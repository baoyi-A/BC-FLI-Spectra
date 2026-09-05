"""Two-stage training of LUMINA, the dual-anchor barcode classifier.

Stage 1 pre-trains the six input stems and the shared trunk on SINGLE-anchor crops --
cells carrying one fluorophore, on the nuclear anchor or on the mitochondrial anchor, so
the other head's label is 0 on those rows. Stage 2 trains on DUAL-anchor crops, where
`clustered.xlsx` supplies both labels, and writes the checkpoint the rest of this folder
consumes.

Which stages run is decided by two switches:

    (neither)       Load --checkpoint, then run stage 2 on every parameter.
    --finetune      Freeze the six input stems (the trunk and both heads keep training),
                    rebuild the optimizer over what is left, then run stage 2.
    --from-scratch  Do not load a checkpoint. Together with --finetune this also runs
                    stage 1 first -- read that flag's help before you do, because that
                    path has two known defects and they are left in place here on purpose.

WHAT IS A FLAG AND WHAT IS NOT
    Every path and every number that `main()` used to hold as a literal is now a flag,
    carrying that literal as its default, so a run that passes no optional flag computes
    exactly what the committed file computed. The class ORDER is deliberately not a flag:
    NU_CLASS_MAP and MITO_CLASS_MAP below are module constants, identical to
    Finetune_LUMINA.py's, and the manifest supplies only which folders on disk hold which
    class. Those indices are baked into every checkpoint and recorded nowhere inside it,
    so a further way to permute them would be a silent way to mislabel every cell. A
    manifest that names only SOME of the panel is fine and changes no index -- the classes
    it leaves out keep theirs, reserved -- and a manifest naming a class outside the two
    maps is an error rather than a silent class 0. The resolved map is printed at startup.

    `--out` is created if absent, and every flag with its resolved value is written once
    to `<out>/train_run_config.csv`, so a directory of results says what produced it.

WHAT IT READS
    Stage 1, single-anchor:  <--data-root>/<folder>/<--seg-folder>/*_5D.tif
        --single-anchor-manifest is a CSV of (class, folder) saying which folder holds
        which barcode. It is read and walked on EVERY run, including the default one in
        which stage 1 never executes. That is what this script does today and it is not
        changed here.
    Stage 2, dual-anchor:    <--dual-root>/<sample>/clustered.xlsx
                             <--dual-root>/<sample>/<--seg-folder>/cell<id>_5D.tif
        --dual-samples or --dual-samples-file names the sample folders. They are loaded
        in the order you give them, and the split below is taken by ROW POSITION, so
        THE ORDER IS PART OF THE VALIDATION SPLIT: the same names in a different order
        hold out different cells at the same --seed. Reproducing a run therefore needs
        the same list in the same order as well as the same --seed. The resolved order
        is printed at startup and recorded in train_run_config.csv.

WHAT IT WRITES, into --out
    best_model_<phase>.pth               rewritten on every validation-loss improvement,
                                         so the file on disk is the best epoch, not the
                                         last one.
    combination_accuracies_<phase>.xlsx  one sheet per epoch, appended to the workbook.
    test_train_val_log.xlsx              the loss/accuracy curve. It carries no phase, so
                                         a second run in the same --out overwrites it.
    val_df.xlsx                          the dual-anchor validation split, so the same
                                         split can be reproduced later. Rebuilding it from
                                         the flags needs the same --dual-samples ORDER as
                                         well as the same --seed; keep this file, or the
                                         --dual-samples-file that produced it, beside the
                                         checkpoint.
    train_run_config.csv                 every flag and its resolved value.

Usage:
    python Train_LUMINA.py --data-root /path/to/single_anchor \
        --single-anchor-manifest single_anchor.csv \
        --dual-root /path/to/dual_anchor --dual-samples-file dual_samples.txt \
        --checkpoint best_model_fine-tune.pth --out ./train_out

This module is import-safe: argparse, file I/O and device creation all sit inside main(),
behind the __main__ guard.
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Class name -> class index. Written out explicitly, and character for character
# Finetune_LUMINA.py's NU_CLASS_MAP and MITO_CLASS_MAP. These indices used to be derived
# from the ORDER of the folder dicts that lived in main(), half of whose entries were
# commented out, so uncommenting one line renumbered every class after it. The checkpoint
# stores no mapping, so that failure was silent. Test_LUMINA.py and Visualize_heatmap.py
# restate the same order; if you change the panel, change it in all four places.
NU_CLASS_MAP = {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
MITO_CLASS_MAP = {'M10': 1, 'M13': 2, 'M4': 3, 'M14': 4, 'M16': 5, 'M8': 6, 'M1': 7}


def comma_list(value):
    """Split a comma-separated flag value into a list of non-empty, stripped strings."""
    return [s.strip() for s in value.split(',') if s.strip()]


def int_list(value, flag):
    """comma_list, as integers, with the flag named in the failure."""
    try:
        return [int(s) for s in comma_list(value)]
    except ValueError:
        raise SystemExit('%s must be a comma-separated list of integers, got: %s'
                         % (flag, value))


def require_dir(path, flag):
    if not os.path.isdir(path):
        raise SystemExit('%s does not exist or is not a folder: %s' % (flag, path))
    return path


def require_file(path, flag):
    if not os.path.isfile(path):
        raise SystemExit('%s does not exist or is not a file: %s' % (flag, path))
    return path


def read_name_file(path, flag):
    """One folder name per line; blank lines and lines starting with # are ignored."""
    require_file(path, flag)
    # utf-8-sig, not utf-8: PowerShell writes a UTF-8 BOM by default (both `>` and
    # Out-File -Encoding utf8), so a list file made the platform's own obvious way
    # would otherwise carry U+FEFF into the first folder name and fail the lookup.
    with open(path, encoding='utf-8-sig') as handle:
        names = [line.strip() for line in handle
                 if line.strip() and not line.strip().startswith('#')]
    if not names:
        raise SystemExit('%s contains no folder names: %s' % (flag, path))
    return names


def load_single_anchor_manifest(path):
    """Read the (class, folder) CSV into the two folder dicts stage 1 walks.

    A `class` of N10/N13/... goes to the nuclear dict, M10/M13/... to the mitochondrial
    one, and any other name is an error rather than a silent class 0. Several rows may
    name the same class. The dicts come back in NU_CLASS_MAP / MITO_CLASS_MAP order, not
    in the manifest's row order, so that the order of rows in the file cannot move the
    --seed split.
    """
    require_file(path, '--single-anchor-manifest')
    df = pd.read_csv(path)
    missing = [c for c in ('class', 'folder') if c not in df.columns]
    if missing:
        raise SystemExit('--single-anchor-manifest %s is missing column(s): %s\n'
                         'Required: class, folder. One row per (barcode, folder) pair; a '
                         'class may appear on several rows.' % (path, ', '.join(missing)))
    nu_files = {}
    mito_files = {}
    unknown = []
    for _, row in df.iterrows():
        name = str(row['class']).strip()
        folder = str(row['folder']).strip()
        if not name or not folder or name == 'nan' or folder == 'nan':
            continue
        if name in NU_CLASS_MAP:
            nu_files.setdefault(name, []).append(folder)
        elif name in MITO_CLASS_MAP:
            mito_files.setdefault(name, []).append(folder)
        else:
            unknown.append(name)
    if unknown:
        raise SystemExit(
            '--single-anchor-manifest %s names class(es) this panel does not have: %s\n'
            'Known nuclear classes: %s\nKnown mitochondrial classes: %s\n'
            'The class indices are baked into the checkpoint, so an unknown name is fatal '
            'rather than a warning: it would otherwise train as class 0, which is the '
            '"no anchor" index.'
            % (path, ', '.join(sorted(set(unknown))),
               ', '.join(NU_CLASS_MAP), ', '.join(MITO_CLASS_MAP)))
    if not nu_files and not mito_files:
        raise SystemExit('--single-anchor-manifest %s has no usable rows.' % path)
    nu_files = {k: nu_files[k] for k in NU_CLASS_MAP if k in nu_files}
    mito_files = {k: mito_files[k] for k in MITO_CLASS_MAP if k in mito_files}
    return nu_files, mito_files


def normalize_intensity(image):
    intensity_channel = image[-1]
    normalized_intensity = intensity_channel / np.max(intensity_channel)
    image[-1] = normalized_intensity
    return image

def pad_image(image, height, width):
    padded_image = np.zeros((image.shape[0], height, width), dtype=image.dtype)
    h, w = image.shape[1:]
    y_start = (height - h) // 2
    x_start = (width - w) // 2
    padded_image[:, y_start:y_start + h, x_start:x_start + w] = image
    return padded_image

class FluorescenceDataset(Dataset):
    def __init__(self, df, base_dir, max_height, max_width, transform=None, is_test=False,
                 seg_folder='seg_5D'):
        self.df = df
        self.base_dir = base_dir
        self.max_height = max_height
        self.max_width = max_width
        self.transform = transform
        self.is_test = is_test
        # Only the is_test branch uses this: the pre-training rows carry an absolute
        # `output_file`. Test_LUMINA.py has a same-named class with a different signature;
        # do not copy a change here into that one without reading it.
        self.seg_folder = seg_folder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        while True:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            row = self.df.iloc[idx]

            if self.is_test:
                cell_label = int(row['Cell_Label'])
                test_dir = row['Directory']
                seg_dir = os.path.join(self.base_dir, test_dir, self.seg_folder)
                img_path = os.path.join(seg_dir, f'cell{cell_label}_5D.tif')
                nu_label = int(row['Nu_cluster'])
                mito_label = int(row['Mito_cluster'])
            else:
                img_path = row['output_file']
                nu_label = int(row['nu_class'])
                mito_label = int(row['mito_class'])
            # only use those nu andmito labels not 0
            if nu_label == 0 or mito_label == 0:
                idx = (idx + 1) % len(self.df)
                continue
            img = tiff.imread(img_path)
            if img.shape[1] > self.max_height or img.shape[2] > self.max_width:
                idx = (idx + 1) % len(self.df)
                continue
            img = np.nan_to_num(img)
            resized_img = pad_image(img, self.max_height, self.max_width)
            normalized_img = normalize_intensity(resized_img)

            sample = (normalized_img, nu_label, mito_label)

            if self.transform:
                sample = self.transform(sample)

            return sample


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class DualHeadConvNet(nn.Module):
    def __init__(self, num_classes, height=256, width=256):
        super(DualHeadConvNet, self).__init__()

        # Six input heads
        self.input_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ) for _ in range(6)
        ])

        # ResNet-like backbone
        self.backbone = nn.Sequential(
            ResNetBlock(384, 384),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlock(384, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResNetBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Calculate the flattened size
        self.flat_size = 512

        # Fully connected layers
        self.fc_nu = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.fc_mito = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Process each input channel through its respective head
        x_heads = [head(x[:, i:i + 1]) for i, head in enumerate(self.input_heads)]

        # Concatenate the outputs of all heads
        x = torch.cat(x_heads, dim=1)

        # Pass through the ResNet-like backbone
        x = self.backbone(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        out_nu = self.fc_nu(x)
        out_mito = self.fc_mito(x)

        return out_nu, out_mito

def load_training_data(nu_files, mito_files, base_folder, seg_folder='seg_5D'):
    data = []
    # These used to be `{key: idx + 1 for idx, key in enumerate(nu_files.keys())}`, i.e.
    # derived from the ORDER of the folder dicts. They are now the module constants, so
    # neither the manifest's row order nor the SUBSET of classes it happens to name can
    # renumber anything: a class keeps its index whether or not the manifest mentions it.
    # load_single_anchor_manifest() has already rejected any name outside these two maps,
    # so every lookup below is a hit.
    nu_class_map = dict(NU_CLASS_MAP)
    mito_class_map = dict(MITO_CLASS_MAP)
    require_dir(base_folder, '--data-root')
    print(f'nu_class_map: {nu_class_map}')
    print(f'mito_class_map: {mito_class_map}')
    for nu_class, files in nu_files.items():
        nu_class_num = nu_class_map[nu_class]
        for file in files:
            img_dir = os.path.join(base_folder, file, seg_folder)
            require_dir(img_dir, '--data-root/--single-anchor-manifest/--seg-folder')
            for img_file in os.listdir(img_dir):
                if img_file.endswith('_5D.tif'):
                    img_path = os.path.join(img_dir, img_file)
                    data.append({
                        'output_file': img_path,
                        'nu_class': nu_class_num,
                        'mito_class': 0  # No mito
                    })

    for mito_class, files in mito_files.items():
        mito_class_num = mito_class_map[mito_class]
        for file in files:
            img_dir = os.path.join(base_folder, file, seg_folder)
            require_dir(img_dir, '--data-root/--single-anchor-manifest/--seg-folder')
            for img_file in os.listdir(img_dir):
                if img_file.endswith('_5D.tif'):
                    img_path = os.path.join(img_dir, img_file)
                    data.append({
                        'output_file': img_path,
                        'nu_class': 0,  # No nu
                        'mito_class': mito_class_num
                    })

    df = pd.DataFrame(data)
    return df, nu_class_map, mito_class_map

def load_finetuning_data(test_dirs, base_folder, nu_class_map, mito_class_map):
    data = []
    require_dir(base_folder, '--dual-root')
    for test_dir in test_dirs:
        sample_dir = os.path.join(base_folder, test_dir)
        if not os.path.isdir(sample_dir):
            raise SystemExit(
                'A sample named by --dual-samples/--dual-samples-file is not a folder '
                'under --dual-root: %s\n'
                'Expected layout: <--dual-root>/<sample>/clustered.xlsx and '
                '<--dual-root>/<sample>/<--seg-folder>/cell<id>_5D.tif.' % sample_dir)
        excel_path = os.path.join(base_folder, test_dir, 'clustered.xlsx')
        if not os.path.isfile(excel_path):
            raise SystemExit(
                'Sample folder %s has no clustered.xlsx.\n'
                'It carries the dual-anchor labels (Cell_Label, Nu_FP, Mito_FP) this stage '
                'trains on, so a sample without one is fatal rather than skipped: skipping '
                'it silently would train on a smaller population than the one you named.'
                % sample_dir)
        df = pd.read_excel(excel_path)
        for _, row in df.iterrows():
            nu_class = row['Nu_FP']
            mito_class = row['Mito_FP']
            # print(f'nu_class: {nu_class}, mito_class: {mito_class}')
            nu_class_num = nu_class_map.get(nu_class, 0) if pd.notna(nu_class) else 0
            mito_class_num = mito_class_map.get(mito_class, 0) if pd.notna(mito_class) else 0
            # print(f'nu_class_num: {nu_class_num}, mito_class_num: {mito_class_num}')
            # only consider those cells with certain nu and mito class
            if nu_class_num == 0 or mito_class_num == 0:
                continue
            cell_label = row['Cell_Label']
            data.append({
                'Directory': test_dir,
                'Cell_Label': cell_label,
                'Nu_cluster': nu_class_num,
                'Mito_cluster': mito_class_num
            })

    df = pd.DataFrame(data)
    return df


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, nu_class_counts, mito_class_counts, device):
        super().__init__()
        # Calculate weights for nuclear classes
        total_nu = sum(nu_class_counts.values())
        nu_weights = torch.zeros(max(nu_class_counts.keys()) + 1)
        for class_idx, count in nu_class_counts.items():
            nu_weights[class_idx] = total_nu / (len(nu_class_counts) * count)

        # Calculate weights for mito classes
        total_mito = sum(mito_class_counts.values())
        mito_weights = torch.zeros(max(mito_class_counts.keys()) + 1)
        for class_idx, count in mito_class_counts.items():
            mito_weights[class_idx] = total_mito / (len(mito_class_counts) * count)

        self.nu_criterion = nn.CrossEntropyLoss(weight=nu_weights.to(device))
        self.mito_criterion = nn.CrossEntropyLoss(weight=mito_weights.to(device))

    def forward(self, outputs_nu, outputs_mito, targets_nu, targets_mito):
        loss_nu = self.nu_criterion(outputs_nu, targets_nu)
        loss_mito = self.mito_criterion(outputs_mito, targets_mito)
        return loss_nu + loss_mito


def get_class_counts(dataloader):
    nu_counts = Counter()
    mito_counts = Counter()

    for _, nu_labels, mito_labels in dataloader:
        nu_counts.update(nu_labels.numpy())
        mito_counts.update(mito_labels.numpy())

    return dict(nu_counts), dict(mito_counts)


# Modify the train_model function
# NOTE: `early_stop_patience=10` here is NOT the value the script runs with. main() passes
# --early-stop-patience, whose default is the literal main() used to hold, 1000. Sourcing
# a default from this signature would end every run after ten stagnant epochs.
def train_model(model, train_loader, val_loader, optimizer, num_epochs, phase, out_folder, device,
                early_stop_patience=10,
                lr_drop_epochs_initial=(30, 70), lr_drop_epochs_finetune=(200, 350, 500),
                lr_drop_factor=0.2):
    model.to(device)

    # Calculate class weights from training data
    print("Calculating class weights...")
    nu_counts, mito_counts = get_class_counts(train_loader)
    print(f"Nuclear class distribution: {nu_counts}")
    print(f"Mitochondrial class distribution: {mito_counts}")

    # Initialize weighted loss
    criterion = WeightedCrossEntropyLoss(nu_counts, mito_counts, device)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience = 0

    combine_acc_path = os.path.join(out_folder, f'combination_accuracies_{phase}.xlsx')
    if not os.path.exists(combine_acc_path):
        initial_df = pd.DataFrame({'Initial': ['This is the initial content']})
        with pd.ExcelWriter(combine_acc_path, engine='openpyxl') as writer:
            initial_df.to_excel(writer, sheet_name='Sheet1', index=False)

    for epoch in range(num_epochs):
        print(f'lr: {optimizer.param_groups[0]["lr"]}')

        if phase == 'initial' and epoch in lr_drop_epochs_initial:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_drop_factor
        elif phase == 'fine-tune' and epoch in lr_drop_epochs_finetune:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_drop_factor

        model.train()
        running_loss = 0.0
        correct_pred_train = 0
        total_pred_train = 0

        for i, (images, nu_labels, mito_labels) in enumerate(train_loader):
            images = images.clone().detach().float().to(device)
            nu_labels = nu_labels.clone().detach().long().to(device)
            mito_labels = mito_labels.clone().detach().long().to(device)

            optimizer.zero_grad()

            outputs_nu, outputs_mito = model(images)
            loss = criterion(outputs_nu, outputs_mito, nu_labels, mito_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred_nu = torch.argmax(outputs_nu, dim=1)
            pred_mito = torch.argmax(outputs_mito, dim=1)
            correct_pred_train += ((pred_nu == nu_labels) & (pred_mito == mito_labels)).sum().item()
            total_pred_train += len(nu_labels)

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracy = correct_pred_train / total_pred_train
        train_accuracies.append(train_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5f}, Accuracy: {train_accuracy:.4f}')

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct_pred_val = 0
            total_pred_val = 0
            combo_stats = {}

            for i, (images, nu_labels, mito_labels) in enumerate(val_loader):
                images = images.clone().detach().float().to(device)
                nu_labels = nu_labels.clone().detach().long().to(device)
                mito_labels = mito_labels.clone().detach().long().to(device)

                outputs_nu, outputs_mito = model(images)
                loss = criterion(outputs_nu, outputs_mito, nu_labels, mito_labels)
                val_loss += loss.item()

                pred_nu = torch.argmax(outputs_nu, dim=1)
                pred_mito = torch.argmax(outputs_mito, dim=1)

                # Calculate overall accuracy
                correct_mask = (pred_nu == nu_labels) & (pred_mito == mito_labels)
                correct_pred_val += correct_mask.sum().item()
                total_pred_val += len(nu_labels)

                # Convert to numpy for easier processing
                nu_labels_np = nu_labels.cpu().numpy()
                mito_labels_np = mito_labels.cpu().numpy()
                pred_nu_np = pred_nu.cpu().numpy()
                pred_mito_np = pred_mito.cpu().numpy()
                correct_mask_np = correct_mask.cpu().numpy()

                # Update combination statistics
                for idx in range(len(nu_labels_np)):
                    true_combo = (int(nu_labels_np[idx]), int(mito_labels_np[idx]))
                    pred_combo = (int(pred_nu_np[idx]), int(pred_mito_np[idx]))

                    # Initialize if this true combination hasn't been seen
                    if true_combo not in combo_stats:
                        combo_stats[true_combo] = {
                            'total': 0,
                            'correct': 0,
                            'predictions': {}  # To store distribution of predictions
                        }

                    # Update statistics
                    combo_stats[true_combo]['total'] += 1
                    if correct_mask_np[idx]:
                        combo_stats[true_combo]['correct'] += 1

                    # Track prediction distribution
                    if pred_combo not in combo_stats[true_combo]['predictions']:
                        combo_stats[true_combo]['predictions'][pred_combo] = 0
                    combo_stats[true_combo]['predictions'][pred_combo] += 1

            # Calculate metrics
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_accuracy = correct_pred_val / total_pred_val
            val_accuracies.append(val_accuracy)
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            # scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), os.path.join(out_folder, f'best_model_{phase}.pth'))
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print("Early stopping")
                    break

                # Prepare data for Excel output
            if (epoch + 1) % 1 == 0:
                # Create lists to store the data
                rows = []
                for true_combo in combo_stats:
                    stats = combo_stats[true_combo]
                    accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

                    # Get top 3 predictions for this combination
                    pred_dist = stats['predictions']
                    sorted_preds = sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)[:3]

                    row = {
                        'True_Nu': f'N{true_combo[0]}',
                        'True_Mito': f'M{true_combo[1]}',
                        'Total_Samples': stats['total'],
                        'Correct_Predictions': stats['correct'],
                        'Accuracy': accuracy,
                        'Top1_Pred': f'N{sorted_preds[0][0][0]}-M{sorted_preds[0][0][1]}',
                        'Top1_Count': sorted_preds[0][1],
                        'Top2_Pred': f'N{sorted_preds[1][0][0]}-M{sorted_preds[1][0][1]}' if len(
                            sorted_preds) > 1 else '',
                        'Top2_Count': sorted_preds[1][1] if len(sorted_preds) > 1 else 0,
                        'Top3_Pred': f'N{sorted_preds[2][0][0]}-M{sorted_preds[2][0][1]}' if len(
                            sorted_preds) > 2 else '',
                        'Top3_Count': sorted_preds[2][1] if len(sorted_preds) > 2 else 0
                    }
                    rows.append(row)

                # Convert to DataFrame
                results_df = pd.DataFrame(rows)

                # Sort by accuracy and sample count
                results_df = results_df.sort_values(['Accuracy', 'Total_Samples'], ascending=[False, False])

                # Write to Excel
                with pd.ExcelWriter(combine_acc_path,
                                    engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    results_df.to_excel(writer,
                                        sheet_name=f'epoch_{epoch + 1}',
                                        index=False)

                # Save fine-tuning losses and accuracies
                test_train_val_log = pd.DataFrame({
                    'Epoch': list(range(1, len(train_losses) + 1)),
                    'Train Loss': train_losses,
                    'Validation Loss': val_losses,
                    'Train Accuracy': train_accuracies,
                    'Validation Accuracy': val_accuracies
                })
                test_train_val_log.to_excel(os.path.join(out_folder, 'test_train_val_log.xlsx'), index=False)
    # return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    ap = argparse.ArgumentParser(
        description='Two-stage training of the LUMINA dual-anchor classifier: pre-train on '
                    'single-anchor crops, then train on dual-anchor crops.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--data-root', required=True,
                    help='Root of the SINGLE-ANCHOR (stage 1) folders: '
                         '<root>/<folder>/seg_5D/*_5D.tif, see --seg-folder. It is walked '
                         'on every run, including the default one in which stage 1 never '
                         'executes.')
    ap.add_argument('--single-anchor-manifest', required=True,
                    help='CSV with columns class,folder saying which folder under '
                         '--data-root holds which barcode. Class names must be this '
                         'panel\'s own (N10, N13, ... / M10, M13, ...) and a class may '
                         'appear on several rows. Row order does not matter: the class '
                         'indices come from the module maps, never from this file.')
    ap.add_argument('--dual-root', required=True,
                    help='Root of the DUAL-ANCHOR (stage 2) folders, each holding a '
                         'clustered.xlsx and a folder of crops. This is TRAINING data, not '
                         'a held-out set: the only cells held out are the --val-split '
                         'fraction of these same folders.')
    ap.add_argument('--dual-samples', default='',
                    help='Comma-separated dual-anchor sample folder names under '
                         '--dual-root. Give this or --dual-samples-file. THE ORDER IS PART '
                         'OF THE SPLIT: the folders are loaded in the order you write them '
                         'and the stage-2 validation split is taken by row position, so '
                         '"a,b" and "b,a" hold out different cells at the same --seed. A '
                         'rerun must pass the same names in the same order to get the same '
                         'validation set and the same checkpoint. The resolved order is '
                         'printed at startup and recorded in train_run_config.csv as '
                         'resolved_dual_samples.')
    ap.add_argument('--dual-samples-file', default='',
                    help='The same names in a text file, one per line; blank lines and '
                         'lines starting with # are ignored. Use this once the list '
                         'outgrows a command line -- and keep the file, because the ORDER '
                         'OF THE LINES is part of the split exactly as it is for '
                         '--dual-samples: reordering the file moves the validation set at '
                         'the same --seed.')
    ap.add_argument('--checkpoint', default='',
                    help='Pre-trained LUMINA state_dict that stage 2 starts from, e.g. '
                         'best_model_fine-tune.pth. Give this or --from-scratch. Its head '
                         'width must equal --num-classes.')
    ap.add_argument('--seg-folder', default='seg_5D',
                    help='Folder of prepared crops read under each sample, under both '
                         'roots. This script has only ever read seg_5D, while Data_prep.py '
                         'writes seg_5D_calib; name that folder here if that is what you '
                         'prepared. There is deliberately no "auto" as in '
                         'Finetune_LUMINA.py -- a checkpoint trained on a mixture of the '
                         'two would not be reproducible from this flag.')
    ap.add_argument('--out', required=True,
                    help='Output directory. Created if absent. A second run pointed at the '
                         'same directory overwrites test_train_val_log.xlsx and val_df.xlsx '
                         'and appends sheets to combination_accuracies_<phase>.xlsx.')

    ap.add_argument('--crop-size', type=int, default=256,
                    help='Both loaders pad every crop onto a square canvas of this size. A '
                         'crop LARGER than this is skipped, not resized.')
    ap.add_argument('--batch-size', type=int, default=128,
                    help='Used by all four loaders. Freezing the six input stems with '
                         '--finetune leaves room for a larger batch.')
    ap.add_argument('--epochs', type=int, default=180,
                    help='STAGE 1 ONLY, and stage 1 runs only under --from-scratch '
                         '--finetune. Stage 2 has its own budget, --finetune-epochs.')
    ap.add_argument('--finetune-epochs', type=int, default=800,
                    help='STAGE 2, which is the run you get by default. Separate from '
                         '--epochs on purpose: one flag for both would silently hand stage '
                         '2 the stage-1 budget. combination_accuracies_fine-tune.xlsx gains '
                         'one sheet per epoch, so this also sets how large that workbook '
                         'grows and how long each epoch\'s write takes.')
    ap.add_argument('--early-stop-patience', type=int, default=1000,
                    help='Stop after this many epochs with no validation-loss improvement. '
                         'The default is larger than either epoch budget, so early stopping '
                         'never fires unless you lower it. best_model_<phase>.pth is '
                         'rewritten on every improvement either way, so the file on disk is '
                         'the best epoch rather than the last one.')
    ap.add_argument('--lr', type=float, default=1e-3,
                    help='AdamW learning rate, for both stages. weight_decay is left at '
                         'torch\'s AdamW default and is deliberately not a flag: this '
                         'script has never overridden it.')
    ap.add_argument('--lr-drop-epochs-initial', default='30,70',
                    help='Comma-separated stage-1 epochs at which the learning rate is '
                         'multiplied by --lr-drop-factor. 0-based, while the progress line '
                         'prints epoch+1.')
    ap.add_argument('--lr-drop-epochs-finetune', default='200,350,500',
                    help='The same for stage 2. Kept separate from the stage-1 list because '
                         'the phase selects which one applies.')
    ap.add_argument('--lr-drop-factor', type=float, default=0.2,
                    help='Multiplier applied at each of those epochs, in both stages.')
    ap.add_argument('--num-classes', type=int, default=8,
                    help='Width of both heads. Index 0 means "no anchor" and is never a '
                         'training target, so a seven-barcode panel needs eight outputs. '
                         'Must match --checkpoint or the state_dict will not load.')
    ap.add_argument('--seed', type=int, default=42,
                    help='random_state for BOTH validation splits, the single-anchor one '
                         'and the dual-anchor one. Not the same quantity as '
                         'Finetune_LUMINA.py\'s --seed, which draws a support set.')
    ap.add_argument('--val-split', type=float, default=0.2,
                    help='Validation fraction, for BOTH splits. It splits CELLS, not '
                         'acquisitions: cells from one dish land on both sides, so the '
                         'validation accuracy is not a held-out-acquisition estimate and '
                         'should not be quoted as one.')
    ap.add_argument('--device', default='cuda:0',
                    help='cuda:0, cuda:1, cpu, ... The literal this script has always used '
                         'is cuda:0 with no CPU fallback, and that is kept: a machine '
                         'without a GPU fails here rather than quietly starting a training '
                         'run that would never finish.')

    ap.add_argument('--finetune', action='store_true',
                    help='Freeze the six input stems before stage 2 and rebuild the '
                         'optimizer over the parameters that remain; the trunk and both '
                         'heads still train. It ALSO gates whether stage 1 runs, which '
                         'matters only together with --from-scratch. Off by default, so '
                         'stage 2 trains every parameter.')
    ap.add_argument('--from-scratch', action='store_true',
                    help='Do not load a checkpoint; start from a random initialisation. '
                         'WARNING: combined with --finetune this also runs stage 1, and '
                         'stage 1 does not work in this file. train_model returns nothing '
                         'while its caller unpacks four values, and before that every '
                         'stage-1 row carries a 0 label for the anchor its cells lack, '
                         'which the loader skips, so no batch ever assembles. Both are '
                         'pre-existing and are left exactly as they are, because either '
                         'repair changes what a training run produces.')

    args = ap.parse_args()

    if not args.dual_samples and not args.dual_samples_file:
        raise SystemExit('Give --dual-samples (comma-separated folder names under '
                         '--dual-root) or --dual-samples-file (the same names, one per '
                         'line).')
    if args.dual_samples and args.dual_samples_file:
        raise SystemExit('Give --dual-samples or --dual-samples-file, not both. Which list '
                         'trained a checkpoint has to be unambiguous.')
    if not args.checkpoint and not args.from_scratch:
        raise SystemExit('Give --checkpoint (the state_dict stage 2 starts from) or '
                         '--from-scratch (start from a random initialisation).')
    if args.checkpoint and args.from_scratch:
        raise SystemExit('--checkpoint and --from-scratch contradict each other: '
                         '--from-scratch ignores the checkpoint entirely. Drop one.')
    if args.batch_size < 1 or args.epochs < 1 or args.finetune_epochs < 1:
        raise SystemExit('--batch-size, --epochs and --finetune-epochs must be at least 1.')
    if args.crop_size < 1:
        raise SystemExit('--crop-size must be at least 1.')
    if not 0.0 < args.val_split < 1.0:
        raise SystemExit('--val-split must be strictly between 0 and 1, got: %g'
                         % args.val_split)
    min_classes = max(max(NU_CLASS_MAP.values()), max(MITO_CLASS_MAP.values())) + 1
    if args.num_classes < min_classes:
        raise SystemExit(
            '--num-classes is %d but this panel needs at least %d outputs per head (index 0 '
            'is reserved for "no anchor", and the highest class index in use is %d). A '
            'narrower head cannot represent every barcode and would mislabel silently.'
            % (args.num_classes, min_classes, min_classes - 1))

    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit('--device is not a device string: %s (%s)' % (args.device, exc))
    use_pretrained = not args.from_scratch
    use_finetune = args.finetune
    lr_drop_initial = int_list(args.lr_drop_epochs_initial, '--lr-drop-epochs-initial')
    lr_drop_finetune = int_list(args.lr_drop_epochs_finetune, '--lr-drop-epochs-finetune')
    if args.dual_samples_file:
        given_dirs = read_name_file(args.dual_samples_file, '--dual-samples-file')
    else:
        given_dirs = comma_list(args.dual_samples)
    # NOT sorted, and not de-duplicated: the list is loaded in exactly the order it is
    # given. load_finetuning_data() appends rows folder by folder in that order and
    # train_test_split() splits by ROW POSITION, so the order is genuinely part of the
    # stage-2 validation split -- `--dual-samples a,b` and `--dual-samples b,a` hold out
    # different cells at the same --seed.
    #
    # Sorting here would make the split a function of the set and the seed alone, which is
    # tidier, but the list this script shipped with was hand-written and NOT in sorted
    # order. Sorting it would mean this file no longer reproduces the split, and therefore
    # no longer reproduces the checkpoint, that the shipped script produced -- with no flag
    # to ask for the original back. So the order is preserved, and the hazard is made
    # visible instead: it is printed below, it is written to train_run_config.csv, and both
    # --dual-samples and --dual-samples-file say so in their help.
    test_dirs = list(given_dirs)

    print('nu_class_map: %s' % NU_CLASS_MAP)
    print('mito_class_map: %s' % MITO_CLASS_MAP)
    print('  (fixed, from this module. Neither the manifest nor --dual-samples can '
          'renumber a class.)')
    print('device: %s   crop: %d   batch: %d   num_classes: %d'
          % (device, args.crop_size, args.batch_size, args.num_classes))
    print('stage 1 (initial): %s   epochs: %d   lr-drops: %s'
          % ('runs' if (not use_pretrained and use_finetune) else 'skipped',
             args.epochs, lr_drop_initial))
    print('stage 2 (fine-tune): runs   epochs: %d   lr-drops: %s   input stems: %s'
          % (args.finetune_epochs, lr_drop_finetune,
             'frozen (--finetune)' if use_finetune else 'training'))
    print('lr: %g   lr-drop factor: %g   early stop patience: %d%s'
          % (args.lr, args.lr_drop_factor, args.early_stop_patience,
             '  (larger than both epoch budgets, so it never fires)'
             if args.early_stop_patience > max(args.epochs, args.finetune_epochs) else ''))
    print('start: %s   seg folder: %s   val split: %g (seed %d)   dual samples: %d'
          % ('--checkpoint %s' % args.checkpoint if use_pretrained
             else 'random init (--from-scratch)',
             args.seg_folder, args.val_split, args.seed, len(test_dirs)))
    print('dual samples, in the order given -- this is the load order, and the --seed '
          'validation split is taken over the rows in this order:')
    print('  %s' % ', '.join(test_dirs))
    print('  ORDER MATTERS: the same names in a different order hold out a different '
          'validation set at the same --seed, so a rerun has to pass this list in this '
          'order. It is also recorded as resolved_dual_samples in train_run_config.csv.')

    # The literals these locals used to hold are now the flags' defaults, so a run with no
    # optional flag computes what the committed file computed.
    base_folder = args.data_root
    test_base_folder = args.dual_root
    out_folder = args.out
    os.makedirs(out_folder, exist_ok=True)
    max_height = args.crop_size
    max_width = args.crop_size
    batchsize = args.batch_size
    num_epochs = args.epochs
    early_stop_patience = args.early_stop_patience

    # Every flag, resolved, written once, so a directory of results says what produced it.
    cfg = {k: v for k, v in sorted(vars(args).items())}
    cfg['resolved_device'] = str(device)
    cfg['resolved_use_pretrained'] = use_pretrained
    cfg['resolved_use_finetune'] = use_finetune
    cfg['resolved_lr_drop_epochs_initial'] = ','.join(str(e) for e in lr_drop_initial)
    cfg['resolved_lr_drop_epochs_finetune'] = ','.join(str(e) for e in lr_drop_finetune)
    cfg['resolved_dual_sample_count'] = len(test_dirs)
    cfg['resolved_dual_samples'] = ';'.join(test_dirs)
    cfg['resolved_nu_class_map'] = ';'.join('%s=%d' % (k, v) for k, v in NU_CLASS_MAP.items())
    cfg['resolved_mito_class_map'] = ';'.join('%s=%d' % (k, v) for k, v in MITO_CLASS_MAP.items())
    pd.DataFrame([{'flag': k, 'value': v} for k, v in cfg.items()]).to_csv(
        os.path.join(out_folder, 'train_run_config.csv'), index=False)
    print('run config: %s' % os.path.join(out_folder, 'train_run_config.csv'))

    nu_files, mito_files = load_single_anchor_manifest(args.single_anchor_manifest)

    # The RESOLVED class map: which classes the manifest actually named, and the index each
    # one gets. A manifest naming a subset of the panel is fine and is not renumbered -- the
    # indices below are read out of the module maps, never assigned from the manifest -- so
    # this line is printed to show that the checkpoint about to be written numbers its
    # classes the same way as every other checkpoint from this file.
    named = ['%s=%d' % (k, NU_CLASS_MAP[k]) for k in NU_CLASS_MAP if k in nu_files]
    named += ['%s=%d' % (k, MITO_CLASS_MAP[k]) for k in MITO_CLASS_MAP if k in mito_files]
    absent = [k for k in NU_CLASS_MAP if k not in nu_files]
    absent += [k for k in MITO_CLASS_MAP if k not in mito_files]
    print('[single-anchor] manifest %s' % args.single_anchor_manifest)
    print('  resolved class map: %s' % ('  '.join(named) if named else '(none)'))
    if absent:
        print('  not named by the manifest, indices unchanged and reserved: %s'
              % ', '.join(absent))

    # Load training and fine-tuning data
    train_df, nu_class_map, mito_class_map = load_training_data(nu_files, mito_files,
                                                                base_folder, args.seg_folder)
    print('[single-anchor] %d crop(s) from %d folder(s) under %s (seg folder: %s)'
          % (len(train_df),
             sum(len(v) for v in nu_files.values()) + sum(len(v) for v in mito_files.values()),
             base_folder, args.seg_folder))
    if len(train_df) == 0:
        raise SystemExit(
            'No *_5D.tif found under any folder named by --single-anchor-manifest.\n'
            'Expected layout: <--data-root>/<folder>/%s/*_5D.tif (as written by '
            'Data_prep.py).' % args.seg_folder)
    test_df = load_finetuning_data(test_dirs, test_base_folder, nu_class_map, mito_class_map)
    print('[dual-anchor] %d cell(s) with two known labels, from %d sample folder(s) under %s'
          % (len(test_df), len(test_dirs), test_base_folder))
    if len(test_df) == 0:
        raise SystemExit(
            'No dual-anchor cell had BOTH a known Nu_FP and a known Mito_FP, so there is '
            'nothing to train stage 2 on.\n'
            'Every cell whose Nu_FP or Mito_FP is blank, or is a name outside %s / %s, is '
            'dropped by design.' % (', '.join(NU_CLASS_MAP), ', '.join(MITO_CLASS_MAP)))

    # Split training data for validation
    train_df, val_df = train_test_split(train_df, test_size=args.val_split,
                                        random_state=args.seed)

    # Split test data for fine-tuning
    test_train_df, test_val_df = train_test_split(test_df, test_size=args.val_split,
                                                  random_state=args.seed)

    # save the val df to excel for later use
    test_val_df.to_excel(os.path.join(out_folder, 'val_df.xlsx'), index=False)
    print(f'test_val_df saved to {os.path.join(out_folder, "val_df.xlsx")}')
    # Create datasets and dataloaders
    train_dataset = FluorescenceDataset(train_df, base_folder, max_height, max_width,
                                        seg_folder=args.seg_folder)
    val_dataset = FluorescenceDataset(val_df, base_folder, max_height, max_width,
                                      seg_folder=args.seg_folder)
    test_train_dataset = FluorescenceDataset(test_train_df, test_base_folder, max_height, max_width,
                                             is_test=True, seg_folder=args.seg_folder)
    test_val_dataset = FluorescenceDataset(test_val_df, test_base_folder, max_height, max_width,
                                           is_test=True, seg_folder=args.seg_folder)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    test_train_loader = DataLoader(test_train_dataset, batch_size=batchsize, shuffle=True)
    test_val_loader = DataLoader(test_val_dataset, batch_size=batchsize, shuffle=False)

    # Initial training
    num_classes = args.num_classes
    model = DualHeadConvNet(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Check if user wants to skip initial training and load a pre-trained model
    if use_pretrained:
        require_file(args.checkpoint, '--checkpoint')
        # use map_location to avoid loading model trained on different device
        state_dict = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise SystemExit(
                'The checkpoint did not load into DualHeadConvNet(%d):\n%s\n'
                '--checkpoint %s\n'
                'A width mismatch on fc_nu.6 / fc_mito.6 means --num-classes does not match '
                'this checkpoint. A partially loaded model still trains and still writes a '
                'checkpoint of its own, so this is fatal rather than a warning.'
                % (num_classes, exc, args.checkpoint))
        print('Loaded --checkpoint %s' % args.checkpoint)
    else:
        if use_finetune:
            print('')
            print('WARNING: --from-scratch --finetune runs stage 1, and stage 1 does not '
                  'work in this file.')
            print('  train_model has no return statement while the call below unpacks four '
                  'values from it, so expect "cannot unpack non-sequence NoneType".')
            print('  Before that: every stage-1 row carries a 0 label for the anchor its '
                  'cells do not have, and the loader skips any row with a 0 label, so no '
                  'batch assembles and the process spins with no output.')
            print('  Both defects are pre-existing and are left in place on purpose. Either '
                  'repair changes what a training run produces, and which side is wrong is '
                  'not something this file settles.')
            print('')
            train_losses, val_losses, train_accuracies, val_accuracies = train_model(
                model, train_loader, val_loader, optimizer, num_epochs, phase='initial',
                early_stop_patience=early_stop_patience, out_folder=out_folder,
                device=device, lr_drop_epochs_initial=lr_drop_initial,
                lr_drop_epochs_finetune=lr_drop_finetune, lr_drop_factor=args.lr_drop_factor
            )

            # Save training and validation losses and accuracies
            train_val_log = pd.DataFrame({
                'Epoch': list(range(1, len(train_losses) + 1)),
                'Train Loss': train_losses,
                'Validation Loss': val_losses,
                'Train Accuracy': train_accuracies,
                'Validation Accuracy': val_accuracies
            })
            train_val_log.to_excel(os.path.join(out_folder, 'train_val_log_pre.xlsx'), index=False)


    # Fine-tuning
    def freeze_conv_layers(model):
        # The six input stems only. The shared trunk and both heads keep training.
        for head in model.input_heads:
            for param in head.parameters():
                param.requires_grad = False

    if use_finetune:
        freeze_conv_layers(model)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # Stage 2 has its own epoch budget, --finetune-epochs, not --epochs.
    num_epochs = args.finetune_epochs
    train_model(
        model, test_train_loader, test_val_loader, optimizer, num_epochs, phase='fine-tune',
        out_folder=out_folder, device=device,
        early_stop_patience=early_stop_patience, lr_drop_epochs_initial=lr_drop_initial,
        lr_drop_epochs_finetune=lr_drop_finetune, lr_drop_factor=args.lr_drop_factor
    )


if __name__ == '__main__':
    main()
