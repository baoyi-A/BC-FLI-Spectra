"""Inference with a trained LUMINA dual-anchor checkpoint.

Reads the per-cell crops `Data_prep.py` wrote (`<root>/<sample>/<seg folder>/
cell<id>_5D.tif`), pushes each one through a `DualHeadConvNet` checkpoint, gates both
head calls on the LUMINA composite confidence score, and writes two workbooks per
sample folder:

    predict_class_confident_<threshold>.xlsx   both heads cleared the gate
    predict_class_uncertain_<threshold>.xlsx   at least one head did not

The threshold is part of the file name, and `Visualize_heatmap.py` rebuilds that same
name from its own threshold flag, so the two must be run with the same value.

Every path and every knob is a command-line flag, the same convention
`Finetune_LUMINA.py` uses; the resolved value of each one is written once per run to
`test_run_config.csv` so a directory of workbooks says what produced it.

WHICH SAMPLE FOLDERS IT TOUCHES IS NEVER A DEFAULT
    Unless `--out` redirects them, those two workbooks are written back INTO each sample
    folder and overwrite any file of the same name already there, so the set of folders is
    asked for explicitly: either name them with `--samples`, or say `--all-samples` to mean
    every qualifying sample folder under the root(s). Passing neither is an error. The
    resolved list is printed in full, one destination per line, before anything is
    written. Same rule and same wording as `Data_prep.py`.

This module is import-safe: nothing outside a function touches a path, a device or
argv. `Finetune_LUMINA.py` imports `DualHeadConvNet` and `pad_image` from here and
relies on that.

Usage:
    python Test_LUMINA.py --checkpoint /path/to/best_model_fine-tune.pth \
        --data-root /path/to/dish --samples sampleA,sampleB --confidence-threshold 0.6
    python Test_LUMINA.py --checkpoint /path/to/best_model_fine-tune.pth \
        --data-root /path/to/dish --all-samples
"""

import argparse
import glob
import torch.nn as nn
import os
import pandas as pd
import tifffile as tiff
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


# Class name -> class index. Written out explicitly on purpose. This file used to derive
# the same mapping from two dicts of lab folder names whose entries were half commented
# out, so uncommenting one line renumbered every class after it. The checkpoint stores no
# mapping, so that failure is silent. Train_LUMINA.py, Visualize_heatmap.py and
# Finetune_LUMINA.py each restate this same order; if you change the panel, change it in
# all four places.
NU_CLASS_MAP = {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
MITO_CLASS_MAP = {'M10': 1, 'M13': 2, 'M4': 3, 'M14': 4, 'M16': 5, 'M8': 6, 'M1': 7}

CROP_SIZE = 256          # the canvas the loader pads to; a larger crop is skipped

# Which folder of prepared crops to read, in preference order. "auto" is the rule this
# script has always used: seg_5D_calib when it is there, seg_5D otherwise. The two are
# NOT interchangeable -- seg_5D_calib holds spectrally calibrated crops -- so on a dish
# that has both, "auto" feeds the network a different input distribution than seg_5D.
SEG_FOLDER_ORDER = {
    'auto': ('seg_5D_calib', 'seg_5D'),
    'seg_5D': ('seg_5D',),
    'seg_5D_calib': ('seg_5D_calib',),
}


def comma_list(value):
    """Split a comma-separated flag value into a list of non-empty, stripped strings."""
    return [s.strip() for s in value.split(',') if s.strip()]


def require_dir(path, flag):
    """Fail with the flag's name rather than with a bare OSError from deep inside a loader."""
    if not os.path.isdir(path):
        raise SystemExit(
            '%s does not exist or is not a folder: %s\n'
            'It must be the root that holds one folder per sample, i.e. '
            '<root>/<sample>/seg_5D/cell<id>_5D.tif as written by Data_prep.py.'
            % (flag, path))


def require_file(path, flag):
    """Same, for a file."""
    if not os.path.isfile(path):
        raise SystemExit(
            '%s does not exist or is not a file: %s\n'
            'Give the full path to the checkpoint itself, e.g. '
            '<run folder>/best_model_fine-tune.pth.' % (flag, path))


def discover_samples(base_folder, base_folder2, seg_order):
    """Immediate subfolders of the root(s) that hold something this script can read.

    A folder qualifies when it has a clustered.xlsx (labelled dish) or one of the
    --seg-folder candidates (unlabelled dish) -- the same two branches
    load_finetuning_data() takes below. Used only for --all-samples.
    """
    found = []
    for root in (base_folder, base_folder2):
        if not root:
            continue
        for name in sorted(os.listdir(root)):
            sample_dir = os.path.join(root, name)
            if not os.path.isdir(sample_dir) or name in found:
                continue
            if os.path.exists(os.path.join(sample_dir, 'clustered.xlsx')):
                found.append(name)
                continue
            if any(os.path.isdir(os.path.join(sample_dir, seg)) for seg in seg_order):
                found.append(name)
    return found


def resolve_samples(base_folder, base_folder2, seg_order, seg_flag, samples, all_samples):
    """The sample folder names to score, in the order they will be scored.

    Exactly one of --samples and --all-samples decides the set, and neither of them has a
    default. Unless --out redirects them, scoring a sample writes two workbooks INTO its
    folder, on top of any predict_class_confident_/uncertain_<threshold>.xlsx already
    there, so "every folder under the root" is something you ask for by name rather than
    something you get by typing the shorter command. Same rule as Data_prep.py, whose
    resolve_samples() this mirrors.
    """
    roots = [r for r in (base_folder, base_folder2) if r]

    if samples and all_samples:
        raise SystemExit(
            '--samples and --all-samples contradict each other.\n'
            '--all-samples scores EVERY qualifying sample folder under the root(s), which '
            'is not the subset --samples names. Drop one.')
    if not samples and not all_samples:
        raise SystemExit(
            'Give --samples or --all-samples; there is no default.\n'
            '  --samples sampleA,sampleB   score exactly these sample folders\n'
            '  --all-samples               score every sample folder under the root(s) '
            'that holds a clustered.xlsx or a %s subfolder\n'
            'Unless --out redirects them, scoring a sample OVERWRITES '
            'predict_class_confident_<threshold>.xlsx and '
            'predict_class_uncertain_<threshold>.xlsx inside its folder, so the wider of '
            'the two is deliberately not what you get for leaving a flag off.'
            % ' or '.join(seg_order))

    if samples:
        names = comma_list(samples)
        missing = [s for s in names
                   if not any(os.path.isdir(os.path.join(r, s)) for r in roots)]
        if missing:
            raise SystemExit(
                'These --samples are not folders under %s: %s\n'
                'A name that does not exist would otherwise be scored as zero cells and '
                'reported as "Total predictions: 0", which reads like a data problem.'
                % (' or '.join(roots), ', '.join(missing)))
        return names

    names = discover_samples(base_folder, base_folder2, seg_order)
    if not names:
        raise SystemExit(
            '--all-samples: no sample folder under %s holds a clustered.xlsx or a %s '
            'subfolder (--seg-folder %s).\n'
            'Expected layout: <root>/<sample>/seg_5D/cell<id>_5D.tif  (as written by '
            'Data_prep.py).\n'
            'Point --data-root at the folder that HOLDS the sample folders, not at one '
            'sample, or name them with --samples.'
            % (' or '.join(roots), ' or '.join(seg_order), seg_flag))
    return names


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
    def __init__(self, df, base_dir, base_dir2, max_height, max_width, transform=None,
                 is_test=False, seg_order=SEG_FOLDER_ORDER['auto']):
        self.df = df
        self.base_dir = base_dir
        self.base_dir2 = base_dir2
        self.max_height = max_height
        self.max_width = max_width
        self.transform = transform
        self.is_test = is_test
        self.seg_order = seg_order

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
                # Folder preference only; this loader never switches root, and never has.
                seg_dir = os.path.join(self.base_dir, test_dir, self.seg_order[-1])
                for seg_name in self.seg_order:
                    candidate = os.path.join(self.base_dir, test_dir, seg_name)
                    if os.path.exists(candidate):
                        seg_dir = candidate
                        break
                img_path = os.path.join(seg_dir, f'cell{cell_label}_5D.tif')
                nu_label = int(row['Nu_cluster'])
                mito_label = int(row['Mito_cluster'])
            else:
                img_path = row['output_file']
                nu_label = int(row['nu_class'])
                mito_label = int(row['mito_class'])

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



def load_finetuning_data(test_dirs, base_folder, base_folder2, nu_class_map, mito_class_map,
                         seg_order=SEG_FOLDER_ORDER['auto']):
    data = []
    for test_dir in test_dirs:
        excel_path = os.path.join(base_folder, test_dir, 'clustered.xlsx')
        if not os.path.exists(excel_path) and base_folder2:
            excel_path = os.path.join(base_folder2, test_dir, 'clustered.xlsx')
        if os.path.exists(excel_path):
            # If clustered.xlsx exists, process it as before
            df = pd.read_excel(excel_path)
            for _, row in df.iterrows():
                nu_class = row['Nu_FP']
                mito_class = row['Mito_FP']
                nu_class_num = nu_class_map.get(nu_class, 0) if pd.notna(nu_class) else 0
                mito_class_num = mito_class_map.get(mito_class, 0) if pd.notna(mito_class) else 0
                cell_label = row['Cell_Label']
                data.append({
                    'Directory': test_dir,
                    'Cell_Label': cell_label,
                    'Nu_cluster': nu_class_num,
                    'Mito_cluster': mito_class_num
                })
        else:
            # If clustered.xlsx doesn't exist, process the seg_5D folder.
            # This fallback is asymmetric on purpose and always has been: the first
            # candidate is the preferred folder under the FIRST root, the second is the
            # last-choice folder under the SECOND root. Under --seg-folder auto that is
            # exactly (--data-root, seg_5D_calib) then (--data-root-2, seg_5D). A dish
            # whose crops sit in seg_5D under --data-root alone is therefore NOT found
            # here; pass --seg-folder seg_5D for that layout.
            seg_5d_path = os.path.join(base_folder, test_dir, seg_order[0])
            if not os.path.exists(seg_5d_path) and base_folder2:
                seg_5d_path = os.path.join(base_folder2, test_dir, seg_order[-1])
            if os.path.exists(seg_5d_path):
                tiff_files = glob.glob(os.path.join(seg_5d_path, 'cell*_5D.tif'))
                for tiff_file in tiff_files:
                    cell_label = int(os.path.basename(tiff_file).split('cell')[1].split('_')[0])
                    data.append({
                        'Directory': test_dir,
                        'Cell_Label': cell_label,
                        'Nu_cluster': 0,  # Unknown, set to 0
                        'Mito_cluster': 0  # Unknown, set to 0
                    })
            else:
                print(f"Warning: Neither clustered.xlsx nor seg_5D folder found in {test_dir}")

    df = pd.DataFrame(data)
    return df


def test_model(model, test_dirs, base_folder, base_folder2, nu_class_map, mito_class_map, device,
               confidence_threshold, seg_order=SEG_FOLDER_ORDER['auto'], crop_size=CROP_SIZE,
               out_root='', out_pred=False):
    model.eval()
    max_height = crop_size
    max_width = crop_size

    def calculate_confidence_score(predictions):
        """
        Calculate confidence score based on prediction distribution.
        Returns confidence score and boolean indicating if prediction is reliable.

        Methods used:
        1. Max probability vs second highest (margin)
        2. Entropy of distribution
        3. Ratio of max to mean of others
        """
        # Convert to numpy for easier manipulation
        pred_np = predictions.cpu().numpy()

        # Sort probabilities in descending order
        sorted_probs = np.sort(pred_np)[::-1]

        # Calculate margin between top two predictions
        margin = sorted_probs[0] - sorted_probs[1]

        # Calculate entropy
        entropy = -np.sum(pred_np * np.log(pred_np + 1e-10))
        max_entropy = -np.log(1 / 7)  # Maximum possible entropy for 7 classes
        normalized_entropy = 1 - (entropy / max_entropy)

        # Calculate ratio of max to mean of others
        max_prob = sorted_probs[0]
        mean_others = np.mean(sorted_probs[1:])
        ratio = max_prob / (mean_others + 1e-10)

        # Combine metrics into final confidence score
        confidence_score = (0.4 * margin + 0.3 * normalized_entropy + 0.3 * min(ratio / 10, 1))

        return confidence_score, confidence_score >= confidence_threshold

    test_df_all = load_finetuning_data(test_dirs, base_folder, base_folder2, nu_class_map,
                                       mito_class_map, seg_order)
    for test_dir in test_dirs:
        results = []
        uncertain_results = []
        out_folder = os.path.join(base_folder, test_dir)
        if not os.path.exists(out_folder) and base_folder2:
            out_folder = os.path.join(base_folder2, test_dir)
        if out_root:
            out_folder = os.path.join(out_root, test_dir)
            os.makedirs(out_folder, exist_ok=True)

        test_df = test_df_all[test_df_all['Directory'] == test_dir]

        # Say what was read, so a run is never ambiguous about which crops it scored.
        read_from = None
        for seg_name in seg_order:
            if os.path.isdir(os.path.join(base_folder, test_dir, seg_name)):
                read_from = seg_name
                break
        print('  [%s] seg folder: %-13s cells: %d'
              % (test_dir, read_from if read_from else 'not under --data-root', len(test_df)))

        test_dataset = FluorescenceDataset(test_df, base_folder, base_folder2, max_height, max_width,
                                           is_test=True, seg_order=seg_order)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for i, (images, _, _) in enumerate(test_loader):
                images = images.float().to(device)
                outputs_nu, outputs_mito = model(images)

                # Apply softmax to get probabilities
                probs_nu = torch.softmax(outputs_nu, dim=1).squeeze()
                probs_mito = torch.softmax(outputs_mito, dim=1).squeeze()

                # Calculate confidence scores
                nu_confidence, nu_reliable = calculate_confidence_score(probs_nu)
                mito_confidence, mito_reliable = calculate_confidence_score(probs_mito)

                # Get predictions
                pred_nu = torch.argmax(probs_nu).item()
                pred_mito = torch.argmax(probs_mito).item()

                nu_class = next((k for k, v in nu_class_map.items() if v == pred_nu), 'Unknown')
                mito_class = next((k for k, v in mito_class_map.items() if v == pred_mito), 'Unknown')

                result = {
                    'Directory': test_dir,
                    'Cell_Label': test_df.iloc[i]['Cell_Label'],
                    'Predicted_Nu_Class': nu_class,
                    'Predicted_Mito_Class': mito_class,
                    'Nu_Confidence': f"{nu_confidence:.3f}",
                    'Mito_Confidence': f"{mito_confidence:.3f}",
                    'Nu_Probabilities': probs_nu.cpu().numpy().tolist(),
                    'Mito_Probabilities': probs_mito.cpu().numpy().tolist()
                }

                # Separate results based on confidence
                if nu_reliable and mito_reliable:
                    results.append(result)
                else:
                    uncertain_results.append(result)

        if out_pred:
            # Save confident predictions
            if results:
                results_df = pd.DataFrame(results)
                results_df.to_excel(os.path.join(out_folder, f'predict_class_confident_{confidence_threshold}.xlsx'), index=False)
                print(f"Confident results saved to {os.path.join(out_folder, 'predict_class_confident.xlsx')}")

            # Save uncertain predictions separately
            if uncertain_results:
                uncertain_df = pd.DataFrame(uncertain_results)
                uncertain_df.to_excel(os.path.join(out_folder, f'predict_class_uncertain_{confidence_threshold}.xlsx'), index=False)
                print(f"Uncertain results saved to {os.path.join(out_folder, 'predict_class_uncertain.xlsx')}")

            # Print statistics
            total = len(results) + len(uncertain_results)
            print(f"\nConfidence Statistics for {test_dir}:")
            print(f"Total predictions: {total}")
            if total:
                print(f"Confident predictions: {len(results)} ({len(results) / total * 100:.1f}%)")
                print(f"Uncertain predictions: {len(uncertain_results)} ({len(uncertain_results) / total * 100:.1f}%)")

def main():
    ap = argparse.ArgumentParser(
        description='Score prepared per-cell crops with a trained LUMINA checkpoint and '
                    'write the confident and uncertain predictions per sample folder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--checkpoint', required=True,
                    help='Trained LUMINA state_dict, e.g. best_model_fine-tune.pth. Give the '
                         'full path to the file: a stage-1 checkpoint '
                         '(best_model_initial.pth) is just a different path, not a code edit.')
    ap.add_argument('--data-root', required=True,
                    help='Root holding one folder per sample: <root>/<sample>/<seg folder>/'
                         'cell<id>_5D.tif, see --seg-folder. Unless --out is given, the two '
                         'result workbooks are written back into each sample folder.')
    ap.add_argument('--data-root-2', default='',
                    help='Optional second root, searched after --data-root. It steers where '
                         'clustered.xlsx and the unlabelled crop glob are looked for and '
                         'where results land, but NOT the image loader, which only ever '
                         'reads --data-root. That asymmetry is long-standing behaviour and '
                         'is preserved: a sample found only under the second root is '
                         'enumerated and then fails to load. Prefer keeping one root '
                         'complete over splitting a dataset across two.')
    ap.add_argument('--samples', default='',
                    help='Comma-separated sample folder names to score. Required unless '
                         '--all-samples is given; there is no default, because unless --out '
                         'redirects them the workbooks are written INTO each sample folder '
                         'and overwrite the ones already there. The shipped script scored a '
                         'hand-edited list of folder names, so naming the folders here is '
                         'the direct equivalent of editing that list.')
    ap.add_argument('--all-samples', action='store_true',
                    help='Score every immediate subfolder of the root(s) that holds a '
                         'clustered.xlsx or a --seg-folder candidate, instead of naming '
                         'them with --samples. Unless --out redirects the output, this '
                         'OVERWRITES predict_class_confident_<threshold>.xlsx and '
                         'predict_class_uncertain_<threshold>.xlsx in every one of those '
                         'folders, including folders you were not thinking about. The '
                         'resolved list of destinations is printed in full before the first '
                         'workbook is written, so check it there.')
    ap.add_argument('--seg-folder', default='auto',
                    choices=['auto', 'seg_5D', 'seg_5D_calib'],
                    help='Which folder of prepared crops to read under each sample. "auto" '
                         'takes seg_5D_calib when present and seg_5D otherwise, which is '
                         'what this script has always done. Data_prep.py writes '
                         'seg_5D_calib and Train_LUMINA.py reads seg_5D, so pass the name '
                         'explicitly when a dish has both and you care which one was '
                         'scored. The folder actually read is printed per sample.')
    ap.add_argument('--out', default='',
                    help='Optional output root. Default: empty, meaning the workbooks are '
                         'written in place, into each sample folder beside its crops, which '
                         'is where Visualize_heatmap.py looks for them. If you redirect, '
                         'results land in <out>/<sample>/ and the heatmap must be pointed '
                         'at <out> instead of at --data-root.')

    ap.add_argument('--num-classes', type=int, default=8,
                    help='Output units per head. Must match the checkpoint, or '
                         'load_state_dict fails on fc_nu.6.weight / fc_mito.6.weight. Index '
                         '0 is reserved for "no anchor" and is never a class name.')
    ap.add_argument('--crop-size', type=int, default=CROP_SIZE,
                    help='Canvas each crop is centre-padded onto. A crop LARGER than this '
                         'is not resized: the loader skips it by advancing to the next row, '
                         'which shifts every following Cell_Label in that folder. Leave it '
                         'at the value the checkpoint was trained with.')
    ap.add_argument('--device', default='cuda:0',
                    help='Torch device string: cuda:0, cuda:1, cpu. There is no automatic '
                         'fallback -- cuda on a machine without a GPU fails rather than '
                         'silently running slowly on the CPU.')

    ap.add_argument('--confidence-threshold', type=float, default=0.6,
                    help='A cell is confident when BOTH heads score at least this on the '
                         'composite score. It is also part of the output file name '
                         '(predict_class_confident_<threshold>.xlsx), so Visualize_heatmap.py '
                         'must be run with the same value or it finds no file. '
                         'Finetune_LUMINA.py defaults the same flag to 0.9; that is not a '
                         'typo, they are different measurements on the same scale.')

    args = ap.parse_args()

    if args.num_classes < 1:
        raise SystemExit('--num-classes must be at least 1.')
    if args.crop_size < 1:
        raise SystemExit('--crop-size must be at least 1.')

    require_dir(args.data_root, '--data-root')
    if args.data_root_2:
        require_dir(args.data_root_2, '--data-root-2')
    require_file(args.checkpoint, '--checkpoint')

    seg_order = SEG_FOLDER_ORDER[args.seg_folder]
    device = torch.device(args.device)
    confidence_threshold = args.confidence_threshold

    test_dirs = resolve_samples(args.data_root, args.data_root_2, seg_order,
                                args.seg_folder, args.samples, args.all_samples)

    print('nu_class_map: %s' % NU_CLASS_MAP)
    print('mito_class_map: %s' % MITO_CLASS_MAP)
    print('device: %s   seg-folder: %s   crop: %d   classes: %d'
          % (device, args.seg_folder, args.crop_size, args.num_classes))
    print('confidence threshold: %.2f   samples: %d   output: %s'
          % (confidence_threshold, len(test_dirs), args.out or 'in place, beside the crops'))

    # Printed BEFORE the model is even built, which is well before the first workbook is
    # written, and printed as one line per folder rather than as a summary: --all-samples
    # resolves its list from a scan of the disk, so this is the only chance to see a
    # mistake while the files it would overwrite are still there.
    print('samples: %d folder(s), from %s'
          % (len(test_dirs), '--samples' if args.samples
             else '--all-samples (every folder under the root(s) with a clustered.xlsx or '
                  'a %s)' % ' or '.join(seg_order)))
    for name in test_dirs:
        dest = os.path.join(args.data_root, name)
        if not os.path.isdir(dest) and args.data_root_2:
            dest = os.path.join(args.data_root_2, name)
        if args.out:
            dest = os.path.join(args.out, name)
        print('    %s' % dest)
    print('writing: <destination>/predict_class_{confident,uncertain}_%s.xlsx'
          % confidence_threshold)
    if not args.out:
        print('NOTE: those two workbooks are written into every folder listed above, on '
              'top of any file of the same name already there. Nothing has been written '
              'yet. Pass --out to send them somewhere else instead.')

    # Every flag, resolved, written once, so a directory of workbooks is self-describing.
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    cfg = {k: v for k, v in sorted(vars(args).items())}
    cfg['resolved_device'] = str(device)
    cfg['resolved_seg_order'] = ','.join(seg_order)
    cfg['resolved_samples'] = ','.join(test_dirs)
    # Exactly how the threshold was spelled into the file names, which is what
    # Visualize_heatmap.py has to match.
    cfg['workbook_suffix'] = '_%s.xlsx' % confidence_threshold
    config_path = os.path.join(args.out or args.data_root, 'test_run_config.csv')
    pd.DataFrame([{'flag': k, 'value': v} for k, v in cfg.items()]).to_csv(config_path, index=False)
    print('run configuration: %s' % config_path)

    # Initial training
    num_classes = args.num_classes
    model = DualHeadConvNet(num_classes).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    # Add this after the fine-tuning section
    test_model(model, test_dirs, args.data_root, args.data_root_2, NU_CLASS_MAP, MITO_CLASS_MAP,
               device, out_pred=True, confidence_threshold=confidence_threshold,
               seg_order=seg_order, crop_size=args.crop_size, out_root=args.out)

if __name__ == '__main__':
    main()
