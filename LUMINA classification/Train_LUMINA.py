from collections import Counter

import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import models
import os
import pandas as pd
import tifffile as tiff
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

def resize_image(image, height, width):
    resized_image = np.zeros((image.shape[0], height, width), dtype=image.dtype)
    for i in range(image.shape[0]):
        resized_image[i] = cv2.resize(image[i], (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image

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
    def __init__(self, df, base_dir, max_height, max_width, transform=None, is_test=False):
        self.df = df
        self.base_dir = base_dir
        self.max_height = max_height
        self.max_width = max_width
        self.transform = transform
        self.is_test = is_test

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
                seg_dir = os.path.join(self.base_dir, test_dir, 'seg_5D')
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

def load_training_data(nu_files, mito_files, base_folder):
    data = []
    nu_class_map = {key: idx + 1 for idx, key in enumerate(nu_files.keys())}
    mito_class_map = {key: idx + 1 for idx, key in enumerate(mito_files.keys())}
    print(f'nu_class_map: {nu_class_map}')
    print(f'mito_class_map: {mito_class_map}')
    for nu_class, files in nu_files.items():
        nu_class_num = nu_class_map[nu_class]
        for file in files:
            img_dir = os.path.join(base_folder, file, 'seg_5D')
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
            img_dir = os.path.join(base_folder, file, 'seg_5D')
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
    for test_dir in test_dirs:
        excel_path = os.path.join(base_folder, test_dir, 'clustered.xlsx')
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
def train_model(model, train_loader, val_loader, optimizer, num_epochs, phase, out_folder, early_stop_patience=10,
                gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}')
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

        if phase == 'initial' and epoch in [30, 70]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2
        elif phase == 'fine-tune' and epoch in [200, 350, 500]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2

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
    # Define file paths and parameters
    base_folder = r'/gpfs/share/home/2301112465/BC_FLIM/Hek293T'
    test_base_folder = r'/gpfs/share/home/2301112465/BC_FLIM/Hek293T-Dual'
    # pre_model_dir = r'/gpfs/share/home/2301112465/BC_FLIM/Hek293T/Dual_class_0804_6-2heads_FTBN'
    pre_model_dir = r'/gpfs/share/home/2301112465/BC_FLIM/Hek293T/Dual_241127'
    # out_folder = os.path.join(base_folder, 'Dual_class_0804_6-2heads_FTBN')
    # out_folder = os.path.join(base_folder, 'Dual_241104-classweight')
    out_folder = os.path.join(base_folder, 'Dual_241203-3')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    max_height = 256
    max_width = 256
    batchsize = 128
    # batchsize = 224 # 256 may OOM, but make it bigger is better, when locking the conv input heads
    num_epochs = 180 # seems enough
    early_stop_patience = 1000
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    # use_finetune = True
    use_finetune = False
    test_dirs = [
        'NTOM1-4-8-14-16-NLS-13-240618', 'NTOM1-8-10-13-14-16-NLS-4-240618',
        'NTOM1-4-8-10-13-14-NLS-16-240618', 'NTOM-14-16-NLS-1-240602-1',
        'NTOM-14-16-NLS-1-240602-2', 'NTOM-1-13-14-NLS-4-240602-1',
        'NTOM-1-13-14-NLS-4-240602-2', 'NTOM-1-4-13-NLS-10-240602-1',
        'NTOM-1-4-13-NLS-10-240602-2', 'NTOM-1-4-16-NLS-13-240602',
        # 'NTOM-1-4-10-NLS-14-240602',
        'NTOM1-4-8-10-13-NLS-14-240618',
        'NTOM-1-4-NLS-16-240602', 'NLS1-NTOM-Mix6-240719',
        'NTOM-4-10-13-14-NLS-8-240602-1', 'NTOM-4-10-13-14-NLS-8-240602-2',
        'NTOM1-4-10-13-14-NLS-8-240618', 'NLS16-NTOM-Mix6-240722-1',
        'NLS16-NTOM-Mix6-240722-2', 'NLS14-NTOM-Mix6-240722-1',
        'NLS14-NTOM-Mix6-240722-2',
        'NTOM1-4-8-13-14-16-NLS-10-240618',
        'NTOM4-8-10-13-14-16-NLS-1-240618', 'NLS1-NTOM8-240922-1',
        'NLS1-NTOM8-240922-2', 'NLS4-NTOM8-240922-1', 'NLS4-NTOM8-240922-2',
        'NLS10-NTOM8-240922-1', 'NLS10-NTOM8-240922-2', 'NLS13-NTOM10-240922-1',
        'NLS13-NTOM10-240922-2', 'NLS10-NTOM16-240924-1', 'NLS10-NTOM16-240924-2',
        'NLS8-NTOM16-240924-1', 'NLS8-NTOM16-240924-2', 'NLS4-NTOM16-240924-1',
        'NLS4-NTOM16-240924-2', 'NLS10-NTOM14-240924-1', 'NLS10-NTOM14-240924-2',
        'NLS8-NTOM13-240924-1', 'NLS8-NTOM13-240924-2', 'NLS8-NTOM13-240924-3',
        'NLS8-NTOM4-240929-1', 'NLS8-NTOM4-240929-2',
        'NLS16-NTOM10-240929-1',
        'NLS16-NTOM10-240929-2',
        'NLS4-NTOM10-240924', 'NLS4-NTOM14-240926',
        'NLS1-NTOM14-240926', 'NLS1-NTOM4-240926', 'NLS8-NTOM1-240926-1',
        'NLS8-NTOM1-240926-2', 'NLS14-NTOM10-240926-1', 'NLS14-NTOM10-240926-2',
        'NLS10-NTOM13-240929-1', 'NLS10-NTOM13-240929-2', 'NLS16-NTOM8-240926-1',
        'NLS16-NTOM8-240926-2', 'NLS1-NTOM13-240922',
        'NLS16-NTOM13-240929-1', 'NLS16-NTOM13-240929-2',
        'NLS14-NTOM16-241020-1',
        'NLS14-NTOM16-241020-2',
        'NLS14-NTOM16-241020-3',
        'NLS4-Mix6-240719',
        'NLS10-Mix6-240719',
        'NLS13-Mix6-240719',
        'NLS14-Mix6-240719',
        'NLS16-Mix6-240719-2',
        'NLS16-Mix6-240719-1',

        'NLS4-NTOM13-241121-2',
        'NLS4-NTOM13-241121-1',
        'NLS16-NTOM1-241121',
        'NLS8-NTOM10-241121',
        'NLS8-NTOM14-241121-1',
        'NLS8-NTOM14-241121-2',
        'NLS14-NTOM13-241121',

        'Mix7-1-241117',
        'Mix7-2-241117',
        'Mix7-3-241117',

        'Mix6-1-241020-1',
        'Mix6-1-241020-2',
        'Mix6-1-241020-3',

        'Mix6-2-241009-1',
        'Mix6-2-241009-2',

        'Mix6-1-241009-1',
        'Mix6-1-241009-2',
        'Mix6-1-241009-3',
    ]

    nu_files = {
        # 'N10': ['NLS-mScarlet3-240222-1', 'NLS-mScarlet3-240222-2'],
        # 'N13': ['NLS-mScarlet-I3-240222-1', 'NLS-mScarlet-I3-240222-2'],
        # 'N4': ['NLS-mScarlet-I-240229-1', 'NLS-mScarlet-I-240229-2'],
        # 'N14': ['NLS-mApple-240229-1', 'NLS-mApple-240229-2'],
        # 'N16': ['NLS-FR-MQ-240308-1', 'NLS-FR-MQ-240308-2'],
        # 'N8': ['NLS-mCherry-240222-1', 'NLS-mCherry-240222-2'],
        # 'N1': ['NLS-mScarlet-H-240222-1', 'NLS-mScarlet-H-240222-2'],
        'N10': ['NLS-N10-240623'],
        'N13': ['NLS-N13-240623'],
        'N4': ['NLS-N4-240623'],
        'N14': ['NLS-N14-240623'],
        'N16': ['NLS-N16-240623'],
        'N8': ['NLS-N8-240623'],
        # 'N1': [ 'NLS-N1-240623-1', 'NLS-N1-240623-2', 'NLS-N1-240622'],
        'N1': ['NLS-N1-240623-1'],
    }

    mito_files = {
        'M10': ['NTOM-M10-240629'],
        'M13': [ 'NTOM-M13-240629'],
        'M4': [ 'NTOM-M4-240629'],
        'M14': [ 'NTOM-M14-240629'],
        'M16': ['NTOM-M16-240629'],
        'M8': [ 'NTOM-M8-240629'],
        'M1': [ 'NTOM-M1-240629'],

    }

    # Load training and fine-tuning data
    train_df, nu_class_map, mito_class_map = load_training_data(nu_files, mito_files, base_folder)
    test_df = load_finetuning_data(test_dirs, test_base_folder, nu_class_map, mito_class_map)

    # Split training data for validation
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Split test data for fine-tuning
    test_train_df, test_val_df = train_test_split(test_df, test_size=0.2, random_state=42)

    # save the val df to excel for later use
    test_val_df.to_excel(os.path.join(out_folder, 'val_df.xlsx'), index=False)
    print(f'test_val_df saved to {os.path.join(out_folder, "val_df.xlsx")}')
    # Create datasets and dataloaders
    train_dataset = FluorescenceDataset(train_df, base_folder, max_height, max_width)
    val_dataset = FluorescenceDataset(val_df, base_folder, max_height, max_width)
    test_train_dataset = FluorescenceDataset(test_train_df, test_base_folder, max_height, max_width, is_test=True)
    test_val_dataset = FluorescenceDataset(test_val_df, test_base_folder, max_height, max_width, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    test_train_loader = DataLoader(test_train_dataset, batch_size=batchsize, shuffle=True)
    test_val_loader = DataLoader(test_val_dataset, batch_size=batchsize, shuffle=False)

    # Initial training
    num_classes = 8
    model = DualHeadConvNet(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Check if user wants to skip initial training and load a pre-trained model
    use_pretrained = True  # Change to True if you want to use a pre-trained model
    if use_pretrained:
        # model.load_state_dict(torch.load(os.path.join(out_folder, 'best_model_initial.pth'), map_location=device))
        # model.load_state_dict(torch.load(os.path.join(pre_model_dir, 'best_model_initial.pth'), map_location=device))
        # use map_location to avoid loading model trained on different device
        model.load_state_dict(torch.load(os.path.join(pre_model_dir, 'best_model_fine-tune.pth'), map_location=device))
    else:
        if use_finetune:
            train_losses, val_losses, train_accuracies, val_accuracies = train_model(
                model, train_loader, val_loader, optimizer, num_epochs, phase='initial', early_stop_patience=early_stop_patience, out_folder=out_folder,
                gpu_id=gpu_id
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
        for head in model.input_heads:
            for param in head.parameters():
                param.requires_grad = False
        # for param in model.backbone.parameters():
        #     param.requires_grad = False

    if use_finetune:
        freeze_conv_layers(model)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # optimizer.add_param_group({'params': model.fc_mito.parameters()})
    num_epochs = 800
    # test_train_losses, test_val_losses, test_train_accuracies, test_val_accuracies = train_model(
    #     model, test_train_loader, test_val_loader, optimizer, num_epochs, phase='fine-tune', out_folder=out_folder,
    #     # model, test_train_loader, test_val_loader, optimizer, num_epochs, phase='initial', base_folder=base_folder,
    #     early_stop_patience=early_stop_patience, gpu_id=gpu_id
    # )
    train_model(
        model, test_train_loader, test_val_loader, optimizer, num_epochs, phase='fine-tune', out_folder=out_folder,
        # model, test_train_loader, test_val_loader, optimizer, num_epochs, phase='initial', base_folder=base_folder,
        early_stop_patience=early_stop_patience, gpu_id=gpu_id
    )


if __name__ == '__main__':
    main()
