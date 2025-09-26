import glob
import torch.nn as nn
import os
import pandas as pd
import tifffile as tiff
import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset
import torch


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
    def __init__(self, df, base_dir, base_dir2, max_height, max_width, transform=None, is_test=False):
        self.df = df
        self.base_dir = base_dir
        self.base_dir2 = base_dir2
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
                # seg_dir = os.path.join(self.base_dir, test_dir, 'seg_5D')
                seg_dir = os.path.join(self.base_dir, test_dir, 'seg_5D_calib')
                if not os.path.exists(seg_dir):
                    # seg_dir = os.path.join(self.base_dir2, test_dir, 'seg_5D')
                    seg_dir = os.path.join(self.base_dir, test_dir, 'seg_5D')
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



def load_finetuning_data(test_dirs, base_folder, base_folder2, nu_class_map, mito_class_map):
    data = []
    for test_dir in test_dirs:
        excel_path = os.path.join(base_folder, test_dir, 'clustered.xlsx')
        if not os.path.exists(excel_path):
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
            # If clustered.xlsx doesn't exist, process the seg_5D folder
            seg_5d_path = os.path.join(base_folder, test_dir, 'seg_5D_calib')
            if not os.path.exists(seg_5d_path):
                seg_5d_path = os.path.join(base_folder2, test_dir, 'seg_5D')
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
               confidence_threshold=0.5, out_pred=False):
    model.eval()
    max_height = 256
    max_width = 256

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

    test_df_all = load_finetuning_data(test_dirs, base_folder, base_folder2, nu_class_map, mito_class_map)
    # _, test_df_all = train_test_split(test_df_all, test_size=0.2, random_state=42)
    # read test df all from excel
    # excel_path = r'/gpfs/share/home/2301112465/BC_FLIM/Hek293T/Dual_241105/val_df.xlsx'
    # test_df_all = pd.read_excel(excel_path)
    for test_dir in test_dirs:
        results = []
        uncertain_results = []
        out_folder = os.path.join(base_folder, test_dir)
        if not os.path.exists(out_folder):
            out_folder = os.path.join(base_folder2, test_dir)

        # test_df = load_finetuning_data([test_dir], base_folder, base_folder2, nu_class_map, mito_class_map)
        # split the test_df into 20% for validation by random state 42
        # _, test_df = train_test_split(test_df, test_size=0.2, random_state=42)

        test_df = test_df_all[test_df_all['Directory'] == test_dir]
        test_dataset = FluorescenceDataset(test_df, base_folder, base_folder2, max_height, max_width, is_test=True)
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
    # Define file paths and parameters
    base_folder = r'G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual'
    base_folder2 = r'I:\BC-FLIM\Hek293T-BJMU-Dual'
    # model_folder = r'E:\BC-FLIM\Hek293T-BJMU\Dual_class\Dual_class_0804_6-2heads_FTBN'
    # model_folder = r'I:\BC-FLIM\Hek293T-BJMU-Dual\Dual_241104-classweight'
    # model_folder = r'I:\BC-FLIM\Hek293T-BJMU-Dual\Dual_241127'
    model_folder = r'G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual\Dual_241127'
    # model_folder = r'I:\BC-FLIM\Hek293T-BJMU-Dual\Dual_241128-2'
    # model_folder = r'I:\BC-FLIM\Hek293T-BJMU-Dual\Dual_241202'
    # model_folder = r'I:\BC-FLIM\Hek293T-BJMU-Dual\Dual_241203-2' # without 241117 mix7 to train
    # model_folder = r'I:\BC-FLIM\Hek293T-BJMU-Dual\Dual_241203-1'
    model_path = os.path.join(model_folder, 'best_model_fine-tune.pth')

    max_height = 256
    max_width = 256
    batchsize = 128
    num_epochs = 180 # seems enough
    early_stop_patience = 1000
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    use_finetune = True
    confidence_threshold = 0.6

    # test_dirs = [
        # 'NLS-NTOM-Mix39-240618-1',
        # 'NLS-NTOM-Mix39-240618-2', # those kept for testing
        # 'NLS-NTOM-Mix39-240618-3', # those kept for testing
        # 'NTOM-4-10-13-14-NLS-8-240602-1', # those are not sure about the gt
        # 'NTOM-4-10-13-14-NLS-8-240602-2', # those are not sure about the gt
        # 'NTOM1-4-8-14-16-NLS-13-240618',
        # 'NTOM1-8-10-13-14-16-NLS-4-240618',
        # 'NTOM1-4-8-10-13-14-NLS-16-240618',
        # 'NTOM-14-16-NLS-1-240602-1',
        # 'NTOM-14-16-NLS-1-240602-2',
        # 'NTOM-1-13-14-NLS-4-240602-1',
        # 'NTOM-1-13-14-NLS-4-240602-2', # those are not processed yet
        #     'NTOM-1-4-13-NLS-10-240602-1',
        #     'NTOM-1-4-13-NLS-10-240602-2',
        # 'NTOM-1-4-16-NLS-13-240602',
        # 'NTOM-1-4-10-NLS-14-240602',
        # 'NTOM1-4-8-10-13-NLS-14-240618', # not so sure about the gt
        # 'NTOM-1-4-NLS-16-240602'

        # 'NLS1-NTOM-Mix6-240719'
        # 'NLS16-NTOM-Mix6-240722-1',
        # 'NLS16-NTOM-Mix6-240722-2',
        # 'Mix6-1-241009-1'

        # 'Mix6-2-241009-1'
        # 'Mix6-2-241009-2'
    # ]
    test_dirs = [
        # 'NTOM1-4-8-14-16-NLS-13-240618', 'NTOM1-8-10-13-14-16-NLS-4-240618',
        # 'NTOM1-4-8-10-13-14-NLS-16-240618', 'NTOM-14-16-NLS-1-240602-1',
        # 'NTOM-14-16-NLS-1-240602-2', 'NTOM-1-13-14-NLS-4-240602-1',
        # 'NTOM-1-13-14-NLS-4-240602-2', 'NTOM-1-4-13-NLS-10-240602-1',
        # 'NTOM-1-4-13-NLS-10-240602-2', 'NTOM-1-4-16-NLS-13-240602',
        # # 'NTOM-1-4-10-NLS-14-240602',
        # 'NTOM1-4-8-10-13-NLS-14-240618',
        # 'NTOM-1-4-NLS-16-240602', 'NLS1-NTOM-Mix6-240719',
        # 'NTOM-4-10-13-14-NLS-8-240602-1', 'NTOM-4-10-13-14-NLS-8-240602-2',
        # 'NTOM1-4-10-13-14-NLS-8-240618', 'NLS16-NTOM-Mix6-240722-1',
        # 'NLS16-NTOM-Mix6-240722-2', 'NLS14-NTOM-Mix6-240722-1',
        # 'NLS14-NTOM-Mix6-240722-2',
        # 'NTOM1-4-8-13-14-16-NLS-10-240618',
        # 'NTOM4-8-10-13-14-16-NLS-1-240618', 'NLS1-NTOM8-240922-1',
        # 'NLS1-NTOM8-240922-2', 'NLS4-NTOM8-240922-1', 'NLS4-NTOM8-240922-2',
        # 'NLS10-NTOM8-240922-1', 'NLS10-NTOM8-240922-2', 'NLS13-NTOM10-240922-1',
        # 'NLS13-NTOM10-240922-2', 'NLS10-NTOM16-240924-1', 'NLS10-NTOM16-240924-2',
        # 'NLS8-NTOM16-240924-1', 'NLS8-NTOM16-240924-2', 'NLS4-NTOM16-240924-1',
        # 'NLS4-NTOM16-240924-2', 'NLS10-NTOM14-240924-1', 'NLS10-NTOM14-240924-2',
        # 'NLS8-NTOM13-240924-1', 'NLS8-NTOM13-240924-2', 'NLS8-NTOM13-240924-3',
        # 'NLS8-NTOM4-240929-1', 'NLS8-NTOM4-240929-2',
        # 'NLS16-NTOM10-240929-1',
        # 'NLS16-NTOM10-240929-2',
        # 'NLS4-NTOM10-240924', 'NLS4-NTOM14-240926',
        # 'NLS1-NTOM14-240926', 'NLS1-NTOM4-240926', 'NLS8-NTOM1-240926-1',
        # 'NLS8-NTOM1-240926-2', 'NLS14-NTOM10-240926-1', 'NLS14-NTOM10-240926-2',
        # 'NLS10-NTOM13-240929-1', 'NLS10-NTOM13-240929-2', 'NLS16-NTOM8-240926-1',
        # 'NLS16-NTOM8-240926-2', 'NLS1-NTOM13-240922',
        # 'NLS16-NTOM13-240929-1', 'NLS16-NTOM13-240929-2',
        # 'NLS14-NTOM16-241020-1',
        # 'NLS14-NTOM16-241020-2',
        # 'NLS14-NTOM16-241020-3',
        # 'NLS4-Mix6-240719',
        # 'NLS10-Mix6-240719',
        # 'NLS13-Mix6-240719',
        # 'NLS14-Mix6-240719',
        # 'NLS16-Mix6-240719-2',
        # 'NLS16-Mix6-240719-1',

        # 'Mix6-1-241009-1',
        # 'Mix6-1-241009-2',
        # 'Mix6-1-241009-3',


        # 'Mix6-2-241009-1',
        # 'Mix6-2-241009-2',

        # 'Mix7-1-241117',
        # 'Mix7-2-241117',
        # 'Mix7-3-241117',

        # 'Mix6-1-241020-1',
        # 'Mix6-1-241020-2',
        # 'Mix6-1-241020-3',

        # 'NLS-NTOM-Mix39-240618-1',
        # 'NLS-NTOM-Mix39-240618-2',
        # 'NLS-NTOM-Mix39-240618-3',

        # 'Mix30-1-240719',
        # 'Mix30-2-240719',
        # 'Mix30-3-240719',

        # 'Mix7-241205-1',
        # 'Mix7-241205-2',

        # 'Mix7-241205-40X-3',
        # 'Mix7-241205-40X-4',
    # 'Mix7-2-250103-1',
    # 'Mix7-2-250103-2',
    # 'Mix7-1-250103-1',
    # 'Mix7-1-250103-2',
    #     'Mix42-250601-1',
    #     'Mix42-250601-2',
    #     'Mix42-250602-1',
    #     'Mix42-250602-2',
    #     'Mix42-250602-3',
        # 'Mix42-250602-4',
        # 'Mix42-250602-5',
        # 'Mix42-250602-6',
        # 'Mix35-250616-1',
        # 'Mix35-250616-2',
        # 'Mix35-250616-3',
        # 'Mix35-250616-4',
        # 'Mix35-250616-5',
        # 'Mix35-250616-6',
        # 'Mix36-250624-1',
        # 'Mix36-250624-2',
        # 'Mix36-250624-3',
        'Mix36-250624-4',
        # 'Mix36-250624-5',
        # 'Mix36-250624-6',
        # 'Mix36-250624-7',
        # 'Mix36-250624-8',
        # 'Mix36-250624-9',
        # 'Mix36-250624-10',
        # 'Mix36-250624-11',
        'Mix36-250624-12',
        # 'Mix36-250624-13',
        # 'Mix36-250624-14',
        # 'Mix36-250624-15',
        # 'Mix36-250624-16',
        # 'Mix36-250624-17',
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
        'M13': ['NTOM-M13-240629'],
        'M4': ['NTOM-M4-240629'],
        'M14': ['NTOM-M14-240629'],
        'M16': ['NTOM-M16-240629'],
        'M8': ['NTOM-M8-240629'],
        'M1': ['NTOM-M1-240629'],

    }


    nu_class_map = {key: idx + 1 for idx, key in enumerate(nu_files.keys())}
    mito_class_map = {key: idx + 1 for idx, key in enumerate(mito_files.keys())}
    print(f'nu_class_map: {nu_class_map}')
    print(f'mito_class_map: {mito_class_map}')

    # Initial training
    num_classes = 8
    model = DualHeadConvNet(num_classes).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    # Add this after the fine-tuning section
    test_model(model, test_dirs, base_folder, base_folder2, nu_class_map, mito_class_map, device, out_pred=True, confidence_threshold=confidence_threshold)

if __name__ == '__main__':
    main()
