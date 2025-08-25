import os
import cv2
import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

# -------------------------------------------------------------------
# USER‐EDITABLE PARAMETERS
# -------------------------------------------------------------------
# Provide 3 factors [f1,f2,f3] (then f4=1.0) or 4 factors [f1,f2,f3,f4]
calibration_factors = [19.0009, 14.3886, 13.2671, 11.8055]
# Phasor calibration parameters
phi_calib = -0.0125
m_calib   =  1.0292
# phi_calib = -0.018
# m_calib   =  1.027
# phi_calib = 0
# m_calib   = 1.0
# Binning / smoothing / lifetime options
bin_size           = 1      # e.g. 1, 2, 4...
smooth_option      = None   # 'median' / 'wavelet' / None
calculate_lifetime = False
intensity_threshold = 100 * bin_size * bin_size

# PEAK / PHASOR parameters
tail_only      = True
PEAK_OFFSET    = 4
END_OFFSET     = 18
peak2_begin    = 77
peak2_end      = 84
tau_resolution = 0.09696969696999999  # ns

# Paths & cell types
cell_types = [
    # 'Mix42-250602-1',
    # 'Mix42-250601-1',
    # 'Mix42-250601-2',
    # 'Mix42-250602-2',
    # 'Mix42-250602-3',
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
cell       = 'Hek293T'
instrument = 'BJMU-Dual'
dataset    = f'{cell}-{instrument}'
data_dir   = fr"G:\BC-FLIM-S\WBY\{dataset}"
# -------------------------------------------------------------------

# unpack calibration_factors
if len(calibration_factors) == 3:
    f1, f2, f3 = calibration_factors
    f4 = 1.0
elif len(calibration_factors) == 4:
    f1, f2, f3, f4 = calibration_factors
else:
    raise ValueError("calibration_factors must have length 3 or 4")

def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau) + c

def calcu_phasor_info(roi_decay, total_intensity, peak_idx):
    # (identical to your existing implementation)
    if tail_only:
        start = peak_idx + PEAK_OFFSET
    else:
        start = 0
    end = len(roi_decay) - END_OFFSET
    if start >= end:
        return 0,0,0,0,0,0
    seg = roi_decay[start:end].astype(float)
    seg /= seg.max()
    t = np.arange(len(seg)) * tau_resolution

    # optional smoothing
    if smooth_option == 'median':
        seg = medfilt(seg, kernel_size=3)
    elif smooth_option == 'wavelet':
        seg = pywt.dwt(seg, 'db1')

    fast = np.sum(seg * t) / np.sum(seg)

    if calculate_lifetime:
        popt, _ = curve_fit(exp_func, t, seg, p0=[1,2,0])
        tau   = popt[1]
        chi2  = np.sum((seg - exp_func(t, *popt))**2)
    else:
        tau, chi2 = 0, 0

    freq = 78.1/1000
    g = np.sum(seg * np.cos(2*np.pi*freq*t)) / np.sum(seg)
    s = np.sum(seg * np.sin(2*np.pi*freq*t)) / np.sum(seg)

    return g, s, tau, chi2, fast, total_intensity

def pad_image(img, bs):
    pad_h = (bs - img.shape[0]%bs)%bs
    pad_w = (bs - img.shape[1]%bs)%bs
    return np.pad(img, ((0,pad_h),(0,pad_w)), mode='constant')

def binning_2d(img, bs):
    if bs==1: return img
    p = pad_image(img, bs)
    return p.reshape(p.shape[0]//bs, bs, p.shape[1]//bs, bs).sum(axis=(1,3))

def binning_3d(img, bs):
    out = []
    for k in range(img.shape[0]):
        out.append(binning_2d(img[k], bs))
    return np.stack(out, axis=0)

# -------------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------------
for cell_type in cell_types:
    raw_dir = os.path.join(data_dir, cell_type, 'raw')
    for fname in os.listdir(raw_dir):
        fov = os.path.splitext(fname)[0]
        print(f"Processing {cell_type} / {fov} …")

        # load segmentation
        seg_path = os.path.join(data_dir, cell_type, 'intensity', f'{fov}-sum_seg.npy')
        masks = np.load(seg_path, allow_pickle=True).item()['masks']

        # load FLIM & intensity frames
        stack_sum = tiff.imread(os.path.join(data_dir, cell_type, 'flim_stack',  f'{fov}-sum.tif'))
        I1 = cv2.imread(os.path.join(data_dir, cell_type, 'intensity', f'{fov}-1.tif'), -1)
        I2 = cv2.imread(os.path.join(data_dir, cell_type, 'intensity', f'{fov}-2.tif'), -1)
        I3 = cv2.imread(os.path.join(data_dir, cell_type, 'intensity', f'{fov}-3.tif'), -1)
        I4 = cv2.imread(os.path.join(data_dir, cell_type, 'intensity', f'{fov}-4.tif'), -1)

        # apply calibration factors
        C1 = f1 * I1
        C2 = f2 * I2
        C3 = f3 * I3
        C4 = f4 * I4
        Csum = C1 + C2 + C3 + C4

        # calibrated ratios
        int_ratio_1 = C1 / Csum
        int_ratio_2 = C2 / Csum
        int_ratio_3 = C3 / Csum
        # (if you ever need the 4th: int_ratio_4 = C4/Csum)

        # binning
        decay_data   = stack_sum
        b_i_sum      = binning_2d(Csum, bin_size)
        b_decay_data = binning_3d(decay_data, bin_size)

        out_dir = os.path.join(data_dir, cell_type, 'seg_5D_calib')
        # if exist, clear it
        if os.path.exists(out_dir):
            for file in os.listdir(out_dir):
                file_path = os.path.join(out_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared existing files in {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        for cid in tqdm(range(1, masks.max() + 1)):
            # 1) extract this cell’s binary mask
            mask = (masks == cid)
            if not mask.any():
                continue

            # 2) bin that mask to match b_i_sum dimensions
            mask_b = binning_2d(mask.astype(np.uint8), bin_size).astype(bool)

            # 3) placeholder arrays for per-pixel phasors
            c_g = np.zeros_like(b_i_sum, dtype=float)
            c_s = np.zeros_like(b_i_sum, dtype=float)

            # 4) loop ONLY over the mask pixels
            ys, xs = np.where(mask_b)
            for i, j in zip(ys, xs):
                tot = b_i_sum[i, j]
                if tot < intensity_threshold:
                    continue
                roi = b_decay_data[:, i, j]
                pidx = np.argmax(roi)
                g, s, _, _, _, _ = calcu_phasor_info(roi, tot, pidx)
                c_g[i, j] = g
                c_s[i, j] = s

            # 5) apply phasor‐space calibration
            phi = np.arctan2(c_s, c_g)
            m = np.sqrt(c_g ** 2 + c_s ** 2)
            phi_c = phi + phi_calib
            m_c = m * m_calib
            c_gc = m_c * np.cos(phi_c)
            c_sc = m_c * np.sin(phi_c)

            # 6) binned intensity ratios (already computed)
            ci1 = binning_2d(int_ratio_1, bin_size) * mask_b
            ci2 = binning_2d(int_ratio_2, bin_size) * mask_b
            ci3 = binning_2d(int_ratio_3, bin_size) * mask_b
            isum = b_i_sum * mask_b  # already binned Csum

            # 7) stack your channels: [g, s, g_cal, s_cal, int1, int2, int3, isum]
            cell_stack = np.stack([c_gc, c_sc, ci1, ci2, ci3, isum], axis=0)

            # 8) crop to ROI bounds and write
            # get the crop bounds
            y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            crop = cell_stack[:, y0:y1, x0:x1]

            tiff.imwrite(os.path.join(out_dir, f'cell{cid}_5D.tif'), crop)

        print(f" → Saved calibrated 5D cells for {fov}")
