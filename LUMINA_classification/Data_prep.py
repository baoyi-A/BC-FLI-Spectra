"""Prepare the per-cell six-plane crops the LUMINA network reads.

For every segmented cell of every field of view, this writes one small float64 stack:

    plane 0  G          calibrated phasor coordinate, per pixel
    plane 1  S          calibrated phasor coordinate, per pixel
    plane 2  ratio 1    calibrated detector-1 intensity / calibrated total
    plane 3  ratio 2    calibrated detector-2 intensity / calibrated total
    plane 4  ratio 3    calibrated detector-3 intensity / calibrated total
    plane 5  intensity  the calibrated total, in RAW units (not normalised here)

WHAT IT EXPECTS ON DISK, under --data-root

    <root>/<sample>/raw/<fov>.<ext>                one entry per field of view; only the
                                                  file STEM is used, the contents are not
    <root>/<sample>/flim_stack/<fov>-sum.tif       the decay stack, (time bins, h, w)
    <root>/<sample>/intensity/<fov>-1.tif  ... -4.tif   the four detector images
    <root>/<sample>/intensity/<fov>-sum_seg.npy    a pickled dict with a 'masks' label
                                                   image: 0 = background, 1..N = cells

WHAT IT WRITES

    <root>/<sample>/<seg folder>/cell<id>_5D.tif   one crop per cell, see --seg-folder
    <root>/<sample>/data_prep_run_config.csv       every flag and its resolved value

THOSE NAMES ARE A CONTRACT, NOT A CONVENIENCE
    Train_LUMINA.py, Test_LUMINA.py and Finetune_LUMINA.py all build these paths
    themselves. In particular the SLIC napari plugin that ships in this repository writes
    its own per-channel images and masks under DIFFERENT names (`<fov>_ch1.tif`,
    `<fov>_sum_seg_n.npy`), and nothing in this folder converts between the two layouts --
    a plugin output folder has to be renamed by hand before this script will see it.

WHAT MUST MATCH THE MICROSCOPE THAT TOOK THE DATA
    --calibration-factors, --phi-calib, --m-calib and --rep-rate-mhz describe the
    instrument, not the sample. The defaults are the values this script shipped with, i.e.
    the ones the prepared crops in this project were made with; they are meaningless on
    another microscope and carrying them over silently moves every G and S. --rep-rate-mhz
    in particular is easy to miss because it is the laser, not a processing choice.

WHICH SAMPLE FOLDERS IT TOUCHES IS NEVER A DEFAULT
    Writing a sample CLEARS its output folder first, so the set of folders is asked for
    explicitly: either name them with --samples, or say --all-samples to mean every sample
    folder under --data-root. Passing neither is an error. The resolved list is printed in
    full, one path per line, before anything is deleted.

Usage:
    python Data_prep.py --data-root /path/to/sample_root --samples sampleA,sampleB
    python Data_prep.py --data-root /path/to/sample_root --all-samples
"""

import argparse
import os

import cv2
import numpy as np
import pywt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import pandas as pd
import tifffile as tiff
from tqdm import tqdm


def comma_list(value):
    """'a, b ,,c' -> ['a', 'b', 'c']. The comma-list idiom used by every script here."""
    return [s.strip() for s in value.split(',') if s.strip()]


def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau) + c

def calcu_phasor_info(roi_decay, total_intensity, peak_idx,
                      tail_only, PEAK_OFFSET, END_OFFSET, smooth_option,
                      calculate_lifetime, tau_resolution, rep_rate_mhz):
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

    freq = rep_rate_mhz/1000
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
# resolving the command line
# -------------------------------------------------------------------

def unpack_calibration_factors(value):
    """'f1,f2,f3[,f4]' -> (f1, f2, f3, f4), with f4 = 1.0 when three are given."""
    parts = comma_list(value)
    try:
        factors = [float(p) for p in parts]
    except ValueError:
        raise SystemExit(
            '--calibration-factors must be plain numbers separated by commas, got: %s\n'
            'Example: --calibration-factors 19.0009,14.3886,13.2671,11.8055' % value)
    if len(factors) == 3:
        f1, f2, f3 = factors
        f4 = 1.0
    elif len(factors) == 4:
        f1, f2, f3, f4 = factors
    else:
        raise SystemExit(
            '--calibration-factors must have length 3 or 4, got %d: %s\n'
            'Give three factors [f1,f2,f3] and the fourth is taken as 1.0, or give all '
            'four. They scale the four detector images before every ratio and every '
            'phasor in the output, so a wrong count is fatal rather than a warning.'
            % (len(factors), value))
    return f1, f2, f3, f4


def resolve_samples(root, samples, all_samples):
    """The sample folder names to process, in the order they will be processed.

    Exactly one of --samples and --all-samples decides the set, and neither of them has a
    default. Processing a sample folder CLEARS its output folder first, so "every folder
    under --data-root" is something you ask for by name rather than something you get by
    typing the shorter command.
    """
    if not os.path.isdir(root):
        raise SystemExit('--data-root does not exist or is not a folder: %s' % root)

    if samples and all_samples:
        raise SystemExit(
            '--samples and --all-samples contradict each other.\n'
            '--all-samples processes EVERY sample folder under --data-root, which is not '
            'the subset --samples names. Drop one.')
    if not samples and not all_samples:
        raise SystemExit(
            'Give --samples or --all-samples; there is no default.\n'
            '  --samples sampleA,sampleB   process exactly these sample folders\n'
            '  --all-samples               process every sample folder under --data-root '
            'that has a raw/\n'
            'Processing a sample CLEARS its output folder before writing, so the wider of '
            'the two is deliberately not what you get for leaving a flag off.')

    if samples:
        names = comma_list(samples)
        missing = [s for s in names if not os.path.isdir(os.path.join(root, s))]
        if missing:
            raise SystemExit('These --samples are not folders under %s: %s'
                             % (root, ', '.join(missing)))
        no_raw = [s for s in names if not os.path.isdir(os.path.join(root, s, 'raw'))]
        if no_raw:
            raise SystemExit(
                'These --samples have no raw/ subfolder under %s: %s\n'
                'Expected layout: <root>/<sample>/raw/<fov>, with flim_stack/ and '
                'intensity/ beside it.\n'
                'raw/ is what the fields of view are enumerated from, so a sample without '
                'it is fatal rather than skipped: a silent skip is indistinguishable from '
                'a sample that legitimately produced no cells.' % (root, ', '.join(no_raw)))
        return names

    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d, 'raw')))
    if not names:
        raise SystemExit(
            '--all-samples: no sample folder under --data-root %s contains a raw/ '
            'subfolder.\n'
            'Expected layout: <root>/<sample>/raw/<fov>, with flim_stack/ and intensity/ '
            'beside it.\n'
            'Point --data-root at the folder that HOLDS the sample folders, not at one '
            'sample, or name them with --samples.' % root)
    return names


def main():
    ap = argparse.ArgumentParser(
        description='Prepare per-cell six-plane crops (G, S, three calibrated intensity '
                    'ratios, calibrated total intensity) for the LUMINA network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--data-root', required=True,
                    help='Root holding one sample folder per acquisition: '
                         '<root>/<sample>/raw/<fov>, <root>/<sample>/flim_stack/'
                         '<fov>-sum.tif, <root>/<sample>/intensity/<fov>-{1..4}.tif and '
                         '<fov>-sum_seg.npy. Crops are written back into each sample '
                         'folder, so this root must be writable.')
    ap.add_argument('--samples', default='',
                    help='Comma-separated sample folder names to process. Required unless '
                         '--all-samples is given; there is no default, because processing a '
                         'sample CLEARS its output folder first. This script used to carry '
                         'a hand-edited list of folder names with most entries commented '
                         'out, so naming the folders here is the direct equivalent of '
                         'uncommenting two of them.')
    ap.add_argument('--all-samples', action='store_true',
                    help='Process every sample folder under --data-root that contains a '
                         'raw/ folder, instead of naming them with --samples. This CLEARS '
                         'and REGENERATES the output folder (see --seg-folder) of every one '
                         'of them: every file already in <sample>/<seg folder>/ is deleted '
                         'and the crops are rebuilt from the raw data. Prepared crops that '
                         'came from anywhere else, and any hand curation done inside those '
                         'folders, are lost. The resolved list of folders is printed in '
                         'full before the first deletion, so check it there.')
    ap.add_argument('--seg-folder', default='seg_5D_calib',
                    help='Subfolder of each sample folder the crops are written to. The '
                         'default is what this script has always written and what '
                         'Test_LUMINA.py and Finetune_LUMINA.py prefer; Train_LUMINA.py '
                         'reads seg_5D only, so pass --seg-folder seg_5D to prepare data '
                         'for training instead of renaming the folder afterwards.')

    ap.add_argument('--calibration-factors', default='19.0009,14.3886,13.2671,11.8055',
                    help='Three or four per-detector gain factors, comma-separated. With '
                         'three, the fourth is 1.0. They multiply the four intensity '
                         'images before the ratios and the total are formed, so they set '
                         'the spectral coordinates of every crop. Instrument-specific: '
                         'measure them on your own microscope, do not carry these over.')
    ap.add_argument('--phi-calib', type=float, default=-0.0125,
                    help='Phasor phase correction, in radians, added to every pixel phase.')
    ap.add_argument('--m-calib', type=float, default=1.0292,
                    help='Phasor modulation correction, multiplying every pixel radius. '
                         '--phi-calib 0 --m-calib 1.0 is the identity, i.e. uncalibrated '
                         'phasor coordinates. Instrument-specific, like --phi-calib: both '
                         'come from a reference dye measured on the same setup.')
    ap.add_argument('--rep-rate-mhz', type=float, default=78.1,
                    help='Laser repetition rate in MHz. This is the phasor frequency and '
                         'it scales every G and S; it is a property of the laser, so check '
                         'the acquisition metadata rather than assuming the default.')
    ap.add_argument('--tau-resolution', type=float, default=0.09696969696999999,
                    help='Nanoseconds per time bin of the decay stack. Check the '
                         'acquisition metadata. --help prints the shortest form that '
                         'round-trips to the shipped value; every G and S moves with it.')

    ap.add_argument('--peak-offset', type=int, default=4,
                    help='Time bins after the per-pixel decay maximum where the phasor '
                         'window starts. Ignored when --no-tail-only is given.')
    ap.add_argument('--end-offset', type=int, default=18,
                    help='Time bins dropped from the noisy end of the decay.')
    ap.add_argument('--bin-size', type=int, default=1,
                    help='Square spatial binning by summation, applied to the decay, the '
                         'intensities and the masks alike. 1 disables it. Note the crops '
                         'get smaller, and that --intensity-threshold scales with it.')
    ap.add_argument('--smooth', default='none', choices=['none', 'median', 'wavelet'],
                    help='Optional smoothing of each pixel decay before the phasor is '
                         'taken. "median" is a 3-bin median filter. "wavelet" is listed '
                         'because the code path exists, but it does not work: pywt.dwt '
                         'returns a (cA, cD) pair and the next line broadcasts it against '
                         'the time axis, so the run dies. It is left as it was found '
                         'rather than quietly changed.')
    ap.add_argument('--intensity-threshold', type=int, default=0,
                    help='Minimum binned total intensity for a pixel to get a phasor; '
                         'quieter pixels keep G = S = 0. 0 means auto, which is '
                         '100 * --bin-size * --bin-size, i.e. 100 at the default binning '
                         '-- the gate scales with the bin so a bin, not a pixel, has to '
                         'clear the same photon count. Pass a negative value to keep every '
                         'pixel of the mask.')

    ap.add_argument('--no-tail-only', action='store_true',
                    help='Start the phasor window at time bin 0 instead of at the decay '
                         'maximum plus --peak-offset. The default (tail-only) is what the '
                         'prepared crops in this project use; the two are not comparable, '
                         'because a full-decay phasor carries the instrument response.')
    ap.add_argument('--calculate-lifetime', action='store_true',
                    help='Also fit a mono-exponential per pixel. It changes NOTHING in the '
                         'output: the caller discards the lifetime and the chi-square, and '
                         'only G and S are written. It is a per-pixel curve_fit, so it '
                         'costs a great deal of time for nothing. Left exposed because the '
                         'switch exists.')
    ap.add_argument('--keep-existing', action='store_true',
                    help='Do not delete the files already in the output folder before '
                         'writing. The default is to clear it, which is what makes a '
                         're-run idempotent. Read this before turning it off: the clearing '
                         'happens once per FIELD OF VIEW, inside the loop, while the '
                         'written names (cell<id>_5D.tif) carry no field-of-view part, so '
                         'a sample with more than one field of view keeps only the LAST '
                         'one either way. --keep-existing does not fix that; it makes '
                         'same-numbered cells from different fields of view overwrite each '
                         'other instead. This is long-standing behaviour and is left '
                         'alone deliberately.')

    args = ap.parse_args()

    if args.bin_size < 1:
        raise SystemExit('--bin-size must be at least 1 (1 disables binning), got %d.'
                         % args.bin_size)
    if args.peak_offset < 0 or args.end_offset < 0:
        raise SystemExit('--peak-offset and --end-offset are counts of time bins and '
                         'cannot be negative, got %d and %d.'
                         % (args.peak_offset, args.end_offset))
    if args.tau_resolution <= 0:
        raise SystemExit('--tau-resolution is nanoseconds per time bin and must be '
                         'positive, got %s.' % args.tau_resolution)
    if args.rep_rate_mhz <= 0:
        raise SystemExit('--rep-rate-mhz must be positive, got %s.' % args.rep_rate_mhz)

    f1, f2, f3, f4 = unpack_calibration_factors(args.calibration_factors)
    intensity_threshold = args.intensity_threshold or 100 * args.bin_size * args.bin_size
    tail_only = not args.no_tail_only
    smooth_option = None if args.smooth == 'none' else args.smooth
    cell_types = resolve_samples(args.data_root, args.samples, args.all_samples)

    # Printed BEFORE the main loop, which is where the deleting happens, and printed as one
    # line per folder rather than as a summary: --all-samples resolves its list from a scan
    # of the disk, so this is the only chance to see a mistake while it is still reversible.
    print('data root: %s' % args.data_root)
    print('samples: %d folder(s), from %s'
          % (len(cell_types), '--samples' if args.samples
             else '--all-samples (every folder under --data-root with a raw/)'))
    for name in cell_types:
        print('    %s' % os.path.join(args.data_root, name, args.seg_folder))
    print('writing: <sample>/%s/cell<id>_5D.tif   (%s)'
          % (args.seg_folder,
             'keeping existing files' if args.keep_existing else 'clearing it first'))
    if not args.keep_existing:
        print('NOTE: every folder listed above is cleared -- every file in it removed -- '
              'before its crops are written. Nothing has been deleted yet.')
    print('calibration factors: f1=%s f2=%s f3=%s f4=%s' % (f1, f2, f3, f4))
    print('phasor: phi_calib=%s   m_calib=%s   rep rate=%s MHz   tau resolution=%s ns'
          % (args.phi_calib, args.m_calib, args.rep_rate_mhz, args.tau_resolution))
    print('window: tail_only=%s   peak offset=%d   end offset=%d   smooth=%s   lifetime=%s'
          % (tail_only, args.peak_offset, args.end_offset, smooth_option,
             args.calculate_lifetime))
    print('binning: bin size=%d   intensity threshold=%d%s'
          % (args.bin_size, intensity_threshold,
             '  (auto)' if not args.intensity_threshold else ''))

    # Every flag, resolved, written once per sample folder, next to the crops it produced.
    # This script writes in place and has no --out, so the record goes beside its output;
    # without it a folder of crops does not say which calibration made it.
    cfg = {k: v for k, v in sorted(vars(args).items())}
    cfg['resolved_calibration_factors'] = '%s,%s,%s,%s' % (f1, f2, f3, f4)
    cfg['resolved_intensity_threshold'] = intensity_threshold
    cfg['resolved_tail_only'] = tail_only
    cfg['resolved_smooth_option'] = smooth_option
    cfg['resolved_samples'] = ','.join(cell_types)
    run_config = pd.DataFrame([{'flag': k, 'value': v} for k, v in cfg.items()])

    # -------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------
    for cell_type in cell_types:
        raw_dir = os.path.join(args.data_root, cell_type, 'raw')
        if not os.path.isdir(raw_dir):
            raise SystemExit(
                '--data-root %s: sample %s has no raw/ subfolder (looked in %s).\n'
                'Expected layout: <root>/<sample>/raw/<fov>, with flim_stack/ and '
                'intensity/ beside it.' % (args.data_root, cell_type, raw_dir))
        fnames = os.listdir(raw_dir)
        run_config.to_csv(
            os.path.join(args.data_root, cell_type, 'data_prep_run_config.csv'), index=False)
        print('  [%s] raw/: %d field(s) of view  ->  %s/' % (cell_type, len(fnames),
                                                             args.seg_folder))
        for fname in fnames:
            fov = os.path.splitext(fname)[0]
            print(f"Processing {cell_type} / {fov} ...")

            # load segmentation
            seg_path = os.path.join(args.data_root, cell_type, 'intensity', f'{fov}-sum_seg.npy')
            masks = np.load(seg_path, allow_pickle=True).item()['masks']

            # load FLIM & intensity frames
            stack_sum = tiff.imread(os.path.join(args.data_root, cell_type, 'flim_stack',  f'{fov}-sum.tif'))
            I1 = cv2.imread(os.path.join(args.data_root, cell_type, 'intensity', f'{fov}-1.tif'), -1)
            I2 = cv2.imread(os.path.join(args.data_root, cell_type, 'intensity', f'{fov}-2.tif'), -1)
            I3 = cv2.imread(os.path.join(args.data_root, cell_type, 'intensity', f'{fov}-3.tif'), -1)
            I4 = cv2.imread(os.path.join(args.data_root, cell_type, 'intensity', f'{fov}-4.tif'), -1)

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
            b_i_sum      = binning_2d(Csum, args.bin_size)
            b_decay_data = binning_3d(decay_data, args.bin_size)

            out_dir = os.path.join(args.data_root, cell_type, args.seg_folder)
            # if exist, clear it
            if os.path.exists(out_dir) and not args.keep_existing:
                for file in os.listdir(out_dir):
                    file_path = os.path.join(out_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"Cleared existing files in {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            for cid in tqdm(range(1, masks.max() + 1)):
                # 1) extract this cell's binary mask
                mask = (masks == cid)
                if not mask.any():
                    continue

                # 2) bin that mask to match b_i_sum dimensions
                mask_b = binning_2d(mask.astype(np.uint8), args.bin_size).astype(bool)

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
                    g, s, _, _, _, _ = calcu_phasor_info(
                        roi, tot, pidx,
                        tail_only, args.peak_offset, args.end_offset, smooth_option,
                        args.calculate_lifetime, args.tau_resolution, args.rep_rate_mhz)
                    c_g[i, j] = g
                    c_s[i, j] = s

                # 5) apply phasor-space calibration
                phi = np.arctan2(c_s, c_g)
                m = np.sqrt(c_g ** 2 + c_s ** 2)
                phi_c = phi + args.phi_calib
                m_c = m * args.m_calib
                c_gc = m_c * np.cos(phi_c)
                c_sc = m_c * np.sin(phi_c)

                # 6) binned intensity ratios (already computed)
                ci1 = binning_2d(int_ratio_1, args.bin_size) * mask_b
                ci2 = binning_2d(int_ratio_2, args.bin_size) * mask_b
                ci3 = binning_2d(int_ratio_3, args.bin_size) * mask_b
                isum = b_i_sum * mask_b  # already binned Csum

                # 7) stack your channels: [g_cal, s_cal, int1, int2, int3, isum]
                cell_stack = np.stack([c_gc, c_sc, ci1, ci2, ci3, isum], axis=0)

                # 8) crop to ROI bounds and write
                # get the crop bounds
                y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
                crop = cell_stack[:, y0:y1, x0:x1]

                tiff.imwrite(os.path.join(out_dir, f'cell{cid}_5D.tif'), crop)

            print(f" -> Saved calibrated 5D cells for {fov}")


if __name__ == '__main__':
    main()
