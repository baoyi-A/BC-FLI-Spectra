import os

import cv2
import numpy as np
import numpy.fft as fft
from matplotlib import pyplot as plt
from readPTU_FLIM import PTUreader
import pandas as pd
from scipy.optimize import curve_fit
import tifffile as tiff

def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau ) + c

def calcu_phasor_info(roi_decay_data,  peak_idx, tau_resolution=0.1, pulse_freq=80, harmonics=1, PEAK_OFFSET=0, END_OFFSET=0):
    # That's the wierd peak for Leica FALCON, not tunable parameter!
    peak2_begin = 77
    peak2_end = 84

    # Create a mask to select desired part of decay curve
    mask_start = peak_idx + PEAK_OFFSET
    mask_end = len(roi_decay_data) - END_OFFSET

    decay_segment_mask = np.zeros_like(roi_decay_data, dtype=bool)
    decay_segment_mask[mask_start:mask_end] = True


    # Use the mask to get the segment of the decay curve
    roi_decay_data_segment = roi_decay_data[decay_segment_mask]
    # Normalize the decay segment with the peak value
    roi_decay_data_normalized = roi_decay_data_segment / np.max(roi_decay_data_segment)

    # Fit the decay curve to an exponential function
    t_arr = np.arange(len(roi_decay_data_normalized)) * tau_resolution

    # print(f'length: {len(roi_decay_data_normalized)}')
    roi_decay_data_normalized = np.delete(roi_decay_data_normalized,
                                          np.arange(peak2_begin - mask_start, peak2_end - mask_start))
    t_arr = np.delete(t_arr, np.arange(peak2_begin - mask_start, peak2_end - mask_start))
    # print(f'length after: {len(roi_decay_data_normalized)}')


    # fastflim calculation
    fastflim = np.sum(roi_decay_data_normalized * t_arr) / np.sum(roi_decay_data_normalized)
    # fastflim = ptu_file.get_lifetime_imge()
    # print(f't_arr: {t_arr}')

    params_init = [1, 2, 0]
    popt, pcov = curve_fit(exp_func, t_arr, roi_decay_data_normalized, p0=params_init)
    # get the lifetime and 卡方值
    lifetime = popt[1]
    chi_square = np.sum((roi_decay_data_normalized - exp_func(t_arr, *popt)) ** 2)

    # Computing g and s coordinates freq: 80MHz, tau_res: 10^9 times, so 1/1000
    phasor_g = np.sum(roi_decay_data_normalized * np.cos(2 * np.pi * pulse_freq * harmonics * t_arr)) / np.sum(
        roi_decay_data_normalized)
    phasor_s = np.sum(roi_decay_data_normalized * np.sin(2 * np.pi * pulse_freq * harmonics * t_arr)) / np.sum(
        roi_decay_data_normalized)

    print(f'g: {phasor_g}, s: {phasor_s}')
    return phasor_g, phasor_s, lifetime, chi_square, fastflim

# Parameter input: mask intensity threshold, pixel intensity threshold, peak offset, end offset, tau resolution, pulse frequency, harmonics, pixel_wise

def Gen_excel(stack1, stack2, stack3, stack4, output_folder, seg_img, mask_int_thres, pixel_int_thres,
              peak_offset, end_offset, tau_resolution, pulse_freq, harmonics, pixel_wise):
    # stack1-4 are tiff stacks, no need to read them
    if pixel_wise:
        save_path = os.path.join(output_folder, 'FLIM-S_pixel.xlsx')
        print('Pixel-wise mode is not supported currently.')
    else:
        save_path = os.path.join(output_folder, 'FLIM-S.xlsx')

    phasor_g_values = []
    phasor_s_values = []
    lifetime_values = []
    chi_square_values = []
    total_intensity_values = []
    mask_labels = []
    fastflim_values = []
    int_570_590_values = []
    int_590_610_values = []
    int_610_638_values = []
    int_638_720_values = []
    norm_1_4_1_values = []  # 1-4 channels normalization
    norm_1_4_2_values = []
    norm_1_4_3_values = []
    norm_1_4_4_values = []

    labels = np.unique(seg_img)
    labels = labels[labels != 0]  # Exclude the background label

    # sum the stack along the time axis to get the intensity image
    intensity_1 = np.sum(stack1, axis=0)
    intensity_2 = np.sum(stack2, axis=0)
    intensity_3 = np.sum(stack3, axis=0)
    intensity_4 = np.sum(stack4, axis=0)
    intensity_image = intensity_1 + intensity_2 + intensity_3 + intensity_4

    # decay data is the sum of the stacks
    decay_data = stack1 + stack2 + stack3 + stack4

    for label in labels:
        cell_mask = seg_img == label
        valid_pixel_mask = (intensity_image >= pixel_int_thres) & cell_mask
        mask_intensity = np.sum(intensity_image[cell_mask])
        if mask_intensity < mask_int_thres:
            print(f"Mask with label {label} excluded due to low total intensity.")
            continue
        roi_decay_data = np.sum(decay_data[:, valid_pixel_mask], axis=-1)
        peak_idx = np.argmax(roi_decay_data)

        # here, only consider pixel_wise = false case, since pixel_wise = true is not supported currently
        total_intensity = np.sum(intensity_image[cell_mask])
        phasor_g, phasor_s, lifetime, chi_square, fastflim = \
            calcu_phasor_info(roi_decay_data, peak_idx=peak_idx, PEAK_OFFSET=peak_offset, END_OFFSET=end_offset,
                              tau_resolution=tau_resolution, pulse_freq=pulse_freq, harmonics=harmonics)
        int_570_590 = np.sum(intensity_1[cell_mask])
        int_590_610 = np.sum(intensity_2[cell_mask])
        int_610_638 = np.sum(intensity_3[cell_mask])
        int_638_720 = np.sum(intensity_4[cell_mask])
        norm_1_4_1 = int_570_590 / total_intensity
        norm_1_4_2 = int_590_610 / total_intensity
        norm_1_4_3 = int_610_638 / total_intensity
        norm_1_4_4 = int_638_720 / total_intensity

        phasor_g_values.append(phasor_g)
        phasor_s_values.append(phasor_s)
        lifetime_values.append(lifetime)
        chi_square_values.append(chi_square)
        total_intensity_values.append(total_intensity)
        mask_labels.append(label)
        fastflim_values.append(fastflim)
        int_570_590_values.append(int_570_590)
        int_590_610_values.append(int_590_610)
        int_610_638_values.append(int_610_638)
        int_638_720_values.append(int_638_720)
        norm_1_4_1_values.append(norm_1_4_1)
        norm_1_4_2_values.append(norm_1_4_2)
        norm_1_4_3_values.append(norm_1_4_3)
        norm_1_4_4_values.append(norm_1_4_4)

    phasor_g_values = np.array(phasor_g_values)
    phasor_s_values = np.array(phasor_s_values)
    lifetime_values = np.array(lifetime_values)
    chi_square_values = np.array(chi_square_values)
    total_intensity_values = np.array(total_intensity_values)
    mask_labels = np.array(mask_labels)
    fastflim_values = np.array(fastflim_values)
    int_570_590_values = np.array(int_570_590_values)
    int_590_610_values = np.array(int_590_610_values)
    int_610_638_values = np.array(int_610_638_values)
    int_638_720_values = np.array(int_638_720_values)
    norm_1_4_1_values = np.array(norm_1_4_1_values)
    norm_1_4_2_values = np.array(norm_1_4_2_values)
    norm_1_4_3_values = np.array(norm_1_4_3_values)
    norm_1_4_4_values = np.array(norm_1_4_4_values)

    data_df = pd.DataFrame({
        'G': phasor_g_values,
        'S': phasor_s_values,
        'Lifetime': lifetime_values,
        'Chi^2': chi_square_values,
        'Total intensity': total_intensity_values,
        'Mask label': mask_labels,
        'FastFLIM': fastflim_values,
        'Int 570-590': int_570_590_values,
        'Int 590-610': int_590_610_values,
        'Int 610-638': int_610_638_values,
        'Int 638-720': int_638_720_values,
        'Int 1/(1-4)': norm_1_4_1_values,
        'Int 2/(1-4)': norm_1_4_2_values,
        'Int 3/(1-4)': norm_1_4_3_values,
        'Int 4/(1-4)': norm_1_4_4_values,
    })

    if os.path.exists(save_path):
        print('Excel file already exists, will overwrite the data.')

    data_df.to_excel(save_path, index=False)
    print(f'Excel file saved at {save_path}')









