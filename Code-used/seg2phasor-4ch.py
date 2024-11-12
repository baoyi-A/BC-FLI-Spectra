import os

import cv2
import numpy as np
import numpy.fft as fft
import pywt
from matplotlib import pyplot as plt
from scipy.signal import medfilt

from readPTU_FLIM import PTUreader
import pandas as pd
from scipy.optimize import curve_fit
import tifffile as tiff
def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau ) + c

# cell_types = ['NLS-mScarlet-H-231205-1',
#               'NLS-mScarlet-H-231205-2','NLS-mScarlet-H-231205-3','NLS-FR-231205-1']
# type_fp_locs = ['mScarlet-H_Nu','mScarlet-H_Nu','mScarlet-H_Nu','FR_Nu']
# cell_types = ['NLS-FR-MQV-231205']
# type_fp_locs = ['FR-MQV_Nu']
# cell_types = ['NLS-FR-231205-2','NLS-FR-231205-3',]
# type_fp_locs = ['FR_Nu','FR_Nu']
# cell_types = ['MS-N1-4-8-9-10-11-13_231127'] # mix
# type_fp_locs = ['1-4-8-9-10-11-13_Nu-MS']
# cell_types = ['NLS-FR-M-231211-1']
# type_fp_locs = ['FR-M_Nu']
# cell_types = ['NLS-mScarlet-H-231219']
# type_fp_locs = ['mScarlet-H_Nu']
# cell_types = ['NLS-mScarlet-I3-231211']
# type_fp_locs = ['mScarlet-I3_Nu']
# cell_types = ['BS-N1-4-8-9-10-15-231226']
# type_fp_locs = ['1-4-8-9-10-15_Nu-BS']
# cell_types = ['NLS-FR-M-231225','NLS-mCherry-231225',
#               'NLS-mScarlet-I3-231225','NLS-mScarlet-H-231225','NLS-mScarlet3-231225']
# type_fp_locs = ['FR-M_Nu','mCherry_Nu','mScarlet-I3_Nu','mScarlet-H_Nu','mScarlet3_Nu']
# cell_types = ['NLS-mRuby3-231225']
# type_fp_locs = ['mRuby3_Nu']
# cell_types = ['NLS-mRuby3-231211-1','NLS-mRuby3-231211-2','NLS-mScarlet3-231211-1','NLS-mScarlet3-231211-2']
# type_fp_locs = ['mRuby3_Nu','mRuby3_Nu','mScarlet3_Nu','mScarlet3_Nu']
# cell_types = ['NLS-FR-M-231211-2',]
# type_fp_locs = ['FR-M_Nu',]
# cell_types = ['NLS-mScarlet3-231211-2']
# type_fp_locs = ['mScarlet3_Nu']
# cell_types = ['NLS-N12-13-14-15-16-240114-DCZ']
# type_fp_locs = ['N12-13-14-15-16_Nu']
# cell_types = ['NLS-N12-13-14-15-16-240114-DMSO']
# type_fp_locs = ['N12-13-14-15-16_Nu']
# cell_types = ['NLS-N12-13-14-15-16-240114-ISO']
# type_fp_locs = ['N12-13-14-15-16_Nu']
# cell_types = ['NLS-mCherry-231211-1']
# type_fp_locs = ['mCherry_Nu']
# cell_types = ['NLS-mApple-240117','NLS-FR-1-240117','NLS-FR-MQ-240117','NLS-mScarlet-I3-240117','NLS-TagRFP-T-240117']
# type_fp_locs = ['mApple_Nu','FR-1_Nu','FR-MQ_Nu','mScarlet-I3_Nu','TagRFP-T_Nu']
# cell_types = ['NLS-N13-14-16-240117-1']
# type_fp_locs = ['N13-14-16_Nu']
# cell_types = ['NLS-N13-14-16-240117-2']
# type_fp_locs = ['N13-14-16_Nu-2']
# cell_types = ['NLS-N1-4-8-9-10-11-13-231211']
# type_fp_locs = ['N1-4-8-9-10-11-13_Nu']
# cell_types = ['NLS-N3-8-10-13-14-240119-FSK', 'NLS-N3-8-10-13-14-240119-ISO', 'NLS-N3-8-10-13-14-240119-DMSO', 'NLS-N3-8-10-13-14-240119-NE']
# type_fp_locs = ['N3-8-10-13-14_Nu-FSK', 'N3-8-10-13-14_Nu-ISO', 'N3-8-10-13-14_Nu-DMSO', 'N3-8-10-13-14_Nu-NE']
# cell_types = ['NLS-N4-8-9-10-13-14-16-240126-act']
# type_fp_locs = ['N4-8-9-10-13-14-16_Nu-act']
# cell_types = ['NLS-mScarlet-I-240126']
# type_fp_locs = ['mScarlet-I_Nu']
# cell_types = ['NLS-mApple-240126']
# type_fp_locs = ['mApple_Nu']
# cell_types = ['NLS-FR-MQ-240117']
# type_fp_locs = ['FR-MQ_Nu']
# cell_types = ['NLS-FR-1-240117']
# type_fp_locs = ['FR-1_Nu']
# cell_types = ['NLS-FR-1-240126']
# type_fp_locs = ['FR-1_Nu']
# cell_types = ['NLS-FR-M-231225']
# type_fp_locs = ['FR-M_Nu']
# cell_types = ['NLS-N4-8-9-10-13-14-16-240126-vcl']
# type_fp_locs = ['N4-8-9-10-13-14-16_Nu-vcl']
# cell_types = ['NLS-N4-8-9-10-13-14-16-240126-act-stained']
# type_fp_locs = ['N4-8-9-10-13-14-16_Nu-act-stained']
# cell_types = ['NLS-N4-8-9-10-13-14-16-240126-vcl-fixed']
# type_fp_locs = ['N4-8-9-10-13-14-16_Nu-vcl-fixed']
# cell_types = ['NLS-N4-8-9-10-13-14-16-240126-vcl-stained']
# type_fp_locs = ['N4-8-9-10-13-14-16_Nu-vcl-stained']
# cell_types = ['NLS-N3-8-10-12-13-14-15-240128-NE']
# type_fp_locs = ['N3-8-10-12-13-14-15_Nu-NE']
# cell_types = ['NLS-N3-8-10-12-13-14-15-240128-NE-2']
# type_fp_locs = ['N3-8-10-12-13-14-15_Nu-NE-2']
# cell_types = ['NLS-mScarlet-I3-240126']
# type_fp_locs = ['mScarlet-I3_Nu']
# cell_types = ['NLS-FR-MQ-240126']
# type_fp_locs = ['FR-MQ_Nu']
# cell_types = ['NLS-mApple-240117']
# type_fp_locs = ['mApple_Nu']
# cell_types = ['NLS-mCherry-231225']
# type_fp_locs = ['mCherry_Nu']
# cell_types = ['NLS-mRuby3-231225', 'NLS-mScarlet3-231225', 'NLS-mScarlet-H-231225']
# type_fp_locs = ['mRuby3_Nu', 'mScarlet3_Nu', 'mScarlet-H_Nu']
# cell_types = ['NLS-mScarlet-I3-231225']
# type_fp_locs = ['mScarlet-I3_Nu']
# cell_types = ['NLS-mScarlet-I3-240117', 'NLS-mScarlet-I3-240126']
# type_fp_locs = ['mScarlet-I3_Nu', 'mScarlet-I3_Nu']
# cell_types = ['NLS-mScarlet-I-240126', 'NLS-TagRFP-T-240117']
# type_fp_locs = ['mScarlet-I_Nu', 'TagRFP-T_Nu']
# cell_types = ['NLS-FR-1-240126']
# type_fp_locs = ['FR-1_Nu']
# cell_types = ['NLS-mScarlet-I-240123', 'NLS-mScarlet-H-240123']
# type_fp_locs = ['mScarlet-I_Nu', 'mScarlet-H_Nu']
# cell_types = ['NLS-mCherry-240123', 'NLS-mRuby3-240123', 'NLS-mScarlet3-240123', 'NLS-FR-M-240123']
# type_fp_locs = ['mCherry_Nu', 'mRuby3_Nu', 'mScarlet3_Nu', 'FR-M_Nu']
# cell_types = ['NLS-mScarlet3-240123']
# type_fp_locs = ['mScarlet3_Nu']
# cell_types = ['NLS-N2-3-5-8-9-10-11-12-13-14-15-240206-DMSO']
# type_fp_locs = ['N2-3-5-8-9-10-11-12-13-14-15_Nu-DMSO']
# cell_types = ['NLS-N2-3-5-8-9-10-11-12-13-14-15-240206-ISO-1']
# type_fp_locs = ['N2-3-5-8-9-10-11-12-13-14-15_Nu-ISO-1']
# cell_types = ['NLS-N2-3-5-8-9-10-11-12-13-14-15-240206-ISO-2']
# type_fp_locs = ['N2-3-5-8-9-10-11-12-13-14-15_Nu-ISO-2']
# cell_types = ['NLS-N2-3-5-8-9-10-11-12-13-14-15-240206-NE-1']
# type_fp_locs = ['N2-3-5-8-9-10-11-12-13-14-15_Nu-NE-1']
# cell_types = ['NLS-N2-3-5-8-9-10-11-12-13-14-15-240206-NE-2']
# type_fp_locs = ['N2-3-5-8-9-10-11-12-13-14-15_Nu-NE-2']
# cell_types = ['NLS-mScarlet-H-240222-1', 'NLS-mScarlet-H-240222-2',
#               'NLS-FR-240222-1', 'NLS-FR-240222-2',
#               'NLS-FR-MQV-240222-1', 'NLS-FR-MQV-240222-2',
#               'NLS-mCherry-240222-1', 'NLS-mCherry-240222-2',
#               'NLS-mScarlet3-240222-1', 'NLS-mScarlet3-240222-2',
#               'NLS-TagRFP-T-240222-1', 'NLS-TagRFP-T-240222-2',
#               'NLS-mScarlet-I3-240222-1', 'NLS-mScarlet-I3-240222-2',
#               'NLS-FR-1-240222-1', 'NLS-FR-1-240222-2',]
# type_fp_locs = ['mScarlet-H_Nu', 'mScarlet-H_Nu',
#                 'FR_Nu', 'FR_Nu',
#                 'FR-MQV_Nu', 'FR-MQV_Nu',
#                 'mCherry_Nu', 'mCherry_Nu',
#                 'mScarlet3_Nu', 'mScarlet3_Nu',
#                 'TagRFP-T_Nu', 'TagRFP-T_Nu',
#                 'mScarlet-I3_Nu', 'mScarlet-I3_Nu',
#                 'FR-1_Nu', 'FR-1_Nu',]
# cell_types = ['NLS-mScarlet-I-240229-1', 'NLS-mScarlet-I-240229-2',
#               'NLS-FR-M-240229-1', 'NLS-FR-M-240229-2',
#               'NLS-mApple-240229-1', 'NLS-mApple-240229-2',]
# type_fp_locs = ['mScarlet-I_Nu', 'mScarlet-I_Nu',
#                 'FR-M_Nu', 'FR-M_Nu',
#                 'mApple_Nu', 'mApple_Nu',]
# cell_types = ['NLS-N1-4-9-11-13-16-240223-DCZ', 'NLS-N1-4-9-11-13-16-240223-DMSO', 'NLS-N1-4-9-11-13-16-240223-ISO']
# type_fp_locs = ['N1-4-9-11-13-16_Nu-DCZ', 'N1-4-9-11-13-16_Nu-DMSO', 'N1-4-9-11-13-16_Nu-ISO']
# cell_types = ['NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240308-DMSO']
# type_fp_locs = ['N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-DMSO']
# cell_types = ['NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240308-FSK']
# type_fp_locs = ['N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-FSK']
# cell_types = ['NLS-mKate2-240308-1', 'NLS-mKate2-240308-2', 'NLS-mRuby3-240308-1', 'NLS-mRuby3-240308-2', 'NLS-FR-MQ-240308-1', 'NLS-FR-MQ-240308-2', ]
# type_fp_locs = ['mKate2_Nu', 'mKate2_Nu', 'mRuby3_Nu', 'mRuby3_Nu', 'FR-MQ_Nu', 'FR-MQ_Nu', ]
# cell_types = ['NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240308']
# type_fp_locs = ['N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu']
# cell_types = ['NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240315']
# type_fp_locs = ['N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu']
# cell_types = ['NLS-N1-4-7-8-9-10-11-12-13-14-15-16-240326-1',]
# type_fp_locs = ['N1-4-7-8-9-10-11-12-13-14-15-16_Nu-1',]
# cell_types = ['NLS-N1-4-7-8-9-10-11-12-13-14-15-16-240326-2',]
# type_fp_locs = ['N1-4-7-8-9-10-11-12-13-14-15-16_Nu-2',]
# cell_types = ['NLS-N1-4-7-8-9-10-11-12-13-14-15-16-240326-1-fixed','NLS-N1-4-7-8-9-10-11-12-13-14-15-16-240326-2-fixed']
# type_fp_locs = ['N1-4-7-8-9-10-11-12-13-14-15-16_Nu-1-fixed','N1-4-7-8-9-10-11-12-13-14-15-16_Nu-2-fixed']
# cell_types = [
    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-FSK-1',
#               'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-FSK-2',
#               'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE',]
# type_fp_locs = [
    # 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-FSK-1',
#                 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-FSK-2',
#                 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-NE',]
# cell_types = ['NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-NE-1',
#                 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-NE-2',
#                 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-NE-3',
#               'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-DMSO',]
# type_fp_locs = ['N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-NE-1',
#                 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-NE-2',
#                 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-NE-3',
#                 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-DMSO',]
# cell_types = ['NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-1',
#                 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-2',]
# type_fp_locs = ['N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-1',
#                 'N1-2-3-4-7-8-9-10-11-12-13-14-15-16_Nu-2',]
# cell_types = [
#                 'NLS-mScarlet-H-240401', 'NLS-FR-240401', 'NLS-FR-MQV-240401',
#     'NLS-mCherry-240401','NLS-mRuby3-240401', 'NLS-mScarlet3-240401',
#     'NLS-mApple-240401', 'NLS-mKate2-240401', 'NLS-mScarlet-I-240401',
#     'NLS-FR-M-240401'
# ]
# type_fp_locs = [
#     'mScarlet-H_Nu', 'FR_Nu', 'FR-MQV_Nu',
#     'mCherry_Nu','mRuby3_Nu', 'mScarlet3_Nu',
#     'mApple_Nu', 'mKate2_Nu', 'mScarlet-I_Nu',
#     'FR-M_Nu'
# ]
# cell_types = [ 'NLS-mScarlet-H-240406',]
# type_fp_locs = ['mScarlet-H_Nu',]

# cell_types = [
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-DMSO',
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-ISO-1',
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-ISO-2',
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-ISO-3',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-1',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-2',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-3',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-4',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-5',
# ]
# type_fp_locs = [
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-DMSO',
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-ISO-1',
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-ISO-2',
#                     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-ISO-3',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-1',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-2',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-3',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-4',
                    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240410-Nu-5',
# ]
# cell_types = [
#     'NLS-N1-3-4-7-8-9-10-11-13-14-15-16-240425-2',
#     'NLS-N1-3-4-7-8-9-10-11-13-14-15-16-240425-3',
#     ]
# type_fp_locs = [
#     'N1-3-4-7-8-9-10-11-13-14-15-16_Nu-2',
#     'N1-3-4-7-8-9-10-11-13-14-15-16_Nu-3',
#     ]
# cell_types = [
#     'NLS-N1-3-4-7-8-9-10-11-13-14-15-16-240425-1'
#     ]
# type_fp_locs = [
#     'N1-3-4-7-8-9-10-11-13-14-15-16_Nu-1'
#     ]
# cell_types = [
#     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-240429-NE-1',
#     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-240429-NE-9',
#     ]
# type_fp_locs = [
#     'N1-2-3-4-7-8-9-10-11-12-13-14-15_Nu-NE-1',
#     'N1-2-3-4-7-8-9-10-11-12-13-14-15_Nu-NE-9',
#     ]
cell_types = [
    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240510-1',
    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240510-1notraw',
    # 'NLS-14Mix-NLYN-5Mix-240618-1'
    # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240308-FSK'
# 'NTOM-N10-240528',
#       'NTOM-N13-240528',
# 'NTOM-N4-240528',
# 'NTOM-N14-240528',
#                  'NTOM-N16-240528',
# 'NTOM-N8-240528',
# 'NTOM-N1-240528',
#     'NLS-TagRFP-T-240117'
#     'NTOM-M2-240814',
#     'NTOM-M9-240814-1',
#     'NTOM-M9-240814-2',
#     'NTOM-M12-240814-1',
#     'NTOM-M12-240814-2',
#     'NTOM-M15-240814',
# 'NLYN-P1-240629'
    'NLYN-P13-240629'
]
continue_save = False
# dataset = 'test_data'
cell = 'Hek293T'
# cell = 'MDA'
# cell = 'MCF7'
# cell = 'MEF'
instrument = 'BJMU'
dataset = f'{cell}-{instrument}'
# dataset = 'Mix_test'
# dataset = 'lifetime_MCF7'
# fov = 'fov3'
processed_fovs = []
data_dir = fr"E:\BC-FLIM\{dataset}"
pixel_wise = False
tail_only = True
remove_peak2 = False
mask_int_thres = 1500 # 20000, 12000, 5000, 3000( for flim is ok?)
pixel_int_thres = 7 # 30, 10
PEAK_OFFSET = 4 # 2 originally
END_OFFSET = 18 # 13 originally
peak2_begin = 77
peak2_end = 84
tau_resolution = 0.09696969696999999 # ns
# frame = '_1'
# frame = '_0_5'
# frame = '_0_10'
# frame = '_0'
frame = ''
def calcu_phasor_info(roi_decay_data, roi_decay_data_1_2_3, total_intensity, type, peak_idx, peak_idx_1_2_3):

    # plot the decay curve
    # plt.figure()
    # plt.plot(roi_decay_data)
    # plt.show()
    # print(f'total intensity: {total_intensity}')
    # Find the index of the peak

    # Create a mask to select desired part of decay curve
    if tail_only:
        mask_start = peak_idx + PEAK_OFFSET
    else:
        mask_start = 0 # start from the beginning
    mask_end = len(roi_decay_data) - END_OFFSET
    mask_start_1_2_3 = peak_idx_1_2_3 + PEAK_OFFSET
    mask_end_1_2_3 = len(roi_decay_data_1_2_3) - END_OFFSET
    decay_segment_mask = np.zeros_like(roi_decay_data, dtype=bool)
    decay_segment_mask[mask_start:mask_end] = True
    decay_segment_mask_1_2_3 = np.zeros_like(roi_decay_data_1_2_3, dtype=bool)
    decay_segment_mask_1_2_3[mask_start_1_2_3:mask_end_1_2_3] = True

    # Use the mask to get the segment of the decay curve
    roi_decay_data_segment = roi_decay_data[decay_segment_mask]
    roi_decay_data_segment_1_2_3 = roi_decay_data_1_2_3[decay_segment_mask_1_2_3]
    # Normalize the decay segment with the peak value
    roi_decay_data_normalized = roi_decay_data_segment / np.max(roi_decay_data_segment)
    roi_decay_data_normalized_1_2_3 = roi_decay_data_segment_1_2_3 / np.max(roi_decay_data_segment_1_2_3)
    # append the raw decay data
    # raw_decay_data.append(roi_decay_data_normalized)
    # Fit the decay curve to an exponential function
    t_arr = np.arange(len(roi_decay_data_normalized)) * tau_resolution
    t_arr_1_2_3 = np.arange(len(roi_decay_data_normalized_1_2_3)) * tau_resolution
    # plot the decay curves in the same figure
    # plt.figure()
    # plt.plot(t_arr, roi_decay_data_normalized)
    # plt.show()
    # median filter the decay curve
    # kernel_size = 5
    # roi_decay_data_normalized = medfilt(roi_decay_data_normalized, kernel_size=3)
    # plot the decay curve after median filter
    # plt.figure()
    # plt.plot(t_arr, roi_decay_data_normalized)
    # plt.show()
    # apply wavelet denoising
    # roi_decay_data_normalized = pywt.dwt(roi_decay_data_normalized, 'db1') # db1 is Haar wavelet
    # plot the decay curve after wavelet denoising
    # plt.figure()
    # plt.plot(t_arr, roi_decay_data_normalized)
    # plt.show()

    # remove the data from 7.41ns to 8.05 ns, since there's always a peak2 there
    # print(f'length: {len(roi_decay_data_normalized)}')
    if remove_peak2:
        roi_decay_data_normalized = np.delete(roi_decay_data_normalized,
                                              np.arange(peak2_begin - mask_start, peak2_end - mask_start))
        t_arr = np.delete(t_arr, np.arange(peak2_begin - mask_start, peak2_end - mask_start))
    # print(f'length after: {len(roi_decay_data_normalized)}')
    roi_decay_data_normalized_1_2_3 = np.delete(roi_decay_data_normalized_1_2_3,
                                                np.arange(peak2_begin - mask_start_1_2_3, peak2_end - mask_start_1_2_3))
    t_arr_1_2_3 = np.delete(t_arr_1_2_3, np.arange(peak2_begin - mask_start_1_2_3, peak2_end - mask_start_1_2_3))
    # plot the decay curve in another color
    # plt.plot(t_arr, roi_decay_data_normalized)
    # plt.show()

    # fastflim calculation
    fastflim = np.sum(roi_decay_data_normalized * t_arr) / np.sum(roi_decay_data_normalized)
    fastflim_1_2_3 = np.sum(roi_decay_data_normalized_1_2_3 * t_arr_1_2_3) / np.sum(roi_decay_data_normalized_1_2_3)
    # fastflim = ptu_file.get_lifetime_imge()
    # print(f't_arr: {t_arr}')
    if type == 'mask':
        params_init = [1, 2, 0]
        popt, pcov = curve_fit(exp_func, t_arr, roi_decay_data_normalized, p0=params_init)
        popt_1_2_3, pcov_1_2_3 = curve_fit(exp_func, t_arr_1_2_3, roi_decay_data_normalized_1_2_3, p0=params_init)
        # get the lifetime and 卡方值
        lifetime = popt[1]
        lifetime_1_2_3 = popt_1_2_3[1]
        chi_square = np.sum((roi_decay_data_normalized - exp_func(t_arr, *popt)) ** 2)
        chi_square_1_2_3 = np.sum((roi_decay_data_normalized_1_2_3 - exp_func(t_arr_1_2_3, *popt_1_2_3)) ** 2)
    elif type == 'pixel':
        lifetime = 0
        lifetime_1_2_3 = 0
        chi_square = 0
        chi_square_1_2_3 = 0
    # print(f'Lifetime: {lifetime}, Chi square: {chi_square}')

    # print(f'total intensity: {total_intensity}')
    # print(f'mask label: {label}')
    # save the intensity image cropped with mask as png (but dont scale, and maitian gray) in a new folder, naming the mask with the fov and label
    # mask_save_folder = f'D:\PKU_STUDY\SynologyDrive\BaoyiWang\BC-FLIM\phasor_data\masks\\{type_fp_loc}'
    # make the folder if it doesn't exist
    # if not os.path.exists(mask_save_folder):
    #     os.makedirs(mask_save_folder)
    # mask_save = intensity_image * cell_mask
    # plt.figure()
    # plt.imshow(mask_save)
    # plt.show()
    # crop it as small as possible
    # mask_save = mask_save[np.ix_(mask_save.any(1), mask_save.any(0))] # np.ix_ is used to index the array with another array,
    # first find '23' or '24' in the cell_type, then get the date of the image, like 231205


    # plot the decay curve
    # plt.figure()
    # plt.plot(roi_decay_data_normalized)
    # plt.plot(exp_func(t_arr, *popt))
    # plt.show()

    # # Calculate phasor values using Fourier Transform
    # transformed_data = np.fft.fft(roi_decay_data_normalized)
    # # Extract the g and s values from the first harmonic
    # phasor_g = np.real(transformed_data[1])
    # phasor_s = np.imag(transformed_data[1])

    # Modulation frequency and harmonic
    # freq = 80/1000  # in 1000 MHz
    freq = 78.1 / 1000
    # freq = 1/tau_resolution/len(roi_decay_data_normalized) / 0.5 # in 1000 MHz
    harmonic = 1

    # Computing g and s coordinates freq: 80MHz, tau_res: 10^9 times, so 1/1000
    phasor_g = np.sum(roi_decay_data_normalized * np.cos(2 * np.pi * freq * harmonic * t_arr)) / np.sum(
        roi_decay_data_normalized)
    phasor_s = np.sum(roi_decay_data_normalized * np.sin(2 * np.pi * freq * harmonic * t_arr)) / np.sum(
        roi_decay_data_normalized)
    phasor_g_1_2_3 = np.sum(
        roi_decay_data_normalized_1_2_3 * np.cos(2 * np.pi * freq * harmonic * t_arr_1_2_3)) / np.sum(
        roi_decay_data_normalized_1_2_3)
    phasor_s_1_2_3 = np.sum(
        roi_decay_data_normalized_1_2_3 * np.sin(2 * np.pi * freq * harmonic * t_arr_1_2_3)) / np.sum(
        roi_decay_data_normalized_1_2_3)

    print(f'g: {phasor_g}, s: {phasor_s}')
    return phasor_g, phasor_s, lifetime, chi_square, fastflim, phasor_g_1_2_3, phasor_s_1_2_3, lifetime_1_2_3, chi_square_1_2_3, total_intensity, fastflim_1_2_3



# for type_fp_loc, cell_type in zip(type_fp_locs, cell_types):
for cell_type in cell_types:
    for fov in os.listdir(f'{data_dir}\\{cell_type}\\raw'):
        fov = fov.split('.')[0]
        if fov in processed_fovs:
            continue
        print(f"Processing {fov}...")
        # save_path = fr'D:\PKU_STUDY\SynologyDrive\BaoyiWang\BC-FLIM\phasor_data\{cell}\NLS-{instrument}\\{type_fp_loc}.xlsx'
        # if pixel_wise:
        #     save_path = os.path.join(data_dir, f'{cell_type}\\{type_fp_loc}_pixel.xlsx')
        # else:
        #     save_path = os.path.join(data_dir, f'{cell_type}\\{type_fp_loc}.xlsx')
        save_path = os.path.join(data_dir, f'{cell_type}\\barcode_info{frame}.xlsx')
        if not tail_only: # add not_tail to the save_path
            save_path = save_path.replace('.xlsx', '_not_tail.xlsx')
        if pixel_wise:
            save_path = save_path.replace('.xlsx', '_pixel.xlsx')
        if not remove_peak2:
            save_path = save_path.replace('.xlsx', '_peak2.xlsx')

        stack_path = [f'{data_dir}\\{cell_type}\\flim_stack{frame}\\{fov}-1.tif',
                        f'{data_dir}\\{cell_type}\\flim_stack{frame}\\{fov}-2.tif',
                        f'{data_dir}\\{cell_type}\\flim_stack{frame}\\{fov}-3.tif',
                        f'{data_dir}\\{cell_type}\\flim_stack{frame}\\{fov}-4.tif',
                        f'{data_dir}\\{cell_type}\\flim_stack{frame}\\{fov}-sum.tif',]
        intensity_path = [f'{data_dir}\\{cell_type}\\intensity{frame}\\{fov}-1.tif',
                            f'{data_dir}\\{cell_type}\\intensity{frame}\\{fov}-2.tif',
                            f'{data_dir}\\{cell_type}\\intensity{frame}\\{fov}-3.tif',
                            f'{data_dir}\\{cell_type}\\intensity{frame}\\{fov}-4.tif',
                            f'{data_dir}\\{cell_type}\\intensity{frame}\\{fov}-sum.tif',]
        masks_path = f'{data_dir}\\{cell_type}\\intensity\\{fov}-sum_seg.npy'
        # masks_path = f'{data_dir}\\{cell_type}\\intensity\\{fov}-sum-pm_seg.npy'

        if not os.path.exists(masks_path):
            masks_path = f'{data_dir}\\{cell_type}\\intensity\\{fov}-sum_masks.npy'
        if not os.path.exists(masks_path):
            masks_path = f'{data_dir}\\{cell_type}\\intensity\\{fov}-sum_1_2_3_masks.npy'
        if not os.path.exists(masks_path):
            masks_path = f'{data_dir}\\{cell_type}\\intensity\\{fov}-sum_1_2_3_seg.npy'
        flim_stack_1 = tiff.imread(stack_path[0]) # the type of flim_stack is numpy.ndarray
        flim_stack_2 = tiff.imread(stack_path[1])
        flim_stack_3 = tiff.imread(stack_path[2])
        flim_stack_4 = tiff.imread(stack_path[3])
        flim_stack = tiff.imread(stack_path[4])
        # sum the 1-3 channels
        flim_stack_1_2_3 = flim_stack_1 + flim_stack_2 + flim_stack_3
        intensity_1 = cv2.imread(intensity_path[0], -1) # -1 means read the image as it is
        intensity_2 = cv2.imread(intensity_path[1], -1)
        intensity_3 = cv2.imread(intensity_path[2], -1)
        intensity_4 = cv2.imread(intensity_path[3], -1)
        intensity_image = cv2.imread(intensity_path[4], -1)


        decay_data = flim_stack
        decay_data_1_2_3 = flim_stack_1_2_3
        # intensity_image = np.sum(flim_data_stack, axis=-1)
        # print(f"Intensity image shape: {intensity_image.shape}")
        # Load the ROI masks
        masks = np.load(masks_path, allow_pickle=True).item()
        masks = masks['masks']

        # Store the phasor coordinates
        phasor_g_values = []
        phasor_s_values = []
        lifetime_values = []
        phasor_g_values_1_2_3 = []
        phasor_s_values_1_2_3 = []
        lifetime_values_1_2_3 = []
        chi_square_values = []
        total_intensity_values = []
        chi_square_values_1_2_3 = []
        total_intensity_values_1_2_3 = []
        mask_labels = []
        fov_labels = []
        fastflim_values = []
        fastflim_values_1_2_3 = []
        int_570_590_values = []
        int_590_610_values = []
        int_610_638_values = []
        int_638_720_values = []
        norm_1_4_1_values = [] # 1-4 channels normalization
        norm_1_4_2_values = []
        norm_1_4_3_values = []
        norm_1_4_4_values = []

        labels = np.unique(masks)
        labels = labels[labels != 0]  # Exclude the background label

        # raw_decay_data = []
        for label in labels:


            cell_mask = masks == label
            valid_pixel_mask = (intensity_image >= pixel_int_thres) & cell_mask


            mask_intensity = np.sum(intensity_image[cell_mask])
            if mask_intensity < mask_int_thres:
                print(f"Mask with label {label} excluded due to low total intensity.")
                continue

            roi_decay_data = np.sum(decay_data[:, valid_pixel_mask],
                                    axis=-1)  # axis=0 means sum along the first axis,
            roi_decay_data_1_2_3 = np.sum(decay_data_1_2_3[:, valid_pixel_mask], axis=-1)
            peak_idx = np.argmax(roi_decay_data)
            peak_idx_1_2_3 = np.argmax(roi_decay_data_1_2_3)
            if pixel_wise:

                valid_pixels = np.where(valid_pixel_mask)
                for i,j in zip(*valid_pixels): # * is used to unpack the tuple
                    phasor_g = 0
                    phasor_s = 0
                    lifetime = 0
                    phasor_g_1_2_3 = 0
                    phasor_s_1_2_3 = 0
                    lifetime_1_2_3 = 0
                    chi_square = 0

                    chi_square_1_2_3 = 0
                    total_intensity_1_2_3 = 0
                    fastflim = 0
                    fastflim_1_2_3 = 0
                    int_570_590 = 0
                    int_590_610 = 0
                    int_610_638 = 0
                    int_638_720 = 0
                    norm_1_4_1 = 0
                    norm_1_4_2 = 0
                    norm_1_4_3 = 0
                    norm_1_4_4 = 0

                    # if pixel intensity is less than 100, make all the phasor values 0
                    total_intensity = intensity_image[i, j]
                    # print(f'total intensity: {total_intensity}')
                    if total_intensity < 100:
                        pass


                    else:
                        roi_decay_data = decay_data[:, i, j]
                        roi_decay_data_1_2_3 = decay_data_1_2_3[:, i, j]
                        # phasor_g, phasor_s, lifetime, phasor_g_1_2_3, phasor_s_1_2_3, lifetime_1_2_3, chi_square, chi_square_1_2_3, total_intensity_1_2_3, fastflim, fastflim_1_2_3 = calcu_phasor_info(roi_decay_data, roi_decay_data_1_2_3, total_intensity, type='pixel', peak_idx=peak_idx, peak_idx_1_2_3=peak_idx_1_2_3)
                        #     return phasor_g, phasor_s, lifetime, chi_square, fastflim, phasor_g_1_2_3, phasor_s_1_2_3, lifetime_1_2_3, chi_square_1_2_3, total_intensity, fastflim_1_2_3, so the order above is wrong
                        phasor_g, phasor_s, lifetime, chi_square, fastflim, phasor_g_1_2_3, phasor_s_1_2_3, lifetime_1_2_3, chi_square_1_2_3, total_intensity, fastflim_1_2_3 = calcu_phasor_info(roi_decay_data, roi_decay_data_1_2_3, total_intensity, type='pixel', peak_idx=peak_idx, peak_idx_1_2_3=peak_idx_1_2_3)
                        int_570_590 = intensity_1[i,j]
                        int_590_610 = intensity_2[i,j]
                        int_610_638 = intensity_3[i,j]
                        int_638_720 = intensity_4[i,j]
                        norm_1_4_1 = int_570_590 / total_intensity
                        norm_1_4_2 = int_590_610 / total_intensity
                        norm_1_4_3 = int_610_638 / total_intensity
                        norm_1_4_4 = int_638_720 / total_intensity

                    phasor_g_values.append(phasor_g)
                    phasor_s_values.append(phasor_s)
                    lifetime_values.append(lifetime)
                    phasor_g_values_1_2_3.append(phasor_g_1_2_3)
                    phasor_s_values_1_2_3.append(phasor_s_1_2_3)
                    lifetime_values_1_2_3.append(lifetime_1_2_3)
                    chi_square_values.append(chi_square)
                    total_intensity_values.append(total_intensity)
                    chi_square_values_1_2_3.append(chi_square_1_2_3)
                    total_intensity_values_1_2_3.append(total_intensity)
                    mask_labels.append(label)
                    fov_labels.append(fov)
                    fastflim_values.append(fastflim)
                    fastflim_values_1_2_3.append(fastflim_1_2_3)
                    int_570_590_values.append(int_570_590)
                    int_590_610_values.append(int_590_610)
                    int_610_638_values.append(int_610_638)
                    int_638_720_values.append(int_638_720)
                    norm_1_4_1_values.append(norm_1_4_1)
                    norm_1_4_2_values.append(norm_1_4_2)
                    norm_1_4_3_values.append(norm_1_4_3)
                    norm_1_4_4_values.append(norm_1_4_4)



            else:
                total_intensity = np.sum(intensity_image[cell_mask])
                phasor_g, phasor_s, lifetime, chi_square, fastflim, phasor_g_1_2_3, phasor_s_1_2_3, lifetime_1_2_3, chi_square_1_2_3, total_intensity, fastflim_1_2_3 = calcu_phasor_info(roi_decay_data, roi_decay_data_1_2_3, total_intensity, type='mask', peak_idx=peak_idx, peak_idx_1_2_3=peak_idx_1_2_3)
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
                phasor_g_values_1_2_3.append(phasor_g_1_2_3)
                phasor_s_values_1_2_3.append(phasor_s_1_2_3)
                lifetime_values_1_2_3.append(lifetime_1_2_3)
                chi_square_values.append(chi_square)
                total_intensity_values.append(total_intensity)
                chi_square_values_1_2_3.append(chi_square_1_2_3)
                total_intensity_values_1_2_3.append(total_intensity)
                mask_labels.append(label)
                fov_labels.append(fov)
                fastflim_values.append(fastflim)
                fastflim_values_1_2_3.append(fastflim_1_2_3)
                int_570_590_values.append(int_570_590)
                int_590_610_values.append(int_590_610)
                int_610_638_values.append(int_610_638)
                int_638_720_values.append(int_638_720)
                norm_1_4_1_values.append(norm_1_4_1)
                norm_1_4_2_values.append(norm_1_4_2)
                norm_1_4_3_values.append(norm_1_4_3)
                norm_1_4_4_values.append(norm_1_4_4)
        # exclude all the nan values in the g and s values
        phasor_g_values = np.array(phasor_g_values)
        phasor_s_values = np.array(phasor_s_values)
        lifetime_values = np.array(lifetime_values)
        phasor_g_values_1_2_3 = np.array(phasor_g_values_1_2_3)
        phasor_s_values_1_2_3 = np.array(phasor_s_values_1_2_3)
        lifetime_values_1_2_3 = np.array(lifetime_values_1_2_3)
        chi_square_values = np.array(chi_square_values)
        total_intensity_values = np.array(total_intensity_values)
        chi_square_values_1_2_3 = np.array(chi_square_values_1_2_3)
        total_intensity_values_1_2_3 = np.array(total_intensity_values_1_2_3)
        mask_labels = np.array(mask_labels)
        fov_labels = np.array(fov_labels)
        fastflim_values = np.array(fastflim_values)
        fastflim_values_1_2_3 = np.array(fastflim_values_1_2_3)

        if '-23' in cell_type:
            # find index of '23'
            idx = cell_type.find('-23') + 1
            # get the date
            date = cell_type[idx:idx + 6]
        elif '-24' in cell_type:
            idx = cell_type.find('-24') + 1
            date = cell_type[idx:idx + 6]
        # print(f'date: {date}')
        date_values = np.full_like(phasor_g_values, date)

        # valid_g_mask = ~np.isnan(phasor_g_values)
        # valid_s_mask = ~np.isnan(phasor_s_values)


        # print(f'g values: {phasor_g_values}')
        # print(f's values: {phasor_s_values}')

        # 4. Save the g, s values to an Excel
        data_df = pd.DataFrame({
            'G': phasor_g_values,
            'S': phasor_s_values,
            'Lifetime': lifetime_values,
            'G 1-3': phasor_g_values_1_2_3,
            'S 1-3': phasor_s_values_1_2_3,
            'Lifetime 1-3': lifetime_values_1_2_3,
            'Chi^2': chi_square_values,
            'Total intensity': total_intensity_values,
            'Chi^2 1-3': chi_square_values_1_2_3,
            'Total intensity 1-3': total_intensity_values_1_2_3,
            'Mask label': mask_labels,
            'FOV': fov_labels,
            'Date': date_values,
            'FastFLIM': fastflim_values,
            'FastFLIM 1-3': fastflim_values_1_2_3,
            'Int 570-590': int_570_590_values,
            'Int 590-610': int_590_610_values,
            'Int 610-638': int_610_638_values,
            'Int 638-720': int_638_720_values,
            'Int 1/(1-4)': norm_1_4_1_values,
            'Int 2/(1-4)': norm_1_4_2_values,
            'Int 3/(1-4)': norm_1_4_3_values,
            'Int 4/(1-4)': norm_1_4_4_values,
        })
        # if
        if os.path.exists(save_path):
            if continue_save:
                org_data_df = pd.read_excel(save_path)
                org_data_df = pd.concat([org_data_df, data_df], axis=0)
                org_data_df.to_excel(save_path, index=False)
                print(f"Data appended to {save_path}")
            else: # overwrite the file
                data_df.to_excel(save_path, index=False)
                print(f'New excel file created at {save_path}')
        else:
            data_df.to_excel(save_path, index=False)
            print(f'New excel file created at {save_path}')




