import os

import numpy as np
from matplotlib import pyplot as plt
from tifffile import imwrite
from ptufile import PtuFile  # Assuming this class provides necessary methods
# cell_types = [
#     'NLS-N1-3-4-7-8-9-10-11-13-14-15-16-240425-1',
#     ]
# cell_types = [
#     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240510-1notraw',
#     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240510-2',
#     ]
# cell_types = [
#     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-240429-NE-9',
#     'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-240429-NE-1',

    # ]
cell_types = [
    # 'NLS-x-231127-MS',
    # 'sfG-240524-466'
    # 'NTOM-1-4-10-NLS-14-240602'
    # 'NTOM-1-4-13-NLS-14-240602-1',
    # 'NTOM-1-4-13-NLS-14-240602-2',
    # 'NTOM-1-4-16-NLS-13-240602',
    # 'NTOM-1-4-NLS-16-240602'
    # 'NTOM-1-13-14-NLS-4-240602-1',
    # 'NTOM-1-13-14-NLS-4-240602-2'
    # 'NTOM-4-10-13-14-NLS-8-240602-1',
    # 'NTOM-4-10-13-14-NLS-8-240602-2',
    # 'NTOM-14-16-NLS-1-240602-1',
    # 'NTOM-14-16-NLS-1-240602-2'
    # 'NLS-NTOM-Mix39-240618-1',
    # 'NLS-NTOM-Mix39-240618-2',
    # 'NLS-NTOM-Mix39-240618-3',
    # 'NLS-14Mix-NLYN-5Mix-240618-1',
    # 'NLS-14Mix-NLYN-5Mix-240618-2',
    # N1,4,8,10,13,14,16
    # 'NLS-N1-240618',
    # 'NLS-N4-240618',
    # 'NLS-N8-240618',
    # 'NLS-N10-240618',
    # 'NLS-N13-240618',
    # 'NLS-N14-240618',
    # 'NLS-N16-240618',
    # 'NTOM-N10-240528',
    #   'NTOM-N13-240528',
# 'NTOM-N4-240528',
# 'NTOM-N14-240528',
#                  'NTOM-N16-240528',
# 'NTOM-N8-240528',
# 'NTOM-N1-240528',

#     'NTOM-M1-240629',
#     'NTOM-M4-240629',
#     'NTOM-M8-240629',
#     'NTOM-M10-240629',
#     'NTOM-M13-240629',
#     'NTOM-M14-240629',
#     'NTOM-M16-240629',
#     'NLS-TagRFP-T-240117'
#     'DLZT-2404'
#     'NTOM-M2-240814',
#     'NTOM-M9-240814-1',
#     'NTOM-M9-240814-2',
#     'NTOM-M12-240814-1',
#     'NTOM-M12-240814-2',
#     'NTOM-M15-240814',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-1-240826'
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-2-240826'
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-3-240826',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-4-240826',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-5-240826',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-6-240826',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-7-240826',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-8-240826',
#     'NLS1-NTOM-Mix6-240719'
#     'NLS16-NTOM-Mix6-240722-1',
#     'NLS16-NTOM-Mix6-240722-2',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-1-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-2-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-3-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-4-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-5-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-6-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-7-240827',
#     '9 Mix N3_N8_N9_M11_M1_M16_P4_P7_P10-8-240827',
#     'NLS16-NTOM-Mix6-240722-1',
#     'NLS16-NTOM-Mix6-240722-2',
#     'NLS14-NTOM-Mix6-240722-1',
#     'NLS14-NTOM-Mix6-240722-2',
#     'NLS1-NTOM13-240922'
#     'Mix6-1-241009-1'
    'Mix6-2-241009-1'
    ]
# dataset = 'MCF7-PKU'
# dataset = 'MEF-BJMU'
frame = -1 # -1 for all frames' sum, 0 for the first frame, 1 for the second frame, etc.
# frame = 0
# frame = '0_4'
# dataset = 'Hek293T-BJMU'
dataset = 'Hek293T-BJMU-Dual'
# dataset = ''
processed_fovs = []
# data_dir = fr"E:/BC-FLIM/{dataset}/"
data_dir = fr"I:/BC-FLIM/{dataset}/"

# data_dir = fr'I:\{dataset}'
for cell_type in cell_types:
    raw_dir = rf'{data_dir}/{cell_type}/raw'

    int_folder = rf'{data_dir}/{cell_type}/intensity'
    stack_folder = rf'{data_dir}/{cell_type}/flim_stack'
    if not frame == -1:
        # if type of frame is int
        if isinstance(frame, int):
            int_folder = rf'{data_dir}/{cell_type}/intensity_{frame+1}'
            stack_folder = rf'{data_dir}/{cell_type}/flim_stack_{frame+1}'
        else:
            int_folder = rf'{data_dir}/{cell_type}/intensity_{frame}'
            stack_folder = rf'{data_dir}/{cell_type}/flim_stack_{frame}'
    for fov in os.listdir(raw_dir):
        if not fov.endswith('.ptu'):
            continue
        fov = fov.split('.')[0]
        if fov in processed_fovs:
            continue
        print(f"Processing {fov}")
        ptu_path = rf'{raw_dir}/{fov}.ptu'

        if not os.path.exists(int_folder):
            os.makedirs(int_folder)
        if not os.path.exists(stack_folder):
            os.makedirs(stack_folder)

        with PtuFile(ptu_path) as ptu:
            # try:
            #     print(f"Number of channels: {ptu.number_channels}")
            #     print(f"Shape: {ptu.shape}")
            #     flim_data = ptu.decode_image(frame=-1)
            # except Exception as e:
            #     print(f"Error: {e}")

            # Directly debug the _info attribute and read_records method
            try:
                _info = ptu._info
                print(f"_info: {_info}")
            except Exception as e:
                print(f"Error retrieving _info: {e}")

            try:
                records = ptu.read_records()
                print(f"Records: {records}")
            except Exception as e:
                print(f"Error reading records: {e}")
            flim_data = ptu.decode_image(frame=frame)
            print(f"flim_data shape: {flim_data.shape}")
            # flim_data = flim_data.astype(np.uint16)

        # if flim_data shape one is 1, remove it, if more than 1, then sum them
        # if flim_data shape 4 is the number of detectors, if only 1, then no need to distinguish, if more than 1, then save them separately in flim_stack, sum them in intensity
        if flim_data.shape[0] == 1:
            flim_data = flim_data[0]
        else:
            flim_data = np.sum(flim_data, axis=0)


        for channel in range(flim_data.shape[2]): # channel 4 is the number of detectors, now it is channel 3, because channel 1 was removed
            channel_data = flim_data[:, :, channel, :]
            channel_data = channel_data.astype(np.uint8) # photons in single frame should be less than 255
            # if flim_data.shape[2] only has one channel, then no need to save separately, only save the summed intensity
            if flim_data.shape[2] > 1:
                # plot the first frame of each channel
                # plt.figure()
                # plt.imshow(channel_data[:, :, 40])
                # plt.title(f"Channel {channel+1}")
                # plt.show()
                # print(f"type of channel_data(uint8/uint16): {channel_data.dtype}")
                stack_path = rf'{stack_folder}/{fov}-{channel+1}.tif'

                imwrite(stack_path, channel_data.transpose(2, 0, 1), imagej=True)

            intensity_image = np.sum(flim_data[:, :, channel, :], axis=2)
            intensity_image = intensity_image.astype(np.uint16)
            int_path = rf'{int_folder}/{fov}-{channel+1}.tif'
            # imwrite(int_path, intensity_image, imagej=True)
            imwrite(int_path, intensity_image)
        # Summed intensity image across all channels and all time bins
        summed_intensity = np.sum(flim_data, axis=(2, 3))
        summed_intensity = summed_intensity.astype(np.uint16)
        summed_intensity_path = rf'{int_folder}/{fov}-sum.tif'
        # if only 1 channel, then the path should without sum
        if flim_data.shape[2] == 1:
            summed_intensity_path = rf'{int_folder}/{fov}.tif'
        imwrite(summed_intensity_path, summed_intensity, imagej=True)
        # Save the summed decay stack (151, 2048, 2048)
        summed_channels = np.sum(flim_data, axis=2)
        summed_channels = summed_channels.astype(np.uint16)
        summed_channels_path = rf'{stack_folder}/{fov}-sum.tif'
        imwrite(summed_channels_path, summed_channels.transpose(2, 0, 1), imagej=True)

        print(f"Processed {fov}")



