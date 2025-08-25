
import glob
import pathlib
import time
import traceback
from itertools import combinations

from matplotlib.lines import Line2D
from napari.layers import Labels
from napari.utils import Colormap, notifications
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from magicgui import widgets
import tifffile as tiff
from typing import TYPE_CHECKING, List
from magicgui.widgets import FloatSpinBox, FloatSlider, Label
from skimage.measure import regionprops, label
from ptufile import PtuFile
from napari.layers import Image as NapariImage
from qtpy.QtWidgets import QFileDialog

import cv2
import tifffile
from skimage.morphology import disk, erosion
from magicgui.widgets import Container,  ComboBox, SpinBox
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import center_of_mass
from typing import Union
from pathlib import Path

import tkinter as tk
from tkinter import messagebox
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
import os
import pandas as pd
import numpy as np
import napari
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from qtpy.QtWidgets import QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

def _load_ptu(ptu_path: Union[str, Path], frame: Union[int, str] = -1) -> np.ndarray:
    """
    Load and decode a PTU file. Returns numpy array; may be high-dimensional.
      - If frame == -1: returns full decoded stack
      - Else: returns only that frame
    """
    with PtuFile(str(ptu_path)) as ptu:
        data = ptu.decode_image(frame=frame)
    return np.asarray(data)

class PTUReader(Container):
    """
    Napari dock widget for reading PTU files and saving intensity and FLIM stacks.
    """
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__(layout='vertical')
        self.viewer = viewer

        # Widgets
        # self.input_dir = FileEdit(label='PTU Folder', mode='d', value=os.getcwd())
        # self.output_dir = FileEdit(label='Output Folder', mode='d', value=os.getcwd())
        # use J: as default input folder
        self.input_dir = widgets.FileEdit(label='PTU Folder', mode='d', value=r'J:/')
        self.output_dir = widgets.FileEdit(label='Output Folder', mode='d', value=r'J:/')
        self.frame = SpinBox(label='Frame (-1 for all)', min=-1, max=80, step=1, value=-1)
        self.process_btn = PushButton(text='Process and Save')
        self.process_btn.changed.connect(self._on_process)
        self.input_dir.changed.connect(self._on_input_dir_changed)

        # Add to layout
        for w in [self.input_dir, self.output_dir, self.frame, self.process_btn]:
            self.append(w)

        # viewer.window.add_dock_widget(self, area='right', name='PTU Reader')

    def _on_input_dir_changed(self):
        """When the PTU folder changes, update output to its parent."""
        new_input = self.input_dir.value
        if new_input and os.path.isdir(new_input):
            # use pathlib for clarity
            parent_dir = str(Path(new_input).parent)
            self.output_dir.value = parent_dir
    def _on_process(self):
        in_dir = Path(self.input_dir.value)
        out_dir = Path(self.output_dir.value)
        frame = self.frame.value
        # Validate
        if not in_dir.is_dir():
            show_warning(f"Input folder not found: {in_dir}")
            return
        out_int = out_dir / 'intensity'
        out_stack = out_dir / 'flim_stack'
        out_int.mkdir(parents=True, exist_ok=True)
        out_stack.mkdir(parents=True, exist_ok=True)

        ptu_files = sorted(in_dir.glob('*.ptu'))
        if not ptu_files:
            show_warning(f"No .ptu files in {in_dir}")
            return

        for p in ptu_files:
            try:
                raw = _load_ptu(p, frame)
                arr = np.array(raw)
                # Collapse extra dims by summing until at most 4 dims
                while arr.ndim > 4:
                    arr = arr.sum(axis=0)
                # Now arr.ndim <= 4
                # Cases:
                # 4D: (H, W, channels, bins)
                # 3D: (H, W, frames) or (H, W, channels)
                # 2D: (H, W)
                if arr.ndim == 4:
                    # Sum over bins (axis=3) to get intensity per channel
                    intensity = arr.sum(axis=3)
                    # Save each channel separately
                    for ch in range(intensity.shape[2]):
                        int_img = intensity[..., ch]
                        tifffile.imwrite(out_int / f"{p.stem}_ch{ch+1}.tif",
                                          int_img.astype(np.uint16), imagej=True)
                        # Save decay stack per channel
                        stack_ch = arr[..., ch, :].transpose(2, 0, 1)
                        tifffile.imwrite(out_stack / f"{p.stem}_ch{ch+1}.tif",
                                          stack_ch.astype(np.uint16), imagej=True)
                        # put the stack in the layer
                        self.viewer.add_image(stack_ch, name=f"{p.stem}_ch{ch+1}")
                    # Combined sum over channels
                    total_int = intensity.sum(axis=2)
                    tifffile.imwrite(out_int / f"{p.stem}_sum.tif",
                                      total_int.astype(np.uint16), imagej=True)
                    summed_stack = arr.sum(axis=2).transpose(2, 0, 1)
                    tifffile.imwrite(out_stack / f"{p.stem}_sum.tif",
                                      summed_stack.astype(np.uint16), imagej=True)
                elif arr.ndim == 3:
                    # Could be (H,W,frames) or (H,W,channels)
                    # Treat last dim as channel or frame and save both intensity and stack same
                    for i in range(arr.shape[2]):
                        slice = arr[..., i]
                        tifffile.imwrite(out_int / f"{p.stem}_slice{i+1}.tif",
                                          slice.astype(np.uint16), imagej=True)
                        tifffile.imwrite(out_stack / f"{p.stem}_slice{i+1}.tif",
                                          slice.astype(np.uint16), imagej=True)
                elif arr.ndim == 2:
                    tifffile.imwrite(out_int / f"{p.stem}.tif",
                                      arr.astype(np.uint16), imagej=True)
                    tifffile.imwrite(out_stack / f"{p.stem}.tif",
                                      arr.astype(np.uint16), imagej=True)
                else:
                    show_warning(f"Unexpected array dims: {arr.ndim} for {p.name}")
                # show_info(f"Processed {p.name}")
                notifications.show_info(f"Processed {p.name}")
            except Exception as e:
                show_warning(f"Failed {p.name}: {e}")
        # show_info("All PTU files processed, please segment the intensity_sum.tif image to get the masks in Cellpose, then come back to the next plugin.")
        notifications.show_info("All PTU files processed, please segment the intensity_sum.tif image to get the masks in Cellpose, then come back to the next plugin.")


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

    roi_decay_data_normalized = np.delete(roi_decay_data_normalized,
                                          np.arange(peak2_begin - mask_start, peak2_end - mask_start))
    t_arr = np.delete(t_arr, np.arange(peak2_begin - mask_start, peak2_end - mask_start))

    # fastflim calculation
    fastflim = np.sum(roi_decay_data_normalized * t_arr) / np.sum(roi_decay_data_normalized)

    params_init = [1, 2, 0]
    popt, pcov = curve_fit(exp_func, t_arr, roi_decay_data_normalized, p0=params_init)
    # get the lifetime and
    lifetime = popt[1]
    chi_square = np.sum((roi_decay_data_normalized - exp_func(t_arr, *popt)) ** 2)
    pulse_freq = pulse_freq / 1000 # MHz to GHz
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
        notifications.show_info('Excel file already exists, will overwrite the data.')
    data_df.to_excel(save_path, index=False)
    print(f'Excel file saved at {save_path}')
    notifications.show_info(f'Excel file saved at {save_path}')
    return data_df

def get_color_map(n_colors):
    color_list_14 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#469990', '#9A6324', '#808000', '#000075', '#800000', '#aaffc3']
    colors = color_list_14[:n_colors]
    return colors

if TYPE_CHECKING:
    import napari

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def plot_signal(self, *signals, titles=None):
        """
        Plots one or more 1D signals in a single row of subplots.

        Parameters:
            *signals: A variable number of 1D arrays representing the signals to plot.
            titles (optional): A list of titles for each subplot. If not provided, default titles will be used.
        """
        # Clear the current figure.
        self.figure.clear()

        n = len(signals)
        if n == 0:
            return  # Nothing to plot.

        # Use default titles if none were provided.
        if titles is None or len(titles) != n:
            titles = [f"Signal {i + 1}" for i in range(n)]

        # Create one row with n columns.
        for i, signal in enumerate(signals):
            ax = self.figure.add_subplot(1, n, i + 1)
            ax.plot(signal)
            ax.set_title(titles[i])

        # Refresh the canvas.
        self.canvas.draw()
        print('Signal plotted')

    def plot_phasor(self, g_values, s_values):
        # self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(g_values, s_values)
        # Draw a semi-circle for reference
        theta = np.linspace(0, np.pi, 100)
        x = 0.5 + 0.5 * np.cos(theta)
        y = 0.5 * np.sin(theta)
        ax.plot(x, y)
        ax.set_xlabel('G')
        ax.set_ylabel('S')
        ax.set_title('Phasor plot')
        self.canvas.draw()

    def plot_phasor_cls(self, g_values, s_values, class_values, colors, checked_cls):
        # self.figure.clear()
        # ax = self.figure.add_subplot(142)
        ax = self.figure.add_subplot(132)
        # set figure size
        self.figure.set_size_inches(7,7)
        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue
            idx = class_values == i
            ax.scatter(g_values[idx], s_values[idx], color=color, label=f'Class {i}')
        ax.set_xlabel('G')
        ax.set_ylabel('S')
        ax.set_title('Phasor plot')
        self.canvas.draw()

    def plot_int12_cls(self, int1, int2, class_values, colors, checked_cls):
        # self.figure.clear()
        ax = self.figure.add_subplot(143)
        # set figure size
        self.figure.set_size_inches(7,7)
        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue
            idx = class_values == i
            ax.scatter(int1[idx], int2[idx], color=color, label=f'Class {i}')
        ax.set_xlabel('Int 1')
        ax.set_ylabel('Int 2')
        ax.set_title('Int 1-2 plot')
        self.canvas.draw()

    def plot_int13_cls(self, int1, int3, class_values, colors, checked_cls):
        # self.figure.clear()
        ax = self.figure.add_subplot(144)
        # set figure size
        self.figure.set_size_inches(7,7)
        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue
            idx = class_values == i
            ax.scatter(int1[idx], int3[idx], color=color, label=f'Class {i}')
        ax.set_xlabel('Int 1')
        ax.set_ylabel('Int 3')
        ax.set_title('Int 1-3 plot')
        self.canvas.draw()

    def plot_int123_cls(self, int1, int2, int3, class_values, colors, checked_cls):
        # self.figure.clear()
        ax = self.figure.add_subplot(133)
        self.figure.set_size_inches(10, 7)  # Adjust the size as necessary
        marker_styles = ['o', '*']

        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue

            # Indices where the class matches
            idx = class_values == i

            # Plot int2 with the first marker style
            ax.scatter(int1[idx], int2[idx], color=color, marker=marker_styles[0], label=f'Class {i} int2')

            # Plot int3 with the second marker style
            ax.scatter(int1[idx], int3[idx], color=color, marker=marker_styles[1], label=f'Class {i} int3')

        ax.set_xlabel('Int 1')
        ax.set_ylabel('Int 2(o) & Int 3(*)')
        ax.set_title('Spectral Intensity Plot')
        # ax.legend()
        self.canvas.draw()







class Calculate_FLIM_S(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create widget for selecting TIFF stacks
        self._stack_selectors = [
            create_widget(label=f"Stack {i + 1}", annotation="napari.layers.Image")
            for i in range(4)
        ]

        # Widget to choose the output directory
        self._output_dir = FileEdit(label="Output Folder", mode='d')
        # self._output_dir.value = ''  # Default value is the current directory
        self._output_dir.value = r'J:/'
        # Widget for selecting a segmentation image
        self._segmentation_image = create_widget(label="Segmentation", annotation="napari.layers.Labels")

        # Parameter input: mask intensity threshold, pixel intensity threshold, peak offset, end offset, tau resolution, pulse frequency, harmonics
        self._mask_int_thres = create_widget(label="Mask Intensity Threshold", widget_type="Slider", value=1500)
        self._mask_int_thres.min = 500
        self._mask_int_thres.max = 100000

        self._pixel_int_thres = create_widget(label="Pixel Intensity Threshold", widget_type="Slider", value=7)
        self._pixel_int_thres.min = 0
        self._pixel_int_thres.max = 500

        self._peak_offset = create_widget(label="Peak Offset(bins)", widget_type="Slider", value=4)
        self._peak_offset.min = 0
        self._peak_offset.max = 20

        self._end_offset = create_widget(label="End Offset(bins)", widget_type="Slider", value=18)
        self._end_offset.min = 0
        self._end_offset.max = 50

        self._tau_resolution = create_widget(
            label="Tau Resolution(ns)",
            widget_type=FloatSpinBox,
            value=0.097,
            options={'min': 0, 'max': 1, 'step': 0.001}  # Use options to specify min, max, step
        )

        self._pulse_frequency = create_widget(
            label="Pulse Frequency(MHz)",
            widget_type=FloatSpinBox,
            value=78.1,
            options={'min': 0, 'max': 100, 'step': 0.1}
        )

        self._harmonics = create_widget(label="Harmonics", widget_type="Slider", value=1)
        self._harmonics.min = 1
        self._harmonics.max = 10

        # add a true or fulse checkbox for pixel-wise or not
        self._pixel_wise = CheckBox(label="Pixel-wise") # there's no checked keyword argument!

        # Summation and processing button
        self._process_button = PushButton(text="Process and Save to Excel")
        self._process_button.clicked.connect(self.process_and_save_to_excel)

        # Extend the container with the new widgets and button
        self.extend(
            self._stack_selectors +
            [self._output_dir, self._segmentation_image, self._mask_int_thres, self._pulse_frequency,
             self._pixel_int_thres, self._peak_offset, self._end_offset, self._tau_resolution,
             self._harmonics, self._pixel_wise, self._process_button]
        )

        # self.plot_widget = PlotWidget()
        # viewer.window.add_dock_widget(self.plot_widget, area='right', name='Phasor Plot')
        self._populate_initial_layers()
    def _populate_initial_layers(self):
        """Auto-fill stack1–4 from any Image layers named *_ch1…_ch4,
           and first Labels/‘seg’ layer into the segmentation selector."""
        # 1) stacks
        for i, selector in enumerate(self._stack_selectors):
            target_tag = f"_ch{i + 1}"
            for layer in self._viewer.layers:
                if isinstance(layer, NapariImage) and target_tag in layer.name:
                    selector.value = layer
                    break

        # 2) segmentation
        for layer in self._viewer.layers:
            if isinstance(layer, Labels) or 'seg' in layer.name.lower():
                self._segmentation_image.value = layer
                break
    def process_and_save_to_excel(self):
        # Get the data from the widgets
        stack_1 = self._stack_selectors[0].value
        stack_2 = self._stack_selectors[1].value
        stack_3 = self._stack_selectors[2].value
        stack_4 = self._stack_selectors[3].value
        output_folder = self._output_dir.value
        segmentation = self._segmentation_image.value
        mask_int_thres = self._mask_int_thres.value
        pixel_int_thres = self._pixel_int_thres.value
        peak_offset = self._peak_offset.value
        end_offset = self._end_offset.value
        tau_resolution = self._tau_resolution.value
        pulse_frequency = self._pulse_frequency.value
        harmonics = self._harmonics.value
        pixel_wise = self._pixel_wise.value

        # Placeholder for the actual implementation of data processing and Excel output
        stack_1 = stack_1.data
        stack_2 = stack_2.data
        stack_3 = stack_3.data
        stack_4 = stack_4.data
        segmentation = segmentation.data

        data_df = Gen_excel(stack_1, stack_2, stack_3, stack_4, output_folder, segmentation, mask_int_thres, pixel_int_thres,
                             peak_offset, end_offset, tau_resolution, pulse_frequency, harmonics, pixel_wise)

        # show G,S scatter plot on the right bar in napari, using matplotlib
        plt.figure()
        plt.scatter(data_df['G'], data_df['S'])
        # also draw a semi-circle for reference, center = (0.5,0), r = 0.5, S > 0
        theta = np.linspace(0, np.pi, 100)
        x = 0.5 + 0.5 * np.cos(theta)
        y = 0.5 * np.sin(theta)
        plt.plot(x, y, 'r--')
        plt.xlabel('G')
        plt.ylabel('S')
        plt.title('G-S plot')
        plt.show()

        # self._update_phasor_plot(data_df['G'], data_df['S'])

    # def _update_phasor_plot(self, g_values, s_values):
    #     self.plot_widget.plot_phasor(g_values, s_values)


def get_color_map(n_colors):
    """
    Return n_colors RGB tuples (0-1). First color is always gray. If n_colors <= 15,
    use predefined hex list; otherwise use gray + tab20 palette.
    """
    base_hex = [
        '#808080', '#e6194B', '#3cb44b', '#ffe119', '#4363d8',
        '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#469990',
        '#9A6324', '#808000', '#000075', '#800000', '#aaffc3'
    ]
    if n_colors <= len(base_hex):
        hex_list = base_hex[:n_colors]
        colors = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex_list]) / 255.0
    else:
        # always first gray
        colors = [np.array([128, 128, 128]) / 255.0]
        cmap = plt.cm.tab20
        for i in range(1, n_colors):
            rgb = cmap((i - 1) % 20)[:3]
            colors.append(np.array(rgb))
        colors = np.stack(colors)
    return colors
class KMeansCluster(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__(layout='vertical')
        # reduce spacing between rows in the main container
        main_layout = self.native.layout()
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = viewer
        self.df_test = None
        self.df_ref = None
        self.dims = []
        self.fig = None
        self.axes = None
        # self.seeds = []
        self.seed_indices: List[int] = []
        self.seed_artists: List[Line2D] = []
        self.selected_class = 1
        self.lasso = None
        self._cid_seed = None
        self.marker_size_pt = 12
        self.drag_threshold_px = 12
        # Test folder pickers (5)
        self.test_folders = []
        for i in range(3):  # only 3 test folders
            row = Container(layout='horizontal')
            row.append(Label(value=f'Test Folder {i+1}'))
            fe = FileEdit(label='', filter='*', mode='d')
            row.append(fe)
            self.append(row)
            self.test_folders.append(fe)

        # Reference folder pickers (14)
        self.ref_folders = []
        for j in range(14):
            row = Container(layout='horizontal')
            row.append(Label(value=f'Ref Folder {j+1}'))
            fe = FileEdit(label='', filter='*', mode='d')
            row.append(fe)
            self.append(row)
            self.ref_folders.append(fe)

        # Intensity Threshold
        row = Container(layout='horizontal')
        row.append(Label(value='Intensity Threshold'))
        self.threshold = FloatSpinBox(min=0, max=1e6, step=1e3, value=200000)
        row.append(self.threshold)
        self.append(row)

        # Number of Clusters
        row = Container(layout='horizontal')
        row.append(Label(value='Number of Clusters'))
        self.n_clusters = SpinBox(min=1, max=50, value=5)
        row.append(self.n_clusters)
        self.append(row)

        # Weights for dimensions
        self.weights = {}
        for d in ['G', 'S', 'Int1', 'Int2', 'Int3']:
            row = Container(layout='horizontal')
            row.append(Label(value=f'Weight {d}'))
            if d in ['G', 'S']:
                value = 4.0
            else:
                value = 1.0
            w = FloatSpinBox(min=0.0, max=10.0, step=0.1, value=value)
            row.append(w)
            self.append(row)
            self.weights[d] = w

        seed_btn_row = Container(layout='horizontal')
        self.save_seeds_btn = PushButton(text='Save Seeds')
        self.save_seeds_btn.clicked.connect(self.save_seeds)
        seed_btn_row.append(self.save_seeds_btn)
        self.load_seeds_btn = PushButton(text='Load Seeds')
        self.load_seeds_btn.clicked.connect(self.load_seeds)
        seed_btn_row.append(self.load_seeds_btn)
        self.append(seed_btn_row)


        # Buttons row
        btn_row = Container(layout='horizontal')
        self.load_button = PushButton(text='Read and Plot')
        self.load_button.clicked.connect(self.load_and_plot)
        btn_row.append(self.load_button)
        self.run_button = PushButton(text='Run K-Means')
        self.run_button.clicked.connect(self.run_kmeans)
        btn_row.append(self.run_button)
        self.save_button = PushButton(text='Save Results')
        self.save_button.clicked.connect(self.save_results)
        btn_row.append(self.save_button)
        self.append(btn_row)

    def _notify(self, msg: str):
        # print(msg)
        napari.utils.notifications.show_info(msg)
    def load_and_plot(self):
        def find_excels(base):
            if not base or not os.path.isdir(base):
                return None
            files = os.listdir(base)
            excel = None
            for name in ['FLIM-S.xlsx'] + [f for f in files if 'barcode' in f.lower() and f.lower().endswith('.xlsx')] + [f for f in files if f.lower().endswith('.xlsx')]:
                fp = os.path.join(base, name)
                if os.path.isfile(fp):
                    print(f'Found Excel: {fp}')
                    excel = fp
                    break
            if not excel:
                return None
            df = pd.read_excel(excel)
            required = ['G','S']
            if not all(col in df.columns for col in required):
                print(f"Skipping {excel}: missing required columns {required}")
                return None
            df['subfolder'] = os.path.basename(base)
            df['base_folder'] = base
            return df

        # Load test data
        dfs_test = []
        for fe in self.test_folders:
            path = fe.value
            if not path or not os.path.isdir(path):
                continue
            df = find_excels(path)
            if df is not None:
                dfs_test.append(df)
        self.df_test = pd.concat(dfs_test, ignore_index=True) if dfs_test else None
        if self.df_test is None:
            print('No test data loaded.'); return
        self._notify('Test data loaded.')

        # Load reference data
        dfs_ref = []
        # for fe in self.ref_folders:
        for i, fe in enumerate(self.ref_folders):
            path = fe.value
            if not path or not os.path.isdir(path):
                continue
            df = find_excels(path)
            if df is None:
                print(f'Skipping empty or invalid reference folder: {path}')
                continue
            # add class column, class i add in
            df = df.copy()
            df['class'] = i + 1  # classes start from 1

            if df is not None:
                dfs_ref.append(df)
        self.df_ref = pd.concat(dfs_ref, ignore_index=True) if dfs_ref else None
        if self.df_ref is not None:
            self._notify('Reference data loaded.')

        # Detect dimensions
        cols = self.df_test.columns
        self.dims = ['G','S'] + [d for d in ['Int 1/(1-4)','Int 2/(1-4)','Int 3/(1-4)'] if d in cols]

        # Filter by intensity
        if 'Total intensity' in cols:
            thr = self.threshold.value
            self.df_test = self.df_test[self.df_test['Total intensity'] > thr]
        self.df_test.dropna(subset=self.dims, inplace=True)

        # Prepare weighted and scaled data
        wvals = [self.weights['G'].value, self.weights['S'].value]
        for d in ['Int1','Int2','Int3']:
            key = f'Int {d[-1]}/(1-4)' if d!='Int1' else 'Int 1/(1-4)'
            if key in self.dims:
                wvals.append(self.weights[d].value)
        W = np.array(wvals)
        X = self.df_test[self.dims].to_numpy() * W
        Xs = StandardScaler().fit_transform(X)
        self.df_scaled = Xs

        # build pairs for subplots
        pairs = [(self.dims[0], self.dims[1])]
        if len(self.dims) >= 4:
            pairs.append((self.dims[2], self.dims[3]))
        if len(self.dims) == 5:
            pairs.append((self.dims[3], self.dims[4]))
        self.pairs = pairs

        # draw fresh figure
        if self.fig:
            plt.close(self.fig)
        self.fig, axes_arr = plt.subplots(
            1, len(pairs), figsize=(4 * len(pairs), 3), constrained_layout=True
        )
        self.axes = list(np.atleast_1d(axes_arr).flatten())
        if self.df_ref is not None:
            colors_ref = get_color_map(self.df_ref['class'].nunique() + 1)

        for ax, (xd, yd) in zip(self.axes, self.pairs):
            ax.scatter(self.df_test[xd], self.df_test[yd], s=20, color='blue', alpha=0.8)
            if self.df_ref is not None:
                for cls, grp in self.df_ref.groupby('class'):
                    #
                    ax.scatter(
                        grp[xd], grp[yd],
                        s=30,
                        color=colors_ref[cls],
                        alpha=0.02,
                        label=f"Ref {cls}"
                    )
            ax.set_xlabel(xd)
            ax.set_ylabel(yd)
            ax.set_title(f'{xd} vs {yd}')
        self._cid_seed = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        self.seeds = []

    def on_click(self, event):
        self.drag_threshold_px = (
                (self.marker_size_pt / 2)
                * self.fig.dpi / 72
        )
        if event.inaxes in self.axes and self.seed_indices:
            ax_idx = self.axes.index(event.inaxes)
            x_dim, y_dim = self.pairs[ax_idx]

            seed_data = self.df_test.iloc[self.seed_indices][[x_dim, y_dim]].to_numpy()
            seed_disp = event.inaxes.transData.transform(seed_data)
            click_disp = np.array([event.x, event.y])
            dists_px = np.linalg.norm(seed_disp - click_disp, axis=1)

            closest = int(dists_px.argmin())
            if dists_px[closest] < self.drag_threshold_px:
                self._dragging_seed = True
                self._drag_ax_idx = ax_idx
                self._drag_artist_idx = closest
                self._drag_artist = self.seed_artists[closest]
                self._cid_motion = self.fig.canvas.mpl_connect(
                    'motion_notify_event', self._on_seed_motion
                )
                return

        if event.inaxes not in self.axes:
            return

        if len(self.seed_indices) >= self.n_clusters.value:
           # remove all old seed markers from
            for art in self.seed_artists:
                art.remove()

            self.seed_indices.clear()
            self.seed_artists.clear()
            self._notify("Seed number exceeded, all seeds cleared. Please select new seeds.")

        ax_idx = self.axes.index(event.inaxes)
        x_dim, y_dim = self.pairs[ax_idx]
        x, y = event.xdata, event.ydata

        #
        coords = self.df_test[[x_dim, y_dim]].to_numpy()
        idx = int(np.argmin(cdist([(x, y)], coords)))
        self.seed_indices.append(idx)
        self._draw_seed(idx)

        #
        # self.seeds.append(self.df_scaled[idx])
        # colors = get_color_map(self.n_clusters.value + 1)
        # seed_idx = len(self.seeds)  # 第几个 seed，从 1 开始
        # color = colors[seed_idx]
        # row = self.df_test.iloc[idx]
        # for ax, (xd, yd) in zip(self.axes, self.pairs):
        #     ax.plot(row[xd], row[yd], '*', markersize=12, color=color)
        # plt.draw()

        if len(self.seeds) == self.n_clusters.value:
            self._notify('All seeds selected, time for K-Means clustering!')
            if self._cid_seed is not None:
                self.fig.canvas.mpl_disconnect(self._cid_seed)
                self._cid_seed = None

    def _draw_seed(self, idx: int):
        """Plot a draggable star at the raw-data coords of row idx."""
        row = self.df_test.iloc[idx]
        for ax, (xd, yd) in zip(self.axes, self.pairs):
            artist, = ax.plot(row[xd], row[yd], '*', markersize=self.marker_size_pt,
                              color=get_color_map(self.n_clusters.value + 1)[len(self.seed_indices)])
            artist.set_picker(5)  # enable picking
            self.seed_artists.append(artist)
        self.fig.canvas.draw_idle()
        # connect drag event once
        if not hasattr(self, '_drag_cid'):
            self._drag_press_cid = self.fig.canvas.mpl_connect('pick_event', self._on_seed_press)
            self._drag_release_cid = self.fig.canvas.mpl_connect('button_release_event', self._on_seed_release)

    def save_seeds(self):
        # from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(None, "Save seeds to .npz", "", "NumPy files (*.npz)")
        if not path: return
        np.savez(path, seed_indices=np.array(self.seed_indices, dtype=int))
        self._notify(f"Seeds saved to {path}")



    def load_seeds(self):
        # from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(None, "Load seeds from .npz", "", "NumPy files (*.npz)")

        if not path:
            self._notify("No file selected for loading seeds.")
            return
        data = np.load(path)
        loaded = data['seed_indices'].tolist()

        if len(loaded) > self.n_clusters.value:
            self._notify(
                f"Attempted to load {len(loaded)} seeds, "
                f"but number of clusters is {self.n_clusters.value}. "
                "Load cancelled."
            )
            return

        for art in self.seed_artists:
            art.remove()
        self.seed_artists.clear()
        self.seed_indices.clear()

        for idx in loaded:
            self.seed_indices.append(idx)
            self._draw_seed(idx)

        self._notify(f"Loaded {len(self.seed_indices)} seeds")

    def _on_seed_press(self, event):
        # begin dragging one of our seed markers
        if event.artist in self.seed_artists:
            self._dragging_artist = event.artist
            self._drag_orig_xy = (event.mouseevent.xdata, event.mouseevent.ydata)
            self._drag_artist_idx = self.seed_artists.index(event.artist)
            self._cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_seed_motion)

    def _on_seed_motion(self, event):
        if not hasattr(self, '_dragging_artist'): return
        self._dragging_artist.set_data(event.xdata, event.ydata)
        self.fig.canvas.draw_idle()

    def _on_seed_release(self, event):
        # finish drag: snap to nearest data‐point
        if not hasattr(self, '_dragging_artist'): return
        artist = self._dragging_artist
        ax_idx = next(i for i, a in enumerate(self.axes) if a == artist.axes)
        xd, yd = self.pairs[ax_idx]
        # find nearest index in raw df_test
        coords = self.df_test[[xd, yd]].to_numpy()
        dists = np.linalg.norm(coords - np.array([event.xdata, event.ydata]), axis=1)
        new_idx = int(dists.argmin())
        # update our records
        self.seed_indices[self._drag_artist_idx] = new_idx
        # redraw that star at the true location
        row = self.df_test.iloc[new_idx]
        artist.set_data(row[xd], row[yd])
        # cleanup
        self.fig.canvas.mpl_disconnect(self._cid_motion)
        delattr(self, '_dragging_artist')
        self._notify(f"Seed {self._drag_artist_idx + 1} reassigned to cell #{new_idx}")
        self.fig.canvas.draw_idle()


    def run_kmeans(self):
        if len(self.seed_indices) != self.n_clusters.value:
            self._notify(f"Please select {self.n_clusters.value} seeds before running K-Means, currently {len(self.seed_indices)} selected.")
            return

        init_seeds = [self.df_scaled[i] for i in self.seed_indices]
        km = KMeans(n_clusters = self.n_clusters.value, init = np.vstack(init_seeds), n_init = 1)
        labels = km.fit_predict(self.df_scaled)
        # shift labels so 1..n and reserve 0 for unassigned
        self.df_test['cluster'] = labels + 1

        # plot clusters with custom colormap
        n_total = self.n_clusters.value + 1
        colors = get_color_map(n_total)

        for ax, (xd, yd) in zip(self.axes, self.pairs):
            ax.clear()
            for k in range(n_total):
                sel = self.df_test['cluster'] == k
                ax.scatter(
                    self.df_test.loc[sel, xd],
                    self.df_test.loc[sel, yd],
                    color=colors[k], s=20
                )
            ax.set_xlabel(xd)
            ax.set_ylabel(yd)
            ax.set_title(f'{xd} vs {yd}')

        plt.draw()

        cid1 = self.fig.canvas.mpl_connect('button_press_event', self._on_manual_click)
        cid2 = self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)
        self._notify('Clustering done. Now you can manually reassign points by shift+clilck for lasso-selector activation on plot, using keys 1-9, a-z for classes 1-35 to assign classes.')

    def _on_keypress(self, event):
        key = event.key
        if not key:
            return

        # 1–9
        if key.isdigit():
            val = int(key)

        # a–z → 10–35
        elif len(key) == 1 and 'a' <= key <= 'z':
            val = ord(key) - ord('a') + 10

        else:
            return

        if 0 <= val <= self.n_clusters.value:
            self.selected_class = val
            self._notify(f"Selected class {val} (for next assignment)")

    def _on_manual_click(self, event):
        #
        if event.inaxes not in self.axes:
            return

        ax_idx = self.axes.index(event.inaxes)
        x_dim, y_dim = self.pairs[ax_idx]
        ix, iy = event.xdata, event.ydata

        # debug
        print(f"[Manual] click at ({ix:.2f},{iy:.2f}) on subplot #{ax_idx} dims=({x_dim},{y_dim})")

        coords = self.df_test[[x_dim, y_dim]].to_numpy()

        #
        if event.key == 'control' and event.button == 1:
            dists = cdist([(ix, iy)], coords).flatten()
            idx = int(dists.argmin())
            print(f" → nearest idx={idx}, coord={coords[idx]}, dist={dists[idx]:.3f}")
            print(f" original cluster: {self.df_test.at[idx, 'cluster']}")
            self.df_test.at[idx, 'cluster'] = self.selected_class
            self.idx_changed = idx
            self._redraw_clusters()
            self._notify(f"Point {idx}→cluster {self.selected_class}")

        elif event.key == 'shift' and event.button == 1:
            self._notify("Starting lasso selection. Click to draw polygon.")
            # start a LassoSelector on this axis
            if self.lasso:
                self.lasso.disconnect_events()
            ax = event.inaxes
            self.lasso = LassoSelector(ax, onselect=self._on_lasso_select)

    def _on_lasso_select(self, verts):
        """Called when the lasso polygon is completed."""
        path = MplPath(verts)
        # build data array matching whichever axis lasso was drawn on
        ax = self.lasso.ax
        if ax == self.axes[0]:
            dims = self.dims[:2]
        else:
            # assume additional dims plotting: use dims[2],dims[3] for axes[1], etc.
            idx = self.axes.index(ax)
            dims = self.dims[2 * idx:2 * idx + 2]
        pts = self.df_test[dims].to_numpy()
        mask = path.contains_points(pts)
        self.df_test.loc[mask, 'cluster'] = self.selected_class
        self._lasso_cleanup()
        self._redraw_clusters()
        self._notify(f"{mask.sum()} points → cluster {self.selected_class}")

    def _lasso_cleanup(self):
        self.lasso.disconnect_events()
        self.lasso = None

    def _redraw_clusters(self):
        try:
            print(f'idx_changed: {self.idx_changed}')
            print(f'cluster: {self.df_test.at[self.idx_changed, "cluster"]}')
        except AttributeError:
            pass
        # same logic as in run_kmeans, but without re-fitting
        n_total = self.n_clusters.value + 1
        colors = get_color_map(n_total)
        for ax, (xd, yd) in zip(self.axes, self.pairs):
            ax.clear()
            for k in range(n_total):
                sel = self.df_test['cluster'] == k
                ax.scatter(
                    self.df_test.loc[sel, xd],
                    self.df_test.loc[sel, yd],
                    color=colors[k], s=20
                )

            ax.set_xlabel(xd)
            ax.set_ylabel(yd)
            ax.set_title(f'{xd} vs {yd}')
        print('Redrew clusters.')
        self.fig.canvas.draw_idle()

    def _select_npy_file(self, initial_folder):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            None,
            "Select segmentation .npy file",
            initial_folder,
            "NumPy files (*.npy);;All Files (*)",
            options=options
        )
        return file or None

    def _process_masks_for_folder(self, folder, df_grp):
        int_folder = os.path.join(folder, 'intensity')
        if not os.path.isdir(int_folder):
            self._notify(f"No intensity folder at {int_folder}, skipping masks for {folder}.")
            return

        # 必要列检查
        if not all(k in df_grp.columns for k in ['FOV', 'Mask label', 'cluster']):
            self._notify(f"Missing required columns in data for {folder}. Need 'FOV','Mask label','cluster'.")
            return

        # cluster 数量（fallback 取 n_clusters 控件的值）
        n_clusters = None
        if hasattr(self, 'n_clusters'):
            try:
                n_clusters = self.n_clusters.value
            except Exception:
                pass
        if n_clusters is None:
            n_clusters = getattr(self, 'num_clusters', 0)
        total_classes = (n_clusters if isinstance(n_clusters, int) else 0) + 1


        for fov in df_grp['FOV'].unique():
            # 找 segmentation npy
            seg_path = os.path.join(int_folder, f'{fov}-sum_seg.npy')
            if not os.path.isfile(seg_path):
                candidates = glob.glob(os.path.join(int_folder, f"{fov}*seg*.npy"))
                if len(candidates) == 1:
                    seg_path = candidates[0]
                else:
                    all_npy = glob.glob(os.path.join(int_folder, '*.npy'))
                    if len(all_npy) == 1:
                        seg_path = all_npy[0]
                    else:
                        self._notify(f"Need segmentation .npy for FOV '{fov}' in {int_folder}. Prompting user.")
                        seg_path = self._select_npy_file(int_folder)
                        if not seg_path:
                            self._notify(f"Skipping FOV {fov}: no segmentation file provided.")
                            continue

            # 载入 segmentation
            try:
                data = np.load(seg_path, allow_pickle=True).item()
            except Exception as e:
                self._notify(f"Failed to load '{seg_path}': {e}")
                continue

            mask_cp = data.get('masks', None)
            if mask_cp is None:
                self._notify(f"No 'masks' key in '{seg_path}', skipping FOV {fov}.")
                continue

            # 侵蚀处理（为每个 label 做局部腐蚀）
            mask_cp_eroded = np.zeros_like(mask_cp)
            erosion_disk = disk(1)
            for mask_label in np.unique(mask_cp):
                if mask_label == 0:
                    continue
                mask_region = mask_cp == mask_label
                props = regionprops(mask_region.astype(int))
                if not props:
                    continue
                prop = props[0]
                minr, minc, maxr, maxc = prop.bbox
                minr = max(0, minr - 1)
                minc = max(0, minc - 1)
                maxr = min(mask_region.shape[0], maxr + 1)
                maxc = min(mask_region.shape[1], maxc + 1)

                cropped = mask_region[minr:maxr, minc:maxc]
                eroded_cropped = erosion(cropped, erosion_disk)
                mask_cp_eroded[minr:maxr, minc:maxc][eroded_cropped] = mask_label

            # 构造 cluster mask
            df_fov = df_grp[df_grp['FOV'] == fov].copy()
            df_fov['cluster'] = df_fov['cluster'].astype(np.uint8)
            mask = np.zeros_like(mask_cp, dtype=np.uint8)
            for _, row in df_fov.iterrows():
                ml = row['Mask label']
                cluster_id = row['cluster']
                mask[mask_cp_eroded == ml] = cluster_id

            # 保存纯数字的 cluster mask
            cls_path = os.path.join(int_folder, f'{fov}-cls.tif')
            try:
                tiff.imwrite(cls_path, mask)
                self._notify(f"Saved cluster mask for FOV {fov} to {cls_path}.")
            except Exception as e:
                self._notify(f"Failed to save cluster mask for FOV {fov}: {e}")

            # 生成带颜色的 mask
            colors = get_color_map(total_classes)
            mask_color_map = np.array(colors)  # 预期是 [0,1] 范围的 RGB tuples

            mask_color_map[0] = (0, 0, 0)

            # 用索引映射成 HxWx3
            try:
                mask_color = mask_color_map[mask]
            except Exception as e:
                self._notify(f"Color mapping failed for FOV {fov}: {e}")
                continue
            mask_color = (mask_color * 255).astype(np.uint8)

            # 写出彩色 mask
            color_path = os.path.join(int_folder, f'{fov}-cls-color.tif')
            try:
                tiff.imwrite(color_path, mask_color)
                self._notify(f"Saved colored mask for FOV {fov} to {color_path}.")
            except Exception as e:
                self._notify(f"Failed to save colored mask for FOV {fov}: {e}")

    def save_results(self):
        # 遍历每个 subfolder
        for folder, grp in self.df_test.groupby('base_folder'):
            filename = 'clustered.xlsx'
            path = os.path.join(folder, filename)

            # 如果已经存在，弹框询问
            if os.path.exists(path):
                from qtpy.QtWidgets import QMessageBox
                reply = QMessageBox.question(
                    None,
                    "Overwrite existing file?",
                    f"‘{filename}’ already exists in:\n{folder}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    # 跳过当前 subfolder，不覆盖它
                    continue

            # 保存到 clustered.xlsx
            try:
                grp.to_excel(path, index=False)
            except Exception as e:
                self._notify(f"Failed to save to {path}: {e}")
                continue

            # 处理 mask（segmentation npy → cluster mask/color mask）
            self._process_masks_for_folder(folder, grp)

        self._notify("Save complete.")




from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QLabel,
    QFormLayout
)
from magicgui.widgets import Container, create_widget, PushButton, FileEdit, CheckBox
import magicgui
# Inject create_widget into magicgui's top-level namespace for compatibility.
magicgui.create_widget = create_widget
from typing import Annotated
from scipy.ndimage import sum as ndi_sum, mean as ndi_mean


# Helper: add a row to a QFormLayout.
# If more than one widget is provided, they are arranged horizontally.
def add_form_row(form: QFormLayout, prompt: str, widgets):
    if len(widgets) == 1:
        form.addRow(prompt, widgets[0].native)
    else:
        container = QWidget()
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            hlayout.addWidget(w.native)
        container.setLayout(hlayout)
        form.addRow(prompt, container)

# Helper: add a row with two label-widget pairs in the same row.
def add_double_row(form: QFormLayout, prompt1: str, widget1, prompt2: str, widget2, bg_color: str):
    container = QWidget()
    container.setStyleSheet(f"background-color: {bg_color};")
    hlayout = QHBoxLayout(container)
    hlayout.setContentsMargins(0, 0, 0, 0)

    # First pair: prompt and widget
    label1 = QLabel(prompt1)
    label1.setStyleSheet("color: black; font-family: Calibri;")
    hlayout.addWidget(label1)
    # Override the interactive widget's style:
    widget1.native.setStyleSheet("background-color: #414851; color: white;")
    hlayout.addWidget(widget1.native)

    hlayout.addSpacing(10)

    # Second pair: prompt and widget
    label2 = QLabel(prompt2)
    label2.setStyleSheet("color: black; font-family: Calibri;")
    hlayout.addWidget(label2)
    widget2.native.setStyleSheet("background-color: #414851; color: white;")
    hlayout.addWidget(widget2.native)

    container.setLayout(hlayout)
    form.addRow(container)


# Helper: create a group container with an external title.
def create_group(title: str, form: QFormLayout, bg_color: str, border_color: str) -> QWidget:
    group_box = QGroupBox("")
    group_box.setLayout(form)
    group_box.setStyleSheet(f"""
        QGroupBox {{
            background-color: {bg_color};
            border: 2px solid {border_color};
            border-radius: 5px;
            padding: 5px;
        }}
        QGroupBox QLabel {{
            color: black;
            font-family: Calibri;
        }}
        QCheckBox {{
            color: black;
            font-family: Calibri;
        }}
    """)
    # Title label: white text, Calibri, 16px font, minimal margins.
    title_label = QLabel(title)
    title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white; font-family: Calibri; margin: 2px;")
    container_layout = QVBoxLayout() # qv means vertical layout
    container_layout.setContentsMargins(2, 2, 2, 2)
    container_layout.setSpacing(2)
    container_layout.addWidget(title_label)
    container_layout.addWidget(group_box)
    container = QWidget()
    container.setLayout(container_layout)
    return container

# Create Part 0: A settings row (light background, no title) with checkboxes and the top-level button.
def create_part0(read_button, mask_checkbox, revise_checkbox, ratio_checkbox):
    container = QWidget()
    hlayout = QHBoxLayout() # qh means horizontal layout
    hlayout.setContentsMargins(2, 2, 2, 2)
    # Add the "Read in all files" button first.
    hlayout.addWidget(read_button.native)
    hlayout.addSpacing(20)
    # Add the checkboxes.
    hlayout.addWidget(mask_checkbox.native)
    hlayout.addWidget(revise_checkbox.native)
    hlayout.addWidget(ratio_checkbox.native)
    container.setLayout(hlayout)
    container.setStyleSheet("background-color: #;")
    return container

class Trackrevise(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.plot_widget = PlotWidget()
        self.viewer.window.add_dock_widget(self.plot_widget, area='bottom', name='Signal')
        self.masks_history = []


        # -------------------------------
        # Part 0: Top-level row with "Read in all files" and checkboxes.
        # -------------------------------
        self.read_in_all_button = PushButton(text="Read in all")
        # light bg and black text
        self.read_in_all_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #F0F0F0;"  # light background
            "  color: black;"  #
            "  font-family: Calibri;"
            "}"
        )
        self.mask_256_checkbox = CheckBox(text="Masks>255")
        self.revise_mode_checkbox = CheckBox(text="Revise+Visualize Mode")
        self.revise_mode_checkbox.value = False
        self.ratio_checkbox = CheckBox(text="FRET for G/B")
        self.ratio_checkbox.value = True

        part0 = create_part0(self.read_in_all_button,
                             self.mask_256_checkbox,
                             self.revise_mode_checkbox,
                             self.ratio_checkbox)

        # -------------------------------
        # Group 1: Tracking Revision
        # -------------------------------
        self.mask_folder = FileEdit(label="Masks Folder", mode='d')
        # self.mask_folder.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\Track-1-11'
        self.mask_folder.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\Track_log_rainbow'
        # self.mask_folder.value = r'D:\PKU_STUDY\Branch\OtherOnes\ZhangXY\20250707_lowOBJ_podosome imaging_THP-1 M0_notFACS_Ac-NEMOs\KO_XY568' # zxy
        self.read_masks_button = PushButton(text="Read")
        # Now, mask folder row only has file input and read button.
        self.tif_input = FileEdit(label="TIF stack Input", mode='r', filter='*.tif')
        self.tif_input.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\stack-b_denoised-1-11.tif'
        self.read_tif_button = PushButton(text="Read")
        form1 = QFormLayout()
        add_form_row(form1, "Mask Folder", [self.mask_folder, self.read_masks_button])
        add_form_row(form1, "TIF stack Input", [self.tif_input, self.read_tif_button])
        # Also add action buttons row.
        self.apply_next_button = PushButton(text="Apply to Next Fr")
        self.apply_button = PushButton(text="Apply to Following Frs")
        self.save_tracking_button = PushButton(text="Save Track")
        add_form_row(form1, "If revised:", [self.apply_next_button, self.apply_button, self.save_tracking_button])
        group1 = create_group("Tracking Revision", form1, "#FFF9E6", "#FFC107")

        # -------------------------------
        # Group 2: Biosensor Channels
        # -------------------------------
        self.stack_b_input = FileEdit(label="Stack B Input", mode='r', filter='*.tif')
        self.stack_b_input.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\stack-b-1-11.tif'
        # self.stack_b_input.value = r'D:\PKU_STUDY\Branch\OtherOnes\ZhangXY\20250707_lowOBJ_podosome imaging_THP-1 M0_notFACS_Ac-NEMOs\KO_XY568\C488power50_Exp200ms_Int1s-20x_009.vsi - C488.tif'
        self.read_stack_b_button = PushButton(text="Read")
        self.stack_g_input = FileEdit(label="Stack G Input", mode='r', filter='*.tif')
        self.stack_g_input.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\stack-g-1-11.tif'
        self.read_stack_g_button = PushButton(text="Read")
        self.stack_nir_input = FileEdit(label="Stack NIR Input", mode='r', filter='*.tif')
        self.stack_nir_input.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\stack-b-1-11.tif'
        self.read_stack_nir_button = PushButton(text="Read")
        form2 = QFormLayout()
        add_form_row(form2, "Stack B Input", [self.stack_b_input, self.read_stack_b_button])
        add_form_row(form2, "Stack G Input", [self.stack_g_input, self.read_stack_g_button])
        add_form_row(form2, "Stack NIR Input", [self.stack_nir_input, self.read_stack_nir_button])
        group2 = create_group("Biosensor Channels", form2, "#F0FFF0", "#66BB6A")

        # -------------------------------
        # Group 2.5: Calibration, Shifting and Overexposure
        # -------------------------------
        ChannelChoice = Annotated[
            str,
            (
                ("widget_type", "ComboBox"),
                ("choices", ["Stack G", "Stack NIR", "Stack B", "Barcodes Masks classified"]),
            )
        ]
        self.channel_to_shift = create_widget(label="Channel to shift", annotation=ChannelChoice, value="Stack G")
        self.channel_to_shift.native.setStyleSheet(
            "QComboBox {"
            "  background-color: #414851;"  # dark background
            "  color: white;"  # white text
            "  font-family: Calibri;"
            "}"
        )
        shift_max = 20
        self.shift_r_param = create_widget(label="Right shift", widget_type="Slider", value=0)
        self.shift_r_param.native.setStyleSheet("""
            QSlider::handle {
                color: black;
            }
        """)
        self.shift_r_param.min = -shift_max
        self.shift_r_param.max = shift_max
        self.shift_r_param.step = 1
        self.shift_u_param = create_widget(label="Up shift", widget_type="Slider", value=0)
        self.shift_u_param.min = -shift_max
        self.shift_u_param.max = shift_max
        self.shift_u_param.step = 1
        self.shift_button = PushButton(text="Shift")
        self.shift_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #414851;"  # dark background 65 72 81 = #
            "  color: white;"  # white text
            "  font-family: Calibri;"
            "}"
        )
        self.shift_save_button = PushButton(text="Save Shift")
        self.shift_save_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #414851;"  # dark background
            "  color: white;"  # white text
            "  font-family: Calibri;"
            "}"
        )
        # Now, combine channel_to_shift and the two action buttons in one row.
        form25 = QFormLayout()
        # Custom row: no prompt for the buttons.
        container = QWidget()
        # make the bg as same as the group
        container.setStyleSheet("background-color: #E8F4FD;")
        hlayout = QHBoxLayout()
        # SET THE bg of hlayout to the same as the group
        # hlayout.setStyleSheet("background-color: #F0FFF0;") wrong
        hlayout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Channel to shift")
        label.setStyleSheet("color: black; font-family: Calibri;")
        hlayout.addWidget(label)
        hlayout.addWidget(self.channel_to_shift.native)
        hlayout.addSpacing(10)
        hlayout.addWidget(self.shift_button.native)
        hlayout.addWidget(self.shift_save_button.native)
        container.setLayout(hlayout)
        form25.addRow(container)
        # Next row: the two sliders in one row.
        add_double_row(form25, "Right shift", self.shift_r_param, "Up shift", self.shift_u_param, bg_color="#E8F4FD")
        # Next row: Overexposure threshold with its buttons.
        # add a spinbox for overexposure threshold, from 0-65535
        self.overexpo_thres_param = create_widget(label="Overexposure Threshold", widget_type="SpinBox", value=65535, options={'min': 0, 'max': 65535})
        self.overexpo_vis_button = PushButton(text="visualize")
        self.overexpo_discard_button = PushButton(text="Discard")
        add_form_row(form25, "Overexpo Threshold", [self.overexpo_thres_param, self.overexpo_vis_button, self.overexpo_discard_button])
        group25 = create_group("Multi-channel registration (Shifting in pixel) and Overexposure Check", form25, "#E8F4FD", "#42A5F5")

        # -------------------------------
        # Group 3: Barcodes Alignment
        # -------------------------------
        self.classification_input = FileEdit(label="Barcode Image", mode='r', filter='*.tif')
        self.classification_input.value = r'E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\intensity\100uM_NE-cls.tif'
        self.classification_resize = create_widget(label="Resize to", widget_type="SpinBox", value=1024,
                                                   options={'min': 512, 'max': 2048})
        self.classification_resize.native.setStyleSheet(
            "QSpinBox {"
            "  background-color: #F3E5F5;"  # light background
            "  color: black;"  # black text
            "  font-family: Calibri;"
            "}"
        )
        self.align_thres_percent = create_widget(label="Align Threshold", widget_type="SpinBox", value=10,
                                                 options={'min': 0, 'max': 100})
        self.align_mask_frame = create_widget(label="Align to Frame", widget_type="SpinBox", value=0,
                                              options={'min': 0, 'max': 2000})
        self.classification_align_button = PushButton(text="Align")
        form3 = QFormLayout()
        add_form_row(form3, "Barcode Image", [self.classification_input])
        add_double_row(form3, "Resize to", self.classification_resize, "Align Threshold", self.align_thres_percent, bg_color="#F3E5F5")
        add_form_row(form3, "Align to Frame", [self.align_mask_frame, self.classification_align_button])
        group3 = create_group("Barcodes-Biosensor Alignment", form3, "#F3E5F5", "#AB47BC")
        self.Bs2Code_save_path = None
        # -------------------------------
        # Group 4: Calculate Signals
        # -------------------------------
        self.ratio_calcu_range = create_widget(label="Ratio Calculation Range", widget_type="RangeSlider",
                                               value=[0, 100],
                                               options={'min': 0, 'max': 2000})
        # self.basal_frame_spinbox = create_widget(label="Basal Frame Number", widget_type="SpinBox", value=34,
        #                                          options={'min': 0, 'max': 2000})
        self.basal_frame_range = create_widget(label="Basal Frame Range", widget_type="RangeSlider",
                                              value=[40,80],
                                                options={'min': 0, 'max': 2000})

        self.ratio_calcu_button = PushButton(text="Calculate")
        form4 = QFormLayout()
        add_form_row(form4, "Ratio Calculation Range", [self.ratio_calcu_range])
        # add_form_row(form4, "Basal Frame Number", [self.basal_frame_spinbox, self.ratio_calcu_button])
        add_form_row(form4, "Basal Frame Range", [self.basal_frame_range, self.ratio_calcu_button])
        # add_double_row(form4, "Basal Frame Range", self.basal_frame_range, "", self.ratio_calcu_button, bg_color="#FFEEEE")
        group4 = create_group("Calculate Signals", form4, "#FFEEEE", "#FF8888")
        # -------------------------------
        self.freq_analysis_checkbox = CheckBox(text="Frequency Domain Analysis")
        self.freq_analysis_checkbox.value = False
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(self.freq_analysis_checkbox.native)
        freq_layout.addStretch(1)

        # -------------------------------
        # New: Percentile sliders for contrast limits
        # self.freq_clim_slider = create_widget(
        #     label="Freq CLIM (%)",
        #     widget_type="RangeSlider",
        #     value=[5, 95],
        #     options={'min': 0, 'max': 100}
        # )
        # # to get the value low, use self.freq_clim_slider.value[
        # self.phase_clim_slider = create_widget(
        #     label="Phase CLIM (%)",
        #     widget_type="RangeSlider",
        #     value=[5, 95],
        #     options={'min': 0, 'max': 100}
        # )
        # clim_layout = QFormLayout()
        # add_form_row(
        #     clim_layout,
        #     "Contrast Limits",
        #     [self.freq_clim_slider, self.phase_clim_slider]
        # )

        # -------------------------------
        # Arrange all parts vertically (one column)
        # -------------------------------
        main_vlayout = QVBoxLayout()
        main_vlayout.addWidget(part0)
        main_vlayout.addWidget(group1)
        main_vlayout.addWidget(group2)
        main_vlayout.addWidget(group25)
        main_vlayout.addWidget(group3)
        main_vlayout.addWidget(group4)
        main_vlayout.addLayout(freq_layout)
        # main_vlayout.addLayout(clim_layout)


        existing = self.native.layout()
        if existing is None:
            self.native.setLayout(main_vlayout)
        else:
            existing.addLayout(main_vlayout)

        # ---------------------------------------------------------
        # Connect callbacks (implement these methods in your class)
        # ---------------------------------------------------------
        self.read_in_all_button.clicked.connect(self.read_in_all)
        self.read_masks_button.clicked.connect(self.read_masks)
        self.read_tif_button.clicked.connect(lambda: self.read_tif('TIF for Tracking'))
        self.read_stack_b_button.clicked.connect(lambda: self.read_tif('Stack B'))
        self.read_stack_g_button.clicked.connect(lambda: self.read_tif('Stack G'))
        self.read_stack_nir_button.clicked.connect(lambda: self.read_tif('Stack NIR'))
        self.apply_button.clicked.connect(self.apply_to_following_frames)
        self.apply_next_button.clicked.connect(self.apply_to_next_frame)
        self.save_tracking_button.clicked.connect(self.save_tracking)
        self.shift_button.clicked.connect(self.shift_stack)
        self.shift_save_button.clicked.connect(self.save_shifted_stack)
        self.overexpo_vis_button.clicked.connect(self.overexpo_visualize)
        self.overexpo_discard_button.clicked.connect(self.overexpo_discard)
        self.revise_mode_checkbox.changed.connect(self.toggle_revise_mode)
        self.classification_input.changed.connect(self.load_classification)
        self.classification_resize.changed.connect(self.load_classification)
        self.classification_align_button.clicked.connect(self.align_classification)
        self.ratio_calcu_button.clicked.connect(self.calculate_signal_ratio)

        self.num_masks = 1000

    def frequency_domain_analysis(self, signal: np.ndarray, sampling_rate: float = 1.0):
        N = len(signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
        fft_vals = np.abs(np.fft.rfft(signal))
        return freqs, fft_vals
    def read_in_all(self):
        # call all the read functions
        self.read_masks()
        self.read_tif('TIF for Tracking')
        self.read_tif('Stack B')
        self.read_tif('Stack G')
        self.read_tif('Stack NIR')
        self.load_classification()

    def calculate_signal_ratio(self):

        # Check that the Bs2Code Excel file exists.
        if self.Bs2Code_save_path is None:
            self.Bs2Code_save_path = os.path.join(self.base_folder, 'Bs2Code.xlsx')

            notifications.show_warning("You have not run the Bs2Code alignment yet. Now check the existence of the excel.")
        if not os.path.exists(self.Bs2Code_save_path):
            # Ask user whether to proceed with default
            choice = self.show_warning_dialog(
                "No 'Bs2Code.xlsx' found.\n"
                "Continue with default alignment (all cells → class 1)?"
            )
            if choice != "continue":
                return
            # Build default alignment: every cell_id maps to class 1
            idx = np.arange(1, self.num_masks + 1)
            alignment_info = pd.DataFrame({'Class': 1}, index=idx)
        else:
            # Normal path: read the provided Excel
            df = pd.read_excel(self.Bs2Code_save_path)
            df = df[['Tracking Mask Index', 'Class']].set_index('Tracking Mask Index')
            alignment_info = df

        out_excel_path = os.path.join(self.base_folder, 'signal_analysis.xlsx')
        # If the Excel file already exists, ask user if they want to overwrite it.
        if os.path.exists(out_excel_path):
            response = self.show_warning_dialog("Excel file already exists. Do you want to overwrite it?")
            if response != "continue":
                return
            # Check writability: if the file is open elsewhere, attempting to open it will fail.
            try:
                with open(out_excel_path, 'a'):
                    pass
            except Exception as e:
                notifications.show_error(
                    "The Excel file is currently open or not writable. Please close it and try again.")
                return

        # Define frame range and baseline parameters.
        stack_start = self.ratio_calcu_range.value[0]
        stack_end = self.ratio_calcu_range.value[1] + 1
        # basal_frame_num = self.basal_frame_spinbox.value
        basal_frame_start = self.basal_frame_range.value[0]
        basal_frame_end = self.basal_frame_range.value[1]
        # Check which channels are available.
        has_stack_b = 'Stack B' in self.viewer.layers
        has_stack_g = 'Stack G' in self.viewer.layers
        has_stack_nir = 'Stack NIR' in self.viewer.layers

        if not (has_stack_b or has_stack_g or has_stack_nir):
            notifications.show_error("None of Stack B, G, or NIR found in layers!")
            return

        # Notify user about missing channels.
        if not has_stack_b:
            notifications.show_info("Stack B is missing, skipping Blue channel processing.")
        if not has_stack_g:
            notifications.show_info("Stack G is missing, skipping Green channel processing.")
        if not has_stack_nir:
            notifications.show_info("Stack NIR is missing, skipping NIR channel processing.")

        # Get masks and number of cells.
        all_masks = self.viewer.layers['Masks'].data[stack_start:stack_end]
        cell_num = len(np.unique(all_masks[0])) - 1
        max_cell_id = np.max(all_masks)
        notifications.show_info(f'Number of Cells: {cell_num} cells')

        # Load alignment info from Excel.
        # alignment_info = pd.read_excel(self.Bs2Code_save_path)
        # alignment_info = alignment_info[['Tracking Mask Index', 'Class']]
        # alignment_info = alignment_info.set_index('Tracking Mask Index')

        # Function to extract intensity per cell for each frame.
        # def extract_intensity(stack, all_masks, cell_num, mode='sum'):
        #     intensity_data = []
        #     for frame_number, frame in tqdm(enumerate(stack), total=len(stack), desc="Extracting intensities"):
        #         current_masks = all_masks[frame_number]
        #         for cell_id in range(1, cell_num + 1):
        #             cell_mask = (current_masks == cell_id)
        #             if np.sum(cell_mask) == 0:
        #                 continue
        #             if mode == 'sum':
        #                 intensity = np.sum(frame[cell_mask])
        #             elif mode == 'mean':
        #                 intensity = np.mean(frame[cell_mask])
        #             else:
        #                 continue
        #             if intensity == 0:
        #                 notifications.show_warning(f'Frame: {frame_number}, Cell: {cell_id}, intensity = 0!')
        #             intensity_data.append({
        #                 "frame": frame_number,
        #                 "cell_id": cell_id,
        #                 "intensity": intensity
        #             })
        #     return intensity_data
        # def extract_intensity(stack, all_masks, max_id, mode='sum'):
        #     records = []
        #     for t in tqdm(range(stack.shape[0]), desc="GPU Extract"):
        #         frame_gpu = cp.asarray(stack[t])
        #         labels_gpu = cp.asarray(all_masks[t])
        #         # counts and sums 同时算
        #         counts = cp.bincount(labels_gpu.ravel(), minlength=max_id + 1)
        #         weighted = cp.bincount(labels_gpu.ravel(), weights=frame_gpu.ravel(), minlength=max_id + 1)
        #         if mode == 'mean':
        #             weighted = weighted / counts
        #         # 拷回 CPU
        #         weighted = cp.asnumpy(weighted)
        #         for cid in range(1, max_id + 1):
        #             val = weighted[cid]
        #             if val == 0:
        #                 notifications.show_warning(f'Frame {t}, Cell {cid} intensity = 0')
        #             records.append({"frame": t, "cell_id": cid, "intensity": float(val)})
        #     return records

        def extract_intensity(stack, all_masks, cell_num, mode='sum'):
            """
            stack:      numpy array of shape (T, H, W)
            all_masks:  numpy array of same shape, integer labels per pixel
            cell_num:   maximum label ID (cells numbered 1..cell_num)
            mode:       'sum' or 'mean'
            """
            print(f'stack shape: {stack.shape}, all_masks shape: {all_masks.shape}, cell_num: {cell_num}, mode: {mode}')
            intensity_data = []
            cell_ids = list(range(1, cell_num + 1))

            for frame_number, frame in tqdm(
                    enumerate(stack), total=stack.shape[0], desc="Extracting intensities"
            ):
                labels = all_masks[frame_number]

                # use C-accelerated ndimage routines
                if mode == 'sum':
                    vals = ndi_sum(frame, labels=labels, index=cell_ids)
                else:  # 'mean'
                    vals = ndi_mean(frame, labels=labels, index=cell_ids)

                # pack results into the same format as before
                for cid, inten in zip(cell_ids, vals):
                    if inten == 0:
                        notifications.show_warning(
                            f'Frame {frame_number}, Cell {cid}, intensity = 0!'
                        )
                    intensity_data.append({
                        "frame": frame_number,
                        "cell_id": int(cid),
                        "intensity": float(inten)
                    })

            return intensity_data

        # Helper to compute summary statistics per class for a normalized pivot table.
        def compute_summary_stats(norm_df):
            # For each cell, compute the overall mean (across frames, excluding the 'Class' column).
            cell_means = norm_df.drop(columns=['Class']).mean(axis=1)
            stats = cell_means.groupby(norm_df['Class']).agg(['mean', 'std', 'count'])
            stats = stats.rename(columns={'mean': 'Average', 'std': 'Std', 'count': 'Cell Count'})
            stats['SE'] = stats['Std'] / np.sqrt(stats['Cell Count'])
            return stats

        # Dictionary to hold the normalized data for each channel.
        pivot_data = {}

        with pd.ExcelWriter(out_excel_path) as writer:
            # if stack_end < stack length, pop out warning
            if stack_end < self.viewer.layers['Masks'].data.shape[0]:
                notifications.show_warning(f'using only frames {stack_start} to {stack_end - 1} of the Masks stack, '
                                           f'which has {self.viewer.layers["Masks"].data.shape[0]} frames.')
            # Process Blue channel.
            if has_stack_b:
                blue_channel = self.viewer.layers['Stack B'].data[stack_start:stack_end]
                notifications.show_info("Extracting blue intensity...")
                blue_intensity = extract_intensity(blue_channel, all_masks, max_cell_id, mode='sum')
                blue_df = pd.DataFrame(blue_intensity)
                blue_pivot = blue_df.pivot(index='cell_id', columns='frame', values='intensity')
                blue_pivot = self.add_class_info(blue_pivot, alignment_info)
                blue_pivot.sort_values(by='Class').to_excel(writer, sheet_name='Blue Channel (Original)')

                baseline_blue = blue_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_blue = blue_pivot.div(baseline_blue, axis=0)
                normalized_blue = self.add_class_info(normalized_blue, alignment_info)
                normalized_blue.sort_values(by='Class').to_excel(writer, sheet_name='Blue Channel (Normalized)')

                pivot_data['B'] = normalized_blue
                stats_blue = compute_summary_stats(normalized_blue)
                stats_blue.to_excel(writer, sheet_name='Statistics - Blue')

                # —— only if frequency_analysis_checkbox is checked ——
                if self.freq_analysis_checkbox.value and has_stack_b:
                    sampling_rate = 1.0  # Hz

                    freq_results = []
                    phase_results = []
                    # compute per-cell peak frequency, phase, and weighted frequency
                    for cell_id, row in normalized_blue.iterrows():
                        signal = row.values.astype(float)
                        fft_vals = np.fft.rfft(signal)
                        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
                        spectrum = np.abs(fft_vals)

                        # skip DC and very low freqs (bins 0,1,2)
                        spectrum_nodc = spectrum[1:]
                        freqs_nodc = freqs[1:]

                        # peak: find index in the sliced arrays, then offset by +1
                        peak_rel_idx = np.argmax(spectrum_nodc[2:]) + 2
                        peak_idx = peak_rel_idx + 1
                        peak_freq = freqs[peak_idx]
                        peak_power = spectrum[peak_idx]

                        # phase at peak
                        raw_phase = np.angle(fft_vals[peak_idx])  # [-π, π]
                        phase = (raw_phase + 2 * np.pi) % (2 * np.pi)

                        # weighted (centroid) frequency excluding DC
                        weighted_freq = np.sum(freqs_nodc * spectrum_nodc) / np.sum(spectrum_nodc)

                        freq_results.append({
                            'cell_id': int(cell_id),
                            'peak_frequency (Hz)': float(peak_freq),
                            'peak_power': float(peak_power),
                            'weighted_frequency (Hz)': float(weighted_freq)
                        })
                        phase_results.append({
                            'cell_id': int(cell_id),
                            'phase (rad)': phase
                        })

                    # write tables to Excel
                    freq_df = pd.DataFrame(freq_results)
                    phase_df = pd.DataFrame(phase_results)
                    freq_df.to_excel(writer, sheet_name='Frequency Analysis', index=False)
                    phase_df.to_excel(writer, sheet_name='Phase Analysis', index=False)

                    # build two maps from the first mask frame
                    mask0 = self.viewer.layers['Masks'].data[stack_start]
                    freq_map = np.zeros_like(mask0, dtype=float)
                    freq_dom_map = np.zeros_like(mask0, dtype=float)
                    phase_map = np.zeros_like(mask0, dtype=float)

                    for row in freq_results:
                        cid = row['cell_id']
                        freq_map[mask0 == cid] = row['weighted_frequency (Hz)']  # use weighted freq on map
                    for row in freq_results:
                        cid = row['cell_id']
                        freq_dom_map[mask0 == cid] = row['peak_frequency (Hz)'] # use peak freq on map for dominant frequency
                    for row in phase_results:
                        cid = row['cell_id']
                        phase_map[mask0 == cid] = row['phase (rad)']

                    # add both layers with full-range turbo colormap
                    self.viewer.add_image(
                        freq_map,
                        name='Freq Map',
                        colormap='turbo'
                    )
                    self.viewer.add_image(
                        freq_dom_map,
                        name='Freq Dom Map',
                        colormap='turbo'
                    )
                    self.viewer.add_image(
                        phase_map,
                        name='Phase Map',
                        colormap='turbo'
                    )

            # Process Green channel.
            if has_stack_g:
                green_channel = self.viewer.layers['Stack G'].data[stack_start:stack_end]
                notifications.show_info("Extracting green intensity...")
                green_intensity = extract_intensity(green_channel, all_masks, max_cell_id, mode='sum')
                green_df = pd.DataFrame(green_intensity)
                green_pivot = green_df.pivot(index='cell_id', columns='frame', values='intensity')
                green_pivot = self.add_class_info(green_pivot, alignment_info)
                green_pivot.sort_values(by='Class').to_excel(writer, sheet_name='Green Channel (Original)')
                # baseline_green = green_pivot.iloc[:, :basal_frame_num].mean(axis=1)
                baseline_green = green_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_green = green_pivot.div(baseline_green, axis=0)
                normalized_green = self.add_class_info(normalized_green, alignment_info)
                normalized_green.sort_values(by='Class').to_excel(writer, sheet_name='Green Channel (Normalized)')
                pivot_data['G'] = normalized_green
                stats_green = compute_summary_stats(normalized_green)
                stats_green.to_excel(writer, sheet_name='Statistics - Green')

            # Process NIR channel.
            if has_stack_nir:
                nir_channel = self.viewer.layers['Stack NIR'].data[stack_start:stack_end]
                notifications.show_info("Extracting NIR intensity...")
                nir_intensity = extract_intensity(nir_channel, all_masks, max_cell_id, mode='sum')
                nir_df = pd.DataFrame(nir_intensity)
                nir_pivot = nir_df.pivot(index='cell_id', columns='frame', values='intensity')
                nir_pivot = self.add_class_info(nir_pivot, alignment_info)
                nir_pivot.sort_values(by='Class').to_excel(writer, sheet_name='NIR Channel (Original)')
                # baseline_nir = nir_pivot.iloc[:, :basal_frame_num].mean(axis=1)
                baseline_nir = nir_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_nir = nir_pivot.div(baseline_nir, axis=0)
                normalized_nir = self.add_class_info(normalized_nir, alignment_info)
                normalized_nir.sort_values(by='Class').to_excel(writer, sheet_name='NIR Channel (Normalized)')
                pivot_data['NIR'] = normalized_nir
                stats_nir = compute_summary_stats(normalized_nir)
                stats_nir.to_excel(writer, sheet_name='Statistics - NIR')

            # Process G/B Ratio if both Blue and Green channels exist and the ratio checkbox is checked.
            if has_stack_b and has_stack_g and self.ratio_checkbox.value:
                ratio_pivot = pivot_data['G'] / pivot_data['B']
                # baseline_ratio = ratio_pivot.iloc[:, :basal_frame_num].mean(axis=1)
                baseline_ratio = ratio_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_ratio = ratio_pivot.div(baseline_ratio, axis=0)
                normalized_ratio = self.add_class_info(normalized_ratio, alignment_info)
                normalized_ratio.sort_values(by='Class').to_excel(writer, sheet_name='Normalized G-B')
                pivot_data['Ratio'] = normalized_ratio
                stats_ratio = compute_summary_stats(normalized_ratio)
                stats_ratio.to_excel(writer, sheet_name='Statistics - G-B')

        notifications.show_info(f'Saved signal analysis results to {out_excel_path}')

        # Decide which channels to plot.
        if 'Ratio' in pivot_data:
            channels_to_plot = {'Ratio': pivot_data['Ratio']}
            if 'NIR' in pivot_data:
                channels_to_plot['NIR'] = pivot_data['NIR']
            overall_plot_title = "G/B Ratio" + (" + NIR" if 'NIR' in pivot_data else "")
        else:
            channels_to_plot = {}
            if 'B' in pivot_data:
                channels_to_plot['Blue'] = pivot_data['B']
            if 'G' in pivot_data:
                channels_to_plot['Green'] = pivot_data['G']
            if 'NIR' in pivot_data:
                channels_to_plot['NIR'] = pivot_data['NIR']
            overall_plot_title = "Channels: " + ", ".join(channels_to_plot.keys())

        # Get the list of classes from one of the pivot tables.
        sample_df = next(iter(pivot_data.values()))
        classes = sorted(sample_df['Class'].unique())
        num_classes = len(classes)

        import math
        ncols = 4  # 4 plots per row
        nrows = math.ceil(num_classes / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1:
            axs = np.array(axs).reshape(1, -1)
        axs = axs.flatten()

        # Plot each class in a separate subplot.
        # for ax, cls in zip(axs, tqdm(classes, desc="Plotting by class")):
        for ax, cls in zip(axs, classes):
            for ch_name, df in channels_to_plot.items():
                class_data = df[df['Class'] == cls]
                # Remove the 'Class' column.
                numeric_data = class_data.drop(columns=['Class'])
                if numeric_data.empty:
                    continue
                mean_curve = numeric_data.mean(axis=0)
                std_curve = numeric_data.std(axis=0)
                n = numeric_data.shape[0]
                # Compute standard error (SE) instead of SD.
                se_curve = std_curve / np.sqrt(n) if n > 0 else std_curve
                frames = mean_curve.index.astype(float)  # assuming frame numbers are numeric
                # Choose color based on channel.
                if ch_name in ['Blue', 'B']:
                    color = 'blue'
                elif ch_name in ['Green', 'G']:
                    color = 'green'
                elif ch_name == 'NIR':
                    color = 'red'
                elif ch_name == 'Ratio':
                    color = 'green'
                else:
                    color = None
                ax.plot(frames, mean_curve, label=ch_name, color=color)
                ax.fill_between(frames, mean_curve - se_curve, mean_curve + se_curve, color=color, alpha=0.3)
            num_cells = len(class_data)
            ax.set_title(f'Class {cls} (n={num_cells})')
            ax.set_xlabel("Frame")
            ax.set_ylabel("Normalized Intensity")
            ax.legend()

        # Remove extra subplots if there are more axes than classes.
        for i in range(num_classes, len(axs)):
            fig.delaxes(axs[i])
        plt.suptitle(overall_plot_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def show_warning_dialog(self, message):
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        # Using askquestion to get a Yes/No/Cancel response.
        response = messagebox.askquestion("Warning", message + "\n\nDo you want to continue?", icon='warning',
                                          type='yesnocancel')
        if response == 'yes':
            return "continue"
        elif response == 'no':
            return "add"
        else:
            return "cancel"

    def add_class_info(self, df, alignment_info):
        df['Class'] = df.index.map(lambda x: alignment_info.loc[x, 'Class'] if x in alignment_info.index else 0)
        cols = df.columns.tolist()
        cols = ['Class'] + [col for col in cols if col != 'Class']
        return df[cols]

    def show_warning_dialog(self, message):
        root = tk.Tk()
        root.withdraw()
        response = messagebox.askquestion("Warning", message + "\n\nDo you want to continue?",
                                          icon='warning', type='yesnocancel')
        if response == 'yes':
            return "continue"
        elif response == 'no':
            return "add"
        else:
            return "cancel"
    def load_classification(self):
        # load the tif file which is masks from 0-14, 0 for bg, 1-14 for cells in different classes
        classification_file = self.classification_input.value
        resize = self.classification_resize.value
        # if classification_file:
        if classification_file and os.path.exists(classification_file):
            masks = tiff.imread(classification_file)
            # resize masks from 2048 2048 to 1024 1024
            masks = cv2.resize(masks, (resize, resize), interpolation=cv2.INTER_NEAREST)
            self.viewer.add_labels(masks, name="Barcodes Masks classified", opacity=0.5)
            notifications.show_info(f'Loaded classification {np.max(masks)} classes, resized to {resize}x{resize}. Now you can align the classification masks to the tracking masks.')

    def align_classification(self):
        align_mask_frame = self.align_mask_frame.value
        # if none of those exist, show warning and return
        if 'Masks' not in self.viewer.layers or 'Barcodes Masks classified' not in self.viewer.layers:
            notifications.show_warning("Please load both 'Masks' and 'Barcodes Masks classified' layers before aligning.")
            return
        tracking_masks = self.viewer.layers['Masks'].data
        cls_masks = self.viewer.layers['Barcodes Masks classified'].data
        print('Shape of cls_masks:', cls_masks.shape)

        # Label all regions in the classification masks to separate them
        cls_masks_labeled = label(cls_masks)

        # Create a mapping from labeled regions to original indices
        label_to_index = {}
        for region in regionprops(cls_masks_labeled):
            label_to_index[region.label] = cls_masks[region.coords[0][0], region.coords[0][1]]

        # Extract properties of the masks in the specified frame
        frame_props = regionprops(tracking_masks[align_mask_frame])

        # Create an array to store the new indices of the masks in tracking_masks
        cls_masks_aligned = np.zeros_like(tracking_masks)

        # Initialize a list to store alignment information for Excel
        alignment_info = []

        # Define the threshold level for considering a class in a mask
        threshold_level = self.align_thres_percent.value / 100

        # Loop over each mask in the tracking frame, use tqdm
        # for frame_mask_id, frame_prop in enumerate(frame_props, start=1):

        for frame_prop in tqdm(frame_props, desc='Aligning Classification Masks'):
            frame_mask_id = frame_prop.label
            current_mask = tracking_masks[align_mask_frame] == frame_mask_id
            overlapping_classes, counts = np.unique(cls_masks[current_mask], return_counts=True)

            # Remove the background class (0) if it exists
            if 0 in overlapping_classes:
                zero_index = np.where(overlapping_classes == 0)
                overlapping_classes = np.delete(overlapping_classes, zero_index)
                counts = np.delete(counts, zero_index)

            if len(overlapping_classes) == 0:
                nearest_cls_id = 0  # No overlapping class
            else:
                total_pixels = np.sum(current_mask)
                class_percentages = counts / total_pixels

                # Find classes that meet the threshold level
                valid_classes = overlapping_classes[class_percentages >= threshold_level]

                if len(valid_classes) == 0:
                    nearest_cls_id = 0  # No class meets the threshold
                else:
                    # Determine the class with the highest count (majority)
                    majority_class_index = np.argmax(class_percentages)
                    nearest_cls_id = overlapping_classes[majority_class_index]

            # Save alignment info
            alignment_info.append((frame_mask_id, nearest_cls_id))

            # Apply the nearest class id to all frames
            for frame in range(len(tracking_masks)):
                cls_masks_aligned[frame][tracking_masks[frame] == frame_mask_id] = nearest_cls_id

        # Save alignment info to Excel
        self.save_alignment_info(alignment_info)

        self.viewer.add_labels(cls_masks_aligned, name="Tracking masks cls", opacity=0.5)
        notifications.show_info(f'Classification masks aligned to tracking masks. Alignment information saved to {self.Bs2Code_save_path}. Label layer "Tracking masks cls" added.')
    def save_alignment_info(self, alignment_info):
        df = pd.DataFrame(alignment_info, columns=['Tracking Mask Index', 'Class'])
        self.Bs2Code_save_path = os.path.join(self.base_folder, 'Bs2Code.xlsx')
        df.to_excel(self.Bs2Code_save_path, index=False)
        print(f'Saved alignment information to {self.Bs2Code_save_path}')
        notifications.show_info(f'Saved alignment information to {self.Bs2Code_save_path}')

    def save_shifted_stack(self):
        folder = QFileDialog.getExistingDirectory(caption="Select Folder to Save Shifted Stack")
        if folder:
            channel_name = self.channel_to_shift.value
            if channel_name not in self.viewer.layers:
                notifications.show_warning(f"{channel_name} layer not found.")
                return
            stack = self.viewer.layers[channel_name].data
            # Construct a filename based on the channel name.
            filename = f"stack-{channel_name.lower().replace(' ', '-')}.tif"
            tiff.imsave(os.path.join(folder, filename), stack)
            print(f"Shifted stack saved to {folder}")
            notifications.show_info(f"Shifted stack saved to {folder}")

    def overexpo_visualize(self):
        overexpo_thres = self.overexpo_thres_param.value
        stack_b = self.viewer.layers['Stack B'].data
        stack_g = self.viewer.layers['Stack G'].data

        overexpo_mask_b = stack_b >= overexpo_thres
        overexpo_mask_g = stack_g >= overexpo_thres

        # If NIR exists, get its data and compute the overexposure mask.
        if 'Stack NIR' in self.viewer.layers:
            stack_nir = self.viewer.layers['Stack NIR'].data
            overexpo_mask_nir = stack_nir >= overexpo_thres

        # Add the overexposure masks as labels.
        self.viewer.add_labels(overexpo_mask_b, name="Overexposed Pixels B")
        self.viewer.add_labels(overexpo_mask_g, name="Overexposed Pixels G")
        if 'Stack NIR' in self.viewer.layers:
            self.viewer.add_labels(overexpo_mask_nir, name="Overexposed Pixels NIR")

        masks = self.viewer.layers['Masks'].data

        # If the masks layer is 2D (applied to all frames), replicate it for each frame.
        if masks.ndim == 2:
            masks = np.stack([masks] * overexpo_mask_b.shape[0], axis=0)

        # Prepare arrays to hold the overexposed mask labels.
        masks_with_overexpo_b = np.zeros_like(masks)
        masks_with_overexpo_g = np.zeros_like(masks)
        if 'Stack NIR' in self.viewer.layers:
            masks_with_overexpo_nir = np.zeros_like(masks)

        # Loop through each frame and each cell within the mask.
        for i in tqdm(range(masks.shape[0]), desc='Classifying Overexposed Pixels'):
            for j in range(1, masks[i].max() + 1):
                current_mask = masks[i]
                # Create a binary mask for the current cell.
                current_mask = np.where(current_mask == j, current_mask, 0)

                if np.any(current_mask[overexpo_mask_b[i][:]]):
                    if self.mask_256_checkbox.value:
                        masks_with_overexpo_b[i] += current_mask.astype(np.uint16) * j
                    else:
                        masks_with_overexpo_b[i] += current_mask.astype(np.uint8) * j

                if np.any(current_mask[overexpo_mask_g[i][:]]):
                    if self.mask_256_checkbox.value:
                        masks_with_overexpo_g[i] += current_mask.astype(np.uint16) * j
                    else:
                        masks_with_overexpo_g[i] += current_mask.astype(np.uint8) * j

                if 'Stack NIR' in self.viewer.layers:
                    if np.any(current_mask[overexpo_mask_nir[i][:]]):
                        if self.mask_256_checkbox.value:
                            masks_with_overexpo_nir[i] += current_mask.astype(np.uint16) * j
                        else:
                            masks_with_overexpo_nir[i] += current_mask.astype(np.uint8) * j

        # Add the overexposed masks as separate layers.
        self.viewer.add_labels(masks_with_overexpo_b, name="Masks with Overexposed pixels B")
        self.viewer.add_labels(masks_with_overexpo_g, name="Masks with Overexposed pixels G")
        if 'Stack NIR' in self.viewer.layers:
            self.viewer.add_labels(masks_with_overexpo_nir, name="Masks with Overexposed pixels NIR")

        notifications.show_info(
            'Overexposed pixels visualized in layers "Overexposed Pixels B", "Overexposed Pixels G"' +
            (', "Overexposed Pixels NIR"' if 'Stack NIR' in self.viewer.layers else '') +
            ', and overexposed masks visualized in layers "Masks with Overexposed pixels ..."'
        )

    def overexpo_discard(self):
        # Retrieve the current masks and the overexposed mask layers for blue and green.
        masks = self.viewer.layers['Masks'].data
        masks_with_overexpo_b = self.viewer.layers['Masks with Overexposed pixels B'].data
        masks_with_overexpo_g = self.viewer.layers['Masks with Overexposed pixels G'].data

        binary_overexpo_b = masks_with_overexpo_b > 0
        binary_overexpo_g = masks_with_overexpo_g > 0

        # Set all pixels in the masks corresponding to overexposed regions (B and G) to 0.
        masks[binary_overexpo_b] = 0
        masks[binary_overexpo_g] = 0

        # If the NIR overexposed layer exists, process it similarly.
        if 'Masks with Overexposed pixels NIR' in self.viewer.layers:
            masks_with_overexpo_nir = self.viewer.layers['Masks with Overexposed pixels NIR'].data
            binary_overexpo_nir = masks_with_overexpo_nir > 0
            masks[binary_overexpo_nir] = 0

        # Update the masks layer with the discarded pixels.
        self.viewer.layers['Masks'].data = masks
        self.viewer.layers['Masks'].refresh()
        notifications.show_info('Overexposed masks discarded from the masks layer')

    def read_masks(self):
        # Set base_folder to the parent directory of the mask folder
        self.base_folder = os.path.dirname(self.mask_folder.value)

        # Find all .npy mask files
        mask_files = sorted(f for f in os.listdir(self.mask_folder.value) if f.endswith('.npy'))
        if not mask_files:
            notifications.show_error("No .npy files found in the mask folder!")
            return

        # Update the ratio calculation range to match mask count
        num_files = len(mask_files)
        self.ratio_calcu_range.value = [0, num_files - 1]

        single_file = (num_files == 1)
        all_masks = []

        for fname in mask_files:
            path = os.path.join(self.mask_folder.value, fname)
            try:
                # Try loading as a regular NumPy array
                data = np.load(path)
                if data.dtype != object:
                    masks = data.astype(np.uint16 if self.mask_256_checkbox.value else np.uint8)
                else:
                    # If dtype is object, treat as failure to trigger the Cellpose branch
                    raise ValueError
            except Exception:
                # Attempt to load as Cellpose output
                notifications.show_info(f"Failed to load {fname} as a standard array; trying Cellpose format...")
                try:
                    cell = np.load(path, allow_pickle=True).item()
                    masks = cell['masks'].astype(np.uint16 if self.mask_256_checkbox.value else np.uint8)
                except Exception:
                    notifications.show_error(f"Could not find 'masks' in {fname}; load failed!")
                    return

            all_masks.append(masks)

        # If only one mask file is present, duplicate it to match a multi‐frame image layer
        if single_file:
            n_frames = None
            for layer in self.viewer.layers:
                arr = getattr(layer, 'data', None)
                if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                    n_frames = arr.shape[0]
                    break
            if n_frames is None:
                notifications.show_warning(
                    "Only one mask file detected, but no multi‐frame image layer found. Loading single frame only."
                )
            else:
                notifications.show_info(f"Only one mask file detected; duplicating it for {n_frames} frames.")
                all_masks = [all_masks[0]] * n_frames

        # Stack into a (frames, height, width) array
        all_masks = np.stack(all_masks, axis=0)
        print(f"All masks shape: {all_masks.shape}")
        self.num_masks = all_masks.shape[0]
        # Add as a Labels layer in Napari
        self.viewer.add_labels(all_masks, name="Masks")
        notifications.show_info(
            f"{self.num_masks} mask frames loaded. "
            f"Maximum label in first frame: {np.max(all_masks[0])}"
        )

    def read_tif(self, name: str):
        if name == 'TIF for Tracking':
            tif_stack = tiff.imread(self.tif_input.value)
        elif name == 'Stack B':
            try:
                tif_stack = tiff.imread(self.stack_b_input.value)
                notifications.show_info(f'{name} read in successfully')
            except Exception as e:
                print(e)
                notifications.show_error("Error reading Stack B. Consider there's no Stack B needed.")
        elif name == 'Stack G':
            try:
                tif_stack = tiff.imread(self.stack_g_input.value)
                notifications.show_info(f'{name} read in successfully')
            except Exception as e:
                print(e)
                notifications.show_error("Error reading Stack G. Consider there's no Stack G needed.")
        elif name == 'Stack NIR':
            try:
                tif_stack = tiff.imread(self.stack_nir_input.value)
                notifications.show_info(f'{name} read in successfully')
            except Exception as e:
                print(e)
                notifications.show_error("Error reading Stack NIR. Consider there's no Stack NIR needed.")
        self.viewer.add_image(tif_stack, name=name)

    def save_tracking(self):
        # Open a dialog for the user to choose the folder to save the masks
        folder = QFileDialog.getExistingDirectory(caption='Select Folder to Save Masks')

        if folder:  # If a folder was chosen
            # for i, mask in enumerate(self.masks_layer.data):
            for i, mask in enumerate(self.viewer.layers['Masks'].data):
                # Save each mask to a .npy file in the chosen folder with the name 00000.npy, 00001.npy, etc.
                if self.mask_256_checkbox.value:
                    np.save(os.path.join(folder, f'{i:05d}.npy'), mask.astype(np.uint16))
                else:
                    np.save(os.path.join(folder, f'{i:05d}.npy'), mask.astype(np.uint8))
            print(f"Tracking saved to {folder}")
            notifications.show_info(f"Tweaked tracking results saved to {folder}")

    def shift_stack(self):
        shift_r = self.shift_r_param.value
        shift_u = -self.shift_u_param.value  # Invert to match 'up' direction.

        # Get the channel selected from the dropdown.
        channel_name = self.channel_to_shift.value
        if channel_name not in self.viewer.layers:
            notifications.show_warning(f"{channel_name} layer not found.")
            return

        active_layer = self.viewer.layers[channel_name]
        stack_shift = active_layer.data

        # Perform the shifting.
        if stack_shift.ndim == 3:
            stack_shift = np.roll(stack_shift, shift_r, axis=2)  # Roll right/left on axis 2.
            stack_shift = np.roll(stack_shift, shift_u, axis=1)  # Roll up/down on axis 1.
        elif stack_shift.ndim == 2:
            stack_shift = np.roll(stack_shift, shift_r, axis=1)
            stack_shift = np.roll(stack_shift, shift_u, axis=0)

        active_layer.data = stack_shift
        active_layer.refresh()

        notifications.show_info(
            f"Shifted {active_layer.name} by {shift_r} pixels to the right and {-shift_u} pixels up")

    def toggle_revise_mode(self):
        undo_key = 'u'
        # make brush size to be 1
        self.viewer.layers['Masks'].brush_size = 1
        self.masks_layer = self.viewer.layers['Masks']
        if self.revise_mode_checkbox.value:
            self.masks_layer.mouse_drag_callbacks.append(self.on_click)
            self.viewer.bind_key(undo_key, self.on_undo, overwrite=True)
            notifications.show_info('Revise Mode enabled. Press Control + Click to delete masks from this frame on, Ctrl + Alt + Click to delete current mask. Press Shift + Click to plot signal cha   nges in the mask. Press u to undo. Do all the operations in the Masks layer.')

        else:
            self.masks_layer.mouse_drag_callbacks.remove(self.on_click)
            # self.viewer.unbind_key(undo_key, None)

    def on_click(self, layer, event):
        # self.masks_layer = layer
        if event.modifiers and 'Control' in event.modifiers:  # Control 按下
            history_keep = 2  # keep certain number of history, in case memory is not enough
            position = tuple(map(int, event.position))
            label = self.masks_layer.data[position]
            current_frame = self.viewer.dims.current_step[0]

            if len(self.masks_history) > history_keep:
                self.masks_history.pop(0)
            self.masks_history.append(self.masks_layer.data.copy())

            if 'Alt' in event.modifiers:
                # Ctrl+Alt+Click → delete the mask in current frame
                self.masks_layer.data[current_frame] = np.where(
                    self.masks_layer.data[current_frame] == label,
                    0,
                    self.masks_layer.data[current_frame]
                )
                notifications.show_info(f'Deleted mask {label} in frame {current_frame} only')
            else:
                # Ctrl+Click → delete the mask in all following frames
                self.masks_layer.data[current_frame:] = np.where(
                    self.masks_layer.data[current_frame:] == label,
                    0,
                    self.masks_layer.data[current_frame:]
                )
                notifications.show_info(f'Deleted mask {label} from frame {current_frame} to the end')

            self.masks_layer.refresh()
            self.masks_layer.selected_label = label  # Set the active label to the deleted label

        elif event.modifiers and event.modifiers[0] == 'Shift':
            # get the mask label and mask positions
            label = self.masks_layer.data[tuple(map(int, event.position))]
            mask_positions = self.masks_layer.data == label

            # Collect available stacks
            available_stacks = {}
            if 'Stack B' in self.viewer.layers:
                available_stacks['B'] = self.viewer.layers['Stack B'].data
            if 'Stack G' in self.viewer.layers:
                available_stacks['G'] = self.viewer.layers['Stack G'].data
            if 'Stack NIR' in self.viewer.layers:
                available_stacks['NIR'] = self.viewer.layers['Stack NIR'].data

            # If none of the stacks exist, notify and exit.
            if not available_stacks:
                notifications.show_info('No Stack B, G, or NIR layer available for plotting.')
                return

            # Use one available stack to ensure the mask is 3D.
            some_stack = next(iter(available_stacks.values()))
            if mask_positions.ndim == 2:
                mask_positions = np.repeat(mask_positions[np.newaxis, :, :], some_stack.shape[0], axis=0)

            # Compute the summed signals for each available channel.
            sum_signals = {}
            for key, stack in available_stacks.items():
                sum_signal = np.zeros(stack.shape[0])
                for i in range(stack.shape[0]):
                    sum_signal[i] = np.sum(stack[i][mask_positions[i]])
                sum_signals[key] = sum_signal

            # Prepare a list of signals and labels to plot.
            signals_to_plot = []
            labels_to_plot = []
            if 'B' in sum_signals:
                signals_to_plot.append(sum_signals['B'])
                labels_to_plot.append("Stack B")
            if 'G' in sum_signals:
                signals_to_plot.append(sum_signals['G'])
                labels_to_plot.append("Stack G")
            if 'NIR' in sum_signals:
                signals_to_plot.append(sum_signals['NIR'])
                labels_to_plot.append("Stack NIR")
            if self.freq_analysis_checkbox.value and 'B' in sum_signals:
                freqs = np.fft.rfftfreq(len(sum_signals['B']), d=1)  # Assuming unit time step

                spectrum = np.abs(np.fft.rfft(sum_signals['B']))
                # make DC to 0
                spectrum[0] = 0
                spectrum[1] = 0
                spectrum[0] = 0
                signals_to_plot.append(spectrum)
                bin_width = freqs[1] - freqs[0]  # Frequency bin width
                peak_freq = freqs[np.argmax(spectrum[3:])] + 3 * bin_width  # Skip DC and first bin
                weighted_freq = np.sum(freqs[1:] * spectrum[1:]) / np.sum(spectrum[1:])
                title_str = f"FFT of B Signal (Δf = {bin_width:.4f} Hz per bin), Peak Freq = {peak_freq:.2f} Hz, Weighted Freq = {weighted_freq:.2f} Hz"
                labels_to_plot.append(title_str)


            # Only compute the ratio if both B and G are available and the checkbox is checked.
            if self.ratio_checkbox.value and ('B' in sum_signals and 'G' in sum_signals):
                ratio_gb = sum_signals['G'] / sum_signals['B']
                signals_to_plot.append(ratio_gb)
                labels_to_plot.append("G/B Ratio")

            # If no valid signal was computed (should not happen, but check for safety), notify.
            if not signals_to_plot:
                notifications.show_info('No valid signal data available for plotting.')
                return

            # Call your plotting function with the signals and their corresponding labels.
            self.plot_widget.plot_signal(*signals_to_plot, titles=labels_to_plot)

            # Mark the mask as selected and notify the user.
            self.masks_layer.selected_label = label
            notifications.show_info(f'Plotted signal changes in mask {label} in the bottom plot widget')

    def on_undo(self, event):
        print('undo called')
        if self.masks_history:
            print('undo')
            notifications.show_info('Undo')
            # Revert to the previous state
            self.masks_layer.data = self.masks_history.pop()
            self.masks_layer.refresh()

    def apply_to_following_frames(self):
        self.masks_layer = self.viewer.layers['Masks']

        # Get the current mask label
        current_mask_label = self.masks_layer.selected_label # get the selected label
        print(f'current mask label: {current_mask_label}')

        current_frame = self.viewer.dims.current_step[0]
        # print(f'current frame: {current_frame}')

        # Get positions where current mask label is present in the current frame
        mask_positions = np.where(self.masks_layer.data[current_frame] == current_mask_label)
        # print(f'mask positions shape: {mask_positions[0].shape}')

        # Apply this mask label to the same positions in all following frames
        num_frames = self.masks_layer.data.shape[0]
        for frame in range(current_frame + 1, num_frames):
            # Set the mask label at the identified positions in the following frames
            self.masks_layer.data[frame][mask_positions] = current_mask_label

        # Refresh the mask layer to update the viewer
        self.masks_layer.refresh()
        notifications.show_info(f'Applied mask label {current_mask_label} to following frames ({current_frame + 1} to {num_frames - 1})')

    def apply_to_next_frame(self):
        self.masks_layer = self.viewer.layers['Masks']

        # Get the current mask label
        current_mask_label = self.masks_layer.selected_label  # get the selected label
        print(f'current mask label: {current_mask_label}')

        current_frame = self.viewer.dims.current_step[0]
        # print(f'current frame: {current_frame}')

        # Get positions where current mask label is present in the current frame
        mask_positions = np.where(self.masks_layer.data[current_frame] == current_mask_label)
        # print(f'mask positions shape: {mask_positions[0].shape}')

        # Apply this mask label to the same positions in the next frame

        self.masks_layer.data[current_frame + 1][mask_positions] = current_mask_label

        # Refresh the mask layer to update the viewer
        self.masks_layer.refresh()
        notifications.show_info(
            f'Applied mask label {current_mask_label} to the next frame ({current_frame + 1})')



import multiprocessing as mp
import torch
from .track_anything_simple import TrackingAnything, parse_augment

# def find_contours(mask: np.ndarray) -> np.ndarray:
def find_contours(mask: np.ndarray):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(c, 1) for c in contours]
    for i, c in enumerate(contours):
        if not np.array_equal(c[0], c[-1]):
            contours[i] = np.vstack([c, c[0]])
    # return np.array(contours, dtype=np.int16)
    return contours
def find_nearest_masks(initial_masks, mask_contours, cell_dist):
    """
    initial_masks: 2D uint8 mask array
    mask_contours: dict[label] = list of np.arrays shape (Ni,2) 浮点或整型
    cell_dist: 距离阈值
    返回：[[label1,label2,...], [...], ...]  连通分量列表
    """

    labels = np.array([lab for lab in mask_contours.keys()], dtype=int)
    n = len(labels)
    # 1) prepare bounding boxes for each label
    bboxes = {}
    for lab in labels:
        bin_mask = (initial_masks == lab).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(bin_mask)
        bboxes[lab] = (x, y, w, h)

    # 2) initialize union-find structure
    parent = {lab: lab for lab in labels}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cd2 = cell_dist * cell_dist
    # 3)
    for i, j in combinations(labels, 2):
        x1, y1, w1, h1 = bboxes[i]
        x2, y2, w2, h2 = bboxes[j]
        # 3.1
        dx = max(0, x2 - (x1 + w1), x1 - (x2 + w2))
        dy = max(0, y2 - (y1 + h1), y1 - (y2 + h2))
        if dx*dx + dy*dy > cd2:
            continue  #

        # 3.2 candidate contours
        min_dist = np.inf
        for c1 in mask_contours[i]:
            for c2 in mask_contours[j]:
                d = cdist(c1, c2, 'euclidean').min()
                if d < min_dist:
                    min_dist = d
                    if min_dist < cell_dist:
                        break
            if min_dist < cell_dist:
                break

        if min_dist < cell_dist:
            union(i, j)

    groups = {}
    for lab in labels:
        root = find(lab)
        groups.setdefault(root, []).append(lab)
    # plot the masks and boxes in 1 figure, of all batches to

    return list(groups.values())



def batch_masks_by_distance(contours: dict[int, np.ndarray], cell_dist: float) -> list[list[int]]:
    labels = np.array(list(contours.keys()), dtype=np.uint8)
    batches, processed = [], set()
    for idx in labels:
        if idx in processed:
            continue
        stack, group = [idx], {idx}
        while stack:
            cur = stack.pop()
            for other in labels:
                if other in processed or other == cur:
                    continue
                c1_list, c2_list = contours[cur], contours[other]
                if c1_list.size and c2_list.size:
                    dmin = min(np.linalg.norm(p1 - p2)
                               for p1 in c1_list.reshape(-1,2)
                               for p2 in c2_list.reshape(-1,2))
                    if dmin < cell_dist:
                        group.add(other)
                        stack.append(other)
                        processed.add(other)
        batches.append(list(group))
    return batches


def compute_trajectories(mask_stack: np.ndarray) -> np.ndarray:
    labels = np.unique(mask_stack)
    labels = labels[labels != 0]
    n_frames = mask_stack.shape[0]
    traj = np.full((n_frames, labels.max()+1, 2), np.nan)
    for t in range(n_frames):
        for lbl in labels:
            traj[t, lbl] = center_of_mass((mask_stack[t]==lbl).astype(np.uint8))
    return traj


def _to_rgb_logscale(stack: np.ndarray) -> np.ndarray:
    scale, C = 0.005, 65535/np.log(1+65535*0.005)
    cmap = plt.get_cmap('viridis')
    rgb = np.zeros((*stack.shape,3), dtype=np.uint8)
    for i, img in enumerate(stack):
        im = np.log1p(img.astype(float)*scale)*C
        im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)
        rgb[i] = (cmap(im)[...,:3]*255).astype(np.uint8)
    return rgb


def _load_stack(path: Union[str, Path]) -> np.ndarray:
    """Load an image or numpy stack, given a file path."""
    path_str = str(path)
    if path_str.lower().endswith(('.tif', '.tiff')):
        return tifffile.imread(path_str)
    elif path_str.lower().endswith('.npy'):
        return np.load(path_str)
    else:
        raise ValueError(f"Unsupported stack format: {path_str}")


def _load_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a mask file (.npy or image); return a 2D array."""
    path_str = str(path)
    if path_str.lower().endswith('.npy'):
        return np.load(path_str, allow_pickle=True).item().get('masks')
    else:
        img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        return img if img.ndim == 2 else img[..., 0]


# --- Utility functions ---
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def preprocess_rgb(img_stack, log_alpha, colormap, start_frame, end_frame, desc="Colorizing"):
    sub = img_stack[start_frame: end_frame + 1]
    cmap = plt.get_cmap(colormap)
    scale = float(log_alpha)

    gmin = float(sub.min())
    gmax = float(sub.max())

    T = sub.shape[0]
    H, W = sub.shape[1], sub.shape[2]
    rgb = np.zeros((T, H, W, 3), dtype=np.uint8)

    if gmax > gmin and scale > 0:
        vmax_log = np.log1p((gmax - gmin) * scale)
        inv_vmax_log = 1.0 / vmax_log
    else:
        vmax_log = 0.0
        inv_vmax_log = 0.0

    for i in tqdm(range(T), desc=desc):
        img = sub[i].astype(np.float32)

        im = img - gmin
        im = np.log1p(im * scale)
        if vmax_log > 0:
            im *= inv_vmax_log
        else:
            im[:] = 0.0

        im = np.clip(im, 0.0, 1.0)

        rgb[i] = (cmap(im)[..., :3] * 255).astype(np.uint8)

    return start_frame, rgb


# reuse _process_ta_batch but now expects rgb_stack offset index

# track_with_cutie removed; only TrackAnything backend remains

def _process_ta_batch(rgb_subset, initial_masks, batch, padding):
    bm = np.zeros_like(initial_masks, np.uint8)
    for oid in batch:
        bm += (initial_masks == oid).astype(np.uint8) * oid
    x, y, w, h = cv2.boundingRect(bm)
    y1 = max(y - padding, 0)
    x1 = max(x - padding, 0)
    y2 = min(y + h + padding, initial_masks.shape[0])
    x2 = min(x + w + padding, initial_masks.shape[1])
    cropped = [f[y1:y2, x1:x2] for f in rgb_subset]
    tmpl    = bm[y1:y2, x1:x2]
    masks = model.generator(cropped, tmpl)
    model.xmem.clear_memory()
    print(f"Processed batch {batch}")
    return masks, y1, y2, x1, x2


def _init_worker(sam_ckpt, xmem_ckpt, e2fgvi_ckpt, args_dict):
    global model
    args = parse_augment()
    args.__dict__.update(args_dict)
    model = TrackingAnything(sam_ckpt, xmem_ckpt, e2fgvi_ckpt, args)


def track_with_tasimple(img_stack: np.ndarray,
                         initial_masks: np.ndarray,
                         start_frame: int,
                         end_frame: int,
                         log_alpha: float,
                         colormap: str,
                         cell_dist: int = 4,
                         padding: int = 50,
                         num_processes: int = 4,
                         sam_checkpoint: str = None,
                         xmem_checkpoint: str = None,
                         e2fgvi_checkpoint: str = None,
                         ) -> np.ndarray:

    if end_frame >= img_stack.shape[0]:
        end_frame = img_stack.shape[0] - 1
    full_out = np.zeros_like(img_stack, dtype=np.uint16)
    print(f'full_out shape: {full_out.shape}')

    offset, rgb_subset = preprocess_rgb(img_stack, log_alpha, colormap, start_frame, end_frame)
    mask_contours = {i: find_contours(initial_masks == i) for i in np.unique(initial_masks) if i != 0}
    batches = find_nearest_masks(initial_masks, mask_contours, cell_dist)

    args = parse_augment()
    args.mask_save = True
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run_serial():
        print("⚠️ Falling back to single-process execution.")
        _init_worker(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args.__dict__)
        return [_process_ta_batch(rgb_subset, initial_masks, b, padding) for b in batches]

    results = None
    if num_processes == 1:
        results = run_serial()
    else:
        try:
            print(f"🧵 Launching parallel pool with {num_processes} processes.")
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args.__dict__)
            )
            inputs = [(rgb_subset, initial_masks, b, padding) for b in batches]
            results = pool.starmap(_process_ta_batch, inputs)
            pool.close()
            pool.join()
        except Exception as e:
            print("❌ Parallel processing failed:", repr(e))
            traceback.print_exc()
            results = run_serial()

    out = np.zeros((end_frame - start_frame + 1, ) + initial_masks.shape, dtype=np.uint16)
    for masks, y1, y2, x1, x2 in results:
        out[:, y1:y2, x1:x2] = np.maximum(out[:, y1:y2, x1:x2], masks)

    full_out[start_frame:end_frame+1, :, :] = out
    return full_out
def smooth_stack(input_tiff, output_tiff, window_size):
    # Calculate the averages
    # for i in range(input_tiff.shape[0]):
    for i in tqdm(range(input_tiff.shape[0])):
        # start = i
        start = min(i, input_tiff.shape[0] - window_size)
        end = min(i + window_size, input_tiff.shape[0])
        output_tiff[i] = np.mean(input_tiff[start:end], axis=0)

    # Replace the first 15 frames with the average of the first 30 frames
    output_tiff[:window_size//2] = np.mean(input_tiff[:window_size], axis=0)
    return output_tiff
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)


    if not os.path.exists(filepath):
        import requests
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")
    else:
        print(f"Checkpoint {filename} already exists at {filepath}, skipping download.")
    return filepath
def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        import gdown
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")
    else:
        print(f"Checkpoint {filename} already exists at {filepath}, skipping download.")

    return filepath
# class MultiModelTracker(Container):
class BPTracker(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__(layout='vertical')
        self.viewer = viewer
        self.window_size = SpinBox(label='Smooth Window', value=40, min=1)
        self.smooth_btn = PushButton(text='Smooth Stack')
        self.smooth_btn.changed.connect(self._on_smooth)
        # default_tiff = r"D:\PKU_STUDY\SynologyDrive\BaoyiWang\BC-FLIM\Figures\Tracking\NLS-N1-3-4-7-8-9-10-11-12-13-14-15-16-240827-10uM 5-HT\smoothed_stack-b.tif"
        # default_tiff = r"D:\PKU_STUDY\SynologyDrive\BaoyiWang\BC-FLIM\Figures\Tracking\NLS-N1-3-4-7-8-9-10-11-12-13-14-15-16-240827-10uM 5-HT\smoothed_stack-b.tif"
        # default_mask = r"D:\PKU_STUDY\SynologyDrive\BaoyiWang\BC-FLIM\Figures\Tracking\NLS-N1-3-4-7-8-9-10-11-12-13-14-15-16-240827-10uM 5-HT\smoothed_stack-1_seg.npy"
        default_tiff = r"E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\stack-b.tif"
        default_mask = r"E:\BC-FLIM\Hek293T-BJMU\NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE\stack-b-1-6-avg_seg.npy"
        # # attempt to load defaults once at startup
        if Path(default_tiff).exists() and Path(default_mask).exists():
            stack = _load_stack(default_tiff)
            mask  = _load_mask(default_mask)
            self.viewer.add_image(stack, name='Default Stack')
            self.viewer.add_labels(mask.astype(np.uint16), name='Default Mask')
            self._stack = stack
            self._mask  = mask
        else:
            show_warning("Default files not found—please load Image Stack and Mask below.")
            self._stack = None
            self._mask  = None
        # load defaults omitted for brevity
        self.tiff_path = FileEdit(label='Image Stack', mode='r', value=default_tiff)
        self.mask_path = FileEdit(label='Mask', mode='r', value=default_mask)

        self.stack_layer = ComboBox(
            label='Use existing Image layer',
            choices=[layer.name for layer in self.viewer.layers if isinstance(layer, NapariImage)],
            nullable=True,
            value=None,
        )
        self.mask_layer = ComboBox(
            label='Use existing Labels layer',
            choices=[layer.name for layer in self.viewer.layers if isinstance(layer, Labels)],
            nullable=True,
            value=None,
        )
        self.stack_layer.changed.connect(self._on_select_stack_layer)
        self.mask_layer.changed.connect(self._on_select_mask_layer)
        self.viewer.layers.events.changed.connect(self._refresh_layer_choices)
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)
        # call once to populate
        self._refresh_layer_choices()

        self.frame_start = SpinBox(label='Start Frame', value=0, min=0)
        self.frame_end = SpinBox(label='End Frame', value=2000, min=0) # default max is 999
        self.log_alpha = FloatSlider(label='Log α', min=0.0, max=0.1, step=0.001, value=0.005)
        self.colormap = ComboBox(label='Colormap', choices=['rainbow','viridis','plasma','magma', 'gray'], value='rainbow') # can also add gray 'gray'
        self.preview_btn = PushButton(text='Preview Colorization')
        self.preview_btn.changed.connect(self._on_preview)
        self.visualize_btn = PushButton(text='Preview Batches')
        self.visualize_btn.changed.connect(self._on_preview_batches)
        self.notify = lambda msg: show_info(msg)
        # File inputs (only used if user wants to override)
        self.tiff_path = FileEdit(label='Image Stack', mode='r', value=default_tiff)
        self.mask_path = FileEdit(label='Mask', mode='r', value=default_mask)
        self.tiff_path.changed.connect(self._on_load)
        self.mask_path.changed.connect(self._on_load)
        # Backend selector now only TrackAnything
        self.backend = ComboBox(label='Backend', choices=['TrackAnything'], value='TrackAnything')
        self.cell_dist = SpinBox(label='Cell dist', value=4, min=1)
        self.padding = SpinBox(label='Padding', value=20, min=0)
        self.num_proc = SpinBox(label='Processes', value=2, min=1)
        self.track_btn = PushButton(text='Track')
        self.track_btn.changed.connect(self._on_track)
        self.save_btn = PushButton(text='Save Tracking')
        self.save_btn.changed.connect(self.save_tracking)
        for w in [self.tiff_path, self.mask_path,
                self.stack_layer, self.mask_layer,
                  self.frame_start, self.frame_end,
                  self.log_alpha, self.colormap,
                  self.window_size, self.smooth_btn,
                  self.preview_btn, self.visualize_btn,
                  self.backend,
                  self.cell_dist, self.padding, self.num_proc,
                  self.track_btn, self.save_btn]:
            self.append(w)
        # viewer.window.add_dock_widget(self, area='right', name='Multi-Model Tracker')


    def _refresh_layer_choices(self, event=None):
        names_img = [layer.name for layer in self.viewer.layers if isinstance(layer, NapariImage)]
        names_lbl = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]

        self.stack_layer.choices = names_img
        self.mask_layer.choices = names_lbl

        if self.stack_layer.value not in names_img:
            self.stack_layer.value = None
        if self.mask_layer.value not in names_lbl:
            self.mask_layer.value = None


    def _on_select_stack_layer(self):
        """Callback when the user picks an existing Image layer."""
        name = self.stack_layer.value
        if name is None:
            return
        data = self.viewer.layers[name].data
        # if it’s a Dask array, compute it now:
        if hasattr(data, 'compute'):
            data = data.compute()
        self._stack = data
        self.notify(f"Stack set from layer “{name}”")

    def _on_select_mask_layer(self):
        """Callback when the user picks an existing Labels layer."""
        name = self.mask_layer.value
        if name is None:
            return
        layer = self.viewer.layers[name]
        if isinstance(layer, Labels):
            self._mask = layer.data
            self.notify(f"Mask set from layer “{name}”")
    def _on_smooth(self):
        if self._stack is None:
            show_warning("Please load an Image Stack first.")
            return

        w = self.window_size.value
        smoothed = smooth_stack(self._stack, np.zeros_like(self._stack), w)
        self._smoothed_stack = smoothed
        self.viewer.add_image(smoothed, name=f'Smoothed (w={w})')
        show_info(f'Stack smoothed (window={w})')
    def save_tracking(self):
        # save to a folder of npy files, 00000-....npy
        # pop out a dialog to choose the folder
        folder = QFileDialog.getExistingDirectory(caption='Select Folder to Save Masks')
        if folder:  # If a folder was chosen
            # save the tracked masks layer to the folder
            tracked_masks = self.viewer.layers['Tracked Masks'].data
            # for i in range(tracked_masks.shape[0]):
            # only save the beginning-end frames
            start, end = self.frame_start.value, self.frame_end.value
            # end = min (end, tracked_masks.shape[0] - 1)  # Ensure end is within bounds
            if end >= tracked_masks.shape[0]:
                end = tracked_masks.shape[0] - 1
            for i in range(start, end + 1):
                np.save(os.path.join(folder, f'{i:05d}.npy'), tracked_masks[i].astype(np.uint8))
            show_info(f"Tracking results saved to {folder}")
            notifications.show_info(f"Tracking results saved to {folder}")

    def _on_preview(self):
        # img = _load_stack(self.tiff_path.value)
        # load the smoothed one, if not exists, use the original stack, if not exists, show warning
        if not hasattr(self, '_stack'):
            show_warning("Please load an Image Stack first.")
            return
        img = self._smoothed_stack if hasattr(self, '_smoothed_stack') else self._stack
        start, end = self.frame_start.value, self.frame_end.value
        if end >= img.shape[0]:
            end = img.shape[0] - 1
        _, rgb = preprocess_rgb(img, self.log_alpha.value, self.colormap.value, start, end)
        self.viewer.add_image(rgb, name='Preprocessed Preview')

    def _on_preview_batches(self):
        # masks = _load_mask(self.mask_path.value)
        if not hasattr(self, '_mask'):
            show_warning("Please load a Mask first.")
            return
        masks = self._mask
        dist, pad = self.cell_dist.value, self.padding.value
        contours = {i: find_contours(masks==i) for i in np.unique(masks) if i!=0}
        for i, ctr in contours.items():
            if len(ctr) < 2:
                continue
            self.viewer.add_shapes([ctr], shape_type='path', name=f'Contour {i}', edge_color='yellow', visible=False)
        batches = find_nearest_masks(masks, contours, dist)
        self.notify(f"Found {len(batches)} batches with cell distance {dist}.")
        shapes = []
        for batch in batches:
            bm = sum((masks==m).astype(np.uint8) for m in batch)
            x,y,w,h = cv2.boundingRect(bm)
            shapes.append([[y-pad, x-pad], [y+h+pad, x+w+pad]])
        self.viewer.add_shapes(shapes, shape_type='rectangle', name='Batch Boxes')

    def _on_load(self, path: pathlib.Path):
        try:
            # decide by comparing which widget sent it
            if path == self.tiff_path.value:
                stack = _load_stack(path)
                self.viewer.add_image(stack, name='Stack (loaded)')
                self._stack = stack
            else:
                mask = _load_mask(path)
                self.viewer.add_labels(mask.astype(np.uint16), name='Mask (loaded)')
                self._mask = mask
        except Exception as e:
            show_warning(f"Failed to load: {e}")

    def _on_track(self):
        if self._stack is None or self._mask is None:
            show_warning("Please load both Image Stack and Mask before tracking.")
            return
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        xmem_checkpoint = "XMem-s012.pth"

        xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
        e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"
        e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"

        # folder = "./checkpoints"
        folder = str(Path(__file__).parent / 'checkpoints')
        SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
        xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
        e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)

        show_info('Starting tracking...')
        notifications.show_info('Tracking started. This may take a while depending on the number of frames and masks.')
        if hasattr(self, '_smoothed_stack') and self._smoothed_stack is not None:
            stack = self._smoothed_stack
            # Use the smoothed stack if available
            notifications.show_info('Using smoothed stack for tracking.')
        else:
            stack = self._stack
        masks = self._mask
        params = dict(
            start_frame=self.frame_start.value,
            end_frame=self.frame_end.value,
            log_alpha=self.log_alpha.value,
            colormap=self.colormap.value,
            cell_dist=self.cell_dist.value,
            padding=min(self.padding.value, *masks.shape)
        )
        # Always use TrackAnything backend
        params.update({
            'num_processes': self.num_proc.value,
            # 'sam_checkpoint': str(Path(__file__).parent/'checkpoints'/'sam_vit_h_4b8939.pth'),
            # 'xmem_checkpoint': str(Path(__file__).parent/'checkpoints'/'XMem-s012.pth'),
            # 'e2fgvi_checkpoint': str(Path(__file__).parent/'checkpoints'/'E2FGVI-HQ-CVPR22.pth'),
            'sam_checkpoint': SAM_checkpoint,
            'xmem_checkpoint': xmem_checkpoint,
            'e2fgvi_checkpoint': e2fgvi_checkpoint,
        })
        time_start = time.time()
        result = track_with_tasimple(stack, masks, **params)
        time_end = time.time()
        self.viewer.add_labels(result.astype(np.uint8), name='Tracked Masks')
        # show_info('Tracking completed.')
        # notifications.show_info('Tracking completed. You can now save the tracking results or visualize them further.')
        notifications.show_info(f'Tracking completed in {time_end - time_start:.2f} seconds. '
                                f'You can now save the tracking results or visualize them further.')