# 王抱一 last modified on 2023-5-31



import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from skimage.measure import regionprops
from skimage.morphology import erosion, disk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import tifffile as tiff
plt.switch_backend('TkAgg')
def get_color_map(n_colors):
    # color_list_14 = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#469990', '#9A6324', '#808000', '#000075', '#800000', '#aaffc3']
    # add a gray color at first for list 15
    color_list_15 = ['#808080', '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#469990', '#9A6324', '#808000', '#000075', '#800000', '#aaffc3'] # 808080 is gray
    # colors = color_list_14[:n_colors]
    colors = color_list_15[:n_colors]
    colors = [(int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)) for c in colors]
    colors = np.array(colors) / 255  # for plt.scatter's 0-1 input
    return colors

class ScatterGUI:
    def __init__(self, root):
        self.root = root
        self.selected_class = 0
        self.coords = []
        self.closest_indices = []
        self.lasso = None
        self.current_ax = None
        # Log text
        self.log_text = tk.Text(root, height=10, width=50)
        self.log_text.config(font=("Courier", 12), bg='white', fg='black')
        self.log_text.pack()


        # GUI elements
        self.save_button = tk.Button(root, text="Save Results", command=self.save_results)
        self.save_button.pack() # pack() is used to display the button
        self.save_button.config(font=("Courier", 20), bg='white', fg='black')



        self.root.bind('<Key>', self.onkey)
        self.root.title('Assign class and Save results GUI')
        self.root.configure(background='white')
        self.root.geometry("500x400")
        self.root.attributes('-topmost', 1) # bring the window to the front
        self.root.lift() # bring the window to the front, but not always effective

        # Load data and clustering
        self.root.update_idletasks() # update the window
        self.load_data()
        # self.root.mainloop()
        self.initialize_clustering()
    def log(self, message):
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END) # Scroll to the bottom after adding text

    def load_data(self):
        self.dims_read = ['G', 'S', 'Int 1/(1-4)', 'Int 2/(1-4)', 'Int 3/(1-4)', 'Total intensity', 'FOV', 'Mask label']
        self.dims_use = ['G', 'S', 'Int 1/(1-4)', 'Int 2/(1-4)', 'Int 3/(1-4)']
        # self.dims_read = ['G', 'S', 'Total intensity', 'FOV', 'Mask label']
        # self.dims_use = ['G', 'S', 'Total intensity']
        # self.ref_FPs = ['mScarlet3', 'mScarlet-I3', 'mScarlet-I', 'mApple', 'mRuby3', 'mKate2', 'FR-MQV', 'TagRFP-T', 'FR-MQ', 'FR-1', 'FR-M', 'FR', 'mCherry', 'mScarlet-H'] # 14 FPs
        self.ref_FPs = ['N10', 'N13', 'N4', 'N14', 'N9', 'N3', 'N7', 'N12', 'N16', 'N15', 'N11', 'N2', 'N8', 'N1'] # 14 FPs
        # self.ref_FPs = ['N10', 'N13', 'N4', 'N14', 'N7', 'N16', 'N8', 'N1'] # 14 FPs
        self.test_FPs = ['N10', 'N13', 'N4', 'N14', 'N9', 'N3', 'N7', 'N12', 'N16', 'N15', 'N11', 'N2', 'N8', 'N1'] # 14 FPs
        # self.test_FPs = ['N4', 'N3', 'N7', 'N2', 'N1'] # 5 FPs for PM
        # self.test_FPs = ['N10']
        # self.test_FPs = ['M10', 'P13', 'N9',  'M3','P7', 'N8'] # 6 FPs for 6 Mix
        # self.add_text = True # or False
        self.add_text = False
        self.font_scale = 0.8
        self.ref_alpha = 0.4
        self.legend_fontsize = 10
        # self.num_clusters = 2
        self.num_clusters = len(self.test_FPs)
        # self.background_color = 'black'
        self.background_color = 'white'
        self.log('Loading data')
        # self.test_base_folder = filedialog.askdirectory(title='Select the test data folder')
        # self.test_base_folder = r'F:\barcode_info'
        # self.test_base_folder = r'G:\BC-FLIM-S\WBY\Hek293T-BJMU'
        # self.test_base_folder = r'G:\BC-FLIM-S\WBY\MEF-BJMU'
        # self.test_base_folder = r'E:\BC-FLIM\MEF-BJMU'
        self.test_base_folder = r'E:\BC-FLIM\Hek293T-BJMU'
        # self.test_base_folder = r'E:\BC-FLIM\MCF7-PKU'
        self.ref_base_folder = r'E:\BC-FLIM\Hek293T-BJMU'
        self.ref_data = [
                         # 'NLS-FR-1-240222-1', 'NLS-FR-1-240222-2',
                         #  'NLS-FR-240222-1', 'NLS-FR-240222-2',
                         #  'NLS-FR-MQV-240222-1', 'NLS-FR-MQV-240222-2',
                         #  'NLS-mCherry-240222-1', 'NLS-mCherry-240222-2',
                         #  'NLS-mScarlet-I3-240222-1', 'NLS-mScarlet-I3-240222-2',
                         #  'NLS-mScarlet-H-240222-1', 'NLS-mScarlet-H-240222-2',
                         #  'NLS-mScarlet3-240222-1', 'NLS-mScarlet3-240222-2',
                         #  'NLS-TagRFP-T-240222-1', 'NLS-TagRFP-T-240222-2',
                         #  'NLS-mScarlet-I-240229-1', 'NLS-mScarlet-I-240229-2',
                         #  'NLS-mApple-240229-1', 'NLS-mApple-240229-2',
                         #  'NLS-FR-M-240229-1', 'NLS-FR-M-240229-2',
                         #  'NLS-mRuby3-240308-1', 'NLS-mRuby3-240308-2',
                         #  'NLS-mKate2-240308-1', 'NLS-mKate2-240308-2',
                         #  'NLS-FR-MQ-240308-1', 'NLS-FR-MQ-240308-2',
            # 'NTOM-N10-240528',
            # 'NTOM-N13-240528',
            # 'NTOM-N4-240528',
            # 'NTOM-N14-240528',
            # 'NTOM-N16-240528',
            # 'NTOM-N8-240528',
            # 'NTOM-N1-240528',
            'NLS-N10-240623',
            'NLS-N13-240623',
            'NLS-N4-240623',
            'NLS-N14-240623',
            'NLS-N16-240623',
            'NLS-N8-240623',
            'NLS-N1-240623-1',
            'NLS-N15-240623',
            'NLS-N11-240623',
            'NLS-N2-240623',
            'NLS-N7-240623',
            'NLS-N3-240623',
            'NLS-N9-240623',
            'NLS-N12-240623',
        ]

        self.test_data = [
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-240429-NE-9',
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240308-FSK',
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE',
            # 'NLS-mScarlet3-231219-MS',
            # 'NLS-N1-13-231127-MS',
            # 'NLS-14Mix-NLYN-5Mix-240618-1',
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240401-DMSO'
            # 'NTOM-N10-240528',
            # 'NTOM-N13-240528',
            # 'NTOM-N4-240528',
            # 'NTOM-N14-240528',
            # 'NTOM-N16-240528',
            # 'NTOM-N8-240528',
            # 'NTOM-N1-240528',
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-FSK-1'
            # 'NTOM-M1-240629',
            # 'NTOM-M4-240629',
            # 'NTOM-M8-240629',
            # 'NTOM-M10-240629',
            # 'NTOM-M13-240629',
            # 'NTOM-M14-240629',
            # 'NTOM-M16-240629',
            # 'NLYN-P1-240629',
            # 'NLYN-P2-240629',
            # 'NLYN-P3-240629',
            # 'NLYN-P4-240629',
            # 'NLYN-P8-240629',
            # 'NLYN-P9-240629',
            # 'NLYN-P10-240629',
            # 'NLYN-P13-240629',
            # 'NLYN-P15-240629',
            # 'NLYN-P16-240629',
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE', # not good
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-FSK-2'
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240510-1'
            # 'NLS-N1-3-4-7-8-9-10-11-13-14-15-16-240425-1'
            # 'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240512-NE-2',
            # 'NLS-14MIX1-1-240627'
            # 'NLS-14MIX-240603-DCZ'
            # '6 Mix N8_N9_P7_P13_M3_M10-6-240728',
            # '6 Mix N8_N9_P7_P13_M3_M10-7-240728',
            # '6 Mix N8_N9_P7_P13_M3_M10-1-240728',
            # '6 Mix N8_N9_P7_P13_M3_M10-2-240728',
            # '6 Mix N8_N9_P7_P13_M3_M10-3-240728',
            # '6 Mix N8_N9_P7_P13_M3_M10-4-240728',
            # '6 Mix N8_N9_P7_P13_M3_M10-5-240728',
            'NLS-N1-2-3-4-7-8-9-10-11-12-13-14-15-16-240329-NE'
        ]
        self.discard_fovs = []
        self.int_thres = 1e3  # 200000
        self.pixel_wise = False
        self.df_test = pd.DataFrame(columns=self.dims_read)
        self.df_ref = pd.DataFrame(columns=self.dims_use)
        for FP in self.ref_FPs:
            for file in self.ref_data:
                if not f'-{FP}-2' in file:
                    continue
                file_path = os.path.join(self.ref_base_folder, file, f'{FP}_Nu.xlsx')
                if not os.path.exists(file_path):
                    # file_path = os.path.join(self.ref_base_folder, file, f'barcode_info_0_10.xlsx')  # for the new data with barcode_info.xlsx
                    file_path = os.path.join(self.ref_base_folder, file, f'barcode_info.xlsx')  # for the new data with barcode_info.xlsx
                df_read = pd.read_excel(file_path, usecols=self.dims_use)
                df_read['file'] = file
                df_read['FP'] = FP
                self.df_ref = pd.concat([self.df_ref, df_read], ignore_index=True)
            print(f'ref: {file_path}')
        # self.df_ref = self.df_ref[self.df_ref['Total intensity'] > self.int_thres]
        self.df_ref = self.df_ref.dropna()
        self.df_ref = self.df_ref.reset_index(drop=True)

        for file in self.test_data:
            file_path = ''
            for file_name in os.listdir(os.path.join(self.test_base_folder, file)):
            # for file_name in os.listdir(self.test_base_folder): # for barcode_info folder
                if self.pixel_wise:
                    excel_keyword = 'pixel'
                else:
                    excel_keyword = 'Nu'
                    # excel_keyword = 'barcode_info'
                if excel_keyword in file_name:
                    file_path = os.path.join(self.test_base_folder, file, file_name)
                    break
                if not os.path.exists(file_path):
                    file_path = os.path.join(self.test_base_folder, file, 'barcode_info.xlsx')  # for the new data with barcode_info.xlsx
                if not os.path.exists(file_path):
                    file_path = os.path.join(self.test_base_folder, f'{file}_barcode_info.xlsx')  # for the new data with barcode_info.xlsx
            print(f'bc: {file_path}')
            df_read = pd.read_excel(file_path, usecols=self.dims_read)
            df_read['file'] = file
            df_read['FP'] = file  # use the file name as the FP temporarily
            self.df_test = pd.concat([self.df_test, df_read], ignore_index=True)
        self.df_test = self.df_test[self.df_test['Total intensity'] > self.int_thres]
        self.df_test = self.df_test.dropna()
        self.df_test = self.df_test.reset_index(drop=True)
        self.df_test = self.df_test[~self.df_test['FOV'].isin(self.discard_fovs)]
        self.log(f'Data loaded, please select {self.num_clusters} seeds for clustering. The order should be the same as the order of: {self.test_FPs}')

    def initialize_clustering(self):
        colors = get_color_map(len(self.ref_FPs) + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        legend_elements = []

        for i, file in enumerate(self.df_test['file'].unique()):
            file_data = self.df_test[self.df_test['file'] == file]
            ax1.scatter(file_data['G'], file_data['S'], label=file, alpha=0.8, s=30, color=colors[i + 1])
            ax2.scatter(file_data['Int 1/(1-4)'], file_data['Int 2/(1-4)'], label=file, alpha=0.8, s=30,
                        color=colors[i + 1])
            ax3.scatter(file_data['Int 1/(1-4)'], file_data['Int 3/(1-4)'], label=file, alpha=0.8, s=30,
                        color=colors[i + 1])
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', label=file, markersize=10, alpha=0.9, linestyle='None',
                           markerfacecolor=colors[i + 1], color='w'))

        for i, FP in enumerate(self.ref_FPs):
            FP_data = self.df_ref[self.df_ref['FP'] == FP]
            ax1.scatter(FP_data['G'], FP_data['S'], color=colors[i + 1], s=40, alpha=self.ref_alpha, marker='+')
            ax2.scatter(FP_data['Int 1/(1-4)'], FP_data['Int 2/(1-4)'], color=colors[i + 1], s=40, alpha=self.ref_alpha)
            ax3.scatter(FP_data['Int 1/(1-4)'], FP_data['Int 3/(1-4)'], color=colors[i + 1], s=40, alpha=self.ref_alpha)
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label=FP, markerfacecolor=colors[i + 1], markersize=10,
                           alpha=0.3))

        for ax in [ax1, ax2, ax3]:
            ax.legend(handles=legend_elements, fontsize=self.legend_fontsize)
            ax.set_xlabel('G' if ax == ax1 else 'Int 1/(1-4)')
            ax.set_ylabel('S' if ax == ax1 else ('Int 2/(1-4)' if ax == ax2 else 'Int 3/(1-4)'))
            ax.set_title(f'Threshold: {int(self.int_thres)}')

        def onclick(event):
            ix, iy = event.xdata, event.ydata
            if ix is not None and iy is not None:
                if event.inaxes == ax1:
                    coords = self.df_test[['G', 'S']].to_numpy()
                elif event.inaxes == ax2:
                    coords = self.df_test[['Int 1/(1-4)', 'Int 2/(1-4)']].to_numpy()
                elif event.inaxes == ax3:
                    coords = self.df_test[['Int 1/(1-4)', 'Int 3/(1-4)']].to_numpy()
                else:
                    return

                distances = cdist([(ix, iy)], coords)
                closest_index = np.argmin(distances)
                closest_point = coords[closest_index]
                self.closest_indices.append(closest_index)
                self.coords.append(closest_point)
                i_seed = len(self.coords)

                ax1.plot(self.df_test.iloc[closest_index]['G'], self.df_test.iloc[closest_index]['S'], '*',
                         label=f'Seed {i_seed}', color=colors[i_seed], markersize=13)
                ax2.plot(self.df_test.iloc[closest_index]['Int 1/(1-4)'],
                         self.df_test.iloc[closest_index]['Int 2/(1-4)'], '*', label=f'Seed {i_seed}',
                         color=colors[i_seed], markersize=13)
                ax3.plot(self.df_test.iloc[closest_index]['Int 1/(1-4)'],
                         self.df_test.iloc[closest_index]['Int 3/(1-4)'], '*', label=f'Seed {i_seed}',
                         color=colors[i_seed], markersize=13)

                for ax in [ax1, ax2, ax3]:
                    handles, labels = ax.get_legend_handles_labels()
                    handles[-1] = plt.Line2D([0], [0], marker='*', color=colors[i_seed], markersize=13,
                                             linestyle='None', label=f'Seed {i_seed}')
                    labels[-1] = f'Seed {i_seed}'
                    ax.legend(handles, labels, fontsize=self.legend_fontsize)
                    plt.draw()

                if len(self.coords) == self.num_clusters:
                    print(f'Selection complete')
                    self.log(
                        'Selection complete. Now you can assign class to the point(s) using the keyboard. Class 0 is for unassigned points.')
                    self.log('Classes more than 9 are assigned to a, b, c, d, e, f...')
                    self.log(
                        'Press Ctrl + Click to assign class to a point. Press Shift + Click on the plot to activate Lasso Selector so as to assign in batches.')
                    plt.close(fig)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()



        seeds_5d = self.df_test.iloc[self.closest_indices][self.dims_use].to_numpy()
        X = self.df_test[self.dims_use].to_numpy()
        weights = np.array([4, 4, 1, 1, 1])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X * weights
        seeds_5d = scaler.transform(seeds_5d)
        seeds_5d = seeds_5d * weights
        kmeans = KMeans(n_clusters=self.num_clusters, init=seeds_5d, n_init=1, max_iter=300, random_state=42)
        kmeans.fit(X)
        self.df_test['cluster'] = kmeans.labels_ + 1
        self.create_figures()
        self.plot_data()

    def create_figures(self):
        print('Creating figures')
        self.fig1, self.ax1 = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots()
        self.fig1.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig2.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig3.canvas.mpl_connect('button_press_event', self.onclick)

    def plot_data(self):
        print('Plotting data')
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        colors = get_color_map(self.num_clusters + 1)

        for i in range(self.num_clusters + 1):
            if i == 0:
                unassigned_data = self.df_test[self.df_test['cluster'] == 0]
                self.ax1.scatter(unassigned_data['G'], unassigned_data['S'], label='Unassigned', color=colors[0])
                self.ax2.scatter(unassigned_data['Int 1/(1-4)'], unassigned_data['Int 2/(1-4)'], label='Unassigned',
                                 color=colors[0])
                self.ax3.scatter(unassigned_data['Int 1/(1-4)'], unassigned_data['Int 3/(1-4)'], label='Unassigned',
                                 color=colors[0])
            else:
                cluster_data = self.df_test[self.df_test['cluster'] == i]
                self.ax1.scatter(cluster_data['G'], cluster_data['S'], label=f'{self.test_FPs[i - 1]}', color=colors[i])
                self.ax2.scatter(cluster_data['Int 1/(1-4)'], cluster_data['Int 2/(1-4)'],
                                 label=f'{self.test_FPs[i - 1]}', color=colors[i])
                self.ax3.scatter(cluster_data['Int 1/(1-4)'], cluster_data['Int 3/(1-4)'],
                                 label=f'{self.test_FPs[i - 1]}', color=colors[i])

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.legend()
            ax.set_xlabel('G' if ax == self.ax1 else 'Int 1/(1-4)')
            ax.set_ylabel('S' if ax == self.ax1 else ('Int 2/(1-4)' if ax == self.ax2 else 'Int 3/(1-4)'))

        self.fig1.canvas.draw()
        self.fig2.canvas.draw()
        self.fig3.canvas.draw()

        plt.ion()
        plt.show()
        plt.pause(0.001)
        print('Data plotted')

    def onselect(self, verts):
        self.log('Polygon selected')
        print('Polygon selected')
        path = Path(verts)
        if self.lasso.ax == self.ax1:
            print('Lasso on ax1')
            self.log(f'Lasso on ax1 activated, assign class: {self.selected_class}')
            coords = self.df_test[['G', 'S']].to_numpy()
        elif self.lasso.ax == self.ax2:
            self.log(f'Lasso on ax2 activated, assign class: {self.selected_class}')
            coords = self.df_test[['Int 1/(1-4)', 'Int 2/(1-4)']].to_numpy()
        elif self.lasso.ax == self.ax3:
            self.log(f'Lasso on ax3 activated, assign class: {self.selected_class}')
            coords = self.df_test[['Int 1/(1-4)', 'Int 3/(1-4)']].to_numpy()
        else:
            print('Lasso not on any axes')
            self.log('Lasso not on any axes, please click on the plot to activate Lasso Selector')
            return
        print(f'Verts: {verts}')
        ind = np.nonzero(path.contains_points(coords))[0]
        self.df_test.loc[ind, 'cluster'] = self.selected_class
        print(f'Points reassigned to cluster {self.selected_class}')
        self.log(f'Points reassigned to cluster {self.selected_class}')
        self.lasso = None
        self.plot_data()

    def onclick(self, event):
        if event.key == 'control' and event.button == 1:  # Ctrl + Click
            ix, iy = event.xdata, event.ydata
            if ix is not None and iy is not None:
                if event.inaxes == self.ax1:
                    coords = self.df_test[['G', 'S']].to_numpy()
                elif event.inaxes == self.ax2:
                    coords = self.df_test[['Int 1/(1-4)', 'Int 2/(1-4)']].to_numpy()
                elif event.inaxes == self.ax3:
                    coords = self.df_test[['Int 1/(1-4)', 'Int 3/(1-4)']].to_numpy()
                else:
                    return

                distances = cdist([(ix, iy)], coords)
                closest_index = np.argmin(distances)
                closest_point = coords[closest_index]

                self.df_test.at[closest_index, 'cluster'] = self.selected_class
                print(f'Point reassigned to cluster {self.selected_class}')
                self.log(f'Point reassigned to cluster {self.selected_class}')
                self.plot_data()

        elif event.key == 'shift' and event.button == 1:  # Shift + Click to start Lasso Selector
            if self.lasso is not None:
                self.lasso.disconnect_events()
            self.lasso = LassoSelector(event.inaxes, onselect=self.onselect)
            print('Lasso Selector activated')
            self.log('Lasso Selector activated')

    # to use this, user should be on the GUI window, not plotting window
    def onkey(self, event):
        print(f'Key pressed: {event.char}')
        # if event.state & 4:  # 4 is the shift key
        if event.char.isdigit() and int(event.char) < self.num_clusters + 1:
            self.selected_class = int(event.char)
        elif event.char == 'a':
            self.selected_class = 10
        elif event.char == 'b':
            self.selected_class = 11
        elif event.char == 'c':
            self.selected_class = 12
        elif event.char == 'd':
            self.selected_class = 13
        elif event.char == 'e':
            self.selected_class = 14
        print(f'Selected class: {self.selected_class}')
        self.log(f'Assign next to class {self.selected_class}')
    def save_results(self):
        # save each of the test files to a new excel, only save FOV, Mask label, and cluster
        for file in self.test_data:
            df_save = self.df_test[self.df_test['file'] == file][['FOV', 'Mask label', 'cluster']]
            save_path = os.path.join(self.test_base_folder, file, 'clustered.xlsx')
            df_save.to_excel(save_path, index=False)
            print(f'Saved clustered data to {save_path}')
            self.log(f'Saved classified data to {save_path}.')
            self.log(f'Now saving mask to tif files.')
            # color the corresponding cells using npy segmentation image, and save it to tif files, one is mask from 1-14, the other is mask with color from 1-14
            int_folder = os.path.join(self.test_base_folder, file, 'intensity')
            for fov in df_save['FOV'].unique():
                mask_path = os.path.join(int_folder, f'{fov}-sum_seg.npy')
                mask_cp = np.load(mask_path, allow_pickle=True).item()['masks'] # mask from cellpose
                mask_cp_eroded = np.zeros_like(mask_cp)
                # erode mask_cp to separate cells
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
                    # make the bbox a bit larger
                    minr = max(0, minr - 1)
                    minc = max(0, minc - 1)
                    maxr = min(mask_region.shape[0], maxr + 1)
                    maxc = min(mask_region.shape[1], maxc + 1)

                    cropped_region = mask_region[minr:maxr, minc:maxc]
                    eroded_cropped = erosion(cropped_region, erosion_disk)
                    # plot cropped region before and after erosion
                    # plt.figure()
                    # plt.imshow(cropped_region, cmap='gray')
                    # plt.imshow(eroded_cropped, cmap='gray', alpha=0.5)
                    # plt.show(block=True)
                    # Put the eroded cropped region back to the full mask
                    mask_cp_eroded[minr:maxr, minc:maxc][eroded_cropped] = mask_label

                print('erosion done')


                # plot mask cp before and after erosion
                # plt.figure()
                # plt.imshow(mask_cp, cmap='gray')
                # plt.imshow(mask_cp_eroded, cmap='gray', alpha=0.5)
                # plt.show(block=True)
                # save mask cp to tif
                df_save['cluster'] = df_save['cluster'].astype(np.uint8)
                df_save['FP'] = df_save['cluster'].apply(lambda x: self.test_FPs[x-1] if x != 0 else 'Unassigned')
                mask = np.zeros_like(mask_cp)

                for idx, row in df_save[df_save['FOV'] == fov].iterrows():
                    mask[mask_cp_eroded == row['Mask label']] = row['cluster']
                print(f'unique mask values: {np.unique(mask)}')
                mask_color = get_color_map(self.num_clusters + 1)
                # make the first color to black, not gray
                if self.background_color == 'black':
                    mask_color[0] = (0, 0, 0)
                elif self.background_color == 'white':
                    mask_color[0] = (1, 1, 1)
                mask_color = np.array(mask_color)
                mask_color = mask_color[mask]
                mask = mask.astype(np.uint8)

                tiff.imwrite(os.path.join(int_folder, f'{fov}-cls.tif'), mask)
                print(f'Saved mask to {int_folder}')
                self.log(f'Saved classified mask in classes numbers to {int_folder}, for Napari visualization.')
                # save mask_color to RGB tif, 255 8 bit for each channel
                mask_color = mask_color * 255
                mask_color = mask_color.astype(np.uint8)
                # add text to the mask_color, each class with FPs' text label, say class 1 is mScarlet3, class 2 is mScarlet-I3, etc., in white color, add it to the center of the cell
                if self.add_text:
                    for idx, row in df_save[df_save['FOV'] == fov].iterrows():
                        cls = row['cluster']
                        if cls != 0:
                            mask_idx = row['Mask label']
                            mask_uint8 = np.where(mask_cp == mask_idx, 255, 0).astype(np.uint8)
                            # draw white outline
                            # contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            # cv2.drawContours(mask_color, contours, -1, (255, 255, 255), 1)
                            # cv2.putText(mask_color, f"{self.FPs[cls]}", (contours[0][0][0][0], contours[0][0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1)
                            center = regionprops(mask_uint8)[0].centroid
                            if self.background_color == 'black':
                                cv2.putText(mask_color, f"{self.test_FPs[cls-1]}", (int(center[1]), int(center[0])), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), 1)
                            elif self.background_color == 'white':
                                cv2.putText(mask_color, f"{self.test_FPs[cls-1]}", (int(center[1]), int(center[0])), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 1)



                tiff.imwrite(os.path.join(int_folder, f'{fov}-cls-color.tif'), mask_color)
                print(f'Saved mask color to {int_folder}')
                if self.add_text:
                    self.log(f'Saved classified mask with text to {int_folder}, for direct visualization and double check.')
                else:
                    self.log(f'Saved classified mask with color to {int_folder}, for direct visualization.')



# Tkinter root
root = tk.Tk()
app = ScatterGUI(root)
root.mainloop()