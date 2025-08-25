import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Use TkAgg backend for embedding in Tkinter
plt.switch_backend('TkAgg')


def process_directory(directory, base_folder, base_folder2, confidence=0.6):
    """
    Reads confident and uncertain prediction Excel files for a given directory,
    prints per-FOV detection stats, and returns the confident dataframe along
    with counts of confident and uncertain cells.
    """
    # Determine the full path for this directory
    full_path = os.path.join(base_folder, directory)
    if not os.path.exists(full_path):
        full_path = os.path.join(base_folder2, directory)
    if not os.path.exists(full_path):
        print(f"Directory not found: {full_path}")
        return None, 0, 0

    # File paths for confident and uncertain predictions
    conf_path = os.path.join(full_path, f"predict_class_confident_{confidence}.xlsx")
    unc_path = os.path.join(full_path, f"predict_class_uncertain_{confidence}.xlsx")

    # Read confident predictions
    if not os.path.exists(conf_path):
        print(f"Confident Excel file not found in: {full_path}")
        return None, 0, 0
    df_conf = pd.read_excel(conf_path)

    # Read uncertain predictions, if available
    if os.path.exists(unc_path):
        df_unc = pd.read_excel(unc_path)
        total_unc = len(df_unc)
    else:
        total_unc = 0
        print(f"Uncertain Excel file not found in: {full_path}")

    # Add directory column and counts
    df_conf['directory'] = directory
    total_conf = len(df_conf)
    total_cells = total_conf + total_unc
    detection_rate = total_conf / total_cells if total_cells > 0 else 0

    # Print per-FOV stats
    print(f"{directory}: {total_conf}/{total_cells} confident (Detection rate: {detection_rate:.2%})")

    return df_conf, total_conf, total_unc


def create_heatmap(data, nu_FPs, mito_FPs, detection_rate):
    """
    Builds and returns a heatmap figure/axis for Nu vs Mito FP counts,
    and includes overall detection rate in the title.
    """
    # Initialize count matrix
    heatmap_data = pd.DataFrame(0, index=nu_FPs, columns=mito_FPs)
    for _, row in data.iterrows():
        nu_fp = row['Predicted_Nu_Class']
        mito_fp = row['Predicted_Mito_Class']
        if nu_fp in nu_FPs and mito_fp in mito_FPs:
            heatmap_data.loc[nu_fp, mito_fp] += 1

    print(f"Total confident cells plotted: {len(data)}")

    # Custom diverging colormap
    # colors = ['blue', 'white', 'red']
    # cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    # use blues as heatmap
    cmap = plt.get_cmap('Blues')
    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(heatmap_data, dtype=bool)
    np.fill_diagonal(mask, True)

    # Plot off-diagonal with blue-white-red
    import seaborn as sns
    sns.heatmap(
        heatmap_data, annot=True, fmt='d', cmap=cmap,
        annot_kws={'size': 14}, cbar_kws={'shrink': .8},
        mask=mask, ax=ax
    )
    # Plot diagonal in gray
    sns.heatmap(
        heatmap_data, annot=True, fmt='d', cmap='gray',
        annot_kws={'size': 14}, cbar=False,
        mask=~mask, ax=ax
    )

    # Set labels and title with detection rate
    ax.set_title(
        f"Heatmap of Nu_FP and Mito_FP Combinations"
        f"\nOverall Detection Rate: {detection_rate:.2%}",
        fontsize=18
    )
    ax.set_xlabel('Mito FP', fontsize=16)
    ax.set_ylabel('Nu FP', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    # save to pdf
    output_pdf = 'heatmap_nu_mito.pdf'
    fig.savefig(output_pdf, format='pdf')
    print(f"Heatmap saved to {output_pdf}")
    return fig, ax


def print_distribution(data, nu_fp, mito_fp):
    """
    Prints the distribution of cells across directories for a given Nu/Mito FP combination.
    """
    filtered = data[
        (data['Predicted_Nu_Class'] == nu_fp) &
        (data['Predicted_Mito_Class'] == mito_fp)
    ]
    counts = filtered['directory'].value_counts()

    print(f"\nCell Distribution for {nu_fp}-{mito_fp}")
    print("-" * 40)
    for d, c in counts.items():
        print(f"{d}: {c} cells")
    print("-" * 40)
    print(f"Total Cells: {counts.sum()}")
    print(f"FOVs with this combo: {len(counts)}")


class HeatmapGUI:
    def __init__(self, master, data, nu_FPs, mito_FPs, detection_rate):
        self.master = master
        self.data = data
        self.nu_FPs = nu_FPs
        self.mito_FPs = mito_FPs
        self.detection_rate = detection_rate

        # Window title includes detection rate
        self.master.title(f"Interactive Heatmap (Detection: {detection_rate:.2%})")
        self.master.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Create the heatmap figure/axis
        self.fig, self.ax = create_heatmap(
            self.data, self.nu_FPs, self.mito_FPs, self.detection_rate
        )

        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Bind click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes == self.ax:
            col = int(event.xdata)
            row = int(event.ydata)
            if 0 <= col < len(self.mito_FPs) and 0 <= row < len(self.nu_FPs):
                nu_fp = self.nu_FPs[row]
                mito_fp = self.mito_FPs[col]
                print_distribution(self.data, nu_fp, mito_fp)


def main():
    # Base folders for data
    base_folder = r'G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual'
    base_folder2 = r'E:\BC-FLIM\Hek293T-BJMU-Dual'

    # List of directories (FOVs)
    directories = [
        # 'Mix7-241205-1', 'Mix7-241205-2', 'Mix7-241205-40X-3',
        # 'Mix7-241205-40X-4', 'Mix7-2-250103-1', 'Mix7-2-250103-2',
        # 'Mix7-1-250103-1', 'Mix7-1-250103-2'
        # 'Mix42-250601-1',
        # 'Mix42-250601-2',
        # 'Mix42-250602-1',
        # 'Mix42-250602-2',
        # 'Mix42-250602-3',
        # 'Mix42-250602-4',
        # 'Mix42-250602-5',
        # 'Mix42-250602-6',

        # 'Mix35-250616-1',
        # 'Mix35-250616-2',
        'Mix35-250616-3',
        # 'Mix35-250616-4',
        # 'Mix35-250616-5',
        # 'Mix35-250616-6',
        # 'Mix36-250624-1',
        # 'Mix36-250624-2',
        # 'Mix36-250624-3',
        # 'Mix36-250624-4',
        # 'Mix36-250624-5',
        # 'Mix36-250624-6',
        # 'Mix36-250624-7',
        # 'Mix36-250624-8',
        # 'Mix36-250624-9',
        # 'Mix36-250624-10',
        # 'Mix36-250624-11',
        # 'Mix36-250624-12',
        # 'Mix36-250624-13',
        # 'Mix36-250624-14',
        # 'Mix36-250624-15',
        # 'Mix36-250624-16',
        # 'Mix36-250624-17',
    ]

    # FP class labels
    nu_FPs = ['N10', 'N13', 'N4', 'N14', 'N16', 'N8', 'N1']
    mito_FPs = ['M10', 'M13', 'M4', 'M14', 'M16', 'M8', 'M1']

    # Aggregate data and stats
    all_data = pd.DataFrame()
    total_conf = 0
    total_unc = 0
    for d in directories:
        df_conf, conf_cnt, unc_cnt = process_directory(
            d, base_folder, base_folder2
        )
        if df_conf is not None:
            all_data = pd.concat([all_data, df_conf], ignore_index=True)
            total_conf += conf_cnt
            total_unc += unc_cnt

    # Compute overall detection rate
    overall_total = total_conf + total_unc
    overall_rate = total_conf / overall_total if overall_total > 0 else 0
    print(f"Overall: {total_conf}/{overall_total} confident (Detection rate: {overall_rate:.2%})")

    # Launch GUI
    root = tk.Tk()
    app = HeatmapGUI(root, all_data, nu_FPs, mito_FPs, overall_rate)
    root.mainloop()


if __name__ == '__main__':
    main()
