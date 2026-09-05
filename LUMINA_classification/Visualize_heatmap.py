"""Plot the Nu x Mito barcode-combination heatmap from LUMINA prediction workbooks.

Reads the `predict_class_confident_<threshold>.xlsx` / `predict_class_uncertain_<threshold>.xlsx`
pairs that `Test_LUMINA.py` writes back into each sample folder, aggregates them, prints
the per-sample and overall detection rate, and draws one heatmap of nuclear barcode call
against mitochondrial barcode call.

THE HEATMAP IS NOT A CONFUSION MATRIX. Both axes are PREDICTIONS -- rows are the predicted
nuclear barcode, columns the predicted mitochondrial barcode. There is no ground truth in
this figure. The grey diagonal marks the cells whose two calls are the same fluorophore
index on both anchors; it is only meaningful while the row list and the column list are in
matching order.

By default the script opens an interactive Tk window in which clicking a cell prints the
per-sample breakdown of that combination, and writes the same figure to a PDF. Pass
--no-gui on a machine with no display: tkinter and the TkAgg backend are imported only on
the GUI path, so importing this module is side-effect free.

Usage:
    python Visualize_heatmap.py --data-root /path/to/dataset
    python Visualize_heatmap.py --data-root /path/to/dataset --no-gui \
        --samples sampleA,sampleB --confidence-threshold 0.6 --out-pdf ./heatmap.pdf
"""

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Class name -> class index, character for character Finetune_LUMINA.py's NU_CLASS_MAP and
# MITO_CLASS_MAP (and Train_LUMINA.py's, and Test_LUMINA.py's). The KEY ORDER is what this
# script uses as the heatmap's row and column order, and it is the same order the other
# three assign integer class indices from. The checkpoint stores no mapping, so a
# permutation here is silent. If you change the panel, change it in all four places.
NU_CLASS_MAP = {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
MITO_CLASS_MAP = {'M10': 1, 'M13': 2, 'M4': 3, 'M14': 4, 'M16': 5, 'M8': 6, 'M1': 7}

# Filename template Test_LUMINA.py builds, in the two to_excel calls at the end of its
# test_model(). The threshold is formatted with str() on both sides, so both scripts must
# hold it as a float or the two names differ ('0.6' vs '0.60') and this script silently
# finds nothing.
CONFIDENT_XLSX = 'predict_class_confident_%s.xlsx'
UNCERTAIN_XLSX = 'predict_class_uncertain_%s.xlsx'


def comma_list(value):
    """Split a comma-separated flag value; the idiom Finetune_LUMINA.py's scan_population()
    uses to read its own --samples."""
    return [s.strip() for s in value.split(',') if s.strip()]


def require_dir(path, flag):
    """Fail immediately, naming the flag, when a user-supplied folder is not usable."""
    if not os.path.isdir(path):
        raise SystemExit('%s does not exist or is not a folder: %s' % (flag, path))


def discover_samples(base_folder, base_folder2, confidence):
    """Sample folders under either root that already hold a confident workbook."""
    wanted = CONFIDENT_XLSX % confidence
    found = []
    for root in (base_folder, base_folder2):
        if not root:
            continue
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if name not in found and os.path.isdir(full) \
                    and os.path.exists(os.path.join(full, wanted)):
                found.append(name)
    return sorted(found)


def process_directory(directory, base_folder, base_folder2, confidence=0.6):
    """
    Reads confident and uncertain prediction Excel files for a given directory,
    prints per-FOV detection stats, and returns the confident dataframe along
    with counts of confident and uncertain cells.

    `confidence` is passed explicitly by main() from --confidence-threshold; the default
    here is kept only so the function is callable on its own. Editing it does not change
    what a command line run reads.
    """
    # Determine the full path for this directory
    full_path = os.path.join(base_folder, directory)
    # `base_folder2` is empty when no --data-root-2 was given; os.path.join('', directory)
    # would otherwise become a relative path that can accidentally hit the cwd.
    if base_folder2 and not os.path.exists(full_path):
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


def create_heatmap(data, nu_FPs, mito_FPs, detection_rate, output_pdf='heatmap_nu_mito.pdf'):
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
    def __init__(self, master, data, nu_FPs, mito_FPs, detection_rate,
                 output_pdf='heatmap_nu_mito.pdf'):
        self.master = master
        self.data = data
        self.nu_FPs = nu_FPs
        self.mito_FPs = mito_FPs
        self.detection_rate = detection_rate
        self.output_pdf = output_pdf

        # Window title includes detection rate
        self.master.title(f"Interactive Heatmap (Detection: {detection_rate:.2%})")
        self.master.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # tkinter and the Tk canvas backend are imported here rather than at module scope:
        # at module scope they made this file fail on IMPORT on a machine with no display,
        # before argparse could even print --help.
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter as tk

        # Create the heatmap figure/axis
        self.fig, self.ax = create_heatmap(
            self.data, self.nu_FPs, self.mito_FPs, self.detection_rate, self.output_pdf
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
    ap = argparse.ArgumentParser(
        description='Plot the Nu x Mito barcode-combination heatmap from the confident '
                    'prediction workbooks Test_LUMINA.py wrote.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--data-root', required=True,
                    help='Root holding one sample folder per acquisition, each containing '
                         'the workbooks Test_LUMINA.py wrote: <root>/<sample>/'
                         'predict_class_confident_<threshold>.xlsx and the matching '
                         '_uncertain_ one. This must be the SAME root Test_LUMINA.py was '
                         'given, because it writes its results back into the sample folder.')
    ap.add_argument('--data-root-2', default='',
                    help='Optional second root, searched only when a sample folder is not '
                         'found under --data-root. Empty means there is no second root. A '
                         'dataset split across two roots is a common cause of "Confident '
                         'Excel file not found"; prefer keeping one root complete.')
    ap.add_argument('--samples', default='',
                    help='Comma-separated sample folder names to plot, in the order given. '
                         'Empty means every folder under --data-root (and --data-root-2 if '
                         'given) that already holds a confident workbook at this threshold. '
                         'The shipped script instead plotted whatever short list was left '
                         'uncommented in main(), so pass --samples to reproduce a particular '
                         'figure rather than relying on the scan.')
    ap.add_argument('--out-pdf', default='heatmap_nu_mito.pdf',
                    help='Where the heatmap PDF is written. A bare filename lands in the '
                         'current working directory and is overwritten on every run. The '
                         'resolved configuration is written beside it as '
                         'visualize_heatmap_run_config.csv.')

    ap.add_argument('--nu-classes', default='',
                    help='Comma-separated nuclear barcode names, in the order they become '
                         'the heatmap ROWS. Empty means the NU_CLASS_MAP order at the top of '
                         'this file, which is the order Train/Test/Finetune number the '
                         'classes in. The list also FILTERS: a cell predicted into a name '
                         'that is not listed is left out of the matrix while still counting '
                         'in the printed "Total confident cells plotted", so those two '
                         'numbers can legitimately disagree.')
    ap.add_argument('--mito-classes', default='',
                    help='Comma-separated mitochondrial barcode names, in the order they '
                         'become the heatmap COLUMNS. Empty means the MITO_CLASS_MAP order. '
                         'Filters as well as orders, exactly like --nu-classes. The grey '
                         'diagonal only means "same fluorophore on both anchors" while this '
                         'list and --nu-classes are in matching order; reordering one and '
                         'not the other makes the diagonal meaningless with no error.')

    ap.add_argument('--confidence-threshold', type=float, default=0.6,
                    help='The confidence gate Test_LUMINA.py was run with. Nothing is gated '
                         'here -- this value is part of the workbook FILENAME, so it must '
                         'equal the value used there or no workbook is found. '
                         'Test_LUMINA.py defaults to the same 0.6; Finetune_LUMINA.py '
                         'defaults to 0.9 under the same flag name, because it is a '
                         'different measurement on the same scale, not a typo.')

    ap.add_argument('--no-gui', action='store_true',
                    help='Write the PDF and exit instead of opening the interactive Tk '
                         'window. The window is what lets you click a cell to print its '
                         'per-sample breakdown; the PDF is the same either way. Required on '
                         'a machine with no display.')

    args = ap.parse_args()

    # FP class labels: the rows and the columns of the figure.
    nu_FPs = comma_list(args.nu_classes) or list(NU_CLASS_MAP)
    mito_FPs = comma_list(args.mito_classes) or list(MITO_CLASS_MAP)
    if len(set(nu_FPs)) != len(nu_FPs):
        raise SystemExit('--nu-classes repeats a name: %s\n'
                         'Each row of the heatmap must be a distinct barcode.'
                         % ', '.join(nu_FPs))
    if len(set(mito_FPs)) != len(mito_FPs):
        raise SystemExit('--mito-classes repeats a name: %s\n'
                         'Each column of the heatmap must be a distinct barcode.'
                         % ', '.join(mito_FPs))

    conf_name = CONFIDENT_XLSX % args.confidence_threshold
    print('nu_class_map: %s' % NU_CLASS_MAP)
    print('mito_class_map: %s' % MITO_CLASS_MAP)
    print('rows (nu): %s' % ', '.join(nu_FPs))
    print('columns (mito): %s' % ', '.join(mito_FPs))
    print('confidence threshold: %s   workbook: %s   gui: %s'
          % (args.confidence_threshold, conf_name, 'off' if args.no_gui else 'on'))

    # Base folders for data
    require_dir(args.data_root, '--data-root')
    if args.data_root_2:
        require_dir(args.data_root_2, '--data-root-2')

    roots = [args.data_root] + ([args.data_root_2] if args.data_root_2 else [])
    roots_text = ' and '.join(roots)

    # List of directories (FOVs)
    directories = comma_list(args.samples)
    if directories:
        missing = [d for d in directories
                   if not any(os.path.isdir(os.path.join(r, d)) for r in roots)]
        if missing:
            raise SystemExit(
                'These --samples are not folders under %s: %s\n'
                'Give the sample folder names as they appear on disk, or drop --samples to '
                'plot every folder that holds a %s.'
                % (roots_text, ', '.join(missing), conf_name))
    else:
        directories = discover_samples(args.data_root, args.data_root_2,
                                       args.confidence_threshold)
        if not directories:
            raise SystemExit(
                'No folder under %s holds a %s.\n'
                'Check that --confidence-threshold (%s) is the value Test_LUMINA.py was run '
                'with, and that --data-root is the root it wrote into -- it writes its '
                'workbooks back into each sample folder.'
                % (roots_text, conf_name, args.confidence_threshold))
        print('samples: %d folder(s) found under %s' % (len(directories), roots_text))

    # Aggregate data and stats
    all_data = pd.DataFrame()
    total_conf = 0
    total_unc = 0
    for d in directories:
        df_conf, conf_cnt, unc_cnt = process_directory(
            d, args.data_root, args.data_root_2, args.confidence_threshold
        )
        if df_conf is not None:
            all_data = pd.concat([all_data, df_conf], ignore_index=True)
            total_conf += conf_cnt
            total_unc += unc_cnt

    if len(all_data) == 0:
        raise SystemExit(
            'No confident predictions were read from %d sample folder(s) under %s, so there '
            'is nothing to plot.\n'
            'Looked for %s in each. Check --confidence-threshold (%s) against the value '
            'Test_LUMINA.py was run with, and check that those folders have been scored.'
            % (len(directories), roots_text, conf_name, args.confidence_threshold))

    # Compute overall detection rate
    overall_total = total_conf + total_unc
    overall_rate = total_conf / overall_total if overall_total > 0 else 0
    print(f"Overall: {total_conf}/{overall_total} confident (Detection rate: {overall_rate:.2%})")

    # Every flag, resolved, written once beside the PDF. This script writes its figure in
    # place rather than into an --out directory, so the config goes next to the figure.
    out_dir = os.path.dirname(os.path.abspath(args.out_pdf))
    os.makedirs(out_dir, exist_ok=True)
    cfg = {k: v for k, v in sorted(vars(args).items())}
    cfg['resolved_nu_classes'] = ','.join(nu_FPs)
    cfg['resolved_mito_classes'] = ','.join(mito_FPs)
    cfg['resolved_samples'] = ','.join(directories)
    cfg['resolved_out_pdf'] = os.path.abspath(args.out_pdf)
    cfg['confident_workbook'] = conf_name
    cfg['uncertain_workbook'] = UNCERTAIN_XLSX % args.confidence_threshold
    cfg['total_confident'] = total_conf
    cfg['total_cells'] = overall_total
    cfg['detection_rate'] = overall_rate
    pd.DataFrame([{'flag': k, 'value': v} for k, v in cfg.items()]).to_csv(
        os.path.join(out_dir, 'visualize_heatmap_run_config.csv'), index=False)

    if args.no_gui:
        # The PDF is written inside create_heatmap, which the GUI otherwise reaches only
        # through HeatmapGUI.create_widgets. Call it directly or --no-gui produces nothing
        # while appearing to succeed.
        fig, _ax = create_heatmap(all_data, nu_FPs, mito_FPs, overall_rate, args.out_pdf)
        plt.close(fig)
    else:
        # Use TkAgg backend for embedding in Tkinter
        import tkinter as tk
        plt.switch_backend('TkAgg')

        # Launch GUI
        root = tk.Tk()
        app = HeatmapGUI(root, all_data, nu_FPs, mito_FPs, overall_rate, args.out_pdf)
        root.mainloop()


if __name__ == '__main__':
    main()
