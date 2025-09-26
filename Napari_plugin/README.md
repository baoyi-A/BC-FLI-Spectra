# BC‑FLIM‑Spectra (NaCha) — a napari plugin

BC‑FLIM‑Spectra (aka **NaCha**) is a napari plugin that supports an end‑to‑end workflow for FLIM and barcode analysis, from raw **`.ptu`** ingestion to alignment and visualization.

It exposes **five widgets** under the napari menu **`Plugins → BC‑FLIM‑Spectra`**:

1. **PTU Reader** — import and decode FLIM `.ptu` files into usable image stacks/metadata.  
2. **Calculate FLIM‑S** — lifetime/phasor computation and related FLIM analysis utilities.  
3. **KMeans Cluster** — interactive clustering and visualization for **single‑anchor barcodes**.  
4. **B&P Tracker** — tracking widget for barcode/object trajectories (B‑Tracker & P‑Tracker combined).  
5. **NaCha** — final **data alignment** and **readout/visualization** across modalities or acquisition runs.

---

## Installation

A clean environment is recommended.

```bash
# 1) Create & activate environment
conda create -n nacha python=3.10 -y
conda activate nacha

# 2) Install PyTorch (choose the command from the official site for your OS/CUDA)
# https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3) Install TrackAnything (required for tracking functionality)
# Please follow the official instructions at:
# https://github.com/gaomingqi/Track-Anything
# Typically:
git clone https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything
pip install -r requirements.txt

# 4) Install this plugin (editable mode recommended)
cd ../BC-FLIM-Spectra
pip install -e .
```

> Notes  
> • We intentionally do **not** include Qt backends (PyQt/PySide) in the plugin dependencies to avoid conflicts.  
> • If you use a headless/CI system, prefer `opencv-python-headless`; for desktop, keep `opencv-python`.  
> • Make sure your CUDA driver/toolkit matches the PyTorch build you install.

---

## Launch in napari

After installation:
1. Start **napari**.
```bash
conda activate nacha
napari
```
2. Open the menu: **`Plugins → BC‑FLIM‑Spectra`**.  
3. Choose one of the widgets: **PTU Reader**, **Calculate FLIM‑S**, **KMeans Cluster**, **B&P Tracker**, or **NaCha**.

---

## Quick workflow

1. **PTU Reader**: load and decode `.ptu` data.  
2. **Calculate FLIM‑S**: compute lifetime/phasor features for downstream analysis.  
3. **KMeans Cluster**: explore and visualize **single‑anchor barcodes** interactively.  
4. **B&P Tracker**: track objects/barcodes through time.  
5. **NaCha**: align results (across modalities/runs) and generate final visualization/readout.

---

## Troubleshooting

- If napari does not list the plugin, it may be due to missing dependencies. Please install the required packages via pip install <package_name> and then restart napari.  
- For GPU/CUDA issues, verify your PyTorch build and drivers.  
- Some components (e.g., TrackAnything) rely on external model files. If the automatic download fails, please refer to their official documentation for manual installation.

---

**Enjoy NaCha!**
