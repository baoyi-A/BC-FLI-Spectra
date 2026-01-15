# SLIC (Spectral-Lifetime Indexing of Cells) Tools

This repository contains two independent tools for barcode analysis in FLIM experiments:

## 1. Napari Plugin — Single‑Anchor Barcode Analysis

A napari plugin named **BC‑FLIM‑Spectra (NaCha)** that provides a full workflow for single‑anchor barcode analysis, from `.ptu` data ingestion to alignment and visualization.

It exposes **five widgets** under `Plugins → BC‑FLIM‑Spectra`:

- **PTU Reader** — import and decode FLIM `.ptu` files.  
- **Calculate FLIM‑S** — lifetime/phasor computation.  
- **KMeans Cluster** — interactive clustering for single‑anchor barcodes.  
- **B&P Tracker** — barcode/object tracking widget.  
- **NaCha** — final alignment and visualization output.

For details, see [Napari plugin/README](Napari%20plugin/README.md).

---

## 2. LUMINA — Dual‑Anchor Barcode Classification Network

**LUMINA** is a PyTorch‑based deep learning framework for **dual‑anchor barcodes classification**.  
It provides scripts for preprocessing, training, inference, and visualization:

- `Data_Prep.py` — preprocess raw data.  
- `Train_LUMINA.py` — train the LUMINA model.  
- `Test_LUMINA.py` — inference on new data.  
- `Visualize_heatmap.py` — visualize results.

For details, see [LUMINA classification/README](LUMINA%20classification/README.md).

---

## Environment
- Python >= 3.8 (tested on Python 3.10)
- Operating system:
  - Tested on: Windows 11
  - Expected to work on: Windows 10/11, Linux (Ubuntu 20.04)
- GPU: optional (CUDA-enabled GPU recommended for acceleration)

**Typical installation time:** ~20–60 minutes depending on whether PyTorch/CUDA wheels
need to be downloaded and the network speed.

---

## Expected runtime (representative)

### Napari Plugin (interactive)
- Typical processing time per dataset: ~10–15 minutes
- Tracking step: ~5 minutes (included above)
- Runtime depends on dataset size, number of cells, and hardware.

### LUMINA
- Inference on a single sample/image: ~1 second
- Including preprocessing (e.g., segmentation / data extraction): typically 3 minutes
- Model training (if performed) can take longer and depends on dataset size and GPU.

---

## License
MIT License (recommended for review and reuse).

---

## Notes

- These tools are under active development.  
- The manuscript describing the methods has been submitted but not yet published. The DOI will be provided once it becomes available.  
- An [instruction video](https://zenodo.org/records/17045806) is available, providing a step-by-step guide on how to use the Napari plugin.  
- The [Dual-Anchor dataset](https://zenodo.org/records/17036213), used for training the LUMINA network, is also provided.  
- A [demo dataset](https://zenodo.org/records/16940026) is included for testing the Napari plugin functionalities.  
- The [original version of all software code](https://zenodo.org/records/17018436) has been archived as well.  

---

**Enjoy using BC‑FLIM Tools!**
