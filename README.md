# BC‑FLIM Tools

This repository contains two complementary tools for barcode analysis in FLIM experiments:

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

## Notes

- These tools are under active development.  
- The manuscript describing the methods has not yet been published. Once released, DOI and Zenodo links will be provided here.  
- Example/test datasets will also be deposited alongside the Zenodo record.

---

**Enjoy using BC‑FLIM Tools!**
