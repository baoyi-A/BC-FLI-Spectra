# 🔬 SLIC (Spectral-Lifetime Indexing of Cells) Tools

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22228957.svg)](https://doi.org/10.5281/zenodo.22228957)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](Napari_plugin/LICENSE)

This repository contains two independent tools for barcode analysis in FLIM experiments:

Each tagged release is archived on Zenodo. The DOI above always resolves to the
most recent version; to cite the exact version described in the manuscript, use
the version DOI listed on the Zenodo record.

## 1. 🧩 Napari Plugin — Single‑Anchor Barcode Analysis

A napari plugin named **BC‑FLIM‑Spectra (NaCha)** that provides a full workflow for single‑anchor barcode analysis, from `.ptu` data ingestion through segmentation, classification, tracking to alignment and visualization.

It exposes **seven widgets** under `Plugins → BC‑FLIM‑Spectra`:

- 📥 **PTU Reader** — import and decode FLIM `.ptu` files.
- 🔬 **Barcode Seg (Cellpose)** — N / P segmentation on the barcode intensity image, with online single‑ or multi‑folder fine‑tuning.
- 🌀 **Calculate FLIM‑S** — lifetime / phasor computation.
- 🧩 **Seeded K-Means** — semi-supervised barcode classifier (Basu et al. 2002 — seeds initialise centroids, then K-Means refines). Also ships K-Means++, MiniBatchKMeans, Gaussian Mixture, Spectral as alternatives; per-class outlier flagging; save / load of class distribution overlays.
- 🟡 **Biosensor Seg (Cellpose)** — dual‑input segmentation on the confocal biosensor stack using the barcode classification mask as an auxiliary channel.
- 🎬 **B&P Tracker** — barcode / object tracking (built on Track‑Anything / XMem).
- 📈 **NaCha** — final alignment and per‑class signal readout / visualization.

For details, see [Napari plugin/README](Napari_plugin/README.md).

---

## 2. 🧠 LUMINA — Dual‑Anchor Barcode Classification Network

**LUMINA** is a PyTorch‑based deep learning framework for **dual‑anchor barcodes classification**.
It provides scripts for preprocessing, training, inference, and visualization:

- 🧹 `Data_Prep.py` — preprocess raw data.
- 🏋️ `Train_LUMINA.py` — train the LUMINA model.
- 🔍 `Test_LUMINA.py` — inference on new data.
- 🔥 `Visualize_heatmap.py` — visualize results.

For details, see [LUMINA classification/README](LUMINA_classification/README.md).

---

## 🖥 Environment
- Python >= 3.8 (tested on Python 3.10)
- Operating system:
  - Tested on: Windows 11
  - Expected to work on: Windows 10/11, Linux (Ubuntu 20.04)
- GPU: optional (CUDA-enabled GPU recommended for acceleration)

**Typical installation time:** ~20–60 minutes depending on whether PyTorch/CUDA wheels
need to be downloaded and the network speed.

---

## ⏱ Expected runtime (representative)

### 🧩 Napari Plugin (interactive)
- Typical processing time per dataset: ~10–15 minutes
- Tracking step: ~5 minutes (included above)
- Runtime depends on dataset size, number of cells, and hardware.

### 🧠 LUMINA
- Inference on a single sample/image: ~1 second
- Including preprocessing (e.g., segmentation / data extraction): typically 3 minutes
- Model training (if performed) can take longer and depends on dataset size and GPU.

---

## 🤖 Using this repository with an AI assistant

Both tools ship machine-readable instructions, so a general-purpose coding
assistant can install the software and answer questions about it without
being told anything else:

| File | For |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Coding agents — layout, install, conventions, testing pattern. Follows the [agents.md](https://agents.md) convention; Claude Code picks it up via the one-line `CLAUDE.md`. |
| [`.claude/skills/bc-flim-spectra/`](.claude/skills/bc-flim-spectra/) | Usage — the seven steps control by control, what every parameter means, and the failure modes with the check for each. [Agent Skills](https://agentskills.io) format. |

Point your assistant at the repository and ask it, for example, *"install this
and open Calculate FLIM-S"* or *"why does my FLIM-S.xlsx contain rows I did not
compute"*. The parameter descriptions in the skill are the same strings the
widgets show as tooltips, so the two cannot drift apart.

## 📜 License
BSD 2-Clause License. See [`Napari_plugin/LICENSE`](Napari_plugin/LICENSE) for the full text.

---

## 📝 Notes

- These tools are under active development.  
- The manuscript describing the methods has been submitted but not yet published. The DOI will be provided once it becomes available.  
- An [instruction video](https://zenodo.org/records/17045806) is available, providing a step-by-step guide on how to use the Napari plugin.  
- The [Dual-Anchor dataset](https://zenodo.org/records/17036213), used for training the LUMINA network, is also provided.  
- A [demo dataset](https://zenodo.org/records/16940026) is included for testing the Napari plugin functionalities.  
- The [original version of all software code](https://zenodo.org/records/17018436) has been archived as well.  

---

**Enjoy using BC‑FLIM Tools!**
