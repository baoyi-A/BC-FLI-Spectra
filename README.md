# 🔬 SLIC — Spectral-Lifetime Indexing of Cells

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22228957.svg)](https://doi.org/10.5281/zenodo.22228957)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](Napari_plugin/LICENSE)

This repository contains two independent tools for barcode analysis in FLIM experiments:

Each tagged release is archived on Zenodo. The DOI above always resolves to the
most recent version; to cite the exact version described in the manuscript, use
the version DOI listed on the Zenodo record.

## 1. 🧩 Napari Plugin — Single‑Anchor Barcode Analysis

The SLIC napari plugin provides the full workflow for single‑anchor barcode analysis, from `.ptu` data ingestion through segmentation, classification and tracking to alignment and visualization. It is distributed as the Python package `bc-flim-spectra` and appears in napari under that name; *NaCha* is the label of its final widget.

It exposes **seven widgets**, listed in napari under `Plugins → bc-flim-spectra`:

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
- Typical processing time per dataset: ~10–15 minutes on a CUDA GPU, most of it
  spent looking at masks rather than waiting.
- Tracking step: ~5 minutes (included above).
- **Check the "Compute:" line in the segmentation panel first.** The default
  PyTorch package has no GPU support, and on CPU segmentation of one 2048 × 2048
  field takes tens of minutes instead of seconds. Nothing fails — it is just
  slow, and the header is the only place that says so.
- Runtime otherwise depends on dataset size, number of cells, and hardware.

### 🧠 LUMINA
- Inference on a single sample/image: ~1 second
- Including preprocessing (e.g., segmentation / data extraction): typically 3 minutes
- Model training (if performed) can take longer and depends on dataset size and GPU.

---

## 🤖 SLIC works with an AI assistant

Seven steps are easy to run and easy to forget. So the repository also carries
its documentation in a form coding assistants read — the
[agents.md](https://agents.md) and [Agent Skills](https://agentskills.io)
conventions, used by Claude Code, Codex, Cursor, Copilot and others. Point one
at this repository and it can install SLIC, walk you through a step, tell you
what a parameter means, and look at your data to say where you are.

### ✅ We tried it, and it works

We gave **Claude Code** nothing but the address of this repository and asked it
to install SLIC and process one dataset. Working only from the files here, it
built a fresh environment, installed the plugin, and ran the workflow through
on a real acquisition:

| Step | What came out |
|---|---|
| 📥 PTU Reader | a 1.5 GB `.ptu` decoded into four spectral channels, the decay stacks and a lifetime map |
| 🔬 Barcode Seg | 418 nuclei and 416 cytoplasm masks on the full 2048 × 2048 field |
| 🌀 Calculate FLIM-S | 834 cells, each reduced to its 5‑D fingerprint |
| 🧩 Seeded K-Means | 373 cells sorted into 10 barcodes; 45 declined as outliers |
| 🟡 Biosensor Seg | 342 cells in the confocal channel, against 349 in our own stored result |
| 📈 NaCha | one signal curve per barcode across 101 frames |

The registration was checked rather than assumed: with the correct 90°
rotation, 96 % of biosensor cells carry a barcode class at 99 % purity, and the
three wrong rotations were run as controls and fall apart, as they should.

Two honest notes. Steps 1–4 above used a public Cellpose model, because our
fine-tuned weights are distributed separately from the code. And the exercise
was worth doing for its own sake: it turned up a dozen real defects — including
a packaging error that stopped `pip install` outright, and a failed
segmentation that hung the interface with no message — all fixed here.

### 🔍 "Where am I in the workflow?"

`Napari_plugin/scripts/workflow_status.py` reads a sample folder and reports
which steps have run, quality checks on what they produced, and what is left.
Read-only; no napari, no GPU.

```bash
python Napari_plugin/scripts/workflow_status.py <sample folder>
```

```
  [   done] 3 Calculate FLIM-S   rows=834, fovs=1, localisations={'N': 418, 'P': 416}
  [   done] 4 Seeded K-Means     clustered=418, classified=373, declined=45, classes=10
  [not run] 5 Biosensor Seg
  [  check] registration         best=90 CW, coverage=0.966, purity=0.998

  PASS  S stays under the semicircle apex (0.5) — max S = 0.4997
  PASS  lifetimes are physical (0.1-10 ns) — 1.24-3.80 ns
  PASS  barcode classes land on biosensor cells — 96.6% of cells, purity 99.8%

  Next
  - Seeded K-Means: 416 row(s) were never clustered — it runs one localisation
    at a time and only ['N'] has been done.
  - Steps 5-7 (biosensor) have not run. They need the confocal B/G/Y stacks.
```

The checks cover the decay stacks expected per field, empty masks, the phasor
coordinates against the universal semicircle, the lifetime range, the declined
fraction against the outlier setting, class sizes, and the registration — for
which it tries all four rotations and reports coverage *and* purity, since
coverage alone cannot tell a correct alignment from a wrong one. `--json` for
scripting; exit status 1 when a check fails, so it works as a gate too.

### 📚 What the assistant reads

| File | Contents |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Repository layout, installation, the three-environment design, the headless testing pattern, and the conventions the code follows. Claude Code picks it up through the one-line `CLAUDE.md`. |
| [`.claude/skills/bc-flim-spectra/`](.claude/skills/bc-flim-spectra/) | The seven steps control by control, how to read each output, the failure modes with the check that identifies each, and installation notes. |

Parameter descriptions in the skill are pulled from the widget tooltips in the
source, so the documentation and the interface cannot drift apart.

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
