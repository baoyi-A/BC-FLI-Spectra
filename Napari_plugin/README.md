# 🔬 BC‑FLIM‑Spectra (NaCha) — a napari plugin

BC‑FLIM‑Spectra (aka **NaCha**) is a napari plugin that supports an end‑to‑end
workflow for FLIM and barcode analysis, from raw **`.ptu`** ingestion through
segmentation, classification, tracking and final alignment / visualisation.

It exposes **seven widgets** under the napari menu
**`Plugins → BC‑FLIM‑Spectra`**:

1. 📥 **PTU Reader** — import and decode FLIM `.ptu` files into usable image
   stacks / metadata.
2. 🔬 **Barcode Seg (Cellpose)** — Cellpose‑based nucleus / cytoplasm (N / P)
   segmentation on the barcode intensity‑sum image, with on‑the‑fly user
   editing and **online single‑ or multi‑folder fine‑tuning**.
3. 🌀 **Calculate FLIM‑S** — lifetime / phasor computation and related FLIM
   analysis utilities.
4. 🧩 **Seeded K-Means** — semi-supervised barcode classifier following
   **Basu, Banerjee & Mooney (ICML 2002)**: user‑placed seeds initialise
   the class centroids, then the K‑Means EM loop refines them. Also
   supports K‑Means++, MiniBatchKMeans, Gaussian Mixture, Spectral as
   alternative methods; per‑class outlier flagging; and save / load of
   class distribution overlays as prior knowledge for manual seeding.
5. 🟡 **Biosensor Seg (Cellpose)** — dual‑input Cellpose segmentation on the
   confocal biosensor stack that takes the barcode classification mask as
   an auxiliary channel, biasing segmentation toward barcode‑positive
   cells and boosting detection rate.
6. 🎬 **B&P Tracker** — tracking widget for barcode / object trajectories
   (B‑Tracker & P‑Tracker combined), built on Track‑Anything / XMem.
7. 📈 **NaCha** — final **data alignment** and **readout / visualisation**
   across modalities; per‑class mean ± SE signal‑vs‑time plots and per‑cell
   inspection via Shift‑click in Revise Mode.

A workflow‑wide **Next** button chains the widgets in order and tears down
viewer layers on transition, keeping the session clean.

---

## ✨ What's new

**2026‑04 major update**

- 🆕 **Two Cellpose‑based segmentation widgets** (Barcode Seg, Biosensor Seg)
  with on‑the‑fly manual editing (right‑click polygon draw, Ctrl+click
  delete, Z/X toggle, S cycle contrast).
- 🎓 **Online fine‑tuning** of Cellpose directly from the edited mask —
  both single‑image (the currently edited sample) and a new
  **multi‑folder dialog**: pick any number of sample folders, per‑row
  auto‑detection of the required image / mask pair with ✓ ⚠ ✗ status,
  trains jointly in one subprocess call.
- 🧵 **Subprocess isolation for PyTorch / CUDA**: all Cellpose training
  and inference runs in a child process (`_finetune_runner.py`), so
  torch and CUDA state are never loaded into the napari main process.
  This fixes a family of vispy access‑violation crashes on Windows.
- 🔍 **Persistent custom‑model discovery**: fine‑tuned models are
  auto‑discovered from the per‑sample `_finetune/` folder, the shared
  plugin model root, and the `~/.cellpose` cache, **and re‑scanned when
  the sample folder changes**. Ordering places target‑matching names
  first, then sorts by modification time.
- 🎯 **Seeded K-Means classifier** (renamed from "KMeans Cluster" for
  clarity): explicitly the semi-supervised **Seeded-KMeans** algorithm of
  Basu et al. 2002 — seeds initialise the class centroids, then the
  K-Means EM loop refines them. Added alternative methods (K-Means++,
  MiniBatchKMeans, Gaussian Mixture, Spectral), per-class outlier
  flagging (Isolation Forest), and save / load of **class distribution
  overlays** (convex hulls) with a user-adjustable expansion factor that
  serve as prior knowledge for manual seeding.
- 🎉 **NaCha finalise**: auto‑broadcasts single‑frame masks to the full
  biosensor stack length, Shift‑click per‑cell signal inspection in
  Revise Mode, and a celebration dialog on final Calculate that reports
  the total elapsed time from PTU Reader open.
- 🛠 **vispy 0x1C crash fix**: a backport of napari PR #8122 is applied
  at plugin load (see `_widget.py` → `_install_vispy_0x1c_patch`), so
  removing layers on Windows / NVIDIA no longer corrupts the shared GL
  context. Self‑tests in `walkthrough/` reproduce the add → remove →
  add pattern stress‑free.

---

## 🔧 Installation

The plugin runs Cellpose in **subprocess‑isolated child processes** (to
keep CUDA / OpenGL state out of napari's main process), and Cellpose 2.x
and 4.x have **incompatible APIs** + different model formats. As a
result the recommended setup is **three conda envs** — one for napari +
the plugin, plus one each for cellpose 2 and cellpose 4. The plugin
auto‑detects which python belongs to which slot, so you don't have to
configure paths by hand.

```bash
# 1️⃣ napari + this plugin (the env you actually launch napari from)
conda create -n nacha python=3.10 -y
conda activate nacha

# Install PyTorch matching your OS / CUDA (https://pytorch.org/get-started/locally/)
# Example (CUDA 12.1):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Track-Anything (used by B&P Tracker and NaCha)
git clone https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything && pip install -r requirements.txt && cd ..

# Install this plugin (editable mode recommended)
cd Path_To_BC-FLI-Spectra/Napari_plugin
pip install -e .   # or: pip install .

# 2️⃣ Cellpose 2.x env (segmentation Barcode/Biosensor — legacy 2-channel models)
conda deactivate
conda create -n cellpose2 python=3.10 -y
conda activate cellpose2
pip install "cellpose==2.2.3"

# 3️⃣ Cellpose 4.x env (CellposeSAM — current 3-channel default models)
conda deactivate
conda create -n cellpose4 python=3.10 -y
conda activate cellpose4
pip install "cellpose>=4.1,<5"
```

The plugin scans your conda envs at startup, picks one v2 candidate +
one v4 candidate (scored by env‑name affinity + version recency), and
writes the chosen pythons to `~/.bc_flim_spectra_envs.json`. Override
the slot anytime with the env vars `BCFLIM_CELLPOSE_V2_PYTHON` and
`BCFLIM_CELLPOSE_V4_PYTHON`, or by editing that JSON file.

### CellposeSAM weights (~1.15 GB)

The first time you run a v4 model with the default name `cpsam`,
Cellpose downloads the weights from `cellpose.org`. **In China this
download is often blocked.** Two workarounds:

```bash
# (a) point Cellpose at a HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com   # bash / zsh
$env:HF_ENDPOINT = "https://hf-mirror.com"  # PowerShell

# (b) download manually and drop into the cache
#     URL: https://hf-mirror.com/mouseland/cellpose-sam/resolve/main/cpsam
#     Target: ~/.cellpose/models/cpsam   (Windows: %USERPROFILE%\.cellpose\models\cpsam)
```

> 📝 Notes
> • We intentionally do **not** pin a Qt backend (PyQt5 / PySide) in the
>   plugin dependencies to avoid conflicts with other plugins or napari
>   distributions. Install whichever Qt backend napari itself is using.
> • For headless / CI systems prefer `opencv-python-headless`; for
>   desktop use keep `opencv-python`.
> • Make sure your CUDA driver / toolkit matches the PyTorch build you
>   install. Cellpose inference and fine‑tuning both accept a `Use GPU`
>   checkbox; if GPU is not available they fall back to CPU.
> • Track‑Anything has loose torch / CUDA constraints; if its
>   `requirements.txt` upgrades torch beyond what your driver supports,
>   pip‑install it with `--no-deps` and resolve dependencies manually.

---

## 🚀 Launch in napari

```bash
conda activate nacha
napari
```

Open the menu **`Plugins → BC‑FLIM‑Spectra`** and pick one of the seven
widgets. A **Next** button at the bottom of each widget advances to the
next stage in the canonical workflow order.

---

## 🧭 How Cellpose env routing works

The plugin never imports Cellpose into the napari process. Every train
or inference call goes through `_finetune_runner.py` launched as a
**subprocess**, and the plugin chooses **which python** to launch based
on the model:

| Model name pattern                              | Routed env | Input shape |
| ----------------------------------------------- | ---------- | ----------- |
| `*-cpsam-*`, `*cellpose4*`, weight file >200 MB | v4         | 3‑channel RGB render |
| `cpsam` (the v4 builtin)                        | v4         | 3‑channel RGB render |
| Anything else (incl. `cyto2`, `nuclei`, custom v2) | v2      | 1‑ or 2‑channel grayscale |

The two BarcodeSeg defaults (N: `NinNC-cpsam-fastflimRGB-…`, P:
`CinNC-cpsam-fastflimRGB-…`) are v4 → routed to the `cellpose4` env.
BiosensorSeg's default (`BS-BC-assist-cls-bgy-260426`) is v2 → routed
to the `cellpose2` env. **You don't pick the env, the model name does.**

If the v4 env isn't installed the routing logs a warning and falls back
to v2. The plugin's status panel in BarcodeSeg shows ✓/✗ per slot at a
glance, with the resolved python paths and override hints.

```python
# Override at runtime via env vars (highest priority)
$env:BCFLIM_CELLPOSE_V2_PYTHON = "D:\envs\my_cellpose2\python.exe"
$env:BCFLIM_CELLPOSE_V4_PYTHON = "D:\envs\my_cellpose4\python.exe"

# Or edit the persistent cache
notepad $env:USERPROFILE\.bc_flim_spectra_envs.json
```

To see which env was picked and why:

```python
import logging
logging.getLogger("bc_flim_spectra").setLevel(logging.INFO)
import flim_s_gen   # logs: v2 python: …, v4 python: …, scoring decisions
```

---

## 🗺 Quick workflow

1. 📥 **PTU Reader** — load a `.ptu` and decode it into an intensity stack
   and FLIM stack under `<sample>/intensity/` and `<sample>/flim/`.
2. 🔬 **Barcode Seg (Cellpose)** — run N and P segmentation on the
   intensity‑sum image. Edit masks in napari. Optionally fine‑tune the
   N / P model from the edits (single‑image or multi‑folder).
3. 🌀 **Calculate FLIM‑S** — compute lifetime / phasor features; write
   `FLIM‑S.xlsx` with per‑cell features.
4. 🧩 **Seeded K-Means** — place seeds on each barcode class (optionally
   load a prior distribution overlay), pick a method, flag outliers, and
   export per‑cell class labels.
5. 🟡 **Biosensor Seg (Cellpose)** — generate / pick the seg image, load
   and align the barcode classification layer as auxiliary channel,
   run the dual‑input Cellpose model, edit masks against the barcode
   reference. Optionally fine‑tune.
6. 🎬 **B&P Tracker** *(optional for time‑lapse)* — track cells through
   the confocal stack.
7. 📈 **NaCha** — final alignment and per‑class signal computation.
   Shift‑click any cell in Revise Mode to inspect its individual
   signal curve before trusting the class averages.

---

## 📁 Repository layout

```
Napari_plugin/
├── README.md              (this file)
├── pyproject.toml
├── src/flim_s_gen/
│   ├── _widget.py         ← all seven widgets live here
│   ├── _finetune_runner.py ← standalone Cellpose subprocess runner
│   ├── napari.yaml        ← napari manifest (widget registration)
│   ├── resources/         ← Cellpose / Track‑Anything logos
│   ├── track_anything_simple.py, tracker/, tools/, inpainter/
│   └── _tests/
└── walkthrough/
    ├── storyboard.md                         ← 13‑slide demo storyboard
    ├── test_0x1c_patch.py                    ← headless GL‑crash patch test
    ├── test_0x1c_with_subproc.py             ← end‑to‑end patch + cellpose
    └── test_model_scan_and_multi_ft.py       ← dropdown + multi‑folder FT
```

---

## 🛟 Troubleshooting

- ❓ **Plugin not listed in napari** — usually a missing dependency. Start
  napari from a terminal (`napari`) and read the stderr for the import
  error, install the missing package, restart.
- 💥 **Access violation on Windows / NVIDIA** — make sure the plugin
  imports cleanly; our backport of napari PR #8122 prints
  `[vispy-patch] installed napari PR#8122 backport (0x1C fix).` on
  import. If you see the crash again, run the reproducer:
  ```bash
  python Napari_plugin/walkthrough/test_0x1c_patch.py
  ```
- 🧪 **Cellpose fine‑tune / inference errors** — all Cellpose runs go
  through the subprocess (`_finetune_runner.py`). Errors are re‑raised
  in the main process with the child's stderr tail attached, so check
  the message for lines starting with `ERROR:`.
- ⚡ **GPU / CUDA mismatch** — verify your PyTorch build matches your
  driver. You can untick the `Use GPU` checkbox in each widget to fall
  back to CPU.
- 📦 **External model files** (Track‑Anything weights) — if the automatic
  download fails, follow the Track‑Anything official docs for manual
  checkpoint placement.

---

**Enjoy NaCha! 🎉**
