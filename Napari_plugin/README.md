# 🔬 SLIC — the napari plugin (NaCha)

*Three names, one thing: the science is **SLIC**, the pip package and the napari
menu entry are **`bc-flim-spectra`**, and **NaCha** is the final widget — the name
the [hosted demo](https://baoyi-a.github.io/nacha-demo/) goes by. This folder is
`Napari_plugin/`.*

*One of the two tools in [SLIC](../README.md). For dual-anchor barcodes, see
[LUMINA](../LUMINA_classification/README.md).*

> 🤖 An agent skill for this plugin ships at [`.claude/skills/slic-napari/`](../.claude/skills/slic-napari/); see the [root README](../README.md#-slic-works-with-an-ai-assistant).

**SLIC** (Spectral‑Lifetime Indexing of Cells) is a napari plugin that supports an end‑to‑end
workflow for FLIM and barcode analysis, from raw **`.ptu`** ingestion through
segmentation, classification, tracking and final alignment / visualisation.

It exposes **seven widgets** under the napari menu
**`Plugins → bc-flim-spectra`** (the package name, kept for compatibility):

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

The current version adds two Cellpose segmentation widgets with in‑viewer
editing and online fine‑tuning, runs every Cellpose call in an isolated
subprocess, and ships the classifier as **Seeded K‑Means**.
Release‑by‑release detail is in the [Changelog](#-changelog).

---

## 🔧 Installation

> 💡 The same workflow runs in a browser at
> <https://baoyi-a.github.io/nacha-demo/>, on the demo dataset, with none of the
> setup below. The three-environment install described here is for local data.

Cellpose 2.x and 4.x have **incompatible APIs** and different model
formats, and the plugin runs Cellpose in a subprocess, so the setup is
**three conda envs** — one for napari + the plugin, one each for cellpose
2 and cellpose 4. The plugin auto‑detects which python belongs to which slot.

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

# napari needs a Qt backend, which this plugin deliberately does not pin
pip install "napari[all]"          # or: pip install napari pyqt5

# Install this plugin (editable mode recommended)
cd Path_To_BC-FLI-Spectra/Napari_plugin
pip install -e ".[segmentation]"   # add ,harmony for the Harmony calibration option
#   `pip install -e .` alone is enough for PTU Reader, Calculate FLIM-S and
#   Seeded K-Means; the [segmentation] extra adds cellpose + torch, which the
#   Barcode / Biosensor segmentation widgets need.

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
Cellpose downloads the weights from `cellpose.org`. **If that download is
blocked**, two workarounds:

```bash
# (a) point Cellpose at a HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com   # bash / zsh
$env:HF_ENDPOINT = "https://hf-mirror.com"  # PowerShell

# (b) download manually and drop into the cache
#     URL: https://hf-mirror.com/mouseland/cellpose-sam/resolve/main/cpsam
#     Target: ~/.cellpose/models/cpsam   (Windows: %USERPROFILE%\.cellpose\models\cpsam)
```

> 📝 Notes
> • Install whichever Qt backend napari itself is using; the plugin pins none.
> • For headless / CI systems prefer `opencv-python-headless`; for
>   desktop use keep `opencv-python`.
> • Make sure your CUDA driver / toolkit matches the PyTorch build you
>   install.
> • Track‑Anything has loose torch / CUDA constraints; if its
>   `requirements.txt` upgrades torch beyond what your driver supports,
>   pip‑install it with `--no-deps` and resolve dependencies manually.

---

## 🚀 Launch in napari

```bash
conda activate nacha
napari
```

Open the menu **`Plugins → BC‑FLIM‑Spectra`** and pick one of the seven widgets.

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

The defaults are the three models published with the paper, all v2:
BarcodeSeg uses `NinNC-260328-1` (N) and `CinNC-260328-1` (P), BiosensorSeg
`BS-BC-assist-cls-260402-forDense`; if one is absent the next candidate
present on the machine is used. **You don't pick the env, the model name does.**

If the v4 env isn't installed the routing logs a warning and falls back
to v2. The plugin's status panel in BarcodeSeg shows ✓/✗ per slot at a
glance, with the resolved python paths and override hints.

```powershell
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

## 📦 Segmentation models

The plugin's default Cellpose models are archived at
**<https://doi.org/10.5281/zenodo.22499321>**. Extract the archive and point the plugin at
`plugin_defaults/`:

```bash
export BCFLIM_MODEL_ROOT=/path/to/plugin_defaults          # bash / zsh
$env:BCFLIM_MODEL_ROOT = "C:\path\to\plugin_defaults"     # PowerShell
```

Without them the plugin falls back to a public Cellpose base model, which
segments but does not reproduce the published masks. The same archive carries
`manuscript_quantified/`, the eight models whose segmentation performance the
manuscript reports, under the manuscript's own names; `MANIFEST.tsv` gives md5
and sha256 for every file and the original training-run name of each.

---

## 🧩 Bring your own Cellpose model

Drop a `config.json` beside the model weight (or in its parent dir, or one level
up — first hit wins) and the plugin applies it: input kind, diameter, Cellpose
thresholds, post-processing. The status hint gains a `📄cfg` tag when one took
effect. Fine-tuning from inside the plugin writes one for you, so this is only
something to hand-author for a model trained elsewhere.
[Full key list](../.claude/skills/slic-napari/references/workflow.md) ·
[worked example](examples/).

---

## 🗺 Quick workflow

1. 📥 **PTU Reader** — load a `.ptu` and decode it into an intensity stack
   and FLIM stack under `<sample>/intensity/` and `<sample>/flim/`.
2. 🔬 **Barcode Seg (Cellpose)** — run N and P segmentation on the
   intensity‑sum image. Edit masks in napari. Optionally fine‑tune the
   N / P model from the edits (single‑image or multi‑folder).
3. 🌀 **Calculate FLIM‑S** — compute lifetime / phasor features from
   1–4 decay channels and the N / M / P masks you select; write
   `FLIM‑S.xlsx` with per‑cell features. A run merges into an existing
   workbook per FOV (tick *Fresh FLIM‑S.xlsx* to start clean); the
   batch button uses the same channel / mask selection.
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

## 📜 Changelog

Numbered as described in [`VERSIONING.md`](../VERSIONING.md): MAJOR when a
file on disk changes meaning, MINOR when the same input can give a different
result, PATCH otherwise.

**1.1.1 — 2026-09-11**

- The environment records shipped with 1.1.0 could not be installed: pins
  were taken from `pip list`, which drops the git origins of packages
  installed from repositories and, with two copies of a package present,
  can name the copy that never loads. They are now generated from what the
  interpreter imports, keep git origins, name the PyTorch index, and
  `release.py locks` refuses to write a record that disagrees with the
  interpreter. Apply the conda file, then the pip file; `VERSIONING.md`
  says so.
- `signal_analysis.xlsx` gains its `_meta` sheet; `clustered.xlsx` records
  what each run actually did (whitening applied or skipped by its guard,
  the seed file that was loaded) instead of the widget state at save time;
  `FLIM-S.xlsx` keeps the previous write's record under `previous.*`;
  workbooks are written to a temp file and renamed into place, so an
  interrupted write no longer leaves an empty file.
- `release.py check` also refuses to tag a commit that already carries
  another tag, and when the installed plugin does not report the
  `pyproject.toml` version.

**1.1.0 — 2026-09-11**

- Whitening by within-cluster spread in Seeded K-Means, ported from the
  research fork: a seed set saved on one acquisition transfers to another
  without the two closest barcodes swapping. Default on; auto-skipped when
  fewer than 8 seeds are claimed by a cluster centre. **Results differ from 1.0.1**, which is
  why this is a MINOR bump; the manuscript's numbers correspond to this version.
- Loaded seeds initialise K-Means from their own coordinates rather than from
  the nearest cell.
- `FLIM-S.xlsx`, `clustered.xlsx`, the seeds file and `Bs2Code.xlsx` gain a
  `_meta` sheet recording the plugin version and the run settings; fine-tuned
  models record `plugin_version` in `config.json`. Data sheets are unchanged
  for existing readers. (`signal_analysis.xlsx` follows in 1.1.1.)
- One version number: `pyproject.toml` is the source, `flim_s_gen.__version__`
  reads the installed metadata, `scripts/release.py check` keeps the tag and
  this changelog in step; `envs/` holds exact environment records per release.
- LUMINA: the K-shot domain-adaptation trainer ships; inference scores the
  cell each row asks for even when its crop is oversized; data preparation
  keeps every field of view of a sample.
- Repository made installable (`pyproject.toml` dependencies completed), with
  agent-readable docs (`AGENTS.md`, `CLAUDE.md`, `.claude/skills/`), DOI and
  licence badges, and the segmentation models archived separately.
- Default models are resolved against the machine at import, so a name that is
  not on it no longer reaches the dropdown.
- The fine-tune base model is resolved in the parent process and passed to the
  child, so a model the dropdown offers is one training can start from.
- The model store accepts ordinary folder layouts instead of a hardcoded drive.
- The segmentation panel reports whether Cellpose will actually use the GPU.
- Fixed a silent hang when the Cellpose child process failed.

**1.0.1 — 2026-09-01 — first public release**

- Per-label hole filling and erosion sped up; Python `faulthandler` enabled
  so a Qt crash leaves a traceback. `v1.0.0` and `v1.0.1` point at the same
  commit, and `pyproject.toml` still said `0.1.0`.

**2026-06 / 07**

- Per-head input-kind picker; `cpsam` offered among the builtins.
- Seeded K-Means: per-FOV mode, optional Harmony calibration to a labelled
  reference, and classification layers added to napari on save.
- PTU Reader: re-render dialog when a folder is already complete.

**2026-05**

- Cellpose v2 and v4 side by side, routed automatically by model name.
- Per-model `config.json`: bring your own model with its own input kind and
  parameters.
- Multi-FOV batch runs and post-processing knobs (erode, close holes, dilate,
  minimum area), with undo.
- Sample folder, model choice and contrast settings persist across sessions.

**2026-04 — major update**

- Two Cellpose segmentation widgets (Barcode Seg, Biosensor Seg) with manual
  mask editing.
- Online fine-tuning from the edited mask, single-image or across any number of
  sample folders.
- Cellpose training and inference moved into a child process, which fixed a
  family of vispy access-violation crashes on Windows.
- Custom models discovered from the sample folder, the shared model root and the
  Cellpose cache, re-scanned when the sample folder changes.
- Seeded K-Means classifier (Basu et al. 2002) with alternative methods, outlier
  flagging, whitening by within-cluster spread, and saved class-distribution
  overlays.
- NaCha finalise: single-frame masks broadcast to the full biosensor stack, with
  per-cell signal inspection.
- vispy 0x1C crash fix, a backport of napari PR #8122 applied at plugin load.

---

**Enjoy SLIC! 🎉**
