# AGENTS.md — BC-FLIM-Spectra

Instructions for coding agents working in this repository. Written for the
[AGENTS.md](https://agents.md) convention; Claude Code reads it through the
one-line `CLAUDE.md`.

## What this is

**BC-FLIM-Spectra** (in-app name *NaCha*) is a [napari](https://napari.org)
plugin for barcoded fluorescence-lifetime imaging (FLIM). It turns raw
PicoQuant `.ptu` acquisitions into per-cell 5-D fingerprints
(phasor *G*, *S* + three spectral intensity ratios), classifies each cell into
its barcode, then reads a biosensor channel per class.

Two independent parts:

| Path | What it is |
|---|---|
| `Napari_plugin/` | The plugin. Seven GUI widgets, all in `src/flim_s_gen/_widget.py`. |
| `LUMINA_classification/` | A standalone PyTorch classifier for dual-anchor barcodes. Not imported by the plugin. |

## Usage knowledge lives in a skill

How to *operate* the software — the seven steps, what each parameter means,
what to do when something looks wrong — is in

```
.claude/skills/bc-flim-spectra/SKILL.md
    references/workflow.md          all seven widgets, control by control
    references/troubleshooting.md   real failure modes and the check for each
    references/install.md           environments, model weights, constraints
```

written to the [Agent Skills](https://agentskills.io) format. Read it before
answering a user's question about how to run the workflow; read this file
before changing code. One canonical copy — do not fork it into another skills
directory, point your tool at this path instead.

## Layout you need to know

```
Napari_plugin/src/flim_s_gen/
├── _widget.py            ← all seven widgets live here (~640 KB, one file)
├── _finetune_runner.py   ← Cellpose fine-tune, run as a CHILD PROCESS
├── postproc.py           ← mask post-processing (merge/erode/close/carve)
├── harmony_calib.py      ← optional Harmony batch calibration
├── napari.yaml           ← widget registration; names here must match the classes
└── walkthrough/storyboard.md   ← the 7-step demo script, in prose
```

`_widget.py` is large. **Do not read it whole** — `grep -n "    def name"` for
the method, then `sed -n 'A,Bp'`. The seven widget classes, in workflow order:
`PTUReader`, `BarcodeSeg`, `Calculate_FLIM_S`, `SeededKMeans`, `BiosensorSeg`,
`BPTracker`, `Trackrevise` (displayed as *NaCha*).

## Install

```bash
conda create -n nacha python=3.10 -y && conda activate nacha
pip install "napari[all]"                 # napari needs a Qt backend; we do not pin one
cd Napari_plugin && pip install -e ".[segmentation]"
```

`pip install -e .` alone gives you PTU reading, Calculate FLIM-S and Seeded
K-Means. The `[segmentation]` extra adds `cellpose` + `torch`, needed only by
the two Cellpose widgets. Install `torch` from
<https://pytorch.org/get-started/locally/> first if you need a specific CUDA
build — the extra will not override an existing one.

Verify without opening a window:

```bash
python -c "import flim_s_gen._widget as w; print(w.__file__)"
python -c "import flim_s_gen._widget as w; print([n for n in dir(w) if n[0].isupper()][:12])"
```

To open the GUI on one widget:

```bash
python -m napari -w bc-flim-spectra "Calculate FLIM-S"
```

Widget display names come from `napari.yaml`: `PTU Reader`,
`Barcode Seg (Cellpose)`, `Calculate FLIM-S`, `Seeded K-Means`,
`Biosensor Seg (Cellpose)`, `B&P Tracker`, `NaCha`.

## The three-environment design (important)

Cellpose 2.x and 4.x have incompatible APIs and model formats, and PyTorch
leaves CUDA/OpenGL state that crashes napari's renderer. So the plugin
**shells out** to a separate interpreter for every segmentation and
fine-tune, and expects up to three conda envs: `nacha` (napari + plugin),
one with cellpose 2.x, one with cellpose 4.x.

The plugin auto-detects them at startup and caches the choice in
`~/.bc_flim_spectra_envs.json`. Override with `BCFLIM_CELLPOSE_V2_PYTHON` /
`BCFLIM_CELLPOSE_V4_PYTHON`.

Custom model weights are searched under `BCFLIM_MODEL_ROOT` (env var, or
`model_root` in `~/.bc_flim_spectra_state.json`, or
`flim_s_gen.set_barcode_model_root(path)`), the per-sample `_finetune/` folder,
and `~/.cellpose`. Several on-disk layouts are accepted, so a plain folder of
weight files works as a model store.

**The fine-tuned models named as widget defaults are not in this repository.**
A clean checkout can only use the public Cellpose base models (`cpsam`,
`cyto3`). Do not assume the defaults resolve.

**If you are only reading, editing or testing code, you do not need the
cellpose envs.** Everything except the two segmentation widgets runs in one env.

## Sample-folder layout

Every widget works on one *sample folder*. The convention, produced by PTU
Reader and consumed by everything downstream:

```
<sample>/
├── flim_stack/<fov>_ch1..4.tif      per-channel decay stacks, axis order T×H×W
├── intensity/<fov>_sum.tif          intensity sum, the segmentation input
│   └── <fov>_sum_seg_n.npy          nucleus mask   ← masks live HERE, next to the image
│   └── <fov>_sum_seg_p.npy          cytoplasm mask
├── <fov>_fastflim_rgb.png           colour render (cached by PTU Reader)
├── <fov>_fastflim_tau.tif           lifetime map
├── FLIM-S.xlsx                      one row per cell (the 5-D features)
└── clustered.xlsx                   FLIM-S.xlsx + barcode class per cell
```

A mask always sits **beside the image it was drawn on**, not in the sample
root. Getting this wrong is a recurring bug source.

## Conventions

- **Python 3.10**, PEP 8, 4 spaces. Type hints where they help; the file is
  not fully annotated and does not need to be.
- **numpy is pinned below 2.0** — napari 0.4.19 loads numba, which requires
  numpy ≤ 1.26. Do not "upgrade numpy to fix" anything.
- Long work goes in a `@thread_worker`, never on the Qt main thread.
- `_tt(widget, text)` attaches a tooltip. Every user-facing control should
  have one, and it should say what the number *does*, not restate its name.
- This widget is a magicgui `Container`, which overrides `__delattr__`.
  **Never `delattr(self, ...)`** — assign `None` instead. `delattr` raises
  `ValueError` here and has silently broken handlers before.
- Never write GUI state you cannot recompute; persist to disk instead.

## Testing

There is no pytest suite for the widgets. The working pattern is a headless
napari script:

```python
import matplotlib; matplotlib.use('Agg')
import napari, flim_s_gen._widget as w
from qtpy import QtWidgets
viewer = napari.Viewer(show=False)
wid = w.Calculate_FLIM_S(viewer)
viewer.window.add_dock_widget(wid, name='Calculate FLIM-S')
app = QtWidgets.QApplication.instance()
def pump():
    for _ in range(20): app.processEvents()
```

Then drive the widget's own methods and assert on files it writes. Patch
`QFileDialog.getExistingDirectory` / `getSaveFileName` and
`QDialog.exec_` so nothing blocks. `walkthrough/` holds standalone checks
(`test_0x1c_patch.py` reproduces the OpenGL crash the plugin patches).

Compile check before committing:

```bash
python -m py_compile Napari_plugin/src/flim_s_gen/_widget.py
```

## Gotchas that have cost real time

- **`Calculate FLIM-S` merges into an existing `FLIM-S.xlsx`.** A run replaces
  rows of the current FOV and keeps every other FOV's rows — including rows
  written earlier with different channels or masks. Tick *Fresh FLIM-S.xlsx*
  for a single-configuration workbook.
- **Training input must match inference input.** `_cellpose_input_for()` routes
  a model to grayscale render / RGB render / raw intensity sum. Fine-tuning
  must go through the same call, or the model is trained on one form and used
  on another. A fine-tune writes `config.json` beside the weights recording
  which form it used.
- **Two lifetime quantities.** `Lifetime` is a fitted mono-exponential τ;
  `FastFLIM` is the photon-weighted mean arrival time. They are not
  interchangeable — keep the names straight.
- **Seeded K-Means seeds are a starting point**, not fixed centres: the
  clustering re-fits them on the cells currently loaded, which is what lets a
  seed set transfer between acquisitions.

## Scope

Do not commit or push unless asked. Do not add a dependency without adding it
to `Napari_plugin/pyproject.toml`. Do not reformat `_widget.py` wholesale —
its diffs are reviewed by hand.
