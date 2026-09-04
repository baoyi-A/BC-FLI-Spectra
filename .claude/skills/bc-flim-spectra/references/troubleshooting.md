# Troubleshooting

Real failures, with the check that distinguishes them. Work down the "check"
column before changing parameters.

---

## The workbook contains things I did not ask for

**Symptom.** `FLIM-S.xlsx` has `P` rows when only N was selected, or non-NaN
`Int 590-610` when only channel 1 was given.

**Cause.** Not the phasor code. A run *merges*: it replaces the rows of the
current FOV and keeps every other FOV's rows, including rows written earlier
with a different channel or mask configuration.

**Check.**

```python
import pandas as pd
df = pd.read_excel('FLIM-S.xlsx')
print(df.groupby([df['FOV'].astype(str), df['Localization'].fillna('')]).size())
```

If the unexpected rows carry a different `FOV`, that is the merge.

**Fix.** Tick **Fresh FLIM-S.xlsx** and re-run. The old workbook is renamed
`FLIM-S_old_<timestamp>.xlsx`, never deleted. The status line and the popup
also report kept-vs-new counts.

---

## Multi-folder fine-tune says a folder has no mask

**Check** that the mask is where the widget looks — **beside the image it was
drawn on**:

```
<sample>/intensity/<stem>_sum.tif
<sample>/intensity/<stem>_sum_seg_n.npy      ← here
```

not in the sample root. For the biosensor model the pair is
`<fov>_seg_image.tif` + `<fov>_seg_image_seg.npy` in the sample folder.

If the files are there and the dialog still shows ⚠, read the status text: it
distinguishes "no `*_sum.tif` at all" (✗) from "FOVs present, none with a
mask" (⚠), and reports partial coverage as "2 of 3 FOVs".

**Also:** a mask whose shape does not match its image is refused by name — that
means the `.npy` belongs to a different FOV.

---

## The fine-tuned model performs worse than the base model

Three usual causes, in order of likelihood.

1. **Too few examples.** Single-image fine-tuning trains on exactly one FOV.
   Use the multi-folder button, which uses every FOV that has a saved mask.
2. **Too many epochs on too little data.** 50–200 is the working range for
   small corrections; more overfits one field of view.
3. **Input-form mismatch.** The model must be trained on the same form it is
   used with. Fine-tunes now write a `config.json` beside the weights
   recording `input_kind`; check it:

```bash
cat <sample>/_finetune/<model>/config.json
```

An older model without a `config.json` falls back to the heuristic
(v4 → RGB render, v2 → grayscale render). If you had forced *Input kind* to
something else when you trained it, set the same override when you use it.

---

## Cellpose does not run / wrong environment

The plugin runs Cellpose in a **child process**, and Cellpose 2.x and 4.x have
incompatible APIs. It auto-detects one v2 and one v4 environment and caches the
choice.

**Check** what it found — the widget header shows the resolved interpreters, or:

```bash
cat ~/.bc_flim_spectra_envs.json
```

**Fix.** Point it explicitly:

```bash
export BCFLIM_CELLPOSE_V2_PYTHON=/path/to/envs/cellpose2/bin/python
export BCFLIM_CELLPOSE_V4_PYTHON=/path/to/envs/cellpose4/bin/python
```

On Windows use `$env:BCFLIM_CELLPOSE_V4_PYTHON = "...\python.exe"`. An
environment variable only reaches processes started *after* it is set, so a
napari launched from a desktop shortcut will not see one you just typed in a
terminal.

**Model not found.** The widget header shows the resolved model root, where it
came from, and how many models are under it. The fine-tuned defaults
(`NinNC-…`, `CinNC-…`, `BS-BC-assist-…`) are **not** in the repository; on a
fresh checkout pick a public base model (`cpsam`, `cyto3`) or point the plugin
at your own store:

```bash
export BCFLIM_MODEL_ROOT=/path/to/models
```

Flat folders, one-directory-per-model (with or without a `_BEST` suffix), and
the trainer's `<name>/models/<name>` layout are all accepted. Weights are also
searched in the per-sample `_finetune/` folder and `~/.cellpose`.

---

## CellposeSAM weights will not download

First use of `cpsam` pulls ~1.15 GB from `cellpose.org`, which is often blocked
in China.

```bash
# either point Cellpose at a mirror
export HF_ENDPOINT=https://hf-mirror.com

# or fetch it by hand into the cache
#   https://hf-mirror.com/mouseland/cellpose-sam/resolve/main/cpsam
#   -> ~/.cellpose/models/cpsam        (Windows: %USERPROFILE%\.cellpose\models\cpsam)
```

---

## napari crashes when layers are removed (Windows / NVIDIA)

`access violation reading 0x1C` in the paint loop. This is a vispy/OpenGL bug;
the plugin applies a backport of napari PR #8122 at import
(`_install_vispy_0x1c_patch`). `walkthrough/test_0x1c_patch.py` reproduces the
add → remove → add pattern to verify the patch is live.

If it still happens, it is usually PyTorch leaving GPU state behind: the plugin
releases torch caches and defers teardown between widgets for exactly this
reason. Run segmentation and fine-tuning through the widgets rather than
importing cellpose into the napari process yourself.

---

## Remote desktop: napari opens a white window

Remote sessions can disable hardware OpenGL. Use the software renderer by
copying Mesa's `opengl32sw.dll` over `opengl32.dll` next to the Python
executable.

---

## Two barcodes swap classes when I reuse a seed file

Expected without whitening, and the reason it exists. The 5-D weights scale one
axis each and cannot describe a cluster that is long, thin and tilted, so the
two closest barcodes can end up separated by less than their own width.

**Fix.** Tick **Whiten by within-cluster spread** (on by default). If it
reports "skipped, fewer than 8 seeds claimed by a cluster centre", the dish
carries only a few barcodes, or the loaded seeds sit far from this data — check
the warning the loader prints about seeds far from every cell.

---

## Everything landed in class 0

Class 0 is the outlier class. Either contamination is too high (0.05–0.15 is
the working range) or the cells really are poor — check `Total intensity` and
`Mask intensity threshold` first. Outlier detection cannot be a confidence
gate: it only finds cells unlike their own cluster.

---

## Lifetimes are all wrong by a constant factor

Check **Pulse frequency** and **Tau resolution** before anything else. The
phasor is evaluated in a window set by the repetition rate, so a wrong value
scales every lifetime. Leica SP8 / STELLARIS: 78.1 MHz, 0.097 ns per bin with a
256-bin decay. The PTU metadata has the true values.

---

## "0/1 FOVs" but the per-cell numbers printed fine

The phasor ran; the *write* failed. pandas needs `openpyxl` to produce an
`.xlsx`, and on an incomplete install the batch reports `0/N FOVs` after
streaming every cell's G and S to the console, which reads like success.

```bash
pip install openpyxl        # or reinstall the plugin, which now requires it
```

## The plugin will not import on a newly installed napari

> `AttributeError: type object 'QtViewer' has no attribute '_remove_layer'`

napari 0.5 removed that method. The plugin targets the 0.4.x Qt viewer and now
declares `napari>=0.4.19,<0.5`; a pre-existing environment may still carry a
newer one:

```bash
pip install "napari[all]==0.4.19.post1"
```

## `npe2 list` crashes with a UnicodeEncodeError

> `UnicodeEncodeError: 'gbk' codec can't encode character '✅'`

A non-UTF-8 Windows console (common with a Chinese system locale) cannot print
npe2's tick marks. Not a plugin problem:

```bash
set PYTHONIOENCODING=utf-8      # PowerShell: $env:PYTHONIOENCODING = "utf-8"
```

## Numbers changed after I upgraded numpy

Do not upgrade numpy. It is pinned below 2.0 because napari 0.4.19 loads numba,
which requires ≤ 1.26. Upgrading breaks the napari tools that share the
environment.
