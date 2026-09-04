---
name: lumina-network
description: Guide a user through LUMINA, the standalone PyTorch classifier for dual-anchor barcodes that ships in this repository alongside the SLIC napari plugin - preparing per-cell crops with Data_prep.py, the two-stage recipe in Train_LUMINA.py, inference and confidence scoring in Test_LUMINA.py, and the co-occurrence figure from Visualize_heatmap.py. Use whenever someone is working with cell<id>_5D.tif crops, seg_5D / seg_5D_calib folders, a DualHeadConvNet checkpoint (best_model_initial.pth, best_model_fine-tune.pth), predict_class_confident_*.xlsx, or asks about dual-anchor barcodes, per-pixel phasor G/S maps as network input, the two independent nuclear and mitochondrial heads, the weight-balanced dual cross-entropy loss, or adapting a trained checkpoint to a new cell line / new domain by few-shot fine-tuning of the heads. None of the four scripts takes command-line arguments - every path and hyperparameter is a constant edited in the source.
license: BSD-3-Clause
---

# LUMINA — dual-anchor barcode classification

A PyTorch classifier for cells carrying **two** barcodes at once: one
fluorescent protein on a nuclear anchor (`NLS…`) and one on a mitochondrial
anchor (`NTOM…`). It lives in `LUMINA_classification/`, is independent of the
napari plugin — its own conda env, its own `requirements.txt`, nothing imports
it — and consists of four scripts run by hand, in order.

Use this skill to answer "how do I run it on my data", "what does this number
mean", and "why did it do that". For editing the repository's code, see
`AGENTS.md` at the root; for the plugin half — SLIC — see the sibling skill
`.claude/skills/slic-napari/`.

## The idea in one paragraph

The plugin's route — five numbers averaged over a whole cell, then clustered —
gives one cell **one** barcode, and an average cannot separate two fluorophores
sitting in two organelles of the same cell. LUMINA keeps the pixels instead.
`Data_prep.py` writes, per segmented cell, a small
image stack whose planes are the calibrated per-pixel phasor *G* and *S*, three
calibrated spectral intensity ratios and the intensity itself. A network gives
each of those six planes its own convolutional stem, concatenates them, runs a
shared ResNet-style trunk, and ends in **two independent classifier heads** —
one for the nuclear anchor, one for the mitochondrial anchor — trained under a
class-balanced cross-entropy on each head, summed 1:1. Because the two anchors
sit on different pixels of the same cell, one forward pass reads both.

> The repository README describes the input as "the phasor *G* map and the
> intensity image". The code feeds **six** planes; `DualHeadConvNet.forward`
> slices `x[:, i:i+1]` for `i in range(6)`. Believe the code.

## The four scripts

No script has an argument parser. Every path, every hyperparameter and the list
of samples to process are module-level or `main()`-level constants, and most
sample lists are shipped with all but one or two entries commented out.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `Data_prep.py` | `<sample>/raw/`, `flim_stack/<fov>-sum.tif`, `intensity/<fov>-{1..4}.tif`, `intensity/<fov>-sum_seg.npy` | `<sample>/seg_5D_calib/cell<id>_5D.tif` — one 6×h×w float64 crop per cell |
| 2 | `Train_LUMINA.py` | single-anchor folders' `seg_5D/`, dual-anchor folders' `clustered.xlsx` + crops | `best_model_<phase>.pth`, `combination_accuracies_<phase>.xlsx`, `test_train_val_log.xlsx`, `val_df.xlsx` |
| 3 | `Test_LUMINA.py` | a checkpoint + crops for each listed sample | `<sample>/predict_class_confident_<thr>.xlsx`, `predict_class_uncertain_<thr>.xlsx` |
| 4 | `Visualize_heatmap.py` | those two workbooks | `heatmap_nu_mito.pdf` in the **current working directory**, plus a Tk window |

**The sample folder is the unit of work**, as in the plugin — but the file names
inside it are not the plugin's. LUMINA wants `<fov>-1.tif` and
`<fov>-sum_seg.npy`; the plugin writes `<fov>_ch1.tif` and `<fov>_sum_seg_n.npy`.
Nothing in this repository converts one to the other. See
`references/data-and-prep.md`.

## Reference files

Read the one you need; do not read them all.

| File | Read it when |
|---|---|
| `references/data-and-prep.md` | **Start here for any "will it find my data" question.** The exact folder and file names each script expects, the six planes and their order, array shapes and dtypes, the spreadsheet columns, and every constant in `Data_prep.py`. |
| `references/training.md` | Training or fine-tuning: the two stages, which checkpoint feeds which, what is frozen when, the loss weighting, the LR schedule, and what the committed switch values actually do. |
| `references/inference-and-heatmap.md` | Running `Test_LUMINA.py` or reading its output: the confidence score formula, the two workbooks, and what the heatmap does and does not show. |
| `references/troubleshooting.md` | Something hung, crashed, or produced a result that cannot be right. |
| `references/domain-adaptation.md` | **Someone wants to use LUMINA on a different cell line.** Why a checkpoint degrades across cell lines, the few-shot head-only recipe that fixes it, how many cells they need to label, the triage gate that precedes it, and what is not in this repository. |

## Answering well here

**There is no command line. Name the line to edit.** "Set `base_folder` on line
346 of `Test_LUMINA.py`" is an answer; "pass `--input`" is wrong, and there is no
flag to pass. Every default quoted in these files is a literal in the source,
with its line number.

**Class indices come from dictionary order, not from the class name.**
`nu_class_map = {key: idx + 1 for idx, key in enumerate(nu_files.keys())}`, so
with the shipped literal `N10` is class 1 and `N1` is class 7. Index 0 means *no
anchor / unknown* and is never a training target. A checkpoint stores no mapping,
so reordering `nu_files` or `mito_files` between training and testing silently
permutes every label. Check that the dicts in `Train_LUMINA.py` (lines 564–591)
and `Test_LUMINA.py` (lines 501–528) are character-for-character the same.

**"5D" means six planes.** The folder is `seg_5D` / `seg_5D_calib` and the file
is `cell<id>_5D.tif`, but the stack is `[G, S, ratio1, ratio2, ratio3,
intensity]` — the five-dimensional fingerprint plus the intensity plane the
network normalises. The stale comment on `Data_prep.py` line 227 lists eight
names for the six planes it writes; the `np.stack` on line 228 is authoritative.

**Only the last plane is normalised.** `normalize_intensity` divides
`image[-1]` by its max and touches nothing else. G, S and the ratios go into the
network in physical units. That is deliberate — they are already bounded — but
it means a stack with the wrong number of planes is normalised in the wrong
place and still runs.

**Both heads must be right.** Every accuracy the training loop prints is the
fraction of cells where the nuclear *and* the mitochondrial prediction are both
correct, never a per-anchor accuracy.

**Do not invent numbers.** No accuracy, dataset size or runtime appears in these
files unless the source states it. If you do not know a valid range for a
parameter, say so and point at the constant.

**Two anchors, two vocabularies.** Nuclear classes are written `N…`
(`N10 N13 N4 N14 N16 N8 N1`) and mitochondrial classes `M…`
(`M10 M13 M4 M14 M16 M8 M1`); the training loop prints combinations as
`N3-M5`. Keep the prefixes.

## Fast checks

What is actually inside a prepared cell crop:

```python
import tifffile as tiff
a = tiff.imread('seg_5D_calib/cell1_5D.tif')
print(a.shape, a.dtype)          # expect (6, h, w), float64
print([float(a[i].max()) for i in range(a.shape[0])])
# planes: 0 G, 1 S, 2 ratio1, 3 ratio2, 4 ratio3, 5 intensity
```

Anything but 6 in the first position will either crash the forward pass or be
silently truncated to the first six planes — see `references/troubleshooting.md`.

How many cells are prepared, and how many are too big for the 256×256 canvas
the loaders pad to:

```python
import glob, tifffile as tiff
f = glob.glob('seg_5D_calib/cell*_5D.tif')
big = [p for p in f if max(tiff.imread(p).shape[1:]) > 256]
print(len(f), 'cells,', len(big), 'oversized (silently skipped)')
```

What a checkpoint contains (it is a bare `state_dict`, no class map, no epoch):

```python
import torch
sd = torch.load('best_model_fine-tune.pth', map_location='cpu')
print(len(sd), 'tensors')
print(sd['fc_nu.6.weight'].shape, sd['fc_mito.6.weight'].shape)  # (num_classes, 256)
```

The last line is the only record of how many classes the model was built with;
`num_classes = 8` is hardcoded in both scripts (`Train_LUMINA.py` line 618,
`Test_LUMINA.py` line 537).
