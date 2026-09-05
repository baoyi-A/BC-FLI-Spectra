---
name: lumina-network
description: Guide a user through LUMINA, the standalone PyTorch classifier for dual-anchor barcodes that ships in this repository alongside the SLIC napari plugin - preparing per-cell crops with Data_prep.py, the two-stage recipe in Train_LUMINA.py, inference and confidence scoring in Test_LUMINA.py, few-shot adaptation to a new cell line with Finetune_LUMINA.py, and the co-occurrence figure from Visualize_heatmap.py. Use whenever someone is working with cell<id>_5D.tif crops, seg_5D / seg_5D_calib folders, a DualHeadConvNet checkpoint (best_model_initial.pth, best_model_fine-tune.pth), predict_class_confident_*.xlsx, or asks about dual-anchor barcodes, per-pixel phasor G/S maps as network input, the two independent nuclear and mitochondrial heads, the weight-balanced dual cross-entropy loss, or adapting a trained checkpoint to a new cell line / new domain by few-shot fine-tuning of the heads. All five scripts are driven by command-line flags; --help is authoritative on each, and no script has editable constants left in it.
license: BSD-3-Clause
---

# LUMINA — dual-anchor barcode classification

A PyTorch classifier for cells carrying **two** barcodes at once: one
fluorescent protein on a nuclear anchor (`NLS…`) and one on a mitochondrial
anchor (`NTOM…`). It lives in `LUMINA_classification/`, is independent of the
napari plugin — its own conda env, its own `requirements.txt`, nothing imports
it — and consists of four scripts run in order, plus an optional fifth for
moving a checkpoint onto a new cell line.

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

## The five scripts

**All five are driven by argparse.** Every path and every hyperparameter that
scripts 1–4 once held as a module-level or `main()`-level literal is now a flag
carrying *that same literal* as its default, so a run that passes no optional
flag computes what the committed file computed. The deliberate exception is the
paths: the lab drive letters and cluster directories did not become defaults,
they became **required** flags, because a default nobody outside the lab can
reach is worse than no default. `--help` on each script is authoritative for the
flag list and for the defaults.

| # | Script | Required flags | Reads | Writes |
|---|---|---|---|---|
| 1 | `Data_prep.py` | `--data-root`, and one of `--samples` / `--all-samples` | `<sample>/raw/`, `flim_stack/<fov>-sum.tif`, `intensity/<fov>-{1..4}.tif`, `intensity/<fov>-sum_seg.npy` | `<sample>/<--seg-folder>/cell<id>_5D.tif` — one 6×h×w float64 crop per cell — plus `<sample>/data_prep_run_config.csv` |
| 2 | `Train_LUMINA.py` | `--data-root`, `--single-anchor-manifest`, `--dual-root`, `--out`, and one of `--dual-samples` / `--dual-samples-file`, and one of `--checkpoint` / `--from-scratch` | single-anchor folders' crops, dual-anchor folders' `clustered.xlsx` + crops | under `--out`: `best_model_<phase>.pth`, `combination_accuracies_<phase>.xlsx`, `test_train_val_log.xlsx`, `val_df.xlsx`, `train_run_config.csv` |
| 3 | `Test_LUMINA.py` | `--checkpoint`, `--data-root`, and one of `--samples` / `--all-samples` | a checkpoint + crops for each sample | `<sample>/predict_class_confident_<thr>.xlsx`, `predict_class_uncertain_<thr>.xlsx`, plus `test_run_config.csv` |
| 4 | `Visualize_heatmap.py` | `--data-root` | those two workbooks | `--out-pdf` (default `heatmap_nu_mito.pdf`, a **relative** path), `visualize_heatmap_run_config.csv`, and a Tk window unless `--no-gui` |
| 5 | `Finetune_LUMINA.py` | `--checkpoint`, `--out`, and a population from `--data-root` or `--manifest` | a checkpoint + crops under `--data-root`, optionally a second dish under `--eval-root`, an optional `--drop-list` | under `--out`: `finetune_predictions_K<k>_seed<s>.csv`, `finetune_summary_K<k>_seed<s>.csv`, `support_cells_K<k>_seed<s>.csv`, `finetune_per_seed_K<k>.csv`, `finetune_run_config.csv`, plus `finetune_across_seeds_K<k>.csv` when more than one seed was run, and with `--save-heads` a `best_model_fewshot_K<k>_seed<s>.pth` |

A minimal run of each, in order:

```bash
python Data_prep.py --data-root /path/to/dataset --samples sampleA,sampleB

python Train_LUMINA.py --data-root /path/to/single_anchor \
    --single-anchor-manifest single_anchor.csv \
    --dual-root /path/to/dual_anchor --dual-samples-file dual_samples.txt \
    --checkpoint best_model_fine-tune.pth --out ./train_out

python Test_LUMINA.py --checkpoint ./train_out/best_model_fine-tune.pth \
    --data-root /path/to/dataset --samples sampleA,sampleB

python Visualize_heatmap.py --data-root /path/to/dataset --no-gui
```

**`--samples` is not optional in scripts 1 and 3, and has no default.** Name the
folders, or ask for every one of them with `--all-samples`; passing neither exits
1 with a message naming both, and passing both exits 1 as well. That is
deliberate, because those two scripts write into the folders they are given:
`Data_prep.py` **clears** a sample's `--seg-folder` before writing it, and
`Test_LUMINA.py` **overwrites** the two workbooks inside it unless `--out`
redirects them. Each prints its resolved list, one destination per line, before
the first deletion or the first write — with a line saying nothing has happened
yet — so a wrong `--all-samples` is visible while it is still reversible.
`Visualize_heatmap.py` is the exception: it only reads, so its `--samples` still
defaults to scanning, and it has no `--all-samples`.

**The dual-anchor list is ordered, and the order is part of the split.**
`Train_LUMINA.py` loads the folders in the order `--dual-samples` /
`--dual-samples-file` gives them, and the stage-2 validation split is taken by row
position, so the same names in a different order hold out different cells at the
same `--seed` — a different `val_df.xlsx` and a different checkpoint. A rerun has
to pass the same list in the same order, not merely the same names. The resolved
order is printed at startup and recorded as `resolved_dual_samples` in
`train_run_config.csv`.

**Every script writes its resolved configuration beside its output** — one
`flag,value` CSV per run, named in the table above. That is the record of what
produced a directory of results; ask for it before trusting a number whose
provenance is in doubt.

**Three flags have to agree across scripts, and nothing enforces it:**

- `--seg-folder`. `Data_prep.py` writes `seg_5D_calib` by default,
  `Train_LUMINA.py` reads `seg_5D` only, and `Test_LUMINA.py` /
  `Finetune_LUMINA.py` default to `auto` (prefer `seg_5D_calib`, fall back to
  `seg_5D`). Preparing crops for training means
  `Data_prep.py --seg-folder seg_5D`, not renaming a folder afterwards.
- `--confidence-threshold`. Test writes the threshold **into the workbook file
  name**; `Visualize_heatmap.py` rebuilds that name from its own flag. Run the
  two with different values and the heatmap silently finds nothing.
- `--num-classes`, and the class order. It has to match the checkpoint, which
  records neither.

Script 5 is optional and only needed when a checkpoint is being moved to a cell
line it was not trained on. It fine-tunes `fc_nu` and `fc_mito` on a few labelled
cells from the new dish, freezes everything else, and reports detection and
accuracy separately. `K` is per barcode **combination**, not per sample folder,
and it reads `seg_5D_calib` in preference to `seg_5D` unless `--seg-folder` says
otherwise — both of which it prints at startup.

**That default is not what the measured runs read.** They read `seg_5D` only, so
on a dish holding both folders the default `--seg-folder auto` feeds the network
calibrated crops where the protocol used uncalibrated ones — a different input
distribution, not the same experiment. Do not tell a user the default reproduces
the protocol; tell them to pass `--seg-folder seg_5D` if that is what they want.
That is one of several ways a run here differs from the manuscript's runs; the
full list, and the honest answer to "does `--seed 0` give me your cells" (it does
not), are in `references/domain-adaptation.md`.

**The sample folder is the unit of work**, as in the plugin — but the file names
inside it are not the plugin's. LUMINA wants `<fov>-1.tif` and
`<fov>-sum_seg.npy`; the plugin writes `<fov>_ch1.tif` and `<fov>_sum_seg_n.npy`.
Nothing in this repository converts one to the other. See
`references/data-and-prep.md`.

## Reference files

Read the one you need; do not read them all.

| File | Read it when |
|---|---|
| `references/data-and-prep.md` | **Start here for any "will it find my data" question.** The exact folder and file names each script expects, the six planes and their order, array shapes and dtypes, the spreadsheet columns, and every flag of `Data_prep.py` with its default. |
| `references/training.md` | Training from scratch, or `Train_LUMINA.py`'s own two-stage fine-tune: which checkpoint feeds which, what is frozen when, the loss weighting, the LR schedule, and what the default switch values actually do. Not the same thing as adapting to a new cell line — for that, read `domain-adaptation.md`. |
| `references/inference-and-heatmap.md` | Running `Test_LUMINA.py` or reading its output: the confidence score formula, the two workbooks, and what the heatmap does and does not show. |
| `references/troubleshooting.md` | Something hung, crashed, or produced a result that cannot be right. |
| `references/domain-adaptation.md` | **Someone wants to use LUMINA on a different cell line.** Why a checkpoint degrades across cell lines, the few-shot head-only recipe that fixes it, how to run `Finetune_LUMINA.py`, how many cells they need to label, the triage gate that precedes it, and the two traps in reading the result. |

## Answering well here

**Name the flag, never a line to edit.** "Pass `--data-root`" is the answer;
"set `base_folder` near the top of `main()`" is now wrong, because no such
constant exists in any of the five scripts. If you are unsure of a flag name or a
default, read the `add_argument` block rather than guessing — `--help` is
authoritative and cheap to run.

**A flag's default is the value the script shipped with.** That is the whole
point of the conversion, so "what does it do by default" and "what did the
original file do" have the same answer for every optional flag. There are two
kinds of exception. The required path flags have no default at all. And the
sample lists of `Data_prep.py` and `Test_LUMINA.py` have none either — those two
write into the folders they are handed, so the choice is `--samples` or
`--all-samples` and neither is assumed. `Visualize_heatmap.py`'s `--samples` is
the one that is deliberately *wider* than the shipped file: empty means every
folder already holding a confident workbook at this threshold, where the shipped
file plotted whatever short list was left uncommented. Tell a user to pass
`--samples` explicitly when they want to reproduce a particular figure.

**Class indices are module constants now, in four places.**
`NU_CLASS_MAP` / `MITO_CLASS_MAP` are written out as module-level literals in
`Train_LUMINA.py`, `Test_LUMINA.py`, `Visualize_heatmap.py` and
`Finetune_LUMINA.py` — `grep -n 'NU_CLASS_MAP = ' *.py` finds all four — and they
are character-for-character identical: `N10` is class 1 and `N1` is class 7.
Index 0 means *no anchor / unknown* and is never a training target. There is
deliberately **no flag** to permute them — a checkpoint stores no mapping, so a
way to reorder them would be a silent way to mislabel every cell. `--num-classes`
sets only the *width* of the heads, not the order. If you change the panel,
change it in all four files; the three that a run touches each print their map at
startup, so a mismatch is visible in the console log of the run that produced a
checkpoint.

**"5D" means six planes.** The folder is `seg_5D` / `seg_5D_calib` and the file
is `cell<id>_5D.tif`, but the stack is `[G, S, ratio1, ratio2, ratio3,
intensity]` — the five-dimensional fingerprint plus the intensity plane the
network normalises. The `np.stack([c_gc, c_sc, ci1, ci2, ci3, isum], axis=0)`
that builds `cell_stack` in `Data_prep.py` is authoritative, and the numbered
comment directly above it now names those same six.

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
parameter, say so and quote the flag's default.

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

How many cells are prepared, and how many are too big for the canvas the loaders
pad to (`--crop-size`, default 256 in both `Train_LUMINA.py` and
`Test_LUMINA.py`):

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

The last line is the only record of how many classes the model was built with.
`--num-classes` defaults to 8 in both `Train_LUMINA.py` and `Test_LUMINA.py` —
read the `add_argument('--num-classes', ...)` block in either, or run `--help` —
and a checkpoint built with a different width fails to load rather than loading
wrongly.

What a finished run was actually given — the file to read before quoting any
number back to a user:

```python
import pandas as pd
print(pd.read_csv('test_run_config.csv').to_string(index=False))
```
