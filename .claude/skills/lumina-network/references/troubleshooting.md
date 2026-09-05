# What goes wrong

Every entry below is a failure mode visible in the source, with the check that
identifies it. None of the behavioural ones is fixed in the repository — read
this before telling a user their data is at fault.

All five scripts are argparse-driven, so a fix is a flag, never a line to edit.

---

## `error: the following arguments are required: …`

argparse, exit code 2, before anything is read. The required flags are the paths
that used to be lab drive letters, and they deliberately carry **no** default —
a default nobody else can reach is worse than none:

| Script | Required |
|---|---|
| `Data_prep.py` | `--data-root` |
| `Train_LUMINA.py` | `--data-root`, `--single-anchor-manifest`, `--dual-root`, `--out` |
| `Test_LUMINA.py` | `--checkpoint`, `--data-root` |
| `Visualize_heatmap.py` | `--data-root` |
| `Finetune_LUMINA.py` | `--checkpoint`, `--out` |

Three more pairs are checked *after* parsing, so the message names both
alternatives rather than one flag, and giving **both** of a pair is an error too:

| Script | Give one of |
|---|---|
| `Data_prep.py` | `--samples` or `--all-samples` |
| `Test_LUMINA.py` | `--samples` or `--all-samples` |
| `Train_LUMINA.py` | `--dual-samples` or `--dual-samples-file`, **and** `--checkpoint` or `--from-scratch` |

Which sample list, or which starting point, trained a checkpoint has to be
unambiguous — hence the second column rather than a default.

## `Give --samples or --all-samples; there is no default.`

Exit 1, from `resolve_samples`, before anything is read or written. It is the
answer to `python Data_prep.py --data-root <root>` and to
`python Test_LUMINA.py --checkpoint <ckpt> --data-root <root>` — both of which
used to run, and neither of which does now.

**Why there is no default.** Both scripts write into the folders they are handed:
`Data_prep.py` **clears** each sample's `--seg-folder` before writing it, and
`Test_LUMINA.py` **overwrites** `predict_class_confident_<threshold>.xlsx` and
`predict_class_uncertain_<threshold>.xlsx` inside each sample folder unless
`--out` redirects them. "Every folder under the root" is therefore something you
ask for by name — `--all-samples` — not something you get by typing the shorter
command.

**What to answer.** Name the folders with `--samples a,b`, or pass
`--all-samples` and read the list it prints. Both scripts print every destination,
one per line, followed by a line saying nothing has been deleted or written yet,
before they touch anything. `Visualize_heatmap.py` is unaffected: it only reads,
so its `--samples` still defaults to scanning and it has no `--all-samples`.
`Finetune_LUMINA.py` is unaffected too — its `--samples` restricts a scan that
otherwise takes every sample folder, and it writes only under `--out`.

## A path flag points somewhere that is not a folder

Every one of these exits 1 with a single line naming the flag, no traceback. If a
user reports a stack trace from a bad path, they are on an old copy of the file.

The common shape mistake is pointing `--data-root` at **one sample** instead of
at the folder that holds the sample folders. Each script says so in its error,
because the two look identical from the command line.

---

## Training starts and prints nothing, forever

**Symptom.** `Train_LUMINA.py` reaches "Calculating class weights..." — the line
`train_model` prints just before `get_class_counts` — or does not even reach it,
one core sits at 100 %, no epoch line ever appears, no error.

**Cause.** `FluorescenceDataset.__getitem__` skips any sample whose nuclear *or*
mitochondrial label is 0 by advancing to the next row and `continue`-ing inside
`while True`. Every row produced by `load_training_data` has exactly one label 0
by construction — its nucleus loop appends `'mito_class': 0  # No mito` and its
mitochondria loop appends `'nu_class': 0,  # No nu`. So the stage-1
dataset can never yield a sample, and the loop spins.

**Check.** Stage 1 only runs under **`--from-scratch --finetune` together**. The
startup banner says which stages will run:

```
stage 1 (initial): skipped   epochs: 180   lr-drops: [30, 70]
```

A default run — with `--checkpoint` and without `--finetune` — cannot hit this.
The script also prints a four-line warning before entering the stage-1 path, so a
user who hit it has that warning in their console; ask for it.

**What to say.** Stage-1 pre-training cannot be re-run from this file as
written. Either the zero-label filter or the label construction has to change,
and which one is right depends on whether class 0 was meant to be a trainable
"no anchor" target — the file does not say. Both defects are left in place on
purpose, because either repair changes what a training run produces. Do not
change it silently on the user's behalf.

---

## `TypeError: cannot unpack non-sequence NoneType`

At the stage-1 call site in `Train_LUMINA.main`, which reads
`train_losses, val_losses, train_accuracies, val_accuracies = train_model(...)`.
`train_model` returns nothing — its `return train_losses, val_losses, ...` is
commented out at the end of the function. Only reachable under
`--from-scratch --finetune`, i.e. the same path as the entry above, and named in
the same warning. The live stage-2 call discards the result and is fine.

---

## `ValueError: weight tensor should be defined either for all 8 classes…`

**Cause.** `WeightedCrossEntropyLoss` sizes its weight vector as
`torch.zeros(max(class_counts.keys()) + 1)`, once per head in
`WeightedCrossEntropyLoss.__init__`, from the classes actually present in the **training split**. The heads emit
`--num-classes` logits, default 8. If the highest class index in your data is 6 —
say the manifest names no folder for `N1`, the last key of the map — the weights
are length 7 and `CrossEntropyLoss` refuses.

**Check.** The two lines `train_model` prints before epoch 1:

```
Nuclear class distribution: {1: …, 2: …, …}
Mitochondrial class distribution: {…}
```

If the largest key is not `--num-classes - 1`, this is your error. It can also
appear from a `--val-split` draw that happens to leave a rare class entirely on
the validation side — try a different `--seed`.

**Fix direction.** Either keep the highest-indexed class in the training split,
or size the vector to `--num-classes`. **Lowering `--num-classes` is not the
fix**: it has to match the checkpoint, and the class maps are fixed — see the
next entry.

---

## Every prediction is confidently wrong, or shifted by one class

**Cause.** Class indices are module-level literals, and the checkpoint stores no
mapping. `Train_LUMINA.py`, `Test_LUMINA.py`, `Visualize_heatmap.py` and
`Finetune_LUMINA.py` each carry their own copy — `grep -n 'NU_CLASS_MAP = ' *.py`
in `LUMINA_classification/` finds all four. Edit one and not the others and every label is permuted with no
error anywhere.

This used to be worse: the indices were *derived* from the order of a dict of
folder names whose entries were half commented out, so uncommenting one line
renumbered every class after it. That derivation is gone, and there is
deliberately no flag to permute the order — but four copies still have to agree,
and nothing checks them against each other.

**Check.** Three of the four scripts print their maps at startup:

```
nu_class_map: {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
```

`Train_LUMINA.py` also writes them into `train_run_config.csv` as
`resolved_nu_class_map` / `resolved_mito_class_map`, which survives after the
console log is gone — compare that file against the inference run's console
output. If neither exists for the run that produced the checkpoint, there is no
way to recover the mapping from the `.pth` file; say so rather than guessing.

Note that `--nu-classes` / `--mito-classes` in `Visualize_heatmap.py` reorder the
**figure** only. They cannot cause this, and they cannot fix it either.

---

## `RuntimeError: Error(s) in loading state_dict for DualHeadConvNet`

Size mismatch on `fc_nu.6.weight` / `fc_mito.6.weight` means the checkpoint was
built with a different width than `--num-classes` (default 8). The checkpoint's
own value:

```python
import torch
print(torch.load('best_model_fine-tune.pth', map_location='cpu')['fc_nu.6.weight'].shape)
```

`Train_LUMINA.py` catches this one — the `try: model.load_state_dict(state_dict)`
in `main` — and exits with a message naming
`--num-classes` and `--checkpoint`, because a partially loaded model still trains
and still writes a checkpoint of its own. `Test_LUMINA.py` lets the `RuntimeError`
through.

Missing or unexpected keys instead means the two copies of `DualHeadConvNet`
have drifted — the class is duplicated verbatim in `Train_LUMINA.py` and
`Test_LUMINA.py`, and nothing keeps them in sync. `Finetune_LUMINA.py` does
`from Test_LUMINA import DualHeadConvNet, pad_image` rather than carrying a third
copy, so a change there reaches two scripts and not the trainer.

---

## `FileNotFoundError: …/seg_5D/cell12_5D.tif`

The four scripts default to different folders, on purpose:

| Script | Flag default |
|---|---|
| `Data_prep.py` writes | `--seg-folder seg_5D_calib` |
| `Train_LUMINA.py` reads | `--seg-folder seg_5D` |
| `Test_LUMINA.py` reads | `--seg-folder auto` → `seg_5D_calib`, else `seg_5D` |
| `Finetune_LUMINA.py` reads | `--seg-folder auto`, same rule |

So preparing crops and then training with no flags looks in the wrong place. Fix
it with a flag on either side — `Data_prep.py --seg-folder seg_5D`, or
`Train_LUMINA.py --seg-folder seg_5D_calib` — not by renaming a folder, so that
the run config records which one was used.

`Test_LUMINA.py` prints the folder it actually read for every sample. Check that
line before assuming the crops are missing.

---

## The forward pass crashes, or the wrong planes are used

The network hardcodes **six** input planes: the `for _ in range(6)` that builds
`self.input_heads` in `DualHeadConvNet.__init__`, and the
`[head(x[:, i:i + 1]) for i, head in enumerate(self.input_heads)]` in its
`forward` — in both `Train_LUMINA.py` and `Test_LUMINA.py`, which carry the class
twice. There is no flag for the plane count, and `Data_prep.py` has no way to
write anything but six.

- **Fewer than six planes** — the slice for `i ≥ n` is empty and the first
  convolution fails on a zero-size channel dimension.
- **More than six planes** — no error at all. The first six are used, in order,
  and `normalize_intensity` still divides the **last** plane by its max, so the
  plane the network reads as intensity is un-normalised and a plane it never
  sees gets scaled. Silent, and it will look like a training problem.

```python
import tifffile as tiff; print(tiff.imread('seg_5D_calib/cell1_5D.tif').shape)
```

Expect `(6, h, w)`. Crops from anywhere other than `Data_prep.py` are the case to
suspect.

---

## Cells are missing from the results, and `Cell_Label` does not match the image

**Cause.** A crop whose bounding box exceeds `--crop-size` (default 256) is
skipped by `FluorescenceDataset.__getitem__`'s
`idx = (idx + 1) % len(self.df); continue` — no warning. At inference the result
row is then built from `test_df.iloc[i]` where `i` counts *loader iterations*, so
after a skip the prediction of cell *i+1* is
written under the `Cell_Label` of cell *i*, and the last cell in the folder is
predicted twice. Every subsequent row in that folder is shifted.

**Check.** Count oversized crops before trusting a folder's predictions:

```python
import glob, tifffile as tiff
big = [p for p in glob.glob('seg_5D_calib/cell*_5D.tif')
       if max(tiff.imread(p).shape[1:]) > 256]
print(len(big), 'oversized:', big[:5])
```

Zero oversized crops means the indices line up. Otherwise the folder's
`Cell_Label` column is unreliable from the first skipped cell onwards.

**Raising `--crop-size` is not a free fix.** The crop is *padded*, never resized,
so a larger canvas changes what the network sees relative to the checkpoint it
was trained with. Leave it at the training value and treat oversized cells as a
segmentation problem.

---

## `seg_5D_calib` only contains the last field of view

**Cause.** `Data_prep.py` clears the output folder — the
`for file in os.listdir(out_dir)` block that runs right after `out_dir` is built —
**inside** the per-FOV loop, and names its output `cell<id>_5D.tif` with no FOV in
the name.
Two FOVs in one sample folder therefore collide on every id, and the clear
removes the earlier FOV's cells entirely.

**`--keep-existing` does not fix this.** It makes same-numbered cells from
different fields of view overwrite each other instead of the earlier FOV being
deleted wholesale — a different way to keep only the last one. The flag's own
help says so. This is long-standing behaviour, left alone deliberately.

**Check.** `len(os.listdir('seg_5D_calib'))` against the number of files in
`raw/` times the cells per field. If the sample has more than one FOV, only the
last survives. The startup line `[<sample>] raw/: N field(s) of view  ->  <seg
folder>/`, printed once per sample, tells you N before the run finishes.

---

## `AssertionError: Torch not compiled with CUDA enabled`

`--device` defaults to `cuda:0` in both `Train_LUMINA.py` and `Test_LUMINA.py`,
with no CPU fallback — that is the literal both scripts have always used, and it
is kept so a machine without a GPU fails immediately rather than quietly starting
a run that would never finish. Pass `--device cpu`.

`CUDA error: invalid device ordinal` instead means `cuda:0` does not exist on
this host; pass the ordinal that does. `Finetune_LUMINA.py` is the exception —
its `--device` defaults to `auto`.

---

## `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`

In `Data_prep.py` at `C1 = f1 * I1`, the first calibration multiply. The four
`cv2.imread(..., -1)` calls just above it return `None` rather than raising when
they cannot read a file, and that multiply is the next thing that touches the
result. Either an
`intensity/<fov>-{1..4}.tif` is missing, or the path contains characters
`cv2.imread` cannot handle on Windows. Confirm by listing the four files for
that FOV.

---

## NaNs reach the network

`np.nan_to_num` is applied to the loaded crop in both copies of
`FluorescenceDataset.__getitem__` and *then* `normalize_intensity` divides by
`np.max`. A crop whose intensity plane is entirely zero therefore produces fresh
NaNs **after** the cleaning step, and they go straight into the model. A cell can
end up with an all-zero intensity plane when its mask covers only background, or
when `--intensity-threshold` was set high enough to zero every pixel.

```python
import tifffile as tiff, numpy as np
a = tiff.imread('seg_5D_calib/cell1_5D.tif')
print(float(a[-1].max()), np.isnan(a).sum())
```

A zero in the first number is the problem case.

---

## Phasor values look wrong for every cell

Check four flags of `Data_prep.py` before anything else, all four of which
describe the **instrument** rather than the sample:

| Flag | Default | Reaches |
|---|---|---|
| `--rep-rate-mhz` | `78.1` | `freq = rep_rate_mhz/1000` in `calcu_phasor_info` — scales the whole phasor |
| `--tau-resolution` | `0.09696969696999999` | ns per time bin, `t = np.arange(len(seg)) * tau_resolution` |
| `--phi-calib` | `-0.0125` | phase added to every pixel |
| `--m-calib` | `1.0292` | modulation multiplying every pixel |

The defaults are the values this project's crops were made with; they are
meaningless on another microscope and carrying them over moves every *G* and *S*
silently. `--rep-rate-mhz` is the one users miss, because it is the laser rather
than a processing choice. `--phi-calib 0 --m-calib 1.0` gives the uncalibrated
values if you need to see them.

Every one of these is recorded in `data_prep_run_config.csv` beside the crops, so
a folder whose phasors look wrong can be checked against the run that made it.

---

## `--smooth wavelet` raises

The `smooth_option == 'wavelet'` branch of `calcu_phasor_info` assigns
`seg = pywt.dwt(seg, 'db1')`, which returns a *pair* of coefficient arrays; the
next lines multiply that by the time axis `t` and the shapes cannot broadcast. Only `none` and `median` work. `wavelet` is still
offered by the parser because the code path exists and was left as found rather
than quietly changed — the flag's help says so.

Similarly, `--calculate-lifetime` runs a `curve_fit` per pixel inside
`calcu_phasor_info` and then throws the result away — the caller unpacks
`g, s, _, _, _, _`, discarding τ and χ². It costs a
great deal of time and changes no output; the output is byte-identical with and
without it.

---

## `Visualize_heatmap.py`: no confident workbook found

The script exits 1 naming the flag and the filename it looked for. Three causes,
in order:

1. **The thresholds do not match.** `Test_LUMINA.py` writes
   `predict_class_confident_<threshold>.xlsx`; the viewer rebuilds the same name
   from **its own** `--confidence-threshold` (default 0.6, same as
   `Test_LUMINA.py`'s). Run inference at 0.7 and plot at the default and the
   viewer looks for a file that does not exist. Pass the same value to both.
   `test_run_config.csv` records the exact spelling as `workbook_suffix`.
2. **The roots do not match.** Point `--data-root` at the root
   `Test_LUMINA.py` wrote into — which is its own `--data-root` by default, but
   its `--out` if one was given. `--data-root-2` is searched only as a fallback,
   and a dataset split across two roots is the other common cause.
3. **Inference produced no confident cells at all** — the writer is guarded by
   `if results:` in `Test_LUMINA.test_model`.

A missing *uncertain* workbook is only a warning; the detection rate then counts
those cells as zero and reads 100 %.

---

## The training workbook gets slower every epoch

`combination_accuracies_<phase>.xlsx` is reopened in append mode and rewritten
once per epoch — the `pd.ExcelWriter(..., mode='a', if_sheet_exists='replace')`
near the end of `train_model`'s epoch loop — gaining a sheet each time, up to 800
at the default `--finetune-epochs`. openpyxl rewrites the whole workbook each time, so
late epochs pay more than early ones. Lowering `--finetune-epochs` lowers it
proportionally.

`test_train_val_log.xlsx` is rewritten whole every epoch too, and its
name carries no phase, so a second run pointed at the same `--out` overwrites the
first run's curve. `val_df.xlsx` is overwritten as well; only
`combination_accuracies_<phase>.xlsx` accumulates. Point each run at its own
`--out` if you want to keep both.
