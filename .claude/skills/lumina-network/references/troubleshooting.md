# What goes wrong

Every entry below is a failure mode visible in the source, with the check that
identifies it. None of them is fixed in the repository — read this before
telling a user their data is at fault.

---

## Training starts and prints nothing, forever

**Symptom.** `Train_LUMINA.py` reaches "Calculating class weights…" (or does not
even reach it), one core sits at 100 %, no epoch line ever appears, no error.

**Cause.** `FluorescenceDataset.__getitem__` skips any sample whose nuclear *or*
mitochondrial label is 0 by advancing to the next row and `continue`-ing inside
`while True` (lines 69–72). Every row produced by `load_training_data` has
exactly one label 0 by construction — nucleus-only folders get `mito_class: 0`
(line 202), mitochondria-only folders get `nu_class: 0` (line 214). So the
stage-1 dataset can never yield a sample, and the loop spins.

**Check.** Which stage is running:

```python
# in main(): line 623 and line 499
use_pretrained = True    # -> stage 1 is skipped entirely, this cannot bite
use_finetune   = False
```

Stage 1 only runs when `use_pretrained` is `False` **and** `use_finetune` is
`True` (line 630). The shipped file avoids the problem by loading a checkpoint
instead.

**What to say.** Stage-1 pre-training cannot be re-run from this file as
written. Either the zero-label filter or the label construction has to change,
and which one is right depends on whether class 0 was meant to be a trainable
"no anchor" target — the file does not say. Do not change it silently on the
user's behalf.

---

## `TypeError: cannot unpack non-sequence NoneType`

At `Train_LUMINA.py` line 631. `train_model` returns nothing — its `return`
statement is commented out on line 477 — but the stage-1 call site still
unpacks four values from it. Same trap in the commented-out stage-2 call on
lines 661–665. The live stage-2 call on line 666 discards the result and is
fine.

---

## `ValueError: weight tensor should be defined either for all 8 classes…`

**Cause.** `WeightedCrossEntropyLoss` sizes its weight vector as
`torch.zeros(max(class_counts.keys()) + 1)` (lines 253 and 259), from the
classes actually present in the **training split**. The heads emit
`num_classes = 8` logits. If the highest class index in your data is 6 — say
you dropped `N1`, the last key of the dict — the weights are length 7 and
`CrossEntropyLoss` refuses.

**Check.** The two lines the trainer prints before epoch 1:

```
Nuclear class distribution: {1: …, 2: …, …}
Mitochondrial class distribution: {…}
```

If the largest key is not `num_classes - 1`, this is your error. It can also
appear from a random 80/20 split that happens to leave a rare class entirely on
the validation side.

**Fix direction.** Either keep the highest-indexed class in the training split,
or size the vector to `num_classes`. Note that dropping a class from the dict
literal *renumbers every class after it* — see the next entry.

---

## Every prediction is confidently wrong, or shifted by one class

**Cause.** Class indices come from the order of the `nu_files` / `mito_files`
dict literal (`{key: idx + 1 for idx, key in enumerate(...)}`), and the
checkpoint stores no mapping. `Train_LUMINA.py` (lines 564–591) and
`Test_LUMINA.py` (lines 501–528) each carry their own copy of that literal.
Uncomment a line in one and not the other, or reorder the keys, and every label
is permuted with no error anywhere.

**Check.** Both scripts print their maps at startup:

```
nu_class_map: {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
```

Compare them character for character with the run that produced the checkpoint.
If the training run's console log is gone, there is no way to recover the
mapping from the `.pth` file — say so rather than guessing.

---

## `RuntimeError: Error(s) in loading state_dict for DualHeadConvNet`

Size mismatch on `fc_nu.6.weight` / `fc_mito.6.weight` means the checkpoint was
built with a different `num_classes` than the 8 hardcoded on `Test_LUMINA.py`
line 537. The checkpoint's own value:

```python
import torch
print(torch.load('best_model_fine-tune.pth', map_location='cpu')['fc_nu.6.weight'].shape)
```

Missing or unexpected keys instead means the two copies of `DualHeadConvNet`
have drifted — the class is duplicated verbatim in `Train_LUMINA.py` (113–184)
and `Test_LUMINA.py` (108–179), and nothing keeps them in sync.

---

## `FileNotFoundError: …/seg_5D/cell12_5D.tif`

`Data_prep.py` writes **`seg_5D_calib`**. `Test_LUMINA.py` tries that first and
falls back to `seg_5D` (lines 56–59). `Train_LUMINA.py` only ever looks in
`seg_5D` (lines 61, 195, 208). Training on freshly prepared data means renaming
the folder or editing those three lines. Check which folders exist before
assuming the crops are missing.

---

## The forward pass crashes, or the wrong planes are used

The network hardcodes **six** input planes: `range(6)` on line 130 and
`x[:, i:i+1]` in `forward`.

- **Fewer than six planes** — the slice for `i ≥ n` is empty and the first
  convolution fails on a zero-size channel dimension.
- **More than six planes** — no error at all. The first six are used, in order,
  and `normalize_intensity` still divides the **last** plane by its max, so the
  plane the network reads as intensity is un-normalised and a plane it never
  sees gets scaled. Silent, and it will look like a training problem.

```python
import tifffile as tiff; print(tiff.imread('seg_5D_calib/cell1_5D.tif').shape)
```

Expect `(6, h, w)`. `Data_prep.py`'s stale comment on line 227 lists eight plane
names; if someone "restored" the missing planes to match that comment, this is
what happens.

---

## Cells are missing from the results, and `Cell_Label` does not match the image

**Cause.** A crop whose bounding box exceeds 256×256 is skipped by advancing the
index (`Test_LUMINA.py` lines 69–71) — no warning. At inference the result row
is then built from `test_df.iloc[i]` where `i` counts *loader iterations*
(line 308), so after a skip the prediction of cell *i+1* is written under the
`Cell_Label` of cell *i*, and the last cell in the folder is predicted twice.
Every subsequent row in that folder is shifted.

**Check.** Count oversized crops before trusting a folder's predictions:

```python
import glob, tifffile as tiff
big = [p for p in glob.glob('seg_5D_calib/cell*_5D.tif')
       if max(tiff.imread(p).shape[1:]) > 256]
print(len(big), 'oversized:', big[:5])
```

Zero oversized crops means the indices line up. Otherwise the folder's
`Cell_Label` column is unreliable from the first skipped cell onwards.

**Also worth knowing:** the crop is *padded*, never resized. `resize_image` is
defined in both scripts and called by neither.

---

## `seg_5D_calib` only contains the last field of view

**Cause.** `Data_prep.py` clears `out_dir` (lines 178–186) **inside** the
per-FOV loop, and names its output `cell<id>_5D.tif` with no FOV in the name.
Two FOVs in one sample folder therefore collide on every id, and the clear
removes the earlier FOV's cells entirely.

**Check.** `len(os.listdir('seg_5D_calib'))` against the number of files in
`raw/` times the cells per field. If the sample has more than one FOV, only the
last survives.

---

## `AssertionError: Torch not compiled with CUDA enabled`

Both scripts build `torch.device(f'cuda:{gpu_id}')` unconditionally
(`Train_LUMINA.py` line 497, `Test_LUMINA.py` line 364) and move the model to it
before anything else. There is no CPU fallback and no flag; a CPU-only machine
needs those lines edited. `CUDA error: invalid device ordinal` instead means
`gpu_id = 0` (line 496 / 363) does not exist on this host.

---

## `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`

In `Data_prep.py` around line 161. `cv2.imread` returns `None` rather than
raising when it cannot read a file, and the calibration multiply is the next
thing that touches it. Either an `intensity/<fov>-{1..4}.tif` is missing, or the
path contains characters `cv2.imread` cannot handle on Windows. Confirm by
listing the four files for that FOV.

---

## NaNs reach the network

`np.nan_to_num` is applied to the loaded crop (line 72 / 77) and *then*
`normalize_intensity` divides by `np.max`. A crop whose intensity plane is
entirely zero therefore produces fresh NaNs **after** the cleaning step, and
they go straight into the model. A cell can end up with an all-zero intensity
plane when its mask covers only background.

```python
import tifffile as tiff, numpy as np
a = tiff.imread('seg_5D_calib/cell1_5D.tif')
print(float(a[-1].max()), np.isnan(a).sum())
```

A zero in the first number is the problem case.

---

## Phasor values look wrong for every cell

Check three constants in `Data_prep.py` before anything else:
`tau_resolution` (line 36, ns per bin), `freq` (**line 118**, inside
`calcu_phasor_info`, `78.1/1000` = 78.1 MHz in GHz), and `phi_calib` / `m_calib`
(lines 18–19). The repetition rate is the one that is *not* in the
user-editable block at the top of the file, and it scales the whole phasor. The
identity calibration `phi_calib = 0, m_calib = 1.0` sits commented out on lines
22–23 if you need to see the uncalibrated values.

---

## `smooth_option = 'wavelet'` raises

Line 106 assigns `seg = pywt.dwt(seg, 'db1')`, which returns a *pair* of
coefficient arrays; the next lines multiply that by the time axis `t` and the
shapes cannot broadcast. Only `None` and `'median'` work.

Similarly, `calculate_lifetime = True` (line 27) fits a mono-exponential per
pixel and then throws the result away — line 209 unpacks τ and χ² into `_`. It
costs time and changes no output.

---

## `Visualize_heatmap.py`: "Confident Excel file not found"

Three causes, in order:

1. **The threshold is in the file name.** `Test_LUMINA.py` writes
   `predict_class_confident_<threshold>.xlsx`; the viewer builds the same name
   from its own `confidence=0.6` default (line 12), which `main()` never
   overrides. Run inference at 0.7 and the viewer looks for a file that does not
   exist. Edit line 12, or pass `confidence` at the call on lines 222–224.
2. **A different second root.** `base_folder2` is `E:\…` here (line 172) and
   `I:\…` in `Test_LUMINA.py` (line 347). A sample found under one is not found
   under the other.
3. Inference was run with `out_pred=False`, or produced no confident cells at
   all — the writer is guarded by `if results:` (`Test_LUMINA.py` line 325).

The uncertain workbook missing is only a warning; the detection rate then
counts those cells as zero and reads 100 %.

---

## The training workbook gets slower every epoch

`combination_accuracies_<phase>.xlsx` is reopened in append mode and rewritten
once per epoch (lines 462–466), gaining a sheet each time — up to 800 for a full
stage-2 run. openpyxl rewrites the whole workbook each time, so late epochs pay
more than early ones. `test_train_val_log.xlsx` is rewritten whole every epoch
too, and its name carries no phase, so a second run in the same `out_folder`
overwrites the first run's curve.
