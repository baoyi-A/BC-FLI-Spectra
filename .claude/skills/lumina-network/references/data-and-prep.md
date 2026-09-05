# What must be on disk, and the script that puts it there

Every path below is traced from the source. Where a script builds a path with
`os.path.join`, the exact string is quoted. Where a value is set by a flag, the
flag is named — none of these scripts has editable constants any more, and
`--help` is authoritative for anything not listed here.

---

## 1. What `Data_prep.py` reads

One *sample folder* per acquisition, under `--data-root` (required, no default),
named by `--samples` or taken wholesale with `--all-samples` — one of those
two is required and there is no default:

```
<--data-root>/<sample>/
├── raw/<fov>.<ext>                 ← only the STEM is used, to name everything else
├── flim_stack/<fov>-sum.tif        ← decay stack, T×H×W, time first, channel-summed
├── intensity/<fov>-1.tif           ← the four spectral intensity images,
├── intensity/<fov>-2.tif             read with cv2.imread(path, -1) = IMREAD_UNCHANGED
├── intensity/<fov>-3.tif
├── intensity/<fov>-4.tif
├── intensity/<fov>-sum_seg.npy     ← np.load(..., allow_pickle=True).item()['masks']
└── seg_5D_calib/                   ← WRITTEN by this script, see --seg-folder
```

`raw/` is enumerated with `os.listdir` and every entry is treated as a field of
view: `fov = os.path.splitext(fname)[0]`. Anything else in that folder — a
stray `.txt`, a subdirectory — becomes a FOV name whose other files do not
exist, and the script dies on the first missing file.

The mask file is the Cellpose `_seg.npy` format: a pickled dict with a `masks`
key holding an integer label image, labels `1..N`. Cells are iterated as
`for cid in tqdm(range(1, masks.max() + 1))` and empty labels are skipped, so gaps
are harmless.

A sample named in `--samples` that has no `raw/` is a **hard error**, not a skip.
That is deliberate: a silent skip is indistinguishable from a sample that
legitimately produced no cells. `--all-samples` applies the same test the other
way round: it selects exactly the immediate subfolders that *do* have a `raw/`,
in sorted order, and refuses to run when there are none.

**These names are not the plugin's.** The napari plugin writes
`intensity/<fov>_ch1.tif`, `intensity/<fov>_sum.tif`,
`flim_stack/<fov>_sum.tif` and `intensity/<fov>_sum_seg_n.npy` — the same kinds
of file with a different separator, and split into an N and a P mask instead of
one. No script in this repository renames one convention into the other; a
plugin sample folder must be renamed by hand before `Data_prep.py` will see it,
and the single `-sum_seg.npy` must be whichever mask you want cells cut from.

---

## 2. What `Data_prep.py` writes

```
<--data-root>/<sample>/<--seg-folder>/cell<label>_5D.tif
<--data-root>/<sample>/data_prep_run_config.csv
```

`--seg-folder` defaults to `seg_5D_calib`. The crop is shape `(6, h, w)`, dtype
**float64**, where `h, w` is the bounding box of that cell's mask *after
binning*. Everything outside the mask is zero. The planes, in the order the
`np.stack([c_gc, c_sc, ci1, ci2, ci3, isum], axis=0)` that builds `cell_stack`
puts them:

| Plane | Variable | What it is |
|---|---|---|
| 0 | `c_gc` | phasor **G** per pixel, after the phase/modulation calibration |
| 1 | `c_sc` | phasor **S** per pixel, same calibration |
| 2 | `ci1` | calibrated intensity ratio ch1 / (ch1+ch2+ch3+ch4) |
| 3 | `ci2` | ratio ch2 / sum |
| 4 | `ci3` | ratio ch3 / sum |
| 5 | `isum` | calibrated total intensity, binned and masked |

Only three ratios are written; the fourth is redundant given the other three, and
the comment beside `int_ratio_3` — `# (if you ever need the 4th: int_ratio_4 =
C4/Csum)` — says as much. Six planes, always.

Plane 5 is the one the training and inference loaders normalise
(`normalize_intensity` divides `image[-1]` by its max). Planes 0–4 reach the
network in physical units.

The run-config CSV is written once **per sample folder** (the `run_config.to_csv`
at the top of the per-sample loop), before that sample's crops. This script writes in place and has no `--out`, so the record
lives beside its output; without it a folder of crops does not say which
calibration produced it.

---

## 3. Every flag of `Data_prep.py`

Seventeen flags, listed in the order `--help` prints them. Each *optional*
flag's default is the literal the script shipped with, so a run that names its
samples and passes nothing else computes what the committed file computed.

### Inputs and outputs

| Flag | Default | What it does |
|---|---|---|
| `--data-root` | **required** | Root holding one sample folder per acquisition. Must be writable — crops are written back into each sample folder. |
| `--samples` | **no default** – give this or `--all-samples` | Comma-separated sample folder names, processed in the order given. The shipped file carried a hand-edited list of folder names with most entries commented out, so naming them here is the direct equivalent of uncommenting two of them. A name that is not a folder, or a folder without a `raw/`, is a hard error. |
| `--all-samples` | off | Process every immediate subfolder of `--data-root` that has a `raw/`, in sorted order, instead of naming them. Passing neither this nor `--samples` exits 1; passing both exits 1. It is a separate flag rather than the empty default because processing a sample **clears** its output folder first, so "every folder under the root" has to be asked for. The resolved list prints one path per line, followed by a line saying nothing has been deleted yet, before the first deletion. |
| `--seg-folder` | `seg_5D_calib` | Output subfolder under each sample. `Test_LUMINA.py` and `Finetune_LUMINA.py` prefer this name; **`Train_LUMINA.py` reads `seg_5D` only**, so pass `--seg-folder seg_5D` to prepare training data rather than renaming afterwards. |

### Instrument calibration — check these against your own microscope

| Flag | Default | What it does |
|---|---|---|
| `--calibration-factors` | `19.0009,14.3886,13.2671,11.8055` | Three or four per-detector gain factors. With three, the fourth is taken as 1.0; any other count is fatal (`unpack_calibration_factors`). They multiply the four intensity images into `C1`…`C4` before the ratios and the total are formed, so they set the spectral coordinates of every crop. |
| `--phi-calib` | `-0.0125` | Phasor **phase** correction, in radians, added to every pixel's φ. |
| `--m-calib` | `1.0292` | Phasor **modulation** correction, multiplying every pixel's radius. `--phi-calib 0 --m-calib 1.0` is the identity, i.e. uncalibrated phasor coordinates. |
| `--rep-rate-mhz` | `78.1` | Laser repetition rate in MHz; reaches `calcu_phasor_info` as `freq = rep_rate_mhz/1000`, to pair with `t` in nanoseconds. This is the phasor frequency and it scales every *G* and *S*. It is a property of the laser — read the acquisition metadata rather than assuming. |
| `--tau-resolution` | `0.09696969696999999` | Nanoseconds per time bin of the decay stack. Also from the acquisition metadata; every *G* and *S* moves with it. |

The first three, together with `--rep-rate-mhz`, describe the **instrument, not
the sample**. The defaults are the values this script shipped with — the ones the
prepared crops in this project were made with. They are meaningless on another
microscope, and carrying them over silently moves every *G* and *S* rather than
raising anything.

### Phasor window, binning, gate

| Flag | Default | What it does |
|---|---|---|
| `--peak-offset` | `4` | Time bins after the per-pixel decay maximum (`pidx = np.argmax(roi)`) where the phasor window starts. Ignored under `--no-tail-only`. |
| `--end-offset` | `18` | Time bins dropped from the noisy end of the decay. |
| `--bin-size` | `1` | Square spatial binning by summation, applied to the decay, the intensities and the masks alike. 1 disables it. Larger bins buy photons per pixel, lose resolution, and shrink the crops. |
| `--smooth` | `none` | `median` runs `medfilt(kernel_size=3)` along the decay inside `calcu_phasor_info`. `wavelet` is exposed because the `smooth_option == 'wavelet'` branch beside it exists, but it **does not work** — see `troubleshooting.md`. |
| `--intensity-threshold` | `0` = auto | Minimum binned total intensity for a pixel to get a phasor; quieter pixels keep *G* = *S* = 0 while their ratio and intensity planes keep their values. Auto resolves in `main` as `args.intensity_threshold or 100 * args.bin_size * args.bin_size`, so the gate scales with the bin: 100 at bin 1, 400 at bin 2, 1600 at bin 4. A negative value keeps every pixel of the mask. |

### Switches

| Flag | Default | What it does |
|---|---|---|
| `--no-tail-only` | off (tail-only **on**) | Start the phasor window at bin 0 instead of at the decay maximum plus `--peak-offset`. The two are not comparable: a full-decay phasor carries the instrument response. |
| `--calculate-lifetime` | off | Fits a mono-exponential per pixel with `curve_fit` inside `calcu_phasor_info`. The caller discards τ and χ² — it unpacks `g, s, _, _, _, _` — so this **changes no output** and costs a great deal of time. Exposed only because the switch exists. |
| `--keep-existing` | off (clear first) | Do not delete what is already in the output folder. Read the flag's own help before turning it off: the clear happens once per *field of view*, inside the loop — it is the `for file in os.listdir(out_dir)` block that runs right after `out_dir` is built — while the written names carry no FOV part, so a multi-FOV sample keeps only the last FOV either way — see `troubleshooting.md`. |

**How the phasor is computed** (`calcu_phasor_info`), for a user
who needs to know whether the numbers are comparable to the plugin's: per pixel,
take the decay from `argmax + --peak-offset` to `len - --end-offset`, divide by
its own maximum, set `t = 0` at the start of that window, then
`g = Σ d·cos(2πft) / Σ d` and `s = Σ d·sin(2πft) / Σ d`. Tail-only,
no deconvolution — the same family as the plugin's tail method, but computed per
pixel rather than per cell, and then calibrated in polar form. Cell-averaged *G*
from `FLIM-S.xlsx` and the mean of plane 0 here are not the same quantity.

---

## 4. What the training and inference loaders expect

Both scripts carry their own copy of `FluorescenceDataset`; the copies differ,
and the difference is how the folder is chosen.

| Script | Path it builds | Folder chosen by |
|---|---|---|
| `Train_LUMINA.py`, single-anchor rows | absolute `output_file` built by `load_training_data`: `<--data-root>/<folder>/<--seg-folder>/*_5D.tif` | `--seg-folder`, default `seg_5D` |
| `Train_LUMINA.py`, dual-anchor rows | `<--dual-root>/<Directory>/<--seg-folder>/cell<Cell_Label>_5D.tif`, built in the `is_test` branch of `FluorescenceDataset.__getitem__` | the same `--seg-folder` |
| `Test_LUMINA.py` | `<--data-root>/<Directory>/<folder>/cell<Cell_Label>_5D.tif`, built in its own `FluorescenceDataset.__getitem__`, which walks `seg_order` | `--seg-folder`, default `auto` = first of `seg_5D_calib`, `seg_5D` that exists |

So `Data_prep.py` writes `seg_5D_calib` by default, `Test_LUMINA.py` prefers it,
and `Train_LUMINA.py` looks only where `--seg-folder` says — whose default is
`seg_5D`. Training on freshly prepared data means matching the two flags, in
either direction. There is deliberately no `auto` in `Train_LUMINA.py`: a
checkpoint trained on a mixture of calibrated and uncalibrated crops would not be
reproducible from the recorded flags.

`Test_LUMINA.py` prints the folder it actually read, per sample, before scoring
it. Read that line rather than assuming.

Note also the asymmetry around `--data-root-2` in `Test_LUMINA.py`: the image
loader **never switches root** — the comment inside `FluorescenceDataset.__getitem__`
says exactly that — it only chooses a folder name under `--data-root`.
`--data-root-2` still steers three other lookups: the `clustered.xlsx` search and
the unlabelled crop glob, both in `load_finetuning_data`, and the output folder
`test_model` writes into. So a sample found only under the second root gets
enumerated and then fails to load. Keep one root complete rather than
splitting a dataset across two. This is long-standing behaviour, preserved
deliberately.

**Both loaders pad, they do not resize.** `pad_image` centres the crop in a
square canvas of zeros, sized by `--crop-size` (default 256 in both scripts;
`Test_LUMINA.py` keeps that literal in its module-level `CROP_SIZE`, which is
where its flag takes the default from). A cell whose bounding box
exceeds either dimension is **skipped** — `idx = (idx + 1) % len(self.df);
continue` — not resized and not reported. See `troubleshooting.md` for what that
does to `Cell_Label` at inference.

---

## 5. The label spreadsheet

`load_finetuning_data` reads one workbook per dual-anchor folder:

```
Train_LUMINA.py:  <--dual-root>/<sample>/clustered.xlsx
Test_LUMINA.py:   <--data-root>/<sample>/clustered.xlsx, then --data-root-2/<sample>/
```

and needs exactly three columns:

| Column | Used as |
|---|---|
| `Nu_FP` | nuclear class **name**, looked up in `NU_CLASS_MAP` (`'N10'`, `'N13'`, …) |
| `Mito_FP` | mitochondrial class name, looked up in `MITO_CLASS_MAP` |
| `Cell_Label` | the integer in `cell<label>_5D.tif` |

A name that is not a key of the map, or a blank cell, becomes class 0. In
`Train_LUMINA.py` a row with either label 0 is dropped by `load_finetuning_data`,
and `FluorescenceDataset.__getitem__` skips any that survive; in `Test_LUMINA.py`
it is kept, because
inference does not need labels. A dual-anchor set in which *no* cell has two
known labels is a hard error in `Train_LUMINA.py` — there would be nothing to
train stage 2 on.

In `Train_LUMINA.py` a sample folder without a `clustered.xlsx` is also fatal
(`load_finetuning_data` raises `SystemExit`), not skipped: skipping it silently would train on a smaller
population than the one named on the command line.

This is **not** the `clustered.xlsx` the plugin writes. The plugin's workbook
carries `Mask label`, `FOV`, `Localization`, `cluster_local`, `cluster_tag`; it
has no `Nu_FP`, `Mito_FP` or `Cell_Label` column, and nothing in this
repository produces those three. They come with the dual-anchor dataset or are
added by hand. If a user asks where their `Nu_FP` column should come from, say
that — do not point them at Seeded K-Means.

When `clustered.xlsx` is missing, `Test_LUMINA.py` falls back to globbing
`cell*_5D.tif` in the crop folder (the `else` branch of its `load_finetuning_data`)
and sets both labels to 0,
which is the normal path for classifying an unlabelled dish. That fallback is
asymmetric on purpose and always has been: it tries the *preferred* folder under
`--data-root`, then the *last-choice* folder under `--data-root-2`. Under
`--seg-folder auto` that is `(--data-root, seg_5D_calib)` then
`(--data-root-2, seg_5D)` — so a dish whose crops sit in `seg_5D` under
`--data-root` alone is not found here. Pass `--seg-folder seg_5D` for that
layout.

---

## 6. Class maps

They are module-level literals, not derived from anything, and identical in all
four files:

| File | Where |
|---|---|
| `Train_LUMINA.py` | module level, under the class-map comment |
| `Test_LUMINA.py` | module level, above `CROP_SIZE` |
| `Visualize_heatmap.py` | module level, above `CONFIDENT_XLSX` |
| `Finetune_LUMINA.py` | module level, just after the `Test_LUMINA` import |

`grep -n 'NU_CLASS_MAP = ' *.py` in `LUMINA_classification/` finds all four.

```python
NU_CLASS_MAP   = {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
MITO_CLASS_MAP = {'M10': 1, 'M13': 2, 'M4': 3, 'M14': 4, 'M16': 5, 'M8': 6, 'M1': 7}
```

| Name | `N10` | `N13` | `N4` | `N14` | `N16` | `N8` | `N1` |
|---|---|---|---|---|---|---|---|
| Index | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

Index **0** means *no anchor / unknown*: it is a real output unit of both heads
(`--num-classes 8` covers 0–7), it is never a training target, and at inference
it reverse-maps to the string `'Unknown'` — the
`next((k for k, v in nu_class_map.items() if v == pred_nu), 'Unknown')` lookups in
`Test_LUMINA.test_model`.

These indices used to be *derived* — `{key: idx + 1 for idx, key in
enumerate(nu_files.keys())}` over a dict of folder names whose entries were half
commented out, so uncommenting one line renumbered every class after it. That
derivation is gone; the maps are stated. Two consequences worth telling a user:

- **There is no flag for the order, by design.** `Train_LUMINA.py`'s
  `--single-anchor-manifest` says which *folder* holds which class, and its row
  order is explicitly ignored (`load_single_anchor_manifest` returns the dicts in
  `NU_CLASS_MAP` order, so the file cannot move the `--seed` split either).
  A checkpoint records no mapping, so a flag that permuted the classes would be a
  silent way to mislabel every cell.
- **Four copies still have to agree.** Nothing checks them against each other.
  `Train_LUMINA.py`, `Test_LUMINA.py` and `Finetune_LUMINA.py` each print their
  maps at startup, and `Train_LUMINA.py` also writes them into
  `train_run_config.csv` as `resolved_nu_class_map` / `resolved_mito_class_map` —
  which is the durable record to compare a later inference run against.

`Visualize_heatmap.py` uses the same key order as the heatmap's row and column
order, and `--nu-classes` / `--mito-classes` override it. Those two flags reorder
and **filter** the figure only; they do not touch any class index. Reordering one
and not the other makes the grey diagonal meaningless with no error.
