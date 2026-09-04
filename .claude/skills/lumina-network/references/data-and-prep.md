# What must be on disk, and the script that puts it there

Every path below is traced from the source. Where a script builds a path with
`os.path.join`, the exact string is quoted.

---

## 1. What `Data_prep.py` reads

One *sample folder* per acquisition, named in the `cell_types` list
(`Data_prep.py` lines 39–71) and sitting under `data_dir` (line 75, shipped as
`G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual`, built from `cell` and `instrument` on
lines 72–74):

```
<data_dir>/<cell_type>/
├── raw/<fov>.<ext>                 ← only the STEM is used, to name everything else
├── flim_stack/<fov>-sum.tif        ← decay stack, T×H×W, time first, channel-summed
├── intensity/<fov>-1.tif           ← the four spectral intensity images,
├── intensity/<fov>-2.tif             read with cv2.imread(path, -1) = IMREAD_UNCHANGED
├── intensity/<fov>-3.tif
├── intensity/<fov>-4.tif
├── intensity/<fov>-sum_seg.npy     ← np.load(..., allow_pickle=True).item()['masks']
└── seg_5D_calib/                   ← WRITTEN by this script
```

`raw/` is enumerated with `os.listdir` and every entry is treated as a field of
view: `fov = os.path.splitext(fname)[0]`. Anything else in that folder — a
stray `.txt`, a subdirectory — becomes a FOV name whose other files do not
exist, and the script dies on the first missing file.

The mask file is the Cellpose `_seg.npy` format: a pickled dict with a `masks`
key holding an integer label image, labels `1..N`. Cells are iterated as
`range(1, masks.max() + 1)` and empty labels are skipped, so gaps are harmless.

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
<data_dir>/<cell_type>/seg_5D_calib/cell<label>_5D.tif
```

Shape `(6, h, w)`, dtype **float64**, where `h, w` is the bounding box of that
cell's mask *after binning*. Everything outside the mask is zero. The planes,
in the order `np.stack` puts them (line 228):

| Plane | Variable | What it is |
|---|---|---|
| 0 | `c_gc` | phasor **G** per pixel, after the phase/modulation calibration |
| 1 | `c_sc` | phasor **S** per pixel, same calibration |
| 2 | `ci1` | calibrated intensity ratio ch1 / (ch1+ch2+ch3+ch4) |
| 3 | `ci2` | ratio ch2 / sum |
| 4 | `ci3` | ratio ch3 / sum |
| 5 | `isum` | calibrated total intensity, binned and masked |

The comment on line 227 lists eight names (`[g, s, g_cal, s_cal, int1, int2,
int3, isum]`) for the six planes actually stacked on line 228. The comment is
stale; the fourth ratio is never written (line 171 says as much).

Plane 5 is the one the training and inference loaders normalise
(`normalize_intensity` divides `image[-1]` by its max). Planes 0–4 reach the
network in physical units.

---

## 3. Every constant in `Data_prep.py`

There are no arguments. Edit the block at the top of the file.

| Line | Constant | Shipped value | What it does |
|---|---|---|---|
| 16 | `calibration_factors` | `[19.0009, 14.3886, 13.2671, 11.8055]` | Per-channel scale applied to the four intensity images before the ratios, correcting the detector/filter response so the ratios are comparable between channels. Give 4 values, or 3 and the fourth becomes 1.0 (lines 79–85). These are instrument-specific — do not carry them to another microscope. |
| 18 | `phi_calib` | `-0.0125` | Phasor **phase** correction, in radians, added to every pixel's φ. |
| 19 | `m_calib` | `1.0292` | Phasor **modulation** correction, multiplying every pixel's radius. Together these two stand in for a reference-dye calibration; there is no IRF deconvolution anywhere in this file. Two other pairs are commented out on lines 20–23, including the identity `(0, 1.0)`. |
| 25 | `bin_size` | `1` | Square spatial binning by summation, `bin_size × bin_size` pixels into one. 1 disables it. Larger bins buy photons per pixel and lose resolution. |
| 26 | `smooth_option` | `None` | `'median'` runs `medfilt(kernel_size=3)` along the decay. `'wavelet'` is **broken** — see `troubleshooting.md`. |
| 27 | `calculate_lifetime` | `False` | Fits a mono-exponential per pixel with `curve_fit`. The fitted τ and χ² are returned and then discarded by the caller (line 209 unpacks them into `_`), so turning this on costs time and changes no output. |
| 28 | `intensity_threshold` | `100 * bin_size**2` | Minimum photons in a binned pixel before its phasor is computed. Below it, G and S stay 0 for that pixel while its ratio and intensity planes keep their values. |
| 31 | `tail_only` | `True` | Start the phasor window after the decay peak instead of at bin 0, skipping the IRF-convolved rising edge. |
| 32 | `PEAK_OFFSET` | `4` | Bins after the per-pixel peak (`np.argmax`) where the window starts. |
| 33 | `END_OFFSET` | `18` | Bins dropped from the end of the decay — the noisy late tail. |
| 36 | `tau_resolution` | `0.09696969696999999` | Nanoseconds per time bin. Check the acquisition metadata; a wrong value scales every lifetime. |
| 39–71 | `cell_types` | 2 active, 30 commented out | One entry per sample folder to process. |
| 72–75 | `cell`, `instrument`, `data_dir` | `Hek293T`, `BJMU-Dual`, `G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual` | Hardcoded lab path on a removable drive. |

**The laser repetition rate is not in that block.** It is `freq = 78.1/1000`
inside `calcu_phasor_info`, **line 118** — 78.1 MHz written in GHz to pair with
`t` in nanoseconds. A different laser needs that line changed, and getting it
wrong moves every G and S. Lines 34–35 (`peak2_begin`, `peak2_end`) are
declared and never used.

**How the phasor is computed** (lines 90–122), for a user who needs to know
whether the numbers are comparable to the plugin's: per pixel, take the decay
from `argmax + PEAK_OFFSET` to `len - END_OFFSET`, divide by its own maximum,
set `t = 0` at the start of that window, then
`g = Σ d·cos(2πft) / Σ d` and `s = Σ d·sin(2πft) / Σ d`. Tail-only, no
deconvolution — the same family as the plugin's tail method, but computed per
pixel rather than per cell, and then calibrated in polar form. Cell-averaged G
from `FLIM-S.xlsx` and the mean of plane 0 here are not the same quantity.

---

## 4. What the training and inference loaders expect

Both scripts carry their own copy of `FluorescenceDataset`; the copies differ,
and the difference is the folder name.

| Script | Path it builds | Fallback |
|---|---|---|
| `Train_LUMINA.py`, pre-training rows | absolute `output_file` from `load_training_data`: `<base_folder>/<folder>/seg_5D/*_5D.tif` (lines 195 and 208) | none |
| `Train_LUMINA.py`, dual-anchor rows | `<test_base_folder>/<Directory>/seg_5D/cell<Cell_Label>_5D.tif` (line 61) | none |
| `Test_LUMINA.py` | `<base_folder>/<Directory>/seg_5D_calib/cell<Cell_Label>_5D.tif` (line 56) | `<base_folder>/<Directory>/seg_5D/…` (line 59) |

So `Data_prep.py` writes `seg_5D_calib`, `Test_LUMINA.py` prefers it, and
`Train_LUMINA.py` only ever looks in `seg_5D`. Training on freshly prepared
data means renaming the folder or editing lines 61 and 195.

Note also that `Test_LUMINA.py`'s dataset takes a `base_dir2` and never uses it:
the fallback on line 59 re-joins `self.base_dir`, and the `base_dir2` branch is
commented out on line 58. `base_folder2` still steers three other lookups — the
`clustered.xlsx` search (line 188), the unlabelled `seg_5D` glob (line 208) and
the output folder (line 276) — so a sample found only under `base_folder2` gets
enumerated and then fails to load, because the loader looks for its images under
`base_folder`. Keep one root, or keep both roots complete.

**Both loaders pad, they do not resize.** `pad_image` centres the crop in a
256×256 canvas of zeros (`max_height`/`max_width`, `Train_LUMINA.py` lines
490–491, `Test_LUMINA.py` lines 229–230). A cell whose bounding box exceeds
either dimension is **skipped** — `idx = (idx + 1) % len(self.df); continue` —
not resized and not reported. `resize_image` is defined in both files and
called by neither.

---

## 5. The label spreadsheet

`load_finetuning_data` reads one workbook per dual-anchor folder:

```
<test_base_folder>/<test_dir>/clustered.xlsx
```

and needs exactly three columns:

| Column | Used as |
|---|---|
| `Nu_FP` | nuclear class **name**, looked up in `nu_class_map` (`'N10'`, `'N13'`, …) |
| `Mito_FP` | mitochondrial class name, looked up in `mito_class_map` |
| `Cell_Label` | the integer in `cell<label>_5D.tif` |

A name that is not a key of the map, or a blank cell, becomes class 0. In
`Train_LUMINA.py` (line 234) a row with either label 0 is dropped; in
`Test_LUMINA.py` it is kept, because inference does not need labels.

This is **not** the `clustered.xlsx` the plugin writes. The plugin's workbook
carries `Mask label`, `FOV`, `Localization`, `cluster_local`, `cluster_tag`; it
has no `Nu_FP`, `Mito_FP` or `Cell_Label` column, and nothing in this
repository produces those three. They come with the dual-anchor dataset or are
added by hand. If a user asks where their `Nu_FP` column should come from, say
that — do not point them at Seeded K-Means.

When `clustered.xlsx` is missing, `Test_LUMINA.py` falls back to globbing
`seg_5D_calib/cell*_5D.tif` (lines 204–218) and sets both labels to 0, which is
the normal path for classifying an unlabelled dish.

---

## 6. Class maps

```python
nu_class_map   = {key: idx + 1 for idx, key in enumerate(nu_files.keys())}
mito_class_map = {key: idx + 1 for idx, key in enumerate(mito_files.keys())}
```

Built from the **order of the dict literal**, not from the number in the name.
With the literal shipped in both scripts (`Train_LUMINA.py` 564–591,
`Test_LUMINA.py` 501–528):

| Name | `N10` | `N13` | `N4` | `N14` | `N16` | `N8` | `N1` |
|---|---|---|---|---|---|---|---|
| Index | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

and the same for `M10 … M1`. Index **0** means *no anchor / unknown*: it is a
real output unit of both heads (`num_classes = 8` covers 0–7), it is never a
training target, and at inference it reverse-maps to the string `'Unknown'`
(`Test_LUMINA.py` lines 303–304).

The checkpoint records none of this. Two dict literals that disagree between
training and inference produce confident, wrong, silently permuted labels.
`Visualize_heatmap.py` hardcodes the same order again on lines 214–215, where
it sets the row/column order of the figure.
