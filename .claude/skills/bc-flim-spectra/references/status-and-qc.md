# "Where am I, and does this look right?"

Most people using this plugin are running someone else's protocol on their own
cells. The questions that come up mid-experiment are not about parameters —
they are *have I done this bit yet*, *is this number normal*, and *what do I do
next*. Answer them from the folder, not from memory.

## Start here, every time

```bash
python Napari_plugin/scripts/workflow_status.py <sample folder>
```

It opens nothing and changes nothing. It reports, per step, whether that step
has run and how much it produced; then a list of quality checks with the actual
numbers; then what is outstanding. `--json` gives the same thing machine-readably.
Exit status is 1 if a check failed, so it also works as a gate in a script.

Run it before answering any "how is my run going" question, and quote the
numbers back. A user who is told *"418 nuclei were segmented, 373 got a barcode,
45 were declined as outliers, and the biosensor half has not started"* knows
where they stand. A user who is told *"looks fine"* does not.

## What the checks mean, and when a failure is real

| Check | Fails when | What to say |
|---|---|---|
| every FOV has decay stacks | `flim_stack/` is missing or short | Calculate FLIM-S cannot run; re-run PTU Reader for the missing FOVs. |
| no empty masks | a `*_seg_*.npy` has zero objects | Segmentation found nothing — usually the wrong model or a diameter far off. |
| S under 0.5 | any S exceeds the semicircle apex | Suspect the pulse frequency or tau resolution; those scale the whole phasor. |
| near the universal semicircle | median radius from (0.5, 0) is well over 0.5 | Slightly outside is normal here: the fit uses a truncated tail with no IRF deconvolution. Far outside means the timing parameters are wrong. |
| lifetimes physical | outside 0.1-10 ns | Same cause: check pulse frequency and tau resolution first. |
| declined fraction | more than ~30 % of *clustered* cells declined | The outlier detector runs per class at its contamination rate, so ~10 % is expected. Much more means dim or fragmented cells — check `Total intensity` and the mask threshold. Cells of a localisation you never clustered are counted separately as `not_clustered`, not as declined. |
| no class collapsed | a class has fewer than 3 cells | Either that barcode is absent from the dish, or two seeds landed on the same cluster. |
| barcode classes land on biosensor cells | coverage < 50 % or purity < 85 % | The registration is wrong. See below — this is the check people skip. |

## The registration check, and why coverage alone is not it

The barcode image is 2048×2048 FLIM; the biosensor stack is 1024×1024 confocal.
The plugin aligns them with a 90° clockwise rotation and a resize. The status
script tries all four rotations and reports two numbers for each:

- **coverage** — the fraction of biosensor cells that receive any class;
- **purity** — within each cell, how much of the covered area agrees on one class.

**Coverage on its own does not prove anything.** The plugin assigns a class when
coverage exceeds *Align Threshold* (default 10 %), and measured on the reference
dataset **two thirds of cells still get a class at the wrong rotation** — the
labels are then at chance. Purity separates them cleanly: about 99 % correct
versus about 70 % wrong. So quote purity, and quote which rotation won.

If the wrong rotation wins, or the two are close, tell the user to click
**Load / Confirm Barcode** and look at the overlay before segmenting. That
button exists precisely so the alignment can be seen before anything depends
on it.

## Reading the result files directly

```python
import pandas as pd
df = pd.read_excel('FLIM-S.xlsx')       # one row per cell, the 5-D features
dc = pd.read_excel('clustered.xlsx')    # the same rows plus cluster_local / cluster_tag
```

`cluster_tag` is `N3`, `P5`, … or `Outlier`; **NaN means that row was never
clustered**, which is normal for the localisation you did not run. In
`signal_analysis.xlsx` each sheet is one channel or ratio, one column per
barcode class across frames.

## Judging a run without ground truth

There usually is no ground truth, so lean on internal consistency:

- **Class means should land on the seeds.** If a seed file was loaded, each
  class's mean G and S should sit within a few thousandths of its seed.
- **Lifetime should fall monotonically as G rises.** That is the geometry of
  the phasor; a class that breaks it is suspect.
- **`Lifetime` and `FastFLIM` should track but not coincide.** They are a
  fitted mono-exponential and a photon-weighted mean.
- **The four intensity ratios sum to 1** for every cell.
- **Whitening reporting "0 cells changed class"** means the first pass was
  already stable — good news, not a no-op.

## What to say when something is wrong

Name the file, the number and the control that would settle it. "The workbook
has 3 FOVs but only 1 is on disk, so rows from an earlier run were merged in —
tick Fresh FLIM-S.xlsx and re-run" is actionable. "Something looks off" is not.
