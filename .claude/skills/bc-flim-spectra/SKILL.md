---
name: bc-flim-spectra
description: Guide a user through SLIC (Spectral-Lifetime Indexing of Cells), the napari plugin for barcoded FLIM imaging distributed as the bc-flim-spectra package - installing it, running the seven-step workflow from .ptu files to per-barcode biosensor readouts, choosing parameters, and diagnosing the failures that actually happen. Use whenever someone is working with this plugin, a barcoded FLIM sample folder (flim_stack/, intensity/, FLIM-S.xlsx, clustered.xlsx), PTU decoding, phasor G/S per cell, seeded K-Means barcode classification, or Cellpose N/P segmentation in this context.
license: BSD-3-Clause
---

# SLIC — Spectral-Lifetime Indexing of Cells

A napari plugin that turns raw PicoQuant `.ptu` FLIM acquisitions into
per-barcode biosensor measurements. It is installed as the `bc-flim-spectra`
package and listed under that name in napari; *NaCha* is its final widget. Seven widgets, run in order, each writing
files the next one picks up automatically.

Use this skill to answer "how do I do step N", "what should this number be",
and "why did it do that". For editing the source, see `AGENTS.md` at the repo
root instead.

## The idea in one paragraph

Cells carry **barcodes**: combinations of fluorophores that differ in
fluorescence lifetime and emission spectrum. One FLIM acquisition gives, for
every pixel, a photon-arrival histogram in four spectral channels. Averaged
over a segmented cell that becomes five numbers — phasor coordinates *G* and
*S* (lifetime, as a point on the universal circle) plus three spectral
intensity ratios. Cells of the same barcode land in the same place in that 5-D
space, so clustering assigns each cell its barcode. A separate biosensor
channel is then read per class, giving one readout per barcode from a single
mixed dish.

## The seven steps

Each widget has a blue **Next ▶** button that opens the following one, so the
whole workflow chains without touching the Plugins menu.

| # | Widget | Turns | Into |
|---|---|---|---|
| 1 | PTU Reader | `raw/*.ptu` | `intensity/*_ch1..4.tif`, `intensity/*_sum.tif`, `flim_stack/*.tif`, `*_fastflim_tau.tif` |
| 2 | Barcode Seg (Cellpose) | `intensity/*_sum.tif` | `intensity/<stem>_seg_n.npy`, `_seg_p.npy` |
| 3 | Calculate FLIM-S | decay stacks + masks | `FLIM-S.xlsx` — one row per cell, the 5-D fingerprint |
| 4 | Seeded K-Means | `FLIM-S.xlsx` | `clustered.xlsx` — a barcode class per cell |
| 5 | Biosensor Seg (Cellpose) | biosensor stack | `<fov>_seg_image_seg.npy` |
| 6 | B&P Tracker *(time-lapse only)* | biosensor stack | tracked masks |
| 7 | NaCha | everything above | per-class signal curves |

Steps 6 and 7 are only needed for time-lapse biosensor work; a fixed-sample
barcode experiment ends at step 4 or 5.

**The sample folder is the unit of work.** Every widget takes one folder and
finds its own inputs inside it. A mask always sits **beside the image it was
drawn on** — `intensity/<stem>_sum_seg_n.npy`, never the sample root.

## Reference files

Read the one you need; do not read them all.

| File | Read it when |
|---|---|
| `references/status-and-qc.md` | **Start here for any "where am I / how is it going / does this look right" question.** Runs one read-only script that reports which steps are done, the quality checks with real numbers, and what is outstanding. |
| `references/workflow.md` | Walking someone through a step, or choosing parameters. Covers all seven widgets, every control, and what each number does. |
| `references/troubleshooting.md` | Something failed, looks wrong, or produced surprising numbers. |
| `references/install.md` | Installing, or diagnosing an install: the three-environment design, Cellpose model weights, Qt backends, China-network workarounds. |

## Answering well here

**Say which file to look at.** Every step's output is a file on disk. "Check
that `intensity/` now has a `_seg_n.npy` next to the sum tif" is a better
answer than a description of what should have happened.

**Parameters have physical meaning — give it.** *Pulse Frequency* is the
laser repetition rate and getting it wrong makes every lifetime wrong;
*Peak Offset* skips the instrument response at the rising edge; *Mask
Intensity Threshold* drops cells with too few photons for a stable fit. A
user who knows what a knob does can set it for their own microscope. Defaults
in the widget match a Leica FALCON at 78.1 MHz.

**Two lifetime quantities, never interchangeable.** `Lifetime` is a fitted
mono-exponential τ. `FastFLIM` is the photon-weighted mean arrival time. Both
are in `FLIM-S.xlsx`; say which one you mean.

**Do not invent numbers.** If you do not know a valid range, say so and point
at the tooltip in the widget, which carries the authoritative value.

**Barcode colours come from the project palette**, never `tab20`.

## Someone asks how their run is going

Do not answer from the conversation. Look at the folder:

```bash
python Napari_plugin/scripts/workflow_status.py <sample folder>
```

Read-only. It says which of the seven steps have run, gives the quality checks
with their numbers, and lists what is outstanding — including the registration
check, which is the one people skip. `references/status-and-qc.md` explains how
to read it and when a failure is real.

## Fast checks

Is the plugin installed and which copy is loaded?

```bash
python -c "import flim_s_gen._widget as w; print(w.__file__)"
```

Open napari straight onto one widget (names from `napari.yaml`: `PTU Reader`,
`Barcode Seg (Cellpose)`, `Calculate FLIM-S`, `Seeded K-Means`,
`Biosensor Seg (Cellpose)`, `B&P Tracker`, `NaCha`):

```bash
python -m napari -w bc-flim-spectra "Calculate FLIM-S"
```

What is in a results workbook?

```python
import pandas as pd
df = pd.read_excel('FLIM-S.xlsx')
print(df.groupby(['FOV', df['Localization'].fillna('')]).size())
print(df[['Int 570-590','Int 590-610','Int 610-638','Int 638-720']].isna().sum())
```

Rows for a FOV you did not just process, or non-NaN columns for channels you
did not select, mean the workbook is carrying an earlier run — see
`references/troubleshooting.md`.
