# SLIC Napari Plugin — Walkthrough Storyboard

> **For the video editor**: each `##` section below is one slide / scene.
> Slides whose title starts with **[TITLE]** or **[TRANSITION]** are simple
> text cards inserted over the screen recording. Everything else is real
> napari footage — use the voiceover / caption copy as the subtitle / narration.
> Keep each card on screen ~3-6 seconds unless stated otherwise.

---

## Slide 1 — [TITLE] Opening card

**Headline (centered, large):**

> **SLIC — a Napari plugin for FLIM-based barcode cell classification**

**Subtitle (smaller, centered):**

- Spectral-Lifetime Integrated Classification
- Plugin: `bc-flim-spectra` (module `flim_s_gen`)
- Audience: bench scientists running multi-biosensor barcode imaging

**Duration:** ~4 s. Fade in.

---

## Slide 2 — [TITLE] What this plugin does (30-second overview)

**Narration / caption (one screen, two columns):**

| **Input** (what you bring) | **Output** (what you get) |
|---|---|
| Leica FLIM `.ptu` files (barcode) | `clustered.xlsx` — per-cell barcode class labels |
| Confocal time-lapse TIF stacks (B / G / Y biosensor channels) | `signal_analysis.xlsx` — per-cell normalised signal curves |
| | Labels layers (N / P nucleus / cytoplasm + barcode cls + tracking) |
| | Per-class mean ± SE signal plots |

**One-line pipeline summary (bottom of slide):**

> *PTU → intensity + FLIM stacks → segment barcode (N/P) → compute FLIM-S → K-Means cluster → segment biosensor → (optional) track → revise + calculate per-class signals.*

**Duration:** ~8 s.

---

## Slide 3 — [TITLE] Data & workspace you need to prepare

**Body (bullets):**

1. **Barcode data (FLIM)**
   - One `.ptu` file per field of view (FOV)
   - Keep them inside a `raw/` subfolder of the sample

2. **Biosensor data (time-lapse confocal)**
   - One multi-frame TIF stack per channel, per FOV
   - Channels: **B** (blue), **G** (green), **Y** (NIR/yellow)
   - Same frame count and same XY resolution across the three channels
   - Filenames like `FOV-1-b.tif`, `FOV-1-g.tif`, `FOV-1-y.tif`
   - Place them at the sample-folder root (NOT inside `raw/`)

3. **Folder layout (required)**

   ```
   <sample>/
   ├── raw/
   │   └── <FOV>.ptu          # one .ptu per FOV
   ├── FOV-1-b.tif            # blue channel time-lapse
   ├── FOV-1-g.tif            # green channel time-lapse
   ├── FOV-1-y.tif            # NIR/yellow channel time-lapse
   └── FOV-2-b.tif            # (repeat for other FOVs)
   ```

   A clean, empty sample folder is ideal — the plugin writes intermediate
   files here, so keep it separate from the raw source data.

**Visual:** file-explorer screenshot of a correctly laid out sample folder
with the `raw/` subfolder expanded.

**Duration:** ~10 s.

---

## Slide 4 — [TITLE] Open napari and launch the plugin

**Voiceover:**

> *Start napari from the `BC-FLIM` conda environment. The SLIC plugin shows
> up under the Plugins menu as a list of seven widgets in workflow order. We'll
> walk them top-to-bottom.*

**Screen record:** start napari → Plugins menu → show the 7 widget entries:

1. PTU Reader
2. Barcode Seg (Cellpose)
3. Calculate FLIM-S
4. Seeded K-Means
5. Biosensor Seg (Cellpose)
6. B&P Tracker
7. NaCha

Each widget has a blue **Next ▶** button at the bottom that advances to
the next one — users can chain them without using the menu.

**Duration:** ~6 s.

---

## Slide 5 — [TRANSITION] Step 1/7 · PTU Reader

**Card text (one-liner):**

> *Decode the FLIM `.ptu` file into per-channel intensity + lifetime maps.*

**Voiceover / caption sequence during screen record:**

- *Pick the `raw/` folder and the sample folder as the output root.*
- *Set the lifetime range (tau min / max), clip percentile, frame index.*
- *Click the green **Process and Save** button.*
- *Progress bar + status line show each step: loading PTU →
  writing per-channel TIFs → FastFLIM tau map.*
- *Napari fills with `_Tau`, `_Intensity`, and per-channel stacks.*
- *If the outputs already exist, the plugin asks you whether to overwrite.*

**Output of this step:**

- `intensity/*_ch1.tif ... *_ch4.tif` (per-channel intensity)
- `intensity/*_sum.tif` (total intensity — this is what gets segmented next)
- `flim_stack/*.tif` (per-channel decay stacks)
- `*_fastflim_tau.tif` (lifetime map)

**End of step:** click **Next ▶** → layer stacks are cleared to save memory;
Barcode Seg opens.

**Duration:** ~10 s on screen + ~20-30 s actual waiting for decoding.

---

## Slide 6 — [TRANSITION] Step 2/7 · Barcode Seg (Cellpose)

**Card text:**

> *Segment each barcode cell into N (nucleus) and P (cytoplasm) masks using
> a Cellpose model fine-tuned on our barcode library.*

**Voiceover / caption sequence:**

- *The widget picks the sample folder's `intensity/*_sum.tif` automatically.*
- *Two Cellpose models: N (nucleus, diameter ≈ 55) and P (cytoplasm,
  diameter ≈ 92). Latest fine-tuned default: `NinNC-260328-1` and
  `CinNC-260328-1`.*
- *Click **Auto Segment N & P**. Progress bar shows model build + inference
  time; first run includes a ~10 s CUDA warmup.*
- *Inspect the `mask_n_fill` and `mask_p_fill` Labels layers. Edit them in
  napari: right-click to draw, Enter to commit, Ctrl+click to delete,
  Z/X to toggle visibility, S to cycle contrast.*
- *Optional: **Fine-tune** the model in-place for this sample — the new
  model is stored alongside the data and added to the dropdown.*

**Output:**

- `intensity/<stem>_seg_n.npy` + `_seg_p.npy` (N / P masks)

**Duration:** ~15 s.

---

## Slide 7 — [TRANSITION] Step 3/7 · Calculate FLIM-S

**Card text:**

> *Turn the four decay stacks + N/P masks into per-cell phasor coordinates
> (G, S) and per-channel intensities — a compact 5-D fingerprint per cell.*

**Voiceover / caption sequence:**

- *The widget auto-loads the four `flim_stack/*_ch[1-4].tif` stacks into the
  Stack 1-4 selectors, and auto-fills the N / P segmentation slots from
  the `.npy` files saved in Step 2.*
- *Adjust phasor parameters only if needed (pulse frequency, peak offset,
  pixel-wise vs mask-wise). Defaults match standard Leica FALCON setups.*
- *Click the green **Process and Save to Excel**. Progress bar runs the
  phasor engine; a G-S scatter plot pops up at the end.*

**Output:**

- `<sample>/FLIM-S.xlsx` — one row per cell, columns include
  `Localization, G, S, Lifetime, Chi^2, Total intensity, Mask label, FastFLIM,
  Int 1..4, Int 1/(1-4) ... Int 4/(1-4), FOV`.

**Duration:** ~12 s.

---

## Slide 8 — [TRANSITION] Step 4/7 · Seeded K-Means

**Card text:**

> *Cluster the 5-D fingerprints into the N barcode classes expected in this
> library. Supports seeded K-Means, K-Means++, MiniBatchKMeans, Gaussian
> Mixture, and Spectral clustering.*

**Voiceover / caption sequence:**

- *Pick the sample folder (auto-filled). Set number of clusters, method,
  5-D weights (G/S/Int1/Int2/Int3). These weights are applied AFTER
  StandardScaler, so they really affect distance.*
- *Click **Read and Plot** → a scatter of G vs S (plus Int pairs in 4D/5D)
  appears. For seed-based K-Means, click to place seeds.*
- *Optional: **Load Seeds** from a previously saved Excel — K
  auto-updates to match the seed count. **Load Distribution** overlays a
  coloured background region per class, sized by the adjustable expansion
  factor.*
- *Per-class outlier detection runs after K-Means — points flagged as
  Outlier are assigned class 0. Tweak contamination and click **Re-flag
  outliers** without re-fitting K-Means.*
- *Click **Run K-Means** → **Save Results**. Progress bar goes through
  assign → sort → write Excel → draw per-class masks.*
- *Tip: for multi-localization data (AUTO over N / M / P), run K-Means
  once per localization and **Save Results** after each — `clustered.xlsx`
  accumulates across locs.*

**Output:**

- `<sample>/clustered.xlsx` — adds `cluster_local`, `cluster_tag`,
  `cluster_global` columns
- Per-class mask PNG / tif in the sample folder

**Duration:** ~15-20 s.

---

## Slide 9 — [TRANSITION] Step 5/7 · Biosensor Seg (Cellpose)

**Card text:**

> *Now switch to the biosensor side: make a single strong-signal image to
> segment biosensor cells, using the barcode class map as an auxiliary
> channel.*

**Voiceover / caption sequence:**

- **Step 1 — Generate Seg Image**: pick which channels to use (B/G/Y) and
  which frames to average (default 1-6). Click **Generate Seg Image**.
  A new `seg_image` layer appears; regenerate with different params to
  overwrite.
- **Step 2 — Load / Confirm Barcode**: set rotation (default **90° CW**
  for Leica tilescan, **0°** for single-FOV) and resize (default 1024,
  0 = match seg image shape). Click **Load / Confirm Barcode**. Toggle
  the new `barcode_cls` layer against `seg_image` to visually confirm
  registration before segmentation.
- **Step 3 — Segment**: Cellpose model `BS-BC-assist-cls-260402-forDense`
  is the default, diameter 45. Click **Segment Cells**; the dual-channel
  result is saved as `<seg_image_stem>_seg.npy` and shown as
  `mask_biosensor` — editable with the same shortcuts as Barcode Seg.
- Optional: **Fine-tune** (single-image) to tweak the model for this
  sample; the new model is saved under `<sample>/_finetune/<name>/`.

**Output:**

- `<sample>/<seg_image_stem>.tif` (seg image)
- `<sample>/<seg_image_stem>_seg.npy` (biosensor cell mask, single 2D)

**Duration:** ~20 s.

---

## Slide 10 — [TRANSITION] Step 6/7 · B&P Tracker (OPTIONAL)

**Card text (with callout):**

> ⚠️ *Tracking is **optional**. If your cells do not move much, skip this
> step and click **Next ▶** — the next widget will broadcast the single
> biosensor mask to every frame automatically.*

**Voiceover / caption sequence (only if user does track):**

- *Use **Build Tracking Stack**: pick B / G / Y channels, set smoothing
  window (default 10), click → the plugin averages the channels
  frame-by-frame, temporally smooths, writes a uint16 stack and loads it
  into napari.*
- *Adjust tracking knobs (cell distance, padding, chunk size, patch
  size). Click the green **Track** to run. Progress is tqdm-based in the
  terminal.*
- *Click **Save Tracking** to dump per-frame `.npy` masks into a folder.*

**Output (if tracked):**

- A folder of per-frame `00000.npy ... NNNNN.npy` tracked masks

**Duration:** ~8 s transition card + optional footage.

---

## Slide 11 — [TRANSITION] Step 7/7 · NaCha (Final step)

**Card text:**

> *Align barcode classes to biosensor cells, then compute and save per-class
> signal curves. This is the final step.*

**Voiceover / caption sequence:**

- *Click **Read in all** — loads mask folder (or falls back to the single
  BiosensorSeg `.npy` if no tracking), biosensor B / G / Y stacks, and
  barcode class image. Progress bar reports each read.*
- *Optional **Multi-channel registration**: shift B/G/Y channels in pixels
  if needed, discard overexposed cells.*
- *Barcode alignment: set rotation (default 90° CW) and resize (default
  1024). Click **Align** → per-cell class labels are saved to
  `Bs2Code.xlsx` and overlaid as `Tracking masks cls`.*
- *Set the **Basal Frame Range** (default 0-21) — used to normalise
  fluorescence baselines per cell.*
- *(Optional) Tick **Frequency Domain Analysis** if the biosensor shows
  clear oscillations (e.g. Ca²⁺): yields per-cell dominant frequency +
  phase maps.*
- *Click the big green **▶  Calculate  (final step)** button. Progress
  bar walks through the Blue / Green / NIR extraction and Excel write.*

**Output:**

- `<sample>/signal_analysis.xlsx` — per-cell raw + normalised curves,
  plus `Statistics - *` sheets for per-class mean ± SE
- Per-class mean ± SE matplotlib figure (shown at the end)

**Duration:** ~15-20 s.

---

## Slide 12 — [TITLE] Celebration card

**What happens in the plugin:**

When Calculate finishes, NaCha pops up a "Walkthrough complete" dialog
that reports the total elapsed time from the moment PTU Reader was opened.

**Video card suggestion (right after the dialog):**

- Headline: **🎉 Walkthrough complete**
- Sub: *PTU Reader → Barcode Seg → Calculate FLIM-S → Seeded K-Means →
  Biosensor Seg → B&P Tracker → NaCha*
- Small line: *Outputs are in your sample folder; `clustered.xlsx` +
  `signal_analysis.xlsx` are the key deliverables.*

**Duration:** ~4 s.

---

## Slide 13 — [TITLE] Closing card

**Text:**

> **Questions / feedback** · baoyi.wang23@gmail.com
> Plugin: `bc-flim-spectra` · Module: `flim_s_gen`
> Repository: *(fill in URL)*

**Duration:** ~4 s. Fade to black.

---

# Appendix — Recording plan

- Recording: user manually, OBS Studio / Xbox Game Bar at 1920×1080 60 fps.
- Suggested section order matches the slide numbering above; stitch each
  step on top of a real napari screen capture.
- The transition cards (Slides 5-11) should appear as a brief full-screen
  text overlay (~3 s) between screen-captured steps. Use a consistent
  colour palette — e.g. dark blue panel for transition cards, white for
  title cards, amber accents for the "optional" callout (matches the
  yellow hint in the plugin UI).
- All elapsed-time figures in voiceover should be rough ranges — the
  plugin's built-in status bar / celebration dialog gives real numbers
  for the particular sample recorded.
