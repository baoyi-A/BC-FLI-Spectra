# The seven steps, control by control

Parameter descriptions here are the same text the widget shows as a tooltip, so
the GUI and this file cannot drift apart. Defaults are for a Leica SP8 /
STELLARIS FALCON at 78.1 MHz; other instruments need the timing numbers
changed.

---

## 1. PTU Reader — decode `.ptu` into images

**In:** a folder of `.ptu` files. **Out:** `intensity/*_ch1..4.tif`,
`intensity/*_sum.tif`, `flim_stack/*.tif`, `*_fastflim_tau.tif`, and a cached
`*_fastflim_rgb.png` colour render.

| Control | What it does |
|---|---|
| Input folder | Folder of `.ptu` files to decode. Output folder auto-updates to its parent. |
| Output folder | Where the decoded TIFs and the FastFLIM snapshot are written. This becomes the **sample folder** for every later step. |
| Frame | −1 sums every frame in the PTU (typical). A positive N picks a single 0-based frame. |
| Tau resolution | Nanoseconds per time bin in the decay. Leica default 0.098 ns. |
| Tau min / Tau max | Blue and red ends of the lifetime colormap, in ns. Auto-set from the data's 12th / 88th percentile on the first run. Display only. |
| Intensity clip | Upper percentile for brightness normalisation. Lower = darker. |
| Brightness gamma / floor | Shadow lift and a minimum brightness so colours do not crush to black. Re-render live. |
| CLAHE + tile | Local contrast equalisation so dim cells show colour as clearly as bright ones. Tile ≈ half a cell diameter. **Display only** — never affects segmentation or quantification. |
| No enhancement | One click bypasses all of the above: CLAHE off, gamma 1.0, floor 0. Toggling back restores your values. |
| Auto contrast (3 buttons) | Cycle 6 presets: master (τ + intensity together), τ only, intensity only. |
| Skip already-processed | Skips a PTU whose three outputs all exist. Lets you resume an interrupted batch. |

Decoding is slow — tens of seconds per PTU. If the outputs already exist, the
button offers **Re-render from existing** instead, which reloads the cached τ
and intensity and re-renders instantly, so contrast can be tuned on old data
without decoding again.

The display settings matter beyond looks: the segmentation models were trained
on this render, so the *default* render parameters are part of the model
contract. Change them for viewing, not before segmenting.

---

## 2. Barcode Seg (Cellpose) — N and P masks

**In:** `intensity/*_sum.tif` plus the FLIM stack or τ map. **Out:**
`intensity/<stem>_seg_n.npy` and `_seg_p.npy`.

The Cellpose input is not the raw intensity: the widget builds the FastFLIM
render (Leica blue→green→red, 30/85 percentile auto-range, CLAHE on, gamma
0.55) and segments that, because that is what the models were trained on.

| Control | What it does |
|---|---|
| Segment N / Segment P | Nucleus and cytoplasm heads. Untick one to skip it; existing masks on disk are left alone. Both off is rejected. |
| N model / P model | The default is the first known model present on the machine: the published `NinNC-260328-1` / `CinNC-260328-1` (Cellpose v2, ~27 MB), else the lab's CellposeSAM models, else a public base model. A name containing `cpsam` routes to the v4 environment with RGB input; other names are v2 with grayscale input. Routing is automatic, so both generations can sit in the dropdown together. |
| Diameter N / P | Approximate object diameter in pixels; 0 lets Cellpose estimate it (slower). P is typically ~2× N. |
| Channels N / P | v2 takes at most 2 of the RGB channels (`0`=gray, `1`=R, `2`=G, `3`=B). v4 `RGB all 3 channels` feeds the full H×W×3. `(auto)` defers to the model's `config.json`. |
| Input kind N / P | Which form of the image the model gets: grayscale render, RGB render, or raw intensity sum. `(auto from model)` uses the model's `config.json`, else v4→RGB / v2→gray. |
| Use GPU | CUDA if available, else CPU. ~10× faster on 2k×2k. |
| Auto Segment N & P | Run both heads on this FOV and save the masks next to the sum tif. |
| Auto Segment ALL FOVs | Same for every `*_sum.tif` in the folder. Then walk the results with ← Prev / Next →. |
| close_holes / erode / dilate / min area | Per-cell post-processing in pixels. `min area` 200 clears specks. All destructive; **↶ Undo** (Shift+Z) restores one level. |
| ♻ Re-apply | Re-run post-processing on the masks in the viewer without a Cellpose run — cheap parameter sweeps. Overwrites both the layer and the `.npy`. |
| Save masks | Only needed after manual edits; Auto Segment already saved its own output. Shift+S. |

**Editing masks in napari:** right-click draws, Enter commits, Ctrl+click
deletes, Z / X toggle layer visibility, S cycles contrast.

**Fine-tuning.** *Fine-tune N / P* trains on **this one FOV** — the image plus
the mask layer you just edited — and saves the new model under
`<sample>/_finetune/`. *Multi-folder fine-tune* opens a dialog: add sample
folders, and **every FOV in each folder that already has a saved mask** is used.
The table shows the count per folder and says why a folder is skipped. Training
images go through the same input routing as inference, so a `cpsam` model is
fine-tuned on the RGB render, not on grayscale. Epochs 50–200 is typical for
small corrections; more risks overfitting. The new model records its input form
in a `config.json` beside the weights.

**Bringing your own model: the `config.json` keys.** Different finetuned models
often need different inputs and inference parameters. Rather than forking the
widget per model, the plugin reads an optional `config.json` sitting next to the
model weight and applies its settings transparently.

| Key | What it does |
|---|---|
| `input_kind` | `intensity_sum` (raw `_sum.tif`), `barcode_seg_grayscale` (FastFLIM luminance, default), or `barcode_seg_rgb` (FastFLIM 3-channel render) |
| `diameter` | Pre-fills the BarcodeSeg diameter spinbox when the model is selected |
| `cellprob_threshold` | Forwarded to `model.eval(...)` — needed for sparse / membrane models |
| `flow_threshold` | Forwarded to `model.eval(...)` |
| `channels` | Cellpose v2 channel indices |
| `post_process.method` | `merge_fragments` (morph close + min-area filter) or `none` |
| `post_process.merge_gap` | Closing radius in pixels |
| `post_process.min_merged_px` | Drop connected components smaller than this after closing |
| `notes` | Free text shown in the status hint when the model is picked |

**Discovery.** When the user picks a model in the dropdown, the plugin looks for
`config.json` (a) right next to the weight file, (b) in the weight's parent dir,
or (c) one level up — the `…/<model_name>/models/<model_name>` layout `git`
checkouts use. First hit wins. Missing → legacy behaviour: FastFLIM grayscale
input, cellpose defaults, no post-processing, so every model without one keeps
working unchanged. A `📄cfg` tag appears in the status hint when an override
took effect. A worked example ships under `Napari_plugin/examples/`.

---

## 3. Calculate FLIM-S — the 5-D fingerprint

**In:** `flim_stack/*_ch[1-4].tif` (T×H×W, time first) + the N/P masks. **Out:** `FLIM-S.xlsx`,
one row per cell, plus a G–S scatter.

| Control | What it does |
|---|---|
| Stack 1–4 | Per-channel decay stacks, axis order **T×H×W** (time first). All optional — auto-filled from `flim_stack/`. At least one is required; missing channels are written as NaN. |
| Segmentation N / M / P / Any | Which masks to quantify. `Any` is a fallback used only when N, M and P are all empty. |
| **Pulse frequency (MHz)** | Laser repetition rate. Leica default 78.1. Sets the time window the phasor is evaluated in, so a wrong value makes every lifetime wrong. |
| **Tau resolution (ns)** | Time-bin width. Leica with a 256-bin decay: 0.097 ns. Check the PTU metadata if unsure. |
| Mask intensity threshold | Minimum **total** photons summed over a cell. Below it the cell is skipped as too dim for a stable fit. Raise if τ outliers persist, lower if real dim cells are dropped. |
| Pixel intensity threshold | Per-**pixel** floor when aggregating photons into the cell decay, so dim background does not dilute the fit. Different from the mask threshold. |
| Peak offset (bins) | Start of the fit window, after the decay peak — skips the IRF-convolved rising edge. Typical 3–6. |
| End offset (bins) | End of the fit window, before the decay ends — drops the noisy late tail. Typical 10–20. |
| Harmonics | Phasor harmonic order. **Use 1** for single-anchor barcode FLIM. Higher orders probe faster components but amplify noise. |
| Fresh FLIM-S.xlsx | See the merge rule below. |
| Process and Save to Excel | This FOV. |
| ▶▶ Process ALL FOVs (from disk) | Every FOV under `flim_stack/`, using the same channel and mask selection as the slots above. Leave every slot empty to use whatever exists on disk. |

**The merge rule, which surprises people.** A run does not overwrite the
workbook. It replaces the rows of *this* FOV and keeps every other FOV's rows,
so single-FOV runs accumulate into one table. The consequence: rows written
earlier with different channels or different masks survive. The status line
reports new-vs-kept counts and flags kept rows from a different configuration,
and the G–S popup draws kept rows in grey. Tick **Fresh FLIM-S.xlsx** to move
the old workbook aside (`FLIM-S_old_<timestamp>.xlsx`, never deleted) and get a
single-configuration file.

**Columns:** `Localization, G, S, Lifetime, Chi^2, Total intensity, Area (px),
Mean intensity, Mask label, FastFLIM, Int 570-590, Int 590-610, Int 610-638,
Int 638-720, Int 1/(1-4) … Int 4/(1-4), FOV`.

`Lifetime` is a fitted mono-exponential τ. `FastFLIM` is the photon-weighted
mean arrival time. They answer different questions; do not average them
together or swap the names.

---

## 4. Seeded K-Means — assign a barcode to each cell

**In:** `FLIM-S.xlsx`. **Out:** `clustered.xlsx` with a class per cell.

The feature space is 5-D: `G`, `S`, `Int 1/(1-4)`, `Int 2/(1-4)`,
`Int 3/(1-4)`, z-scored on the cells currently loaded and then multiplied by
the per-axis weights.

Order of operations: **Read and Plot** → place or load seeds → **Run K-Means**
→ inspect → **Save Results**.

| Control | What it does |
|---|---|
| Number of clusters | Barcode classes. Auto-syncs to the row count of a loaded seeds file. |
| Method | `Seeded K-Means` (Basu 2002) uses your seeds as centroid initialisation and refines them. The others (K-Means++, MiniBatchKMeans, Gaussian Mixture, Spectral) ignore seeds and cluster blind. |
| **Whiten by within-cluster spread** | After a first clustering, measures how wide the clusters are and rescales the space so one distance unit means one cluster width, then clusters again from the same seeds. Uses no labels. This is what lets a seed set from one acquisition transfer to another without the two closest barcodes swapping. Auto-skipped when fewer than 8 clusters are populated. While ticked, the weights below barely matter. |
| Weight G / S / Int1–3 | Stretch one axis each. Cannot describe a long, thin, tilted cluster — that is what whitening is for. |
| Auto-detect outliers → class 0 | Per-class Isolation Forest. Cells that do not look like the rest of their cluster are declined to class 0 and excluded from per-class results. It catches segmentation fragments, dim and non-expressing cells; it **cannot** catch a cell sitting neatly inside the wrong cluster. |
| Outlier contamination | Fraction declined per class. 0.05–0.15 typical. |
| Per-FOV K-Means | Cluster each ticked FOV separately instead of pooling. Not available for seeded K-Means (seeds are global). |
| Save / Load Seeds | Seeds are a **starting point**, not fixed centres: the clustering re-fits them on the cells now loaded, which is what makes a reference reusable. Stars are drawn on the nearest cells for display; K-Means starts from the loaded coordinates. Clicking a new seed or dragging a star makes that seed a cell again. |
| Save / Load Distribution | Per-class convex hulls saved as `.npz`, redrawn as translucent polygons behind the scatter to guide manual seeding. Expand factor 1.0 = raw hull. |

**Placing seeds by hand:** click once per class on the scatter, in the same
localisation you are about to run. Clicking more than K times clears them all
and starts over. Drag a star to move it.

**After clustering:** shift+click for the lasso, keys 1–9 and a–z reassign a
class by hand. Then **Save Results**, which also adds the classification layers
to napari.

**Harmony calibration (optional, default off).** Aligns query cells onto a
labelled reference and predicts labels directly, bypassing manual seeds. Needs
`harmonypy`. The reference CSV's label column differs per cell line: A549 uses
`NLabelDisplay` (not `NLabel`, which has stale numbers), HEK and MDA use
`NLabel`, SKOV3 uses `CorrectedBarcode`. Defaults θ=4, 20 soft clusters, k=15;
raising the cluster count toward the harmonypy default of 100 collapses rare
classes and must not be done without checking class preservation.

---

## 5. Biosensor Seg (Cellpose) — segment the biosensor channel

**In:** the confocal B/G/Y stacks + the barcode classification image. **Out:**
`<fov>_seg_image.tif` and `<fov>_seg_image_seg.npy`.

| Control | What it does |
|---|---|
| Stack B / G / Y + Use B/G/Y | The confocal channels. Untick a channel that was not acquired. |
| Frame start / end | 1-based, inclusive. Default 1–1: the BGY-render models were trained on frame 1 only, so multi-frame averages deviate from the training distribution. |
| Generate seg image | For a BGY-render model: each enabled channel is CLAHE + gamma corrected, stacked Y→R, G→G, B→B, then reduced to BT.601 luminance, matching training. Disabled channels are zeros. |
| Model | Default `BS-BC-assist-cls-bgy-260426`, a v2 model taking the render grayscale **plus** the barcode class as a second channel. |
| Barcode cls TIF | One class label per cell, from step 4. Auto-filled. |
| Rotation | Leica tilescan barcodes are rotated 90° CW relative to the confocal stack — hence the default. Use 0° for a single FOV. |
| Load / Confirm Barcode | Loads, rotates and resizes the barcode layer so you can **see** the registration before committing to a Cellpose run. Do this first with an assist model. |
| Save mask | Writes `<stem>_seg.npy` next to the seg image. |

Fine-tuning works as in step 2. Multi-folder fine-tuning is refused for
`BS-BC-assist` models, because their second channel is built from *this* FOV's
barcode classification and other folders cannot supply it.

---

## 6. B&P Tracker — time-lapse only

Propagates the single-frame mask through the confocal time series
(XMem / Track-Anything). Skip for fixed samples. Its input is normally
`seg_image_seg.npy` from step 5.

---

## 7. NaCha — per-class readout

Final alignment plus per-class signal computation. Shift+click any cell in
Revise Mode to inspect its individual curve before trusting a class average —
one bad cell in a small class moves the mean visibly.

A completion dialog reports the total elapsed time since PTU Reader opened.
