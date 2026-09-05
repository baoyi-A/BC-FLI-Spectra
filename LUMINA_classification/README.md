# 🧠 LUMINA — Dual-Anchor Barcode Classification Network

*One of the two tools in [SLIC](../README.md). For the single-anchor workflow,
see the [napari plugin](../Napari_plugin/README.md).*

> 🤖 This repository ships an agent skill for LUMINA, [`.claude/skills/lumina-network/`](../.claude/skills/lumina-network/), so a coding assistant can drive these scripts — see the [root README](../README.md#-slic-works-with-an-ai-assistant).

![LUMINA network](../docs/lumina_network.png)

LUMINA classifies cells carrying **two** barcodes at once, one fluorophore on a
nuclear anchor and one on a mitochondrial anchor.

Averaging a whole cell into five numbers, as the single-anchor workflow does,
cannot separate two fluorophores sitting in two organelles: the cell average
mixes them. LUMINA therefore keeps the pixels. For each segmented cell it
builds a **six-plane stack** — the per-pixel phasor coordinates *G* and *S*,
three spectral intensity ratios, and the intensity — gives each plane its own
convolutional stem, fuses them in a shared trunk, and ends in **two independent
heads** under a class-balanced loss. One forward pass reads both anchors.

Training is two-stage: pre-train on single-anchor data, then fine-tune on
dual-anchor data.

---

## 🔧 Installation

A clean conda environment is strongly recommended.

```bash
# 1) Create and activate environment
conda create -n lumina python=3.10 -y
conda activate lumina

# 2) Install PyTorch
# Follow the official instructions for your OS / CUDA version:
# https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3) Install other dependencies
pip install -r requirements.txt
```

This environment is independent of the napari plugin's. Nothing here is shared
with it, so the two can be installed in either order or on their own.

---

## 🚀 Usage

Run the scripts in this order. Preprocessing must finish before training or
inference.

Every script is driven by command-line flags — `--help` on any of them lists the
full set with its defaults. The input and output roots are **required** and carry
no default, because there is no path that would be right on someone else's
machine; optional extras such as a second root are empty until you pass them. So
is the choice of which sample folders to touch, in the two scripts that write into
them — see `--samples` / `--all-samples` under steps 1 and 3. Every other flag
defaults to the value the script was developed with, so a run passing only the
required flags reproduces the reference configuration. Each run
also writes a `*_run_config.csv` beside its output recording every flag it
resolved, so a directory of results says what produced it.

### 1. 🧹 `Data_prep.py` — build the per-cell crops

```bash
python Data_prep.py --data-root /path/to/dataset --samples sampleA,sampleB
```

Expects `<root>/<sample>/` holding `raw/`, `flim_stack/<fov>-sum.tif`,
`intensity/<fov>-{1..4}.tif` and `intensity/<fov>-sum_seg.npy`, and writes one
six-plane `cell<id>_5D.tif` per segmented cell into `<sample>/seg_5D_calib/`.

Which sample folders it touches is never a default. Name them with `--samples`,
or pass `--all-samples` for every sample folder under `--data-root` that has a
`raw/`; passing neither is an error. The distinction matters because writing a
sample **clears** its output folder first, so `--all-samples` clears and
regenerates every one of them. The resolved list is printed one path per line
before the first deletion.

The calibration flags — `--calibration-factors`, `--phi-calib`, `--m-calib`,
`--rep-rate-mhz`, `--tau-resolution` — describe **your microscope**, not the
sample. Their defaults are the values this dataset was prepared with; check each
against your own acquisition metadata before trusting the phasor coordinates.

### 2. 🏋️ `Train_LUMINA.py` — train the classifier

```bash
python Train_LUMINA.py \
    --data-root /path/to/single_anchor \
    --single-anchor-manifest single_anchor.csv \
    --dual-root /path/to/dual_anchor \
    --dual-samples-file dual_samples.txt \
    --checkpoint best_model_fine-tune.pth \
    --out ./train_out
```

`--single-anchor-manifest` is a CSV of `class,folder` saying which folder holds
which barcode. Give `--dual-samples` (comma-separated) or `--dual-samples-file`
(one name per line), and `--checkpoint` or `--from-scratch`. Note that
`Train_LUMINA.py` reads `--seg-folder seg_5D` while `Data_prep.py` writes
`seg_5D_calib`, so pass the matching name to one of the two.

**To reproduce a training run you need the dual-anchor list in the same order,
not just the same names.** The folders are loaded in the order you give them and
the validation split is taken by row position, so the same set of samples in a
different order holds out different cells at the same `--seed` — and therefore
produces a different `val_df.xlsx` and a different checkpoint. Keep the
`--dual-samples-file` beside the results; the resolved order is printed at
startup and recorded as `resolved_dual_samples` in `train_run_config.csv`.

### 3. 🔍 `Test_LUMINA.py` — run inference

```bash
python Test_LUMINA.py \
    --checkpoint ./train_out/best_model_fine-tune.pth \
    --data-root /path/to/dataset --samples sampleA,sampleB
```

Writes `predict_class_confident_<threshold>.xlsx` and
`predict_class_uncertain_<threshold>.xlsx` back into each sample folder. Add
`--out` to redirect them, `--device cpu` on a machine without a GPU.

Which sample folders it touches is never a default, for the same reason as in
`Data_prep.py`: name them with `--samples`, or pass `--all-samples` for every
folder under the root(s) that holds a `clustered.xlsx` or a folder of crops;
passing neither is an error. Without `--out` the two workbooks are written back
into each sample folder and overwrite the ones already there, so `--all-samples`
overwrites them everywhere. The resolved list of destinations is printed one per
line before the first workbook is written.

### 4. 🔥 `Visualize_heatmap.py` — plot the combination heatmap

```bash
python Visualize_heatmap.py --data-root /path/to/dataset --no-gui
```

Drop `--no-gui` for the interactive window, in which clicking a cell prints its
per-sample breakdown; the PDF is written either way. **Run it with the same
`--confidence-threshold` as `Test_LUMINA.py`** — the threshold is part of the
workbook filename, so a mismatch means no file is found.

Both axes of the figure are *predictions*, so it is a co-occurrence table, not a
confusion matrix.

`Finetune_LUMINA.py` is optional and sits between training and inference — see
below.

---

## 🎯 Moving a checkpoint to a new cell line

A checkpoint learns its decision boundaries on one cell line under one set of
imaging conditions. On a different cell line the spectral features shift while
those boundaries stay where they were, and accuracy drops. The barcodes are
still separable; the two heads are simply miscalibrated.

`Finetune_LUMINA.py` fixes that by **few-shot fine-tuning of the two classifier
heads only**. Everything else — the six input stems and the shared backbone —
stays frozen, so a small number of labelled cells from the new dish is enough,
and the fine-tuning step itself will run on a CPU. The run as a whole still
pushes every crop of every dish through the frozen backbone once before it gets
there, so budget for that pass rather than for the fine-tune.

```bash
python Finetune_LUMINA.py --checkpoint best_model_fine-tune.pth \
    --data-root /path/to/new_dish --out ./adapt_out --k 20
```

That adapts on *K* cells per barcode combination — the pair of nuclear and
mitochondrial barcodes, which is not necessarily one sample folder — and reads
out every remaining cell of the dish. To test whether the adaptation survives a
change of imaging day, add `--eval-root /path/to/another_dish`, which scores a
completely different dish instead — it has to be a genuinely different one, and
the script refuses to score any cell it just fine-tuned on. Run several `--seeds`
and report the spread; a single support draw is not a result. `--help` lists
every knob.

At startup the run prints the combinations it will draw from, which folder of
crops it read for each sample (`seg_5D` or `seg_5D_calib`, see `--seg-folder`)
and where each sample's labels came from. Read those three lines before reading
the numbers.

All five scripts take command-line arguments, but here the flags *are* the
experiment rather than plumbing: `--k`, `--seed`, `--eval-root` and the curation
flags each move the number that comes out, and several move the denominator
rather than the score. A result from this script means nothing without them,
which is why every one is recorded in `finetune_run_config.csv`.

### The two numbers it prints

For every barcode combination, and overall:

- **Detection** — the fraction of evaluated cells whose nuclear *and*
  mitochondrial call both clear the confidence threshold.
- **Accuracy** — among *those* cells only, the fraction where both calls are
  correct.

Accuracy is conditional on detection, so a higher threshold buys accuracy by
shrinking the denominator. Read them as a pair, and compare two settings only at
matched detection. By default the un-adapted checkpoint is scored on exactly the
same held-out cells, so the before/after comparison is on identical cells.

The heads keep their original width. Your panel may be a subset of the classes
the checkpoint was trained on, but the barcode names have to be the same names,
and a barcode the checkpoint never saw cannot be added this way.

---

## 📦 Data

The [Dual-Anchor dataset](https://zenodo.org/records/17036213) used to train
the network is deposited on Zenodo.

---

## ⏱ Expected runtime

- Inference on a single sample or image: **~1 second**.
- Including preprocessing, such as segmentation and data extraction:
  typically **~3 minutes**.
- Training time depends on dataset size and GPU, and can be considerably
  longer.

---

## 📝 Notes

- Check that your GPU and CUDA drivers are configured before training.
  `Train_LUMINA.py` and `Test_LUMINA.py` default to `--device cuda:0` with no
  CPU fallback, so they fail immediately rather than running unusably slowly.
- Adjust the hyperparameters to your dataset size and GPU memory with flags —
  `--batch-size`, `--crop-size`, `--epochs`, `--finetune-epochs`, `--lr` — not by
  editing the script. The defaults are the reference configuration, and the flags
  you passed are recorded in `train_run_config.csv`.
- The manuscript describing the method has been submitted but is not yet
  published. The schematic above is a figure panel from it.

## 📜 License

BSD 2-Clause, the same as the rest of the repository. See
[`Napari_plugin/LICENSE`](../Napari_plugin/LICENSE).
