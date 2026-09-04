# 🧠 LUMINA — Dual-Anchor Barcode Classification Network

*One of the two tools in [SLIC](../README.md). For the single-anchor workflow,
see the [napari plugin](../Napari_plugin/README.md).*

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

| Script | What it does |
|---|---|
| 🧹 `Data_Prep.py` | Preprocess the raw dataset into the training format. |
| 🏋️ `Train_LUMINA.py` | Train the classification model, two-stage as above. |
| 🔍 `Test_LUMINA.py` | Run inference on new data. |
| 🔥 `Visualize_heatmap.py` | Render the classification results as heatmaps. |

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

Unlike the four scripts above, this one takes command-line arguments. That is
deliberate: the numbers only mean something alongside the support set, the
held-out set and the seed that produced them, so each is a flag and each is
recorded in the output CSV.

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
- Adjust the hyperparameters in the training script to your dataset size and
  GPU memory.
- The manuscript describing the method has been submitted but is not yet
  published. The schematic above is a figure panel from it.

## 📜 License

BSD 2-Clause, the same as the rest of the repository. See
[`Napari_plugin/LICENSE`](../Napari_plugin/LICENSE).
