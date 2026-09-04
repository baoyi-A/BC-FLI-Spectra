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
