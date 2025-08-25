# LUMINA — Dual-Anchor Barcodes Classification Network

LUMINA is a deep learning framework for classifying dual-anchor barcodes.

## Installation

It is strongly recommended to use a clean conda environment.

```bash
# 1) Create and activate environment
conda create -n lumina python=3.10 -y
conda activate lumina

# 2) Install PyTorch
# Please follow the official instructions for your OS / CUDA version:
# https://pytorch.org/get-started/locally/
# Example (CUDA 12.1):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3) Install other dependencies
pip install -r requirements.txt
```

## Usage

- **Data_Prep.py**: preprocess the raw dataset into the required format for training.  
- **Train_LUMINA.py**: train the LUMINA classification model.  
- **Test_LUMINA.py**: run inference (testing) on new data.  
- **Visualize_heatmap.py**: visualize classification results as heatmaps.  

## Notes

- Ensure your GPU and CUDA drivers are properly configured before training.  
- Preprocessing must be completed before running training or inference.  
- Adjust hyperparameters in the training script according to your dataset size and GPU memory.  

---

**Enjoy using LUMINA!**
