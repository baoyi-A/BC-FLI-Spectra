# BC-FLIM-Spectra — a napari plugin

## Quick Installation Guide

Follow these steps to install the plugin in an isolated environment:

### 1) Create and activate a clean environment

Using **conda** (recommended):

```bash
conda create -n nacha python=3.10 -y
conda activate nacha
Or using venv:

python -m venv nacha
source nacha/bin/activate    # macOS/Linux
nacha\Scripts\activate       # Windows
2) Install PyTorch (follow official instructions)
Visit the PyTorch installation guide and use the command that matches your OS and hardware. For example, with CUDA 12.1:

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
3) Install Track-Anything (follow official instructions)
Please refer to the Track‑Anything GitHub repository and follow their installation guide to ensure any required models or dependencies are properly set up.

4) Install this plugin (editable mode recommended)
Navigate to the plugin root directory (where pyproject.toml is located) and run:

pip install -e .
The editable install allows you to modify the code and see changes immediately without reinstalling. If you prefer a standard install, use:

pip install .