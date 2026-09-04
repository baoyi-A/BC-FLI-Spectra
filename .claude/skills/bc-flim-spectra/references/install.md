# Installing SLIC

## Minimum: the analysis half, one environment

PTU Reader, Calculate FLIM-S and Seeded K-Means need no deep learning. If the
user only wants to decode data and get per-cell features, this is enough:

```bash
conda create -n nacha python=3.10 -y && conda activate nacha
pip install "napari[all]"            # napari needs a Qt backend; the plugin does not pin one
cd BC-FLI-Spectra/Napari_plugin && pip install -e .
```

Verify without opening a window:

```bash
python -c "import flim_s_gen._widget as w; print(w.__file__)"
```

Launch straight onto a widget:

```bash
python -m napari -w bc-flim-spectra "PTU Reader"
```

## Adding segmentation

The two Cellpose widgets need `cellpose` and `torch`.

**Install a CUDA build of torch FIRST if the machine has an NVIDIA GPU.** The
default torch wheel on PyPI is CPU-only, so `pip install -e ".[segmentation]"`
on its own gives you a CPU install even on a machine with a good GPU — and
nothing fails, it just runs about ten times slower (tens of minutes for one
2k×2k field instead of under a minute). Take the command for your CUDA version
from <https://pytorch.org/get-started/locally/>, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[segmentation]"      # will not replace the torch you just installed
```

On a machine with no GPU, `pip install -e ".[segmentation]"` alone is correct.

Check what you actually got — and note that the answer that matters is the one
from the environment Cellpose runs in, not from napari's:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The segmentation widgets show the same thing in their header as a
**Compute:** line, per Cellpose environment.

## One environment is usually enough

`pip install -e ".[segmentation]"` puts a current Cellpose in the same
environment as the plugin, and that environment is then a valid v4 slot by
itself. Steps 1-4 — PTU decoding, segmentation, FLIM-S and clustering — run
without any extra environment. Point the plugin at itself if the auto-detection
picks something else:

```bash
export BCFLIM_CELLPOSE_V4_PYTHON=/path/to/envs/<your-env>/bin/python
```

The three-environment setup below matters when you need **both** Cellpose
generations, because models trained on 2.x and 4.x are not interchangeable.

## Why three environments

Cellpose 2.x and 4.x have incompatible APIs and model formats, and the models
in use span both generations. PyTorch also leaves CUDA/OpenGL state that
crashes napari's renderer, so the plugin runs **every** segmentation and
fine-tune in a child interpreter. Hence:

```bash
# cellpose 2.x — legacy 2-channel models
conda create -n cellpose2 python=3.10 -y && conda activate cellpose2
pip install "cellpose==2.2.3"

# cellpose 4.x — CellposeSAM, the current default models
conda deactivate && conda create -n cellpose4 python=3.10 -y && conda activate cellpose4
pip install "cellpose>=4.1,<5"
```

The plugin scans conda environments at startup, scores candidates by name
affinity and version, and caches the choice in `~/.bc_flim_spectra_envs.json`.
Override with `BCFLIM_CELLPOSE_V2_PYTHON` / `BCFLIM_CELLPOSE_V4_PYTHON`.

**Only the two segmentation widgets need this.** Reading, editing and testing
the code needs one environment.

## Model weights

**The fine-tuned models are not in this repository** — the weights are
distributed separately. The published set is three Cellpose v2 models of about
27 MB each, the same ones the reviewer demo runs:

| Model | Role |
|---|---|
| `NinNC-260328-1` | barcode nucleus (N) |
| `CinNC-260328-1` | barcode cytoplasm (P) |
| `BS-BC-assist-cls-260402-forDense` | biosensor cells |

Each widget's default is the **first model it can actually find**, in this
order: the published set, then the lab's larger CellposeSAM models on machines
that have them, then a public Cellpose base model (`nuclei`, `cyto2`) so a
clean checkout still segments something. The resolved default is named in the
model dropdown's tooltip, and the barcode dropdown remembers your last choice.
With no custom weights at all, segmentation runs but will not reproduce the
published masks.

Point the plugin at a folder of models with either:

```bash
export BCFLIM_MODEL_ROOT=/path/to/models      # PowerShell: $env:BCFLIM_MODEL_ROOT = "..."
```

```python
import flim_s_gen; flim_s_gen.set_barcode_model_root(r"D:/path/to/models")  # remembered
```

The folder may be flat (`<root>/<name>`), one directory per model
(`<root>/<name>/<name>`, with or without a `_BEST` suffix), or the layout the
plugin's own fine-tuning writes (`<root>/<name>/models/<name>`). The widget
header reports which root it resolved and how many models it found there.

`cpsam` (~1.15 GB) downloads from `cellpose.org` on first use, which is often
blocked in China:

```bash
export HF_ENDPOINT=https://hf-mirror.com
# or fetch https://hf-mirror.com/mouseland/cellpose-sam/resolve/main/cpsam
# into ~/.cellpose/models/cpsam
```

Custom and fine-tuned weights are searched under `BCFLIM_MODEL_ROOT`, the
per-sample `_finetune/` folder, and `~/.cellpose`.

## Time-lapse only

B&P Tracker and NaCha use Track-Anything:

```bash
git clone https://github.com/gaomingqi/Track-Anything.git
cd Track-Anything && pip install -r requirements.txt
```

Its torch constraint is loose and can upgrade torch past your driver. If that
happens, install it with `--no-deps` and resolve by hand.

## Constraints that bite

- **numpy stays below 2.0.** napari 0.4.19 loads numba, which requires ≤ 1.26.
- **No Qt backend is pinned**, deliberately, to avoid fighting other napari
  plugins. `napari[all]` brings one; otherwise install PyQt5 or PySide2 to
  match your napari.
- **Headless / CI:** prefer `opencv-python-headless` over `opencv-python`.
- A first end-to-end run on a fresh machine downloads several GB (torch, the
  Cellpose weights, and any demo data). Plan for bandwidth, not CPU.

## Standalone classifier

`LUMINA_classification/` is independent of the plugin — its own conda
environment, its own `requirements.txt`, PyTorch installed separately.
