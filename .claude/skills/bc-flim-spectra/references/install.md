# Installing BC-FLIM-Spectra

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

The two Cellpose widgets need `cellpose` and `torch`:

```bash
pip install -e ".[segmentation]"
```

Install `torch` first from <https://pytorch.org/get-started/locally/> if you
need a particular CUDA build — the extra will not replace one that is already
there. CPU-only torch works; Cellpose falls back to CPU automatically and is
roughly 10× slower on 2k×2k images.

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
