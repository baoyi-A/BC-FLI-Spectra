"""Standalone Cellpose runner (fine-tune + inference) — v2 / v4 aware.

Run as a subprocess to keep the main napari process free of PyTorch CUDA /
OpenGL state that otherwise corrupts vispy's context and crashes every
subsequent Image paint in the session.

The runner detects the installed Cellpose major version at startup and
dispatches to the correct API:

  * Cellpose 2.x:
      models.CellposeModel(gpu, device, model_type, pretrained_model)
      eval(img, diameter, channels) -> (masks, flows, styles)
      model.train(imgs, masks, channels, ...) saves under save_path/
  * Cellpose 4.x (CellposeSAM):
      models.CellposeModel(gpu, device, pretrained_model='cpsam' or path)
      eval(img, diameter, channel_axis=-1 if RGB) -> (masks, flows, styles)
      train.train_seg(model.net, train_data=imgs, train_labels=masks,
                      n_epochs=..., save_path=..., model_name=...)

The caller (`_widget.py`) chooses which python interpreter to launch
based on the model (file size / name heuristic) so this runner doesn't
have to bootstrap a different environment — it just speaks whichever
cellpose version it imports.

Usage:
    python _finetune_runner.py <cfg.pkl>

cfg.pkl (pickled dict) keys (common):
    op: 'train' (default) or 'infer'
    base_name: builtin (cyto2/nuclei/cpsam/...) OR custom local model name
    use_gpu: bool
    cellpose_src: optional path to a custom cellpose-main checkout
                  (only used by v2 — v4 must be installed in the env)
    extra_roots: list[str], default_finetune_roots: list[str]

For op='train':
    Single-image: img, mask, new_name, save_dir, n_epochs, channels
    Multi-image:  imgs, masks, new_name, save_dir, n_epochs, channels
        When both forms are present, the list form wins.
    -> stdout: RESULT:<trained-model-path>

For op='infer':
    img (2D or HxWxC), diameter, channels
    out_path: target .npy path to save masks to
    -> stdout: RESULT:<out_path>|<ncells>

Errors: ERROR:<msg>
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


# v2 builtin set; v4 just has 'cpsam'.
_V2_BUILTIN = {"cyto", "cyto2", "nuclei", "tissuenet", "livecell", "general"}
_V4_BUILTIN = {"cpsam"}


def _cellpose_major() -> int:
    try:
        from importlib.metadata import version
        v = version("cellpose")
    except Exception:
        # Local checkout w/o package metadata: assume v2 (BC-FLIM env).
        return 2
    try:
        return int(v.split(".")[0])
    except Exception:
        return 2


def _resolve_local_model(name: str, extra_roots, default_finetune_roots):
    p = Path(name)
    if p.exists():
        return p
    cache = Path.home() / ".cellpose" / "models" / name
    if cache.exists():
        return cache
    for d in default_finetune_roots:
        c = Path(d) / name / "models" / name
        if c.exists():
            return c
    for root in extra_roots:
        for cand in (
            Path(root) / "_finetune" / name / "models" / name,
            Path(root) / name / "models" / name,
        ):
            if cand.exists():
                return cand
    return None


def _device(use_gpu: bool):
    try:
        import torch  # type: ignore
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# v2 dispatch
# ---------------------------------------------------------------------------
def _run_v2(cfg) -> int:
    from cellpose import models  # type: ignore
    import numpy as np  # type: ignore

    base_name = str(cfg["base_name"]).strip()
    use_gpu = bool(cfg.get("use_gpu", True))
    device = _device(use_gpu)

    if base_name in _V2_BUILTIN:
        base_model = models.CellposeModel(
            gpu=use_gpu, device=device, model_type=base_name,
        )
    else:
        local = _resolve_local_model(
            base_name,
            extra_roots=cfg.get("extra_roots", []),
            default_finetune_roots=cfg.get("default_finetune_roots", []),
        )
        if local is None:
            print(f"ERROR:base model '{base_name}' not found locally (v2)")
            return 4
        base_model = models.CellposeModel(
            gpu=use_gpu, device=device, model_type=None,
            pretrained_model=str(local),
        )

    op = str(cfg.get("op", "train")).lower()
    if op == "infer":
        eval_kw = dict(
            diameter=float(cfg.get("diameter", 0) or None),
            channels=cfg.get("channels", [0, 0]),
        )
        if "cellprob_threshold" in cfg and cfg["cellprob_threshold"] is not None:
            eval_kw["cellprob_threshold"] = float(cfg["cellprob_threshold"])
        if "flow_threshold" in cfg and cfg["flow_threshold"] is not None:
            eval_kw["flow_threshold"] = float(cfg["flow_threshold"])
        out = base_model.eval(cfg["img"], **eval_kw)
        masks = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(masks, list):
            masks = masks[0]
        masks = np.asarray(masks, dtype=np.uint16)
        out_path = Path(cfg["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), masks)
        print(f"RESULT:{out_path}|{int(masks.max())}")
        return 0

    # train
    save_dir = Path(cfg["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    imgs_list = cfg.get("imgs"); masks_list = cfg.get("masks")
    if imgs_list is not None and masks_list is not None:
        if len(imgs_list) != len(masks_list):
            print(f"ERROR:imgs/masks length mismatch: {len(imgs_list)} vs {len(masks_list)}"); return 5
        if len(imgs_list) == 0:
            print("ERROR:imgs/masks lists are empty"); return 5
        train_imgs = list(imgs_list); train_masks = list(masks_list)
    else:
        train_imgs = [cfg["img"]]; train_masks = [cfg["mask"]]

    new_path = base_model.train(
        train_imgs, train_masks,
        channels=cfg.get("channels", [0, 0]),
        min_train_masks=1,
        save_path=str(save_dir),
        n_epochs=int(cfg["n_epochs"]),
        model_name=cfg["new_name"],
    )
    print(f"INFO:n_train={len(train_imgs)}")
    print(f"RESULT:{new_path}")
    return 0


# ---------------------------------------------------------------------------
# v4 dispatch (CellposeSAM)
# ---------------------------------------------------------------------------
def _run_v4(cfg) -> int:
    from cellpose import models  # type: ignore
    import numpy as np  # type: ignore

    base_name = str(cfg["base_name"]).strip()
    use_gpu = bool(cfg.get("use_gpu", True))
    device = _device(use_gpu)

    # v4: only one builtin name; otherwise pretrained_model is a path.
    if base_name.lower() in _V4_BUILTIN:
        base_model = models.CellposeModel(
            gpu=use_gpu, device=device, pretrained_model=base_name.lower(),
        )
    else:
        local = _resolve_local_model(
            base_name,
            extra_roots=cfg.get("extra_roots", []),
            default_finetune_roots=cfg.get("default_finetune_roots", []),
        )
        if local is None:
            print(f"ERROR:base model '{base_name}' not found locally (v4)")
            return 4
        base_model = models.CellposeModel(
            gpu=use_gpu, device=device, pretrained_model=str(local),
        )

    op = str(cfg.get("op", "train")).lower()
    if op == "infer":
        img = cfg["img"]
        # If image is HxWx3 RGB, tell cellpose 4 the channel axis.
        eval_kw = dict(diameter=float(cfg.get("diameter", 0) or None))
        try:
            arr_shape = np.asarray(img).shape
            if len(arr_shape) == 3 and arr_shape[-1] in (3, 4):
                eval_kw["channel_axis"] = -1
        except Exception:
            pass
        # Optional per-model thresholds passed from the config.json.
        if "cellprob_threshold" in cfg and cfg["cellprob_threshold"] is not None:
            eval_kw["cellprob_threshold"] = float(cfg["cellprob_threshold"])
        if "flow_threshold" in cfg and cfg["flow_threshold"] is not None:
            eval_kw["flow_threshold"] = float(cfg["flow_threshold"])
        out = base_model.eval(img, **eval_kw)
        masks = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(masks, list):
            masks = masks[0]
        masks = np.asarray(masks, dtype=np.uint16)
        out_path = Path(cfg["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), masks)
        print(f"RESULT:{out_path}|{int(masks.max())}")
        return 0

    # train (v4: cellpose.train.train_seg, takes the model.net)
    from cellpose import train as _train  # type: ignore
    save_dir = Path(cfg["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    imgs_list = cfg.get("imgs"); masks_list = cfg.get("masks")
    if imgs_list is not None and masks_list is not None:
        if len(imgs_list) != len(masks_list):
            print(f"ERROR:imgs/masks length mismatch: {len(imgs_list)} vs {len(masks_list)}"); return 5
        if len(imgs_list) == 0:
            print("ERROR:imgs/masks lists are empty"); return 5
        train_imgs = list(imgs_list); train_masks = list(masks_list)
    else:
        train_imgs = [cfg["img"]]; train_masks = [cfg["mask"]]

    new_path = _train.train_seg(
        base_model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        n_epochs=int(cfg["n_epochs"]),
        save_path=str(save_dir),
        model_name=cfg["new_name"],
    )
    print(f"INFO:n_train={len(train_imgs)}")
    print(f"RESULT:{new_path}")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("ERROR:expected cfg.pkl path as single argument"); return 2
    cfg_path = sys.argv[1]
    try:
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
    except Exception as e:
        print(f"ERROR:failed to load cfg: {e}"); return 2

    # v2 only: prepend the cellpose-main checkout if provided.
    cp_src = cfg.get("cellpose_src")
    if cp_src and Path(cp_src).exists() and cp_src not in sys.path:
        sys.path.insert(0, cp_src)

    try:
        import cellpose  # type: ignore  # noqa: F401
    except Exception as e:
        print(f"ERROR:cellpose import failed: {e}"); return 3

    major = _cellpose_major()
    print(f"INFO:cellpose_major={major}")
    try:
        if major >= 4:
            return _run_v4(cfg)
        return _run_v2(cfg)
    except Exception as e:
        import traceback
        print(f"ERROR:{e}")
        traceback.print_exc()
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
