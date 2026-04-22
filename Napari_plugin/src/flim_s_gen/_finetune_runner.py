"""Standalone Cellpose runner (fine-tune + inference).

Run as a subprocess to keep the main napari process free of PyTorch CUDA /
OpenGL state that otherwise corrupts vispy's context and crashes every
subsequent Image paint in the session.

Usage:
    python _finetune_runner.py <cfg.pkl>

cfg.pkl (pickled dict) keys (common):
    op: str, either 'train' (default) or 'infer'
    base_name: str (builtin name like "cyto2" or a custom local model name)
    use_gpu: bool
    cellpose_src: str (optional path to custom cellpose-main checkout)
    extra_roots: list[str], default_finetune_roots: list[str]

For op='train':
    Single-image form:  img,  mask,  new_name, save_dir, n_epochs, channels
    Multi-image form:   imgs, masks, new_name, save_dir, n_epochs, channels
        `imgs` and `masks` are lists of ndarrays (same length).
        When both forms are present in cfg, the multi-image form wins.
    -> stdout: RESULT:<trained-model-path>

For op='infer':
    img (2D or HxWxC), diameter, channels
    out_path: str (target .npy path to save masks to)
    -> stdout: RESULT:<out_path>|<ncells>

Errors: ERROR:<msg>
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path


_BUILTIN = {"cyto", "cyto2", "nuclei", "tissuenet", "livecell", "general"}


def _resolve_local_model(name: str, extra_roots, default_finetune_roots):
    # exact path?
    p = Path(name)
    if p.exists():
        return p
    # cellpose cache
    cache = Path.home() / ".cellpose" / "models" / name
    if cache.exists():
        return cache
    # plugin default roots (glob expansion done by caller — treat each as dir)
    for d in default_finetune_roots:
        c = Path(d) / name / "models" / name
        if c.exists():
            return c
    # extra roots (sample folder's _finetune, etc.)
    for root in extra_roots:
        for cand in (
            Path(root) / "_finetune" / name / "models" / name,
            Path(root) / name / "models" / name,
        ):
            if cand.exists():
                return cand
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print("ERROR:expected cfg.pkl path as single argument")
        return 2
    cfg_path = sys.argv[1]
    try:
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
    except Exception as e:
        print(f"ERROR:failed to load cfg: {e}")
        return 2

    cp_src = cfg.get("cellpose_src")
    if cp_src and Path(cp_src).exists() and cp_src not in sys.path:
        sys.path.insert(0, cp_src)

    try:
        from cellpose import models  # type: ignore
    except Exception as e:
        print(f"ERROR:cellpose import failed: {e}")
        return 3

    try:
        import torch  # type: ignore
        if cfg.get("use_gpu") and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    except Exception:
        device = None

    base_name = str(cfg["base_name"]).strip()
    use_gpu = bool(cfg.get("use_gpu", True))
    if base_name in _BUILTIN:
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
            print(f"ERROR:base model '{base_name}' not found locally")
            return 4
        base_model = models.CellposeModel(
            gpu=use_gpu, device=device, model_type=None,
            pretrained_model=str(local),
        )

    op = str(cfg.get("op", "train")).lower()
    if op == "infer":
        try:
            import numpy as np  # type: ignore
            out = base_model.eval(
                cfg["img"],
                diameter=float(cfg.get("diameter", 0) or None),
                channels=cfg.get("channels", [0, 0]),
            )
            masks = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(masks, list):
                masks = masks[0]
            masks = np.asarray(masks, dtype=np.uint16)
            out_path = Path(cfg["out_path"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), masks)
            print(f"RESULT:{out_path}|{int(masks.max())}")
            return 0
        except Exception as e:
            print(f"ERROR:infer failed: {e}")
            return 6

    # Default: train
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    imgs_list = cfg.get("imgs")
    masks_list = cfg.get("masks")
    if imgs_list is not None and masks_list is not None:
        if len(imgs_list) != len(masks_list):
            print(f"ERROR:imgs/masks length mismatch: {len(imgs_list)} vs {len(masks_list)}")
            return 5
        if len(imgs_list) == 0:
            print("ERROR:imgs/masks lists are empty")
            return 5
        train_imgs = list(imgs_list)
        train_masks = list(masks_list)
    else:
        train_imgs = [cfg["img"]]
        train_masks = [cfg["mask"]]

    try:
        new_path = base_model.train(
            train_imgs, train_masks,
            channels=cfg.get("channels", [0, 0]),
            min_train_masks=1,
            save_path=str(save_dir),
            n_epochs=int(cfg["n_epochs"]),
            model_name=cfg["new_name"],
        )
    except Exception as e:
        print(f"ERROR:train failed: {e}")
        return 5

    print(f"INFO:n_train={len(train_imgs)}")
    print(f"RESULT:{new_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
