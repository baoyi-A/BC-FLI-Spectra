"""End-to-end self-test: subprocess cellpose infer → del+add layers.

Reproduces the *exact* widget flow that was crashing:
  add_image(raw) → add_labels(mask from subprocess) → del both → repeat.

Uses the builtin Cellpose 'cyto2' model on a tiny 256x256 synthetic image so
the subprocess finishes in a few seconds even on CPU.

Run with:
    D:/Softwares/Anaconda/Anaconda3/envs/BC-FLIM/python.exe \
        walkthrough/test_0x1c_with_subproc.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


N_CYCLES = 6
IMG_SHAPE = (256, 256)


def _import_plugin_with_patch():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if here not in sys.path:
        sys.path.insert(0, here)
    import flim_s_gen  # noqa: F401
    from flim_s_gen import _widget as _w  # noqa: F401
    return _w


def main() -> int:
    _w = _import_plugin_with_patch()

    from napari._qt.qt_viewer import QtViewer
    if not getattr(QtViewer, '_bcflim_0x1c_patched', False):
        print('[FAIL] patch not installed')
        return 2
    import napari
    print(f'[check] napari {napari.__version__} + patch active')

    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    v = napari.Viewer(show=True)
    try:
        v.window._qt_window.setWindowState(0x1)
    except Exception:
        pass

    rng = np.random.default_rng(0)

    def _pump(ms=120):
        t0 = time.time()
        while (time.time() - t0) * 1000 < ms:
            app.processEvents()

    tmp_dir = Path(os.environ.get('TEMP') or '.').resolve() / 'bcflim_0x1c_test'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i in range(N_CYCLES):
            img = (rng.random(IMG_SHAPE, dtype=np.float32) * 30 +
                   np.clip(rng.random(IMG_SHAPE, dtype=np.float32) * 255, 0, 255))
            out_path = tmp_dir / f'mask_{i}.npy'
            t0 = time.time()
            mask = _w._run_infer_subprocess(
                img=img, base_name='cyto2',
                diameter=30.0, channels=[0, 0], use_gpu=False,
                extra_roots=[], out_path=out_path,
            )
            dt = time.time() - t0
            print(f'  cycle {i+1}/{N_CYCLES}: subproc {dt:.1f}s, mask.max={int(mask.max())}')
            v.add_image(img.astype(np.float32), name='seg_img')
            v.add_labels(mask.astype(np.int32), name='mask_test')
            _pump(150)
            if 'seg_img' in v.layers:
                del v.layers['seg_img']
            if 'mask_test' in v.layers:
                del v.layers['mask_test']
            _pump(150)
        print(f'[PASS] survived {N_CYCLES} subproc + del+add cycles.')
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        try:
            v.close()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
