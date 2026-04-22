"""Headless self-test for the napari 0x1C access-violation patch.

Reproduces the add-labels → del-layer → add-labels loop that our BarcodeSeg /
BiosensorSeg widgets hit. If the fix is live, this should run for N cycles
without the process dying.

Run with:
    D:/Softwares/Anaconda/Anaconda3/envs/BC-FLIM/python.exe walkthrough/test_0x1c_patch.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np


N_CYCLES = 30
LAYER_SHAPE = (1024, 1024)


def _import_plugin_with_patch():
    """Import flim_s_gen — triggers _install_vispy_0x1c_patch() at module load."""
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if here not in sys.path:
        sys.path.insert(0, here)
    import flim_s_gen  # noqa: F401


def main() -> int:
    _import_plugin_with_patch()

    from napari._qt.qt_viewer import QtViewer
    patched = getattr(QtViewer, '_bcflim_0x1c_patched', False)
    print(f'[check] QtViewer._bcflim_0x1c_patched = {patched}')
    if not patched:
        print('[FAIL] patch did not install')
        return 2

    import napari
    print(f'[check] napari {napari.__version__}')

    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    # Create a real viewer (shows a window — we minimise it).
    v = napari.Viewer(show=True)
    try:
        v.window._qt_window.setWindowState(0x1)  # minimised
    except Exception:
        pass

    rng = np.random.default_rng(0)

    def _pump(ms: int = 100):
        # Pump the Qt event loop so vispy actually does its paint/cleanup.
        t0 = time.time()
        while (time.time() - t0) * 1000 < ms:
            app.processEvents()

    try:
        for i in range(N_CYCLES):
            img = rng.integers(0, 5000, size=LAYER_SHAPE, dtype=np.uint16)
            mask = rng.integers(0, 50, size=LAYER_SHAPE, dtype=np.int32)
            v.add_image(img, name='seg_img', contrast_limits=(0, 5000))
            v.add_labels(mask, name='mask_test')
            _pump(80)
            # The exact pattern that used to crash: del old, add new
            if 'seg_img' in v.layers:
                del v.layers['seg_img']
            if 'mask_test' in v.layers:
                del v.layers['mask_test']
            _pump(80)
            if (i + 1) % 5 == 0:
                print(f'  cycle {i+1}/{N_CYCLES} ok, layers now: {list(v.layers)}')
        print(f'[PASS] survived {N_CYCLES} add/remove cycles, no access violation.')
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
