
import glob
import pathlib
import re
import time
import traceback
from datetime import datetime
from itertools import combinations

from matplotlib.lines import Line2D
from napari.layers import Labels
from napari.qt.threading import thread_worker
from napari.utils import Colormap, notifications
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from magicgui import widgets
import tifffile as tiff
from typing import TYPE_CHECKING, List
from magicgui.widgets import FloatSpinBox, FloatSlider, Label
from skimage.measure import regionprops, label
from ptufile import PtuFile
from napari.layers import Image as NapariImage
from qtpy.QtWidgets import QFileDialog

import cv2
import tifffile
from skimage.morphology import disk, erosion
from magicgui.widgets import Container,  ComboBox, SpinBox
from napari.utils.notifications import show_info, show_warning
from scipy.ndimage import center_of_mass
from typing import Union
from pathlib import Path

import tkinter as tk
from tkinter import messagebox
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
import os
import pandas as pd
import numpy as np
import napari
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from qtpy.QtWidgets import QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from qtpy.QtWidgets import QMessageBox


def _install_vispy_0x1c_patch():
    """Backport napari PR #8122 to napari 0.4.x / 0.5.x on Windows+NVIDIA.

    Without this patch, removing a Labels/Image layer on Windows with NVIDIA
    drivers can leave vispy's OpenGL context in a half-freed state. The NEXT
    paint (e.g. when the next layer is added, or when the user pans/zooms)
    then crashes the process with `OSError: exception: access violation
    reading 0x0000000000000001C` at `WARNING: Error drawing visual <Image>`.

    The upstream fix in napari 0.6.3 is two lines inserted after layer.close():

        gc.collect()
        self._scene_canvas.context.finish()

    See https://github.com/napari/napari/pull/8122. We replicate it here by
    wrapping QtViewer._remove_layer. Safe to install even on fixed napari
    versions — the extra gc.collect()/glFinish is cheap.
    """
    try:
        from napari._qt.qt_viewer import QtViewer  # noqa: E402
    except Exception as e:  # pragma: no cover - napari missing
        print(f'[vispy-patch] napari QtViewer import failed: {e}')
        return
    if getattr(QtViewer, '_bcflim_0x1c_patched', False):
        return
    import functools as _ft
    _orig_remove = QtViewer._remove_layer

    @_ft.wraps(_orig_remove)
    def _remove_layer(self, event):
        import gc as _gc
        _orig_remove(self, event)
        try:
            _gc.collect()
        except Exception:
            pass
        # In 0.4.x the vispy canvas is `self.canvas`; in 0.6.x it's
        # `self._scene_canvas`. Try both.
        scene = getattr(self, '_scene_canvas', None) or getattr(self, 'canvas', None)
        if scene is None:
            return
        try:
            scene.context.finish()
        except Exception:
            try:
                from vispy.gloo import gl as _gl  # noqa: E402
                _gl.glFinish()
            except Exception:
                pass

    # napari's event system uses callback.__name__ to validate the bound
    # method, so @functools.wraps above (which copies __name__) is critical.
    QtViewer._remove_layer = _remove_layer
    QtViewer._bcflim_0x1c_patched = True
    print('[vispy-patch] installed napari PR#8122 backport (0x1C fix).')


_install_vispy_0x1c_patch()


def _tt(widget, text: str) -> None:
    """Attach a Qt tooltip to a magicgui control or a raw Qt widget.

    Lets us write ``_tt(self.tau_min, "...")`` in one line instead of the
    verbose ``try: self.tau_min.native.setToolTip(...) except: pass``. The
    same call works on ``Container``-derived magicgui widgets (which expose
    ``.native``) and on bare Qt widgets that have ``setToolTip`` directly.
    """
    if widget is None:
        return
    target = getattr(widget, 'native', widget)
    try:
        target.setToolTip(text)
    except Exception:
        pass


def _load_ptu(ptu_path: Union[str, Path], frame: Union[int, str] = -1) -> np.ndarray:
    """
    Load and decode a PTU file. Returns numpy array; may be high-dimensional.
      - If frame == -1: returns full decoded stack
      - Else: returns only that frame
    """
    with PtuFile(str(ptu_path)) as ptu:
        data = ptu.decode_image(frame=frame)
    return np.asarray(data)

# =========================================================================
# Widget workflow order + Next-button helper
# Each entry: (ClassName, display name shown in napari dock)
# =========================================================================
_WIDGET_ORDER = [
    ('PTUReader',        'PTU Reader'),
    ('BarcodeSeg',       'Barcode Seg (Cellpose)'),
    ('Calculate_FLIM_S', 'Calculate FLIM-S'),
    ('SeededKMeans',     'Seeded K-Means'),
    ('BiosensorSeg',     'Biosensor Seg (Cellpose)'),
    ('BPTracker',        'B&P Tracker'),
    ('Trackrevise',      'NaCha'),
]

# Per-viewer workflow start time. Used by _go_next_widget to show a
# celebration dialog + total elapsed time when the user finishes the last step.
_WALKTHROUGH_START: dict = {}


def _note_workflow_start(viewer):
    vid = id(viewer)
    if vid not in _WALKTHROUGH_START:
        _WALKTHROUGH_START[vid] = datetime.now()


def _format_elapsed(td) -> str:
    secs = int(td.total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h {m}m {s}s'
    if m:
        return f'{m}m {s}s'
    return f'{s}s'


def _show_walkthrough_celebration(viewer):
    """Show a completion dialog with total elapsed time."""
    from qtpy.QtWidgets import QMessageBox
    vid = id(viewer)
    start = _WALKTHROUGH_START.get(vid)
    if start is None:
        elapsed_text = 'unknown (PTU Reader was not opened via this session)'
    else:
        elapsed_text = _format_elapsed(datetime.now() - start)

    steps = ' -> '.join(display for _, display in _WIDGET_ORDER)
    msg = (
        f'<h2 style="color:#2E7D32;">🎉 Walkthrough complete!</h2>'
        f'<p>You went through the full BC-FLIM-Spectra pipeline:</p>'
        f'<p style="font-family:Consolas; color:#555555;">{steps}</p>'
        f'<p><b>Total elapsed time:</b> {elapsed_text}</p>'
        f'<p>Outputs for this sample are in the <b>Sample Folder</b> you picked '
        f'at the start. Enjoy your clustered cells!</p>'
    )
    box = QMessageBox()
    box.setWindowTitle('Walkthrough finished')
    box.setTextFormat(1)  # Qt.RichText
    box.setText(msg)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()
    _WALKTHROUGH_START.pop(vid, None)  # reset for next run


def _find_dock_parent(qwidget):
    """Walk up the parent chain until we find the QDockWidget hosting this widget."""
    from qtpy.QtWidgets import QDockWidget
    p = qwidget
    while p is not None:
        if isinstance(p, QDockWidget):
            return p
        p = p.parent()
    return None


def _clear_all_viewer_layers(viewer):
    """Remove every layer from a napari viewer. Used when advancing between
    workflow widgets so each step starts from a clean canvas — downstream
    widgets always reload what they need from disk.

    Uses the public napari LayerList API (clear / remove) and lets Qt finish
    any pending paint cycle before / after to avoid vispy GL access violations
    triggered by tearing down layers mid-draw.
    """
    try:
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()
    except Exception:
        pass
    try:
        # napari >= 0.4.16: LayerList.clear() is available and handles the
        # GL teardown properly.
        viewer.layers.clear()
    except Exception:
        # Fallback: remove one by one via the public API.
        try:
            for layer in list(viewer.layers):
                try:
                    viewer.layers.remove(layer)
                except Exception as e:
                    print(f'[clear layers] {getattr(layer, "name", "?")}: {e}')
        except Exception as e:
            print(f'[clear layers] iteration failed: {e}')
    try:
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()
    except Exception:
        pass


def _run_infer_subprocess(
    *, img, base_name, diameter, channels, use_gpu, extra_roots, out_path,
):
    """Run Cellpose inference in a child Python. Returns a uint16 mask ndarray.

    Shares the isolation logic with fine-tune: importing torch in the child
    means the napari main process never touches CUDA/GL-reactive code, so
    vispy's rendering does not crash later in the session.
    """
    import pickle, subprocess, sys, tempfile, os as _os, numpy as _np

    default_roots: list[str] = []
    try:
        default_roots = [str(d) for d in sorted(_BARCODE_MODEL_ROOT.glob('_cellpose_finetune_*'))]
    except Exception:
        pass

    out_path = Path(out_path)
    cfg = dict(
        op='infer',
        img=img,
        base_name=base_name,
        diameter=float(diameter) if diameter and diameter > 0 else 0,
        channels=list(channels),
        use_gpu=bool(use_gpu),
        cellpose_src=str(_CELLPOSE_SRC_PATH) if _CELLPOSE_SRC_PATH.exists() else '',
        extra_roots=[str(r) for r in (extra_roots or [])],
        default_finetune_roots=default_roots,
        out_path=str(out_path),
    )
    fd, cfg_path = tempfile.mkstemp(prefix='infer_cfg_', suffix='.pkl')
    _os.close(fd)
    with open(cfg_path, 'wb') as f:
        pickle.dump(cfg, f)

    runner = Path(__file__).resolve().parent / '_finetune_runner.py'
    proc = subprocess.run(
        [sys.executable, '-u', str(runner), cfg_path],
        capture_output=True, text=True,
    )
    try:
        _os.remove(cfg_path)
    except Exception:
        pass

    out = (proc.stdout or '') + '\n' + (proc.stderr or '')
    for line in out.splitlines()[::-1]:
        s = line.strip()
        if s.startswith('RESULT:'):
            # format: RESULT:<out_path>|<ncells>
            body = s[len('RESULT:'):]
            saved = Path(body.split('|', 1)[0].strip())
            return _np.load(str(saved))
        if s.startswith('ERROR:'):
            raise RuntimeError(s[len('ERROR:'):].strip())
    raise RuntimeError(f'infer subprocess ended without RESULT/ERROR '
                       f'(exit={proc.returncode}); tail: {out[-500:]}')


def _run_finetune_subprocess(
    *, img=None, mask=None, imgs=None, masks=None,
    base_name, new_name, save_dir, n_epochs, channels,
    use_gpu, extra_roots,
) -> "Path":
    """Run Cellpose fine-tune in a child Python process.

    Keeping training out of napari's main process is the only reliable way
    to avoid PyTorch leaving dangling GL/CUDA state that later crashes vispy
    with `access violation reading 0x1C` the next time an Image is drawn.

    Pass either ``img`` + ``mask`` (single-image form) or ``imgs`` + ``masks``
    (lists of equal length for multi-folder fine-tuning). If both forms are
    supplied, the list form wins.

    Returns the Path to the trained model file.
    """
    import pickle, subprocess, sys, tempfile, os as _os

    default_roots: list[str] = []
    try:
        default_roots = [str(d) for d in sorted(_BARCODE_MODEL_ROOT.glob('_cellpose_finetune_*'))]
    except Exception:
        pass

    cfg = dict(
        base_name=base_name, new_name=new_name,
        save_dir=str(save_dir), n_epochs=int(n_epochs),
        channels=list(channels), use_gpu=bool(use_gpu),
        cellpose_src=str(_CELLPOSE_SRC_PATH) if _CELLPOSE_SRC_PATH.exists() else '',
        extra_roots=[str(r) for r in (extra_roots or [])],
        default_finetune_roots=default_roots,
    )
    if imgs is not None and masks is not None:
        if len(imgs) != len(masks):
            raise ValueError(f'imgs/masks length mismatch: {len(imgs)} vs {len(masks)}')
        if len(imgs) == 0:
            raise ValueError('imgs/masks lists are empty')
        cfg['imgs'] = list(imgs)
        cfg['masks'] = list(masks)
    else:
        if img is None or mask is None:
            raise ValueError('either (img, mask) or (imgs, masks) must be provided')
        cfg['img'] = img
        cfg['mask'] = mask
    fd, cfg_path = tempfile.mkstemp(prefix='ft_cfg_', suffix='.pkl')
    _os.close(fd)
    with open(cfg_path, 'wb') as f:
        pickle.dump(cfg, f)

    runner = Path(__file__).resolve().parent / '_finetune_runner.py'
    proc = subprocess.run(
        [sys.executable, '-u', str(runner), cfg_path],
        capture_output=True, text=True,
    )
    try:
        _os.remove(cfg_path)
    except Exception:
        pass

    # Parse RESULT / ERROR line from child stdout
    out = (proc.stdout or '') + '\n' + (proc.stderr or '')
    for line in out.splitlines()[::-1]:
        s = line.strip()
        if s.startswith('RESULT:'):
            return Path(s[len('RESULT:'):].strip())
        if s.startswith('ERROR:'):
            raise RuntimeError(s[len('ERROR:'):].strip())
    raise RuntimeError(f'fine-tune subprocess ended without RESULT/ERROR '
                       f'(exit={proc.returncode}); stdout tail: {out[-500:]}')


# =========================================================================
# Multi-folder fine-tune dialog
# =========================================================================
# Detectors: (sample_dir: Path) -> (image_path | None, mask_path | None, status)
# - image_path / mask_path: resolved candidate (may not exist)
# - status: short human-readable note for the table
# All detectors are cheap — they just probe filesystem paths, not load data.

def _detect_finetune_pair_barcode_n(sample_dir: Path):
    intensity = sample_dir / 'intensity'
    img = None
    if intensity.is_dir():
        tifs = sorted(intensity.glob('*_sum.tif'))
        if tifs:
            img = tifs[0]
    if img is None:
        return None, None, '✗ no intensity/*_sum.tif'
    mask = img.parent.parent / f'{img.stem}_seg_n.npy'
    if mask.exists():
        return img, mask, '✓ n mask found'
    return img, None, '⚠ missing *_seg_n.npy'


def _detect_finetune_pair_barcode_p(sample_dir: Path):
    intensity = sample_dir / 'intensity'
    img = None
    if intensity.is_dir():
        tifs = sorted(intensity.glob('*_sum.tif'))
        if tifs:
            img = tifs[0]
    if img is None:
        return None, None, '✗ no intensity/*_sum.tif'
    mask = img.parent.parent / f'{img.stem}_seg_p.npy'
    if mask.exists():
        return img, mask, '✓ p mask found'
    return img, None, '⚠ missing *_seg_p.npy'


def _detect_finetune_pair_biosensor(sample_dir: Path):
    img = sample_dir / 'seg_image.tif'
    if not img.is_file():
        return None, None, '✗ no seg_image.tif'
    mask = sample_dir / 'seg_image_seg.npy'
    if mask.exists():
        return img, mask, '✓ seg_image_seg.npy'
    return img, None, '⚠ missing seg_image_seg.npy'


_FINETUNE_DETECTORS = {
    'n':  (_detect_finetune_pair_barcode_n,  'Barcode — N mask'),
    'p':  (_detect_finetune_pair_barcode_p,  'Barcode — P mask'),
    'bs': (_detect_finetune_pair_biosensor, 'Biosensor — cell mask'),
}


def _load_mask_npy_any(path: Path) -> "np.ndarray":
    """Load a mask .npy that may be a plain ndarray or a dict with 'masks'."""
    import numpy as _np
    raw = _np.load(str(path), allow_pickle=True)
    if isinstance(raw, _np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict) and 'masks' in raw:
        raw = raw['masks']
    arr = _np.asarray(raw)
    if arr.ndim > 2:
        arr = _np.squeeze(arr)
    return arr.astype(_np.int32)


def _open_multi_finetune_dialog(parent_widget, target: str, base_name: str,
                                 default_new_name: str, default_epochs: int,
                                 use_gpu: bool, save_root: Path,
                                 on_done, on_error):
    """Show a modal dialog to pick multiple sample folders and fine-tune.

    Parameters
    ----------
    parent_widget : QWidget-like — used as the Qt parent for the dialog.
    target : 'n' | 'p' | 'bs' — which file-pair detector to run per folder.
    base_name : str — Cellpose base model name (builtin or custom).
    default_new_name : str — prefilled name for the trained-model output.
    default_epochs : int — prefilled epoch count.
    use_gpu : bool — prefilled Use-GPU checkbox.
    save_root : Path — where the trained model folder is created
                       (``save_root / <new_name>`` is then created).
    on_done(new_name, new_path, n_samples) : callback on successful training.
    on_error(exc) : callback on failure.
    """
    from qtpy.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
        QTableWidgetItem, QLabel, QLineEdit, QSpinBox, QCheckBox, QFileDialog,
        QProgressBar, QMessageBox, QHeaderView,
    )
    from qtpy.QtCore import Qt

    detector_fn, mask_label = _FINETUNE_DETECTORS[target]

    parent = getattr(parent_widget, 'native', parent_widget)

    dlg = QDialog(parent)
    dlg.setWindowTitle(f'Multi-folder fine-tune — {mask_label}')
    dlg.resize(900, 520)

    root = QVBoxLayout(dlg)

    hdr = QLabel(
        f'<b>Fine-tune <code>{base_name}</code> on multiple sample folders.</b><br>'
        f'Add each folder; the dialog auto-detects the image + mask pair '
        f'<i>({mask_label})</i>. Rows with ⚠ or ✗ will be skipped.'
    )
    hdr.setWordWrap(True)
    root.addWidget(hdr)

    table = QTableWidget(0, 4)
    table.setHorizontalHeaderLabels(['Folder', 'Image', 'Mask', 'Status'])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    root.addWidget(table)

    def _add_folder(dir_str: str):
        if not dir_str:
            return
        sd = Path(dir_str)
        if not sd.is_dir():
            return
        # Skip if already in the table.
        for r in range(table.rowCount()):
            if table.item(r, 0) and table.item(r, 0).text() == str(sd):
                return
        img, mask, status = detector_fn(sd)
        r = table.rowCount()
        table.insertRow(r)
        table.setItem(r, 0, QTableWidgetItem(str(sd)))
        table.setItem(r, 1, QTableWidgetItem(str(img) if img else '(none)'))
        table.setItem(r, 2, QTableWidgetItem(str(mask) if mask else '(none)'))
        item = QTableWidgetItem(status)
        if status.startswith('✓'):
            item.setForeground(Qt.darkGreen)
        elif status.startswith('⚠'):
            item.setForeground(Qt.darkYellow)
        else:
            item.setForeground(Qt.red)
        table.setItem(r, 3, item)

    def _pick_and_add():
        d = QFileDialog.getExistingDirectory(dlg, 'Add sample folder')
        _add_folder(d)

    def _remove_selected():
        rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
        for r in rows:
            table.removeRow(r)

    row_btns = QHBoxLayout()
    b_add = QPushButton('+ Add folder...')
    b_add.clicked.connect(_pick_and_add)
    row_btns.addWidget(b_add)
    b_rm = QPushButton('– Remove selected')
    b_rm.clicked.connect(_remove_selected)
    row_btns.addWidget(b_rm)
    row_btns.addStretch(1)
    root.addLayout(row_btns)

    # Training config row
    cfg_row = QHBoxLayout()
    cfg_row.addWidget(QLabel('New model name:'))
    name_edit = QLineEdit(default_new_name)
    cfg_row.addWidget(name_edit, 2)
    cfg_row.addWidget(QLabel('Epochs:'))
    ep_spin = QSpinBox()
    ep_spin.setRange(1, 5000)
    ep_spin.setValue(int(default_epochs))
    cfg_row.addWidget(ep_spin)
    gpu_chk = QCheckBox('Use GPU')
    gpu_chk.setChecked(bool(use_gpu))
    cfg_row.addWidget(gpu_chk)
    root.addLayout(cfg_row)

    # Status + progress
    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    root.addWidget(progress)
    status_lbl = QLabel('Ready. Add at least one folder with a valid ✓ pair.')
    status_lbl.setWordWrap(True)
    root.addWidget(status_lbl)

    # OK / Cancel
    btn_row = QHBoxLayout()
    btn_row.addStretch(1)
    start_btn = QPushButton('Start training')
    cancel_btn = QPushButton('Close')
    btn_row.addWidget(start_btn)
    btn_row.addWidget(cancel_btn)
    root.addLayout(btn_row)

    cancel_btn.clicked.connect(dlg.reject)

    def _on_start():
        # Gather valid rows.
        pairs: list[tuple[Path, Path]] = []
        for r in range(table.rowCount()):
            status = table.item(r, 3).text() if table.item(r, 3) else ''
            if not status.startswith('✓'):
                continue
            img_p = Path(table.item(r, 1).text())
            mask_p = Path(table.item(r, 2).text())
            pairs.append((img_p, mask_p))
        if not pairs:
            QMessageBox.warning(dlg, 'Nothing to train',
                                'No folders with a complete ✓ image + mask pair.')
            return
        new_name = name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(dlg, 'Name required', 'Enter a name for the trained model.')
            return
        save_dir = Path(save_root) / new_name
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(dlg, 'Save path', f'Cannot create {save_dir}: {e}')
            return

        start_btn.setEnabled(False)
        b_add.setEnabled(False)
        b_rm.setEnabled(False)
        status_lbl.setText(f'Loading {len(pairs)} image/mask pairs...')
        progress.setValue(5)

        try:
            imgs = []
            masks = []
            for i, (img_p, mask_p) in enumerate(pairs):
                img = tifffile.imread(str(img_p))
                if img.ndim > 2:
                    img = np.squeeze(img)
                imgs.append(np.asarray(img, dtype=np.float32))
                masks.append(_load_mask_npy_any(mask_p))
                progress.setValue(5 + int(15 * (i + 1) / max(1, len(pairs))))

            status_lbl.setText(
                f'Launching Cellpose subprocess on {len(pairs)} pairs '
                f'for {ep_spin.value()} epochs...')
            progress.setValue(25)

            # Channels: barcode single-channel (0,0). Biosensor is single-channel
            # too in _prep_biosensor_seg_input output, so [0,0] is correct for
            # all three targets.
            channels = [0, 0]

            new_path = _run_finetune_subprocess(
                imgs=imgs, masks=masks,
                base_name=base_name, new_name=new_name,
                save_dir=save_dir, n_epochs=int(ep_spin.value()),
                channels=channels, use_gpu=bool(gpu_chk.isChecked()),
                extra_roots=[str(save_root)],
            )
            progress.setValue(100)
            status_lbl.setText(f'Done: {new_path}')
            on_done(new_name, new_path, len(pairs))
            dlg.accept()
        except Exception as e:
            progress.setValue(0)
            status_lbl.setText(f'ERROR: {e}')
            try:
                on_error(e)
            except Exception:
                pass
            start_btn.setEnabled(True)
            b_add.setEnabled(True)
            b_rm.setEnabled(True)

    start_btn.clicked.connect(_on_start)

    dlg.exec_()


def _release_gpu_caches():
    """Best-effort: drop any cached PyTorch CUDA / Python GC state. Called before
    we tear down napari layers across a Next transition — fine-tuning Cellpose
    can leave vispy's GL shared-context in a fragile state; dropping torch
    caches + forcing a GC makes the subsequent layer teardown less crash-prone.
    """
    import gc
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f'[release gpu] torch: {e}')
    try:
        gc.collect()
    except Exception:
        pass


def _go_next_widget(container, viewer):
    """Close the current widget's dock and open the next workflow widget.

    Fine-tuning Cellpose between widgets has been observed to leave vispy/GL
    in a state where immediately clearing layers + adding new ones triggers
    `access violation reading 0x1C` during the paint loop. We work around
    that by:
      1. releasing torch CUDA caches + running a GC,
      2. deferring the whole teardown + new-widget creation via QTimer so
         the current paint event finishes first.
    """
    import sys as _sys
    current_name = type(container).__name__
    names = [n for n, _ in _WIDGET_ORDER]
    if current_name not in names:
        show_warning(f"Unknown widget '{current_name}' in workflow order.")
        return
    idx = names.index(current_name)
    if idx + 1 >= len(_WIDGET_ORDER):
        try:
            _show_walkthrough_celebration(viewer)
        except Exception as e:
            print(f'[walkthrough celebration] {e}')
            show_info('Workflow complete.')
        return
    next_cls_name, next_display = _WIDGET_ORDER[idx + 1]
    mod = _sys.modules[__name__]
    next_cls = getattr(mod, next_cls_name, None)
    if next_cls is None:
        show_warning(f"Next widget class '{next_cls_name}' not found.")
        return

    dock = _find_dock_parent(container.native)

    def _do_transition():
        _release_gpu_caches()
        _clear_all_viewer_layers(viewer)
        try:
            new_widget = next_cls(viewer)
            viewer.window.add_dock_widget(new_widget, area='right', name=next_display)
        except Exception as e:
            show_warning(f"Failed to open next widget '{next_display}': {e}")
            return
        if dock is not None:
            try:
                dock.close()
            except Exception:
                pass

    try:
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, _do_transition)
    except Exception:
        # Fall back to synchronous if Qt timer is unavailable.
        _do_transition()


_NEXT_BTN_STYLE = (
    "QPushButton {"
    "  background-color: #1E88E5;"
    "  color: white;"
    "  font-weight: bold;"
    "  padding: 6px 12px;"
    "  border-radius: 4px;"
    "  font-family: Calibri;"
    "} "
    "QPushButton:hover { background-color: #1565C0; } "
    "QPushButton:disabled { background-color: #9E9E9E; color: #EEEEEE; }"
)

_PROCESS_BTN_STYLE = (
    "QPushButton {"
    "  background-color: #43A047;"
    "  color: white;"
    "  font-weight: bold;"
    "  padding: 6px 12px;"
    "  border-radius: 4px;"
    "  font-family: Calibri;"
    "} "
    "QPushButton:hover { background-color: #2E7D32; } "
    "QPushButton:disabled { background-color: #9E9E9E; color: #EEEEEE; }"
)


def _style_process_button(btn):
    """Apply the green process-button style to a magicgui PushButton."""
    try:
        btn.native.setStyleSheet(_PROCESS_BTN_STYLE)
    except Exception:
        pass


_PLUGIN_RESOURCES = Path(__file__).resolve().parent / 'resources'
_CELLPOSE_LOGO_PATH = _PLUGIN_RESOURCES / 'cellpose_logo.png'
_TRACK_ANYTHING_LOGO_PATH = _PLUGIN_RESOURCES / 'track_anything_logo.png'


def _tighten_container(container, spacing: int = 2, margins=(4, 4, 4, 4)):
    """Reduce the default row spacing / margins on a magicgui Container so
    dense widgets fit without squeezing the napari canvas."""
    try:
        lay = container.native.layout()
        if lay is not None:
            lay.setSpacing(spacing)
            lay.setContentsMargins(*margins)
    except Exception:
        pass


def _append_section_divider(container, text: str):
    """Append a small styled header Label to group the rows that follow.
    Shared by all workflow widgets for visual consistency."""
    from magicgui.widgets import Label as _Label
    lbl = _Label(value=text)
    try:
        lbl.native.setStyleSheet(
            'QLabel {'
            '  color: #CE93D8;'
            '  font-weight: bold;'
            '  font-family: Calibri;'
            '  padding: 3px 4px 1px 4px;'
            '  border-bottom: 1px solid #7B1FA2;'
            '}'
        )
    except Exception:
        pass
    container.append(lbl)


def _add_logo_header(container, title, subtitle, logo_path, logo_size=40):
    """Prepend a compact "logo + title + subtitle" header to a magicgui Container."""
    from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel
    from qtpy.QtGui import QPixmap
    from qtpy.QtCore import Qt

    header = QWidget()
    lay = QHBoxLayout()
    lay.setContentsMargins(4, 2, 4, 2)
    lay.setSpacing(8)
    header.setLayout(lay)

    if logo_path and Path(logo_path).exists():
        logo_lbl = QLabel()
        pix = QPixmap(str(logo_path))
        if not pix.isNull():
            pix = pix.scaledToHeight(
                logo_size, Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl.setPixmap(pix)
        lay.addWidget(logo_lbl)

    text_lbl = QLabel(
        f'<div style="font-weight:bold;">{title}</div>'
        f'<div style="color:#888888; font-size:10px;">{subtitle}</div>'
    )
    text_lbl.setTextFormat(Qt.TextFormat.RichText)
    lay.addWidget(text_lbl)
    lay.addStretch(1)

    try:
        container.native.layout().insertWidget(0, header)
    except Exception:
        try:
            from magicgui.widgets import Label as _Label
            container.append(_Label(value=f'🔬  {title}  ({subtitle})'))
        except Exception:
            pass


def _add_cellpose_header(container, title='Cellpose Segmentation', logo_size=40):
    """Prepend a compact "logo + title" header to a magicgui Container widget.

    Uses Qt directly because magicgui Labels don't render local image paths
    reliably across versions.
    """
    from qtpy.QtWidgets import QWidget, QHBoxLayout, QLabel
    from qtpy.QtGui import QPixmap
    from qtpy.QtCore import Qt

    header = QWidget()
    lay = QHBoxLayout()
    lay.setContentsMargins(4, 2, 4, 2)
    lay.setSpacing(8)
    header.setLayout(lay)

    if _CELLPOSE_LOGO_PATH.exists():
        logo_lbl = QLabel()
        pix = QPixmap(str(_CELLPOSE_LOGO_PATH))
        if not pix.isNull():
            pix = pix.scaled(
                logo_size, logo_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_lbl.setPixmap(pix)
        lay.addWidget(logo_lbl)

    text_lbl = QLabel(
        f'<div style="font-weight:bold;">{title}</div>'
        f'<div style="color:#888888; font-size:10px;">Powered by Cellpose</div>'
    )
    text_lbl.setTextFormat(Qt.TextFormat.RichText)
    lay.addWidget(text_lbl)
    lay.addStretch(1)

    # Container.native is a QWidget with a QVBoxLayout; insert our header at the top.
    try:
        container.native.layout().insertWidget(0, header)
    except Exception:
        # Fallback: just append via magicgui (will appear at bottom)
        try:
            from magicgui.widgets import Label as _Label
            container.append(_Label(value=f'🔬  {title}  (Powered by Cellpose)'))
        except Exception:
            pass


# Per-widget "before you click Next" checklist. Surfaced on the Next
# button's tooltip so users know (a) where they're going and (b) whether
# they're ready to advance.
_NEXT_STEP_CHECKLIST = {
    'PTUReader': (
        "Before clicking Next, make sure every .ptu was decoded into "
        "intensity/*_sum.tif + flim_stack/*_ch*.tif, and the FastFLIM "
        "RGB overlay looks reasonable."
    ),
    'BarcodeSeg': (
        "Before clicking Next, make sure the N and P masks cover the "
        "cells you want and *_seg_n.npy / *_seg_p.npy are saved "
        "(Auto-Segment saves them; click Save masks after manual edits)."
    ),
    'Calculate_FLIM_S': (
        "Before clicking Next, make sure Process finished and FLIM-S.xlsx "
        "was written to the Base folder."
    ),
    'SeededKMeans': (
        "Before clicking Next, make sure seeds are placed on every class, "
        "Run K-Means produced a sensible clustering, and you clicked "
        "Save Results (writes clustered.xlsx)."
    ),
    'BiosensorSeg': (
        "Before clicking Next, make sure seg_image.tif + seg_image_seg.npy "
        "are saved, and the mask looks right when overlaid on the "
        "barcode classification layer."
    ),
    'BPTracker': (
        "Before clicking Next, save per-frame tracking masks if you ran "
        "tracking. SKIP this step (just click Next) if your sample has a "
        "static mask — NaCha will reuse the single seg."
    ),
}
_NEXT_STEP_NAME_SUFFIX = {
    'PTUReader':        'Barcode Seg',
    'BarcodeSeg':       'Calculate FLIM-S',
    'Calculate_FLIM_S': 'Seeded K-Means',
    'SeededKMeans':     'Biosensor Seg',
    'BiosensorSeg':     'B&P Tracker',
    'BPTracker':        'NaCha (final step)',
}


def _add_next_button(container, viewer, pre_next=None):
    """Append a 'Next ▶' button to a magicgui Container widget.

    `pre_next` (optional callable) runs before advancing — use it to clean up
    layers or persist state. The button's hover tooltip names the next
    widget and lists the "did you finish X?" checklist for the current
    step, driven by ``_NEXT_STEP_CHECKLIST`` + ``_NEXT_STEP_NAME_SUFFIX``.
    """
    from magicgui.widgets import PushButton as _PushButton
    btn = _PushButton(text='Next \u25B6')
    try:
        btn.native.setStyleSheet(_NEXT_BTN_STYLE)
    except Exception:
        pass

    cls_name = type(container).__name__
    next_name = _NEXT_STEP_NAME_SUFFIX.get(cls_name, '(no further step)')
    checklist = _NEXT_STEP_CHECKLIST.get(cls_name, '')
    _tt(btn,
        f'Advance to: {next_name}. '
        + (checklist + ' ' if checklist else '')
        + "Clicking Next also tears down this widget's layers to keep "
          "the session clean.")

    def _on_click():
        if pre_next is not None:
            try:
                pre_next()
            except Exception as e:
                print(f'[Next pre-callback] {e}')
        _go_next_widget(container, viewer)
    btn.changed.connect(_on_click)
    container.append(btn)
    return btn


class PTUReader(Container):
    """
    Napari dock widget for reading PTU files and saving intensity and FLIM stacks.
    """
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__(layout='vertical')
        self.viewer = viewer
        _note_workflow_start(viewer)  # starts the walkthrough timer
        # FastFLIM 参数
        self.tau_min = FloatSpinBox(
            label='Tau min (ns)',
            min=0.0, max=10.0, step=0.1, value=3.1
        )
        self.tau_max = FloatSpinBox(
            label='Tau max (ns)',
            min=0.0, max=10.0, step=0.1, value=5.1
        )
        self.tau_res = FloatSpinBox(
            label='Tau resolution (ns/bin)',
            min=0.001, max=1.0, step=0.001, value=0.098
        )
        # 强度 autocontrast 百分位，比如 99 表示按 99th percentile 做上界
        self.intensity_clip = FloatSpinBox(
            label='Intensity clip (%)',
            min=60.0, max=100.0, step=0.5, value=99.0
        )
        # FastFLIM display (applies live to the RGB render; does not change
        # saved tau values). All re-render from the cached tau_map+intensity,
        # no PTU re-decoding.
        self.brightness_gamma = FloatSpinBox(
            label='Brightness gamma',
            min=0.3, max=1.0, step=0.05, value=0.55,
        )
        self.brightness_floor = FloatSpinBox(
            label='Brightness floor',
            min=0.0, max=0.4, step=0.02, value=0.10,
        )
        self.use_clahe = widgets.CheckBox(text='CLAHE', value=True)
        self.clahe_tile = SpinBox(
            label='CLAHE tile size (px)', min=8, max=512, step=4, value=64,
        )
        # Auto-contrast button: declared before tooltip pass below so the
        # _tt(self.auto_contrast_btn, ...) call has something to attach to.
        self.auto_contrast_btn = PushButton(text='Auto contrast ↻')
        self.auto_contrast_btn.clicked.connect(self._on_auto_contrast)
        # Tooltips for every control — the user hovers to see what each does.
        _tt(self.tau_min,
            'Colormap blue end (short tau, ns). Auto-set from data 12th '
            'percentile on first run.')
        _tt(self.tau_max,
            'Colormap red end (long tau, ns). Auto-set from data 88th '
            'percentile on first run.')
        _tt(self.tau_res,
            'Nanoseconds per time bin in the PTU decay. Leica SP8 / '
            'STELLARIS default: 0.098 ns.')
        _tt(self.intensity_clip,
            'Upper percentile for brightness normalisation. Lower value '
            '= darker overall.')
        _tt(self.brightness_gamma,
            'Lower = brighter shadows. 1.0 = linear. Re-renders live.')
        _tt(self.brightness_floor,
            'Minimum brightness for signal pixels so colours do not '
            'crush to black. Re-renders live.')
        _tt(self.use_clahe,
            'Contrast-Limited Adaptive Histogram Equalisation on the '
            'brightness channel. Dim cells show colour as clearly as '
            'bright ones. Display-only — does not affect segmentation '
            'or per-cell quantification.')
        _tt(self.clahe_tile,
            'Approximately half of your typical cell diameter in pixels. '
            'Larger tile = broader equalisation, less noise amplification.')
        _tt(self.auto_contrast_btn,
            'One-click contrast preset. Cycles through 6 presets and '
            'loops: (100,0) → (95,10) → (90,20) → (85,30) → (80,40) → '
            '(75,50) → back to (100,0). Each preset re-derives tau '
            'min/max as the (lower%, upper%) percentiles over signal '
            'pixels, and sets Intensity clip to the upper percent. The '
            'low end is clipped harder than the high end because the '
            'blue (short-tau) region carries noisier background pixels.')

        # Cached per-FOV FastFLIM data for live re-render. Keyed by layer
        # name (stem + "_FastFLIM"). Each value is a dict with keys
        # {'tau': 2D float32, 'inten': 2D float32}.
        self._fastflim_cache: dict = {}

        # Auto-contrast cycle state. -1 so the first click lands on idx 0.
        self._auto_cycle_idx = -1

        # Widgets
        # self.input_dir = FileEdit(label='PTU Folder', mode='d', value=os.getcwd())
        # self.output_dir = FileEdit(label='Output Folder', mode='d', value=os.getcwd())
        # use J: as default input folder
        self.input_dir = widgets.FileEdit(label='PTU Folder', mode='d', value=r'J:/Mix16-N-P-260306-DCZ-2-1/raw')
        self.output_dir = widgets.FileEdit(label='Output Folder', mode='d', value=r'J:/Mix16-N-P-260306-DCZ-2-1')
        self.frame = SpinBox(label='Frame (-1 for all)', min=-1, max=80, step=1, value=-1)
        self.process_btn = PushButton(text='Process and Save')
        self.process_btn.changed.connect(self._on_process)
        _style_process_button(self.process_btn)
        self.input_dir.changed.connect(self._on_input_dir_changed)

        _tt(self.input_dir,
            'Folder containing .ptu files to decode. Output Folder auto-'
            'updates to the parent when this changes.')
        _tt(self.output_dir,
            'Where decoded intensity / flim_stack TIFs and the FastFLIM '
            'RGB snapshot are written.')
        _tt(self.frame,
            '−1 sums every frame in the PTU (typical). Positive N picks '
            'a single 0-based frame index.')
        _tt(self.process_btn,
            'Decodes every .ptu in the folder, writes intensity and '
            'flim_stack TIFs, computes FastFLIM. Slow for large PTUs '
            '(tens of seconds each).')

        # Progress feedback (PTU decoding can take ~30-60s for 1GB+ files)
        self.progress = widgets.ProgressBar(label='Progress', value=0, min=0, max=100)
        self.status_label = Label(value='Ready')

        # Add to layout with section dividers for scientific-but-playful feel.
        _append_section_divider(self, '— 📁 Input / Output paths —')
        self.append(self.input_dir)
        self.append(self.output_dir)
        self.append(self.frame)

        _append_section_divider(self, '— ⏱ FastFLIM parameters —')
        self.append(self.tau_min)
        self.append(self.tau_max)
        self.append(self.tau_res)
        self.append(self.intensity_clip)

        _append_section_divider(self, '— 🎨 FastFLIM display (live apply) —')
        self.append(self.auto_contrast_btn)
        self.append(self.brightness_gamma)
        self.append(self.brightness_floor)
        self.append(self.use_clahe)
        self.append(self.clahe_tile)

        _append_section_divider(self, '— ▶ Process —')
        self.append(self.process_btn)
        self.append(self.progress)
        self.append(self.status_label)

        # Live-apply: whenever any display control changes, re-render every
        # cached FOV. tau_min / tau_max also trigger a re-render (the worker
        # captures their value once at launch time, but after processing the
        # user can still tweak).
        for ctrl in (self.tau_min, self.tau_max, self.brightness_gamma,
                     self.brightness_floor, self.use_clahe, self.clahe_tile):
            try:
                ctrl.changed.connect(self._redraw_all_fastflim)
            except Exception:
                pass

        _add_next_button(self, viewer, pre_next=self._cleanup_layers)
        _tighten_container(self)

    def _cleanup_layers(self):
        """Remove PTU-Reader layers before moving to the next widget.

        Downstream widgets read TIFs from disk; keeping the heavy layers just
        bloats memory.
        """
        doomed = []
        for layer in list(self.viewer.layers):
            n = layer.name
            if ('_ch' in n and n.rsplit('_ch', 1)[-1][:1].isdigit()):
                doomed.append(n)
            elif n.endswith('_Tau') or n.endswith('_Intensity') or n.endswith('_FastFLIM'):
                doomed.append(n)
            elif n == 'sum' or '_sum' in n:
                doomed.append(n)
        for n in doomed:
            try:
                del self.viewer.layers[n]
            except Exception as e:
                print(f'[PTUReader cleanup] {n}: {e}')
        # Drop the in-memory FastFLIM cache so stale tau/intensity don't
        # get re-rendered if the user comes back and drags a slider.
        self._fastflim_cache = {}
        # viewer.window.add_dock_widget(self, area='right', name='PTU Reader')

    # ---- Intensity-weighted FastFLIM RGB renderer -----------------------
    # Classic FLIM visualisation: hue = tau (blue → green → red), brightness
    # = intensity with gamma + floor so short-tau regions do not crush to
    # black. Optional CLAHE equalises per-cell brightness when barcode
    # expression varies widely. All live — re-renders from cached
    # tau_map + total_int in milliseconds.

    _LEICA_FLIM_CMAP = None  # built lazily on first use

    @classmethod
    def _leica_cmap(cls):
        if cls._LEICA_FLIM_CMAP is None:
            import matplotlib.colors as _mc
            cls._LEICA_FLIM_CMAP = _mc.LinearSegmentedColormap.from_list(
                'leica_flim',
                [(0.00, '#5faaff'),   # sky blue, short tau
                 (0.50, '#00ff00'),   # green,   mid tau
                 (1.00, '#ff3030')],  # red,     long tau
                N=256,
            )
        return cls._LEICA_FLIM_CMAP

    def _render_fastflim_rgb(self, tau, inten):
        """Return a uint8 HxWx3 RGB image for a given (tau, intensity) pair.

        Pulls all display parameters from the current widget values so the
        caller does not have to pass them.
        """
        tau = np.asarray(tau, dtype=np.float32)
        inten = np.asarray(inten, dtype=np.float32)
        tau_lo = float(self.tau_min.value)
        tau_hi = float(self.tau_max.value)
        if tau_hi <= tau_lo:
            tau_hi = tau_lo + 1e-3

        # tau -> colormap
        tau_norm = np.clip((tau - tau_lo) / (tau_hi - tau_lo), 0.0, 1.0)
        tau_norm = np.where(np.isfinite(tau), tau_norm, 0.0)
        rgb = self._leica_cmap()(tau_norm)[..., :3]  # drop alpha

        # intensity -> brightness
        clip_pct = float(self.intensity_clip.value)
        v_max = float(np.percentile(inten, clip_pct)) if inten.size else 1.0
        if v_max <= 0:
            v_max = 1.0
        bright = np.clip(inten / v_max, 0.0, 1.0)

        # optional CLAHE on the brightness channel
        if bool(self.use_clahe.value):
            try:
                from skimage import exposure as _exp
                tile = int(self.clahe_tile.value)
                tile = max(2, tile)
                h, w = bright.shape
                # skimage expects kernel_size (tile per axis); it adapts if
                # the image is not divisible.
                bright = _exp.equalize_adapthist(
                    bright.astype(np.float32),
                    kernel_size=(min(tile, max(2, h // 2)),
                                 min(tile, max(2, w // 2))),
                    clip_limit=0.01,
                ).astype(np.float32)
            except Exception as _e:
                print(f'[FastFLIM CLAHE] skipped: {_e}')

        # gamma + floor lift
        gamma = float(self.brightness_gamma.value)
        floor = float(self.brightness_floor.value)
        bright = floor + (1.0 - floor) * np.power(bright, gamma)
        bright = np.clip(bright, 0.0, 1.0)

        out = rgb * bright[..., None]
        return np.clip(out * 255, 0, 255).astype(np.uint8)

    def _push_fastflim_layer(self, name, tau, inten):
        """Compute RGB, cache (tau, inten), and add / update the napari layer."""
        self._fastflim_cache[name] = {
            'tau': tau.astype(np.float32, copy=False),
            'inten': inten.astype(np.float32, copy=False),
        }
        rgb = self._render_fastflim_rgb(tau, inten)
        try:
            if name in self.viewer.layers:
                del self.viewer.layers[name]
        except Exception:
            pass
        try:
            self.viewer.add_image(rgb, name=name, rgb=True, blending='translucent')
        except Exception as e:
            show_warning(f'FastFLIM layer {name} failed: {e}')

    # (upper_pct, lower_pct) presets for the Auto button. ASYMMETRIC —
    # we clip the low-tau end harder than the high-tau end because the
    # blue end (short tau) carries more background / noise pixels whose
    # tau estimate is unreliable. Empirical rule of thumb:
    # lower_pct ≈ 2 × (100 − upper_pct).
    _AUTO_CONTRAST_PRESETS = (
        (100, 0),    # idx 0 — no clip, full tau range
        (95, 10),    # idx 1 — mild, trim blue tail
        (90, 20),    # idx 2 — typical default (applied on first Process)
        (85, 30),    # idx 3 — tighter
        (80, 40),    # idx 4 — aggressive
        (75, 50),    # idx 5 — very aggressive
    )

    def _on_auto_contrast(self, *_args):
        """Cycle the tau + intensity contrast one step tighter.

        Uses the currently cached (tau, intensity) of the FIRST FOV to
        compute asymmetric percentiles of tau over signal pixels and
        writes the result into the tau_min / tau_max / intensity_clip
        spinboxes. The live-apply connections on those spinboxes then
        trigger a re-render of every cached FastFLIM layer.

        Asymmetric because the low-tau (blue) end is noisier: we clip
        the low end about 2× harder than the high end. At idx 0 the
        full tau range is shown and no intensity clipping is applied.
        """
        if not self._fastflim_cache:
            show_info('Auto contrast: run Process first — no FastFLIM data '
                      'cached yet.')
            return
        presets = PTUReader._AUTO_CONTRAST_PRESETS
        self._auto_cycle_idx = (self._auto_cycle_idx + 1) % len(presets)
        upper, lower = presets[self._auto_cycle_idx]
        upper = float(upper)
        lower = float(lower)

        first = next(iter(self._fastflim_cache.values()))
        tau = np.asarray(first['tau'], dtype=np.float32)
        inten = np.asarray(first['inten'], dtype=np.float32)

        # Signal mask: ignore pixels with near-zero photon count so
        # percentiles are not dragged toward 0 by empty background.
        thr = float(np.percentile(inten, 10)) if inten.size else 0.0
        mask = (inten > thr) & np.isfinite(tau)
        if not mask.any():
            mask = np.isfinite(tau)

        if upper >= 100.0 or lower <= 0.0:
            tau_lo = float(np.nanmin(tau[mask]))
            tau_hi = float(np.nanmax(tau[mask]))
        else:
            tau_lo = float(np.percentile(tau[mask], lower))
            tau_hi = float(np.percentile(tau[mask], upper))
        if tau_hi <= tau_lo:
            tau_hi = tau_lo + 1e-3

        # Write values; the .changed signals drive _redraw_all_fastflim.
        self.tau_min.value = round(tau_lo, 3)
        self.tau_max.value = round(tau_hi, 3)
        self.intensity_clip.value = round(upper, 1)
        try:
            self.status_label.value = (
                f'Auto contrast preset {self._auto_cycle_idx + 1}/'
                f'{len(presets)} — '
                f'tau [{tau_lo:.2f}, {tau_hi:.2f}] ns '
                f'(pctl {int(lower)}–{int(upper)}), '
                f'intensity clip {int(upper)}%.'
            )
        except Exception:
            pass

    def _redraw_all_fastflim(self, *_args):
        """Re-render every cached FastFLIM layer using the current control values.

        Wired to ``changed`` on tau_min / tau_max / gamma / floor / CLAHE /
        clahe_tile. Runs in the main thread; cost is O(pixels * n_layers),
        typically sub-second for a 2k x 2k image.
        """
        if not self._fastflim_cache:
            return
        for name, pair in list(self._fastflim_cache.items()):
            try:
                rgb = self._render_fastflim_rgb(pair['tau'], pair['inten'])
                if name in self.viewer.layers:
                    try:
                        del self.viewer.layers[name]
                    except Exception:
                        pass
                self.viewer.add_image(
                    rgb, name=name, rgb=True, blending='translucent',
                )
            except Exception as e:
                print(f'[FastFLIM redraw] {name}: {e}')

    def _compute_tau_only(self, stack_sum, total_int, tau_res):
        eps = 1e-6
        B = stack_sum.shape[2]
        t = (np.arange(B, dtype=np.float32) * tau_res).reshape(1, 1, B)
        denom = np.maximum(total_int.astype(np.float32), eps)
        tau_map = (stack_sum.astype(np.float32) * t).sum(axis=2) / denom
        return tau_map
    def _on_input_dir_changed(self):
        """When the PTU folder changes, update output to its parent."""
        new_input = self.input_dir.value
        if new_input and os.path.isdir(new_input):
            # use pathlib for clarity
            parent_dir = str(Path(new_input).parent)
            self.output_dir.value = parent_dir
    def _on_process(self):
        in_dir = Path(self.input_dir.value)
        out_dir = Path(self.output_dir.value)
        frame = self.frame.value
        if not in_dir.is_dir():
            show_warning(f"Input folder not found: {in_dir}")
            return
        out_int = out_dir / 'intensity'
        out_stack = out_dir / 'flim_stack'
        out_int.mkdir(parents=True, exist_ok=True)
        out_stack.mkdir(parents=True, exist_ok=True)

        ptu_files = sorted(in_dir.glob('*.ptu'))
        if not ptu_files:
            show_warning(f"No .ptu files in {in_dir}")
            return

        # Detect already-processed PTUs and ask whether to overwrite.
        already_done = []
        for p in ptu_files:
            sum_int = out_int / f'{p.stem}_sum.tif'
            sum_stack = out_stack / f'{p.stem}_sum.tif'
            if sum_int.exists() or sum_stack.exists():
                already_done.append(p.name)
        if already_done:
            preview = '\n  '.join(already_done[:5])
            more = f'\n  ... and {len(already_done) - 5} more' if len(already_done) > 5 else ''
            msg = (
                f'{len(already_done)} PTU file(s) already have outputs in:\n'
                f'  {out_int}\n  {out_stack}\n\n'
                f'Already processed:\n  {preview}{more}\n\n'
                f'Re-run and OVERWRITE existing TIFs?'
            )
            reply = QMessageBox.question(
                None, 'PTU already processed', msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                show_info('Skipped — outputs kept as-is. You can click Next to continue.')
                return

        # Capture GUI values on main thread — worker runs in a different thread
        tau_min = float(self.tau_min.value)
        tau_max = float(self.tau_max.value)
        tau_res = float(self.tau_res.value)
        clip_pct = float(self.intensity_clip.value)

        n_total = len(ptu_files)
        self.progress.min = 0
        self.progress.max = max(1, n_total * 100)
        self.progress.value = 0
        self.status_label.value = f'Starting {n_total} file(s)...'
        self.process_btn.enabled = False

        worker = self._process_worker(
            ptu_files=ptu_files, out_dir=out_dir, out_int=out_int, out_stack=out_stack,
            frame=frame, tau_min=tau_min, tau_max=tau_max,
            tau_res=tau_res, clip_pct=clip_pct,
        )
        worker.yielded.connect(self._on_worker_yield)
        worker.returned.connect(self._on_worker_done)
        worker.errored.connect(self._on_worker_error)
        worker.start()

    @thread_worker
    def _process_worker(self, ptu_files, out_dir, out_int, out_stack, frame,
                        tau_min, tau_max, tau_res, clip_pct):
        n_total = len(ptu_files)
        for file_idx, p in enumerate(ptu_files):
            base = file_idx * 100
            try:
                yield ('status', base + 2, f'[{file_idx+1}/{n_total}] Loading {p.name} (may take ~30-60s)...')
                raw = _load_ptu(p, frame)
                arr = np.array(raw)
                while arr.ndim > 4:
                    arr = arr.sum(axis=0)

                if arr.ndim == 4:
                    yield ('status', base + 35, f'[{file_idx+1}/{n_total}] Summing intensity / decay stacks...')
                    intensity = arr.sum(axis=3)
                    stack_sum = arr.sum(axis=2)
                    total_int = stack_sum.sum(axis=2)

                    n_ch = intensity.shape[2]
                    for ch in range(n_ch):
                        yield ('status', base + 40 + int(30 * ch / max(1, n_ch)),
                               f'[{file_idx+1}/{n_total}] Writing channel {ch+1}/{n_ch} TIFs...')
                        int_img = intensity[..., ch]
                        tifffile.imwrite(out_int / f"{p.stem}_ch{ch + 1}.tif",
                                         int_img.astype(np.uint16), imagej=True)
                        stack_ch = arr[..., ch, :].transpose(2, 0, 1)
                        tifffile.imwrite(out_stack / f"{p.stem}_ch{ch + 1}.tif",
                                         stack_ch.astype(np.uint16), imagej=True)
                        yield ('layer', f"{p.stem}_ch{ch + 1}", stack_ch.astype(np.uint16), {})

                    yield ('status', base + 75, f'[{file_idx+1}/{n_total}] Writing sum TIFs...')
                    tifffile.imwrite(out_int / f"{p.stem}_sum.tif",
                                     total_int.astype(np.uint16), imagej=True)
                    summed_stack = stack_sum.transpose(2, 0, 1)
                    tifffile.imwrite(out_stack / f"{p.stem}_sum.tif",
                                     summed_stack.astype(np.uint16), imagej=True)

                    # FastFLIM
                    yield ('status', base + 85, f'[{file_idx+1}/{n_total}] Computing FastFLIM tau map...')
                    try:
                        tau_map = self._compute_tau_only(
                            stack_sum=stack_sum, total_int=total_int, tau_res=tau_res,
                        )
                        # Keep the raw tau .tif for debugging / downstream
                        # quantification; the user-visible representation is
                        # the intensity-weighted RGB below.
                        tifffile.imwrite(out_dir / f"{p.stem}_fastflim_tau.tif", tau_map)
                        yield ('fastflim', f"{p.stem}_FastFLIM",
                               tau_map.astype(np.float32),
                               total_int.astype(np.float32))
                    except Exception as e_fast:
                        yield ('warn', f'FastFLIM failed for {p.name}: {e_fast}')

                elif arr.ndim == 3:
                    for i in range(arr.shape[2]):
                        slice_ = arr[..., i]
                        tifffile.imwrite(out_int / f"{p.stem}_slice{i+1}.tif",
                                         slice_.astype(np.uint16), imagej=True)
                        tifffile.imwrite(out_stack / f"{p.stem}_slice{i+1}.tif",
                                         slice_.astype(np.uint16), imagej=True)
                elif arr.ndim == 2:
                    tifffile.imwrite(out_int / f"{p.stem}.tif",
                                     arr.astype(np.uint16), imagej=True)
                    tifffile.imwrite(out_stack / f"{p.stem}.tif",
                                     arr.astype(np.uint16), imagej=True)
                else:
                    yield ('warn', f'Unexpected array dims {arr.ndim} for {p.name}')

                yield ('status', (file_idx + 1) * 100, f'[{file_idx+1}/{n_total}] Done: {p.name}')
            except Exception as e:
                yield ('warn', f'Failed {p.name}: {e}')
        return True

    def _on_worker_yield(self, payload):
        try:
            kind = payload[0]
        except Exception:
            return
        if kind == 'status':
            _, val, msg = payload
            try:
                self.progress.value = int(val)
            except Exception:
                pass
            self.status_label.value = msg
        elif kind == 'layer':
            _, name, data, kwargs = payload
            try:
                self.viewer.add_image(data, name=name, **(kwargs or {}))
            except Exception as e:
                show_warning(f'Add layer {name} failed: {e}')
        elif kind == 'fastflim':
            # FastFLIM RGB payload: cache on the widget, render, add layer,
            # save PNG next to the raw tau .tif.
            _, name, tau_map, total_int = payload
            try:
                # First FOV this session — advance auto-contrast to the
                # default "data-driven" preset (idx=2 → 10/90 percentiles)
                # so the user does not see hardcoded 3.1 / 5.1 ns on data
                # whose real range is elsewhere. Subsequent FOVs inherit
                # the current controls.
                first_fov = not self._fastflim_cache
                self._push_fastflim_layer(name, tau_map, total_int)
                if first_fov and self._auto_cycle_idx < 0:
                    # Jump straight to 90% preset (idx 2) on first Process.
                    self._auto_cycle_idx = 1
                    self._on_auto_contrast()
                # Persist the RGB with current display settings alongside
                # the raw tau .tif. Users can later drag a slider and
                # re-export by calling the render helper manually if
                # needed — the RGB here is a one-shot snapshot with the
                # settings that were active when Process finished.
                try:
                    from PIL import Image as _PILImage
                    out_dir = Path(str(self.output_dir.value))
                    rgb = self._render_fastflim_rgb(tau_map, total_int)
                    _PILImage.fromarray(rgb).save(
                        str(out_dir / f"{name.replace('_FastFLIM','')}_fastflim_rgb.png"))
                except Exception as e_save:
                    print(f'[FastFLIM RGB save] {name}: {e_save}')
            except Exception as e:
                show_warning(f'FastFLIM layer {name} failed: {e}')
        elif kind == 'warn':
            show_warning(payload[1])

    def _on_worker_done(self, _result=None):
        self.progress.value = self.progress.max
        self.status_label.value = 'All PTU files processed. Next: Barcode Seg.'
        self.process_btn.enabled = True
        show_info('All PTU files processed. Click Next to continue to Barcode Seg.')

    def _on_worker_error(self, exc):
        self.status_label.value = f'ERROR: {exc}'
        self.process_btn.enabled = True
        show_warning(f'Processing failed: {exc}')


def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau ) + c

def calcu_phasor_info(
    roi_decay_data,
    peak_idx,
    tau_resolution=0.1,
    pulse_freq=80,
    harmonics=1,
    PEAK_OFFSET=0,
    END_OFFSET=0
):
    mask_start = peak_idx + PEAK_OFFSET
    mask_end = len(roi_decay_data) - END_OFFSET
    mask_start = max(0, mask_start)
    mask_end = max(mask_start + 1, mask_end)

    decay_segment_mask = np.zeros_like(roi_decay_data, dtype=bool)
    decay_segment_mask[mask_start:mask_end] = True

    roi_decay_data_segment = roi_decay_data[decay_segment_mask]
    peak_val = np.max(roi_decay_data_segment) if roi_decay_data_segment.size else 0.0
    if peak_val <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    roi_decay_data_normalized = roi_decay_data_segment / peak_val

    t_arr = np.arange(len(roi_decay_data_normalized)) * tau_resolution

    denom = np.sum(roi_decay_data_normalized)
    if denom <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    fastflim = np.sum(roi_decay_data_normalized * t_arr) / denom

    params_init = [1, 2, 0]
    popt, pcov = curve_fit(exp_func, t_arr, roi_decay_data_normalized, p0=params_init)
    lifetime = popt[1]
    chi_square = np.sum((roi_decay_data_normalized - exp_func(t_arr, *popt)) ** 2)

    pf = pulse_freq / 1000.0
    phasor_g = np.sum(roi_decay_data_normalized * np.cos(2 * np.pi * pf * harmonics * t_arr)) / denom
    phasor_s = np.sum(roi_decay_data_normalized * np.sin(2 * np.pi * pf * harmonics * t_arr)) / denom

    print(f'g: {phasor_g}, s: {phasor_s}')
    return phasor_g, phasor_s, lifetime, chi_square, fastflim

# Parameter input: mask intensity threshold, pixel intensity threshold, peak offset, end offset, tau resolution, pulse frequency, harmonics

def Gen_excel_multi(
    stack1, stack2, stack3, stack4,
    basefolder: str,
    seg_dict: dict,  # {'n': seg_img, 'm': seg_img, 'p': seg_img} OR {'': seg_img}
    mask_int_thres, pixel_int_thres,
    peak_offset, end_offset,
    tau_resolution, pulse_freq,
    harmonics,
    fov: str
):
    """
    seg_dict:
      - keys are 'n','m','p' or '' (empty means unknown/unspecified localization)
      - values are seg_img (H,W) int masks
    """
    save_path = os.path.join(basefolder, 'FLIM-S.xlsx')

    all_rows = []

    # ---- precompute intensities once (same for all segmentations) ----
    intensity_1 = np.sum(stack1, axis=0)
    intensity_2 = np.sum(stack2, axis=0)
    intensity_3 = np.sum(stack3, axis=0)
    intensity_4 = np.sum(stack4, axis=0)
    intensity_image = intensity_1 + intensity_2 + intensity_3 + intensity_4
    decay_data = stack1 + stack2 + stack3 + stack4

    for loc_key, seg_img in seg_dict.items():
        # loc_key: 'n'/'m'/'p'/''  -> write as 'N'/'M'/'P'/'' in excel
        loc_out = loc_key.upper() if isinstance(loc_key, str) else ''
        if loc_out not in ('N', 'M', 'P'):
            loc_out = ''

        labels = np.unique(seg_img)
        labels = labels[labels != 0]

        for label in labels:
            cell_mask = seg_img == label
            valid_pixel_mask = (intensity_image >= pixel_int_thres) & cell_mask

            mask_intensity = np.sum(intensity_image[cell_mask])
            if mask_intensity < mask_int_thres:
                # 你原来是 print，我保留
                print(f"[{loc_out}] Mask label {label} excluded (low total intensity).")
                continue

            roi_decay_data = np.sum(decay_data[:, valid_pixel_mask], axis=-1)
            if roi_decay_data.size == 0:
                continue

            peak_idx = int(np.argmax(roi_decay_data))
            total_intensity = np.sum(intensity_image[cell_mask])

            phasor_g, phasor_s, lifetime, chi_square, fastflim = calcu_phasor_info(
                roi_decay_data,
                peak_idx=peak_idx,
                PEAK_OFFSET=peak_offset,
                END_OFFSET=end_offset,
                tau_resolution=tau_resolution,
                pulse_freq=pulse_freq,
                harmonics=harmonics
            )

            int_570_590 = np.sum(intensity_1[cell_mask])
            int_590_610 = np.sum(intensity_2[cell_mask])
            int_610_638 = np.sum(intensity_3[cell_mask])
            int_638_720 = np.sum(intensity_4[cell_mask])

            # 防止极端情况下 total_intensity==0
            denom = total_intensity if total_intensity != 0 else 1.0
            norm_1_4_1 = int_570_590 / denom
            norm_1_4_2 = int_590_610 / denom
            norm_1_4_3 = int_610_638 / denom
            norm_1_4_4 = int_638_720 / denom

            all_rows.append({
                'Localization': loc_out,
                'G': phasor_g,
                'S': phasor_s,
                'Lifetime': lifetime,
                'Chi^2': chi_square,
                'Total intensity': total_intensity,
                'Mask label': label,
                'FastFLIM': fastflim,
                'Int 570-590': int_570_590,
                'Int 590-610': int_590_610,
                'Int 610-638': int_610_638,
                'Int 638-720': int_638_720,
                'Int 1/(1-4)': norm_1_4_1,
                'Int 2/(1-4)': norm_1_4_2,
                'Int 3/(1-4)': norm_1_4_3,
                'Int 4/(1-4)': norm_1_4_4,
                'FOV': fov
            })

    data_df = pd.DataFrame(all_rows)

    if os.path.exists(save_path):
        print('Excel file already exists, will overwrite the data.')
        notifications.show_info('Excel file already exists, will overwrite the data.')

    data_df.to_excel(save_path, index=False)
    print(f'Excel file saved at {save_path}')
    notifications.show_info(f'Excel file saved at {save_path}')
    return data_df

if TYPE_CHECKING:
    import napari

class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def plot_signal(self, *signals, titles=None):
        """
        Plots one or more 1D signals in a single row of subplots.

        Parameters:
            *signals: A variable number of 1D arrays representing the signals to plot.
            titles (optional): A list of titles for each subplot. If not provided, default titles will be used.
        """
        # Clear the current figure.
        self.figure.clear()

        n = len(signals)
        if n == 0:
            return  # Nothing to plot.

        # Use default titles if none were provided.
        if titles is None or len(titles) != n:
            titles = [f"Signal {i + 1}" for i in range(n)]

        # Create one row with n columns.
        for i, signal in enumerate(signals):
            ax = self.figure.add_subplot(1, n, i + 1)
            ax.plot(signal)
            ax.set_title(titles[i])

        # Refresh the canvas.
        self.canvas.draw()
        print('Signal plotted')

    def plot_phasor(self, g_values, s_values):
        # self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(g_values, s_values)
        # Draw a semi-circle for reference
        theta = np.linspace(0, np.pi, 100)
        x = 0.5 + 0.5 * np.cos(theta)
        y = 0.5 * np.sin(theta)
        ax.plot(x, y)
        ax.set_xlabel('G')
        ax.set_ylabel('S')
        ax.set_title('Phasor plot')
        self.canvas.draw()

    def plot_phasor_cls(self, g_values, s_values, class_values, colors, checked_cls):
        # self.figure.clear()
        # ax = self.figure.add_subplot(142)
        ax = self.figure.add_subplot(132)
        # set figure size
        self.figure.set_size_inches(7,7)
        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue
            idx = class_values == i
            ax.scatter(g_values[idx], s_values[idx], color=color, label=f'Class {i}')
        ax.set_xlabel('G')
        ax.set_ylabel('S')
        ax.set_title('Phasor plot')
        self.canvas.draw()

    def plot_int12_cls(self, int1, int2, class_values, colors, checked_cls):
        # self.figure.clear()
        ax = self.figure.add_subplot(143)
        # set figure size
        self.figure.set_size_inches(7,7)
        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue
            idx = class_values == i
            ax.scatter(int1[idx], int2[idx], color=color, label=f'Class {i}')
        ax.set_xlabel('Int 1')
        ax.set_ylabel('Int 2')
        ax.set_title('Int 1-2 plot')
        self.canvas.draw()

    def plot_int13_cls(self, int1, int3, class_values, colors, checked_cls):
        # self.figure.clear()
        ax = self.figure.add_subplot(144)
        # set figure size
        self.figure.set_size_inches(7,7)
        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue
            idx = class_values == i
            ax.scatter(int1[idx], int3[idx], color=color, label=f'Class {i}')
        ax.set_xlabel('Int 1')
        ax.set_ylabel('Int 3')
        ax.set_title('Int 1-3 plot')
        self.canvas.draw()

    def plot_int123_cls(self, int1, int2, int3, class_values, colors, checked_cls):
        # self.figure.clear()
        ax = self.figure.add_subplot(133)
        self.figure.set_size_inches(10, 7)  # Adjust the size as necessary
        marker_styles = ['o', '*']

        for i, color in enumerate(colors):
            if i not in checked_cls:
                continue

            # Indices where the class matches
            idx = class_values == i

            # Plot int2 with the first marker style
            ax.scatter(int1[idx], int2[idx], color=color, marker=marker_styles[0], label=f'Class {i} int2')

            # Plot int3 with the second marker style
            ax.scatter(int1[idx], int3[idx], color=color, marker=marker_styles[1], label=f'Class {i} int3')

        ax.set_xlabel('Int 1')
        ax.set_ylabel('Int 2(o) & Int 3(*)')
        ax.set_title('Spectral Intensity Plot')
        # ax.legend()
        self.canvas.draw()





def load_cellpose_mask_from_npy(npy_path: str) -> np.ndarray:
    """Load cellpose-style npy and return masks (H,W) int array."""
    obj = np.load(npy_path, allow_pickle=True)
    # cellpose common: np.save(..., dat) where dat is dict -> ndarray(object) of len 1
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        # could be dict packed in array
        if obj.size == 1 and isinstance(obj.item(), dict):
            d = obj.item()
            if 'masks' in d:
                return d['masks'].astype(np.int32)
            # fallback: try common keys
            for k in ['mask', 'labels', 'seg', 'masks_pred']:
                if k in d:
                    return d[k].astype(np.int32)
            raise ValueError(f"Cellpose npy dict has no 'masks' key: {npy_path}")
        # could be list-like
        if obj.size > 1 and isinstance(obj.flat[0], dict):
            d = obj.flat[0]
            if 'masks' in d:
                return d['masks'].astype(np.int32)
            raise ValueError(f"Unexpected object array content in: {npy_path}")
    # if it's directly numeric array
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number):
        if obj.ndim != 2:
            raise ValueError(f"Segmentation array must be 2D (H,W), got {obj.shape} in {npy_path}")
        return obj.astype(np.int32)

    raise ValueError(f"Unrecognized npy format: {npy_path}")

def find_intensity_folder(basefolder: str) -> str | None:
    """Return intensity folder path if exists, else None."""
    p = Path(basefolder)
    if not p.exists():
        return None
    # prefer exact "Intensity"
    cand = p / "Intensity"
    if cand.exists() and cand.is_dir():
        return str(cand)
    # case-insensitive fallback
    for child in p.iterdir():
        if child.is_dir() and child.name.lower() == "intensity":
            return str(child)
    return None

def find_seg_npys(intensity_folder: str):
    """
    Return:
      - seg_map: dict loc -> npy_path (loc in {'n','m','p'})
      - fallback_npy: first npy path (or None)
    """
    p = Path(intensity_folder)
    npys = sorted([x for x in p.glob("*.npy") if x.is_file()])
    if not npys:
        return {}, None

    seg_map = {}
    # match: xxx-n.npy / xxx_m.npy / xxxN.npy? 你说的是 xxx-n.npy，我按 -/_ 都支持一下
    pat = re.compile(r"(.+)[-_]([nmpNMP])\.npy$")
    for f in npys:
        m = pat.match(f.name)
        if m:
            loc = m.group(2).lower()
            if loc in ("n", "m", "p"):
                seg_map[loc] = str(f)

    fallback_npy = str(npys[0])
    return seg_map, fallback_npy

class Calculate_FLIM_S(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        print('version 250828 (NMP barcode-ready)')
        super().__init__()
        self._viewer = viewer

        # stacks
        self._stack_selectors = [
            create_widget(label=f"Stack {i + 1}", annotation="napari.layers.Image")
            for i in range(4)
        ]

        # Base folder (replaces output folder)
        self._base_dir = FileEdit(label="Base Folder", mode='d')
        self._base_dir.value = r'J:/Mix16-N-P-260306-DCZ-2-1'

        # Optional manual masks for N/M/P (Labels layers) — nullable so they don't
        # auto-default to the first Labels layer in the viewer.
        self._seg_n = create_widget(
            label="Segmentation (N) [optional]", annotation="napari.layers.Labels",
            options={"nullable": True},
        )
        self._seg_m = create_widget(
            label="Segmentation (M) [optional]", annotation="napari.layers.Labels",
            options={"nullable": True},
        )
        self._seg_p = create_widget(
            label="Segmentation (P) [optional]", annotation="napari.layers.Labels",
            options={"nullable": True},
        )

        # fallback single segmentation (optional) — only use if N/M/P not provided
        self._seg_any = create_widget(
            label="Segmentation (Any) [optional]", annotation="napari.layers.Labels",
            options={"nullable": True},
        )

        # params
        self._mask_int_thres = create_widget(label="Mask Intensity Threshold", widget_type="Slider", value=1500)
        self._mask_int_thres.min = 500
        self._mask_int_thres.max = 100000

        self._pixel_int_thres = create_widget(label="Pixel Intensity Threshold", widget_type="Slider", value=7)
        self._pixel_int_thres.min = 0
        self._pixel_int_thres.max = 500

        self._peak_offset = create_widget(label="Peak Offset(bins)", widget_type="Slider", value=4)
        self._peak_offset.min = 0
        self._peak_offset.max = 20

        self._end_offset = create_widget(label="End Offset(bins)", widget_type="Slider", value=18)
        self._end_offset.min = 0
        self._end_offset.max = 50

        self._tau_resolution = create_widget(
            label="Tau Resolution(ns)",
            widget_type=FloatSpinBox,
            value=0.097,
            options={'min': 0, 'max': 1, 'step': 0.001}
        )

        self._pulse_frequency = create_widget(
            label="Pulse Frequency(MHz)",
            widget_type=FloatSpinBox,
            value=78.1,
            options={'min': 0, 'max': 100, 'step': 0.1}
        )

        self._harmonics = create_widget(label="Harmonics", widget_type="Slider", value=1)
        self._harmonics.min = 1
        self._harmonics.max = 10

        self._process_button = PushButton(text="Process and Save to Excel")
        self._process_button.clicked.connect(self.process_and_save_to_excel)
        _style_process_button(self._process_button)

        self._progress = widgets.ProgressBar(label='Progress', value=0, min=0, max=100)
        self._status_label = Label(value='Ready')

        _tt(self._base_dir,
            'Sample folder. Calculate FLIM-S reads flim_stack/*_ch*.tif and '
            'writes FLIM-S.xlsx here.')
        for i, sel in enumerate(self._stack_selectors):
            _tt(sel, f'Channel {i+1} FLIM decay stack (H×W×T). Auto-picked '
                     f'from flim_stack/*_ch{i+1}.tif if available.')
        _tt(self._seg_n,
            'Optional nucleus mask (e.g. mask_n_fill from Barcode Seg). '
            'Leave empty if you only have a combined mask.')
        _tt(self._seg_m,
            'Optional middle / membrane mask. Usually left empty.')
        _tt(self._seg_p,
            'Optional cytoplasm mask (e.g. mask_p_fill from Barcode Seg).')
        _tt(self._seg_any,
            'Fallback single mask used only if N / M / P are all empty.')
        _tt(self._mask_int_thres,
            'Minimum TOTAL photon count summed across a cell mask. Cells '
            'whose total intensity is below this threshold are skipped '
            '(too few photons for a reliable phasor fit). Raise if you '
            'still see noisy tau outliers; lower if legitimate dim cells '
            'are being dropped.')
        _tt(self._pulse_frequency,
            'Laser repetition rate in MHz. Leica SP8 / STELLARIS default: '
            '78.1 MHz. This sets the time window the phasor g, s are '
            'evaluated in (T = 1 / (n · f_rep)), so getting it right is '
            'essential — wrong freq = wrong tau.')
        _tt(self._pixel_int_thres,
            'Per-PIXEL intensity floor applied when aggregating photons '
            'into the cell decay — pixels below this are discarded so '
            'dim background does not dilute the phasor fit. Different '
            'from Mask Intensity Threshold (which gates the whole cell).')
        _tt(self._peak_offset,
            'Start of the fitting window, measured in bins AFTER the '
            'decay peak. Used to skip the IRF-convolved rising edge so '
            'only the exponential tail is fit. Typical: 3–6 bins.')
        _tt(self._end_offset,
            'End of the fitting window, measured in bins BEFORE the end '
            'of the decay. Drops the late-time noisy tail where signal '
            'has decayed into the dark count. Typical: 10–20 bins.')
        _tt(self._tau_resolution,
            'Time-bin width in ns. Leica SP8 / STELLARIS with 256-bin '
            'decay default: 0.097 ns (≈ 25 ns / 256 bins at 40 MHz, or '
            '12.8 / 256 at 78 MHz). Check your PTU metadata if unsure.')
        _tt(self._harmonics,
            'Phasor harmonic order n. Evaluates g, s at n · f_rep (n=1 '
            'at the fundamental laser rep rate, n=2 at double, etc.). '
            'Use n=1 for single-anchor barcode FLIM (standard, what we '
            'use). Higher n (2–3) probes faster decay components and '
            'can help separate multi-exponential lifetimes but '
            'amplifies noise — not needed here.')
        _tt(self._process_button,
            'Fit phasor + lifetime per cell and write FLIM-S.xlsx to the '
            'Base folder.')

        _append_section_divider(self, '— 📚 FLIM stacks —')
        self.extend(self._stack_selectors + [self._base_dir])

        _append_section_divider(self, '— 🧩 Segmentations —')
        self.extend([self._seg_n, self._seg_m, self._seg_p, self._seg_any])

        _append_section_divider(self, '— ⚙ Phasor parameters —')
        self.extend([self._mask_int_thres, self._pulse_frequency,
                     self._pixel_int_thres, self._peak_offset, self._end_offset,
                     self._tau_resolution, self._harmonics])

        _append_section_divider(self, '— ▶ Process —')
        self.extend([self._process_button, self._progress, self._status_label])

        _add_next_button(self, viewer)
        _tighten_container(self)
        self._base_dir.changed.connect(self._populate_initial_layers)
        self._populate_initial_layers()

    def _auto_load_flim_stack(self, ch_idx: int):
        """Try to load `<base>/flim_stack/*_ch{ch_idx}.tif` into napari, return the new layer or None."""
        base = str(self._base_dir.value) if self._base_dir.value else ''
        if not base:
            return None
        flim_dir = Path(base) / 'flim_stack'
        if not flim_dir.is_dir():
            return None
        tifs = sorted(flim_dir.glob(f'*_ch{ch_idx}.tif'))
        if not tifs:
            return None
        tif_path = tifs[0]
        # already loaded?
        for layer in self._viewer.layers:
            if isinstance(layer, NapariImage) and layer.name == tif_path.stem:
                return layer
        try:
            data = tifffile.imread(str(tif_path))
            return self._viewer.add_image(data, name=tif_path.stem)
        except Exception as e:
            print(f'[auto-load ch{ch_idx}] {e}')
            return None

    def _populate_initial_layers(self):
        # stacks: first try matching existing napari Image layers; fall back to disk.
        for i, selector in enumerate(self._stack_selectors):
            target_tag = f"_ch{i + 1}"
            matched = None
            for layer in self._viewer.layers:
                if isinstance(layer, NapariImage) and target_tag in layer.name:
                    matched = layer
                    break
            if matched is None:
                matched = self._auto_load_flim_stack(i + 1)
            if matched is not None:
                # The magicgui layer-selector's choices list was snapshotted when
                # the widget was built; refresh it so newly added layers are valid.
                try:
                    selector.reset_choices()
                except Exception:
                    pass
                selector.value = matched

        # Auto-fill segmentations from Labels layers by name.
        # Reset all four first so they don't auto-default to the first layer in the viewer.
        for sel in (self._seg_n, self._seg_m, self._seg_p, self._seg_any):
            try:
                sel.reset_choices()
            except Exception:
                pass
            try:
                sel.value = None
            except Exception:
                pass

        # Also auto-load N/P masks from intensity/*_seg_n.npy and *_seg_p.npy if they exist
        base = str(self._base_dir.value) if self._base_dir.value else ''
        if base:
            int_dir = Path(base) / 'intensity'
            if int_dir.is_dir():
                for tag, selector in (('_seg_n.npy', self._seg_n), ('_seg_p.npy', self._seg_p)):
                    hits = sorted(int_dir.glob(f'*{tag}'))
                    if hits:
                        # Prefer a layer already in napari
                        already = None
                        for lay in self._viewer.layers:
                            if isinstance(lay, Labels) and lay.name == hits[0].stem:
                                already = lay
                                break
                        if already is None:
                            try:
                                mask = _load_mask(hits[0])
                                already = self._viewer.add_labels(
                                    mask.astype(np.uint32), name=hits[0].stem,
                                )
                            except Exception as e:
                                print(f'[auto-load {tag}] {e}')
                        if already is not None:
                            try:
                                selector.reset_choices()
                            except Exception:
                                pass
                            selector.value = already

        # Best-effort match from existing napari Labels layer names — N and P only;
        # M and Any stay None unless user explicitly picks them.
        for layer in self._viewer.layers:
            if not isinstance(layer, Labels):
                continue
            name = layer.name.lower()
            if self._seg_n.value is None and any(k in name for k in ['_seg_n', 'mask_n', '-n', '_n_']):
                try:
                    self._seg_n.reset_choices()
                except Exception:
                    pass
                self._seg_n.value = layer
            elif self._seg_p.value is None and any(k in name for k in ['_seg_p', 'mask_p', '-p', '_p_']):
                try:
                    self._seg_p.reset_choices()
                except Exception:
                    pass
                self._seg_p.value = layer

    def _resolve_segmentations(self, basefolder: str):
        """
        Returns seg_dict suitable for Gen_excel_multi.
        Priority:
          1) manual N/M/P if provided
          2) manual Any if provided (as '' key)
          3) auto from basefolder/Intensity (*.npy)
        """
        seg_dict = {}

        # 1) manual N/M/P
        if self._seg_n.value is not None:
            seg_dict['n'] = self._seg_n.value.data
        if self._seg_m.value is not None:
            seg_dict['m'] = self._seg_m.value.data
        if self._seg_p.value is not None:
            seg_dict['p'] = self._seg_p.value.data

        if seg_dict:
            return seg_dict

        # 2) manual Any
        if self._seg_any.value is not None:
            seg_dict[''] = self._seg_any.value.data
            return seg_dict

        # 3) auto from Intensity folder
        intensity_folder = find_intensity_folder(basefolder)
        if intensity_folder is None:
            return None  # will trigger prompt

        seg_map, fallback_npy = find_seg_npys(intensity_folder)
        if seg_map:
            for loc, npy_path in seg_map.items():
                try:
                    seg_dict[loc] = load_cellpose_mask_from_npy(npy_path)
                except Exception as e:
                    print(f"[WARN] Failed to load {loc.upper()} mask from {npy_path}: {e}")
            if seg_dict:
                return seg_dict

        # no N/M/P -> fallback first npy
        if fallback_npy is not None:
            try:
                seg_dict[''] = load_cellpose_mask_from_npy(fallback_npy)
                return seg_dict
            except Exception as e:
                print(f"[WARN] Failed to load fallback mask from {fallback_npy}: {e}")

        return None

    def process_and_save_to_excel(self):
        stack_layers = [s.value for s in self._stack_selectors]
        if any(x is None for x in stack_layers):
            notifications.show_error("Please select Stack 1-4.")
            return

        basefolder = str(self._base_dir.value)
        seg_dict = self._resolve_segmentations(basefolder)
        if seg_dict is None or len(seg_dict) == 0:
            msg = ("No segmentation found.\n"
                   "Either:\n"
                   "  - Put cellpose .npy in BaseFolder/intensity (e.g. xxx_seg_n.npy / xxx_seg_p.npy), OR\n"
                   "  - Manually select a Labels layer in N/M/P or 'Any'.")
            notifications.show_error(msg)
            return

        fov = os.path.split(stack_layers[0].name)[-1].split('_ch')[0]

        params = dict(
            basefolder=basefolder,
            mask_int_thres=self._mask_int_thres.value,
            pixel_int_thres=self._pixel_int_thres.value,
            peak_offset=self._peak_offset.value,
            end_offset=self._end_offset.value,
            tau_resolution=self._tau_resolution.value,
            pulse_frequency=self._pulse_frequency.value,
            harmonics=self._harmonics.value,
            fov=fov,
        )
        stacks = [np.asarray(s.data) for s in stack_layers]

        self._progress.min = 0
        self._progress.max = 100
        self._progress.value = 0
        self._status_label.value = f'Starting FLIM-S for {fov} '\
                                   f'(seg groups: {list(seg_dict.keys())})...'
        self._process_button.enabled = False

        worker = self._flims_worker(stacks=stacks, seg_dict=seg_dict, params=params)
        worker.yielded.connect(self._on_flims_yield)
        worker.returned.connect(self._on_flims_done)
        worker.errored.connect(self._on_flims_error)
        worker.start()

    @thread_worker
    def _flims_worker(self, stacks, seg_dict, params):
        import time as _time
        yield ('status', 10, 'Running Gen_excel_multi (this can take a while for large FOVs)...')
        t0 = _time.time()
        data_df = Gen_excel_multi(
            stacks[0], stacks[1], stacks[2], stacks[3],
            params['basefolder'], seg_dict,
            params['mask_int_thres'], params['pixel_int_thres'],
            params['peak_offset'], params['end_offset'],
            params['tau_resolution'], params['pulse_frequency'],
            params['harmonics'],
            params['fov'],
        )
        yield ('status', 95, f'Gen_excel_multi done in {_time.time()-t0:.1f}s.')
        return data_df

    def _on_flims_yield(self, payload):
        try:
            kind = payload[0]
        except Exception:
            return
        if kind == 'status':
            _, val, msg = payload
            try:
                self._progress.value = int(val)
            except Exception:
                pass
            self._status_label.value = msg
        elif kind == 'warn':
            show_warning(payload[1])

    def _on_flims_done(self, data_df):
        self._progress.value = self._progress.max
        self._status_label.value = 'FLIM-S done. Excel saved. Plotting G-S...'
        self._process_button.enabled = True
        # Plot on main thread (matplotlib is not thread-safe)
        try:
            plt.figure()
            plt.scatter(data_df['G'], data_df['S'])
            plt.xlabel('G')
            plt.ylabel('S')
            plt.title('G-S plot')
            plt.show()
        except Exception as e:
            show_warning(f'G-S plot failed: {e}')
        show_info('FLIM-S processing complete. Click Next to continue.')

    def _on_flims_error(self, exc):
        self._status_label.value = f'ERROR: {exc}'
        self._process_button.enabled = True
        show_warning(f'FLIM-S failed: {exc}')
        traceback.print_exc()



def get_color_map(n_colors: int):
    """
    Return n_colors RGB tuples (0-1).
    Color[0] is always gray (#808080) for background/unassigned.
    """
    base_hex = [
        '#808080',  # <-- fixed gray at index 0
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#469990', '#9A6324', '#808000', '#000075',
        '#800000', '#aaffc3'
    ]

    if n_colors <= len(base_hex):
        arr = np.array(
            [[int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)] for h in base_hex[:n_colors]],
            dtype=np.float32
        ) / 255.0
        print(f"Using predefined color map for {n_colors} colors.")
        return arr

    # extend: keep first gray, then tab20 for the rest
    colors = [np.array([128, 128, 128], dtype=np.float32) / 255.0]  # index 0
    cmap = plt.cm.tab20
    for i in range(1, n_colors):
        colors.append(np.array(cmap((i - 1) % 20)[:3], dtype=np.float32))
    print(f"Generated extended color map for {n_colors} colors.")
    return np.stack(colors, axis=0)


def _find_intensity_dir(folder: str) -> str | None:
    """Find intensity folder under 'folder' (case-insensitive)."""
    p = Path(folder)
    if not p.exists():
        return None
    # prefer intensity / Intensity
    for name in ["intensity", "Intensity"]:
        cand = p / name
        if cand.is_dir():
            return str(cand)
    # fallback case-insensitive scan
    for child in p.iterdir():
        if child.is_dir() and child.name.lower() == "intensity":
            return str(child)
    return None

def _select_seg_file_by_loc(int_folder: str, fov: str, loc: str | None):
    """
    If loc is N/M/P:
      1) strictly look for fov-n.npy or fov_n.npy
      2) if not found, WARN and fallback to the first .npy in folder
    If loc is '' or None:
      - fallback to first .npy in folder
    """
    loc = (loc or "").strip().lower()

    # ---------- Case 1: explicit localization ----------
    if loc in ("n", "m", "p"):
        cand1 = os.path.join(int_folder, f"{fov}_sum_seg-{loc}.npy")
        cand2 = os.path.join(int_folder, f"{fov}_sum_seg_{loc}.npy")

        if os.path.isfile(cand1):
            return cand1
        if os.path.isfile(cand2):
            return cand2

        # nothing found → explicit warning
        print(
            f"[WARN] No '{loc.upper()}' segmentation found for FOV '{fov}' "
            f"in {int_folder}. Falling back to first .npy."
        )

        all_npy = sorted(glob.glob(os.path.join(int_folder, "*.npy")))
        if all_npy:
            return all_npy[0]

        return None

    # ---------- Case 2: no localization specified ----------
    all_npy = sorted(glob.glob(os.path.join(int_folder, "*.npy")))
    if all_npy:
        return all_npy[0]

    return None

def _load_cellpose_masks(seg_path: str) -> np.ndarray | None:
    """Load cellpose npy and return masks (H,W)."""
    try:
        obj = np.load(seg_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1 and isinstance(obj.item(), dict):
            d = obj.item()
            return d.get("masks", None)
        if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number) and obj.ndim == 2:
            return obj
    except Exception:
        return None
    return None


class SeededKMeans(Container):
    """Seeded K-Means barcode classifier widget.

    Implements the Seeded-KMeans algorithm of Basu, Banerjee & Mooney (ICML
    2002): user-provided class prototypes initialise the centroids, then the
    standard K-Means EM loop refines them to absorb within-class biological
    variability. This is a **classifier**, not an unsupervised clusterer —
    the decision rule reduces to Rocchio / Nearest Centroid when iterations
    converge immediately. See README for citations.
    """
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__(layout='vertical')
        # reduce spacing between rows in the main container
        main_layout = self.native.layout()
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = viewer
        self.df_test = None
        self.df_ref = None
        self.dims = []
        self.fig = None
        self.axes = None
        # self.seeds = []
        self.seed_indices: List[int] = []
        self.seed_artists: List[Line2D] = []
        self.selected_class = 1
        self.lasso = None
        self._cid_seed = None
        self.marker_size_pt = 12
        self.drag_threshold_px = 12
        # Single sample-folder input (defaults to the walkthrough sample).
        # `test_folders` / `ref_folders` lists are kept so the existing
        # load_and_plot / run_kmeans code keeps working without rewrite.
        self.sample_folder = FileEdit(
            label='Sample Folder', filter='*', mode='d',
            value=r'J:/Mix16-N-P-260306-DCZ-2-1',
        )
        self.append(self.sample_folder)
        self.test_folders = [self.sample_folder]
        self.ref_folders = []

        _append_section_divider(self,'— ⚙ Filters & clustering parameters —')
        # Intensity Threshold
        row = Container(layout='horizontal')
        row.append(Label(value='Intensity Threshold'))
        self.threshold = FloatSpinBox(min=0, max=1e6, step=1e3, value=2000)
        row.append(self.threshold)
        self.append(row)

        # Number of Clusters
        row = Container(layout='horizontal')
        row.append(Label(value='Number of Clusters'))
        self.n_clusters = SpinBox(min=1, max=50, value=5)
        row.append(self.n_clusters)
        self.append(row)

        # Clustering method — methods other than "KMeans (seeds)" ignore manual seeds.
        row = Container(layout='horizontal')
        row.append(Label(value='Method'))
        self.method = ComboBox(
            choices=[
                'KMeans (seeds)',        # current behaviour: manual seeds as init
                'KMeans++',              # auto smart init, ignores seeds
                'MiniBatchKMeans++',     # faster KMeans++, same API
                'GaussianMixture',       # soft assignment, overlap-friendly
                'Spectral',              # for non-convex cluster shapes
            ],
            value='KMeans (seeds)',
        )
        row.append(self.method)
        self.append(row)

        row = Container(layout='horizontal')
        row.append(Label(value='Localization to cluster'))
        self.loc_choice = ComboBox(choices=['AUTO', 'ALL', 'N', 'M', 'P'], value='AUTO')
        row.append(self.loc_choice)
        self.append(row)

        # --- Outlier detection (marks bad cells as cluster 0 before K-Means) ---
        row = Container(layout='horizontal')
        from magicgui.widgets import CheckBox as _CheckBox
        self.outlier_enable = _CheckBox(text='Auto-detect outliers -> class 0', value=True)
        row.append(self.outlier_enable)
        self.append(row)

        row = Container(layout='horizontal')
        row.append(Label(value='Outlier contamination'))
        self.outlier_contam = FloatSpinBox(min=0.0, max=0.3, step=0.01, value=0.1)
        row.append(self.outlier_contam)
        self.rerun_outlier_btn = PushButton(text='Re-flag outliers')
        self.rerun_outlier_btn.clicked.connect(self.rerun_outliers)
        row.append(self.rerun_outlier_btn)
        self.append(row)

        _append_section_divider(self,'— ⚖ 5D weights for scaling —')
        # Weights for dimensions
        self.weights = {}
        for d in ['G', 'S', 'Int1', 'Int2', 'Int3']:
            row = Container(layout='horizontal')
            row.append(Label(value=f'Weight {d}'))
            if d in ['G', 'S']:
                value = 4.0
            else:
                value = 1.0
            w = FloatSpinBox(min=0.0, max=10.0, step=0.1, value=value)
            row.append(w)
            self.append(row)
            self.weights[d] = w

        _append_section_divider(self,'— 💾 Seeds & distribution —')
        # --- Seeds row: Save + FileEdit (defaults to a discovered seeds file) + Load ---
        self.seed_file_path = FileEdit(
            label='Seeds file', mode='r', filter='*.xlsx',
        )
        self.append(self.seed_file_path)
        seed_btn_row = Container(layout='horizontal')
        self.save_seeds_btn = PushButton(text='Save Seeds')
        self.save_seeds_btn.clicked.connect(self.save_seeds)
        seed_btn_row.append(self.save_seeds_btn)
        self.load_seeds_btn = PushButton(text='Load Seeds')
        self.load_seeds_btn.clicked.connect(self.load_seeds)
        seed_btn_row.append(self.load_seeds_btn)
        self.append(seed_btn_row)

        # --- Distribution row: same pattern (.npz) ---
        self.dist_file_path = FileEdit(
            label='Distribution file', mode='r', filter='*.npz',
        )
        self.append(self.dist_file_path)
        dist_btn_row = Container(layout='horizontal')
        self.save_dist_btn = PushButton(text='Save Distribution')
        self.save_dist_btn.clicked.connect(self.save_distribution)
        dist_btn_row.append(self.save_dist_btn)
        self.load_dist_btn = PushButton(text='Load Distribution')
        self.load_dist_btn.clicked.connect(self.load_distribution)
        dist_btn_row.append(self.load_dist_btn)
        self.append(dist_btn_row)

        # Distribution-overlay expansion. 1.0 = raw convex hull (tight); larger
        # values inflate the hull around each cluster centroid so the region
        # is easier to see as a light background on the scatter plot.
        dist_scale_row = Container(layout='horizontal')
        dist_scale_row.append(Label(value='Distribution expand ×'))
        self.dist_inflate = FloatSpinBox(min=1.0, max=3.0, step=0.05, value=2.0)
        self.dist_inflate.changed.connect(self._redraw_distribution_if_loaded)
        dist_scale_row.append(self.dist_inflate)
        self.dist_apply_btn = PushButton(text='Apply')
        self.dist_apply_btn.clicked.connect(self._redraw_distribution_if_loaded)
        dist_scale_row.append(self.dist_apply_btn)
        self.append(dist_scale_row)

        # Auto-fill both file paths on startup + when the sample folder changes.
        self.sample_folder.changed.connect(self._refresh_file_combos)
        self._refresh_file_combos()

        # Tooltips — hover any control to see what it does.
        _tt(self.n_clusters,
            'Number of barcode classes. Auto-syncs to the row count of '
            'the Seeds file when one is loaded.')
        _tt(self.method,
            '"Seeded K-Means" (Basu 2002) uses your manually-placed seeds '
            'as centroid init and refines with EM. The other methods '
            '(K-Means++, MiniBatchKMeans, Gaussian Mixture, Spectral) '
            'ignore the seeds and cluster blind.')
        _tt(self.outlier_enable,
            'Flag per-class outliers (Isolation Forest) and reassign '
            'them to class 0 (unassigned). Helps weed out dim or '
            'defocused cells without re-editing masks.')
        _tt(self.outlier_contam,
            'Fraction of points per class flagged as outliers. 0.05–0.15 '
            'is typical. Higher = more aggressive weeding.')
        _tt(self.save_seeds_btn,
            'Save the currently-placed seeds to an .xlsx file so you can '
            'reload them in a future session.')
        _tt(self.load_seeds_btn,
            'Load a previously-saved seeds .xlsx. Number of seeds in the '
            'file auto-fills Clusters above.')
        _tt(self.save_dist_btn,
            'Save the current per-class distributions (convex hulls in '
            'feature space) as .npz — so you can reload them as prior '
            'knowledge for manual seeding in future sessions.')
        _tt(self.load_dist_btn,
            'Load previously-saved class distribution overlays as '
            'semi-transparent polygons behind the scatter plot, to '
            'guide seed placement.')
        _tt(self.dist_inflate,
            'Expand loaded convex hulls by this factor so nearby points '
            'still fall inside. 1.0 = raw hull, 2.0 = 2× larger.')
        _tt(self.dist_apply_btn,
            'Redraw the distribution overlay with the current expand '
            'factor.')
        _tt(self.dist_file_path,
            'Path to the saved distribution .npz. Auto-filled from the '
            'sample folder.')

        # Stores {cluster_id (int): {'hull_xy': (M,2) array, 'color_rgb': (r,g,b)}}.
        # Populated by save_distribution / load_distribution; drawn as a background
        # overlay in load_and_plot when present.
        self._distribution_regions: dict = {}


        _append_section_divider(self,'— ▶ Run & save —')
        # Buttons row
        btn_row = Container(layout='horizontal')
        self.load_button = PushButton(text='Read and Plot')
        self.load_button.clicked.connect(self.load_and_plot)
        btn_row.append(self.load_button)
        self.run_button = PushButton(text='Run K-Means')
        self.run_button.clicked.connect(self.run_kmeans)
        _style_process_button(self.run_button)
        btn_row.append(self.run_button)
        self.save_button = PushButton(text='Save Results')
        self.save_button.clicked.connect(self.save_results)
        btn_row.append(self.save_button)
        _tt(self.load_button,
            'Reads FLIM-S.xlsx from the sample folder and plots the 5-D '
            'feature scatter. Required before seeds / Run K-Means.')
        _tt(self.run_button,
            'Run the selected clustering method on the currently-loaded '
            'points. Uses placed seeds as initial centroids when method is '
            '"Seeded K-Means".')
        _tt(self.save_button,
            'Write per-cell class labels to clustered.xlsx in the sample '
            'folder. Do this before clicking Next.')
        _tt(self.rerun_outlier_btn,
            'Re-flag outliers using the current Contamination value, '
            'without re-fitting the clustering.')
        self.append(btn_row)

        # Progress + tip for Save Results (the Excel write + mask drawing can take
        # several seconds; users have run K-Means multiple times per loc before saving).
        self.save_progress = widgets.ProgressBar(label='Save progress', value=0, min=0, max=100)
        self.save_status = Label(value='Ready.')
        self.append(self.save_progress)
        self.append(self.save_status)
        self.append(Label(
            value=(
                'Tip: For multi-localization data (e.g. AUTO over N / M / P),\n'
                'run K-Means for each localization separately, and click\n'
                'Save Results after EACH localization so that cluster_local /\n'
                'cluster_tag accumulate correctly across locs in clustered.xlsx.'
            )
        ))

        _add_next_button(self, viewer)
        _tighten_container(self)

    def _notify(self, msg: str):
        # print(msg)
        napari.utils.notifications.show_info(msg)

    def _append_section_header(self, text: str):
        """Append a small styled header to visually group subsequent rows."""
        lbl = Label(value=text)
        try:
            # Light lavender on napari's dark theme — readable; falls back
            # gracefully on the light theme too.
            lbl.native.setStyleSheet(
                'QLabel {'
                '  color: #CE93D8;'
                '  font-weight: bold;'
                '  font-family: Calibri;'
                '  padding: 4px 4px 2px 4px;'
                '  border-bottom: 1px solid #7B1FA2;'
                '}'
            )
        except Exception:
            pass
        self.append(lbl)

    def _nearest_index_in_5d(self, seed_raw_row: np.ndarray) -> int:
        """
        seed_raw_row: shape [n_dims] in raw units, ordered as self.dims
        Returns nearest point index in current df_test, using current weights+scaler.
        """
        # weights aligned to self.dims
        W = []
        for d in self.dims:
            if d == 'G':
                W.append(self.weights['G'].value)
            elif d == 'S':
                W.append(self.weights['S'].value)
            elif d.startswith('Int 1'):
                W.append(self.weights['Int1'].value)
            elif d.startswith('Int 2'):
                W.append(self.weights['Int2'].value)
            elif d.startswith('Int 3'):
                W.append(self.weights['Int3'].value)
            else:
                W.append(1.0)
        W = np.asarray(W, dtype=float)

        # Scale FIRST, then apply weights — must match _ensure_scaled_ready,
        # otherwise the seed lands in a different space than df_scaled.
        seed_scaled = self.scaler.transform(
            seed_raw_row.astype(float).reshape(1, -1)
        ) * W  # shape (1, D)

        dists = cdist(seed_scaled, self.df_scaled).flatten()
        return int(dists.argmin())

    def load_and_plot(self):
        def find_excels(base):
            if not base or not os.path.isdir(base):
                return None

            files = os.listdir(base)

            # 1) 优先使用已经聚好类的 clustered.xlsx
            clustered_path = os.path.join(base, "clustered.xlsx")
            if os.path.isfile(clustered_path):
                excel = clustered_path
            else:
                # 2) 没有 clustered 的话，再按原逻辑找 FLIM-S / barcode / 其他 xlsx
                excel = None
                for name in (
                        ['FLIM-S.xlsx']
                        + [f for f in files if 'barcode' in f.lower() and f.lower().endswith('.xlsx')]
                        + [f for f in files if f.lower().endswith('.xlsx')]
                ):
                    fp = os.path.join(base, name)
                    if os.path.isfile(fp):
                        excel = fp
                        break

            if not excel:
                return None

            print(f'Found Excel: {excel}')
            df = pd.read_excel(excel)

            required = ['G', 'S']
            if not all(col in df.columns for col in required):
                print(f"Skipping {excel}: missing required columns {required}")
                return None

            df['subfolder'] = os.path.basename(base)
            df['base_folder'] = base

            # ensure Localization exists (upper)
            if 'Localization' not in df.columns:
                df['Localization'] = ''
            df['Localization'] = df['Localization'].fillna('').astype(str).str.upper()

            return df

        # --------- load test ----------
        dfs_test = []
        for fe in self.test_folders:
            path = fe.value
            if not path or not os.path.isdir(path):
                continue
            df = find_excels(path)
            if df is not None and len(df) > 0:
                dfs_test.append(df)

        self.df_test_all = pd.concat(dfs_test, ignore_index=True) if dfs_test else None
        if self.df_test_all is None or len(self.df_test_all) == 0:
            print('No test data loaded.')
            return
        self._notify('Test data loaded.')

        def _norm_loc(x):
            x = '' if x is None else str(x).strip().upper()
            return '' if x in ('NAN', 'NONE') else x

        self.df_test_all['Localization'] = self.df_test_all['Localization'].fillna('').astype(str).str.upper()
        self.df_test_all['Localization'] = self.df_test_all['Localization'].apply(_norm_loc)

        # 建稳定 key：Mask label + FOV + Localization
        # 注意 Mask label 可能是 float/NaN，统一转 int/str
        self.df_test_all['Mask label'] = self.df_test_all['Mask label'].fillna(0)
        try:
            self.df_test_all['Mask label'] = self.df_test_all['Mask label'].astype(int)
        except Exception:
            # 万一是字符串/混合，至少保证稳定
            self.df_test_all['Mask label'] = self.df_test_all['Mask label'].astype(str)

        self.df_test_all['FOV'] = self.df_test_all['FOV'].fillna('').astype(str)

        self.df_test_all['_cell_key'] = (
                self.df_test_all['Mask label'].astype(str) + '|' +
                self.df_test_all['FOV'].astype(str) + '|' +
                self.df_test_all['Localization'].astype(str)
        )

        # 初始化结果列（只做一次）
        for col in ['cluster_local', 'cluster_tag', 'cluster_global']:
            if col not in self.df_test_all.columns:
                self.df_test_all[col] = np.nan if col != 'cluster_tag' else ''

        # --------- load ref (unchanged, but also enforce Localization col if exists) ----------
        dfs_ref = []
        for i, fe in enumerate(self.ref_folders):
            path = fe.value
            if not path or not os.path.isdir(path):
                continue
            df = find_excels(path)
            if df is None or len(df) == 0:
                print(f'Skipping empty or invalid reference folder: {path}')
                continue
            df = df.copy()
            df['class'] = i + 1
            dfs_ref.append(df)

        self.df_ref = pd.concat(dfs_ref, ignore_index=True) if dfs_ref else None
        if self.df_ref is not None:
            self._notify('Reference data loaded.')

        # --------- pick localization to show ----------
        choice = getattr(self, 'loc_choice', None)
        loc = (choice.value if choice is not None else 'AUTO')
        loc = '' if loc is None else str(loc).upper()

        # AUTO 这里我不建议：因为你要“处理完一类再下一类”
        # 所以 AUTO 就当成 ''（无定位）或者直接提示用户选 N/M/P
        if loc == 'AUTO':
            # 若数据里只有一种定位，自动选它；否则提示
            present = set(self.df_test_all['Localization'].unique().tolist())
            present_order = [x for x in ['N', 'M', 'P', ''] if x in present]
            if len(present_order) == 1:
                loc = present_order[0]
            else:
                self._notify(
                    f"AUTO found multiple Localization {present_order}. Please choose N/M/P/'' and re-click Read and Plot.")
                return

        # Filter test by loc for plotting & kmeans
        if loc == '':
            df_test = self.df_test_all[self.df_test_all['Localization'].isin(['', 'NAN'])].copy()
        else:
            df_test = self.df_test_all[self.df_test_all['Localization'] == loc].copy()

        if len(df_test) == 0:
            self._notify(f"No test rows for Localization='{loc}'.")
            return

        # --------- Detect dims ----------
        cols = df_test.columns
        self.dims = ['G', 'S'] + [d for d in ['Int 1/(1-4)', 'Int 2/(1-4)', 'Int 3/(1-4)'] if d in cols]

        # --------- Filter by intensity ----------
        if 'Total intensity' in cols:
            thr = self.threshold.value
            df_test = df_test[df_test['Total intensity'] > thr]
        df_test.dropna(subset=self.dims, inplace=True)

        if len(df_test) == 0:
            self._notify(f"No valid rows after filtering for Localization='{loc}'.")
            return

        # IMPORTANT:
        # self.df_test becomes the CURRENT localization subset, for seed picking + kmeans
        self.df_test = df_test.reset_index(drop=True)
        # 确保子集也有 key
        if '_cell_key' not in df_test.columns:
            df_test['_cell_key'] = (
                    df_test['Mask label'].astype(str) + '|' +
                    df_test['FOV'].astype(str) + '|' +
                    df_test['Localization'].astype(str)
            )
        self.df_test = df_test
        self.current_loc = loc  # remember what we're working on now

        # --------- Prepare weighted and scaled data ----------
        wvals = [self.weights['G'].value, self.weights['S'].value]
        for d in ['Int1', 'Int2', 'Int3']:
            key = f'Int {d[-1]}/(1-4)' if d != 'Int1' else 'Int 1/(1-4)'
            if key in self.dims:
                wvals.append(self.weights[d].value)
        W = np.array(wvals)

        X = self.df_test[self.dims].to_numpy() * W
        self.scaler = StandardScaler()
        Xs = StandardScaler().fit_transform(X)
        self.df_scaled = Xs

        # --------- build pairs for subplots ----------
        pairs = [(self.dims[0], self.dims[1])]
        if len(self.dims) >= 4:
            pairs.append((self.dims[2], self.dims[3]))
        if len(self.dims) == 5:
            pairs.append((self.dims[3], self.dims[4]))
        self.pairs = pairs

        # --------- draw figure ----------
        if self.fig:
            plt.close(self.fig)
        self.fig, axes_arr = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3), constrained_layout=True)
        self.axes = list(np.atleast_1d(axes_arr).flatten())

        if self.df_ref is not None:
            colors_ref = get_color_map(self.df_ref['class'].nunique() + 1)

        for ax, (xd, yd) in zip(self.axes, self.pairs):
            ax.scatter(self.df_test[xd], self.df_test[yd], s=20, color='blue', alpha=0.8)

            # reference overlay (optional)
            if self.df_ref is not None:
                for cls, grp in self.df_ref.groupby('class'):
                    # optionally also filter ref by same loc, if ref has Localization
                    if 'Localization' in grp.columns and loc in ('N', 'M', 'P', ''):
                        if loc == '':
                            grp2 = grp[grp['Localization'].fillna('').astype(str).str.upper().isin(['', 'NAN'])]
                        else:
                            grp2 = grp[grp['Localization'].fillna('').astype(str).str.upper() == loc]
                    else:
                        grp2 = grp

                    if len(grp2) == 0:
                        continue
                    ax.scatter(grp2[xd], grp2[yd], s=30, color=colors_ref[cls], alpha=0.02, label=f"Ref {cls}")

            ax.set_xlabel(xd)
            ax.set_ylabel(yd)
            ax.set_title(f'{xd} vs {yd} (Loc={loc if loc else "None"})')

        # Background overlay of previously saved cluster distribution (if loaded)
        self._draw_distribution_overlay()

        # reconnect click for seed selection
        self._cid_seed = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

        # reset seeds each time we reload a loc
        self.seed_indices = []
        self._notify(f"Plot ready for Localization='{loc}'. Select {self.n_clusters.value} seeds then Run K-Means.")

    def on_click(self, event):
        self.drag_threshold_px = (
                (self.marker_size_pt / 2)
                * self.fig.dpi / 72
        )
        if event.inaxes in self.axes and self.seed_indices:
            ax_idx = self.axes.index(event.inaxes)
            x_dim, y_dim = self.pairs[ax_idx]

            seed_data = self.df_test.iloc[self.seed_indices][[x_dim, y_dim]].to_numpy()
            seed_disp = event.inaxes.transData.transform(seed_data)
            click_disp = np.array([event.x, event.y])
            dists_px = np.linalg.norm(seed_disp - click_disp, axis=1)

            closest = int(dists_px.argmin())
            if dists_px[closest] < self.drag_threshold_px:
                self._dragging_seed = True
                self._drag_ax_idx = ax_idx
                self._drag_artist_idx = closest
                self._drag_artist = self.seed_artists[closest]
                self._cid_motion = self.fig.canvas.mpl_connect(
                    'motion_notify_event', self._on_seed_motion
                )
                return

        if event.inaxes not in self.axes:
            return

        if len(self.seed_indices) >= self.n_clusters.value:
           # remove all old seed markers from
            for art in self.seed_artists:
                art.remove()

            self.seed_indices.clear()
            self.seed_artists.clear()
            self._notify("Seed number exceeded, all seeds cleared. Please select new seeds.")

        ax_idx = self.axes.index(event.inaxes)
        x_dim, y_dim = self.pairs[ax_idx]
        x, y = event.xdata, event.ydata

        #
        coords = self.df_test[[x_dim, y_dim]].to_numpy()
        idx = int(np.argmin(cdist([(x, y)], coords)))
        self.seed_indices.append(idx)
        self._draw_seed(idx)

        #
        # self.seeds.append(self.df_scaled[idx])
        # colors = get_color_map(self.n_clusters.value + 1)
        # seed_idx = len(self.seeds)  # 第几个 seed，从 1 开始
        # color = colors[seed_idx]
        # row = self.df_test.iloc[idx]
        # for ax, (xd, yd) in zip(self.axes, self.pairs):
        #     ax.plot(row[xd], row[yd], '*', markersize=12, color=color)
        # plt.draw()

        if len(self.seed_indices) == self.n_clusters.value:
            self._notify('All seeds selected, time for K-Means clustering!')
            if self._cid_seed is not None:
                self.fig.canvas.mpl_disconnect(self._cid_seed)
                self._cid_seed = None

    def _draw_seed(self, idx: int):
        """Plot a draggable star at the raw-data coords of row idx."""
        row = self.df_test.iloc[idx]
        for ax, (xd, yd) in zip(self.axes, self.pairs):
            artist, = ax.plot(row[xd], row[yd], '*', markersize=self.marker_size_pt,
                              color=get_color_map(self.n_clusters.value + 1)[len(self.seed_indices)])
            artist.set_picker(5)  # enable picking
            self.seed_artists.append(artist)
        self.fig.canvas.draw_idle()
        # connect drag event once
        if not hasattr(self, '_drag_cid'):
            self._drag_press_cid = self.fig.canvas.mpl_connect('pick_event', self._on_seed_press)
            self._drag_release_cid = self.fig.canvas.mpl_connect('button_release_event', self._on_seed_release)

    def _default_io_dir(self) -> str:
        """Default directory for file dialogs — the current sample folder if set."""
        try:
            v = str(self.sample_folder.value) if getattr(self, 'sample_folder', None) else ''
            if v and os.path.isdir(v):
                return v
        except Exception:
            pass
        return ''

    def _refresh_file_combos(self):
        """Populate the seeds / distribution dropdowns.

        Searches the sample folder AND its parent (e.g. the J: drive root), so a
        seeds xlsx dropped anywhere in the workspace is findable.
        """
        base = self._default_io_dir()
        search_dirs: list[str] = []
        if base:
            search_dirs.append(base)
            try:
                parent = str(Path(base).parent)
                if parent and parent not in search_dirs and os.path.isdir(parent):
                    search_dirs.append(parent)
            except Exception:
                pass

        # Collect across all search dirs first, then apply the name filter globally
        # (so a seed-named file in any dir wins over unrelated xlsx files in other dirs).
        all_xlsx: list[str] = []
        all_npz: list[str] = []
        for d in search_dirs:
            try:
                all_xlsx.extend(sorted(glob.glob(os.path.join(d, '*.xlsx'))))
            except Exception:
                pass
            try:
                all_npz.extend(sorted(glob.glob(os.path.join(d, '*.npz'))))
            except Exception:
                pass

        seed_named = [p for p in all_xlsx if 'seed' in os.path.basename(p).lower()]
        seed_choices: list[str] = seed_named if seed_named else all_xlsx

        dist_named = [p for p in all_npz
                      if any(k in os.path.basename(p).lower() for k in ('dist', 'distribution'))]
        dist_choices: list[str] = dist_named if dist_named else all_npz
        # De-duplicate while preserving order (sample folder first)
        def _uniq(xs):
            seen = set(); out = []
            for x in xs:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        seed_choices = _uniq(seed_choices)
        dist_choices = _uniq(dist_choices)

        # Default the FileEdit paths to the best candidates found. User can always
        # click the browse icon on the FileEdit to pick another file.
        try:
            if seed_choices:
                self.seed_file_path.value = seed_choices[0]
        except Exception as e:
            print(f'[refresh seeds path] {e}')
        try:
            if dist_choices:
                self.dist_file_path.value = dist_choices[0]
        except Exception as e:
            print(f'[refresh dist path] {e}')

        summary = (
            f'[SeededKMeans] Seeds files found: {len(seed_choices)} in {search_dirs} -> '
            f'{seed_choices}. Distribution files: {len(dist_choices)} -> {dist_choices}.'
        )
        print(summary, flush=True)
        try:
            self._notify(
                f'Seeds: {os.path.basename(seed_choices[0]) if seed_choices else "(none)"} | '
                f'Distribution: {os.path.basename(dist_choices[0]) if dist_choices else "(none)"}.'
            )
        except Exception:
            pass

    def save_seeds(self):
        if not self.seed_indices:
            self._notify("No seeds to save.")
            return

        default = os.path.join(self._default_io_dir(), 'seeds.xlsx') if self._default_io_dir() else ''
        path, _ = QFileDialog.getSaveFileName(None, "Save seeds to Excel", default, "Excel files (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        # store raw coordinates (all dims)
        seed_df = self.df_test.iloc[self.seed_indices][self.dims].copy()
        seed_df.insert(0, "seed_id", np.arange(1, len(seed_df) + 1))
        seed_df.to_excel(path, index=False)

        self._notify(f"Seeds saved to {path}")
        try:
            self.seed_file_path.value = path
            self._refresh_file_combos()
        except Exception:
            pass

    # ---------- Distribution (per-class convex hull on each pair of dims) ----------
    def save_distribution(self):
        """Save convex-hull polygons per cluster across each plotted (x, y) pair.

        The stored .npz is a dict-like with a single 'regions' pickled object:
          { cluster_id -> { 'X_dim-Y_dim': (M, 2) hull points in data coords } }
        Also stores the list of pair keys so load knows which plots to overlay on.
        """
        if self.df_test is None or 'cluster_local' not in self.df_test.columns:
            self._notify("Run K-Means first so each row has a cluster_local.")
            return
        pairs = getattr(self, 'pairs', None) or [('G', 'S')]

        default = os.path.join(self._default_io_dir(), 'cluster_distribution.npz') if self._default_io_dir() else ''
        path, _ = QFileDialog.getSaveFileName(
            None, "Save cluster distribution", default, "NumPy npz (*.npz)",
        )
        if not path:
            return
        if not path.lower().endswith('.npz'):
            path += '.npz'

        regions = {}
        for cid, grp in self.df_test.groupby('cluster_local'):
            cid = int(cid)
            if cid <= 0 or len(grp) < 3:
                continue
            per_pair = {}
            for xd, yd in pairs:
                if xd not in grp.columns or yd not in grp.columns:
                    continue
                pts = grp[[xd, yd]].to_numpy(dtype=float)
                pts = pts[~np.isnan(pts).any(axis=1)]
                if len(pts) < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                    per_pair[f'{xd}|{yd}'] = pts[hull.vertices]
                except Exception as e:
                    print(f'[save_distribution] {cid} {xd}-{yd}: {e}')
            if per_pair:
                regions[cid] = per_pair

        np.savez(path, regions=np.asarray(regions, dtype=object),
                 pairs=np.asarray([f'{a}|{b}' for a, b in pairs], dtype=object))
        self._notify(f'Distribution saved to {path} (clusters: {sorted(regions.keys())})')
        try:
            self.dist_file_path.value = path
            self._refresh_file_combos()
        except Exception:
            pass

    def load_distribution(self):
        """Load a previously saved distribution and render it as a light background."""
        picked = ''
        try:
            picked = str(self.dist_file_path.value) if getattr(self, 'dist_file_path', None) else ''
        except Exception:
            picked = ''
        if picked and os.path.isfile(picked):
            path = picked
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Load cluster distribution", self._default_io_dir(), "NumPy npz (*.npz)",
            )
        if not path:
            return
        try:
            data = np.load(path, allow_pickle=True)
            regions = data['regions'].item()
        except Exception as e:
            self._notify(f'Load distribution failed: {e}')
            return
        self._distribution_regions = {int(k): v for k, v in regions.items()}
        self._notify(f'Loaded distribution for clusters {sorted(self._distribution_regions.keys())}. '
                     'Click "Read and Plot" to overlay.')
        # Use the redraw path so any polygons from a previously-loaded
        # distribution (e.g. when user loads N then P) are cleared first
        # instead of stacking on top of each other.
        if self.fig is not None and self.axes is not None:
            self._redraw_distribution_if_loaded()

    def _draw_distribution_overlay(self):
        """Paint the stored cluster hulls as light filled polygons on each axis.

        Inflate factor comes from self.dist_inflate (UI) — larger = more visible.
        """
        if not self._distribution_regions or self.axes is None or not getattr(self, 'pairs', None):
            return
        from matplotlib.patches import Polygon as _MplPolygon
        try:
            inflate = float(self.dist_inflate.value)
        except Exception:
            inflate = 1.5
        max_cid = max(self._distribution_regions.keys()) if self._distribution_regions else 0
        colors = get_color_map(max(max_cid + 1, 2))
        for ax, (xd, yd) in zip(self.axes, self.pairs):
            key = f'{xd}|{yd}'
            for cid, per_pair in self._distribution_regions.items():
                if key not in per_pair:
                    continue
                pts = np.asarray(per_pair[key], dtype=float)
                if len(pts) < 3:
                    continue
                centroid = pts.mean(axis=0)
                pts_inflated = centroid + (pts - centroid) * inflate
                color = tuple(colors[min(cid, len(colors) - 1)])
                poly = _MplPolygon(
                    pts_inflated, closed=True, facecolor=color, edgecolor=color,
                    alpha=0.18, linewidth=1.2, zorder=0,
                )
                ax.add_patch(poly)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass

    def _redraw_distribution_if_loaded(self, *_args, **_kwargs):
        """Clear existing overlay patches and redraw with the new inflate factor."""
        if not self._distribution_regions or self.axes is None:
            return
        from matplotlib.patches import Polygon as _MplPolygon
        for ax in self.axes:
            # Remove only Polygon patches (our overlay); keep scatter markers, etc.
            for p in list(ax.patches):
                if isinstance(p, _MplPolygon):
                    try:
                        p.remove()
                    except Exception:
                        pass
        self._draw_distribution_overlay()

    def _ensure_scaled_ready(self) -> bool:
        """Ensure df_test, dims, weights, scaler, df_scaled are ready for nearest-neighbor mapping."""
        if self.df_test is None or self.df_test.empty:
            self._notify("No test data. Please click 'Read and Plot' first.")
            return False
        if not hasattr(self, "dims") or not self.dims:
            self._notify("Dims not ready. Please click 'Read and Plot' first.")
            return False

        # build weights aligned to dims
        W = []
        for d in self.dims:
            if d == "G":
                W.append(self.weights["G"].value)
            elif d == "S":
                W.append(self.weights["S"].value)
            elif d.startswith("Int 1"):
                W.append(self.weights["Int1"].value)
            elif d.startswith("Int 2"):
                W.append(self.weights["Int2"].value)
            elif d.startswith("Int 3"):
                W.append(self.weights["Int3"].value)
            else:
                W.append(1.0)
        W = np.asarray(W, dtype=float)

        # CRITICAL: normalise FIRST, then multiply by weights. If we did
        # (X * W) -> StandardScaler, the scaler would divide by W*std and the
        # weight cancels exactly, meaning the sliders have no effect on KMeans.
        X_raw = self.df_test[self.dims].to_numpy(dtype=float)

        if not hasattr(self, "scaler") or not hasattr(self.scaler, "mean_"):
            self.scaler = StandardScaler()
            X_std = self.scaler.fit_transform(X_raw)
        else:
            if not hasattr(self, "df_scaled") or (self.df_scaled is None) or (len(self.df_scaled) != len(self.df_test)):
                X_std = self.scaler.fit_transform(X_raw)
            else:
                # Re-fit so we always use a fresh scaler (caller clears it to
                # force rebuild; but keep the defensive path here too).
                X_std = self.scaler.fit_transform(X_raw)

        # Apply weights AFTER normalisation so they actually affect the distance.
        self.df_scaled = X_std * W
        return True

    def load_seeds(self):
        if not self._ensure_scaled_ready():
            return
        picked = ''
        try:
            picked = str(self.seed_file_path.value) if getattr(self, 'seed_file_path', None) else ''
        except Exception:
            picked = ''
        if picked and os.path.isfile(picked):
            path = picked
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Load seeds from Excel", self._default_io_dir(), "Excel files (*.xlsx *.xls)",
            )
        if not path:
            self._notify("No file selected for loading seeds.")
            return

        df = pd.read_excel(path)
        # find available dims in file
        dims_in_file = [d for d in self.dims if d in df.columns]
        if len(dims_in_file) < 2:
            self._notify(f"Seed file missing required dims. Need at least 2 of: {self.dims}")
            return

        # clear old artists
        for art in self.seed_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.seed_artists.clear()
        self.seed_indices.clear()

        # Proceed with whatever dim columns the seed file has in common with
        # the current data. Nearest-neighbor is done in weighted raw space over
        # that intersection — not as precise as the full-D scaled lookup, but
        # handles the common "seed saved with only G,S" case without bailing.
        rows = df[dims_in_file].to_numpy(dtype=float)
        if dims_in_file != self.dims:
            self._notify(
                f"Seed file dims={dims_in_file}, current dims={self.dims}. "
                f"Matching on intersection."
            )

        # weights for the intersection dims (same mapping as _nearest_index_in_5d)
        def _weight_for_dim(d: str) -> float:
            if d == 'G':
                return float(self.weights['G'].value)
            if d == 'S':
                return float(self.weights['S'].value)
            if d.startswith('Int 1'):
                return float(self.weights['Int1'].value)
            if d.startswith('Int 2'):
                return float(self.weights['Int2'].value)
            if d.startswith('Int 3'):
                return float(self.weights['Int3'].value)
            return 1.0

        W_sub = np.asarray([_weight_for_dim(d) for d in dims_in_file], dtype=float)
        test_xy = self.df_test[dims_in_file].to_numpy(dtype=float) * W_sub

        # IMPORTANT: sync n_clusters BEFORE drawing seeds. _draw_seed indexes into
        # a color table sized by `n_clusters + 1`; if we loaded more seeds than K
        # the draw loop would crash on the (K+2)-th seed and leave seeds partial.
        n_loaded = len(rows)
        old_k = int(self.n_clusters.value)
        if n_loaded > 0 and old_k != n_loaded:
            try:
                self.n_clusters.value = n_loaded
            except Exception as e:
                self._notify(f"Could not auto-update K to {n_loaded}: {e}")

        for r in rows:
            r_w = r * W_sub
            dists = np.linalg.norm(test_xy - r_w, axis=1)
            idx = int(dists.argmin())
            self.seed_indices.append(idx)
            self._draw_seed(idx)

        if n_loaded > 0 and old_k != n_loaded:
            self._notify(
                f"Loaded {n_loaded} seeds; auto-updated Number of Clusters: {old_k} -> {n_loaded}."
            )
        else:
            self._notify(f"Loaded {n_loaded} seeds (snapped to nearest points).")

    def _on_seed_press(self, event):
        # begin dragging one of our seed markers
        if event.artist in self.seed_artists:
            self._dragging_artist = event.artist
            self._drag_orig_xy = (event.mouseevent.xdata, event.mouseevent.ydata)
            self._drag_artist_idx = self.seed_artists.index(event.artist)
            self._cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_seed_motion)

    def _on_seed_motion(self, event):
        if not hasattr(self, '_dragging_artist'): return
        self._dragging_artist.set_data(event.xdata, event.ydata)
        self.fig.canvas.draw_idle()

    def _on_seed_release(self, event):
        # finish drag: snap to nearest data‐point
        if not hasattr(self, '_dragging_artist'): return
        artist = self._dragging_artist
        ax_idx = next(i for i, a in enumerate(self.axes) if a == artist.axes)
        xd, yd = self.pairs[ax_idx]
        # find nearest index in raw df_test
        coords = self.df_test[[xd, yd]].to_numpy()
        dists = np.linalg.norm(coords - np.array([event.xdata, event.ydata]), axis=1)
        new_idx = int(dists.argmin())
        # update our records
        self.seed_indices[self._drag_artist_idx] = new_idx
        # redraw that star at the true location
        row = self.df_test.iloc[new_idx]
        artist.set_data(row[xd], row[yd])
        # cleanup
        self.fig.canvas.mpl_disconnect(self._cid_motion)
        delattr(self, '_dragging_artist')
        self._notify(f"Seed {self._drag_artist_idx + 1} reassigned to cell #{new_idx}")
        self.fig.canvas.draw_idle()

    def _clear_seed_mode(self):
        """Exit seed-selection mode safely (magicgui Container overrides __delattr__).

        Keeps `self.seed_indices` populated so the user can re-run K-Means after
        tweaking weights / K / outlier settings without having to re-click seeds.
        Seed indices are row-indices into `df_test` and stay valid as long as
        the data hasn't been reloaded.
        """
        # 1) remove seed artists (visible markers on the plot)
        for art in list(self.seed_artists):
            try:
                art.remove()
            except Exception:
                pass
        self.seed_artists.clear()
        # NOTE: intentionally NOT clearing self.seed_indices — they are cheap,
        # valid for subsequent re-runs, and get reset by load_and_plot anyway.

        # 2) disconnect seed click handler
        if getattr(self, "_cid_seed", None) is not None and getattr(self, "fig", None) is not None:
            try:
                self.fig.canvas.mpl_disconnect(self._cid_seed)
            except Exception:
                pass
            self._cid_seed = None

        # 3) disconnect drag handlers if any
        if getattr(self, "fig", None) is not None:
            for name in ["_drag_press_cid", "_drag_release_cid", "_cid_motion"]:
                cid = getattr(self, name, None)
                if cid is not None:
                    try:
                        self.fig.canvas.mpl_disconnect(cid)
                    except Exception:
                        pass
                # IMPORTANT: don't delattr on magicgui Container
                setattr(self, name, None)

        # 4) remove any dragging state flags
        for name in ["_dragging_artist", "_drag_orig_xy", "_drag_artist_idx",
                     "_dragging_seed", "_drag_ax_idx", "_drag_artist"]:
            if hasattr(self, name):
                # again: avoid delattr
                setattr(self, name, None)

    def _get_loc_run_list(self):
        """Return localization list to run in order."""
        # make sure column exists
        if self.df_test is None or len(self.df_test) == 0:
            return []
        if 'Localization' not in self.df_test.columns:
            self.df_test['Localization'] = ''
        self.df_test['Localization'] = self.df_test['Localization'].fillna('').astype(str).str.upper()

        choice = getattr(self, 'loc_choice', None)
        choice = choice.value if choice is not None else 'AUTO'
        choice = '' if choice is None else choice
        choice = str(choice).upper()

        if choice in ('N', 'M', 'P', ''):
            return [choice]

        # AUTO: detect present and order N,M,P,''
        present = set(self.df_test['Localization'].unique().tolist())
        order = ['N', 'M', 'P', '']
        locs = [x for x in order if x in present]
        if not locs:
            locs = ['']
        return locs

    def _load_existing_clustered_max(self, folder: str) -> int:
        """If clustered.xlsx exists, return max global cluster id; else 0."""
        path = os.path.join(folder, 'clustered.xlsx')
        if not os.path.isfile(path):
            return 0
        try:
            old = pd.read_excel(path)
            if 'cluster' in old.columns and len(old) > 0:
                m = np.nanmax(old['cluster'].to_numpy())
                if np.isfinite(m):
                    return int(m)
        except Exception:
            pass
        return 0

    def _calc_offset_for_loc(self, folder: str, loc: str) -> int:
        """
        Compute starting offset for this folder+loc:
        - if clustered.xlsx exists, start from its max cluster
        - else start from 0
        This allows '补全'：先跑N再跑M，会自动往后延。
        """
        return self._load_existing_clustered_max(folder)

    def run_kmeans(self):
        if self.df_test is None or len(self.df_test) == 0:
            self._notify("No test data loaded.")
            return

        locs_to_run = self._get_loc_run_list()
        if not locs_to_run:
            self._notify("No localization found to run.")
            return

        # Force a rebuild of df_scaled so re-runs after knob changes (weights,
        # method) use fresh values. _ensure_scaled_ready caches the scaler so a
        # simple re-call would reuse the old fit.
        if hasattr(self, 'scaler'):
            try:
                del self.scaler
            except Exception:
                self.scaler = None
        self.df_scaled = None
        if not self._ensure_scaled_ready():
            return

        method = str(self.method.value) if getattr(self, 'method', None) else 'KMeans (seeds)'
        seeds_required = method == 'KMeans (seeds)'

        # Seed-count sanity check only matters for the seed-initialised method.
        if seeds_required and len(self.seed_indices) != self.n_clusters.value:
            self._notify(
                f"Please select {self.n_clusters.value} seeds before running K-Means, "
                f"currently {len(self.seed_indices)} selected."
            )
            return

        # ensure helper cols exist
        if 'Localization' not in self.df_test.columns:
            self.df_test['Localization'] = ''
        self.df_test['Localization'] = self.df_test['Localization'].fillna('').astype(str).str.upper()

        # init seeds from current scaled array (IMPORTANT: must match subset indexing!)
        # We'll run per folder+loc subset; need seeds selected within that subset figure.
        # So here: assume user selected seeds on current plot, which corresponds to current df_test order.
        # We'll keep that, but for each subset we must re-map seed indices to subset indices.
        # Therefore: enforce user only runs one loc at a time OR use AUTO but require reselect seeds for each loc.
        # To keep your UX: when AUTO has multiple locs, we will reuse same seeds only if subset size == full size.
        # Otherwise we will stop and ask user to rerun for that loc with seeds selected.
        # (这个是必须的，否则 seed_indices 对不上子集)
        #
        # 我给你更实用的策略：当 AUTO 且 locs>1：
        #   每个 loc 单独跑，要求用户先选 loc 再选 seeds 再 run。
        #   所以 AUTO 这里直接提示并返回。
        if getattr(self, 'loc_choice', None) is not None:
            choice = str(self.loc_choice.value).upper()
            if choice == 'AUTO' and len(locs_to_run) > 1:
                self._notify(
                    f"AUTO detected multiple localizations {locs_to_run}. "
                    "For seed-based KMeans, please run one localization at a time:\n"
                    "Select Localization = N (pick seeds) -> Run, then M -> Run, then P -> Run."
                )
                return

        # run one loc at a time (seed indices valid for current df ordering)
        loc = locs_to_run[0]

        # subset rows
        if loc == '':
            sel = self.df_test['Localization'].isin(['', 'NAN'])
        else:
            sel = self.df_test['Localization'] == loc
        df_sub = self.df_test.loc[sel].copy()
        if len(df_sub) == 0:
            self._notify(f"No rows for Localization='{loc}'.")
            return

        # Map global seed indices -> subset indices
        idx_global = np.where(sel.to_numpy())[0]  # positions in df_test that are in subset
        idx_global_set = set(idx_global.tolist())

        # Build subset scaled features (used by all methods)
        Xs_sub = self.df_scaled[idx_global, :]

        init_seeds = None
        if seeds_required:
            seed_global = [i for i in self.seed_indices if i in idx_global_set]
            if len(seed_global) != self.n_clusters.value:
                self._notify(
                    f"Seeds must be selected within Localization='{loc}'. "
                    f"Currently selected {len(seed_global)} seeds in this localization, "
                    f"need {self.n_clusters.value}. Please reselect seeds after filtering."
                )
                return
            pos_map = {g: j for j, g in enumerate(idx_global.tolist())}
            seed_sub_idx = [pos_map[g] for g in seed_global]
            init_seeds = np.vstack([Xs_sub[i] for i in seed_sub_idx])

        # clear seed mode (your original)
        self._clear_seed_mode()

        K = int(self.n_clusters.value)
        self._notify(f"Running method='{method}' with K={K} on {len(Xs_sub)} points...")
        try:
            if method == 'KMeans (seeds)':
                km = KMeans(n_clusters=K, init=init_seeds, n_init=1)
                labels0 = km.fit_predict(Xs_sub)
            elif method == 'KMeans++':
                km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0)
                labels0 = km.fit_predict(Xs_sub)
            elif method == 'MiniBatchKMeans++':
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(
                    n_clusters=K, init='k-means++', n_init=10,
                    batch_size=min(256, max(32, len(Xs_sub) // 4)),
                    random_state=0,
                )
                labels0 = km.fit_predict(Xs_sub)
            elif method == 'GaussianMixture':
                from sklearn.mixture import GaussianMixture
                gmm = GaussianMixture(
                    n_components=K, covariance_type='full',
                    n_init=3, random_state=0,
                )
                gmm.fit(Xs_sub)
                labels0 = gmm.predict(Xs_sub)
            elif method == 'Spectral':
                from sklearn.cluster import SpectralClustering
                sc = SpectralClustering(
                    n_clusters=K, affinity='nearest_neighbors',
                    assign_labels='kmeans', random_state=0,
                )
                labels0 = sc.fit_predict(Xs_sub)
            else:
                self._notify(f"Unknown method '{method}', falling back to KMeans++.")
                km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0)
                labels0 = km.fit_predict(Xs_sub)
        except Exception as e:
            self._notify(f"Clustering failed with method='{method}': {e}")
            traceback.print_exc()
            return

        cluster_local = labels0.astype(int) + 1  # 1..K

        # Cache the pre-outlier clustering so "Re-flag outliers" can re-apply a
        # new contamination value without re-fitting the full clusterer.
        self._last_run = {
            'loc': loc,
            'Xs_sub': Xs_sub.copy(),
            'labels0_base': labels0.astype(int).copy(),
            'df_sub_index': df_sub.index.copy(),
            'K': K,
        }

        use_outlier = bool(getattr(self, 'outlier_enable', None) and self.outlier_enable.value)
        contam = float(getattr(self, 'outlier_contam', None).value) if getattr(self, 'outlier_contam', None) else 0.0
        if use_outlier and contam > 0:
            try:
                from sklearn.ensemble import IsolationForest
                flagged = 0
                K = int(self.n_clusters.value)
                for k in range(1, K + 1):
                    idx_k = np.where(cluster_local == k)[0]
                    if len(idx_k) < 10:  # too few points to reliably estimate
                        continue
                    iso = IsolationForest(
                        contamination=min(contam, 0.49), random_state=0, n_estimators=100,
                    )
                    pred = iso.fit_predict(Xs_sub[idx_k])  # +1 inlier, -1 outlier
                    out_idx = idx_k[pred == -1]
                    cluster_local[out_idx] = 0
                    flagged += len(out_idx)
                self._notify(
                    f'Per-class outliers flagged as 0: {flagged} / {len(cluster_local)} '
                    f'(contamination={contam:.2f})'
                )
            except Exception as e:
                self._notify(f'Per-class outlier detection failed: {e}. Keeping all points.')

        df_sub['cluster_local'] = cluster_local
        # tag 只是显示/导出用；cluster 0 = outlier
        def _fmt_tag(x, loc=loc):
            n = int(x)
            if n == 0:
                return 'Outlier'
            return f"{loc}{n}" if loc in ('N', 'M', 'P') else f"C{n}"
        df_sub['cluster_tag'] = df_sub['cluster_local'].apply(_fmt_tag)

        # write back to main df_test
        # only update subset rows; keep other localizations untouched
        self.df_test.loc[df_sub.index, 'cluster_local'] = df_sub['cluster_local'].astype(int)
        self.df_test.loc[df_sub.index, 'cluster_tag'] = df_sub['cluster_tag'].astype(str)

        # 把当前 loc 的结果同步回全量表，保证切换 loc 后不丢
        m = df_sub[['_cell_key', 'cluster_local', 'cluster_tag']].copy()
        m['cluster_local'] = m['cluster_local'].astype(int)

        # 用 key 做 map 写回
        map_local = dict(zip(m['_cell_key'], m['cluster_local']))
        map_tag = dict(zip(m['_cell_key'], m['cluster_tag']))

        self.df_test_all.loc[self.df_test_all['_cell_key'].isin(map_local.keys()), 'cluster_local'] = \
            self.df_test_all.loc[self.df_test_all['_cell_key'].isin(map_local.keys()), '_cell_key'].map(map_local)

        self.df_test_all.loc[self.df_test_all['_cell_key'].isin(map_tag.keys()), 'cluster_tag'] = \
            self.df_test_all.loc[self.df_test_all['_cell_key'].isin(map_tag.keys()), '_cell_key'].map(map_tag)
        colors = get_color_map(self.n_clusters.value + 1)  # 0..K

        for ax, (xd, yd) in zip(self.axes, self.pairs):
            ax.clear()
            # 先画 0（如果你有）
            for k in range(0, self.n_clusters.value + 1):
                selk = (self.df_test['cluster_local'].fillna(0).astype(int) == k)
                ax.scatter(self.df_test.loc[selk, xd], self.df_test.loc[selk, yd],
                           color=colors[k], s=20)
            ax.set_xlabel(xd);
            ax.set_ylabel(yd)
            ax.set_title(f'{xd} vs {yd} (Loc={self.current_loc if self.current_loc else "None"})')

        self.fig.canvas.draw_idle()

        cid1 = self.fig.canvas.mpl_connect('button_press_event', self._on_manual_click)
        cid2 = self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)
        self._notify(
            f"Clustering done for Localization='{loc}'. "
            "Now you can manually reassign points by shift+click for lasso-selector activation on plot, "
            "using keys 1-9, a-z for classes 1-35 to assign classes."
        )

    def rerun_outliers(self):
        """Re-apply per-class outlier detection on the last K-Means result,
        using the current 'Outlier contamination' value. Does NOT re-fit clusters.
        """
        cache = getattr(self, '_last_run', None)
        if not cache or self.df_test is None:
            self._notify("Run K-Means first before re-flagging outliers.")
            return

        Xs_sub = cache['Xs_sub']
        labels0 = cache['labels0_base']
        loc = cache['loc']
        df_sub_idx = cache['df_sub_index']
        K = cache['K']

        cluster_local = labels0.astype(int) + 1
        contam = float(self.outlier_contam.value)
        use_outlier = bool(self.outlier_enable.value)
        if use_outlier and contam > 0:
            try:
                from sklearn.ensemble import IsolationForest
                flagged = 0
                for k in range(1, K + 1):
                    idx_k = np.where(cluster_local == k)[0]
                    if len(idx_k) < 10:
                        continue
                    iso = IsolationForest(
                        contamination=min(contam, 0.49),
                        random_state=0, n_estimators=100,
                    )
                    pred = iso.fit_predict(Xs_sub[idx_k])
                    out_idx = idx_k[pred == -1]
                    cluster_local[out_idx] = 0
                    flagged += len(out_idx)
                self._notify(
                    f'Re-flag outliers: {flagged} / {len(cluster_local)} flagged '
                    f'(contamination={contam:.2f})'
                )
            except Exception as e:
                self._notify(f'Outlier detection failed: {e}')
                return

        # Build tags and write back (mirrors the run_kmeans writeback block)
        def _fmt_tag(x, loc=loc):
            n = int(x)
            if n == 0:
                return 'Outlier'
            return f"{loc}{n}" if loc in ('N', 'M', 'P') else f"C{n}"

        tags = [_fmt_tag(x) for x in cluster_local]
        self.df_test.loc[df_sub_idx, 'cluster_local'] = cluster_local
        self.df_test.loc[df_sub_idx, 'cluster_tag'] = tags

        # Sync back into df_test_all by _cell_key
        try:
            keys = self.df_test.loc[df_sub_idx, '_cell_key'].values
            map_local = dict(zip(keys, cluster_local.astype(int)))
            map_tag = dict(zip(keys, tags))
            in_mask = self.df_test_all['_cell_key'].isin(map_local.keys())
            self.df_test_all.loc[in_mask, 'cluster_local'] = (
                self.df_test_all.loc[in_mask, '_cell_key'].map(map_local)
            )
            self.df_test_all.loc[in_mask, 'cluster_tag'] = (
                self.df_test_all.loc[in_mask, '_cell_key'].map(map_tag)
            )
        except Exception as e:
            print(f'[rerun_outliers] df_test_all sync: {e}')

        try:
            self._redraw_clusters()
        except Exception as e:
            print(f'[rerun_outliers] redraw: {e}')

    def _on_keypress(self, event):
        key = event.key
        if not key:
            return

        # 1–9
        if key.isdigit():
            val = int(key)

        # a–z → 10–35
        elif len(key) == 1 and 'a' <= key <= 'z':
            val = ord(key) - ord('a') + 10

        else:
            return

        if 0 <= val <= self.n_clusters.value:
            self.selected_class = val
            self._notify(f"Selected class {val} (for next assignment)")

    def _on_manual_click(self, event):
        #
        if event.inaxes not in self.axes:
            return

        ax_idx = self.axes.index(event.inaxes)
        x_dim, y_dim = self.pairs[ax_idx]
        ix, iy = event.xdata, event.ydata

        # debug
        print(f"[Manual] click at ({ix:.2f},{iy:.2f}) on subplot #{ax_idx} dims=({x_dim},{y_dim})")

        coords = self.df_test[[x_dim, y_dim]].to_numpy()

        #
        if event.key == 'control' and event.button == 1:
            dists = cdist([(ix, iy)], coords).flatten()
            idx = int(dists.argmin())
            print(f" → nearest idx={idx}, coord={coords[idx]}, dist={dists[idx]:.3f}")
            print(f" original cluster: {self.df_test.at[idx, 'cluster']}")
            self.df_test.at[idx, 'cluster_local'] = int(self.selected_class)

            # tag 也更新一下（可选）
            loc = self.current_loc
            if loc in ('N', 'M', 'P') and self.selected_class > 0:
                self.df_test.at[idx, 'cluster_tag'] = f"{loc}{int(self.selected_class)}"
            elif self.selected_class > 0:
                self.df_test.at[idx, 'cluster_tag'] = f"C{int(self.selected_class)}"
            else:
                self.df_test.at[idx, 'cluster_tag'] = ""

            # 同步回全量（用 key）
            key = self.df_test.at[idx, '_cell_key']
            m = (self.df_test_all['_cell_key'] == key)
            self.df_test_all.loc[m, 'cluster_local'] = int(self.selected_class)
            self.df_test_all.loc[m, 'cluster_tag'] = self.df_test.at[idx, 'cluster_tag']

            self.idx_changed = idx
            self._redraw_clusters()
            self._notify(f"Point {idx}→cluster {self.selected_class}")

        elif event.key == 'shift' and event.button == 1:
            self._notify("Starting lasso selection. Click to draw polygon.")
            # start a LassoSelector on this axis
            if self.lasso:
                self.lasso.disconnect_events()
            ax = event.inaxes
            self.lasso = LassoSelector(ax, onselect=self._on_lasso_select)

    def _on_lasso_select(self, verts):
        path = MplPath(verts)
        ax = self.lasso.ax
        ax_idx = self.axes.index(ax)
        x_dim, y_dim = self.pairs[ax_idx]

        pts = self.df_test[[x_dim, y_dim]].to_numpy()
        mask = path.contains_points(pts)

        # 更新当前子集
        self.df_test.loc[mask, 'cluster_local'] = int(self.selected_class)
        loc = self.current_loc
        if int(self.selected_class) > 0:
            tag = (f"{loc}{int(self.selected_class)}" if loc in ('N', 'M', 'P') else f"C{int(self.selected_class)}")
        else:
            tag = ""
        self.df_test.loc[mask, 'cluster_tag'] = tag

        # 同步回全量（按 key 批量）
        keys = self.df_test.loc[mask, '_cell_key'].tolist()
        m = self.df_test_all['_cell_key'].isin(keys)
        self.df_test_all.loc[m, 'cluster_local'] = int(self.selected_class)
        self.df_test_all.loc[m, 'cluster_tag'] = tag

        self._lasso_cleanup()
        self._redraw_clusters()

    def _lasso_cleanup(self):
        self.lasso.disconnect_events()
        self.lasso = None

    def _redraw_clusters(self):
        try:
            print(f'idx_changed: {self.idx_changed}')
            print(f'cluster: {self.df_test.at[self.idx_changed, "cluster"]}')
        except AttributeError:
            pass
        n_total = self.n_clusters.value + 1
        colors = get_color_map(n_total)

        for ax, (xd, yd) in zip(self.axes, self.pairs):
            ax.clear()
            for k in range(n_total):
                sel = (self.df_test['cluster_local'].fillna(0).astype(int) == k)
                ax.scatter(self.df_test.loc[sel, xd], self.df_test.loc[sel, yd],
                           color=colors[k], s=20)
            ax.set_xlabel(xd);
            ax.set_ylabel(yd)
            ax.set_title(f'{xd} vs {yd} (Loc={self.current_loc if self.current_loc else "None"})')

        self.fig.canvas.draw_idle()

    def _select_npy_file(self, initial_folder):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(
            None,
            "Select segmentation .npy file",
            initial_folder,
            "NumPy files (*.npy);;All Files (*)",
            options=options
        )
        return file or None

    def _draw_tags_on_rgb(self, rgb: np.ndarray, mask: np.ndarray, id_to_tag: dict[int, str]):
        from PIL import Image, ImageDraw, ImageFont
        from skimage.measure import label as cc_label, regionprops

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        # try:
        #     font = ImageFont.load_default()
        # except Exception:
        #     font = None
        font = ImageFont.truetype("arial.ttf", size=24)
        ids = [int(i) for i in np.unique(mask) if i != 0]

        for cid in ids:
            tag = id_to_tag.get(int(cid), "")
            if not tag:
                continue

            binary = (mask == cid)
            # 连通域分解：每个岛都标
            cc = cc_label(binary, connectivity=1)
            props = regionprops(cc)

            for rp in props:
                # rp.centroid 是 (row, col)，但可能落在边界外/空洞附近（极少数）
                y = int(round(rp.centroid[0]))
                x = int(round(rp.centroid[1]))

                # 确保点在该连通域里：不在的话就用该域的第一个像素
                minr, minc, maxr, maxc = rp.bbox
                y = min(max(y, minr), maxr - 1)
                x = min(max(x, minc), maxc - 1)
                if cc[y, x] != rp.label:
                    coords = rp.coords  # (N,2)
                    y, x = int(coords[0, 0]), int(coords[0, 1])

                # 白字 + 黑描边（每个都标，不再跳过“小的”）
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    draw.text((x + dx, y + dy), tag, fill=(0, 0, 0), font=font)
                draw.text((x, y), tag, fill=(255, 255, 255), font=font)

        return np.array(img)

    def _process_masks_for_folder(self, folder, df_grp):
        int_folder = _find_intensity_dir(folder)
        if not int_folder:
            self._notify(f"No intensity folder under {folder}, skipping.")
            return

        # NOTE: now require cluster_global (final 1..T id)
        need_cols = ['FOV', 'Mask label', 'cluster_global']
        if not all(k in df_grp.columns for k in need_cols):
            self._notify(
                f"Missing required columns in data for {folder}. Need {need_cols}."
            )
            return

        # total_classes for colormap: use max cluster_global + 1
        try:
            max_cluster = int(np.nanmax(df_grp['cluster_global'].to_numpy()))
        except Exception:
            max_cluster = 0
        total_classes = max_cluster + 1  # include 0

        colors = get_color_map(total_classes)  # RGB 0-1, index 0 gray by design
        mask_color_map = np.array(colors, dtype=np.float32)
        mask_color_map[0] = (0, 0, 0)  # keep 0 as black background (your preference)

        # build global id -> tag mapping (for text overlay)
        id_to_tag = {}
        if 'cluster_tag' in df_grp.columns:
            tmp = df_grp.copy()
            tmp = tmp.dropna(subset=['cluster_global'])
            # group by global id
            for cid, sub in tmp.groupby('cluster_global'):
                try:
                    cid_int = int(cid)
                except Exception:
                    continue
                tag = ""
                if 'cluster_tag' in sub.columns and len(sub) > 0:
                    tag = str(sub['cluster_tag'].iloc[0])
                id_to_tag[cid_int] = tag

        for fov in df_grp['FOV'].unique():
            df_fov_all = df_grp[df_grp['FOV'] == fov].copy()

            # make sure Localization exists
            if 'Localization' not in df_fov_all.columns:
                df_fov_all['Localization'] = ''

            # Determine which locs exist for this fov (for picking seg files)
            locs = sorted({str(x).upper() for x in df_fov_all['Localization'].fillna('').tolist()})
            loc_order = ['N', 'M', 'P', '']
            locs = [x for x in loc_order if x in locs]
            if not locs:
                locs = ['']

            mask_global = None

            for loc in locs:
                if loc != '':
                    df_fov = df_fov_all[df_fov_all['Localization'].fillna('').astype(str).str.upper() == loc].copy()
                else:
                    df_fov = df_fov_all[
                        df_fov_all['Localization'].fillna('').astype(str).str.upper().isin(['', 'NAN'])].copy()

                if len(df_fov) == 0:
                    continue

                seg_path = _select_seg_file_by_loc(int_folder, fov, loc)
                if not seg_path or not os.path.isfile(seg_path):
                    self._notify(f"Need segmentation .npy for FOV '{fov}' loc '{loc}' in {int_folder}. Prompting user.")
                    seg_path = self._select_npy_file(int_folder)
                    if not seg_path:
                        self._notify(f"Skipping FOV {fov} loc {loc}: no segmentation file provided.")
                        continue

                mask_cp = _load_cellpose_masks(seg_path)
                if mask_cp is None:
                    self._notify(f"Failed to load 'masks' from '{seg_path}', skipping FOV {fov} loc {loc}.")
                    continue

                # init global mask shape
                if mask_global is None:
                    mask_global = np.zeros_like(mask_cp, dtype=np.uint16)

                # erosion (unchanged)
                mask_cp_eroded = np.zeros_like(mask_cp)
                erosion_disk = disk(1)
                for ml in np.unique(mask_cp):
                    if ml == 0:
                        continue
                    mask_region = mask_cp == ml
                    props = regionprops(mask_region.astype(int))
                    if not props:
                        continue
                    prop = props[0]
                    minr, minc, maxr, maxc = prop.bbox
                    minr = max(0, minr - 1);
                    minc = max(0, minc - 1)
                    maxr = min(mask_region.shape[0], maxr + 1)
                    maxc = min(mask_region.shape[1], maxc + 1)
                    cropped = mask_region[minr:maxr, minc:maxc]
                    eroded_cropped = erosion(cropped, erosion_disk)
                    mask_cp_eroded[minr:maxr, minc:maxc][eroded_cropped] = ml

                # apply FINAL global clusters into global mask
                # NOTE: Mask label may be float in Excel; cast safely
                df_fov['Mask label'] = df_fov['Mask label'].fillna(0)
                try:
                    df_fov['Mask label'] = df_fov['Mask label'].astype(int)
                except Exception:
                    df_fov['Mask label'] = df_fov['Mask label'].astype(str)

                df_fov['cluster_global'] = df_fov['cluster_global'].fillna(0).astype(np.uint16)

                for _, row in df_fov.iterrows():
                    ml = int(row['Mask label'])
                    cid = int(row['cluster_global'])
                    if ml == 0:
                        continue
                    mask_global[mask_cp_eroded == ml] = cid

            if mask_global is None:
                self._notify(f"FOV {fov}: no segmentation loaded for any localization, skip saving masks.")
                continue

            # save cls (now it is GLOBAL class ids)
            cls_path = os.path.join(int_folder, f'{fov}-cls.tif')
            try:
                tiff.imwrite(cls_path, mask_global)
                self._notify(f"Saved cluster mask for FOV {fov} to {cls_path}.")
            except Exception as e:
                self._notify(f"Failed to save cluster mask for FOV {fov}: {e}")

            # color
            try:
                # IMPORTANT: mask_global max must be < len(mask_color_map)
                if int(mask_global.max()) >= len(mask_color_map):
                    self._notify(
                        f"[WARN] FOV {fov}: mask has class id up to {int(mask_global.max())}, "
                        f"but colormap size is {len(mask_color_map)}. Check cluster_global assignment."
                    )
                mask_color = mask_color_map[mask_global]
                mask_color = (mask_color * 255).astype(np.uint8)
            except Exception as e:
                self._notify(f"Color mapping failed for FOV {fov}: {e}")
                continue

            color_path = os.path.join(int_folder, f'{fov}-cls-color.tif')
            try:
                tiff.imwrite(color_path, mask_color)
                self._notify(f"Saved colored mask for FOV {fov} to {color_path}.")
            except Exception as e:
                self._notify(f"Failed to save colored mask for FOV {fov}: {e}")

            # add text labels (N1.. / P7.. etc.) based on cluster_global -> tag
            try:
                mask_color_text = self._draw_tags_on_rgb(mask_color, mask_global, id_to_tag)
                text_path = os.path.join(int_folder, f'{fov}-cls-color-text.tif')
                tiff.imwrite(text_path, mask_color_text)
                self._notify(f"Saved colored+text mask for FOV {fov} to {text_path}.")
            except Exception as e:
                self._notify(f"Failed to draw/save text labels for FOV {fov}: {e}")

    def _merge_clustered(self, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:

        def _dbg_cols(df, name):
            cols = list(df.columns)
            dups = pd.Index(cols)[pd.Index(cols).duplicated()].tolist()
            print(f"[{name}] ncols={len(cols)} dup_cols={dups}")
            # 也看下是不是带空格/不可见字符导致“看起来一样但其实不一样”
            weird = [(c, repr(c)) for c in cols if str(c).strip() != str(c)]
            if weird:
                print(f"[{name}] has_whitespace_cols:", weird[:10])

        _dbg_cols(old, "old")
        _dbg_cols(new, "new")
        # must-have cols in new
        for c in ['FOV', 'Mask label', 'Localization']:
            if c not in new.columns:
                raise ValueError(f"New data missing column: {c}")

        old = old.copy()
        new = new.copy()

        def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # 1) 统一列名：去首尾空格
            df.columns = [str(c).strip() for c in df.columns]

            # 2) 如果仍然重复列名：保留最后一个（或第一个都行，但要固定）
            if df.columns.duplicated().any():
                # 保留最后一个：通常更像“新数据覆盖旧数据”
                df = df.loc[:, ~df.columns.duplicated(keep="last")]
            return df

        old = _clean_columns(old)
        new = _clean_columns(new)

        # ensure Localization exists
        if 'Localization' not in old.columns:
            old['Localization'] = ''

        # normalize Localization (upper, NAN->'')
        def _norm_loc(s):
            s = '' if s is None else str(s).strip().upper()
            return '' if s in ('NAN', 'NONE') else s

        old['Localization'] = old['Localization'].fillna('').apply(_norm_loc)
        new['Localization'] = new['Localization'].fillna('').apply(_norm_loc)

        # normalize FOV to string (safer)
        old['FOV'] = old['FOV'].fillna('').astype(str)
        new['FOV'] = new['FOV'].fillna('').astype(str)

        # normalize Mask label (avoid 1 vs 1.0 mismatch)
        old['Mask label'] = old['Mask label'].fillna(0)
        new['Mask label'] = new['Mask label'].fillna(0)
        try:
            old['Mask label'] = old['Mask label'].astype(int)
        except Exception:
            old['Mask label'] = old['Mask label'].astype(str)
        try:
            new['Mask label'] = new['Mask label'].astype(int)
        except Exception:
            new['Mask label'] = new['Mask label'].astype(str)

        key_cols = ['FOV', 'Localization', 'Mask label']

        old_idx = old.set_index(key_cols)
        new_idx = new.set_index(key_cols)

        # ensure old has all columns that appear in new (create empty columns if missing)
        key_cols = ['FOV', 'Localization', 'Mask label']

        # ensure old has all non-key columns that appear in new
        for col in new.columns:
            if col in key_cols:
                continue
            if col not in old_idx.columns:
                old_idx[col] = '' if col == 'cluster_tag' else np.nan

        # update: new overwrites old on intersection keys
        old_idx.update(new_idx)

        # append: keys only in new
        add_idx = new_idx.index.difference(old_idx.index)
        if len(add_idx) > 0:
            old_idx = pd.concat([old_idx, new_idx.loc[add_idx]], axis=0)

        out = old_idx.reset_index()

        # optional: keep dtypes clean for cluster cols
        if 'cluster_local' in out.columns:
            out['cluster_local'] = out['cluster_local'].fillna(0).astype(int)
        if 'cluster_tag' in out.columns:
            out['cluster_tag'] = out['cluster_tag'].fillna('').astype(str)

        # DO NOT trust old cluster_global: we'll recompute per folder after merge in save_results()
        # If it exists, leave it; it will be overwritten by _assign_cluster_global_per_folder(out)

        return out

    def _assign_cluster_global_per_folder(self, out: pd.DataFrame) -> pd.DataFrame:
        """
        For ONE base_folder's merged dataframe `out`,
        assign cluster_global by concatenating local clusters in order:
        N -> M -> P -> ''(None)
        cluster_local is 0..K (0=background/unassigned).
        cluster_global is 0..T (0=background), with N locals first.
        """
        out = out.copy()

        # normalize Localization
        if 'Localization' not in out.columns:
            out['Localization'] = ''
        out['Localization'] = out['Localization'].fillna('').astype(str).str.upper()
        out['Localization'] = out['Localization'].replace({'NAN': ''})

        # normalize cluster_local
        if 'cluster_local' not in out.columns:
            # fallback: if old file uses 'cluster' as local, try to use it
            if 'cluster' in out.columns:
                out['cluster_local'] = out['cluster'].fillna(0).astype(int)
            else:
                out['cluster_local'] = 0
        out['cluster_local'] = out['cluster_local'].fillna(0).astype(int)

        out['cluster_global'] = 0

        offset = 0
        loc_order = ['N', 'M', 'P', '']  # None as ''

        for loc in loc_order:
            if loc == '':
                sel = out['Localization'].isin([''])
            else:
                sel = (out['Localization'] == loc)

            sub = out.loc[sel & (out['cluster_local'] > 0), 'cluster_local']
            if len(sub) == 0:
                continue

            kmax = int(sub.max())  # strict 1..K
            out.loc[sel & (out['cluster_local'] > 0), 'cluster_global'] = (
                    offset + out.loc[sel & (out['cluster_local'] > 0), 'cluster_local'].astype(int)
            )
            offset += kmax

        return out

    def save_results(self):
        # IMPORTANT: save should use the global/full table, not self.df_test (current loc subset)
        if not hasattr(self, "df_test_all") or self.df_test_all is None or len(self.df_test_all) == 0:
            self._notify("No global test table (df_test_all). Please click 'Read and Plot' first.")
            return

        def _set_progress(pct: int, msg: str):
            try:
                self.save_progress.value = int(pct)
                self.save_status.value = msg
            except Exception:
                pass
            try:
                from qtpy.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass

        # Step weights based on roughly observed costs: cheap prep (~5%),
        # Excel write (~30%), mask drawing (~65%). Progress only moves AFTER a
        # step actually completes, so "100%" really means "done".
        STAGE_PCTS = {
            'prep':      5,   # after assign_global + sort
            'confirmed': 10,  # after overwrite prompt answered Yes
            'wrote':     35,  # after to_excel
            'drew':     100,  # after process_masks
        }

        self.save_button.enabled = False
        try:
            groups = list(self.df_test_all.groupby('base_folder'))
            n_folders = max(1, len(groups))
            _set_progress(0, f'Preparing {n_folders} folder(s)...')

            for fi, (folder, grp_all) in enumerate(groups):
                filename = 'clustered.xlsx'
                path = os.path.join(folder, filename)
                base = os.path.basename(folder) or folder

                def pct(stage_key: str) -> int:
                    # Map per-folder stage into [fi/N*100, (fi+1)/N*100]
                    span = 100.0 / n_folders
                    return int(fi * span + STAGE_PCTS[stage_key] * span / 100.0)

                out = grp_all.copy()
                try:
                    out = self._assign_cluster_global_per_folder(out)
                except Exception as e:
                    self._notify(f"Failed to assign cluster_global for {folder}: {e}")
                sort_cols = [c for c in ['FOV', 'Localization', 'cluster_global', 'Mask label']
                             if c in out.columns]
                if sort_cols:
                    out = out.sort_values(sort_cols, kind='stable')
                _set_progress(pct('prep'), f'[{base}] prepared, asking overwrite if needed...')

                if os.path.exists(path):
                    from qtpy.QtWidgets import QMessageBox
                    reply = QMessageBox.question(
                        None,
                        "Overwrite existing file?",
                        f"‘{filename}’ already exists in:\n{folder}\n\nOverwrite (merged result)?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        _set_progress(pct('drew'), f'[{base}] skipped (overwrite declined).')
                        continue
                _set_progress(pct('confirmed'), f'[{base}] writing {filename}...')

                try:
                    out.to_excel(path, index=False)
                except Exception as e:
                    self._notify(f"Failed to save to {path}: {e}")
                    continue
                _set_progress(pct('wrote'), f'[{base}] drawing per-class masks...')

                self._process_masks_for_folder(folder, out)
                _set_progress(pct('drew'), f'[{base}] done.')

            _set_progress(100, 'Save complete.')
            self._notify("Save complete.")
        finally:
            self.save_button.enabled = True


from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QWidget,
    QLabel,
    QFormLayout
)
from magicgui.widgets import Container, create_widget, PushButton, FileEdit, CheckBox
import magicgui
# Inject create_widget into magicgui's top-level namespace for compatibility.
magicgui.create_widget = create_widget
from typing import Annotated
from scipy.ndimage import sum as ndi_sum, mean as ndi_mean


# Helper: add a row to a QFormLayout.
# If more than one widget is provided, they are arranged horizontally.
def add_form_row(form: QFormLayout, prompt: str, widgets):
    if len(widgets) == 1:
        form.addRow(prompt, widgets[0].native)
    else:
        container = QWidget()
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            hlayout.addWidget(w.native)
        container.setLayout(hlayout)
        form.addRow(prompt, container)

# Helper: add a row with two label-widget pairs in the same row.
def add_double_row(form: QFormLayout, prompt1: str, widget1, prompt2: str, widget2, bg_color: str):
    container = QWidget()
    container.setStyleSheet(f"background-color: {bg_color};")
    hlayout = QHBoxLayout(container)
    hlayout.setContentsMargins(0, 0, 0, 0)

    # First pair: prompt and widget
    label1 = QLabel(prompt1)
    label1.setStyleSheet("color: black; font-family: Calibri;")
    hlayout.addWidget(label1)
    # Override the interactive widget's style:
    widget1.native.setStyleSheet("background-color: #414851; color: white;")
    hlayout.addWidget(widget1.native)

    hlayout.addSpacing(10)

    # Second pair: prompt and widget
    label2 = QLabel(prompt2)
    label2.setStyleSheet("color: black; font-family: Calibri;")
    hlayout.addWidget(label2)
    widget2.native.setStyleSheet("background-color: #414851; color: white;")
    hlayout.addWidget(widget2.native)

    container.setLayout(hlayout)
    form.addRow(container)


# Helper: create a group container with an external title.
def create_group(title: str, form: QFormLayout, bg_color: str, border_color: str) -> QWidget:
    group_box = QGroupBox("")
    group_box.setLayout(form)
    # Tight vertical spacing inside the form to reclaim canvas height.
    try:
        form.setVerticalSpacing(4)
        form.setHorizontalSpacing(6)
        form.setContentsMargins(6, 4, 6, 4)
    except Exception:
        pass
    group_box.setStyleSheet(f"""
        QGroupBox {{
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: 4px;
        }}
        QGroupBox QLabel {{
            color: black;
            font-family: Calibri;
        }}
        QCheckBox {{
            color: black;
            font-family: Calibri;
        }}
    """)
    title_label = QLabel(title)
    title_label.setStyleSheet(
        "font-size: 15px; font-weight: bold; color: white; "
        "font-family: Calibri; margin: 0px; padding: 4px 8px;"
    )
    container_layout = QVBoxLayout()
    container_layout.setContentsMargins(2, 2, 2, 2)
    container_layout.setSpacing(2)
    container_layout.addWidget(title_label)
    container_layout.addWidget(group_box)
    container = QWidget()
    container.setLayout(container_layout)
    return container

# Create Part 0: A settings row (light background, no title) with checkboxes and the top-level button.
def create_part0(read_button, mask_checkbox, revise_checkbox, ratio_checkbox):
    container = QWidget()
    hlayout = QHBoxLayout() # qh means horizontal layout
    hlayout.setContentsMargins(2, 2, 2, 2)
    # Add the "Read in all files" button first.
    hlayout.addWidget(read_button.native)
    hlayout.addSpacing(20)
    # Add the checkboxes.
    hlayout.addWidget(mask_checkbox.native)
    hlayout.addWidget(revise_checkbox.native)
    hlayout.addWidget(ratio_checkbox.native)
    container.setLayout(hlayout)
    container.setStyleSheet("background-color: #;")
    return container

class Trackrevise(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.plot_widget = PlotWidget()
        self.viewer.window.add_dock_widget(self.plot_widget, area='bottom', name='Signal')
        self.masks_history = []


        # -------------------------------
        # Part 0: Top-level row with "Read in all files" and checkboxes.
        # -------------------------------
        self.read_in_all_button = PushButton(text="▶ Read in all")
        self.read_in_all_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #1E88E5;"
            "  color: white;"
            "  font-weight: bold;"
            "  padding: 4px 10px;"
            "  border-radius: 4px;"
            "  font-family: Calibri;"
            "} "
            "QPushButton:hover { background-color: #1565C0; }"
        )
        self.mask_256_checkbox = CheckBox(text="Masks>255", value=True)
        self.revise_mode_checkbox = CheckBox(text="Revise+Visualize Mode")
        self.revise_mode_checkbox.value = False
        # Highlight Revise mode so users notice it's an interactive editor toggle.
        self.revise_mode_checkbox.native.setStyleSheet(
            "QCheckBox {"
            "  background-color: #FFF3E0;"
            "  border: 1px solid #FB8C00;"
            "  border-radius: 3px;"
            "  padding: 2px 4px;"
            "  color: #E65100;"
            "  font-weight: bold;"
            "  font-family: Calibri;"
            "}"
        )
        self.ratio_checkbox = CheckBox(text="FRET for G/B")
        self.ratio_checkbox.value = True

        part0 = create_part0(self.read_in_all_button,
                             self.mask_256_checkbox,
                             self.revise_mode_checkbox,
                             self.ratio_checkbox)

        # -------------------------------
        # Group 1: Tracking Revision
        # -------------------------------
        self.mask_folder = FileEdit(label="Masks Folder", mode='d')
        self.mask_folder.value = r'J:/Mix16-N-P-260306-DCZ-2-1/Track_log_rainbow'
        self.read_masks_button = PushButton(text="Read")
        # Now, mask folder row only has file input and read button.
        self.tif_input = FileEdit(label="TIF stack Input", mode='r', filter='*.tif')
        self.tif_input.value = r'J:/Mix16-N-P-260306-DCZ-2-1/FOV-1-b.tif'
        self.read_tif_button = PushButton(text="Read")
        form1 = QFormLayout()
        add_form_row(form1, "Mask Folder", [self.mask_folder, self.read_masks_button])
        add_form_row(form1, "TIF stack Input", [self.tif_input, self.read_tif_button])
        # Also add action buttons row.
        self.apply_next_button = PushButton(text="Apply to Next Fr")
        self.apply_button = PushButton(text="Apply to Following Frs")
        self.save_tracking_button = PushButton(text="Save Track")
        add_form_row(form1, "If revised:", [self.apply_next_button, self.apply_button, self.save_tracking_button])
        group1 = create_group("Tracking Revision", form1, "#FFF9E6", "#FFC107")

        # -------------------------------
        # Group 2: Biosensor Channels
        # -------------------------------
        self.stack_b_input = FileEdit(label="Stack B Input", mode='r', filter='*.tif')
        self.stack_b_input.value = r'J:/Mix16-N-P-260306-DCZ-2-1/FOV-1-b.tif'
        self.read_stack_b_button = PushButton(text="Read")
        self.stack_g_input = FileEdit(label="Stack G Input", mode='r', filter='*.tif')
        self.stack_g_input.value = r'J:/Mix16-N-P-260306-DCZ-2-1/FOV-1-g.tif'
        self.read_stack_g_button = PushButton(text="Read")
        self.stack_nir_input = FileEdit(label="Stack NIR Input", mode='r', filter='*.tif')
        self.stack_nir_input.value = r'J:/Mix16-N-P-260306-DCZ-2-1/FOV-1-y.tif'
        self.read_stack_nir_button = PushButton(text="Read")
        form2 = QFormLayout()
        add_form_row(form2, "Stack B Input", [self.stack_b_input, self.read_stack_b_button])
        add_form_row(form2, "Stack G Input", [self.stack_g_input, self.read_stack_g_button])
        add_form_row(form2, "Stack NIR Input", [self.stack_nir_input, self.read_stack_nir_button])
        group2 = create_group("Biosensor Channels", form2, "#F0FFF0", "#66BB6A")

        # -------------------------------
        # Group 2.5: Calibration, Shifting and Overexposure
        # -------------------------------
        ChannelChoice = Annotated[
            str,
            (
                ("widget_type", "ComboBox"),
                ("choices", ["Stack G", "Stack NIR", "Stack B", "Barcodes Masks classified"]),
            )
        ]
        self.channel_to_shift = create_widget(label="Channel to shift", annotation=ChannelChoice, value="Stack G")
        self.channel_to_shift.native.setStyleSheet(
            "QComboBox {"
            "  background-color: #414851;"  # dark background
            "  color: white;"  # white text
            "  font-family: Calibri;"
            "}"
        )
        shift_max = 20
        self.shift_r_param = create_widget(label="Right shift", widget_type="Slider", value=0)
        self.shift_r_param.native.setStyleSheet("""
            QSlider::handle {
                color: black;
            }
        """)
        self.shift_r_param.min = -shift_max
        self.shift_r_param.max = shift_max
        self.shift_r_param.step = 1
        self.shift_u_param = create_widget(label="Up shift", widget_type="Slider", value=0)
        self.shift_u_param.min = -shift_max
        self.shift_u_param.max = shift_max
        self.shift_u_param.step = 1
        self.shift_button = PushButton(text="Shift")
        self.shift_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #414851;"  # dark background 65 72 81 = #
            "  color: white;"  # white text
            "  font-family: Calibri;"
            "}"
        )
        self.shift_save_button = PushButton(text="Save Shift")
        self.shift_save_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #414851;"  # dark background
            "  color: white;"  # white text
            "  font-family: Calibri;"
            "}"
        )
        # Now, combine channel_to_shift and the two action buttons in one row.
        form25 = QFormLayout()
        # Custom row: no prompt for the buttons.
        container = QWidget()
        # make the bg as same as the group
        container.setStyleSheet("background-color: #E8F4FD;")
        hlayout = QHBoxLayout()
        # SET THE bg of hlayout to the same as the group
        # hlayout.setStyleSheet("background-color: #F0FFF0;") wrong
        hlayout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Channel to shift")
        label.setStyleSheet("color: black; font-family: Calibri;")
        hlayout.addWidget(label)
        hlayout.addWidget(self.channel_to_shift.native)
        hlayout.addSpacing(10)
        hlayout.addWidget(self.shift_button.native)
        hlayout.addWidget(self.shift_save_button.native)
        container.setLayout(hlayout)
        form25.addRow(container)
        # Next row: the two sliders in one row.
        add_double_row(form25, "Right shift", self.shift_r_param, "Up shift", self.shift_u_param, bg_color="#E8F4FD")
        # Next row: Overexposure threshold with its buttons.
        # add a spinbox for overexposure threshold, from 0-65535
        self.overexpo_thres_param = create_widget(label="Overexposure Threshold", widget_type="SpinBox", value=65535, options={'min': 0, 'max': 65535})
        self.overexpo_vis_button = PushButton(text="visualize")
        self.overexpo_discard_button = PushButton(text="Discard")
        add_form_row(form25, "Overexpo Threshold", [self.overexpo_thres_param, self.overexpo_vis_button, self.overexpo_discard_button])
        group25 = create_group("Multi-channel registration (Shifting in pixel) and Overexposure Check", form25, "#E8F4FD", "#42A5F5")

        # -------------------------------
        # Group 3: Barcodes Alignment
        # -------------------------------
        self.classification_input = FileEdit(label="Barcode Image", mode='r', filter='*.tif')
        self.classification_input.value = r'J:/Mix16-N-P-260306-DCZ-2-1/intensity/TileScan_001_s1-cls.tif'
        self.classification_resize = create_widget(label="Resize to", widget_type="SpinBox", value=1024,
                                                   options={'min': 512, 'max': 2048})
        self.classification_resize.native.setStyleSheet(
            "QSpinBox {"
            "  background-color: #F3E5F5;"  # light background
            "  color: black;"  # black text
            "  font-family: Calibri;"
            "}"
        )
        self.align_thres_percent = create_widget(label="Align Threshold", widget_type="SpinBox", value=10,
                                                 options={'min': 0, 'max': 100})
        self.align_mask_frame = create_widget(label="Align to Frame", widget_type="SpinBox", value=0,
                                              options={'min': 0, 'max': 2000})
        # Barcode rotation: matches BiosensorSeg. Leica tilescan defaults to
        # 90° CW; single-FOV acquisitions usually need 0°.
        self.classification_rotate = create_widget(
            label='Barcode rotation', widget_type='ComboBox',
            value='90° CW',
            options={'choices': ['0°', '90° CW', '180°', '270° CW']},
        )
        self.classification_align_button = PushButton(text="▶ Align")
        self.classification_align_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #8E24AA;"
            "  color: white;"
            "  font-weight: bold;"
            "  padding: 4px 10px;"
            "  border-radius: 4px;"
            "  font-family: Calibri;"
            "} "
            "QPushButton:hover { background-color: #6A1B9A; }"
        )
        form3 = QFormLayout()
        add_form_row(form3, "Barcode Image", [self.classification_input])
        add_double_row(form3, "Resize to", self.classification_resize, "Align Threshold", self.align_thres_percent, bg_color="#F3E5F5")
        add_form_row(form3, "Rotate", [self.classification_rotate])
        add_form_row(form3, "Align to Frame", [self.align_mask_frame, self.classification_align_button])
        group3 = create_group("Barcodes-Biosensor Alignment", form3, "#F3E5F5", "#AB47BC")
        self.Bs2Code_save_path = None
        # -------------------------------
        # Group 4: Calculate Signals
        # -------------------------------
        self.ratio_calcu_range = create_widget(label="Ratio Calculation Range", widget_type="RangeSlider",
                                               value=[0, 100],
                                               options={'min': 0, 'max': 2000})
        # self.basal_frame_spinbox = create_widget(label="Basal Frame Number", widget_type="SpinBox", value=34,
        #                                          options={'min': 0, 'max': 2000})
        self.basal_frame_range = create_widget(label="Basal Frame Range", widget_type="RangeSlider",
                                              value=[0, 21],
                                                options={'min': 0, 'max': 2000})

        self.ratio_calcu_button = PushButton(text="▶ Calculate (final)")
        self.ratio_calcu_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #2E7D32;"
            "  color: white;"
            "  font-weight: bold;"
            "  padding: 4px 10px;"
            "  border-radius: 4px;"
            "  border: 1px solid #1B5E20;"
            "  font-family: Calibri;"
            "} "
            "QPushButton:hover { background-color: #1B5E20; } "
            "QPushButton:disabled { background-color: #9E9E9E; color: #EEEEEE; }"
        )
        form4 = QFormLayout()
        add_form_row(form4, "Ratio Calculation Range", [self.ratio_calcu_range])
        # add_form_row(form4, "Basal Frame Number", [self.basal_frame_spinbox, self.ratio_calcu_button])
        add_form_row(form4, "Basal Frame Range", [self.basal_frame_range, self.ratio_calcu_button])
        # add_double_row(form4, "Basal Frame Range", self.basal_frame_range, "", self.ratio_calcu_button, bg_color="#FFEEEE")
        group4 = create_group("Calculate Signals", form4, "#FFEEEE", "#FF8888")
        # -------------------------------
        self.freq_analysis_checkbox = CheckBox(text="Frequency Domain Analysis")
        self.freq_analysis_checkbox.value = False
        self.freq_analysis_hint = QLabel(
            'Tip: enable for per-cell analysis when the biosensor shows clear '
            'oscillations (e.g. Ca²⁺ signals). Produces per-cell dominant '
            'frequency + phase maps.'
        )
        self.freq_analysis_hint.setWordWrap(True)
        self.freq_analysis_hint.setStyleSheet(
            'color: #BDBDBD; font-style: italic; font-family: Calibri; font-size: 12px;'
        )
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(self.freq_analysis_checkbox.native)
        freq_layout.addWidget(self.freq_analysis_hint)
        freq_layout.addStretch(1)

        # -------------------------------
        # New: Percentile sliders for contrast limits
        # self.freq_clim_slider = create_widget(
        #     label="Freq CLIM (%)",
        #     widget_type="RangeSlider",
        #     value=[5, 95],
        #     options={'min': 0, 'max': 100}
        # )
        # # to get the value low, use self.freq_clim_slider.value[
        # self.phase_clim_slider = create_widget(
        #     label="Phase CLIM (%)",
        #     widget_type="RangeSlider",
        #     value=[5, 95],
        #     options={'min': 0, 'max': 100}
        # )
        # clim_layout = QFormLayout()
        # add_form_row(
        #     clim_layout,
        #     "Contrast Limits",
        #     [self.freq_clim_slider, self.phase_clim_slider]
        # )

        # -------------------------------
        # Arrange all parts vertically (one column)
        # -------------------------------
        main_vlayout = QVBoxLayout()
        main_vlayout.setContentsMargins(4, 4, 4, 4)
        main_vlayout.setSpacing(4)
        main_vlayout.addWidget(part0)
        main_vlayout.addWidget(group1)
        main_vlayout.addWidget(group2)
        main_vlayout.addWidget(group25)
        main_vlayout.addWidget(group3)
        main_vlayout.addWidget(group4)
        main_vlayout.addLayout(freq_layout)
        # main_vlayout.addLayout(clim_layout)

        # Progress bar + status label — shared by Read / Align / Calculate
        # so users see what the long-running synchronous work is doing.
        from qtpy.QtWidgets import QProgressBar
        self._nacha_progress = QProgressBar()
        self._nacha_progress.setRange(0, 100)
        self._nacha_progress.setValue(0)
        self._nacha_progress.setMaximumHeight(12)
        self._nacha_progress.setTextVisible(False)
        self._nacha_status = QLabel('Ready.')
        self._nacha_status.setStyleSheet(
            'color: #FFEB3B; font-family: Calibri; font-size: 11px; font-weight: bold;'
        )
        main_vlayout.addWidget(self._nacha_progress)
        main_vlayout.addWidget(self._nacha_status)

        self.preview_gb_button = PushButton(text="Preview G/B map (masked)")
        self.preview_gb_button.native.setStyleSheet(
            "QPushButton {"
            "  background-color: #414851;"
            "  color: white;"
            "  font-family: Calibri;"
            "}"
        )
        main_vlayout.addWidget(self.preview_gb_button.native)

        existing = self.native.layout()
        if existing is None:
            self.native.setLayout(main_vlayout)
        else:
            existing.addLayout(main_vlayout)

        # ---------------------------------------------------------
        # Connect callbacks (implement these methods in your class)
        # ---------------------------------------------------------
        self.read_in_all_button.clicked.connect(self.read_in_all)
        self.read_masks_button.clicked.connect(self.read_masks)
        self.read_tif_button.clicked.connect(lambda: self.read_tif('TIF for Tracking'))
        self.read_stack_b_button.clicked.connect(lambda: self.read_tif('Stack B'))
        self.read_stack_g_button.clicked.connect(lambda: self.read_tif('Stack G'))
        self.read_stack_nir_button.clicked.connect(lambda: self.read_tif('Stack NIR'))
        self.apply_button.clicked.connect(self.apply_to_following_frames)
        self.apply_next_button.clicked.connect(self.apply_to_next_frame)
        self.save_tracking_button.clicked.connect(self.save_tracking)
        self.shift_button.clicked.connect(self.shift_stack)
        self.shift_save_button.clicked.connect(self.save_shifted_stack)
        self.overexpo_vis_button.clicked.connect(self.overexpo_visualize)
        self.overexpo_discard_button.clicked.connect(self.overexpo_discard)
        self.revise_mode_checkbox.changed.connect(self.toggle_revise_mode)
        self.classification_input.changed.connect(self.load_classification)
        self.classification_resize.changed.connect(self.load_classification)
        self.classification_align_button.clicked.connect(self.align_classification)
        self.ratio_calcu_button.clicked.connect(self.calculate_signal_ratio)
        self.preview_gb_button.clicked.connect(self.preview_g_over_b_masked)

        # Tooltips — hover any control for details.
        _tt(self.read_in_all_button,
            'One-click: load tracking masks + B/G/Y channel stacks + '
            'barcode classification from the selected sample folder.')
        _tt(self.mask_256_checkbox,
            'Masks contain more than 255 cell labels. Keep on unless you '
            'know your labels fit in uint8.')
        _tt(self.revise_mode_checkbox,
            'Revise + Visualize mode: Shift-click any cell in the viewer '
            'to pop up its individual signal curve. Useful QC before '
            'trusting the class averages.')
        _tt(self.ratio_checkbox,
            'Compute FRET ratio as G/B (otherwise: single-channel signal).')
        _tt(self.read_masks_button, 'Load per-frame tracking masks (.npy) from a folder.')
        _tt(self.read_tif_button, 'Load the tracking image stack (.tif).')
        _tt(self.apply_next_button,
            'Copy the currently-selected cell mask label forward by ONE frame.')
        _tt(self.apply_button,
            'Copy the currently-selected cell mask label forward to EVERY '
            'subsequent frame.')
        _tt(self.save_tracking_button,
            'Save the possibly-edited per-frame tracking masks back to '
            'disk (one .npy per frame).')
        _tt(self.read_stack_b_button, 'Load the B channel stack.')
        _tt(self.read_stack_g_button, 'Load the G channel stack.')
        _tt(self.read_stack_nir_button, 'Load the NIR / Y channel stack.')
        _tt(self.channel_to_shift,
            'Which stack to shift when aligning channels (drift / chromatic '
            'offset).')
        _tt(self.shift_r_param, 'Rightward pixel shift applied to the selected channel.')
        _tt(self.shift_u_param, 'Upward pixel shift applied to the selected channel.')
        _tt(self.shift_button, 'Apply the shift to the selected channel in memory.')
        _tt(self.shift_save_button, 'Save the shifted stack back to disk.')
        _tt(self.overexpo_vis_button,
            'Visualise over-exposed pixels as a mask layer so you can judge '
            'the threshold.')
        _tt(self.overexpo_discard_button,
            'Zero out over-exposed pixels in the stacks before computing '
            'per-cell signals.')
        _tt(self.classification_resize,
            'Resize the barcode classification image to this edge length '
            'before overlaying on the tracking stack.')
        _tt(self.align_thres_percent,
            'Minimum overlap PERCENTAGE required for a tracked cell to be '
            'assigned to a barcode class. e.g. 10 = the tracked cell must '
            'share at least 10%% of its pixels with the barcode class '
            'region. Higher = stricter, fewer cells get labels but cleaner.')
        _tt(self.align_mask_frame,
            '0-based tracking-stack frame used as the alignment reference: '
            'the barcode classification mask is overlaid on THIS frame '
            'when assigning labels to tracked cells. Typically 0 (first '
            'frame) because cells drift less early on.')
        _tt(self.classification_align_button,
            'Align barcode classification -> tracking mask labels and write '
            'Bs2Code.xlsx with the mapping table.')
        _tt(self.ratio_calcu_range,
            'Frame range (start, end inclusive) used to compute the per-cell '
            'signal ratio. Trim tightly to the window where you expect a '
            'response — including long stretches of noise just dilutes the '
            'per-class curve.')
        _tt(self.basal_frame_range,
            'Frames treated as baseline. Each cell\'s signal is normalised '
            'to the mean over this range, so all curves start near 1.0 — '
            'factors out cell-to-cell brightness differences and focuses '
            'on RELATIVE change. Typical: the first ~5 pre-stimulus frames.')
        _tt(self.overexpo_thres_param,
            'Pixels at or above this DN value are treated as saturated '
            '(detector pegged). 65535 = off (no pixels flagged) for '
            'uint16 data. Lower if you see clipping in bright cells.')
        _tt(self.ratio_calcu_button,
            'Compute per-class mean ± SE signal curves and write '
            'signal_analysis.xlsx. FINAL step of the workflow.')
        _tt(self.freq_analysis_checkbox,
            'Additionally run frequency-domain analysis (FFT) on the '
            'per-cell signals.')
        _tt(self.preview_gb_button,
            'Preview the G/B ratio map masked by the current cell labels, '
            'so you can eyeball the FRET response before Calculate.')

        self.num_masks = 1000

        # NaCha is the last step of the workflow — no Next button; the
        # celebration dialog fires from calculate_signal_ratio when it finishes.

    def _set_nacha_progress(self, pct: int, msg: str):
        try:
            self._nacha_progress.setValue(int(pct))
            self._nacha_status.setText(msg)
        except Exception:
            pass
        try:
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

    def _get_active_or_first_labels_layer(self):
        """Return the active Labels layer, or the first one if none selected."""
        try:
            from napari.layers import Labels
        except Exception:
            return None

        selected = list(self.viewer.layers.selection)
        if selected:
            lyr = selected[0]
            if getattr(lyr, 'data', None) is not None and lyr.__class__.__name__ == 'Labels':
                return lyr

        for lyr in self.viewer.layers:
            if getattr(lyr, 'data', None) is not None and lyr.__class__.__name__ == 'Labels':
                return lyr
        return None

    def _get_image_layer_by_name(self, name):
        """Find an Image layer by exact name."""
        for lyr in self.viewer.layers:
            if lyr.name == name and getattr(lyr, 'data', None) is not None and lyr.__class__.__name__ == 'Image':
                return lyr
        return None

    def preview_g_over_b_masked(self):
        """
        Preview pixel-wise G/B ratio inside tracked masks only.
        Requires: 'Stack G', 'Stack B', and at least one Labels layer.
        Adds a new Image layer named 'G/B (masked)' with turbo colormap.
        """
        import numpy as np
        from napari.utils.notifications import show_error, show_info

        # --- 1. Fetch required layers ---
        g_layer = self._get_image_layer_by_name('Stack G')
        b_layer = self._get_image_layer_by_name('Stack B')
        labels_layer = self._get_active_or_first_labels_layer()

        missing = []
        if g_layer is None:
            missing.append("Stack G")
        if b_layer is None:
            missing.append("Stack B")
        if labels_layer is None:
            missing.append("Tracking masks (Labels)")
        if missing:
            show_error(
                f"Missing required layer(s): {', '.join(missing)}.\n"
                f"Please ensure Stack G, Stack B, and one Labels (tracking) layer are loaded."
            )
            return

        G = np.asarray(g_layer.data)
        B = np.asarray(b_layer.data)
        M = np.asarray(labels_layer.data)

        # --- 2. Shape validation ---
        if G.shape != B.shape or G.shape != M.shape:
            show_error(
                f"Shape mismatch:\nG:{G.shape}  B:{B.shape}  Masks:{M.shape}\n"
                f"All layers must have the same dimensions (e.g. T×Y×X or Y×X)."
            )
            return

        # --- 3. Compute G/B inside mask only ---
        Gf = G.astype(np.float32)
        Bf = B.astype(np.float32)
        eps = 1e-6
        ratio = np.full_like(Gf, np.nan, dtype=np.float32)

        inside = M > 0
        valid = inside & (Bf > 0)
        ratio[valid] = Gf[valid] / (Bf[valid] + eps)

        # --- 4. Robust contrast limits ---
        try:
            vals = ratio[~np.isnan(ratio)]
            if vals.size >= 16:
                lo, hi = np.nanpercentile(vals, (2, 98))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
            else:
                lo, hi = 0.0, 1.0
        except Exception:
            lo, hi = 0.0, 1.0

        # --- 5. Remove existing layer with same name ---
        name = "G/B (masked)"
        for lyr in list(self.viewer.layers):
            if lyr.name == name and getattr(lyr, 'data', None) is not None and lyr.__class__.__name__ == 'Image':
                try:
                    self.viewer.layers.remove(lyr)
                except Exception:
                    pass

        # --- 6. Add new visualization layer ---
        cmap_candidates = ["turbo", "rainbow", "viridis"]
        added = False
        for cm in cmap_candidates:
            try:
                self.viewer.add_image(
                    ratio,
                    name=name,
                    blending="additive",
                    colormap=cm,
                    contrast_limits=(float(lo), float(hi)),
                    metadata={"source": "G/B masked", "note": "ratio inside tracked masks only"},
                )
                show_info(
                    f"G/B map created successfully (colormap='{cm}'). "
                    f"You can adjust contrast and colormap in Napari."
                )
                added = True
                break
            except Exception:
                continue

        if not added:
            show_error("Failed to create G/B map layer. Please check your Napari version or colormap support.")


    def frequency_domain_analysis(self, signal: np.ndarray, sampling_rate: float = 1.0):
        N = len(signal)
        freqs = np.fft.rfftfreq(N, d=1.0 / sampling_rate)
        fft_vals = np.abs(np.fft.rfft(signal))
        return freqs, fft_vals
    def read_in_all(self):
        # call all the read functions
        self.read_masks()
        self.read_tif('TIF for Tracking')
        self.read_tif('Stack B')
        self.read_tif('Stack G')
        self.read_tif('Stack NIR')
        self.load_classification()
        # No-tracking safety net: if the Masks layer is still a single frame
        # but biosensor stacks now have many frames, broadcast Masks so that
        # on_click / calculate_signal_ratio / shift-click plotting all work.
        self._broadcast_masks_to_stack_length()

    def _broadcast_masks_to_stack_length(self):
        """Ensure viewer.layers['Masks'].data has >=1 frame per biosensor frame.

        When the user skipped tracking and only a single seg.npy was read, the
        Masks layer can stay as (1, H, W). Downstream code (on_click plotting,
        calculate_signal_ratio) assumes per-frame indexing, so broadcast it in
        place to the length of Stack B/G/NIR.
        """
        if 'Masks' not in self.viewer.layers:
            return
        masks = np.asarray(self.viewer.layers['Masks'].data)
        if masks.ndim != 3 or masks.shape[0] > 1:
            return
        target_len = None
        for _name in ('Stack B', 'Stack G', 'Stack NIR'):
            if _name in self.viewer.layers:
                _arr = self.viewer.layers[_name].data
                if isinstance(_arr, np.ndarray) and _arr.ndim >= 3:
                    target_len = int(_arr.shape[0])
                    break
        if not target_len or target_len <= masks.shape[0]:
            return
        try:
            new = np.broadcast_to(masks[0:1], (target_len,) + masks.shape[1:]).copy()
            self.viewer.layers['Masks'].data = new
            # read_masks set ratio_calcu_range to [0, num_files-1] = [0, 0]
            # for the single-file case. Widen it to the new frame count so
            # Calculate processes every frame.
            try:
                self.ratio_calcu_range.value = [0, target_len - 1]
            except Exception:
                pass
            notifications.show_info(
                f'Broadcast single Masks frame to {target_len} frames; '
                f'ratio range reset to [0, {target_len - 1}] (no-tracking fallback).'
            )
        except Exception as e:
            print(f'[broadcast masks] {e}')

    def calculate_signal_ratio(self):

        # Check that the Bs2Code Excel file exists.
        if self.Bs2Code_save_path is None:
            self.Bs2Code_save_path = os.path.join(self.base_folder, 'Bs2Code.xlsx')

            notifications.show_warning("You have not run the Bs2Code alignment yet. Now check the existence of the excel.")
        if not os.path.exists(self.Bs2Code_save_path):
            # Ask user whether to proceed with default
            choice = self.show_warning_dialog(
                "No 'Bs2Code.xlsx' found.\n"
                "Continue with default alignment (all cells → class 1)?"
            )
            if choice != "continue":
                return
            # Build default alignment: every cell_id maps to class 1
            idx = np.arange(1, self.num_masks + 1)
            alignment_info = pd.DataFrame({'Class': 1}, index=idx)
        else:
            # Normal path: read the provided Excel
            df = pd.read_excel(self.Bs2Code_save_path)
            df = df[['Tracking Mask Index', 'Class']].set_index('Tracking Mask Index')
            alignment_info = df

        out_excel_path = os.path.join(self.base_folder, 'signal_analysis.xlsx')
        # If the Excel file already exists, ask user if they want to overwrite it.
        if os.path.exists(out_excel_path):
            response = self.show_warning_dialog("Excel file already exists. Do you want to overwrite it?")
            if response != "continue":
                return
            # Check writability: if the file is open elsewhere, attempting to open it will fail.
            try:
                with open(out_excel_path, 'a'):
                    pass
            except Exception as e:
                notifications.show_error(
                    "The Excel file is currently open or not writable. Please close it and try again.")
                return

        self._set_nacha_progress(5, 'Preparing frame range + channels...')
        # Define frame range and baseline parameters.
        stack_start = self.ratio_calcu_range.value[0]
        stack_end = self.ratio_calcu_range.value[1] + 1
        basal_frame_start = self.basal_frame_range.value[0]
        basal_frame_end = self.basal_frame_range.value[1]
        # Check which channels are available.
        has_stack_b = 'Stack B' in self.viewer.layers
        has_stack_g = 'Stack G' in self.viewer.layers
        has_stack_nir = 'Stack NIR' in self.viewer.layers

        if not (has_stack_b or has_stack_g or has_stack_nir):
            notifications.show_error("None of Stack B, G, or NIR found in layers!")
            return

        # Notify user about missing channels.
        if not has_stack_b:
            notifications.show_info("Stack B is missing, skipping Blue channel processing.")
        if not has_stack_g:
            notifications.show_info("Stack G is missing, skipping Green channel processing.")
        if not has_stack_nir:
            notifications.show_info("Stack NIR is missing, skipping NIR channel processing.")

        # Get masks and number of cells.
        all_masks = self.viewer.layers['Masks'].data[stack_start:stack_end]

        # --- Handle "no tracking" case: Masks layer may be a single frame
        # (shape (1, H, W)) when the user skipped B&P Tracker. We need it to
        # match the biosensor stack length so per-frame indexing works. Look
        # up any biosensor stack in the viewer for the expected frame count.
        target_len = None
        for _name in ('Stack B', 'Stack G', 'Stack NIR'):
            if _name in self.viewer.layers:
                _arr = self.viewer.layers[_name].data
                if isinstance(_arr, np.ndarray) and _arr.ndim >= 3:
                    target_len = int(_arr.shape[0])
                    break
        if target_len is None and hasattr(all_masks, 'shape'):
            target_len = int(all_masks.shape[0])
        if all_masks.shape[0] < max(1, (stack_end - stack_start)):
            # single-frame mask → broadcast to all requested frames
            needed = max(1, (target_len or (stack_end - stack_start)))
            notifications.show_info(
                f'Masks has {all_masks.shape[0]} frame(s); broadcasting single mask to '
                f'{needed} frames (no-tracking fallback).'
            )
            all_masks = np.broadcast_to(all_masks[0:1], (needed,) + all_masks.shape[1:]).copy()
        cell_num = len(np.unique(all_masks[0])) - 1
        max_cell_id = np.max(all_masks)
        notifications.show_info(f'Number of Cells: {cell_num} cells')

        def extract_intensity(stack, all_masks, cell_num, mode='sum'):
            intensity_data = []
            cell_ids = list(range(1, cell_num + 1))
            zero_hits = 0

            for frame_number, frame in tqdm(enumerate(stack), total=stack.shape[0], desc="Extracting intensities"):
                labels = all_masks[frame_number]
                if mode == 'sum':
                    vals = ndi_sum(frame, labels=labels, index=cell_ids)
                else:
                    vals = ndi_mean(frame, labels=labels, index=cell_ids)

                for cid, inten in zip(cell_ids, vals):
                    if inten == 0:
                        zero_hits += 1
                    intensity_data.append({"frame": frame_number, "cell_id": int(cid), "intensity": float(inten)})

            if zero_hits > 0:
                notifications.show_warning(
                    f"{zero_hits} zero-intensity events encountered (suppressed per-frame warnings).")
            return intensity_data

        # Helper to compute summary statistics per class for a normalized pivot table.
        def compute_summary_stats(norm_df):
            # For each cell, compute the overall mean (across frames, excluding the 'Class' column).
            cell_means = norm_df.drop(columns=['Class']).mean(axis=1)
            stats = cell_means.groupby(norm_df['Class']).agg(['mean', 'std', 'count'])
            stats = stats.rename(columns={'mean': 'Average', 'std': 'Std', 'count': 'Cell Count'})
            stats['SE'] = stats['Std'] / np.sqrt(stats['Cell Count'])
            return stats

        # Dictionary to hold the normalized data for each channel.
        pivot_data = {}

        self._set_nacha_progress(15, 'Opening Excel writer + extracting intensities...')
        with pd.ExcelWriter(out_excel_path) as writer:
            # if stack_end < stack length, pop out warning
            if stack_end < self.viewer.layers['Masks'].data.shape[0]:
                notifications.show_warning(f'using only frames {stack_start} to {stack_end - 1} of the Masks stack, '
                                           f'which has {self.viewer.layers["Masks"].data.shape[0]} frames.')
            # Process Blue channel.
            if has_stack_b:
                self._set_nacha_progress(25, 'Processing Blue channel...')
                blue_channel = self.viewer.layers['Stack B'].data[stack_start:stack_end]
                notifications.show_info("Extracting blue intensity...")
                blue_intensity = extract_intensity(blue_channel, all_masks, max_cell_id, mode='sum')
                blue_df = pd.DataFrame(blue_intensity)
                blue_pivot = blue_df.pivot(index='cell_id', columns='frame', values='intensity')
                blue_pivot = self.add_class_info(blue_pivot, alignment_info)
                blue_pivot.sort_values(by='Class').to_excel(writer, sheet_name='Blue Channel (Original)')

                baseline_blue = blue_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_blue = blue_pivot.div(baseline_blue, axis=0)
                normalized_blue = self.add_class_info(normalized_blue, alignment_info)
                normalized_blue.sort_values(by='Class').to_excel(writer, sheet_name='Blue Channel (Normalized)')

                pivot_data['B'] = normalized_blue
                stats_blue = compute_summary_stats(normalized_blue)
                stats_blue.to_excel(writer, sheet_name='Statistics - Blue')

                # —— only if frequency_analysis_checkbox is checked ——
                if self.freq_analysis_checkbox.value and has_stack_b:
                    sampling_rate = 1.0  # Hz

                    freq_results = []
                    phase_results = []
                    # compute per-cell peak frequency, phase, and weighted frequency
                    for cell_id, row in normalized_blue.iterrows():
                        signal = row.values.astype(float)
                        fft_vals = np.fft.rfft(signal)
                        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
                        spectrum = np.abs(fft_vals)

                        # skip DC and very low freqs (bins 0,1,2)
                        spectrum_nodc = spectrum[1:]
                        freqs_nodc = freqs[1:]

                        # peak: find index in the sliced arrays, then offset by +1
                        peak_rel_idx = np.argmax(spectrum_nodc[2:]) + 2
                        peak_idx = peak_rel_idx + 1
                        peak_freq = freqs[peak_idx]
                        peak_power = spectrum[peak_idx]

                        # phase at peak
                        raw_phase = np.angle(fft_vals[peak_idx])  # [-π, π]
                        phase = (raw_phase + 2 * np.pi) % (2 * np.pi)

                        # weighted (centroid) frequency excluding DC
                        weighted_freq = np.sum(freqs_nodc * spectrum_nodc) / np.sum(spectrum_nodc)

                        freq_results.append({
                            'cell_id': int(cell_id),
                            'peak_frequency (Hz)': float(peak_freq),
                            'peak_power': float(peak_power),
                            'weighted_frequency (Hz)': float(weighted_freq)
                        })
                        phase_results.append({
                            'cell_id': int(cell_id),
                            'phase (rad)': phase
                        })

                    # write tables to Excel
                    freq_df = pd.DataFrame(freq_results)
                    phase_df = pd.DataFrame(phase_results)
                    freq_df.to_excel(writer, sheet_name='Frequency Analysis', index=False)
                    phase_df.to_excel(writer, sheet_name='Phase Analysis', index=False)

                    # build two maps from the first mask frame
                    mask0 = self.viewer.layers['Masks'].data[stack_start]
                    freq_map = np.zeros_like(mask0, dtype=float)
                    freq_dom_map = np.zeros_like(mask0, dtype=float)
                    phase_map = np.zeros_like(mask0, dtype=float)

                    for row in freq_results:
                        cid = row['cell_id']
                        freq_map[mask0 == cid] = row['weighted_frequency (Hz)']  # use weighted freq on map
                    for row in freq_results:
                        cid = row['cell_id']
                        freq_dom_map[mask0 == cid] = row['peak_frequency (Hz)'] # use peak freq on map for dominant frequency
                    for row in phase_results:
                        cid = row['cell_id']
                        phase_map[mask0 == cid] = row['phase (rad)']

                    # add both layers with full-range turbo colormap
                    self.viewer.add_image(
                        freq_map,
                        name='Freq Map',
                        colormap='turbo'
                    )
                    self.viewer.add_image(
                        freq_dom_map,
                        name='Freq Dom Map',
                        colormap='turbo'
                    )
                    self.viewer.add_image(
                        phase_map,
                        name='Phase Map',
                        colormap='turbo'
                    )

            # Process Green channel.
            if has_stack_g:
                self._set_nacha_progress(45, 'Processing Green channel...')
                green_channel = self.viewer.layers['Stack G'].data[stack_start:stack_end]
                notifications.show_info("Extracting green intensity...")
                green_intensity = extract_intensity(green_channel, all_masks, max_cell_id, mode='sum')
                green_df = pd.DataFrame(green_intensity)
                green_pivot = green_df.pivot(index='cell_id', columns='frame', values='intensity')
                green_pivot = self.add_class_info(green_pivot, alignment_info)
                green_pivot.sort_values(by='Class').to_excel(writer, sheet_name='Green Channel (Original)')
                # baseline_green = green_pivot.iloc[:, :basal_frame_num].mean(axis=1)
                baseline_green = green_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_green = green_pivot.div(baseline_green, axis=0)
                normalized_green = self.add_class_info(normalized_green, alignment_info)
                normalized_green.sort_values(by='Class').to_excel(writer, sheet_name='Green Channel (Normalized)')
                pivot_data['G'] = normalized_green
                stats_green = compute_summary_stats(normalized_green)
                stats_green.to_excel(writer, sheet_name='Statistics - Green')

            # Process NIR channel.
            if has_stack_nir:
                self._set_nacha_progress(65, 'Processing NIR channel...')
                nir_channel = self.viewer.layers['Stack NIR'].data[stack_start:stack_end]
                notifications.show_info("Extracting NIR intensity...")
                nir_intensity = extract_intensity(nir_channel, all_masks, max_cell_id, mode='sum')
                nir_df = pd.DataFrame(nir_intensity)
                nir_pivot = nir_df.pivot(index='cell_id', columns='frame', values='intensity')
                nir_pivot = self.add_class_info(nir_pivot, alignment_info)
                nir_pivot.sort_values(by='Class').to_excel(writer, sheet_name='NIR Channel (Original)')
                # baseline_nir = nir_pivot.iloc[:, :basal_frame_num].mean(axis=1)
                baseline_nir = nir_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_nir = nir_pivot.div(baseline_nir, axis=0)
                normalized_nir = self.add_class_info(normalized_nir, alignment_info)
                normalized_nir.sort_values(by='Class').to_excel(writer, sheet_name='NIR Channel (Normalized)')
                pivot_data['NIR'] = normalized_nir
                stats_nir = compute_summary_stats(normalized_nir)
                stats_nir.to_excel(writer, sheet_name='Statistics - NIR')

            # Process G/B Ratio if both Blue and Green channels exist and the ratio checkbox is checked.
            if has_stack_b and has_stack_g and self.ratio_checkbox.value:
                ratio_pivot = pivot_data['G'] / pivot_data['B']
                # baseline_ratio = ratio_pivot.iloc[:, :basal_frame_num].mean(axis=1)
                baseline_ratio = ratio_pivot.iloc[:, basal_frame_start:basal_frame_end].mean(axis=1)
                normalized_ratio = ratio_pivot.div(baseline_ratio, axis=0)
                normalized_ratio = self.add_class_info(normalized_ratio, alignment_info)
                normalized_ratio.sort_values(by='Class').to_excel(writer, sheet_name='Normalized G-B')
                pivot_data['Ratio'] = normalized_ratio
                stats_ratio = compute_summary_stats(normalized_ratio)
                stats_ratio.to_excel(writer, sheet_name='Statistics - G-B')

        self._set_nacha_progress(90, 'Plotting per-class signals...')
        notifications.show_info(f'Saved signal analysis results to {out_excel_path}')

        # Decide which channels to plot.
        if 'Ratio' in pivot_data:
            channels_to_plot = {'Ratio': pivot_data['Ratio']}
            if 'NIR' in pivot_data:
                channels_to_plot['NIR'] = pivot_data['NIR']
            overall_plot_title = "G/B Ratio" + (" + NIR" if 'NIR' in pivot_data else "")
        else:
            channels_to_plot = {}
            if 'B' in pivot_data:
                channels_to_plot['Blue'] = pivot_data['B']
            if 'G' in pivot_data:
                channels_to_plot['Green'] = pivot_data['G']
            if 'NIR' in pivot_data:
                channels_to_plot['NIR'] = pivot_data['NIR']
            overall_plot_title = "Channels: " + ", ".join(channels_to_plot.keys())

        # Get the list of classes from one of the pivot tables.
        sample_df = next(iter(pivot_data.values()))
        classes = sorted(sample_df['Class'].unique())
        num_classes = len(classes)

        import math
        ncols = 4  # 4 plots per row
        nrows = math.ceil(num_classes / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1:
            axs = np.array(axs).reshape(1, -1)
        axs = axs.flatten()

        # Plot each class in a separate subplot.
        # for ax, cls in zip(axs, tqdm(classes, desc="Plotting by class")):
        for ax, cls in zip(axs, classes):
            for ch_name, df in channels_to_plot.items():
                class_data = df[df['Class'] == cls]
                # Remove the 'Class' column.
                numeric_data = class_data.drop(columns=['Class'])
                if numeric_data.empty:
                    continue
                mean_curve = numeric_data.mean(axis=0)
                std_curve = numeric_data.std(axis=0)
                n = numeric_data.shape[0]
                # Compute standard error (SE) instead of SD.
                se_curve = std_curve / np.sqrt(n) if n > 0 else std_curve
                frames = mean_curve.index.astype(float)  # assuming frame numbers are numeric
                # Choose color based on channel.
                if ch_name in ['Blue', 'B']:
                    color = 'blue'
                elif ch_name in ['Green', 'G']:
                    color = 'green'
                elif ch_name == 'NIR':
                    color = 'red'
                elif ch_name == 'Ratio':
                    color = 'green'
                else:
                    color = None
                ax.plot(frames, mean_curve, label=ch_name, color=color)
                ax.fill_between(frames, mean_curve - se_curve, mean_curve + se_curve, color=color, alpha=0.3)
            num_cells = len(class_data)
            ax.set_title(f'Class {cls} (n={num_cells})')
            ax.set_xlabel("Frame")
            ax.set_ylabel("Normalized Intensity")
            ax.legend()

        # Remove extra subplots if there are more axes than classes.
        for i in range(num_classes, len(axs)):
            fig.delaxes(axs[i])
        plt.suptitle(overall_plot_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        self._set_nacha_progress(100, 'Calculate complete! 🎉')
        # NaCha is the last step of the walkthrough. Surface the celebration
        # dialog + total elapsed time here so users see it without having to
        # click a Next button.
        try:
            _show_walkthrough_celebration(self.viewer)
        except Exception as _ce:
            print(f'[NaCha celebration] {_ce}')

    def show_warning_dialog(self, message: str):
        mbox = QMessageBox(self.viewer.window._qt_window)  # 绑定到 napari 主窗
        mbox.setIcon(QMessageBox.Warning)
        mbox.setWindowTitle("Warning")
        mbox.setText(message + "\n\nDo you want to continue?")
        yes_btn = mbox.addButton("Continue", QMessageBox.YesRole)
        add_btn = mbox.addButton("Add", QMessageBox.NoRole)
        cancel_btn = mbox.addButton("Cancel", QMessageBox.RejectRole)
        mbox.exec_()
        clicked = mbox.clickedButton()
        if clicked is yes_btn:
            return "continue"
        elif clicked is add_btn:
            return "add"
        else:
            return "cancel"

    def add_class_info(self, df, alignment_info):
        df['Class'] = df.index.map(lambda x: alignment_info.loc[x, 'Class'] if x in alignment_info.index else 0)
        cols = df.columns.tolist()
        cols = ['Class'] + [col for col in cols if col != 'Class']
        return df[cols]

    def show_warning_dialog(self, message):
        root = tk.Tk()
        root.withdraw()
        response = messagebox.askquestion("Warning", message + "\n\nDo you want to continue?",
                                          icon='warning', type='yesnocancel')
        if response == 'yes':
            return "continue"
        elif response == 'no':
            return "add"
        else:
            return "cancel"
    def load_classification(self):
        # load the tif file which is masks from 0-14, 0 for bg, 1-14 for cells in different classes
        classification_file = self.classification_input.value
        resize = self.classification_resize.value
        if classification_file and os.path.exists(classification_file):
            self._set_nacha_progress(10, f'Reading {os.path.basename(classification_file)}...')
            masks = tiff.imread(classification_file)
            # Apply rotation to match BiosensorSeg convention. UI value -> np.rot90 k (CCW):
            #   0° -> 0, 90° CW -> 3, 180° -> 2, 270° CW -> 1
            rot_map = {'0°': 0, '90° CW': 3, '180°': 2, '270° CW': 1}
            try:
                rot_k = int(rot_map.get(str(self.classification_rotate.value), 0))
            except Exception:
                rot_k = 0
            self._set_nacha_progress(40, f'Rotating ({self.classification_rotate.value})...')
            if rot_k % 4:
                masks = np.rot90(masks, k=rot_k)
            self._set_nacha_progress(70, f'Resizing to {resize}×{resize}...')
            masks = cv2.resize(masks, (resize, resize), interpolation=cv2.INTER_NEAREST)
            self.viewer.add_labels(masks, name="Barcodes Masks classified", opacity=0.5)
            self._set_nacha_progress(100, f'Barcode loaded: {int(np.max(masks))} classes.')
            notifications.show_info(
                f'Loaded classification {np.max(masks)} classes, '
                f'rotated {self.classification_rotate.value}, resized to {resize}×{resize}.'
            )

    def align_classification(self):
        align_mask_frame = self.align_mask_frame.value
        # if none of those exist, show warning and return
        if 'Masks' not in self.viewer.layers or 'Barcodes Masks classified' not in self.viewer.layers:
            notifications.show_warning("Please load both 'Masks' and 'Barcodes Masks classified' layers before aligning.")
            return
        tracking_masks = self.viewer.layers['Masks'].data
        cls_masks = self.viewer.layers['Barcodes Masks classified'].data
        print('Shape of cls_masks:', cls_masks.shape)

        self._set_nacha_progress(5, 'Labeling barcode regions per class...')
        # Label connected barcode regions per class (avoid cross-class merge when touching).
        cls_masks_labeled = np.zeros_like(cls_masks, dtype=np.int32)
        region_to_class = {}
        region_area = {}
        next_rid = 1
        class_ids = np.unique(cls_masks)
        class_ids = class_ids[class_ids > 0]
        for cid in class_ids.tolist():
            m_c = cls_masks == int(cid)
            if not np.any(m_c):
                continue
            cc = label(m_c, connectivity=1)
            n_cc = int(cc.max())
            if n_cc <= 0:
                continue
            for k in range(1, n_cc + 1):
                rid = next_rid
                next_rid += 1
                m = cc == k
                cls_masks_labeled[m] = rid
                region_to_class[rid] = int(cid)
                region_area[rid] = int(np.count_nonzero(m))

        # Extract properties of the masks in the specified frame
        frame_props = regionprops(tracking_masks[align_mask_frame])

        # Create an array to store the new indices of the masks in tracking_masks
        cls_masks_aligned = np.zeros_like(tracking_masks)

        # Initialize a list to store alignment information for Excel
        alignment_info = []

        # Define the threshold level for considering a class in a mask
        threshold_level = self.align_thres_percent.value / 100
        tie_eps = 1e-12
        drop_ambiguous_tie = True

        # Loop over each mask in the tracking frame, use tqdm
        n_props = len(frame_props)
        self._set_nacha_progress(15, f'Aligning {n_props} cells to barcode classes...')
        for _pi, frame_prop in enumerate(tqdm(frame_props, desc='Aligning Classification Masks')):
            if n_props and _pi % max(1, n_props // 20) == 0:
                self._set_nacha_progress(
                    15 + int(80 * _pi / max(1, n_props)),
                    f'Aligning {_pi}/{n_props} cells...'
                )
            frame_mask_id = frame_prop.label
            current_mask = tracking_masks[align_mask_frame] == frame_mask_id
            overlapping_regions, counts = np.unique(cls_masks_labeled[current_mask], return_counts=True)

            # Remove background region id 0
            if 0 in overlapping_regions:
                zero_index = np.where(overlapping_regions == 0)
                overlapping_regions = np.delete(overlapping_regions, zero_index)
                counts = np.delete(counts, zero_index)

            if len(overlapping_regions) == 0:
                nearest_cls_id = 0  # No overlapping class
            else:
                # New rule: among overlapping barcode regions, choose the one with
                # highest coverage ratio inside this biosensor mask:
                #   overlap_pixels / barcode_region_area
                # If best ratio ties across different classes, drop this cell (Class=0).
                cand = []
                for rid, inter in zip(overlapping_regions.tolist(), counts.tolist()):
                    rid_i = int(rid)
                    area = max(region_area.get(rid_i, 0), 1)
                    ratio = float(inter) / float(area)
                    cand.append({
                        "rid": rid_i,
                        "inter": int(inter),
                        "ratio": float(ratio),
                        "cls": int(region_to_class.get(rid_i, 0)),
                    })
                best_ratio = max(c["ratio"] for c in cand)
                best = [c for c in cand if abs(c["ratio"] - best_ratio) <= tie_eps]

                if best_ratio < threshold_level:
                    nearest_cls_id = 0
                elif drop_ambiguous_tie and len({c["cls"] for c in best}) > 1:
                    nearest_cls_id = 0
                else:
                    # deterministic fallback among same-class ties
                    best_sorted = sorted(best, key=lambda c: (-c["inter"], c["rid"]))
                    nearest_cls_id = int(best_sorted[0]["cls"])

            # Save alignment info
            alignment_info.append((frame_mask_id, nearest_cls_id))

            # Apply the nearest class id to all frames
            for frame in range(len(tracking_masks)):
                cls_masks_aligned[frame][tracking_masks[frame] == frame_mask_id] = nearest_cls_id

        self._set_nacha_progress(95, 'Saving Bs2Code.xlsx...')
        # Save alignment info to Excel
        self.save_alignment_info(alignment_info)

        self.viewer.add_labels(cls_masks_aligned, name="Tracking masks cls", opacity=0.5)
        self._set_nacha_progress(100, 'Alignment done.')
        notifications.show_info(f'Classification masks aligned to tracking masks. Alignment information saved to {self.Bs2Code_save_path}. Label layer "Tracking masks cls" added.')
    def save_alignment_info(self, alignment_info):
        df = pd.DataFrame(alignment_info, columns=['Tracking Mask Index', 'Class'])
        self.Bs2Code_save_path = os.path.join(self.base_folder, 'Bs2Code.xlsx')
        df.to_excel(self.Bs2Code_save_path, index=False)
        print(f'Saved alignment information to {self.Bs2Code_save_path}')
        notifications.show_info(f'Saved alignment information to {self.Bs2Code_save_path}')

    def save_shifted_stack(self):
        folder = QFileDialog.getExistingDirectory(caption="Select Folder to Save Shifted Stack")
        if folder:
            channel_name = self.channel_to_shift.value
            if channel_name not in self.viewer.layers:
                notifications.show_warning(f"{channel_name} layer not found.")
                return
            stack = self.viewer.layers[channel_name].data
            # Construct a filename based on the channel name.
            filename = f"stack-{channel_name.lower().replace(' ', '-')}.tif"
            tiff.imsave(os.path.join(folder, filename), stack)
            print(f"Shifted stack saved to {folder}")
            notifications.show_info(f"Shifted stack saved to {folder}")

    def overexpo_visualize(self):
        overexpo_thres = self.overexpo_thres_param.value
        stack_b = self.viewer.layers['Stack B'].data
        stack_g = self.viewer.layers['Stack G'].data

        overexpo_mask_b = stack_b >= overexpo_thres
        overexpo_mask_g = stack_g >= overexpo_thres

        # If NIR exists, get its data and compute the overexposure mask.
        if 'Stack NIR' in self.viewer.layers:
            stack_nir = self.viewer.layers['Stack NIR'].data
            overexpo_mask_nir = stack_nir >= overexpo_thres

        # Add the overexposure masks as labels.
        self.viewer.add_labels(overexpo_mask_b, name="Overexposed Pixels B")
        self.viewer.add_labels(overexpo_mask_g, name="Overexposed Pixels G")
        if 'Stack NIR' in self.viewer.layers:
            self.viewer.add_labels(overexpo_mask_nir, name="Overexposed Pixels NIR")

        masks = self.viewer.layers['Masks'].data

        # If the masks layer is 2D (applied to all frames), replicate it for each frame.
        if masks.ndim == 2:
            masks = np.stack([masks] * overexpo_mask_b.shape[0], axis=0)

        # Prepare arrays to hold the overexposed mask labels.
        masks_with_overexpo_b = np.zeros_like(masks)
        masks_with_overexpo_g = np.zeros_like(masks)
        if 'Stack NIR' in self.viewer.layers:
            masks_with_overexpo_nir = np.zeros_like(masks)

        # Loop through each frame and each cell within the mask.
        for i in tqdm(range(masks.shape[0]), desc='Classifying Overexposed Pixels'):
            for j in range(1, masks[i].max() + 1):
                current_mask = masks[i]
                # Create a binary mask for the current cell.
                current_mask = np.where(current_mask == j, current_mask, 0)

                if np.any(current_mask[overexpo_mask_b[i][:]]):
                    if self.mask_256_checkbox.value:
                        masks_with_overexpo_b[i] += current_mask.astype(np.uint16) * j
                    else:
                        masks_with_overexpo_b[i] += current_mask.astype(np.uint8) * j

                if np.any(current_mask[overexpo_mask_g[i][:]]):
                    if self.mask_256_checkbox.value:
                        masks_with_overexpo_g[i] += current_mask.astype(np.uint16) * j
                    else:
                        masks_with_overexpo_g[i] += current_mask.astype(np.uint8) * j

                if 'Stack NIR' in self.viewer.layers:
                    if np.any(current_mask[overexpo_mask_nir[i][:]]):
                        if self.mask_256_checkbox.value:
                            masks_with_overexpo_nir[i] += current_mask.astype(np.uint16) * j
                        else:
                            masks_with_overexpo_nir[i] += current_mask.astype(np.uint8) * j

        # Add the overexposed masks as separate layers.
        self.viewer.add_labels(masks_with_overexpo_b, name="Masks with Overexposed pixels B")
        self.viewer.add_labels(masks_with_overexpo_g, name="Masks with Overexposed pixels G")
        if 'Stack NIR' in self.viewer.layers:
            self.viewer.add_labels(masks_with_overexpo_nir, name="Masks with Overexposed pixels NIR")

        notifications.show_info(
            'Overexposed pixels visualized in layers "Overexposed Pixels B", "Overexposed Pixels G"' +
            (', "Overexposed Pixels NIR"' if 'Stack NIR' in self.viewer.layers else '') +
            ', and overexposed masks visualized in layers "Masks with Overexposed pixels ..."'
        )

    def overexpo_discard(self):
        # Retrieve the current masks and the overexposed mask layers for blue and green.
        masks = self.viewer.layers['Masks'].data
        masks_with_overexpo_b = self.viewer.layers['Masks with Overexposed pixels B'].data
        masks_with_overexpo_g = self.viewer.layers['Masks with Overexposed pixels G'].data

        binary_overexpo_b = masks_with_overexpo_b > 0
        binary_overexpo_g = masks_with_overexpo_g > 0

        # Set all pixels in the masks corresponding to overexposed regions (B and G) to 0.
        masks[binary_overexpo_b] = 0
        masks[binary_overexpo_g] = 0

        # If the NIR overexposed layer exists, process it similarly.
        if 'Masks with Overexposed pixels NIR' in self.viewer.layers:
            masks_with_overexpo_nir = self.viewer.layers['Masks with Overexposed pixels NIR'].data
            binary_overexpo_nir = masks_with_overexpo_nir > 0
            masks[binary_overexpo_nir] = 0

        # Update the masks layer with the discarded pixels.
        self.viewer.layers['Masks'].data = masks
        self.viewer.layers['Masks'].refresh()
        notifications.show_info('Overexposed masks discarded from the masks layer')

    def read_masks(self):
        # Set base_folder to the parent directory of the mask folder
        self.base_folder = os.path.dirname(self.mask_folder.value)

        # --- Auto-detect "no tracking" case ---
        # If the tracking folder does not exist or is empty, fall back to the
        # single BiosensorSeg mask (*_seg.npy in the sample folder) and
        # broadcast it to every frame. This lets users skip B&P Tracker.
        mask_folder = str(self.mask_folder.value)
        mask_files: list[str] = []
        if os.path.isdir(mask_folder):
            mask_files = sorted(f for f in os.listdir(mask_folder) if f.endswith('.npy'))

        if not mask_files:
            # Look in the sample folder (parent of mask_folder, or mask_folder's parent tree)
            fallback = None
            for cand_base in [Path(mask_folder).parent, Path(mask_folder)]:
                if not cand_base or not cand_base.exists():
                    continue
                matches = (
                    sorted(cand_base.glob('*_seg_img_seg.npy'))
                    or sorted(cand_base.glob('*_seg.npy'))
                )
                if matches:
                    fallback = matches[0]
                    break
            if fallback is None:
                notifications.show_error(
                    "No tracking masks found, and no fallback *_seg.npy in the "
                    "sample folder. Either build tracking, or put the biosensor "
                    "seg.npy next to the sample folder."
                )
                return
            notifications.show_info(
                f"No tracking folder / files; using single mask {fallback.name} "
                f"and broadcasting to every frame."
            )
            mask_folder = str(fallback.parent)
            mask_files = [fallback.name]
            # update displayed folder so users see what we picked
            try:
                self.mask_folder.value = mask_folder
            except Exception:
                pass
            self.base_folder = mask_folder

        # Update the ratio calculation range to match mask count
        num_files = len(mask_files)
        self.ratio_calcu_range.value = [0, num_files - 1]

        single_file = (num_files == 1)
        all_masks = []

        for fname in mask_files:
            path = os.path.join(mask_folder, fname)
            try:
                # Try loading as a regular NumPy array
                data = np.load(path)
                if data.dtype != object:
                    masks = data.astype(np.uint16 if self.mask_256_checkbox.value else np.uint8)
                else:
                    # If dtype is object, treat as failure to trigger the Cellpose branch
                    raise ValueError
            except Exception:
                # Attempt to load as Cellpose output
                notifications.show_info(f"Failed to load {fname} as a standard array; trying Cellpose format...")
                try:
                    cell = np.load(path, allow_pickle=True).item()
                    masks = cell['masks'].astype(np.uint16 if self.mask_256_checkbox.value else np.uint8)
                except Exception:
                    notifications.show_error(f"Could not find 'masks' in {fname}; load failed!")
                    return

            all_masks.append(masks)

        # If only one mask file is present, duplicate it to match a multi‐frame image layer
        if single_file:
            n_frames = None
            for layer in self.viewer.layers:
                arr = getattr(layer, 'data', None)
                if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                    n_frames = arr.shape[0]
                    break
            if n_frames is None:
                notifications.show_warning(
                    "Only one mask file detected, but no multi‐frame image layer found. Loading single frame only."
                )
            else:
                notifications.show_info(f"Only one mask file detected; duplicating it for {n_frames} frames.")
                all_masks = [all_masks[0]] * n_frames

        # Stack into a (frames, height, width) array
        all_masks = np.stack(all_masks, axis=0)
        print(f"All masks shape: {all_masks.shape}")
        self.num_masks = all_masks.shape[0]
        # Add as a Labels layer in Napari
        self.viewer.add_labels(all_masks, name="Masks")
        notifications.show_info(
            f"{self.num_masks} mask frames loaded. "
            f"Maximum label in first frame: {np.max(all_masks[0])}"
        )

    def read_tif(self, name: str):
        if name == 'TIF for Tracking':
            tif_stack = tiff.imread(self.tif_input.value)
        elif name == 'Stack B':
            try:
                tif_stack = tiff.imread(self.stack_b_input.value)
                notifications.show_info(f'{name} read in successfully')
            except Exception as e:
                print(e)
                notifications.show_error("Error reading Stack B. Consider there's no Stack B needed.")
        elif name == 'Stack G':
            try:
                tif_stack = tiff.imread(self.stack_g_input.value)
                notifications.show_info(f'{name} read in successfully')
            except Exception as e:
                print(e)
                notifications.show_error("Error reading Stack G. Consider there's no Stack G needed.")
        elif name == 'Stack NIR':
            try:
                tif_stack = tiff.imread(self.stack_nir_input.value)
                notifications.show_info(f'{name} read in successfully')
            except Exception as e:
                print(e)
                notifications.show_error("Error reading Stack NIR. Consider there's no Stack NIR needed.")
        self.viewer.add_image(tif_stack, name=name)

    def save_tracking(self):
        folder = QFileDialog.getExistingDirectory(caption='Select Folder to Save Masks')
        if folder:
            tracked = self.viewer.layers['Tracked Masks'].data
            start, end = self.frame_start.value, self.frame_end.value
            end = min(end, tracked.shape[0] - 1)

            use_uint16 = bool(self.uint16_mode.value) or (tracked.max() > 255)
            save_dtype = np.uint16 if use_uint16 else np.uint8

            for i in range(start, end + 1):
                np.save(os.path.join(folder, f'{i:05d}.npy'),
                        tracked[i].astype(save_dtype))
            msg = f"Tracking results saved to {folder} (dtype={save_dtype})."
            show_info(msg);
            notifications.show_info(msg)

    def shift_stack(self):
        shift_r = self.shift_r_param.value
        shift_u = -self.shift_u_param.value  # Invert to match 'up' direction.

        # Get the channel selected from the dropdown.
        channel_name = self.channel_to_shift.value
        if channel_name not in self.viewer.layers:
            notifications.show_warning(f"{channel_name} layer not found.")
            return

        active_layer = self.viewer.layers[channel_name]
        stack_shift = active_layer.data

        # Perform the shifting.
        if stack_shift.ndim == 3:
            stack_shift = np.roll(stack_shift, shift_r, axis=2)  # Roll right/left on axis 2.
            stack_shift = np.roll(stack_shift, shift_u, axis=1)  # Roll up/down on axis 1.
        elif stack_shift.ndim == 2:
            stack_shift = np.roll(stack_shift, shift_r, axis=1)
            stack_shift = np.roll(stack_shift, shift_u, axis=0)

        active_layer.data = stack_shift
        active_layer.refresh()

        notifications.show_info(
            f"Shifted {active_layer.name} by {shift_r} pixels to the right and {-shift_u} pixels up")

    def toggle_revise_mode(self):
        undo_key = 'u'
        # make brush size to be 1
        self.viewer.layers['Masks'].brush_size = 1
        self.masks_layer = self.viewer.layers['Masks']
        if self.revise_mode_checkbox.value:
            self.masks_layer.mouse_drag_callbacks.append(self.on_click)
            self.viewer.bind_key(undo_key, self.on_undo, overwrite=True)
            notifications.show_info('Revise Mode enabled. Press Control + Click to delete masks from this frame on, Ctrl + Alt + Click to delete current mask. Press Shift + Click to plot signal cha   nges in the mask. Press u to undo. Do all the operations in the Masks layer.')

        else:
            self.masks_layer.mouse_drag_callbacks.remove(self.on_click)
            # self.viewer.unbind_key(undo_key, None)

    def on_click(self, layer, event):
        # self.masks_layer = layer
        if event.modifiers and 'Control' in event.modifiers:  # Control 按下
            history_keep = 2  # keep certain number of history, in case memory is not enough
            position = tuple(map(int, event.position))
            label = self.masks_layer.data[position]
            current_frame = self.viewer.dims.current_step[0]

            if len(self.masks_history) > history_keep:
                self.masks_history.pop(0)
            self.masks_history.append(self.masks_layer.data.copy())

            if 'Alt' in event.modifiers:
                # Ctrl+Alt+Click → delete the mask in current frame
                self.masks_layer.data[current_frame] = np.where(
                    self.masks_layer.data[current_frame] == label,
                    0,
                    self.masks_layer.data[current_frame]
                )
                notifications.show_info(f'Deleted mask {label} in frame {current_frame} only')
            else:
                # Ctrl+Click → delete the mask in all following frames
                self.masks_layer.data[current_frame:] = np.where(
                    self.masks_layer.data[current_frame:] == label,
                    0,
                    self.masks_layer.data[current_frame:]
                )
                notifications.show_info(f'Deleted mask {label} from frame {current_frame} to the end')

            self.masks_layer.refresh()
            self.masks_layer.selected_label = label  # Set the active label to the deleted label

        elif event.modifiers and event.modifiers[0] == 'Shift':
            # get the mask label and mask positions
            label = self.masks_layer.data[tuple(map(int, event.position))]
            mask_positions = self.masks_layer.data == label

            # Collect available stacks
            available_stacks = {}
            if 'Stack B' in self.viewer.layers:
                available_stacks['B'] = self.viewer.layers['Stack B'].data
            if 'Stack G' in self.viewer.layers:
                available_stacks['G'] = self.viewer.layers['Stack G'].data
            if 'Stack NIR' in self.viewer.layers:
                available_stacks['NIR'] = self.viewer.layers['Stack NIR'].data

            # If none of the stacks exist, notify and exit.
            if not available_stacks:
                notifications.show_info('No Stack B, G, or NIR layer available for plotting.')
                return

            # Use one available stack to ensure the mask is 3D.
            some_stack = next(iter(available_stacks.values()))
            if mask_positions.ndim == 2:
                mask_positions = np.repeat(mask_positions[np.newaxis, :, :], some_stack.shape[0], axis=0)

            # Compute the summed signals for each available channel.
            sum_signals = {}
            for key, stack in available_stacks.items():
                sum_signal = np.zeros(stack.shape[0])
                for i in range(stack.shape[0]):
                    sum_signal[i] = np.sum(stack[i][mask_positions[i]])
                sum_signals[key] = sum_signal

            # Prepare a list of signals and labels to plot.
            signals_to_plot = []
            labels_to_plot = []
            if 'B' in sum_signals:
                signals_to_plot.append(sum_signals['B'])
                labels_to_plot.append("Stack B")
            if 'G' in sum_signals:
                signals_to_plot.append(sum_signals['G'])
                labels_to_plot.append("Stack G")
            if 'NIR' in sum_signals:
                signals_to_plot.append(sum_signals['NIR'])
                labels_to_plot.append("Stack NIR")
            if self.freq_analysis_checkbox.value and 'B' in sum_signals:
                freqs = np.fft.rfftfreq(len(sum_signals['B']), d=1)  # Assuming unit time step

                spectrum = np.abs(np.fft.rfft(sum_signals['B']))
                # make DC to 0
                spectrum[0] = 0
                spectrum[1] = 0
                spectrum[0] = 0
                signals_to_plot.append(spectrum)
                bin_width = freqs[1] - freqs[0]  # Frequency bin width
                peak_freq = freqs[np.argmax(spectrum[3:])] + 3 * bin_width  # Skip DC and first bin
                weighted_freq = np.sum(freqs[1:] * spectrum[1:]) / np.sum(spectrum[1:])
                title_str = f"FFT of B Signal (Δf = {bin_width:.4f} Hz per bin), Peak Freq = {peak_freq:.2f} Hz, Weighted Freq = {weighted_freq:.2f} Hz"
                labels_to_plot.append(title_str)


            # Only compute the ratio if both B and G are available and the checkbox is checked.
            if self.ratio_checkbox.value and ('B' in sum_signals and 'G' in sum_signals):
                ratio_gb = sum_signals['G'] / sum_signals['B']
                signals_to_plot.append(ratio_gb)
                labels_to_plot.append("G/B Ratio")

            # If no valid signal was computed (should not happen, but check for safety), notify.
            if not signals_to_plot:
                notifications.show_info('No valid signal data available for plotting.')
                return

            # Call your plotting function with the signals and their corresponding labels.
            self.plot_widget.plot_signal(*signals_to_plot, titles=labels_to_plot)

            # Mark the mask as selected and notify the user.
            self.masks_layer.selected_label = label
            notifications.show_info(f'Plotted signal changes in mask {label} in the bottom plot widget')

    def on_undo(self, event):
        print('undo called')
        if self.masks_history:
            print('undo')
            notifications.show_info('Undo')
            # Revert to the previous state
            self.masks_layer.data = self.masks_history.pop()
            self.masks_layer.refresh()

    def apply_to_following_frames(self):
        self.masks_layer = self.viewer.layers['Masks']

        # Get the current mask label
        current_mask_label = self.masks_layer.selected_label # get the selected label
        print(f'current mask label: {current_mask_label}')

        current_frame = self.viewer.dims.current_step[0]
        # print(f'current frame: {current_frame}')

        # Get positions where current mask label is present in the current frame
        mask_positions = np.where(self.masks_layer.data[current_frame] == current_mask_label)
        # print(f'mask positions shape: {mask_positions[0].shape}')

        # Apply this mask label to the same positions in all following frames
        num_frames = self.masks_layer.data.shape[0]
        for frame in range(current_frame + 1, num_frames):
            # Set the mask label at the identified positions in the following frames
            self.masks_layer.data[frame][mask_positions] = current_mask_label

        # Refresh the mask layer to update the viewer
        self.masks_layer.refresh()
        notifications.show_info(f'Applied mask label {current_mask_label} to following frames ({current_frame + 1} to {num_frames - 1})')

    def apply_to_next_frame(self):
        self.masks_layer = self.viewer.layers['Masks']

        # Get the current mask label
        current_mask_label = self.masks_layer.selected_label  # get the selected label
        print(f'current mask label: {current_mask_label}')

        current_frame = self.viewer.dims.current_step[0]
        # print(f'current frame: {current_frame}')

        # Get positions where current mask label is present in the current frame
        mask_positions = np.where(self.masks_layer.data[current_frame] == current_mask_label)
        # print(f'mask positions shape: {mask_positions[0].shape}')

        # Apply this mask label to the same positions in the next frame

        self.masks_layer.data[current_frame + 1][mask_positions] = current_mask_label

        # Refresh the mask layer to update the viewer
        self.masks_layer.refresh()
        notifications.show_info(
            f'Applied mask label {current_mask_label} to the next frame ({current_frame + 1})')



import multiprocessing as mp
# NOTE: torch + track_anything_simple are imported LAZILY inside the tracker
# functions that actually need them. Importing them at module load would bring
# PyTorch / CUDA into the napari main process and corrupt vispy's shared GL
# context → `access violation reading 0x1C` on every Image paint. See
# _lazy_track_anything_imports() below.


def _lazy_track_anything_imports():
    """Return (torch, TrackingAnything, parse_augment). Called only inside
    tracker run paths so a user who never clicks Track never imports torch
    in the main process."""
    import torch as _torch
    from .track_anything_simple import TrackingAnything, parse_augment
    return _torch, TrackingAnything, parse_augment

# def find_contours(mask: np.ndarray) -> np.ndarray:
def find_contours(mask: np.ndarray):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(c, 1) for c in contours]
    for i, c in enumerate(contours):
        if not np.array_equal(c[0], c[-1]):
            contours[i] = np.vstack([c, c[0]])
    # return np.array(contours, dtype=np.int16)
    return contours
def find_nearest_masks(initial_masks, mask_contours, cell_dist):
    """
    initial_masks: 2D uint8 mask array
    mask_contours: dict[label] = list of np.arrays shape (Ni,2) 浮点或整型
    cell_dist: 距离阈值
    返回：[[label1,label2,...], [...], ...]  连通分量列表
    """

    labels = np.array([lab for lab in mask_contours.keys()], dtype=int)
    n = len(labels)
    # 1) prepare bounding boxes for each label
    bboxes = {}
    for lab in labels:
        bin_mask = (initial_masks == lab).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(bin_mask)
        bboxes[lab] = (x, y, w, h)

    # 2) initialize union-find structure
    parent = {lab: lab for lab in labels}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cd2 = cell_dist * cell_dist
    # 3)
    for i, j in combinations(labels, 2):
        x1, y1, w1, h1 = bboxes[i]
        x2, y2, w2, h2 = bboxes[j]
        # 3.1
        dx = max(0, x2 - (x1 + w1), x1 - (x2 + w2))
        dy = max(0, y2 - (y1 + h1), y1 - (y2 + h2))
        if dx*dx + dy*dy > cd2:
            continue  #

        # 3.2 candidate contours
        min_dist = np.inf
        for c1 in mask_contours[i]:
            for c2 in mask_contours[j]:
                d = cdist(c1, c2, 'euclidean').min()
                if d < min_dist:
                    min_dist = d
                    if min_dist < cell_dist:
                        break
            if min_dist < cell_dist:
                break

        if min_dist < cell_dist:
            union(i, j)

    groups = {}
    for lab in labels:
        root = find(lab)
        groups.setdefault(root, []).append(lab)
    # plot the masks and boxes in 1 figure, of all batches to

    return list(groups.values())



def batch_masks_by_distance(contours: dict[int, np.ndarray], cell_dist: float) -> list[list[int]]:
    labels = np.array(list(contours.keys()), dtype=np.uint8)
    batches, processed = [], set()
    for idx in labels:
        if idx in processed:
            continue
        stack, group = [idx], {idx}
        while stack:
            cur = stack.pop()
            for other in labels:
                if other in processed or other == cur:
                    continue
                c1_list, c2_list = contours[cur], contours[other]
                if c1_list.size and c2_list.size:
                    dmin = min(np.linalg.norm(p1 - p2)
                               for p1 in c1_list.reshape(-1,2)
                               for p2 in c2_list.reshape(-1,2))
                    if dmin < cell_dist:
                        group.add(other)
                        stack.append(other)
                        processed.add(other)
        batches.append(list(group))
    return batches


def compute_trajectories(mask_stack: np.ndarray) -> np.ndarray:
    labels = np.unique(mask_stack)
    labels = labels[labels != 0]
    n_frames = mask_stack.shape[0]
    traj = np.full((n_frames, labels.max()+1, 2), np.nan)
    for t in range(n_frames):
        for lbl in labels:
            traj[t, lbl] = center_of_mass((mask_stack[t]==lbl).astype(np.uint8))
    return traj


def _to_rgb_logscale(stack: np.ndarray) -> np.ndarray:
    scale, C = 0.005, 65535/np.log(1+65535*0.005)
    cmap = plt.get_cmap('viridis')
    rgb = np.zeros((*stack.shape,3), dtype=np.uint8)
    for i, img in enumerate(stack):
        im = np.log1p(img.astype(float)*scale)*C
        im = cv2.normalize(im, None, 0, 1, cv2.NORM_MINMAX)
        rgb[i] = (cmap(im)[...,:3]*255).astype(np.uint8)
    return rgb


def _load_stack(path: Union[str, Path]) -> np.ndarray:
    """Load an image or numpy stack, given a file path."""
    path_str = str(path)
    if path_str.lower().endswith(('.tif', '.tiff')):
        return tifffile.imread(path_str)
    elif path_str.lower().endswith('.npy'):
        return np.load(path_str)
    else:
        raise ValueError(f"Unsupported stack format: {path_str}")


def _load_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a mask file (.npy or image); return a 2D array.

    Supports two .npy formats:
    - dict wrapped in 0-d object array: {'masks': ndarray, ...}
    - raw 2D ndarray (produced by the BarcodeSeg widget / NP review script)
    """
    path_str = str(path)
    if path_str.lower().endswith('.npy'):
        obj = np.load(path_str, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
            item = obj.item()
            if isinstance(item, dict):
                return np.asarray(item.get('masks'))
        if isinstance(obj, np.ndarray) and obj.ndim == 2:
            return obj
        raise ValueError(f"Unsupported .npy mask format: {path_str}")
    img = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
    return img if img.ndim == 2 else img[..., 0]


# --- Utility functions ---
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def preprocess_rgb(img_stack, log_alpha, colormap, start_frame, end_frame, desc="Colorizing"):
    sub = img_stack[start_frame: end_frame + 1]
    cmap = plt.get_cmap(colormap)
    scale = float(log_alpha)

    gmin = float(sub.min())
    gmax = float(sub.max())

    T = sub.shape[0]
    H, W = sub.shape[1], sub.shape[2]
    rgb = np.zeros((T, H, W, 3), dtype=np.uint8)

    if gmax > gmin and scale > 0:
        vmax_log = np.log1p((gmax - gmin) * scale)
        inv_vmax_log = 1.0 / vmax_log
    else:
        vmax_log = 0.0
        inv_vmax_log = 0.0

    for i in tqdm(range(T), desc=desc):
        img = sub[i].astype(np.float32)

        im = img - gmin
        im = np.log1p(im * scale)
        if vmax_log > 0:
            im *= inv_vmax_log
        else:
            im[:] = 0.0

        im = np.clip(im, 0.0, 1.0)

        rgb[i] = (cmap(im)[..., :3] * 255).astype(np.uint8)

    return start_frame, rgb


def _chunk_batches(batches, max_size=240):
    """将大于 max_size 的 batch 切成更小块，避免一次对象数 >255。"""
    out = []
    for b in batches:
        if len(b) <= max_size:
            out.append(b)
        else:
            for i in range(0, len(b), max_size):
                out.append(b[i:i+max_size])
    return out

def _remap_template_to_local_ids(tmpl: np.ndarray, batch_labels: list[int]):
    """
    把裁剪区域内的模板标签，从全局ID映射到局部 1..K（uint8），
    返回 (local_tmpl_uint8, lut) 其中 lut[v_local] = v_global。
    """
    lut = np.zeros(256, dtype=np.int64)  # 仅局部使用；若K<=255足够
    local_tmpl = np.zeros_like(tmpl, dtype=np.uint8)
    for i, gid in enumerate(batch_labels, start=1):
        if i >= 256:
            raise ValueError("Single batch has >=256 objects after chunking, lower max_size.")
        local_tmpl[tmpl == gid] = i
        lut[i] = gid
    return local_tmpl, lut


# track_with_cutie removed; only TrackAnything backend remains

def _process_ta_batch(rgb_subset, initial_masks, batch, padding):
    # === 1) 外接框（按二值 OR，避免ID溢出） ===
    bin_map = np.zeros(initial_masks.shape, dtype=np.uint8)
    for oid in batch:
        bin_map |= (initial_masks == oid).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(bin_map)
    if w == 0 or h == 0:
        print(f"[WARN] Empty bbox for batch {batch}. Skip.")
        return np.zeros((len(rgb_subset), 0, 0), dtype=np.uint16), 0, 0, 0, 0

    y1 = max(y - padding, 0)
    x1 = max(x - padding, 0)
    y2 = min(y + h + padding, initial_masks.shape[0])
    x2 = min(x + w + padding, initial_masks.shape[1])

    cropped = [f[y1:y2, x1:x2] for f in rgb_subset]

    # === 2) 裁剪模板（全局ID），并重映射到 1..K（uint8）喂模型 ===
    tmpl_global = np.zeros((y2 - y1, x2 - x1), dtype=np.uint16)
    for oid in batch:
        sel = (initial_masks[y1:y2, x1:x2] == oid)
        if sel.any():
            tmpl_global[sel] = oid

    if tmpl_global.max() == 0:
        print(f"[WARN] No positive pixels in template for batch {batch}.")
        return np.zeros((len(rgb_subset), y2 - y1, x2 - x1), dtype=np.uint16), y1, y2, x1, x2

    local_ids = {gid: i for i, gid in enumerate(batch, start=1)}
    local_tmpl = np.zeros_like(tmpl_global, dtype=np.uint8)
    for gid, lid in local_ids.items():
        local_tmpl[tmpl_global == gid] = lid

    # === 3) 送入模型 ===
    pred = model.generator(cropped, local_tmpl.astype(np.uint8))
    model.xmem.clear_memory()

    # === 4) 规范化返回形态，得到 masks_local in {0..K} ===
    masks_local = None  # 最终应为 [T, Hc, Wc] 的整数标签

    def _to_uint8(arr):
        x = np.asarray(arr)
        if x.dtype == bool:
            x = x.astype(np.uint8)
        elif x.dtype == np.uint16 or x.dtype == np.int32:
            # 保留标签；若是 0/255 二值，也先不缩放
            pass
        return x

    pred_np = _to_uint8(pred)

    if isinstance(pred, list):
        # 可能是：list 长度 T，每个 [Hc,Wc]；或 list 长度 K，每个 [T,Hc,Wc]
        # 情况 A：list 长度 == T 且每项是 2D
        if len(pred) == len(cropped) and np.ndim(pred[0]) == 2:
            # 假设每帧是一张“多对象标签图”或二值图
            arr = np.stack([_to_uint8(p) for p in pred], axis=0)
            if arr.max() <= 1:  # 二值 => 无法区分对象，按“模板腐蚀分解”做个软合并
                # 用每个 lid 的模板做掩膜与预测相交，优先级按 lid 顺序
                masks_local = np.zeros_like(arr, dtype=np.uint8)
                for lid in range(1, len(batch)+1):
                    # 这里简单地把所有前景像素分配给当前 lid（可换成连通域/最近模板像素）
                    to_assign = (arr > 0) & (masks_local == 0)
                    masks_local[to_assign] = lid
            else:
                # 已经是 0..K 的标签
                masks_local = arr.astype(np.uint8)

        # 情况 B：list 长度 == K，每项是 [T,Hc,Wc] 的二值
        elif len(pred) == len(batch) and np.ndim(pred[0]) == 3:
            K = len(batch)
            T = pred[0].shape[0]
            Hc, Wc = pred[0].shape[1:]
            arr = np.stack([_to_uint8(p) for p in pred], axis=1)  # [T,K,Hc,Wc]
            # 取 argmax 通道作为 lid（背景为0）
            masks_local = np.zeros((T, Hc, Wc), dtype=np.uint8)
            # 先把二值化为 {0,1}
            arr01 = (arr > 0).astype(np.uint8)
            has_fg = arr01.any(axis=1)
            if has_fg.any():
                # 给前景像素按 lid 的先后顺序分配（或用 argmax）
                # 这里用 argmax，若多通道同为1，取通道索引最大的 lid
                lids = np.argmax(arr01, axis=1) + 1  # [T,Hc,Wc] in 1..K
                masks_local[has_fg] = lids[has_fg]
            # 背景保持 0

        else:
            raise ValueError(f"Unsupported list return shape for batch {batch}.")
    else:
        # 非 list：array-like
        if pred_np.ndim == 3:
            T, Hc, Wc = pred_np.shape
            if pred_np.max() <= 1:
                # 二值 [T,Hc,Wc]：无法区分对象 => 用模板引导分配
                masks_local = np.zeros_like(pred_np, dtype=np.uint8)
                fg = pred_np > 0
                # 简单策略：把所有前景分配给 lid=1（或根据模板最近标签分配——可替换为距离变换）
                # 为了更合理，这里按“最近模板标签”分配
                from scipy.ndimage import distance_transform_edt
                # 预先为每个 lid 生成模板二值
                tmpl_lid_bin = [(local_tmpl == lid).astype(np.uint8) for lid in range(1, len(batch)+1)]
                # 为每个 lid 计算距离图
                dists = np.stack([distance_transform_edt(1 - b) for b in tmpl_lid_bin], axis=0)  # [K,Hc,Wc]
                # 对每个前景像素选距离最近的 lid
                lids = np.argmin(dists, axis=0) + 1
                for t in range(T):
                    masks_local[t][fg[t]] = lids[fg[t]]
            else:
                # 已经是标签图
                masks_local = pred_np.astype(np.uint8)

        elif pred_np.ndim == 4:
            # 可能是 [T,K,Hc,Wc] 或 [K,T,Hc,Wc]
            if pred_np.shape[1] == len(batch):
                arr = (pred_np > 0).astype(np.uint8)  # [T,K,Hc,Wc]
                lids = np.argmax(arr, axis=1) + 1
                masks_local = lids.astype(np.uint8)
            elif pred_np.shape[0] == len(batch):
                arr = (pred_np > 0).astype(np.uint8).transpose(1,0,2,3)  # [T,K,Hc,Wc]
                lids = np.argmax(arr, axis=1) + 1
                masks_local = lids.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported array shape {pred_np.shape} for batch {batch}.")
        else:
            raise ValueError(f"Unsupported pred ndim={pred_np.ndim} for batch {batch}.")

    # 防御：如仍全 0，立即提示
    if masks_local is None or masks_local.max() == 0:
        print(f"[WARN] masks_local empty for batch {batch}. "
              f"pred shape={np.shape(pred)} dtype={getattr(pred, 'dtype', type(pred))}")
        return np.zeros((len(rgb_subset), y2 - y1, x2 - x1), dtype=np.uint16), y1, y2, x1, x2

    # === 5) 局部ID → 全局ID ===
    masks_global = np.zeros_like(masks_local, dtype=np.uint16)
    # 建立 LUT：lid -> gid
    lut = np.zeros(len(batch) + 1, dtype=np.uint32)
    for gid, lid in local_ids.items():
        lut[lid] = gid
    # 映射
    for lid in range(1, len(batch) + 1):
        gid = int(lut[lid])
        if gid == 0:
            continue
        masks_global[masks_local == lid] = gid

    print(f"Processed batch {batch} | local unique={np.unique(masks_local)[:10]} ...")
    return masks_global, y1, y2, x1, x2




def _init_worker(sam_ckpt, xmem_ckpt, e2fgvi_ckpt, args_dict):
    global model
    _torch, TrackingAnything, parse_augment = _lazy_track_anything_imports()
    args = parse_augment()
    args.__dict__.update(args_dict)
    model = TrackingAnything(sam_ckpt, xmem_ckpt, e2fgvi_ckpt, args)


def track_with_tasimple(
    img_stack: np.ndarray,
    initial_masks: np.ndarray,
    start_frame: int,
    end_frame: int,
    log_alpha: float,
    colormap: str,
    cell_dist: int = 4,
    padding: int = 50,
    num_processes: int = 4,
    sam_checkpoint: str = None,
    xmem_checkpoint: str = None,
    e2fgvi_checkpoint: str = None,
    max_patch_size: int = 400,   # NEW: patch 最大边长
    max_batch_size: int = 10,    # NEW: 一个 patch 里最多追多少个 cell
) -> np.ndarray:
    def _bbox_of_labels(mask2d: np.ndarray, ids, padding: int, H: int, W: int):
        # mask2d: label mask (H,W)
        bm = np.zeros_like(mask2d, dtype=np.uint8)
        for i in ids:
            bm |= (mask2d == i).astype(np.uint8)

        if bm.max() == 0:
            return None

        x, y, w, h = cv2.boundingRect(bm)
        y1 = max(0, y - padding)
        y2 = min(H, y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(W, x + w + padding)
        return (y1, y2, x1, x2)

    def _split_batch_by_limits(mask2d, batch, padding: int, max_patch_size: int, max_batch_size: int, H: int, W: int):
        """
        贪心拆分：尽量维持原 batch 的顺序，把它切成多个 sub-batch
        约束：
          - sub-batch cell 数 <= max_batch_size
          - sub-batch 的 bbox（含 padding）宽/高都 <= max_patch_size
        """
        sub_batches = []
        cur = []

        def fits(candidate_ids):
            bb = _bbox_of_labels(mask2d, candidate_ids, padding, H, W)
            if bb is None:
                return False
            y1, y2, x1, x2 = bb
            hh = y2 - y1
            ww = x2 - x1
            return (hh <= max_patch_size) and (ww <= max_patch_size)

        for cid in batch:
            # 如果空的，就先放进去（哪怕它自己很大，后面再兜底）
            if len(cur) == 0:
                cur = [cid]
                continue

            # 先检查 max_batch_size
            if len(cur) >= max_batch_size:
                sub_batches.append(cur)
                cur = [cid]
                continue

            # 再检查 max_patch_size（加入 cid 会不会超）
            cand = cur + [cid]
            if fits(cand):
                cur = cand
            else:
                # 放不下：先把当前 cur 封存，另起一组
                sub_batches.append(cur)
                cur = [cid]

        if len(cur) > 0:
            sub_batches.append(cur)

        return sub_batches

    if end_frame >= img_stack.shape[0]:
        end_frame = img_stack.shape[0] - 1
    full_out = np.zeros_like(img_stack, dtype=np.uint16)
    print(f'full_out shape: {full_out.shape}')

    offset, rgb_subset = preprocess_rgb(img_stack, log_alpha, colormap, start_frame, end_frame)
    mask_contours = {i: find_contours(initial_masks == i) for i in np.unique(initial_masks) if i != 0}
    batches = find_nearest_masks(initial_masks, mask_contours, cell_dist)

    H, W = initial_masks.shape  # 注意：这里是 mask 的尺寸
    batches_limited = []
    for b in batches:
        # 先按约束拆
        sub_bs = _split_batch_by_limits(
            initial_masks, b,
            padding=padding,
            max_patch_size=max_patch_size,
            max_batch_size=max_batch_size,
            H=H, W=W
        )
        batches_limited.extend(sub_bs)

    batches = batches_limited
    batches = _chunk_batches(batches, max_size=240)

    _torch, _TrackingAnything, parse_augment = _lazy_track_anything_imports()
    args = parse_augment()
    args.mask_save = True
    args.device = 'cuda' if _torch.cuda.is_available() else 'cpu'

    def run_serial():
        print("⚠️ Falling back to single-process execution.")
        _init_worker(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args.__dict__)
        return [_process_ta_batch(rgb_subset, initial_masks, b, padding) for b in batches]

    results = None
    if num_processes == 1:
        results = run_serial()
    else:
        try:
            print(f"🧵 Launching parallel pool with {num_processes} processes.")
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(
                processes=num_processes,
                initializer=_init_worker,
                initargs=(sam_checkpoint, xmem_checkpoint, e2fgvi_checkpoint, args.__dict__)
            )
            inputs = [(rgb_subset, initial_masks, b, padding) for b in batches]
            results = pool.starmap(_process_ta_batch, inputs)
            pool.close()
            pool.join()
        except Exception as e:
            print("❌ Parallel processing failed:", repr(e))
            traceback.print_exc()
            results = run_serial()

    out = np.zeros((end_frame - start_frame + 1, ) + initial_masks.shape, dtype=np.uint16)
    for masks, y1, y2, x1, x2 in results:
        out[:, y1:y2, x1:x2] = np.maximum(out[:, y1:y2, x1:x2], masks.astype(np.uint16))

    full_out[start_frame:end_frame+1, :, :] = out
    return full_out
def smooth_stack(input_tiff, output_tiff, window_size):
    # Calculate the averages
    # for i in range(input_tiff.shape[0]):
    for i in tqdm(range(input_tiff.shape[0])):
        # start = i
        start = min(i, input_tiff.shape[0] - window_size)
        end = min(i + window_size, input_tiff.shape[0])
        output_tiff[i] = np.mean(input_tiff[start:end], axis=0)

    # Replace the first 15 frames with the average of the first 30 frames
    output_tiff[:window_size//2] = np.mean(input_tiff[:window_size], axis=0)
    return output_tiff
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)


    if not os.path.exists(filepath):
        import requests
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")
    else:
        print(f"Checkpoint {filename} already exists at {filepath}, skipping download.")
    return filepath
def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        import gdown
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")
    else:
        print(f"Checkpoint {filename} already exists at {filepath}, skipping download.")

    return filepath
# class MultiModelTracker(Container):
def _first_existing(it):
    """Return the first Path from an iterable that exists, or None."""
    for p in it:
        if p and Path(p).exists():
            return Path(p)
    return None


class BPTracker(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__(layout='vertical')
        self.viewer = viewer
        self._stop_requested = False

        # Top-of-widget info: make it obvious this step is skippable.
        self.skip_hint = Label(
            value=(
                '💡 Tracking is OPTIONAL. If your cells do not move much, skip '
                'by clicking the blue "Next ▶" at the bottom — the next step '
                'will reuse the single seg.npy for every frame.'
            ),
        )
        try:
            self.skip_hint.native.setWordWrap(True)
            self.skip_hint.native.setStyleSheet(
                'QLabel {'
                '  background-color: #FFF8E1;'
                '  border: 1px solid #FFB300;'
                '  border-radius: 4px;'
                '  padding: 6px 8px;'
                '  color: #5D4037;'
                '  font-family: Calibri;'
                '}'
            )
        except Exception:
            pass

        # --- Build a smoothed multi-channel tracking stack from biosensor TIFs ---
        sample_dir = Path(r"J:/Mix16-N-P-260306-DCZ-2-1")
        fov_b = _first_existing(sample_dir.glob('FOV-*-b.tif'))
        fov_g = _first_existing(sample_dir.glob('FOV-*-g.tif'))
        fov_y = _first_existing(sample_dir.glob('FOV-*-y.tif'))
        self.tracker_stack_b = FileEdit(label='Stack B (track)', mode='r', filter='*.tif',
                                        value=str(fov_b) if fov_b else '')
        self.tracker_stack_g = FileEdit(label='Stack G (track)', mode='r', filter='*.tif',
                                        value=str(fov_g) if fov_g else '')
        self.tracker_stack_y = FileEdit(label='Stack Y (track)', mode='r', filter='*.tif',
                                        value=str(fov_y) if fov_y else '')
        self.tracker_use_b = CheckBox(text='Use B', value=True)
        self.tracker_use_g = CheckBox(text='Use G', value=True)
        self.tracker_use_y = CheckBox(text='Use Y', value=True)
        self.window_size = SpinBox(label='Smooth Window', value=10, min=1)
        self.build_stack_btn = PushButton(text='Build Tracking Stack (avg channels + smooth)')
        self.build_stack_btn.changed.connect(self._on_build_track_stack)
        _style_process_button(self.build_stack_btn)
        # Legacy button kept for direct smoothing of an already-loaded stack.
        self.smooth_btn = PushButton(text='Smooth Currently Loaded Stack')
        self.smooth_btn.changed.connect(self._on_smooth)

        # Default paths: prefer pre-built tracking stack; mask comes from BiosensorSeg step.
        default_tiff = str(sample_dir / f'stack-bgysum-smooth{self.window_size.value}.tif')
        # First biosensor seg.npy in the sample folder, if any.
        seg_candidates = sorted(sample_dir.glob('*_seg_img_seg.npy')) + sorted(sample_dir.glob('*_seg.npy'))
        default_mask = str(seg_candidates[0]) if seg_candidates else ''

        if Path(default_tiff).exists() and Path(default_mask).exists():
            stack = _load_stack(default_tiff)
            mask  = _load_mask(default_mask)
            self.viewer.add_image(stack, name='Default Stack')
            self.viewer.add_labels(mask.astype(np.uint16), name='Default Mask')
            self._stack = stack
            self._mask  = mask
        else:
            show_warning(
                "Default tracking stack / mask not found. Either click "
                "'Build Tracking Stack' above or pick Image Stack / Mask paths below."
            )
            self._stack = None
            self._mask  = None

        self.tiff_path = FileEdit(label='Image Stack', mode='r', value=default_tiff)
        self.mask_path = FileEdit(label='Mask', mode='r', value=default_mask)

        self.stack_layer = ComboBox(
            label='Use existing Image layer',
            choices=[layer.name for layer in self.viewer.layers if isinstance(layer, NapariImage)],
            nullable=True,
            value=None,
        )
        self.mask_layer = ComboBox(
            label='Use existing Labels layer',
            choices=[layer.name for layer in self.viewer.layers if isinstance(layer, Labels)],
            nullable=True,
            value=None,
        )
        self.stack_layer.changed.connect(self._on_select_stack_layer)
        self.mask_layer.changed.connect(self._on_select_mask_layer)
        self.viewer.layers.events.changed.connect(self._refresh_layer_choices)
        self.viewer.layers.events.inserted.connect(self._refresh_layer_choices)
        self.viewer.layers.events.removed.connect(self._refresh_layer_choices)
        # call once to populate
        self._refresh_layer_choices()

        self.frame_start = SpinBox(label='Start Frame', value=0, min=0)
        self.frame_end = SpinBox(label='End Frame', value=2000, min=0) # default max is 999
        self.log_alpha = FloatSlider(label='Log α', min=0.0, max=0.1, step=0.001, value=0.005)
        self.colormap = ComboBox(label='Colormap', choices=['rainbow','viridis','plasma','magma', 'gray'], value='rainbow') # can also add gray 'gray'
        self.preview_btn = PushButton(text='Preview Colorization')
        self.preview_btn.changed.connect(self._on_preview)
        self.visualize_btn = PushButton(text='Preview Batches')
        self.visualize_btn.changed.connect(self._on_preview_batches)
        self.notify = lambda msg: show_info(msg)
        # File inputs (only used if user wants to override)
        self.tiff_path = FileEdit(label='Image Stack', mode='r', value=default_tiff)
        self.mask_path = FileEdit(label='Mask', mode='r', value=default_mask)
        self.tiff_path.changed.connect(self._on_load)
        self.mask_path.changed.connect(self._on_load)
        # Backend selector now only TrackAnything
        self.backend = ComboBox(label='Backend', choices=['TrackAnything'], value='TrackAnything')
        self.cell_dist = SpinBox(label='Cell dist', value=4, min=1)
        self.padding = SpinBox(label='Padding', value=20, min=0)
        self.num_proc = SpinBox(label='Processes', value=1, min=1)
        self.chunk_size = SpinBox(label='Chunk length (frames, 0=off)', value=10, min=0)
        self.max_patch_size = SpinBox(label='Max patch size', value=400, min=64, max=4096)
        self.max_batch_size = SpinBox(label='Max batch size', value=10, min=1, max=512)
        self.track_btn = PushButton(text='Track')
        self.track_btn.changed.connect(self._on_track)
        _style_process_button(self.track_btn)
        self.stop_btn = PushButton(text='Stop Tracking')
        self.stop_btn.changed.connect(self._on_stop)
        self.save_btn = PushButton(text='Save Tracking')
        self.save_btn.changed.connect(self.save_tracking)
        self.uint16_mode = CheckBox(label='>255 masks', value=True)

        _tt(self.tracker_stack_b, 'Blue-channel time-lapse TIF used to build the tracking stack.')
        _tt(self.tracker_stack_g, 'Green-channel time-lapse TIF.')
        _tt(self.tracker_stack_y, 'Yellow / NIR-channel time-lapse TIF. Optional.')
        _tt(self.tracker_use_b, 'Include B channel in the averaged tracking stack.')
        _tt(self.tracker_use_g, 'Include G channel in the averaged tracking stack.')
        _tt(self.tracker_use_y, 'Include Y channel in the averaged tracking stack.')
        _tt(self.window_size,
            'Temporal smoothing window (frames). Larger = steadier '
            'tracking but loses fine motion.')
        _tt(self.build_stack_btn,
            'Averages the selected channels across frames and smooths with '
            'the window above. Writes stack-bgysum-smooth<N>.tif.')
        _tt(self.smooth_btn,
            'Temporal smooth on an existing loaded Image layer (no rebuild).')
        _tt(self.tiff_path, 'Tracking Image stack (overrides the auto-picked one).')
        _tt(self.mask_path,
            'Initial-frame mask to propagate through time. Usually seg_image_seg.npy '
            'from Biosensor Seg.')
        _tt(self.stack_layer, 'Pick an existing napari Image layer instead of loading from disk.')
        _tt(self.mask_layer, 'Pick an existing napari Labels layer as the initial mask.')
        _tt(self.frame_start, 'First frame to propagate from (0-based).')
        _tt(self.frame_end, 'Last frame (inclusive). Clipped to stack length.')
        _tt(self.log_alpha,
            'Log-scale α used when colourising the tracking stack for preview. '
            'Smaller = more aggressive log compression (brightens dim regions).')
        _tt(self.colormap, 'Colormap for the preview visualisation only.')
        _tt(self.preview_btn,
            'Colourises the tracking stack with current α + colormap for '
            'visual inspection before tracking.')
        _tt(self.visualize_btn,
            'Shows the per-batch bounding boxes that the tracker will use. '
            'Use to catch cells grouped into too-big or too-tight batches.')
        _tt(self.cell_dist,
            'Cells closer than this pixel distance are grouped into the '
            'same tracking batch. Smaller = more parallel batches.')
        _tt(self.padding,
            'Extra pixels added around each batch bounding box before '
            'cropping, so propagating masks have context.')
        _tt(self.num_proc,
            'Parallel tracker processes. Usually 1 (multi-proc rarely '
            'beats single on CUDA workloads).')
        _tt(self.chunk_size,
            'Split long stacks into chunks of this length for GPU memory '
            'safety. 0 = no chunking.')
        _tt(self.max_patch_size,
            'Upper bound on per-batch bounding-box edge length. Safety cap '
            'against out-of-memory on dense fields.')
        _tt(self.max_batch_size,
            'Upper bound on cells per tracking batch.')
        _tt(self.track_btn,
            'Run Track-Anything / XMem propagation from the initial mask '
            'through the stack. Slow; watch the progress bar.')
        _tt(self.stop_btn, 'Request the tracker to stop at the next batch boundary.')
        _tt(self.save_btn,
            'Save per-frame tracking masks to a folder you pick (one .npy '
            'per frame).')
        _tt(self.uint16_mode,
            'Save masks as uint16 instead of uint8 — needed if you have '
            'more than 255 tracked cells.')
        self.append(self.skip_hint)

        _append_section_divider(self, '— 🛠 Build tracking stack (optional) —')
        self.append(self.tracker_stack_b)
        self.append(self.tracker_stack_g)
        self.append(self.tracker_stack_y)
        self.append(self.tracker_use_b)
        self.append(self.tracker_use_g)
        self.append(self.tracker_use_y)
        self.append(self.window_size)
        self.append(self.build_stack_btn)

        _append_section_divider(self, '— 📁 Inputs for tracker —')
        self.append(self.tiff_path)
        self.append(self.mask_path)
        self.append(self.stack_layer)
        self.append(self.mask_layer)

        _append_section_divider(self, '— 👁 Preview —')
        self.append(self.frame_start)
        self.append(self.frame_end)
        self.append(self.log_alpha)
        self.append(self.colormap)
        self.append(self.smooth_btn)
        self.append(self.preview_btn)
        self.append(self.visualize_btn)

        _append_section_divider(self, '— ▶ Track —')
        self.append(self.backend)
        self.append(self.cell_dist)
        self.append(self.padding)
        self.append(self.num_proc)
        self.append(self.chunk_size)
        self.append(self.max_patch_size)
        self.append(self.max_batch_size)
        self.append(self.uint16_mode)
        self.append(self.track_btn)
        self.append(self.stop_btn)
        self.append(self.save_btn)

        _add_next_button(self, viewer)
        _tighten_container(self)
        _add_logo_header(
            self,
            title='B&P Tracker',
            subtitle='Powered by Track-Anything / XMem (biology-tuned)',
            logo_path=_TRACK_ANYTHING_LOGO_PATH,
            logo_size=30,
        )
        # viewer.window.add_dock_widget(self, area='right', name='Multi-Model Tracker')

    def _on_stop(self):
        # 标记一下，下一次 chunk 检查到就停
        self._stop_requested = True
        notifications.show_info("Stop requested. Tracking will stop after current chunk.")

    def _refresh_layer_choices(self, event=None):
        names_img = [layer.name for layer in self.viewer.layers if isinstance(layer, NapariImage)]
        names_lbl = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]

        self.stack_layer.choices = names_img
        self.mask_layer.choices = names_lbl

        if self.stack_layer.value not in names_img:
            self.stack_layer.value = None
        if self.mask_layer.value not in names_lbl:
            self.mask_layer.value = None


    def _on_select_stack_layer(self):
        """Callback when the user picks an existing Image layer."""
        name = self.stack_layer.value
        if name is None:
            return
        data = self.viewer.layers[name].data
        # if it’s a Dask array, compute it now:
        if hasattr(data, 'compute'):
            data = data.compute()
        self._stack = data
        self.notify(f"Stack set from layer “{name}”")

    def _on_select_mask_layer(self):
        """Callback when the user picks an existing Labels layer."""
        name = self.mask_layer.value
        if name is None:
            return
        layer = self.viewer.layers[name]
        if isinstance(layer, Labels):
            self._mask = layer.data
            self.notify(f"Mask set from layer “{name}”")
    def _on_smooth(self):
        if self._stack is None:
            show_warning("Please load an Image Stack first.")
            return

        w = self.window_size.value
        smoothed = smooth_stack(self._stack, np.zeros_like(self._stack), w)
        self._smoothed_stack = smoothed
        self.viewer.add_image(smoothed, name=f'Smoothed (w={w})')
        show_info(f'Stack smoothed (window={w})')

    def _on_build_track_stack(self):
        """Average selected B/G/Y channel stacks frame-by-frame, temporally
        smooth, save as stack-<tag>-smooth<w>.tif, and load into napari."""
        selected = []
        for use, path, tag in (
            (self.tracker_use_b.value, self.tracker_stack_b.value, 'b'),
            (self.tracker_use_g.value, self.tracker_stack_g.value, 'g'),
            (self.tracker_use_y.value, self.tracker_stack_y.value, 'y'),
        ):
            if not use:
                continue
            p = Path(str(path)) if path else None
            if p is None or not p.is_file():
                show_warning(f'Stack {tag.upper()} path invalid: {p}')
                return
            selected.append((tag, p))
        if not selected:
            show_warning('Pick at least one channel (B/G/Y) to build the tracking stack.')
            return

        try:
            accum = None
            shape = None
            for tag, p in selected:
                arr = np.asarray(tifffile.imread(str(p)), dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                if shape is None:
                    shape = arr.shape
                elif arr.shape != shape:
                    show_warning(f'Channel shape mismatch: {p.name} is {arr.shape}, expected {shape}.')
                    return
                accum = arr if accum is None else accum + arr
            accum = accum / float(len(selected))  # per-frame channel mean

            w = int(self.window_size.value)
            if w > 1:
                smoothed = smooth_stack(accum, np.zeros_like(accum), w)
            else:
                smoothed = accum
            smoothed_u16 = np.clip(np.round(smoothed), 0, 65535).astype(np.uint16)

            tag_str = ''.join(t for t, _ in selected)
            sample_dir = Path(str(self.tracker_stack_b.value or self.tracker_stack_g.value or
                                 self.tracker_stack_y.value)).parent
            out_path = sample_dir / f'stack-{tag_str}sum-smooth{w}.tif'
            tifffile.imwrite(str(out_path), smoothed_u16, imagej=True)

            self._stack = smoothed_u16
            self._smoothed_stack = smoothed_u16
            layer_name = f'track_stack ({tag_str}, w={w})'
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = smoothed_u16
            else:
                self.viewer.add_image(smoothed_u16, name=layer_name)
            try:
                self.tiff_path.value = str(out_path)
            except Exception:
                pass
            show_info(f'Tracking stack saved: {out_path.name} '
                      f'(channels={tag_str.upper()}, w={w}, shape={smoothed_u16.shape})')
        except Exception as e:
            show_warning(f'Build tracking stack failed: {e}')
            traceback.print_exc()
    def save_tracking(self):
        # save to a folder of npy files, 00000-....npy
        # pop out a dialog to choose the folder
        folder = QFileDialog.getExistingDirectory(caption='Select Folder to Save Masks')
        if folder:  # If a folder was chosen
            # save the tracked masks layer to the folder
            tracked_masks = self.viewer.layers['Tracked Masks'].data
            # for i in range(tracked_masks.shape[0]):
            # only save the beginning-end frames
            start, end = self.frame_start.value, self.frame_end.value
            # end = min (end, tracked_masks.shape[0] - 1)  # Ensure end is within bounds
            if end >= tracked_masks.shape[0]:
                end = tracked_masks.shape[0] - 1
            if self.uint16_mode.value:
                for i in range(start, end + 1):
                    np.save(os.path.join(folder, f'{i:05d}.npy'), tracked_masks[i].astype(np.uint16))
            else:
                for i in range(start, end + 1):
                    np.save(os.path.join(folder, f'{i:05d}.npy'), tracked_masks[i].astype(np.uint8))
            show_info(f"Tracking results saved to {folder}")
            notifications.show_info(f"Tracking results saved to {folder}")

    def _on_preview(self):
        # img = _load_stack(self.tiff_path.value)
        # load the smoothed one, if not exists, use the original stack, if not exists, show warning
        if not hasattr(self, '_stack'):
            show_warning("Please load an Image Stack first.")
            return
        img = self._smoothed_stack if hasattr(self, '_smoothed_stack') else self._stack
        start, end = self.frame_start.value, self.frame_end.value
        if end >= img.shape[0]:
            end = img.shape[0] - 1
        _, rgb = preprocess_rgb(img, self.log_alpha.value, self.colormap.value, start, end)
        self.viewer.add_image(rgb, name='Preprocessed Preview')

    def _on_preview_batches(self):
        # masks = _load_mask(self.mask_path.value)
        if not hasattr(self, '_mask'):
            show_warning("Please load a Mask first.")
            return
        masks = self._mask
        dist, pad = self.cell_dist.value, self.padding.value
        contours = {i: find_contours(masks==i) for i in np.unique(masks) if i!=0}
        # for i, ctr in contours.items():
        #     if len(ctr) < 2:
        #         continue
        #     self.viewer.add_shapes([ctr], shape_type='path', name=f'Contour {i}', edge_color='yellow', visible=False)
        batches = find_nearest_masks(masks, contours, dist)
        self.notify(f"Found {len(batches)} batches with cell distance {dist}.")
        shapes = []
        for batch in batches:
            bm = np.zeros_like(masks, dtype=np.uint8)
            for m in batch:
                bm |= (masks == m).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(bm)
            shapes.append([[y-pad, x-pad], [y+h+pad, x+w+pad]])
        self.viewer.add_shapes(shapes, shape_type='rectangle', name='Batch Boxes')

    def _on_load(self, path: pathlib.Path):
        try:
            # decide by comparing which widget sent it
            if path == self.tiff_path.value:
                stack = _load_stack(path)
                self.viewer.add_image(stack, name='Stack (loaded)')
                self._stack = stack
            else:
                mask = _load_mask(path)
                self.viewer.add_labels(mask.astype(np.uint16), name='Mask (loaded)')
                self._mask = mask
        except Exception as e:
            show_warning(f"Failed to load: {e}")

    def _on_track(self):
        if self._stack is None or self._mask is None:
            show_warning("Please load both Image Stack and Mask before tracking.")
            return

        # 每次开始追踪前，清空停止标记
        self._stop_requested = False

        # ---- 1. 准备 checkpoint（保持原来的逻辑） ----
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        xmem_checkpoint = "XMem-s012.pth"
        xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
        e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"
        e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"

        folder = str(Path(__file__).parent / 'checkpoints')
        SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
        xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
        e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)

        # ---- 2. 选择使用原 stack 还是 smoothed stack ----
        if hasattr(self, '_smoothed_stack') and self._smoothed_stack is not None:
            stack = self._smoothed_stack
            notifications.show_info('Using smoothed stack for tracking.')
        else:
            stack = self._stack

        masks = self._mask
        # ===============================
        # Sanity check: mask vs stack
        # ===============================
        if masks.ndim != 2:
            show_warning(
                f"Mask must be 2D (H×W), but got shape {masks.shape}.\n"
                "Please provide a single-frame segmentation mask."
            )
            return

        if stack.ndim != 3:
            show_warning(
                f"Stack must be 3D (T×H×W), but got shape {stack.shape}."
            )
            return

        H, W = stack.shape[1], stack.shape[2]
        mH, mW = masks.shape

        if (H, W) != (mH, mW):
            show_warning(
                "Mask / Image size mismatch!\n\n"
                f"Image stack size: (H={H}, W={W})\n"
                f"Mask size:        (H={mH}, W={mW})\n\n"
                "They must be exactly the same.\n"
                "Please check whether you loaded the correct mask\n"
                "or resized the image stack without resizing the mask."
            )
            return

        # ---- 3. 起止帧 & chunk 长度 ----
        start_frame = self.frame_start.value
        end_frame = self.frame_end.value
        if end_frame >= stack.shape[0]:
            end_frame = stack.shape[0] - 1

        total_frames = end_frame - start_frame + 1
        chunk = int(self.chunk_size.value)  # 0 = 不分段

        show_info('Starting tracking...')
        notifications.show_info(
            f'Tracking started. Frames [{start_frame}, {end_frame}] '
            f'({"no chunks" if chunk <= 0 else f"chunk={chunk}"})'
        )

        time_start = time.time()

        # ---- 4. 不分段：保持原来的行为 ----
        if chunk <= 0 or chunk >= total_frames:
            params = dict(
                start_frame=start_frame,
                end_frame=end_frame,
                log_alpha=self.log_alpha.value,
                colormap=self.colormap.value,
                cell_dist=self.cell_dist.value,
                padding=min(self.padding.value, *masks.shape),
                num_processes=self.num_proc.value,
                sam_checkpoint=SAM_checkpoint,
                xmem_checkpoint=xmem_checkpoint,
                e2fgvi_checkpoint=e2fgvi_checkpoint,
                max_patch_size=int(self.max_patch_size.value),
                max_batch_size=int(self.max_batch_size.value),
            )
            result = track_with_tasimple(stack, masks, **params)

        # ---- 5. 分段追踪：每一段重启 memory，并在段与段之间检查 stop ----
        else:
            result = np.zeros_like(stack, dtype=np.uint16)

            current_start = start_frame
            current_init_masks = masks.copy()

            while current_start <= end_frame:
                # 如果中途用户点了 Stop，就不要再跑新的 chunk 了
                if self._stop_requested:
                    notifications.show_info(
                        f"Tracking stopped by user at frame {current_start}. "
                        f"Results up to previous chunk are kept."
                    )
                    break

                current_end = min(current_start + chunk - 1, end_frame)

                notifications.show_info(
                    f'Tracking chunk [{current_start}, {current_end}]...'
                )

                params = dict(
                    start_frame=current_start,
                    end_frame=current_end,
                    log_alpha=self.log_alpha.value,
                    colormap=self.colormap.value,
                    cell_dist=self.cell_dist.value,
                    padding=min(self.padding.value, *masks.shape),
                    num_processes=self.num_proc.value,
                    sam_checkpoint=SAM_checkpoint,
                    xmem_checkpoint=xmem_checkpoint,
                    e2fgvi_checkpoint=e2fgvi_checkpoint,
                    max_patch_size=int(self.max_patch_size.value),
                    max_batch_size=int(self.max_batch_size.value),
                )

                # 这一段从 current_start → current_end 单独跑
                chunk_result = track_with_tasimple(stack, current_init_masks, **params)

                # 写入结果
                result[current_start:current_end + 1] = chunk_result[current_start:current_end + 1]

                if current_end == end_frame:
                    break

                # 下一段的初始 mask = 本段最后一帧的追踪结果
                next_init = chunk_result[current_end]
                current_init_masks = next_init.copy()
                current_start = current_end  # 下一段从 current_end 开始

        time_end = time.time()

        # ---- 6. 加到 napari 里 ----
        use_uint16 = bool(self.uint16_mode.value) or (self._mask.max() > 255)
        result_dtype = np.uint16 if use_uint16 else np.uint8
        self.viewer.add_labels(result.astype(result_dtype), name='Tracked Masks')

        notifications.show_info(
            f"Tracking completed in {time_end - time_start:.2f} seconds. "
            f"(Stopped early: {self._stop_requested}). "
            f"You can now save the tracking results or visualize them further."
        )


# =========================================================================
# Cellpose segmentation widgets
# =========================================================================

def _reduce_stack_to_2d(arr: np.ndarray, mode: str) -> np.ndarray:
    """Reduce a 3D stack (T, H, W) to a single 2D image for segmentation."""
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
    if mode == 'first':
        return arr[0]
    if mode == 'mean':
        return arr.mean(axis=0)
    if mode == 'max':
        return arr.max(axis=0)
    if mode == 'sum':
        return arr.sum(axis=0)
    raise ValueError(f"Unknown reduction mode: {mode}")


def _run_cellpose(image_2d: np.ndarray, *, model_type: str, diameter: float,
                  flow_threshold: float, cellprob_threshold: float, use_gpu: bool) -> np.ndarray:
    """Run Cellpose on a 2D image; return integer mask (0 = background)."""
    from cellpose import models
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)
    dia = diameter if diameter and diameter > 0 else None
    masks, _flows, _styles, _diams = model.eval(
        image_2d,
        diameter=dia,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks


def _save_seg_npy(dest_path: Path, masks: np.ndarray) -> None:
    """Save a mask dict compatible with the plugin's _load_mask() reader."""
    np.save(str(dest_path), {'masks': masks.astype(np.uint32)})


# ---- BarcodeSeg N/P model configuration ----
_BARCODE_MODEL_ROOT = Path(r"G:/BC-FLIM-S/LYH")  # where _cellpose_finetune_* folders live
_CELLPOSE_SRC_PATH = Path(r"D:/PKU_STUDY/DeepLearining/BC-FLIM/cellpose-main")
_DEFAULT_N_MODEL = "NinNC-260328-1"
_DEFAULT_P_MODEL = "CinNC-260328-1"
_DEFAULT_N_DIAMETER = 55.0
_DEFAULT_P_DIAMETER = 92.0
_CELLPOSE_BUILTIN = {"cyto", "cyto2", "nuclei", "tissuenet", "livecell", "general"}


def _resolve_barcode_model_path(model_type: str, extra_roots=()) -> "Path | None":
    """Resolve a Cellpose custom model name to a concrete file path."""
    p = Path(model_type)
    if p.exists():
        return p
    cached = Path.home() / ".cellpose" / "models" / model_type
    if cached.exists():
        return cached
    for d in sorted(_BARCODE_MODEL_ROOT.glob("_cellpose_finetune_*")):
        c = d / model_type / "models" / model_type
        if c.exists():
            return c
    for root in extra_roots:
        for cand in (
            Path(root) / "_finetune" / model_type / "models" / model_type,
            Path(root) / model_type / "models" / model_type,
        ):
            if cand.exists():
                return cand
    return None


def _build_barcode_cellpose(model_type: str, use_gpu: bool, extra_roots=()):
    """Return a cellpose.models.CellposeModel for built-in or custom name."""
    import sys as _sys
    if _CELLPOSE_SRC_PATH.exists() and str(_CELLPOSE_SRC_PATH) not in _sys.path:
        _sys.path.insert(0, str(_CELLPOSE_SRC_PATH))
    from cellpose import models as cp_models
    try:
        import torch
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    except Exception:
        device = None
    mt = model_type.strip()
    if mt in _CELLPOSE_BUILTIN:
        return cp_models.CellposeModel(gpu=use_gpu, device=device, model_type=mt)
    local = _resolve_barcode_model_path(mt, extra_roots=extra_roots)
    if local is None:
        raise FileNotFoundError(f"Cellpose model '{mt}' not found locally.")
    return cp_models.CellposeModel(
        gpu=use_gpu, device=device, model_type=None, pretrained_model=str(local),
    )


def _run_barcode_cellpose(model, img2d: np.ndarray, diameter: float) -> np.ndarray:
    out = model.eval(img2d, diameter=diameter, channels=[0, 0])
    masks = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(masks, list):
        masks = masks[0]
    return np.asarray(masks, dtype=np.uint16)


def _list_sample_finetuned_models(sample_dir: Path) -> list[str]:
    """Return names of models fine-tuned locally under <sample>/_finetune/.

    Kept for backward-compatibility; new code should call
    _list_all_custom_models(), which scans both per-sample and the shared
    BC-FLIM-S model root, and sorts by name-match + recency.
    """
    out: list[str] = []
    fd = Path(sample_dir) / "_finetune"
    if fd.is_dir():
        for mdir in sorted(fd.iterdir()):
            p = mdir / "models" / mdir.name
            if p.exists():
                out.append(mdir.name)
    return out


def _list_all_custom_models(sample_dirs=(), target_hint: str = "") -> list[str]:
    """Return custom Cellpose model names discovered in all known locations.

    Scans:
      * Each ``<sample_dir>/_finetune/<name>/models/<name>``
      * The shared ``_BARCODE_MODEL_ROOT/_cellpose_finetune_*/<name>/models/<name>``
      * ``~/.cellpose/models/*`` (cache — anything not in the builtin set)

    Ordering (so the most relevant model is always on top of the dropdown):
      1. Models whose name matches ``target_hint`` (e.g. "n" for the N
         dropdown, "p"/"c" for P, "bs"/"biosensor" for biosensor). Among
         these, most-recently-modified first.
      2. Everything else, most-recently-modified first.

    ``target_hint`` is lower-cased; pass "" to skip the target-match bucket
    and just return recency order.
    """
    hint = (target_hint or "").lower().strip()
    # {name: (is_hit, mtime)}
    seen: dict[str, tuple[int, float]] = {}

    def _consider(path: Path, name: str) -> None:
        if not path.exists():
            return
        if name in _CELLPOSE_BUILTIN:
            return
        try:
            mt = path.stat().st_mtime
        except OSError:
            mt = 0.0
        low = name.lower()
        is_hit = 0
        if hint:
            if hint == 'n' and low.startswith('n'):
                is_hit = 1
            elif hint in {'p', 'c'} and (low.startswith('p') or low.startswith('c')):
                is_hit = 1
            elif hint in {'bs', 'biosensor'} and (
                low.startswith('bs') or 'biosensor' in low
            ):
                is_hit = 1
        prev = seen.get(name)
        if prev is None or mt > prev[1]:
            seen[name] = (max(prev[0] if prev else 0, is_hit), mt)

    for sd in sample_dirs:
        try:
            sd = Path(sd) if sd else None
        except Exception:
            sd = None
        if sd and sd.is_dir():
            fd = sd / "_finetune"
            if fd.is_dir():
                for mdir in fd.iterdir():
                    if not mdir.is_dir():
                        continue
                    _consider(mdir / "models" / mdir.name, mdir.name)

    try:
        root_iter = list(_BARCODE_MODEL_ROOT.glob("_cellpose_finetune_*"))
    except Exception:
        root_iter = []
    for d in root_iter:
        if not d.is_dir():
            continue
        for mdir in d.iterdir():
            if not mdir.is_dir():
                continue
            _consider(mdir / "models" / mdir.name, mdir.name)

    try:
        cache_root = Path.home() / ".cellpose" / "models"
        if cache_root.is_dir():
            for p in cache_root.iterdir():
                if p.is_file():
                    _consider(p, p.name)
    except Exception:
        pass

    # Sort: hit desc, then mtime desc, then name for stability.
    ordered = sorted(
        seen.items(),
        key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]),
    )
    return [name for name, _ in ordered]


class BarcodeSeg(Container):
    """Single-FOV N/P segmentation on the barcode intensity (sum) image.

    Workflow after PTU Reader:
    1. Pick sample folder → auto-finds intensity/*_sum.tif
    2. Click "Auto Segment N & P" → runs both Cellpose models, saves *_seg_n.npy / *_seg_p.npy
    3. Edit masks interactively in napari (right-click draw, Ctrl+click delete, Z/X toggle, etc.)
    4. Optional: "Fine-tune" on the current edited masks; new model is saved in the sample
       folder and added to the model dropdown for reuse.
    """
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(layout='vertical')
        self.viewer = viewer
        self._model_cache: dict = {}
        self._current_img: np.ndarray | None = None
        self._current_src: Path | None = None
        self._draw_state = {"layer": "", "pts": [], "active": False}
        self._contrast_ratios = [1.0, 0.5, 0.25, 0.12, 0.06]
        self._contrast_idx = 0

        self.sample_dir = FileEdit(
            label='Sample Folder', mode='d',
            value=r'J:/Mix16-N-P-260306-DCZ-2-1',
        )
        self.tif_override = FileEdit(
            label='Override TIF (optional)', mode='r',
            filter='*.tif', value='',
        )

        self.n_model = ComboBox(
            label='N model',
            choices=[_DEFAULT_N_MODEL, 'nuclei'],
            value=_DEFAULT_N_MODEL,
        )
        self.p_model = ComboBox(
            label='P model',
            choices=[_DEFAULT_P_MODEL, 'cyto2'],
            value=_DEFAULT_P_MODEL,
        )
        self.n_diameter = FloatSpinBox(label='N diameter (px)', min=5, max=300, step=1, value=_DEFAULT_N_DIAMETER)
        self.p_diameter = FloatSpinBox(label='P diameter (px)', min=5, max=300, step=1, value=_DEFAULT_P_DIAMETER)
        self.use_gpu = CheckBox(text='Use GPU', value=True)

        self.run_btn = PushButton(text='Auto Segment N & P')
        self.run_btn.changed.connect(self._on_run_auto)
        _style_process_button(self.run_btn)
        self.reseg_n_btn = PushButton(text='Re-seg N (current image)')
        self.reseg_n_btn.changed.connect(self._on_reseg_n)
        self.reseg_p_btn = PushButton(text='Re-seg P (current image)')
        self.reseg_p_btn.changed.connect(self._on_reseg_p)
        self.save_btn = PushButton(text='Save masks')
        self.save_btn.changed.connect(self._on_save)

        # Finetune section
        self.ft_epochs = SpinBox(label='Fine-tune epochs', min=1, max=2000, value=100)
        self.ft_n_btn = PushButton(text='Fine-tune N (from edits)')
        self.ft_n_btn.changed.connect(lambda: self._on_finetune('n'))
        self.ft_p_btn = PushButton(text='Fine-tune P (from edits)')
        self.ft_p_btn.changed.connect(lambda: self._on_finetune('p'))
        self.ft_multi_n_btn = PushButton(text='Fine-tune N — multi-folder...')
        self.ft_multi_n_btn.changed.connect(lambda: self._on_finetune_multi('n'))
        self.ft_multi_p_btn = PushButton(text='Fine-tune P — multi-folder...')
        self.ft_multi_p_btn.changed.connect(lambda: self._on_finetune_multi('p'))

        _tt(self.sample_dir,
            'Sample folder — must contain intensity/*_sum.tif from PTU Reader.')
        _tt(self.tif_override,
            'Optional: override the auto-detected sum.tif with a specific '
            'file. Leave empty for the default.')
        _tt(self.n_model,
            'Cellpose model for nucleus (N) segmentation. Custom fine-'
            'tuned models under <sample>/_finetune/ or the shared plugin '
            'root are auto-discovered and ranked by recency.')
        _tt(self.p_model,
            'Cellpose model for cytoplasm (P) segmentation. Custom fine-'
            'tuned models auto-discovered like N model.')
        _tt(self.n_diameter,
            'Approximate nucleus diameter in pixels. Cellpose auto-'
            'detects if you set 0 (slower).')
        _tt(self.p_diameter,
            'Approximate cytoplasm diameter in pixels. Typically ~2× the '
            'nucleus diameter.')
        _tt(self.use_gpu,
            'Uses CUDA if available; falls back to CPU automatically. '
            'GPU is ~10× faster on 2k×2k images.')
        _tt(self.run_btn,
            'Runs N and P Cellpose models in sequence. Results auto-save '
            'to <image>_seg_n.npy / <image>_seg_p.npy next to the source.')
        _tt(self.reseg_n_btn,
            'Re-runs the current N model on the image (e.g. after you '
            'changed the model or diameter).')
        _tt(self.reseg_p_btn,
            'Re-runs the current P model on the image.')
        _tt(self.save_btn,
            'Re-saves the mask layers to disk AFTER your manual edits. '
            'Auto-Segment already saved the raw output, so this is only '
            'needed if you drew / deleted cells in the viewer.')
        _tt(self.ft_epochs,
            'Fine-tune epochs. 50–200 is typical for small mask '
            'corrections. More = more adaptation, but also overfitting '
            'risk on a single image.')
        _tt(self.ft_n_btn,
            'Fine-tune the selected N model on THIS image + edited mask. '
            'New model is saved under <sample>/_finetune/ and added to '
            'the dropdown.')
        _tt(self.ft_p_btn,
            'Fine-tune the selected P model on THIS image + edited mask.')
        _tt(self.ft_multi_n_btn,
            'Open a dialog to pick MULTIPLE sample folders and fine-tune '
            'N jointly on every (image, edited mask) pair found.')
        _tt(self.ft_multi_p_btn,
            'Multi-folder fine-tune for the P model (same UI as N).')

        self.progress = widgets.ProgressBar(label='Progress', value=0, min=0, max=100)
        self.status_label = Label(value='Ready')

        self.tips_label = Label(
            value=(
                '<b>Edit shortcuts</b> — select <code>mask_n_fill</code> / '
                '<code>mask_p_fill</code> first, then:<br>'
                '• <b>Right-click</b> to draw polygon, <b>Enter</b> to commit, '
                '<b>Esc</b> to cancel<br>'
                '• <b>Z / X</b> toggle N / P visibility, '
                '<b>Ctrl+click</b> delete label, <b>S</b> cycle contrast<br>'
                '• <b>Auto-save</b>: Auto Segment / Re-seg write '
                '<code>*_seg_n.npy</code> / <code>*_seg_p.npy</code> on disk. '
                'Click <b>Save masks</b> after manual edits.'
            ),
        )
        try:
            self.tips_label.native.setTextFormat(1)  # Qt.RichText
            self.tips_label.native.setWordWrap(True)
            self.tips_label.native.setStyleSheet(
                'QLabel {'
                '  background-color: #E3F2FD;'
                '  border: 1px solid #90CAF9;'
                '  border-radius: 4px;'
                '  padding: 6px 8px;'
                '  color: #1A237E;'
                '  font-family: Calibri;'
                '}'
            )
        except Exception:
            pass

        _append_section_divider(self, '— 📁 Input image —')
        self.append(self.sample_dir)
        self.append(self.tif_override)

        _append_section_divider(self, '— 🧠 Models & parameters —')
        self.append(self.n_model)
        self.append(self.p_model)
        self.append(self.n_diameter)
        self.append(self.p_diameter)
        self.append(self.use_gpu)

        _append_section_divider(self, '— ▶ Segment & edit —')
        self.append(self.run_btn)
        self.append(self.reseg_n_btn)
        self.append(self.reseg_p_btn)
        self.append(self.save_btn)

        _append_section_divider(self, '— 🎓 Fine-tune —')
        self.append(self.ft_epochs)
        self.append(self.ft_n_btn)
        self.append(self.ft_p_btn)
        self.append(self.ft_multi_n_btn)
        self.append(self.ft_multi_p_btn)

        self.append(self.progress)
        self.append(self.status_label)
        self.append(self.tips_label)

        _add_next_button(self, viewer)
        _add_cellpose_header(self, title='Barcode Segmentation (N & P)')
        _tighten_container(self)
        self._refresh_model_choices()
        try:
            self.sample_dir.changed.connect(self._refresh_model_choices)
        except Exception:
            pass
        self._bind_viewer_callbacks()

    # ----- choices / paths -----

    def _refresh_model_choices(self, *_args):
        """Rebuild the N / P dropdowns from all known custom-model locations.

        Scans the per-sample ``_finetune/`` folder, the shared
        ``_BARCODE_MODEL_ROOT`` finetune folders, and the cellpose cache.
        Every custom model appears in BOTH dropdowns; ordering puts
        target-matching models (N-starts / P-or-C-starts) first, then sorts
        by mtime so the most-recently-trained model is on top.

        Always wired to ``self.sample_dir.changed`` so switching sample
        folder auto-refreshes.
        """
        sd = self.sample_dir.value
        sample_dirs = [sd] if sd else []
        n_ranked = _list_all_custom_models(sample_dirs, target_hint='n')
        p_ranked = _list_all_custom_models(sample_dirs, target_hint='p')

        # Preserve the current selection if still available.
        cur_n = str(self.n_model.value) if self.n_model.value else _DEFAULT_N_MODEL
        cur_p = str(self.p_model.value) if self.p_model.value else _DEFAULT_P_MODEL

        def _with_defaults(ranked: list[str], default: str, builtins: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for n in [default] + ranked + builtins:
                if n and n not in seen:
                    out.append(n)
                    seen.add(n)
            return out

        self.n_model.choices = tuple(_with_defaults(n_ranked, _DEFAULT_N_MODEL, ['nuclei']))
        self.p_model.choices = tuple(_with_defaults(p_ranked, _DEFAULT_P_MODEL, ['cyto2']))
        if cur_n in self.n_model.choices:
            self.n_model.value = cur_n
        if cur_p in self.p_model.choices:
            self.p_model.value = cur_p

    def _intensity_sum_tif(self) -> Path | None:
        override = str(self.tif_override.value).strip()
        if override and Path(override).is_file():
            return Path(override)
        sd = Path(str(self.sample_dir.value))
        if not sd.is_dir():
            return None
        int_dir = sd / 'intensity'
        if not int_dir.is_dir():
            return None
        tifs = sorted(int_dir.glob('*_sum.tif'))
        return tifs[0] if tifs else None

    def _load_image_2d(self, path: Path) -> np.ndarray:
        img = tifffile.imread(str(path))
        if img.ndim > 2:
            img = np.squeeze(img)
        return np.asarray(img, dtype=np.float32)

    def _get_model(self, name: str):
        if name not in self._model_cache:
            self._model_cache[name] = _build_barcode_cellpose(
                model_type=name,
                use_gpu=self.use_gpu.value,
                extra_roots=[str(self.sample_dir.value)] if self.sample_dir.value else [],
            )
        return self._model_cache[name]

    # ----- actions -----

    def _on_run_auto(self):
        src = self._intensity_sum_tif()
        if src is None:
            show_warning("Could not find intensity/*_sum.tif — run PTU Reader first or set 'Override TIF'.")
            return
        self._current_src = src
        img = self._load_image_2d(src)
        self._current_img = img

        self.progress.min = 0
        self.progress.max = 100
        self.progress.value = 0
        self.status_label.value = 'Starting segmentation...'
        self.run_btn.enabled = False
        self.reseg_n_btn.enabled = False
        self.reseg_p_btn.enabled = False

        worker = self._seg_worker(
            img=img, src=src,
            n_model_name=self.n_model.value, n_diameter=float(self.n_diameter.value),
            p_model_name=self.p_model.value, p_diameter=float(self.p_diameter.value),
            use_gpu=bool(self.use_gpu.value),
            extra_roots=[str(self.sample_dir.value)] if self.sample_dir.value else [],
        )
        worker.yielded.connect(self._on_seg_yield)
        worker.returned.connect(self._on_seg_done)
        worker.errored.connect(self._on_seg_error)
        worker.start()

    @thread_worker
    def _seg_worker(self, img, src, n_model_name, n_diameter, p_model_name, p_diameter,
                    use_gpu, extra_roots):
        import time as _time
        stem = src.stem
        n_out = src.parent / f'{stem}_seg_n.npy'
        p_out = src.parent / f'{stem}_seg_p.npy'

        yield ('status', 10, f'Running N model ({n_model_name}) in subprocess...')
        t0 = _time.time()
        n_mask = _run_infer_subprocess(
            img=img, base_name=n_model_name,
            diameter=n_diameter, channels=[0, 0],
            use_gpu=use_gpu, extra_roots=extra_roots, out_path=n_out,
        )
        yield ('status', 50, f'N: {int(n_mask.max())} cells in {_time.time()-t0:.1f}s.')

        yield ('status', 55, f'Running P model ({p_model_name}) in subprocess...')
        t0 = _time.time()
        p_mask = _run_infer_subprocess(
            img=img, base_name=p_model_name,
            diameter=p_diameter, channels=[0, 0],
            use_gpu=use_gpu, extra_roots=extra_roots, out_path=p_out,
        )
        yield ('status', 95, f'P: {int(p_mask.max())} cells in {_time.time()-t0:.1f}s.')

        return (img, n_mask, p_mask)

    def _get_cached_model(self, name, use_gpu, extra_roots):
        """Thread-safe accessor that builds a cellpose model once per name."""
        if name not in self._model_cache:
            self._model_cache[name] = _build_barcode_cellpose(
                model_type=name, use_gpu=use_gpu, extra_roots=extra_roots,
            )
        return self._model_cache[name]

    def _on_seg_yield(self, payload):
        try:
            kind = payload[0]
        except Exception:
            return
        if kind == 'status':
            _, val, msg = payload
            try:
                self.progress.value = int(val)
            except Exception:
                pass
            self.status_label.value = msg
        elif kind == 'warn':
            show_warning(payload[1])

    def _on_seg_done(self, result):
        img, n_mask, p_mask = result
        self._setup_viewer_layers(img, n_mask, p_mask)
        self.progress.value = self.progress.max
        self.status_label.value = f'Done. N={int(n_mask.max())} cells, P={int(p_mask.max())} cells.'
        self.run_btn.enabled = True
        self.reseg_n_btn.enabled = True
        self.reseg_p_btn.enabled = True
        show_info('Segmentation done.')

    def _on_seg_error(self, exc):
        self.status_label.value = f'ERROR: {exc}'
        self.run_btn.enabled = True
        self.reseg_n_btn.enabled = True
        self.reseg_p_btn.enabled = True
        show_warning(f'Segmentation failed: {exc}')
        traceback.print_exc()

    def _setup_viewer_layers(self, img, n_mask, p_mask):
        # Remove + re-add instead of .data = — avoids vispy GL access violations
        # when the canvas is mid-draw.
        for _name in ('sum', 'mask_n_fill', 'mask_p_fill'):
            if _name in self.viewer.layers:
                try:
                    del self.viewer.layers[_name]
                except Exception:
                    pass
        self.viewer.add_image(img, name='sum', colormap='gray')
        ln = self.viewer.add_labels(n_mask.astype(np.int32), name='mask_n_fill')
        ln.mouse_drag_callbacks.append(self._ctrl_click_delete)
        lp = self.viewer.add_labels(p_mask.astype(np.int32), name='mask_p_fill')
        lp.mouse_drag_callbacks.append(self._ctrl_click_delete)
        if 'draw_points' not in self.viewer.layers:
            self.viewer.add_points(
                np.zeros((0, 2), dtype=float), name='draw_points',
                size=8, face_color='yellow', edge_color='black',
            )
        else:
            self.viewer.layers['draw_points'].data = np.zeros((0, 2), dtype=float)

    def _on_reseg_n(self):
        if self._current_img is None:
            show_warning('No image loaded. Click "Auto Segment" first.')
            return
        out_path = self._current_src.parent / f'{self._current_src.stem}_seg_n.npy'
        try:
            mask = _run_infer_subprocess(
                img=self._current_img, base_name=str(self.n_model.value),
                diameter=float(self.n_diameter.value), channels=[0, 0],
                use_gpu=bool(self.use_gpu.value),
                extra_roots=[str(self.sample_dir.value)] if self.sample_dir.value else [],
                out_path=out_path,
            )
        except Exception as e:
            show_warning(f'Re-seg N failed: {e}')
            return
        if 'mask_n_fill' in self.viewer.layers:
            try:
                del self.viewer.layers['mask_n_fill']
            except Exception:
                pass
        ln = self.viewer.add_labels(mask.astype(np.int32), name='mask_n_fill')
        ln.mouse_drag_callbacks.append(self._ctrl_click_delete)
        show_info(f'[N] re-seg: {int(mask.max())} cells')

    def _on_reseg_p(self):
        if self._current_img is None:
            show_warning('No image loaded. Click "Auto Segment" first.')
            return
        out_path = self._current_src.parent / f'{self._current_src.stem}_seg_p.npy'
        try:
            mask = _run_infer_subprocess(
                img=self._current_img, base_name=str(self.p_model.value),
                diameter=float(self.p_diameter.value), channels=[0, 0],
                use_gpu=bool(self.use_gpu.value),
                extra_roots=[str(self.sample_dir.value)] if self.sample_dir.value else [],
                out_path=out_path,
            )
        except Exception as e:
            show_warning(f'Re-seg P failed: {e}')
            return
        if 'mask_p_fill' in self.viewer.layers:
            try:
                del self.viewer.layers['mask_p_fill']
            except Exception:
                pass
        lp = self.viewer.add_labels(mask.astype(np.int32), name='mask_p_fill')
        lp.mouse_drag_callbacks.append(self._ctrl_click_delete)
        show_info(f'[P] re-seg: {int(mask.max())} cells')

    def _on_save(self):
        if self._current_src is None:
            show_warning('No source TIF loaded.')
            return
        stem = self._current_src.stem
        if 'mask_n_fill' in self.viewer.layers:
            arr = np.asarray(self.viewer.layers['mask_n_fill'].data).astype(np.uint16)
            np.save(str(self._current_src.parent / f'{stem}_seg_n.npy'), arr)
        if 'mask_p_fill' in self.viewer.layers:
            arr = np.asarray(self.viewer.layers['mask_p_fill'].data).astype(np.uint16)
            np.save(str(self._current_src.parent / f'{stem}_seg_p.npy'), arr)
        show_info('Saved N and P masks.')

    def _on_finetune(self, target: str):
        """Fine-tune the selected N or P model on the currently edited mask (single-image training)."""
        if self._current_img is None:
            show_warning('Segment first so there is an image to fine-tune on.')
            return
        layer_name = 'mask_n_fill' if target == 'n' else 'mask_p_fill'
        if layer_name not in self.viewer.layers:
            show_warning(f'{layer_name} not found — run Auto Segment first.')
            return
        mask = np.asarray(self.viewer.layers[layer_name].data).astype(np.int32)
        if int(mask.max()) == 0:
            show_warning(f'{layer_name} is empty — draw / keep some labels before fine-tuning.')
            return

        base_name = self.n_model.value if target == 'n' else self.p_model.value
        ts = datetime.now().strftime('%m%d-%H%M')
        new_name = f'{base_name}-ft{ts}'
        sample_dir = Path(str(self.sample_dir.value))
        save_dir = sample_dir / '_finetune' / new_name
        save_dir.mkdir(parents=True, exist_ok=True)

        n_epochs = int(self.ft_epochs.value)
        self.progress.min = 0
        self.progress.max = 100
        self.progress.value = 0
        self.status_label.value = f'Fine-tuning {target.upper()} from {base_name} ({n_epochs} epochs)...'
        self.run_btn.enabled = False
        self.reseg_n_btn.enabled = False
        self.reseg_p_btn.enabled = False
        self.ft_n_btn.enabled = False
        self.ft_p_btn.enabled = False

        worker = self._finetune_worker(
            target=target, base_name=base_name, new_name=new_name,
            save_dir=save_dir, img=self._current_img, mask=mask,
            n_epochs=n_epochs, use_gpu=bool(self.use_gpu.value),
            extra_roots=[str(sample_dir)],
        )
        worker.yielded.connect(self._on_seg_yield)
        worker.returned.connect(self._on_finetune_done)
        worker.errored.connect(self._on_finetune_error)
        worker.start()

    @thread_worker
    def _finetune_worker(self, target, base_name, new_name, save_dir, img, mask,
                         n_epochs, use_gpu, extra_roots):
        import time as _time
        yield ('status', 10, f'Spawning fine-tune subprocess ({base_name}, '
                              f'{n_epochs} epochs on 1 image)... '
                              f'Running in a child Python so GL/CUDA state '
                              f'cannot leak back into napari.')
        t0 = _time.time()
        new_path = _run_finetune_subprocess(
            img=img, mask=mask,
            base_name=base_name, new_name=new_name,
            save_dir=save_dir, n_epochs=n_epochs,
            channels=[0, 0], use_gpu=use_gpu, extra_roots=extra_roots,
        )
        yield ('status', 95, f'Training done in {_time.time()-t0:.1f}s. Saving...')
        return (target, new_name, new_path)

    def _on_finetune_done(self, result):
        target, new_name, new_path = result
        # Invalidate cache so re-seg picks up new weights if the name collides
        self._model_cache.pop(new_name, None)
        # Full rescan so the new model appears in BOTH dropdowns (with correct
        # mtime-based ordering), not just the target dropdown.
        self._refresh_model_choices()
        # Select it on the target dropdown.
        combo = self.n_model if target == 'n' else self.p_model
        if new_name in combo.choices:
            combo.value = new_name
        self.progress.value = self.progress.max
        self.status_label.value = f'Fine-tuned: {new_name}. Selected.'
        self._reenable_buttons()
        show_info(f'Fine-tuned model saved: {new_path}')

    def _on_finetune_error(self, exc):
        self.status_label.value = f'Fine-tune ERROR: {exc}'
        self._reenable_buttons()
        show_warning(f'Fine-tune failed: {exc}')
        traceback.print_exc()

    def _on_finetune_multi(self, target: str):
        """Open the multi-folder fine-tune dialog for target 'n' or 'p'."""
        base_name = str(self.n_model.value if target == 'n' else self.p_model.value)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        default_new = f'{"N" if target == "n" else "P"}-multi-{ts}'
        sd = self.sample_dir.value
        save_root = (Path(str(sd)) / '_finetune') if sd else _BARCODE_MODEL_ROOT / '_cellpose_finetune_multi'

        def _done(new_name, new_path, n_samples):
            self._model_cache.pop(new_name, None)
            self._refresh_model_choices()
            combo = self.n_model if target == 'n' else self.p_model
            if new_name in combo.choices:
                combo.value = new_name
            self.status_label.value = (
                f'Multi-folder fine-tune done: {new_name} '
                f'(trained on {n_samples} samples). Saved to {new_path}.'
            )
            show_info(f'Multi-folder fine-tune done: {new_name} ({n_samples} samples).')

        def _err(exc):
            self.status_label.value = f'Multi-folder fine-tune ERROR: {exc}'
            show_warning(f'Multi-folder fine-tune failed: {exc}')

        _open_multi_finetune_dialog(
            parent_widget=self,
            target=target,
            base_name=base_name,
            default_new_name=default_new,
            default_epochs=int(self.ft_epochs.value),
            use_gpu=bool(self.use_gpu.value),
            save_root=save_root,
            on_done=_done,
            on_error=_err,
        )

    def _reenable_buttons(self):
        self.run_btn.enabled = True
        self.reseg_n_btn.enabled = True
        self.reseg_p_btn.enabled = True
        self.ft_n_btn.enabled = True
        self.ft_p_btn.enabled = True

    # ----- viewer callbacks -----

    def _bind_viewer_callbacks(self):
        try:
            @self.viewer.bind_key('Z', overwrite=True)
            def _toggle_n(_v):
                if 'mask_n_fill' in self.viewer.layers:
                    layer = self.viewer.layers['mask_n_fill']
                    layer.visible = not layer.visible
                    self.viewer.layers.selection.active = layer

            @self.viewer.bind_key('X', overwrite=True)
            def _toggle_p(_v):
                if 'mask_p_fill' in self.viewer.layers:
                    layer = self.viewer.layers['mask_p_fill']
                    layer.visible = not layer.visible
                    self.viewer.layers.selection.active = layer

            @self.viewer.bind_key('Escape', overwrite=True)
            def _cancel_poly(_v):
                self._draw_state = {"layer": "", "pts": [], "active": False}
                if 'draw_points' in self.viewer.layers:
                    self.viewer.layers['draw_points'].data = np.zeros((0, 2), dtype=float)

            @self.viewer.bind_key('Enter', overwrite=True)
            def _commit_poly_key(_v):
                self._commit_polygon()

            @self.viewer.bind_key('S', overwrite=True)
            def _cycle_contrast(_v):
                if 'sum' not in self.viewer.layers:
                    return
                layer = self.viewer.layers['sum']
                data = np.asarray(layer.data)
                if data.size == 0:
                    return
                vmin = float(np.nanmin(data))
                vmax = float(np.nanmax(data))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    return
                self._contrast_idx = (self._contrast_idx + 1) % len(self._contrast_ratios)
                ratio = self._contrast_ratios[self._contrast_idx]
                layer.contrast_limits = (vmin, vmin + (vmax - vmin) * ratio)
        except Exception as e:
            print(f'[BarcodeSeg] key binding error: {e}')

        if self._right_click_add not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.append(self._right_click_add)
        if self._mouse_track not in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.append(self._mouse_track)

    def _selected_mask_layer(self) -> str:
        try:
            active = self.viewer.layers.selection.active
            if active is not None and active.name in ('mask_n_fill', 'mask_p_fill'):
                return active.name
        except Exception:
            pass
        return ''

    def _right_click_add(self, _viewer, event):
        btn = getattr(event, 'button', None)
        is_right = (btn == 2) or (str(btn).lower() == 'right')
        if not is_right:
            return
        layer_name = self._selected_mask_layer()
        if layer_name not in ('mask_n_fill', 'mask_p_fill'):
            return
        layer = self.viewer.layers[layer_name]
        try:
            yx = layer.world_to_data(event.position)[:2]
        except Exception:
            yx = event.position[:2]
        yx = (float(yx[0]), float(yx[1]))
        if not self._draw_state['active'] or self._draw_state['layer'] != layer_name:
            self._draw_state = {'active': True, 'layer': layer_name, 'pts': [yx]}
            if 'draw_points' in self.viewer.layers:
                self.viewer.layers['draw_points'].data = np.array([yx], dtype=float)
            return
        self._commit_polygon()

    def _mouse_track(self, _viewer, event):
        if not self._draw_state['active']:
            return
        layer_name = self._draw_state['layer']
        if layer_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[layer_name]
        try:
            yx = layer.world_to_data(event.position)[:2]
        except Exception:
            yx = event.position[:2]
        yx = (float(yx[0]), float(yx[1]))
        pts = self._draw_state['pts']
        if len(pts) == 0:
            pts.append(yx)
        else:
            y_last, x_last = pts[-1]
            d_last = ((yx[0] - y_last) ** 2 + (yx[1] - x_last) ** 2) ** 0.5
            if d_last >= 3.0:
                pts.append(yx)
        self._draw_state['pts'] = pts
        if 'draw_points' in self.viewer.layers:
            self.viewer.layers['draw_points'].data = np.array(pts, dtype=float)
        if len(pts) >= 8:
            y0, x0 = pts[0]
            d0 = ((yx[0] - y0) ** 2 + (yx[1] - x0) ** 2) ** 0.5
            if d0 <= 18:
                self._commit_polygon()

    def _commit_polygon(self):
        layer_name = self._draw_state['layer'] or self._selected_mask_layer()
        if layer_name not in ('mask_n_fill', 'mask_p_fill'):
            return
        pts = self._draw_state['pts']
        if len(pts) < 3:
            return
        layer = self.viewer.layers[layer_name]
        mask = np.asarray(layer.data)
        used = set(np.unique(mask).astype(int).tolist())
        used.discard(0)
        new_id = 1
        while new_id in used:
            new_id += 1
        # cv2.fillPoly is O(polygon_area) vs MplPath.contains_points O(H*W).
        # On 2048x2048 this is ~20x faster, which removes the draw lag.
        new_mask = mask.copy()
        poly_xy = np.array([[int(round(x)), int(round(y))] for y, x in pts], dtype=np.int32)
        cv2.fillPoly(new_mask, [poly_xy], int(new_id))
        layer.data = new_mask
        self._draw_state = {'active': False, 'layer': layer_name, 'pts': []}
        if 'draw_points' in self.viewer.layers:
            self.viewer.layers['draw_points'].data = np.zeros((0, 2), dtype=float)

    def _ctrl_click_delete(self, layer, event):
        if 'Control' not in event.modifiers:
            return
        v = layer.get_value(event.position, world=True)
        if v is None:
            return
        try:
            v = int(v)
        except Exception:
            return
        if v <= 0:
            return
        dat = np.asarray(layer.data)
        dat[dat == v] = 0
        layer.data = dat


# ---- BiosensorSeg configuration ----
_BIOSENSOR_MODEL_DEFAULT = "BS-BC-assist-cls-260402-forDense"
_BIOSENSOR_MODEL_FALLBACKS = [
    "BS-BC-assist-cls-260402-forDense",
    "BS-BC-assist-cls-260328-1",
    "BS-BC-assist-cls-260328",
    "cyto2",
    "cyto",
]
_BIOSENSOR_DEFAULT_DIAM = 45.0
_BARCODE_ASSIST_ROTATE_K = 3  # 90° * k clockwise; matches BS-BC-assist training


def _norm01_percentile(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    p1 = float(np.nanpercentile(a, 1))
    p99 = float(np.nanpercentile(a, 99))
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32)


def _resize_labels_nearest(lbl: np.ndarray, target_hw) -> np.ndarray:
    h, w = target_hw
    if lbl.shape[:2] == (h, w):
        return lbl
    in_h, in_w = lbl.shape[:2]
    if min(in_h, in_w, h, w) <= 0:
        return np.zeros((h, w), dtype=lbl.dtype)
    ys = (np.arange(h) * in_h // h).astype(np.int64)
    xs = (np.arange(w) * in_w // w).astype(np.int64)
    return np.asarray(lbl[ys[:, None], xs[None, :]], dtype=lbl.dtype)


def _prep_biosensor_seg_input(img2d: np.ndarray, barcode_cls_path: "Path | None",
                              use_assist: bool, rotate_k: int = 0):
    """Build the Cellpose input. Returns (img_in, channels) tuple.

    If use_assist and barcode_cls is available, stacks a 2-channel HxWx2 image
    (seg image, barcode assist) and returns channels=[1, 2]. Otherwise returns
    the single-channel img2d and channels=[0, 0].

    `rotate_k` rotates the barcode cls image by k*90° counter-clockwise (np.rot90
    convention). Leica tilescan exports often need k=3 (equiv. 270° CCW / 90° CW)
    to align with the confocal orientation; single-FOV acquisitions usually need k=0.
    """
    if use_assist and barcode_cls_path and Path(barcode_cls_path).exists():
        aux = np.asarray(tifffile.imread(str(barcode_cls_path)), dtype=np.float32)
        if aux.ndim > 2:
            aux = np.squeeze(aux)
        if int(rotate_k) % 4:
            aux = np.rot90(aux, k=int(rotate_k))
        if aux.shape != img2d.shape:
            aux = _resize_labels_nearest(aux, img2d.shape).astype(np.float32, copy=False)
        main01 = _norm01_percentile(img2d)
        aux01 = _norm01_percentile(aux)
        img_in = np.stack([main01, aux01], axis=-1).astype(np.float32, copy=False)
        return img_in, [1, 2]
    return img2d, [0, 0]


class BiosensorSeg(Container):
    """Two-step biosensor cell segmentation with barcode-assist Cellpose model.

    Step 1 (seg image): average the first N frames of the selected biosensor
           channels (B/G/Y) into one strong-signal 2D image. This single image
           is used as the per-cell mask for the full stack (no per-frame
           tracking required when cells don't move much).
    Step 2 (segmentation): run Cellpose (BS-BC-assist-cls-*) on that image
           using the barcode cls.tif as the auxiliary channel.
    Optional: single-image fine-tune of the dual-channel model.
    """
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(layout='vertical')
        self.viewer = viewer
        self._seg_img: np.ndarray | None = None
        self._seg_img_save_path: Path | None = None
        self._last_mask: np.ndarray | None = None
        self._last_mask_save_path: Path | None = None
        self._model_cache: dict = {}
        self._draw_state = {'layer': '', 'pts': [], 'active': False}
        self._contrast_ratios = [1.0, 0.5, 0.25, 0.12, 0.06]
        self._contrast_idx = 0

        # --- Step 1: seg image (the image we segment on) ---
        self.sample_folder = FileEdit(
            label='Sample Folder', mode='d',
            value=r'J:/Mix16-N-P-260306-DCZ-2-1',
        )
        self.stack_b_path = FileEdit(label='Stack B', mode='r', filter='*.tif')
        self.stack_g_path = FileEdit(label='Stack G', mode='r', filter='*.tif')
        self.stack_y_path = FileEdit(label='Stack Y', mode='r', filter='*.tif')
        self.use_b = CheckBox(text='Use B', value=True)
        self.use_g = CheckBox(text='Use G', value=True)
        self.use_y = CheckBox(text='Use Y', value=True)
        self.frame_start = SpinBox(label='First frame (1-based)', min=1, max=10000, value=1)
        self.frame_end = SpinBox(label='Last frame (inclusive)', min=1, max=10000, value=6)
        self.gen_btn = PushButton(text='Generate Seg Image')
        self.gen_btn.changed.connect(self._on_generate)
        _style_process_button(self.gen_btn)

        # --- Step 2: segmentation ---
        self.barcode_cls_path = FileEdit(
            label='Barcode cls.tif (assist)', mode='r', filter='*.tif',
        )
        self.seg_model = ComboBox(
            label='Cellpose model', choices=list(_BIOSENSOR_MODEL_FALLBACKS),
            value=_BIOSENSOR_MODEL_DEFAULT,
        )
        self.diameter = FloatSpinBox(
            label='Diameter (px)', min=5, max=500, step=1, value=_BIOSENSOR_DEFAULT_DIAM,
        )
        # Barcode cls rotation. Leica tilescan barcodes are rotated 90° CW
        # relative to the confocal biosensor stack — that's why the default is
        # 90° CW. For non-tilescan single-FOV acquisitions, pick 0°.
        # Internally we translate to np.rot90 k (which is CCW):
        #   0° -> k=0, 90° CW -> k=3, 180° -> k=2, 270° CW -> k=1
        self.barcode_rot_choice = ComboBox(
            label='Barcode rotation',
            choices=['0°', '90° CW', '180°', '270° CW'],
            value='90° CW',
        )
        self.barcode_resize = SpinBox(
            label='Barcode resize (px, 0=match seg image)',
            min=0, max=8192, value=1024,
        )
        self.confirm_barcode_btn = PushButton(text='Load / Confirm Barcode')
        self.confirm_barcode_btn.changed.connect(self._on_confirm_barcode)
        self.use_gpu = CheckBox(text='Use GPU', value=True)
        self.seg_btn = PushButton(text='Segment Cells')
        self.seg_btn.changed.connect(self._on_segment)
        _style_process_button(self.seg_btn)
        self.reseg_btn = PushButton(text='Re-seg (current image)')
        self.reseg_btn.changed.connect(self._on_segment)  # same path, reuses cached seg image
        self.save_btn = PushButton(text='Save mask')
        self.save_btn.changed.connect(self._on_save)

        # --- Fine-tune ---
        self.ft_epochs = SpinBox(label='Fine-tune epochs', min=1, max=2000, value=100)
        self.ft_btn = PushButton(text='Fine-tune (current image + mask)')
        self.ft_btn.changed.connect(self._on_finetune)
        self.ft_multi_btn = PushButton(text='Fine-tune — multi-folder...')
        self.ft_multi_btn.changed.connect(self._on_finetune_multi)

        # --- Progress + tips ---
        self.progress = widgets.ProgressBar(label='Progress', value=0, min=0, max=100)
        self.status_label = Label(value='Ready. Start with Step 1 -> Generate Seg Image.')
        self.tips_label = Label(
            value=(
                '<b>Step 1 — Seg image</b>: pick channels + frame range → '
                '<b>Generate Seg Image</b>. Inspect the new layer; tweak params '
                'and click again to overwrite. This single image is the mask '
                'for the whole stack (no per-frame tracking required).<br>'
                '<b>Step 2 — Barcode</b>: set <i>Rotation</i> (90° CW for Leica '
                'tilescan, 0° for single-FOV) and <i>Resize</i> (default 1024, '
                '0 = match seg image shape), then <b>Load / Confirm Barcode</b>. '
                'Toggle the new <code>barcode_cls</code> layer on/off to confirm '
                'registration. Segment uses exactly what you see.<br>'
                '<b>Step 3 — Segment & edit</b>: right-click polygon, Enter '
                'commit, Esc cancel, <b>Z</b> toggle mask_biosensor, <b>X</b> '
                'toggle barcode_cls, Ctrl+click delete, S cycle contrast. '
                'Auto-save to '
                '<code>&lt;sample&gt;/&lt;seg_image_stem&gt;_seg.npy</code>.'
            ),
        )
        try:
            self.tips_label.native.setTextFormat(1)  # Qt.RichText
            self.tips_label.native.setWordWrap(True)
            self.tips_label.native.setStyleSheet(
                'QLabel {'
                '  background-color: #E3F2FD;'
                '  border: 1px solid #90CAF9;'
                '  border-radius: 4px;'
                '  padding: 6px 8px;'
                '  color: #1A237E;'
                '  font-family: Calibri;'
                '}'
            )
        except Exception:
            pass

        # Tooltips for every control. Hover reveals what each one does.
        _tt(self.sample_folder,
            'Sample folder. Biosensor Seg reads the confocal stacks, the '
            'barcode classification image, and writes seg_image.tif + '
            'seg_image_seg.npy here.')
        _tt(self.stack_b_path, 'Blue-channel confocal stack (.tif). Time-lapse.')
        _tt(self.stack_g_path, 'Green-channel confocal stack (.tif). Time-lapse.')
        _tt(self.stack_y_path,
            'NIR / yellow-channel confocal stack (.tif). Optional — untick '
            'Use Y if this channel is missing.')
        _tt(self.use_b, 'Include B channel when building the seg image.')
        _tt(self.use_g, 'Include G channel when building the seg image.')
        _tt(self.use_y, 'Include Y channel when building the seg image.')
        _tt(self.frame_start,
            '1-based first frame to sum into the seg image. Early frames '
            '(1–6) are usually the sharpest.')
        _tt(self.frame_end,
            '1-based last frame (inclusive) to sum into the seg image.')
        _tt(self.gen_btn,
            'Sums the checked channels across [first, last] frames into '
            'seg_image.tif and loads it as a napari layer.')
        _tt(self.barcode_cls_path,
            'Barcode classification TIF (one class label per cell) from '
            'the Seeded K-Means step. Auto-filled when the sample folder '
            'is set; override to use a different one.')
        _tt(self.barcode_rot_choice,
            'Leica tilescan barcodes are rotated 90° CW relative to the '
            'confocal biosensor stack — hence the default. Pick 0° for a '
            'single-FOV acquisition.')
        _tt(self.barcode_resize,
            'Resize the barcode mask to this edge length (pixels) before '
            'overlaying. 0 = match the seg image size exactly.')
        _tt(self.confirm_barcode_btn,
            'Loads the barcode classification, rotates + resizes to match '
            'the seg image, and previews it as an overlay layer.')
        _tt(self.seg_model,
            'Dual-input Cellpose model that takes the biosensor seg '
            'image + aligned barcode as input. BS-* / *biosensor* names '
            'sort to the top.')
        _tt(self.diameter,
            'Approximate cell diameter in pixels. Set 0 for Cellpose '
            'auto-detect (slower).')
        _tt(self.use_gpu,
            'Uses CUDA if available; falls back to CPU automatically.')
        _tt(self.seg_btn,
            'Runs the selected Cellpose model in a subprocess and saves '
            'the mask to seg_image_seg.npy next to the seg image.')
        _tt(self.reseg_btn,
            'Re-runs segmentation on the CURRENT seg image (useful after '
            'changing model or diameter).')
        _tt(self.save_btn,
            'Re-save mask_biosensor to disk AFTER your manual edits. '
            'Segment already saved the raw output.')
        _tt(self.ft_epochs,
            'Fine-tune epochs. 50–200 is typical for small corrections.')
        _tt(self.ft_btn,
            'Fine-tune the selected model on THIS seg image + edited '
            'mask. New model is saved under <sample>/_finetune/.')
        _tt(self.ft_multi_btn,
            'Open a dialog to pick MULTIPLE sample folders and fine-tune '
            'jointly on every (seg_image, seg_image_seg.npy) pair found.')

        _append_section_divider(self, '— 📁 Sample folder —')
        self.append(self.sample_folder)

        _append_section_divider(self, '— 🧪 Step 1: Seg image —')
        self.append(self.stack_b_path)
        self.append(self.stack_g_path)
        self.append(self.stack_y_path)
        self.append(self.use_b)
        self.append(self.use_g)
        self.append(self.use_y)
        self.append(self.frame_start)
        self.append(self.frame_end)
        self.append(self.gen_btn)

        _append_section_divider(self, '— 🔖 Step 2: Barcode assist —')
        self.append(self.barcode_cls_path)
        self.append(self.barcode_rot_choice)
        self.append(self.barcode_resize)
        self.append(self.confirm_barcode_btn)

        _append_section_divider(self, '— 🧠 Step 3: Segment & edit —')
        self.append(self.seg_model)
        self.append(self.diameter)
        self.append(self.use_gpu)
        self.append(self.seg_btn)
        self.append(self.reseg_btn)
        self.append(self.save_btn)

        _append_section_divider(self, '— 🎓 Fine-tune —')
        self.append(self.ft_epochs)
        self.append(self.ft_btn)
        self.append(self.ft_multi_btn)

        self.append(self.progress)
        self.append(self.status_label)
        self.append(self.tips_label)

        _add_next_button(self, viewer)
        _add_cellpose_header(self, title='Biosensor Segmentation')
        _tighten_container(self)
        # Cap the widget width — long FOV-*/_cls.tif paths were letting napari
        # size the whole dock super wide. Individual FileEdits can still show
        # the tail of a long path; hovering reveals the full path as a tooltip.
        try:
            self.native.setMaximumWidth(720)
            for _fe in (self.sample_folder, self.stack_b_path, self.stack_g_path,
                         self.stack_y_path, self.barcode_cls_path):
                _fe.native.setMaximumWidth(600)
        except Exception:
            pass
        self.sample_folder.changed.connect(self._auto_fill_paths)
        self.sample_folder.changed.connect(self._refresh_seg_model_choices)
        self._auto_fill_paths()
        self._refresh_seg_model_choices()
        self._bind_viewer_callbacks()

    def _refresh_seg_model_choices(self, *_args):
        """Rebuild the Cellpose-model dropdown with all known custom models
        from this sample's ``_finetune/``, the shared model root, and the
        cellpose cache. Biosensor-trained models (BS-* or names containing
        'biosensor') sort to the top.
        """
        sd = self.sample_folder.value
        sample_dirs = [sd] if sd else []
        ranked = _list_all_custom_models(sample_dirs, target_hint='bs')
        cur = str(self.seg_model.value) if self.seg_model.value else _BIOSENSOR_MODEL_DEFAULT
        out: list[str] = []
        seen: set[str] = set()
        for n in [_BIOSENSOR_MODEL_DEFAULT] + ranked + list(_BIOSENSOR_MODEL_FALLBACKS):
            if n and n not in seen:
                out.append(n)
                seen.add(n)
        self.seg_model.choices = tuple(out)
        if cur in self.seg_model.choices:
            self.seg_model.value = cur

    # ---------- Path discovery ----------

    def _rotation_k(self) -> int:
        """Translate the Barcode rotation dropdown into an np.rot90 k value."""
        mapping = {'0°': 0, '90° CW': 3, '180°': 2, '270° CW': 1}
        return int(mapping.get(str(self.barcode_rot_choice.value), 0))

    def _build_seg_input(self, img2d: np.ndarray, use_assist: bool):
        """Build Cellpose input (img_in, channels). Prefers the already-confirmed
        `barcode_cls` napari layer so the user's visually-verified alignment is
        what gets fed to Cellpose. Falls back to re-reading + rotating from path
        if the layer is missing."""
        if use_assist and 'barcode_cls' in self.viewer.layers:
            aux = np.asarray(self.viewer.layers['barcode_cls'].data, dtype=np.float32)
            if aux.shape != img2d.shape:
                aux = _resize_labels_nearest(
                    aux.astype(np.int32), img2d.shape,
                ).astype(np.float32)
            main01 = _norm01_percentile(img2d)
            aux01 = _norm01_percentile(aux)
            img_in = np.stack([main01, aux01], axis=-1).astype(np.float32, copy=False)
            return img_in, [1, 2]
        # Fallback: old behaviour (read + rotate + resize from path)
        aux_path = Path(str(self.barcode_cls_path.value)) if self.barcode_cls_path.value else None
        return _prep_biosensor_seg_input(
            img2d, aux_path, use_assist, rotate_k=self._rotation_k(),
        )

    def _on_confirm_barcode(self):
        """Load barcode cls.tif, apply rotation + resize, show as napari layer.

        Segment reuses this aligned layer directly — so the user can visually
        confirm registration against the seg image BEFORE running Cellpose.
        """
        p = Path(str(self.barcode_cls_path.value)) if self.barcode_cls_path.value else None
        if p is None or not p.is_file():
            show_warning(f'Barcode cls.tif path invalid: {p}')
            return

        try:
            aux = np.asarray(tifffile.imread(str(p)), dtype=np.float32)
        except Exception as e:
            show_warning(f'Failed to read {p.name}: {e}')
            return
        if aux.ndim > 2:
            aux = np.squeeze(aux)
        if aux.ndim != 2:
            show_warning(f'Barcode cls must be 2D, got shape {aux.shape}.')
            return

        # rotate
        k = self._rotation_k()
        if k % 4:
            aux = np.rot90(aux, k=k)

        # resize target: if 0, match the seg image (if loaded); else user value
        target_size = int(self.barcode_resize.value)
        target_hw = None
        if target_size <= 0:
            seg = self._get_seg_img()
            if seg is None:
                show_warning('Resize=0 needs the seg image loaded (Step 1) to match its shape.')
                return
            target_hw = seg.shape[:2]
        else:
            target_hw = (target_size, target_size)
        if aux.shape != target_hw:
            aux = _resize_labels_nearest(aux.astype(np.int32), target_hw).astype(np.float32)

        name = 'barcode_cls'
        # Remove + re-add instead of reassigning .data — avoids vispy GL
        # access violations when the canvas is mid-draw.
        if name in self.viewer.layers:
            try:
                del self.viewer.layers[name]
            except Exception:
                pass
        self.viewer.add_labels(aux.astype(np.int32), name=name, opacity=0.45)
        show_info(
            f'Barcode loaded: rot={self.barcode_rot_choice.value}, '
            f'shape={aux.shape}. Toggle visibility and compare with seg_image.'
        )

    def _auto_fill_paths(self):
        """Pick default stack / barcode paths by scanning the sample folder."""
        base = str(self.sample_folder.value) if self.sample_folder.value else ''
        if not base or not os.path.isdir(base):
            return
        base_p = Path(base)
        for ch, widget in (('b', self.stack_b_path), ('g', self.stack_g_path),
                           ('y', self.stack_y_path)):
            hits = sorted(base_p.glob(f'FOV-*-{ch}.tif'))
            if hits:
                try:
                    widget.value = str(hits[0])
                except Exception:
                    pass
        int_dir = base_p / 'intensity'
        if int_dir.is_dir():
            cls_hits = [p for p in sorted(int_dir.glob('*-cls.tif'))
                        if 'color' not in p.name.lower() and 'text' not in p.name.lower()]
            if cls_hits:
                try:
                    self.barcode_cls_path.value = str(cls_hits[0])
                except Exception:
                    pass

    # ---------- Progress helpers ----------

    def _set_progress(self, pct: int, msg: str):
        try:
            self.progress.value = int(pct)
            self.status_label.value = msg
        except Exception:
            pass
        try:
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

    # ---------- Step 1: seg image ----------

    def _on_generate(self):
        start = int(self.frame_start.value)  # 1-based inclusive
        end = int(self.frame_end.value)
        if end < start:
            show_warning(f'Last frame ({end}) < First frame ({start}).')
            return

        stacks = []
        for use, path, tag in (
            (self.use_b.value, self.stack_b_path.value, 'B'),
            (self.use_g.value, self.stack_g_path.value, 'G'),
            (self.use_y.value, self.stack_y_path.value, 'Y'),
        ):
            if not use:
                continue
            p = Path(str(path)) if path else None
            if p is None or not p.is_file():
                show_warning(f'Stack {tag} path invalid: {p}')
                return
            stacks.append((tag, p))
        if not stacks:
            show_warning('Pick at least one channel (B/G/Y) for the seg image.')
            return

        self.gen_btn.enabled = False
        self._set_progress(5, f'Reading {len(stacks)} stack(s)...')
        try:
            accum = None
            n_channels = len(stacks)
            for ci, (tag, p) in enumerate(stacks):
                self._set_progress(
                    5 + int(80 * ci / max(1, n_channels)),
                    f'Loading {tag}: {p.name}...',
                )
                arr = tifffile.imread(str(p))
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                T = arr.shape[0]
                s0 = max(0, start - 1)
                e0 = min(T, end)
                if e0 <= s0:
                    show_warning(f'{tag}: frame range {start}..{end} outside stack with {T} frames.')
                    return
                sub = arr[s0:e0].astype(np.float32)  # (F, H, W)
                frame_mean = sub.mean(axis=0)  # (H, W) — safe from overflow
                if accum is None:
                    accum = frame_mean
                else:
                    accum = accum + frame_mean

            accum = accum / float(n_channels)  # avg across selected channels
            track_u16 = np.clip(np.round(accum), 0, 65535).astype(np.uint16)

            # persist
            sample_dir = Path(str(self.sample_folder.value))
            sample_dir.mkdir(parents=True, exist_ok=True)
            tags = [t for t, _ in stacks]
            stem = f"FOV_{''.join(tags).lower()}_frames{start}-{end}_seg_img"
            out_path = sample_dir / f'{stem}.tif'
            self._set_progress(90, f'Saving {out_path.name}...')
            tifffile.imwrite(str(out_path), track_u16, imagej=True)

            self._seg_img = track_u16
            self._seg_img_save_path = out_path

            # napari layer (single name so re-runs overwrite). We REMOVE the old
            # layer first and add a fresh one instead of reassigning `.data`,
            # because the latter has triggered vispy/GL access violations when
            # the canvas is mid-draw.
            if 'seg_image' in self.viewer.layers:
                try:
                    del self.viewer.layers['seg_image']
                except Exception:
                    pass
            self.viewer.add_image(track_u16, name='seg_image', colormap='gray')
            self._set_progress(
                100,
                f'Tracking image ready: {out_path.name} '
                f'(channels={"+".join(tags)}, frames {start}-{end}).'
            )
            show_info(f'Tracking image saved: {out_path}')
        except Exception as e:
            self._set_progress(0, f'ERROR: {e}')
            show_warning(f'Generate tracking image failed: {e}')
            traceback.print_exc()
        finally:
            self.gen_btn.enabled = True

    # ---------- Step 2: segmentation ----------

    def _get_seg_img(self) -> np.ndarray | None:
        if self._seg_img is not None:
            return self._seg_img
        # fallback: tracking_image layer in viewer
        if 'seg_image' in self.viewer.layers:
            return np.asarray(self.viewer.layers['seg_image'].data)
        return None

    def _get_cached_model(self, name: str, use_gpu: bool):
        if name not in self._model_cache:
            self._model_cache[name] = _build_barcode_cellpose(
                model_type=name, use_gpu=use_gpu,
            )
        return self._model_cache[name]

    def _on_segment(self):
        img = self._get_seg_img()
        if img is None:
            show_warning('No seg image. Run Step 1 first.')
            return
        img2d = np.asarray(img, dtype=np.float32)
        if img2d.ndim > 2:
            img2d = np.squeeze(img2d)
        if img2d.ndim != 2:
            show_warning(f'Seg image must be 2D, got shape {img2d.shape}.')
            return

        model_name = str(self.seg_model.value)
        diameter = float(self.diameter.value)
        use_assist = model_name.lower().startswith('bs-bc-assist')

        self.seg_btn.enabled = False
        self.reseg_btn.enabled = False
        try:
            import time as _time
            self._set_progress(15, f'Preparing input for {model_name}...')
            img_in, channels = self._build_seg_input(img2d, use_assist)
            if use_assist and channels == [0, 0]:
                show_warning('BS-BC-assist model selected but no barcode layer loaded. '
                             'Click "Load / Confirm Barcode" first, or running '
                             'single-channel fallback.')
            # Decide where to save the mask first (subprocess writes it for us).
            seg_ref_path = self._seg_img_save_path
            if seg_ref_path is None:
                seg_ref_path = Path(str(self.sample_folder.value)) / 'seg_image.tif'
            save_path = seg_ref_path.parent / (seg_ref_path.stem + '_seg.npy')

            self._set_progress(
                40,
                f'Running {model_name} in a child process (keeps PyTorch / '
                f'CUDA state out of napari)...'
            )
            t0 = _time.time()
            masks = _run_infer_subprocess(
                img=img_in, base_name=model_name,
                diameter=diameter, channels=channels,
                use_gpu=bool(self.use_gpu.value),
                extra_roots=[str(self.sample_folder.value)] if self.sample_folder.value else [],
                out_path=save_path,
            )
            self._set_progress(85, f'Got {int(masks.max())} cells in {_time.time()-t0:.1f}s.')

            self._last_mask = masks
            self._last_mask_save_path = save_path
            self._set_progress(100, f'Saved {save_path.name}. Edit with right-click / Ctrl+click.')

            # Put mask into viewer for editing — del + add instead of .data =
            if 'mask_biosensor' in self.viewer.layers:
                try:
                    del self.viewer.layers['mask_biosensor']
                except Exception:
                    pass
            lm = self.viewer.add_labels(masks.astype(np.int32), name='mask_biosensor')
            lm.mouse_drag_callbacks.append(self._ctrl_click_delete)
            if 'draw_points' not in self.viewer.layers:
                self.viewer.add_points(
                    np.zeros((0, 2), dtype=float), name='draw_points',
                    size=8, face_color='yellow', edge_color='black',
                )
            show_info(f'Biosensor segmentation: {int(masks.max())} cells.')
        except Exception as e:
            self._set_progress(0, f'ERROR: {e}')
            show_warning(f'Segmentation failed: {e}')
            traceback.print_exc()
        finally:
            self.seg_btn.enabled = True
            self.reseg_btn.enabled = True

    def _on_save(self):
        if 'mask_biosensor' not in self.viewer.layers:
            show_warning('No mask_biosensor layer to save.')
            return
        arr = np.asarray(self.viewer.layers['mask_biosensor'].data).astype(np.uint16)
        seg_ref_path = self._seg_img_save_path
        if seg_ref_path is None:
            seg_ref_path = Path(str(self.sample_folder.value)) / 'seg_image.tif'
        dest = seg_ref_path.parent / (seg_ref_path.stem + '_seg.npy')
        np.save(str(dest), arr)
        self._last_mask_save_path = dest
        show_info(f'Saved {dest.name}')

    # ---------- Fine-tune ----------

    def _on_finetune(self):
        if 'mask_biosensor' not in self.viewer.layers:
            show_warning('Segment first so there is a mask to fine-tune on.')
            return
        img = self._get_seg_img()
        if img is None:
            show_warning('No tracking image loaded.')
            return
        img2d = np.asarray(img, dtype=np.float32)
        if img2d.ndim > 2:
            img2d = np.squeeze(img2d)

        mask = np.asarray(self.viewer.layers['mask_biosensor'].data, dtype=np.int32)
        if int(mask.max()) == 0:
            show_warning('Mask is empty — nothing to fine-tune on.')
            return

        base_name = str(self.seg_model.value)
        use_assist = base_name.lower().startswith('bs-bc-assist')
        if use_assist and 'barcode_cls' not in self.viewer.layers:
            show_warning('BS-BC-assist model needs a confirmed barcode_cls layer. '
                         'Click "Load / Confirm Barcode" first.')
            return

        img_in, channels = self._build_seg_input(img2d, use_assist)
        ts = datetime.now().strftime('%m%d-%H%M')
        new_name = f'{base_name}-ft{ts}'
        sample_dir = Path(str(self.sample_folder.value))
        save_dir = sample_dir / '_finetune' / new_name
        save_dir.mkdir(parents=True, exist_ok=True)

        epochs = int(self.ft_epochs.value)
        self.ft_btn.enabled = False
        self._set_progress(
            10,
            f'Fine-tuning {base_name} -> {new_name} ({epochs} epochs) in a '
            f'subprocess so PyTorch GL/CUDA state does not leak into napari...'
        )
        try:
            import time as _time
            t0 = _time.time()
            new_path = _run_finetune_subprocess(
                img=img_in, mask=mask,
                base_name=base_name, new_name=new_name,
                save_dir=save_dir, n_epochs=epochs,
                channels=channels, use_gpu=bool(self.use_gpu.value),
                extra_roots=[str(sample_dir)],
            )
            self._set_progress(95, f'Training done in {_time.time()-t0:.1f}s. Saving...')
            self._model_cache.pop(new_name, None)
            # Full rescan so the new model lands in the right slot (mtime ord)
            # and is also visible on the next napari session.
            self._refresh_seg_model_choices()
            if new_name in self.seg_model.choices:
                self.seg_model.value = new_name
            self._set_progress(100, f'Fine-tuned model saved: {new_name}.')
            show_info(f'Fine-tuned model: {new_path}')
        except Exception as e:
            self._set_progress(0, f'Fine-tune ERROR: {e}')
            show_warning(f'Fine-tune failed: {e}')
            traceback.print_exc()
        finally:
            self.ft_btn.enabled = True

    def _on_finetune_multi(self):
        """Open the multi-folder fine-tune dialog for the biosensor model.

        Each added folder is expected to contain a ``seg_image.tif`` and an
        edited ``seg_image_seg.npy`` mask saved from a prior BiosensorSeg
        session. The dialog loads all valid pairs and trains them jointly.
        """
        base_name = str(self.seg_model.value)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        default_new = f'BS-multi-{ts}'
        sd = self.sample_folder.value
        save_root = (Path(str(sd)) / '_finetune') if sd else _BARCODE_MODEL_ROOT / '_cellpose_finetune_multi'

        def _done(new_name, new_path, n_samples):
            self._model_cache.pop(new_name, None)
            self._refresh_seg_model_choices()
            if new_name in self.seg_model.choices:
                self.seg_model.value = new_name
            self._set_progress(
                100,
                f'Multi-folder fine-tune done: {new_name} ({n_samples} samples).'
            )
            show_info(f'Multi-folder fine-tune done: {new_name} ({n_samples} samples).')

        def _err(exc):
            self._set_progress(0, f'Multi-folder fine-tune ERROR: {exc}')
            show_warning(f'Multi-folder fine-tune failed: {exc}')

        _open_multi_finetune_dialog(
            parent_widget=self,
            target='bs',
            base_name=base_name,
            default_new_name=default_new,
            default_epochs=int(self.ft_epochs.value),
            use_gpu=bool(self.use_gpu.value),
            save_root=save_root,
            on_done=_done,
            on_error=_err,
        )

    # ---------- Viewer interaction (edit mask) ----------

    def _bind_viewer_callbacks(self):
        try:
            @self.viewer.bind_key('Z', overwrite=True)
            def _toggle_mask(_v):
                if 'mask_biosensor' in self.viewer.layers:
                    layer = self.viewer.layers['mask_biosensor']
                    layer.visible = not layer.visible
                    self.viewer.layers.selection.active = layer

            @self.viewer.bind_key('X', overwrite=True)
            def _toggle_barcode(_v):
                if 'barcode_cls' in self.viewer.layers:
                    layer = self.viewer.layers['barcode_cls']
                    layer.visible = not layer.visible
                    self.viewer.layers.selection.active = layer

            @self.viewer.bind_key('Escape', overwrite=True)
            def _cancel_poly(_v):
                self._draw_state = {'layer': '', 'pts': [], 'active': False}
                if 'draw_points' in self.viewer.layers:
                    self.viewer.layers['draw_points'].data = np.zeros((0, 2), dtype=float)

            @self.viewer.bind_key('Enter', overwrite=True)
            def _commit_poly_key(_v):
                self._commit_polygon()

            @self.viewer.bind_key('S', overwrite=True)
            def _cycle_contrast(_v):
                name = 'seg_image'
                if name not in self.viewer.layers:
                    return
                layer = self.viewer.layers[name]
                data = np.asarray(layer.data)
                if data.size == 0:
                    return
                vmin = float(np.nanmin(data))
                vmax = float(np.nanmax(data))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    return
                self._contrast_idx = (self._contrast_idx + 1) % len(self._contrast_ratios)
                ratio = self._contrast_ratios[self._contrast_idx]
                layer.contrast_limits = (vmin, vmin + (vmax - vmin) * ratio)
        except Exception as e:
            print(f'[BiosensorSeg] key binding error: {e}')

        if self._right_click_add not in self.viewer.mouse_drag_callbacks:
            self.viewer.mouse_drag_callbacks.append(self._right_click_add)
        if self._mouse_track not in self.viewer.mouse_move_callbacks:
            self.viewer.mouse_move_callbacks.append(self._mouse_track)

    def _right_click_add(self, _viewer, event):
        btn = getattr(event, 'button', None)
        is_right = (btn == 2) or (str(btn).lower() == 'right')
        if not is_right:
            return
        if 'mask_biosensor' not in self.viewer.layers:
            return
        active = self.viewer.layers.selection.active
        if active is None or active.name != 'mask_biosensor':
            return
        layer = self.viewer.layers['mask_biosensor']
        try:
            yx = layer.world_to_data(event.position)[:2]
        except Exception:
            yx = event.position[:2]
        yx = (float(yx[0]), float(yx[1]))
        if not self._draw_state['active']:
            self._draw_state = {'active': True, 'layer': 'mask_biosensor', 'pts': [yx]}
            if 'draw_points' in self.viewer.layers:
                self.viewer.layers['draw_points'].data = np.array([yx], dtype=float)
            return
        self._commit_polygon()

    def _mouse_track(self, _viewer, event):
        if not self._draw_state['active']:
            return
        if 'mask_biosensor' not in self.viewer.layers:
            return
        layer = self.viewer.layers['mask_biosensor']
        try:
            yx = layer.world_to_data(event.position)[:2]
        except Exception:
            yx = event.position[:2]
        yx = (float(yx[0]), float(yx[1]))
        pts = self._draw_state['pts']
        if len(pts) == 0:
            pts.append(yx)
        else:
            y_last, x_last = pts[-1]
            if ((yx[0] - y_last) ** 2 + (yx[1] - x_last) ** 2) ** 0.5 >= 3.0:
                pts.append(yx)
        self._draw_state['pts'] = pts
        if 'draw_points' in self.viewer.layers:
            self.viewer.layers['draw_points'].data = np.array(pts, dtype=float)
        if len(pts) >= 8:
            y0, x0 = pts[0]
            if ((yx[0] - y0) ** 2 + (yx[1] - x0) ** 2) ** 0.5 <= 18:
                self._commit_polygon()

    def _commit_polygon(self):
        if 'mask_biosensor' not in self.viewer.layers:
            return
        pts = self._draw_state['pts']
        if len(pts) < 3:
            return
        layer = self.viewer.layers['mask_biosensor']
        mask = np.asarray(layer.data)
        used = set(np.unique(mask).astype(int).tolist())
        used.discard(0)
        new_id = 1
        while new_id in used:
            new_id += 1
        new_mask = mask.copy()
        poly_xy = np.array([[int(round(x)), int(round(y))] for y, x in pts], dtype=np.int32)
        cv2.fillPoly(new_mask, [poly_xy], int(new_id))
        layer.data = new_mask
        self._draw_state = {'active': False, 'layer': '', 'pts': []}
        if 'draw_points' in self.viewer.layers:
            self.viewer.layers['draw_points'].data = np.zeros((0, 2), dtype=float)

    def _ctrl_click_delete(self, layer, event):
        if 'Control' not in event.modifiers:
            return
        v = layer.get_value(event.position, world=True)
        if v is None:
            return
        try:
            v = int(v)
        except Exception:
            return
        if v <= 0:
            return
        dat = np.asarray(layer.data)
        dat[dat == v] = 0
        layer.data = dat


# Backward-compat alias: the widget used to be named ``KMeansCluster``; we
# renamed it to ``SeededKMeans`` once it became clear the implementation is
# a Seeded-KMeans classifier (Basu et al. 2002), not unsupervised clustering.
# Keep the old name exported so external code (saved napari layouts,
# older napari.yaml caches) still resolves.
KMeansCluster = SeededKMeans
