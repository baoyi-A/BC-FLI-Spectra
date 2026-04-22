"""Self-tests for the two new features:

1. Model-scan ordering: custom finetune folders show up in the dropdown,
   sorted by target-match then mtime-recent.
2. Multi-folder fine-tune subprocess: the runner accepts imgs/masks lists
   and trains jointly on cyto2.

Run with:
    D:/Softwares/Anaconda/Anaconda3/envs/BC-FLIM/python.exe \
        walkthrough/test_model_scan_and_multi_ft.py
"""
from __future__ import annotations

import os
import sys
import time
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _import_plugin():
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    if here not in sys.path:
        sys.path.insert(0, here)
    from flim_s_gen import _widget as _w
    return _w


def _fake_finetune_model(root: Path, name: str) -> Path:
    """Create the folder layout that _list_all_custom_models scans for.

    Structure created:
        <root>/<name>/models/<name>          (the 'weights file' — just a placeholder)
    """
    mdir = root / name / 'models'
    mdir.mkdir(parents=True, exist_ok=True)
    f = mdir / name
    f.write_bytes(b'fake-weights')
    return f


def test_scan_ordering(_w) -> bool:
    """Target-matching + mtime ordering."""
    tmp = Path(tempfile.mkdtemp(prefix='bcflim_scan_'))
    try:
        sd1 = tmp / 'sample_A' / '_finetune'
        sd2 = tmp / 'sample_B' / '_finetune'
        sd1.mkdir(parents=True)
        sd2.mkdir(parents=True)

        # Create 4 models, interleave by mtime to prove sort works.
        f1 = _fake_finetune_model(sd1, 'NinNC-old-260101')
        f2 = _fake_finetune_model(sd1, 'CinNC-old-260101')
        f3 = _fake_finetune_model(sd2, 'NinNC-new-260420')
        f4 = _fake_finetune_model(sd2, 'MiscModel-260415')

        # Set explicit mtimes: f1 oldest, then f2, f4, f3 newest.
        base = time.time() - 100
        for i, f in enumerate([f1, f2, f4, f3]):
            os.utime(f, (base + i, base + i))

        n_order = _w._list_all_custom_models(
            sample_dirs=[sd1.parent, sd2.parent], target_hint='n',
        )
        p_order = _w._list_all_custom_models(
            sample_dirs=[sd1.parent, sd2.parent], target_hint='p',
        )

        print(f'  N-hint order: {n_order}')
        print(f'  P-hint order: {p_order}')

        # Expected for N: target-hit (N-starts) first, then others, each by
        # mtime desc. So NinNC-new first, NinNC-old second; then non-hit
        # MiscModel (mtime newer), then CinNC-old (mtime older).
        # NOTE: hint 'p' also matches names starting with 'c' (our plugin
        # convention), so CinNC-old counts as a P hit.
        assert n_order[0] == 'NinNC-new-260420', f'N top wrong: {n_order[0]}'
        assert n_order[1] == 'NinNC-old-260101', f'N second wrong: {n_order[1]}'
        # Under P-hint: CinNC-old is the only hit, so it's first even though
        # it's the oldest mtime.
        assert p_order[0] == 'CinNC-old-260101', f'P top wrong: {p_order[0]}'

        print('[PASS] scan ordering respects target-hit then mtime desc.')
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multi_ft_subprocess(_w) -> bool:
    """Train cyto2 on 2 synthetic image/mask pairs, confirm subprocess accepts lists."""
    tmp = Path(tempfile.mkdtemp(prefix='bcflim_multi_ft_'))
    try:
        rng = np.random.default_rng(42)
        # 2 tiny synthetic "cell-like" images with one blob each.
        imgs = []
        masks = []
        for _ in range(2):
            img = (rng.random((128, 128), dtype=np.float32) * 30).astype(np.float32)
            yy, xx = np.ogrid[:128, :128]
            cy, cx = int(rng.integers(40, 90)), int(rng.integers(40, 90))
            blob = (yy - cy) ** 2 + (xx - cx) ** 2 < 25 ** 2
            img[blob] += 200
            m = np.zeros((128, 128), dtype=np.int32)
            m[blob] = 1
            imgs.append(img)
            masks.append(m)

        save_dir = tmp / 'multi_train'
        save_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        out = _w._run_finetune_subprocess(
            imgs=imgs, masks=masks,
            base_name='cyto2', new_name='test-multi-ft',
            save_dir=save_dir, n_epochs=2,
            channels=[0, 0], use_gpu=False, extra_roots=[],
        )
        print(f'  subproc returned: {out}  ({time.time()-t0:.1f}s)')
        assert Path(out).exists(), f'trained model not at {out}'
        print('[PASS] multi-image subprocess trained cyto2 on 2 pairs.')
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    _w = _import_plugin()
    results = []
    try:
        print('-- test_scan_ordering --')
        results.append(test_scan_ordering(_w))
    except AssertionError as e:
        print(f'[FAIL] scan ordering: {e}')
        results.append(False)
    try:
        print('-- test_multi_ft_subprocess --')
        results.append(test_multi_ft_subprocess(_w))
    except AssertionError as e:
        print(f'[FAIL] multi ft: {e}')
        results.append(False)

    print(f'\n{sum(results)}/{len(results)} tests passed.')
    return 0 if all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
