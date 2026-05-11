"""Post-processing helpers for Cellpose segmentation output.

Currently houses ``merge_fragments`` — a morphological close + min-area
filter used when a finetuned model (e.g. the JQW NBL2 membrane model)
produces heavily over-segmented predictions. The widget calls this only
when a per-model ``config.json`` requests it; default behaviour is a
no-op so the main flow is unchanged.
"""
from __future__ import annotations

import numpy as np


def merge_fragments(
    masks: np.ndarray,
    *,
    gap_px: int = 3,
    min_px: int = 400,
) -> np.ndarray:
    """Close gaps then drop small components.

    Algorithm:
      1. Threshold the mask to a boolean foreground (any non-zero label).
      2. Binary dilate by ``gap_px`` → connected components → binary
         erode by ``gap_px`` so neighbouring fragments separated by a
         thin gap end up sharing one label.
      3. Re-label by connected components on the closed mask.
      4. Drop labels with area < ``min_px``.
      5. Repack labels 1..N (no gaps in the label space).

    Parameters
    ----------
    masks : ndarray, int
        Cellpose mask output (HxW int with label 0 = background).
    gap_px : int
        Morphological closing radius. ``0`` skips the closing step (just
        re-labels + min-area filter).
    min_px : int
        Drop connected components with fewer than this many pixels.

    Returns
    -------
    ndarray, int32
        Merged & filtered mask, repacked to 1..N.
    """
    from scipy import ndimage as _ndi

    if masks is None:
        return masks
    a = np.asarray(masks)
    if a.ndim != 2:
        return a.astype(np.int32, copy=False)

    fg = a > 0
    if not fg.any():
        return np.zeros_like(a, dtype=np.int32)

    if gap_px and gap_px > 0:
        # Square structuring element of side (2*gap_px+1). Cellpose
        # fragments are typically thin-gap on membrane data — square is
        # close enough to a disk for this radius range and ~10x cheaper.
        struct = np.ones((2 * gap_px + 1, 2 * gap_px + 1), dtype=bool)
        closed = _ndi.binary_closing(fg, structure=struct)
    else:
        closed = fg

    labels, _n = _ndi.label(closed)
    if _n == 0:
        return np.zeros_like(a, dtype=np.int32)

    # Drop small components.
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(min_px)
    keep[0] = False  # background

    if not keep.any():
        return np.zeros_like(a, dtype=np.int32)

    # Build a lookup that maps old label → new packed label (or 0).
    new_labels = np.zeros_like(sizes, dtype=np.int32)
    new_idx = 0
    for old in range(1, sizes.shape[0]):
        if keep[old]:
            new_idx += 1
            new_labels[old] = new_idx

    return new_labels[labels].astype(np.int32, copy=False)
