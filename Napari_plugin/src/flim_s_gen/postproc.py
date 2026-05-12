"""Post-processing helpers for Cellpose segmentation output.

Houses two membrane-segmentation post-proc steps:

* ``merge_fragments`` — morphological close + min-area filter. Used when a
  finetuned model produces heavily over-segmented predictions (e.g. the
  JQW NBL2 cell-membrane model) so neighbouring fragments belonging to the
  same membrane curve get one label.

* ``split_at_kinks`` — split a label wherever its skeleton bends by more
  than a threshold angle. Counterpart to ``merge_fragments``: it undoes
  over-merging where the closing step accidentally fused two different
  membranes into one label by walking the centerline and cutting at
  geometric kinks.

The widget calls these via the per-model ``config.json``. Default
behaviour is a no-op so legacy models keep working.
"""
from __future__ import annotations

import numpy as np


def merge_fragments(
    masks: np.ndarray,
    *,
    gap_px: int = 3,
    min_px: int = 400,
    close_holes_px: int = 0,
    erode_px: int = 0,
) -> np.ndarray:
    """Close gaps, drop small components, optionally close holes / shrink.

    Algorithm (in order):
      1. Threshold the mask to a boolean foreground (any non-zero label).
      2. Binary dilate by ``gap_px`` → connected components → binary
         erode by ``gap_px`` so neighbouring fragments separated by a
         thin gap end up sharing one label.
      3. Re-label by connected components on the closed mask.
      4. Drop labels with area < ``min_px``.
      5. Repack labels 1..N (no gaps in the label space).
      6. Optionally fill interior holes <= ``close_holes_px`` pixels per
         label — closes pinprick gaps inside thick membrane masks so the
         cell becomes a solid region rather than just its outline.
      7. Optionally erode every label by ``erode_px`` pixels (peel one
         ring off the outline) — useful when the predicted membrane is
         too thick and the user wants to thin the masks before
         downstream feature extraction.

    Parameters
    ----------
    masks : ndarray, int
        Cellpose mask output (HxW int with label 0 = background).
    gap_px : int
        Morphological closing radius. ``0`` skips the closing step (just
        re-labels + min-area filter).
    min_px : int
        Drop connected components with fewer than this many pixels.
    close_holes_px : int
        Per-cell hole filling cap. For each label, holes (background
        regions fully enclosed by that label) with area <= this many
        pixels are filled in. ``0`` (default) = no filling. Useful when
        the model traces the cell as a ring and you want a solid disc.
    erode_px : int
        Per-cell erosion radius applied last. ``0`` (default) = no
        erosion. Pixels within this many pixels of any non-foreground
        boundary are demoted to background. Adjacent labels are treated
        as separate regions so the eroded boundary follows each cell.

    Returns
    -------
    ndarray, int32
        Merged & filtered mask, labels repacked to 1..N. close_holes and
        erode operate per-label and preserve IDs.
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

    out = new_labels[labels].astype(np.int32, copy=False)

    if close_holes_px and close_holes_px > 0:
        # Per-label hole filling. For each cell, find background regions
        # fully enclosed by that cell, and fill them in if their area is
        # below the cap. Per-label rather than global so a hole next to
        # another cell doesn't get "filled" by both — only the enclosing
        # cell claims it.
        filled = out.copy()
        for lbl in range(1, int(out.max()) + 1):
            region = out == lbl
            if not region.any():
                continue
            fully_filled = _ndi.binary_fill_holes(region)
            holes = fully_filled & ~region
            if not holes.any():
                continue
            hole_labels, _nh = _ndi.label(holes)
            if _nh == 0:
                continue
            hole_sizes = np.bincount(hole_labels.ravel())
            for h in range(1, _nh + 1):
                if hole_sizes[h] <= close_holes_px:
                    filled[hole_labels == h] = lbl
        out = filled

    if erode_px and erode_px > 0:
        # Per-label erosion: distance_transform_edt on the boolean mask of
        # each label tells us how far each pixel is from THIS label's
        # boundary. Demote any pixel within erode_px of the boundary to
        # background. We do it per-label so two touching labels don't
        # melt into each other through erosion.
        eroded = np.zeros_like(out, dtype=np.int32)
        for lbl in range(1, int(out.max()) + 1):
            region = out == lbl
            if not region.any():
                continue
            dist = _ndi.distance_transform_edt(region)
            eroded[dist > erode_px] = lbl
        out = eroded

    return out


# ---------------------------------------------------------------------------
# Kink splitting
# ---------------------------------------------------------------------------


def _walk_skeleton_segments(skel_bool):
    """Yield ordered (N, 2) integer arrays for each branch-free segment of
    a 1-pixel-wide skeleton.

    A "segment" is a maximal chain of skeleton pixels whose interior
    pixels each have exactly 2 skeleton neighbours (so the chain has no
    branches). Endpoints (1 neighbour) and branch-points (>=3 neighbours)
    terminate a segment.

    Closed loops (every pixel has exactly 2 neighbours) are returned as a
    single segment starting from an arbitrary pixel and visiting each
    pixel once.
    """
    from scipy import ndimage as _ndi
    if not skel_bool.any():
        return []
    kernel = np.ones((3, 3), dtype=np.uint8); kernel[1, 1] = 0
    deg = _ndi.convolve(skel_bool.astype(np.uint8), kernel,
                        mode='constant', cval=0) * skel_bool
    endpoints = np.argwhere(deg == 1)
    branches  = np.argwhere(deg >= 3)
    visited = np.zeros_like(skel_bool, dtype=bool)
    # Branch points anchor multiple segments; mark them visited so a walk
    # stops at them, then we will pull them out of the visited set for
    # them to also start their own segments toward each branch direction.
    for by, bx in branches:
        visited[by, bx] = True

    segments = []

    def _walk_from(start, allow_branch_pass=False):
        path = [tuple(start)]
        visited[tuple(start)] = True
        cur = tuple(start)
        prev = None
        while True:
            y, x = cur
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < skel_bool.shape[0] and 0 <= nx < skel_bool.shape[1]):
                        continue
                    if not skel_bool[ny, nx]:
                        continue
                    if visited[ny, nx]:
                        continue
                    nbrs.append((ny, nx))
            if not nbrs:
                break
            # Prefer 4-connected neighbours, then 8-connected
            nbrs.sort(key=lambda p: 0 if (p[0] - y == 0 or p[1] - x == 0) else 1)
            nxt = nbrs[0]
            path.append(nxt)
            visited[nxt] = True
            prev = cur
            cur = nxt
        return np.array(path, dtype=np.int32)

    # 1) Walk from each endpoint
    for ep in endpoints:
        if visited[tuple(ep)]:
            continue
        segments.append(_walk_from(ep))

    # 2) Walk between branch points (each branch starts a segment toward
    # every unvisited neighbour direction).
    for by, bx in branches:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = by + dy, bx + dx
                if not (0 <= ny < skel_bool.shape[0] and 0 <= nx < skel_bool.shape[1]):
                    continue
                if not skel_bool[ny, nx] or visited[ny, nx]:
                    continue
                # This neighbour seeds a new segment; the branch itself is
                # treated as the segment's first point so the geometry is
                # contiguous.
                start_path = [(by, bx), (ny, nx)]
                visited[ny, nx] = True
                cur = (ny, nx)
                while True:
                    y, x = cur
                    cands = []
                    for dy2 in (-1, 0, 1):
                        for dx2 in (-1, 0, 1):
                            if dy2 == 0 and dx2 == 0:
                                continue
                            ny2, nx2 = y + dy2, x + dx2
                            if not (0 <= ny2 < skel_bool.shape[0] and 0 <= nx2 < skel_bool.shape[1]):
                                continue
                            if not skel_bool[ny2, nx2]:
                                continue
                            if visited[ny2, nx2]:
                                continue
                            cands.append((ny2, nx2))
                    if not cands:
                        break
                    cands.sort(key=lambda p: 0 if (p[0] - y == 0 or p[1] - x == 0) else 1)
                    nxt = cands[0]
                    start_path.append(nxt)
                    visited[nxt] = True
                    cur = nxt
                segments.append(np.array(start_path, dtype=np.int32))

    # 3) Closed loops with no branch / endpoint — pick any unvisited skeleton
    # pixel as start
    remaining = (skel_bool & ~visited)
    while remaining.any():
        ys, xs = np.where(remaining)
        start = np.array([ys[0], xs[0]], dtype=np.int32)
        segments.append(_walk_from(start))
        remaining = (skel_bool & ~visited)

    return segments


def _find_kink_indices(path, window=8, angle_deg=60.0, min_segment_len=0):
    """Return a sorted list of indices into ``path`` where the curve bends
    by more than ``angle_deg`` (measured as the turning angle between the
    incoming and outgoing tangents averaged over ``window`` pixels).

    A candidate cut at index ``i`` is rejected if it would create a tail
    shorter than ``min_segment_len`` on either side of the cut, considering
    the cuts already accepted.
    """
    n = len(path)
    if n < 2 * window + 2:
        return []
    cos_thr = float(np.cos(np.deg2rad(angle_deg)))
    candidates = []
    for i in range(window, n - window):
        v_back = path[i] - path[i - window]
        v_fwd  = path[i + window] - path[i]
        nb = float(np.linalg.norm(v_back))
        nf = float(np.linalg.norm(v_fwd))
        if nb < 1.0 or nf < 1.0:
            continue
        cos_a = float(np.dot(v_back, v_fwd)) / (nb * nf)
        if cos_a < cos_thr:
            # turning angle = acos(cos_a). Larger = sharper.
            sharpness = 1.0 - cos_a  # for NMS sort
            candidates.append((i, sharpness))
    if not candidates:
        return []

    # Non-max suppression by `window` distance — keep the sharpest in each
    # neighbourhood.
    candidates.sort(key=lambda t: -t[1])
    accepted_pos = []
    for i, _ in candidates:
        if all(abs(i - j) >= window for j in accepted_pos):
            accepted_pos.append(i)
    accepted_pos.sort()

    # Enforce min_segment_len on the segments between cuts (and the two tails).
    # Walk the accepted cuts left to right; drop a cut if it leaves a tail
    # shorter than min_segment_len when paired with its left neighbour OR
    # the start of the path.
    if min_segment_len > 0:
        kept = []
        prev_pos = 0
        for i in accepted_pos:
            if i - prev_pos < min_segment_len:
                continue  # would leave a tiny segment before this cut
            kept.append(i)
            prev_pos = i
        # Final tail check: if last cut leaves <min_segment_len to the end,
        # drop it.
        while kept and (n - 1) - kept[-1] < min_segment_len:
            kept.pop()
        return kept
    return accepted_pos


def split_at_kinks(
    masks: np.ndarray,
    *,
    kink_angle_deg: float = 60.0,
    kink_window: int = 8,
    min_segment_px: int = 100,
    cut_width: int = 1,
) -> np.ndarray:
    """Split each label of ``masks`` at sharp bends along its centerline.

    Algorithm per label:
      1. Skeletonize the label into a 1-pixel-wide centerline.
      2. Break the skeleton into branch-free segments. Branches by
         themselves are already "natural" cut points.
      3. Inside each segment, walk along the path with a ``kink_window``
         look-ahead / look-behind, and flag pixels where the tangent turns
         by more than ``kink_angle_deg``.
      4. Reject cuts whose resulting tail would be shorter than
         ``min_segment_px`` pixels (along the skeleton — roughly = physical
         length since skel is 1-px wide).
      5. Erase a ``cut_width``-pixel-wide notch in the skeleton at each
         accepted cut so the connected components separate.
      6. Re-label the cut skeleton. Each surviving sub-skeleton becomes a
         new label.
      7. For every original-region pixel, assign it to the sub-skeleton
         whose centerline is nearest (Voronoi via ``distance_transform_edt``
         with ``return_indices=True``). This regrows full-thickness masks
         from the cut skeletons.
      8. Drop sub-pieces smaller than ``min_segment_px`` (their area in
         the regrown mask, not just the skeleton).

    Parameters
    ----------
    masks : ndarray
        Cellpose-style integer label image (0 = background).
    kink_angle_deg : float
        Cut where the centerline turns by more than this many degrees.
        60° is a reasonable starting value for membrane curves.
    kink_window : int
        Half-window used to estimate the tangent direction. Bigger window
        = smoother, fewer false cuts on jittery skeletons.
    min_segment_px : int
        Minimum length (skeleton step 5) AND minimum area (step 8). A cut
        is skipped if it would leave a piece shorter than this, and any
        regrown sub-piece smaller than this many pixels is dropped.
    cut_width : int
        Number of consecutive skeleton pixels to erase at each cut point
        (1 is usually enough for 4-connected skeletons; bump to 2-3 if
        you observe diagonal "leaks" reconnecting cut ends).

    Returns
    -------
    ndarray, int32
        Mask with the same dtype/shape as input. Labels are re-packed
        1..M with the split sub-pieces appended after the original IDs.
    """
    try:
        from skimage.morphology import skeletonize
    except ImportError as e:
        raise ImportError(
            'split_at_kinks needs scikit-image: pip install scikit-image'
        ) from e
    from scipy import ndimage as _ndi

    if masks is None:
        return masks
    a = np.asarray(masks)
    if a.ndim != 2 or a.max() == 0:
        return a.astype(np.int32, copy=False)

    out = np.zeros_like(a, dtype=np.int32)
    next_lbl = 0

    for lbl in range(1, int(a.max()) + 1):
        region = a == lbl
        area = int(region.sum())
        if area == 0:
            continue
        # Skip splitting on labels that are already too small to subdivide.
        if area < 2 * min_segment_px:
            next_lbl += 1
            out[region] = next_lbl
            continue

        skel = skeletonize(region)
        if not skel.any():
            next_lbl += 1
            out[region] = next_lbl
            continue

        cut_skel = skel.copy()
        segments = _walk_skeleton_segments(skel)
        for seg in segments:
            cuts = _find_kink_indices(
                seg, window=kink_window, angle_deg=kink_angle_deg,
                min_segment_len=min_segment_px,
            )
            for ci in cuts:
                for k in range(max(0, ci - (cut_width // 2)),
                               min(len(seg), ci + (cut_width // 2) + 1)):
                    y, x = int(seg[k][0]), int(seg[k][1])
                    cut_skel[y, x] = False

        sub_labels, n_sub = _ndi.label(cut_skel)
        if n_sub == 0:
            # All skeleton was cut away — keep original as one piece.
            next_lbl += 1
            out[region] = next_lbl
            continue
        if n_sub == 1:
            # No effective cut; keep as a single label.
            next_lbl += 1
            out[region] = next_lbl
            continue

        # Re-grow each sub-skeleton to full thickness via Voronoi.
        sources = sub_labels > 0
        if not sources.any():
            next_lbl += 1
            out[region] = next_lbl
            continue
        _dist, (iy, ix) = _ndi.distance_transform_edt(
            ~sources, return_indices=True,
        )
        nearest = sub_labels[iy, ix]

        # For each sub-label, mask to the region and check size; collect.
        local_assign = np.zeros_like(out)
        for sub in range(1, n_sub + 1):
            piece = region & (nearest == sub)
            piece_area = int(piece.sum())
            if piece_area < min_segment_px:
                # Too small — fold it into the largest neighbour sub later.
                continue
            next_lbl += 1
            local_assign[piece] = next_lbl

        # Anything in region not yet assigned (because its nearest sub was
        # too small or got dropped) — re-attach to the nearest *surviving*
        # sub. Re-run Voronoi using the surviving sub-skeleton pixels.
        leftover = region & (local_assign == 0)
        if leftover.any() and local_assign.max() > 0:
            surviving = local_assign > 0
            _, (iy2, ix2) = _ndi.distance_transform_edt(
                ~surviving, return_indices=True,
            )
            local_assign[leftover] = local_assign[iy2, ix2][leftover]

        out[region] = local_assign[region]
        # If somehow nothing survived, fall back to keeping as one piece.
        if not (out[region] > 0).any():
            next_lbl += 1
            out[region] = next_lbl

    # Repack labels 1..M (close any gaps the per-label loop left).
    uniq = np.unique(out)
    uniq = uniq[uniq > 0]
    if uniq.size and (uniq[-1] != uniq.size):
        remap = np.zeros(int(uniq.max()) + 1, dtype=np.int32)
        for i, old in enumerate(uniq, start=1):
            remap[old] = i
        out = remap[out]
    return out.astype(np.int32, copy=False)
