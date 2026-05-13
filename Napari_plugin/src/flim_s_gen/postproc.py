"""Post-processing helpers for Cellpose segmentation output.

Houses:

* ``merge_fragments`` — morphological close + min-area filter, plus
  optional interior-hole closing and per-label erosion. Used when a
  finetuned model produces heavily over-segmented predictions (e.g. the
  JQW NBL2 cell-membrane model) so neighbouring fragments belonging to
  the same membrane curve get one label.

* ``split_at_kinks`` — counterpart to ``merge_fragments``: split a label
  wherever its skeleton has a Y/T/X junction (default — branch-only),
  optionally also at Douglas-Peucker geometric corners (set ``epsilon``
  smaller to enable).

* ``membrane_pipeline`` — convenience one-call entry that runs the full
  membrane post-proc chain (merge → split → close holes → erode → final
  min-area filter) with sane defaults tuned for the JQW NBL2 model.

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
    dilate_px: int = 0,
) -> np.ndarray:
    """Close gaps, drop small components, optionally close holes / shrink / grow.

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
      8. Optionally dilate every label by ``dilate_px`` pixels — grows
         each cell outward, but only into background pixels. Adjacent
         labels never merge. Useful to restore a smoother boundary
         after over-erosion or to extend a thin membrane mask outward.

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
        pixels are filled in. ``0`` (default) = no filling.
    erode_px : int
        Per-cell erosion radius. ``0`` (default) = no erosion. Pixels
        within ``erode_px`` of any non-foreground boundary are demoted
        to background. Adjacent labels stay separate.
    dilate_px : int
        Per-cell dilation radius applied AFTER erode. ``0`` (default) =
        no dilation. Each label grows outward by this many pixels, but
        only into pixels that are currently background — labels never
        eat into their neighbours.

    Returns
    -------
    ndarray, int32
        Merged & filtered mask, labels repacked to 1..N. close_holes,
        erode and dilate operate per-label and preserve IDs.
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
        # Re-label only when we changed connectivity (closing can merge
        # previously-separate labels). Otherwise we'd destroy any label
        # structure created by an earlier step (e.g. split_at_kinks).
        labels, _n = _ndi.label(closed)
        if _n == 0:
            return np.zeros_like(a, dtype=np.int32)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= int(min_px)
        keep[0] = False
        if not keep.any():
            return np.zeros_like(a, dtype=np.int32)
        new_labels = np.zeros_like(sizes, dtype=np.int32)
        new_idx = 0
        for old in range(1, sizes.shape[0]):
            if keep[old]:
                new_idx += 1
                new_labels[old] = new_idx
        out = new_labels[labels].astype(np.int32, copy=False)
    else:
        # gap_px=0 path: preserve the caller's existing labels (don't re-CC).
        # Only apply the min_px filter per-label.
        out = a.astype(np.int32, copy=True)
        if min_px > 0:
            uniq = np.unique(out)
            for lbl in uniq:
                if lbl == 0:
                    continue
                m = (out == lbl)
                if m.sum() < min_px:
                    out[m] = 0
            # Repack 1..N.
            uniq2 = np.unique(out); uniq2 = uniq2[uniq2 > 0]
            remap = np.zeros(int(out.max()) + 1, dtype=np.int32)
            for i, old in enumerate(uniq2, start=1):
                remap[old] = i
            out = remap[out]

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

    if dilate_px and dilate_px > 0:
        # Per-label dilation, but only into pixels that are CURRENTLY
        # background. If two labels are adjacent, neither overruns the
        # other — dilation just fills the surrounding empty space. We
        # use distance_transform on the inverse-of-label-bg-mask to find
        # the nearest label per background pixel; pixels within
        # dilate_px of any label get that label's id.
        bg = out == 0
        if bg.any():
            # Distance from each bg pixel to the nearest non-bg pixel,
            # plus the indices of that nearest pixel.
            dist, inds = _ndi.distance_transform_edt(
                bg, return_indices=True,
            )
            assign = dist <= dilate_px
            if assign.any():
                grown = out.copy()
                grown[assign] = out[tuple(inds[:, assign])]
                out = grown

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


def _find_corners_dp(path, epsilon=8.0, min_segment_len=0):
    """Find geometrically significant corner indices along ``path`` using
    Douglas-Peucker polyline simplification.

    DP recursively replaces a polyline by the chord between endpoints
    whenever every internal point lies within ``epsilon`` pixels of that
    chord. Internal points that DP keeps are by definition the geometric
    corners — points the polyline cannot be approximated past without
    exceeding the tolerance. This is the standard "find the corners of
    a polygon" algorithm in computational geometry.

    Parameters
    ----------
    path : (N, 2) int ndarray
        Ordered list of skeleton pixel positions.
    epsilon : float
        DP tolerance in pixels. Smaller → more corners detected.
        For typical NBL2 membrane skeletons (median length 176 px),
        ε = 5–15 gives reasonable behaviour.
    min_segment_len : int
        Reject a candidate corner if it would leave a tail shorter than
        this many skeleton pixels on either side.

    Returns
    -------
    list of int
        Sorted indices in ``path`` where corners were found (excluding
        the two endpoints).
    """
    n = len(path)
    if n < 4:
        return []
    try:
        import cv2
    except ImportError:
        # No OpenCV — fall back to "no corners". Caller will keep label whole.
        return []
    pts = path.astype(np.int32).reshape(-1, 1, 2)
    simp = cv2.approxPolyDP(pts, float(epsilon), False).reshape(-1, 2)
    if len(simp) < 3:
        return []

    # Map every simplified vertex back to its index in `path`.
    # DP preserves order, so we walk both in lockstep.
    indices = []
    si = 0
    for i in range(n):
        if si >= len(simp):
            break
        if path[i][0] == simp[si][0] and path[i][1] == simp[si][1]:
            indices.append(i)
            si += 1
    # Drop the two endpoints (indices[0] and indices[-1]).
    corners = indices[1:-1] if len(indices) >= 2 else []

    # Min-tail check.
    if min_segment_len > 0 and corners:
        kept = []
        prev = 0
        for i in corners:
            if i - prev < min_segment_len:
                continue
            kept.append(i)
            prev = i
        while kept and (n - 1) - kept[-1] < min_segment_len:
            kept.pop()
        corners = kept
    return corners


def split_at_kinks(
    masks: np.ndarray,
    *,
    epsilon: float = 8.0,
    min_skel_len: int = 45,
    min_piece_area: int = 500,
    cut_width: int = 1,
    # Legacy aliases (old tangent-window API) — accepted for back-compat.
    # If supplied, they're translated into roughly-equivalent ``epsilon``
    # behaviour (smaller window / sharper angle ≈ smaller epsilon).
    kink_angle_deg: float | None = None,
    kink_window: int | None = None,
    min_segment_px: int | None = None,
) -> np.ndarray:
    """Split each label of ``masks`` at geometric corners along its centerline.

    Algorithm per label:
      1. Skeletonize the label into a 1-pixel-wide centerline.
      2. Always cut at branch points (skeleton pixels with >=3 neighbours).
         A Y/T/X-junction is by definition the right place to split.
      3. In each branch-free skeleton chain, run Douglas-Peucker polyline
         simplification with tolerance ``epsilon``. The vertices DP keeps
         (other than the two chain endpoints) are the geometric corners
         — points the polyline cannot be approximated past without
         exceeding the tolerance. Cut at each corner.
      4. Reject cuts whose resulting tail would be shorter than
         ``min_skel_len`` pixels.
      5. Erase a ``cut_width``-pixel-wide notch at each accepted cut.
      6. Connected-component label the cut skeleton. Each surviving
         sub-skeleton becomes a new label.
      7. Voronoi-regrow each sub-skeleton to the full-thickness mask by
         assigning every original-region pixel to its nearest sub-skel
         (via ``distance_transform_edt`` with ``return_indices=True``).
      8. Drop sub-pieces with area < ``min_piece_area`` and merge their
         pixels into the nearest surviving neighbour.

    Parameters
    ----------
    masks : ndarray
        Cellpose-style integer label image (0 = background).
    epsilon : float
        Douglas-Peucker tolerance in pixels. The polyline is simplified
        so every original point lies within ``epsilon`` of the simplified
        line. Internal vertices of the simplification are the corners.
        Larger ε → fewer, sharper corners. Suggested range for NBL2
        membrane skeletons (median length 176 px): ε = 5–20.
    min_skel_len : int
        Minimum skeleton length (in pixels) to allow a cut on either side
        of it. NBL2 GT 1%-percentile skel length is 45 px.
    min_piece_area : int
        Minimum pixel area for a sub-piece (after Voronoi regrowth) to
        survive. Smaller pieces are folded into the nearest neighbour.
        NBL2 GT 1%-percentile area is 377 px.
    cut_width : int
        Number of consecutive skeleton pixels to erase at each cut point
        (1 is usually enough for 4-connected skeletons).
    kink_angle_deg, kink_window, min_segment_px : legacy
        Accepted for backwards compatibility with the old tangent-window
        API. If supplied, ``min_segment_px`` falls back into the two
        explicit min thresholds. ``kink_angle_deg`` / ``kink_window`` are
        ignored by the new DP-based algorithm.

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

    # Resolve the param aliases. Explicit values win; min_segment_px is a
    # legacy fallback so existing configs keep working.
    if min_segment_px is not None:
        if min_skel_len == 45:      # still default => override
            min_skel_len = int(min_segment_px)
        if min_piece_area == 500:   # still default => override
            min_piece_area = int(min_segment_px)

    out = np.zeros_like(a, dtype=np.int32)
    next_lbl = 0

    # 8-connectivity kernel for skeleton-pixel degree (count of skel neighbours).
    _DEG_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    for lbl in range(1, int(a.max()) + 1):
        region = a == lbl
        area = int(region.sum())
        if area == 0:
            continue
        # Skip splitting on labels that are already too small to subdivide.
        if area < 2 * min_piece_area:
            next_lbl += 1
            out[region] = next_lbl
            continue

        skel = skeletonize(region)
        if not skel.any():
            next_lbl += 1
            out[region] = next_lbl
            continue

        # === Always cut at skeleton branch points (degree >= 3). ===
        # User observation: at a Y/T/X-junction the natural split is the
        # junction pixel itself, not somewhere 8 px down each arm. So we
        # remove branch pixels from cut_skel BEFORE walking segments —
        # this leaves the arms as independent branch-free chains for the
        # subsequent kink detection.
        cut_skel = skel.copy()
        deg = _ndi.convolve(skel.astype(np.uint8), _DEG_KERNEL,
                            mode='constant', cval=0) * skel
        branch_mask = (deg >= 3) & skel
        if branch_mask.any():
            # Erase branch pixels AND their immediate 8-neighbours that are
            # also branch pixels (densely packed junctions) so the cut is
            # clean (no 8-connectivity leak across the junction).
            cut_skel = cut_skel & ~branch_mask

        # Re-walk on the branch-free skeleton.
        segments = _walk_skeleton_segments(cut_skel)
        for seg in segments:
            cuts = _find_corners_dp(
                seg, epsilon=epsilon, min_segment_len=min_skel_len,
            )
            for ci in cuts:
                for k in range(max(0, ci - (cut_width // 2)),
                               min(len(seg), ci + (cut_width // 2) + 1)):
                    y, x = int(seg[k][0]), int(seg[k][1])
                    cut_skel[y, x] = False

        # 8-connectivity for skeleton CC — skeletons are 8-connected (include
        # diagonal neighbours). The default 4-connectivity would shatter a
        # diagonal chain into a fragment per pixel.
        sub_labels, n_sub = _ndi.label(cut_skel, structure=np.ones((3, 3), dtype=np.uint8))
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
            if piece_area < min_piece_area:
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


# ---------------------------------------------------------------------------
# membrane_pipeline — fixed-flow entry combining the four operations above
# ---------------------------------------------------------------------------


def membrane_pipeline(
    masks: np.ndarray,
    *,
    merge_gap: int = 3,
    min_merged_px: int = 400,
    epsilon: float = 10000.0,
    min_skel_len: int = 45,
    min_piece_area: int = 500,
    close_holes_px: int = 400,
    erode_px: int = 7,
    min_final_px: int = 200,
) -> np.ndarray:
    """Run the full membrane-segmentation post-proc chain in one call.

    Stages (applied in order):
      1. merge_fragments(gap=``merge_gap``, min=``min_merged_px``)
         Bridge nearby cellpose fragments via morphological closing;
         drop pieces smaller than ``min_merged_px``.
      2. split_at_kinks(epsilon=``epsilon``, min_skel_len=``min_skel_len``,
                         min_piece_area=``min_piece_area``)
         Cut at skeleton Y/T/X junctions (branch-only when
         ``epsilon`` is large; lower ``epsilon`` to also cut at DP corners).
      3. close pinprick interior holes <= ``close_holes_px`` per-label.
      4. Erode each label by ``erode_px`` pixels (thins thick membrane
         predictions toward a centerline of ~5-8 px).
      5. Drop labels whose final area is < ``min_final_px``. Erosion can
         leave tiny ghosts that this step removes.

    All defaults are tuned for the JQW NBL2 cell-membrane model:
      * Raw cellpose predicts membranes ~21 px thick with many 1-pixel
        gaps between fragments belonging to the same curve.
      * GT skeleton-length 1st percentile is 45 px; GT area 1st percentile
        is 377 px → ``min_skel_len=45``, ``min_piece_area=500``.
      * close_holes_px=400 = line-width-squared, fills pin-hole noise but
        keeps real cell interiors (those are >>500 px).
      * erode_px=7 brings the 21 px median line down to ~6-7 px.
      * After erosion, any label that shrank below 200 px is presumed
        spurious and dropped.

    Parameters
    ----------
    masks : ndarray
        Raw cellpose integer label image (0 = background).
    Other args : as in ``merge_fragments`` and ``split_at_kinks``.

    Returns
    -------
    ndarray, int32
        Final post-processed labels, packed 1..N.
    """
    if masks is None:
        return masks
    a = np.asarray(masks)
    if a.ndim != 2 or a.max() == 0:
        return a.astype(np.int32, copy=False)

    # 1. merge
    out = merge_fragments(a, gap_px=merge_gap, min_px=min_merged_px)
    # 2. split
    out = split_at_kinks(out, epsilon=epsilon,
                          min_skel_len=min_skel_len,
                          min_piece_area=min_piece_area)
    # 3+4. close holes & erode (gap=0/min=0 path preserves labels)
    out = merge_fragments(out, gap_px=0, min_px=0,
                          close_holes_px=close_holes_px,
                          erode_px=erode_px)
    # 5. final per-label area filter
    if min_final_px and min_final_px > 0 and out.max() > 0:
        sizes = np.bincount(out.ravel())
        drop = np.zeros_like(sizes, dtype=bool)
        for lbl in range(1, sizes.size):
            if 0 < sizes[lbl] < min_final_px:
                drop[lbl] = True
        if drop.any():
            out = out.copy()
            for lbl in np.where(drop)[0]:
                out[out == lbl] = 0
            # repack
            uniq = np.unique(out); uniq = uniq[uniq > 0]
            if uniq.size:
                remap = np.zeros(int(out.max()) + 1, dtype=np.int32)
                for i, old in enumerate(uniq, start=1):
                    remap[old] = i
                out = remap[out].astype(np.int32, copy=False)
    return out.astype(np.int32, copy=False)
