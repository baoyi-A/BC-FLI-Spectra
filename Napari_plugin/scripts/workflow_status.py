#!/usr/bin/env python
"""Report where a sample folder stands in the BC-FLIM-Spectra workflow.

Answers three questions about a folder, without opening napari and without
changing anything:

    which steps have run?      what should I do next?      does the output look right?

Written for an assistant to run on the user's behalf: a biologist part-way
through the seven steps can ask "how am I doing" and get numbers rather than an
opinion. Every check is read-only.

    python workflow_status.py <sample folder>
    python workflow_status.py <sample folder> --json

Exit status is 0 when the folder is coherent, 1 when a quality check failed, so
the script can also be used as a gate. A step that has not run yet is not a
failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STEPS = [
    "1 PTU Reader", "2 Barcode Seg", "3 Calculate FLIM-S", "4 Seeded K-Means",
    "5 Biosensor Seg", "6 B&P Tracker", "7 NaCha",
]


def _load_mask(path: Path):
    """Cellpose .npy, either a bare array or a dict with 'masks'."""
    raw = np.load(str(path), allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        raw = raw.get("masks", raw)
    return np.asarray(raw)


def _n_objects(path: Path) -> int:
    try:
        m = _load_mask(path)
        return int(len(np.unique(m)) - (1 if (m == 0).any() else 0))
    except Exception:
        return -1


class Report:
    def __init__(self, folder: Path):
        self.folder = folder
        self.steps: dict[str, dict] = {}
        self.checks: list[tuple[str, bool, str]] = []
        self.next: list[str] = []

    def step(self, name, done, **info):
        self.steps[name] = dict(done=bool(done), **info)

    def check(self, label, ok, detail=""):
        self.checks.append((label, bool(ok), detail))

    def todo(self, msg):
        self.next.append(msg)


def inspect(folder: Path) -> Report:
    r = Report(folder)
    intensity = folder / "intensity"
    flim = folder / "flim_stack"

    # ---- step 1 -----------------------------------------------------------
    ptus = sorted((folder / "raw").glob("*.ptu")) if (folder / "raw").is_dir() else []
    sums = sorted(intensity.glob("*_sum.tif")) if intensity.is_dir() else []
    stacks = sorted(flim.glob("*_ch*.tif")) if flim.is_dir() else []
    taus = sorted(folder.glob("*_fastflim_tau.tif"))
    fovs = [p.stem[:-4] for p in sums]          # strip the trailing "_sum"
    r.step(STEPS[0], bool(sums), ptu=len(ptus), fovs=len(sums),
           decay_stacks=len(stacks), tau_maps=len(taus),
           fov_names=[p.name for p in sums][:12])
    if not sums:
        r.todo("Run PTU Reader on the raw/ folder — nothing is decoded yet.")
        return r
    if stacks:
        r.check("every FOV has decay stacks", len(stacks) >= len(sums),
                f"{len(stacks)} stack files for {len(sums)} FOV(s)")
    else:
        r.check("decay stacks present", False,
                "flim_stack/ is empty — Calculate FLIM-S cannot run")

    # ---- step 2 -----------------------------------------------------------
    n_masks = sorted(intensity.glob("*_seg_n.npy"))
    p_masks = sorted(intensity.glob("*_seg_p.npy"))
    counts = {p.name: _n_objects(p) for p in (n_masks + p_masks)[:24]}
    r.step(STEPS[1], bool(n_masks or p_masks), n_masks=len(n_masks),
           p_masks=len(p_masks), objects=counts)
    if not (n_masks or p_masks):
        r.todo("Run Barcode Seg — no *_seg_n.npy / *_seg_p.npy beside the intensity sums.")
    else:
        if len(n_masks) < len(sums):
            r.todo(f"Barcode Seg: {len(sums) - len(n_masks)} of {len(sums)} FOV(s) "
                   f"still have no N mask.")
        empty = [k for k, v in counts.items() if v == 0]
        r.check("no empty masks", not empty, f"empty: {empty}" if empty else "")

    # ---- step 3 -----------------------------------------------------------
    fs = folder / "FLIM-S.xlsx"
    if fs.is_file():
        import pandas as pd
        df = pd.read_excel(fs)
        loc = df.get("Localization")
        locs = (loc.fillna("").astype(str).str.upper().value_counts().to_dict()
                if loc is not None else {})
        by_fov = (df["FOV"].astype(str).value_counts().to_dict()
                  if "FOV" in df.columns else {})
        int_cols = [c for c in ("Int 570-590", "Int 590-610", "Int 610-638", "Int 638-720")
                    if c in df.columns]
        chans = [i + 1 for i, c in enumerate(int_cols) if df[c].notna().any()]
        r.step(STEPS[2], True, rows=len(df), fovs=len(by_fov), per_fov=by_fov,
               localisations=locs, channels_with_data=chans)

        if {"G", "S"} <= set(df.columns):
            g, s = df["G"].to_numpy(float), df["S"].to_numpy(float)
            fin = np.isfinite(g) & np.isfinite(s)
            r.check("S stays under the semicircle apex (0.5)",
                    bool(np.nanmax(s[fin]) <= 0.5 + 1e-6),
                    f"max S = {np.nanmax(s[fin]):.4f}")
            # how far outside the universal semicircle the points sit
            rad = np.hypot(g[fin] - 0.5, s[fin])
            r.check("points lie on or near the universal semicircle",
                    float(np.median(rad)) <= 0.52,
                    f"median radius {np.median(rad):.3f} (0.5 is the circle); "
                    f"a tail-window fit without IRF deconvolution sits slightly outside")
        if "Lifetime" in df.columns:
            lt = df["Lifetime"].to_numpy(float)
            fin = np.isfinite(lt)
            r.check("lifetimes are physical (0.1-10 ns)",
                    bool(fin.any() and lt[fin].min() > 0.1 and lt[fin].max() < 10),
                    f"{np.nanmin(lt):.2f}-{np.nanmax(lt):.2f} ns")
        if len(by_fov) > len(sums):
            r.check("workbook holds only this folder's FOVs", False,
                    f"{len(by_fov)} FOVs in FLIM-S.xlsx but {len(sums)} on disk — "
                    f"rows from an earlier run were merged in; tick 'Fresh FLIM-S.xlsx'")
        if len(chans) not in (0, 1, 2, 3, 4):
            pass
    else:
        r.step(STEPS[2], False)
        r.todo("Run Calculate FLIM-S — no FLIM-S.xlsx yet.")

    # ---- step 4 -----------------------------------------------------------
    cl = folder / "clustered.xlsx"
    if cl.is_file():
        import pandas as pd
        dc = pd.read_excel(cl)
        tag = dc.get("cluster_tag")
        loc_col = dc.get("cluster_local")
        dist = (tag.value_counts().to_dict() if tag is not None else {})
        # Seeded K-Means runs ONE localisation at a time, so rows of the other
        # localisation are simply untouched. Counting those as "declined" turns
        # a normal half-finished run into a scary number — the denominator has
        # to be the rows that were actually clustered.
        if tag is not None:
            considered = int(tag.notna().sum())
            n_out = int((tag.astype(str) == "Outlier").sum())
        else:
            considered = int((loc_col.notna()).sum()) if loc_col is not None else 0
            n_out = int((loc_col == 0).sum()) if loc_col is not None else 0
        classified = considered - n_out
        untouched = len(dc) - considered
        locs_done = sorted({str(t)[0] for t in (tag.dropna().unique() if tag is not None else [])
                            if str(t)[:1] in ("N", "M", "P")})
        r.step(STEPS[3], True, rows=len(dc), clustered=considered,
               classified=classified, declined=n_out, not_clustered=untouched,
               localisations_done=locs_done,
               classes=len([k for k in dist if k != "Outlier"]),
               distribution={k: int(v) for k, v in list(dist.items())[:20]})
        if considered > 0:
            frac = n_out / considered
            r.check("declined fraction is near the contamination setting",
                    frac <= 0.30,
                    f"{n_out} of {considered} clustered cells declined ({frac:.1%}); "
                    f"the default contamination is 0.10 per class")
        if untouched:
            r.todo(f"Seeded K-Means: {untouched} row(s) were never clustered — it runs one "
                   f"localisation at a time and only {locs_done or 'none'} has been done. "
                   f"Select the other localisation, place its seeds and run again if you "
                   f"need it.")
            sizes = [v for k, v in dist.items() if k != "Outlier"]
            if sizes:
                r.check("no class collapsed",
                        min(sizes) >= 3,
                        f"smallest class has {min(sizes)} cells "
                        f"(largest {max(sizes)})")
    else:
        r.step(STEPS[3], False)
        if fs.is_file():
            r.todo("Run Seeded K-Means — FLIM-S.xlsx exists but there is no clustered.xlsx.")

    # ---- step 5 -----------------------------------------------------------
    seg_imgs = sorted(folder.glob("*_seg_image.tif"))
    bare = folder / "seg_image.tif"
    if bare.is_file() and bare not in seg_imgs:
        seg_imgs.append(bare)
    bio_masks = [p for p in (folder.glob("*_seg_image_seg.npy")) if p.is_file()]
    confocal = sorted(p for p in folder.glob("*.tif")
                      if p.stem.lower().endswith(("-b", "-g", "-y")))
    r.step(STEPS[4], bool(bio_masks), seg_images=len(seg_imgs),
           masks=len(bio_masks), confocal_stacks=len(confocal),
           objects={p.name: _n_objects(p) for p in bio_masks[:6]})
    if seg_imgs and not bio_masks:
        r.todo("Biosensor Seg: a seg image exists but no *_seg_image_seg.npy was saved.")
    elif not bio_masks:
        if confocal:
            r.todo(f"Run Biosensor Seg — {len(confocal)} confocal channel stack(s) are here "
                   f"but nothing has been segmented on them yet.")
        elif cl.is_file():
            r.todo("Steps 5-7 (biosensor) have not run. They need the confocal B/G/Y stacks "
                   "in this folder; a barcode-only experiment is finished at step 4.")

    # ---- registration quality (the check people skip) ----------------------
    cls_tifs = sorted(intensity.glob("*-cls.tif")) if intensity.is_dir() else []
    if bio_masks and cls_tifs:
        try:
            import tifffile
            mask = _load_mask(bio_masks[0])
            cls = np.asarray(tifffile.imread(str(cls_tifs[0])))
            if cls.ndim > 2:
                cls = np.squeeze(cls)
            best = None
            for k, label in ((3, "90 CW"), (0, "0"), (2, "180"), (1, "270 CW")):
                a = np.rot90(cls, k=k)
                if a.shape != mask.shape:
                    ys = np.linspace(0, a.shape[0] - 1, mask.shape[0]).astype(int)
                    xs = np.linspace(0, a.shape[1] - 1, mask.shape[1]).astype(int)
                    a = a[np.ix_(ys, xs)]
                hit = purity = 0
                labels = [v for v in np.unique(mask) if v]
                for v in labels[:400]:
                    vals = a[mask == v]
                    vals = vals[vals > 0]
                    if vals.size:
                        counts_ = np.bincount(vals.astype(int))
                        hit += 1
                        purity += counts_.max() / vals.size
                n = max(len(labels[:400]), 1)
                score = (hit / n, purity / max(hit, 1))
                if best is None or score[0] * score[1] > best[1][0] * best[1][1]:
                    best = (label, score)
                if label == "90 CW":
                    default = score
            r.step("registration", True, best_rotation=best[0],
                   default_90cw_coverage=round(default[0], 3),
                   default_90cw_purity=round(default[1], 3),
                   best_coverage=round(best[1][0], 3), best_purity=round(best[1][1], 3))
            r.check("barcode classes land on biosensor cells",
                    default[0] >= 0.5 and default[1] >= 0.85,
                    f"at 90 CW: {default[0]:.1%} of cells carry a class, "
                    f"purity {default[1]:.1%}. Best rotation tested: {best[0]}. "
                    f"A wrong rotation still covers many cells but with low purity, "
                    f"so coverage alone does not prove alignment.")
        except Exception as e:
            r.check("registration measurable", False, f"could not measure: {e}")

    # ---- step 7 -----------------------------------------------------------
    sig = folder / "signal_analysis.xlsx"
    r.step(STEPS[6], sig.is_file() and sig.stat().st_size > 0,
           file=str(sig) if sig.is_file() else None)
    if bio_masks and not sig.is_file():
        r.todo("Run NaCha to get the per-class signal curves.")
    return r


def render(r: Report) -> str:
    out = [f"Sample folder: {r.folder}", ""]
    for name in STEPS:
        st = r.steps.get(name)
        if st is None:
            continue
        mark = "done" if st["done"] else "not run"
        detail = ", ".join(f"{k}={v}" for k, v in st.items()
                           if k != "done" and v not in (None, {}, [], 0))
        out.append(f"  [{mark:>7}] {name}" + (f"   {detail}" if detail else ""))
    if "registration" in r.steps:
        st = r.steps["registration"]
        out.append(f"  [  check] registration   best={st['best_rotation']}, "
                   f"90CW coverage={st['default_90cw_coverage']}, "
                   f"purity={st['default_90cw_purity']}")
    out.append("")
    if r.checks:
        out.append("Quality checks")
        for label, ok, detail in r.checks:
            out.append(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
        out.append("")
    out.append("Next")
    if r.next:
        out.extend(f"  - {t}" for t in r.next)
    else:
        out.append("  - Nothing obvious outstanding for this folder.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    if not a.folder.is_dir():
        print(f"not a folder: {a.folder}", file=sys.stderr)
        return 2
    r = inspect(a.folder)
    if a.json:
        print(json.dumps({"folder": str(r.folder), "steps": r.steps,
                          "checks": [{"check": c, "ok": o, "detail": d} for c, o, d in r.checks],
                          "next": r.next}, indent=2, default=str))
    else:
        print(render(r))
    return 1 if any(not ok for _, ok, _ in r.checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
