"""Harmony calibration to a labelled reference (A549 14-barcode by default).

OPTIONAL classifier path for SeededKMeans. The default seeded-KMeans flow
is unchanged — this is only invoked when the user toggles the Harmony
section on and clicks the calibrate button.

Pipeline (matches spec at
/dfs/share/liubeiLab/WBY/BC-FLIM/Code/mix_overlay/HARMONY_CALIBRATE_SPEC_260606.md):

  1. Load a labelled reference CSV (per-cell rows; label column varies
     by reference — A549 uses ``NLabelDisplay``, NOT ``NLabel``).
  2. Subsample to ``per_class`` cells per barcode class.
  3. Build a 5D feature matrix ``[G, S, Int1/(1-4), Int2/(1-4), Int3/(1-4)]``
     scaled by per-feature weights (default ``[2,2,1,1,1]``).
  4. Per-dataset z-score (reference + query each normalised with their
     own mean/std before concat).
  5. Run ``harmonypy.run_harmony`` on the stacked matrix with two batches
     (ref, query). ``nclust=20`` and ``theta=4`` are the spec defaults —
     the ``nclust=100`` Harmony default crushes rare classes (N16 → 0).
  6. kNN(k=15) on corrected ref → predict labels for corrected query.

Dependencies: ``harmonypy``, ``scikit-learn``, ``numpy``, ``pandas``.
The first is the only undeclared one — the widget surfaces a clear
install hint when import fails.

The function returns a pandas Series of predicted labels (string),
indexed like the input query DataFrame, so the caller can write them
straight back into the SeededKMeans ``clustered.xlsx`` output as the
``cluster_tag`` column without re-running KMeans.
"""
from __future__ import annotations

import warnings
import logging

import numpy as np
import pandas as pd

_log = logging.getLogger("bc_flim_spectra.harmony")


# Default features + weights from the spec.
FEATURE_COLS = ('G', 'S', 'Int 1/(1-4)', 'Int 2/(1-4)', 'Int 3/(1-4)')
DEFAULT_WEIGHTS = (2.0, 2.0, 1.0, 1.0, 1.0)


def build_5d_features(
    df: pd.DataFrame,
    *,
    weights=DEFAULT_WEIGHTS,
    feature_cols=FEATURE_COLS,
) -> np.ndarray:
    """Return an (N, 5) float matrix from a FLIM-S-style DataFrame.

    Drops rows with any NaN in the 5D features (caller should filter
    before harmony). Multiplies each column by the matching weight.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing FLIM-S columns for Harmony features: {missing}. "
            f"Available: {list(df.columns)[:20]}"
        )
    X = df.loc[:, list(feature_cols)].to_numpy(dtype=float)
    if X.shape[1] != len(weights):
        raise ValueError(
            f"weights length ({len(weights)}) doesn't match features ({X.shape[1]})"
        )
    X = X * np.asarray(weights, dtype=float)
    return X


def subsample_per_class(
    df: pd.DataFrame,
    *,
    label_col: str,
    per_class: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Take up to ``per_class`` rows from each label group.

    Classes with fewer rows pass through whole — we don't upsample,
    that would just duplicate noise into the harmony fit.
    """
    if label_col not in df.columns:
        raise KeyError(
            f"Label column {label_col!r} not in reference; "
            f"got {list(df.columns)[:20]}"
        )
    rng = np.random.default_rng(seed)
    pieces = []
    for lbl, group in df.groupby(label_col, sort=False):
        n = min(per_class, len(group))
        if n <= 0:
            continue
        idx = rng.choice(len(group), size=n, replace=False)
        pieces.append(group.iloc[idx])
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def _zscore(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Column-wise z-score; eps guards against zero-variance features."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def calibrate_and_classify(
    query_df: pd.DataFrame,
    *,
    ref_csv_path: str,
    ref_label_col: str = 'NLabelDisplay',
    per_class: int = 300,
    weights=DEFAULT_WEIGHTS,
    theta: float = 4.0,
    nclust: int = 20,
    max_iter_harmony: int = 30,
    knn_k: int = 15,
    seed: int = 0,
) -> pd.Series:
    """Calibrate query cells onto a labelled reference, return predicted labels.

    Parameters
    ----------
    query_df : DataFrame
        Per-cell rows from FLIM-S.xlsx (or similar). Must contain the
        5D feature columns (see ``FEATURE_COLS``).
    ref_csv_path : str
        Path to the reference CSV. Defaults to the A549 14-class lasso
        in the spec; the widget passes through whatever the user picks.
    ref_label_col : str
        Label column on the reference. Spec lists:
            A549    -> 'NLabelDisplay'   (NOT 'NLabel')
            HEK     -> 'NLabel'
            MDA     -> 'NLabel'
            SKOV3   -> 'CorrectedBarcode'
    per_class : int
        Subsample cap per class. Default 300 (≪ tens of thousands per
        class in the raw reference) — fits in seconds, plenty for kNN.
    weights : tuple of 5 floats
        Feature weights, default (2,2,1,1,1).
    theta : float
        Harmony integration strength; spec default 4.
    nclust : int
        Harmony cluster count; spec default 20 (NOT harmonypy's 100,
        which collapses rare classes).
    max_iter_harmony : int
        Harmony max iterations. Typically converges in ~4.
    knn_k : int
        k for the post-harmony kNN classifier on the corrected reference.
    seed : int
        RNG seed for the subsample step (kept stable across re-runs).

    Returns
    -------
    Series of str, indexed like ``query_df``.
        Predicted barcode label per query row. Rows whose 5D features
        contain NaN are returned as ``''`` (empty string) so the caller
        can leave them out of downstream class-tag counts.
    """
    # Lazy imports so the plain napari plugin doesn't choke on missing
    # harmonypy at startup — the GUI shows a clear hint instead.
    try:
        import harmonypy as hm  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Harmony calibration needs the 'harmonypy' package. Install "
            "with:  pip install harmonypy   (also: scikit-learn)"
        ) from e
    try:
        from sklearn.neighbors import KNeighborsClassifier
    except Exception as e:
        raise RuntimeError(
            "Harmony calibration needs scikit-learn. Install with: "
            "pip install scikit-learn"
        ) from e

    ref_df = pd.read_csv(ref_csv_path)
    _log.info('harmony ref: %s rows from %s', len(ref_df), ref_csv_path)
    if ref_label_col not in ref_df.columns:
        raise KeyError(
            f"Reference is missing label column {ref_label_col!r}. "
            f"Got: {list(ref_df.columns)[:20]}"
        )
    # Drop ref rows without labels OR missing 5D feature values.
    ref_df = ref_df.dropna(subset=[ref_label_col] + list(FEATURE_COLS)).copy()
    ref_df[ref_label_col] = ref_df[ref_label_col].astype(str)
    ref_df = subsample_per_class(
        ref_df, label_col=ref_label_col, per_class=per_class, seed=seed,
    )
    _log.info(
        'harmony ref after subsample: %s rows, %s classes',
        len(ref_df), ref_df[ref_label_col].nunique(),
    )

    # Query rows with any NaN in features are excluded from harmony but
    # tracked so the returned Series matches query_df.index.
    keep_mask = query_df[list(FEATURE_COLS)].notna().all(axis=1).to_numpy()
    if not keep_mask.any():
        return pd.Series([''] * len(query_df), index=query_df.index, dtype=object)

    X_ref = build_5d_features(ref_df, weights=weights)
    X_query = build_5d_features(query_df.loc[keep_mask], weights=weights)
    y_ref = ref_df[ref_label_col].to_numpy()

    # Per-dataset z-score BEFORE concat — keeps each dataset's distribution
    # centred on its own stats. The spec is explicit on this step.
    X_ref_z = _zscore(X_ref)
    X_query_z = _zscore(X_query)
    X_stack = np.vstack([X_ref_z, X_query_z]).astype(np.float64)
    meta = pd.DataFrame({
        'batch': ['ref'] * len(X_ref_z) + ['query'] * len(X_query_z),
    })

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ho = hm.run_harmony(
            X_stack, meta, vars_use=['batch'],
            theta=theta, nclust=nclust,
            max_iter_harmony=max_iter_harmony,
        )
    Z = np.asarray(ho.Z_corr).T  # (n_obs, n_features)
    Z_ref = Z[:len(X_ref_z)]
    Z_query = Z[len(X_ref_z):]
    _log.info(
        'harmony done; Z shape=%s, ref=%s, query=%s',
        Z.shape, Z_ref.shape[0], Z_query.shape[0],
    )

    knn = KNeighborsClassifier(n_neighbors=int(knn_k))
    knn.fit(Z_ref, y_ref)
    y_query = knn.predict(Z_query)

    # Reassemble into a Series matching query_df.index
    out = pd.Series([''] * len(query_df), index=query_df.index, dtype=object)
    keep_index = query_df.index[keep_mask]
    out.loc[keep_index] = y_query
    return out


def harmony_available() -> bool:
    """Quick truthy check used by the widget to grey out the button when
    harmonypy isn't installed."""
    try:
        import harmonypy  # noqa: F401
        from sklearn.neighbors import KNeighborsClassifier  # noqa: F401
        return True
    except Exception:
        return False
