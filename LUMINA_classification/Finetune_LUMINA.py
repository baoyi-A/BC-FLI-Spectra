"""Few-shot domain adaptation of a trained LUMINA checkpoint.

A LUMINA checkpoint is trained on one cell line under one set of imaging conditions.
Applied to a different cell line it degrades: the spectral features move while the two
classifier heads keep the decision boundaries they were trained with. This script fixes
that by fine-tuning ONLY the two heads (`fc_nu`, `fc_mito`) on a handful of labelled
cells from the new domain, then evaluating on the cells that were not used.

It is the trainer for the protocol described in
`.claude/skills/lumina-network/references/domain-adaptation.md`.

WHY THIS SCRIPT TAKES COMMAND-LINE ARGUMENTS
    The other four scripts in this folder do not: every path and hyperparameter is a
    constant you edit in `main()`. This one is deliberately different. Adaptation is run
    many times over -- several support-set seeds, several dishes, a support dish and a
    different evaluation dish -- and the whole point of the protocol is that the numbers
    are only meaningful when the support set, the held-out set and the seed are stated
    together. Editing a constant between runs makes that impossible to record. Every knob
    is therefore a flag. The flags that change a number are echoed into every row of the
    prediction CSV; the complete command line, every flag and its resolved value, is
    written once per run to `finetune_run_config.csv`.

WHAT IT DOES
    1. Enumerates the cells of a *support* population and, if you give one, a separate
       *evaluation* population.
    2. Applies the quality gate (mask size, peak raw intensity) and an optional manual
       drop list. Both shrink the EVALUATION population, not just the support pool -- see
       "Reading the numbers" below.
    3. Draws K cells per barcode combination, without replacement, with a seeded RNG.
    4. Fine-tunes the two heads on them.
    5. Scores every remaining cell, gates each call on the LUMINA composite confidence
       score, and reports DETECTION and ACCURACY as two separate numbers.

    Step 5 never scores a cell from step 4. That is enforced, not assumed: giving the same
    dish to --data-root and --eval-root is switched to the within-batch path with a printed
    note, and every seed additionally asserts that the support crops and the scored crops
    are disjoint before any number is produced. A cell scored right after being trained on
    is classified almost perfectly and would inflate both rates with no visible symptom.

THE CLASS COUNT, AND WHAT HAPPENS IF YOUR PANEL IS SMALLER
    The heads are NOT resized. `num_classes` is read back out of the checkpoint
    (`fc_nu.6.weight`), the model is rebuilt with exactly that width, and the pre-trained
    final layer is fine-tuned in place. That is the whole reason a handful of cells is
    enough: you are re-weighting an already-trained classifier, not training a new one.

    Consequences you must accept:
      * Your barcode names must be the checkpoint's names (`N10`, `M13`, ...) so they map
        onto the SAME integer indices. A name this script does not know is an error, never
        a silent class 0.
      * A smaller panel is fine and needs no flag. Barcodes you did not image simply get
        no support cells; their output units stay live and are pushed down only indirectly,
        as the negative term of the cross-entropy. They remain reachable at test time,
        which is part of what the confidence gate absorbs.
      * A barcode the checkpoint never saw cannot be added. This recipe re-weights existing
        outputs; it does not grow new ones.

READING THE NUMBERS
    Two numbers, always together, never merged:
      DETECTION -- the fraction of evaluated cells whose nuclear AND mitochondrial call
                   both clear the confidence threshold.
      ACCURACY  -- among the detected cells only, the fraction where BOTH calls are right.
    Accuracy is conditional on detection, so raising the threshold buys accuracy by
    shrinking the denominator. Two settings may only be compared at matched detection.
    Both the quality gate and the drop list change the denominator as well, so every rate
    printed here is a rate on a curated population; the header records which curation ran.

Usage (one dish, adapt and read out the rest of the same dish):
    python Finetune_LUMINA.py --checkpoint best_model_fine-tune.pth \
        --data-root /path/to/dish1 --out ./adapt_out --k 20

Cross-batch (adapt on one dish, read out a different one):
    python Finetune_LUMINA.py --checkpoint best_model_fine-tune.pth \
        --data-root /path/to/dish1 --eval-root /path/to/dish2 \
        --out ./adapt_out --k 20 --seeds 0,1,2,3,4
"""

import argparse
import copy
import glob
import hashlib
import os
import re

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
import torch.nn as nn

# The model definition and the centre-padding rule are reused from Test_LUMINA.py rather
# than copied, so a checkpoint that loads there loads here. Test_LUMINA.py is safe to
# import: everything that touches a path or a device is inside its main(), behind the
# __main__ guard. Do NOT import Data_prep.py (it runs a loop and deletes folders at import
# time) or Visualize_heatmap.py (it switches matplotlib to TkAgg at import time).
try:
    from Test_LUMINA import DualHeadConvNet, pad_image
except ImportError as exc:
    raise SystemExit(
        'Could not import Test_LUMINA.py: %s\n'
        'Finetune_LUMINA.py must sit in the same folder as Test_LUMINA.py '
        '(LUMINA_classification/).' % exc
    )


# Class name -> class index. Written out explicitly here on purpose, and now the same
# literal in all four places: Train_LUMINA.py, Test_LUMINA.py and Visualize_heatmap.py each
# hold their own NU_CLASS_MAP / MITO_CLASS_MAP, character for character these ones. None of
# them derives the mapping any more. They used to: each built it from a folder-name dict
# whose entries were half commented out, so uncommenting one line there renumbered every
# class after it. Writing the map out is what removed that failure mode, not a style choice.
#
# The warning it was making still stands, because the reason it was silent has not changed:
# a checkpoint stores no mapping. Nothing writes these indices into a .pth and nothing
# checks them when one is loaded, so a checkpoint read back under a different order
# mislabels every cell and raises nothing. If you change the panel, change it in all four
# places at once, and do not point an existing checkpoint at the new order.
NU_CLASS_MAP = {'N10': 1, 'N13': 2, 'N4': 3, 'N14': 4, 'N16': 5, 'N8': 6, 'N1': 7}
MITO_CLASS_MAP = {'M10': 1, 'M13': 2, 'M4': 3, 'M14': 4, 'M16': 5, 'M8': 6, 'M1': 7}
NU_REV = {v: k for k, v in NU_CLASS_MAP.items()}
MITO_REV = {v: k for k, v in MITO_CLASS_MAP.items()}

CROP_SIZE = 256          # the canvas both loaders pad to
N_PLANES = 6             # [G, S, ratio1, ratio2, ratio3, intensity]
INTENSITY_PLANE = 5


# ----------------------------------------------------------------------------------
# confidence
# ----------------------------------------------------------------------------------

def calculate_confidence_score(predictions):
    """Composite confidence, transcribed from the calculate_confidence_score() nested
    inside Test_LUMINA.py's test_model().

    Kept bit-identical to that function on purpose, including the two quirks:
    `max_entropy` is normalised against SEVEN classes while the softmax has eight, so a
    near-uniform vector scores slightly negative; and the ratio term saturates at its 0.3
    ceiling once the top probability passes ~0.59. The thresholds in use are points on
    this exact scale -- "fixing" either quirk moves every detection rate ever reported.

    Returns the bare score. The caller applies the gate, because a cell counts as detected
    only when BOTH heads clear it -- Test_LUMINA.py's `if nu_reliable and mito_reliable:`,
    which is what sorts a cell into the confident workbook rather than the uncertain one.

    float32, not float64, and deliberately so: Test_LUMINA.py computes this on
    `predictions.cpu().numpy()` of a float32 softmax, so it accumulates the entropy sum in
    float32. Promoting to float64 here would move the score in the last few decimals of
    every vector, and the thresholds in use are points on this exact scale -- the two
    scripts have to agree bit for bit or the same cell can be detected by one and not the
    other.
    """
    pred_np = np.asarray(predictions, dtype=np.float32)

    sorted_probs = np.sort(pred_np)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]

    entropy = -np.sum(pred_np * np.log(pred_np + 1e-10))
    max_entropy = -np.log(1 / 7)  # Maximum possible entropy for 7 classes
    normalized_entropy = 1 - (entropy / max_entropy)

    max_prob = sorted_probs[0]
    mean_others = np.mean(sorted_probs[1:])
    ratio = max_prob / (mean_others + 1e-10)

    return float(0.4 * margin + 0.3 * normalized_entropy + 0.3 * min(ratio / 10, 1))


# ----------------------------------------------------------------------------------
# input normalisation
# ----------------------------------------------------------------------------------

def simplex_project(spec):
    """Clip the three spectral ratios at zero and, where they sum above one, scale down."""
    spec = torch.clamp(spec, min=0.0)
    s = spec.sum(1, keepdim=True)
    return torch.where(s > 1.0, spec / (s + 1e-6), spec)


def normalize_six_plane(img6):
    """Normalise a batch of padded six-plane crops, (B, 6, H, W) float.

    Five steps, in this order:
      1. mask = intensity > 0
      2. clamp G and S into [0, 1]
      3. project the three ratio planes into the 3-simplex
      4. zero all five feature planes outside the mask
      5. divide the intensity plane by its per-cell maximum, floored at 1e-6

    This is NOT `Test_LUMINA.normalize_intensity`, which performs step 5 only (and without
    the zero guard, so an all-zero crop yields NaN there). The checkpoint was adapted under
    the five-step version; feeding it the one-step version changes the input distribution
    and quietly costs accuracy. If you evaluate the same cells with Test_LUMINA.py you are
    not feeding the network the same thing -- that is expected, and it is why this script
    reports its own numbers rather than deferring to that one.
    """
    intensity = img6[:, INTENSITY_PLANE:INTENSITY_PLANE + 1]
    mask = (intensity > 0).float()
    gs = torch.clamp(img6[:, 0:2], 0.0, 1.0)
    spec = simplex_project(img6[:, 2:5])
    feats = torch.cat([gs, spec], 1) * mask
    mx = intensity.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    return torch.cat([feats, intensity / mx], 1)


# ----------------------------------------------------------------------------------
# population: finding the cells and their labels
# ----------------------------------------------------------------------------------

def natural_key(text):
    """Sort key that orders cell10 after cell2 rather than before it.

    Row order is part of what a seed means: `sample_support` draws from positional indices,
    so re-ordering the rows within a combination re-labels which cells a given --seed picks.
    Plain lexicographic sorting would order cell1, cell10, cell11, cell2 ..., which is a
    different draw from the same seed. Every enumeration of samples and crops in this file
    goes through here so that a seed is reproducible against the same folder. (The order of
    the COMBINATIONS is not part of it -- see `sample_support`, which seeds each combination
    separately -- but the order of the rows inside one still is.)
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(text))]


def parse_cell_id(path):
    m = re.search(r'cell(\d+)_5D', os.path.basename(str(path)))
    return int(m.group(1)) if m else -1


def expected_from_sample(sample):
    """Folder-name convention: NT<mito>-<nu>, e.g. NT8-10 -> ('N10', 'M8')."""
    m = re.search(r'NT(\d+)-(\d+)', str(sample))
    if not m:
        return '', ''
    return 'N%s' % m.group(2), 'M%s' % m.group(1)


SEG_FOLDER_ORDER = {
    'auto': ('seg_5D_calib', 'seg_5D'),
    'seg_5D': ('seg_5D',),
    'seg_5D_calib': ('seg_5D_calib',),
}


def find_seg_folder(sample_dir, preference='auto'):
    """Locate the folder of prepared crops under one sample folder.

    "auto" tries seg_5D_calib and falls back to seg_5D, which is the rule encoded in
    Test_LUMINA.py's SEG_FOLDER_ORDER and therefore the rest of this repository's
    behaviour. The two are NOT interchangeable: seg_5D_calib holds crops that have been
    spectrally calibrated and seg_5D holds the uncalibrated ones, so on a dish that has
    both, "auto" feeds the network a different input distribution than seg_5D would. The
    measured adaptation runs read seg_5D only. --seg-folder pins the choice; the chosen
    folder is printed for every sample so a run can never be ambiguous about which one it
    read.
    """
    for name in SEG_FOLDER_ORDER[preference]:
        p = os.path.join(sample_dir, name)
        if os.path.isdir(p):
            return p
    return None


WORKBOOK_COLUMNS = ('Cell_Label', 'Nu_FP', 'Mito_FP')


def read_workbook_labels(sample_dir, strict):
    """Read `clustered.xlsx` -> ({Cell_Label: (Nu_FP, Mito_FP)}, note), or (None, reason).

    Two different files are called clustered.xlsx. This script wants the dual-anchor
    workbook, whose columns are Cell_Label / Nu_FP / Mito_FP. The napari plugin shipped in
    this same repository writes its own clustered.xlsx into sample folders, with columns
    Mask label / FOV / Localization / cluster_local / cluster_tag -- a perfectly good file
    that simply does not carry dual-anchor labels.

    `strict` is True only when the user asked for --labels workbook by name. Then any
    problem is fatal, because they said which file to read. Under --labels auto a wrong
    schema, an unreadable file or a missing openpyxl is a reason to fall back to the folder
    name, not a reason to stop: refusing to run because a folder happens to contain the
    plugin's output would make the default flag unusable on this repository's own data.
    """
    path = os.path.join(sample_dir, 'clustered.xlsx')
    if not os.path.exists(path):
        if strict:
            raise SystemExit('--labels workbook was asked for but %s has no clustered.xlsx'
                             % sample_dir)
        return None, 'no clustered.xlsx'
    try:
        df = pd.read_excel(path)
    except ImportError as exc:
        if strict:
            raise SystemExit(
                'Reading %s needs the openpyxl package, which pandas uses for .xlsx and '
                'does not install itself.\n'
                'It is listed in requirements.txt; this environment does not have it. '
                'Either "pip install openpyxl", or pass --labels foldername (folders named '
                'NT<mito>-<nu>), or pass --manifest with a CSV of labels.' % path)
        return None, 'openpyxl not installed (%s)' % exc
    except Exception as exc:
        if strict:
            raise SystemExit('Could not read %s: %s' % (path, exc))
        return None, 'unreadable (%s)' % exc
    missing = [c for c in WORKBOOK_COLUMNS if c not in df.columns]
    if missing:
        detail = ('%s has no "%s" column. This script wants the dual-anchor workbook with '
                  'Cell_Label / Nu_FP / Mito_FP; this file has %s -- most likely the napari '
                  'plugin\'s clustered.xlsx, which is a different file with the same name.'
                  % (path, missing[0], ', '.join(str(c) for c in df.columns[:8])))
        if strict:
            raise SystemExit(detail + '\nPass --labels foldername or --labels auto to use '
                                      'the folder name instead.')
        return None, 'not the dual-anchor workbook (no %s column)' % missing[0]
    out = {}
    for _, row in df.iterrows():
        if pd.isna(row['Cell_Label']):
            continue
        nu = row['Nu_FP'] if pd.notna(row['Nu_FP']) else ''
        mito = row['Mito_FP'] if pd.notna(row['Mito_FP']) else ''
        out[int(row['Cell_Label'])] = (str(nu).strip(), str(mito).strip())
    return out, 'clustered.xlsx (%d labelled cells)' % len(out)


def scan_population(root, samples, labels_mode, tag, seg_pref='auto'):
    """Enumerate <root>/<sample>/<seg folder>/cell*_5D.tif into a DataFrame.

    Columns: batch, sample, cell_global, path, expected_nu, expected_mito.
    Prints, per sample folder, which seg folder was read and where the labels came from.
    """
    if not os.path.isdir(root):
        raise SystemExit('--data-root/--eval-root does not exist or is not a folder: %s' % root)

    if samples:
        sample_names = [s.strip() for s in samples.split(',') if s.strip()]
        missing = [s for s in sample_names if not os.path.isdir(os.path.join(root, s))]
        if missing:
            raise SystemExit('These --samples are not folders under %s: %s' % (root, ', '.join(missing)))
    else:
        sample_names = sorted((d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))), key=natural_key)

    rows = []
    scanned = 0
    for sample in sample_names:
        sample_dir = os.path.join(root, sample)
        seg_dir = find_seg_folder(sample_dir, seg_pref)
        if seg_dir is None:
            continue
        scanned += 1
        crops = sorted(glob.glob(os.path.join(seg_dir, 'cell*_5D.tif')),
                       key=lambda p: natural_key(os.path.basename(p)))
        if not crops:
            print('  warning: %s has a %s folder but no cell*_5D.tif in it'
                  % (sample, os.path.basename(seg_dir)))
            continue

        workbook, note = None, 'folder name'
        if labels_mode in ('auto', 'workbook'):
            workbook, note = read_workbook_labels(sample_dir, strict=(labels_mode == 'workbook'))
            if workbook is None:
                note = 'folder name (%s)' % note
        folder_nu, folder_mito = expected_from_sample(sample)
        if workbook is None and not (folder_nu and folder_mito):
            note += ' -> UNPARSEABLE, expected NT<mito>-<nu>'
        print('  [%s] seg folder: %-13s labels: %s'
              % (sample, os.path.basename(seg_dir), note))

        for path in crops:
            cell = parse_cell_id(path)
            if workbook is not None and cell in workbook:
                nu, mito = workbook[cell]
            else:
                nu, mito = folder_nu, folder_mito
            rows.append({'batch': tag, 'sample': sample, 'cell_global': cell, 'path': path,
                         'expected_nu': nu, 'expected_mito': mito})

    if scanned == 0:
        raise SystemExit(
            'No sample folder under %s contains a %s subfolder (--seg-folder %s).\n'
            'Expected layout: <root>/<sample>/seg_5D/cell<id>_5D.tif  (as written by '
            'Data_prep.py).' % (root, ' or '.join(SEG_FOLDER_ORDER[seg_pref]), seg_pref)
        )
    if not rows:
        raise SystemExit('Found sample folders under %s but no cell*_5D.tif crops in them.' % root)
    return pd.DataFrame(rows)


def load_manifest(path, tag):
    """Explicit manifest CSV: path, sample, cell_global, expected_nu, expected_mito.
    A `batch` column is used if present, otherwise every row is tagged with `tag`."""
    if not os.path.exists(path):
        raise SystemExit('--manifest does not exist: %s' % path)
    df = pd.read_csv(path)
    required = ['path', 'sample', 'expected_nu', 'expected_mito']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit('--manifest %s is missing column(s): %s\n'
                         'Required: path, sample, expected_nu, expected_mito. '
                         'Optional: batch, cell_global.' % (path, ', '.join(missing)))
    df = df.copy()
    if 'batch' not in df.columns:
        df['batch'] = tag
    df['batch'] = df['batch'].astype(str)
    if 'cell_global' not in df.columns:
        df['cell_global'] = [parse_cell_id(p) for p in df['path']]
    df['expected_nu'] = df['expected_nu'].fillna('').astype(str).str.strip()
    df['expected_mito'] = df['expected_mito'].fillna('').astype(str).str.strip()
    bad = [p for p in df['path'] if not os.path.exists(p)]
    if bad:
        raise SystemExit('--manifest lists %d file(s) that do not exist, first: %s'
                         % (len(bad), bad[0]))
    return df.reset_index(drop=True)


def validate_labels(df, source):
    """Every label must map onto a pre-trained class index. Never default to class 0.

    Class 0 is the "no anchor" output. A folder name or workbook entry that does not parse
    would otherwise fall through to it, which produces a mislabelled training cell instead
    of an error: the run completes, the support set is quietly wrong, and the only symptom
    is that adaptation helps less than it should. Failing here is the point of the function.
    """
    bad_nu = sorted(set(df.loc[~df['expected_nu'].isin(set(NU_CLASS_MAP)), 'expected_nu']))
    bad_mito = sorted(set(df.loc[~df['expected_mito'].isin(set(MITO_CLASS_MAP)), 'expected_mito']))
    if bad_nu or bad_mito:
        raise SystemExit(
            'Unrecognised barcode label(s) in %s.\n'
            '  nuclear   : %s\n'
            '  mitochondrial: %s\n'
            'Known nuclear classes  : %s\n'
            'Known mitochondrial    : %s\n'
            'The checkpoint stores no class map, so a name it does not know cannot be '
            'adapted -- it would have to be trained from scratch. Fix the labels, or drop '
            'those cells from the manifest.'
            % (source, bad_nu or '(none)', bad_mito or '(none)',
               ', '.join(NU_CLASS_MAP), ', '.join(MITO_CLASS_MAP))
        )


def batch_key_forms(value):
    """Every spelling of a batch identifier, in two tiers plus its full dates.

    The same batch gets written down three ways across a project: as the folder name
    ("dual_20260101_2"), as the acquisition date ("20260101") and as the month-day tail
    ("0101"). The tail is the dangerous one, in BOTH directions, and the two dangers pull
    against each other:

      * A curation CSV with a zero-padded `batch` column is parsed by pandas as int64, so
        `astype(str)` hands back "101"; a string comparison against "0101" matches nothing
        at all, and the drop list appears to load, reports its entry count, and removes
        zero cells. Generating the padded and unpadded spellings on both sides stops that.
      * But emitting the MMDD tail as a PEER of the full date makes "20250101" match a dish
        named "dual_20260101_2" -- a drop list from a different year silently removing
        cells from the support pool and the evaluation set. The year is the whole point of
        writing a full date down.

    So MMDD is a FALLBACK, not a peer, and the tiers say which is which:
        strict -- the literal, its digit variants, and every embedded 8-digit date.
        loose  -- strict, plus each embedded date's MMDD tail, padded and unpadded.
        dates  -- the embedded 8-digit runs on their own.

    Use `batch_matches`, not these sets directly: the rule is a pairwise one and cannot be
    expressed by unioning form sets across many identifiers and intersecting once.
    """
    s = str(value).strip()
    if not s or s.lower() in ('nan', 'none', '<na>'):
        return frozenset(), frozenset(), frozenset()
    # A third spelling, and the nastiest: pandas types an integer column as float64 as soon
    # as ONE cell is blank, so a zero-padded "0101" comes back as "101.0". isdigit() rejects
    # that, no padded variant is generated, and the entry matches nothing -- while the blank
    # rows still act as wildcards and remove cells, so n_dropped > 0 and the zero-removals
    # diagnostic never fires. The run then scores a population that still holds every cell
    # the curator marked bad. Normalise it back to the integer spelling before anything else
    # looks at it.
    m_float = re.fullmatch(r'(\d+)\.0+', s)
    if m_float:
        s = m_float.group(1)
    strict = {s.lower()}
    if s.isdigit():
        strict.add(s.lstrip('0') or '0')
        if len(s) <= 4:
            strict.add(s.zfill(4))
    dates = set()
    for run in re.findall(r'\d+', s):
        if len(run) >= 8:                 # an embedded YYYYMMDD
            dates.add(run)
            strict.add(run)
    loose = set(strict)
    for run in dates:
        tail = run[4:8]                   # ... and its MMDD tail, fallback tier only
        loose.add(tail)
        loose.add(tail.lstrip('0') or '0')
    return frozenset(strict), frozenset(loose), frozenset(dates)


def batch_is_wildcard(value):
    """True when a drop-list entry carries no usable batch identifier at all.

    An entry without a `batch` column, or with a blank/NaN one, means "every batch". That
    is only safe when cell ids are globally unique, which is the caller's problem, but it
    can never be the REASON a list matched nothing -- so the zero-removals diagnostic has
    to be able to tell a wildcard from a batch that simply did not match.
    """
    return not batch_key_forms(value)[0]


def batch_matches(a, b):
    """True when two batch identifiers name the same batch.

    The rule, and the only rule:
      * If BOTH sides carry a full 8-digit date, those dates must be equal. A full date is
        an unambiguous statement of which batch is meant, and two of them that disagree
        disagree -- no amount of shared MMDD tail makes 20250101 and 20260101 the same
        dish.
      * Otherwise fall back to the loose forms, which is where the MMDD tail lives. That
        is the case the padding fix exists for: a bare `0101`-style column (or the `101`
        pandas hands back for it) against a dish folder carrying a full date.

    Symmetric in its arguments on purpose -- neither side is privileged, and a drop list
    written either way behaves the same.
    """
    a_strict, a_loose, a_dates = batch_key_forms(a)
    b_strict, b_loose, b_dates = batch_key_forms(b)
    if not a_strict or not b_strict:
        return False                      # an absent identifier is the caller's wildcard
    if a_dates and b_dates:
        return bool(a_dates & b_dates)
    return bool(a_loose & b_loose)


def read_drop_list(path):
    """Manual curation list: batch, sample, cell_global.

    Returns {(sample, cell_global): [raw batch value, ...]}. The RAW value is kept, not a
    pre-expanded form set, because matching is pairwise (see `batch_matches`): whether the
    MMDD tail may be used depends on what the OTHER side looks like, so a form set expanded
    without knowing the other side cannot answer the question. A `batch` column is optional;
    an entry without one carries None, meaning "every batch", which is only safe if cell ids
    are globally unique.
    """
    if not os.path.exists(path):
        raise SystemExit('--drop-list does not exist: %s' % path)
    df = pd.read_csv(path)
    for col in ('sample', 'cell_global'):
        if col not in df.columns:
            raise SystemExit('--drop-list %s is missing the "%s" column. Expected columns: '
                             'batch, sample, cell_global.' % (path, col))
    batches = (df['batch'] if 'batch' in df.columns else pd.Series([None] * len(df)))
    entries = {}
    for b, s, c in zip(batches, df['sample'].astype(str).str.strip(),
                       df['cell_global'].astype(int)):
        entries.setdefault((s, int(c)), []).append(b)
    return entries


def apply_drop_list(df, Z, drop_entries, label):
    """Remove manually curated cells from BOTH the table and the cached feature tensor.

    Returns (df, Z, removed). The tensor must be subset here, in the same call, on the same
    boolean mask: `Z` is addressed by positional index everywhere downstream, so a df that
    has been shrunk while Z has not silently pairs row i with a different cell's features
    from the first dropped row onward. `assert_aligned` at the end is the guard that turns
    any future version of that mistake into a crash instead of a wrong number.
    """
    if not drop_entries:
        return df, Z, 0
    keep = []
    for b, s, c in zip(df['batch'].astype(str), df['sample'].astype(str),
                       df['cell_global'].astype(int)):
        hits = drop_entries.get((s, int(c)))
        if not hits:
            keep.append(True)
            continue
        keep.append(not any(batch_is_wildcard(want) or batch_matches(want, b)
                            for want in hits))
    keep = np.array(keep, bool)
    removed = int((~keep).sum())
    print('[drop-list] %s: removed %d manually curated cells, kept %d'
          % (label, removed, int(keep.sum())))
    df = df[keep].reset_index(drop=True)
    if Z is not None:
        Z = Z[torch.from_numpy(np.where(keep)[0]).to(Z.device)]
    assert_aligned(df, Z, 'drop list (%s)' % label)
    return df, Z, removed


def same_location(a, b):
    """True when two path strings name the same file or folder on this filesystem.

    `normcase` is what makes this both correct and portable: on Windows it lower-cases and
    flips separators, so `D:\\dish` and `d:/DISH/` compare equal, while on POSIX it is the
    identity and two genuinely different case-sensitive paths stay different. `realpath`
    resolves a junction, a symlink or a `..`, which is the usual way one dish arrives under
    two spellings.
    """
    if not a or not b:
        return False
    return (os.path.normcase(os.path.realpath(os.path.abspath(a))) ==
            os.path.normcase(os.path.realpath(os.path.abspath(b))))


def cell_identities(df, positions, _dir_cache=None):
    """Identity of each named cell, as the crop file it was read from.

    The file path, not `(batch, sample, cell_global)`: `batch` is a folder basename that two
    dishes can share, and `cell_global` restarts at 1 in every sample folder, so the triple
    is unique only by luck -- on the measured layout EVERY dish has an `NT8-10/cell1`, so
    the triple would flag every legitimate cross-batch run.

    Two tokens per cell, and a collision on EITHER is a collision. `(st_dev, st_ino)`
    catches a hard link or a symlink to the file. The normalised `realpath` of the file --
    exactly as `same_location` normalises -- catches everything reached through two
    spellings of its root: a junction, a symlink, a `..`, a different case. `realpath` is
    resolved per DIRECTORY and memoised, since it is a filesystem call and a dish has a
    handful of directories and thousands of crops.

    Returns `(absolute_path, token_set)` per cell, so a caller that finds a collision can
    report the path a human will recognise rather than an inode number.
    """
    if _dir_cache is None:
        _dir_cache = {}
    paths = df['path'].values
    out = []
    for p in positions:
        raw = os.path.abspath(str(paths[p]))
        tokens = set()
        # (device, inode) is the only identity that survives a per-FILE link. `realpath` on
        # the directory resolves a junction or a symlinked ROOT, but a hard link -- or a
        # symlink to the file itself -- gives the same bytes a different path, and an
        # evaluation dish built with `cp -al` or `mklink /H` would otherwise pass the
        # disjointness check while scoring the very cells the heads were trained on.
        # Windows populates st_ino on NTFS; where a filesystem does not, it reports 0 and
        # the path token below carries the check on its own.
        try:
            st = os.stat(raw)
            if st.st_ino:
                tokens.add('inode:%d:%d' % (st.st_dev, st.st_ino))
        except OSError:
            pass
        head, tail = os.path.split(raw)
        real = _dir_cache.get(head)
        if real is None:
            real = _dir_cache[head] = os.path.realpath(head)
        tokens.add('path:' + os.path.normcase(os.path.join(real, tail)))
        out.append((raw, frozenset(tokens)))
    return out


def assert_support_held_out(sup_df, support, eval_df, held_pos, mode, seed):
    """The cells trained on and the cells scored must be disjoint. No exceptions.

    This decides nothing -- `mode` already chose which cells to hold out -- and only checks
    the result, because getting it wrong is invisible in the numbers. A support cell scored
    as held-out is a cell the heads were just fine-tuned on; it is classified almost
    perfectly, it inflates both detection and accuracy, and the run completes with a
    better-looking result and no warning at all. There is no downstream symptom to notice.

    It catches what the --data-root/--eval-root path comparison cannot see: an
    --eval-manifest listing the support dish's crops, an --eval-root that reaches the same
    crops through a different folder tree, and any future edit to the held-out mask. The
    path comparison is the convenience; this is the guarantee.

    Links are caught too, by inode: an evaluation dish built with `cp -al` or `mklink /H`
    is the same files under new names, and the paths alone would not show it.

    What it does NOT catch: a byte-for-byte COPY of the support dish at another path. Those
    are genuinely different files, and telling a copy from a second dish that happens to
    look similar would mean hashing every crop. `--eval-root` and
    `--data-root` are then both printed and both land in finetune_run_config.csv, so the
    duplication is visible in the record even though it is not refused.
    """
    dir_cache = {}
    sup_ids = cell_identities(sup_df, support, dir_cache)
    eval_ids = cell_identities(eval_df, held_pos, dir_cache)
    eval_tokens = set()
    for _, toks in eval_ids:
        eval_tokens |= toks
    leaked = sorted(raw for raw, toks in sup_ids if toks & eval_tokens)
    if not leaked:
        return
    raise SystemExit(
        'Support cells leaked into the evaluation set: %d of the %d cells the heads were '
        'fine-tuned on (seed %d) are also among the %d cells being scored.\n'
        'Every rate from such a run is inflated -- a cell just trained on is classified '
        'almost perfectly -- and nothing downstream would show it, so this is fatal.\n'
        '  mode as run: %s\n'
        '  first leaked crop(s):\n    %s\n'
        'The usual cause is an evaluation population that is the support population under '
        'another name: an --eval-manifest listing crops from the support dish, or an '
        '--eval-root reaching the same crops through a different folder tree. (--eval-root '
        'naming the same dish outright is detected earlier and switched to the within-batch '
        'path, so it does not reach here.) Either drop the evaluation flags to run '
        'within-batch, or point them at a genuinely different dish.'
        % (len(leaked), len(support), seed, len(held_pos), mode,
           '\n    '.join(leaked[:5]) + ('\n    ...' if len(leaked) > 5 else '')))


def assert_aligned(df, Z, where):
    """Row i of `df` must describe the cell whose features are row i of `Z`.

    Every step that filters the population has to shrink both together. Nothing downstream
    can detect a mismatch -- positional indices into a stale tensor are perfectly valid
    indices, they just point at the wrong cell -- so the invariant is checked explicitly
    after each filter rather than left to be noticed in the numbers.
    """
    if Z is not None and len(df) != len(Z):
        raise SystemExit(
            'Internal error: after %s the population has %d rows but the cached feature '
            'tensor has %d. A filtering step shrank one and not the other; every row after '
            'the first removed cell would be scored against a different cell\'s features.'
            % (where, len(df), len(Z)))


# ----------------------------------------------------------------------------------
# crops
# ----------------------------------------------------------------------------------

def load_crop(path, oversize):
    """Read one cell crop and centre-pad it to 256x256 float32, or return None if it is
    oversized and the policy is to skip.

    Neither of the obvious behaviours is safe by default. Test_LUMINA.py's dataset advances
    to the NEXT row and returns that image instead, which silently shifts every downstream
    Cell_Label. Downscaling to fit keeps the cell but changes its scale. This script does
    one or the other, explicitly, on a flag, and counts what it did -- a few-shot run has to
    know exactly which cells are in its support set. The measured adaptation runs
    downscaled, which is why that is the default.
    """
    img = tiff.imread(path)
    if img.ndim != 3 or img.shape[0] != N_PLANES:
        raise SystemExit(
            '%s has shape %s. A prepared LUMINA crop is (6, h, w): '
            '[G, S, ratio1, ratio2, ratio3, intensity]. Re-run Data_prep.py.'
            % (path, tuple(img.shape))
        )
    img = np.nan_to_num(np.asarray(img, dtype=np.float32))
    h, w = img.shape[1], img.shape[2]
    if h > CROP_SIZE or w > CROP_SIZE:
        if oversize == 'skip':
            return None
        scale = min(CROP_SIZE / h, CROP_SIZE / w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        img = np.stack([cv2.resize(img[i], (nw, nh), interpolation=cv2.INTER_LINEAR)
                        for i in range(img.shape[0])])
    return pad_image(img, CROP_SIZE, CROP_SIZE)


def load_crops(df, positions, oversize):
    """Load a fixed set of rows into one (n, 6, 256, 256) float32 array."""
    stack = []
    for pos in positions:
        img = load_crop(df['path'].values[pos], oversize)
        if img is None:
            raise SystemExit('Crop %s became oversized between passes -- this should not '
                             'happen.' % df['path'].values[pos])
        stack.append(img)
    return np.stack(stack)


def survey_population(model, df, device, args, want_features, label):
    """One streaming pass over every crop.

    Returns (df_kept, features_or_None, px, mx):
      px  -- non-zero pixel count of the padded RAW intensity plane (= segmented cell area)
      mx  -- per-cell maximum of the RAW intensity plane, BEFORE any normalisation
      features -- the frozen 512-d backbone output, if want_features

    Features are extracted once, under model.eval(), so BatchNorm keeps the source-domain
    running statistics. That is a real difference from --ft-mode full, which runs the model
    in train() mode and therefore adapts BatchNorm as well; the trade-off is written up in
    `.claude/skills/lumina-network/references/domain-adaptation.md`, section "Two traps".
    """
    keep_pos, feats, px_all, mx_all = [], [], [], []
    skipped = 0
    n = len(df)
    bs = args.load_batch_size

    model.eval()
    with torch.no_grad():
        for start in range(0, n, bs):
            chunk, chunk_pos = [], []
            for pos in range(start, min(start + bs, n)):
                img = load_crop(df['path'].values[pos], args.oversize)
                if img is None:
                    skipped += 1
                    continue
                chunk.append(img)
                chunk_pos.append(pos)
            if not chunk:
                continue
            raw = np.stack(chunk)
            flat = raw[:, INTENSITY_PLANE].reshape(len(raw), -1)
            px_all.append((flat > 0).sum(1))
            mx_all.append(flat.max(1))
            keep_pos.extend(chunk_pos)
            if want_features:
                img6 = normalize_six_plane(torch.from_numpy(raw).float().to(device))
                x = torch.cat([h(img6[:, i:i + 1]) for i, h in enumerate(model.input_heads)], 1)
                feats.append(model.backbone(x).view(x.size(0), -1))
            if start // bs % 20 == 0:
                print('  %s: %d / %d crops read' % (label, min(start + bs, n), n))

    if skipped:
        print('[oversize] %s: dropped %d crop(s) larger than %dx%d (--oversize skip)'
              % (label, skipped, CROP_SIZE, CROP_SIZE))
    if not keep_pos:
        raise SystemExit('%s: every crop was dropped. With --oversize skip that means all '
                         'crops exceed %dx%d; try --oversize downscale.'
                         % (label, CROP_SIZE, CROP_SIZE))

    df_kept = df.iloc[keep_pos].reset_index(drop=True)
    Z = torch.cat(feats) if want_features else None
    assert_aligned(df_kept, Z, 'the survey pass (%s)' % label)
    return df_kept, Z, np.concatenate(px_all), np.concatenate(mx_all)


def apply_quality_gate(df, Z, px, mx, args, label):
    """Automatic cell-quality filter, on the RAW intensity plane.

    min_px  -- segmented area in pixels; drops fragmentary masks.
    min_max -- peak raw intensity, in the units Data_prep.py wrote (photon sums); drops
               dim, effectively non-expressing cells.

    This runs BEFORE support sampling and BEFORE evaluation, so it defines the denominator
    of every rate this script prints, not just which cells may be labelled.
    """
    if args.min_px <= 0 and args.min_max <= 0:
        return df, Z
    keep = (px >= args.min_px) & (mx >= args.min_max)
    kept = int(keep.sum())
    print('[quality gate] %s: kept %d / %d cells (--min-px %d --min-max %g)'
          % (label, kept, len(keep), args.min_px, args.min_max))
    if kept == 0:
        raise SystemExit(
            '%s: the quality gate removed every cell.\n'
            'Peak raw intensity over this population was %.4g, and --min-max is %g. '
            '--min-max is measured on the RAW intensity plane, in the units Data_prep.py '
            'wrote. If your crops were already normalised into [0, 1], pass --min-max 0.'
            % (label, float(mx.max()), args.min_max)
        )
    df = df[keep].reset_index(drop=True)
    if Z is not None:
        Z = Z[torch.from_numpy(np.where(keep)[0]).to(Z.device)]
    assert_aligned(df, Z, 'the quality gate (%s)' % label)
    return df, Z


# ----------------------------------------------------------------------------------
# support sampling and fine-tuning
# ----------------------------------------------------------------------------------

def combination_keys(df):
    """The stratification key: the barcode PAIR a cell carries, "<nu>+<mito>".

    Not the sample folder. On the layout the protocol was measured with, one folder holds
    exactly one combination and the two are the same partition -- but --labels workbook and
    --manifest both allow one folder to hold many combinations, and stratifying on the
    folder there would draw K cells for the whole folder and split them across its
    combinations instead of drawing K for each. K means K per combination everywhere in
    this script and in the documentation, so the key has to be the combination itself.
    """
    return np.array(['%s+%s' % (n, m)
                     for n, m in zip(df['expected_nu'].astype(str),
                                     df['expected_mito'].astype(str))], dtype=object)


def stable_group_hash(name):
    """A deterministic 64-bit integer from a group name, stable ACROSS PROCESSES.

    Python's built-in `hash()` of a str is salted by PYTHONHASHSEED and therefore differs
    between interpreter runs, which would make `--seed 0` mean a different support set on
    every invocation. BLAKE2b has no such salt.
    """
    return int.from_bytes(hashlib.blake2b(str(name).encode('utf-8'),
                                          digest_size=8).digest(), 'big')


def sample_support(df, groups, k, seed):
    """K cells per barcode COMBINATION, without replacement, each group drawn from its OWN
    RNG derived from (seed, group name).

    Two properties of the protocol this deliberately keeps:
      * Stratification is by COMBINATION, not by class. A nuclear barcode that appears in
        three combinations therefore contributes 3K support cells and one that appears in
        a single combination contributes K. Per-class support is imbalanced by construction
        and no class weighting compensates for it.
      * A scarce combination contributes everything it has, min(K, available).

    WHY ONE RNG PER GROUP RATHER THAN ONE STREAM OVER ALL OF THEM
        A single `default_rng(seed)` consumed group by group makes the draw depend on the
        ORDER the groups are visited in and on how many groups precede each one, because
        every group advances the shared stream for the ones after it. Then adding a dish
        that carries one extra combination, or renaming an unrelated combination so it
        sorts elsewhere, silently changes which cells every later group labels. Seeding
        each group separately from (seed, that group's name) removes that coupling: what a
        group draws depends only on the seed, its own name, and the order of its own rows.

        The ROW order within a group is still part of what a seed means -- the draw is over
        positional indices -- which is why every enumeration in this file sorts through
        `natural_key`. That order is printed at startup.

    A K-sweep at one seed is NOT nested: the RNG restarts for each K, so K=3 and K=5 draw
    different cells rather than "the same three plus two". Average over seeds.

    Returns the picks in group-concatenation order, NOT sorted. Sorting would be a
    cosmetic change with a numerical consequence: the support rows are handed to the
    fine-tuner in this order, so it sets the mini-batch composition, and a fine-tune on the
    same cells in a different order lands on a different head.
    """
    keys = combination_keys(df)
    picks = []
    for g in groups:
        ci = np.where(keys == g)[0]
        if len(ci) == 0:
            continue
        if len(ci) < k:
            print('  warning: combination %s has only %d cell(s), fewer than K=%d'
                  % (g, len(ci), k))
        rng = np.random.default_rng([int(seed), stable_group_hash(g)])
        picks += list(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.array(picks, dtype=int)


def train_heads_cached(model, Zsup, yn, ym, device, args, lr):
    """Heads-only fine-tune on cached backbone features.

    The backbone is frozen, so its 512-d output was computed once, under eval(), and only
    `fc_nu` and `fc_mito` are trained. Freezing is structural here -- the optimizer never
    sees anything else -- which also means BatchNorm keeps its source-domain running
    statistics. The heads themselves are in train() mode, so the Dropout(0.5) inside each
    of them is active during fine-tuning; that is part of the recipe, not an oversight.
    """
    fn = copy.deepcopy(model.fc_nu).to(device)
    fm = copy.deepcopy(model.fc_mito).to(device)
    opt = torch.optim.Adam(list(fn.parameters()) + list(fm.parameters()),
                           lr=lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    ynt = torch.from_numpy(yn).long().to(device)
    ymt = torch.from_numpy(ym).long().to(device)
    fn.train()
    fm.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(Zsup), device=Zsup.device)
        for s in range(0, len(Zsup), args.batch_size):
            idx = perm[s:s + args.batch_size]
            loss = ce(fn(Zsup[idx]), ynt[idx]) + ce(fm(Zsup[idx]), ymt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    fn.eval()
    fm.eval()
    return fn, fm


def finetune_full(model, xsup, yn, ym, device, args, lr):
    """Comparison arm: fine-tune every parameter, at the lower learning rate.

    The whole model goes to train() here, so BatchNorm normalises with the support batch's
    own statistics and updates its running estimates -- the target-domain adaptation comes
    partly from the heads and partly from that. This arm exists for comparison; the shipped
    recipe is --ft-mode heads.
    """
    m = copy.deepcopy(model).to(device)
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()
    xt = torch.from_numpy(xsup).float()
    ynt = torch.from_numpy(yn).long().to(device)
    ymt = torch.from_numpy(ym).long().to(device)
    m.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), args.batch_size):
            idx = perm[s:s + args.batch_size]
            img = normalize_six_plane(xt[idx].to(device))
            ln, lm = m(img)
            loss = ce(ln, ynt[idx]) + ce(lm, ymt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


# ----------------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------------

def predict_from_features(fn, fm, Z, batch_size):
    pn, pm, cn, cm = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(Z), batch_size):
            z = Z[s:s + batch_size]
            probs_nu = torch.softmax(fn(z), 1).cpu().numpy()
            probs_mito = torch.softmax(fm(z), 1).cpu().numpy()
            pn += [NU_REV.get(int(np.argmax(p)), 'Unknown') for p in probs_nu]
            pm += [MITO_REV.get(int(np.argmax(p)), 'Unknown') for p in probs_mito]
            cn += [calculate_confidence_score(p) for p in probs_nu]
            cm += [calculate_confidence_score(p) for p in probs_mito]
    return pn, pm, np.array(cn), np.array(cm)


def predict_from_images(m, df, positions, device, args):
    pn, pm, cn, cm = [], [], [], []
    m.eval()
    with torch.no_grad():
        for s in range(0, len(positions), args.infer_batch_size):
            block = positions[s:s + args.infer_batch_size]
            raw = load_crops(df, block, args.oversize)
            img = normalize_six_plane(torch.from_numpy(raw).float().to(device))
            ln, lm = m(img)
            probs_nu = torch.softmax(ln, 1).cpu().numpy()
            probs_mito = torch.softmax(lm, 1).cpu().numpy()
            pn += [NU_REV.get(int(np.argmax(p)), 'Unknown') for p in probs_nu]
            pm += [MITO_REV.get(int(np.argmax(p)), 'Unknown') for p in probs_mito]
            cn += [calculate_confidence_score(p) for p in probs_nu]
            cm += [calculate_confidence_score(p) for p in probs_mito]
    return pn, pm, np.array(cn), np.array(cm)


# ----------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------

def build_rows(df, combos, positions, pn, pm, cn, cm, threshold, common):
    """One row per evaluated cell. Detection needs BOTH heads over the threshold.

    `combination` and `sample` are both recorded because they are not the same thing: the
    combination is the barcode pair (the unit K is counted in, and the unit the summary
    table groups by), the sample is the folder the crop came from. On a one-combination-
    per-folder layout they coincide; under --labels workbook or --manifest one folder can
    hold several combinations.
    """
    det = np.minimum(cn, cm) >= threshold
    out = []
    for j, pos in enumerate(positions):
        row = dict(common)
        row.update({
            'sample': df['sample'].values[pos],
            'combination': combos[pos],
            'cell_global': int(df['cell_global'].values[pos]),
            'true_nu': df['expected_nu'].values[pos],
            'true_mito': df['expected_mito'].values[pos],
            'pred_nu': pn[j],
            'pred_mito': pm[j],
            'conf_nu': round(float(cn[j]), 4),
            'conf_mito': round(float(cm[j]), 4),
            'detected': int(det[j]),
        })
        row['correct_nu'] = int(row['pred_nu'] == row['true_nu'])
        row['correct_mito'] = int(row['pred_mito'] == row['true_mito'])
        row['correct_pair'] = int(row['correct_nu'] and row['correct_mito'])
        out.append(row)
    return out


def folder_is_combination(rows_df):
    """True when each sample folder holds exactly one combination and vice versa."""
    return (rows_df.groupby('sample')['combination'].nunique().max() == 1 and
            rows_df.groupby('combination')['sample'].nunique().max() == 1)


def summarise(rows_df, key):
    """Per-group detection and accuracy. Accuracy is over DETECTED cells only."""
    recs = []
    for name, g in rows_df.groupby(key, sort=True):
        d = g[g['detected'] == 1]
        recs.append({
            'group': name,
            'n_eval': len(g),
            'n_detected': len(d),
            'detect_pct': 100.0 * len(d) / len(g) if len(g) else np.nan,
            'pair_acc_pct': 100.0 * d['correct_pair'].mean() if len(d) else np.nan,
            'nu_acc_pct': 100.0 * d['correct_nu'].mean() if len(d) else np.nan,
            'mito_acc_pct': 100.0 * d['correct_mito'].mean() if len(d) else np.nan,
        })
    d = rows_df[rows_df['detected'] == 1]
    recs.append({
        'group': 'OVERALL',
        'n_eval': len(rows_df),
        'n_detected': len(d),
        'detect_pct': 100.0 * len(d) / len(rows_df) if len(rows_df) else np.nan,
        'pair_acc_pct': 100.0 * d['correct_pair'].mean() if len(d) else np.nan,
        'nu_acc_pct': 100.0 * d['correct_nu'].mean() if len(d) else np.nan,
        'mito_acc_pct': 100.0 * d['correct_mito'].mean() if len(d) else np.nan,
    })
    return pd.DataFrame(recs)


def print_table(title, table):
    def fmt(v):
        # An empty detected set prints n/a, never 0: an accuracy over zero detected cells
        # is undefined, and printing 0% there reads exactly like "every detected cell was
        # wrong", which is the opposite of what happened.
        return '  n/a' if (v is None or (isinstance(v, float) and np.isnan(v))) else '%5.1f' % v

    width = max([len(str(g)) for g in table['group']] + [8])
    print('')
    print(title)
    print('  %-*s %8s %9s %8s %8s %7s %7s'
          % (width, 'group', 'n_eval', 'detected', 'detect%', 'pair%', 'nu%', 'mito%'))
    for _, r in table.iterrows():
        print('  %-*s %8d %9d %8s %8s %7s %7s'
              % (width, r['group'], r['n_eval'], r['n_detected'], fmt(r['detect_pct']),
                 fmt(r['pair_acc_pct']), fmt(r['nu_acc_pct']), fmt(r['mito_acc_pct'])))


# ----------------------------------------------------------------------------------

def resolve_device(name):
    if name != 'auto':
        return torch.device(name)
    # Train_LUMINA.py and Test_LUMINA.py hardcode cuda:0 with no fallback. This one falls
    # back to CPU on purpose: with the backbone frozen, the heads-only fine-tune itself is
    # small enough to run without a GPU. The survey pass in front of it is not small -- it
    # pushes every crop of every population through the frozen backbone once -- so a CPU run
    # is bounded by that, not by the fine-tuning.
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise SystemExit('--checkpoint does not exist: %s' % path)
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict) or 'fc_nu.6.weight' not in sd:
        raise SystemExit(
            '%s does not look like a LUMINA state_dict (no fc_nu.6.weight). Train_LUMINA.py '
            'saves a bare state_dict; a full torch.save(model) or another architecture will '
            'not load here.' % path
        )
    num_classes = int(sd['fc_nu.6.weight'].shape[0])
    model = DualHeadConvNet(num_classes).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        raise SystemExit(
            'The checkpoint did not load cleanly into DualHeadConvNet(%d):\n%s\n'
            'A partially loaded model looks like "few-shot barely helps", so this is fatal '
            'rather than a warning.' % (num_classes, exc)
        )
    model.eval()
    print('Loaded %s: %d tensors, %d output classes per head'
          % (path, len(sd), num_classes))
    return model, num_classes


def main():
    ap = argparse.ArgumentParser(
        description='Few-shot adaptation of a LUMINA checkpoint to a new domain by '
                    'fine-tuning the two classifier heads.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--checkpoint', required=True,
                    help='Pre-trained LUMINA state_dict, e.g. best_model_fine-tune.pth.')
    ap.add_argument('--data-root', default='',
                    help='Root of the SUPPORT population: <root>/<sample>/<seg folder>/'
                         'cell<id>_5D.tif, see --seg-folder. Required unless --manifest is '
                         'given.')
    ap.add_argument('--eval-root', default='',
                    help='Root of the EVALUATION population. Omit for within-batch '
                         '(evaluate the rest of the support dish). Give a different dish '
                         'for the cross-batch arm.')
    ap.add_argument('--manifest', default='',
                    help='CSV listing the support cells explicitly (columns: path, sample, '
                         'expected_nu, expected_mito; optional batch, cell_global). '
                         'Replaces scanning --data-root.')
    ap.add_argument('--eval-manifest', default='',
                    help='Same, for the evaluation population.')
    ap.add_argument('--samples', default='',
                    help='Comma-separated sample folder names to restrict the scan to. '
                         'Default: every sample folder found.')
    ap.add_argument('--labels', default='auto', choices=['auto', 'workbook', 'foldername'],
                    help='Where labels come from when scanning a root. "workbook" reads '
                         'clustered.xlsx (Cell_Label / Nu_FP / Mito_FP) and fails if it is '
                         'missing or has other columns; "foldername" parses NT<mito>-<nu>; '
                         '"auto" uses the workbook when it is the dual-anchor one and falls '
                         'back to the folder name otherwise -- including when the folder '
                         'holds the napari plugin\'s clustered.xlsx, which has the same name '
                         'and different columns. The source used is printed per sample.')
    ap.add_argument('--seg-folder', default='auto',
                    choices=['auto', 'seg_5D', 'seg_5D_calib'],
                    help='Which folder of prepared crops to read under each sample. "auto" '
                         'takes seg_5D_calib when present and seg_5D otherwise, matching '
                         'Test_LUMINA.py. The measured adaptation runs read seg_5D only, so '
                         'pass --seg-folder seg_5D to reproduce them on a dish that has '
                         'both. The chosen folder is printed per sample either way.')
    ap.add_argument('--out', required=True, help='Output directory. Created if absent.')

    ap.add_argument('--k', type=int, default=20,
                    help='Support cells per barcode combination.')
    ap.add_argument('--epochs', type=int, default=30,
                    help='Fixed budget. No validation split and no early stopping -- that '
                         'is the recipe, not an omission.')
    ap.add_argument('--lr', type=float, default=0.0,
                    help='0 means auto: 1e-3 for --ft-mode heads, 1e-4 for full.')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=32, help='Fine-tuning batch size.')
    ap.add_argument('--infer-batch-size', type=int, default=64)
    ap.add_argument('--load-batch-size', type=int, default=64,
                    help='Crops read from disk at a time during the survey pass.')
    ap.add_argument('--ft-mode', default='heads', choices=['heads', 'full'],
                    help='"heads" trains fc_nu and fc_mito only, on cached frozen-backbone '
                         'features. "full" trains every parameter, at the lower lr.')
    ap.add_argument('--seed', type=int, default=0, help='Support-set seed.')
    ap.add_argument('--seeds', default='',
                    help='Comma-separated seed list; overrides --seed. Report mean and '
                         'spread across seeds, never a single run.')
    ap.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')

    ap.add_argument('--confidence-threshold', type=float, default=0.9,
                    help='A cell is detected when BOTH heads score at least this.')
    ap.add_argument('--min-px', type=int, default=3000,
                    help='Minimum segmented area in pixels. 0 disables.')
    ap.add_argument('--min-max', type=float, default=100.0,
                    help='Minimum peak RAW intensity, in the units Data_prep.py wrote. '
                         '0 disables. Set 0 if your crops are already normalised.')
    ap.add_argument('--drop-list', default='',
                    help='CSV of manually curated bad cells (batch, sample, cell_global). '
                         'Removes them from the support pool AND the evaluation set.')
    ap.add_argument('--oversize', default='downscale', choices=['downscale', 'skip'],
                    help='What to do with a crop larger than 256x256. "downscale" is what '
                         'the measured adaptation runs did; "skip" drops the cell entirely.')
    ap.add_argument('--skip-baseline', action='store_true',
                    help='Do not score the un-adapted checkpoint. By default it is scored '
                         'on exactly the same held-out cells, so before/after is honest.')
    ap.add_argument('--save-heads', action='store_true',
                    help='Write the adapted checkpoint, loadable by Test_LUMINA.py.')

    args = ap.parse_args()

    if not args.data_root and not args.manifest:
        raise SystemExit('Give --data-root (a folder of sample folders) or --manifest (a CSV).')
    if args.k < 1:
        raise SystemExit('--k must be at least 1.')
    if args.batch_size < 1 or args.epochs < 1:
        raise SystemExit('--batch-size and --epochs must be at least 1.')

    lr = args.lr or (1e-3 if args.ft_mode == 'heads' else 1e-4)
    device = resolve_device(args.device)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()] if args.seeds else [args.seed]

    print('nu_class_map: %s' % NU_CLASS_MAP)
    print('mito_class_map: %s' % MITO_CLASS_MAP)
    print('device: %s   ft-mode: %s   lr: %g   epochs: %d   batch: %d   K: %d'
          % (device, args.ft_mode, lr, args.epochs, args.batch_size, args.k))
    print('confidence threshold: %.2f   seeds: %s' % (args.confidence_threshold, seeds))

    model, num_classes = load_checkpoint(args.checkpoint, device)
    if num_classes < max(max(NU_CLASS_MAP.values()), max(MITO_CLASS_MAP.values())) + 1:
        raise SystemExit(
            'The checkpoint has %d outputs per head but the class map needs at least %d '
            '(index 0 is reserved for "no anchor"). This checkpoint was trained on a '
            'different panel and cannot be adapted onto this one.'
            % (num_classes, max(max(NU_CLASS_MAP.values()), max(MITO_CLASS_MAP.values())) + 1))

    # ---- populations -------------------------------------------------------------
    sup_tag = os.path.basename(os.path.normpath(args.manifest or args.data_root)) or 'support'
    if args.manifest:
        sup_df = load_manifest(args.manifest, sup_tag)
        sup_src = args.manifest
    else:
        sup_df = scan_population(args.data_root, args.samples, args.labels, sup_tag,
                                 args.seg_folder)
        sup_src = args.data_root
    validate_labels(sup_df, sup_src)
    print('[support population] %s: %d crops in %d sample folder(s), %d barcode combination(s)'
          % (sup_tag, len(sup_df), sup_df['sample'].nunique(),
             len(set(combination_keys(sup_df)))))

    # Passing the SAME dish to --data-root and --eval-root is an easy thing to type and used
    # to be silently catastrophic: it flipped the run to mode='cross', and the cross path
    # holds nothing out, so every cell just fine-tuned on was scored again as if it were
    # held out. The result looks excellent and is meaningless. The cluster script could not
    # express this (it looped `for tb in D: for eb in D: if tb != eb`), so the mistake is
    # native to the flag interface and belongs to it to catch.
    #
    # Resolving and comparing the paths is the convenient half of the fix, and it is only a
    # heuristic: it cannot see a manifest that lists the support dish's cells. The
    # disjointness assertion inside the seed loop is what actually guarantees the property.
    cross = bool(args.eval_root or args.eval_manifest)
    if cross:
        if same_location(args.eval_root, args.data_root):
            same_as = ('--eval-root', '--data-root', args.eval_root, 'dish')
        elif same_location(args.eval_manifest, args.manifest):
            same_as = ('--eval-manifest', '--manifest', args.eval_manifest, 'file')
        else:
            same_as = None
        if same_as is not None:
            print('[eval population] %s names the same %s as %s (%s), so this is a '
                  'WITHIN-batch run, not a cross-batch one.'
                  % (same_as[0], same_as[3], same_as[1], os.path.normpath(same_as[2])))
            print('  Taking the within-batch path: the support cells are held out of the '
                  'evaluation set. Scoring them would be scoring the cells just trained on, '
                  'which inflates every rate and looks like a very good result.')
            print('  Pass a genuinely different dish for the cross-batch arm, or drop the '
                  'flag -- omitting it is what "within-batch" already means.')
            cross = False

    mode = 'cross' if cross else 'within'
    if cross:
        eval_tag = os.path.basename(os.path.normpath(args.eval_manifest or args.eval_root)) or 'eval'
        if args.eval_manifest:
            eval_df = load_manifest(args.eval_manifest, eval_tag)
            eval_src = args.eval_manifest
        else:
            eval_df = scan_population(args.eval_root, args.samples, args.labels, eval_tag,
                                      args.seg_folder)
            eval_src = args.eval_root
        validate_labels(eval_df, eval_src)
        print('[evaluation population] %s: %d crops in %d sample folder(s), '
              '%d barcode combination(s)'
              % (eval_tag, len(eval_df), eval_df['sample'].nunique(),
                 len(set(combination_keys(eval_df)))))
    else:
        eval_tag = sup_tag

    drop_entries = read_drop_list(args.drop_list) if args.drop_list else {}
    if drop_entries:
        print('[drop-list] %s: %d curated (sample, cell) key(s)'
              % (args.drop_list, len(drop_entries)))

    want_feats = args.ft_mode == 'heads'
    print('Reading support crops ...')
    sup_df, sup_Z, sup_px, sup_mx = survey_population(model, sup_df, device, args,
                                                      want_feats, sup_tag)
    sup_df, sup_Z = apply_quality_gate(sup_df, sup_Z, sup_px, sup_mx, args, sup_tag)
    sup_df, sup_Z, n_dropped = apply_drop_list(sup_df, sup_Z, drop_entries, sup_tag)
    if len(sup_df) == 0:
        raise SystemExit('No support cells survived the quality gate and the drop list.')

    if cross:
        print('Reading evaluation crops ...')
        eval_df, eval_Z, ev_px, ev_mx = survey_population(model, eval_df, device, args,
                                                          want_feats, eval_tag)
        eval_df, eval_Z = apply_quality_gate(eval_df, eval_Z, ev_px, ev_mx, args, eval_tag)
        eval_df, eval_Z, n_ev_dropped = apply_drop_list(eval_df, eval_Z, drop_entries, eval_tag)
        n_dropped += n_ev_dropped
        if len(eval_df) == 0:
            raise SystemExit('No evaluation cells survived the quality gate and the drop list.')
    else:
        eval_df, eval_Z = sup_df, sup_Z

    # A curation file that silently matches nothing is worse than no curation file: the run
    # completes, prints a rate on the UNcurated population, and records the drop list's name
    # in the header as if it had been applied. But "matched nothing" has two causes and they
    # need different answers, because the protocol keeps ONE combined drop list covering
    # every dish while this script reads one dish per invocation:
    #   * the SAMPLE vocabulary is disjoint -- no sample name in the list occurs in the
    #     data. That is the genuine key-shape mismatch: the sample column is not holding
    #     sample folder names, so the list is not being applied ANYWHERE and nobody would
    #     notice. Fatal.
    #   * the sample names line up, but this particular dish holds none of the listed
    #     cells. That is what a multi-dish list looks like from inside one dish, and it is
    #     correct behaviour; say so once and carry on.
    #
    # Batch non-overlap on its own is NOT fatal, and used to be. A shared curation list is
    # the intended usage -- point every dish at one CSV -- and a dish that holds zero
    # curated cells has no reason to appear in the list at all, so its batch does not
    # either. Refusing to run there is precisely what would make a shared list unusable,
    # which is the opposite of the intent. It is reported, not enforced.
    if drop_entries and n_dropped == 0:
        seen_batches = sorted(set(sup_df['batch'].astype(str)) |
                              set(eval_df['batch'].astype(str)))
        seen_samples = sorted(set(sup_df['sample'].astype(str)) |
                              set(eval_df['sample'].astype(str)))

        list_samples = {s for s, _ in drop_entries}
        sample_overlap = sorted(list_samples & set(seen_samples))

        # Pairwise, because `batch_matches` is pairwise: unioning every list form and every
        # data form and intersecting once would let one entry's full date pair up with a
        # different entry's MMDD tail and report an overlap neither of them has. An entry
        # with no batch value is a wildcard and can never be the reason nothing matched.
        batch_wildcard = any(batch_is_wildcard(v)
                             for vals in drop_entries.values() for v in vals)
        batch_overlap = sorted({str(v).strip()
                                for vals in drop_entries.values() for v in vals
                                if not batch_is_wildcard(v)
                                and any(batch_matches(v, b) for b in seen_batches)})
        batch_ok = batch_wildcard or bool(batch_overlap)

        if sample_overlap:
            print('[drop-list] %s: %d entries, none of which name a cell present in this '
                  'run.' % (args.drop_list, len(drop_entries)))
            print('  The sample names (%s) do overlap this data, so the keys are the right '
                  'shape -- this is a list that covers several dishes and this invocation '
                  'reads one of them. Continuing.'
                  % (', '.join(sample_overlap[:6])
                     + (' ...' if len(sample_overlap) > 6 else '')))
            if not batch_ok:
                # Spellings, not the raw cell values: a `batch` column of 0102 comes back
                # from pandas as the integer 102, and printing that alone leaves a reader
                # unable to see why it did not match a dish called dual_20260102.
                listed = set()
                for vals in drop_entries.values():
                    for v in vals:
                        listed |= batch_key_forms(v)[0]
                print('  Its batch values (%s) match no batch in this data (%s), which is '
                      'what a list written for OTHER dishes looks like from inside this '
                      'one. Informational, not an error.'
                      % (', '.join(sorted(listed)[:8]), ', '.join(seen_batches)))
            print('  The curation header still records the list\'s name; it removed 0 cells '
                  'HERE. If you expected removals in this dish, check whether the quality '
                  'gate took those cells first.')
        else:
            raise SystemExit(
                '--drop-list %s has %d entries and removed ZERO cells, and no "sample" '
                'value in the list occurs in this data.\n'
                'That is a key-shape mismatch, not a list that happens to cover other '
                'dishes: the sample column is not holding sample folder names, so this '
                'list is being applied nowhere.\n'
                '  batch values in the data   : %s\n'
                '  sample values in the data  : %s\n'
                '  drop-list (sample, cell)   : %s%s\n'
                'The sample column has to hold the sample FOLDER name; the batch column may '
                'be the dish folder name, the acquisition date or its MMDD tail, padded or '
                'not -- but two FULL dates must agree, so a list written for 20250101 will '
                'not match a dish from 20260101. If the listed cells were legitimately '
                'already removed by the quality gate, either relax --min-px/--min-max or '
                'drop the --drop-list flag.'
                % (args.drop_list, len(drop_entries),
                   ', '.join(seen_batches),
                   ', '.join(seen_samples[:8]) + (' ...' if len(seen_samples) > 8 else ''),
                   ', '.join('%s/%d' % k for k in list(drop_entries)[:8]),
                   ' ...' if len(drop_entries) > 8 else ''))

    # Sorted only so the printed line and finetune_run_config.csv are stable to read. The
    # draw itself does not depend on this order: sample_support seeds every combination
    # from (seed, combination name), so re-ordering, adding or renaming OTHER combinations
    # leaves each combination's own support cells unchanged.
    groups = sorted(set(combination_keys(sup_df)), key=natural_key)
    print('support is drawn per barcode COMBINATION (expected_nu + expected_mito), K=%d '
          'each: %s' % (args.k, ', '.join(groups)))
    print('  each combination is seeded from (--seed, its own name), so this listing order '
          'does not affect the draw; the ROW order within a combination does, and is the '
          'natural-sorted order of the crop files.')

    yn_all = sup_df['expected_nu'].map(NU_CLASS_MAP).values.astype(np.int64)
    ym_all = sup_df['expected_mito'].map(MITO_CLASS_MAP).values.astype(np.int64)
    sup_combos = combination_keys(sup_df)
    eval_combos = combination_keys(eval_df)

    # ---- run ---------------------------------------------------------------------
    curation = 'min_px=%d;min_max=%g;drop_list=%s;oversize=%s' % (
        args.min_px, args.min_max, os.path.basename(args.drop_list) if args.drop_list else 'none',
        args.oversize)
    os.makedirs(args.out, exist_ok=True)

    # Every flag, resolved, written once. The per-cell rows carry only the flags that change
    # a number in them; this file is what makes a directory of results self-describing.
    cfg = {k: v for k, v in sorted(vars(args).items())}
    cfg['resolved_lr'] = lr
    cfg['resolved_device'] = str(device)
    cfg['resolved_seeds'] = ','.join(str(s) for s in seeds)
    cfg['support_batch'] = sup_tag
    cfg['eval_batch'] = eval_tag
    cfg['mode'] = mode
    cfg['combinations'] = ';'.join(groups)
    pd.DataFrame([{'flag': k, 'value': v} for k, v in cfg.items()]).to_csv(
        os.path.join(args.out, 'finetune_run_config.csv'), index=False)

    per_seed = []

    for seed in seeds:
        support = sample_support(sup_df, groups, args.k, seed)
        if len(support) == 0:
            raise SystemExit('Seed %d selected no support cells at all.' % seed)

        pd.DataFrame({
            'batch': sup_df['batch'].values[support],
            'sample': sup_df['sample'].values[support],
            'combination': sup_combos[support],
            'cell_global': sup_df['cell_global'].values[support],
            'K': args.k,
            'seed': seed,
        }).to_csv(os.path.join(args.out, 'support_cells_K%d_seed%d.csv' % (args.k, seed)),
                  index=False)

        if mode == 'within':
            held = np.ones(len(eval_df), bool)
            held[support] = False
        else:
            held = np.ones(len(eval_df), bool)
        held_pos = np.where(held)[0]
        held_t = torch.from_numpy(held_pos).long()
        if len(held_pos) == 0:
            raise SystemExit('Nothing left to evaluate: the support set is the whole '
                             'population. Lower --k.')
        assert_support_held_out(sup_df, support, eval_df, held_pos, mode, seed)

        common = {'mode': mode, 'stage': 'finetuned', 'train_batch': sup_tag,
                  'eval_batch': eval_tag, 'seed': seed, 'K': args.k,
                  'ft_mode': args.ft_mode, 'epochs': args.epochs, 'lr': lr,
                  'weight_decay': args.weight_decay, 'batch_size': args.batch_size,
                  'labels': args.labels, 'seg_folder': args.seg_folder,
                  'threshold': args.confidence_threshold, 'curation': curation}
        rows = []

        if not args.skip_baseline:
            # The un-adapted checkpoint, on EXACTLY the held-out cells the adapted model is
            # scored on, so before/after is a comparison on identical cells. Scoring the
            # baseline on the whole population instead gives it a slightly larger and
            # differently composed denominator, and the difference between the two numbers
            # then mixes the adaptation with a change of denominator.
            if want_feats:
                bn, bm, bcn, bcm = predict_from_features(
                    model.fc_nu, model.fc_mito, eval_Z[held_t.to(eval_Z.device)], args.infer_batch_size)
            else:
                bn, bm, bcn, bcm = predict_from_images(model, eval_df, held_pos, device, args)
            base_common = dict(common)
            base_common.update({'stage': 'baseline', 'K': 0})
            rows += build_rows(eval_df, eval_combos, held_pos, bn, bm, bcn, bcm,
                               args.confidence_threshold, base_common)

        # torch is seeded here, which the runs the manuscript reports did not do: they
        # seeded only the numpy draw of the support set, so the batch order and the dropout
        # masks inside the heads varied between two otherwise identical invocations and a
        # run could not be reproduced exactly. Seeding torch changes nothing about the
        # recipe, but it does mean a given --seed lands on a different trained head than the
        # same seed would have there. Declared here so a discrepancy is not a mystery.
        torch.manual_seed(seed)

        print('')
        print('[fine-tune] seed %d, K=%d, %d support cells, %d held-out cells'
              % (seed, args.k, len(support), len(held_pos)))

        if args.ft_mode == 'heads':
            fn, fm = train_heads_cached(model, sup_Z[torch.from_numpy(support).to(sup_Z.device)],
                                        yn_all[support], ym_all[support], device, args, lr)
            pn, pm, cn, cm = predict_from_features(fn, fm, eval_Z[held_t.to(eval_Z.device)],
                                                   args.infer_batch_size)
            if args.save_heads:
                adapted = copy.deepcopy(model)
                adapted.fc_nu.load_state_dict(fn.state_dict())
                adapted.fc_mito.load_state_dict(fm.state_dict())
                state = adapted.state_dict()
        else:
            xsup = load_crops(sup_df, support, args.oversize)
            m = finetune_full(model, xsup, yn_all[support], ym_all[support], device, args, lr)
            pn, pm, cn, cm = predict_from_images(m, eval_df, held_pos, device, args)
            state = m.state_dict() if args.save_heads else None

        rows += build_rows(eval_df, eval_combos, held_pos, pn, pm, cn, cm,
                           args.confidence_threshold, common)

        if args.save_heads:
            ck = os.path.join(args.out, 'best_model_fewshot_K%d_seed%d.pth' % (args.k, seed))
            torch.save({k: v.cpu() for k, v in state.items()}, ck)
            print('adapted checkpoint written to %s' % ck)

        rows_df = pd.DataFrame(rows)
        pred_path = os.path.join(args.out, 'finetune_predictions_K%d_seed%d.csv'
                                 % (args.k, seed))
        rows_df.to_csv(pred_path, index=False)

        summaries = []
        for stage in ('baseline', 'finetuned'):
            part = rows_df[rows_df['stage'] == stage]
            if len(part) == 0:
                continue
            # Grouped by combination, because that is the unit K is counted in. The
            # per-folder breakdown is printed as well, but only when a folder is not simply
            # one combination -- otherwise it is the same table twice.
            table = summarise(part, 'combination')
            table.insert(0, 'stage', stage)
            table.insert(0, 'seed', seed)
            table.insert(0, 'mode', mode)
            summaries.append(table)
            print_table('%s | %s-batch | seed %d | K=%d | threshold %.2f  (per combination)'
                        % (stage, mode, seed, args.k if stage == 'finetuned' else 0,
                           args.confidence_threshold), table)
            if not folder_is_combination(part):
                print_table('    ... by sample folder', summarise(part, 'sample'))
            print_table('    ... by nuclear class', summarise(part, 'true_nu'))
            print_table('    ... by mitochondrial class', summarise(part, 'true_mito'))
            overall = table[table['group'] == 'OVERALL'].iloc[0]
            per_seed.append({'mode': mode, 'stage': stage, 'seed': seed,
                             'K': args.k if stage == 'finetuned' else 0,
                             'n_eval': overall['n_eval'], 'detect_pct': overall['detect_pct'],
                             'pair_acc_pct': overall['pair_acc_pct']})

        sum_path = os.path.join(args.out, 'finetune_summary_K%d_seed%d.csv' % (args.k, seed))
        pd.concat(summaries, ignore_index=True).to_csv(sum_path, index=False)
        print('')
        print('predictions -> %s' % pred_path)
        print('summary     -> %s' % sum_path)

    # ---- across seeds ------------------------------------------------------------
    per_seed_df = pd.DataFrame(per_seed)
    per_seed_df.to_csv(os.path.join(args.out, 'finetune_per_seed_K%d.csv' % args.k), index=False)
    if len(seeds) > 1:
        # Aggregate by (mode, stage, seed) FIRST, then across seeds. Pooling the per-cell
        # rows instead would weight each seed by its cell count and collapse the
        # seed-to-seed spread, which is the quantity the seeds exist to measure.
        print('')
        print('across %d seeds (per-seed overall figures aggregated):' % len(seeds))
        # Each statistic gets its OWN contributing count, and the SEM is divided by that,
        # not by the number of seeds run. A seed whose held-out cells all fell below the
        # confidence threshold contributes a NaN accuracy: pandas' mean() and std() skip it,
        # so the mean beside it is already a mean over fewer seeds, and dividing that sd by
        # sqrt(seeds RUN) makes the SEM smaller than the data supports -- the one direction
        # an error bar must never be wrong in. `count()` counts non-NaN, which is exactly
        # the number mean() and std() used.
        #
        # `n_seeds_run` and the two `*_n_seeds` columns are named apart on purpose: they are
        # equal in the ordinary case and a reader who saw only one of them could not tell
        # which question it answers.
        agg = per_seed_df.groupby(['mode', 'stage'], sort=False).agg(
            n_seeds_run=('seed', 'nunique'),
            detect_n_seeds=('detect_pct', 'count'),
            detect_mean=('detect_pct', 'mean'), detect_sd=('detect_pct', 'std'),
            pair_n_seeds=('pair_acc_pct', 'count'),
            pair_mean=('pair_acc_pct', 'mean'), pair_sd=('pair_acc_pct', 'std')).reset_index()
        # BOTH spreads are written, and each is labelled for what it is. The two answer
        # different questions -- sd is how much a single run moves when the seed changes,
        # sem is how precisely the mean over THOSE seeds is pinned down -- and relabelling
        # one as the other is a silent factor of sqrt(n). sd here is pandas' default ddof=1
        # (sample sd), and sem = sd / sqrt(contributing seeds) accordingly. A single
        # contributing seed has no spread at all and both columns are empty for it.
        agg['detect_sem'] = agg['detect_sd'] / np.sqrt(agg['detect_n_seeds'])
        agg['pair_sem'] = agg['pair_sd'] / np.sqrt(agg['pair_n_seeds'])
        agg = agg[['mode', 'stage', 'n_seeds_run',
                   'detect_n_seeds', 'detect_mean', 'detect_sd', 'detect_sem',
                   'pair_n_seeds', 'pair_mean', 'pair_sd', 'pair_sem']]
        for _, r in agg.iterrows():
            # Same convention as print_table: a mean accuracy over seeds that detected
            # nothing is n/a, not a number. "nan" in a results table gets copied as one.
            def num(v, fmt='%5.1f'):
                return '  n/a' if pd.isna(v) else fmt % v

            def spread(v):
                # 'n/a', never ' 0.0'. An unavailable spread and a measured spread of zero
                # are opposite claims -- "we cannot say" against "every seed agreed
                # exactly" -- and printing them byte-identical hands a reader the stronger
                # one for free. Padded to the same 4 columns as '%4.1f' so the table lines
                # up either way.
                return ' n/a' if pd.isna(v) else '%4.1f' % v
            print('  %-9s %-6s n=%d  detection %s +/- %s sd (+/- %s sem) %% [n=%d]   '
                  'accuracy %s +/- %s sd (+/- %s sem) %% (of detected) [n=%d]'
                  % (r['stage'], r['mode'], int(r['n_seeds_run']), num(r['detect_mean']),
                     spread(r['detect_sd']), spread(r['detect_sem']),
                     int(r['detect_n_seeds']),
                     num(r['pair_mean']),
                     spread(r['pair_sd']), spread(r['pair_sem']),
                     int(r['pair_n_seeds'])))
        print('  n=<seeds run>; the [n=..] after each figure is how many of them CONTRIBUTED '
              'to it -- a seed that detected nothing has no accuracy to average, and its '
              'mean, sd and sem are all over the smaller number.')
        print('  sd = spread of one run over seeds; sem = sd / sqrt(contributing seeds) = '
              'precision of the mean. Quote whichever you mean, by name.')
        agg.to_csv(os.path.join(args.out, 'finetune_across_seeds_K%d.csv' % args.k), index=False)

    print('')
    print('Accuracy is computed over DETECTED cells only, so it moves against detection: '
          'raising --confidence-threshold buys accuracy by shrinking the denominator. '
          'Quote the two together, and compare settings only at matched detection.')
    print('Population as scored: %s' % curation)


if __name__ == '__main__':
    main()
