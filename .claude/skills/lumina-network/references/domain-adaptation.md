# Adapting LUMINA to a new domain

A LUMINA checkpoint is trained on one cell line under one set of imaging
conditions. Applied to a different cell line it degrades badly — not because
the barcodes stop being separable, but because the *spectral* features move
while the network's decision boundaries stay where they were.

This page describes the adaptation protocol that has been used to move a
checkpoint onto a new cell line, and the script in this repository that runs it:
**`LUMINA_classification/Finetune_LUMINA.py`**. It is the one script here with a
command line — see the last section for why, and for what it does *not* cover.

## What actually shifts between cell lines

Moving a trained checkpoint onto a different cell line:

- **The spectral intensity ratios move.** A small near-uniform offset in the
  first ratio, in the same direction for every fluorophore.
- **The phasor coordinates barely move.** *G* and *S* stay put.

The shift is a near-uniform offset shared by all classes, not a reshuffling of
the classes relative to each other. That is why the fix works and why it is
cheap: the *features* are still good, only the *heads* are miscalibrated.

**Zero-shot transfer is not usable.** Applying the checkpoint to a new cell
line without adaptation lands far below usable, and some barcodes collapse
almost entirely. Do not report a zero-shot number as the method's performance,
and do not let a user conclude the model is broken from one — it means the heads
need recalibrating, nothing more.

The manuscript carries the measured shift, the measured zero-shot figures and
the cell lines they were measured on. It is not published yet, so none of those
numbers appear anywhere in this repository. Quote the manuscript, or run
`Finetune_LUMINA.py` and quote its output.

## The recipe

Few-shot fine-tuning of **the two classifier heads only**. This table is what
`Finetune_LUMINA.py` implements; every row is either a hardcoded property of the
script or the default of the flag named in the right-hand column.

| Item | Value | Flag |
|---|---|---|
| Frozen | every parameter whose name does not contain `fc_nu` or `fc_mito` — all six input stems **and** the shared backbone | `--ft-mode heads` |
| Trained | `fc_nu` and `fc_mito` only | |
| Head size | unchanged — the heads keep their original class count and their pre-trained weights; they are not resized or re-initialised for the new panel | none, by design |
| Support set | *K* cells per barcode **combination**, sampled without replacement | `--k 20`, `--seed` / `--seeds` |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` | `--lr`, `--weight-decay` |
| Epochs | 30, fixed. No validation split, no early stopping | `--epochs 30` |
| Batch size | 32, reshuffled each epoch | `--batch-size 32` |
| Loss | `CE(nucleus) + CE(mitochondria)`, unweighted, summed 1:1 | none |
| BatchNorm | frozen — features are extracted once under `eval`, so the backbone keeps its source-domain running statistics | implied by `--ft-mode heads` |
| Augmentation | none | |
| Input normalisation | clamp *G*/*S* to [0,1], simplex-project the three ratio planes, zero every feature plane outside the intensity mask, divide intensity by its per-cell max | none |

Sampling is stratified by combination, not by class, so per-class support is
imbalanced by construction: a nuclear barcode appearing in three combinations
contributes 3*K* support cells and one appearing in a single combination
contributes *K*. No class weighting compensates. Barcodes not present in any
combination get zero support cells and their output units stay live.

The combination is the barcode pair `(expected_nu, expected_mito)`, **not** the
sample folder. On the layout the protocol was measured with, one folder holds
exactly one combination and the two are the same partition — but `--labels
workbook` and `--manifest` both let one folder hold several combinations, and
there *K* still means *K* per combination. The script prints the groups it will
draw from at startup; read that line rather than assuming.

**What a `--seed` does and does not pin down.** Each combination is drawn from
its own RNG, derived from `(--seed, that combination's name)`, so a
combination's support cells depend only on the seed, its own name and the order
of its own rows. Re-ordering the other combinations, adding a dish that carries
one more combination, or renaming a combination it does not draw from all leave
it untouched. The row order *within* a combination is still part of it — the
draw is over positional indices — which is why every enumeration in the script
is natural-sorted, and why a seed is reproducible against the same folder but
not against a re-ordered one.

That is deliberately **not** what the measured runs did. There the support set
came out of one RNG stream walked over a hardcoded list of the experiment's
combination names, which is unpublished and has been removed from this
repository, so the order that produced those draws no longer exists here. A
`--seed` here therefore does **not** select the cells a same-numbered seed
selected for the manuscript. What is preserved is the protocol — *K* per
combination, without replacement, balanced across combinations — and that is
what the seeds exist to average over: the seed is a nuisance parameter you run
several of, not an identifier of a particular support set.

Fine-tuning the whole network instead of the heads is the fallback, at
`lr=1e-4` (`--ft-mode full`); it has not been needed.

The input normalisation row matters more than it looks. It is **not**
`Test_LUMINA.normalize_intensity`, which performs only the last of the steps in
that row. Adaptation was measured with the full sequence, and
`Finetune_LUMINA.py` implements that; a checkpoint adapted here and then run
through `Test_LUMINA.py` is being fed a slightly different distribution. If
someone reports that the numbers do not reproduce under `Test_LUMINA.py`, this
is the first thing to check.

### Which crops get read

A prepared dish may contain `seg_5D`, `seg_5D_calib`, or both. `--seg-folder
auto` (the default) takes `seg_5D_calib` when it is there and `seg_5D`
otherwise, matching `Test_LUMINA.py` and the rest of the repository. **The
measured adaptation runs read `seg_5D` only.** On a dish that has both, pass
`--seg-folder seg_5D` to reproduce them; otherwise the network is being fed
calibrated crops where the protocol used uncalibrated ones, which is a different
input distribution and not the same experiment. The folder actually read is
printed for every sample.

### Where the labels come from

`--labels auto` (the default) reads the dual-anchor `clustered.xlsx`
(`Cell_Label` / `Nu_FP` / `Mito_FP`) when the sample folder has one, and falls
back to parsing the folder name otherwise. The fallback matters here: the napari
plugin in this same repository writes its own `clustered.xlsx` (`Mask label` /
`FOV` / `Localization` / `cluster_local` / `cluster_tag`) into sample folders. It
is a different file with the same name, `auto` skips it and uses the folder name,
and the source chosen is printed per sample folder. `--labels workbook` is the
strict form: if the user named the workbook, a missing or wrong-schema file is
fatal rather than silently replaced.

### How many cells do you need to label?

**Accuracy saturates almost immediately — detection does not.** With a handful
of cells per barcode the labels the model commits to are already almost all
correct; what a larger support set buys is that the model becomes *willing to
commit* on more cells, especially in the weakest barcode. So:

- If you want correct labels on the cells the model is confident about, a few
  cells per barcode is enough.
- If you want a high fraction of cells to receive a call at all, budget a few
  tens per barcode.

Increasing the epoch count does not substitute for support cells: the epoch
budget was swept at 10 and 50 epochs either side of the default 30, and the
result barely moves. If adaptation is not working, the answer is more labelled
cells, not more epochs.

### Evaluation protocol

Report both **within-batch** and **cross-batch**:

- **within-batch** — fine-tune on *K* cells from one dish, test on every
  remaining cell of that same dish, with the support cells explicitly excluded.
  This is `--data-root <dish>` with no `--eval-root`.
- **cross-batch** — fine-tune on one dish, test on the whole of a *different*
  dish. This is `--data-root <dish A> --eval-root <dish B>`. Run both orderings.

`--eval-root` must name a **different** dish. Pointing it at the same one asks
for the cross-batch behaviour — score the whole evaluation population — on a
population that contains the support cells, which would score the cells just
fine-tuned on and inflate both rates. The script resolves both paths and, if
they name the same dish, says so and takes the within-batch path instead. It
also asserts outright that no support cell is among the cells being scored, in
either mode, and refuses to run if one is: that catches the cases a path
comparison cannot see, such as an `--eval-manifest` listing crops from the
support dish. A byte-for-byte *copy* of the dish at another path is the one thing
neither check can distinguish from a genuine second dish; both roots are recorded
in `finetune_run_config.csv`, so it is at least visible in the record.

Repeat over several support-set seeds and several dishes, and report
mean ± SEM. A small within-minus-cross gap is the evidence that the adaptation
generalises rather than memorising the dish it was tuned on.

`finetune_across_seeds_K<k>.csv` writes **both** spreads, one column each, and
you have to pick the one you mean by name:

| Column | Is |
|---|---|
| `detect_sd`, `pair_sd` | sample standard deviation (ddof=1) across seeds — how much a *single* run moves when the seed changes |
| `detect_sem`, `pair_sem` | `sd / sqrt(contributing seeds)` — how precisely the mean over the seeds you ran is pinned down |
| `n_seeds_run` | how many seeds this `(mode, stage)` was run with |
| `detect_n_seeds`, `pair_n_seeds` | how many of them **contributed** to that statistic — the *n* in its square root |

For the protocol above, report the **`sem`** columns. Both are printed on the
console line as well, each labelled. Neither column exists for a single seed —
one run has no spread — so a `±` on a one-seed result is not available and
should not be invented.

**`n_seeds_run` and the `*_n_seeds` columns are not the same number and are
named apart for that reason.** A seed whose held-out cells all fell below the
confidence threshold has no accuracy to contribute: it is absent from
`pair_mean`, from `pair_sd` and from `pair_n_seeds`, while `n_seeds_run` still
counts it. Quote the mean together with the count that produced it — a mean over
four seeds reported as a mean over five is a real misstatement, and the SEM
attached to it would be too small by `sqrt(5/4)`. On the console the contributing
count is the `[n=…]` after each figure.

An unavailable spread prints `n/a`, never `0.0` — those are opposite claims
("we cannot say" against "every seed agreed exactly") and the CSV leaves the
cell empty for the same reason.

Report **two numbers, never one**: accuracy (of the cells that got a call, how
many are right) and detection rate (what fraction of cells got a call at all).
They respond differently to *K*, and quoting only accuracy hides the fact that
a low-*K* model is confident about very little.

**Aggregate by seed first.** `Finetune_LUMINA.py` writes one row per evaluated
cell, and a cell appears once per seed. Pooling the prediction CSVs and taking a
grand mean weights each seed by its cell count and collapses the seed-to-seed
spread — which is the quantity the seeds exist to measure. Group by
`(mode, seed)`, then average across seeds. The script's own
`finetune_across_seeds_K<k>.csv` does this; the per-cell file is there for
confusion matrices, not for headline numbers.

**Compare against the un-adapted checkpoint on the same cells.** The script
scores the baseline on exactly the held-out mask by default, so before/after is
a comparison on identical cells rather than on two different denominators. Pass
`--skip-baseline` only when you already have that number.

When a barcode combination shows accuracy `n/a`, that means **nothing was
detected** in it, not that everything detected was wrong. Read the detection
column before interpreting any accuracy.

## Triage — the curation gate that comes first

"Triage" is not part of the fine-tuning; it is the curation stage that defines
an honest denominator. It removes exactly three categories:

1. **Segmentation fragments** — broken or partial masks.
2. **Non-expression** — the nuclear or the mitochondrial anchor is absent.
3. **Contamination** — a different barcode is present in the cell.

**Genuine look-alike confusions are kept.** Cells that are hard because two
barcodes really do resemble each other stay in, and the residual error rate is
biological ambiguity, not dirt. Dropping them would inflate the result and would
be cherry-picking; if someone proposes removing "cells the model gets wrong",
that is the objection to raise.

Cells removed by triage are excluded from **both** the support pool and the
evaluation set.

In practice the gate is a quality threshold on mask size and peak intensity,
then a human review pass, then a single global keep/drop classifier used only
to auto-drop at high confidence. Per-class drop classifiers were tried and
rejected — their drop-precision was not good enough, and inspection showed most
of what they dropped was fine.

In `Finetune_LUMINA.py` the automatic threshold is `--min-px` (segmented area in
pixels) and `--min-max` (peak intensity); the human pass arrives as
`--drop-list`, a CSV of `batch, sample, cell_global`. All three shrink the
denominator of every rate the script prints, not merely which cells may be
labelled, so the run echoes the curation settings into every output row — a rate
quoted without them is not interpretable.

The `batch` column of a drop list is matched loosely, but **the year is never
discarded**. The dish folder name (`dual_20260101_2`), the acquisition date
(`20260101`) and the `MMDD` tail (`0101`, or the `101` pandas hands back for it)
all count as the same batch — with one rule on top:

> When **both** sides carry a full 8-digit date, those dates must be equal.
> `MMDD` is a *fallback*, used only when one side has no full date at all.

The loose half exists because a curation file is routinely written with an
`MMDD`-style key, which pandas reads back as the integer `101`, so a strict
string comparison would match nothing while still printing that the list had
loaded. The strict half exists because the tail alone would make a list written
for `20250101` remove cells from a dish imaged on `20260101` — silently, from
both the support pool and the evaluation set. So `20250101` does **not** match
`dual_20260101_2`; `0101` and `20260101` both do.

A non-empty drop list that removes **zero** cells is checked, and the two reasons
that can happen are answered differently:

- **The `sample` vocabulary is disjoint** — no `sample` value in the list occurs
  in the data. The `sample` column is not holding sample folder names, so the
  list is being applied *nowhere* and nobody would notice. Hard error, with the
  batch and sample values actually present printed alongside.
- **The sample names line up but this dish holds none of the listed cells.**
  The authoritative protocol keeps one combined drop list covering every dish
  while this script reads one dish per invocation, so a dish the list does not
  touch is normal and correct. The script prints one line saying the list covers
  other dishes and continues.

**Batch non-overlap on its own is informational, not fatal.** A dish that holds
zero curated cells has no reason to appear in the list at all, so neither does
its batch; refusing to run there is exactly what would make a shared list
unusable. The script says so in a second line — the batch spellings the list
carries and the ones the data carries — and carries on. This is the common case:
pointing every dish at the same curation CSV is the intended usage. What is not
tolerated is a list that silently does nothing *everywhere*, because the run
completes and records the list's name in the header as if it had been applied.

`--min-max` is measured on the **raw** intensity plane, in the units
`Data_prep.py` wrote, before the per-cell max normalisation. If someone's crops
have already been scaled into [0, 1], the default threshold removes every cell;
the script raises rather than reporting on an empty set, and tells them to pass
`--min-max 0`.

## Two traps

**Cached-feature training changes the BatchNorm mode.** Because the backbone is
frozen, its 512-d output can be computed once per dish and only the heads
trained on the cache — faster, since the backbone then runs once instead of once
per epoch. But caching runs the backbone in `eval` mode where
the direct path runs it in `train` mode. Putting the whole model in `train` mode
does not un-freeze the frozen weights, but it does let BatchNorm normalise with
the support batch's own statistics and update its running estimates, so the
direct path is a head fine-tune *plus* an unannounced BatchNorm adaptation. The
two agree once the support set is reasonably sized; at very small *K* the cached
path reads measurably **higher**. Any small-*K* claim must say which path
produced it. (The size of that gap is in the manuscript, not here.)

`--ft-mode heads` in `Finetune_LUMINA.py` is the **cached, BatchNorm-frozen**
path — the one whose behaviour actually matches the phrase "we fine-tuned the
two heads", and the one the final protocol used. `--ft-mode full` trains
everything in `train` mode, so BatchNorm does move there; that is expected for
that arm and is not the same experiment.

**Some barcode pairs overlap by design.** At least one nucleus/mitochondria
pair is not separable in the 5-D feature space in any dataset, including the
model's own training data — it is separated only by *which organelle* the
signal is in, i.e. by the two heads being separate, never by the features. That
pair sets the floor on every metric, and no amount of fine-tuning moves it.
This is a property of the barcode panel, not a defect of the network.

## What is in this repository, and what is not

**The trainer is here**: `LUMINA_classification/Finetune_LUMINA.py`. It
implements the recipe table, the support sampling, the confidence gate and both
evaluation arms, and it is the answer to "how do I move LUMINA onto my own
cells".

```bash
# within-batch: adapt on K cells per combination from a dish, read out the rest
python Finetune_LUMINA.py --checkpoint best_model_fine-tune.pth \
    --data-root /path/to/new_dish --out ./adapt_out --k 20 --seeds 0,1,2,3,4

# cross-batch: adapt on one dish, score a different one
python Finetune_LUMINA.py --checkpoint best_model_fine-tune.pth \
    --data-root /path/to/dish_A --eval-root /path/to/dish_B \
    --out ./adapt_out --k 20 --seeds 0,1,2,3,4
```

Everything it writes, under `--out`:

| File | Written | Holds |
|---|---|---|
| `finetune_predictions_K<k>_seed<s>.csv` | per seed | one row per evaluated cell, baseline and fine-tuned stages both |
| `finetune_summary_K<k>_seed<s>.csv` | per seed | the per-combination detection/accuracy table |
| `support_cells_K<k>_seed<s>.csv` | per seed | exactly which cells were labelled, so a reviewer can check none of them was a known-bad cell |
| `finetune_per_seed_K<k>.csv` | **every run** | the one overall `(mode, stage, seed)` row per seed that the across-seeds aggregate is computed from — the file to check when a mean looks wrong, and the only across-seed file a single-seed run produces |
| `finetune_across_seeds_K<k>.csv` | when >1 seed | mean, `sd`, `sem`, `n_seeds_run` and a per-statistic contributing count per `(mode, stage)` |
| `finetune_run_config.csv` | every run | every flag and its resolved value, so a results directory says what produced it without the shell history |
| `best_model_fewshot_K<k>_seed<s>.pth` | with `--save-heads` | the adapted checkpoint, loadable by `Test_LUMINA.py` |

**This is the one script in the folder with an argument parser.** Everywhere
else the correct answer is "edit line N"; here it is a flag. Say so rather than
sending someone to edit a constant that does not exist. `--help` is authoritative
for the flag list.

Two things it does *not* ship, both of which are data, not code:

- **The pre-trained checkpoint.** `--checkpoint` is required and has no default.
  The label space is the checkpoint's, and it stores no class map, so the class
  indices in `Finetune_LUMINA.py` must match the ones the checkpoint was trained
  under. The script refuses a barcode name it does not know instead of silently
  assigning class 0.
- **The manual drop list.** `--drop-list` is optional with no default. If a set
  of numbers was produced with one, the list has to travel with the numbers, or
  the numbers have to be reported without it as well.

`Train_LUMINA.py` also has a `use_finetune` path. It is **not** this protocol,
and is disabled in the shipped configuration:

- `freeze_conv_layers()` freezes only `model.input_heads`; the backbone freeze
  is commented out — so it trains the backbone as well as the heads.
- It uses AdamW, 800 epochs, batch size 128 and early stopping.
- It has no *K*-shot support-set sampling, no confidence gate and no
  within/cross-batch evaluation.

If someone says "I fine-tuned it with `Train_LUMINA.py`", they did something
else. Point them at `Finetune_LUMINA.py`.

## How a run here differs from the runs the manuscript reports

The **recipe** is the same — the table above is what was measured. What is not
the same is which cells a given `--seed` labels and which trained head it lands
on, for five reasons, plus one more that changes a number without changing which
cells are used. None is an accident; say so plainly rather than implying a
`--seed` reproduces a published run.

1. **Each combination is seeded separately** from `(--seed, its name)`, where
   the original walked one RNG stream over a hardcoded list of the experiment's
   combination names. That list is unpublished and was removed, so its order
   cannot be reproduced here — and depending on it would have meant that adding
   or renaming any combination silently changed every other one's support set.
   Different cells, same protocol.
2. **The support indices are not sorted** before training. They come out in
   group-concatenation order, as the original produced them. This looks
   cosmetic and is not: it sets the mini-batch composition, so sorting the same
   support cells trains a different head.
3. **PyTorch is seeded** as well as the support draw, so a given `--seed`
   reproduces the whole run. The original seeded only the numpy draw, leaving
   its batch order and the dropout masks inside the heads to vary between two
   otherwise identical invocations.
4. **`--seg-folder auto` is the default**, and the measured runs read `seg_5D`.
   This one you can undo: pass `--seg-folder seg_5D`.
5. **`--labels auto` is the default**, and the original derived labels from the
   sample folder name unconditionally. `auto` prefers a dual-anchor
   `clustered.xlsx` when the sample folder has one, and a workbook that
   disagrees with the folder name relabels cells — which moves them between
   barcode combinations, changes the partition *K* is drawn per, and therefore
   changes every rate. Undo it the same way: pass `--labels foldername`. (The
   two are exactly parallel; if you are pinning one, pin both.)

Items 4 and 5 you can undo. Items 1–3 you cannot.

One further difference, on a different axis — it changes a number rather than
which cells are used:

6. **The baseline is scored on the held-out cells, not on the whole dish.** The
   original's *K* = 0 row ran the un-adapted checkpoint over the entire
   population; this script scores it on exactly the mask the adapted model is
   scored on, so before/after is a difference on identical cells rather than on
   two denominators. In the cross-batch arm the two sets are the same and the
   number is unchanged; in the within-batch arm this baseline is computed over
   exactly the support set fewer cells. Reported figures for the *adapted* model
   are unaffected either way. Pass `--skip-baseline` if you do not want it.

The consequence to state out loud when someone asks: numbers from this script
are a re-run of the protocol, not a replay of the manuscript's runs, and they
are expected to differ by about the seed-to-seed spread the script itself
prints. Run several seeds and compare distributions, never single runs.

Do not invent the missing numbers. If asked "how accurate is it", say what this
page says qualitatively, point at the manuscript for the measured values, and
offer to run the script — its own output is the honest way to get a number.
