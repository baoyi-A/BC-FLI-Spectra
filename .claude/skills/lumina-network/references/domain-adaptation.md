# Adapting LUMINA to a new domain

A LUMINA checkpoint is trained on one cell line under one set of imaging
conditions. Applied to a different cell line it degrades badly — not because
the barcodes stop being separable, but because the *spectral* features move
while the network's decision boundaries stay where they were.

This page describes the adaptation protocol that has been used to move a
checkpoint onto a new cell line. **Read the last section first if you intend to
run it**: the trainer that implements this protocol is not in this repository.

## What actually shifts between cell lines

Measured moving a HEK293T-trained checkpoint onto A549:

- **The spectral intensity ratios move.** The first ratio shifted by about
  +0.07 in the same direction for every fluorophore.
- **The phasor coordinates barely move.** *G* and *S* stay put.

The shift is a near-uniform offset shared by all classes, not a reshuffling of
the classes relative to each other. That is why the fix works and why it is
cheap: the *features* are still good, only the *heads* are miscalibrated.

**Zero-shot transfer is not usable.** Applying the checkpoint to a new cell
line without adaptation gives accuracies in the tens of percent, and some
barcodes collapse to near zero. Do not report a zero-shot number as the
method's performance, and do not let a user conclude the model is broken from
one — it means the heads need recalibrating, nothing more.

## The recipe

Few-shot fine-tuning of **the two classifier heads only**.

| Item | Value |
|---|---|
| Frozen | every parameter whose name does not contain `fc_nu` or `fc_mito` — all six input stems **and** the shared backbone |
| Trained | `fc_nu` and `fc_mito` only |
| Head size | unchanged — the heads keep their original class count and their pre-trained weights; they are not resized or re-initialised for the new panel |
| Support set | *K* cells per barcode, sampled without replacement — balanced by construction, so no class weighting is applied |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Epochs | 30, fixed. No validation split, no early stopping |
| Batch size | 32, reshuffled each epoch |
| Loss | `CE(nucleus) + CE(mitochondria)`, unweighted, summed 1:1 |
| BatchNorm | left in **train** mode during fine-tuning, `eval` afterwards |
| Augmentation | none |

Fine-tuning the whole network instead of the heads is the fallback, at
`lr=1e-4`; it has not been needed.

### How many cells do you need to label?

**Accuracy saturates almost immediately — detection does not.** With a handful
of cells per barcode the labels the model commits to are already almost all
correct; what a larger support set buys is that the model becomes *willing to
commit* on more cells, especially in the weakest barcode. So:

- If you want correct labels on the cells the model is confident about, a few
  cells per barcode is enough.
- If you want a high fraction of cells to receive a call at all, budget a few
  tens per barcode.

Increasing the epoch count does not substitute for support cells: between 10
and 50 epochs the result moves by a few tenths of a percent.

### Evaluation protocol

Report both **within-batch** and **cross-batch**:

- **within-batch** — fine-tune on *K* cells from one dish, test on every
  remaining cell of that same dish, with the support cells explicitly excluded.
- **cross-batch** — fine-tune on one dish, test on the whole of a *different*
  dish.

Repeat over several support-set seeds and several dishes, and report
mean ± SEM. A small within-minus-cross gap is the evidence that the adaptation
generalises rather than memorising the dish it was tuned on.

Report **two numbers, never one**: accuracy (of the cells that got a call, how
many are right) and detection rate (what fraction of cells got a call at all).
They respond differently to *K*, and quoting only accuracy hides the fact that
a low-*K* model is confident about very little.

## Triage — the curation gate that comes first

"Triage" is not part of the fine-tuning; it is the curation stage that defines
an honest denominator. It removes exactly three categories:

1. **Segmentation fragments** — broken or partial masks.
2. **Non-expression** — the nuclear or the mitochondrial anchor is absent.
3. **Contamination** — a different barcode is present in the cell.

**Genuine look-alike confusions are kept.** Cells that are hard because two
barcodes really do resemble each other stay in, and the residual error rate is
biological ambiguity, not dirt. Dropping them would inflate the number by
several points and would be cherry-picking; if someone proposes removing
"cells the model gets wrong", that is the objection to raise.

Cells removed by triage are excluded from **both** the support pool and the
evaluation set.

In practice the gate is a quality threshold on mask size and peak intensity,
then a human review pass, then a single global keep/drop classifier used only
to auto-drop at high confidence. Per-class drop classifiers were tried and
rejected — their drop-precision was not good enough, and inspection showed most
of what they dropped was fine.

## Two traps

**Cached-feature training changes the BatchNorm mode.** Because the backbone is
frozen, its 512-d output can be computed once per dish and only the heads
trained on the cache — about twenty times faster. But caching runs the backbone
in `eval` mode where the direct path runs it in `train` mode. The two agree
once the support set is reasonably sized; at very small *K* the cached path
reads a couple of points **higher**. Any small-*K* claim must say which path
produced it.

**Some barcode pairs overlap by design.** At least one nucleus/mitochondria
pair is not separable in the 5-D feature space in any dataset, including the
model's own training data — it is separated only by *which organelle* the
signal is in, i.e. by the two heads being separate, never by the features. That
pair sets the floor on every metric, and no amount of fine-tuning moves it.
This is a property of the barcode panel, not a defect of the network.

## What is in this repository, and what is not

**The trainer that implements the protocol above is not here.** It lives
outside the public tree, together with the calibration adapter and the model
definition it imports.

`Train_LUMINA.py` in this repository has a `use_finetune` path, but it is
**not** this protocol and is disabled in the shipped configuration:

- `freeze_conv_layers()` freezes only `model.input_heads`; the backbone freeze
  is commented out — so it trains the backbone as well as the heads.
- It uses AdamW, 800 epochs, batch size 128 and early stopping.
- It has no *K*-shot support-set sampling, no confidence gate and no
  within/cross-batch evaluation.

So: describe the protocol on this page to a user who asks how to move LUMINA to
their own cells, and be straight with them that reproducing it means writing
the loop — freeze all but `fc_nu`/`fc_mito`, sample *K* per class, Adam 1e-3,
30 epochs, batch 32, summed CE — rather than calling a script that ships here.
It is a short loop; the recipe above is complete enough to write it.

Do not invent the missing numbers. If asked "how accurate is it", say what this
page says qualitatively and point at the manuscript for the measured values.
