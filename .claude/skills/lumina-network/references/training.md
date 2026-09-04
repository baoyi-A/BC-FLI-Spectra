# Training: the two stages as the code implements them

`Train_LUMINA.py` has no arguments. Everything below is a literal in `main()`
or in `train_model`, quoted with its line number.

---

## The network

`DualHeadConvNet` (lines 113–184; an identical copy lives in `Test_LUMINA.py`
lines 108–179 — the two must be edited together).

| Stage | Layers | Output for a 256×256 input |
|---|---|---|
| 6 input stems, one per plane | `Conv(1→16,3×3) BN ReLU`, `Conv(16→32) BN ReLU`, `Conv(32→64) BN ReLU`, `MaxPool2` | 6 × (64, 128, 128) |
| concatenate | `torch.cat(dim=1)` | (384, 128, 128) |
| trunk | `ResNetBlock(384→384)`, `MaxPool2`, `ResNetBlock(384→512)`, `MaxPool2`, `ResNetBlock(512→512)`, `AdaptiveAvgPool2d(1,1)` | (512, 1, 1) |
| `fc_nu` | `512→512 ReLU Drop(0.5) →256 ReLU Drop(0.5) →num_classes` | 8 logits |
| `fc_mito` | same, separate weights | 8 logits |

The **plane count is fixed at six** by the `range(6)` on line 130 and the
`x[:, i:i+1]` slicing in `forward`. The `height`/`width` constructor arguments
are accepted and unused — the adaptive pool makes the trunk size-agnostic, so
only the loaders enforce 256×256.

The two heads share every feature and predict independently. Nothing couples
them except the summed loss, so the model can and will emit combinations that
never occurred in training.

---

## The switches, and what the shipped values actually do

| Line | Name | Shipped | Effect |
|---|---|---|---|
| 623 | `use_pretrained` | `True` | **Skips stage 1 entirely** and loads `pre_model_dir/best_model_fine-tune.pth` (line 628). Two other `load_state_dict` calls, including one for `best_model_initial.pth`, are commented out on lines 625–626. |
| 499 | `use_finetune` | `False` | Guards two different things: whether stage 1 runs at all (line 630, only reachable when `use_pretrained` is `False`), and whether the stems are frozen and the optimizer rebuilt before stage 2 (lines 655–658). At `False`, **nothing is frozen** and stage 2 trains every parameter. |
| 618 | `num_classes` | `8` | Both heads. Indices 0–7; index 0 is never a training target. |
| 492 | `batchsize` | `128` | Line 493 notes 224 fits when the stems are locked and 256 may OOM — that comment is the only memory guidance in the file, and it is not a measurement you should quote as one. |
| 494 / 660 | `num_epochs` | `180`, then **reassigned to 800** on line 660 | 180 applies to stage 1 only. Stage 2 always gets 800. |
| 495 | `early_stop_patience` | `1000` | Larger than either epoch count, so **early stopping never fires**. The checkpoint is still written on every val-loss improvement, so the file on disk is the best epoch, not the last. |
| 496–497 | `gpu_id`, `device` | `0`, `cuda:0` | Hardcoded. There is no CPU path — line 619 moves the model to `cuda:0` before anything else. |

So a run of the file **as committed** is: build the model, load a fine-tuned
checkpoint from `pre_model_dir`, do not freeze anything, and train all
parameters on the dual-anchor split for up to 800 epochs under `phase =
'fine-tune'`.

That is not the recipe the README describes ("pre-trained on single-anchor
data, then fine-tuned on dual-anchor data with the feature extractor frozen").
The recipe is what the code *supports*; the committed constants are one
particular continuation run. Say which one you mean.

---

## Stage 1 — pre-training on single-anchor data

**Data.** `load_training_data(nu_files, mito_files, base_folder)` (lines
186–219) walks every `*_5D.tif` in `<base_folder>/<folder>/seg_5D` for each
folder listed under a class key. A nucleus-only folder yields rows with
`nu_class = k, mito_class = 0`; a mitochondria-only folder yields
`nu_class = 0, mito_class = k`. Split 80/20 with `random_state=42` (line 598).

**This path cannot run as written.** The dataset skips any row where either
label is 0 (`Train_LUMINA.py` lines 69–72) — and by construction *every* row
from `load_training_data` has one. `__getitem__` advances to the next index and
loops, so the first batch never assembles: the process spins at 100 % CPU with
no error and no output. See `troubleshooting.md`. Do not tell a user to "just
set `use_pretrained = False`" without warning them about this.

**Intended shape of the stage.** Single-anchor cells teach the stems and trunk
what one fluorophore looks like in G/S and in the three ratios; the absent
anchor carries class 0, which is why index 0 exists in both heads. Whether
class 0 was meant to be trained as a real "no anchor" target or filtered out is
not something the file settles — the rows are built with it and then skipped by
the loader. The checkpoint this stage would write is
`out_folder/best_model_initial.pth`, and its log is `train_val_log_pre.xlsx`
(line 644).

---

## Stage 2 — fine-tuning on dual-anchor data

**Data.** `load_finetuning_data(test_dirs, test_base_folder, nu_class_map,
mito_class_map)` (lines 221–245) reads `clustered.xlsx` per folder and keeps
only cells where **both** `Nu_FP` and `Mito_FP` map to a known class (line 234).
`test_dirs` (lines 500–562) is a long explicit list of dual-anchor
acquisitions, all but one active — line 506 is commented out without a reason.
(Several folders are annotated "not so sure about the gt" in the parallel list
inside `Test_LUMINA.py`; those annotations do not appear here.)

**Split.** `train_test_split(test_df, test_size=0.2, random_state=42)`, line
601. This splits **cells**, not acquisitions — cells from the same dish land on
both sides, so the validation accuracy is not a held-out-acquisition estimate.
Say so when quoting it. The validation rows are written to
`out_folder/val_df.xlsx` (line 604) so the same split can be reproduced later.

**Freezing.** `freeze_conv_layers` (lines 648–653) sets `requires_grad = False`
on the **six input stems only**. The loop that would also freeze the trunk is
commented out on lines 652–653, so with freezing on, the ResNet trunk and both
heads still train. It is called only when `use_finetune` is `True` (line 656),
and is followed by rebuilding the optimizer over the surviving parameters
(line 658) — if you enable freezing by hand, keep that rebuild, or the
optimizer will still hold state for parameters that no longer receive gradients.

---

## The loss

`WeightedCrossEntropyLoss` (lines 248–269): one `nn.CrossEntropyLoss` per head,
and the total is their **unweighted sum** — the two anchors count equally.

Within a head, class *c* gets

```
w_c = total_samples / (n_classes_present * count_c)
```

so a class with half the average number of cells contributes twice as much per
cell. Absent classes — and index 0 — keep weight 0.

The counts come from `get_class_counts(train_loader)` (line 291), which
**iterates the whole training loader once before epoch 1**, reading every crop
from disk to count labels. On a slow filesystem that is a silent delay before
training appears to start.

The weight vector is built as `torch.zeros(max(class_counts.keys()) + 1)`. If
the highest class index present in the training data is smaller than
`num_classes - 1`, the vector is shorter than the logit dimension and
`CrossEntropyLoss` raises — see `troubleshooting.md`.

---

## Optimizer and schedule

`optim.AdamW(model.parameters(), lr=1e-3)`, line 620, rebuilt over trainable
parameters only if freezing is enabled (line 658). No weight-decay override, no
gradient clipping, no augmentation (`transform` exists on the dataset and is
never passed), no `num_workers` on the loaders.

Learning rate is stepped by hand at the top of the epoch loop (lines 315–320),
`lr *= 0.2` at 0-based epochs:

| Phase | Decay at epochs | Printed as |
|---|---|---|
| `'initial'` | 30, 70 | epoch 31, 71 |
| `'fine-tune'` | 200, 350, 500 | epoch 201, 351, 501 |

The epoch index in the code is 0-based; the progress line prints `epoch + 1`.
A `ReduceLROnPlateau`-style call is commented out on line 414.

---

## What a run writes, and how to read it

All in `out_folder` (line 487, shipped as
`/gpfs/share/home/2301112465/BC_FLIM/Hek293T/Dual_241203-3`).

| File | Contents |
|---|---|
| `best_model_<phase>.pth` | Bare `state_dict`, saved on every improvement in **validation loss** (line 419). No epoch, no class map, no optimizer state — it cannot be resumed from, only loaded. |
| `combination_accuracies_<phase>.xlsx` | One sheet per epoch, named `epoch_<n>`, plus a placeholder `Sheet1` created on the first run. |
| `test_train_val_log.xlsx` | `Epoch, Train Loss, Validation Loss, Train Accuracy, Validation Accuracy`, rewritten in full every epoch. The name carries **no phase**, so a second run in the same folder overwrites the first. |
| `val_df.xlsx` | The 20 % validation rows of the dual-anchor split: `Directory, Cell_Label, Nu_cluster, Mito_cluster`. |

**Accuracy means both heads.** `(pred_nu == nu_labels) & (pred_mito ==
mito_labels)` (lines 344 and 373) — the combination is right or the cell is
wrong. A model at 0.9 on each anchor independently would not print 0.9 here.

**`combination_accuracies_*.xlsx` is the confusion information**, one row per
*true* combination:

| Column | Meaning |
|---|---|
| `True_Nu`, `True_Mito` | the true pair, formatted `N3` / `M5` — these are **class indices**, not the `N10`-style names |
| `Total_Samples`, `Correct_Predictions`, `Accuracy` | for that true pair, this epoch, on the validation set |
| `Top1_Pred … Top3_Pred` | the three most frequent *predicted* pairs, formatted `N3-M5`, with counts |

Rows are sorted by accuracy then sample count, so the worst-performing
combinations sit at the bottom of the sheet. When one pair's `Top1_Pred` is
consistently a different pair, that is the pair being confused, and it is the
number to quote — not the headline accuracy.

Reading it:

```python
import pandas as pd
sheets = pd.read_excel('combination_accuracies_fine-tune.xlsx', sheet_name=None)
last = sheets[sorted((k for k in sheets if k.startswith('epoch_')),
                     key=lambda s: int(s.split('_')[1]))[-1]]
print(last.sort_values('Accuracy').head(10))
```

Note that the workbook is reopened and rewritten in append mode every epoch
(lines 462–466), so it accumulates one sheet per epoch — 800 sheets over a full
stage-2 run — and each write costs more than the last.
