# Training: the two stages as the code implements them

`Train_LUMINA.py` is driven entirely by flags. Every number below is a flag's
default, and every default is the literal the committed file held, so a run that
passes only the required path flags computes what the committed file computed.
`--help` is authoritative; the code below is named by function, class or
statement rather than by line number, so a citation still finds its target after
the file is edited.

A minimal invocation:

```bash
python Train_LUMINA.py --data-root /path/to/single_anchor \
    --single-anchor-manifest single_anchor.csv \
    --dual-root /path/to/dual_anchor --dual-samples-file dual_samples.txt \
    --checkpoint best_model_fine-tune.pth --out ./train_out
```

Four flags are required — `--data-root`, `--single-anchor-manifest`,
`--dual-root`, `--out` — and two more are give-one-or-the-other rather than
required, so the script names the missing one instead of failing on a path:

| Give one of | Or | Because |
|---|---|---|
| `--dual-samples a,b,c` | `--dual-samples-file list.txt` | which sample list trained a checkpoint has to be unambiguous, so passing both is an error. **The order is part of the list**, not just the names — see stage 2 |
| `--checkpoint model.pth` | `--from-scratch` | `--from-scratch` ignores a checkpoint entirely; passing both is an error |

---

## The network

`DualHeadConvNet` (an identical copy of the class lives in `Test_LUMINA.py` —
the two must be edited together, and `Finetune_LUMINA.py` imports the
`Test_LUMINA.py` copy rather than carrying a third).

| Stage | Layers | Output for a 256×256 input |
|---|---|---|
| 6 input stems, one per plane | `Conv(1→16,3×3) BN ReLU`, `Conv(16→32) BN ReLU`, `Conv(32→64) BN ReLU`, `MaxPool2` | 6 × (64, 128, 128) |
| concatenate | `torch.cat(dim=1)` | (384, 128, 128) |
| trunk | `ResNetBlock(384→384)`, `MaxPool2`, `ResNetBlock(384→512)`, `MaxPool2`, `ResNetBlock(512→512)`, `AdaptiveAvgPool2d(1,1)` | (512, 1, 1) |
| `fc_nu` | `512→512 ReLU Drop(0.5) →256 ReLU Drop(0.5) →num_classes` | 8 logits |
| `fc_mito` | same, separate weights | 8 logits |

The **plane count is fixed at six** by the `for _ in range(6)` that builds
`self.input_heads` in `DualHeadConvNet.__init__` and by the
`[head(x[:, i:i + 1]) for i, head in enumerate(self.input_heads)]` in `forward`.
It is not a flag. The
`height`/`width` constructor arguments are accepted and unused — the adaptive
pool makes the trunk size-agnostic, so only the loaders enforce `--crop-size`.

The two heads share every feature and predict independently. Nothing couples
them except the summed loss, so the model can and will emit combinations that
never occurred in training.

---

## The switches, and what the defaults actually do

| Flag | Default | Effect |
|---|---|---|
| `--from-scratch` | off, i.e. **load a checkpoint** | Off means `--checkpoint` is loaded through `model.load_state_dict` and **stage 1 is skipped entirely**. On means a random initialisation — and, together with `--finetune`, also runs stage 1, which does not work in this file; read the warning below. |
| `--finetune` | off | Guards two different things: whether stage 1 runs at all (only reachable under `--from-scratch`), and whether the six input stems are frozen and the optimizer rebuilt before stage 2 (the `if use_finetune:` block just before the stage-2 `train_model` call). Off, **nothing is frozen** and stage 2 trains every parameter. |
| `--num-classes` | `8` | Width of both heads. Indices 0–7; index 0 is never a training target. Must match the checkpoint. The script refuses a value below what the class maps need. |
| `--batch-size` | `128` | Used by all four loaders. Freezing the stems with `--finetune` leaves room for a larger batch. |
| `--epochs` | `180` | **Stage 1 only**, and stage 1 runs only under `--from-scratch --finetune`. |
| `--finetune-epochs` | `800` | **Stage 2**, which is the run you get by default. Kept separate from `--epochs` on purpose: one flag for both would silently hand stage 2 the stage-1 budget. |
| `--early-stop-patience` | `1000` | Larger than either epoch budget, so **early stopping never fires** unless lowered. The checkpoint is still written on every val-loss improvement, so the file on disk is the best epoch, not the last. |
| `--device` | `cuda:0` | No CPU fallback and no auto-detect: a machine without a GPU fails here rather than quietly starting a run that would never finish. |
| `--crop-size` | `256` | The canvas both loaders pad to. A larger crop is skipped, not resized. |
| `--seg-folder` | `seg_5D` | Read under **both** roots. `Data_prep.py` writes `seg_5D_calib` by default, so one of the two flags has to move. There is deliberately no `auto` here (`Test_LUMINA.py` and `Finetune_LUMINA.py` have one): a checkpoint trained on a mixture of calibrated and uncalibrated crops would not be reproducible from the recorded flags. |

So a run with **no optional flags** is: build the model, load `--checkpoint`,
freeze nothing, and train all parameters on the dual-anchor split for up to 800
epochs under `phase = 'fine-tune'`.

That is not the recipe the README describes ("pre-trained on single-anchor
data, then fine-tuned on dual-anchor data with the feature extractor frozen").
The recipe is what the code *supports*; the defaults are one particular
continuation run. Say which one you mean.

The startup banner prints which stages will run, which lr-drops apply, whether
early stopping can fire at all, and whether the stems will be frozen. Read it
back to a user rather than reasoning about the flag combination in your head.

---

## Stage 1 — pre-training on single-anchor data

**Data.** `load_training_data` walks every `*_5D.tif` in
`<--data-root>/<folder>/<--seg-folder>` for each folder listed under a class key.
A nucleus-only folder yields rows with `nu_class = k, mito_class = 0`; a
mitochondria-only folder yields `nu_class = 0, mito_class = k`. Split by
`--val-split` (default 0.2) with `random_state=--seed` (default 42), in the first
of `main`'s two `train_test_split` calls.

Which folder holds which class comes from `--single-anchor-manifest`, a CSV of
`class,folder`. Several rows may name the same class. **Row order does not
matter** — `load_single_anchor_manifest` returns the dicts in
`NU_CLASS_MAP` / `MITO_CLASS_MAP` order, so the order of rows in the file cannot
move the `--seed` split, and a class name outside the panel is an error rather
than a silent class 0.

The manifest is read and walked on **every** run, including the default one in
which stage 1 never executes. That is what this script does today and it is not
changed; it means a bad `--data-root` fails a default run even though the data is
never trained on.

**This path cannot run as written.** The dataset skips any row where either
label is 0 by advancing to the next index and `continue`-ing inside `while True`
(in `FluorescenceDataset.__getitem__`) — and by construction *every* row from
`load_training_data` has one.
So the first batch never assembles: the process spins with no error and no
output. The script prints an explicit warning before entering that path, and
`troubleshooting.md` covers it. Do not tell a user to "just pass `--from-scratch
--finetune`" without repeating that warning.

**Intended shape of the stage.** Single-anchor cells teach the stems and trunk
what one fluorophore looks like in G/S and in the three ratios; the absent
anchor carries class 0, which is why index 0 exists in both heads. Whether
class 0 was meant to be trained as a real "no anchor" target or filtered out is
not something the file settles — the rows are built with it and then skipped by
the loader. The checkpoint this stage would write is
`<--out>/best_model_initial.pth`, and its log is `train_val_log_pre.xlsx`,
written only on that branch.

---

## Stage 2 — fine-tuning on dual-anchor data

**Data.** `load_finetuning_data` reads
`<--dual-root>/<sample>/clustered.xlsx` for every sample named by
`--dual-samples` / `--dual-samples-file`, and keeps only cells where **both**
`Nu_FP` and `Mito_FP` map to a known class. Both a missing sample folder and a
missing `clustered.xlsx` raise `SystemExit` rather than being skipped: skipping
one silently would train on a smaller population than the one named on the
command line.

**The list is loaded in the order you give it, and the order is part of the
split.** `main` does `test_dirs = list(given_dirs)` — no sort, no de-duplication —
`load_finetuning_data` appends rows folder by folder in that order, and the split
below is taken by row position. So `--dual-samples a,b` and `--dual-samples b,a`
hold out different cells at the same `--seed`, and therefore write a different
`val_df.xlsx` and a different checkpoint. A rerun has to pass the same names in
the same order; the resolved order is printed at startup, under a line saying
`ORDER MATTERS`, and recorded as `resolved_dual_samples` in
`train_run_config.csv`. The list is not sorted on purpose: the hand-written list
this script shipped with was not in sorted order, so sorting would mean the file
no longer reproduces the split, and therefore the checkpoint, that the shipped
script produced — with no flag to ask for the original back.

**Split.** the second `train_test_split(test_df, test_size=--val-split,
random_state=--seed)` in `main`. This splits **cells**, not acquisitions — cells
from the same dish land on both sides, so the validation accuracy is not a
held-out-acquisition estimate. Say so when quoting it. The validation rows are
written to `<--out>/val_df.xlsx` so the same split can be reproduced later.

`--dual-root` is **training data, not a held-out set**. The only cells held out
are the `--val-split` fraction of these same folders.

**Freezing.** `freeze_conv_layers`, defined just above the stage-2 call, sets
`requires_grad = False` on the **six input stems only**; the shared trunk and both
heads keep training. It is called only under `--finetune`, and is followed
immediately by rebuilding the optimizer over the surviving parameters — the two
belong together,
because an optimizer built before the freeze would still hold state for
parameters that no longer receive gradients.

---

## The loss

`WeightedCrossEntropyLoss`: one `nn.CrossEntropyLoss` per head,
and the total is their **unweighted sum** — the two anchors count equally. The
1:1 ratio is not a flag.

Within a head, class *c* gets

```
w_c = total_samples / (n_classes_present * count_c)
```

so a class with half the average number of cells contributes twice as much per
cell. Absent classes — and index 0 — keep weight 0.

The counts come from `get_class_counts(train_loader)`, called at the top of
`train_model`, which **iterates the whole training loader once before epoch 1**,
reading every crop from disk to count labels. On a slow filesystem that is a
silent delay before training appears to start; the line
"Calculating class weights..." is printed just before it.

The weight vector is built as `torch.zeros(max(class_counts.keys()) + 1)`,
once per head in `WeightedCrossEntropyLoss.__init__`. If the highest class index present in the training data is
smaller than `--num-classes - 1`, the vector is shorter than the logit dimension
and `CrossEntropyLoss` raises — see `troubleshooting.md`.

---

## Optimizer and schedule

`optim.AdamW(model.parameters(), lr=--lr)` (`--lr` defaults to `1e-3`), rebuilt
over trainable parameters only under `--finetune`. `weight_decay` is
left at torch's AdamW default and is deliberately **not** a flag: this script has
never overridden it. No gradient clipping, no augmentation (`transform` exists on
the dataset and is never passed), no `num_workers` on the loaders.

Learning rate is stepped by hand at the top of `train_model`'s epoch loop, in the
`if phase == 'initial' and epoch in lr_drop_epochs_initial` / `elif phase ==
'fine-tune'` pair, multiplied by `--lr-drop-factor` (default 0.2) at 0-based
epochs:

| Phase | Flag | Default | Printed as |
|---|---|---|---|
| `'initial'` | `--lr-drop-epochs-initial` | `30,70` | epoch 31, 71 |
| `'fine-tune'` | `--lr-drop-epochs-finetune` | `200,350,500` | epoch 201, 351, 501 |

The epoch index in the code is 0-based; the progress line prints `epoch + 1`. The two lists are kept separate because the phase selects which one
applies — one flag for both would step stage 2 at stage 1's epochs.

---

## What a run writes, and how to read it

All in `--out`, which is created if absent. A second run pointed at the same
directory overwrites `test_train_val_log.xlsx` and `val_df.xlsx` and appends
sheets to `combination_accuracies_<phase>.xlsx`.

| File | Contents |
|---|---|
| `best_model_<phase>.pth` | Bare `state_dict`, saved by the `torch.save` that runs on every improvement in **validation loss**. No epoch, no class map, no optimizer state — it cannot be resumed from, only loaded. |
| `combination_accuracies_<phase>.xlsx` | One sheet per epoch, named `epoch_<n>`, plus a placeholder `Sheet1` created on the first run, by the `if not os.path.exists(combine_acc_path)` block at the top of `train_model`. |
| `test_train_val_log.xlsx` | `Epoch, Train Loss, Validation Loss, Train Accuracy, Validation Accuracy`, rewritten in full every epoch. The name carries **no phase**, so a second run in the same folder overwrites the first. |
| `val_df.xlsx` | The `--val-split` validation rows of the dual-anchor split: `Directory, Cell_Label, Nu_cluster, Mito_cluster`. |
| `train_run_config.csv` | Every flag and its resolved value, plus `resolved_nu_class_map` / `resolved_mito_class_map` and the resolved dual-sample list. This is the durable record of what produced the checkpoint — read it before quoting any number from the directory. |

**Accuracy means both heads.** `(pred_nu == nu_labels) & (pred_mito ==
mito_labels)`, computed once in the train loop and once in the validation loop —
the combination is right or the cell is
wrong. A model at 0.9 on each anchor independently would not print 0.9 here.

**`combination_accuracies_*.xlsx` is the confusion information**, one row per
*true* combination:

| Column | Meaning |
|---|---|
| `True_Nu`, `True_Mito` | the true pair, formatted `N3` / `M5` — these are **class indices**, not the `N10`-style names |
| `Total_Samples`, `Correct_Predictions`, `Accuracy` | for that true pair, this epoch, on the validation set |
| `Top1_Pred … Top3_Pred` | the three most frequent *predicted* pairs, formatted `N3-M5`, with counts |

Rows are sorted by accuracy then sample count, descending, so the
worst-performing combinations sit at the bottom of the sheet. When one pair's
`Top1_Pred` is consistently a different pair, that is the pair being confused,
and it is the number to quote — not the headline accuracy.

Reading it:

```python
import pandas as pd
sheets = pd.read_excel('combination_accuracies_fine-tune.xlsx', sheet_name=None)
last = sheets[sorted((k for k in sheets if k.startswith('epoch_')),
                     key=lambda s: int(s.split('_')[1]))[-1]]
print(last.sort_values('Accuracy').head(10))
```

Note that the workbook is reopened and rewritten in append mode every epoch
(the `pd.ExcelWriter(..., mode='a', if_sheet_exists='replace')` near the end of
the epoch loop), so it accumulates one sheet per epoch — 800 sheets at the
default `--finetune-epochs` — and each write costs more than the last. Lowering
`--finetune-epochs` lowers that cost proportionally.
