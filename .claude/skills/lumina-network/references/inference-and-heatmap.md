# Inference, confidence, and the heatmap

---

## `Test_LUMINA.py` — how to run it

Two required path flags and a required choice of sample list; everything else is
optional with the shipped literal as its default:

```bash
python Test_LUMINA.py --checkpoint /path/to/best_model_fine-tune.pth \
    --data-root /path/to/dataset --samples sampleA,sampleB
```

Swap `--samples sampleA,sampleB` for `--all-samples` to score every qualifying
folder under the root(s). Passing neither exits 1; passing both exits 1.

| Flag | Default | Note |
|---|---|---|
| `--checkpoint` | **required** | Trained `state_dict`. A stage-1 checkpoint (`best_model_initial.pth`) is just a different path. |
| `--data-root` | **required** | Root holding one folder per sample. Unless `--out` is given, the two result workbooks are written **back into each sample folder**, which is where `Visualize_heatmap.py` looks for them. |
| `--data-root-2` | `''` | Optional second root, searched after the first — for `clustered.xlsx` and the unlabelled crop glob, both in `load_finetuning_data`, and for the output folder `test_model` writes into, but **not** by the image loader, which only ever reads `--data-root`. A sample found only under the second root gets enumerated and then fails to load. Keep one root complete rather than splitting a dataset across two; the asymmetry is long-standing behaviour, preserved deliberately. |
| `--samples` | **no default** — give this or `--all-samples` | Comma-separated sample folder names, scored in the order given. The shipped file scored a hand-edited list instead, so naming them here is the direct equivalent of editing that list. A name that is not a folder under either root is a hard error, because it would otherwise be scored as zero cells and reported as "Total predictions: 0", which reads like a data problem. |
| `--all-samples` | off | Score every immediate subfolder of the root(s) holding a `clustered.xlsx` or one of the `--seg-folder` candidates — the same two branches `load_finetuning_data` takes. Unless `--out` redirects them, this **overwrites** both workbooks in every one of those folders. It is a separate flag rather than the empty default for exactly that reason; `resolve_samples` mirrors `Data_prep.py`'s. The resolved destinations print one per line, followed by a line saying nothing has been written yet, before the first workbook. |
| `--seg-folder` | `auto` | `auto` takes `seg_5D_calib` when present, `seg_5D` otherwise — the `SEG_FOLDER_ORDER` table at the top of the file spells out all three choices, and `auto` is what this script has always done. The two folders are **not** interchangeable; on a dish holding both, `auto` feeds the network a different input distribution than `seg_5D`. The folder actually read is printed per sample. |
| `--out` | `''` = in place | Empty writes the workbooks into each sample folder. Given, results land in `<out>/<sample>/`, which is created if absent, and `Visualize_heatmap.py` must be pointed at `<out>` instead of at `--data-root`. |
| `--num-classes` | `8` | Must match the checkpoint, or `load_state_dict` fails on `fc_nu.6.weight` / `fc_mito.6.weight`. Index 0 is reserved for "no anchor" and is never a class name. |
| `--crop-size` | `256` | Canvas each crop is centre-padded onto; the literal lives in the module-level `CROP_SIZE`. A **larger** crop is not resized: `FluorescenceDataset.__getitem__` skips it with `idx = (idx + 1) % len(self.df); continue`, which shifts every following `Cell_Label` in that folder. Leave it at the value the checkpoint was trained with. |
| `--device` | `cuda:0` | No automatic fallback — cuda on a machine without a GPU fails rather than silently running slowly on the CPU. |
| `--confidence-threshold` | `0.6` | Both the gate and part of the output file name. See below. |

Every run also writes `test_run_config.csv` (into `--out` if given, else into
`--data-root`) with every flag resolved, plus `resolved_seg_order`,
`resolved_samples` and `workbook_suffix` — the last being exactly how the
threshold was spelled into the file names, which is what `Visualize_heatmap.py`
has to match.

Per sample folder, the run: build the row list (from `clustered.xlsx` if
present, else by globbing the crops), load each crop, pad to `--crop-size`,
normalise the last plane, forward, softmax each head, score, and file the result
as confident or uncertain. `test_model` calls `model.eval()` before the loop, so
dropout is off and batch-norm uses its running statistics. Inference always runs
at `DataLoader(..., batch_size=1, shuffle=False)`, which the confidence helper
depends on; that is not a flag.

---

## The confidence score

`calculate_confidence_score`, nested inside `test_model`, is applied to each
head's softmax vector separately:

```
margin  = p(1) - p(2)                       # top minus runner-up
H       = -Σ p·ln(p + 1e-10)                # entropy over ALL outputs
H_norm  = 1 - H / (-ln(1/7))                # max_entropy = -np.log(1 / 7) = 1.9459
ratio   = p(1) / mean(p(2..n))

score   = 0.4·margin + 0.3·H_norm + 0.3·min(ratio/10, 1)
```

A cell is written to the **confident** workbook only when *both* heads score at
or above `--confidence-threshold` — the `if nu_reliable and mito_reliable:`
branch; otherwise the whole row goes to the
**uncertain** workbook. There is no per-head split of the output.

The three weights, the `/10` cap and the `1/7` in the entropy normaliser are
**not** flags — they are the score's definition, and changing one would make two
runs' thresholds incomparable while both still printing "0.6".

Things to know before quoting a score:

- **It is not a probability.** It is an ad-hoc weighted blend on roughly the
  interval 0–1, and 0.6 is a point on that scale, not 60 % confidence.
- **The entropy term is normalised against 7 classes while the vector has 8**
  (`--num-classes 8` includes the unused index 0). When the distribution is
  near-uniform over eight outputs, `H_norm` goes slightly negative — about
  −0.07, so ≈ −0.02 on the score. Harmless in practice, but it means the score
  can dip just below zero.
- **The ratio term saturates early.** With 8 outputs, `mean(p(2..n)) =
  (1−p(1))/7`, so `ratio ≥ 10` — the cap — as soon as the top probability
  passes ≈ 0.59. Above that, only the margin and entropy terms still move.
- **`Finetune_LUMINA.py` defaults the same flag name to 0.9.** That is not a
  typo; they are different measurements on the same scale.

Worked examples, computed from the formula above (arithmetic, not measurements):

| Softmax | margin term | entropy term | ratio term | score | at 0.6 |
|---|---|---|---|---|---|
| one-hot | 0.400 | 0.300 | 0.300 | 1.000 | confident |
| 0.9, rest 0.0143 | 0.354 | 0.220 | 0.300 | 0.874 | confident |
| two-way tie 0.5 / 0.5 | 0.000 | 0.193 | 0.210 | 0.403 | uncertain |

A clean two-way tie scores 0.40, so a threshold of 0.6 does reject a coin flip
between two barcodes — which is the failure this score exists to catch.

---

## The two output workbooks

Written into the **sample folder itself** (or under `--data-root-2` if the sample
is not under `--data-root`), or into `<--out>/<sample>/` when `--out` is given —
the three-line `out_folder` cascade at the top of `test_model`'s per-sample loop:

```
<sample>/predict_class_confident_<threshold>.xlsx
<sample>/predict_class_uncertain_<threshold>.xlsx
```

At the default threshold that is `predict_class_confident_0.6.xlsx`. **The
threshold is in the name**, so `--confidence-threshold 0.7` produces a *new pair
of files*, not an overwritten one — and `Visualize_heatmap.py` run at its own
default will then not find them. Run both scripts with the same value. (The two
"results saved to" console messages print the names *without* the suffix; the
files on disk have it. Trust the disk, or `workbook_suffix` in
`test_run_config.csv`.)

| Column | Contents |
|---|---|
| `Directory` | the sample folder |
| `Cell_Label` | the `<label>` in `cell<label>_5D.tif` |
| `Predicted_Nu_Class` | class **name** (`'N10'`…), or `'Unknown'` for index 0 — the `next((k for k, v in nu_class_map.items() if v == pred_nu), 'Unknown')` lookup |
| `Predicted_Mito_Class` | same for the mitochondrial head |
| `Nu_Confidence`, `Mito_Confidence` | the scores above, written as **strings** formatted to 3 decimals — cast them before comparing |
| `Nu_Probabilities`, `Mito_Probabilities` | the full softmax vector, 8 entries, as a Python list rendered into the cell |

Neither workbook carries a ground-truth column: inference does not need labels
and does not copy over the ones in `clustered.xlsx`. Comparing predictions with
truth is a join on `Directory` + `Cell_Label` that you have to write.

```python
import pandas as pd, ast
df = pd.read_excel('predict_class_confident_0.6.xlsx')
df['Nu_Confidence'] = df['Nu_Confidence'].astype(float)
print(df.groupby(['Predicted_Nu_Class', 'Predicted_Mito_Class']).size())
print(pd.DataFrame(df['Nu_Probabilities'].map(ast.literal_eval).tolist()).round(3).head())
```

`Cell_Label` is taken from `test_df.iloc[i]` where `i` is the loader's own
counter, not the row the image came from — see `troubleshooting.md` for when that
stops lining up with the image that was actually classified.

The repository README quotes "~1 second" per cell for inference and "typically
3 minutes" including preprocessing. Those are the README's numbers, not the
code's; no script in this folder times itself.

---

## `Visualize_heatmap.py`

```bash
python Visualize_heatmap.py --data-root /path/to/dataset --no-gui
```

| Flag | Default | Note |
|---|---|---|
| `--data-root` | **required** | Must be the **same** root `Test_LUMINA.py` was given (or its `--out`), because that is where the workbooks were written. |
| `--data-root-2` | `''` | Searched only when a sample folder is not found under `--data-root`. A dataset split across two roots is a common cause of "Confident Excel file not found". |
| `--samples` | `''` = scan | Sample folder names, **in the order given**. Empty means every folder under the root(s) that already holds a confident workbook at this threshold. The shipped file plotted whatever short list was left uncommented, so pass `--samples` to reproduce a particular figure. |
| `--out-pdf` | `heatmap_nu_mito.pdf` | A bare filename lands in the current working directory and is overwritten every run (`fig.savefig(output_pdf, format='pdf')` at the end of `create_heatmap`). `visualize_heatmap_run_config.csv` is written beside it. |
| `--nu-classes` | `''` = `NU_CLASS_MAP` order | Row order **and filter**. |
| `--mito-classes` | `''` = `MITO_CLASS_MAP` order | Column order and filter. |
| `--confidence-threshold` | `0.6` | Nothing is gated here — this value is only interpolated into the module-level `CONFIDENT_XLSX` / `UNCERTAIN_XLSX` **file-name** templates, so it must equal the value `Test_LUMINA.py` was run with or no workbook is found. |
| `--no-gui` | off | Write the PDF and exit instead of opening the interactive Tk window. Required on a machine with no display. |

`tkinter` and the TkAgg backend are imported inside the GUI branch of `main`
(and again inside `HeatmapGUI.create_widgets`), not at module scope, so importing
this module is side-effect free and `--no-gui` works headless. On the `--no-gui`
path `main` calls `create_heatmap` directly; on the GUI path the same call happens
inside `HeatmapGUI.create_widgets`, so the PDF is written either way.

Everything else about the figure is a literal and not a flag: `figsize=(7, 6)`,
the annotation/title/axis/tick font sizes, the `Blues` and `gray` colormaps, and
the colorbar shrink.

**What the figure shows.** Rows are the predicted *nuclear* class, columns the
predicted *mitochondrial* class, and each entry counts confident cells with
that pair. Both axes are predictions — **there is no ground truth in this
figure.** It is a co-occurrence table of the dual-anchor calls, not a confusion
matrix. Do not describe it as one, and do not read the diagonal as "correct".

The diagonal is drawn on a separate grey colormap (`np.fill_diagonal(mask, True)`
and the `mask` / `~mask` pair of heatmaps in `create_heatmap`); with the two class
lists in matching order it
picks out `N10-M10`, `N13-M13`, … — the cells carrying the *same* fluorophore
on both anchors. Reordering `--nu-classes` and not `--mito-classes` makes that
diagonal meaningless with no error. Whether those cells are expected or are a
sign of bleed-through between the anchors is a question about the dish, not
about the plot.

**Detection rate** = confident / (confident + uncertain), printed per sample by
`process_directory`, pooled across all listed samples by `main` on its `Overall:`
line, and written into the figure title by `create_heatmap` and into the window
title by `HeatmapGUI`. It measures how many
cells cleared the confidence threshold, and says nothing about whether the calls
are right.

**Two counts that can disagree.** `create_heatmap` prints `Total confident cells
plotted: {len(data)}` but only increments the matrix inside
`if nu_fp in nu_FPs and mito_fp in mito_FPs:`. Cells predicted `'Unknown'`,
or predicted into a class you removed with `--nu-classes` / `--mito-classes`, are
counted in the printed total and missing from the matrix. If the entries do not
sum to the printed number, that is why — and it is why those two flags filter as
well as order.
