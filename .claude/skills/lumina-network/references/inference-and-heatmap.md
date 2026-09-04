# Inference, confidence, and the heatmap

---

## `Test_LUMINA.py` — what to edit

No arguments. The constants live in `main()`:

| Line | Name | Shipped | Note |
|---|---|---|---|
| 346 | `base_folder` | `G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual` | Primary sample root. Hardcoded lab path on a removable drive. |
| 347 | `base_folder2` | `I:\BC-FLIM\Hek293T-BJMU-Dual` | Second root, searched for `clustered.xlsx`, for the unlabelled `seg_5D` glob, and for the output folder — but **not** by the image loader. Keep one root complete rather than splitting a dataset across the two. |
| 351 | `model_folder` | `G:\…\Dual_241127` | Six other checkpoints are commented out on lines 348–355. |
| 356 | `model_path` | `<model_folder>/best_model_fine-tune.pth` | Fixed file name; to load a stage-1 checkpoint, edit this line. |
| 363–364 | `gpu_id`, `device` | `0`, `cuda:0` | Hardcoded, no CPU path. |
| 366 | `confidence_threshold` | `0.6` | Passed to `test_model` on line 542. The function's own default is `0.5` (line 227) and is never used. **The value ends up in the output file name.** |
| 396–500 | `test_dirs` | 2 active, ~90 commented out | Sample folders to classify. |
| 501–528 | `nu_files`, `mito_files` | 7 + 7 | Present only to rebuild the class maps (lines 531–532). The file lists themselves are not read at inference. |
| 537 | `num_classes` | `8` | Must match the checkpoint, or `load_state_dict` fails on the two final layers. |

`batchsize`, `num_epochs`, `early_stop_patience` and `use_finetune` (lines
360–365) are declared in `main()` and never used — inference always runs at
`batch_size=1` (line 284), which the confidence helper depends on.

Per sample folder, the run: build the row list (from `clustered.xlsx` if
present, else by globbing the crops), load each crop, pad to 256×256, normalise
the last plane, forward, softmax each head, score, and file the result as
confident or uncertain. `model.eval()` is set on line 228, so dropout is off and
batch-norm uses its running statistics.

---

## The confidence score

`calculate_confidence_score` (lines 232–264) is applied to each head's softmax
vector separately:

```
margin  = p(1) - p(2)                       # top minus runner-up
H       = -Σ p·ln(p + 1e-10)                # entropy over ALL outputs
H_norm  = 1 - H / (-ln(1/7))                # -ln(1/7) = ln 7 = 1.9459
ratio   = p(1) / mean(p(2..n))

score   = 0.4·margin + 0.3·H_norm + 0.3·min(ratio/10, 1)
```

A cell is written to the **confident** workbook only when *both* heads score at
or above the threshold (line 318); otherwise the whole row goes to the
**uncertain** workbook. There is no per-head split of the output.

Things to know before quoting a score:

- **It is not a probability.** It is an ad-hoc weighted blend on roughly the
  interval 0–1, and 0.6 is a point on that scale, not 60 % confidence.
- **The entropy term is normalised against 7 classes while the vector has 8**
  (`num_classes = 8` includes the unused index 0). When the distribution is
  near-uniform over eight outputs, `H_norm` goes slightly negative — about
  −0.07, so ≈ −0.02 on the score. Harmless in practice, but it means the score
  can dip just below zero.
- **The ratio term saturates early.** With 8 outputs, `mean(p(2..n)) =
  (1−p(1))/7`, so `ratio ≥ 10` — the cap — as soon as the top probability
  passes ≈ 0.59. Above that, only the margin and entropy terms still move.

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

Written into the **sample folder itself** (line 274, or `base_folder2` on line
276 if the sample is not under `base_folder`), only when `out_pred=True`:

```
<sample>/predict_class_confident_<threshold>.xlsx
<sample>/predict_class_uncertain_<threshold>.xlsx
```

With the shipped threshold that is `predict_class_confident_0.6.xlsx`. Change
the threshold and you get a *new pair of files*, not an overwritten one — and
`Visualize_heatmap.py` will then not find them. (The console messages on lines
328 and 334 print the names without the suffix; the files on disk have it.)

| Column | Contents |
|---|---|
| `Directory` | the sample folder |
| `Cell_Label` | the `<label>` in `cell<label>_5D.tif` |
| `Predicted_Nu_Class` | class **name** (`'N10'`…), or `'Unknown'` for index 0 |
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

`Cell_Label` is taken from `test_df.iloc[i]` where `i` is the loader's counter
(line 308) — see `troubleshooting.md` for when that stops lining up with the
image that was actually classified.

The repository README quotes "~1 second" per cell for inference and "typically
3 minutes" including preprocessing. Those are the README's numbers, not the
code's; nothing in these four scripts times itself.

---

## `Visualize_heatmap.py`

| Line | Name | Shipped |
|---|---|---|
| 171 | `base_folder` | `G:\BC-FLIM-S\WBY\Hek293T-BJMU-Dual` |
| 172 | `base_folder2` | `E:\BC-FLIM\Hek293T-BJMU-Dual` — a **different** second root from `Test_LUMINA.py`'s `I:\…` |
| 175–211 | `directories` | 1 active, ~30 commented out |
| 214–215 | `nu_FPs`, `mito_FPs` | `['N10','N13','N4','N14','N16','N8','N1']`, `['M10',…,'M1']` — the row and column order of the figure |
| 12 | `process_directory(..., confidence=0.6)` | the threshold used to build the file names; `main()` never passes it, so **0.6 is effectively hardcoded** |

**What the figure shows.** Rows are the predicted *nuclear* class, columns the
predicted *mitochondrial* class, and each entry counts confident cells with
that pair. Both axes are predictions — **there is no ground truth in this
figure.** It is a co-occurrence table of the dual-anchor calls, not a confusion
matrix. Do not describe it as one, and do not read the diagonal as "correct".

The diagonal is drawn on a separate grey colormap (`np.fill_diagonal` +
`mask`/`~mask`, lines 77–92); with the two class lists in matching order it
picks out `N10-M10`, `N13-M13`, … — the cells carrying the *same* fluorophore
on both anchors. Whether those are expected or are a sign of bleed-through
between the anchors is a question about the dish, not about the plot.

**Detection rate** = confident / (confident + uncertain), printed per directory
(line 51) and pooled across all listed directories (line 233), and written into
the figure title and the window title. It measures how many cells cleared the
confidence threshold, and says nothing about whether the calls are right.

**Two counts that can disagree.** `create_heatmap` prints `Total confident
cells plotted: len(data)` (line 69) but only increments the matrix for pairs
whose *names* are in `nu_FPs` and `mito_FPs` (line 66). Cells predicted
`'Unknown'`, or predicted into a class you removed from those lists, are
counted in the printed total and missing from the matrix. If the entries do not
sum to the printed number, that is why.

**Outputs.** `heatmap_nu_mito.pdf` — a *relative* path (line 106), so it lands
in whatever directory you launched Python from, and is overwritten on every
run. Then a Tk window opens (`root.mainloop()`, line 238) and blocks; clicking
a matrix entry prints that combination's per-directory breakdown to the
console. It is a GUI script — do not run it on a headless machine, and do not
run it when you only need the PDF.
