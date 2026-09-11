# Versioning

How releases of this repository are numbered, cut, archived, and rolled back.
The rules exist so that a result produced with one version can always be
matched to the code that produced it, and so that an old version can still be
installed after the ecosystem underneath it has moved on.

## One number, kept in three places

The version lives in `Napari_plugin/pyproject.toml` (`version = "X.Y.Z"`).
Two other places must agree with it, and `scripts/release.py check` refuses to
tag until they do:

| Place | Form | Who reads it |
|---|---|---|
| `Napari_plugin/pyproject.toml` | `version = "1.1.0"` | pip, `flim_s_gen.__version__`, every output file |
| git tag | `v1.1.0` on the released commit | `pip install git+...@v1.1.0`, rollback |
| `Napari_plugin/README.md` → Changelog | `**1.1.0 — 2026-09-11**` heading | people |

Zenodo carries the same number as its record version, with one DOI per
release and one concept DOI that always resolves to the latest.

The installed package is the only source of `__version__`; there is no
generated `_version.py`. After editing `pyproject.toml`, run
`pip install -e .` again so the metadata catches up.

## What a number means

`MAJOR.MINOR.PATCH`, in the usual sense, with these definitions for this
project:

- **MAJOR** changes when something already on disk stops being readable by the
  new code, or reads differently: the columns of `FLIM-S.xlsx`,
  `clustered.xlsx`, `Bs2Code.xlsx`, `signal_analysis.xlsx`; the seeds and
  distribution files; the per-cell crops LUMINA consumes; the checkpoint
  layout; the model `config.json`; the sample-folder layout the widgets
  expect. A MAJOR bump is also the answer when a base dependency breaks
  underneath us the way Cellpose did between 2 and 4 — two versions of this
  plugin may then be needed side by side, and they must not share a number.
- **MINOR** changes when the same inputs can produce different numbers:
  a new classification step (whitening was one), a changed default
  parameter, a changed default model, a different rejection rule. The old
  files still load; the results are not the same. The manuscript cites a
  MINOR version for this reason.
- **PATCH** changes fix behaviour without changing what a correct run
  produces: a crash, a hang, a dialog, a wrong error message, documentation.

When in doubt between two levels, take the higher one.

## What every output file records

Every workbook the plugin writes has a second sheet, `_meta`, after the data
sheet: the `plugin_version` that wrote it, when, and the settings of that run
(phasor window, thresholds, weights, whitening, contamination, seed file,
alignment threshold, and so on). Readers that call `pd.read_excel(path)`
see the data sheet only, as before. Fine-tuned models carry `plugin_version`
in their `config.json`.

So a file can always be matched to a release and to the parameters that
produced it, and a MAJOR version can tell an old file from a new one and
convert it instead of misreading it.

## Cutting a release

1. Land everything on `main`. Working tree clean.
2. Set `version` in `Napari_plugin/pyproject.toml`; add the Changelog entry
   under `## 📜 Changelog` in `Napari_plugin/README.md` with the same number
   and the date.
3. Refresh the environment records:
   `python scripts/release.py locks`
   (writes `Napari_plugin/envs/lock-<env>-win64.txt` and
   `conda-<env>-win64.txt` for the napari, cellpose2 and cellpose4
   environments from the machine the release was verified on).
4. `python scripts/release.py check` — must pass.
5. `python scripts/release.py tag` — creates the annotated tag; then
   `git push origin main --tags`.
6. On Zenodo, add a new version to the existing record with the same
   number, and update the version DOI in the README badge if it is cited.
7. Deploy where the plugin is used (below).

## Installing a particular version

Any released version, at any later date:

```bash
git clone --branch v1.1.0 https://github.com/baoyi-A/BC-FLI-Spectra.git
cd BC-FLI-Spectra/Napari_plugin
pip install -r envs/lock-napari-win64.txt      # the exact environment it was verified in
pip install -e .
```

Or, for the conda form of the same environment:

```bash
conda create -n bc-flim-1.1.0 --file envs/conda-napari-win64.txt
```

Environments are named by version and never upgraded in place. Two versions
that must coexist — the situation a MAJOR bump creates — live in two
environments, exactly as `cellpose2` and `cellpose4` do today.

## Rolling back a deployment

A deployment is a git checkout with an editable install, never a copied
folder, so a rollback is a checkout:

```bash
cd <deployment>/BC-FLI-Spectra
git fetch --tags
git checkout v1.0.1
pip install -e Napari_plugin       # refresh the version metadata
```

then restart napari. Nothing is deleted; the newer tag is one checkout away.

## What is not promised

Files written by a newer MINOR version load in an older one, but the older
one cannot know about parameters it does not have; use the `_meta` sheet to
see what was actually run. Across a MAJOR boundary, converters are provided
where feasible and the Changelog says which files need them.

## History of the numbering

`v1.0.0` and `v1.0.1` (2026-09-01) point at the same commit and were cut while
`pyproject.toml` still said `0.1.0`. `1.1.0` is the first release where the
three places agree. Results in the manuscript were produced with the
classification pipeline as of `1.1.0` (whitening by within-cluster spread,
per-cluster isolation-forest rejection); `1.0.1` predates whitening and
should not be cited for them.
