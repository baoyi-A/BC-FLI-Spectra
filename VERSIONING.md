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
| `Napari_plugin/pyproject.toml` | `version = "1.1.1"` | pip, `flim_s_gen.__version__`, every output file |
| git tag | `v1.1.1` on the released commit | `pip install git+...@v1.1.1`, rollback |
| `Napari_plugin/README.md` → Changelog | `**1.1.1 — 2026-09-11**` heading | people |

Zenodo carries the same number as its record version, with one DOI per
release and one concept DOI that always resolves to the latest.

The installed package is the only source of `__version__`; there is no
generated `_version.py`. The metadata is written at install time, so after
any change to `pyproject.toml`, and after any `git pull` or checkout that
changes it, run `pip install -e Napari_plugin` again — otherwise
`__version__` and every `_meta` stamp keep reporting the previous number.
`scripts/release.py check` compares the installed number with `pyproject.toml`
and refuses to tag while they differ.

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

The workbooks the plugin writes — `FLIM-S.xlsx`, `clustered.xlsx` (both
the Seeded K-Means and the Harmony path), the seeds file, `Bs2Code.xlsx` and
`signal_analysis.xlsx` — carry a `_meta` sheet after the data sheets: the
`plugin_version` that wrote the file, when, and what the run did. For
`clustered.xlsx` that record is per localisation and says what actually
happened, not what was ticked: the method and K, the weights, whether
whitening was applied or skipped by its guard, the contamination that was
used, and which seed file (if any) was loaded. Readers that call
`pd.read_excel(path)` see the first sheet only, as before. Fine-tuned models
carry `plugin_version` in their `config.json`.

`FLIM-S.xlsx` is merged across runs (a run replaces its own FOV's rows and
keeps the others), so its `_meta` sheet describes the most recent write and
carries the previous write's rows under `previous.*`; rows from older runs
than that are described only by the data they contain.

A file can therefore be matched to a release and to the run that produced
it, and a MAJOR version can tell an old file from a new one and convert it
instead of misreading it. Files written before 1.1.0 have no `_meta` sheet.

## Cutting a release

1. Land everything on `main`. Working tree clean.
2. Set `version` in `Napari_plugin/pyproject.toml`, then
   `pip install -e Napari_plugin` so the installed metadata matches; add the
   Changelog entry under `## 📜 Changelog` in `Napari_plugin/README.md` with
   the same number and the date.
3. Refresh the environment records:
   `python Napari_plugin/scripts/release.py locks`
   (writes `Napari_plugin/envs/lock-<env>-win64.txt` and
   `conda-<env>-win64.txt` for the napari, cellpose2 and cellpose4
   environments from the machine the release was verified on; it refuses
   to write a record that disagrees with what the interpreter imports).
4. Commit the release: `git commit -am "Release X.Y.Z"` (pyproject, the
   changelog, `envs/`).
5. `python Napari_plugin/scripts/release.py check` — must pass.
6. `python Napari_plugin/scripts/release.py tag` — creates the annotated
   tag; then `git push origin main --tags`.
7. On Zenodo, add a new version to the existing record with the same
   number, and update the version DOI in the README badge if it is cited.
8. Deploy where the plugin is used (below).

## Installing a particular version

Any release from 1.1.1 on, at any later date:

```bash
git clone --branch v1.1.1 https://github.com/baoyi-A/BC-FLI-Spectra.git
cd BC-FLI-Spectra/Napari_plugin
conda create -n bc-flim-1.1.1 --file envs/conda-napari-win64.txt   # the conda layer
conda activate bc-flim-1.1.1
pip install -r envs/lock-napari-win64.txt                          # the pip layer
pip install -e .
```

The two files are two layers of one environment, not alternatives:
`conda list --explicit` records only what conda installed, and the pip
record is what the interpreter actually imports on top of it (with the
PyTorch index and the git commits of packages installed from repositories
named in the file). Apply the conda file first, then the pip file. The
same pair exists for the `cellpose2` and `cellpose4` environments.

Environments are named by version and never upgraded in place. Two versions
that must coexist — the situation a MAJOR bump creates — live in two
environments, exactly as `cellpose2` and `cellpose4` do today.

`v1.0.0` and `v1.0.1` predate all of this: no environment records, and the
package installs as `0.1.0`. They can be checked out and read, but the
recipe above starts at 1.1.1.

## Rolling back a deployment

A deployment is a git checkout with an editable install, never a copied
folder. Updating and rolling back are the same operation with a different
tag:

```bash
cd <deployment>/BC-FLI-Spectra
git fetch --tags
git checkout v1.1.1                # or any other tag
pip install -e Napari_plugin       # refresh the version metadata, always
```

then restart napari. Nothing is deleted; every other tag is one checkout
away. Rolling back to `v1.0.1` works as a checkout, but that version reports
itself as `0.1.0` and its outputs carry no `_meta` sheet; if a stale
`src/flim_s_gen/_version.py` is present from an old install, delete it, or
`__version__` will report a 2025 dev build.

## What is not promised

Files written by a newer MINOR version load in an older one, but the older
one cannot know about parameters it does not have; use the `_meta` sheet to
see what was actually run. Across a MAJOR boundary, converters are provided
where feasible and the Changelog says which files need them.

## History of the numbering

`v1.0.0` and `v1.0.1` (2026-09-01) point at the same commit and were cut while
`pyproject.toml` still said `0.1.0`. `1.1.0` (2026-09-11) is the first release
where the three places agree; `1.1.1` (same day) corrects its environment
records, which as first published were not installable, and completes the
`_meta` stamping. Results in the manuscript were produced with the
classification pipeline as of `1.1.x` (whitening by within-cluster spread,
per-cluster isolation-forest rejection); `1.0.1` predates whitening and
should not be cited for them.
