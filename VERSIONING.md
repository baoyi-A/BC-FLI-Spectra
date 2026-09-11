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
| `Napari_plugin/pyproject.toml` | `version = "1.1.3"` | pip, `flim_s_gen.__version__`, the `_meta` sheets |
| git tag | `v1.1.3` on the released commit | `pip install "git+https://github.com/baoyi-A/BC-FLI-Spectra.git@v1.1.3#subdirectory=Napari_plugin"`, rollback |
| `Napari_plugin/README.md` → Changelog | `**1.1.3 — 2026-09-11**` heading | people |

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

## What the output files record

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

The class-distribution `.npz` carries `plugin_version` and `written_at` as
arrays. Fine-tuned models carry `plugin_version` in `config.json`. Nothing
else the plugin writes is stamped: the per-FOV masks (`*_seg_n.npy`, the
`-cls.tif` class maps, tracking masks) and the rendered images identify
themselves only through the workbook of the run that made them.

A stamped file can therefore be matched to a release and to the run that
produced it, and a MAJOR version can tell an old file from a new one and
convert it instead of misreading it. Files written before 1.1.0 have no
`_meta` sheet; `signal_analysis.xlsx` is stamped from 1.1.1 and the `.npz`
from 1.1.2.

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
   environments from the machine the release was verified on). It refuses
   to write when it cannot tell which of two installed copies of a package
   is the one that imports, when a key package's imported version differs
   from the record, or when the recorded versions violate the dependency
   ranges in `pyproject.toml` — a record that `pip install -e` would then
   undo is not a record.
4. Commit the release: `git commit -am "Release X.Y.Z"` (pyproject, the
   changelog, `envs/`).
5. `python Napari_plugin/scripts/release.py check` — must pass. It refuses
   while `git status` lists anything, untracked files included (a new
   `envs/` record is untracked until added; a stray crash dump must be
   deleted or ignored).
6. `python Napari_plugin/scripts/release.py tag` — creates the annotated
   tag; then `git push origin main --tags`.
7. Create a GitHub Release for the tag (`gh release create vX.Y.Z`); the
   Zenodo integration that archived v1.0.1 archives each GitHub Release and
   mints its version DOI. A tag alone is not archived. Check the record
   afterwards, and update the version DOI in the README badge if it is cited.
8. Deploy where the plugin is used (below).

## Installing a particular version

Any release from 1.1.3 on, at any later date:

```bash
git clone --branch v1.1.3 https://github.com/baoyi-A/BC-FLI-Spectra.git
cd BC-FLI-Spectra/Napari_plugin
conda create -n bc-flim-1.1.3 --file envs/conda-napari-win64.txt   # the conda layer
conda activate bc-flim-1.1.3
pip install --no-deps -r envs/lock-napari-win64.txt                # the pip layer
pip install --no-deps -e .                                         # the plugin itself
```

The two files are two layers of one environment, not alternatives:
`conda list --explicit` records only what conda installed, and the pip
record is the set of copies the interpreter actually imports on top of it
(with the PyTorch index and the git commits of packages installed from
repositories named in the file). Apply the conda file first, then the pip
file. `--no-deps` on both pip steps is deliberate and is the form the record
was validated with: the record is complete, and letting pip re-resolve
dependencies would reject it for the internal inconsistencies any
long-lived environment accumulates (a package whose declared range excludes
the version that was in fact installed beside it). The same pair of files
exists for the `cellpose2` and `cellpose4` environments.

Environments are named by version and never upgraded in place. Two versions
that must coexist — the situation a MAJOR bump creates — live in two
environments, exactly as `cellpose2` and `cellpose4` do today.

`v1.0.0` and `v1.0.1` predate all of this: no environment records, and
their `pyproject.toml` is rejected by current setuptools
(`project.license must be string`), so they cannot be pip-installed at all.
They can be checked out and read; the recipe above starts at 1.1.3 (the
1.1.0–1.1.2 records exist, but 1.1.0's were not installable and 1.1.1's and
1.1.2's named some copies that do not load — see the changelog).

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
away. Rolling back to `v1.0.1` is a bare checkout: skip the `pip install`
step (that tree does not install), and know that `__version__` will keep
reporting whatever was installed last, its outputs carry no `_meta` sheet,
and a stale `src/flim_s_gen/_version.py` from an old install, if present,
makes `__version__` report a 2025 dev build until it is deleted.

## What is not promised

Files written by a newer MINOR version load in an older one, but the older
one cannot know about parameters it does not have; use the `_meta` sheet to
see what was actually run. Across a MAJOR boundary, converters are provided
where feasible and the Changelog says which files need them.

## History of the numbering

`v1.0.0` and `v1.0.1` (2026-09-01) point at the same commit and were cut while
`pyproject.toml` still said `0.1.0`. `1.1.0` (2026-09-11) is the first release
where the three places agree. `1.1.1`, `1.1.2` and `1.1.3` (same day) are
the rounds it took to make the environment records true and installable —
the first records were not installable, the next two each still named some
copies that do not load — and to finish the `_meta` stamping. Results in the manuscript
were produced with the classification pipeline as of `1.1.x` (whitening by
within-cluster spread, per-cluster isolation-forest rejection); `1.0.1`
predates whitening and should not be cited for them.
