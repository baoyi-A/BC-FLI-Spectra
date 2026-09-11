"""Release helper: keep pyproject, the git tag and the changelog on one number.

    python scripts/release.py check   # report every mismatch; exit 1 if any
    python scripts/release.py locks   # refresh envs/lock-*.txt and envs/conda-*.txt
    python scripts/release.py tag     # run check, then create the annotated tag

Run from the repository root or from Napari_plugin/. Prints ASCII only, so it
survives a non-UTF-8 Windows console. Never pushes; the push is yours.
See VERSIONING.md.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.path.dirname(HERE)
ROOT = os.path.dirname(PLUGIN)
PYPROJECT = os.path.join(PLUGIN, 'pyproject.toml')
README = os.path.join(PLUGIN, 'README.md')
ENVS = os.path.join(PLUGIN, 'envs')

# environment name -> python.exe. Edit for the machine releases are verified on.
ENV_PYTHON = {
    'napari': r'D:\Softwares\Anaconda\Anaconda3\envs\BC-FLIM\python.exe',
    'cellpose2': r'D:\Softwares\Anaconda\Anaconda3\envs\cellpose\python.exe',
    'cellpose4': r'D:\Softwares\Anaconda\Anaconda3\envs\cellpose4\python.exe',
}
CONDA = r'D:\Softwares\Anaconda\Anaconda3\Scripts\conda.exe'
# Modules whose imported version must agree with the record, per environment.
KEY_IMPORTS = {
    'napari': ['napari', 'numpy', 'pandas', 'sklearn', 'torch', 'scipy', 'skimage'],
    'cellpose2': ['cellpose', 'numpy', 'torch'],
    'cellpose4': ['cellpose', 'numpy', 'torch'],
}
DIST_OF_MODULE = {'sklearn': 'scikit-learn', 'skimage': 'scikit-image'}
# distribution name -> conda package name, where they differ
CONDA_NAME = {'torch': 'pytorch', 'msgpack': 'msgpack-python', 'pyqt5': 'pyqt', 'pyyaml': 'pyyaml'}
OWN = ('bc-flim-spectra', 'bc-flim-s', 'napari-cutie', 'napari-mito-flim')
PYTORCH_INDEX = 'https://download.pytorch.org/whl/{cuda}'
# The documented install command. `_validate_pins` runs exactly this form, so
# what is validated is what the reader types. --no-deps because the record is
# complete: letting pip re-resolve dependencies would reject a real machine's
# environment for the inconsistencies every long-lived environment carries.
PIP_FORM = 'pip install --no-deps -r envs/lock-{env}-win64.txt'


def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True,
                          text=True, check=False).stdout.strip()


def read_version():
    with open(PYPROJECT, encoding='utf-8') as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    return m.group(1) if m else None


def read_dependencies():
    """The `dependencies = [...]` list of pyproject.toml, comments stripped."""
    with open(PYPROJECT, encoding='utf-8') as f:
        text = f.read()
    m = re.search(r'^dependencies\s*=\s*\[(.*?)^\]', text, re.S | re.M)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        line = line.split('#', 1)[0].strip().strip(',').strip()
        if line.startswith(('"', "'")):
            out.append(line.strip('"\''))
    return out


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('%s failed (%d): %s' % (' '.join(cmd[:3]), r.returncode,
                                                  (r.stderr or r.stdout).strip()[:300]))
    return r.stdout


def installed_version(py):
    """Version the plugin reports inside an environment, or None."""
    r = subprocess.run([py, '-c', 'import importlib.metadata as m; print(m.version("bc-flim-spectra"))'],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


# ---------------------------------------------------------------- check

def problems(version):
    out = []
    if not version or not re.fullmatch(r'\d+\.\d+\.\d+', version):
        out.append('pyproject.toml: version "%s" is not MAJOR.MINOR.PATCH' % version)
        return out
    tag = 'v' + version
    with open(README, encoding='utf-8') as f:
        readme = f.read()
    if not re.search(r'^\*\*%s\b' % re.escape(version), readme, re.M):
        out.append('README.md changelog: no heading starting with "**%s"' % version)
    if git('status', '--porcelain'):
        out.append('working tree is not clean: `git status` must print nothing '
                   '(commit, delete or .gitignore each listed file; untracked files count)')
    head = git('rev-parse', 'HEAD')
    others = [t for t in git('tag', '--points-at', 'HEAD').split() if t != tag]
    if others:
        out.append('HEAD already carries tag(s) %s; one commit, one tag' % ' '.join(others))
    if tag in git('tag', '-l').split():
        at = git('rev-list', '-n', '1', tag)
        if at != head:
            out.append('tag %s exists on %s, HEAD is %s' % (tag, at[:8], head[:8]))
    inst = installed_version(ENV_PYTHON['napari'])
    if inst is None:
        out.append('plugin is not installed in the napari environment (pip install -e Napari_plugin)')
    elif inst != version:
        out.append('installed plugin reports %s, pyproject says %s (pip install -e Napari_plugin)'
                   % (inst, version))
    for env in ENV_PYTHON:
        for kind in ('lock', 'conda'):
            p = os.path.join(ENVS, '%s-%s-win64.txt' % (kind, env))
            rel = os.path.relpath(p, ROOT)
            if not os.path.isfile(p):
                out.append('missing %s (run: release.py locks)' % rel)
                continue
            with open(p, encoding='utf-8') as f:
                lines = f.read().splitlines()
            if ('bc-flim-spectra %s ' % version) not in (lines[0] if lines else ''):
                out.append('%s is not from version %s (run: release.py locks)' % (rel, version))
            body = [l for l in lines if l.strip() and not l.startswith('#') and not l.startswith('--')]
            if len(body) < 5:
                out.append('%s has no package lines (run: release.py locks)' % rel)
    return out


def cmd_check():
    v = read_version()
    print('pyproject version : %s' % v)
    print('installed (napari): %s' % installed_version(ENV_PYTHON['napari']))
    print('HEAD              : %s' % git('rev-parse', '--short', 'HEAD'))
    print('tags              : %s' % (git('tag', '-l') or '(none)').replace('\n', ' '))
    bad = problems(v)
    for b in bad:
        print('  PROBLEM  ' + b)
    print('OK: ready to tag v%s' % v if not bad else '%d problem(s)' % len(bad))
    return 0 if not bad else 1


# ---------------------------------------------------------------- locks

# Runs inside the target interpreter. When one distribution name has several
# dist-infos on the machine, the copy that actually imports is the one whose
# RECORD describes the files on disk: the hashes are compared for every file
# the candidates disagree on. Only when the package files are identical does
# the module's __version__ decide (it may itself be importlib.metadata's
# answer, which is directory order and proves nothing), then the location.
INSPECT_CODE = r'''
import importlib, importlib.metadata as m, json, os, platform
from collections import defaultdict
own = set(%r); mods = %r; conda_ver = %r; conda_name = %r
groups = defaultdict(list)
for d in m.distributions():
    name = (d.metadata['Name'] or '').strip(); key = name.lower().replace('_', '-')
    if name and key not in own:
        groups[key].append(d)
def norm(p):
    return os.path.normcase(os.path.abspath(str(p))) if p else None
def top_modules(d):
    try:
        t = d.read_text('top_level.txt')
        if t and t.split():
            return [x for x in t.split() if x and not x.startswith('_')] or t.split()
    except Exception:
        pass
    return [(d.metadata['Name'] or '').replace('-', '_')]
def import_info(modname):
    try:
        mod = importlib.import_module(modname)
        return norm(getattr(mod, '__file__', None)), getattr(mod, '__version__', None)
    except Exception:
        return None, None
def file_hash_b64(path):
    import hashlib, base64
    try:
        h = hashlib.sha256(open(path, 'rb').read()).digest()
        return base64.urlsafe_b64encode(h).rstrip(b'=').decode()
    except Exception:
        return None
def record_hash(d, path):
    """RECORD's sha256 (urlsafe b64, unpadded) for the on-disk file, or None."""
    try:
        for f in (d.files or []):
            if norm(d.locate_file(f)) == path and f.hash is not None and f.hash.mode == 'sha256':
                return f.hash.value
    except Exception:
        pass
    return None
chosen, ambiguous, dup_note = {}, [], []
for key, ds in groups.items():
    if len(ds) == 1:
        chosen[key] = ds[0]; continue
    if len(set(d.version for d in ds)) == 1:
        # same version twice (conda and pip both installed it): identical code,
        # and the conda layer already records it
        conda_ones = [d for d in ds if (d.read_text('INSTALLER') or '').strip().lower() == 'conda']
        chosen[key] = (conda_ones or ds)[0]
        dup_note.append('%%s: %%s installed twice (same version)' %% (key, ds[0].version)); continue
    pick = None; how = ''
    # 0. the RECORD that describes the files on disk: hash every file the
    #    candidates disagree on (at most 400) and count agreements
    recs = []
    for d in ds:
        rec = {}
        try:
            for f in (d.files or []):
                rel = str(f).replace(os.sep, '/')
                if (f.hash is None or f.hash.mode != 'sha256' or rel.endswith('.pyc')
                        or '.dist-info/' in rel or '.egg-info/' in rel):
                    continue
                rec[rel] = f.hash.value
        except Exception:
            rec = None
        recs.append(rec)
    if all(r is not None for r in recs):
        every = set().union(*[set(r) for r in recs])
        paths = sorted(q for q in every if len(set(r.get(q) for r in recs)) > 1)[:400]
        if paths:
            score = [0] * len(ds)
            for q in paths:
                on_disk = None
                for d, r in zip(ds, recs):
                    if q in r:
                        on_disk = file_hash_b64(str(d.locate_file(q))); break
                for i, r in enumerate(recs):
                    if (r.get(q) == on_disk) if q in r else (on_disk is None):
                        score[i] += 1
            best = max(score)
            if best > 0 and score.count(best) == 1:
                pick = ds[score.index(best)]
                how = 'RECORD, %%d of %%d differing files on disk' %% (best, len(paths))
    for modname in ([] if pick is not None else top_modules(ds[0])):
        f, v = import_info(modname)
        if f is None:
            continue
        # 1. the version the module reports about itself
        if v is not None:
            same = [d for d in ds if d.version == str(v)]
            if len(same) == 1:
                pick, how = same[0], '__version__ (package files identical)'; break
        # 2. the hash RECORD holds for the file that was imported
        hv = file_hash_b64(f)
        if hv:
            same = [d for d in ds if record_hash(d, f) == hv]
            if len(same) == 1:
                pick, how = same[0], 'RECORD hash'; break
        # 3. dist-infos in different directories: the one beside the module
        here = os.path.dirname(f)
        same = [d for d in ds if norm(d.locate_file('')).rstrip(os.sep) == here.rstrip(os.sep)
                or here.startswith(norm(d.locate_file('')).rstrip(os.sep) + os.sep)]
        if len(same) == 1:
            pick, how = same[0], 'location'; break
    if pick is None:
        ambiguous.append('%%s: copies %%s, cannot tell which imports' %% (key, ', '.join(sorted(set(d.version for d in ds))))); continue
    chosen[key] = pick
    dup_note.append('%%s: copies %%s; %%s imports (by %%s)' %% (key, ', '.join(sorted(set(d.version for d in ds))), pick.version, how))
pip_pins, conda_layer, dropped = [], [], []
for key, d in chosen.items():
    name = (d.metadata['Name'] or '').strip()
    installer = ''
    try:
        installer = (d.read_text('INSTALLER') or '').strip().lower()
    except Exception:
        pass
    cname = conda_name.get(key, key)
    in_conda = conda_ver.get(cname) == d.version or conda_ver.get(key) == d.version
    if installer == 'conda' or (installer != 'pip' and in_conda):
        conda_layer.append(key); continue
    du = None
    try:
        raw = d.read_text('direct_url.json'); du = json.loads(raw) if raw else None
    except Exception:
        du = None
    url = str((du or {}).get('url', ''))
    if du and du.get('dir_info', {}).get('editable'):
        dropped.append('-e %%s  (%%s %%s: editable checkout)' %% (url, name, d.version)); continue
    if du and du.get('vcs_info'):
        vi = du['vcs_info']
        pip_pins.append('%%s @ %%s+%%s@%%s' %% (name, vi.get('vcs', 'git'), url, vi.get('commit_id', ''))); continue
    if du and url.startswith('file:'):
        if in_conda:
            conda_layer.append(key); continue
        dropped.append('%%s==%%s  (installed from %%s)' %% (name, d.version, url)); continue
    pip_pins.append('%%s==%%s' %% (name, d.version))
imported = {}
for n in mods:
    f, v = import_info(n)
    if f is None:
        imported[n] = 'ERR import failed'; continue
    if v is None:
        hv = file_hash_b64(f)
        v = next((d.version for ds in groups.values() for d in ds if hv and record_hash(d, f) == hv), None)
    imported[n] = str(v) if v is not None else 'ERR no version'
print(json.dumps(dict(pins=sorted(pip_pins, key=str.lower), conda=sorted(conda_layer), dropped=dropped,
                      ambiguous=ambiguous, duplicates=dup_note, imported=imported,
                      python=platform.python_version())))
'''


def _conda_versions(spec_text):
    """{name: version} from a `conda list --explicit` spec."""
    out = {}
    for line in spec_text.splitlines():
        m = re.search(r'/([^/]+)-([^-/]+)-[^-/]+\.(?:conda|tar\.bz2)$', line.strip())
        if m:
            # the same normalisation distribution names get (mkl_fft -> mkl-fft)
            out[re.sub(r'[-_.]+', '-', m.group(1)).lower()] = m.group(2)
    return out


def _inspect(py, env, conda_ver):
    code = INSPECT_CODE % ([o.lower() for o in OWN], KEY_IMPORTS.get(env, []), conda_ver, CONDA_NAME)
    return json.loads(_run([py, '-c', code]))


def _cuda_tag(pins):
    """The CUDA tag of any +cuNNN pin (torch, torchvision, torchaudio, ...)."""
    for l in pins:
        m = re.match(r'[A-Za-z0-9_.\-]+==\d[^+]*\+(cu\d+)', l)
        if m:
            return m.group(1)
    return None


def _validate_pins(py, pins, cuda):
    """Ask pip to resolve the record the way it will be installed (--no-deps),
    without installing. Returns (pins that resolve, [lines that do not])."""
    import tempfile
    unresolvable, pins = [], list(pins)
    for _ in range(12):
        fd, req = tempfile.mkstemp(suffix='.txt'); os.close(fd)
        with open(req, 'w', encoding='utf-8') as f:
            if cuda:
                f.write('--extra-index-url %s\n' % PYTORCH_INDEX.format(cuda=cuda))
            f.write('\n'.join(pins) + '\n')
        r = subprocess.run([py, '-m', 'pip', 'install', '--dry-run', '--no-deps', '--ignore-installed',
                            '-q', '-r', req], capture_output=True, text=True)
        os.remove(req)
        if r.returncode == 0:
            return pins, unresolvable
        m = re.search(r'No matching distribution found for ([^\s]+)', r.stderr + r.stdout)
        if not m:
            raise RuntimeError('pip dry-run failed: %s' % (r.stderr or r.stdout).strip()[-400:])
        bad = m.group(1).split('==')[0].split('@')[0].strip().lower().replace('_', '-')
        hit = [l for l in pins if l.split('==')[0].split('@')[0].strip().lower().replace('_', '-') == bad]
        if not hit:
            raise RuntimeError('pip rejected %s but it is not in the record' % bad)
        unresolvable.extend(hit)
        pins = [l for l in pins if l not in hit]
    raise RuntimeError('too many unresolvable pins; giving up')


def _pyproject_violations(recorded):
    """pyproject dependencies not satisfied by the recorded versions."""
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version
    except ImportError:
        return ['packaging is not importable here; cannot check pyproject specifiers']
    out = []
    for spec in read_dependencies():
        try:
            req = Requirement(spec)
        except Exception:
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        name = req.name.lower().replace('_', '-')
        ver = recorded.get(name)
        if ver is None:
            out.append('%s: not in either layer' % spec)
        elif not req.specifier.contains(Version(ver.split('+')[0]), prereleases=True):
            out.append('%s: recorded %s' % (spec, ver))
    return out


def cmd_locks():
    import datetime
    import platform
    v = read_version()
    os.makedirs(ENVS, exist_ok=True)
    today = datetime.date.today().isoformat()
    rc = 0
    for env, py in ENV_PYTHON.items():
        if not os.path.isfile(py):
            print('  skip %s: no %s' % (env, py)); rc = 1; continue
        env_name = os.path.basename(os.path.dirname(py))
        spec = _run([CONDA, 'list', '-n', env_name, '--explicit'])
        conda_ver = _conda_versions(spec)
        info = _inspect(py, env, conda_ver)
        fatal = list(info['ambiguous'])
        pins = list(info['pins'])
        pinned = {l.split('==')[0].lower().replace('_', '-'): l.split('==')[1] for l in pins if '==' in l}
        recorded = dict(conda_ver)
        for k, vv in pinned.items():
            recorded[k] = vv
        for mod, ver in info['imported'].items():
            d = DIST_OF_MODULE.get(mod, mod).lower()
            rec = recorded.get(d) or recorded.get(CONDA_NAME.get(d, d))
            if ver.startswith('ERR'):
                fatal.append('%s: %s' % (mod, ver))
            elif rec is None:
                fatal.append('%s: imports %s but appears in neither layer' % (mod, ver))
            elif rec.split('+')[0] != ver.split('+')[0]:
                fatal.append('%s: record says %s, import gives %s' % (mod, rec, ver))
        if env == 'napari':
            viol = _pyproject_violations(recorded)
            if viol:
                fatal.append('the recorded environment violates pyproject.toml: ' + '; '.join(viol)
                             + '  (fix pyproject or the environment; a record that pip install -e would undo is not a record)')
        if fatal:
            print('  %s: not writing:' % env)
            for m_ in fatal:
                print('      ' + m_)
            rc = 1
            continue
        pins = sorted(pins, key=str.lower)
        cuda = _cuda_tag(pins)
        print('  %-10s asking pip to resolve %d pins (takes a few minutes)...' % (env, len(pins)))
        pins, unresolvable = _validate_pins(py, pins, cuda)
        stuck = [u for u in unresolvable if re.search(r'\+cu\d+', u)]
        if stuck:
            # a CUDA build that imports here and cannot be fetched is a hole in
            # the record, not a package the plugin does not need
            print('  %s: not writing: CUDA-tagged pins do not resolve from the index: %s'
                  % (env, ', '.join(stuck)))
            rc = 1
            continue
        for u in unresolvable:
            info['dropped'].append('%s  (not resolvable from the configured indexes)' % u)
        lock = os.path.join(ENVS, 'lock-%s-win64.txt' % env)
        with open(lock, 'w', encoding='utf-8', newline='\n') as f:
            f.write("# bc-flim-spectra %s -- pip layer of the '%s' environment (%s), frozen %s on %s, Python %s.\n"
                    % (v, env, env_name, today, platform.platform(), info['python']))
            f.write('# The copies the interpreter actually imports, on top of the conda layer in\n'
                    '# envs/conda-%s-win64.txt. Apply the conda file first, then exactly:\n'
                    '#   %s\n'
                    '# (--no-deps: the record is complete; this is the form it was validated with.)\n'
                    % (env, PIP_FORM.format(env=env)))
            if cuda:
                f.write('# torch and friends carry a +%s tag and come from the PyTorch index below, not PyPI.\n'
                        '--extra-index-url %s\n' % (cuda, PYTORCH_INDEX.format(cuda=cuda)))
            f.write('# Lines with " @ git+" are packages installed from a repository at a fixed commit.\n')
            if info['duplicates']:
                f.write('# Names with more than one copy on the freezing machine; the record is the copy\n'
                        '# whose files the interpreter imports:\n')
                for n_ in info['duplicates']:
                    f.write('#   %s\n' % n_)
            if info['dropped']:
                f.write('# Present on the freezing machine but not recorded (the plugin does not need them):\n')
                for d_ in info['dropped']:
                    f.write('#   %s\n' % d_)
            f.write('\n'.join(pins) + '\n')
        conda = os.path.join(ENVS, 'conda-%s-win64.txt' % env)
        with open(conda, 'w', encoding='utf-8', newline='\n') as f:
            f.write("# bc-flim-spectra %s -- conda layer of the '%s' environment (%s), %s.\n"
                    % (v, env, env_name, today))
            f.write('# conda create -n <name> --file <this file>   then   %s\n' % PIP_FORM.format(env=env))
            f.write(spec)
        print('  %-10s pip layer %3d pins (%s; %d duplicates resolved, %d not recorded), conda layer %3d packages'
              % (env, len(pins), ('index +' + cuda) if cuda else 'PyPI only',
                 len(info['duplicates']), len(info['dropped']), len(conda_ver)))
    return rc


# ---------------------------------------------------------------- tag

def cmd_tag():
    if cmd_check() != 0:
        return 1
    v = read_version()
    tag = 'v' + v
    if tag in git('tag', '-l').split():
        print('tag %s already on HEAD; nothing to do' % tag)
        return 0
    subprocess.run(['git', '-C', ROOT, 'tag', '-a', tag, '-m', 'bc-flim-spectra %s' % v], check=True)
    print('created %s' % tag)
    print('next:  git push origin main --tags')
    print('then:  add version %s on Zenodo (concept DOI 10.5281/zenodo.22228957)' % v)
    return 0


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    fn = {'check': cmd_check, 'locks': cmd_locks, 'tag': cmd_tag}.get(what)
    if fn is None:
        print(__doc__)
        sys.exit(2)
    sys.exit(fn())
