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
# Modules whose imported version must agree with the record, per environment,
# so a pin can never name a copy that is installed but never loads.
KEY_IMPORTS = {
    'napari': ['napari', 'numpy', 'pandas', 'sklearn', 'torch'],
    'cellpose2': ['cellpose', 'numpy', 'torch'],
    'cellpose4': ['cellpose', 'numpy', 'torch'],
}
DIST_OF_MODULE = {'sklearn': 'scikit-learn'}
OWN = ('bc-flim-spectra', 'bc-flim-s', 'napari-cutie', 'napari-mito-flim')
PYTORCH_INDEX = 'https://download.pytorch.org/whl/{cuda}'


def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True,
                          text=True, check=False).stdout.strip()


def read_version():
    with open(PYPROJECT, encoding='utf-8') as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    return m.group(1) if m else None


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
        out.append('working tree is not clean (commit the release first)')
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

# Runs inside the target interpreter. Walks importlib.metadata in sys.path
# order (first hit is the copy that imports), splits distributions by who
# installed them, and keeps the origin of anything installed from a repository.
INSPECT_CODE = r'''
import importlib, importlib.metadata as m, json, platform
own = set(%r); mods = %r
seen, pip_pins, conda_names, dropped = set(), [], [], []
for d in m.distributions():
    name = (d.metadata['Name'] or '').strip()
    key = name.lower().replace('_', '-')
    if not name or key in seen:
        continue
    seen.add(key)
    if key in own:
        continue
    installer = (d.read_text('INSTALLER') or '').strip().lower()
    if installer == 'conda':
        conda_names.append(key)             # recorded by the conda layer
        continue
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
        dropped.append('%%s==%%s  (installed from %%s: not on an index)' %% (name, d.version, url)); continue
    pip_pins.append('%%s==%%s' %% (name, d.version))
imported = {}
for n in mods:
    try:
        mod = importlib.import_module(n); imported[n] = str(getattr(mod, '__version__', '?'))
    except Exception as e:
        imported[n] = 'ERR ' + type(e).__name__
print(json.dumps(dict(pins=sorted(pip_pins, key=str.lower), conda=sorted(conda_names),
                      dropped=dropped, imported=imported, python=platform.python_version())))
'''


def _inspect(py, env):
    return json.loads(_run([py, '-c', INSPECT_CODE % ([o.lower() for o in OWN],
                                                       KEY_IMPORTS.get(env, []))]))


def _conda_versions(spec_text):
    """{name: version} from a `conda list --explicit` spec."""
    out = {}
    for line in spec_text.splitlines():
        m = re.search(r'/([^/]+)-([^-/]+)-[^-/]+\.(?:conda|tar\.bz2)$', line.strip())
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def _cuda_tag(pins):
    for l in pins:
        m = re.match(r'torch==\d[^+]*\+(cu\d+)', l)
        if m:
            return m.group(1)
    return None


def _validate_pins(py, pins, cuda):
    """Ask pip to resolve the record without installing; drop what no index has.

    Returns (pins that resolve, [lines that do not]). Iterates because pip stops
    at the first unresolvable requirement.
    """
    import tempfile
    unresolvable = []
    pins = list(pins)
    for _ in range(12):
        fd, req = tempfile.mkstemp(suffix='.txt'); os.close(fd)
        with open(req, 'w', encoding='utf-8') as f:
            if cuda:
                f.write('--extra-index-url %s' % PYTORCH_INDEX.format(cuda=cuda) + chr(10))
            f.write(chr(10).join(pins) + chr(10))
        r = subprocess.run([py, '-m', 'pip', 'install', '--dry-run', '--no-deps', '--ignore-installed',
                            '-q', '-r', req], capture_output=True, text=True)
        os.remove(req)
        if r.returncode == 0:
            return pins, unresolvable
        m = re.search(r'No matching distribution found for ([^\s]+)', r.stderr + r.stdout)
        if not m:
            raise RuntimeError('pip dry-run failed for another reason: %s' % (r.stderr or r.stdout).strip()[-400:])
        bad = m.group(1).split('==')[0].split('@')[0].strip().lower().replace('_', '-')
        hit = [l for l in pins if l.split('==')[0].split('@')[0].strip().lower().replace('_', '-') == bad]
        if not hit:
            raise RuntimeError('pip rejected %s but it is not in the record' % bad)
        unresolvable.extend(hit)
        pins = [l for l in pins if l not in hit]
    raise RuntimeError('too many unresolvable pins; giving up')


def cmd_locks():
    import datetime
    import platform
    v = read_version()
    os.makedirs(ENVS, exist_ok=True)
    today = datetime.date.today().isoformat()
    rc = 0
    for env, py in ENV_PYTHON.items():
        if not os.path.isfile(py):
            print('  skip %s: no %s' % (env, py))
            rc = 1
            continue
        info = _inspect(py, env)
        env_name = os.path.basename(os.path.dirname(py))
        spec = _run([CONDA, 'list', '-n', env_name, '--explicit'])
        conda_ver = _conda_versions(spec)
        pins = list(info['pins'])
        pinned = {l.split('==')[0].lower().replace('_', '-'): l.split('==')[1] for l in pins if '==' in l}
        # cross-check against what actually imports, over both layers
        notes, fatal = [], []
        for mod, ver in info['imported'].items():
            d = DIST_OF_MODULE.get(mod, mod).lower()
            if ver.startswith('ERR'):
                fatal.append('%s: %s' % (mod, ver)); continue
            rec = pinned.get(d) or conda_ver.get(d)
            layer = 'pip' if d in pinned else ('conda' if d in conda_ver else None)
            if rec is None:
                fatal.append('%s: imports %s but appears in neither layer' % (mod, ver)); continue
            if ver != '?' and rec.split('+')[0] != ver.split('+')[0]:
                if layer == 'pip':
                    pins = [l for l in pins if l.split('==')[0].lower().replace('_', '-') != d]
                    pins.append('%s==%s' % (d, ver)); pinned[d] = ver
                    notes.append('%s: metadata said %s, import gives %s; pinned %s' % (d, rec, ver, ver))
                else:
                    fatal.append('%s: conda layer says %s, import gives %s' % (mod, rec, ver))
        if fatal:
            print('  %s: cannot reconcile the record with the interpreter; not writing:' % env)
            for m_ in fatal:
                print('      ' + m_)
            rc = 1
            continue
        pins = sorted(pins, key=str.lower)
        cuda = _cuda_tag(pins)
        print('  %-10s asking pip to resolve %d pins (takes a few minutes)...' % (env, len(pins)))
        pins, unresolvable = _validate_pins(py, pins, cuda)
        for u in unresolvable:
            info['dropped'].append('%s  (no index has this version; installed from a local build)' % u)
        lock = os.path.join(ENVS, 'lock-%s-win64.txt' % env)
        with open(lock, 'w', encoding='utf-8', newline='\n') as f:
            f.write("# bc-flim-spectra %s -- pip layer of the '%s' environment (%s), frozen %s on %s, Python %s.\n"
                    % (v, env, env_name, today, platform.platform(), info['python']))
            f.write('# What the interpreter imports on top of the conda layer in envs/conda-%s-win64.txt.\n'
                    '# Apply the conda file first, then:   pip install -r envs/lock-%s-win64.txt\n'
                    % (env, env))
            if cuda:
                f.write('# torch and friends carry a +%s tag and come from the PyTorch index below, not PyPI.\n'
                        '--extra-index-url %s\n' % (cuda, PYTORCH_INDEX.format(cuda=cuda)))
            f.write('# Lines with " @ git+" are packages installed from a repository at a fixed commit.\n')
            if info['dropped']:
                f.write('# Present on the freezing machine but not recorded (editable checkouts and\n'
                        '# local-only packages; the plugin does not need them):\n')
                for d_ in info['dropped']:
                    f.write('#   %s\n' % d_)
            if notes:
                f.write('# Two copies were installed; the pin is the copy that imports:\n')
                for n_ in notes:
                    f.write('#   %s\n' % n_)
            f.write('\n'.join(pins) + '\n')
        conda = os.path.join(ENVS, 'conda-%s-win64.txt' % env)
        with open(conda, 'w', encoding='utf-8', newline='\n') as f:
            f.write("# bc-flim-spectra %s -- conda layer of the '%s' environment (%s), %s.\n"
                    % (v, env, env_name, today))
            f.write('# conda create -n <name> --file <this file>   then   pip install -r envs/lock-%s-win64.txt\n'
                    % env)
            f.write(spec)
        print('  %-10s pip layer %3d pins (%s; %d not recorded), conda layer %3d packages'
              % (env, len(pins), ('index +' + cuda) if cuda else 'PyPI only',
                 len(info['dropped']), len(conda_ver)))
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
