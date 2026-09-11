"""Release helper: keep pyproject, the git tag and the changelog on one number.

    python scripts/release.py check   # report every mismatch; exit 1 if any
    python scripts/release.py locks   # refresh envs/lock-*.txt and envs/conda-*.txt
    python scripts/release.py tag     # run check, then create the annotated tag

Run from the repository root or from Napari_plugin/. Prints ASCII only, so it
survives a non-UTF-8 Windows console. Never pushes; the push is yours.
See VERSIONING.md.
"""
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
# our own editable installs are not third-party pins
OWN = ('bc-flim-spectra', 'bc-flim-s', 'napari-cutie', 'napari-mito-flim')


def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True,
                          text=True, check=False).stdout.strip()


def read_version():
    with open(PYPROJECT, encoding='utf-8') as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    return m.group(1) if m else None


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
        out.append('working tree is not clean')
    tags = git('tag', '-l').split()
    if tag in tags:
        head = git('rev-parse', 'HEAD')
        at = git('rev-list', '-n', '1', tag)
        if at != head:
            out.append('tag %s exists on %s, HEAD is %s' % (tag, at[:8], head[:8]))
    for env in ENV_PYTHON:
        for kind in ('lock', 'conda'):
            p = os.path.join(ENVS, '%s-%s-win64.txt' % (kind, env))
            if not os.path.isfile(p):
                out.append('missing %s (run: release.py locks)' % os.path.relpath(p, ROOT))
            else:
                with open(p, encoding='utf-8') as f:
                    if ('bc-flim-spectra %s ' % version) not in f.readline():
                        out.append('%s is not from version %s (run: release.py locks)'
                                   % (os.path.relpath(p, ROOT), version))
    return out


def cmd_check():
    v = read_version()
    print('pyproject version : %s' % v)
    print('HEAD              : %s' % git('rev-parse', '--short', 'HEAD'))
    print('tags              : %s' % (git('tag', '-l') or '(none)').replace('\n', ' '))
    bad = problems(v)
    for b in bad:
        print('  PROBLEM  ' + b)
    print('OK: ready to tag v%s' % v if not bad else '%d problem(s)' % len(bad))
    return 0 if not bad else 1


def cmd_locks():
    import datetime
    import platform
    v = read_version()
    os.makedirs(ENVS, exist_ok=True)
    today = datetime.date.today().isoformat()
    for env, py in ENV_PYTHON.items():
        if not os.path.isfile(py):
            print('  skip %s: no %s' % (env, py))
            continue
        pyv = subprocess.run([py, '-c', 'import platform;print(platform.python_version())'],
                             capture_output=True, text=True).stdout.strip()
        frozen = subprocess.run([py, '-m', 'pip', 'list', '--format=freeze'],
                                capture_output=True, text=True).stdout.splitlines()
        frozen = [l for l in frozen if not l.lower().startswith(OWN)]
        lock = os.path.join(ENVS, 'lock-%s-win64.txt' % env)
        with open(lock, 'w', encoding='utf-8', newline='\n') as f:
            f.write("# bc-flim-spectra %s -- the '%s' environment, frozen %s on %s, Python %s.\n"
                    % (v, env, today, platform.platform(), pyv))
            f.write('# Exact versions of a working environment: what to install when the ranges in\n'
                    '# pyproject.toml no longer resolve to something that runs. pip form:\n'
                    '#   pip install -r envs/lock-%s-win64.txt\n'
                    '# The conda form of the same environment is envs/conda-%s-win64.txt\n'
                    '# (conda create -n <name> --file envs/conda-%s-win64.txt).\n' % (env, env, env))
            f.write('\n'.join(frozen) + '\n')
        conda = os.path.join(ENVS, 'conda-%s-win64.txt' % env)
        env_name = os.path.basename(os.path.dirname(py))
        res = subprocess.run([CONDA, 'list', '-n', env_name, '--explicit'],
                             capture_output=True, text=True)
        with open(conda, 'w', encoding='utf-8', newline='\n') as f:
            f.write('# bc-flim-spectra %s -- conda explicit spec of the %s environment, %s\n'
                    % (v, env, today))
            f.write(res.stdout)
        print('  %-10s %4d pip pins, %4d conda lines' % (env, len(frozen), len(res.stdout.splitlines())))
    return 0


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
