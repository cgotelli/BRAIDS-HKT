"""Make the qualified installed runtime the single local BRAIDS application.

This is a local development update, not a release builder.  It preserves the
existing installation path so the original Windows uninstaller remains valid.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
INSTALL = Path(os.environ['LOCALAPPDATA']) / 'Programs' / 'BRAIDS-HKT'
RUNTIME = INSTALL / 'runtime'
BACKUP = ROOT / 'validation/sediment/bedload_validation/reports/unified152/before'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, data):
    temporary = path.with_name(path.name + '.braids-unify-tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main():
    paths = [INSTALL / 'launch.py', RUNTIME / 'braids_desktop/app.py',
             RUNTIME / 'braids_desktop/shell.py', RUNTIME / 'RUNTIME.json',
             ROOT / 'Launch BRAIDS Desktop.cmd']
    if BACKUP.exists():
        raise RuntimeError('Backup already exists; do not apply this update twice')
    manifest = json.loads(paths[3].read_text(encoding='utf-8'))
    for path in paths[1:3]:
        name = path.relative_to(RUNTIME).as_posix()
        if sha(path) != manifest['files'][name]:
            raise RuntimeError('Installed runtime has changed: ' + name)
    originals = {p: p.read_bytes() for p in paths}
    app = originals[paths[1]].decode('utf-8')
    shell = originals[paths[2]].decode('utf-8')
    launcher = originals[paths[0]].decode('utf-8')
    old_app = "app.setApplicationName('BRAIDS-HKT');app.setApplicationDisplayName('BRAIDS-HKT')"
    new_app = "app.setApplicationName('BRAIDS');app.setApplicationDisplayName('BRAIDS')"
    if app.count(old_app) != 1 or app.count("'BRAIDS.HKT'") != 1 or shell.count("self.setWindowTitle('BRAIDS-HKT')") != 1:
        raise RuntimeError('Unexpected branding in the installed application')
    app = app.replace(old_app, new_app).replace("'BRAIDS.HKT'", "'BRAIDS.Desktop'")
    shell = shell.replace("self.setWindowTitle('BRAIDS-HKT')", "self.setWindowTitle('BRAIDS')")
    launcher = launcher.replace('Installed BRAIDS-HKT entry point.', 'Installed BRAIDS entry point.')
    # Preserve the existing log directory and settings: this is a branding and
    # launch unification, not a project or checkpoint migration.
    app_bytes, shell_bytes = app.encode('utf-8'), shell.encode('utf-8')
    manifest['files']['braids_desktop/app.py'] = hashlib.sha256(app_bytes).hexdigest()
    manifest['files']['braids_desktop/shell.py'] = hashlib.sha256(shell_bytes).hexdigest()
    manifest['distribution'] = 'BRAIDS local unified runtime (original installer 0.1.0)'
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    command = ('@echo off\r\nsetlocal\r\n'
               'set "BRAIDS_APP=%LOCALAPPDATA%\\Programs\\BRAIDS-HKT"\r\n'
               'if not exist "%BRAIDS_APP%\\python\\pythonw.exe" (\r\n'
               '  echo BRAIDS is not installed at "%BRAIDS_APP%".\r\n  pause\r\n  exit /b 1\r\n)\r\n'
               'if /i "%~1"=="--smoke" (\r\n'
               '  "%BRAIDS_APP%\\python\\python.exe" -I -B "%BRAIDS_APP%\\launch.py" %*\r\n'
               '  exit /b\r\n)\r\n'
               'start "" /b "%BRAIDS_APP%\\python\\pythonw.exe" -I -B "%BRAIDS_APP%\\launch.py" %*\r\n'
               'exit /b\r\n').encode('utf-8')
    BACKUP.mkdir(parents=True)
    for i, path in enumerate(paths):
        (BACKUP / (str(i) + '-' + path.name)).write_bytes(originals[path])
    try:
        for path, data in zip(paths, [launcher.encode('utf-8'), app_bytes,
                                      shell_bytes, manifest_bytes, command]):
            atomic(path, data)
    except Exception:
        for path, data in originals.items():
            atomic(path, data)
        raise
    receipt = {'old_manifest_sha256': hashlib.sha256(originals[paths[3]]).hexdigest(),
               'new_manifest_sha256': sha(paths[3]), 'installed_runtime': str(RUNTIME),
               'changed_runtime_files': ['braids_desktop/app.py', 'braids_desktop/shell.py'],
               'solver_changes': False, 'installer_created': False, 'published': False}
    atomic(BACKUP.parent / 'INSTALL.json', (json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
