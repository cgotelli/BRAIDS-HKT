"""Preserve the original development sources without copying build/run caches.

Only explicitly listed software directories and direct maintenance scripts are
read. Files already present byte-for-byte in src are referenced, not duplicated.
Tests, project inputs, measurements, binaries and simulation results are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.py', '.h', '.hpp', '.c', '.cpp', '.cu', '.f', '.f90', '.inc',
              '.json', '.md', '.toml', '.txt', '.svg', '.qss', '.html', '.qrc', '.cmake'}
SKIP = {'build', 'dist', '__pycache__', '.venv', '.git', 'tests', 'test_projects',
        'qualification', 'runtime', 'evidence', 'examples', 'results', 'runs', 'checkpoints'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspaces', type=Path, nargs='+')
    args = parser.parse_args()
    entries = {}
    for workspace in args.workspaces:
        workspace = workspace.resolve()
        files = []
        for name in ('braids_core', 'braids_desktop', 'braids_gpu', 'braids_qgis', 'solver', 'docs'):
            folder = workspace / name
            if not folder.is_dir():
                continue
            for base, dirs, names in os.walk(folder, followlinks=False):
                dirs[:] = [n for n in dirs if n not in SKIP and
                           not (Path(base) / n).is_symlink() and
                           not getattr(Path(base) / n, 'is_junction', lambda: False)()]
                files.extend(Path(base) / n for n in names)
        for name in ('desktop', 'tools'):
            if (workspace / name).is_dir():
                files.extend(p for p in (workspace / name).iterdir() if p.is_file())
        files.extend(workspace / n for n in ('pyproject.toml', '.gitignore', 'README.md')
                     if (workspace / n).is_file())
        for path in files:
            if (path.suffix.lower() not in EXTENSIONS or path.name.startswith('test_')
                    or path.stat().st_size > 10_000_000):
                continue
            name = path.relative_to(workspace).as_posix()
            data = path.read_bytes()
            fingerprint = sha(data)
            if name in entries:
                if entries[name]['sha256'] != fingerprint:
                    raise ValueError('Split workspace has conflicting source copies: ' + name)
                continue
            current = ROOT / 'src' / name
            if current.is_file() and sha(current.read_bytes()) == fingerprint:
                retained = current
            else:
                retained = ROOT / 'archive/original-development' / name
                retained.parent.mkdir(parents=True, exist_ok=True)
                retained.write_bytes(data)
            entries[name] = dict(file=retained.relative_to(ROOT).as_posix(),
                                 sha256=fingerprint, bytes=len(data))
    target = ROOT / 'archive/development-sources.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(dict(schema=1, date='2026-11-12', files=entries),
                                indent=2, sort_keys=True) + '\n', encoding='utf-8')
    copied = [row for row in entries.values() if row['file'].startswith('archive/')]
    print(json.dumps(dict(original_files=len(entries), unique_archived_files=len(copied),
                          archived_bytes=sum(row['bytes'] for row in copied)), indent=2))


if __name__ == '__main__':
    main()
