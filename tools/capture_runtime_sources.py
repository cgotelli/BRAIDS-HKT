"""Capture authenticated application sources, never projects or compiled payloads.

Used for the 2026-11-12 recovery audit. This is an explicit maintenance command,
not part of the application or its startup path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {'.py', '.h', '.hpp', '.c', '.cpp', '.cu', '.f', '.f90',
                   '.inc', '.html', '.svg', '.qss', '.qrc', '.json', '.txt'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    manifest = json.loads((runtime / 'RUNTIME.json').read_text(encoding='utf-8'))
    copied = {}
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.suffix.lower() not in SOURCE_SUFFIXES or name == 'PRECOMPILED.json':
            continue
        source = (runtime / relative).resolve()
        if not source.is_relative_to(runtime):
            raise ValueError('Manifest path escapes runtime: ' + name)
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Installed source checksum differs: ' + name)
        target = ROOT / 'src' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
        copied[name] = expected
    shutil.copy2(runtime / 'BRAIDS_logo.png', ROOT / 'src/BRAIDS_logo.png')
    for name, target in [('RUNTIME.json', 'installed-runtime.reference.json'),
                         ('PRECOMPILED.json', 'native-bindings.reference.json')]:
        shutil.copy2(runtime / name, ROOT / 'packaging' / target)
    receipt = dict(schema=1, date='2026-11-12',
                   description='Authenticated installed source capture; binaries are rebuilt, not versioned.',
                   files=copied)
    (ROOT / 'packaging/source-snapshot.json').write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Captured {len(copied)} authenticated source/resource files')


if __name__ == '__main__':
    main()
