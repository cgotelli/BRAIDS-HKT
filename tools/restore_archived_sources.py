"""Restore the original source-only development tree into a NEW directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError('Choose a new, empty destination: ' + str(destination))
    entries = json.loads((ROOT / 'archive/development-sources.json').read_text(encoding='utf-8'))['files']
    for name, entry in entries.items():
        source = (ROOT / entry['file']).resolve()
        target = (destination / name).resolve()
        if not source.is_relative_to(ROOT) or not target.is_relative_to(destination):
            raise ValueError('Inventory path escapes its source/destination')
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Archived source checksum differs: ' + name)
    for name, entry in entries.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / entry['file'], target)
    print(f'Restored {len(entries)} original source files to {destination}')


if __name__ == '__main__':
    main()
