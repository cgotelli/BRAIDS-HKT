"""Stage ONLY reviewed Python geometry updates over the frozen desktop runtime.

No QGIS package/install or native rebuild. The base manifest is checked before
copying; every inherited artifact remains exact. An existing target is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CHANGED = ('braids_core/sediment/river_polygon_banks.py',
           'braids_core/sediment/river_amr_transfer.py')
ADDED = ('braids_core/sediment/river_polygon_geometry.py',)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(base, target):
    base, target = base.resolve(), target.resolve()
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((base/'RUNTIME.json').read_text())
    for name, digest in manifest['files'].items():
        p = (base/name).resolve()
        if not p.is_relative_to(base) or sha(p) != digest:
            raise ValueError('Base runtime changed: '+name)
    if not set(CHANGED) <= set(manifest['files']) or set(ADDED) & set(manifest['files']):
        raise ValueError('Unexpected base inventory')
    target.mkdir(parents=True)
    for name in manifest['files']:
        dest = target/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base/name, dest)
    changes = []
    for name in (*CHANGED, *ADDED):
        old = manifest['files'].get(name)
        shutil.copy2(ROOT/name, target/name)
        manifest['files'][name] = sha(target/name)
        changes.append(dict(path=name, before=old, after=manifest['files'][name]))
    for name, digest in manifest['files'].items():
        if sha(target/name) != digest:
            raise ValueError('Staged artifact differs: '+name)
    (target/'RUNTIME.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    receipt = dict(base=str(base), base_manifest_sha256=sha(base/'RUNTIME.json'),
                   target=str(target), manifest_sha256=sha(target/'RUNTIME.json'),
                   changes=changes, native_artifacts_unchanged=True,
                   qgis_modified=False, numerical_methods_unchanged=True)
    (target.parent/(target.name+'_STAGING.json')).write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT/'dist/unified-alpha136/braids_qgis/river_runtime')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.base, args.output)
