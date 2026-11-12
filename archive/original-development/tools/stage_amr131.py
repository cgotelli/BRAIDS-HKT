"""Authenticated compiler-free benchmark copies; never writes installed QGIS.

The alpha129 payload and native binding table are frozen. Only reviewed Python
files are replaced; every non-Python runtime source must match the release.
The before copy reconstructs milestone130 using retained pre-edit snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_unified_qgis import verify, payload_files, write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(destination, before=False):
    original = ROOT/'dist/unified-alpha129/braids_qgis'
    old = verify(original)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(original, destination, ignore=shutil.ignore_patterns('__pycache__'))
    runtime = destination/'river_runtime'
    snapshots = ROOT/'validation/_amr131/source_before'
    changes = []
    for target in payload_files(runtime):
        rel = target.relative_to(runtime)
        source = ROOT/rel
        if target.suffix not in ('.py', '.h', '.cu', '.cpp', '.f90', '.f', '.inc') or not source.is_file():
            continue
        if before and (snapshots/rel).is_file():
            source = snapshots/rel
        if sha(source) == sha(target):
            continue
        if target.suffix != '.py':
            raise ValueError('Native source changed; recorded binding reuse forbidden: '+str(rel))
        changes.append(dict(path=rel.as_posix(), before=sha(target), after=sha(source), source=str(source)))
        shutil.copy2(source, target)
    rdoc = json.loads((runtime/'RUNTIME.json').read_text())
    rdoc['files'] = {p.relative_to(runtime).as_posix():sha(p) for p in payload_files(runtime) if p.name!='RUNTIME.json'}
    write(runtime/'RUNTIME.json', rdoc)
    write(destination/'AMR131_BENCHMARK.json',dict(release=False, original_release=old['version'],
        baseline130=before, native_bindings_unchanged=True, replacements=changes,
        original_manifest_sha256=sha(original/'MANIFEST.json')))
    doc=dict(old)
    doc['files']={p.relative_to(destination).as_posix():sha(p) for p in payload_files(destination) if p.name!='MANIFEST.json'}
    write(destination/'MANIFEST.json',doc)
    verify(destination)
    print(json.dumps(dict(runtime=str(runtime),before=before,python_changes=changes),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--before',action='store_true')
    a=p.parse_args()
    stage(a.destination.resolve(),a.before)
