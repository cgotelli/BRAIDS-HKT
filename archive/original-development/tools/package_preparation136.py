"""Record the bounded archive/setup optimization; preserve frozen alpha135."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_unified_qgis import record, package, merge_bindings, verify, sha, write

REVIEWED = {
    'braids_core/sediment/river_runtime_cache.py',
    'braids_core/sediment/river_runtime_lease.py',
    'braids_gpu/general_live_transport.py',
}
ADDED = {'braids_core/sediment/river_source_io.py'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'dist/unified-alpha136')
    args = parser.parse_args()
    base = ROOT/'validation/_preparation136'
    evidence = base/'release_evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    if any(evidence.iterdir()):
        raise FileExistsError('Preserve existing release evidence: '+str(evidence))
    old = ROOT/'dist/unified-alpha135/braids_qgis'
    manifest = verify(old)
    runtime = json.loads((old/'river_runtime/RUNTIME.json').read_text())
    changes = []
    for relative in runtime['files']:
        # These are recorded copies bundled beside native DLLs, not checkout
        # paths. merge_bindings separately authenticates retained libraries.
        if Path(relative).parts[0] == 'precompiled':
            continue
        before, after = old/'river_runtime'/relative, ROOT/relative
        if before.suffix not in ('.py', '.h', '.cu', '.cpp', '.f90', '.f', '.inc'):
            continue
        if not after.is_file():
            raise ValueError('Runtime source missing: '+relative)
        if sha(before) != sha(after):
            if relative not in REVIEWED:
                raise ValueError('Unreviewed runtime change: '+relative)
            changes.append(dict(path=relative, before=sha(before), after=sha(after)))
    for relative in sorted(ADDED):
        if relative in runtime['files']:
            raise ValueError('Expected new preparation implementation: '+relative)
        changes.append(dict(path=relative, before=None, after=sha(ROOT/relative)))
    retained = ROOT/'validation/_preparation135/release_evidence'
    prior = json.loads((retained/'BINDINGS.json').read_text())
    projects = ROOT/'validation/_performance134/release_projects'
    paths = [projects/f'check_{mode}_cpu_{physics}.json'
             for mode in ('dynamic', 'static') for physics in ('hydro', 'bedload')]
    # Reuse the recorded short cropped-river deck without modifying it.
    paths.append(ROOT/'validation/_preparation135/record_cropped_cpu.json')
    record(evidence, paths)
    fresh = json.loads((evidence/'BINDINGS.json').read_text())
    write(evidence/'BINDINGS.json', merge_bindings(prior, fresh))
    rec = json.loads((evidence/'RECORD.json').read_text())
    oldrec = json.loads((retained/'RECORD.json').read_text())
    rec['seeds'] = sorted(set(rec['seeds']+oldrec['seeds']))
    rec['retained_bindings'] = dict(
        source=str(retained/'BINDINGS.json'), sha256=sha(retained/'BINDINGS.json'),
        previous_release_manifest_sha256=sha(old/'MANIFEST.json'),
        previous_source_identity=manifest['source_identity'],
        unchanged_existing_native_sources=True, reviewed_runtime_changes=changes,
        scope='Source archive preparation and verification only; unchanged numerical kernels',
        current_public_recordings=len(fresh),
        retained_unexercised=sorted(set(prior)-set(fresh)))
    write(evidence/'RECORD.json', rec)
    package(evidence, args.output.resolve())


if __name__ == '__main__':
    main()
