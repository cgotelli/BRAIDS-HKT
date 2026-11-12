"""Package folder persistence with unchanged qualified numerical bindings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_unified_qgis import record, package, merge_bindings, verify, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT/'validation/_project133/release_evidence')
    parser.add_argument('--output', type=Path, default=ROOT/'dist/unified-alpha133')
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    if evidence.exists():
        raise FileExistsError(evidence)
    evidence.mkdir(parents=True)
    old = ROOT/'dist/unified-alpha132-final/braids_qgis'
    manifest = verify(old)
    runtime = json.loads((old/'river_runtime/RUNTIME.json').read_text())
    reviewed = {'braids_core/river_worker.py', 'tools/build_unified_qgis.py'}
    differences = []
    for relative in runtime['files']:
        before, after = old/'river_runtime'/relative, ROOT/relative
        if before.suffix not in ('.py', '.h', '.cu', '.cpp', '.f90', '.f', '.inc') or not after.is_file():
            continue
        if sha(before) != sha(after):
            if relative not in reviewed:
                raise ValueError('Unreviewed runtime source change: '+relative)
            differences.append(dict(path=relative, before=sha(before), after=sha(after)))
    # Record fresh public executions to bind the new provenance owner. Native
    # operators not exercised here retain their separately qualified receipts.
    retained = ROOT/'validation/_amr132/release_final/BINDINGS.json'
    prior = json.loads(retained.read_text())
    projects = ROOT/'validation/_amr131/release_projects'
    record(evidence, [projects/f'check_{mode}_cpu_{physics}.json'
                     for mode in ('dynamic', 'static') for physics in ('hydro', 'bedload')])
    fresh = json.loads((evidence/'BINDINGS.json').read_text())
    write(evidence/'BINDINGS.json', merge_bindings(prior, fresh))
    rec = json.loads((evidence/'RECORD.json').read_text())
    prior_record = json.loads(retained.with_name('RECORD.json').read_text())
    rec['seeds'] = sorted(set(rec['seeds']+prior_record['seeds']))
    rec['retained_bindings'] = dict(source=str(retained), sha256=sha(retained),
        release_sha256=sha(old/'MANIFEST.json'), previous_source_identity=manifest['source_identity'],
        unchanged_numerical_sources=True, reviewed_python_changes=differences,
        added_folder_source_sha256=sha(ROOT/'braids_core/river_folder.py'),
        current_public_recordings=len(fresh), retained_unexercised=sorted(set(prior)-set(fresh)),
        scope='Persistence-only changes; no hydraulic, CFL, sediment or AMR algorithm change.')
    write(evidence/'RECORD.json', rec)
    package(evidence, args.output.resolve())


if __name__ == '__main__':
    main()
