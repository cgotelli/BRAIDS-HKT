"""Build fresh milestone134 validation decks and record the I/O-only release.

This is an identified validation-input construction, not a product migration
reader. Original alpha133 projects and frozen runtimes are never overwritten.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_unified_qgis import record,package,merge_bindings,verify,sha,write

REVIEWED={
    'braids_gpu/atomic_output.py',
    'braids_core/sediment/shoreline_solver.py',
    'braids_core/sediment/river_mixtures.py',
    'braids_core/sediment/river_leaf_controller.py',
    'braids_core/sediment/river_field_policy.py',
    'braids_core/sediment/adaptive_river_worker.py',
    'braids_core/river_worker.py',
    'braids_core/river_project.py',
}


def inputs(root):
    from braids_core.river_project import SCHEMA,validate_project
    receipts=[]
    for previous,target in ((ROOT/'validation/_amr131/release_projects',root/'release_projects'),
                            (ROOT/'validation/_performance134/matched',root/'current_inputs')):
        target.mkdir(parents=True,exist_ok=True)
        for path in sorted(previous.glob('*.json')):
            doc=json.loads(path.read_text())
            if not isinstance(doc,dict) or doc.get('schema')!='braids.river-project.v19':continue
            doc['schema']=SCHEMA;doc['settings']['checkpoint_interval_s']=60.
            validate_project(doc)
            result=target/path.name
            if result.exists():
                if json.loads(result.read_text())!=doc:raise ValueError('Existing validation input differs: '+str(result))
            else:write(result,doc)
            receipts.append(dict(source=str(path),source_sha256=sha(path),created=str(result),created_sha256=sha(result),
                changes={'schema':SCHEMA,'settings.checkpoint_interval_s':60.},
                purpose='Fresh validation input with unchanged geometry, physics, CFL, outer and solution clocks'))
    write(root/'CURRENT_INPUTS.json',receipts)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inputs','package'])
    p.add_argument('--root',type=Path,default=ROOT/'validation/_performance134')
    p.add_argument('--destination',type=Path,default=ROOT/'dist/unified-alpha134')
    a=p.parse_args();root=a.root.resolve()
    if a.action=='inputs':inputs(root);return
    evidence=root/'release_evidence';evidence.mkdir(parents=True,exist_ok=False)
    old=ROOT/'dist/unified-alpha133/braids_qgis';manifest=verify(old)
    runtime=json.loads((old/'river_runtime/RUNTIME.json').read_text());differences=[]
    for relative in runtime['files']:
        before,after=old/'river_runtime'/relative,ROOT/relative
        if before.suffix not in ('.py','.h','.cu','.cpp','.f90','.f','.inc') or not after.is_file():continue
        if sha(before)!=sha(after):
            if relative not in REVIEWED:raise ValueError('Unreviewed runtime change: '+relative)
            differences.append(dict(path=relative,before=sha(before),after=sha(after)))
    retained=ROOT/'validation/_project133/release_final/BINDINGS.json'
    if not retained.is_file():retained=ROOT/'validation/_project133/release_evidence/BINDINGS.json'
    prior=json.loads(retained.read_text())
    projects=root/'release_projects'
    record(evidence,[projects/f'check_{mode}_cpu_{physics}.json'
                     for mode in ('dynamic','static') for physics in ('hydro','bedload')])
    fresh=json.loads((evidence/'BINDINGS.json').read_text())
    write(evidence/'BINDINGS.json',merge_bindings(prior,fresh))
    rec=json.loads((evidence/'RECORD.json').read_text());previous=json.loads(retained.with_name('RECORD.json').read_text())
    rec['seeds']=sorted(set(rec['seeds']+previous['seeds']))
    rec['retained_bindings']=dict(source=str(retained),sha256=sha(retained),
        previous_release_sha256=sha(old/'MANIFEST.json'),previous_source_identity=manifest['source_identity'],
        unchanged_native_sources=True,reviewed_python_changes=differences,
        current_public_recordings=len(fresh),retained_unexercised=sorted(set(prior)-set(fresh)),
        scope='Checkpoint serialization/cadence and publication only; no hydraulic, sediment or AMR numerical changes')
    write(evidence/'RECORD.json',rec)
    package(evidence,a.destination.resolve())


if __name__=='__main__':main()
