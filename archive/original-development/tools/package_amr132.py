"""Record alpha132's initializer and retain authenticated unchanged native bindings."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_unified_qgis import record,package,merge_bindings,verify,sha,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path,default=ROOT/'validation/_amr132/release_evidence')
    parser.add_argument('--output',type=Path,default=ROOT/'dist/unified-alpha132')
    args=parser.parse_args()
    evidence=args.evidence.resolve()
    if evidence.exists():raise FileExistsError(evidence)
    evidence.mkdir(parents=True)
    old=ROOT/'dist/unified-alpha131/braids_qgis'
    manifest=verify(old)
    allowed={
        'braids_core/sediment/adaptive_river_worker.py',
        'braids_core/sediment/river_leaf_io.py',
        'braids_core/sediment/portable_river_runtime.py',
    }
    differences=[]
    runtime=json.loads((old/'river_runtime/RUNTIME.json').read_text())
    for relative in runtime['files']:
        previous=old/'river_runtime'/relative;current=ROOT/relative
        if previous.suffix not in ('.py','.h','.cu','.cpp','.f90','.f','.inc') or not current.is_file():continue
        if sha(current)!=sha(previous):
            if relative not in allowed:raise ValueError('Unreviewed retained runtime source change: '+relative)
            differences.append(dict(path=relative,before=sha(previous),after=sha(current)))
    retained=ROOT/'validation/_amr131/release_evidence/BINDINGS.json'
    prior=json.loads(retained.read_text())
    projects=ROOT/'validation/_amr131/release_projects'
    record(evidence,[projects/f'check_{mode}_cpu_{physics}.json'
        for mode in ('dynamic','static') for physics in ('hydro','bedload')])
    fresh=json.loads((evidence/'BINDINGS.json').read_text())
    merged=merge_bindings(prior,fresh)
    if 'braids_core.sediment.river_mesh_records.build' not in fresh:
        raise ValueError('New initializer was not recorded through public execution')
    write(evidence/'BINDINGS.json',merged)
    rec=json.loads((evidence/'RECORD.json').read_text())
    prior_record=json.loads((ROOT/'validation/_amr131/release_evidence/RECORD.json').read_text())
    rec['seeds']=sorted(set(rec['seeds']+prior_record['seeds']))
    rec['retained_bindings']=dict(source=str(retained),sha256=sha(retained),
        release_sha256=sha(old/'MANIFEST.json'),previous_source_identity=manifest['source_identity'],
        unchanged_native_sources=True,reviewed_python_changes=differences,
        current_public_recordings=len(fresh),retained_unexercised=sorted(set(prior)-set(fresh)),
        scope='Unchanged previously qualified native operators retained; not relabelled as newly tested physics.')
    write(evidence/'RECORD.json',rec)
    package(evidence,args.output.resolve())


if __name__=='__main__':main()
