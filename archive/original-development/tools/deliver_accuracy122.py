"""Explicit validation input copies for alpha122, never a project migration API."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_accuracy122'


def inputs():
    from braids_core.river_project import SCHEMA,validate_project
    from braids_core.sediment.current_river_solver import CurrentRiverControls
    from braids_core.river_model import resolve_model
    from braids_gpu.atomic_output import write_json_atomic
    changes=[]
    def copy(source,target,*,backend=None):
        raw=source.read_bytes();doc=json.loads(raw);before=dict(doc['settings']['controls'])
        doc['schema']=SCHEMA
        doc['settings']['controls'].update(nonlinear_atol=CurrentRiverControls.nonlinear_atol,
            temporal_depth_atol=CurrentRiverControls.temporal_depth_atol,
            temporal_momentum_atol=CurrentRiverControls.temporal_momentum_atol,temporal_rtol=CurrentRiverControls.temporal_rtol)
        if backend:
            doc['model']['request']['execution']['backend']=backend
            doc['model']=resolve_model(doc['model']['request']).document()
        dem=Path(doc['terrain']['dem_path'])
        if not dem.is_absolute():doc['terrain']['dem_path']=str((source.parent/dem).resolve())
        validate_project(doc);write_json_atomic(target,doc)
        changes.append(dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),target=str(target),
            previous_controls=before,controls=doc['settings']['controls'],physical_inputs_changed=False))
        assert source.read_bytes()==raw
    out=BASE/'release_inputs';out.mkdir(exist_ok=False)
    record=json.loads((ROOT/'validation/_mesh_modes117/release_record/RECORD.json').read_text())
    for i,row in enumerate(record['projects']):
        source=Path(row['project']);assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
        copy(source,out/f'cpu_{i:03d}.project.json')
    projects=BASE/'projects';projects.mkdir(exist_ok=False)
    for p in (ROOT/'validation/_mesh_modes117/clock_projects').glob('*.json'):
        if p.name.startswith('check_'):copy(p,projects/p.name)
    for backend in ('cpu','gpu'):
        copy(ROOT/'validation/_axis_boundary120/ALIGNED.json',projects/f'River1m_{backend}.json',backend=backend)
    write_json_atomic(BASE/'INPUT_CHANGES.json',dict(scope='Explicit new validation projects; no user-project migration',changes=changes))
    print(len(changes),'new input decks')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inputs','bench','bench-pair'])
    ap.add_argument('--backend',choices=['cpu','gpu'],default='gpu');ap.add_argument('--label')
    ap.add_argument('--outer-dt',type=float);args=ap.parse_args()
    if args.action=='inputs':inputs()
    elif args.action=='bench-pair':
        for backend,label,extra in [('gpu','final_gpu',[]),('cpu','final_cpu',[]),('gpu','final_gpu_outer125',['--outer-dt','.125'])]:
            subprocess.run([sys.executable,__file__,'bench','--backend',backend,'--label',label,*extra],check=True)
    else:
        from braids_core import river_worker
        from tools import profile_test1_amr
        source=BASE/'public_source'
        if not source.exists():river_worker.prepare(BASE/'projects/River1m_cpu.json',source)
        sys.argv=[profile_test1_amr.__file__,'--source',str(source),'--output',str(BASE/args.label),
            '--backend',args.backend,'--until','3.75','--checkout']
        if args.outer_dt:sys.argv+=['--dt-max',str(args.outer_dt)]
        profile_test1_amr.main()
