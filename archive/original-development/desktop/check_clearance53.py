"""Short compiler-free packaged checks on the supplied 1 m trapezoidal case."""
import argparse,json,shutil,sys
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_clearance53';TARGET=OUT/'candidate'
sys.path.insert(0,str(TARGET))
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_core.river_project import validate_project
from braids_desktop import preparation


def last(folder):
    d=json.loads((folder/'frames/INDEX.json').read_text());row=d['frames'][-1]
    with np.load(folder/'frames'/row['file']) as f:return {n:f[n] for n in f.files}


def check(backend):
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET);shutil.which=lambda _:None
    folder=OUT/('coupled_'+backend);folder.mkdir(exist_ok=False)
    base=json.loads((OUT/'channel_template/input.json').read_text());request=base['model']['request'];request['execution']['backend']=backend;base['model']=resolve_model(request).document();base['settings'].update(end_time_s=2.,checkpoint_interval_s=.5)
    summaries={};paths={}
    for name in ['normal','surface_off','bed_off']:
        root=folder/name;root.mkdir();doc=deepcopy(base)
        if name=='surface_off':doc['turbines']['turbines'][0]['minimum_surface_clearance_m']=1.0
        if name=='bed_off':
            row=doc['turbines']['turbines'][0];row['center_xy'][1]=1200007.9;row['hub_height_m']=1.;row['minimum_bed_clearance_m']=.4
        doc=validate_project(doc);source=root/'input.json';source.write_text(json.dumps(doc,indent=2)+'\n')
        p=preparation.prepare(source,root/'prepared',asset_cache=OUT/'assets');paths[name]=p
        summary=river_worker.run(p,root/'run',until=2. if name=='normal' else .5,checkpoint_dir=root/'checkpoints');assert summary['status'] in ['complete','segment_complete']
        rows=json.loads((root/'run/TURBINE_PERFORMANCE.json').read_text())
        if name=='normal':assert all(r['active'] for r in rows)
        else:
            reason='insufficient_surface_clearance' if name=='surface_off' else 'insufficient_bed_clearance'
            assert all(not r['active'] and r['inactive_reason']==reason and r['thrust_N']==r['useful_power_W']==r['hydraulic_work_W']==0 for r in rows),[(r['inactive_reason'],r['minimum_coverage_margin_m'],r['minimum_bed_clearance_m']) for r in rows]
        summaries[name]=dict(status=summary['status'],time_s=summary['time_s'],rows=rows)
    checkpoint=next((folder/'normal/checkpoints').glob('t_1_step_*.npz'))
    result=river_worker.run(paths['normal'],folder/'resume',restart=checkpoint);assert result['status']=='complete'
    for field in ['q','z','water']:np.testing.assert_array_equal(last(folder/'normal/run')[field],last(folder/'resume')[field])
    try:river_worker.run(paths['surface_off'],folder/'wrong_limit',restart=checkpoint)
    except ValueError:pass
    else:raise AssertionError('Changed clearance accepted an old checkpoint')
    assert not (folder/'wrong_limit').exists()
    receipt=dict(backend=backend,compiler_discovery_disabled=True,restart_exact=True,changed_clearance_restart_rejected=True,cases=summaries)
    (folder/'CHECK.json').write_text(json.dumps(receipt,indent=2)+'\n');print(backend,'normal + both clearance shutdowns + exact restart passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=['cpu','gpu']);check(p.parse_args().backend)
