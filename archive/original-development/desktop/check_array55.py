"""Short matched array/single-rotor runs, constant inflow and restart."""
import argparse,csv,json,shutil,sys
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_array55'


def last(folder):
    row=json.loads((folder/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(folder/'frames'/row['file']) as a:q={k:a[k].copy() for k in a.files}
    with np.load(folder/'frames'/row['epoch']) as a:m={k:a[k].copy() for k in a.files}
    return q,m


def run(backend,variant,mode,duration=None):
    runtime=ROOT/('desktop/qualification/decimal_mesh54/candidate' if variant=='baseline' else 'desktop/qualification/turbine_array55/candidate')
    sys.path.insert(0,str(runtime))
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_core import river_worker
    from braids_desktop import preparation
    verify();install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime);shutil.which=lambda _:None
    folder=OUT/(variant+'_'+mode+'_'+backend);folder.mkdir(exist_ok=False)
    doc=json.loads((ROOT/'desktop/qualification/turbine_clearance53/channel_template/input.json').read_text());doc.pop('editor_state',None)
    end=duration if duration is not None else (60. if mode=='array' else 2.);doc['settings'].update(end_time_s=end,output_interval_s=1. if mode=='array' else .5,checkpoint_interval_s=end-1.)
    doc['settings']['levels']=4;doc['settings']['grid'].update(dx_m=2.,dy_m=2.,nx=15,ny=6)
    doc['model']['request']['execution']['backend']=backend;doc['model']=resolve_model(doc['model']['request']).document()
    if mode=='array':
        base=doc['turbines']['turbines'][0]
        a=dict(deepcopy(base),id='West',center_xy=[2600010.,1200005.]);b=dict(deepcopy(base),id='East RPM',center_xy=[2600010.,1200007.]);c=dict(deepcopy(base),id='Downstream TSR',center_xy=[2600018.,1200006.])
        b['control']=dict(mode='fixed_rpm_table',model_name='Synthetic verification curve',rotational_speed_rpm=90.,outside_curve='off',performance=[dict(tip_speed_ratio=x,thrust_coefficient=ct,power_coefficient=cp) for x,ct,cp in [(0.,0.,0.),(2.,.6,.2),(4.,.8,.4),(6.,.7,.35),(10.,.2,.1)]])
        c['control']=dict(mode='prescribed_tsr_table',tip_speed_ratio=3.67,performance=[dict(tip_speed_ratio=x,thrust_coefficient=ct,power_coefficient=cp) for x,ct,cp in [(2.,.8,.3),(3.67,.8,.4),(6.,.7,.3)]])
        doc['turbines']['turbines']=[a,b,c]
    doc=validate_project(doc);source=folder/'input.json';source.write_text(json.dumps(doc,indent=2)+'\n')
    prepared=preparation.prepare(source,folder/'prepared',asset_cache=OUT/(variant+'_assets'))
    result=river_worker.run(prepared,folder/'run',checkpoint_dir=folder/'checkpoints');assert result['status']=='complete'
    rows=json.loads((folder/'run/TURBINE_PERFORMANCE.json').read_text());names=[r['id'] for r in doc['turbines']['turbines']]
    for time in sorted({r['time_s'] for r in rows}):assert [r['turbine_id'] for r in rows if r['time_s']==time]==names
    assert all(r['active'] for r in rows)
    table=list(csv.DictReader((folder/'run/TURBINE_PERFORMANCE.csv').open(encoding='utf-8')));assert len(table)==len(rows)
    for a,b in zip(table,rows):assert a['turbine_id']==b['turbine_id'] and float(a['useful_power_W'])==b['useful_power_W']
    restart=next((folder/'checkpoints').glob('t_'+str(int(end-1))+'_step_*.npz'))
    resumed=river_worker.run(prepared,folder/'resume',restart=restart);assert resumed['status']=='complete'
    for key in ['q','z','water']:np.testing.assert_array_equal(last(folder/'run')[0][key],last(folder/'resume')[0][key])
    diagnostics=json.loads((folder/'run/DIAGNOSTICS.json').read_text());summary=dict(backend=backend,variant=variant,mode=mode,end_s=end,turbines=names,rows=len(rows),restart_exact=True,compiler_discovery_disabled=True,
        maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),minimum_area_m2=float(np.min(last(folder/'run')[1]['area'])))
    (folder/'CHECK.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=['cpu','gpu']);p.add_argument('--variant',default='candidate',choices=['baseline','candidate']);p.add_argument('--mode',default='array',choices=['array','single']);p.add_argument('--duration',type=float);a=p.parse_args();run(a.backend,a.variant,a.mode,a.duration)
