"""Bounded packaged replay of the user's decimal-grid turbine channel."""
import argparse,json,shutil,sys
from copy import deepcopy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/decimal_mesh54';TARGET=OUT/'candidate'
sys.path.insert(0,str(TARGET))
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_core.river_project import validate_project
from braids_desktop import preparation


def frame(folder):
    rows=json.loads((folder/'frames/INDEX.json').read_text())['frames']
    with np.load(folder/'frames'/rows[-1]['file']) as f:return {k:f[k] for k in f.files}


def check(backend):
    verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET);shutil.which=lambda _:None
    user=Path(r'C:\Users\STREEM\Desktop\BRAIDS_1m_Turbine_Channel')
    base=json.loads((OUT/'user_snapshot/Channel.braids.json').read_text())['project']
    base['terrain']['dem_path']=str(user/'project_data'/base['terrain']['dem_path'])
    base['settings'].update(end_time_s=.1,output_interval_s=.05,checkpoint_interval_s=.05)
    base['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=.05)
    base['model']['request']['execution']['backend']=backend;base['model']=resolve_model(base['model']['request']).document()
    old=json.loads((ROOT/'desktop/qualification/turbine_clearance53/channel_template/input.json').read_text())
    folder=OUT/('qualified_'+backend);folder.mkdir(exist_ok=False);checks={}
    for name in ['off','on','rpm']:
        p=folder/name;p.mkdir();doc=deepcopy(base);doc['turbines']=None if name=='off' else deepcopy(old['turbines'])
        if name=='rpm':
            doc['turbines']['turbines'][0]['control']=dict(mode='fixed_rpm_table',model_name='Synthetic verification curve — replace for real turbine',rotational_speed_rpm=90.,outside_curve='off',
                performance=[dict(tip_speed_ratio=x,thrust_coefficient=ct,power_coefficient=cp) for x,ct,cp in [(0.,0.,0.),(2.,.6,.2),(4.,.8,.4),(6.,.7,.35),(10.,.2,.1)]])
        doc=validate_project(doc);source=p/'input.json';source.write_text(json.dumps(doc,indent=2)+'\n')
        prepared=preparation.prepare(source,p/'prepared',asset_cache=OUT/'assets')
        result=river_worker.run(prepared,p/'run',checkpoint_dir=p/'checkpoints');assert result['status']=='complete'
        diagnostics=json.loads((p/'run/DIAGNOSTICS.json').read_text())
        check=dict(time_s=result['time_s'],maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics))
        assert check['maximum_water_residual_m3']<1e-8
        if name!='off':
            rows=json.loads((p/'run/TURBINE_PERFORMANCE.json').read_text());assert all(r['active'] for r in rows)
            checkpoint=next((p/'checkpoints').glob('t_0.05_step_*.npz'))
            resumed=river_worker.run(prepared,p/'resume',restart=checkpoint)
            assert resumed['status']=='complete'
            for k in ['q','z','water']:np.testing.assert_array_equal(frame(p/'run')[k],frame(p/'resume')[k])
            check['restart_exact']=True;check['last_turbine']=rows[-1]
            import csv
            table=list(csv.DictReader((p/'run/TURBINE_PERFORMANCE.csv').open(encoding='utf-8')))
            assert len(table)==len(rows)
            assert {'time_s','turbine_id','upstream_velocity_m_s','local_velocity_m_s','rotational_speed_rpm','tip_speed_ratio','thrust_coefficient','power_coefficient','thrust_N','useful_power_W','performance_model','active'}<=set(table[0])
            assert float(table[-1]['useful_power_W'])==rows[-1]['useful_power_W']
            if name=='rpm':assert rows[0]['tip_speed_ratio']!=rows[-1]['tip_speed_ratio'] and rows[0]['power_coefficient']!=rows[-1]['power_coefficient']
        else:assert not (p/'run/TURBINE_PERFORMANCE.json').exists()
        checks[name]=check
    (folder/'CHECK.json').write_text(json.dumps(checks,indent=2)+'\n');print(backend,'decimal-grid on/off, budgets and exact restart passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',required=True,choices=['cpu','gpu']);check(p.parse_args().backend)
