"""Qualify the JC stage-wave outer CFL predictor and exact restart."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent/'BRAIDS_JiangCao_Experiment228'
RUNTIME=ROOT/'validation/_experiment229/candidate_v2'
OUT=ROOT/'validation/_experiment229/predictor_v2'
PREVIOUS=ROOT/'validation/_experiment229/timestep'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def imports():
    sys.path[:0]=[str(RUNTIME),str(ROOT)]
    os.environ.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',QT_QPA_PLATFORM='offscreen')
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)


def frame(output,time_s):
    root=output/'frames';index=json.loads((root/'INDEX.json').read_text())
    row=next(r for r in index['frames'] if abs(r['time_s']-time_s)<1e-9)
    with np.load(root/row['file'],allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def compare(a,b):
    fields=('q','z','mobile_bedload_solid_m','secondary_hI_m2_s',
            'bedload_vector_m2_s','bedload_capacity_m2_s','state')
    return {k:float(np.max(np.abs(a[k]-b[k]))) for k in fields}


def run(prepared,label,until,restart=None):
    from braids_core import river_worker
    folder=OUT/label
    if folder.exists():raise FileExistsError(folder)
    started=time.perf_counter()
    result=river_worker.run(prepared,folder/'run',until=until,restart=restart,
        checkpoint_dir=folder/'checkpoints')
    row=dict(label=label,until_s=until,execution_s=time.perf_counter()-started,
        result=result,restart=None if restart is None else str(restart))
    save(folder/'RESULT.json',row);return row


def main():
    if OUT.exists():raise FileExistsError(f'Preserve evidence: {OUT}')
    imports()
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core.sediment.jiang_cao_desktop import outer_cfl_hint
    assert outer_cfl_hint(.03,.03,.9,.5,.03)<.017
    assert outer_cfl_hint(.015,.015,.45,.5,.03)>.016
    document=services.load_project_folder(PROJECT);request=deepcopy(document['model']['request'])
    request['execution']['backend']='cpu';document['model']=resolve_model(request).document()
    document['settings']['controls'].update(cfl=.5,threads=8)
    document['settings'].update(dt_max_s=.03,end_time_s=30.,output_interval_s=15.,checkpoint_interval_s=15.)
    document['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=15.)
    project=OUT/'project.json';save(project,validate_project(document))
    prepared=prepare(project,OUT/'prepared',asset_cache=PROJECT/'cache')
    full=run(prepared,'full30',30.)
    first=run(prepared,'first15',15.)
    checkpoint=OUT/'first15/checkpoints/t_15_step_{:08d}.npz'.format(first['result']['steps'])
    continued=run(prepared,'restart30',30.,checkpoint)
    final=frame(OUT/'full30/run',30.);restarted=frame(OUT/'restart30/run',30.)
    restart=compare(final,restarted)
    old_candidate=json.loads((PREVIOUS/'cfl050_dt0030/RESULT.json').read_text())
    old_control=json.loads((PREVIOUS/'cfl050_dt0020/RESULT.json').read_text())
    against_old=compare(final,frame(PREVIOUS/'cfl050_dt0030/run',30.))
    against_control=compare(final,frame(PREVIOUS/'cfl050_dt0020/run',30.))
    diagnostics=json.loads((OUT/'full30/run/DIAGNOSTICS.json').read_text())[-1]
    checks=dict(rejections=len(full['result']['rejections']),steps=full['result']['steps'],
        execution_s=full['execution_s'],old_retry_execution_s=old_candidate['execution_s'],
        speedup_over_retry=old_candidate['execution_s']/full['execution_s'],
        bed_vs_old_retry_mm=1000*against_old['z'],depth_vs_old_retry_mm=1000*against_old['q'],
        bed_vs_dt002_control_mm=1000*against_control['z'],depth_vs_dt002_control_mm=1000*against_control['q'],
        restart_max=max(restart.values()),solid_residual_m3=diagnostics['solid_residual_m3'],
        water_plus_pore_residual_m3=diagnostics['water_plus_pore_residual_m3'])
    accepted=(checks['rejections']<=5 and checks['speedup_over_retry']>1.2 and
        checks['bed_vs_old_retry_mm']<1. and checks['depth_vs_old_retry_mm']<2. and
        checks['restart_max']==0. and abs(checks['solid_residual_m3'])<1e-12 and
        abs(checks['water_plus_pore_residual_m3'])<1e-12)
    record=dict(runtime=sha(RUNTIME/'RUNTIME.json'),full=full,first=first,continued=continued,
        previous=dict(dt003=old_candidate,dt002=old_control),against_old_retry=against_old,
        against_dt002_control=against_control,restart=restart,checks=checks,accepted=accepted,
        selected=dict(cfl=.5,dt_max_s=.03,controller='stage-wave outer CFL predictor') if accepted else None)
    save(OUT/'QUALIFICATION.json',record);print(json.dumps(dict(checks=checks,accepted=accepted,
        runtime=record['runtime']),indent=2),flush=True)
    if not accepted:raise RuntimeError('Outer predictor did not qualify')


if __name__=='__main__':main()
