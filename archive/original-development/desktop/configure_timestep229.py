"""Configure preview50 project controls and run installed CPU/GPU smoke checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import sys,time
import numpy as np


ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent/'BRAIDS_JiangCao_Experiment228'
RUNTIME=PROJECT/'runtime';OUT=ROOT/'validation/_experiment229/installed'
MANIFEST='63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9'
SOURCE='d55826eca99c82867a79744458157b24606f784a1e5e9102cdcfc33e320ed005'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
def frame(output):
    root=output/'frames';ix=json.loads((root/'INDEX.json').read_text());row=ix['frames'][-1]
    with np.load(root/row['file'],allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    sys.path[:0]=[str(RUNTIME),str(ROOT)]
    os.environ.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',QT_QPA_PLATFORM='offscreen')
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)
    from braids_core.river_worker import source_identity
    if sha(RUNTIME/'RUNTIME.json')!=MANIFEST or source_identity()!=SOURCE:raise ValueError('Runtime differs')
    from braids_desktop import services
    from braids_desktop.preparation import prepare
    from braids_core.river_model import resolve_model
    from braids_core.river_project import validate_project
    from braids_core import river_worker
    document=services.load_project_folder(PROJECT);session=services.project_session(PROJECT)
    document['name']='Jiang-Cao Case 1 Run 2 — preview50 CFL0.5 / outer max 0.03 s'
    document['settings']['controls'].update(cfl=.5,threads=8)
    document['settings'].update(dt_max_s=.03,end_time_s=18000.,output_interval_s=60.,checkpoint_interval_s=300.)
    document['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=300.)
    services.save_project_folder(PROJECT,document,source_base=PROJECT,session=session)
    results={}
    frames={}
    prepared_cpu=None
    for backend in ('cpu','gpu'):
        folder=OUT/backend;value=deepcopy(document);request=deepcopy(value['model']['request'])
        request['execution']['backend']=backend;value['model']=resolve_model(request).document()
        value['settings'].update(end_time_s=60.,output_interval_s=5.,checkpoint_interval_s=5.)
        value['editor_state']['checkpoint_clock']=dict(mode='custom',custom_interval_s=5.)
        project=folder/'project.json';save(project,validate_project(value))
        prepared=prepare(project,folder/'prepared',asset_cache=PROJECT/'cache')
        if backend=='cpu':prepared_cpu=prepared
        began=time.perf_counter();result=river_worker.run(prepared,folder/'run',until=5.,checkpoint_dir=folder/'checkpoints')
        results[backend]=dict(result=result,execution_s=time.perf_counter()-began);frames[backend]=frame(folder/'run')
    fields=('q','z','mobile_bedload_solid_m','secondary_hI_m2_s','bedload_vector_m2_s','bedload_capacity_m2_s')
    parity={k:float(np.max(np.abs(frames['cpu'][k]-frames['gpu'][k]))) for k in fields}
    old=PROJECT/'qualification/smoke_cpu/checkpoints/t_5_step_00000250.npz'
    rejection=OUT/'old_checkpoint_rejection';message=''
    try:river_worker.run(prepared_cpu,rejection,restart=old,until=5.)
    except Exception as exc:message=str(exc)
    rejected=('Checkpoint project, physics, backend, runtime or payload differs' in message and not rejection.exists())
    record=dict(runtime_manifest=MANIFEST,source_identity=SOURCE,project=str(PROJECT),
        settings=dict(cfl=.5,dt_max_s=.03,end_time_s=18000.,output_interval_s=60.,checkpoint_interval_s=300.,threads=8),
        results=results,cpu_gpu=parity,old_checkpoint_rejected=rejected,rejection_message=message)
    save(OUT/'CHECK.json',record);save(PROJECT/'TIMESTEP_QUALIFICATION_229.json',record)
    if not (all(r['result']['status']=='complete' and len(r['result']['rejections'])<=5 for r in results.values()) and
            max(parity.values())<1e-10 and rejected):raise RuntimeError(record)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
