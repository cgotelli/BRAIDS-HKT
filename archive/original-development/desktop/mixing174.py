"""Authorized single matched constant-viscosity diagnostic, not a default change."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop'),
             str(ROOT/'validation/sediment/delft3d_current')]
from stage_wall172 import save,sha
from check_corner174 import setup,project
BASE=ROOT/'validation/_corner174/mixing'
VISCOSITY=.05


def run(backend,resume=False):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    from braids_core.river_model import resolve_model
    setup();out=BASE/backend
    if not resume:
        out.mkdir(parents=True,exist_ok=False)
        d=project(backend)
        d['model']['request']['hydrodynamics']['turbulence']='constant_eddy_viscosity'
        d['model']=resolve_model(d['model']['request']).document()
        d['settings']['controls']['viscosity']=VISCOSITY
        save(out/'project.json',d)
        prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
        result=river_worker.run(out/'prepared',out/'warmup',until=300.)
        name='WARMUP.json'
    else:
        assert json.loads((BASE/'SETTLING.json').read_text())['both_pass']
        result=river_worker.run(out/'prepared',out/'mobile',restart=out/'warmup/checkpoint.npz')
        name='MOBILE.json'
    save(out/name,dict(status=result,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        script_sha256=sha(__file__),diagnostic_only=True,constant_viscosity_m2_s=VISCOSITY))
    print(json.dumps(result),flush=True)


def native():
    from check_bedload161 import replace
    import runtime
    parent=ROOT/'validation/_corner174/delft';out=BASE/'delft'
    out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest;shutil.copy2(parent/name,out/name)
    p=out/'sim_P057.mdf';p.write_text(replace(p.read_text(),'Vicouv',str(VISCOSITY)))
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=dict(Vicouv=VISCOSITY),diagnostic_only=True,
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=240)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('native','cpu','gpu'))
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    if a.action=='native':native()
    else:run(a.action,a.resume)
