"""One bounded 600 s no-physics-change settling control, preserving 300 s failure."""
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
BASE=ROOT/'validation/_corner174/long'


def run(backend):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    setup();out=BASE/backend;out.mkdir(parents=True,exist_ok=False)
    d=project(backend);d['settings'].update(end_time_s=640.,sediment_start_time_s=600.,checkpoint_interval_s=640.)
    from braids_core.river_open_boundary import resolve
    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],dict(Upstream='inlet',Downstream='outlet'),
        sediment=d['model']['physics']['sediment'],ng=1,end_time=640.)
    save(out/'project.json',d);prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
    result=river_worker.run(out/'prepared',out/'warmup',until=600.)
    save(out/'WARMUP.json',dict(status=result,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),script_sha256=sha(__file__)))
    print(json.dumps(result),flush=True)


def native():
    from check_bedload161 import replace
    import runtime
    parent=ROOT/'validation/_corner174/delft';out=BASE/'delft';out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest;shutil.copy2(parent/name,out/name)
    p=out/'sim_P057.mdf';code=p.read_text()
    for key,value in dict(Tstop='640',Flmap='0 1 640',Flhis='0 1 640').items():code=replace(code,key,value)
    p.write_text(code);p=out/'mor.mor';p.write_text(replace(p.read_text(),'MorStt','600'))
    for name in ('flume.bct','bcm.bcm'):
        p=out/name;lines=p.read_text().splitlines()
        for i,line in enumerate(lines):
            tokens=line.split()
            if tokens and tokens[0]==f'{340/60:.14g}':lines[i]=f'{640/60:.14g} '+' '.join(tokens[1:])
        p.write_text('\n'.join(lines)+'\n')
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=dict(end_time_s=640,sediment_start_time_s=600,constant_tables_extended_to_minutes=640/60),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=300)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('native','cpu','gpu'));a=ap.parse_args()
    if a.action=='native':native()
    else:run(a.action)
