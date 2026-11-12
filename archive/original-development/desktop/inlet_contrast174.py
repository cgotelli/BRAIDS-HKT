"""Existing total-discharge inlet instead of reference-stage inlet; fixed bed."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_corner174 import setup,BASE
from stage_wall172 import save,sha


def run(backend):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    from braids_core.river_open_boundary import resolve
    setup();out=BASE/'inlet_total'/backend;out.mkdir(parents=True,exist_ok=False)
    parent=BASE/'mixing'/backend/'project.json';d=json.loads(parent.read_text())
    d['settings']['hydraulics']['Upstream']=dict(method='inlet.total_discharge',discharge_m3_s=4.)
    d['boundary_resolution']=resolve(d['settings']['hydraulics'],d['boundary_states'],dict(Upstream='inlet',Downstream='outlet'),
        sediment=d['model']['physics']['sediment'],ng=1,end_time=340.)
    save(out/'project.json',d);prepare(out/'project.json',out/'prepared',asset_cache=BASE/'mixing/assets')
    result=river_worker.run(out/'prepared',out/'warmup',until=300.)
    save(out/'WARMUP.json',dict(status=result,parent_project_sha256=sha(parent),
        change='Existing total-discharge inlet instead of reference-stage inlet; no other physical setting changed',
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),script_sha256=sha(__file__)))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('backend',choices=('cpu','gpu'));run(ap.parse_args().backend)
