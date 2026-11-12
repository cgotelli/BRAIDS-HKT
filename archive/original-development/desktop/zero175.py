"""Isolated zero-viscosity hydraulic diagnostics; immutable installed runtime."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_jet173 import document,setup
from stage_wall172 import save,sha
BASE=ROOT/'validation/_zero175'


def run(backend,cfl,end,label,spatial='baseline',case='groin'):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    setup();out=BASE/label/backend
    if spatial=='tangent':
        from corner_candidate167 import install
        install(backend,'tangent')
    elif spatial!='baseline':
        from reconstruction175 import install
        install(backend,spatial)
    out.mkdir(parents=True,exist_ok=False)
    doc=document(backend,end)
    doc['settings']['controls']['cfl']=cfl
    if case=='straight':
        doc['settings']['domain']['vertices_xy']=[[0.,0.],[40.,0.],[40.,8.],[0.,8.],[0.,0.]]
    if case=='free_inlet':
        from braids_core.river_open_boundary import resolve
        doc['settings']['hydraulics']['Upstream']=dict(method='inlet.total_discharge',discharge_m3_s=4.)
        doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
            dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=end)
    save(out/'project.json',doc)
    prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
    status=river_worker.run(out/'prepared',out/'run')
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        script_sha256=sha(__file__),changes=dict(cfl=cfl,end_time_s=end,viscosity=0,spatial=spatial,case=case),
        production_modified=False,Delft3D_modified=False))
    print(json.dumps(status),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu')
    ap.add_argument('--cfl',type=float,default=.5)
    ap.add_argument('--end',type=float,default=300.)
    ap.add_argument('--label',default='half_cfl')
    ap.add_argument('--spatial',choices=('baseline','minmod','surface_constant','tangent'),default='baseline')
    ap.add_argument('--case',choices=('groin','straight','free_inlet'),default='groin')
    args=ap.parse_args();run(args.backend,args.cfl,args.end,args.label,args.spatial,args.case)
