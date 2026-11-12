"""Short public-worker qualification of polygon/ellipse walls on CPU and CUDA."""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_solids151'
runtime=OUT/'candidate';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_desktop.demo import create_demo
from braids_desktop import preparation,services
from braids_core import river_worker
from braids_core.river_project import worker_profile
from braids_core.sediment.adaptive_river_worker import hierarchy,read_live_frame
from braids_core.river_model import new_bedload_selection,resolve_model
from braids_core.river_open_boundary import new_state,resolve
from braids_core.boundaries.domain import PolygonDomain

parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','gpu'],required=True)
parser.add_argument('--case',choices=['lake','wetfront','bedload'],required=True)
parser.add_argument('--name',default='');args=parser.parse_args()
case=OUT/((args.name or args.case)+'_'+args.backend);duration=6. if args.case=='wetfront' else 2.
doc=create_demo(case,backend=args.backend,end_time_s=duration)
angles=np.arange(64)*2*np.pi/64
doc['settings']['solids']=[dict(id='polygon',points_xy=[[3.2,3.1],[6.4,3.3],[6.1,5.4],[3.5,5.2]]),
    dict(id='ellipse',points_xy=([12.,4.]+np.column_stack((np.cos(angles),np.sin(angles)))*[2.,1.3]).tolist())]
doc['settings']['breaklines']=[dict(id='permeable',closed=False,points_xy=[[1.,4.],[18.,4.]])]
doc['settings']['output_interval_s']=.25;doc['settings']['dt_max_s']=.05
doc['settings']['adaptation']['interval']=.25
if args.case=='wetfront':
    doc['settings']['adaptive']=True;doc['settings']['adaptation']['terrain_enabled']=False
    doc['settings']['hydraulics']['Upstream']['discharge_m3_s']=3.2
    doc['settings']['levels']=3;doc['settings']['adaptation']['buffer_cells']=1
    doc['settings']['grid']=services.geometry_grid(PolygonDomain(**doc['settings']['domain']),1.,3)
else:
    doc['initial']['stage_m']=.5
    if args.case=='lake':
        for name in list(doc['settings']['hydraulics']):doc=services.clear_boundary(doc,name)
    else:
        doc['initial']['velocity_m_s']=[1.,0.]
        doc['settings']['hydraulics']['Upstream']['discharge_m3_s']=3.2
        selection=new_bedload_selection()
        selection['bed']=dict(composition='bed.layered',transport_layer='transport_layer.constant',coarse_layer=False,sorting='sorting.off')
        request=deepcopy(doc['model']['request']);request['sediment']=dict(enabled=True,selection=selection)
        doc['model']=resolve_model(request).document()
        for name,control in doc['settings']['hydraulics'].items():doc['boundary_states'][name]=new_state(control['method'],sediment=True)
        doc['settings']['sediment_boundaries']['Upstream']['solid_flux_m2_s']=[1e-5]
doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
    {a['bc_id']:a['kind'] for a in doc['settings']['assignments'] if a['kind']!='wall'},
    sediment=doc['model']['physics']['sediment'],ng=1,end_time=duration)
services.save_project_folder(case,doc)
prepared=preparation.prepare(case/'braids_project.json',case/'prepared/current',asset_cache=case/'model_layers/preparation_cache')
tree=hierarchy(worker_profile(doc));start=time.perf_counter();result=river_worker.run(prepared,case/'run')
elapsed=time.perf_counter()-start
if args.case=='bedload':
    river_worker.run(prepared,case/'partial',until=1.)
    river_worker.run(prepared,case/'restarted',restart=case/'partial/checkpoint.npz')
rows=json.loads((case/'run/DIAGNOSTICS.json').read_text());events=json.loads((case/'run/ADAPTATION.json').read_text())
row,mesh,values=read_live_frame(case/'run')
report=dict(case=args.case,backend=args.backend,run=result,wall_seconds=elapsed,
    minimum_area_m2=min(c.area_m2 for c in tree.finest.cells),finest_cells=len(tree.finest.cells),
    mesh_metrics=tree.finest.metrics(),minimum_depth_m=float(values['q'][0].min()),
    maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
    maximum_class_residual_kg=max(max(abs(x) for x in r['class_residual_kg']) for r in rows),
    maximum_wall_integral=max(max(abs(x) for x in r['wall_integrals']) for r in rows),
    adaptations=[dict(time_s=e['time_s'],changed=e['changed'],cells=e['to_cells']) for e in events])
if args.case=='lake':
    report['lake_stage_error_m']=float(abs(values['q'][0]+values['z']-.5).max())
    report['lake_momentum_max']=float(abs(values['q'][1:]).max())
    assert report['lake_stage_error_m']<1e-12 and report['lake_momentum_max']<1e-12
if args.case=='bedload':
    _,_,other=read_live_frame(case/'restarted')
    report['restart_maximum_difference']={k:float(abs(a-other[k]).max()) for k,a in values.items() if a.size}
    assert max(report['restart_maximum_difference'].values())==0
assert report['maximum_water_residual_m3']<1e-8 and report['maximum_wall_integral']==0
assert report['minimum_depth_m']>=0 and report['maximum_class_residual_kg']<1e-7
(case/'CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
