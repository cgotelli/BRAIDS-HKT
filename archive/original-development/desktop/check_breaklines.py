"""Short actual CPU/CUDA water and bedload checks on constrained polygons."""
from pathlib import Path
import argparse
import json
import sys
from copy import deepcopy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','gpu'],required=True)
parser.add_argument('--case',choices=['lake','wetfront','bedload'],required=True)
parser.add_argument('--name',default='');args=parser.parse_args()
runtime=ROOT/'validation/_breaklines147/candidate'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
from osgeo import osr
osr.UseExceptions()
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_desktop.demo import create_demo
from braids_desktop import preparation,services
from braids_core.river_model import new_bedload_selection,resolve_model
from braids_core.river_open_boundary import new_state,resolve
from braids_core import river_worker
from braids_core.river_project import worker_profile
from braids_core.sediment.adaptive_river_worker import hierarchy

root=ROOT/'validation/_breaklines147'/((args.name or args.case)+'_'+args.backend)
doc=create_demo(root,backend=args.backend,end_time_s=2.)
doc['settings']['breaklines']=[dict(id='levee outline',closed=True,points_xy=[[2.3,2.4],[16.4,3.1],[15.6,6.3],[2.7,5.7]])]
doc['settings']['output_interval_s']=.25;doc['settings']['dt_max_s']=.05
doc['settings']['adaptation']['interval']=.25
if args.case=='wetfront':
    doc['settings']['adaptive']=True;doc['settings']['adaptation']['terrain_enabled']=False
    doc['settings']['hydraulics']['Upstream']['discharge_m3_s']=3.2
    doc['settings']['levels']=3
    from braids_core.boundaries.domain import PolygonDomain
    doc['settings']['grid']=services.geometry_grid(PolygonDomain(**doc['settings']['domain']),1.,3)
    doc['settings']['adaptation']['buffer_cells']=1
    doc['settings']['breaklines'][0]['points_xy']=[[10.3,2.4],[16.4,3.1],[15.6,6.3],[10.7,5.7]]
else:
    doc['initial']['stage_m']=.5
    if args.case=='lake':
        for name in list(doc['settings']['hydraulics']):doc=services.clear_boundary(doc,name)
    else:
        doc['initial']['velocity_m_s']=[1.,0.]
        doc['settings']['hydraulics']['Upstream']['discharge_m3_s']=3.2
        selection=new_bedload_selection();selection['bed']=dict(composition='bed.layered',transport_layer='transport_layer.constant',coarse_layer=False,sorting='sorting.off')
        request=deepcopy(doc['model']['request']);request['sediment']=dict(enabled=True,selection=selection)
        doc['model']=resolve_model(request).document()
        for name,control in doc['settings']['hydraulics'].items():doc['boundary_states'][name]=new_state(control['method'],sediment=True)
        doc['settings']['sediment_boundaries']['Upstream']['solid_flux_m2_s']=[1e-5]
doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
    {a['bc_id']:a['kind'] for a in doc['settings']['assignments'] if a['kind']!='wall'},
    sediment=doc['model']['physics']['sediment'],ng=1,end_time=2.)
services.save_project_folder(root,doc)
prepared=preparation.prepare(root/'braids_project.json',root/'prepared/current',asset_cache=root/'model_layers/preparation_cache')
tree=hierarchy(worker_profile(doc));print('MESH',len(tree.finest.cells),min(c.area_m2 for c in tree.finest.cells),flush=True)
result=river_worker.run(prepared,root/'run');print('RESULT',result,flush=True)
if args.case=='bedload':
    river_worker.run(prepared,root/'partial',until=1.)
    river_worker.run(prepared,root/'restarted',restart=root/'partial/checkpoint.npz')
report=dict(case=args.case,backend=args.backend,status=result,minimum_area_m2=min(c.area_m2 for c in tree.finest.cells),
    finest_cells=len(tree.finest.cells),mesh_metrics=tree.finest.metrics())
(root/'CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
