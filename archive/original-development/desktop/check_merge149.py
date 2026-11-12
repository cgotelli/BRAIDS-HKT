"""Qualify the user's unchanged geometry via an isolated authenticated runtime."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_merge149'
runtime=OUT/'candidate';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(runtime))
from braids_core.river_launcher import verify
verify()
from braids_core.sediment.portable_river_runtime import install
install(json.loads((runtime/'PRECOMPILED.json').read_text()),runtime)
from braids_desktop import services,preparation
from braids_core import river_worker
from braids_core.river_model import resolve_model
from braids_core.river_project import worker_profile
from braids_core.sediment.adaptive_river_worker import hierarchy
from shapely.geometry import LineString,Polygon
from shapely.ops import unary_union
from shapely.affinity import translate
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['cpu','gpu'],required=True);parser.add_argument('--name',default='user_v2');args=parser.parse_args()
source=Path('C:/Users/STREEM/Desktop/0101ttt/River_data/braids_project.json')
raw=source.read_bytes();doc=json.loads(raw);doc['terrain']['dem_path']=str(source.parent/doc['terrain']['dem_path'])
doc['model']['request']['execution']['backend']=args.backend
doc['model']=resolve_model(doc['model']['request']).document()
case=OUT/(args.name+'_'+args.backend)
if case.exists():raise FileExistsError(case)
services.save_project_folder(case,doc)
stamp=time.perf_counter();prepared=preparation.prepare(case/'braids_project.json',case/'prepared/current',asset_cache=case/'model_layers/preparation_cache')
seconds=time.perf_counter()-stamp
tree=hierarchy(worker_profile(doc));mesh=tree.finest
area=sum(c.area_m2 for c in mesh.cells);expected=Polygon(doc['settings']['domain']['vertices_xy']).area
for row in doc['settings']['breaklines']:
    wanted=LineString(row['points_xy']).intersection(Polygon(doc['settings']['domain']['vertices_xy']))
    got=unary_union([LineString([f.start_xy,f.end_xy]) for f in mesh.faces if row['id'] in f.breakline_ids])
    # GEOS nanometre-width buffers can collapse at LV95 coordinate magnitudes.
    # Compare in the same local frame used by construction; do not loosen line
    # accuracy or infer that a zero-area global buffer means a missing edge.
    x,y=row['points_xy'][0];wanted=translate(wanted,-x,-y);got=translate(got,-x,-y)
    assert wanted.difference(got.buffer(1e-8)).length<1e-7
    assert got.difference(wanted.buffer(1e-8)).length<1e-7
assert min(c.area_m2 for c in mesh.cells)>=4*(1-1e-11)
assert abs(area-expected)<1e-6
print('PREPARED',args.backend,len(mesh.cells),seconds,flush=True)
stamp=time.perf_counter();result=river_worker.run(prepared,case/'run',until=.02)
rows=json.loads((case/'run/DIAGNOSTICS.json').read_text())
report=dict(source=str(source),source_sha256=hashlib.sha256(raw).hexdigest(),backend=args.backend,
            prepared_seconds=seconds,run_seconds=time.perf_counter()-stamp,run=result,
            finest_cells=len(mesh.cells),minimum_area_m2=min(c.area_m2 for c in mesh.cells),
            domain_area_m2=expected,mesh_area_m2=area,mesh_metrics=mesh.metrics(),
            water_budget_m3=max(abs(r['water_residual_m3']) for r in rows))
(case/'CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
assert source.read_bytes()==raw,'Source project changed during the check'
