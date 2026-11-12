"""Inspect the failed geometry guard without changing or executing the mesh."""
import json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'desktop/qualification/workflow61/candidate'
sys.path.insert(0,str(TARGET));sys.path.insert(1,str(ROOT))
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_core.river_project import load_project,worker_profile
from braids_core.sediment.adaptive_river_worker import hierarchy
from braids_core.boundaries.cut_cells import CutCellMesh
folder=next((TARGET.parent).glob('cpu_mapped_*/target/input.json'))
doc=load_project(folder);original=CutCellMesh.metrics;seen=[]
def metrics(self):
    m=original(self);seen.append((self,m));return m
CutCellMesh.metrics=metrics
try:hierarchy(worker_profile(doc))
except ValueError as e:error=str(e)
else:raise AssertionError('Original guard should reproduce')
mesh,metrics=seen[-1];closure=np.zeros((len(mesh.cells),2))
for f in mesh.faces:
    v=f.length_m*np.array(f.outward_normal);closure[f.owner]+=v
    if f.neighbour>=0:closure[f.neighbour]-=v
bad=np.flatnonzero(np.max(abs(closure),axis=1)>1e-11)
result=dict(error=error,metrics=metrics,origin=[mesh.grid.xlower_m,mesh.grid.ylower_m],
    coordinate_ulp_m=float(np.spacing(max(abs(mesh.grid.xlower_m),abs(mesh.grid.ylower_m)))),
    bad_cells=[dict(cell=int(i),closure=closure[i].tolist(),vertices=mesh.cells[i].vertices_xy,
        faces=[dict(length=f.length_m,normal=f.outward_normal,start=f.start_xy,end=f.end_xy,role=f.boundary_role) for f in mesh.faces if i in (f.owner,f.neighbour)]) for i in bad])
path=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61/GEOMETRY_AUDIT.json';path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['error','metrics','coordinate_ulp_m']},indent=2))
