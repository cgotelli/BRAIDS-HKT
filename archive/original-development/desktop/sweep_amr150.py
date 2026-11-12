"""Geometry-only full wall refinement/coarsening across both user hierarchies."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_amr150'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'candidate'))
from braids_desktop.preparation import decode_tree
from braids_core.sediment.river_leaf_controller import RiverLeafController
from braids_core.sediment.river_amr_transfer import boundary_map

def data(mesh):
    faces=[f for f in mesh.faces if f.neighbour<0]
    names={n:i for i,n in enumerate(sorted({f.bc_id for f in faces if f.kind!='wall'}))}
    return dict(nf=len(faces),neighbour=np.full(len(faces),-1),
        bid=np.array([names[f.bc_id] if f.kind!='wall' else -1 for f in faces]),
        normal=np.array([f.outward_normal for f in faces]).T,
        mid=np.array([.5*(np.array(f.start_xy)+f.end_xy) for f in faces]).T,
        length=np.array([f.length_m for f in faces]))

rows=[]
for name in ('fixed_cpu','breakline_cpu'):
    tree=decode_tree(json.loads(next((OUT/name/'model_layers').rglob('TREE.json')).read_text()))
    meshes=[]
    for level in range(tree.levels):
        leaves=RiverLeafController.pin_boundaries(tree,tree.balanced_leaves(np.full(tree.count,level)))
        meshes.append(data(tree.select(leaves)))
    for a,old in enumerate(meshes):
        for b,new in enumerate(meshes):
            try:boundary_map(old,new)
            except ValueError:
                np.savez(OUT/f'SWEEP_FAILED_{name}_{a}_{b}.npz',**{p+k:v for p,d in [('old_',old),('new_',new)] for k,v in d.items()})
                raise
            rows.append(dict(case=name,from_level=a,to_level=b,source_faces=old['nf'],target_faces=new['nf']))
(OUT/'SWEEP.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',len(rows),'full-boundary partition pairs')
