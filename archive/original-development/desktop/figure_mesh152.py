"""Unsmoothed mesh and CPU/GPU output evidence for the mesh152 qualification."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_mesh152'
sys.path[:0]=[str(OUT/'candidate'),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from braids_desktop.results import ResultsStore
from braids_desktop.preparation import decode_tree
comparison={}
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for i,case in enumerate(('wetfront','bedload')):
    states=[]
    for j,backend in enumerate(('cpu','gpu')):
        store=ResultsStore(OUT/(case+'_'+backend)/'run');frame=store.read();first=store.read(0);states.append(frame)
        values=frame.q[0] if case=='wetfront' else 1000*(frame.z-first.z)
        ax=axes[i,j];collection=PolyCollection([np.asarray(p.exterior.coords) for p in frame.polygons],
            array=values,cmap='Blues' if case=='wetfront' else 'RdBu_r',edgecolors='#555555',linewidths=.3)
        if j==0:bound=max(float(abs(values).max()),1e-8)
        collection.set_clim(0,.8) if case=='wetfront' else collection.set_clim(-bound,bound)
        ax.add_collection(collection)
        for solid in store.project['settings']['solids']:
            p=np.asarray(solid['points_xy']);ax.fill(p[:,0],p[:,1],color='#8c939a',zorder=3)
        ax.set_xlim(0,20);ax.set_ylim(0,8);ax.set_aspect('equal');ax.set_title(f'{backend.upper()} - {case} - {frame.time_s:g} s')
        fig.colorbar(collection,ax=ax,label='Depth (m)' if case=='wetfront' else 'Bed change (mm)',shrink=.8)
    a,b=states
    np.testing.assert_array_equal(a.mesh['leaves'],b.mesh['leaves'])
    comparison[case]={k:float(abs(a.arrays[k]-b.arrays[k]).max()) for k in ('q','z','mass','water','bed_boundary')}
    comparison[case]['depth_m']=float(abs(a.q[0]-b.q[0]).max())
    comparison[case]['strict_roundoff_screen_passed']=comparison[case]['q']<1e-9 and comparison[case]['z']<1e-10
    # The user accepts sub-mm coupled differences, not bitwise trajectory
    # identity. Keep the failed roundoff-only screen visible; require depth
    # and bed within 0.1 mm and momentum within 0.001 m²/s, plus separate
    # conservation and exact within-backend restart checks.
    assert comparison[case]['depth_m']<1e-4 and comparison[case]['z']<1e-4 and comparison[case]['q']<1e-3
fig.savefig(OUT/'COMPARISON.png',dpi=160);plt.close(fig)
a=ResultsStore(OUT/'project_cpu/run').read();b=ResultsStore(OUT/'project_gpu/run').read()
np.testing.assert_array_equal(a.mesh['leaves'],b.mesh['leaves'])
comparison['project']={k:float(abs(a.arrays[k]-b.arrays[k]).max()) for k in ('q','z','mass','water','bed_boundary')}
assert comparison['project']['q']<1e-9
path=next((OUT/'project_cpu/model_layers/preparation_cache/geometry').glob('*/TREE.json'))
tree=decode_tree(json.loads(path.read_text()));mesh=tree.select(a.mesh['leaves'])
solid=[f for f in mesh.faces if f.boundary_role=='solid_obstacle']
assert all(mesh.cells[f.owner].level==tree.finest_level for f in solid)
assert min(c.area_m2 for c in mesh.cells)>=1-1e-9
comparison['geometry']=dict(active_cells=len(mesh.cells),finest_cells=len(tree.finest.cells),minimum_area_m2=min(c.area_m2 for c in mesh.cells),
    obstacle_faces=len(solid),obstacle_owner_levels=sorted(set(mesh.cells[f.owner].level for f in solid)),finest_level=tree.finest_level)
fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
origin=np.array([2571300.,1107300.])
for ax,limits,title in zip(axes,[(48,70,3,35),(-40,15,90,140)],['Finest cells around the obstacle','Merged cells along the outer wall']):
    collection=PolyCollection([np.asarray(p.exterior.coords)-origin for p in a.polygons],array=np.array([c.area_m2 for c in mesh.cells]),
        cmap='viridis',norm=LogNorm(vmin=1,vmax=256),edgecolors='#202020',linewidths=.5)
    ax.add_collection(collection)
    for row in ResultsStore(OUT/'project_cpu/run').project['settings']['solids']:
        p=np.asarray(row['points_xy'])-origin;ax.fill(p[:,0],p[:,1],color='#bbbbbb',zorder=3)
    ax.set_xlim(*limits[:2]);ax.set_ylim(*limits[2:]);ax.set_aspect('equal');ax.set_title(title)
    ax.set_xlabel('Easting - 2571300 (m)');ax.set_ylabel('Northing - 1107300 (m)')
    fig.colorbar(collection,ax=ax,label='Actual cell area (m²)',shrink=.8)
fig.suptitle('01obstacle: 1 m minimum size, every active fluid cell >= 1 m²')
fig.savefig(OUT/'PROJECT_MESH.png',dpi=170);plt.close(fig)
(OUT/'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n');print(json.dumps(comparison,indent=2))
