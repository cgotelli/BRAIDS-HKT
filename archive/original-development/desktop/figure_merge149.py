"""Raw local geometry before/after the atomic repair, and backend startup check."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_merge149'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT/'candidate'))
from braids_desktop.preparation import decode_tree
from shapely.geometry import shape,Point
from shapely.affinity import translate
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

failure=json.loads((OUT/'FAILURES.json').read_text());old=failure['small'][0]
tree=decode_tree(json.loads(next((OUT/'user_v2_cpu/model_layers').rglob('TREE.json')).read_text()))
anchor=np.array(failure['origin'])+[54,456]
oldpoly=[shape(old['geometry']),*[shape(r['geometry']) for r in old['neighbors']]]
oldpoly=[translate(p,-54,-456) for p in oldpoly]
newpoly=[translate(shape(dict(type='Polygon',coordinates=[c.vertices_xy])),-anchor[0],-anchor[1]) for c in tree.finest.cells]
newpoly=[p for p in newpoly if p.distance(Point(0,0))<5]
target=next(p for p in newpoly if p.covers(oldpoly[0].centroid))
fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
for ax,polys,title in zip(axes,[oldpoly,newpoly],['Before: binary-only merging stalls','After: valid atomic three-cell union']):
    colors=['#f6d3b3' if p.covers(oldpoly[0].centroid) else '#eef1f6' for p in polys]
    ax.add_collection(PolyCollection([list(p.exterior.coords) for p in polys],facecolors=colors,edgecolors='#555',linewidths=.8))
    p=oldpoly[0] if ax==axes[0] else target
    ax.text(p.centroid.x,p.centroid.y,f'{p.area:.6f} m²',ha='center',va='center',fontsize=10)
    doc=json.loads((OUT/'SOURCE.json').read_text())
    for r in doc['settings']['breaklines']:
        v=np.array(r['points_xy'])-anchor;ax.plot(v[:,0],v[:,1],color='#c43fad',lw=2)
    ax.set_xlim(-5,5);ax.set_ylim(-5,6);ax.set_aspect('equal');ax.set_title(title)
    ax.set_xlabel('x relative to local origin (m)');ax.set_ylabel('y relative to local origin (m)')
fig.suptitle('Original breakline retained; 4 m² minimum and shape checks unchanged')
fig.savefig(OUT/'REPAIR.png',dpi=160);plt.close(fig)
states=[]
for backend in ('cpu','gpu'):
    folder=OUT/('user_v2_'+backend)/'run/frames';index=json.loads((folder/'INDEX.json').read_text())
    with np.load(folder/index['frames'][-1]['file']) as f:states.append({k:f[k].copy() for k in ('q','z')})
result=dict(max_water_state_difference=float(abs(states[0]['q']-states[1]['q']).max()),
            max_bed_difference_m=float(abs(states[0]['z']-states[1]['z']).max()),
            repaired_area_m2=target.area,old_area_m2=oldpoly[0].area)
assert result['max_water_state_difference']<1e-10 and result['max_bed_difference_m']==0
(OUT/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
