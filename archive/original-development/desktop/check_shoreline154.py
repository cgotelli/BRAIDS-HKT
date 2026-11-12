"""Straight shoreline and solid-boundary visual checks on immutable fields."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/_shoreline154';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'tests')]
from test_desktop_shoreline153 import mixed_circle,sample
from test_desktop_shoreline154 import solid_case
from braids_desktop.results import ResultsStore
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

fig,axes=plt.subplots(3,2,figsize=(12,14),layout='constrained');timings={}
real=ResultsStore(ROOT/'validation/_mesh152/project_cpu/run').read()
solids,domain,objects=solid_case()
for row,(name,frame) in enumerate((('Mixed 2 m / 1 m cells',mixed_circle()),('Angular and elliptical solids (synthetic display check)',solids),('01obstacle - saved 2 s snapshot',real))):
    before={k:v.copy() for k,v in frame.arrays.items()};start=time.perf_counter()
    rgba,x,y=sample(frame,maximum=700);timings[name]=time.perf_counter()-start
    origin=np.array([0,0]) if row<2 else np.array([2571300.,1107500.])
    raw,smooth=axes[row];values=frame.q[0].copy();values[values<=1e-6]=np.nan
    collection=PolyCollection([np.asarray(p.exterior.coords)-origin for p in frame.polygons],array=values,
        cmap='Blues',clim=(.001,2.),edgecolors='none')
    raw.add_collection(collection)
    dx=x[0,1]-x[0,0];dy=y[0,0]-y[1,0]
    smooth.imshow(rgba,origin='upper',interpolation='nearest',extent=(x.min()-dx/2-origin[0],x.max()+dx/2-origin[0],y.min()-dy/2-origin[1],y.max()+dy/2-origin[1]))
    if row<2:limits=(0,24,0,20)
    else:
        bounds=np.array([frame.polygons[i].bounds for i in np.flatnonzero(frame.q[0]>1e-6)])
        limits=(bounds[:,0].min()-origin[0]-2,bounds[:,2].max()-origin[0]+2,bounds[:,1].min()-origin[1]-2,bounds[:,3].max()-origin[1]+2)
    for ax,label in ((raw,'Original cell colors / wet footprint'),(smooth,'Straight shoreline / linear colors')):
        if row==1:
            for solid in objects:
                pts=np.asarray(solid.exterior.coords);ax.fill(pts[:,0],pts[:,1],color='#777777',edgecolor='#333333',linewidth=.8)
        ax.set_xlim(*limits[:2]);ax.set_ylim(*limits[2:]);ax.set_aspect('equal');ax.set_facecolor('#e8e3d5');ax.set_title(name+'\n'+label)
        ax.set_xlabel('Local x (m)');ax.set_ylabel('Local y (m)')
    for k,value in before.items():np.testing.assert_array_equal(value,frame.arrays[k])
fig.savefig(OUT/'SHORELINE.png',dpi=160);plt.close(fig)
(OUT/'CHECK.json').write_text(json.dumps(dict(render_seconds_700_pixels=timings,source_arrays_unchanged=True,
    method='local-scale straight chords, wall-pinned, bounded linear colors; display only',runtime_unchanged=True),indent=2)+'\n')
print(json.dumps(timings,indent=2))
