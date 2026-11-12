"""Geometry scaling evidence and figures, without a global fine-grid build."""
import json,time
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from braids_core import river_turbines as t
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_core.boundaries.breaklines import build_breakline_hierarchy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'validation/hydrokinetic_turbines/reports/local_mesh63'
v=t.default();v['turbines']=[t.definition([80.,40.],[1.,.2])]
d=PolygonDomain('large',[[0,0],[400,0],[400,80],[0,80]],'EPSG:2056');g=NativeGrid(200,40,0,0,2,2)
metrics={}
for mode in ['off','local']:
 start=time.perf_counter();tree=build_breakline_hierarchy(d,[],g,[] if mode=='off' else t.constraints(v),levels=1,local_refinement=[] if mode=='off' else t.mesh_regions(v))
 metrics[mode]=dict(cells=len(tree.finest.cells),geometry_seconds=time.perf_counter()-start,area=sum(c.area_m2 for c in tree.finest.cells))
metrics['uniform_025m_cell_count_analytic']=400*80/.25**2
metrics['local_vs_uniform_cell_reduction_percent']=100*(1-metrics['local']['cells']/metrics['uniform_025m_cell_count_analytic'])
(OUT/'GEOMETRY.json').write_text(json.dumps(metrics,indent=2)+'\n')
fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
polys=[np.asarray(c.vertices_xy) for c in tree.finest.cells];sizes=[np.sqrt(c.nominal_area_m2) for c in tree.finest.cells]
for ax in axes:
 artist=PolyCollection(polys,array=np.array(sizes),cmap='viridis',norm=LogNorm(.25,2),edgecolors='#333333',linewidths=.12)
 ax.add_collection(artist);ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)');line=np.array(t.rotor_line(v['turbines'][0]));ax.plot(*line.T,color='red',lw=2)
 wake=np.array(t.wake_footprint(v['turbines'][0]).exterior.coords);ax.plot(*wake.T,'--',color='red',lw=1)
axes[0].set(xlim=(0,400),ylim=(0,80),title=f"2 m river mesh; {len(polys):,} cells with local turbine refinement")
axes[1].set(xlim=(70,100),ylim=(30,52),title='0.7 m rotor: local 0.25 m cells; dashed outline is its 10D wake')
fig.colorbar(artist,ax=axes,label='Nominal local square size (m)',ticks=[.25,.5,1,2]);fig.savefig(OUT/'LOCAL_MESH.png',dpi=170)
print(json.dumps(metrics,indent=2))
