"""Display/mapping evidence from real Desktop projects and saved cell fields."""
import os,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/'desktop/qualification/workflow61/candidate';sys.path.insert(0,str(RUNTIME));sys.path.insert(1,str(ROOT))
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from braids_core import river_state_snapshot as snapshot,river_turbines as turbines
from braids_core.sediment import adaptive_river_worker as core
REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_workflow61'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    ui=read(REPORT/'UI_CHECK.json');output=Path(ui['turbine_run']);context=read(output/'RUN_CONTEXT.json');prepared=(output/context['input']).resolve()
    if not prepared.is_dir():prepared=(output.parent.parent/context['input']).resolve()
    # Run contexts use paths relative to the output directory.
    if not prepared.is_dir():raise ValueError(prepared)
    tree=snapshot._tree(prepared);project=read(prepared/'PROJECT.json');profile=read(prepared/'core/prepared/profile.json')
    with np.load(prepared/'core/prepared/initial.npz') as a:initial={k:a[k].copy() for k in a.files}
    bed=initial['rock']+initial['thickness'].sum(0);cases=[]
    fig,axes=plt.subplots(2,2,figsize=(12,6.6),layout='constrained')
    for ax,length in zip(axes.flat,[0.,5.,10.,20.]):
        doc=json.loads(json.dumps(project['turbines']));rotor=doc['turbines'][0];rotor['wake_refinement']=dict(enabled=length>0,length_diameters=max(length,10.) if length==0 else length)
        mapping=turbines.prepare(doc,tree,bed);leaves=core.initial_leaves(profile,initial,tree,tuple(mapping['required_leaves']));mesh=tree.select(leaves)
        finest=tree.offsets[-2];values=np.array([1. if i>=finest else 0. for i in leaves])
        collection=PolyCollection([c.vertices_xy for c in mesh.cells],array=values,cmap='Blues',clim=(0,1),edgecolors='#566675',linewidths=.15);ax.add_collection(collection)
        ax.plot([8.,8.],[3.5,4.5],color='#c93226',lw=3);ax.set(xlim=(0,32),ylim=(0,8),aspect='equal',title=f'{"Wake off" if length==0 else str(int(length))+"D wake"}: {len(leaves):,} active cells',xlabel='x (m)',ylabel='y (m)')
        cases.append(dict(length_D=length,cells=len(leaves),finest_wake_cells=mapping['rotors'][0]['wake_refinement']['finest_cells']))
    fig.suptitle('Automatic wake corridor: same turbine, smallest cell size 0.25 m, three levels');fig.savefig(REPORT/'WAKE_REFINEMENT.png',dpi=160);plt.close(fig)
    store=ResultsStore(output);line=np.array([[9.,1.],[9.,7.]])
    raw=store.spatial_profile(line,'velocity_m_s',-1);smooth=store.spatial_profile(line,'velocity_m_s',-1,interpolate=True)
    fig,ax=plt.subplots(figsize=(8.5,4.4),layout='constrained');ax.plot(raw.distance_m,raw.values,color='#8c8c8c',lw=1.6,label='Saved cell values (raw steps)')
    ax.plot(smooth.distance_m,smooth.values,'o-',color='#1c78a6',lw=1.8,ms=3,label='Desktop linear profile')
    ax.set(xlabel='Distance along section (m)',ylabel='Speed (m/s)',title='Same saved turbine wake at 8 s; display interpolation only');ax.grid(alpha=.2);ax.legend();fig.savefig(REPORT/'PROFILE_COMPARISON.png',dpi=160);plt.close(fig)
    source=next(Path(ui['source_checkpoint']).parent.parent.glob('01_Hydraulic_state/Project_data/runs/*/frames/INDEX.json'))
    a=ResultsStore(source.parent.parent).read(-1);b=ResultsStore(Path(ui['mapped_run'])).read(0)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');vmin=min(a.q[0].min(),b.q[0].min());vmax=max(a.q[0].max(),b.q[0].max())
    for ax,f,label in zip(axes,[a,b],['Source hydraulic mesh','Explicitly mapped bedload mesh']):
        c=PolyCollection([np.array(p.exterior.coords) for p in f.polygons],array=f.q[0],cmap='Blues',clim=(vmin,vmax),edgecolors='#555555',linewidths=.3);ax.add_collection(c);ax.autoscale();ax.set_aspect('equal');ax.set(title=f'{label}: {len(f.polygons)} cells, t = {f.time_s:g} s',xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(c,ax=axes,label='Water depth (m)',shrink=.8);fig.savefig(REPORT/'STATE_MAPPING.png',dpi=160);plt.close(fig)
    (REPORT/'FIGURE_CHECKS.json').write_text(json.dumps(dict(status='passed',wake_cases=cases,profile_sampling=smooth.sampling,profile_saved_min=float(np.nanmin(raw.values)),profile_saved_max=float(np.nanmax(raw.values)),profile_display_min=float(np.nanmin(smooth.values)),profile_display_max=float(np.nanmax(smooth.values))),indent=2)+'\n')
    print('Saved wake, profile and mapping figures',flush=True)
if __name__=='__main__':main()
