"""Raw-cell plots and budgets from the short MPM groin run; no smoothing."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'validation/_bedload157'
out=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_bedload157';out.mkdir(parents=True,exist_ok=True)

def load(name):
    folder=base/name/'run/frames';index=json.loads((folder/'INDEX.json').read_text());frames=[]
    for row in index['frames']:
        path=folder/row['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with np.load(path) as f:frames.append({k:f[k] for k in f.files})
    with np.load(folder/index['frames'][0]['epoch']) as f:
        polygons=[f['vertices'][a:b].copy() for a,b in zip(f['offsets'][:-1],f['offsets'][1:])];area=f['area'].copy()
    return frames,polygons,area

cpu,polys,area=load('final_cpu');gpu,_,_=load('final_gpu');control,cp,ca=load('control_cpu')
from shapely.geometry import Polygon
xy=np.array([Polygon(p).centroid.coords[0] for p in polys]);times=np.array([float(f['time']) for f in cpu]);bed=np.array([f['z']-cpu[0]['z'] for f in cpu])
summary={}
for name,frames in [('cpu',cpu),('gpu',gpu)]:
    b=np.array([f['z']-frames[0]['z'] for f in frames]);t=np.array([float(f['time']) for f in frames])
    warm=t<=40.+1e-9
    assert np.max(np.abs(b[warm]))==0.
    assert all(np.max(np.abs(f['bed_boundary']))==0 for f in np.array(frames,dtype=object)[warm])
    last=b[-1];pos=int(np.argmax(last));neg=int(np.argmin(last));wake=(xy[:,0]>=10)&(xy[:,0]<=16)&(xy[:,1]<3)
    summary[name]=dict(max_deposition_mm=float(last[pos]*1000),max_erosion_mm=float(last[neg]*1000),
        deposition_xy_m=xy[pos].tolist(),erosion_xy_m=xy[neg].tolist(),
        deposited_bulk_m3=float(np.maximum(last,0)@area),eroded_bulk_m3=float(-np.minimum(last,0)@area),
        wake_max_deposition_mm=float(last[wake].max()*1000),warmup_bed_change_max_m=float(np.abs(b[warm]).max()),
        warmup_boundary_ledger_max_kg=0.,
        final_water_state_change_last_5s_max_m=float(np.max(np.abs(frames[-1]['q'][0]-frames[-2]['q'][0]))))
summary['backend_max_abs']={k:float(np.max(np.abs(cpu[-1][k]-gpu[-1][k]))) for k in ('q','z','mass','bed_boundary')}
assert summary['backend_max_abs']['z']<1e-4 and summary['backend_max_abs']['q']<1e-4
limit=float(np.max(np.abs(bed))*1000);norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
def panel(ax,polygons,values,title,*,norm=norm,cmap='RdBu_r'):
    artist=PolyCollection(polygons,array=values,cmap=cmap,norm=norm,edgecolors='none');ax.add_collection(artist)
    ax.set(xlim=(0,20),ylim=(0,8),aspect='equal',title=title,xlabel='Distance along channel (m)',ylabel='Across channel (m)')
    return artist
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ax,target in zip(axes.flat,(40,60,90,120)):
    i=int(np.argmin(abs(times-target)));art=panel(ax,polys,1000*bed[i],f't = {times[i]:g} s · {max(0,times[i]-40):g} s of bedload')
    ax.fill([8,8,10,10],[0,3,3,0],color='#4c5260');ax.text(9,1.5,'Groin',color='white',rotation=90,ha='center',va='center')
fig.colorbar(art,ax=axes.ravel().tolist(),label='Bed change (mm): blue erosion / red deposition',shrink=.85)
fig.suptitle('MPM bedload around a bank-attached groin · raw 0.5 m cells · CPU\nQ = 4 m³/s, d = 1 mm, 40 s fixed-bed warm-up, MF = 1')
fig.savefig(out/'BED_EVOLUTION.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
for ax,frames,pg,title in [(axes[0],cpu,polys,'BRAIDS CPU · groin'),(axes[1],gpu,polys,'BRAIDS GPU · groin'),(axes[2],control,cp,'Control · same channel without groin')]:
    art=panel(ax,pg,1000*(frames[-1]['z']-frames[0]['z']),title)
fig.colorbar(art,ax=axes.tolist(),label='Bed change after 80 s of MPM bedload (mm)',shrink=.8)
fig.savefig(out/'CPU_GPU_CONTROL.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained');h=cpu[-1]['q'][0];uv=cpu[-1]['q'][1:]/np.maximum(h,1e-12)
art=panel(axes[0],polys,h,'Water depth at 120 s',norm=None,cmap='Blues');fig.colorbar(art,ax=axes[0],label='Depth (m)',shrink=.7)
art=panel(axes[1],polys,np.linalg.norm(uv,axis=0),'Velocity and impermeable groin',norm=None,cmap='Reds')
sel=(np.arange(len(xy))%3==0)&(h>.001);axes[1].quiver(xy[sel,0],xy[sel,1],uv[0,sel],uv[1,sel],scale=25,width=.003)
fig.colorbar(art,ax=axes[1],label='Speed (m/s)',shrink=.7);fig.savefig(out/'HYDRAULICS.png',dpi=180);plt.close(fig)
summary['control']=dict(max_deposition_mm=float(1000*np.max(control[-1]['z']-control[0]['z'])),max_erosion_mm=float(1000*np.min(control[-1]['z']-control[0]['z'])))
adaptive_cpu,_,_=load('adaptive_cpu2');adaptive_gpu,_,_=load('adaptive_gpu2')
summary['adaptive_backend_max_abs']={k:float(np.max(np.abs(adaptive_cpu[-1][k]-adaptive_gpu[-1][k]))) for k in ('q','z','mass','bed_boundary')}
assert summary['adaptive_backend_max_abs']['z']<1e-4
for name,frames in [('cpu',adaptive_cpu),('gpu',adaptive_gpu)]:
    assert all(np.max(np.abs(f['bed_boundary']))==0. for f in frames if float(f['time'])<=2.)
    events=json.loads((base/('adaptive_'+name+'2')/'run/ADAPTATION.json').read_text())
    summary['adaptive_'+name]=[dict(time_s=e['time_s'],from_cells=e['from_cells'],to_cells=e['to_cells']) for e in events if e['changed']]
    assert len(summary['adaptive_'+name])>=3
(out/'COMPARISON.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
