"""Raw-cell bare-base groin figures, paired fields and conservative budgets."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_desktop.results import ResultsStore
BASE=ROOT/'validation/_sediment158';OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_sediment158'
OUT.mkdir(parents=True,exist_ok=True)
def store(name,run='first'):return ResultsStore(BASE/name/'project/runs'/run)
cpu=store('cpu_final');gpu=store('gpu_final');cpu240=store('cpu_final','resumed');gpu240=store('gpu_final','resumed')
low=store('cpu_half_cfl');a=cpu.read(-1);b=gpu.read(-1);c=cpu240.read(-1);d=gpu240.read(-1)
def stats(s):
    f=s.read(-1);bed=s.values(f,'bed_change_m');area=f.mesh['area'];xy=np.array([p.centroid.coords[0] for p in f.polygons]);wake=(xy[:,0]>=10)&(xy[:,0]<=16)&(xy[:,1]<3)
    ledger=f.arrays['bed_boundary'];mass=float(np.sum(f.arrays['mass']*area));rho=s.project['settings']['columns']['solid_density_kg_m3'][0]
    # Bed boundary is signed integrated solid volume, not mass.
    supplied=float(np.maximum(ledger,0).sum()*rho);exported=float(-np.minimum(ledger,0).sum()*rho)
    return dict(time_s=f.time_s,maximum_deposition_mm=float(bed.max()*1000),minimum_bed_change_mm=float(bed.min()*1000),
        peak_xy_m=xy[int(bed.argmax())].tolist(),wake_maximum_deposition_mm=float(bed[wake].max()*1000),
        bed_inventory_kg=mass,inlet_supply_kg=supplied,outlet_export_kg=exported,
        minimum_stock_kg_m2=float(f.arrays['mass'].min()),budget_residual_kg=mass-supplied+exported)
summary={name:stats(s) for name,s in [('cpu120',cpu),('gpu120',gpu),('cpu240',cpu240),('gpu240',gpu240)]}
for label,x,y in [('backend120',a,b),('backend240',c,d),('cfl120',a,low.read(-1))]:
    summary[label]={key:float(np.max(np.abs(x.arrays[key]-y.arrays[key]))) for key in ('q','z','mass','bed_boundary')}
    summary[label]['depth_m']=float(np.max(np.abs(x.q[0]-y.q[0])))
    assert summary[label]['z']<1e-4 and summary[label]['q']<.01,summary[label]
for v in ('cpu120','gpu120','cpu240','gpu240'):
    assert summary[v]['minimum_bed_change_mm']>=-1e-10 and abs(summary[v]['budget_residual_kg'])<1e-7
for method in ('total','ratio'):
    x=store(method+'_cpu').read(-1);y=store(method+'_gpu').read(-1)
    supplied=float(np.maximum(x.arrays['bed_boundary'],0).sum())
    assert abs(supplied-.0004)<1e-14 # 4 s * 1e-4 m3/s = 4 s * Q(4) * qs/q(2.5e-5)
    summary[method]=dict(supplied_solid_m3=supplied,backend_bed_m=float(np.max(abs(x.z-y.z))))
for label,cn,gn in [('manning_amr_mf10','zones_cpu_final','zones_gpu'),('colebrook_mf10','colebrook_cpu','colebrook_gpu')]:
    x=store(cn).read(-1);y=store(gn).read(-1)
    summary[label]={k:float(np.max(abs(x.arrays[k]-y.arrays[k]))) for k in ('q','z','mass','bed_boundary')}
    assert summary[label]['z']<1e-7
    summary[label]['note']='Short coupled numerical check; not morphological acceleration accuracy validation.'
limit=1000*cpu240.values(c,'bed_change_m').max()
def panel(ax,s,f,title,field='bed_change_m',scale=1000,cmap='YlOrBr',vmax=limit):
    polys=[np.asarray(p.exterior.coords) for p in f.polygons]
    art=PolyCollection(polys,array=s.values(f,field)*scale,cmap=cmap,clim=(0,vmax),edgecolors='none');ax.add_collection(art)
    ax.fill([8,8,10,10],[0,3,3,0],color='#414b55');ax.text(9,1.4,'Groin',color='white',rotation=90,ha='center')
    ax.set(xlim=(0,20),ylim=(0,8),aspect='equal',xlabel='Along channel (m)',ylabel='Across channel (m)',title=title);return art
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for ax,s,t in zip(axes.flat,(cpu,cpu,cpu,cpu240),(40,80,120,240)):
    f=s.read(int(np.argmin(abs(s.times-t))));art=panel(ax,s,f,f'{t} s: {max(0,t-40)} s of MPM transport')
fig.colorbar(art,ax=axes.ravel().tolist(),label='Deposition above initially bare fixed base (mm)',shrink=.8)
fig.suptitle('Capacity-fed MPM on a non-erodible base: no initial sediment stock\n0.5 m mesh, Q = 4 m³/s, d = 1 mm, MF = 1, CFL target = 1')
fig.savefig(OUT/'BARE_BED_EVOLUTION.png',dpi=175);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
for ax,s,f,title in [(axes[0],cpu240,c,'CPU — 240 s'),(axes[1],gpu240,d,'GPU — 240 s')]:art=panel(ax,s,f,title)
fig.colorbar(art,ax=axes.tolist(),label='Deposition (mm)',shrink=.8);fig.savefig(OUT/'CPU_GPU.png',dpi=175);plt.close(fig)
frames=[cpu.read(i) for i in range(len(cpu.times))]+[cpu240.read(i) for i in range(1,len(cpu240.times))]
times=[f.time_s for f in frames];rho=2650.
stock=[float(np.sum(f.arrays['mass']*f.mesh['area'])) for f in frames]
feed=[float(np.maximum(f.arrays['bed_boundary'],0).sum()*rho) for f in frames]
export=[float(-np.minimum(f.arrays['bed_boundary'],0).sum()*rho) for f in frames]
fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.plot(times,feed,label='Cumulative solid supply');ax.plot(times,export,label='Cumulative export');ax.plot(times,stock,label='Sediment retained in bed')
ax.axvline(40,color='gray',linestyle=':',label='Bedload activated');ax.axvline(120,color='black',linestyle='--',label='Selected checkpoint')
ax.set(xlabel='Physical time (s)',ylabel='Solid sediment mass (kg)',ylim=(0,None));ax.legend();ax.grid(alpha=.2)
fig.savefig(OUT/'SEDIMENT_BUDGET.png',dpi=175);plt.close(fig)
(OUT/'COMPARISON.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
