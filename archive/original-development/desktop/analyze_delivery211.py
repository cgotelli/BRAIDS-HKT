"""Installed flume startup and release comparison figures from native cells."""
import sys,json
from pathlib import Path
from stage_delivery211 import ROOT,OUT,TARGET,sha,save
REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_delivery211'
PROJECT=ROOT.parent/'BRAIDS_JiangCao_MF1';RUNTIME=ROOT/'desktop/runtime'
sys.path[:0]=[str(RUNTIME),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore

def checks(folder):
    s=json.loads((folder/'STATUS.json').read_text());d=json.loads((folder/'DIAGNOSTICS.json').read_text())
    assert s['status']=='complete'
    return dict(status=s['status'],time_s=s['time_s'],steps=s['steps'],rejections=len(s['rejections']),
       wall_seconds=s['wall_seconds'],budgets={k:max(abs(x[k]) for x in d) for k in ('solid_residual_m3','water_plus_pore_residual_m3')})

def compare(a,b):
    np.testing.assert_allclose(a.times,b.times,rtol=0,atol=1e-12)
    result={}
    for i in range(len(a.times)):
        f,g=a.read(i),b.read(i);np.testing.assert_array_equal(f.mesh['leaves'],g.mesh['leaves'])
        for name,x,y in [('depth_mm',f.q[0]*1000,g.q[0]*1000),('bed_mm',f.z*1000,g.z*1000)]:
            e=y-x;v=result.setdefault(name,dict(maximum=0.,rms=0.))
            v['maximum']=max(v['maximum'],float(abs(e).max()))
            v['rms']=max(v['rms'],float(np.sqrt(np.average(e*e,weights=f.mesh['area']))))
    return result

def patch(ax,frame,values,limits,cmap):
    p=PolyCollection([np.asarray(p.exterior.coords) for p in frame.polygons],array=values,cmap=cmap,linewidths=0,rasterized=True)
    p.set_clim(*limits);ax.add_collection(p);ax.autoscale();return p

def main():
    REPORT.mkdir(exist_ok=True,parents=True);records=dict(runtime=sha(RUNTIME/'RUNTIME.json'))
    for scenario in ('front','oblique'):
        a=ResultsStore(OUT/(scenario+'_cpu')/'run');b=ResultsStore(OUT/(scenario+'_gpu')/'run')
        records[scenario]=dict(cpu=checks(a.output),gpu=checks(b.output),difference=compare(a,b))
    store=ResultsStore(PROJECT/'runs/flume211_ready');frames=[store.read(i) for i in range(len(store.times))]
    rows=[]
    for f in frames:
        dz=f.z-f.arrays['initial_bed_m'];area=f.mesh['area'];xy=f.mesh['centres'] if 'centres' in f.mesh else np.array([[p.centroid.x,p.centroid.y] for p in f.polygons])
        wet=f.q[0]>.001
        rows.append(dict(time_s=f.time_s,depth_max_mm=float(f.q[0].max()*1000),min_bed_mm=float(dz.min()*1000),max_bed_mm=float(dz.max()*1000),
          erosion_bulk_m3=float(np.dot(np.maximum(-dz,0),area)),deposition_bulk_m3=float(np.dot(np.maximum(dz,0),area)),
          wet_area_m2=float(area[wet].sum()),wet_front_m=float(xy[wet,0].max()) if wet.any() else 0.,
          concentration_max=float(f.arrays['mobile_bedload_concentration'].max()),
          secondary_max_m_s=float(abs(f.arrays['secondary_intensity_m_s']).max())))
    records['experiment']=dict(checks=checks(store.output),rows=rows,cells=len(frames[-1].z))
    for bc in ('inlet','outlet'):
        records['experiment'][bc]={}
        for quantity in ('water','bedload'):
            series=store.boundary_flow(bc,quantity=quantity)
            records['experiment'][bc][quantity]=dict(minimum=float(series.values.min()),maximum=float(series.values.max()),last=float(series.values[-1]))
    selected=[min(range(len(frames)),key=lambda i:abs(frames[i].time_s-t)) for t in (2,5,10,20)]
    hi=max(abs(f.z-f.arrays['initial_bed_m']).max() for f in frames)*1000
    xmax=min(25,max(r['wet_front_m'] for r in rows)+.5)
    fig,axes=plt.subplots(2,2,figsize=(12,5.8),layout='constrained')
    for ax,index in zip(axes.flat,selected):
        f=frames[index];dz=(f.z-f.arrays['initial_bed_m'])*1000
        p=patch(ax,f,dz,(-hi,hi),'RdBu_r');ax.set(xlim=(0,xmax),ylim=(0,1.5),xlabel='Along-flume x (m)',ylabel='y (m)',title=f'{f.time_s:g} s physical = morphological time')
        ax.set_aspect('equal');ax.axvline(rows[index]['wet_front_m'],color='#168aad',ls='--',lw=1,label='Wet front (> 1 mm)')
        ax.legend(fontsize=7,loc='upper right');fig.colorbar(p,ax=ax,label='Bed change (mm)',shrink=.7)
    fig.suptitle('Jiang-Cao reconstruction, MF1: raw cell bed changes in the wetted reach\nRed = deposition; blue = erosion. No imposed bed perturbation.')
    fig.savefig(REPORT/'BED_EVOLUTION.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained');t=store.times
    ax[0,0].plot(t,[r['min_bed_mm'] for r in rows],label='Minimum bed change')
    ax[0,0].plot(t,[r['max_bed_mm'] for r in rows],label='Maximum bed change')
    ax[0,0].set(ylabel='Bed change (mm)',title='Scour and deposition begin during startup')
    for k,label in [('erosion_bulk_m3','Eroded'),('deposition_bulk_m3','Deposited')]:ax[0,1].plot(t,[1e6*r[k] for r in rows],label=label)
    ax[0,1].set(ylabel='Bulk bed volume (cm³)',title='Integrated bed changes')
    ax[1,0].plot(t,[r['wet_front_m'] for r in rows],label='Depth > 1 mm')
    ax[1,0].axhline(25,ls=':',color='k',label='Outlet');ax[1,0].set(ylabel='Wet-front x (m)',title='Initially dry flume')
    diagnostics=json.loads((store.output/'DIAGNOSTICS.json').read_text())
    for k,label in [('solid_residual_m3','Solid'),('water_plus_pore_residual_m3','Water plus pore')]:ax[1,1].plot([x['time_s'] for x in diagnostics],[x[k] for x in diagnostics],label=label)
    ax[1,1].set(ylabel='Conservation residual (m³)',title='Phase budgets')
    for a in ax.flat:a.set_xlabel('Physical time (s), MF1');a.grid(alpha=.2);a.legend(fontsize=9)
    fig.savefig(REPORT/'EVOLUTION_AND_BUDGETS.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    for scenario,column in [('front',0),('oblique',1)]:
        for backend,style in [('cpu','-'),('gpu','--')]:
            s=ResultsStore(OUT/(scenario+'_'+backend)/'run');ff=[s.read(i) for i in range(len(s.times))]
            ax[0,column].plot(s.times,[1000*(f.z-f.arrays['initial_bed_m']).min() for f in ff],style,label=backend.upper())
            ax[1,column].plot(s.times,[1000*f.q[0].max() for f in ff],style,label=backend.upper())
        ax[0,column].set(title=scenario.capitalize()+' dry-start AMR',ylabel='Minimum bed change (mm)')
        ax[1,column].set(ylabel='Maximum depth (mm)')
    for a in ax.flat:a.set_xlabel('Physical time (s)');a.legend();a.grid(alpha=.2)
    fig.suptitle('Preview36 CPU/GPU release controls')
    fig.savefig(REPORT/'CPU_GPU.png',dpi=170);plt.close(fig)
    save(REPORT/'COMPARISON.json',records);print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':main()
