"""Analytic corner-budget verification and spatial hydraulic settling diagnosis."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from review_settling174 import BASE,load,gate,rms
from braids_desktop.results import ResultsStore
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_corner174'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b,n,xy,area=load();analytic=[]
    for path in sorted((BASE/'probes_v2/gpu').glob('*.npz')):
        a=np.load(path);c=np.load(BASE/'probes_v2/cpu'/path.name)
        m=(xy[:,0]>=7.25)&(xy[:,0]<=8.75)&(xy[:,1]>=2.25)&(xy[:,1]<=4.25)
        assert abs(a['rate'][:,m]-a['exact'][:,m]).max()<1e-13
        assert abs(a['rate'][:,m]-c['rate'][:,m]).max()<1e-13
        analytic.append(dict(name=path.stem,cells=int(m.sum()),
            rate_error_max=float(abs(a['rate'][:,m]-a['exact'][:,m]).max()),
            CPU_GPU_max=float(abs(a['rate'][:,m]-c['rate'][:,m]).max())))
    regions=dict(approach=xy[:,0]<7,groin=(xy[:,0]>=7)&(xy[:,0]<12),wake=xy[:,0]>=12)
    metrics={};indices={}
    for label,data in [('BRAIDS',b),('FLOW',n)]:
        eta=data['h']+data['z'];deta=np.std(eta[-21:],axis=0)
        c=int(deta.argmax());indices[label]=c
        metrics[label]=dict(max_eta_std_xy=xy[c].tolist(),max_eta_std_m=float(deta[c]),
            regions={name:gate(data['t'],data['h'][:,m],data['z'][:,m],data['uv'][:,:,m],300,area[m]) for name,m in regions.items()})
    g=ResultsStore(BASE/'gpu/warmup');c=ResultsStore(BASE/'cpu/warmup')
    dh=[];dv=[]
    for j in range(len(c.times)):
        cf,gf=c.read(j),g.read(j);dh.append(float(abs(cf.q[0]-gf.q[0]).max()))
        dv.append(float(abs(cf.q[1:]/cf.q[0]-gf.q[1:]/gf.q[0]).max()))
    save(OUT/'AUDIT.json',dict(analytic=analytic,settling=metrics,
        CPU_GPU_depth_max_m=max(dh),CPU_GPU_velocity_max_m_s=max(dv),
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (BASE/'SETTLING.json',BASE/'cpu/WARMUP.json',BASE/'gpu/WARMUP.json')},
        new_mobile_comparison_qualified=False))
    verts=[np.asarray(p.exterior.coords) for p in c.read(0).polygons]
    fig,axs=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for row,(label,data) in enumerate([('BRAIDS',b),('FLOW',n)]):
        eta=data['h']+data['z'];std=np.std(eta[-21:],axis=0)*1000
        p=PolyCollection(verts,array=std,cmap='magma',clim=(0,30));axs[row,0].add_collection(p)
        axs[row,0].autoscale();axs[row,0].set_aspect('equal');axs[row,0].set(title=label+' water-surface variability, 280–300 s',xlabel='x (m)',ylabel='y (m)');fig.colorbar(p,ax=axs[row,0],label='Temporal standard deviation (mm)',shrink=.75)
        for name,point in [('Upstream',[4.25,3.25]),('First corner cell',[8.25,3.25]),('Wake',[15.25,3.25])]:
            k=int(np.flatnonzero(np.all(xy==point,axis=1))[0]);axs[row,1].plot(data['t'],eta[:,k],label=name)
        axs[row,1].set(title=label+' fixed-bed water surface',xlabel='Time (s)',ylabel='Elevation (m)');axs[row,1].legend(fontsize=8);axs[row,1].grid(alpha=.2)
    fig.savefig(OUT/'SETTLING.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,6),layout='constrained')
    for row,(label,data) in enumerate([('BRAIDS',b),('FLOW',n)]):
        for ax,t in zip(axs[row],(120,220,300)):
            speed=np.linalg.norm(data['uv'][t],axis=0)
            p=PolyCollection(verts,array=speed,cmap='viridis',clim=(0,2.5));ax.add_collection(p)
            ax.autoscale();ax.set_aspect('equal');ax.set(title=f'{label} {t} s',xlabel='x (m)',ylabel='y (m)')
            every=np.arange(len(xy))%3==0
            ax.quiver(xy[every,0],xy[every,1],data['uv'][t,0,every],data['uv'][t,1,every],scale=35,width=.002)
    fig.colorbar(p,ax=axs.ravel().tolist(),label='Speed (m/s)',shrink=.8)
    fig.suptitle('Fixed-bed flow: no sediment activation before 300 s')
    fig.savefig(OUT/'WAKE_HISTORY.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(analytic=analytic,spatial=metrics,backend_depth_max_m=max(dh)),indent=2))


if __name__=='__main__':main()
