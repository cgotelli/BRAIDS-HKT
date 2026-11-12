"""Measured zero-viscosity contrasts and raw-cell figures."""
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
from braids_desktop.results import ResultsStore
from review_mixing174 import pair
from stage_wall172 import save,sha
from probe_zero175 import energy
BASE=ROOT/'validation/_zero175'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_zero175'


def load(path):
    s=ResultsStore(path);fs=[s.read(i) for i in range(len(s.times))]
    return dict(t=s.times,h=np.array([f.q[0] for f in fs]),z=np.array([f.z for f in fs]),
        uv=np.array([f.q[1:]/f.q[0] for f in fs]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b,n,xy,area,s,fs,_=pair(ROOT/'validation/_corner174')
    sets={'BRAIDS baseline':b,'FLOW 2DH':n}
    if (BASE/'native_cyclic/execution.json').exists():
        _,cyclic,*_=pair(ROOT/'validation/_corner174',native_folder=BASE/'native_cyclic')
        sets['FLOW Cyclic']=cyclic
    for label,path in [('BRAIDS half CFL','half_cfl'),('Hydraulic minmod','minmod'),
                       ('Constant face surface','surface_constant'),('Consistent wall shear','tangent'),
                       ('Free-stage inlet','free_inlet')]:
        p=BASE/path/'cpu/run'
        if (BASE/path/'cpu/CHECK.json').exists():sets[label]=load(p)
    metrics={}
    for name,d in sets.items():
        eta=d['h']+d['z'];j=d['t']>=280
        std=np.std(eta[j],axis=0)
        r=dict(late_eta_std_rms_mm=float(np.sqrt(np.mean(std**2))*1000),
            late_eta_std_max_mm=float(std.max()*1000),max_std_xy=xy[std.argmax()].tolist())
        for key,mask in [('upstream',xy[:,0]<7),('groin',(xy[:,0]>=7)&(xy[:,0]<11)),
                         ('wake',(xy[:,0]>=11)&(xy[:,0]<30)),('outlet',xy[:,0]>=30)]:
            r[key]=dict(eta_std_rms_mm=float(np.sqrt(np.mean(std[mask]**2))*1000),
                velocity_std_rms_m_s=float(np.sqrt(np.var(d['uv'][j][:,:,mask],axis=0).mean())))
        r['history']=[dict(time_s=float(t),eta_std_rms_mm=float(np.sqrt(np.var(eta[max(0,k-20):k+1],axis=0).mean())*1000))
                      for k,t in enumerate(d['t'])]
        metrics[name]=r
    probes={}
    for spatial in ('baseline','minmod'):
        probes[spatial]={p.stem:energy(np.load(p))[0] for p in sorted((BASE/'probes_v2'/spatial/'gpu').glob('*.npz'))}
    parity={}
    for spatial in ('baseline','minmod'):
        parity[spatial]={}
        for p in sorted((BASE/'probes_v2'/spatial/'cpu').glob('*.npz')):
            c=np.load(p);g=np.load(BASE/'probes_v2'/spatial/'gpu'/p.name)
            parity[spatial][p.stem]={k:float(abs(c[k]-g[k]).max()) for k in ('rate','water')}
    from scipy.integrate import solve_ivp
    straight={}
    for backend in ('cpu','gpu'):
        d=load(BASE/'straight_v2'/backend/'run')
        straight[backend]=dict(late_eta_std_rms_mm=float(np.sqrt(np.var(d['h'][-21:],axis=0).mean())*1000))
        if backend=='cpu':cpu=d
        else:straight['backend']={k:float(abs(cpu[k]-d[k]).max()) for k in ('h','uv','z')}
    ss=ResultsStore(BASE/'straight_v2/cpu/run');ff=ss.read(-1);sxy=np.array([p.centroid.coords[0] for p in ff.polygons])
    exact=solve_ivp(lambda x,h:(.001-(.5**2)/(45**2*h**3))/(1-.5**2/(9.81*h**3)),
                    (40,0),[.58-(.1-.001*40)],rtol=1e-11,atol=1e-13,dense_output=True)
    true_h=exact.sol(sxy[:,0])[0]
    straight['backwater_depth_error_mm']=dict(maximum=float(abs(ff.q[0]-true_h).max()*1000),
        rms=float(np.sqrt(np.mean((ff.q[0]-true_h)**2))*1000))
    budgets={}
    for p in sorted(BASE.glob('*/*/run/DIAGNOSTICS.json')):
        rows=json.loads(p.read_text())
        budgets[str(p.parent.parent.parent.name)+'_'+p.parent.parent.name]=dict(
            max_water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
            max_solid_residual_kg=max(abs(v) for r in rows for v in r['class_residual_kg']),
            max_wall_integral=max(abs(v) for r in rows for v in r['wall_integrals']))
    save(OUT/'DIAGNOSTICS.json',dict(metrics=metrics,energy=probes,operator_backend=parity,
        straight=straight,budgets=budgets,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json')))
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for name,d in sets.items():
        if name in ('Hydraulic minmod','Constant face surface','Consistent wall shear','Free-stage inlet'):continue
        r=metrics[name]['history'];axes[0,0].plot([v['time_s'] for v in r],[v['eta_std_rms_mm'] for v in r],label=name)
        for ax,point in zip(axes.flat[1:],[(4.25,3.25),(8.25,3.25),(10.25,3.25),(20.25,3.25),(35.25,3.25)]):
            c=int(np.flatnonzero(np.all(xy==point,axis=1))[0])
            ax.plot(d['t'],d['h'][:,c]+d['z'][:,c],label=name,lw=1)
            ax.set(title=f'Water surface at {point} m',ylabel='Elevation (m)',xlabel='Time (s)')
    axes[0,0].set(title='Trailing 20 s water-surface variability',ylabel='RMS temporal fluctuation (mm)',xlabel='Time (s)')
    for ax in axes.flat:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'ZERO_VISCOSITY_HISTORY.png',dpi=150);plt.close(fig)
    verts=[np.array(p.exterior.coords) for p in fs[0].polygons]
    maps={k:sets[k] for k in ('BRAIDS baseline','FLOW 2DH','FLOW Cyclic') if k in sets}
    fig,axes=plt.subplots(len(maps),2,figsize=(14,3*len(maps)),layout='constrained')
    for row,(name,d) in enumerate(maps.items()):
        std=np.std((d['h']+d['z'])[d['t']>=280],axis=0)*1000
        for ax,val,title,cmap,limits in zip(axes[row],(std,d['uv'][-1,1]),
            ('Water-surface variability 280–300 s','Transverse velocity at 300 s'),('magma','RdBu_r'),((0,30),(-1,1))):
            pc=PolyCollection(verts,array=val,cmap=cmap,clim=limits,edgecolors='none');ax.add_collection(pc)
            ax.set(xlim=(0,40),ylim=(0,8),aspect='equal',title=name+': '+title,xlabel='x (m)',ylabel='y (m)')
            fig.colorbar(pc,ax=ax,shrink=.7)
    fig.savefig(OUT/'ZERO_VISCOSITY_MAPS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for name,d in sets.items():
        r=metrics[name]['history'];axes[0].plot([v['time_s'] for v in r],[v['eta_std_rms_mm'] for v in r],label=name)
        sel=xy[:,0]==8.25
        axes[1].plot(xy[sel,1],d['uv'][-21:,0,sel].mean(0),'.-',label=name)
    axes[0].set(xlim=(70,300),ylim=(0,25),xlabel='Time (s)',ylabel='20 s fluctuation RMS (mm)',title='All isolated controls')
    axes[1].set(xlabel='y (m)',ylabel='Mean u (m/s)',ylim=(0,None),title='Wall-side jet at x=8.25 m, mean 280–300 s')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(OUT/'NUMERICAL_CONTROLS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    mask=sxy[:,1]==4.25;order=np.flatnonzero(mask);order=order[np.argsort(sxy[order,0])]
    axes[0].plot(sxy[order,0],ff.q[0,order]+ff.z[order],label='BRAIDS CPU/GPU')
    axes[0].plot(sxy[order,0],true_h[order]+ff.z[order],'--',label='Steady backwater ODE')
    axes[0].set(xlabel='x (m)',ylabel='Water elevation (m)',title='Straight channel, zero viscosity')
    for label,d in [('Straight channel',cpu),('Groin',b)]:
        std=[np.sqrt(np.var(d['h'][max(0,j-20):j+1],axis=0).mean())*1000 for j in range(len(d['t']))]
        axes[1].semilogy(d['t'][1:],std[1:],label=label)
    axes[1].set(xlabel='Time (s)',ylabel='20 s fluctuation RMS (mm)',title='Same inlet, outlet and resistance')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'STRAIGHT_CHANNEL_CONTROL.png',dpi=160);plt.close(fig)
    print(json.dumps({k:{f:v for f,v in r.items() if f!='history'} for k,r in metrics.items()},indent=2))


if __name__=='__main__':main()
