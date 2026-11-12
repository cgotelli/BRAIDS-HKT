"""Reproducible S5 scientific figures from the packaged solver and saved runs."""
import os,sys,json
from pathlib import Path
from stage_suspended218 import ROOT,OUT,TARGET,REPORT,sha,save
sys.path[:0]=[str(TARGET),str(ROOT)]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.integrate import quad
from braids_core.sediment.portable_river_runtime import install
install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
from braids_core.sediment.river_de_leeuw import DeLeeuw,parameters
from braids_core.sediment.bed_columns_v101 import host
from braids_core.sediment.adaptive_river_worker import read_live_frame

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    d=.0002;speed=np.geomspace(.001,8.,160);h=np.ones_like(speed);z0=h*.001
    arrays={k:host(DeLeeuw(k,4,parameters(d,.0003)).probe(h,speed,z0)) for k in ('cpu','gpu')}
    a,b=arrays.values()
    ut=speed*.41/np.log1p(.36800000071525574*h/z0);sf=ut**2/(9.81*h)
    hsk=(speed*.0009**(1/6)/(8.1*np.sqrt(9.81*sf)))**1.5
    us=np.minimum(ut,np.sqrt(9.81*hsk*sf));z=(us/a[0])**1.5*speed/np.sqrt(9.81*h)
    e=4.74e-4*np.maximum(z-.015,0)**1.18;reference=e/(1+3*e)
    law_error=float(np.max(abs(a[1]-reference)/np.maximum(reference,1e-100)))
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    axes[0].loglog(z,reference,'k-',label='de Leeuw Eq. 26a (direct)')
    axes[0].loglog(z[::6],a[1,::6],'o',mfc='none',label='BRAIDS CPU')
    axes[0].loglog(z[3::6],b[1,3::6],'+',label='BRAIDS GPU')
    axes[0].set(xlabel='(u*skin / ws)^1.5 Fr',ylabel='Reference volume concentration Es',title='Published entrainment relation')
    axes[0].legend(fontsize=8)
    axes[1].plot(speed,a[3],label='Total shear velocity')
    axes[1].plot(speed,a[4],label='Skin shear velocity')
    axes[1].set(xlabel='Depth-mean speed (m/s)',ylabel='Shear velocity (m/s)',title='Explicit bounded Hsk partition');axes[1].legend(fontsize=8)
    axes[2].plot(speed,2650*a[1],label='Reference-level equilibrium')
    axes[2].plot(speed,2650*a[1]*a[2],label='Depth-mean equilibrium')
    axes[2].set(xlabel='Depth-mean speed (m/s)',ylabel='Concentration (kg/m³)',title='They are not the same concentration');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('S5 component verification — not an experimental river comparison')
    fig.savefig(REPORT/'DE_LEEUW_COMPONENTS.png',dpi=170);plt.close(fig)
    save(OUT/'OPERATOR.json',dict(runtime=sha(TARGET/'RUNTIME.json'),law_relative_max=law_error,
        cpu_gpu_max_abs=float(np.max(abs(a-b))),diameter_m=d,source='de Leeuw2020 Eqs7,8,26a; sections3.3,4.2',auxiliaries=parameters(d,.0003)))
    # Compare genuinely different physical bundles at identical hydraulic inputs.
    from braids_core.sediment.river_garcia_parker import GarciaParker,parameters as gp_parameters
    gp=host(GarciaParker('cpu',4,gp_parameters(d)).probe(h,speed,z0))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].loglog(speed,2650*a[1]*a[2],label='de Leeuw 2020 bundle')
    axes[0].loglog(speed,2650*gp[1]*gp[2],label='Garcia-Parker 1991 bundle')
    axes[0].set(xlabel='Depth-mean speed (m/s)',ylabel='Equilibrium depth-mean C (kg/m³)',
                title='Different equations, not backend error')
    i=np.argmin(abs(speed-1.));y=np.linspace(.1,1,200)
    for name,v,alpha,below in (('de Leeuw',a,.1,1.66),('Garcia-Parker',gp,.05,1.)):
        yy=np.r_[np.linspace(0,alpha,50,endpoint=False),np.linspace(alpha,1,150)]
        profile=np.where(yy<alpha,below,(alpha*(1-yy)/((1-alpha)*np.maximum(yy,alpha)))**v[6,i])
        axes[1].plot(2650*v[1,i]*profile,yy,label=name)
    axes[1].set(xlabel='Equilibrium C(z) (kg/m³)',ylabel='z/h',title=f'Profile at U={speed[i]:.3f} m/s')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same h=1 m, z0=1 mm, Ds=0.2 mm; de Leeuw bed D84=0.3 mm')
    fig.savefig(REPORT/'METHOD_COMPARISON.png',dpi=170);plt.close(fig)
    checks=Path(os.environ.get('BRAIDS_S5_RESULTS',str(OUT/'packaged')))
    cases={}
    for p in checks.glob('*/METRICS.json'):
        r=json.loads(p.read_text())
        source=p.parent/'project.json'
        if 'bed' in r and source.exists() and json.loads(source.read_text())['model']['physics']['sediment']['suspension']=='suspension.de_leeuw_2020':
            cases[r['case'],r['bed'],r['backend']]=p.parent
    selected=('amr','cropped','entrainment')
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
    fields=[]
    for row,case in enumerate(selected):
        p=cases[case,'capacity.mpm','cpu'];pg=cases[case,'capacity.mpm','gpu']
        meta,mesh,f=read_live_frame(p/'run');g=read_live_frame(pg/'run')[2]
        polys=[mesh['vertices'][i:j] for i,j in zip(mesh['offsets'][:-1],mesh['offsets'][1:])]
        mean=np.divide(f['suspended'][0],f['q'][0],out=np.zeros_like(f['z']),where=f['q'][0]>1e-8)
        vals=(mean,(f['z']-f['initial_bed_m'])*1000,(g['z']-f['z'])*1000)
        titles=('CPU suspended concentration\n(kg/m³)','CPU bed change\n(mm)','GPU minus CPU bed\n(mm)')
        for col,(value,title) in enumerate(zip(vals,titles)):
            ax=axes[row,col];pc=PolyCollection(polys,array=value,cmap='Blues' if col==0 else 'RdBu_r',edgecolors='#444444',linewidths=.3)
            if col:limit=max(float(np.max(abs(value))),1e-12);pc.set_clim(-limit,limit)
            else:pc.set_clim(0.,max(float(np.max(value)),1e-12))
            ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title,fontsize=10)
            ax.set(xlabel='x (m)',ylabel=f'{case}, t={meta["time_s"]:g} s\ny (m)');fig.colorbar(pc,ax=ax,shrink=.7)
        fields.append(dict(case=case,bed_difference_m=float(np.max(abs(g['z']-f['z']))),
            suspended_difference_kg_m2=float(np.max(abs(g['suspended']-f['suspended'])))))
    fig.suptitle('Short coupled controls — raw cells, no smoothing; separate color scales')
    fig.savefig(REPORT/'COUPLED_CONTROLS.png',dpi=170);plt.close(fig)
    save(OUT/'FIGURE_METRICS.json',fields)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for backend in ('cpu','gpu'):
        for bed in (None,'capacity.mpm'):
            p=cases['amr',bed,backend]
            rows=json.loads((p/'run/DIAGNOSTICS.json').read_text())
            # Diagnostic time key is recorded by the shared worker.
            t=[r.get('time_s',r.get('time',i)) for i,r in enumerate(rows)]
            style='-' if backend=='cpu' else '--';label=backend.upper()+(' + MPM' if bed else ' suspension only')
            axes[0].plot(t,[r['class_residual_kg'][0] for r in rows],style,label=label)
            axes[1].plot(t,[r['water_residual_m3'] for r in rows],style,label=label)
    axes[0].set(title='Sediment budget residual',ylabel='kg',xlabel='Time (s)')
    axes[1].set(title='Water budget residual',ylabel='m³',xlabel='Time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('AMR conservation includes boundary flux and finite bed exchange')
    fig.savefig(REPORT/'AMR_BUDGETS.png',dpi=170);plt.close(fig)
    print('S5 figures generated',flush=True)

if __name__=='__main__':main()
