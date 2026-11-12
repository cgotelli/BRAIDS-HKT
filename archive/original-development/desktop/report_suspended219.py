"""S6 scientific comparisons, using packaged operators and raw coupled results."""
import os,sys,json
from pathlib import Path
from stage_suspended219 import ROOT,OUT,TARGET,REPORT,sha,save
sys.path[:0]=[str(TARGET),str(ROOT),str(ROOT/'tests')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_core.sediment.portable_river_runtime import install
install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
from braids_core.sediment.river_classic_suspension import ClassicSuspension,parameters,METHODS,VR84,ZF94,CL08
from braids_core.sediment.bed_columns_v101 import host
from braids_core.sediment.adaptive_river_worker import read_live_frame
from test_suspended_classic219 import oracle,SIZES
NAMES={VR84:'Van Rijn 1984',ZF94:'Zyserman–Fredsøe 1994',CL08:'Camenen–Larson 2008 (current)'}

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    speed=np.geomspace(.02,4.,140);h=np.ones_like(speed);z0=h*.001
    records=[];operators={}
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for row,method in enumerate(METHODS):
        a,b=(host(ClassicSuspension(backend,4,parameters(method,SIZES)).probe(h,speed,z0)) for backend in ('cpu','gpu'))
        expected=np.array([oracle(method,1.,u,.001,**SIZES) for u in speed]).T;operators[method]=a
        rel=float(np.max(abs(a[1]-expected[1])/np.maximum(abs(expected[1]),1e-100)))
        records.append(dict(method=method,reference_relative_max=rel,profile_absolute_max=float(np.max(abs(a[2]-expected[2]))),
            cpu_gpu_max_abs=float(np.max(abs(a-b))),parameters=parameters(method,SIZES)))
        ax=axes[row,0];ax.loglog(speed,expected[1],'k-',label='Direct source equation')
        ax.loglog(speed[::8],a[1,::8],'o',mfc='none',label='BRAIDS CPU')
        ax.loglog(speed[4::8],b[1,4::8],'+',label='BRAIDS GPU')
        ax.set(title=NAMES[method],xlabel='Speed (m/s)',ylabel='Reference volume concentration',ylim=(1e-8,1.));ax.legend(fontsize=7)
        ax=axes[row,1];ax.loglog(speed,expected[2],'k-',label='Independent height integration')
        ax.loglog(speed[::8],a[2,::8],'o',mfc='none',label='CPU logit quadrature')
        ax.loglog(speed[4::8],b[2,4::8],'+',label='GPU');ax.set(xlabel='Speed (m/s)',ylabel='Depth-mean / reference concentration',title='Profile conversion');ax.legend(fontsize=7)
        ax=axes[row,2];ax.plot(speed,a[3],label='Total shear velocity');ax.plot(speed,a[4],'--',label='Entrainment shear velocity')
        ax.set(xlabel='Speed (m/s)',ylabel='Shear velocity (m/s)',title='Resolved shear');ax.legend(fontsize=7)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('S6 equation checks — h=1 m, z0=1 mm, Ds=0.1 mm, D50=0.2 mm, D90=0.4 mm')
    fig.savefig(REPORT/'CLASSIC_COMPONENTS.png',dpi=160);plt.close(fig)
    save(OUT/'OPERATOR.json',dict(runtime=sha(TARGET/'RUNTIME.json'),cases=records))

    from braids_core.sediment.river_vr1993 import RiverVR1993,FIELDS,DESKTOP_PARAMETERS
    import validation.sediment.river_morphodynamics as validation_package
    validation_package.__path__.append(str(ROOT/'validation/sediment/river_morphodynamics'))
    from validation.sediment.river_morphodynamics.unified_vr1993_reference import grain
    dstar,critical=grain(SIZES['d50'])
    vals=dict(depth=1.,u=speed,v=0.,observation_height=.36800000071525574,current_z0=.001,apparent_z0=.001,
        d50=SIZES['d50'],mixture_d90=SIZES['d90'],solid_density=2650.,water_density=1000.,gravity=9.81,viscosity=1e-6,
        settling_velocity=operators[VR84][0],dstar=dstar,class_critical_stress=critical,bed_critical_stress=.2,
        surface_fraction=1.,mud_fraction=0.,suspended_diameter=SIZES['ds'],additional_stress=0.,von_karman=.41,
        epsilon=1e-6,maximum_volume_concentration=.65,molecular_schmidt=1.)
    forcing=np.ascontiguousarray([np.broadcast_to(vals[k],speed.shape) for k in FIELDS])
    flow=RiverVR1993('cpu',DESKTOP_PARAMETERS).evaluate(forcing)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for method,a in operators.items():
        axes[0].loglog(speed,2650*a[1]*a[2],label=NAMES[method])
    axes[0].loglog(speed,flow[2],'k--',label='Existing FLOW 1993 producer')
    axes[0].set(xlabel='Speed (m/s)',ylabel='Equilibrium depth-mean C (kg/m³)',title='Same physical inputs, different closures',ylim=(1e-6,1e3))
    axes[0].legend(fontsize=7)
    i=np.argmin(abs(speed-1.));yy=np.linspace(0,1,600)
    for method,a in operators.items():
        p=a[6,i];alpha=a[5,i]
        if method==CL08:profile=np.exp(-p*yy)
        else:
            profile=np.full_like(yy,np.nan);mask=yy>=alpha
            if method==VR84:
                lower=mask&(yy<=.5);upper=yy>.5
                profile[lower]=(alpha*(1-yy[lower])/((1-alpha)*yy[lower]))**p
                profile[upper]=(alpha/(1-alpha))**p*np.exp(-4*p*(yy[upper]-.5))
            else:profile[mask]=(alpha*(1-yy[mask])/((1-alpha)*yy[mask]))**p
        axes[1].semilogx(2650*a[1,i]*profile,yy,label=NAMES[method])
    axes[1].set(xlabel='Equilibrium C(z) (kg/m³)',ylabel='z/h',title=f'Profile shapes at U={speed[i]:.2f} m/s',xlim=(1e-4,200.))
    axes[1].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Physical-model comparison, not an experimental accuracy ranking')
    fig.savefig(REPORT/'METHOD_COMPARISON.png',dpi=170);plt.close(fig)
    if '--operators-only' in sys.argv:
        print('S6 equation figures generated',flush=True);return

    checks=Path(os.environ.get('BRAIDS_S6_RESULTS',str(OUT/'packaged_final_v2')));cases={}
    for p in checks.glob('*/METRICS.json'):
        r=json.loads(p.read_text());source=p.parent/'project.json'
        if not source.exists() or 'bed' not in r:continue
        method=json.loads(source.read_text())['model']['physics']['sediment']['suspension']
        if method in METHODS:cases[method,r['case'],r['bed'],r['backend']]=p.parent
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained');fields=[]
    for row,method in enumerate(METHODS):
        p=cases[method,'cropped','capacity.mpm','cpu'];pg=cases[method,'cropped','capacity.mpm','gpu']
        meta,mesh,f=read_live_frame(p/'run');g=read_live_frame(pg/'run')[2]
        polys=[mesh['vertices'][i:j] for i,j in zip(mesh['offsets'][:-1],mesh['offsets'][1:])]
        mean=np.divide(f['suspended'][0],f['q'][0],out=np.zeros_like(f['z']),where=f['q'][0]>1e-8)
        vals=(mean,(f['z']-f['initial_bed_m'])*1e6,(g['z']-f['z'])*1000)
        titles=('CPU concentration (kg/m³)','CPU bed change (µm)','GPU minus CPU bed (mm)')
        for col,(value,title) in enumerate(zip(vals,titles)):
            ax=axes[row,col];pc=PolyCollection(polys,array=value,cmap='Blues' if col==0 else 'RdBu_r',edgecolors='#444',linewidths=.35)
            if col:limit=max(float(np.max(abs(value))),1e-12);pc.set_clim(-limit,limit)
            else:pc.set_clim(0.,max(float(np.max(value)),1e-12))
            ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(title,fontsize=10)
            ax.set(xlabel='x (m)',ylabel=NAMES[method]+'\ny (m)');fig.colorbar(pc,ax=ax,shrink=.7)
        fields.append(dict(method=method,bed_difference_m=float(np.max(abs(g['z']-f['z']))),
            suspended_difference_kg_m2=float(np.max(abs(g['suspended']-f['suspended'])))))
    fig.suptitle('Short cropped-wall controls at 0.04 s — raw cells, separate scales, no smoothing')
    fig.savefig(REPORT/'COUPLED_CONTROLS.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for method in METHODS:
        for backend in ('cpu','gpu'):
            p=cases[method,'amr','capacity.mpm',backend];rows=json.loads((p/'run/DIAGNOSTICS.json').read_text())
            t=[r.get('time_s',r.get('time',i)) for i,r in enumerate(rows)];style='-' if backend=='cpu' else '--'
            axes[0].plot(t,[r['class_residual_kg'][0] for r in rows],style,label=NAMES[method]+' '+backend.upper())
            axes[1].plot(t,[r['water_residual_m3'] for r in rows],style,label=NAMES[method]+' '+backend.upper())
    axes[0].set(title='Sediment budget residual',ylabel='kg',xlabel='Time (s)')
    axes[1].set(title='Water budget residual',ylabel='m³',xlabel='Time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('AMR budgets include boundary export and finite-bed exchange')
    fig.savefig(REPORT/'AMR_BUDGETS.png',dpi=170);plt.close(fig)
    save(OUT/'FIGURE_METRICS.json',fields)
    print('S6 scientific figures generated',flush=True)

if __name__=='__main__':main()
