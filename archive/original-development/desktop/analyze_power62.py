"""Power-focused reanalysis; no relabelling of velocity proxies as measured power."""
import csv,json,sys,os,hashlib
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'validation/hydrokinetic_turbines/reports/power_array62';HOME=Path.home()/'Desktop/BRAIDS_Turbine_Power62'
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def csvout(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def retained():
    src=REPORT.parent/'nearwake60/COMPARISON.csv'
    with src.open() as f:raw=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.3),layout='constrained');metrics={};out=[]
    for res,color in [('D/4','#66889c'),('D/8','#146d9c')]:
        for ref,style in [('experiment_depth_average','-'),('published_2d_model','--')]:
            rows=[r for r in raw if r['resolution']==res and r['reference']==ref and 5<=float(r['x_D'])<=10]
            x,p,t=np.array([[float(r[k]) for k in ('x_D','prediction','target')] for r in rows]).T
            e=100*(p-t);proxy=(1-p)**3/(1-t)**3-1
            metrics[res+'_'+ref]=dict(samples=len(x),RMS_deficit_pp=float(np.sqrt(np.mean(e**2))),maximum_deficit_pp=float(max(abs(e))),
                constant_Cp_centreline_power_proxy_error_percent_range=(100*float(min(proxy)),100*float(max(proxy))))
            axes[1].plot(x,100*proxy,style,color=color,label=res+' vs '+('experiment' if ref.startswith('experiment') else '2D model'))
            if ref.startswith('experiment'):
                axes[0].plot(x,100*p,color=color,label='BRAIDS '+res)
                if res=='D/8':axes[0].plot(x,100*t,'ko-',ms=3,label='Published depth-mean experiment')
            for xx,pp,tt,pr in zip(x,p,t,proxy):out.append(dict(resolution=res,reference=ref,x_D=xx,deficit_braids=pp,deficit_reference=tt,centreline_power_proxy_relative_error=pr))
    axes[0].set(xlabel='Distance downstream (D)',ylabel='Centreline velocity deficit (%)',title='Existing single-rotor wake validation: 5–10D')
    axes[1].set(xlabel='Distance downstream (D)',ylabel='Power proxy error (%)',title='Sensitivity only: constant Cp × centreline U³')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(REPORT/'FAR_WAKE_POWER_SENSITIVITY.png',dpi=170);plt.close(fig)
    save(REPORT/'FAR_WAKE.json',dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),metrics=metrics,
        limitation='The cubic centreline proxy is not disk-integrated power or independent downstream turbine validation.'))
    csvout(REPORT/'FAR_WAKE.csv',out)

def coupled():
    entries=[];histories=[]
    for folder in sorted(HOME.iterdir()):
        f=folder/'run/TURBINE_PERFORMANCE.json'
        if not (folder/'DONE.json').exists() or not f.exists():continue
        rows=json.loads(f.read_text());spec=json.loads((folder/'SPEC.json').read_text())
        for name in sorted(set(r['turbine_id'] for r in rows)):
            selected=[r for r in rows if r['turbine_id']==name];steady=[r for r in selected if r['time_s']>=40]
            time=np.array([r['time_s'] for r in selected]);power=np.array([r['useful_power_W'] for r in selected]);U=np.array([r['upstream_velocity_m_s'] for r in selected])
            assert len(steady)>=3 and time[-1]==60.
            def mean(field):
                ts=np.array([r['time_s'] for r in steady]);v=np.array([r[field] for r in steady]);return float(np.trapezoid(v,ts)/(ts[-1]-ts[0]))
            A=np.pi*.7**2/4;rho=1000
            assert max(abs(r['useful_power_W']-.5*rho*A*r['power_coefficient']*r['upstream_velocity_m_s']**3) for r in selected)<1e-10
            entries.append(dict(case=folder.name,turbine=name,mean_power_W=mean('useful_power_W'),mean_thrust_N=mean('thrust_N'),mean_approach_m_s=mean('upstream_velocity_m_s'),
                temporal_power_range_percent=100*(max(r['useful_power_W'] for r in steady)-min(r['useful_power_W'] for r in steady))/mean('useful_power_W'),
                shaft_energy_sampled_Wh=float(np.trapezoid(power,time)/3600),end_s=float(time[-1]),sampling_s=2.,
                mean_Cp=mean('power_coefficient'),mean_Ct=mean('thrust_coefficient')))
            histories.append((folder.name,name,time,power,U))
    if not entries:return
    csvout(REPORT/'POWER_SUMMARY.csv',entries);save(REPORT/'POWER_SUMMARY.json',entries)
    measured=json.loads((REPORT/'references/DIGITIZATION.json').read_text())['selected_points']
    comparison=[]
    for row in entries:
        if 'rpm4' not in row['case']:continue
        spacing=6 if '6D' in row['case'] else 10
        target=measured[str(spacing if row['turbine']=='Downstream rotor' else 0)]
        scale=.5*1000*(np.pi*.7**2/4)
        ct=row['mean_thrust_N']/(scale*.8**2);cp=row['mean_power_W']/(scale*.8**3)
        comparison.append(dict(case=row['case'],turbine=row['turbine'],spacing_D=spacing,Ct_nominal_braids=ct,Ct_nominal_measured=target['Ct'],
            Cp_nominal_braids=cp,Cp_nominal_measured=target['Cp'],power_W_braids=row['mean_power_W'],power_W_measured=scale*.8**3*target['Cp'],
            power_relative_error_percent=100*(cp/target['Cp']-1),thrust_relative_error_percent=100*(ct/target['Ct']-1)))
    if comparison:
        csvout(REPORT/'EXPERIMENT_COMPARISON.csv',comparison);save(REPORT/'EXPERIMENT_COMPARISON.json',comparison)
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained');down=sorted([r for r in comparison if r['turbine']=='Downstream rotor'],key=lambda r:r['spacing_D'])
        labels=[str(r['spacing_D'])+'D' for r in down];x=np.arange(len(down))
        for ax,field in zip(axes,['Cp','Ct']):
            ax.bar(x-.18,[r[field+'_nominal_measured'] for r in down],.36,label='Measured (Mycek Part II)',color='#444444')
            ax.bar(x+.18,[r[field+'_nominal_braids'] for r in down],.36,label='BRAIDS',color='#237ba8')
            ax.set_xticks(x,labels);ax.set(xlabel='Turbine spacing',ylabel=field+' using nominal flume velocity',title='Downstream rotor '+field);ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle('Fixed RPM: nominal TSR 4, U=0.8 m/s; no fitted loading or mixing')
        fig.savefig(REPORT/'MEASURED_ARRAY_POWER.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10.5,7),layout='constrained')
    for case,name,t,p,u in histories:
        label=case+' / '+name;axes[0].plot(t,p,label=label);axes[1].plot(t,u,label=label)
    axes[0].set(ylabel='Mechanical shaft power (W)',title='Controlled tandem diagnostic — not an experimental array pass')
    axes[1].set(xlabel='Simulation time (s)',ylabel='Corrected local approach velocity (m/s)')
    for ax in axes:ax.axvspan(40,60,color='#aaaaaa',alpha=.1);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,ncol=2);fig.savefig(REPORT/'TANDEM_POWER.png',dpi=160);plt.close(fig)
    print(json.dumps(entries,indent=2))

def vertical_audit():
    sys.path.insert(0,str(ROOT/'desktop/qualification/power_array62/candidate'))
    if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
    from braids_desktop.results import ResultsStore
    from scipy.interpolate import LinearNDInterpolator
    folder=HOME/'single_gpu/run'
    if not (folder.parent/'DONE.json').exists():return
    store=ResultsStore(folder);frame=store.read(-1);xy=np.array([np.array(p.centroid.coords)[0] for p in frame.polygons]);interpolate=LinearNDInterpolator(xy,frame.q[1]/frame.q[0])
    with (ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figure_A12_velocity.csv').open() as f:raw=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.4),layout='constrained');out=[]
    for ax,station in zip(axes,[6.,10.]):
        data=[r for r in raw if float(r['ambient_turbulence_percent'])==3 and float(r['x_over_d'])==station]
        y,u=np.array([[float(r[k]) for k in ('y_over_d','mean_u_over_u_infinity')] for r in data]).T
        order=np.argsort(y);y=y[order];u=u[order]
        r=np.linspace(0,max(abs(y.min()),abs(y.max())),1201)
        symmetric=.5*(np.interp(r,y,u)+np.interp(-r,y,u))
        radius=np.linspace(0,.5,1001);disk=float(8*np.trapezoid(radius*np.interp(radius,r,symmetric),radius))
        zz=np.linspace(-.9/.7,1.1/.7,1601);depthmean=np.array([np.trapezoid(np.interp(np.hypot(yy,zz),r,symmetric,right=1.),zz)/(2/.7) for yy in y])
        prediction=interpolate(5+station*.7+np.zeros_like(y),2+y*.7)/.8
        # This is a planar width/disk projection of depth-mean velocity, not
        # a reconstruction of unknown vertical wake structure.
        yy=np.linspace(-.5,.5,1001);weights=2*np.sqrt(np.maximum(0,.25-yy**2));uh=interpolate(5+station*.7+np.zeros_like(yy),2+yy*.7)/.8
        numerical_disk=float(np.trapezoid(weights*uh,yy)/np.trapezoid(weights,yy))
        out.append(dict(station_D=station,experimental_rotor_mean_axisymmetric=disk,braids_depthmean_projected_rotor_mean=numerical_disk,
            equivalent_constant_Cp_power_ratio_error_percent=100*((numerical_disk/disk)**3-1),
            assumption='Rotational symmetry of measured hub-plane profile; no direct rotor-area measurement. Last BRAIDS frame at 60s.'))
        ax.plot(y,1-u,'ko-',ms=3,label='Measured hub plane')
        ax.plot(y,1-depthmean,'--',color='#ce8d30',label='Assumed full-depth projection')
        ax.plot(y,1-prediction,color='#237ba8',label='BRAIDS depth mean')
        ax.axvspan(-.5,.5,color='#aaaaaa',alpha=.12);ax.set(xlabel='Lateral position / D',ylabel='Axial velocity deficit',title=f'{station:g}D: rotor span shaded');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(REPORT/'VERTICAL_WAKE_AUDIT.png',dpi=170);plt.close(fig);save(REPORT/'VERTICAL_WAKE_AUDIT.json',out)
    print('Vertical audit:',out)
if __name__=='__main__':retained();coupled();vertical_audit()
