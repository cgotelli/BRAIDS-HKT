"""Frozen-state qualification of vertical inflow, not a coupled power release."""
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'desktop/qualification/power_array62/candidate'))
sys.path.insert(1,str(ROOT))
if os.name=='nt':
    dlls=[os.add_dll_directory(str(p)) for p in (Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin') if p.is_dir()]
from braids_desktop.results import ResultsStore
from validation.hydrokinetic_turbines.rotor_inflow64 import reconstruct, disk_quadrature, moments, METHOD, gaussian_deficit, vertical_mean

REPORT=ROOT/'validation/hydrokinetic_turbines/reports/rotor_inflow64'
OLD=REPORT.parent/'power_array62'
HOME=Path.home()/'Desktop/BRAIDS_Turbine_Power62'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    U=.8;D=.7;H=2.;hub=.9
    store=ResultsStore(HOME/'single_gpu/run')
    ids=np.flatnonzero(store.times>=40)
    frames=[store.read(int(i)) for i in ids]
    xy=np.array([p.centroid.coords[0] for p in frames[0].polygons])
    times=store.times[ids]
    vel=np.trapezoid(np.array([f.q[1]/f.q[0] for f in frames]),times,axis=0)/(times[-1]-times[0])
    interpolate=LinearNDInterpolator(xy,vel)
    data=rows(ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figure_A12_velocity.csv')
    table=read(HOME/'tandem_10D_rpm4/run/PROJECT.json')['turbines']['turbines'][0]['control']['performance']
    tsr=np.array([r['tip_speed_ratio'] for r in table]); ct=np.array([r['thrust_coefficient'] for r in table]);cp=np.array([r['power_coefficient'] for r in table])
    measurements=read(OLD/'references/DIGITIZATION.json')['selected_points']
    previous=read(OLD/'EXPERIMENT_COMPARISON.json')
    ct_source=float(np.interp(3.67,tsr,ct))
    train=[r for r in data if float(r['ambient_turbulence_percent'])==3 and float(r['x_over_d']) in (4.,8.)]
    train_x=np.array([float(r['x_over_d']) for r in train]);train_y=np.array([float(r['y_over_d']) for r in train]);train_u=np.array([float(r['mean_u_over_u_infinity']) for r in train])
    # Constrain width at the FIRST evaluated far-wake station (3D), not at
    # the rotor plane, where this far-wake formula need not be real-valued.
    # Exactly two shared parameters, no power fitting.
    lower_width=np.sqrt(ct_source/8)+1e-8
    def residual(p):
        return np.array([1-float(gaussian_deficit(y,0.,x,ct_source,p[0],p[1]-3*p[0])) for x,y in zip(train_x,train_y)])-train_u
    calibrated=least_squares(residual,[.01,lower_width+.01],bounds=([0,lower_width],[.08,.7]))
    growth,width3=calibrated.x;initial_width=width3-3*growth
    save(REPORT/'GAUSSIAN_CALIBRATION.json',dict(source='Jordan et al. 2022 Eq14, DOI 10.1007/s40722-022-00225-2',
         upstream_Ct=ct_source,growth=float(growth),initial_width=float(initial_width),training_stations_D=[4,8],
         validation_stations_D=[3,5,6,7,9,10],training_hub_RMS_pp=float(100*np.sqrt(np.mean(calibrated.fun**2))),
         caveat='BRAIDS-specific measured-profile calibration, not published parameter replication. No power/Ct fit or viscosity change.'))
    def loads(m):
        # Rotor-average TSR, area moments for load/energy. This is a tested
        # research hypothesis, not blade-element aerodynamics.
        lam=4*U/m[0]
        if not tsr[0]<=lam<=tsr[-1]:raise ValueError('Operating point outside supplied curve')
        return dict(U_mean=float(m[0]),U_thrust=float(m[1]**.5),U_power=float(m[2]**(1/3)),TSR=float(lam),
                    Ct_nominal=float(np.interp(lam,tsr,ct)*m[1]/U**2),Cp_nominal=float(np.interp(lam,tsr,cp)*m[2]/U**3))
    fig,axes=plt.subplots(2,4,figsize=(16,7.5),layout='constrained')
    diagnostics=[];power=[];ys,zs,weights=disk_quadrature(D/2)
    stations=sorted({float(r['x_over_d']) for r in data if float(r['ambient_turbulence_percent'])==3 and float(r['x_over_d'])>=3})
    for ax,station in zip(axes.flat,stations):
        subset=[r for r in data if float(r['ambient_turbulence_percent'])==3 and float(r['x_over_d'])==station]
        yu=sorted((float(r['y_over_d']),float(r['mean_u_over_u_infinity'])) for r in subset)
        y,observed=np.array(yu).T
        yf=np.linspace(-1.2,1.2,241)
        ub=interpolate(np.full_like(yf,5+station*D),2+yf)
        # Fit only the saved numerical lateral profile, never measured velocity
        # or downstream power. U_inf is the specified experimental inflow.
        def fitfun(p):return U-p[0]*np.exp(-.5*((yf-p[1])/p[2])**2)
        fit=least_squares(lambda p:fitfun(p)-ub,[.08,0,.3],bounds=([0,-.35,.05],[U,.35,1.2]))
        amp,centre,sigma=fit.x
        mean=interpolate(np.full_like(y,5+station*D),2+y*D)
        deficit=amp*np.exp(-.5*((y*D-centre)/sigma)**2)
        rec=reconstruct(mean,deficit,hub,H,hub,sigma)
        rmean=interpolate(np.full_like(ys,5+station*D),2+ys)
        rdef=amp*np.exp(-.5*((ys-centre)/sigma)**2)
        rotor=reconstruct(rmean,rdef,hub+zs,H,hub,sigma)
        sg=D*(growth*station+initial_width)
        model_hub=U*(1-gaussian_deficit(y,0.,station,ct_source,growth,initial_width))
        model_depth_def=U*gaussian_deficit(y,0.,station,ct_source,growth,initial_width)*vertical_mean(H,hub,sg)
        embedded=reconstruct(mean,model_depth_def,hub,H,hub,sg)
        gaussian_rotor=U*(1-gaussian_deficit(ys/D,zs/D,station,ct_source,growth,initial_width))
        embedded_rotor=reconstruct(rmean,U*gaussian_deficit(ys/D,0.,station,ct_source,growth,initial_width)*vertical_mean(H,hub,sg),hub+zs,H,hub,sg)
        # Reproduce the paper's axisymmetric rotor-area projection, using its
        # measured horizontal profile. This is not a direct vertical measurement.
        rr=np.hypot(ys,zs)/D
        expdisk=U*.5*(np.interp(rr,y,observed)+np.interp(-rr,y,observed))
        diagnostic=dict(station_D=station,sigma_over_D=float(sigma/D),amplitude_depthmean_m_s=float(amp),centre_m=float(centre),
            numerical_fit_RMS_m_s=float(np.sqrt(np.mean((fitfun(fit.x)-ub)**2))),
            raw_hub_RMS_pp=float(100*np.sqrt(np.mean((mean/U-observed)**2))),
            reconstructed_hub_RMS_pp=float(100*np.sqrt(np.mean((rec/U-observed)**2))),
            rotor_min_m_s=float(rotor.min()),raw=loads(moments(rmean,weights)),reconstructed=loads(moments(rotor,weights)),
            measured_profile_proxy=loads(moments(expdisk,weights)),gaussian=loads(moments(gaussian_rotor,weights)),embedded=loads(moments(embedded_rotor,weights)),
            gaussian_hub_RMS_pp=float(100*np.sqrt(np.mean((model_hub/U-observed)**2))),
            embedded_hub_RMS_pp=float(100*np.sqrt(np.mean((embedded/U-observed)**2))),
            gaussian_depthmean_mismatch_RMS_m_s=float(np.sqrt(np.mean((U-model_depth_def-mean)**2))),
            used_for_calibration=station in (4.,8.))
        diagnostics.append(diagnostic)
        ax.plot(y,1-observed,'ko',ms=3,label='Measured hub plane')
        ax.plot(y,1-mean/U,label='BRAIDS depth mean',color='#267ca5')
        ax.plot(y,1-rec/U,label='Reconstructed hub plane',color='#c85127')
        ax.plot(y,1-model_hub/U,'--',label='Calibrated published Gaussian',color='#7956a4')
        ax.plot(y,1-embedded/U,label='Gaussian + BRAIDS mean',color='#279a57')
        ax.set(title=f'{station:g}D',xlabel='y/D',ylabel='Velocity deficit');ax.grid(alpha=.2)
        if station in (6.,10.):
            exp=measurements[str(int(station))]
            original=next(r for r in previous if r['spacing_D']==station and r['turbine']=='Downstream rotor')
            entry=dict(station_D=station,measured=exp,previous_coupled=dict(Cp_nominal=original['Cp_nominal_braids'],Ct_nominal=original['Ct_nominal_braids']),
                       reconstructed=diagnostic['reconstructed'],measured_profile_proxy=diagnostic['measured_profile_proxy'],gaussian=diagnostic['gaussian'],embedded=diagnostic['embedded'])
            for name in ('previous_coupled','reconstructed','measured_profile_proxy','gaussian','embedded'):
                entry[name].update({f'{key}_error_percent':100*(entry[name][key+'_nominal']/exp[key]-1) for key in ('Cp','Ct')})
            power.append(entry)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,fontsize=8)
    fig.suptitle('Rotor-inflow reconstruction: Gaussian calibrated at 4D/8D only; other stations held out')
    fig.savefig(REPORT/'WAKE_PROFILES.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,key in zip(axes,('Cp','Ct')):
        x=np.arange(len(power))
        for j,(name,label,color) in enumerate([('measured','Measured tandem','#333333'),('previous_coupled','Previous coupled BRAIDS','#267ca5'),('reconstructed','Uncalibrated reconstruction','#c85127'),('gaussian','Published Gaussian (calibrated)','#7956a4'),('embedded','Gaussian + BRAIDS mean','#279a57'),('measured_profile_proxy','Measured-profile + isolated curve','#a5a55e')]):
            ax.bar(x+(j-2.5)*.135,[r[name][key if name=='measured' else key+'_nominal'] for r in power],.135,label=label,color=color)
        ax.set_xticks(x,['6D','10D']);ax.set(ylabel=key+' (nominal inflow)',title=key);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Downstream loads: fixed nominal TSR 4; no tandem coefficient fitting')
    fig.savefig(REPORT/'ROTOR_LOADS.png',dpi=170);plt.close(fig)
    save(REPORT/'METRICS.json',dict(method=METHOD,averaging_s=[40,60],profiles=diagnostics,loads=power,
          scope='One-way frozen single-wake inference. Not a new coupled run; no force feedback. Wake measurements TSR3.67, tandem TSR4. Axisymmetry assumption.'))
    paths=[Path(__file__),ROOT/'validation/hydrokinetic_turbines/rotor_inflow64.py',ROOT/'desktop/runtime/RUNTIME.json',
           HOME/'single_gpu/run/frames/INDEX.json',OLD/'references/DIGITIZATION.json',
           ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figure_A12_velocity.csv',HOME/'tandem_10D_rpm4/run/PROJECT.json']
    save(REPORT/'INPUT_IDENTITIES.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(power,indent=2))
if __name__=='__main__':main()
