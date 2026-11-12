"""Independent held-out loads and wake comparison for the TI15 coupled run."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'validation/hydrokinetic_turbines/reports/performance66';RUNS=ROOT/'desktop/qualification/performance66/runs'
sys.path[:0]=[str(ROOT/'desktop/qualification/performance66/candidate'),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_desktop.results import ResultsStore
from validation.hydrokinetic_turbines.rotor_inflow64 import gaussian_deficit,vertical_mean
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    audit=read(REPORT/'AUDIT.json');k=audit['growth'];eps=audit['initial_width'];U=.83;D=.7
    with (REPORT.parent/'coupled_inflow65/MYCEK_TI15_TARGETS.csv').open() as f:targets=list(csv.DictReader(f))
    def target(spacing,field):
        x,y=np.array(sorted((float(r['nominal_TSR']),float(r['coefficient'])) for r in targets if int(r['spacing_D'])==spacing and r['field']==field)).T
        return float(np.interp(4,x,y))
    rows=[];fields={};initials={};histories={};fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for folder in sorted(RUNS.iterdir()):
        if not (folder/'DONE.json').exists():continue
        spec=read(folder/'SPEC.json');spacing=spec['spacing_D'];data=read(folder/'run/TURBINE_PERFORMANCE.json')
        selected=[r for r in data if r['turbine_id']==('Upstream rotor' if spacing==0 else 'Downstream rotor')]
        late=[r for r in selected if r['time_s']>=50];t=np.array([r['time_s'] for r in late])
        def avg(key):return float(np.trapezoid([r[key] for r in late],t)/(t[-1]-t[0]))
        water=max(abs(r['water_residual_m3']) for r in read(folder/'run/DIAGNOSTICS.json'))
        row=dict(case=folder.name,spacing_D=spacing,backend=spec['backend'],nu_scale=spec['nu_scale'],power_W=avg('useful_power_W'),thrust_N=avg('thrust_N'),
          power_range_percent=100*np.ptp([r['useful_power_W'] for r in late])/avg('useful_power_W'),rotor_speed_m_s=avg('rotor_mean_velocity_m_s') if spacing else avg('upstream_velocity_m_s'),
          max_water_residual_m3=water,max_shaft_sourcework_fraction=max(r['useful_power_W']/r['hydraulic_work_W'] for r in data if r['active']))
        earlier=[r for r in selected if 40<=r['time_s']<=50]
        earlier_mean=float(np.trapezoid([r['useful_power_W'] for r in earlier],[r['time_s'] for r in earlier])/10)
        row['power_40_50_to_50_60_change_percent']=100*(row['power_W']/earlier_mean-1)
        assert water<1e-8 and row['max_shaft_sourcework_fraction']<=1+1e-10
        store=ResultsStore(folder/'run');frame=store.read(-1);fields[folder.name]=frame.q;histories[folder.name]=selected;initials[folder.name]=store.read(0)
        if spacing:
            for i,(field,key,scale) in enumerate([('Cp','useful_power_W',.5*1000*np.pi*D**2/4*U**3),('Ct','thrust_N',.5*1000*np.pi*D**2/4*U**2)]):
                value=avg(key);measured=target(spacing,field)*scale;row[field+'_error_percent']=100*(value/measured-1);row[field+'_measured']=target(spacing,field)
                ax=axes[i,0 if spacing==6 else 1];ax.plot([r['time_s'] for r in selected],[r[key] for r in selected],label=spec['backend'].upper(),linestyle='--' if spec['backend']=='cpu' else '-',linewidth=2.7 if spec['backend']=='cpu' else 1.4)
                ax.axhline(measured,color='black',linestyle='--');ax.set(xlabel='Time (s)',ylabel='Power (W)' if i==0 else 'Thrust (N)',title=f'{spacing}D');ax.grid(alpha=.2);ax.legend(fontsize=7)
        rows.append(row)
    fig.suptitle('TI15% coupled runs; dashed line = independent Mycek tandem measurement')
    fig.savefig(REPORT/'COUPLED_LOADS.png',dpi=160);plt.close(fig)
    pairs=[]
    for spacing in [0,6,10]:
        a=f'{spacing}D_cpu_nu1';b=f'{spacing}D_gpu_nu1'
        if a in fields and b in fields:
            np.testing.assert_array_equal(initials[a].q,initials[b].q);np.testing.assert_array_equal(initials[a].z,initials[b].z)
            assert [r['time_s'] for r in histories[a]]==[r['time_s'] for r in histories[b]]
            pairs.append(dict(spacing_D=spacing,initial_q_z_identical=True,max_q_difference=float(np.max(abs(fields[a]-fields[b]))),max_power_difference_W=max(abs(x['useful_power_W']-y['useful_power_W']) for x,y in zip(histories[a],histories[b]))))
    profiles=[]
    single=RUNS/'0D_cpu_nu1/run'
    if (single.parent/'DONE.json').exists():
        store=ResultsStore(single);frames=[store.read(i) for i,t in enumerate(store.times) if t>=50]
        xy=np.array([p.centroid.coords[0] for p in frames[0].polygons]);v=np.mean([f.q[1]/f.q[0] for f in frames],axis=0);interp=LinearNDInterpolator(xy,v)
        with (ROOT/'validation/hydrokinetic_turbines/measurements/mycek_2014_figure_A12_velocity.csv').open() as f:meas=list(csv.DictReader(f))
        ct=np.mean([r['thrust_coefficient'] for r in read(single/'TURBINE_PERFORMANCE.json') if r['time_s']>=50])
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for ax,x in zip(axes.flat,[4,6,8,10]):
            y,u=np.array(sorted((float(r['y_over_d']),float(r['mean_u_over_u_infinity'])) for r in meas if float(r['ambient_turbulence_percent'])==15 and float(r['x_over_d'])==x)).T
            mean=interp(np.full(len(y),5+x*D),2+D*y)/U
            hub=mean+gaussian_deficit(y,0,x,ct,k,eps)*(vertical_mean(2,.9,D*(k*x+eps))-1)
            row=dict(station_D=x,held_out=x not in [4,8],raw_depthmean_as_hub_rms_pp=float(100*np.sqrt(np.mean((mean-u)**2))),reconstructed_hub_rms_pp=float(100*np.sqrt(np.mean((hub-u)**2))),max_hub_error_pp=float(100*np.max(abs(hub-u))))
            profiles.append(row);ax.plot(y,u,'k.',label='Measured hub plane');ax.plot(y,mean,'--',label='BRAIDS depth mean');ax.plot(y,hub,label='BRAIDS reconstructed hub')
            ax.set(xlabel='y/D',ylabel='u/U',title=f'{x}D '+('held out' if row['held_out'] else 'shape fit station'));ax.set_ylim(.65,1.08);ax.grid(alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3);fig.suptitle('Single-turbine cold run, mean 50–60 s; no tandem power fitting');fig.savefig(REPORT/'COUPLED_WAKE.png',dpi=160);plt.close(fig)
    out=dict(runs=rows,paired=pairs,profiles=profiles,averaging_s=[50,60],scope='Mean-flow high-TI surrogate; no resolved TI15 fluctuations. Finite 60s cold run, not 100s experimental statistics.')
    (REPORT/'METRICS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
