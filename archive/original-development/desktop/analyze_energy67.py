"""Power-first acceptance evidence; no fitting of tandem measurements."""
import csv,json,os,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/'validation/hydrokinetic_turbines/reports/energy67';RUNS=ROOT/'desktop/qualification/energy67/runs'
sys.path[:0]=[str(ROOT/'desktop/qualification/energy67/candidate'),str(ROOT)]
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in [Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin'] if p.is_dir()]
from braids_desktop.results import ResultsStore
from braids_core.river_turbine_surface import integrate_energy
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    low=read(REPORT.parent/'power_array62/references/DIGITIZATION.json')['selected_points']
    with (REPORT.parent/'coupled_inflow65/MYCEK_TI15_TARGETS.csv').open() as f:high=list(csv.DictReader(f))
    rows=[];raw={};frames={};init={};fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for folder in sorted(RUNS.iterdir()):
        if not (folder/'DONE.json').exists():continue
        spec=read(folder/'SPEC.json');data=read(folder/'run/TURBINE_PERFORMANCE.json');raw[folder.name]=data
        store=ResultsStore(folder/'run');frames[folder.name]=store.read(-1);init[folder.name]=store.read(0)
        energy=integrate_energy(data);row=dict(case=folder.name,backend=spec['backend'],mesh_changed=read(folder/'INITIALIZATION.json')['mesh_changed'],start_s=spec['start_s'],end_s=spec['end_s'],
          water_residual_max_m3=max(abs(r['water_residual_m3']) for r in read(folder/'run/DIAGNOSTICS.json')),
          max_induction_residual_m_s=max(abs(r.get('own_induction_residual_m_s',0)) for r in data),
          max_shaft_work_fraction=max(r['useful_power_W']/r['hydraulic_work_W'] for r in data if r['active']),
          energy_Wh={n:[r['shaft_energy_Wh'] for r in energy if r['turbine_id']==n][-1] for n in {r['turbine_id'] for r in energy}})
        assert row['water_residual_max_m3']<1e-8 and row['max_shaft_work_fraction']<1+1e-10
        if not spec['check'] and spec['case']!='variable':
            down=[r for r in data if r['turbine_id']=='Downstream rotor'];late=[r for r in down if r['time_s']>=spec['end_s']-5];t=[r['time_s'] for r in late]
            P=float(np.trapezoid([r['useful_power_W'] for r in late],t)/(t[-1]-t[0]));ti=spec['ti'];spacing=spec['spacing_D']
            cp=float(low[str(spacing)]['Cp']) if ti==3 else float(np.interp(4,*np.array(sorted((float(r['nominal_TSR']),float(r['coefficient'])) for r in high if r['field']=='Cp' and int(r['spacing_D'])==spacing)).T))
            target=.5*1000*np.pi*.7**2/4*(.8 if ti==3 else .83)**3*cp
            row.update(power_W=P,measured_power_W=target,power_error_percent=100*(P/target-1),power_range_percent=100*np.ptp([r['useful_power_W'] for r in late])/P)
            ax=axes[0 if ti==3 else 1,0 if spacing==6 else 1];ax.plot([r['time_s'] for r in down],[r['useful_power_W'] for r in down],label='Speed/TSR surface');ax.axhline(target,color='black',linestyle='--',label='Independent experiment');ax.set(title=f'TI {ti}% — {spacing}D',xlabel='Time (s)',ylabel='Shaft power (W)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        rows.append(row)
    for ax in axes.flat:ax.set_ylim(0,1.15*max(float(np.max(line.get_ydata())) for line in ax.lines))
    fig.suptitle('Power regression and adverse low-TI control: no tandem-load fitting');fig.savefig(REPORT/'POWER_COMPARISON.png',dpi=160);plt.close(fig)
    pairs=[]
    if 'variable_cpu' in frames and 'variable_gpu' in frames:
        a,b='variable_cpu','variable_gpu';np.testing.assert_array_equal(init[a].q,init[b].q);np.testing.assert_array_equal(init[a].z,init[b].z)
        assert [(r['time_s'],r['turbine_id']) for r in raw[a]]==[(r['time_s'],r['turbine_id']) for r in raw[b]]
        pairs.append(dict(case='variable',initial_fields_identical=True,q_max_difference=float(np.max(abs(frames[a].q-frames[b].q))),power_max_difference_W=max(abs(x['useful_power_W']-y['useful_power_W']) for x,y in zip(raw[a],raw[b]))))
    revisions=[]
    for backend in ['cpu','gpu']:
        a,b=f'variable_{backend}',f'variable_{backend}_v2'
        if b in frames:
            revisions.append(dict(backend=backend,q_max_difference=float(np.max(abs(frames[a].q-frames[b].q))),power_max_difference_W=max(abs(x['useful_power_W']-y['useful_power_W']) for x,y in zip(raw[a],raw[b]))))
    restarts={p.parent.name:read(p)['exact_restart'] for p in RUNS.glob('*_check_v3/DONE.json')}
    sensitivity={}
    if 'variable_cpu' in raw:
        data=raw['variable_cpu'];fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
        for name in ['Upstream rotor','Downstream rotor']:
            r=[x for x in data if x['turbine_id']==name];t=[x['time_s'] for x in r]
            for ax,key in zip([axes[0,0],axes[0,1],axes[1,0],axes[1,1]],['rotor_mean_velocity_m_s','useful_power_W','power_coefficient','tip_speed_ratio']):ax.plot(t,[x[key] for x in r],label=name)
            e=integrate_energy(r);coarse=integrate_energy(r[::2]);axes[2,0].plot(t,[x['shaft_energy_Wh'] for x in e],label=name)
            sensitivity[name]=dict(energy_Wh=e[-1]['shaft_energy_Wh'],sampling_05_vs_1s_percent=100*(coarse[-1]['shaft_energy_Wh']/e[-1]['shaft_energy_Wh']-1),
              power_min_W=min(x['useful_power_W'] for x in r),power_max_W=max(x['useful_power_W'] for x in r),Cp_min=min(x['power_coefficient'] for x in r),Cp_max=max(x['power_coefficient'] for x in r))
        axes[2,1].plot([60,65,70,75,80],[6.64,5.6,5.6,6.64,6.64],color='black')
        for ax,label in zip(axes.flat,['Rotor mean speed (m/s)','Shaft power (W)','Cp','TSR','Accumulated shaft energy (Wh)','Inlet discharge (m³/s)']):ax.set(xlabel='Time (s)',ylabel=label);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle('Variable-discharge functional stress test — not transient experimental validation');fig.savefig(REPORT/'VARIABLE_ENERGY.png',dpi=160);plt.close(fig)
    result=dict(runs=rows,paired=pairs,variable=sensitivity,revision_replays=revisions,final_exact_restarts=restarts)
    (REPORT/'METRICS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
