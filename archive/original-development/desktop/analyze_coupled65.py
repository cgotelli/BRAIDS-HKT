"""Compare completed coupled trials with independent Mycek tandem loads."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'desktop/qualification/coupled_inflow65/runs';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/coupled_inflow65'
sys.path.insert(0,str(ROOT/'desktop/qualification/coupled_inflow65/candidate'));sys.path.insert(1,str(ROOT))
if os.name=='nt':dlls=[os.add_dll_directory(str(p)) for p in (Path(sys.base_prefix),Path(sys.base_prefix)/'Library/bin') if p.is_dir()]
from braids_desktop.results import ResultsStore
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    measured=read(REPORT.parent/'power_array62/references/DIGITIZATION.json')['selected_points'];scale=.5*1000*np.pi*.7**2/4
    metrics=[];finals={};histories={};fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for folder in sorted(RUNS.iterdir()):
        if not (folder/'DONE.json').exists():continue
        spec=read(folder/'SPEC.json');check=spec['check'];spacing=spec['spacing_D'];backend=spec['backend'];reference=spec['reference']
        data=read(folder/'run/TURBINE_PERFORMANCE.json');down=[r for r in data if r['turbine_id']=='Downstream rotor'];late=[r for r in down if r['time_s']>=75]
        diagnostics=read(folder/'run/DIAGNOSTICS.json');water=max(abs(r['water_residual_m3']) for r in diagnostics)
        row=dict(case=folder.name,spacing_D=spacing,backend=backend,reference=reference,check=check,water_residual_max_m3=water,
            max_induction_residual_m_s=max(abs(r.get('own_induction_residual_m_s',0)) for r in data),
            max_shaft_work_fraction=max(r['useful_power_W']/r['hydraulic_work_W'] for r in data if r['active']),
            end_s=down[-1]['time_s'],exact_restart=read(folder/'DONE.json')['exact_restart'] if check else None)
        assert water<1e-8 and row['max_shaft_work_fraction']<=1+1e-10
        if late:
            t=np.array([r['time_s'] for r in late])
            def avg(k):return float(np.trapezoid([r[k] for r in late],t)/(t[-1]-t[0]))
            P=avg('useful_power_W');T=avg('thrust_N');target=measured[str(spacing)]
            row.update(power_W=P,thrust_N=T,power_error_percent=100*(P/(scale*.8**3*target['Cp'])-1),
                thrust_error_percent=100*(T/(scale*.8**2*target['Ct'])-1),
                power_window_range_percent=100*(max(r['useful_power_W'] for r in late)-min(r['useful_power_W'] for r in late))/P,
                power_change_first_to_last_W=down[-1]['useful_power_W']-down[0]['useful_power_W'])
            label=backend+' '+('previous source' if reference else 'coupled reconstruction')
            col=0 if spacing==6 else 1
            for i,k in enumerate(['useful_power_W','thrust_N']):
                axes[i,col].plot([r['time_s'] for r in down],[r[k] for r in down],label='Previous source' if reference else label,
                    color='#267ca5' if reference else ('#db7626' if backend=='cpu' else '#279857'),
                    linestyle='--' if backend=='cpu' and not reference else '-',linewidth=3 if backend=='cpu' and not reference else 1.5)
        metrics.append(row)
        frame=ResultsStore(folder/'run').read(-1);finals[folder.name]=frame.q
        histories[folder.name]=down
    pairs=[]
    for spacing in [6,10]:
        col=0 if spacing==6 else 1
        for i,(key,unit,expscale) in enumerate([('Cp','Shaft power (W)',scale*.8**3),('Ct','Thrust (N)',scale*.8**2)]):
            axes[i,col].axhline(measured[str(spacing)][key]*expscale,color='black',linestyle='--',label='Mycek measured')
            axes[i,col].set(title=f'{spacing}D',xlabel='Time (s)',ylabel=unit);axes[i,col].grid(alpha=.2)
            axes[i,col].set_ylim(bottom=0)
        for suffix in ('','_check_binary'):
            cpu=f'{spacing}D_cpu_research{suffix}';gpu=f'{spacing}D_gpu_research{suffix}'
            if cpu in finals and gpu in finals:
                err=float(np.max(abs(finals[cpu]-finals[gpu])))
                pair=dict(cpu=cpu,gpu=gpu,max_conserved_field_difference=err)
                pair['max_power_history_difference_W']=max(abs(a['useful_power_W']-b['useful_power_W']) for a,b in zip(histories[cpu],histories[gpu]))
                pairs.append(pair)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Coupled source transition from identical accepted fields at 60 s')
    fig.savefig(REPORT/'COUPLED_LOADS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    import matplotlib.tri as mtri
    field_comparison=[]
    for i,spacing in enumerate([6,10]):
        refname=f'{spacing}D_'+('cpu' if spacing==6 else 'gpu')+'_reference'
        newname=f'{spacing}D_gpu_research'
        if refname not in finals or newname not in finals:continue
        refstore=ResultsStore(RUNS/refname/'run');newstore=ResultsStore(RUNS/newname/'run')
        old=refstore.read(-1);new=newstore.read(-1)
        np.testing.assert_array_equal(refstore.read(0).q,newstore.read(0).q)
        xy=np.array([p.centroid.coords[0] for p in new.polygons]);tri=mtri.Triangulation(xy[:,0],xy[:,1])
        for j,frame in enumerate([old,new]):
            axes[i,j].tricontourf(tri,frame.q[1]/frame.q[0],levels=np.linspace(.4,.9,26),extend='both',cmap='viridis')
            for x in [5,5+spacing*.7]:axes[i,j].plot([x,x],[1.65,2.35],color='red',lw=2)
            axes[i,j].set(xlabel='x (m)',ylabel='y (m)',title=f'{spacing}D '+('previous source' if j==0 else 'coupled reconstruction'),ylim=(0,4),xlim=(3,16))
        field_comparison.append(dict(spacing_D=spacing,initial_fields_exact=True,
            final_depth_max_difference_m=float(np.max(abs(new.q[0]-old.q[0]))),
            final_streamwise_velocity_max_difference_m_s=float(np.max(abs(new.q[1]/new.q[0]-old.q[1]/old.q[0])))))
    fig.suptitle('Depth-averaged streamwise velocity at 80 s (m/s); same 0.4–0.9 colour scale')
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(.4,.9),cmap='viridis'),ax=axes.ravel().tolist(),label='u (m/s)',shrink=.85)
    fig.savefig(REPORT/'COUPLED_FLOW.png',dpi=170);plt.close(fig)
    save(REPORT/'METRICS.json',dict(runs=metrics,paired=pairs,field_comparison=field_comparison,averaging_s=[75,80],limitation='20-second source transition; not statistical equilibrium or general river validation'))
    print(json.dumps(dict(runs=metrics,paired=pairs),indent=2))
if __name__=='__main__':main()
