"""Figures and explicit partial/full qualification receipts for the candidate."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from . import run_case as r
from . import experiment as e
from . import analyze_runs as a


def intensity(d):
    rows=np.loadtxt(d['folder']/'secondary_flow.txt')
    rows=rows[(rows[:,0]==d['t'][-1])&(rows[:,2]==0)]
    result=np.zeros(len(d['area']));result[rows[:,1].astype(int)]=rows[:,6]
    return result


def evidence(d):
    I=intensity(d);bend=(d['s']>=e.APPROACH)&(d['s']<=e.BEND_END)
    budgets=np.loadtxt(d['folder']/'secondary_flow_budget.txt')
    momentum=np.loadtxt(d['folder']/'secondary_momentum_budget.txt')
    return dict(metrics=a.audit(d),
        bend_mean_intensity_m_s=float(np.average(I[bend],weights=d['area'][bend])),
        bend_rms_intensity_m_s=float(np.sqrt(np.average(I[bend]**2,weights=d['area'][bend]))),
        max_abs_intensity_m_s=float(np.max(abs(I))),
        secondary_inventory_budget_max_residual=float(np.max(abs(budgets[:,-1]))),
        momentum_budget_max_residual=float(np.max(abs(momentum[:,-2:]))),
        integrated_secondary_boundary_impulse=momentum[-1,7:9].tolist(),
        frames_sha256=r.sha256(r.shared.native_frames_path(d['folder'])),
        execution_sha256=r.sha256(d['folder']/'execution.json'),
        input_sha256=r.sha256(d['folder']/'input_provenance.json'))


def profiles(data,labels,target):
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    n=np.linspace(.04,e.WIDTH-.04,100)
    for j,angle in enumerate((45,90,135)):
        s=np.full_like(n,e.APPROACH+e.RADIUS*np.deg2rad(angle))
        for d,label in zip(data,labels):
            for row,values in enumerate((d['speed'][-1],intensity(d))):
                y,_=a.sample(d,values,s,n)
                axes[row,j].plot(n,y,label=label)
        axes[0,j].set(title=f'{angle} degree section',ylabel='Speed (m/s)')
        axes[1,j].set(xlabel='Outer to inner bank (m)',ylabel='Signed secondary I (m/s)')
        for ax in axes[:,j]:ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.5)
    speed_top=max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:ax.set_ylim(0,speed_top)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Raw model profiles; secondary feedback retained, experimental accuracy not yet qualified')
    fig.savefig(target,dpi=160);plt.close(fig)


def seeded():
    report=r.HERE/'reports/fine_regularized_v1'
    tags=['fine_hydro_controls_v1_baseline','fine_hydro_controls_v1_no_momentum_feedback','fine_regularized_v1_t8']
    labels=['Original feedback','Feedback off (diagnostic)','Regularized feedback (candidate)']
    data=[a.load(r.HERE/'runs'/tag) for tag in tags]
    fig=plt.figure(figsize=(12,7.5),layout='constrained');grid=fig.add_gridspec(2,3)
    top=fig.add_subplot(grid[0,:]);axes=[]
    for col,(d,label) in enumerate(zip(data,labels)):
        delta=np.diff(d['h']+d['z'],axis=0)*1000
        top.semilogy(d['t'][1:],np.max(abs(delta),axis=1),'o-',label=label)
        ax=fig.add_subplot(grid[1,col]);axes.append(ax)
        im=PolyCollection(d['polygons'],array=delta[-1],cmap='RdBu_r',clim=(-11,11),edgecolors='none')
        ax.add_collection(im);ax.autoscale();ax.set_aspect('equal')
        ax.set(title=label,xlabel='x (m)',ylabel='y (m)')
    top.axhline(.01,color='gray',ls=':',label='Steady gate: 0.01 mm')
    top.set(xlabel='Seconds after identical saved hydraulic initialization',ylabel='Max stage change per 10 s (mm)')
    top.grid(alpha=.2);top.legend(fontsize=9)
    fig.colorbar(im,ax=axes,label='Raw stage change, 50 to 60 s (mm)',shrink=.8)
    fig.suptitle('Regularized secondary-flow forcing: short nonlinear decay screen\nSame fine mesh, viscosity, boundary conditions and physical initial state')
    fig.savefig(report/'decay_comparison.png',dpi=160);plt.close(fig)
    profiles(data,labels,report/'secondary_profiles.png')
    records=[dict(label=label,**evidence(d)) for d,label in zip(data,labels)]
    old,new=records[0]['metrics'],records[-1]['metrics']
    improvement=old['final_stage_change_over_output_interval_m']/new['final_stage_change_over_output_interval_m']
    thread_tag='fine_regularized_threads_v1'
    f=[r.shared.read_frames(r.shared.native_frames_path(r.HERE/'runs'/f'{thread_tag}_t{n}'))[-1] for n in (1,8)]
    thread=dict(end_s=1.,max_state_difference=float(np.max(abs(f[0]['q']-f[1]['q']))),
                max_ledger_difference=float(np.max(abs(f[0]['ledger']-f[1]['ledger']))))
    result=dict(cases=records,reduction_factor=improvement,thread_check=thread,
                steady_hydraulics_qualified=False,experimental_validation_passed=False,
                production_promoted=False,delft_source_modified=False)
    r.write_json(report/'comparison.json',result)
    r.write(report/'RESULTS.md',f'''# Regularized secondary-flow candidate: short test

The last 10-second maximum stage change drops from
{old['final_stage_change_over_output_interval_m']*1000:.4f} mm to
{new['final_stage_change_over_output_interval_m']*1000:.4f} mm ({improvement:.1f} times smaller).
This is decay, **not yet the 0.01 mm steady gate**. No sediment run is released by this result.

Momentum feedback remains beta=1, viscosity remains 0.001 m2/s, and
secondary intensity remains nonzero in the bend. The signed mean bend
intensity is {records[-1]['bend_mean_intensity_m_s']:.6f} m/s;
the water budget residual is {new['water_budget_residual_m3']:.3e} m3.
The 1-second 1/8-thread comparison has maximum state difference
{thread['max_state_difference']:.3e} and ledger difference {thread['max_ledger_difference']:.3e}.

Figures: [decay and maps](decay_comparison.png), [velocity/intensity profiles](secondary_profiles.png).
Receipts: [comparison.json](comparison.json), [build.json](build.json).
Formulation, limitations and promotion gates: [REGULARIZATION.md](../../REGULARIZATION.md).

Next: cold-start 300-second hydraulics on both resolutions, then conservative
mobile-bed validation and experimental-profile comparison. No experimental
accuracy claim or general AMR qualification is made. The plugin/production
runtime is unchanged; Delft3D remains untouched.
''')
    print(json.dumps(dict(reduction_factor=improvement,threads=thread,candidate=records[-1]),indent=2))


def cold(tag):
    report=r.HERE/'reports'/tag
    data=[a.load(r.HERE/'runs'/f'{tag}_{name}_t{threads}') for name,threads in [('coarse',4),('fine',8)]]
    labels=['Regularized coarse: 0.2125 m','Regularized fine: 0.10625 m']
    records=[]
    for d,label in zip(data,labels):
        record=evidence(d);m=record['metrics']
        gates=dict(steady_stage=m['final_stage_change_over_output_interval_m']<1e-5,
            inlet=m['inlet_discharge_relative_error']<.005,outlet=m['outlet_discharge_relative_error']<.005,
            positive_depth=m['min_depth_m']>0,water_budget=abs(m['water_budget_residual_m3'])<1e-6,
            secondary_budget=record['secondary_inventory_budget_max_residual']<1e-8,
            momentum_budget=record['momentum_budget_max_residual']<1e-8,
            secondary_active=record['bend_rms_intensity_m_s']>1e-4)
        records.append(dict(label=label,gates=gates,**record))
    profiles(data,labels,report/'cold_secondary_profiles.png')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for d,label in zip(data,labels):
        ax.semilogy(d['t'][1:],np.max(abs(np.diff(d['h']+d['z'],axis=0)),axis=1)*1000,'o-',label=label)
    ax.axhline(.01,color='gray',ls=':',label='Steady gate')
    ax.set(xlabel='Time since cold start (s)',ylabel='Max stage change per 50 s (mm)',title='300-second cold-start hydraulic qualification')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(report/'cold_decay.png',dpi=160);plt.close(fig)
    passed=all(all(row['gates'].values()) for row in records)
    result=dict(cases=records,hydraulic_gates_passed=passed,production_promoted=False,
                sediment_run_started=False,experimental_validation_passed=False,delft_source_modified=False)
    r.write_json(report/'qualification.json',result)
    r.write(report/'RESULTS.md',f'''# Cold-start secondary-flow qualification

Hydraulic operational gates passed on both resolutions: **{passed}**.
This is not sediment/experimental validation and does not promote a plugin release.

See [qualification receipts](qualification.json), [decay](cold_decay.png),
[bend-scale profiles](cold_secondary_profiles.png).

Next: review coarse/fine bend response before mobile-bed comparison with the
preserved unmodified Delft3D run and digitized experiment. No automatic sediment
simulation has been started. Delft3D was neither modified nor rerun.
''')
    return result


if __name__=='__main__':seeded()
