"""Nash's already depth-averaged experiment: matched-control normalization."""
import csv,importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'desktop'))
from turbine_validation59 import HOME,REPORT,REF,CASES
import analyze_turbine_validation58 as common
common.HOME=HOME;common.REPORT=REPORT
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection


def speed(state):return np.hypot(state[1],state[2])
def speed_state(case,states=None):
    history=case.history if states is None else states
    return np.array([case.mean[0],np.mean([speed(s) for _,s in history],axis=0),np.zeros_like(case.mean[0])])


def deficit(turbine,control,xD,yD=0.,turbine_state=None,control_state=None):
    t=speed_state(turbine) if turbine_state is None else turbine_state
    c=speed_state(control) if control_state is None else control_state
    return 1-turbine.sample(xD,np.atleast_1d(yD),t)/control.sample(xD,np.atleast_1d(yD),c)


def series(turbine,control,xD):
    return np.array([deficit(turbine,control,float(x))[0] for x in xD])


def metrics(error,target):
    return dict(RMSE_pp=float(100*np.sqrt(np.mean(error**2))),maximum_absolute_pp=float(100*np.max(abs(error))),
        relative_L2_deficit=float(np.linalg.norm(error)/np.linalg.norm(target)),mean_bias_pp=float(100*np.mean(error)))


def widths(y,d):
    peak=int(np.argmax(d));half=d[peak]/2
    left=np.flatnonzero(d[:peak+1]<=half);right=np.flatnonzero(d[peak:]<=half)+peak
    if not len(left) or not len(right):return None
    i=left[-1];j=right[0]
    l=float(np.interp(half,d[i:i+2],y[i:i+2]));r=float(np.interp(half,d[j-1:j+1][::-1],y[j-1:j+1][::-1]))
    return r-l


def analyze():
    cases={n:common.Case(n) for n in ['nash_gpu','zero_gpu']};a=cases['nash_gpu'];control=cases['zero_gpu']
    np.testing.assert_array_equal(a.xy,control.xy)
    raw=list(csv.DictReader((REF/'measurements/nash_2015_figure6b.csv').open(encoding='utf-8')))
    refs={k:np.array([[float(r['x_over_d']),float(r['velocity_deficit'])] for r in raw if r['series']==k]) for k in {r['series'] for r in raw}}
    result=dict(comparison={},lateral={},configurations_held_fixed=True,controls='1 - speed_turbine / speed_zero_thrust, not nominal-inlet normalization',
        projection='None: author-published full-water-column experimental average is used directly. Raw measurements not acquired.',
        physical_solver_modified=False,arithmetic_check_modified=True,runtime_sha256=json.loads((REPORT/'LOCKED_SPEC.json').read_text())['runtime_sha256'])
    rows=[];fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    xd=np.linspace(2.5,20,141);pred=series(a,control,xd)
    for ax in axes:
        ax.plot(xd,100*pred,color='#236da2',label='BRAIDS GPU / matched zero-thrust control')
        for key,col,label in [('published_2d_model','#353535','Published Nash 2D'),('experiment_depth_average','#398958','Experiment, author depth average')]:
            r=refs[key];ax.plot(r[:,0],100*r[:,1],'--',color=col,label=label)
        ax.set(xlabel='Distance downstream / rotor diameter',ylabel='Velocity deficit (%)',ylim=(0,30));ax.grid(alpha=.2)
    axes[0].set_title('Full comparison, including near wake');axes[1].set_title('Far wake from 4D');axes[1].set_xlim(4,20);axes[1].set_ylim(0,22)
    axes[0].legend(fontsize=8);fig.suptitle('Direct published depth-mean target; frozen steady-flood reconstruction, not the author tidal run')
    fig.savefig(REPORT/'DIRECT_EXPERIMENT.png',dpi=170);plt.close(fig)
    for name,curve in refs.items():
        if name=='experiment_hub_plane':continue
        p=series(a,control,curve[:,0]);error=p-curve[:,1];mask=curve[:,0]>=4
        strict=curve[:,0]>4
        worst=int(np.argmax(abs(error[mask])))
        result['comparison'][name]=dict(all=metrics(error,curve[:,1]),farwake_ge4D=metrics(error[mask],curve[mask,1]),
            supplemental_strictly_beyond4D=metrics(error[strict],curve[strict,1]),
            largest_ge4D_error_at_D=float(curve[mask,0][worst]),
            note='Original >=4D screen retained. Strictly >4D is supplemental because the paper describes agreement beyond 4D; it does not replace the frozen gate.')
        for x,b,t in zip(curve[:,0],p,curve[:,1]):rows.append(dict(kind='centreline',reference=name,x_D=x,y_D=0.,braids_deficit=b,reference_deficit=t))
    lateral=list(csv.DictReader((REF/'measurements/nash_2015_figure7.csv').open(encoding='utf-8')))
    fig,axes=plt.subplots(1,4,figsize=(13,4),layout='constrained')
    for ax,x in zip(axes,[2.,4.,10.,20.]):
        data=[r for r in lateral if float(r['x_over_d'])==x];y=np.array([float(r['y_over_d']) for r in data]);target=np.array([float(r['velocity_deficit']) for r in data])
        p=deficit(a,control,x,y);grid=np.linspace(-5,5,201);dg=deficit(a,control,x,grid)
        result['lateral'][f'{x:g}D']=dict(**metrics(p-target,target),peak_deficit_percent=float(100*np.max(dg)),
            reference_peak_percent=float(100*max(target)),full_half_height_width_D=widths(grid,dg),reference_full_half_height_width_D=widths(y,target))
        ax.plot(100*dg,grid,color='#236da2',label='BRAIDS');ax.plot(100*target,y,'k--',label='Published Nash 2D')
        ax.set(xlabel='Velocity deficit (%)',ylabel='Transverse position / D',title=f'{x:g}D',ylim=(-3,3));ax.grid(alpha=.2)
        for yy,b,t in zip(y,p,target):rows.append(dict(kind='lateral',reference='published_2d_model',x_D=x,y_D=yy,braids_deficit=b,reference_deficit=t))
    axes[0].legend(fontsize=8);fig.suptitle('Wake shape and spreading: published model comparison, not measured lateral data')
    fig.savefig(REPORT/'LATERAL_WAKE.png',dpi=170);plt.close(fig)
    halves=[]
    for case in [a,control]:
        mid=(case.c['average']+case.c['end'])/2
        halves.append([speed_state(case,[(t,s) for t,s in case.history if t<=mid]),speed_state(case,[(t,s) for t,s in case.history if t>mid])])
    first=np.array([deficit(a,control,float(x),turbine_state=halves[0][0],control_state=halves[1][0])[0] for x in xd])
    second=np.array([deficit(a,control,float(x),turbine_state=halves[0][1],control_state=halves[1][1])[0] for x in xd])
    result['maximum_centreline_halfwindow_change_pp']=float(100*np.max(abs(second-first)))
    result['backend_bounded_100s']={}
    for cpu_name,gpu_name in [('nash_cpu','nash_gpu'),('zero_cpu','zero_gpu')]:
        record=json.loads((REPORT/'bounded_cpu'/(cpu_name+'.json')).read_text())
        with np.load(REPORT/'bounded_cpu'/(cpu_name+'_100s.npz')) as f:cpu={k:f[k].copy() for k in f.files}
        case=cases[gpu_name];idx=int(np.flatnonzero(case.store.times==100.)[0]);gpu_frame=case.store.read(idx)
        gpu=case.store.index['frames'][idx];__import__('shutil').copy2(case.output/'frames'/gpu['file'],REPORT/'bounded_cpu'/(gpu_name+'_100s.npz'))
        result['backend_bounded_100s'][cpu_name]=dict(comparison_time_s=100.,
            maximum_depth_m=float(np.max(abs(cpu['q'][0]-gpu_frame.q[0]))),
            maximum_velocity_m_s=float(np.max(abs(cpu['q'][1:]/cpu['q'][0]-gpu_frame.q[1:]/gpu_frame.q[0]))),
            cpu_output=record['output'],gpu_output=str(case.output),
            scope='Matched 100s startup; not a full 2000s CPU mean-wake comparison.')
    result['runs']={}
    for name,case in cases.items():
        ds=json.loads((case.output/'DIAGNOSTICS.json').read_text());status=json.loads((case.output/'STATUS.json').read_text())
        force=float(np.mean([r['thrust_N'] for r in case.rows]));refspeed=float(np.mean([r['upstream_velocity_m_s'] for r in case.rows]))
        area=np.pi*case.c['D']**2/4;recovered=force/(.5*1000*area*refspeed**2) if refspeed else 0.
        result['runs'][name]=dict(output=str(case.output),thrust_N=force,mean_reconstructed_approach_m_s=refspeed,
            recovered_Ct_at_approach=recovered,input_Ct=case.c['ct'],mean_power_W=float(np.mean([r['useful_power_W'] for r in case.rows])),
            maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in ds),
            accepted_substeps=status['hydraulic']['accepted_substeps'],outer_rejections=len(status.get('rejected_trials',[])))
    result['zero_control_speed_departure_percent']=float(100*np.max(abs(speed_state(control)[1]-control.c['U']))/control.c['U'])
    fig,axes=plt.subplots(2,1,figsize=(11,6),layout='constrained')
    for ax,case in zip(axes,[a,control]):
        c=case.c;mask=(case.xy[:,0]>c['X']-2*c['D'])&(case.xy[:,0]<c['X']+25*c['D'])&(abs(case.xy[:,1]-c['Y'])<4*c['D'])
        ids=np.flatnonzero(mask);polys=[(np.asarray(case.polygons[i].exterior.coords)-[c['X'],c['Y']])/c['D'] for i in ids]
        artist=PolyCollection(polys,array=speed_state(case)[1,ids]/c['U'],cmap='viridis',clim=(.7,1.03),edgecolors='none');ax.add_collection(artist)
        ax.plot([0,0],[-.5,.5],color='red',lw=2);ax.set(xlim=(-2,25),ylim=(-4,4),aspect='equal',xlabel='x/D',ylabel='y/D',title=case.name)
        fig.colorbar(artist,ax=ax,label='Mean speed / nominal inlet')
    fig.savefig(REPORT/'WAKE_MAPS.png',dpi=170);plt.close(fig)
    common.export(REPORT/'COMPARISON.csv',rows)
    limits=json.loads((HOME/CASES['nash_cpu']['folder']/'VALIDATION_SPEC.json').read_text())['screens']
    result['screens']={}
    for key in ['published_2d_model','experiment_depth_average']:
        m=result['comparison'][key]['farwake_ge4D'];result['screens'][key]=m['RMSE_pp']<=limits['centreline_RMS_pp'] and m['maximum_absolute_pp']<=limits['centreline_maximum_pp'] and m['relative_L2_deficit']<=limits['relative_L2_deficit']
    result['screens']['stationarity']=result['maximum_centreline_halfwindow_change_pp']<=limits['profile_halfwindow_change_pp']
    common.save(REPORT/'METRICS.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':analyze()
