"""Read-only, all-output inlet qualification on exact common mesh supports.

RMS values are cell-area weighted. Aggregate RMS is the largest instantaneous
RMS, not an average that hides an early transient. Different adaptive meshes
are explicitly skipped; no interpolation is used to disguise support changes.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

TIME_ATOL_S=1e-9
WET_DEPTH_M=1e-6


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}


def labels(folders):
    counts=Counter(p.name for p in folders)
    result=[p.name if counts[p.name]==1 else p.parent.name+'/'+p.name for p in folders]
    if len(set(result))!=len(result):raise ValueError('Ambiguous folder labels; use distinct case/group names')
    return result


def configuration(project):
    settings=project['settings'];request=project['model']['request']
    return dict(backend=request['execution']['backend'],
        inlet_methods={k:v['method'] for k,v in settings['hydraulics'].items() if v['method'].startswith('inlet.')},
        cfl=settings['controls']['cfl'],dt_max_s=settings['dt_max_s'],
        adaptive=settings['adaptive'],grid=settings['grid'],levels=settings['levels'])


def input_signature(project,*,omit_backend=False,omit_time_controls=False):
    # Only metadata and explicitly named comparison axes are excluded.
    result=deepcopy({k:project[k] for k in ('settings','model','boundary_states')})
    result['model']=result['model']['request']
    if omit_backend:result['model']['execution'].pop('backend',None)
    if omit_time_controls:
        for key in ('dt_max_s','end_time_s','output_interval_s'):result['settings'].pop(key,None)
        result['settings']['controls'].pop('cfl',None)
    return result


def pair_category(a,b):
    pa,pb=a['project'],b['project'];ca,cb=configuration(pa),configuration(pb)
    if ca['inlet_methods']!=cb['inlet_methods']:return 'different_boundary_methods'
    if input_signature(pa,omit_backend=True)==input_signature(pb,omit_backend=True):
        return 'same_method_cpu_gpu' if ca['backend']!=cb['backend'] else 'same_method_repeat'
    if ca['backend']==cb['backend'] and input_signature(pa,omit_time_controls=True)==input_signature(pb,omit_time_controls=True):
        return 'same_method_timestep_sensitivity'
    return 'other_input_differences'


def budget_metrics(rows):
    result=dict(output_samples=len(rows),max_water_budget_residual_m3=0.,
        max_water_budget_tolerance_ratio=0.,water_budget_pass=bool(rows),
        max_class_budget_residual_kg=0.,max_class_budget_tolerance_ratio=0.,
        class_budget_pass=bool(rows),max_wall_integral=0.,wall_flux_pass=bool(rows))
    for row in rows:
        water=abs(float(row['water_residual_m3']));wlimit=float(row['water_roundoff_limit_m3'])
        stock=float(np.max(np.abs(row['class_residual_kg']),initial=0.));slimit=float(row['class_roundoff_limit_kg'])
        if not np.isfinite([water,wlimit,stock,slimit]).all() or min(wlimit,slimit)<=0:
            raise ValueError('Nonfinite budget or nonpositive recorded tolerance')
        wall=float(np.max(np.abs(row['wall_integrals']),initial=0.))
        result['max_water_budget_residual_m3']=max(result['max_water_budget_residual_m3'],water)
        result['max_water_budget_tolerance_ratio']=max(result['max_water_budget_tolerance_ratio'],water/wlimit)
        result['water_budget_pass'] &= water<=wlimit
        result['max_class_budget_residual_kg']=max(result['max_class_budget_residual_kg'],stock)
        result['max_class_budget_tolerance_ratio']=max(result['max_class_budget_tolerance_ratio'],stock/slimit)
        result['class_budget_pass'] &= stock<=slimit
        result['max_wall_integral']=max(result['max_wall_integral'],wall)
        result['wall_flux_pass'] &= wall==0.
    return result


def discharge_metrics(project,history):
    inlets={k:v for k,v in project['settings']['hydraulics'].items() if v['method'].startswith('inlet.')}
    result=dict(scope='Recorded output-time inlet flux only; not every internal substep',per_time=[])
    if len(inlets)!=1:
        return dict(result,skipped_reason='History records one inlet; project does not have exactly one inlet')
    name,control=next(iter(inlets.items()))
    method=project['boundary_states'][name]['hydraulic_schedule']['method']
    if method!='forcing.constant':
        return dict(result,skipped_reason='Scheduled target requires time-table evaluation; no constant-target assumption made')
    expected=float(control['discharge_m3_s'])
    for row in history:
        actual=float(row['discharge_m3_s'])
        result['per_time'].append(dict(time_s=row['time_s'],expected_m3_s=expected,actual_m3_s=actual,error_m3_s=actual-expected))
    errors=[abs(row['error_m3_s']) for row in result['per_time']]
    bounded=[r for r in history if 'boundary_Q_roundoff_bound_m3_s' in r]
    result['recorded_boundary_roundoff_check']=dict(samples=len(bounded),
        all_samples_have_bound=len(bounded)==len(history),
        passed=all(abs(r['boundary_Q_error_m3_s'])<=r['boundary_Q_roundoff_bound_m3_s'] for r in bounded) if bounded else None,
        maximum_bound_ratio=max((abs(r['boundary_Q_error_m3_s'])/r['boundary_Q_roundoff_bound_m3_s'] for r in bounded),default=None))
    return dict(result,target_m3_s=expected,maximum_absolute_error_m3_s=max(errors,default=None),
        maximum_relative_error=max(errors)/abs(expected) if errors and expected else None,
        roundoff_tolerance_m3_s=1e-10*max(1.,abs(expected)),
        discharge_pass=bool(errors) and max(errors)<=1e-10*max(1.,abs(expected)))


def load_run(folder):
    qualification=read(folder/'QUALIFICATION.json');project=read(folder/'PROJECT.json')
    index=read(folder/'run/frames/INDEX.json');frames=[];meshes={}
    for row in index['frames']:
        if row['epoch'] not in meshes:meshes[row['epoch']]=arrays(folder/'run/frames'/row['epoch'])
        state=arrays(folder/'run/frames'/row['file']);mesh=meshes[row['epoch']]
        q=state['q'];area=mesh['area']
        if not np.isfinite(q).all() or not np.isfinite(area).all() or np.any(area<=0):
            raise ValueError('Invalid output state/area: '+str(folder/row['file']))
        velocity=np.divide(q[1:],q[0],out=np.zeros_like(q[1:]),where=q[0]>WET_DEPTH_M)
        stats=dict(time_s=float(row['time_s']),cells=len(area),minimum_depth_m=float(q[0].min()),
            maximum_depth_m=float(q[0].max()),maximum_velocity_m_s=float(np.linalg.norm(velocity,axis=0).max()),
            water_volume_m3=float(q[0]@area),wet_area_m2=float(area[q[0]>WET_DEPTH_M].sum()))
        frames.append(dict(row=row,state=state,mesh=mesh,statistics=stats))
    if not frames:raise ValueError('No output frames: '+str(folder))
    if any(b['row']['time_s']<=a['row']['time_s'] for a,b in zip(frames,frames[1:])):
        raise ValueError('Output times must be unique and increasing: '+str(folder))
    diagnostics=read(folder/'run/DIAGNOSTICS.json');status=qualification['status']
    budget=budget_metrics(diagnostics);discharge=discharge_metrics(project,qualification['history'])
    count=status['hydraulic']['accepted_substeps']
    summary=dict(folder=str(folder.resolve()),group=folder.parent.name,case=folder.name,
        configuration=configuration(project),status=status,complete=status['status']=='complete',
        worker_wall_s=qualification['wall_s'],preparation_s=qualification.get('preparation_s'),
        mean_hydraulic_dt_s=status['time_s']/count if count else None,
        min_depth_m=min(f['statistics']['minimum_depth_m'] for f in frames),
        max_depth_m=max(f['statistics']['maximum_depth_m'] for f in frames),
        maximum_velocity_m_s=max(f['statistics']['maximum_velocity_m_s'] for f in frames),
        water_volume_m3=frames[-1]['statistics']['water_volume_m3'],
        maximum_output_inlet_outward_velocity_m_s=max((r['maximum_inlet_outward_velocity_m_s'] for r in qualification['history']),default=None),
        maximum_output_gross_inflow_m3_s=max((r['gross_inflow_m3_s'] for r in qualification['history'] if 'gross_inflow_m3_s' in r),default=None),
        maximum_output_gross_backflow_m3_s=max((r['gross_backflow_m3_s'] for r in qualification['history'] if 'gross_backflow_m3_s' in r),default=None),
        local_backflow_interpretation='Signed local recirculation is allowed; exact net discharge and bounded trajectory are the checks, not zero outward velocity everywhere',
        final_output=qualification['history'][-1] if qualification['history'] else None,
        runtime=qualification['runtime'],frame_extrema_scope='All saved output frames, not every internal substep',
        per_time=[f['statistics'] for f in frames],boundary_per_time=qualification['history'],
        boundary_trace_scope='Saved actual trace statistics; reference water line is not a strong/common stage condition',
        budget=budget,discharge=discharge,
        max_water_budget_residual_m3=budget['max_water_budget_residual_m3'],
        max_Q_error_m3_s=discharge.get('maximum_absolute_error_m3_s'))
    return dict(folder=folder,project=project,qualification=qualification,diagnostics=diagnostics,
        frames=frames,summary=summary)


def same_mesh(a,b):
    if set(a)!=set(b):return False,'mesh array names differ'
    changed=[k for k in a if not np.array_equal(a[k],b[k])]
    return not changed,'mesh arrays differ: '+', '.join(changed)


def field_metrics(a,b,area):
    difference=a-b
    result=dict(depth_max_m=float(np.abs(difference[0]).max()),
        depth_rms_m=float(np.sqrt(np.average(difference[0]**2,weights=area))),
        momentum_max=float(np.abs(difference[1:]).max()),velocity={})
    for threshold in (WET_DEPTH_M,.01):
        wet=(a[0]>threshold)&(b[0]>threshold)
        item=dict(minimum_depth_both_m=threshold,cells=int(wet.sum()),area_m2=float(area[wet].sum()))
        if np.any(wet):
            delta=a[1:,wet]/a[0,wet]-b[1:,wet]/b[0,wet]
            squared=np.sum(delta**2,axis=0)
            item.update(vector_max_m_s=float(np.sqrt(squared.max())),
                vector_rms_m_s=float(np.sqrt(np.average(squared,weights=area[wet]))),
                component_max_m_s=float(np.abs(delta).max()))
        else:item.update(vector_max_m_s=None,vector_rms_m_s=None,component_max_m_s=None)
        result['velocity'][str(threshold)]=item
    return result


def compare_runs(a,b):
    frames_a,frames_b=a['frames'],b['frames'];i=j=0;entries=[];unmatched_a=[];unmatched_b=[]
    while i<len(frames_a) and j<len(frames_b):
        fa,fb=frames_a[i],frames_b[j];ta,tb=fa['row']['time_s'],fb['row']['time_s']
        if abs(ta-tb)<=TIME_ATOL_S:
            match,reason=same_mesh(fa['mesh'],fb['mesh']);entry=dict(time_s=ta,time_b_s=tb)
            if match:entry.update(status='compared',**field_metrics(fa['state']['q'],fb['state']['q'],fa['mesh']['area']))
            else:entry.update(status='skipped',reason=reason+'; no interpolation or adaptive support mapping performed',
                cells_a=len(fa['mesh']['area']),cells_b=len(fb['mesh']['area']))
            entries.append(entry);i+=1;j+=1
        elif ta<tb:unmatched_a.append(ta);i+=1
        else:unmatched_b.append(tb);j+=1
    unmatched_a.extend(f['row']['time_s'] for f in frames_a[i:]);unmatched_b.extend(f['row']['time_s'] for f in frames_b[j:])
    compared=[e for e in entries if e['status']=='compared']
    result=dict(category=pair_category(a,b),group_a=a['folder'].parent.name,group_b=b['folder'].parent.name,
        configuration_a=configuration(a['project']),configuration_b=configuration(b['project']),
        runtime_identity_equal=a['qualification']['runtime']==b['qualification']['runtime'],
        per_time=entries,compared_times=len(compared),skipped_times=len(entries)-len(compared),
        unmatched_times_a_s=unmatched_a,unmatched_times_b_s=unmatched_b,
        time_matching_atol_s=TIME_ATOL_S,aggregate_scope='Maximum over common-time, identical-mesh saved outputs; RMS is maximum instantaneous area-weighted RMS')
    for key in ('depth_max_m','depth_rms_m','momentum_max'):
        result[key]=max((e[key] for e in compared),default=None)
    result['velocity']={}
    for threshold in (WET_DEPTH_M,.01):
        values=[e['velocity'][str(threshold)] for e in compared]
        result['velocity'][str(threshold)]={key:max((v[key] for v in values if v[key] is not None),default=None)
            for key in ('vector_max_m_s','vector_rms_m_s','component_max_m_s')}
    if compared:result['maximum_depth_error_time_s']=max(compared,key=lambda e:e['depth_max_m'])['time_s']
    if not entries:result['skipped_reason']='No matching saved output times'
    return result


def figures(runs,pairs,out):
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,run in runs.items():
        d=run['qualification'];h=d['history'];t=[r['time_s'] for r in h];steps=d['steps']
        axs[0,0].plot(t,[r['maximum_inlet_outward_velocity_m_s'] for r in h],label=name)
        axs[0,1].plot([r['time_s'] for r in steps],[r['diagnostics']['hydraulic_next_substep_hint_s'] for r in steps],label=name)
        # A boundary trace is not a prescribed/common water line. Retain the
        # historical sampler name only when reading explicitly labeled old runs.
        mean=[r.get('mean_trace_stage_m',r.get('actual_trace_stage_mean_m',r.get('compatible_stage_m'))) for r in h]
        line,=axs[1,0].plot(t,mean,label=name)
        if all('actual_trace_stage_min_m' in row for row in h):
            low=[row['actual_trace_stage_min_m'] if row['actual_trace_stage_min_m'] is not None else np.nan for row in h]
            high=[row['actual_trace_stage_max_m'] if row['actual_trace_stage_max_m'] is not None else np.nan for row in h]
            axs[1,0].fill_between(t,low,high,color=line.get_color(),alpha=.08)
        # Measured full ledger includes outflow; do not subtract a hard-coded Qin.
        diag=run['diagnostics'];axs[1,1].plot([r['time_s'] for r in diag],[r['water_residual_m3'] for r in diag],label=name)
    axs[0,0].set(title='Largest outward inlet-cell velocity',ylabel='m/s (inflow is negative)')
    axs[0,1].set(title='Unclipped next hydraulic timestep',ylabel='s')
    axs[1,0].set(title='Mean actual inlet-trace elevation; shaded face range',ylabel='m (not a strong/common stage condition)')
    axs[1,1].set(title='Water conservation ledger residual',ylabel='m³')
    for ax in axs.flat:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7)
    fig.savefig(out/'BOUNDARY_HISTORY.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,len(runs),figsize=(max(6,3.2*len(runs)),5),layout='constrained',squeeze=False)
    for ax,(name,run) in zip(axs.flat,runs.items()):
        state,g=run['frames'][-1]['state'],run['frames'][-1]['mesh']
        polys=[g['vertices'][g['offsets'][i]:g['offsets'][i+1]] for i in range(len(g['area']))]
        domain=np.asarray(run['project']['settings']['domain']['vertices_xy']);origin=domain.min(0)
        # This campaign's inlet is at the south end; raw cells, no smoothing.
        pick=np.array([np.mean(p[:,1])-origin[1]<60 for p in polys])
        q=state['q'];v=np.divide(q[2],q[0],out=np.zeros_like(q[0]),where=q[0]>WET_DEPTH_M)
        pc=PolyCollection([p-origin for p,k in zip(polys,pick) if k],array=v[pick],
            cmap='RdBu',clim=(-3,3),edgecolors='.65',linewidths=.15)
        ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(name,fontsize=9)
        ax.set(xlabel='Local easting (m)',ylabel='Local northing (m)')
    fig.colorbar(pc,ax=list(axs.flat),label='Northward velocity (m/s)',shrink=.7)
    fig.savefig(out/'INLET_VELOCITY.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for name,pair in pairs.items():
        if pair['category'] not in ('same_method_cpu_gpu','same_method_timestep_sensitivity','same_method_repeat'):continue
        rows=[row for row in pair['per_time'] if row['status']=='compared']
        if not rows:continue
        label=pair['category'].replace('same_method_','')+': '+name
        axs[0].plot([r['time_s'] for r in rows],[1000*r['depth_max_m'] for r in rows],label=label)
        axs[1].plot([r['time_s'] for r in rows],[r['velocity']['0.01']['vector_max_m_s'] for r in rows],label=label)
    axs[0].set(title='Maximum depth difference at each common output',ylabel='mm')
    axs[1].set(title='Maximum velocity-vector difference; both depths > 1 cm',ylabel='m/s')
    for ax in axs:ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.2)
    if axs[0].lines:axs[0].legend(fontsize=6)
    fig.savefig(out/'TIME_COMPARISONS.png',dpi=170);plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folders',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--no-figures',action='store_true');args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    runs={name:load_run(folder) for name,folder in zip(labels(args.folders),args.folders)}
    pairs={};names=list(runs)
    for i,a in enumerate(names):
        for b in names[i+1:]:pairs[a+'__'+b]=compare_runs(runs[a],runs[b])
    result=dict(runs={name:r['summary'] for name,r in runs.items()},pairs=pairs)
    (out/'METRICS.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if not args.no_figures:figures(runs,pairs,out)
    print(json.dumps(dict(output=str(out/'METRICS.json'),runs=len(runs),pairs=len(pairs),
        compared_times=sum(p['compared_times'] for p in pairs.values()),skipped_times=sum(p['skipped_times'] for p in pairs.values())),indent=2))


if __name__=='__main__':main()
