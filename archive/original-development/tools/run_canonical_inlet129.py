"""Sequential four-inlet CPU/GPU fixed/AMR qualification, without tuning.

The 100-second dry-start channel is a transient, not an equilibrium validation.
Analytical normal/critical depths qualify the incoming reference roots only.
Actual traces are sampled from boundary Riemann solutions, including shocks;
their water elevations need not equal each other or the external reference.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
METHODS=('normal','critical','stage','depth')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def signature(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame(output,record):
    folder=Path(output)/'run/frames'
    with np.load(folder/record['epoch']) as f:mesh={k:f[k] for k in f.files}
    with np.load(folder/record['file']) as f:fields={k:f[k] for k in f.files}
    return mesh,fields


def raster(mesh,value):
    """Canonical rectangles: conservative constant prolongation to finest5m.

    This is used only for cross-mesh comparisons/profiles; raw arrays remain
    separately compared whenever the actual saved geometry is identical.
    """
    result=np.full((*value.shape[:-1],4,20),np.nan)
    for c,(left,right) in enumerate(zip(mesh['offsets'][:-1],mesh['offsets'][1:])):
        v=mesh['vertices'][left:right];lo=v.min(0);hi=v.max(0)
        i0,j0=np.rint(lo/5).astype(int);i1,j1=np.rint(hi/5).astype(int)
        if not np.allclose(np.r_[lo,hi]/5,np.rint(np.r_[lo,hi]/5),atol=1e-9):
            raise ValueError('Canonical comparison requires actual grid-aligned rectangular cells')
        result[...,j0:j1,i0:i1]=value[...,c,None,None]
    if not np.isfinite(result).all():raise ValueError('Canonical mesh does not cover the100x20m rectangle')
    return result


def comparison(first,second):
    ia=read(Path(first)/'run/frames/INDEX.json')['frames'];ib=read(Path(second)/'run/frames/INDEX.json')['frames']
    if [r['time_s'] for r in ia]!=[r['time_s'] for r in ib]:raise ValueError('Saved comparison clocks differ')
    same=True;max_depth=rms_depth=max_speed=0.;max_raw=np.zeros(3);max_bed=0.;rows=[]
    for ar,br in zip(ia,ib):
        ma,a=frame(first,ar);mb,b=frame(second,br)
        equal=all(np.array_equal(ma[k],mb[k]) for k in ('leaves','offsets','vertices','area'))
        same=bool(same and equal)
        qa,qb=(a['q'],b['q']) if equal else (raster(ma,a['q']),raster(mb,b['q']))
        delta=qa-qb;maximum=float(np.max(abs(delta[0])));rms=float(np.sqrt(np.mean(delta[0]**2)))
        wet=(qa[0]>.01)&(qb[0]>.01)
        va=np.divide(qa[1:],qa[0],out=np.zeros_like(qa[1:]),where=qa[0]>1e-6)
        vb=np.divide(qb[1:],qb[0],out=np.zeros_like(qb[1:]),where=qb[0]>1e-6)
        speed=float(np.max(np.linalg.norm(va-vb,axis=0)[wet],initial=0.))
        if equal:
            max_raw=np.maximum(max_raw,np.max(abs(delta),axis=1));max_bed=max(max_bed,float(np.max(abs(a['z']-b['z']))))
        max_depth=max(max_depth,maximum);rms_depth=max(rms_depth,rms);max_speed=max(max_speed,speed)
        rows.append(dict(time_s=ar['time_s'],raw_geometry_equal=equal,depth_max_m=maximum,depth_rms_m=rms,
            velocity_vector_max_m_s_both_depth_above1cm=speed))
    return dict(raw_geometry_equal_all_frames=same,raw_q_component_max_difference=max_raw.tolist() if same else None,
        raw_bed_max_difference_m=max_bed if same else None,depth_max_m=max_depth,depth_rms_max_m=rms_depth,
        velocity_vector_max_m_s_both_depth_above1cm=max_speed,history=rows,
        mapping='raw identical geometry' if same else 'raw where identical; conservative constant finest-grid prolongation otherwise')


def run_case(source,output,analytic,*,reference_cfl=None):
    from braids_core import river_worker as worker
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.shoreline_solver import ShorelineSolver
    from braids_core.sediment.bed_columns_v101 import host
    from braids_gpu.atomic_output import write_json_atomic
    output=Path(output);output.mkdir(parents=True,exist_ok=False);doc=read(source)
    if reference_cfl is not None:doc['settings']['controls']['cfl']=reference_cfl
    write_json_atomic(output/'PROJECT.json',doc);definition=doc['settings']['hydraulics']['inlet']
    names=sorted(doc['settings']['hydraulics']);inlet=names.index('inlet');outlet=names.index('outlet')
    context=dict(source=str(Path(source).resolve()),source_sha256=signature(source),
        changes={} if reference_cfl is None else {'settings.controls.cfl':reference_cfl},
        interpretation='100-second dry-start transient; analytical depths apply to external reference, not an equilibrium target; actual Riemann face water lines may differ',
        runtime_id=worker.source_identity())
    write_json_atomic(output/'INPUT_CHANGES.json',context)
    began=time.perf_counter();prepared=worker.prepare(output/'PROJECT.json',output/'prepared');prepare_s=time.perf_counter()-began
    owners=[];history=[];steps=[];original_factory=core.factory;original_step=ShorelineSolver.step
    def factory(*args,**kw):
        make=original_factory(*args,**kw)
        def create(*args,**kw):
            owner=make(*args,**kw);owners.append(owner);return owner
        create.required_leaves=make.required_leaves;return create
    def step(self,dt,*args,**kw):
        start=time.perf_counter();result=original_step(self,dt,*args,**kw)
        steps.append(dict(time_s=self.time,outer_dt_s=dt,wall_s=time.perf_counter()-start,diagnostics=result['diagnostics']))
        return result
    def sample(folder,row):
        a=owners[-1];e=a.exterior_bed;d=a.data;p=a.physics;q=host(a.q);z=host(a.z)
        e.evaluate(a.q,a.z,time=a.time);trace=host(e.trace);flux=host(e.flux);bed=host(p.exterior_bed)
        compatible=e.inlet_compatibility;root_iterations=host(compatible.iterations)
        target=host(compatible.target_stage);L=d['length'];outside=d['neighbour']<0
        incoming=outside&(d['bid']==inlet);outgoing=outside&(d['bid']==outlet)
        Q=-float(flux[0,incoming]@L[incoming]);Qout=float(flux[0,outgoing]@L[outgoing])
        active=incoming&(trace[0]>p.parameters.dry);h=trace[0,active];A=float(h@L[active]);T=float(L[active].sum())
        face_eta=h+bed[active]
        inward_per_width=-flux[0,incoming]
        gross_in=float(np.maximum(inward_per_width,0.)@L[incoming])
        gross_out=float(np.maximum(-inward_per_width,0.)@L[incoming])
        face_u=np.divide(-np.sum(trace[1:,active]*d['normal'][:,active],axis=0),h,
            out=np.zeros_like(h),where=h>p.parameters.dry)
        face_fr=face_u/np.sqrt(p.parameters.gravity*h)
        ht=np.maximum(float(target[inlet])-bed[incoming],0.);wet=ht>p.parameters.dry
        At=float(ht[wet]@L[incoming][wet]);Tt=float(L[incoming][wet].sum());gravity=p.parameters.gravity
        Fr=(Q*Q*T/(gravity*A**3))**.5 if A else 0.;Frt=(Q*Q*Tt/(gravity*At**3))**.5 if At else 0.
        inlet_cells=d['owner'][incoming];outlet_cells=d['owner'][outgoing]
        v=np.divide(q[1:],q[0],out=np.zeros_like(q[1:]),where=q[0]>p.parameters.dry)
        outward=float(np.max(np.sum(v[:,inlet_cells]*d['normal'][:,incoming],axis=0),initial=0.))
        normal_metric=None
        if e.normal_section is not None:
            metric=host(e.normal_section.metrics())[:,inlet]
            normal_metric=dict(area_m2=float(metric[0]),wetted_perimeter_m=float(metric[1]),drag_integral=float(metric[2]),
                implied_discharge_m3_s=float(metric[3]),implied_slope=Q*Q*float(metric[2])/float(metric[0])**3 if metric[0] else None)
        ledger=host(a.water_ledger)
        history.append(dict(time_s=a.time,cells=p.n,inlet_Q_m3_s=Q,outlet_Q_m3_s=Qout,
            inlet_accepted_volume_m3=float(ledger[incoming].sum()),outlet_accepted_volume_m3=-float(ledger[outgoing].sum()),
            stored_water_m3=float(q[0]@d['area']),water_residual_m3=float(row['water_residual_m3']),
            minimum_depth_m=float(q[0].min()),maximum_depth_m=float(q[0].max()),maximum_speed_m_s=float(np.linalg.norm(v,axis=0).max()),
            maximum_inlet_outward_cell_velocity_m_s=outward,outlet_maximum_cell_depth_m=float(q[0,outlet_cells].max()),
            actual_equivalent_section_froude=Fr,reference_section_froude=Frt,
            actual_trace_stage_min_m=float(face_eta.min()) if len(face_eta) else None,
            actual_trace_stage_max_m=float(face_eta.max()) if len(face_eta) else None,
            actual_trace_stage_width_mean_m=float(face_eta@L[active]/T) if T else None,
            reference_common_stage_m=float(target[inlet]),
            actual_section_mean_depth_m=A/T if T else 0.,reference_mean_depth_m=At/Tt if Tt else 0.,
            actual_local_inward_froude_min=float(face_fr.min()) if len(face_fr) else None,
            actual_local_inward_froude_max=float(face_fr.max()) if len(face_fr) else None,
            gross_inlet_inflow_m3_s=gross_in,gross_inlet_backflow_m3_s=gross_out,
            inward_faces=int(np.count_nonzero(inward_per_width>1e-12)),
            backflow_faces=int(np.count_nonzero(inward_per_width < -1e-12)),
            zero_mass_faces=int(np.count_nonzero(abs(inward_per_width)<=1e-12)),normal_reference_metric=normal_metric,
            virtual_exterior_velocity_m_s=float(host(compatible.exterior_velocity)[inlet]),
            outer_section_root_iterations=int(root_iterations[0,inlet]),
            maximum_inner_riemann_iterations=int(root_iterations[1,inlet]),
            inlet_method_identity=compatible.identity,
            section_froude_interpretation='Equivalent scalar sqrt(Qnet^2*T/(g*A^3)); variable face velocity/depth means this is not a local critical-flow test'))
    core.factory=factory;ShorelineSolver.step=step
    started=time.perf_counter()
    try:status=worker.run(prepared,output/'run',on_frame=sample)
    finally:core.factory=original_factory;ShorelineSolver.step=original_step
    elapsed=time.perf_counter()-started
    ds=[r['diagnostics'] for r in steps];count=sum(d['hydraulic_substeps'] for d in ds)
    dt=dict(mean_s=status['time_s']/count if count else None,
        minimum_s=min((d['hydraulic_minimum_dt'] for d in ds),default=None),
        maximum_s=max((d['hydraulic_maximum_dt'] for d in ds),default=None),
        mean_final20s_s=sum(r['outer_dt_s'] for r in steps if r['time_s']>80)/sum((r['diagnostics']['hydraulic_substeps'] for r in steps if r['time_s']>80),start=0) if any(r['time_s']>80 for r in steps) else None,
        accepted_substeps=count,endpoint_remainders=sum(d.get('hydraulic_endpoint_remainders',0) for d in ds),
        accepted_cfl_max=max((d['accepted_cfl_max'] for d in ds),default=0.),configured_face_sum_cfl=doc['settings']['controls']['cfl'],
        warning='The minimum includes exact output/AMR clock remainders; compare mean and final20s mean for persistent timestep size')
    expected=analytic['critical_depth_m'] if definition['method']=='inlet.critical_discharge' else analytic['normal_depth_m']
    target_error=max(abs(r['reference_mean_depth_m']-expected) for r in history)
    checks=dict(completed=status['status']=='complete',positive_depth=all(r['minimum_depth_m']>=0 for r in history),
        imposed_Q=max(abs(r['inlet_Q_m3_s']-11.) for r in history)<1e-8,
        conservation=max(abs(r['water_residual_m3']) for r in history)<1e-8,
        analytical_incoming_reference=target_error<1e-9,
        outlet_reached_by100s=any(r['outlet_Q_m3_s']>1e-6 and r['outlet_maximum_cell_depth_m']>1e-6 for r in history),
        accepted_cfl_within_bound=dt['accepted_cfl_max']<=dt['configured_face_sum_cfl']*(1+1e-10),
        finite_riemann_trace_metrics=all(np.isfinite([r['actual_trace_stage_min_m'],r['actual_trace_stage_max_m'],
            r['actual_local_inward_froude_min'],r['actual_local_inward_froude_max']]).all() for r in history))
    result=dict(context=context,status=status,prepare_s=prepare_s,worker_s=elapsed,hydraulic_dt=dt,
        analytical_reference_depth_m=expected,
        boundary_interpretation='Exact net section Q; signed local Riemann flux is retained, not clipped to hide backflow. The common external reference stage differs from actual sampled face stages; an outgoing rarefaction invariant is not imposed across a shock.',
        reference_depth_max_error_m=target_error,history=history,steps=steps,checks=checks,
        timing_scope='Instrumented validation incl. output/callbacks; not a clean isolated speed benchmark')
    write_json_atomic(output/'METRICS.json',result)
    print(output.name,json.dumps(dict(checks=checks,dt=dt,worker_s=elapsed)),flush=True)
    return result


def summarize(output):
    from braids_gpu.atomic_output import write_json_atomic
    out=Path(output);names=[f'{mesh}_{backend}_{method}' for mesh in ('static','dynamic') for backend in ('cpu','gpu') for method in METHODS]
    runs={n:read(out/n/'METRICS.json') for n in names if (out/n/'METRICS.json').is_file()}
    result=dict(runs={n:dict(checks=r['checks'],dt=r['hydraulic_dt'],worker_s=r['worker_s'],prepare_s=r['prepare_s']) for n,r in runs.items()},
        comparisons={},cfl_references={},interpretation='Dry-start100s transient, not an equilibrium validation; reference roots and actual backwater dynamics are distinct',
        acceptance=dict(backend_depth_max_m=.005,temporal_depth_max_m=.005,
            strict_backend_diagnostic_m=1e-6,
            reason='User permits a few millimetres; the 1 micrometre screen is retained as a stricter diagnostic, not silently relabelled as passed. Conservation and all physical per-run checks remain mandatory.'))
    for mesh in ('static','dynamic'):
        for method in METHODS:
            cpu=f'{mesh}_cpu_{method}';gpu=f'{mesh}_gpu_{method}'
            if cpu in runs and gpu in runs:
                value=comparison(out/cpu,out/gpu);value['paired_depth_below1micrometre']=value['depth_max_m']<1e-6
                value['paired_depth_within5mm']=value['depth_max_m']<=.005
                result['comparisons'][mesh+'_'+method+'_cpu_gpu']=value
        for backend in ('cpu','gpu'):
            stage=f'{mesh}_{backend}_stage';depth=f'{mesh}_{backend}_depth'
            if stage in runs and depth in runs:
                value=comparison(out/stage,out/depth);value['equivalent_below1nanometre']=value['depth_max_m']<1e-9
                result['comparisons'][mesh+'_'+backend+'_stage_depth']=value
        for method in ('normal','critical'):
            main=f'{mesh}_cpu_{method}';reference=main+'_cflref'
            if main in runs and (out/reference/'METRICS.json').is_file():
                value=comparison(out/main,out/reference);value['depth_within5mm']=value['depth_max_m']<=.005
                value['reference_checks']=read(out/reference/'METRICS.json')['checks']
                result['cfl_references'][main]=value
    write_json_atomic(out/'COMPARISONS.json',result)
    if runs:figures(out,runs,result)
    return result


def figures(out,runs,comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(normal='#246aa8',critical='#cc5535',stage='#33966d',depth='#995bb1')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,mesh in enumerate(('static','dynamic')):
        for method in METHODS:
            name=f'{mesh}_cpu_{method}'
            if name not in runs:continue
            r=runs[name];h=r['history'];t=[x['time_s'] for x in h]
            axes[i,0].plot(t,[x['inlet_Q_m3_s'] for x in h],color=colors[method],ls='--',alpha=.5)
            axes[i,0].plot(t,[x['outlet_Q_m3_s'] for x in h],label=method,color=colors[method])
            axes[i,1].plot(t,[x['actual_equivalent_section_froude'] for x in h],label=method,color=colors[method])
        axes[i,0].set_title(mesh+' mesh: dashedQin / solidQout');axes[i,0].set_ylabel('Discharge (m³/s)')
        axes[i,1].set_title(mesh+' mesh: equivalent Fr from netQ, area, width');axes[i,1].set_ylabel('Equivalent section Froude number')
        for ax in axes[i]:ax.set_xlabel('Physical time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylim(bottom=0)
    fig.suptitle('Four inlet references — exact11m³/s and evolving dry-start/backwater dynamics')
    fig.tight_layout();fig.savefig(out/'CANONICAL_BOUNDARIES.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,mesh in enumerate(('static','dynamic')):
        for method in METHODS:
            name=f'{mesh}_cpu_{method}'
            if name not in runs:continue
            h=runs[name]['history'];t=[row['time_s'] for row in h];color=colors[method]
            axes[i,0].plot(t,[row['reference_common_stage_m'] for row in h],color=color,ls='--',alpha=.6)
            axes[i,0].plot(t,[row['actual_trace_stage_width_mean_m'] for row in h],color=color,label=method)
            axes[i,0].fill_between(t,[row['actual_trace_stage_min_m'] for row in h],
                [row['actual_trace_stage_max_m'] for row in h],color=color,alpha=.12)
            axes[i,1].plot(t,[row['gross_inlet_backflow_m3_s'] for row in h],color=color,label=method)
        axes[i,0].set_title(mesh+': reference dashed / actual mean + range')
        axes[i,0].set_ylabel('Boundary water elevation (m)')
        axes[i,1].set_title(mesh+': signed local backflow, without clipping')
        axes[i,1].set_ylabel('Gross outward flow on inlet faces (m3/s)');axes[i,1].set_ylim(bottom=0)
        for ax in axes[i]:ax.set_xlabel('Physical time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('External reference is common; actual face traces follow shock/rarefaction Riemann solutions')
    fig.tight_layout();fig.savefig(out/'CANONICAL_REFERENCE_TRACE.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(20)*5+2.5
    for i,meshmode in enumerate(('static','dynamic')):
        for method in METHODS:
            for backend in ('cpu','gpu'):
                name=f'{meshmode}_{backend}_{method}'
                if name not in runs:continue
                record=read(out/name/'run/frames/INDEX.json')['frames'][-1];mesh,fields=frame(out/name,record)
                q=raster(mesh,fields['q']);h=q[0].mean(0);u=np.divide(q[1].sum(0),q[0].sum(0),out=np.zeros(20),where=q[0].sum(0)>1e-6)
                kwargs=dict(color=colors[method],label=method if backend=='cpu' else None,
                    linestyle='-' if backend=='cpu' else 'None',marker=None if backend=='cpu' else '.',alpha=.85)
                axes[i,0].plot(x,h,**kwargs);axes[i,1].plot(x,u,**kwargs)
        for method,style in [('normal',':'),('critical','--')]:
            key=f'{meshmode}_cpu_{method}'
            if key in runs:axes[i,0].axhline(runs[key]['analytical_reference_depth_m'],color='gray',ls=style,label=method.title()+' incoming reference')
        axes[i,0].set_ylabel('Width-mean depth (m)');axes[i,1].set_ylabel('Depth-weighted section speed (m/s)')
        for ax in axes[i]:ax.set_title(meshmode+' mesh');ax.set_xlabel('Distance downstream (m)');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('100s dry-start transient — CPU lines / GPU dots; reference depths are NOT an equilibrium target')
    fig.tight_layout();fig.savefig(out/'CANONICAL_PROFILES.png',dpi=160);plt.close(fig)
    selected=[(n,r) for n,r in runs.items() if '_cpu_' in n]
    fig,axes=plt.subplots(1,2,figsize=(12,4));positions=np.arange(len(selected));labels=[n.replace('_cpu_','\n') for n,r in selected]
    axes[0].bar(positions,[r['hydraulic_dt']['mean_s'] for n,r in selected],label='Whole-run mean')
    axes[0].plot(positions,[r['hydraulic_dt']['mean_final20s_s'] for n,r in selected],'o',color='#cc5535',label='Last20s mean')
    axes[1].bar(positions,[max(abs(x['water_residual_m3']) for x in r['history']) for n,r in selected])
    axes[0].set_ylabel('Accepted hydraulic dt (s)');axes[0].legend();axes[1].set_ylabel('Maximum water-budget residual (m³)')
    for ax in axes:ax.set_xticks(positions,labels,fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Clock and conservation checks; minimum dt also includes output/AMR remainders')
    fig.tight_layout();fig.savefig(out/'CANONICAL_TIMESTEPS_BUDGET.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,result in comparisons['comparisons'].items():
        if name.endswith('_cpu_gpu'):
            h=result['history'];axes[0].plot([r['time_s'] for r in h],
                [1e3*r['depth_max_m'] for r in h],label=name.replace('_cpu_gpu',''))
    for name,result in comparisons['cfl_references'].items():
        h=result['history'];axes[1].plot([r['time_s'] for r in h],
            [1e3*r['depth_max_m'] for r in h],label=name.replace('_cpu_',' '))
    axes[0].set_title('CPU/GPU difference at every saved output')
    axes[1].set_title('CFL .45 versus .1125, same physical outputs')
    axes[1].axhline(5.,ls='--',color='black',alpha=.6,label='5 mm temporal screen')
    for ax in axes:
        ax.set_xlabel('Physical time (s)');ax.set_ylabel('Maximum depth difference (mm)')
        ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Raw same-mesh fields; conservative finest-grid prolongation only when actual meshes differ')
    fig.tight_layout();fig.savefig(out/'CANONICAL_DIFFERENCES.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--select',nargs='+');p.add_argument('--cfl-reference',action='store_true')
    p.add_argument('--analyze-only',action='store_true');p.add_argument('--resume-completed',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);analytic=read(a.inputs/'INPUTS.json')
    if not a.analyze_only:
        names=[Path(n).stem for n in analytic['decks']]
        if a.select:
            if not set(a.select)<=set(names):raise ValueError('Unknown canonical deck')
            names=[n for n in names if n in a.select]
        for name in names:
            folder=a.output/name
            if a.resume_completed and (folder/'METRICS.json').is_file():
                from braids_core.river_worker import source_identity
                old=read(folder/'METRICS.json')
                if old['context']['runtime_id']!=source_identity() or old['context']['source_sha256']!=signature(a.inputs/(name+'.json')):
                    raise ValueError('Cannot reuse completed canonical case with changed source/input')
                if not old['checks']['completed']:raise ValueError('Cannot skip an incomplete canonical run')
            else:run_case(a.inputs/(name+'.json'),folder,analytic)
        if a.cfl_reference:
            for mesh in ('static','dynamic'):
                for method in ('normal','critical'):
                    name=f'{mesh}_cpu_{method}'
                    if name not in names:continue
                    folder=a.output/(name+'_cflref')
                    if not (a.resume_completed and (folder/'METRICS.json').is_file()):
                        run_case(a.inputs/(name+'.json'),folder,analytic,reference_cfl=.1125)
    result=summarize(a.output);print(json.dumps(result['runs'],indent=2),flush=True)
    failed=[n+':'+k for n,r in result['runs'].items() for k,v in r['checks'].items() if not v]
    for name,row in result['comparisons'].items():
        for key in ('paired_depth_within5mm','equivalent_below1nanometre'):
            if key in row and not row[key]:failed.append(name+':'+key)
    failed.extend(name+':depth_within5mm' for name,row in result['cfl_references'].items() if not row['depth_within5mm'])
    failed.extend(name+':reference_'+key for name,row in result['cfl_references'].items()
        for key,passed in row['reference_checks'].items() if not passed)
    if failed:raise SystemExit('Canonical checks requiring review: '+', '.join(failed))


if __name__=='__main__':main()
