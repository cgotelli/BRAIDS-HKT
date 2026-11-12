"""Same-state, short-window audit; no production source or project mutation."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core import river_worker
from braids_core.sediment import adaptive_river_worker as core
from braids_core.sediment.bed_columns_v101 import host
from braids_core.river_boundary import carrier_schedules
from braids_gpu.atomic_output import write_json_atomic,save_arrays_atomic
from validation.hydraulics.merged_fwave115 import native
from validation.hydraulics.explicit_fwave124 import explicit


def read(p):return json.loads(p.read_text())


def build(source,backend,work):
    prepared=source/'prepared';project,receipt=river_worker.load_prepared(prepared)
    p,initial,t=core.load_prepared(prepared/'core/prepared')
    params=dict(slope=project['slope_parameters'],bank_survey=read(prepared/'BANK_SURVEY.json'),
        boundaries=dict(hydraulics=project['settings']['hydraulics'],states=project['boundary_states'],resolved=project['boundary_resolution']),
        sediment_schedules=carrier_schedules(project['sediment_schedules'],project['settings']['sediment_boundaries'],project['settings']['columns']),
        transport_boundaries=project['settings']['sediment_boundaries'])
    for key in ('hiding','roughness','skin_shear','wet_avalanche','van_rijn_reference','dependent_layers'):
        params[key]=project[('dependent_layer' if key=='dependent_layers' else key)+'_parameters']
    cache=core.cache_for(prepared/'core/prepared/cache',p,initial,t)
    make=core.factory(backend,p,t,cache,work,resolved_model=project['model'],process_parameters=params)
    leaves=core.initial_leaves(p,initial,t,make.required_leaves)
    return make(leaves,t.select(leaves))


def frame(source,at):
    folder=source/'run/frames';rows=read(folder/'INDEX.json')['frames']
    row=next(r for r in rows if abs(r['time_s']-at)<1e-10)
    return dict(np.load(folder/row['file']))


def state(a,q,i):
    p=a.physics;d=a.data;q=host(q);c=host(p.bedform_chezy)
    faces=np.flatnonzero((d['owner']==i)|(d['neighbour']==i))
    cf=host(p.bedform_faces)[0,faces];weights=host(p.effective_wet_weights)
    active=np.array([weights[0 if d['owner'][f]==i else 1,f] if weights.ndim==2 else weights[(0 if d['owner'][f]==i else p.nf)+f] for f in faces])
    records=[]
    for j,f in enumerate(faces):
        o,n=int(d['owner'][f]),int(d['neighbour'][f])
        records.append(dict(face=int(f),owner=o,neighbour=n,bid=int(d['bid'][f]),
            depths=[float(q[0,o]),None if n<0 else float(q[0,n])],chezy=float(cf[j]),weight=float(active[j]),
            physical_face_depth_from_manning_m=float((cf[j]*.03)**6) if cf[j]>0 else None))
    return dict(h=float(q[0,i]),velocity=(q[1:,i]/max(q[0,i],1e-30)).tolist(),chezy=float(c[i]),faces=records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resolution',choices=['1m','5m'],required=True)
    ap.add_argument('--backend',choices=['cpu','gpu'],required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--drag-control',choices=['baseline','frozen'],default='baseline')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    native.install();identity=river_worker.source_identity();base=ROOT/'validation/_explicit_fwave124'
    source=base/f'{args.resolution}_explicit_{args.backend}_cfl0.225'
    if not source.exists():source=base/f'{args.resolution}_explicit_{args.backend}'
    a=build(source,args.backend,out/'runtime');explicit.install(a.solver,.225)
    i,at=(4358,2.) if args.resolution=='1m' else (127,3.5)
    variants={'explicit225':base/f'{args.resolution}_explicit_gpu_cfl0.225',
        'fine':base/f'{args.resolution}_explicit_gpu_cfl{0.05625 if args.resolution=="1m" else 0.028125}',
        'implicit':base/f'{args.resolution}_implicit_gpu'}
    samples=[];histories={};ends={};timings={};p=a.physics;original=p.friction
    for label,folder in variants.items():
        for t in (at,at+.5):
            s=frame(folder,t);q=a.xp.asarray(s['q']);original(q,0.)
            m=a.solver.midpoint;m.interval_base=q;m.minimum_theta=1.;m.frequency_peak=0.
            rate,water=m.rate(q,1e-6,t)
            sample=dict(label=label,time=t,cell=i,rate=host(rate[:,i]).tolist(),**state(a,q,i))
            ext=a.exterior_bed
            for f in sample['faces']:
                j=f['face'];f['water_flux_m2_s']=float(water[j])
                if args.backend=='gpu':
                    b=a.hydraulics.flux.buffers
                    f['hydraulic_closed']=int(b['closed'][j])
                    f['hydraulic_local']=host(b['local'][:,j]).tolist()
                if f['bid']>=0:
                    f['boundary_trace']=host(ext.trace[:,j]).tolist();f['boundary_flux']=host(ext.flux[:,j]).tolist()
            samples.append(sample)
    # All controllers below start from the IDENTICAL saved explicit225 state.
    start=frame(variants['explicit225'],at)['q']
    physical_friction=original
    if args.drag_control=='frozen':
        physical_friction(a.xp.asarray(start),0.);fixed=p.bedform_chezy.copy()
        def fixed_friction(q,dt,*rest):
            physical_friction(q,0.,*rest)
            safe,wet=p.safe(q);xp=a.xp
            speed=xp.sqrt(safe[1]**2+safe[2]**2)/safe[0]
            safe[1:]/=1.+dt*p.parameters.gravity*speed/(fixed**2*safe[0])
            q[1:]=xp.where(wet,safe[1:],0.)
        original=fixed_friction
    for label,cfl in [('explicit225',.225),('explicit05625',.05625),('explicit0140625',.0140625),('implicit',None)]:
        if cfl is None:
            from braids_core.sediment.river_midpoint import RiverMidpoint
            m=RiverMidpoint(a.solver)
        else:m=explicit.Explicit(a.solver,cfl)
        a.solver.midpoint=m;history=[];trial=[None];original_single=m._single
        def single(q,dt,t):
            trial[0]=(float(t),float(dt));return original_single(q,dt,t)
        m._single=single
        def friction(q,dt,*rest):
            before=host(q[:,i]).copy();original(q,dt,*rest)
            history.append(dict(trial=trial[0],drag_dt=float(dt),before=before.tolist(),
                applied_chezy=float(fixed[i]) if args.drag_control=='frozen' else float(p.bedform_chezy[i]),**state(a,q,i)))
        p.friction=friction
        began=time.perf_counter();q,w,diag=m.advance(a.xp.asarray(start),.5,at)
        if args.backend=='gpu':a.xp.cuda.get_current_stream().synchronize()
        timings[label]=dict(wall_s=time.perf_counter()-began,diagnostics=diag)
        ends[label]=host(q);histories[label]=history
        p.friction=original
    save_arrays_atomic(out/'ENDS.npz',initial=start,**ends)
    write_json_atomic(out/'AUDIT.json',dict(cell=i,xy=a.data['xy'][:,i].tolist(),area=float(a.data['area'][i]),
        backend=args.backend,resolution=args.resolution,drag_control=args.drag_control,start_s=at,end_s=at+.5,
        samples=samples,histories=histories,timings=timings,runtime_before=identity,runtime_after=river_worker.source_identity(),
        note='Same-state diagnostic continuations, not public authenticated restarts; all other physics disabled. Frozen drag is a causal experiment, not a proposed physical model.'))
    print(json.dumps(dict(output=str(out),timings=timings)))


if __name__=='__main__':main()
