"""Longer, independently adaptive GPU hydraulics against frozen alpha82.

Research only: immutable FP64 augmented f-wave kernels and SSPRK43 equations.
The optional graph variant changes four scalar dt arguments into a resident
device parameter. It does not recapture per step, change CFL, or copy fields
between stages. Native-compatible retry/growth/output clipping remains on CPU.
Timing with the ongoing CPU bend is exploratory, never an isolated speed claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from . import hydraulic_pipeline as hp
from . import hydraulic_reference as hr
from . import reconstruction_oracle as oracle
from . import resident_update as ru
from .graph_execution import validate_graph_scope

HERE=Path(__file__).resolve().parent
TAG='gpu_adaptive_hydraulics_20261001_v1'
PIPELINE_TAG='gpu_hydraulic_20261001_v6'
CONTROLS={
    'bend_small':dict(end=20.,dt=.05,output=1.),
    'bend_large':dict(end=20.,dt=.05,output=1.),
    'cutcell_lake':dict(end=1.,dt=.05,output=.1),
    'dry_dam_break':dict(end=1.,dt=.05,output=.1),
}
DT_KERNELS={
    'resident_update.cu':{'resident_draining':(10,.5),'resident_gather_update':(14,.5)},
    'boundaries.cu':{'gpu_boundary_ledger':(5,.5)},
    'time_integration.cu':{'ti_manning':(3,1.)},
}


def next_hint(dt,peak,target,dtmax,*,scheduled_clip=False,first_attempt=True,unclipped=None):
    """Same accepted-step rule as frozen run_cutcell; Euler-stage CFL target."""
    if scheduled_clip and first_attempt:
        if unclipped is None or not np.isfinite(unclipped) or unclipped<=0:raise ValueError('Missing unclipped hint')
        return min(dtmax,unclipped)
    factor=max(.5,min(1.05,.95*target/peak)) if peak>0 else 1.05
    return min(dtmax,factor*dt)


def retry_step(dt,peak,target):
    """Native CFL retry when measured too large, otherwise positivity halving."""
    factor=max(.1,min(.95,.95*target/peak)) if np.isfinite(peak) and peak>target else .5
    return factor*dt


def transform_dt_arguments(source,kernels):
    """Change only the source of dt, retaining the complete numerical body.

    Existing immutable source is never edited. Exact declaration and argument
    position checks make unexpected upstream changes fail closed.
    """
    for name,(position,factor) in kernels.items():
        pattern=r'(extern "C" __global__ void '+name+r'\s*\()(.*?)(\)\s*\{)'
        matches=list(re.finditer(pattern,source,re.S))
        if len(matches)!=1:raise ValueError('Expected one kernel declaration: '+name)
        match=matches[0];arguments=match.group(2).split(',')
        if arguments[position].strip()!='double dt':raise ValueError('Unexpected dt ABI: '+name)
        arguments[position]='const double* adaptive_dt'
        header=match.group(1)+','.join(arguments)+match.group(3)
        header+='\n    const double dt='+('0.5*' if factor==.5 else '')+'adaptive_dt[0];'
        source=source[:match.start()]+header+source[match.end():]
    return source


class _DynamicArgument:
    def __init__(self,kernel,position,parameter):
        self.kernel,self.position,self.parameter=kernel,position,parameter

    def __call__(self,grid,block,args,**kwargs):
        values=list(args);values[self.position]=self.parameter
        return self.kernel(grid,block,tuple(values),**kwargs)


class AdaptiveGraph:
    """One graph capture for arbitrary positive dt and constant forcing.

    Four cloned kernel declarations consume the same device dt. Their bodies
    are unchanged, and Euler calls read half dt while Manning reads full dt.
    Retained private pool prevents temporary reduction addresses being reused.
    The owned pipeline may not be used concurrently or externally mutated.
    """
    def __init__(self,pipeline,generated_source,*,warm_dt=.0001):
        import cupy as cp
        validate_graph_scope(pipeline.data,warm_dt,pipeline.reconstruction.outlet_extension)
        self.cp=cp;self.pipeline=pipeline;self.stream=cp.cuda.Stream(non_blocking=True)
        self.pool=cp.cuda.MemoryPool();self.parameter=cp.asarray([warm_dt],dtype=cp.float64)
        self.modules=[];generated_source=Path(generated_source)
        for filename,description in DT_KERNELS.items():
            code=(generated_source/filename).read_text()
            if filename=='resident_update.cu':code=(pipeline.folder/'augmented_fwave.cu').read_text()+'\n'+code
            module=cp.RawModule(code=code,options=ru.OPTIONS,name_expressions=tuple(description))
            target=pipeline.flux if filename=='resident_update.cu' else pipeline.boundary if filename=='boundaries.cu' else pipeline.stepper
            for name,(position,_) in description.items():
                target.kernels[name]=_DynamicArgument(module.get_function(name),position,self.parameter)
            self.modules.append(module)
        cp.cuda.get_current_stream().synchronize();started=time.perf_counter();calls=pipeline.euler_calls
        with self.stream,cp.cuda.using_allocator(self.pool.malloc):
            for _ in range(2):pipeline.step(warm_dt,0.,commit=False)
        self.stream.synchronize()
        try:
            with self.stream,cp.cuda.using_allocator(self.pool.malloc):
                self.stream.begin_capture()
                self.result=pipeline.step(warm_dt,0.)
                self.graph=self.stream.end_capture()
        except Exception:
            if self.stream.is_capturing():self.stream.end_capture()
            raise
        finally:pipeline.euler_calls=calls
        self.graph.upload(stream=self.stream);self.stream.synchronize()
        self.setup_s=time.perf_counter()-started;self.replays=0
        self.pointers=(pipeline.q.data.ptr,pipeline.ledger.data.ptr)

    def step(self,dt,current):
        if not np.isfinite([dt,current]).all() or dt<=0:raise ValueError('Finite positive dt and finite time required')
        if (self.pipeline.q.data.ptr,self.pipeline.ledger.data.ptr)!=self.pointers:
            raise ValueError('Accepted array pointers changed after graph capture')
        with self.stream:
            self.parameter.fill(float(dt))
            self.graph.launch(stream=self.stream)
            code=int(self.result['status'].item())
            peak=float(self.result['cfl'].item())
        self.replays+=1;self.pipeline.euler_calls+=4
        return code,peak


def freeze(tag=TAG,pipeline_tag=PIPELINE_TAG):
    root,report=ru.paths(tag);root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    source=HERE/'runs'/pipeline_tag/'source';hp.verify_frozen(source)
    generated=root/'adaptive_source';generated.mkdir()
    for filename,description in DT_KERNELS.items():
        (generated/filename).write_text(transform_dt_arguments((source/filename).read_text(),description))
    shutil.copy2(__file__,root/'adaptive_hydraulics.py')
    identity=oracle.verify_parent()
    record=dict(driver_sha256=ru.sha(__file__),pipeline_tag=pipeline_tag,
        pipeline_manifest_sha256=ru.sha(source/'pipeline_build.json'),
        cpu_runtime_path=str(oracle.PARENT),cpu_executable_sha256=identity['executable_sha256'],
        generated_sources={p.name:ru.sha(p) for p in generated.iterdir()},
        transform='Four dt scalar arguments become one device full-dt parameter; equations unchanged',
        controls=CONTROLS,precision='FP64',cuda_options=ru.OPTIONS,
        user_CFL_preserved=True,ongoing_CPU_job_untouched=True,
        timing='Exploratory under concurrent eight-thread mobile CPU bend; no isolated CPU/GPU speed claim',
        acceptance=dict(depth_rms_m=2e-8,velocity_rms_m_s=2e-7,water_residual_m3=1e-8,boundary_ledger_m3=1e-8),
        full_coupled_simulation=False,production_modified=False,delft_modified=False)
    ru.write(root/'build.json',record);ru.write(report/'plan.json',record)
    return record


def verify(tag):
    root,_=ru.paths(tag);record=json.loads((root/'build.json').read_text())
    if ru.sha(__file__)!=record['driver_sha256']:raise ValueError('Adaptive driver changed; freeze a new qualification')
    source=HERE/'runs'/record['pipeline_tag']/'source';hp.verify_frozen(source)
    if ru.sha(source/'pipeline_build.json')!=record['pipeline_manifest_sha256']:raise ValueError('Pipeline identity changed')
    for name,digest in record['generated_sources'].items():
        if ru.sha(root/'adaptive_source'/name)!=digest:raise ValueError('Adaptive CUDA source changed')
    return root,record,source


def run_cpu(case,integrator='ssp43',tag=TAG):
    """Unmodified native runtime; only validation time controls are changed."""
    from braids_support.sediment.lfm_bend_2d import run_case as reader
    root,record,_=verify(tag);identity=oracle.verify_parent();controls=CONTROLS[case]
    source=HERE/'runs'/oracle.TAG/case;initial=hp.load_snapshot(source)
    target=root/('cpu_'+case+'_'+integrator);target.mkdir(exist_ok=False)
    for name in ('case.data.bin','geometry.npz','analysis_geometry.npz'):
        if (source/name).exists():shutil.copy2(source/name,target/name)
    deck,changes=hr.transform_deck((source/'case.data').read_text(),**controls)
    (target/'case.data').write_text(deck)
    provenance=dict(case=case,integrator_requested=integrator,threads=8,changes=changes,
        cfl_target=float(initial['cfl_target']),input_origin=str(source),
        origin_sha256=ru.sha(source/'input_provenance.json'),
        input_files={p.name:ru.sha(p) for p in target.iterdir() if p.is_file()},
        runtime_path=str(oracle.PARENT),executable_sha256=identity['executable_sha256'],
        source_sha256=identity['source_sha256'],frozen_source_modified=False,delft_modified=False,
        isolated_performance_measurement=False,concurrent_CPU_job=True)
    ru.write(target/'input_provenance.json',provenance)
    environment={k:v for k,v in os.environ.items() if not k.startswith('BRAIDS_')}
    environment.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
        BRAIDS_TIME_INTEGRATOR=integrator,BRAIDS_PROFILE='1')
    print(json.dumps(dict(event='cpu_start',case=case,integrator=integrator,controls=controls)),flush=True)
    started=time.perf_counter()
    with (target/'native.log').open('w') as log:
        result=subprocess.run([str(oracle.PARENT/'xbraids_candidate.exe'),'case.data'],cwd=target,
            env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
    wall=time.perf_counter()-started;result.check_returncode()
    frames_path=reader.shared.native_frames_path(target);frames=reader.shared.read_frames(frames_path)
    times=np.array([f['time_s'] for f in frames]);states=np.stack([f['q'][:,:3].T for f in frames])
    ledgers=np.stack([f['ledger'][:,:,] for f in frames]);steps=np.array([f['steps'] for f in frames])
    np.testing.assert_array_equal(states[0],initial['q'])
    np.testing.assert_allclose(times,np.arange(round(controls['end']/controls['output'])+1)*controls['output'],atol=2e-12,rtol=0)
    water=(states[:,0]-states[0,0])@initial['area']-(ledgers[:,:,0]-ledgers[0,:,0]).sum(axis=1)
    np.savez_compressed(target/'trajectory.npz',times=times,q=states,ledger=ledgers[:,:,0],steps=steps,water_residual=water)
    text=(target/'native.log').read_text();counts=re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)',text)
    measured=re.search(r'profile: measured run wall seconds=\s*([\d.]+)',text)
    receipt=dict(case=case,integrator_requested=integrator,final_time_s=float(times[-1]),steps=int(steps[-1]),
        rejected_trials=int(counts.group(2)) if counts else None,cfl_target=float(initial['cfl_target']),
        controls=controls,wall_s_including_process=wall,native_profile_wall_s=float(measured.group(1)) if measured else None,
        arrays_sha256=ru.sha(target/'trajectory.npz'),frames_sha256=ru.sha(frames_path),
        log_sha256=ru.sha(target/'native.log'),input_provenance_sha256=ru.sha(target/'input_provenance.json'),
        runtime_sha256=identity['executable_sha256'],min_depth_m=float(states[:,0].min()),
        water_residual_m3=float(abs(water).max()),finite=bool(np.isfinite(states).all()),
        q_layout='time,component,cell',ledger_layout='time,boundary',isolated_timing=False)
    ru.write(target/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    return receipt


def validate_graph(tag=TAG):
    """Changing dt, output clipping and deliberately rejected graph trials."""
    root,_,source=verify(tag);rows=[]
    for case in ('bend_small','dry_dam_break','cutcell_lake'):
        data=hp.load_snapshot(HERE/'runs'/oracle.TAG/case)
        plain=hp.HydraulicPipeline(data,source,outlet_extension=0)
        captured=hp.HydraulicPipeline(data,source,outlet_extension=0)
        graph=AdaptiveGraph(captured,root/'adaptive_source');now=0.
        sequence=(.0001,.0003,.00015,.0008,10.,.0002,.0007,.00001)
        checks=[]
        for dt in sequence:
            before_q,before_l=captured.fields()
            result=plain.step(dt,now);code=int(result['status'].item());peak=float(result['cfl'].item())
            other_code,other_peak=graph.step(dt,now)
            q1,l1=plain.fields();q2,l2=captured.fields()
            equal=np.array_equal(q1,q2) and np.array_equal(l1,l2)
            rollback=bool(np.array_equal(q2,before_q) and np.array_equal(l2,before_l)) if code else None
            passed=equal and code==other_code and peak==other_peak and (not code or rollback)
            checks.append(dict(dt=dt,status=code,cfl=peak,bitwise_equal=equal,rejected_rollback=rollback,passed=bool(passed)))
            if not code:now+=dt
        rows.append(dict(case=case,checks=checks,passed=all(c['passed'] for c in checks),captures=1,replays=graph.replays))
    result=dict(rows=rows,all_passed=all(row['passed'] for row in rows),no_per_step_graph_recapture=True)
    _,report=ru.paths(tag);ru.write(report/'graph_braids_support.json',result)
    print(json.dumps(result),flush=True)
    if not result['all_passed']:raise RuntimeError('Adaptive graph qualification failed')
    return result


def run_gpu(case,tag=TAG,mode='graph'):
    root,record,source=verify(tag);controls=CONTROLS[case]
    folder=root/('cpu_'+case+'_ssp43');receipt=json.loads((folder/'receipt.json').read_text())
    if ru.sha(folder/'trajectory.npz')!=receipt['arrays_sha256']:raise ValueError('CPU trajectory changed')
    cpu=np.load(folder/'trajectory.npz');data=hp.load_snapshot(HERE/'runs'/oracle.TAG/case)
    before_setup=time.perf_counter();pipeline=hp.HydraulicPipeline(data,source,outlet_extension=0)
    graph=AdaptiveGraph(pipeline,root/'adaptive_source') if mode=='graph' else None
    if mode=='plain':
        pipeline.step(.0001,0.,commit=False);pipeline.cp.cuda.Stream.null.synchronize()
    setup=time.perf_counter()-before_setup;dtmax=controls['dt'];hint=dtmax;current=0.;steps=0;rejected=0
    history=[];states=[];ledgers=[];frame_steps=[];output_seconds=0.;first_rejection=None
    def trial(dt,current):
        if graph:return graph.step(dt,current)
        result=pipeline.step(dt,current);return int(result['status'].item()),float(result['cfl'].item())
    started=time.perf_counter()
    print(json.dumps(dict(event='gpu_start',case=case,mode=mode,cells=pipeline.nc,setup_s=setup)),flush=True)
    for target in cpu['times']:
        while current<target-1e-12:
            unclipped=hint;dt=min(hint,float(target)-current);clipped=dt<unclipped
            for attempt in range(1,36):
                if current+dt<=current:raise RuntimeError('Nonadvancing adaptive step')
                code,peak=trial(dt,current)
                history.append((current,dt,peak,code))
                if not code:break
                rejected+=1
                if first_rejection is None:first_rejection=dict(time_s=current,dt_s=dt,cfl=peak)
                dt=retry_step(dt,peak,data['cfl_target'])
            else:raise RuntimeError(f'{case}: failed 35 adaptive trials at {current}')
            hint=next_hint(dt,peak,data['cfl_target'],dtmax,scheduled_clip=clipped,first_attempt=attempt==1,unclipped=unclipped)
            current+=dt;steps+=1
        output_started=time.perf_counter();q,l=pipeline.fields();states.append(q);ledgers.append(l);frame_steps.append(steps)
        output_seconds+=time.perf_counter()-output_started
        print(json.dumps(dict(event='gpu_output',case=case,time_s=current,steps=steps,rejected=rejected)),flush=True)
    wall=time.perf_counter()-started;states=np.array(states);ledgers=np.array(ledgers);history=np.array(history)
    difference=states-cpu['q'];weights=data['area']/data['area'].sum()
    depth_rms=np.sqrt((difference[:,0]**2*weights).sum(axis=1))
    velocity=states[:,1:]/np.maximum(states[:,:1],data['dry']);reference_v=cpu['q'][:,1:]/np.maximum(cpu['q'][:,:1],data['dry'])
    velocity_rms=np.sqrt((np.sum((velocity-reference_v)**2,axis=1)*weights).sum(axis=1))
    water=(states[:,0]-states[0,0])@data['area']-ledgers.sum(axis=1)
    ledger_error=abs(ledgers-cpu['ledger']);accepted=history[:,3]==0
    checks=dict(case=case,mode=mode,cells=pipeline.nc,faces=pipeline.nf,end_s=float(cpu['times'][-1]),
        steps=steps,rejected_trials=rejected,CPU_steps=int(cpu['steps'][-1]),CPU_rejected_trials=receipt['rejected_trials'],
        cfl_target=float(data['cfl_target']),max_accepted_stage_cfl=float(history[accepted,2].max()),
        accepted_dt_min_s=float(history[accepted,1].min()),accepted_dt_max_s=float(history[accepted,1].max()),
        depth_rms_m=float(depth_rms.max()),velocity_rms_m_s=float(velocity_rms.max()),
        max_state_difference=float(abs(difference).max()),water_residual_m3=float(abs(water).max()),
        boundary_ledger_difference_m3=float(ledger_error.max()) if ledger_error.size else 0.,
        min_depth_m=float(states[:,0].min()),finite=bool(np.isfinite(states).all()),
        first_rejection=first_rejection,independent_native_compatible_adaptation=True,
        exact_CPU_step_schedule_replayed=False,frame_step_counts_equal=bool(np.array_equal(frame_steps,cpu['steps'])),
        wall_evolution_s=wall,setup_JIT_graph_upload_s=setup,output_download_s=output_seconds,
        CPU_ssp43_process_wall_s=receipt['wall_s_including_process'],isolated_timing=False,
        CPU_background_and_other_load='Concurrent mobile bend job; exploratory wall times, not isolated speedup',
        graph_captures=1 if graph else 0,graph_replays=graph.replays if graph else 0,
        field_transfers_per_Euler_stage=0,control_scalar_reads_per_trial=2,
        output_times_exact=True,CPU_reference_sha256=receipt['arrays_sha256'])
    checks['passed']=bool(checks['depth_rms_m']<=2e-8 and checks['velocity_rms_m_s']<=2e-7 and
        checks['water_residual_m3']<=1e-8 and checks['boundary_ledger_difference_m3']<=1e-8 and checks['min_depth_m']>=0 and checks['finite'])
    target=root/('gpu_'+case+'_'+mode);target.mkdir(exist_ok=False)
    np.savez_compressed(target/'trajectory.npz',q=states,ledger=ledgers,times=cpu['times'],steps=frame_steps,
        history=history,water_residual=water,depth_rms=depth_rms,velocity_rms=velocity_rms)
    checks['arrays_sha256']=ru.sha(target/'trajectory.npz');ru.write(target/'receipt.json',checks)
    print(json.dumps(checks),flush=True)
    if not checks['passed']:raise RuntimeError('Adaptive trajectory failed; inspect '+str(target))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','cpu','graph','gpu'))
    parser.add_argument('--tag',default=TAG);parser.add_argument('--pipeline-tag',default=PIPELINE_TAG)
    parser.add_argument('--cases',nargs='+',choices=tuple(CONTROLS),default=tuple(CONTROLS))
    parser.add_argument('--integrator',choices=('ssp43','auto'),default='ssp43')
    parser.add_argument('--mode',choices=('graph','plain'),default='graph');args=parser.parse_args()
    os.environ.setdefault('CUPY_CACHE_DIR',str(HERE/'.cupy_cache'))
    if args.action=='freeze':freeze(args.tag,args.pipeline_tag)
    elif args.action=='graph':validate_graph(args.tag)
    elif args.action=='cpu':
        for case in args.cases:run_cpu(case,args.integrator,args.tag)
    else:
        for case in args.cases:run_gpu(case,args.tag,args.mode)


if __name__=='__main__':main()
