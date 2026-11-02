"""Archive-bound GPU runner for the four G3 live-bank engineering fixtures."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from . import live_bank
from .runner import (a1_auxiliary_device_arrays, a1_checkpoint_extra_schema,
    a1_event_cursor, bank_toe_persistent_arrays, distributed_bank_persistent_arrays,
    fields_digest, live_internal_knots, reject_deferred_public_controls,
    save_arrays, sha, time_controls, validate_bank_toe_checkpoint,
    validate_checkpoint, validate_distributed_bank_checkpoint, write_json)


def _method(variant, case, source_identity, input_identity, events, dtmax):
    return dict(schema=1, variant=variant, case=case, backend='gpu',
        integrator='SSPRK33', stages=3, stage_slots=[0, 2, 1],
        launch_mode='plain', precision='FP64', cfl=.4, dtmax_s=dtmax,
        bedlaw='graded_MPM_Exner', bank_mode=live_bank.VARIANTS[variant],
        bank_failure=True, post_rk_order=['bank_toe_or_distributed_failure',
            'general_wet_dry_avalanche'], avalanche_enabled=case.endswith('_on'),
        secondary='prognostic_model2', live_boundaries='normal9_critical4',
        exchange=0, input_identity=input_identity, source_identity=source_identity,
        event_identity=events.identity, gate_sha256=live_bank.GATE_SHA,
        transaction='segmented_all_reachable_v5', auxiliary_checkpoint_schema=4)


def _archive(data, input_path, variant, folder):
    """Freeze CUDA, generated bridge and every imported Python source before evolution."""
    from . import a1_full_array_transaction as transaction
    from braids_support.gpu_fwave import a1_combined_current_adapter as ancestor
    folder=Path(folder).resolve()
    with live_bank.isolated_adapter(folder=input_path,variant=variant) as candidate:
        candidate.prepare_sources(folder,data)
        candidate.verify_sources(folder)
    python_sources={}
    g3_inputs=('combined_export.py','native_case.py','ablation_case.py')
    source_lists=(sorted(Path(__file__).parent.glob('*.py')),
        [live_bank.ROOT/'braids_support/g3_live_bank'/name for name in g3_inputs],
        sorted((live_bank.ROOT/'braids_support/gpu_fwave').glob('*.py')))
    for sources in source_lists:
        for source in sources:
            relative=str(source.relative_to(live_bank.ROOT)).replace('\\','/')
            copied=folder/'python_snapshot'/relative
            copied.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,copied)
            python_sources[relative]=sha(source)
            if sha(copied)!=python_sources[relative]:
                raise RuntimeError('G3 live-bank Python source changed during archive')
    python_sources['adapter']=sha(live_bank.__file__)
    python_sources['runner']=sha(__file__)
    python_sources['transaction']=sha(transaction.__file__)
    python_sources['ancestor']=sha(ancestor.__file__)
    artifact_hashes={str(path.relative_to(folder)).replace('\\','/'):sha(path)
        for path in folder.rglob('*') if path.is_file()}
    receipt=dict(schema=1,variant=variant,gate_sha256=live_bank.GATE_SHA,
        prepared_manifest_sha256=live_bank.MANIFEST_SHA,
        input_case=Path(input_path).name,source_files=artifact_hashes,
        live_python_sha256=python_sources,
        candidate_receipt_sha256=sha(folder/'candidate_sources.json'),
        python_executable=str(Path(os.sys.executable).resolve()),
        python_executable_sha256=sha(os.sys.executable),
        numerical_cuda_unchanged=True,Delft3D_source_modified=False)
    write_json(folder/'runtime_manifest.json',receipt)
    return sha(folder/'runtime_manifest.json')


def _checkpoint_schema(data,events,mode):
    nc=int(data['nc']);nf=int(data['nf']);ng=int(data['nfrac']);nb=int(data['nb'])
    knots=live_internal_knots(events,0.,1.)
    state={'q':np.asarray(data['q']).shape,
        'layers':np.asarray(data['layer_mass']).shape,'secondary':(nc,)}
    ledger={'water':(nb,),'bed':(ng,nb),'secondary':(nb,),'source':()}
    extras=a1_checkpoint_extra_schema(data,len(knots))
    extras.update({f'bank_history_{name}':((nc,),np.int32 if name=='collapse_count'
        else np.float64) for name in live_bank.HISTORY})
    if mode==2:
        extras.update(bank_toe_face_signed=((ng,nf),np.float64),
            bank_toe_face_gross=((ng,nf),np.float64))
    expected=set(state)|{'ledger_'+key for key in ledger}|set(extras)|{
        'time','dt_hint','steps','rejected','input_identity','source_identity',
        'method_identity','event_identity','next_boundary_knot','a1_payload_sha256'}
    return state,ledger,extras,expected,knots


def _restore_preflight(resume,data,events,*,input_identity,source_identity,
                       method_identity,start,end,mode):
    from braids_support.gpu_fwave.boundary_schedule import evaluate_host,validate_tables
    state,ledger,extras,expected,knots=_checkpoint_schema(data,events,mode)
    with np.load(resume,allow_pickle=False) as checkpoint:
        if set(checkpoint.files)!=expected:
            raise ValueError('G3 live-bank restart differs from complete checkpoint schema')
        payload=np.asarray(checkpoint['a1_payload_sha256'])
        if (payload.shape!=() or payload.dtype.kind not in 'US' or
                payload.item()!=fields_digest({key:checkpoint[key] for key in checkpoint.files
                    if key!='a1_payload_sha256'})):
            raise ValueError('G3 live-bank checkpoint payload digest differs')
        arrays,scalars=validate_checkpoint(checkpoint,state,ledger,
            input_identity=input_identity,source_identity=source_identity,
            method_identity=method_identity,events=events,secondary_model=2,
            start=start,end=end,extra_schema=extras,
            allow_nan_extras=('a1_outlet_failed_speed',))
    if mode==2:validate_bank_toe_checkpoint(arrays,data['is_toe_face'])
    else:validate_distributed_bank_checkpoint(arrays)
    if int(arrays['a1_event_cursor'])!=a1_event_cursor(events,scalars['time']):
        raise ValueError('G3 live-bank restart event cursor differs')
    if not np.array_equal(arrays['a1_knot_hits'],
                          (knots<=float(scalars['time'])+1.e-12).astype(np.int32)):
        raise ValueError('G3 live-bank forcing-knot history differs')
    if (not np.isin(arrays['a1_normal_cache_valid'],(0,1)).all() or
            not np.array_equal(arrays['a1_normal_recorded'],
                               arrays['a1_outlet_recorded']) or
            np.any(arrays['a1_normal_latch']) or np.any(arrays['a1_outlet_latch']) or
            np.any(arrays['a1_schedule_status']) or
            np.any(arrays['a1_normal_root_status']) or
            np.any(arrays['a1_normal_trace_status']) or
            np.any(arrays['a1_outlet_status']) or
            np.any(arrays['a1_normal_failure_latch']) or
            np.any(arrays['a1_outlet_failure_latch']) or
            np.any(arrays['a1_outlet_face_status'])):
        raise ValueError('G3 live-bank accepted boundary diagnostics are invalid')
    if scalars['steps']:
        clock=arrays['a1_schedule_clock'];stages=arrays['a1_schedule_stage_time']
        if (clock[1]<=0 or abs(clock[0]+clock[1]-scalars['time'])>1.e-12 or
                not np.array_equal(arrays['a1_normal_recorded'],[1,1,1,0])):
            raise ValueError('G3 live-bank restart SSPRK33 stage records differ')
        tables=validate_tables(dict(data,section_control=np.zeros(int(data['nb']),np.int32),
            sampled=np.zeros(int(data['nf']),np.int32)))
        scheduled=evaluate_host(tables,float(clock[0]),float(clock[1]),validated=True)
        for key in ('stage_time','hydraulic','concentration','feed'):
            if not np.allclose(arrays['a1_schedule_'+key],scheduled[key],
                               rtol=0,atol=1.e-12):
                raise ValueError('G3 live-bank restart schedule differs: '+key)
        if not np.allclose(stages,scheduled['stage_time'],rtol=0,atol=1.e-12):
            raise ValueError('G3 live-bank resident stage clock differs')
    elif np.any(arrays['a1_normal_recorded']) or np.any(arrays['a1_outlet_recorded']):
        raise ValueError('G3 live-bank initial restart has nonempty section records')
    return arrays,scalars


def run_live_bank(*,input_path,output_path,until=None,output_every=None,dt_max=None,
                  max_steps=None,stop_file=None,resume=None,launch_mode='graph',
                  integrator='ssp43',capture_stages=False,capture_step_trace=False,
                  cancelled=lambda:False,numerical_variant):
    """Run one authenticated live-bank case; never silently use CPU or graph."""
    from braids_support.gpu_fwave.adaptive_hydraulics import next_hint,retry_step
    if capture_stages or capture_step_trace:
        raise ValueError('G3 live-bank public stage/step trace is not yet qualified')
    if max_steps is not None and (type(max_steps) is not int or max_steps<1):
        raise ValueError('G3 live-bank max_steps must be positive integer')
    folder=Path(input_path).resolve();destination=Path(output_path).resolve()
    if destination.exists():raise FileExistsError('GPU output folder already exists')
    data,input_identity=live_bank.read_case(folder,numerical_variant)
    reject_deferred_public_controls(data)
    start,end,interval,dtmax=time_controls(data,until,output_every,dt_max)
    events=live_bank.validate_scope(data,folder=folder,variant=numerical_variant,
        start=start,end=end,interval=interval,dtmax=dtmax,
        launch_mode=launch_mode,integrator=integrator)
    mode=live_bank.VARIANTS[numerical_variant]
    with tempfile.TemporaryDirectory(prefix='braids_g3_live_bank_source_') as temporary:
        archive=Path(temporary)/'source'
        source_identity=_archive(data,folder,numerical_variant,archive)
        method=_method(numerical_variant,folder.name,source_identity,input_identity,events,dtmax)
        method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        restored=(_restore_preflight(resume,data,events,input_identity=input_identity,
            source_identity=source_identity,method_identity=method_identity,
            start=start,end=end,mode=mode) if resume is not None else None)
        destination.mkdir(parents=True,exist_ok=False)
        source_folder=destination/'source';shutil.copytree(archive,source_folder)
        if sha(source_folder/'runtime_manifest.json')!=source_identity:
            raise RuntimeError('G3 live-bank source changed during archive')
    manifest=dict(schema=1,backend='gpu',backend_selected_by='user',
        automatic_backend_selection=False,variant=numerical_variant,
        case=folder.name,input_folder=str(folder),input_identity=input_identity,
        source_identity=source_identity,method_identity=method_identity,
        method=method,gate_sha256=live_bank.GATE_SHA,
        preparation_manifest_sha256=live_bank.MANIFEST_SHA,
        numerical_CUDA_unchanged=True,physical_validation=False,
        Delft3D_source_modified=False,QGIS_release=False,
        frames=[],status='initializing',start_s=start,end_s=end,
        output_every_s=interval,dtmax_s=dtmax)
    write_json(destination/'run.json',manifest)
    began=time.perf_counter();current=start;steps=0;rejected=0;hint=dtmax
    frame_index=0;status='initializing';solver=None;auxiliary={};bank={}
    # Keep the full 1 s forcing-knot history even for a 0.4 s prefix.  The
    # resumed process must carry exactly the same cursor/hits as a continuous
    # run at that accepted clock.
    knots=live_internal_knots(events,start,1.)
    knot_hits=np.zeros(len(knots),dtype=np.int32)
    event_rows=[];trial_rows=[];last_progress=began
    stack=ExitStack()

    def progress(event):
        row=dict(event=event,status=status,backend='gpu',time_s=current,
            end_s=end,steps=steps,rejected=rejected,
            percent=(100. if status=='completed' else
                100.*(current-start)/(end-start)),
            wall_s=time.perf_counter()-began)
        write_json(destination/'status.json',row)
        print(json.dumps(row,allow_nan=False),flush=True)

    def output(frame=True):
        nonlocal frame_index
        fields=solver.fields()
        checkpoint=dict(fields,time=np.float64(current),steps=np.int64(steps),
            rejected=np.int64(rejected),dt_hint=np.float64(hint),
            input_identity=np.asarray(input_identity),
            source_identity=np.asarray(source_identity),
            method_identity=np.asarray(method_identity),
            event_identity=np.asarray(events.identity),
            next_boundary_knot=np.float64(events.next_after(current)),
            a1_event_cursor=np.int64(a1_event_cursor(events,current)),
            a1_knot_hits=knot_hits.copy())
        checkpoint.update({key:solver.cp.asnumpy(value)
            for key,value in auxiliary.items()})
        checkpoint['a1_payload_sha256']=np.asarray(fields_digest(checkpoint))
        save_arrays(destination/'checkpoint.npz',**checkpoint)
        if frame:
            name=f'frame_{frame_index:06d}.npz'
            save_arrays(destination/name,**checkpoint)
            manifest['frames'].append(dict(file=name,time_s=current,steps=steps,
                sha256=sha(destination/name)))
            write_json(destination/'run.json',manifest)
            frame_index+=1

    try:
        progress('initializing')
        Solver=stack.enter_context(live_bank.constructors(source_folder,data,
            input_folder=folder,variant=numerical_variant))
        solver=Solver(data,source_folder=source_folder)
        auxiliary=a1_auxiliary_device_arrays(solver)
        schema=_checkpoint_schema(data,events,mode)[2]
        expected_aux=set(schema)-{'bed_elevation','a1_event_cursor','a1_knot_hits'}
        expected_aux-={f'bank_history_{name}' for name in live_bank.HISTORY}
        if mode==2:expected_aux-={'bank_toe_face_signed','bank_toe_face_gross'}
        if set(auxiliary)!=expected_aux or any(value.shape!=schema[key][0] or
            value.dtype!=np.dtype(schema[key][1]) for key,value in auxiliary.items()):
            raise RuntimeError('G3 live-bank resident boundary checkpoint schema differs')
        bank=(bank_toe_persistent_arrays(solver) if mode==2 else
              distributed_bank_persistent_arrays(solver))
        if len(bank)!=(9 if mode==2 else 7):
            raise RuntimeError('G3 live-bank persistent bank history schema differs')
        if restored is not None:
            arrays,scalars=restored
            for key,value in solver.state.items():value.set(arrays[key])
            for key,value in solver.ledger.items():value.set(arrays['ledger_'+key])
            for key,value in auxiliary.items():value.set(arrays[key])
            for key,value in bank.items():value.set(arrays[key])
            current=float(scalars['time']);steps=int(scalars['steps'])
            rejected=int(scalars['rejected']);hint=min(float(scalars['dt_hint']),dtmax)
            knot_hits[:]=arrays['a1_knot_hits']
            if not np.array_equal(solver.fields()['bed_elevation'],
                                  arrays['bed_elevation']):
                raise ValueError('G3 live-bank resumed bed differs from accepted layers')
        cp=solver.cp;device=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        if isinstance(device,bytes):device=device.decode()
        manifest.update(device=str(device),cupy=cp.__version__,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
            python_executable=str(Path(os.sys.executable).resolve()),
            python_executable_sha256=sha(os.sys.executable),
            process_id=os.getpid(),cells=solver.nc,faces=solver.nf,
            fractions=solver.ng)
        save_arrays(destination/'geometry.npz',xy=data['xy'],area=data['area'],
            mid=data['mid'],owner=data['owner'],neighbour=data['neighbour'],
            normal=data['normal'],length=data['length'],rock=data['rock'])
        status='running';output();progress('started')
        initial_steps=steps;initial_rejected=rejected;initial_euler=solver.euler_calls
        target_index=math.floor((current-start)/interval+1.e-10)+1
        target=min(end,start+target_index*interval)
        while current<end-1.e-12:
            if cancelled() or (stop_file is not None and Path(stop_file).exists()):
                status='cancelled';break
            if max_steps is not None and steps>=max_steps:
                status='step_limit';break
            clock=events.trial(current,hint,output=target,end=end)
            dt=clock.dt
            for attempt in range(1,36):
                if not math.isfinite(dt) or current+dt<=current:
                    raise RuntimeError('G3 live-bank nonadvancing adaptive timestep')
                result=solver.step(dt,current)
                code=int(result['status'].item());peak=float(result['cfl'].item())
                trial_rows.append(dict(step=steps+1,attempt=attempt,time_s=current,
                    dt_s=dt,status=code,cfl=peak,accepted=code==0,
                    avalanche_substeps=list(result['avalanche_substeps'])))
                if code==0:break
                rejected+=1;dt=retry_step(dt,peak,.4)
            else:raise RuntimeError(f'35 rejected G3 live-bank trials at {current:g} s')
            hint=next_hint(dt,peak,.4,dtmax,
                scheduled_clip=clock.scheduled_clip,first_attempt=attempt==1,
                unclipped=clock.unclipped_hint)
            steps+=1;current=clock.accepted_time(dt)
            if (start<clock.boundary_knot<=end and
                    clock.boundary_knot<1.-1.e-12 and
                    abs(current-clock.boundary_knot)<=1.e-12):
                event_rows.append(dict(step=steps,time_s=current,knot_s=clock.boundary_knot))
                index=np.flatnonzero(np.isclose(knots,clock.boundary_knot,rtol=0,atol=1.e-12))
                if len(index)!=1:raise RuntimeError('Unexpected G3 live-bank knot hit')
                knot_hits[int(index[0])]=1
            if current>=target-1.e-12:
                output();progress('output');last_progress=time.perf_counter()
                target_index+=1;target=min(end,start+target_index*interval)
            elif time.perf_counter()-last_progress>=15:
                progress('progress');last_progress=time.perf_counter()
        if status=='running':status='completed'
        if not (manifest['frames'] and manifest['frames'][-1]['time_s']==current):
            output(frame=False)
        due=(knots<=end+1.e-12).astype(np.int32)
        if status=='completed' and not np.array_equal(knot_hits,due):
            raise RuntimeError('G3 live-bank completed trajectory missed a live forcing knot')
        expected_euler=3*((steps-initial_steps)+(rejected-initial_rejected))
        if solver.euler_calls-initial_euler!=expected_euler:
            raise RuntimeError('G3 live-bank accepted/rejected Euler count differs')
        manifest.update(status=status,final_time_s=current,steps=steps,
            rejected=rejected,euler_calls_in_process=solver.euler_calls-initial_euler,
            expected_euler_calls=expected_euler,knot_hits=knot_hits.tolist(),
            boundary_event_rows=event_rows,total_wall_s=time.perf_counter()-began)
        write_json(destination/'trials.json',trial_rows)
        manifest['trial_trace_sha256']=sha(destination/'trials.json')
        write_json(destination/'run.json',manifest)
        progress('finished')
        return manifest
    except Exception as exc:
        status='failed';manifest.update(status=status,final_time_s=current,
            steps=steps,rejected=rejected,error=str(exc))
        write_json(destination/'run.json',manifest)
        progress('failed')
        raise
    finally:
        stack.close()
