"""Exact-fixture, explicitly selected composed GPU AMR runner bridge.

This is an experimental public-worker route for one authenticated case, not a
general AMR implementation. The numerical adapter remains the frozen private
Gate-B implementation; this module supplies runner admission, progress,
transaction-boundary cancellation, output and restart without CPU fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .atomic_output import save_arrays_atomic, write_json_atomic
from .runner import fields_digest, sha


VARIANT = 'authenticated_composed_amr_ssp33'
PHYSICS_SCOPE = ('Frozen 91-to-64-leaf wet/dry MPM/Exner, two suspended sand '
                 'classes with exchange and signed prognostic hI; one native '
                 '2:1 AMR coarsening event; no other input or process supported')


def _reference():
    from braids_support.gpu_fwave import amr_composed_native_fixture_v2 as case
    from braids_support.gpu_fwave import amr_composed_b3_provenance as b3_gate
    from braids_support.gpu_fwave import amr_composed_b3_frames_provenance as frame_gate
    b3_gate.preflight()
    frame_gate.preflight()
    plan=json.loads((case.ROOT/'plan.json').read_text(encoding='utf-8'))
    expected={Path(name).name:digest for name,digest in plan['inputs'].items()}
    if len(expected)!=len(plan['inputs']):
        raise ValueError('Frozen AMR input basenames are not unique')
    return case,plan,expected,b3_gate,frame_gate


def authenticate_input(input_path):
    """CPU-only, exact native-input guard; independent of caller path spelling."""
    case,plan,expected,b3_gate,frame_gate=_reference()
    folder=Path(input_path).resolve()
    if not folder.is_dir() or {p.name for p in folder.iterdir() if p.is_file()}!=set(expected):
        raise ValueError('Authenticated composed AMR requires exactly the frozen native input artifacts')
    for name,digest in expected.items():
        if sha(folder/name)!=digest:
            raise ValueError('Authenticated composed AMR input changed: '+name)
    if (plan['runtime_sha256']!=
            '21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5'):
        raise ValueError('Authenticated composed AMR native reference changed')
    identity=hashlib.sha256(json.dumps(expected,sort_keys=True,
        separators=(',',':')).encode()).hexdigest()
    return dict(input_identity=identity, input_folder=str(folder),
        native_plan_sha256=sha(case.ROOT/'plan.json'),
        Gate_B3_source_lock_sha256=sha(b3_gate.LOCK),
        Gate_B_all_frames_lock_sha256=sha(frame_gate.LOCK),
        expected_artifacts=expected)


def _method_identity(admission):
    package=Path(__file__).resolve().parent
    public_sources={name:sha(package/name) for name in
        ('amr_composed_runner.py','runner.py','__main__.py','atomic_output.py')}
    method=dict(variant=VARIANT, integrator='SSPRK33', launch_mode='plain',
        initial_leaves=91, post_coarsening_leaves=64, end_s=.08,
        outer_dt_s=.002, output_every_s=.02, cfl=.4,
        source_lock=admission['Gate_B3_source_lock_sha256'],
        frame_lock=admission['Gate_B_all_frames_lock_sha256'],
        public_source_sha256=public_sources)
    digest=hashlib.sha256(json.dumps(method,sort_keys=True,
        separators=(',',':')).encode()).hexdigest()
    return method,digest


def _state_fields(model,time,admission,method_identity):
    values=model.persistent(time)
    values['area']=model.area.copy()
    values.update({f'solver_ledger_{key}':model.cp.asnumpy(value).copy()
                   for key,value in model.solver.ledger.items()})
    values.update(input_identity=np.asarray(admission['input_identity']),
        source_identity=np.asarray(admission['Gate_B3_source_lock_sha256']),
        method_identity=np.asarray(method_identity))
    values['checkpoint_payload_sha256']=np.asarray(fields_digest(values))
    return values


def _read_checkpoint(path,admission,method_identity):
    """Verify accepted-state identity and minimum structural schema before CUDA."""
    with np.load(Path(path),allow_pickle=False) as archive:
        fields={key:archive[key] for key in archive.files}
    digest=fields.pop('checkpoint_payload_sha256',None)
    if digest is None or digest.shape!=() or digest.dtype.kind not in 'US' or \
            digest.item()!=fields_digest(fields):
        raise ValueError('Authenticated AMR checkpoint payload digest differs')
    for key,expected in (('input_identity',admission['input_identity']),
                         ('source_identity',admission['Gate_B3_source_lock_sha256']),
                         ('method_identity',method_identity)):
        value=fields.get(key)
        if value is None or value.shape!=() or value.dtype.kind not in 'US' or \
                value.item()!=expected:
            raise ValueError('Authenticated AMR checkpoint '+key+' differs')
    steps=np.asarray(fields.get('accepted_steps'))
    time_value=np.asarray(fields.get('time_s'))
    mesh=np.asarray(fields.get('mesh_index'))
    if (steps.shape!=() or steps.dtype.kind not in 'iu' or
            not 0<=int(steps)<=40 or time_value.shape!=() or
            time_value.dtype.kind!='f' or
            abs(float(time_value)-.002*int(steps))>1e-12 or
            mesh.shape!=() or int(mesh)!=(0 if int(steps)<30 else 3)):
        raise ValueError('Authenticated AMR checkpoint clock/topology differs')
    leaves=91 if int(mesh)==0 else 64
    if (np.asarray(fields.get('state')).shape!=(leaves,10) or
            np.asarray(fields.get('node_ids')).shape!=(leaves,) or
            np.asarray(fields.get('area')).shape!=(leaves,)):
        raise ValueError('Authenticated AMR checkpoint resident field shape differs')
    return fields


def run_authenticated_amr(*,input_path,output_path,until=None,
        output_every=None,dt_max=None,max_steps=None,stop_file=None,resume=None,
        launch_mode='plain',integrator='ssp33',capture_stages=False,
        capture_step_trace=False,cancelled=lambda:False,**unused):
    """Advance only the authenticated case with native AMR output clocks."""
    if (launch_mode!='plain' or integrator!='ssp33' or capture_stages or
            capture_step_trace):
        raise ValueError('Authenticated composed AMR requires plain SSPRK33 without stage/step trace')
    if (until is not None and float(until)!=.08 or
            output_every is not None and float(output_every)!=.02 or
            dt_max is not None and float(dt_max)!=.002):
        raise ValueError('Authenticated composed AMR requires frozen 0.08 s end, 0.02 s output and 0.002 s outer dt')
    if max_steps is not None and (type(max_steps) is not int or not 1<=max_steps<=40):
        raise ValueError('Authenticated composed AMR max_steps must be 1..40')
    admission=authenticate_input(input_path)
    method,method_identity=_method_identity(admission)
    restored=None if resume is None else _read_checkpoint(resume,admission,
                                                          method_identity)
    destination=Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError('Authenticated AMR output must be a new directory')
    if destination==Path(input_path).resolve() or \
            destination.is_relative_to(Path(input_path).resolve()):
        raise ValueError('Authenticated AMR output cannot be inside input artifacts')
    from braids_support.gpu_fwave.amr_composed_b3_gpu import ComposedB3
    model=ComposedB3()
    if restored is not None:
        model.restore(restored)
        for key,accepted in model.solver.ledger.items():
            name=f'solver_ledger_{key}'
            if name not in restored or restored[name].shape!=accepted.shape or \
                    restored[name].dtype!=np.float64:
                raise ValueError('Authenticated AMR checkpoint accepted ledger differs')
            model.cp.copyto(accepted,model.cp.asarray(restored[name]))
        if not np.array_equal(model.area,restored['area']):
            raise ValueError('Authenticated AMR checkpoint cell areas differ')
    destination.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    manifest=dict(schema=1,backend='gpu',backend_selected_by='user',
        automatic_backend_selection=False,variant=VARIANT,
        scope=PHYSICS_SCOPE,experimental_exact_fixture=True,
        general_AMR_supported=False,QGIS_release=False,
        input=admission,method=method,method_identity=method_identity,
        resume_from=None if resume is None else str(Path(resume).resolve()),
        frames=[],accepted_step_checks=[],status='running',
        Delft3D_modified=False)
    write_json_atomic(destination/'run.json',manifest)
    current=.002*model.accepted_steps
    frame_index=0
    def progress(event,**extra):
        row=dict(event=event,backend='gpu',variant=VARIANT,time_s=current,
            end_s=.08,steps=model.accepted_steps,
            percent=100.*current/.08,wall_s=time.perf_counter()-began,**extra)
        write_json_atomic(destination/'status.json',row)
        print(json.dumps(row,allow_nan=False),flush=True)
    def output(*,frame):
        nonlocal frame_index
        fields=_state_fields(model,current,admission,method_identity)
        save_arrays_atomic(destination/'checkpoint.npz',**fields)
        if frame:
            name=f'frame_{frame_index:06d}.npz'
            save_arrays_atomic(destination/name,**fields)
            manifest['frames'].append(dict(file=name,time_s=current,
                steps=model.accepted_steps,mesh_index=model.mesh_index,
                leaves=len(model.ids),sha256=sha(destination/name)))
            write_json_atomic(destination/'run.json',manifest)
            frame_index+=1
    output(frame=abs(current/.02-round(current/.02))<1e-10)
    progress('initialized',mesh_index=model.mesh_index,leaves=len(model.ids))
    while model.accepted_steps<40:
        if cancelled() or (stop_file is not None and Path(stop_file).exists()):
            manifest.update(status='cancelled',final_time_s=current,
                            steps=model.accepted_steps)
            write_json_atomic(destination/'run.json',manifest)
            progress('cancelled',mesh_index=model.mesh_index)
            return dict(status='cancelled',output=str(destination),
                        steps=model.accepted_steps,time_s=current)
        if max_steps is not None and model.accepted_steps>=max_steps:
            manifest.update(status='stopped',final_time_s=current,
                            steps=model.accepted_steps)
            write_json_atomic(destination/'run.json',manifest)
            progress('stopped',reason='max_steps',mesh_index=model.mesh_index)
            return dict(status='stopped',output=str(destination),
                        steps=model.accepted_steps,time_s=current)
        outer=model.accepted_steps
        row=model.step(.002,outer*.002)
        current=round((outer+1)*.002,12)
        if (outer+1)%10==0:
            model.regrid_evaluation(current)
            output(frame=True)
        else:
            output(frame=False)
        manifest['accepted_step_checks'].append(row)
        write_json_atomic(destination/'run.json',manifest)
        progress('step',mesh_index=model.mesh_index,leaves=len(model.ids))
    manifest.update(status='completed',final_time_s=current,steps=40)
    write_json_atomic(destination/'run.json',manifest)
    progress('completed',mesh_index=model.mesh_index,leaves=len(model.ids))
    return dict(status='completed',output=str(destination),steps=40,
                time_s=.08,frames=len(manifest['frames']),
                method_identity=method_identity)
