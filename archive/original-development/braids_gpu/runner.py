"""Finite, explicit GPU runs with progress, cancellation and recoverable output."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, ExitStack
import numpy as np
from braids_core.windows_paths import filesystem_path
from .atomic_output import save_arrays_atomic as save_arrays, write_json_atomic as write_json

ROOT=Path(__file__).resolve().parents[1]
CUDA_SOURCE=ROOT/'validation/gpu_fwave'
_COMBINED_BINDING_LOCK=threading.Lock()
A1_VARIANT='combined_live_scheduled_a1_ssp43'
AMR_VARIANT='authenticated_composed_amr_ssp33'
ON_DEMAND_AMR_VARIANT='on_demand_bank_amr_ssp33'
ON_DEMAND_TRANSPORT_AMR_VARIANT='on_demand_transport_amr_ssp33'
SECONDARY_LIVE_VARIANT='live_normal_critical_secondary_mpm_ssp43'
LIVE_SUSPENSION_VARIANT='live_normal_critical_suspension_mpm_ssp33'
LIVE_VR_VARIANT='live_normal_critical_van_rijn_mpm_ssp33'
GRADED_LIVE_VR_VARIANT='live_normal_critical_van_rijn_graded_ssp33'
TARGET_B_ADAPTATION_VARIANT='live_normal_critical_graded_adaptation_ssp33'
GRADED_BEDLOAD_GENERAL_VARIANT='live_normal_critical_graded_bedload_ssp33'
LIVE_INLET_VARIANT='live_mode9_section_choice_ssp33'
GENERAL_LIVE_BANK_VARIANT='live_normal_critical_bank_ssp33'
GENERAL_LIVE_TRANSPORT_VARIANT='live_normal_critical_transport_ssp33'
INTEGRATED_VARIANT='integrated'
G3_VARIANT='physical_toe_mode2_avalanche_secondary_ssp33'
G3_LIVE_BANK_MODE2_VARIANT='g3_live_bank_mode2_ssp33'
G3_LIVE_BANK_MODE3_VARIANT='g3_live_bank_mode3_ssp33'
G3_LIVE_BANK_VARIANTS=(G3_LIVE_BANK_MODE2_VARIANT,G3_LIVE_BANK_MODE3_VARIANT)


def sha(path):return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()


def fields_digest(fields):
    """Stable byte identity of accepted host fields and signed ledgers."""
    digest=hashlib.sha256()
    for key in sorted(fields):
        value=np.ascontiguousarray(fields[key]);digest.update(key.encode()+b'\0')
        digest.update(value.dtype.str.encode()+b'\0')
        digest.update(json.dumps(value.shape).encode()+b'\0')
        digest.update(value.tobytes())
    return digest.hexdigest()


def reject_deferred_public_controls(data):
    """Fail closed on the first-goal GPU menu before output or CUDA work.

    Component kernels may recognize some of these controls, but no selectable
    whole-step implementation has qualified them. This guard applies to the
    fixed-grid public runner; AMR has a separate prepared-schema admission.
    """
    def active(key):
        return bool(np.any(np.asarray(data.get(key, 0)) != 0))

    if int(np.asarray(data.get('exchange_mode', 0)).item()) in (1, 2) and active('settling_mode'):
        raise ValueError('GPU hindered settling is deferred; use constant settling velocity')
    if active('active_layer_mode'):
        raise ValueError('GPU water-depth-scaled active layer is deferred; use constant thickness')
    if (active('nmorfac') or
            np.any(np.asarray(data.get('morphological_factor', 1.)) != 1.) or
            np.any(np.asarray(data.get('morfac', 1.)) != 1.)):
        raise ValueError('GPU morphological acceleration is deferred; use MorFac 1')
    if active('bank_front_tracking'):
        raise ValueError('GPU persistent bank-front tracking is excluded')
    if active('bank_failure') and int(np.asarray(data.get('bank_toe_erosion_mode', 0)).item()) == 1:
        raise ValueError('GPU cohesive bank failure is excluded from the noncohesive solver')
    if active('viscosity_equilibrated_inlet'):
        raise ValueError('GPU viscosity-equilibrated total-discharge inlet is deferred')
    if (np.any(np.asarray(data.get('horizontal_viscosity', 0.)) > 0.) and
            not active('horizontal_viscosity_mode')):
        raise ValueError('GPU depth-conservative viscosity is deferred; use velocity-laplacian viscosity')
    if np.isin(np.asarray(data.get('mode', ())), (3, 5, 7)).any():
        raise ValueError(
            'GPU supercritical, weir and prescribed-stage-trace boundaries are deferred')


def load_prepared(folder, *, bank_avalanche_mode2=False,
                  distributed_bank_avalanche_mode3=False,
                  general_live_sections=False, g3_bank_secondary=False):
    from validation.gpu_fwave.coupled_reference import read_snapshot
    folder=filesystem_path(folder).resolve()
    receipt=json.loads((folder/'preparation_receipt.json').read_text())
    if receipt.get('schema') not in (1,2,3):
        raise ValueError('Unsupported GPU preparation schema')
    if receipt['schema']==3 and (
            receipt.get('variant')!=GRADED_BEDLOAD_GENERAL_VARIANT or
            set(receipt.get('artifacts',{}))!={
                'coupled_input.bin','coupled_input.npz','normal_metadata.bin'} or
            receipt.get('native_solver_steps_performed')!=0 or
            receipt.get('GPU_solver_executed') is not False):
        raise ValueError('GPU preparation schema 3 requires the exact zero-step graded-bedload contract')
    if receipt['schema']==2:
        if g3_bank_secondary:
            if (receipt.get('case') not in ('all_on','no_toe','no_avalanche','no_secondary') or
                    set(receipt.get('artifacts',{})) != {
                        'coupled_input.bin','bank_toe_metadata.bin','avalanche_metadata.bin'} or
                    not (folder/'bank_toe_metadata.bin').is_file() or
                    not (folder/'avalanche_metadata.bin').is_file()):
                raise ValueError('G3 GPU selector requires authenticated core and two sidecars')
        elif (not (folder/'distributed_bank_metadata.bin').is_file() or
                receipt.get('native_reference_id') not in (
                    'i4_distributed_bank_mode3_native_20261004_v2',
                    'i4_mode3_active_overlap_native_20261004_v1',
                    'i4_mode3_avalanche_native_20261004_v1')):
            raise ValueError('GPU preparation schema 2 is bound to authenticated mode-3 or explicit G3 reference')
    required={'coupled_input.bin'}
    if (folder/'coupled_diffusion.bin').exists():required.add('coupled_diffusion.bin')
    if (folder/'scheduled_metadata.bin').exists():required.add('scheduled_metadata.bin')
    if (folder/'avalanche_metadata.bin').exists():required.add('avalanche_metadata.bin')
    if (folder/'bank_toe_metadata.bin').exists():required.add('bank_toe_metadata.bin')
    if (folder/'distributed_bank_metadata.bin').exists():required.add('distributed_bank_metadata.bin')
    if not required<=set(receipt.get('artifacts',{})):raise ValueError('Missing input identity')
    for name,digest in receipt['artifacts'].items():
        path=(folder/name).resolve()
        if path.parent!=folder:raise ValueError('Input artifact must be a direct child')
        if not path.is_file():raise ValueError('Prepared input artifact missing: '+name)
        if sha(path)!=digest:raise ValueError('Prepared input changed: '+name)
    if (folder/'normal_metadata.bin').exists() and (folder/'scheduled_metadata.bin').exists():
        raise ValueError('Normal/critical and stock scheduled metadata cannot coexist')
    if (folder/'avalanche_metadata.bin').exists() and any((folder/name).exists() for name in
            ('normal_metadata.bin','scheduled_metadata.bin')):
        raise ValueError('General-avalanche metadata cannot coexist with scheduled/section boundaries')
    if ((folder/'bank_toe_metadata.bin').exists() and
            (any((folder/name).exists() for name in
                 ('normal_metadata.bin','scheduled_metadata.bin')) or
             ((folder/'avalanche_metadata.bin').exists() and not bank_avalanche_mode2))):
        raise ValueError('Physical bank-toe metadata cannot coexist with other physics/boundary sidecars')
    distributed_conflicts=('normal_metadata.bin','scheduled_metadata.bin',
                           'bank_toe_metadata.bin','coupled_diffusion.bin')
    if not distributed_bank_avalanche_mode3:
        distributed_conflicts+=('avalanche_metadata.bin',)
    if (folder/'distributed_bank_metadata.bin').exists() and any((folder/name).exists() for name in
            distributed_conflicts):
        raise ValueError('Distributed-bank metadata cannot coexist with other physics/boundary sidecars')
    if (folder/'normal_metadata.bin').exists():
        if 'normal_metadata.bin' not in receipt['artifacts']:
            raise ValueError('Missing normal boundary metadata identity')
        if general_live_sections:
            from .live_secondary import read_case
        else:
            from validation.gpu_fwave.normal_outlet_cases import read_case
        data=read_case(folder)
    else:
        data=read_snapshot(folder)
        if (folder/'scheduled_metadata.bin').exists():
            from validation.gpu_fwave.coupled_reference import read_records
            extra=read_records(folder/'scheduled_metadata.bin')
            for key,value in extra.items():
                if key in data and not np.array_equal(data[key],value):
                    raise ValueError('Scheduled duplicate export differs: '+key)
                data[key]=value
        if (folder/'avalanche_metadata.bin').exists():
            from validation.gpu_fwave.coupled_reference import read_records
            extra=read_records(folder/'avalanche_metadata.bin')
            controls={'general_avalanche_wet_slope','general_avalanche_dry_slope',
                'general_avalanche_time','general_avalanche_wet_depth_switch',
                'general_avalanche_relaxation_length'}
            if set(extra)!=controls|{'nc','nf','nfrac','nbed_layers','deck_version','general_avalanche'}:
                raise ValueError('Unsupported general-avalanche metadata schema')
            for key,value in extra.items():
                if key in data and not np.array_equal(data[key],value):
                    raise ValueError('General-avalanche duplicate prepared value differs: '+key)
                data[key]=value
        if (folder/'bank_toe_metadata.bin').exists():
            from validation.gpu_fwave.coupled_reference import read_records
            extra=read_records(folder/'bank_toe_metadata.bin')
            identity={'nc','nf','nfrac','nbed_layers','deck_version'}
            flags={'bank_toe_erosion_mode','bank_failure','toe_mapping_status'}
            scalars={'bank_toe_capacity_fraction','bank_critical_undercut_m'}
            cells={'bank_material_mask','physical_toe_cell','physical_toe_donor',
                'physical_toe_face','physical_toe_distance','is_toe_cell',
                'is_toe_donor','physical_toe_status'}
            faces={'is_toe_face','candidate_bank','candidate_toe'}
            if set(extra)!=identity|flags|scalars|cells|faces:
                raise ValueError('Unsupported physical bank-toe metadata schema')
            nc=int(data['nc']);nf=int(data['nf'])
            for key in identity|flags|scalars:
                if np.asarray(extra[key]).shape!=():
                    raise ValueError('Invalid scalar physical bank-toe metadata: '+key)
            for key in cells:
                if np.asarray(extra[key]).shape!=(nc,):
                    raise ValueError('Invalid cell physical bank-toe metadata: '+key)
            for key in faces:
                if np.asarray(extra[key]).shape!=(nf,):
                    raise ValueError('Invalid face physical bank-toe metadata: '+key)
            if int(extra['toe_mapping_status'])!=0:
                raise ValueError('Native physical bank-toe mapping failed')
            for key,value in extra.items():
                if key in data and not np.array_equal(data[key],value):
                    raise ValueError('Physical bank-toe duplicate prepared value differs: '+key)
                data[key]=value
        if (folder/'distributed_bank_metadata.bin').exists():
            from validation.gpu_fwave.coupled_reference import read_records
            extra=read_records(folder/'distributed_bank_metadata.bin')
            identity={'nc','nf','nfrac','nbed_layers','deck_version'}
            flags={'bank_toe_erosion_mode','bank_failure'}
            scalars={'bank_distributed_transport_fraction',
                'bank_distributed_critical_slope','bank_critical_undercut_m'}
            mask='bank_material_mask'
            if set(extra)!=identity|flags|scalars|{mask}:
                raise ValueError('Unsupported distributed-bank metadata schema')
            nc=int(data['nc'])
            for key in identity|flags:
                original=extra[key];value=np.asarray(original)
                if (value.shape!=() or isinstance(original,(bool,np.bool_)) or
                        not (type(original) is int or value.dtype==np.dtype('<i4')) or
                        not np.iinfo(np.int32).min<=int(original)<=np.iinfo(np.int32).max):
                    raise ValueError('Invalid integer distributed-bank metadata: '+key)
            for key in scalars:
                original=extra[key];value=np.asarray(original)
                if (value.shape!=() or not (type(original) is float or
                        value.dtype==np.dtype('<f8')) or not np.isfinite(value)):
                    raise ValueError('Invalid real distributed-bank metadata: '+key)
            bank_mask=np.asarray(extra[mask])
            if (bank_mask.shape!=(nc,) or bank_mask.dtype!=np.dtype('<i4') or
                    not np.isin(bank_mask,(0,1)).all() or not np.any(bank_mask)):
                raise ValueError('Invalid distributed-bank material mask')
            if (int(extra['bank_toe_erosion_mode'])!=3 or int(extra['bank_failure'])!=1 or
                    not 0.<=float(extra['bank_distributed_transport_fraction'])<=1. or
                    float(extra['bank_distributed_critical_slope'])<=0. or
                    float(extra['bank_critical_undercut_m'])<=0.):
                raise ValueError('Invalid distributed-bank controls')
            for key,value in extra.items():
                if key in data and not np.array_equal(data[key],value):
                    raise ValueError('Distributed-bank duplicate prepared value differs: '+key)
                data[key]=value
    return data,sha(folder/'preparation_receipt.json')


def combined_scope(data):
    """The current integrated adapter admits the measured mode9/mode4 section pair."""
    from validation.gpu_fwave.normal_outlet_boundary import validate_boundaries
    from validation.gpu_fwave.boundary_events import active_knots
    if int(data['cpu_time_integrator'])!=43 or int(data['cpu_outlet_extension'])!=1:
        raise ValueError('Combined GPU boundary scope requires native SSPRK43 and outlet extension 1')
    validate_boundaries(data);active_knots(data)


def g3_frozen_preflight(data, input_path, *, start, end, interval, dtmax,
                        launch_mode, integrator):
    """Pin the four-case G3 composition to its immutable native/input gate."""
    folder=filesystem_path(input_path).resolve()
    case=folder.name
    prepared_root=CUDA_SOURCE/'runs/g3_bank_secondary_prepared_20261004_v2'
    receipt_sha={
        'all_on':'3941836421396f6feac679f3f0bad6ed57238d0ecd650d168ef71fcef010cada',
        'no_toe':'e4e9d0fbd0a654e145c91ac9668505173b86f3c57f7fc6205804c535029f683f',
        'no_avalanche':'204b3c13929b17f887e249b6b97226c8aaed88799c8a980d0b3b81f04b4c7e3b',
        'no_secondary':'fc08b757b8fb16a734d7147444f495fa33d6ad429580d5edcd4128b22f17d19a'}
    if case not in receipt_sha or folder!=prepared_root.resolve()/case:
        raise ValueError('G3 selector requires one of the four frozen prepared case folders')
    if launch_mode!='plain' or integrator!='ssp33':
        raise ValueError('G3 bank/avalanche/secondary requires explicit plain SSPRK33')
    if ((start,end,interval,dtmax) not in ((0.,.5,.1,.002),(0.,.2,.1,.002)) or
            float(data['cfl_target'])!=.4):
        raise ValueError('G3 selector requires frozen 0-0.5 or 0-0.2 s clock, 0.1 s output, 0.002 s dtmax and CFL 0.4')
    gate=CUDA_SOURCE/'reports/g3_bank_secondary_native_20261004_v1/GPU_ACCEPTANCE_GATE.md'
    gate_sha='986be9e6707f6be8f23bc724a65b7a81a9c6f044868fc85b1d21171c38f70537'
    manifest=prepared_root/'preparation_manifest.json'
    manifest_sha='e360678d89530dcfc8e7e78fb4d29a0310600a7c417c2df13d84508c8a8d70b9'
    if sha(gate)!=gate_sha or sha(manifest)!=manifest_sha:
        raise ValueError('Frozen G3 GPU gate or preparation manifest differs')
    reference=dict(
        gate_sha256='8fda5d9d60d9296f519e75be3ec48e87eff78c29017d19c94a28b4a32b2cb687',
        plan_sha256='2e3da861328b34ea4a4be25d02574fd26aaee30cbe6d91ee975865adf93b3b38',
        analysis_sha256='826c9635cd47af86c49a2c0c0c4fea0a9e28ef6112bd753c4a82dd7bc6a093f9',
        native_executable_sha256='21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5')
    aggregate=json.loads(manifest.read_text(encoding='utf-8'))
    receipt_path=folder/'preparation_receipt.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    if (sha(receipt_path)!=receipt_sha[case] or
            aggregate.get('receipt_sha256',{}).get(case)!=receipt_sha[case] or
            aggregate.get('native_reference')!=reference or
            receipt.get('native_reference')!=reference or
            receipt.get('artifacts')!=aggregate.get('artifacts',{}).get(case) or
            receipt.get('solver_steps_performed')!=0 or
            receipt.get('numerical_routines_unchanged') is not True or
            receipt.get('Delft3D_source_modified') is not False or
            receipt.get('GPU_solver_executed') is not False or
            any(sha(folder/name)!=digest for name,digest in receipt['artifacts'].items())):
        raise ValueError('Frozen G3 prepared/native provenance differs')
    return dict(case=case,gate_sha256=gate_sha,
                prepared_manifest_sha256=manifest_sha,
                native_reference=reference,
                preparation_receipt_sha256=receipt_sha[case])


def a1_frozen_preflight(data,input_path,*,start,end,interval,dtmax,launch_mode,integrator):
    """Admit only the two authenticated full-trajectory A1 native fixtures."""
    from . import a1_source_lock_v2 as source_lock
    from validation.gpu_fwave import a1_combined_current_adapter as candidate
    from validation.gpu_fwave import a1_combined_full_gate_preflight as gate
    from validation.gpu_fwave import a1_combined_live_scheduled_native_case as native
    from validation.gpu_fwave.boundary_events import BoundaryEvents
    if launch_mode!='plain' or integrator!='ssp43':
        raise ValueError('A1 live scheduled variant requires explicit plain SSPRK43')
    if (start,end,interval,dtmax)!=(0.,.2,.025,.002) or float(data['cfl_target'])!=.4:
        raise ValueError('A1 live scheduled variant requires frozen 0-0.20 s, 0.025 s output, 0.002 s dtmax and CFL 0.4')
    folder=filesystem_path(input_path).resolve()
    if any((folder/name).exists() for name in ('scheduled_metadata.bin','avalanche_metadata.bin',
            'bank_toe_metadata.bin','distributed_bank_metadata.bin','coupled_diffusion.bin')):
        raise ValueError('A1 native normal_metadata sidecar cannot be combined with another physics sidecar')
    matches=[name for name,row in gate.PINNED.items()
        if sha(folder/'coupled_input.bin')==row['input'] and
           sha(folder/'normal_metadata.bin')==row['schedule'] and
           sha(folder/'preparation_receipt.json')==sha(native.ROOT/name/'prepared/preparation_receipt.json')]
    if len(matches)!=1:raise ValueError('A1 prepared input/normal_metadata identity is not a frozen pool or wet/dry case')
    variant=matches[0]
    with source_lock.compatible_source_lock():
        gate.preflight(variant)
        candidate.validate_input(data)
    combined_scope(data)
    events=BoundaryEvents.from_tables(data)
    if not np.array_equal(events.knots,[0.,.07,.13,.2]):
        raise ValueError('A1 live forcing knots differ from the frozen full gate')
    return variant,events,gate


def a1_auxiliary_device_arrays(solver):
    """Every accepted A1 diagnostic and persistent root-cache buffer."""
    boundary=solver.boundary
    return {
        'a1_schedule_clock':solver.schedule.clock,
        **{'a1_schedule_'+key:value for key,value in solver.schedule.outputs.items()},
        **{'a1_normal_'+key:value for key,value in boundary.normal_records.items()},
        **{'a1_outlet_'+key:value for key,value in boundary.outlet_records.items()},
        **{'a1_normal_cache_'+key:value for key,value in boundary.normal_root.cache.items()},
        'a1_normal_latch':boundary._normal_latch,
        'a1_outlet_latch':boundary._outlet_latch}


def a1_checkpoint_extra_schema(data,event_count=2):
    """Host-only exact A1 schema, usable before importing or constructing CUDA."""
    nc=int(data['nc']);ng=int(data['nfrac']);nb=int(data['nb'])
    offsets=np.asarray(data['boundary_offset']);normal_faces=int(offsets[1]-offsets[0])
    outlet_faces=int(offsets[2]-offsets[1])
    f=np.float64;i=np.int32
    rows={'bed':((4,nc),f),'nodes':((4,normal_faces+1),f),
          'stage':((4,1),f),'discharge':((4,1),f),'trace':((4,3,normal_faces),f),
          **{key:((4,1),i) for key in ('root_status','trace_status','failure_latch',
                                      'cache_hit','bisections','metric_evaluations')},
          'recorded':((4,),i)}
    outlet={'bed':((4,nc),f),'q':((4,3,nc),f),'trace':((4,3,outlet_faces),f),
            'failure_latch':((4,1),i),'recorded':((4,),i),
            **{key:((4,outlet_faces),f) for key in ('invariant','face_speed')},
            **{key:((4,outlet_faces),i) for key in ('branch','initial_branch','removal_pass',
                        'removal_reason','face_status')},
            **{key:((4,1),f) for key in ('stage','area','top_width','characteristic_Q',
                        'critical_Q','residual','outward_discharge','lower_residual',
                        'upper_residual','failed_speed')},
            **{key:((4,1),i) for key in ('status','active_passes','expansions','bisections',
                        'metric_evaluations','failed_face')}}
    cache={'valid':((1,),i),'bed':((normal_faces,),f),
           **{key:((1,),f) for key in ('discharge','stage','area','top_width','conveyance','residual')},
           'selected_branch':((1,),i),'selected_kind':((1,),i)}
    return {'bed_elevation':((nc,),f),'a1_schedule_clock':((2,),f),
        'a1_schedule_stage_time':((4,),f),'a1_schedule_hydraulic':((4,2,nb),f),
        'a1_schedule_concentration':((4,ng,nb),f),'a1_schedule_feed':((4,ng,nb),f),
        'a1_schedule_status':((4,nb),i),
        **{'a1_normal_'+key:value for key,value in rows.items()},
        **{'a1_outlet_'+key:value for key,value in outlet.items()},
        **{'a1_normal_cache_'+key:value for key,value in cache.items()},
        'a1_normal_latch':((1,),i),'a1_outlet_latch':((1,),i),
        'a1_event_cursor':((),np.int64),'a1_knot_hits':((event_count,),np.int32)}


def a1_event_cursor(events,current):
    from validation.gpu_fwave.boundary_events import tolerance
    return int(np.searchsorted(events.knots,float(current)+tolerance(current),side='right'))


def a1_checkpoint_preflight(resume,data,*,input_identity,source_identity,method_identity,events,start,end):
    """Reject missing/corrupt auxiliary restart data before CUDA or output."""
    state={'q':np.asarray(data['q']).shape,'layers':np.asarray(data['layer_mass']).shape,
           'secondary':(int(data['nc']),)}
    ledger={'water':(int(data['nb']),),'bed':(int(data['nfrac']),int(data['nb'])),
            'secondary':(int(data['nb']),),'source':()}
    extras=a1_checkpoint_extra_schema(data)
    expected=set(state)|{'ledger_'+key for key in ledger}|set(extras)|{
        'time','dt_hint','steps','rejected','input_identity','source_identity',
        'method_identity','event_identity','next_boundary_knot','a1_payload_sha256'}
    with np.load(resume,allow_pickle=False) as checkpoint:
        if set(checkpoint.files)!=expected:raise ValueError('A1 checkpoint fields differ from complete auxiliary schema')
        payload=np.asarray(checkpoint['a1_payload_sha256'])
        if (payload.shape!=() or payload.dtype.kind not in 'US' or
                payload.item()!=fields_digest({key:checkpoint[key] for key in checkpoint.files
                                               if key!='a1_payload_sha256'})):
            raise ValueError('A1 checkpoint payload digest differs')
        arrays,scalars=validate_checkpoint(checkpoint,state,ledger,input_identity=input_identity,
            source_identity=source_identity,method_identity=method_identity,events=events,
            secondary_model=int(data['secondary_flow_model']),start=start,end=end,
            extra_schema=extras,allow_nan_extras=('a1_outlet_failed_speed',))
    if int(arrays['a1_event_cursor'])!=a1_event_cursor(events,scalars['time']):
        raise ValueError('A1 restart event cursor differs from accepted clock')
    expected_hits=(events.knots[1:-1]<=float(scalars['time'])+1.e-12).astype(np.int32)
    if not np.array_equal(arrays['a1_knot_hits'],expected_hits):
        raise ValueError('A1 restart internal-knot history differs from accepted clock')
    if (not np.isin(arrays['a1_normal_cache_valid'],(0,1)).all() or
            not np.isin(arrays['a1_normal_recorded'],(0,1)).all() or
            not np.array_equal(arrays['a1_normal_recorded'],arrays['a1_outlet_recorded']) or
            np.any(arrays['a1_normal_latch']) or np.any(arrays['a1_outlet_latch']) or
            np.any(arrays['a1_schedule_status']) or np.any(arrays['a1_normal_root_status']) or
            np.any(arrays['a1_normal_trace_status']) or np.any(arrays['a1_outlet_status']) or
            np.any(arrays['a1_normal_failure_latch']) or
            np.any(arrays['a1_outlet_failure_latch']) or
            np.any(arrays['a1_outlet_face_status'])):
        raise ValueError('A1 accepted section/schedule diagnostics are invalid')
    if scalars['steps']:
        from validation.gpu_fwave import a1_combined_live_scheduled_native_case as native
        clock=arrays['a1_schedule_clock'];stages=arrays['a1_schedule_stage_time']
        expected_stage=np.asarray([clock[0],clock[0]+.5*clock[1],
                                   clock[0]+clock[1],clock[0]+.5*clock[1]])
        if (clock[1]<=0 or abs(clock[0]+clock[1]-scalars['time'])>1.e-12 or
                np.max(np.abs(stages-expected_stage))>1.e-12 or
                not np.array_equal(arrays['a1_normal_recorded'],[1,1,1,1])):
            raise ValueError('A1 restart resident stage clock/records differ from accepted step')
        q=np.interp(stages,native.KNOTS_S,native.Q_M3_S)
        feed=np.stack([np.interp(stages,native.KNOTS_S,native.FEED_KG_S[g])
                       for g in range(2)],axis=-1)
        if (np.max(np.abs(arrays['a1_schedule_hydraulic'][:,0,0]-q))>1.e-12 or
                np.max(np.abs(arrays['a1_schedule_feed'][:,:,0]-feed))>1.e-12):
            raise ValueError('A1 restart resident Q/class-feed outputs differ from frozen schedule')
    elif (np.any(arrays['a1_normal_recorded']) or np.any(arrays['a1_outlet_recorded'])):
        raise ValueError('A1 initial restart has nonempty stage records')
    return arrays,scalars


def a1_method(data,source_identity,events,dtmax,capture_stages):
    return dict(variant=A1_VARIANT,integrator='SSPRK43',euler_stages=4,
        launch_mode='plain',precision='FP64',dtmax_s=dtmax,
        capture_stages=bool(capture_stages),cfl=float(data['cfl_target']),
        morfac=float(data['morphological_factor']),bedlaw=int(data['bedlaw']),
        secondary_flow_model=int(data['secondary_flow_model']),
        exchange_mode=int(data['exchange_mode']),settling_mode=int(data.get('settling_mode',0)),
        exchange_ledger=dict(algorithm='none'),source_identity=source_identity,
        event_identity=events.identity,mpm_friction_mode=int(data['mpm_friction_mode']),
        mpm_cf=float(data['mpm_cf']),a1_gate_sha256=(
            'd26632b692d508f372e9ccbf09c696c33f75c585dfce09072df67d77be15e823'),
        authenticated_native_sidecar=True,auxiliary_checkpoint_schema=1,
        trial_transaction='segmented_all_reachable_v5')


def a1_transaction_sources():
    """Live implementation files authenticated in each frozen A1 run."""
    from . import a1_full_array_transaction as transaction
    from . import a1_source_lock_v2 as source_lock
    from validation.gpu_fwave import a2_segmented_array_copy as segmented
    from validation.gpu_fwave import a2_solver_array_inventory as inventory
    return {name:Path(module.__file__).resolve() for name,module in (
        ('a1_full_array_transaction.py',transaction),
        ('a1_source_lock_v2.py',source_lock),
        ('a2_segmented_array_copy.py',segmented),
        ('a2_solver_array_inventory.py',inventory))}


def a1_verify_transaction_sources(folder):
    folder=filesystem_path(folder).resolve()
    identity=json.loads((folder/'runtime_manifest.json').read_text(encoding='utf-8'))
    pinned=identity.get('transaction_sources')
    live=a1_transaction_sources()
    if not isinstance(pinned,dict) or set(pinned)!=set(live):
        raise ValueError('A1 complete transaction source manifest differs')
    for name,path in live.items():
        archived=folder/'transaction_python'/name
        if sha(path)!=pinned[name] or sha(archived)!=pinned[name]:
            raise ValueError('A1 complete transaction source identity differs: '+name)


def prepare_a1_run(data,input_path,destination,*,input_identity,events,start,end,dtmax,
                   capture_stages,resume,gate):
    """Freeze sources and authenticate restart before creating the target run."""
    from validation.gpu_fwave import a1_combined_current_adapter as candidate
    from . import a1_source_lock_v2 as source_lock
    with tempfile.TemporaryDirectory(prefix='braids_a1_source_') as temp:
        archive=Path(temp)/'source'
        with source_lock.compatible_source_lock():candidate.prepare_sources(archive,data)
        transaction_folder=archive/'transaction_python'
        transaction_folder.mkdir()
        transaction_sources={}
        for name,path in a1_transaction_sources().items():
            target=transaction_folder/name
            shutil.copy2(path,target)
            transaction_sources[name]=sha(path)
            if sha(target)!=transaction_sources[name]:
                raise ValueError('A1 complete transaction source changed during archive')
        files={str(path.relative_to(archive)).replace('\\','/'):sha(path)
               for path in archive.rglob('*') if path.is_file()}
        identity=dict(schema=1,variant=A1_VARIANT,source_files=files,
            transaction_sources=transaction_sources,
            runner_sha256=sha(__file__),cli_sha256=sha(Path(__file__).parent/'__main__.py'),
            gate_sha256=gate.GATE_SHA,
            candidate_receipt_sha256=sha(archive/'candidate_sources.json'))
        write_json(archive/'runtime_manifest.json',identity)
        source_identity=sha(archive/'runtime_manifest.json')
        method=a1_method(data,source_identity,events,dtmax,capture_stages)
        method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        restored=None
        if resume is not None:
            restored=a1_checkpoint_preflight(resume,data,input_identity=input_identity,
                source_identity=source_identity,method_identity=method_identity,
                events=events,start=start,end=end)
        destination=filesystem_path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
        source_folder=destination/'source';shutil.copytree(archive,source_folder)
        if (sha(source_folder/'runtime_manifest.json')!=source_identity or
                any(sha(source_folder/name)!=digest for name,digest in files.items())):
            raise ValueError('A1 archived numerical source changed while freezing')
        with source_lock.compatible_source_lock():candidate.verify_sources(source_folder)
        a1_verify_transaction_sources(source_folder)
    return source_identity,method,restored


@contextmanager
def a1_constructors(folder,data,bindings):
    from validation.gpu_fwave import a1_combined_current_adapter as candidate
    from . import a1_source_lock_v2 as source_lock
    from . import a1_full_array_transaction as transaction
    a1_verify_transaction_sources(folder)
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('A1 GPU constructor requires the main worker thread')
    if not _COMBINED_BINDING_LOCK.acquire(blocking=False):
        raise ValueError('Concurrent A1 GPU construction is unsupported')
    try:
        with source_lock.compatible_source_lock():
            with candidate.constructors(folder,data) as Solver:
                PublicA1Solver=transaction.wrap_solver(Solver)
                bindings.append(dict(kind='a1_current_source_live_scheduled',
                    source_receipt_sha256=sha(Path(folder)/'candidate_sources.json'),
                    runtime_manifest_sha256=sha(Path(folder)/'runtime_manifest.json'),
                    graph_supported=False,normal_root_cache_transaction=True,
                    complete_trial_transaction='segmented_all_reachable_v5'))
                yield PublicA1Solver,None
    finally:_COMBINED_BINDING_LOCK.release()


def live_internal_knots(events,start,end):
    return events.knots[(events.knots>start)&(events.knots<end)]


def secondary_checkpoint_preflight(resume,data,*,input_identity,source_identity,
                                   method_identity,events,start,end):
    """Authenticate the complete model-2 restart before output or CUDA use."""
    from validation.gpu_fwave.boundary_schedule import evaluate_host
    knots=live_internal_knots(events,start,end)
    nc=int(data['nc']);nb=int(data['nb']);ng=int(data['nfrac'])
    state={'q':np.asarray(data['q']).shape,'layers':np.asarray(data['layer_mass']).shape,
           'secondary':(nc,)}
    ledger={'water':(nb,),'bed':(ng,nb),'secondary':(nb,),'source':()}
    extras=a1_checkpoint_extra_schema(data,len(knots))
    expected=set(state)|{'ledger_'+key for key in ledger}|set(extras)|{
        'time','dt_hint','steps','rejected','input_identity','source_identity',
        'method_identity','event_identity','next_boundary_knot','a1_payload_sha256'}
    with np.load(resume,allow_pickle=False) as checkpoint:
        if set(checkpoint.files)!=expected:
            raise ValueError('Live secondary checkpoint fields differ from complete auxiliary schema')
        payload=np.asarray(checkpoint['a1_payload_sha256'])
        if (payload.shape!=() or payload.dtype.kind not in 'US' or
                payload.item()!=fields_digest({key:checkpoint[key] for key in checkpoint.files
                                               if key!='a1_payload_sha256'})):
            raise ValueError('Live secondary checkpoint payload digest differs')
        arrays,scalars=validate_checkpoint(checkpoint,state,ledger,
            input_identity=input_identity,source_identity=source_identity,
            method_identity=method_identity,events=events,secondary_model=2,
            start=start,end=end,extra_schema=extras,
            allow_nan_extras=('a1_outlet_failed_speed',))
    if int(arrays['a1_event_cursor'])!=a1_event_cursor(events,scalars['time']):
        raise ValueError('Live secondary restart event cursor differs')
    expected_hits=(knots<=float(scalars['time'])+1.e-12).astype(np.int32)
    if not np.array_equal(arrays['a1_knot_hits'],expected_hits):
        raise ValueError('Live secondary restart forcing-knot history differs')
    if (not np.isin(arrays['a1_normal_cache_valid'],(0,1)).all() or
            not np.isin(arrays['a1_normal_recorded'],(0,1)).all() or
            not np.array_equal(arrays['a1_normal_recorded'],arrays['a1_outlet_recorded']) or
            np.any(arrays['a1_normal_latch']) or np.any(arrays['a1_outlet_latch']) or
            np.any(arrays['a1_schedule_status']) or np.any(arrays['a1_normal_root_status']) or
            np.any(arrays['a1_normal_trace_status']) or np.any(arrays['a1_outlet_status']) or
            np.any(arrays['a1_normal_failure_latch']) or
            np.any(arrays['a1_outlet_failure_latch']) or
            np.any(arrays['a1_outlet_face_status'])):
        raise ValueError('Live secondary accepted section/schedule diagnostics are invalid')
    if scalars['steps']:
        clock=arrays['a1_schedule_clock'];stages=arrays['a1_schedule_stage_time']
        expected_stage=np.asarray([clock[0],clock[0]+.5*clock[1],
                                   clock[0]+clock[1],clock[0]+.5*clock[1]])
        if (clock[1]<=0 or abs(clock[0]+clock[1]-scalars['time'])>1.e-12 or
                np.max(np.abs(stages-expected_stage))>1.e-12 or
                not np.array_equal(arrays['a1_normal_recorded'],[1,1,1,1])):
            raise ValueError('Live secondary restart stage clock/records differ')
        from validation.gpu_fwave.boundary_schedule import validate_tables
        tables=validate_tables(dict(data,section_control=np.zeros(int(data['nb']),np.int32),
                                    sampled=np.zeros(int(data['nf']),np.int32)))
        expected_schedule=evaluate_host(tables,float(clock[0]),float(clock[1]),validated=True)
        for key in ('stage_time','hydraulic','concentration','feed'):
            if np.max(np.abs(arrays['a1_schedule_'+key]-expected_schedule[key]))>1.e-12:
                raise ValueError('Live secondary restart schedule differs: '+key)
    elif np.any(arrays['a1_normal_recorded']) or np.any(arrays['a1_outlet_recorded']):
        raise ValueError('Live secondary initial restart has nonempty stage records')
    return arrays,scalars


def prepare_secondary_run(data,input_path,destination,*,input_identity,events,start,end,
                          dtmax,capture_stages,resume):
    """Freeze the transformed composition and preflight restart before output."""
    from . import live_secondary
    from validation.gpu_fwave import a2_combined_prognostic_native_case as native
    from . import a1_full_array_transaction as transaction
    from validation.gpu_fwave import a1_combined_current_adapter as ancestor
    if sha(native.GATE)!=native.GATE_SHA:
        raise ValueError('Frozen A2 engineering gate identity differs')
    if sha(live_secondary.WETDRY_GATE)!=live_secondary.WETDRY_GATE_SHA:
        raise ValueError('Frozen A2 evolving wet/dry gate identity differs')
    with tempfile.TemporaryDirectory(prefix='braids_live_secondary_source_') as temp:
        archive=Path(temp)/'source'
        with live_secondary.isolated_adapter() as candidate:
            candidate.prepare_sources(archive,data)
            candidate.verify_sources(archive)
        transaction_folder=archive/'transaction_python';transaction_folder.mkdir()
        transaction_sources={}
        for name,path in a1_transaction_sources().items():
            target=transaction_folder/name;shutil.copy2(path,target)
            transaction_sources[name]=sha(path)
            if sha(target)!=transaction_sources[name]:
                raise ValueError('Live secondary transaction source changed during archive')
        files={str(path.relative_to(archive)).replace('\\','/'):sha(path)
               for path in archive.rglob('*') if path.is_file()}
        identity=dict(schema=1,variant=SECONDARY_LIVE_VARIANT,source_files=files,
            transaction_sources=transaction_sources,
            runner_sha256=sha(__file__),cli_sha256=sha(Path(__file__).parent/'__main__.py'),
            adapter_sha256=sha(live_secondary.__file__),
            ancestor_sha256=sha(ancestor.__file__),
            transaction_sha256=sha(transaction.__file__),
            historical_a2_gate_sha256=native.GATE_SHA,
            candidate_receipt_sha256=sha(archive/'candidate_sources.json'))
        write_json(archive/'runtime_manifest.json',identity)
        source_identity=sha(archive/'runtime_manifest.json')
        method=dict(variant=SECONDARY_LIVE_VARIANT,integrator='SSPRK43',euler_stages=4,
            launch_mode='plain',precision='FP64',dtmax_s=dtmax,
            capture_stages=bool(capture_stages),cfl=float(data['cfl_target']),
            morfac=float(data['morphological_factor']),bedlaw=2,
            secondary_flow_model=2,exchange_mode=0,settling_mode=int(data.get('settling_mode',0)),
            exchange_ledger=dict(algorithm='none'),source_identity=source_identity,
            event_identity=events.identity,mpm_friction_mode=int(data['mpm_friction_mode']),
            mpm_cf=float(data['mpm_cf']),historical_a2_gate_sha256=native.GATE_SHA,
            evolving_wet_dry_gate_sha256=live_secondary.WETDRY_GATE_SHA,
            authenticated_native_sidecar=True,auxiliary_checkpoint_schema=2,
            trial_transaction='segmented_all_reachable_v5')
        method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        restored=None
        if resume is not None:
            restored=secondary_checkpoint_preflight(resume,data,input_identity=input_identity,
                source_identity=source_identity,method_identity=method_identity,
                events=events,start=start,end=end)
        destination=filesystem_path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
        source_folder=destination/'source';shutil.copytree(archive,source_folder)
        if (sha(source_folder/'runtime_manifest.json')!=source_identity or
                any(sha(source_folder/name)!=digest for name,digest in files.items())):
            raise ValueError('Live secondary source changed while archiving')
        with live_secondary.isolated_adapter() as candidate:
            candidate.verify_sources(source_folder)
        a1_verify_transaction_sources(source_folder)
    return source_identity,method,restored


@contextmanager
def secondary_constructors(folder,data,bindings):
    from . import live_secondary
    from . import a1_full_array_transaction as transaction
    a1_verify_transaction_sources(folder)
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Live secondary GPU constructor requires the main worker thread')
    if not _COMBINED_BINDING_LOCK.acquire(blocking=False):
        raise ValueError('Concurrent live GPU construction is unsupported')
    try:
        with live_secondary.isolated_adapter() as candidate:
            with candidate.constructors(folder,data) as Solver:
                PublicSolver=transaction.wrap_solver(Solver)
                bindings.append(dict(kind='live_normal_critical_secondary_mpm',
                    source_receipt_sha256=sha(Path(folder)/'candidate_sources.json'),
                    runtime_manifest_sha256=sha(Path(folder)/'runtime_manifest.json'),
                    graph_supported=False,complete_trial_transaction='segmented_all_reachable_v5'))
                yield PublicSolver,None
    finally:_COMBINED_BINDING_LOCK.release()


def live_suspension_checkpoint_preflight(resume,data,*,input_identity,source_identity,
                                         method_identity,events,start,end):
    """Authenticate G2 state, exchange ledgers and every resident boundary record."""
    from validation.gpu_fwave.boundary_schedule import evaluate_host,validate_tables
    from validation.gpu_fwave.coupled_solver import normalized
    data=normalized(data)
    knots=live_internal_knots(events,start,end)
    nc=int(data['nc']);nb=int(data['nb']);ng=int(data['nfrac'])
    state={'q':np.asarray(data['q']).shape,'layers':np.asarray(data['layer_mass']).shape,
           'secondary':(nc,),'suspended':(ng,nc)}
    ledger={'water':(nb,),'bed':(ng,nb),'secondary':(nb,),'source':(),
            'suspended':(ng,nb),'exchange':(ng,2)}
    extras=a1_checkpoint_extra_schema(data,len(knots))
    expected=set(state)|{'ledger_'+key for key in ledger}|set(extras)|{
        'time','dt_hint','steps','rejected','input_identity','source_identity',
        'method_identity','event_identity','next_boundary_knot','a1_payload_sha256'}
    with np.load(resume,allow_pickle=False) as checkpoint:
        if set(checkpoint.files)!=expected:
            raise ValueError('Live suspension checkpoint differs from complete G2 schema')
        payload=np.asarray(checkpoint['a1_payload_sha256'])
        if (payload.shape!=() or payload.dtype.kind not in 'US' or
                payload.item()!=fields_digest({key:checkpoint[key] for key in checkpoint.files
                                               if key!='a1_payload_sha256'})):
            raise ValueError('Live suspension checkpoint payload digest differs')
        arrays,scalars=validate_checkpoint(checkpoint,state,ledger,
            input_identity=input_identity,source_identity=source_identity,
            method_identity=method_identity,events=events,
            secondary_model=int(data['secondary_flow_model']),start=start,end=end,
            extra_schema=extras,allow_nan_extras=('a1_outlet_failed_speed',))
    if int(arrays['a1_event_cursor'])!=a1_event_cursor(events,scalars['time']):
        raise ValueError('Live suspension restart event cursor differs')
    expected_hits=(knots<=float(scalars['time'])+1.e-12).astype(np.int32)
    if not np.array_equal(arrays['a1_knot_hits'],expected_hits):
        raise ValueError('Live suspension forcing-knot history differs')
    if (not np.isin(arrays['a1_normal_cache_valid'],(0,1)).all() or
            not np.isin(arrays['a1_normal_recorded'],(0,1)).all() or
            not np.array_equal(arrays['a1_normal_recorded'],arrays['a1_outlet_recorded']) or
            np.any(arrays['a1_normal_latch']) or np.any(arrays['a1_outlet_latch']) or
            np.any(arrays['a1_schedule_status']) or np.any(arrays['a1_normal_root_status']) or
            np.any(arrays['a1_normal_trace_status']) or np.any(arrays['a1_outlet_status']) or
            np.any(arrays['a1_normal_failure_latch']) or
            np.any(arrays['a1_outlet_failure_latch']) or
            np.any(arrays['a1_outlet_face_status'])):
        raise ValueError('Live suspension accepted section/schedule diagnostics are invalid')
    if scalars['steps']:
        clock=arrays['a1_schedule_clock'];stages=arrays['a1_schedule_stage_time']
        expected_stage=np.asarray([clock[0],clock[0]+.5*clock[1],
                                   clock[0]+clock[1],clock[0]+.5*clock[1]])
        if (clock[1]<=0 or abs(clock[0]+clock[1]-scalars['time'])>1.e-12 or
                np.max(np.abs(stages-expected_stage))>1.e-12 or
                not np.array_equal(arrays['a1_normal_recorded'],[1,1,1,0])):
            raise ValueError('Live suspension restart SSPRK33 stage records differ')
        tables=validate_tables(dict(data,section_control=np.zeros(nb,np.int32),
                                    sampled=np.zeros(int(data['nf']),np.int32)))
        expected_schedule=evaluate_host(tables,float(clock[0]),float(clock[1]),validated=True)
        for key in ('stage_time','hydraulic','concentration','feed'):
            if np.max(np.abs(arrays['a1_schedule_'+key]-expected_schedule[key]))>1.e-12:
                raise ValueError('Live suspension restart schedule differs: '+key)
    elif np.any(arrays['a1_normal_recorded']) or np.any(arrays['a1_outlet_recorded']):
        raise ValueError('Live suspension initial restart has nonempty stage records')
    return arrays,scalars


def prepare_live_suspension_run(data,input_path,destination,*,input_identity,events,start,end,
                                dtmax,capture_stages,resume,exchange_mode=1,vr_reference=None,
                                graded=False):
    """Freeze current numerical sources and verify G2 restart before output."""
    from . import live_suspension
    from . import a1_full_array_transaction as transaction
    from validation.gpu_fwave import a1_combined_current_adapter as ancestor
    gate,gate_sha=live_suspension.gate_for_exchange(exchange_mode,graded=graded)
    if sha(gate)!=gate_sha:
        raise ValueError('Frozen pre-GPU G2 engineering gate identity differs')
    with tempfile.TemporaryDirectory(prefix='braids_live_suspension_source_') as temp:
        archive=Path(temp)/'source'
        with live_suspension.isolated_adapter(expected_exchange=exchange_mode,
                                              graded=graded) as candidate:
            candidate.prepare_sources(archive,data)
            candidate.verify_sources(archive)
        transaction_folder=archive/'transaction_python';transaction_folder.mkdir()
        transaction_sources={}
        for name,path in a1_transaction_sources().items():
            target=transaction_folder/name;shutil.copy2(path,target)
            transaction_sources[name]=sha(path)
            if sha(target)!=transaction_sources[name]:
                raise ValueError('Live suspension transaction source changed during archive')
        files={str(path.relative_to(archive)).replace('\\','/'):sha(path)
               for path in archive.rglob('*') if path.is_file()}
        selected_variant=(GRADED_LIVE_VR_VARIANT if graded else
                          LIVE_VR_VARIANT if exchange_mode==2 else LIVE_SUSPENSION_VARIANT)
        identity=dict(schema=1,variant=selected_variant,
            source_files=files,
            transaction_sources=transaction_sources,
            runner_sha256=sha(__file__),cli_sha256=sha(Path(__file__).parent/'__main__.py'),
            adapter_sha256=sha(live_suspension.__file__),ancestor_sha256=sha(ancestor.__file__),
            transaction_sha256=sha(transaction.__file__),
            frozen_g2_gate_sha256=gate_sha,
            candidate_receipt_sha256=sha(archive/'candidate_sources.json'))
        if exchange_mode==2:
            identity['frozen_vr_reference']={key:vr_reference[key] for key in
                ('gate_sha256','native_plan_sha256','native_audit_sha256',
                 'native_executable_sha256','archive_bound')}
        write_json(archive/'runtime_manifest.json',identity)
        source_identity=sha(archive/'runtime_manifest.json')
        if exchange_mode==2:
            candidate_path=CUDA_SOURCE/('runs/g2_live_vr_graded_gpu_20261004_v3/candidate_plan.json'
                if graded else 'runs/g2_live_vr_gpu_20261004_v4/candidate_plan.json')
            if not candidate_path.is_file():
                raise ValueError('Live Van Rijn GPU source/runtime must be frozen before evolution')
            candidate=json.loads(candidate_path.read_text(encoding='utf-8'))
            if (candidate.get('status')!='candidate_frozen_before_GPU_evolution' or
                    candidate.get('gate_sha256')!=gate_sha or
                    candidate.get('source_identity')!=source_identity or
                    candidate.get('source_files')!=files or
                    candidate.get('reference')!=identity['frozen_vr_reference']):
                raise ValueError('Live Van Rijn candidate source/runtime differs from pre-run freeze')
        from validation.gpu_fwave.exchange_ledger_parallel import (
            DEFAULT_DIRECT_LIMIT,TILE,method_for_cells)
        ledger_algorithm=dict(algorithm='deterministic_cell_parallel_v1',
            dispatch=method_for_cells(int(data['nc'])),tile_cells=TILE,
            direct_limit_cells=DEFAULT_DIRECT_LIMIT)
        method=dict(variant=selected_variant,
            integrator='SSPRK33',euler_stages=3,
            euler_schedule_slots=[0,2,1],launch_mode='plain',precision='FP64',dtmax_s=dtmax,
            capture_stages=bool(capture_stages),cfl=float(data['cfl_target']),
            morfac=float(data['morphological_factor']),bedlaw=2,
            secondary_flow_model=int(data['secondary_flow_model']),exchange_mode=exchange_mode,
            settling_mode=int(data['settling_mode']),exchange_ledger=ledger_algorithm,
            source_identity=source_identity,event_identity=events.identity,
            mpm_friction_mode=int(data['mpm_friction_mode']),mpm_cf=float(data['mpm_cf']),
            frozen_g2_gate_sha256=gate_sha,
            authenticated_native_sidecar=True,auxiliary_checkpoint_schema=3,
            trial_transaction='segmented_all_reachable_v5')
        if exchange_mode==2:
            method['frozen_vr_reference']=vr_reference
        if graded:
            method.update(grain_classes=7,bed_layers=4,hiding_exposure=4,
                          bed_slope=2,graded_stratigraphy=True)
        method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        restored=None
        if resume is not None:
            restored=live_suspension_checkpoint_preflight(resume,data,
                input_identity=input_identity,source_identity=source_identity,
                method_identity=method_identity,events=events,start=start,end=end)
        destination=filesystem_path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
        source_folder=destination/'source';shutil.copytree(archive,source_folder)
        if (sha(source_folder/'runtime_manifest.json')!=source_identity or
                any(sha(source_folder/name)!=digest for name,digest in files.items())):
            raise ValueError('Live suspension source changed while archiving')
        with live_suspension.isolated_adapter(expected_exchange=exchange_mode,
                                              graded=graded) as candidate:
            candidate.verify_sources(source_folder)
        a1_verify_transaction_sources(source_folder)
    return source_identity,method,restored


@contextmanager
def live_suspension_constructors(folder,data,bindings,*,exchange_mode=1,
                                 graded=False):
    from . import live_suspension
    from . import a1_full_array_transaction as transaction
    a1_verify_transaction_sources(folder)
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Live suspension GPU constructor requires the main worker thread')
    if not _COMBINED_BINDING_LOCK.acquire(blocking=False):
        raise ValueError('Concurrent live GPU construction is unsupported')
    try:
        with live_suspension.constructors(folder,data,expected_exchange=exchange_mode,
                                          graded=graded) as Solver:
            PublicSolver=transaction.wrap_solver(Solver)
            bindings.append(dict(kind='live_normal_critical_van_rijn_graded' if graded else
                                  'live_normal_critical_van_rijn_mpm' if exchange_mode==2
                                  else 'live_normal_critical_suspension_mpm',
                source_receipt_sha256=sha(Path(folder)/'candidate_sources.json'),
                runtime_manifest_sha256=sha(Path(folder)/'runtime_manifest.json'),
                graph_supported=False,complete_trial_transaction='segmented_all_reachable_v5'))
            yield PublicSolver,None
    finally:_COMBINED_BINDING_LOCK.release()


def target_b_checkpoint_preflight(resume,data,*,input_identity,source_identity,
                                  method_identity,events,start,end):
    """Authenticate exact live bedload/adaptation checkpoint before output/CUDA."""
    from validation.gpu_fwave.boundary_schedule import evaluate_host,validate_tables
    from validation.gpu_fwave.coupled_solver import normalized
    data=normalized(data)
    knots=live_internal_knots(events,start,end)
    nc=int(data['nc']);nb=int(data['nb']);ng=int(data['nfrac'])
    state={'q':np.asarray(data['q']).shape,'layers':np.asarray(data['layer_mass']).shape,
           'secondary':(nc,)}
    ledger={'water':(nb,),'bed':(ng,nb),'secondary':(nb,),'source':()}
    extras=a1_checkpoint_extra_schema(data,len(knots))
    adaptation=bool(int(data['bedload_adaptation']))
    if adaptation:
        extras.update(bedload_adaptation_equilibrium=((2,ng,nc),np.float64),
            bedload_adaptation_capacity=((2,ng,nc),np.float64),
            bedload_adaptation_iterations=((),np.int32),
            bedload_adaptation_residual=((),np.float64))
    expected=set(state)|{'ledger_'+key for key in ledger}|set(extras)|{
        'time','dt_hint','steps','rejected','input_identity','source_identity',
        'method_identity','event_identity','next_boundary_knot','a1_payload_sha256'}
    with np.load(resume,allow_pickle=False) as checkpoint:
        if set(checkpoint.files)!=expected:
            raise ValueError('Target-B checkpoint differs from complete OFF/ON schema')
        payload=np.asarray(checkpoint['a1_payload_sha256'])
        if (payload.shape!=() or payload.dtype.kind not in 'US' or
                payload.item()!=fields_digest({key:checkpoint[key] for key in checkpoint.files
                                               if key!='a1_payload_sha256'})):
            raise ValueError('Target-B checkpoint payload digest differs')
        arrays,scalars=validate_checkpoint(checkpoint,state,ledger,
            input_identity=input_identity,source_identity=source_identity,
            method_identity=method_identity,events=events,secondary_model=0,
            start=start,end=end,extra_schema=extras,
            allow_nan_extras=('a1_outlet_failed_speed',))
    if int(arrays['a1_event_cursor'])!=a1_event_cursor(events,scalars['time']):
        raise ValueError('Target-B restart event cursor differs')
    expected_hits=(knots<=float(scalars['time'])+1.e-12).astype(np.int32)
    if not np.array_equal(arrays['a1_knot_hits'],expected_hits):
        raise ValueError('Target-B restart forcing-knot history differs')
    if (not np.isin(arrays['a1_normal_cache_valid'],(0,1)).all() or
            not np.isin(arrays['a1_normal_recorded'],(0,1)).all() or
            not np.array_equal(arrays['a1_normal_recorded'],arrays['a1_outlet_recorded']) or
            np.any(arrays['a1_normal_latch']) or np.any(arrays['a1_outlet_latch']) or
            np.any(arrays['a1_schedule_status']) or np.any(arrays['a1_normal_root_status']) or
            np.any(arrays['a1_normal_trace_status']) or
            np.any(arrays['a1_outlet_status']) or
            np.any(arrays['a1_normal_failure_latch']) or
            np.any(arrays['a1_outlet_failure_latch']) or
            np.any(arrays['a1_outlet_face_status'])):
        raise ValueError('Target-B accepted section/schedule diagnostics invalid')
    if (np.any(arrays['secondary']) or
            np.any(arrays['ledger_secondary']) or np.any(arrays['ledger_source'])):
        raise ValueError('Target-B bedload-only restart has hidden suspended/secondary inventory')
    if adaptation:
        if (int(arrays['bedload_adaptation_iterations'])<0 or
                float(arrays['bedload_adaptation_residual'])<0):
            raise ValueError('Target-B derived adaptation diagnostics are invalid')
    if scalars['steps']:
        clock=arrays['a1_schedule_clock'];stages=arrays['a1_schedule_stage_time']
        expected_stage=np.asarray([clock[0],clock[0]+.5*clock[1],
                                   clock[0]+clock[1],clock[0]+.5*clock[1]])
        if (clock[1]<=0 or abs(clock[0]+clock[1]-scalars['time'])>1.e-12 or
                np.max(np.abs(stages-expected_stage))>1.e-12 or
                not np.array_equal(arrays['a1_normal_recorded'],[1,1,1,0])):
            raise ValueError('Target-B restart SSPRK33 stage records differ')
        tables=validate_tables(dict(data,section_control=np.zeros(nb,np.int32),
                                    sampled=np.zeros(int(data['nf']),np.int32)))
        expected_schedule=evaluate_host(tables,float(clock[0]),float(clock[1]),validated=True)
        for key in ('stage_time','hydraulic','concentration','feed'):
            if np.max(np.abs(arrays['a1_schedule_'+key]-expected_schedule[key]))>1.e-12:
                raise ValueError('Target-B restart schedule differs: '+key)
    elif np.any(arrays['a1_normal_recorded']) or np.any(arrays['a1_outlet_recorded']):
        raise ValueError('Target-B initial restart has nonempty stage records')
    return arrays,scalars


def prepare_target_b_run(data,input_path,destination,*,input_identity,events,start,end,
                         dtmax,capture_stages,resume,reference):
    """Freeze exact source/runtime and preflight restart before output/CUDA."""
    from . import live_bedload_adaptation as live
    from . import a1_full_array_transaction as transaction
    from validation.gpu_fwave import a1_combined_current_adapter as ancestor
    from validation.g2_target_b import gpu_scope
    if sha(gpu_scope.GATE)!=gpu_scope.GATE_SHA:
        raise ValueError('Target-B frozen GPU gate identity differs')
    with tempfile.TemporaryDirectory(prefix='braids_target_b_source_') as temp:
        archive=Path(temp)/'source'
        with live.isolated_adapter() as candidate:
            candidate.prepare_sources(archive,data)
            candidate.verify_sources(archive)
        transaction_folder=archive/'transaction_python';transaction_folder.mkdir()
        transaction_sources={}
        for name,path in a1_transaction_sources().items():
            target=transaction_folder/name;shutil.copy2(path,target)
            transaction_sources[name]=sha(path)
            if sha(target)!=transaction_sources[name]:
                raise ValueError('Target-B transaction source changed during archive')
        files={str(path.relative_to(archive)).replace('\\','/'):sha(path)
               for path in archive.rglob('*') if path.is_file()}
        reference_core={key:reference[key] for key in
            ('gate_sha256','native_plan_sha256','native_audit_sha256',
             'native_executable_sha256','archive_bound')}
        identity=dict(schema=1,variant=TARGET_B_ADAPTATION_VARIANT,
            source_files=files,transaction_sources=transaction_sources,
            runner_sha256=sha(__file__),cli_sha256=sha(Path(__file__).parent/'__main__.py'),
            adapter_sha256=sha(live.__file__),scope_sha256=sha(gpu_scope.__file__),
            ancestor_sha256=sha(ancestor.__file__),
            transaction_sha256=sha(transaction.__file__),
            frozen_target_b_gate_sha256=gpu_scope.GATE_SHA,
            candidate_receipt_sha256=sha(archive/'candidate_sources.json'),
            frozen_target_b_reference=reference_core)
        write_json(archive/'runtime_manifest.json',identity)
        source_identity=sha(archive/'runtime_manifest.json')
        candidate_path=ROOT/'validation/g2_target_b/runs/target_b_gpu_20261004_v3/candidate_plan.json'
        if not candidate_path.is_file():
            raise ValueError('Target-B GPU source/runtime must be frozen before evolution')
        candidate_plan=json.loads(candidate_path.read_text(encoding='utf-8'))
        if (candidate_plan.get('status')!='candidate_frozen_before_GPU_evolution' or
                candidate_plan.get('gate_sha256')!=gpu_scope.GATE_SHA or
                candidate_plan.get('source_identity')!=source_identity or
                candidate_plan.get('source_files')!=files or
                candidate_plan.get('reference')!=reference_core):
            raise ValueError('Target-B source/runtime differs from pre-run freeze')
        enabled=bool(int(data['bedload_adaptation']))
        method=dict(variant=TARGET_B_ADAPTATION_VARIANT,
            integrator='SSPRK33',euler_stages=3,euler_schedule_slots=[0,2,1],
            launch_mode='plain',precision='FP64',dtmax_s=dtmax,
            capture_stages=bool(capture_stages),cfl=float(data['cfl_target']),
            morfac=float(data['morphological_factor']),bedlaw=2,
            secondary_flow_model=0,exchange_mode=0,
            settling_mode=int(data['settling_mode']),
            exchange_ledger=dict(algorithm='none'),
            source_identity=source_identity,event_identity=events.identity,
            mpm_friction_mode=int(data['mpm_friction_mode']),mpm_cf=float(data['mpm_cf']),
            grain_classes=7,bed_layers=4,hiding_exposure=4,bed_slope=2,
            bedload_adaptation=dict(enabled=enabled,
                length_diameters=float(data['bedload_adaptation_length_diameters']) if enabled else None,
                closure='stage_local_classwise_streamline_jacobi' if enabled else 'local_equilibrium',
                derived_capacity_not_checkpointed=True),
            frozen_target_b_gate_sha256=gpu_scope.GATE_SHA,
            frozen_target_b_reference=reference,
            authenticated_native_sidecar=True,auxiliary_checkpoint_schema=4,
            trial_transaction='segmented_all_reachable_v5')
        method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        restored=None
        if resume is not None:
            restored=target_b_checkpoint_preflight(resume,data,
                input_identity=input_identity,source_identity=source_identity,
                method_identity=method_identity,events=events,start=start,end=end)
        destination=filesystem_path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
        source_folder=destination/'source';shutil.copytree(archive,source_folder)
        if (sha(source_folder/'runtime_manifest.json')!=source_identity or
                any(sha(source_folder/name)!=digest for name,digest in files.items())):
            raise ValueError('Target-B source changed during archive')
        with live.isolated_adapter() as candidate:
            candidate.verify_sources(source_folder)
        a1_verify_transaction_sources(source_folder)
    return source_identity,method,restored


def prepare_general_bedload_run(data,input_path,destination,*,input_identity,
                                events,start,end,dtmax,capture_stages,resume,
                                reference,source_archive=None):
    """Freeze a portable prepared-input contract, not a fixture identity.

    The original target-B source/gate path above remains untouched.  This
    path archives exactly the same private numerical kernels but derives the
    method/input identity from a regular checked preparation receipt and its
    decoded controls.  All source freezing and restart checks precede CUDA.
    """
    from . import live_bedload_adaptation as live
    from . import a1_full_array_transaction as transaction
    from . import graded_bedload_prebuilt as prebuilt
    if reference['preparation_receipt_sha256'] != input_identity or reference['archive_bound']:
        raise ValueError('General graded bedload prepared contract differs')
    with tempfile.TemporaryDirectory(prefix='braids_graded_bedload_source_') as temp:
        archive=Path(temp)/'source'
        if source_archive is None:
            from validation.gpu_fwave import a1_combined_current_adapter as ancestor
            with live.isolated_adapter(scope_validator=live.validate_general_scope) as candidate:
                candidate.prepare_sources(archive,data)
                candidate.verify_sources(archive)
            transaction_folder=archive/'transaction_python';transaction_folder.mkdir()
            transaction_sources={}
            for name,path in a1_transaction_sources().items():
                target=transaction_folder/name;shutil.copy2(path,target)
                transaction_sources[name]=sha(path)
                if sha(target)!=transaction_sources[name]:
                    raise ValueError('General graded bedload transaction source changed')
            files={str(path.relative_to(archive)).replace('\\','/'):sha(path)
                   for path in archive.rglob('*') if path.is_file()}
            identity=dict(schema=1,variant=GRADED_BEDLOAD_GENERAL_VARIANT,
                source_files=files,transaction_sources=transaction_sources,
                runner_sha256=sha(__file__),
                adapter_sha256=sha(live.__file__),
                prebuilt_adapter_sha256=sha(prebuilt.__file__),
                ancestor_sha256=sha(ancestor.__file__),
                transaction_sha256=sha(transaction.__file__),
                preparation_contract_schema=1,
                candidate_receipt_sha256=sha(archive/'candidate_sources.json'))
            write_json(archive/'runtime_manifest.json',identity)
        else:
            bundle=filesystem_path(source_archive).resolve()
            identity=json.loads((bundle/'runtime_manifest.json').read_text())
            if (identity.get('schema')!=1 or
                    identity.get('variant')!=GRADED_BEDLOAD_GENERAL_VARIANT or
                    identity.get('preparation_contract_schema')!=1 or
                    identity.get('runner_sha256')!=sha(__file__) or
                    identity.get('adapter_sha256')!=sha(live.__file__) or
                    identity.get('prebuilt_adapter_sha256')!=sha(prebuilt.__file__) or
                    identity.get('ancestor_sha256')!=sha(
                        CUDA_SOURCE/'a1_combined_current_adapter.py') or
                    identity.get('transaction_sha256')!=sha(transaction.__file__) or
                    identity.get('candidate_receipt_sha256')!=sha(
                        bundle/'candidate_sources.json')):
                raise ValueError('General graded bedload source bundle identity differs')
            files=identity.get('source_files',{})
            actual={str(path.relative_to(bundle)).replace('\\','/')
                    for path in bundle.rglob('*') if path.is_file()}
            if actual!=set(files)|{'runtime_manifest.json'}:
                raise ValueError('General graded bedload source bundle inventory differs')
            for name,digest in files.items():
                path=(bundle/name).resolve()
                if not path.is_relative_to(bundle) or sha(path)!=digest:
                    raise ValueError('General graded bedload source bundle file differs: '+name)
            for name,digest in identity.get('transaction_sources',{}).items():
                if files.get('transaction_python/'+name)!=digest:
                    raise ValueError('General graded bedload transaction bundle differs: '+name)
            shutil.copytree(bundle,archive)
            if any(sha(archive/name)!=digest for name,digest in files.items()):
                raise ValueError('General graded bedload source bundle changed during copy')
        source_identity=sha(archive/'runtime_manifest.json')
        enabled=bool(int(data['bedload_adaptation']))
        method=dict(variant=GRADED_BEDLOAD_GENERAL_VARIANT,
            integrator='SSPRK33',euler_stages=3,euler_schedule_slots=[0,2,1],
            launch_mode='plain',precision='FP64',dtmax_s=dtmax,
            capture_stages=bool(capture_stages),cfl=float(data['cfl_target']),
            morfac=float(data['morphological_factor']),bedlaw=2,
            secondary_flow_model=0,exchange_mode=0,
            settling_mode=int(data['settling_mode']),
            exchange_ledger=dict(algorithm='none'),
            source_identity=source_identity,event_identity=events.identity,
            mpm_friction_mode=int(data['mpm_friction_mode']),
            mpm_cf=float(data['mpm_cf']),grain_classes=int(data['nfrac']),
            bed_layers=int(data['nbed_layers']),
            hiding_exposure=int(data['hiding_exposure']),
            bed_slope=int(data['bed_slope']),
            bedload_adaptation=dict(enabled=enabled,
                length_diameters=float(data['bedload_adaptation_length_diameters'])
                    if enabled else None,
                closure='stage_local_classwise_streamline_jacobi'
                    if enabled else 'local_equilibrium',
                derived_capacity_not_checkpointed=True),
            preparation_contract=reference,
            auxiliary_checkpoint_schema=4,
            trial_transaction='segmented_all_reachable_v5')
        method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        restored=None
        if resume is not None:
            restored=target_b_checkpoint_preflight(resume,data,
                input_identity=input_identity,source_identity=source_identity,
                method_identity=method_identity,events=events,start=start,end=end)
        destination=filesystem_path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
        source_folder=destination/'source';shutil.copytree(archive,source_folder)
        if (sha(source_folder/'runtime_manifest.json')!=source_identity or
                any(sha(source_folder/name)!=digest for name,digest in files.items())):
            raise ValueError('General graded bedload source changed during archive')
        if source_archive is None:
            with live.isolated_adapter(scope_validator=live.validate_general_scope) as candidate:
                candidate.verify_sources(source_folder)
        prebuilt.verify_archive(source_folder)
    return source_identity,method,restored


@contextmanager
def target_b_constructors(folder,data,bindings,*,general=False):
    from . import live_bedload_adaptation as live
    from . import a1_full_array_transaction as transaction
    if general:
        from . import graded_bedload_prebuilt as prebuilt
        prebuilt.verify_archive(folder)
    else:
        a1_verify_transaction_sources(folder)
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Target-B GPU constructor requires main worker thread')
    if not _COMBINED_BINDING_LOCK.acquire(blocking=False):
        raise ValueError('Concurrent live GPU construction is unsupported')
    try:
        validator=live.validate_general_scope if general else None
        selected_gate=sha(live.__file__) if general else None
        with live.constructors(folder,data,scope_validator=validator,
                               selected_gate_sha256=selected_gate) as Solver:
            PublicSolver=transaction.wrap_solver(Solver)
            bindings.append(dict(kind='live_normal_critical_graded_bedload_general'
                if general else 'live_normal_critical_graded_adaptation',
                source_receipt_sha256=sha(Path(folder)/'candidate_sources.json'),
                runtime_manifest_sha256=sha(Path(folder)/'runtime_manifest.json'),
                graph_supported=False,
                complete_trial_transaction='segmented_all_reachable_v5'))
            yield PublicSolver,None
    finally:_COMBINED_BINDING_LOCK.release()


def _combined_live_sources(baseline):
    """Pin the imported live Python as well as the frozen numerical archive."""
    names={CUDA_SOURCE/name for name in baseline if name.endswith('.py') and not name.startswith('backend_')
           and (CUDA_SOURCE/name).is_file()}
    names.update(Path(__file__).parent/name.removeprefix('backend_') for name in baseline
                 if name.startswith('backend_') and name.endswith('.py')
                 and (Path(__file__).parent/name.removeprefix('backend_')).is_file())
    names.update(CUDA_SOURCE/name for name in (
        'boundary_schedule.py','curved_wetdry_worker.py','scheduled_coupled.py',
        'scheduled_coupled_guarded.py','normal_coupled.py','normal_outlet_coupled.py',
        'normal_outlet_boundary.py','normal_outlet_cases.py','live_normal_boundary.py',
        'outlet_boundary.py','live_critical_outlet.py','normal_coupled_oracle.py',
        'reconstruction_oracle.py','hydraulic_reference.py',
        'suspended_faces.py','suspended_advection.py','suspended_exchange.py',
        'suspension_resident.py','sediment_stratigraphy.py','exchange_ledger_parallel.py'))
    names.add(Path(__file__).resolve())
    for name,module in tuple(sys.modules.items()):
        if not name.startswith(('validation.gpu_fwave.','braids_gpu.','braids_core.')):continue
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT):names.add(path)
    for path in names:
        if not path.is_file():raise ValueError('Missing combined GPU runtime dependency: '+str(path))
    return {str(path.resolve()):sha(path) for path in sorted(names)}


def check_combined_imports(source_folder,live):
    """Reject a new live BRAIDS import that was absent from the frozen receipt."""
    source_folder=filesystem_path(source_folder).resolve()
    for name,module in tuple(sys.modules.items()):
        if not name.startswith(('validation.gpu_fwave.','braids_gpu.','braids_core.')):continue
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix!='.py' or not path.is_relative_to(ROOT) or path.is_relative_to(source_folder):continue
        if str(path) not in live:raise ValueError('Unpinned combined GPU module import: '+name)
        if sha(path)!=live[str(path)]:raise ValueError('Combined GPU module changed during run: '+name)
    return True


def combined_module_receipt(source_folder,live):
    check_combined_imports(source_folder,live)
    source_folder=filesystem_path(source_folder).resolve();modules={}
    for name,module in sorted(tuple(sys.modules.items())):
        if not name.startswith(('validation.gpu_fwave.','braids_gpu.','braids_core.')):continue
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        modules[name]=dict(path=str(path),sha256=sha(path),
            origin='run_source' if path.is_relative_to(source_folder) else 'pinned_live')
    return modules


def freeze_combined_sources(destination):
    """Copy the qualified base, generate the bounded adapter, and pin live imports."""
    from validation.gpu_fwave import curved_wetdry_worker as qualified
    from validation.gpu_fwave import normal_outlet_coupled as adapter
    if sha(qualified.QUALIFIED_PLAN)!=qualified.QUALIFIED_SHA:
        raise ValueError('Qualified GPU base plan changed')
    plan=json.loads(qualified.QUALIFIED_PLAN.read_text())
    origin=Path(plan['output_path'])/'source'
    baseline=json.loads((origin/'manifest.json').read_text())
    qualified.checked_hashes({str(origin/name):digest for name,digest in baseline.items()})
    expected=dict(plan['baseline_source_sha256']);expected.update(plan['candidate_source_sha256'])
    expected.update({name:row['generated_file_sha256'] for name,row in plan['repair_specs'].items()})
    if any(baseline.get(name)!=digest for name,digest in expected.items()):
        raise ValueError('Qualified GPU base source manifest differs')
    destination=filesystem_path(destination).resolve();shutil.copytree(origin,destination)
    adapter.prepare_sources(destination,baseline)
    live=_combined_live_sources(baseline)
    live_archive=destination/'live_python';live_archive.mkdir()
    archived={}
    for path,digest in live.items():
        source=Path(path);label=str(source.relative_to(ROOT)).replace('\\','__').replace('/','__')
        target=live_archive/label;shutil.copy2(source,target)
        if sha(target)!=digest:raise ValueError('Live GPU dependency changed while freezing: '+path)
        archived[label]=digest
    source_files={str(path.relative_to(destination)).replace('\\','/'):sha(path)
                  for path in destination.rglob('*') if path.is_file() and path.suffix in ('.py','.cu')}
    identity=dict(schema=1,variant='scheduled_normal9_critical4',base_manifest=baseline,
                  source_files=source_files,live_python=archived)
    identity_digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(destination/'runtime_manifest.json',dict(**identity,identity_sha256=identity_digest,
        live_paths=live,base_archive=str(origin),adapter_receipt_sha256=sha(destination/'normal_outlet_sources.json')))
    return baseline,live,identity_digest


def qualified_curved_scope(data,input_path,*,start,end,interval,dtmax):
    """Admit only the exact native bar256 fixture used by the frozen V3 proof."""
    from validation.gpu_fwave import curved_wetdry_worker as qualified
    plan_path=CUDA_SOURCE/'reports/curved_wetdry_execution_20261002_v3/bar256_gpu_plan.json'
    expected='9287692af58a3cc746955a23251a9ac39f8ee2f09012b97b54339296a1bd91af'
    if sha(plan_path)!=expected:raise ValueError('Qualified curved wet/dry plan changed')
    plan=json.loads(plan_path.read_text())
    original=Path(plan['input_folder']).resolve()
    supplied=filesystem_path(input_path).resolve()
    frozen_receipt=plan['authenticated_files'][str(original/'preparation_receipt.json')]
    if sha(original/'preparation_receipt.json')!=frozen_receipt or sha(supplied/'preparation_receipt.json')!=frozen_receipt:
        raise ValueError('Qualified curved wet/dry preparation identity differs')
    if (start,end,interval,dtmax)!=(0.,4.,.25,.002):
        raise ValueError('Qualified curved wet/dry requires its declared 0–4 s, 0.25 s output, 0.002 s trial clocks')
    if plan['qualified_source_plan_sha256']!=qualified.QUALIFIED_SHA or plan['filter_controls']!=qualified.FILTER:
        raise ValueError('Qualified curved wet/dry method controls changed')
    initial_path=Path(plan['case']['input']).parent/'initial_analysis.npz'
    if sha(initial_path)!=plan['case']['input_files'][str(initial_path)]:
        raise ValueError('Qualified curved wet/dry initial analysis changed')
    qualified.snapshot_identity(data,plan['case'],plan['controls'])
    return expected,plan


def freeze_qualified_curved_sources(destination):
    """Pin the complete vetted numerical combination, not the stock live kernels."""
    from validation.gpu_fwave import curved_wetdry_worker as qualified
    if sha(qualified.QUALIFIED_PLAN)!=qualified.QUALIFIED_SHA:
        raise ValueError('Qualified GPU base plan changed')
    plan=json.loads(qualified.QUALIFIED_PLAN.read_text())
    origin=Path(plan['output_path'])/'source'
    baseline=json.loads((origin/'manifest.json').read_text())
    qualified.checked_hashes({str(origin/name):digest for name,digest in baseline.items()})
    expected=dict(plan['baseline_source_sha256']);expected.update(plan['candidate_source_sha256'])
    expected.update({name:row['generated_file_sha256'] for name,row in plan['repair_specs'].items()})
    if any(baseline.get(name)!=digest for name,digest in expected.items()):
        raise ValueError('Qualified GPU base source manifest differs')
    destination=filesystem_path(destination).resolve();shutil.copytree(origin,destination)
    live=_combined_live_sources(baseline)
    live_archive=destination/'live_python';live_archive.mkdir()
    archived={}
    for path,digest in live.items():
        source=Path(path);label=str(source.relative_to(ROOT)).replace('\\','__').replace('/','__')
        target=live_archive/label;shutil.copy2(source,target)
        if sha(target)!=digest:raise ValueError('Live GPU dependency changed while freezing: '+path)
        archived[label]=digest
    source_files={str(path.relative_to(destination)).replace('\\','/'):sha(path)
                  for path in destination.rglob('*') if path.is_file() and path.suffix in ('.py','.cu')}
    identity=dict(schema=1,variant='qualified_curved_wetdry',base_manifest=baseline,
                  source_files=source_files,live_python=archived,
                  filter_controls=qualified.FILTER,effective_outlet_extension=1)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(destination/'runtime_manifest.json',dict(**identity,identity_sha256=digest,
        live_paths=live,base_archive=str(origin),qualified_plan_sha256=qualified.QUALIFIED_SHA))
    return baseline,live,digest


@contextmanager
def combined_constructors(folder,baseline,live,bindings):
    """Bridge the old worker's live-runner hash without altering frozen sources."""
    from validation.gpu_fwave import curved_wetdry_worker as qualified
    from validation.gpu_fwave import normal_outlet_coupled as adapter
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Combined GPU constructor requires the main worker thread')
    if not _COMBINED_BINDING_LOCK.acquire(blocking=False):
        raise ValueError('Concurrent combined GPU construction is unsupported')
    previous=qualified.source_identity
    folder=filesystem_path(folder).resolve()
    def pinned_identity():
        qualified.checked_hashes({str(folder/name):digest for name,digest in baseline.items()})
        qualified.checked_hashes(live)
        return folder,baseline,live
    try:
        qualified.source_identity=pinned_identity
        with adapter.constructors(folder,baseline,bindings) as constructors:
            yield constructors
        pinned_identity()
    finally:
        qualified.source_identity=previous
        _COMBINED_BINDING_LOCK.release()


@contextmanager
def qualified_curved_constructors(folder,baseline,live,bindings):
    """Scope the archived constructor swaps to one authenticated main-process run."""
    from validation.gpu_fwave import curved_wetdry_worker as qualified
    from validation.gpu_fwave import coupled_solver
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Qualified GPU constructor requires the main worker thread')
    if not _COMBINED_BINDING_LOCK.acquire(blocking=False):
        raise ValueError('Concurrent qualified GPU construction is unsupported')
    folder=filesystem_path(folder).resolve();previous=qualified.source_identity
    old_sediment=coupled_solver.ResidentSedimentTransport
    archive_name='validation.gpu_fwave._wetdry_sediment_archive'
    if archive_name in sys.modules:
        _COMBINED_BINDING_LOCK.release()
        raise ValueError('Private archived sediment module already registered')
    def pinned_identity():
        qualified.checked_hashes({str(folder/name):digest for name,digest in baseline.items()})
        qualified.checked_hashes(live)
        return folder,baseline,live
    try:
        qualified.source_identity=pinned_identity
        pinned_identity()
        sediment_python=folder/'sediment_transport.py';sediment_cuda=folder/'sediment_transport.cu'
        spec=importlib.util.spec_from_file_location(archive_name,sediment_python)
        if spec is None or spec.loader is None:raise ValueError('Archived sediment Python cannot be loaded')
        module=importlib.util.module_from_spec(spec);sys.modules[archive_name]=module
        spec.loader.exec_module(module)
        coupled_solver.ResidentSedimentTransport=module.ResidentSedimentTransport
        bindings.append(dict(kind='archived_sediment_python',executed_python=str(sediment_python),
            python_sha256=sha(sediment_python),executed_cuda=str(sediment_cuda),
            cuda_sha256=sha(sediment_cuda),before_graph_capture=True))
        with qualified.constructors(folder,baseline,bindings):
            with stock_constructors() as constructors:yield constructors
        pinned_identity()
    finally:
        coupled_solver.ResidentSedimentTransport=old_sediment
        sys.modules.pop(archive_name,None)
        qualified.source_identity=previous
        _COMBINED_BINDING_LOCK.release()


@contextmanager
def stock_constructors():
    from validation.gpu_fwave.coupled_solver import CoupledSolver
    from validation.gpu_fwave.coupled_graph import CoupledTransportGraph
    yield CoupledSolver,CoupledTransportGraph


def observed_combined_solver(parent):
    """Optional GPU-resident four-stage taps around the selected combined Euler."""
    class Observed(parent):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);cp=self.cp
            shapes=dict(q=(3,self.nc),layers=self.state['layers'].shape,bed=(self.nc,),trace=(10,self.nf),
                raw_fl=(3,self.nf),fl=(3,self.nf),bed_raw=(self.ng,self.nf),bed_fl=(self.ng,self.nf),
                traction=(2,self.nf),theta=(self.nc,),bed_theta=(self.ng,self.nc),
                hydraulic=(2,self.nb),sediment_hydraulic=(2,self.nb),feed=(self.ng,self.nb),
                water_increment=(self.nb,),bed_increment=(self.ng,self.nb))
            self.observed={key:cp.empty((4,*shape),dtype=cp.float64) for key,shape in shapes.items()}
            self.sed_observed={key:cp.empty((4,*self.sediment.buffers[key].shape),dtype=cp.float64)
                for key in ('capacity','availability','downslope')}
        def euler(self,state,dt,current):
            i=self._schedule_stage;cp=self.cp
            cp.copyto(self.observed['q'][i],state['q']);cp.copyto(self.observed['layers'][i],state['layers'])
            result=super().euler(state,dt,current)
            fields=dict(bed=self.reconstruction.geometry['bed'],trace=self.reconstruction.buffers['trace'],
                raw_fl=self.flux.buffers['raw_fl'],fl=self.flux.buffers['fl'],
                bed_raw=self.sediment.buffers['raw'],bed_fl=self.exner.buffers['flux'],
                traction=self.secondary.buffers['traction'],theta=self.flux.buffers['theta'],
                bed_theta=self.exner.buffers['theta'],hydraulic=self.boundary.arrays['forcing'],
                sediment_hydraulic=self.sediment.geometry['bc_hydraulic'],feed=self.sediment.geometry['feed'],
                water_increment=result['ledger']['water'],bed_increment=result['ledger']['bed'])
            for key,value in fields.items():cp.copyto(self.observed[key][i],value)
            for key,value in self.sed_observed.items():cp.copyto(value[i],self.sediment.buffers[key])
            return result
    return Observed


def combined_stage_fields(solver):
    return {**{key:solver.cp.asnumpy(value) for key,value in solver.observed.items()},
            **{'sediment_'+key:solver.cp.asnumpy(value) for key,value in solver.sed_observed.items()},
            **{'normal_'+key:solver.cp.asnumpy(value) for key,value in solver.boundary.normal_records.items()},
            **{'outlet_'+key:solver.cp.asnumpy(value) for key,value in solver.boundary.outlet_records.items()}}


def freeze_sources(destination,*,distributed_bank=False,bedload_adaptation=False):
    destination=filesystem_path(destination);destination.mkdir(exist_ok=False)
    names=('augmented_fwave.cu','resident_update.cu','reconstruction.cu','boundaries.cu','time_integration.cu',
           'coupled_time.cu','boundary_schedule.cu','sediment_transport.cu','secondary_transport.cu','secondary_curvature.cu',
           'sediment_stratigraphy.cu','suspended_faces.cu','suspended_advection.cu','suspended_exchange.cu',
           'suspension_resident.cu','exchange_ledger_parallel.cu','exner_update.cu','bedform_roughness.cu',
           'momentum_diffusion.cu','momentum_geometry.cu','avalanche_proposal.cu','surface_transfer.cu',
           'toe_supply.cu','bank_undercut.cu','toe_face_diagnostic.cu')
    if distributed_bank:names+=('distributed_bank.cu',)
    if bedload_adaptation:names+=('bedload_adaptation.cu',)
    manifest={}
    for name in names:
        source=CUDA_SOURCE/name
        if not source.is_file():raise ValueError('Missing GPU numerical source: '+name)
        shutil.copy2(source,destination/name);manifest[name]=sha(source)
    runtime_modules=('adaptive_hydraulics.py','boundaries.py','boundary_events.py','boundary_schedule.py',
        'scheduled_suspension_inlet.py','coupled_reference.py','coupled_solver.py','coupled_time.py','coupled_graph.py','graph_execution.py',
        'spatial_material.py',
        'exner_update.py','hydraulic_pipeline.py','momentum_diffusion.py','momentum_geometry.py','reconstruction.py','resident_update.py',
        'secondary_curvature.py','secondary_transport.py','sediment_transport.py','sediment_stratigraphy.py',
        'bedform_roughness.py','avalanche_operator.py','avalanche_proposal.py','surface_transfer.py',
        'toe_supply.py','bank_undercut.py','toe_face_diagnostic.py',
        'suspended_faces.py','suspended_advection.py','suspended_exchange.py','suspension_resident.py',
        'exchange_ledger_parallel.py','time_integration.py')
    if distributed_bank:runtime_modules+=('distributed_bank.py',)
    if bedload_adaptation:runtime_modules+=('bedload_adaptation.py',)
    for source in [CUDA_SOURCE/name for name in runtime_modules]+list(Path(__file__).parent.glob('*.py')):
        name=('backend_' if source.parent==Path(__file__).parent else '')+source.name
        shutil.copy2(source,destination/name);manifest[name]=sha(source)
    write_json(destination/'manifest.json',manifest)
    return manifest


def stage_long_execution_sources(source_folder, stack, *, legacy_limit=260):
    """Give legacy readers a short authenticated copy of a long source tree.

    The complete source snapshot remains in the run output and owns the
    persisted source identity.  This copy is used only for source-text reads
    by CUDA constructors which internally coerce paths back to ``Path``.
    """
    source_folder=filesystem_path(source_folder).resolve()
    probe=source_folder/'reconstruction.cu'
    if len(str(probe))<legacy_limit:
        return source_folder,dict(policy='authenticated_short_stage_v1',used=False,
            published_source=str(source_folder),published_probe_characters=len(str(probe)),
            staged_source_persisted=False,files_verified=0)
    inventory={}
    for path in source_folder.rglob('*'):
        if path.is_symlink():
            raise ValueError('Published GPU source snapshot contains a link')
        if path.is_file():
            inventory[path.relative_to(source_folder).as_posix()]=sha(path)
    if not inventory:
        raise ValueError('Published GPU source snapshot is empty')
    temporary=stack.enter_context(tempfile.TemporaryDirectory(
        prefix='braids_gpu_execution_source_'))
    staged=Path(temporary)/'source'
    shutil.copytree(source_folder,staged)
    copied={path.relative_to(staged).as_posix():sha(path)
            for path in staged.rglob('*') if path.is_file()}
    if copied!=inventory or any(path.is_symlink() for path in staged.rglob('*')):
        raise ValueError('Short GPU execution-source stage differs from published snapshot')
    return staged,dict(policy='authenticated_short_stage_v1',used=True,
        published_source=str(source_folder),published_probe_characters=len(str(probe)),
        staged_source_persisted=False,files_verified=len(inventory))


def time_controls(data,until,output_every,dt_max):
    start=float(data.get('time',0.))
    end=float(data['endtime'] if until is None else until)
    interval=float(data['output_dt'] if output_every is None else output_every)
    dt=float(data['dtmax'] if dt_max is None else dt_max)
    if not np.isfinite([start,end,interval,dt]).all() or start<0 or end<=start or min(interval,dt)<=0:
        raise ValueError('Require finite end > start and positive output interval / dt maximum')
    return start,end,interval,dt


def validate_checkpoint(checkpoint,state_shapes,ledger_shapes,*,input_identity,source_identity,start,end,
                        method_identity=None,events=None,secondary_model=None,extra_schema=None,
                        allow_nan_extras=()):
    """Validate every restart value before mutating any accepted device state."""
    arrays={}
    for key,shape in {**state_shapes,**{'ledger_'+k:v for k,v in ledger_shapes.items()}}.items():
        if key not in checkpoint:raise ValueError('Missing restart array '+key)
        value=np.asarray(checkpoint[key])
        if value.shape!=shape or value.dtype!=np.float64 or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError('Invalid restart array '+key)
        arrays[key]=value
    for key,(shape,dtype) in (extra_schema or {}).items():
        if key not in checkpoint:raise ValueError('Missing restart array '+key)
        value=np.asarray(checkpoint[key])
        finite=(not np.isinf(value).any() if key in allow_nan_extras else np.isfinite(value).all())
        if (value.shape!=shape or value.dtype!=np.dtype(dtype) or
                not value.flags.c_contiguous or not finite):
            raise ValueError('Invalid restart array '+key)
        arrays[key]=value
    if 'bank_history_collapse_count' in arrays and np.any(arrays['bank_history_collapse_count']<0):
        raise ValueError('Negative bank-collapse restart count')
    if 'bank_toe_face_gross' in arrays and np.any(arrays['bank_toe_face_gross']<0):
        raise ValueError('Negative gross bank-toe face flux')
    if np.any(arrays['q'][0]<0) or np.any(arrays['layers']<0):raise ValueError('Negative restart depth/inventory')
    if 'suspended' in arrays and np.any(arrays['suspended']<0):
        raise ValueError('Negative restart suspended inventory')
    if secondary_model in (0,1) and (np.any(arrays['secondary']) or np.any(arrays['ledger_secondary'])):
        raise ValueError('Inactive/equilibrium secondary checkpoint must carry zero prognostic inventory')
    identities=[('input_identity',input_identity),('source_identity',source_identity)]
    if method_identity is not None:identities.append(('method_identity',method_identity))
    if events is not None:identities.append(('event_identity',events.identity))
    for key,expected in identities:
        value=np.asarray(checkpoint[key])
        if value.shape!=() or value.dtype.kind not in 'US' or value.item()!=expected:
            raise ValueError('Restart identity mismatch: '+key)
    scalars={}
    for key in ('time','dt_hint','steps','rejected'):
        value=np.asarray(checkpoint[key])
        if value.shape!=() or value.dtype.kind not in 'fiu' or not np.isfinite(value):raise ValueError('Invalid restart '+key)
        if key in ('steps','rejected') and (value.dtype.kind not in 'iu' or value.item()<0):raise ValueError('Invalid restart '+key)
        scalars[key]=value.item()
    if not start<=scalars['time']<end or scalars['dt_hint']<=0:raise ValueError('Restart time is outside requested run')
    if events is not None:
        knot=np.asarray(checkpoint['next_boundary_knot'])
        if knot.shape!=() or knot.dtype.kind!='f' or knot.item()!=events.next_after(scalars['time']):
            raise ValueError('Restart boundary event clock mismatch')
    return arrays,scalars


def bank_toe_persistent_arrays(solver):
    """Accepted bank histories/diagnostics are checkpointed outside RK state."""
    history=getattr(solver,'bank_history',None)
    if history is None:return {}
    required={'undercut','reference_height','toe_height','reference_length',
              'toe_cumulative','collapse_cumulative','collapse_count'}
    if set(history)!=required:
        raise ValueError('Incomplete accepted physical bank-toe history')
    return {**{'bank_history_'+key:value for key,value in history.items()},
            'bank_toe_face_signed':solver.toe_face_signed,
            'bank_toe_face_gross':solver.toe_face_gross}


def validate_bank_toe_checkpoint(arrays, is_toe_face):
    """Reject auxiliary restart data impossible for the fixed physical map."""
    for name in ('undercut','reference_height','toe_height','reference_length',
                 'toe_cumulative','collapse_cumulative'):
        if np.any(arrays['bank_history_'+name]<0):
            raise ValueError('Negative physical bank-toe restart history: '+name)
    signed=arrays['bank_toe_face_signed'];gross=arrays['bank_toe_face_gross']
    active=np.asarray(is_toe_face)
    if (active.shape!=(signed.shape[1],) or not np.isin(active,(0,1)).all() or
            gross.shape!=signed.shape):
        raise ValueError('Physical bank-toe restart map/diagnostic shape differs')
    if np.any(signed[:,active==0]) or np.any(gross[:,active==0]):
        raise ValueError('Unmapped bank-toe face has nonzero restart flux')
    tolerance=1.e-12*np.maximum(1.,np.maximum(np.abs(signed),gross))
    if np.any(gross+tolerance<np.abs(signed)):
        raise ValueError('Gross bank-toe restart flux is below absolute signed flux')


def distributed_bank_persistent_arrays(solver):
    """Mode 3 has seven accepted histories but no persistent physical-toe face map."""
    history=getattr(solver,'bank_history',None)
    required={'undercut','reference_height','toe_height','reference_length',
              'toe_cumulative','collapse_cumulative','collapse_count'}
    if history is None or set(history)!=required:
        raise ValueError('Incomplete accepted distributed-bank history')
    return {'bank_history_'+key:value for key,value in history.items()}


def validate_distributed_bank_checkpoint(arrays):
    """Reject corrupt mode-3 auxiliary state before touching device arrays."""
    for name in ('undercut','reference_height','toe_height','reference_length',
                 'toe_cumulative','collapse_cumulative'):
        if np.any(arrays['bank_history_'+name]<0):
            raise ValueError('Negative distributed-bank restart history: '+name)
    if np.any(arrays['bank_history_collapse_count']<0):
        raise ValueError('Negative distributed-bank restart collapse count')


def run_gpu(**kwargs):
    """Defer Ctrl+C to a transaction boundary, never split state and its time."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError('Run the GPU worker in a separate main process')
    cancelled=threading.Event()
    previous=signal.signal(signal.SIGINT,lambda *_:cancelled.set())
    try:return _run_gpu(cancelled=cancelled.is_set,**kwargs)
    finally:signal.signal(signal.SIGINT,previous)


def _run_gpu(*,input_path,output_path,until=None,output_every=None,dt_max=None,
             max_steps=None,stop_file=None,resume=None,launch_mode='graph',capture_stages=False,
             numerical_variant='auto',integrator='ssp43',capture_step_trace=False,
             source_archive=None,
             cancelled=lambda:False,**unused):
    if numerical_variant==INTEGRATED_VARIANT:
        from .integrated import resolve_prepared
        numerical_variant=resolve_prepared(input_path)
    if numerical_variant==GENERAL_LIVE_TRANSPORT_VARIANT:
        from .general_live_transport_runner import run as run_transport
        return run_transport(input_path=input_path,output_path=output_path,
            until=until,output_every=output_every,dt_max=dt_max,max_steps=max_steps,
            stop_file=stop_file,resume=resume,launch_mode=launch_mode,integrator=integrator,
            capture_stages=capture_stages,capture_step_trace=capture_step_trace,
            source_archive=source_archive,cancelled=cancelled)
    if numerical_variant in (ON_DEMAND_AMR_VARIANT,ON_DEMAND_TRANSPORT_AMR_VARIANT):
        from .amr_on_demand_runner import run as run_on_demand
        return run_on_demand(input_path=input_path,output_path=output_path,
            until=until,output_every=output_every,dt_max=dt_max,max_steps=max_steps,
            stop_file=stop_file,resume=resume,launch_mode=launch_mode,integrator=integrator,
            capture_stages=capture_stages,capture_step_trace=capture_step_trace,
            source_archive=source_archive,cancelled=cancelled)
    if numerical_variant==GENERAL_LIVE_BANK_VARIANT:
        from .general_live_bank_runner import run as run_general_bank
        return run_general_bank(input_path=input_path,output_path=output_path,
            until=until,output_every=output_every,dt_max=dt_max,
            max_steps=max_steps,stop_file=stop_file,resume=resume,
            launch_mode=launch_mode,integrator=integrator,
            capture_stages=capture_stages,capture_step_trace=capture_step_trace,
            source_archive=source_archive,cancelled=cancelled)
    if numerical_variant==LIVE_INLET_VARIANT:
        from .live_inlet_runner import run as run_live_inlet
        return run_live_inlet(input_path=input_path,output_path=output_path,
            until=until,output_every=output_every,dt_max=dt_max,
            max_steps=max_steps,stop_file=stop_file,resume=resume,
            launch_mode=launch_mode,integrator=integrator,
            capture_stages=capture_stages,capture_step_trace=capture_step_trace,
            source_archive=source_archive,cancelled=cancelled)
    if numerical_variant in G3_LIVE_BANK_VARIANTS:
        from .live_bank_runner import run_live_bank
        return run_live_bank(input_path=input_path,output_path=output_path,
            until=until,output_every=output_every,dt_max=dt_max,
            max_steps=max_steps,stop_file=stop_file,resume=resume,
            launch_mode=launch_mode,integrator=integrator,
            capture_stages=capture_stages,capture_step_trace=capture_step_trace,
            cancelled=cancelled,numerical_variant=numerical_variant)
    if numerical_variant==AMR_VARIANT:
        from .amr_composed_runner import run_authenticated_amr
        return run_authenticated_amr(input_path=input_path,output_path=output_path,
            until=until,output_every=output_every,dt_max=dt_max,
            max_steps=max_steps,stop_file=stop_file,resume=resume,
            launch_mode=launch_mode,integrator=integrator,
            capture_stages=capture_stages,
            capture_step_trace=capture_step_trace,cancelled=cancelled)
    from validation.gpu_fwave.adaptive_hydraulics import next_hint,retry_step
    g3_variant=numerical_variant==G3_VARIANT
    bank_avalanche_variant=numerical_variant in (
        'physical_toe_mode2_avalanche_ssp33',G3_VARIANT)
    distributed_bank_avalanche_variant=(
        numerical_variant=='distributed_slope_mode3_avalanche_ssp33')
    secondary_live_variant=numerical_variant==SECONDARY_LIVE_VARIANT
    target_b_variant=numerical_variant==TARGET_B_ADAPTATION_VARIANT
    general_bedload_variant=numerical_variant==GRADED_BEDLOAD_GENERAL_VARIANT
    if source_archive is not None and not general_bedload_variant:
        raise ValueError('GPU source archive is only supported for the graded-bedload selector')
    bedload_live_variant=target_b_variant or general_bedload_variant
    live_graded_variant=numerical_variant==GRADED_LIVE_VR_VARIANT
    live_vr_variant=numerical_variant in (LIVE_VR_VARIANT,GRADED_LIVE_VR_VARIANT)
    live_suspension_variant=numerical_variant in (
        LIVE_SUSPENSION_VARIANT,LIVE_VR_VARIANT,GRADED_LIVE_VR_VARIANT)
    live_exchange_mode=2 if live_vr_variant else 1
    data,input_identity=load_prepared(input_path,
        bank_avalanche_mode2=bank_avalanche_variant,
        distributed_bank_avalanche_mode3=distributed_bank_avalanche_variant,
        general_live_sections=secondary_live_variant or live_suspension_variant or bedload_live_variant,
        g3_bank_secondary=g3_variant)
    input_path=filesystem_path(input_path).resolve()
    resume=(None if resume is None else filesystem_path(resume).resolve())
    receipt_path=input_path/'preparation_receipt.json'
    if (not general_bedload_variant and receipt_path.is_file() and
            json.loads(receipt_path.read_text()).get('schema')==3):
        raise ValueError('GPU preparation schema 3 requires its explicit graded-bedload selector')
    reject_deferred_public_controls(data)
    if not isinstance(capture_step_trace,bool):
        raise ValueError('GPU accepted-step trace selector must be boolean')
    if launch_mode not in ('graph','plain'):raise ValueError('GPU launch mode must be graph or plain')
    if integrator not in ('ssp43','ssp33','implicit_fwave_midpoint'):
        raise ValueError('Unsupported GPU coupled integrator')
    # The common fixed-grid transport dispatch returns above and owns the
    # complete implicit capability gate.  No historical route may reinterpret
    # this user selection as an explicit method.
    if integrator=='implicit_fwave_midpoint':
        raise ValueError('GPU implicit f-wave midpoint requires the common fixed-grid transport route')
    if integrator=='ssp33' and launch_mode!='plain':
        raise ValueError('GPU SSPRK33 requires plain launch; CUDA graph replay is not qualified')
    start,end,interval,dtmax=time_controls(data,until,output_every,dt_max)
    if max_steps is not None and (not isinstance(max_steps,int) or max_steps<1):raise ValueError('Positive max_steps required')
    combined=(input_path/'normal_metadata.bin').exists()
    a1_variant=numerical_variant==A1_VARIANT
    bounded_live=a1_variant or secondary_live_variant or live_suspension_variant or bedload_live_variant
    no_bedload=int(data['bedlaw'])==0
    if no_bedload and (launch_mode!='plain' or integrator!='ssp33'):
        raise ValueError('GPU no-bedload mode requires explicitly selected plain SSPRK33')
    if numerical_variant not in ('auto','scheduled_normal9_critical4',INTEGRATED_VARIANT,A1_VARIANT,SECONDARY_LIVE_VARIANT,LIVE_SUSPENSION_VARIANT,LIVE_VR_VARIANT,GRADED_LIVE_VR_VARIANT,TARGET_B_ADAPTATION_VARIANT,GRADED_BEDLOAD_GENERAL_VARIANT,LIVE_INLET_VARIANT,GENERAL_LIVE_BANK_VARIANT,GENERAL_LIVE_TRANSPORT_VARIANT,AMR_VARIANT,ON_DEMAND_AMR_VARIANT,ON_DEMAND_TRANSPORT_AMR_VARIANT,'qualified_curved_wetdry',
                                 'scheduled_inlet_ssp33','bedform_feedback_ssp33',
                                 'general_avalanche_plain_ssp33',
                                 'physical_toe_mode2_plain_ssp33',
                                 'physical_toe_mode2_avalanche_ssp33',
                                 G3_VARIANT,
                                 G3_LIVE_BANK_MODE2_VARIANT,G3_LIVE_BANK_MODE3_VARIANT,
                                 'distributed_slope_mode3_plain_ssp33',
                                 'distributed_slope_mode3_avalanche_ssp33',
                                 'bedload_adaptation_plain_ssp33'):
        raise ValueError('Unsupported explicit GPU numerical variant')
    scheduled_inlet=numerical_variant=='scheduled_inlet_ssp33'
    bedform_variant=numerical_variant=='bedform_feedback_ssp33'
    avalanche_variant=numerical_variant=='general_avalanche_plain_ssp33'
    bank_toe_variant=numerical_variant in ('physical_toe_mode2_plain_ssp33',
                                         'physical_toe_mode2_avalanche_ssp33',G3_VARIANT)
    distributed_bank_variant=numerical_variant in (
        'distributed_slope_mode3_plain_ssp33',
        'distributed_slope_mode3_avalanche_ssp33')
    adaptation_variant=numerical_variant=='bedload_adaptation_plain_ssp33'
    avalanche_sidecar=(input_path/'avalanche_metadata.bin').is_file()
    bank_toe_sidecar=(input_path/'bank_toe_metadata.bin').is_file()
    distributed_bank_sidecar=(input_path/'distributed_bank_metadata.bin').is_file()
    if avalanche_sidecar and not (avalanche_variant or bank_avalanche_variant or distributed_bank_avalanche_variant):
        raise ValueError('Prepared general-avalanche ON/OFF sidecar requires an explicit GPU general-avalanche variant')
    if bank_toe_sidecar and not bank_toe_variant:
        raise ValueError('Prepared physical bank-toe ON/OFF sidecar requires explicit GPU physical_toe_mode2_plain_ssp33 variant')
    if distributed_bank_sidecar and not distributed_bank_variant:
        raise ValueError('Prepared distributed-bank sidecar requires explicit GPU distributed_slope_mode3_plain_ssp33 variant')
    if bedform_variant and (combined or launch_mode!='plain' or integrator!='ssp33' or no_bedload):
        raise ValueError('Bedform feedback variant requires stock MPM plain SSPRK33')
    if avalanche_variant and (combined or launch_mode!='plain' or integrator!='ssp33' or not no_bedload or
            not avalanche_sidecar):
        raise ValueError('General-avalanche variant requires prepared bedlaw-0 plain SSPRK33 sidecar')
    if bank_toe_variant and (combined or launch_mode!='plain' or integrator!='ssp33' or no_bedload or
            not bank_toe_sidecar or int(data['bedlaw'])!=2 or
            int(data['bank_toe_erosion_mode'])!=2 or int(data['bank_failure'])!=1):
        raise ValueError('Physical bank-toe mode 2 requires prepared MPM plain SSPRK33 sidecar')
    if bank_avalanche_variant and not avalanche_sidecar:
        raise ValueError('Bank/avalanche mode 2 requires authenticated general-avalanche sidecar')
    if distributed_bank_avalanche_variant and not avalanche_sidecar:
        raise ValueError('Distributed bank/avalanche mode 3 requires authenticated general-avalanche sidecar')
    if distributed_bank_variant and (combined or launch_mode!='plain' or integrator!='ssp33' or
            no_bedload or not distributed_bank_sidecar or int(data['bedlaw'])!=2 or
            int(data['bank_toe_erosion_mode'])!=3 or int(data['bank_failure'])!=1):
        raise ValueError('Distributed bank mode 3 requires prepared MPM plain SSPRK33 sidecar')
    if adaptation_variant and (combined or launch_mode!='plain' or integrator!='ssp33' or
            no_bedload or int(data['bedlaw'])!=2 or not int(data['bedload_adaptation'])):
        raise ValueError('Bedload adaptation requires prepared MPM plain SSPRK33 input with native adaptation enabled')
    mode3_reference=None
    if distributed_bank_variant:
        receipt=json.loads((input_path/'preparation_receipt.json').read_text())
        reference_id=receipt.get('native_reference_id',
            'i4_distributed_bank_mode3_native_20261004_v1')
        references={
            'i4_distributed_bank_mode3_native_20261004_v1':(
                False,'GPU_ACCEPTANCE_GATE.txt'),
            'i4_distributed_bank_mode3_native_20261004_v2':(
                True,'GPU_ACCEPTANCE_GATE_POSTDIAGNOSTIC.txt'),
            'i4_mode3_active_overlap_native_20261004_v1':(
                True,'GPU_ACCEPTANCE_GATE.txt'),
            'i4_mode3_avalanche_native_20261004_v1':(
                True,'GPU_ACCEPTANCE_GATE.txt')}
        if reference_id not in references:
            raise ValueError('Unsupported distributed-bank native reference identity')
        root=CUDA_SOURCE/'runs'/reference_id
        source_identical,gate_name=references[reference_id]
        gate=CUDA_SOURCE/'reports'/reference_id/gate_name
        identities={'native_plan_sha256':root/'plan.json',
                    'native_pre_gpu_gate_sha256':gate,
                    'native_reference_executable_sha256':root/'runtime/xbraids_mode3_reference.exe'}
        if any(not isinstance(receipt.get(key),str) or len(receipt[key])!=64 or
               not path.is_file() or sha(path)!=receipt[key]
               for key,path in identities.items()):
            raise ValueError('Distributed-bank prepared native reference identity differs')
        if reference_id in ('i4_mode3_active_overlap_native_20261004_v1',
                            'i4_mode3_avalanche_native_20261004_v1'):
            analysis_path=root/'analysis.json'
            if (not analysis_path.is_file() or
                    sha(analysis_path)!=receipt.get('native_analysis_sha256')):
                raise ValueError('Distributed-bank active-overlap native analysis identity differs')
            plan=json.loads((root/'plan.json').read_text())
            for source_name,digest in plan['source_snapshot_sha256'].items():
                source_path=(root/'runtime/source_snapshot'/source_name).resolve()
                if (not source_path.is_relative_to((root/'runtime/source_snapshot').resolve()) or
                        not source_path.is_file() or sha(source_path)!=digest):
                    raise ValueError('Distributed-bank active-overlap native source snapshot differs')
            cases=(('bank0_aval0','bank1_aval0','bank0_aval1','bank1_aval1')
                   if reference_id=='i4_mode3_avalanche_native_20261004_v1'
                   else ('erosion','control'))
            for case in cases:
                for artifact,digest in plan['inputs'][case].items():
                    if sha(root/'inputs'/case/artifact)!=digest:
                        raise ValueError('Distributed-bank active-overlap native input differs')
                execution=json.loads((root/case/'full/execution.json').read_text())
                if (execution.get('status')!='completed' or execution.get('return_code')!=0 or
                        execution.get('plan_sha256')!=receipt['native_plan_sha256'] or
                        execution.get('native_executable_sha256')!=receipt['native_reference_executable_sha256'] or
                        any(sha(root/case/'full'/artifact)!=digest
                            for artifact,digest in execution.get('output_files',{}).items())):
                    raise ValueError('Distributed-bank active-overlap native execution differs')
            if reference_id=='i4_mode3_avalanche_native_20261004_v1':
                frozen_preparation={
                    'bank0_aval0':'c0846e0aff3b1f228d330e77f60d589c15602ddbbf9a8171f96297aca156d653',
                    'bank1_aval0':'cb1d054e15e31d338c9f2f480dd82560d671e88a8dbaf5fcb80c2fde2cc643ce',
                    'bank0_aval1':'d57dbf2f89bc5da38aa8f1e06a2f545297b8b46ea0599de1886687869eeb34c3',
                    'bank1_aval1':'7d4ed96e51994ed9cfa7f23e632c299ab83c535962ce062b12783a86f128bf44'}
                if (receipt.get('case') not in frozen_preparation or
                        input_identity!=frozen_preparation[receipt['case']]):
                    raise ValueError('Distributed-bank avalanche prepared input differs from frozen case')
                exporter=root.parent/'i4_mode3_avalanche_prepared_20261004_v1'/'runtime'
                export_identities={
                    'exporter_build_sha256':exporter/'build.json',
                    'exporter_executable_sha256':exporter/'xbraids_mode3_avalanche_export.exe'}
                if any(not path.is_file() or sha(path)!=receipt.get(key)
                       for key,path in export_identities.items()):
                    raise ValueError('Distributed-bank avalanche exporter identity differs')
                if (receipt.get('zero_step_export') is not True or
                        receipt.get('solver_steps_performed')!=0 or
                        receipt.get('numerical_parser_body_unchanged') is not True):
                    raise ValueError('Distributed-bank avalanche parser provenance differs')
        if (receipt.get('source_identical_to_native_reference') is not source_identical or
                receipt.get('native_internal_stencils_independently_verified') is not False):
            raise ValueError('Distributed-bank parser provenance must be explicit')
        mode3_reference={key:receipt[key] for key in identities}
        mode3_reference['native_reference_id']=reference_id
        mode3_reference['source_identical_to_native_reference']=source_identical
        mode3_reference['native_internal_stencils_independently_verified']=False
        if reference_id in ('i4_mode3_active_overlap_native_20261004_v1',
                            'i4_mode3_avalanche_native_20261004_v1'):
            mode3_reference['native_analysis_sha256']=receipt['native_analysis_sha256']
    if numerical_variant=='scheduled_normal9_critical4' and not combined:
        raise ValueError('Scheduled normal/critical variant requires authenticated native boundary metadata')
    if a1_variant and not combined:
        raise ValueError('A1 live scheduled variant requires authenticated native normal_metadata.bin')
    if secondary_live_variant and not combined:
        raise ValueError('Live secondary variant requires authenticated native normal_metadata.bin')
    if live_suspension_variant and not combined:
        raise ValueError('Live suspension variant requires authenticated native normal_metadata.bin')
    if bedload_live_variant and not combined:
        raise ValueError('Live graded bedload variant requires authenticated native normal_metadata.bin')
    if numerical_variant=='qualified_curved_wetdry' and combined:
        raise ValueError('Qualified curved wet/dry variant does not support scheduled normal/critical boundaries')
    if scheduled_inlet:
        if combined or launch_mode!='plain' or integrator!='ssp33':
            raise ValueError('Scheduled per-class inlet requires stock plain SSPRK33')
        if not np.any(data.get('nfraction_series',0)):
            raise ValueError('Scheduled per-class inlet requires an active fraction series')
        if int(data.get('cpu_outlet_extension',data.get('outlet_extension',0)))!=0:
            raise ValueError('Scheduled per-class inlet requires native outlet extension 0')
    curved=numerical_variant=='qualified_curved_wetdry'
    if no_bedload and (combined or curved):
        raise ValueError('GPU no-bedload mode is qualified only for stock constant boundaries')
    if integrator=='ssp33' and ((combined and not (live_suspension_variant or bedload_live_variant)) or curved):
        raise ValueError('GPU SSPRK33 is only enabled for the stock constant-boundary solver')
    if capture_stages and not combined:
        raise ValueError('Stage diagnostics currently require the combined normal/critical boundary scope')
    if bounded_live:
        if capture_stages:
            raise ValueError('Live scheduled variants use --gpu-capture-step-trace; the older stage wrapper is unqualified')
    if g3_variant:
        g3_reference=g3_frozen_preflight(data,input_path,start=start,end=end,
            interval=interval,dtmax=dtmax,launch_mode=launch_mode,
            integrator=integrator)
    else:g3_reference=None
    if a1_variant:
        a1_case,events,a1_gate=a1_frozen_preflight(data,input_path,start=start,end=end,
            interval=interval,dtmax=dtmax,launch_mode=launch_mode,integrator=integrator)
        constructor_data=data
        scope=('Frozen 128-cell pool/wetdry A1 live mode9 normal inlet, mode4 critical outlet, '
            'scheduled Q and two distinct class feeds; fixed-grid MPM, secondary OFF, plain SSPRK43')
    elif secondary_live_variant:
        from . import live_secondary
        events=live_secondary.validate_scope(data,start=start,end=end,interval=interval,
            dtmax=dtmax,launch_mode=launch_mode,integrator=integrator)
        constructor_data=data
        scope=('Fixed-grid initially wet or mixed wet/dry two-class/two-layer MPM with one live normal inlet, '
            'one critical outlet, common scheduled Q/class-feed knots and prognostic secondary flow; '
            'plain SSPRK43; guarded research selector')
    elif live_suspension_variant:
        from . import live_suspension
        events=live_suspension.validate_scope(data,start=start,end=end,interval=interval,
            dtmax=dtmax,launch_mode=launch_mode,integrator=integrator,
            expected_exchange=live_exchange_mode,graded=live_graded_variant)
        constructor_data=data
        scope=(f'Fixed-grid initially mixed wet/dry {int(data["nfrac"])}-class/{int(data["nbed_layers"])}-layer MPM with one live normal inlet, '
            f'one critical outlet, scheduled Q/class bed feed and concentration, noncohesive exchange {live_exchange_mode}, '
            'secondary OFF or prognostic model 2; plain SSPRK33; guarded engineering selector')
    elif target_b_variant:
        from . import live_bedload_adaptation as target_b
        events=target_b.validate_scope(data,start=start,end=end,interval=interval,
            dtmax=dtmax,launch_mode=launch_mode,integrator=integrator)
        constructor_data=data
        scope=('Frozen 128-cell initially mixed wet/dry seven-class/four-layer MPM '
            'bedload-only case, live normal9/critical4, scheduled Q/class feed, '
            'native adaptation OFF/ON, hiding 4/slope 2, plain SSPRK33')
    elif general_bedload_variant:
        from . import live_bedload_adaptation as target_b
        events=target_b.validate_general_scope(data,start=start,end=end,
            interval=interval,dtmax=dtmax,launch_mode=launch_mode,
            integrator=integrator)
        constructor_data=data
        scope=('Manifest-authenticated fixed-grid mixed wet/dry seven-class/four-layer '
            'MPM bedload-only, live normal9/critical4, scheduled Q/class feed, '
            'classwise adaptation OFF/ON, hiding 4/slope 2, plain SSPRK33; '
            'not a QGIS release or physical validation')
    elif combined:
        from validation.gpu_fwave.boundary_events import BoundaryEvents
        combined_scope(data);events=BoundaryEvents.from_tables(data)
        constructor_data=dict(data,native_outlet_extension=int(data['cpu_outlet_extension']),
                              outlet_extension=0,cpu_outlet_extension=0)
        scope='Fixed-grid scheduled mode9/control1 normal inlet and mode4 critical outlet; MPM and prognostic secondary flow'
    elif curved:
        if launch_mode!='graph':raise ValueError('Qualified curved wet/dry variant requires its captured graph route')
        curved_plan_sha,curved_plan=qualified_curved_scope(data,input_path,
            start=start,end=end,interval=interval,dtmax=dtmax)
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        constructor_data=dict(data,outlet_extension=0,cpu_outlet_extension=0)
        validate_scope(normalized(constructor_data));events=None
        scope='Exact archived bar256 curved wet/dry secondary-only qualified source combination; graph and fixed 0.002 s trial'
    elif scheduled_inlet:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        from validation.gpu_fwave.boundary_events import BoundaryEvents
        checked=normalized(data);validate_scope(checked,scheduled=True)
        events=BoundaryEvents.from_tables(checked);constructor_data=data
        scope=(f'Fixed-grid stock scheduled per-class inlet; plain SSPRK33; '
               f'bedlaw {int(data["bedlaw"])}; suspended exchange {int(data["exchange_mode"])}; '
               'no live sections, AMR or graph replay')
    elif bedform_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data);validate_scope(checked,bedform_variant=True)
        events=None;constructor_data=data
        scope=('Fixed-grid stock MPM with explicitly selected one-flag bedform-feedback '
               'ablation; native SSPRK33, plain launch, no suspended exchange or AMR')
    elif avalanche_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data);validate_scope(checked,general_avalanche_variant=True)
        events=None;constructor_data=data
        scope=('Fixed-grid closed bedlaw-0 with explicitly selected one-flag general-avalanche '
               'ablation; native-order SSPRK33, plain launch, no graph/AMR/bank failure')
    elif bank_avalanche_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data)
        validate_scope(checked,bank_avalanche_mode2_variant=True,
                       g3_bank_secondary_variant=g3_variant)
        events=None;constructor_data=data
        scope=(('Frozen G3 fixed-grid graded MPM physical-toe mode 2, post-RK general '
                'wet/dry avalanche and prognostic secondary composition; '
                'native-order plain SSPRK33, no graph/AMR') if g3_variant else
               ('Fixed-grid MPM physical-toe mode 2 followed by one post-RK '
                'general-avalanche transaction; native-order plain SSPRK33, no graph/AMR'))
    elif bank_toe_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data);validate_scope(checked,physical_toe_mode2_variant=True)
        events=None;constructor_data=data
        scope=('Fixed-grid stock two-grain MPM with explicitly selected noncohesive physical '
               'bank-toe mode 2; native-order post-combination SSPRK33, plain launch, no graph/AMR')
    elif distributed_bank_avalanche_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data)
        validate_scope(checked,distributed_bank_avalanche_mode3_variant=True)
        events=None;constructor_data=data
        scope=('Fixed-grid two-grain MPM distributed-slope bank mode 3 followed by one '
               'post-RK general wet/dry avalanche transaction; plain SSPRK33, no graph/AMR')
    elif distributed_bank_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data);validate_scope(checked,distributed_bank_mode3_variant=True)
        events=None;constructor_data=data
        scope=('Fixed-grid stock two-grain MPM with explicitly selected noncohesive distributed '
               'slope-bank mode 3; native-order post-combination SSPRK33, plain launch, no graph/AMR')
    elif adaptation_variant:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data);validate_scope(checked,bedload_adaptation_variant=True)
        events=None;constructor_data=data
        scope=('Fixed-grid stock mode6/mode2 MPM with explicitly selected classwise bedload '
               'adaptation; stage-local native Jacobi, plain SSPRK33, no graph/AMR/bank physics')
    else:
        from validation.gpu_fwave.coupled_solver import normalized,validate_scope
        checked=normalized(data);validate_scope(checked);events=None;constructor_data=data
        if launch_mode=='graph' and (int(data['exchange_mode'])!=0 or
                bool(data.get('suspended_transport_active',False)) or
                np.any(checked['suspended_mass']) or np.any(data['fraction_concentration'])):
            raise ValueError('Suspended GPU transport requires plain launch until adaptive graph replay is qualified')
        scope=(f'Fixed-grid constant-boundary {"no bedload" if no_bedload else "MPM"}; secondary model '
                f'{int(data["secondary_flow_model"])}; '
                f'{np.asarray(data["layer_mass"]).shape[0]} bed layers; '
                f'outlet extension {int(data.get("outlet_extension",0))}; '
                f'suspended exchange mode {int(data["exchange_mode"])}; '
                f'{integrator.upper()}')
    # Preserve explicit variant/sidecar/native-reference rejection precedence
    # on incomplete inputs, then check material identity before output or CUDA.
    # Per-cell bed and roughness arrays pass unchanged to resident kernels.
    from validation.gpu_fwave.spatial_material import validate_prepared_spatial_material
    spatial_material_receipt=validate_prepared_spatial_material(data)
    vr_reference=(live_suspension.verify_graded_reference(input_path) if live_graded_variant
                  else live_suspension.verify_vr_reference(input_path) if live_vr_variant
                  else None)
    target_b_reference=(target_b.verify_reference(input_path) if target_b_variant
                        else target_b.verify_general_prepared(input_path,data)
                        if general_bedload_variant else None)
    destination=filesystem_path(output_path).resolve();a1_restored=None;a1_preflight_method=None
    if a1_variant:
        source_identity,a1_preflight_method,a1_restored=prepare_a1_run(data,input_path,destination,
            input_identity=input_identity,events=events,start=start,end=end,dtmax=dtmax,
            capture_stages=capture_stages,resume=resume,gate=a1_gate)
    elif secondary_live_variant:
        source_identity,a1_preflight_method,a1_restored=prepare_secondary_run(data,input_path,destination,
            input_identity=input_identity,events=events,start=start,end=end,dtmax=dtmax,
            capture_stages=capture_stages,resume=resume)
    elif live_suspension_variant:
        source_identity,a1_preflight_method,a1_restored=prepare_live_suspension_run(data,input_path,destination,
            input_identity=input_identity,events=events,start=start,end=end,dtmax=dtmax,
            capture_stages=capture_stages,resume=resume,exchange_mode=live_exchange_mode,
            vr_reference=vr_reference,graded=live_graded_variant)
    elif target_b_variant:
        source_identity,a1_preflight_method,a1_restored=prepare_target_b_run(data,input_path,destination,
            input_identity=input_identity,events=events,start=start,end=end,dtmax=dtmax,
            capture_stages=capture_stages,resume=resume,reference=target_b_reference)
    elif general_bedload_variant:
        source_identity,a1_preflight_method,a1_restored=prepare_general_bedload_run(
            data,input_path,destination,input_identity=input_identity,
            events=events,start=start,end=end,dtmax=dtmax,
            capture_stages=capture_stages,resume=resume,reference=target_b_reference,
            source_archive=source_archive)
    else:destination.mkdir(parents=True,exist_ok=False)
    source_folder=destination/'source';bindings=[];source_stage_allowed=False
    if a1_variant:
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=a1_constructors(source_folder,data,bindings)
    elif secondary_live_variant:
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=secondary_constructors(source_folder,data,bindings)
    elif live_suspension_variant:
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=live_suspension_constructors(source_folder,data,bindings,
            exchange_mode=live_exchange_mode,graded=live_graded_variant)
    elif target_b_variant:
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=target_b_constructors(source_folder,data,bindings)
    elif general_bedload_variant:
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=target_b_constructors(source_folder,data,bindings,general=True)
    elif combined:
        baseline,live,source_identity=freeze_combined_sources(source_folder)
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=combined_constructors(source_folder,baseline,live,bindings)
    elif curved:
        baseline,live,source_identity=freeze_qualified_curved_sources(source_folder)
        sources=json.loads((source_folder/'runtime_manifest.json').read_text())['source_files']
        binding=qualified_curved_constructors(source_folder,baseline,live,bindings)
    else:
        sources=freeze_sources(source_folder,distributed_bank=distributed_bank_variant,
                               bedload_adaptation=adaptation_variant)
        source_identity=sha(source_folder/'manifest.json')
        binding=stock_constructors()
        source_stage_allowed=True
    ledger_algorithm=dict(algorithm='none')
    if (not combined or live_suspension_variant) and not curved and int(data['exchange_mode']) in (1,2):
        from validation.gpu_fwave.exchange_ledger_parallel import (
            DEFAULT_DIRECT_LIMIT,TILE,method_for_cells)
        ledger_algorithm=dict(algorithm='deterministic_cell_parallel_v1',
            dispatch=method_for_cells(int(np.asarray(data['q']).shape[1])),
            tile_cells=TILE,direct_limit_cells=DEFAULT_DIRECT_LIMIT)
    method=dict(variant='scheduled_normal9_critical4' if combined else
                'qualified_curved_wetdry' if curved else
                'scheduled_inlet_ssp33' if scheduled_inlet else 'constant_boundary',
                integrator='SSPRK33' if integrator=='ssp33' else 'SSPRK43',
                euler_stages=3 if integrator=='ssp33' else 4,
                launch_mode=launch_mode,precision='FP64',dtmax_s=dtmax,
                capture_stages=bool(capture_stages),
                cfl=float(data['cfl_target']),morfac=float(data['morphological_factor']),
                bedlaw=int(data['bedlaw']),
                secondary_flow_model=int(data['secondary_flow_model']),
                exchange_mode=int(data['exchange_mode']),settling_mode=int(data.get('settling_mode',0)),
                exchange_ledger=ledger_algorithm,
                source_identity=source_identity,event_identity=None if events is None else events.identity)
    if int(data['bedlaw'])==2:
        method['mpm_friction_mode']=int(data['mpm_friction_mode'])
        method['mpm_cf']=float(data['mpm_cf'])
    if bedform_variant:
        method['variant']='bedform_feedback_ssp33'
        method['bedlaw']=int(data['bedlaw'])
        method['bedform_roughness']=dict(enabled=bool(data['bedform_roughness']),
            model='fredsoe_van_rijn_1984_equilibrium',
            mpm_friction_mode=int(data['mpm_friction_mode']),
            d50_m=np.asarray(data['d50_fraction']).tolist(),
            critical_shields=float(data['mpm_critical']),
            evaluations_per_euler=1 if bool(data['bedform_roughness']) else 0,
            friction_evaluation_per_trial=1 if bool(data['bedform_roughness']) else 0,
            diagnostic_from_accepted_state=bool(data['bedform_roughness']))
        method['bedform_mode1']=dict(enabled=True,
            kind='unsampled_total_Q_characteristic_stage_root',root_iterations=90,
            sampled_boundary_bed=False,boundary_bed_offset=False,section_control=0)
    if avalanche_variant:
        controls=('general_avalanche_wet_slope','general_avalanche_dry_slope',
                  'general_avalanche_time','general_avalanche_wet_depth_switch',
                  'general_avalanche_relaxation_length')
        method['variant']='general_avalanche_plain_ssp33'
        method['bedlaw']=0
        method['general_avalanche']=dict(enabled=bool(data['general_avalanche']),
            controls={key:float(data[key]) for key in controls},
            placement='post_euler_hydro_bed_pre_rebalance',
            post_combination_rebalance=True,accepted_state_transaction=True)
    if bank_toe_variant:
        method['variant']=(G3_VARIANT if g3_variant else
                           'physical_toe_mode2_avalanche_ssp33' if bank_avalanche_variant
                           else 'physical_toe_mode2_plain_ssp33')
        method['bedlaw']=2
        method['physical_bank_toe']=dict(mode=2,bank_failure=True,
            capacity_fraction=float(data['bank_toe_capacity_fraction']),
            critical_undercut_m=float(data['bank_critical_undercut_m']),
            placement='post_ssp33_combination_and_source_pre_commit',
            physical_map_from_prepared_native_sidecar=True,
            donor_limited_face_flux_weights=[1./6.,1./6.,2./3.],
            history_accepted_state_transaction=True,
            face_flux_signed_and_gross_accepted_state=True)
    if bank_avalanche_variant or distributed_bank_avalanche_variant:
        controls=('general_avalanche_wet_slope','general_avalanche_dry_slope',
                  'general_avalanche_time','general_avalanche_wet_depth_switch',
                  'general_avalanche_relaxation_length')
        method['general_avalanche']=dict(enabled=bool(data['general_avalanche']),
            controls={key:float(data[key]) for key in controls},
            placement='once_post_rk_after_bank_failure',
            post_bank_rebalance=True,accepted_state_transaction=True)
    if g3_variant:
        method['frozen_g3_reference']=g3_reference
        method['secondary_composition']=dict(
            model=int(data['secondary_flow_model']),
            conserved_state_index=int(data['secondary_index']),
            momentum_beta=float(data['secondary_flow_momentum_beta']),
            placement='each_ssp33_euler_stage_and_accepted_source',
            accepted_state_transaction=True)
    if distributed_bank_variant:
        method['variant']=('distributed_slope_mode3_avalanche_ssp33'
                           if distributed_bank_avalanche_variant else
                           'distributed_slope_mode3_plain_ssp33')
        method['bedlaw']=2
        method['native_reference']=mode3_reference
        method['distributed_bank']=dict(mode=3,bank_failure=True,
            transport_fraction=float(data['bank_distributed_transport_fraction']),
            critical_slope=float(data['bank_distributed_critical_slope']),
            critical_undercut_m=float(data['bank_critical_undercut_m']),
            placement='post_ssp33_combination_and_source_pre_commit',
            signed_donor_limited_face_flux_weights=[1./6.,1./6.,2./3.],
            dynamic_face_eligibility_from_accepted_bed=True,
            history_accepted_state_transaction=True)
    if adaptation_variant:
        method['variant']='bedload_adaptation_plain_ssp33'
        method['bedload_adaptation']=dict(enabled=True,
            length_diameters=float(data['bedload_adaptation_length_diameters']),
            closure='stage_local_classwise_streamline_jacobi',
            current_stage_feed=True,derived_capacity_not_checkpointed=True)
    if scheduled_inlet:
        method['scheduled_inlet']=dict(record_indices=[0,2,1],stage_fractions=[0.,1.,.5],
            scheduled_concentration=True,scheduled_bed_feed=int(data['bedlaw'])==2,
            accepted_state_transaction=True)
    if no_bedload:method['bedlaw']=0
    if bounded_live:method=a1_preflight_method
    method_identity=hashlib.sha256(json.dumps(method,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=dict(schema=1,backend='gpu',backend_selected_by='user',automatic_backend_selection=False,
        process_id=os.getpid(),
        input_preparation_sha256=input_identity,input_folder=str(input_path),
        source_sha256=sources,source_identity=source_identity,method_identity=method_identity,
        method=method,scope=scope,combined_adapter=combined and not bounded_live,
        a1_live_scheduled_adapter=a1_variant,
        live_secondary_adapter=secondary_live_variant,
        live_suspension_adapter=live_suspension_variant,
        target_b_adaptation_adapter=target_b_variant,
        graded_bedload_general_adapter=general_bedload_variant,
        integrator=method['integrator'],euler_stages=method['euler_stages'],
        precision='FP64',end_s=end,output_every_s=interval,dtmax_s=dtmax,
        cfl=float(data['cfl_target']),morfac=float(data['morphological_factor']),launch_mode=launch_mode,frames=[],
        spatial_material_preflight=spatial_material_receipt,
        accepted_step_trace_requested=capture_step_trace)
    if g3_variant:
        manifest['g3_bank_secondary']=dict(**g3_reference,
            public_variant_selectable=True,full_frozen_gate_passed=False,
            QGIS_release=False,observational_validation=False)
    if curved:
        from validation.gpu_fwave import curved_wetdry_worker as qualified
        manifest['qualified_curved_wetdry']=dict(plan_sha256=curved_plan_sha,
            prepared_receipt_sha256=input_identity,filter_controls=qualified.FILTER,
            constructor_outlet_extension=0,effective_reconstruction_outlet_extension=1,
            bound_to_exact_bar256_fixture=True,fixed_trial_dt_s=.002,
            rejected_trial_policy='abort; no retry or CPU fallback',
            diagnostic_instrumentation=True,performance_timing_claim=False)
    if a1_variant:
        manifest['a1_live_scheduled']=dict(case=a1_case,gate_sha256=a1_gate.GATE_SHA,
            prepared_normal_metadata_sha256=sha(input_path/'normal_metadata.bin'),
            source_receipt_sha256=sha(source_folder/'candidate_sources.json'),
            auxiliary_checkpoint_schema=1,graph_supported=False,
            public_variant_selectable=True,private_constructor_receipt_is_historical=True,
            QGIS_release=False,physical_validation=False,reach_scale_speedup_claim=False)
    if secondary_live_variant:
        manifest['live_secondary']=dict(historical_gate_sha256=method['historical_a2_gate_sha256'],
            prepared_normal_metadata_sha256=sha(input_path/'normal_metadata.bin'),
            source_receipt_sha256=sha(source_folder/'candidate_sources.json'),
            auxiliary_checkpoint_schema=2,graph_supported=False,
            public_variant_selectable=True,exact_fixture_hash_required=False,
            initial_all_wet_required=False,
            initial_dry_cells=int(np.count_nonzero(np.asarray(data['q'])[0]<=float(data['dry_tolerance']))),
            evolving_wet_dry_gate_sha256=method['evolving_wet_dry_gate_sha256'],
            wet_dry_runtime_guard=False,
            QGIS_release=False,physical_validation=False,reach_scale_speedup_claim=False)
    if live_suspension_variant:
        manifest['live_suspension']=dict(frozen_g2_gate_sha256=method['frozen_g2_gate_sha256'],
            prepared_normal_metadata_sha256=sha(input_path/'normal_metadata.bin'),
            source_receipt_sha256=sha(source_folder/'candidate_sources.json'),
            auxiliary_checkpoint_schema=3,graph_supported=False,
            public_variant_selectable=True,exact_fixture_hash_required=False,
            initial_dry_cells=int(np.count_nonzero(np.asarray(data['q'])[0]<=float(data['dry_tolerance']))),
            euler_schedule_slots=[0,2,1],suspended_exchange_enabled=True,
            QGIS_release=False,physical_validation=False,reach_scale_speedup_claim=False)
        if live_vr_variant:
            manifest['live_suspension']['frozen_vr_reference']=vr_reference
            manifest['live_suspension']['archive_bound_to_two_prepared_cases']=True
            manifest['live_suspension']['graded_seven_class_four_layer']=live_graded_variant
    if target_b_variant:
        manifest['target_b_adaptation']=dict(
            frozen_gate_sha256=target_b_reference['gate_sha256'],
            frozen_reference=target_b_reference,
            prepared_normal_metadata_sha256=sha(input_path/'normal_metadata.bin'),
            source_receipt_sha256=sha(source_folder/'candidate_sources.json'),
            auxiliary_checkpoint_schema=4,graph_supported=False,
            public_variant_selectable=True,archive_bound=True,
            adaptation_enabled=bool(int(data['bedload_adaptation'])),
            initial_dry_cells=int(np.count_nonzero(np.asarray(data['q'])[0]<=float(data['dry_tolerance']))),
            euler_schedule_slots=[0,2,1],QGIS_release=False,
            observational_validation=False)
    if general_bedload_variant:
        manifest['graded_bedload_live']=dict(
            preparation_contract=target_b_reference,
            prepared_normal_metadata_sha256=sha(input_path/'normal_metadata.bin'),
            source_receipt_sha256=sha(source_folder/'candidate_sources.json'),
            auxiliary_checkpoint_schema=4,graph_supported=False,
            public_variant_selectable=True,archive_bound=False,
            adaptation_enabled=bool(int(data['bedload_adaptation'])),
            initial_dry_cells=int(np.count_nonzero(
                np.asarray(data['q'])[0]<=float(data['dry_tolerance']))),
            euler_schedule_slots=[0,2,1],QGIS_release=False,
            observational_validation=False)
    if events is not None:
        manifest['boundary_events']=dict(identity=events.identity,knots_s=events.knots.tolist(),
            internal_hits=0,mode=np.asarray(data['mode']).tolist(),
            section_control=np.asarray(data['section_control']).tolist())
        if combined and not (live_suspension_variant or bedload_live_variant):
            manifest['boundary_events'].update(native_outlet_extension=int(data['cpu_outlet_extension']),
                constructor_outlet_extension=0,effective_reconstruction_outlet_extension=1)
        if live_suspension_variant or bedload_live_variant:
            manifest['boundary_events'].update(native_outlet_extension=0,
                constructor_outlet_extension=0,effective_reconstruction_outlet_extension=0,
                nfraction_series=np.asarray(data['nfraction_series']).tolist(),
                nhydro_series=np.asarray(data['nhydro_series']).tolist())
        if scheduled_inlet:
            manifest['boundary_events'].update(native_outlet_extension=0,
                constructor_outlet_extension=0,effective_reconstruction_outlet_extension=0,
                nfraction_series=np.asarray(data['nfraction_series']).tolist(),
                nhydro_series=np.asarray(data['nhydro_series']).tolist())
    write_json(destination/'run.json',manifest)
    began=time.perf_counter();current=start;steps=0;rejected=0;hint=dtmax;frame_index=0
    state='initializing';solver=None;graph=None;activity=None;event_hits=0;event_hit_rows=[];stack=ExitStack()
    stages=[];stage_times=[];stage_dts=[];accepted=[];accepted_times=[];trial_rows=[]
    scheduled_trial_rows=[];avalanche_trial_rows=[]
    a1_step_rows=[];a1_trial_rows=[];a1_stage_beds=[];a1_accepted_beds=[];a1_persistent={}
    live_knots=live_internal_knots(events,start,end) if bounded_live else np.empty(0)
    a1_knot_hits=np.zeros(len(live_knots),dtype=np.int32)
    live_label='graded_bedload' if general_bedload_variant else 'target_b' if target_b_variant else 'g2' if live_suspension_variant else 'a2' if secondary_live_variant else 'a1'
    live_manifest_key='graded_bedload_live' if general_bedload_variant else 'target_b_adaptation' if target_b_variant else 'live_suspension' if live_suspension_variant else 'live_secondary' if secondary_live_variant else 'a1_live_scheduled'
    fixed_step_times=[];fixed_step_dts=[]
    trace_steps=[];trace_times=[];trace_dts=[];trace_cfl=[]
    def progress(event,**extra):
        record=dict(event=event,status=state,backend='gpu',time_s=current,end_s=end,steps=steps,rejected=rejected,
            boundary_event_hits=event_hits,
            percent=100. if state=='completed' else 100.*(current-start)/(end-start),wall_s=time.perf_counter()-began,**extra)
        write_json(destination/'status.json',record);print(json.dumps(record,allow_nan=False),flush=True)
    def output(checkpoint_only=False):
        nonlocal frame_index
        fields=solver.fields()
        checkpoint=dict(**fields,time=np.float64(current),steps=np.int64(steps),rejected=np.int64(rejected),
            dt_hint=np.float64(hint),input_identity=np.asarray(input_identity),
            source_identity=np.asarray(source_identity),method_identity=np.asarray(method_identity))
        if events is not None:
            checkpoint.update(event_identity=np.asarray(events.identity),
                next_boundary_knot=np.float64(events.next_after(current)))
        if bounded_live:
            checkpoint.update({key:solver.cp.asnumpy(value) for key,value in a1_persistent.items()})
            checkpoint['a1_event_cursor']=np.int64(a1_event_cursor(events,current))
            checkpoint['a1_knot_hits']=a1_knot_hits.copy()
            checkpoint['a1_payload_sha256']=np.asarray(fields_digest(checkpoint))
        save_arrays(destination/'checkpoint.npz',**checkpoint)
        if not checkpoint_only:
            filename=f'frame_{frame_index:06d}.npz';save_arrays(destination/filename,**checkpoint)
            manifest['frames'].append(dict(file=filename,time_s=current,steps=steps,sha256=sha(destination/filename)))
            write_json(destination/'run.json',manifest);frame_index+=1
    try:
        progress('initializing')
        if source_stage_allowed:
            execution_source_folder,source_stage=stage_long_execution_sources(
                source_folder,stack)
        else:
            execution_source_folder=source_folder
            source_stage=dict(policy='authenticated_short_stage_v1',used=False,
                published_source=str(source_folder),
                published_probe_characters=len(str(source_folder/'reconstruction.cu')),
                staged_source_persisted=False,files_verified=0,
                reason='source-bound constructor route')
        manifest['source_execution_staging']=source_stage
        write_json(destination/'run.json',manifest)
        Solver,Graph=stack.enter_context(binding)
        if capture_stages:Solver=observed_combined_solver(Solver)
        solver=(Solver(constructor_data,source_folder=execution_source_folder) if combined or curved
                else Solver(constructor_data,source_folder=execution_source_folder,integrator=integrator,
                            scheduled=scheduled_inlet,bedform_variant=bedform_variant,
                            general_avalanche_variant=avalanche_variant,
                            physical_toe_mode2_variant=bank_toe_variant and not bank_avalanche_variant,
                            bank_avalanche_mode2_variant=bank_avalanche_variant,
                            g3_bank_secondary_variant=g3_variant,
                            distributed_bank_mode3_variant=(
                                distributed_bank_variant and not distributed_bank_avalanche_variant),
                            distributed_bank_avalanche_mode3_variant=(
                                distributed_bank_avalanche_variant),
                            bedload_adaptation_variant=adaptation_variant))
        if bounded_live:
            # The private V2 adapter zeros exposed diagnostics; the invalid
            # normal-root cache payload is otherwise undefined before step 1.
            for value in solver.boundary.normal_root.cache.values():value.fill(0)
            a1_persistent=a1_auxiliary_device_arrays(solver)
            schema=a1_checkpoint_extra_schema(data,len(live_knots))
            expected=set(schema)-{'bed_elevation','a1_event_cursor','a1_knot_hits'}
            if set(a1_persistent)!=expected or any(
                    value.shape!=schema[key][0] or value.dtype!=np.dtype(schema[key][1])
                    for key,value in a1_persistent.items()):
                raise RuntimeError('A1 resident auxiliary checkpoint schema differs from host preflight')
        if scheduled_inlet:
            if solver.inlet_schedule.events.identity!=events.identity:
                raise RuntimeError('Resident schedule and runner event identities differ')
            manifest['scheduled_inlet']=solver.inlet_schedule.receipt()
        if (not combined or live_suspension_variant or bedload_live_variant) and not curved and solver.exchange_ledger_algorithm!=ledger_algorithm.get('dispatch','none'):
            raise RuntimeError('Exchange-ledger algorithm dispatch differs from frozen method identity')
        if (combined and not (live_suspension_variant or bedload_live_variant)) or curved:
            solver.reconstruction.outlet_extension=1
        if curved:
            activity=qualified.Activity(solver)
            cp=solver.cp
            before={**{k:cp.asnumpy(v) for k,v in solver.state.items()},
                    **{'ledger_'+k:cp.asnumpy(v) for k,v in solver.ledger.items()}}
            activity.phase='preflight_plain'
            plain=qualified.host_trial(solver,solver.step(dtmax,start,commit=False))
            if int(plain['status']):raise RuntimeError('Qualified curved wet/dry plain preflight rejected')
        if launch_mode=='graph':
            if curved:activity.phase='graph_warmup'
            graph=Graph(solver)
        if curved:
            warm=qualified.accepted_unchanged(solver,before)
            if not all(warm.values()):raise RuntimeError('Qualified curved wet/dry warmup mutated accepted state')
            activity.phase='preflight_graph'
            replay=qualified.host_trial(solver,solver.step(dtmax,start,commit=False))
            parity={key:qualified.same_bits(plain[key],replay[key]) for key in plain}
            if not all(parity.values()) or not all(qualified.accepted_unchanged(solver,before).values()):
                raise RuntimeError('Qualified curved wet/dry plain/graph preflight parity failed')
            manifest['qualified_curved_wetdry']['preflight']=dict(plain_graph_bitwise=parity,
                accepted_state_ledger_unchanged=warm,plain_status=int(plain['status']),
                graph_status=int(replay['status']))
            activity.phase='evolution'
        if combined:solver.boundary.normal_root.invalidate_cache()
        if (combined and not bounded_live) or curved:
            manifest['executed_python_modules']=combined_module_receipt(source_folder,live)
        cp=solver.cp;props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        device=props['name'];device=device.decode() if isinstance(device,bytes) else str(device)
        manifest.update(device=device,cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cells=solver.nc,faces=solver.nf,fractions=solver.ng,
            diffusion_enabled=solver.diffusion is not None,
            suspended_enabled=bool(getattr(solver,'has_suspended',False)),source_bindings=bindings)
        if bounded_live:manifest[live_manifest_key]['constructor_receipt']=solver.candidate_receipt()
        elif combined:manifest['combined_receipt']=solver.receipt()
        bank_persistent=(bank_toe_persistent_arrays(solver) if bank_toe_variant else
            distributed_bank_persistent_arrays(solver) if distributed_bank_variant else {})
        if bank_toe_variant and len(bank_persistent)!=9:
            raise RuntimeError('Physical bank-toe variant lacks complete accepted restart arrays')
        if distributed_bank_variant and len(bank_persistent)!=7:
            raise RuntimeError('Distributed-bank variant lacks complete accepted restart arrays')
        if bounded_live and a1_restored is not None:
            arrays,scalars=a1_restored
            for key,array in solver.state.items():array.set(arrays[key])
            for key,array in solver.ledger.items():array.set(arrays['ledger_'+key])
            for key,array in a1_persistent.items():array.set(arrays[key])
            current=float(scalars['time']);steps=int(scalars['steps']);rejected=int(scalars['rejected'])
            hint=min(float(scalars['dt_hint']),dtmax)
            a1_knot_hits[:]=arrays['a1_knot_hits']
            if not np.array_equal(solver.fields()['bed_elevation'],arrays['bed_elevation']):
                raise ValueError('A1 resumed bed differs from authenticated accepted layers')
        elif resume is not None:
            with np.load(resume,allow_pickle=False) as checkpoint:
                arrays,scalars=validate_checkpoint(checkpoint,{k:v.shape for k,v in solver.state.items()},
                    {k:v.shape for k,v in solver.ledger.items()},input_identity=input_identity,
                    source_identity=source_identity,method_identity=method_identity,
                    events=events,secondary_model=int(data['secondary_flow_model']),start=start,end=end,
                    extra_schema={k:(v.shape,v.dtype) for k,v in bank_persistent.items()})
                if bank_toe_variant:
                    validate_bank_toe_checkpoint(arrays,data['is_toe_face'])
                if distributed_bank_variant:
                    validate_distributed_bank_checkpoint(arrays)
                for key,array in solver.state.items():array.set(arrays[key])
                for key,array in solver.ledger.items():array.set(arrays['ledger_'+key])
                for key,array in bank_persistent.items():array.set(arrays[key])
                current=float(scalars['time']);steps=int(scalars['steps']);rejected=int(scalars['rejected']);hint=min(float(scalars['dt_hint']),dtmax)
            if combined and not bounded_live:solver.boundary.normal_root.invalidate_cache()
        save_arrays(destination/'geometry.npz',xy=data['xy'],area=data['area'],mid=data['mid'],
                    owner=data['owner'],neighbour=data['neighbour'],normal=data['normal'],length=data['length'],
                    rock=data['rock'],manning_cell=data['manning_cell'])
        if bank_toe_variant:
            map_names=('bank_material_mask','physical_toe_cell','physical_toe_donor',
                'physical_toe_face','physical_toe_distance','is_toe_cell','is_toe_donor',
                'is_toe_face','candidate_bank','candidate_toe','physical_toe_status')
            map_path=destination/'bank_toe_map.npz'
            save_arrays(map_path,**{key:np.asarray(data[key]) for key in map_names})
            manifest['physical_bank_toe']=dict(map_file=map_path.name,map_sha256=sha(map_path),
                mode=2,capacity_fraction=float(data['bank_toe_capacity_fraction']),
                critical_undercut_m=float(data['bank_critical_undercut_m']))
        if distributed_bank_variant:
            mask_path=destination/'distributed_bank_mask.npz'
            save_arrays(mask_path,bank_material_mask=np.asarray(data['bank_material_mask']))
            manifest['distributed_bank']=dict(mask_file=mask_path.name,
                mask_sha256=sha(mask_path),mode=3,
                transport_fraction=float(data['bank_distributed_transport_fraction']),
                critical_slope=float(data['bank_distributed_critical_slope']),
                critical_undercut_m=float(data['bank_critical_undercut_m']))
        if capture_stages:
            accepted.append(solver.fields());accepted_times.append(current)
        if bounded_live and capture_step_trace:
            a1_accepted_beds.append(solver.fields()['bed_elevation'].copy())
        state='running';output();progress('started',device=device)
        fixed_start_steps=steps
        process_initial_rejected=rejected
        initial_euler_calls=solver.euler_calls
        last_progress=time.perf_counter()
        # Derive every output target from its integer index. Repeatedly adding
        # ``interval`` drifts by a few ULPs; a cut run then clips its last
        # step to the exact end while the uninterrupted run takes the drifted
        # target, breaking bitwise restart at an otherwise identical clock.
        target_index=math.floor((current-start)/interval+1.e-10)+1
        target=min(end,start+target_index*interval)
        while current<end-1.e-12:
            if cancelled() or (stop_file is not None and filesystem_path(stop_file).exists()):state='cancelled';break
            if max_steps is not None and steps>=max_steps:state='step_limit';break
            if curved:
                unclipped=dtmax;dt=dtmax;clipped=False;clock=None
                if steps*dtmax!=current or current+dt>end+1.e-12:
                    raise RuntimeError('Qualified curved wet/dry fixed clock changed')
            elif events is None:
                unclipped=hint;dt=min(hint,target-current);clipped=dt<unclipped;clock=None
            else:
                clock=events.trial(current,hint,output=target,end=end)
                unclipped=clock.unclipped_hint;dt=clock.dt;clipped=clock.scheduled_clip
            pretrial_hash=fields_digest(solver.fields()) if capture_stages else None
            if curved:activity.step=steps+1
            for attempt in range(1,36):
                if not math.isfinite(dt) or current+dt<=current:raise RuntimeError('Nonadvancing adaptive timestep')
                result=solver.step(dt,current)
                code=int(result['status'].item());peak=float(result['cfl'].item())
                if bounded_live:
                    a1_trial_rows.append(dict(step=steps+1,attempt=attempt,time_s=current,
                        dt_s=dt,status=code,cfl=None if not math.isfinite(peak) else peak,
                        accepted=code==0,next_boundary_knot_s=(None if clock.boundary_knot>1.e300
                                                                else clock.boundary_knot)))
                if scheduled_inlet:
                    scheduled_trial_rows.append(dict(step=steps+1,attempt=attempt,time_s=current,dt_s=dt,
                        euler_stage_times_s=[current,current+dt,current+.5*dt],
                        next_boundary_knot_s=None if clock.boundary_knot>1.e300 else clock.boundary_knot,
                        status=code,cfl=None if not math.isfinite(peak) else peak,accepted=code==0))
                if avalanche_variant:
                    avalanche_trial_rows.append(dict(step=steps+1,attempt=attempt,time_s=current,dt_s=dt,
                        euler_stage_times_s=[current,current+dt,current+.5*dt],
                        status=code,accepted=code==0,
                        euler_substeps=list(result['avalanche_substeps'])))
                if bank_avalanche_variant or distributed_bank_avalanche_variant:
                    avalanche_trial_rows.append(dict(step=steps+1,attempt=attempt,time_s=current,dt_s=dt,
                        placement='once_post_rk_after_bank_failure',status=code,accepted=code==0,
                        post_rk_substeps=result['avalanche_substeps'][0]))
                if capture_stages:
                    post_hash=fields_digest(solver.fields())
                    trial_rows.append(dict(accepted_step=steps+1,attempt=attempt,time_s=current,dt_s=dt,
                        status=code,cfl=peak,pretrial_accepted_sha256=pretrial_hash,
                        post_trial_accepted_sha256=post_hash,committed=code==0))
                    if code and post_hash!=pretrial_hash:
                        raise RuntimeError('Rejected GPU trial mutated accepted state')
                if not code:break
                if curved:raise RuntimeError('Qualified curved wet/dry fixed-step trial rejected; no adaptive retry')
                rejected+=1;dt=retry_step(dt,peak,float(data['cfl_target']))
            else:raise RuntimeError(f'35 rejected GPU trials at t={current:g}')
            if curved:activity.accept()
            if capture_stages:
                stages.append(combined_stage_fields(solver));stage_times.append(current);stage_dts.append(dt)
            hint=(dtmax if curved else next_hint(dt,peak,float(data['cfl_target']),dtmax,
                scheduled_clip=clipped,first_attempt=attempt==1,unclipped=unclipped))
            steps+=1
            current=steps*dtmax if curved else clock.accepted_time(dt) if clock is not None else current+dt
            if bounded_live and capture_step_trace:
                cp=solver.cp
                schedule={key:cp.asnumpy(value) for key,value in solver.schedule.outputs.items()}
                normal={key:cp.asnumpy(value) for key,value in solver.boundary.normal_records.items()}
                outlet={key:cp.asnumpy(value) for key,value in solver.boundary.outlet_records.items()}
                a1_stage_beds.append(normal['bed'].copy())
                a1_step_rows.append(dict(step=steps,start_s=clock.current,end_s=current,
                    dt_s=dt,cfl=peak,status=code,
                    stage_clocks_s=schedule['stage_time'].tolist(),
                    inlet_q_m3_s=schedule['hydraulic'][:,0,0].tolist(),
                    inlet_feed_kg_s=schedule['feed'][:,:,0].tolist(),
                    normal_recorded=normal['recorded'].tolist(),
                    normal_root_status=normal['root_status'].tolist(),
                    normal_trace_status=normal['trace_status'].tolist(),
                    normal_failure_latch=normal['failure_latch'].tolist(),
                    outlet_recorded=outlet['recorded'].tolist(),
                    outlet_status=outlet['status'].tolist(),
                    outlet_failure_latch=outlet['failure_latch'].tolist()))
                a1_accepted_beds.append(solver.fields()['bed_elevation'].copy())
            if capture_step_trace:
                trace_steps.append(steps);trace_times.append(current)
                trace_dts.append(dt);trace_cfl.append(peak)
            if curved:
                fixed_step_times.append(current);fixed_step_dts.append(dt)
            if capture_stages:
                accepted.append(solver.fields());accepted_times.append(current)
            if events is not None and start<clock.boundary_knot<end and abs(current-clock.boundary_knot)<=1.e-12:
                event_hits+=1;event_hit_rows.append(dict(step=steps,time_s=current,knot_s=clock.boundary_knot,dt_s=dt))
                if bounded_live:
                    knot_index=int(np.flatnonzero(np.isclose(live_knots,clock.boundary_knot,
                        rtol=0,atol=1.e-12))[0])
                    a1_knot_hits[knot_index]=1
                if bounded_live and capture_step_trace:
                    knot_fields=solver.fields()
                    a1_step_rows[-1]['knot_ledger']=dict(
                        water_m3=knot_fields['ledger_water'].tolist(),
                        bed_kg=knot_fields['ledger_bed'].tolist())
            if current>=target-1.e-12:
                output();progress('output');last_progress=time.perf_counter()
                target_index+=1;target=min(end,start+target_index*interval)
            elif time.perf_counter()-last_progress>=15:
                progress('progress');last_progress=time.perf_counter()
        if state=='running':state='completed'
        # An output at this accepted clock already wrote the same checkpoint.
        # Avoid a redundant Windows replace of a file that another reader or
        # indexer may still be opening after the immediately preceding output.
        if not (manifest['frames'] and manifest['frames'][-1]['time_s']==current):
            output()
        progress('finished')
        if capture_stages and stages:
            save_arrays(destination/'stages.npz',**{key:np.stack([row[key] for row in stages]) for key in stages[0]},
                        time=np.asarray(stage_times),dt=np.asarray(stage_dts))
            fields={key:np.stack([row[key] for row in accepted]) for key in accepted[0]}
            fields['time']=np.asarray(accepted_times);fields['bed']=fields['bed_elevation']
            save_arrays(destination/'accepted.npz',**fields)
            manifest['stage_diagnostics']=dict(file='stages.npz',sha256=sha(destination/'stages.npz'),
                accepted_file='accepted.npz',accepted_sha256=sha(destination/'accepted.npz'),
                accepted_steps=len(stages),four_euler_queries_per_step=True,performance_timing_claim=False)
        if capture_stages:
            write_json(destination/'trials.json',trial_rows)
            manifest['trial_diagnostics']=dict(file='trials.json',sha256=sha(destination/'trials.json'),
                attempts=len(trial_rows),accepted=sum(row['committed'] for row in trial_rows),
                rejected=sum(not row['committed'] for row in trial_rows),
                rejected_state_identity_checked=True)
        if scheduled_inlet:
            write_json(destination/'schedule_trials.json',scheduled_trial_rows)
            manifest['scheduled_inlet']['trial_trace']=dict(file='schedule_trials.json',
                sha256=sha(destination/'schedule_trials.json'),attempts=len(scheduled_trial_rows),
                accepted=sum(row['accepted'] for row in scheduled_trial_rows),
                rejected=sum(not row['accepted'] for row in scheduled_trial_rows),
                fields_are_host_clock_and_trial_status_only=True)
        if bounded_live:
            trial_file=live_label+'_trials.json'
            write_json(destination/trial_file,a1_trial_rows)
            manifest[live_manifest_key]['trial_trace']=dict(file=trial_file,
                sha256=sha(destination/trial_file),attempts=len(a1_trial_rows),
                accepted=sum(row['accepted'] for row in a1_trial_rows),
                rejected=sum(not row['accepted'] for row in a1_trial_rows))
            if capture_step_trace:
                stage_file=live_label+'_stage_trace.json'
                stage_bed_file=live_label+'_stage_beds.npz'
                accepted_bed_file=live_label+'_accepted_beds.npz'
                write_json(destination/stage_file,a1_step_rows)
                save_arrays(destination/stage_bed_file,bed=np.asarray(a1_stage_beds))
                save_arrays(destination/accepted_bed_file,bed=np.asarray(a1_accepted_beds))
                manifest[live_manifest_key]['stage_trace']=dict(file=stage_file,
                    sha256=sha(destination/stage_file),
                    stage_bed_file=stage_bed_file,
                    stage_bed_sha256=sha(destination/stage_bed_file),
                    accepted_bed_file=accepted_bed_file,
                    accepted_bed_sha256=sha(destination/accepted_bed_file),
                    accepted_steps=len(a1_step_rows),
                    resident_stage_records=3 if (live_suspension_variant or bedload_live_variant) else 4)
        if avalanche_variant or bank_avalanche_variant or distributed_bank_avalanche_variant:
            write_json(destination/'avalanche_trials.json',avalanche_trial_rows)
            count=lambda row: sum(row['euler_substeps']) if avalanche_variant else row['post_rk_substeps']
            accepted_substeps=sum(count(row) for row in avalanche_trial_rows if row['accepted'])
            attempted_substeps=sum(count(row) for row in avalanche_trial_rows)
            manifest['general_avalanche']=dict(accepted_substeps_in_process=accepted_substeps,
                attempted_substeps_in_process=attempted_substeps,
                trial_trace=dict(file='avalanche_trials.json',sha256=sha(destination/'avalanche_trials.json'),
                    attempts=len(avalanche_trial_rows),accepted=sum(row['accepted'] for row in avalanche_trial_rows),
                    rejected=sum(not row['accepted'] for row in avalanche_trial_rows),
                    accepted_substeps=accepted_substeps,attempted_substeps=attempted_substeps))
        euler_calls_in_process=solver.euler_calls-initial_euler_calls
        expected_euler_calls=method['euler_stages']*((steps-fixed_start_steps)+(rejected-process_initial_rejected))
        if euler_calls_in_process!=expected_euler_calls:
            raise RuntimeError('GPU Euler-call count does not match accepted/rejected trials')
        if capture_step_trace:
            if (len(trace_steps)!=steps-fixed_start_steps or
                    trace_steps!=list(range(fixed_start_steps+1,steps+1)) or
                    (trace_times and trace_times[-1]!=current) or
                    not np.isfinite(trace_times+trace_dts+trace_cfl).all() or
                    any(dt<=0 for dt in trace_dts)):
                raise RuntimeError('GPU accepted-step trace differs from accepted clock')
            trace_path=destination/'accepted_step_trace.npz'
            save_arrays(trace_path,step=np.asarray(trace_steps,dtype=np.int64),
                        time=np.asarray(trace_times,dtype=np.float64),
                        dt=np.asarray(trace_dts,dtype=np.float64),
                        cfl=np.asarray(trace_cfl,dtype=np.float64))
            manifest['accepted_step_trace']=dict(file=trace_path.name,sha256=sha(trace_path),
                accepted_steps=len(trace_steps),
                first_step=trace_steps[0] if trace_steps else None,
                last_step=trace_steps[-1] if trace_steps else None,
                contains_only_steps_accepted_in_this_process=True)
        manifest.update(status=state,final_time_s=current,steps=steps,rejected=rejected,
            steps_initial_in_process=fixed_start_steps,rejected_initial_in_process=process_initial_rejected,
            euler_calls_initial=initial_euler_calls,euler_calls_final=solver.euler_calls,
            euler_calls_in_process=euler_calls_in_process,
            euler_calls_expected_from_attempts=expected_euler_calls,
            boundary_event_hits=event_hits,total_wall_s=time.perf_counter()-began)
        if bounded_live:
            manifest[live_manifest_key]['total_knot_hits']=a1_knot_hits.tolist()
            if state=='completed' and not np.array_equal(a1_knot_hits,np.ones(len(live_knots),np.int32)):
                raise RuntimeError('Live scheduled trajectory missed an internal forcing knot')
        if events is not None:
            manifest['boundary_events']['internal_hits']=event_hits
            manifest['boundary_events']['hit_rows']=event_hit_rows
        if graph is not None:manifest['transport_graph']=graph.receipt()
        if curved:
            fixed_ok=(len(fixed_step_dts)==steps-fixed_start_steps and all(dt==dtmax for dt in fixed_step_dts)
                      and all(t==(fixed_start_steps+i+1)*dtmax for i,t in enumerate(fixed_step_times)))
            if not fixed_ok:raise RuntimeError('Qualified curved wet/dry accepted clock differs')
            save_arrays(destination/'fixed_step_trace.npz',
                step=np.arange(fixed_start_steps+1,steps+1,dtype=np.int64),time=np.asarray(fixed_step_times),
                dt=np.asarray(fixed_step_dts))
            manifest['qualified_curved_wetdry']['fixed_step_trace_sha256']=sha(destination/'fixed_step_trace.npz')
            manifest['qualified_curved_wetdry']['all_accepted_steps_fixed_dt_verified']=fixed_ok
            manifest['qualified_curved_wetdry']['fixed_trace_start_step']=fixed_start_steps
            write_json(destination/'filter_history.json',activity.filters)
            write_json(destination/'source_history.json',activity.sources)
            manifest['qualified_curved_wetdry']['activity']=activity.summary()
            manifest['qualified_curved_wetdry']['filter_history_sha256']=sha(destination/'filter_history.json')
            manifest['qualified_curved_wetdry']['source_history_sha256']=sha(destination/'source_history.json')
        if (combined and not bounded_live) or curved:
            manifest['executed_python_modules']=combined_module_receipt(source_folder,live)
        stack.close()
        write_json(destination/'run.json',manifest)
        return manifest
    except Exception as exc:
        state='failed'
        try:stack.close()
        except Exception as source_exc:manifest['source_exit_error']=str(source_exc)
        if capture_stages and trial_rows:
            write_json(destination/'trials.json',trial_rows)
            manifest['trial_diagnostics']=dict(file='trials.json',sha256=sha(destination/'trials.json'),
                attempts=len(trial_rows),accepted=sum(row['committed'] for row in trial_rows),
                rejected=sum(not row['committed'] for row in trial_rows))
        if scheduled_inlet and scheduled_trial_rows:
            write_json(destination/'schedule_trials.json',scheduled_trial_rows)
            manifest.setdefault('scheduled_inlet',{})['trial_trace']=dict(file='schedule_trials.json',
                sha256=sha(destination/'schedule_trials.json'),attempts=len(scheduled_trial_rows),
                accepted=sum(row['accepted'] for row in scheduled_trial_rows),
                rejected=sum(not row['accepted'] for row in scheduled_trial_rows))
        if bounded_live and a1_trial_rows:
            trial_file=live_label+'_trials.json'
            write_json(destination/trial_file,a1_trial_rows)
            manifest[live_manifest_key]['trial_trace']=dict(file=trial_file,
                sha256=sha(destination/trial_file),attempts=len(a1_trial_rows),
                accepted=sum(row['accepted'] for row in a1_trial_rows),
                rejected=sum(not row['accepted'] for row in a1_trial_rows))
        if (avalanche_variant or bank_avalanche_variant or
                distributed_bank_avalanche_variant) and avalanche_trial_rows:
            write_json(destination/'avalanche_trials.json',avalanche_trial_rows)
            count=lambda row: sum(row['euler_substeps']) if avalanche_variant else row['post_rk_substeps']
            accepted_substeps=sum(count(row) for row in avalanche_trial_rows if row['accepted'])
            attempted_substeps=sum(count(row) for row in avalanche_trial_rows)
            manifest['general_avalanche']=dict(accepted_substeps_in_process=accepted_substeps,
                attempted_substeps_in_process=attempted_substeps,
                trial_trace=dict(file='avalanche_trials.json',sha256=sha(destination/'avalanche_trials.json'),
                    attempts=len(avalanche_trial_rows),accepted=sum(row['accepted'] for row in avalanche_trial_rows),
                    rejected=sum(not row['accepted'] for row in avalanche_trial_rows),
                    accepted_substeps=accepted_substeps,attempted_substeps=attempted_substeps))
        if solver is not None:
            try:output(checkpoint_only=True)
            except Exception:pass
        manifest.update(status=state,final_time_s=current,steps=steps,rejected=rejected,error=str(exc))
        write_json(destination/'run.json',manifest)
        progress('failed',error=str(exc));raise


def run_cpu(*,input_path,output_path,cpu_executable=None,until=None,output_every=None,dt_max=None,
            max_steps=None,stop_file=None,resume=None,**unused):
    """Explicit existing CPU executable; no automatic route from a GPU failure."""
    from validation.gpu_fwave.coupled_reference import copy_inputs
    if any(v is not None for v in (until,output_every,dt_max,max_steps,stop_file,resume)):
        raise ValueError('CPU route uses native deck controls; GPU runner overrides do not apply')
    if cpu_executable is None:raise ValueError('CPU backend requires --cpu-executable explicitly')
    executable=filesystem_path(cpu_executable).resolve();deck=filesystem_path(input_path).resolve();output=filesystem_path(output_path).resolve()
    if not executable.is_file() or not deck.is_file():raise ValueError('CPU executable and native input deck must exist')
    origin=copy_inputs(deck,output)
    env={k:v for k,v in os.environ.items() if not k.startswith('BRAIDS_')}
    write_json(output/'backend.json',dict(backend='cpu',selected_by='user',executable=str(executable),
        executable_sha256=sha(executable),origin=origin,automatic_backend_selection=False))
    with (output/'execution.log').open('w',encoding='utf-8') as log:
        completed=subprocess.run([str(executable),deck.name],cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT)
    completed.check_returncode()
    return dict(backend='cpu',status='completed',output=str(output))
