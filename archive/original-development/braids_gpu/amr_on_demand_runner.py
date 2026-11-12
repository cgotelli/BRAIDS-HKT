"""Common CLI lifecycle for the on-demand synchronous GPU bank-AMR route.

This is an explicit experimental backend, not an automatic mesh-size choice.
The augmented f-wave operators are retained. Revised bank-interface and normal
root kernels are paired explicitly with their Python ABI in each new archive;
preparation/derived geometry stay on host while timesteps/transfers use CUDA.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .amr_on_demand import GeometrySession,OnDemandBankRoute,sha,digest,verify_runtime
from .amr_prepared import load_amr_prepared
from .atomic_output import (replace_with_windows_retry, save_arrays_atomic,
                            write_json_atomic)


ROOT=Path(__file__).resolve().parents[1]
VARIANT='on_demand_bank_amr_ssp33'
GENERAL_VARIANT='on_demand_transport_amr_ssp33'
GENERAL_IMPLICIT_VARIANT='on_demand_transport_amr_implicit_fwave_midpoint'
RECURSIVE_BANK_CHECKPOINT='recursive_bank_amr_v1'
RECURSIVE_GENERAL_CHECKPOINT='recursive_general_amr_v1'
RECURSIVE_LEVEL_STEPPING='recursive_ssprk33_fine_replaced_register_v1'
RECURSIVE_GENERAL_AVALANCHE_POLICY='parent_post_reflux_once_per_macro_v1'
RECURSIVE_GENERAL_VISCOSITY_POLICY='parent_post_reflux_ssprk2_once_per_macro_v1'
RECURSIVE_GENERAL_VISCOSITY_METHOD=dict(
    policy=RECURSIVE_GENERAL_VISCOSITY_POLICY,
    placement='external_parent_post_reflux_before_avalanche',
    calls_per_macro=1,
    integrator='SSPRK2_subcycled_area_weighted_velocity_L2_limited',
    limited_norm='area_weighted_velocity_L2',
    area_weighted_velocity_l2_nonincrease=True,
    pairwise_velocity_face_flux=True,
    momentum_conservation='variable_depth_not_exact',
    dynamic_wet_geometry=True,
    general_avalanche_modes=[0,1],
    composition_order='viscosity_then_avalanche')
RECURSIVE_BEDFORM_TRACE_KEYS=('bedform_stage_evaluations_per_level',
    'bedform_friction_evaluations_per_level')
RECURSIVE_BEDFORM_FRAME_KEYS=('bedform_effective_manning',
    'bedform_dune_height','bedform_dune_length','bedform_total_ks',
    'bedform_iterations')
RUNTIME=ROOT/'validation/g4_bank_amr/runs/on_demand_geometry_runtime_20261008_v3'
EXPORTER_SHA='98806f1cb766ea4b5bdeefcd86d3c343a3bbee1f1fe255cd070c57b73048ef9a'


def require(value,message):
    if not value:
        raise ValueError(message)


def recursive_geometry_deck_adapter(source, *, expected_base_version):
    """Expose a current deck's unchanged physical payload to a frozen reader.

    QGIS v54 decks wrap the older physical bank geometry in length-delimited
    method/execution identity records.  The signed recursive v38/v40 native
    initializers intentionally predate those records.  This orchestration-only
    adapter removes exactly the v50--v54 metadata envelopes, retains the deck
    fingerprint and complete physical payload, and publishes a content-addressed
    durable input for later on-demand operators and restart.
    """
    source=Path(source).resolve()
    require(expected_base_version in (38,40),
            'Recursive geometry adapter requires physical deck v38 or v40')
    rows=source.read_text(encoding='ascii').splitlines()
    require(len(rows)>10 and len(rows[1])==64 and
            all(character in '0123456789abcdef' for character in rows[1]),
            'Current recursive geometry deck lacks its authenticated fingerprint')
    logical=[rows[0],*rows[2:]]
    cursor=0;versions=[]
    try:
        outer=int(logical[cursor])
        while int(logical[cursor])>41:
            version=int(logical[cursor]);versions.append(version)
            if version==54:
                counts=[int(value) for value in logical[cursor+1].split()]
                require(len(counts)==5 and counts[0]==1 and
                        all(value>=0 for value in counts[1:]),
                        'Invalid v54 recursive geometry identity envelope')
                cursor+=2+counts[2]+counts[4]
                require(int(logical[cursor])==53,
                        'V54 recursive geometry adapter did not find nested v53')
            elif version==53:
                components=int(logical[cursor+1])
                require(components>0,
                        'Invalid v53 recursive geometry component count')
                cursor+=2
                for _ in range(components):
                    header=logical[cursor].split()
                    require(len(header)==4 and int(header[3])>0,
                            'Invalid v53 recursive geometry component header')
                    cursor+=1+int(header[3])
                require(int(logical[cursor])==52,
                        'V53 recursive geometry adapter did not find nested v52')
            elif version==52:
                header=[int(value) for value in logical[cursor+1].split()]
                require(len(header)==4 and header[0]==2 and
                        all(value>0 for value in header[1:]),
                        'Invalid v52 recursive geometry composition envelope')
                fractions,chunks=header[1],header[3]
                cursor+=2+4+fractions+1+chunks
                require(int(logical[cursor])==2,
                        'Invalid v52 recursive geometry execution-profile schema')
                preimage_bytes=int(logical[cursor+3])
                require(preimage_bytes>0,
                        'Invalid v52 recursive geometry execution-profile size')
                cursor+=4+(2*preimage_bytes+127)//128
                require(int(logical[cursor])==51,
                        'V52 recursive geometry adapter did not find nested v51')
            elif version==51:
                cursor+=6
                require(int(logical[cursor])==50,
                        'V51 recursive geometry adapter did not find nested v50')
            elif version==50:
                cursor+=8
            else:
                raise ValueError('Frozen recursive geometry reader cannot adapt '
                                 f'deck version {version}')
        base=int(logical[cursor])
    except (IndexError,TypeError,ValueError) as error:
        if isinstance(error,ValueError) and str(error).startswith(
                ('Invalid ','V','Frozen ')):
            raise
        raise ValueError('Malformed recursive geometry identity envelope') from error
    require(base==expected_base_version,
            f'Recursive geometry adapter expected physical v{expected_base_version}, got v{base}')
    source_sha=sha(source)
    if not versions:
        adapted=source;adapted_sha=source_sha
    else:
        require(versions==[54,53,52,51,50],
                'Recursive geometry identity envelope chain differs')
        payload=('\n'.join((logical[cursor],rows[1],
                            *logical[cursor+1:]))+'\n').encode('ascii')
        adapted_sha=hashlib.sha256(payload).hexdigest()
        # Keep the durable name short: it is copied below a private native
        # bootstrap directory and must remain below legacy Windows MAX_PATH.
        # The full source/adapted hashes and collision check remain authoritative.
        adapted=source.with_name(f'.br{base}_{source_sha[:16]}.dat')
        if adapted.exists():
            require(sha(adapted)==adapted_sha,
                    'Existing recursive geometry adapter differs')
        else:
            descriptor,temporary_name=tempfile.mkstemp(
                prefix='.braids-recursive-',suffix='.partial',dir=source.parent)
            temporary=Path(temporary_name)
            try:
                with os.fdopen(descriptor,'wb') as stream:
                    stream.write(payload)
                replace_with_windows_retry(temporary,adapted)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    receipt=dict(schema=1,applied=bool(versions),outer_version=outer,
        base_version=base,expected_base_version=expected_base_version,
        stripped_metadata_versions=versions,source_deck_sha256=source_sha,
        adapted_deck_name=adapted.name,adapted_deck_sha256=adapted_sha,
        physical_payload_unchanged=True)
    receipt['identity']=digest(receipt)
    return adapted,receipt


def described_bank_mode(bank_description):
    """Resolve an exact bank composition before selecting its native exporter."""
    require(isinstance(bank_description,dict),
            'Exact recursive bedform-bank preparation needs bank metadata')
    mode={'noncohesive_transport_capacity':2,
          'distributed_slope_transport':3}.get(
              bank_description.get('toe_erosion_mode'))
    require(mode in (2,3),
            'Exact recursive bedform-bank preparation needs an explicit '
            'physical-toe or distributed toe-erosion mode')
    return mode


def recursive_bedform_step_diagnostics(record, enabled):
    """Validate and normalize the physical level-local bedform work trace."""
    from .general_live_transport import RECURSIVE_BEDFORM_ORDER
    require(type(enabled) is bool,
            'Recursive bedform diagnostic switch must be boolean')
    order=record.get('bedform_order')
    arrays=[]
    for name in RECURSIVE_BEDFORM_TRACE_KEYS:
        value=record.get(name)
        require(isinstance(value,(list,tuple)) and len(value)>0 and
                all(not isinstance(item,(bool,np.bool_)) and
                    isinstance(item,(int,np.integer)) and int(item)>=0
                    for item in value),
                'Recursive general bedform diagnostic differs: '+name)
        arrays.append([int(item) for item in value])
    stage,friction=arrays
    total=record.get('bedform_evaluations_per_macro')
    require(not isinstance(total,(bool,np.bool_)) and
            isinstance(total,(int,np.integer)) and int(total)>=0 and
            len(stage)==len(friction) and int(total)==sum(stage)+sum(friction),
            'Recursive general bedform total diagnostic differs')
    if enabled:
        require(order==RECURSIVE_BEDFORM_ORDER and sum(friction)>0 and
                all(s==3*f for s,f in zip(stage,friction,strict=True)),
                'Recursive general bedform ON diagnostic differs')
    else:
        require(order=='disabled' and int(total)==0 and
                not any(stage) and not any(friction),
                'Recursive general bedform OFF diagnostic differs')
    return dict(bedform_order=order,
        bedform_stage_evaluations_per_level=stage,
        bedform_friction_evaluations_per_level=friction,
        bedform_evaluations_per_macro=int(total))


def scope(session, integrator=None):
    """Host-only admission; unsupported physics are never disabled implicitly."""
    from .runner import reject_deferred_public_controls
    from validation.gpu_fwave.coupled_solver import normalized,validate_scope
    data=session.bootstrap.topologies[0].data
    bank_mode=int(data['bank_toe_erosion_mode'])
    recursive=bool(session.selector_arguments['settings']['subcycling'])
    if bank_mode == 0:
        require(getattr(session, 'general_transport', False) is True and
                not session.evolving_banks,
                'General transport AMR requires explicit nonbank preparation')
        from .general_live_transport import (_validate_prepared,
            recursive_bedform_envelope)
        declared_implicit='suspended_amr_contract' in data
        selected=('implicit_fwave_midpoint' if declared_implicit
                  else 'ssp33') if integrator is None else integrator
        require(selected in ('ssp33','implicit_fwave_midpoint') and
                (selected=='implicit_fwave_midpoint') is declared_implicit,
                'General AMR selected integrator and prepared ownership route differ')
        require(not recursive or selected=='ssp33',
                'Recursive implicit f-wave AMR is not qualified')
        bedform_envelope=bool(recursive and recursive_bedform_envelope(data))
        closed_m2c=(declared_implicit and
                    np.asarray(data['mode']).shape == (0,))
        require((np.array_equal(data['mode'], [9, 4]) or bedform_envelope or
                 closed_m2c) and
                int(data['amr_enabled']) == 0 and float(data['time']) == 0.,
                'General transport AMR needs original live normal/critical state '
                'or the exact recursive mode1/mode2 bedform pair')
        reject_deferred_public_controls(data)
        _validate_prepared(data,integrator=selected)
        if recursive:
            viscosity=float(data['horizontal_viscosity'])
            ordinary=(int(data['bedlaw']) == 2 and
                int(data['exchange_mode']) == 2 and
                int(data['secondary_flow_model']) == 2 and
                np.isfinite(viscosity) and viscosity >= 0. and
                (viscosity == 0. or
                 (int(data['horizontal_viscosity_mode']) == 1 and
                  int(data['horizontal_viscosity_geometry']) == 1)) and
                not int(data.get('bedform_roughness', 0)) and
                not int(data['bank_failure']))
            require(bedform_envelope or ordinary,
                    'Recursive general AMR requires either the exact mode1/mode2 '
                    'bedform ON/OFF pair or MPM2/exchange2/prognostic-secondary; '
                    'positive viscosity requires native mode1/geometry1')
        return data
    require(integrator in (None,'ssp33'),
            'Bank AMR supports explicit SSPRK33 only')
    require(bank_mode in (2,3) and (bank_mode==2 or session.evolving_banks),
            'Distributed bank AMR requires explicit dynamic-bank preparation')
    reject_deferred_public_controls(data)
    live=np.array_equal(data['mode'],[9,4])
    if recursive:
        require((bank_mode == 2) or (bank_mode == 3 and session.evolving_banks),
                'Recursive bank AMR requires physical-toe mode 2 or explicit evolving mode 3')
        if live:
            require(bank_mode == 3,
                    'Recursive surveyed bank sections currently require distributed mode 3')
            from .amr_boundary_forcing import validate_surveyed_scope
            validate_surveyed_scope(normalized(data))
        else:
            composed=bool(getattr(session,'bedform_bank_composition',False) and
                          int(data.get('bedform_roughness', 0)))
            require(not composed or getattr(session,'bedform_bank_mode',None)==bank_mode,
                    'Recursive bedform-bank initializer mode differs')
            flags=(dict(recursive_bedform_bank_avalanche_mode2_variant=True,
                        evolving_toe_variant=session.evolving_banks,
                        external_physical_toe=True) if composed and bank_mode==2 else
                   dict(bank_avalanche_mode2_variant=True,
                        g3_bank_secondary_variant=True,
                        evolving_toe_variant=session.evolving_banks,
                        external_physical_toe=True) if bank_mode == 2 else
                   dict(recursive_bedform_bank_avalanche_mode3_variant=True)
                   if composed else
                   dict(distributed_bank_avalanche_mode3_variant=True))
            validate_scope(normalized(data),external_boundary_schedule=True,
                external_level_secondary=int(data['secondary_flow_model']) in (1,2), **flags)
        require(int(data['amr_enabled'])==0 and float(data['time'])==0.,
                'Recursive AMR geometry must contain the original cold initial state')
        return data
    if live:
        require(session.evolving_banks,
                'Live AMR sections require the evolving-bank preparation contract')
        require(int(data['hiding_exposure']) in (0,4),
                'Live bank AMR currently requires hiding 0/4')
        from .general_live_bank import _validate_prepared
        _validate_prepared(data)
    else:
        flags=(dict(bank_avalanche_mode2_variant=True,g3_bank_secondary_variant=True,
                    evolving_toe_variant=session.evolving_banks) if bank_mode==2 else
               dict(distributed_bank_avalanche_mode3_variant=True))
        validate_scope(normalized(data),**flags)
        require(np.array_equal(data['mode'],[2]) and
                all(not np.any(data[name]) for name in ('nseries','nhydro_series','nfraction_series')),
                'Legacy AMR boundaries require one constant stage outlet')
    require(int(data['secondary_flow_model'])==(0 if bank_mode==3 and not live else 2) and
            int(data['exchange_mode'])==0 and
            int(data['amr_enabled'])==0,
            'On-demand AMR requires a supported bank/secondary pairing and no suspension')
    require(float(data['time'])==0.,'On-demand input must contain original cold initial state')
    return data


def prepare(deck,output,*,stop_file=None,progress=None,evolving_banks=False,
            selector_policy=None,recursive_bank_bedform_avalanche=False):
    deck,output=Path(deck).resolve(),Path(output).resolve()
    description_path=deck.with_name(deck.name+'.json')
    if description_path.is_file():
        description=json.loads(description_path.read_text(encoding='utf-8'))
        bank_description=description.get('bank_failure')
        require(isinstance(bank_description,dict) and
                type(bank_description.get('enabled')) is bool,
                'AMR preparation needs an explicit ordinary bank control')
        general=bank_description['enabled'] is False
    else:
        # Historical bank preparations predate the descriptive JSON sidecar.
        # Keep that route readable; numerical scope() still rejects a nonbank
        # deck after the authenticated native initializer has decoded it.
        general=False
        bank_description=None
    # The ordinary v49 sidecar is the authority for selecting the M2-C
    # implicit route.  Do this before publishing the preparation request so a
    # law-11/mode-4 package is never mislabeled as the historical SSPRK33
    # general variant.  The same binding is independently revalidated while
    # loading each selected operator.
    preparation_integrator='ssp33'
    if general:
        from .implicit_suspension_amr import prepared_binding_from_case_metadata
        if prepared_binding_from_case_metadata(
                description, deck_path=deck) is not None:
            preparation_integrator='implicit_fwave_midpoint'
    from .amr_general_session import validate_selector_policy,selector_source_hashes
    validate_selector_policy(selector_policy)
    require(type(recursive_bank_bedform_avalanche) is bool,
            'Recursive bedform-bank preparation selection must be boolean')
    require(selector_policy is None or general,
            'General AMR selector policy requires nonbank transport AMR')
    policy_sources=selector_source_hashes(selector_policy)
    require(not general or not evolving_banks,
            'General transport AMR cannot request evolving banks')
    require(not recursive_bank_bedform_avalanche or
            (evolving_banks and not general and selector_policy is None),
            'Recursive bedform-bank preparation requires evolving bank AMR')
    bedform_bank_mode=(described_bank_mode(bank_description)
                       if recursive_bank_bedform_avalanche else None)
    session_deck=deck
    geometry_adapter=None
    if recursive_bank_bedform_avalanche:
        session_deck,geometry_adapter=recursive_geometry_deck_adapter(
            deck,expected_base_version=38 if bedform_bank_mode==2 else 40)
    runtime=RUNTIME
    if general:
        from .amr_general_native_geometry import DEFAULT_RUNTIME
        runtime=DEFAULT_RUNTIME
    elif evolving_banks:
        if recursive_bank_bedform_avalanche:
            if bedform_bank_mode==2:
                from .amr_bank_physical_toe_bedform_native_geometry import DEFAULT_RUNTIME
            else:
                from .amr_bank_bedform_native_geometry import DEFAULT_RUNTIME
        else:
            from .amr_bank_native_geometry import DEFAULT_RUNTIME
        runtime=DEFAULT_RUNTIME
    if (ROOT/'PACKAGE-MANIFEST.json').exists():
        from .amr_portable_runtime import verify_packaged_runtime
        bundled=verify_packaged_runtime()
        # The exact recursive bedform/bank initializers already include and
        # authenticate their own live-boundary geometry contract.  A compact
        # QGIS product worker may therefore omit the separate general-purpose
        # live runtime.  Non-composed evolving-bank preparation still needs it.
        require(not evolving_banks or recursive_bank_bedform_avalanche or
                'live_runtime' in bundled,
                'This worker package lacks the current evolving/live-boundary initializer; rebuild it')
        runtime_key=('bedform_physical_toe_runtime' if bedform_bank_mode==2 else
                     'bedform_bank_runtime')
        require(not recursive_bank_bedform_avalanche or runtime_key in bundled,
                'This worker package lacks the exact recursive bedform-bank initializer; rebuild it')
        runtime=bundled['general_runtime' if general else
                        runtime_key if recursive_bank_bedform_avalanche else
                        'live_runtime' if evolving_banks else 'runtime']
    record=verify_runtime(runtime)
    if general:
        require(record.get('kind') == 'selected-general-transport-amr-v1',
                'General transport AMR needs its versioned zero-step initializer')
    elif evolving_banks:
        require(record.get('live_boundary_metadata') is True,
                'Evolving preparation lacks the live-section initializer metadata')
        require((record.get('recursive_bank_bedform_avalanche') is True) is
                recursive_bank_bedform_avalanche,
                'Evolving initializer bedform-bank scope differs')
        if recursive_bank_bedform_avalanche:
            require(record.get('bank_modes')==[bedform_bank_mode] and
                    (record.get('toe_geometry_seed_only') is True)==
                        (bedform_bank_mode==2),
                    'Evolving initializer exact bank mode differs')
    else:
        require(record['executable_sha256']==EXPORTER_SHA,'Pinned on-demand initializer differs')
    variant=(GENERAL_IMPLICIT_VARIANT if general and
             preparation_integrator=='implicit_fwave_midpoint' else
             GENERAL_VARIANT if general else VARIANT)
    request=dict(schema=1,variant=variant,deck=str(deck),deck_sha256=sha(deck),
                 hierarchy_sha256=sha(deck.parent/'amr_hierarchy.npz'),runtime_build_sha256=sha(runtime/'build.json'))
    require(type(evolving_banks) is bool,'Evolving bank preparation must be explicit boolean')
    if general:
        request.update(schema=3,general_transport=True)
        if preparation_integrator=='implicit_fwave_midpoint':
            request['integrator']=preparation_integrator
    elif evolving_banks:
        request.update(schema=2,evolving_banks=True)
        if recursive_bank_bedform_avalanche:
            request['recursive_bank_bedform_avalanche']=True
            request['bank_mode']=bedform_bank_mode
            request['recursive_geometry_adapter']=geometry_adapter
    if selector_policy is not None:
        request.update(selector_policy=selector_policy,
                       selector_source_sha256=policy_sources)
    def stopped():
        return stop_file is not None and Path(stop_file).exists()
    if stopped():
        raise InterruptedError('GPU AMR preparation cancelled')
    if output.exists():
        require((output/'preparation_request.json').is_file() and
                json.loads((output/'preparation_request.json').read_text())==request and
                not (output/'preparation_receipt.json').exists(),
                'Preserve existing GPU AMR preparation; only the same incomplete request can resume')
    else:
        output.mkdir(parents=True)
        write_json_atomic(output/'preparation_request.json',request)
    def report(fraction,message,status='preparing',*,committed=False):
        row=dict(status=status,variant=variant,percent=100*float(fraction),message=message)
        actions=[lambda:write_json_atomic(output/'preparation_status.json',row),
                 lambda:print(json.dumps(row),flush=True)]
        if progress is not None:
            actions.append(lambda:progress(float(fraction),message))
        for action in actions:
            try:
                action()
            except Exception:
                # Once the authenticated receipt exists, a UI callback or
                # disconnected observer cannot turn the preparation into failure.
                if not committed:
                    raise
    try:
        report(0.,'Reading original AMR input and checking native initializer')
        if general:
            from .amr_general_session import GeneralGeometrySession
            session_type=GeneralGeometrySession
        else:
            session_type=GeometrySession
        if (output/'session').exists():
            session=session_type(output/'session',cancelled=stopped)
        else:
            session=session_type.create(output/'session',session_deck,runtime=runtime,
                cancelled=stopped,progress=lambda fraction,message:report(.05+.9*fraction,message),
                **(dict(evolving_banks=True) if evolving_banks else {}),
                **(dict(selector_policy=selector_policy) if selector_policy is not None else {}))
        require(session.evolving_banks==evolving_banks,'Prepared session bank mapping contract differs')
        require(bool(getattr(session,'bedform_bank_composition',False)) is
                recursive_bank_bedform_avalanche,
                'Prepared session bedform-bank initializer contract differs')
        require(not recursive_bank_bedform_avalanche or
                session.bank_mode==bedform_bank_mode,
                'Prepared session exact bedform-bank mode differs')
        require(bool(getattr(session,'general_transport',False))==general,
                'Prepared session transport contract differs')
        require(getattr(session,'selector_policy',None)==selector_policy,
                'Prepared session selector policy differs')
        if stopped():
            raise InterruptedError('GPU AMR preparation cancelled')
        data=(scope(session) if preparation_integrator=='ssp33' else
              scope(session,preparation_integrator))
        session.get(session.manifest['initial_node_ids'])
        if stopped():
            raise InterruptedError('GPU AMR preparation cancelled before publication')
        receipt=dict(schema=6,variant=variant,session_identity=session.identity,
            session_manifest_sha256=sha(output/'session/session.json'),
            original_input_files=session.manifest['original_input_files'],
            native_solver_steps_performed=0,initialization_CPU_only=True,GPU_solver_executed=False,
            runtime_build_sha256=sha(runtime/'build.json'),portable=False,
            source_worker_packaged=(ROOT/'PACKAGE-MANIFEST.json').exists())
        if preparation_integrator=='implicit_fwave_midpoint':
            receipt['integrator']=preparation_integrator
        if geometry_adapter is not None:
            receipt['recursive_geometry_adapter']=geometry_adapter
        if selector_policy is not None:
            receipt.update(selector_policy=selector_policy,
                           selector_source_sha256=policy_sources)
        write_json_atomic(output/'preparation_receipt.json',receipt)
        report(1.,'GPU AMR geometry and initial mesh ready','prepared',committed=True)
        return dict(status='prepared',output=str(output),variant=variant,
            cells=len(session.manifest['initial_node_ids']),fractions=int(data['nfrac']),
            native_solver_steps_performed=0,preparation_receipt_sha256=sha(output/'preparation_receipt.json'))
    except Exception as error:
        try:
            report(0.,str(error),'cancelled' if isinstance(error,InterruptedError) else 'failed')
        except Exception:
            pass  # Preserve the actual preparation error, not an observer error.
        raise


def read_prepared(folder,*,cancelled=lambda:False,integrator=None):
    folder=Path(folder).resolve()
    receipt=json.loads((folder/'preparation_receipt.json').read_text())
    request=json.loads((folder/'preparation_request.json').read_text())
    variant=receipt.get('variant')
    require(receipt.get('schema')==6 and variant in
        (VARIANT,GENERAL_VARIANT,GENERAL_IMPLICIT_VARIANT) and
        receipt.get('initialization_CPU_only') is True and receipt.get('GPU_solver_executed') is False and
        receipt.get('native_solver_steps_performed')==0 and
        receipt.get('session_manifest_sha256')==sha(folder/'session/session.json'),
        'GPU on-demand preparation receipt differs')
    if variant in (GENERAL_VARIANT,GENERAL_IMPLICIT_VARIANT):
        from .amr_general_session import GeneralGeometrySession
        session=GeneralGeometrySession(folder/'session',cancelled=cancelled)
    else:
        session=GeometrySession(folder/'session',cancelled=cancelled)
    require(receipt['session_identity']==session.identity and
        receipt['original_input_files']==session.manifest['original_input_files'] and
        receipt['runtime_build_sha256']==session.bootstrap.source['runtime_build_sha256'],
        'GPU on-demand preparation dependencies differ')
    recursive_adapter=request.get('recursive_geometry_adapter')
    if request.get('recursive_bank_bedform_avalanche') is True:
        require(isinstance(recursive_adapter,dict) and
                receipt.get('recursive_geometry_adapter')==recursive_adapter and
                recursive_adapter.get('identity')==digest({key:value for key,value in
                    recursive_adapter.items() if key!='identity'}) and
                recursive_adapter.get('physical_payload_unchanged') is True and
                Path(session.manifest['deck']).name==
                    recursive_adapter.get('adapted_deck_name') and
                sha(Path(session.manifest['deck']))==
                    recursive_adapter.get('adapted_deck_sha256'),
                'Recursive geometry identity adapter differs')
    else:
        require(recursive_adapter is None and
                'recursive_geometry_adapter' not in receipt,
                'Unexpected recursive geometry identity adapter')
    expected_integrator=('implicit_fwave_midpoint'
        if variant==GENERAL_IMPLICIT_VARIANT else 'ssp33')
    if variant==GENERAL_IMPLICIT_VARIANT:
        require(receipt.get('integrator')==expected_integrator and
                request.get('variant')==variant and
                request.get('integrator')==expected_integrator,
                'Implicit M2-C preparation integrator identity differs')
    else:
        require('integrator' not in receipt and 'integrator' not in request,
                'Historical on-demand preparation unexpectedly selects an integrator')
    selected=expected_integrator if integrator is None else integrator
    require((selected==expected_integrator) or variant==VARIANT,
            'Requested integrator differs from prepared general AMR route')
    policy=getattr(session,'selector_policy',None)
    if policy is None:
        require('selector_policy' not in receipt and
                'selector_source_sha256' not in receipt,
                'Legacy GPU AMR receipt has selector policy fields')
    else:
        require(variant in (GENERAL_VARIANT,GENERAL_IMPLICIT_VARIANT) and
                receipt.get('selector_policy')==policy and
                receipt.get('selector_source_sha256')==
                    session.manifest['selector_source_sha256'] and
                request.get('selector_policy')==policy and
                request.get('selector_source_sha256')==
                    session.manifest['selector_source_sha256'],
                'GPU AMR selector policy preparation differs')
    return session,scope(session,selected),sha(folder/'preparation_receipt.json')


def source_snapshot(folder,*,evolving_banks=False,live_data=None,general_data=None,
                    integrator='ssp33'):
    from .amr_portable_runtime import (current_interface_kernel_sources,
        verify_interface_source_pair,REVISED_KERNEL_ABIS)
    if (ROOT/'PACKAGE-MANIFEST.json').exists():
        from .amr_portable_runtime import verify_packaged_runtime
        packaged=verify_packaged_runtime()
        source=packaged['general_kernels' if general_data is not None else 'kernels']
        kernels={path.name:path for path in source.glob('*.cu')}
        pair=packaged['general_interface_source_pair' if general_data is not None else
                      'interface_source_pair']
    else:
        from validation.gpu_fwave.amr_general_gpu_step import SOURCE
        source=SOURCE
        kernels,pair=current_interface_kernel_sources(ROOT,source)
    evolving_pair=None
    if evolving_banks:
        from .amr_evolving_runtime import current_sources,source_pair,KERNEL_ABIS
        if (ROOT/'PACKAGE-MANIFEST.json').exists():
            extra={name:source/name for name in KERNEL_ABIS}
            evolving_pair=source_pair(ROOT,extra)
        else:
            extra,evolving_pair=current_sources(ROOT)
        kernels.update(extra)
    if general_data is not None:
        indicator=(source/'amr_general_indicators.cu' if
                   (ROOT/'PACKAGE-MANIFEST.json').exists() else
                   ROOT/'validation/gpu_fwave/amr_general_indicators.cu')
        require(indicator.is_file(),
                'General transport AMR indicator kernel is missing')
        kernels[indicator.name]=indicator
    folder=Path(folder)
    require(not folder.exists(),'Preserve existing AMR numerical source snapshot')
    files={}
    originals={}
    groups=((ROOT/'braids_gpu','*.py','python/braids_gpu'),
            (ROOT/'braids_gpu','*.cu','python/braids_gpu'),
            (ROOT/'validation/gpu_fwave','*.py','python/validation/gpu_fwave'),
            (ROOT/'braids_core','gpu_live_boundary_metadata.py','python/braids_core'),
            (ROOT/'braids_core','windows_paths.py','python/braids_core'),
            (ROOT/'braids_core','gpu_capabilities.py','python/braids_core'))
    for origin,pattern,prefix in groups:
        for path in sorted(origin.glob(pattern)):
            name=prefix+'/'+path.name
            destination=folder/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            files[name]=sha(destination);originals[path]=files[name]
    for name,path in sorted(kernels.items()):
        destination=folder/'kernels'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        files['kernels/'+name]=sha(destination);originals[path]=files['kernels/'+name]
    verify_interface_source_pair(folder/'python',
        {name:folder/'kernels'/name for name in REVISED_KERNEL_ABIS},pair)
    if evolving_banks:
        from .amr_evolving_runtime import verify
        verify(folder/'python',{name:folder/'kernels'/name for name in KERNEL_ABIS},evolving_pair)
    if live_data is not None:
        require(evolving_banks and np.array_equal(live_data['mode'],[9,4]),
                'Live AMR source archive needs explicit evolving surveyed/critical controls')
        from .general_live_bank import prepare_sources
        live_folder=folder/'kernels/live_bank'
        live_receipt=prepare_sources(live_folder,live_data)
        for path in sorted(live_folder.rglob('*')):
            if path.is_file():
                files[path.relative_to(folder).as_posix()]=sha(path)
        for relative,identity in live_receipt['sources'].items():
            originals[ROOT/relative]=identity
    if general_data is not None:
        from .general_live_transport import recursive_bedform_envelope
        from braids_core.sediment.prognostic_secondary_nonzero_execution import (
            EXECUTION_KEY as NONZERO_EXECUTION_KEY,
        )
        live_v59 = (integrator == 'implicit_fwave_midpoint' and
                    NONZERO_EXECUTION_KEY in general_data)
        if live_v59:
            from .implicit_midpoint_fwave_amr_v59 import prepare_sources
        else:
            from .general_live_transport import prepare_sources
        modes = np.asarray(general_data['mode'])
        closed_v59 = (
            live_v59 and int(general_data['nb']) == 0 and
            modes.shape == (0,) and modes.dtype == np.int32 and
            np.all(np.asarray(general_data['bid']) == -1) and
            not np.any(np.asarray(
                general_data.get('secondary_open_face', 0))))
        admitted_envelope = (
            closed_v59 if live_v59 else
            (np.array_equal(modes, [9, 4]) or
             recursive_bedform_envelope(general_data)))
        require(live_data is None and not evolving_banks and
                int(general_data['bank_toe_erosion_mode']) == 0 and
                admitted_envelope,
                'General AMR transport source requires the closed Q3-H3/v59 '
                'envelope, nonbank surveyed/critical input, or the exact '
                'recursive bedform pair')
        general_folder=folder/'kernels/live_transport'
        general_receipt=prepare_sources(general_folder,general_data,
                                        integrator=integrator)
        for path in sorted(general_folder.rglob('*')):
            if path.is_file():
                files[path.relative_to(folder).as_posix()]=sha(path)
        for relative,identity in general_receipt['sources'].items():
            originals[ROOT/relative]=identity
    require(all(sha(path)==identity for path,identity in originals.items()),
        'AMR source changed during numerical snapshot')
    manifest=dict(schema=2,variant=(GENERAL_IMPLICIT_VARIANT
        if general_data is not None and integrator=='implicit_fwave_midpoint'
        else GENERAL_VARIANT if general_data is not None else VARIANT),
        source_files=files,numerical_cuda_unchanged=False,
        intentional_cuda_revisions=sorted(REVISED_KERNEL_ABIS),interface_source_pair=pair)
    if evolving_banks:
        manifest.update(evolving_interface_source_pair=evolving_pair,
                        added_cuda_kernels=sorted(KERNEL_ABIS))
    if live_data is not None:
        manifest['live_boundary_source_sha256']=sha(live_folder/'general_live_bank_sources.json')
    if general_data is not None:
        manifest['general_transport_source_sha256']=sha(
            general_folder/'general_live_transport_sources.json')
        v59_manifest = general_folder/'q3h3_amr_live_v59_sources.json'
        if v59_manifest.is_file():
            manifest['q3h3_amr_live_v59_source_sha256'] = sha(v59_manifest)
    write_json_atomic(folder/'runtime_manifest.json',manifest)
    return files,sha(folder/'runtime_manifest.json'),originals


def next_clock(current,interval):
    return (math.floor(current/interval+1e-10)+1)*interval


def boundary_events(data):
    """Immutable forcing clock; retries/regrids never consume a table cursor."""
    from validation.gpu_fwave.boundary_events import BoundaryEvents
    return BoundaryEvents.from_tables(data)


def trial_clock(current,hint,*,end,next_output,regrid_interval,events=None):
    """Do not step over a forcing, output or mesh-selection event."""
    regrid_time=next_clock(current,regrid_interval)
    target=min(end,next_output,regrid_time)
    if events is not None:
        target=min(target,events.next_after(current))
    dt=hint if current+hint<=target+1e-13 else target-current
    require(np.isfinite(dt) and dt>0 and current+dt>current,
            'Nonadvancing on-demand AMR trial clock')
    return dt,regrid_time


def _publish_directory(source,target):
    for attempt in range(12):
        if target.exists():
            raise FileExistsError('GPU AMR output publication is write-once')
        try:
            os.rename(source,target)
            return
        except OSError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==11:
                raise
            time.sleep(min(.05*(attempt+1),.5))


def validate_worker_checkpoint(resume,*,method_identity,input_identity,source_identity,dtmax,start,end):
    """Cross-check worker metadata and payload clock before any CUDA allocation."""
    resume=Path(resume)
    worker=json.loads((resume/'worker.json').read_text())
    require(set(worker)=={'schema','method_identity','input_identity','source_identity','time_s',
        'steps','dt_hint','rejected','payload_sha256','operators_sha256'} and worker['schema']==1 and
        worker['method_identity']==method_identity and worker['input_identity']==input_identity and
        worker['source_identity']==source_identity and worker['payload_sha256']==sha(resume/'state.npz') and
        worker['operators_sha256']==sha(resume/'operators.json') and
        type(worker['steps']) is int and worker['steps']>=0 and type(worker['rejected']) is int and
        worker['rejected']>=0 and type(worker['dt_hint']) in (int,float) and
        np.isfinite(worker['dt_hint']) and 0<worker['dt_hint']<=dtmax and
        type(worker['time_s']) in (int,float) and np.isfinite(worker['time_s']) and
        start<=worker['time_s']<=end+1e-12,
        'GPU AMR worker checkpoint controls differ')
    with np.load(resume/'state.npz',allow_pickle=False) as fields:
        require(fields['time_s'].shape==() and fields['time_s'].dtype==np.float64 and
            fields['accepted_steps'].shape==() and fields['accepted_steps'].dtype.kind in 'iu' and
            float(fields['time_s'])==worker['time_s'] and int(fields['accepted_steps'])==worker['steps'] and
            str(fields['method_identity'])==method_identity and str(fields['catalog_identity'])==input_identity,
            'GPU AMR checkpoint accepted clock or identity differs')
    return worker


def checkpoint_descriptor(target,*,time_s,steps,method_identity,input_identity,source_identity,hint,rejected):
    """Recover only the authenticated current commit, never a stale manifest pointer."""
    target=Path(target)
    worker=validate_worker_checkpoint(target,method_identity=method_identity,input_identity=input_identity,
        source_identity=source_identity,dtmax=hint,start=0.,end=time_s)
    require(worker['time_s']==time_s and worker['steps']==steps and worker['dt_hint']==hint and
        worker['rejected']==rejected,'Existing AMR checkpoint differs from current accepted state')
    receipt=json.loads((target/'checkpoint.json').read_text())
    if receipt.get('schema') in (5,7):
        require(receipt.get('session_identity')==input_identity and
                receipt.get('catalog_identity')==input_identity and
                receipt.get('method_identity')==method_identity and
                receipt.get('payload_file_sha256')==worker['payload_sha256'] and
                (receipt.get('schema')==5 or
                 (receipt.get('kind')==
                    'implicit_fwave_midpoint_suspension_amr' and
                  receipt.get('integrator')=='implicit_fwave_midpoint')),
                'Committed general AMR checkpoint receipt differs')
        with np.load(target/'state.npz',allow_pickle=False) as archive:
            require(archive['schema'].shape==() and archive['schema'].dtype==np.int64 and
                    int(archive['schema'])==receipt['schema'],
                    'General AMR checkpoint schema receipt/payload differs')
        return dict(path='checkpoints/'+target.name,worker_sha256=sha(target/'worker.json'),
                    payload_sha256=worker['payload_sha256'],operators_sha256=worker['operators_sha256'])
    if receipt.get('kind')=='recursive_bank_amr':
        require(receipt.get('schema') in (1,2) and receipt.get('payload_sha256')==worker['payload_sha256'] and
            receipt.get('catalog_identity')==input_identity and receipt.get('method_identity')==method_identity,
            'Committed recursive checkpoint receipt differs')
        return dict(path='checkpoints/'+target.name,worker_sha256=sha(target/'worker.json'),
                    payload_sha256=worker['payload_sha256'],operators_sha256=worker['operators_sha256'])
    if receipt.get('kind')=='recursive_general_amr':
        require(set(receipt)=={'schema','kind','session_identity','catalog_identity',
            'topology_identity','method_identity','payload_file','payload_file_sha256',
            'payload_sha256','accepted_steps','time_s','event_count'} and
            receipt.get('schema')==6 and receipt.get('session_identity')==input_identity and
            receipt.get('payload_file')=='state.npz' and
            receipt.get('payload_file_sha256')==worker['payload_sha256'] and
            receipt.get('catalog_identity')==input_identity and
            receipt.get('method_identity')==method_identity and
            receipt.get('accepted_steps')==steps and receipt.get('time_s')==time_s and
            type(receipt.get('event_count')) is int and receipt['event_count']>=0 and
            isinstance(receipt.get('payload_sha256'),str) and len(receipt['payload_sha256'])==64,
            'Committed recursive general checkpoint receipt differs')
        return dict(path='checkpoints/'+target.name,worker_sha256=sha(target/'worker.json'),
                    payload_sha256=worker['payload_sha256'],operators_sha256=worker['operators_sha256'])
    payload_schema=receipt.get('schema')
    require(type(payload_schema) is int and payload_schema in (1,2,3,4) and
        json.loads((target/'checkpoint.json').read_text())==dict(schema=payload_schema,kind='coupled_bank_amr',
        payload_sha256=worker['payload_sha256'],catalog_identity=input_identity,method_identity=method_identity),
        'Committed AMR checkpoint receipt differs')
    with np.load(target/'state.npz',allow_pickle=False) as archive:
        require(archive['schema'].shape==() and archive['schema'].dtype==np.int64 and
                int(archive['schema'])==payload_schema,'Checkpoint schema receipt/payload differs')
    return dict(path='checkpoints/'+target.name,worker_sha256=sha(target/'worker.json'),
                payload_sha256=worker['payload_sha256'],operators_sha256=worker['operators_sha256'])


def run(*,input_path,output_path,until=None,output_every=None,dt_max=None,max_steps=None,
        stop_file=None,resume=None,launch_mode='plain',integrator='ssp33',capture_stages=False,
        capture_step_trace=False,source_archive=None,cancelled=lambda:False):
    from .runner import time_controls
    from validation.gpu_fwave.adaptive_hydraulics import next_hint,retry_step
    output,input_path=Path(output_path).resolve(),Path(input_path).resolve()
    require(not output.exists(),'Preserve existing GPU AMR output')
    require(launch_mode=='plain' and
            integrator in ('ssp33','implicit_fwave_midpoint') and
            not capture_stages and source_archive is None,
            'On-demand AMR requires an explicit admitted plain integrator '
            'with current authenticated sources')
    require(max_steps is None or (type(max_steps) is int and max_steps>0),'Invalid AMR max_steps')
    def stopped():
        return bool(cancelled() or (stop_file is not None and Path(stop_file).exists()))
    if stopped():
        return dict(status='cancelled',output=str(output),time_s=0.,steps=0)
    session,data,prepared_sha=read_prepared(input_path,cancelled=stopped,
                                            integrator=integrator)
    general=bool(getattr(session,'general_transport',False))
    recursive=bool(session.selector_arguments['settings']['subcycling'])
    require(integrator=='ssp33' or (general and not recursive),
            'Implicit f-wave on-demand execution requires synchronous '
            'nonrecursive general AMR')
    bank_bedform_composition=bool(recursive and not general and
        getattr(session,'bedform_bank_composition',False) and
        int(data.get('bedform_roughness',0))==1)
    physical_toe_bedform_composition=bool(
        bank_bedform_composition and session.bank_mode==2)
    if general:
        from braids_core.sediment.prognostic_secondary_nonzero_execution import (
            EXECUTION_KEY as NONZERO_EXECUTION_KEY,
        )
        live_v59_route = (
            integrator == 'implicit_fwave_midpoint' and
            NONZERO_EXECUTION_KEY in data)
    else:
        live_v59_route = False
    if general:
        if recursive:
            from .amr_general_recursive_route import RecursiveGeneralOnDemandRoute
            route_type=RecursiveGeneralOnDemandRoute
        elif live_v59_route:
            from .amr_general_route_v59 import LiveV59GeneralOnDemandRoute
            route_type=LiveV59GeneralOnDemandRoute
        else:
            from .amr_general_route import GeneralOnDemandRoute
            route_type=GeneralOnDemandRoute
    elif recursive:
        from .amr_recursive_route import RecursiveOnDemandBankRoute
        route_type=RecursiveOnDemandBankRoute
    else:
        route_type=OnDemandBankRoute
    if stopped():
        return dict(status='cancelled',output=str(output),time_s=0.,steps=0)
    start,end,interval,dtmax=time_controls(data,until,output_every,dt_max)
    require(end<=float(data['endtime']) and dtmax<=float(data['dtmax']),
            'AMR time overrides exceed prepared limits')
    regrid_interval=float(session.tree['amr_interval'])
    live=np.array_equal(data['mode'],[9,4])
    events=boundary_events(data) if live or recursive else None
    output.mkdir(parents=True)
    sources,source_identity,original_sources=source_snapshot(output/'source',
        **(dict(evolving_banks=True) if session.evolving_banks else {}),
        **(dict(live_data=data) if live and not general else {}),
        **(dict(general_data=data,integrator=integrator) if general else {}))
    if general:
        from .general_live_transport import integration_identity
        selected_integration=integration_identity(integrator,'plain')
    else:
        selected_integration=dict(integrator='SSPRK33',euler_stages=3,
                                  stage_fractions=[0.,1.,.5])
    method=dict(variant=(GENERAL_IMPLICIT_VARIANT if general and
                        integrator=='implicit_fwave_midpoint' else
                        GENERAL_VARIANT if general else VARIANT),
        source_identity=source_identity,session_identity=session.identity,
        integrator=selected_integration['integrator'],
        integrator_token=integrator, launch_mode='plain',
        precision='FP64',dtmax_s=dtmax,
        output_every_s=interval,regrid_interval_s=regrid_interval,cfl=float(data['cfl_target']),
        subcycling=recursive,selector_contract=session.manifest['selector_contract'],
        checkpoint_schema=(RECURSIVE_GENERAL_CHECKPOINT if general and recursive else
                           RECURSIVE_BANK_CHECKPOINT if recursive else
                           7 if general and
                               integrator=='implicit_fwave_midpoint' else
                           session.checkpoint_schema))
    if recursive:
        method['level_stepping']=RECURSIVE_LEVEL_STEPPING
        if not general:
            method['recursive_bank_contract']='shared_recursive_bank_modes_v2'
            if bank_bedform_composition:
                from .amr_bank_subcycle import (BEDFORM_BANK_METHOD_CONTRACT,
                    BEDFORM_PHYSICAL_TOE_METHOD_CONTRACT)
                method['recursive_bank_bedform_avalanche']=dict(
                    BEDFORM_PHYSICAL_TOE_METHOD_CONTRACT
                    if physical_toe_bedform_composition else
                    BEDFORM_BANK_METHOD_CONTRACT)
        if bool(data.get('bedload_adaptation',False)):
            method['bedload_adaptation']=dict(policy='full_leaf_steady_v1',
                length_diameters=float(data['bedload_adaptation_length_diameters']))
        if general and bool(data.get('general_avalanche',False)):
            method['recursive_general_avalanche']=dict(
                policy=RECURSIVE_GENERAL_AVALANCHE_POLICY,
                placement='external_parent_post_reflux',calls_per_macro=1,
                conservative_per_fraction=True)
        if general and float(data.get('horizontal_viscosity',0.)) > 0.:
            from .general_live_transport import (PARENT_VISCOSITY_ORDER,
                VISCOSITY_INTEGRATOR,VISCOSITY_LIMITED_NORM)
            require(PARENT_VISCOSITY_ORDER == RECURSIVE_GENERAL_VISCOSITY_POLICY and
                    VISCOSITY_INTEGRATOR ==
                    RECURSIVE_GENERAL_VISCOSITY_METHOD['integrator'] and
                    VISCOSITY_LIMITED_NORM ==
                    RECURSIVE_GENERAL_VISCOSITY_METHOD['limited_norm'],
                    'Recursive viscosity runner/core method contract differs')
            method['recursive_general_viscosity']=dict(
                RECURSIVE_GENERAL_VISCOSITY_METHOD)
        if general and bool(data.get('bedform_roughness',False)):
            from .general_live_transport import RECURSIVE_BEDFORM_METHOD
            method['recursive_general_bedform']=dict(RECURSIVE_BEDFORM_METHOD)
    if session.evolving_banks:
        method['evolving_banks']=True
    if session.bank_mode==3 or (recursive and not general):
        method['bank_mode']=session.bank_mode
    if general:
        method['general_transport']=dict(bedlaw=int(data['bedlaw']),
            exchange_mode=int(data['exchange_mode']),
            secondary_flow_model=int(data['secondary_flow_model']),
            bedload_adaptation=bool(data['bedload_adaptation']),
            general_avalanche=bool(data.get('general_avalanche', 0)),
            horizontal_viscosity=float(data['horizontal_viscosity']),
            state_transfer=('conservative_all_rows_closed_recursive_epoch_v1'
                            if recursive else
                            'conservative_all_rows_synchronous_v1'))
        if integrator=='implicit_fwave_midpoint':
            from .implicit_suspension_amr import method_identity
            method['general_transport'].update(method_identity(data))
        from .amr_typed_roughness import (
            BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD,
            amr_forcing_identity)
        if AMR_ROUGHNESS_BINDING_FIELD in data:
            # One hierarchy identity survives every topology-local schema-2
            # rebind and is therefore the forcing identity checked on restart.
            method['general_transport'].update(amr_forcing_identity(data))
        from .general_live_transport import recursive_bedform_envelope
        if bool(data.get('bedform_roughness', 0)) or (
                recursive and recursive_bedform_envelope(data)):
            # Preserve all historical ordinary-OFF method identities.  The
            # explicit field identifies either an actually enabled closure or
            # the new paired mode1/mode2 OFF control admitted by v15.
            method['general_transport']['bedform_roughness'] = bool(
                data.get('bedform_roughness', 0))
        if session.selector_policy is not None:
            policy_sources=session.manifest['selector_source_sha256']
            require(all(sources['python/braids_gpu/'+name]==value for name,value in
                        policy_sources.items()),
                    'GPU AMR selector source snapshot differs from preparation')
            method['selector_policy']=session.selector_policy
            method['selector_source_sha256']=policy_sources
    if session.selector_arguments['settings']['refine_sediment']:
        method['selector_reference']='accepted_regrid_leaf_bed_v1'
    if live or recursive:
        method['boundary_forcing']=dict(schema=1,event_identity=events.identity,
            stage_fractions=list(selected_integration['stage_fractions']),
            clock='forcing_knot_clipped')
    method_identity=digest(method)
    manifest=dict(schema=3,format='braids-gpu-amr-run-v2',
        input_kind='amr-on-demand-general' if general else 'amr-on-demand',
        backend='gpu',backend_selected_by='user',automatic_backend_selection=False,
        input_folder=str(input_path),input_preparation_sha256=prepared_sha,input_identity=session.identity,
        source_identity=source_identity,source_sha256=sources,method=method,method_identity=method_identity,
        frames=[],operators=[],topologies=[],status='initializing',start_s=start,end_s=end,
        output_every_s=interval,dtmax_s=dtmax,preparation_receipt_sha256=prepared_sha,
        QGIS_release=False,physical_validation=False,portable=False,Delft3D_source_modified=False)
    shutil.copy2(input_path/'preparation_receipt.json',output/'preparation_receipt.json')
    state='initializing';route=None;admitted=False;delay=None;hint=dtmax;rejected=0;accepted=[];began=time.perf_counter()
    def progress(event):
        current=0. if route is None else route.model.time_s
        steps=0 if route is None else route.model.accepted_steps
        row=dict(event=event,status=state,backend='gpu',time_s=current,end_s=end,steps=steps,
            rejected=rejected,percent=100*min(1.,max(0.,(current-start)/(end-start))),
            wall_s=time.perf_counter()-began)
        write_json_atomic(output/'status.json',row)
        print(json.dumps(row),flush=True)
    def publish_operators():
        if general:
            from .amr_general_prepared import load_general_amr_prepared as load_operator
        else:
            load_operator=load_amr_prepared
        for entry in route.entries[len(manifest['operators']):]:
            relative='operators/'+entry.package_identity
            target=output/relative
            target.parent.mkdir(exist_ok=True)
            pending=Path(tempfile.mkdtemp(prefix='.mesh-',dir=target.parent))
            committed=False
            try:
                shutil.copytree(entry.package.folder,pending/'package')
                require(load_operator(pending/'package').identity==entry.package_identity,
                        'Copied AMR result geometry differs')
                _publish_directory(pending/'package',target)
                committed=True
            finally:
                if pending.exists():
                    try:
                        shutil.rmtree(pending)
                    except OSError:
                        if not committed:
                            raise
            top=entry.package.topologies[0]
            manifest['operators'].append(dict(name=top.name,path=relative,
                package_identity=entry.package_identity,operator_identity=top.source_identity))
            manifest['topologies'].append(top.name)
    def checkpoint():
        # The final run manifest points only to a fully published four-file
        # checkpoint. Previous checkpoints are immutable and remain recoverable.
        current=route.model.time_s;steps=route.model.accepted_steps
        root=output/'checkpoints';root.mkdir(exist_ok=True)
        target=root/f'accepted_{steps:09d}'
        if target.exists():
            previous=validate_worker_checkpoint(target,method_identity=method_identity,
                input_identity=session.identity,source_identity=source_identity,dtmax=dtmax,start=0.,end=current)
            require(previous['time_s']==current and previous['steps']==steps,
                    'Existing checkpoint directory has a different accepted clock')
            if previous['dt_hint']!=hint or previous['rejected']!=rejected:
                # A rejected trial followed by Stop changes recovery counters,
                # not the accepted solution. Preserve the old commit and publish
                # a new immutable revision of its worker metadata.
                target=root/f'accepted_{steps:09d}_trial_{rejected:09d}_{digest(hint)[:16]}'
        def descriptor():
            return checkpoint_descriptor(target,time_s=current,steps=steps,method_identity=method_identity,
                input_identity=session.identity,source_identity=source_identity,hint=hint,rejected=rejected)
        if target.exists():
            return descriptor()
        pending=Path(tempfile.mkdtemp(prefix='.accepted-',dir=root))
        committed=False
        try:
            route.write_checkpoint(pending/'complete',delay,method_identity=method_identity)
            worker=dict(schema=1,method_identity=method_identity,input_identity=session.identity,
                source_identity=source_identity,time_s=current,steps=steps,dt_hint=hint,rejected=rejected,
                payload_sha256=sha(pending/'complete/state.npz'),
                operators_sha256=sha(pending/'complete/operators.json'))
            write_json_atomic(pending/'complete/worker.json',worker)
            _publish_directory(pending/'complete',target)
            committed=True
        finally:
            if pending.exists():
                try:
                    shutil.rmtree(pending)
                except OSError:
                    if not committed:
                        raise
        return descriptor()
    def publish(frame=True):
        publish_operators()
        manifest['checkpoint']=checkpoint()
        if frame:
            raw=route.frame();top=route.model.topology
            name=f"frame_{len(manifest['frames']):06d}.npz"
            fields=dict(frame_schema=np.int64(8 if physical_toe_bedform_composition else
                                             7 if bank_bedform_composition else
                                             6 if general and recursive else
                                            5 if general else
                                            4 if recursive and session.bank_mode==2 else
                                            3 if session.bank_mode==3 else
                                            2 if session.evolving_banks else 1),
                state=raw['state'],node_ids=raw['node_ids'],
                topology_name=np.asarray(top.name),operator_identity=np.asarray(top.source_identity),
                regrid_mesh_index=np.int64(manifest['topologies'].index(top.name)),
                time_s=np.float64(route.model.time_s),steps=np.int64(route.model.accepted_steps),
                input_identity=np.asarray(prepared_sha),source_identity=np.asarray(source_identity),
                method_identity=np.asarray(method_identity),dt_hint=np.float64(hint),rejected=np.int64(rejected))
            fields.update({key:value for key,value in raw.items() if key.startswith('ledger_')})
            if general:
                fields['topology_identity']=raw['topology_identity']
                fields['package_identity']=raw['package_identity']
                if integrator=='implicit_fwave_midpoint':
                    for key in ('selected_integrator',
                        'suspended_amr_contract_semantic_id',
                        'suspended_amr_contract_fingerprint',
                        'suspended_amr_contract_json','wetdry_phase_ledger',
                        'wetdry_last_suspended_change_kg'):
                        fields[key]=raw[key]
                if recursive:
                    fields.update({key:value for key,value in raw.items()
                                   if key.startswith('recursive_')})
                    enabled=bool(data.get('bedform_roughness', 0))
                    present={key for key in RECURSIVE_BEDFORM_FRAME_KEYS
                             if key in raw}
                    require((enabled and present==set(RECURSIVE_BEDFORM_FRAME_KEYS)) or
                            (not enabled and not present),
                            'Recursive bedform accepted-frame diagnostics differ')
                    if enabled:
                        fields.update({key:raw[key]
                                       for key in RECURSIVE_BEDFORM_FRAME_KEYS})
            else:
                for key,value in route.model.solver.bank_history.items():
                    fields['bank_history_'+key]=route.cp.asnumpy(value)
                if bank_bedform_composition:
                    present={key for key in RECURSIVE_BEDFORM_FRAME_KEYS if key in raw}
                    require(present==set(RECURSIVE_BEDFORM_FRAME_KEYS),
                            'Recursive bank bedform accepted-frame diagnostics differ')
                    fields.update({key:raw[key] for key in RECURSIVE_BEDFORM_FRAME_KEYS})
            if session.bank_mode==3:
                fields['bank_mode']=np.int64(3)
                fields['topology_identity']=raw['topology_identity']
                for key in ('distributed_stage_index','distributed_collapse_waves'):
                    fields[key]=raw[key]
                if recursive:
                    fields.update({key:value for key,value in raw.items() if key.startswith('recursive_')})
            elif session.evolving_banks:
                fields.update({key:value for key,value in raw.items() if
                    key.startswith(('toe_map_','face_epoch_')) or
                    key in ('toe_signed','toe_gross','topology_identity')})
                if recursive:
                    fields.update({key:value for key,value in raw.items()
                                   if key.startswith('recursive_')})
            elif recursive and session.bank_mode==2:
                fields.update({key:value for key,value in raw.items()
                               if key.startswith('recursive_') or
                               key in ('toe_segment_ids','toe_signed','toe_gross',
                                       'topology_identity')})
            save_arrays_atomic(output/name,**fields)
            manifest['frames'].append(dict(file=name,time_s=route.model.time_s,
                steps=route.model.accepted_steps,sha256=sha(output/name),topology=top.name,
                operator_identity=top.source_identity,cell_count=route.model.n))
        manifest['status']=state
        write_json_atomic(output/'run.json',manifest)
    try:
        write_json_atomic(output/'run.json',manifest);progress('initializing')
        if stopped():
            raise InterruptedError('GPU AMR initialization cancelled')
        if resume is not None:
            resume=Path(resume).resolve()
            worker=validate_worker_checkpoint(resume,method_identity=method_identity,
                input_identity=session.identity,source_identity=source_identity,dtmax=dtmax,start=start,end=end)
            if stopped():
                raise InterruptedError('GPU AMR restart admission cancelled')
            route,delay=route_type.restore(session,resume,source_folder=output/'source/kernels',
                **(dict(transport_source_folder=output/'source/kernels/live_transport') if general else {}),
                **(dict(integrator=integrator) if general and not recursive else {}),
                **(dict(cancelled=stopped) if general and not recursive else {}),
                method_identity=method_identity)
            require(route.model.time_s==worker['time_s'] and route.model.accepted_steps==worker['steps'],
                    'GPU AMR checkpoint accepted clock differs')
            hint,rejected=float(worker['dt_hint']),worker['rejected']
            admitted=True
        else:
            route=route_type(session,source_folder=output/'source/kernels',
                **(dict(transport_source_folder=output/'source/kernels/live_transport',
                        integrator=integrator, cancelled=stopped)
                   if general and not recursive else
                   dict(transport_source_folder=output/'source/kernels/live_transport')
                   if general else {}))
            delay=np.zeros(len(session.tree['parent']),dtype=np.int64)
            if session.evolving_banks or general:
                initial=session.initial_selection()
            else:
                initial=route.selector.propose(route.model.topology,route.model.full(),
                    cached_tree_state=route.tree_state,delay_state=delay,time_s=0.,initial=True)
            require(tuple(route.model.ids)==initial.node_ids,'AMR initial selected mesh differs')
            delay=initial.delay_state
            admitted=True
        manifest['device']=str(route.cp.cuda.runtime.getDeviceProperties(route.cp.cuda.Device().id)['name'])
        if stopped():
            raise InterruptedError('GPU AMR initialization cancelled')
        state='running';publish();progress('started')
        next_output=min(end,next_clock(route.model.time_s,interval))
        while route.model.time_s<end-1e-13:
            if stopped():
                state='cancelled';break
            if max_steps is not None and route.model.accepted_steps>=max_steps:
                state='step_limit';break
            current=route.model.time_s
            dt,regrid_time=trial_clock(current,hint,end=end,next_output=next_output,
                regrid_interval=regrid_interval,events=events)
            unclipped=hint
            for attempt in range(20):
                try:
                    record,event,proposal=route.advance(dt,delay_state=delay,
                        regrid=abs(current+dt-regrid_time)<1e-12)
                    break
                except InterruptedError:
                    state='cancelled';break
                except RuntimeError as error:
                    if stopped():
                        state='cancelled';break
                    from .amr_bank_subcycle import RecursiveStepRejected
                    if not isinstance(error,RecursiveStepRejected) and str(error) not in (
                            'Synchronous bank CUDA status/CFL rejected',
                            'Synchronous general CUDA status/CFL rejected',
                            'Synchronous general AMR CUDA status/CFL rejected',
                            'General subcycle active-level CFL rejected'):
                        raise
                    rejected+=1
                    dt=retry_step(dt,getattr(error,'cfl',float('inf')),float(data['cfl_target']))
            else:
                raise RuntimeError('GPU AMR retry limit exceeded')
            if state=='cancelled':
                break
            if proposal is not None:
                delay=proposal.delay_state
            peak=record['cfl']
            hint=next_hint(dt,peak,float(data['cfl_target']),dtmax,
                scheduled_clip=dt<unclipped-1e-13,first_attempt=attempt==0,unclipped=unclipped)
            accepted_row=dict(time_before=current,time_after=route.model.time_s,dt=dt,cfl=peak,
                              cells=route.model.n,regrid=event is not None)
            if 'avalanche_substeps' in record:
                avalanche_substeps=record['avalanche_substeps']
                require(not isinstance(avalanche_substeps,bool) and
                        isinstance(avalanche_substeps,(int,np.integer)) and
                        int(avalanche_substeps)>=0,
                        'Recursive general avalanche substep diagnostic differs')
                accepted_row['avalanche_substeps']=int(avalanche_substeps)
            if recursive and bool(data.get('general_avalanche',False)) and (
                    general or bank_bedform_composition):
                require('avalanche_substeps' in accepted_row,
                        'Recursive avalanche omitted its accepted diagnostic')
            if recursive and (general or bank_bedform_composition):
                accepted_row.update(recursive_bedform_step_diagnostics(
                    record,bool(data.get('bedform_roughness', 0))))
            if general and recursive:
                integer_diagnostics=('viscosity_calls_per_macro','viscosity_substeps',
                    'viscosity_velocity_l2_retries',
                    'viscosity_geometry_updates','viscosity_geometry_rebuilds')
                for name in integer_diagnostics:
                    value=record.get(name)
                    require(not isinstance(value,bool) and
                            isinstance(value,(int,np.integer)) and int(value)>=0,
                            'Recursive general viscosity diagnostic differs: '+name)
                    accepted_row[name]=int(value)
                correction=record.get('viscosity_minimum_correction')
                require(isinstance(correction,(int,float,np.integer,np.floating)) and
                        not isinstance(correction,(bool,np.bool_)) and
                        np.isfinite(correction) and 0. <= float(correction) <= 1.,
                        'Recursive general viscosity diagnostic differs: '
                        'viscosity_minimum_correction')
                accepted_row['viscosity_minimum_correction']=float(correction)
                active=float(data['horizontal_viscosity']) > 0.
                require((active and
                         accepted_row['viscosity_calls_per_macro'] == 1 and
                         accepted_row['viscosity_substeps'] > 0) or
                        (not active and
                         accepted_row['viscosity_calls_per_macro'] == 0 and
                         accepted_row['viscosity_substeps'] == 0 and
                         accepted_row['viscosity_velocity_l2_retries'] == 0 and
                         accepted_row['viscosity_minimum_correction'] == 1. and
                         accepted_row['viscosity_geometry_updates'] == 0 and
                         accepted_row['viscosity_geometry_rebuilds'] == 0),
                        'Recursive general viscosity ON/OFF diagnostic differs')
            if live or recursive:
                # BoundaryEvents uses +inf when no later forcing knot exists;
                # published JSON is strict and the run end is the relevant
                # finite clock bound in that case.
                accepted_row['next_boundary_knot']=min(end,events.next_after(current))
            if 'amr_regrid_veto' in record:
                accepted_row['amr_regrid_veto']=record['amr_regrid_veto']
            if proposal is not None and route.selector.extended_indicators:
                accepted_row['amr_selection']=proposal.receipt
                accepted_row['amr_selection_wall_s']=record['amr_selection_wall_s']
                accepted_row['amr_regrid_wall_s']=record['amr_regrid_wall_s']
            accepted.append(accepted_row)
            if route.model.time_s>=next_output-1e-13:
                publish();progress('output')
                next_output=min(end,next_clock(route.model.time_s,interval))
        if route.model.time_s>=end-1e-13:
            state='completed'
        require(all(sha(path)==value for path,value in original_sources.items()),
                'GPU AMR implementation changed during run')
        publish(frame=manifest['frames'][-1]['time_s']!=route.model.time_s)
    except InterruptedError:
        state='cancelled'
        if admitted:
            publish(frame=not manifest['frames'] or manifest['frames'][-1]['time_s']!=route.model.time_s)
    except Exception as error:
        state='failed';manifest['error']=repr(error)
        if admitted:
            publish(frame=not manifest['frames'] or manifest['frames'][-1]['time_s']!=route.model.time_s)
        raise
    finally:
        manifest.update(status=state,time_s=0. if route is None else route.model.time_s,
            steps=0 if route is None else route.model.accepted_steps,rejected=rejected,
            wall_s=time.perf_counter()-began)
        write_json_atomic(output/'accepted_steps.json',dict(rows=accepted))
        write_json_atomic(output/'run.json',manifest);progress(state)
    result=dict(status=state,output=str(output),time_s=0. if route is None else route.model.time_s,
        steps=0 if route is None else route.model.accepted_steps,rejected=rejected)
    if 'checkpoint' in manifest:
        result['checkpoint']=str(output/manifest['checkpoint']['path'])
    return result
