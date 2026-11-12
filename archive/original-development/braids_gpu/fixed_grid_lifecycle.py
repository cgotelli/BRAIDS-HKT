"""Shared accepted-step lifecycle for ordinary fixed-grid GPU physics.

Drivers supply physics admission, constructors and persistent-state contracts.
This module owns clocks, retries, cancellation, output and restart orchestration;
it never selects a backend or alters a physical control.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .atomic_output import save_arrays_atomic as save_arrays, write_json_atomic as write_json


def run(driver, *, input_path, output_path, until=None, output_every=None, dt_max=None,
        max_steps=None, stop_file=None, resume=None, launch_mode='graph', integrator='ssp43',
        capture_stages=False, capture_step_trace=False, source_archive=None, cancelled=lambda: False):
    from .runner import reject_deferred_public_controls, time_controls
    from validation.gpu_fwave.adaptive_hydraulics import next_hint, retry_step

    require, sha = driver.require, driver.sha
    candidate = driver.candidate_module()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError('Preserve existing GPU output')
    require(not capture_stages and not capture_step_trace and source_archive is None,
            'Ordinary fixed-grid execution requires its current-source route')
    require(max_steps is None or (type(max_steps) is int and max_steps > 0), 'Invalid max_steps')
    data, input_identity, prepared_identity = driver.read_prepared(input_path)
    profile_contract = (driver.execution_profile_contract(data)
        if hasattr(driver, 'execution_profile_contract') else
        dict(identities={}, method={}, manifest={}, result={}))
    require(isinstance(profile_contract, dict) and
            set(profile_contract) == {'identities', 'method', 'manifest', 'result'} and
            all(isinstance(profile_contract[key], dict) for key in profile_contract),
            'Execution-profile lifecycle contract differs')

    def lifecycle_result(**values):
        require(not (set(values) & set(profile_contract['result'])),
                'Execution-profile result fields overlap lifecycle result')
        values.update(profile_contract['result'])
        return values

    reject_deferred_public_controls(data)
    start, end, interval, dtmax = time_controls(data, until, output_every, dt_max)
    events = driver.scope(data, start=start, end=end, interval=interval, dtmax=dtmax,
                          launch_mode=launch_mode, integrator=integrator)
    require(end <= float(data['endtime']) and dtmax <= float(data['dtmax']),
            'Time overrides exceed prepared run')

    def stopped():
        return bool(cancelled() or (stop_file is not None and Path(stop_file).exists()))

    if stopped():
        return lifecycle_result(status='cancelled', output=str(destination), time_s=start, steps=0)
    contract = driver.execution_contract(data)
    if hasattr(driver, 'integrator_execution_contract'):
        contract = driver.integrator_execution_contract(data, integrator, contract)
    integration = (driver.integration_identity(integrator, launch_mode)
                   if hasattr(driver, 'integration_identity') else
                   dict(integrator='SSPRK33', euler_stages=3, launch_mode='plain'))
    with tempfile.TemporaryDirectory(prefix='braids_fixed_source_') as temporary:
        archive = Path(temporary)/'source'
        source_identity = (driver.archive_sources(archive, data, integrator=integrator)
                           if hasattr(driver, 'integration_identity') else
                           driver.archive_sources(archive, data))
        method = dict(variant=driver.VARIANT, **integration,
            precision='FP64', cfl=float(data['cfl_target']),
            dtmax_s=dtmax, output_every_s=interval, fractions=int(data['nfrac']),
            bed_layers=int(data['nbed_layers']), source_identity=source_identity,
            event_identity=events.identity, derived_boundary_cache_policy='invalidate_and_recompute',
            **contract['method'])
        require(not (set(method) & set(profile_contract['method'])),
                'Execution-profile method fields overlap lifecycle method')
        method.update(profile_contract['method'])
        method_identity = hashlib.sha256(json.dumps(method, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        identities = dict(input_identity=input_identity, source_identity=source_identity,
                          method_identity=method_identity, event_identity=events.identity)
        require(not (set(identities) & set(profile_contract['identities'])),
                'Execution-profile identity fields overlap lifecycle identities')
        identities.update(profile_contract['identities'])
        restored = driver.checkpoint_preflight(resume, data, events, identities=identities,
            start=start, end=end, expected_schema=contract['schema']) if resume is not None else None
        if stopped():
            return lifecycle_result(status='cancelled', output=str(destination), time_s=start, steps=0)
        destination.mkdir(parents=True)
        source = destination/'source'
        shutil.copytree(archive, source)
    manifest = dict(schema=1, backend='gpu', backend_selected_by='user', automatic_backend_selection=False,
        input_folder=str(Path(input_path).resolve()), input_preparation_sha256=prepared_identity,
        **identities, method=method, integrator=integration['integrator'],
        euler_stages=integration['euler_stages'],
        source_sha256=json.loads((source/'runtime_manifest.json').read_text())['source_files'],
        frames=[], status='initializing', start_s=start, end_s=end, output_every_s=interval,
        dtmax_s=dtmax, dx=float(data['dx']), dy=float(data['dy']),
        boundary_events=dict(identity=events.identity, knots_s=events.knots.tolist()),
        QGIS_release=False, physical_validation=False, Delft3D_source_modified=False,
        **contract['manifest'])
    require(not (set(manifest) & set(profile_contract['manifest'])),
            'Execution-profile manifest fields overlap lifecycle manifest')
    manifest.update(profile_contract['manifest'])
    began = time.perf_counter()
    current, hint, steps, rejected = start, dtmax, 0, 0
    state, index, solver, accepted = 'initializing', 0, None, []
    endpoint_written = False

    def progress(event):
        row = dict(event=event, status=state, backend='gpu', time_s=current, end_s=end,
            steps=steps, rejected=rejected, percent=100.*(current-start)/(end-start),
            wall_s=time.perf_counter()-began)
        write_json(destination/'status.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)

    def output(frame=True):
        nonlocal index, endpoint_written
        fields = dict(solver.fields(current=current), time=np.float64(current), dt_hint=np.float64(hint),
            steps=np.int64(steps), rejected=np.int64(rejected),
            next_boundary_knot=np.float64(events.next_after(current)),
            **{contract['schema_key']:np.int32(contract['schema'])},
            **{key:np.asarray(value) for key, value in identities.items()})
        # Capacities/filters are derived from accepted state, not restart state.
        checkpoint = {key:value for key,value in fields.items()
                      if key not in contract.get('diagnostics', ())}
        if hasattr(driver, 'checkpoint_extra'):
            extra = driver.checkpoint_extra(solver)
            require(not (set(extra) & set(checkpoint)),
                    'Integrator checkpoint fields overlap physical state')
            checkpoint.update({key:np.asarray(value) for key,value in extra.items()})
        checkpoint[contract['payload_key']] = np.asarray(driver.digest(checkpoint))
        save_arrays(destination/'checkpoint.npz', **checkpoint)
        if frame:
            fields[contract['payload_key']] = np.asarray(driver.digest(fields))
            name = 'frame_%06d.npz' % index
            save_arrays(destination/name, **fields)
            manifest['frames'].append(dict(file=name, time_s=current, steps=steps, sha256=sha(destination/name)))
            index += 1
        if (not endpoint_written and current >= end-1.e-13 and
                hasattr(driver,'endpoint_artifacts')):
            # The accepted frame/checkpoint above is frozen before any
            # endpoint-only production-pipeline diagnostic mutates scratch.
            endpoint=driver.endpoint_artifacts(solver,current,destination)
            if endpoint is not None:manifest['endpoint_artifacts']=endpoint
            endpoint_written=True
        write_json(destination/'run.json', manifest)

    write_json(destination/'run.json', manifest)
    try:
        progress('initializing')
        constructors = ({'integrator': integrator}
                        if hasattr(driver, 'integration_identity') else {})
        with candidate.constructors(source, data, **constructors) as Solver:
            solver = Solver(data, source_folder=source)
            if hasattr(solver, 'implicit_midpoint') and solver.implicit_midpoint is not None:
                solver.implicit_midpoint.bind_cancelled(stopped)
            cp = solver.cp
            if restored is not None:
                for key, value in solver.state.items():
                    value.set(restored[key])
                for key, value in solver.ledger.items():
                    value.set(restored['ledger_'+key])
                driver.restore_extra(solver, restored)
                current, hint = float(restored['time']), float(restored['dt_hint'])
                steps, rejected = int(restored['steps']), int(restored['rejected'])
                require(np.array_equal(solver.fields(current=current)['bed_elevation'], restored['bed_elevation']),
                        'Restored bed differs from accepted inventory')
            if hasattr(solver.boundary, 'normal_root'):
                solver.boundary.normal_root.invalidate_cache()
            elif getattr(solver.boundary, 'inlet_root', None) is not None:
                for value in solver.boundary.inlet_root.cache.values():
                    value.fill(0)
            device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
            manifest.update(device=device.decode() if isinstance(device, bytes) else str(device),
                cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cells=solver.nc, faces=solver.nf, fractions=solver.ng,
                constructor_s=time.perf_counter()-began)
            save_arrays(destination/'geometry.npz', **{key:np.asarray(data[key]) for key in
                ('xy', 'area', 'mid', 'owner', 'neighbour', 'normal', 'length', 'rock', 'manning_cell')})
            manifest['geometry_sha256'] = sha(destination/'geometry.npz')
            output()
            state = 'running'
            progress('started')
            next_output = min(end, start+(math.floor((current-start)/interval+1e-10)+1)*interval)
            while current < end-1e-13:
                if stopped():
                    state = 'cancelled'
                    break
                if max_steps is not None and steps >= max_steps:
                    state = 'step_limit'
                    break
                trial = events.trial(current, hint, output=next_output, end=end)
                dt, first = trial.dt, True
                for _ in range(20):
                    result = solver.step(dt, current)
                    status, peak = int(result['status'].item()), float(result['cfl'].item())
                    if status == 0:
                        break
                    rejected += 1
                    first = False
                    if stopped():
                        state = 'cancelled'
                        break
                    dt = retry_step(dt, peak, float(data['cfl_target']))
                else:
                    raise RuntimeError('GPU retry limit reached, status='+str(status))
                if state == 'cancelled':
                    break
                before = current
                current = float(current+dt)
                steps += 1
                hint = next_hint(dt, peak, float(data['cfl_target']), dtmax,
                    scheduled_clip=trial.scheduled_clip, first_attempt=first, unclipped=trial.unclipped_hint)
                accepted.append(dict(time_before=before, time_after=current, dt=dt, cfl=peak))
                if current >= next_output-1e-13:
                    output()
                    progress('output')
                    next_output = min(end, start+(math.floor((current-start)/interval+1e-10)+1)*interval)
            if current >= end-1e-13:
                state = 'completed'
            output(frame=manifest['frames'][-1]['time_s'] != current)
    except Exception as exc:
        state = 'failed'
        manifest['error'] = repr(exc)
        if solver is not None:
            output(frame=False)
        raise
    finally:
        manifest.update(status=state, time_s=current, final_time_s=current, steps=steps,
            rejected=rejected, wall_s=time.perf_counter()-began)
        write_json(destination/'accepted_steps.json', dict(rows=accepted))
        write_json(destination/'run.json', manifest)
        progress(state)
    return lifecycle_result(status=state, output=str(destination), time_s=current, steps=steps,
                            rejected=rejected, checkpoint=str(destination/'checkpoint.npz'))
