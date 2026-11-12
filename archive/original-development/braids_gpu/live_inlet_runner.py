"""Common-worker lifecycle for the qualified surveyed stage/critical inlet.

Accepted state, signed ledgers and the forcing clock are restart state. CUDA
root workspaces and exact-value caches are derived and are rebuilt on resume.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from . import g1_inlet_solver as candidate
from . import live_inlet_modes as inlet
from .atomic_output import save_arrays_atomic as save_arrays, write_json_atomic as write_json

ROOT = Path(__file__).resolve().parents[1]
VARIANT = inlet.VARIANT
ARTIFACTS = frozenset(('coupled_input.bin', 'coupled_input.npz', 'normal_metadata.bin'))
sha = candidate.SHA


def digest(fields):
    h = hashlib.sha256()
    for key in sorted(fields):
        value = np.ascontiguousarray(fields[key])
        h.update(key.encode() + b'\0' + value.dtype.str.encode() + b'\0')
        h.update(json.dumps(value.shape).encode() + b'\0' + value.tobytes())
    return h.hexdigest()


def read_prepared(folder):
    """Authenticate ordinary prepared records without a fixture-path selector."""
    folder = Path(folder).resolve()
    path = folder / 'preparation_receipt.json'
    inlet.require(path.is_file() and not path.is_symlink(), 'Missing regular inlet preparation receipt')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    inlet.require(receipt.get('schema') in (1, 4) and
                  ARTIFACTS <= set(receipt.get('artifacts', {})),
                  'Inlet preparation requires all three numerical artifacts')
    if receipt['schema'] == 4:
        from .graded_native_exporter import EXPORTER_SHA256
        inlet.require(receipt.get('variant') == VARIANT and
            set(receipt['artifacts']) == ARTIFACTS and
            receipt.get('exporter_executable_sha256') == EXPORTER_SHA256 and
            receipt.get('initialization_CPU_only') is True and
            receipt.get('native_solver_steps_performed') == 0 and
            receipt.get('GPU_solver_executed') is False,
            'Inlet schema-4 zero-step preparation identity differs')
    else:
        inlet.require(receipt.get('native_solver_steps_performed',
            receipt.get('solver_steps_performed')) == 0 and
            receipt.get('GPU_solver_executed', False) is False,
            'Inlet preparation must be a zero-step native export')
    for name, expected in receipt['artifacts'].items():
        member = (folder / name).resolve()
        inlet.require(member.parent == folder and member.is_file() and
                      not (folder / name).is_symlink() and sha(member) == expected,
                      'Prepared inlet artifact differs: ' + name)
    for name in ('scheduled_metadata.bin', 'coupled_diffusion.bin', 'avalanche_metadata.bin',
                 'bank_toe_metadata.bin', 'distributed_bank_metadata.bin',
                 'bank_history_metadata.bin'):
        inlet.require(not (folder / name).exists(), 'Inlet worker excludes sidecar: ' + name)
    if receipt['schema'] == 4:
        origins = receipt.get('origin', {}).get('files', {})
        inlet.require({'case.data', 'case.data.bin', 'geometry.npz'} <= set(origins),
                      'Inlet native deck and geometry identity is missing')
        for name, expected in origins.items():
            member = (folder / name).resolve()
            inlet.require(member.is_relative_to(folder) and member.is_file() and
                          not member.is_symlink() and sha(member) == expected,
                          'Prepared inlet deck origin differs: ' + name)
    data = inlet.read_case(folder)
    # The legacy handoff stores native one-based seeds; the reader derives
    # these two zero-based vectors. Schema 4 stores the complete decoded set.
    archive_keys = set(data)
    if receipt['schema'] == 1:
        archive_keys -= {'boundary_seed_position', 'section_seed_node'}
    with np.load(folder / 'coupled_input.npz', allow_pickle=False) as archive:
        inlet.require(set(archive.files) == archive_keys and all(
            archive[key].dtype == np.asarray(data[key]).dtype and
            np.array_equal(archive[key], data[key]) for key in archive.files),
            'Inlet NPZ differs from authenticated native records')
    numerical_identity = hashlib.sha256(json.dumps(
        {name:receipt['artifacts'][name] for name in sorted(ARTIFACTS)},
        sort_keys=True).encode()).hexdigest()
    return data, numerical_identity, sha(path)


def prepare(deck, output, *, stop_file=None):
    """Initialize ordinary native input with the pinned I/O-only exporter."""
    from . import graded_native_exporter as exporter
    from validation.gpu_fwave.coupled_reference import copy_inputs

    deck, output = Path(deck).resolve(), Path(output).resolve()
    inlet.require(deck.name == 'case.data', 'Inlet preparation requires a native case.data deck')
    def check_stop():
        if stop_file is not None and Path(stop_file).exists():
            raise InterruptedError('GPU inlet preparation cancelled')
    check_stop()
    if (ROOT / 'PACKAGE-MANIFEST.json').is_file():
        build = exporter.verify_packaged_exporter()
        executable = Path(build['executable'])
        environment = exporter.export_environment(build['runtime'])
    else:
        runtime = ROOT / 'validation/gpu_fwave/runs/i2_adaptation_export_20261004_v1/runtime'
        executable = runtime / 'xbraids_normal_export.exe'
        inlet.require(executable.is_file() and sha(executable) == exporter.EXPORTER_SHA256,
                      'Pinned inlet zero-step exporter differs')
        build = json.loads((runtime / 'build.json').read_text())
        environment = exporter.export_environment(runtime)
        compiler_directory = Path(build['command'][0]).parent
        inlet.require(all((compiler_directory / name).is_file()
                          for name in exporter.REQUIRED_DLLS), 'Native exporter runtime DLLs are missing')
        environment['PATH'] = str(compiler_directory) + os.pathsep + environment['PATH']
    origin = copy_inputs(deck, output, stop_file=stop_file)
    with (output / 'initial.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([str(executable), deck.name], cwd=output,
            env=environment, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        began = time.monotonic()
        try:
            while process.poll() is None:
                check_stop()
                if time.monotonic() - began > 240:
                    raise TimeoutError('GPU inlet zero-step preparation exceeded 240 seconds')
                time.sleep(.05)
        except (InterruptedError, TimeoutError):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, [str(executable), deck.name])
    data = inlet.read_case(output)
    events = candidate.validate(data)
    check_stop()
    np.savez_compressed(output / 'coupled_input.npz', **data)
    receipt = dict(schema=4, variant=VARIANT, origin=origin,
        artifacts={name:sha(output / name) for name in sorted(ARTIFACTS)},
        exporter_executable_sha256=exporter.EXPORTER_SHA256,
        initialization_CPU_only=True, native_solver_steps_performed=0,
        GPU_solver_executed=False)
    write_json(output / 'preparation_receipt.json', receipt)
    read_prepared(output)
    return dict(status='prepared', output=str(output), variant=VARIANT,
        cells=int(data['nc']), faces=int(data['nf']),
        forcing_knots_s=events.knots.tolist(),
        preparation_receipt_sha256=sha(output / 'preparation_receipt.json'),
        native_solver_steps_performed=0, archive_bound=False)


def archive_sources(folder, data):
    candidate.prepare_sources(folder, data)
    folder = Path(folder)
    for name in ('live_inlet_runner.py', 'runner.py', '__main__.py',
                 'atomic_output.py', 'graded_native_exporter.py'):
        target = folder / 'worker_python' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'braids_gpu' / name, target)
    files = {path.relative_to(folder).as_posix():sha(path)
             for path in folder.rglob('*') if path.is_file()}
    write_json(folder / 'runtime_manifest.json', dict(schema=1, variant=VARIANT,
        source_files=files, candidate_source_sha256=sha(folder / 'g1_sources.json'),
        compact_checkpoint_schema=1, numerical_CUDA_unchanged=True))
    return sha(folder / 'runtime_manifest.json')


def checkpoint_preflight(path, data, events, *, input_identity, source_identity,
                         method_identity, start, end):
    with np.load(path, allow_pickle=False) as archive:
        fields = {key:archive[key].copy() for key in archive.files}
    nc, ng, nl, nb = (int(data[key]) for key in ('nc', 'nfrac', 'nbed_layers', 'nb'))
    shapes = dict(q=(3, nc), layers=(nl, ng, nc), secondary=(nc,), bed_elevation=(nc,),
                  ledger_water=(nb,), ledger_bed=(ng, nb), ledger_secondary=(nb,), ledger_source=())
    text = dict(input_identity=input_identity, source_identity=source_identity,
                method_identity=method_identity, event_identity=events.identity)
    extra = {'time', 'steps', 'rejected', 'dt_hint', 'next_boundary_knot',
             'compact_checkpoint_schema', 'g1_payload_sha256'}
    inlet.require(set(fields) == set(shapes) | set(text) | extra,
                  'Inlet checkpoint schema differs')
    payload = fields.pop('g1_payload_sha256')
    inlet.require(payload.shape == () and payload.dtype.kind in 'US' and
                  payload.item() == digest(fields), 'Inlet checkpoint payload differs')
    for key, expected in text.items():
        value = fields[key]
        inlet.require(value.shape == () and value.dtype.kind in 'US' and
                      value.item() == expected, 'Inlet checkpoint identity differs: ' + key)
    for key, shape in shapes.items():
        value = fields[key]
        inlet.require(value.shape == shape and value.dtype == np.float64 and
                      np.isfinite(value).all(), 'Inlet checkpoint field differs: ' + key)
    for key in ('time', 'dt_hint', 'next_boundary_knot'):
        value = fields[key]
        inlet.require(value.shape == () and value.dtype == np.float64 and np.isfinite(value),
                      'Inlet checkpoint scalar differs: ' + key)
    for key in ('steps', 'rejected', 'compact_checkpoint_schema'):
        value = fields[key]
        inlet.require(value.shape == () and value.dtype.kind in 'iu' and int(value) >= 0,
                      'Inlet checkpoint integer differs: ' + key)
    current = float(fields['time'])
    inlet.require(start <= current <= end and float(fields['dt_hint']) > 0 and
                  fields['next_boundary_knot'].item() == events.next_after(current) and
                  int(fields['compact_checkpoint_schema']) == 1,
                  'Inlet checkpoint clock/schema differs')
    inlet.require(np.min(fields['q'][0]) >= 0 and np.min(fields['layers']) >= 0 and
                  not np.any(fields['secondary']) and not np.any(fields['ledger_secondary']) and
                  not np.any(fields['ledger_source']), 'Inlet checkpoint inactive or negative state differs')
    dry = fields['q'][0] <= float(data['dry_tolerance'])
    inlet.require(not np.any(fields['q'][1:, dry]), 'Inlet checkpoint dry-cell momentum differs')
    return fields


def run(*, input_path, output_path, until=None, output_every=None, dt_max=None,
        max_steps=None, stop_file=None, resume=None, launch_mode='graph', integrator='ssp43',
        capture_stages=False, capture_step_trace=False, source_archive=None,
        cancelled=lambda:False):
    from .runner import reject_deferred_public_controls, time_controls
    from validation.gpu_fwave.adaptive_hydraulics import next_hint, retry_step

    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError('GPU output folder already exists')
    inlet.require(not capture_stages and not capture_step_trace and source_archive is None,
                  'Inlet worker does not yet accept graph traces or packaged source overrides')
    inlet.require(max_steps is None or (type(max_steps) is int and max_steps > 0),
                  'Inlet worker requires positive integer max_steps')
    data, input_identity, preparation_identity = read_prepared(input_path)
    reject_deferred_public_controls(data)
    start, end, interval, dtmax = time_controls(data, until, output_every, dt_max)
    events = inlet.validate_scope(data, start=start, end=end, interval=interval,
        dtmax=dtmax, launch_mode=launch_mode, integrator=integrator)
    inlet.require(end <= float(data['endtime']) and dtmax <= float(data['dtmax']),
                  'Inlet worker time overrides exceed prepared duration/timestep')
    def stop_requested():
        return bool(cancelled() or (stop_file is not None and Path(stop_file).exists()))
    if stop_requested():
        return dict(status='cancelled', output=str(destination), time_s=start, steps=0)
    with tempfile.TemporaryDirectory(prefix='braids_inlet_source_') as temporary:
        archive = Path(temporary) / 'source'
        source_identity = archive_sources(archive, data)
        method = dict(variant=VARIANT, integrator='SSPRK33', euler_stages=3,
            launch_mode='plain', precision='FP64', cfl=float(data['cfl_target']),
            dtmax_s=dtmax, output_every_s=interval, morfac=1., bedlaw=2,
            exchange_mode=0, secondary_flow_model=0, source_identity=source_identity,
            event_identity=events.identity, compact_checkpoint_schema=1,
            derived_boundary_cache_policy='invalidate_and_recompute')
        method_identity = hashlib.sha256(json.dumps(method, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        restored = (checkpoint_preflight(resume, data, events,
            input_identity=input_identity, source_identity=source_identity,
            method_identity=method_identity, start=start, end=end) if resume else None)
        if stop_requested():
            return dict(status='cancelled', output=str(destination), time_s=start, steps=0)
        destination.mkdir(parents=True)
        source = destination / 'source'
        shutil.copytree(archive, source)
    manifest = dict(schema=1, backend='gpu', backend_selected_by='user',
        automatic_backend_selection=False, input_folder=str(Path(input_path).resolve()),
        input_preparation_sha256=preparation_identity, input_identity=input_identity,
        source_identity=source_identity, method_identity=method_identity, method=method,
        source_sha256=json.loads((source / 'runtime_manifest.json').read_text())['source_files'],
        integrator='SSPRK33', euler_stages=3, start_s=start, end_s=end,
        output_every_s=interval, dtmax_s=dtmax, dx=float(data['dx']), dy=float(data['dy']),
        frames=[], status='initializing', scope='Fixed-grid two-class surveyed stage/critical inlet MPM',
        boundary_events=dict(identity=events.identity, knots_s=events.knots.tolist(),
            mode=np.asarray(data['mode']).tolist(), section_control=np.asarray(data['section_control']).tolist()),
        compact_checkpoint_schema=1, physical_validation=False, QGIS_release=False)
    began = time.perf_counter()
    current, hint, steps, rejected = start, dtmax, 0, 0
    state = 'initializing'
    frame_index = 0
    solver = None
    accepted = []
    def progress(event):
        record = dict(event=event, status=state, backend='gpu', time_s=current,
            end_s=end, steps=steps, rejected=rejected,
            percent=100. if state == 'completed' else 100.*(current-start)/(end-start),
            wall_s=time.perf_counter()-began)
        write_json(destination / 'status.json', record)
        print(json.dumps(record, allow_nan=False), flush=True)
    def output(frame=True):
        nonlocal frame_index
        fields = dict(solver.fields(), time=np.float64(current), dt_hint=np.float64(hint),
            steps=np.int64(steps), rejected=np.int64(rejected),
            input_identity=np.asarray(input_identity), source_identity=np.asarray(source_identity),
            method_identity=np.asarray(method_identity), event_identity=np.asarray(events.identity),
            next_boundary_knot=np.float64(events.next_after(current)), compact_checkpoint_schema=np.int32(1))
        fields['g1_payload_sha256'] = np.asarray(digest(fields))
        save_arrays(destination / 'checkpoint.npz', **fields)
        if frame:
            filename = 'frame_%06d.npz' % frame_index
            save_arrays(destination / filename, **fields)
            manifest['frames'].append(dict(file=filename, time_s=current, steps=steps,
                                          sha256=sha(destination / filename)))
            frame_index += 1
        write_json(destination / 'run.json', manifest)
    write_json(destination / 'run.json', manifest)
    try:
        progress('initializing')
        with candidate.constructors(source, data) as Solver:
            solver = Solver(data, source_folder=source)
            cp = solver.cp
            if restored:
                for key, value in solver.state.items():
                    value.set(restored[key])
                for key, value in solver.ledger.items():
                    value.set(restored['ledger_' + key])
                current, hint = float(restored['time']), float(restored['dt_hint'])
                steps, rejected = int(restored['steps']), int(restored['rejected'])
                inlet.require(np.array_equal(solver.fields()['bed_elevation'], restored['bed_elevation']),
                              'Inlet restart bed differs from accepted layer inventory')
            if solver.boundary.inlet_root is not None:
                solver.boundary.inlet_root.invalidate_cache()
            device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
            manifest.update(device=device.decode() if isinstance(device, bytes) else str(device),
                cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cells=solver.nc, faces=solver.nf, fractions=solver.ng,
                suspended_enabled=False, diffusion_enabled=False,
                constructor_s=time.perf_counter()-began)
            save_arrays(destination / 'geometry.npz', **{key:np.asarray(data[key]) for key in
                ('xy', 'area', 'mid', 'owner', 'neighbour', 'normal', 'length', 'rock', 'manning_cell')})
            manifest['geometry_sha256'] = sha(destination / 'geometry.npz')
            output()
            state = 'running'
            progress('started')
            next_output = min(end, start + (math.floor((current-start)/interval + 1e-10)+1)*interval)
            while current < end - 1e-13:
                if stop_requested():
                    state = 'cancelled'
                    break
                if max_steps is not None and steps >= max_steps:
                    state = 'step_limit'
                    break
                trial = events.trial(current, hint, output=next_output, end=end)
                dt = trial.dt
                first = True
                for _ in range(20):
                    result = solver.step(dt, current)
                    status, peak = int(result['status'].item()), float(result['cfl'].item())
                    if status == 0:
                        break
                    rejected += 1
                    first = False
                    if stop_requested():
                        state = 'cancelled'
                        break
                    dt = retry_step(dt, peak, float(data['cfl_target']))
                else:
                    raise RuntimeError('Inlet GPU retry limit reached, status=' + str(status))
                if state == 'cancelled':
                    break
                before = current
                current = float(current + dt)
                steps += 1
                hint = next_hint(dt, peak, float(data['cfl_target']), dtmax,
                    scheduled_clip=trial.scheduled_clip, first_attempt=first,
                    unclipped=trial.unclipped_hint)
                accepted.append(dict(time_before=before, time_after=current, dt=dt, cfl=peak))
                if current >= next_output - 1e-13:
                    output()
                    progress('output')
                    next_output = min(end, start + (math.floor((current-start)/interval + 1e-10)+1)*interval)
            if current >= end - 1e-13:
                state = 'completed'
            output(frame=not manifest['frames'] or manifest['frames'][-1]['time_s'] != current)
    except Exception as exc:
        state = 'failed'
        manifest['error'] = repr(exc)
        if solver is not None:
            output(frame=False)
        raise
    finally:
        manifest.update(status=state, time_s=current, steps=steps, rejected=rejected,
                        wall_s=time.perf_counter()-began)
        write_json(destination / 'accepted_steps.json', dict(rows=accepted))
        write_json(destination / 'run.json', manifest)
        progress(state)
    return dict(status=state, output=str(destination), time_s=current, steps=steps,
                rejected=rejected, checkpoint=str(destination / 'checkpoint.npz'))
