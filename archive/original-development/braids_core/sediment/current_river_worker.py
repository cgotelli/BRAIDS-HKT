"""Prepared-input worker for the maintained fully wet CPU/CUDA river bundle.

One lifecycle on both backends. No laboratory geometry imports, backend
fallback, input physics disabling, or automatic admission to the QGIS UI.
"""
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import numpy as np

from braids_gpu.atomic_output import save_arrays_atomic, write_json_atomic
from .bend_methods_v1 import BendTrialRejected, digest
from .current_river_solver import (CurrentRiverControls, create_current_river_solver,
                                   validate_current_river_input)

SCHEMA = 'braids.current_river.prepared.v1'
PROFILE = 'current_river.json'
RECEIPT = 'current_river_prepared.json'


def _stop(path):
    return path is not None and Path(path).exists()


def _check_stop(path):
    if _stop(path):
        raise InterruptedError('Current river worker cancelled')


def validate_profile(profile):
    expected = {'schema', 'controls', 'hydraulic_spinup_s', 'end_time_s',
                'output_interval_s', 'dt_max_s', 'initial_momentum', 'boundaries'}
    if not isinstance(profile, dict) or set(profile) != expected or profile['schema'] != SCHEMA:
        raise ValueError('Current river profile fields/schema differ')
    controls = CurrentRiverControls(**profile['controls'])
    controls.validate()
    spin, end, output, dt = (profile[k] for k in
        ('hydraulic_spinup_s', 'end_time_s', 'output_interval_s', 'dt_max_s'))
    if (not np.isfinite([spin, end, output, dt]).all() or spin < 0
            or end <= spin or min(output, dt) <= 0):
        raise ValueError('Invalid current river clocks')
    if profile['initial_momentum'] not in ('prepared', 'zero'):
        raise ValueError('Explicit prepared or zero initial momentum required')
    if not isinstance(profile['boundaries'], dict):
        raise ValueError('Explicit sediment boundary definitions required')
    return controls


def boundary_arrays(data, definitions):
    """Resolve explicit zero-based prepared boundary IDs, never geometric guesses."""
    open_faces = (data['neighbour'] < 0) & (data['bid'] >= 0)
    ids = set(int(i) for i in np.unique(data['bid'][open_faces]))
    if set(definitions) != {str(i) for i in ids}:
        raise ValueError('Define each open sediment boundary exactly once')
    bc = np.zeros(int(data['nf']), dtype=np.int32)
    feed = np.zeros(int(data['nf']))
    for bid in sorted(ids):
        row = definitions[str(bid)]
        if not isinstance(row, dict) or row.get('kind') not in ('feed', 'outflow'):
            raise ValueError('Sediment boundary must be feed or outflow')
        faces = open_faces & (data['bid'] == bid)
        if row['kind'] == 'feed':
            if set(row) != {'kind', 'solid_flux_m2_s'}:
                raise ValueError('Feed requires solid volume flux per unit width')
            value = row['solid_flux_m2_s']
            if not np.isfinite(value) or value < 0:
                raise ValueError('Feed must be finite and nonnegative')
            feed[faces] = value
            bc[faces] = 1
        else:
            if set(row) != {'kind'}:
                raise ValueError('Outflow cannot carry unused feed settings')
            bc[faces] = 2
    return bc, feed


def _member(root, name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or ':' in name
            or '\\' in name or p.as_posix() != name):
        raise ValueError('Invalid prepared artifact path')
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Prepared artifact escaped its folder')
    return path


def prepare(deck, profile, output, *, stop_file=None):
    """Fresh native zero-step export, then freeze exact inputs and GPU sources."""
    from braids_gpu import general_live_transport as transport
    from braids_gpu.graded_native_exporter import EXPORTER_SHA256, export_environment
    from validation.gpu_fwave.coupled_reference import copy_inputs
    _check_stop(stop_file)
    controls = validate_profile(profile)
    deck = Path(deck).resolve(); output = Path(output).resolve()
    if deck.name != 'case.data':
        raise ValueError('Expected native case.data input')
    executable = Path(__file__).resolve().parents[2] / 'braids_gpu/native_exporter_v1/xbraids_normal_export.exe'
    if digest(executable) != EXPORTER_SHA256:
        raise ValueError('Native initializer identity differs')
    origin = copy_inputs(deck, output, stop_file=stop_file)
    with (output/'preparation.log').open('x') as log:
        process = subprocess.Popen([str(executable), 'case.data'], cwd=output,
            env=export_environment(executable.parent), stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        began = time.monotonic()
        try:
            while process.poll() is None:
                _check_stop(stop_file)
                if time.monotonic()-began > 240:
                    raise TimeoutError('Current river native preparation timed out')
                time.sleep(.05)
        except BaseException:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if process.returncode:
        raise RuntimeError('Native initialization failed; see preparation.log')
    _check_stop(stop_file)
    data = transport.read_records(output)
    validate_current_river_input(data,controls)
    boundary_arrays(data, profile['boundaries'])
    # This does not disable unsupported input settings. Constructors and the
    # underlying source preparation reject incompatible native configurations.
    transport.prepare_sources(output/'gpu_sources', data, integrator='implicit_fwave_midpoint')
    _check_stop(stop_file)
    write_json_atomic(output/PROFILE, dict(profile, controls=asdict(controls)))
    artifacts = {p.relative_to(output).as_posix(): digest(p)
                 for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'preparation.log'}
    receipt = dict(schema=SCHEMA, artifacts=artifacts, origin=origin,
                   native_exporter_sha256=EXPORTER_SHA256, solver_steps_performed=0,
                   worker_sha256=digest(__file__), public_qgis_admission=False)
    write_json_atomic(output/RECEIPT, receipt)
    return dict(status='prepared', input=str(output), cells=int(data['nc']),
                faces=int(data['nf']), receipt_sha256=digest(output/RECEIPT))


def load_prepared(folder):
    from braids_gpu.general_live_transport import read_records, verify_sources
    root = Path(folder).resolve()
    receipt = json.loads((root/RECEIPT).read_text(encoding='utf8'))
    artifacts = receipt.get('artifacts', {})
    if (receipt.get('schema') != SCHEMA or not isinstance(artifacts, dict)
            or not {PROFILE, 'case.data', 'coupled_input.bin', 'normal_metadata.bin'} <= set(artifacts)):
        raise ValueError('Incomplete current river preparation receipt')
    for name, expected in artifacts.items():
        path = _member(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Prepared artifact differs: '+name)
    profile = json.loads((root/PROFILE).read_text(encoding='utf8'))
    controls = validate_profile(profile)
    data = read_records(root)
    validate_current_river_input(data,controls)
    bc, feed = boundary_arrays(data, profile['boundaries'])
    verify_sources(root/'gpu_sources')
    return root, profile, controls, data, bc, feed


def run(input_path, output_path, *, backend, until=None, resume=None,
        stop_file=None, max_steps=None, progress=None):
    """Run a prepared case; publish accepted states only and preserve restarts."""
    from braids_gpu.general_live_transport import constructors
    if backend not in ('cpu', 'gpu'):
        raise ValueError('Choose backend explicitly: cpu or gpu')
    _check_stop(stop_file)
    root, profile, controls, data, bc, feed = load_prepared(input_path)
    target = profile['end_time_s'] if until is None else float(until)
    if not np.isfinite(target) or not 0 < target <= profile['end_time_s']:
        raise ValueError('Target must lie within the prepared run clock')
    if max_steps is not None and (isinstance(max_steps,bool) or int(max_steps)!=max_steps or max_steps<1):
        raise ValueError('Positive integer maximum step count required')
    output = Path(output_path).resolve(); output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output/'STATUS.json', dict(status='initializing', backend=backend))
    context = (constructors(root/'gpu_sources', data, integrator='implicit_fwave_midpoint')
               if backend == 'gpu' else nullcontext(None))
    with context as Solver:
        try:
            options = dict(gpu_solver_type=Solver, gpu_source_folder=root/'gpu_sources') if backend=='gpu' else {}
            solver, owner, p = create_current_river_solver(root/'case.data', data,
                backend=backend, controls=controls, bedbc=bc, feed=feed, **options)
            owner.identity['current_river_worker'] = dict(schema=SCHEMA,
                source_sha256=digest(__file__), prepared_receipt_sha256=digest(root/RECEIPT))
            if backend == 'cpu': solver.cancelled = lambda: _stop(stop_file)
            else: solver.implicit_midpoint.bind_cancelled(lambda: _stop(stop_file))
            q = lambda: solver.q if backend=='cpu' else solver.state['q']
            z = lambda: solver.z if backend=='cpu' else solver.bed()
            if resume is not None:
                owner.restart(resume)
            elif profile['initial_momentum'] == 'zero':
                q()[1:] = 0.
            if p.time >= target-1e-10:
                raise ValueError('Restart is at or beyond requested target')
        except BaseException as exc:
            write_json_atomic(output/'STATUS.json',dict(status='setup_failed',error=str(exc),backend=backend))
            raise
        write_json_atomic(output/'IDENTITY.json', owner.identity)
        write_json_atomic(output/'run.json', dict(schema=SCHEMA, backend=backend,
            prepared_input=str(root), controls=profile, restart=str(resume) if resume else None,
            frame_format='current_river_frames_v1', public_qgis_admission=False))
        frames=[]
        def save_frame():
            frame=dict(time=p.time, q=p.engine.host(q()).copy(), z=p.engine.host(z()).copy(),
                       hI=p.engine.host(p.W).copy())
            if not frames or abs(frame['time']-frames[-1]['time'])>1e-10: frames.append(frame)
            save_arrays_atomic(output/'frames.npz', **{k:np.stack([f[k] for f in frames]) for k in frame})
        save_frame()
        start=p.time; initial_water=float(p.xp.sum(q()[0]*p.g['area']))
        initial_solid=float(p.xp.sum((1-p.parameters.porosity)*z()*p.g['area']))
        bed_ledger0=float(p.xp.sum(p.boundary_ledger))
        water0=float(solver.water_ledger.sum()) if backend=='cpu' else 0.
        water=0.; worst_water=0.; worst_solid=0.; accepted=0; rejected=[]
        interval=profile['output_interval_s']; spin=profile['hydraulic_spinup_s']
        next_output=min(target,(math.floor((p.time+1e-8)/interval)+1)*interval)
        began=time.perf_counter(); status='complete'; error=None
        def publish():
            value=dict(status='running', backend=backend, time_s=p.time, target_s=target,
                phase='hydraulic_spinup' if p.time<spin else 'morphodynamics',
                percent=100*p.time/target, steps=accepted,
                morphological_time_s=max(0.,p.time-spin)*controls.morfac,
                wall_s=time.perf_counter()-began,
                maximum_water_residual_m3=worst_water, maximum_solid_residual_m3=worst_solid)
            write_json_atomic(output/'PROGRESS.json',value)
            if progress is not None: progress(value)
        publish()
        try:
            while p.time<target-1e-8:
                _check_stop(stop_file)
                if max_steps is not None and accepted>=max_steps:
                    status='step_limit'; break
                owner.morphology=p.time>=spin-1e-9
                dt=min(profile['dt_max_s'],target-p.time,next_output-p.time)
                if p.time<spin-1e-9: dt=min(dt,spin-p.time)
                oldtime=p.time
                for attempt in range(9):
                    try:
                        result=solver.step(dt,p.time)
                        _check_stop(stop_file) if p.time==oldtime else None
                        if backend=='gpu' and int(result['status'].item()):
                            raise BendTrialRejected(str(result['diagnostics']))
                        break
                    except BendTrialRejected as exc:
                        _check_stop(stop_file)
                        if p.time!=oldtime or p._open:
                            raise RuntimeError('Rejected step changed accepted clock') from exc
                        rejected.append(dict(time_s=oldtime,dt_s=dt,reason=str(exc)))
                        if attempt==8: raise
                        dt*=.5
                accepted+=1
                if backend=='gpu': water+=float(p.xp.sum(solver.implicit_midpoint.water_ledger))
                else: water=float(solver.water_ledger.sum())-water0
                wr=float(p.xp.sum(q()[0]*p.g['area']))-initial_water-water
                sr=float(p.xp.sum((1-p.parameters.porosity)*z()*p.g['area']))-initial_solid-(float(p.xp.sum(p.boundary_ledger))-bed_ledger0)
                worst_water=max(worst_water,abs(wr)); worst_solid=max(worst_solid,abs(sr))
                if not np.isfinite([wr,sr]).all() or abs(wr)>1e-7 or abs(sr)>1e-8:
                    raise RuntimeError('Current river conservation guard failed')
                if p.time>=next_output-1e-8:
                    save_frame(); owner.checkpoint(output/'checkpoint.npz'); publish()
                    next_output=min(target,next_output+interval)
        except InterruptedError as exc: status='cancelled'; error=str(exc)
        except BaseException as exc: status='failed'; error=f'{type(exc).__name__}: {exc}'
        save_frame(); owner.checkpoint(output/'checkpoint.npz'); publish()
        summary=dict(status=status, error=error, backend=backend, time_s=p.time,
            segment_start_s=start, target_s=target, percent=100*p.time/target,
            steps=accepted, rejected_trials=rejected, wall_s=time.perf_counter()-began,
            maximum_water_residual_m3=worst_water, maximum_solid_residual_m3=worst_solid)
        write_json_atomic(output/'STATUS.json',summary)
        write_json_atomic(output/'PROGRESS.json',summary)
        if status=='failed': raise RuntimeError(error)
        return summary
