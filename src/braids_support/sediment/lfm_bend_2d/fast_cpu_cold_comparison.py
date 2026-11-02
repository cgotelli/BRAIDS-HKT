"""Finite MF25 cold bend rerun against the preserved 6.27-hour CPU trajectory.

This runner does not compile, install, extend, poll or restart a simulation.
It accepts only a frozen MF25 validation runtime. The two native executions are
sequential, and a failed cold hydraulic gate prevents the mobile-bed phase.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from . import cold_optimized_comparison as previous
from . import profile_current

r = previous.r
a = previous.a
TAG = 'fast_cpu_cold20h_20261001_v1'
REFERENCE = previous.REFERENCE
HYDRAULIC_REFERENCE = previous.HYDRAULIC_REFERENCE
CFL = .8
PHYSICAL_KEYS = tuple(key for key in previous.MATCH_KEYS if key != 'cfl')
LIMITS = dict(bed_rms_m=.003, bed_maximum_m=.005, water_budget_m3=1e-6,
              sediment_budget_kg=1e-5, secondary_budget=1e-8,
              momentum_budget=1e-8, minimum_inventory_kg_m2=-1e-10)


def resolve_runtime(executable=None, build_json=None):
    """Verify the executable and retained compiled sources before each phase."""
    if (executable is None) != (build_json is None):
        raise ValueError('Supply both --executable and --build-json, or neither')
    if executable is None:
        from . import fast_cpu_release
        executable, build = fast_cpu_release.runtime(variant='mf25')
        executable = Path(executable).resolve()
    else:
        executable = Path(executable).resolve()
        build_json = Path(build_json).resolve()
        build = previous.read(build_json)
    if build.get('constant_mf_limit') != 25:
        raise ValueError('The cold runner requires the MF25 validation-copy runtime, not production')
    if not build.get('source_sha256') or r.sha256(executable) != build.get('executable_sha256'):
        raise ValueError('Frozen executable identity is missing or changed')
    # New release runtimes retain each compiled source alongside the executable.
    sources = {}
    for name, digest in build['source_sha256'].items():
        source = executable.parent / Path(name.replace('\\', '/')).name
        if not source.is_file() or r.sha256(source) != digest:
            raise ValueError(f'Frozen compiled source changed or missing: {source}')
        sources[str(source)] = digest
    identity = dict(build, executable=str(executable), verified_source_files=sources)
    if build_json is not None:
        identity.update(build_json=str(build_json), build_json_sha256=r.sha256(build_json))
    return executable, identity


def assert_matching(left, right):
    """CFL is the sole allowed deck-setting change in this matched case."""
    for key in PHYSICAL_KEYS:
        if not np.isclose(left[key], right[key], rtol=1e-12, atol=0):
            raise ValueError(f'Changed physical/run setting: {key}')
    if not np.isclose(left['cfl'], .4, rtol=0, atol=1e-14):
        raise ValueError('Historical CFL must be 0.4')
    if not np.isclose(right['cfl'], CFL, rtol=0, atol=1e-14):
        raise ValueError('New case CFL must be the declared 0.8')


def assert_geometry(source, target):
    with np.load(source / 'analysis_geometry.npz') as left, np.load(target / 'analysis_geometry.npz') as right:
        if set(left.files) != set(right.files):
            raise ValueError('Analysis geometry fields changed')
        for key in left.files:
            np.testing.assert_allclose(left[key], right[key], rtol=0, atol=1e-12,
                                       err_msg=f'Changed geometry: {key}')


def prepare_cold(target, report):
    """Rewrite the verified original cold case through the fingerprinting writer.

    The binary contains physical mesh/state arrays, not the text fingerprint.
    Its exact identity proves no evolved bed or old hydraulic state was reused.
    """
    original = previous.verify_inputs(HYDRAULIC_REFERENCE)
    if not original['hydraulic'] or original['seed'] != 'None':
        raise ValueError('Reference must be original cold hydraulic inputs')
    previous.clone_cold_inputs(HYDRAULIC_REFERENCE, report / 'original_cold_inputs')
    r.prepare_native(target, dx=.10625, end=300., output=50., dt=.0125,
                     morfac=1., hydraulic=True, seed=None, cfl=CFL)
    current = previous.verify_inputs(target)
    assert_matching(original, current)
    assert_geometry(HYDRAULIC_REFERENCE, target)
    original_binary = r.sha256(HYDRAULIC_REFERENCE / 'case.data.bin')
    if r.sha256(target / 'case.data.bin') != original_binary:
        raise ValueError('Regenerated cold physical binary differs from the original')
    current.update(initialization='Original cold physical arrays; no checkpoint or evolved bed',
        cold_reference_folder=str(HYDRAULIC_REFERENCE),
        original_cold_binary_sha256=original_binary,
        original_input_provenance_sha256=r.sha256(HYDRAULIC_REFERENCE / 'input_provenance.json'),
        numerical_changes=dict(historical_cfl=.4, candidate_cfl=CFL,
            method='Frozen new CPU default with documented eligibility/fallback'),
        production_source_modified=False)
    r.write_json(target / 'input_provenance.json', current)
    return current


def execute(folder, executable, build, threads, timeout_s):
    """Record a native completion or timeout without inheriting research toggles."""
    previous.verify_inputs(folder)
    if r.sha256(executable) != build['executable_sha256']:
        raise ValueError('Runtime changed before execution')
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith('BRAIDS_')}
    settings = dict(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
        OMP_WAIT_POLICY='PASSIVE', BRAIDS_PROFILE='1', BRAIDS_PROGRESS_STEPS='20000')
    environment.update(settings)
    started = time.perf_counter()
    return_code = None
    timed_out = False
    failure = None
    with (folder / 'native.log').open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run([str(executable), 'case.data'], cwd=folder,
                env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as exc:
            failure = f'{type(exc).__name__}: {exc}'
    text = (folder / 'native.log').read_text(encoding='utf-8', errors='replace')
    profile = profile_current.parse_profile(text)
    wall = time.perf_counter() - started
    record = dict(return_code=return_code, wall_s=wall, threads=threads,
        timed_out=timed_out, launch_error=failure, environment=settings,
        executable_sha256=r.sha256(executable), source_sha256=build['source_sha256'],
        input_provenance_sha256=r.sha256(folder / 'input_provenance.json'),
        native_log_sha256=r.sha256(folder / 'native.log'), profile=profile,
        compute_s=profile.get('profile: measured run wall seconds', wall)
                  - profile.get('profile: result output wall seconds', 0),
        steps=profile.get('steps'), rejected=profile.get('rejected'),
        inherited_braids_settings_removed=True,
        numerical_settings='CFL comes from the fingerprinted input; CPU method uses frozen runtime default')
    r.write_json(folder / 'execution.json', record)
    if timed_out or failure or return_code != 0:
        raise RuntimeError(f'Native execution failed: {folder}; return_code={return_code}, timeout={timed_out}, error={failure}')
    previous.verify_inputs(folder)
    if record['executable_sha256'] != build['executable_sha256']:
        raise RuntimeError('Executable changed during execution')
    return record


def compare_series(old, new):
    assert_matching(old['p'], new['p'])
    for key in ('xy', 'area', 'th'):
        np.testing.assert_allclose(old[key], new[key], rtol=0, atol=1e-10)
    np.testing.assert_allclose(old['z'][0], new['z'][0], rtol=0, atol=1e-12)
    result = dict(morphological_hours=(old['th'] / 3600).tolist())
    for label, key in (('bed', 'z'), ('depth', 'h'), ('speed', 'speed')):
        difference = new[key] - old[key]
        if not np.isfinite(difference).all():
            raise ValueError(f'Nonfinite {label} comparison')
        result[label + '_rms_difference'] = np.sqrt(
            np.average(difference ** 2, axis=1, weights=old['area'])).tolist()
        result[label + '_maximum_difference'] = np.max(abs(difference), axis=1).tolist()
    return result


def trajectory_checks(data, series):
    """Whole-history checks, so an inaccurate intermediate state cannot pass."""
    area = data['area']
    water = (data['h'] - data['h'][0]) @ area
    water -= data['win'] - data['win'][0]
    water += data['wout'] - data['wout'][0]
    sediment = ((data['z'] - data['z'][0]) @ area) * r.BULK
    sediment -= data['sin'] - data['sin'][0]
    sediment += data['sout'] - data['sout'][0]
    bed_rms = np.asarray(series['bed_rms_difference'])
    bed_maximum = np.asarray(series['bed_maximum_difference'])
    gates = dict(
        every_frame_bed_rms=bool(np.all(bed_rms <= LIMITS['bed_rms_m'])),
        every_frame_bed_maximum=bool(np.all(bed_maximum <= LIMITS['bed_maximum_m'])),
        every_frame_water_balance=bool(np.all(abs(water) <= LIMITS['water_budget_m3'])),
        every_frame_sediment_balance=bool(np.all(abs(sediment) <= LIMITS['sediment_budget_kg'])),
        nonnegative_depth=bool(np.min(data['h']) >= 0),
        finite_fields=bool(all(np.isfinite(data[key]).all() for key in ('h', 'z', 'speed'))))
    return dict(gates=gates, passed=all(gates.values()), limits=LIMITS,
        maximum_bed_rms_difference_m=float(bed_rms.max()),
        maximum_bed_difference_m=float(bed_maximum.max()),
        failed_bed_frames=np.flatnonzero((bed_rms > LIMITS['bed_rms_m']) |
                                        (bed_maximum > LIMITS['bed_maximum_m'])).tolist(),
        water_residual_m3=water.tolist(), sediment_residual_kg=sediment.tolist(),
        maximum_water_residual_m3=float(abs(water).max()),
        maximum_sediment_residual_kg=float(abs(sediment).max()))


def morphology_history(old, new):
    result = {}
    for label, data in (('historical', old), ('candidate', new)):
        dz = data['z'] - data['z'][0]
        result[label] = dict(erosion_volume_m3=np.maximum(-dz, 0).dot(data['area']).tolist(),
            deposition_volume_m3=np.maximum(dz, 0).dot(data['area']).tolist())
    signs = []
    for old_bed, new_bed in zip(old['z'] - old['z'][0], new['z'] - new['z'][0]):
        significant = abs(old_bed) > .002
        signs.append(float(np.average(np.sign(old_bed[significant]) == np.sign(new_bed[significant]),
            weights=old['area'][significant])) if significant.any() else None)
    result['sign_agreement_fraction_where_reference_change_exceeds_2mm'] = signs
    return result


def render_figures(old, new, report):
    """Reuse the established raw-cell renderer with this comparison contract.

    A private globals mapping avoids modifying the historical module or passing
    fictional metadata merely to bypass its old requirement of equal CFL.
    """
    from types import FunctionType
    renderer = FunctionType(previous.visuals.__code__,
        dict(previous.visuals.__globals__, compare_series=compare_series),
        previous.visuals.__name__, previous.visuals.__defaults__, previous.visuals.__closure__)
    renderer.__kwdefaults__ = previous.visuals.__kwdefaults__
    renderer(old, new, report)


def finish(folder, report, old):
    new = a.load(folder)
    series = compare_series(old, new)
    checks = trajectory_checks(new, series)
    accounts = previous.secondary.evidence(new)
    checks['gates'].update(
        secondary_balance=accounts['secondary_inventory_budget_max_residual'] <= LIMITS['secondary_budget'],
        momentum_balance=accounts['momentum_budget_max_residual'] <= LIMITS['momentum_budget'])
    frames = r.shared.read_frames(r.shared.native_frames_path(folder))
    minimum_inventory = min(float(frame['q'][:, 3:].min()) for frame in frames)
    checks['minimum_sediment_inventory_kg_m2'] = minimum_inventory
    checks['gates']['nonnegative_sediment'] = minimum_inventory >= LIMITS['minimum_inventory_kg_m2']
    checks['gates']['all_native_states_finite'] = all(np.isfinite(frame['q']).all() for frame in frames)
    checks['passed'] = bool(all(checks['gates'].values()))
    stationarity = previous.equilibrium.equilibrium_review(new)
    old_wall, new_wall = old['execution']['wall_s'], new['execution']['wall_s']
    result = dict(checks=checks, series=series, new_accounts=accounts,
        historical_accounts=a.audit(old), morphology=morphology_history(old, new),
        new_stationarity=stationarity, old_mobile_wall_s=old_wall,
        new_mobile_wall_s=new_wall, mobile_speedup=old_wall / new_wall,
        mobile_wall_reduction_percent=100 * (1 - new_wall / old_wall),
        wall_time_caveat='Historical/current whole mobile-run timings, not paired isolated timing measurements; warm-up is separate',
        numerical_changes=dict(historical_cfl=.4, candidate_cfl=CFL),
        previous_run_stationary=False, experimental_accuracy_validated=False,
        delft_modified=False, automatic_extension=False)
    r.write_json(report / 'comparison.json', result)
    render_figures(old, new, report)
    r.write(report / 'RESULTS.txt', f'''FAST CPU COLD BEND COMPARISON

The new cold hydraulic run completed 300 seconds on the original uneroded bed.
The mobile phase completed 2880 hydraulic seconds at MF25 = 20 morphological hours.
Mesh, forcing, sediment, geometry, timestep ceiling and output cadence match the
preserved 6.27-hour historical run. CFL changes from 0.4 to 0.8; the frozen CPU
runtime identity and selected method/fallback are recorded in build.json and native.log.

All-frame numerical agreement and conservation gates passed: {checks['passed']}
Largest area-weighted bed RMS difference: {checks['maximum_bed_rms_difference_m'] * 1000:.6f} mm (limit 3 mm).
Largest cell bed difference: {checks['maximum_bed_difference_m'] * 1000:.6f} mm (limit 5 mm).
Water residual maximum: {checks['maximum_water_residual_m3']:.6e} m3.
Sediment residual maximum: {checks['maximum_sediment_residual_kg']:.6e} kg.
Mobile wall time: {new_wall / 3600:.4f} h versus {old_wall / 3600:.4f} h historically;
measured historical/current speed ratio {old_wall / new_wall:.3f}. Warm-up is separate.
These are not alternating paired timings and neither numerical model is truth.

New final three-hour stationarity screen: {stationarity['passed']}.
Twenty hours is a comparison duration, not an assumed equilibrium duration.

Figures: EVOLUTION.png, EVOLUTION.gif, animation_png/, FINAL_PROFILES.png,
DIFFERENCES.png. Raw-cell maps use shared scales; no bed smoothing is applied.
Full histories, all-frame gate failures, erosion/deposition volumes, experimental
profile context and budget evidence are in comparison.json.

Next milestone: inspect the completed trajectory and any failed criteria before
extending method eligibility. This job does not install a release, modify Delft3D,
claim GPU acceleration or automatically extend the simulation.
''')
    return result


def run(tag=TAG, *, executable=None, build_json=None, threads=8, timeout_s=172800):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        raise ValueError('Use a simple unique tag')
    if threads < 1 or timeout_s <= 0:
        raise ValueError('Threads and timeout must be positive')
    report = r.HERE / 'reports' / tag
    hydro = r.HERE / 'runs' / f'{tag}_hydraulic_braids'
    mobile = r.HERE / 'runs' / f'{tag}_mobile_braids'
    if any(path.exists() for path in (report, hydro, mobile)):
        raise FileExistsError('Refusing to overwrite a previous run')
    exe, identity = resolve_runtime(executable, build_json)
    report.mkdir(parents=True)
    executions = {}
    def status(state, phase, **extra):
        record = dict(status=state, phase=phase, supervisor_pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(), executions=executions, **extra)
        r.write_json(report / 'status.json', record)
        print(json.dumps(record), flush=True)
    original_build, original_writer = r.shared.verified_braids_build, r.write_cut_case
    phase = 'preflight'
    try:
        status('preparing', phase)
        reference_meta = previous.verify_inputs(REFERENCE)
        old = a.load(REFERENCE)
        r.write_json(report / 'build.json', identity)
        r.write_json(report / 'reference.json', dict(folder=str(REFERENCE),
            input_provenance_sha256=r.sha256(REFERENCE / 'input_provenance.json'),
            execution_sha256=r.sha256(REFERENCE / 'execution.json'),
            frames_sha256=r.sha256(r.shared.native_frames_path(REFERENCE)),
            execution=old['execution']))
        r.write_json(report / 'plan.json', dict(tag=tag, threads=threads,
            hydraulic_end_s=300, mobile_hydraulic_end_s=2880, morfac=25,
            morphological_end_s=72000, output_morphological_s=600, dx_m=.10625,
            cells=5012, dt_max_s=.0125, historical_cfl=.4, candidate_cfl=CFL,
            limits=LIMITS, initialization='New 300-second cold hydraulics and original bed; no old checkpoint',
            secondary_transition='Public writer resets secondary intensity, matching the historical protocol',
            hydraulic_folder=str(hydro), mobile_folder=str(mobile), reference_folder=str(REFERENCE),
            executable_sha256=identity['executable_sha256'], script_sha256=r.sha256(Path(__file__)),
            timeout_per_phase_s=timeout_s, delft_executed=False, delft_modified=False,
            periodic_polling=False, auto_extension=False))
        r.write(report / 'RUN.txt', 'Finite cold300 -> MF25 mobile2880 bend comparison.\n'
            'Status: status.json. Inputs, runtime identity and limits: plan.json/build.json.\n'
            'No checkpoint seed, periodic polling, automatic extension or installation.\n'
            'All-frame bed limits: RMS 3 mm; maximum 5 mm. Conservation remains required.\n')
        r.shared.verified_braids_build = lambda: (exe, identity)
        r.write_cut_case = previous.mf25.validation_writer(report)
        phase = 'hydraulic'
        prepare_cold(hydro, report)
        resolve_runtime(executable, build_json)
        status('running', phase, folder=str(hydro))
        executions[phase] = execute(hydro, exe, identity, threads, timeout_s)
        water = a.load(hydro)
        check = previous.warmup_gates(water)
        r.write_json(report / 'hydraulic_checks.json', check)
        if not check['passed']:
            raise RuntimeError(f"Cold warm-up failed: {check['gates']}")
        np.testing.assert_allclose(water['z'][-1], water['z'][0], rtol=0, atol=1e-12)
        phase = 'mobile'
        status('preparing', phase, folder=str(mobile))
        r.prepare_native(mobile, dx=.10625, end=2880, output=24, dt=.0125,
            morfac=25, hydraulic=False, seed=hydro, cfl=CFL)
        meta = previous.verify_inputs(mobile)
        assert_matching(reference_meta, meta)
        assert_geometry(REFERENCE, mobile)
        meta.update(candidate='Frozen new CPU default; validation-copy constant MF25 guard',
            production_source_modified=False, curvature_regularization_depth_factor=2.,
            hydraulic_seed_frames_sha256=r.sha256(r.shared.native_frames_path(hydro)),
            comparison_reference=str(REFERENCE), no_evolved_bed_restart=True,
            numerical_changes=dict(historical_cfl=.4, candidate_cfl=CFL))
        r.write_json(mobile / 'input_provenance.json', meta)
        resolve_runtime(executable, build_json)
        status('running', phase, folder=str(mobile))
        executions[phase] = execute(mobile, exe, identity, threads, timeout_s)
        phase = 'analysis'
        status('analyzing', phase, folder=str(mobile))
        result = finish(mobile, report, old)
        status('complete' if result['checks']['passed'] else 'completed_failed_gates', phase,
            report=str(report / 'RESULTS.txt'), passed=result['checks']['passed'],
            stationary=result['new_stationarity']['passed'])
        return result
    except Exception as exc:
        status('failed', phase, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        r.shared.verified_braids_build = original_build
        r.write_cut_case = original_writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--build-json', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=172800)
    args = parser.parse_args()
    result = run(args.tag, executable=args.executable, build_json=args.build_json,
                 threads=args.threads, timeout_s=args.timeout)
    return 0 if result['checks']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
