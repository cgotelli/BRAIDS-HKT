"""Frozen small native reference for GPU suspended sediment integration.

This is an engineering CPU/GPU comparator, not an experimental braids_support.
The native evolution executable and its I/O-only preparation derivative are
pre-existing, unchanged binaries. Each new output folder is write-once.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from braids_core.boundaries.cut_cell_io import CutBoundary, write_cut_case
from braids_qgis.core.results import read_frames
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen_cpu
from . import i2_multilayer_native_case as i2
from . import normal_coupled_oracle as tight_exporter


HERE = Path(__file__).resolve().parent
TAG = 'i3_suspension_native_20261003_v3'
ROOT = HERE / 'runs' / TAG
EXPORT_RUNTIME = HERE / 'runs' / 'i3_suspension_native_export_20261003_v1' / 'runtime'
VARIANTS = {'settling': 1, 'van_rijn': 2}
INITIAL_CONCENTRATION = np.array([.020, .016, .012, .009, .006, .004, .002])
INLET_CONCENTRATION = np.array([.030, .024, .018, .013, .009, .006, .003])
SETTLING_M_S = np.array([.0004, .0006, .0010, .0015, .0020, .0030, .0040])
END_TIME_S, DT_MAX_S, OUTPUT_DT_S = 1., .002, .25
ENV = dict(i2.ENV, BRAIDS_TIME_INTEGRATOR='legacy')
EXPECTED_INTEGRATOR = 33
EXPECTED_OUTLET_EXTENSION = 0


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sha(path):
    return i2.sha(path)


def save(path, value):
    i2.save(path, value)


def verify_file_map(mapping):
    i2.verify_file_map(mapping)


def fractions():
    return tuple(replace(fraction, settling_velocity_m_s=float(ws))
                 for fraction, ws in zip(i2.fractions(), SETTLING_M_S))


def initial_fields():
    mesh, xy, area, state, bed, rock, shoal = i2.mesh_and_initial()
    x, y = xy.T
    multiplier = 1. + .25 * np.sin(.85 * x) + .15 * np.cos(2. * y)
    state[:, 3:3 + i2.GRAIN_COUNT] = (
        state[:, 0, None] * multiplier[:, None] * INITIAL_CONCENTRATION[None, :])
    require(np.all(state[:, 3:10] >= 0.) and np.all(state[state[:, 0] == 0., 3:10] == 0.) and
            np.all(np.max(state[:, 3:10], axis=0) > 0.),
            'All seven suspended classes must start active without dry-cell mass')
    return mesh, xy, area, state, bed, rock, shoal


def source_files():
    return {**i2.dependencies(), str(Path(__file__).resolve()): sha(__file__)}


def prepare(output=ROOT):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a fresh output folder; existing fixture is immutable')
    _, native_files = frozen_cpu.runtime_identity()
    build = tight_exporter.build(EXPORT_RUNTIME)
    runtime = Path(build['runtime'])
    require(build['parent_executable_sha256'] == frozen_cpu.TIGHT_SHA,
            'I/O-only exporter must derive from the exact native evolution runtime')
    sources = {**source_files(), **native_files,
               str(runtime / 'build.json'): sha(runtime / 'build.json'),
               str(runtime / 'xbraids_normal_export.exe'): build['executable_sha256']}
    verify_file_map(sources)
    mesh, xy, area, state, bed, rock, shoal = initial_fields()
    output.mkdir(parents=True)
    cases = []
    grain_ids = [item.fraction_id for item in fractions()]
    inlet = {grain_id: float(value) for grain_id, value in zip(grain_ids, INLET_CONCENTRATION)}
    for variant, exchange in VARIANTS.items():
        folder = output / 'inputs' / variant
        folder.mkdir(parents=True)
        write_cut_case(folder / 'case.data', mesh, state, rock,
            {'in': CutBoundary('total_discharge', discharge_m3_s=.08,
                               fraction_concentration_kg_m3=inlet),
             'out': CutBoundary('stage', stage_m=.08)},
            end_time_s=END_TIME_S, dt_max_s=DT_MAX_S, output_interval_s=OUTPUT_DT_S,
            cfl=.4, scheme=6, bedload=2, exchange=exchange, manning=1. / 36.,
            dry_tolerance_m=.00025, sediment_fractions=fractions(),
            active_layer_thickness_m=.04, substrate_layer_count=3,
            substrate_layer_thickness_m=.04, mpm_friction_mode=1,
            mpm_critical_shields=.047, bed_slope=2, hiding_exposure=4,
            secondary_flow=False, morphological_factor=1., compact_sidecars=True,
            source_sha256={'fixture': sha(__file__), 'base_geometry': sha(i2.__file__)})
        mesh.write_geometry(folder / 'geometry.npz')
        np.savez_compressed(folder / 'initial_analysis.npz', xy=xy, area=area, q=state,
            bed=bed, rock=rock, shoal=shoal, initial_concentration=INITIAL_CONCENTRATION,
            inlet_concentration=INLET_CONCENTRATION, settling_m_s=SETTLING_M_S,
            diameters_m=i2.DIAMETERS_M)
        cases.append(dict(name=variant, exchange_mode=exchange,
            input=str(folder / 'case.data'),
            input_files={str(path): sha(path) for path in folder.iterdir() if path.is_file()}))
    plan = dict(schema=1, tag=TAG, output=str(output), cases=cases, source_files=sources,
        native_executable=str(frozen_cpu.TIGHT_EXE), native_executable_sha256=frozen_cpu.TIGHT_SHA,
        native_runtime_build=str(frozen_cpu.TIGHT_RUNTIME / 'build.json'),
        native_runtime_build_sha256=sha(frozen_cpu.TIGHT_RUNTIME / 'build.json'),
        zero_step_export_runtime=str(runtime),
        zero_step_export_executable_sha256=build['executable_sha256'],
        export_parent_executable_sha256=build['parent_executable_sha256'],
        environment=ENV, controls=dict(duration_s=END_TIME_S, dt_max_s=DT_MAX_S,
            output_dt_s=OUTPUT_DT_S, cfl=.4, morfac=1., grains=7, bed_layers=4,
            mpm_bedload=2, hiding_mode=4, exchange_modes=VARIANTS,
            initial_concentration_kg_m3=INITIAL_CONCENTRATION,
            inlet_concentration_kg_m3=INLET_CONCENTRATION,
            settling_velocity_m_s=SETTLING_M_S),
        intention='Matched unchanged native suspension references; no GPU solver pass',
        Delft3D_source_modified=False)
    save(output / 'plan.json', plan)
    return dict(plan=str(output / 'plan.json'), plan_sha256=sha(output / 'plan.json'),
                cases=len(cases), prepared_only=True)


def load_plan(output=ROOT):
    plan_path = Path(output).resolve() / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    require(plan['tag'] == TAG and plan['schema'] == 1 and len(plan['cases']) == 2,
            'Unexpected I3 plan')
    verify_file_map(plan['source_files'])
    require(sha(plan['native_executable']) == plan['native_executable_sha256'],
            'Native executable changed')
    require(sha(Path(plan['zero_step_export_runtime']) / 'xbraids_normal_export.exe') ==
            plan['zero_step_export_executable_sha256'], 'Exporter executable changed')
    tight_exporter.verify_build(Path(plan['zero_step_export_runtime']))
    for case in plan['cases']:
        verify_file_map(case['input_files'])
    return plan, sha(plan_path)


def environment():
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(ENV)
    return env


def execute_case(variant, output=ROOT):
    require(variant in VARIANTS, 'Unknown I3 variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    row = next(item for item in plan['cases'] if item['name'] == variant)
    deck = Path(row['input'])
    prepared, cpu = output / variant / 'prepared', output / variant / 'cpu'
    require(not prepared.exists() and not cpu.exists(), 'Fresh execution required')
    prepared.mkdir(parents=True)
    raw = prepared / 'raw_export'
    origin = exporter.copy_inputs(deck, raw)
    build = tight_exporter.verify_build(Path(plan['zero_step_export_runtime']))
    export_env = tight_exporter.environment(True, Path(plan['zero_step_export_runtime']))
    export_env.update(ENV)
    with (raw / 'initial.log').open('x', encoding='utf-8') as log:
        run = subprocess.run([build['executable'], deck.name], cwd=raw,
            env=export_env,
            stdout=log, stderr=subprocess.STDOUT, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    require(run.returncode == 0 and (raw / 'coupled_input.bin').is_file(),
            'Native zero-step export failed; see raw_export/initial.log')
    shutil.copy2(raw / 'coupled_input.bin', prepared / 'coupled_input.bin')
    data = exporter.read_snapshot(prepared)
    initial = np.load(deck.parent / 'initial_analysis.npz', allow_pickle=False)
    checks = dict(cells=int(data['nc']) == 256, grains=int(data['nfrac']) == 7,
        layers=int(data['nbed_layers']) == 4,
        exchange_mode=int(data['exchange_mode']) == row['exchange_mode'],
        suspended_active=bool(data['suspended_transport_active']),
        legacy_SSP33_integrator=int(data['cpu_time_integrator']) == EXPECTED_INTEGRATOR,
        native_outlet_extension=int(data['cpu_outlet_extension']) == EXPECTED_OUTLET_EXTENSION,
        initial_hydraulics_bitwise=np.array_equal(data['q'].T, initial['q'][:, :3]),
        initial_suspended_bitwise=np.array_equal(
            data['state'][data['suspended_indices']].T, initial['q'][:, 3:10]),
        initial_layers_bitwise=np.array_equal(data['layer_mass'],
            initial['q'][:, 10:].reshape(256, 4, 7).transpose(1, 2, 0)),
        concentration_bitwise=np.array_equal(data['fraction_concentration'][:, 0],
                                             INLET_CONCENTRATION),
        settling_bitwise=np.array_equal(data['ws_fraction'], SETTLING_M_S),
        nonzero_all_classes=bool(np.all(np.max(data['state'][data['suspended_indices']], axis=1) > 0.)),
        wet_dry_present=bool(np.any(data['q'][0] <= .0005) and np.any(data['q'][0] >= .002)))
    require(all(checks.values()), 'Native zero-step preparation differs: ' + str(checks))
    save(prepared / 'preparation_receipt.json', dict(schema=1, origin=origin,
        exporter_executable_sha256=build['executable_sha256'],
        exporter_parent_executable_sha256=build['parent_executable_sha256'],
        raw_normal_metadata_sha256=sha(raw / 'normal_metadata.bin'),
        raw_normal_metadata_is_not_a_runner_input=True,
        artifacts={'coupled_input.bin': sha(prepared / 'coupled_input.bin')},
        initialization_CPU_only=True, solver_steps_performed=0, checks=checks))
    np.savez_compressed(prepared / 'coupled_input.npz', **data)
    exporter.copy_inputs(deck, cpu)
    command = [plan['native_executable'], deck.name]
    started = time.perf_counter()
    with (cpu / 'native.log').open('x', encoding='utf-8') as log:
        run = subprocess.run(command, cwd=cpu, env=environment(),
            stdout=log, stderr=subprocess.STDOUT, timeout=240,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    elapsed = time.perf_counter() - started
    save(cpu / 'execution.json', dict(schema=1,
        status='completed' if run.returncode == 0 else 'failed', return_code=run.returncode,
        plan_sha256=plan_sha, command=command, environment=ENV,
        native_executable_sha256=sha(command[0]), elapsed_s=elapsed,
        timing_scope='Diagnostic elapsed time, not a benchmark',
        output_files={str(path): sha(path) for path in cpu.iterdir() if path.is_file()},
        Delft3D_source_modified=False, GPU_steps_performed=0))
    require(run.returncode == 0, 'Unchanged native CPU failed; see native.log')
    return dict(case=variant, status='completed', elapsed_s=elapsed,
                prepared=str(prepared), cpu=str(cpu))


def analyze_case(variant, output=ROOT):
    require(variant in VARIANTS, 'Unknown I3 variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    cpu = output / variant / 'cpu'
    execution = json.loads((cpu / 'execution.json').read_text(encoding='utf-8'))
    require(execution['status'] == 'completed' and execution['plan_sha256'] == plan_sha,
            'Native execution identity or status differs')
    verify_file_map(execution['output_files'])
    frames = read_frames(cpu / 'frames.txt')
    times = np.array([frame['time_s'] for frame in frames])
    require(np.allclose(times, np.arange(5) * OUTPUT_DT_S, rtol=0, atol=1e-10),
            'Native output clocks differ')
    state = np.stack([frame['q'] for frame in frames])
    ledger = np.stack([frame['ledger'] for frame in frames])
    require(state.shape == (5, 256, 38) and ledger.shape == (5, 2, 15),
            'Native seven-class suspension frame layout differs')
    data = exporter.read_snapshot(output / variant / 'prepared')
    area = data['area']
    suspended = state[:, :, 3:10]
    layers = state[:, :, 10:].reshape(5, 256, 4, 7)
    suspended_mass = suspended.transpose(0, 2, 1) @ area
    bed_mass = layers.sum(axis=2).transpose(0, 2, 1) @ area
    suspended_ledger = ledger[:, :, 1:8].sum(axis=1)
    bed_ledger = ledger[:, :, 8:15].sum(axis=1)
    class_residual = ((suspended_mass + bed_mass) - (suspended_mass[0] + bed_mass[0]) -
                      ((suspended_ledger + bed_ledger) - (suspended_ledger[0] + bed_ledger[0])))
    water = state[:, :, 0] @ area
    water_residual = (water - water[0]) - (ledger[:, :, 0].sum(axis=1) - ledger[0, :, 0].sum())
    depth = state[:, :, 0]
    wet0, dry0 = depth[0] >= .002, depth[0] <= .0005
    activity = dict(initial_suspended_kg=suspended_mass[0],
        final_suspended_kg=suspended_mass[-1],
        final_suspended_boundary_ledger_kg=suspended_ledger[-1],
        final_bedload_boundary_ledger_kg=bed_ledger[-1],
        maximum_class_conservation_residual_kg=np.max(np.abs(class_residual), axis=0),
        maximum_water_residual_m3=float(np.max(np.abs(water_residual))),
        minimum_suspended_kg_m2=float(suspended.min()),
        minimum_bed_layer_kg_m2=float(layers.min()),
        wet_to_dry_cells=int(np.count_nonzero(wet0 & np.any(depth[1:] <= .0005, axis=0))),
        dry_to_wet_cells=int(np.count_nonzero(dry0 & np.any(depth[1:] >= .002, axis=0))),
        native_steps=int(frames[-1]['steps']),
        rejected_steps=int(frames[-1]['rejected_steps']))
    activity['basic_native_gates_passed'] = bool(
        activity['minimum_suspended_kg_m2'] >= -1e-12 and
        activity['minimum_bed_layer_kg_m2'] >= -1e-12 and
        activity['maximum_water_residual_m3'] <= 1e-9 and
        max(activity['maximum_class_conservation_residual_kg']) <= 1e-7 and
        np.all(np.max(np.abs(suspended_ledger), axis=0) > 1e-14) and
        activity['wet_to_dry_cells'] > 0 and activity['dry_to_wet_cells'] > 0)
    result = dict(schema=1, case=variant, plan_sha256=plan_sha,
        input_case_sha256=sha(next(item['input'] for item in plan['cases'] if item['name'] == variant)),
        native_frames_sha256=sha(cpu / 'frames.txt'),
        frame_times_s=times, activity=activity,
        CPU_executable_sha256=execution['native_executable_sha256'],
        GPU_solver_executed=False, physical_validation=False)
    save(output / variant / 'analysis.json', result)
    require(activity['basic_native_gates_passed'], 'Native suspension basic gates failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--case', choices=tuple(VARIANTS))
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.output)
    else:
        if args.case is None:
            parser.error('--case is required for run/analyze')
        result = execute_case(args.case, args.output) if args.action == 'run' else analyze_case(args.case, args.output)
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.tolist()))


if __name__ == '__main__':
    main()
