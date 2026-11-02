"""Small, immutable-input native references for graded GPU coupled qualification.

The two cases share an analytic wet/dry bar and a seven-class, four-layer bed.
Only the native (unchanged) CPU executable advances them.  The existing
I/O-only zero-step exporter creates the GPU-prepared input; it is not a GPU
simulation and this driver makes no GPU acceptance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction, write_cut_case
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_qgis.core.results import read_frames
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen_cpu
from . import curved_wetdry_readiness_v3 as analytic_bar
from . import normal_coupled_oracle as tight_exporter


HERE = Path(__file__).resolve().parent
TAG = 'i2_multilayer_native_20261003_v4'
ROOT = HERE / 'runs' / TAG
DIAMETERS_M = np.array([.00005, .00007, .00010, .00014, .00020, .00028, .00040])
GRAIN_COUNT = 7
LAYER_THICKNESS_M = np.array([.04, .04, .04, .08])
LAYER_COMPOSITION = np.array([
    [.03, .07, .12, .20, .26, .20, .12],
    [.08, .12, .16, .18, .18, .16, .12],
    [.15, .15, .15, .15, .15, .15, .10],
    [.25, .20, .15, .12, .10, .10, .08],
])
BULK = 2650. * (1. - .4)
END_TIME_S, DT_MAX_S, OUTPUT_DT_S = 1., .002, .25
ENV = dict(OMP_NUM_THREADS='8', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE',
           BRAIDS_TIME_INTEGRATOR='ssp43', BRAIDS_PROFILE='1')
VARIANTS = ('off', 'equilibrium')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    def plain(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, dict):
            return {key: plain(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(entry) for entry in item]
        return item
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, description):
    if not condition:
        raise ValueError(description)


def dependencies():
    from braids_core.boundaries import cut_cell_io, cut_cells, domain, inlet_io
    helpers = (Path(__file__).resolve(), Path(analytic_bar.__file__).resolve(),
               Path(exporter.__file__).resolve(), Path(frozen_cpu.__file__).resolve(),
               Path(tight_exporter.__file__).resolve(),
               *(Path(module.__file__).resolve() for module in
                 (cut_cell_io, cut_cells, domain, inlet_io)))
    return {str(path): sha(path) for path in helpers}


def mesh_and_initial():
    require(np.allclose(LAYER_COMPOSITION.sum(axis=1), 1., atol=1e-15, rtol=0),
            'Every layer composition must sum to one')
    require(len(set(DIAMETERS_M.tolist())) == GRAIN_COUNT and
            np.all(np.diff(DIAMETERS_M) > 0), 'Seven ordered distinct grain sizes required')
    domain = PolygonDomain('I2 graded wet/dry bar', [(0., 0.), (8., 0.), (8., 2.), (0., 2.)],
                           'EPSG:2056')
    assignments = [BoundaryAssignment(domain.edges[3].edge_id, 'in', 'inlet'),
                   BoundaryAssignment(domain.edges[1].edge_id, 'out', 'outlet')]
    mesh = map_cut_cells(domain, assignments, NativeGrid(32, 8, 0., 0., .25, .25))
    xy = np.array([cell.centroid_xy for cell in mesh.cells])
    area = np.array([cell.area_m2 for cell in mesh.cells])
    require(len(xy) == 256 and np.array_equal(area, np.full(256, .25**2)),
            'Only the uncut 256-cell Cartesian fixture is supported')
    original, bed, _, shoal = analytic_bar.initial_fields(xy)
    state = np.zeros((256, 3 + (1 + len(LAYER_THICKNESS_M)) * GRAIN_COUNT))
    state[:, :3] = original[:, :3]
    for layer, (thickness, composition) in enumerate(zip(LAYER_THICKNESS_M, LAYER_COMPOSITION)):
        begin = 3 + GRAIN_COUNT + layer * GRAIN_COUNT
        state[:, begin:begin + GRAIN_COUNT] = BULK * thickness * composition
    rock = bed - float(LAYER_THICKNESS_M.sum())
    require(np.all(state[:, 0] >= 0) and np.any(state[:, 0] <= .0005) and
            np.any(state[:, 0] >= .002), 'Initial wet and dry cells required')
    return mesh, xy, area, state, bed, rock, shoal


def fractions():
    return tuple(CutSedimentFraction(
        fraction_id=f'sand_{index}', d50_m=float(diameter), d90_m=float(1.5 * diameter),
        grain_density_kg_m3=2650., porosity=.4, settling_velocity_m_s=0.,
        z0_m=float(diameter / 30.), minimum_bedload_depth_m=.001,
        minimum_exchange_depth_m=.001, roughness_mode='manning',
        initial_volume_fraction=1. / GRAIN_COUNT,
        initial_active_volume_fraction=float(LAYER_COMPOSITION[0, index]))
        for index, diameter in enumerate(DIAMETERS_M))


def verify_file_map(mapping):
    for name, digest in mapping.items():
        require(Path(name).is_file() and sha(name) == digest,
                'Protected file identity differs: ' + name)


def prepare(output=ROOT):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a fresh output folder; existing fixture is immutable')
    tight, native_files = frozen_cpu.runtime_identity()
    export_build = tight_exporter.verify_build()
    runtime = Path(export_build['runtime'])
    require(export_build['parent_executable_sha256'] == frozen_cpu.TIGHT_SHA,
            'I/O-only exporter must derive from the exact native evolution runtime')
    required = dependencies()
    mesh, xy, area, state, bed, rock, shoal = mesh_and_initial()
    sources = {**required, **native_files,
               str(runtime / 'build.json'): sha(runtime / 'build.json'),
               str(runtime / 'xbraids_normal_export.exe'): export_build['executable_sha256']}
    verify_file_map(sources)
    output.mkdir(parents=True)
    cases = []
    for variant in VARIANTS:
        folder = output / 'inputs' / variant
        folder.mkdir(parents=True)
        equilibrium = variant == 'equilibrium'
        write_cut_case(folder / 'case.data', mesh, state, rock,
            {'in': CutBoundary('total_discharge', discharge_m3_s=.08),
             'out': CutBoundary('stage', stage_m=.08)},
            end_time_s=END_TIME_S, dt_max_s=DT_MAX_S, output_interval_s=OUTPUT_DT_S,
            cfl=.4, scheme=6, bedload=2, exchange=0, manning=1. / 36.,
            dry_tolerance_m=.00025, sediment_fractions=fractions(),
            active_layer_thickness_m=float(LAYER_THICKNESS_M[0]),
            substrate_layer_count=3, substrate_layer_thickness_m=float(LAYER_THICKNESS_M[1]),
            mpm_friction_mode=1, mpm_critical_shields=.047, bed_slope=2,
            hiding_exposure=4, secondary_flow=equilibrium,
            secondary_flow_model='equilibrium',
            secondary_flow_transport_direction=equilibrium,
            secondary_flow_momentum_beta=0., morphological_factor=1.,
            compact_sidecars=True, source_sha256={'fixture': sha(__file__),
                'analytic_bar': sha(analytic_bar.__file__)})
        mesh.write_geometry(folder / 'geometry.npz')
        np.savez_compressed(folder / 'initial_analysis.npz', xy=xy, area=area, q=state,
            bed=bed, rock=rock, shoal=shoal, diameters_m=DIAMETERS_M,
            layer_thickness_m=LAYER_THICKNESS_M, layer_composition=LAYER_COMPOSITION)
        cases.append(dict(name=variant, input=str(folder / 'case.data'), cells=len(state),
            input_files={str(path): sha(path) for path in folder.iterdir() if path.is_file()},
            secondary_model=1 if equilibrium else 0))
    plan = dict(schema=1, tag=TAG, output=str(output), cases=cases, source_files=sources,
        native_executable=str(frozen_cpu.TIGHT_EXE),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        native_runtime_build=str(frozen_cpu.TIGHT_RUNTIME / 'build.json'),
        native_runtime_build_sha256=sha(frozen_cpu.TIGHT_RUNTIME / 'build.json'),
        zero_step_export_runtime=str(runtime),
        zero_step_export_executable_sha256=export_build['executable_sha256'],
        export_parent_executable_sha256=export_build['parent_executable_sha256'],
        old_generic_exporter_20261002_v4_not_used=True,
        environment=ENV, controls=dict(duration_s=END_TIME_S, dt_max_s=DT_MAX_S,
            output_dt_s=OUTPUT_DT_S, cfl=.4, morfac=1., mpm=2, hiding_mode=4,
            grains=GRAIN_COUNT, grain_d50_m=DIAMETERS_M.tolist(),
            layers=4, layer_thickness_m=LAYER_THICKNESS_M.tolist(),
            secondary_models={'off': 0, 'equilibrium': 1}),
        intention='Native coupled bedload/Exner plus zero-step GPU input; no GPU solver pass',
        Delft3D_source_modified=False)
    save(output / 'plan.json', plan)
    return dict(plan=str(output / 'plan.json'), plan_sha256=sha(output / 'plan.json'),
                cases=len(cases), prepared_only=True)


def load_plan(output=ROOT):
    plan_path = Path(output).resolve() / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    require(plan['tag'] == TAG and plan['schema'] == 1 and len(plan['cases']) == 2,
            'Unexpected I2 plan')
    verify_file_map(plan['source_files'])
    require(sha(plan['native_executable']) == plan['native_executable_sha256'],
            'Native executable changed')
    require(sha(Path(plan['zero_step_export_runtime']) / 'xbraids_normal_export.exe') ==
            plan['zero_step_export_executable_sha256'], 'Exporter executable changed')
    tight_exporter.verify_build(Path(plan['zero_step_export_runtime']))
    for case in plan['cases']:
        verify_file_map(case['input_files'])
    return plan, sha(plan_path)


def _environment(extra=None):
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(ENV)
    if extra:
        env.update(extra)
    return env


def execute_case(variant, output=ROOT):
    require(variant in VARIANTS, 'Unknown I2 variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    row = next(item for item in plan['cases'] if item['name'] == variant)
    deck = Path(row['input'])
    prepared = output / variant / 'prepared'
    cpu = output / variant / 'cpu'
    require(not prepared.exists() and not cpu.exists(),
            'Fresh reference execution required; preserve existing outputs')
    prepared.mkdir(parents=True)
    raw = prepared / 'raw_export'
    origin = exporter.copy_inputs(deck, raw)
    built = tight_exporter.verify_build(Path(plan['zero_step_export_runtime']))
    command_export = [built['executable'], deck.name]
    with (raw / 'initial.log').open('x', encoding='utf-8') as log:
        process_export = subprocess.run(command_export, cwd=raw,
            env=tight_exporter.environment(True, Path(plan['zero_step_export_runtime'])),
            stdout=log, stderr=subprocess.STDOUT, timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    require(process_export.returncode == 0, 'CPU21b7-derived zero-step export failed; see raw_export/initial.log')
    require((raw / 'coupled_input.bin').is_file() and (raw / 'normal_metadata.bin').is_file(),
            'Expected two raw I/O-only export artifacts')
    # The normal-derived exporter always writes the extra mode-9 metadata.  It
    # is archived above, but this mode-6/mode-2 deck must follow the generic
    # runner path, which detects combined mode-9 from a direct-child metadata
    # artifact.  Copy ONLY its native coupled input, without modifying values.
    shutil.copy2(raw / 'coupled_input.bin', prepared / 'coupled_input.bin')
    data = exporter.read_snapshot(prepared)
    require(int(data['cpu_outlet_extension']) == 1 and int(data['cpu_time_integrator']) == 43,
            'Prepared control must match the unchanged CPU21b7 SSP43/outlet1 runtime')
    receipt = dict(schema=1, origin=origin, status='completed',
        exporter_executable_sha256=built['executable_sha256'],
        exporter_parent_executable_sha256=built['parent_executable_sha256'],
        raw_normal_metadata_sha256=sha(raw / 'normal_metadata.bin'),
        raw_normal_metadata_is_not_a_runner_input=True,
        artifacts={'coupled_input.bin': sha(prepared / 'coupled_input.bin')},
        initialization_CPU_only=True, solver_steps_performed=0,
        same_native_runtime_for_preparation_and_evolution=True,
        production_modified=False, delft_modified=False)
    save(prepared / 'preparation_receipt.json', receipt)
    np.savez_compressed(prepared / 'coupled_input.npz', **data)
    initial = np.load(deck.parent / 'initial_analysis.npz', allow_pickle=False)
    expected_layers = initial['q'][:, 3 + GRAIN_COUNT:].reshape(256, 4, GRAIN_COUNT).transpose(1, 2, 0)
    checks = dict(cells=int(data['nc']) == 256,
        grains=int(data['nfrac']) == GRAIN_COUNT,
        layers=int(data['nbed_layers']) == 4,
        substrate_layers=int(data['nsubstrate_layers']) == 3,
        secondary_model=int(data['secondary_flow_model']) == row['secondary_model'],
        hiding_mode=int(data['hiding_exposure']) == 4,
        SSP43_integrator=int(data['cpu_time_integrator']) == 43,
        native_outlet_extension=int(data['cpu_outlet_extension']) == 1,
        exchange_disabled=int(data['exchange_mode']) == 0,
        mpm_bedload=int(data['bedlaw']) == 2,
        initial_hydraulics_bitwise=np.array_equal(data['q'].T, initial['q'][:, :3]),
        initial_layers_bitwise=np.array_equal(data['layer_mass'], expected_layers),
        initial_rock_bitwise=np.array_equal(data['rock'], initial['rock']),
        all_suspended_zero=np.all(data['state'][data['suspended_indices']] == 0),
        all_ledgers_zero=np.all(data['ledger'] == 0),
        diameter_values=np.array_equal(data['d50_fraction'], DIAMETERS_M),
        wet_dry_present=bool(np.any(data['q'][0] <= .0005) and np.any(data['q'][0] >= .002)))
    require(all(checks.values()), 'Native zero-step preparation differs: ' + str(checks))
    exporter.copy_inputs(deck, cpu)
    command = [plan['native_executable'], deck.name]
    started = time.perf_counter()
    with (cpu / 'native.log').open('x', encoding='utf-8') as log:
        process = subprocess.run(command, cwd=cpu, env=_environment(),
            stdout=log, stderr=subprocess.STDOUT, timeout=240,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    elapsed = time.perf_counter() - started
    record = dict(schema=1, status='completed' if process.returncode == 0 else 'failed',
        return_code=process.returncode, plan_sha256=plan_sha,
        preparation_receipt_sha256=sha(prepared / 'preparation_receipt.json'),
        command=command, environment=ENV, native_executable_sha256=sha(command[0]),
        source_files=plan['source_files'], initial_checks=checks,
        elapsed_s=elapsed, timing_scope='Diagnostic elapsed time, not a benchmark',
        output_files={str(path): sha(path) for path in cpu.iterdir() if path.is_file()},
        Delft3D_source_modified=False, GPU_steps_performed=0)
    save(cpu / 'execution.json', record)
    require(process.returncode == 0, 'Unchanged native CPU failed; see native.log')
    return dict(case=variant, status=record['status'], elapsed_s=elapsed,
                prepared=str(prepared), cpu=str(cpu))


def analyze_case(variant, output=ROOT):
    require(variant in VARIANTS, 'Unknown I2 variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    row = next(item for item in plan['cases'] if item['name'] == variant)
    prepared = output / variant / 'prepared'
    cpu = output / variant / 'cpu'
    execution = json.loads((cpu / 'execution.json').read_text(encoding='utf-8'))
    require(execution['status'] == 'completed' and execution['plan_sha256'] == plan_sha,
            'Native execution identity or status differs')
    verify_file_map(execution['output_files'])
    require(sha(prepared / 'preparation_receipt.json') == execution['preparation_receipt_sha256'],
            'Zero-step receipt changed')
    data = exporter.read_snapshot(prepared)
    frames = read_frames(cpu / 'frames.txt')
    times = np.array([frame['time_s'] for frame in frames])
    expected_times = np.arange(5) * OUTPUT_DT_S
    require(np.allclose(times, expected_times, atol=1e-10, rtol=0),
            'Native output clocks differ from expected')
    state = np.stack([frame['q'] for frame in frames])
    ledger = np.stack([frame['ledger'] for frame in frames])
    require(state.shape == (5, 256, 38) and ledger.shape == (5, 2, 15),
            'Native seven-class four-layer frame layout differs')
    require(all(frame['nfrac'] == 7 and frame['bed_layers'] == 4 for frame in frames),
            'Native frames did not declare seven classes/four layers')
    require(np.isfinite(state).all() and np.isfinite(ledger).all(),
            'Nonfinite native state or ledger')
    layers = state[:, :, 3 + GRAIN_COUNT:].reshape(5, 256, 4, GRAIN_COUNT)
    area = data['area']
    sediment_mass = layers.sum(axis=2).transpose(0, 2, 1) @ area
    bed_ledger = ledger[:, :, 1 + GRAIN_COUNT:].sum(axis=1)
    sediment_residual = (sediment_mass - sediment_mass[0]) - (bed_ledger - bed_ledger[0])
    water_mass = state[:, :, 0] @ area
    water_ledger = ledger[:, :, 0].sum(axis=1)
    water_residual = (water_mass - water_mass[0]) - (water_ledger - water_ledger[0])
    bed = data['rock'][None, :] + (layers / data['bulk_fraction'][None, None, None, :]).sum(axis=(2, 3))
    depth = state[:, :, 0]
    wet0, dry0 = depth[0] >= .002, depth[0] <= .0005
    activity = dict(initial_wet_cells=int(wet0.sum()), initial_dry_cells=int(dry0.sum()),
        wet_to_dry_cells=int(np.count_nonzero(wet0 & np.any(depth[1:] <= .0005, axis=0))),
        dry_to_wet_cells=int(np.count_nonzero(dry0 & np.any(depth[1:] >= .002, axis=0))),
        maximum_bed_change_m=float(np.max(np.abs(bed - bed[0]))),
        final_bed_change_range_m=[float(np.min(bed[-1] - bed[0])),
                                  float(np.max(bed[-1] - bed[0]))],
        minimum_depth_m=float(depth.min()), minimum_layer_inventory_kg_m2=float(layers.min()),
        maximum_water_residual_m3=float(np.max(np.abs(water_residual))),
        maximum_each_grain_sediment_residual_kg=np.max(np.abs(sediment_residual), axis=0).tolist(),
        nonzero_bed_boundary_classes=int(np.count_nonzero(np.max(np.abs(bed_ledger), axis=0) > 1e-14)),
        final_sediment_boundary_ledger_kg=bed_ledger[-1].tolist(),
        native_steps=int(frames[-1]['steps']))
    activity['basic_native_gates_passed'] = bool(activity['minimum_depth_m'] >= -1e-12 and
        activity['minimum_layer_inventory_kg_m2'] >= -1e-12 and
        activity['maximum_water_residual_m3'] <= 1e-9 and
        max(activity['maximum_each_grain_sediment_residual_kg']) <= 1e-7 and
        activity['maximum_bed_change_m'] > 1e-8 and
        activity['initial_wet_cells'] > 0 and activity['initial_dry_cells'] > 0)
    from . import coupled_solver as gpu_solver
    try:
        gpu_solver.validate_scope(gpu_solver.normalized(data))
    except ValueError as error:
        guard = dict(host_scope_accepted=False, rejection=str(error))
    else:
        guard = dict(host_scope_accepted=True, still_not_a_GPU_execution=True)
    # Reveal the next fail-closed guard without relabeling or changing the
    # accepted input.  The generic runner has no authorized outlet1 adapter.
    diagnostic = dict(data, cpu_outlet_extension=0, outlet_extension=0)
    try:
        gpu_solver.validate_scope(gpu_solver.normalized(diagnostic))
    except ValueError as error:
        guard['next_rejection_after_diagnostic_outlet_zero_only'] = str(error)
    else:
        guard['next_rejection_after_diagnostic_outlet_zero_only'] = None
    guard['diagnostic_mutation_is_not_a_run_or_accepted_input'] = True
    guard['coupled_solver_sha256'] = sha(gpu_solver.__file__)
    result = dict(schema=1, case=variant, plan_sha256=plan_sha, activity=activity,
        frame_times_s=times.tolist(), frame_steps=[int(frame['steps']) for frame in frames],
        CPU_executable_sha256=execution['native_executable_sha256'],
        GPU_preparation_receipt_sha256=execution['preparation_receipt_sha256'],
        input_case_sha256=sha(row['input']), native_frames_sha256=sha(cpu / 'frames.txt'),
        native_boundary_ledger_sha256=sha(cpu / 'boundary_ledger.txt'),
        current_GPU_host_guard=guard, GPU_solver_executed=False, physical_validation=False)
    save(output / variant / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--case', choices=VARIANTS)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.output)
    else:
        if args.case is None:
            parser.error('--case is required for run/analyze')
        result = execute_case(args.case, args.output) if args.action == 'run' else analyze_case(args.case, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
