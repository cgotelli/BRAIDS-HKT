"""Unchanged-native seven-class/four-layer live Van Rijn composition reference.

The numerical executable and Delft3D are read-only. The gate is written before
native or GPU evolution; outputs and exported prepared inputs are write-once.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from braids_core.boundaries import cut_cell_io as writer
from braids_core.boundaries.cut_cell_io import CutSedimentBoundarySeries
from braids_qgis.core.results import read_frames
from . import a1_combined_live_scheduled_native_case as a1
from . import coupled_reference as exporter
from . import cpu_filter_cold20h as frozen_cpu
from . import g2_live_vr_native as g2
from . import i2_multilayer_native_case as i2
from . import i3_suspension_native_case as i3
from . import normal_coupled_oracle as zero
from . import normal_outlet_cases as outlet


HERE = Path(__file__).resolve().parent
TAG = 'g2_live_vr_graded_native_20261004_v1'
ROOT = HERE / 'runs' / TAG
GATE = HERE / 'reports' / TAG / 'GATE.json'
EXPORT_RUNTIME = HERE / 'runs/g2_live_suspension_export_20261004_v1/runtime'
VARIANTS = {'off': False, 'prognostic': True}
ENV = dict(g2.ENV)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return i2.sha(path)


def save(path, value):
    i2.save(path, value)


def gate():
    controls = json.loads(GATE.read_text(encoding='utf-8'))
    require(controls['schema'] == 1 and
            controls['status'] == 'frozen_before_native_or_GPU_evolution' and
            controls['controls']['fractions'] == 7 and
            controls['controls']['bed_layers'] == 4 and
            controls['controls']['exchange_mode'] == 2,
            'Seven-class live Van Rijn gate is absent or altered')
    return controls


def inputs():
    controls = gate()['controls']
    base = a1.fixture('wetdry')
    q = np.zeros((128, 3 + 7 + 4 * 7), dtype=np.float64)
    q[:, :3] = base['q'][:, :3]
    concentration = np.asarray(controls['initial_concentration_kg_m3'])
    layers = np.asarray(controls['layer_thickness_m'])
    composition = np.asarray(controls['layer_composition'])
    require(concentration.shape == (7,) and layers.shape == (4,) and
            composition.shape == (4, 7) and
            np.allclose(composition.sum(axis=1), 1., atol=1e-15, rtol=0) and
            np.array_equal(layers, i2.LAYER_THICKNESS_M) and
            np.array_equal(composition, i2.LAYER_COMPOSITION),
            'Frozen seven-class layer or concentration structure differs')
    q[:, 3:10] = q[:, 0, None] * concentration[None, :]
    q[:, 10:38] = np.broadcast_to((i2.BULK * layers[:, None] * composition).reshape(28),
                                  (128, 28))
    rock = base['bed'] - float(layers.sum())
    fractions = tuple(replace(fraction, roughness_mode='fixed')
                      for fraction in i3.fractions())
    require(np.array_equal([fraction.d50_m for fraction in fractions],
                           controls['grain_d50_m']) and
            np.array_equal([fraction.settling_velocity_m_s for fraction in fractions],
                           controls['settling_velocity_m_s']),
            'Frozen grain and settling controls differ')
    inlet_c = np.asarray(controls['inlet_concentration_kg_m3_by_class'])
    feed = np.asarray(controls['inlet_bed_feed_kg_s_by_class'])
    require(inlet_c.shape == feed.shape == (7, 4) and
            np.array_equal(controls['forcing_knots_s'], a1.KNOTS_S),
            'Frozen class schedule dimensions differ')
    schedule = {f'sand_{g}': CutSedimentBoundarySeries(
        tuple(a1.KNOTS_S), tuple(inlet_c[g]), tuple(feed[g])) for g in range(7)}
    inlet = replace(base['boundaries']['in'], fraction_sediment_series=schedule)
    return dict(base, q=q, rock=rock, fractions=fractions,
                boundaries={'in': inlet, 'out': base['boundaries']['out']})


def prepare(output=ROOT):
    output = Path(output).resolve()
    require(not output.exists(), 'Use a fresh seven-class native reference folder')
    frozen = gate()
    controls = frozen['controls']
    require(controls['cells'] == 128 and controls['scheme'] == 6 and
            controls['bedlaw'] == 2 and controls['hiding_exposure'] == 4 and
            controls['bed_slope'] == 2 and controls['mpm_friction_mode'] == 1 and
            controls['morfac'] == 1. and controls['duration_s'] == a1.END_S and
            controls['dt_max_s'] == a1.DT_MAX_S and
            controls['output_interval_s'] == a1.OUTPUT_S,
            'Frozen native controls differ from fixture')
    _, runtime = frozen_cpu.runtime_identity()
    modules = (writer, a1, exporter, frozen_cpu, g2, i2, i3, zero, outlet)
    sources = {str(Path(module.__file__).resolve()): sha(module.__file__)
               for module in modules}
    sources.update(runtime)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    sources[str(GATE.resolve())] = sha(GATE)
    i2.verify_file_map(sources)
    data = inputs()
    output.mkdir(parents=True)
    cases = []
    for name, secondary in VARIANTS.items():
        folder = output / 'inputs' / name
        folder.mkdir(parents=True)
        writer.write_cut_case(folder / 'case.data', data['mesh'], data['q'],
            data['rock'], data['boundaries'], sections={'in': data['section']},
            end_time_s=a1.END_S, dt_max_s=a1.DT_MAX_S,
            output_interval_s=a1.OUTPUT_S, cfl=.4, scheme=6, bedload=2,
            exchange=2, manning=.03, dry_tolerance_m=g2.DRY_M,
            mpm_friction_mode=1, mpm_critical_shields=.047,
            bed_slope=2, hiding_exposure=4,
            sediment_fractions=data['fractions'],
            active_layer_thickness_m=float(i2.LAYER_THICKNESS_M[0]),
            substrate_layer_count=3,
            substrate_layer_thickness_m=float(i2.LAYER_THICKNESS_M[1]),
            secondary_flow=secondary, secondary_flow_model='prognostic',
            secondary_flow_transport_direction=secondary,
            secondary_flow_momentum_beta=1. if secondary else 0.,
            secondary_flow_espir=1., morphological_factor=1.,
            compact_sidecars=True,
            source_sha256={'fixture': sources[str(Path(__file__).resolve())],
                           'gate': sources[str(GATE.resolve())]})
        data['mesh'].write_geometry(folder / 'geometry.npz')
        np.savez_compressed(folder / 'initial_analysis.npz', q=data['q'],
            bed=data['bed'], rock=data['rock'], xy=data['xy'],
            area=np.array([cell.area_m2 for cell in data['mesh'].cells]),
            grain_d50_m=controls['grain_d50_m'],
            layer_thickness_m=i2.LAYER_THICKNESS_M,
            layer_composition=i2.LAYER_COMPOSITION,
            inlet_concentration_kg_m3=controls['inlet_concentration_kg_m3_by_class'],
            initial_concentration_kg_m3=controls['initial_concentration_kg_m3'],
            inlet_bed_feed_kg_s=controls['inlet_bed_feed_kg_s_by_class'])
        cases.append(dict(variant=name, secondary=secondary,
            deck=str((folder / 'case.data').resolve()),
            files={str(path.resolve()): sha(path) for path in folder.iterdir()
                   if path.is_file()}))
    plan = dict(schema=1, tag=TAG, gate_sha256=sha(GATE), sources=sources,
        cases=cases, environment=ENV,
        native_executable=str(frozen_cpu.TIGHT_EXE.resolve()),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        zero_step_exporter=str((EXPORT_RUNTIME / 'xbraids_normal_export.exe').resolve()),
        zero_step_exporter_sha256=sha(EXPORT_RUNTIME / 'xbraids_normal_export.exe'),
        controls=controls, GPU_solver_executed=False,
        observational_validation=False, Delft3D_source_modified=False)
    save(output / 'plan.json', plan)
    return dict(plan=str(output / 'plan.json'), plan_sha256=sha(output / 'plan.json'),
                gate_sha256=sha(GATE))


def load_plan(output=ROOT):
    output = Path(output).resolve()
    plan = json.loads((output / 'plan.json').read_text(encoding='utf-8'))
    require(plan['schema'] == 1 and plan['tag'] == TAG and
            plan['gate_sha256'] == sha(GATE) and
            plan['native_executable_sha256'] == frozen_cpu.TIGHT_SHA and
            {row['variant'] for row in plan['cases']} == set(VARIANTS),
            'Frozen seven-class native plan differs')
    i2.verify_file_map(plan['sources'])
    for row in plan['cases']:
        i2.verify_file_map(row['files'])
    require(sha(plan['native_executable']) == frozen_cpu.TIGHT_SHA and
            sha(plan['zero_step_exporter']) == plan['zero_step_exporter_sha256'],
            'Unchanged native/export executable identity differs')
    return plan, sha(output / 'plan.json')


def execute(variant, output=ROOT):
    require(variant in VARIANTS, 'Unknown seven-class native variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    deck = Path(next(row['deck'] for row in plan['cases']
                     if row['variant'] == variant))
    cpu = output / variant / 'cpu'
    require(not cpu.exists(), 'Native evolution output must be fresh')
    exporter.copy_inputs(deck, cpu)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    env.update(ENV)
    began = time.perf_counter()
    with (cpu / 'native.log').open('x', encoding='utf-8') as log:
        process = subprocess.run([plan['native_executable'], deck.name],
            cwd=cpu, env=env, stdout=log, stderr=subprocess.STDOUT,
            timeout=240, creationflags=subprocess.CREATE_NO_WINDOW
            if os.name == 'nt' else 0)
    save(cpu / 'execution.json', dict(schema=1, variant=variant,
        status='completed' if process.returncode == 0 else 'failed',
        return_code=process.returncode, elapsed_s=time.perf_counter()-began,
        plan_sha256=plan_sha, native_executable_sha256=frozen_cpu.TIGHT_SHA,
        files={str(path.resolve()): sha(path) for path in cpu.iterdir()
               if path.is_file()}, Delft3D_source_modified=False))
    require(process.returncode == 0,
            'Unchanged native seven-class run failed; inspect native.log')
    return dict(variant=variant, return_code=process.returncode,
                native_folder=str(cpu))


def audit(output=ROOT):
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    frozen = gate()
    floor = frozen['native_acceptance']
    rows = {}
    for variant, secondary in VARIANTS.items():
        cpu = output / variant / 'cpu'
        execution = json.loads((cpu / 'execution.json').read_text(encoding='utf-8'))
        require(execution['status'] == 'completed' and
                execution['plan_sha256'] == plan_sha, 'Native execution differs')
        i2.verify_file_map(execution['files'])
        frames = read_frames(cpu / 'frames.txt')
        state = np.stack([frame['q'] for frame in frames])
        ledger = np.stack([frame['ledger'] for frame in frames])
        times = np.array([frame['time_s'] for frame in frames])
        initial = np.load(output / 'inputs' / variant / 'initial_analysis.npz',
                          allow_pickle=False)
        area = np.asarray(initial['area'])
        layers = state[:, :, 10:38].reshape(len(frames), 128, 4, 7)
        suspended = state[:, :, 3:10].transpose(0, 2, 1) @ area
        bed = layers.sum(axis=2).transpose(0, 2, 1) @ area
        water = state[:, :, 0] @ area
        water_boundary = ledger[:, :, 0].sum(axis=1)
        suspended_boundary = ledger[:, :, 1:8].sum(axis=1)
        bed_boundary = ledger[:, :, 8:15].sum(axis=1)
        water_error = water-water[0]-(water_boundary-water_boundary[0])
        class_error = suspended+bed-(suspended[0]+bed[0])-(
            suspended_boundary+bed_boundary-
            suspended_boundary[0]-bed_boundary[0])
        exchange = suspended-suspended[0]-(suspended_boundary-suspended_boundary[0])
        wet = np.sum(state[:, :, 0] > g2.DRY_M, axis=1)
        log = (cpu / 'native.log').read_text(encoding='utf-8')
        match = re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)', log)
        prog = None
        if secondary:
            raw = np.loadtxt(cpu / 'secondary_flow.txt').reshape(9, 128, 2, 10)
            hi = state[:, :, 0] * raw[:, :, 0, 6]
            prog = float(np.max(np.abs(hi)))
        checks = dict(
            nine_frames=len(frames) == frozen['controls']['frame_count'],
            frame_clock=np.allclose(times, a1.FRAME_TIMES, atol=1e-10, rtol=0),
            public_layout=state.shape == (9, 128, 39 if secondary else 38) and
                ledger.shape == (9, 2, 15),
            finite=np.isfinite(state).all() and np.isfinite(ledger).all(),
            nonnegative_depth=np.min(state[:, :, 0]) >= floor['minimum_allowed_depth_m'],
            nonnegative_sediment=np.min(state[:, :, 3:38]) >=
                floor['minimum_allowed_sediment_kg_m2'],
            active_wet_dry=int(wet.max()-wet.min()) >= floor['minimum_wet_count_increase'],
            every_class_inlet_feed=np.all(ledger[-1, 0, 8:15] >=
                floor['minimum_each_class_inlet_feed_kg']),
            every_class_suspended_inlet=np.all(ledger[-1, 0, 1:8] >=
                floor['minimum_each_class_suspended_inlet_kg']),
            class_zero_erosion=exchange[-1, 0] >=
                floor['minimum_class_zero_erosion_to_water_kg'],
            class_six_deposition=-exchange[-1, 6] >=
                floor['minimum_class_six_deposition_to_bed_kg'],
            substrate_exchange=np.max(np.abs(layers[-1, :, 1:, :] -
                layers[0, :, 1:, :])) >=
                floor['minimum_substrate_layer_mass_change_kg_m2'],
            water_budget=np.max(np.abs(water_error)) <=
                floor['maximum_water_budget_residual_m3'],
            each_class_budget=np.max(np.abs(class_error)) <=
                floor['maximum_each_class_budget_residual_kg'],
            steps=bool(match and int(match.group(1)) >= floor['minimum_accepted_steps']
                and int(match.group(2)) <= floor['maximum_rejected_steps']),
            prognostic_hI=(not secondary or prog >=
                floor['minimum_prognostic_hI_m2_s']))
        rows[variant] = dict(passed=all(checks.values()), checks=checks,
            accepted_steps=None if match is None else int(match.group(1)),
            rejected_steps=None if match is None else int(match.group(2)),
            wet_counts=wet.tolist(),
            maximum_water_budget_residual_m3=float(np.max(np.abs(water_error))),
            maximum_each_class_budget_residual_kg=np.max(
                np.abs(class_error), axis=0).tolist(),
            final_signed_exchange_to_water_kg=exchange[-1].tolist(),
            maximum_substrate_mass_change_kg_m2=float(np.max(np.abs(
                layers[-1, :, 1:, :] - layers[0, :, 1:, :]))),
            maximum_prognostic_hI_m2_s=prog,
            final_inlet_feed_kg=ledger[-1, 0, 8:15].tolist(),
            final_inlet_suspended_kg=ledger[-1, 0, 1:8].tolist(),
            frames_sha256=sha(cpu / 'frames.txt'))
    result = dict(schema=1, passed=all(row['passed'] for row in rows.values()),
        gate_sha256=sha(GATE), plan_sha256=plan_sha,
        native_executable_sha256=frozen_cpu.TIGHT_SHA, variants=rows,
        GPU_solver_executed=False, observational_validation=False,
        Delft3D_source_modified=False)
    target = output / 'native_audit.json'
    require(not target.exists(), 'Native audit is write-once')
    save(target, result)
    require(result['passed'], 'Native seven-class gate failed: ' +
            str({variant: row['checks'] for variant, row in rows.items()}))
    return result


def export_prepared(variant, output=ROOT):
    require(variant in VARIANTS, 'Unknown seven-class preparation variant')
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    audit_path = output / 'native_audit.json'
    passed = json.loads(audit_path.read_text(encoding='utf-8'))
    require(passed['passed'] and passed['plan_sha256'] == plan_sha,
            'Unchanged-native gate must pass before zero-step preparation')
    built = zero.verify_build(EXPORT_RUNTIME)
    require(built['parent_executable_sha256'] == frozen_cpu.TIGHT_SHA and
            built['executable_sha256'] == plan['zero_step_exporter_sha256'],
            'Zero-step exporter provenance differs')
    deck = Path(next(row['deck'] for row in plan['cases']
                     if row['variant'] == variant))
    folder = output / variant / 'prepared_gpu_ssp33_v1'
    require(not folder.exists(), 'Zero-step prepared input must be fresh')
    zero.cr.copy_inputs(deck, folder)
    env = zero.environment(True, EXPORT_RUNTIME)
    env['BRAIDS_TIME_INTEGRATOR'] = 'legacy'
    process = subprocess.run([built['executable'], deck.name], cwd=folder,
        env=env, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (folder / 'initial.log').write_text(process.stdout + process.stderr,
                                         encoding='utf-8')
    require(process.returncode == 0, 'Zero-step native exporter failed')
    data = outlet.read_case(folder)
    control = gate()['controls']
    checks = dict(cells=int(data['nc']) == 128,
        seven_classes=int(data['nfrac']) == 7,
        four_layers=int(data['nbed_layers']) == 4,
        state_layout=np.asarray(data['state']).shape ==
            (39 if VARIANTS[variant] else 38, 128),
        mode9_critical4=np.array_equal(data['mode'], [9, 4]),
        scheduled_Q=np.array_equal(data['hydro_series_q'][:4, 0], a1.Q_M3_S),
        class_schedules=np.array_equal(data['nfraction_series'][:, 0],
            np.full(7, 4)) and
            np.array_equal(data['fraction_series_concentration'][:4, :, 0],
                           np.asarray(control['inlet_concentration_kg_m3_by_class']).T) and
            np.array_equal(data['fraction_series_bed_input'][:4, :, 0],
                           np.asarray(control['inlet_bed_feed_kg_s_by_class']).T),
        exchange2=int(data['exchange_mode']) == 2,
        hiding4=int(data['hiding_exposure']) == 4,
        slope2=int(data['bed_slope']) == 2,
        native_SSPRK33=int(data['cpu_time_integrator']) == 33,
        no_outlet_extension=int(data['cpu_outlet_extension']) == 0,
        initial_layer_mass=np.array_equal(data['state'][10:38].T,
            inputs()['q'][:, 10:38]),
        finite=np.isfinite(data['state']).all())
    require(all(checks.values()), 'Zero-step seven-class export differs: ' + str(checks))
    np.savez_compressed(folder / 'coupled_input.npz', **data)
    receipt = dict(schema=1, variant=variant, status='completed',
        plan_sha256=plan_sha, native_audit_sha256=sha(audit_path),
        gate_sha256=sha(GATE), exporter_sha256=built['executable_sha256'],
        parent_executable_sha256=frozen_cpu.TIGHT_SHA,
        helper_sha256=sha(__file__), solver_steps_performed=0,
        GPU_solver_executed=False, checks=checks,
        artifacts={path.name: sha(path) for path in folder.iterdir() if path.is_file()},
        observational_validation=False, Delft3D_source_modified=False)
    save(folder / 'preparation_receipt.json', receipt)
    return dict(variant=variant, prepared=str(folder),
        receipt_sha256=sha(folder / 'preparation_receipt.json'),
        input_sha256=sha(folder / 'coupled_input.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run-native', 'audit', 'export'))
    parser.add_argument('--variant', choices=tuple(VARIANTS))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.output)
    elif args.action == 'audit':
        result = audit(args.output)
    else:
        if args.variant is None:
            parser.error(args.action + ' requires --variant')
        result = execute(args.variant, args.output) if args.action == 'run-native' else (
            export_prepared(args.variant, args.output))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
