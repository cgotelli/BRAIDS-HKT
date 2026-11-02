"""Audit/export helper for the frozen seven-class native input and outputs.

The first diagnostic reader incorrectly assumed two secondary rows per cell.
The unchanged native executable writes one row per *grain class*: seven here.
Only this diagnostic/parser helper is corrected; native decks, gate, source,
evolution outputs and executable remain frozen and unmodified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from braids_qgis.core.results import read_frames
from . import a1_combined_live_scheduled_native_case as a1
from . import g2_live_vr_graded_native as native
from . import i2_multilayer_native_case as i2
from . import normal_coupled_oracle as zero
from . import normal_outlet_cases as outlet


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit(output=native.ROOT):
    output = Path(output).resolve()
    plan, plan_sha = native.load_plan(output)
    frozen = native.gate()
    floor = frozen['native_acceptance']
    rows = {}
    for variant, secondary in native.VARIANTS.items():
        cpu = output / variant / 'cpu'
        execution = json.loads((cpu / 'execution.json').read_text(encoding='utf-8'))
        require(execution['status'] == 'completed' and
                execution['return_code'] == 0 and
                execution['plan_sha256'] == plan_sha and
                execution['native_executable_sha256'] == native.frozen_cpu.TIGHT_SHA,
                'Native execution differs from frozen plan')
        i2.verify_file_map(execution['files'])
        frames = read_frames(cpu / 'frames.txt')
        state = np.stack([frame['q'] for frame in frames])
        ledger = np.stack([frame['ledger'] for frame in frames])
        times = np.array([frame['time_s'] for frame in frames])
        with np.load(output / 'inputs' / variant / 'initial_analysis.npz',
                     allow_pickle=False) as initial:
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
        wet = np.sum(state[:, :, 0] > native.g2.DRY_M, axis=1)
        log = (cpu / 'native.log').read_text(encoding='utf-8')
        match = re.search(r'steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)', log)
        secondary_result = None
        if secondary:
            raw = np.loadtxt(cpu / 'secondary_flow.txt').reshape(9, 128, 7, 10)
            budget = np.loadtxt(cpu / 'secondary_flow_budget.txt')
            require(np.allclose(raw[:, :, :, 0], times[:, None, None],
                                atol=1e-10, rtol=0) and
                    np.allclose(budget[:, 0], times, atol=1e-10, rtol=0),
                    'Native seven-class secondary diagnostic clock differs')
            hi = state[:, :, 0] * raw[:, :, 0, 6]
            closure = budget[:, 1]-budget[:, 2]-budget[:, 3]-budget[:, 4]
            secondary_result = dict(maximum_hI_m2_s=float(np.max(np.abs(hi))),
                maximum_inventory_difference=float(np.max(np.abs(hi @ area-budget[:, 1]))),
                maximum_closure_error=float(np.max(np.abs(closure))),
                maximum_reported_error=float(np.max(np.abs(budget[:, 5]))),
                source_and_boundary_final=budget[-1, 3:5].tolist())
        substrate_change = float(np.max(np.abs(
            layers[-1, :, 1:, :] - layers[0, :, 1:, :])))
        expected_feed = np.asarray(frozen['controls']['inlet_bed_feed_kg_s_by_class'])
        feed_integral = np.array([[a1.integral_at(a1.KNOTS_S, expected_feed[g], t)
                                   for g in range(7)] for t in times])
        expected_q = np.array([a1.integral_at(a1.KNOTS_S, a1.Q_M3_S, t)
                               for t in times])
        checks = dict(
            nine_frames=len(frames) == frozen['controls']['frame_count'],
            frame_clock=bool(np.allclose(times, a1.FRAME_TIMES, atol=1e-10, rtol=0)),
            public_layout=state.shape == (9, 128, 38) and ledger.shape == (9, 2, 15),
            finite=bool(np.isfinite(state).all() and np.isfinite(ledger).all()),
            nonnegative_depth=bool(np.min(state[:, :, 0]) >=
                floor['minimum_allowed_depth_m']),
            nonnegative_sediment=bool(np.min(state[:, :, 3:38]) >=
                floor['minimum_allowed_sediment_kg_m2']),
            active_wet_dry=int(wet.max()-wet.min()) >= floor['minimum_wet_count_increase'],
            every_class_inlet_feed=bool(np.all(ledger[-1, 0, 8:15] >=
                floor['minimum_each_class_inlet_feed_kg'])),
            every_class_suspended_inlet=bool(np.all(ledger[-1, 0, 1:8] >=
                floor['minimum_each_class_suspended_inlet_kg'])),
            prescribed_Q_integral=bool(np.max(np.abs(ledger[:, 0, 0]-expected_q)) <= 1e-8),
            prescribed_feed_integral=bool(np.max(np.abs(ledger[:, 0, 8:15]-feed_integral)) <= 1e-8),
            class_zero_erosion=bool(exchange[-1, 0] >=
                floor['minimum_class_zero_erosion_to_water_kg']),
            class_six_deposition=bool(-exchange[-1, 6] >=
                floor['minimum_class_six_deposition_to_bed_kg']),
            substrate_exchange=substrate_change >=
                floor['minimum_substrate_layer_mass_change_kg_m2'],
            water_budget=bool(np.max(np.abs(water_error)) <=
                floor['maximum_water_budget_residual_m3']),
            each_class_budget=bool(np.max(np.abs(class_error)) <=
                floor['maximum_each_class_budget_residual_kg']),
            steps=bool(match and int(match.group(1)) >= floor['minimum_accepted_steps']
                and int(match.group(2)) <= floor['maximum_rejected_steps']),
            prognostic_hI=(not secondary or bool(secondary_result['maximum_hI_m2_s'] >=
                floor['minimum_prognostic_hI_m2_s'] and
                secondary_result['maximum_inventory_difference'] <= 1e-8 and
                secondary_result['maximum_closure_error'] <= 1e-8 and
                secondary_result['maximum_reported_error'] <= 1e-8)))
        rows[variant] = dict(passed=all(checks.values()), checks=checks,
            accepted_steps=None if match is None else int(match.group(1)),
            rejected_steps=None if match is None else int(match.group(2)),
            wet_counts=wet.tolist(),
            maximum_water_budget_residual_m3=float(np.max(np.abs(water_error))),
            maximum_each_class_budget_residual_kg=np.max(np.abs(class_error), axis=0).tolist(),
            final_signed_exchange_to_water_kg=exchange[-1].tolist(),
            maximum_substrate_mass_change_kg_m2=substrate_change,
            secondary=secondary_result,
            final_inlet_feed_kg=ledger[-1, 0, 8:15].tolist(),
            final_inlet_suspended_kg=ledger[-1, 0, 1:8].tolist(),
            maximum_Q_integral_error_m3=float(np.max(np.abs(ledger[:, 0, 0]-expected_q))),
            maximum_feed_integral_error_kg=float(np.max(np.abs(
                ledger[:, 0, 8:15]-feed_integral))),
            frames_sha256=native.sha(cpu / 'frames.txt'))
    result = dict(schema=1, passed=all(row['passed'] for row in rows.values()),
        gate_sha256=native.sha(native.GATE), plan_sha256=plan_sha,
        audit_helper_sha256=native.sha(__file__),
        native_executable_sha256=native.frozen_cpu.TIGHT_SHA,
        variants=rows, GPU_solver_executed=False,
        observational_validation=False, Delft3D_source_modified=False)
    target = output / 'native_audit.json'
    require(not target.exists(), 'Native seven-class audit is write-once')
    native.save(target, result)
    require(result['passed'], 'Native seven-class gate failed: ' +
            str({variant: row['checks'] for variant, row in rows.items()}))
    return result


def export_prepared(variant, output=native.ROOT):
    require(variant in native.VARIANTS, 'Unknown seven-class export variant')
    output = Path(output).resolve()
    plan, plan_sha = native.load_plan(output)
    audit_path = output / 'native_audit.json'
    passed = json.loads(audit_path.read_text(encoding='utf-8'))
    require(passed['passed'] and passed['plan_sha256'] == plan_sha and
            passed['audit_helper_sha256'] == native.sha(__file__),
            'Frozen seven-class native audit differs')
    built = zero.verify_build(native.EXPORT_RUNTIME)
    require(built['parent_executable_sha256'] == native.frozen_cpu.TIGHT_SHA and
            built['executable_sha256'] == plan['zero_step_exporter_sha256'],
            'Zero-step exporter provenance differs')
    deck = Path(next(row['deck'] for row in plan['cases']
                     if row['variant'] == variant))
    folder = output / variant / 'prepared_gpu_ssp33_v1'
    require(not folder.exists(), 'Prepared input must be fresh')
    zero.cr.copy_inputs(deck, folder)
    env = zero.environment(True, native.EXPORT_RUNTIME)
    env['BRAIDS_TIME_INTEGRATOR'] = 'legacy'
    process = subprocess.run([built['executable'], deck.name], cwd=folder,
        env=env, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (folder / 'initial.log').write_text(process.stdout + process.stderr,
                                         encoding='utf-8')
    require(process.returncode == 0, 'Zero-step seven-class exporter failed')
    data = outlet.read_case(folder)
    control = native.gate()['controls']
    expected_state = native.inputs()['q']
    checks = dict(cells=int(data['nc']) == 128,
        seven_classes=int(data['nfrac']) == 7,
        four_layers=int(data['nbed_layers']) == 4,
        state_layout=np.asarray(data['state']).shape == (38, 128),
        mode9_critical4=np.array_equal(data['mode'], [9, 4]),
        eight_faces_each=np.array_equal(np.diff(data['boundary_offset']), [8, 8]),
        surveyed_inlet=int(np.count_nonzero(data['sampled'])) == 8 and
            bool(data['production_section_geometry']),
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
                                          expected_state[:, 10:38]),
        secondary_variant=bool(data['secondary_flow']) == native.VARIANTS[variant] and
            int(data['secondary_flow_model']) ==
                (2 if native.VARIANTS[variant] else 0),
        finite=np.isfinite(data['state']).all())
    require(all(checks.values()), 'Zero-step seven-class export differs: ' + str(checks))
    np.savez_compressed(folder / 'coupled_input.npz', **data)
    receipt = dict(schema=1, variant=variant, status='completed',
        plan_sha256=plan_sha, native_audit_sha256=native.sha(audit_path),
        gate_sha256=native.sha(native.GATE),
        exporter_sha256=built['executable_sha256'],
        parent_executable_sha256=native.frozen_cpu.TIGHT_SHA,
        helper_sha256=native.sha(__file__), solver_steps_performed=0,
        GPU_solver_executed=False, checks=checks,
        artifacts={path.name: native.sha(path) for path in folder.iterdir() if path.is_file()},
        observational_validation=False, Delft3D_source_modified=False)
    native.save(folder / 'preparation_receipt.json', receipt)
    return dict(variant=variant, prepared=str(folder),
        receipt_sha256=native.sha(folder / 'preparation_receipt.json'),
        input_sha256=native.sha(folder / 'coupled_input.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'export'))
    parser.add_argument('--variant', choices=tuple(native.VARIANTS))
    parser.add_argument('--output', type=Path, default=native.ROOT)
    args = parser.parse_args()
    if args.action == 'audit':
        result = audit(args.output)
    else:
        if args.variant is None:
            parser.error('export requires --variant')
        result = export_prepared(args.variant, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
