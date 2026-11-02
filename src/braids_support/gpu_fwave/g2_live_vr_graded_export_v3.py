"""Schema-1 zero-step export for the frozen seven-class live native pair.

The earlier v2 receipt used schema 2, which the public runner reserves for
bank sidecars. This creates new write-once prepared folders without changing
the numerical native decks, reference, or original exports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from . import a1_combined_live_scheduled_native_case as a1
from . import g2_live_vr_graded_native as native
from . import g2_live_vr_graded_process as audit_helper
from . import normal_coupled_oracle as zero
from . import normal_outlet_cases as outlet


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prepare(variant, output=native.ROOT):
    require(variant in native.VARIANTS, 'Unknown graded export variant')
    output = Path(output).resolve()
    plan, plan_sha = native.load_plan(output)
    audit_path = output / 'native_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    require(audit['passed'] and audit['plan_sha256'] == plan_sha and
            audit['audit_helper_sha256'] == native.sha(audit_helper.__file__),
            'Frozen graded native audit differs')
    built = zero.verify_build(native.EXPORT_RUNTIME)
    require(built['parent_executable_sha256'] == native.frozen_cpu.TIGHT_SHA and
            built['executable_sha256'] == plan['zero_step_exporter_sha256'],
            'Zero-step exporter provenance differs')
    deck = Path(next(row['deck'] for row in plan['cases']
                     if row['variant'] == variant))
    folder = output / variant / 'prepared_gpu_ssp33_v3'
    require(not folder.exists(), 'Use a fresh schema-1 prepared-input folder')
    zero.cr.copy_inputs(deck, folder)
    env = zero.environment(True, native.EXPORT_RUNTIME)
    env['BRAIDS_TIME_INTEGRATOR'] = 'legacy'
    process = subprocess.run([built['executable'], deck.name], cwd=folder,
        env=env, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (folder / 'initial.log').write_text(process.stdout + process.stderr,
                                         encoding='utf-8')
    require(process.returncode == 0, 'Zero-step graded exporter failed')
    data = outlet.read_case(folder)
    control = native.gate()['controls']
    expected_rows = 39 if native.VARIANTS[variant] else 38
    checks = dict(cells=int(data['nc']) == 128,
        seven_classes=int(data['nfrac']) == 7,
        four_layers=int(data['nbed_layers']) == 4,
        state_layout=np.asarray(data['state']).shape == (expected_rows, 128),
        hidden_prognostic_state=(not native.VARIANTS[variant] or
            int(data['secondary_index']) == 38),
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
                                          native.inputs()['q'][:, 10:38]),
        secondary_variant=bool(data['secondary_flow']) == native.VARIANTS[variant] and
            int(data['secondary_flow_model']) ==
                (2 if native.VARIANTS[variant] else 0),
        finite=np.isfinite(data['state']).all())
    require(all(checks.values()), 'Schema-1 zero-step export differs: ' + str(checks))
    np.savez_compressed(folder / 'coupled_input.npz', **data)
    receipt = dict(schema=1, variant=variant, status='completed',
        plan_sha256=plan_sha, native_audit_sha256=native.sha(audit_path),
        gate_sha256=native.sha(native.GATE),
        exporter_sha256=built['executable_sha256'],
        parent_executable_sha256=native.frozen_cpu.TIGHT_SHA,
        audit_helper_sha256=native.sha(audit_helper.__file__),
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
    parser.add_argument('variant', choices=tuple(native.VARIANTS))
    parser.add_argument('--output', type=Path, default=native.ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.variant, args.output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
