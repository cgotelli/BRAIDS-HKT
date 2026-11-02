"""Private first-step CUDA composition of MPM, suspension and signed hI.

This uses the exact native-AMR initial WENO map on the matching 91 selected
leaves. It is the synchronous wet/dry *precursor* gate only: no L1/L2
subcycling, reflux, coarsening or mid-run AMR restart is claimed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import amr_composed_native_fixture_v2 as case
from . import amr_composed_one_step_native as native_case
from . import amr_hydro_native_fixture as base
from . import coupled_reference as exported
from . import coupled_solver


ROOT = case.ROOT / 'gpu_composed_one_step_candidate_v1'
SOURCE = case.ROOT / 'gpu_composed_source_v1'
INITIAL = case.ROOT / 'amr_full_state_export_v1/initial'
FIXED = native_case.ROOT / 'mesh_0'
FINAL = native_case.ROOT / 'amr_final'


def full(solver):
    cp = solver.cp
    n = solver.nc
    result = cp.empty((10, n), dtype=cp.float64)
    result[:3] = solver.state['q']
    result[3:5] = solver.state['suspended']
    result[5:7] = solver.state['layers'][0]
    result[7:9] = solver.state['layers'][1]
    result[9] = solver.state['secondary']
    return result


def accepted(solver):
    cp = solver.cp
    return dict(state=cp.asnumpy(full(solver)),
        ledgers={key: cp.asnumpy(value).copy() for key, value in
                 solver.ledger.items()})


def identical(left, right):
    return (np.array_equal(left['state'], right['state']) and
            left['ledgers'].keys() == right['ledgers'].keys() and
            all(np.array_equal(left['ledgers'][key], right['ledgers'][key])
                for key in left['ledgers']))


def model():
    data = exported.read_snapshot(FIXED)
    native = exported.read_snapshot(INITIAL)
    if (int(data['amr_enabled']) != 0 or int(native['amr_enabled']) != 1 or
            data['state'].shape != (10, 91) or
            not np.array_equal(data['state'], native['state'])):
        raise ValueError('Selected-leaf composed initial state differs from native AMR')
    for key in ('xy', 'area', 'owner', 'neighbour', 'length', 'normal',
                'distance', 'mid', 'hydraulic_ls_weight', 'manning_cell'):
        if not np.array_equal(data[key], native[key]):
            raise ValueError('Selected-leaf composed geometry changed: '+key)
    data['weno_cells'] = np.ascontiguousarray(native['weno_cells'].copy())
    data['cpu_time_integrator'] = 33
    solver = coupled_solver.CoupledSolver(data, source_folder=SOURCE,
                                           integrator='ssp33')
    if (solver.nb or solver.ng != 2 or not solver.has_suspended or
            solver.secondary.model != 2 or solver.suspended_exchange is None):
        raise ValueError('Composed closed MPM/suspension/hI scope was lost')
    return solver, data


def run(output=ROOT):
    from . import amr_composed_one_step_provenance as provenance
    lock = provenance.preflight()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Composed one-step GPU candidate is write-once')
    output.mkdir(parents=True)
    solver, data = model()
    before = accepted(solver)
    result = solver.step(.002, 0.)
    if int(result['status'].item()) != 0:
        raise RuntimeError('Composed CUDA SSPRK33 first step rejected')
    actual = solver.cp.asnumpy(full(solver))
    reference = exported.read_snapshot(FINAL)['state']
    area = np.asarray(data['area'])
    bulk = np.asarray(data['bulk_fraction'])
    bed_actual = np.asarray(data['rock'])+np.sum(
        actual[5:7]/bulk[:, None]+actual[7:9]/bulk[:, None], axis=0)
    bed_native = np.asarray(data['rock'])+np.sum(
        reference[5:7]/bulk[:, None]+reference[7:9]/bulk[:, None], axis=0)
    initial = before['state']
    total_initial = initial[3:5]+initial[5:7]+initial[7:9]
    total_final = actual[3:5]+actual[5:7]+actual[7:9]
    class_residual = np.sum(area[None, :]*(total_final-total_initial), axis=1)
    water_residual = float(np.sum(area*(actual[0]-initial[0])))
    source = float(solver.ledger['source'].item())
    secondary_residual = float(np.sum(area*(actual[9]-initial[9]))-source)
    cp = solver.cp
    bed_flux = cp.asnumpy(cp.max(cp.abs(solver.exner.buffers['flux']), axis=1))
    suspended_flux = cp.asnumpy(cp.max(cp.abs(
        solver.suspended_advection.buffers['flux']), axis=1))
    secondary_flux = float(cp.max(cp.abs(
        solver.secondary.buffers['flux'])).item())
    exchange = cp.asnumpy(solver.ledger['exchange'])

    # Force failure *after* the exchange kernels execute but before the global
    # commit. A rejected trial must not mutate accepted state/ledgers.
    rejected, _ = model()
    rejected_before = accepted(rejected)
    original_exchange = rejected.suspended_exchange.step

    def fail_late(*args, **kwargs):
        proposal = original_exchange(*args, **kwargs)
        proposal['status'].fill(1)
        return proposal

    rejected.suspended_exchange.step = fail_late
    late = rejected.step(.002, 0.)
    rejected.suspended_exchange.step = original_exchange
    rejection_fired = int(late['status'].item()) != 0
    rollback_equal = identical(rejected_before, accepted(rejected))
    retry = rejected.step(.002, 0.)
    retry_equal = int(retry['status'].item()) == 0 and \
        identical(accepted(solver), accepted(rejected))
    failures = []
    if not np.array_equal(actual[0] <= .0001, reference[0] <= .0001) or \
            np.count_nonzero(actual[0] <= .0001):
        failures.append('Wet/dry mask differs from native')
    metrics = dict(depth_max_abs_m=float(np.max(np.abs(actual[0]-reference[0]))),
        bed_max_abs_m=float(np.max(np.abs(bed_actual-bed_native))),
        suspended_class_max_abs_kg_m2=np.max(np.abs(actual[3:5]-reference[3:5]),
                                            axis=1).tolist(),
        total_bed_class_max_abs_kg_m2=np.max(np.abs(
            actual[5:7]+actual[7:9]-reference[5:7]-reference[7:9]),
            axis=1).tolist(),
        secondary_hI_max_abs=float(np.max(np.abs(actual[9]-reference[9]))),
        water_residual_m3=water_residual,
        total_class_residual_kg=class_residual.tolist(),
        secondary_budget_residual=secondary_residual,
        signed_hI_min=float(np.min(actual[9])),
        signed_hI_max=float(np.max(actual[9])),
        cfl=float(result['cfl'].item()))
    if (metrics['depth_max_abs_m'] > 2.e-5 or
            metrics['bed_max_abs_m'] > 2.e-5 or
            max(metrics['suspended_class_max_abs_kg_m2']) > 1.e-5 or
            max(metrics['total_bed_class_max_abs_kg_m2']) > .02 or
            metrics['secondary_hI_max_abs'] > 1.e-7):
        failures.append('Native/GPU one-step difference exceeds pre-GPU gate')
    if (abs(water_residual) > 1.e-9 or
            max(np.abs(class_residual)) > 1.e-8 or
            abs(secondary_residual) > 1.e-8):
        failures.append('Water, class or hI source budget failed')
    if (np.any(bed_flux <= 1.e-8) or
            np.any(suspended_flux <= 1.e-8) or
            secondary_flux <= 1.e-12 or not np.any(exchange)):
        failures.append('Composed flux/exchange activity missing')
    if not (rejection_fired and rollback_equal and retry_equal):
        failures.append('Late exchange rejection/retry transaction failed')
    record = dict(scope='Private first synchronous wet/dry AMR step only',
        metrics=metrics, bed_flux_peak_kg_m_s=bed_flux.tolist(),
        suspended_face_flux_peak_kg_m_s=suspended_flux.tolist(),
        secondary_face_flux_peak=secondary_flux,
        class_exchange_ledger_kg=exchange.tolist(),
        late_rejection_fired=rejection_fired,
        accepted_rollback_bitwise=rollback_equal,
        retry_bitwise=retry_equal,
        frozen_gate_A_passed=not failures, failures=failures,
        native_plan_sha256=base.sha(native_case.ROOT / 'plan.json'),
        native_final_state_sha256=base.sha(FINAL / 'coupled_input.bin'),
        source_preflight=lock, no_CPU_numerical_fallback=True,
        no_AMR_subcycling_or_regrid_claim=True,
        Delft3D_modified=False, plugin_modified=False)
    base.save(output / 'READBACK.json', record)
    if failures:
        raise RuntimeError('Composed GPU AMR first-step gate failed: '+'; '.join(failures))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
