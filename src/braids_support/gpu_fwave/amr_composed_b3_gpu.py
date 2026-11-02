"""Private post-coarsening composed CUDA AMR trajectory through 0.08 s."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from braids_qgis.core.results import read_frames

from . import amr_composed_b2_gpu as b2
from . import amr_composed_b1_gpu_v2 as b1
from . import amr_composed_native_fixture_v2 as case
from . import amr_hydro_native_fixture as base
from . import coupled_reference


ROOT = case.ROOT/'gpu_composed_b3_candidate_v1'
FINAL = case.ROOT/'amr_full_state_export_v2/final'


class ComposedB3(b2.ComposedB2):
    def __init__(self):
        super().__init__()
        self.uniform_steps = 0
        self.post_peaks = {key:self.cp.zeros(2, dtype=self.cp.float64)
            for key in ('bed_flux', 'suspended_flux')}
        self.post_secondary_flux = 0.
        self.post_exchange_ledger = np.zeros((2, 2), dtype=np.float64)
        self.post_source_integral = 0.

    def _euler(self, state, dt, time):
        if self.mesh_index == 0:
            return super()._euler(state, dt, time)
        cp = self.cp
        hydro = self.solver.stepper.hydro
        hydro.status.fill(0)
        hydro._prepare(state['q'], None)
        if int(hydro.status.item()):
            raise RuntimeError('Post-regrid hydraulic preparation rejected')
        result = self.solver.euler(state, dt, time)
        if bool(cp.any(result['status'] != 0)):
            raise RuntimeError('Post-regrid composed Euler stage rejected')
        cfl = float(cp.max(result['cfl']).item())
        if not np.isfinite(cfl) or cfl > .4+1e-12:
            raise RuntimeError(f'Post-regrid composed CFL rejected: {cfl}')
        self.maximum_cfl = max(self.maximum_cfl, cfl)
        for name, flux in (('bed_flux', self.solver.exner.buffers['flux']),
                           ('suspended_flux',
                            self.solver.suspended_advection.buffers['flux'])):
            values = cp.max(cp.abs(flux), axis=1)
            cp.maximum(self.post_peaks[name], values,
                       out=self.post_peaks[name])
            cp.maximum(self.peaks[name], values, out=self.peaks[name])
        peak = float(cp.max(cp.abs(self.solver.secondary.buffers['flux'])).item())
        self.post_secondary_flux = max(self.post_secondary_flux, peak)
        self.secondary_peaks['face_flux'] = max(
            self.secondary_peaks['face_flux'], peak)
        return self._copy(result['state']), cp.zeros((0, 2, 10),
                                                     dtype=cp.float64)

    def _uniform_step(self, dt, time, *, reject_after_second_half=False):
        cp = self.cp
        prior_state = self._copy(self.state)
        prior = (self.accepted_steps, self.uniform_steps,
                 self.level_steps.copy(), self.solver.euler_calls,
                 self.maximum_cfl, self.source_integral,
                 self.exchange_ledger.copy(), self.post_source_integral,
                 self.post_exchange_ledger.copy(), self.post_secondary_flux)
        peaks = {name:value.copy() for name, value in self.peaks.items()}
        post_peaks = {name:value.copy() for name, value in self.post_peaks.items()}
        secondary_peaks = self.secondary_peaks.copy()
        before = self._budgets(self.state)
        try:
            half, _, source0, exchange0 = self._rk(self.state, 1, dt/2., time)
            final, _, source1, exchange1 = self._rk(
                half, 1, dt/2., time+dt/2.)
            if reject_after_second_half:
                raise RuntimeError('injected post-second-half pre-commit rejection')
            after = self._budgets(final)
            water = after[0]-before[0]
            classes = after[1]-before[1]
            source = source0+source1
            secondary = after[2]-before[2]-source
            if (abs(water) > 1e-9 or np.max(np.abs(classes)) > 1e-8 or
                    abs(secondary) > 1e-8 or
                    bool(cp.any(final['q'][0] < -1e-12)) or
                    bool(cp.any(final['suspended'] < -1e-12)) or
                    bool(cp.any(final['layers'] < -1e-12))):
                raise RuntimeError('Post-regrid conservation/positivity rejected')
            for key in b1.KEYS:
                cp.copyto(self.state[key], final[key])
            transfer = exchange0+exchange1
            self.source_integral += source
            self.post_source_integral += source
            self.exchange_ledger[:, 0] += transfer
            self.exchange_ledger[:, 1] -= transfer
            self.post_exchange_ledger[:, 0] += transfer
            self.post_exchange_ledger[:, 1] -= transfer
            self.accepted_steps += 1
            self.uniform_steps += 1
            return dict(time_s=time+dt, route='uniform_level1_subcycled',
                water_residual_m3=water, class_residual_kg=classes.tolist(),
                secondary_source_residual=secondary,
                minimum_depth_m=float(cp.min(self.state['q'][0]).item()),
                maximum_cfl=self.maximum_cfl)
        except Exception:
            for key in b1.KEYS:
                cp.copyto(self.state[key], prior_state[key])
            (self.accepted_steps, self.uniform_steps, self.level_steps,
             self.solver.euler_calls, self.maximum_cfl, self.source_integral,
             self.exchange_ledger, self.post_source_integral,
             self.post_exchange_ledger, self.post_secondary_flux) = prior
            for name, value in peaks.items():
                cp.copyto(self.peaks[name], value)
            for name, value in post_peaks.items():
                cp.copyto(self.post_peaks[name], value)
            self.secondary_peaks = secondary_peaks
            raise

    def step(self, dt, time, *, reject_after_second_half=False):
        if self.mesh_index == 0:
            if reject_after_second_half:
                raise ValueError('Late uniform rejection requires post-regrid mesh')
            return super().step(dt, time)
        if (self.mesh_index != 3 or abs(dt-.002) > 1e-12 or
                abs(time-self.accepted_steps*.002) > 1e-12):
            raise ValueError('Only frozen post-regrid .002 s outer schedule')
        return self._uniform_step(dt, time,
            reject_after_second_half=reject_after_second_half)

    def regrid_evaluation(self, time, *, inject_failure=False):
        if time == .08:
            if self.mesh_index != 3 or self.regrid_evaluations != 3 or \
                    self.accepted_steps != 40 or inject_failure:
                raise ValueError('Native final regrid evaluation changed')
            self.regrid_evaluations += 1
            return None
        return super().regrid_evaluation(time, inject_failure=inject_failure)

    def persistent(self, time):
        result = super().persistent(time)
        result.update(uniform_steps=np.int64(self.uniform_steps),
            post_source_integral=np.float64(self.post_source_integral),
            post_exchange_ledger=self.post_exchange_ledger.copy(),
            post_secondary_flux=np.float64(self.post_secondary_flux))
        result.update({f'post_peak_{name}':self.cp.asnumpy(value)
                       for name, value in self.post_peaks.items()})
        return result

    def restore(self, archive):
        mesh_index = int(archive['mesh_index'])
        if mesh_index == 0:
            super().restore(archive)
        elif mesh_index == 3:
            cp = self.cp
            full = cp.asarray(archive['state'])
            if full.shape != (64, 10) or bool(archive['reflux_pending']):
                raise ValueError('Invalid accepted post-regrid B3 checkpoint')
            solver, data = self._post_solver(full)
            ids = np.flatnonzero(self.tree.node_level == 1).astype(np.int64)
            if not np.array_equal(ids, archive['node_ids']):
                raise ValueError('Post-regrid checkpoint leaf IDs changed')
            self.solver, self.state = solver, solver.state
            self.ids, self.n, self.levels = ids, 64, np.ones(64, dtype=np.int64)
            self.area = np.asarray(data['area']).copy()
            self.area_device = cp.asarray(self.area)
            self.index = {1:cp.arange(64), 2:cp.empty(0, dtype=cp.int64)}
            self.owner = cp.asarray(data['owner'])
            self.neighbour = cp.asarray(data['neighbour'])
            self.length = cp.asarray(data['length'])
            self.interface = cp.empty(0, dtype=cp.int64)
            self.reflux = None
            self.mesh_index = 3
            self.accepted_steps = int(archive['accepted_steps'])
            self.sync_steps = int(archive['sync_steps'])
            self.mixed_steps = int(archive['mixed_steps'])
            self.level_steps = archive['level_steps'].astype(int).tolist()
            self.solver.euler_calls = int(archive['euler_calls'])
            self.maximum_cfl = float(archive['maximum_cfl'])
            self.source_integral = float(archive['source_integral'])
            self.exchange_ledger = archive['exchange_ledger'].copy()
            self.secondary_peaks = json.loads(str(archive['secondary_peaks_json']))
            for name in self.peaks:
                cp.copyto(self.peaks[name], cp.asarray(archive[f'peak_{name}']))
            self.regrid_evaluations = int(archive['regrid_evaluations'])
            self.regrid_events = json.loads(str(archive['regrid_events_json']))
            self.euler_calls_completed = int(archive['euler_calls_completed'])
        else:
            raise ValueError('Unsupported B3 checkpoint mesh index')
        self.uniform_steps = int(archive['uniform_steps'])
        self.post_source_integral = float(archive['post_source_integral'])
        self.post_exchange_ledger = archive['post_exchange_ledger'].copy()
        self.post_secondary_flux = float(archive['post_secondary_flux'])
        for name in self.post_peaks:
            self.cp.copyto(self.post_peaks[name],
                           self.cp.asarray(archive[f'post_peak_{name}']))


def compare(model):
    if (model.mesh_index != 3 or not np.array_equal(model.ids,
            read_frames(case.ROOT/'cpu/frames.txt')[4]['node_ids'])):
        raise ValueError('Final native/GPU selected-leaf IDs differ')
    reference = coupled_reference.read_snapshot(FINAL)['state']
    actual = model.cp.asnumpy(model._full_for_n(model.state, 64)).T
    data = model.solver.data
    bulk = np.asarray(data['bulk_fraction'])
    rock = np.asarray(data['rock'])
    bed = lambda a: rock+np.sum(a[5:7]/bulk[:, None]+a[7:9]/bulk[:, None], axis=0)
    return dict(depth_max_abs_m=float(np.max(np.abs(actual[0]-reference[0]))),
        bed_max_abs_m=float(np.max(np.abs(bed(actual)-bed(reference)))),
        suspended_class_max_abs_kg_m2=np.max(np.abs(actual[3:5]-reference[3:5]),
                                            axis=1).tolist(),
        total_bed_class_max_abs_kg_m2=np.max(np.abs(
            actual[5:7]+actual[7:9]-reference[5:7]-reference[7:9]),
            axis=1).tolist(),
        secondary_hI_max_abs=float(np.max(np.abs(actual[9]-reference[9]))),
        dry_mask_equal=bool(np.array_equal(actual[0] <= .0001,
                                          reference[0] <= .0001)))


def run(output=ROOT):
    from . import amr_composed_b3_provenance as provenance
    lock = provenance.preflight()
    output = Path(output)
    if output.exists():
        raise FileExistsError('B3 readback is write-once')
    output.mkdir(parents=True)
    model = ComposedB3()
    rows = []
    error = None
    try:
        for outer in range(40):
            rows.append(model.step(.002, outer*.002))
            if (outer+1) % 10 == 0:
                model.regrid_evaluation(round((outer+1)*.002, 12))
    except Exception as exc:
        error = str(exc)
    metrics = compare(model) if error is None else None
    np.savez_compressed(output/'trajectory.npz', **model.persistent(
        model.accepted_steps*.002))
    failures = [error] if error else []
    if metrics is not None and (metrics['depth_max_abs_m'] > 2e-5 or
            metrics['bed_max_abs_m'] > 2e-5 or
            max(metrics['suspended_class_max_abs_kg_m2']) > 1e-5 or
            max(metrics['total_bed_class_max_abs_kg_m2']) > .02 or
            metrics['secondary_hI_max_abs'] > 1e-7 or
            not metrics['dry_mask_equal']):
        failures.append('Native/GPU final difference exceeds frozen B3 limits')
    if (model.accepted_steps != 40 or model.sync_steps != 1 or
            model.mixed_steps != 29 or model.uniform_steps != 10 or
            model.level_steps != [0, 78, 116] or model.mesh_index != 3 or
            model.regrid_evaluations != 4 or len(model.regrid_events) != 1 or
            model.euler_calls_completed != 525 or
            model.solver.euler_calls != 60):
        failures.append('Native post-regrid route/level schedule changed')
    if rows and (max(abs(row['water_residual_m3']) for row in rows) > 1e-9 or
            max(max(abs(x) for x in row['class_residual_kg']) for row in rows) > 1e-8 or
            max(abs(row['secondary_source_residual']) for row in rows) > 1e-8 or
            min(row['minimum_depth_m'] for row in rows) < -1e-12):
        failures.append('Accepted outer conservation/source/positivity failed')
    post_peaks = {key:model.cp.asnumpy(value).tolist()
                  for key, value in model.post_peaks.items()}
    if (any(x <= 1e-8 for pair in post_peaks.values() for x in pair) or
            model.post_secondary_flux <= 1e-12 or
            not np.any(model.post_exchange_ledger) or
            abs(model.post_source_integral) <= 1e-12):
        failures.append('Post-regrid composed physics inactive')
    record = dict(scope='Private full 0.08 s composed GPU AMR fixture only',
        rows=rows, metrics=metrics,
        accepted_steps=model.accepted_steps,
        level_steps=model.level_steps, uniform_steps=model.uniform_steps,
        mesh_index=model.mesh_index, regrid_events=model.regrid_events,
        post_regrid_flux_peaks=post_peaks,
        post_regrid_secondary_face_peak=model.post_secondary_flux,
        post_regrid_exchange_ledger=model.post_exchange_ledger.tolist(),
        post_regrid_source_integral=model.post_source_integral,
        gate_B3_numerical_passed=not failures, failures=failures,
        source_preflight=lock,
        trajectory_sha256=base.sha(output/'trajectory.npz'),
        no_CPU_numerical_fallback=True, no_general_AMR_claim=True,
        Delft3D_modified=False, plugin_modified=False)
    base.save(output/'READBACK.json', record)
    if failures:
        raise RuntimeError('B3 numerical gate failed: '+'; '.join(failures))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
