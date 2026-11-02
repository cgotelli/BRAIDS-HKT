"""Private 10-row composed AMR subcycling experiment, ending before regrid.

This is a deliberately bounded validation adapter.  The CUDA numerical
operators are the locked Gate-A operators; Python only schedules native AMR
levels, copies resident device fields, and checks scalar acceptance gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import amr_composed_one_step_gpu as gate_a
from . import amr_composed_native_fixture_v2 as native_case
from . import amr_composed_b1_native as b1_native
from . import amr_hydro_native_fixture as base
from . import amr_mpm_wetdry_export as identity_export
from . import coupled_reference
from .amr_reflux import ResidentAMRReflux
from .amr_hydro_route import requires_synchronous_wet_front


ROOT = native_case.ROOT / 'gpu_composed_b1_candidate_v2'
REFERENCE = b1_native.ROOT
KEYS = ('q', 'suspended', 'layers', 'secondary')


class ComposedB1:
    def __init__(self):
        solver, data = gate_a.model()
        cp = solver.cp
        self.cp, self.solver, self.state = cp, solver, solver.state
        with np.load(identity_export.ROOT / 'mesh_0/leaf_identity.npz') as leaf:
            self.ids = leaf['node_ids'].copy()
            self.levels = leaf['levels'].copy()
        native_initial = coupled_reference.read_snapshot(gate_a.INITIAL)
        if (len(self.ids) != 91 or data['state'].shape != (10, 91) or
                not np.array_equal(data['state'], native_initial['state'])):
            raise ValueError('Frozen composed 91-leaf input/topology changed')
        self.n = len(self.ids)
        self.index = {level: cp.asarray(np.flatnonzero(self.levels == level))
                      for level in (1, 2)}
        self.area = np.asarray(data['area'], dtype=np.float64).copy()
        self.area_device = cp.asarray(self.area)
        self.length = cp.asarray(data['length'])
        owner = np.asarray(data['owner'], dtype=np.int64)
        neighbour = np.asarray(data['neighbour'], dtype=np.int64)
        self.owner, self.neighbour = cp.asarray(owner), cp.asarray(neighbour)
        interface = np.flatnonzero((neighbour >= 0) &
            (self.levels[owner] != self.levels[np.maximum(neighbour, 0)]))
        if len(interface) != 18:
            raise ValueError('Frozen eighteen-face AMR interface changed')
        self.interface = cp.asarray(interface)
        groups, offsets, local, sides = [], [0], [], []
        for coarse in np.flatnonzero(self.levels == 1):
            entries = [(j, 0 if owner[f] == coarse else 1)
                       for j, f in enumerate(interface)
                       if owner[f] == coarse or neighbour[f] == coarse]
            if entries:
                groups.append(int(coarse))
                for j, side in entries:
                    local.append(j)
                    sides.append(side)
                offsets.append(len(local))
        if len(local) != 18:
            raise ValueError('Interface group map is incomplete')
        self.reflux = ResidentAMRReflux(self._full(self.state), self.area,
            np.asarray(groups), np.asarray(offsets), np.asarray(local),
            np.asarray(sides), 18, grains=2, prognostic_secondary=True,
            source_path=gate_a.SOURCE / 'amr_reflux.cu')
        self.accepted_steps = 0
        self.level_steps = [0, 0, 0]
        self.sync_steps = 0
        self.mixed_steps = 0
        self.maximum_cfl = 0.
        self.source_integral = 0.
        self.exchange_ledger = np.zeros((2, 2), dtype=np.float64)
        self.peaks = {name: cp.zeros(2, dtype=cp.float64) for name in
            ('bed_flux', 'suspended_flux', 'coarse_bed', 'fine_bed',
             'bed_reflux', 'coarse_suspended', 'fine_suspended',
             'suspended_reflux')}
        self.secondary_peaks = {name: 0. for name in
            ('face_flux', 'coarse_integral', 'fine_integral', 'reflux_delta')}

    def _copy(self, state):
        return {key: state[key].copy() for key in KEYS}

    def _full(self, state):
        cp = self.cp
        full = cp.empty((self.n, 10), dtype=cp.float64)
        full[:, :3] = state['q'].T
        full[:, 3:5] = state['suspended'].T
        full[:, 5:7] = state['layers'][0].T
        full[:, 7:9] = state['layers'][1].T
        full[:, 9] = state['secondary']
        return cp.ascontiguousarray(full)

    def _unpack(self, full):
        cp = self.cp
        return dict(q=cp.ascontiguousarray(full[:, :3].T),
            suspended=cp.ascontiguousarray(full[:, 3:5].T),
            layers=cp.ascontiguousarray(cp.stack((full[:, 5:7].T,
                                                  full[:, 7:9].T), axis=0)),
            secondary=cp.ascontiguousarray(full[:, 9]))

    def _mask(self, before, after, level):
        result = self._copy(before)
        ix = self.index[level]
        result['q'][:, ix] = after['q'][:, ix]
        result['suspended'][:, ix] = after['suspended'][:, ix]
        result['layers'][:, :, ix] = after['layers'][:, :, ix]
        result['secondary'][ix] = after['secondary'][ix]
        return result

    def _combine(self, a, b, wa, wb):
        return {key: wa*a[key]+wb*b[key] for key in KEYS}

    def _ghost(self, state, t, start, dt, old, new):
        if old is None:
            return
        alpha = min(1., max(0., (t-start)/dt))
        ix = self.index[1]
        for key in ('q', 'suspended'):
            state[key][:, ix] = (1.-alpha)*old[key][:, ix]+alpha*new[key][:, ix]
        state['layers'][:, :, ix] = ((1.-alpha)*old['layers'][:, :, ix]
                                      + alpha*new['layers'][:, :, ix])
        state['secondary'][ix] = ((1.-alpha)*old['secondary'][ix]
                                   + alpha*new['secondary'][ix])

    def _rebalance(self, state):
        status = self.solver.sediment.rebalance(state['layers'])
        if bool(self.cp.any(status != 0)):
            raise RuntimeError('AMR composed active-layer rebalance rejected')

    def _relax_level(self, state, dt, level):
        cp = self.cp
        before = state['secondary'].copy()
        _, status = self.solver.relax(state, dt)
        if bool(cp.any(status != 0)):
            raise RuntimeError('AMR composed secondary relaxation rejected')
        ix = self.index[level]
        source = float(cp.sum(self.area_device[ix]*
                              (state['secondary'][ix]-before[ix])).item())
        # Curvature is solved on the whole selected grid, but only the native
        # active level receives this half-step's hI source.
        inactive = self.index[2 if level == 1 else 1]
        state['secondary'][inactive] = before[inactive]
        return source

    def _exchange_level(self, state, dt, level):
        cp = self.cp
        before = self._copy(state)
        _, status = self.solver.exchange(state, dt)
        if bool(cp.any(status != 0)):
            raise RuntimeError('AMR composed suspended exchange rejected')
        ix = self.index[level]
        transfer = cp.asnumpy(cp.sum(self.area_device[ix][None, :]*
            (state['suspended'][:, ix]-before['suspended'][:, ix]), axis=1))
        inactive = self.index[2 if level == 1 else 1]
        state['suspended'][:, inactive] = before['suspended'][:, inactive]
        state['layers'][:, :, inactive] = before['layers'][:, :, inactive]
        return transfer

    def _manning(self, state, dt):
        hydro = self.solver.stepper.hydro
        hydro._launch('ti_manning', self.n,
            (state['q'], hydro.manning, np.int64(self.n), np.float64(dt),
             hydro.grav, hydro.dry, hydro.chezy_min, hydro.cell_status))
        if bool(self.cp.any(hydro.cell_status != 0)):
            raise RuntimeError('AMR composed Manning source rejected')

    def _euler(self, state, dt, time):
        cp = self.cp
        hydro = self.solver.stepper.hydro
        hydro.status.fill(0)
        hydro._prepare(state['q'], None)
        if int(hydro.status.item()):
            raise RuntimeError('AMR composed hydraulic preparation rejected')
        result = self.solver.euler(state, dt, time)
        if bool(cp.any(result['status'] != 0)):
            raise RuntimeError('AMR composed Euler stage rejected')
        cfl = float(cp.max(result['cfl']).item())
        if not np.isfinite(cfl) or cfl > .4+1.e-12:
            raise RuntimeError(f'AMR composed CFL rejected: {cfl}')
        self.maximum_cfl = max(self.maximum_cfl, cfl)
        rows = self.interface
        factor = dt*self.length[rows, None]
        integral = cp.zeros((18, 2, 10), dtype=cp.float64)
        fl, fr = self.solver.flux.buffers['fl'], self.solver.flux.buffers['fr']
        bed = self.solver.exner.buffers['flux']
        suspended = self.solver.suspended_advection.buffers['flux']
        secondary = self.solver.secondary.buffers['flux']
        integral[:, 0, :3] = factor*fl[:, rows].T
        integral[:, 1, :3] = factor*fr[:, rows].T
        for part, flux in ((slice(3, 5), suspended), (slice(5, 7), bed)):
            integral[:, 0, part] = factor*flux[:, rows].T
            integral[:, 1, part] = factor*flux[:, rows].T
        integral[:, :, 9] = factor*secondary[rows, None]
        for name, flux in (('bed_flux', bed), ('suspended_flux', suspended)):
            cp.maximum(self.peaks[name], cp.max(cp.abs(flux), axis=1),
                       out=self.peaks[name])
        self.secondary_peaks['face_flux'] = max(
            self.secondary_peaks['face_flux'],
            float(cp.max(cp.abs(secondary)).item()))
        return self._copy(result['state']), integral

    def _rk(self, state, level, dt, start, *, coarse_old=None,
            coarse_new=None, coarse_start=None, coarse_dt=None):
        cp = self.cp
        initial = self._copy(state)
        if level == 2:
            self._ghost(initial, start, coarse_start, coarse_dt,
                        coarse_old, coarse_new)
        source = self._relax_level(initial, .5*dt, level)
        next_a, fa = self._euler(initial, dt, start)
        a = self._mask(initial, next_a, level)
        if level == 2:
            self._ghost(a, start+dt, coarse_start, coarse_dt,
                        coarse_old, coarse_new)
        next_b, fb = self._euler(a, dt, start+dt)
        b = self._combine(initial, self._mask(a, next_b, level), .75, .25)
        self._rebalance(b)
        if level == 2:
            self._ghost(b, start+.5*dt, coarse_start, coarse_dt,
                        coarse_old, coarse_new)
        next_c, fc = self._euler(b, dt, start+.5*dt)
        final = self._combine(initial, self._mask(b, next_c, level), 1./3., 2./3.)
        self._rebalance(final)
        prior_q = final['q'].copy()
        self._manning(final, dt)
        inactive = self.index[2 if level == 1 else 1]
        final['q'][:, inactive] = prior_q[:, inactive]
        exchange = self._exchange_level(final, dt, level)
        self._rebalance(final)
        source += self._relax_level(final, .5*dt, level)
        updated = self._mask(state, final, level)
        self.level_steps[level] += 1
        return updated, cp.ascontiguousarray((fa+fb)/6.+(2./3.)*fc), source, exchange

    def _budgets(self, state):
        cp = self.cp
        area = self.area_device
        water = float(cp.sum(area*state['q'][0]).item())
        class_stock = state['suspended']+cp.sum(state['layers'], axis=0)
        classes = cp.asnumpy(cp.sum(class_stock*area[None, :], axis=1))
        secondary = float(cp.sum(area*state['secondary']).item())
        return water, classes, secondary

    def _mixed_half(self, dt, time, *, reject_after_reflux=False):
        cp = self.cp
        coarse_old = self._copy(self.state)
        coarse, cf, cs, ce = self._rk(self.state, 1, dt, time)
        fine0, ff0, fs0, fe0 = self._rk(coarse, 2, dt/2., time,
            coarse_old=coarse_old, coarse_new=coarse,
            coarse_start=time, coarse_dt=dt)
        fine1, ff1, fs1, fe1 = self._rk(fine0, 2, dt/2., time+dt/2.,
            coarse_old=coarse_old, coarse_new=coarse,
            coarse_start=time, coarse_dt=dt)
        ff = ff0+ff1
        for prefix, part in (('bed', slice(5, 7)),
                             ('suspended', slice(3, 5))):
            for name, value in ((f'coarse_{prefix}', cf[:, :, part]),
                                (f'fine_{prefix}', ff[:, :, part]),
                                (f'{prefix}_reflux', ff[:, :, part]-cf[:, :, part])):
                cp.maximum(self.peaks[name], cp.max(cp.abs(value), axis=(0, 1)),
                           out=self.peaks[name])
        for name, value in (('coarse_integral', cf[:, :, 9]),
                            ('fine_integral', ff[:, :, 9]),
                            ('reflux_delta', ff[:, :, 9]-cf[:, :, 9])):
            self.secondary_peaks[name] = max(self.secondary_peaks[name],
                                              float(cp.max(cp.abs(value)).item()))
        cp.copyto(self.reflux.accepted, self._full(fine1))
        trial, _, status = self.reflux.propose(cf,
            cp.stack((ff0, ff1), axis=0))
        if reject_after_reflux:
            raise RuntimeError('injected late post-reflux rejection')
        if bool(cp.any(status != 0)):
            raise RuntimeError('AMR composed reflux requires native phase repair')
        candidate = self._unpack(trial)
        self._rebalance(candidate)
        self.reflux.commit()
        for key in KEYS:
            cp.copyto(self.state[key], candidate[key])
        cp.copyto(self.reflux.accepted, self._full(self.state))
        self.source_integral += cs+fs0+fs1
        self.exchange_ledger[:, 0] += ce+fe0+fe1
        self.exchange_ledger[:, 1] -= ce+fe0+fe1

    def step(self, dt, time, *, reject_after_second_reflux=False):
        if abs(dt-.002) > 1.e-12 or abs(time-self.accepted_steps*.002) > 1.e-12:
            raise ValueError('Frozen B1 outer schedule only')
        cp = self.cp
        state = self._copy(self.state)
        reflux = self.reflux.accepted.copy()
        register = self.reflux.accepted_register.copy()
        counters = (self.accepted_steps, self.sync_steps, self.mixed_steps,
                    self.level_steps.copy(), self.solver.euler_calls,
                    self.maximum_cfl, self.source_integral,
                    self.exchange_ledger.copy())
        peaks = {key: value.copy() for key, value in self.peaks.items()}
        secondary_peaks = self.secondary_peaks.copy()
        before = self._budgets(self.state)
        try:
            if requires_synchronous_wet_front(self.state['q'], self.owner,
                                               self.neighbour):
                if self.accepted_steps != 0:
                    raise RuntimeError('Unexpected later synchronous AMR route')
                result = self.solver.step(dt, time)
                if int(result['status'].item()):
                    raise RuntimeError('AMR composed synchronous step rejected')
                self.maximum_cfl = max(self.maximum_cfl,
                                       float(result['cfl'].item()))
                self.source_integral += float(self.solver.ledger['source'].item())
                self.exchange_ledger += cp.asnumpy(self.solver.ledger['exchange'])
                cp.maximum(self.peaks['bed_flux'], cp.max(cp.abs(
                    self.solver.exner.buffers['flux']), axis=1),
                    out=self.peaks['bed_flux'])
                cp.maximum(self.peaks['suspended_flux'], cp.max(cp.abs(
                    self.solver.suspended_advection.buffers['flux']), axis=1),
                    out=self.peaks['suspended_flux'])
                self.sync_steps += 1
                route = 'synchronous_wetdry'
            else:
                self._mixed_half(dt/2., time)
                self._mixed_half(dt/2., time+dt/2.,
                                 reject_after_reflux=reject_after_second_reflux)
                self.mixed_steps += 1
                route = 'mixed_subcycled_reflux'
            after = self._budgets(self.state)
            water_residual = after[0]-before[0]
            class_residual = after[1]-before[1]
            source_step = self.source_integral-counters[6]
            secondary_residual = after[2]-before[2]-source_step
            if (abs(water_residual) > 1.e-9 or
                    np.max(np.abs(class_residual)) > 1.e-8 or
                    abs(secondary_residual) > 1.e-8 or
                    bool(cp.any(self.state['q'][0] < -1.e-12)) or
                    bool(cp.any(self.state['suspended'] < -1.e-12)) or
                    bool(cp.any(self.state['layers'] < -1.e-12))):
                raise RuntimeError('AMR composed outer conservation/positivity rejected')
            self.accepted_steps += 1
            return dict(time_s=time+dt, route=route,
                water_residual_m3=water_residual,
                class_residual_kg=class_residual.tolist(),
                secondary_source_residual=secondary_residual,
                minimum_depth_m=float(cp.min(self.state['q'][0]).item()),
                maximum_cfl=self.maximum_cfl)
        except Exception:
            if self.reflux.pending:
                self.reflux.rollback()
            for key in KEYS:
                cp.copyto(self.state[key], state[key])
            cp.copyto(self.reflux.accepted, reflux)
            cp.copyto(self.reflux.accepted_register, register)
            (self.accepted_steps, self.sync_steps, self.mixed_steps,
             self.level_steps, self.solver.euler_calls,
             self.maximum_cfl, self.source_integral,
             self.exchange_ledger) = counters
            for key, value in peaks.items():
                cp.copyto(self.peaks[key], value)
            self.secondary_peaks = secondary_peaks
            raise

    def persistent(self, time):
        cp = self.cp
        result = dict(state=cp.asnumpy(self._full(self.state)),
            reflux_accepted=cp.asnumpy(self.reflux.accepted),
            reflux_register=cp.asnumpy(self.reflux.accepted_register),
            reflux_pending=np.bool_(self.reflux.pending),
            accepted_steps=np.int64(self.accepted_steps),
            sync_steps=np.int64(self.sync_steps),
            mixed_steps=np.int64(self.mixed_steps),
            level_steps=np.asarray(self.level_steps, dtype=np.int64),
            euler_calls=np.int64(self.solver.euler_calls),
            maximum_cfl=np.float64(self.maximum_cfl),
            source_integral=np.float64(self.source_integral),
            exchange_ledger=self.exchange_ledger.copy(),
            secondary_peaks_json=np.asarray(json.dumps(self.secondary_peaks,
                                                        sort_keys=True)),
            time_s=np.float64(time), node_ids=self.ids.copy())
        result.update({f'peak_{name}': cp.asnumpy(value)
                       for name, value in self.peaks.items()})
        return result

    def restore(self, archive):
        cp = self.cp
        if not np.array_equal(archive['node_ids'], self.ids) or \
                bool(archive['reflux_pending']):
            raise ValueError('B1 checkpoint topology/register changed')
        state = self._unpack(cp.asarray(archive['state']))
        for key in KEYS:
            cp.copyto(self.state[key], state[key])
        cp.copyto(self.reflux.accepted, cp.asarray(archive['reflux_accepted']))
        cp.copyto(self.reflux.accepted_register,
                  cp.asarray(archive['reflux_register']))
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


def compare(model):
    native = coupled_reference.read_snapshot(REFERENCE / 'amr_final')['state']
    actual = model.cp.asnumpy(model._full(model.state)).T
    data = model.solver.data
    bulk = np.asarray(data['bulk_fraction'])
    rock = np.asarray(data['rock'])
    bed = lambda a: rock+np.sum(a[5:7]/bulk[:, None]+a[7:9]/bulk[:, None], axis=0)
    return dict(depth_max_abs_m=float(np.max(np.abs(actual[0]-native[0]))),
        bed_max_abs_m=float(np.max(np.abs(bed(actual)-bed(native)))),
        suspended_class_max_abs_kg_m2=np.max(np.abs(actual[3:5]-native[3:5]),
                                            axis=1).tolist(),
        total_bed_class_max_abs_kg_m2=np.max(np.abs(
            actual[5:7]+actual[7:9]-native[5:7]-native[7:9]), axis=1).tolist(),
        secondary_hI_max_abs=float(np.max(np.abs(actual[9]-native[9]))),
        dry_mask_equal=bool(np.array_equal(actual[0] <= .0001,
                                          native[0] <= .0001)))


def run(output=ROOT):
    from . import amr_composed_b1_provenance_v2 as provenance
    lock = provenance.preflight()
    output = Path(output)
    if output.exists():
        raise FileExistsError('B1 readback is write-once')
    output.mkdir(parents=True)
    model = ComposedB1()
    rows = []
    error = None
    try:
        for outer in range(10):
            rows.append(model.step(.002, outer*.002))
    except Exception as exc:
        error = str(exc)
    metrics = compare(model) if error is None else None
    np.savez_compressed(output/'trajectory.npz', **model.persistent(
        model.accepted_steps*.002))
    peaks = {key: model.cp.asnumpy(value).tolist()
             for key, value in model.peaks.items()}
    failures = []
    if error is not None:
        failures.append(error)
    if metrics is not None:
        if (metrics['depth_max_abs_m'] > 2e-5 or
                metrics['bed_max_abs_m'] > 2e-5 or
                max(metrics['suspended_class_max_abs_kg_m2']) > 1e-5 or
                max(metrics['total_bed_class_max_abs_kg_m2']) > .02 or
                metrics['secondary_hI_max_abs'] > 1e-7 or
                not metrics['dry_mask_equal']):
            failures.append('Native/GPU difference exceeds frozen B1 limits')
    record = dict(scope='Private ten-row active composed AMR prefix through 0.02 s',
        accepted_steps=model.accepted_steps, level_steps=model.level_steps,
        sync_steps=model.sync_steps, mixed_steps=model.mixed_steps,
        euler_calls=model.solver.euler_calls, rows=rows, metrics=metrics,
        peak_flux_and_integral=peaks,
        peak_secondary=model.secondary_peaks,
        exchange_ledger=model.exchange_ledger.tolist(),
        source_integral=model.source_integral,
        gate_B1_passed=not failures, failures=failures,
        source_preflight=lock,
        trajectory_sha256=base.sha(output/'trajectory.npz'),
        no_CPU_numerical_fallback=True, no_regrid_claim=True,
        Delft3D_modified=False, plugin_modified=False)
    base.save(output/'READBACK.json', record)
    if failures:
        raise RuntimeError('B1 gate failed: '+'; '.join(failures))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
