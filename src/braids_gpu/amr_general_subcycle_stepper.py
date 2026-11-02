"""Research recursive SSPRK33 GPU subcycling for general nonbank transport.

This module owns a private macro transaction and selected-cell advancement;
all fluxes, sources, state blends and reflux operate on CUDA.  It is *not* the
production route. Bedform roughness retains its established instantaneous
equilibrium timing: current-state capacity feedback in every Euler stage and a
prepared pre-friction evaluation in every level-local RK step. Horizontal
viscosity and general avalanche are applied once on the synchronized post-reflux
parent state, in fixed-grid source order: viscosity first, then avalanche.
The current Euler bridge evaluates the full graph but accepts only level-local
cells, so this is a numerical integration proof, not yet a sparse speedup.  A
shared scheduler supplies arbitrary-depth 2:1 clocks without owning physical
state or transaction rollback.
"""
from __future__ import annotations

import numpy as np

from .amr_general_stepper import SynchronousGeneralStepper
from .amr_general_subcycle_register import GeneralFaceRegister, require
from .amr_recursive_engine import RecursiveLevelScheduler


STATE = ('q', 'layers', 'secondary', 'suspended')


class RecursiveGeneralStepper:
    """One accepted model, arbitrary balanced 2:1 clocks and CUDA reflux."""

    route = 'research_recursive_general_subcycle'

    def __init__(self, prepared, topology, *, source_folder):
        require(prepared.schema == 3 and
                any(topology is item for item in prepared.topologies),
                'Authenticated general schema-3 topology required')
        data = topology.data
        bedform = bool(int(data['bedform_roughness']))
        modes = np.asarray(data['mode'])
        from .general_live_transport import recursive_bedform_envelope
        bedform_envelope = recursive_bedform_envelope(data)
        require(not int(data['bank_failure']) and
                not int(data['bank_toe_erosion_mode']) and
                int(data['bedlaw']) == 2 and
                (bedform_envelope or
                 (not bedform and int(data['exchange_mode']) == 2 and
                  int(data['secondary_flow_model']) == 2)),
                'Research subcycle admits either the exact bedform-only MPM '
                'envelope or MPM/exchange-2/prognostic-secondary with parent '
                'viscosity/avalanche, and excludes banks')
        if bedform_envelope:
            require(modes.shape == (2,) and sorted(map(int, modes)) == [1, 2],
                    'Recursive bedform pair requires one mode1 inlet and one '
                    'mode2 stage outlet')
        require(np.isfinite(float(data['horizontal_viscosity'])) and
                float(data['horizontal_viscosity']) >= 0. and
                (not float(data['horizontal_viscosity']) or
                 (int(data['horizontal_viscosity_mode']) == 1 and
                  int(data['horizontal_viscosity_geometry']) == 1)),
                'Research subcycle viscosity requires native mode1 geometry1')
        from .general_live_transport import (PARENT_AVALANCHE_ORDER,
                                              PARENT_VISCOSITY_ORDER,
                                              RECURSIVE_BEDFORM_ORDER)
        self.model = SynchronousGeneralStepper(prepared, topology,
            source_folder=source_folder,
            avalanche_placement=PARENT_AVALANCHE_ORDER,
            viscosity_placement=PARENT_VISCOSITY_ORDER,
            bedform_placement=RECURSIVE_BEDFORM_ORDER)
        self.cp = self.model.cp
        self.solver = self.model.solver
        self.data = data
        self.bedform_envelope = bedform_envelope
        self.bedform_order = (RECURSIVE_BEDFORM_ORDER if bedform else 'disabled')
        self.state_keys = tuple(key for key in STATE if key in self.solver.state)
        self.levels = prepared.tree['node_level'][topology.node_ids].astype(np.int32)
        require(self.levels.ndim == 1 and len(self.levels) == self.model.n and
                np.all(self.levels >= 0),
                'Research general subcycle requires nonnegative selected levels')
        self.register = GeneralFaceRegister(data, self.levels)
        self.root_level = int(np.min(self.levels))
        self.finest_level = int(np.max(self.levels))
        self.scheduled_levels = tuple(range(self.root_level, self.finest_level+1))
        self.index = {level:self.cp.asarray(np.flatnonzero(self.levels == level))
                      for level in self.scheduled_levels}
        self.inactive = {level:self.cp.asarray(np.flatnonzero(self.levels != level))
                         for level in self.scheduled_levels}
        owners = np.asarray(data['owner'])[np.asarray(data['boundary_faces'])]
        require(np.all(self.levels[owners] == self.finest_level),
                'Live physical boundary owners must be on the finest level')
        self.boundary_faces = [self.cp.asarray(np.flatnonzero(
            np.asarray(data['bid']) == boundary))
            for boundary in range(int(data['nb']))]
        self.area = self.register.area
        self.level_steps = np.zeros(self.finest_level+1, dtype=np.int64)
        self.bedform_stage_evaluations = np.zeros(
            self.finest_level+1, dtype=np.int64)
        self.bedform_friction_evaluations = np.zeros(
            self.finest_level+1, dtype=np.int64)
        self.epoch_origin = dict(accepted_steps=0, time_s=0., euler_calls=0)
        self.macro_cfl = 0.
        self.work = self._copy(self.solver.state)
        self.dense_old = self._copy(self.solver.state)
        self.dense_new = self._copy(self.solver.state)
        self.dense_times = np.zeros((len(self.scheduled_levels), 2), np.float64)
        self.level_integrals = [self.register.empty()
                                for _ in self.scheduled_levels]
        self.subtree_integrals = [self.register.empty()
                                  for _ in self.scheduled_levels]
        own_faces = [self.cp.asarray(self.register.layout[level]['own_faces'])
                     for level in self.scheduled_levels]
        self.scheduler = RecursiveLevelScheduler(
            level_has_work=[bool(self.index[level].size)
                            for level in self.scheduled_levels],
            own_faces=own_faces, level_integrals=self.level_integrals,
            subtree_integrals=self.subtree_integrals)
        self.accepted_integral = self.register.empty()
        self.last_exchange = self.cp.zeros((self.model.ng, 2), self.cp.float64)
        self.last_source = self.cp.zeros((), self.cp.float64)
        self.trial_exchange = self.cp.zeros_like(self.last_exchange)
        self.trial_source = self.cp.zeros_like(self.last_source)

    @property
    def prepared(self):
        return self.model.prepared

    @prepared.setter
    def prepared(self, value):
        self.model.prepared = value

    @property
    def topology(self):
        return self.model.topology

    @property
    def ids(self):
        return self.model.ids

    @property
    def source(self):
        return self.model.source

    @property
    def state(self):
        return self.model.state

    @property
    def n(self):
        return self.model.n

    @property
    def ns(self):
        return self.model.ns

    @property
    def ng(self):
        return self.model.ng

    @property
    def nl(self):
        return self.model.nl

    @property
    def accepted_steps(self):
        return self.model.accepted_steps

    @accepted_steps.setter
    def accepted_steps(self, value):
        self.model.accepted_steps = int(value)

    @property
    def time_s(self):
        return self.model.time_s

    @time_s.setter
    def time_s(self, value):
        self.model.time_s = float(value)

    @property
    def maximum_cfl(self):
        return self.model.maximum_cfl

    @maximum_cfl.setter
    def maximum_cfl(self, value):
        self.model.maximum_cfl = float(value)

    @property
    def last_root_integral(self):
        return self.accepted_integral

    def full(self):
        return self.model.full()

    def install(self, rows):
        return self.model.install(rows)

    def totals(self):
        return self.model.totals()

    def ledger_totals(self):
        return self.model.ledger_totals()

    def snapshot(self, *, independent=False):
        require(not self.register.pending,
                'Cannot snapshot a pending general subcycle reflux')
        return dict(model=self.model.snapshot(independent=independent),
            level_steps=self.level_steps.copy(),
            bedform_stage_evaluations=self.bedform_stage_evaluations.copy(),
            bedform_friction_evaluations=self.bedform_friction_evaluations.copy(),
            epoch_origin=dict(self.epoch_origin), macro_cfl=self.macro_cfl,
            dense_times=self.dense_times.copy(),
            work=self._copy(self.work), dense_old=self._copy(self.dense_old),
            dense_new=self._copy(self.dense_new),
            level_integrals=[value.copy() for value in self.level_integrals],
            subtree_integrals=[value.copy() for value in self.subtree_integrals],
            accepted_integral=self.accepted_integral.copy(),
            last_exchange=self.last_exchange.copy(),
            last_source=self.last_source.copy(),
            trial_exchange=self.trial_exchange.copy(),
            trial_source=self.trial_source.copy())

    def rollback(self, saved):
        if self.register.pending:
            self.register.rollback()
        self.model.rollback(saved['model'])
        np.copyto(self.level_steps, saved['level_steps'])
        np.copyto(self.bedform_stage_evaluations,
                  saved['bedform_stage_evaluations'])
        np.copyto(self.bedform_friction_evaluations,
                  saved['bedform_friction_evaluations'])
        self.epoch_origin = dict(saved['epoch_origin'])
        self.macro_cfl = saved['macro_cfl']
        np.copyto(self.dense_times, saved['dense_times'])
        for target_name in ('work', 'dense_old', 'dense_new'):
            target, source = getattr(self, target_name), saved[target_name]
            for key in self.state_keys:
                self.cp.copyto(target[key], source[key])
        for name in ('level_integrals', 'subtree_integrals'):
            for target, source in zip(getattr(self, name), saved[name], strict=True):
                self.cp.copyto(target, source)
        for name in ('accepted_integral', 'last_exchange', 'last_source',
                     'trial_exchange', 'trial_source'):
            self.cp.copyto(getattr(self, name), saved[name])

    def sync_accepted_clock(self):
        return self.model.sync_accepted_clock()

    def accepted_gradients(self):
        return self.model.accepted_gradients()

    def accepted_bed_gradient(self):
        return self.model.accepted_bed_gradient()

    def begin_topology_epoch(self, previous):
        """Carry cumulative clocks after the caller installs a conservative remap."""
        require(isinstance(previous, RecursiveGeneralStepper) and previous is not self and
                self.accepted_steps == 0 and self.time_s == 0. and
                self.solver.euler_calls == 0 and not self.register.pending and
                not previous.register.pending and self.ng == previous.ng and
                self.nl == previous.nl and self.ns == previous.ns and
                all(item.solver._glt_stage is None and
                    item.solver._glt_times is None for item in (self, previous)),
                'Invalid recursive general topology epoch handoff')
        self.epoch_origin = dict(accepted_steps=previous.accepted_steps,
            time_s=previous.time_s, euler_calls=previous.solver.euler_calls)
        self.accepted_steps = previous.accepted_steps
        self.time_s = previous.time_s
        self.maximum_cfl = previous.maximum_cfl
        self.solver.euler_calls = previous.solver.euler_calls
        self.level_steps.fill(0)
        self.bedform_stage_evaluations.fill(0)
        self.bedform_friction_evaluations.fill(0)
        self.accepted_integral.fill(0.)
        self.last_exchange.fill(0.)
        self.last_source.fill(0.)
        self.sync_accepted_clock()

    def frame(self):
        values = self.model.frame()
        values.update(recursive_level_steps=self.level_steps.copy(),
            recursive_epoch_start_steps=np.int64(self.epoch_origin['accepted_steps']),
            recursive_epoch_start_time=np.float64(self.epoch_origin['time_s']),
            recursive_epoch_start_euler_calls=np.int64(self.epoch_origin['euler_calls']),
            recursive_register=self.cp.asnumpy(self.accepted_integral))
        return values

    def _copy(self, state):
        return {key:state[key].copy() for key in self.state_keys}

    def _pack(self, state):
        cp, schema = self.cp, self.register.schema
        rows = cp.zeros((schema.nc, schema.ns), cp.float64)
        rows[:, :3] = state['q'].T
        if 'suspended' in state:
            rows[:, cp.asarray(schema.suspended)] = state['suspended'].T
        for layer, indices in enumerate(schema.bed):
            rows[:, cp.asarray(indices)] = state['layers'][layer].T
        if schema.secondary >= 0:
            rows[:, schema.secondary] = state['secondary']
        return cp.ascontiguousarray(rows)

    def _unpack(self, rows):
        cp, schema = self.cp, self.register.schema
        result = dict(q=cp.ascontiguousarray(rows[:, :3].T),
            layers=cp.ascontiguousarray(cp.stack([
                rows[:, cp.asarray(indices)].T for indices in schema.bed])),
            secondary=cp.ascontiguousarray(rows[:, schema.secondary]
                if schema.secondary >= 0 else cp.zeros(schema.nc, cp.float64)))
        if self.solver.has_suspended:
            result['suspended'] = cp.ascontiguousarray(
                rows[:, cp.asarray(schema.suspended)].T)
        return result

    def _mask(self, before, after, level):
        result = self._copy(before)
        ids = self.index[level]
        for key in self.state_keys:
            result[key][..., ids] = after[key][..., ids]
        return result

    def _ghost(self, state, current, local_level):
        for coarse_local in range(local_level):
            level = self.scheduled_levels[coarse_local]
            ids = self.index[level]
            if not int(ids.size):
                continue
            start, end = self.dense_times[coarse_local]
            alpha = (0. if end <= start else
                     min(1., max(0., (current-start)/(end-start))))
            for key in self.state_keys:
                state[key][..., ids] = ((1.-alpha)*self.dense_old[key][..., ids] +
                                        alpha*self.dense_new[key][..., ids])

    def _rebalance(self, state):
        flag = self.solver.sediment.rebalance(state['layers'])
        if bool(self.cp.any(flag)):
            raise RuntimeError('General subcycle active-layer rebalance rejected')

    def _relax(self, state, dt, level):
        cp, ids = self.cp, self.index[level]
        before = state['secondary'].copy()
        _, status = self.solver.relax(state, dt)
        if bool(cp.any(status)):
            raise RuntimeError('General subcycle secondary source rejected')
        source = cp.sum(self.area[ids]*(state['secondary'][ids]-before[ids]))
        inactive = self.inactive[level]
        state['secondary'][inactive] = before[inactive]
        return source

    def _exchange(self, state, dt, level):
        cp, ids = self.cp, self.index[level]
        if self.solver.suspended_exchange is None:
            return cp.zeros((self.model.ng, 2), cp.float64)
        old_suspended = state['suspended'].copy()
        old_layers = state['layers'].copy()
        _, status = self.solver.exchange(state, dt)
        if bool(cp.any(status)):
            raise RuntimeError('General subcycle suspended exchange rejected')
        delta = cp.sum(self.area[ids][None, :]*
            (state['suspended'][:, ids]-old_suspended[:, ids]), axis=1)
        inactive = self.inactive[level]
        state['suspended'][:, inactive] = old_suspended[:, inactive]
        state['layers'][:, :, inactive] = old_layers[:, :, inactive]
        return cp.stack((delta, -delta), axis=1)

    def _prepare(self, state, message):
        hydro = self.solver.stepper.hydro
        hydro.status.fill(0)
        hydro._prepare(state['q'], None)
        if bool(hydro.status.item()):
            raise RuntimeError(message)

    def _manning(self, state, dt, level):
        cp = self.cp
        hydro = self.solver.stepper.hydro
        before = state['q'].copy()
        self._prepare(state, 'General subcycle final hydraulic preparation rejected')
        roughness = hydro.manning
        if self.solver.bedform is not None:
            roughness, bedform_status = self.solver.friction_roughness(state)
            if bool(cp.any(bedform_status)):
                raise RuntimeError('General subcycle pre-friction bedform rejected')
            self.bedform_friction_evaluations[level] += 1
        hydro._launch('ti_manning', self.model.n,
            (state['q'], roughness, np.int64(self.model.n), np.float64(dt),
             hydro.grav, hydro.dry, hydro.chezy_min, hydro.cell_status))
        if bool(cp.any(hydro.cell_status)):
            raise RuntimeError('General subcycle Manning source rejected')
        inactive = self.inactive[level]
        state['q'][:, inactive] = before[:, inactive]

    def _euler(self, state, dt, current, level):
        cp, solver = self.cp, self.solver
        for key in self.state_keys:
            cp.copyto(solver.stepper.stage[key], state[key])
        hydro = solver.stepper.hydro
        hydro.status.fill(0)
        hydro._prepare(solver.stepper.stage['q'], None)
        if bool(hydro.status.item()):
            raise RuntimeError('General subcycle hydraulic stage preparation rejected')
        result = solver.euler(solver.stepper.stage, dt, current)
        if bool(cp.any(result['status'])):
            raise RuntimeError('General subcycle Euler CUDA status rejected')
        if solver.bedform is not None:
            self.bedform_stage_evaluations[level] += 1
        cfl = float(cp.max(result['cfl'][self.index[level]]).item())
        if not np.isfinite(cfl) or cfl > float(self.data['cfl_target'])+1e-12:
            raise RuntimeError('General subcycle active-level CFL rejected')
        self.maximum_cfl = max(self.maximum_cfl, cfl)
        self.macro_cfl = max(self.macro_cfl, cfl)
        integral = self.register.face_integral(solver, dt)
        return (self._copy(result['state']), integral,
                {name:value.copy() for name,value in result['ledger'].items()})

    def _rk(self, local_level, dt, current):
        cp, solver = self.cp, self.solver
        level = self.scheduled_levels[local_level]
        state = self.work
        solver.boundary.check_storage()
        solver.boundary.begin_trial()
        solver.schedule.evaluate(current, dt)
        solver._glt_stage = 0
        solver._glt_times = (float(current), float(current+dt),
                             float(current+.5*dt))
        try:
            initial = self._copy(state)
            self._ghost(initial, current, local_level)
            self._prepare(initial,
                          'General subcycle initial hydraulic preparation rejected')
            source = self._relax(initial, .5*dt, level)
            first, fa, _ = self._euler(initial, dt, current, level)
            a = self._mask(initial, first, level)
            self._ghost(a, current+dt, local_level)
            second, fb, _ = self._euler(a, dt, current+dt, level)
            b = {key: .75*initial[key]+.25*self._mask(a,second,level)[key]
                 for key in self.state_keys}
            self._rebalance(b)
            self._ghost(b, current+.5*dt, local_level)
            third, fc, _ = self._euler(b, dt, current+.5*dt, level)
            final = {key: initial[key]/3.+(2./3.)*self._mask(b,third,level)[key]
                     for key in self.state_keys}
            self._rebalance(final)
            self._manning(final, dt, level)
            exchange = self._exchange(final, dt, level)
            self._rebalance(final)
            source += self._relax(final, .5*dt, level)
            ids = self.index[level]
            for key in self.state_keys:
                self.dense_old[key][..., ids] = state[key][..., ids]
                self.dense_new[key][..., ids] = final[key][..., ids]
            self.dense_times[local_level] = current, current+dt
            self.work = self._mask(state, final, level)
            self.level_integrals[local_level][...] = self.register.rk33(fa, fb, fc)
            self.trial_source += source
            self.trial_exchange += exchange
            self.level_steps[level] += 1
            require(solver._glt_stage == 3,
                    'General subcycle live forcing omitted an RK stage')
        finally:
            solver._glt_stage = solver._glt_times = None

    def _reflux(self, local_level, coarse, fine):
        level = self.scheduled_levels[local_level]
        proposed = self.register.propose(self._pack(self.work), coarse, fine, level)
        try:
            work = self._unpack(proposed)
            self._rebalance(work)
            self.work = work
            self.register.commit()
        except BaseException:
            if self.register.pending:
                self.register.rollback()
            raise

    def _budget(self, state):
        cp, rows = self.cp, self._pack(state)
        schema = self.register.schema
        classes = schema.class_inventory(rows)
        secondary = (cp.sum(self.area*rows[:, schema.secondary])[None]
                     if schema.secondary >= 0 else cp.zeros(1, cp.float64))
        return cp.concatenate((cp.sum(self.area*rows[:, 0])[None],
            cp.sum(self.area[:, None]*classes, axis=0), secondary))

    def _boundary_ledger(self, integral):
        cp, schema = self.cp, self.register.schema
        nb = int(self.data['nb'])
        result = dict(water=cp.zeros(nb, cp.float64),
                      bed=cp.zeros((schema.ng, nb), cp.float64),
                      secondary=cp.zeros(nb, cp.float64))
        if self.solver.has_suspended:
            result['suspended'] = cp.zeros((schema.ng, nb), cp.float64)
        for boundary, faces in enumerate(self.boundary_faces):
            result['water'][boundary] = -cp.sum(integral[faces, 0, 0])
            result['bed'][:, boundary] = -cp.sum(
                integral[faces, 0][:, cp.asarray(schema.bed[0])], axis=0)
            if self.solver.has_suspended:
                result['suspended'][:, boundary] = -cp.sum(
                    integral[faces, 0][:, cp.asarray(schema.suspended)], axis=0)
            if schema.secondary >= 0:
                result['secondary'][boundary] = -cp.sum(
                    integral[faces, 0, schema.secondary])
        return result

    def step(self, dt, *, snapshot=None, reject_after_reflux=False,
              reject_after_viscosity=False, reject_after_avalanche=False,
              reject_after_active_toe=False, reject_after_bedform=False):
        cp, solver, model = self.cp, self.solver, self.model
        current = model.time_s
        require(not reject_after_active_toe,
                'General AMR stepper has no physical-toe rejection hook')
        require(np.isfinite(dt) and 0 < dt <= float(self.data['dtmax']) and
                current+dt <= float(self.data['endtime'])+1e-12,
                'General subcycle macro clock exceeds input')
        saved = self.snapshot() if snapshot is None else snapshot
        before = model.totals()-model.ledger_totals()
        try:
            self.macro_cfl = 0.
            self.bedform_stage_evaluations.fill(0)
            self.bedform_friction_evaluations.fill(0)
            for key in self.state_keys:
                cp.copyto(self.work[key], solver.state[key])
                cp.copyto(self.dense_old[key], solver.state[key])
                cp.copyto(self.dense_new[key], solver.state[key])
            self.dense_times.fill(current)
            self.trial_source.fill(0.)
            self.trial_exchange.fill(0.)
            solver.avalanche_trial_substeps = []
            root = self.scheduler.advance(float(dt), current,
                                          self._rk, self._reflux)
            if reject_after_bedform:
                raise RuntimeError('Injected general subcycle post-bedform rejection')
            if reject_after_reflux:
                raise RuntimeError('Injected general subcycle post-reflux rejection')
            work = self.work
            viscosity_calls = 0
            viscosity_substeps = 0
            viscosity_velocity_l2_retries = 0
            viscosity_minimum_correction = 1.
            viscosity_geometry_updates = 0
            viscosity_geometry_rebuilds = 0
            if solver.diffusion is not None:
                diffusion = solver.diffusion
                before_updates = int(diffusion.geometry_updates)
                before_rebuilds = (0 if diffusion.dynamic_geometry is None else
                                   int(diffusion.dynamic_geometry.rebuilds))
                status = solver.parent_post_reflux_viscosity(
                    work['q'], work['layers'], dt)
                if bool(status.item()):
                    raise RuntimeError('General subcycle parent-clock viscosity rejected')
                diagnostic = diffusion.last_result
                viscosity_calls = 1
                viscosity_substeps = int(diagnostic['substeps'])
                viscosity_velocity_l2_retries = int(diagnostic['energy_retries'])
                viscosity_minimum_correction = float(
                    diagnostic['minimum_correction'])
                viscosity_geometry_updates = (
                    int(diffusion.geometry_updates)-before_updates)
                rebuilds = (0 if diffusion.dynamic_geometry is None else
                            int(diffusion.dynamic_geometry.rebuilds))
                viscosity_geometry_rebuilds = rebuilds-before_rebuilds
                require(viscosity_substeps > 0 and
                        viscosity_velocity_l2_retries >= 0 and
                        np.isfinite(viscosity_minimum_correction) and
                        0. <= viscosity_minimum_correction <= 1. and
                        viscosity_geometry_updates in (0, 1) and
                        viscosity_geometry_rebuilds >= 0,
                        'General subcycle viscosity diagnostic differs')
            if reject_after_viscosity:
                raise RuntimeError('Injected general subcycle post-viscosity rejection')
            avalanche_substeps = 0
            if solver.general_avalanche_enabled:
                result = solver.parent_post_reflux_avalanche(
                    work['layers'], work['q'][0], dt)
                if bool(result['status'].item()):
                    raise RuntimeError('General subcycle parent-clock avalanche rejected')
                cp.copyto(work['layers'], result['layers'])
                self._rebalance(work)
                avalanche_substeps = int(result['substeps'])
                require(solver.avalanche_trial_substeps == [avalanche_substeps],
                        'General subcycle avalanche did not run exactly once')
            else:
                require(not solver.avalanche_trial_substeps,
                        'Disabled general subcycle avalanche recorded work')
            if reject_after_avalanche:
                raise RuntimeError('Injected general subcycle post-avalanche rejection')
            boundary = self._boundary_ledger(root)
            source, exchange = self.trial_source, self.trial_exchange
            candidate = self._budget(work)
            sediment_boundary = boundary['bed']
            if 'suspended' in boundary:
                sediment_boundary = sediment_boundary + boundary['suspended']
            supplied = cp.concatenate((cp.sum(boundary['water'])[None],
                cp.sum(sediment_boundary, axis=1),
                (cp.sum(boundary['secondary'])+source)[None]))
            residual = candidate-before-model.ledger_totals()-supplied
            if (not bool(cp.all(cp.isfinite(residual))) or
                    abs(float(residual[0].item())) > 1e-8 or
                    float(cp.max(cp.abs(residual[1:-1])).item()) > 1e-7 or
                    abs(float(residual[-1].item())) > 1e-8 or
                    bool(cp.any(work['q'][0] < 0)) or
                    bool(cp.any(work['layers'] < 0)) or
                    ('suspended' in work and bool(cp.any(work['suspended'] < 0))) or
                    float(cp.max(cp.abs(cp.sum(exchange, axis=1))).item()) > 1e-8):
                raise RuntimeError('General subcycle budget/positivity rejected')
            model.install(cp.ascontiguousarray(self._pack(work)))
            for key, value in boundary.items():
                solver.ledger[key] += value
            solver.ledger['source'] += source
            if 'exchange' in solver.ledger:
                solver.ledger['exchange'] += exchange
            model.accepted_steps += 1
            model.time_s += dt
            solver._glt_accepted_time = model.time_s
            cp.copyto(self.accepted_integral, root)
            cp.copyto(self.last_source, source)
            cp.copyto(self.last_exchange, exchange)
            interfaces = sum(int(self.register.layout[level]['faces'].size)
                             for level in self.scheduled_levels[:-1])
            bedform_stage = self.bedform_stage_evaluations.tolist()
            bedform_friction = self.bedform_friction_evaluations.tolist()
            bedform_calls = int(self.bedform_stage_evaluations.sum() +
                                self.bedform_friction_evaluations.sum())
            require((solver.bedform is None and bedform_calls == 0) or
                    (solver.bedform is not None and
                     self.bedform_stage_evaluations.sum() ==
                        3*self.bedform_friction_evaluations.sum() and
                     np.array_equal(self.bedform_friction_evaluations,
                                    np.asarray(self.level_steps)-
                                    np.asarray(saved['level_steps']))),
                    'General subcycle bedform evaluation schedule differs')
            return dict(time_s=model.time_s, accepted_steps=model.accepted_steps,
                step=model.accepted_steps,
                level_steps=self.level_steps.tolist(), euler_calls=solver.euler_calls,
                cfl=self.macro_cfl, macro_cfl=self.macro_cfl,
                maximum_cfl=self.maximum_cfl,
                viscosity_calls_per_macro=viscosity_calls,
                viscosity_substeps=viscosity_substeps,
                viscosity_velocity_l2_retries=viscosity_velocity_l2_retries,
                viscosity_minimum_correction=viscosity_minimum_correction,
                viscosity_geometry_updates=viscosity_geometry_updates,
                viscosity_geometry_rebuilds=viscosity_geometry_rebuilds,
                bedform_order=self.bedform_order,
                bedform_stage_evaluations_per_level=bedform_stage,
                bedform_friction_evaluations_per_level=bedform_friction,
                bedform_evaluations_per_macro=bedform_calls,
                avalanche_substeps=avalanche_substeps,
                residual=cp.asnumpy(residual).tolist(),
                budget_residual=cp.asnumpy(residual).tolist(),
                interface_faces=interfaces, route=self.route)
        except BaseException:
            self.rollback(saved)
            raise


class GeneralTwoLevelSubcycle(RecursiveGeneralStepper):
    """Compatibility wrapper for the original populated levels 1 and 2."""

    route = 'research_general_two_level_subcycle'

    def __init__(self, prepared, topology, *, source_folder):
        require(prepared.schema == 3 and
                any(topology is item for item in prepared.topologies),
                'Authenticated general schema-3 topology required')
        levels = prepared.tree['node_level'][topology.node_ids]
        require(set(map(int, levels)) == {1, 2},
                'Research general subcycle requires two populated 2:1 levels')
        super().__init__(prepared, topology, source_folder=source_folder)
