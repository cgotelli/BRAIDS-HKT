"""Synchronous general-transport GPU evolution on one authenticated AMR layout.

The selected operator supplies geometry and initial material only.  Accepted
hydraulic, suspended, bed-layer and secondary state remains resident on CUDA;
the caller owns mesh selection and conservative topology transitions.  This
route advances all selected cells on one clock (no level subcycling).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .a1_full_array_transaction import _require_disjoint_storage, _unique_arrays
from braids_support.gpu_fwave.a2_segmented_array_copy import SegmentedCopyPlan, signature


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SynchronousGeneralStepper:
    """One physical GPU solver per selected topology, with whole-array rollback."""

    def __init__(self, prepared, topology, *, source_folder,
                 avalanche_placement=None, viscosity_placement=None,
                 bedform_placement=None, integrator='ssp33',
                 cancelled=lambda: False):
        require(prepared.schema == 3 and any(topology is item for item in
                prepared.topologies),
                'General AMR stepper requires an authenticated schema-3 topology')
        data = topology.data
        require(int(data['bank_failure']) == 0 and
                int(data['bank_toe_erosion_mode']) == 0 and
                int(data['amr_enabled']) == 0,
                'General AMR transport cannot enter a bank or native AMR variant')
        self.prepared, self.topology = prepared, topology
        self.ids = topology.node_ids.copy()
        self.source = Path(source_folder).resolve()
        from .general_live_transport import (AVALANCHE_ORDER, BEDFORM_ORDER,
                                              VISCOSITY_ORDER,
                                              integration_identity)
        integration_identity(integrator, 'plain')
        placement = AVALANCHE_ORDER if avalanche_placement is None else avalanche_placement
        diffusion = VISCOSITY_ORDER if viscosity_placement is None else viscosity_placement
        bedform = BEDFORM_ORDER if bedform_placement is None else bedform_placement
        from braids_core.sediment.prognostic_secondary_nonzero_execution import (
            EXECUTION_KEY as NONZERO_EXECUTION_KEY,
        )
        live_v59 = (integrator == 'implicit_fwave_midpoint' and
                    NONZERO_EXECUTION_KEY in data)
        if live_v59:
            require(placement == AVALANCHE_ORDER and
                    diffusion == VISCOSITY_ORDER and
                    bedform == BEDFORM_ORDER,
                    'Q3-H3/v59 live selected route excludes reordered owners')
            from .amr_prognostic_nonzero_v59_route import constructors
            constructor = constructors(self.source, prepared, topology)
        else:
            from .general_live_transport import constructors
            constructor = constructors(
                self.source, data, integrator=integrator,
                avalanche_placement=placement,
                viscosity_placement=diffusion,
                bedform_placement=bedform)
        with constructor as Solver:
            self.solver = Solver(data, source_folder=self.source)
        require(callable(cancelled),
                'General AMR cancellation callback must be callable')
        self.cancelled = cancelled
        if integrator == 'implicit_fwave_midpoint':
            require(self.solver.implicit_midpoint is not None,
                    'M2-C route lost its implicit controller')
            self.solver.implicit_midpoint.bind_cancelled(cancelled)
        self.integrator = integrator
        self.prognostic_secondary_amr_live_v59 = getattr(
            self.solver, 'prognostic_secondary_amr_live_v59', None)
        self.cp = self.solver.cp
        self.state = self.solver.state
        self.n, self.ns = len(self.ids), int(data['nstate'])
        self.ng, self.nl = int(data['nfrac']), int(data['nbed_layers'])
        self.area = self.cp.asarray(data['area'], dtype=self.cp.float64)
        self.suspended_amr_contract = None
        self.wetdry_ownership = None
        self.wetdry_phase_ledger = self.cp.zeros((self.ng, 2),
                                                  dtype=self.cp.float64)
        self.last_wetdry_phase_change = self.cp.zeros(self.ng,
                                                       dtype=self.cp.float64)
        implicit_base = self.solver.implicit_midpoint
        if self.prognostic_secondary_amr_live_v59 is not None:
            implicit_base = implicit_base.base
        suspended_implicit = bool(
            integrator == 'implicit_fwave_midpoint' and
            implicit_base.suspended_composition)
        if suspended_implicit:
            from .implicit_suspension_amr import (
                ResidentSuspendedWetDryOwnership, contract_from_data)
            self.suspended_amr_contract = contract_from_data(data)
            self.wetdry_ownership = ResidentSuspendedWetDryOwnership(
                np.asarray(data['area']), self.ng,
                self.suspended_amr_contract.hydraulic_dry_tolerance_m,
                source_path=self.source/'implicit_suspension_amr.cu')
        self.accepted_steps, self.time_s, self.maximum_cfl = 0, 0., 0.
        self._copy_plan = None

    def full(self):
        """All conserved native state rows, in selected-cell order."""
        cp, data = self.cp, self.topology.data
        rows = cp.zeros((self.n, self.ns), dtype=cp.float64)
        rows[:, :3] = self.state['q'].T
        if self.solver.has_suspended:
            rows[:, cp.asarray(data['suspended_indices'])] = self.state['suspended'].T
        for layer, indices in enumerate(np.asarray(data['bed_indices'])):
            rows[:, cp.asarray(indices)] = self.state['layers'][layer].T
        if int(data['secondary_flow_model']) == 2:
            rows[:, int(data['secondary_index'])] = self.state['secondary']
        return rows

    def _owned_candidate(self, rows):
        """Return a disjoint candidate satisfying accepted dry ownership."""
        if self.wetdry_ownership is None:
            return rows, self.cp.zeros(self.ng, dtype=self.cp.float64)
        cp, data = self.cp, self.topology.data
        candidate = rows.copy()
        suspended = cp.asarray(data['suspended_indices'])
        active = cp.asarray(np.asarray(data['bed_indices'])[0])
        result = self.wetdry_ownership.propose(
            cp.ascontiguousarray(candidate[:, 0]),
            cp.ascontiguousarray(candidate[:, suspended].T),
            cp.ascontiguousarray(candidate[:, active].T))
        require(not bool(cp.any(result['status'] != 0)),
                'M2-C wet/dry ownership CUDA proposal rejected')
        candidate[:, suspended] = result['suspended'].T
        candidate[:, active] = result['active_bed'].T
        change = cp.sum(result['suspended_phase_change_kg'], axis=1)
        before = cp.sum(self.area[None, :]*(
            rows[:, suspended].T+rows[:, active].T), axis=1)
        after = cp.sum(self.area[None, :]*(
            candidate[:, suspended].T+candidate[:, active].T), axis=1)
        scale = cp.maximum(1.0, cp.maximum(cp.abs(before), cp.abs(after)))
        require(not bool(cp.any(cp.abs(after-before) > 1.e-12*scale)),
                'M2-C wet/dry ownership changed class inventory')
        return cp.ascontiguousarray(candidate), change

    def install(self, rows):
        """Install only a previously validated conservative candidate."""
        cp, data = self.cp, self.topology.data
        suspended = cp.asarray(data['suspended_indices'])
        bed = cp.asarray(np.asarray(data['bed_indices']).ravel())
        require(isinstance(rows, cp.ndarray) and rows.shape == (self.n, self.ns) and
                rows.dtype == cp.float64 and rows.flags.c_contiguous and
                bool(cp.all(cp.isfinite(rows))) and bool(cp.all(rows[:, 0] >= 0)) and
                bool(cp.all(rows[:, suspended] >= 0)) and
                bool(cp.all(rows[:, bed] >= 0)),
                'Invalid accepted general AMR state')
        rows, change = self._owned_candidate(rows)
        dry_tolerance = (float(self.suspended_amr_contract.
                               hydraulic_dry_tolerance_m)
                         if self.suspended_amr_contract is not None else 0.0)
        dry = rows[:, 0] <= dry_tolerance
        require(not bool(cp.any(rows[dry, 1:3])) and
                not bool(cp.any(rows[dry][:, suspended])) and
                (int(data['secondary_flow_model']) != 2 or
                 not bool(cp.any(rows[dry, int(data['secondary_index'])]))),
                'Dry accepted AMR state carries momentum, suspended mass or secondary flow')
        if not self.solver.has_suspended:
            require(not bool(cp.any(rows[:, suspended])),
                    'Inactive suspension has nonzero accepted inventory')
        cp.copyto(self.state['q'], cp.ascontiguousarray(rows[:, :3].T))
        if self.solver.has_suspended:
            cp.copyto(self.state['suspended'],
                      cp.ascontiguousarray(rows[:, suspended].T))
        for layer, indices in enumerate(np.asarray(data['bed_indices'])):
            cp.copyto(self.state['layers'][layer],
                      cp.ascontiguousarray(rows[:, cp.asarray(indices)].T))
        if int(data['secondary_flow_model']) == 2:
            cp.copyto(self.state['secondary'], rows[:, int(data['secondary_index'])])
        else:
            self.state['secondary'].fill(0.)
        self.wetdry_phase_ledger[:, 0] += change
        self.wetdry_phase_ledger[:, 1] -= change
        self.cp.copyto(self.last_wetdry_phase_change, change)

    def totals(self):
        """Water, total class mass (bed plus suspension), signed secondary."""
        cp = self.cp
        classes = cp.sum(self.state['layers'], axis=0)
        if self.solver.has_suspended:
            classes = classes + self.state['suspended']
        return cp.concatenate((cp.sum(self.area*self.state['q'][0])[None],
            cp.sum(self.area[None, :]*classes, axis=1),
            cp.sum(self.area*self.state['secondary'])[None]))

    def ledger_totals(self):
        cp, ledger = self.cp, self.solver.ledger
        classes = cp.sum(ledger['bed'], axis=1)
        if self.solver.has_suspended:
            classes = classes + cp.sum(ledger['suspended'], axis=1)
        return cp.concatenate((cp.sum(ledger['water'])[None], classes,
            (cp.sum(ledger['secondary'])+ledger['source'])[None]))

    def snapshot(self, *, independent=False):
        arrays = _unique_arrays(self.solver)
        current = signature(arrays)
        if independent:
            _require_disjoint_storage(arrays)
            plan = SegmentedCopyPlan(self.cp, arrays)
        elif self._copy_plan is None or self._copy_plan.signature != current:
            _require_disjoint_storage(arrays)
            self._copy_plan = SegmentedCopyPlan(self.cp, arrays)
            plan = self._copy_plan
        else:
            plan = self._copy_plan
        plan.snapshot()
        boundary = self.solver.boundary
        return dict(plan=plan, accepted_steps=self.accepted_steps,
            time_s=self.time_s, maximum_cfl=self.maximum_cfl,
            euler_calls=self.solver.euler_calls,
            wetdry_phase_ledger=self.wetdry_phase_ledger.copy(),
            last_wetdry_phase_change=self.last_wetdry_phase_change.copy(),
            diagnostics=self.solver._a1_trial_diagnostics(),
            boundary_stage_present=hasattr(boundary, '_g1_next_stage'),
            boundary_next_stage=getattr(boundary, '_g1_next_stage', None))

    def rollback(self, saved):
        # A rejected first wet/dry rebuild can replace diffusion-cache array
        # bindings. Restore those bindings before either complete byte plan is
        # replayed, so no byte is restored into a now-detached replacement.
        self.solver._a1_restore_diagnostics(saved['diagnostics'])
        saved['plan'].restore()
        self.accepted_steps = saved['accepted_steps']
        self.time_s = saved['time_s']
        self.maximum_cfl = saved['maximum_cfl']
        self.solver.euler_calls = saved['euler_calls']
        self.cp.copyto(self.wetdry_phase_ledger,
                       saved['wetdry_phase_ledger'])
        self.cp.copyto(self.last_wetdry_phase_change,
                       saved['last_wetdry_phase_change'])
        # The wrapped fixed-grid transaction uses these detached arrays only
        # to return a rejected ``solver.step`` result.  This owner restores a
        # snapshot directly, so retaining them would change the reachable CUDA
        # inventory after rollback and poison the next segmented-copy plan.
        if hasattr(self.solver, '_a1_trial_result'):
            del self.solver._a1_trial_result
        boundary = self.solver.boundary
        if saved['boundary_stage_present']:
            boundary._g1_next_stage = saved['boundary_next_stage']
        elif hasattr(boundary, '_g1_next_stage'):
            del boundary._g1_next_stage

    def sync_accepted_clock(self):
        require(self.solver._glt_stage is None and self.solver._glt_times is None,
                'Cannot reset general AMR clock during a trial')
        self.solver._glt_accepted_time = float(self.time_s)

    def step(self, dt, *, snapshot=None, reject_after_active_toe=False):
        require(not reject_after_active_toe,
                'General AMR stepper has no physical-toe rejection hook')
        require(np.isfinite(dt) and 0 < dt <= float(self.topology.data['dtmax']) and
                self.time_s+dt <= float(self.topology.data['endtime'])+1e-12,
                'Synchronous general AMR step exceeds its clock bounds')
        saved = self.snapshot() if snapshot is None else snapshot
        before = self.totals()-self.ledger_totals()
        cp = self.cp
        try:
            result = self.solver.step(dt, self.time_s)
            if bool(result.get('diagnostics', {}).get('cancelled', False)):
                raise InterruptedError('M2-C implicit CUDA step cancelled')
            cfl = float(cp.max(result['cfl']).item())
            if (bool(cp.any(result['status'])) or not np.isfinite(cfl) or
                    cfl > float(self.topology.data['cfl_target'])+1e-12):
                raise RuntimeError('Synchronous general AMR CUDA status/CFL rejected')
            # This is normally a zero transfer because the currently admitted
            # implicit hydraulic envelope remains safely wet.  Keeping the
            # actual CUDA ownership transaction in the accepted path prevents
            # a future wet/dry extension from stranding suspended inventory.
            if self.wetdry_ownership is not None:
                owned = self.wetdry_ownership.propose(
                    self.state['q'][0], self.state['suspended'],
                    self.state['layers'][0])
                if bool(cp.any(owned['status'] != 0)):
                    raise RuntimeError('M2-C accepted wet/dry ownership rejected')
                change = cp.sum(owned['suspended_phase_change_kg'], axis=1)
                cp.copyto(self.state['suspended'], owned['suspended'])
                cp.copyto(self.state['layers'][0], owned['active_bed'])
                self.wetdry_phase_ledger[:, 0] += change
                self.wetdry_phase_ledger[:, 1] -= change
                cp.copyto(self.last_wetdry_phase_change, change)
                rebalance = self.solver.sediment.rebalance(self.state['layers'])
                if rebalance is not None and bool(cp.any(rebalance != 0)):
                    raise RuntimeError('M2-C post-dry active-layer rebalance rejected')
            residual = cp.asnumpy(self.totals()-self.ledger_totals()-before)
            dry_tolerance = (float(self.suspended_amr_contract.
                                   hydraulic_dry_tolerance_m)
                             if self.suspended_amr_contract is not None else 0.0)
            dry = self.state['q'][0] <= dry_tolerance
            if (abs(residual[0]) > 1e-8 or
                    np.max(np.abs(residual[1:-1])) > 1e-7 or
                    abs(residual[-1]) > 1e-8 or
                    bool(cp.any(self.state['q'][0] < 0)) or
                    bool(cp.any(self.state['layers'] < 0)) or
                    (self.solver.has_suspended and
                     bool(cp.any(self.state['suspended'] < 0))) or
                    (self.solver.has_suspended and
                     bool(cp.any(self.state['suspended'][:, dry] != 0))) or
                    bool(cp.any(self.state['q'][1:, dry] != 0))):
                raise RuntimeError('Synchronous general AMR conservation/positivity rejected')
            if self.solver.has_suspended and bool(cp.any(cp.abs(
                    cp.sum(self.solver.ledger['exchange'], axis=1)) > 1e-8)):
                raise RuntimeError('Synchronous general AMR exchange ledger unbalanced')
            self.accepted_steps += 1
            self.time_s += dt
            self.maximum_cfl = max(self.maximum_cfl, cfl)
            return dict(time_s=self.time_s, step=self.accepted_steps,
                route='synchronous_general_catalog',
                budget_residual=residual.tolist(), cfl=cfl)
        except BaseException:
            self.rollback(saved)
            raise

    def accepted_gradients(self):
        """Native transfer primitives: eta, velocity, concentration, hI, bed."""
        cp, solver = self.cp, self.solver
        bed = solver.bed()
        hydro = solver.reconstruction.reconstruct(self.state['q'])
        secondary = (solver.secondary.reconstruct(self.state['q'], bed,
            self.state['secondary']) if int(self.topology.data['secondary_flow_model']) == 2
            else None)
        require(not bool(cp.any(hydro['primitive_status'])) and
                not bool(cp.any(hydro['reconstruction_status'])) and
                (secondary is None or not bool(cp.any(secondary['status']))),
                'Accepted general AMR primitive gradient reconstruction failed')
        nsecondary = int(secondary is not None)
        gradients = cp.zeros((self.n, 4+self.ng+nsecondary, 2), cp.float64)
        gradients[:, :3] = hydro['grad'][:3].transpose(2, 0, 1)
        if solver.has_suspended:
            values = solver.schedule.evaluate(self.time_s,
                max(float(solver.data['dtmax']), float(np.spacing(self.time_s))))
            require(not bool(cp.any(values['status'][0])),
                    'Accepted general AMR concentration forcing differs')
            cp.copyto(solver.bc_concentration, values['concentration'][0])
            concentration = solver.suspended_faces.step(self.state['q'][0],
                self.state['suspended'], solver.bc_concentration, hydro)
            require(not bool(cp.any(concentration['status'])),
                    'Accepted general AMR concentration reconstruction failed')
            gradients[:, 3:3+self.ng] = concentration['grad'].transpose(2, 0, 1)
        if secondary is not None:
            gradients[:, 3+self.ng] = solver.secondary.buffers['gradient'].T
        gradients[:, 3+self.ng+nsecondary] = hydro['grad'][3].T
        return gradients

    def accepted_bed_gradient(self):
        cp, solver = self.cp, self.solver
        solver.bed()
        hydro = solver.reconstruction.reconstruct(self.state['q'])
        require(not bool(cp.any(hydro['primitive_status'])) and
                not bool(cp.any(hydro['reconstruction_status'])),
                'Accepted general AMR bed gradient reconstruction failed')
        return cp.ascontiguousarray(hydro['grad'][3].T)

    def frame(self):
        cp = self.cp
        values = self.solver.fields(current=self.time_s)
        result = dict(time_s=np.float64(self.time_s), node_ids=self.ids.copy(),
            state=cp.asnumpy(self.full()),
            accepted_steps=np.int64(self.accepted_steps),
            euler_calls=np.int64(self.solver.euler_calls),
            maximum_cfl=np.float64(self.maximum_cfl),
            wetdry_phase_ledger=cp.asnumpy(self.wetdry_phase_ledger),
            wetdry_last_suspended_change_kg=cp.asnumpy(
                self.last_wetdry_phase_change),
            selected_integrator=np.asarray(self.integrator), **values)
        if self.suspended_amr_contract is not None:
            result.update(
                suspended_amr_contract_semantic_id=np.asarray(
                    self.suspended_amr_contract.semantic_id),
                suspended_amr_contract_fingerprint=np.asarray(
                    self.suspended_amr_contract.fingerprint),
                suspended_amr_contract_json=np.asarray(json.dumps(
                    self.suspended_amr_contract.as_dict(), sort_keys=True,
                    separators=(',', ':'), allow_nan=False)))
        return result
