"""Selected-state synchronous GPU AMR for the integrated nonbank solver.

Native initialization supplies authenticated selected geometry on demand.  It
never advances a solution.  This route owns the one accepted CUDA model and
atomically changes its topology after conservative physical-state transfer.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from .amr_catalog_mixed_transition import plan_mixed_transition, propose_mixed_transition
from .amr_general_session import state_selector
from .amr_general_stepper import SynchronousGeneralStepper, require
from .general_live_transport import static_bed_enabled


_GLOBAL_BOUNDARY = ('mode', 'boundary_offset', 'forcing',
    'fraction_concentration', 'fraction_bed_input', 'section_control',
    'section_energy_slope', 'boundary_seed_position',
    'section_seed_node',
    'production_section_geometry', 'reversal_policy', 'bed_boundary_mode',
    'viscosity_equilibrated_inlet', 'nseries', 'nhydro_series',
    'nfraction_series', 'nfrac', 'nb', 'nledger', 'secondary_ledger_index',
    'nstate', 'secondary_index', 'bed_indices', 'suspended_indices',
    'bc_hydraulic', 'bc_bed_input')
_PHYSICAL_FACES = ('mid', 'normal', 'length', 'sampled', 'bid',
    'boundary_mode',
    'boundary_bed_offset', 'boundary_bed_left_offset',
    'boundary_bed_right_offset')


def general_boundary_owners(prepared, top):
    """Ordered global owners for every admitted nonbank open-boundary face."""
    data = top.data
    mode = np.asarray(data['mode'])
    faces = np.asarray(data['boundary_faces'])
    offset = np.asarray(data['boundary_offset'])
    owner = np.asarray(data['owner'])
    neighbour = np.asarray(data['neighbour'])
    bid = np.asarray(data['bid'])
    nb = int(data['nb'])
    source_ids = np.asarray(data.get('face_source_ids'))
    require(prepared.schema == 3 and nb > 0 and mode.shape == (nb,) and
            mode.dtype.kind in 'iu' and
            faces.ndim == 1 and faces.dtype.kind in 'iu' and
            offset.shape == (nb+1,) and offset.dtype.kind in 'iu' and
            offset[0] == 0 and offset[-1] == len(faces) and
            np.all(np.diff(offset) > 0) and
            owner.ndim == 1 and owner.dtype.kind in 'iu' and
            neighbour.shape == owner.shape and neighbour.dtype.kind in 'iu' and
            bid.shape == owner.shape and bid.dtype.kind in 'iu' and
            source_ids.shape == owner.shape and source_ids.dtype.kind in 'iu' and
            np.all(source_ids > 0) and
            np.all((faces >= 0) & (faces < len(owner))) and
            len(np.unique(faces)) == len(faces) and
            np.all((owner[faces] >= 0) & (owner[faces] < len(top.node_ids))),
            'General AMR boundary CSR/owner/source inventory differs')
    owners = top.node_ids[owner[faces]]
    require(len(faces) > 0 and
            np.array_equal(np.sort(faces), np.flatnonzero(bid >= 0)) and
            np.all(neighbour[faces] < 0) and
            all(np.all(bid[faces[offset[boundary]:offset[boundary+1]]] ==
                       boundary) for boundary in range(nb)) and
            len(np.unique(source_ids[faces])) == len(faces) and
            np.all(prepared.tree['node_level'][owners] ==
                   np.max(prepared.tree['node_level'])),
            'General AMR requires unique finest ordered exterior '
            'boundary owners')
    return owners


def validate_general_boundary_handoff(prepared, old, new):
    """Open sections and their actual forcing remain the same physical faces."""
    require(np.array_equal(general_boundary_owners(prepared, old),
                           general_boundary_owners(prepared, new)),
            'General AMR regrid changes finest open-boundary owners')
    a, b = old.data, new.data
    from .amr_typed_roughness import BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD
    typed = AMR_ROUGHNESS_BINDING_FIELD in a or AMR_ROUGHNESS_BINDING_FIELD in b
    if typed:
        require(AMR_ROUGHNESS_BINDING_FIELD in a and
                AMR_ROUGHNESS_BINDING_FIELD in b and
                a[AMR_ROUGHNESS_BINDING_FIELD] ==
                b[AMR_ROUGHNESS_BINDING_FIELD] and
                a.get('amr_typed_roughness_fingerprint') ==
                b.get('amr_typed_roughness_fingerprint') and
                a.get('amr_roughness_restriction_policy') ==
                b.get('amr_roughness_restriction_policy') and
                a.get('amr_roughness_prolongation_policy') ==
                b.get('amr_roughness_prolongation_policy'),
                'General AMR typed roughness identity changes')
    from .implicit_suspension_amr import (CONTRACT_FIELD, FINGERPRINT_FIELD,
                                           SEMANTIC_FIELD)
    declared = CONTRACT_FIELD in a or CONTRACT_FIELD in b
    if declared:
        require(all(key in a and key in b and
                    a[key] == b[key] for key in
                    (CONTRACT_FIELD, FINGERPRINT_FIELD, SEMANTIC_FIELD)),
                'General AMR suspended ownership contract changes')
    for key in (*_GLOBAL_BOUNDARY, *(name for name in set(a)|set(b)
             if name.startswith(('hydro_series_', 'fraction_series_', 'series_')))):
        require((key in a) == (key in b),
                'General AMR boundary record missing: '+key)
        if key in a:
            require(np.asarray(a[key]).dtype == np.asarray(b[key]).dtype and
                    np.array_equal(a[key], b[key]),
                    'General AMR forcing/table changes: '+key)
    af, bf = np.asarray(a['boundary_faces']), np.asarray(b['boundary_faces'])
    require('face_source_ids' in a and 'face_source_ids' in b and
            np.array_equal(np.asarray(a['face_source_ids'])[af],
                           np.asarray(b['face_source_ids'])[bf]) and
            len(np.unique(np.asarray(a['face_source_ids'])[af])) == len(af),
            'General AMR open boundary stable candidate-face IDs change')
    require(af.shape == bf.shape and
            np.array_equal(old.node_ids[np.asarray(a['owner'])[af]],
                           new.node_ids[np.asarray(b['owner'])[bf]]),
            'General AMR ordered boundary faces change')
    for key in _PHYSICAL_FACES:
        require((key in a) == (key in b),
                'General AMR boundary geometry record missing: '+key)
        if key in a:
            require(np.array_equal(np.asarray(a[key])[..., af],
                                   np.asarray(b[key])[..., bf]),
                    'General AMR physical boundary changes: '+key)
    require(np.array_equal(np.asarray(a['manning_cell'])[np.asarray(a['owner'])[af]],
                           np.asarray(b['manning_cell'])[np.asarray(b['owner'])[bf]]),
            'General AMR boundary owner roughness changes')


class GeneralOnDemandRoute:
    """Integrated transport on a selected mesh with accepted-state regrids."""

    def __init__(self, session, *, source_folder, transport_source_folder,
                 integrator='ssp33', cancelled=lambda: False):
        self.session = session
        self.source = Path(source_folder).resolve()
        self.transport_source = Path(transport_source_folder).resolve()
        from .general_live_transport import integration_identity
        integration_identity(integrator, 'plain')
        self.integrator = integrator
        require(callable(cancelled),
                'General AMR cancellation callback must be callable')
        self.cancelled = cancelled
        self.entries = [session.get(session.manifest['initial_node_ids'])]
        self.prepared = session.view(self.entries)
        require(self.prepared.schema == 3 and len(self.prepared.topologies) == 1,
                'General AMR initial selected package differs')
        initial = self.prepared.topologies[0]
        state = np.asarray(session.initial_tree_state)
        require(state.dtype == np.float64 and
                state.shape == (len(self.prepared.tree['parent']), int(initial.data['nstate'])) and
                np.isfinite(state).all() and
                np.allclose(state[initial.node_ids], initial.data['state'].T,
                            rtol=0, atol=1e-10),
                'Original AMR whole-tree state differs from selected native input')
        self.model = self._make_stepper(initial)
        self.cp = self.model.cp
        self.tree_state = self.cp.asarray(state)
        self.tree_area = self.cp.asarray(self.prepared.tree['area'])
        self.groups = []
        parent = self.prepared.tree['parent']
        levels = self.prepared.tree['node_level']
        for lev in range(int(np.max(levels))-1, -1, -1):
            ids = np.flatnonzero((levels == lev) & np.isin(np.arange(len(parent)), parent))
            children = np.asarray([np.flatnonzero(parent == node) for node in ids])
            self.groups.append((self.cp.asarray(ids), self.cp.asarray(children)))
        self.by_leaves = {tuple(map(int, initial.node_ids)): initial}
        self.events = []
        self.selector = self._selector(self.prepared)
        self.selector_last_bed = None
        if self.selector.settings['refine_sediment']:
            rows = self.selector._bed(state, np.arange(len(state)), initial.data)
            self.selector_last_bed = self.cp.asarray(rows)
            self._observe_selector_bed()

    def _make_stepper(self, top):
        return SynchronousGeneralStepper(self.prepared, top,
                                         source_folder=self.transport_source,
                                         integrator=self.integrator,
                                         cancelled=self.cancelled)

    def _handoff_stepper(self, old, new):
        """Carry synchronous accepted clocks after the conservative remap."""
        new.accepted_steps = old.accepted_steps
        new.time_s = old.time_s
        new.maximum_cfl = old.maximum_cfl
        new.solver.euler_calls = old.solver.euler_calls
        implicit = self.integrator == 'implicit_fwave_midpoint'
        receipt = {}
        if implicit:
            require(old.solver.implicit_midpoint is not None and
                    new.solver.implicit_midpoint is not None,
                    'M2-C topology handoff lost the selected implicit solver')
            old_controller = old.solver.implicit_midpoint
            new_controller = new.solver.implicit_midpoint
            require(old_controller.suspended_composition and
                    new_controller.suspended_composition and
                    old.solver.flow2dh_boundary_diffusion is not None and
                    new.solver.flow2dh_boundary_diffusion is not None,
                    'M2-C topology handoff requires law-11/mode-4 suspension')
            memory = old.solver.flow2dh_boundary_diffusion.boundary_memory_fields()
            new.solver.flow2dh_boundary_diffusion.restore_boundary_memory(memory)
            # The predictor is a topology-local hydraulic rate, not physical
            # conserved state.  Invalidating it is exact and only forfeits a
            # Newton initial guess; interpolating it would invent an AMR law.
            new_controller.predictor_rate.fill(0.0)
            new_controller.have_predictor = False
            new_controller.accepted_steps = old_controller.accepted_steps
            new_controller.rejected_steps = old_controller.rejected_steps
            new_controller.fwave_evaluations = old_controller.fwave_evaluations
            new_controller.linear_matvecs = old_controller.linear_matvecs
            new_controller.last_diagnostics = {}
            new_controller.equilibrium_candidate = False
            new_controller._refresh_equilibrium_scales()
            new.wetdry_phase_ledger += old.wetdry_phase_ledger
            receipt.update(
                implicit_predictor_handoff=
                    'invalidated_zero_topology_local_initial_guess_v1',
                flow2dh_boundary_memory_handoff=
                    'ordered_unchanged_finest_physical_faces_bitwise_v1',
                flow2dh_boundary_memory_fields=sorted(memory),
                suspended_amr_contract_semantic_id=
                    new.suspended_amr_contract.semantic_id,
                suspended_amr_contract_fingerprint=
                    new.suspended_amr_contract.fingerprint,
                wetdry_phase_ledger_kg=
                    self.cp.asnumpy(new.wetdry_phase_ledger).tolist(),
                automatic_CPU_fallback=False)
        new.sync_accepted_clock()
        return receipt

    def _selector(self, prepared):
        options = dict(self.session.selector_arguments)
        options['indicator_source_path'] = self.source/'amr_general_indicators.cu'
        return state_selector(prepared, options,
                              getattr(self.session, 'selector_policy', None))

    def synchronized_tree(self):
        """Update current leaves/ancestors while retaining dormant child rows."""
        cp = self.cp
        rows = self.tree_state.copy()
        covered = np.zeros(len(rows), dtype=bool)
        covered[self.model.ids] = True
        for node, parent in enumerate(self.prepared.tree['parent']):
            if parent >= 0:
                covered[node] |= covered[parent]
        covered = cp.asarray(covered)
        rows[cp.asarray(self.model.ids)] = self.model.full()
        for parents, children in self.groups:
            weighted = cp.sum(rows[children]*self.tree_area[children, None], axis=1)
            restricted = weighted/self.tree_area[parents, None]
            rows[parents] = cp.where(covered[parents, None], rows[parents], restricted)
        return rows

    def _selector_state(self):
        extra = {}
        if self.selector_last_bed is not None:
            extra['selector_last_bed'] = self.selector_last_bed
        if self.selector.settings['refine_bathymetry']:
            extra['bed_gradient'] = self.model.accepted_bed_gradient()
        return extra

    def _observe_selector_bed(self):
        if self.selector_last_bed is None:
            return
        cp = self.cp
        bed = self.model.solver.bed()
        require(bed.shape == (self.model.n,) and bed.dtype == cp.float64 and
                bool(cp.all(cp.isfinite(bed))),
                'Invalid accepted general AMR reference bed')
        candidate = self.selector_last_bed.copy()
        candidate[cp.asarray(self.model.ids)] = bed
        self.selector_last_bed = candidate

    def _regrid(self, selected_ids, *, reject_after_candidate=False):
        ids = np.asarray(selected_ids)
        require(ids.ndim == 1 and ids.dtype.kind in 'iu' and
                np.all(np.diff(ids) > 0),
                'General AMR proposal needs unique sorted global IDs')
        key = tuple(map(int, ids))
        registered = (self.entries, self.prepared, self.by_leaves.copy(),
                      self.selector, self.model.prepared)
        physical = (self.model, self.tree_state, list(self.events),
                    self.selector_last_bed)
        try:
            if key not in self.by_leaves:
                entry = self.session.get(ids)
                entries = [*self.entries, entry]
                prepared = self.session.view(entries)
                target = prepared.topologies[-1]
                validate_general_boundary_handoff(prepared, self.model.topology, target)
                selector = self._selector(prepared)
                self.entries, self.prepared = entries, prepared
                self.by_leaves[key] = target
                self.model.prepared = prepared
                self.selector = selector
            target = self.by_leaves[key]
            old = self.model
            if target is old.topology:
                self._observe_selector_bed()
                return None
            validate_general_boundary_handoff(self.prepared, old.topology, target)
            plan = plan_mixed_transition(self.prepared, old.topology, target)
            roughness_candidate = None
            roughness_receipt = {}
            from .amr_typed_roughness import (
                BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD,
                propose_resident_transfer)
            roughness_binding = old.topology.data.get(
                AMR_ROUGHNESS_BINDING_FIELD)
            if roughness_binding is not None:
                typed_tree = dict(
                    parent=np.ascontiguousarray(
                        self.prepared.tree['parent'], dtype=np.int64),
                    node_level=np.ascontiguousarray(
                        self.prepared.tree['node_level'], dtype=np.int64),
                    area=np.ascontiguousarray(
                        self.prepared.tree['area'], dtype=np.float64),
                    roughness_parameter=np.ascontiguousarray(
                        self.session.tree['roughness_parameter'],
                        dtype=np.float64),
                    raw_roughness_mode=np.asarray(
                        self.session.tree['raw_roughness_mode']))
                roughness_candidate, roughness_receipt = \
                    propose_resident_transfer(
                        roughness_binding, typed_tree, old.ids, target.node_ids,
                        old.solver.sediment.geometry['manning_cell'],
                        backend='cuda')
            tree = self.synchronized_tree()
            gradients = old.accepted_gradients() if plan.refine_parents else None
            old_map = {int(node): i for i, node in enumerate(old.ids)}
            inputs = dict(refinement_children={p:self.cp.ascontiguousarray(
                    tree[self.cp.asarray(group)]) for p, group in
                    zip(plan.refine_parents, plan.refine_children)},
                primitive_gradients={p:self.cp.ascontiguousarray(gradients[old_map[p]])
                    for p in plan.refine_parents},
                reject_after_candidate=reject_after_candidate)
            if static_bed_enabled(old.topology.data):
                from .amr_general_static_transfer import propose_static_transition
                proposal = propose_static_transition(self.prepared, plan,
                    old.full(), **inputs)
            else:
                proposal = propose_mixed_transition(self.prepared, plan,
                    old.full(), source_folder=self.source,
                    suspended_wetdry_source=(
                        self.transport_source/'implicit_suspension_amr.cu'
                        if self.integrator == 'implicit_fwave_midpoint'
                        else None),
                    transfer_bank_history=False, **inputs)
            new = self._make_stepper(target)
            new.install(proposal.state)
            if roughness_candidate is not None:
                require(self.cp.array_equal(
                    new.solver.sediment.geometry['manning_cell'],
                    roughness_candidate),
                    'General AMR CUDA raw roughness handoff differs')
            for name, value in old.solver.ledger.items():
                require(name in new.solver.ledger and
                        new.solver.ledger[name].shape == value.shape and
                        new.solver.ledger[name].dtype == value.dtype,
                        'General AMR boundary ledger shape changes: '+name)
                self.cp.copyto(new.solver.ledger[name], value)
            handoff = self._handoff_stepper(old, new)
            if self.integrator == 'implicit_fwave_midpoint':
                # ``propose_mixed_transition`` has already moved every
                # shallow-dry suspended class into its matching active-bed
                # reservoir.  ``new.install`` therefore sees clean dry rows
                # and quite correctly reports no second move.  Carry the
                # proposal's signed phase ledger exactly once into the new
                # accepted model before publishing the topology transaction.
                phase = np.asarray(proposal.receipt.get(
                    'wetdry_suspended_phase_change_kg'))
                require(phase.shape == (new.ng,) and
                        phase.dtype.kind == 'f' and np.isfinite(phase).all() and
                        np.all(phase <= 0.0),
                        'M2-C regrid suspended phase ledger differs')
                device_phase = self.cp.asarray(phase, dtype=self.cp.float64)
                new.wetdry_phase_ledger[:, 0] += device_phase
                new.wetdry_phase_ledger[:, 1] -= device_phase
                new.last_wetdry_phase_change += device_phase
                handoff['accepted_regrid_suspended_phase_change_kg'] = (
                    phase.tolist())
                handoff['wetdry_phase_ledger_kg'] = self.cp.asnumpy(
                    new.wetdry_phase_ledger).tolist()
            tree[self.cp.asarray(new.ids)] = new.full()
            event = dict(time_s=old.time_s, step=old.accepted_steps,
                old_topology=old.topology.name, new_topology=target.name,
                **proposal.receipt, **handoff)
            if roughness_receipt:
                event['typed_roughness_transfer'] = roughness_receipt
            self.tree_state, self.model = tree, new
            self.events.append(event)
            self._observe_selector_bed()
            return event
        except BaseException:
            (self.model, self.tree_state, self.events,
             self.selector_last_bed) = physical
            (self.entries, self.prepared, self.by_leaves,
             self.selector, self.model.prepared) = registered
            raise

    def advance(self, dt, *, delay_state, regrid=False,
                reject_after_candidate=False):
        old = self.model
        saved = old.snapshot()
        try:
            step = old.step(dt, snapshot=saved)
            if not regrid:
                return step, None, None
            started = time.perf_counter()
            proposal = self.selector.propose(old.topology, old.full(),
                cached_tree_state=self.synchronized_tree(),
                delay_state=delay_state, time_s=old.time_s,
                **self._selector_state())
            step['amr_selection_wall_s'] = time.perf_counter()-started
            started = time.perf_counter()
            event = self._regrid(np.asarray(proposal.node_ids, dtype=np.int64),
                                 reject_after_candidate=reject_after_candidate)
            step['amr_regrid_wall_s'] = time.perf_counter()-started
            return step, event, proposal
        except BaseException:
            old.rollback(saved)
            raise

    def frame(self):
        values = self.model.frame()
        values['topology_identity'] = np.asarray(self.model.topology.source_identity)
        current = [entry for entry in self.entries if
                   entry.package.topologies[0].name == self.model.topology.name]
        require(len(current) == 1,
                'General AMR frame lacks its selected operator package')
        values['package_identity'] = np.asarray(current[0].package_identity)
        return values

    def write_checkpoint(self, folder, delay_state, *, method_identity):
        from .amr_general_checkpoint import write
        return write(folder, self, delay_state, method_identity=method_identity)

    @classmethod
    def restore(cls, session, folder, *, source_folder,
                transport_source_folder, method_identity,
                integrator='ssp33', cancelled=lambda: False):
        from .amr_general_checkpoint import restore
        return restore(folder, session, source_folder=source_folder,
                       transport_source_folder=transport_source_folder,
                       method_identity=method_identity,
                       integrator=integrator, cancelled=cancelled)
