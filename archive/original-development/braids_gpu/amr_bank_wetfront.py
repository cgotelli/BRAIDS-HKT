"""Bounded schema-2 CUDA bank-AMR wet-front step and two-layout handoff.

This deliberately does not extend the legacy mixed subcycled/reflux stepper.
The admitted selected layouts have a persistent wet/dry face and therefore
use one synchronous SSPRK33 CUDA step on all leaves, as in the native
fixed-selected-topology reference. Mixed-level bank reflux is a later gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .amr_catalog_transition import coarsen_selected_state, refine_selected_state
from .amr_prepared import AMRTopology, PreparedAMR
from validation.gpu_fwave.amr_bank_history import ResidentAMRBankHistory
from validation.gpu_fwave.amr_bank_history_prolong import ResidentAMRBankHistoryProlong
from validation.gpu_fwave.amr_hydro_route import requires_synchronous_wet_front
from validation.gpu_fwave.coupled_solver import CoupledSolver


HISTORY = (
    ('undercut', 'bank_undercut'),
    ('reference_height', 'bank_reference_height'),
    ('toe_height', 'bank_toe_height'),
    ('reference_length', 'bank_reference_length'),
    ('toe_cumulative', 'bank_toe_cumulative'),
    ('collapse_cumulative', 'bank_collapse_cumulative'),
)
HISTORY_NAMES = tuple(name for name, _ in HISTORY)
COUNT = 'collapse_count'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _copy_device(cp, fields):
    return {key: value.copy() for key, value in fields.items()}


def _history_matrix(solver):
    return solver.cp.ascontiguousarray(solver.cp.stack(
        [solver.bank_history[name] for name in HISTORY_NAMES], axis=1))


def _install_history(solver, values, counts):
    cp = solver.cp
    _require(isinstance(values, cp.ndarray) and
        values.shape == (solver.nc, 6) and values.dtype == cp.float64 and
        isinstance(counts, cp.ndarray) and
        counts.shape == (solver.nc,) and counts.dtype == cp.int32,
        'Resident seven-bank-history shape/type differs')
    for column, name in enumerate(HISTORY_NAMES):
        cp.copyto(solver.bank_history[name], values[:, column])
    cp.copyto(solver.bank_history[COUNT], counts)


class BankWetFrontStepper:
    """A single selected topology with accepted mode-2 bank auxiliary state."""

    def __init__(self, prepared: PreparedAMR, topology: AMRTopology, *,
                 source_folder, outer_dt=.002):
        _require(prepared.schema == 2 and
            any(item is topology for item in prepared.topologies),
            'Bank wet-front step requires authenticated schema-2 topology')
        data = topology.data
        _require(int(data['nfrac']) == 2 and int(data['nbed_layers']) == 2 and
            int(data['nstate']) == 10 and int(data['bedlaw']) == 2 and
            int(data['exchange_mode']) == 0 and
            int(data['bank_toe_erosion_mode']) == 2 and
            int(data['general_avalanche']) == 1 and
            int(data['secondary_flow_model']) == 2 and
            int(data['amr_enabled']) == 0 and
            np.array_equal(data['mode'], [2]) and
            float(data['endtime']) == .08 and
            float(data['dtmax']) == .002 and
            float(data['output_dt']) == .002 and
            np.all(np.asarray(data['state'])[
                np.asarray(data['suspended_indices'])] == 0),
            'Bank wet-front fixture physics/clock differs')
        faces = np.asarray(data['boundary_faces'], dtype=np.int64)
        owner = np.asarray(data['owner'], dtype=np.int64)
        nodes = topology.node_ids[owner[faces]]
        _require(len(faces) == 8 and len(np.unique(nodes)) == 8 and
            np.all(prepared.tree['node_level'][nodes] == 2),
            'Bank wet-front stage outlet must have eight finest owners')
        source = Path(source_folder).resolve()
        for filename in ('amr_transfer.cu', 'amr_bank_history.cu',
                         'amr_bank_history_prolong.cu', 'toe_supply.cu',
                         'bank_undercut.cu', 'toe_face_diagnostic.cu',
                         'avalanche_proposal.cu'):
            _require((source/filename).is_file(),
                     'Bank wet-front CUDA source is missing: '+filename)
        _require(np.isfinite(outer_dt) and outer_dt == .002,
                 'Bank wet-front outer dt differs')
        self.prepared = prepared
        self.topology = topology
        self.source = source
        self.outer_dt = float(outer_dt)
        self.ids = topology.node_ids.copy()
        self.n = len(self.ids)
        self.solver = CoupledSolver(data, source_folder=source,
            integrator='ssp33', bank_avalanche_mode2_variant=True,
            g3_bank_secondary_variant=True)
        self.cp = self.solver.cp
        _require(not self.solver.has_suspended and
            set(self.solver.state) == {'q','layers','secondary'} and
            set(self.solver.ledger) == {'water','bed','secondary','source'},
            'Exchange-0 bank route must not construct suspended physics')
        for gpu_name, native_name in HISTORY:
            self.cp.copyto(self.solver.bank_history[gpu_name],
                           self.cp.asarray(data[native_name]))
        self.cp.copyto(self.solver.bank_history[COUNT],
                       self.cp.asarray(data['bank_collapse_count']))
        self.state = self.solver.state
        self.area = self.cp.asarray(data['area'])
        self.owner = self.cp.asarray(data['owner'])
        self.neighbour = self.cp.asarray(data['neighbour'])
        self.accepted_steps = 0
        self.maximum_cfl = 0.
        self.route_steps = 0

    def _full(self):
        cp = self.cp
        result = cp.empty((self.n, 10), dtype=cp.float64)
        result[:, :3] = self.state['q'].T
        result[:, 3:5] = 0.
        result[:, 5:7] = self.state['layers'][0].T
        result[:, 7:9] = self.state['layers'][1].T
        result[:, 9] = self.state['secondary']
        return cp.ascontiguousarray(result)

    def install_full(self, full):
        cp = self.cp
        _require(isinstance(full, cp.ndarray) and
            full.shape == (self.n, 10) and full.dtype == cp.float64 and
            full.flags.c_contiguous and
            bool(cp.all(cp.isfinite(full))) and
            bool(cp.all(full[:, 3:5] == 0)) and
            bool(cp.all(full[:, 0] >= -1e-12)) and
            bool(cp.all(full[:, 5:9] >= -1e-12)),
            'Invalid resident bank-AMR conserved state')
        cp.copyto(self.state['q'], cp.ascontiguousarray(full[:, :3].T))
        cp.copyto(self.state['layers'][0],
                  cp.ascontiguousarray(full[:, 5:7].T))
        cp.copyto(self.state['layers'][1],
                  cp.ascontiguousarray(full[:, 7:9].T))
        cp.copyto(self.state['secondary'], full[:, 9])

    def budgets(self):
        cp = self.cp
        water = float(cp.sum(self.area*self.state['q'][0]).item())
        grains = cp.asnumpy(cp.sum(self.area[None, :]*
            cp.sum(self.state['layers'], axis=0), axis=1))
        secondary = float(cp.sum(self.area*self.state['secondary']).item())
        return water, grains, secondary

    def _snapshot(self):
        solver = self.solver
        return dict(state=_copy_device(self.cp, solver.state),
            ledger=_copy_device(self.cp, solver.ledger),
            history=_copy_device(self.cp, solver.bank_history),
            toe_signed=solver.toe_face_signed.copy(),
            toe_gross=solver.toe_face_gross.copy(),
            euler_calls=solver.euler_calls,
            toe_stage_index=solver.toe_stage_index,
            avalanche_trial_substeps=solver.avalanche_trial_substeps.copy(),
            bank_trial=solver.bank_trial if hasattr(solver, 'bank_trial') else None,
            accepted_steps=self.accepted_steps,
            maximum_cfl=self.maximum_cfl,
            route_steps=self.route_steps)

    def _restore(self, saved):
        cp = self.cp
        solver = self.solver
        for name, value in saved['state'].items():
            cp.copyto(solver.state[name], value)
        for name, value in saved['ledger'].items():
            cp.copyto(solver.ledger[name], value)
        for name, value in saved['history'].items():
            cp.copyto(solver.bank_history[name], value)
        cp.copyto(solver.toe_face_signed, saved['toe_signed'])
        cp.copyto(solver.toe_face_gross, saved['toe_gross'])
        solver.euler_calls = saved['euler_calls']
        solver.toe_stage_index = saved['toe_stage_index']
        solver.avalanche_trial_substeps = saved['avalanche_trial_substeps']
        solver.bank_trial = saved['bank_trial']
        self.accepted_steps = saved['accepted_steps']
        self.maximum_cfl = saved['maximum_cfl']
        self.route_steps = saved['route_steps']

    def step(self, dt, time, *, reject_after_active_toe=False):
        _require(np.isfinite((dt,time)).all() and dt == self.outer_dt and
            abs(time-self.accepted_steps*self.outer_dt) <= 1e-12 and
            time+dt <= float(self.topology.data['endtime'])+1e-12,
            'Bank wet-front accepted clock differs')
        if not requires_synchronous_wet_front(self.state['q'], self.owner,
                                               self.neighbour):
            raise ValueError('Bank AMR mixed-level/reflux route is not qualified')
        cp = self.cp
        saved = self._snapshot()
        before = self.budgets()
        try:
            result = self.solver.step(dt, time)
            if bool(cp.any(result['status'] != 0)):
                raise RuntimeError('Bank wet-front CUDA step rejected')
            cfl = float(cp.max(result['cfl']).item())
            if not np.isfinite(cfl) or cfl > float(self.topology.data['cfl_target'])+1e-12:
                raise RuntimeError('Bank wet-front CUDA CFL rejected')
            if reject_after_active_toe:
                activity = cp.sum(cp.abs(self.solver.toe_face_diagnostic.step_net))
                if not bool(activity > 0):
                    raise RuntimeError('Injected rejection did not reach active toe flux')
                raise RuntimeError('injected late active-toe bank-AMR rejection')
            after = self.budgets()
            led = self.solver.ledger
            boundary_water = float(cp.sum(led['water']-saved['ledger']['water']).item())
            boundary_grain = cp.asnumpy(cp.sum(led['bed']-saved['ledger']['bed'], axis=1))
            boundary_secondary = float(cp.sum(
                led['secondary']-saved['ledger']['secondary']).item())
            source = float((led['source']-saved['ledger']['source']).item())
            water_residual = after[0]-before[0]-boundary_water
            grain_residual = after[1]-before[1]-boundary_grain
            secondary_residual = after[2]-before[2]-boundary_secondary-source
            if (abs(water_residual) > 1e-8 or
                    np.max(np.abs(grain_residual)) > 1e-7 or
                    abs(secondary_residual) > 1e-7 or
                    bool(cp.any(self.state['q'][0] < -1e-12)) or
                    bool(cp.any(self.state['layers'] < -1e-12))):
                raise RuntimeError('Bank wet-front accepted budget/positivity rejected')
            self.accepted_steps += 1
            self.route_steps += 1
            self.maximum_cfl = max(self.maximum_cfl, cfl)
            return dict(time_s=time+dt, route='bank_synchronous_wetfront',
                water_residual_m3=water_residual,
                class_residual_kg=grain_residual.tolist(),
                secondary_residual=secondary_residual,
                boundary_water_m3=boundary_water,
                boundary_class_kg=boundary_grain.tolist(),
                boundary_secondary=boundary_secondary,
                maximum_cfl=self.maximum_cfl)
        except Exception:
            self._restore(saved)
            raise

    def persistent(self):
        cp = self.cp
        fields = dict(state=cp.asnumpy(self._full()),
            node_ids=self.ids.copy(),
            accepted_steps=np.int64(self.accepted_steps),
            route_steps=np.int64(self.route_steps),
            euler_calls=np.int64(self.solver.euler_calls),
            maximum_cfl=np.float64(self.maximum_cfl),
            time_s=np.float64(self.accepted_steps*self.outer_dt),
            toe_signed=cp.asnumpy(self.solver.toe_face_signed),
            toe_gross=cp.asnumpy(self.solver.toe_face_gross))
        fields.update({'history_'+name:cp.asnumpy(value) for name,value in
                       self.solver.bank_history.items()})
        fields.update({'ledger_'+name:cp.asnumpy(value) for name,value in
                       self.solver.ledger.items()})
        return fields

    def restore(self, fields):
        cp = self.cp
        _require(np.array_equal(fields['node_ids'], self.ids) and
            fields['state'].shape == (self.n,10) and
            int(fields['accepted_steps']) >= 0 and
            int(fields['route_steps']) == int(fields['accepted_steps']),
            'Bank wet-front checkpoint topology/clock differs')
        self.install_full(cp.asarray(fields['state']))
        for name in self.solver.bank_history:
            cp.copyto(self.solver.bank_history[name],
                      cp.asarray(fields['history_'+name]))
        for name in self.solver.ledger:
            cp.copyto(self.solver.ledger[name],cp.asarray(fields['ledger_'+name]))
        cp.copyto(self.solver.toe_face_signed, cp.asarray(fields['toe_signed']))
        cp.copyto(self.solver.toe_face_gross, cp.asarray(fields['toe_gross']))
        self.accepted_steps = int(fields['accepted_steps'])
        self.route_steps = int(fields['route_steps'])
        self.solver.euler_calls = int(fields['euler_calls'])
        self.maximum_cfl = float(fields['maximum_cfl'])


def _transfer_history(prepared, old, new, parent, children, action, source):
    """Use existing qualified CUDA native-rule restriction/inheritance kernels."""
    cp = old.cp
    old_values = _history_matrix(old.solver)
    old_counts = old.solver.bank_history[COUNT]
    old_map = {int(node): i for i,node in enumerate(old.ids)}
    new_map = {int(node): i for i,node in enumerate(new.ids)}
    retained = sorted(set(old_map) & set(new_map))
    if action == 'coarsen':
        positions = np.asarray([old_map[int(node)] for node in children],
                               dtype=np.int64)
        weights = np.asarray(prepared.tree['area'][children],
                             dtype=np.float64)[None, :]
        operator = ResidentAMRBankHistory(weights,
            cp.zeros((1,6),dtype=cp.float64),
            cp.zeros(1,dtype=cp.int32),
            source_path=source/'amr_bank_history.cu')
        proposal, counts, status = operator.propose(
            cp.ascontiguousarray(old_values[cp.asarray(positions)][None]),
            cp.ascontiguousarray(old_counts[cp.asarray(positions)][None]))
        if bool(cp.any(status != 0)):
            operator.rollback()
            raise RuntimeError('AMR bank-history restriction rejected')
        operator.commit()
        values = cp.empty((new.n,6),dtype=cp.float64)
        next_counts = cp.empty(new.n,dtype=cp.int32)
        values[cp.asarray([new_map[node] for node in retained])] = \
            old_values[cp.asarray([old_map[node] for node in retained])]
        next_counts[cp.asarray([new_map[node] for node in retained])] = \
            old_counts[cp.asarray([old_map[node] for node in retained])]
        values[new_map[parent]] = operator.accepted[0]
        next_counts[new_map[parent]] = operator.accepted_count[0]
    else:
        total = len(prepared.tree['parent'])
        values_tree = cp.zeros((total,6),dtype=cp.float64)
        count_tree = cp.zeros(total,dtype=cp.int32)
        values_tree[cp.asarray(old.ids)] = old_values
        count_tree[cp.asarray(old.ids)] = old_counts
        old_leaf = np.zeros(total,dtype=np.int32)
        old_leaf[old.ids] = 1
        selected = np.zeros(total,dtype=np.int32)
        selected[new.ids] = 1
        cover = np.full(total,-1,dtype=np.int32)
        cover[np.asarray(retained,dtype=np.int64)] = retained
        cover[np.asarray(children,dtype=np.int64)] = parent
        operator = ResidentAMRBankHistoryProlong(values_tree,count_tree,
            old_leaf,source_path=source/'amr_bank_history_prolong.cu')
        proposal, counts, status = operator.propose(selected,cover)
        if bool(cp.any(status != 0)):
            operator.rollback()
            raise RuntimeError('AMR bank-history inheritance rejected')
        operator.commit()
        values = cp.ascontiguousarray(operator.accepted[cp.asarray(new.ids)])
        next_counts = cp.ascontiguousarray(
            operator.accepted_count[cp.asarray(new.ids)])
    _install_history(new.solver, values, next_counts)
    if not bool(cp.array_equal(values[cp.asarray([new_map[node] for node in retained])],
                               old_values[cp.asarray([old_map[node] for node in retained])])):
        raise RuntimeError('AMR retained bank history changed')
    return dict(bank_history_transfer='native_max_area_or_current_parent_cuda',
                all_seven_bank_histories_transferred_on_CUDA=True)


class BankWetFrontRoute:
    """One accepted bank wet-front stepper with two authenticated mesh choices."""

    def __init__(self, prepared, parent, children, *, source_folder,
                 mesh_index=0, outer_dt=.002, min_dwell=2,
                 coarsen_spread_m=.080, refine_jump_m=.070):
        _require(prepared.schema == 2 and len(prepared.topologies) == 2 and
            mesh_index in (0,1) and parent == 56 and
            np.array_equal(children,[224,225,256,257]) and
            min_dwell == 2 and coarsen_spread_m == .080 and
            refine_jump_m == .070 and outer_dt == .002,
            'Frozen bank wet-front two-layout policy differs')
        self.prepared = prepared
        self.parent = int(parent)
        self.children = np.asarray(children,dtype=np.int64).copy()
        self.source = Path(source_folder).resolve()
        self.policy = dict(min_dwell=min_dwell,
            coarsen_spread_m=float(coarsen_spread_m),
            refine_jump_m=float(refine_jump_m))
        self.outer_dt = float(outer_dt)
        self.mesh_index = int(mesh_index)
        self.model = BankWetFrontStepper(prepared, prepared.topologies[mesh_index],
            source_folder=self.source, outer_dt=self.outer_dt)
        self.cp = self.model.cp
        self.cached_children = self.cp.empty((0,4,10),dtype=self.cp.float64)
        self.last_transition_step = 0
        self.events = []
        self.closed_epochs = []

    def indicator(self):
        cp = self.cp
        model = self.model
        if self.mesh_index == 0:
            positions = [int(np.flatnonzero(model.ids == node)[0])
                         for node in self.children]
            depth = model.state['q'][0,cp.asarray(positions)]
            return 'coarsen', float((cp.max(depth)-cp.min(depth)).item())
        pos = int(np.flatnonzero(model.ids == self.parent)[0])
        owner = np.asarray(model.topology.data['owner'])
        neighbour = np.asarray(model.topology.data['neighbour'])
        adjacent = np.unique(np.concatenate((
            neighbour[(owner == pos) & (neighbour >= 0)],
            owner[neighbour == pos])))
        depth = model.state['q'][0]
        return 'refine', float(cp.max(cp.abs(
            depth[cp.asarray(adjacent)]-depth[pos])).item())

    def coarsen_guard(self):
        """Protect unresolved wet fronts/stage structure from restriction.

        A conservative parent cannot retain a subcell shoreline or its
        secondary-flow source over subsequent coarse steps.  Only the
        candidate four-child group is inspected; no state is changed.
        """
        _require(self.mesh_index == 0,
                 'Bank-AMR coarsening guard requires refined topology')
        positions = [int(np.flatnonzero(self.model.ids == node)[0])
                     for node in self.children]
        child = self.cp.asnumpy(
            self.model._full()[self.cp.asarray(positions)])
        data = self.model.topology.data
        depth = child[:, 0]
        dry = float(data['dry_tolerance'])
        wet = depth > dry
        mixed = bool(np.any(wet) and np.any(~wet))
        rock = np.asarray(data['rock'])[positions]
        bulk = np.asarray(data['bulk_fraction'])
        bed = rock + np.sum((child[:, 5:7] + child[:, 7:9]) / bulk,
                            axis=1)
        stage = depth + bed
        wet_stage_span = float(np.ptp(stage[wet])) if np.any(wet) else 0.
        stage_limit = .020  # Native AMR hydraulic eta tolerance in this deck.
        return dict(veto=bool(mixed or wet_stage_span > stage_limit),
                    mixed_solver_wet_dry=mixed,
                    solver_wet=wet.tolist(),
                    depth_m=depth.tolist(),
                    wet_stage_span_m=wet_stage_span,
                    stage_limit_m=stage_limit,
                    dry_tolerance_m=dry)

    def proposed_action(self):
        if self.model.accepted_steps-self.last_transition_step < \
                self.policy['min_dwell']:
            return None, None
        action, value = self.indicator()
        threshold = self.policy['coarsen_spread_m'] if action == 'coarsen' \
                    else self.policy['refine_jump_m']
        accept = value < threshold if action == 'coarsen' else value > threshold
        if accept and action == 'coarsen' and self.coarsen_guard()['veto']:
            accept = False
        return (action if accept else None), value

    def transition(self, action, indicator_m, *, reject_after_candidate=False):
        _require(action == ('coarsen' if self.mesh_index == 0 else 'refine'),
                 'Bank-AMR transition direction differs')
        expected, actual = self.proposed_action()
        _require(expected == action and actual == indicator_m,
                 'Bank-AMR transition is not justified by accepted state')
        old = self.model
        target_index = 1-self.mesh_index
        target = self.prepared.topologies[target_index]
        full = old._full()
        if action == 'coarsen':
            positions = [int(np.flatnonzero(old.ids == node)[0])
                         for node in self.children]
            cache = self.cp.ascontiguousarray(
                full[self.cp.asarray(positions)][None])
            candidate, transfer = coarsen_selected_state(self.prepared,
                old.topology,target,full,
                transfer_source=self.source/'amr_transfer.cu')
        else:
            cache = self.cp.empty((0,4,10),dtype=self.cp.float64)
            candidate, transfer = refine_selected_state(self.prepared,
                old.topology,target,full,self.cached_children,
                transfer_source=self.source/'amr_transfer.cu')
        replacement = BankWetFrontStepper(self.prepared,target,
            source_folder=self.source,outer_dt=self.outer_dt)
        replacement.install_full(candidate)
        bank_transfer = _transfer_history(self.prepared,old,replacement,
            self.parent,self.children,action,self.source)
        for name,value in old.solver.ledger.items():
            self.cp.copyto(replacement.solver.ledger[name],value)
        replacement.accepted_steps = old.accepted_steps
        replacement.route_steps = old.route_steps
        replacement.solver.euler_calls = old.solver.euler_calls
        replacement.maximum_cfl = old.maximum_cfl
        _require(bool(self.cp.array_equal(replacement._full(), candidate)),
                 'Installed bank-AMR transfer state differs')
        if reject_after_candidate:
            raise RuntimeError('injected late bank-AMR regrid rejection')
        epoch = dict(topology_identity=old.topology.source_identity,
            name=old.topology.name,
            signed=old.solver.toe_face_signed.copy(),
            gross=old.solver.toe_face_gross.copy())
        event = dict(step=old.accepted_steps,
            time_s=old.accepted_steps*self.outer_dt,action=action,
            indicator_m=indicator_m,parent_id=self.parent,
            old_leaves=old.n,new_leaves=replacement.n,
            old_operator_identity=old.topology.source_identity,
            new_operator_identity=target.source_identity,
            face_epoch_sealed=True)
        event.update(transfer)
        event.update(bank_transfer)
        self.model = replacement
        self.mesh_index = target_index
        self.cached_children = cache
        self.closed_epochs.append(epoch)
        self.last_transition_step = old.accepted_steps
        self.events.append(event)
        return event

    def advance(self):
        current = self.model.accepted_steps
        result = self.model.step(self.outer_dt,current*self.outer_dt)
        action, indicator = self.proposed_action()
        event = None if action is None else self.transition(action,indicator)
        return result,event

    def persistent(self):
        fields = self.model.persistent()
        fields.update(regrid_mesh_index=np.int64(self.mesh_index),
            regrid_last_step=np.int64(self.last_transition_step),
            regrid_events_json=np.asarray(json.dumps(self.events,
                sort_keys=True,separators=(',',':'),allow_nan=False)),
            regrid_policy_json=np.asarray(json.dumps(self.policy,
                sort_keys=True,separators=(',',':'),allow_nan=False)),
            regrid_cached_children=self.cp.asnumpy(self.cached_children),
            toe_epoch_count=np.int64(len(self.closed_epochs)))
        for i,epoch in enumerate(self.closed_epochs):
            fields['toe_epoch_'+str(i)+'_topology'] = np.asarray(
                epoch['topology_identity'])
            fields['toe_epoch_'+str(i)+'_signed'] = self.cp.asnumpy(
                epoch['signed'])
            fields['toe_epoch_'+str(i)+'_gross'] = self.cp.asnumpy(
                epoch['gross'])
        return fields

    def restore(self, fields):
        _require(int(fields['regrid_mesh_index']) == self.mesh_index,
                 'Bank-AMR checkpoint mesh selector differs')
        self.model.restore(fields)
        self.last_transition_step = int(fields['regrid_last_step'])
        self.events = json.loads(str(fields['regrid_events_json']))
        self.cached_children = self.cp.asarray(fields['regrid_cached_children'])
        self.closed_epochs = []
        for i in range(int(fields['toe_epoch_count'])):
            prefix = 'toe_epoch_'+str(i)+'_'
            self.closed_epochs.append(dict(
                topology_identity=str(fields[prefix+'topology']),
                name=self.prepared.topologies[i%2].name,
                signed=self.cp.asarray(fields[prefix+'signed']),
                gross=self.cp.asarray(fields[prefix+'gross'])))
