"""Synchronous coupled bank evolution and atomic catalog-wide AMR handoffs.

Numerical work uses the existing augmented f-wave, MPM, secondary, bank and
avalanche CUDA operators. This is not a subcycled bank/reflux implementation.
The catalog supplies geometry only. Selection belongs to the accepted-state
selector; no native trajectory or prescribed mesh schedule is read here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .amr_bank_wetfront import HISTORY, COUNT, _history_matrix, _install_history
from .amr_catalog_mixed_transition import plan_mixed_transition, propose_mixed_transition
from .a1_full_array_transaction import _unique_arrays, _require_disjoint_storage
from braids_support.gpu_fwave.a2_segmented_array_copy import SegmentedCopyPlan, signature
from braids_support.gpu_fwave.coupled_solver import CoupledSolver


def require(value, message):
    if not value:
        raise ValueError(message)


def outlet_owners(prepared, top):
    """Ordered owners of every open face, including both live-section ends."""
    data = top.data
    mode=np.asarray(data['mode']);nb=len(mode)
    legacy=np.array_equal(mode,[2]);live=np.array_equal(mode,[9,4])
    require(legacy or live,'Synchronous bank boundary mode set is unsupported')
    faces=np.asarray(data['boundary_faces']);offset=np.asarray(data['boundary_offset'])
    owner=np.asarray(data['owner']);neighbour=np.asarray(data['neighbour']);bid=np.asarray(data['bid'])
    require(faces.ndim==1 and faces.dtype.kind in 'iu' and
            offset.shape==(nb+1,) and offset.dtype.kind in 'iu' and
            offset[0]==0 and offset[-1]==len(faces) and np.all(np.diff(offset)>0) and
            np.all((faces>=0)&(faces<len(owner))) and len(np.unique(faces))==len(faces) and
            owner.ndim==1 and owner.dtype.kind in 'iu' and
            neighbour.shape==owner.shape and neighbour.dtype.kind in 'iu' and
            bid.shape==owner.shape and bid.dtype.kind in 'iu' and
            np.all((owner>=0)&(owner<len(top.node_ids))),
            'Synchronous bank boundary CSR/owner shape differs')
    owners = top.node_ids[owner[faces]]
    require(len(faces) > 0 and
            np.array_equal(np.sort(faces), np.flatnonzero(data['bid'] >= 0)) and
            np.all(neighbour[faces]<0) and
            all(np.all(bid[faces[offset[b]:offset[b+1]]]==b) for b in range(nb)) and
            np.all(prepared.tree['node_level'][owners] ==
                   np.max(prepared.tree['node_level'])),
            'Synchronous bank catalog requires finest ordered exterior boundary owners')
    if legacy:
        require(len(set(map(int,owners)))==len(faces),
                'Legacy stage route requires unique finest stage-outlet owners')
    return owners


def validate_boundary_handoff(prepared,old,new):
    """No boundary geometry, forcing table or ledger convention may regrid."""
    require(np.array_equal(outlet_owners(prepared,old),outlet_owners(prepared,new)),
            'Catalog changes boundary global owner ordering')
    a,b=old.data,new.data
    keys={'mode','boundary_offset','forcing','fraction_concentration','fraction_bed_input',
          'section_control','section_energy_slope','section_seed_node','boundary_seed_position',
          'production_section_geometry','reversal_policy','bed_boundary_mode',
          'viscosity_equilibrated_inlet','nseries','nhydro_series','nfraction_series',
          'nfrac','nb','nledger','secondary_ledger_index','nstate','secondary_index',
          'bed_indices','suspended_indices','bc_hydraulic','bc_bed_input'}
    keys.update(k for d in (a,b) for k in d if k.startswith(
        ('hydro_series_','fraction_series_','series_')))
    for key in keys:
        require((key in a)==(key in b),'Boundary handoff record missing: '+key)
        if key in a:
            require(np.asarray(a[key]).dtype==np.asarray(b[key]).dtype and
                    np.array_equal(a[key],b[key]),'Boundary handoff control/table changed: '+key)
    af,bf=np.asarray(a['boundary_faces']),np.asarray(b['boundary_faces'])
    for key in ('mid','normal','length','sampled','bid','boundary_bed_offset',
                'boundary_bed_left_offset','boundary_bed_right_offset'):
        require((key in a)==(key in b),'Boundary handoff geometry missing: '+key)
        if key in a:
            require(np.array_equal(np.asarray(a[key])[...,af],np.asarray(b[key])[...,bf]),
                    'Boundary handoff physical faces changed: '+key)
    require(np.array_equal(np.asarray(a['manning_cell'])[np.asarray(a['owner'])[af]],
                           np.asarray(b['manning_cell'])[np.asarray(b['owner'])[bf]]),
            'Boundary handoff roughness changed')


class SynchronousBankStepper:
    """One arbitrary authenticated leaf layout, with full device rollback."""

    def __init__(self, prepared, topology, *, source_folder, evolving_toe_variant=False):
        require(prepared.schema == 2 and any(topology is item for item in
                prepared.topologies), 'Bank stepper needs an authenticated schema-2 member')
        data = topology.data
        self.bank_mode = int(data['bank_toe_erosion_mode'])
        require(self.bank_mode in (2, 3) and not
                (self.bank_mode == 3 and evolving_toe_variant),
                'Distributed bank mode cannot use a physical-toe map')
        self.live_boundary=bool((evolving_toe_variant or self.bank_mode == 3) and
                                np.array_equal(data['mode'],[9,4]))
        secondary_model = 2 if self.live_boundary or self.bank_mode == 2 else 0
        require(int(data['exchange_mode']) == 0 and
                int(data['secondary_flow_model']) == secondary_model and
                int(data['amr_enabled']) == 0 and
                (np.array_equal(data['mode'], [2]) or self.live_boundary) and
                np.all(data['state'][np.asarray(data['suspended_indices'])] == 0),
                'Synchronous bank route requires supported unsuspended bank/secondary boundaries')
        self.outlet_ids = outlet_owners(prepared, topology)
        self.prepared, self.topology = prepared, topology
        self.ids = topology.node_ids.copy()
        self.source = Path(source_folder).resolve()
        self.evolving_toe_variant = bool(evolving_toe_variant)
        if self.evolving_toe_variant:
            require(all(not np.any(data[native]) for _, native in HISTORY[:4]),
                    'Cold evolving AMR input cannot authenticate preexisting geometric history')
        if self.live_boundary:
            from .general_live_bank import constructors
            with constructors(self.source/'live_bank',data) as Solver:
                self.solver=Solver(data,source_folder=self.source/'live_bank')
        elif self.bank_mode == 2:
            self.solver = CoupledSolver(data, source_folder=self.source,
                integrator='ssp33', bank_avalanche_mode2_variant=True,
                g3_bank_secondary_variant=True, evolving_toe_variant=self.evolving_toe_variant)
        else:
            self.solver = CoupledSolver(data, source_folder=self.source,
                integrator='ssp33', distributed_bank_avalanche_mode3_variant=True)
        self.cp = self.solver.cp
        self.state = self.solver.state
        self.n, self.ns = len(self.ids), int(data['nstate'])
        self.ng, self.nl = int(data['nfrac']), int(data['nbed_layers'])
        self.area = self.cp.asarray(data['area'])
        for device, native in HISTORY:
            self.cp.copyto(self.solver.bank_history[device], self.cp.asarray(data[native]))
        self.cp.copyto(self.solver.bank_history[COUNT],
                       self.cp.asarray(data['bank_collapse_count']))
        # Stabilize the stage alias before the complete physical-array inventory.
        self.solver.flux.inputs['q'] = self.solver.stepper.stage['q']
        self.accepted_steps, self.time_s, self.maximum_cfl = 0, 0., 0.
        self._copy_plan = None

    def full(self):
        cp, data = self.cp, self.topology.data
        out = cp.zeros((self.n, self.ns), dtype=cp.float64)
        out[:, :3] = self.state['q'].T
        for layer, indices in enumerate(np.asarray(data['bed_indices'])):
            out[:, cp.asarray(indices)] = self.state['layers'][layer].T
        if int(data['secondary_flow_model']) == 2:
            out[:, int(data['secondary_index'])] = self.state['secondary']
        return out

    def install(self, rows):
        cp, data = self.cp, self.topology.data
        require(isinstance(rows, cp.ndarray) and rows.dtype == cp.float64 and
                rows.shape == (self.n, self.ns) and rows.flags.c_contiguous and
                bool(cp.all(cp.isfinite(rows))) and bool(cp.all(rows[:, 0] >= 0)) and
                bool(cp.all(rows[:, cp.asarray(data['suspended_indices'])] == 0)) and
                bool(cp.all(rows[:, cp.asarray(data['bed_indices']).ravel()] >= 0)),
                'Invalid accepted bank state')
        cp.copyto(self.state['q'], cp.ascontiguousarray(rows[:, :3].T))
        for layer, indices in enumerate(np.asarray(data['bed_indices'])):
            cp.copyto(self.state['layers'][layer],
                      cp.ascontiguousarray(rows[:, cp.asarray(indices)].T))
        if int(data['secondary_flow_model']) == 2:
            cp.copyto(self.state['secondary'], rows[:, int(data['secondary_index'])])
        else:
            self.state['secondary'].fill(0.)

    def totals(self):
        cp = self.cp
        return cp.concatenate((cp.sum(self.area * self.state['q'][0])[None],
            cp.sum(self.area[None, :] * cp.sum(self.state['layers'], axis=0), axis=1),
            cp.sum(self.area * self.state['secondary'])[None]))

    def ledger_totals(self):
        cp, led = self.cp, self.solver.ledger
        return cp.concatenate((cp.sum(led['water'])[None], cp.sum(led['bed'], axis=1),
            (cp.sum(led['secondary']) + led['source'])[None]))

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
        saved=dict(plan=plan, accepted_steps=self.accepted_steps,
            time_s=self.time_s, maximum_cfl=self.maximum_cfl,
            euler_calls=self.solver.euler_calls,
            toe_stage_index=self.solver.toe_stage_index,
            bank_trial=getattr(self.solver, 'bank_trial', None),
            avalanche_trial_substeps=self.solver.avalanche_trial_substeps.copy())
        if self.bank_mode == 3:
            saved['distributed_host'] = (self.solver.distributed_stage_index,
                                        self.solver.distributed_bank.collapse_waves)
        if self.live_boundary:
            saved['live_host']=dict(accepted_time=self.solver._glb_accepted_time,
                boundary_next_stage=getattr(self.solver.boundary,'_g1_next_stage',None),
                boundary_stage_present=hasattr(self.solver.boundary,'_g1_next_stage'))
            if self.solver.bedload_adaptation_variant:
                saved['live_host']['adaptation']=(self.solver.sediment.adaptation.iterations,
                    self.solver.sediment.adaptation.residual,
                    None if self.solver.sediment.adaptation_result is None else
                    dict(self.solver.sediment.adaptation_result))
        return saved

    def rollback(self, saved):
        saved['plan'].restore()
        for key in ('accepted_steps', 'time_s', 'maximum_cfl'):
            setattr(self, key, saved[key])
        for key in ('euler_calls', 'toe_stage_index', 'bank_trial',
                    'avalanche_trial_substeps'):
            setattr(self.solver, key, saved[key])
        if 'distributed_host' in saved:
            (self.solver.distributed_stage_index,
             self.solver.distributed_bank.collapse_waves) = saved['distributed_host']
        if 'live_host' in saved:
            host=saved['live_host'];self.solver._glb_accepted_time=host['accepted_time']
            if host['boundary_stage_present']:
                self.solver.boundary._g1_next_stage=host['boundary_next_stage']
            elif hasattr(self.solver.boundary,'_g1_next_stage'):
                del self.solver.boundary._g1_next_stage
            if 'adaptation' in host:
                (self.solver.sediment.adaptation.iterations,self.solver.sediment.adaptation.residual,
                 self.solver.sediment.adaptation_result)=host['adaptation']

    def sync_accepted_clock(self):
        """Call after regrid/restart installs an accepted model clock."""
        if self.live_boundary:
            require(self.solver._glb_stage is None and self.solver._glb_times is None,
                    'Cannot reset accepted live-boundary clock during a trial')
            self.solver._glb_accepted_time=float(self.time_s)

    def step(self, dt, *, snapshot=None, reject_after_active_toe=False):
        require(np.isfinite(dt) and 0 < dt <= float(self.topology.data['dtmax']) and
                self.time_s + dt <= float(self.topology.data['endtime']) + 1e-12,
                'Synchronous bank step exceeds its clock bounds')
        cp = self.cp
        saved = self.snapshot() if snapshot is None else snapshot
        before = self.totals() - self.ledger_totals()
        try:
            result = self.solver.step(dt, self.time_s)
            cfl = float(cp.max(result['cfl']).item())
            if (bool(cp.any(result['status'])) or not np.isfinite(cfl) or
                    cfl > float(self.topology.data['cfl_target']) + 1e-12):
                raise RuntimeError('Synchronous bank CUDA status/CFL rejected')
            if reject_after_active_toe:
                if self.bank_mode == 3:
                    trial = self.solver.bank_trial
                    require(trial is not None and any(bool(cp.any(trial[key]))
                            for key in ('limited', 'removed')),
                            'Late rejection did not reach active distributed bank transport')
                else:
                    require(bool(cp.any(self.solver.toe_face_diagnostic.step_net)),
                            'Late rejection did not reach active toe transport')
                raise RuntimeError('injected late active bank rejection')
            residual = cp.asnumpy(self.totals() - self.ledger_totals() - before)
            if (abs(residual[0]) > 1e-8 or np.max(np.abs(residual[1:])) > 1e-7 or
                    bool(cp.any(self.state['q'][0] < 0)) or
                    bool(cp.any(self.state['layers'] < 0))):
                raise RuntimeError('Synchronous bank conservation/positivity rejected')
            self.accepted_steps += 1
            self.time_s += dt
            self.maximum_cfl = max(self.maximum_cfl, cfl)
            return dict(time_s=self.time_s, step=self.accepted_steps,
                route='synchronous_bank_catalog', budget_residual=residual.tolist(), cfl=cfl)
        except Exception:
            self.rollback(saved)
            raise

    def accepted_gradients(self):
        """Current accepted eta/u/v and signed-intensity gradients, on CUDA."""
        cp, solver = self.cp, self.solver
        bed = solver.bed()
        hydro = solver.reconstruction.reconstruct(self.state['q'])
        has_secondary = int(self.topology.data['secondary_flow_model']) == 2
        secondary = (solver.secondary.reconstruct(self.state['q'], bed, self.state['secondary'])
                     if has_secondary else None)
        require(not bool(cp.any(hydro['primitive_status'])) and
                not bool(cp.any(hydro['reconstruction_status'])) and
                (secondary is None or not bool(cp.any(secondary['status']))),
                'Accepted AMR gradient reconstruction failed')
        result = cp.zeros((self.n, 4 + self.ng + int(has_secondary), 2), dtype=cp.float64)
        result[:, :3] = hydro['grad'][:3].transpose(2, 0, 1)
        # Suspension is explicitly zero in this route. Last primitive is bed.
        if has_secondary:
            result[:, 3 + self.ng] = solver.secondary.buffers['gradient'].T
        result[:, 3 + self.ng + int(has_secondary)] = hydro['grad'][3].T
        return result

    def accepted_bed_gradient(self):
        """Reuse the limited hydraulic bed gradient at current accepted state."""
        cp,solver=self.cp,self.solver
        solver.bed()
        hydro=solver.reconstruction.reconstruct(self.state['q'])
        require(not bool(cp.any(hydro['primitive_status'])) and
                not bool(cp.any(hydro['reconstruction_status'])),
                'Accepted AMR bed-gradient reconstruction failed')
        return cp.ascontiguousarray(hydro['grad'][3].T)

    def frame(self):
        cp = self.cp
        out = dict(time_s=np.float64(self.time_s), node_ids=self.ids.copy(),
            state=cp.asnumpy(self.full()), accepted_steps=np.int64(self.accepted_steps),
            euler_calls=np.int64(self.solver.euler_calls), maximum_cfl=np.float64(self.maximum_cfl))
        out.update({'history_' + name: cp.asnumpy(value)
                    for name, value in self.solver.bank_history.items()})
        out.update({'ledger_' + name: cp.asnumpy(value)
                    for name, value in self.solver.ledger.items()})
        if self.bank_mode == 3:
            out.update(bank_mode=np.int64(3),
                       distributed_stage_index=np.int64(self.solver.distributed_stage_index),
                       distributed_collapse_waves=np.int64(self.solver.distributed_bank.collapse_waves))
        if self.evolving_toe_variant:
            out.update({'toe_map_'+name:cp.asnumpy(value)
                        for name,value in self.solver.evolving_toe.persistent_arrays().items()})
        return out


def validate_lineages(prepared, records):
    """Bind immutable segment IDs to actual selected geometry, before CUDA."""
    require(set(records) == {top.name for top in prepared.topologies},
            'Every catalog operator requires a toe lineage')
    result, expected = {}, None
    for top in prepared.topologies:
        require('toe_geometry_seed_only' not in top.data,
                'Geometry-only toe seeds cannot define immutable bank lineage')
        rows = sorted(records[top.name], key=lambda row: row['candidate_source_face'])
        faces = np.asarray([row['local_face'] for row in rows], dtype=np.int64)
        ids = np.asarray([row['candidate_source_face'] for row in rows], dtype=np.int64)
        require(len(set(map(int, ids))) == len(ids) and np.all(ids > 0) and
                np.array_equal(np.sort(faces), np.flatnonzero(top.data['is_toe_face'])),
                'Missing, duplicate or unexpected bank-toe segments')
        identity = []
        for face, row in zip(faces, rows):
            bank = int(top.node_ids[int(top.data['candidate_bank'][face])])
            toe = int(top.node_ids[int(top.data['candidate_toe'][face])])
            require(bank == row['bank_node'] and toe == row['toe_node'],
                    'Toe lineage does not match selected global bank/receiver nodes')
            identity.append((row['candidate_source_face'], bank, toe,
                *np.asarray(top.data['mid'])[:, face], float(top.data['length'][face])))
        if expected is None:
            expected = identity
        require(identity == expected, 'Physical bank-toe segments changed across catalog')
        result[top.name] = (ids, faces)
    return result


class CatalogBankRoute:
    """One coupled model with atomic multi-group regrid and stable toe history."""

    def __init__(self, prepared, *, source_folder, initial_tree_state, lineages=None,
                 initial_topology=0, evolving_toe_variant=False):
        initial = prepared.topologies[initial_topology]
        self.bank_mode = int(initial.data['bank_toe_erosion_mode'])
        require(self.bank_mode in (2, 3) and all(int(top.data['bank_toe_erosion_mode']) ==
                self.bank_mode for top in prepared.topologies),
                'Catalog bank modes differ or are unsupported')
        require(not (self.bank_mode == 3 and evolving_toe_variant),
                'Distributed bank mode cannot use a physical-toe map')
        self.evolving_toe_variant = bool(evolving_toe_variant)
        self.lineages = ({} if self.evolving_toe_variant or self.bank_mode == 3 else
                         validate_lineages(prepared, lineages))
        state = np.asarray(initial_tree_state)
        require(state.dtype == np.float64 and state.shape ==
                (len(prepared.tree['parent']), int(initial.data['nstate'])) and
                np.isfinite(state).all() and
                np.allclose(state[initial.node_ids], initial.data['state'].T,
                            rtol=0, atol=1e-11),
                'Explicit original full-tree state differs from initial selected inputs')
        self.prepared, self.source = prepared, Path(source_folder).resolve()
        self.model = self._make_stepper(initial)
        self.cp = self.model.cp
        self.tree_state = self.cp.asarray(state)
        self.tree_area = self.cp.asarray(prepared.tree['area'])
        self.groups = []
        parent, level = prepared.tree['parent'], prepared.tree['node_level']
        for lev in range(int(np.max(level)) - 1, -1, -1):
            parents = np.flatnonzero((level == lev) & np.isin(np.arange(len(parent)), parent))
            children = np.asarray([np.flatnonzero(parent == node) for node in parents])
            self.groups.append((self.cp.asarray(parents), self.cp.asarray(children)))
        self.events = []
        self.face_epochs = []
        self.by_leaves = {tuple(map(int, top.node_ids)): top for top in prepared.topologies}
        require(len(self.by_leaves) == len(prepared.topologies), 'Duplicate catalog leaf sets')
        for top in prepared.topologies:
            validate_boundary_handoff(prepared,initial,top)

    def _make_stepper(self, topology):
        return SynchronousBankStepper(self.prepared, topology, source_folder=self.source,
                                      evolving_toe_variant=self.evolving_toe_variant)

    def _handoff_stepper(self, old, new):
        """Carry accepted clocks; alternate steppers own their epoch accounting."""
        new.accepted_steps, new.time_s, new.maximum_cfl = (
            old.accepted_steps, old.time_s, old.maximum_cfl)
        new.solver.euler_calls = old.solver.euler_calls
        if self.bank_mode == 3:
            new.solver.distributed_stage_index = old.solver.distributed_stage_index
            new.solver.distributed_bank.collapse_waves = old.solver.distributed_bank.collapse_waves
        new.sync_accepted_clock()
        return {}

    def synchronized_tree(self):
        """Sync current leaves and ancestors, retaining accepted dormant children."""
        cp = self.cp
        candidate = self.tree_state.copy()
        selected = cp.zeros(len(candidate), dtype=cp.bool_)
        selected[cp.asarray(self.model.ids)] = True
        covered = np.zeros(len(candidate), dtype=bool)
        covered[self.model.ids] = True
        for node, parent in enumerate(self.prepared.tree['parent']):
            if parent >= 0:
                covered[node] |= covered[parent]
        covered = cp.asarray(covered)
        candidate[cp.asarray(self.model.ids)] = self.model.full()
        for parents, children in self.groups:
            values = cp.sum(candidate[children] * self.tree_area[children, None], axis=1)
            restricted = values / self.tree_area[parents, None]
            candidate[parents] = cp.where(covered[parents, None], candidate[parents], restricted)
        return candidate

    def regrid(self, selected_ids, *, reject_after_candidate=False, snapshot=None):
        old = self.model
        saved = old.snapshot() if snapshot is None else snapshot
        try:
            return self._regrid(selected_ids, reject_after_candidate=reject_after_candidate)
        except BaseException:
            old.rollback(saved)
            raise

    def _regrid(self, selected_ids, *, reject_after_candidate=False):
        ids = np.asarray(selected_ids)
        require(ids.ndim == 1 and ids.dtype.kind in 'iu' and np.all(np.diff(ids) > 0),
                'Proposed leaves must be sorted unique global IDs')
        key = tuple(map(int, ids))
        require(key in self.by_leaves, 'State-selected AMR leaf set is absent from frozen catalog')
        target = self.by_leaves[key]
        old, cp = self.model, self.cp
        if target is old.topology:
            return None
        validate_boundary_handoff(self.prepared,old.topology,target)
        plan = plan_mixed_transition(self.prepared, old.topology, target)
        tree = self.synchronized_tree()
        gradients = old.accepted_gradients() if plan.refine_parents else None
        old_map = {int(node): i for i, node in enumerate(old.ids)}
        candidate = propose_mixed_transition(self.prepared, plan, old.full(),
            refinement_children={p:cp.ascontiguousarray(tree[cp.asarray(group)])
                for p, group in zip(plan.refine_parents, plan.refine_children)},
            primitive_gradients={p:cp.ascontiguousarray(gradients[old_map[p]])
                for p in plan.refine_parents}, source_folder=self.source,
            bank_history=(None if self.evolving_toe_variant else _history_matrix(old.solver)),
            bank_collapse_count=(None if self.evolving_toe_variant else old.solver.bank_history[COUNT]),
            transfer_bank_history=not self.evolving_toe_variant)
        new = self._make_stepper(target)
        new.install(candidate.state)
        extra_receipt, archived = {}, None
        if self.evolving_toe_variant:
            from .amr_evolving_bank import transfer_evolving_bank
            extra_receipt, archived = transfer_evolving_bank(self.prepared,plan,old,new,
                                                            source_folder=self.source)
        else:
            _install_history(new.solver, candidate.bank_history, candidate.bank_collapse_count)
        for name, value in old.solver.ledger.items():
            require(name in new.solver.ledger and new.solver.ledger[name].shape==value.shape and
                    new.solver.ledger[name].dtype==value.dtype,
                    'Regrid boundary ledger shape/type changed: '+name)
            cp.copyto(new.solver.ledger[name], value)
        segment_ids=np.empty(0,np.int64)
        if self.bank_mode == 2 and not self.evolving_toe_variant:
            segment_ids, old_faces = self.lineages[old.topology.name]
            new_ids, new_faces = self.lineages[target.name]
            require(np.array_equal(segment_ids, new_ids), 'Toe segment identity changed')
            inactive = np.flatnonzero(np.asarray(old.topology.data['is_toe_face']) == 0)
            for name in ('toe_face_signed', 'toe_face_gross'):
                value = getattr(old.solver, name)
                require(not bool(cp.any(value[:, cp.asarray(inactive)])),
                        'Non-toe face carries cumulative physical-toe history')
                getattr(new.solver, name)[:, cp.asarray(new_faces)] = value[:, cp.asarray(old_faces)]
        extra_receipt.update(self._handoff_stepper(old, new))
        if reject_after_candidate:
            raise RuntimeError('injected late coupled bank regrid rejection')
        tree[cp.asarray(new.ids)] = candidate.state
        event = dict(time_s=old.time_s, step=old.accepted_steps,
            old_topology=old.topology.name, new_topology=target.name,
            stable_toe_segment_ids=segment_ids.tolist(), **candidate.receipt)
        event.update(extra_receipt)
        # Only this final block commits topology, state, bank and toe history.
        self.tree_state, self.model = tree, new
        if archived is not None:
            self.face_epochs.append(archived)
        self.events.append(event)
        return event

    def advance(self, dt, *, selector=None, delay_state=None, regrid=False,
                reject_after_candidate=False):
        """Commit a step and optional regrid together, or restore both exactly."""
        old = self.model
        saved = old.snapshot()
        try:
            step = old.step(dt, snapshot=saved)
            proposal = event = None
            if regrid:
                require(selector is not None and delay_state is not None,
                        'Accepted-state AMR selector and delay state are required')
                proposal = selector.propose(old.topology, old.full(),
                    cached_tree_state=self.synchronized_tree(),
                    delay_state=delay_state, time_s=old.time_s)
                event = self.regrid(np.asarray(proposal.node_ids, dtype=np.int64),
                    reject_after_candidate=reject_after_candidate, snapshot=saved)
            return step, event, proposal
        except BaseException:
            old.rollback(saved)
            raise

    def frame(self):
        out = self.model.frame()
        if self.bank_mode == 3:
            out['topology_identity'] = np.asarray(self.model.topology.source_identity)
            return out
        if self.evolving_toe_variant:
            out.update(toe_signed=self.cp.asnumpy(self.model.solver.toe_face_signed),
                toe_gross=self.cp.asnumpy(self.model.solver.toe_face_gross),
                topology_identity=np.asarray(self.model.topology.source_identity))
            # Epochs are immutable closed diagnostic interfaces, not extra
            # sediment inventory. Checkpointing retains their topology identity.
            out['face_epoch_count']=np.int64(len(self.face_epochs))
            for i,epoch in enumerate(self.face_epochs):
                for key,value in epoch.items():
                    out[f'face_epoch_{i}_{key}']=(self.cp.asnumpy(value)
                        if isinstance(value,self.cp.ndarray) else np.asarray(value))
            return out
        ids, faces = self.lineages[self.model.topology.name]
        out.update(toe_segment_ids=ids.copy(),
            toe_signed=self.cp.asnumpy(self.model.solver.toe_face_signed[:, self.cp.asarray(faces)]),
            toe_gross=self.cp.asnumpy(self.model.solver.toe_face_gross[:, self.cp.asarray(faces)]),
            topology_identity=np.asarray(self.model.topology.source_identity))
        return out
