"""Atomic, GPU-resident transfer between authenticated AMR catalog layouts.

This is a transfer component, not a regrid selector or a coupled stepper.
The caller supplies current accepted rows, cached child rows and primitive
gradients. Zero-step topology data supplies geometry only; it is never used as
future numerical state. A returned candidate is not installed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .amr_prepared import AMRTopology, PreparedAMR


@dataclass(frozen=True)
class MixedTransitionPlan:
    old: AMRTopology
    new: AMRTopology
    retained_ids: tuple[int, ...]
    coarsen_parents: tuple[int, ...]
    coarsen_children: tuple[tuple[int, ...], ...]
    refine_parents: tuple[int, ...]
    refine_children: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MixedTransitionCandidate:
    state: object
    bank_history: object | None
    bank_collapse_count: object | None
    receipt: dict


def plan_mixed_transition(prepared: PreparedAMR, old: AMRTopology,
                          new: AMRTopology) -> MixedTransitionPlan:
    """Partition one catalog change into complete, disjoint 2:1 leaf groups."""
    if old is new or not any(top is old for top in prepared.topologies) or not any(
            top is new for top in prepared.topologies):
        raise ValueError('Both distinct topologies must belong to the authenticated catalog')
    old_ids = set(map(int, old.node_ids))
    new_ids = set(map(int, new.node_ids))
    removed, added = old_ids-new_ids, new_ids-old_ids
    parent = np.asarray(prepared.tree['parent'])

    def children(node):
        group = tuple(map(int, np.flatnonzero(parent == node)))
        return group if len(group) == 4 else ()

    coarsen = {node: children(node) for node in added
               if children(node) and set(children(node)) <= removed}
    refine = {node: children(node) for node in removed
              if children(node) and set(children(node)) <= added}
    coarsen = dict(sorted(coarsen.items()))
    refine = dict(sorted(refine.items()))
    if (not coarsen and not refine or
            removed != (set(refine) | {c for group in coarsen.values() for c in group}) or
            added != (set(coarsen) | {c for group in refine.values() for c in group}) or
            len(removed) != len(refine)+4*len(coarsen) or
            len(added) != len(coarsen)+4*len(refine)):
        raise ValueError('Catalog change is not a partition of complete one-level groups')
    old_map = {int(node): i for i, node in enumerate(old.node_ids)}
    new_map = {int(node): i for i, node in enumerate(new.node_ids)}
    retained = tuple(sorted(old_ids & new_ids))
    if not np.array_equal(old.data['rock'][[old_map[n] for n in retained]],
                          new.data['rock'][[new_map[n] for n in retained]]):
        raise ValueError('Retained catalog material changed')
    if prepared.schema == 2:
        old_mask = np.asarray(old.data['bank_material_mask'])
        new_mask = np.asarray(new.data['bank_material_mask'])
        if (not np.array_equal(old_mask[[old_map[n] for n in retained]],
                               new_mask[[new_map[n] for n in retained]]) or
                any(not np.all(old_mask[[old_map[c] for c in group]] ==
                               new_mask[new_map[node]])
                    for node, group in coarsen.items()) or
                any(not np.all(new_mask[[new_map[c] for c in group]] ==
                               old_mask[old_map[node]])
                    for node, group in refine.items())):
            raise ValueError('Bank material differs across catalog transfer')
    return MixedTransitionPlan(old, new, retained, tuple(coarsen),
        tuple(coarsen.values()), tuple(refine), tuple(refine.values()))


def _geometry(prepared, topology, parents, groups):
    ids = {int(node): i for i, node in enumerate(topology.node_ids)}
    tree = prepared.tree
    group_index = np.asarray([[ids[c] for c in group] for group in groups],
                             dtype=np.int64)
    child_ids = np.asarray(groups, dtype=np.int64)
    weights = np.ascontiguousarray(tree['area'][child_ids], dtype=np.float64)
    parent_area = np.ascontiguousarray(tree['area'][list(parents)], dtype=np.float64)
    rock = np.ascontiguousarray(topology.data['rock'][group_index], dtype=np.float64)
    position = np.ascontiguousarray(tree['xy'][child_ids]-
        tree['xy'][np.asarray(parents, dtype=np.int64), None], dtype=np.float64)
    return group_index, weights, parent_area, rock, position


def _placeholder(topology, count):
    nstate = int(topology.data['nstate'])
    seed = np.zeros((count, 4, nstate), dtype=np.float64)
    seed[:, :, 0] = max(float(topology.data['dry_tolerance']), 1e-4)
    seed[:, :, np.asarray(topology.data['bed_indices']).ravel()] = .01
    return seed


def _device_rows(cp, value, shape, name, *, dtype=None, nonnegative=False):
    dtype = cp.float64 if dtype is None else dtype
    if (not isinstance(value, cp.ndarray) or value.shape != shape or
            value.dtype != dtype or not value.flags.c_contiguous or
            (dtype == cp.float64 and not bool(cp.all(cp.isfinite(value)))) or
            (nonnegative and bool(cp.any(value < 0)))):
        raise ValueError('Invalid resident '+name)
    return value


def _physical_state(cp, rows, sediment, suspended, secondary_index, name,
                    *, dry_tolerance=0.0):
    if (bool(cp.any(rows[...,0] < -1e-12)) or
            bool(cp.any(rows[...,cp.asarray(sediment)] < -1e-12))):
        raise ValueError('Invalid resident '+name+' water/sediment')
    dry = rows[...,0] <= dry_tolerance
    if (bool(cp.any((rows[...,1] != 0) & dry)) or
            bool(cp.any((rows[...,2] != 0) & dry)) or
            bool(cp.any((rows[...,cp.asarray(suspended)] != 0) & dry[...,None])) or
            (secondary_index is not None and
             bool(cp.any((rows[...,secondary_index] != 0) & dry)))):
        raise ValueError('Dry resident '+name+' carries momentum/suspension/secondary')


def propose_mixed_transition(prepared: PreparedAMR,
                             plan: MixedTransitionPlan, accepted_state, *,
                             refinement_children: dict[int, object],
                             primitive_gradients: dict[int, object],
                             source_folder, bank_history=None,
                             bank_collapse_count=None,
                             transfer_bank_history=True,
                             suspended_wetdry_source=None,
                             reject_after_candidate=False) -> MixedTransitionCandidate:
    """Build and validate one candidate; accepted device arrays stay untouched.

    ``refinement_children`` and ``primitive_gradients`` are resident arrays
    keyed by global parent ID. The child rows are accepted caches or explicit
    caller seeds, never silently read from the target prepared topology.
    """
    import cupy as cp
    from braids_support.gpu_fwave.amr_transfer import ResidentAMRTransfer

    if not isinstance(plan, MixedTransitionPlan):
        raise ValueError('AMR mixed transition plan differs from catalog')
    expected = plan_mixed_transition(prepared, plan.old, plan.new)
    if (plan.old is not expected.old or plan.new is not expected.new or
            any(getattr(plan,key) != getattr(expected,key) for key in (
                'retained_ids','coarsen_parents','coarsen_children',
                'refine_parents','refine_children'))):
        raise ValueError('AMR mixed transition plan differs from catalog')
    old, new = plan.old, plan.new
    nstate = int(old.data['nstate'])
    nfrac = int(old.data['nfrac'])
    if (int(new.data['nstate']) != nstate or int(new.data['nfrac']) != nfrac or
            not np.array_equal(old.data['bulk_fraction'],new.data['bulk_fraction'])):
        raise ValueError('AMR state or grain layout changed')
    _device_rows(cp, accepted_state, (len(old.node_ids),nstate),
                 'accepted state')
    suspended = np.asarray(old.data['suspended_indices']).ravel()
    sediment = np.concatenate((suspended,
                               np.asarray(old.data['bed_indices']).ravel()))
    secondary_index = (int(old.data['secondary_index']) if
                       int(old.data['secondary_flow_model']) == 2 else None)
    m2c = ('suspended_amr_contract' in old.data or
           'suspended_amr_contract' in new.data)
    if m2c:
        from .implicit_suspension_amr import contract_from_data
        old_contract = contract_from_data(old.data)
        new_contract = contract_from_data(new.data)
        if old_contract != new_contract:
            raise ValueError('M2-C transition ownership instance changed')
        dry_tolerance = old_contract.hydraulic_dry_tolerance_m
    else:
        dry_tolerance = 0.0
    _physical_state(cp,accepted_state,sediment,suspended,secondary_index,
                    'accepted state',dry_tolerance=dry_tolerance)
    if (set(refinement_children) != set(plan.refine_parents) or
            set(primitive_gradients) != set(plan.refine_parents)):
        raise ValueError('Every refined parent requires explicit child rows and gradients')
    source = Path(source_folder).resolve()
    for filename in ('amr_transfer.cu','amr_bank_history.cu',
                     'amr_bank_history_prolong.cu'):
        if not (source/filename).is_file():
            raise ValueError('Missing AMR CUDA source: '+filename)
    source_sha256 = {filename:hashlib.sha256((source/filename).read_bytes()).hexdigest()
        for filename in ('amr_transfer.cu','amr_bank_history.cu',
                         'amr_bank_history_prolong.cu')}
    old_map = {int(node): i for i,node in enumerate(old.node_ids)}
    new_map = {int(node): i for i,node in enumerate(new.node_ids)}
    # CuPy infers FP64 for an empty Python list; explicitly retain integer
    # indexing so a complete four-child <-> parent transition (no retained
    # leaves) is a valid transaction too.
    retained_old = cp.asarray([old_map[n] for n in plan.retained_ids],
                              dtype=cp.int64)
    retained_new = cp.asarray([new_map[n] for n in plan.retained_ids],
                              dtype=cp.int64)
    candidate = cp.empty((len(new.node_ids),nstate),dtype=cp.float64)
    candidate[retained_new] = accepted_state[retained_old]
    bulk = np.asarray(old.data['bulk_fraction'],dtype=np.float64)
    dry = float(old.data['dry_tolerance'])
    secondary = int(old.data['secondary_flow_model']) == 2
    coarsen_weights = None

    if plan.coarsen_parents:
        group_ix, weights, parent_area, rock, position = _geometry(
            prepared, old, plan.coarsen_parents, plan.coarsen_children)
        coarsen_weights = weights
        operator = ResidentAMRTransfer(_placeholder(old,len(group_ix)),
            weights,parent_area,rock,position,bulk,dry=dry,
            prognostic_secondary=secondary,source_path=source/'amr_transfer.cu')
        cp.copyto(operator.accepted,cp.ascontiguousarray(
            accepted_state[cp.asarray(group_ix)]))
        parent_rows = operator.restrict()
        candidate[cp.asarray([new_map[n] for n in plan.coarsen_parents])] = parent_rows

    if plan.refine_parents:
        group_ix, weights, parent_area, rock, position = _geometry(
            prepared, new, plan.refine_parents, plan.refine_children)
        primitives = 4+nfrac+int(secondary)
        seed_rows = []
        gradient_rows = []
        for parent in plan.refine_parents:
            seed = _device_rows(cp, refinement_children[parent],
                (4,nstate),'refinement child cache')
            _physical_state(cp,seed,sediment,suspended,secondary_index,
                            'refinement child cache',
                            dry_tolerance=dry_tolerance)
            seed_rows.append(seed)
            gradient_rows.append(_device_rows(cp,primitive_gradients[parent],
                (primitives,2),'refinement primitive gradient'))
        seeds = cp.ascontiguousarray(cp.stack(seed_rows))
        gradients = cp.ascontiguousarray(cp.stack(gradient_rows))
        operator = ResidentAMRTransfer(_placeholder(new,len(group_ix)),
            weights,parent_area,rock,position,bulk,dry=dry,
            prognostic_secondary=secondary,source_path=source/'amr_transfer.cu')
        cp.copyto(operator.accepted,seeds)
        parents = cp.ascontiguousarray(accepted_state[cp.asarray(
            [old_map[n] for n in plan.refine_parents])])
        children,status = operator.propose(parents,gradients)
        if bool(cp.any(status != 0)):
            operator.rollback()
            raise RuntimeError('Mixed AMR refinement kernel rejected')
        candidate[cp.asarray(group_ix)] = children
        operator.rollback()

    phase_change = cp.zeros(nfrac, dtype=cp.float64)
    safety_delta = cp.zeros(nstate, dtype=cp.float64)
    if m2c:
        from .implicit_suspension_amr import ResidentSuspendedWetDryOwnership
        if suspended_wetdry_source is None:
            raise ValueError('M2-C transition requires explicit wet/dry CUDA source')
        wetdry_source = Path(suspended_wetdry_source).resolve()
        if not wetdry_source.is_file():
            raise ValueError('Missing M2-C wet/dry CUDA source')
        active = np.asarray(old.data['bed_indices'])[0].ravel()
        ownership = ResidentSuspendedWetDryOwnership(
            np.asarray(new.data['area']), nfrac, dry_tolerance,
            source_path=wetdry_source)
        result = ownership.propose(
            cp.ascontiguousarray(candidate[:, 0]),
            cp.ascontiguousarray(candidate[:, cp.asarray(suspended)].T),
            cp.ascontiguousarray(candidate[:, cp.asarray(active)].T))
        if bool(cp.any(result['status'] != 0)):
            raise RuntimeError('M2-C transition wet/dry ownership rejected')
        candidate[:, cp.asarray(suspended)] = result['suspended'].T
        candidate[:, cp.asarray(active)] = result['active_bed'].T
        phase_change = cp.sum(result['suspended_phase_change_kg'], axis=1)
        new_area = cp.asarray(new.data['area'])
        shallow_dry = candidate[:, 0] <= dry_tolerance
        for row in (1, 2, *(() if secondary_index is None else
                              (secondary_index,))):
            before_cleanup = cp.sum(new_area*candidate[:, row])
            candidate[shallow_dry, row] = 0.0
            safety_delta[row] = (cp.sum(new_area*candidate[:, row])-
                                 before_cleanup)

    if (not bool(cp.all(cp.isfinite(candidate))) or
            bool(cp.any(candidate[:,0] < -1e-12))):
        raise RuntimeError('Mixed AMR candidate has invalid water/state')
    if bool(cp.any(candidate[:,cp.asarray(sediment)] < -1e-12)):
        raise RuntimeError('Mixed AMR candidate has negative sediment')
    if not bool(cp.array_equal(candidate[retained_new],accepted_state[retained_old])):
        raise RuntimeError('Mixed AMR retained state changed')
    old_area = cp.asarray(old.data['area'])
    new_area = cp.asarray(new.data['area'])
    residual = cp.asnumpy(cp.sum(new_area[:,None]*candidate,axis=0)-
                           cp.sum(old_area[:,None]*accepted_state,axis=0))
    safety_host = cp.asnumpy(safety_delta)
    # Restriction/prolongation is conservative before ownership.  The M2-C
    # post-transfer ownership operation is a *phase* move, however: each
    # suspended row loses ``phase_change`` and the corresponding active-bed
    # row gains its negative.  Checking every sediment row against zero would
    # therefore reject exactly the shallow-dry repair this transaction owns.
    # Momentum (and prognostic secondary state, when present) has a separate
    # explicit safety delta because those rows are not sediment inventories.
    expected_residual = safety_host.copy()
    phase_host = cp.asnumpy(phase_change)
    if m2c:
        active = np.asarray(old.data['bed_indices'])[0].ravel()
        expected_residual[suspended] += phase_host
        expected_residual[active] -= phase_host
    class_total_residual = np.asarray([
        residual[int(suspended[g])] +
        np.sum(residual[np.asarray(old.data['bed_indices'])[:, g]])
        for g in range(nfrac)], dtype=np.float64)
    if (np.max(np.abs(residual-expected_residual)) > 1e-8 or
            np.max(np.abs(class_total_residual)) > 1e-8):
        raise RuntimeError('Mixed AMR state transfer is not conservative')

    history_candidate = count_candidate = None
    history_residual = None
    if prepared.schema == 2 and transfer_bank_history:
        from braids_support.gpu_fwave.amr_bank_history import ResidentAMRBankHistory
        from braids_support.gpu_fwave.amr_bank_history_prolong import ResidentAMRBankHistoryProlong

        history = _device_rows(cp,bank_history,(len(old.node_ids),6),
                               'accepted bank history',nonnegative=True)
        counts = _device_rows(cp,bank_collapse_count,(len(old.node_ids),),
                              'accepted bank collapse count',dtype=cp.int32,
                              nonnegative=True)
        total = len(prepared.tree['parent'])
        tree_history = cp.zeros((total,6),dtype=cp.float64)
        tree_counts = cp.zeros(total,dtype=cp.int32)
        tree_history[cp.asarray(old.node_ids)] = history
        tree_counts[cp.asarray(old.node_ids)] = counts
        cover = np.full(total,-1,dtype=np.int32)
        for node in plan.retained_ids:
            cover[node] = node
        if plan.coarsen_parents:
            group_ix = np.asarray([[old_map[c] for c in group]
                                   for group in plan.coarsen_children])
            operator = ResidentAMRBankHistory(coarsen_weights,
                cp.zeros((len(group_ix),6),dtype=cp.float64),
                cp.zeros(len(group_ix),dtype=cp.int32),
                source_path=source/'amr_bank_history.cu')
            restricted,restricted_counts,status = operator.propose(
                cp.ascontiguousarray(history[cp.asarray(group_ix)]),
                cp.ascontiguousarray(counts[cp.asarray(group_ix)]))
            if bool(cp.any(status != 0)):
                operator.rollback()
                raise RuntimeError('Mixed AMR bank restriction rejected')
            tree_history[cp.asarray(plan.coarsen_parents)] = restricted
            tree_counts[cp.asarray(plan.coarsen_parents)] = restricted_counts
            operator.rollback()
            for node in plan.coarsen_parents:
                cover[node] = node
        for node,group in zip(plan.refine_parents,plan.refine_children):
            for child in group:
                cover[child] = node
        old_leaf = np.zeros(total,dtype=np.int32)
        old_leaf[old.node_ids] = 1
        old_leaf[list(plan.coarsen_parents)] = 1
        selected = np.zeros(total,dtype=np.int32)
        selected[new.node_ids] = 1
        operator = ResidentAMRBankHistoryProlong(tree_history,tree_counts,
            old_leaf,source_path=source/'amr_bank_history_prolong.cu')
        proposed,proposed_counts,status = operator.propose(selected,cover)
        if bool(cp.any(status != 0)):
            operator.rollback()
            raise RuntimeError('Mixed AMR bank inheritance rejected')
        history_candidate = cp.ascontiguousarray(proposed[cp.asarray(new.node_ids)])
        count_candidate = cp.ascontiguousarray(proposed_counts[cp.asarray(new.node_ids)])
        operator.rollback()
        if (not bool(cp.array_equal(history_candidate[retained_new],
                                    history[retained_old])) or
                not bool(cp.array_equal(count_candidate[retained_new],
                                        counts[retained_old]))):
            raise RuntimeError('Mixed AMR retained bank history changed')
        history_residual = cp.asnumpy(cp.sum(new_area[:,None]*history_candidate[:,3:6],axis=0)-
                                         cp.sum(old_area[:,None]*history[:,3:6],axis=0))
        if np.max(np.abs(history_residual)) > 1e-10:
            raise RuntimeError('Mixed AMR mean bank history changed inventory')
    elif bank_history is not None or bank_collapse_count is not None:
        raise ValueError('Physical-only AMR transfer cannot accept bank histories')

    if reject_after_candidate:
        raise RuntimeError('injected late mixed AMR candidate rejection')
    receipt = dict(catalog_identity=prepared.identity,
        transfer_source_sha256=source_sha256,
        old_operator_identity=old.source_identity,
        new_operator_identity=new.source_identity,
        old_leaves=len(old.node_ids),new_leaves=len(new.node_ids),
        retained_nodes=len(plan.retained_ids),
        coarsened_parent_ids=list(plan.coarsen_parents),
        refined_parent_ids=list(plan.refine_parents),
        state_row_residuals=residual.tolist(),
        expected_state_row_residuals=expected_residual.tolist(),
        sediment_class_total_residuals_kg=class_total_residual.tolist(),
        wetdry_momentum_secondary_safety_delta=safety_host.tolist(),
        wetdry_suspended_phase_change_kg=phase_host.tolist(),
        wetdry_predicate=('h_le_hydraulic_dry_tolerance' if m2c else
                          'legacy_h_eq_zero'),
        bank_mean_history_residuals=(None if history_residual is None else
                                     history_residual.tolist()),
        all_state_and_history_rows_transferred_on_CUDA=True,
        candidate_only_no_accepted_mutation=True,
        refinement_gradients='explicit_caller_supplied',
        bank_history_rule=('native_max_area_current_parent' if transfer_bank_history
                           and prepared.schema == 2 else 'separate_interface_remap_required'))
    return MixedTransitionCandidate(candidate,history_candidate,count_candidate,
                                    receipt)
