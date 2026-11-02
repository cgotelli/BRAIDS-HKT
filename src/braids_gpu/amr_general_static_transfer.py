"""Conservative CUDA array transfer for native hydraulic-only inert beds.

The general layered transfer requires an active and at least one substrate
layer.  An actual hydraulic-only BRAIDS deck has one inert bed row instead.
This narrow transfer keeps that row and its original child relief, while
redistributing water and momentum across 2:1 children on the GPU.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .amr_catalog_mixed_transition import (
    MixedTransitionCandidate, MixedTransitionPlan, plan_mixed_transition)
from .amr_general_stepper import require
from .general_live_transport import static_bed_enabled


def propose_static_transition(prepared, plan: MixedTransitionPlan,
                              accepted_state, *, refinement_children,
                              primitive_gradients, reject_after_candidate=False):
    """Return a private all-five-row candidate; never mutate accepted arrays."""
    import cupy as cp

    require(prepared.schema == 3 and isinstance(plan, MixedTransitionPlan) and
            static_bed_enabled(plan.old.data) and
            static_bed_enabled(plan.new.data) and
            int(plan.old.data['nfrac']) == int(plan.new.data['nfrac']) == 1 and
            int(plan.old.data['nstate']) == int(plan.new.data['nstate']) == 5 and
            int(plan.old.data['exchange_mode']) == int(plan.new.data['exchange_mode']) == 0 and
            int(plan.old.data['secondary_flow_model']) ==
            int(plan.new.data['secondary_flow_model']) == 0 and
            int(plan.old.data['bedlaw']) == int(plan.new.data['bedlaw']) == 0,
            'Static AMR transfer requires the original inert single bed layer')
    expected = plan_mixed_transition(prepared, plan.old, plan.new)
    require(plan.old is expected.old and plan.new is expected.new and
            all(getattr(plan, key) == getattr(expected, key) for key in
                ('retained_ids', 'coarsen_parents', 'coarsen_children',
                 'refine_parents', 'refine_children')),
            'Static AMR plan differs from authenticated complete groups')
    old, new = plan.old, plan.new
    require(isinstance(accepted_state, cp.ndarray) and
            accepted_state.shape == (len(old.node_ids), 5) and
            accepted_state.dtype == cp.float64 and
            accepted_state.flags.c_contiguous and
            bool(cp.all(cp.isfinite(accepted_state))) and
            bool(cp.all(accepted_state[:, [0, 3, 4]] >= 0)),
            'Static AMR accepted state differs')
    require(set(refinement_children) == set(plan.refine_parents) and
            set(primitive_gradients) == set(plan.refine_parents),
            'Static AMR refinement requires accepted child cache and gradients')
    old_map = {int(node): i for i, node in enumerate(old.node_ids)}
    new_map = {int(node): i for i, node in enumerate(new.node_ids)}
    candidate = cp.empty((len(new.node_ids), 5), cp.float64)
    oi = cp.asarray([old_map[node] for node in plan.retained_ids])
    ni = cp.asarray([new_map[node] for node in plan.retained_ids])
    candidate[ni] = accepted_state[oi]
    area_old = cp.asarray(old.data['area'])
    area_new = cp.asarray(new.data['area'])

    for parent, children in zip(plan.coarsen_parents, plan.coarsen_children):
        indices = cp.asarray([old_map[child] for child in children])
        weighted = cp.sum(area_old[indices, None]*accepted_state[indices], axis=0)
        candidate[new_map[parent]] = weighted/area_new[new_map[parent]]

    bulk = float(old.data['bulk_fraction'][0])
    dry = float(old.data['dry_tolerance'])
    for parent, children in zip(plan.refine_parents, plan.refine_children):
        cached = refinement_children[parent]
        gradient = primitive_gradients[parent]
        require(isinstance(cached, cp.ndarray) and cached.shape == (4, 5) and
                cached.dtype == cp.float64 and cached.flags.c_contiguous and
                isinstance(gradient, cp.ndarray) and gradient.shape == (5, 2) and
                gradient.dtype == cp.float64 and gradient.flags.c_contiguous and
                bool(cp.all(cp.isfinite(cached))) and
                bool(cp.all(cp.isfinite(gradient))) and
                bool(cp.all(cached[:, [0, 3, 4]] >= 0)),
                'Static AMR child cache/primitive gradient differs')
        parent_row = accepted_state[old_map[parent]]
        indices = cp.asarray([new_map[child] for child in children])
        weights = area_new[indices]
        pa = area_old[old_map[parent]]
        position = cp.asarray(prepared.tree['xy'][list(children)] -
                              prepared.tree['xy'][parent])
        # The inert bed is original accepted child relief.  A uniform mass
        # correction enforces its parent integral without creating a layer.
        bed_mass = cached[:, 4] + (parent_row[4] -
            cp.sum(weights*cached[:, 4])/pa)
        require(bool(cp.all(bed_mass >= -1e-12)),
                'Static AMR child bed cannot inherit parent inert material')
        bed_mass = cp.maximum(bed_mass, 0.)
        child_bed = cp.asarray(new.data['rock'])[indices] + bed_mass/bulk
        parent_bed = (cp.asarray(old.data['rock'])[old_map[parent]] +
                      parent_row[4]/bulk)
        eta = parent_bed + parent_row[0] + cp.sum(position*gradient[0], axis=1)
        depth = cp.maximum(eta-child_bed, 0.)
        weighted_depth = cp.sum(weights*depth)/pa
        depth = cp.where(weighted_depth > 0,
                         depth*parent_row[0]/cp.maximum(weighted_depth, 1e-300),
                         parent_row[0])
        speed = cp.where(parent_row[0] > dry,
                         parent_row[1:3]/cp.maximum(parent_row[0], 1e-300), 0.)
        speed = speed[None, :] + position@gradient[1:3].T
        momentum = depth[:, None]*speed
        correction = parent_row[1:3] - cp.sum(weights[:, None]*momentum, axis=0)/pa
        momentum += depth[:, None]*correction[None, :]/cp.maximum(parent_row[0], 1e-300)
        result = cp.empty((4, 5), cp.float64)
        result[:, 0] = depth
        result[:, 1:3] = momentum
        result[:, 3] = 0.
        result[:, 4] = bed_mass
        candidate[indices] = result

    require(bool(cp.all(cp.isfinite(candidate))) and
            bool(cp.all(candidate[:, [0, 3, 4]] >= 0)) and
            not bool(cp.any(candidate[candidate[:, 0] == 0, 1:3])) and
            bool(cp.array_equal(candidate[ni], accepted_state[oi])),
            'Static AMR candidate is invalid or changes retained cells')
    residual = cp.asnumpy(cp.sum(area_new[:, None]*candidate, axis=0) -
                           cp.sum(area_old[:, None]*accepted_state, axis=0))
    require(abs(residual[0]) <= 1e-10 and
            np.max(np.abs(residual[1:])) <= 1e-8,
            'Static AMR state transfer is not conservative')
    if reject_after_candidate:
        raise RuntimeError('injected late static AMR candidate rejection')
    receipt = dict(catalog_identity=prepared.identity,
        transfer_source_sha256={'amr_general_static_transfer.py':
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        old_operator_identity=old.source_identity,
        new_operator_identity=new.source_identity,
        old_leaves=len(old.node_ids), new_leaves=len(new.node_ids),
        retained_nodes=len(plan.retained_ids),
        coarsened_parent_ids=list(plan.coarsen_parents),
        refined_parent_ids=list(plan.refine_parents),
        state_row_residuals=residual.tolist(),
        bank_mean_history_residuals=None,
        all_state_and_history_rows_transferred_on_CUDA=True,
        candidate_only_no_accepted_mutation=True,
        refinement_gradients='explicit_caller_supplied',
        bank_history_rule='not_applicable_no_bank',
        inert_single_bed_layer=True)
    return MixedTransitionCandidate(candidate, None, None, receipt)
