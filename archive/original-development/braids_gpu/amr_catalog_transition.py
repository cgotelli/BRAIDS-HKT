"""GPU-resident conservative transfers between authenticated AMR topologies.

The transition is geometry-generic: it discovers complete four-child groups
from the prepared hierarchy. Refinement requires accepted cached children and
currently uses zero primitive gradient; it is a bounded research route, not a
general native-equivalent AMR reconstruction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .amr_prepared import AMRTopology, PreparedAMR


def coarsen_selected_state(prepared: PreparedAMR, old: AMRTopology,
                           new: AMRTopology, accepted_full, *,
                           transfer_source):
    """Restrict all conserved rows on CUDA, leaving input accepted bytes intact.

    ``accepted_full`` is a contiguous device matrix ``(old_leaves,nstate)``.
    The returned matrix is also resident; no physical state is obtained from a
    future CPU frame or copied to host to perform the transfer.
    """
    import cupy as cp
    from validation.gpu_fwave.amr_transfer import ResidentAMRTransfer

    if (not any(item is old for item in prepared.topologies) or
            not any(item is new for item in prepared.topologies)):
        raise ValueError('AMR topology is not in the authenticated catalog')
    if (not isinstance(accepted_full, cp.ndarray) or
            accepted_full.shape != (len(old.node_ids), int(old.data['nstate'])) or
            accepted_full.dtype != cp.float64 or
            not accepted_full.flags.c_contiguous):
        raise ValueError('Accepted AMR state must be resident contiguous FP64')
    source = Path(transfer_source).resolve()
    if not source.is_file():
        raise ValueError('AMR transfer CUDA source is missing')
    parent = prepared.tree['parent']
    area = prepared.tree['area']
    xy = prepared.tree['xy']
    old_map = {int(node): position for position, node in enumerate(old.node_ids)}
    new_map = {int(node): position for position, node in enumerate(new.node_ids)}
    retained = sorted(set(old_map) & set(new_map))
    removed = set(old_map)-set(new_map)
    parents = sorted(set(new_map)-set(old_map))
    if not parents or not removed:
        raise ValueError('AMR transition requires at least one coarsened parent')
    groups = [np.flatnonzero(parent == node).astype(np.int64) for node in parents]
    if (any(len(group) != 4 or not set(group).issubset(old_map)
            for group in groups) or
            set(np.concatenate(groups).tolist()) != removed):
        raise ValueError('AMR catalog transition is not complete four-child coarsening')
    group_index = np.asarray([[old_map[int(child)] for child in group]
                              for group in groups], dtype=np.int64)
    weights = np.asarray(area[np.concatenate(groups)].reshape(-1, 4),
                         dtype=np.float64)
    positions = np.asarray([[xy[int(child)]-xy[node] for child in group]
                            for node, group in zip(parents, groups)],
                           dtype=np.float64)
    rock = np.asarray(old.data['rock'][group_index], dtype=np.float64)
    nstate = int(old.data['nstate'])
    nfrac = int(old.data['nfrac'])
    bulk = np.asarray(old.data['bulk_fraction'], dtype=np.float64)
    if (int(new.data['nstate']) != nstate or int(new.data['nfrac']) != nfrac or
            not np.array_equal(new.data['bulk_fraction'], bulk)):
        raise ValueError('AMR coarsening cannot change state/material layout')
    # Only allocate a valid device buffer here.  Current accepted child rows,
    # never host placeholders or native future states, overwrite it on CUDA.
    placeholder = np.zeros((len(parents), 4, nstate), dtype=np.float64)
    placeholder[:, :, 0] = max(float(old.data['dry_tolerance']), 1e-4)
    for row in np.asarray(old.data['bed_indices']).ravel():
        placeholder[:, :, int(row)] = .01
    transfer = ResidentAMRTransfer(np.ascontiguousarray(placeholder),
        weights, np.asarray(area[parents]), rock, positions, bulk,
        dry=float(old.data['dry_tolerance']),
        prognostic_secondary=int(old.data['secondary_flow_model']) == 2,
        source_path=source)
    cp.copyto(transfer.accepted,
              cp.ascontiguousarray(accepted_full[cp.asarray(group_index)]))
    restricted = transfer.restrict()
    candidate = cp.empty((len(new.node_ids), nstate), dtype=cp.float64)
    old_ix = cp.asarray([old_map[node] for node in retained], dtype=cp.int64)
    new_ix = cp.asarray([new_map[node] for node in retained], dtype=cp.int64)
    candidate[new_ix] = accepted_full[old_ix]
    candidate[cp.asarray([new_map[node] for node in parents], dtype=cp.int64)] = restricted
    if not bool(cp.all(cp.isfinite(candidate))) or bool(cp.any(candidate[:, 0] < -1e-12)):
        raise RuntimeError('AMR GPU coarsening produced an invalid candidate')
    old_area = cp.asarray(old.data['area'])
    new_area = cp.asarray(new.data['area'])
    class_rows = []
    bed_indices = np.asarray(old.data['bed_indices'])
    for grain in range(nfrac):
        class_rows.append([int(old.data['suspended_indices'][grain])]+[
            int(bed_indices[layer, grain]) for layer in range(bed_indices.shape[0])])
    water = float(cp.sum(new_area*candidate[:, 0]).item()-
                  cp.sum(old_area*accepted_full[:, 0]).item())
    classes = [float(cp.sum(new_area*cp.sum(candidate[:, rows], axis=1)).item()-
                     cp.sum(old_area*cp.sum(accepted_full[:, rows], axis=1)).item())
               for rows in class_rows]
    secondary = 0.
    if int(old.data['secondary_flow_model']) == 2:
        row = int(old.data['secondary_index'])
        secondary = float(cp.sum(new_area*candidate[:, row]).item()-
                          cp.sum(old_area*accepted_full[:, row]).item())
    if abs(water) > 1e-10 or max(map(abs, classes)) > 1e-8 or \
            abs(secondary) > 1e-8 or any(bool(cp.any(candidate[:, rows] < -1e-12))
                                               for rows in class_rows):
        raise RuntimeError('AMR GPU coarsening violated conservation/positivity')
    receipt = dict(old_leaves=len(old.node_ids), new_leaves=len(new.node_ids),
        coarsened_parent_ids=parents,
        transfer_water_residual_m3=water,
        transfer_sediment_class_residual_kg=classes,
        transfer_secondary_residual=secondary,
        all_state_rows_transferred_on_CUDA=True,
        no_CPU_numerical_fallback=True)
    return candidate, receipt


def refine_selected_state(prepared: PreparedAMR, old: AMRTopology,
                          new: AMRTopology, accepted_full, cached_children,
                          *, transfer_source):
    """Prolong evolved parents on CUDA with accepted pre-coarsening children.

    Zero primitive gradient is explicit. The native transfer kernel preserves
    the cached subcell bed distribution, fills water over that bed, and balances
    every conserved row to the *current* parent. No native initial/future state
    is used for the numerical transfer.
    """
    import cupy as cp
    from validation.gpu_fwave.amr_transfer import ResidentAMRTransfer

    if (not any(item is old for item in prepared.topologies) or
            not any(item is new for item in prepared.topologies)):
        raise ValueError('AMR topology is not in the authenticated catalog')
    nstate = int(old.data['nstate'])
    nfrac = int(old.data['nfrac'])
    if (not isinstance(accepted_full, cp.ndarray) or
            accepted_full.shape != (len(old.node_ids), nstate) or
            accepted_full.dtype != cp.float64 or
            not accepted_full.flags.c_contiguous or
            int(new.data['nstate']) != nstate or
            int(new.data['nfrac']) != nfrac or
            not np.array_equal(old.data['bulk_fraction'],
                               new.data['bulk_fraction'])):
        raise ValueError('Invalid AMR parent state/material layout')
    source = Path(transfer_source).resolve()
    if not source.is_file():
        raise ValueError('AMR transfer CUDA source is missing')
    parent = prepared.tree['parent']
    area = prepared.tree['area']
    xy = prepared.tree['xy']
    old_map = {int(node): position for position, node in enumerate(old.node_ids)}
    new_map = {int(node): position for position, node in enumerate(new.node_ids)}
    retained = sorted(set(old_map) & set(new_map))
    parents = sorted(set(old_map)-set(new_map))
    added = set(new_map)-set(old_map)
    if not parents or not added:
        raise ValueError('AMR refinement requires at least one split parent')
    groups = [np.flatnonzero(parent == node).astype(np.int64) for node in parents]
    if (any(len(group) != 4 or not set(group).issubset(new_map)
            for group in groups) or
            set(np.concatenate(groups).tolist()) != added):
        raise ValueError('AMR catalog transition is not complete four-child refinement')
    if (not isinstance(cached_children, cp.ndarray) or
            cached_children.shape != (len(parents), 4, nstate) or
            cached_children.dtype != cp.float64 or
            not cached_children.flags.c_contiguous or
            not bool(cp.all(cp.isfinite(cached_children))) or
            bool(cp.any(cached_children[:, :, 0] < -1e-12)) or
            bool(cp.any(cached_children[:, :, 3:3+nfrac*3] < -1e-12))):
        raise ValueError('Accepted resident AMR child cache is invalid')
    group_index = np.asarray([[new_map[int(child)] for child in group]
                              for group in groups], dtype=np.int64)
    weights = np.asarray(area[np.concatenate(groups)].reshape(-1, 4),
                         dtype=np.float64)
    positions = np.asarray([[xy[int(child)]-xy[node] for child in group]
                            for node, group in zip(parents, groups)],
                           dtype=np.float64)
    rock = np.asarray(new.data['rock'][group_index], dtype=np.float64)
    placeholder = np.zeros((len(parents), 4, nstate), dtype=np.float64)
    placeholder[:, :, 0] = max(float(old.data['dry_tolerance']), 1e-4)
    for row in np.asarray(old.data['bed_indices']).ravel():
        placeholder[:, :, int(row)] = .01
    transfer = ResidentAMRTransfer(np.ascontiguousarray(placeholder),
        weights, np.asarray(area[parents]), rock, positions,
        np.asarray(old.data['bulk_fraction'], dtype=np.float64),
        dry=float(old.data['dry_tolerance']),
        prognostic_secondary=int(old.data['secondary_flow_model']) == 2,
        source_path=source)
    cp.copyto(transfer.accepted, cached_children)
    parent_state = cp.ascontiguousarray(accepted_full[
        cp.asarray([old_map[node] for node in parents], dtype=cp.int64)])
    gradient = cp.zeros((len(parents), transfer.primitives, 2), dtype=cp.float64)
    children, status = transfer.propose(parent_state, gradient)
    if bool(cp.any(status != 0)):
        transfer.rollback()
        raise RuntimeError('AMR GPU refinement prolongation rejected')
    candidate = cp.empty((len(new.node_ids), nstate), dtype=cp.float64)
    candidate[cp.asarray([new_map[node] for node in retained], dtype=cp.int64)] = \
        accepted_full[cp.asarray([old_map[node] for node in retained], dtype=cp.int64)]
    candidate[cp.asarray(group_index)] = children
    transfer.rollback()
    if (not bool(cp.all(cp.isfinite(candidate))) or
            bool(cp.any(candidate[:, 0] < -1e-12))):
        raise RuntimeError('AMR GPU refinement produced an invalid candidate')
    old_area = cp.asarray(old.data['area'])
    new_area = cp.asarray(new.data['area'])
    class_rows = []
    bed_indices = np.asarray(old.data['bed_indices'])
    for grain in range(nfrac):
        class_rows.append([int(old.data['suspended_indices'][grain])]+[
            int(bed_indices[layer, grain]) for layer in range(bed_indices.shape[0])])
    water = float(cp.sum(new_area*candidate[:, 0]).item()-
                  cp.sum(old_area*accepted_full[:, 0]).item())
    classes = [float(cp.sum(new_area*cp.sum(candidate[:, rows], axis=1)).item()-
                     cp.sum(old_area*cp.sum(accepted_full[:, rows], axis=1)).item())
               for rows in class_rows]
    secondary = 0.
    if int(old.data['secondary_flow_model']) == 2:
        row = int(old.data['secondary_index'])
        secondary = float(cp.sum(new_area*candidate[:, row]).item()-
                          cp.sum(old_area*accepted_full[:, row]).item())
    if (abs(water) > 1e-10 or max(map(abs, classes)) > 1e-8 or
            abs(secondary) > 1e-8 or
            any(bool(cp.any(candidate[:, rows] < -1e-12)) for rows in class_rows)):
        raise RuntimeError('AMR GPU refinement violated conservation/positivity')
    receipt = dict(old_leaves=len(old.node_ids), new_leaves=len(new.node_ids),
        refined_parent_ids=parents, transfer_water_residual_m3=water,
        transfer_sediment_class_residual_kg=classes,
        transfer_secondary_residual=secondary, primitive_gradient='zero',
        accepted_child_cache_used=True, all_state_rows_transferred_on_CUDA=True,
        no_CPU_numerical_fallback=True)
    return candidate, receipt
