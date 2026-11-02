"""Bounded accepted regridding across two independently prepared GPU layouts.

The pair differs by one complete four-child group and both selected meshes
retain mixed levels, so the existing subcycled/reflux stepper can advance on
either side. Indicators use the current accepted CUDA state. The explicit
zero-gradient refinement and two-mesh policy are research limitations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .amr_prepared import (PreparedAMR, load_amr_prepared,
                           _bank_same_global_controls)
from .amr_catalog_checkpoint import read_mixed_checkpoint
from .amr_catalog_subcycle import CatalogMixedStepper, STATE_KEYS
from .amr_catalog_transition import coarsen_selected_state, refine_selected_state
from .atomic_output import save_arrays_atomic
from .runner import fields_digest


EXTRA_FIELDS = ('regrid_mesh_index', 'regrid_last_step', 'regrid_events_json',
                'regrid_cached_children', 'regrid_policy_json')
MATERIAL_KEYS = ('nfrac', 'nstate', 'nbed_layers', 'bedlaw', 'exchange_mode',
                 'secondary_flow_model', 'dry_tolerance', 'cfl_target',
                 'morphological_factor', 'bulk_fraction', 'suspended_indices',
                 'bed_indices', 'secondary_index', 'mode', 'forcing',
                 'fraction_bed_input', 'fraction_concentration')
_BANK_MATERIAL_KEYS = ('bank_failure', 'bank_toe_erosion_mode',
    'bank_toe_capacity_fraction', 'bank_critical_undercut_m',
    'general_avalanche', 'general_avalanche_wet_slope',
    'general_avalanche_dry_slope', 'general_avalanche_time',
    'general_avalanche_wet_depth_switch',
    'general_avalanche_relaxation_length', 'mpm_friction_mode',
    'mpm_critical', 'cpu_time_integrator')
_BANK_HISTORY_FLOATS = ('bank_undercut', 'bank_reference_height',
    'bank_toe_height', 'bank_reference_length', 'bank_toe_cumulative',
    'bank_collapse_cumulative')


def combine_regrid_catalog(refined_package, coarse_package):
    """Authenticate and combine distinct zero-step packages without rewriting either."""
    refined = load_amr_prepared(refined_package)
    coarse = load_amr_prepared(coarse_package)
    if refined.schema != coarse.schema:
        raise ValueError('AMR prepared package schemas differ')
    if refined.identity == coarse.identity:
        raise ValueError('Repeated AMR gate requires independent prepared packages')
    if any(not np.array_equal(refined.tree[key], coarse.tree[key])
           for key in refined.tree):
        raise ValueError('AMR prepared hierarchies differ')
    larger, smaller = refined.topologies[0], coarse.topologies[0]
    a, b = set(map(int, larger.node_ids)), set(map(int, smaller.node_ids))
    parents = b-a
    added = a-b
    tree = refined.tree
    if len(parents) != 1 or len(added) != 4:
        raise ValueError('AMR layouts must differ by one complete child group')
    parent = parents.pop()
    children = np.flatnonzero(tree['parent'] == parent).astype(np.int64)
    if len(children) != 4 or set(map(int, children)) != added:
        raise ValueError('AMR selected difference is not a four-child group')
    controls = MATERIAL_KEYS + (_BANK_MATERIAL_KEYS if refined.schema == 2 else ())
    if any(not np.array_equal(larger.data[key], smaller.data[key])
           for key in controls) or (refined.schema == 2 and
                not _bank_same_global_controls(larger.data, smaller.data)):
        raise ValueError('AMR prepared physical controls differ')
    old_map = {int(node): pos for pos, node in enumerate(larger.node_ids)}
    new_map = {int(node): pos for pos, node in enumerate(smaller.node_ids)}
    retained = sorted(a & b)
    if (not np.allclose(larger.data['rock'][[old_map[node] for node in retained]],
                        smaller.data['rock'][[new_map[node] for node in retained]],
                        rtol=0, atol=1e-12) or
            np.count_nonzero(tree['node_level'][larger.node_ids] == 2) == 0 or
            np.count_nonzero(tree['node_level'][smaller.node_ids] == 2) == 0):
        raise ValueError('AMR retained material or mixed-level layout differs')
    if refined.schema == 2:
        old_ix = [old_map[node] for node in retained]
        new_ix = [new_map[node] for node in retained]
        for key in ('state','bed','rock','layer_mass','manning_cell','active'):
            if not np.array_equal(np.take(larger.data[key], old_ix, axis=-1),
                                  np.take(smaller.data[key], new_ix, axis=-1)):
                raise ValueError('AMR retained water/sediment/material differs: '+key)
        for key in ('bank_material_mask', *_BANK_HISTORY_FLOATS,
                    'bank_collapse_count'):
            if not np.array_equal(larger.data[key][old_ix],
                                  smaller.data[key][new_ix]):
                raise ValueError('AMR retained bank material/history differs: '+key)
        child_ix = [old_map[int(node)] for node in children]
        parent_ix = new_map[parent]
        if (not np.all(larger.data['bank_material_mask'][child_ix] ==
                       smaller.data['bank_material_mask'][parent_ix])):
            raise ValueError('AMR parent/children bank material differs')
        weights = tree['area'][children]
        parent_area = tree['area'][parent]
        for key in _BANK_HISTORY_FLOATS:
            child_values = larger.data[key][child_ix]
            expected = (np.max(child_values) if key in
                _BANK_HISTORY_FLOATS[:3] else
                np.dot(weights, child_values)/parent_area)
            if not np.isclose(smaller.data[key][parent_ix], expected,
                              rtol=0, atol=1e-12):
                raise ValueError('AMR native bank-history restriction differs: '+key)
        if (smaller.data['bank_collapse_count'][parent_ix] !=
                np.max(larger.data['bank_collapse_count'][child_ix])):
            raise ValueError('AMR native bank-collapse count restriction differs')
        outlets = []
        for topology in (larger, smaller):
            faces = np.asarray(topology.data['boundary_faces'], dtype=np.int64)
            owner = np.asarray(topology.data['owner'], dtype=np.int64)
            outlets.append(sorted(map(int, topology.node_ids[owner[faces]])))
        if outlets[0] != outlets[1]:
            raise ValueError('AMR finest stage-outlet owners differ')
    identity = hashlib.sha256(json.dumps({
        'schema': refined.schema, 'refined': refined.identity,
        'coarse': coarse.identity,
        'operators': [larger.source_identity, smaller.source_identity],
    }, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    catalog = PreparedAMR(refined.folder, identity, tree, (larger, smaller),
        dict(refined_package=refined.identity, coarse_package=coarse.identity),
        refined.schema)
    return catalog, parent, children


class CatalogRegridRoute:
    """One accepted mixed-level stepper with transactional mesh replacement."""

    def __init__(self, prepared, parent, children, *, source_folder,
                 outer_dt=.002, min_dwell=2, coarsen_spread_m=1e-5,
                 refine_jump_m=1e-7, mesh_index=0,
                 stepper_class=CatalogMixedStepper):
        if prepared.schema != 1:
            raise ValueError('Bank AMR schema-2 GPU evolution is not qualified')
        if (mesh_index not in (0, 1) or type(min_dwell) is not int or
                min_dwell < 1 or not np.isfinite(outer_dt) or outer_dt <= 0 or
                not 0 < refine_jump_m < coarsen_spread_m):
            raise ValueError('Invalid bounded AMR regrid policy')
        self.prepared = prepared
        self.parent = int(parent)
        self.children = np.asarray(children, dtype=np.int64).copy()
        self.source_folder = Path(source_folder).resolve()
        self.outer_dt = float(outer_dt)
        self.policy = dict(min_dwell=min_dwell,
            coarsen_spread_m=float(coarsen_spread_m),
            refine_jump_m=float(refine_jump_m))
        self.mesh_index = mesh_index
        if not issubclass(stepper_class, CatalogMixedStepper):
            raise TypeError('AMR regrid route requires a mixed-level GPU stepper')
        self.stepper_class = stepper_class
        self.model = stepper_class(prepared, prepared.topologies[mesh_index],
            source_folder=self.source_folder, outer_dt=self.outer_dt)
        self.cp = self.model.cp
        self.cached_children = self.cp.empty((0, 4, 10), dtype=self.cp.float64)
        self.last_transition_step = 0
        self.events = []

    def indicator(self):
        """Read only a scalar tag from the evolved, accepted CUDA depth field."""
        cp = self.cp
        model = self.model
        if self.mesh_index == 0:
            positions = [int(np.flatnonzero(model.ids == node)[0])
                         for node in self.children]
            depth = model.state['q'][0, cp.asarray(positions)]
            return 'coarsen', float((cp.max(depth)-cp.min(depth)).item())
        position = int(np.flatnonzero(model.ids == self.parent)[0])
        owner = np.asarray(model.topology.data['owner'])
        neighbour = np.asarray(model.topology.data['neighbour'])
        adjacent = np.unique(np.concatenate((
            neighbour[(owner == position) & (neighbour >= 0)],
            owner[neighbour == position])))
        if not len(adjacent):
            raise ValueError('AMR parent has no selected face neighbor')
        depth = model.state['q'][0]
        return 'refine', float(cp.max(cp.abs(
            depth[cp.asarray(adjacent)]-depth[position])).item())

    def proposed_action(self):
        if self.model.accepted_steps-self.last_transition_step < \
                self.policy['min_dwell']:
            return None, None
        action, value = self.indicator()
        threshold = self.policy['coarsen_spread_m'] if action == 'coarsen' \
            else self.policy['refine_jump_m']
        accept = value < threshold if action == 'coarsen' else value > threshold
        return (action if accept else None), value

    def _install_replacement(self, replacement, candidate):
        cp = self.cp
        old = self.model
        state = replacement._unpack(candidate)
        for key in STATE_KEYS:
            cp.copyto(replacement.state[key], state[key])
        cp.copyto(replacement.reflux.accepted, candidate)
        replacement.reflux.accepted_register.fill(0)
        replacement.accepted_steps = old.accepted_steps
        replacement.level_steps = old.level_steps.copy()
        replacement.solver.euler_calls = old.solver.euler_calls
        replacement.maximum_cfl = old.maximum_cfl
        replacement.source_integral = old.source_integral
        replacement.exchange_ledger = old.exchange_ledger.copy()
        for counter in ('sync_steps', 'mixed_steps'):
            if hasattr(old, counter):
                setattr(replacement, counter, getattr(old, counter))
        replacement.secondary_peaks = old.secondary_peaks.copy()
        for name in old.peaks:
            cp.copyto(replacement.peaks[name], old.peaks[name])
        for name in old.hydro_peaks:
            cp.copyto(replacement.hydro_peaks[name], old.hydro_peaks[name])
        for name in old.solver.ledger:
            cp.copyto(replacement.solver.ledger[name], old.solver.ledger[name])
        if not bool(cp.array_equal(replacement._full(replacement.state), candidate)):
            raise RuntimeError('AMR regrid replacement state installation differs')

    def transition(self, action, indicator_m, *, reject_after_candidate=False):
        if action != ('coarsen' if self.mesh_index == 0 else 'refine'):
            raise ValueError('AMR transition action differs from active topology')
        expected, actual = self.proposed_action()
        if expected != action or actual != indicator_m:
            raise ValueError('AMR transition is not justified by accepted state')
        old = self.model
        target_index = 1-self.mesh_index
        target = self.prepared.topologies[target_index]
        full = old._full(old.state)
        if action == 'coarsen':
            positions = np.asarray([int(np.flatnonzero(old.ids == node)[0])
                                    for node in self.children], dtype=np.int64)
            cache = self.cp.ascontiguousarray(full[self.cp.asarray(positions)][None])
            candidate, transfer = coarsen_selected_state(self.prepared,
                old.topology, target, full,
                transfer_source=self.source_folder/'amr_transfer.cu')
        else:
            cache = self.cp.empty((0, 4, 10), dtype=self.cp.float64)
            candidate, transfer = refine_selected_state(self.prepared,
                old.topology, target, full, self.cached_children,
                transfer_source=self.source_folder/'amr_transfer.cu')
        replacement = self.stepper_class(self.prepared, target,
            source_folder=self.source_folder, outer_dt=self.outer_dt)
        self._install_replacement(replacement, candidate)
        if reject_after_candidate:
            raise RuntimeError('injected late AMR regrid rejection')
        event = dict(step=old.accepted_steps,
            time_s=old.accepted_steps*self.outer_dt, action=action,
            indicator_m=indicator_m, parent_id=self.parent,
            old_interface_faces=old.interface_count,
            new_interface_faces=replacement.interface_count,
            reflux_register_reset_at_accepted_regrid=True)
        event.update(transfer)
        self.model = replacement
        self.mesh_index = target_index
        self.cached_children = cache
        self.last_transition_step = old.accepted_steps
        self.events.append(event)
        return event

    def advance(self):
        step = self.model.accepted_steps
        result = self.model.step(self.outer_dt, step*self.outer_dt)
        action, indicator = self.proposed_action()
        event = None if action is None else self.transition(action, indicator)
        return result, event

    def persistent(self):
        fields = self.model.persistent()
        fields.update(regrid_mesh_index=np.int64(self.mesh_index),
            regrid_last_step=np.int64(self.last_transition_step),
            regrid_events_json=np.asarray(json.dumps(self.events,
                sort_keys=True, separators=(',', ':'), allow_nan=False)),
            regrid_cached_children=self.cp.asnumpy(self.cached_children),
            regrid_policy_json=np.asarray(json.dumps(self.policy,
                sort_keys=True, separators=(',', ':'))))
        return fields

    def restore(self, fields):
        self.model.restore(fields)
        self.last_transition_step = int(fields['regrid_last_step'])
        self.events = json.loads(str(fields['regrid_events_json']))
        self.cached_children = self.cp.asarray(fields['regrid_cached_children'])


def write_regrid_checkpoint(path, route, *, source_identity, method_identity):
    if route.model.reflux.pending:
        raise ValueError('Cannot checkpoint pending AMR reflux')
    fields = route.persistent()
    fields.update(input_identity=np.asarray(route.prepared.identity),
        operator_identity=np.asarray(route.model.topology.source_identity),
        source_identity=np.asarray(source_identity),
        method_identity=np.asarray(method_identity))
    fields['checkpoint_payload_sha256'] = np.asarray(fields_digest(fields))
    save_arrays_atomic(path, **fields)
    return str(fields['checkpoint_payload_sha256'])


def read_regrid_checkpoint(path, prepared, parent, children, *,
                           source_identity, method_identity, outer_dt,
                           max_steps, policy, wetfront_schedule=False):
    """Authenticate the entire event/cache payload before any CUDA allocation."""
    with np.load(Path(path), allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    mesh = np.asarray(raw.get('regrid_mesh_index'))
    if mesh.shape != () or mesh.dtype.kind not in 'iu' or int(mesh) not in (0, 1):
        raise ValueError('AMR regrid checkpoint mesh index differs')
    fields = read_mixed_checkpoint(path, prepared, prepared.topologies[int(mesh)],
        source_identity=source_identity, method_identity=method_identity,
        outer_dt=outer_dt, max_steps=max_steps,
        extra_fields=EXTRA_FIELDS + (('sync_steps', 'mixed_steps')
                                    if wetfront_schedule else ()),
        wetfront_schedule=wetfront_schedule)
    encoded_policy = np.asarray(fields['regrid_policy_json'])
    if (encoded_policy.shape != () or encoded_policy.dtype.kind not in 'US' or
            str(encoded_policy.item()) != json.dumps(policy, sort_keys=True,
                separators=(',', ':'))):
        raise ValueError('AMR regrid checkpoint policy differs')
    last = np.asarray(fields['regrid_last_step'])
    cache = np.asarray(fields['regrid_cached_children'])
    if (last.shape != () or last.dtype.kind not in 'iu' or
            not 0 <= int(last) <= int(fields['accepted_steps']) or
            cache.shape != ((1, 4, 10) if int(mesh) else (0, 4, 10)) or
            cache.dtype != np.float64 or not cache.flags.c_contiguous or
            not np.isfinite(cache).all() or
            np.any(cache[:, :, 0] < -1e-12) or
            np.any(cache[:, :, 3:9] < -1e-12)):
        raise ValueError('AMR regrid checkpoint child cache/clock differs')
    encoded = np.asarray(fields['regrid_events_json'])
    if encoded.shape != () or encoded.dtype.kind not in 'US':
        raise ValueError('AMR regrid checkpoint event history differs')
    try:
        events = json.loads(str(encoded.item()))
    except (TypeError, ValueError) as exc:
        raise ValueError('AMR regrid checkpoint event history differs') from exc
    if not isinstance(events, list) or len(events) % 2 != int(mesh):
        raise ValueError('AMR regrid checkpoint event count differs')
    previous = 0
    for index, event in enumerate(events):
        if (not isinstance(event, dict) or
                event.get('action') != ('coarsen' if index % 2 == 0 else 'refine') or
                event.get('parent_id') != int(parent) or
                event.get('step') is None or type(event['step']) is not int or
                event['step']-previous < policy['min_dwell'] or
                event['step'] > int(fields['accepted_steps']) or
                event.get('time_s') != event['step']*outer_dt or
                not isinstance(event.get('indicator_m'), (int, float)) or
                not np.isfinite(event['indicator_m']) or
                event['indicator_m'] < 0 or
                event.get('old_leaves') != len(prepared.topologies[index % 2].node_ids) or
                event.get('new_leaves') != len(prepared.topologies[1-index % 2].node_ids)):
            raise ValueError('AMR regrid checkpoint event history differs')
        previous = event['step']
    if previous != int(last):
        raise ValueError('AMR regrid checkpoint last event clock differs')
    # A rehashed ledger is not sufficient evidence of an accepted trajectory.
    # Regridding preserves every conserved inventory, so the initial prepared
    # stock plus accepted boundary/source ledgers must equal the checkpoint
    # stock on either selected mesh. This is a host-only pre-CUDA admission.
    start = prepared.topologies[0]
    start_state = np.asarray(start.data['state']).T
    start_area = prepared.tree['area'][start.node_ids]
    end_state = np.asarray(fields['state'])
    end_area = prepared.tree['area'][prepared.topologies[int(mesh)].node_ids]
    water_error = (np.sum(end_area*end_state[:, 0])-
        np.sum(start_area*start_state[:, 0])-
        np.sum(fields['solver_ledger_water']))
    class_error = np.array([
        np.sum(end_area*(end_state[:, 3+g]+end_state[:, 5+g]+end_state[:, 7+g]))-
        np.sum(start_area*(start_state[:, 3+g]+start_state[:, 5+g]+start_state[:, 7+g]))-
        np.sum(fields['solver_ledger_bed'][g]+fields['solver_ledger_suspended'][g])
        for g in range(2)])
    secondary_error = (np.sum(end_area*end_state[:, 9])-
        np.sum(start_area*start_state[:, 9])-
        float(fields['source_integral'])-
        np.sum(fields['solver_ledger_secondary']))
    if (abs(water_error) > 1e-9 or np.max(np.abs(class_error)) > 1e-8 or
            abs(secondary_error) > 1e-8):
        raise ValueError('AMR regrid checkpoint cumulative conservation/ledger differs')
    return fields
