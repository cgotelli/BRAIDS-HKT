"""Versioned accepted-state checkpoint for on-demand general transport AMR.

All identities, shapes, event history and operator-cache entries are checked
on the host before a CUDA solver is constructed for continuation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from braids_core.windows_paths import filesystem_path

from .amr_general_stepper import require
from .amr_native_geometry import sha
from .atomic_output import save_arrays_atomic, write_json_atomic
from .runner import fields_digest


SCHEMA = 5
RECURSIVE_SCHEMA = 6
IMPLICIT_SUSPENSION_SCHEMA = 7
RECURSIVE_KIND = 'recursive_general_amr'
IMPLICIT_SUSPENSION_KIND = 'implicit_fwave_midpoint_suspension_amr'
PAYLOAD = 'state.npz'
RECEIPT = 'checkpoint.json'
OPERATORS = 'operators.json'


def _operator_rows(entries):
    return [dict(selected_ids=entry.selected_ids.tolist(),
                 request_key=entry.request_key,
                 package_identity=entry.package_identity,
                 entry_sha256=entry.entry_sha256) for entry in entries]


def _payload(route, delay_state, method_identity, *, recursive=False):
    model, cp = route.model, route.cp
    from .amr_general_subcycle_stepper import RecursiveGeneralStepper
    require(bool(recursive) == isinstance(model, RecursiveGeneralStepper),
            'General AMR checkpoint schema and stepper route differ')
    implicit = (not recursive and
                getattr(model, 'integrator', 'ssp33') ==
                'implicit_fwave_midpoint')
    nstate = int(model.topology.data['nstate'])
    delay = np.asarray(delay_state)
    require(delay.shape == (len(route.prepared.tree['parent']),) and
            delay.dtype.kind in 'iu' and np.all(delay >= 0),
            'General AMR checkpoint delay state differs')
    require(type(method_identity) is str and len(method_identity) == 64,
            'General AMR checkpoint method identity differs')
    state, tree = cp.asnumpy(model.full()), cp.asnumpy(route.synchronized_tree())
    require(state.shape == (model.n, nstate) and
            tree.shape == (len(delay), nstate),
            'General AMR accepted state dimensions differ')
    schema = (RECURSIVE_SCHEMA if recursive else
              IMPLICIT_SUSPENSION_SCHEMA if implicit else SCHEMA)
    fields = dict(schema=np.int64(schema),
        session_identity=np.asarray(route.session.identity),
        catalog_identity=np.asarray(route.prepared.identity),
        topology_identity=np.asarray(model.topology.source_identity),
        method_identity=np.asarray(method_identity),
        node_ids=model.ids.copy(), state=state, tree_state=tree,
        delay_state=delay.astype(np.int64, copy=True),
        time_s=np.float64(model.time_s),
        accepted_steps=np.int64(model.accepted_steps),
        euler_calls=np.int64(model.solver.euler_calls),
        maximum_cfl=np.float64(model.maximum_cfl),
        events_json=np.asarray(json.dumps(route.events, sort_keys=True,
                                          separators=(',', ':'), allow_nan=False)))
    fields.update({'ledger_'+key:cp.asnumpy(value).copy()
                   for key, value in model.solver.ledger.items()})
    if route.selector_last_bed is not None:
        fields['selector_last_bed'] = cp.asnumpy(route.selector_last_bed).copy()
    if recursive:
        fields.update(recursive_level_steps=model.level_steps.copy(),
            recursive_epoch_start_steps=np.int64(
                model.epoch_origin['accepted_steps']),
            recursive_epoch_start_time_s=np.float64(
                model.epoch_origin['time_s']),
            recursive_epoch_start_euler_calls=np.int64(
                model.epoch_origin['euler_calls']),
            recursive_register=cp.asnumpy(model.accepted_integral).copy(),
            recursive_last_source=cp.asnumpy(model.last_source).copy(),
            recursive_last_exchange=cp.asnumpy(model.last_exchange).copy())
    if implicit:
        controller = model.solver.implicit_midpoint
        require(controller is not None and controller.suspended_composition,
                'M2-C checkpoint requires implicit law-11 suspension')
        fields.update(controller.checkpoint_fields())
        fields.update(
            model.solver.flow2dh_boundary_diffusion.boundary_memory_fields())
        fields.update(
            selected_integrator=np.asarray('implicit_fwave_midpoint'),
            suspended_amr_contract_semantic_id=np.asarray(
                model.suspended_amr_contract.semantic_id),
            suspended_amr_contract_fingerprint=np.asarray(
                model.suspended_amr_contract.fingerprint),
            suspended_amr_contract_json=np.asarray(json.dumps(
                model.suspended_amr_contract.as_dict(), sort_keys=True,
                separators=(',', ':'), allow_nan=False)),
            wetdry_phase_ledger=cp.asnumpy(
                model.wetdry_phase_ledger).copy(),
            wetdry_last_suspended_change_kg=cp.asnumpy(
                model.last_wetdry_phase_change).copy(),
            implicit_controller_accepted_steps=np.int64(
                controller.accepted_steps),
            implicit_controller_rejected_steps=np.int64(
                controller.rejected_steps),
            implicit_fwave_evaluations=np.int64(
                controller.fwave_evaluations),
            implicit_linear_matvecs=np.int64(controller.linear_matvecs))
    fields['payload_sha256'] = np.asarray(fields_digest(fields))
    return fields


def write(folder, route, delay_state, *, method_identity, recursive=False):
    """Publish a complete write-once checkpoint after an accepted step."""
    folder = filesystem_path(folder).resolve()
    require(not folder.exists(), 'General AMR checkpoint is write-once')
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = filesystem_path(tempfile.mkdtemp(prefix='.general-amr-checkpoint-',
                                               dir=folder.parent))
    complete = staging/'complete'
    try:
        complete.mkdir()
        fields = _payload(route, delay_state, method_identity,
                          recursive=recursive)
        save_arrays_atomic(complete/PAYLOAD, **fields)
        schema = int(np.asarray(fields['schema']))
        receipt = dict(schema=schema,
            session_identity=route.session.identity,
            catalog_identity=route.prepared.identity,
            topology_identity=route.model.topology.source_identity,
            method_identity=method_identity,
            payload_file=PAYLOAD,
            payload_file_sha256=sha(complete/PAYLOAD),
            payload_sha256=str(fields['payload_sha256']),
            accepted_steps=route.model.accepted_steps,
            time_s=route.model.time_s,
            event_count=len(route.events))
        if recursive:
            receipt['kind'] = RECURSIVE_KIND
        elif schema == IMPLICIT_SUSPENSION_SCHEMA:
            receipt['kind'] = IMPLICIT_SUSPENSION_KIND
            receipt['integrator'] = 'implicit_fwave_midpoint'
            receipt['suspended_amr_contract_semantic_id'] = str(
                fields['suspended_amr_contract_semantic_id'])
            receipt['suspended_amr_contract_fingerprint'] = str(
                fields['suspended_amr_contract_fingerprint'])
        write_json_atomic(complete/RECEIPT, receipt)
        operators = dict(schema=1, session_identity=route.session.identity,
            checkpoint_sha256=sha(complete/RECEIPT),
            entries=_operator_rows(route.entries))
        write_json_atomic(complete/OPERATORS, operators)
        for attempt in range(8):
            require(not folder.exists(), 'Concurrent general AMR checkpoint exists')
            try:
                os.rename(complete, folder)
                break
            except OSError as error:
                if (os.name != 'nt' or getattr(error, 'winerror', None) not in (5, 32, 33)
                        or attempt == 7):
                    raise
                time.sleep(min(.02*2**attempt, .5))
        return receipt
    finally:
        require(staging.parent == folder.parent and
                staging.name.startswith('.general-amr-checkpoint-') and
                not staging.is_symlink(),
                'Invalid owned general AMR checkpoint staging')
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                # A scanner may still hold an unpublished Windows staging file.
                # The published checkpoint, if any, is already complete.
                pass


def write_recursive(folder, route, delay_state, *, method_identity):
    """Publish the recursive schema without changing historical schema 5."""
    return write(folder, route, delay_state, method_identity=method_identity,
                 recursive=True)


def _scalar(fields, key, dtype):
    value = np.asarray(fields.get(key))
    require(value.shape == () and value.dtype == np.dtype(dtype),
            'General AMR checkpoint scalar differs: '+key)
    return value.item()


def _text(fields, key):
    value = np.asarray(fields.get(key))
    require(value.shape == () and value.dtype.kind in 'US',
            'General AMR checkpoint identity differs: '+key)
    return str(value.item())


def _array(fields, key, shape, dtype):
    value = np.asarray(fields.get(key))
    require(value.shape == shape and value.dtype == np.dtype(dtype) and
            value.flags.c_contiguous and
            (value.dtype.kind != 'f' or np.isfinite(value).all()),
            'General AMR checkpoint array differs: '+key)
    return value


def read(folder, session, *, method_identity, recursive=False,
         integrator='ssp33'):
    """Authenticate operator family and physical payload before CUDA."""
    folder = filesystem_path(folder).resolve()
    require(integrator in ('ssp33', 'implicit_fwave_midpoint'),
            'General AMR checkpoint integrator differs')
    require(not recursive or integrator == 'ssp33',
            'Recursive implicit AMR is not qualified')
    implicit = not recursive and integrator == 'implicit_fwave_midpoint'
    expected_schema = (RECURSIVE_SCHEMA if recursive else
                       IMPLICIT_SUSPENSION_SCHEMA if implicit else SCHEMA)
    members = ({item.name for item in folder.iterdir()}
               if folder.is_dir() and not folder.is_symlink() else set())
    require(folder.is_dir() and not folder.is_symlink() and
            members in ({PAYLOAD, RECEIPT, OPERATORS},
                        {PAYLOAD, RECEIPT, OPERATORS, 'worker.json'}) and
            all((folder/name).is_file() and not (folder/name).is_symlink()
                for name in members),
            'General AMR checkpoint artifact set differs')
    receipt = json.loads((folder/RECEIPT).read_text(encoding='utf-8'))
    receipt_keys = ({'schema', 'session_identity', 'catalog_identity',
            'topology_identity', 'method_identity', 'payload_file',
            'payload_file_sha256', 'payload_sha256', 'accepted_steps', 'time_s',
            'event_count'} | ({'kind'} if recursive or implicit else set()) |
            ({'integrator', 'suspended_amr_contract_semantic_id',
              'suspended_amr_contract_fingerprint'} if implicit else set()))
    require(set(receipt) == receipt_keys and
            receipt['schema'] == expected_schema and
            (not recursive or receipt['kind'] == RECURSIVE_KIND) and
            (not implicit or (receipt['kind'] == IMPLICIT_SUSPENSION_KIND and
                              receipt['integrator'] ==
                              'implicit_fwave_midpoint' and
                              isinstance(receipt[
                                  'suspended_amr_contract_semantic_id'], str) and
                              isinstance(receipt[
                                  'suspended_amr_contract_fingerprint'], str))) and
            receipt['session_identity'] == session.identity and
            receipt['catalog_identity'] == session.identity and
            receipt['method_identity'] == method_identity and
            receipt['payload_file'] == PAYLOAD and
            receipt['payload_file_sha256'] == sha(folder/PAYLOAD),
            'General AMR checkpoint receipt/identity differs')
    inventory = json.loads((folder/OPERATORS).read_text(encoding='utf-8'))
    require(set(inventory) == {'schema', 'session_identity',
            'checkpoint_sha256', 'entries'} and
            inventory['schema'] == 1 and
            inventory['session_identity'] == session.identity and
            inventory['checkpoint_sha256'] == sha(folder/RECEIPT) and
            isinstance(inventory['entries'], list) and inventory['entries'],
            'General AMR checkpoint operator inventory differs')
    entries = []
    for row in inventory['entries']:
        require(isinstance(row, dict) and
                set(row) == {'selected_ids', 'request_key',
                             'package_identity', 'entry_sha256'},
                'General AMR checkpoint operator row differs')
        entry = session.store.get(row['selected_ids'])
        require(entry is not None and _operator_rows([entry])[0] == row,
                'General AMR checkpoint cached operator differs')
        entries.append(entry)
    require(entries[0].selected_ids.tolist() == session.manifest['initial_node_ids'],
            'General AMR checkpoint initial topology differs')
    prepared = session.view(entries)
    targets = [top for top in prepared.topologies
               if top.source_identity == receipt['topology_identity']]
    require(len(targets) == 1, 'General AMR checkpoint selected topology differs')
    top = targets[0]
    with np.load(folder/PAYLOAD, allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    payload_digest = fields.pop('payload_sha256', None)
    require(payload_digest is not None and payload_digest.shape == () and
            payload_digest.dtype.kind in 'US' and
            str(payload_digest.item()) == fields_digest(fields) and
            str(payload_digest.item()) == receipt['payload_sha256'],
            'General AMR checkpoint payload digest differs')
    required = {'schema', 'session_identity', 'catalog_identity',
        'topology_identity', 'method_identity', 'node_ids', 'state',
        'tree_state', 'delay_state', 'time_s', 'accepted_steps',
        'euler_calls', 'maximum_cfl', 'events_json'}
    recursive_fields = {'recursive_level_steps',
        'recursive_epoch_start_steps', 'recursive_epoch_start_time_s',
        'recursive_epoch_start_euler_calls', 'recursive_register',
        'recursive_last_source', 'recursive_last_exchange'} if recursive else set()
    implicit_fields = ({'selected_integrator', 'implicit_predictor_rate',
        'implicit_have_predictor',
        'suspended_amr_contract_semantic_id',
        'suspended_amr_contract_fingerprint',
        'suspended_amr_contract_json', 'wetdry_phase_ledger',
        'wetdry_last_suspended_change_kg',
        'flow2dh_boundary_last_outflow_concentration',
        'flow2dh_boundary_return_clock',
        'flow2dh_amr_boundary_face_source_ids',
        'flow2dh_amr_boundary_last_outflow_concentration',
        'flow2dh_amr_boundary_return_clock',
        'implicit_controller_accepted_steps',
        'implicit_controller_rejected_steps',
        'implicit_fwave_evaluations',
        'implicit_linear_matvecs'} if implicit else set())
    ledger_keys = {'ledger_'+key for key in ('water', 'bed', 'secondary', 'source')}
    from .general_live_transport import suspended_enabled
    suspended = suspended_enabled(top.data)
    if suspended:
        ledger_keys.update({'ledger_suspended', 'ledger_exchange'})
    reference = session.selector_arguments['settings']['refine_sediment']
    require(set(fields) == required|ledger_keys|recursive_fields|implicit_fields|
            ({'selector_last_bed'} if reference else set()),
            'General AMR checkpoint physical/selector inventory differs')
    require(_scalar(fields, 'schema', np.int64) == expected_schema and
            _text(fields, 'session_identity') == session.identity and
            _text(fields, 'catalog_identity') == prepared.identity and
            _text(fields, 'topology_identity') == top.source_identity and
            _text(fields, 'method_identity') == method_identity,
            'General AMR checkpoint method/topology identity differs')
    n, nn = len(top.node_ids), len(prepared.tree['parent'])
    ns, ng, nb = int(top.data['nstate']), int(top.data['nfrac']), int(top.data['nb'])
    require(np.array_equal(_array(fields, 'node_ids', (n,), np.int64), top.node_ids),
            'General AMR checkpoint global nodes differ')
    state = _array(fields, 'state', (n, ns), np.float64)
    tree = _array(fields, 'tree_state', (nn, ns), np.float64)
    delay = _array(fields, 'delay_state', (nn,), np.int64)
    require(np.all(delay >= 0) and np.array_equal(tree[top.node_ids], state),
            'General AMR checkpoint accepted tree differs')
    sediment = np.r_[np.asarray(top.data['suspended_indices']).ravel(),
                     np.asarray(top.data['bed_indices']).ravel()]
    dry_tolerance = float(top.data.get('dry', top.data['dry_tolerance']))
    for name, rows in (('state', state), ('tree_state', tree)):
        dry = rows[:, 0] <= dry_tolerance
        require(np.all(rows[:, 0] >= 0) and np.all(rows[:, sediment] >= 0) and
                not np.any(rows[dry, 1:3]) and
                not np.any(rows[dry][:, top.data['suspended_indices']]) and
                (int(top.data['secondary_flow_model']) != 2 or
                 not np.any(rows[dry, int(top.data['secondary_index'])])),
                'General AMR checkpoint invalid wet/dry sediment: '+name)
    if reference:
        _array(fields, 'selector_last_bed', (nn,), np.float64)
    shapes = dict(ledger_water=(nb,), ledger_bed=(ng, nb),
        ledger_secondary=(nb,), ledger_source=())
    if suspended:
        shapes.update(ledger_suspended=(ng, nb), ledger_exchange=(ng, 2))
    for key, shape in shapes.items():
        _array(fields, key, shape, np.float64)
    if suspended:
        require(np.max(np.abs(np.sum(fields['ledger_exchange'], axis=1))) <= 1e-8,
                'General AMR checkpoint exchange ledger unbalanced')
    implicit_clock = True
    if implicit:
        from .implicit_suspension_amr import contract_from_data
        contract = contract_from_data(top.data)
        require(int(top.data['bedlaw']) == 11 and
                int(top.data['exchange_mode']) == 4 and ng == 1,
                'M2-C checkpoint physical route differs')
        require(_text(fields, 'selected_integrator') ==
                'implicit_fwave_midpoint',
                'M2-C checkpoint selected integrator differs')
        require(_text(fields, 'suspended_amr_contract_semantic_id') ==
                contract.semantic_id and
                _text(fields, 'suspended_amr_contract_fingerprint') ==
                contract.fingerprint and
                _text(fields, 'suspended_amr_contract_json') == json.dumps(
                    contract.as_dict(), sort_keys=True, separators=(',', ':'),
                    allow_nan=False) and
                receipt['suspended_amr_contract_semantic_id'] ==
                    contract.semantic_id and
                receipt['suspended_amr_contract_fingerprint'] ==
                    contract.fingerprint,
                'M2-C checkpoint contract identity differs')
        phase = _array(fields, 'wetdry_phase_ledger', (ng, 2), np.float64)
        last_phase = _array(fields, 'wetdry_last_suspended_change_kg',
                            (ng,), np.float64)
        require(np.allclose(np.sum(phase, axis=1), 0., rtol=0., atol=1e-12) and
                np.all(last_phase <= 0.),
                'M2-C checkpoint phase ledger differs')
        _array(fields, 'implicit_predictor_rate', (3, n), np.float64)
        valid = _scalar(fields, 'implicit_have_predictor', np.int32)
        last = _array(fields,
            'flow2dh_boundary_last_outflow_concentration',
            (ng, int(top.data['nf'])), np.float64)
        boundary_clock = _array(fields, 'flow2dh_boundary_return_clock',
                                last.shape, np.float64)
        boundary_faces = np.asarray(top.data['boundary_faces'], dtype=np.int64)
        compact_source = _array(fields,
            'flow2dh_amr_boundary_face_source_ids',
            (len(boundary_faces),), np.int64)
        compact_last = _array(fields,
            'flow2dh_amr_boundary_last_outflow_concentration',
            (ng, len(boundary_faces)), np.float64)
        compact_clock = _array(fields,
            'flow2dh_amr_boundary_return_clock', compact_last.shape,
            np.float64)
        require(np.array_equal(compact_source,
                    np.asarray(top.data['face_source_ids'])[boundary_faces]) and
                len(np.unique(compact_source)) == len(compact_source) and
                np.array_equal(compact_last, last[:, boundary_faces]) and
                np.array_equal(compact_clock,
                               boundary_clock[:, boundary_faces]),
                'M2-C compact physical boundary memory differs')
        controller_steps = _scalar(fields,
            'implicit_controller_accepted_steps', np.int64)
        rejected_steps = _scalar(fields,
            'implicit_controller_rejected_steps', np.int64)
        fwave_evaluations = _scalar(fields, 'implicit_fwave_evaluations',
                                    np.int64)
        linear_matvecs = _scalar(fields, 'implicit_linear_matvecs', np.int64)
        bid = np.asarray(top.data['bid'])
        return_time = np.asarray(top.data['boundary_return_time_s'],
                                 dtype=np.float64)
        if return_time.shape == (nb,):
            return_time = return_time[None, :]
        face_return = np.zeros((ng, int(top.data['nf'])), dtype=np.float64)
        boundary = bid >= 0
        face_return[:, boundary] = return_time[:, bid[boundary]]
        tolerance = 256*np.finfo(np.float64).eps*np.maximum(1., face_return)
        implicit_clock = (valid in (0, 1) and controller_steps >= 0 and
            rejected_steps >= 0 and fwave_evaluations >= 0 and
            linear_matvecs >= 0 and np.all(last >= 0.) and
            np.all(boundary_clock >= 0.) and
            np.all(boundary_clock <= face_return+tolerance) and
            np.all(boundary_clock[face_return == 0.] == 0.))
    step = _scalar(fields, 'accepted_steps', np.int64)
    calls = _scalar(fields, 'euler_calls', np.int64)
    clock = _scalar(fields, 'time_s', np.float64)
    maximum = _scalar(fields, 'maximum_cfl', np.float64)
    recursive_clock = True
    if recursive:
        levels = prepared.tree['node_level'][top.node_ids].astype(np.int32)
        owner, neighbour = (np.asarray(top.data[name]) for name in
                            ('owner', 'neighbour'))
        internal = neighbour >= 0
        require(levels.size and np.all(levels >= 0) and
                np.all(np.abs(levels[owner[internal]]-
                              levels[neighbour[internal]]) <= 1),
                'Recursive general checkpoint levels are not balanced')
        root, finest = int(np.min(levels)), int(np.max(levels))
        level_steps = _array(fields, 'recursive_level_steps',
                             (finest+1,), np.int64)
        origin_step = _scalar(fields, 'recursive_epoch_start_steps', np.int64)
        origin_time = _scalar(fields, 'recursive_epoch_start_time_s', np.float64)
        origin_calls = _scalar(fields, 'recursive_epoch_start_euler_calls',
                               np.int64)
        register = _array(fields, 'recursive_register',
                          (int(top.data['nf']), 2, ns), np.float64)
        last_source = _scalar(fields, 'recursive_last_source', np.float64)
        last_exchange = _array(fields, 'recursive_last_exchange',
                               (ng, 2), np.float64)
        elapsed = step-origin_step
        expected = np.zeros(finest+1, np.int64)
        for level in range(root, finest+1):
            if np.any(levels == level):
                expected[level] = elapsed*(1 << (level-root))
        recursive_clock = (0 <= origin_step <= step and
            0 <= origin_time <= clock and 0 <= origin_calls <= calls and
            ((origin_step == 0 and origin_time == 0. and origin_calls == 0) or
             (origin_step > 0 and origin_time > 0. and origin_calls > 0)) and
            np.array_equal(level_steps, expected) and
            calls == origin_calls+3*int(expected.sum()) and
            np.isfinite(last_source) and
            np.max(np.abs(np.sum(last_exchange, axis=1))) <= 1e-8 and
            (elapsed > 0 or (not np.any(register) and last_source == 0. and
                             not np.any(last_exchange))))
    require(step >= 0 and
            (recursive_clock if recursive else
             (implicit_clock and calls == step and
              controller_steps == step) if implicit else calls == 3*step) and
            0 <= clock <= float(top.data['endtime'])+1e-12 and
            (step == 0) == (clock == 0) and
            0 <= maximum <= float(top.data['cfl_target'])+1e-12 and
            receipt['accepted_steps'] == step and receipt['time_s'] == clock,
            'General AMR checkpoint accepted clock differs')
    try:
        events = json.loads(_text(fields, 'events_json'))
    except (TypeError, ValueError) as exc:
        raise ValueError('General AMR checkpoint event JSON differs') from exc
    require(isinstance(events, list) and len(events) == receipt['event_count'],
            'General AMR checkpoint event count differs')
    by_name = {item.name: item for item in prepared.topologies}
    previous, previous_step, previous_time = prepared.topologies[0].name, 0, 0.
    recursive_origin = dict(accepted_steps=0, time_s=0., euler_calls=0)
    for event in events:
        require(isinstance(event, dict) and event.get('old_topology') == previous and
                event.get('new_topology') in by_name and
                event['new_topology'] != previous and
                type(event.get('step')) is int and
                previous_step <= event['step'] <= step and
                type(event.get('time_s')) in (int, float) and
                previous_time <= event['time_s'] <= clock and
                event.get('old_operator_identity') == by_name[previous].source_identity and
                event.get('catalog_identity') == prepared.identity and
                event.get('new_operator_identity') ==
                by_name[event['new_topology']].source_identity,
                'General AMR checkpoint topology-event chain differs')
        if recursive:
            epoch = event.get('recursive_epoch')
            old_top = by_name[previous]
            old_levels = prepared.tree['node_level'][old_top.node_ids]
            old_finest = int(np.max(old_levels))
            epoch_steps = np.asarray(epoch.get('level_steps')
                if isinstance(epoch, dict) else None)
            expected_epoch_steps = np.zeros(old_finest+1, np.int64)
            old_root = int(np.min(old_levels))
            epoch_macros = event['step']-recursive_origin['accepted_steps']
            for level in range(old_root, old_finest+1):
                if np.any(old_levels == level):
                    expected_epoch_steps[level] = epoch_macros*(1 << (level-old_root))
            require(isinstance(epoch, dict) and
                    set(epoch) == {'previous_origin', 'end_step', 'end_time_s',
                        'euler_calls', 'level_steps', 'old_operator_identity',
                        'consumed_register_shape', 'consumed_register_sha256',
                        'remapped_as_pending_flux'} and
                    epoch['previous_origin'] == recursive_origin and
                    epoch['end_step'] == event['step'] and
                    epoch['end_time_s'] == event['time_s'] and
                    type(epoch['euler_calls']) is int and
                    epoch['euler_calls'] == (recursive_origin['euler_calls']+
                                             3*int(expected_epoch_steps.sum())) and
                    epoch_steps.dtype.kind in 'iu' and
                    np.array_equal(epoch_steps, expected_epoch_steps) and
                    epoch['old_operator_identity'] == old_top.source_identity and
                    epoch['consumed_register_shape'] ==
                        [int(old_top.data['nf']), 2, int(old_top.data['nstate'])] and
                    isinstance(epoch['consumed_register_sha256'], str) and
                    len(epoch['consumed_register_sha256']) == 64 and
                    all(character in '0123456789abcdef' for character in
                        epoch['consumed_register_sha256']) and
                    epoch['remapped_as_pending_flux'] is False,
                    'Recursive general checkpoint topology epoch differs')
            recursive_origin = dict(accepted_steps=event['step'],
                time_s=event['time_s'], euler_calls=epoch['euler_calls'])
        previous, previous_step, previous_time = (event['new_topology'],
                                                  event['step'], event['time_s'])
    require(previous == top.name, 'General AMR checkpoint events miss selected mesh')
    if recursive:
        require(recursive_origin == dict(accepted_steps=origin_step,
                time_s=origin_time, euler_calls=origin_calls),
                'Recursive general checkpoint current epoch origin differs')
    return prepared, top, entries, fields, events, delay


def read_recursive(folder, session, *, method_identity):
    """Read only the recursive schema; schema-5 checkpoints remain separate."""
    return read(folder, session, method_identity=method_identity, recursive=True)


def restore(folder, session, *, source_folder, transport_source_folder,
            method_identity, integrator='ssp33', cancelled=lambda: False):
    prepared, top, entries, fields, events, delay = read(
        folder, session, method_identity=method_identity,
        integrator=integrator)
    from .amr_general_route import GeneralOnDemandRoute

    route = GeneralOnDemandRoute(session, source_folder=source_folder,
        transport_source_folder=transport_source_folder,
        integrator=integrator, cancelled=cancelled)
    route.entries = entries
    route.prepared = prepared
    route.by_leaves = {tuple(map(int, item.node_ids)): item
                       for item in prepared.topologies}
    route.selector = route._selector(prepared)
    route.model = route._make_stepper(top)
    route.model.install(route.cp.asarray(fields['state']))
    for name, value in route.model.solver.ledger.items():
        route.cp.copyto(value, route.cp.asarray(fields['ledger_'+name]))
    route.tree_state = route.cp.asarray(fields['tree_state'])
    route.model.time_s = float(fields['time_s'])
    route.model.accepted_steps = int(fields['accepted_steps'])
    route.model.maximum_cfl = float(fields['maximum_cfl'])
    route.model.solver.euler_calls = int(fields['euler_calls'])
    if integrator == 'implicit_fwave_midpoint':
        controller = route.model.solver.implicit_midpoint
        require(controller is not None and controller.suspended_composition,
                'M2-C restored solver route differs')
        controller.restore_checkpoint(fields)
        route.model.solver.flow2dh_boundary_diffusion.restore_boundary_memory(
            fields)
        controller.accepted_steps = int(
            fields['implicit_controller_accepted_steps'])
        controller.rejected_steps = int(
            fields['implicit_controller_rejected_steps'])
        controller.fwave_evaluations = int(fields['implicit_fwave_evaluations'])
        controller.linear_matvecs = int(fields['implicit_linear_matvecs'])
        route.cp.copyto(route.model.wetdry_phase_ledger,
                        route.cp.asarray(fields['wetdry_phase_ledger']))
        route.cp.copyto(route.model.last_wetdry_phase_change,
                        route.cp.asarray(
                            fields['wetdry_last_suspended_change_kg']))
    route.model.sync_accepted_clock()
    route.events = events
    route.selector_last_bed = (route.cp.asarray(fields['selector_last_bed'])
        if 'selector_last_bed' in fields else None)
    return route, delay


def restore_recursive(folder, session, *, source_folder,
                      transport_source_folder, method_identity):
    prepared, top, entries, fields, events, delay = read_recursive(
        folder, session, method_identity=method_identity)
    from .amr_general_recursive_route import RecursiveGeneralOnDemandRoute

    route = RecursiveGeneralOnDemandRoute(session, source_folder=source_folder,
        transport_source_folder=transport_source_folder)
    route.entries = entries
    route.prepared = prepared
    route.by_leaves = {tuple(map(int, item.node_ids)): item
                       for item in prepared.topologies}
    route.selector = route._selector(prepared)
    route.model = route._make_stepper(top)
    model, cp = route.model, route.cp
    model.install(cp.asarray(fields['state']))
    for name, value in model.solver.ledger.items():
        cp.copyto(value, cp.asarray(fields['ledger_'+name]))
    route.tree_state = cp.asarray(fields['tree_state'])
    model.time_s = float(fields['time_s'])
    model.accepted_steps = int(fields['accepted_steps'])
    model.maximum_cfl = float(fields['maximum_cfl'])
    model.solver.euler_calls = int(fields['euler_calls'])
    np.copyto(model.level_steps, fields['recursive_level_steps'])
    model.epoch_origin = dict(
        accepted_steps=int(fields['recursive_epoch_start_steps']),
        time_s=float(fields['recursive_epoch_start_time_s']),
        euler_calls=int(fields['recursive_epoch_start_euler_calls']))
    cp.copyto(model.accepted_integral, cp.asarray(fields['recursive_register']))
    cp.copyto(model.last_source, cp.asarray(fields['recursive_last_source']))
    cp.copyto(model.last_exchange, cp.asarray(fields['recursive_last_exchange']))
    for state in (model.work, model.dense_old, model.dense_new):
        for key in state:
            cp.copyto(state[key], model.solver.state[key])
    model.dense_times.fill(model.time_s)
    model.macro_cfl = 0.
    model.trial_source.fill(0.)
    model.trial_exchange.fill(0.)
    for values in (model.level_integrals, model.subtree_integrals):
        for value in values:
            value.fill(0.)
    model.sync_accepted_clock()
    route.events = events
    route.selector_last_bed = (cp.asarray(fields['selector_last_bed'])
        if 'selector_last_bed' in fields else None)
    return route, delay
