"""Strict NumPy-only validation of the recursive bank-AMR persistence format.

This module never imports a worker, CUDA, or executes archived Python. Source
identities are reconstructed from the authenticated, copied source inventory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

import numpy as np

from .gpu_frame_io import _array, _scalar, _string_scalar, _identity, _fail

SCHEMA = 'recursive_bank_amr_v1'
_HISTORY = ('undercut', 'reference_height', 'toe_height', 'reference_length',
            'toe_cumulative', 'collapse_cumulative')
_ADAPTER = ('amr_recursive_route.py', 'amr_catalog_bank.py', 'amr_catalog_mixed_transition.py',
            'amr_bank_wetfront.py', 'amr_prepared.py', 'amr_on_demand.py',
            'amr_general_selector.py', 'amr_evolving_bank.py',
            'amr_face_correspondence.py')


def _require(value, message):
    if not value:
        _fail('Recursive AMR: ' + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _forcing(data):
    surveyed = np.array_equal(data['mode'], [9, 4])
    scheduled = surveyed or any(np.any(data.get(key, 0)) for key in
        ('nhydro_series', 'nfraction_series', 'fraction_bed_input')) or not np.all(data['mode'] == 2)
    return scheduled, surveyed


def validate_adaptation_method(method, data):
    """Bind the declared recursive closure to the authenticated input controls."""
    from .gpu_on_demand_prepared import _recursive_adaptation_scope
    if _recursive_adaptation_scope(data):
        expected = dict(policy='full_leaf_steady_v1',
                        length_diameters=float(data['bedload_adaptation_length_diameters']))
        value = method.get('bedload_adaptation')
        _require(isinstance(value, dict) and value == expected and
                 type(value.get('length_diameters')) is float,
                 'bedload adaptation method differs from prepared controls')
    else:
        _require('bedload_adaptation' not in method,
                 'adaptation OFF cannot carry an ON method')


def validate_bank_bedform_method(method, data):
    """Bind the combined method label to the exact authenticated input."""
    from .gpu_on_demand_prepared import _recursive_bank_bedform_avalanche_scope
    from .gpu_on_demand_frame_io import (
        RECURSIVE_BANK_BEDFORM_METHODS,
        RECURSIVE_BANK_BEDFORM_METHOD_KEY, _typed_equal)
    enabled = _recursive_bank_bedform_avalanche_scope(data)
    value = method.get(RECURSIVE_BANK_BEDFORM_METHOD_KEY)
    if enabled:
        bank_mode = int(data['bank_toe_erosion_mode'])
        _require(bank_mode in RECURSIVE_BANK_BEDFORM_METHODS and
                 _typed_equal(value, RECURSIVE_BANK_BEDFORM_METHODS[bank_mode]),
                  'bank bedform/avalanche method differs from prepared controls')
    else:
        _require(RECURSIVE_BANK_BEDFORM_METHOD_KEY not in method,
                 'bank bedform OFF cannot carry the combined method')
    return enabled


def source_contract(root, manifest, data):
    """Reconstruct the engine's logical hash inventory, not its executable code."""
    inventory = manifest['source_sha256']
    combined_bedform = validate_bank_bedform_method(manifest['method'], data)
    sources = {}
    def add(logical, relative):
        _require(relative in inventory, 'source is absent: ' + relative)
        sources[logical] = inventory[relative]
    for relative in sorted(inventory):
        path = PurePosixPath(relative)
        if path.parent.as_posix() == 'python/validation/gpu_fwave' and path.suffix == '.py':
            add('operators/' + path.name, relative)
        if relative.startswith('kernels/'):
            tail = relative.removeprefix('kernels/')
            if path.suffix in ('.cu', '.py'):
                add(('cuda/' if path.suffix == '.cu' else 'generated/') + tail, relative)
            if path.name == 'general_live_bank_sources.json':
                add('receipts/' + tail, relative)
                receipt = json.loads((root/'source'/relative).read_text(encoding='utf-8'))
                for name in receipt['sources']:
                    item = PurePosixPath(name)
                    _require(not item.is_absolute() and '..' not in item.parts and '\\' not in name,
                             'unsafe live source member')
                    if item.suffix == '.py':
                        add('live/' + name, 'python/' + name)
    drivers = ['amr_bank_subcycle.py', 'a1_full_array_transaction.py', 'atomic_output.py',
               'amr_level_euler.py', 'amr_bank_repair.py', 'amr_bank_wet.py']
    # Schema predates the shared engine. Preserve read-only inspection of old
    # archives while binding all newly produced checkpoints to the scheduler.
    if 'python/braids_gpu/amr_recursive_engine.py' in inventory:
        drivers.insert(1, 'amr_recursive_engine.py')
    for name in drivers:
        add('driver/' + name, 'python/braids_gpu/' + name)
    for logical, name in (('recursive_register', 'amr_bank_subcycle'), ('level_euler', 'amr_level_euler'),
                          ('reflux_repair', 'amr_bank_repair'), ('wet_preparation', 'amr_bank_wet')):
        add('cuda/' + logical + '.cu', 'python/braids_gpu/' + name + '.cu')
    scheduled, surveyed = _forcing(data)
    if scheduled:
        add('driver/amr_boundary_forcing.py', 'python/braids_gpu/amr_boundary_forcing.py')
    bank_mode = int(data['bank_toe_erosion_mode'])
    _require(bank_mode in (2, 3) and
             manifest['method'].get('bank_mode') == bank_mode,
             'source/method bank mode differs')
    evolving = bool(manifest['method'].get('evolving_banks', False))
    _require(not (bank_mode == 3 and not evolving),
             'distributed recursive source requires evolving banks')
    secondary = int(data['secondary_flow_model']) != 0
    if secondary:
        add('driver/amr_level_secondary.py', 'python/braids_gpu/amr_level_secondary.py')
        add('cuda/level_secondary.cu', 'python/braids_gpu/amr_level_secondary.cu')
    identity = _digest(sources)
    method = ('research_recursive_mode3_secondary_v6_' +
              ('surveyed' if surveyed else 'scheduled' if scheduled else 'constant') if secondary else
              'research_recursive_mode3_scheduled_forcing_v5' if scheduled else
              'research_recursive_mode3_wet_preparation_v4')
    if bank_mode == 2:
        method = method.replace('mode3', 'mode2_physical_toe')
    from .gpu_on_demand_prepared import _recursive_adaptation_scope
    if _recursive_adaptation_scope(data):
        method += '_adaptation_full_leaf_v1'
    if combined_bedform:
        method += '_' + (
            'recursive_bedform_mode2_physical_toe_parent_avalanche_v1'
            if bank_mode == 2 else
            'recursive_bedform_mode3_parent_avalanche_v1')
    base_schema = 7 if bank_mode == 2 else 5 if secondary else 4 if scheduled else 3
    return dict(source_identity=identity,
        method_identity=_digest(dict(method=method, source_identity=identity, wet_depth=1.e-6)),
        adapter_sources={name: inventory['python/braids_gpu/' + name] for name in _ADAPTER},
        bank_mode=bank_mode, evolving_banks=evolving,
        engine_schema=base_schema,
        engine_schemas=((7, 8) if bank_mode == 2 else (base_schema, 6)))


def _topology_digest(initial, levels):
    # Mirror normalized input representation only; no numerical operation.
    data = dict(initial)
    data.update(q=np.ascontiguousarray(data['state'][:3]), dry=data['dry_tolerance'],
                dt=data['dtmax'], bedoffset=data['boundary_bed_offset'],
                bc_hydraulic=np.ascontiguousarray(data['forcing'][:2]),
                bc_bed_input=data['fraction_bed_input'].copy())
    data['outlet_extension'] = int(data.get('outlet_extension', data.get('cpu_outlet_extension', 0)))
    for target, source in (('dry', 'dry_tolerance'), ('cfl_target', 'cfl'), ('manning_cell', 'manning'),
                           ('layer_mass', 'layers'), ('rock', 'bedrock'), ('morfac', 'morphological_factor')):
        if target not in data and source in data:
            data[target] = data[source]
    for name in ('dry', 'grav', 'cfl_target', 'speed_limit', 'dx', 'dy'):
        data[name] = float(np.asarray(data[name]).item())
    data['dt'] = float(np.asarray(data.get('dt', data.get('dtmax', .001))).item())
    if int(data['secondary_flow_model']) == 2:
        data['secondary_mass'] = np.ascontiguousarray(data['state'][int(data['secondary_index'])])
    elif 'secondary_mass' not in data:
        data['secondary_mass'] = np.zeros(data['nc'], np.float64)
    data['suspended_mass'] = np.ascontiguousarray(data['state'][np.asarray(data['suspended_indices']).astype(int)])
    data['recursive_cell_levels'] = levels
    digest = hashlib.sha256()
    for key, value in sorted(data.items()):
        # The recursive worker's NumPy-2 Windows scalar integers are int64.
        # QGIS still ships NumPy 1, whose default scalar integer is int32;
        # preserve the producer representation rather than the viewer ABI.
        array = np.asarray(value, dtype=np.int64) if type(value) is int else np.asarray(value)
        _require(array.dtype.kind in 'fiub', 'nonnumeric operator field ' + key)
        header = json.dumps([key, array.dtype.str, array.shape], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'little')); digest.update(header)
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def validate_frame_diagnostics(fields, topology, index, steps, clock):
    data = topology['initial_data']
    levels = index['tree']['node_level'][topology['node_ids']]
    nlevels = int(np.max(levels)) + 1
    result = {key: _array(fields, key, shape, dtype).copy() for key, shape, dtype in (
        ('recursive_level_steps', (nlevels,), 'int64'),
        ('recursive_epoch_start_steps', (), 'int64'), ('recursive_epoch_start_time', (), 'float64'),
        ('recursive_epoch_start_euler_calls', (), 'int64'),
        ('recursive_register', (data['nf'], 2, data['nstate']), 'float64'),
        ('recursive_repair_totals', (data['nfrac']+1, 4), 'float64'),
        ('recursive_preparation_impulse', (2, 2), 'float64'))}
    origin, time, euler = (result[key].item() for key in
        ('recursive_epoch_start_steps', 'recursive_epoch_start_time', 'recursive_epoch_start_euler_calls'))
    _require(0 <= origin <= steps and 0 <= time <= clock and euler >= 0 and euler % 3 == 0 and
             (origin == 0) == (time == 0) == (euler == 0), 'invalid topology epoch clock')
    expected = np.array([(steps-origin)*(1 << level) if np.any(levels == level) else 0
                         for level in range(nlevels)], np.int64)
    _require(np.array_equal(result['recursive_level_steps'], expected), 'level-step counters differ')
    reg = result['recursive_register']; bed = np.asarray(data['bed_indices'])[0]
    secondary = int(data['secondary_index'])
    used = np.r_[np.arange(3), bed, [secondary] if secondary >= 0 else []].astype(int)
    _require(np.array_equal(reg[:, 0, bed], reg[:, 1, bed]) and
             not np.any(reg[:, :, np.setdiff1d(np.arange(data['nstate']), used)]) and
             (steps > origin or not np.any(reg)) and
             (secondary < 0 or np.array_equal(reg[:, 0, secondary], reg[:, 1, secondary])),
             'invalid or nonconservative accepted register')
    totals = result['recursive_repair_totals']
    _require(np.all(totals[:, :2] >= 0) and np.all(totals[:, 0] == np.floor(totals[:, 0])) and
             not np.any(totals[1:, 2:]) and (steps > 0 or not np.any(totals)) and
             (steps > 0 or not np.any(result['recursive_preparation_impulse'])),
             'invalid conservative-repair diagnostics')
    return result


def _static_lineage(prepared, topology):
    """Recover the already-authenticated local face order for static mode 2."""
    matches = [entry for entry in prepared['entries']
               if np.array_equal(entry['selected_ids'], topology['node_ids'])]
    _require(len(matches) == 1, 'static physical-toe operator entry differs')
    path = matches[0]['folder'] / 'entry.json'
    try:
        with path.open('rb') as stream:
            identity = hashlib.file_digest(stream, 'sha256').hexdigest()
        row = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('Recursive AMR: cannot read static toe lineage') from exc
    _require(identity == matches[0]['entry_sha256'] and
             row.get('selected_node_ids') == topology['node_ids'].tolist() and
             isinstance(row.get('lineage'), list) and row['lineage'],
             'static physical-toe lineage identity differs')
    lineage = sorted(row['lineage'], key=lambda item: item.get('segment_id', -1))
    _require(all(isinstance(item, dict) and
                 type(item.get('segment_id')) is int and item['segment_id'] > 0 and
                 type(item.get('local_face')) is int
                 for item in lineage), 'invalid static physical-toe lineage')
    ids = np.asarray([item['segment_id'] for item in lineage], np.int64)
    faces = np.asarray([item['local_face'] for item in lineage], np.int64)
    data = topology['initial_data']
    expected = np.flatnonzero(np.asarray(data['is_toe_face']) != 0)
    _require(len(np.unique(ids)) == len(ids) and len(np.unique(faces)) == len(faces) and
             np.array_equal(np.sort(faces), expected),
             'static physical-toe lineage faces differ')
    return ids, faces


def validate_payload(fields, *, prepared, index, method_identity, worker, end, contract):
    """Return selected topology and synchronous-shaped fields for final comparison."""
    strings = ('catalog_identity', 'method_identity', 'topology_identity', 'events_json')
    values = {name: _string_scalar(fields, name) for name in strings}
    _require(values['catalog_identity'] == prepared['session_identity'] and
             values['method_identity'] == method_identity, 'worker method/session differs')
    candidates = [top for top in index['topologies'].values()
                  if top['operator_sha256'] == values['topology_identity']]
    _require(len(candidates) == 1, 'unknown selected operator')
    top = candidates[0]; data = top['initial_data']
    nc, nf, ns, ng, nl, nb = (int(data[k]) for k in ('nc', 'nf', 'nstate', 'nfrac', 'nbed_layers', 'nb'))
    nn = len(index['tree']['parent']); levels = index['tree']['node_level'][top['node_ids']]
    engine = {key.removeprefix('engine_'): value for key, value in fields.items() if key.startswith('engine_')}
    bank_mode = int(data['bank_toe_erosion_mode'])
    evolving = bool(prepared['session_manifest']['schema'] == 2)
    _require(bank_mode == contract['bank_mode'] and
             evolving == contract['evolving_banks'],
             'checkpoint bank route differs from numerical source')
    schema = int(_scalar(engine, 'schema', 'int64'))
    _require(schema in contract['engine_schemas'],
             'unsupported engine schema for physics')
    floats = dict(time_s=(), maximum_cfl=(), state_q=(3, nc), state_layers=(nl, ng, nc), state_secondary=(nc,),
        accepted_integral=(nf, 2, ns), repair_totals=(ng+1, 4), hydraulic_preparation_impulse=(2, 2),
        ledger_water=(nb,), ledger_bed=(ng, nb), ledger_secondary=(nb,), ledger_source=())
    floats.update({'history_' + key: (nc,) for key in _HISTORY})
    ints = {key: () for key in ('schema', 'accepted_steps', 'euler_calls', 'bank_updates', 'avalanche_updates',
                                'collapse_waves')}
    stage_name = 'distributed_stage_index' if bank_mode == 3 else 'toe_stage_index'
    ints[stage_name] = ()
    ints['level_steps'] = (int(np.max(levels))+1,)
    epoch_schema = schema == (6 if bank_mode == 3 else 8)
    if epoch_schema:
        floats['epoch_start_time_s'] = ()
        ints.update(epoch_start_steps=(), epoch_start_euler_calls=())
    if bank_mode == 2:
        floats.update(toe_signed=(ng, nf), toe_gross=(ng, nf))
        if evolving:
            from .gpu_frame_io import _toe_map_schema
            for key, (shape, dtype) in _toe_map_schema(nc, nf).items():
                (ints if dtype == 'int64' else floats if dtype == 'float64' else ints)[key] = shape
    names = {'method_identity', 'source_identity', 'topology_identity'}
    _require(set(engine) == set(floats) | set(ints) | names | {'history_collapse_count'}, 'engine field inventory differs')
    for key, shape in floats.items(): _array(engine, key, shape, 'float64')
    for key, shape in ints.items():
        dtype = ('int32' if key.startswith('toe_map_') else 'int64')
        value = _array(engine, key, shape, dtype)
        if not key.startswith('toe_map_'):
            _require(np.all(value >= 0), 'negative engine counter ' + key)
    _array(engine, 'history_collapse_count', (nc,), 'int32')
    for key in ('source_identity', 'method_identity'):
        _require(_string_scalar(engine, key) == contract[key], 'engine ' + key + ' differs')
    _require(_string_scalar(engine, 'topology_identity') == _topology_digest(data, levels), 'engine input topology differs')
    temporal = prepared['amr_settings']['refine_sediment']
    outer = set(strings) | {'tree_state', 'delay_state', 'node_ids', 'time_s', 'accepted_steps'}
    if temporal: outer.add('selector_last_bed')
    face_count = 0
    if bank_mode == 2:
        meta = {'recursive_bank_mode', 'recursive_evolving_toe_variant',
                'face_epoch_count'}
        for key in meta:
            _scalar(fields, key, 'int64')
        face_count = int(fields['face_epoch_count'])
        _require(int(fields['recursive_bank_mode']) == 2 and
                 int(fields['recursive_evolving_toe_variant']) == int(evolving) and
                 0 <= face_count <= len(fields)//6 and
                 (evolving or face_count == 0),
                 'recursive physical-toe metadata differs')
        outer |= meta | {f'face_epoch_{i}_{key}' for i in range(face_count)
                         for key in ('operator_identity', 'node_ids', 'old_face',
                                     'fraction', 'signed', 'gross')}
    _require(set(fields) == outer | {'engine_' + key for key in engine}, 'outer field inventory differs')
    _array(fields, 'tree_state', (nn, ns), 'float64')
    _require(np.all(_array(fields, 'delay_state', (nn,), 'int64') >= 0) and
             np.array_equal(_array(fields, 'node_ids', (nc,), 'int64'), top['node_ids']), 'selector/global nodes differ')
    if temporal: _array(fields, 'selector_last_bed', (nn,), 'float64')
    step, clock = int(_scalar(fields, 'accepted_steps', 'int64')), float(_scalar(fields, 'time_s', 'float64'))
    origin = dict(accepted_steps=int(engine.get('epoch_start_steps', 0)),
                  time_s=float(engine.get('epoch_start_time_s', 0.)),
                  euler_calls=int(engine.get('epoch_start_euler_calls', 0)))
    diagnostic = dict(recursive_level_steps=engine['level_steps'], recursive_register=engine['accepted_integral'],
        recursive_repair_totals=engine['repair_totals'], recursive_preparation_impulse=engine['hydraulic_preparation_impulse'],
        recursive_epoch_start_steps=np.asarray(origin['accepted_steps'], np.int64),
        recursive_epoch_start_time=np.asarray(origin['time_s'], np.float64),
        recursive_epoch_start_euler_calls=np.asarray(origin['euler_calls'], np.int64))
    validate_frame_diagnostics(diagnostic, top, index, step, clock)
    _require(step == worker['steps'] == int(engine['accepted_steps']) and
             clock == worker['time_s'] == float(engine['time_s']) and (clock == 0) == (step == 0) and
             clock <= min(end, origin['time_s'] + (step-origin['accepted_steps'])*prepared['dtmax_s'])+1.e-12 and
             int(engine['euler_calls']) == origin['euler_calls'] + 3*int(engine['level_steps'].sum()) and
             int(engine['bank_updates']) == step and
             int(engine['avalanche_updates']) == (step if data['general_avalanche'] else 0) and
             int(engine[stage_name]) == 0 and
             int(engine['collapse_waves']) <= (nc if bank_mode == 3 else 0) and
             0 <= float(engine['maximum_cfl']) <= prepared['cfl']+1.e-12,
             'accepted clock/counters differ')
    state = np.zeros((nc, ns), np.float64); state[:, :3] = engine['state_q'].T
    for layer in range(nl): state[:, data['bed_indices'][layer]] = engine['state_layers'][layer].T
    si = int(data['secondary_index'])
    if si >= 0: state[:, si] = engine['state_secondary']
    _require(si >= 0 or not np.any(engine['state_secondary']), 'secondary OFF state is nonzero')
    for name, value in (('selected', state), ('tree', fields['tree_state'])):
        _require(np.all(value[:, 0] >= 0) and np.all(value[:, np.asarray(data['bed_indices']).ravel()] >= 0) and
                 not np.any(value[:, data['suspended_indices']]), 'invalid physical inventory in ' + name)
        dry = value[:, 0] == 0
        _require(not np.any(value[dry, 1:3]) and (si < 0 or not np.any(value[dry, si])), 'dry inventory in ' + name)
    _require(not np.any(engine['state_q'][1:, engine['state_q'][0] <= data['dry_tolerance']]) and
             all(np.all(engine['history_' + key] >= 0) for key in (*_HISTORY, 'collapse_count')),
             'negative bank history or dry momentum')
    if bank_mode == 2:
        tolerance = 1.e-12*np.maximum(1., np.maximum(
            np.abs(engine['toe_signed']), engine['toe_gross']))
        boundary = np.asarray(data['neighbour']) < 0
        _require(np.all(engine['toe_gross'] >= 0.) and
                 np.all(engine['toe_gross']+tolerance >= np.abs(engine['toe_signed'])) and
                 not np.any(engine['toe_signed'][:, boundary]) and
                 not np.any(engine['toe_gross'][:, boundary]),
                 'invalid recursive physical-toe face diagnostics')
        if not evolving:
            inactive = np.asarray(data['is_toe_face']) == 0
            _require(not np.any(engine['toe_signed'][:, inactive]) and
                     not np.any(engine['toe_gross'][:, inactive]),
                     'static recursive non-toe face carries history')
    events = json.loads(values['events_json'])
    _validate_events(events, index, prepared['session_identity'], top, step,
                     clock, origin, schema, bank_mode=bank_mode)
    if bank_mode == 2:
        _require(face_count == (len(events) if evolving else 0),
                 'recursive physical-toe face epoch count differs')
        if evolving:
            for number, event in enumerate(events):
                old = index['topologies'][event['old_topology']]
                _require(_string_scalar(fields,
                    f'face_epoch_{number}_operator_identity') ==
                    old['operator_sha256'],
                    'retired physical-toe epoch/event operator differs')
    flattened = dict(fields, state=state, distributed_stage_index=engine['distributed_stage_index'],
                     distributed_collapse_waves=engine['collapse_waves']) if bank_mode == 3 else dict(fields, state=state)
    flattened.update({key: value for key, value in engine.items() if key.startswith(('history_', 'ledger_'))})
    if bank_mode == 2:
        flattened.update({key:value for key,value in engine.items()
                          if key.startswith('toe_map_') or key in ('toe_signed','toe_gross')})
        if evolving:
            from .gpu_on_demand_frame_io import validate_evolving_bank_fields
            translated = dict(flattened)
            translated.update({'bank_'+key:value for key,value in flattened.items()
                               if key.startswith('history_')})
            validate_evolving_bank_fields(translated, top, index)
        else:
            ids, faces = _static_lineage(prepared, top)
            flattened.update(toe_segment_ids=ids,
                toe_signed=engine['toe_signed'][:, faces].copy(),
                toe_gross=engine['toe_gross'][:, faces].copy())
    flattened['recursive'] = diagnostic
    return top, flattened


def _validate_events(events, index, session_identity, selected, steps, clock,
                     origin, schema, *, bank_mode):
    _require(isinstance(events, list), 'invalid mesh event list')
    by_name = index['topologies']; previous = next(iter(by_name.values()))
    prior = dict(accepted_steps=0, time_s=0., euler_calls=0)
    for event in events:
        _require(isinstance(event, dict) and event.get('old_topology') == previous['name'] and
                 event.get('new_topology') in by_name and event.get('new_topology') != previous['name'] and
                 event.get('catalog_identity') == session_identity and
                 event.get('old_operator_identity') == previous['operator_sha256'] and
                 event.get('new_operator_identity') == by_name[event['new_topology']]['operator_sha256'],
                 'mesh event topology chain differs')
        epoch = event.get('recursive_epoch')
        epoch_fields = {'previous_origin', 'end_step', 'end_time_s', 'euler_calls',
            'level_steps', 'old_operator_identity', 'consumed_register_shape', 'consumed_register_sha256',
            'consumed_by_bank', 'remapped_as_pending_flux'}
        if bank_mode == 2:
            epoch_fields |= {'bank_mode', 'physical_toe_integral_consumed',
                             'evolving_toe_variant'}
        _require(isinstance(epoch, dict) and set(epoch) == epoch_fields,
                 'mesh epoch inventory differs')
        end_step, end_time = epoch['end_step'], epoch['end_time_s']
        _require(type(end_step) is int and type(end_time) in (int, float) and np.isfinite(end_time) and
                 prior['accepted_steps'] <= end_step <= steps and prior['time_s'] <= end_time <= clock and
                 epoch['previous_origin'] == prior and event.get('step') == end_step and event.get('time_s') == end_time and
                 epoch['old_operator_identity'] == previous['operator_sha256'] and
                 epoch['consumed_by_bank'] is True and epoch['remapped_as_pending_flux'] is False and
                 (bank_mode != 2 or (epoch['bank_mode'] == 2 and
                  epoch['physical_toe_integral_consumed'] is True and
                  type(epoch['evolving_toe_variant']) is bool)),
                 'mesh epoch accepted ownership differs')
        data = previous['initial_data']; levels = index['tree']['node_level'][previous['node_ids']]
        expected = [(end_step-prior['accepted_steps'])*(1 << level) if np.any(levels == level) else 0
                    for level in range(int(np.max(levels))+1)]
        _require(epoch['level_steps'] == expected and type(epoch['euler_calls']) is int and
                 epoch['euler_calls'] == prior['euler_calls']+3*sum(expected) and
                 epoch['consumed_register_shape'] == [data['nf'], 2, data['nstate']], 'retired register/counters differ')
        _identity(epoch['consumed_register_sha256'], 'consumed recursive register')
        prior = dict(accepted_steps=end_step, time_s=end_time, euler_calls=epoch['euler_calls'])
        previous = by_name[event['new_topology']]
    _require(previous['name'] == selected['name'] and origin == prior and
             (not events or schema == (6 if bank_mode == 3 else 8)),
             'mesh event chain does not reach current epoch')
