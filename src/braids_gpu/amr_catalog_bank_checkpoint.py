"""Host-validated restart of a synchronous, state-selected bank-AMR route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .atomic_output import save_arrays_atomic, write_json_atomic


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _validate_legacy(fields, prepared, *, method_identity, lineages):
    """No CuPy import or device construction occurs before complete braids_support."""
    from .amr_catalog_bank import validate_lineages
    maps = validate_lineages(prepared, lineages)
    _require(str(fields['catalog_identity']) == prepared.identity and
             str(fields['method_identity']) == method_identity and
             int(fields['schema']) == 1, 'Bank-AMR checkpoint method/catalog differs')
    candidates = [top for top in prepared.topologies
                  if top.source_identity == str(fields['topology_identity'])]
    _require(len(candidates) == 1, 'Checkpoint operator is outside catalog')
    top = candidates[0]
    n, nn, ns = len(top.node_ids), len(prepared.tree['parent']), int(top.data['nstate'])
    ng = int(top.data['nfrac'])
    ids, _ = maps[top.name]
    shapes = dict(time_s=(), maximum_cfl=(), state=(n,ns), tree_state=(nn,ns),
        ledger_water=(1,), ledger_bed=(ng,1), ledger_secondary=(1,), ledger_source=(),
        toe_signed=(ng,len(ids)), toe_gross=(ng,len(ids)))
    history = ('undercut','reference_height','toe_height','reference_length',
               'toe_cumulative','collapse_cumulative')
    shapes.update({'history_'+key:(n,) for key in history})
    integers = dict(accepted_steps=(), euler_calls=(), node_ids=(n,),
                    toe_segment_ids=(len(ids),), history_collapse_count=(n,), delay_state=(nn,))
    strings = ('catalog_identity','method_identity','topology_identity','events_json')
    _require(set(fields) == set(shapes)|set(integers)|set(strings)|{'schema'},
             'Checkpoint payload fields differ')
    for name, shape in shapes.items():
        value = np.asarray(fields[name])
        _require(value.shape == shape and value.dtype == np.float64 and
                 np.isfinite(value).all(), 'Invalid finite checkpoint field '+name)
    for name, shape in integers.items():
        value = np.asarray(fields[name])
        _require(value.shape == shape and value.dtype.kind in 'iu' and np.all(value>=0),
                 'Invalid integer checkpoint field '+name)
    for name in strings:
        _require(np.asarray(fields[name]).shape == () and np.asarray(fields[name]).dtype.kind == 'U',
                 'Invalid checkpoint string '+name)
    _require(np.array_equal(fields['node_ids'], top.node_ids) and
             np.array_equal(fields['toe_segment_ids'], ids), 'Checkpoint global identities differ')
    step, current = int(fields['accepted_steps']), float(fields['time_s'])
    _require(int(fields['euler_calls']) == 3*step and 0 <= current <= float(top.data['endtime'])+1e-12 and
             (step == 0) == (current == 0) and 0 <= float(fields['maximum_cfl']) <=
             float(top.data['cfl_target'])+1e-12, 'Invalid accepted checkpoint clock/CFL')
    for key in ('state','tree_state'):
        state = fields[key]
        _require(np.all(state[:,0]>=0) and
                 np.all(state[:,np.asarray(top.data['bed_indices']).ravel()]>=0) and
                 np.all(state[:,np.asarray(top.data['suspended_indices'])]==0),
                 'Invalid checkpoint water/sediment '+key)
        _require(np.all(state[state[:,0]==0,1:3]==0) and
                 np.all(state[state[:,0]==0,int(top.data['secondary_index'])]==0),
                 'Dry checkpoint rows carry momentum/secondary '+key)
    _require(all(np.all(fields['history_'+key]>=0) for key in history) and
             np.all(fields['toe_gross']>=np.abs(fields['toe_signed'])-1e-12),
             'Invalid checkpoint bank/toe history')
    events = json.loads(str(fields['events_json']))
    _require(isinstance(events,list), 'Invalid checkpoint event list')
    by_name = {item.name:item for item in prepared.topologies}
    previous = prepared.topologies[0].name
    previous_step = -1
    for event in events:
        _require(isinstance(event,dict) and event.get('old_topology')==previous and
                 event.get('new_topology') in by_name and
                 previous_step < event['step'] <= step and 0 < event['time_s'] <= current and
                 event.get('catalog_identity')==prepared.identity and
                 event.get('old_operator_identity')==by_name[previous].source_identity and
                 event.get('new_operator_identity')==by_name[event['new_topology']].source_identity,
                 'Invalid checkpoint mesh-event chain')
        previous, previous_step = event['new_topology'], event['step']
    _require(previous == top.name, 'Checkpoint events do not reach selected topology')
    return top, events


def _field(fields, name, shape, dtype):
    value = np.asarray(fields.get(name))
    _require(value.shape == shape and value.dtype == np.dtype(dtype),
             'Invalid checkpoint shape/dtype '+name)
    if value.dtype.kind == 'f':
        _require(np.isfinite(value).all(), 'Nonfinite checkpoint field '+name)
    return value


def _string(fields, name):
    value = np.asarray(fields.get(name))
    _require(value.shape == () and value.dtype.kind == 'U',
             'Invalid checkpoint string '+name)
    return str(value.item())


def _signed_gross(signed, gross, label):
    _require(np.all(gross >= 0.) and
             np.all(gross + 1.e-12*np.maximum(1., np.abs(signed)) >= np.abs(signed)),
             'Invalid signed/gross checkpoint ledger '+label)


def _validate_map(fields, top):
    """Validate all persistent map/support indices before any CUDA use.

    The resident mapper additionally verifies canonical physical consistency
    when restoring from the accepted depth/bed/layers. No map is silently
    regenerated in place of missing checkpoint support.
    """
    from braids_support.gpu_fwave.evolving_toe_map import checkpoint_specs
    nc, nf = len(top.node_ids), len(top.data['owner'])
    specs = checkpoint_specs(nc, nf)
    state = {key: _field(fields, 'toe_map_'+key, shape, dtype)
             for key, (shape, dtype) in specs.items()}
    _require(int(state['initialized']) == 1, 'Uninitialized evolving checkpoint map')
    for key in ('channel','bank','is_toe_cell','is_toe_donor','is_toe_face','ambiguous',
                'support_ambiguous','support_root','support_live'):
        _require(np.isin(state[key], (0, 1)).all(), 'Invalid checkpoint map flag '+key)
    _require(not np.any(state['bank'] & state['channel']),
             'Checkpoint map bank/channel roles overlap')
    for key in ('candidate_bank','candidate_toe','mapped_donor','mapped_toe',
                'support_donor','support_toe','support_face_bank','support_face_toe'):
        _require(np.all((state[key] >= -1) & (state[key] < nc)),
                 'Invalid checkpoint map cell IDs '+key)
    for key in ('mapped_face','support_face'):
        _require(np.all((state[key] >= -1) & (state[key] < nf)),
                 'Invalid checkpoint map face IDs '+key)
    for key in ('distance','support_distance'):
        _require(np.all(state[key] >= 0.), 'Negative checkpoint map distance '+key)
    owner, neighbour = (np.asarray(top.data[k]) for k in ('owner','neighbour'))
    for first, second, active in (
            ('candidate_bank', 'candidate_toe', state['is_toe_face'] != 0),
            ('support_face_bank', 'support_face_toe', state['support_face_bank'] >= 0)):
        bank, toe = state[first], state[second]
        _require(np.all(bank[~active] == -1) and np.all(toe[~active] == -1) and
                 np.all(bank[active] >= 0) and np.all(toe[active] >= 0) and
                 np.all(neighbour[active] >= 0), 'Invalid checkpoint map face support')
        _require(np.all(((owner[active] == bank[active]) & (neighbour[active] == toe[active])) |
                        ((owner[active] == toe[active]) & (neighbour[active] == bank[active]))),
                 'Checkpoint face support differs from topology')
    active = state['is_toe_face'] != 0
    _require(np.all(state['bank'][state['candidate_bank'][active]] == 1) and
             np.all(state['channel'][state['candidate_toe'][active]] == 1),
             'Checkpoint active face roles differ from mapped cells')
    for donor, toe, face, distance, bank_face, toe_face in (
            ('mapped_donor','mapped_toe','mapped_face','distance','candidate_bank','candidate_toe'),
            ('support_donor','support_toe','support_face','support_distance',
             'support_face_bank','support_face_toe')):
        valid = state[donor] >= 0
        _require(np.all((state[toe] >= 0) == valid) and
                 np.all((state[face] >= 0) == valid) and
                 np.all(state[distance][~valid] == np.finfo(np.float64).max),
                 'Incomplete checkpoint donor/toe/face support')
        _require(np.all(state[bank_face][state[face][valid]] == state[donor][valid]) and
                 np.all(state[toe_face][state[face][valid]] == state[toe][valid]),
                 'Checkpoint mapped support seed differs from face endpoints')
    # CSR tails hold all unused IDs as a permutation, never -1 padding.
    groups = (('toe_offset','toe_faces',state['candidate_toe'],None,nf),
              ('donor_offset','donor_faces',state['candidate_bank'],None,nf),
              ('candidate_offset','candidates',state['mapped_donor'],state['distance'],nc),
              ('toe_donor_offset','toe_donors',
               np.where(state['is_toe_donor'] != 0, state['mapped_toe'], -1),None,nc))
    for offset, item, keys, secondary, size in groups:
        canonical = np.where(keys >= 0, keys, nc)
        identifiers = np.arange(size, dtype=np.int32)
        order = np.lexsort((identifiers, canonical) if secondary is None else
                           (identifiers, secondary, canonical))
        counts = np.bincount(canonical, minlength=nc+1)[:nc]
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        _require(np.array_equal(state[offset], offsets) and
                 np.array_equal(state[item], identifiers[order]),
                 'Noncanonical checkpoint map CSR '+offset)
    return specs


def _validate_evolving(fields, prepared, *, method_identity):
    _require(_string(fields, 'catalog_identity') == prepared.identity and
             _string(fields, 'method_identity') == method_identity,
             'Bank-AMR checkpoint method/catalog differs')
    selected = _string(fields, 'topology_identity')
    candidates = [top for top in prepared.topologies if top.source_identity == selected]
    _require(len(candidates) == 1, 'Checkpoint operator is outside catalog')
    top = candidates[0]
    n, nn, ns = len(top.node_ids), len(prepared.tree['parent']), int(top.data['nstate'])
    ng, nf = int(top.data['nfrac']), len(top.data['owner'])
    nb = int(top.data['nb'])
    count = int(_field(fields, 'face_epoch_count', (), 'int64'))
    _require(count >= 0 and count <= len(fields)//6, 'Invalid checkpoint face epoch count')
    floats = dict(time_s=(), maximum_cfl=(), state=(n,ns), tree_state=(nn,ns),
        ledger_water=(nb,), ledger_bed=(ng,nb), ledger_secondary=(nb,), ledger_source=(),
        toe_signed=(ng,nf), toe_gross=(ng,nf))
    if int(fields['schema']) == 3:
        floats['selector_last_bed'] = (nn,)
    history = ('undercut','reference_height','toe_height','reference_length',
               'toe_cumulative','collapse_cumulative')
    floats.update({'history_'+key:(n,) for key in history})
    integers = dict(accepted_steps=(), euler_calls=(), node_ids=(n,),
                    history_collapse_count=(n,), delay_state=(nn,), face_epoch_count=())
    strings = {'catalog_identity','method_identity','topology_identity','events_json'}
    from braids_support.gpu_fwave.evolving_toe_map import checkpoint_specs
    map_names = {'toe_map_'+key for key in checkpoint_specs(n, nf)}
    epoch_names = {f'face_epoch_{i}_{key}' for i in range(count) for key in
                   ('operator_identity','node_ids','old_face','fraction','signed','gross')}
    _require(set(fields) == set(floats)|set(integers)|strings|map_names|epoch_names|{'schema'},
             'Checkpoint payload fields differ')
    for key, shape in floats.items():
        _field(fields, key, shape, 'float64')
    for key, shape in integers.items():
        value = _field(fields, key, shape, 'int32' if key == 'history_collapse_count' else 'int64')
        _require(np.all(value >= 0), 'Negative integer checkpoint field '+key)
    for key in strings:
        _string(fields, key)
    _require(np.array_equal(fields['node_ids'], top.node_ids), 'Checkpoint global node IDs differ')
    step, current = int(fields['accepted_steps']), float(fields['time_s'])
    _require(int(fields['euler_calls']) == 3*step and
             0 <= current <= float(top.data['endtime'])+1e-12 and
             (step == 0) == (current == 0) and
             0 <= float(fields['maximum_cfl']) <= float(top.data['cfl_target'])+1e-12,
             'Invalid accepted checkpoint clock/CFL')
    for key in ('state','tree_state'):
        state = fields[key]
        _require(np.all(state[:,0] >= 0) and
                 np.all(state[:,np.asarray(top.data['bed_indices']).ravel()] >= 0) and
                 not np.any(state[:,np.asarray(top.data['suspended_indices'])]),
                 'Invalid checkpoint water/sediment '+key)
        _require(not np.any(state[state[:,0] == 0,1:3]) and
                 not np.any(state[state[:,0] == 0,int(top.data['secondary_index'])]),
                 'Dry checkpoint rows carry momentum/secondary '+key)
    _require(all(np.all(fields['history_'+key] >= 0) for key in history),
             'Invalid checkpoint bank history')
    _signed_gross(fields['toe_signed'], fields['toe_gross'], 'current')
    boundary = np.asarray(top.data['neighbour']) < 0
    _require(not np.any(fields['toe_signed'][:,boundary]) and
             not np.any(fields['toe_gross'][:,boundary]),
             'External boundary carries physical-toe checkpoint history')
    _validate_map(fields, top)
    try:
        events = json.loads(_string(fields, 'events_json'))
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid checkpoint event JSON') from exc
    _require(isinstance(events, list) and len(events) == count,
             'Checkpoint face epochs/event count differs')
    by_name = {item.name:item for item in prepared.topologies}
    previous, previous_step, previous_time = prepared.topologies[0].name, 0, 0.
    for i, event in enumerate(events):
        _require(isinstance(event, dict) and event.get('old_topology') == previous and
                 event.get('new_topology') in by_name and event['new_topology'] != previous and
                 type(event.get('step')) is int and previous_step <= event['step'] <= step and
                 type(event.get('time_s')) in (int, float) and np.isfinite(event['time_s']) and
                 previous_time <= event['time_s'] <= current and
                 (event['step'] == 0) == (event['time_s'] == 0) and
                 event.get('catalog_identity') == prepared.identity and
                 event.get('old_operator_identity') == by_name[previous].source_identity and
                 event.get('new_operator_identity') == by_name[event['new_topology']].source_identity,
                 'Invalid checkpoint mesh-event chain')
        prior = by_name[previous]
        prefix = f'face_epoch_{i}_'
        _require(_string(fields, prefix+'operator_identity') == prior.source_identity,
                 'Checkpoint face epoch operator differs from regrid event')
        ids = _field(fields, prefix+'node_ids', (len(prior.node_ids),), 'int64')
        _require(np.array_equal(ids, prior.node_ids), 'Checkpoint face epoch global nodes differ')
        faces = np.asarray(fields[prefix+'old_face'])
        _require(faces.ndim == 1, 'Invalid checkpoint archived face vector')
        faces = _field(fields, prefix+'old_face', faces.shape, 'int64')
        _require(np.all((faces >= 0) & (faces < len(prior.data['owner']))) and
                 np.all(np.diff(faces) > 0), 'Invalid/duplicate checkpoint archived face IDs')
        fraction = _field(fields, prefix+'fraction', faces.shape, 'float64')
        _require(np.all((fraction > 0.) & (fraction <= 1.)),
                 'Invalid checkpoint archived face fractions')
        signed = _field(fields, prefix+'signed', (ng,len(faces)), 'float64')
        gross = _field(fields, prefix+'gross', (ng,len(faces)), 'float64')
        _signed_gross(signed, gross, prefix)
        external = np.asarray(prior.data['neighbour'])[faces] < 0
        _require(not np.any(signed[:,external]) and not np.any(gross[:,external]),
                 'External boundary carries archived physical-toe history')
        previous, previous_step, previous_time = event['new_topology'], event['step'], event['time_s']
    _require(previous == top.name, 'Checkpoint events do not reach selected topology')
    return top, events


def _validate_distributed(fields, prepared, *, method_identity, requires_selector_last_bed):
    """Mode 3 owns cell history only: no physical-toe support or face ledgers."""
    _require(type(requires_selector_last_bed) is bool,
             'Mode-3 checkpoint requires explicit temporal-selector method admission')
    _require(_string(fields, 'catalog_identity') == prepared.identity and
             _string(fields, 'method_identity') == method_identity,
             'Bank-AMR checkpoint method/catalog differs')
    candidates = [top for top in prepared.topologies
                  if top.source_identity == _string(fields, 'topology_identity')]
    _require(len(candidates) == 1 and all(int(top.data['bank_toe_erosion_mode']) == 3
             for top in prepared.topologies), 'Mode-3 checkpoint operator/catalog differs')
    top = candidates[0]
    n, nn, ns = len(top.node_ids), len(prepared.tree['parent']), int(top.data['nstate'])
    ng, nb = int(top.data['nfrac']), int(top.data['nb'])
    marker = int(_field(fields, 'selector_reference_present', (), 'int64'))
    _require(marker in (0, 1) and bool(marker) == requires_selector_last_bed,
             'Mode-3 checkpoint temporal reference differs from required method')
    history = ('undercut','reference_height','toe_height','reference_length',
               'toe_cumulative','collapse_cumulative')
    floats = dict(time_s=(), maximum_cfl=(), state=(n,ns), tree_state=(nn,ns),
        ledger_water=(nb,), ledger_bed=(ng,nb), ledger_secondary=(nb,), ledger_source=())
    floats.update({'history_'+key:(n,) for key in history})
    if marker:
        floats['selector_last_bed'] = (nn,)
    integers = dict(accepted_steps=(), euler_calls=(), node_ids=(n,), delay_state=(nn,),
        history_collapse_count=(n,), bank_mode=(), distributed_stage_index=(),
        distributed_collapse_waves=(), selector_reference_present=())
    strings = {'catalog_identity','method_identity','topology_identity','events_json'}
    _require(set(fields) == set(floats)|set(integers)|strings|{'schema'},
             'Mode-3 checkpoint payload fields differ')
    for key, shape in floats.items():
        _field(fields, key, shape, 'float64')
    for key, shape in integers.items():
        value = _field(fields, key, shape, 'int32' if key == 'history_collapse_count' else 'int64')
        _require(np.all(value >= 0), 'Negative integer checkpoint field '+key)
    for key in strings:
        _string(fields, key)
    _require(int(fields['bank_mode']) == 3 and np.array_equal(fields['node_ids'], top.node_ids),
             'Mode-3 checkpoint bank/global node identity differs')
    step, current = int(fields['accepted_steps']), float(fields['time_s'])
    _require(int(fields['euler_calls']) == 3*step and
             0 <= current <= float(top.data['endtime'])+1.e-12 and
             (step == 0) == (current == 0) and
             0 <= float(fields['maximum_cfl']) <= float(top.data['cfl_target'])+1.e-12,
             'Invalid accepted checkpoint clock/CFL')
    _require(int(fields['distributed_stage_index']) == (3 if step else 0) and
             0 <= int(fields['distributed_collapse_waves']) <= nn and
             (step > 0 or int(fields['distributed_collapse_waves']) == 0),
             'Invalid distributed-bank checkpoint stage/wave counters')
    secondary_index = int(top.data['secondary_index'])
    for key in ('state','tree_state'):
        state = fields[key]
        _require(np.all(state[:,0] >= 0) and
                 np.all(state[:,np.asarray(top.data['bed_indices']).ravel()] >= 0) and
                 not np.any(state[:,np.asarray(top.data['suspended_indices'])]),
                 'Invalid checkpoint water/sediment '+key)
        _require(not np.any(state[state[:,0] == 0,1:3]) and
                 (secondary_index < 0 or not np.any(state[state[:,0] == 0,secondary_index])),
                 'Dry checkpoint rows carry momentum/secondary '+key)
    _require(all(np.all(fields['history_'+key] >= 0) for key in history),
             'Invalid checkpoint bank history')
    try:
        events = json.loads(_string(fields, 'events_json'))
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid checkpoint event JSON') from exc
    _require(isinstance(events, list), 'Invalid checkpoint event list')
    by_name = {item.name:item for item in prepared.topologies}
    previous, previous_step, previous_time = prepared.topologies[0].name, 0, 0.
    for event in events:
        _require(isinstance(event, dict) and event.get('old_topology') == previous and
                 event.get('new_topology') in by_name and event['new_topology'] != previous and
                 type(event.get('step')) is int and previous_step <= event['step'] <= step and
                 type(event.get('time_s')) in (int,float) and np.isfinite(event['time_s']) and
                 previous_time <= event['time_s'] <= current and
                 (event['step'] == 0) == (event['time_s'] == 0) and
                 event.get('catalog_identity') == prepared.identity and
                 event.get('old_operator_identity') == by_name[previous].source_identity and
                 event.get('new_operator_identity') == by_name[event['new_topology']].source_identity,
                 'Invalid checkpoint mesh-event chain')
        previous, previous_step, previous_time = event['new_topology'], event['step'], event['time_s']
    _require(previous == top.name, 'Checkpoint events do not reach selected topology')
    return top, events


def validate(fields, prepared, *, method_identity, lineages, expected_schema=None,
             requires_selector_last_bed=None):
    """Versioned host validation; legacy static checkpoints stay unchanged."""
    schema = np.asarray(fields.get('schema'))
    _require(schema.shape == () and schema.dtype == np.dtype('int64') and
             int(schema) in (1, 2, 3, 4), 'Invalid bank-AMR checkpoint schema')
    if expected_schema is not None:
        _require(type(expected_schema) is int and expected_schema in (1, 2, 3, 4) and
                 int(schema) == expected_schema, 'Checkpoint schema differs from required method')
    if int(schema) == 1:
        return _validate_legacy(fields, prepared, method_identity=method_identity, lineages=lineages)
    if int(schema) == 4:
        return _validate_distributed(fields, prepared, method_identity=method_identity,
                                     requires_selector_last_bed=requires_selector_last_bed)
    return _validate_evolving(fields, prepared, method_identity=method_identity)


def write(folder, route, delay_state, *, method_identity, requires_selector_last_bed=None):
    folder = Path(folder)
    fields = route.frame()
    distributed = getattr(route, 'bank_mode', 2) == 3
    schema = 4 if distributed else (2 if getattr(route, 'evolving_toe_variant', False) else 1)
    last_bed = getattr(route, 'selector_last_bed', None)
    if distributed:
        _require(type(requires_selector_last_bed) is bool and
                 (last_bed is not None) == requires_selector_last_bed,
                 'Mode-3 checkpoint temporal reference differs from required method')
        fields['selector_reference_present'] = np.int64(last_bed is not None)
    if last_bed is not None:
        _require(schema in (2, 4), 'Temporal sediment reference requires dynamic bank checkpoint')
        fields['selector_last_bed'] = route.cp.asnumpy(last_bed)
        _field(fields, 'selector_last_bed', (len(route.prepared.tree['parent']),), 'float64')
        if not distributed:
            schema = 3
    else:
        _require('selector_last_bed' not in fields,
                 'Checkpoint frame has an unowned temporal sediment reference')
    fields.update(schema=np.int64(schema), catalog_identity=np.asarray(route.prepared.identity),
        method_identity=np.asarray(method_identity), tree_state=route.cp.asnumpy(route.tree_state),
        delay_state=np.asarray(delay_state,dtype=np.int64),
        events_json=np.asarray(json.dumps(route.events,sort_keys=True,allow_nan=False)))
    if distributed:
        validate(fields, route.prepared, method_identity=method_identity, lineages=None,
                 expected_schema=4, requires_selector_last_bed=requires_selector_last_bed)
    folder.mkdir(parents=True, exist_ok=False)
    save_arrays_atomic(folder/'state.npz', **fields)
    write_json_atomic(folder/'checkpoint.json', dict(schema=schema, kind='coupled_bank_amr',
        payload_sha256=_sha(folder/'state.npz'), catalog_identity=route.prepared.identity,
        method_identity=method_identity))


def read(folder, prepared, *, method_identity, lineages, expected_schema=None,
         requires_selector_last_bed=None):
    folder = Path(folder)
    manifest = json.loads((folder/'checkpoint.json').read_text())
    schema = manifest.get('schema')
    _require(type(schema) is int and schema in (1, 2, 3, 4) and
        manifest == dict(schema=schema, kind='coupled_bank_amr',
        payload_sha256=_sha(folder/'state.npz'), catalog_identity=prepared.identity,
        method_identity=method_identity), 'Checkpoint receipt or payload changed')
    with np.load(folder/'state.npz', allow_pickle=False) as archive:
        _require(len(archive.files) == len(set(archive.files)), 'Duplicate bank-AMR checkpoint member')
        fields = {key:archive[key] for key in archive.files}
    _require(np.asarray(fields.get('schema')).shape == () and
             fields['schema'] == schema, 'Checkpoint receipt/payload schema differs')
    top, events = validate(fields, prepared, method_identity=method_identity, lineages=lineages,
                           expected_schema=expected_schema,
                           requires_selector_last_bed=requires_selector_last_bed)
    return fields, top, events


def restore(folder, prepared, *, source_folder, initial_tree_state, method_identity, lineages,
            expected_schema=None, requires_selector_last_bed=None):
    fields, top, events = read(folder, prepared, method_identity=method_identity, lineages=lineages,
                              expected_schema=expected_schema,
                              requires_selector_last_bed=requires_selector_last_bed)
    # Device construction starts only after the complete host check above.
    from .amr_catalog_bank import CatalogBankRoute
    index = next(i for i,item in enumerate(prepared.topologies) if item is top)
    evolving = int(fields['schema']) in (2, 3)
    route = CatalogBankRoute(prepared, source_folder=source_folder,
        initial_tree_state=initial_tree_state, lineages=lineages, initial_topology=index,
        evolving_toe_variant=evolving)
    cp, model = route.cp, route.model
    # Restore the pending temporal displacement reference before any subsequent
    # selection. Rebuilding it from the accepted bed would erase pre-regrid dz.
    route.selector_last_bed = (cp.asarray(fields['selector_last_bed'])
                               if 'selector_last_bed' in fields else None)
    model.install(cp.asarray(fields['state']))
    for name,value in model.solver.bank_history.items():
        cp.copyto(value, cp.asarray(fields['history_'+name]))
    for name,value in model.solver.ledger.items():
        cp.copyto(value, cp.asarray(fields['ledger_'+name]))
    if evolving:
        mapper = model.solver.evolving_toe
        _require(mapper is not None, 'Evolving checkpoint needs the evolving bank solver')
        mapper.restore({key.removeprefix('toe_map_'):value for key,value in fields.items()
                        if key.startswith('toe_map_')}, model.solver.bed(),
                       model.state['q'][0], model.state['layers'])
        cp.copyto(model.solver.toe_face_signed, cp.asarray(fields['toe_signed']))
        cp.copyto(model.solver.toe_face_gross, cp.asarray(fields['toe_gross']))
        route.face_epochs = []
        for i in range(int(fields['face_epoch_count'])):
            prefix = f'face_epoch_{i}_'
            epoch = dict(operator_identity=str(fields[prefix+'operator_identity'].item()),
                         node_ids=fields[prefix+'node_ids'].copy())
            epoch.update({key:cp.asarray(fields[prefix+key])
                          for key in ('old_face','fraction','signed','gross')})
            route.face_epochs.append(epoch)
    elif int(fields['schema']) != 4:
        _, faces = route.lineages[top.name]
        model.solver.toe_face_signed[:,cp.asarray(faces)] = cp.asarray(fields['toe_signed'])
        model.solver.toe_face_gross[:,cp.asarray(faces)] = cp.asarray(fields['toe_gross'])
    model.accepted_steps = int(fields['accepted_steps'])
    model.time_s = float(fields['time_s'])
    model.maximum_cfl = float(fields['maximum_cfl'])
    model.solver.euler_calls = int(fields['euler_calls'])
    if int(fields['schema']) == 4:
        model.solver.distributed_stage_index = int(fields['distributed_stage_index'])
        model.solver.distributed_bank.collapse_waves = int(fields['distributed_collapse_waves'])
    model.sync_accepted_clock()
    route.tree_state = cp.asarray(fields['tree_state'])
    route.events = events
    return route, fields['delay_state'].copy()
