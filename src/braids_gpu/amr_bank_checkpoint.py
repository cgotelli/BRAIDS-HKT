"""Strict schema-2 checkpoint for the bounded synchronous bank-AMR route."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .amr_bank_wetfront import HISTORY_NAMES, BankWetFrontRoute
from .atomic_output import save_arrays_atomic
from .runner import fields_digest


FLOAT_HISTORY = tuple('history_'+name for name in HISTORY_NAMES)
INT_HISTORY = 'history_collapse_count'
ROUTE = 'bank_synchronous_wetfront_schema2'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _float(fields, name, shape):
    value = np.asarray(fields[name])
    _require(value.shape == shape and value.dtype == np.float64 and
        value.flags.c_contiguous and np.isfinite(value).all(),
        'Bank-AMR checkpoint invalid floating field: '+name)
    return value


def _integer(fields, name, shape, *, dtype=np.int64):
    value = np.asarray(fields[name])
    _require(value.shape == shape and value.dtype == dtype and
        value.flags.c_contiguous,
        'Bank-AMR checkpoint invalid integer field: '+name)
    return value


def _string(fields, name, expected=None):
    value = np.asarray(fields[name])
    _require(value.shape == () and value.dtype.kind in 'US' and
        (expected is None or value.item() == expected),
        'Bank-AMR checkpoint identity/JSON differs: '+name)
    return str(value.item())


def write_bank_checkpoint(path, route: BankWetFrontRoute, *,
                          source_identity: str, method_identity: str):
    """Atomically publish only a fully accepted route state."""
    _require(isinstance(route, BankWetFrontRoute) and
        route.model.accepted_steps >= 0 and
        len(route.events) == len(route.closed_epochs),
        'Bank-AMR route is not committed')
    fields = route.persistent()
    fields.update(checkpoint_schema=np.int64(2),
        route_kind=np.asarray(ROUTE),
        input_identity=np.asarray(route.prepared.identity),
        operator_identity=np.asarray(route.model.topology.source_identity),
        source_identity=np.asarray(source_identity),
        method_identity=np.asarray(method_identity))
    fields['checkpoint_payload_sha256'] = np.asarray(fields_digest(fields))
    save_arrays_atomic(path, **fields)
    return str(fields['checkpoint_payload_sha256'].item())


def read_bank_checkpoint(path, prepared, *, source_identity: str,
                         method_identity: str, max_steps=40):
    """Authenticate complete payload and physical semantics before CUDA use."""
    _require(prepared.schema == 2 and len(prepared.topologies) == 2 and
        type(max_steps) is int and 1 <= max_steps <= 40,
        'Bank-AMR checkpoint prepared scope differs')
    with np.load(Path(path), allow_pickle=False) as archive:
        fields = {key:archive[key] for key in archive.files}
    digest = fields.pop('checkpoint_payload_sha256', None)
    _require(digest is not None and np.asarray(digest).shape == () and
        np.asarray(digest).dtype.kind in 'US' and
        digest.item() == fields_digest(fields),
        'Bank-AMR checkpoint payload digest differs')
    _integer(fields, 'checkpoint_schema', (), dtype=np.int64)
    _require(int(fields['checkpoint_schema']) == 2,
             'Bank-AMR checkpoint schema differs')
    _string(fields, 'route_kind', ROUTE)
    _string(fields, 'input_identity', prepared.identity)
    _string(fields, 'source_identity', source_identity)
    _string(fields, 'method_identity', method_identity)
    mesh = int(_integer(fields, 'regrid_mesh_index', ()))
    epochs = int(_integer(fields, 'toe_epoch_count', ()))
    _require(mesh in (0,1) and epochs in (0,1,2) and mesh == epochs % 2,
             'Bank-AMR checkpoint topology/epoch sequence differs')
    current = prepared.topologies[mesh]
    _string(fields, 'operator_identity', current.source_identity)
    n = len(current.node_ids)
    nf = int(current.data['nf'])
    expected = {
        'checkpoint_schema','route_kind','input_identity','operator_identity',
        'source_identity','method_identity','state','node_ids','accepted_steps',
        'route_steps','euler_calls','maximum_cfl','time_s','toe_signed',
        'toe_gross','ledger_water','ledger_bed','ledger_secondary',
        'ledger_source',*FLOAT_HISTORY,INT_HISTORY,
        'regrid_mesh_index','regrid_last_step','regrid_events_json',
        'regrid_policy_json','regrid_cached_children','toe_epoch_count',
    }
    for i in range(epochs):
        expected.update({'toe_epoch_'+str(i)+'_'+suffix for suffix in
                         ('topology','signed','gross')})
    _require(set(fields) == expected,
             'Bank-AMR checkpoint accepted field schema differs')
    ids = _integer(fields, 'node_ids', (n,))
    _require(np.array_equal(ids, current.node_ids),
             'Bank-AMR checkpoint selected global nodes differ')
    state = _float(fields, 'state', (n,10))
    _require(np.all(state[:,3:5] == 0) and
        np.all(state[:,0] >= -1e-12) and
        np.all(state[:,5:9] >= -1e-12),
        'Bank-AMR checkpoint unsuspended state/positivity differs')
    steps = int(_integer(fields, 'accepted_steps', ()))
    count = int(_integer(fields, 'route_steps', ()))
    calls = int(_integer(fields, 'euler_calls', ()))
    last = int(_integer(fields, 'regrid_last_step', ()))
    clock = float(_float(fields, 'time_s', ()))
    maximum_cfl = float(_float(fields, 'maximum_cfl', ()))
    _require(0 <= steps <= max_steps and count == steps and
        calls == 3*steps and abs(clock-steps*.002) <= 1e-12 and
        0 <= maximum_cfl <= float(current.data['cfl_target'])+1e-12,
        'Bank-AMR checkpoint accepted step/clock/CFL differs')
    _float(fields, 'ledger_water', (1,))
    _float(fields, 'ledger_bed', (2,1))
    _float(fields, 'ledger_secondary', (1,))
    _float(fields, 'ledger_source', ())
    for name in FLOAT_HISTORY:
        _require(np.all(_float(fields,name,(n,)) >= 0),
                 'Bank-AMR checkpoint negative history: '+name)
    _require(np.all(_integer(fields,INT_HISTORY,(n,),dtype=np.int32) >= 0),
             'Bank-AMR checkpoint negative collapse count')

    def toe_arrays(prefix, topology):
        nf_local = int(topology.data['nf'])
        signed = _float(fields,prefix+'signed',(2,nf_local))
        gross = _float(fields,prefix+'gross',(2,nf_local))
        toe = np.asarray(topology.data['is_toe_face'],dtype=bool)
        _require(np.all(gross >= 0) and
            np.all(gross+1e-12 >= np.abs(signed)) and
            np.all(signed[:,~toe] == 0) and
            np.all(gross[:,~toe] == 0),
            'Bank-AMR checkpoint toe-face activity/mapping differs')
        return signed,gross

    toe_arrays('toe_',current)
    for i in range(epochs):
        prior = prepared.topologies[i%2]
        prefix = 'toe_epoch_'+str(i)+'_'
        _string(fields,prefix+'topology',prior.source_identity)
        toe_arrays(prefix,prior)
    policy = dict(min_dwell=2,coarsen_spread_m=.080,refine_jump_m=.070)
    _string(fields,'regrid_policy_json',json.dumps(policy,sort_keys=True,
                                                 separators=(',',':')))
    cache = _float(fields,'regrid_cached_children',
                   (1,4,10) if mesh == 1 else (0,4,10))
    _require(np.all(cache[:,:,3:5] == 0) and
        np.all(cache[:,:,0] >= -1e-12) and
        np.all(cache[:,:,5:9] >= -1e-12),
        'Bank-AMR checkpoint child cache is not physical')
    try:
        events = json.loads(_string(fields,'regrid_events_json'))
    except (TypeError,ValueError) as exc:
        raise ValueError('Bank-AMR checkpoint events JSON differs') from exc
    _require(isinstance(events,list) and len(events) == epochs and
        last == (events[-1]['step'] if events else 0),
        'Bank-AMR checkpoint regrid event count/clock differs')
    for i,event in enumerate(events):
        a = prepared.topologies[i%2]
        b = prepared.topologies[1-i%2]
        _require(isinstance(event,dict) and
            event.get('action') == ('coarsen' if i%2 == 0 else 'refine') and
            event.get('step') == 2*(i+1) and
            event.get('time_s') == event['step']*.002 and
            event.get('parent_id') == 56 and
            event.get('old_leaves') == len(a.node_ids) and
            event.get('new_leaves') == len(b.node_ids) and
            event.get('old_operator_identity') == a.source_identity and
            event.get('new_operator_identity') == b.source_identity and
            event.get('face_epoch_sealed') is True and
            isinstance(event.get('indicator_m'),(int,float)) and
            np.isfinite(event['indicator_m']) and
            (event['indicator_m'] < .080 if i%2 == 0 else
             event['indicator_m'] > .070),
            'Bank-AMR checkpoint accepted event semantics differ')

    initial = prepared.topologies[0]
    first = np.asarray(initial.data['state']).T
    ai = np.asarray(initial.data['area'])
    ac = np.asarray(current.data['area'])
    water_error = (np.dot(ac,state[:,0])-np.dot(ai,first[:,0])-
                   np.sum(fields['ledger_water']))
    class_error = np.asarray([np.dot(ac,state[:,3+g]+state[:,5+g]+state[:,7+g])-
        np.dot(ai,first[:,3+g]+first[:,5+g]+first[:,7+g])-
        np.sum(fields['ledger_bed'][g]) for g in range(2)])
    secondary_error = (np.dot(ac,state[:,9])-np.dot(ai,first[:,9])-
        np.sum(fields['ledger_secondary'])-float(fields['ledger_source']))
    _require(abs(water_error) <= 1e-8 and
        np.max(np.abs(class_error)) <= 1e-7 and
        abs(secondary_error) <= 1e-7,
        'Bank-AMR checkpoint cumulative inventory/ledger differs')
    return fields


def restore_bank_route(path, prepared, parent, children, *, source_folder,
                       source_identity, method_identity, max_steps=40):
    """No CUDA allocation occurs until the entire host checkpoint is valid."""
    fields = read_bank_checkpoint(path,prepared,
        source_identity=source_identity,method_identity=method_identity,
        max_steps=max_steps)
    route = BankWetFrontRoute(prepared,parent,children,
        source_folder=source_folder,
        mesh_index=int(fields['regrid_mesh_index']))
    route.restore(fields)
    return route
