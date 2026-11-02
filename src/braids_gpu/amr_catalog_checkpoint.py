"""Accepted-state checkpoint guards for the experimental AMR catalog route.

Checkpoint admission is deliberately strict and entirely host-side before
constructing or mutating CUDA state.  This covers the current deterministic
one-way catalog transition; it does not authorize arbitrary AMR schedules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .amr_prepared import PreparedAMR
from .atomic_output import save_arrays_atomic
from .runner import fields_digest


def _schema(topology):
    data = topology.data
    n = len(topology.node_ids)
    grains = int(data['nfrac'])
    layers = int(data['nbed_layers'])
    boundaries = len(data['mode'])
    return dict(q=(3, n), layers=(layers, grains, n),
        suspended=(grains, n), secondary=(n,),
        ledger_water=(boundaries,), ledger_bed=(grains, boundaries),
        ledger_secondary=(boundaries,), ledger_source=(),
        ledger_suspended=(grains, boundaries), ledger_exchange=(grains, 2))


def _state_fields(solver):
    result = {key: solver.cp.asnumpy(solver.state[key]).copy()
              for key in ('q', 'layers', 'suspended', 'secondary')}
    result.update({'ledger_'+key: solver.cp.asnumpy(value).copy()
                   for key, value in solver.ledger.items()})
    return result


def write_catalog_checkpoint(path, prepared: PreparedAMR, topology_index: int,
                             solver, *, step: int, dt: float,
                             transition_step: int, source_identity: str,
                             method_identity: str):
    """Write only a committed state; the caller owns the step transaction."""
    if (type(topology_index) is not int or
            not 0 <= topology_index < len(prepared.topologies) or
            type(step) is not int or step < 0 or
            not np.isfinite(dt) or dt <= 0 or
            type(transition_step) is not int or transition_step < 1 or
            topology_index != int(step >= transition_step)):
        raise ValueError('AMR checkpoint topology/clock is not committed')
    topology = prepared.topologies[topology_index]
    if topology_index >= 2:
        raise ValueError('Current AMR checkpoint schema supports one transition')
    fields = _state_fields(solver)
    expected = _schema(topology)
    if set(fields) != set(expected) or any(fields[key].shape != shape or
            fields[key].dtype != np.float64 or not np.isfinite(fields[key]).all()
            for key, shape in expected.items()):
        raise ValueError('Invalid AMR solver accepted state/ledger schema')
    fields.update(node_ids=topology.node_ids.copy(),
        input_identity=np.asarray(prepared.identity),
        operator_identity=np.asarray(topology.source_identity),
        source_identity=np.asarray(source_identity),
        method_identity=np.asarray(method_identity),
        topology_index=np.int64(topology_index),
        accepted_steps=np.int64(step),
        time_s=np.float64(step*dt), dt_s=np.float64(dt),
        transition_step=np.int64(transition_step),
        euler_calls=np.int64(solver.euler_calls))
    fields['checkpoint_payload_sha256'] = np.asarray(fields_digest(fields))
    save_arrays_atomic(Path(path), **fields)
    return fields['checkpoint_payload_sha256'].item()


def read_catalog_checkpoint(path, prepared: PreparedAMR, *, dt: float,
                            transition_step: int, source_identity: str,
                            method_identity: str, max_steps: int):
    """Reject tampering, stale code, wrong topology or wrong clock before CUDA."""
    with np.load(Path(path), allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    digest = fields.pop('checkpoint_payload_sha256', None)
    if (digest is None or digest.shape != () or digest.dtype.kind not in 'US' or
            digest.item() != fields_digest(fields)):
        raise ValueError('AMR checkpoint accepted payload digest differs')
    for key, expected in (('input_identity', prepared.identity),
                          ('source_identity', source_identity),
                          ('method_identity', method_identity)):
        value = fields.get(key)
        if value is None or value.shape != () or value.dtype.kind not in 'US' or \
                value.item() != expected:
            raise ValueError('AMR checkpoint '+key+' differs')
    step = np.asarray(fields.get('accepted_steps'))
    mesh = np.asarray(fields.get('topology_index'))
    time = np.asarray(fields.get('time_s'))
    saved_dt = np.asarray(fields.get('dt_s'))
    transition = np.asarray(fields.get('transition_step'))
    calls = np.asarray(fields.get('euler_calls'))
    if (not np.isfinite(dt) or dt <= 0 or type(transition_step) is not int or
            transition_step < 1 or type(max_steps) is not int or
            max_steps <= transition_step or
            step.shape != () or step.dtype.kind not in 'iu' or
            not 0 <= int(step) <= max_steps or
            mesh.shape != () or mesh.dtype.kind not in 'iu' or
            int(mesh) != int(int(step) >= transition_step) or int(mesh) >= 2 or
            time.shape != () or time.dtype != np.float64 or
            not np.isfinite(time) or abs(float(time)-int(step)*dt) > 1e-12 or
            saved_dt.shape != () or saved_dt.dtype != np.float64 or
            float(saved_dt) != dt or transition.shape != () or
            transition.dtype.kind not in 'iu' or int(transition) != transition_step or
            calls.shape != () or calls.dtype.kind not in 'iu' or int(calls) < 0):
        raise ValueError('AMR checkpoint clock/topology differs')
    topology = prepared.topologies[int(mesh)]
    ids = np.asarray(fields.get('node_ids'))
    operator = np.asarray(fields.get('operator_identity'))
    if (ids.dtype.kind not in 'iu' or not np.array_equal(ids, topology.node_ids) or
            operator.shape != () or operator.dtype.kind not in 'US' or
            operator.item() != topology.source_identity):
        raise ValueError('AMR checkpoint selected topology/operator differs')
    schema = _schema(topology)
    expected = set(schema) | {'node_ids', 'input_identity', 'operator_identity',
        'source_identity', 'method_identity', 'topology_index', 'accepted_steps',
        'time_s', 'dt_s', 'transition_step', 'euler_calls'}
    if set(fields) != expected:
        raise ValueError('AMR checkpoint contains missing/unknown accepted fields')
    for key, shape in schema.items():
        value = np.asarray(fields[key])
        if (value.shape != shape or value.dtype != np.float64 or
                not value.flags.c_contiguous or not np.isfinite(value).all()):
            raise ValueError('AMR checkpoint accepted field invalid: '+key)
    if (np.any(fields['q'][0] < -1e-12) or
            np.any(fields['layers'] < -1e-12) or
            np.any(fields['suspended'] < -1e-12)):
        raise ValueError('AMR checkpoint has negative water/sediment inventory')
    return fields


def restore_catalog_solver(solver, fields):
    """Install a fully prevalidated accepted state in a newly built solver."""
    cp = solver.cp
    for key in ('q', 'layers', 'suspended', 'secondary'):
        cp.copyto(solver.state[key], cp.asarray(fields[key]))
    for key in solver.ledger:
        cp.copyto(solver.ledger[key], cp.asarray(fields['ledger_'+key]))
    solver.euler_calls = int(fields['euler_calls'])


def write_mixed_checkpoint(path, model, *, source_identity: str,
                           method_identity: str):
    """Atomically publish one committed mixed-level GPU reflux state."""
    if model.reflux.pending or model.accepted_steps < 0:
        raise ValueError('Cannot checkpoint an unresolved AMR reflux proposal')
    fields = model.persistent()
    fields.update(input_identity=np.asarray(model.prepared.identity),
        operator_identity=np.asarray(model.topology.source_identity),
        source_identity=np.asarray(source_identity),
        method_identity=np.asarray(method_identity))
    fields['checkpoint_payload_sha256'] = np.asarray(fields_digest(fields))
    save_arrays_atomic(path, **fields)
    return fields['checkpoint_payload_sha256'].item()


def read_mixed_checkpoint(path, prepared: PreparedAMR, topology, *,
                          source_identity: str, method_identity: str,
                          outer_dt: float, max_steps: int,
                          extra_fields=(), wetfront_schedule=False):
    """Authenticate every mixed-level field before constructing a solver."""
    if not any(item is topology for item in prepared.topologies):
        raise ValueError('AMR checkpoint topology is not in the catalog')
    with np.load(Path(path), allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    digest = fields.pop('checkpoint_payload_sha256', None)
    if (digest is None or digest.shape != () or digest.dtype.kind not in 'US' or
            digest.item() != fields_digest(fields)):
        raise ValueError('Mixed AMR checkpoint payload digest differs')
    for key, expected in (('input_identity', prepared.identity),
                          ('operator_identity', topology.source_identity),
                          ('source_identity', source_identity),
                          ('method_identity', method_identity)):
        value = np.asarray(fields.get(key))
        if value.shape != () or value.dtype.kind not in 'US' or \
                value.item() != expected:
            raise ValueError('Mixed AMR checkpoint '+key+' differs')
    ids = np.asarray(fields.get('node_ids'))
    levels = prepared.tree['node_level'][topology.node_ids]
    owner = topology.data['owner']
    neighbour = topology.data['neighbour']
    faces = int(np.count_nonzero((neighbour >= 0) &
        (levels[owner] != levels[np.maximum(neighbour, 0)])))
    n = len(topology.node_ids)
    count = np.asarray(fields.get('interface_count'))
    if (ids.dtype.kind not in 'iu' or
            not np.array_equal(ids, topology.node_ids) or
            count.shape != () or count.dtype.kind not in 'iu' or
            int(count) != faces):
        raise ValueError('Mixed AMR checkpoint topology/interface differs')
    steps = np.asarray(fields.get('accepted_steps'))
    clock = np.asarray(fields.get('time_s'))
    level_steps = np.asarray(fields.get('level_steps'))
    calls = np.asarray(fields.get('euler_calls'))
    sync = np.asarray(fields.get('sync_steps')) if wetfront_schedule else None
    mixed = np.asarray(fields.get('mixed_steps')) if wetfront_schedule else None
    counts_present = (steps.shape == () and steps.dtype.kind in 'iu' and
        level_steps.shape == (3,) and level_steps.dtype.kind in 'iu' and
        calls.shape == () and calls.dtype.kind in 'iu')
    if wetfront_schedule and counts_present:
        schedule_valid = (sync.shape == () and sync.dtype.kind in 'iu' and
            mixed.shape == () and mixed.dtype.kind in 'iu' and
            int(sync) >= 0 and int(mixed) >= 0 and
            int(sync)+int(mixed) == int(steps) and
            np.array_equal(level_steps, [0,2*int(mixed),4*int(mixed)]) and
            int(calls) == 3*int(sync)+18*int(mixed))
    elif counts_present:
        schedule_valid = (np.array_equal(level_steps,
            [0,2*int(steps),4*int(steps)]) and
            int(calls) == 18*int(steps))
    else:
        schedule_valid = False
    if (not np.isfinite(outer_dt) or outer_dt <= 0 or
            type(max_steps) is not int or max_steps < 1 or
            steps.shape != () or steps.dtype.kind not in 'iu' or
            not 0 <= int(steps) <= max_steps or
            clock.shape != () or clock.dtype != np.float64 or
            abs(float(clock)-int(steps)*outer_dt) > 1e-12 or
            level_steps.shape != (3,) or level_steps.dtype.kind not in 'iu' or
            not schedule_valid):
        raise ValueError('Mixed AMR checkpoint clock/level schedule differs')
    boundaries = len(topology.data['mode'])
    float_shapes = dict(state=(n, 10), reflux_accepted=(n, 10),
        reflux_register=(faces, 2, 10), exchange_ledger=(2, 2),
        maximum_cfl=(), source_integral=(),
        **{'peak_'+name: (2,) for name in (
            'bed_flux', 'suspended_flux', 'coarse_bed', 'fine_bed',
            'bed_reflux', 'coarse_suspended', 'fine_suspended',
            'suspended_reflux')},
        **{'hydro_peak_'+name: (3,) for name in ('coarse','fine','reflux')},
        solver_ledger_water=(boundaries,),
        solver_ledger_bed=(2,boundaries),
        solver_ledger_secondary=(boundaries,), solver_ledger_source=(),
        solver_ledger_suspended=(2,boundaries),
        solver_ledger_exchange=(2,2))
    expected = set(float_shapes) | {'reflux_pending','accepted_steps',
        'level_steps','euler_calls','secondary_peaks_json','node_ids',
        'interface_count','time_s','input_identity','operator_identity',
        'source_identity','method_identity'} | set(extra_fields)
    if set(fields) != expected:
        raise ValueError('Mixed AMR checkpoint accepted field schema differs')
    for key, shape in float_shapes.items():
        value = np.asarray(fields[key])
        if (value.shape != shape or value.dtype != np.float64 or
                not value.flags.c_contiguous or not np.isfinite(value).all()):
            raise ValueError('Mixed AMR checkpoint invalid accepted field: '+key)
    pending = np.asarray(fields['reflux_pending'])
    if pending.shape != () or pending.dtype != np.bool_ or bool(pending):
        raise ValueError('Mixed AMR checkpoint has pending reflux')
    exchange_balanced = (np.max(np.abs(fields['exchange_ledger'][:,0] +
        fields['exchange_ledger'][:,1])) <= 1e-12 if wetfront_schedule else
        np.array_equal(fields['exchange_ledger'][:,0],
                       -fields['exchange_ledger'][:,1]))
    if (not np.array_equal(fields['state'], fields['reflux_accepted']) or
            np.any(fields['state'][:,0] < -1e-12) or
            np.any(fields['state'][:,3:9] < -1e-12) or
            np.any(fields['maximum_cfl'] < 0) or
            np.any(fields['maximum_cfl'] > float(topology.data['cfl_target'])+1e-12) or
            not exchange_balanced):
        raise ValueError('Mixed AMR checkpoint conservation/positivity differs')
    register = np.abs(fields['reflux_register'])
    if (np.any(fields['hydro_peak_fine'] < 0) or
            any(np.any(fields['peak_'+name] < 0) for name in (
                'bed_flux','suspended_flux','coarse_bed','fine_bed',
                'bed_reflux','coarse_suspended','fine_suspended',
                'suspended_reflux')) or
            np.any(register[:,:,7:9] != 0) or
            np.any(np.max(register[:,:,:3], axis=(0,1)) >
                   fields['hydro_peak_fine']+1e-12) or
            np.any(np.max(register[:,:,3:5], axis=(0,1)) >
                   fields['peak_fine_suspended']+1e-12) or
            np.any(np.max(register[:,:,5:7], axis=(0,1)) >
                   fields['peak_fine_bed']+1e-12)):
        raise ValueError('Mixed AMR checkpoint reflux register exceeds accepted flux history')
    encoded = np.asarray(fields['secondary_peaks_json'])
    if encoded.shape != () or encoded.dtype.kind not in 'US':
        raise ValueError('Mixed AMR checkpoint secondary peaks are invalid')
    import json
    try:
        peaks = json.loads(str(encoded.item()))
    except (TypeError, ValueError) as exc:
        raise ValueError('Mixed AMR checkpoint secondary peaks are invalid') from exc
    if set(peaks) != {'face_flux','coarse_integral','fine_integral',
                      'reflux_delta'} or any(not isinstance(value, (int,float)) or
                      not np.isfinite(value) or value < 0 for value in peaks.values()):
        raise ValueError('Mixed AMR checkpoint secondary peaks are invalid')
    if np.max(register[:,:,9]) > peaks['fine_integral']+1e-12:
        raise ValueError('Mixed AMR checkpoint secondary reflux exceeds accepted flux history')
    return fields
