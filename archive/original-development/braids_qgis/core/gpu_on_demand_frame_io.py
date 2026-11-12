"""Host-only reader for self-contained schema-3 on-demand GPU AMR runs.

Each run carries complete, content-addressed single-topology schema-2
operator packages.  No native initializer, CUDA library, or original input
folder is needed to view committed frames after the run has been moved.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np
from braids_core.windows_paths import filesystem_path

from . import gpu_amr_prepared_io as prepared_io
from .gpu_amr_frame_io import (_FRAME, _frame as _legacy_frame, _method_identity,
    _fail, _REQUIRED, _OPTIONAL, _HISTORY_FLOAT, _HISTORY_COUNT)
from .gpu_frame_io import (
    _digest, _geometry, _identity, _json, _number, _source_identity,
    _archive, _array, _scalar, _string_scalar, _toe_map_frame, _toe_map_schema,
)
from .gpu_prepared_io import _read_records, read_gpu_prepared


_STATUSES = {"initializing", "running", "completed", "step_limit",
             "cancelled", "failed"}
_ACTIVE_STATUSES = {"initializing", "running"}
_PENDING_OPERATOR = re.compile(r"(?:[0-9a-f]{64}|\.mesh-[a-z0-9_]+)\Z")
_RECURSIVE_SCHEMA = 'recursive_bank_amr_v1'
_RECURSIVE_FIELDS = frozenset(('recursive_level_steps', 'recursive_epoch_start_steps',
    'recursive_epoch_start_time', 'recursive_epoch_start_euler_calls', 'recursive_register',
    'recursive_repair_totals', 'recursive_preparation_impulse'))
RECURSIVE_BANK_BEDFORM_FRAME_SCHEMA = 7
RECURSIVE_PHYSICAL_TOE_BEDFORM_FRAME_SCHEMA = 8
RECURSIVE_BANK_BEDFORM_METHOD_KEY = 'recursive_bank_bedform_avalanche'
RECURSIVE_BANK_BEDFORM_AVALANCHE_METHOD = dict(
    policy='recursive_bedform_mode3_parent_avalanche_v1',
    bedform_order='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    bank_transaction='distributed_mode3_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    calls_per_macro={'bank': 1, 'avalanche': 1})
RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_METHOD = dict(
    policy='recursive_bedform_mode2_physical_toe_parent_avalanche_v1',
    bedform_order='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    accepted_flux_owner='completed_root_exner_register_v1',
    bank_transaction=
        'physical_toe_supply_undercut_collapse_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    mapping_transaction='reconcile_then_evolving_toe_rebuild_before_commit_v1',
    calls_per_macro={'bank': 1, 'avalanche': 1})
RECURSIVE_BANK_BEDFORM_METHODS = {
    2: RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_METHOD,
    3: RECURSIVE_BANK_BEDFORM_AVALANCHE_METHOD,
}


def _base_input_digest(originals: dict) -> str:
    names=[name for name in originals if name=='case.data' or
           (name.startswith('.br') and name.endswith('.dat'))]
    if len(names)!=1:
        _fail('On-demand original-input deck inventory differs')
    return _identity(originals[names[0]],'original input deck')


def _adapter_identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
BEDFORM_FIELDS = frozenset(('bedform_effective_manning',
    'bedform_dune_height', 'bedform_dune_length', 'bedform_total_ks',
    'bedform_iterations'))
_READABLE_BEDFORM_BANK_RUNTIMES = {
    # v2 produced authenticated zero-step packages through the wrapper, but
    # its raw executable did not itself enforce export-only invocation.  Keep
    # those completed packages readable without admitting v2 for new runs.
    'live-selected-bank-mode3-bedform-avalanche-amr-v2':
        'exact_recursive_mode3_bedform_parent_avalanche_v2',
    'live-selected-bank-mode3-bedform-avalanche-amr-v3':
        'exact_recursive_mode3_bedform_parent_avalanche_v3',
    'live-selected-bank-mode2-bedform-avalanche-amr-v1':
        'exact_recursive_mode2_physical_toe_bedform_parent_avalanche_v1',
}


class _SnapshotAdvanced(Exception):
    """The commit pointer changed while inspecting a live-only snapshot."""


def _typed_equal(actual, expected) -> bool:
    """Compare a small JSON-like contract without bool/int aliasing."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (set(actual) == set(expected) and
                all(_typed_equal(actual[key], value)
                    for key, value in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (len(actual) == len(expected) and
                all(_typed_equal(left, right)
                    for left, right in zip(actual, expected)))
    return actual == expected


def _recursive_bank_bedform_method(method: dict, bank_mode: int | None = None) -> bool:
    """Validate the optional exact mode-specific bedform/bank/avalanche method."""
    if RECURSIVE_BANK_BEDFORM_METHOD_KEY not in method:
        return False
    record = method[RECURSIVE_BANK_BEDFORM_METHOD_KEY]
    if bank_mode is None:
        matches = [candidate for candidate in RECURSIVE_BANK_BEDFORM_METHODS.values()
                   if _typed_equal(record, candidate)]
    else:
        candidate = RECURSIVE_BANK_BEDFORM_METHODS.get(bank_mode)
        matches = [] if candidate is None else [candidate] if _typed_equal(
            record, candidate) else []
    if len(matches) != 1:
        _fail('Recursive bank bedform/avalanche method contract differs')
    return True


def _combined_bedform_frame(manifest: dict, initial: dict, *,
                            recursive: bool, distributed: bool) -> bool:
    """Bind schemas 7/8 to exact prepared compositions, not labels alone."""
    flag = np.asarray(initial.get('bedform_roughness', 0))
    if (flag.shape != () or flag.dtype.kind not in 'iu' or
            int(flag) not in (0, 1)):
        _fail('Recursive bank bedform switch differs')
    enabled = bool(int(flag))
    bank_mode = int(initial.get('bank_toe_erosion_mode', -1))
    declared = _recursive_bank_bedform_method(manifest['method'], bank_mode)
    if enabled != declared:
        _fail('Recursive bank bedform operator/method pairing differs')
    if not enabled:
        return False

    integers = dict(bank_toe_erosion_mode=bank_mode, bank_failure=1,
        general_avalanche=1, bedlaw=2, mpm_friction_mode=1,
        bed_slope=0, hiding_exposure=0, exchange_mode=0,
        secondary_flow_model=0, bedload_adaptation=0,
        nsubstrate_layers=1)
    integer_scope = all(
        np.asarray(initial.get(name)).shape == () and
        np.asarray(initial[name]).dtype.kind in 'iu' and
        int(initial[name]) == expected
        for name, expected in integers.items())
    mode = np.asarray(initial.get('mode'))
    viscosity = np.asarray(initial.get('horizontal_viscosity'))
    morfac = np.asarray(initial.get('morphological_factor'))
    if (bank_mode not in (2, 3) or distributed != (bank_mode == 3) or
            not recursive or
            manifest['method'].get('bank_mode') != bank_mode or
            manifest['method'].get('evolving_banks') is not True or
            manifest['method'].get('recursive_bank_contract') !=
                'shared_recursive_bank_modes_v2' or
            not integer_scope or
            mode.shape != (1,) or mode.dtype.kind not in 'iu' or
            int(mode[0]) != 2 or int(initial.get('nb', -1)) != 1 or
            viscosity.shape != () or viscosity.dtype.kind not in 'fiu' or
            not np.isfinite(viscosity) or float(viscosity) != 0. or
            morfac.shape != () or morfac.dtype.kind not in 'fiu' or
            not np.isfinite(morfac) or float(morfac) != 1.):
        _fail('Recursive bank bedform/avalanche operator scope differs')
    return True


def _evolving_method(manifest: dict) -> bool:
    method = manifest["method"]
    schema = method.get("checkpoint_schema")
    if schema == _RECURSIVE_SCHEMA:
        bank_mode=method.get('bank_mode')
        evolving=method.get('evolving_banks',False)
        route_contract=method.get('recursive_bank_contract')
        bedform_composition=_recursive_bank_bedform_method(method, bank_mode)
        if (method.get('subcycling') is not True or
                type(bank_mode) is not int or bank_mode not in (2,3) or
                type(evolving) is not bool or (bank_mode == 3 and evolving is not True) or
                (bank_mode == 2 and route_contract != 'shared_recursive_bank_modes_v2') or
                (bank_mode == 3 and route_contract not in
                    (None,'shared_recursive_bank_modes_v2')) or
                (bedform_composition and
                 (bank_mode not in (2, 3) or evolving is not True or
                  route_contract != 'shared_recursive_bank_modes_v2')) or
                method.get('level_stepping') != 'recursive_ssprk33_fine_replaced_register_v1' or
                ('selector_reference' in method and method['selector_reference'] != 'accepted_regrid_leaf_bed_v1')):
            _fail('Recursive on-demand method contract differs')
        forcing = method.get('boundary_forcing')
        if (not isinstance(forcing, dict) or set(forcing) != {'schema', 'event_identity', 'stage_fractions', 'clock'} or
                type(forcing['schema']) is not int or forcing['schema'] != 1 or
                forcing['stage_fractions'] != [0., 1., .5] or forcing['clock'] != 'forcing_knot_clipped'):
            _fail('Recursive on-demand boundary clock differs')
        _identity(forcing['event_identity'], 'recursive boundary clock')
        return True
    if RECURSIVE_BANK_BEDFORM_METHOD_KEY in method:
        _fail('Synchronous on-demand method carries recursive bank bedform physics')
    if method.get('subcycling', False) is not False or 'level_stepping' in method:
        _fail('Synchronous on-demand method cannot carry recursive stepping')
    if type(schema) is not int or schema not in (1, 2, 3, 4):
        _fail("On-demand checkpoint schema is unsupported")
    if schema == 4:
        if (type(method.get('bank_mode')) is not int or method['bank_mode'] != 3 or
                method.get('evolving_banks') is not True or
                ('selector_reference' in method and method['selector_reference'] != 'accepted_regrid_leaf_bed_v1')):
            _fail('On-demand distributed-bank method contract differs')
        return True
    if 'bank_mode' in method:
        _fail('Legacy on-demand method cannot carry distributed-bank mode')
    if schema == 3:
        if method.get("selector_reference") != "accepted_regrid_leaf_bed_v1":
            _fail("On-demand schema-3 checkpoint needs the temporal bed reference")
    elif "selector_reference" in method:
        _fail("Legacy on-demand method cannot carry a temporal bed reference")
    if schema in (2, 3):
        if method.get("evolving_banks") is not True:
            _fail("On-demand evolving checkpoint needs evolving banks")
        return True
    if "evolving_banks" in method:
        _fail("Legacy on-demand checkpoint cannot carry an evolving-bank method")
    return False


def _signed_gross(signed, gross, label):
    if (np.any(gross < 0.) or
            np.any(gross + 1e-12*np.maximum(1., np.abs(signed)) < np.abs(signed))):
        _fail("Invalid on-demand signed/gross ledger " + label)


def _evolving_map(archive, initial, nc, nf):
    _toe_map_frame(archive, initial, nc, nf)
    state = {key.removeprefix("toe_map_"): archive[key] for key in _toe_map_schema(nc, nf)}
    owner, other = initial["owner"], initial["neighbour"]
    active = state["support_face_bank"] >= 0
    bank, toe = state["support_face_bank"], state["support_face_toe"]
    if (np.any(toe[~active] != -1) or np.any(toe[active] < 0) or np.any(other[active] < 0) or
            not np.all(((owner[active] == bank[active]) & (other[active] == toe[active])) |
                       ((owner[active] == toe[active]) & (other[active] == bank[active])))):
        _fail("On-demand bank support endpoints differ from topology")
    for donor, to, face, distance, bf, tf in (
            ("mapped_donor", "mapped_toe", "mapped_face", "distance", "candidate_bank", "candidate_toe"),
            ("support_donor", "support_toe", "support_face", "support_distance", "support_face_bank", "support_face_toe")):
        valid = state[donor] >= 0
        if (not np.array_equal(state[to] >= 0, valid) or not np.array_equal(state[face] >= 0, valid) or
                np.any(state[distance][~valid] != np.finfo(np.float64).max) or
                not np.array_equal(state[bf][state[face][valid]], state[donor][valid]) or
                not np.array_equal(state[tf][state[face][valid]], state[to][valid])):
            _fail("On-demand bank donor/toe/face support is incomplete")
    groups = (("toe_offset", "toe_faces", state["candidate_toe"], None, nf),
              ("donor_offset", "donor_faces", state["candidate_bank"], None, nf),
              ("candidate_offset", "candidates", state["mapped_donor"], state["distance"], nc),
              ("toe_donor_offset", "toe_donors", np.where(state["is_toe_donor"] != 0,
                                                        state["mapped_toe"], -1), None, nc))
    for offset, item, keys, secondary, size in groups:
        canonical = np.where(keys >= 0, keys, nc)
        identifiers = np.arange(size, dtype=np.int32)
        order = np.lexsort((identifiers, canonical) if secondary is None else
                           (identifiers, secondary, canonical))
        offsets = np.r_[0, np.cumsum(np.bincount(canonical, minlength=nc+1)[:nc], dtype=np.int64)]
        if not np.array_equal(state[offset], offsets) or not np.array_equal(state[item], identifiers[order]):
            _fail("On-demand bank map has noncanonical CSR " + offset)
    return {key: value.copy() for key, value in state.items()}


def validate_evolving_bank_fields(archive: dict, topology: dict, index: dict) -> dict:
    """Validate host frame/checkpoint bank auxiliaries without CUDA imports.

    Checkpoint callers translate ``history_*`` to ``bank_history_*`` first.
    The caller owns the exact complete payload inventory and state/clock checks;
    this routine requires every map, history, full-face and epoch array. Index
    topologies carry authenticated node IDs, operator SHA and initial_data (or
    their authenticated folder/receipt for lazy host decoding).
    """
    initial = topology["initial_data"]
    nc, nf, ng = (int(initial[key]) for key in ("nc", "nf", "nfrac"))
    history = {key: _array(archive, key, (nc,), "float64").copy() for key in _HISTORY_FLOAT}
    history[_HISTORY_COUNT] = _array(archive, _HISTORY_COUNT, (nc,), "int32").copy()
    if any(np.any(value < 0) for value in history.values()):
        _fail("Evolving on-demand bank history is negative")
    if int(initial['bank_toe_erosion_mode']) == 3:
        if int(_scalar(archive, 'bank_mode', 'int64')) != 3 or any(
                key.startswith(('toe_map_', 'face_epoch_')) or key in ('toe_signed', 'toe_gross', 'toe_segment_ids')
                for key in archive):
            _fail('Distributed-bank state cannot contain physical-toe histories')
        diagnostics = {key: int(_scalar(archive, key, 'int64')) for key in
                       ('distributed_stage_index', 'distributed_collapse_waves')}
        if any(value < 0 for value in diagnostics.values()):
            _fail('Invalid distributed-bank diagnostic state')
        return dict(bank_history=history, bank_mode=3, **diagnostics)
    toe_map = _evolving_map(archive, initial, nc, nf)
    signed, gross = (_array(archive, key, (ng, nf), "float64") for key in ("toe_signed", "toe_gross"))
    _signed_gross(signed, gross, "current faces")
    boundary = np.asarray(initial["neighbour"]) < 0
    if np.any(signed[:, boundary]) or np.any(gross[:, boundary]):
        _fail("On-demand exterior faces carry bank transfer")
    count = int(_scalar(archive, "face_epoch_count", "int64"))
    if count < 0 or count > len(archive)//6:
        _fail("On-demand face epoch count is invalid")
    expected = {f"face_epoch_{i}_{key}" for i in range(count) for key in
                ("operator_identity", "node_ids", "old_face", "fraction", "signed", "gross")}
    if {key for key in archive if key.startswith("face_epoch_") and key != "face_epoch_count"} != expected:
        _fail("On-demand face epoch fields are incomplete or unknown")
    epochs = []
    for i in range(count):
        prefix = f"face_epoch_{i}_"
        identity = _identity(_string_scalar(archive, prefix+"operator_identity"), "retired face operator")
        matches = [top for top in index["topologies"].values() if top["operator_sha256"] == identity]
        if len(matches) != 1:
            _fail("On-demand retired face operator is unregistered")
        old = matches[0]
        if "initial_data" not in old:
            old["initial_data"] = _decode_initial(old)
        old_nodes = _array(archive, prefix+"node_ids", (len(old["node_ids"]),), "int64")
        if not np.array_equal(old_nodes, old["node_ids"]):
            _fail("On-demand retired face global nodes differ")
        faces = np.asarray(archive[prefix+"old_face"])
        if faces.ndim != 1 or faces.dtype != np.int64 or np.any(faces < 0) or np.any(faces >= old["initial_data"]["nf"]) or np.any(np.diff(faces) <= 0):
            _fail("On-demand retired face IDs are invalid")
        fraction = _array(archive, prefix+"fraction", faces.shape, "float64")
        if np.any(fraction <= 0) or np.any(fraction > 1):
            _fail("On-demand retired face fractions are invalid")
        es, eg = (_array(archive, prefix+key, (ng, len(faces)), "float64") for key in ("signed", "gross"))
        _signed_gross(es, eg, "retired faces")
        boundary = old["initial_data"]["neighbour"][faces] < 0
        if np.any(es[:, boundary]) or np.any(eg[:, boundary]):
            _fail("On-demand retired exterior faces carry bank transfer")
        epochs.append({key: archive[prefix+key].copy() for key in
                       ("operator_identity", "node_ids", "old_face", "fraction", "signed", "gross")})
    return dict(bank_history=history, bank_mode=2,
                toe_map=toe_map, toe_signed=signed.copy(),
                toe_gross=gross.copy(), face_epochs=epochs)


def _frame(path, row, manifest, topology, topology_index, index):
    if not _evolving_method(manifest):
        return _legacy_frame(path, row, manifest, topology, topology_index)
    archive = _archive(path)
    initial = topology["initial_data"]
    nc, nf, ng, nl, nb = (int(initial[key]) for key in ("nc", "nf", "nfrac", "nbed_layers", "nb"))
    recursive = manifest['method']['checkpoint_schema'] == _RECURSIVE_SCHEMA
    bank_mode=int(initial['bank_toe_erosion_mode'])
    distributed = bank_mode == 3
    evolving=manifest['method'].get('evolving_banks',False)
    bedform = _combined_bedform_frame(manifest, initial,
                                      recursive=recursive,
                                      distributed=distributed)
    if recursive and manifest['method'].get('bank_mode') != bank_mode:
        _fail('Frame method bank mode differs from native operator')
    required = _REQUIRED | set(_HISTORY_FLOAT) | {_HISTORY_COUNT}
    if distributed:
        required |= {'bank_mode', 'distributed_stage_index', 'distributed_collapse_waves', 'topology_identity'}
        if recursive:
            required |= _RECURSIVE_FIELDS
    elif evolving:
        maps = _toe_map_schema(nc, nf)
        count = int(_scalar(archive, "face_epoch_count", "int64"))
        if count < 0 or count > len(archive)//6:
            _fail("On-demand face epoch count is invalid")
        epoch_names = {f"face_epoch_{i}_{key}" for i in range(count) for key in
                       ("operator_identity", "node_ids", "old_face", "fraction", "signed", "gross")}
        required |= {"toe_signed", "toe_gross", "face_epoch_count"} | set(maps) | epoch_names
        if recursive:
            required |= _RECURSIVE_FIELDS | {'topology_identity'}
    else:
        segment_ids=np.asarray(archive.get('toe_segment_ids'))
        if (not recursive or segment_ids.ndim != 1 or
                segment_ids.dtype != np.dtype(np.int64)):
            _fail('Recursive physical-toe segment identities differ')
        required |= {'toe_segment_ids','toe_signed','toe_gross','topology_identity'}
        required |= _RECURSIVE_FIELDS
    if bedform:
        required |= BEDFORM_FIELDS
    if not required <= set(archive) or set(archive) - required - _OPTIONAL:
        _fail("Unsupported evolving on-demand frame fields")
    expected_frame_schema=(
        RECURSIVE_PHYSICAL_TOE_BEDFORM_FRAME_SCHEMA
        if bedform and bank_mode == 2 else
        RECURSIVE_BANK_BEDFORM_FRAME_SCHEMA if bedform else
                           3 if distributed else 4 if recursive else 2)
    if int(_scalar(archive, "frame_schema", "int64")) != expected_frame_schema:
        _fail("Evolving on-demand frame schema differs")
    if (distributed or recursive) and _string_scalar(
            archive, 'topology_identity') != topology['operator_sha256']:
        _fail('Recursive/distributed frame topology identity differs')
    if (_string_scalar(archive, "topology_name") != topology["name"] or
            _string_scalar(archive, "operator_identity") != topology["operator_sha256"] or
            int(_scalar(archive, "regrid_mesh_index", "int64")) != topology_index):
        _fail("Evolving on-demand frame topology differs")
    for key, expected in (("input_identity", manifest["input_preparation_sha256"]),
                          ("source_identity", manifest["source_identity"]),
                          ("method_identity", manifest["method_identity"])):
        if _string_scalar(archive, key) != expected:
            _fail("Evolving on-demand frame identity differs: " + key)
    nodes = _array(archive, "node_ids", (nc,), "int64")
    if not np.array_equal(nodes, topology["node_ids"]):
        _fail("Evolving on-demand global nodes differ")
    base = 3 + ng * (1+nl)
    secondary = int(initial['secondary_index'])
    expected_secondary = -1 if bedform or (
        distributed and np.array_equal(initial['mode'], [2])) else base
    if int(initial["nstate"]) != base+(expected_secondary >= 0) or secondary != expected_secondary:
        _fail("Unsupported evolving on-demand numerical layout")
    state = _array(archive, "state", (nc, int(initial['nstate'])), "float64")
    if np.any(state[:, 0] < -1e-10) or np.any(state[:, 3:base] < -1e-10):
        _fail("Evolving on-demand frame has negative water or material")
    if distributed:
        dry = state[:, 0] == 0
        if (np.any(state[:, 3:3+ng]) or np.any(state[dry, 1:3]) or
                (secondary >= 0 and np.any(state[dry, secondary]))):
            _fail('Distributed-bank frame carries suspended or dry momentum/secondary inventory')
    for key, shape in (("ledger_water", (nb,)), ("ledger_bed", (ng, nb)),
                       ("ledger_secondary", (nb,)), ("ledger_source", ())):
        _array(archive, key, shape, "float64")
    clock, steps = float(_scalar(archive, "time_s", "float64")), int(_scalar(archive, "steps", "int64"))
    if clock < 0 or steps < 0 or not math.isclose(clock, row["time_s"], rel_tol=1e-12, abs_tol=1e-12) or steps != row["steps"]:
        _fail("Evolving on-demand frame clock differs")
    if distributed:
        stage_index = int(_scalar(archive, 'distributed_stage_index', 'int64'))
        waves = int(_scalar(archive, 'distributed_collapse_waves', 'int64'))
        if (stage_index != (0 if recursive else 3 if steps else 0) or waves > len(index['tree']['parent']) or
                (steps == 0 and waves != 0)):
            _fail('Distributed-bank accepted stage/wave counters differ')
    if "rejected" in archive and _scalar(archive, "rejected", "int64") < 0:
        _fail("Evolving on-demand rejected-step count is negative")
    if "dt_hint" in archive and _scalar(archive, "dt_hint", "float64") <= 0:
        _fail("Evolving on-demand timestep hint is invalid")
    if distributed or evolving:
        bank = validate_evolving_bank_fields(archive, topology, index)
    else:
        history = {key:_array(archive,key,(nc,),"float64").copy()
                   for key in _HISTORY_FLOAT}
        history[_HISTORY_COUNT]=_array(archive,_HISTORY_COUNT,(nc,),"int32").copy()
        if any(np.any(value < 0) for value in history.values()):
            _fail('Recursive physical-toe bank history is negative')
        ids=np.asarray(archive['toe_segment_ids'])
        expected_count=int(np.count_nonzero(initial['is_toe_face']))
        if (ids.shape != (expected_count,) or np.any(ids <= 0) or
                len(np.unique(ids)) != len(ids)):
            _fail('Recursive physical-toe segment identities are invalid')
        signed=_array(archive,'toe_signed',(ng,expected_count),'float64')
        gross=_array(archive,'toe_gross',(ng,expected_count),'float64')
        _signed_gross(signed,gross,'physical-toe segments')
        bank=dict(bank_history=history,toe_segment_ids=ids.copy(),
                  toe_signed=signed.copy(),toe_gross=gross.copy())
    if recursive:
        from .gpu_recursive_checkpoint import validate_frame_diagnostics, validate_adaptation_method
        from .gpu_on_demand_prepared import recursive_boundary_event_identity
        if (manifest['method']['boundary_forcing']['event_identity'] !=
                recursive_boundary_event_identity(initial)):
            _fail('Recursive boundary event clock differs from authenticated operator')
        validate_adaptation_method(manifest['method'], initial)
        bank['recursive'] = validate_frame_diagnostics(archive, topology, index, steps, clock)
    bedform_values = None
    if bedform:
        bedform_values = {key:_array(archive, key, (nc,),
            'int32' if key == 'bedform_iterations' else 'float64').copy()
            for key in BEDFORM_FIELDS}
        if any(np.any(value < 0) for value in bedform_values.values()):
            _fail('Recursive bank bedform diagnostics are negative')
    rock, bulk = np.asarray(initial["rock"]), np.asarray(initial["bulk_fraction"])
    if rock.shape != (nc,) or bulk.shape != (ng,) or np.any(bulk <= 0):
        _fail("Invalid on-demand bed material geometry")
    bed = rock + (state[:, 3+ng:base].reshape(nc, nl, ng) / bulk[None, None, :]).sum(axis=(1, 2))
    if "bed_elevation" in archive and not np.allclose(_array(archive, "bed_elevation", (nc,), "float64"), bed, rtol=0, atol=1e-8):
        _fail("Evolving on-demand bed elevation differs from layers")
    return dict(time_s=clock, steps=steps, mesh_id=topology_index, topology_name=topology["name"],
        node_ids=nodes.copy(), q=state[:, :base].copy(),
        secondary=state[:, base].copy() if secondary >= 0 else np.zeros(nc), bed_elevation=bed,
        nfrac=ng, bed_layers=nl, suspended_enabled=bool(int(initial["exchange_mode"]) != 0),
        bedform=bedform_values, **bank,
        ledger={key: archive[key].copy() for key in ("ledger_water", "ledger_bed", "ledger_secondary", "ledger_source")})


def _pending_operator_directory(path: Path) -> bool:
    if _PENDING_OPERATOR.fullmatch(path.name) is None:
        return False
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        # A staging directory can be removed after iterdir's snapshot.
        return True
    return stat.S_ISDIR(mode) and not stat.S_ISLNK(mode)


def _preparation(root: Path, manifest: dict, prepared_root: str | Path | None) -> dict:
    expected = _identity(manifest.get("input_preparation_sha256"),
                         "AMR preparation")
    if manifest.get("preparation_receipt_sha256") != expected:
        _fail("On-demand preparation receipt identity differs")
    path = root / "preparation_receipt.json"
    if path.is_symlink() or _digest(path) != expected:
        _fail("Copied on-demand preparation receipt differs")
    receipt = _json(path)
    if (receipt.get("schema") != 6 or
            receipt.get("variant") != "on_demand_bank_amr_ssp33" or
            receipt.get("native_solver_steps_performed") != 0 or
            receipt.get("initialization_CPU_only") is not True or
            receipt.get("GPU_solver_executed") is not False or
            receipt.get("portable") is not False):
        _fail("Unsupported on-demand preparation receipt")
    session = _identity(receipt.get("session_identity"), "AMR session")
    if (manifest.get("input_identity") != session or
            manifest["method"].get("session_identity") != session):
        _fail("On-demand run/session identity differs")
    _identity(receipt.get("session_manifest_sha256"), "AMR session manifest")
    _identity(receipt.get("runtime_build_sha256"), "AMR runtime build")
    originals = receipt.get("original_input_files")
    if (not isinstance(originals, dict) or not originals or
            "amr_hierarchy.npz" not in originals):
        _fail("On-demand original-input family is absent")
    base_digest=_base_input_digest(originals)
    adapter=receipt.get('recursive_geometry_adapter')
    if 'case.data' not in originals:
        if (not isinstance(adapter,dict) or adapter.get('schema')!=1 or
                adapter.get('applied') is not True or
                adapter.get('physical_payload_unchanged') is not True or
                adapter.get('adapted_deck_sha256')!=base_digest or
                originals.get(adapter.get('adapted_deck_name'))!=base_digest or
                adapter.get('identity')!=_adapter_identity({key:value for key,value
                    in adapter.items() if key!='identity'})):
            _fail('On-demand recursive geometry adapter differs')
    elif adapter is not None:
        _fail('Unexpected on-demand recursive geometry adapter')
    for name, value in originals.items():
        if (not isinstance(name, str) or not name or "/" in name or
                "\\" in name or name in (".", "..")):
            _fail("On-demand original-input name is unsafe")
        _identity(value, "original input " + name)
    if prepared_root is not None:
        external = filesystem_path(Path(prepared_root).expanduser()).resolve()
        candidate = external / "preparation_receipt.json"
        if candidate.is_symlink() or _digest(candidate) != expected:
            _fail("Supplied on-demand preparation receipt differs")
        session_path = external / "session" / "session.json"
        if (session_path.is_symlink() or
                _digest(session_path) != receipt["session_manifest_sha256"]):
            _fail("Supplied on-demand session manifest differs")
        session_record = _json(session_path)
        if (session_record.get("identity") != session or
                session_record.get("original_input_files") != originals or
                session_record.get("native_solver_steps") != 0):
            _fail("Supplied on-demand session family differs")
    return receipt


def _package(root: Path, row: dict, receipt: dict) -> tuple[dict, dict]:
    if not isinstance(row, dict):
        _fail("On-demand operator inventory row is invalid")
    identity = _identity(row.get("package_identity"), "AMR operator package")
    operator = _identity(row.get("operator_identity"), "AMR operator")
    if row.get("path") != "operators/" + identity:
        _fail("On-demand operator package path is unsafe")
    folder = root / "operators" / identity
    if (folder.is_symlink() or not folder.is_dir() or
            not folder.resolve().is_relative_to(root)):
        _fail("On-demand operator package is missing or linked")
    manifest_path = folder / "amr_prepared.json"
    if manifest_path.is_symlink() or _digest(manifest_path) != identity:
        _fail("On-demand operator package identity differs")
    package = _json(manifest_path)
    if (package.get("schema") != 2 or
            package.get("kind") != "braids-gpu-amr-prepared" or
            package.get("ratio") != 2 or
            package.get("CPU_numerical_fallback") is not False or
            package.get("solver_steps_performed_by_exporter") != 0):
        _fail("Unsupported on-demand operator package")
    source = package.get("source")
    originals = receipt["original_input_files"]
    if (not isinstance(source, dict) or
            source.get("on_demand_initialization") is not True or
            source.get("base_input_sha256") != _base_input_digest(originals) or
            source.get("hierarchy_sha256") != originals["amr_hierarchy.npz"] or
            source.get("runtime_build_sha256") != receipt["runtime_build_sha256"]):
        _fail("On-demand operator source family differs")
    artifacts = package.get("artifacts")
    records = package.get("topologies")
    if (not isinstance(artifacts, dict) or not artifacts or
            not isinstance(records, list) or len(records) != 1 or
            not isinstance(records[0], dict)):
        _fail("On-demand operator is not a single authenticated topology")
    for name, digest in artifacts.items():
        _identity(digest, "AMR package artifact")
        try:
            artifact = prepared_io._artifact(folder, name)
        except ValueError as exc:
            _fail("Invalid on-demand package artifact: " + str(exc))
        if _digest(artifact) != digest:
            _fail("On-demand package artifact differs: " + name)
    if (artifacts.get(package.get("runtime_build")) !=
            receipt["runtime_build_sha256"] or "tree.npz" not in artifacts):
        _fail("On-demand operator runtime or hierarchy artifact differs")
    expected_files = set(artifacts) | {"amr_prepared.json"}
    for member in folder.rglob("*"):
        if member.is_symlink():
            _fail("On-demand operator package contains a link")
    actual_files = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
                    if p.is_file()}
    if actual_files != expected_files:
        _fail("On-demand operator package file inventory differs")
    topo = records[0]
    name = topo.get("name")
    if (not isinstance(name, str) or not name or row.get("name") != name or
            any(not isinstance(topo.get(key), str) or topo[key] not in artifacts
                for key in ("node_ids", "geometry", "operator", "receipt")) or
            artifacts[topo["operator"]] != operator):
        _fail("On-demand topology/operator binding differs")
    try:
        node_path = prepared_io._artifact(folder, topo["node_ids"])
        geometry_path = prepared_io._artifact(folder, topo["geometry"])
        receipt_path = prepared_io._artifact(folder, topo["receipt"])
        operator_path = prepared_io._artifact(folder, topo["operator"])
        tree = prepared_io._tree(folder / "tree.npz", 2)
        nodes = prepared_io._selected_nodes(node_path, tree)
    except ValueError as exc:
        _fail("On-demand operator geometry differs: " + str(exc))
    topology_folder = node_path.parent
    if any(path.parent != topology_folder for path in
           (geometry_path, receipt_path, operator_path)):
        _fail("On-demand topology artifacts have different folders")
    sidecars = topo.get("sidecars", {})
    if not isinstance(sidecars, dict):
        _fail("On-demand topology sidecar inventory differs")
    for relative in sidecars.values():
        if not isinstance(relative, str) or relative not in artifacts:
            _fail("On-demand topology sidecar is unauthenticated")
        try:
            if prepared_io._artifact(folder, relative).parent != topology_folder:
                _fail("On-demand topology sidecar folder differs")
        except ValueError as exc:
            _fail("Invalid on-demand topology sidecar: " + str(exc))
    topology_receipt = _json(receipt_path)
    bank_mode = source.get('bank_mode', 2)
    if type(bank_mode) is not int or bank_mode not in (2, 3):
        _fail('On-demand operator bank mode is invalid')
    build = _json(folder / package['runtime_build'])
    if bank_mode == 3:
        expected_sidecars = {'distributed_bank': 'distributed_bank_metadata.bin',
            'bank_history': 'bank_history_metadata.bin', 'avalanche': 'avalanche_metadata.bin',
            'normal': 'normal_metadata.bin'}
        historical=(build.get('kind') == 'live-selected-bank-mode23-amr' and
            build.get('bank_modes') == [2, 3] and
            build.get('bank_metadata_contract') == 'mode_dispatch_v1' and
            'recursive_bank_bedform_avalanche' not in source and
            'recursive_bank_bedform_avalanche' not in topology_receipt)
        exact_revision = _READABLE_BEDFORM_BANK_RUNTIMES.get(build.get('kind'))
        exact=(exact_revision is not None and
            build.get('parser_scope_revision') == exact_revision and
            build.get('bank_modes') == [3] and
            build.get('bank_metadata_contract') ==
                'recursive_bedform_mode3_parent_avalanche_v1' and
            build.get('geometry_initialization_version') == 6 and
            build.get('export_only') is True and
            build.get('recursive_bank_bedform_avalanche') is True and
            source.get('recursive_bank_bedform_avalanche') is True and
            topology_receipt.get('recursive_bank_bedform_avalanche') is True)
        if (not (historical or exact) or
                source.get('live_boundary_metadata') is not True or
                source.get('toe_geometry_seed_only') is not False or
                type(topology_receipt.get('bank_mode')) is not int or topology_receipt['bank_mode'] != 3 or
                topology_receipt.get('toe_geometry_seed_only', False) is not False or 'toe_faces' in topology_receipt or
                sidecars != {k: topology_folder.name+'/'+v for k, v in expected_sidecars.items()} or
                set(topology_receipt.get('artifacts', {})) != {'coupled_input.bin', *expected_sidecars.values()} or
                any(topology_receipt['artifacts'][v] != artifacts.get(topology_folder.name+'/'+v)
                    for v in ('coupled_input.bin', *expected_sidecars.values()))):
            _fail('On-demand distributed-bank metadata dispatch differs')
    elif (build.get('kind') ==
            'live-selected-bank-mode2-bedform-avalanche-amr-v1' or
            source.get('recursive_bank_bedform_avalanche') is True or
            topology_receipt.get('recursive_bank_bedform_avalanche') is True):
        expected_sidecars = {'bank_toe': 'bank_toe_metadata.bin',
            'bank_history': 'bank_history_metadata.bin',
            'avalanche': 'avalanche_metadata.bin',
            'normal': 'normal_metadata.bin'}
        exact_revision = _READABLE_BEDFORM_BANK_RUNTIMES.get(build.get('kind'))
        exact = (build.get('kind') ==
                    'live-selected-bank-mode2-bedform-avalanche-amr-v1' and
            exact_revision ==
                'exact_recursive_mode2_physical_toe_bedform_parent_avalanche_v1' and
            build.get('parser_scope_revision') == exact_revision and
            build.get('bank_modes') == [2] and
            build.get('bank_metadata_contract') ==
                'recursive_bedform_mode2_physical_toe_parent_avalanche_v1' and
            build.get('geometry_initialization_version') == 7 and
            build.get('export_only') is True and
            build.get('recursive_bank_bedform_avalanche') is True and
            build.get('live_boundary_metadata') is True and
            build.get('toe_geometry_seed_only') is True and
            build.get('evolving_toe_geometry') is True and
            build.get('source_original_modified') is False and
            build.get('Delft3D_source_modified') is False and
            source.get('recursive_bank_bedform_avalanche') is True and
            source.get('live_boundary_metadata') is True and
            source.get('toe_geometry_seed_only') is True and
            topology_receipt.get('recursive_bank_bedform_avalanche') is True and
            topology_receipt.get('live_boundary_metadata') is True and
            topology_receipt.get('toe_geometry_seed_only') is True and
            type(topology_receipt.get('bank_mode')) is int and
            topology_receipt['bank_mode'] == 2)
        if (not exact or
                type(topology_receipt.get('toe_faces')) is not int or
                topology_receipt['toe_faces'] < 0 or
                sidecars != {k: topology_folder.name+'/'+v
                             for k, v in expected_sidecars.items()} or
                set(topology_receipt.get('artifacts', {})) !=
                    {'coupled_input.bin', *expected_sidecars.values()} or
                any(topology_receipt['artifacts'][v] !=
                    artifacts.get(topology_folder.name+'/'+v)
                    for v in ('coupled_input.bin', *expected_sidecars.values()))):
            _fail('On-demand physical-toe bedform metadata dispatch differs')
    origin = topology_receipt.get("origin")
    files = origin.get("files") if isinstance(origin, dict) else None
    if not isinstance(files, dict):
        _fail("On-demand topology receipt has no original-input inventory")
    family = {key: value for key, value in files.items()
              if key != "selected_nodes.txt"}
    if (family != originals or
            topology_receipt.get("runtime_build_sha256") !=
            receipt["runtime_build_sha256"] or
            topology_receipt.get("initialization_CPU_only") is not True or
            topology_receipt.get("solver_steps_performed") != 0 or
            topology_receipt.get("GPU_solver_executed") is not False):
        _fail("On-demand topology receipt source family differs")
    # Initializer receipts identify the original deck rather than the copied
    # topology geometry.  The latter is authenticated by the package artifact
    # table, so pass its digest to the existing geometry decoder in memory.
    geometry_receipt = {"origin": {"files": {
        "geometry.npz": artifacts[topo["geometry"]]}}}
    geometry = _geometry(topology_folder, geometry_receipt)
    if (geometry["cell_count"] != len(nodes) or
            not np.allclose(geometry["area_m2"], tree["area"][nodes],
                            rtol=1e-11, atol=1e-11) or
            not np.allclose(geometry["centroid_xy"], tree["xy"][nodes],
                            rtol=1e-11, atol=1e-11)):
        _fail("On-demand selected nodes and copied geometry differ")
    topology = dict(name=name, folder=topology_folder, node_ids=nodes,
                    geometry=geometry, receipt_sha256=artifacts[topo["receipt"]],
                    operator_sha256=operator, bank_mode=bank_mode,
                    normal_metadata=(prepared_io._artifact(folder, sidecars['normal'])
                                     if 'normal' in sidecars else None))
    return tree, topology


def _index(root: Path, manifest: dict, receipt: dict, *, live: bool = False) -> dict:
    rows = manifest.get("operators")
    if (live and rows == [] and manifest.get("topologies") == [] and
            manifest.get("frames") == [] and manifest.get("status") == "initializing"):
        return dict(root=root, manifest_sha256=manifest["input_preparation_sha256"],
                    tree=None, topologies={})
    if not isinstance(rows, list) or not rows:
        _fail("On-demand run has no registered operators")
    operators_folder = root / "operators"
    if operators_folder.is_symlink() or not operators_folder.is_dir():
        _fail("On-demand operator inventory is missing or linked")
    topologies = {}
    tree = None
    identities = set()
    for row in rows:
        package_identity = row.get("package_identity") if isinstance(row, dict) else None
        _identity(package_identity, "AMR operator package")
        if package_identity in identities:
            _fail("Duplicate on-demand operator package")
        identities.add(package_identity)
        candidate_tree, topology = _package(root, row, receipt)
        if topology['bank_mode'] != manifest['method'].get('bank_mode', 2):
            _fail('Run method bank mode differs from operator metadata')
        if tree is None:
            tree = candidate_tree
        elif (set(candidate_tree) != set(tree) or
              any(not np.array_equal(candidate_tree[key], tree[key]) for key in tree)):
            _fail("On-demand operator hierarchy changed during run")
        if topology["name"] in topologies:
            _fail("Duplicate on-demand topology name")
        topologies[topology["name"]] = topology
    if manifest.get("topologies") != list(topologies):
        _fail("On-demand registered topology/operator order differs")
    members = list(operators_folder.iterdir())
    actual = {path.name for path in members}
    if actual != identities:
        # Publishing an immutable operator directory precedes publishing the
        # run.json commit pointer. Never read these unreferenced payloads, nor
        # let them change this snapshot's topology order. Terminal/full reads
        # still require the exact directory inventory.
        pending = [path for path in members if path.name not in identities]
        if (not live or manifest.get("status") not in _ACTIVE_STATUSES or
                not identities <= actual or any(
                    not _pending_operator_directory(path) for path in pending)):
            _fail("On-demand registered topology/operator order differs")
    return dict(root=root, manifest_sha256=manifest["input_preparation_sha256"],
                tree=tree, topologies=topologies)


def _rows(manifest: dict, index: dict, *, require_nonempty: bool) -> tuple[list, float, int]:
    rows = manifest.get("frames")
    if not isinstance(rows, list) or (require_nonempty and not rows):
        _fail("On-demand frame inventory is invalid")
    previous_time, previous_steps = -math.inf, -1
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            _fail("On-demand frame inventory row is invalid")
        filename = row.get("file")
        if (not isinstance(filename, str) or _FRAME.fullmatch(filename) is None or
                filename != f"frame_{number:06d}.npz"):
            _fail("On-demand frame filename is unsafe or out of order")
        _identity(row.get("sha256"), f"AMR frame {number}")
        clock = _number(row.get("time_s"), "AMR frame time", minimum=0)
        steps = _number(row.get("steps"), "AMR frame steps", integer=True, minimum=0)
        if number and (clock <= previous_time or steps <= previous_steps):
            _fail("On-demand frame clocks/steps are not increasing")
        previous_time, previous_steps = clock, steps
        name = row.get("topology")
        if not isinstance(name, str) or name not in index["topologies"]:
            _fail("On-demand frame names an unknown topology")
        topology = index["topologies"][name]
        if (row.get("operator_identity") != topology["operator_sha256"] or
                row.get("cell_count") != len(topology["node_ids"])):
            _fail("On-demand frame topology binding differs")
    return rows, previous_time, previous_steps


def _common(run_root: str | Path, prepared_root: str | Path | None, *, live=False):
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    manifest = _json(root / "run.json")
    status = _json(root / "status.json")
    if (manifest.get("schema") != 3 or
            manifest.get("format") != "braids-gpu-amr-run-v2" or
            manifest.get("input_kind") != "amr-on-demand" or
            manifest.get("backend") != "gpu" or
            manifest.get("backend_selected_by") != "user" or
            manifest.get("automatic_backend_selection") is not False):
        _fail("Run is not an explicitly selected schema-3 on-demand GPU AMR output")
    _method_identity(manifest)
    _evolving_method(manifest)
    _source_identity(root, manifest)
    folder = manifest.get("input_folder")
    if not isinstance(folder, str) or not folder:
        _fail("On-demand input-preparation folder is absent")
    receipt = _preparation(root, manifest, prepared_root)
    index = _index(root, manifest, receipt, live=live)
    if (status.get("backend") != "gpu" or status.get("status") not in _STATUSES or
            manifest.get("status") not in _STATUSES):
        _fail("On-demand worker status differs")
    if manifest["status"] != status["status"]:
        if live and _json(root / "run.json") != manifest:
            # A new authoritative pointer proves that our status and manifest
            # snapshots are from different commits. No payload error is caught.
            raise _SnapshotAdvanced()
        # The writer commits run.json before status.json. A valid active old
        # status is deferred only by the live reader after frame authentication.
        pending_transition = (status["status"] == "initializing" and
                              manifest["status"] != "initializing") or (
                              status["status"] == "running" and
                              manifest["status"] not in _ACTIVE_STATUSES)
        if not live or not pending_transition:
            _fail("On-demand worker status differs")
    return root, manifest, status, index


def _status(status: dict, manifest: dict, previous_time: float,
            previous_steps: int, *, live=False) -> bool:
    status_time = _number(status.get("time_s"), "GPU AMR status time", minimum=0)
    status_steps = _number(status.get("steps"), "GPU AMR status steps",
                           integer=True, minimum=0)
    if live and status["status"] in _ACTIVE_STATUSES and (
            manifest["status"] != status["status"] or
            (status_time <= previous_time + 1e-12 and status_steps <= previous_steps and
             (status_time + 1e-12 < previous_time or status_steps < previous_steps))):
        # No frame is returned as valid while the separate progress document
        # has yet to catch up with the authenticated run.json commit.
        return False
    if status_time + 1e-12 < previous_time or status_steps < previous_steps:
        _fail("On-demand status predates last committed frame")
    if manifest["status"] == "completed" and (
            not math.isclose(status_time, previous_time, rel_tol=1e-12,
                             abs_tol=1e-12) or status_steps != previous_steps):
        _fail("Completed on-demand status differs from final frame")
    return True


def _decode_initial(topology: dict) -> dict:
    try:
        data = read_gpu_prepared(topology["folder"],
                                 receipt_sha256=topology["receipt_sha256"])
        normal = topology.get('normal_metadata')
        if normal is not None:
            from braids_core.gpu_live_boundary_metadata import \
                validate_live_boundary_metadata
            data = validate_live_boundary_metadata(data, _read_records(normal))
    except (OSError, ValueError, KeyError) as exc:
        _fail("Cannot decode on-demand operator initial state: " + str(exc))
    if int(data["nc"]) != len(topology["node_ids"]):
        _fail("On-demand operator state and selected nodes differ")
    bank_mode = topology.get('bank_mode')
    if bank_mode in (2, 3):
        from .gpu_on_demand_prepared import _scope
        sidecars = (('bank_toe_metadata.bin', 'bank_history_metadata.bin',
                     'avalanche_metadata.bin') if bank_mode == 2 else
                    ('distributed_bank_metadata.bin',
                     'bank_history_metadata.bin', 'avalanche_metadata.bin'))
        for filename in sidecars:
            for key, value in _read_records(topology['folder']/filename).items():
                if key in data and not np.array_equal(data[key], value):
                    _fail('Bank sidecar conflicts: '+key)
                data[key] = value
        try:
            _scope(data, live_boundaries=True, bank_mode=bank_mode,
                   toe_geometry_seed_only=(
                       bank_mode == 2 and
                       data.get('toe_geometry_seed_only') == 1))
        except ValueError as exc:
            _fail('Invalid evolving-bank operator controls: '+str(exc))
    return data


def read_gpu_on_demand_run(run_root: str | Path, *,
                           prepared_root: str | Path | None = None) -> dict:
    """Authenticate all committed frames, with no external preparation needed."""
    # Dispatch peeks without consuming the reader's authoritative commit read.
    # A live writer may advance run.json between these two operations.
    manifest = json.loads((filesystem_path(Path(run_root).expanduser()).resolve() /
                           'run.json').read_text(encoding='utf-8'))
    if manifest.get('input_kind') == 'amr-on-demand-general':
        from .gpu_general_on_demand_io import read_run
        return read_run(run_root, prepared_root=prepared_root)
    root, manifest, status, index = _common(run_root, prepared_root)
    rows, previous_time, previous_steps = _rows(manifest, index,
                                                  require_nonempty=True)
    # QGIS mesh/raster materialization may inspect an operator registered
    # between output clocks even when no frame ultimately selected it.
    for topology in index["topologies"].values():
        topology["initial_data"] = _decode_initial(topology)
        if manifest['method']['checkpoint_schema'] == _RECURSIVE_SCHEMA:
            from .gpu_recursive_checkpoint import validate_adaptation_method
            validate_adaptation_method(manifest['method'], topology['initial_data'])
    converted = []
    for row in rows:
        path = root / row["file"]
        if path.is_symlink() or _digest(path) != row["sha256"]:
            _fail("On-demand frame hash differs: " + row["file"])
        topology = index["topologies"][row["topology"]]
        item = _frame(path, row, manifest, topology,
                      list(index["topologies"]).index(row["topology"]), index)
        if item["bank_history"] is None:
            _fail("On-demand bank-history frame is incomplete")
        if converted and (item["nfrac"], item["bed_layers"]) != (
                converted[0]["nfrac"], converted[0]["bed_layers"]):
            _fail("On-demand material state layout changed")
        converted.append(item)
    if manifest["status"] in {"completed", "step_limit", "cancelled"}:
        if {p.name for p in root.glob("frame_*.npz")} != {
                row["file"] for row in rows}:
            _fail("Final on-demand output contains unlisted frames")
    _status(status, manifest, previous_time, previous_steps)
    return dict(manifest=manifest, status=status, prepared_index=index,
                frames=converted)


def read_gpu_on_demand_live_frame(run_root: str | Path, *,
                                  prepared_root: str | Path | None = None,
                                  after_index: int = -1) -> dict | None:
    """Authenticate the latest committed frame without reopening prior payloads."""
    if isinstance(after_index, bool) or not isinstance(after_index, int) or after_index < -1:
        _fail("Invalid on-demand live-preview cursor")
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    if not (root / "run.json").is_file():
        return None
    if json.loads((root / 'run.json').read_text(encoding='utf-8')).get(
            'input_kind') == 'amr-on-demand-general':
        from .gpu_general_on_demand_io import read_live_frame
        return read_live_frame(root, prepared_root=prepared_root,
                               after_index=after_index)
    try:
        root, manifest, status, index = _common(root, prepared_root, live=True)
    except _SnapshotAdvanced:
        return None
    rows, previous_time, previous_steps = _rows(manifest, index,
                                                  require_nonempty=False)
    latest = len(rows) - 1
    if latest <= after_index:
        return None
    row = rows[latest]
    path = root / row["file"]
    if path.is_symlink() or _digest(path) != row["sha256"]:
        _fail("On-demand live frame hash differs")
    topology = index["topologies"][row["topology"]]
    topology["initial_data"] = _decode_initial(topology)
    frame = _frame(path, row, manifest, topology,
                   list(index["topologies"]).index(row["topology"]), index)
    if frame["bank_history"] is None:
        _fail("On-demand bank-history frame is incomplete")
    if not _status(status, manifest, previous_time, previous_steps, live=True):
        return None
    return dict(frame_index=latest, frame=frame, manifest=manifest,
                topology=topology, prepared_root=root)
