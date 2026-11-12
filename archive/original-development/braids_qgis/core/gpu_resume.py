"""Read-only preflight for GPU restarts selected in QGIS.

This authenticates the published run envelope and checkpoint identities before
the dock offers a restart. The GPU worker still validates all state arrays,
shapes and numerical method before creating its new output folder.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from braids_core.gpu_capabilities import gpu_integrator_label
from braids_core.sediment_method_catalog import (
    CATALOG_SCHEMA, CATALOG_SCOPE_ID, semantic_catalog_id)
from braids_core.windows_paths import filesystem_path


_RESUMABLE = frozenset(("cancelled", "step_limit", "completed"))
_IMPLICIT_TOKEN = "implicit_fwave_midpoint"


def validate_fixed_grid_restart_selection(record: dict, *, token: str,
                                           variant: str, launch_mode: str,
                                           dtmax_s: float | None = None,
                                           output_every_s: float | None = None) -> dict:
    """Bind a requested fixed-grid restart to its authenticated saved method.

    ``inspect_gpu_resume_checkpoint`` has already authenticated the frozen
    source, input, event and checkpoint identities.  This final pure host gate
    prevents the QGIS selection from changing the variant or integration
    method; implicit restarts additionally retain their timestep/output
    controls because both participate in the qualified nonlinear method.
    """
    if not isinstance(record, dict) or not isinstance(record.get("method"), dict):
        raise ValueError("GPU fixed-grid restart method identity is incomplete")
    method = record["method"]
    digest = hashlib.sha256(json.dumps(method, sort_keys=True,
                            separators=(",", ":")).encode()).hexdigest()
    if (record.get("method_identity") != digest or
            method.get("source_identity") != record.get("source_identity")):
        raise ValueError("GPU fixed-grid restart method/source identity differs")
    try:
        expected_integrator = gpu_integrator_label(token)
    except ValueError as error:
        raise ValueError("Unsupported GPU restart integrator") from error
    if (method.get("variant") != variant or
            method.get("integrator") != expected_integrator or
            method.get("launch_mode") != launch_mode):
        raise ValueError(
            "GPU fixed-grid restart uses a different variant, integrator or launch mode")
    if token == _IMPLICIT_TOKEN:
        for value, name in ((dtmax_s, "timestep"),
                            (output_every_s, "output interval")):
            if (type(value) not in (int, float) or isinstance(value, bool) or
                    not math.isfinite(value) or value <= 0.):
                raise ValueError("GPU implicit restart needs a finite positive " + name)
        if (method.get("dtmax_s") != float(dtmax_s) or
                method.get("output_every_s") != float(output_every_s)):
            raise ValueError(
                "GPU implicit restart changes the exact saved timestep/output method")
    return method


def _sha256(path: Path) -> str:
    with filesystem_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(path: Path) -> dict:
    value = json.loads(filesystem_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path.name + " must contain an object")
    return value


def _scalar(archive, name: str):
    if name not in archive.files:
        raise ValueError("GPU checkpoint is missing " + name)
    value = archive[name]
    if value.shape != ():
        raise ValueError("GPU checkpoint has non-scalar " + name)
    return value.item()


def inspect_gpu_resume_checkpoint(run_root: str | Path,
                                  prepared_root: str | Path) -> dict:
    """Return an authenticated checkpoint envelope, or fail before launch."""
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    prepared = filesystem_path(Path(prepared_root).expanduser()).resolve()
    from .sediment_product import validate_gpu_result_product
    validate_gpu_result_product(root, prepared)
    if (root / "run.json").is_file() and _object(root / "run.json").get("schema") == 3:
        return inspect_gpu_on_demand_resume_checkpoint(run_root, prepared_root)
    checkpoint = root / "checkpoint.npz"
    if not checkpoint.is_file() or not (root / "run.json").is_file() or not (root / "status.json").is_file():
        raise ValueError("GPU checkpoint folder is incomplete")
    manifest = _object(root / "run.json")
    status = _object(root / "status.json")
    state = manifest.get("status")
    if (manifest.get("schema") != 1 or manifest.get("backend") != "gpu" or
            manifest.get("backend_selected_by") != "user" or
            manifest.get("automatic_backend_selection") is not False or
            state not in _RESUMABLE or status.get("status") != state or
            status.get("backend") != "gpu"):
        raise ValueError("Selected folder is not a resumable completed/stopped GPU run")
    receipt = prepared / "preparation_receipt.json"
    input_identity = _sha256(receipt)
    receipt_record = _object(receipt)
    if manifest.get("input_preparation_sha256") != input_identity:
        raise ValueError("GPU checkpoint belongs to a different prepared input")
    source = root / "source" / "runtime_manifest.json"
    if not source.is_file():
        source = root / "source" / "manifest.json"
    source_identity = _sha256(source)
    if manifest.get("source_identity") != source_identity:
        raise ValueError("GPU checkpoint's frozen source identity differs")
    method = manifest.get("method")
    if not isinstance(method, dict) or method.get("source_identity") != source_identity:
        raise ValueError("GPU checkpoint method/source identity is incomplete")
    if method.get("variant") == "live_normal_critical_transport_ssp33":
        catalog_fields = (
            "sediment_catalog_schema", "sediment_catalog_scope_id",
            "semantic_catalog_id",
        )
        catalog_values = tuple(method.get(name) for name in catalog_fields)
        if all(value is None for value in catalog_values):
            # Frozen schema-6 transport checkpoints predate the catalogue
            # triplet.  Their preparation receipt, source, method and state
            # identities are still authenticated below.  This exception is
            # deliberately absence-only: current schema-7 preparation and a
            # partial modern identity must both fail closed.
            if receipt_record.get("schema") != 6:
                raise ValueError("GPU checkpoint sediment-method catalogue differs")
        elif (method.get("sediment_catalog_schema") != CATALOG_SCHEMA or
              method.get("sediment_catalog_scope_id") != CATALOG_SCOPE_ID or
              method.get("semantic_catalog_id") != semantic_catalog_id()):
            raise ValueError("GPU checkpoint sediment-method catalogue differs")
    method_identity = hashlib.sha256(json.dumps(method, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    if manifest.get("method_identity") != method_identity:
        raise ValueError("GPU checkpoint method identity differs")
    if method.get('variant') == 'live_normal_critical_bank_ssp33':
        return _inspect_fixed_bank_checkpoint(root, prepared, manifest, status)
    if method.get('variant') == 'live_normal_critical_transport_ssp33':
        return _inspect_fixed_bank_checkpoint(root, prepared, manifest, status, transport=True)
    with np.load(checkpoint, allow_pickle=False) as archive:
        for name, expected in (("input_identity", input_identity),
                               ("source_identity", source_identity),
                               ("method_identity", method_identity)):
            if _scalar(archive, name) != expected:
                raise ValueError("GPU checkpoint " + name + " differs")
        time_s = float(_scalar(archive, "time"))
        steps = int(_scalar(archive, "steps"))
    final_time = float(manifest["final_time_s"])
    if (not math.isfinite(time_s) or time_s < 0 or
            not math.isfinite(final_time) or
            not math.isclose(time_s, final_time, rel_tol=0, abs_tol=1e-12) or
            steps < 0 or steps != manifest.get("steps") or steps != status.get("steps")):
        raise ValueError("GPU checkpoint accepted clock differs from run status")
    return {"checkpoint": checkpoint, "time_s": time_s, "steps": steps,
            "status": state, "method": method,
            "input_identity": input_identity, "source_identity": source_identity,
            "method_identity": method_identity}


def _inspect_fixed_bank_checkpoint(root, prepared, manifest, status, *, transport=False):
    """Authenticate compact fixed-grid state and nonprognostic neutral fields."""
    from . import gpu_frame_io as frame_io
    from .gpu_prepared_io import read_gpu_prepared, _bank_numerical_identity
    live = frame_io.read_gpu_live_frame(root, prepared)
    if live is None:
        raise ValueError('GPU bank checkpoint has no committed final frame')
    receipt = _object(prepared/'preparation_receipt.json')
    data = read_gpu_prepared(prepared, receipt_sha256=manifest['input_preparation_sha256'])
    if transport:
        from . import gpu_live_transport_io as transport_io
        identity = transport_io.numerical_identity(receipt)
        context = transport_io.preparation_context(prepared, receipt, manifest, int(data['nc']))
    else:
        identity = _bank_numerical_identity(receipt)
        context = frame_io._bank_preparation(prepared, receipt, manifest, int(data['nc']))
    checkpoint = root/'checkpoint.npz'
    if checkpoint.is_symlink():
        raise ValueError('GPU bank checkpoint cannot be linked')
    row = manifest['frames'][-1]
    if transport:
        checkpoint_context = dict(context, checkpoint=True)
        kwargs = {'transport_context': checkpoint_context}
    else:
        kwargs = {'bank_context': context}
    frame_io._frame(checkpoint, row, manifest, int(data['nc']), int(data['nf']), **kwargs)
    fields = frame_io._archive(checkpoint)
    final = frame_io._archive(root/row['file'])
    expected = set(final) - frame_io._ADAPTATION
    checkpoint_only = set()
    if transport:
        expected -= transport_io.BEDFORM
        if context.get('implicit', False):
            checkpoint_only = set(transport_io._IMPLICIT_COMPACT)
            expected |= checkpoint_only
    if set(fields) != expected or any(not np.array_equal(fields[key], final[key])
            for key in expected - checkpoint_only -
                {'bank_payload_sha256', 'transport_payload_sha256'}):
        raise ValueError('GPU bank checkpoint differs from final accepted physical state')
    time_s, steps = float(fields['time']), int(fields['steps'])
    if (not math.isfinite(time_s) or time_s < 0. or steps < 0 or
            type(manifest.get('final_time_s')) not in (int, float) or
            not math.isclose(time_s, manifest['final_time_s'], rel_tol=0., abs_tol=1.e-12) or
            steps != manifest.get('steps') or steps != status.get('steps') or
            type(status.get('time_s')) not in (int, float) or
            not math.isclose(time_s, status['time_s'], rel_tol=0., abs_tol=1.e-12)):
        raise ValueError('GPU bank checkpoint accepted clock differs from run status')
    profile_result = {}
    if transport and context.get('execution_profile') is not None:
        profile_result.update(context['execution_profile']['identities'])
        profile_result['execution_profile_v2'] = \
            context['execution_profile']['record']
    return dict(checkpoint=checkpoint, time_s=time_s, steps=steps, status=manifest['status'],
        method=manifest['method'], input_identity=identity,
        source_identity=manifest['source_identity'], method_identity=manifest['method_identity'],
        **profile_result)


_CHECKPOINT = re.compile(r"accepted_([0-9]{9,})(?:_trial_([0-9]{9,})_([0-9a-f]{16}))?\Z")
_CHECKPOINT_FILES = frozenset(("checkpoint.json", "state.npz", "operators.json", "worker.json"))
_BANK_HISTORY = ("undercut", "reference_height", "toe_height", "reference_length",
                 "toe_cumulative", "collapse_cumulative")
_HISTORICAL_MODE3_BEDFORM_ADAPTERS = (
    "amr_recursive_route.py", "amr_catalog_bank.py",
    "amr_catalog_mixed_transition.py", "amr_bank_wetfront.py",
    "amr_prepared.py", "amr_on_demand.py", "amr_general_selector.py",
)
_HISTORICAL_MODE3_BEDFORM_IDENTITIES = {
    "live-selected-bank-mode3-bedform-avalanche-amr-v2": (
        "ef6740ead46c2fbde29f80b54fba6a4fc33dcf80cd7dcd072e2723bb3d422220",
        "21fb73b51c586bb21352eff4d3d38c6c7e583a237c4baa6ebb01d29f11dbb1b0",
    ),
    "live-selected-bank-mode3-bedform-avalanche-amr-v3": (
        "7eff401f72a979d079f126bc09f77339cbd1a9afdcb58d19fe9b3a22a11b9f23",
        "c8bc2096cd34747f3c4260bcdd271b65116b80868a7ac7e512c4fa89b985aeb0",
    ),
}


def _amr_require(condition, message):
    if not condition:
        raise ValueError("On-demand GPU checkpoint: " + message)


def _amr_regular(path, *, directory=False):
    path = filesystem_path(path)
    linked = path.is_symlink() or getattr(path, "is_junction", lambda: False)()
    _amr_require(not linked and (path.is_dir() if directory else path.is_file()),
                 "missing or linked " + str(path))
    return path


def _amr_number(value, name, *, integer=False, positive=False):
    _amr_require(type(value) is int if integer else type(value) in (int, float),
                 "invalid " + name)
    _amr_require(math.isfinite(value) and (value > 0 if positive else value >= 0),
                 "invalid " + name)
    return value


def _amr_checkpoint_schema(prepared):
    evolving = prepared['session_manifest']['schema'] == 2
    mode = prepared['session_manifest'].get('bank_mode', 2)
    _amr_require(type(mode) is int and mode in (2, 3) and (mode != 3 or evolving) and
                 prepared['initial_data'].get('bank_toe_erosion_mode', 2) == mode,
                 'prepared bank mode differs')
    temporal = prepared.get('amr_settings', {}).get('refine_sediment', False)
    _amr_require(type(temporal) is bool and (not temporal or evolving),
                 'temporal sediment indicator needs the evolving-bank route')
    recursive = prepared.get('amr_settings', {}).get('subcycling', False)
    _amr_require(type(recursive) is bool and
                 (not recursive or mode == 2 or (mode == 3 and evolving)),
                 'recursive checkpoint requires physical-toe mode 2 or explicit distributed banks')
    expected = ('recursive_bank_amr_v1' if recursive else
                4 if mode == 3 else (3 if temporal else (2 if evolving else 1)))
    declared = prepared.get('checkpoint_schema', expected)
    _amr_require(type(declared) is type(expected) and declared == expected,
                 'prepared temporal checkpoint contract differs')
    return expected


def _historical_mode3_bedform_checkpoint_receipt(
        receipt, expected, manifest, prepared):
    """Admit only the two immutable alpha91 pre-mode-2 result identities."""
    method = manifest.get("method")
    build = prepared.get("runtime_build")
    inventory = manifest.get("source_sha256")
    if (not isinstance(method, dict) or
            prepared.get("bank_mode") != 3 or
            not isinstance(build, dict) or
            build.get("kind") not in _HISTORICAL_MODE3_BEDFORM_IDENTITIES or
            method.get("bank_mode") != 3 or
            method.get("evolving_banks") is not True or
            not isinstance(inventory, dict)):
        return False
    combined = method.get("recursive_bank_bedform_avalanche")
    if (not isinstance(combined, dict) or
            combined.get("policy") !=
                "recursive_bedform_mode3_parent_avalanche_v1" or
            (manifest.get("source_identity"), manifest.get("method_identity")) !=
                _HISTORICAL_MODE3_BEDFORM_IDENTITIES[build["kind"]]):
        return False
    adapters = {}
    for name in _HISTORICAL_MODE3_BEDFORM_ADAPTERS:
        digest = inventory.get("python/braids_gpu/" + name)
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef"
                    for character in digest)):
            return False
        adapters[name] = digest
    historical = dict(expected, adapter_sources=adapters)
    return (type(receipt.get("schema")) is int and
            receipt == historical)


def _amr_controls(manifest, prepared):
    method = manifest["method"]
    evolving=prepared['session_manifest']['schema']==2
    checkpoint_schema = _amr_checkpoint_schema(prepared)
    recursive = checkpoint_schema == 'recursive_bank_amr_v1'
    if recursive:
        from .gpu_recursive_checkpoint import validate_adaptation_method
        validate_adaptation_method(method, prepared['initial_data'])
    fixed = dict(variant="on_demand_bank_amr_ssp33", integrator="SSPRK33",
        launch_mode="plain", precision="FP64", checkpoint_schema=checkpoint_schema, subcycling=recursive,
        session_identity=prepared["session_identity"], selector_contract=prepared["selector_contract"],
        cfl=prepared["cfl"], regrid_interval_s=prepared["amr_settings"]["regrid_interval_s"])
    if prepared.get('amr_settings', {}).get('refine_sediment', False):
        fixed['selector_reference'] = 'accepted_regrid_leaf_bed_v1'
    else:
        _amr_require('selector_reference' not in method,
                     'legacy checkpoint method carries temporal reference rule')
    if evolving:
        fixed['evolving_banks']=True
    else:
        _amr_require('evolving_banks' not in method,'Legacy method carries evolving-bank controls')
    if recursive:
        fixed['level_stepping'] = 'recursive_ssprk33_fine_replaced_register_v1'
        route_contract=method.get('recursive_bank_contract')
        if prepared['bank_mode'] == 2:
            fixed['recursive_bank_contract']='shared_recursive_bank_modes_v2'
        else:
            _amr_require(route_contract in (None,'shared_recursive_bank_modes_v2'),
                         'distributed recursive bank contract differs')
    else:
        _amr_require('level_stepping' not in method, 'Synchronous method carries recursive stepping')
        _amr_require('recursive_bank_contract' not in method,
                     'Synchronous method carries recursive bank contract')
    if checkpoint_schema == 4:
        fixed['bank_mode'] = 3
    elif recursive:
        fixed['bank_mode'] = prepared['bank_mode']
    else:
        _amr_require('bank_mode' not in method, 'Mode-2 method carries distributed-bank controls')
    live=np.array_equal(prepared['initial_data']['mode'],[9,4])
    if live or recursive:
        from .gpu_on_demand_prepared import recursive_boundary_event_identity
        fixed['boundary_forcing']=dict(schema=1,
            event_identity=recursive_boundary_event_identity(
                prepared['initial_data']),
            stage_fractions=[0.,1.,.5],clock='forcing_knot_clipped')
    else:
        _amr_require('boundary_forcing' not in method,'Unexpected live boundary clock')
    _amr_require(all(type(method.get(key)) is type(value) and method[key] == value
                     for key, value in fixed.items()), "method/prepared controls differ")
    dtmax = _amr_number(method.get("dtmax_s"), "maximum timestep", positive=True)
    interval = _amr_number(method.get("output_every_s"), "output interval", positive=True)
    start = _amr_number(manifest.get("start_s"), "start time")
    end = _amr_number(manifest.get("end_s"), "end time", positive=True)
    _amr_require(dtmax <= prepared["dtmax_s"] and manifest.get("dtmax_s") == dtmax and
                 manifest.get("output_every_s") == interval and start == prepared["time_s"] and
                 start < end <= prepared["end_s"], "run controls exceed or differ from prepared input")
    return dtmax, interval, start, end


def _amr_payload(fields, *, prepared, index, method_identity, worker, end):
    """Validate versioned bank payloads using only authenticated host data."""
    strings = ("catalog_identity", "method_identity", "topology_identity", "events_json")
    for name in strings:
        value = fields.get(name)
        _amr_require(isinstance(value, np.ndarray) and value.shape == () and value.dtype.kind == "U",
                     "invalid string " + name)
    _amr_require(str(fields["catalog_identity"]) == prepared["session_identity"] and
                 str(fields["method_identity"]) == method_identity, "payload session/method identity differs")
    candidates = [top for top in index["topologies"].values()
                  if top["operator_sha256"] == str(fields["topology_identity"])]
    _amr_require(len(candidates) == 1, "payload operator is outside the registered inventory")
    topology = candidates[0]
    data = topology["initial_data"]
    n, nn, ns, ng = len(topology["node_ids"]), len(index["tree"]["parent"]), data["nstate"], data["nfrac"]
    evolving=prepared['session_manifest']['schema']==2
    schema = _amr_checkpoint_schema(prepared)
    distributed = schema == 4
    _amr_require(data['bank_toe_erosion_mode'] == (3 if distributed else 2), 'payload native bank mode differs')
    toe_ids = (np.empty(0,np.int64) if evolving else np.asarray(sorted(row["segment_id"] for row in
                                prepared["session_manifest"]["global_toe_lineage"]), dtype=np.int64))
    face_count=int(data['nf']) if evolving else len(toe_ids)
    nb=int(data['nb'])
    shapes = dict(time_s=(), maximum_cfl=(), state=(n, ns), tree_state=(nn, ns),
        ledger_water=(nb,), ledger_bed=(ng, nb), ledger_secondary=(nb,), ledger_source=(),
        toe_signed=(ng, face_count), toe_gross=(ng, face_count))
    temporal = prepared.get('amr_settings', {}).get('refine_sediment', False)
    if temporal:
        shapes['selector_last_bed'] = (nn,)
    shapes.update({"history_" + key: (n,) for key in _BANK_HISTORY})
    integers = dict(schema=(), accepted_steps=(), euler_calls=(), node_ids=(n,),
        toe_segment_ids=(len(toe_ids),), history_collapse_count=(n,), delay_state=(nn,))
    extras=set()
    if distributed:
        del shapes['toe_signed'], shapes['toe_gross'], integers['toe_segment_ids']
        integers.update(bank_mode=(), distributed_stage_index=(), distributed_collapse_waves=(),
                        selector_reference_present=())
    elif evolving:
        from .gpu_frame_io import _scalar,_toe_map_schema
        del integers['toe_segment_ids']
        integers['face_epoch_count']=()
        count=int(_scalar(fields,'face_epoch_count','int64'))
        _amr_require(0<=count<=len(fields)//6,'invalid face epoch count')
        extras=set(_toe_map_schema(n,face_count))|{f'face_epoch_{i}_{key}' for i in range(count)
            for key in ('operator_identity','node_ids','old_face','fraction','signed','gross')}
    _amr_require(set(fields) == set(shapes) | set(integers) | set(strings) | extras, "payload fields differ")
    for name, shape in shapes.items():
        value = fields[name]
        _amr_require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(),
                     "invalid finite payload field " + name)
    for name, shape in integers.items():
        value = fields[name]
        _amr_require(value.shape == shape and value.dtype.kind in "iu" and np.all(value >= 0),
                     "invalid integer payload field " + name)
        if evolving:
            _amr_require(value.dtype==np.dtype('int32' if name=='history_collapse_count' else 'int64'),
                         'invalid versioned integer dtype '+name)
    _amr_require(int(fields["schema"]) == schema and
                 np.array_equal(fields["node_ids"], topology["node_ids"]) and
                 (evolving or np.array_equal(fields["toe_segment_ids"], toe_ids)),
                 "payload schema/global node identities differ")
    if distributed:
        _amr_require(int(fields['bank_mode']) == 3 and
                     int(fields['selector_reference_present']) == int(temporal),
                     'distributed-bank mode/temporal marker differs')
    steps, current = int(fields["accepted_steps"]), float(fields["time_s"])
    if distributed:
        _amr_require(int(fields['distributed_stage_index']) == (3 if steps else 0) and
                     int(fields['distributed_collapse_waves']) <= nn and
                     (steps > 0 or int(fields['distributed_collapse_waves']) == 0),
                     'distributed-bank accepted stage/wave counters differ')
    _amr_require(steps == worker["steps"] and current == worker["time_s"] and
                 int(fields["euler_calls"]) == 3 * steps and (steps == 0) == (current == 0) and
                 current <= end + 1e-12 and 0 <= float(fields["maximum_cfl"]) <= prepared["cfl"] + 1e-12,
                 "payload accepted clock/CFL differs")
    for name in ("state", "tree_state"):
        state = fields[name]
        _amr_require(np.all(state[:, 0] >= 0) and np.all(state[:, np.asarray(data["bed_indices"]).ravel()] >= 0) and
                     not np.any(state[:, np.asarray(data["suspended_indices"])]),
                     "invalid water/sediment inventory in " + name)
        dry = state[:, 0] == 0
        _amr_require(not np.any(state[dry, 1:3]) and (data['secondary_index'] < 0 or
                     not np.any(state[dry, data["secondary_index"]])),
                     "dry momentum/secondary inventory in " + name)
    _amr_require(all(np.all(fields["history_" + key] >= 0) for key in _BANK_HISTORY) and
                 (distributed or np.all(fields["toe_gross"] >= np.abs(fields["toe_signed"]) - 1e-12)),
                 "invalid bank/toe history")
    if evolving:
        from .gpu_on_demand_frame_io import validate_evolving_bank_fields
        translated={('bank_'+key if key.startswith('history_') else key):value
                    for key,value in fields.items()}
        validate_evolving_bank_fields(translated,topology,index)
    events = json.loads(str(fields["events_json"]))
    _amr_require(isinstance(events, list), "invalid mesh-event list")
    if evolving and not distributed:
        _amr_require(len(events)==count,'mesh event/face epoch count differs')
    by_name = index["topologies"]
    previous = next(iter(by_name))
    previous_step = -1
    previous_time=0.
    for event_index,event in enumerate(events):
        _amr_require(isinstance(event, dict) and event.get("old_topology") == previous and
                     isinstance(event.get("new_topology"), str) and event["new_topology"] in by_name,
                     "invalid mesh-event topology chain")
        step = _amr_number(event.get("step"), "mesh-event step", integer=True)
        time = _amr_number(event.get("time_s"), "mesh-event time", positive=not evolving)
        step_order=(previous_step<=step<=steps if evolving else previous_step<step<=steps)
        _amr_require(step_order and previous_time<=time<=current and
                     event.get("catalog_identity") == prepared["session_identity"] and
                     event.get("old_operator_identity") == by_name[previous]["operator_sha256"] and
                     event.get("new_operator_identity") == by_name[event["new_topology"]]["operator_sha256"],
                     "invalid mesh-event identity/clock chain")
        if evolving:
            _amr_require(event['new_topology']!=previous and (step==0)==(time==0) and (distributed or
                str(fields[f'face_epoch_{event_index}_operator_identity'])==by_name[previous]['operator_sha256']),
                'mesh event/retired face identity differs')
        previous, previous_step, previous_time = event["new_topology"], step,time
    _amr_require(previous == topology["name"], "mesh events do not reach selected topology")
    return topology


def _inspect_amr_resume(run_root, prepared_root):
    from .gpu_frame_io import _archive
    from .gpu_on_demand_frame_io import read_gpu_on_demand_run
    from .gpu_on_demand_prepared import inspect_gpu_on_demand_prepared

    root = _amr_regular(Path(run_root).expanduser(), directory=True).resolve()
    manifest = _object(_amr_regular(root / "run.json"))
    _amr_regular(root / "status.json")
    _amr_require(manifest.get("schema") == 3 and manifest.get("status") in _RESUMABLE,
                 "selected folder is not a resumable completed/stopped run")
    descriptor = manifest.get("checkpoint")
    _amr_require(isinstance(descriptor, dict) and set(descriptor) ==
                 {"path", "worker_sha256", "payload_sha256", "operators_sha256"}, "checkpoint descriptor differs")
    relative = descriptor.get("path")
    _amr_require(isinstance(relative, str) and relative.startswith("checkpoints/") and
                 relative.count("/") == 1, "unsafe checkpoint path")
    match = _CHECKPOINT.fullmatch(relative.split("/", 1)[1])
    _amr_require(match is not None, "unsafe checkpoint path")
    _amr_regular(root / "checkpoints", directory=True)
    checkpoint = _amr_regular(root / relative, directory=True)
    _amr_require(checkpoint.resolve().is_relative_to(root) and
                 {p.name for p in checkpoint.iterdir()} == _CHECKPOINT_FILES, "checkpoint file inventory differs")
    for name in _CHECKPOINT_FILES:
        _amr_regular(checkpoint / name)
    for name, key in (("worker.json", "worker_sha256"), ("state.npz", "payload_sha256"),
                       ("operators.json", "operators_sha256")):
        _amr_require(_sha256(checkpoint / name) == descriptor[key], "checkpoint digest differs: " + name)
    prepared = inspect_gpu_on_demand_prepared(prepared_root,
        expected_receipt_sha256=manifest.get("input_preparation_sha256"))
    # Existing readers authenticate the frozen source, every copied operator,
    # and all published frames. They never import the worker or execute a solver.
    run = read_gpu_on_demand_run(root, prepared_root=prepared_root)
    _amr_require(run["manifest"] == manifest, "run manifest changed during inspection")
    status, index = run["status"], run["prepared_index"]
    dtmax, interval, start, end = _amr_controls(manifest, prepared)
    worker = _object(checkpoint / "worker.json")
    _amr_require(set(worker) == {"schema", "method_identity", "input_identity", "source_identity", "time_s",
        "steps", "dt_hint", "rejected", "payload_sha256", "operators_sha256"} and
        type(worker["schema"]) is int and worker["schema"] == 1 and
        worker["method_identity"] == manifest["method_identity"] and
        worker["input_identity"] == prepared["session_identity"] and
        worker["source_identity"] == manifest["source_identity"] and
        all(worker[key] == descriptor[key] for key in ("payload_sha256", "operators_sha256")),
        "worker identities differ")
    current = _amr_number(worker["time_s"], "accepted time")
    steps = _amr_number(worker["steps"], "accepted steps", integer=True)
    rejected = _amr_number(worker["rejected"], "rejected trials", integer=True)
    hint = _amr_number(worker["dt_hint"], "timestep hint", positive=True)
    _amr_require(start <= current <= end + 1e-12 and hint <= dtmax and int(match[1]) == steps,
                 "worker accepted clock/timestep controls differ")
    if match[2] is not None:
        hint_digest = hashlib.sha256(json.dumps(hint, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode()).hexdigest()[:16]
        _amr_require(int(match[2]) == rejected and match[3] == hint_digest, "trial revision controls differ")
    for record in (manifest, status):
        _amr_require(_amr_number(record.get("time_s"), "published time") == current and
                     _amr_number(record.get("steps"), "published steps", integer=True) == steps and
                     _amr_number(record.get("rejected"), "published rejected trials", integer=True) == rejected,
                     "accepted clock/rejected trials differ from run status")
    _amr_require(status.get("end_s") == end and (manifest["status"] != "completed" or
                 math.isclose(current, end, rel_tol=0, abs_tol=1e-12)), "completed/end clock differs")
    receipt = _object(checkpoint / "checkpoint.json")
    recursive = _amr_checkpoint_schema(prepared) == 'recursive_bank_amr_v1'
    expected_receipt = dict(schema=manifest['method']['checkpoint_schema'], kind="coupled_bank_amr",
        payload_sha256=descriptor["payload_sha256"], catalog_identity=prepared["session_identity"],
        method_identity=manifest["method_identity"])
    if recursive:
        from .gpu_recursive_checkpoint import source_contract
        contract = source_contract(root, manifest, prepared['initial_data'])
        expected_receipt.update(schema=(2 if prepared['bank_mode'] == 2 else 1),
            kind='recursive_bank_amr',
            source_identity=contract['source_identity'], adapter_sources=contract['adapter_sources'])
    receipt_matches = (type(receipt.get("schema")) is int and
                       receipt == expected_receipt)
    if not receipt_matches and recursive:
        receipt_matches = _historical_mode3_bedform_checkpoint_receipt(
            receipt, expected_receipt, manifest, prepared)
    _amr_require(receipt_matches, "checkpoint receipt differs")
    inventory = _object(checkpoint / "operators.json")
    _amr_require(set(inventory) == {"schema", "session_identity", "checkpoint_sha256", "entries"} and
        type(inventory["schema"]) is int and inventory["schema"] == 1 and
        inventory["session_identity"] == prepared["session_identity"] and
        inventory["checkpoint_sha256"] == _sha256(checkpoint / "checkpoint.json") and
        isinstance(inventory["entries"], list) and bool(inventory["entries"]), "checkpoint operator receipt differs")
    known = {entry["package_identity"]: dict(selected_ids=entry["selected_ids"].tolist(),
        request_key=entry["request_key"], package_identity=entry["package_identity"], entry_sha256=entry["entry_sha256"])
        for entry in prepared["entries"]}
    selected = []
    for row in inventory["entries"]:
        _amr_require(isinstance(row, dict) and isinstance(row.get("package_identity"), str) and
                     isinstance(row.get("selected_ids"), list) and
                     all(type(value) is int for value in row["selected_ids"]) and
                     row == known.get(row["package_identity"]), "checkpoint operator is unknown or changed")
        selected.append(row["package_identity"])
    _amr_require(len(set(selected)) == len(selected) and
                 selected == [row["package_identity"] for row in manifest["operators"]] and
                 inventory["entries"][0]["selected_ids"] == prepared["session_manifest"]["initial_node_ids"],
                 "checkpoint operator order/initial selection differs")
    _amr_require(all(np.array_equal(value, prepared["initial_index"]["tree"][key])
                     for key, value in index["tree"].items()), "checkpoint hierarchy differs from prepared session")
    fields = _archive(checkpoint / "state.npz")
    if recursive:
        from .gpu_recursive_checkpoint import validate_payload
        topology, fields = validate_payload(fields, prepared=prepared, index=index,
            method_identity=manifest['method_identity'], worker=worker, end=end, contract=contract)
    else:
        topology = _amr_payload(fields, prepared=prepared, index=index,
                                method_identity=manifest["method_identity"], worker=worker, end=end)
    final = run["frames"][-1]
    secondary_index = int(topology['initial_data']['secondary_index'])
    q = fields['state'][:, :secondary_index] if secondary_index >= 0 else fields['state']
    secondary = fields['state'][:, secondary_index] if secondary_index >= 0 else np.zeros(len(q))
    _amr_require(final["time_s"] == current and final["steps"] == steps and
                 final["topology_name"] == topology["name"] and
                 np.array_equal(final["q"], q) and
                 np.array_equal(final["secondary"], secondary) and
                 all(np.array_equal(value, fields[key]) for key, value in final["ledger"].items()) and
                 all(np.array_equal(value, fields[key.removeprefix("bank_")])
                     for key, value in final["bank_history"].items()), "checkpoint differs from final published frame")
    if recursive:
        _amr_require(all(np.array_equal(value, fields['recursive'][key])
                         for key, value in final['recursive'].items()),
                     'checkpoint recursive accepted register/epoch differs from final frame')
    if _amr_checkpoint_schema(prepared) == 4 or (recursive and prepared['bank_mode'] == 3):
        _amr_require(final.get('bank_mode') == 3 and all(final[key] == int(fields[key]) for key in
                     ('distributed_stage_index', 'distributed_collapse_waves')),
                     'checkpoint distributed-bank diagnostics differ from final frame')
    elif prepared['session_manifest']['schema']==2:
        _amr_require(all(np.array_equal(value,fields['toe_map_'+key])
                         for key,value in final['toe_map'].items()) and
                     np.array_equal(final['toe_signed'],fields['toe_signed']) and
                     np.array_equal(final['toe_gross'],fields['toe_gross']) and
                     len(final['face_epochs'])==int(fields['face_epoch_count']) and
                     all(np.array_equal(value,fields[f'face_epoch_{i}_{key}'])
                         for i,epoch in enumerate(final['face_epochs']) for key,value in epoch.items()),
                     'checkpoint evolving bank state differs from final published frame')
    elif recursive:
        _amr_require(np.array_equal(final['toe_segment_ids'],fields['toe_segment_ids']) and
                     np.array_equal(final['toe_signed'],fields['toe_signed']) and
                     np.array_equal(final['toe_gross'],fields['toe_gross']),
                     'checkpoint static physical-toe diagnostics differ from final frame')
    # A cancelled rejected trial may update hint/count without creating another
    # accepted frame; its immutable revision is selected only by run.json.
    return dict(checkpoint=checkpoint, run_root=root, prepared_root=prepared["root"],
        time_s=current, steps=steps, rejected=rejected, dt_hint=hint, status=manifest["status"],
        method=manifest["method"], input_identity=prepared["receipt_sha256"],
        preparation_receipt_sha256=prepared["receipt_sha256"], session_identity=prepared["session_identity"],
        source_identity=manifest["source_identity"], method_identity=manifest["method_identity"],
        output_every_s=interval, dt_max=dtmax, dtmax_s=dtmax, start_s=start, end_s=end)


def inspect_gpu_on_demand_resume_checkpoint(run_root: str | Path, prepared_root: str | Path) -> dict:
    """Authenticate the schema-3 manifest-selected checkpoint, without execution.

    ``checkpoint`` is a directory, not a frame or an arbitrarily newest folder.
    For schema-1 caller compatibility ``input_identity`` is the preparation
    receipt SHA; ``session_identity`` is the numerical checkpoint/catalog ID.
    The external worker must still admit its executable/package and live inputs.
    """
    try:
        manifest = _object(_amr_regular(Path(run_root).expanduser(), directory=True) / 'run.json')
        if manifest.get('input_kind') == 'amr-on-demand-general':
            from .gpu_general_on_demand_io import inspect_resume
            return inspect_resume(run_root, prepared_root)
        return _inspect_amr_resume(run_root, prepared_root)
    except (OSError, KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError("Cannot inspect on-demand GPU checkpoint: " + str(exc)) from exc
