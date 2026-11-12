"""Host-only, topology-aware reader for a public GPU AMR result contract.

Schema 2 keeps a separate, authenticated result frame per accepted output
clock.  Each frame names one topology in a *single* AMR preparation package;
its global node IDs and operator identity must equal that package.  A complete
checkpoint may contain more fields, but must not be substituted for a frame.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from braids_core.windows_paths import filesystem_path

from .gpu_amr_prepared_io import inspect_gpu_amr_prepared, read_gpu_amr_topology
from .gpu_frame_io import (
    GPUFrameError, _archive, _digest, _identity, _json, _number,
    _source_identity, _string_scalar,
)


_FRAME = re.compile(r"frame_[0-9]{6}\.npz\Z")
_HISTORY_FLOAT = (
    "bank_history_undercut", "bank_history_reference_height",
    "bank_history_toe_height", "bank_history_reference_length",
    "bank_history_toe_cumulative", "bank_history_collapse_cumulative",
)
_HISTORY_COUNT = "bank_history_collapse_count"
_REQUIRED = frozenset({
    "frame_schema", "state", "node_ids", "topology_name", "regrid_mesh_index",
    "operator_identity", "time_s", "steps", "input_identity",
    "source_identity", "method_identity", "ledger_water", "ledger_bed",
    "ledger_secondary", "ledger_source",
})
_OPTIONAL = frozenset((*_HISTORY_FLOAT, _HISTORY_COUNT,
                       "bed_elevation", "rejected", "dt_hint"))


def _fail(message: str):
    raise GPUFrameError(message)


def _scalar(archive: dict, name: str, dtype):
    value = archive[name]
    if value.shape != () or value.dtype != np.dtype(dtype):
        _fail("GPU AMR frame has invalid scalar " + name)
    result = value.item()
    if np.issubdtype(value.dtype, np.floating) and not math.isfinite(result):
        _fail("GPU AMR frame has nonfinite scalar " + name)
    return result


def _array(archive: dict, name: str, shape: tuple[int, ...], dtype):
    value = archive[name]
    if value.shape != shape or value.dtype != np.dtype(dtype):
        _fail("GPU AMR frame has invalid shape/dtype for " + name)
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        _fail("GPU AMR frame has nonfinite " + name)
    return value


def _method_identity(manifest: dict):
    for key in ("input_preparation_sha256", "source_identity", "method_identity"):
        _identity(manifest.get(key), key)
    method = manifest.get("method")
    if not isinstance(method, dict) or method.get("source_identity") != manifest["source_identity"]:
        _fail("GPU AMR method/source identity is inconsistent")
    digest = hashlib.sha256(json.dumps(method, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["method_identity"]:
        _fail("GPU AMR method identity differs from method record")


def _frame(path: Path, row: dict, manifest: dict, topology: dict,
           topology_index: int) -> dict:
    archive = _archive(path)
    names = set(archive)
    if not _REQUIRED.issubset(names) or names - _REQUIRED - _OPTIONAL:
        _fail("Unsupported GPU AMR frame schema")
    if int(_scalar(archive, "frame_schema", "int64")) != 1:
        _fail("Unsupported GPU AMR frame version")
    name = topology["name"]
    if (_string_scalar(archive, "topology_name") != name or
            _string_scalar(archive, "operator_identity") != topology["operator_sha256"] or
            int(_scalar(archive, "regrid_mesh_index", "int64")) != topology_index):
        _fail("GPU AMR frame topology/operator differs from authenticated preparation")
    for key, expected in (("input_identity", manifest["input_preparation_sha256"]),
                          ("source_identity", manifest["source_identity"]),
                          ("method_identity", manifest["method_identity"])):
        if _string_scalar(archive, key) != expected:
            _fail("GPU AMR frame identity differs: " + key)
    nodes = topology["node_ids"]
    nc = len(nodes)
    ids = _array(archive, "node_ids", (nc,), "int64")
    if not np.array_equal(ids, nodes):
        _fail("GPU AMR frame global nodes differ from authenticated topology")
    initial = topology["initial_data"]
    nfrac = int(initial["nfrac"])
    nlayer = int(initial["nbed_layers"])
    nb = int(initial["nb"])
    base = 3 + nfrac + nlayer * nfrac
    if int(initial["nstate"]) != base + 1 or int(initial["secondary_index"]) != base:
        _fail("GPU AMR numerical state layout is unsupported")
    state = _array(archive, "state", (nc, base + 1), "float64")
    if np.any(state[:, 0] < -1e-10) or np.any(state[:, 3:base] < -1e-10):
        _fail("GPU AMR frame has negative water or material")
    _array(archive, "ledger_water", (nb,), "float64")
    _array(archive, "ledger_bed", (nfrac, nb), "float64")
    _array(archive, "ledger_secondary", (nb,), "float64")
    _scalar(archive, "ledger_source", "float64")
    clock = float(_scalar(archive, "time_s", "float64"))
    steps = int(_scalar(archive, "steps", "int64"))
    if (clock < 0 or steps < 0 or
            not math.isclose(clock, row["time_s"], rel_tol=1e-12, abs_tol=1e-12) or
            steps != row["steps"]):
        _fail("GPU AMR frame clock differs from run manifest")
    if "rejected" in archive and _scalar(archive, "rejected", "int64") < 0:
        _fail("GPU AMR frame has negative rejected-step count")
    if "dt_hint" in archive and _scalar(archive, "dt_hint", "float64") <= 0:
        _fail("GPU AMR frame has invalid timestep hint")
    present = names & (set(_HISTORY_FLOAT) | {_HISTORY_COUNT})
    if present and present != set(_HISTORY_FLOAT) | {_HISTORY_COUNT}:
        _fail("GPU AMR bank history is incomplete")
    bank_history = None
    if present:
        bank_history = {}
        for key in _HISTORY_FLOAT:
            value = _array(archive, key, (nc,), "float64")
            if np.any(value < 0):
                _fail("GPU AMR bank history is negative: " + key)
            bank_history[key] = value.copy()
        count = _array(archive, _HISTORY_COUNT, (nc,), "int32")
        if np.any(count < 0):
            _fail("GPU AMR bank-collapse count is negative")
        bank_history[_HISTORY_COUNT] = count.copy()
    rock = np.asarray(initial["rock"], dtype=np.float64)
    bulk = np.asarray(initial["bulk_fraction"], dtype=np.float64)
    if rock.shape != (nc,) or bulk.shape != (nfrac,) or np.any(bulk <= 0):
        _fail("GPU AMR preparation has invalid bed material geometry")
    layers = state[:, 3+nfrac:base].reshape(nc, nlayer, nfrac)
    bed = rock + (layers / bulk[None, None, :]).sum(axis=(1, 2))
    if "bed_elevation" in archive:
        reported = _array(archive, "bed_elevation", (nc,), "float64")
        if not np.allclose(reported, bed, rtol=0, atol=1e-8):
            _fail("GPU AMR bed elevation differs from conserved layers")
    return {"time_s": clock, "steps": steps, "mesh_id": topology_index,
            "topology_name": name, "node_ids": nodes.copy(),
            "q": state[:, :base].copy(), "secondary": state[:, base].copy(),
            "bed_elevation": bed, "nfrac": nfrac, "bed_layers": nlayer,
            "suspended_enabled": bool(int(initial["exchange_mode"]) != 0),
            "bank_history": bank_history,
            "ledger": {key: archive[key].copy() for key in
                       ("ledger_water", "ledger_bed", "ledger_secondary", "ledger_source")}}


def read_gpu_amr_run(run_root: str | Path, *,
                     prepared_root: str | Path | None = None) -> dict:
    """Authenticate a completed or partial schema-2 GPU AMR result sequence.

    This is an input/output contract reader, not a claim that any AMR route has
    passed physical qualification or is already selectable in QGIS.
    """
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    manifest = _json(root / "run.json")
    if (manifest.get("schema") == 3 and
            manifest.get("input_kind") in ("amr-on-demand", "amr-on-demand-general")):
        from .gpu_on_demand_frame_io import read_gpu_on_demand_run
        return read_gpu_on_demand_run(root, prepared_root=prepared_root)
    status = _json(root / "status.json")
    if (manifest.get("schema") != 2 or
            manifest.get("format") != "braids-gpu-amr-run-v1" or
            manifest.get("input_kind") != "amr-prepared" or
            manifest.get("backend") != "gpu" or
            manifest.get("backend_selected_by") != "user" or
            manifest.get("automatic_backend_selection") is not False):
        _fail("Run is not an explicitly selected GPU AMR schema-2 output")
    _method_identity(manifest)
    _source_identity(root, manifest)
    input_folder = manifest.get("input_folder")
    if not isinstance(input_folder, str) or not input_folder:
        _fail("GPU AMR prepared-input folder is absent")
    prepared = filesystem_path(Path(prepared_root if prepared_root is not None else input_folder).expanduser()).resolve()
    index = inspect_gpu_amr_prepared(prepared,
        expected_manifest_sha256=manifest["input_preparation_sha256"])
    names = list(index["topologies"])
    if (manifest.get("topologies") != names or
            status.get("backend") != "gpu" or
            status.get("status") not in {"initializing", "running", "completed",
                                         "step_limit", "cancelled", "failed"} or
            manifest.get("status") != status.get("status")):
        _fail("GPU AMR topology inventory or worker status differs")
    for name in names:
        index["topologies"][name] = read_gpu_amr_topology(index, name)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        _fail("GPU AMR run has no committed frames")
    converted = []
    previous_time = -math.inf
    previous_steps = -1
    for number, row in enumerate(frames):
        if not isinstance(row, dict):
            _fail("GPU AMR frame inventory row is invalid")
        filename = row.get("file")
        if (not isinstance(filename, str) or _FRAME.fullmatch(filename) is None or
                filename != f"frame_{number:06d}.npz"):
            _fail("GPU AMR frame filename is unsafe or out of order")
        _identity(row.get("sha256"), f"AMR frame {number}")
        clock = _number(row.get("time_s"), "AMR frame time", minimum=0)
        steps = _number(row.get("steps"), "AMR frame steps", integer=True, minimum=0)
        if number and (clock <= previous_time or steps <= previous_steps):
            _fail("GPU AMR frame clocks/steps are not increasing")
        previous_time, previous_steps = clock, steps
        topology_name = row.get("topology")
        if not isinstance(topology_name, str) or topology_name not in index["topologies"]:
            _fail("GPU AMR frame names an unknown topology")
        topology = index["topologies"][topology_name]
        mesh_id = names.index(topology_name)
        if (row.get("operator_identity") != topology["operator_sha256"] or
                row.get("cell_count") != len(topology["node_ids"])):
            _fail("GPU AMR frame topology binding differs")
        path = root / filename
        if _digest(path) != row["sha256"]:
            _fail("GPU AMR frame hash differs: " + filename)
        item = _frame(path, row, manifest, topology, mesh_id)
        if converted and (item["nfrac"], item["bed_layers"]) != (
                converted[0]["nfrac"], converted[0]["bed_layers"]):
            _fail("GPU AMR material state layout changed during run")
        converted.append(item)
    if manifest["status"] in {"completed", "step_limit", "cancelled"}:
        if {path.name for path in root.glob("frame_*.npz")} != {
                row["file"] for row in frames}:
            _fail("Final GPU AMR output contains unlisted frames")
    status_time = _number(status.get("time_s"), "GPU AMR status time", minimum=0)
    status_steps = _number(status.get("steps"), "GPU AMR status steps", integer=True, minimum=0)
    if status_time + 1e-12 < previous_time or status_steps < previous_steps:
        _fail("GPU AMR status predates last committed frame")
    if manifest["status"] == "completed" and (
            not math.isclose(status_time, previous_time, rel_tol=1e-12, abs_tol=1e-12)
            or status_steps != previous_steps):
        _fail("Completed GPU AMR status differs from final frame")
    return {"manifest": manifest, "status": status, "prepared_index": index,
            "frames": converted}


def read_gpu_amr_live_frame(run_root: str | Path, *,
                            prepared_root: str | Path | None = None,
                            after_index: int = -1) -> dict | None:
    """Read only the latest committed AMR frame for a live QGIS preview.

    All prior manifest rows are checked for ordering and topology identity,
    but their potentially large frame payloads are not reopened each poll.
    """
    if isinstance(after_index, bool) or not isinstance(after_index, int) or after_index < -1:
        _fail("Invalid GPU AMR live-preview cursor")
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    if not (root / "run.json").is_file():
        return None
    manifest = _json(root / "run.json")
    if (manifest.get("schema") == 3 and
            manifest.get("input_kind") in ("amr-on-demand", "amr-on-demand-general")):
        from .gpu_on_demand_frame_io import read_gpu_on_demand_live_frame
        return read_gpu_on_demand_live_frame(root, prepared_root=prepared_root,
                                             after_index=after_index)
    if (manifest.get("schema") != 2 or
            manifest.get("format") != "braids-gpu-amr-run-v1" or
            manifest.get("input_kind") != "amr-prepared" or
            manifest.get("backend") != "gpu" or
            manifest.get("backend_selected_by") != "user" or
            manifest.get("automatic_backend_selection") is not False):
        _fail("Run is not an explicitly selected GPU AMR schema-2 output")
    rows = manifest.get("frames")
    if not isinstance(rows, list):
        _fail("GPU AMR live frame inventory is invalid")
    latest = len(rows) - 1
    if latest <= after_index:
        return None
    _method_identity(manifest)
    _source_identity(root, manifest)
    folder = manifest.get("input_folder")
    if not isinstance(folder, str) or not folder:
        _fail("GPU AMR prepared-input folder is absent")
    prepared = filesystem_path(Path(prepared_root if prepared_root is not None else folder).expanduser()).resolve()
    index = inspect_gpu_amr_prepared(prepared,
        expected_manifest_sha256=manifest["input_preparation_sha256"])
    names = list(index["topologies"])
    if manifest.get("topologies") != names:
        _fail("GPU AMR live topology inventory differs")
    previous_time = -math.inf
    previous_steps = -1
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            _fail("GPU AMR live frame inventory row is invalid")
        filename = row.get("file")
        if (not isinstance(filename, str) or _FRAME.fullmatch(filename) is None or
                filename != f"frame_{number:06d}.npz"):
            _fail("GPU AMR live frame filename is unsafe or out of order")
        _identity(row.get("sha256"), f"AMR frame {number}")
        clock = _number(row.get("time_s"), "AMR live frame time", minimum=0)
        steps = _number(row.get("steps"), "AMR live frame steps", integer=True, minimum=0)
        if number and (clock <= previous_time or steps <= previous_steps):
            _fail("GPU AMR live frame clocks/steps are not increasing")
        previous_time, previous_steps = clock, steps
        name = row.get("topology")
        if not isinstance(name, str) or name not in index["topologies"]:
            _fail("GPU AMR live frame names an unknown topology")
        topology = index["topologies"][name]
        if (row.get("operator_identity") != topology["operator_sha256"] or
                row.get("cell_count") != len(topology["node_ids"])):
            _fail("GPU AMR live topology binding differs")
    row = rows[latest]
    name = row["topology"]
    topology = read_gpu_amr_topology(index, name)
    path = root / row["file"]
    if _digest(path) != row["sha256"]:
        _fail("GPU AMR live frame hash differs")
    frame = _frame(path, row, manifest, topology, names.index(name))
    return {"frame_index": latest, "frame": frame, "manifest": manifest,
            "topology": topology, "prepared_root": prepared}
