"""Authenticated, CPU-only reader for fixed-grid BRAIDS GPU worker output.

This module deliberately has no QGIS, GDAL, CuPy, or CUDA dependency.  It
does not infer a CRS or construct polygons: the prepared geometry's declared
CRS and verified cell metrics are returned for the caller to use explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from braids_core.windows_paths import filesystem_path


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FRAME = re.compile(r"frame_[0-9]{6}\.npz\Z")
_G1_VARIANT = "live_mode9_section_choice_ssp33"
_G1_FIELDS = frozenset({"compact_checkpoint_schema", "g1_payload_sha256"})
_BANK_VARIANT = "live_normal_critical_bank_ssp33"
from . import gpu_live_transport_io as _transport_io
_BANK_COMPACT = frozenset({"bank_checkpoint_schema", "bank_payload_sha256"})
_REQUIRED = frozenset({
    "q", "layers", "secondary", "bed_elevation", "ledger_water",
    "ledger_bed", "ledger_secondary", "ledger_source", "time", "steps",
    "rejected", "dt_hint", "input_identity", "source_identity",
    "method_identity",
})
_OPTIONAL = frozenset({
    "suspended", "ledger_suspended", "ledger_exchange",
    "event_identity", "next_boundary_knot", "bank_toe_face_signed", "bank_toe_face_gross",
    "bedload_adaptation_equilibrium", "bedload_adaptation_capacity",
    "bedload_adaptation_iterations", "bedload_adaptation_residual",
    "bedform_effective_manning", "bedform_dune_height", "bedform_dune_length",
    "bedform_total_ks", "bedform_iterations",
})
_BANK_FLOAT = frozenset({
    "bank_history_undercut", "bank_history_reference_height",
    "bank_history_toe_height", "bank_history_reference_length",
    "bank_history_toe_cumulative", "bank_history_collapse_cumulative",
})
_BANK_HISTORY = _BANK_FLOAT | {"bank_history_collapse_count"}
_ADAPTATION = frozenset({
    "bedload_adaptation_equilibrium", "bedload_adaptation_capacity",
    "bedload_adaptation_iterations", "bedload_adaptation_residual",
})


def _toe_map_schema(nc: int, nf: int) -> dict[str, tuple[tuple[int, ...], str]]:
    """CPU-only mirror of evolving_toe_map.checkpoint_specs (ABI 2).

    Keep the reader independent of the development/CUDA package; a host test
    checks this exact inventory against the numerical module's metadata.
    """
    result = {name: ((nc,), "int32") for name in (
        "channel", "bank", "is_toe_cell", "is_toe_donor", "mapped_toe",
        "mapped_donor", "mapped_face", "ambiguous", "support_donor",
        "support_toe", "support_face", "support_ambiguous", "support_root",
        "support_live")}
    result.update({name: ((nf,), "int32") for name in (
        "is_toe_face", "candidate_bank", "candidate_toe", "toe_faces",
        "donor_faces", "support_face_bank", "support_face_toe")})
    result.update({name: ((nc + 1,), "int32") for name in (
        "toe_offset", "donor_offset", "candidate_offset", "toe_donor_offset")})
    result.update({name: ((nc,), "int32") for name in ("candidates", "toe_donors")})
    result.update(distance=((nc,), "float64"), support_distance=((nc,), "float64"),
                  initialized=((), "int32"))
    return {"toe_map_" + name: spec for name, spec in result.items()}


class GPUFrameError(ValueError):
    """The worker output cannot be trusted or mapped to native cell state."""


def _fail(message: str):
    raise GPUFrameError(message)


def _json(path: Path) -> dict:
    path = filesystem_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GPUFrameError(f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{path.name} must contain a JSON object")
    return value


def _digest(path: Path) -> str:
    path = filesystem_path(path)
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        raise GPUFrameError(f"Cannot hash {path}: {exc}") from exc


def _identity(value, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        _fail(f"Invalid {label} SHA-256 identity")
    return value


def _number(value, label: str, *, integer=False, minimum=None):
    if integer:
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(f"Invalid {label}")
        result = int(value)
    else:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            _fail(f"Invalid {label}")
        result = float(value)
        if not math.isfinite(result):
            _fail(f"Nonfinite {label}")
    if minimum is not None and result < minimum:
        _fail(f"Invalid {label}")
    return result


def _same_time(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _array(archive, name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
    if name not in archive:
        _fail(f"Missing GPU array {name}")
    value = archive[name]
    if value.shape != shape or value.dtype != np.dtype(dtype):
        _fail(f"GPU array {name} has shape/dtype {value.shape}/{value.dtype}, expected {shape}/{np.dtype(dtype)}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        _fail(f"GPU array {name} contains nonfinite values")
    return value


def _scalar(archive, name: str, dtype):
    return _array(archive, name, (), dtype).item()


def _string_scalar(archive, name: str) -> str:
    if name not in archive:
        _fail(f"Missing GPU identity {name}")
    value = archive[name]
    if value.shape != () or value.dtype.kind != "U":
        _fail(f"GPU identity {name} must be a Unicode scalar")
    return str(value.item())


def _archive(path: Path) -> dict[str, np.ndarray]:
    path = filesystem_path(path)
    try:
        with np.load(path, allow_pickle=False) as source:
            if len(source.files) != len(set(source.files)):
                _fail(f"Duplicate NPZ member in {path.name}")
            return {key: source[key] for key in source.files}
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise GPUFrameError(f"Cannot decode {path.name}: {exc}") from exc


def _a1_schema(nc: int = 128, nfrac: int = 2, nb: int = 2,
               normal_faces: int = 8, outlet_faces: int = 8,
               event_count: int = 2) -> dict[str, tuple[tuple[int, ...], str]]:
    """A1 auxiliary layout; defaults retain the frozen 128-cell A1 contract."""
    f = "float64"
    i = "int32"
    result = {"a1_schedule_clock": ((2,), f),
              "a1_schedule_stage_time": ((4,), f),
              "a1_schedule_hydraulic": ((4, 2, nb), f),
              "a1_schedule_concentration": ((4, nfrac, nb), f),
              "a1_schedule_feed": ((4, nfrac, nb), f),
              "a1_schedule_status": ((4, nb), i),
              "a1_normal_bed": ((4, nc), f),
              "a1_normal_nodes": ((4, normal_faces + 1), f),
              "a1_normal_stage": ((4, 1), f),
              "a1_normal_discharge": ((4, 1), f),
              "a1_normal_trace": ((4, 3, normal_faces), f),
              "a1_normal_recorded": ((4,), i),
              "a1_outlet_bed": ((4, nc), f),
              "a1_outlet_q": ((4, 3, nc), f),
              "a1_outlet_trace": ((4, 3, outlet_faces), f),
              "a1_outlet_recorded": ((4,), i),
              "a1_normal_cache_valid": ((1,), i),
              "a1_normal_cache_bed": ((normal_faces,), f),
              "a1_normal_cache_selected_branch": ((1,), i),
              "a1_normal_cache_selected_kind": ((1,), i),
              "a1_normal_latch": ((1,), i),
              "a1_outlet_latch": ((1,), i),
              "a1_event_cursor": ((), "int64"),
              "a1_knot_hits": ((event_count,), i)}
    for key in ("root_status", "trace_status", "failure_latch", "cache_hit",
                "bisections", "metric_evaluations"):
        result["a1_normal_" + key] = ((4, 1), i)
    for key in ("invariant", "face_speed"):
        result["a1_outlet_" + key] = ((4, outlet_faces), f)
    for key in ("branch", "initial_branch", "removal_pass", "removal_reason", "face_status"):
        result["a1_outlet_" + key] = ((4, outlet_faces), i)
    for key in ("stage", "area", "top_width", "characteristic_Q", "critical_Q",
                "residual", "outward_discharge", "lower_residual", "upper_residual",
                "failed_speed"):
        result["a1_outlet_" + key] = ((4, 1), f)
    for key in ("status", "active_passes", "expansions", "bisections",
                "metric_evaluations", "failed_face", "failure_latch"):
        result["a1_outlet_" + key] = ((4, 1), i)
    for key in ("discharge", "stage", "area", "top_width", "conveyance", "residual"):
        result["a1_normal_cache_" + key] = ((1,), f)
    return result


def _graded_a1_schema(prepared: Path, receipt: dict, manifest: dict,
                      ncell: int) -> dict[str, tuple[tuple[int, ...], str]] | None:
    if manifest["method"].get("variant") != "live_normal_critical_graded_bedload_ssp33":
        return None
    if receipt.get("schema") != 3:
        _fail("Graded GPU frame needs the authenticated schema-3 preparation")
    from .gpu_prepared_io import read_gpu_prepared
    try:
        data = read_gpu_prepared(prepared,
            receipt_sha256=manifest["input_preparation_sha256"])
    except (OSError, ValueError) as exc:
        raise GPUFrameError(f"Cannot authenticate graded GPU preparation: {exc}") from exc
    offsets = np.asarray(data.get("boundary_offset"))
    event = manifest.get("boundary_events")
    knots = event.get("knots_s") if isinstance(event, dict) else None
    if (int(data["nc"]) != ncell or int(data["nfrac"]) != 7 or
            np.shape(offsets) != (3,) or offsets.dtype.kind not in "iu" or
            int(offsets[0]) != 0 or not (0 < offsets[1] < offsets[2]) or
            not isinstance(knots, list) or len(knots) < 3):
        _fail("Graded GPU frame layout differs from authenticated preparation")
    return _a1_schema(ncell, int(data["nfrac"]), int(data["nb"]),
        int(offsets[1] - offsets[0]), int(offsets[2] - offsets[1]), len(knots) - 2)


def _g1_preparation(prepared: Path, receipt: dict, manifest: dict, ncell: int) -> None:
    variant = manifest["method"].get("variant")
    if receipt.get("schema") == 4 and variant != _G1_VARIANT:
        _fail("Schema-4 inlet preparation requires the inlet GPU variant")
    if variant != _G1_VARIANT:
        return
    if (receipt.get("schema") != 4 or receipt.get("variant") != _G1_VARIANT or
            manifest.get("compact_checkpoint_schema") != 1 or
            manifest["method"].get("compact_checkpoint_schema") != 1 or
            manifest["method"].get("integrator") != "SSPRK33" or
            manifest["method"].get("launch_mode") != "plain" or
            manifest.get("suspended_enabled") is not False):
        _fail("GPU inlet run needs the schema-4 plain SSPRK33 preparation")
    from .gpu_prepared_io import _g1_numerical_identity, read_gpu_prepared
    try:
        data = read_gpu_prepared(prepared,
            receipt_sha256=manifest["input_preparation_sha256"])
        identity = _g1_numerical_identity(receipt)
    except (OSError, ValueError, KeyError) as exc:
        raise GPUFrameError(f"Cannot authenticate inlet GPU preparation: {exc}") from exc
    if (int(data["nc"]) != ncell or
            manifest.get("input_identity") != identity or
            manifest.get("fractions") != 2):
        _fail("GPU inlet run differs from its numerical preparation")
    event = manifest.get("boundary_events")
    if (not isinstance(event, dict) or
            _identity(event.get("identity"), "inlet event") !=
            manifest["method"].get("event_identity") or
            not isinstance(event.get("knots_s"), list)):
        _fail("GPU inlet run has no authenticated boundary event identity")


def _bank_preparation(prepared: Path, receipt: dict, manifest: dict,
                      ncell: int) -> dict | None:
    variant = manifest["method"].get("variant")
    if receipt.get("schema") == 5 and variant != _BANK_VARIANT:
        _fail("Schema-5 bank preparation requires the bank GPU variant")
    if variant != _BANK_VARIANT:
        return None
    method = manifest["method"]
    mode = receipt.get("bank_mode")
    checkpoint_schema = manifest.get("bank_checkpoint_schema")
    if (receipt.get("schema") != 5 or receipt.get("variant") != _BANK_VARIANT or
            type(mode) is not int or mode not in (2, 3) or
            manifest.get("bank_mode") != mode or method.get("bank_mode") != mode or
            type(checkpoint_schema) is not int or checkpoint_schema not in (1, 2) or
            type(method.get("bank_checkpoint_schema")) is not int or
            method.get("bank_checkpoint_schema") != checkpoint_schema or
            method.get("integrator") != "SSPRK33" or
            method.get("launch_mode") != "plain" or
            type(method.get("secondary_flow_model")) is not int or method['secondary_flow_model'] not in (0, 1, 2) or
            manifest.get("suspended_enabled") is not False):
        _fail("GPU bank run needs the schema-5 plain SSPRK33 preparation")
    if checkpoint_schema == 2:
        if (mode != 2 or
                method.get("physical_toe_mapping") != "geometry_connected_wet_v1" or
                method.get("toe_face_orientation") != "owner_to_neighbour"):
            _fail("GPU evolving bank map needs its schema-2 mapping/orientation contract")
    elif (method.get("physical_toe_mapping") == "geometry_connected_wet_v1" or
          method.get("toe_face_orientation") == "owner_to_neighbour"):
        _fail("GPU evolving bank map cannot use the static schema-1 contract")
    from .gpu_prepared_io import _bank_numerical_identity, read_gpu_prepared
    try:
        data = read_gpu_prepared(prepared,
            receipt_sha256=manifest["input_preparation_sha256"])
        identity = _bank_numerical_identity(receipt)
    except (OSError, ValueError, KeyError) as exc:
        raise GPUFrameError(f"Cannot authenticate bank GPU preparation: {exc}") from exc
    nc, ng, nl, nb = (int(data[key]) for key in
                      ("nc", "nfrac", "nbed_layers", "nb"))
    from .gpu_live_transport_io import validate_mpm_method
    try:
        validate_mpm_method(method,data)
    except ValueError as error:
        _fail(str(error))
    if (nc != ncell or manifest.get("input_identity") != identity or
            method.get("fractions") != ng or method.get("bed_layers") != nl or
            method['secondary_flow_model'] != data['secondary_flow_model']):
        _fail("GPU bank run differs from its numerical preparation")
    # Older schema-5 methods omitted these controls only while they were off.
    # New graded methods bind their recorded values to the actual native input.
    for name, default in (('hiding_exposure', 0), ('bed_slope', 0), ('bedload_adaptation', False)):
        expected = bool(data[name]) if type(default) is bool else int(data[name])
        value = method.get(name, default)
        if type(value) is not type(expected) or value != expected:
            _fail('GPU bank method differs from prepared '+name)
    length = method.get('bedload_adaptation_length_diameters')
    if (length is not None or bool(data['bedload_adaptation'])) and (
            type(length) is not float or length != float(data['bedload_adaptation_length_diameters'])):
        _fail('GPU bank adaptation length differs from preparation')
    viscosity = float(data['horizontal_viscosity'])
    if (type(manifest.get('diffusion_enabled',False)) is not bool or
            manifest.get('diffusion_enabled',False) != (viscosity > 0.)):
        _fail('GPU bank diffusion manifest differs from preparation')
    if viscosity > 0. or 'horizontal_viscosity' in method:
        if (type(method.get('horizontal_viscosity')) is not float or
                method['horizontal_viscosity'] != viscosity or
                any(type(method.get(key)) is not int or method[key] != data[key]
                    for key in ('horizontal_viscosity_mode','horizontal_viscosity_geometry'))):
            _fail('GPU bank viscosity method differs from preparation')
        if viscosity > 0. and method.get('diffusion_order') != 'after_bank_and_avalanche':
            _fail('GPU bank viscosity requires the authenticated post-bank/avalanche order')
    event = manifest.get("boundary_events")
    if (not isinstance(event, dict) or
            _identity(event.get("identity"), "bank event") !=
            manifest.get("event_identity") or
            event["identity"] != method.get("event_identity") or
            not isinstance(event.get("knots_s"), list)):
        _fail("GPU bank run has no authenticated boundary event identity")
    return {"mode": mode, "nfrac": ng, "nlayer": nl, "nb": nb,
            "secondary_model": int(data['secondary_flow_model']),
            "dry_tolerance": float(data['dry_tolerance']),
            "checkpoint_schema": checkpoint_schema,
            "owner": np.asarray(data["owner"]),
            "neighbour": np.asarray(data["neighbour"]),
            "toe_mask": (np.asarray(data["is_toe_face"], dtype=bool)
                         if mode == 2 else None)}


def _toe_map_frame(archive: dict, context: dict, nc: int, nf: int) -> None:
    """Check the exact persisted layout without rerunning physical mapping."""
    for name, (shape, dtype) in _toe_map_schema(nc, nf).items():
        _array(archive, name, shape, dtype)
    state = {name.removeprefix("toe_map_"): archive[name]
             for name in _toe_map_schema(nc, nf)}
    if int(state["initialized"]) != 1:
        _fail("GPU evolving toe map is not initialized")
    for name in ("channel", "bank", "is_toe_cell", "is_toe_donor", "is_toe_face",
                 "ambiguous", "support_ambiguous", "support_root", "support_live"):
        if np.any((state[name] != 0) & (state[name] != 1)):
            _fail(f"GPU evolving toe map {name} is not a binary flag")
    if np.any(state["bank"] & state["channel"]):
        _fail("GPU evolving toe map bank/channel roles overlap")
    for name in ("candidate_bank", "candidate_toe", "mapped_toe", "mapped_donor",
                 "support_toe", "support_donor", "support_face_bank", "support_face_toe"):
        if np.any((state[name] < -1) | (state[name] >= nc)):
            _fail(f"GPU evolving toe map {name} has an invalid cell ID")
    for name in ("mapped_face", "support_face"):
        if np.any((state[name] < -1) | (state[name] >= nf)):
            _fail(f"GPU evolving toe map {name} has an invalid face ID")
    for name in ("distance", "support_distance"):
        if np.any(state[name] < 0.):
            _fail(f"GPU evolving toe map {name} is negative")
    for offset, items, bound in (("toe_offset", "toe_faces", nf),
                                  ("donor_offset", "donor_faces", nf),
                                  ("candidate_offset", "candidates", nc),
                                  ("toe_donor_offset", "toe_donors", nc)):
        values = state[offset]
        if (values[0] != 0 or np.any((values < 0) | (values > bound)) or
                np.any(np.diff(values) < 0) or
                np.any((state[items] < 0) | (state[items] >= bound))):
            _fail(f"GPU evolving toe map {offset}/{items} is invalid")
    active = state["is_toe_face"] != 0
    bank, toe = state["candidate_bank"], state["candidate_toe"]
    owner, neighbour = context["owner"], context["neighbour"]
    if (owner.shape != (nf,) or neighbour.shape != (nf,) or
            np.any(bank[~active] != -1) or np.any(toe[~active] != -1) or
            np.any(bank[active] < 0) or np.any(toe[active] < 0) or
            np.any(neighbour[active] < 0)):
        _fail("GPU evolving toe map active face endpoints are invalid")
    endpoints = (((owner[active] == bank[active]) & (neighbour[active] == toe[active])) |
                 ((owner[active] == toe[active]) & (neighbour[active] == bank[active])))
    if (not endpoints.all() or np.any(state["bank"][bank[active]] != 1) or
            np.any(state["channel"][toe[active]] != 1)):
        _fail("GPU evolving toe map face roles differ from its cell topology")


def _source_identity(root: Path, manifest: dict) -> None:
    source = root / "source"
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        _fail("GPU source inventory is missing")
    if source.is_symlink() or not source.is_dir():
        _fail("GPU source snapshot is missing or linked")
    for name, expected in sources.items():
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            _fail("GPU source inventory has an unsafe filename")
        candidate = source
        for part in name.split("/"):
            candidate = candidate / part
            if candidate.is_symlink():
                _fail("GPU source inventory has a linked filename")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(source.resolve()):
            _fail("GPU source inventory has an unsafe filename")
        _identity(expected, f"source {name}")
        if _digest(candidate) != expected:
            _fail(f"GPU source {name} changed")
    runtime = source / "runtime_manifest.json"
    if runtime.is_file():
        source_manifest = _json(runtime)
        if source_manifest.get("source_files") != sources:
            _fail("GPU runtime source inventory differs from run.json")
    else:
        runtime = source / "manifest.json"
        source_manifest = _json(runtime)
        if source_manifest != sources:
            _fail("GPU source inventory differs from run.json")
    if _digest(runtime) != manifest["source_identity"]:
        _fail("GPU source identity differs from frozen manifest")


def _inline_geometry(prepared: Path, receipt: dict) -> dict:
    """Read the original QGIS polygons, authenticated independently of state."""
    from .gpu_prepared_io import _authenticated_member, read_gpu_prepared
    if (receipt.get('schema') not in (6,7) or
            receipt.get('variant') != _transport_io.VARIANT):
        _fail('JSON geometry is only admitted for ordinary schema6/7 inline input')
    _transport_io.validate_receipt(prepared,receipt)
    files = receipt['origin']['files']
    case = _json(prepared/'case.data.json')
    if case.get('sidecars',{}).get('geometry') != 'geometry.json':
        _fail('Inline QGIS input does not declare its original geometry.json')
    expected = _identity(files.get('geometry.json'),'inline geometry')
    _authenticated_member(prepared,'geometry.json',expected,nested=False)
    metadata = _json(prepared/'geometry.json')
    if (metadata.get('format') != 'braids-cut-cell-geometry' or
            type(metadata.get('version')) is not int or metadata['version'] != 1):
        _fail('Inline QGIS polygon geometry format differs')
    cells, faces = metadata.get('cells'), metadata.get('faces')
    if not isinstance(cells,list) or not cells or not isinstance(faces,list):
        _fail('Inline QGIS polygon geometry is incomplete')
    nc = len(cells)
    metrics = metadata.get('metrics',{})
    if metrics.get('cell_count') != nc or metrics.get('face_count') != len(faces):
        _fail('Inline QGIS polygon counts differ from metadata')
    area, xy = np.empty(nc,np.float64), np.empty((nc,2),np.float64)
    for index,cell in enumerate(cells):
        if not isinstance(cell,dict) or type(cell.get('cell_id')) is not int or cell['cell_id'] != index:
            _fail('Inline QGIS cell IDs are not sequential')
        area[index] = _number(cell.get('area_m2'),'inline cell area',minimum=1e-300)
        center = np.asarray(cell.get('centroid_xy'))
        if center.shape != (2,) or center.dtype.kind not in 'fiu' or not np.isfinite(center).all():
            _fail('Inline QGIS cell centroid is invalid')
        xy[index] = center
        holes = cell.get('holes_xy',[])
        if not isinstance(holes,list):
            _fail('Inline QGIS cell holes are invalid')
        for ring in [cell.get('vertices_xy'),*holes]:
            vertices = np.asarray(ring)
            if (vertices.ndim != 2 or vertices.shape[0] < 4 or vertices.shape[1] != 2 or
                    vertices.dtype.kind not in 'fiu' or not np.isfinite(vertices).all() or
                    not np.array_equal(vertices[0],vertices[-1])):
                _fail('Inline QGIS cell polygon ring is invalid')
    grid = metadata.get('grid')
    if not isinstance(grid,dict):
        _fail('Inline QGIS polygon geometry has no grid')
    for key in ('nx','ny'):
        _number(grid.get(key),'grid '+key,integer=True,minimum=1)
    for key in ('xlower_m','ylower_m','dx_m','dy_m'):
        _number(grid.get(key),'grid '+key,minimum=1e-300 if key.startswith('d') else None)
    crs = metadata.get('crs_wkt')
    if not isinstance(crs,str) or not crs:
        _fail('Inline QGIS polygon geometry does not declare a CRS')
    data = read_gpu_prepared(prepared)
    if (int(data['nc']) != nc or int(data['nf']) != len(faces) or
            not np.allclose(xy,np.asarray(data['xy']).T,rtol=0.,atol=1.e-9) or
            not np.allclose(area,data['area'],rtol=1.e-10,atol=1.e-12)):
        _fail('Inline QGIS cell order, centroid or area differs from native preparation')
    return dict(format=metadata['format'],sha256=expected,cell_count=nc,grid=grid,
        declared_crs_wkt=crs,cell_ids=np.arange(nc,dtype=np.int32),area_m2=area,
        centroid_xy=xy,metadata=metadata)


def _geometry(prepared: Path, receipt: dict) -> dict:
    prepared = filesystem_path(prepared)
    origin = receipt.get("origin")
    files = origin.get("files") if isinstance(origin, dict) else None
    if not isinstance(files, dict):
        _fail("Prepared receipt has no origin file inventory")
    if 'geometry.npz' not in files:
        return _inline_geometry(prepared,receipt)
    expected = _identity(files.get("geometry.npz"), "geometry")
    path = prepared / "geometry.npz"
    if _digest(path) != expected:
        _fail("Prepared geometry hash differs from receipt")
    archive = _archive(path)
    required = {
        "metadata_json", "cell_integer", "cell_scalar", "member_scale",
        "regular_cell", "vertex_offsets", "vertices_xy", "cell_hole_offsets",
        "hole_vertex_offsets", "hole_vertices_xy", "member_offsets",
        "source_member_indices",
    }
    if set(archive) != required:
        _fail("Prepared geometry array schema is unsupported")
    text = _string_scalar(archive, "metadata_json")
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GPUFrameError("Prepared geometry metadata JSON is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("format") != "braids-packed-cell-geometry-v1":
        _fail("Prepared geometry format is unsupported")
    nc = _number(metadata.get("cell_count"), "geometry cell count", integer=True, minimum=1)
    integer = _array(archive, "cell_integer", (nc, 6), "int32")
    scalar = _array(archive, "cell_scalar", (nc, 4), "float64")
    _array(archive, "member_scale", (nc,), "int8")
    _array(archive, "regular_cell", (nc,), "uint8")
    if not np.array_equal(integer[:, 0], np.arange(nc, dtype="int32")):
        _fail("Prepared cell IDs are not sequential")
    if np.any(scalar[:, 0] <= 0):
        _fail("Prepared cell areas must be positive")
    vertices = archive["vertices_xy"]
    if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.dtype != np.float64 or not np.isfinite(vertices).all():
        _fail("Prepared vertices are invalid")
    holes = archive["hole_vertices_xy"]
    if holes.ndim != 2 or holes.shape[1] != 2 or holes.dtype != np.float64 or not np.isfinite(holes).all():
        _fail("Prepared hole vertices are invalid")
    members = archive["source_member_indices"]
    if members.ndim != 2 or members.shape[1] != 3 or members.dtype != np.int32:
        _fail("Prepared source members are invalid")
    for name, length, final in (
        ("vertex_offsets", nc + 1, len(vertices)),
        ("cell_hole_offsets", nc + 1, len(archive["hole_vertex_offsets"]) - 1),
        ("hole_vertex_offsets", len(archive["hole_vertex_offsets"]), len(holes)),
        ("member_offsets", nc + 1, len(members)),
    ):
        offsets = _array(archive, name, (length,), "int64")
        if offsets[0] != 0 or offsets[-1] != final or np.any(np.diff(offsets) < 0):
            _fail(f"Prepared {name} is invalid")
    grid = metadata.get("grid")
    if not isinstance(grid, dict):
        _fail("Prepared geometry has no grid metadata")
    for key in ("nx", "ny"):
        _number(grid.get(key), f"grid {key}", integer=True, minimum=1)
    for key in ("xlower_m", "ylower_m", "dx_m", "dy_m"):
        _number(grid.get(key), f"grid {key}", minimum=1e-300 if key.startswith("d") else None)
    crs_wkt = metadata.get("crs_wkt")
    if not isinstance(crs_wkt, str) or not crs_wkt:
        _fail("Prepared geometry does not declare a CRS")
    return {"format": metadata["format"], "sha256": expected, "cell_count": nc,
            "grid": grid, "declared_crs_wkt": crs_wkt,
            "cell_ids": integer[:, 0].copy(), "area_m2": scalar[:, 0].copy(),
            "centroid_xy": scalar[:, 1:3].copy(), "metadata": metadata}


def _frame(path: Path, row: dict, manifest: dict, nc: int, nf: int,
           graded_schema: dict | None = None,
           bank_context: dict | None = None,
           transport_context: dict | None = None) -> dict:
    archive = _archive(path)
    names = set(archive)
    variant = manifest["method"].get("variant")
    is_g1 = variant == _G1_VARIANT
    is_bank = variant == _BANK_VARIANT
    is_transport = variant == _transport_io.VARIANT
    if is_bank and bank_context is None:
        _fail("GPU bank frame has no authenticated preparation context")
    bank_schema = bank_context.get("checkpoint_schema", 1) if is_bank else None
    map_schema = _toe_map_schema(nc, nf) if is_bank and bank_schema == 2 else {}
    is_a1 = variant in {"combined_live_scheduled_a1_ssp43",
                        "live_normal_critical_graded_bedload_ssp33"}
    a1_schema = (graded_schema if graded_schema is not None else
                 _a1_schema() if variant == "combined_live_scheduled_a1_ssp43" else {})
    if is_a1 and not a1_schema:
        _fail("GPU A1 frame has no authenticated auxiliary layout")
    if variant == "combined_live_scheduled_a1_ssp43" and (
            nc != 128 or archive.get("ledger_water", np.empty((0,))).shape != (2,)):
        _fail("GPU A1 frame is outside the frozen 128-cell/two-boundary scope")
    a1_names = set(a1_schema) | ({"a1_payload_sha256"} if is_a1 else set())
    unknown = (names - _REQUIRED - _OPTIONAL - _BANK_FLOAT -
               {"bank_history_collapse_count"} - a1_names -
               (_G1_FIELDS if is_g1 else frozenset()) -
               (_BANK_COMPACT if is_bank else frozenset()) - set(map_schema) -
               ((_transport_io.COMPACT | _transport_io.GAIA_DIAGNOSTICS)
                if is_transport else frozenset()))
    if unknown or not _REQUIRED.issubset(names):
        _fail(f"Unsupported GPU frame schema: missing={sorted(_REQUIRED - names)}, unknown={sorted(unknown)}")
    if is_a1 and not a1_names.issubset(names):
        _fail(f"GPU A1 auxiliary frame schema is incomplete: {sorted(a1_names - names)}")
    if is_g1 and not _G1_FIELDS.issubset(names):
        _fail("GPU inlet compact frame schema is incomplete")
    if is_g1 and names != (_REQUIRED | _G1_FIELDS |
                           {"event_identity", "next_boundary_knot"}):
        _fail("GPU inlet compact frame field inventory differs")
    if is_bank:
        expected_names = (_REQUIRED | _BANK_COMPACT | _BANK_HISTORY |
                          {"event_identity", "next_boundary_knot"})
        if bank_context["mode"] == 2:
            expected_names |= {"bank_toe_face_signed", "bank_toe_face_gross"}
        expected_names |= set(map_schema)
        if _ADAPTATION & names:
            if manifest["method"].get("bedload_adaptation") is not True:
                _fail("GPU bank adaptation diagnostics need the enabled method")
            expected_names |= _ADAPTATION
        if names != expected_names:
            _fail("GPU bank compact frame field inventory differs")
    q = _array(archive, "q", (3, nc), "float64")
    layers = archive["layers"]
    if layers.ndim != 3 or layers.dtype != np.float64 or layers.shape[2] != nc:
        _fail("GPU layers have invalid shape/dtype")
    nlayer, nclass, _ = layers.shape
    if nlayer < 1 or nclass < 1 or nlayer > 33 or not np.isfinite(layers).all():
        _fail("GPU layers contain invalid values or count")
    if is_bank and (nlayer != bank_context["nlayer"] or
                    nclass != bank_context["nfrac"]):
        _fail("GPU bank frame material layout differs from preparation")
    for key in ("secondary", "bed_elevation"):
        _array(archive, key, (nc,), "float64")
    nb = archive["ledger_water"].size
    if is_bank and nb != bank_context["nb"]:
        _fail("GPU bank frame boundary count differs from preparation")
    _array(archive, "ledger_water", (nb,), "float64")
    bed = _array(archive, "ledger_bed", (nclass, nb), "float64")
    _array(archive, "ledger_secondary", (nb,), "float64")
    _array(archive, "ledger_source", (), "float64")
    if is_bank:
        dry = q[0] <= bank_context.get('dry_tolerance', 0.)
        if (np.any(q[0] < 0.) or np.any(layers < 0.) or np.any(q[1:, dry]) or
                np.any(archive['secondary'][dry])):
            _fail('GPU bank frame contains negative inventory or dry momentum/secondary state')
        if bank_context.get('secondary_model', 2) != 2 and any(np.any(archive[name]) for name in
                ('secondary', 'ledger_secondary', 'ledger_source')):
            _fail('GPU bank nonprognostic secondary state and ledgers must be zero')
    has_suspended = "suspended" in archive
    if has_suspended != ("ledger_suspended" in archive) or has_suspended != ("ledger_exchange" in archive):
        _fail("GPU suspended state and ledgers must be present together")
    suspended = (_array(archive, "suspended", (nclass, nc), "float64") if has_suspended
                 else np.zeros((nclass, nc), dtype=np.float64))
    suspended_ledger = (_array(archive, "ledger_suspended", (nclass, nb), "float64") if has_suspended
                        else np.zeros((nclass, nb), dtype=np.float64))
    if has_suspended:
        _array(archive, "ledger_exchange", (nclass, 2), "float64")
    time = float(_scalar(archive, "time", "float64"))
    steps = int(_scalar(archive, "steps", "int64"))
    if time < 0 or steps < 0 or not _same_time(time, row["time_s"]) or steps != row["steps"]:
        _fail("GPU frame time/step disagrees with manifest")
    if _scalar(archive, "rejected", "int64") < 0 or _scalar(archive, "dt_hint", "float64") <= 0:
        _fail("GPU frame counters or timestep hint are invalid")
    for key, expected in (("input_identity", manifest["input_identity"] if (is_g1 or is_bank or is_transport)
                           else manifest["input_preparation_sha256"]),
                          ("source_identity", manifest["source_identity"]),
                          ("method_identity", manifest["method_identity"])):
        if _string_scalar(archive, key) != expected:
            _fail(f"GPU frame {key} differs from run identity")
    event = manifest.get("boundary_events")
    if ("event_identity" in archive) != (event is not None):
        _fail("GPU frame boundary event identity is inconsistent")
    if event is not None:
        if _string_scalar(archive, "event_identity") != event.get("identity"):
            _fail("GPU frame boundary event identity differs from run")
        _array(archive, "next_boundary_knot", (), "float64")
    elif "next_boundary_knot" in archive:
        _fail("Unexpected GPU boundary knot in frame")
    if is_g1:
        if ("suspended" in names or "ledger_suspended" in names or
                "ledger_exchange" in names or
                int(_scalar(archive, "compact_checkpoint_schema", "int32")) != 1):
            _fail("GPU inlet compact frame has an unsupported state schema")
        expected = _identity(_string_scalar(archive, "g1_payload_sha256"),
                             "inlet payload")
        digest = hashlib.sha256()
        for key in sorted(names - {"g1_payload_sha256"}):
            value = np.ascontiguousarray(archive[key])
            digest.update(key.encode() + b"\0" + value.dtype.str.encode() + b"\0")
            digest.update(json.dumps(value.shape).encode() + b"\0" + value.tobytes())
        if digest.hexdigest() != expected:
            _fail("GPU inlet payload identity differs from frame state")
    if is_bank:
        if (has_suspended or
                int(_scalar(archive, "bank_checkpoint_schema", "int32")) != bank_schema):
            _fail("GPU bank compact frame has an unsupported state schema")
        expected = _identity(_string_scalar(archive, "bank_payload_sha256"),
                             "bank payload")
        digest = hashlib.sha256()
        for key in sorted(names - {"bank_payload_sha256"}):
            value = np.ascontiguousarray(archive[key])
            digest.update(key.encode() + b"\0" + value.dtype.str.encode() + b"\0")
            digest.update(json.dumps(value.shape).encode() + b"\0" + value.tobytes())
        if digest.hexdigest() != expected:
            _fail("GPU bank payload identity differs from frame state")
    gaia_secondary = None
    if is_transport:
        gaia_secondary = _transport_io.validate_frame(
            archive, manifest, transport_context, _REQUIRED)
    for key, (shape, dtype) in a1_schema.items():
        if key == "a1_outlet_failed_speed":
            value = archive[key]
            if value.shape != shape or value.dtype != np.float64 or np.isinf(value).any():
                _fail("GPU A1 failed-speed sentinel has invalid shape/dtype/value")
        else:
            _array(archive, key, shape, dtype)
    if is_a1:
        if _scalar(archive, "a1_event_cursor", "int64") < 0:
            _fail("GPU A1 event cursor is invalid")
        expected = _identity(_string_scalar(archive, "a1_payload_sha256"), "A1 payload")
        digest = hashlib.sha256()
        for key in sorted(names - {"a1_payload_sha256"}):
            value = np.ascontiguousarray(archive[key])
            digest.update(key.encode() + b"\0")
            digest.update(value.dtype.str.encode() + b"\0")
            digest.update(json.dumps(value.shape).encode() + b"\0")
            digest.update(value.tobytes())
        if digest.hexdigest() != expected:
            _fail("GPU A1 payload identity differs from frame state")
    for key in _BANK_FLOAT & names:
        if np.any(_array(archive, key, (nc,), "float64") < 0.):
            _fail(f"GPU array {key} contains negative bank history")
    for key in ("bank_toe_face_signed", "bank_toe_face_gross"):
        if key in archive:
            _array(archive, key, (nclass, nf), "float64")
    if ("bank_toe_face_signed" in archive) != ("bank_toe_face_gross" in archive):
        _fail("GPU bank face diagnostic pair is incomplete")
    if is_bank and bank_context["mode"] == 2:
        signed = archive["bank_toe_face_signed"]
        gross = archive["bank_toe_face_gross"]
        # Schema 2 stores cumulative owner->neighbour transfers on fixed
        # topology. Retired internal faces retain their historical ledgers;
        # the current map is not a mask for erasing accepted past transfers.
        if bank_schema == 2:
            _toe_map_frame(archive, bank_context, nc, nf)
            mask = bank_context["neighbour"] >= 0
        else:
            mask = bank_context["toe_mask"]
        if (mask.shape != (nf,) or np.any(signed[:, ~mask]) or
                np.any(gross[:, ~mask]) or np.any(gross < 0) or
                np.any(gross + 1e-12 * np.maximum(1., np.maximum(np.abs(signed), gross))
                       < np.abs(signed))):
            _fail("GPU bank physical-toe frame flux differs from prepared map")
    for key in ("bedload_adaptation_equilibrium", "bedload_adaptation_capacity"):
        if key in archive:
            # Two signed Cartesian components, not nonnegative magnitudes.
            _array(archive, key, (2, nclass, nc), "float64")
    if ("bedload_adaptation_equilibrium" in archive) != ("bedload_adaptation_capacity" in archive):
        _fail("GPU bedload adaptation diagnostic pair is incomplete")
    adaptation_keys = {"bedload_adaptation_equilibrium", "bedload_adaptation_capacity",
                       "bedload_adaptation_iterations", "bedload_adaptation_residual"}
    if adaptation_keys & names and not adaptation_keys.issubset(names):
        _fail("GPU bedload adaptation diagnostic group is incomplete")
    if "bedload_adaptation_iterations" in archive:
        value = _scalar(archive, "bedload_adaptation_iterations", "int32")
        if is_bank and value < 0:
            _fail("GPU bank adaptation iteration count is negative")
    if "bedload_adaptation_residual" in archive:
        value = _scalar(archive, "bedload_adaptation_residual", "float64")
        if is_bank and value < 0.:
            _fail("GPU bank adaptation residual is negative")
    for key in ("bedform_effective_manning", "bedform_dune_height", "bedform_dune_length", "bedform_total_ks"):
        if key in archive:
            _array(archive, key, (nc,), "float64")
    if "bedform_iterations" in archive:
        _array(archive, "bedform_iterations", (nc,), "int32")
    bedform_keys = {"bedform_effective_manning", "bedform_dune_height",
                    "bedform_dune_length", "bedform_total_ks", "bedform_iterations"}
    if bedform_keys & names and not bedform_keys.issubset(names):
        _fail("GPU bedform diagnostic group is incomplete")
    if "bank_history_collapse_count" in archive:
        if np.any(_array(archive, "bank_history_collapse_count", (nc,), "int32") < 0):
            _fail("GPU bank-collapse count is negative")
    bank_keys = _BANK_FLOAT | {"bank_history_collapse_count"}
    if bank_keys & names and not bank_keys.issubset(names):
        _fail("GPU bank-history diagnostic group is incomplete")
    bank_history = ({key: archive[key].copy() for key in sorted(bank_keys)}
                    if bank_keys.issubset(names) else None)
    # Native state columns: h, hu, hv; suspended by class; then active and
    # each substrate layer by class.  Bed elevation remains a separate,
    # authenticated diagnostic for caller-side rock/bulk consistency checks.
    native = np.concatenate((q.T, suspended.T,
                             layers.transpose(2, 0, 1).reshape(nc, nlayer*nclass)), axis=1)
    ledger = np.concatenate((archive["ledger_water"][:, None], suspended_ledger.T, bed.T), axis=1)
    result = {"time_s": time, "steps": steps, "q": native, "ledger": ledger,
            "nfrac": nclass, "bed_layers": nlayer, "bed_elevation": archive["bed_elevation"].copy(),
            "secondary": archive["secondary"].copy(), "suspended_enabled": has_suspended,
            "bank_history": bank_history, "gaia_secondary_stress": gaia_secondary}
    if bedform_keys.issubset(names):
        result['bedform'] = {key: archive[key].copy() for key in sorted(bedform_keys)}
    return result


def read_gpu_run(run_root: str | Path, prepared_root: str | Path | None = None) -> dict:
    """Verify a fixed-grid worker run and return native-shaped per-cell frames.

    ``prepared_root`` permits a previously authenticated prepared input to be
    relocated.  The receipt bytes must still match ``run.json``.  A changed or
    missing source snapshot, geometry, method, or frame fails closed.
    """
    root = filesystem_path(run_root).resolve()
    manifest = _json(root / "run.json")
    status = _json(root / "status.json")
    if (manifest.get("schema") != 1 or manifest.get("backend") != "gpu"
            or manifest.get("backend_selected_by") != "user"
            or manifest.get("automatic_backend_selection") is not False):
        _fail("Run is not an explicit GPU schema-1 worker output")
    for key in ("input_preparation_sha256", "source_identity", "method_identity"):
        _identity(manifest.get(key), key)
    method = manifest.get("method")
    if not isinstance(method, dict) or method.get("source_identity") != manifest["source_identity"]:
        _fail("GPU method/source identity is inconsistent")
    digest = hashlib.sha256(json.dumps(method, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["method_identity"]:
        _fail("GPU method identity differs from method record")
    _source_identity(root, manifest)
    input_folder = manifest.get("input_folder")
    if not isinstance(input_folder, str) or not input_folder:
        _fail("GPU prepared-input folder is missing")
    prepared = filesystem_path(prepared_root if prepared_root is not None else input_folder).resolve()
    receipt_path = prepared / "preparation_receipt.json"
    if _digest(receipt_path) != manifest["input_preparation_sha256"]:
        _fail("GPU input preparation identity differs from receipt")
    receipt = _json(receipt_path)
    geometry = _geometry(prepared, receipt)
    if status.get("backend") != "gpu" or status.get("status") not in {
            "initializing", "running", "completed", "step_limit", "cancelled", "failed"}:
        _fail("GPU status record is inconsistent")
    if "status" in manifest and manifest["status"] != status["status"]:
        _fail("GPU run/status state differs")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        _fail("GPU run has no committed frames")
    ncell = geometry["cell_count"]
    nface = geometry["metadata"].get("metrics", {}).get("face_count")
    nface = _number(nface, "prepared face count", integer=True, minimum=0)
    if "cells" in manifest and _number(manifest["cells"], "GPU cell count", integer=True) != ncell:
        _fail("GPU cell count differs from prepared geometry")
    if "faces" in manifest:
        metrics = geometry["metadata"].get("metrics", {})
        if manifest["faces"] != metrics.get("face_count"):
            _fail("GPU face count differs from prepared geometry")
    graded_schema = _graded_a1_schema(prepared, receipt, manifest, ncell)
    _g1_preparation(prepared, receipt, manifest, ncell)
    bank_context = _bank_preparation(prepared, receipt, manifest, ncell)
    transport_context = _transport_io.preparation_context(prepared, receipt, manifest, ncell)
    converted = []
    previous_time = -math.inf
    previous_steps = -1
    for index, row in enumerate(frames):
        if not isinstance(row, dict):
            _fail("GPU frame manifest row is invalid")
        name = row.get("file")
        if not isinstance(name, str) or _FRAME.fullmatch(name) is None or name != f"frame_{index:06d}.npz":
            _fail("GPU frame filename is unsafe or out of order")
        _identity(row.get("sha256"), f"frame {index}")
        frame_time = _number(row.get("time_s"), f"frame {index} time", minimum=0)
        frame_steps = _number(row.get("steps"), f"frame {index} steps", integer=True, minimum=0)
        if index and (frame_time <= previous_time or frame_steps <= previous_steps):
            _fail("GPU frame times/steps are not strictly increasing")
        previous_time, previous_steps = frame_time, frame_steps
        path = root / name
        if _digest(path) != row["sha256"]:
            _fail(f"GPU frame {name} hash differs from manifest")
        item = _frame(path, row, manifest, ncell, nface, graded_schema,
                      bank_context, transport_context)
        if "fractions" in manifest and item["nfrac"] != manifest["fractions"]:
            _fail("GPU frame fraction count differs from run")
        if "suspended_enabled" in manifest and item["suspended_enabled"] != manifest["suspended_enabled"]:
            _fail("GPU suspended state differs from run")
        if converted and (item["nfrac"], item["bed_layers"], item["ledger"].shape[0]) != (
                converted[0]["nfrac"], converted[0]["bed_layers"], converted[0]["ledger"].shape[0]):
            _fail("GPU frame material/boundary shape changed during run")
        if converted and ((item["bank_history"] is None) != (converted[0]["bank_history"] is None)):
            _fail("GPU bank-history availability changed during run")
        converted.append(item)
    if manifest.get("status") in {"completed", "step_limit", "cancelled"}:
        listed = {row["file"] for row in frames}
        if {path.name for path in root.glob("frame_*.npz")} != listed:
            _fail("Final GPU output contains unlisted frames")
    status_time = _number(status.get("time_s"), "GPU status time", minimum=0)
    status_steps = _number(status.get("steps"), "GPU status steps", integer=True, minimum=0)
    if status_time + 1e-12 < previous_time or status_steps < previous_steps:
        _fail("GPU status predates last committed frame")
    if manifest.get("status") == "completed" and (
            not _same_time(status_time, previous_time) or status_steps != previous_steps):
        _fail("Completed GPU status differs from final frame")
    return {"manifest": manifest, "status": status, "geometry": geometry,
            "prepared_receipt": receipt, "frames": converted}


def read_gpu_live_frame(run_root: str | Path, prepared_root: str | Path | None = None,
                        *, after_index: int = -1) -> dict | None:
    """Read only the newest *committed* fixed-grid GPU frame after ``after_index``.

    The worker publishes a closed frame before atomically appending its digest
    to ``run.json``.  This reader never scans/decodes preceding frame payloads,
    so a QGIS polling task does not recompute the full run at each output time.
    Source, input, geometry, method and the selected frame are authenticated
    before any values are exposed to the rasterizer.  An unlisted NPZ is never
    treated as a live frame.  Absence of a committed new frame is ordinary.
    """
    if isinstance(after_index, bool) or not isinstance(after_index, int) or after_index < -1:
        _fail("Invalid GPU live-preview frame cursor")
    root = filesystem_path(run_root).resolve()
    if not (root / "run.json").is_file():
        return None
    manifest = _json(root / "run.json")
    if (manifest.get("schema") != 1 or manifest.get("backend") != "gpu"
            or manifest.get("backend_selected_by") != "user"
            or manifest.get("automatic_backend_selection") is not False):
        _fail("Run is not an explicit GPU schema-1 worker output")
    frames = manifest.get("frames")
    if not isinstance(frames, list):
        _fail("GPU frame inventory is invalid")
    if not frames or len(frames) - 1 <= after_index:
        return None
    for key in ("input_preparation_sha256", "source_identity", "method_identity"):
        _identity(manifest.get(key), key)
    method = manifest.get("method")
    if not isinstance(method, dict) or method.get("source_identity") != manifest["source_identity"]:
        _fail("GPU method/source identity is inconsistent")
    digest = hashlib.sha256(json.dumps(method, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["method_identity"]:
        _fail("GPU method identity differs from method record")
    latest = len(frames) - 1
    previous_time = -math.inf
    previous_steps = -1
    for index, row in enumerate(frames):
        if not isinstance(row, dict):
            _fail("GPU frame manifest row is invalid")
        name = row.get("file")
        if not isinstance(name, str) or _FRAME.fullmatch(name) is None or name != f"frame_{index:06d}.npz":
            _fail("GPU frame filename is unsafe or out of order")
        _identity(row.get("sha256"), f"frame {index}")
        time_s = _number(row.get("time_s"), f"frame {index} time", minimum=0)
        steps = _number(row.get("steps"), f"frame {index} steps", integer=True, minimum=0)
        if index and (time_s <= previous_time or steps <= previous_steps):
            _fail("GPU frame times/steps are not strictly increasing")
        previous_time, previous_steps = time_s, steps
    _source_identity(root, manifest)
    input_folder = manifest.get("input_folder")
    if not isinstance(input_folder, str) or not input_folder:
        _fail("GPU prepared-input folder is missing")
    prepared = filesystem_path(prepared_root if prepared_root is not None else input_folder).resolve()
    receipt_path = prepared / "preparation_receipt.json"
    if _digest(receipt_path) != manifest["input_preparation_sha256"]:
        _fail("GPU input preparation identity differs from receipt")
    receipt = _json(receipt_path)
    geometry = _geometry(prepared, receipt)
    ncell = geometry["cell_count"]
    nface = _number(geometry["metadata"].get("metrics", {}).get("face_count"),
                    "prepared face count", integer=True, minimum=0)
    if "cells" in manifest and _number(manifest["cells"], "GPU cell count", integer=True) != ncell:
        _fail("GPU cell count differs from prepared geometry")
    graded_schema = _graded_a1_schema(prepared, receipt, manifest, ncell)
    _g1_preparation(prepared, receipt, manifest, ncell)
    bank_context = _bank_preparation(prepared, receipt, manifest, ncell)
    transport_context = _transport_io.preparation_context(prepared, receipt, manifest, ncell)
    worker_geometry = root / "geometry.npz"
    try:
        with np.load(worker_geometry, allow_pickle=False) as worker:
            xy = _array(worker, "xy", (2, ncell), "float64").T
            area = _array(worker, "area", (ncell,), "float64")
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise GPUFrameError(f"Cannot decode GPU live geometry: {exc}") from exc
    if (not np.allclose(xy, geometry["centroid_xy"], rtol=0., atol=1e-9)
            or not np.allclose(area, geometry["area_m2"], rtol=1e-10, atol=1e-12)):
        _fail("GPU live geometry differs from authenticated preparation")
    row = frames[latest]
    path = root / row["file"]
    if _digest(path) != row["sha256"]:
        _fail(f"GPU frame {row['file']} hash differs from manifest")
    frame = _frame(path, row, manifest, ncell, nface, graded_schema,
                   bank_context, transport_context)
    return {"frame_index": latest, "frame": frame, "manifest": manifest,
            "geometry": geometry, "prepared_root": prepared}
