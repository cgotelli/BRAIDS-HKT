"""Read the GPU worker's initialized state without a validation-tree import.

Only the CPU initializer writes ``coupled_input.bin``.  The QGIS result viewer
uses these authenticated initial values to georeference and label GPU frames;
it never runs a numerical CPU step or changes the GPU execution backend.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
from braids_core.windows_paths import filesystem_path


MAGIC = b"BRDGPU01"
_REQUIRED = ("nc", "nfrac", "nbed_layers", "state", "rock",
             "manning_cell", "bulk_fraction", "dry_tolerance",
             "bedlaw", "exchange_mode")
_G1_VARIANT = "live_mode9_section_choice_ssp33"
_G1_EXPORTER_SHA256 = "a50f91dd99e5b48af94535761a7ec2f5b933e1e21ca9479370e78d03275eac33"
_G1_ARTIFACTS = frozenset(("coupled_input.bin", "coupled_input.npz", "normal_metadata.bin"))
_BANK_VARIANT = "live_normal_critical_bank_ssp33"
_BANK_EXPORTER_SHA256 = "f9dc86987a04328e778ce790bb89133aa7c97943b550b2b335508aa3a8195ab5"
_BANK_COMMON = frozenset(("coupled_input.bin", "coupled_input.npz",
                          "normal_metadata.bin", "avalanche_metadata.bin",
                          "bank_history_metadata.bin"))
_BANK_MODE_SIDECAR = {2: "bank_toe_metadata.bin", 3: "distributed_bank_metadata.bin"}
_BANK_HISTORY = ("undercut", "reference_height", "toe_height", "reference_length",
                 "toe_cumulative", "collapse_cumulative", "collapse_count")
_DIFFUSION_ARTIFACT = 'coupled_diffusion.bin'


def _bank_artifact_inventory(mode, names):
    if type(mode) is not int or mode not in _BANK_MODE_SIDECAR:
        return False
    base = _BANK_COMMON | {_BANK_MODE_SIDECAR[mode]}
    return set(names) in (base, base | {_DIFFUSION_ARTIFACT})


def _g1_numerical_identity(receipt: dict) -> str:
    """Match the worker's three-artifact identity, distinct from receipt SHA."""
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _G1_ARTIFACTS:
        raise ValueError("GPU inlet numerical artifact inventory differs")
    value = {name: artifacts[name] for name in sorted(_G1_ARTIFACTS)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _bank_numerical_identity(receipt: dict) -> str:
    """Match the worker's complete-artifact identity, not the receipt hash."""
    mode = receipt.get("bank_mode")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not _bank_artifact_inventory(mode, artifacts):
        raise ValueError("GPU bank numerical artifact inventory differs")
    return hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()


def _bank_npz_dtype(value) -> np.dtype:
    """Worker NumPy-2 scalar serialization, independent of host NumPy/OS.

    Native rank-zero records become Python scalars in `_read_records` and the
    worker writes them as int64/float64.  NumPy 1.x on 32-bit-int Windows would
    otherwise infer int32; actual native vector dtypes must remain unchanged.
    """
    if type(value) is int:
        return np.dtype("int64")
    if type(value) is float:
        return np.dtype("float64")
    return np.asarray(value).dtype


def _sha256(path: Path) -> str:
    with filesystem_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _authenticated_member(root: Path, name: str, digest: str, *, nested: bool) -> Path:
    """Resolve one receipt member without following a link or leaving the package."""
    root = filesystem_path(root)
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or
            not isinstance(digest, str) or len(digest) != 64 or
            any(ch not in "0123456789abcdef" for ch in digest)):
        raise ValueError("GPU preparation has an invalid member identity")
    member = PurePosixPath(name)
    if (member.is_absolute() or member.as_posix() != name or
            any(part in ("", ".", "..") for part in member.parts) or
            (not nested and len(member.parts) != 1)):
        raise ValueError("GPU preparation has an unsafe member path: " + name)
    path = root
    for part in member.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("GPU preparation has a linked member: " + name)
    if not path.is_file() or not path.resolve().is_relative_to(root) or _sha256(path) != digest:
        raise ValueError("GPU preparation member differs: " + name)
    return path


def _g1_records(root: Path, data: dict) -> None:
    """Check that the native sidecar and NPZ describe the same mode-9 export."""
    sidecar = _read_records(root / "normal_metadata.bin")
    for name, value in sidecar.items():
        if name in data and not np.array_equal(data[name], value):
            raise ValueError("GPU inlet duplicate native export differs: " + name)
    merged = {**data, **sidecar}
    control = np.asarray(merged.get("section_control"))
    if (int(data["nfrac"]) != 2 or int(data["nbed_layers"]) != 2 or
            int(data["bedlaw"]) != 2 or int(data["exchange_mode"]) != 0 or
            int(data.get("production_section_geometry", -1)) != 1 or
            not np.array_equal(merged.get("mode"), [9, 4]) or
            control.shape != (2,) or control.dtype != np.int32 or
            control[0] not in (0, 2) or control[1] != 0):
        raise ValueError("GPU inlet preparation has the wrong native mode-9 state")
    # The worker's snapshot reader adds these exact aliases before writing the
    # NPZ.  Checking every NPZ member prevents a rehashed, inconsistent export
    # from being accepted by the host viewer.
    derived = {
        "q": np.ascontiguousarray(data["state"][:3]),
        "dry": data["dry_tolerance"],
        "outlet_extension": data["cpu_outlet_extension"],
        "dt": data["dtmax"],
        "bedoffset": data["boundary_bed_offset"],
        "bc_hydraulic": np.ascontiguousarray(data["forcing"][:2]),
        "bc_bed_input": np.asarray(data["fraction_bed_input"]).copy(),
    }
    for native, canonical in (("native_section_seed_node", "section_seed_node"),
                              ("native_boundary_seed_position", "boundary_seed_position")):
        raw = np.asarray(merged[native])
        if raw.dtype != np.int32 or raw.shape != (2,) or np.any(raw < 0):
            raise ValueError("GPU inlet preparation has invalid native section seeds")
        derived[canonical] = np.ascontiguousarray(raw - 1)
    merged.update(derived)
    with np.load(root / "coupled_input.npz", allow_pickle=False) as archive:
        if (len(archive.files) != len(set(archive.files)) or
                set(archive.files) != set(merged) or
                any(not np.array_equal(archive[name], merged[name])
                    for name in archive.files)):
            raise ValueError("GPU inlet NPZ differs from native records")


def _bank_secondary_layout(data: dict, *, allow_gaia: bool = False) -> int:
    """Native fixed-grid layout: only model2 owns hI and its ledger.

    Model 3 is the state-free GAIA stress correction.  It is admitted only by
    the exact v43/law-9 transport caller; bank preparations retain their
    historical model-0/1/2 envelope.
    """
    model = data.get('secondary_flow_model')
    ng, nl = int(data['nfrac']), int(data['nbed_layers'])
    prognostic = model == 2
    permitted = (0, 1, 2, 3) if allow_gaia else (0, 1, 2)
    if (type(model) is not int or model not in permitted or
            data.get('secondary_index') != (3 + ng*(1+nl) if prognostic else -1) or
            data.get('secondary_ledger_index') != (1+2*ng if prognostic else -1) or
            data.get('nstate') != 3 + ng*(1+nl) + prognostic or
            data.get('nledger') != 1 + 2*ng + prognostic):
        raise ValueError('GPU bank preparation has an invalid secondary state/ledger layout')
    enabled, direction = data.get('secondary_flow'), data.get('secondary_flow_transport_direction')
    beta, espir = data.get('secondary_flow_momentum_beta'), data.get('secondary_flow_espir')
    if (type(enabled) is not int or enabled != int(model != 0) or
            type(direction) is not int or direction not in (0, 1) or
            any(type(value) not in (int, float) or not np.isfinite(value) or value < 0
                for value in (beta, espir)) or
            (model == 0 and (direction or beta)) or
            (model in (1, 3) and (not direction or beta)) or
            (prognostic and data.get('secondary_flow_inflow_mode') != 1)):
        raise ValueError('GPU bank preparation has inconsistent secondary-flow controls')
    return model


def _bank_records(root: Path, data: dict, mode: int) -> dict:
    """Merge every authenticated native sidecar and check the complete NPZ."""
    merged = dict(data)
    for filename in ("normal_metadata.bin", "avalanche_metadata.bin",
                     "bank_history_metadata.bin", _BANK_MODE_SIDECAR[mode]):
        for name, value in _read_records(root / filename).items():
            if name in merged and not np.array_equal(merged[name], value):
                raise ValueError("GPU bank duplicate native record differs: " + name)
            merged[name] = value
    diffusion = root/_DIFFUSION_ARTIFACT
    if diffusion.is_file() != (data.get('horizontal_viscosity',0.) > 0.):
        raise ValueError('GPU bank diffusion artifact does not match viscosity')
    if diffusion.is_file():
        for name,value in _read_records(diffusion).items():
            if 'diffusion_'+name in merged:
                raise ValueError('GPU bank duplicate diffusion record')
            merged['diffusion_'+name] = value
    nb, nc, ng, nl = (int(merged[key]) for key in
                      ("nb", "nc", "nfrac", "nbed_layers"))
    if (mode != int(merged["bank_toe_erosion_mode"]) or
            nb < 1 or nc <= 0 or not 1 <= ng <= 32 or not 2 <= nl <= 33 or
            int(merged["bedlaw"]) != 2 or int(merged["exchange_mode"]) != 0 or
            int(merged["bank_failure"]) != 1):
        raise ValueError("GPU bank preparation has the wrong native live-bank state")
    from .gpu_live_transport_io import validate_boundary_scope, validate_diffusion, validate_mpm_controls
    validate_boundary_scope(merged)
    validate_diffusion(merged)
    validate_mpm_controls(merged)
    _bank_secondary_layout(merged)
    hiding = merged.get('hiding_exposure')
    if (type(hiding) is not int or hiding not in (0, 2, 3, 4, 5) or (hiding and ng < 2) or
            merged.get('active_layer_mode') != 0 or
            not np.isfinite(merged.get('active_layer_thickness', np.nan)) or
            merged['active_layer_thickness'] <= 0 or
            not all(np.isfinite(merged.get(key, np.nan)) for key in ('hiding_parker_alpha', 'hiding_wu_exponent'))):
        raise ValueError('GPU bank preparation has unsupported material or viscosity controls')
    derived = {
        "q": np.ascontiguousarray(data["state"][:3]),
        "dry": data["dry_tolerance"],
        "outlet_extension": data["cpu_outlet_extension"],
        "dt": data["dtmax"],
        "bedoffset": data["boundary_bed_offset"],
        "bc_hydraulic": np.ascontiguousarray(data["forcing"][:2]),
        "bc_bed_input": np.asarray(data["fraction_bed_input"]).copy(),
    }
    for native, canonical in (("native_section_seed_node", "section_seed_node"),
                              ("native_boundary_seed_position", "boundary_seed_position")):
        value = np.asarray(merged[native])
        if value.shape != (nb,) or value.dtype != np.int32 or np.any(value < 0):
            raise ValueError("GPU bank preparation has invalid native section seeds")
        derived[canonical] = np.ascontiguousarray(value - 1)
    merged.update(derived)
    for name in _BANK_HISTORY:
        value = np.asarray(merged["bank_" + name])
        expected = np.int32 if name == "collapse_count" else np.float64
        if (value.shape != (nc,) or value.dtype != expected or
                not np.isfinite(value).all() or np.any(value < 0)):
            raise ValueError("GPU bank preparation has invalid native history: " + name)
    if mode == 2:
        toe = np.asarray(merged.get("is_toe_face"))
        if (toe.shape != (int(merged["nf"]),) or toe.dtype != np.int32 or
                not np.isin(toe, (0, 1)).all() or not np.any(toe)):
            raise ValueError("GPU bank preparation has invalid physical-toe map")
    with np.load(root / "coupled_input.npz", allow_pickle=False) as archive:
        if (len(archive.files) != len(set(archive.files)) or
                set(archive.files) != set(merged) or
                any(archive[name].dtype != _bank_npz_dtype(merged[name]) or
                    not np.array_equal(archive[name], merged[name])
                    for name in archive.files)):
            raise ValueError("GPU bank NPZ differs from native records")
    return merged


def _read_records(path: Path) -> dict[str, np.ndarray | int | float]:
    path = filesystem_path(path)
    fields = {}
    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError("Invalid GPU initializer record magic")
        version = np.fromfile(stream, dtype="<i4", count=1)
        if version.size != 1 or int(version[0]) != 1:
            raise ValueError("Unsupported GPU initializer record version")
        while True:
            raw_name = stream.read(64)
            header = np.fromfile(stream, dtype="<i4", count=6)
            if len(raw_name) != 64 or header.size != 6:
                raise ValueError("Truncated GPU initializer record header")
            try:
                name = raw_name.decode("ascii").strip()
            except UnicodeError as exc:
                raise ValueError("Non-ASCII GPU initializer record name") from exc
            kind, rank, *extent = map(int, header)
            if name == "END":
                if kind != 0 or rank != 0 or stream.read(1):
                    raise ValueError("Invalid GPU initializer record terminator")
                break
            if (not name or name in fields or kind not in (1, 2) or
                    not 0 <= rank <= 4 or any(value < 0 for value in extent[:rank])):
                raise ValueError("Invalid GPU initializer record header")
            shape = tuple(extent[:rank])
            count = int(np.prod(shape)) if rank else 1
            if count > 100_000_000:
                raise ValueError("Implausibly large GPU initializer record")
            values = np.fromfile(stream, dtype="<i4" if kind == 1 else "<f8",
                                 count=count)
            if values.size != count or not np.isfinite(values).all():
                raise ValueError("Invalid GPU initializer record values: " + name)
            fields[name] = (values[0].item() if not rank else
                            np.ascontiguousarray(values.reshape(shape, order="F")))
    return fields


def read_gpu_prepared(folder: str | Path, *, receipt_sha256: str | None = None) -> dict:
    """Return the initial GPU state after verifying the preparation receipt."""
    root = filesystem_path(Path(folder).expanduser()).resolve()
    receipt_path = root / "preparation_receipt.json"
    if receipt_sha256 is not None and _sha256(receipt_path) != receipt_sha256:
        raise ValueError("GPU preparation receipt differs from run identity")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    schema = receipt.get("schema")
    if type(schema) is not int or schema not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("Unsupported GPU preparation receipt schema")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or "coupled_input.bin" not in artifacts:
        raise ValueError("GPU preparation lacks authenticated coupled input")
    if schema == 3:
        if (receipt.get("variant") != "live_normal_critical_graded_bedload_ssp33" or
                set(artifacts) != {"coupled_input.bin", "coupled_input.npz",
                                   "normal_metadata.bin"} or
                type(receipt.get("native_solver_steps_performed")) is not int or
                receipt["native_solver_steps_performed"] != 0 or
                receipt.get("GPU_solver_executed") is not False or
                receipt.get("initialization_CPU_only") is not True):
            raise ValueError("GPU preparation schema 3 is not a zero-step graded-bedload export")
        origin = receipt.get("origin")
        files = origin.get("files") if isinstance(origin, dict) else None
        if not isinstance(files, dict) or not {"case.data", "geometry.npz"} <= set(files):
            raise ValueError("GPU preparation schema 3 lacks copied deck/geometry provenance")
        for name, digest in files.items():
            candidate = root / name
            path = candidate.resolve()
            if (not isinstance(name, str) or not name or
                    not path.is_relative_to(root) or candidate.is_symlink() or
                    not path.is_file() or not isinstance(digest, str) or
                    len(digest) != 64 or _sha256(path) != digest):
                raise ValueError("GPU preparation origin file differs: " + str(name))
    if schema == 4:
        if (receipt_path.is_symlink() or
                receipt.get("variant") != _G1_VARIANT or
                set(artifacts) != _G1_ARTIFACTS or
                receipt.get("exporter_executable_sha256") != _G1_EXPORTER_SHA256 or
                receipt.get("initialization_CPU_only") is not True or
                type(receipt.get("native_solver_steps_performed")) is not int or
                receipt["native_solver_steps_performed"] != 0 or
                receipt.get("GPU_solver_executed") is not False):
            raise ValueError("GPU preparation schema 4 is not a pinned zero-step inlet export")
        origin = receipt.get("origin")
        files = origin.get("files") if isinstance(origin, dict) else None
        if (not isinstance(files, dict) or
                not {"case.data", "case.data.bin", "geometry.npz"} <= set(files)):
            raise ValueError("GPU preparation schema 4 lacks copied deck/geometry provenance")
        for name, digest in files.items():
            _authenticated_member(root, name, digest, nested=True)
        for name in ("scheduled_metadata.bin", "coupled_diffusion.bin",
                     "avalanche_metadata.bin", "bank_toe_metadata.bin",
                     "distributed_bank_metadata.bin", "bank_history_metadata.bin"):
            if (root / name).exists():
                raise ValueError("GPU inlet preparation has an excluded sidecar: " + name)
    if schema == 5:
        mode = receipt.get("bank_mode")
        if (receipt_path.is_symlink() or
                receipt.get("variant") != _BANK_VARIANT or
                not _bank_artifact_inventory(mode, artifacts) or
                receipt.get("exporter_executable_sha256") != _BANK_EXPORTER_SHA256 or
                receipt.get("initialization_CPU_only") is not True or
                type(receipt.get("native_solver_steps_performed")) is not int or
                receipt["native_solver_steps_performed"] != 0 or
                receipt.get("GPU_solver_executed") is not False):
            raise ValueError("GPU preparation schema 5 is not a pinned zero-step bank export")
        origin = receipt.get("origin")
        files = origin.get("files") if isinstance(origin, dict) else None
        if (not isinstance(files, dict) or
                not {"case.data", "geometry.npz"} <= set(files)):
            raise ValueError("GPU preparation schema 5 lacks copied deck/geometry provenance")
        for name, digest in files.items():
            _authenticated_member(root, name, digest, nested=True)
        for line in (root / "case.data").read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if len(tokens) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", tokens[1]):
                if files.get(tokens[0]) != tokens[1].lower():
                    raise ValueError("GPU bank native deck dependency is not authenticated")
        for name in ("scheduled_metadata.bin",
                     _BANK_MODE_SIDECAR[3 if mode == 2 else 2]):
            if (root / name).exists():
                raise ValueError("GPU bank preparation has an excluded sidecar: " + name)
        expected_bins = set(artifacts) - {'coupled_input.npz'}
        if {p.name for p in root.glob('*.bin') if p.name != 'case.data.bin'} != expected_bins:
            raise ValueError('GPU bank preparation native sidecar inventory differs')
    if schema in (6, 7):
        from .gpu_live_transport_io import validate_receipt
        validate_receipt(root, receipt)
    for name, digest in artifacts.items():
        if schema in (4, 5, 6, 7):
            _authenticated_member(root, name, digest, nested=False)
            continue
        candidate = root / name
        path = candidate.resolve()
        if (path.parent != root or candidate.is_symlink() or
                not isinstance(digest, str) or len(digest) != 64 or
                _sha256(path) != digest):
            raise ValueError("GPU preparation artifact differs: " + str(name))
    data = _read_records(root / "coupled_input.bin")
    if not set(_REQUIRED) <= data.keys():
        raise ValueError("GPU preparation is missing viewer fields")
    nc, ng, nl = (int(data[key]) for key in ("nc", "nfrac", "nbed_layers"))
    if not (nc > 0 and ng > 0 and nl > 0):
        raise ValueError("GPU preparation has invalid state dimensions")
    if schema == 3 and (ng != 7 or nl != 4 or
                        int(data["bedlaw"]) != 2 or
                        int(data["exchange_mode"]) != 0):
        raise ValueError("GPU preparation schema 3 has the wrong graded-bedload state")
    if schema == 4:
        _g1_records(root, data)
    if schema == 5:
        data = _bank_records(root, data, mode)
    if schema in (6, 7):
        from .gpu_live_transport_io import read_records
        data = read_records(root, data, receipt)
    base_state = 3 + (1 + nl) * ng
    secondary_index = int(data.get("secondary_index", -1))
    if secondary_index not in (-1, base_state):
        raise ValueError("GPU preparation has unexpected secondary state layout")
    expected_state = (base_state + (secondary_index >= 0), nc)
    expected_shapes = {"state": expected_state, "rock": (nc,),
                       "manning_cell": (nc,), "bulk_fraction": (ng,)}
    for key, shape in expected_shapes.items():
        if np.shape(data[key]) != shape:
            raise ValueError("GPU preparation has invalid " + key + " shape")
    if (np.any(np.asarray(data["bulk_fraction"]) <= 0) or
            float(data["dry_tolerance"]) < 0):
        raise ValueError("GPU preparation has invalid material or dry threshold")
    return data
