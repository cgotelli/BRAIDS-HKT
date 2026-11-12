"""Public, fail-closed CPU/QGIS envelope for Q3-H4b bed-slope transport.

The existing Q3-C writer remains the authority for the base capacity deck.
This module accepts only its authenticated slope-off v54 output, changes the
single slope record to the explicit ten-field H4 ABI, and binds every changed
byte to a version-62 sidecar before exposing a launch specification.

Keeping this route additive preserves frozen v54--v61 meanings and the frozen
native cut-cell source.  The executable is the derived adapter built by the
H4b builder; it is copied into each prepared run and re-hashed immediately
before launch.  Direct invocation without this verifier is not a v62 run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from collections.abc import Mapping, Sequence

import numpy as np

from braids_core.sediment.bed_slope_live_v62 import (
    PREPARED_EXECUTION_KEY,
    PROJECT_BINDING_KEY,
    RESTART_IDENTITY_KEY,
    RESULT_IDENTITY_KEY,
    canonical_bytes,
    component_document,
    lifecycle_identity_document,
    prepared_execution_document,
    project_binding_document,
    stage_fingerprint,
    validate_lifecycle_identity,
    validate_prepared_execution,
    validate_project_binding,
)
from braids_core.sediment.capacity_foundation_execution import (
    load_capacity_foundation_execution,
)
from braids_core.sediment.channel_composition import (
    load_channel_composition_profile,
)


PROJECT_SIDECAR = "bed_slope_cpu_v62_project.json"
PREPARED_SIDECAR = "bed_slope_cpu_v62_prepared.json"
RESULT_SIDECAR = "bed_slope_cpu_v62_result.json"
RESTART_SIDECAR = "bed_slope_cpu_v62_restart.json"
BASE_DECK = "case.v54.slope_off.data"
BASE_METADATA = "case.v54.slope_off.data.json"
LIVE_DECK = "case.data"
LIVE_METADATA = "case.data.json"
CAPACITY_SIDECAR = "capacity_execution.json"
ADAPTER = "xbraids_q3h4b_cpu_v62.exe" if os.name == "nt" else \
    "xbraids_q3h4b_cpu_v62"
ADAPTER_RECEIPT = "bed_slope_cpu_v62_adapter_receipt.json"
ADAPTER_SOURCE = "cutcell_solver_h4b_v62.f90"


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Q3-H4b file is missing or linked: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid Q3-H4b JSON file: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Q3-H4b JSON root must be an object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n", encoding="utf-8")


def _adapter(receipt_path: Path, executable: Path,
             source_path: Path) -> dict[str, object]:
    receipt = _json(receipt_path)
    required = {
        "schema": 1,
        "scope_id": "q3h4b-fixed-grid-cpu-qgis-adapter-v62",
        "envelope_version": 62,
        "status": "public_bounded_cpu_qgis_v62",
        "backend": "cpu",
        "automatic_backend_fallback": False,
        "parent_source_edited": False,
        "Delft3D_source_modified": False,
        "legacy_face_selector_bypassed_exactly_once": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Q3-H4b adapter receipt is outside the public v62 scope")
    scope = receipt.get("scope")
    if (not isinstance(scope, Mapping) or scope.get("grid") != "fixed" or
            scope.get("internal_faces") != "fully_wet_runtime_guarded" or
            scope.get("capacity_laws") != [2, 8, 11] or
            scope.get("slope_persistence_codes") != [3, 5, 6] or
            scope.get("exchange") != 0 or
            scope.get("bedload_adaptation") is not False or
            scope.get("secondary_steering") != "off" or
            scope.get("generic_hiding") != "off" or
            scope.get("prescribed_sediment_feed") != "zero_only"):
        raise ValueError("Q3-H4b adapter scope declaration differs")
    if (_sha(executable) != receipt.get("executable_sha256") or
            _sha(source_path) != receipt.get("derived_source_sha256")):
        raise ValueError("Q3-H4b adapter executable/source hash differs")
    sources = receipt.get("component_source_hashes")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Q3-H4b adapter lacks component source identities")
    return receipt


def _all_zero(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value)) and float(value) == 0.0
    if isinstance(value, Mapping):
        return all(_all_zero(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_zero(item) for item in value)
    return False


def _require_zero_prescribed_feed(metadata: Mapping[str, object]) -> None:
    bed_input = metadata.get("resolved_bed_input")
    if not isinstance(bed_input, Mapping):
        raise ValueError("Q3-H4b requires resolved bed-boundary inputs")
    for row in bed_input.values():
        if (not isinstance(row, Mapping) or
                not _all_zero(row.get("per_width_kg_m_s")) or
                not _all_zero(row.get("total_kg_s"))):
            raise ValueError("Q3-H4b excludes nonzero prescribed sediment feed")
    resolved = metadata.get("resolved_fraction_input")
    if not isinstance(resolved, Mapping):
        raise ValueError("Q3-H4b requires resolved fraction-boundary inputs")
    for boundary in resolved.values():
        if not isinstance(boundary, Mapping):
            raise ValueError("Q3-H4b fraction-boundary input is malformed")
        for fraction in boundary.values():
            if not isinstance(fraction, Mapping):
                raise ValueError("Q3-H4b fraction-boundary row is malformed")
            if (not _all_zero(fraction.get("bed_input_kg_s")) or
                    fraction.get("time_series") is not None):
                raise ValueError(
                    "Q3-H4b excludes prescribed fraction feeds and schedules")


def _base_inputs(root: Path, deck: Path, metadata_path: Path
                 ) -> tuple[dict[str, object], dict[str, object], str, str,
                            list[str]]:
    metadata = _json(metadata_path)
    if (metadata.get("format") != "xbraids_cutcell-v54" or
            deck.read_bytes().splitlines()[0] != b"54"):
        raise ValueError("Q3-H4b requires an authenticated Q3-C v54 base deck")
    if metadata.get("amr") is not None:
        raise ValueError("Q3-H4b v62 excludes AMR")
    adaptation = metadata.get("bedload_adaptation")
    secondary = metadata.get("secondary_flow")
    hiding = metadata.get("hiding_exposure")
    if (not isinstance(adaptation, Mapping) or
            adaptation.get("enabled") is not False or
            not isinstance(secondary, Mapping) or
            secondary.get("enabled") is not False or
            not isinstance(hiding, Mapping) or hiding.get("mode") != 0 or
            metadata.get("suspended_exchange_name") != "off"):
        raise ValueError(
            "Q3-H4b requires exchange/adaptation/secondary/hiding off")
    slope = metadata.get("bed_slope")
    if not isinstance(slope, Mapping) or slope.get("islope") != 0:
        raise ValueError("Q3-H4b requires an unmodified slope-off base deck")
    _require_zero_prescribed_feed(metadata)
    fields = root / "initial_fields.npz"
    if not fields.is_file() or fields.is_symlink():
        raise ValueError("Q3-H4b requires authenticated initial fields")
    with np.load(fields, allow_pickle=False) as arrays:
        q = np.asarray(arrays["q"], dtype=np.float64)
    dry = metadata.get("dry_tolerance_m")
    if (q.ndim != 2 or q.shape[1] < 1 or isinstance(dry, bool) or
            not isinstance(dry, (int, float)) or not np.isfinite(q).all() or
            np.any(q[:, 0] <= float(dry))):
        raise ValueError("Q3-H4b initial grid must be fully wet")

    capacity_sidecar = _json(root / CAPACITY_SIDECAR)
    capacity = load_capacity_foundation_execution(
        capacity_sidecar.get("capacity_foundation_execution_identity"))
    formula = metadata.get("bedload_law")
    if capacity["formula_code"] != formula or formula not in (2, 8, 11):
        raise ValueError("Q3-H4b capacity law/identity differs")
    channel_identity = metadata.get("channel_composition_identity")
    if not isinstance(channel_identity, Mapping):
        raise ValueError("Q3-H4b base deck lacks channel composition identity")
    channel = load_channel_composition_profile(channel_identity.get("profile"))
    if channel.fingerprint != channel_identity.get("profile_fingerprint_authority") \
            and channel.fingerprint != channel_identity.get("profile_fingerprint"):
        # Current v53 stores authority as prose and the exact identity under
        # native_profile_preimage_sha256/profile.fingerprint.
        if channel.fingerprint != channel_identity.get(
                "native_profile_preimage_sha256"):
            raise ValueError("Q3-H4b channel-composition identity differs")
    profile = metadata.get("execution_profile_v2")
    methods = (profile.get("profile", {}).get("method_selections", [])
               if isinstance(profile, Mapping) else [])
    slope_methods = [row.get("entry_id") for row in methods
                     if isinstance(row, Mapping) and
                     str(row.get("entry_id", "")).startswith("bed_slope.")]
    if slope_methods != ["bed_slope.off"]:
        raise ValueError("Q3-H4b base execution profile is not exactly slope-off")
    fractions = [row.fraction_id for row in channel.fractions]
    metadata_fractions = metadata.get("fraction_ids")
    if (not fractions or not isinstance(metadata_fractions, list) or
            (metadata_fractions and metadata_fractions != fractions) or
            (not metadata_fractions and fractions != ["single"])):
        raise ValueError("Q3-H4b base deck has inconsistent fraction identities")
    return (metadata, capacity, str(channel.fingerprint),
            _sha(root / CAPACITY_SIDECAR), fractions)


def _minimum_bedload_depth(metadata: Mapping[str, object]) -> float:
    materials = metadata.get("materials")
    fractions = materials.get("fractions") if isinstance(materials, Mapping) else None
    if isinstance(materials, Mapping) and fractions is None:
        fractions = [materials]
    if not isinstance(fractions, list) or not fractions:
        raise ValueError("Q3-H4b requires explicit material fractions")
    values = [row.get("minimum_bedload_depth_m")
              for row in fractions if isinstance(row, Mapping)]
    if (len(values) != len(fractions) or any(isinstance(value, bool) or
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in values) or len({float(value) for value in values}) != 1):
        raise ValueError("Q3-H4b requires one finite bedload-depth threshold")
    return float(values[0])


def _transform_deck(base: bytes, metadata: Mapping[str, object],
                    binding: Mapping[str, object]) -> bytes:
    try:
        text = base.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("Q3-H4b native deck must be ASCII") from error
    slope = metadata["bed_slope"]
    minimum = _minimum_bedload_depth(metadata)
    candidates: list[int] = []
    rows = text.splitlines(keepends=True)
    for index, row in enumerate(rows):
        tokens = row.strip().split()
        if len(tokens) != 5:
            continue
        try:
            values = [float(value) for value in tokens]
        except ValueError:
            continue
        if (values[0] == 0.0 and
                values[1] == float(slope["alfabs"]) and
                values[2] == float(slope["alfabn"]) and
                values[3] == float(slope["friction_angle_deg"]) and
                values[4] == minimum):
            candidates.append(index)
    if len(candidates) != 1:
        raise ValueError(
            "Q3-H4b expected exactly one authenticated slope-off deck record")
    controls = binding["parameters"]
    values = [
        int(binding["persistence_code"]), controls["alfabs"],
        controls["alfabn"], controls["ashld"], controls["bshld"],
        controls["cshld"], controls["dshld"], controls["alfpa"],
        controls["thcrpa"], minimum,
    ]
    ending = "\r\n" if rows[candidates[0]].endswith("\r\n") else "\n"
    rows[candidates[0]] = " ".join(
        str(value) if isinstance(value, int) else format(float(value), ".17g")
        for value in values) + ending
    result = "".join(rows).encode("ascii")
    if result == base:
        raise ValueError("Q3-H4b deck transform did not activate")
    return result


def _face_fingerprint(metadata: Mapping[str, object],
                      receipt: Mapping[str, object]) -> str:
    return stage_fingerprint("face_reconstruction_execution", {
        "method_id": "braids.native_cutcell_reconstruction_h4b_v62",
        "reconstruction_scheme": metadata.get("reconstruction_scheme"),
        "parent_cutcell_sha256": receipt.get("parent_cutcell_sha256"),
        "derived_source_sha256": receipt.get("derived_source_sha256"),
    })


def prepare_cpu_v62(
    base_run: str | Path, target_run: str | Path, *,
    method_id: str, controls: Mapping[str, object],
    adapter_executable: str | Path, adapter_receipt: str | Path,
    adapter_source: str | Path,
) -> Path:
    """Prepare a self-contained v62 run from one fresh Q3-C CPU run."""
    base = Path(base_run).expanduser().resolve()
    target = Path(target_run).expanduser().resolve()
    if not base.is_dir() or base.is_symlink():
        raise ValueError("Q3-H4b base run must be a real directory")
    if target.exists():
        raise ValueError("Q3-H4b target run already exists")
    try:
        target.relative_to(base)
    except ValueError:
        pass
    else:
        raise ValueError("Q3-H4b target run cannot be inside its base run")
    if any((base / name).exists() for name in
           ("frames.txt", "checkpoint.bin", "checkpoint_status.txt")):
        raise ValueError("Q3-H4b requires a fresh, not-yet-executed Q3-C base run")
    base_metadata, capacity, channel_fp, _, fractions = _base_inputs(
        base, base / LIVE_DECK, base / LIVE_METADATA)
    binding = project_binding_document(
        method_id=method_id, formula_code=int(base_metadata["bedload_law"]),
        fraction_ids=fractions, controls=controls)
    executable = Path(adapter_executable).expanduser().resolve()
    receipt_path = Path(adapter_receipt).expanduser().resolve()
    source_path = Path(adapter_source).expanduser().resolve()
    receipt = _adapter(receipt_path, executable, source_path)
    face_fp = _face_fingerprint(base_metadata, receipt)
    component = component_document(
        project_binding=binding, capacity_execution=capacity,
        channel_composition_fingerprint=channel_fp,
        face_reconstruction_fingerprint=face_fp,
        implementation_source_hashes=receipt["component_source_hashes"])
    live_deck = _transform_deck(
        (base / LIVE_DECK).read_bytes(), base_metadata, binding)

    shutil.copytree(base, target, copy_function=shutil.copy2)
    shutil.copy2(target / LIVE_DECK, target / BASE_DECK)
    shutil.copy2(target / LIVE_METADATA, target / BASE_METADATA)
    (target / LIVE_DECK).write_bytes(live_deck)
    shutil.copy2(executable, target / ADAPTER)
    shutil.copy2(receipt_path, target / ADAPTER_RECEIPT)
    shutil.copy2(source_path, target / ADAPTER_SOURCE)

    project_sidecar = {
        "format": "braids-q3h4b-cpu-project-v62",
        PROJECT_BINDING_KEY: binding,
        "base_project_sha256": (
            _sha(target / "project.json") if (target / "project.json").is_file()
            else None),
    }
    _write_json(target / PROJECT_SIDECAR, project_sidecar)
    live_metadata = copy.deepcopy(base_metadata)
    live_metadata["format"] = "xbraids_cutcell-v54+h4b-external-v62"
    live_metadata["bed_slope"] = {
        "islope": binding["persistence_code"],
        "method_id": binding["method_id"],
        **dict(binding["parameters"]),
        "application_count": 1,
        "owner": "external_authenticated_v62_envelope",
    }
    live_metadata[PROJECT_BINDING_KEY] = binding
    live_metadata["bed_slope_execution_fingerprint"] = component["fingerprint"]
    live_metadata["base_deck_sha256"] = _sha(target / BASE_DECK)
    _write_json(target / LIVE_METADATA, live_metadata)

    prepared = prepared_execution_document(
        project_binding=binding, component=component,
        capacity_execution=capacity,
        channel_composition_fingerprint=channel_fp,
        face_reconstruction_fingerprint=face_fp,
        implementation_source_hashes=receipt["component_source_hashes"],
        base_deck_sha256=_sha(target / BASE_DECK),
        live_deck_sha256=_sha(target / LIVE_DECK),
        base_metadata_sha256=_sha(target / BASE_METADATA),
        live_metadata_sha256=_sha(target / LIVE_METADATA),
        capacity_sidecar_sha256=_sha(target / CAPACITY_SIDECAR),
        project_binding_sidecar_sha256=_sha(target / PROJECT_SIDECAR),
        adapter_executable_sha256=_sha(target / ADAPTER),
        adapter_receipt_sha256=_sha(target / ADAPTER_RECEIPT),
        adapter_derived_source_sha256=_sha(target / ADAPTER_SOURCE),
    )
    _write_json(target / PREPARED_SIDECAR, {
        "format": "braids-q3h4b-cpu-prepared-v62",
        PREPARED_EXECUTION_KEY: prepared,
    })
    validate_prepared_run(target)
    return target


def validate_prepared_run(run_root: str | Path) -> dict[str, object]:
    """Revalidate project, base deck, transformed deck and adapter bytes."""
    root = Path(run_root).expanduser().resolve()
    project_sidecar = _json(root / PROJECT_SIDECAR)
    if project_sidecar.get("format") != "braids-q3h4b-cpu-project-v62":
        raise ValueError("Q3-H4b project sidecar format differs")
    binding = validate_project_binding(project_sidecar.get(PROJECT_BINDING_KEY))
    base_metadata, capacity, channel_fp, capacity_hash, _ = _base_inputs(
        root, root / BASE_DECK, root / BASE_METADATA)
    receipt = _adapter(
        root / ADAPTER_RECEIPT, root / ADAPTER, root / ADAPTER_SOURCE)
    expected_live = _transform_deck(
        (root / BASE_DECK).read_bytes(), base_metadata, binding)
    if (root / LIVE_DECK).read_bytes() != expected_live:
        raise ValueError("Q3-H4b live deck differs from its one exact transform")
    live_metadata = _json(root / LIVE_METADATA)
    if (live_metadata.get("format") != "xbraids_cutcell-v54+h4b-external-v62" or
            live_metadata.get(PROJECT_BINDING_KEY) != binding or
            live_metadata.get("base_deck_sha256") != _sha(root / BASE_DECK)):
        raise ValueError("Q3-H4b live metadata differs from project/base identity")
    face_fp = _face_fingerprint(base_metadata, receipt)
    prepared_sidecar = _json(root / PREPARED_SIDECAR)
    if prepared_sidecar.get("format") != "braids-q3h4b-cpu-prepared-v62":
        raise ValueError("Q3-H4b prepared sidecar format differs")
    prepared = validate_prepared_execution(
        prepared_sidecar.get(PREPARED_EXECUTION_KEY),
        project_binding=binding, capacity_execution=capacity,
        channel_composition_fingerprint=channel_fp,
        face_reconstruction_fingerprint=face_fp,
        implementation_source_hashes=receipt["component_source_hashes"])
    hashes = prepared["artifact_hashes"]
    actual = {
        "base_deck_sha256": _sha(root / BASE_DECK),
        "live_deck_sha256": _sha(root / LIVE_DECK),
        "base_metadata_sha256": _sha(root / BASE_METADATA),
        "live_metadata_sha256": _sha(root / LIVE_METADATA),
        "capacity_sidecar_sha256": capacity_hash,
        "project_binding_sidecar_sha256": _sha(root / PROJECT_SIDECAR),
        "adapter_executable_sha256": _sha(root / ADAPTER),
        "adapter_receipt_sha256": _sha(root / ADAPTER_RECEIPT),
        "adapter_derived_source_sha256": _sha(root / ADAPTER_SOURCE),
    }
    if hashes != actual:
        raise ValueError("Q3-H4b prepared artifact hashes differ")
    if live_metadata.get("bed_slope_execution_fingerprint") != prepared.get(
            "bed_slope_execution_fingerprint"):
        raise ValueError("Q3-H4b live metadata/component identity differs")
    return prepared


def verified_launch_spec(run_root: str | Path, *,
                         restart_identity: str | Path | None = None
                         ) -> dict[str, object]:
    """Return a QGIS-process launch spec only after complete v62 verification."""
    root = Path(run_root).expanduser().resolve()
    prepared = validate_prepared_run(root)
    arguments = [LIVE_DECK]
    restart = None
    if restart_identity is not None:
        restart_path = Path(restart_identity).expanduser().resolve()
        restart_sidecar = _json(restart_path)
        restart = validate_lifecycle_identity(
            restart_sidecar.get(RESTART_IDENTITY_KEY))
        if (restart["stage"] != "restart" or
                restart["prepared_execution_fingerprint"] !=
                prepared["fingerprint"]):
            raise ValueError("Q3-H4b restart/prepared identities differ")
        for name, expected in restart["file_hashes"].items():
            if _sha(root / name) != expected:
                raise ValueError("Q3-H4b restart file hash differs")
        checkpoint = next(iter(restart["file_hashes"]))
        arguments.append(checkpoint)
    return {
        "executable": root / ADAPTER,
        "arguments": arguments,
        "working_directory": root,
        "prepared_fingerprint": prepared["fingerprint"],
        "restart_fingerprint": None if restart is None else restart["fingerprint"],
        "cancellable_by_qgis_process_termination": True,
        "automatic_backend_fallback": False,
    }


def run_prepared(run_root: str | Path, *, timeout_s: float = 120.0,
                 omp_threads: int = 1) -> subprocess.CompletedProcess[str]:
    """Bounded convenience runner; QGIS should use ``verified_launch_spec``."""
    if (isinstance(omp_threads, bool) or not isinstance(omp_threads, int) or
            omp_threads < 1):
        raise ValueError("Q3-H4b OpenMP thread count must be positive")
    spec = verified_launch_spec(run_root)
    return subprocess.run(
        [str(spec["executable"]), *spec["arguments"]],
        cwd=spec["working_directory"], capture_output=True, text=True,
        timeout=float(timeout_s),
        env={**os.environ, "OMP_NUM_THREADS": str(omp_threads),
             "OMP_DYNAMIC": "FALSE"},
    )


def write_result_identity(run_root: str | Path, *, accepted_step: int,
                          time_s: float, file_names: Sequence[str]
                          ) -> Path:
    root = Path(run_root).expanduser().resolve()
    prepared = validate_prepared_run(root)
    files = {name: _sha(root / name) for name in file_names}
    identity = lifecycle_identity_document(
        stage="result", prepared_fingerprint=prepared["fingerprint"],
        accepted_step=accepted_step, time_s=time_s, file_hashes=files)
    path = root / RESULT_SIDECAR
    _write_json(path, {"format": "braids-q3h4b-cpu-result-v62",
                       RESULT_IDENTITY_KEY: identity})
    return path


def write_restart_identity(run_root: str | Path, *, result_identity: str | Path,
                           accepted_step: int, time_s: float,
                           checkpoint_file_names: Sequence[str]) -> Path:
    root = Path(run_root).expanduser().resolve()
    prepared = validate_prepared_run(root)
    result_sidecar = _json(Path(result_identity).expanduser().resolve())
    result = validate_lifecycle_identity(result_sidecar.get(RESULT_IDENTITY_KEY))
    if (result["stage"] != "result" or
            result["prepared_execution_fingerprint"] != prepared["fingerprint"]):
        raise ValueError("Q3-H4b result/prepared identities differ")
    if len(checkpoint_file_names) != 1:
        raise ValueError("Q3-H4b native restart requires exactly one checkpoint")
    files = {name: _sha(root / name) for name in checkpoint_file_names}
    identity = lifecycle_identity_document(
        stage="restart", prepared_fingerprint=prepared["fingerprint"],
        accepted_step=accepted_step, time_s=time_s, file_hashes=files,
        parent_result_fingerprint=result["fingerprint"])
    path = root / RESTART_SIDECAR
    _write_json(path, {"format": "braids-q3h4b-cpu-restart-v62",
                       RESTART_IDENTITY_KEY: identity})
    return path


__all__ = [
    "ADAPTER", "ADAPTER_RECEIPT", "ADAPTER_SOURCE", "BASE_DECK",
    "BASE_METADATA", "CAPACITY_SIDECAR", "LIVE_DECK", "LIVE_METADATA",
    "PREPARED_SIDECAR", "PROJECT_SIDECAR", "RESTART_SIDECAR",
    "RESULT_SIDECAR", "prepare_cpu_v62", "run_prepared",
    "validate_prepared_run", "verified_launch_spec",
    "write_restart_identity", "write_result_identity",
]
