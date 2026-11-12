"""QGIS-facing helpers for the pure Q10 sediment product contract."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
from typing import Mapping

from braids_core.sediment.qgis_product_v109 import (
    GPU_BACKEND,
    PRODUCT_SIDECAR,
    ProductResolutionV109,
    canonical_backend,
    product_catalogue_v109,
    product_manual_v109,
    load_product_document_v109,
    resolve_qgis_product_v109,
    validate_product_document_v109,
    verify_worker_capabilities_v109,
)
from braids_core.sediment.cross_physics_v108 import output_registry_v108


PROJECT_KEY = "qgis_product_v109"
PREPARED_BINDING = "qgis_product_prepared_v109.json"


def morphology_output_choices() -> tuple[dict, ...]:
    """Return stable rows used to generate the result-selection widgets."""
    return tuple(dict(row) for row in output_registry_v108()["mappings"])


def selected_output_settings(rows: Mapping[str, object] | None) -> dict[str, object]:
    if rows is None:
        return {}
    if not isinstance(rows, Mapping):
        raise ValueError("Morphology output settings must be a mapping")
    return {str(key): value for key, value in rows.items()}


def bind_qgis_product(
    project: Mapping, *, backend: str,
    output_settings: Mapping[str, object] | None = None,
    apply_output_defaults: bool = False,
) -> dict:
    profile = resolve_qgis_product_v109(
        project, backend=canonical_backend(backend),
        output_selection=selected_output_settings(output_settings),
        apply_output_defaults=apply_output_defaults,
    )
    return profile.as_dict()


def validate_qgis_product(project: Mapping) -> ProductResolutionV109 | None:
    document = project.get(PROJECT_KEY)
    if document is None:
        return None
    return validate_product_document_v109(project, document)


def write_prepared_product(run_root: str | Path, project: Mapping) -> Path | None:
    """Write one canonical sidecar after project identity validation."""
    profile = validate_qgis_product(project)
    if profile is None:
        return None
    path = Path(run_root) / PRODUCT_SIDECAR
    path.write_text(json.dumps(profile.as_dict(), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=True,
                               allow_nan=False), encoding="ascii")
    return path


def verify_gpu_worker_for_project(project: Mapping, worker_scope: Mapping) -> dict:
    profile = validate_qgis_product(project)
    if profile is None:
        raise ValueError("Project has no portable Q10 CPU/GPU product identity")
    if profile.backend != GPU_BACKEND:
        raise ValueError("Project product identity is not the GPU backend")
    return verify_worker_capabilities_v109(profile, worker_scope)


def materialize_gpu_prepared_product(
    native_root: str | Path, prepared_root: str | Path,
    project: Mapping, worker_scope: Mapping,
) -> Path:
    """Bind the verified worker and portable profile to prepared GPU input."""
    native_root = Path(native_root)
    prepared_root = Path(prepared_root)
    profile = validate_qgis_product(project)
    if profile is None:
        raise ValueError("GPU product preparation requires a portable Q10 identity")
    verification = verify_worker_capabilities_v109(profile, worker_scope)
    source = native_root / PRODUCT_SIDECAR
    if not source.is_file():
        raise ValueError("Native preparation omitted the Q10 product sidecar")
    document = json.loads(source.read_text(encoding="ascii"))
    if document != profile.as_dict():
        raise ValueError("Native Q10 product sidecar differs from the project")
    target = prepared_root / PRODUCT_SIDECAR
    if target.exists():
        raise ValueError("GPU prepared product sidecar already exists")
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt = {
        "schema": "braids.qgis_product_prepared.v109",
        "product_fingerprint": profile.fingerprint,
        "product_sidecar": PRODUCT_SIDECAR,
        "product_sidecar_sha256": digest,
        "worker_capability": verification["capability"],
    }
    receipt_path = prepared_root / PREPARED_BINDING
    receipt_path.write_text(json.dumps(receipt, sort_keys=True,
                                       separators=(",", ":"),
                                       ensure_ascii=True, allow_nan=False),
                            encoding="ascii")
    return receipt_path


def validate_gpu_result_product(
    run_root: str | Path, prepared_root: str | Path,
) -> ProductResolutionV109 | None:
    """Authenticate the portable Q10 identity used by a result/restart.

    Historical runs without a Q10 sidecar remain readable.  A current run is
    accepted only when the prepared sidecar, worker binding, copied result
    sidecars and published run envelope are all byte-identical or carry the
    corresponding digest.  This check does not import or allocate CUDA state.
    """
    run_root = Path(run_root)
    prepared_root = Path(prepared_root)
    prepared_sidecar = prepared_root / PRODUCT_SIDECAR
    prepared_binding = prepared_root / PREPARED_BINDING
    run_sidecar = run_root / PRODUCT_SIDECAR
    run_binding = run_root / PREPARED_BINDING
    present = tuple(path.is_file() for path in (
        prepared_sidecar, prepared_binding, run_sidecar, run_binding))
    if not any(present):
        return None
    if not all(present):
        raise ValueError("Incomplete Q10 result/restart product identity")
    if prepared_sidecar.read_bytes() != run_sidecar.read_bytes():
        raise ValueError("Q10 result product differs from prepared input")
    if prepared_binding.read_bytes() != run_binding.read_bytes():
        raise ValueError("Q10 result worker binding differs from prepared input")
    profile = load_product_document_v109(json.loads(
        prepared_sidecar.read_text(encoding="ascii")))
    digest = hashlib.sha256(prepared_sidecar.read_bytes()).hexdigest()
    binding = json.loads(prepared_binding.read_text(encoding="ascii"))
    if (binding.get("schema") != "braids.qgis_product_prepared.v109" or
            binding.get("product_fingerprint") != profile.fingerprint or
            binding.get("product_sidecar") != PRODUCT_SIDECAR or
            binding.get("product_sidecar_sha256") != digest or
            not isinstance(binding.get("worker_capability"), Mapping)):
        raise ValueError("Q10 prepared worker/product binding differs")
    manifest_path = run_root / "run.json"
    if not manifest_path.is_file():
        raise ValueError("Q10 result is missing run.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "fingerprint": profile.fingerprint,
        "backend": profile.backend,
        "integrator": profile.integrator,
        "sidecar": PRODUCT_SIDECAR,
        "sidecar_sha256": digest,
        "prepared_binding": PREPARED_BINDING,
    }
    if manifest.get("qgis_product_v109") != expected:
        raise ValueError("Q10 result manifest product identity differs")
    return profile


def generated_product_assets() -> dict[str, str]:
    """Build/package hook; both assets come from the exact runtime catalogue."""
    return {
        "q10_product_catalogue_v109.json": json.dumps(
            product_catalogue_v109(), sort_keys=True, indent=2,
            ensure_ascii=True, allow_nan=False) + "\n",
        "Q10_SEDIMENT_METHOD_MANUAL.md": product_manual_v109(),
    }


__all__ = [
    "PREPARED_BINDING", "PROJECT_KEY", "bind_qgis_product",
    "generated_product_assets", "materialize_gpu_prepared_product",
    "morphology_output_choices", "selected_output_settings",
    "validate_gpu_result_product", "validate_qgis_product",
    "verify_gpu_worker_for_project",
    "write_prepared_product",
]
