"""Authenticate an extracted external GPU worker without importing its code.

This checks the packaged source/data payload, not the separate Python, CuPy,
CUDA driver or GPU hardware. A package passing this check is not automatically
a supported QGIS GPU release.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re


_REQUIRED_PAYLOAD = frozenset(("braids_gpu/__main__.py", "braids_gpu/runner.py",
                               "requirements-gpu.txt"))
_EXECUTABLE_SUFFIXES = frozenset((".py", ".pyw", ".pyc", ".pyo", ".pyd",
                                 ".cu", ".dll", ".exe"))
_SEPARATE_ENVIRONMENT_FOLDERS = frozenset((".venv", ".venv_gpu", "__pycache__"))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _payload_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("Invalid GPU worker payload path")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts)
            or relative.as_posix() != name):
        raise ValueError("Invalid GPU worker payload path: " + name)
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("GPU worker payload escapes its folder: " + name)
    return path


def inspect_gpu_source_worker(folder: str | Path, *,
                              expected_manifest_sha256: str | None = None) -> dict:
    """Verify every declared payload file and executable source namespace.

    Returns the frozen manifest digest and package scope for the eventual
    external-process launcher. The caller must separately verify its chosen
    Python environment and numerical capability for the prepared input.
    """
    root = Path(folder).expanduser().resolve()
    manifest_path = root / "PACKAGE-MANIFEST.json"
    digest = _sha256(manifest_path)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("GPU worker manifest differs from the selected identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or
            manifest.get("kind") != "experimental Windows source worker" or
            manifest.get("automatic_backend_selection") is not False or
            manifest.get("QGIS_release") is not True):
        raise ValueError("Unsupported or misidentified GPU worker package")
    files = manifest.get("files")
    if not isinstance(files, dict) or not _REQUIRED_PAYLOAD <= files.keys():
        raise ValueError("GPU worker payload manifest is incomplete")
    for name, expected in files.items():
        path = _payload_path(root, name)
        if (not isinstance(expected, str) or len(expected) != 64 or
                not path.is_file() or _sha256(path) != expected):
            raise ValueError("GPU worker payload differs: " + name)
    # Additional code beside the package could shadow a declared import.
    # A separately installed Python environment has its own trust boundary;
    # it is checked by the launcher, not by this source-payload verifier.
    for directory, subdirs, names in os.walk(root):
        base = Path(directory)
        subdirs[:] = [name for name in subdirs
                      if name not in _SEPARATE_ENVIRONMENT_FOLDERS]
        for name in subdirs:
            if (base / name).is_symlink():
                raise ValueError("Unexpected GPU worker source link: " + name)
        for name in names:
            path = base / name
            if path.suffix.lower() in _EXECUTABLE_SUFFIXES:
                relative = path.relative_to(root).as_posix()
                if relative not in files:
                    raise ValueError("Undeclared GPU worker executable source: " + relative)
    scope = manifest.get("source_worker_scope")
    if (not isinstance(scope, dict) or
            not isinstance(scope.get("supported_gpu_variants"), list) or
            not isinstance(scope.get("excluded_gpu_variants"), dict) or
            not isinstance(scope.get("excluded_prepared_sidecars"), dict)):
        raise ValueError("GPU worker capability scope is missing")
    catalog = scope.get("sediment_method_catalog")
    required = {"schema", "scope_id", "semantic_catalog_id", "source",
                "source_sha256"}
    if (not isinstance(catalog, dict) or set(catalog) != required or
            type(catalog.get("schema")) is not int or catalog["schema"] <= 0 or
            not isinstance(catalog.get("scope_id"), str) or not catalog["scope_id"] or
            not isinstance(catalog.get("semantic_catalog_id"), str) or
            re.fullmatch(r"[0-9a-f]{64}", catalog["semantic_catalog_id"]) is None or
            not isinstance(catalog.get("source_sha256"), str) or
            re.fullmatch(r"[0-9a-f]{64}", catalog["source_sha256"]) is None or
            catalog.get("source") != "braids_core/sediment_method_catalog.py" or
            files.get(catalog["source"]) != catalog["source_sha256"]):
        raise ValueError("GPU worker sediment-method catalogue contract differs")
    product = scope.get("qgis_product_v109")
    product_required = {
        "schema", "product_id", "product_catalogue_id",
        "semantic_catalog_id", "cross_physics_catalogue_id",
        "output_registry_id", "backend", "integrator",
        "automatic_backend_fallback",
    }
    if (not isinstance(product, dict) or set(product) != product_required or
            product.get("schema") != "braids.qgis_product.v109" or
            product.get("backend") != "implicit_fwave_midpoint_gpu" or
            product.get("integrator") != "implicit_fwave_midpoint" or
            product.get("automatic_backend_fallback") is not False or
            any(re.fullmatch(r"[0-9a-f]{64}", str(product.get(key))) is None
                for key in ("product_catalogue_id", "semantic_catalog_id",
                            "cross_physics_catalogue_id", "output_registry_id"))):
        raise ValueError("GPU worker Q10 product contract differs")
    recursive = scope.get("recursive_worker_v118")
    if recursive is not None:
        source = "braids_core/sediment/recursive_worker_v118.py"
        if (not isinstance(recursive, dict) or recursive != {
                "schema": "braids.recursive_worker.v118",
                "required_worker_variant": "recursive_bank_amr_v118",
                "source": source,
                "source_sha256": recursive.get("source_sha256"),
                "automatic_backend_fallback": False,
        } or files.get(source) != recursive.get("source_sha256")):
            raise ValueError("GPU worker recursive-v118 contract differs")
    return {"root": root, "manifest_sha256": digest, "payload_files": len(files),
            "scope": scope, "release_ready": True}
