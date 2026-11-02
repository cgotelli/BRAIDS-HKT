"""Portable, zero-step native exporter identity for graded live bedload.

The packaged GPU worker carries this exact BRAIDS/GeoClaw initializer and its
redistributable runtime DLL closure.  This module never imports a validation
fixture, compiler, QGIS, or GDAL.  The executable is used only to initialize
geometry and boundary records; the subsequent solver is the selected GPU path.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


EXPORTER_SHA256 = "a50f91dd99e5b48af94535761a7ec2f5b933e1e21ca9479370e78d03275eac33"
EXPORTER_RELATIVE = "braids_gpu/native_exporter_v1/xbraids_normal_export.exe"
REQUIRED_DLLS = frozenset(("libgfortran-5.dll", "libgomp-1.dll"))


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_packaged_exporter() -> dict:
    """Authenticate the shipped initializer and its declared runtime DLLs.

    QGIS separately pins and hashes the entire worker package.  This direct
    CLI guard also rejects a missing, substituted, or partially extracted
    native runtime before preparation creates any output folder.
    """
    worker = Path(__file__).resolve().parents[1]
    manifest_path = worker / "PACKAGE-MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("Packaged graded exporter requires a source-worker manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if (manifest.get("kind") != "experimental Windows source worker" or
            not isinstance(files, dict) or
            files.get(EXPORTER_RELATIVE) != EXPORTER_SHA256):
        raise ValueError("Packaged graded exporter identity is absent or differs")
    executable = worker / EXPORTER_RELATIVE
    runtime = executable.parent
    if (runtime.is_symlink() or executable.is_symlink() or not executable.is_file() or
            _sha256(executable) != EXPORTER_SHA256):
        raise ValueError("Packaged graded exporter executable differs")
    declared_dlls = {path for path in files if
        path.startswith("braids_gpu/native_exporter_v1/") and
        path.lower().endswith(".dll")}
    if not {"braids_gpu/native_exporter_v1/" + name for name in REQUIRED_DLLS} <= declared_dlls:
        raise ValueError("Packaged graded exporter DLL closure is incomplete")
    actual_dlls = {path.relative_to(worker).as_posix() for path in runtime.glob("*.dll")}
    if actual_dlls != declared_dlls:
        raise ValueError("Packaged graded exporter has an undeclared or missing DLL")
    for name in declared_dlls:
        path = worker / name
        if (path.parent != runtime or path.is_symlink() or not path.is_file() or
                _sha256(path) != files[name]):
            raise ValueError("Packaged graded exporter DLL differs: " + name)
    return {"executable": str(executable),
            "executable_sha256": EXPORTER_SHA256, "runtime": str(runtime)}


def export_environment(runtime: str | Path) -> dict[str, str]:
    """Reproduce the locked zero-step native environment without old helpers."""
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("BRAIDS_", "OMP_"))}
    environment.update(OMP_NUM_THREADS="8", OMP_DYNAMIC="FALSE",
        OMP_WAIT_POLICY="PASSIVE", BRAIDS_TIME_INTEGRATOR="legacy",
        BRAIDS_PROFILE="1", BRAIDS_GPU_INITIAL_EXPORT="1")
    environment["PATH"] = str(Path(runtime).resolve()) + os.pathsep + environment.get("PATH", "")
    return environment
