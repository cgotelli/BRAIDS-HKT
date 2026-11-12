#!/usr/bin/env python3
"""Build and verify the installable 64-bit Windows BRAIDS QGIS archive."""
from __future__ import annotations

import configparser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
from pathlib import PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any
import urllib.request
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "validation/sediment/johannot2026_braided_flume"
sys.path.insert(0, str(CASE))
from atomic_status import replace_with_windows_retry, write_text_atomic  # noqa: E402

PLUGIN_SOURCE = ROOT / "braids_qgis"
CORE_SOURCE = ROOT / "braids_core"
BUILD_DIR = ROOT / "solver/cut_cells/build"
NATIVE_SOURCE = BUILD_DIR / "xbraids_cutcell.exe"
NATIVE_BUILD = BUILD_DIR / "build.json"
DIST = ROOT / "dist/windows-x86_64"
PLUGIN_ID = "braids_qgis"
ADAPTIVE_RUNTIME = ROOT / 'dist/adaptive-river-alpha107'


def adaptive_runtime_files() -> dict[str, str]:
    path=ADAPTIVE_RUNTIME/'RUNTIME.json'
    if not path.is_file():raise RuntimeError('Build the qualified adaptive runtime before packaging')
    doc=json.loads(path.read_text())
    result={**doc['files'],'RUNTIME.json':digest(path)}
    for name,expected in result.items():
        p=(ADAPTIVE_RUNTIME/name).resolve()
        if not p.is_relative_to(ADAPTIVE_RUNTIME.resolve()) or digest(p)!=expected:
            raise RuntimeError('Adaptive runtime changed: '+name)
    return result

RAW_RUN_FILENAMES = {
    "boundary_ledger.txt", "case.data", "checkpoint.bin", "frames.bin",
    "frames.txt", "pilot_initial.npz", "trim-basin.nc", "trih-basin.nc",
}

RUNTIME_HASHES = {
    "libdl.dll": "5371648a682af6c79544a0799fca73a3363c20c59d59a7d29fd9f9230032bc0e",
    "libgcc_s_seh-1.dll": "89f1eaf8576f691e1aafc8707173618a74b8c82d6d0ac17a1be62c76a6c79aa1",
    "libgfortran-5.dll": "211b9a2d096d7a6ede71c444b557860a992f027f34390f88280f74d408cb5f0e",
    "libgomp-1.dll": "796b4f63a78769783923d3d5bc682c263d7a16284471f1028a4003cd2db7a2c0",
    "libquadmath-0.dll": "452542573b8d81725086742d69b7eb135d83ab77f1f872fd05fac83cf4da59a1",
    "libwinpthread-1.dll": "4bde68fbb2c6a9a64535928fc505056ebf6f5dc44f82ea2eaac3c714dc22ce79",
}

LICENSE_FILES = (
    ("Clawpack-LICENSE.txt", "668317944192d8217d63c0fd0730f889bd07f759b69ed6b657e1fb9902145ebe",
     "https://raw.githubusercontent.com/clawpack/clawpack/v5.14.0/LICENSE"),
    ("gcc-COPYING3", "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
     "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-13.2.0/COPYING3"),
    ("gcc-COPYING.LIB", "a9bdde5616ecdd1e980b44f360600ee8783b1f99b8cc83a2beb163a0a390e861",
     "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-13.2.0/COPYING.LIB"),
    ("gcc-COPYING.RUNTIME", "9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74",
     "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-13.2.0/COPYING.RUNTIME"),
    ("libquadmath-COPYING.LIB", "32434afcc8666ba060e111d715bfdb6c2d5dd8a35fa4d3ab8ad67d8f850d2f2b",
     "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-13.2.0/libquadmath/COPYING.LIB"),
    ("mingw-w64-COPYING", "99a69660981156c21336fdb5661f89341b013c94e4bf9e1c7467b4745718397f",
     "https://raw.githubusercontent.com/mingw-w64/mingw-w64/v11.0.1/COPYING"),
    ("winpthreads-COPYING", "63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177",
     "https://raw.githubusercontent.com/mingw-w64/mingw-w64/v11.0.1/mingw-w64-libraries/winpthreads/COPYING"),
    ("dlfcn-win32-COPYING", "4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7",
     "https://raw.githubusercontent.com/dlfcn-win32/dlfcn-win32/v1.4.1/COPYING"),
    ("dlfcn-win32-dlfcn.c", "0b7a7e71d4beeef7df6ad9422da7b311bb15edd05ac7016d2d93fa8f90a4944e",
     "https://raw.githubusercontent.com/dlfcn-win32/dlfcn-win32/v1.4.1/src/dlfcn.c"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(path: Path) -> str:
    """Hash source content independent of checkout line-ending conversion."""
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def metadata_version() -> str:
    parser = configparser.ConfigParser(); parser.read(PLUGIN_SOURCE / "metadata.txt", encoding="utf-8")
    version = parser.get("general", "version").strip()
    if not re.fullmatch(r"0\.1\.0-alpha\.\d+", version):
        raise RuntimeError(f"Unexpected alpha version: {version}")
    return version


def require_platform() -> None:
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("The Windows builder requires 64-bit Windows")


def packaged_source_files(source: Path) -> list[Path]:
    """Return the deterministic source subset copied into the plugin."""
    return [
        path for path in sorted(source.rglob("*"))
        if path.is_file()
        and not any(
            part in {"__pycache__", ".pytest_cache", "bin", "braids_core"}
            for part in path.relative_to(source).parts
        )
        and path.name != ".DS_Store"
        and path.suffix not in {".pyc", ".pyo"}
        and path.suffix.lower() in {".py", ".txt", ".md", ".json", ".png"}
    ]


def copy_python_tree(source: Path, destination: Path) -> None:
    for path in packaged_source_files(source):
        relative = path.relative_to(source)
        target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def current_packaged_source_sha256() -> dict[str, str]:
    """Fingerprint every current Python/plugin source shipped in the ZIP."""
    result: dict[str, str] = {}
    for source, prefix in (
        (PLUGIN_SOURCE, Path()), (CORE_SOURCE, Path("braids_core")),
    ):
        for path in packaged_source_files(source):
            relative = (prefix / path.relative_to(source)).as_posix()
            result[relative] = digest(path)
    for relative, payload in generated_product_assets().items():
        result[relative] = hashlib.sha256(payload).hexdigest()
    for name,value in adaptive_runtime_files().items():result['adaptive_runtime/'+name]=value
    return dict(sorted(result.items()))


def generated_product_assets() -> dict[str, bytes]:
    """Generate the shipped manual/catalogue from the executable contract."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from braids_core.sediment.qgis_product_v109 import (
        product_catalogue_v109, product_manual_v109,
    )
    return {
        "docs/q10_product_catalogue_v109.json": (
            json.dumps(product_catalogue_v109(), sort_keys=True, indent=2,
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii"),
        "docs/Q10_SEDIMENT_METHOD_MANUAL.md":
            product_manual_v109().encode("utf-8"),
    }


def write_generated_product_assets(plugin: Path) -> None:
    for relative, payload in generated_product_assets().items():
        path = plugin / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def verify_native_build() -> dict[str, Any]:
    if not NATIVE_SOURCE.is_file() or not NATIVE_BUILD.is_file():
        raise RuntimeError("Build solver/cut_cells/build/xbraids_cutcell.exe before packaging")
    record = json.loads(NATIVE_BUILD.read_text(encoding="utf-8"))
    exact_hashes = record.get("source_sha256", {})
    text_hashes = record.get("source_text_sha256", {})
    if not isinstance(exact_hashes, dict) or not exact_hashes:
        raise RuntimeError("Native build has no source hash manifest")
    if text_hashes and set(text_hashes) != set(exact_hashes):
        raise RuntimeError("Native source hash manifests have different file sets")
    for relative, expected in exact_hashes.items():
        source = ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"Native source is missing: {relative}")
        if digest(source) == expected:
            continue
        if not text_hashes or text_digest(source) != text_hashes.get(relative):
            raise RuntimeError(f"Native build is stale for {relative}")
    if digest(NATIVE_SOURCE) != record.get("executable_sha256"):
        raise RuntimeError("Native executable does not match build.json")
    return record


def imported_dlls(path: Path, objdump: str) -> list[str]:
    output = subprocess.check_output([objdump, "-p", str(path)], text=True, errors="replace")
    return [match.group(1) for match in re.finditer(r"DLL Name:\s*([^\s]+)", output)]


def runtime_sources() -> dict[str, Path]:
    compiler = shutil.which("gfortran"); objdump = shutil.which("objdump")
    if compiler is None or objdump is None:
        raise RuntimeError("gfortran and objdump are required for Windows packaging")
    directory = Path(compiler).resolve().parent
    available = {path.name.lower(): path for path in directory.glob("*.dll")}
    found: dict[str, Path] = {}; queue = imported_dlls(NATIVE_SOURCE, objdump)
    while queue:
        name = queue.pop(0); source = available.get(name.lower())
        if source is None or source.name.lower() in found:
            continue
        found[source.name.lower()] = source
        queue.extend(imported_dlls(source, objdump))
    resolved = {path.name: path for path in found.values()}
    if set(resolved) != set(RUNTIME_HASHES):
        raise RuntimeError(f"Unexpected MinGW runtime closure: {sorted(resolved)}")
    for name, expected in RUNTIME_HASHES.items():
        if digest(resolved[name]) != expected:
            raise RuntimeError(f"Unrecognized runtime DLL bytes: {name}")
    return resolved


def copy_native(destination: Path) -> dict[str, Any]:
    development = verify_native_build(); destination.mkdir(parents=True, exist_ok=True)
    executable = destination / NATIVE_SOURCE.name; shutil.copy2(NATIVE_SOURCE, executable)
    copied_executable_sha256 = digest(executable)
    if (
        copied_executable_sha256 != development.get("executable_sha256")
        or digest(NATIVE_SOURCE) != development.get("executable_sha256")
    ):
        raise RuntimeError("Native executable changed while packaging")
    libraries = runtime_sources()
    for name, source in libraries.items():
        target = destination / name
        shutil.copy2(source, target)
        if digest(target) != RUNTIME_HASHES[name] or digest(source) != RUNTIME_HASHES[name]:
            raise RuntimeError(f"Runtime DLL changed while packaging: {name}")
    record = {
        "format": "braids-native-alpha-build-v1",
        "platform": "windows-x86_64",
        "compiler": development.get("compiler"),
        "openmp": True,
        "source_sha256": development.get("source_sha256", {}),
        "source_text_sha256": development.get("source_text_sha256", {}),
        "executable_sha256": copied_executable_sha256,
        "bundled_runtime_sha256": {name: digest(destination / name) for name in sorted(libraries)},
        "scope": "Relocatable Windows runtime with pinned GNU Fortran/OpenMP dependencies",
    }
    (destination / "build.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "BRAIDS-Windows-packager/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def copy_licenses(plugin: Path) -> None:
    destination = plugin / "licenses"; destination.mkdir(parents=True, exist_ok=True)
    files = []
    for name, expected, url in LICENSE_FILES:
        data = _download(url)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Downloaded license/source hash mismatch: {name}")
        (destination / name).write_bytes(data)
        files.append({"path": name, "sha256": expected, "url": url})
    manifest = {
        "format": 1,
        "toolchain": "WinLibs GCC 13.2.0 / MinGW-w64 11.0.1 UCRT POSIX SEH r8",
        "toolchain_release": "https://github.com/brechtsanders/winlibs_mingw/releases/tag/13.2.0posix-18.1.5-11.0.1-ucrt-r8",
        "scope": "Pinned license texts, source-access references and exact unmodified runtime DLL identities.",
        "libraries": RUNTIME_HASHES,
        "files": files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_manifests(plugin: Path, version: str, native: dict[str, Any]) -> None:
    build_info = {
        "format": "braids-qgis-alpha-build-v1", "plugin_id": PLUGIN_ID,
        "version": version, "channel": "alpha", "platform": "windows-x86_64",
        "qgis_minimum_version": "3.40", "built_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": current_packaged_source_sha256(),
        "native_executable_sha256": native["executable_sha256"],
        "phase2_qualification_sha256": digest(plugin / "phase2_qualification.json"),
        "phase3_qualification_sha256": digest(plugin / "phase3_qualification.json"),
        "general_avalanche_qualification_sha256": digest(plugin / "general_avalanche_qualification.json"),
        "secondary_flow_qualification_sha256": digest(plugin / "secondary_flow_qualification.json"),
        "bank_failure_qualification_sha256": digest(plugin / "bank_failure_qualification.json"),
        "adaptation_length_qualification_sha256": digest(plugin / "adaptation_length_qualification.json"),
        "morphological_acceleration_qualification_sha256": digest(
            plugin / "morphological_acceleration_qualification.json"),
        "bedform_roughness_qualification_sha256": digest(
            plugin / "bedform_roughness_qualification.json"),
    }
    (plugin / "BUILD_INFO.json").write_text(json.dumps(build_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {str(path.relative_to(plugin)).replace("\\", "/"): digest(path)
             for path in sorted(plugin.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}
    manifest = {"format": "braids-qgis-alpha-manifest-v1", "files_sha256": files}
    (plugin / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def zip_plugin(plugin: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(plugin.rglob("*")):
            if not path.is_file():
                continue
            relative = (Path(PLUGIN_ID) / path.relative_to(plugin)).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 9, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.suffix.lower() in {".exe", ".dll"} else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            output.writestr(info, path.read_bytes(), compresslevel=9)


def publish_archive(plugin: Path, archive: Path) -> dict[str, Any]:
    """Build and verify a ZIP off-path before atomically publishing it."""
    archive = archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(
        f".{archive.name}.{os.getpid()}.{uuid4().hex}.partial"
    )
    try:
        zip_plugin(plugin, temporary)
        verification = verify_archive(temporary)
        archive_hash = digest(temporary)
        replace_with_windows_retry(temporary, archive)
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        write_text_atomic(checksum, f"{archive_hash}  {archive.name}\n")
        return {
            "archive": str(archive), "sha256": archive_hash,
            "size_bytes": archive.stat().st_size,
            "verification": verification,
        }
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def forbidden_archive_members(names: list[str]) -> list[str]:
    """Find upstream source trees and generated solver state in an archive."""
    forbidden = []
    for original in names:
        normalized = original.replace("\\", "/")
        parts = tuple(part.lower() for part in PurePosixPath(normalized).parts)
        basename = parts[-1] if parts else ""
        # A checkout name is a directory component.  Match every renamed
        # Delft3D checkout (not just ``Delft3D-main``), while allowing an
        # ordinary prose file such as ``docs/Delft3D-notes.md``.
        directory_parts = parts if normalized.endswith("/") else parts[:-1]
        has_upstream_source = any(
            part.startswith("delft3d") for part in directory_parts
        )
        has_raw_run_directory = any(part in {"run", "runs"} for part in parts)
        has_raw_solver_product = (
            basename in RAW_RUN_FILENAMES
            or basename.startswith(("trim-", "trih-")) and basename.endswith(".nc")
        )
        if has_upstream_source or has_raw_run_directory or has_raw_solver_product:
            forbidden.append(original)
    return forbidden


def is_generated_python_cache(path: Path, plugin: Path) -> bool:
    """Return true only for an ordinary CPython cache file below the plugin.

    QGIS creates ``__pycache__/*.pyc`` after a verified release is first
    imported.  Those derived files are not distributable payload and cannot be
    listed in the immutable release manifest.  Keep this predicate deliberately
    narrow: a cache-directory source file, an adjacent legacy ``.pyc``, or a
    symlink remains an unexpected installed file.
    """
    try:
        relative = path.relative_to(plugin)
    except ValueError:
        return False
    parents_below_plugin = []
    for parent in path.parents:
        if parent == plugin:
            break
        parents_below_plugin.append(parent)
    return (
        path.is_file()
        and not path.is_symlink()
        and not any(parent.is_symlink() for parent in parents_below_plugin)
        and path.suffix.lower() in {".pyc", ".pyo"}
        and "__pycache__" in relative.parts
    )


def verify_manifest_files(
    plugin: Path, *, context: str,
    allow_generated_python_cache: bool = False,
) -> dict[str, str]:
    """Require every payload file, and no persistent extra, to match manifest.

    Archive and staging verification stays strict by default.  A caller
    auditing an already imported QGIS profile may explicitly ignore only the
    generated cache files recognized by :func:`is_generated_python_cache`.
    """
    manifest_path = plugin / "MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{context} manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_files = manifest.get("files_sha256")
    if not isinstance(manifest_files, dict) or not manifest_files:
        raise RuntimeError(f"{context} manifest has no files")
    actual_paths = {
        path.relative_to(plugin).as_posix(): path
        for path in plugin.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    actual_files = set(actual_paths)
    missing = sorted(set(manifest_files) - actual_files)
    unlisted = sorted(actual_files - set(manifest_files))
    if allow_generated_python_cache:
        unlisted = [
            relative for relative in unlisted
            if not is_generated_python_cache(actual_paths[relative], plugin)
        ]
    if missing or unlisted:
        raise RuntimeError(
            f"{context} manifest/file set mismatch; "
            f"missing={missing[:10]}, unlisted={unlisted[:10]}"
        )
    for relative, expected in manifest_files.items():
        if digest(plugin / relative) != expected:
            raise RuntimeError(f"{context} file hash mismatch: {relative}")
    return manifest_files


def verify_packaged_source_manifest(
    plugin: Path, manifest_files: dict[str, str], *, context: str,
) -> dict[str, str]:
    """Bind BUILD_INFO's source claim to the actual manifested file bytes."""
    path = plugin / "BUILD_INFO.json"
    try:
        build_info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{context} BUILD_INFO is invalid") from error
    sources = build_info.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError(f"{context} BUILD_INFO has no source manifest")
    expected_sources = {
        relative for relative in manifest_files
        if relative != "BUILD_INFO.json"
        and not relative.startswith(("bin/", "licenses/"))
    }
    if set(sources) != expected_sources:
        missing = sorted(expected_sources - set(sources))
        extra = sorted(set(sources) - expected_sources)
        raise RuntimeError(
            f"{context} source manifest/file set mismatch; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    for relative, expected in sources.items():
        if (
            not isinstance(relative, str) or not relative
            or not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or manifest_files.get(relative) != expected
        ):
            raise RuntimeError(
                f"{context} source manifest mismatch: {relative}"
            )
    return sources


def verify_archive(archive: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="braids-windows-verify-") as temporary:
        folder = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            forbidden = forbidden_archive_members(names)
            if forbidden:
                raise RuntimeError(
                    "Archive contains forbidden source or raw-run content: "
                    + ", ".join(forbidden[:10])
                )
            zipped.extractall(folder)
        plugin = folder / PLUGIN_ID
        manifest_files = verify_manifest_files(plugin, context="Archive")
        sources = verify_packaged_source_manifest(
            plugin, manifest_files, context="Archive",
        )
        phase3 = json.loads((plugin / "phase3_qualification.json").read_text(encoding="utf-8"))
        if phase3.get("status") != "PASS" or phase3.get("gates_passed") != phase3.get("gates_total"):
            raise RuntimeError("Archive does not contain a passing Phase 3 manifest")
        avalanche = json.loads((plugin / "general_avalanche_qualification.json").read_text(encoding="utf-8"))
        if avalanche.get("status") != "PASS" or avalanche.get("gates_passed") != avalanche.get("gates_total"):
            raise RuntimeError("Archive does not contain a passing general-avalanche manifest")
        secondary = json.loads((plugin / "secondary_flow_qualification.json").read_text(encoding="utf-8"))
        if secondary.get("status") != "PASS" or secondary.get("gates_passed") != secondary.get("gates_total"):
            raise RuntimeError("Archive does not contain a passing secondary-flow manifest")
        bank_failure = json.loads((plugin / "bank_failure_qualification.json").read_text(encoding="utf-8"))
        if bank_failure.get("status") != "PASS" or bank_failure.get("gates_passed") != bank_failure.get("gates_total"):
            raise RuntimeError("Archive does not contain a passing bank-failure manifest")
        adaptation = json.loads((plugin / "adaptation_length_qualification.json").read_text(encoding="utf-8"))
        if adaptation.get("status") != "PASS" or adaptation.get("gates_passed") != adaptation.get("gates_total"):
            raise RuntimeError("Archive does not contain a passing adaptation-length manifest")
        acceleration = json.loads((plugin / "morphological_acceleration_qualification.json").read_text(encoding="utf-8"))
        if (acceleration.get("status") != "PASS" or
                acceleration.get("gates_passed") != acceleration.get("gates_total") or
                acceleration.get("published_regime_limits", {}).get("wet_dry_hydraulic_transient") != 1):
            raise RuntimeError("Archive does not contain a passing fail-closed morphological-acceleration manifest")
        bedform = json.loads((plugin / "bedform_roughness_qualification.json").read_text(encoding="utf-8"))
        if (bedform.get("status") != "PASS" or
                bedform.get("gates_passed") != bedform.get("gates_total") or
                bedform.get("default_enabled") is not False):
            raise RuntimeError("Archive does not contain a passing default-off bedform-roughness manifest")
        native = plugin / "bin/windows-x86_64/xbraids_cutcell.exe"
        build = json.loads((plugin / "bin/windows-x86_64/build.json").read_text(encoding="utf-8"))
        if digest(native) != build["executable_sha256"]:
            raise RuntimeError("Archived native executable identity is invalid")
        return {"files_verified": len(manifest_files),
                "source_files_verified": len(sources),
                "manifest_sha256": digest(plugin / "MANIFEST.json"),
                "phase3_gates": phase3["gates_total"],
                "general_avalanche_gates": avalanche["gates_total"],
                "secondary_flow_gates": secondary["gates_total"],
                "bank_failure_gates": bank_failure["gates_total"],
                "adaptation_length_gates": adaptation["gates_total"],
                "morphological_acceleration_gates": acceleration["gates_total"],
                "bedform_roughness_gates": bedform["gates_total"],
                "native_sha256": digest(native)}


def main() -> None:
    require_platform(); version = metadata_version(); DIST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".braids_qgis_release_stage_", dir=DIST,
    ) as temporary:
        stage = Path(temporary)
        plugin = stage / PLUGIN_ID
        copy_python_tree(PLUGIN_SOURCE, plugin)
        copy_python_tree(CORE_SOURCE, plugin / "braids_core")
        write_generated_product_assets(plugin)
        for name in adaptive_runtime_files():
            target=plugin/'adaptive_runtime'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ADAPTIVE_RUNTIME/name,target)
        native = copy_native(plugin / "bin/windows-x86_64")
        copy_licenses(plugin)
        write_manifests(plugin, version, native)
        archive = DIST / f"BRAIDS-{version}-QGIS-windows-x86_64.zip"
        result = publish_archive(plugin, archive)
    print(json.dumps({
        **result, "version": version,
        "staging_policy": "unique ephemeral directory; atomic ZIP publication",
    }, indent=2))


if __name__ == "__main__":
    main()
