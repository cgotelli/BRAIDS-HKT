#!/usr/bin/env python3
"""Build the installable BRAIDS QGIS alpha archive for macOS Apple Silicon."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SOURCE = ROOT / "braids_qgis"
CORE_SOURCE = ROOT / "braids_core"
NATIVE_SOURCE = ROOT / "solver" / "cut_cells" / "build" / "xbraids_cutcell"
NATIVE_BUILD = NATIVE_SOURCE.parent / "build.json"
DIST = ROOT / "dist"
PLUGIN_ID = "braids_qgis"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(*arguments: str | Path) -> str:
    result = subprocess.run(
        [str(value) for value in arguments], check=True, capture_output=True, text=True)
    return result.stdout


def metadata_version() -> str:
    parser = configparser.ConfigParser()
    parser.read(PLUGIN_SOURCE / "metadata.txt", encoding="utf-8")
    version = parser.get("general", "version").strip()
    if "alpha" not in version.lower():
        raise RuntimeError("The alpha builder requires an alpha version in metadata.txt")
    return version


def distribution_tag() -> str:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("This alpha builder currently supports macOS Apple Silicon only")
    return "macos-arm64"


def copy_python_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in {"__pycache__", ".pytest_cache", "bin", "braids_core"}
               for part in relative.parts):
            continue
        if path.is_dir() or path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.suffix.lower() not in {".py", ".txt", ".md", ".json", ".png"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def linked_libraries(path: Path) -> list[str]:
    rows = []
    for line in command("otool", "-L", path).splitlines()[1:]:
        line = line.strip()
        if line:
            rows.append(line.split(" (compatibility version", 1)[0])
    return rows


def runtime_sources() -> dict[str, Path]:
    """Resolve the non-system Fortran/OpenMP closure of the development binary."""
    found: dict[str, Path] = {}
    queue = [Path(value) for value in linked_libraries(NATIVE_SOURCE)
             if value.startswith("/opt/homebrew/")]
    while queue:
        source = queue.pop(0).resolve()
        if source.name in found:
            continue
        if not source.is_file():
            raise RuntimeError(f"Missing native runtime dependency: {source}")
        found[source.name] = source
        for dependency in linked_libraries(source):
            if dependency.startswith("/opt/homebrew/"):
                queue.append(Path(dependency))
            elif dependency.startswith("@rpath/"):
                candidate = source.parent / Path(dependency).name
                if candidate.is_file():
                    queue.append(candidate)
    required = {"libgfortran.5.dylib", "libgomp.1.dylib", "libquadmath.0.dylib",
                "libgcc_s.1.1.dylib"}
    if set(found) != required:
        raise RuntimeError(f"Unexpected native runtime closure: {sorted(found)}")
    return found


def make_relocatable(destination: Path) -> dict:
    if not NATIVE_SOURCE.is_file() or not NATIVE_BUILD.is_file():
        raise RuntimeError("Build solver/cut_cells/build/xbraids_cutcell before packaging")
    destination.mkdir(parents=True, exist_ok=True)
    executable = destination / NATIVE_SOURCE.name
    shutil.copy2(NATIVE_SOURCE, executable)
    executable.chmod(0o755)
    libraries = runtime_sources()
    for name, source in libraries.items():
        target = destination / name
        shutil.copy2(source, target)
        target.chmod(0o755)

    targets = [executable, *(destination / name for name in sorted(libraries))]
    bundled_names = set(libraries)
    for target in targets:
        for dependency in linked_libraries(target):
            name = Path(dependency).name
            if name in bundled_names and dependency != f"@loader_path/{name}":
                command("install_name_tool", "-change", dependency,
                        f"@loader_path/{name}", target)
        if target.suffix == ".dylib":
            command("install_name_tool", "-id", f"@loader_path/{target.name}", target)
        command("codesign", "--force", "--sign", "-", target)

    unresolved = {
        target.name: [value for value in linked_libraries(target)
                      if value.startswith("/opt/homebrew/")]
        for target in targets
    }
    unresolved = {name: values for name, values in unresolved.items() if values}
    if unresolved:
        raise RuntimeError(f"Unbundled Homebrew runtime references remain: {unresolved}")

    development = json.loads(NATIVE_BUILD.read_text(encoding="utf-8"))
    record = {
        "format": "braids-native-alpha-build-v1",
        "platform": distribution_tag(),
        "compiler": development.get("compiler"),
        "openmp": True,
        "source_sha256": development.get("source_sha256", {}),
        "executable_sha256": digest(executable),
        "bundled_runtime_sha256": {
            name: digest(destination / name) for name in sorted(libraries)},
        "scope": "Relocatable alpha runtime with bundled GNU Fortran and OpenMP libraries",
    }
    (destination / "build.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def copy_third_party_licenses(plugin: Path) -> None:
    destination = plugin / "licenses"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "Clawpack-LICENSE.txt": ROOT / "solver" / "clawpack-v5.14.0" / "LICENSE",
    }
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise RuntimeError("gfortran is required to locate bundled runtime licenses")
    gcc_root = Path(compiler).resolve().parents[1]
    sources.update({
        "GCC-COPYING.txt": gcc_root / "COPYING",
        "GCC-COPYING.LIB.txt": gcc_root / "COPYING.LIB",
        "GCC-COPYING.RUNTIME.txt": gcc_root / "COPYING.RUNTIME",
    })
    for name, source in sources.items():
        if not source.is_file():
            raise RuntimeError(f"Missing third-party license: {source}")
        shutil.copy2(source, destination / name)


def write_manifest(plugin: Path, version: str, native: dict) -> None:
    build_info = {
        "format": "braids-qgis-alpha-build-v1",
        "plugin_id": PLUGIN_ID,
        "version": version,
        "channel": "alpha",
        "platform": distribution_tag(),
        "qgis_minimum_version": "3.40",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "native_executable_sha256": native["executable_sha256"],
    }
    (plugin / "BUILD_INFO.json").write_text(
        json.dumps(build_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {
        str(path.relative_to(plugin)): digest(path)
        for path in sorted(plugin.rglob("*")) if path.is_file()
    }
    manifest = {"format": "braids-qgis-alpha-manifest-v1", "files_sha256": files}
    (plugin / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def zip_plugin(plugin: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as output:
        for path in sorted(plugin.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(PLUGIN_ID) / path.relative_to(plugin)
            info = zipfile.ZipInfo(str(relative), date_time=(2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.name == "xbraids_cutcell" or path.suffix == ".dylib" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            output.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> None:
    version = metadata_version()
    tag = distribution_tag()
    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"BRAIDS-{version}-QGIS-{tag}.zip"
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    with tempfile.TemporaryDirectory(prefix="braids-alpha-") as temporary:
        plugin = Path(temporary) / PLUGIN_ID
        copy_python_tree(PLUGIN_SOURCE, plugin)
        copy_python_tree(CORE_SOURCE, plugin / "braids_core")
        native = make_relocatable(plugin / "bin" / tag)
        copy_third_party_licenses(plugin)
        write_manifest(plugin, version, native)
        zip_plugin(plugin, archive)
    checksum.write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "sha256": digest(archive),
                      "size_bytes": archive.stat().st_size, "version": version,
                      "platform": tag}, indent=2))


if __name__ == "__main__":
    main()
