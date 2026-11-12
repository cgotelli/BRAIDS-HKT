#!/usr/bin/env python3
"""Install a verified BRAIDS Windows ZIP with a recoverable directory swap."""

from __future__ import annotations

import argparse
import configparser
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from typing import Any, Callable
import zipfile

import build_qgis_windows as packager


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "validation/sediment/johannot2026_braided_flume"
sys.path.insert(0, str(CASE))
from atomic_status import replace_with_windows_retry, write_json_atomic  # noqa: E402

DEFAULT_REPORT = CASE / "reports/windows_release_installation.json"
PLUGIN_ID = "braids_qgis"


def default_destination() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is unavailable; pass --destination")
    return (
        Path(appdata) / "QGIS/QGIS3/profiles/default/python/plugins" / PLUGIN_ID
    )


def qgis_processes() -> list[str]:
    completed = subprocess.run(
        ["tasklist.exe", "/FO", "CSV", "/NH"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot determine whether QGIS is running")
    names = []
    for row in csv.reader(completed.stdout.splitlines()):
        if row and "qgis" in row[0].lower():
            names.append(row[0])
    return sorted(set(names))


def _metadata_version(plugin: Path) -> str:
    parser = configparser.ConfigParser()
    if not parser.read(plugin / "metadata.txt", encoding="utf-8"):
        raise RuntimeError("staged plugin metadata.txt is missing")
    return parser.get("general", "version").strip()


def _safe_members(zipped: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = zipped.infolist()
    for member in members:
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive member: {member.filename}")
        if not path.parts or path.parts[0] != PLUGIN_ID:
            raise RuntimeError(f"archive member escapes {PLUGIN_ID}: {member.filename}")
    return members


def verify_installed_plugin(
    plugin: Path, expected_version: str, *,
    allow_generated_python_cache: bool = True,
) -> dict[str, Any]:
    """Verify an installed tree without treating QGIS bytecode as payload.

    A clean staging tree is checked with ``allow_generated_python_cache=False``
    during installation.  Post-use audits may ignore only ordinary
    ``__pycache__/*.py[co]`` files; every manifested byte and every other extra
    file remains fail-closed.
    """
    manifest_path = plugin / "MANIFEST.json"
    files = packager.verify_manifest_files(
        plugin, context="Installed plugin",
        allow_generated_python_cache=allow_generated_python_cache,
    )
    sources = packager.verify_packaged_source_manifest(
        plugin, files, context="Installed plugin",
    )
    version = _metadata_version(plugin)
    if version != expected_version:
        raise RuntimeError(
            f"installed version {version} differs from {expected_version}"
        )
    forbidden = packager.forbidden_archive_members([
        (Path(PLUGIN_ID) / path.relative_to(plugin)).as_posix()
        for path in plugin.rglob("*") if path.is_file()
    ])
    if forbidden:
        raise RuntimeError("installed plugin contains forbidden run/source content")
    generated_cache_files = sorted(
        path.relative_to(plugin).as_posix()
        for path in plugin.rglob("*")
        if path.name != "MANIFEST.json"
        and path.relative_to(plugin).as_posix() not in files
        and packager.is_generated_python_cache(path, plugin)
    )
    return {
        "version": version,
        "files_verified": len(files),
        "source_files_verified": len(sources),
        "source_sha256": sources,
        "generated_python_cache_files_ignored": len(generated_cache_files),
        "manifest_sha256": packager.digest(manifest_path),
        "native_executable_sha256": packager.digest(
            plugin / "bin/windows-x86_64/xbraids_cutcell.exe"
        ),
    }


def installation_qualification_audit(
    record: dict[str, Any], *,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct source and archive lineage for an installed plugin."""
    installed = (
        record.get("installation_verification", {})
        if verification is None else verification
    )
    archive = record.get("archive_verification", {})
    sources = installed.get("source_sha256")
    checks = {
        "installed_manifest_was_reverified": (
            isinstance(installed, dict)
            and bool(installed)
            and installed == record.get("installation_verification")
        ),
        "installed_source_manifest_is_complete": (
            isinstance(sources, dict) and bool(sources)
            and installed.get("source_files_verified") == len(sources)
        ),
        "installed_sources_match_current_tree": (
            sources == packager.current_packaged_source_sha256()
        ),
        "archive_and_install_source_counts_match": (
            isinstance(archive, dict)
            and archive.get("source_files_verified") == len(sources or {})
        ),
    }
    return {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def install_archive(
    archive: Path, destination: Path, *, expected_version: str,
    backup_root: Path | None = None,
    archive_verifier: Callable[[Path], dict[str, Any]] = packager.verify_archive,
) -> dict[str, Any]:
    archive = archive.resolve()
    destination = destination.resolve()
    if destination.name != PLUGIN_ID:
        raise RuntimeError(f"destination must end in {PLUGIN_ID}")
    if not archive.is_file():
        raise FileNotFoundError(f"release archive is missing: {archive}")
    archive_evidence = archive_verifier(archive)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    backup_root = (
        parent / ".braids_release_backups" if backup_root is None
        else backup_root.resolve()
    )
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_root / f"{PLUGIN_ID}_pre_{expected_version}_{stamp}"
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")

    with tempfile.TemporaryDirectory(
        prefix=".braids_release_stage_", dir=parent
    ) as temporary:
        staging_root = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            members = _safe_members(zipped)
            zipped.extractall(staging_root, members=members)
        staged = staging_root / PLUGIN_ID
        staged_verification = verify_installed_plugin(
            staged, expected_version, allow_generated_python_cache=False,
        )
        staged_audit = installation_qualification_audit({
            "archive_verification": archive_evidence,
            "installation_verification": staged_verification,
        })
        if staged_audit["status"] != "pass":
            raise RuntimeError("staged plugin source qualification failed")
        old_moved = False
        try:
            if destination.exists():
                replace_with_windows_retry(destination, backup)
                old_moved = True
            replace_with_windows_retry(staged, destination)
        except BaseException:
            if old_moved and not destination.exists() and backup.exists():
                replace_with_windows_retry(backup, destination)
            raise

    installed = verify_installed_plugin(destination, expected_version)
    record = {
        "schema_version": 1,
        "status": "pass",
        "installed_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_sha256": packager.digest(archive),
        "archive_verification": archive_evidence,
        "destination": str(destination),
        "backup": str(backup) if backup.exists() else None,
        "installation_verification": installed,
    }
    record["qualification_audit"] = installation_qualification_audit(record)
    if record["qualification_audit"]["status"] != "pass":
        raise RuntimeError("installed plugin source qualification failed")
    return record


def publish_installation_report(path: Path, record: dict[str, Any]) -> None:
    """Publish a complete report without sharing a fixed partial filename."""
    write_json_atomic(path, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        running = qgis_processes()
        if running:
            raise RuntimeError(
                "close QGIS before installing the release; running: "
                + ", ".join(running)
            )
        result = install_archive(
            args.archive, args.destination or default_destination(),
            expected_version=args.expected_version,
        )
        publish_installation_report(args.report, result)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
