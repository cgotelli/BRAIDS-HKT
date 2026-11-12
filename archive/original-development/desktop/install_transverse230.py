"""Install the qualified two-direction Jiang--Cao capacity reconstruction.

This publishes Desktop preview51 and updates only the frozen runtime carried by
the Jiang--Cao experiment project.  Existing preview50 runs and checkpoints are
preserved as evidence and are deliberately incompatible with the new runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "desktop/runtime"
CANDIDATE = ROOT / "validation/_transverse230/candidate_2d_compact"
OUT = ROOT / "validation/_transverse230"
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
OLD = "63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9"
NEW = "a882ea4f3b07cdbb2e4c53c637ab62830dc94b8fc17f7f5e4c52bb3e5212cbe1"
SOURCE = "e89341080f14ed1f87ea30e790784b43b086ede2a1ef55083c5aed20d00b71a2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def verify(root: Path) -> dict:
    manifest = json.loads((root / "RUNTIME.json").read_text(encoding="utf8"))
    for name, digest in manifest["files"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Runtime payload differs: {root / name}")
    return manifest


def processes() -> str:
    command = (
        "Get-CimInstance Win32_Process | Where-Object {"
        "($_.Name -match '^python|^BRAIDS') -and "
        "($_.CommandLine -match 'desktop[\\\\/]launch.py|"
        "desktop[\\\\/]worker.py|braids_desktop.app')} | "
        "Select-Object -ExpandProperty ProcessId"
    )
    return subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", command], text=True
    ).strip()


def publish(target: Path, backup: Path, changed: set[str], added: set[str]) -> None:
    if backup.exists():
        raise FileExistsError(f"Preserve existing backup: {backup}")
    for name in sorted(changed | {"RUNTIME.json"}):
        destination = backup / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / name, destination)
    for name in sorted(changed | added):
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CANDIDATE / name, destination)
    shutil.copy2(CANDIDATE / "RUNTIME.json", target / "RUNTIME.json")
    verify(target)
    if sha(target / "RUNTIME.json") != NEW:
        raise ValueError("Published runtime identity differs")


def main() -> None:
    if processes():
        raise RuntimeError("Close BRAIDS Desktop before installation")
    if sha(BASE / "RUNTIME.json") != OLD:
        raise ValueError("Installed Desktop runtime is not preview50")
    if sha(CANDIDATE / "RUNTIME.json") != NEW:
        raise ValueError("Candidate runtime differs from the qualified build")
    old = verify(BASE)
    new = verify(CANDIDATE)
    old_names, new_names = set(old["files"]), set(new["files"])
    removed = old_names - new_names
    added = new_names - old_names
    changed = {name for name in old_names & new_names
               if old["files"][name] != new["files"][name]}
    expected_changed = {
        "PRECOMPILED.json",
        "braids_core/sediment/jiang_cao_coupled.py",
        "braids_core/sediment/jiang_cao_coupled.h",
    }
    build = json.loads((OUT / "BUILD_2D_COMPACT.json").read_text(encoding="utf8"))
    if removed or changed != expected_changed or added != set(build["added"]):
        raise ValueError({"changed": sorted(changed), "added": sorted(added),
                          "removed": sorted(removed)})

    publish(BASE, OUT / "preview50_backup", changed, added)
    frozen = PROJECT / "runtime"
    if sha(frozen / "RUNTIME.json") != OLD:
        raise ValueError("Project frozen runtime is not preview50")
    verify(frozen)
    publish(frozen, OUT / "project_preview50_backup", changed, added)

    report = (
        "validation/sediment/river_morphodynamics/reports/"
        "desktop_experiment225/transverse230/REPORT.md"
    )
    provenance_path = ROOT / "desktop/PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    provenance.update(version="0.1.0-preview51", ui_qualification=report)
    provenance["solver"].update(
        source="validation/_transverse230/candidate_2d_compact",
        runtime_manifest_sha256=NEW,
        qualification=report,
        modification_scope=(
            "Retain preview50 CFL-0.5 stage-wave controller and Jiang-Cao "
            "physics. Reconstruct the capacity forcing through compact "
            "shared-face passes along and across local flow so a transverse "
            "one-cell mode cannot remain uncoupled. Physical capacity, "
            "exchange, secondary-flow and flux formulas are unchanged."
        ),
    )
    save(provenance_path, provenance)
    receipt = {
        "version": "0.1.0-preview51",
        "runtime": NEW,
        "source_identity": SOURCE,
        "previous": OLD,
        "changed": sorted(changed),
        "added": sorted(added),
        "desktop_backup": str(OUT / "preview50_backup"),
        "project_backup": str(OUT / "project_preview50_backup"),
        "physical_formulas_changed": False,
        "project_frozen_runtime_updated": True,
        "old_checkpoints_supported": False,
        "QGIS_changed": False,
        "Delft3D_modified": False,
    }
    save(OUT / "INSTALL.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
