"""Install the qualified Jiang--Cao face-compatible capacity correction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "desktop/runtime"
CANDIDATE = ROOT / "validation/_checkerboard227/candidate_v3"
BUILD = ROOT / "validation/_checkerboard227/BUILD.json"
OUT = ROOT / "validation/_checkerboard228"
REPORT = (ROOT / "validation/sediment/river_morphodynamics/reports/"
          "desktop_experiment225/checkerboard_delivery228")
EXPECTED_BASE = "9caa8c7cbea26bb8b98efe4efd37923985845d731e75d213e31a3a6ec498b054"
EXPECTED_CANDIDATE = "f8f896a590b38f0a0563b2d04fb6db823948a42ad688829849a4dfab370da4c5"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def load_manifest(root: Path) -> dict:
    return json.loads((root / "RUNTIME.json").read_text(encoding="utf8"))


def verify(root: Path, *, progress: bool = False) -> dict:
    manifest = load_manifest(root)
    rows = list(manifest["files"].items())
    for index, (name, digest) in enumerate(rows, 1):
        path = root / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Runtime payload differs: {root}: {name}")
        if progress and index % 500 == 0:
            print(f"Verified {index}/{len(rows)} files in {root}", flush=True)
    return manifest


def desktop_processes() -> str:
    command = (
        "Get-CimInstance Win32_Process | Where-Object {"
        "($_.Name -match '^python|^BRAIDS') -and "
        "($_.CommandLine -match 'desktop[\\\\/]launch.py|desktop[\\\\/]worker.py|braids_desktop.app')"
        "} | Select-Object -ExpandProperty ProcessId"
    )
    return subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", command], text=True
    ).strip()


def main() -> None:
    if sha(BASE / "RUNTIME.json") != EXPECTED_BASE:
        raise ValueError("Installed runtime is not the qualified preview48 base")
    if sha(CANDIDATE / "RUNTIME.json") != EXPECTED_CANDIDATE:
        raise ValueError("Candidate runtime identity differs")
    if desktop_processes():
        raise RuntimeError("Save and close BRAIDS Desktop before installation")

    print("Verifying preview48 and candidate payloads...", flush=True)
    old = verify(BASE, progress=True)
    new = verify(CANDIDATE, progress=True)
    build = json.loads(BUILD.read_text(encoding="utf8"))
    changed = {name for name, digest in old["files"].items()
               if new["files"].get(name) != digest}
    added = set(new["files"]) - set(old["files"])
    expected_changed = set(build["changed"]) | {"PRECOMPILED.json"}
    if changed != expected_changed or added != set(build["added"]):
        raise ValueError({"changed": sorted(changed), "added": sorted(added)})

    backup = OUT / "preview48_backup"
    if backup.exists():
        raise FileExistsError(f"Preserve existing installation evidence: {backup}")
    for name in sorted(changed | {"RUNTIME.json"}):
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BASE / name, target)
    shutil.copy2(ROOT / "desktop/PROVENANCE.json", backup / "DESKTOP_PROVENANCE.json")

    for name in sorted(changed | added):
        target = BASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CANDIDATE / name, target)
    shutil.copy2(CANDIDATE / "RUNTIME.json", BASE / "RUNTIME.json")
    installed = verify(BASE, progress=True)
    if sha(BASE / "RUNTIME.json") != EXPECTED_CANDIDATE:
        raise ValueError("Installed manifest identity differs after publication")

    provenance_path = ROOT / "desktop/PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf8"))
    qualification = str((REPORT / "REPORT.md").relative_to(ROOT)).replace("\\", "/")
    provenance.update(version="0.1.0-preview49", ui_qualification=qualification)
    provenance["solver"].update(
        source=str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"),
        runtime_manifest_sha256=EXPECTED_CANDIDATE,
        qualification=qualification,
        modification_scope=(
            "Jiang-Cao capacity-only face-compatible pair reconstruction and robust "
            "affine-exact flow-aligned bed derivative. Physical hydraulic state, bed, "
            "capacity/exchange/resistance/secondary/repose laws, ordinary formula paths, "
            "QGIS and Delft3D remain unchanged. Public project v36; previous prepared "
            "artifacts and checkpoints are rejected by runtime identity."
        ),
    )
    save(provenance_path, provenance)
    receipt = {
        "version": "0.1.0-preview49",
        "runtime": EXPECTED_CANDIDATE,
        "previous": EXPECTED_BASE,
        "source_identity_pending_worker_check": True,
        "backup": str(backup),
        "changed": sorted(changed),
        "added": sorted(added),
        "new_binding": build["new_binding"],
        "manifest_files": len(installed["files"]),
        "physical_formulas_changed": False,
        "user_projects_modified": False,
        "QGIS_changed": False,
        "Delft3D_modified": False,
    }
    save(OUT / "INSTALL.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
