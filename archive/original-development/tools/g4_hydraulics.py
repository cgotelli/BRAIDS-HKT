#!/usr/bin/env python3
"""Execute G4 channel cases against the preserved, authenticated G3 build."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import g3_hydraulics as base
from validation.hydraulics.channels import ledger_storage

VALIDATION = ROOT / "validation/hydraulics/channels"
REPORTS = VALIDATION / "reports"
RUNS = VALIDATION / "runs"
BUILD = VALIDATION / "build"
OVERLAY = VALIDATION / "native"
SHARED_OVERLAY_TARGETS = {
    name: f"clawpack-v5.14.0/geoclaw/src/2d/shallow/{name}"
    for name in ("stepgrid.f", "tick.f", "advanc.f")
}
WAVE_OVERLAY_TARGETS = {
    "Makefile", "bc2amr.f90", "braids_boundary_module.f90",
    "braids_draining_module.f90", "braids_flux_module.f90",
    "braids_positivity_module.f90", "hydraulic_bc_module.f90",
}
sha256 = base.sha256
write_json = base.write_json


def baseline_identity():
    build = base.check_build()
    qualification = base.REPORTS / "verification_g3.json"
    record = json.loads(qualification.read_text())
    if record["status"] != "PASS" or record["executable_sha256"] != build["executable_sha256"]:
        raise RuntimeError("G4 requires the preserved qualified G3 executable")
    return {"build_provenance_sha256": sha256(base.REPORTS / "build_provenance.json"),
            "g3_verification_sha256": sha256(qualification),
            "executable_sha256": build["executable_sha256"]}


def overlay_hashes():
    return {path.name: sha256(path) for path in sorted(OVERLAY.glob("*"))
            if path.is_file() and (path.suffix.lower() in {".f", ".f90", ".inc"} or path.name == "Makefile")}


def overlay_targets(names=None):
    """Map only declared G4 additions/replacements into the isolated build."""
    result = {}
    for name in overlay_hashes() if names is None else names:
        if name in SHARED_OVERLAY_TARGETS:
            target = SHARED_OVERLAY_TARGETS[name]
        elif name in WAVE_OVERLAY_TARGETS:
            target = f"wave/{name}"
        else:
            raise RuntimeError(f"Unregistered G4 source replacement: {name}")
        if Path(name).name != name or target in result.values():
            raise RuntimeError(f"Invalid/duplicate G4 source target: {name}")
        result[name] = target
    return result


def overlay_source_hashes():
    hashes = overlay_hashes()
    return {target: hashes[name] for name, target in overlay_targets(hashes).items()}


def frozen_build():
    baseline = baseline_identity()
    path = REPORTS / "build_provenance.json"
    if not path.exists():
        return {"baseline": baseline, "executable_sha256": baseline["executable_sha256"],
                "executable": str(base.WAVE / "xgeoclaw"), "overlay": None}
    record = json.loads(path.read_text())
    if record["baseline"] != baseline or record["overlay_sha256"] != overlay_hashes():
        raise RuntimeError("G4 native overlay changed; rebuild its isolated executable")
    if record["source_sha256"] != base.source_hashes(BUILD) or record["executable_sha256"] != sha256(BUILD / "wave/xgeoclaw"):
        raise RuntimeError("G4 staged source or executable changed")
    return {"baseline": baseline, "executable_sha256": record["executable_sha256"],
            "executable": str(BUILD / "wave/xgeoclaw"), "overlay": record["overlay_sha256"],
            "build_provenance_sha256": sha256(path)}


def build():
    baseline = baseline_identity()
    overrides = overlay_hashes()
    if not overrides:
        raise RuntimeError("No G4 native overlay to compile")
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name in ("wave", "clawpack-v5.14.0"):
        destination = BUILD / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(base.BUILD / name, destination, ignore=shutil.ignore_patterns(
            "*.o", "*.mod", "*.pyc", "__pycache__", "xgeoclaw", "xgeoclaw.exe"))
    for name, relative_target in overlay_targets(overrides).items():
        target = BUILD / relative_target
        shutil.copy2(OVERLAY / name, target)
    sources = base.source_hashes(BUILD)
    fc = os.environ.get("FC") or shutil.which("gfortran")
    make = shutil.which("make")
    if not fc or not make:
        raise RuntimeError("GNU Fortran and GNU Make are required")
    link = "-fopenmp -llapack -lblas"
    if sys.platform == "darwin" and Path("/opt/homebrew/opt/openblas/lib").is_dir():
        link = "-fopenmp -L/opt/homebrew/opt/openblas/lib -llapack -lblas"
    env = {**os.environ, "CLAW": str(BUILD / "clawpack-v5.14.0"), "FC": fc,
           "CLAW_PYTHON": sys.executable, "OMP_NUM_THREADS": "1", "FFLAGS": "-O2 -fopenmp", "LFLAGS": link}
    with (REPORTS / "build.log").open("w") as log:
        compiled = subprocess.run([make, "-B", ".exe"], cwd=BUILD / "wave", env=env, stdout=log, stderr=subprocess.STDOUT)
    if compiled.returncode:
        raise RuntimeError(f"G4 build failed; see {REPORTS / 'build.log'}")
    if baseline_identity() != baseline or overlay_hashes() != overrides or base.source_hashes(BUILD) != sources:
        raise RuntimeError("Sources changed while compiling G4")
    write_json(REPORTS / "build_provenance.json", {
        "baseline": baseline, "overlay_sha256": overrides, "source_sha256": sources,
        "executable_sha256": sha256(BUILD / "wave/xgeoclaw"),
        "compiler": subprocess.check_output([fc, "--version"], text=True).splitlines()[0],
        "FFLAGS": env["FFLAGS"], "LFLAGS": link,
        "built_utc": datetime.now(timezone.utc).isoformat()})
    print(f"Built isolated G4 executable: {BUILD / 'wave/xgeoclaw'}", flush=True)


def input_hashes(work: Path):
    paths = {p for p in work.iterdir() if p.is_file() and
             (p.suffix in {".data", ".xyz"} or p.name in {"braids_boundary.data.json", "g4_case.json"})}
    paths.update(p for p in work.parent.rglob("*") if p.is_file() and p.suffix in {".asc", ".yaml"})
    parameters = {line.split("=:", 1)[1].split()[0]: line.split("=:", 1)[0].strip()
                  for line in (work / "claw.data").read_text().splitlines() if "=:" in line}
    if parameters.get("restart") == "T":
        paths.add(work / parameters["restart_file"].strip("'\""))
    return {os.path.relpath(p, work): sha256(p) for p in sorted(paths)}


def artifact_hashes(work: Path):
    """Authenticate logical CSV artifacts regardless of completed storage form."""
    work = Path(work)
    result = {p.name: sha256(p) for p in sorted(work.iterdir()) if p.is_file() and
              (re.fullmatch(r"fort\.[tqba]\d+", p.name) or
               p.name in {"solver.log", "solver_execution_metrics.json"})}
    names = {p.name.removesuffix(".gz") for p in work.iterdir()
             if p.name.startswith("braids_") and (p.name.endswith(".csv") or p.name.endswith(".csv.gz"))}
    if (work / ledger_storage.MANIFEST_NAME).exists():
        names.add(ledger_storage.FACE_NAME)
    for name in sorted(names):
        result[name] = ledger_storage.logical_sha256(work / name)
    record = ledger_storage.storage_record(work)
    execution_path = work / "g4_execution.json"
    if record is not None and execution_path.exists():
        execution = json.loads(execution_path.read_text())
        if "ledger_storage" in execution and (execution["ledger_storage"] != record or
                execution.get("ledger_storage_manifest_sha256") != sha256(work / ledger_storage.MANIFEST_NAME)):
            raise RuntimeError("Compressed ledger storage differs from execution provenance")
    return dict(sorted(result.items()))


def run_native(work: Path, *, cores=1):
    if ledger_storage.storage_record(work) is not None:
        raise RuntimeError("Restore the completed face ledger before native restart/appending")
    identity = frozen_build()
    runtime = base.configure_runtime()
    inputs = input_hashes(work)
    execution = {"status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
                 "inherited_build": identity, "input_sha256": inputs,
                 "preparation_source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                     Path(__file__), VALIDATION / "cases.py", VALIDATION / "inlet_io.py", VALIDATION / "ledger_storage.py", ROOT / "tools/g3_hydraulics.py",
                     base.VENDOR / "validation/avac4qgis_validation/runtime.py")}}
    write_json(work / "g4_execution.json", execution)
    try:
        metrics = runtime.run_solver("wave", work, cores=cores, executable_override=Path(identity["executable"]))
        if frozen_build() != identity or input_hashes(work) != inputs:
            raise RuntimeError("G4 source, executable, retained evidence or inputs changed during execution")
        if any(sha256(ROOT / name) != digest for name, digest in execution["preparation_source_sha256"].items()):
            raise RuntimeError("G4 preparation sources changed during execution")
    except BaseException as failure:
        metrics_path = work / "solver_execution_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"returncode": -1}
        execution.update(status="FAIL", error=f"{type(failure).__name__}: {failure}", metrics=metrics,
                         completed_utc=datetime.now(timezone.utc).isoformat(),
                         native_artifact_sha256=artifact_hashes(work))
        write_json(work / "g4_execution.json", execution)
        raise
    execution.update(metrics=metrics, status="PASS",
                     completed_utc=datetime.now(timezone.utc).isoformat(),
                     native_artifact_sha256=artifact_hashes(work))
    write_json(work / "g4_execution.json", execution)
    # Publish the completed raw artifact identity before archiving. Storage is
    # independently authenticated and never changes the scientific CSV hashes.
    if ledger_storage.ledger_exists(work / ledger_storage.FACE_NAME):
        try:
            record = ledger_storage.compress_completed_faces(work,
                expected_sha256=execution["native_artifact_sha256"][ledger_storage.FACE_NAME],
                expected_input_sha256=inputs, input_hashes=input_hashes)
            if artifact_hashes(work) != execution["native_artifact_sha256"]:
                raise RuntimeError("Logical native artifacts changed during ledger compression")
            execution.update(ledger_storage=record,
                ledger_storage_manifest_sha256=sha256(work / ledger_storage.MANIFEST_NAME))
        except BaseException as failure:
            execution.update(status="FAIL", error=f"Completed-ledger storage failed: {failure}")
            write_json(work / "g4_execution.json", execution)
            raise
        write_json(work / "g4_execution.json", execution)
    return execution


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    arguments = parser.parse_args()
    if arguments.action == "build":
        build()
    else:
        print(json.dumps(frozen_build(), indent=2))
