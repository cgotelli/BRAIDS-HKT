#!/usr/bin/env python3
"""Build and run G3 hydraulics separately from the retained G2 executable.

Only new inlet cases are supported here. Source/build identity is checked before
execution; no historical benchmark or its outputs are touched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "solver"
VENDOR = ROOT / "vendor/avac-qgis"
VALIDATION = ROOT / "validation/hydraulics/inlets"
BUILD = VALIDATION / "build"
REPORTS = VALIDATION / "reports"
WAVE = BUILD / "wave"
CLAW = BUILD / "clawpack-v5.14.0"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_hashes(root: Path = SOURCE) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for directory in (root / "wave", root / "clawpack-v5.14.0")
        for path in sorted(directory.rglob("*"))
        if path.is_file() and (
            path.suffix.lower() in {".f", ".f90", ".f95", ".inc", ".h", ".c", ".py"}
            or path.name.lower().startswith("makefile")
        )
    }


def build() -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    # Only these generated build directories are replaced. Native G2 binaries,
    # input decks, reports and source-provenance records remain untouched.
    for name in ("wave", "clawpack-v5.14.0"):
        destination = BUILD / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(SOURCE / name, destination, ignore=shutil.ignore_patterns(
            "*.o", "*.mod", "*.pyc", "__pycache__", "xgeoclaw", "xgeoclaw.exe"))
    if source_hashes(BUILD) != before:
        raise RuntimeError("Source changed while staging the G3 build")
    fc = os.environ.get("FC") or shutil.which("gfortran")
    make = shutil.which("make")
    if not fc or not make:
        raise RuntimeError("G3 requires GNU Fortran and GNU Make")
    link = "-fopenmp -llapack -lblas"
    if sys.platform == "darwin" and Path("/opt/homebrew/opt/openblas/lib").is_dir():
        link = "-fopenmp -L/opt/homebrew/opt/openblas/lib -llapack -lblas"
    env = os.environ.copy()
    env.update(CLAW=str(CLAW), FC=fc, CLAW_PYTHON=sys.executable,
               OMP_NUM_THREADS="1", FFLAGS="-O2 -fopenmp", LFLAGS=link)
    command = [make, "-B", ".exe"]
    with (REPORTS / "build.log").open("w") as log:
        result = subprocess.run(command, cwd=WAVE, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"G3 build failed; see {REPORTS / 'build.log'}")
    if source_hashes() != before or source_hashes(BUILD) != before:
        raise RuntimeError("Source changed during G3 compilation; rebuild before qualification")
    executable = WAVE / "xgeoclaw"
    write_json(REPORTS / "build_provenance.json", {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": before, "executable_sha256": sha256(executable),
        "executable": str(executable), "command": command,
        "compiler": subprocess.check_output([fc, "--version"], text=True).splitlines()[0],
        "FFLAGS_environment": env["FFLAGS"], "LFLAGS": link,
        "makefile_flags": "See authenticated wave/Makefile for appended preprocessor symbols",
        "vendor_commit": subprocess.check_output(
            ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
    })
    print(f"Built isolated G3 executable: {executable}", flush=True)
    return executable


def check_build() -> dict:
    provenance = json.loads((REPORTS / "build_provenance.json").read_text())
    if provenance["source_sha256"] != source_hashes():
        raise RuntimeError("G3 source changed since compilation; rebuild first")
    if provenance["source_sha256"] != source_hashes(BUILD):
        raise RuntimeError("Staged G3 build source changed")
    if provenance["executable_sha256"] != sha256(WAVE / "xgeoclaw"):
        raise RuntimeError("G3 executable changed since compilation")
    return provenance


def configure_runtime():
    check_build()
    for path in (ROOT, VENDOR, VENDOR / "validation"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runtime = importlib.import_module("avac4qgis_validation.runtime")
    if getattr(runtime, "_braids_g3_runtime", False):
        return runtime
    runtime.CLAWPACK_SOURCE = CLAW
    runtime.SOURCE_ROOTS = {"wave": WAVE}
    runtime.BUILD_STAMP_ROOT = REPORTS / "build_stamps"

    def source(kind):
        if kind != "wave":
            raise ValueError("G3 only executes WAVE")
        return WAVE

    runtime.runtime = source
    original_setrun = runtime._run_setrun

    def setrun(kind, case, qinit, **kwargs):
        (case / "impulse_configuration.yaml").rename(case / "hydraulic_configuration.yaml")
        return original_setrun(kind, case, qinit, **kwargs)

    runtime._run_setrun = setrun

    def no_source(path, entries, times):
        if entries:
            raise ValueError("G3 inlet cases require no internal shoreline forcing")

    runtime._write_internal_inflow = no_source
    original_prepare = runtime.prepare_wave_hydraulic_case

    def prepare(*args, **kwargs):
        work = original_prepare(*args, **kwargs)
        for name in ("AVAC", "CL"):
            unused = work.parent / name
            if unused.exists():
                shutil.rmtree(unused)
        return work

    runtime.prepare_wave_hydraulic_case = prepare
    runtime._activate_packaged_clawpack(CLAW)
    runtime._braids_g3_runtime = True
    return runtime


def run_native(work: Path, *, cores: int = 1) -> dict:
    runtime = configure_runtime()
    before = check_build()
    input_paths = {path for path in work.iterdir() if path.is_file() and
                   (path.suffix in {".data", ".xyz"} or path.name == "braids_boundary.data.json")}
    input_paths.update(path for path in work.parent.rglob("*") if path.is_file() and
                       path.suffix in {".asc", ".yaml"})
    claw_text = (work / "claw.data").read_text()
    if re.search(r"^\s*T\s*=:\s*restart\s*$", claw_text, re.MULTILINE):
        filename = re.search(r"^\s*['\"]([^'\"]+)['\"]\s*=:\s*restart_file\s*$", claw_text, re.MULTILINE)
        if not filename:
            raise RuntimeError("Missing checkpoint path for restart provenance")
        input_paths.add(work / filename.group(1))
    inputs = {os.path.relpath(path, work): sha256(path) for path in sorted(input_paths)}
    try:
        metrics = runtime.run_solver("wave", work, cores=cores,
                                     executable_override=WAVE / "xgeoclaw")
    except RuntimeError:
        # Preserve diagnostic evidence before a corrected case is regenerated.
        archive = REPORTS / "development_failures" / work.parent.name / before["executable_sha256"]
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("solver.log", "solver_time.log", "solver_execution_metrics.json", "braids_boundary.data"):
            path = work / name
            if path.is_file():
                shutil.copy2(path, archive / name)
        write_json(archive / "context.json", {"input_sha256": inputs,
                                              "build_provenance": before})
        log = work / "solver.log"
        detail = "\n".join(log.read_text().splitlines()[-15:]) if log.exists() else "No solver log"
        raise RuntimeError(f"G3 native execution failed in {work}:\n{detail}") from None
    if check_build() != before:
        raise RuntimeError("G3 source/build identity changed during native execution")
    if any(sha256(work / name) != digest for name, digest in inputs.items()):
        raise RuntimeError("G3 numerical inputs changed during execution")
    metrics["input_sha256"] = inputs
    metrics["native_artifact_sha256"] = {
        path.name: sha256(path) for path in sorted(work.iterdir())
        if path.is_file() and (re.fullmatch(r"fort\.[tqba]\d+", path.name)
                               or path.name in {"braids_flux_faces.csv", "braids_flux_steps.csv",
                                                "braids_boundary_diagnostics.csv", "braids_flux_restart.csv"})}
    metrics["build_provenance_sha256"] = sha256(REPORTS / "build_provenance.json")
    write_json(work / "g3_execution.json", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args()
    if args.action == "build":
        build()
    else:
        check_build()
        print("G3 source, staged build and executable identities match")
