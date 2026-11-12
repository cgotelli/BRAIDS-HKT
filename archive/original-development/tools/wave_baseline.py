#!/usr/bin/env python3
"""Extract, build and reproduce the pinned WAVE hydraulic baseline headlessly."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PIN = "cfbaaf4d2c6a8c4bbb648dcfb5749fe488b30b32"
UPSTREAM = "https://github.com/cgotelli/AVAC-QGIS.git"
VENDOR = ROOT / "vendor/avac-qgis"
SOLVER = ROOT / "solver"
WAVE = SOLVER / "wave"
CLAW = SOLVER / "clawpack-v5.14.0"
VALIDATION = ROOT / "validation/hydraulics/wave_baseline"
REPORTS = VALIDATION / "reports"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    """Authenticate all native sources/build controls and Python data writers."""
    return {str(path.relative_to(SOLVER)): sha256(path)
            for directory in (WAVE, CLAW)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and (path.suffix.lower() in {".f", ".f90", ".f95", ".inc", ".h", ".c", ".py"}
                                   or path.name.lower().startswith("makefile"))}


def json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def archive_source(path):
    """Keep the exact Python source associated with an execution-time hash."""
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    destination = REPORTS / "executed_sources" / (digest + ".py")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise RuntimeError("Source archive hash collision")
    destination.write_bytes(payload)
    return digest, str(destination.relative_to(REPORTS))


def git(*args):
    return subprocess.check_output(["git", "-C", str(VENDOR), *args], text=True).strip()


SETPROB = '''! BRAIDS water-only default; the original shoreline source is explicit opt-in.
subroutine setprob
    use internal_inflow_module, only: read_internal_inflow
    use hydraulic_bc_module, only: setup_hydraulic_bc
    implicit none
    character(len=1024) :: inflow_file
    integer :: unit, ios
    logical :: exists

    ! A hydraulic run needs no setprob.data, CL directory or AVAC outputs.
    inquire(file="setprob.data", exist=exists)
    if (exists) then
        unit = 72
        call opendatafile(unit, "setprob.data")
        read(unit,*,iostat=ios) inflow_file
        close(unit)
        if (ios /= 0) then
            write(6,*) "ERROR: malformed optional setprob.data"
            stop 1
        endif
        if (trim(inflow_file) /= "none") then
            call read_internal_inflow(trim(inflow_file))
        endif
    endif
    ! Existing rectangular benchmark boundaries; not BRAIDS's section API.
    call setup_hydraulic_bc()
end subroutine setprob
'''


def prepare():
    if not VENDOR.exists():
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM, str(VENDOR)], check=True)
        subprocess.run(["git", "-C", str(VENDOR), "checkout", "--detach", PIN], check=True)
    if git("rev-parse", "HEAD") != PIN or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Vendor checkout must be clean at the declared pin; refusing to overwrite it.")
    if WAVE.exists() or CLAW.exists():
        if not (SOLVER / "source_provenance.json").exists():
            raise RuntimeError("Existing extraction has no manifest; refusing to overwrite source.")
        return
    SOLVER.mkdir(exist_ok=True)
    for original, target in ((VENDOR / "avac-main/src/WAVE", WAVE),
                             (VENDOR / "avac-main/clawpack-v5.14.0", CLAW)):
        shutil.copytree(original, target, ignore=shutil.ignore_patterns("*.o", "*.mod", "__pycache__", "xgeoclaw"))
    (WAVE / "setprob.f90").write_text(SETPROB)
    setrun = WAVE / "setrun.py"
    contents = setrun.read_text()
    contents = contents.replace('proj_dir / "impulse_configuration.yaml"', 'proj_dir / "hydraulic_configuration.yaml"')
    old = '''    probdata.add_param('inflow_file', str(proj_dir / "CL" / "internal_inflow.data"),
                       'Internal AVAC-to-WAVE shoreline inflow')'''
    new = '''    # Water-only is the default. A source file must be explicitly requested.
    probdata.add_param('inflow_file', str(config.get('internal_inflow_file', 'none')),
                       'Optional shoreline source; none for hydraulic runs')'''
    if old not in contents:
        raise RuntimeError("Pinned setrun coupling block differs from audited source")
    setrun.write_text(contents.replace(old, new))
    modified = []
    patches = []
    for name in ("setprob.f90", "setrun.py"):
        original = VENDOR / "avac-main/src/WAVE" / name
        extracted = WAVE / name
        modified.append({"file": f"wave/{name}", "upstream_sha256": sha256(original),
                         "extracted_sha256": sha256(extracted)})
        patches.extend(difflib.unified_diff(original.read_text().splitlines(True), extracted.read_text().splitlines(True),
                                          fromfile=f"a/wave/{name}", tofile=f"b/wave/{name}"))
    (SOLVER / "wave_hydraulic.patch").write_text("".join(patches))
    manifest = {}
    for source in (WAVE, CLAW):
        for path in sorted(source.rglob("*")):
            if path.is_file():
                manifest[str(path.relative_to(SOLVER))] = sha256(path)
    json_write(SOLVER / "source_provenance.json", {
        "upstream_url": UPSTREAM, "upstream_commit": PIN, "files": manifest,
        "modified_files": modified, "modified_clawpack_files": [],
        "patch_sha256": sha256(SOLVER / "wave_hydraulic.patch"),
        "notes": "Complete upstream modified GeoClaw tree retained. Only two WAVE setup files changed; optional shoreline source module preserved unchanged.",
    })


def build():
    prepare()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fc = os.environ.get("FC") or shutil.which("gfortran")
    make = shutil.which("make")
    if not fc or not make:
        raise RuntimeError("Install gfortran and GNU make to build the native solver")
    environment = os.environ.copy()
    environment.update(CLAW=str(CLAW), FC=fc, CLAW_PYTHON=sys.executable, OMP_NUM_THREADS="1")
    flags = "-O2 -fopenmp"
    link = "-fopenmp -llapack -lblas"
    if sys.platform == "darwin" and Path("/opt/homebrew/opt/openblas/lib").exists():
        link = "-fopenmp -L/opt/homebrew/opt/openblas/lib -llapack -lblas"
    environment.update(FFLAGS=flags, LFLAGS=link)
    command = [make, "-B", ".exe"]
    with (REPORTS / "build.log").open("w") as log:
        result = subprocess.run(command, cwd=WAVE, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"WAVE build failed; see {REPORTS / 'build.log'}")
    executable = WAVE / "xgeoclaw"
    json_write(REPORTS / "build_provenance.json", {
        "upstream_commit": PIN, "built_utc": datetime.now(timezone.utc).isoformat(),
        "executable": str(executable), "executable_sha256": sha256(executable),
        "source_manifest_sha256": sha256(SOLVER / "source_provenance.json"),
        "source_files_sha256": source_hashes(),
        "compiler": subprocess.check_output([fc, "--version"], text=True).splitlines()[0],
        "make": subprocess.check_output([make, "--version"], text=True).splitlines()[0],
        "platform": platform.platform(), "python": sys.version,
        "command": command, "FFLAGS": flags + " -cpp -DWAVE_CFL_FAST_PATH", "LFLAGS": link,
    })
    print(f"Built {executable}", flush=True)


def configure_runtime():
    """Point the unmodified upstream drivers only at the extracted WAVE binary."""
    if git("rev-parse", "HEAD") != PIN or git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Validation requires the clean pinned upstream driver checkout")
    build = json.loads((REPORTS / "build_provenance.json").read_text())
    if build["source_files_sha256"] != source_hashes():
        raise RuntimeError("Source changed since build; run tools/wave_baseline.py build")
    for path in (VENDOR, VENDOR / "validation", VENDOR / "validation/WAVE", VALIDATION):
        sys.path.insert(0, str(path))
    runtime = importlib.import_module("avac4qgis_validation.runtime")
    runtime.CLAWPACK_SOURCE = CLAW
    runtime.SOURCE_ROOTS = {"wave": WAVE}
    runtime.BUILD_STAMP_ROOT = REPORTS / "build_stamps"
    expected = build["executable_sha256"]
    if sha256(WAVE / "xgeoclaw") != expected:
        raise RuntimeError("Extracted executable differs from the recorded build")
    def current(kind):
        if kind != "wave":
            raise RuntimeError("AVAC execution is deliberately excluded from this WAVE baseline")
        return WAVE
    runtime.runtime = current
    # Legacy harness writes an empty shoreline source for historical reasons.
    # Suppress it entirely, and reject attempts to hide a nonzero source.
    def no_inflow(path, entries, times):
        if entries:
            raise RuntimeError("The hydraulic baseline must not use a shoreline source")
    runtime._write_internal_inflow = no_inflow
    original_setrun = runtime._run_setrun
    def hydraulic_setrun(kind, case, qinit, **kwargs):
        (case / "impulse_configuration.yaml").rename(case / "hydraulic_configuration.yaml")
        return original_setrun(kind, case, qinit, **kwargs)
    runtime._run_setrun = hydraulic_setrun
    original_prepare = runtime.prepare_wave_hydraulic_case
    def hydraulic_prepare(*args, **kwargs):
        work = original_prepare(*args, **kwargs)
        # No empty files/directories stand in for an avalanche simulation.
        for name in ("AVAC", "CL"):
            path = work.parent / name
            if path.exists():
                shutil.rmtree(path)
        return work
    runtime.prepare_wave_hydraulic_case = hydraulic_prepare
    runtime._activate_packaged_clawpack(CLAW)
    swashes = importlib.import_module("swashes_reference")
    target = SOLVER / "SWASHES-1.05.01"
    if not target.exists():
        shutil.copytree(VENDOR / "validation/vendor/SWASHES-1.05.01", target)
    swashes.SWASHES_ROOT = target
    swashes.SWASHES = target / "bin/swashes"
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "run", "report", "all"))
    parser.add_argument("--suite", default="all", choices=("all", "swashes", "baines", "amr", "wrr", "smoke"))
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Reuse successful cases authenticated to this binary")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "build":
        build()
    else:
        if args.action == "all":
            build()
        sys.path.insert(0, str(VALIDATION))
        import regression
        if args.action != "report":
            regression.run(args.suite, args.cores, resume=args.resume)
        regression.report()


if __name__ == "__main__":
    main()
