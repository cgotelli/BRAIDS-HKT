#!/usr/bin/env python3
"""Verify current G2 sources and attach completed native-baseline evidence.

Run tools/wave_baseline.py all first. This command reruns section/reference and
compiled probe tests, but does not repeat the long physical WAVE regressions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import wave_baseline as baseline  # noqa: E402
from validation.hydraulics.sections.verify_production import verify  # noqa: E402
from validation.hydraulics.sections.verify_gis import verify as verify_gis  # noqa: E402

REPORTS = ROOT / "validation/hydraulics/sections/reports"
RECORD = REPORTS / "verification_step2.json"


def qualification_sources() -> dict[str, str]:
    paths = {*ROOT.glob("braids_core/**/*.py"), *ROOT.glob("tests/*.py"),
             *ROOT.glob("validation/hydraulics/sections/**/*.py"),
             *ROOT.glob("validation/hydraulics/wave_baseline/tests/*.py"),
             ROOT / "validation/hydraulics/wave_baseline/check_thacker_mass.py",
             Path(__file__), ROOT / "pyproject.toml"}
    return {str(p.relative_to(ROOT)): baseline.sha256(p) for p in sorted(paths)}


def require_current_native_evidence(native: dict, reports: Path, runs: Path) -> None:
    """Do not reuse a cached COMPLETE after an interrupted or failed rerun."""
    state_path = reports / "suite_status.json"
    states = json.loads(state_path.read_text()) if state_path.exists() else {}
    for path in (reports / "suite_statuses").glob("*.json"):
        states[path.stem] = json.loads(path.read_text())
    executions = {str(p.parent.relative_to(runs)): json.loads(p.read_text())
                  for p in runs.rglob("solver_execution_metrics.json")}
    metrics = {str(p.relative_to(runs)): json.loads(p.read_text())
               for p in runs.rglob("results/summary.json")}
    if (native["suites"] != states or native["executions"] != executions or
            native["metrics"] != metrics):
        raise RuntimeError("Native aggregate is stale relative to current suite/run records; refresh the baseline report")
    if len(executions) != 15:
        raise RuntimeError("Expected exactly 15 current native execution records")


def require_reviewed_adapters(root: Path, reports: Path, vendor: Path) -> dict:
    """Bind accepted historical runs to reviewed adapters and exact archives."""
    review = json.loads((reports / "adapter_review.json").read_text())
    expected = {"tools/wave_baseline.py", "validation/hydraulics/wave_baseline/regression.py"}
    if review["upstream_commit"] != baseline.PIN or set(review["reviewed_sources"]) != expected:
        raise RuntimeError("Incomplete adapter review or wrong upstream revision")
    for name, digest in review["reviewed_sources"].items():
        if baseline.sha256(root / name) != digest:
            raise RuntimeError(f"Baseline adapter changed after the recorded review: {name}")
    for name, digest in review["unchanged_vendor_driver_hashes"].items():
        if baseline.sha256(vendor / name) != digest:
            raise RuntimeError(f"Pinned numerical driver changed: {name}")
    for digest, archive in review["archives"].items():
        if baseline.sha256(reports / archive) != digest:
            raise RuntimeError(f"Executed adapter archive hash mismatch: {archive}")
    paths = sorted((reports / "suite_provenance").glob("*.json"))
    if not paths:
        raise RuntimeError("No executed suite adapter provenance")
    for path in paths:
        provenance = json.loads(path.read_text())
        if (provenance["upstream_commit"] != baseline.PIN or
                provenance["upstream_driver_hashes"] != review["unchanged_vendor_driver_hashes"]):
            raise RuntimeError(f"Unreviewed numerical drivers: {path.name}")
        for kind in ("adapter", "orchestration"):
            digest, archive = provenance[f"{kind}_sha256"], provenance[f"{kind}_archive"]
            if (digest not in review[f"approved_{kind}_revisions"] or
                    review["archives"].get(digest) != archive or
                    baseline.sha256(reports / archive) != digest):
                raise RuntimeError(f"Unreviewed executed {kind}: {path.name}")
    return review


def require_native_acceptance(metrics: dict) -> None:
    """Recheck criteria even if a resume skipped the original suite function."""
    smoke = metrics["00_water_only_smoke/results/summary.json"]
    if (smoke["passed"] is not True or smoke["setprob_data_exists"] is not False or
            not 0 <= smoke["maximum_depth_error_m"] <= 1e-7 or
            not 0 <= smoke["maximum_momentum_m2_s"] <= 1e-12):
        raise RuntimeError("Native water-only smoke acceptance failed")
    amr = metrics["08_amr_parallel/results/summary.json"]
    equal = [f"{prefix}{suffix}" for prefix in ("parallel_", "amr_parallel_")
             for suffix in ("bitwise_equal_float32_fgout", "final_native_binary_bitwise_equal",
                            "final_fgout_binary_bitwise_equal")]
    if any(amr.get(key) is not True for key in equal):
        raise RuntimeError("Native one/eight-thread bitwise comparison failed")
    if any(amr.get(key) != 2 for key in
           ("amr_maximum_written_level", "amr_maximum_written_level_1core", "amr_maximum_written_level_8core")):
        raise RuntimeError("Native forced AMR did not reach level 2")
    ratio = amr["amr_error_ratio_to_coarse"]
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise RuntimeError("Native AMR did not improve on the coarse baseline")


def run(wait_for_native: bool = False) -> dict:
    record = {"gate": "G2", "status": "RUNNING", "python": platform.python_version(),
              "scope": "Section geometry/GIS and standalone source baseline; G3 numerical inlet flux remains open."}
    tested_sources = qualification_sources()
    baseline.json_write(RECORD, record)
    try:
        print("Verifying production sections against independent references ...", flush=True)
        section = verify()
        if section["status"] != "PASS":
            raise RuntimeError("Required section checks failed; see G2_SECTION_REPORT.md")
        gis = verify_gis()
        if gis["status"] != "PASS":
            raise RuntimeError("Required GIS hydraulic checks failed; see G2_GIS_REPORT.md")
        with tempfile.TemporaryDirectory(prefix="braids-g2-reproduce-") as directory:
            other = Path(directory)
            verify(output=other)
            verify_gis(output=other)
            outputs = sorted(path for path in other.iterdir() if path.is_file())
            if any(baseline.sha256(path) != baseline.sha256(REPORTS / path.name) for path in outputs):
                raise RuntimeError("G2 reports did not reproduce byte-identically")
            record["byte_identical_g2_reports"] = len(outputs)

        print("Running section, GIS, G1 and compiled WAVE probe tests ...", flush=True)
        xml_path = REPORTS / "g2_tests.xml"
        command = [sys.executable, "-m", "pytest", "tests",
                   "validation/hydraulics/sections/tests",
                   "validation/hydraulics/wave_baseline/tests", "-q", f"--junitxml={xml_path}"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (REPORTS / "g2_test_results.txt").write_text(result.stdout + result.stderr)
        record["test_command"] = command
        record["tests_returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Tests failed; see {REPORTS / 'g2_test_results.txt'}")
        suites = ET.parse(xml_path).getroot().iter("testsuite")
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
        record["tests"] = counts
        print(result.stdout.strip(), flush=True)

        if wait_for_native:
            record["status"] = "WAITING_FOR_NATIVE"
            baseline.json_write(RECORD, record)
            print("Section/probe checks passed; waiting for all native regressions ...", flush=True)
            while True:
                path = baseline.REPORTS / "baseline_results.json"
                try:
                    available = json.loads(path.read_text()) if path.exists() else {}
                except json.JSONDecodeError:
                    available = {}  # The separate report writer may be mid-write.
                current_review = json.loads((baseline.REPORTS / "adapter_review.json").read_text())
                if (available.get("status") == "COMPLETE" and
                        available.get("adapter_review") == current_review):
                    break
                for status_path in (baseline.REPORTS / "suite_statuses").glob("*.json"):
                    try:
                        state = json.loads(status_path.read_text())
                    except json.JSONDecodeError:
                        continue
                    if state.get("status") == "failed":
                        raise RuntimeError(f"Native suite failed while waiting: {status_path.stem}")
                time.sleep(5)

        print("Checking completed native regressions against current source/build ...", flush=True)
        if qualification_sources() != tested_sources:
            raise RuntimeError("Qualification sources changed during verification; rerun for the current code")
        if baseline.git("rev-parse", "HEAD") != baseline.PIN or baseline.git("status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError("Vendor checkout is not clean at the pinned source revision")
        build_path = baseline.REPORTS / "build_provenance.json"
        build = json.loads(build_path.read_text())
        if build["source_files_sha256"] != baseline.source_hashes():
            raise RuntimeError("WAVE source changed since the recorded native build")
        if build["executable_sha256"] != baseline.sha256(baseline.WAVE / "xgeoclaw"):
            raise RuntimeError("WAVE executable does not match the recorded build")
        baseline_path = baseline.REPORTS / "baseline_results.json"
        native = json.loads(baseline_path.read_text())
        require_current_native_evidence(native, baseline.REPORTS, baseline.VALIDATION / "runs")
        if native["status"] != "COMPLETE" or native["native_run_count"] != 15:
            raise RuntimeError("Native baseline incomplete; finish tools/wave_baseline.py all before closing Step 2")
        expected_suites = {"smoke", "swashes", "baines", "amr", "wrr"}
        if set(native["suites"]) != expected_suites or any(
                s["status"] != "completed" or s["binary_sha256"] != build["executable_sha256"]
                for s in native["suites"].values()):
            raise RuntimeError("Native suite statuses or executable hashes do not match the build")
        for name, execution in native["executions"].items():
            checks = execution.get("output_checks", {})
            if (execution["returncode"] != 0 or
                    execution["executable_sha256"] != build["executable_sha256"] or
                    execution.get("executable_sha256_after") != build["executable_sha256"] or
                    not all(checks.get(key) is True for key in
                            ("finite_states", "nonnegative_depth", "no_avac_or_cl_directory"))):
                raise RuntimeError(f"Native execution checks failed: {name}")
        review = require_reviewed_adapters(ROOT, baseline.REPORTS, baseline.VENDOR)
        if native.get("adapter_review") != review:
            raise RuntimeError("Refresh the native aggregate with the current reviewed reporting adapter")
        require_native_acceptance(native["metrics"])
        # Preserve the known failed hydraulic diagnostic while reproducing the
        # source baseline. G2 completion never authorizes the later G4/G5 gate.
        mass_check = subprocess.run(
            [sys.executable, "validation/hydraulics/wave_baseline/check_thacker_mass.py"],
            cwd=ROOT, capture_output=True, text=True, check=True)
        print(mass_check.stdout.strip(), flush=True)

        evidence = [build_path, baseline_path, baseline.REPORTS / "adapter_review.json",
                    baseline.REPORTS / "thacker_native_mass.json",
                    baseline.SOLVER / "source_provenance.json",
                    REPORTS / "g2_summary.json", REPORTS / "g2_test_results.txt",
                    REPORTS / "g2_gis_summary.json",
                    ROOT / "docs/GEOFLOOD_BOUNDARY_AUDIT.md"]
        record.update(status="PASS", upstream_commit=baseline.PIN,
                      native_run_count=native["native_run_count"],
                      executable_sha256=build["executable_sha256"],
                      source_sha256=tested_sources,
                      evidence_sha256={str(p.relative_to(ROOT)): baseline.sha256(p) for p in evidence},
                      known_diagnostics=section["additional_diagnostics"],
                      retained_solver_water_budget_issue=json.loads(
                          (baseline.REPORTS / "thacker_native_mass.json").read_text()),
                      hydraulic_conservation_qualified=False,
                      initial_orchestration_provenance_gap=review["initial_orchestration_provenance_gap"],
                      accuracy_scope=native["accuracy_gate"])
        baseline.json_write(RECORD, record)
        print(f"Step 2 PASS: {RECORD}", flush=True)
        return record
    except Exception as error:
        record.update(status="FAIL", error=str(error))
        baseline.json_write(RECORD, record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-native", action="store_true",
                        help="Run section/probe checks now, then await an already running native suite")
    run(parser.parse_args().wait_for_native)
