#!/usr/bin/env python3
"""Recheck G3 tests and recorded native cases without building or running WAVE.

Run tools/g3_hydraulics.py build and the native verification matrix first.
This command checks current files, replays independent readers, then records
an immutable hash manifest of the evidence it actually inspected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import g3_hydraulics as native  # noqa: E402
from braids_core.boundaries import Face, Hydrograph, InletSpec, Section  # noqa: E402
from braids_core.boundaries.inlet import build_inlet_target  # noqa: E402
from braids_core.boundaries.inlet_io import NativeFace  # noqa: E402
from validation.hydraulics.inlets import verify_native as verify  # noqa: E402

REPORTS = native.REPORTS
RECORD = REPORTS / "verification_g3.json"
SUMMARY = REPORTS / "G3_VERIFICATION_REPORT.md"
REQUIRED_CASES = {
    *(f"uniform_{side}" for side in ("west", "east", "south", "north")),
    *(f"{shape}_{mode}" for shape in ("trapezoid", "semicircle")
      for mode in ("stage", "depth", "normal", "critical")),
    "ramp_shutdown", "zero_q", "dry_start", "split_1thread", "split_4threads",
    "trapezoid_conveyance", "backwater", "depth_nonzero_datum", "wet_supercritical",
    "mixed_regimes", "incoming_conveyance", "multiple_same_side", "multiple_same_side_split",
    "multiple_different_sides", "observer_only", "restart_continuous", "restart_resumed",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_record(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def qualification_sources() -> dict[str, str]:
    paths = {*ROOT.glob("braids_core/**/*.py"), *ROOT.glob("tests/*.py"),
             *ROOT.glob("validation/hydraulics/sections/**/*.py"),
             *ROOT.glob("validation/hydraulics/inlets/*.py"),
             *ROOT.glob("validation/hydraulics/wave_baseline/tests/*.py"),
             ROOT / "tools/g3_hydraulics.py", Path(__file__), ROOT / "pyproject.toml"}
    return {str(path.relative_to(ROOT)): native.sha256(path) for path in sorted(paths)}


def authenticate_run_records(work: Path, saved: dict, build: dict, build_digest: str) -> dict:
    """Reject cached success if the execution, binary or input deck changed."""
    execution = load_json(work / "g3_execution.json")
    metrics = load_json(work / "solver_execution_metrics.json")
    require(saved.get("native_execution") == execution, f"Stale per-case execution: {work}")
    require(all(execution.get(key) == value for key, value in metrics.items()),
            f"Current solver execution differs from G3 execution: {work}")
    require(execution.get("returncode") == 0, f"Failed native execution: {work}")
    for key in ("executable_sha256", "executable_sha256_after"):
        require(execution.get(key) == build["executable_sha256"], f"Wrong executed binary ({key}): {work}")
    require(execution.get("build_provenance_sha256") == build_digest,
            f"Run refers to another build: {work}")
    recorded_inputs = execution.get("input_sha256", {})
    input_paths = {path for path in work.iterdir() if path.is_file() and
                   (path.suffix in {".data", ".xyz"} or path.name == "braids_boundary.data.json")}
    input_paths.update(path for path in work.parent.rglob("*") if path.is_file() and
                       path.suffix in {".asc", ".yaml"})
    parameters = {line.split("=:", 1)[1].split()[0]: line.split("=:", 1)[0].strip()
                  for line in (work / "claw.data").read_text().splitlines() if "=:" in line}
    if parameters.get("restart") == "T":
        input_paths.add(work / parameters["restart_file"].strip("'\""))
    required_inputs = {os.path.relpath(path, work): native.sha256(path) for path in input_paths}
    require(required_inputs and any(path.suffix == ".asc" for path in input_paths),
            f"Missing native terrain input provenance: {work}")
    require(recorded_inputs == required_inputs, f"Native input deck changed after execution: {work}")
    for name, expected in recorded_inputs.items():
        require(native.sha256(work / name) == expected, f"Recorded native input changed: {work / name}")
    artifacts = {path.name: native.sha256(path) for path in work.iterdir() if path.is_file() and
                 (re.fullmatch(r"fort\.[tqba]\d+", path.name) or path.name in {
                     "braids_flux_faces.csv", "braids_flux_steps.csv", "braids_boundary_diagnostics.csv",
                     "braids_flux_restart.csv"})}
    require(artifacts and execution.get("native_artifact_sha256") == artifacts,
            f"Native artifacts changed after execution: {work}")
    return execution


def load_boundaries(work: Path):
    metadata = load_json(work / "braids_boundary.data.json")
    require(metadata["data_sha256"] == native.sha256(work / "braids_boundary.data"),
            f"Boundary metadata refers to a different native table: {work}")
    specs, sections, mapping = [], {}, {}
    for bc_id, entry in metadata["boundaries"].items():
        spec_data = entry["spec"].copy()
        spec_data["hydrograph"] = Hydrograph(**spec_data["hydrograph"])
        specs.append(InletSpec(**spec_data))
        profile = entry["exported_profile"]
        normal = tuple(entry["native_faces"][0]["face"]["inward_normal"])
        sections[bc_id] = Section(bc_id, profile["stations_m"], profile["bed_elevations_m"],
                                  entry["datum_m"], normal, max_stage_m=entry["max_stage_m"])
        mapping[bc_id] = [NativeFace(Face(**row["face"]), row["side"], row["global_index"])
                          for row in entry["native_faces"]]
    return specs, sections, mapping, metadata["grid"]


def require_recomputed_metrics(saved: dict, recomputed: dict, name: str) -> None:
    for key, value in recomputed.items():
        require(saved.get(key) == value, f"Stale native result {name}: {key}")
    require(recomputed.get("pass") is True, f"Independent native replay failed: {name}")


def qualify_case(saved: dict, build: dict, build_digest: str) -> dict:
    name = saved["name"]
    work = native.VALIDATION / "runs" / name / "Wave"
    require(load_json(work / "g3_result.json") == saved, f"Aggregate differs from current case: {name}")
    require(saved.get("pass") is True, f"Saved native case failed: {name}")
    execution = authenticate_run_records(work, saved, build, build_digest)
    final_time = .2 if name == "dry_start" else .4
    if name == "observer_only":
        specs = [InletSpec("upstream", "stage", Hydrograph((0., .4), (.4, .4)), value_m=1.)]
        mapping = {"upstream": [SimpleNamespace(side=1, global_index=index) for index in range(8)]}
        sections = {}
        grid = {"nx": 32, "ny": 8, "dx_m": .5, "dy_m": .5}
        require(not (work / "braids_boundary.data").exists() and
                not (work / "braids_boundary_diagnostics.csv").exists(),
                "Observer-only case unexpectedly enabled the new boundary")
    else:
        specs, sections, mapping, grid = load_boundaries(work)
    options = {}
    if name == "restart_resumed":
        options["initial_work"] = native.VALIDATION / "runs/restart_continuous/Wave"
    repeated = verify.verify_output(
        work, specs, mapping, final_time=final_time,
        expected_area=grid["nx"] * grid["ny"] * grid["dx_m"] * grid["dy_m"],
        expected_face_keys={(side, index) for side in range(1, 5)
                            for index in range(grid["ny"] if side <= 2 else grid["nx"])}, **options)
    require_recomputed_metrics(saved, repeated, name)
    if sections:
        repeated["target_and_characteristics"] = verify.verify_boundary_diagnostics(work, specs, sections, mapping)
        require(repeated["target_and_characteristics"] == saved.get("target_and_characteristics"),
                f"Stale target/characteristic evidence: {name}")
        require(all(item["pass"] for item in repeated["target_and_characteristics"].values()),
                f"Native target/characteristic checks failed: {name}")
    if saved.get("split_requested"):
        require(repeated["maximum_patch_count"] > 1, f"Split case used one patch: {name}")
    if name == "backwater":
        diagnostic = next(iter(repeated["target_and_characteristics"].values()))
        require(diagnostic["checked_subcritical_faces"] > 0 and
                diagnostic["max_compatible_target_difference_m"] > .5, "Backwater branch was not exercised")
    if name in {"wet_supercritical", "mixed_regimes", "incoming_conveyance"}:
        diagnostic = next(iter(repeated["target_and_characteristics"].values()))
        require(diagnostic["supercritical_target_faces"] > 0, f"No incoming target branch: {name}")
        if name == "mixed_regimes":
            require(diagnostic["checked_subcritical_faces"] > 0, "Mixed local regimes were not exercised")
    if name == "dry_start":
        require(repeated["initial_volume_m3"] == 0. and any(
            item["dry_target_faces"] > 0 for item in repeated["target_and_characteristics"].values()),
            "Dry startup was not exercised")
    if name == "incoming_conveyance":
        spec = specs[0]
        target = build_inlet_target(spec, sections[spec.bc_id],
                                    [face.face for face in mapping[spec.bc_id]], 0.)
        samples = verify.rows(work / "braids_flux_faces.csv")
        first_step = min(int(row["step"]) for row in samples)
        by_key = {(int(row["side"]), int(row["global_index"])): row for row in samples
                  if int(row["step"]) == first_step}
        for face, state in zip(mapping[spec.bc_id], target.faces):
            if state.h <= 1.e-12:
                continue
            row = by_key[(face.side, face.global_index)]
            velocity = row["ghost_state_inward_m3_s"] / (row["ghost_h_m"] * face.face.length_m)
            require(math.isclose(velocity, state.normal_velocity_m_s, rel_tol=1.e-10, abs_tol=1.e-10),
                    "Incoming conveyance profile does not match its declared target allocation")
    if name == "multiple_different_sides":
        signed_backflow = any(row["accepted_inward_m3_s"] < 0. for row in
                              verify.rows(work / "braids_flux_faces.csv") if int(row["side"]) in (1, 3))
        require(signed_backflow and saved.get("signed_backflow_observed") is True and sum(
            item["local_backflow_faces"] for item in repeated["target_and_characteristics"].values()) > 0,
            "Explicitly allowed local backflow was not exercised")
    if name == "restart_resumed":
        rows = verify.rows(work / "braids_flux_restart.csv")
        require(len(rows) == 1 and abs(rows[0]["checkpoint_time"] - .2) < 1.e-12,
                "Restart observer did not restore the requested checkpoint")
        prefix = [row for row in verify.rows(work / "braids_flux_steps.csv")
                  if row["t_start"] + row["dt"] <= .2 + 1.e-12]
        require(rows[0]["retained_steps"] == len(prefix) and len(prefix) < repeated["accepted_steps"] and
                rows[0]["next_step"] == len(prefix) + 1, "Restart retained-step metadata is inconsistent")
    artifacts = {str(path.relative_to(work.parent)): native.sha256(path)
                 for path in sorted(work.parent.rglob("*")) if path.is_file()}
    return {"status": "PASS", "duration_s": final_time, "recomputed": repeated, "artifacts_sha256": artifacts,
            "executed_input_sha256": execution["input_sha256"],
            "executed_native_artifact_sha256": execution["native_artifact_sha256"]}


def require_complete_matrix(matrix: dict) -> dict[str, dict]:
    require(matrix.get("status") == "PASS", "Native matrix is incomplete or failed")
    cases = matrix.get("cases", [])
    names = [case["name"] for case in cases]
    require(len(set(names)) == len(names), "Native matrix repeats a case")
    require(set(names) == REQUIRED_CASES, f"Native matrix has the wrong case set: {set(names) ^ REQUIRED_CASES}")
    require(matrix.get("case_count") == len(cases), "Native matrix case count is inconsistent")
    return {case["name"]: case for case in cases}


def test_counts(xml_path: Path) -> dict[str, int]:
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in ET.parse(xml_path).getroot().iter("testsuite"):
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return counts


def write_summary(record: dict) -> None:
    if record["status"] != "PASS":
        SUMMARY.write_text(f"# G3 verification\n\nStatus: **{record['status']}**.\n\n"
                           + record.get("error", "Verification is in progress; no passing claim is current.") + "\n")
        return
    maxima = record["maximum_errors"]
    figure = ("\n![Native stage, accepted hydrograph and water-budget diagnostics](G3_NATIVE_DIAGNOSTICS.png)\n"
              if (REPORTS / "G3_NATIVE_DIAGNOSTICS.png").exists() else "")
    SUMMARY.write_text(
        "# G3 verification\n\nStatus: **PASS** for the stated Cartesian fixed-level scope.\n\n"
        f"{record['tests']['passed']} tests passed; {record['native_case_count']} recorded native cases "
        "were authenticated and independently reread. The numerical cases cover 0.2–0.4 seconds of model time. "
        f"This verification took {record['elapsed_wall_s']:.1f} seconds and did not rerun the native matrix.\n\n"
        "Checks cover four orientations, both section shapes and all four target modes, hydrograph events, "
        "dry/zero flow, subcritical compatibility and backwater, incoming/mixed regimes, conveyance, "
        "multiple IDs, split patches, one/four threads, and checkpoint continuation. Restart states are "
        "compared at matching physical times; legacy native frame numbers may differ.\n\n"
        "The accepted boundary flux, ghost-state discharge and native-cell volume are checked separately. "
        "These short boundary verification cases do not qualify long reservoir filling runs. "
        "AMR/subcycling/reflux, cut or oblique faces and sediment remain outside this qualification.\n\n"
        "| Maximum independently rechecked error | Value |\n|---|---:|\n"
        f"| Accumulated absolute accepted-Q error / requested volume | {maxima['relative_accumulated_absolute_discharge']:.6g} |\n"
        f"| Relative native water-budget residual | {maxima['relative_water_budget']:.6g} |\n"
        f"| Target-stage difference | {maxima['target_stage_m']:.6g} m |\n"
        f"| Outgoing characteristic invariant difference | {maxima['outgoing_invariant_m_s']:.6g} m/s |\n"
        f"| Preparation volume change | {maxima['preparation_volume_m3']:.6g} m³ |\n\n"
        "Exact commands, counts, source/build/input hashes and all inspected per-case artifacts are recorded in "
        "[verification_g3.json](verification_g3.json). Numerical thresholds are preserved from the matrix.\n\n"
        "Run `.venv/bin/python tools/verify_g3.py` to reproduce this check against the existing native evidence.\n"
        + figure)


def run() -> dict:
    started = time.perf_counter()
    record = {"gate": "G3", "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "scope": "Cartesian fixed level 1, axis-aligned whole faces",
              "native_matrix_runs_performed_by_this_command": 0, "amr_qualified": False, "sediment_qualified": False}
    write_record(RECORD, record)
    write_summary(record)
    try:
        sources = qualification_sources()
        build = native.check_build()
        build_path = REPORTS / "build_provenance.json"
        build_digest = native.sha256(build_path)
        matrix_path = REPORTS / "native_results.json"
        matrix = load_json(matrix_path)
        cases = require_complete_matrix(matrix)
        require(matrix.get("build_provenance_sha256") == build_digest, "Native matrix refers to another build")
        matrix_digest = native.sha256(matrix_path)
        xml_path = REPORTS / "g3_tests.xml"
        log_path = REPORTS / "g3_test_results.txt"
        command = [sys.executable, "-m", "pytest", "tests", "validation/hydraulics/sections/tests",
                   "validation/hydraulics/wave_baseline/tests", "-q", f"--junitxml={xml_path}"]
        print("Running section, inlet, source-contract and compiled native probe tests ...", flush=True)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log_path.write_text(result.stdout + result.stderr)
        record.update(test_command=command, tests_returncode=result.returncode,
                      tests=test_counts(xml_path) if xml_path.exists() else {})
        record["test_artifacts_sha256"] = {str(path.relative_to(ROOT)): native.sha256(path)
                                           for path in (xml_path, log_path) if path.exists()}
        require(result.returncode == 0, f"Tests failed; see {log_path}")
        require(record["tests"]["passed"] > 0 and record["tests"]["skipped"] == 0,
                "Qualification requires a nonempty test suite without skips")
        print(result.stdout.strip(), flush=True)
        native.configure_runtime()
        record["cases"] = {}
        for name in sorted(cases):
            record["cases"][name] = qualify_case(cases[name], build, build_digest)
        threads = verify.compare_thread_runs()
        require(matrix.get("one_four_thread_bitwise_comparisons") == threads, "Stale thread comparison")
        restart = verify.compare_restart_runs(native.VALIDATION / "runs/restart_continuous/Wave",
                                               native.VALIDATION / "runs/restart_resumed/Wave")
        require(all(restart.values()) and cases["restart_resumed"].get(
            "continuous_resumed_bitwise_comparisons") == restart, "Stale or failed restart comparison")
        record["one_four_thread_bitwise_comparisons"] = threads
        record["restart_comparisons"] = restart
        require(native.check_build() == build and native.sha256(build_path) == build_digest,
                "Build changed during verification")
        require(native.sha256(matrix_path) == matrix_digest, "Native matrix changed during verification")
        require(qualification_sources() == sources, "Qualification code changed during verification")
        for name, result in record["cases"].items():
            case_root = native.VALIDATION / "runs" / name
            current = {str(path.relative_to(case_root)): native.sha256(path)
                       for path in sorted(case_root.rglob("*")) if path.is_file()}
            require(current == result["artifacts_sha256"], f"Case artifacts changed during verification: {name}")
        evidence = [build_path, matrix_path, ROOT / "solver/source_provenance.json"]
        figure = REPORTS / "G3_NATIVE_DIAGNOSTICS.png"
        if figure.exists():
            evidence.append(figure)
        repeated = [case["recomputed"] for case in record["cases"].values()]
        record["maximum_errors"] = {
            "relative_accumulated_absolute_discharge": max(
                boundary["relative_accumulated_absolute_error"] for case in repeated
                for boundary in case["boundaries"].values()),
            "relative_water_budget": max(case["relative_water_budget_residual"] for case in repeated),
            "preparation_volume_m3": max(case["max_preparation_volume_change_m3"] for case in repeated),
            "target_stage_m": max(diagnostic["max_target_stage_error_m"] for case in repeated
                                   for diagnostic in case.get("target_and_characteristics", {}).values()),
            "outgoing_invariant_m_s": max(diagnostic["max_outgoing_invariant_error_m_s"] for case in repeated
                                           for diagnostic in case.get("target_and_characteristics", {}).values()),
        }
        record.update(status="PASS", completed_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_wall_s=time.perf_counter() - started,
                      native_case_count=len(cases), source_sha256=sources,
                      executable_sha256=build["executable_sha256"],
                      evidence_sha256={str(path.relative_to(ROOT)): native.sha256(path) for path in evidence},
                      thresholds=matrix["thresholds"],
                      provenance_scope="Execution-time input and native-artifact hashes were authenticated, "
                      "including terrain and the referenced restart checkpoint. All current case artifacts "
                      "were hashed and native results reread.")
        write_record(RECORD, record)
        write_summary(record)
        print(f"G3 PASS: {RECORD}", flush=True)
        return record
    except BaseException as error:
        record.update(status="FAIL", error=f"{type(error).__name__}: {error}",
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write_record(RECORD, record)
        write_summary(record)
        raise


if __name__ == "__main__":
    run()
