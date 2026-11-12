#!/usr/bin/env python3
"""Authenticate and reread the declared G4 matrix without running its solver.

Evidence integrity and numerical acceptance are separate results. A complete,
authenticated experiment may correctly report a failed numerical gate.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import g4_hydraulics as native  # noqa: E402
from tools.verify_g3 import load_json, require, test_counts, write_record  # noqa: E402
from validation.hydraulics.channels import run_verification as runner  # noqa: E402
from validation.hydraulics.channels import reference  # noqa: E402
from validation.hydraulics.channels.ledger import THRESHOLDS as LEDGER_THRESHOLDS  # noqa: E402

REPORTS = native.REPORTS
RECORD = REPORTS / "verification_g4.json"
SUMMARY = REPORTS / "G4_VERIFICATION_REPORT.md"
COARSE_PROFILE_CASES = frozenset(
    f"{shape}_{experiment}_dx{str(dx).replace('.', 'p')}"
    for shape in ("trapezoid", "semicircle")
    for experiment in ("throughflow", "friction") for dx in (.5, .25))


def canonical(value):
    """Match JSON's tuple/list representation and reject nonfinite evidence."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def qualification_sources():
    paths = {*ROOT.glob("braids_core/**/*.py"), *ROOT.glob("tests/*.py"),
             *ROOT.glob("validation/hydraulics/sections/**/*.py"),
             *ROOT.glob("validation/hydraulics/inlets/*.py"),
             *ROOT.glob("validation/hydraulics/channels/*.py"),
             *ROOT.glob("validation/hydraulics/channels/native/*"),
             *ROOT.glob("validation/hydraulics/wave_baseline/*.py"),
             *ROOT.glob("validation/hydraulics/wave_baseline/tests/*.py"),
             *ROOT.glob("vendor/avac-qgis/validation/**/*.py"),
             *ROOT.glob("tools/*.py"), ROOT / "pyproject.toml"}
    return {str(path.relative_to(ROOT)): native.sha256(path)
            for path in sorted(paths) if path.is_file()}


def required_configurations():
    if hasattr(runner, "required_configurations"):
        return runner.required_configurations()
    configs = list(runner.configurations())
    if hasattr(runner, "restart_configurations"):
        configs.extend(runner.restart_configurations())
    return configs


def bind_matrix(matrix, configs):
    require(matrix.get("status") in {"PASS", "FAIL"}, "Native matrix is incomplete")
    expected = {config.name: canonical(asdict(config)) for config in configs}
    require(expected and len(expected) == len(configs), "Required configuration factory has duplicate or no cases")
    declared = matrix.get("configuration", [])
    by_name = {config["name"]: config for config in declared}
    require(len(by_name) == len(declared) and by_name == expected,
            "Native matrix differs from the exact required configurations")
    cases = matrix.get("cases", [])
    observed = {case["name"]: case for case in cases}
    require(len(observed) == len(cases) and set(observed) == set(expected),
            "Native matrix has missing, duplicate or undeclared case results")
    require(all(case.get("config") == expected[name] for name, case in observed.items()),
            "Case results do not match their declared configuration")
    return observed


def authenticate_build(identity):
    require(native.sha256(Path(identity["executable"])) == identity["executable_sha256"],
            "Active executable differs from the recorded identity")
    overlays = native.overlay_hashes()
    if overlays:
        require(identity.get("overlay") == overlays and identity.get("build_provenance_sha256"),
                "Current native overlay has not been built and authenticated")
        expected = native.base.source_hashes(native.base.BUILD)
        expected.update(native.overlay_source_hashes())
        require(native.base.source_hashes(native.BUILD) == expected,
                "G4 staged source is not exactly the preserved G3 source plus declared overlay")


def authenticate_execution(work, saved, identity):
    execution_path = work / "g4_execution.json"
    execution = load_json(execution_path)
    require(execution.get("status") in {"PASS", "FAIL"}, f"Native execution is incomplete: {work}")
    require(execution.get("inherited_build") == identity, f"Execution used another solver identity: {work}")
    metrics_path = work / "solver_execution_metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {"returncode": -1}
    require(execution.get("metrics") == metrics, f"Stale execution metrics: {work}")
    restart_names = {config.name for config in runner.restart_configurations()}
    expected_threads = 4 if saved["name"].endswith("_4t") or saved["name"] in restart_names else 1
    require(type(metrics.get("openmp_threads")) is int and metrics["openmp_threads"] == expected_threads,
            f"Native run did not use its declared {expected_threads} OpenMP threads: {work}")
    if "executable_sha256" in metrics:
        require(metrics["executable_sha256"] == identity["executable_sha256"] and
                metrics.get("executable_sha256_after") == identity["executable_sha256"],
                f"Wrong executed binary: {work}")
    else:
        require(execution["status"] == "FAIL", f"Missing executed binary identity: {work}")
    if execution["status"] == "PASS":
        require(metrics.get("returncode") == 0, f"Failed execution recorded as PASS: {work}")
    else:
        require(saved.get("pass") is False and saved.get("status") == "ERROR" and execution.get("error"),
                f"Native failure hidden by its result: {work}")
    require(execution.get("input_sha256") == native.input_hashes(work), f"Numerical inputs changed: {work}")
    require(execution.get("native_artifact_sha256") == native.artifact_hashes(work),
            f"Native artifacts changed after execution: {work}")
    from validation.hydraulics.channels.ledger_storage import storage_record, MANIFEST_NAME
    storage = storage_record(work)
    if storage is not None:
        require(execution.get("ledger_storage") == storage and
                execution.get("ledger_storage_manifest_sha256") == native.sha256(work / MANIFEST_NAME),
                f"Compressed ledger is missing its execution-bound storage record: {work}")
    else:
        require("ledger_storage" not in execution and "ledger_storage_manifest_sha256" not in execution,
                f"Execution storage provenance has no matching container: {work}")
    expected_preparation = {str(path.relative_to(ROOT)) for path in (
        ROOT / "tools/g4_hydraulics.py", native.VALIDATION / "cases.py", native.VALIDATION / "inlet_io.py",
        native.VALIDATION / "ledger_storage.py",
        ROOT / "tools/g3_hydraulics.py",
        native.base.VENDOR / "validation/avac4qgis_validation/runtime.py")}
    preparation = execution.get("preparation_source_sha256", {})
    require(set(preparation) == expected_preparation, f"Missing preparation source provenance: {work}")
    require(all(native.sha256(ROOT / name) == digest for name, digest in preparation.items()),
            f"Preparation sources changed after execution: {work}")
    if saved.get("status") != "ERROR":
        require(saved.get("execution_sha256") == native.sha256(execution_path),
                f"Result refers to another execution: {work}")
    return execution


def expected_output_times(config, *, restart=False):
    start = config.checkpoint_time_s if restart else 0.
    require(start is not None, "Restart configuration has no checkpoint time")
    count = round((config.duration_s-start)/config.output_interval_s)
    require(math.isclose(start+count*config.output_interval_s, config.duration_s, abs_tol=1.e-9),
            "Declared output times do not cover the final time")
    return {round(start+index*config.output_interval_s, 10) for index in range(count+1)}


def require_output_times(work, config, *, restart=False):
    times = [round(float(path.read_text().split()[0]), 10) for path in work.glob("fort.t[0-9]*")]
    require(all(math.isfinite(value) for value in times) and len(times) == len(set(times)) and
            set(times) == expected_output_times(config, restart=restart),
            f"Missing, duplicate, nonfinite or undeclared native output times: {work}")


def restart_prefix(work, config, saved):
    is_restart = config.name in {item.name for item in runner.restart_configurations()}
    expected = native.RUNS / runner.checkpoint_source_name(config) / "Wave" if is_restart else None
    if saved.get("status") != "ERROR":
        require(saved.get("restart_prefix_work") == (str(expected) if expected else None),
                f"Restart result uses another continuous prefix: {work}")
    return expected


def authenticate_blocked_case(saved, work):
    restarts = {config.name: config for config in runner.restart_configurations()}
    require(saved["name"] in restarts and saved.get("pass") is False and
            saved.get("execution_performed") is False and saved.get("error"),
            "Only a declared unexecuted restart may be prerequisite-blocked")
    prerequisite = runner.checkpoint_source_name(restarts[saved["name"]])
    path = native.RUNS / prerequisite / "Wave/g4_result.json"
    origin = load_json(path)
    require(saved.get("prerequisite") == prerequisite and
            saved.get("prerequisite_result_sha256") == native.sha256(path) and
            origin.get("pass") is False and origin.get("status") == "ERROR" and "ledger" not in origin,
            f"Restart block is not supported by its declared failed continuous case: {work}")
    return {"name": saved["name"], "evidence_status": "PASS", "numerical_pass": False,
            "execution_status": "NOT_PERFORMED", "replay_status": "PREREQUISITE_BLOCKED",
            "result_sha256": native.sha256(work / "g4_result.json"), "error": saved["error"],
            "prerequisite": prerequisite, "prerequisite_result_sha256": native.sha256(path)}


def replay_case(saved, identity):
    work = native.RUNS / saved["name"] / "Wave"
    require(load_json(work / "g4_result.json") == saved, f"Stale aggregate result: {saved['name']}")
    if saved.get("status") == "BLOCKED":
        return authenticate_blocked_case(saved, work), saved
    execution = authenticate_execution(work, saved, identity)
    result = {"name": saved["name"], "evidence_status": "PASS", "numerical_pass": saved["pass"],
              "execution_status": execution["status"],
              "openmp_threads": execution["metrics"]["openmp_threads"],
              "execution_sha256": native.sha256(work / "g4_execution.json"),
              "result_sha256": native.sha256(work / "g4_result.json"),
              "input_sha256": execution["input_sha256"],
              "native_artifact_sha256": execution["native_artifact_sha256"],
              "preparation_source_sha256": execution["preparation_source_sha256"]}
    if execution["status"] == "FAIL":
        result["error"] = execution["error"]
        result["replay_status"] = "NATIVE_EXECUTION_FAILED"
        return result, saved
    config, prepared = runner.restore_prepared(work)
    require(canonical(asdict(config)) == saved["config"], f"Restored deck has another configuration: {work}")
    prefix = restart_prefix(work, config, saved)
    try:
        repeated = runner.assess_case(config, prepared, initial_work=prefix)
        if prefix is not None:
            comparison = runner.compare_native_by_time(prefix, work,
                require_times=expected_output_times(config, restart=True))
            repeated.update(restart_comparisons=comparison, restart_prefix_work=str(prefix))
            repeated["pass"] &= all(comparison.values())
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        require(saved.get("status") == "ERROR" and saved.get("pass") is False and saved.get("error") == message,
                f"Independent assessment changed for {saved['name']}: {message}")
        result.update(replay_status="ASSESSMENT_ERROR_REPRODUCED", error=message)
        return result, saved
    require_output_times(work, config, restart=prefix is not None)
    repeated = canonical(repeated)
    require(all(saved.get(key) == value for key, value in repeated.items()),
            f"Saved numerical metrics differ from independent reread: {saved['name']}")
    require(repeated.get("pass") == saved.get("pass"), f"Stale case acceptance: {saved['name']}")
    result.update(replay_status="COMPLETE", duration_s=repeated["duration_s"],
                  accepted_steps=repeated["ledger"]["accepted_steps"],
                  budget_error=repeated["ledger"]["maximum_relative_water_budget_error"],
                  hydraulic_profile_pass=repeated.get("hydraulic_profile_pass"),
                  fill_assessment=repeated.get("fill_assessment"),
                  final_relaxation_pass=repeated.get("final_relaxation_pass"),
                  positivity=repeated["ledger"].get("positivity"),
                  branch_history=repeated.get("branch_history"),
                  final_profile=repeated.get("final_profile"), worst_profile=repeated.get("worst_profile"),
                  maximum_intermediate_residuals=repeated["ledger"]["maximum_intermediate_residuals"],
                  boundary_metrics=repeated["ledger"]["boundaries"])
    return result, {**saved, **repeated}


def numerical_result(cases, suite):
    require(isinstance(suite, dict) and isinstance(suite.get("pass"), bool),
            "Suite evaluation must expose an explicit boolean pass")
    accepted = all((case.get("run_integrity_pass") is True if case["name"] in COARSE_PROFILE_CASES
                    else case.get("pass") is True) for case in cases)
    return "PASS" if accepted and suite["pass"] else "FAIL"


def numerical_maxima(cases):
    complete = [case for case in cases.values() if case.get("replay_status") == "COMPLETE"]
    if not complete:
        return {}
    boundaries = [value for case in complete for value in case["boundary_metrics"].values()]
    return {"maximum_relative_water_budget_error": max(case["budget_error"] for case in complete),
            "maximum_accepted_q_relative_error": max(
                (value["relative_accumulated_absolute_error"] for value in boundaries), default=0.),
            "maximum_state_q_relative_error": max(
                (value["state_relative_accumulated_absolute_error"] for value in boundaries), default=0.),
            "maximum_wet_stage_spread_m": max(
                (value["max_wet_stage_spread_m"] for value in boundaries), default=0.),
            "maximum_outgoing_invariant_error_m_s": max(
                (value["max_outgoing_invariant_error_m_s"] for value in boundaries), default=0.),
            "maximum_zero_q_leakage_m3": max(
                (value["zero_q_leakage_m3"] for value in boundaries), default=0.),
            "maximum_exact_trace_flux_relative_error": max(
                (value.get("max_exact_trace_flux_relative_error", 0.) for value in boundaries), default=0.),
            "maximum_clipping_added_volume_m3": max(
                ((case.get("positivity") or {}).get("negative_depth_added_m3", 0.) for case in complete), default=0.),
            "maximum_absolute_source_water_change_m3": max(
                ((case.get("positivity") or {}).get("absolute_source_water_change_m3", 0.) for case in complete), default=0.),
            "maximum_intermediate_residuals_m3": {
                key: max(case["maximum_intermediate_residuals"][key] for case in complete)
                for key in complete[0]["maximum_intermediate_residuals"]}}


def open_requirements(cases, suite):
    pending = ["Water AMR, synchronized reflux accounting and regridding"]
    for group_name, group in suite.items():
        if isinstance(group, dict):
            for name, result in group.items():
                if isinstance(result, dict) and result.get("pass") is False:
                    pending.append(f"Numerical comparison: {group_name}/{name}")
    for case in cases:
        accepted = case.get("run_integrity_pass") if case["name"] in COARSE_PROFILE_CASES else case.get("pass")
        if accepted is not True:
            pending.append(f"Numerical case: {case['name']}")
    return pending


def write_summary(record):
    paragraphs = ["# G4 verification", f"Status: **{record['status']}**. "
                  f"Evidence: **{record['evidence_status']}**. Numerical gates: **{record['numerical_status']}**."]
    if record.get("error"):
        paragraphs.append(record["error"])
    if record.get("tests"):
        count = record["tests"]
        paragraphs.append(f"Tests: {count['passed']} passed, {count['failures']} failed, "
                          f"{count['errors']} errors, {count['skipped']} skipped.")
    if record.get("cases"):
        complete = sum(case.get("replay_status") == "COMPLETE" for case in record["cases"].values())
        passing = sum(case.get("replay_status") == "COMPLETE" and case["numerical_pass"]
                      for case in record["cases"].values())
        blocked = sum(case.get("replay_status") == "PREREQUISITE_BLOCKED" for case in record["cases"].values())
        paragraphs.append(f"Authenticated evidence for {len(record['cases'])} declared cases; independently reread "
                          f"{complete} complete native outputs, of which {passing} pass every individual gate; "
                          f"{blocked} restarts were not executed because their "
                          "declared continuous prerequisite failed. This command did not run the native matrix.")
        failed = [name for name, case in record["cases"].items() if not case["numerical_pass"]]
        if failed:
            paragraphs.append("Cases failing their individual gates or execution (the declared coarse profile "
                              "exceptions are assessed through convergence at suite level):\n\n" +
                              "\n".join(f"- `{name}`" for name in failed))
        durations = sorted({case["duration_s"] for case in record["cases"].values() if "duration_s" in case})
        if durations:
            paragraphs.append("Completed case durations: " + ", ".join(f"{value:g} s" for value in durations) + ".")
    if record.get("numerical_maxima"):
        values = record["numerical_maxima"]
        paragraphs.append("Largest rechecked errors: water budget "
                          f"{values['maximum_relative_water_budget_error']:.6g} relative; accepted inlet Q "
                          f"{values['maximum_accepted_q_relative_error']:.6g} relative; ghost-state Q "
                          f"{values['maximum_state_q_relative_error']:.6g} relative; wet-face stage spread "
                          f"{values['maximum_wet_stage_spread_m']:.6g} m; outgoing invariant "
                          f"{values['maximum_outgoing_invariant_error_m_s']:.6g} m/s.")
        paragraphs.append("Largest independently recorded clipping addition: "
                          f"{values.get('maximum_clipping_added_volume_m3', 0.):.6g} m³; accumulated absolute source-water change: "
                          f"{values.get('maximum_absolute_source_water_change_m3', 0.):.6g} m³; exact interface-flux discrepancy: "
                          f"{values.get('maximum_exact_trace_flux_relative_error', 0.):.6g} relative.")
    if record.get("suite_evaluation", {}).get("long_reservoir_relaxation"):
        paragraphs.append("The versioned matrix separates 60 s transient filling from four mandatory 3,000 s rest checks. "
                          "Transient passes do not establish rest. Friction uses four prescribed grids, raw 1% accuracy on all four, "
                          "and the fixed finest-three trend while retaining the full trend. These prospective scope revisions "
                          "do not reclassify the archived first-attempt failures; see [the correction record](../G4_CORRECTIONS.md).")
    paragraphs.append("Scope: Cartesian fixed-grid channels, including the declared split-patch and restart comparisons. "
                      "Numerical status above applies to this fixed-grid suite. AMR/subcycling/reflux remains unsupported "
                      "and unqualified independently of that status. Sediment and real-reservoir qualification are later gates.")
    if record.get("open_requirements"):
        paragraphs.append("Open requirements:\n\n" + "\n".join(f"- {item}" for item in record["open_requirements"]))
    paragraphs.append("[verification_g4.json](verification_g4.json) records the exact configuration matrix, tests, "
                      "active executable and overlay identity, input/output hashes, rechecks and numerical failures. "
                      "Reproduce with `.venv/bin/python tools/verify_g4.py` after recording the complete native matrix.")
    figures = record.get("diagnostic_figures_sha256", {})
    for filename in figures:
        paragraphs.append(f"![G4 native diagnostics]({Path(filename).name})")
    SUMMARY.write_text("\n\n".join(paragraphs) + "\n")


def run():
    started = time.perf_counter()
    record = {"gate": "G4", "status": "RUNNING", "evidence_status": "RUNNING", "numerical_status": "NOT_EVALUATED",
              "started_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "native_matrix_runs_performed_by_this_command": 0, "amr_qualified": False,
              "g4_complete": False, "open_requirements": ["Water AMR, synchronized reflux accounting and regridding"]}
    write_record(RECORD, record)
    write_summary(record)
    try:
        sources = qualification_sources()
        identity = native.frozen_build()
        record.update(source_sha256=sources, active_solver=identity)
        authenticate_build(identity)
        matrix_path = REPORTS / "native_results.json"
        matrix = load_json(matrix_path)
        matrix_digest = native.sha256(matrix_path)
        configs = required_configurations()
        cases = bind_matrix(matrix, configs)
        require(matrix.get("inherited_build") == identity, "Matrix uses another active executable or overlay")
        require(matrix.get("profile_thresholds") == canonical(asdict(reference.THRESHOLDS)) and
                matrix.get("ledger_thresholds") == LEDGER_THRESHOLDS, "Matrix thresholds changed")
        xml_path, log_path = REPORTS / "g4_tests.xml", REPORTS / "g4_test_results.txt"
        command = [sys.executable, "-m", "pytest", "tests", "validation/hydraulics/sections/tests",
                   "validation/hydraulics/wave_baseline/tests", "-q", f"--junitxml={xml_path}"]
        print("Running G4 reference/ledger/native tests and inherited regression tests ...", flush=True)
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        log_path.write_text(completed.stdout + completed.stderr)
        record.update(test_command=command, tests_returncode=completed.returncode,
                      tests=test_counts(xml_path) if xml_path.exists() else {})
        record["test_artifacts_sha256"] = {str(path.relative_to(ROOT)): native.sha256(path)
                                           for path in (xml_path, log_path) if path.exists()}
        require(completed.returncode == 0 and record["tests"].get("passed", 0) > 0 and
                record["tests"].get("skipped") == 0, f"Qualification tests failed or skipped; see {log_path}")
        print(completed.stdout.strip(), flush=True)
        native.base.configure_runtime()
        record["cases"], repeated = {}, []
        for index, config in enumerate(configs, 1):
            record.update(replay_case_in_progress=config.name, elapsed_wall_s=time.perf_counter()-started)
            write_record(RECORD, record)
            print(f"Rechecking G4 {index}/{len(configs)}: {config.name} ...", flush=True)
            result, observed = replay_case(cases[config.name], identity)
            record["cases"][config.name] = result
            repeated.append(observed)
            record.update(elapsed_wall_s=time.perf_counter()-started)
            write_record(RECORD, record)
            print(f"  Evidence PASS; {result['replay_status']}; individual numerical "
                  f"{'PASS' if result['numerical_pass'] else 'FAIL'}", flush=True)
        record.pop("replay_case_in_progress", None)
        require(callable(getattr(runner, "evaluate_suite", None)), "Runner has no complete suite evaluation API")
        suite = canonical(runner.evaluate_suite(repeated))
        require(matrix.get("suite_evaluation") == suite, "Cached suite comparisons differ from independent recheck")
        numerical = numerical_result(repeated, suite)
        require(matrix["status"] == numerical, "Matrix acceptance does not match its case and comparison gates")
        require(native.frozen_build() == identity, "Active solver identity changed during verification")
        require(native.sha256(matrix_path) == matrix_digest, "Native matrix changed during verification")
        require(qualification_sources() == sources, "Qualification sources changed during verification")
        for name, result in record["cases"].items():
            work = native.RUNS / name / "Wave"
            if result["replay_status"] == "PREREQUISITE_BLOCKED":
                require(authenticate_blocked_case(cases[name], work) == result,
                        f"Blocked prerequisite changed during verification: {name}")
                continue
            require(native.input_hashes(work) == result["input_sha256"] and
                    native.artifact_hashes(work) == result["native_artifact_sha256"] and
                    native.sha256(work / "g4_result.json") == result["result_sha256"] and
                    native.sha256(work / "g4_execution.json") == result["execution_sha256"],
                    f"Case evidence changed during verification: {name}")
        figures = {str(path.relative_to(ROOT)): native.sha256(path)
                   for path in sorted(REPORTS.glob("G4*.png"))}
        record.update(status=numerical, evidence_status="PASS", numerical_status=numerical,
                      fixed_grid_qualified=numerical == "PASS", configuration=matrix["configuration"],
                      native_case_count=len(cases), suite_evaluation=suite, active_solver=identity,
                      native_execution_count=sum(case["execution_status"] != "NOT_PERFORMED" for case in record["cases"].values()),
                      prerequisite_blocked_count=sum(case["execution_status"] == "NOT_PERFORMED" for case in record["cases"].values()),
                      complete_native_output_count=sum(case["replay_status"] == "COMPLETE" for case in record["cases"].values()),
                      complete_individual_pass_count=sum(case["replay_status"] == "COMPLETE" and case["numerical_pass"]
                                                         for case in record["cases"].values()),
                      open_requirements=open_requirements(repeated, suite),
                      coarse_profile_exception_cases=sorted(COARSE_PROFILE_CASES),
                      numerical_maxima=numerical_maxima(record["cases"]),
                      source_sha256=sources, native_matrix_sha256=matrix_digest,
                      diagnostic_figures_sha256=figures, elapsed_wall_s=time.perf_counter()-started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
        write_record(RECORD, record)
        write_summary(record)
        print(f"G4 evidence PASS; numerical gates {numerical}: {RECORD}", flush=True)
        return record
    except BaseException as error:
        record.update(status="FAIL", evidence_status="FAIL", error=f"{type(error).__name__}: {error}",
                      elapsed_wall_s=time.perf_counter()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        write_record(RECORD, record)
        write_summary(record)
        raise


if __name__ == "__main__":
    if run()["status"] != "PASS":
        raise SystemExit(1)
