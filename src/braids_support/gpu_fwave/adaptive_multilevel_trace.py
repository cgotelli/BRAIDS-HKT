"""Read-only schema3 verification with truthful requested/effective tolerances.

Reuses immutable streaming trial checks. No source record is rewritten to look
like the old1e-4 contract; finite accepted CFL must additionally be <=0.8.
"""
from pathlib import Path
import numpy as np

from .adaptive_trace_analysis import digest, read_json, require, verify, child, finite, summarize_streams

HERE = Path(__file__).resolve().parent
HELPERS = ("adaptive_multilevel_candidate.py", "adaptive_multilevel_trace.py", "adaptive_trace_analysis.py")
REQUESTED = dict(rtol=1e-4, atol=1e-10, maxiter=500, graph_batch=5)


def check_controls(plan, static):
    chosen = plan["selection"]
    require(chosen["filter"] == "multilevel" and chosen.get("backend") == "gpu" and
            chosen.get("automatic_backend_selection") is False, "Explicit original multilevel GPU required")
    require(chosen["rtol"] in (1e-6, 1e-8) and chosen["atol"] == 1e-10 and chosen["maxiter"] == 500,
            "Declared tighter-MG tolerance/cap differs")
    effective = dict(REQUESTED, rtol=chosen["rtol"])
    require(plan["requested_filter_controls"] == REQUESTED and plan["effective_filter_controls"] == effective,
            "Requested/effective plan controls differ")
    controls, receipt = static["controls"], static["receipt"]
    require({key: controls.get(key) for key in effective} == effective, "Executed effective controls differ")
    requested = receipt.get("requested_controls", {})
    require({key: requested.get(key) for key in REQUESTED} == REQUESTED, "Original source request differs")
    require(requested.get("depth_factor") == controls.get("depth_factor") == 2. and requested.get("dry") == controls.get("dry"),
            "Dry/depth-factor controls changed")
    finite(controls["dry"], "filter dry", nonnegative=True)
    require(receipt.get("effective_tolerance") == effective and receipt.get("graph_batch") == 5 and
            receipt.get("parallel_coarse") is True, "MG numerical mode differs")
    require(receipt.get("original_source_request_unchanged") is True and receipt.get("CPU_reference_retuned") is False,
            "Original request/native reference modification declared")
    return effective


def accepted_cfl_gate(trace, maximum=.8):
    value = trace["accepted_cfl"]["maximum"]
    require(value is not None and finite(value, "maximum accepted CFL", nonnegative=True) <= maximum,
            "Accepted CFL exceeds declared0.8 limit")
    return True


def complete_20h(status, until, final):
    """Match the established121-output grid tolerance, not binary clock identity."""
    return bool(status == "completed" and until == 2880. and np.isfinite(final) and abs(final - 2880.) <= 1e-7)


def analyze(output_path, *, require_complete_20h=False):
    output = Path(output_path).resolve()
    diagnostics = output.with_name(output.name + "_audit")
    audit = read_json(diagnostics / "audit.json")
    require(audit.get("all_identities_unchanged") is True and audit.get("status") in ("completed", "step_limit", "cancelled"),
            "MG audit did not finish with intact provenance")
    verify(audit["plan"], audit["plan_sha256"])
    plan = read_json(audit["plan"])
    require(plan.get("wrapper_schema") == 3 and plan.get("wrapper_entrypoint") == "adaptive_multilevel_candidate.py",
            "Explicit MG wrapper schema3 required")
    require(plan.get("base_wrapper_archive") == "audit_driver.py" and plan.get("original_numeric_sources_unchanged") is True and
            plan.get("CPU_reference_retuned") is False, "Reference/source change declared")
    require(Path(plan["output_path"]).resolve() == output and plan["selection"] == audit["selection"], "Plan/output/selection differs")
    require((plan["physical_start_s"], plan["cfl"], plan["morfac"], plan["dt_max_s"], plan["output_every_s"], plan["accepted_CFL_maximum"]) ==
            (0., .8, 25., .0125, 24., .8), "Original cold-run controls changed")
    require(audit.get("production_modified") is False and audit.get("delft_modified") is False, "Reference modification declared")
    require(audit["protected_files"] == plan["protected_files"], "Protected reference set differs")
    for path, expected in {**plan["authenticated_files"], **plan["protected_files"]}.items():
        verify(path, expected)
    require({"step_trials.jsonl", "filter_calls.jsonl", "filter_controls.json"} <= audit["artifacts"].keys(), "Missing scalar audit artifacts")
    for name, expected in audit["artifacts"].items():
        verify(child(diagnostics, name), expected)
    source = output / "source"
    verify(source / "manifest.json", audit["frozen_manifest_sha256"])
    manifest = read_json(source / "manifest.json")
    ids = [value for key, value in plan["authenticated_files"].items() if Path(key).name == "adaptive_filter_candidate.py"]
    require(len(ids) == 1, "Missing unique base driver identity")
    require(manifest == {**plan["baseline_source_sha256"], **plan["candidate_source_sha256"],
                         "audit_driver.py": ids[0], "audit_plan.json": audit["plan_sha256"]}, "Archived source set differs")
    for name, expected in manifest.items():
        verify(child(source, name), expected)
    for name in HELPERS:
        require(plan["candidate_source_sha256"].get(name) == digest(HERE / name) and
                plan["authenticated_files"].get(str(HERE / name)) == digest(HERE / name), "Unpinned MG dependency: " + name)
    for name, expected in plan["baseline_source_sha256"].items():
        verify(child(plan["baseline_source"], name), expected)
    require(plan["active_filter_cuda"] == "secondary_curvature.cu" and
            manifest["secondary_curvature.cu"] == plan["baseline_source_sha256"]["secondary_curvature.cu"], "Original MG CUDA changed")
    static = read_json(diagnostics / "filter_controls.json")
    effective = check_controls(plan, static)
    worker = read_json(output / "run.json")
    require(worker["source_sha256"] == manifest and worker["status"] == audit["status"], "Worker identity/status differs")
    for key, plan_key in (("end_s", "until_s"), ("dtmax_s", "dt_max_s"), ("output_every_s", "output_every_s"),
                           ("cfl", "cfl"), ("morfac", "morfac"), ("launch_mode", "launch_mode")):
        require(worker[key] == plan[plan_key], "Worker control differs: " + key)
    trace = summarize_streams(diagnostics / "step_trials.jsonl", diagnostics / "filter_calls.jsonl",
                              start=0., dtmax=.0125, launch_mode=plan["launch_mode"])
    accepted_cfl_gate(trace, plan["accepted_CFL_maximum"])
    for key, worker_key in (("accepted", "steps"), ("rejected", "rejected")):
        require(trace[key] == audit[key] == worker[worker_key] == audit["worker_" + worker_key], "Worker/observer count differs")
    require(trace["trials"] == audit["trials"] == trace["accepted"] + trace["rejected"], "Trial count mismatch")
    final = finite(worker["final_time_s"], "final time", nonnegative=True)
    require(trace["final_clock_s"] == final == audit["final_time_s"], "Final time differs from accepted trials")
    require(abs(trace["clock_minus_compensated_dt_sum_s"]) <= 1e-8 * max(1., final / 2880.), "Accepted dt sum differs")
    for phase, stats in trace["filters"].items():
        for key in ("calls", "total_iterations", "maximum_iterations", "maximum_true_residual", "maximum_recurrence_residual", "wall_s", "all_passed"):
            require(stats[key] == audit["filter_statistics"][phase][key], "Recorded filter aggregate differs")
    checkpoint = output / "checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        for key in ("q", "layers", "secondary", "bed_elevation", "ledger_water", "ledger_bed", "ledger_secondary", "ledger_source"):
            require(np.isfinite(data[key]).all(), "Nonfinite final checkpoint: " + key)
        require(float(data["time"]) == final and int(data["steps"]) == trace["accepted"] and int(data["rejected"]) == trace["rejected"], "Checkpoint clock/count differs")
        require(data["source_identity"].item() == audit["frozen_manifest_sha256"] and
                data["input_identity"].item() == worker["input_preparation_sha256"], "Checkpoint identity differs")
    complete = complete_20h(worker["status"], plan["until_s"], final)
    if require_complete_20h:
        require(complete, "Full completed20h required; bounded smoke is not long-run qualification")
    require(audit["worker_total_wall_s"] == worker["total_wall_s"], "Worker timing differs")
    finite(audit["audit_total_wall_s"], "inclusive walltime", nonnegative=True)
    finite(audit["worker_total_wall_s"], "worker walltime", nonnegative=True)
    return dict(schema=1, wrapper_schema=3, verified=True, complete_20h=complete, completion_time_tolerance_s=1e-7, physical_accuracy_qualified=False,
                accepted_CFL_gate_passed=True, selection=plan["selection"], requested_filter_controls=REQUESTED,
                effective_filter_controls=effective, CPU_reference_retuned=False, original_numeric_sources_unchanged=True,
                active_filter_cuda_sha256=manifest["secondary_curvature.cu"],
                audit_total_wall_s=audit["audit_total_wall_s"], worker_total_wall_s=worker["total_wall_s"],
                audit_sha256=digest(diagnostics / "audit.json"), run_sha256=digest(output / "run.json"),
                checkpoint_sha256=digest(checkpoint), plan_sha256=audit["plan_sha256"], frozen_manifest_sha256=audit["frozen_manifest_sha256"],
                static_filter_controls=static, **trace)
