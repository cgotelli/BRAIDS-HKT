"""Read-only v2 workspace provenance adapter; v1 streaming algorithms reused.

Unlike v1, the active CUDA identity is selected explicitly by workspace. Both
the inherited MIC files and actual shared implementation must be authenticated.
No archived record is rewritten or substituted to satisfy the checks.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .adaptive_trace_analysis import (digest, read_json, require, verify, child, finite,
                                     check_static, summarize_streams)

HERE = Path(__file__).resolve().parent
WRAPPER_FILES = ("adaptive_filter_candidate_v2.py", "adaptive_trace_analysis.py", "adaptive_trace_analysis_v2.py")


def check_workspace(plan, static, manifest, cells):
    chosen, receipt = plan["selection"], static["receipt"]
    workspace = chosen.get("workspace")
    triangular = chosen.get("triangular")
    require(workspace in ("global", "shared-z", "shared-z-pivot"), "Explicit workspace missing")
    require(triangular in ("division", "reciprocal"), "Explicit triangular arithmetic missing")
    require(triangular != "reciprocal" or (chosen["filter"] == "native-mic" and workspace != "global"),
            "Unsupported reciprocal/global selection")
    require(chosen["filter"] == "native-mic" or workspace == "global", "Shared workspace is not multilevel")
    active = "secondary_curvature.cu" if chosen["filter"] == "multilevel" else (
        "native_filter_reciprocal.cu" if triangular == "reciprocal" else
        "native_filter_candidate.cu" if workspace == "global" else "native_filter_shared.cu")
    require(plan.get("active_filter_cuda") == active, "Selected active CUDA filename differs")
    if chosen["filter"] == "native-mic":
        require(receipt["source_sha256"] == manifest[active], "Executed selected CUDA identity differs")
        native_ids = [value for key, value in plan["authenticated_files"].items()
                      if Path(key).name == "braids_curvature_regularization_module.f90"]
        require(native_ids == [receipt["native_source_sha256"]], "Frozen native method identity differs")
    if workspace != "global":
        require(receipt.get("workspace") == ("z" if workspace == "shared-z" else "z_pivot"), "Executed workspace differs")
        require(receipt.get("inherited_python_sha256") == manifest["native_filter_candidate.py"], "Inherited MIC Python differs")
        require(receipt.get("original_cuda_sha256") == manifest["native_filter_candidate.cu"], "Inherited MIC CUDA differs")
        require(receipt.get("storage_only_change") is (triangular == "division") and
                receipt.get("global_diagnostic_z_and_optional_pivot_not_populated") is True,
                "Shared storage/arithmetic contract differs")
        if triangular == "reciprocal":
            require(receipt.get("pivot_arithmetic") == "precompute FP64 reciprocal once; multiply in triangular sweeps" and
                    receipt.get("inherited_shared_python_sha256") == manifest["native_filter_shared.py"],
                    "Reciprocal arithmetic/inherited shared identity differs")
        else:
            require("pivot_arithmetic" not in receipt, "Unexpected reciprocal arithmetic")
        dynamic = finite(receipt["dynamic_shared_bytes"], "dynamic shared bytes", positive=True)
        static_bytes = finite(receipt["static_shared_bytes"], "static shared bytes", nonnegative=True)
        limit = finite(receipt["device_shared_limit_bytes"], "device shared limit", positive=True)
        require(dynamic == 8 * cells * (1 if workspace == "shared-z" else 2), "Shared allocation size differs")
        require(dynamic + static_bytes <= limit, "Shared allocation exceeded device guard")
    else:
        require("workspace" not in receipt, "Unexpected shared implementation in global selection")
    return active


def analyze(output_path, *, require_complete_20h=False):
    output = Path(output_path).resolve()
    diagnostics = output.with_name(output.name + "_audit")
    audit = read_json(diagnostics / "audit.json")
    require(audit.get("all_identities_unchanged") is True and audit.get("status") in ("completed", "step_limit", "cancelled"),
            "Audit did not finish with intact provenance")
    verify(audit["plan"], audit["plan_sha256"])
    plan = read_json(audit["plan"])
    require(plan.get("wrapper_schema") == 2 and plan.get("wrapper_entrypoint") == "adaptive_filter_candidate_v2.py",
            "V2 wrapper plan discriminator required")
    require(plan.get("base_wrapper_archive") == "audit_driver.py" and
            plan.get("workspace_implementation_only") is (plan["selection"].get("triangular") == "division"),
            "V2 wrapper inheritance/storage scope differs")
    require(plan.get("triangular_arithmetic_changed") is (plan["selection"].get("triangular") == "reciprocal"),
            "V2 arithmetic change declaration differs")
    require(Path(plan["output_path"]).resolve() == output and plan["selection"] == audit["selection"], "Plan/output/selection differs")
    selected = plan["selection"]
    require(selected["filter"] in ("multilevel", "native-mic") and selected.get("backend") == "gpu" and
            selected.get("automatic_backend_selection") is False, "Backend/filter selection changed")
    require((selected["rtol"], selected["atol"], selected["maxiter"]) == (1e-4, 1e-10, 500), "Filter controls changed")
    require((plan["physical_start_s"], plan["cfl"], plan["morfac"], plan["dt_max_s"], plan["output_every_s"]) ==
            (0., .8, 25., .0125, 24.), "Cold adaptive experiment controls changed")
    require(audit.get("production_modified") is False and audit.get("delft_modified") is False, "Reference modification declared")
    require(audit["protected_files"] == plan["protected_files"], "Protected reference set differs")
    for path, expected in {**plan["authenticated_files"], **plan["protected_files"]}.items():
        verify(path, expected)
    needed = {"step_trials.jsonl", "filter_calls.jsonl", "filter_controls.json"}
    require(needed <= audit["artifacts"].keys(), "Missing scalar trace/static identity")
    for name, expected in audit["artifacts"].items():
        verify(child(diagnostics, name), expected)
    source = output / "source"
    verify(source / "manifest.json", audit["frozen_manifest_sha256"])
    manifest = read_json(source / "manifest.json")
    require(manifest.get("audit_plan.json") == audit["plan_sha256"], "Archived plan identity differs")
    driver_ids = [value for key, value in plan["authenticated_files"].items() if Path(key).name == "adaptive_filter_candidate.py"]
    require(len(driver_ids) == 1 and manifest.get("audit_driver.py") == driver_ids[0], "Archived v1 wrapper identity differs")
    expected_sources = {**plan["baseline_source_sha256"], **plan["candidate_source_sha256"],
                        "audit_driver.py": driver_ids[0], "audit_plan.json": audit["plan_sha256"]}
    require(manifest == expected_sources, "Frozen runtime source set differs")
    for name, expected in manifest.items():
        verify(child(source, name), expected)
    for name in WRAPPER_FILES:
        require(plan["candidate_source_sha256"].get(name) == digest(HERE / name) and
                plan["authenticated_files"].get(str(HERE / name)) == digest(HERE / name), "Unpinned v2 dependency: " + name)
    for name, expected in plan["baseline_source_sha256"].items():
        verify(child(plan["baseline_source"], name), expected)
    static = read_json(diagnostics / "filter_controls.json")
    check_static(static, selected)
    worker = read_json(output / "run.json")
    active = check_workspace(plan, static, manifest, worker["cells"])
    require(worker["source_sha256"] == manifest and worker["status"] == audit["status"], "Worker source/status differs")
    for key, plan_key in (("end_s", "until_s"), ("dtmax_s", "dt_max_s"), ("output_every_s", "output_every_s"),
                           ("cfl", "cfl"), ("morfac", "morfac"), ("launch_mode", "launch_mode")):
        require(worker[key] == plan[plan_key], "Worker control differs: " + key)
    result = summarize_streams(diagnostics / "step_trials.jsonl", diagnostics / "filter_calls.jsonl",
                               start=plan["physical_start_s"], dtmax=plan["dt_max_s"], launch_mode=plan["launch_mode"])
    for key, worker_key in (("accepted", "steps"), ("rejected", "rejected")):
        require(result[key] == audit[key] == worker[worker_key] == audit["worker_" + worker_key], "Worker/observer count mismatch: " + key)
    require(result["trials"] == audit["trials"] and result["trials"] == result["accepted"] + result["rejected"], "Trial count mismatch")
    final = finite(worker["final_time_s"], "worker final time", nonnegative=True)
    require(result["final_clock_s"] == final == audit["final_time_s"], "Final time differs from accepted trial sum")
    require(abs(result["clock_minus_compensated_dt_sum_s"]) <= 1e-8 * max(1., final / 2880.), "Accepted dt sum inconsistent with clock")
    for phase, stats in result["filters"].items():
        recorded = audit["filter_statistics"][phase]
        for key in ("calls", "total_iterations", "maximum_iterations", "maximum_true_residual", "maximum_recurrence_residual", "wall_s", "all_passed"):
            require(stats[key] == recorded[key], "Recorded filter aggregate differs: " + phase + "/" + key)
    checkpoint = output / "checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        required = {"q", "layers", "secondary", "bed_elevation", "ledger_water", "ledger_bed", "ledger_secondary", "ledger_source"}
        require(required <= set(data.files), "Final checkpoint lacks required state/budgets")
        for key in required:
            require(np.isfinite(data[key]).all(), "Nonfinite final checkpoint field: " + key)
        require(float(data["time"]) == final and int(data["steps"]) == result["accepted"] and int(data["rejected"]) == result["rejected"],
                "Final checkpoint clock/counts differ")
        require(str(data["source_identity"].item()) == audit["frozen_manifest_sha256"], "Checkpoint source identity differs")
        require(str(data["input_identity"].item()) == worker["input_preparation_sha256"], "Checkpoint input identity differs")
    complete = worker["status"] == "completed" and plan["until_s"] == 2880. and final == 2880.
    if require_complete_20h:
        require(complete, "A completed finite 20 morphological hour run is required; short smoke is not sufficient")
    for key in ("audit_total_wall_s", "worker_total_wall_s"):
        finite(audit[key], key, nonnegative=True)
    require(audit["worker_total_wall_s"] == worker["total_wall_s"], "Worker timing differs")
    return dict(schema=1, wrapper_schema=2, verified=True, output=str(output), status=worker["status"], selection=selected,
                complete_20h=complete, complete_20h_trace=complete, physical_accuracy_qualified=False,
                scope="Trace/provenance/completion only; root report must assess fields and conservation against reference",
                original_filter_controls_verified=True, final_checkpoint_finite=True,
                active_filter_cuda=active, active_filter_cuda_sha256=manifest[active],
                audit_total_wall_s=audit["audit_total_wall_s"], worker_total_wall_s=audit["worker_total_wall_s"],
                audit_sha256=digest(diagnostics / "audit.json"), run_sha256=digest(output / "run.json"),
                checkpoint_sha256=digest(checkpoint), plan_sha256=audit["plan_sha256"],
                frozen_manifest_sha256=audit["frozen_manifest_sha256"], static_filter_controls=static, **result)
