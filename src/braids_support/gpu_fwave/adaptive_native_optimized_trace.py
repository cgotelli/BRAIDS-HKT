"""Streaming schema4 identity/control/trace verifier; no CUDA execution."""
from pathlib import Path

import numpy as np

from .adaptive_trace_analysis import digest, read_json, require, verify, child, finite, summarize_streams
from .adaptive_native_optimized_candidate import (HERE, HELPERS, REQUESTED, EFFECTIVE, selection,
                                                  module_chain, candidate_files, active_cuda, stop_path, base)


def accepted_cfl_gate(trace, maximum=.8):
    value = trace["accepted_cfl"]["maximum"]
    require(value is not None and finite(value, "accepted CFL", nonnegative=True) <= maximum,
            "Accepted CFL exceeds declared0.8 limit")
    return True


def complete_20h(status, until, final):
    return bool(status == "completed" and until == 2880. and np.isfinite(final) and abs(final - 2880.) <= 1e-7)


def check_controls(plan, static, manifest, cells):
    chosen = selection(plan["selection"]["variant"])
    require(plan["selection"] == chosen, "Explicit numerical selection differs")
    require(plan["requested_filter_controls"] == REQUESTED and plan["effective_filter_controls"] == EFFECTIVE,
            "Requested/effective original controls differ")
    controls, receipt = static["controls"], static["receipt"]
    actual_request = dict(controls, maxiter=controls.get("maxiter", 500))
    require({key: actual_request.get(key) for key in REQUESTED} == REQUESTED,
            "Source did not request original controls")
    require(receipt.get("requested_controls") == actual_request and receipt.get("effective_controls") == EFFECTIVE,
            "Executed requested/effective controls differ")
    require(controls.get("depth_factor") == 2., "Depth factor changed")
    finite(controls["dry"], "dry tolerance", nonnegative=True)
    require((receipt.get("rtol"), receipt.get("atol"), receipt.get("maxiter"), receipt.get("graph_batch"),
             receipt.get("graph_batch_requested")) == (1e-4, 1e-10, 500, 0, 5), "Native immediate-stopping controls differ")
    require(receipt.get("arithmetic") == "parallel_tree" and receipt.get("block_size") == 256,
            "Selected arithmetic/block size differs")
    require(receipt.get("original_source_request_unchanged") is True and receipt.get("CPU_reference_retuned") is False,
            "Original request/reference change declared")
    active = active_cuda(chosen["variant"])
    require(plan["active_filter_cuda"] == active and receipt["source_sha256"] == manifest[active],
            "Executed selected CUDA identity differs")
    require(receipt.get("native_source_sha256") == base.NATIVE_SHA, "Frozen native algorithm identity differs")
    for key, name in (("inherited_python_sha256", "native_filter_candidate.py"),
                      ("original_cuda_sha256", "native_filter_candidate.cu"),
                      ("inherited_shared_python_sha256", "native_filter_shared.py"),
                      ("inherited_reciprocal_python_sha256", "native_filter_reciprocal.py")):
        require(receipt.get(key) == manifest[name], "Inherited executed source differs: " + key)
    require(receipt.get("workspace") == "z_pivot" and receipt.get("storage_only_change") is False and
            receipt.get("global_diagnostic_z_and_optional_pivot_not_populated") is True,
            "Shared workspace/rounding contract differs")
    dynamic = finite(receipt["dynamic_shared_bytes"], "dynamic shared bytes", positive=True)
    static_bytes = finite(receipt["static_shared_bytes"], "static shared bytes", nonnegative=True)
    limit = finite(receipt["device_shared_limit_bytes"], "shared device limit", positive=True)
    require(dynamic == 16 * cells and dynamic + static_bytes <= limit, "Shared workspace guard differs")
    variant = chosen["variant"]
    if variant == "fma":
        require(receipt.get("triangular_subtraction") == "explicit FP64 __fma_rn(-factor,previous,value), one rounding instead of two" and
                receipt.get("factorization_operator_PCG_arithmetic") == "unchanged FP64; original --fmad=false retained",
                "Explicit FMA-only arithmetic receipt differs")
        require(receipt.get("pivot_arithmetic") == "precompute FP64 reciprocal once; multiply in triangular sweeps" and
                "row_normalized_triangular_application" not in receipt, "Unexpected normalization in FMA-only selection")
    else:
        require(receipt.get("row_normalized_triangular_application") is True and receipt.get("row_normalization_per_solve") is True and
                receipt.get("normalized_coefficients_dtype") == "float64" and
                receipt.get("factorization_operator_krylov_and_jacobi_unchanged") is True,
                "Row-normalized pure FP64 contract differs")
        require(receipt.get("inherited_reciprocal_cuda_sha256") == manifest["native_filter_reciprocal.cu"],
                "Normalized reciprocal CUDA guard differs")
        require(receipt.get("explicit_fma") is (variant == "rownormalized-fma"), "Explicit normalized FMA choice differs")
        require(receipt.get("pivot_arithmetic") == "FP64 reciprocal once; pre-scale each triangular row before dependency subtraction",
                "Normalized pivot arithmetic differs")
        size = finite(receipt["normalized_workspace_bytes"], "normalized allocation", nonnegative=True)
        require(size % 16 == 0, "Normalized workspace allocation differs")
        if variant == "rownormalized-fma":
            require(receipt.get("inherited_normalized_python_sha256") == manifest["native_filter_rownormalized.py"] and
                    receipt.get("inherited_normalized_cuda_sha256") == manifest["native_filter_rownormalized.cu"],
                    "Normalized FMA parent identity differs")
    require("preconditioner_factor_dtype" not in receipt and "residual_corrections" not in receipt,
            "Unselected mixed-precision implementation")
    return active


def analyze(output_path, *, require_complete_20h=False):
    output = Path(output_path).resolve()
    diagnostics = output.with_name(output.name + "_audit")
    audit = read_json(diagnostics / "audit.json")
    require(audit.get("all_identities_unchanged") is True and audit.get("status") in ("completed", "step_limit", "cancelled"),
            "Schema4 audit did not finish with intact provenance")
    verify(audit["plan"], audit["plan_sha256"])
    plan = read_json(audit["plan"])
    require(plan.get("wrapper_schema") == 4 and plan.get("wrapper_entrypoint") == "adaptive_native_optimized_candidate.py" and
            plan.get("base_wrapper_archive") == "audit_driver.py", "Schema4 plan discriminator differs")
    require(plan.get("CPU_reference_retuned") is False and plan.get("original_numeric_sources_unchanged") is True,
            "Reference/original source change declared")
    require(Path(plan["output_path"]).resolve() == output and plan["selection"] == audit["selection"], "Plan/output/selection differs")
    chosen = selection(plan["selection"]["variant"])
    require(plan["selection"] == chosen and plan["candidate_module_chain"] == list(module_chain(chosen["variant"])),
            "Explicit candidate ancestry differs")
    require((plan["physical_start_s"], plan["cfl"], plan["morfac"], plan["dt_max_s"], plan["output_every_s"], plan["accepted_CFL_maximum"]) ==
            (0., .8, 25., .0125, 24., .8), "Original adaptive experiment controls changed")
    require(audit.get("production_modified") is False and audit.get("delft_modified") is False, "Reference modification declared")
    require(audit["protected_files"] == plan["protected_files"], "Protected reference set differs")
    for name in ("execution.json", "analysis_geometry.npz", "input_provenance.json"):
        require(str(base.CPU_REFERENCE / name) in plan["protected_files"], "CPU comparison provenance not protected: " + name)
    for path, expected in {**plan["authenticated_files"], **plan["protected_files"]}.items():
        verify(path, expected)
    require({"step_trials.jsonl", "filter_calls.jsonl", "filter_controls.json", "stop_control.json"} <= audit["artifacts"].keys(),
            "Missing scalar trace/control artifact")
    for name, expected in audit["artifacts"].items():
        verify(child(diagnostics, name), expected)
    source = output / "source"
    verify(source / "manifest.json", audit["frozen_manifest_sha256"])
    manifest = read_json(source / "manifest.json")
    drivers = [value for path, value in plan["authenticated_files"].items() if Path(path).name == "adaptive_filter_candidate.py"]
    require(len(drivers) == 1, "Unique v1 driver identity required")
    require(manifest == {**plan["baseline_source_sha256"], **plan["candidate_source_sha256"],
                         "audit_driver.py": drivers[0], "audit_plan.json": audit["plan_sha256"]}, "Archived source set differs")
    for name, expected in manifest.items():
        verify(child(source, name), expected)
    for name in HELPERS + candidate_files(chosen["variant"]):
        require(plan["candidate_source_sha256"].get(name) == digest(HERE / name) and
                plan["authenticated_files"].get(str(HERE / name)) == digest(HERE / name), "Unpinned schema4 dependency: " + name)
    for name, expected in plan["baseline_source_sha256"].items():
        verify(child(plan["baseline_source"], name), expected)
    worker = read_json(output / "run.json")
    require(worker["source_sha256"] == manifest and worker["status"] == audit["status"], "Worker source/status differs")
    for key, plan_key in (("end_s", "until_s"), ("dtmax_s", "dt_max_s"), ("output_every_s", "output_every_s"),
                           ("cfl", "cfl"), ("morfac", "morfac"), ("launch_mode", "launch_mode")):
        require(worker[key] == plan[plan_key], "Worker control differs: " + key)
    static = read_json(diagnostics / "filter_controls.json")
    active = check_controls(plan, static, manifest, worker["cells"])
    control = read_json(diagnostics / "stop_control.json")
    require(plan.get("stop_file_initially_absent") is True and
            control["path"] == str(stop_path(output, plan["stop_file"])) and
            control.get("initially_absent") is True and control.get("passed_to_original_worker") is True and
            control["original_worker_sha256"] == manifest["backend_runner.py"] and control["status"] == worker["status"],
            "Plan-pinned stop control execution differs")
    trace = summarize_streams(diagnostics / "step_trials.jsonl", diagnostics / "filter_calls.jsonl",
                              start=0., dtmax=.0125, launch_mode=plan["launch_mode"])
    accepted_cfl_gate(trace)
    for key, worker_key in (("accepted", "steps"), ("rejected", "rejected")):
        require(trace[key] == audit[key] == worker[worker_key] == audit["worker_" + worker_key], "Worker/observer count differs")
    require(trace["trials"] == audit["trials"] == trace["accepted"] + trace["rejected"], "Trial count differs")
    final = finite(worker["final_time_s"], "final time", nonnegative=True)
    require(trace["final_clock_s"] == final == audit["final_time_s"] == control["final_time_s"], "Final clock differs")
    require(abs(trace["clock_minus_compensated_dt_sum_s"]) <= 1e-8 * max(1., final / 2880.), "Accepted dt sum differs")
    for phase, stats in trace["filters"].items():
        require(stats["native_mic_calls"] == stats["calls"] and stats["numerical_jacobi_fallbacks"] == 0,
                "Source used undeclared fallback")
        for key in ("calls", "total_iterations", "maximum_iterations", "maximum_true_residual", "maximum_recurrence_residual", "wall_s", "all_passed"):
            require(stats[key] == audit["filter_statistics"][phase][key], "Filter aggregate differs")
    checkpoint = output / "checkpoint.npz"
    with np.load(checkpoint, allow_pickle=False) as data:
        for key in ("q", "layers", "secondary", "bed_elevation", "ledger_water", "ledger_bed", "ledger_secondary", "ledger_source"):
            require(np.isfinite(data[key]).all(), "Nonfinite final checkpoint: " + key)
        require(float(data["time"]) == final and int(data["steps"]) == trace["accepted"] and int(data["rejected"]) == trace["rejected"],
                "Checkpoint clock/count differs")
        require(data["source_identity"].item() == audit["frozen_manifest_sha256"] and
                data["input_identity"].item() == worker["input_preparation_sha256"], "Checkpoint identity differs")
    complete = complete_20h(worker["status"], plan["until_s"], final)
    if require_complete_20h:
        require(complete, "Completed finite20h required; bounded smoke/cancellation is not full qualification")
    require(audit["worker_total_wall_s"] == worker["total_wall_s"], "Worker timing differs")
    finite(audit["audit_total_wall_s"], "inclusive walltime", nonnegative=True)
    finite(audit["worker_total_wall_s"], "worker walltime", nonnegative=True)
    return dict(schema=1, wrapper_schema=4, verified=True, complete_20h=complete, completion_time_tolerance_s=1e-7,
                physical_accuracy_qualified=False, accepted_CFL_gate_passed=True,
                selection=chosen, requested_filter_controls=REQUESTED, effective_filter_controls=EFFECTIVE,
                CPU_reference_retuned=False, original_numeric_sources_unchanged=True,
                active_filter_cuda=active, active_filter_cuda_sha256=manifest[active], stop_control=control,
                audit_total_wall_s=audit["audit_total_wall_s"], worker_total_wall_s=worker["total_wall_s"],
                audit_sha256=digest(diagnostics / "audit.json"), run_sha256=digest(output / "run.json"),
                checkpoint_sha256=digest(checkpoint), plan_sha256=audit["plan_sha256"], frozen_manifest_sha256=audit["frozen_manifest_sha256"],
                static_filter_controls=static, **trace)
