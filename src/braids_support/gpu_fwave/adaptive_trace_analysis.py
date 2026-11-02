"""Read-only, streaming integrity/summary checks for adaptive filter experiments.

No GPU imports or execution. JSONL rows are paired and discarded immediately;
only two FP64 arrays (accepted dt and CFL) are retained for exact medians.
Passing these checks proves trace completeness, not physical accuracy.
"""
from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(path, expected):
    require(digest(path) == expected, "SHA256 mismatch: " + str(path))


def child(folder, name):
    require(isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name,
            "Artifact must have a flat filename")
    return Path(folder) / name


def finite(value, label, *, nonnegative=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), "Nonfinite/non-numeric " + label)
    require(not nonnegative or value >= 0, "Negative " + label)
    require(not positive or value > 0, "Nonpositive " + label)
    return float(value)


def integer(value, label, *, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer " + label)
    return value


def rows(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(line.strip() != "", f"Empty JSONL row {path}:{number}")
            value = json.loads(line)
            require(isinstance(value, dict), f"Non-object JSONL row {path}:{number}")
            yield value


def number_summary(values):
    if not values:
        return dict(count=0, minimum=None, maximum=None, median=None, mean=None)
    data = np.frombuffer(values, dtype=np.float64)
    return dict(count=len(values), minimum=float(data.min()), maximum=float(data.max()),
                median=float(np.median(data)), mean=float(math.fsum(values) / len(values)))


def histogram_median(histogram):
    count = sum(histogram.values())
    positions = ((count - 1) // 2, count // 2)
    selected, consumed = [], 0
    for value, amount in sorted(histogram.items()):
        selected.extend(value for position in positions if consumed <= position < consumed + amount)
        consumed += amount
    return sum(selected) / 2 if count else None


def check_static(static, selection):
    controls, receipt = static["controls"], static["receipt"]
    require(controls.get("rtol") == 1e-4 and controls.get("atol") == 1e-10 and controls.get("maxiter", 500) == 500,
            "Original filter tolerance/cap changed")
    require(controls.get("depth_factor") == 2. and controls.get("graph_batch") == 5,
            "Source depth factor or original graph-batch request changed")
    finite(controls["dry"], "filter dry", nonnegative=True)
    if selection["filter"] == "native-mic":
        require(receipt.get("rtol") == 1e-4 and receipt.get("atol") == 1e-10 and receipt.get("maxiter") == 500,
                "Native-MIC returned controls differ")
        require(receipt.get("arithmetic") == selection["arithmetic"] and receipt.get("block_size") == selection["block_size"],
                "Native-MIC arithmetic selection differs")
        require(receipt.get("graph_batch_requested") == 5 and receipt.get("graph_batch") == 0,
                "Native-MIC immediate stopping contract differs")
    else:
        require(receipt.get("graph_batch") == 5 and receipt.get("parallel_coarse") is True,
                "Original multilevel filter mode differs")


def new_filter_stats():
    return dict(calls=0, total_iterations=0, iteration_histogram=Counter(), maximum_true_residual=0.,
                maximum_recurrence_residual=0., maximum_tolerance=0., minimum_tolerance=None,
                maximum_true_residual_to_tolerance=0., maximum_recurrence_residual_to_tolerance=0.,
                native_mic_calls=0, numerical_jacobi_fallbacks=0, wall_s=0., all_passed=True)


def add_filter(stats, row):
    receipt = row["receipt"]
    require(receipt.get("passed") is True and "error" not in receipt, "Failed source-filter receipt")
    require(receipt.get("status", 0) == 0, "Nonzero source-filter status")
    count = integer(receipt["iterations"], "filter iterations")
    require(count <= 500, "Filter exceeded original iteration cap")
    tolerance = finite(receipt["tolerance"], "filter tolerance", nonnegative=True)
    stats["calls"] += 1
    stats["total_iterations"] += count
    stats["iteration_histogram"][count] += 1
    stats["maximum_tolerance"] = max(stats["maximum_tolerance"], tolerance)
    stats["minimum_tolerance"] = tolerance if stats["minimum_tolerance"] is None else min(stats["minimum_tolerance"], tolerance)
    for name, limit in (("true_residual", 10.), ("recurrence_residual", 1.)):
        residual = finite(receipt[name], name, nonnegative=True)
        require(residual <= limit * tolerance, "Filter violated original residual contract: " + name)
        ratio = residual / tolerance if tolerance else 0.
        stats["maximum_" + name] = max(stats["maximum_" + name], residual)
        stats["maximum_" + name + "_to_tolerance"] = max(stats["maximum_" + name + "_to_tolerance"], ratio)
    stats["native_mic_calls"] += bool(receipt.get("native_mic_used", False))
    stats["numerical_jacobi_fallbacks"] += bool(receipt.get("numerical_jacobi_fallback", False))
    stats["wall_s"] += finite(row["wall_s"], "filter walltime", nonnegative=True)


def summarize_streams(step_path, filter_path, *, start, dtmax, launch_mode):
    filters = iter(rows(filter_path))
    accepted_dt, accepted_cfl = array("d"), array("d")
    accepted = rejected = trials = warmups = rejected_nonfinite_cfl = 0
    clock = start
    phases = {}
    source_counts = Counter()
    maximum_trial_cfl = None
    for step in rows(step_path):
        phase = step["phase"]
        require(phase in ("warmup_noncommit", "evolution"), "Unexpected trial phase")
        dt = finite(step["dt_s"], "step dt", positive=True)
        current = finite(step["time_s"], "trial time", nonnegative=True)
        require(dt <= dtmax, "Trial exceeds declared dt maximum")
        require(step["accepted_steps_before"] == accepted, "Accepted-step index discontinuity")
        require(integer(step["source_ordinal"], "source count") == 2, "Every SSPRK43 trial requires exactly two sources")
        require(type(step["commit_requested"]) is bool and type(step["accepted"]) is bool, "Nonboolean step acceptance")
        code = integer(step["status"], "trial status")
        if phase == "evolution":
            trials += 1
            require(step["trial"] == trials and step["commit_requested"] is True, "Trial index/commit discontinuity")
            require(current == clock, "Accepted time continuity failed")
            require(step["accepted"] == (code == 0), "Status and acceptance disagree")
        else:
            warmups += 1
            require(trials == 0 and step["trial"] == 0 and current == 0. and not step["commit_requested"] and not step["accepted"],
                    "Warmup committed or appeared during evolution")
        finite(step["wall_s"], "trial walltime", nonnegative=True)
        cfl = step["cfl"]
        if type(cfl) in (int, float) and math.isfinite(cfl):
            require(cfl >= 0, "Negative CFL")
            maximum_trial_cfl = cfl if maximum_trial_cfl is None else max(maximum_trial_cfl, cfl)
        else:
            require(phase == "evolution" and not step["accepted"] and cfl in ("inf", "-inf", "nan"),
                    "Invalid/nonfinite accepted CFL")
            rejected_nonfinite_cfl += 1
        stats = phases.setdefault(phase, new_filter_stats())
        for ordinal in (1, 2):
            source = next(filters, None)
            require(source is not None, "Missing source-filter row")
            for key in ("phase", "trial", "accepted_steps_before", "time_s", "dt_s"):
                require(source.get(key) == step[key], "Source/trial alignment differs: " + key)
            require(source["source_ordinal"] == ordinal, "Source ordinal discontinuity")
            add_filter(stats, source)
        source_counts[2] += 1
        if phase == "evolution":
            if step["accepted"]:
                accepted += 1
                accepted_dt.append(dt)
                accepted_cfl.append(float(cfl))
                clock += dt
            else:
                rejected += 1
    require(next(filters, None) is None, "Unpaired extra source-filter rows")
    require(warmups == (2 if launch_mode == "graph" else 0), "Transport-graph warmup count differs")
    for stats in phases.values():
        histogram = stats["iteration_histogram"]
        stats.update(minimum_iterations=min(histogram) if histogram else None,
                     maximum_iterations=max(histogram) if histogram else None,
                     median_iterations=histogram_median(histogram),
                     mean_iterations=stats["total_iterations"] / stats["calls"] if stats["calls"] else None)
        stats["iteration_histogram"] = {str(key): value for key, value in sorted(histogram.items())}
    total_dt = math.fsum(accepted_dt)
    return dict(accepted=accepted, rejected=rejected, trials=trials, warmup_noncommit_trials=warmups,
                final_clock_s=clock, accepted_dt_sum_s=total_dt, clock_minus_compensated_dt_sum_s=clock - start - total_dt,
                accepted_dt_s=number_summary(accepted_dt), accepted_cfl=number_summary(accepted_cfl),
                maximum_finite_trial_cfl=maximum_trial_cfl, rejected_nonfinite_cfl=rejected_nonfinite_cfl,
                source_count_per_trial_histogram={str(key): value for key, value in source_counts.items()},
                filters=phases, total_filter_wall_s=math.fsum(stats["wall_s"] for stats in phases.values()),
                memory_note="JSONL streamed; only accepted dt/CFL FP64 vectors retained for exact medians")


def analyze(output_path, *, require_complete_20h=False):
    output = Path(output_path).resolve()
    diagnostics = output.with_name(output.name + "_audit")
    audit = read_json(diagnostics / "audit.json")
    require(audit.get("all_identities_unchanged") is True and audit.get("status") in ("completed", "step_limit", "cancelled"),
            "Audit did not finish with intact provenance")
    verify(audit["plan"], audit["plan_sha256"])
    plan = read_json(audit["plan"])
    require(Path(plan["output_path"]).resolve() == output, "Plan output folder differs")
    require(plan["selection"] == audit["selection"], "Plan/audit selection differs")
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
    driver_digests = [value for key, value in plan["authenticated_files"].items() if Path(key).name == "adaptive_filter_candidate.py"]
    require(len(driver_digests) == 1 and manifest.get("audit_driver.py") == driver_digests[0], "Archived wrapper identity differs")
    expected_sources = {**plan["baseline_source_sha256"], **plan["candidate_source_sha256"],
                        "audit_driver.py": driver_digests[0], "audit_plan.json": audit["plan_sha256"]}
    require(manifest == expected_sources, "Frozen runtime source set differs")
    for name, expected in manifest.items():
        verify(child(source, name), expected)
    for name, expected in plan["baseline_source_sha256"].items():
        verify(child(plan["baseline_source"], name), expected)
    static = read_json(diagnostics / "filter_controls.json")
    check_static(static, selected)
    if selected["filter"] == "native-mic":
        require(static["receipt"]["source_sha256"] == manifest["native_filter_candidate.cu"], "Executed MIC CUDA identity differs")
        native_ids = [value for key, value in plan["authenticated_files"].items()
                      if Path(key).name == "braids_curvature_regularization_module.f90"]
        require(native_ids == [static["receipt"]["native_source_sha256"]], "Frozen native method identity differs")
    worker = read_json(output / "run.json")
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
    return dict(schema=1, verified=True, output=str(output), status=worker["status"], selection=selected,
                complete_20h=complete, complete_20h_trace=complete, physical_accuracy_qualified=False,
                scope="Trace/provenance/completion only; root report must assess fields and conservation against reference",
                original_filter_controls_verified=True, final_checkpoint_finite=True,
                audit_total_wall_s=audit["audit_total_wall_s"], worker_total_wall_s=worker["total_wall_s"],
                audit_sha256=digest(diagnostics / "audit.json"), run_sha256=digest(output / "run.json"),
                checkpoint_sha256=digest(checkpoint), plan_sha256=audit["plan_sha256"],
                frozen_manifest_sha256=audit["frozen_manifest_sha256"], static_filter_controls=static, **result)
