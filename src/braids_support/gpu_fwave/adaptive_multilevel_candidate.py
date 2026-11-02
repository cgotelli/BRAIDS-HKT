"""Isolated tighter multilevel filter around the unchanged adaptive GPU worker.

Original source requests1e-4; this explicit experiment uses1e-8 (or1e-6).
Requested and effective controls are both recorded; CPU reference is untouched.
No shader, preconditioner, reconstruction, boundary or transport method changes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

from . import adaptive_filter_candidate as base

HERE = Path(__file__).resolve().parent
V1_SELECTION = base.selection
REQUESTED = dict(rtol=1e-4, atol=1e-10, maxiter=500, graph_batch=5)
HELPERS = ("adaptive_multilevel_candidate.py", "adaptive_multilevel_trace.py", "adaptive_trace_analysis.py")


def selection(rtol):
    if rtol not in (1e-6, 1e-8):
        raise ValueError("Explicit prequalified1e-6 or1e-8 relative tolerance required")
    chosen = V1_SELECTION("multilevel")
    chosen.update(rtol=rtol, numerical_change="Only GPU multilevel relative tolerance changes from1e-4; unchanged atol/cap/operator/shaders")
    return chosen


def effective_controls(rtol, requested):
    if requested.get("rtol") != 1e-4 or requested.get("atol") != 1e-10 or requested.get("maxiter", 500) != 500:
        raise ValueError("Unexpected original source tolerance/cap")
    if requested.get("graph_batch") != 5 or requested.get("depth_factor") != 2.:
        raise ValueError("Unexpected source batch/depth factor")
    selection(rtol)
    return dict(requested, rtol=rtol, atol=1e-10, maxiter=500)


class ToleranceObservedFilter:
    def __init__(self, original, trace, rtol):
        self.original, self.trace, self.rtol = original, trace, rtol

    def solve(self, h, raw, **requested):
        effective = effective_controls(self.rtol, requested)
        began = time.perf_counter()
        try:
            field, receipt = self.original.solve(h, raw, **effective)
        except Exception as exc:
            self.trace.filter_record(effective, dict(passed=False, error=str(exc)), time.perf_counter() - began)
            raise
        receipt = dict(receipt, requested_controls=dict(requested, maxiter=requested.get("maxiter", 500)),
                       effective_tolerance=dict(rtol=self.rtol, atol=1e-10, maxiter=500, graph_batch=5),
                       original_source_request_unchanged=True, CPU_reference_retuned=False)
        self.trace.filter_record(effective, receipt, time.perf_counter() - began)
        return field, receipt


@contextmanager
def adapters(rtol):
    chosen = selection(rtol)
    original_selection, original_observer = base.selection, base.ObservedFilter
    def select(name, arithmetic="native_order", block_size=128):
        if name != "multilevel":
            raise ValueError("Tighter-MG adapter cannot choose another filter/backend")
        return chosen.copy()
    base.selection = select
    base.ObservedFilter = lambda original, trace: ToleranceObservedFilter(original, trace, rtol)
    try:
        yield
    finally:
        base.selection, base.ObservedFilter = original_selection, original_observer


def prepare(plan_path, output_path, *, rtol=1e-8, until=2880., max_steps=None, launch_mode="graph"):
    chosen = selection(rtol)
    plan_path = Path(plan_path).resolve()
    with adapters(rtol):
        result = base.prepare(plan_path, output_path, filter_name="multilevel", until=until,
                              max_steps=max_steps, launch_mode=launch_mode)
    plan = json.loads(plan_path.read_text())
    # The frozen CPU timing is a later speed-gate denominator, so protect it
    # together with the already pinned CPU fields and budgets.
    cpu_timing = base.CPU_REFERENCE / "execution.json"
    plan["protected_files"][str(cpu_timing)] = base.sha(cpu_timing)
    for name in HELPERS:
        identity = base.sha(HERE / name)
        plan["candidate_source_sha256"][name] = identity
        plan["authenticated_files"][str(HERE / name)] = identity
    plan.update(wrapper_schema=3, wrapper_entrypoint=Path(__file__).name, base_wrapper_archive="audit_driver.py",
                requested_filter_controls=REQUESTED, effective_filter_controls=dict(REQUESTED, rtol=rtol),
                original_numeric_sources_unchanged=True, CPU_reference_retuned=False,
                accepted_CFL_maximum=.8, active_filter_cuda="secondary_curvature.cu",
                interpretation="Tighter independent GPU solve, not an exact reproduction of frozen loose CPU finite iterations")
    base.write_json(plan_path, plan)
    return dict(result, sha256=base.sha(plan_path), wrapper_schema=3, selection=chosen)


def run(plan_path, approved_sha256):
    plan_path = Path(plan_path).resolve()
    if base.sha(plan_path) != approved_sha256:
        raise ValueError("Reviewed MG candidate plan SHA required")
    plan = json.loads(plan_path.read_text())
    if plan.get("wrapper_schema") != 3 or plan.get("wrapper_entrypoint") != Path(__file__).name:
        raise ValueError("Explicit MG wrapper schema3 required")
    chosen = selection(plan["selection"]["rtol"])
    if plan["selection"] != chosen or plan["requested_filter_controls"] != REQUESTED or plan["effective_filter_controls"] != dict(REQUESTED, rtol=chosen["rtol"]):
        raise ValueError("Declared requested/effective MG controls differ")
    for name in HELPERS:
        identity = base.sha(HERE / name)
        if plan["candidate_source_sha256"].get(name) != identity or plan["authenticated_files"].get(str(HERE / name)) != identity:
            raise ValueError("Unpinned MG wrapper dependency: " + name)
    with adapters(chosen["rtol"]):
        return base.run(plan_path, approved_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rtol", type=float, choices=(1e-6, 1e-8), required=True)
    p.add_argument("--until", type=float, default=2880.)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--launch-mode", choices=("graph", "plain"), default="graph")
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--approved-plan-sha256", required=True)
    args = parser.parse_args()
    result = (prepare(args.plan, args.output, rtol=args.rtol, until=args.until, max_steps=args.max_steps, launch_mode=args.launch_mode)
              if args.command == "prepare" else run(args.plan, args.approved_plan_sha256))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
