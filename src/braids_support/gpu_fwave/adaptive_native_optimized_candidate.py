"""Schema4: explicitly selected FP64 native-MIC application optimizations.

Research only; original worker/controller and physical numerical sources stay
unchanged. Candidate ancestry is executed from registered, authenticated copies.
No CUDA is imported by prepare. Hardware scheduling is the caller's authority.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import time

from . import adaptive_filter_candidate as base

HERE = Path(__file__).resolve().parent
V1_SELECTION = base.selection
REQUESTED = dict(rtol=1e-4, atol=1e-10, maxiter=500, graph_batch=5)
EFFECTIVE = dict(REQUESTED, graph_batch=0)
HELPERS = ("adaptive_native_optimized_candidate.py", "adaptive_native_optimized_trace.py", "adaptive_trace_analysis.py")
VARIANTS = {
    "fma": ("native_filter_fma", "FMANativeMICCurvature"),
    "rownormalized": ("native_filter_rownormalized", "RowNormalizedNativeMICCurvature"),
    "rownormalized-fma": ("native_filter_rownormalized_fma", "RowNormalizedFMANativeMICCurvature"),
}


def selection(variant):
    if variant not in VARIANTS:
        raise ValueError("Explicit fma, rownormalized or rownormalized-fma variant required")
    result = V1_SELECTION("native-mic", "parallel_tree", 256)
    result.update(variant=variant, workspace="shared-z-pivot", triangular="reciprocal",
                  numerical_change="FP64 MIC triangular application: " + variant + "; rounding may differ, no FP32")
    return result


def module_chain(variant):
    selection(variant)
    chain = ["native_filter_candidate", "native_filter_shared", "native_filter_reciprocal"]
    if variant.startswith("rownormalized"):
        chain.append("native_filter_rownormalized")
    if variant != "rownormalized":
        chain.append(VARIANTS[variant][0])
    return tuple(chain)


def active_cuda(variant):
    return VARIANTS[selection(variant)["variant"]][0] + ".cu"


def candidate_files(variant):
    return tuple(stem + suffix for stem in module_chain(variant) for suffix in (".py", ".cu"))


def stop_path(output, requested=None):
    output = Path(output).resolve()
    target = Path(requested).resolve() if requested is not None else output.with_name(output.name + ".stop")
    # Keep the mutable cancellation signal outside the frozen output/source set.
    if target.parent != output.parent or target in (output, output.with_name(output.name + "_audit")):
        raise ValueError("Stop signal must be a separate file beside the output folder")
    return target


def _load(path, stem, installed):
    name = "braids_support.gpu_fwave._schema4_frozen_" + stem
    if name in sys.modules:
        raise ValueError("Frozen private module already registered: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    installed.append(name)
    spec.loader.exec_module(module)
    return module


def frozen_factory(folder, variant, installed):
    """Bind the complete archived class/helper inheritance; restore aliases."""
    folder = Path(folder).resolve()
    aliases = {}
    modules = {}
    try:
        # Even the base geometry functions and compilation option tuple bind
        # to the frozen helper, not an incidental live module import.
        for stem in ("secondary_curvature",) + module_chain(variant):
            module = _load(folder / (stem + ".py"), stem, installed)
            modules[stem] = module
            alias = "braids_support.gpu_fwave." + stem
            aliases[alias] = sys.modules.get(alias)
            sys.modules[alias] = module
        parent = modules["native_filter_candidate"]
        parent.PARENT = base.NATIVE
        if parent.NATIVE_SOURCE_SHA256 != base.NATIVE_SHA:
            raise ValueError("Frozen native provenance differs")
    finally:
        for alias, previous in reversed(tuple(aliases.items())):
            if previous is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous
    stem, class_name = VARIANTS[variant]
    constructor = getattr(modules[stem], class_name)

    def construct(data, *, source_path, arithmetic, block_size):
        if (Path(source_path).resolve() != folder / "native_filter_candidate.cu" or
                arithmetic != "parallel_tree" or block_size != 256):
            raise ValueError("Unexpected original factory request")
        return constructor(data, source_path=folder / active_cuda(variant), arithmetic=arithmetic,
                           block_size=block_size, workspace="z_pivot")
    # Host tests can inspect the actual archived class without constructing CUDA.
    construct.archived_constructor = constructor
    construct.archived_modules = modules
    return construct


def original_controls(controls):
    effective = dict(controls, maxiter=controls.get("maxiter", 500))
    if any(effective.get(key) != value for key, value in REQUESTED.items()):
        raise ValueError("Original requested filter tolerance/batch/cap changed")
    if controls.get("depth_factor") != 2.:
        raise ValueError("Original depth factor changed")
    return effective


class OriginalObservedFilter:
    def __init__(self, original, traces):
        self.original, self.traces = original, traces

    def solve(self, h, raw, **controls):
        requested = original_controls(controls)
        began = time.perf_counter()
        try:
            value, receipt = self.original.solve(h, raw, **controls)
        except Exception as exc:
            self.traces.filter_record(controls, dict(passed=False, error=str(exc)), time.perf_counter() - began)
            raise
        receipt = dict(receipt, requested_controls=requested, effective_controls=EFFECTIVE.copy(),
                       original_source_request_unchanged=True, CPU_reference_retuned=False)
        self.traces.filter_record(controls, receipt, time.perf_counter() - began)
        return value, receipt


class ControlTraces(base.Traces):
    def close(self):
        result = super().close()
        path = self.destination / "stop_control.json"
        if path.exists():
            result["artifacts"][path.name] = base.sha(path)
        return result


@contextmanager
def adapters(variant, *, plan=None):
    chosen = selection(variant)
    saved = base.selection, base.frozen_candidate, base.ObservedFilter, base.Traces, base.worker.run_gpu
    installed = []

    def choose(name, arithmetic="native_order", block_size=128):
        if name != "native-mic" or arithmetic != "parallel_tree" or block_size != 256:
            raise ValueError("Schema4 requires explicit native-mic parallel_tree/256")
        return chosen.copy()

    def run_gpu(**kwargs):
        signal = stop_path(plan["output_path"], plan["stop_file"])
        if signal.exists() or "stop_file" in kwargs or Path(kwargs["output_path"]).resolve() != Path(plan["output_path"]):
            raise ValueError("Stop control already exists or worker request differs")
        evidence_path = Path(plan["output_path"]).with_name(Path(plan["output_path"]).name + "_audit") / "stop_control.json"
        evidence = dict(path=str(signal), initially_absent=True, passed_to_original_worker=True,
                        original_worker_sha256=plan["baseline_source_sha256"]["backend_runner.py"], status="running")
        base.write_json(evidence_path, evidence)
        try:
            result = saved[4](**kwargs, stop_file=str(signal))
            evidence.update(status=result["status"], final_time_s=result["final_time_s"])
            return result
        except Exception as exc:
            evidence.update(status="failed", error=str(exc))
            raise
        finally:
            evidence["signal_present_at_return"] = signal.exists()
            evidence["signal_sha256_at_return"] = base.sha(signal) if signal.is_file() else None
            base.write_json(evidence_path, evidence)

    base.selection = choose
    base.frozen_candidate = lambda folder: frozen_factory(folder, variant, installed)
    base.ObservedFilter, base.Traces = OriginalObservedFilter, ControlTraces
    if plan is not None:
        base.worker.run_gpu = run_gpu
    try:
        yield
    finally:
        base.selection, base.frozen_candidate, base.ObservedFilter, base.Traces, base.worker.run_gpu = saved
        for name in reversed(installed):
            sys.modules.pop(name, None)


def prepare(plan_path, output_path, *, variant, until=2880., max_steps=None, launch_mode="graph", stop_file=None):
    chosen = selection(variant)
    signal = stop_path(output_path, stop_file)
    if signal.exists():
        raise FileExistsError("Stop signal must be absent before plan preparation")
    plan_path = Path(plan_path).resolve()
    with adapters(variant):
        result = base.prepare(plan_path, output_path, filter_name="native-mic", arithmetic="parallel_tree", block_size=256,
                              until=until, max_steps=max_steps, launch_mode=launch_mode)
    plan = json.loads(plan_path.read_text())
    for name in ("execution.json", "analysis_geometry.npz", "input_provenance.json"):
        path = base.CPU_REFERENCE / name
        plan["protected_files"][str(path)] = base.sha(path)
    for name in HELPERS + candidate_files(variant):
        identity = base.sha(HERE / name)
        plan["candidate_source_sha256"][name] = identity
        plan["authenticated_files"][str(HERE / name)] = identity
    plan.update(wrapper_schema=4, wrapper_entrypoint=Path(__file__).name, base_wrapper_archive="audit_driver.py",
                requested_filter_controls=REQUESTED.copy(), effective_filter_controls=EFFECTIVE.copy(),
                active_filter_cuda=active_cuda(variant), candidate_module_chain=list(module_chain(variant)),
                CPU_reference_retuned=False, original_numeric_sources_unchanged=True,
                candidate_numerical_scope="FP64 MIC application reassociation/FMA only; original operator, factorization, Krylov and stopping",
                accepted_CFL_maximum=.8, stop_file=str(signal), stop_file_initially_absent=True,
                stop_contract="Presence-only signal passed to unchanged worker; checked at accepted transaction boundary; never deleted by wrapper")
    base.write_json(plan_path, plan)
    return dict(result, sha256=base.sha(plan_path), wrapper_schema=4, selection=chosen, stop_file=str(signal))


def run(plan_path, approved_sha256):
    plan_path = Path(plan_path).resolve()
    if base.sha(plan_path) != approved_sha256:
        raise ValueError("Reviewed schema4 plan SHA required")
    plan = json.loads(plan_path.read_text())
    if plan.get("wrapper_schema") != 4 or plan.get("wrapper_entrypoint") != Path(__file__).name:
        raise ValueError("Explicit schema4 plan discriminator required")
    chosen = selection(plan["selection"]["variant"])
    if (plan["selection"] != chosen or plan["requested_filter_controls"] != REQUESTED or
            plan["effective_filter_controls"] != EFFECTIVE or plan["active_filter_cuda"] != active_cuda(chosen["variant"]) or
            plan["candidate_module_chain"] != list(module_chain(chosen["variant"]))):
        raise ValueError("Declared numerical selection or controls differ")
    if plan.get("stop_file_initially_absent") is not True or stop_path(plan["output_path"], plan["stop_file"]).exists():
        raise ValueError("Plan-pinned cancellation signal must be absent at launch")
    for name in HELPERS + candidate_files(chosen["variant"]):
        identity = base.sha(HERE / name)
        if plan["candidate_source_sha256"].get(name) != identity or plan["authenticated_files"].get(str(HERE / name)) != identity:
            raise ValueError("Unpinned schema4 dependency: " + name)
    with adapters(chosen["variant"], plan=plan):
        return base.run(plan_path, approved_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--until", type=float, default=2880.)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--launch-mode", choices=("graph", "plain"), default="graph")
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--approved-plan-sha256", required=True)
    args = parser.parse_args()
    result = (prepare(args.plan, args.output, variant=args.variant, until=args.until, max_steps=args.max_steps,
                      launch_mode=args.launch_mode, stop_file=args.stop_file) if args.command == "prepare"
              else run(args.plan, args.approved_plan_sha256))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
