"""Isolated, explicit curvature-filter choice around the unchanged GPU worker.

`prepare` is CPU-only. `run` requires the reviewed plan SHA and an exclusive
hardware window from the coordinating caller. Production is never patched.
The original adaptive controller, graph, fluxes and reconstruction are retained.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np

from braids_gpu import runner as worker

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LONG = HERE / "runs/gpu_bend20h_20261001_v1"
NATIVE = ROOT / "braids_support/sediment/lfm_bend_2d/runs/fast_cpu_default_20261001_v1/runtime_mf25"
NATIVE_NAME = "braids_curvature_regularization_module.f90"
NATIVE_SHA = "f8770b50a2a5eb7c2cf188ac46a62a5fcaad5bc1ecf59df4ac48a3d80f63c253"
CPU_REFERENCE = ROOT / "braids_support/sediment/lfm_bend_2d/runs/fast_cpu_cold20h_20261001_v1_mobile_braids"
PROTECTED = {
    "solver/cut_cells/build/xbraids_cutcell.exe": "9b1fd6d23a0721069c4a168f60a45a7e261283fe16356a0d14f126f756c5f273",
    "dist/windows-x86_64/BRAIDS-0.1.0-alpha.82-QGIS-windows-x86_64.zip": "773308c4c010f18f2acd6f29344d647c7244fb1c597c21b12d0efec5c8cda544",
    "dist/gpu-worker/BRAIDS-GPU-0.1.0-experimental.1-windows-x86_64.zip": "0a46e36c60eea0c81ef91b0e97892d4b15ae1d92313404fe81d2d55d28cd3854",
}
sha, write_json = worker.sha, worker.write_json


def selection(filter_name, arithmetic="native_order", block_size=128):
    if filter_name not in ("multilevel", "native-mic"):
        raise ValueError("Explicit filter must be multilevel or native-mic")
    if arithmetic not in ("native_order", "parallel_tree") or block_size not in (32, 64, 128, 256):
        raise ValueError("Explicit candidate arithmetic and supported block size required")
    return dict(filter=filter_name, arithmetic=arithmetic if filter_name == "native-mic" else None,
                block_size=block_size if filter_name == "native-mic" else None,
                backend="gpu", selected_by="user/test", automatic_backend_selection=False,
                rtol=1.e-4, atol=1.e-10, maxiter=500,
                numerical_change="MIC preconditioner/order only" if filter_name == "native-mic" else "none")


def verify_baseline(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest or "backend_runner.py" not in manifest:
        raise ValueError("Missing frozen worker identity")
    for name, digest in manifest.items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Source manifest requires flat filenames")
        live = ROOT / "braids_gpu" / name.removeprefix("backend_") if name.startswith("backend_") else HERE / name
        if sha(folder / name) != digest or sha(live) != digest:
            raise ValueError("Frozen/live original source differs: " + name)
    return manifest


def verify_files(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Authenticated file changed: " + str(path))


def prepare(plan_path, output_path, *, filter_name, arithmetic="native_order", block_size=128,
            input_path=LONG / "prepared", baseline=LONG / "gpu/source", until=2880.,
            max_steps=None, launch_mode="graph"):
    """Pin inputs, original sources, candidate and driver without importing CUDA."""
    chosen = selection(filter_name, arithmetic, block_size)
    if not np.isfinite(until) or not 0 < until <= 2880:
        raise ValueError("Require 0 < until <= 2880 physical seconds")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Positive integer max_steps required")
    if launch_mode not in ("graph", "plain"):
        raise ValueError("Explicit graph or plain launch required")
    plan_path, output_path, input_path, baseline = map(lambda p: Path(p).resolve(),
                                                    (plan_path, output_path, input_path, baseline))
    if plan_path.exists() or output_path.exists():
        raise FileExistsError("Plan and output must both be new")
    original = verify_baseline(baseline)
    data, identity = worker.load_prepared(input_path)
    if identity != sha(LONG / "prepared/preparation_receipt.json"):
        raise ValueError("Require the preserved long-run cold prepared input")
    if float(data.get("time", 0.)) != 0. or float(data["cfl_target"]) != .8 or float(data["morphological_factor"]) != 25.:
        raise ValueError("Cold time=0, original CFL=0.8 and MF=25 required")
    source_files = {str(baseline / "manifest.json"): sha(baseline / "manifest.json"),
                    str(Path(__file__).resolve()): sha(__file__), str(NATIVE / NATIVE_NAME): NATIVE_SHA}
    candidate = {}
    if filter_name == "native-mic":
        candidate = {name: sha(HERE / name) for name in ("native_filter_candidate.py", "native_filter_candidate.cu")}
        source_files.update({str(HERE / name): digest for name, digest in candidate.items()})
    receipt = json.loads((input_path / "preparation_receipt.json").read_text())
    source_files[str(input_path / "preparation_receipt.json")] = identity
    source_files.update({str(input_path / name): digest for name, digest in receipt["artifacts"].items()})
    protected = {str(ROOT / name): digest for name, digest in PROTECTED.items()}
    protected[str(NATIVE / "xbraids_candidate.exe")] = "9325aae9e8acdbbe9ed091044a17e847bc04881bd924e83d137565947b07c590"
    protected.update({str(CPU_REFERENCE / name): sha(CPU_REFERENCE / name)
                      for name in ("frames.bin", "checkpoint.bin", "secondary_flow.txt", "secondary_flow_budget.txt")})
    verify_files({**source_files, **protected})
    plan = dict(schema=1, selection=chosen, output_path=str(output_path), input_path=str(input_path),
                baseline_source=str(baseline), baseline_source_sha256=original, candidate_source_sha256=candidate,
                authenticated_files=source_files, protected_files=protected,
                until_s=float(until), dt_max_s=.0125, output_every_s=24., max_steps=max_steps,
                launch_mode=launch_mode, physical_start_s=0., cfl=.8, morfac=25.,
                diagnostic_scope="Every step trial and source filter receipt; no state-field downloads beyond original output",
                walltime_scope="Includes diagnostic scalar synchronization, trace writes, freezing and final verification",
                qualification="Research candidate, not production adoption or proof of long-run agreement",
                production_modified=False, delft_modified=False)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    return dict(plan=str(plan_path), sha256=sha(plan_path), output=str(output_path), gpu_executed=False)


class Traces:
    """Buffered append-only scalar diagnostics; constant Python memory usage."""
    def __init__(self, destination):
        self.destination = Path(destination)
        self.steps = (self.destination / "step_trials.jsonl").open("x", encoding="utf-8", buffering=65536)
        self.filters = (self.destination / "filter_calls.jsonl").open("x", encoding="utf-8", buffering=65536)
        self.context = dict(phase="setup", trial=0, accepted_steps_before=0, source_ordinal=0)
        self.accepted = self.rejected = self.trials = 0
        self.statistics = {}
        self.filter_static = None

    def begin_step(self, dt, current, commit):
        if commit:
            self.trials += 1
        self.context = dict(phase="evolution" if commit else "warmup_noncommit", trial=self.trials if commit else 0,
                            accepted_steps_before=self.accepted, source_ordinal=0, time_s=float(current), dt_s=float(dt))

    @staticmethod
    def emit(stream, row):
        # Diagnostics must not prevent the original retry controller from
        # handling a nonfinite rejected trial. Preserve these explicitly as
        # strings, never coerce them into finite values or round finite data.
        def safe(value):
            if isinstance(value, dict):
                return {key: safe(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [safe(item) for item in value]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return str(value)
            return value
        stream.write(json.dumps(safe(row), allow_nan=False, separators=(",", ":")) + "\n")

    def filter_record(self, controls, receipt, elapsed):
        self.context["source_ordinal"] += 1
        dynamic_names = {"iterations", "recurrence_residual", "true_residual", "tolerance", "passed", "status",
                         "native_mic_used", "numerical_jacobi_fallback", "error"}
        if "error" not in receipt:
            static = dict(controls=controls, receipt={key: value for key, value in receipt.items() if key not in dynamic_names})
            if self.filter_static is None:
                self.filter_static = static
                write_json(self.destination / "filter_controls.json", static)
            elif self.filter_static != static:
                raise ValueError("Filter algorithm, controls or source identity changed during run")
        row = dict(**self.context, receipt={key: value for key, value in receipt.items() if key in dynamic_names}, wall_s=elapsed)
        self.emit(self.filters, row)
        phase = self.context["phase"]
        stats = self.statistics.setdefault(phase, dict(calls=0, total_iterations=0, maximum_iterations=0,
                                                     maximum_true_residual=0., maximum_recurrence_residual=0.,
                                                     wall_s=0., all_passed=True))
        stats["calls"] += 1
        stats["total_iterations"] += int(receipt.get("iterations", 0))
        stats["maximum_iterations"] = max(stats["maximum_iterations"], int(receipt.get("iterations", 0)))
        for name in ("true_residual", "recurrence_residual"):
            stats["maximum_" + name] = max(stats["maximum_" + name], float(receipt.get(name, 0.)))
        stats["wall_s"] += elapsed
        stats["all_passed"] &= bool(receipt.get("passed", False))

    def end_step(self, result, elapsed, commit):
        code, cfl = int(result["status"].item()), float(result["cfl"].item())
        self.emit(self.steps, dict(**self.context, commit_requested=bool(commit), status=code, cfl=cfl,
                                  accepted=bool(commit and code == 0), wall_s=elapsed))
        if commit:
            self.accepted += code == 0
            self.rejected += code != 0

    def failed_step(self, exc, elapsed, commit):
        self.emit(self.steps, dict(**self.context, commit_requested=bool(commit), accepted=False,
                                  status="exception", error=str(exc), wall_s=elapsed))

    def close(self):
        self.steps.close()
        self.filters.close()
        names = ["step_trials.jsonl", "filter_calls.jsonl"]
        if self.filter_static is not None:
            names.append("filter_controls.json")
        return dict(trials=self.trials, accepted=self.accepted, rejected=self.rejected,
                    filter_statistics=self.statistics,
                    artifacts={name: sha(self.destination / name) for name in names})


class ObservedFilter:
    def __init__(self, original, trace):
        self.original, self.trace = original, trace

    def solve(self, h, raw, **controls):
        began = time.perf_counter()
        try:
            result, receipt = self.original.solve(h, raw, **controls)
        except Exception as exc:
            self.trace.filter_record(controls, dict(passed=False, error=str(exc)), time.perf_counter() - began)
            raise
        self.trace.filter_record(controls, receipt, time.perf_counter() - began)
        return result, receipt


def frozen_candidate(folder):
    """Execute the authenticated COPY; relative helper imports are baseline-checked."""
    path = Path(folder) / "native_filter_candidate.py"
    spec = importlib.util.spec_from_file_location("braids_support.gpu_fwave._frozen_native_filter_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Relocation only: retain the immutable native provenance target, not a
    # spurious path inferred from the new archived Python module's directory.
    module.PARENT = NATIVE
    if module.NATIVE_SOURCE_SHA256 != NATIVE_SHA:
        raise ValueError("Candidate native provenance guard differs")
    return module.NativeMICCurvature


@contextmanager
def isolated_worker(plan, plan_path, traces):
    from . import coupled_solver
    original_solver, original_freeze = coupled_solver.CoupledSolver, worker.freeze_sources
    chosen = plan["selection"]

    def freeze(destination):
        manifest = original_freeze(destination)
        if manifest != plan["baseline_source_sha256"]:
            raise ValueError("Worker source set changed since plan review")
        extras = {"audit_driver.py": Path(__file__), "audit_plan.json": Path(plan_path)}
        extras.update({name: HERE / name for name in plan["candidate_source_sha256"]})
        for name, path in extras.items():
            shutil.copy2(path, Path(destination) / name)
            digest = sha(path)
            expected = plan["candidate_source_sha256"].get(name, plan["authenticated_files"].get(str(path.resolve()), digest))
            if digest != expected or sha(Path(destination) / name) != digest:
                raise ValueError("Audit source changed while freezing: " + name)
            manifest[name] = digest
        write_json(Path(destination) / "manifest.json", manifest)
        return manifest

    class AuditSolver(original_solver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if chosen["filter"] == "native-mic":
                candidate = frozen_candidate(self.folder)
                self.secondary.filter = candidate(self.data, source_path=self.folder / "native_filter_candidate.cu",
                                                  arithmetic=chosen["arithmetic"], block_size=chosen["block_size"])
            self.secondary.filter = ObservedFilter(self.secondary.filter, traces)

        def step(self, dt, current, *, commit=True):
            traces.begin_step(dt, current, commit)
            began = time.perf_counter()
            try:
                result = super().step(dt, current, commit=commit)
            except Exception as exc:
                traces.failed_step(exc, time.perf_counter() - began, commit)
                raise
            traces.end_step(result, time.perf_counter() - began, commit)
            return result

    worker.freeze_sources, coupled_solver.CoupledSolver = freeze, AuditSolver
    try:
        yield
    finally:
        worker.freeze_sources, coupled_solver.CoupledSolver = original_freeze, original_solver


def run(plan_path, approved_sha256):
    """Call only after the coordinating caller has granted a GPU window."""
    began = time.perf_counter()
    plan_path = Path(plan_path).resolve()
    if sha(plan_path) != approved_sha256:
        raise ValueError("Reviewed plan SHA is required")
    plan = json.loads(plan_path.read_text())
    if plan["selection"] != selection(plan["selection"]["filter"], plan["selection"].get("arithmetic") or "native_order",
                                      plan["selection"].get("block_size") or 128):
        raise ValueError("Invalid explicit selection")
    if verify_baseline(plan["baseline_source"]) != plan["baseline_source_sha256"]:
        raise ValueError("Original source manifest changed")
    verify_files({**plan["authenticated_files"], **plan["protected_files"]})
    destination = Path(plan["output_path"])
    if destination.exists():
        raise FileExistsError(destination)
    diagnostics = destination.with_name(destination.name + "_audit")
    diagnostics.mkdir(parents=True, exist_ok=False)
    traces = Traces(diagnostics)
    receipt = dict(schema=1, plan=str(plan_path), plan_sha256=approved_sha256, selection=plan["selection"],
                   protected_files=plan["protected_files"], production_modified=False, delft_modified=False,
                   extra_step_scalar_synchronizations=True, diagnostic_walltime_included=True,
                   copied_candidate_native_provenance_path=str(NATIVE), status="running")
    write_json(diagnostics / "audit.json", receipt)
    try:
        with isolated_worker(plan, plan_path, traces):
            result = worker.run_gpu(input_path=plan["input_path"], output_path=destination,
                                    until=plan["until_s"], output_every=plan["output_every_s"],
                                    dt_max=plan["dt_max_s"], max_steps=plan["max_steps"], launch_mode=plan["launch_mode"])
        receipt.update(status=result["status"], worker_total_wall_s=result["total_wall_s"],
                       final_time_s=result["final_time_s"], worker_steps=result["steps"], worker_rejected=result["rejected"])
        if traces.accepted != result["steps"] or traces.rejected != result["rejected"]:
            raise RuntimeError("Observer and unchanged worker trial counts disagree")
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        receipt.update(traces.close())
        try:
            verify_files({**plan["authenticated_files"], **plan["protected_files"], str(plan_path): approved_sha256})
            verify_baseline(plan["baseline_source"])
            source = destination / "source"
            if source.exists():
                manifest = json.loads((source / "manifest.json").read_text())
                verify_files({str(source / name): digest for name, digest in manifest.items()})
                receipt["frozen_manifest_sha256"] = sha(source / "manifest.json")
            receipt["all_identities_unchanged"] = True
        except Exception as exc:
            receipt.update(status="provenance_failed", all_identities_unchanged=False, provenance_error=str(exc))
            raise
        finally:
            receipt["audit_total_wall_s"] = time.perf_counter() - began
            receipt["timing_note"] = "Includes traces and final identity hashing; excludes only this final small receipt write"
            write_json(diagnostics / "audit.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--filter", choices=("multilevel", "native-mic"), required=True)
    p.add_argument("--arithmetic", choices=("native_order", "parallel_tree"), default="native_order")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--until", type=float, default=2880.)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--launch-mode", choices=("graph", "plain"), default="graph")
    p = subs.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--approved-plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.plan, args.output, filter_name=args.filter, arithmetic=args.arithmetic,
                         block_size=args.block_size, until=args.until, max_steps=args.max_steps, launch_mode=args.launch_mode)
    else:
        result = run(args.plan, args.approved_plan_sha256)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
