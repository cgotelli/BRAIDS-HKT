"""One corrected-CPU late24s continuation with ONLY tighter filter rtol.

The acceptance reference is unchanged. This private research executable is
derived from the authenticated corrected CPU; its native checkpoint and deck
are copied byte-for-byte from the approved baseline timestep study.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

from . import outlet_bounds_cpu20h as cpu
from . import outlet_bounds_timestep_sensitivity as study
from . import coupled_reference as cr

base = cpu.base
HERE, ROOT = base.HERE, base.ROOT
TAG = "cpu_filter_convergence_control_20261002_v1"
BASELINE_PLAN = HERE / "reports/outlet_bounds_timestep_20261002_v1/plan.json"
BASELINE_SHA = "8849340c9696f1f2168c832eb67ae6b6afd885a892b785f3f4d2008e56daf1ba"
OLD = b"relative_tolerance=1.0000000000000000d-04,absolute_tolerance=1.0000000000000000d-10"
NEW = b"relative_tolerance=1.0000000000000000d-08,absolute_tolerance=1.0000000000000000d-10"


def transform(source):
    if not isinstance(source, bytes) or source.count(OLD) != 1 or NEW in source:
        raise ValueError("Expected exactly one unchanged native filter control call")
    return source.replace(OLD, NEW)


def verify_transform(original, candidate):
    if candidate != transform(original) or candidate.replace(NEW, OLD) != original:
        raise ValueError("Unexpected source edit beyond the filter tolerance literal")


def dependencies():
    from braids_qgis.core.results import read_frames
    import inspect
    paths = [Path(module.__file__).resolve() for module in
             (cpu, base, study, study.adapter, study.analysis, cr)]
    paths += [Path(__file__).resolve(), Path(inspect.getfile(read_frames)).resolve()]
    return {str(path): base.sha(path) for path in paths}


def prepare():
    if base.sha(BASELINE_PLAN) != BASELINE_SHA:
        raise ValueError("Approved baseline sensitivity plan changed")
    original_study = json.loads(BASELINE_PLAN.read_text()); study.validate(original_study)
    baseline_entry = original_study["children"]["baseline"]
    baseline = json.loads(Path(baseline_entry["path"]).read_text())
    if (baseline["dt_max_s"], baseline["initial_time_s"], baseline["end_time_s"]) != (.0125, 2880., 2904.):
        raise ValueError("Require exact approved baseline24s continuation")
    parent, protected = cpu.verify_runtime()
    root, report = HERE / "runs" / TAG, HERE / "reports" / TAG
    if root.exists() or report.exists():
        raise FileExistsError("Fresh filter convergence paths required")
    root.mkdir(); report.mkdir(); runtime = root / "runtime"; runtime.mkdir()
    sources = []
    for name, digest in parent["source_sha256"].items():
        source, target = cpu.RUNTIME / name, runtime / name
        if name == "cutcell_solver.f90":
            target.write_bytes(transform(source.read_bytes()))
            verify_transform(source.read_bytes(), target.read_bytes())
        else:
            shutil.copy2(source, target)
            if base.sha(target) != digest:
                raise ValueError("Unchanged native source copy differs")
        sources.append(target)
    for source in cpu.RUNTIME.glob("*.dll"):
        shutil.copy2(source, runtime / source.name)
    source_root = Path(baseline["prepared_input"]).parent
    prepared = root / "prepared"; prepared.mkdir()
    for name in ("native_checkpoint.bin", "exact_initial.npz"):
        shutil.copy2(source_root / name, root / name)
        protected[str(source_root / name)] = base.sha(source_root / name)
        if base.sha(root / name) != protected[str(source_root / name)]:
            raise ValueError("Checkpoint/state copy differs")
    for name in ("coupled_input.bin", "coupled_diffusion.bin", "preparation_receipt.json"):
        source = source_root / "prepared" / name
        shutil.copy2(source, prepared / name); protected[str(source)] = base.sha(source)
    case = root / "cpu"
    cr.copy_inputs(Path(baseline["CPU_output"]) / "case.data", case)
    for name in ("case.data", "case.data.bin", "case.data.json"):
        source = Path(baseline["CPU_output"]) / name
        if not (case / name).exists():
            shutil.copy2(source, case / name)
        protected[str(source)] = base.sha(source)
        if base.sha(case / name) != protected[str(source)]:
            raise ValueError("Baseline input bytes differ")
    compiler = Path(r"C:\Strawberry\c\bin\gfortran.exe")
    executable = runtime / "xbraids_filter_tight.exe"
    command = [str(compiler), "-O3", "-flto", "-fopenmp", "-ffree-line-length-none", "-ffixed-line-length-none",
               *map(str, sources), "-o", str(executable)]
    child = dict(baseline, CPU_output=str(case), prepared_input=str(prepared))
    protected.update({str(BASELINE_PLAN): BASELINE_SHA, str(Path(baseline_entry["path"])): baseline_entry["sha256"]})
    authenticated = dependencies()
    for folder in (runtime, prepared, case):
        authenticated.update({str(p): base.sha(p) for p in folder.iterdir() if p.is_file()})
    authenticated.update({str(root / name): base.sha(root / name) for name in ("native_checkpoint.bin", "exact_initial.npz")})
    authenticated[str(compiler)] = base.sha(compiler)
    shutil.copy2(__file__, report / Path(__file__).name)
    authenticated[str(report / Path(__file__).name)] = base.sha(report / Path(__file__).name)
    plan = dict(schema=1, tag=TAG, root=str(root), report=str(report), child=child,
        baseline_study_plan=str(BASELINE_PLAN), baseline_study_sha256=BASELINE_SHA,
        corrected_parent_runtime=str(cpu.RUNTIME), corrected_parent_identity=parent,
        protected_files=protected, authenticated_files=authenticated,
        private_source_sha256={p.name: base.sha(p) for p in sources}, build_command=command,
        executable=str(executable), native_command=[str(executable), "case.data", str(root / "native_checkpoint.bin")],
        only_source_change=dict(file="cutcell_solver.f90", before=OLD.decode(), after=NEW.decode()),
        controls=dict(rtol=1.e-8, atol=1.e-10, CFL=.8, MF=25., dtmax=.0125, begin=2880., end=2904., output=24., threads=8),
        checkpoint_and_inputs_byte_identical=True, reference_replaced=False, production_modified=False, Delft3D_modified=False,
        scope="One local filter-convergence control; not the CPU acceptance reference or preceding20h convergence",
        expected_cost="One20-40s compile and approximately40-60s CPU diagnostic; not a benchmark",
        limits_unchanged=dict(study.analysis.LIMITS))
    validate(plan); base.save_json(report / "plan.json", plan)
    return dict(plan=str(report / "plan.json"), sha256=base.sha(report / "plan.json"), jobs_launched=0)


def validate(plan):
    if (plan["tag"] != TAG or plan["controls"] != dict(rtol=1.e-8, atol=1.e-10, CFL=.8, MF=25., dtmax=.0125, begin=2880., end=2904., output=24., threads=8)
            or plan["limits_unchanged"] != study.analysis.LIMITS or plan["reference_replaced"]):
        raise ValueError("Research scope changed")
    cpu.verify_files(plan["protected_files"]); cpu.verify_files(plan["authenticated_files"])
    cpu.verify_runtime()
    runtime = Path(plan["root"]) / "runtime"
    for name, digest in plan["private_source_sha256"].items():
        if base.sha(runtime / name) != digest:
            raise ValueError("Private source changed")
        if name == "cutcell_solver.f90":
            verify_transform((cpu.RUNTIME / name).read_bytes(), (runtime / name).read_bytes())
        elif base.sha(cpu.RUNTIME / name) != digest:
            raise ValueError("Unexpected changed native routine")


def compare_fields(left, right, data):
    comparison, li, ri = study.analysis.comparison(left, right, data["area"], float(data["dry_tolerance"]))
    if len(li) != 2 or len(ri) != 2 or np.max(abs(left["time"] - right["time"])) > 1.e-7:
        raise ValueError("Unmatched exact continuation outputs")
    comparison["focus"] = {str(c): dict(bed_difference_m=float(left["bed"][-1, c] - right["bed"][-1, c]),
        depth_difference_m=float(left["q"][-1, 0, c] - right["q"][-1, 0, c]),
        secondary_mass_difference=float(left["secondary"][-1, c] - right["secondary"][-1, c])) for c in (3113, 4911, 776, 1140)}
    return comparison


def execute(path, approved, hardware_released=False):
    path = Path(path)
    if not hardware_released or base.sha(path) != approved:
        raise ValueError("Reviewed plan and hardware release required")
    plan = json.loads(path.read_text()); validate(plan)
    root, report = Path(plan["root"]), Path(plan["report"])
    if (report / "execution.json").exists():
        raise FileExistsError("Preserve previous attempt")
    baseline = json.loads(BASELINE_PLAN.read_text())
    baseline_execution = json.loads((Path(baseline["report"]) / "execution.json").read_text())
    if baseline_execution["status"] != "completed" or baseline_execution["plan_sha256"] != BASELINE_SHA:
        raise ValueError("Finish the original four diagnostic cases first")
    result = dict(status="running", plan_sha256=approved, source_change=plan["only_source_change"],
        scope=plan["scope"], reference_replaced=False, performance_claim=False)
    base.save_json(report / "execution.json", result)
    try:
        begin = time.perf_counter()
        process = subprocess.run(plan["build_command"], cwd=root / "runtime", text=True, capture_output=True, timeout=300)
        (root / "runtime/build.log").write_text(process.stdout + process.stderr)
        process.check_returncode()
        result["build_wall_s"] = time.perf_counter() - begin
        executable_hash = base.sha(plan["executable"])
        build = dict(command=plan["build_command"], executable_sha256=executable_hash,
            source_sha256=plan["private_source_sha256"], parent_executable_sha256=cpu.EXE_SHA,
            only_source_change=plan["only_source_change"], all_other_source_bytes_identical=True)
        base.save_json(root / "runtime/build.json", build)
        inherited = {k:v for k,v in os.environ.items() if k.upper().startswith(("BRAIDS_", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))}
        env = {k:v for k,v in os.environ.items() if k not in inherited}
        env.update(study.adapter.NATIVE_ENV, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        result.update(executable_sha256=executable_hash, removed_inherited_environment=inherited,
            overrides={k:env[k] for k in env if k.upper().startswith(("BRAIDS_", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))})
        base.save_json(report / "execution.json", result)
        case = Path(plan["child"]["CPU_output"])
        with (case / "native.log").open("x") as stream:
            begin = time.perf_counter()
            process = subprocess.run(plan["native_command"], cwd=case, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=900, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        result.update(returncode=process.returncode, diagnostic_process_wall_s=time.perf_counter() - begin)
        process.check_returncode()
        actual, data, receipt = study.load_cpu(plan["child"])
        with np.load(root / "exact_initial.npz", allow_pickle=False) as expected:
            if any(not study.exact(actual[k][0], expected[k]) for k in study.FIELDS):
                raise ValueError("Initial physical field or cumulative ledger changed")
        budget = study.analysis.budgets(actual, data["area"], float(data["dry_tolerance"]))
        if not budget["passed"]:
            raise ValueError("Incremental conservation/nonnegativity failed")
        child = json.loads(Path(baseline["children"]["baseline"]["path"]).read_text())
        cpu_base, _, cpu_receipt = study.load_cpu(child)
        gpu_job = next(j for j in baseline["jobs"] if j["name"] == "gpu_baseline")
        gpu_base, _, gpu_receipt = study.load_gpu(child, gpu_job, BASELINE_SHA)
        comparisons = {"tightCPU_minus_defaultCPU": compare_fields(actual, cpu_base, data),
            "tightGPU_minus_tightCPU": compare_fields(gpu_base, actual, data),
            "tightGPU_minus_defaultCPU": compare_fields(gpu_base, cpu_base, data)}
        np.savez(report / "tight_CPU_frames.npz", **actual)
        validate(plan)
        if base.sha(plan["executable"]) != executable_hash:
            raise ValueError("Executed private runtime changed")
        result.update(status="completed", receipt=receipt, budgets=budget, comparisons=comparisons,
            initial_state_and_ledgers_bitwise=True, checkpoint_inputs_unchanged=True,
            baseline_CPU_receipt=cpu_receipt, baseline_GPU_receipt=gpu_receipt,
            baseline_execution_sha256=base.sha(Path(baseline["report"]) / "execution.json"),
            fields_sha256=base.sha(report / "tight_CPU_frames.npz"))
    except Exception as exc:
        result.update(status="failed", error=repr(exc)); raise
    finally:
        study.save(report / "execution.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--hardware-released", action="store_true")
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else execute(args.plan, args.approved_plan_sha256, args.hardware_released)
    print(json.dumps(study.analysis.serializable(result), indent=2, allow_nan=False))
