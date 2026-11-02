"""Prepare/run one additional cold20h CPU filter-convergence control.

Uses the existing authenticated21b7 executable; never compiles. The user has
conditionally approved this tightly converged CPU as the acceptance reference
after full-run verification. All a9/original comparisons and limits remain.
Reuses the original cold CPU workflow with an isolated, restored adapter context.
"""
from contextlib import contextmanager
import argparse
import inspect
import json
from pathlib import Path
import shutil
import numpy as np

from . import outlet_bounds_cpu20h as cpu
from . import cpu_filter_convergence_control as control
from . import outlet_bounds_long_report as report_tools
from . import long_bend_analysis as long
from . import candidate_analysis_review as review

base = cpu.base
TAG = "cpu_filter_cold20h_20261002_v1"
APPROVED_RUNTIME, APPROVED_SHA = cpu.RUNTIME, cpu.EXE_SHA
APPROVED_CASE = base.HERE / "runs/outlet_bounds_cpu20h_20261002_v1/control"
TIGHT_RUNTIME = base.HERE / "runs/cpu_filter_convergence_control_20261002_v1/runtime"
TIGHT_EXE = TIGHT_RUNTIME / "xbraids_filter_tight.exe"
TIGHT_SHA = "21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5"
GPU = base.HERE / "runs/outlet_bounds_tight_gpu20h_20261002_v1/gpu"
TIMES = np.arange(121, dtype=float) * 24.
USER_STATEMENT = "Use the converged CPU reference"
APPROVAL_PATH = base.HERE / "reports/scalable_gpu_qualification_20261002_v1/converged_reference_approval.json"


def reference_approval():
    """Read root's actual user-answer record; do not invent a second authority."""
    record=long.read_json(APPROVAL_PATH)
    if (record.get("user_approved") is not True or record.get("user_statement")!=USER_STATEMENT
            or record.get("converged_CPU_executable_sha256")!=TIGHT_SHA
            or record.get("parent_repaired_CPU_executable_sha256")!=APPROVED_SHA
            or record.get("maximum_bed_difference_limit_m")!=.005
            or record.get("area_weighted_bed_RMS_limit_m")!=.003
            or record.get("full_cold_20h_verification_required") is not True
            or record.get("all_previous_comparisons_retained") is not True):
        raise ValueError("Canonical conditional approval scope differs")
    return record


def runtime_identity():
    parent = long.read_json(APPROVED_RUNTIME / "build.json")
    tight = long.read_json(TIGHT_RUNTIME / "build.json")
    if (parent["executable_sha256"] != APPROVED_SHA or base.sha(APPROVED_RUNTIME / "xbraids_candidate.exe") != APPROVED_SHA
            or tight["parent_executable_sha256"] != APPROVED_SHA or tight["executable_sha256"] != TIGHT_SHA
            or base.sha(TIGHT_EXE) != TIGHT_SHA or set(parent["source_sha256"]) != set(tight["source_sha256"])):
        raise ValueError("Authenticated corrected/tight runtimes differ")
    files = {str(TIGHT_EXE): TIGHT_SHA, str(APPROVED_RUNTIME / "xbraids_candidate.exe"): APPROVED_SHA}
    for folder in (APPROVED_RUNTIME, TIGHT_RUNTIME):
        files[str(folder / "build.json")] = base.sha(folder / "build.json")
    for name, digest in tight["source_sha256"].items():
        original, candidate = APPROVED_RUNTIME / name, TIGHT_RUNTIME / name
        if base.sha(candidate) != digest or base.sha(original) != parent["source_sha256"][name]:
            raise ValueError("Native source identity changed")
        if name == "cutcell_solver.f90":
            control.verify_transform(original.read_bytes(), candidate.read_bytes())
        elif digest != parent["source_sha256"][name]:
            raise ValueError("Unexpected source change")
        files[str(original)] = parent["source_sha256"][name]; files[str(candidate)] = digest
    return tight, files


@contextmanager
def adapter_context():
    """Process-local injection; original module and frozen files are untouched."""
    names = ("TAG", "RUNTIME", "EXE_SHA", "verify_runtime", "analyze")
    saved = {name: getattr(cpu, name) for name in names}
    cpu.TAG, cpu.RUNTIME, cpu.EXE_SHA = TAG, TIGHT_RUNTIME, TIGHT_SHA
    cpu.verify_runtime, cpu.analyze = runtime_identity, analyze
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(cpu, name, value)


def analysis_dependencies():
    result = report_tools.analysis_dependencies()
    for module in (cpu, control, report_tools, long, review):
        result[str(Path(module.__file__).resolve())] = base.sha(module.__file__)
    result[str(Path(__file__).resolve())] = base.sha(__file__)
    return result


def prepare():
    runtime_identity()
    with adapter_context():
        prepared = cpu.prepare()
    path = Path(prepared["plan"]); plan = long.read_json(path)
    # Original helper uses its own executable basename; select the already
    # qualified existing tight executable explicitly before the plan is frozen.
    plan.update(executable=str(TIGHT_EXE), previous_corrected_reference_case=str(APPROVED_CASE),
        previous_corrected_reference_executable_sha256=APPROVED_SHA, additional_convergence_reference=True,
        user_reference_approval=reference_approval(), canonical_approval_path=str(APPROVAL_PATH),
        reference_verified=False, prepared_plan_path=str(path),
        numerical_change="Only curvature filter rtol1e-4 to1e-8 relative to prior correctedCPU; atol1e-10 unchanged",
        reference_policy="User-approved tightCPU21b7 becomes acceptance reference ONLY after complete121-frame runtime/input/initial-state/conservation/nonnegativity verification; retain a9/original comparisons",
        comparison_policy="All121outputs against prior correctedCPU, originalCPU and existing tightGPU; fixed5mm/3mm diagnostics, no interpolation",
        timing_policy="Record actual diagnostic wall time; no speed qualification or automatic promotion",
        estimated_wall_minutes=[80,100], compile_required=False, filter_controls=dict(rtol=1e-8,atol=1e-10,maxiter=500),
        existing_GPU=str(GPU), full_evolution_controls=dict(begin_s=0.,end_s=2880.,output_s=24.,MF=25.,CFL=.8,dtmax=.0125,threads=8,integrator=43),
        expected_frames=121, accuracy_limits=dict(long.LIMITS))
    protected = plan["protected_files"]
    protected[str(APPROVAL_PATH)]=base.sha(APPROVAL_PATH)
    for folder in (APPROVED_CASE,):
        for name in ("case.data", "case.data.bin", "case.data.json", "input_provenance.json", "analysis_geometry.npz",
                     "frames.bin", "checkpoint.bin", "secondary_flow.txt", "secondary_flow_budget.txt", "execution.json"):
            protected[str(folder / name)] = base.sha(folder / name)
    manifest = long.read_json(GPU / "run.json"); report_tools.check_gpu_controls(manifest)
    protected[str(GPU / "run.json")] = base.sha(GPU / "run.json")
    for record in manifest["frames"]:
        protected[str(GPU / record["file"])] = record["sha256"]
    for name, digest in manifest["source_sha256"].items():
        protected[str(GPU / "source" / name)] = digest
    audit_folder = GPU.with_name("gpu_audit"); audit = long.read_json(audit_folder / "audit.json")
    protected[str(audit_folder / "audit.json")] = base.sha(audit_folder / "audit.json")
    protected[str(Path(audit["plan"]))] = audit["plan_sha256"]
    for name, digest in audit["artifacts"].items():
        protected[str(audit_folder / name)] = digest
    plan["authenticated_files"].update(analysis_dependencies())
    shutil.copy2(__file__, Path(plan["report"]) / Path(__file__).name)
    plan["authenticated_files"][str(Path(plan["report"]) / Path(__file__).name)] = base.sha(__file__)
    validate_plan(plan); base.save_json(path, plan)
    base.save_json(path.with_name("reference_approval.json"), dict(canonical_record_path=str(APPROVAL_PATH),
        canonical_record_sha256=base.sha(APPROVAL_PATH), prepared_plan_sha256=base.sha(path),
        target_executable_sha256=TIGHT_SHA, conditional_on_full_run_verification=True))
    return dict(plan=str(path), sha256=base.sha(path), launched=False, compile_required=False, expected_minutes=[80,100])


def validate_plan(plan):
    if (plan["tag"] != TAG or plan["executable"] != str(TIGHT_EXE) or plan["executable_sha256"] != TIGHT_SHA
            or plan["filter_controls"] != dict(rtol=1e-8,atol=1e-10,maxiter=500)
            or plan["full_evolution_controls"] != dict(begin_s=0.,end_s=2880.,output_s=24.,MF=25.,CFL=.8,dtmax=.0125,threads=8,integrator=43)
            or plan["accuracy_limits"] != long.LIMITS or plan["expected_frames"] != 121
            or plan["user_reference_approval"] != reference_approval() or plan["reference_verified"] or plan["compile_required"]):
        raise ValueError("Predeclared full convergence scope changed")
    runtime_identity(); cpu.verify_files(plan["authenticated_files"]); cpu.verify_files(plan["protected_files"])


def verify_approval(path):
    path=Path(path)
    record=reference_approval()
    expected=dict(canonical_record_path=str(APPROVAL_PATH), canonical_record_sha256=base.sha(APPROVAL_PATH),
        prepared_plan_sha256=base.sha(path), target_executable_sha256=TIGHT_SHA, conditional_on_full_run_verification=True)
    if long.read_json(path.with_name("reference_approval.json")) != expected:
        raise ValueError("Conditional user approval is not bound to this exact plan/runtime")
    return dict(binding=expected, canonical_record=record)


def check_initial_and_grid(fields):
    for name, value in fields.items():
        if np.shape(value["time"]) != (121,) or not np.allclose(value["time"], TIMES, rtol=0, atol=1e-7):
            raise ValueError("Require all121original output times: " + name)
        for key in ("q", "layers", "bed", "secondary"):
            a, b = value[key][0], fields["corrected_cpu"][key][0]
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                raise ValueError("Nonidentical cold initial field: " + name + "/" + key)
        for key in ("ledger_water", "ledger_bed", "ledger_secondary", "ledger_source"):
            if np.any(value[key][0] != 0.):
                raise ValueError("Cold initial ledger is not zero")


def analyze(plan):
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    validate_plan(plan); approval=verify_approval(plan["prepared_plan_path"])
    before = analysis_dependencies(); report = Path(plan["report"])
    candidate_folder=Path(plan["case"]); executed=long.read_json(candidate_folder/"execution.json")
    if (executed.get("status")!="completed" or executed.get("return_code")!=0
            or executed.get("executable_sha256")!=TIGHT_SHA
            or executed.get("plan_sha256")!=base.sha(plan["prepared_plan_path"])
            or executed.get("source_sha256")!=plan["runtime_identity"]["source_sha256"]
            or executed.get("threads")!=8 or executed.get("environment")!=plan["environment"]):
        raise ValueError("Completed tightCPU execution identity differs from reviewed plan")
    outputs={str(candidate_folder/name):base.sha(candidate_folder/name) for name in
        ("frames.bin","checkpoint.bin","secondary_flow.txt","secondary_flow_budget.txt","execution.json","native.log")}
    gpu, manifest, data, _ = long.load_worker(GPU)
    audit = long.read_json(GPU.with_name("gpu_audit") / "audit.json")
    gpu_plan = long.read_json(audit["plan"])
    trace = report_tools.trace_auditor(gpu_plan)(GPU, require_complete_20h=True)
    report_tools.check_gpu_controls(manifest)
    fields = dict(tight_cpu=cpu.read_history(candidate_folder), corrected_cpu=cpu.read_history(APPROVED_CASE),
                  original_cpu=cpu.read_history(base.SOURCE), gpu=gpu)
    check_initial_and_grid(fields)
    live = {name:getattr(previous.r,name) for name in review.CONSTANTS}
    constants = review.check_constants(data, {k:v for k,v in fields.items() if k != "gpu"}, live)
    area, dry = data["area"], float(data["dry_tolerance"])
    pairs = dict(tightCPU_vs_correctedCPU=("tight_cpu","corrected_cpu"), tightCPU_vs_originalCPU=("tight_cpu","original_cpu"),
        GPU_vs_tightCPU=("gpu","tight_cpu"), GPU_vs_correctedCPU=("gpu","corrected_cpu"), GPU_vs_originalCPU=("gpu","original_cpu"))
    comparisons = {key:long.comparison(fields[left],fields[right],area,dry)[0] for key,(left,right) in pairs.items()}
    for name,(left,right) in pairs.items():
        value = comparisons[name]; location = value["worst_bed_location"]
        location["xy_m"] = data["xy"][:,location["cell"]]
        value["inlet_corner_curves_m"] = {str(c):fields[left]["bed"][:,c]-fields[right]["bed"][:,c] for c in (3113,4911)}
    budgets = {name:long.budgets(value,area,dry) for name,value in fields.items()}
    result = dict(all121outputs_aligned=True, initial_fields_bitwise_equal=True, comparisons=comparisons,
        conservation=budgets, CPU_decoding_checks=constants, GPU_trace=trace,
        morphology={name:long.morphology(value,fields["corrected_cpu"]["geometry"]) for name,value in fields.items()},
        user_reference_approval=approval, reference_verification_passed=bool(budgets["tight_cpu"]["passed"]),
        selected_acceptance_reference="tightCPU21b7" if budgets["tight_cpu"]["passed"] else None,
        previous_corrected_reference="correctedCPU a9; retained diagnostic comparison",
        acceptance_gate_relaxed=False, accuracy_limits=dict(long.LIMITS),
        GPU_accuracy_passed_against_selected_reference=bool(budgets["tight_cpu"]["passed"] and budgets["gpu"]["passed"]
            and comparisons["GPU_vs_tightCPU"]["passed"] and trace["verified"] and trace["complete_20h"]),
        performance_qualified=False, production_adopted=False, interpretation="Conditional user-approved converged CPU reference after full-run verification; all previous diagnostics retained, no automatic production promotion",
        candidate_identity=fields["tight_cpu"]["identity"], previous_corrected_reference_identity=fields["corrected_cpu"]["identity"],
        timing_note=plan["timing_policy"], analysis_dependency_sha256=before,
        verified_native_output_sha256=outputs, exact_executed_reference_sha256=TIGHT_SHA)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hours=TIMES*25/3600; fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    chosen=("tightCPU_vs_correctedCPU","GPU_vs_tightCPU","GPU_vs_correctedCPU")
    for name in chosen:
        value=comparisons[name]; curves=value["curves"]
        axes[0,0].plot(hours,curves["bed_maximum_m"]*1000,label=name)
        axes[0,1].plot(hours,curves["bed_rms_m"]*1000,label=name)
        for ax,c in zip(axes[1],(3113,4911)):
            ax.plot(hours,value["inlet_corner_curves_m"][str(c)]*1000,label=name)
    axes[0,0].axhline(5,color="black",ls="--",label="Unchanged5mm diagnostic")
    axes[0,1].axhline(3,color="black",ls="--",label="Unchanged3mm diagnostic")
    for ax,title in zip(axes.flat,("Maximum bed difference","Area-RMS bed difference","Signed inlet3113 difference","Signed inlet4911 difference")):
        ax.set(title=title,xlabel="Morphological hours",ylabel="mm");ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Full-cold filter convergence; tightly convergedCPU reference conditionally approved\nCPU tight andGPU rtol1e-8; prior correctedCPU rtol1e-4; all atol1e-10")
    figure=report/"FILTER_CONVERGENCE.png";fig.savefig(figure,dpi=160);plt.close(fig)
    result["figures"]=[dict(file=figure.name,sha256=base.sha(figure))]
    np.savez(report/"tight_CPU_frames.npz",**{k:fields["tight_cpu"][k] for k in ("time","q","layers","bed","secondary","ledger_water","ledger_bed","ledger_secondary","ledger_source")})
    result["fields_sha256"]=base.sha(report/"tight_CPU_frames.npz")
    cpu.verify_files(before);cpu.verify_files(outputs);validate_plan(plan)
    base.save_json(report/"comparison.json",long.serializable(result))
    lines=["FULL COLD20h CPU FILTER CONVERGENCE; CONDITIONAL USER-APPROVED REFERENCE",result["interpretation"],
           f"User statement: {USER_STATEMENT}. Full reference verification passed: {result['reference_verification_passed']}.",
           "All121frames0:24:2880; no interpolation;5mm max and3mm area-RMS diagnostics unchanged."]
    for name,value in comparisons.items():
        lines.append(f"{name}: max {1000*value['maximum_bed_difference_m']:.9g}mm; areaRMS {1000*value['maximum_bed_rms_m']:.9g}mm; diagnostic gates {value['passed']}.")
    lines += [f"All four models' own budgets/nonnegativity pass: {all(v['passed'] for v in budgets.values())}.",
              f"Selected acceptance reference: {result['selected_acceptance_reference']}; GPU accuracy against it passed: {result['GPU_accuracy_passed_against_selected_reference']}.",
              "Previous corrected/originalCPU comparisons retained. No physical validation, performance qualification or production adoption."]
    (report/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return long.serializable(result)


def execute(path, approved):
    path=Path(path)
    if base.sha(path)!=approved:
        raise ValueError("Explicitly approved full convergence plan SHA required")
    validate_plan(long.read_json(path));verify_approval(path)
    with adapter_context():
        return cpu.execute(path,approved)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=("prepare","execute"))
    parser.add_argument("--plan",type=Path);parser.add_argument("--approved-plan-sha256")
    args=parser.parse_args()
    value=prepare() if args.action=="prepare" else execute(args.plan,args.approved_plan_sha256)
    print(json.dumps(value if args.action=="prepare" else dict(status="completed",report=str(Path(args.plan).parent)),indent=2))
