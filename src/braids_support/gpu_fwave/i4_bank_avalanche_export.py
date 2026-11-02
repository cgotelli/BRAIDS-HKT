"""Zero-step two-sidecar export for native mode-2 bank plus general avalanche.

Only initial-state I/O is added to a copied, hash-locked BRAIDS CPU source.
Its numerical routines and all Delft3D files remain untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from . import coupled_reference as cr
from . import cpu_filter_cold20h as parent
from . import i4_avalanche_export as avalanche_export
from . import i4_bank_avalanche_native_case as native
from . import i4_physical_toe_mode2_export as toe_export


HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs/i4_bank_avalanche_prepared_20261004_v2"
RUNTIME = ROOT / "runtime"
NATIVE_PLAN_SHA = "8a8920b64fee92f6cba912f905a558b3918de825b9b28fe4bda77df5753ad589"
NATIVE_ANALYSIS_SHA = "87680c796298fb6b948c062df3be7852e441c9e6f91e2f6003ad8fff8729ab20"
PRE_GPU_GATE_SHA = "e6d9d35621f6bb3ce5df70277f260ba69baccc35a999dabd8d8e3a173a1138b2"
PARENT_EXE_SHA = "21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5"
SIDECARS = ("bank_toe_metadata.bin", "avalanche_metadata.bin")


def sha(path) -> str:
    return native.base.sha(Path(path))


def require(value, message) -> None:
    native.base.require(value, message)


def _native_gate() -> dict:
    plan = native.DEFAULT_ROOT / "plan.json"
    analysis = native.DEFAULT_ROOT / "analysis.json"
    gate = HERE / "reports/i4_bank_avalanche_native_20261004_v1/GPU_ACCEPTANCE_GATE.txt"
    native.verify_plan(native.DEFAULT_ROOT)
    require(sha(plan) == NATIVE_PLAN_SHA and sha(analysis) == NATIVE_ANALYSIS_SHA and
            sha(gate) == PRE_GPU_GATE_SHA and
            sha(native.base.SOURCE_EXECUTABLE) == PARENT_EXE_SHA,
            "Frozen native four-way reference or pre-GPU gate differs")
    contents = json.loads(analysis.read_text())
    require(contents["passed"] is True and contents["GPU_solver_executed"] is False,
            "Native four-way prerequisite did not pass")
    return dict(plan_sha256=NATIVE_PLAN_SHA, analysis_sha256=NATIVE_ANALYSIS_SHA,
                gate_sha256=PRE_GPU_GATE_SHA)


def transform(source: str) -> str:
    changed = cr.instrument(source)
    anchor = "  call gpu_export_input()\n  stop "
    marker = "end module\n\nprogram run_cutcell"
    require(changed.count(anchor) == 1 and changed.count(marker) == 1,
            "Unexpected zero-step native exporter anchors")
    changed = changed.replace(anchor,
        "  call gpu_export_input()\n"
        "  call i4_export_physical_toe_metadata()\n"
        "  call i4_export_avalanche_metadata()\n  stop ", 1)
    extras = toe_export.extra_export() + avalanche_export.extra_export()
    return changed.replace(marker, extras + marker, 1)


def verify_transform(source: str, changed: str) -> None:
    extras = toe_export.extra_export() + avalanche_export.extra_export()
    restored = changed.replace(extras, "", 1).replace(
        "  call i4_export_physical_toe_metadata()\n", "", 1).replace(
        "  call i4_export_avalanche_metadata()\n", "", 1)
    require(changed == transform(source) and restored == cr.instrument(source),
            "Combined exporter changed more than zero-step I/O")
    for name in ("load_case", "build_physical_toe_map",
                 "process_noncohesive_bank_failure", "avalanche_erodible_banks",
                 "advance_ssp33"):
        start = source.index("subroutine " + name + "(")
        end = source.index("end subroutine", start)
        require(changed.count(source[start:end]) == 1,
                "Native numerical/parser body changed: " + name)


def build(runtime: Path = RUNTIME) -> dict:
    runtime = Path(runtime).resolve()
    require(not runtime.exists(), "Combined I/O-only runtime already exists")
    _native_gate()
    identity, protected = parent.runtime_identity()
    require(parent.TIGHT_SHA == PARENT_EXE_SHA,
            "Combined exporter parent must be frozen native runtime")
    runtime.mkdir(parents=True)
    sources = []
    for name in identity["source_sha256"]:
        original = parent.TIGHT_RUNTIME / name
        target = runtime / name
        if name == "cutcell_solver.f90":
            source = original.read_text(encoding="utf-8")
            changed = transform(source)
            verify_transform(source, changed)
            target.write_text(changed, encoding="utf-8")
        else:
            shutil.copy2(original, target)
        sources.append(str(target))
    for path in parent.TIGHT_RUNTIME.glob("*.dll"):
        shutil.copy2(path, runtime / path.name)
    flags = identity["command"][1:6]
    require(flags == ["-O3", "-flto", "-fopenmp", "-ffree-line-length-none",
                      "-ffixed-line-length-none"],
            "Frozen native compiler flags differ")
    compiler = Path(identity["command"][0])
    executable = runtime / "xbraids_bank_avalanche_export.exe"
    command = [str(compiler), *flags, *sources, "-o", str(executable)]
    process = subprocess.run(command, cwd=runtime, capture_output=True,
                             text=True, timeout=300)
    (runtime / "build.log").write_text(process.stdout + process.stderr,
                                       encoding="utf-8")
    require(process.returncode == 0,
            "Combined I/O-only export compilation failed: " + process.stderr[-2000:])
    record = dict(schema=1, command=command, executable=str(executable),
        executable_sha256=sha(executable), parent_executable_sha256=PARENT_EXE_SHA,
        parent_protected_files=protected,
        source_sha256={Path(path).name: sha(path) for path in sources},
        compiler_sha256=sha(compiler), driver_sha256=sha(__file__),
        coupled_reference_sha256=sha(cr.__file__),
        toe_sidecar_generator_sha256=sha(toe_export.__file__),
        avalanche_sidecar_generator_sha256=sha(avalanche_export.__file__),
        native_reference=_native_gate(),
        transformation="cr.instrument initial I/O plus two metadata sidecars; numerical body untouched",
        solver_steps_performed=0, numerical_source_unchanged=True,
        Delft3D_source_modified=False)
    native.base.save_json(runtime / "build.json", record)
    return verify_build(runtime)


def verify_build(runtime: Path = RUNTIME) -> dict:
    runtime = Path(runtime).resolve()
    record = json.loads((runtime / "build.json").read_text())
    parent.runtime_identity()
    require(record["schema"] == 1 and
            record["parent_executable_sha256"] == PARENT_EXE_SHA == parent.TIGHT_SHA and
            record["driver_sha256"] == sha(__file__) and
            record["coupled_reference_sha256"] == sha(cr.__file__) and
            record["toe_sidecar_generator_sha256"] == sha(toe_export.__file__) and
            record["avalanche_sidecar_generator_sha256"] == sha(avalanche_export.__file__) and
            record["compiler_sha256"] == sha(record["command"][0]) and
            record["executable_sha256"] == sha(record["executable"]),
            "Combined I/O-only build identity differs")
    for name, digest in record["source_sha256"].items():
        require(sha(runtime / name) == digest,
                "Combined I/O-only source changed: " + name)
    for path, digest in record["parent_protected_files"].items():
        require(sha(path) == digest,
                "Parent native source/runtime changed: " + path)
    source = (parent.TIGHT_RUNTIME / "cutcell_solver.f90").read_text(encoding="utf-8")
    changed = (runtime / "cutcell_solver.f90").read_text(encoding="utf-8")
    verify_transform(source, changed)
    return record


def read_prepared(folder: Path, case: str) -> dict:
    require(case in native.CASES, "Unknown combined native case")
    folder = Path(folder).resolve()
    require((folder / "coupled_input.bin").is_file() and
            all((folder / name).is_file() for name in SIDECARS) and
            not any((folder / name).exists() for name in
                    ("normal_metadata.bin", "distributed_bank_metadata.bin",
                     "coupled_diffusion.bin")),
            "Combined prepared sidecars absent or incompatible")
    data = cr.read_snapshot(folder)
    for sidecar, expected in ((SIDECARS[0], toe_export.EXPECTED_KEYS),
                              (SIDECARS[1], set(avalanche_export.CONTROLS) |
                               set(toe_export.IDENTITY_KEYS) | {"general_avalanche"})):
        extra = cr.read_records(folder / sidecar)
        require(set(extra) == set(expected),
                "Combined sidecar schema differs: " + sidecar)
        for name, value in extra.items():
            require(name not in data or np.array_equal(data[name], value),
                    "Duplicate native prepared value differs: " + name)
            data[name] = value
    bank_on, avalanche_on = native.flags(case)
    nc, nf = int(data["nc"]), int(data["nf"])
    require(nc == 18 and nf == 45 and int(data["nfrac"]) == 2 and
            int(data["nbed_layers"]) == 2 and int(data["scheme"]) == 6 and
            int(data["bedlaw"]) == 2 and int(data["exchange_mode"]) == 0 and
            int(data["mpm_friction_mode"]) == 1 and
            float(data["mpm_critical"]) == .047 and
            int(data["bank_failure"]) == 1 and
            int(data["bank_toe_erosion_mode"]) == 2 and
            int(data["toe_mapping_status"]) == 0 and
            int(data["general_avalanche"]) == int(avalanche_on) and
            int(data["bank_avalanche"]) == 0 and
            int(data["amr_enabled"]) == 0 and
            float(data["morphological_factor"]) == 1. and
            float(data["bank_toe_capacity_fraction"]) == float(bank_on),
            "Combined prepared numerical scope differs")
    expected_controls = (
        (("general_avalanche_wet_slope", native.WET_SLOPE),
         ("general_avalanche_dry_slope", native.DRY_SLOPE),
         ("general_avalanche_time", native.AVALANCHE_TIME_S),
         ("general_avalanche_wet_depth_switch", .1),
         ("general_avalanche_relaxation_length", native.RELAXATION_LENGTH_M))
        if avalanche_on else
        (("general_avalanche_wet_slope", 10.),
         ("general_avalanche_dry_slope", 1.),
         ("general_avalanche_time", 86400.),
         ("general_avalanche_wet_depth_switch", .1),
         ("general_avalanche_relaxation_length", 1.)))
    for name, value in expected_controls:
        require(float(data[name]) == value,
                "Combined prepared avalanche control differs: " + name)
    original = np.load(native.DEFAULT_ROOT / "inputs" / case /
                       "initial_analysis.npz", allow_pickle=False)
    require(np.array_equal(data["state"].T, original["state"]) and
            np.array_equal(data["xy"].T, original["xy"]) and
            np.array_equal(data["area"], original["area"]) and
            np.array_equal(data["rock"], original["rock"]) and
            np.array_equal(data["bulk_fraction"], original["bulk"]) and
            np.array_equal(data["bank_material_mask"], original["bank_mask"]) and
            np.array_equal(data["layer_mass"],
                           original["state"][:, 5:9].T.reshape(2, 2, nc)),
            "Combined prepared initial state/mask differs from native input")
    for key in (toe_export.CELL_FLAGS + toe_export.CELL_IDS +
                ("physical_toe_distance", "physical_toe_status")):
        require(np.shape(data[key]) == (nc,),
                "Combined cell map shape differs: " + key)
    for key in toe_export.FACE_FLAGS + toe_export.FACE_CELL_IDS:
        require(np.shape(data[key]) == (nf,),
                "Combined face map shape differs: " + key)
    mask = data["bank_material_mask"].astype(bool)
    toe_faces = data["is_toe_face"].astype(bool)
    require(np.any(mask) and np.any(toe_faces) and
            np.all(data["physical_toe_status"][mask] == 1) and
            np.all(data["physical_toe_status"][~mask] == 0) and
            np.all(np.isfinite(data["physical_toe_distance"])) and
            np.all(data["physical_toe_distance"][mask] >= 0),
            "Combined native bank material-to-toe map invalid")
    return data


def prepare(case: str, root: Path = ROOT) -> dict:
    require(case in native.CASES, "Unknown native combination case")
    reference = _native_gate()
    built = verify_build()
    root = Path(root).resolve()
    source = native.DEFAULT_ROOT / "inputs" / case / "case.data"
    destination = root / case
    require(not destination.exists(), "Combined prepared case already exists")
    origin = cr.copy_inputs(source, destination)
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("BRAIDS_", "OMP_"))}
    environment.update(BRAIDS_GPU_INITIAL_EXPORT="1",
                       BRAIDS_TIME_INTEGRATOR="legacy", OMP_NUM_THREADS="1",
                       OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
    environment["PATH"] = str(RUNTIME.resolve()) + os.pathsep + environment.get("PATH", "")
    process = subprocess.run([built["executable"], source.name], cwd=destination,
                             env=environment, capture_output=True, text=True,
                             timeout=60)
    (destination / "initial.log").write_text(process.stdout + process.stderr,
                                              encoding="utf-8")
    require(process.returncode == 0,
            "Combined zero-step native export failed: " + process.stderr[-2000:])
    require(not (destination / "frames.txt").exists() and
            not (destination / "checkpoint.bin").exists(),
            "Combined zero-step export unexpectedly evolved state")
    data = read_prepared(destination, case)
    artifacts = {name: sha(destination / name) for name in
                 ("coupled_input.bin", *SIDECARS)}
    bank_on, avalanche_on = native.flags(case)
    receipt = dict(schema=1, status="completed", case=case, origin=origin,
        native_reference=reference,
        parent_native_executable_sha256=PARENT_EXE_SHA,
        exporter_build_sha256=sha(RUNTIME / "build.json"),
        exporter_executable_sha256=built["executable_sha256"],
        preparation_driver_sha256=sha(__file__),
        coupled_reference_sha256=sha(cr.__file__),
        solver_steps_performed=0, initialization_CPU_only=True,
        controls=dict(bank_failure=1, bank_toe_erosion_mode=2,
                      bank_toe_capacity_fraction=int(bank_on),
                      general_avalanche=int(avalanche_on),
                      avalanche_wet_slope=float(data["general_avalanche_wet_slope"]),
                      avalanche_dry_slope=float(data["general_avalanche_dry_slope"]),
                      avalanche_time_s=float(data["general_avalanche_time"])),
        geometry=dict(cells=int(data["nc"]), faces=int(data["nf"]),
                      fractions=int(data["nfrac"]), layers=int(data["nbed_layers"]),
                      bank_material_cells=int(np.count_nonzero(data["bank_material_mask"])),
                      mapped_toe_faces=int(np.count_nonzero(data["is_toe_face"]))),
        artifacts=artifacts, numerical_source_unchanged=True,
        Delft3D_source_modified=False)
    native.base.save_json(destination / "preparation_receipt.json", receipt)
    return dict(folder=str(destination), receipt_sha256=sha(
        destination / "preparation_receipt.json"), artifacts=artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "prepare", "verify-build"))
    parser.add_argument("--case", choices=native.CASES)
    args = parser.parse_args()
    if args.action == "build":
        result = build()
    elif args.action == "verify-build":
        result = verify_build()
    else:
        require(args.case is not None, "--case required")
        result = prepare(args.case)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
