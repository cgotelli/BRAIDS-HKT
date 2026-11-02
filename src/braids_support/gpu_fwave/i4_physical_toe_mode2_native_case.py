"""Frozen-candidate unchanged-native mode-2 physical bank-toe ON/OFF case.

The numerical executable is a copied, hash-locked CPU reference. This module
only prepares validation inputs, runs that executable, and audits its outputs.
It never rebuilds or modifies native or Delft3D numerical source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from braids_core.boundaries.cut_cell_io import CutSedimentFraction, write_cut_case
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid
from braids_qgis.core.results import read_frames


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs/i4_physical_toe_mode2_native_20261004_v2"
SOURCE_RUNTIME = HERE / "runs/cpu_filter_convergence_control_20261002_v1/runtime"
SOURCE_EXECUTABLE = SOURCE_RUNTIME / "xbraids_filter_tight.exe"
CASES = ("erosion", "control")
END_S = .5
PREFIX_S = .2
OUTPUT_S = .1
DT_MAX_S = .002
DRY_TOLERANCE_M = .00025
ACTIVE_M = .00001
CRITICAL_UNDERCUT_M = .0012
FRACTIONS = (
    CutSedimentFraction(fraction_id="fine", d50_m=.0002, d90_m=.0003,
                        settling_velocity_m_s=0., initial_volume_fraction=.5,
                        initial_active_volume_fraction=.5),
    CutSedimentFraction(fraction_id="coarse", d50_m=.001, d90_m=.0015,
                        settling_velocity_m_s=0., initial_volume_fraction=.5,
                        initial_active_volume_fraction=.5),
)
BULK = np.array([fraction.bulk_kg_m3 for fraction in FRACTIONS])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def files(folder: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in sorted(folder.iterdir())
            if path.is_file() and path.name != "execution.json"}


def geometry_and_initial() -> tuple:
    domain = PolygonDomain("physical_toe_mode2_reference",
                           [(0, 0), (6, 0), (6, 3), (0, 3)], "EPSG:2056")
    mesh = map_cut_cells(domain, [], NativeGrid(6, 3, 0, 0, 1, 1))
    xy = np.asarray([cell.centroid_xy for cell in mesh.cells], dtype=float)
    area = np.asarray([cell.area_m2 for cell in mesh.cells], dtype=float)
    require(len(mesh.cells) == 18 and np.all(area == 1.),
            "Expected 18 aligned one-square-metre cells")
    bank_mask = xy[:, 1] > 1.
    state = np.zeros((len(mesh.cells), 9), dtype=float)
    for cell, (x, y) in enumerate(xy):
        column, row = int(np.floor(x)), int(np.floor(y))
        thickness = (.16, .38, .78)[row] + .001 * (5 - column)
        fine_active = .58 + .015 * column
        fine_substrate = .24 + .02 * column
        state[cell, 5:7] = [ACTIVE_M * fine_active * BULK[0],
                            ACTIVE_M * (1. - fine_active) * BULK[1]]
        state[cell, 7:9] = [(thickness - ACTIVE_M) * fine_substrate * BULK[0],
                            (thickness - ACTIVE_M) * (1. - fine_substrate) * BULK[1]]
        if row == 0:
            state[cell, 0] = .45
            state[cell, 1] = state[cell, 0] * 2. * (.2 + .16 * column)
        elif row == 1:
            state[cell, 0] = .10
            state[cell, 1] = state[cell, 0] * 2. * .12
    rock = np.zeros(len(mesh.cells))
    require(np.count_nonzero(bank_mask) == 12 and
            np.count_nonzero(state[:, 0] <= DRY_TOLERANCE_M) == 6 and
            np.all(state[6:12, 0] > DRY_TOLERANCE_M),
            "Fixture must contain physical submerged donors and dry bank columns")
    return mesh, xy, area, state, rock, bank_mask


def write_input(folder: Path, mesh, xy, area, state, rock, bank_mask,
                *, case: str, duration: float) -> None:
    folder.mkdir(parents=True)
    capacity = 1. if case == "erosion" else 0.
    metadata = write_cut_case(
        folder / "case.data", mesh, state, rock, {},
        end_time_s=duration, dt_max_s=DT_MAX_S, output_interval_s=OUTPUT_S,
        cfl=.4, scheme=6, bedload=2, exchange=0, manning=.03,
        sediment_fractions=FRACTIONS, active_layer_thickness_m=ACTIVE_M,
        dry_tolerance_m=DRY_TOLERANCE_M,
        bank_erodible_by_cell=bank_mask, bank_failure=True,
        bank_toe_erosion_mode="noncohesive_transport_capacity",
        bank_toe_capacity_fraction=capacity,
        bank_critical_undercut_m=CRITICAL_UNDERCUT_M,
        source_sha256={"i4_native_case_script": sha(Path(__file__))},
    )
    require(metadata["format"] == "xbraids_cutcell-v38" and
            metadata["bank_failure"]["toe_erosion_mode"] ==
            "noncohesive_transport_capacity", "Writer did not encode mode 2")
    np.savez_compressed(folder / "initial_analysis.npz", xy=xy, area=area,
                        state=state, rock=rock, bulk=BULK, bank_mask=bank_mask)


def prepare(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    require(not output.exists(), "Native mode-2 reference already exists")
    require(SOURCE_EXECUTABLE.is_file() and
            (SOURCE_RUNTIME / "build.json").is_file(),
            "Frozen native runtime or manifest missing")
    build = json.loads((SOURCE_RUNTIME / "build.json").read_text())
    require(build["executable_sha256"] == sha(SOURCE_EXECUTABLE),
            "Frozen native runtime differs from its build manifest")
    for name, digest in build["source_sha256"].items():
        require(sha(SOURCE_RUNTIME / name) == digest,
                "Frozen native source snapshot changed: " + name)
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    executable = runtime / "xbraids_mode2_reference.exe"
    shutil.copy2(SOURCE_EXECUTABLE, executable)
    mesh, xy, area, state, rock, bank_mask = geometry_and_initial()
    for case in CASES:
        write_input(output / "inputs" / case, mesh, xy, area, state, rock,
                    bank_mask, case=case, duration=END_S)
        write_input(output / "inputs_prefix" / case, mesh, xy, area, state,
                    rock, bank_mask, case=case, duration=PREFIX_S)
    plan = dict(
        schema=1, tag=output.name,
        copied_native_executable=str(executable),
        copied_native_executable_sha256=sha(executable),
        parent_native_executable=str(SOURCE_EXECUTABLE.resolve()),
        parent_native_executable_sha256=sha(SOURCE_EXECUTABLE),
        parent_build_manifest=str((SOURCE_RUNTIME / "build.json").resolve()),
        parent_build_manifest_sha256=sha(SOURCE_RUNTIME / "build.json"),
        source_sha256={"case_script": sha(Path(__file__)),
                       "writer": sha(Path(write_cut_case.__code__.co_filename))},
        settings=dict(duration_s=END_S, prefix_s=PREFIX_S, output_s=OUTPUT_S,
                      dtmax_s=DT_MAX_S, cfl=.4, scheme=6, bedlaw=2,
                      exchange=0, MorFac=1., friction_mode=1,
                      toe_capacity_fraction={"erosion": 1., "control": 0.},
                      critical_undercut_m=CRITICAL_UNDERCUT_M,
                      active_layer_thickness_m=ACTIVE_M,
                      dry_tolerance_m=DRY_TOLERANCE_M,
                      bank_material_cells_zero_based=[int(value) for value in
                                                      np.flatnonzero(bank_mask)]),
        inputs={case: files(output / "inputs" / case) for case in CASES},
        prefix_inputs={case: files(output / "inputs_prefix" / case)
                       for case in CASES},
        Delft3D_source_modified=False, GPU_solver_executed=False,
    )
    save_json(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"),
            "plan_sha256": sha(output / "plan.json"),
            "native_executable_sha256": sha(executable)}


def verify_plan(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    require(plan["schema"] == 1 and plan["tag"] == output.name and
            plan["source_sha256"]["case_script"] == sha(Path(__file__)) and
            plan["source_sha256"]["writer"] ==
            sha(Path(write_cut_case.__code__.co_filename)),
            "Native mode-2 plan/source identity differs")
    require(plan["copied_native_executable_sha256"] ==
            sha(Path(plan["copied_native_executable"])) and
            plan["parent_build_manifest_sha256"] ==
            sha(Path(plan["parent_build_manifest"])),
            "Native mode-2 runtime or parent build identity differs")
    for case in CASES:
        require(files(output / "inputs" / case) == plan["inputs"][case] and
                files(output / "inputs_prefix" / case) ==
                plan["prefix_inputs"][case],
                "Native mode-2 input changed: " + case)
    return plan


def run(case: str, phase: str, output: Path = DEFAULT_ROOT) -> dict:
    require(case in CASES and phase in ("full", "prefix", "resume"),
            "Unknown native case or phase")
    output = Path(output).resolve()
    plan = verify_plan(output)
    folder = output / case / phase
    require(not folder.exists(), "Native mode-2 execution already exists")
    if phase == "resume":
        prefix = output / case / "prefix" / "checkpoint.bin"
        require(prefix.is_file(), "Native prefix checkpoint missing")
        source = output / "inputs" / case
    else:
        prefix = None
        source = output / ("inputs_prefix" if phase == "prefix" else "inputs") / case
    folder.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, folder / item.name)
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(("BRAIDS_", "OMP_"))}
    environment.update(BRAIDS_TIME_INTEGRATOR="legacy", OMP_NUM_THREADS="1",
                       OMP_DYNAMIC="FALSE", OMP_WAIT_POLICY="PASSIVE")
    command = [plan["copied_native_executable"], "case.data"]
    if prefix is not None:
        command.append(str(prefix))
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                            timeout=120, env=environment)
    (folder / "stdout.txt").write_text(result.stdout + result.stderr,
                                        encoding="utf-8")
    receipt = dict(schema=1, case=case, phase=phase,
                   status="completed" if result.returncode == 0 else "failed",
                   return_code=result.returncode, command=command,
                   restart_checkpoint_sha256=sha(prefix) if prefix else None,
                   plan_sha256=sha(output / "plan.json"),
                   native_executable_sha256=plan["copied_native_executable_sha256"],
                   environment={key: environment[key] for key in
                                ("BRAIDS_TIME_INTEGRATOR", "OMP_NUM_THREADS",
                                 "OMP_DYNAMIC", "OMP_WAIT_POLICY")},
                   output_files=files(folder))
    save_json(folder / "execution.json", receipt)
    require(result.returncode == 0,
            "Native mode-2 run failed: " + (result.stdout + result.stderr)[-1800:])
    return receipt


def checked_execution(output: Path, case: str, phase: str, plan: dict) -> Path:
    folder = output / case / phase
    receipt = json.loads((folder / "execution.json").read_text())
    require(receipt["return_code"] == 0 and
            receipt["plan_sha256"] == sha(output / "plan.json") and
            receipt["native_executable_sha256"] ==
            plan["copied_native_executable_sha256"],
            "Native mode-2 execution identity differs")
    for name, digest in receipt["output_files"].items():
        require(sha(folder / name) == digest,
                "Native mode-2 output changed: " + str(folder / name))
    return folder


def analyze(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    plan = verify_plan(output)
    data = {}
    for case in CASES:
        folder = checked_execution(output, case, "full", plan)
        rows = read_frames(folder / "frames.txt")
        state = np.stack([row["q"] for row in rows])
        times = np.asarray([row["time_s"] for row in rows])
        initial = np.load(output / "inputs" / case / "initial_analysis.npz",
                          allow_pickle=False)
        bank = np.loadtxt(folder / "bank_failure.txt", comments="#", ndmin=2)
        flux = np.loadtxt(folder / "physical_bank_toe_bedload_flux.txt",
                          comments="#", ndmin=2)
        data[case] = dict(folder=folder, rows=rows, state=state, times=times,
                          initial=initial, bank=bank, flux=flux)
        for phase in ("prefix", "resume"):
            checked_execution(output, case, phase, plan)
    on, off = data["erosion"], data["control"]
    q, control = on["state"], off["state"]
    area = on["initial"]["area"]
    bulk = on["initial"]["bulk"]
    bed = lambda s: np.sum((s[:, :, 5:7] + s[:, :, 7:9]) /
                           bulk[None, None, :], axis=2)
    bed_difference = bed(q) - bed(control)
    mass = np.sum((q[:, :, 5:7] + q[:, :, 7:9]) * area[None, :, None], axis=1)
    water = np.sum(q[:, :, 0] * area[None, :], axis=1)
    bank = on["bank"]
    bank_final = bank[np.isclose(bank[:, 0], END_S, atol=1e-12)]
    control_final = off["bank"][np.isclose(off["bank"][:, 0], END_S, atol=1e-12)]
    collapse_by_frame = {}
    for time in on["times"]:
        segment = bank[np.isclose(bank[:, 0], time, atol=1e-12)]
        collapse_by_frame[f"{time:.1f}"] = {
            str(cell): int(segment[segment[:, 1] == cell, 10][0])
            for cell in (8, 9)
        }
    expected = {"0.0": {"8": 0, "9": 0},
                "0.1": {"8": 0, "9": 0},
                "0.2": {"8": 0, "9": 1},
                "0.3": {"8": 1, "9": 2},
                "0.4": {"8": 1, "9": 2},
                "0.5": {"8": 1, "9": 2}}
    substrate_diff = q[-1, :, 7:9] - control[-1, :, 7:9]
    on_flux_final = on["flux"][np.isclose(on["flux"][:, 0], END_S, atol=1e-12)]
    control_flux_final = off["flux"][np.isclose(off["flux"][:, 0], END_S,
                                             atol=1e-12)]
    checks = dict(
        matched_six_frames=q.shape == (6, 18, 9) and control.shape == q.shape and
            np.allclose(on["times"], np.arange(6) * OUTPUT_S, atol=1e-12, rtol=0) and
            np.array_equal(on["times"], off["times"]),
        same_initial_state=np.array_equal(q[0], control[0]) and
            np.array_equal(on["initial"]["state"], off["initial"]["state"]),
        finite_nonnegative=np.isfinite(q).all() and np.isfinite(control).all() and
            np.min(q[:, :, 0]) >= 0 and np.min(q[:, :, 5:9]) >= 0,
        mixed_wet_dry=np.all(np.count_nonzero(q[:, :, 0] <= DRY_TOLERANCE_M,
                                            axis=1) == 6) and
            np.all(q[:, :12, 0] > DRY_TOLERANCE_M),
        class_conservation=np.max(np.abs(mass - mass[0])) < 1e-8,
        water_conservation=np.max(np.abs(water - water[0])) < 1e-8,
        toe_erodes_and_routes=bank_final[:, 8].max() > .0007 and
            bank_final[:, 9].max() > .0009 and
            np.all(control_final[:, 8:11] == 0),
        robust_collapse_sequence=collapse_by_frame == expected,
        both_grains_from_donor_substrate=np.all(substrate_diff[[8, 9], :] < -.1),
        both_grains_from_dry_bank_substrate=np.all(substrate_diff[[14, 15], :] < -.1),
        wet_toe_gains=np.max(bed_difference[-1, :6]) > .0015 and
            np.all(substrate_diff[3, :] > .5),
        dry_bank_loses=np.min(bed_difference[-1, 12:]) < -.0009,
        nonzero_physical_toe_flux=np.max(np.abs(on_flux_final[:, 10])) > .1 and
            np.all(on_flux_final[:, 11] + 1e-14 >=
                   np.abs(on_flux_final[:, 10])) and
            np.max(np.abs(control_flux_final[:, 10])) > .1,
    )
    # Verify that native prefix/resume reproduces each full final state and
    # accepted history. Text outputs are compared numerically and checkpoints
    # bytewise, independent of transient log formatting.
    restart_details = {}
    for case in CASES:
        full = data[case]["folder"]
        prefix = output / case / "prefix"
        resume = output / case / "resume"
        full_frames = read_frames(full / "frames.txt")
        prefix_frames = read_frames(prefix / "frames.txt")
        resume_frames = read_frames(resume / "frames.txt")
        full_final = full_frames[-1]
        resume_final = resume_frames[-1]
        matching_final = (full_final["time_s"] == resume_final["time_s"] and
                          np.array_equal(full_final["q"], resume_final["q"]))
        matching_checkpoint = (full / "checkpoint.bin").read_bytes() == \
            (resume / "checkpoint.bin").read_bytes()
        full_bank = np.loadtxt(full / "bank_failure.txt", comments="#", ndmin=2)
        resume_bank = np.loadtxt(resume / "bank_failure.txt", comments="#", ndmin=2)
        full_flux = np.loadtxt(full / "physical_bank_toe_bedload_flux.txt",
                               comments="#", ndmin=2)
        resume_flux = np.loadtxt(resume / "physical_bank_toe_bedload_flux.txt",
                                 comments="#", ndmin=2)
        matching_bank = np.array_equal(full_bank[np.isclose(full_bank[:, 0], END_S)],
                                       resume_bank[np.isclose(resume_bank[:, 0], END_S)])
        matching_flux = np.array_equal(full_flux[np.isclose(full_flux[:, 0], END_S)],
                                       resume_flux[np.isclose(resume_flux[:, 0], END_S)])
        prefix_valid = (prefix_frames[-1]["time_s"] == PREFIX_S and
                        np.array_equal(prefix_frames[-1]["q"], full_frames[2]["q"]))
        restart_details[case] = dict(prefix_state_exact=prefix_valid,
                                     final_state_exact=matching_final,
                                     final_bank_exact=matching_bank,
                                     final_toe_flux_exact=matching_flux,
                                     checkpoint_bytes_exact=matching_checkpoint,
                                     resumed_frames=len(resume_frames))
        checks[f"{case}_native_restart_exact"] = all(value for key, value in
            restart_details[case].items() if key != "resumed_frames")
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(
        schema=1, passed=all(checks.values()), checks=checks,
        plan_sha256=sha(output / "plan.json"),
        native_executable_sha256=plan["copied_native_executable_sha256"],
        frame_times_s=on["times"].tolist(),
        accepted_steps_by_frame=[int(row["steps"]) for row in on["rows"]],
        collapse_count_by_frame=collapse_by_frame,
        maximum_on_off_bed_difference_by_frame_m=np.max(
            np.abs(bed_difference), axis=1).tolist(),
        final_wet_toe_gain_m=float(np.max(bed_difference[-1, :6])),
        final_submerged_bank_loss_m=float(np.min(bed_difference[-1, 6:12])),
        final_dry_bank_loss_m=float(np.min(bed_difference[-1, 12:])),
        final_substrate_on_off_by_cell_kg_m2={
            str(cell): substrate_diff[cell].tolist()
            for cell in (3, 8, 9, 14, 15)},
        maximum_toe_supply_m3=float(bank_final[:, 8].max()),
        maximum_collapse_supply_m3=float(bank_final[:, 9].max()),
        max_class_mass_residual_kg=float(np.max(np.abs(mass - mass[0]))),
        max_water_residual_m3=float(np.max(np.abs(water - water[0]))),
        max_signed_toe_face_flux_kg=float(np.max(np.abs(on_flux_final[:, 10]))),
        restart=restart_details,
        erosion_frames_sha256=sha(on["folder"] / "frames.txt"),
        control_frames_sha256=sha(off["folder"] / "frames.txt"),
        erosion_bank_history_sha256=sha(on["folder"] / "bank_failure.txt"),
        erosion_toe_flux_sha256=sha(on["folder"] /
                                    "physical_bank_toe_bedload_flux.txt"),
    )
    save_json(output / "analysis.json", result)
    require(result["passed"], "Native mode-2 activity, conservation, or restart prerequisite failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--phase", choices=("full", "prefix", "resume"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output)
    elif args.action == "run":
        require(args.case is not None and args.phase is not None,
                "--case and --phase required")
        result = run(args.case, args.phase, args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
