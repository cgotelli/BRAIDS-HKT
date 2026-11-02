"""Unchanged-native physical bank-toe reference with GPU-supported MPM mode 1.

This is a separate reference from the preserved mode-0 v2 case. The v2
module supplies geometry, frozen runtime and generic execution/receipt
utilities; this module explicitly writes MPM mode 1 plus Shields 0.047,
owns its own source identity, activity analysis and later pre-GPU gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from braids_core.boundaries.cut_cell_io import write_cut_case
from braids_qgis.core.results import read_frames
from braids_support.gpu_fwave import i4_physical_toe_mode2_native_case as v2


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs/i4_physical_toe_mode2_native_20261004_v3"
SOURCE_EXECUTABLE = v2.SOURCE_EXECUTABLE
checked_execution = v2.checked_execution
CASES = v2.CASES
END_S = .5
PREFIX_S = .2
OUTPUT_S = .1
DT_MAX_S = .002
CRITICAL_UNDERCUT_M = .0005
SPEED_FACTOR = 1.5


def geometry_and_initial() -> tuple:
    mesh, xy, area, state, rock, bank_mask = v2.geometry_and_initial()
    state[:12, 1] *= SPEED_FACTOR
    v2.require(np.all(state[:12, 0] > v2.DRY_TOLERANCE_M) and
               np.all(state[12:, 0] == 0.),
               "Native mode-1 case must have wet toe/donor and dry upper bank")
    return mesh, xy, area, state, rock, bank_mask


def write_input(folder: Path, mesh, xy, area, state, rock, bank_mask,
                *, case: str, duration: float) -> None:
    folder.mkdir(parents=True)
    capacity = 1. if case == "erosion" else 0.
    metadata = write_cut_case(
        folder / "case.data", mesh, state, rock, {},
        end_time_s=duration, dt_max_s=DT_MAX_S, output_interval_s=OUTPUT_S,
        cfl=.4, scheme=6, bedload=2, exchange=0, manning=.03,
        sediment_fractions=v2.FRACTIONS,
        active_layer_thickness_m=v2.ACTIVE_M,
        dry_tolerance_m=v2.DRY_TOLERANCE_M,
        mpm_friction_mode=1, mpm_critical_shields=.047,
        bank_erodible_by_cell=bank_mask, bank_failure=True,
        bank_toe_erosion_mode="noncohesive_transport_capacity",
        bank_toe_capacity_fraction=capacity,
        bank_critical_undercut_m=CRITICAL_UNDERCUT_M,
        source_sha256={"i4_mode1_case_script": v2.sha(Path(__file__)),
                       "v2_geometry_dependency": v2.sha(Path(v2.__file__))},
    )
    v2.require(metadata["format"] == "xbraids_cutcell-v38" and
               metadata["bank_failure"]["toe_erosion_mode"] ==
               "noncohesive_transport_capacity" and
               metadata["mpm_friction_mode"]["mode"] == 1 and
               metadata["mpm_critical_shields"] == .047,
               "Writer did not encode physical toe / MPM mode 1")
    np.savez_compressed(folder / "initial_analysis.npz", xy=xy, area=area,
                        state=state, rock=rock, bulk=v2.BULK,
                        bank_mask=bank_mask)


def prepare(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    v2.require(not output.exists(), "Native mode-1 reference already exists")
    runtime_manifest = v2.SOURCE_RUNTIME / "build.json"
    build = json.loads(runtime_manifest.read_text())
    v2.require(build["executable_sha256"] == v2.sha(v2.SOURCE_EXECUTABLE),
               "Frozen native executable differs from build manifest")
    for name, digest in build["source_sha256"].items():
        v2.require(v2.sha(v2.SOURCE_RUNTIME / name) == digest,
                   "Frozen native source snapshot differs: " + name)
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    executable = runtime / "xbraids_mode2_mpm1_reference.exe"
    shutil.copy2(v2.SOURCE_EXECUTABLE, executable)
    mesh, xy, area, state, rock, bank_mask = geometry_and_initial()
    for case in CASES:
        write_input(output / "inputs" / case, mesh, xy, area, state,
                    rock, bank_mask, case=case, duration=END_S)
        write_input(output / "inputs_prefix" / case, mesh, xy, area, state,
                    rock, bank_mask, case=case, duration=PREFIX_S)
    plan = dict(
        schema=1, tag=output.name,
        copied_native_executable=str(executable),
        copied_native_executable_sha256=v2.sha(executable),
        parent_native_executable=str(v2.SOURCE_EXECUTABLE.resolve()),
        parent_native_executable_sha256=v2.sha(v2.SOURCE_EXECUTABLE),
        parent_build_manifest=str(runtime_manifest.resolve()),
        parent_build_manifest_sha256=v2.sha(runtime_manifest),
        source_sha256={
            "case_script": v2.sha(Path(v2.__file__)),
            "mode1_case_script": v2.sha(Path(__file__)),
            "writer": v2.sha(Path(write_cut_case.__code__.co_filename)),
        },
        settings=dict(duration_s=END_S, prefix_s=PREFIX_S,
                      output_s=OUTPUT_S, dtmax_s=DT_MAX_S,
                      cfl=.4, scheme=6, bedlaw=2, exchange=0,
                      MorFac=1., friction_mode=1,
                      mpm_critical_shields=.047,
                      initial_velocity_scale_from_v2=SPEED_FACTOR,
                      toe_capacity_fraction={"erosion": 1., "control": 0.},
                      critical_undercut_m=CRITICAL_UNDERCUT_M,
                      active_layer_thickness_m=v2.ACTIVE_M,
                      dry_tolerance_m=v2.DRY_TOLERANCE_M,
                      bank_material_cells_zero_based=[int(cell) for cell in
                                                      np.flatnonzero(bank_mask)]),
        inputs={case: v2.files(output / "inputs" / case) for case in CASES},
        prefix_inputs={case: v2.files(output / "inputs_prefix" / case)
                       for case in CASES},
        Delft3D_source_modified=False, GPU_solver_executed=False,
    )
    v2.save_json(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"),
            "plan_sha256": v2.sha(output / "plan.json"),
            "native_executable_sha256": v2.sha(executable)}


def verify_plan(output: Path = DEFAULT_ROOT) -> dict:
    plan = v2.verify_plan(output)
    v2.require(plan["source_sha256"]["mode1_case_script"] ==
               v2.sha(Path(__file__)), "Native mode-1 generator identity differs")
    settings = plan["settings"]
    v2.require(settings["friction_mode"] == 1 and
               settings["mpm_critical_shields"] == .047 and
               settings["critical_undercut_m"] == CRITICAL_UNDERCUT_M and
               settings["initial_velocity_scale_from_v2"] == SPEED_FACTOR,
               "Native mode-1 controls differ")
    for case in CASES:
        metadata = json.loads((Path(output) / "inputs" / case /
                               "case.data.json").read_text())
        v2.require(metadata["mpm_friction_mode"]["mode"] == 1 and
                   metadata["mpm_critical_shields"] == .047,
                   "Native mode-1 deck metadata differs")
    return plan


def run(case: str, phase: str, output: Path = DEFAULT_ROOT) -> dict:
    verify_plan(output)
    return v2.run(case, phase, output)


def analyze(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    plan = verify_plan(output)
    data = {}
    for case in CASES:
        folder = v2.checked_execution(output, case, "full", plan)
        rows = read_frames(folder / "frames.txt")
        state = np.stack([row["q"] for row in rows])
        bank = np.loadtxt(folder / "bank_failure.txt", comments="#", ndmin=2)
        flux = np.loadtxt(folder / "physical_bank_toe_bedload_flux.txt",
                          comments="#", ndmin=2)
        data[case] = dict(folder=folder, rows=rows, state=state,
                          bank=bank, flux=flux)
        for phase in ("prefix", "resume"):
            v2.checked_execution(output, case, phase, plan)
    on, off = data["erosion"], data["control"]
    q, control = on["state"], off["state"]
    initial = np.load(output / "inputs/erosion/initial_analysis.npz",
                      allow_pickle=False)
    area, bulk = initial["area"], initial["bulk"]
    bed_diff = np.sum((q[:, :, 5:7] + q[:, :, 7:9]
                       - control[:, :, 5:7] - control[:, :, 7:9]) /
                      bulk[None, None, :], axis=2)
    mass = np.sum((q[:, :, 5:7] + q[:, :, 7:9]) * area[None, :, None], axis=1)
    water = np.sum(q[:, :, 0] * area[None, :], axis=1)
    bank = on["bank"]
    counts = {}
    for frame in on["rows"]:
        time = frame["time_s"]
        rows = bank[np.isclose(bank[:, 0], time, atol=1e-12)]
        counts[f"{time:.1f}"] = {str(cell): int(rows[rows[:, 1] == cell, 10][0])
                                 for cell in (8, 9)}
    expected = {"0.0": {"8": 0, "9": 0},
                "0.1": {"8": 0, "9": 0},
                "0.2": {"8": 1, "9": 1},
                "0.3": {"8": 1, "9": 2},
                "0.4": {"8": 1, "9": 3},
                "0.5": {"8": 2, "9": 3}}
    substrate = q[-1, :, 7:9] - control[-1, :, 7:9]
    final_bank = bank[np.isclose(bank[:, 0], END_S, atol=1e-12)]
    off_bank = off["bank"][np.isclose(off["bank"][:, 0], END_S, atol=1e-12)]
    final_flux = on["flux"][np.isclose(on["flux"][:, 0], END_S, atol=1e-12)]
    off_final_flux = off["flux"][np.isclose(off["flux"][:, 0], END_S,
                                           atol=1e-12)]
    restart = {}
    for case in CASES:
        full = data[case]["folder"]
        prefix = output / case / "prefix"
        resume = output / case / "resume"
        frames = data[case]["rows"]
        prefix_frames = read_frames(prefix / "frames.txt")
        resume_frames = read_frames(resume / "frames.txt")
        end_bank = np.loadtxt(resume / "bank_failure.txt", comments="#", ndmin=2)
        end_flux = np.loadtxt(resume / "physical_bank_toe_bedload_flux.txt",
                              comments="#", ndmin=2)
        restart[case] = dict(
            prefix_state_exact=np.array_equal(prefix_frames[-1]["q"],
                                               frames[2]["q"]),
            final_state_exact=np.array_equal(resume_frames[-1]["q"],
                                              frames[-1]["q"]),
            checkpoint_bytes_exact=(resume / "checkpoint.bin").read_bytes() ==
                (full / "checkpoint.bin").read_bytes(),
            final_bank_exact=np.array_equal(
                end_bank[np.isclose(end_bank[:, 0], END_S)],
                data[case]["bank"][np.isclose(data[case]["bank"][:, 0], END_S)]),
            final_toe_flux_exact=np.array_equal(
                end_flux[np.isclose(end_flux[:, 0], END_S)],
                data[case]["flux"][np.isclose(data[case]["flux"][:, 0], END_S)]),
        )
    checks = dict(
        six_matched_frames=q.shape == (6, 18, 9) and control.shape == q.shape and
            np.array_equal(q[0], control[0]) and
            [row["steps"] for row in on["rows"]] == [0, 50, 100, 150, 200, 250],
        mixed_wet_dry=np.all(np.count_nonzero(q[:, :, 0] <= v2.DRY_TOLERANCE_M,
                                            axis=1) == 6) and
            np.all(q[:, :12, 0] > v2.DRY_TOLERANCE_M),
        finite_nonnegative=np.isfinite(q).all() and np.isfinite(control).all() and
            np.min(q[:, :, 0]) >= 0 and np.min(q[:, :, 5:9]) >= 0,
        water_conservative=np.max(np.abs(water - water[0])) < 1e-8,
        each_grain_conservative=np.max(np.abs(mass - mass[0])) < 1e-8,
        active_toe_supply=final_bank[:, 8].max() > .0004 and
            np.all(off_bank[:, 8:11] == 0),
        active_collapse=final_bank[:, 9].max() > .00059 and counts == expected,
        both_grains_from_wet_donor_substrate=np.all(substrate[[8, 9]] < -.1),
        both_grains_from_dry_bank_substrate=np.all(substrate[[14, 15]] < -.1),
        wet_toe_gain=np.max(bed_diff[-1, :6]) > .001 and
            np.all(substrate[3] > .3),
        dry_bank_loss=np.min(bed_diff[-1, 12:]) < -.00059,
        signed_gross_toe_face_diagnostic=np.max(np.abs(final_flux[:, 10])) > .07 and
            np.all(final_flux[:, 11] + 1e-14 >= np.abs(final_flux[:, 10])) and
            np.max(np.abs(off_final_flux[:, 10])) > .07,
        erosion_native_restart_exact=all(restart["erosion"].values()),
        control_native_restart_exact=all(restart["control"].values()),
    )
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(
        schema=1, passed=all(checks.values()), checks=checks,
        plan_sha256=v2.sha(output / "plan.json"),
        native_executable_sha256=plan["copied_native_executable_sha256"],
        frame_times_s=[row["time_s"] for row in on["rows"]],
        accepted_steps_by_frame=[int(row["steps"]) for row in on["rows"]],
        collapse_count_by_frame=counts,
        maximum_on_off_bed_difference_by_frame_m=np.max(
            np.abs(bed_diff), axis=1).tolist(),
        final_wet_toe_gain_m=float(np.max(bed_diff[-1, :6])),
        final_submerged_bank_loss_m=float(np.min(bed_diff[-1, 6:12])),
        final_dry_bank_loss_m=float(np.min(bed_diff[-1, 12:])),
        final_substrate_on_off_by_cell_kg_m2={
            str(cell): substrate[cell].tolist() for cell in (3, 8, 9, 14, 15)},
        maximum_toe_supply_m3=float(final_bank[:, 8].max()),
        maximum_collapse_supply_m3=float(final_bank[:, 9].max()),
        max_class_mass_residual_kg=float(np.max(np.abs(mass - mass[0]))),
        max_water_residual_m3=float(np.max(np.abs(water - water[0]))),
        max_signed_toe_face_flux_kg=float(np.max(np.abs(final_flux[:, 10]))),
        restart=restart,
        erosion_frames_sha256=v2.sha(on["folder"] / "frames.txt"),
        control_frames_sha256=v2.sha(off["folder"] / "frames.txt"),
        erosion_bank_history_sha256=v2.sha(on["folder"] / "bank_failure.txt"),
        erosion_toe_flux_sha256=v2.sha(on["folder"] /
                                      "physical_bank_toe_bedload_flux.txt"),
    )
    v2.save_json(output / "analysis.json", result)
    v2.require(result["passed"], "Native mode-1 physical-toe prerequisite failed")
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
        v2.require(args.case is not None and args.phase is not None,
                   "--case and --phase required")
        result = run(args.case, args.phase, args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
