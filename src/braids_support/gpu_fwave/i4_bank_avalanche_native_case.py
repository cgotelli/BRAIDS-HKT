"""Native-only, four-way bank-mode-2/general-avalanche engineering fixture.

The numerical executable is copied from an authenticated frozen BRAIDS CPU
runtime. This module changes input decks only and never builds source code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from braids_core.boundaries.cut_cell_io import write_cut_case
from braids_qgis.core.results import read_frames
from braids_support.gpu_fwave import i4_physical_toe_mode2_native_case as base
from braids_support.gpu_fwave import i4_physical_toe_mode2_native_case_v3 as mode1


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "runs/i4_bank_avalanche_native_20261004_v1"
CASES = ("bank0_aval0", "bank1_aval0", "bank0_aval1", "bank1_aval1")
END_S, PREFIX_S, OUTPUT_S, DT_MAX_S = .5, .2, .1, .002
WET_SLOPE = .08
DRY_SLOPE = .08
AVALANCHE_TIME_S = 100.
RELAXATION_LENGTH_M = 1.


def flags(case: str) -> tuple[bool, bool]:
    base.require(case in CASES, "Unknown bank/avalanche case")
    return case[4] == "1", case[-1] == "1"


def write_input(folder: Path, mesh, xy, area, state, rock, bank_mask,
                *, case: str, duration: float) -> None:
    folder.mkdir(parents=True)
    bank_on, avalanche_on = flags(case)
    metadata = write_cut_case(
        folder / "case.data", mesh, state, rock, {},
        end_time_s=duration, dt_max_s=DT_MAX_S,
        output_interval_s=OUTPUT_S, cfl=.4, scheme=6,
        bedload=2, exchange=0, manning=.03,
        sediment_fractions=base.FRACTIONS,
        active_layer_thickness_m=base.ACTIVE_M,
        dry_tolerance_m=base.DRY_TOLERANCE_M,
        mpm_friction_mode=1, mpm_critical_shields=.047,
        bank_erodible_by_cell=bank_mask, bank_failure=True,
        bank_toe_erosion_mode="noncohesive_transport_capacity",
        bank_toe_capacity_fraction=1. if bank_on else 0.,
        bank_critical_undercut_m=mode1.CRITICAL_UNDERCUT_M,
        general_avalanche=avalanche_on,
        general_avalanche_wet_slope=WET_SLOPE,
        general_avalanche_dry_slope=DRY_SLOPE,
        general_avalanche_time_s=AVALANCHE_TIME_S,
        general_avalanche_wet_depth_switch_m=.1,
        general_avalanche_relaxation_length_m=RELAXATION_LENGTH_M,
        source_sha256={"bank_avalanche_case_script": base.sha(Path(__file__)),
                       "mode1_geometry_source": base.sha(Path(mode1.__file__))},
    )
    base.require(metadata["format"] == "xbraids_cutcell-v38" and
                 metadata["bank_failure"]["toe_erosion_mode"] ==
                 "noncohesive_transport_capacity" and
                 metadata["general_avalanche"]["enabled"] == avalanche_on,
                 "Combined controls not encoded by native case writer")
    np.savez_compressed(folder / "initial_analysis.npz", xy=xy, area=area,
                        state=state, rock=rock, bulk=base.BULK,
                        bank_mask=bank_mask)


def prepare(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    base.require(not output.exists(), "Native combination reference already exists")
    manifest = base.SOURCE_RUNTIME / "build.json"
    build = json.loads(manifest.read_text())
    base.require(build["executable_sha256"] == base.sha(base.SOURCE_EXECUTABLE),
                 "Frozen native executable differs from build manifest")
    for name, digest in build["source_sha256"].items():
        base.require(base.sha(base.SOURCE_RUNTIME / name) == digest,
                     "Frozen native source snapshot differs: " + name)
    output.mkdir(parents=True)
    runtime = output / "runtime"
    runtime.mkdir()
    executable = runtime / "xbraids_bank_avalanche_reference.exe"
    shutil.copy2(base.SOURCE_EXECUTABLE, executable)
    mesh, xy, area, state, rock, bank_mask = mode1.geometry_and_initial()
    for case in CASES:
        write_input(output / "inputs" / case, mesh, xy, area, state, rock,
                    bank_mask, case=case, duration=END_S)
        write_input(output / "inputs_prefix" / case, mesh, xy, area, state,
                    rock, bank_mask, case=case, duration=PREFIX_S)
    plan = dict(
        schema=1, tag=output.name,
        copied_native_executable=str(executable),
        copied_native_executable_sha256=base.sha(executable),
        parent_native_executable=str(base.SOURCE_EXECUTABLE.resolve()),
        parent_native_executable_sha256=base.sha(base.SOURCE_EXECUTABLE),
        parent_build_manifest=str(manifest.resolve()),
        parent_build_manifest_sha256=base.sha(manifest),
        source_sha256={
            "case_script": base.sha(Path(__file__)),
            "mode1_geometry": base.sha(Path(mode1.__file__)),
            "base_utilities": base.sha(Path(base.__file__)),
            "writer": base.sha(Path(write_cut_case.__code__.co_filename)),
        },
        settings=dict(duration_s=END_S, prefix_s=PREFIX_S,
                      output_s=OUTPUT_S, dtmax_s=DT_MAX_S, cfl=.4,
                      scheme=6, bedlaw=2, exchange=0, MorFac=1.,
                      friction_mode=1, critical_shields=.047,
                      toe_capacity_fraction={case: int(flags(case)[0])
                                             for case in CASES},
                      general_avalanche={case: flags(case)[1]
                                         for case in CASES},
                      avalanche_wet_slope=WET_SLOPE,
                      avalanche_dry_slope=DRY_SLOPE,
                      avalanche_time_s=AVALANCHE_TIME_S,
                      avalanche_relaxation_length_m=RELAXATION_LENGTH_M,
                      critical_undercut_m=mode1.CRITICAL_UNDERCUT_M,
                      active_layer_thickness_m=base.ACTIVE_M,
                      dry_tolerance_m=base.DRY_TOLERANCE_M,
                      bank_material_cells_zero_based=np.flatnonzero(
                          bank_mask).astype(int).tolist()),
        inputs={case: base.files(output / "inputs" / case) for case in CASES},
        prefix_inputs={case: base.files(output / "inputs_prefix" / case)
                       for case in CASES},
        Delft3D_source_modified=False, GPU_solver_executed=False,
    )
    base.save_json(output / "plan.json", plan)
    return {"plan_sha256": base.sha(output / "plan.json"),
            "native_executable_sha256": base.sha(executable),
            "cases": CASES}


def verify_plan(output: Path = DEFAULT_ROOT) -> dict:
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    source = plan["source_sha256"]
    base.require(plan["schema"] == 1 and plan["tag"] == output.name and
                 source["case_script"] == base.sha(Path(__file__)) and
                 source["mode1_geometry"] == base.sha(Path(mode1.__file__)) and
                 source["base_utilities"] == base.sha(Path(base.__file__)) and
                 source["writer"] == base.sha(Path(write_cut_case.__code__.co_filename)),
                 "Native combination source identity differs")
    base.require(plan["copied_native_executable_sha256"] ==
                 base.sha(Path(plan["copied_native_executable"])) and
                 plan["parent_build_manifest_sha256"] ==
                 base.sha(Path(plan["parent_build_manifest"])),
                 "Native combination runtime identity differs")
    for case in CASES:
        base.require(base.files(output / "inputs" / case) == plan["inputs"][case] and
                     base.files(output / "inputs_prefix" / case) ==
                     plan["prefix_inputs"][case],
                     "Native combination input changed: " + case)
    return plan


def run(case: str, phase: str, output: Path = DEFAULT_ROOT) -> dict:
    base.require(case in CASES and phase in ("full", "prefix", "resume"),
                 "Unknown case or phase")
    output = Path(output).resolve()
    plan = verify_plan(output)
    folder = output / case / phase
    base.require(not folder.exists(), "Native combination execution already exists")
    if phase == "resume":
        prefix = output / case / "prefix" / "checkpoint.bin"
        base.require(prefix.is_file(), "Native prefix checkpoint missing")
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
    (folder / "stdout.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    receipt = dict(schema=1, case=case, phase=phase,
                   status="completed" if result.returncode == 0 else "failed",
                   return_code=result.returncode, command=command,
                   restart_checkpoint_sha256=base.sha(prefix) if prefix else None,
                   plan_sha256=base.sha(output / "plan.json"),
                   native_executable_sha256=plan["copied_native_executable_sha256"],
                   environment={key: environment[key] for key in
                                ("BRAIDS_TIME_INTEGRATOR", "OMP_NUM_THREADS",
                                 "OMP_DYNAMIC", "OMP_WAIT_POLICY")},
                   output_files=base.files(folder))
    base.save_json(folder / "execution.json", receipt)
    base.require(result.returncode == 0,
                 "Native combination run failed: " +
                 (result.stdout + result.stderr)[-1800:])
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--phase", choices=("full", "prefix", "resume"))
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output)
    else:
        base.require(args.case is not None and args.phase is not None,
                     "--case and --phase required")
        result = run(args.case, args.phase, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
