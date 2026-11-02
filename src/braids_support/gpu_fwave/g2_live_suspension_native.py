"""Frozen unchanged-native live-section/scheduled suspended-exchange comparator.

This is an engineering reference for a future GPU composition gate, not a
physical braids_support. It never edits the native solver or Delft3D source.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from braids_core.boundaries import cut_cell_io as writer
from braids_core.boundaries.cut_cell_io import CutSedimentBoundarySeries
from braids_qgis.core.results import read_frames
from . import a1_combined_live_scheduled_native_case as a1
from . import coupled_reference as export_reader
from . import cpu_filter_cold20h as frozen_cpu


HERE = Path(__file__).resolve().parent
TAG = "g2_live_suspension_native_20261004_v1"
ROOT = HERE / "runs" / TAG
GATE = HERE / "reports" / TAG / "GATE.json"
VARIANTS = {"off": False, "prognostic": True}
INITIAL_C = np.array([.003, .002], dtype=np.float64)
INLET_C = np.array([[.020, .030, .025, .018],
                    [.010, .008, .015, .012]], dtype=np.float64)
SETTLING_M_S = np.array([.005, .010], dtype=np.float64)
DRY_M = .00025
ENV = dict(a1.ENV, BRAIDS_TIME_INTEGRATOR="legacy")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    def plain(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                     default=plain) + "\n", encoding="utf-8")


def verify_files(mapping):
    for name, digest in mapping.items():
        require(Path(name).is_file() and sha(name) == digest,
                "Frozen file identity differs: " + str(name))


def inputs():
    base = a1.fixture("wetdry")
    q = base["q"].copy()
    q[:, 3:5] = q[:, 0, None] * INITIAL_C[None, :]
    fractions = tuple(replace(fraction,
                              settling_velocity_m_s=float(SETTLING_M_S[index]))
                      for index, fraction in enumerate(base["fractions"]))
    feed = {f"sand_{g}": CutSedimentBoundarySeries(
        tuple(a1.KNOTS_S), tuple(INLET_C[g]), tuple(a1.FEED_KG_S[g]))
        for g in range(2)}
    inlet = replace(base["boundaries"]["in"], fraction_sediment_series=feed)
    return dict(base, q=q, fractions=fractions,
                boundaries={"in": inlet, "out": base["boundaries"]["out"]})


def prepare(output=ROOT):
    output = Path(output).resolve()
    require(not output.exists(), "Use a fresh live-suspension reference folder")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    require(gate["schema"] == 1 and gate["frozen_before_native_or_GPU_evolution"],
            "Pre-GPU live-suspension gate is missing")
    _, native_files = frozen_cpu.runtime_identity()
    sources = {str(Path(module.__file__).resolve()): sha(module.__file__)
               for module in (writer, a1, export_reader, frozen_cpu)}
    sources[str(Path(__file__).resolve())] = sha(__file__)
    sources[str(GATE.resolve())] = sha(GATE)
    sources.update(native_files)
    verify_files(sources)
    data = inputs()
    output.mkdir(parents=True)
    rows = []
    for name, secondary in VARIANTS.items():
        folder = output / "inputs" / name
        folder.mkdir(parents=True)
        writer.write_cut_case(folder / "case.data", data["mesh"], data["q"],
            data["rock"], data["boundaries"], sections={"in": data["section"]},
            end_time_s=a1.END_S, dt_max_s=a1.DT_MAX_S,
            output_interval_s=a1.OUTPUT_S, cfl=.4, scheme=6, bedload=2,
            exchange=1, manning=.03, dry_tolerance_m=DRY_M,
            mpm_friction_mode=1, mpm_critical_shields=.047,
            sediment_fractions=data["fractions"], active_layer_thickness_m=.05,
            substrate_layer_count=1, substrate_layer_thickness_m=.05,
            secondary_flow=secondary, secondary_flow_model="prognostic",
            secondary_flow_transport_direction=secondary,
            secondary_flow_momentum_beta=1. if secondary else 0.,
            secondary_flow_espir=1., morphological_factor=1.,
            compact_sidecars=True,
            source_sha256={"fixture": sources[str(Path(__file__).resolve())],
                           "gate": sources[str(GATE.resolve())]})
        data["mesh"].write_geometry(folder / "geometry.npz")
        np.savez_compressed(folder / "initial_analysis.npz", q=data["q"],
            bed=data["bed"], rock=data["rock"], xy=data["xy"],
            area=np.array([cell.area_m2 for cell in data["mesh"].cells]),
            knots_s=a1.KNOTS_S, q_m3_s=a1.Q_M3_S, bed_feed_kg_s=a1.FEED_KG_S,
            inlet_concentration_kg_m3=INLET_C, initial_concentration_kg_m3=INITIAL_C,
            settling_velocity_m_s=SETTLING_M_S)
        rows.append(dict(variant=name, secondary=secondary,
                         deck=str((folder / "case.data").resolve()),
                         files={str(path.resolve()): sha(path)
                                for path in folder.iterdir() if path.is_file()}))
    plan = dict(schema=1, tag=TAG, gate=str(GATE.resolve()), gate_sha256=sha(GATE),
        sources=sources, cases=rows, environment=ENV,
        native_executable=str(frozen_cpu.TIGHT_EXE.resolve()),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        controls=dict(cells=128, bedlaw=2, exchange=1, integrator="legacy",
                      secondary_variants=VARIANTS, duration_s=a1.END_S,
                      dt_max_s=a1.DT_MAX_S, output_s=a1.OUTPUT_S,
                      q_knots_s=a1.KNOTS_S, concentration_kg_m3=INLET_C),
        GPU_solver_executed=False, observational_validation=False,
        Delft3D_source_modified=False)
    save(output / "plan.json", plan)
    return dict(plan=str(output / "plan.json"), plan_sha256=sha(output / "plan.json"))


def load_plan(output=ROOT):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    require(plan["schema"] == 1 and plan["tag"] == TAG and
            plan["gate_sha256"] == sha(GATE) and
            {row["variant"] for row in plan["cases"]} == set(VARIANTS),
            "Live-suspension plan or gate differs")
    verify_files(plan["sources"])
    for row in plan["cases"]:
        verify_files(row["files"])
    require(sha(plan["native_executable"]) ==
            plan["native_executable_sha256"] == frozen_cpu.TIGHT_SHA,
            "Unchanged native executable differs")
    return plan, sha(output / "plan.json")


def execute(variant, output=ROOT):
    require(variant in VARIANTS, "Unknown live-suspension variant")
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    deck = Path(next(row["deck"] for row in plan["cases"]
                     if row["variant"] == variant))
    cpu = output / variant / "cpu"
    require(not cpu.exists(), "Native execution output must be fresh")
    export_reader.copy_inputs(deck, cpu)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("BRAIDS_", "OMP_"))}
    env.update(ENV)
    start = time.perf_counter()
    with (cpu / "native.log").open("x", encoding="utf-8") as log:
        process = subprocess.run([plan["native_executable"], deck.name],
            cwd=cpu, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    save(cpu / "execution.json", dict(schema=1, variant=variant,
        status="completed" if process.returncode == 0 else "failed",
        return_code=process.returncode, elapsed_s=time.perf_counter()-start,
        plan_sha256=plan_sha, native_executable_sha256=frozen_cpu.TIGHT_SHA,
        files={str(path.resolve()): sha(path) for path in cpu.iterdir()
               if path.is_file()}, Delft3D_source_modified=False))
    require(process.returncode == 0,
            "Unchanged native live-suspension run failed; inspect native.log")
    return dict(variant=variant, native_folder=str(cpu), return_code=process.returncode)


def audit(output=ROOT):
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    limit = gate["native_acceptance"]
    results = {}
    for variant in VARIANTS:
        cpu = output / variant / "cpu"
        execution = json.loads((cpu / "execution.json").read_text(encoding="utf-8"))
        require(execution["status"] == "completed" and
                execution["plan_sha256"] == plan_sha, "Native execution failed")
        verify_files(execution["files"])
        frames = read_frames(cpu / "frames.txt")
        state = np.stack([frame["q"] for frame in frames])
        ledger = np.stack([frame["ledger"] for frame in frames])
        initial = np.load(output / "inputs" / variant / "initial_analysis.npz",
                          allow_pickle=False)
        area = initial["area"]
        layers = state[:, :, 5:9].reshape(len(frames), len(area), 2, 2)
        water = state[:, :, 0] @ area
        class_mass = (state[:, :, 3:5] + layers.sum(axis=2)).transpose(0, 2, 1) @ area
        water_ledger = ledger[:, :, 0].sum(axis=1)
        class_ledger = ledger[:, :, 1:3].sum(axis=1) + ledger[:, :, 3:5].sum(axis=1)
        water_residual = water-water[0]-(water_ledger-water_ledger[0])
        class_residual = class_mass-class_mass[0]-(class_ledger-class_ledger[0])
        wet = np.sum(state[:, :, 0] > DRY_M, axis=1)
        log = (cpu / "native.log").read_text(encoding="utf-8")
        match = re.search(r"steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)", log)
        checks = dict(
            nine_frames=len(frames) == gate["controls"]["frames"],
            expected_clock=np.allclose([frame["time_s"] for frame in frames],
                                       a1.FRAME_TIMES, rtol=0, atol=1e-10),
            state_shape=state.shape == (9, 128, 10 if VARIANTS[variant] else 9),
            ledger_shape=ledger.shape == (9, 2, 6 if VARIANTS[variant] else 5),
            finite=bool(np.isfinite(state).all() and np.isfinite(ledger).all()),
            nonnegative_depth=bool(np.min(state[:, :, 0]) >= limit["minimum_allowed_depth_m"]),
            nonnegative_sediment=bool(np.min(state[:, :, 3:9]) >= limit["minimum_allowed_sediment_kg_m2"]),
            active_wet_dry=int(wet.max()-wet.min()) >= limit["minimum_wet_count_change"],
            live_inlet_water=float(ledger[-1, 0, 0]) >= limit["minimum_live_inlet_water_m3"],
            active_suspension=bool(np.max(np.abs((state[:, :, 3:5]-state[0, :, 3:5]).transpose(0, 2, 1) @ area))
                                   >= limit["minimum_suspended_inventory_change_kg"]),
            both_bed_feeds=bool(np.all(ledger[-1, 0, 3:5] >= limit["minimum_each_class_bed_feed_kg"])),
            water_budget=bool(np.max(np.abs(water_residual)) <= limit["maximum_water_budget_residual_m3"]),
            class_budgets=bool(np.max(np.abs(class_residual)) <= limit["maximum_each_class_budget_residual_kg"]),
            steps=bool(match and int(match.group(1)) >= limit["minimum_steps"]),
            active_secondary=(not VARIANTS[variant] or
                bool(np.max(np.abs(state[1:, :, 9])) > 1e-12 and
                     np.max(np.abs(ledger[-1, :, 5])) > 1e-12)))
        results[variant] = dict(passed=all(checks.values()), checks=checks,
            steps=None if match is None else int(match.group(1)),
            rejected_trials=None if match is None else int(match.group(2)),
            wet_counts=wet.tolist(),
            maximum_water_budget_residual_m3=float(np.max(np.abs(water_residual))),
            maximum_class_budget_residual_kg=float(np.max(np.abs(class_residual))),
            final_suspended_inventory_kg=(state[-1, :, 3:5].T @ area).tolist(),
            final_bed_feed_kg=ledger[-1, 0, 3:5].tolist(),
            frames_sha256=sha(cpu / "frames.txt"))
    report = dict(schema=1, passed=all(row["passed"] for row in results.values()),
        gate_sha256=sha(GATE), plan_sha256=plan_sha,
        native_executable_sha256=frozen_cpu.TIGHT_SHA, variants=results,
        GPU_solver_executed=False, observational_validation=False,
        Delft3D_source_modified=False)
    save(output / "native_audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run-native", "audit"))
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output)
    elif args.action == "run-native":
        if args.variant is None:
            parser.error("run-native requires --variant")
        result = execute(args.variant, args.output)
    else:
        result = audit(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
