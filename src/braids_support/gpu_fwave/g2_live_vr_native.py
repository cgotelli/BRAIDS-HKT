"""Write-once unchanged-native live Van Rijn exchange-2 engineering reference.

This is an implementation composition gate, not observational braids_support.  It
does not edit the numerical executable, Delft3D, or the existing G2 reference.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
from . import g2_live_suspension_native as g2


HERE = Path(__file__).resolve().parent
TAG = "g2_live_vr_native_20261004_v1"
ROOT = HERE / "runs" / TAG
GATE = HERE / "reports" / TAG / "GATE.json"
VARIANTS = {"off": False, "prognostic": True}
INITIAL_C = np.array([1e-5, 10.0], dtype=np.float64)
INLET_C = np.array([[1e-5, 2e-5, 1.5e-5, 1e-5],
                    [10.0, 8.0, 12.0, 9.0]], dtype=np.float64)
SETTLING_M_S = np.array([.005, .010], dtype=np.float64)
DRY_M = .00025
ENV = dict(a1.ENV, BRAIDS_TIME_INTEGRATOR="legacy")


def sha(path):
    return g2.sha(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    g2.save(path, value)


def verify_files(mapping):
    g2.verify_files(mapping)


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
    require(not output.exists(), "Use a fresh live-Van-Rijn reference folder")
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    require(gate["schema"] == 1 and gate["frozen_before_native_or_GPU_evolution"]
            and gate["controls"]["exchange_mode"] == 2,
            "The frozen exchange-2 gate is missing")
    require(np.array_equal(INITIAL_C, gate["controls"]["initial_concentration_kg_m3"])
            and np.array_equal(INLET_C, gate["controls"]["inlet_concentration_kg_m3_by_class"])
            and np.array_equal(SETTLING_M_S, gate["controls"]["settling_velocity_m_s"]),
            "Frozen concentration or settling controls differ")
    _, native_files = frozen_cpu.runtime_identity()
    sources = {str(Path(module.__file__).resolve()): sha(module.__file__)
               for module in (writer, a1, export_reader, frozen_cpu, g2)}
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
            exchange=2, manning=.03, dry_tolerance_m=DRY_M,
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
            inlet_concentration_kg_m3=INLET_C,
            initial_concentration_kg_m3=INITIAL_C,
            settling_velocity_m_s=SETTLING_M_S)
        rows.append(dict(variant=name, secondary=secondary,
                         deck=str((folder / "case.data").resolve()),
                         files={str(path.resolve()): sha(path)
                                for path in folder.iterdir() if path.is_file()}))
    plan = dict(schema=1, tag=TAG, gate=str(GATE.resolve()), gate_sha256=sha(GATE),
        sources=sources, cases=rows, environment=ENV,
        native_executable=str(frozen_cpu.TIGHT_EXE.resolve()),
        native_executable_sha256=frozen_cpu.TIGHT_SHA,
        controls=dict(cells=128, bedlaw=2, exchange=2, integrator="legacy",
                      secondary_variants=VARIANTS, duration_s=a1.END_S,
                      dt_max_s=a1.DT_MAX_S, output_s=a1.OUTPUT_S,
                      q_knots_s=a1.KNOTS_S, concentration_kg_m3=INLET_C,
                      initial_concentration_kg_m3=INITIAL_C,
                      settling_velocity_m_s=SETTLING_M_S),
        GPU_solver_executed=False, observational_validation=False,
        Delft3D_source_modified=False)
    save(output / "plan.json", plan)
    return dict(plan=str(output / "plan.json"), plan_sha256=sha(output / "plan.json"),
                gate_sha256=sha(GATE))


def load_plan(output=ROOT):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    require(plan["schema"] == 1 and plan["tag"] == TAG and
            plan["gate_sha256"] == sha(GATE) and
            {row["variant"] for row in plan["cases"]} == set(VARIANTS),
            "Frozen live-Van-Rijn plan or gate differs")
    verify_files(plan["sources"])
    for row in plan["cases"]:
        verify_files(row["files"])
    require(sha(plan["native_executable"]) ==
            plan["native_executable_sha256"] == frozen_cpu.TIGHT_SHA,
            "Unchanged native executable differs")
    return plan, sha(output / "plan.json")


def execute(variant, output=ROOT):
    require(variant in VARIANTS, "Unknown live-Van-Rijn variant")
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    deck = Path(next(row["deck"] for row in plan["cases"]
                     if row["variant"] == variant))
    cpu = output / variant / "cpu"
    require(not cpu.exists(), "Native evolution output must be fresh")
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
            "Unchanged native live-Van-Rijn run failed; inspect native.log")
    return dict(variant=variant, native_folder=str(cpu), return_code=process.returncode)


def analyze(output=ROOT):
    output = Path(output).resolve()
    plan, plan_sha = load_plan(output)
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    limit = gate["native_acceptance"]
    rows = {}
    for variant, secondary_on in VARIANTS.items():
        cpu = output / variant / "cpu"
        execution = json.loads((cpu / "execution.json").read_text(encoding="utf-8"))
        require(execution["status"] == "completed" and
                execution["return_code"] == 0 and
                execution["plan_sha256"] == plan_sha and
                execution["native_executable_sha256"] == frozen_cpu.TIGHT_SHA,
                "Native execution/plan identity differs")
        verify_files(execution["files"])
        frames = read_frames(cpu / "frames.txt")
        times = np.array([frame["time_s"] for frame in frames])
        state = np.stack([frame["q"] for frame in frames])
        ledger = np.stack([frame["ledger"] for frame in frames])
        initial = np.load(output / "inputs" / variant / "initial_analysis.npz",
                          allow_pickle=False)
        area = np.asarray(initial["area"])
        nframe = len(frames)
        layers = state[:, :, 5:9].reshape(nframe, 128, 2, 2)
        suspended = state[:, :, 3:5].transpose(0, 2, 1) @ area
        bed = layers.sum(axis=2).transpose(0, 2, 1) @ area
        water = state[:, :, 0] @ area
        water_boundary = ledger[:, :, 0].sum(axis=1)
        suspended_boundary = ledger[:, :, 1:3].sum(axis=1)
        bed_boundary = ledger[:, :, 3:5].sum(axis=1)
        water_residual = water-water[0]-(water_boundary-water_boundary[0])
        class_residual = suspended+bed-(suspended[0]+bed[0])-(
            suspended_boundary+bed_boundary-
            (suspended_boundary[0]+bed_boundary[0]))
        exchange = suspended-suspended[0]-(suspended_boundary-suspended_boundary[0])
        wet = np.sum(state[:, :, 0] > DRY_M, axis=1)
        log = (cpu / "native.log").read_text(encoding="utf-8")
        match = re.search(r"steps, rejected trials, CPU seconds:\s*(\d+)\s+(\d+)", log)
        expected_q = np.array([a1.integral_at(a1.KNOTS_S, a1.Q_M3_S, t)
                               for t in times])
        expected_feed = np.array([[a1.integral_at(a1.KNOTS_S,
            a1.FEED_KG_S[g], t) for g in range(2)] for t in times])
        secondary = None
        if secondary_on:
            raw = np.loadtxt(cpu / "secondary_flow.txt").reshape(9, 128, 2, 10)
            budget = np.loadtxt(cpu / "secondary_flow_budget.txt")
            require(budget.shape == (9, 6) and
                    np.allclose(raw[:, :, :, 0], times[:, None, None], atol=1e-10, rtol=0)
                    and np.allclose(budget[:, 0], times, atol=1e-10, rtol=0),
                    "Native secondary diagnostic layout/clock differs")
            intensity = raw[:, :, 0, 6]
            hi = state[:, :, 0] * intensity
            residual = budget[:, 1]-budget[:, 2]-budget[:, 3]-budget[:, 4]
            secondary = dict(maximum_intensity_m_s=float(np.max(np.abs(intensity))),
                maximum_hI_m2_s=float(np.max(np.abs(hi))),
                final_boundary_hI=float(budget[-1, 3]),
                maximum_inventory_budget_difference=float(
                    np.max(np.abs(hi @ area-budget[:, 1]))),
                maximum_closure_residual=float(np.max(np.abs(residual))),
                maximum_reported_residual=float(np.max(np.abs(budget[:, 5]))))
        checks = dict(
            nine_frames=nframe == gate["controls"]["frames"],
            expected_clock=np.allclose(times, a1.FRAME_TIMES, rtol=0, atol=1e-10),
            public_layout=state.shape == (9, 128, 9) and ledger.shape == (9, 2, 5),
            finite=bool(np.isfinite(state).all() and np.isfinite(ledger).all()),
            nonnegative_depth=bool(np.min(state[:, :, 0]) >= limit["minimum_allowed_depth_m"]),
            nonnegative_sediment=bool(np.min(state[:, :, 3:9]) >= limit["minimum_allowed_sediment_kg_m2"]),
            active_wet_dry=int(wet.max()-wet.min()) >= limit["minimum_wet_count_change"],
            live_inlet_water=float(ledger[-1, 0, 0]) >= limit["minimum_live_inlet_water_m3"],
            each_class_suspended_inlet=bool(np.all(ledger[-1, 0, 1:3] >=
                limit["minimum_each_class_suspended_inlet_kg"])),
            each_class_bed_feed=bool(np.all(ledger[-1, 0, 3:5] >=
                limit["minimum_each_class_bed_feed_kg"])),
            inlet_q_integral=bool(np.max(np.abs(ledger[:, 0, 0]-expected_q)) <= 1e-8),
            inlet_feed_integrals=bool(np.max(np.abs(ledger[:, 0, 3:5]-expected_feed)) <= 1e-8),
            class_zero_erosion_to_water=float(exchange[-1, 0]) >=
                limit["minimum_class_zero_net_erosion_to_water_kg"],
            class_one_deposition_to_bed=float(-exchange[-1, 1]) >=
                limit["minimum_class_one_net_deposition_to_bed_kg"],
            distinct_exchange=float(abs(exchange[-1, 0]-exchange[-1, 1])) >=
                limit["minimum_distinct_class_exchange_kg"],
            water_budget=bool(np.max(np.abs(water_residual)) <=
                limit["maximum_water_budget_residual_m3"]),
            class_budgets=bool(np.max(np.abs(class_residual)) <=
                limit["maximum_each_class_budget_residual_kg"]),
            steps=bool(match and int(match.group(1)) >= limit["minimum_steps"]),
            secondary_activity=(not secondary_on or bool(
                secondary["maximum_hI_m2_s"] >= limit["minimum_prognostic_hI_m2_s"]
                and secondary["maximum_inventory_budget_difference"] <=
                    limit["maximum_secondary_budget_residual"]
                and secondary["maximum_closure_residual"] <=
                    limit["maximum_secondary_budget_residual"]
                and secondary["maximum_reported_residual"] <=
                    limit["maximum_secondary_budget_residual"])))
        rows[variant] = dict(passed=all(checks.values()), checks=checks,
            accepted_steps=None if match is None else int(match.group(1)),
            rejected_trials=None if match is None else int(match.group(2)),
            wet_counts=wet.tolist(),
            maximum_water_budget_residual_m3=float(np.max(np.abs(water_residual))),
            maximum_each_class_budget_residual_kg=np.max(np.abs(class_residual), axis=0).tolist(),
            final_signed_exchange_to_water_kg=exchange[-1].tolist(),
            final_suspended_inventory_kg=suspended[-1].tolist(),
            final_bed_inventory_kg=bed[-1].tolist(),
            final_suspended_inlet_kg=ledger[-1, 0, 1:3].tolist(),
            final_bed_feed_kg=ledger[-1, 0, 3:5].tolist(),
            final_suspended_outlet_kg=ledger[-1, 1, 1:3].tolist(),
            final_bed_outlet_kg=ledger[-1, 1, 3:5].tolist(),
            maximum_inlet_q_error_m3=float(np.max(np.abs(ledger[:, 0, 0]-expected_q))),
            maximum_each_class_feed_error_kg=np.max(
                np.abs(ledger[:, 0, 3:5]-expected_feed), axis=0).tolist(),
            secondary=secondary, frames_sha256=sha(cpu / "frames.txt"))
    result = dict(schema=1, passed=all(row["passed"] for row in rows.values()),
        gate_sha256=sha(GATE), plan_sha256=plan_sha,
        frozen_driver_sha256=sha(__file__),
        native_executable_sha256=frozen_cpu.TIGHT_SHA, variants=rows,
        GPU_solver_executed=False, observational_validation=False,
        Delft3D_source_modified=False)
    destination = output / "native_audit.json"
    require(not destination.exists(), "Native audit is write-once")
    save(destination, result)
    require(result["passed"], "Native live-Van-Rijn gate failed: " +
            str({name: row["checks"] for name, row in rows.items()}))
    return result


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
        result = analyze(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
