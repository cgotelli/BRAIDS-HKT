"""Read-only A1 full-trajectory prerequisite and GPU-array acceptance checks.

The immutable human-readable gate is reports/a1_combined_full_gate_20261004_v1/
GPU_FULL_ACCEPTANCE_GATE.txt. This helper cannot qualify rollback, restart,
native/GPU timings, or the dynamic section-bed provenance by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from braids_qgis.core.results import read_frames
from . import a1_combined_live_scheduled_native_case as native


HERE = Path(__file__).resolve().parent
GATE = HERE / "reports/a1_combined_full_gate_20261004_v1/GPU_FULL_ACCEPTANCE_GATE.txt"
GATE_SHA = "d26632b692d508f372e9ccbf09c696c33f75c585dfce09072df67d77be15e823"
PLAN_SHA = "53ba92f818daf0924134ec7dddee26d6f149f879291eaaea4651a3c791222dcf"
EXE_SHA = "21b7b9290384f5b7b0c6536403cfe67ddd98a65a321f0ae31c24a57085b3aba5"
PINNED = {
    "pool": {
        "input": "1b5bef3b4b586b50353620fd63ee6105eb6b6b46396bf4255f6a6c9af23a59d3",
        "schedule": "0b1079e179250a3d268940bb700bfeb9760db251f07b4f9ff5305b7bc0743597",
        "frames": "ee4e2f74574014c5b36e055e9e110dda74d09f3911fd3d08a37685b5fbadb987",
        "dry_counts": [0] * 9,
    },
    "wetdry": {
        "input": "9a839e737dabc69cdd69188617d1f1d74b337c5f961c7d01240b430ba17417c0",
        "schedule": "0b1079e179250a3d268940bb700bfeb9760db251f07b4f9ff5305b7bc0743597",
        "frames": "8456cd4f0e5a016ea3486e27919e2ed1825ed53c760577d3a2236119fb1beaa7",
        "dry_counts": [32, 30, 30, 30, 28, 28, 28, 28, 28],
    },
}
LIMITS = dict(depth_m=5e-5, momentum_m2_s=5e-5, bed_m=5e-6,
              each_layer_kg_m2=1e-4, water_ledger_m3=1e-6,
              each_class_bed_ledger_kg=1e-5)


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def preflight(variant: str) -> tuple[dict, list, dict]:
    """Read only: verify full native identity and the existing analysis receipt."""
    require(variant in PINNED, "Unknown A1 variant")
    require(native.sha(GATE) == GATE_SHA, "Frozen full gate changed")
    plan, plan_sha = native.load_plan()
    require(plan_sha == PLAN_SHA and plan["native_executable_sha256"] == EXE_SHA,
            "Native plan/executable identity changed")
    paths = native.ROOT / variant
    pinned = PINNED[variant]
    for name, expected in (("prepared/coupled_input.bin", pinned["input"]),
                           ("prepared/normal_metadata.bin", pinned["schedule"]),
                           ("cpu/frames.txt", pinned["frames"])):
        require(native.sha(paths / name) == expected, "Native artifact changed: " + name)
    analysis = json.loads((paths / "analysis.json").read_text(encoding="utf-8"))
    require(analysis["passed"] and all(analysis["checks"].values()) and
            analysis["plan_sha256"] == PLAN_SHA and
            analysis["native_executable_sha256"] == EXE_SHA and
            analysis["native_frames_sha256"] == pinned["frames"] and
            analysis["dry_cell_counts"] == pinned["dry_counts"] and
            analysis["accepted_steps"] == 104 and analysis["rejected_trials"] == 0,
            "Native analysis prerequisite failed")
    execution = json.loads((paths / "cpu/execution.json").read_text(encoding="utf-8"))
    require(execution["status"] == "completed" and execution["return_code"] == 0 and
            execution["native_executable_sha256"] == EXE_SHA,
            "Native execution receipt failed")
    native.verify_files(execution["output_files"])
    frames = read_frames(paths / "cpu/frames.txt")
    data = native.ancestor.read_case(paths / "prepared")
    require(len(frames) == 9 and int(data["nc"]) == 128 and
            int(data["nfrac"]) == 2 and
            np.array_equal(data["mode"], [9, 4]), "Native layout differs")
    return analysis, frames, data


def evaluate_frames(variant: str, gpu: dict[str, np.ndarray]) -> dict:
    """Compare nine resident GPU output snapshots against frozen native frames.

    Arrays in gpu are frame-major. Required keys and shapes are time_s (9,),
    q (9,3,128), suspended (9,2,128), layers (9,2,2,128),
    bed_elevation (9,128), ledger_water (9,2), ledger_suspended (9,2,2),
    ledger_bed (9,2,2). Ledgers retain native signed orientation.
    """
    analysis, frames, data = preflight(variant)
    shapes = {"time_s": (9,), "q": (9, 3, 128),
              "suspended": (9, 2, 128), "layers": (9, 2, 2, 128),
              "bed_elevation": (9, 128), "ledger_water": (9, 2),
              "ledger_suspended": (9, 2, 2), "ledger_bed": (9, 2, 2)}
    arrays = {}
    for name, shape in shapes.items():
        require(name in gpu, "GPU frame archive lacks " + name)
        arrays[name] = np.asarray(gpu[name], dtype=np.float64)
        require(arrays[name].shape == shape, "GPU frame shape differs: " + name)
    times = arrays["time_s"]
    q = arrays["q"]
    suspended = arrays["suspended"]
    layers = arrays["layers"]
    bed = arrays["bed_elevation"]
    lw = arrays["ledger_water"]
    ls = arrays["ledger_suspended"]
    lb = arrays["ledger_bed"]
    reference_q = np.stack([frame["q"] for frame in frames])
    reference_ledger = np.stack([frame["ledger"] for frame in frames])
    native_layers = reference_q[:, :, 5:9].reshape(9, 128, 2, 2)
    native_bed = np.asarray(data["rock"])[None, :] + np.sum(
        native_layers / np.asarray(data["bulk_fraction"])[None, None, None, :],
        axis=(2, 3))
    diff = {
        "depth_m": np.max(np.abs(q[:, 0, :] - reference_q[:, :, 0]), axis=1),
        "momentum_m2_s": np.max(np.abs(q[:, 1:3, :] -
                                   reference_q[:, :, 1:3].transpose(0, 2, 1)), axis=(1, 2)),
        "bed_m": np.max(np.abs(bed - native_bed), axis=1),
        "each_layer_kg_m2": np.max(np.abs(layers - native_layers.transpose(0, 2, 3, 1)),
                                    axis=(1, 2, 3)),
        "water_ledger_m3": np.max(np.abs(lw - reference_ledger[:, :, 0]), axis=1),
        "each_class_bed_ledger_kg": np.max(np.abs(lb.transpose(0, 2, 1) -
                                                  reference_ledger[:, :, 3:5]), axis=(1, 2)),
    }
    area = np.asarray(data["area"])
    water_inventory = q[:, 0, :] @ area
    # Resident layers and native layer_mass are (layer, class, cell), whereas
    # suspended mass is (class, cell). Sum over layers, not grain classes.
    class_inventory = (suspended + layers.sum(axis=1)) @ area
    water_residual = water_inventory - water_inventory[0] - (
        lw.sum(axis=1) - lw[0].sum())
    class_boundary = (ls + lb).sum(axis=2)
    class_residual = class_inventory - class_inventory[0] - (
        class_boundary - class_boundary[0])
    inlet = int(np.flatnonzero(data["mode"] == 9)[0])
    outlet = int(np.flatnonzero(data["mode"] == 4)[0])
    expected_q = np.array([native.integral_at(native.KNOTS_S, native.Q_M3_S, t)
                           for t in times])
    expected_feed = np.array([[native.integral_at(native.KNOTS_S, native.FEED_KG_S[g], t)
                               for g in range(2)] for t in times])
    dry = (q[:, 0, :] <= float(data["dry_tolerance"]))
    checks = {
        "finite": all(np.isfinite(value).all() for value in arrays.values()),
        "nine_frame_clocks": bool(np.max(np.abs(times - native.FRAME_TIMES)) <= 1e-10),
        "nonnegative": bool(np.min(q[:, 0, :]) >= -1e-12 and
                            np.min(suspended) >= -1e-12 and np.min(layers) >= -1e-12),
        "no_suspension_activation": bool(np.max(np.abs(suspended)) <= 1e-12 and
                                         np.max(np.abs(ls)) <= 1e-12),
        "frame_limits": all(np.max(diff[key]) <= limit for key, limit in LIMITS.items()),
        "water_budget": bool(np.max(np.abs(water_residual)) <= 1e-8),
        "class_budgets": bool(np.max(np.abs(class_residual)) <= 1e-7),
        "inlet_Q_integral": bool(np.max(np.abs(lw[:, inlet] - expected_q)) <= 1e-6),
        "inlet_class_feed_integral": bool(np.max(np.abs(lb[:, :, inlet] - expected_feed)) <= 1e-6),
        "outlet_water_outward": bool(np.max(lw[:, outlet]) <= 1e-12),
        "dry_counts": dry.sum(axis=1).tolist() == PINNED[variant]["dry_counts"],
        "wetdry_active": (not dry.any() if variant == "pool" else
                           bool(dry[0].any() and (~dry[0]).any() and np.any(dry != dry[0]))),
    }
    return dict(variant=variant, passed_numeric=all(checks.values()), checks=checks,
                native_frames_sha256=PINNED[variant]["frames"],
                original_native_analysis_passed=analysis["passed"],
                frame_differences={key:value.tolist() for key,value in diff.items()},
                maximum_water_residual_m3=float(np.max(np.abs(water_residual))),
                maximum_class_residual_kg=np.max(np.abs(class_residual), axis=0).tolist(),
                maximum_inlet_Q_error_m3=float(np.max(np.abs(lw[:, inlet] - expected_q))),
                maximum_inlet_feed_error_kg=np.max(np.abs(lb[:, :, inlet] - expected_feed),
                                                   axis=0).tolist(),
                dry_cell_counts=dry.sum(axis=1).tolist(),
                scope="Nine GPU frames only; accepted-step, rollback, restart and timing gates separate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "frames"))
    parser.add_argument("--variant", choices=tuple(PINNED), required=True)
    parser.add_argument("--gpu-frames", type=Path)
    args = parser.parse_args()
    if args.action == "preflight":
        analysis, frames, data = preflight(args.variant)
        result = dict(variant=args.variant, ready=True, gate_sha256=GATE_SHA,
                      native_plan_sha256=PLAN_SHA, native_executable_sha256=EXE_SHA,
                      native_frames_sha256=PINNED[args.variant]["frames"],
                      native_steps=analysis["accepted_steps"], frames=len(frames),
                      cells=int(data["nc"]))
    else:
        parser.error("--gpu-frames is required") if args.gpu_frames is None else None
        with np.load(args.gpu_frames, allow_pickle=False) as archive:
            result = evaluate_frames(args.variant, dict(archive))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
