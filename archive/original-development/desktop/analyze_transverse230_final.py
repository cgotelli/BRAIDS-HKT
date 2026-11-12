"""Seal the 300 s transverse-mode comparison and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
BASE = PROJECT / "runs/experiment229_gate_300s/frames"
CANDIDATE = ROOT / "validation/_transverse230/candidate_restart300/run/frames"
OUT = ROOT / "validation/_transverse230/final"


def geometry(folder: Path):
    index = json.loads((folder / "INDEX.json").read_text(encoding="utf8"))
    with np.load(folder / index["frames"][0]["epoch"]) as mesh:
        vertices, offsets = mesh["vertices"], mesh["offsets"]
    cx = np.array([
        (vertices[offsets[i]:offsets[i + 1], 0].min()
         + vertices[offsets[i]:offsets[i + 1], 0].max()) / 2
        for i in range(len(offsets) - 1)
    ])
    cy = np.array([
        (vertices[offsets[i]:offsets[i + 1], 1].min()
         + vertices[offsets[i]:offsets[i + 1], 1].max()) / 2
        for i in range(len(offsets) - 1)
    ])
    return index, np.rint((cx - .025) / .05).astype(int), np.rint((cy - .025) / .05).astype(int)


def load(folder: Path):
    index, ix, iy = geometry(folder)

    def grid(values):
        result = np.empty((30, 500))
        result[iy, ix] = values
        return result

    result = {}
    raw = {}
    for row in index["frames"]:
        with np.load(folder / row["file"]) as frame:
            h = grid(frame["q"][0])
            u = grid(np.divide(
                frame["q"][1], frame["q"][0],
                out=np.zeros_like(frame["q"][0]), where=frame["q"][0] > 0,
            ))
            bed = grid(frame["z"] - frame["initial_bed_m"])
            transport = grid(np.hypot(*frame["bedload_vector_m2_s"]))
            state = frame["state"].copy()
        result[float(row["time_s"])] = dict(
            depth=h, velocity=u, bed=bed, transport=transport
        )
        raw[float(row["time_s"])] = state
    return result, raw


def parity(field: np.ndarray, columns=20) -> float:
    sign = (-1.) ** np.arange(field.shape[0])
    return float(np.sqrt(np.mean((sign @ field[:, :columns] / field.shape[0]) ** 2)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, baseline_raw = load(BASE)
    candidate, candidate_raw = load(CANDIDATE)
    times = [60., 120., 180., 240., 300.]
    metrics = {"times_s": times, "baseline": {}, "candidate": {}}
    scales = dict(bed=1000., depth=1000., velocity=1000., transport=1.e6)
    units = dict(bed="mm", depth="mm", velocity="mm/s", transport="mm2/s")
    for label, data in (("baseline", baseline), ("candidate", candidate)):
        for field, scale in scales.items():
            metrics[label][field + "_transverse_parity_" + units[field].replace("/", "_")] = [
                parity(data[time][field]) * scale for time in times
            ]
        last = data[300.]
        metrics[label].update(
            bed_change_min_mm=float(last["bed"].min() * 1000),
            bed_change_max_mm=float(last["bed"].max() * 1000),
            mean_bed_change_first_1m_mm=float(last["bed"][:, :20].mean() * 1000),
            positive_bed_volume_m3=float(np.maximum(last["bed"], 0.).sum() * .0025),
            negative_bed_volume_m3=float(np.minimum(last["bed"], 0.).sum() * .0025),
        )
    baseline_mean = baseline[300.]["bed"].mean(axis=0)
    candidate_mean = candidate[300.]["bed"].mean(axis=0)
    mean_difference = candidate_mean - baseline_mean
    metrics["mean_longitudinal_bed_difference_300s"] = {
        "rms_mm": float(np.sqrt(np.mean(mean_difference ** 2)) * 1000),
        "max_abs_mm": float(np.max(np.abs(mean_difference)) * 1000),
        "first_1m_rms_mm": float(np.sqrt(np.mean(mean_difference[:20] ** 2)) * 1000),
    }
    metrics["reduction_at_300s_percent"] = {
        field: 100. * (
            1. - parity(candidate[300.][field]) / parity(baseline[300.][field])
        )
        for field in scales
    }
    # The continuation must reproduce its source checkpoint exactly at 60 s.
    source60, source_raw = load(ROOT / "validation/_transverse230/candidate_run60/run/frames")
    metrics["restart_60s_state_max_abs"] = float(
        np.max(np.abs(source_raw[60.] - candidate_raw[60.]))
    )
    diagnostics = json.loads((
        ROOT / "validation/_transverse230/candidate_restart300/run/DIAGNOSTICS.json"
    ).read_text(encoding="utf8"))[-1]
    metrics["candidate_300s_conservation"] = {
        "solid_residual_m3": diagnostics["solid_residual_m3"],
        "water_plus_pore_residual_m3": diagnostics["water_plus_pore_residual_m3"],
    }
    (OUT / "METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_edges = np.arange(21) * .05
    y_edges = np.arange(31) * .05
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for row, (label, data) in enumerate((("current solver", baseline), ("2-D compact candidate", candidate))):
        values = data[300.]
        panels = (
            (values["bed"][:, :20] * 1000, "bed change (mm)", "RdBu_r"),
            (values["velocity"][:, :20], "streamwise velocity (m/s)", "viridis"),
            (values["transport"][:, :20] * 1e6, "bedload magnitude (mm²/s)", "viridis"),
        )
        for column, (field, title, cmap) in enumerate(panels):
            kwargs = {}
            if cmap == "RdBu_r":
                limit = max(
                    np.max(np.abs(baseline[300.]["bed"][:, :20] * 1000)),
                    np.max(np.abs(candidate[300.]["bed"][:, :20] * 1000)),
                )
                kwargs = dict(vmin=-limit, vmax=limit)
            image = axes[row, column].pcolormesh(
                x_edges, y_edges, field, cmap=cmap, shading="flat", **kwargs
            )
            axes[row, column].set(xlim=(0, 1), ylim=(0, 1.5), ylabel=label)
            if row == 0:
                axes[row, column].set_title(title)
            if row == 1:
                axes[row, column].set_xlabel("x (m)")
            fig.colorbar(image, ax=axes[row, column], shrink=.78)
    fig.suptitle("Raw-cell Jiang–Cao comparison at 300 s")
    fig.savefig(OUT / "RAW_TRANSVERSE_COMPARISON_300S.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, field, scale, title in (
        (axes[0, 0], "bed", 1000., "bed parity (mm)"),
        (axes[0, 1], "depth", 1000., "depth parity (mm)"),
        (axes[1, 0], "velocity", 1000., "velocity parity (mm/s)"),
        (axes[1, 1], "transport", 1.e6, "bedload parity (mm²/s)"),
    ):
        for label, data in (("current", baseline), ("candidate", candidate)):
            ax.semilogy(times, [parity(data[t][field]) * scale for t in times], "o-", label=label)
        ax.set(xlabel="time (s)", ylabel=title)
        ax.grid(True)
        ax.legend()
    fig.suptitle("Transverse one-cell mode growth")
    fig.savefig(OUT / "TRANSVERSE_PARITY_EVOLUTION.png", dpi=180)
    plt.close(fig)

    x = .025 + .05 * np.arange(500)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    axes[0].plot(x, baseline_mean * 1000, label="current")
    axes[0].plot(x, candidate_mean * 1000, label="candidate")
    axes[0].set(xlim=(0, 2), ylabel="mean bed change (mm)")
    axes[0].grid(True)
    axes[0].legend()
    section = 8
    y = .025 + .05 * np.arange(30)
    axes[1].plot(y, baseline[300.]["bed"][:, section] * 1000, "o-", label="current")
    axes[1].plot(y, candidate[300.]["bed"][:, section] * 1000, "o-", label="candidate")
    axes[1].set(xlabel="y (m)", ylabel="bed change at x=0.425 m (mm)")
    axes[1].grid(True)
    axes[1].legend()
    fig.suptitle("Retained mean morphology and removed row oscillation")
    fig.savefig(OUT / "MEAN_AND_TRANSVERSE_BED_300S.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
