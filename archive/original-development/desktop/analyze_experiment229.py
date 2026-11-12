"""Raw-field morphology assessment for the preview50 Jiang--Cao experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
RUN = PROJECT / "runs/experiment229_gate_300s"
FRAMES = RUN / "frames"
REPORT = (ROOT / "validation/sediment/river_morphodynamics/reports/"
          "desktop_experiment225/timestep229")
FEED_PER_WIDTH = (0.00286 / 2650.0) / 1.5


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def mesh_and_order(index: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epoch = FRAMES / index["frames"][0]["epoch"]
    with np.load(epoch) as mesh:
        vertices = mesh["vertices"]
        offsets = mesh["offsets"]
        area = mesh["area"].copy()
    cx = np.array([(vertices[offsets[i]:offsets[i + 1], 0].min() +
                    vertices[offsets[i]:offsets[i + 1], 0].max()) / 2
                   for i in range(len(offsets) - 1)])
    cy = np.array([(vertices[offsets[i]:offsets[i + 1], 1].min() +
                    vertices[offsets[i]:offsets[i + 1], 1].max()) / 2
                   for i in range(len(offsets) - 1)])
    ix = np.rint((cx - .025) / .05).astype(int)
    iy = np.rint((cy - .025) / .05).astype(int)
    if len(ix) != 15000 or len(set(zip(ix, iy))) != 15000:
        raise ValueError("The milestone diagnostic requires the fixed 500 x 30 grid")
    return ix, iy, area


def load(row: dict, ix: np.ndarray, iy: np.ndarray) -> dict:
    def grid(values: np.ndarray) -> np.ndarray:
        result = np.empty((30, 500), dtype=float)
        result[iy, ix] = values
        return result
    with np.load(FRAMES / row["file"]) as frame:
        return {
            "time_s": float(row["time_s"]),
            "steps": int(row["steps"]),
            "bed": grid(frame["z"] - frame["initial_bed_m"]),
            "depth": grid(frame["q"][0]),
            "mobile": grid(frame["mobile_bedload_solid_m"]),
            "capacity": grid(frame["bedload_capacity_m2_s"]),
            "transport": grid(np.hypot(*frame["bedload_vector_m2_s"])),
        }


def front_x(field: np.ndarray, threshold: float) -> float:
    active = np.flatnonzero(np.max(field, axis=0) > threshold)
    return float(.025 + .05 * active[-1]) if len(active) else 0.0


def metrics(data: dict) -> dict:
    bed = data["bed"]
    width_mean = bed.mean(axis=0)
    # Exclude the first two columns where prescribed feed enters.  The
    # three-point residual detects one-cell longitudinal energy without
    # subtracting any transverse signal or smoothing the displayed fields.
    window = width_mean[2:120]
    high = window[1:-1] - .5 * (window[:-2] + window[2:])
    transport_mean = data["transport"].mean(axis=0)
    meaningful = transport_mean > FEED_PER_WIDTH * .01
    transverse_cv = np.zeros_like(transport_mean)
    transverse_cv[meaningful] = (data["transport"].std(axis=0)[meaningful] /
                                 transport_mean[meaningful])
    return {
        "time_s": data["time_s"],
        "steps": data["steps"],
        "accepted_dt_mean_s": (data["time_s"] / data["steps"]
                                if data["steps"] else None),
        "bed_min_mm": float(bed.min() * 1000),
        "bed_max_mm": float(bed.max() * 1000),
        "positive_bed_volume_m3": float(np.maximum(bed, 0).sum() * .0025),
        "negative_bed_volume_m3": float(np.minimum(bed, 0).sum() * .0025),
        "three_point_high_frequency_rms_first_6m_mm":
            float(np.sqrt(np.mean(high * high)) * 1000),
        "adjacent_width_mean_rms_first_6m_mm":
            float(np.sqrt(np.mean(np.diff(window) ** 2)) * 1000),
        "transverse_bed_std_max_first_6m_mm":
            float(bed[:, :120].std(axis=0).max() * 1000),
        "transverse_bed_std_mean_first_6m_mm":
            float(bed[:, :120].std(axis=0).mean() * 1000),
        "transport_front_x_m_1pct_feed": front_x(data["transport"], FEED_PER_WIDTH * .01),
        "capacity_front_x_m_1pct_feed": front_x(data["capacity"], FEED_PER_WIDTH * .01),
        "mobile_store_front_x_m_1nm": front_x(data["mobile"], 1e-9),
        "transport_transverse_cv_median_active":
            float(np.median(transverse_cv[meaningful])) if np.any(meaningful) else 0.0,
        "transport_transverse_cv_max_active":
            float(np.max(transverse_cv[meaningful])) if np.any(meaningful) else 0.0,
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    index = json.loads((FRAMES / "INDEX.json").read_text(encoding="utf8"))
    if abs(index["frames"][-1]["time_s"] - 300.0) > 1e-9:
        raise ValueError("The sealed 300 s morphology gate is required")
    ix, iy, _ = mesh_and_order(index)
    data = [load(row, ix, iy) for row in index["frames"]]
    result = {
        "run": str(RUN),
        "runtime_manifest_sha256":
            "63b330057b70c20f8066ebeded4b113495fec82e97c734b5bf97a9a9973572f9",
        "controls": {"cfl": .5, "maximum_outer_timestep_s": .03},
        "feed_solid_per_width_m2_s": FEED_PER_WIDTH,
        "frames": [metrics(item) for item in data],
        "conservation": json.loads((RUN / "DIAGNOSTICS.json").read_text(encoding="utf8")),
        "display": "raw cell values; no smoothing or interpolation",
    }
    save(REPORT / "METRICS.json", result)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = .025 + .05 * np.arange(500)
    y = .025 + .05 * np.arange(30)
    edges_x = np.arange(501) * .05
    edges_y = np.arange(31) * .05
    chosen = [min(data, key=lambda item: abs(item["time_s"] - target))
              for target in (60., 180., 300.)]
    vmax = max(np.percentile(np.abs(item["bed"][:, :200]) * 1000, 99.5)
               for item in chosen)
    vmax = max(vmax, .01)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.8), layout="constrained")
    for ax, item in zip(axes, chosen):
        image = ax.pcolormesh(edges_x, edges_y, item["bed"] * 1000,
                              cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        ax.set(title=f'Raw bed change at {item["time_s"]:g} s', xlim=(0, 10),
               ylim=(0, 1.5), ylabel="y (m)")
    axes[-1].set_xlabel("x (m)")
    fig.colorbar(image, ax=axes, label="bed change (mm)", shrink=.82)
    fig.savefig(REPORT / "RAW_BED_EVOLUTION.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout="constrained")
    for item in data:
        if item["time_s"] == 0:
            continue
        color = plt.cm.viridis(item["time_s"] / 300.)
        axes[0].plot(x, item["bed"].mean(axis=0) * 1000, color=color,
                     alpha=.55, lw=.8)
        axes[1].plot(x, item["bed"].std(axis=0) * 1000, color=color,
                     alpha=.55, lw=.8)
    axes[0].axhline(0, color=".6", lw=.7)
    axes[0].set(title="Cross-width mean bed change (raw frames)", ylabel="bed change (mm)",
                xlim=(0, 10))
    axes[1].set(title="Transverse bed variability", xlabel="x (m)",
                ylabel="transverse std. (mm)", xlim=(0, 10))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 300))
    fig.colorbar(sm, ax=axes, label="simulation time (s)", shrink=.85)
    fig.savefig(REPORT / "LONGITUDINAL_AND_TRANSVERSE_BED.png", dpi=180)
    plt.close(fig)

    final = data[-1]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), layout="constrained")
    image = axes[0].pcolormesh(edges_x, edges_y, final["transport"] * 1e6,
                               cmap="viridis", shading="flat")
    axes[0].set(title="Raw transported bedload magnitude at 300 s", xlim=(0, 10),
                ylim=(0, 1.5), ylabel="y (m)")
    fig.colorbar(image, ax=axes[0], label="transport (mm2/s)")
    axes[1].semilogy(x, np.maximum(final["transport"].mean(axis=0), 1e-14),
                     label="transport")
    axes[1].semilogy(x, np.maximum(final["capacity"].mean(axis=0), 1e-14),
                     label="capacity")
    axes[1].axhline(FEED_PER_WIDTH, color="k", ls="--", lw=.8,
                    label="prescribed feed / width")
    axes[1].set(title="Cross-width means", xlim=(0, 10), xlabel="x (m)",
                ylabel="solid discharge (m2/s)")
    axes[1].legend()
    fig.savefig(REPORT / "TRANSPORT_FRONT_300S.png", dpi=180)
    plt.close(fig)

    frame_metrics = result["frames"]
    times = [row["time_s"] for row in frame_metrics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    axes[0].plot(times, [row["three_point_high_frequency_rms_first_6m_mm"]
                         for row in frame_metrics], marker="o", ms=3)
    axes[0].set(title="One-cell longitudinal energy", xlabel="time (s)",
                ylabel="three-point RMS (mm)")
    for key, label in (("transport_front_x_m_1pct_feed", "transport"),
                       ("capacity_front_x_m_1pct_feed", "capacity"),
                       ("mobile_store_front_x_m_1nm", "mobile store")):
        axes[1].plot(times, [row[key] for row in frame_metrics], marker="o", ms=3,
                     label=label)
    axes[1].set(title="Downstream sediment fronts", xlabel="time (s)",
                ylabel="furthest x (m)")
    axes[1].legend()
    fig.savefig(REPORT / "NUMERICAL_MODE_AND_FRONTS.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
