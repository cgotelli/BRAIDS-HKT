"""Diagnose the transverse row striping in the 300 s Jiang--Cao gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
FRAMES = PROJECT / "runs/experiment229_gate_300s/frames"
OUT = ROOT / "validation/_transverse230"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = json.loads((FRAMES / "INDEX.json").read_text(encoding="utf8"))
    row = index["frames"][-1]
    with np.load(FRAMES / row["epoch"]) as mesh:
        vertices, offsets = mesh["vertices"], mesh["offsets"]
    cx = np.array([(vertices[offsets[i]:offsets[i+1], 0].min() +
                    vertices[offsets[i]:offsets[i+1], 0].max()) / 2
                   for i in range(len(offsets)-1)])
    cy = np.array([(vertices[offsets[i]:offsets[i+1], 1].min() +
                    vertices[offsets[i]:offsets[i+1], 1].max()) / 2
                   for i in range(len(offsets)-1)])
    ix = np.rint((cx-.025)/.05).astype(int)
    iy = np.rint((cy-.025)/.05).astype(int)
    def grid(value):
        result = np.empty((30, 500)); result[iy, ix] = value; return result
    with np.load(FRAMES / row["file"]) as frame:
        initial = grid(frame["initial_bed_m"])
        change = grid(frame["z"] - frame["initial_bed_m"])
        transport = grid(np.hypot(*frame["bedload_vector_m2_s"]))
        depth = grid(frame["q"][0])
        velocity = grid(frame["q"][1] / frame["q"][0])
    x = .025 + .05*np.arange(500); y = .025 + .05*np.arange(30)
    design = np.column_stack((np.ones(initial.size),
                              np.broadcast_to(x, initial.shape).ravel(),
                              np.broadcast_to(y[:, None], initial.shape).ravel()))
    plane = (design @ np.linalg.lstsq(design, initial.ravel(), rcond=None)[0]).reshape(initial.shape)
    perturb = initial-plane
    region = np.s_[:, :20]
    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    parity_y = (-1.)**np.arange(30)
    parity_x = (-1.)**np.arange(500)
    metrics = {
        "time_s": float(row["time_s"]),
        "initial_perturbation_max_mm": float(abs(perturb).max()*1000),
        "initial_perturbation_rms_mm": float(np.sqrt(np.mean(perturb**2))*1000),
        "correlation_change_vs_initial_perturbation_first_1m": corr(change[region], perturb[region]),
        "correlation_transport_vs_initial_perturbation_first_1m": corr(transport[region], perturb[region]),
        "correlation_depth_vs_initial_perturbation_first_1m": corr(depth[region], perturb[region]),
        "correlation_velocity_vs_initial_perturbation_first_1m": corr(velocity[region], perturb[region]),
        "transverse_parity_projection_change_mm":
            float(np.sqrt(np.mean((parity_y @ change[:, :20] / 30.)**2))*1000),
        "longitudinal_parity_projection_change_mm":
            float(np.sqrt(np.mean((change[:, :20] @ parity_x[:20] / 20.)**2))*1000),
        "change_rms_first_1m_mm": float(np.sqrt(np.mean(change[region]**2))*1000),
        "row_mean_change_first_1m_mm": (change[:, :20].mean(axis=1)*1000).tolist(),
        "row_mean_initial_perturbation_first_1m_mm":
            (perturb[:, :20].mean(axis=1)*1000).tolist(),
    }
    (OUT / "AUDIT.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), layout="constrained")
    for ax, field, title, cmap in (
        (axes[0], perturb*1000, "Initial bed perturbation after fitted plane", "RdBu_r"),
        (axes[1], change*1000, "Bed change at 300 s", "RdBu_r"),
        (axes[2], transport*1e6, "Transport at 300 s", "viridis"),
        (axes[3], velocity, "Streamwise velocity at 300 s", "viridis")):
        lim = np.max(np.abs(field[:, :20])) if cmap == "RdBu_r" else None
        image = ax.pcolormesh(np.arange(21)*.05, np.arange(31)*.05, field[:, :20],
                              cmap=cmap, vmin=-lim if lim else None,
                              vmax=lim if lim else None, shading="flat")
        ax.set(title=title, ylabel="y (m)", xlim=(0, 1), ylim=(0, 1.5))
        fig.colorbar(image, ax=ax, shrink=.75)
    axes[-1].set_xlabel("x (m)")
    fig.savefig(OUT / "TRANSVERSE_SOURCE_AUDIT.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
