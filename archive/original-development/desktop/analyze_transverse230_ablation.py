"""Compare transverse-mode growth in the isolated 60 s ablations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent / "BRAIDS_JiangCao_Experiment228"
OUT = ROOT / "validation/_transverse230/ablations"
RUNS = {
    "full bundle": PROJECT / "runs/experiment229_gate_300s/frames",
    "no momentum feedback": OUT / "runs/no_secondary_momentum_60/frames",
    "no momentum or steering": OUT / "runs/no_secondary_all_60/frames",
    "face-visible bed slope": ROOT / "validation/_transverse230/face_slope/run60/frames",
    "2-D compact capacity": ROOT / "validation/_transverse230/candidate_run60/run/frames",
}


def load(folder: Path) -> dict[str, np.ndarray]:
    index = json.loads((folder / "INDEX.json").read_text(encoding="utf8"))
    row = min(index["frames"], key=lambda item: abs(float(item["time_s"]) - 60.0))
    if abs(float(row["time_s"]) - 60.0) > 1e-8:
        raise ValueError(f"Exact 60 s frame missing in {folder}")
    with np.load(folder / row["epoch"]) as mesh:
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
    ix = np.rint((cx - .025) / .05).astype(int)
    iy = np.rint((cy - .025) / .05).astype(int)

    def grid(values):
        result = np.empty((30, 500))
        result[iy, ix] = values
        return result

    with np.load(folder / row["file"]) as frame:
        h = grid(frame["q"][0])
        u = grid(frame["q"][1] / frame["q"][0])
        bed = grid(frame["z"] - frame["initial_bed_m"])
        transport = grid(np.hypot(*frame["bedload_vector_m2_s"]))
    return dict(depth=h, velocity=u, bed=bed, transport=transport)


def parity(field: np.ndarray) -> float:
    sign = (-1.) ** np.arange(field.shape[0])
    return float(np.sqrt(np.mean((sign @ field[:, :20] / field.shape[0]) ** 2)))


def main() -> None:
    data = {name: load(folder) for name, folder in RUNS.items()}
    metrics = {}
    for name, values in data.items():
        metrics[name] = {
            "bed_transverse_parity_mm": parity(values["bed"]) * 1000,
            "depth_transverse_parity_mm": parity(values["depth"]) * 1000,
            "velocity_transverse_parity_mm_s": parity(values["velocity"]) * 1000,
            "transport_transverse_parity_mm2_s": parity(values["transport"]) * 1e6,
            "bed_change_min_mm": float(values["bed"].min() * 1000),
            "bed_change_max_mm": float(values["bed"].max() * 1000),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(data), 3, figsize=(13, 8), layout="constrained")
    for row, (name, values) in enumerate(data.items()):
        fields = (
            (values["bed"][:, :20] * 1000, "bed change (mm)", "RdBu_r"),
            (values["velocity"][:, :20], "streamwise velocity (m/s)", "viridis"),
            (values["transport"][:, :20] * 1e6, "bedload magnitude (mm²/s)", "viridis"),
        )
        for column, (field, title, cmap) in enumerate(fields):
            kwargs = {}
            if cmap == "RdBu_r":
                limit = float(np.max(np.abs(field)))
                kwargs = dict(vmin=-limit, vmax=limit)
            image = axes[row, column].pcolormesh(
                np.arange(21) * .05,
                np.arange(31) * .05,
                field,
                shading="flat",
                cmap=cmap,
                **kwargs,
            )
            axes[row, column].set(xlim=(0, 1), ylim=(0, 1.5), ylabel=name)
            if row == 0:
                axes[row, column].set_title(title)
            if row == len(data) - 1:
                axes[row, column].set_xlabel("x (m)")
            fig.colorbar(image, ax=axes[row, column], shrink=.75)
    fig.suptitle("Jiang–Cao transverse-mode ablation at 60 s")
    fig.savefig(OUT / "TRANSVERSE_ABLATION_60S.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
