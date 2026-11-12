"""All-output qualification of the bounded 2D predictor/cap follow-up.

The original five-millimetre fine-reference screen is kept unchanged.  This
script never changes runtime, input, old summaries, or interpretation of the
failed cap1/cap1.5 experiments.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_2d_cfl130 import compare, frames, read


def checked_compare(a, b):
    if len(a) != len(b):
        raise ValueError("Output frame counts differ")
    return compare(a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("validation/_speed130/closed2d_v2"))
    args = parser.parse_args()
    out = args.input.resolve()
    refroot = args.reference.resolve()
    runs = {p.parent.name: read(p) for p in out.glob("*/RESEARCH.json")}
    cache = {name: frames(out/name) for name in runs}
    references = {f"{case}_{cap:g}": frames(refroot/f"{case}_cpu_cfl{cap:g}")
                  for case in ("dam", "lake") for cap in (.125, .5, 1.)}
    result = dict(cases={}, comparisons={}, all_outputs_compared=True,
        unchanged_acceptance_screen="Maximum raw all-output depth difference <= 0.005 m against cap0.125",
        no_new_stability_theorem=True,
        timing_scope="Numerical qualification only; no timing claim",
        reference_root=str(refroot))
    for name, run in runs.items():
        history = run["history"]
        initial = np.load(out/name/"INITIAL_OVERRIDE.npz")["q"]
        result["cases"][name] = dict(case=run["case"], backend=run["backend"],
            research=run["research"], source_runtime=run["source_runtime"],
            substeps=run["substeps"], rejections=len(run["rejections"]),
            rejection_kinds=sorted({row["kind"] for row in run["rejections"]}),
            minimum_depth_m=min(row["minimum_depth_m"] for row in history),
            max_water_residual_m3=max(abs(row["water_residual_m3"]) for row in history),
            max_wall_ledger_m3=max(row["wall_integral"] for row in history),
            minimum_cell_area_m2=min(row["minimum_cell_area_m2"] for row in history),
            cropped_cells=history[0]["cropped_cells"],
            all_output_lake_drift_q_components=(np.max([
                np.max(abs(state["q"]-initial), axis=1) for _, _, state in cache[name]
                ], axis=0).tolist() if run["case"]=="lake" else None),
            all_output_wetfront=history)
        for cap, label in [(.125,"fine"),(.5,"baseline"),(1.,"failed_cap1")]:
            comparison = checked_compare(cache[name],references[f"{run['case']}_{cap:g}"])
            result["comparisons"][name+"__"+label] = comparison
        if run["backend"]=="gpu":
            pair = name.replace("_gpu_","_cpu_")
            if pair not in cache:
                raise ValueError(f"Missing CPU pair: {pair}")
            result["comparisons"][name+"__CPU"] = checked_compare(cache[name],cache[pair])
    (out/"SUMMARY.json").write_text(json.dumps(result,indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    for name, run in runs.items():
        if run["case"]!="dam":
            continue
        rows=result["comparisons"][name+"__fine"]["all_outputs"]
        label=name.removeprefix("dam_").replace("_", " ")
        line="-" if run["backend"]=="cpu" else ":"
        axes[0].plot([r["time_s"] for r in rows], [1000*r["depth_max_m"] for r in rows],line,label=label)
        axes[1].plot([r["time_s"] for r in rows], [r["wet_symmetric_area_m2"]["0.001"] for r in rows],line,label=label)
    axes[0].axhline(5.,color="k",ls="--",lw=1.,label="Unchanged 5 mm gate")
    axes[0].set_ylabel("Maximum depth difference vs cap0.125 (mm)")
    axes[1].set_ylabel("Symmetric wet area at 1 mm threshold (m²)")
    for ax in axes:
        ax.set_xlabel("Physical time (s)")
        ax.set_ylim(bottom=0.)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Strong diagonal dry dam with cropped closed walls — raw all-output checks")
    fig.tight_layout()
    fig.savefig(out/"CLOSED_2D_CLOCK.png",dpi=160)
    plt.close(fig)
    for name, row in result["comparisons"].items():
        if name.endswith("__fine") or name.endswith("__CPU"):
            print(name,dict(depth_mm=1000*row["depth_max_m"],rms_mm=1000*row["depth_rms_max_m"],
                           pass5mm=row["depth_screen5mm"]),flush=True)


if __name__=="__main__":
    main()
