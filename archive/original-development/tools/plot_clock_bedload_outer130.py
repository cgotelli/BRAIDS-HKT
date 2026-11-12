"""Plot already-completed outer-transaction bedload checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument("--folder",type=Path,required=True);a=p.parse_args()
    data=json.loads((a.folder/"SUMMARY.json").read_text())
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for backend,style in (("cpu","-"),("gpu","--")):
        for outer,color in (("outer2","#20639b"),("outer05","#e07a18")):
            rows=data["comparisons"][backend+"_"+outer+"__fine"]["per_time"]
            label=f"{backend.upper()}, outer {2 if outer=='outer2' else .5:g} s"
            axs[0].plot([r["time_s"] for r in rows],[1000*r["depth_max_m"] for r in rows],style,color=color,label=label)
            axs[1].plot([r["time_s"] for r in rows],[1000*r["bed_max_m"] for r in rows],style,color=color,label=label)
    names=["outer2","outer05","outer0125"]
    cases=[data["cases"]["cpu_"+name] for name in names];x=np.arange(3)
    axs[2].bar(x-.18,[c["status"]["accepted_steps"] for c in cases],.36,label="Physical outer updates")
    axs[2].bar(x+.18,[c["status"]["hydraulic"]["accepted_substeps"] for c in cases],.36,label="Hydraulic SSPRK22 steps")
    axs[2].set_xticks(x,["2 s","0.5 s","0.125 s"])
    axs[0].set(title="Depth difference against outer 0.125 s",ylabel="Maximum difference (mm)",xlabel="Simulation time (s)")
    axs[1].set(title="Bed difference against outer 0.125 s",ylabel="Maximum difference (mm)",xlabel="Simulation time (s)")
    axs[2].set(title="Actual work over 5 s\n(CPU/GPU counts match)",ylabel="Count",xlabel="Requested maximum outer timestep")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle("Fixed mesh, one-grain bedload; actual 2 s outer transactions, unchanged hard CFL 0.5",fontsize=12)
    fig.savefig(a.folder/"BEDLOAD_OUTER.png",dpi=180);plt.close(fig)


if __name__=="__main__":main()
