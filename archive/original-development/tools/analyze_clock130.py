"""All-output accuracy screen of isolated clock candidates; no speed claims."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_inlet129 import load_run, compare_runs


def counters(run, until):
    rows=[r for r in run["qualification"]["steps"] if r["time_s"] <= until+1e-10]
    data=[r["diagnostics"] for r in rows]
    return dict(outer_transactions=len(rows),
        hydraulic_substeps=sum(r["hydraulic_substeps"] for r in data),
        retries=sum(len(r["hydraulic_rejections"]) for r in data),
        remainders=sum(r["hydraulic_endpoint_remainders"] for r in data),
        fwave_evaluations=sum(r["fwave_evaluations"] for r in data),
        accepted_cfl=max(r["accepted_cfl_max"] for r in data),
        mean_hydraulic_dt_s=until/sum(r["hydraulic_substeps"] for r in data))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--matrix",type=Path,required=True)
    p.add_argument("--reference",type=Path,default=ROOT/"validation/_inlet129/natural5m_final/5m_critical_cpu")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    reference=load_run(a.reference)
    rows={};runs={}
    preferred=["carry","near_limit","combined","outer1","cap100_outer1"]
    children=sorted(a.matrix.iterdir(),key=lambda p: preferred.index(p.name) if p.name in preferred else len(preferred))
    for child in children:
        if not (child/"QUALIFICATION.json").exists():continue
        run=load_run(child);runs[child.name]=run
        until=run["summary"]["status"]["time_s"]
        # The reference lasts100s, while these bounded tests stop60s. Compare
        # every candidate time and ignore only the explicit later reference tail.
        pair=compare_runs(reference,run)
        rows[child.name]=dict(experiment=json.loads((child/"EXPERIMENT.json").read_text()),
            comparison=pair,counters=counters(run,until),reference_counters=counters(reference,until),
            budget=run["summary"]["budget"], discharge=run["summary"]["discharge"],
            minimum_saved_depth_m=run["summary"]["min_depth_m"],
            screen_5mm=pair["skipped_times"]==0 and pair["depth_max_m"] is not None and pair["depth_max_m"]<=.005,
            timings_qualified=False)
    result=dict(scope="isolated_clock_accuracy_only",reference=str(a.reference),cases=rows,
        warning="Concurrent jobs contaminate walltimes. No speedup is inferred. CFL overrides are experiment identities, not the public project cap.")
    (a.output/"CLOCK_COMPARISON.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    if runs:
        end=max(run["summary"]["status"]["time_s"] for run in runs.values())
        base_steps=[s for s in reference["qualification"]["steps"] if s["time_s"]<=end+1e-10]
        axs[0,1].plot([s["time_s"] for s in base_steps],
            [s["diagnostics"]["hydraulic_next_substep_hint_s"] for s in base_steps],
            color="black",linestyle="--",label="baseline")
    for name,run in runs.items():
        d=rows[name];pairs=[r for r in d["comparison"]["per_time"] if r["status"]=="compared"]
        axs[0,0].plot([r["time_s"] for r in pairs],[1000*r["depth_max_m"] for r in pairs],marker=".",label=name)
        steps=run["qualification"]["steps"]
        axs[0,1].plot([s["time_s"] for s in steps],[s["diagnostics"]["hydraulic_next_substep_hint_s"] for s in steps],label=name)
        axs[1,0].plot([f["row"]["time_s"] for f in run["frames"]],[len(f["mesh"]["area"]) for f in run["frames"]],marker=".",label=name)
    names=list(rows)
    if names:
        base=rows[names[0]]["reference_counters"]
        names=["baseline"]+names
        counts=[base]+[rows[n]["counters"] for n in names[1:]]
        xpos=np.arange(len(names))
        axs[1,1].bar(xpos-.18,[c["hydraulic_substeps"] for c in counts],.36,label="Accepted hydraulic steps")
        axs[1,1].bar(xpos+.18,[c["retries"] for c in counts],.36,label="Rejected CFL trials")
        axs[1,1].set_xticks(xpos,names,rotation=25,ha="right")
    axs[0,0].set(title="Maximum depth difference over identical meshes",ylabel="mm",xlabel="Simulation time (s)")
    axs[0,1].set(title="Unclipped next hydraulic dt",ylabel="s",xlabel="Simulation time (s)")
    axs[1,0].set(title="AMR cells at every saved output",ylabel="Cells",xlabel="Simulation time (s)")
    axs[1,1].set(title="Work counts over 60 s (not a walltime benchmark)",ylabel="Count")
    for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(a.output/"CLOCK_COMPARISON.png",dpi=180);plt.close(fig)
    for name,d in rows.items():
        print(json.dumps(dict(name=name,counts=d["counters"],
            max_depth_mm=1000*d["comparison"]["depth_max_m"],
            rms_depth_mm=1000*d["comparison"]["depth_rms_m"],
            compared=d["comparison"]["compared_times"],skipped=d["comparison"]["skipped_times"],
            pass_5mm=d["screen_5mm"])))


if __name__=="__main__":main()
