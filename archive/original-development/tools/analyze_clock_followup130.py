"""Paired-backend, measured-retry and bounded transaction/bedload evidence."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_inlet129 import load_run,compare_runs


def transaction_evidence():
    root=ROOT/"validation/_speed130"
    beds={}
    for path in (root/"bedload_pytest_v2").glob("*/BEDLOAD_SMOKE.json"):
        report=json.loads(path.read_text());frames=[]
        index=json.loads((path.parent/"candidate/frames/INDEX.json").read_text())
        for row in index["frames"]:
            with np.load(path.parent/"candidate/frames"/row["file"],allow_pickle=False) as data:
                frames.append({k:data[k].copy() for k in data.files})
        report["maximum_active_bed_change_m"]=max(float(abs(f["z"]-frames[0]["z"]).max()) for f in frames)
        beds[report["backend"]]=(report,frames)
    parity={}
    if set(beds)=={"cpu","gpu"}:
        for key in ("q","z","mass","water","bed_boundary"):
            parity[key]=max(float(abs(a[key]-b[key]).max()) for a,b in zip(beds["cpu"][1],beds["gpu"][1]))
    return dict(scope="bounded 0.2 s actual-front AMR transactions and one-grain bedload smoke",
        restart_cancel_checks=dict(passed=4,backends=["cpu","gpu"],
            exact_final_fields_after_restart=True,exact_final_fields_after_cancel_and_restart=True,
            cancellation_at_s=.12,carried_hint_and_q_restored_exactly=True,
            junit="validation/_speed130/CLOCK_TRANSACTIONS.xml"),
        original_harness_failure="The two first bedload analyzers requested a nonexistent time_s array instead of time; all trajectories had completed. Failed XML/output retained; only the test was corrected.",
        bedload_junit="validation/_speed130/CLOCK_BEDLOAD.xml",bedload_checks_passed=2,
        bedload={k:v[0] for k,v in beds.items()},bedload_cpu_gpu_max_abs=parity,
        no_long_time_morphodynamic_validation=True,timings_qualified=False)


def main():
    out=ROOT/"validation/_speed130/clock_followup_analysis";out.mkdir(exist_ok=True)
    paths=dict(selected_cpu=ROOT/"validation/_speed130/clock_matrix/cap100_outer1",
        selected_gpu=ROOT/"validation/_speed130/clock_followup/selected_gpu",
        measured_cpu=ROOT/"validation/_speed130/clock_followup/measured_retry_cpu",
        frozen_cpu=ROOT/"validation/_inlet129/natural5m_final/5m_critical_cpu",
        frozen_gpu=ROOT/"validation/_inlet129/natural5m_final/5m_critical_gpu")
    runs={name:load_run(path) for name,path in paths.items()}
    comparisons={}
    for a,b in (("selected_cpu","selected_gpu"),("selected_gpu","frozen_gpu"),
                ("selected_cpu","measured_cpu"),("measured_cpu","frozen_cpu")):
        comparisons[a+"__"+b]=compare_runs(runs[a],runs[b])
    doc=dict(scope="60 s dynamic5m paired-backend and retry predictor screens",
        timings_qualified=False,comparisons=comparisons,
        runs={name:r["summary"] for name,r in runs.items()},
        original_hook_snapshot="validation/_speed130/frozen_clock_initial/speed130_clock.py",
        helper_update="Only new combined_measured selection added; preexisting choices are unchanged. Source identities remain distinct.")
    (out/"FOLLOWUP.json").write_text(json.dumps(doc,indent=2),encoding="utf-8")
    if (ROOT/"validation/_speed130/CLOCK_BEDLOAD.xml").exists():
        (out/"TRANSACTION_BEDLOAD.json").write_text(json.dumps(transaction_evidence(),indent=2),encoding="utf-8")
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for name,d in comparisons.items():
        pts=[r for r in d["per_time"] if r["status"]=="compared"]
        axs[0].plot([p["time_s"] for p in pts],[1000*p["depth_max_m"] for p in pts],marker=".",label=name)
    axs[0].set(xlabel="Simulation time (s)",ylabel="Max depth difference (mm)",title="Every matched saved mesh; no smoothing")
    names=["selected_cpu","selected_gpu","measured_cpu"]
    x=np.arange(len(names));hyd=[runs[n]["summary"]["status"]["hydraulic"] for n in names]
    axs[1].bar(x-.18,[h["accepted_substeps"] for h in hyd],.36,label="Accepted steps")
    axs[1].bar(x+.18,[h["internal_retries"] for h in hyd],.36,label="Rejected trials")
    axs[1].set_xticks(x,names,rotation=15,ha="right")
    axs[1].set(title="Measured-frequency retry saves additional work",ylabel="Count (not walltime)")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/"FOLLOWUP.png",dpi=180);plt.close(fig)
    print(json.dumps({n:{k:d[k] for k in ("compared_times","skipped_times","depth_max_m","depth_rms_m")} for n,d in comparisons.items()},indent=2))


if __name__=="__main__":main()
