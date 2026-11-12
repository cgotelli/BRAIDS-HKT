"""Measure complete speed and physical differences separately."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_inlet129 import load_run, compare_runs


def main():
    out=ROOT/'validation/_amr131';bench=json.loads((out/'clean_final/BENCHMARK.json').read_text())
    results={}
    for case in ('cpu5m','gpu2m'):
        records=[r for r in bench['records'] if r['case']==case]
        rows={v:[r for r in records if r['variant']==v] for v in ('baseline','candidate','cfl1')}
        timing={v:{key:float(np.mean([r[key] for r in rr])) for key in ('worker_s','preparation_s','process_s')} for v,rr in rows.items()}
        for v in timing:
            timing[v]['worker_reduction_percent']=100*(1-timing[v]['worker_s']/timing['baseline']['worker_s'])
        candidate=load_run(out/'clean_final'/f'{case}_4_cfl1')
        baseline=load_run(out/'clean_final'/f'{case}_0_baseline')
        fine=ROOT/'validation/_inlet129'/('natural5m_final_fine/5m_critical_cpu' if case=='cpu5m' else 'natural2m_final_fine/2m_critical_gpu')
        comparisons={'previous_CFL05':compare_runs(candidate,baseline)}
        if fine.exists():comparisons['fine']=compare_runs(candidate,load_run(fine))
        results[case]=dict(timing=timing,comparisons=comparisons,budget=candidate['summary']['budget'],
            hydraulic={v:rr[0]['hydraulic'] for v,rr in rows.items()},
            implementation_fields_exact=all(r['exact'] for r in records if r['variant']!='cfl1'))
    (out/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,case,title in zip(axes,results,['CPU: dynamic AMR, 5 m, 60 s simulated','GPU: static mesh, 2 m, 100 s simulated']):
        vals=[results[case]['timing'][v]['worker_s'] for v in ('baseline','candidate','cfl1')]
        bars=ax.bar(['Previous\nCFL 0.5','Optimized\nCFL 0.5','Optimized\nCFL 1'],vals,color=['#73808c','#287eaa','#ef9f30'])
        ax.bar_label(bars,labels=[f'{v:.2f} s' for v in vals],padding=3)
        ax.set_ylim(0,max(vals)*1.22);ax.set_ylabel('Worker wall time (s)');ax.set_title(title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Measured compiler-free execution: implementation gains and CFL gains separated')
    fig.savefig(out/'AMR_SPEED.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v['timing'] for k,v in results.items()},indent=2))


if __name__=='__main__':main()
