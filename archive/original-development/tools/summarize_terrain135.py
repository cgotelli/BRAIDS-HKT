"""Summarize completed terrain-only ABBA controls; never launch a solver."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'validation/_performance135')
    args=parser.parse_args();paths=[args.root/f'terrain_{n}/BENCHMARK.json' for n in (1,8,32)]
    docs=[json.loads(path.read_text()) for path in paths]
    if len({d['project_sha256'] for d in docs})!=1 or len({d['dem_sha256'] for d in docs})!=1:
        raise ValueError('Terrain comparisons require identical project and DEM inputs')
    if len({json.dumps(d['native_sources'],sort_keys=True) for d in docs})!=1:
        raise ValueError('Native terrain sources changed across thread controls')
    arrays=[]
    for path,doc in zip(paths,docs):
        if len(doc['rows'])!=6:raise ValueError('Expected first calls plus one complete ABBA cycle')
        with np.load(path.with_name('AVERAGES.npz'),allow_pickle=False) as data:
            candidate=data['warm_abba_1_after'].copy();repeat=data['warm_abba_2_after']
            if not np.array_equal(candidate,repeat):raise ValueError('Native repeated result differs')
            arrays.append(candidate)
    thread_exact=all(np.array_equal(arrays[0],value) for value in arrays[1:])
    if not thread_exact:raise ValueError('Native thread-count results differ')
    rows=[]
    for path,doc in zip(paths,docs):
        rows.append(dict(threads=doc['requested_candidate_threads'],source=str(path.resolve()),
            before=doc['summary']['before'],after=doc['summary']['after'],speedup=doc['warm_speedup'],
            maximum_bed_difference_m=max(r['maximum_abs_difference_m'] for r in doc['rows']),
            first_call_s={r['variant']:r['wall_s'] for r in doc['rows'] if r['phase']=='first_call'}))
    summary=dict(schema='braids.terrain135.summary.v1',rows=rows,native_thread_results_exact=thread_exact,
        note='Two warm observations per variant in each independent ABBA; ranges are not confidence intervals. Cold/first calls retained separately. This isolates terrain integration, not whole preparation or solver speed. The small workload does not show a measured additional multicore gain.')
    (args.root/'TERRAIN_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');x=np.arange(3)
    for offset,variant,label,color in [(-.18,'before','Original Python/OGR','#a66d4b'),(.18,'after','Native batched','#276b9c')]:
        v=[r[variant] for r in rows];m=np.array([r['median_s'] for r in v]);lo=np.array([r['minimum_s'] for r in v]);hi=np.array([r['maximum_s'] for r in v])
        bars=axes[0].bar(x+offset,m,.34,yerr=[m-lo,hi-m],capsize=3,label=label,color=color)
        for bar,value,upper in zip(bars,m,hi):
            axes[0].annotate(f'{value:.3f}',(bar.get_x()+bar.get_width()/2,upper),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set(ylabel='Terrain integration wall seconds',title='Same mesh and DEM; whole integral call')
    axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.12);axes[0].legend(fontsize=9)
    v=[r['after'] for r in rows];m=np.array([r['median_s'] for r in v]);lo=np.array([r['minimum_s'] for r in v]);hi=np.array([r['maximum_s'] for r in v])
    axes[1].bar(x,m,.55,yerr=[m-lo,hi-m],capsize=3,color='#276b9c')
    for i,(value,upper) in enumerate(zip(m,hi)):
        axes[1].annotate(f'{value:.3f} s',(i,upper),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
    axes[1].set(ylabel='Native integration wall seconds',title='Native detail: threading not the main saving',ylim=(0,max(hi)*1.35))
    for ax in axes:
        ax.set(xticks=x,xticklabels=['1','8','32'],xlabel='Requested native preparation threads');ax.grid(axis='y',alpha=.2)
    fig.suptitle(f"Exact terrain integration: {docs[0]['mesh_cells']:,} cells\nClean sequential ABBA; medians and observed ranges; no hydraulic solver",fontsize=13)
    fig.savefig(args.root/'TERRAIN_SPEEDUP.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
