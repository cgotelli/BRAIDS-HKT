"""Summarize unchanged-input packaged AMR timings and non-overlapping profiles."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--extra-benchmark',type=Path,action='append',default=[])
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'validation/_amr132')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    bench=json.loads(a.benchmark.read_text())
    for extra in a.extra_benchmark:
        bench['records'].extend(json.loads(extra.read_text())['records'])
    result={}
    for case in dict.fromkeys(r['case'] for r in bench['records']):
        rows={v:[r for r in bench['records'] if r['case']==case and r['variant']==v]
              for v in ('baseline','candidate')}
        values={v:{key:dict(mean=statistics.mean(r[key] for r in rr),
                           samples=[r[key] for r in rr])
                   for key in ('worker_s','preparation_s','process_s')} for v,rr in rows.items()}
        result[case]=dict(timing=values,
            reduction_percent={key:100*(1-values['candidate'][key]['mean']/values['baseline'][key]['mean'])
                               for key in ('worker_s','process_s')},
            all_candidate_fields_exact=all(r['exact'] is True for r in rows['candidate']),
            hydraulic=rows['candidate'][0]['hydraulic'])
        # Preserve all observations, but do not attribute an anomalous launcher
        # delay outside both timed phases to the solver optimization.
        outside={v:[r['process_s']-r['worker_s']-r['preparation_s'] for r in rr]
                 for v,rr in rows.items()}
        result[case]['outside_timed_phases_s']=outside
        if max(x for group in outside.values() for x in group)>10.:
            result[case]['unadjusted_process_reduction_percent']=result[case]['reduction_percent']['process_s']
            result[case]['reduction_percent']['process_s']=None
            result[case]['process_timing_warning']='Large delay outside timed phases; no pooled process-speed claim. All raw samples retained.'
    profiles=[json.loads(path.read_text()) for path in (a.before,a.after)]
    summary=dict(cases=result,profiles=[dict(worker_s=r['worker_wall_s'],phases=r['phases'],calls=r['calls'])
                                      for r in profiles],
                 benchmark=str(a.benchmark.resolve()),
                 extra_benchmarks=[str(p.resolve()) for p in a.extra_benchmark],
                 profile_paths=[str(p.resolve()) for p in (a.before,a.after)])
    (out/'RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    titles={'cpu5m':'CPU: 5 m dynamic AMR\n60 simulated seconds',
            'gpu5m':'GPU: 5 m dynamic AMR\n60 simulated seconds',
            'gpu2m':'GPU: 2 m static mesh\n100 simulated seconds'}
    fig,axes=plt.subplots(1,len(result),figsize=(12,4.8),layout='constrained',squeeze=False)
    for ax,(case,row) in zip(axes.flat,result.items()):
        values=[row['timing'][v]['worker_s']['mean'] for v in ('baseline','candidate')]
        bars=ax.bar(['Alpha131','Alpha132'],values,color=['#718096','#167d9a'],width=.6)
        for i,v in enumerate(('baseline','candidate')):
            ax.scatter([i]*len(row['timing'][v]['worker_s']['samples']),
                       row['timing'][v]['worker_s']['samples'],color='#152536',s=18,zorder=3)
            ax.text(i,max(row['timing'][v]['worker_s']['samples'])+.35,
                    f'{values[i]:.2f} s',ha='center',va='bottom')
        ax.set_ylim(0,max(row['timing'][v]['worker_s']['samples'][j]
                          for v in ('baseline','candidate')
                          for j in range(len(row['timing'][v]['worker_s']['samples'])))*1.25)
        ax.set_title(titles[case]);ax.set_ylabel('Worker wall time (s)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Unchanged CFL 1, physics and AMR schedule — sequential packaged runs')
    fig.savefig(out/'AMR_SPEED.png',dpi=180);plt.close(fig)
    keys=['load','construction','amr','output','checkpoint','step','other']
    labels=['Load','Initial construction','AMR','Output','Checkpoint','Hydraulic stepping','Other']
    colors=['#cbd5e1','#64748b','#e87722','#a78bfa','#6d28d9','#167d9a','#94a3b8']
    fig,ax=plt.subplots(figsize=(11,4.5));left=[0.,0.]
    for key,label,color in zip(keys,labels,colors):
        values=[r['phases'].get(key,0.) for r in profiles]
        ax.barh([1,0],values,left=left,label=label,color=color,height=.45)
        if key in ('amr','step'):
            for i,y in enumerate([1,0]):
                ax.text(left[i]+values[i]/2,y,f'{values[i]:.2f} s',ha='center',va='center',color='white',fontsize=10)
        left=[x+y for x,y in zip(left,values)]
    ax.set_yticks([1,0],['Alpha131','Alpha132']);ax.set_xlim(0,max(left)*1.04)
    ax.set_xlabel('Instrumented worker time (s); phases do not overlap')
    ax.set_title('CPU 5 m dynamic AMR: where time is spent',loc='left',weight='bold')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=4,frameon=False)
    fig.text(.01,.015,'Profiling includes instrumentation overhead. Use the clean timing figure for speed claims.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(out/'AMR_PROFILE.png',dpi=180);plt.close(fig)
    print(json.dumps({case:row['reduction_percent'] for case,row in result.items()},indent=2))


if __name__=='__main__':main()
