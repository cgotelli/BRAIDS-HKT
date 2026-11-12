"""Scientific figures separating checkout and installed-path timing evidence."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'validation/_speed130'


def read(path):
    return json.loads(path.read_text())


def main():
    checkout=read(ROOT/'clean_execution_and_selected_v1/BENCHMARK.json')
    initial=read(ROOT/'installed_profile_and_clean_v1/BENCHMARK.json')
    repeated=read(ROOT/'installed_clean_repeats_v2/BENCHMARK.json')
    installed={}
    for case in ('cpu5m','gpu2m'):
        # Initial CPU samples preceded the external scan. Initial GPU timings
        # have a possible overlapping file scan and are intentionally excluded.
        eligible=(initial['records']+repeated['records']) if case=='cpu5m' else repeated['records']
        installed[case]={}
        for variant in ('baseline','conservative'):
            rows=[r for r in eligible if r['clean'] and r['case']==case and r['variant']==variant]
            installed[case][variant]={key:float(np.mean([r[key] for r in rows]))
                for key in ('worker_s','preparation_s','process_s')}
            installed[case][variant]['samples']=[dict(worker_s=r['worker_s'],preparation_s=r['preparation_s'],process_s=r['process_s']) for r in rows]
        a,b=(installed[case][v] for v in ('baseline','conservative'))
        installed[case]['worker_reduction_percent']=100*(1-b['worker_s']/a['worker_s'])
        installed[case]['worker_speedup']=a['worker_s']/b['worker_s']
        installed[case]['whole_process_reduction_percent']=100*(1-b['process_s']/a['process_s'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    labels=['CPU, 5 m dynamic\n60 simulation seconds','GPU, 2 m fixed\n100 simulation seconds']
    x=np.arange(2);width=.36
    for ax,context in zip(axes,('checkout','installed')):
        if context=='checkout':
            before=[checkout['summary'][c]['mean_worker_s']['none'] for c in ('cpu5m','gpu2m')]
            after=[checkout['summary'][c]['mean_worker_s']['both'] for c in ('cpu5m','gpu2m')]
            title='Checkout: exact implementation prototypes'
            note='2 samples each (ABBA); compiler pinning + mesh reuse'
        else:
            before=[installed[c]['baseline']['worker_s'] for c in ('cpu5m','gpu2m')]
            after=[installed[c]['conservative']['worker_s'] for c in ('cpu5m','gpu2m')]
            title='Installed: proposed conservative settings'
            note='CPU 2 samples; GPU 1 clean repeat; no compiler caching'
        a=ax.bar(x-width/2,before,width,color='#707070',label='alpha129')
        b=ax.bar(x+width/2,after,width,color='#0072B2',label='Candidate')
        ax.bar_label(a,fmt='%.2f s',fontsize=9,padding=3)
        ax.bar_label(b,fmt='%.2f s',fontsize=9,padding=3)
        ax.set_xticks(x,labels,fontsize=9);ax.set_ylabel('Worker wall time (s)')
        ax.set_ylim(0,max(before+after)*1.20);ax.set_title(title,fontsize=11)
        ax.text(.5,-.24,note,transform=ax.transAxes,ha='center',fontsize=8)
        ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Checkout compiler gains are not installed-plugin gains',fontsize=13)
    fig.text(.02,.015,'Installed candidate: mesh reuse + CFL predictor 0.98 + AMR hint carry; cap remains 0.5; outer intervals 1 s CPU / 2 s GPU.\n'
        'Different meshes and durations: the panels do not compare CPU versus GPU throughput. Preparation excluded; recorded separately.',fontsize=8)
    fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(ROOT/'CLEAN_EXECUTION_SPEED.png',dpi=180);plt.close(fig)
    folders=[('Checkout CPU, dynamic','profile_cpu5m'),('Installed CPU, dynamic','installed_profile_and_clean_v1/cpu5m_baseline_instrumented'),('Checkout GPU, fixed','profile_gpu2m')]
    phases=['step','amr','construction','load','output','checkpoint','other']
    labels=['Hydraulic/physical stepping','AMR + owner rebuilding','Initial owner construction','Load','Output + geometry','Checkpoint','Other']
    colors=['#0072B2','#D55E00','#E69F00','#CC79A7','#009E73','#56B4E9','#888888']
    fig,ax=plt.subplots(figsize=(11,5.1));left=np.zeros(3)
    phases_json={}
    for key,label,color in zip(phases,labels,colors):
        values=np.array([read(ROOT/folder/'PROFILE.json')['phases'].get(key,0) for _,folder in folders])
        ax.barh(range(3),values,left=left,label=label,color=color,height=.55)
        for i,value in enumerate(values):
            if value>1.4:ax.text(left[i]+value/2,i,f'{value:.1f}',ha='center',va='center',color='white',fontsize=9)
        left+=values
    for name,folder in folders:
        p=read(ROOT/folder/'PROFILE.json');phases_json[name]=dict(worker_s=p['worker_wall_s'],phases_s=p['phases'])
    ax.set_yticks(range(3),[name for name,_ in folders]);ax.set_xlabel('Instrumented worker wall time (s)')
    ax.set_title('Installed CPU remains dominated by rebuilding adaptive mesh owners')
    ax.legend(loc='upper center',bbox_to_anchor=(.42,-.18),ncol=3,frameon=False,fontsize=8)
    fig.text(.02,.015,'Attribution includes cProfile/wrapping overhead; not used for clean speedup claims. Different cases are labelled explicitly.',fontsize=8)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(ROOT/'INSTALLED_EXECUTION_PHASES.png',dpi=180);plt.close(fig)
    (ROOT/'INSTALLED_EXECUTION_FINDINGS.json').write_text(json.dumps(dict(clean=installed,phases=phases_json,
        scope='Numerical comparisons and conservation are in each run COMPARISONS.json/BENCHMARK.json. Initial GPU timings excluded due disclosed possible file-scan overlap. Maintained installation unchanged.',
        excluded_changes='The subsequently implemented maintained exporter proc.wait(timeout) optimization is not present in these frozen installed tests. No measured CPU speedup is attributed to it.'),indent=2)+'\n')


if __name__=='__main__':
    main()
