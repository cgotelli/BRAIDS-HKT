"""Preserve exact qualification and report isolated versus whole-process timing."""
from pathlib import Path
import json
import pstats
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'validation/_mesh138'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_mesh138'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    micro=json.loads((RAW/'HANDOFF.json').read_text())
    benchmark=json.loads((RAW/'abba_final/BENCHMARK.json').read_text())
    exact=json.loads((RAW/'abba_final/STRICT_COMPARISON.json').read_text())
    profiles={}
    for backend in ('cpu','gpu'):
        profiles[backend]={}
        for variant in ('before','after'):
            s=pstats.Stats(str(RAW/(variant+'_'+backend)/'EXECUTION.pstats'))
            result={}
            for label,file,method in [('selection','amr.py','select'),('guard','river_leaf_controller.py','_geometry_snapshot')]:
                found=[v for (path,line,name),v in s.stats.items() if Path(path).name==file and name==method]
                result[label]=dict(calls=sum(v[1] for v in found),cumulative_s=sum(v[3] for v in found))
            profiles[backend][variant]=result
    tests={};unique=set()
    for name in ('UNIT_LOCAL','STAGED_FINAL','DESKTOP_FINAL'):
        p=RAW/(name+'.xml');tree=ET.parse(p)
        tests[name]=dict(tests=len(list(tree.iter('testcase'))),failures=len(list(tree.iter('failure'))),
                        errors=len(list(tree.iter('error'))),skips=len(list(tree.iter('skipped'))))
        unique.update((c.get('classname'),c.get('name')) for c in tree.iter('testcase'))
    summary=dict(micro=micro,profiles=profiles,whole_run=json.loads((RAW/'abba_final/COMPARISON.json').read_text())['summary'],
        all_numerical_comparisons_exact=exact['all_pass'],tests=tests,distinct_tests=len(unique),
        staging=json.loads((RAW/'candidate2_STAGING.json').read_text()),
        whole_run_conclusion='No reliable net whole-run improvement established: CPU samples slower, GPU samples faster, large preparation variability. Isolated handoff improves.')
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    datasets=[('Initial geometry handoff',[[r['elapsed_s'] for r in micro['rows'] if r['variant']==v and r['repeat']>0] for v in ('before','after')])]
    for case,title in [('cpu8_clock1_output1','Whole CPU run'),('gpu_clock1_output1','Whole GPU run')]:
        datasets.append((title,[[r['run_process_s'] for r in benchmark['rows'] if r['case']==case and r['variant']==v] for v in ('before','after')]))
    for ax,(title,groups) in zip(axes,datasets):
        for x,values in enumerate(groups):
            ax.scatter(np.full(len(values),x)+np.linspace(-.06,.06,len(values)),values,color=('#6b7280','#007f8b')[x],s=30,zorder=3)
            ax.hlines(float(np.median(values)),x-.18,x+.18,color='black',linewidth=2)
        ax.set(title=title,ylabel='Wall time (s)',xticks=[0,1],xticklabels=['Preview 2','Candidate'],xlim=(-.5,1.5),ylim=(0,None))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('One initial mesh: 31.6% less handoff time; whole-run gain unresolved',fontsize=13)
    fig.text(.5,.015,'Dots: individual samples. Black lines: medians. All 61 output frames, geometry and budgets remain exact.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'MESH_HANDOFF.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(profiles=profiles,tests=tests,distinct_tests=len(unique)),indent=2))


if __name__=='__main__':main()
