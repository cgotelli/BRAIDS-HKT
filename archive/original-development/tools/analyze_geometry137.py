"""Summarize measured geometry reuse, raw runtimes and historical test deltas."""
import argparse
import json
from pathlib import Path
import pstats
import statistics
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def timings(path):
    data=pstats.Stats(str(path)).stats
    def cost(file, function):
        rows=[v for (f,line,name),v in data.items() if Path(f).name==file and name==function]
        return dict(calls=sum(v[1] for v in rows),inclusive_s=sum(v[3] for v in rows))
    return {name:cost(file,fn) for name,file,fn in (
        ('polygon_topology','river_polygon_banks.py','topology'),
        ('cell_order','river_amr_transfer.py','polygons'),
        ('local_integrations','river_polygon_banks.py','_local_geometry'),
        ('leaf_controller','river_leaf_controller.py','__init__'),
        ('hierarchy_selection','amr.py','select'))}


def tests(path):
    tree=ET.parse(path)
    return {c.attrib['classname']+'.'+c.attrib['name']:
        'failed' if c.find('failure') is not None or c.find('error') is not None else
        'skipped' if c.find('skipped') is not None else 'passed'
        for c in tree.iter('testcase')}


def analyze(root,out):
    raw=json.loads((root/'abba_final/BENCHMARK.json').read_text())
    strict=json.loads((root/'abba_final/STRICT_COMPARISON.json').read_text())
    if not strict['all_pass']:raise ValueError('Strict comparison failed')
    result=dict(cases={},profiles={},qualification={})
    for case in sorted({r['case'] for r in raw['rows']}):
        rows=[r for r in raw['rows'] if r['case']==case]
        metrics={}
        for key in ('preparation_process_s','run_process_s','total_process_s'):
            sample={v:[r[key] for r in rows if r['variant']==v] for v in ('before','after')}
            median={v:statistics.median(a) for v,a in sample.items()}
            metrics[key]=dict(samples=sample,median=median,
                reduction_percent=100*(1-median['after']/median['before']))
        result['cases'][case]=metrics
    for backend in ('cpu','gpu'):
        result['profiles'][backend]={v:timings(root/f'{v}_{backend}'/'EXECUTION.pstats')
                                     for v in ('before','after')}
    baseline=tests(root/'HISTORICAL_BASELINE.xml');candidate=tests(root/'COUPLED.xml')
    result['qualification']['historical_same_outcomes']=baseline==candidate
    result['qualification']['historical_failures']=[k for k,v in candidate.items() if v=='failed']
    current={}
    for name in ('UNIT_LOCAL.xml','STAGED_FINAL.xml','DESKTOP_FINAL.xml'):
        if (root/name).exists():
            run=tests(root/name);current.update(run)
            result['qualification'][name]={v:list(run.values()).count(v) for v in ('passed','failed','skipped')}
    result['qualification']['current_unique']={v:list(current.values()).count(v) for v in ('passed','failed','skipped')}
    result['strict_fields_pass']=strict['all_pass']
    out.mkdir(parents=True,exist_ok=True)
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=np.arange(2);width=.34;colors=['#7e8c98','#007f8b']
    for k,v in enumerate(('before','after')):
        top=[result['profiles'][b][v]['polygon_topology']['inclusive_s'] for b in ('cpu','gpu')]
        order=[result['profiles'][b][v]['cell_order']['inclusive_s'] for b in ('cpu','gpu')]
        axes[0].bar(x+(k-.5)*width,top,width,color=colors[k],label=v.title()+' topology')
        axes[0].bar(x+(k-.5)*width,order,width,bottom=top,color=colors[k],hatch='///',alpha=.65,label=v.title()+' order check')
    axes[0].set(xticks=x,xticklabels=['CPU8','GPU'],ylabel='Seconds (instrumented)',title='Polygon setup and cell-order validation')
    axes[0].set_ylim(0,1.3)
    axes[0].legend(fontsize=8,ncol=2,loc='upper center')
    for k,v in enumerate(('before','after')):
        values=[result['cases'][case]['total_process_s']['median'][v] for case in sorted(result['cases'])]
        bars=axes[1].bar(x+(k-.5)*width,values,width,color=colors[k],label=v.title())
        axes[1].bar_label(bars,fmt='%.3f s',fontsize=9)
    axes[1].set(xticks=x,xticklabels=['CPU8','GPU'],ylabel='Seconds (clean median)',title='Complete preparation plus 60 s simulation')
    axes[1].legend()
    fig.suptitle('Geometry reuse only — unchanged fields, timesteps and physical methods')
    fig.savefig(out/'GEOMETRY_PERFORMANCE.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'validation/_geometry137')
    p.add_argument('--output',type=Path,default=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_geometry137')
    a=p.parse_args();analyze(a.root,a.output)
