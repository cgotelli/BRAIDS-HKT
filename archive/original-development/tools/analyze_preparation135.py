"""Analyze completed same-input alpha134/135 preparation/run ABBA evidence.

Read-only with respect to runs, source runtimes and user projects. This tool
never launches preparation or a solver. Run only after COMPARISON.json exists.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_execution134 import compare_runs


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def statistics_of(values):
    values=list(map(float,values))
    if not values or not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid timing sample')
    return dict(samples=len(values),values=values,mean=statistics.mean(values),median=statistics.median(values),
                minimum=min(values),maximum=max(values))


def conservation(row):
    if not row['conservation_pass']:raise ValueError('Reported conservation check failed')
    path=Path(row['run'])/'DIAGNOSTICS.json';records=read(path)
    if not records:raise ValueError('Missing conservation records')
    maximum=0.;largest_ratio=0.;max_class=0.
    for record in records:
        value=abs(record['water_residual_m3']);limit=record['water_roundoff_limit_m3']
        if not math.isfinite(value) or value>limit:raise ValueError('Water conservation failed')
        maximum=max(maximum,value);largest_ratio=max(largest_ratio,value/limit if limit else 0.)
        for residual in record['class_residual_kg']:
            if not math.isfinite(residual) or abs(residual)>record['class_roundoff_limit_kg']:
                raise ValueError('Sediment conservation failed')
            max_class=max(max_class,abs(residual))
        if any(record['wall_integrals']):raise ValueError('Impermeable wall has a nonzero integral')
    if not math.isclose(maximum,row['maximum_water_residual_m3'],rel_tol=1e-12,abs_tol=1e-20):
        raise ValueError('Reported water residual differs')
    return dict(source=str(path.resolve()),sha256=sha(path),records=len(records),
                maximum_water_residual_m3=maximum,maximum_water_limit_ratio=largest_ratio,
                maximum_class_residual_kg=max_class,impermeable_wall_integrals_exactly_zero=True)


def field_check(first,second,limit,*,scope=None):
    result=compare_runs(first,second,scope=scope or 'Same input alpha134/135, exact bilinear surface with changed summation. Record roundoff differences; not a byte-identical prepared-field claim.')
    left=read(Path(first)/'frames/INDEX.json');right=read(Path(second)/'frames/INDEX.json')
    if [r['time_s'] for r in left['frames']]!=[r['time_s'] for r in right['frames']]:
        raise ValueError('Output times changed')
    if not result['all_common_geometry_exact'] or not result['hydraulic_counts_equal']:
        raise ValueError('Geometry or hydraulic counts changed')
    for frame in result['rows']:
        for name,difference in frame['arrays'].items():
            if difference['exact']:continue
            error=difference.get('maximum_abs')
            if (not difference['shape_equal'] or not difference['dtype_equal'] or
                error is None or not math.isfinite(error) or error>limit):
                raise ValueError(f'Published field {name} changed beyond roundoff screen')
    physical_keys=('depth_maximum_abs_m','depth_rms_m','bed_maximum_abs_m',
                   'momentum_maximum_abs_m2_s','velocity_maximum_abs_m_s')
    result['physical_maxima_over_common_frames']={key:max((r['physical'].get(key,0.) for r in result['rows']),default=0.) for key in physical_keys}
    result['absolute_numeric_field_limit']=limit
    return result


def analyze(root,*,expected_limit=1e-7,field_scope=None):
    benchmark_path=root/'BENCHMARK.json';receipt_path=root/'COMPARISON.json'
    benchmark=read(benchmark_path);receipt=read(receipt_path);groups=defaultdict(list)
    limit=receipt['absolute_field_limit']
    if limit!=expected_limit:raise ValueError('Unexpected qualification limit; review before analysis')
    for row in benchmark['rows']:
        if row['status']['status']!='complete':raise ValueError('Incomplete benchmark run')
        groups[row['case']].append(row)
    if set(groups)!=set(receipt['summary']):raise ValueError('Comparison and timing case inventories differ')
    if len(receipt['comparisons'])!=3*len(groups):raise ValueError('Incomplete comparison receipt')
    metrics=('preparation_process_s','core_wall_s','run_process_s','total_process_s')
    cases={}
    for name,rows in groups.items():
        rows.sort(key=lambda r:r['index'])
        if [r['index'] for r in rows]!=[0,1,2,3] or [r['variant'] for r in rows]!=['before','after','after','before']:
            raise ValueError('Expected two samples per variant in complete ABBA order')
        if len({r['source_sha256'] for r in rows})!=1 or len({r['source'] for r in rows})!=1:
            raise ValueError('Before/after project input differs')
        if sha(rows[0]['source'])!=rows[0]['source_sha256']:raise ValueError('Project changed after benchmark')
        checks=[r for r in receipt['comparisons'] if r['case']==name]
        if sorted((r['index'],r['variant']) for r in checks)!=[(1,'after'),(2,'after'),(3,'before')]:
            raise ValueError('Comparison receipt does not cover the ABBA rows')
        if not all(r['roundoff_bound_pass'] and r['same_counts'] for r in checks):
            raise ValueError('Reported field or hydraulic check failed')
        variants={}
        for variant in ('before','after'):
            vr=[r for r in rows if r['variant']==variant]
            for identity in ('runtime','runtime_manifest_sha256'):
                if len({r[identity] for r in vr})!=1:raise ValueError('Runtime identity changed during repeat')
            for row in vr:
                if not math.isclose(row['total_process_s'],row['preparation_process_s']+row['run_process_s'],abs_tol=1e-9):
                    raise ValueError('End-to-end timing is not the measured sum')
                if row['core_wall_s']>row['run_process_s']:raise ValueError('Worker time exceeds run process')
            value={key:statistics_of(row[key] for row in vr) for key in metrics}
            value.update(outside_worker_s=statistics_of(r['run_process_s']-r['core_wall_s'] for r in vr),
                runtime=vr[0]['runtime'],runtime_manifest_sha256=vr[0]['runtime_manifest_sha256'],
                runs=[r['run'] for r in vr],conservation=[conservation(r) for r in vr])
            for key in metrics:
                if not math.isclose(value[key]['median'],receipt['summary'][name][variant][key],rel_tol=1e-12):
                    raise ValueError('Reported timing median differs from raw ABBA observations')
            variants[variant]=value
        improvements={}
        for key in metrics:
            before=variants['before'][key];after=variants['after'][key]
            improvements[key]=dict(before_median_s=before['median'],after_median_s=after['median'],
                speedup=before['median']/after['median'],reduction_percent=100*(1-after['median']/before['median']),
                mean_reduction_percent=100*(1-after['mean']/before['mean']))
        fields=field_check(rows[0]['run'],rows[1]['run'],limit,scope=field_scope)
        cases[name]=dict(source=rows[0]['source'],source_sha256=rows[0]['source_sha256'],
            variants=variants,improvement=improvements,receipt_comparisons=checks,independent_fields=fields,
            budgets_byte_identical=all(r['same_budgets'] for r in checks),
            fields_byte_identical=all(r['exact'] for r in checks))
    return dict(schema='braids.preparation135.analysis.v1',analysis_sha256=sha(__file__),
        benchmark=str(benchmark_path.resolve()),benchmark_sha256=sha(benchmark_path),
        comparison=str(receipt_path.resolve()),comparison_sha256=sha(receipt_path),scope=benchmark['scope'],cases=cases,
        interpretation=['Two independently prepared fresh-process observations per variant, in ABBA order. Means, medians and observed ranges are descriptive, not confidence intervals.',
            'Preparation plus run uses each actual paired total before aggregating; worker time is nested within run-process time and must not be added again.',
            'Both variants use the same v20 project. Terrain arithmetic may differ at roundoff; geometry and hydraulic counts remain exact, and original conservation limits are checked independently.',
            'These full preparation/run timings, not isolated terrain-kernel gains or previous cold-start outliers, govern the release-level performance claim.'])


def figure(summary,out,*,release_labels=('Alpha134','Alpha135'),title='Same-input alpha134 / alpha135 preparation and execution'):
    names=list(summary['cases']);x=np.arange(len(names));fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    entries=(('preparation_process_s','Fresh preparation'),('core_wall_s','Worker execution'),
             ('run_process_s','Fresh run process (includes worker)'),('total_process_s','Preparation + run: end-to-end'))
    for ax,(metric,panel_title) in zip(axes.flat,entries):
        top=0.
        for offset,variant,label,color in [(-.19,'before',release_labels[0],'#a66d4b'),(.19,'after',release_labels[1],'#276b9c')]:
            values=[summary['cases'][name]['variants'][variant][metric] for name in names]
            m=np.array([r['median'] for r in values]);lo=np.array([r['minimum'] for r in values]);hi=np.array([r['maximum'] for r in values])
            top=max(top,float(max(hi)));bars=ax.bar(x+offset,m,.36,yerr=[m-lo,hi-m],capsize=3,label=label,color=color)
            for bar,value,upper in zip(bars,m,hi):
                ax.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,upper),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
        for i,name in enumerate(names):
            change=summary['cases'][name]['improvement'][metric]
            reduction=change['reduction_percent'];direction='less' if reduction>=0 else 'more'
            ax.text(i,.98,f"{change['speedup']:.2f}x / {abs(reduction):.1f}% {direction}",ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=10)
        ax.set(xticks=x,xticklabels=['CPU8' if n.startswith('cpu8') else 'GPU' if n.startswith('gpu') else n for n in names],
               ylabel='Wall seconds',ylim=(0,top*1.28),title=panel_title)
        ax.grid(axis='y',alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=11)
    fig.suptitle(title+'\n60 simulated seconds; 1 s result interval; clean ABBA medians and observed ranges',fontsize=14)
    fig.savefig(out/'PREPARATION_RELEASE_SPEEDUP.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark-root',type=Path,default=ROOT/'validation/_preparation135/release_abba')
    parser.add_argument('--output',type=Path,default=ROOT/'validation/_preparation135/analysis')
    args=parser.parse_args();summary=analyze(args.benchmark_root)
    args.output.mkdir(parents=True,exist_ok=True);figure(summary,args.output)
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:case['improvement'] for name,case in summary['cases'].items()},indent=2))


if __name__=='__main__':main()
