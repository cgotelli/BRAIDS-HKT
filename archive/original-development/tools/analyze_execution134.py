"""Summarize completed, sequential hydraulic timing and output evidence.

This never launches a solver. Run only while no clean timing job is active:
frame verification and plotting perform real disk reads and CPU work.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'validation/_performance134'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    values=list(map(float,values))
    return dict(samples=len(values),values=values,median=statistics.median(values),
                minimum=min(values),maximum=max(values))


def benchmark(path):
    doc=read(path);groups=defaultdict(list)
    for row in doc['rows']:
        if row['status']['status']!='complete':raise ValueError('Incomplete benchmark run')
        groups[row['case']].append(row)
    result={}
    for name,rows in groups.items():
        rows.sort(key=lambda row:row['repeat'])
        if len({row['repeat'] for row in rows})!=len(rows):raise ValueError('Duplicate repeat')
        if len({row['source_sha256'] for row in rows})!=1:raise ValueError('Changed repeated input')
        prep={row['preparation_process_s'] for row in rows}
        if len(prep)!=1:raise ValueError('Unexpected repeated preparation convention')
        result[name]=dict(repeats=len(rows),source=rows[0]['source'],source_sha256=rows[0]['source_sha256'],
            preparation_process_s=next(iter(prep)),preparation_samples=1,
            core_wall_s=stats(row['core_wall_s'] for row in rows),
            run_process_s=stats(row['run_process_s'] for row in rows),
            outside_core_s=stats(row['process_outside_core_s'] for row in rows),
            frames=[row['frames'] for row in rows],runs=[row['run'] for row in rows],
            hydraulic=[row['status']['hydraulic'] for row in rows],
            outer_transactions=[row['status']['accepted_steps'] for row in rows],
            simulated_end_s=[row['status']['time_s'] for row in rows])
    return dict(source=str(Path(path).resolve()),source_sha256=sha(path),runtime=doc['runtime'],
                runtime_manifest_sha256=doc['runtime_manifest_sha256'],scope=doc['scope'],cases=result,
                note='Run timings are independent fresh-process observations. Each case was prepared only once; the same preparation value is repeated in raw rows and is NOT multiple preparation samples. Ranges are observed ranges, not confidence intervals.')


def arrays_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc' else a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)


def delta(a,b):
    same=arrays_equal(a,b)
    record=dict(exact=bool(same),shape_equal=a.shape==b.shape,dtype_equal=a.dtype==b.dtype)
    if a.shape==b.shape and a.dtype.kind in 'fiuc' and b.dtype.kind in 'fiuc':
        difference=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
        record.update(maximum_abs=float(np.max(np.abs(difference),initial=0.)),
                      rms=float(np.sqrt(np.mean(difference*difference))) if difference.size else 0.)
    return record


class Frames:
    def __init__(self,path):
        self.run=Path(path);self.root=self.run/'frames';self.index=read(self.root/'INDEX.json')
        self.frames={float(row['time_s']):row for row in self.index['frames']}
        self.cache={};self.meshes={}
    def load(self,time):
        if time in self.cache:return self.cache[time]
        row=self.frames[time];frame=self.root/row['file'];mesh=self.root/row['epoch']
        if sha(frame)!=row['sha256'] or sha(mesh)!=self.index['epochs'][row['epoch']]:raise ValueError('Frame or geometry hash differs')
        with np.load(frame,allow_pickle=False) as values:arrays={key:values[key].copy() for key in values.files}
        if row['epoch'] not in self.meshes:
            with np.load(mesh,allow_pickle=False) as values:self.meshes[row['epoch']]={key:values[key].copy() for key in values.files}
        geometry=self.meshes[row['epoch']]
        if str(arrays['mesh_epoch'])!=row['epoch']:raise ValueError('Frame geometry reference differs')
        self.cache[time]=(arrays,geometry)
        return arrays,geometry


def compare_runs(first,second,*,scope):
    a=Frames(first);b=Frames(second);common=sorted(set(a.frames)&set(b.frames))
    if not common:raise ValueError('No shared published times')
    rows=[];all_exact=True;numeric_max={};geometry_exact=True
    for t in common:
        av,ag=a.load(t);bv,bg=b.load(t)
        if set(av)!=set(bv) or set(ag)!=set(bg):raise ValueError('Published array inventory differs')
        geom={k:delta(ag[k],bg[k]) for k in ag}
        gexact=all(row['exact'] for row in geom.values());geometry_exact &= gexact
        values={k:delta(av[k],bv[k]) for k in av};exact=all(row['exact'] for row in values.values())
        all_exact &= exact
        for key,row in values.items():
            if 'maximum_abs' in row:numeric_max[key]=max(numeric_max.get(key,0.),row['maximum_abs'])
        physical={}
        if av['q'].shape==bv['q'].shape and gexact:
            ah=av['q'][0];bh=bv['q'][0];dh=ah-bh
            au=np.divide(av['q'][1:],ah,out=np.zeros_like(av['q'][1:]),where=ah>1e-6)
            bu=np.divide(bv['q'][1:],bh,out=np.zeros_like(bv['q'][1:]),where=bh>1e-6)
            physical=dict(depth_maximum_abs_m=float(np.max(abs(dh),initial=0.)),depth_rms_m=float(np.sqrt(np.mean(dh*dh))),
                momentum_maximum_abs_m2_s=float(np.max(abs(av['q'][1:]-bv['q'][1:]),initial=0.)),
                velocity_maximum_abs_m_s=float(np.max(abs(au-bu),initial=0.)),
                bed_maximum_abs_m=float(np.max(abs(av['z']-bv['z']),initial=0.)),
                first_wet_area_1mm_m2=float(np.sum(av['area'][ah>.001])),
                second_wet_area_1mm_m2=float(np.sum(bv['area'][bh>.001])),
                first_water_volume_m3=float(ah@av['area']),second_water_volume_m3=float(bh@bv['area']))
        rows.append(dict(time_s=t,all_arrays_exact=exact,geometry_exact=gexact,arrays=values,geometry=geom,physical=physical))
    sa=read(a.run/'STATUS.json');sb=read(b.run/'STATUS.json')
    return dict(first=str(a.run.resolve()),second=str(b.run.resolve()),scope=scope,
                common_times_s=common,all_common_arrays_exact=all_exact,all_common_geometry_exact=geometry_exact,
                maximum_difference_by_array=numeric_max,rows=rows,
                hydraulic_counts_equal=all(sa['hydraulic'][k]==sb['hydraulic'][k] for k in ('accepted_substeps','internal_retries','fwave_evaluations')),
                first_hydraulic=sa['hydraulic'],second_hydraulic=sb['hydraulic'],
                first_outer_transactions=sa['accepted_steps'],second_outer_transactions=sb['accepted_steps'])


def comparison_groups(cases):
    result={}
    # Cross-output controls keep all outer transactions at one second. Check
    # every shared time and every published array, not merely final water depth.
    for backend in ('cpu8','gpu'):
        anchor=backend+'_clock1_output1'
        for interval in (10,60):
            candidate=backend+'_clock1_output'+str(interval)
            if anchor in cases and candidate in cases:
                result[candidate]=compare_runs(cases[anchor]['runs'][0],cases[candidate]['runs'][0],
                    scope='Fixed 1 s outer clock; output/checkpoint cadence only. Exact common fields and unchanged hydraulic counts are required for this claim.')
    for name in ('cpu8_clock05_output10','cpu8_clock5_output10'):
        if name in cases:
            result[name]=compare_runs(cases['cpu8_clock1_output10']['runs'][0],cases[name]['runs'][0],
                scope='Different outer transaction clock at the same 10 s output cadence. This changes numerical step partitioning; record q/z/wet-area differences rather than claiming exactness.')
    if 'cpu1_output1' in cases:
        result['cpu_threads']=compare_runs(cases['cpu8_clock1_output1']['runs'][0],cases['cpu1_output1']['runs'][0],
            scope='CPU8 versus CPU1 on identical 1 s effective transactions/output; declared dtmax differs 1 versus 2 s but output caps both to 1 s. Thread-dependent roundoff is reported.')
    if 'cpu8_clock1_output1' in cases and 'gpu_clock1_output1' in cases:
        result['backend']=compare_runs(cases['cpu8_clock1_output1']['runs'][0],cases['gpu_clock1_output1']['runs'][0],
            scope='Same physical configuration and clocks, separate explicit CPU/GPU backends. Report numeric differences without assuming bitwise backend identity.')
    # Repeated fresh processes must themselves reproduce their inputs.
    result['repeat_exact']={}
    for name,case in cases.items():
        if len(case['runs'])>1:
            result['repeat_exact'][name]=compare_runs(case['runs'][0],case['runs'][1],scope='Fresh-process repeated-run reproducibility')
    return result


def phase_figure(output):
    profiles=[read(EVIDENCE/name/'PROFILE.json') for name in ('profile_cpu','profile_gpu')]
    categories=[('step','Hydraulic/transaction stepping','#276b9c'),('checkpoint','Restart checkpoint','#cb6c45'),
        ('output','Viewable frames','#d8a342'),('construction','Owner construction','#6c739c'),
        ('load','Load prepared mesh','#6a9a6b'),('preflight','Preflight','#a9b779'),
        ('other','Other runtime work','#b5b5b5')]
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    left=np.zeros(2)
    values={}
    named={k for k,_,_ in categories if k!='other'}
    for key,label,color in categories:
        v=np.array([p['phases'].get(key,0.) if key!='other' else sum(n for k,n in p['phases'].items() if k not in named) for p in profiles])
        values[key]=v.tolist();ax.barh([0,1],v,left=left,label=label,color=color,height=.5)
        for i,w in enumerate(v):
            if w>.7:ax.text(left[i]+w/2,i,f'{w:.2f}s',ha='center',va='center',fontsize=9,color='white' if key in ('step','checkpoint','construction') else 'black')
        left+=v
    ax.set(yticks=[0,1],yticklabels=['CPU8','GPU'],xlabel='Instrumented public-worker wall seconds',
           title='Where time goes: 60 simulated seconds, 5,624 cells, output every 1 s')
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=9)
    ax.set_ylim(-.5,1.8)
    ax.text(.01,.98,'Non-overlapping phases; profiler overhead included. Clean timings govern speed claims.',transform=ax.transAxes,va='top',fontsize=9)
    fig.savefig(output/'PROFILE_PHASES.png',dpi=160);plt.close(fig)
    return dict(values=values,total_instrumented_worker_s=left.tolist())


def phase_release_figure(output):
    names=('profile_cpu','profile_after_cpu','profile_gpu','profile_after_gpu')
    paths=[EVIDENCE/name/'PROFILE.json' for name in names]
    if not all(path.exists() for path in paths):return None
    profiles=[read(path) for path in paths]
    categories=[('step','Hydraulic/transaction stepping','#276b9c'),('checkpoint','Restart checkpoint','#cb6c45'),
        ('output','Viewable frames','#d8a342'),('construction','Owner construction','#6c739c'),
        ('load','Load prepared mesh','#6a9a6b'),('preflight','Preflight','#a9b779'),
        ('other','Other runtime work','#b5b5b5')]
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');left=np.zeros(4);values={}
    named={key for key,_,_ in categories if key!='other'}
    for key,label,color in categories:
        v=np.array([p['phases'].get(key,0.) if key!='other' else sum(n for k,n in p['phases'].items() if k not in named) for p in profiles])
        values[key]=v.tolist();ax.barh(np.arange(4),v,left=left,label=label,color=color,height=.6)
        for i,w in enumerate(v):
            if w>.65:ax.text(left[i]+w/2,i,f'{w:.2f}',ha='center',va='center',fontsize=9,color='white' if key in ('step','checkpoint','construction') else 'black')
        left+=v
    for i,total in enumerate(left):ax.text(total+.15,i,f'{total:.2f} s',va='center',fontsize=9)
    ax.set(yticks=np.arange(4),yticklabels=['CPU8 before','CPU8 after','GPU before','GPU after'],
           xlabel='Instrumented public-worker wall seconds',xlim=(0,max(left)*1.11),ylim=(3.5,-1.0),
           title='Checkpoint cadence removes repeated persistence cost; results still every 1 s')
    ax.text(.01,.98,'Instrumented phase attribution only; independent clean ABBA determines release speedup.',transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
    fig.savefig(output/'RELEASE_PHASES.png',dpi=160);plt.close(fig)
    return dict(order=names,sources=[str(path.resolve()) for path in paths],
                sha256=[sha(path) for path in paths],values=values,total_instrumented_worker_s=left.tolist(),
                note='Instrumented, separately observed runs. Do not attribute all construction/step timing differences to the I/O optimization, or use these totals instead of clean release ABBA speedup.')


def timing_figures(cases,output):
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,measure,title in zip(axes,('core_wall_s','run_process_s'),('Worker only','Fresh process including startup')):
        for offset,backend,color in [(-.18,'cpu8','#276b9c'),(.18,'gpu','#9f4b80')]:
            names=[backend+'_clock1_output'+str(i) for i in (1,10,60)]
            m=np.array([cases[n][measure]['median'] for n in names]);lo=np.array([cases[n][measure]['minimum'] for n in names]);hi=np.array([cases[n][measure]['maximum'] for n in names])
            bars=ax.bar(np.arange(3)+offset,m,.34,yerr=[m-lo,hi-m],capsize=3,label='CPU8' if backend=='cpu8' else 'GPU',color=color)
            for bar,value,upper in zip(bars,m,hi):ax.text(bar.get_x()+bar.get_width()/2,upper+.25,f'{value:.2f}',ha='center',fontsize=9)
        ax.set(xticks=np.arange(3),xticklabels=['1s','10s','60s'],xlabel='Output interval',ylabel='Wall seconds',title=title)
        ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Output cadence changes cost while the 1 s outer clock stays fixed\nMedians and observed ranges; 60 physical seconds; no profiler',fontsize=12)
    fig.savefig(output/'OUTPUT_FREQUENCY.png',dpi=160);plt.close(fig)

    ordered=[n for n in ('cpu8_clock1_output1','gpu_clock1_output1','cpu8_clock1_output10','gpu_clock1_output10',
                         'cpu8_clock1_output60','gpu_clock1_output60','cpu1_output1','cpu8_clock05_output10','cpu8_clock5_output10') if n in cases]
    fig,ax=plt.subplots(figsize=(11,6.2),layout='constrained');y=np.arange(len(ordered));left=np.zeros(len(ordered))
    for key,label,color in [('preparation_process_s','Preparation (one measurement)','#8da66b'),
                             ('outside_core_s','Startup/preflight/wrapping (median)','#c7a166'),
                             ('core_wall_s','Worker (median)','#276b9c')]:
        values=np.array([cases[n][key] if key=='preparation_process_s' else cases[n][key]['median'] for n in ordered])
        ax.barh(y,values,left=left,label=label,color=color);left+=values
    labels=[n.replace('_clock1_',' / clock1s / ').replace('_output',' / output').replace('_clock',' / clock') for n in ordered]
    ax.set(yticks=y,yticklabels=labels,xlabel='Wall seconds',title='End-to-end cost: preparation + a fresh run process')
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
    ax.set_ylim(len(ordered)+.4,-.8)
    ax.text(.01,.01,'Preparation is not repeated; first cold setup retained, not silently discarded.\nStacked medians summarize stages, not a single measured median total.',transform=ax.transAxes,fontsize=8,va='bottom')
    fig.savefig(output/'SETUP_AND_EXECUTION.png',dpi=160);plt.close(fig)


def timestep_figure(output):
    p=read(EVIDENCE/'profile_cpu/PROFILE.json');audit=read(EVIDENCE/'PREPARATION_CLOCK_AUDIT.json')
    rows=p['steps'];times=np.array([r['time_s'] for r in rows]);d=[r['diagnostics'] for r in rows]
    low=np.array([r['hydraulic_minimum_dt'] for r in d]);high=np.array([r['hydraulic_maximum_dt'] for r in d])
    mean=np.array([r['outer_dt_s']/r['diagnostics']['hydraulic_substeps'] for r in rows])
    irows=audit['iber']['sampled_rows'][1:]
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    ax.fill_between(times,low,high,color='#276b9c',alpha=.18,label='BRAIDS per 1 s interval min–max')
    ax.plot(times,mean,color='#276b9c',label='BRAIDS per 1 s interval accepted mean')
    ax.plot([r['time_s'] for r in irows],[r['reported_dt_s'] for r in irows],'.-',lw=.7,color='#cb6c45',label='Iber logged dt samples (not every step)')
    ax.set(xlabel='Simulated time (s)',ylabel='Hydraulic timestep (s)',ylim=(0,.46),
           title='Comparable cell areas, different explicit methods/CFL definitions\nBRAIDS 3.25 m squares / CFL 1; Iber ≈5 m-sided triangles / CFL 0.45')
    ax.legend(loc='lower right',fontsize=9);ax.grid(alpha=.2)
    fig.savefig(output/'MATCHED_TIMESTEPS.png',dpi=160);plt.close(fig)
    return audit


def candidate_analysis(base,candidate):
    comparison={}
    for name,b in base['cases'].items():
        if name not in candidate['cases']:continue
        c=candidate['cases'][name]
        comparison[name]=dict(worker_speedup=b['core_wall_s']['median']/c['core_wall_s']['median'],
            worker_reduction_percent=100*(1-c['core_wall_s']['median']/b['core_wall_s']['median']),
            process_speedup=b['run_process_s']['median']/c['run_process_s']['median'],
            fields=compare_runs(b['runs'][0],c['runs'][0],scope='Frozen release versus qualified candidate, same named input and cadence'))
    return comparison


def changed_fields(first,second,path=''):
    """Record actual configuration differences, not merely matching case names."""
    if isinstance(first,dict) and isinstance(second,dict):
        result=[]
        for key in sorted(set(first)|set(second)):
            child=path+'.'+key if path else key
            if key not in first:result.append(dict(field=child,change='added',after=second[key]))
            elif key not in second:result.append(dict(field=child,change='removed',before=first[key]))
            else:result.extend(changed_fields(first[key],second[key],child))
        return result
    return [] if first==second else [dict(field=path,change='changed',before=first,after=second)]


def speed_change(before,after):
    return dict(before_median_s=before['median'],after_median_s=after['median'],
                speedup=before['median']/after['median'],
                reduction_percent=100*(1-after['median']/before['median']))


def release_analysis(path):
    """Read the independent fresh prepare+run ABBA, never the old cold setup."""
    path=Path(path);raw=read(path);exact_path=path.with_name('EXACT_AND_TIMING.json');exact=read(exact_path)
    groups=defaultdict(list)
    for row in raw['rows']:
        if row['status']['status']!='complete':raise ValueError('Incomplete release run')
        groups[row['case']].append(row)
    cases={}
    keys=('preparation_process_s','run_process_s','core_wall_s','total_process_s')
    if set(groups)!=set(exact['summary']):raise ValueError('Release timing case inventory differs')
    comparisons=exact['comparisons']
    if len(comparisons)!=3*len(groups):raise ValueError('Incomplete release exact checks')
    for name,rows in groups.items():
        rows.sort(key=lambda r:r['index'])
        if [r['index'] for r in rows]!=[0,1,2,3] or [r['variant'] for r in rows]!=['before','after','after','before']:
            raise ValueError('Release sequence must be complete ABBA')
        selected=[r for r in comparisons if r['case']==name]
        if sorted((r['index'],r['variant']) for r in selected)!=[(1,'after'),(2,'after'),(3,'before')]:
            raise ValueError('Release exact-check inventory differs')
        if not all(r['exact'] and r['same_clock'] and r['same_budgets'] for r in selected):
            raise ValueError('Release failed field, clock, or budget check')
        variants={}
        for variant in ('before','after'):
            vr=[r for r in rows if r['variant']==variant]
            for key in ('runtime','runtime_manifest_sha256','source','source_sha256'):
                if len({r[key] for r in vr})!=1:raise ValueError('Repeated release identity differs: '+key)
            for r in vr:
                if not math.isclose(r['total_process_s'],r['preparation_process_s']+r['run_process_s'],abs_tol=1e-9):
                    raise ValueError('Invalid release end-to-end timing')
                if r['core_wall_s']>r['run_process_s']:raise ValueError('Worker timer exceeds process timer')
            variants[variant]={key:stats(r[key] for r in vr) for key in keys}
            variants[variant].update(outside_worker_s=stats(r['run_process_s']-r['core_wall_s'] for r in vr),
                runtime=vr[0]['runtime'],runtime_manifest_sha256=vr[0]['runtime_manifest_sha256'],
                source=vr[0]['source'],source_sha256=vr[0]['source_sha256'],runs=[r['run'] for r in vr])
            for key in keys:
                if not math.isclose(variants[variant][key]['median'],exact['summary'][name][variant][key],rel_tol=1e-12):
                    raise ValueError('Stored release median differs from raw observations')
        before=variants['before'];after=variants['after']
        for variant in (before,after):
            if sha(variant['source'])!=variant['source_sha256']:raise ValueError('Release input changed after benchmark')
        changes=changed_fields(read(before['source']),read(after['source']))
        # The release receipt compares all frames and budgets. Independently
        # verify array and geometry content for one full before/after trajectory.
        fields=compare_runs(before['runs'][0],after['runs'][0],
            scope='Fresh release ABBA; explicit input differences are recorded. Same physics, transaction times and output times must be established from these inputs and exact clock/budget checks.')
        if not fields['all_common_arrays_exact'] or not fields['all_common_geometry_exact'] or not fields['hydraulic_counts_equal']:
            raise ValueError('Independent release array/geometry/count check failed')
        cases[name]=dict(variants=variants,improvement={key:speed_change(before[key],after[key]) for key in keys},
            project_field_changes=changes,exact_and_timing_checks=selected,independent_fields=fields,
            before_hydraulic=rows[0]['status']['hydraulic'],after_hydraulic=rows[1]['status']['hydraulic'])
    return dict(source=str(path.resolve()),source_sha256=sha(path),exact_source=str(exact_path.resolve()),
        exact_sha256=sha(exact_path),scope=raw['scope'],cases=cases,
        interpretation='Each variant has two independently prepared fresh-process runs in ABBA order. End-to-end medians use actual preparation+run observations, not a sum of separately selected medians. No speedup uses the earlier clean experiment first-cold preparation outlier. Observed ranges are not confidence intervals.')


def memory_analysis(path):
    """An ablation of nested checkpoint memory buffers, not the final release."""
    path=Path(path);raw=read(path);exact_path=path.with_name('EXACT.json');exact=read(exact_path)
    groups=defaultdict(list)
    for row in raw['rows']:
        if row['status']['status']!='complete':raise ValueError('Incomplete memory ablation')
        groups[row['case']].append(row)
    cases={}
    for name,rows in groups.items():
        rows.sort(key=lambda r:r['order'])
        if [r['order'] for r in rows]!=[0,1,2,3] or [r['variant'] for r in rows]!=['baseline','memory','memory','baseline']:
            raise ValueError('Incomplete memory ABBA')
        checks=[r for r in exact if r['case']==name]
        if len(checks)!=3 or not all(r['exact'] for r in checks):raise ValueError('Memory exact checks failed')
        if len({r['project_sha256'] for r in rows})!=1:raise ValueError('Memory input changed')
        if not all(r['status']['hydraulic']==rows[0]['status']['hydraulic'] for r in rows):
            raise ValueError('Memory changed hydraulic work')
        variants={}
        for variant in ('baseline','memory'):
            vr=[r for r in rows if r['variant']==variant]
            variants[variant]=dict(core_wall_s=stats(r['status']['wall_s'] for r in vr),
                public_worker_s=stats(r['worker_s'] for r in vr),process_s=stats(r['process_s'] for r in vr),
                runs=[r['run'] for r in vr])
        cases[name]=dict(variants=variants,
            improvement={key:speed_change(variants['baseline'][key],variants['memory'][key])
                         for key in ('core_wall_s','public_worker_s','process_s')},
            all_fields_exact=True,hydraulic_work_exact=True)
    return dict(source=str(path.resolve()),source_sha256=sha(path),exact_source=str(exact_path.resolve()),
        exact_sha256=sha(exact_path),cases=cases,
        interpretation='Memory-only validation hook on the old release. It still publishes full checkpoints at every output and is not the final cadence optimization. Preparation is outside this experiment; wrapper timing differs from the final public-launcher ABBA. Use within-experiment relative changes, not subtraction of independently measured phase costs.')


def release_figure(release,output):
    names=list(release['cases']);x=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    for ax,key,title in zip(axes,('core_wall_s','run_process_s','total_process_s'),
                           ('Worker only','Fresh run process','Fresh preparation + run')):
        top=0.
        for offset,variant,label,color in [(-.19,'before','Before','#a66d4b'),(.19,'after','After','#276b9c')]:
            values=[release['cases'][n]['variants'][variant][key] for n in names]
            m=np.array([r['median'] for r in values]);lo=np.array([r['minimum'] for r in values]);hi=np.array([r['maximum'] for r in values]);top=max(top,float(max(hi)))
            bars=ax.bar(x+offset,m,.36,yerr=[m-lo,hi-m],capsize=3,label=label,color=color)
            for bar,value,upper in zip(bars,m,hi):
                ax.annotate(f'{value:.2f}',(bar.get_x()+bar.get_width()/2,upper),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
        for i,name in enumerate(names):
            improvement=release['cases'][name]['improvement'][key]
            ax.text(i,.98,f"{improvement['speedup']:.2f}x / {improvement['reduction_percent']:.1f}% less",
                    transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=9)
        ax.set(xticks=x,xticklabels=['CPU8' if n.startswith('cpu8') else 'GPU' if n.startswith('gpu') else n for n in names],
               ylim=(0,top*1.25),ylabel='Wall seconds',title=title)
        ax.grid(axis='y',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2,fontsize=9)
    fig.suptitle('Qualified release: same 1 s result frequency and hydraulic work\nIndependent before/after ABBA; two samples each; median and observed range',fontsize=12)
    fig.savefig(output/'RELEASE_SPEEDUP.png',dpi=160);plt.close(fig)


def ablation_table(memory,release=None):
    result=[]
    for name,case in memory['cases'].items():
        result.append(dict(experiment='Nested checkpoint memory buffers only',case=name,
            worker=case['improvement']['core_wall_s'],run_process=case['improvement']['process_s'],
            total_prepare_run=None,exact_fields_and_hydraulic_work=True))
    if release:
        for name,case in release['cases'].items():
            result.append(dict(experiment='Complete qualified release',case=name,
                worker=case['improvement']['core_wall_s'],run_process=case['improvement']['run_process_s'],
                total_prepare_run=case['improvement']['total_process_s'],exact_fields_and_hydraulic_work=True))
    return dict(rows=result,note='Separate ABBA experiments with their own controls; improvements must not be added or attributed by subtracting absolute timings. Memory-only and final-release process wrappers differ.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',type=Path,default=EVIDENCE/'clean/BENCHMARK.json')
    parser.add_argument('--output',type=Path,default=EVIDENCE/'analysis')
    parser.add_argument('--candidate-benchmark',type=Path)
    parser.add_argument('--release-benchmark',type=Path,help='Fresh public-launcher ABBA from benchmark_release134.py')
    parser.add_argument('--memory-benchmark',type=Path,help='Optional completed nested-checkpoint memory ablation')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=benchmark(args.benchmark);comparisons=comparison_groups(base['cases'])
    profiles=phase_figure(args.output);timing_figures(base['cases'],args.output);clocks=timestep_figure(args.output)
    summary=dict(schema='braids.execution-audit134.v1',analysis_sha256=sha(__file__),baseline=base,
        field_comparisons=comparisons,instrumented_phases=profiles,clock_audit=clocks,
        interpretation=['Output-frequency comparisons hold the outer clock fixed. They isolate persistence/notification work only if all common fields and hydraulic counts match.',
            'Outer-clock 0.5/5 s controls retain 10 s output, but can change numerical substep partitioning. Their q/z/wet-area differences and step counts are reported separately.',
            'Iber sampled timestep median cannot be directly substituted for its unknown exact accepted-step mean. Equal numerical CFL values are not equivalent across different definitions.',
            'This 3.25 m square matched-area comparison is not the previous 5 m dynamic AMR case. Domain/open-section/raster differences remain explicitly recorded in matched/MATCHED_INPUTS.json.'])
    if args.candidate_benchmark:
        candidate=benchmark(args.candidate_benchmark);summary['candidate']=candidate
        summary['candidate_comparison']=candidate_analysis(base,candidate)
    if args.release_benchmark:
        summary['release']=release_analysis(args.release_benchmark)
        release_figure(summary['release'],args.output)
        summary['instrumented_release_phases']=phase_release_figure(args.output)
    if args.memory_benchmark:
        summary['memory_ablation']=memory_analysis(args.memory_benchmark)
        summary['ablation_table']=ablation_table(summary['memory_ablation'],summary.get('release'))
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:{key:value[key] for key in ('core_wall_s','run_process_s','preparation_process_s')} for name,value in base['cases'].items()},indent=2))


if __name__=='__main__':main()
