"""All-output natural-river screening; private effective settings are explicit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_inlet129 import load_run, compare_runs, same_mesh


def centers(mesh):
    return np.asarray([mesh['vertices'][a:b].mean(0)
                       for a,b in zip(mesh['offsets'][:-1],mesh['offsets'][1:])])


def front_comparison(candidate, reference):
    result=[]
    for fa in candidate['frames']:
        matches=[f for f in reference['frames'] if abs(f['row']['time_s']-fa['row']['time_s'])<1e-9]
        if not matches:
            continue
        fb=matches[0]
        match,reason=same_mesh(fa['mesh'],fb['mesh'])
        if not match:
            result.append(dict(time_s=fa['row']['time_s'],status='unmapped',reason=reason))
            continue
        area=fa['mesh']['area'];xy=centers(fa['mesh'])
        for threshold in (.001,.01):
            a=fa['state']['q'][0]>threshold;b=fb['state']['q'][0]>threshold
            front_a=float(xy[a,1].max()) if a.any() else None
            front_b=float(xy[b,1].max()) if b.any() else None
            result.append(dict(time_s=fa['row']['time_s'],status='compared',threshold_m=threshold,
                wet_area_candidate_m2=float(area[a].sum()),wet_area_reference_m2=float(area[b].sum()),
                symmetric_difference_m2=float(area[a!=b].sum()),
                northward_front_difference_m=None if front_a is None or front_b is None else front_a-front_b,
                note='This case flows north. Front is the most northern wet cell centroid, quantized by the mesh, not a continuous front.'))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folders',type=Path,nargs='+',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--fine',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    refs={'baseline':load_run(a.reference)}
    if a.fine:refs['fine']=load_run(a.fine)
    rows=[]
    for folder in a.folders:
        expt=json.loads((folder/'EXPERIMENT.json').read_text()) if (folder/'EXPERIMENT.json').exists() else None
        if not (folder/'QUALIFICATION.json').exists():
            rows.append(dict(name=folder.name,folder=str(folder),experiment=expt,completed=False))
            continue
        run=load_run(folder)
        row=dict(name=folder.name,folder=str(folder),experiment=expt,completed=True,
                 summary=run['summary'],comparisons={})
        for name,ref in refs.items():
            pair=compare_runs(run,ref)
            pair['category']='explicitly_declared_private_numerical_experiment'
            pair['front']=front_comparison(run,ref)
            paired=[e for e in pair['per_time'] if e['status']=='compared']
            pair['depth_screen_pass']=bool(paired) and pair['skipped_times']==0 and pair['depth_max_m']<=.005
            row['comparisons'][name]=pair
        rows.append(row)
    output=dict(reference=str(a.reference),fine=str(a.fine),
        measurement='Concurrent/instrumented runtimes excluded from speed comparisons.',
        scope='All saved times; no claim about extrema between saved outputs. Front checks must be reviewed in addition to depth screen.',cases=rows)
    (a.output/'SUMMARY.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row in rows:
        if not row['completed']:continue
        summary=row['summary'];pair=row['comparisons']['baseline'];name=row['name']
        run=load_run(Path(row['folder']))
        steps=run['qualification']['steps']
        axes[0,0].plot([s['time_s'] for s in steps],
                       [s['diagnostics']['hydraulic_next_substep_hint_s'] for s in steps],label=name)
        per=[r for r in pair['per_time'] if r['status']=='compared']
        axes[0,1].plot([r['time_s'] for r in per],[1000*r['depth_max_m'] for r in per],label=name)
        for threshold,style in ((.001,'-'),(.01,'--')):
            fs=[r for r in pair['front'] if r['status']=='compared' and r['threshold_m']==threshold]
            axes[1,0].plot([r['time_s'] for r in fs],[r['northward_front_difference_m'] for r in fs],
                           style,label=name+f' ({threshold*1000:g} mm)')
        axes[1,1].bar(name,summary['status']['hydraulic']['accepted_substeps'])
    axes[0,0].set(title='Unclipped next hydraulic timestep',ylabel='s')
    axes[0,1].set(title='Largest saved depth difference vs alpha129',ylabel='mm')
    axes[1,0].set(title='Wet-front displacement vs alpha129 (cell-centroid resolution)',ylabel='m')
    axes[1,1].set(title='Accepted hydraulic steps for 100 s',ylabel='steps')
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in (axes[0,0],axes[0,1],axes[1,0]):ax.set_xlabel('Simulation time (s)')
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=7,ncol=2)
    fig.savefig(a.output/'NATURAL_CFL.png',dpi=170);plt.close(fig)
    for row in rows:
        if row['completed']:
            print(row['name'],row['summary']['status']['hydraulic']['accepted_substeps'],
                  {k:dict(depth_mm=1000*v['depth_max_m'],rms_mm=1000*v['depth_rms_m'],
                          maximum_front_m=max((abs(f['northward_front_difference_m']) for f in v['front']
                            if f.get('northward_front_difference_m') is not None),default=None))
                   for k,v in row['comparisons'].items()})


if __name__=='__main__':main()
