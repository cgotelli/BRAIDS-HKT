"""All-output hydraulic research comparisons, with no timing claims."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_canonical_inlet129 import comparison, frame, raster, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--extra', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    out = args.input.resolve()
    summary = dict(scope='100 s dry-start fixed canonical channel, not a general stability proof or speed benchmark',
        cases={}, comparisons={}, fronts={}, reference='Private cap0.5; also frozen alpha129 cap0.1125',
        accepted_depth_screen_m=.005)
    sources = [out] + [p.resolve() for p in args.extra]
    folders = {p.parent.name:p.parent for source in sources for p in source.glob('*/RESEARCH.json')}
    for research in sorted(p/'RESEARCH.json' for p in folders.values()):
        folder = research.parent; m = read(folder/'METRICS.json'); r = read(research)
        rejections = [e for step in m['steps'] for e in step['diagnostics']['hydraulic_rejections']]
        summary['cases'][folder.name] = dict(
            executed_cap=r['actual_face_sum_cfl'], dry_m=r['dry_tolerance_m'],
            substeps=m['hydraulic_dt']['accepted_substeps'], mean_dt_s=m['hydraulic_dt']['mean_s'],
            maximum_dt_s=m['hydraulic_dt']['maximum_s'], accepted_cfl_max=m['hydraulic_dt']['accepted_cfl_max'],
            rejected_steps=len(rejections),
            rejection_kinds={k:sum(e['kind']==k for e in rejections) for k in sorted({e['kind'] for e in rejections})},
            minimum_saved_depth_m=min(row['minimum_depth_m'] for row in m['history']),
            max_water_residual_m3=max(abs(row['water_residual_m3']) for row in m['history']),
            max_inlet_Q_error_m3_s=max(abs(row['inlet_Q_m3_s']-11.) for row in m['history']),
            checks=r['checks_except_original_public_cap'], actual_cap_pass=r['accepted_actual_cap'])
        method='critical' if '_critical_' in folder.name else 'normal'
        for tag, ref in [
            ('baseline05', out/f'cpu_{method}_cfl0.5_dry1e-06'),
            ('lowCFL', ROOT/f'validation/_inlet129/canonical_final/static_cpu_{method}_cflref'),
            ('same_cap1', out/f'cpu_{method}_cfl1_dry1e-06')]:
            if folder == ref or not (ref/'run/frames/INDEX.json').is_file():
                continue
            row=comparison(ref,folder)
            summary['comparisons'][folder.name+'__'+tag]=row
        front=[]
        for row in read(folder/'run/frames/INDEX.json')['frames']:
            mesh, fields=frame(folder,row); q=raster(mesh,fields['q'])
            record=dict(time_s=row['time_s'])
            for threshold in (.001,.01):
                mask=q[0]>threshold
                record[f'area_h_gt_{threshold:g}_m2']=float(mask.sum()*25.)
                xs=np.where(mask.any(0))[0]
                record[f'front_h_gt_{threshold:g}_m']=float((xs.max()+1)*5.) if len(xs) else 0.
            front.append(record)
        summary['fronts'][folder.name]=front
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={.5:'#34495e',.75:'#168a78',1.:'#e67e22',1.5:'#ad3e62'}
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    x=np.arange(20)*5+2.5
    for j,method in enumerate(('critical','normal')):
        for cap,color in colors.items():
            name=f'cpu_{method}_cfl{cap:g}_dry1e-06';folder=folders.get(name,out/name)
            if name not in summary['cases']:continue
            m=read(folder/'METRICS.json');steps=m['steps']
            axes[0,j].plot([s['time_s'] for s in steps],
                [s['diagnostics']['hydraulic_next_substep_hint_s'] for s in steps],
                color=color,label=f'Cap {cap:g}: {summary["cases"][name]["substeps"]} steps')
            comp=summary['comparisons'].get(name+'__lowCFL')
            if comp:
                axes[1,j].plot([r['time_s'] for r in comp['history']],
                    [1000*r['depth_max_m'] for r in comp['history']],color=color,label=f'Cap {cap:g}')
        axes[0,j].set_title(method.title()+' inlet / dry start')
        axes[0,j].set_ylabel('Next internal dt hint (s)')
        axes[1,j].set_ylabel('Max depth difference vs cap0.1125 (mm)')
        axes[1,j].axhline(5.,color='black',ls='--',label='5 mm screen')
        for ax in axes[:,j]:
            ax.set_xlabel('Simulated time (s)');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same SSPRK22 + augmented f-wave flux: isolated face-sum CFL controls')
    fig.tight_layout();fig.savefig(out/'CANONICAL_CFL.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for j,method in enumerate(('critical','normal')):
        for dry,color in [(1e-6,'#34495e'),(1e-4,'#168a78'),(1e-3,'#ad3e62')]:
            name=f'cpu_{method}_cfl1_dry{dry:g}';folder=folders.get(name,out/name)
            if name not in summary['cases']:continue
            rows=summary['fronts'][name]
            axes[0,j].plot([r['time_s'] for r in rows],[r['front_h_gt_0.01_m'] for r in rows],
                          color=color,label=f'dry={dry:g} m')
            row=read(folder/'run/frames/INDEX.json')['frames'][-1];mesh,fields=frame(folder,row)
            q=raster(mesh,fields['q'])
            axes[1,j].plot(x,q[0].mean(0),color=color,label=f'dry={dry:g} m')
        axes[0,j].set_title(method.title()+' inlet / same cap1.0')
        axes[0,j].set_xlabel('Simulated time (s)');axes[0,j].set_ylabel('Furthest wet cell edge, h>1cm (m)')
        axes[1,j].set_xlabel('Downstream distance (m)');axes[1,j].set_ylabel('Width-mean final depth (m)')
        for ax in axes[:,j]:ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Separate wet/dry-threshold contrasts; conserved water retained below threshold')
    fig.tight_layout();fig.savefig(out/'CANONICAL_DRY.png',dpi=160);plt.close(fig)
    print(json.dumps(summary['cases'],indent=2),flush=True)


if __name__=='__main__':main()
