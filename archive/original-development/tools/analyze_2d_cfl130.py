"""Raw all-output comparisons for cropped, closed, genuinely 2D tests."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())


def frames(folder):
    root=folder/'run/frames';result=[]
    for row in read(root/'INDEX.json')['frames']:
        with np.load(root/row['epoch']) as data:mesh={k:data[k] for k in data.files}
        with np.load(root/row['file']) as data:state={k:data[k] for k in data.files}
        result.append((row['time_s'],mesh,state))
    return result


def compare(a,b):
    rows=[]
    for (ta,ma,sa),(tb,mb,sb) in zip(a,b):
        if ta!=tb or any(not np.array_equal(ma[k],mb[k]) for k in ('leaves','area','vertices','offsets')):
            raise ValueError('Raw closed-test geometry/clocks differ')
        qa,qb=sa['q'],sb['q'];delta=qa-qb;area=ma['area'];worst=int(np.argmax(abs(delta[0])))
        va=np.divide(qa[1:],qa[0],out=np.zeros_like(qa[1:]),where=qa[0]>1e-6)
        vb=np.divide(qb[1:],qb[0],out=np.zeros_like(qb[1:]),where=qb[0]>1e-6)
        wet=(qa[0]>.01)&(qb[0]>.01)
        bounds=ma['offsets'][worst:worst+2];xy=ma['vertices'][bounds[0]:bounds[1]].mean(0)
        rows.append(dict(time_s=ta,depth_max_m=float(abs(delta[0]).max()),
            depth_rms_m=float(np.sqrt(delta[0]**2@area/area.sum())),
            velocity_max_m_s_both_h_above1cm=float(np.linalg.norm(va-vb,axis=0)[wet].max(initial=0.)),
            max_depth_cell=worst,max_depth_cell_center_approx_m=xy.tolist(),max_depth_cell_area_m2=float(area[worst]),
            max_depth_cell_depths_m=[float(qa[0,worst]),float(qb[0,worst])],
            wet_symmetric_area_m2={str(h):float(area[(qa[0]>h)!=(qb[0]>h)].sum()) for h in (.001,.01)}))
    return dict(depth_max_m=max(r['depth_max_m'] for r in rows),
        depth_rms_max_m=max(r['depth_rms_m'] for r in rows),
        velocity_max_m_s_both_h_above1cm=max(r['velocity_max_m_s_both_h_above1cm'] for r in rows),
        all_outputs=rows,depth_screen5mm=max(r['depth_max_m'] for r in rows)<=.005)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);args=parser.parse_args()
    out=args.input.resolve();runs={p.parent.name:read(p) for p in out.glob('*/RESEARCH.json')}
    cache={name:frames(out/name) for name in runs};summary=dict(cases={},comparisons={},all_outputs_compared=True,
        scope='2 s bounded closed-wall regression; diagonal dam and oblique partly-dry lake; no general nonlinear stability theorem')
    for name,r in runs.items():
        rows=r['history'];q0=np.load(out/name/'INITIAL_OVERRIDE.npz')['q']
        summary['cases'][name]=dict(substeps=r['substeps'],rejections=len(r['rejections']),
            rejection_kinds=sorted({d['kind'] for d in r['rejections']}),
            min_depth_m=min(s['minimum_depth_m'] for s in rows),
            max_water_residual_m3=max(abs(s['water_residual_m3']) for s in rows),
            max_wall_ledger_m3=max(s['wall_integral'] for s in rows),
            min_cell_area_m2=min(s['minimum_cell_area_m2'] for s in rows),cropped_cells=rows[0]['cropped_cells'],
            all_output_lake_drift_q_components=(np.max([np.max(abs(state['q']-q0),axis=1) for _,_,state in cache[name]],axis=0).tolist()
                                               if r['case']=='lake' else None))
        base=f'{r["case"]}_cpu_cfl0.125'
        if name!=base:summary['comparisons'][name+'__fine']=compare(cache[name],cache[base])
        cpu=f'{r["case"]}_cpu_cfl{r["research"]["actual_face_sum_cfl"]:g}'
        if r['backend']=='gpu':summary['comparisons'][name+'__CPU']=compare(cache[name],cache[cpu])
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,case in enumerate(('dam','lake')):
        for cap in (.5,1.,1.5):
            for backend,style in [('cpu','-'),('gpu',':')]:
                name=f'{case}_{backend}_cfl{cap:g}__fine'
                if name not in summary['comparisons']:continue
                rows=summary['comparisons'][name]['all_outputs']
                axes[j].plot([r['time_s'] for r in rows],[1e3*r['depth_max_m'] for r in rows],style,
                             label=f'{backend.upper()} cap{cap:g}')
        axes[j].set_title('Diagonal dry dam' if case=='dam' else 'Oblique partially dry lake')
        axes[j].set_xlabel('Physical time (s)');axes[j].set_ylabel('Maximum depth difference vs cap0.125 (mm)')
        axes[j].set_ylim(bottom=0);axes[j].grid(alpha=.2);axes[j].legend(fontsize=8)
    fig.suptitle('Cropped closed walls; same SSPRK22 augmented f-wave flux')
    fig.tight_layout();fig.savefig(out/'CLOSED_2D_DIFFERENCES.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for row,case in enumerate(('dam','lake')):
        _,mesh,ref=cache[f'{case}_cpu_cfl0.125'][-1]
        polys=[mesh['vertices'][a:b] for a,b in zip(mesh['offsets'][:-1],mesh['offsets'][1:])]
        for col,cap in enumerate((.125,1.,1.5)):
            q=cache[f'{case}_cpu_cfl{cap:g}'][-1][2]['q']
            value=q[0] if col==0 else 1000*(q[0]-ref['q'][0])
            scale=max(float(abs(value).max()),1e-12)
            patch=PolyCollection(polys,array=value,cmap='Blues' if col==0 else 'RdBu_r',edgecolors='none')
            patch.set_clim(0 if col==0 else -scale,scale);axes[row,col].add_collection(patch)
            axes[row,col].autoscale_view();axes[row,col].set_aspect('equal');axes[row,col].set_xlabel('x (m)');axes[row,col].set_ylabel('y (m)')
            axes[row,col].set_title(f'{case}: '+('reference depth (m)' if col==0 else f'cap{cap:g} - reference (mm)'))
            fig.colorbar(patch,ax=axes[row,col],shrink=.8)
    fig.suptitle('Final raw cells: cropped walls retained; no interpolation or smoothing')
    fig.tight_layout();fig.savefig(out/'CLOSED_2D_MAPS.png',dpi=160);plt.close(fig)
    for name,row in summary['comparisons'].items():
        print(name,dict(depth_mm=1000*row['depth_max_m'],rms_mm=1000*row['depth_rms_max_m'],pass5mm=row['depth_screen5mm']),flush=True)


if __name__=='__main__':main()
