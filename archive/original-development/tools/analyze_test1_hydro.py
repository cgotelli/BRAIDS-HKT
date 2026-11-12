"""Read-only audit of the user's 60 s dry-start CPU/GPU AMR trajectories."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_test1_hydro110'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/test1_hydro110'


def read(folder):
    index=json.loads((folder/'frames/INDEX.json').read_text())
    return {float(row['time_s']):(folder,row) for row in index['frames']}


def frame(item):
    folder,row=item
    with np.load(folder/'frames'/row['file']) as src:
        fields={k:src[k] for k in src.files}
    with np.load(folder/'frames'/row['epoch']) as src:
        mesh={k:src[k] for k in src.files}
    return fields,mesh


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cpu=read(BASE/'cpu_full');gpu=read(BASE/'gpu_case/run');gpu.update(read(BASE/'gpu_continued'))
    assert sorted(cpu)==sorted(gpu)==list(np.arange(61,dtype=float))
    metrics={};paired=[];volumes={'cpu':[],'gpu':[]}
    for backend,series,folders in (('cpu',cpu,[BASE/'cpu_full']),
            ('gpu',gpu,[BASE/'gpu_case/run',BASE/'gpu_continued'])):
        diagnostics=sum([json.loads((p/'DIAGNOSTICS.json').read_text()) for p in folders],[])
        statuses=[json.loads((p/'STATUS.json').read_text()) for p in folders]
        for t,item in sorted(series.items()):
            f,m=frame(item);h=f['q'][0]
            assert np.isfinite(f['q']).all() and np.min(h)>=0
            assert not np.any(f['suspended']) and not np.any(f['bed_boundary'])
            assert not np.any(f['bank_gross']) and not np.any(f['collapse_gross'])
            volumes[backend].append(float(np.dot(h,m['area'])))
        metrics[backend]=dict(status=statuses,frames=len(series),
            initial_volume_m3=volumes[backend][0],final_volume_m3=volumes[backend][-1],
            max_water_residual_m3=max(abs(r['water_residual_m3']) for r in diagnostics),
            max_wall_flux=max(abs(v) for r in diagnostics for v in r['wall_integrals']),
            cell_count_min=min(r['cells'] for r in diagnostics),cell_count_max=max(r['cells'] for r in diagnostics),
            no_sediment_transport=True)
        assert all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in diagnostics)
    for t in cpu:
        a,am=frame(cpu[t]);b,bm=frame(gpu[t])
        assert all(np.array_equal(am[k],bm[k]) for k in am),'CPU/GPU AMR meshes differ'
        paired.append(dict(time_s=t,max_depth_difference_m=float(np.max(np.abs(a['q'][0]-b['q'][0]))),
            max_momentum_difference=float(np.max(np.abs(a['q'][1:]-b['q'][1:]))),
            max_bed_difference_m=float(np.max(np.abs(a['z']-b['z'])))))
    metrics['paired']=paired
    metrics['scope']='Unchanged Test1 hydraulics/AMR; GPU resumed at 2 s. Concurrent runs are not a speed benchmark.'
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(11,9),constrained_layout=True)
    hmax=max(float(frame(series[60.])[0]['q'][0].max()) for series in (cpu,gpu))
    origin=np.array([2571200.,1106900.])
    for row,(backend,series) in enumerate((('CPU',cpu),('GPU',gpu))):
        for col,t in enumerate((0.,30.,60.)):
            f,m=frame(series[t]);polys=[m['vertices'][s:e]-origin for s,e in zip(m['offsets'][:-1],m['offsets'][1:])]
            ax=axes[row,col];pc=PolyCollection(polys,array=np.ma.masked_less(f['q'][0],.001),
                cmap='Blues',clim=(0,hmax),edgecolors='none');ax.add_collection(pc);ax.autoscale_view()
            ax.set_aspect('equal');ax.set_title(f'{backend}: {t:g} s');ax.set_xlabel('Easting - 2571200 (m)')
            if col==0:ax.set_ylabel('Northing - 1106900 (m)')
            outline=PolyCollection(polys,facecolors='none',edgecolors='0.8',linewidths=.15)
            ax.add_collection(outline)
    fig.colorbar(pc,ax=axes.ravel().tolist(),label='Water depth (m)',shrink=.65)
    fig.suptitle('Test1: hydrodynamics only, original dry start and AMR')
    fig.savefig(OUT/'DEPTH_EVOLUTION.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k!='paired'},indent=2))
    print('Maximum paired depth difference (m):',max(r['max_depth_difference_m'] for r in paired))


if __name__=='__main__':main()
