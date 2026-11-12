"""Summarize unchanged-runtime replays, original results and CFL locations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_aaa128'
USER=Path('C:/Users/STREEM/Desktop/aaa/runs')
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/aaa_audit128'
RUNS={'gpu':'run_20261022_204331_367153','cpu':'run_20261022_204720_f34397'}


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def states(p):
    index=read(p/'frames/INDEX.json')
    return [(r,dict(np.load(p/'frames'/r['file'])),dict(np.load(p/'frames'/r['epoch']))) for r in index['frames']]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metrics={}
    for backend,name in RUNS.items():
        original=USER/name
        replay=BASE/(backend+'_diagnostic')
        control=BASE/(backend+'_timing')
        profile=read(replay/'PROFILE_SUMMARY.json');lim=read(replay/'LIMITERS.json')
        old=states(original);new=states(replay/'run')
        comparison=[]
        for (r,a,g),(rr,b,gg) in zip(old,new):
            assert abs(r['time_s']-rr['time_s'])<1e-10
            assert all(np.array_equal(g[k],gg[k]) for k in g)
            comparison.append(dict(time_s=r['time_s'],depth_max_m=float(abs(a['q'][0]-b['q'][0]).max()),
                momentum_max=float(abs(a['q'][1:]-b['q'][1:]).max())))
        timing=read(control/'PROFILE_SUMMARY.json') if (control/'PROFILE_SUMMARY.json').exists() else None
        status=read(original/'STATUS.json');prep=Path(read(original/'RUN_CONTEXT.json')['input']);project=read(prep/'PROJECT.json')
        accepted=[s for s in lim['singles'] if s['error'] is None]
        metrics[backend]=dict(original_status=status,settings=project['settings'],
            original_runtime_id=read(prep/'PREPARED.json')['runtime_id'],replay_identity=profile['runtime_identity'],
            original_mean_dt_s=status['time_s']/status['hydraulic']['accepted_substeps'],
            original_mesh_counts=[len(g['area']) for _,_,g in old],replay_comparison=comparison,
            diagnostic_phases_s=profile['phases'],timing_phases_s=None if timing is None else timing['phases'],
            timing_preparation_s=None if timing is None else timing['preparation_s'],
            limiter_samples=lim['samples'],dt_quantiles_s=np.quantile([s['dt_s'] for s in accepted],[0,.1,.5,.9,1]).tolist(),
            original_hashes={n:hashlib.sha256((original/n).read_bytes()).hexdigest() for n in ('STATUS.json','DIAGNOSTICS.json','frames/INDEX.json')})

    cpu=read(BASE/'boundary_cpu/BOUNDARY.json');gpu=read(BASE/'boundary_gpu/BOUNDARY.json')
    metrics['same_state_boundary_cpu_gpu_rate_max_difference']=max(float(abs(np.array(x['spatial_rate'])-y['spatial_rate']).max())
        for r,s in zip(cpu['samples'],gpu['samples']) for x,y in zip(r['cells'],s['cells']))
    metrics['boundary_samples']=gpu['samples']

    # Raw-cell inlet view: preserve cropped polygons, dry cells and signed flow.
    row,s,g=states(USER/RUNS['gpu'])[-1]
    polygons=[g['vertices'][g['offsets'][i]:g['offsets'][i+1]] for i in range(len(g['area']))]
    xy=np.array([p[:-1].mean(0) for p in polygons]);origin=np.array([2571500.,1106857.])
    local=xy-origin;pick=(local[:,0]>0)&(local[:,0]<49)&(local[:,1]<24)
    selected=[p-origin for p,k in zip(polygons,pick) if k]
    v=np.divide(s['q'][2],s['q'][0],out=np.zeros_like(s['q'][0]),where=s['q'][0]>1e-6)
    fig,axs=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax,values,title,cmap,clim in zip(axs[:2],[s['q'][0],v],
            ['Water depth at 100 s','Northward velocity at 100 s'],['Blues','RdBu'],[(0,1.5),(-14,14)]):
        pc=PolyCollection(selected,array=values[pick],edgecolors='.55',linewidths=.45,cmap=cmap,clim=clim)
        ax.add_collection(pc);ax.set(xlim=(3,47),ylim=(-4,22),aspect='equal',xlabel='Easting from 2571500 m',ylabel='Distance north of inlet (m)',title=title)
        ax.plot([7,43],[0,0],color='limegreen',lw=3,label='Inlet (flow prescribed northward)')
        ax.scatter(*local[989],s=65,facecolors='none',edgecolors='black',linewidths=1.5)
        fig.colorbar(pc,ax=ax,shrink=.75,label='Depth (m)' if cmap=='Blues' else 'Velocity (m/s)')
    axs[1].annotate('Cell 989\n13.69 m/s outward',xy=local[989],xytext=(25,13),arrowprops=dict(arrowstyle='->'),fontsize=9,
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='.7'))
    samples=metrics['gpu']['limiter_samples']
    axs[2].plot([r['time_s'] for r in samples],[r['top'][0]['cfl_limit_s'] for r in samples],label='CFL upper bound',color='#ad3c37')
    axs[2].set(xlabel='Simulation time (s)',ylabel='Hydraulic timestep (s)',title='GPU limiting timestep',ylim=(0,.14))
    axs[2].grid(alpha=.2);axs[2].legend()
    fig.suptitle('The limiting cell remains at the inlet; this is not the advancing downstream front')
    fig.savefig(OUT/'INLET_AND_TIMESTEP.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    labels=['Setup / load / other','Hydraulic stepping','AMR','Output / checkpoints']
    colors=['#969696','#4b7f9c','#c97642','#799b67']
    for j,backend in enumerate(('cpu','gpu')):
        p=metrics[backend]['timing_phases_s'] or metrics[backend]['diagnostic_phases_s']
        groups=[p.get('load',0)+p.get('initialization',0)+p.get('other',0),p['stepping'],p.get('amr',0),p['output']+p['checkpoint']]
        left=0
        for label,color,value in zip(labels,colors,groups):
            axs[0].barh(j,value,left=left,color=color,label=label if j==0 else None)
            if value>2:axs[0].text(left+value/2,j,f'{value:.2f}s',ha='center',va='center',fontsize=9)
            left+=value
        lim=read(BASE/(backend+'_diagnostic')/'LIMITERS.json')
        ss=[s for s in lim['singles'] if s['error'] is None]
        axs[1].scatter([s['time_s'] for s in ss],[s['dt_s'] for s in ss],s=3,alpha=.45,label=backend.upper())
    axs[0].set(yticks=[0,1],yticklabels=['CPU 5 m AMR / 60 s','GPU 2 m fixed / 100 s'],xlabel='Worker wall time (s)',title='Different cases and different bottlenecks')
    axs[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
    axs[1].set(xlabel='Simulation time (s)',ylabel='Accepted hydraulic timestep (s)',ylim=(0,.14),title='Clock clipping adds short substeps')
    axs[1].legend();axs[1].grid(alpha=.2)
    fig.suptitle('Installed alpha127 replays; timing excludes preparation; not a CPU/GPU speed comparison')
    fig.savefig(OUT/'COST_AND_SUBSTEPS.png',dpi=180);plt.close(fig)
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps({k:{'phases':v['timing_phases_s'],'depth_replay_max_m':max(r['depth_max_m'] for r in v['replay_comparison'])} for k,v in metrics.items() if k in ('cpu','gpu')},indent=2))


if __name__=='__main__':main()
