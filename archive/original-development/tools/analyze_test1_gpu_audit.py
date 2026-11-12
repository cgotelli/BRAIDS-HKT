"""Summarize the cancelled 1 m Test1 run and its bounded profiling replay."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon,Point,LineString

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment


def main():
    parent=Path('C:/Users/STREEM/Desktop/Test1/runs')
    run=parent/'run_20261021_162057_b146ad';prepared=parent/'prepare_20261021_162018_0bf405'
    out=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/test1_gpu_audit_v1'
    out.mkdir(parents=True,exist_ok=True)
    doc=json.loads((prepared/'PROJECT.json').read_text());status=json.loads((run/'STATUS.json').read_text())
    events=json.loads((run/'ADAPTATION.json').read_text())
    prof=json.loads((ROOT/'validation/_test1_gpu_audit/replay_v1/PROFILE.json').read_text())
    row=json.loads((run/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(run/'frames'/row['epoch']) as src:mesh={k:src[k] for k in src.files}
    with np.load(run/'frames'/row['file']) as src:fields={k:src[k] for k in src.files}
    polygons=[mesh['vertices'][a:b] for a,b in zip(mesh['offsets'][:-1],mesh['offsets'][1:])]
    cell=3949;shape=Polygon(polygons[cell]);center=np.array(shape.centroid.coords[0])
    domain=PolygonDomain(**doc['settings']['domain'])
    boundaries=domain.boundary_segments([BoundaryAssignment(**v) for v in doc['settings']['assignments']])
    distances={s.bc_id:Point(center).distance(LineString([s.start_xy,s.end_xy])) for s in boundaries if s.kind!='wall'}
    good=[r for r in prof['singles'] if r['error'] is None]
    bad=[r for r in prof['singles'] if r['error'] and not r['error'].startswith('Interrupted')]
    metrics=dict(run=str(run),prepared=str(prepared),status=status,
        cell=dict(index=cell,leaf=int(mesh['leaves'][cell]),area_m2=float(mesh['area'][cell]),
            centroid_xy=center.tolist(),q=fields['q'][:,cell].tolist(),bed_m=float(fields['z'][cell]),
            boundary_distances_m=distances),
        adaptation=dict(events=len(events),changes=sum(e['changed'] for e in events),
            wall_s=sum(e['adaptation_total_s'] for e in events),
            planning_s=sum(e['indicator_planning_s'] for e in events)),
        replay=dict(timings=prof['timings'],failed_intervals=len(bad),successful_internal_intervals=len(good),
            evaluations=sum(r['evaluations'] for r in prof['singles']),
            minimum_internal_dt_s=min(r['dt_s'] for r in prof['singles']),
            failed_interval_wall_s=sum(r['wall_s'] for r in bad),
            provisional_internal_time_advance_s=max(r['time_s']+r['dt_s'] for r in good)-good[0]['time_s'],
            accepted_outer_time_advance_s=prof['status']['time_s']-status['time_s']))
    (out/'METRICS.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,5.5),constrained_layout=True)
    origin=np.array([2571200.,1106900.])
    for ax,zoom in zip(axes[:2],(False,True)):
        ids=[i for i,p in enumerate(polygons) if not zoom or Polygon(p).distance(shape)<10]
        pc=PolyCollection([polygons[i]-origin for i in ids],array=fields['q'][0,ids],cmap='Blues',
            clim=(0,float(fields['q'][0].max())),edgecolors='.65',linewidths=.15 if not zoom else .5)
        ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal')
        for seg in boundaries:
            if seg.kind=='wall':continue
            xy=np.array([seg.start_xy,seg.end_xy])-origin
            ax.plot(xy[:,0],xy[:,1],color='green' if seg.kind=='inlet' else 'purple',lw=2)
            if not zoom:ax.text(*xy.mean(0),seg.bc_id,fontsize=9)
        xy=polygons[cell]-origin;ax.plot(xy[:,0],xy[:,1],color='red',lw=2)
        if zoom:
            c=center-origin;ax.set_xlim(c[0]-7,c[0]+7);ax.set_ylim(c[1]-5,c[1]+9)
            ax.annotate('Cell 3949\n0.632 m²',(c[0],c[1]),xytext=(c[0]-6,c[1]+6),arrowprops=dict(arrowstyle='->',color='red'))
        ax.set_xlabel('Easting − 2571200 (m)');ax.set_ylabel('Northing − 1106900 (m)')
        ax.set_title('Saved state at 0.928125 s' if not zoom else 'Failure directly on inlet i1')
    fig.colorbar(pc,ax=axes[:2],label='Water depth (m)',shrink=.7)
    ax=axes[2]
    for failed,color,label in ((False,'#268547','Successful internal interval'),(True,'#c23a38','Failed internal interval')):
        data=[(i,r) for i,r in enumerate(prof['singles']) if (r['error'] is not None)==failed]
        ax.scatter([i for i,r in data],[r['dt_s'] for i,r in data],s=16,color=color,label=label)
    ax.set_yscale('log');ax.set_xlabel('Internal attempt in 35 s replay');ax.set_ylabel('Attempted timestep (s)')
    ax.set_title('Retry cascade; no outer step committed');ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.2)
    fig.savefig(out/'INLET_AND_RETRIES.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
