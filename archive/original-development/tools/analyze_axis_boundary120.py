"""Raw near-boundary figures, matched timings, common-grid comparisons."""
from pathlib import Path
import hashlib,json,re,zipfile
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon
from shapely import points, STRtree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_axis_boundary120'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/axis_boundary120'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
LABELS=dict(original_gpu_main='Cropped GPU',aligned_gpu_main='Whole-cell GPU (first)',
    original_cpu_main='Cropped CPU',aligned_cpu_main='Whole-cell CPU',
    aligned_gpu_half='Whole-cell GPU (dt cap)',aligned_gpu_warm='Whole-cell GPU (repeat)')


def frame(folder):
    root=folder/'run/frames';r=read(root/'INDEX.json')['frames'][-1]
    g=dict(np.load(root/r['epoch']));s=dict(np.load(root/r['file']))
    polygons=[g['vertices'][g['offsets'][i]:g['offsets'][i+1]] for i in range(len(g['area']))]
    xy=np.array([p[:-1].mean(0) for p in polygons])
    return g,s,polygons,xy


def compare(left,right):
    ga,a,pa,xa=frame(left);gb,b,pb,xb=frame(right)
    # Sample both piecewise-constant finite-volume solutions at the same 1 m
    # centres. Exclude points outside either domain, never nearest-fill a bank.
    low=np.maximum(np.min(ga['vertices'],axis=0),np.min(gb['vertices'],axis=0))
    high=np.minimum(np.max(ga['vertices'],axis=0),np.max(gb['vertices'],axis=0))
    xx,yy=np.meshgrid(np.arange(np.ceil(low[0])+.5,high[0],1),np.arange(np.ceil(low[1])+.5,high[1],1))
    xy=np.column_stack([xx.ravel(),yy.ravel()]);pp=points(xy-low)
    def indices(polygons):
        tree=STRtree([Polygon(p-low) for p in polygons]);pairs=tree.query(pp,predicate='within')
        ids=np.full(len(xy),-1,int);ids[pairs[0]]=pairs[1];return ids
    ia,ib=indices(pa),indices(pb);ok=(ia>=0)&(ib>=0);xy=xy[ok];ia=ia[ok];ib=ib[ok]
    ha,hb=a['q'][0,ia],b['q'][0,ib];wet=(ha>.001)|(hb>.001);both=(ha>.001)&(hb>.001)
    va=np.divide(a['q'][1:,ia],ha,out=np.zeros((2,len(ia))),where=ha>1e-6)
    vb=np.divide(b['q'][1:,ib],hb,out=np.zeros((2,len(ib))),where=hb>1e-6)
    delta=hb-ha
    return dict(common_centres=int(ok.sum()),wet_union_centres=int(wet.sum()),
        depth_wet_rms_m=float(np.sqrt(np.mean(delta[wet]**2))),
        depth_wet_max_m=float(abs(delta[wet]).max()),
        velocity_bothwet_rms_m_s=float(np.sqrt(np.mean(np.sum((vb[:,both]-va[:,both])**2,axis=0)))),
        wet_area_left_m2=int((ha>.001).sum()),wet_area_right_m2=int((hb>.001).sum()),
        note='1 m common cell-centre samples, no fill outside either polygon; geometry differs.'), (xy,ha,hb,delta,wet)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    geometry=read(BASE/'GEOMETRY.json');stats={};fields={}
    for name in ('original_gpu_main','aligned_gpu_main','original_cpu_main','aligned_cpu_main','aligned_gpu_half','aligned_gpu_warm'):
        folder=BASE/name
        if not (folder/'PROFILE_SUMMARY.json').is_file():continue
        p=read(folder/'PROFILE_SUMMARY.json');g,s,polys,xy=frame(folder);fields[name]=(g,s,polys,xy)
        from braids_gpu.general_live_transport import read_records
        native=read_records(next((folder/'prepared/core/prepared/cache').glob('*/case.data')).parent)
        status=p['result'];d=read(folder/'run/DIAGNOSTICS.json')[-1]
        rejects=[row for row in p['singles'] if row['error']]
        diagnosed=[]
        for row in rejects:
            m=re.search(r'cell=(\d+)',row['error'])
            if not m:continue
            i=int(m[1]);exterior=(native['owner']==i)&(native['neighbour']<0)
            diagnosed.append(dict(time_s=row['time'],dt_s=row['dt'],cell=i,
                xy=native['xy'][:,i].tolist(),area_m2=float(g['area'][i]),
                exterior_ids=native['bid'][exterior].tolist(),
                distance_to_inlet_plane_m=float(abs(native['xy'][1,i]-geometry['sections']['2']['position'])),
                error=row['error']))
        stats[name]=dict(worker_s=p['worker_call_s'],preparation_s=p['preparation_s'],
            phases=p['phases'],status=status,water_budget=d,
            cells=len(g['area']),min_area_m2=float(g['area'].min()),
            partial_cells=int((g['area']<16/4.**g['levels']*(1-1e-10)).sum()),
            water_volume_m3=float(g['area']@s['q'][0]),maximum_depth_m=float(s['q'][0].max()),
            minimum_depth_m=float(s['q'][0].min()),
            wet_area_m2=float(g['area'][s['q'][0]>.001].sum()),
            attempted_solves=len(p['singles']),failed_solves=len(rejects),failure_cells=diagnosed,
            runtime_identity=p['runtime_identity'])
        velocity=np.divide(s['q'][1:],s['q'][0],out=np.zeros_like(s['q'][1:]),where=s['q'][0]>1e-6)
        wet=s['q'][0]>.001
        stats[name]['maximum_wet_speed_m_s']=float(np.linalg.norm(velocity[:,wet],axis=0).max())
    if 'aligned_gpu_main' not in fields:return
    # Geometry and depth on RAW polygons, not smoothed/interpolated raster.
    names=[n for n in ('original_gpu_main','aligned_gpu_main') if n in fields]
    fig,axes=plt.subplots(2,len(names),figsize=(6*len(names),9),squeeze=False,layout='constrained')
    origin=np.mean(geometry['sections']['2']['original_coordinates'],axis=0)
    vmax=max(float(fields[n][1]['q'][0].max()) for n in names)
    for column,name in enumerate(names):
        g,s,polys,xy=fields[name]
        for row,lim in enumerate(((-22,27,-12,35),(-25,25,-10,8))):
            ax=axes[row,column];pick=np.max(abs(xy-origin),axis=1)<55
            pc=PolyCollection([p-origin for p,k in zip(polys,pick) if k],array=s['q'][0,pick],
                cmap='Blues',clim=(0,vmax),edgecolors='.6',lw=.2)
            ax.add_collection(pc)
            for key,color in [('original_coordinates','darkorange'),('aligned_coordinates','crimson')]:
                line=np.array(geometry['sections']['2'][key])-origin
                ax.plot(*line.T,c=color,lw=2,label=key.split('_')[0]+' inlet')
            ax.set(xlim=lim[:2],ylim=lim[2:],aspect='equal',xlabel='Relative easting (m)',
                ylabel='Relative northing (m)',title=LABELS[name]+' at 3.75 s')
            if row==0:ax.legend(fontsize=8)
    fig.colorbar(pc,ax=axes,label='Depth (m)',shrink=.7)
    fig.savefig(OUT/'INLET_DEPTH_AND_MESH.png',dpi=170);plt.close(fig)
    # Full domain and both open sections, with actual shift/width visible.
    old=read(BASE/'ORIGINAL.json')['settings']['domain']['vertices_xy']
    new=read(BASE/'ALIGNED.json')['settings']['domain']['vertices_xy']
    fig,axes=plt.subplots(1,3,figsize=(13,6),layout='constrained')
    for ax,section in zip(axes,(None,'2','1')):
        offset=np.min(np.array(old),axis=0) if section is None else np.mean(geometry['sections'][section]['original_coordinates'],axis=0)
        ax.plot(*(np.array(old)-offset).T,c='darkorange',label='Original polygon',lw=1)
        ax.plot(*(np.array(new)-offset).T,c='navy',label='Whole-cell polygon',lw=.8)
        if section:
            for key,color in [('original_coordinates','darkorange'),('aligned_coordinates','red')]:
                ax.plot(*(np.array(geometry['sections'][section][key])-offset).T,c=color,lw=2.5)
            ax.set(xlim=(-27,27),ylim=(-20,22))
        ax.set(aspect='equal',title='Domain' if section is None else ('Inlet' if section=='2' else 'Outlet'),
            xlabel='Relative easting (m)',ylabel='Relative northing (m)')
        ax.legend(fontsize=8)
    fig.savefig(OUT/'BOUNDARY_GEOMETRY.png',dpi=170);plt.close(fig)
    # Accepted hydraulic dt, excluding outer transaction clock.
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in names:
        p=read(BASE/name/'PROFILE_SUMMARY.json');steps=p['steps']
        axes[0].step([r['time_s'] for r in steps],[r['hydraulic_minimum_dt'] for r in steps],where='post',label=LABELS[name])
    axes[0].set(yscale='log',xlabel='Simulation time (s)',ylabel='Minimum accepted substep in outer interval (s)');axes[0].legend(fontsize=8)
    labels=list(stats);values=[stats[n]['phases']['stepping'] for n in labels]
    axes[1].bar(np.arange(len(labels)),values)
    axes[1].set(xticks=np.arange(len(labels)),xticklabels=[LABELS[n].replace(' GPU','\nGPU').replace(' CPU','\nCPU').replace(' (','\n(') for n in labels],ylabel='Stepping wall time (s)')
    fig.savefig(OUT/'TIMESTEP_AND_RUNTIME.png',dpi=170);plt.close(fig)
    if len(names)==2:
        c,data=compare(BASE/names[0],BASE/names[1]);stats['geometry_comparison']=c
        xy,ha,hb,delta,wet=data;sel=wet&(np.max(abs(xy-origin),axis=1)<70)
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
        bound=max(abs(delta[sel]).max(),1e-6)
        pc=ax.scatter(*(xy[sel]-origin).T,c=delta[sel],s=15,marker='s',cmap='RdBu_r',vmin=-bound,vmax=bound)
        ax.set(aspect='equal',xlabel='Relative easting (m)',ylabel='Relative northing (m)',title='Whole-cell minus cropped depth at common 1 m centres')
        fig.colorbar(pc,ax=ax,label='Depth difference (m)');fig.savefig(OUT/'DEPTH_DIFFERENCE.png',dpi=170);plt.close(fig)
    for a,b,label in [('aligned_gpu_main','aligned_cpu_main','backend'),('aligned_gpu_main','aligned_gpu_half','timestep'),('aligned_gpu_main','aligned_gpu_warm','repeat')]:
        if a in fields and b in fields:
            ga,sa,*_=fields[a];gb,sb,*_=fields[b]
            assert np.array_equal(ga['vertices'],gb['vertices'])
            diff=sb['q']-sa['q'];wet=(sa['q'][0]>.001)|(sb['q'][0]>.001)
            stats[label+'_comparison']=dict(maximum_q_difference=float(abs(diff).max()),
                maximum_depth_difference_m=float(abs(diff[0]).max()),
                wet_depth_rms_m=float(np.sqrt(np.mean(diff[0,wet]**2))))
            if label=='timestep':
                _,_,polys,xy=fields[a];peak=int(np.argmax(abs(diff[0])))
                stats[label+'_comparison']['maximum_depth_cell_xy']=xy[peak].tolist()
                fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
                limit=max(float(abs(diff[0]).max()*1000),1e-6)
                pc=PolyCollection([p-origin for p,k in zip(polys,wet) if k],
                    array=1000*diff[0,wet],cmap='RdBu_r',clim=(-limit,limit),edgecolors='.6',lw=.25)
                ax.add_collection(pc);ax.autoscale_view();ax.set_aspect('equal')
                ax.scatter(*(xy[peak]-origin),marker='x',c='black',s=70)
                ax.set(xlabel='Relative easting (m)',ylabel='Relative northing (m)',
                    title='Same geometry: stricter hydraulic dt minus default\nBlack cross marks maximum depth difference')
                fig.colorbar(pc,ax=ax,label='Depth difference (mm)')
                fig.savefig(OUT/'TEMPORAL_CHECK.png',dpi=170);plt.close(fig)
    (OUT/'SUMMARY.json').write_text(json.dumps(dict(geometry=geometry,runs=stats),indent=2))
    with zipfile.ZipFile(OUT/'SOURCES.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('braids_core/boundaries/axis_aligned.py','tools/axis_boundary120.py',
                     'tools/analyze_axis_boundary120.py','tests/test_axis_aligned_boundaries.py',
                     'braids_core/sediment/river_midpoint.py','tools/profile_test1_amr.py',
                     'validation/RIVER_MORPHODYNAMICS_STATUS.md',
                     'validation/sediment/river_morphodynamics/UNIFIED_QGIS_PLAN.md'):
            archive.write(ROOT/name,name)
    artifacts=[p for p in OUT.iterdir() if p.name!='MANIFEST.json']
    artifacts+=list(BASE.glob('*.json'))+list(BASE.glob('*.xml'))
    for name in stats:
        folder=BASE/name
        if not folder.is_dir():continue
        artifacts+=list(folder.glob('*.json'))+list((folder/'run').glob('*.json'))+list((folder/'run/frames').glob('*'))
    (OUT/'MANIFEST.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in artifacts if p.is_file()},indent=2))
    print(json.dumps({n:{k:v for k,v in s.items() if k not in ('failure_cells','runtime_identity','status','water_budget')} for n,s in stats.items()},indent=2))


if __name__=='__main__':main()
