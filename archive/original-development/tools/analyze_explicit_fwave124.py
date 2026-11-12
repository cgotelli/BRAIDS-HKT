"""Raw equal-cell temporal accuracy, wet-front, budget and timing evidence."""
from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_explicit_fwave124'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/explicit_fwave124'


def read(p):return json.loads(p.read_text())


def load(folder):
    profile=read(folder/'PROFILE_SUMMARY.json');exp=read(folder/'EXPERIMENT.json')
    frames=folder/'run/frames';index=read(frames/'INDEX.json');rows=index['frames']
    assert len(index['epochs'])==1, 'This comparison requires fixed identical cells'
    g=dict(np.load(frames/rows[0]['epoch']))
    states={r['time_s']:dict(np.load(frames/r['file'])) for r in rows}
    polygons=[g['vertices'][g['offsets'][i]:g['offsets'][i+1]] for i in range(len(g['area']))]
    from braids_gpu.general_live_transport import read_records
    native=read_records(next((folder/'prepared/core/prepared/cache').glob('*/case.data')).parent)
    xy=native['xy'].T
    ds=read(folder/'run/DIAGNOSTICS.json')
    final=states[max(states)];wet=final['q'][0]>.001
    trials=exp['explicit_trials'] if exp['settings']['integration']=='explicit' else profile['singles']
    accepted=[r['dt'] for r in trials if not r['error']]
    h=profile['result']['hydraulic']
    record=dict(preparation_s=profile['preparation_s'],worker_s=profile['worker_call_s'],
        total_prepare_plus_worker_s=profile['preparation_s']+profile['worker_call_s'],
        stepping_s=profile['phases']['stepping'],phases=profile['phases'],status=profile['result'],
        cells=len(g['area']),min_area_m2=float(g['area'].min()),
        minimum_depth_m=float(min(s['q'][0].min() for s in states.values())),
        final_max_depth_m=float(final['q'][0].max()),final_wet_area_m2=float(g['area'][wet].sum()),
        final_volume_m3=float(g['area']@final['q'][0]),
        max_conservation_residual_m3=max(abs(r['water_residual_m3']) for r in ds),
        walls_max=max(abs(v) for r in ds for v in r['wall_integrals']),
        conservation_pass=all(abs(r['water_residual_m3'])<=r['water_roundoff_limit_m3'] for r in ds),
        finite=all(np.isfinite(s['q']).all().item() for s in states.values()),
        mean_accepted_hydraulic_dt_s=max(states)/h['accepted_substeps'],
        runtime_id=profile['runtime_identity'],fwave_evaluations=h['fwave_evaluations'],
        accepted_substeps=h['accepted_substeps'],retries=h['internal_retries'],
        all_research_sources=exp['sources'],substep_trial_dt_quantiles_s=np.quantile(accepted,[0,.5,1]).tolist(),
        dt_quantile_note='Explicit: accepted SSP steps. Implicit: all successful nonlinear solves INCLUDING estimator full trials.')
    names=sorted(read(folder/'PROJECT.json')['settings']['hydraulics'])
    boundary=[]
    for bid,name in enumerate(names):
        rows=[];last_t=0.;last_v=0.
        for t,s in sorted(states.items()):
            volume=float(s['water'][native['bid']==bid].sum())
            if t>last_t:rows.append(dict(time_s=t,mean_discharge_into_domain_m3_s=(volume-last_v)/(t-last_t),cumulative_m3=volume))
            last_t=t;last_v=volume
        boundary.append(dict(id=name,rows=rows))
    record['boundary_discharge']=boundary
    if (folder/'CFL_FINAL.npz').exists():
        c=dict(np.load(folder/'CFL_FINAL.npz'));i=int(c['frequency'].argmax())
        faces=(native['owner']==i)&(native['neighbour']<0)
        record['final_cfl_limiting_cell']=dict(index=i,xy=xy[i].tolist(),area_m2=float(g['area'][i]),
            level=int(g['levels'][i]),depth_m=float(final['q'][0,i]),
            face_sum_frequency_s_inv=float(c['frequency'][i]),
            maximum_cfl_dt_s=exp['settings']['cfl']/float(c['frequency'][i]),
            exterior_boundary_ids=native['bid'][faces].tolist())
        record['explicit_retry_reasons']=[r['error'] for r in trials if r['error']]
        record['legacy_status_counter_note']='Public status categorizes non-accuracy retries as nonlinear_retries. Here those are explicit CFL/stage retries; no nonlinear solver runs.'
    return dict(g=g,states=states,polygons=polygons,xy=xy,record=record,profile=profile)


def comparison(a,b):
    for key in a['g']:np.testing.assert_array_equal(a['g'][key],b['g'][key],err_msg=key)
    for key in ('q','z','rock','mass','W','bedforms','suspended'):
        np.testing.assert_array_equal(a['states'][0.][key],b['states'][0.][key],err_msg='initial '+key)
    area=a['g']['area'];xy=a['xy'];rows=[]
    for t in sorted(set(a['states'])&set(b['states'])):
        if t==0:continue
        qa=a['states'][t]['q'];qb=b['states'][t]['q'];dh=qa[0]-qb[0]
        wa=qa[0]>.001;wb=qb[0]>.001;wet=wa|wb;both=(qa[0]>.01)&(qb[0]>.01)
        velocity=np.linalg.norm(qa[1:,both]/qa[0,both]-qb[1:,both]/qb[0,both],axis=0)
        iv=int(np.flatnonzero(both)[velocity.argmax()]) if velocity.size else None
        front=max(float(cKDTree(xy[wb]).query(xy[wa])[0].max()),float(cKDTree(xy[wa]).query(xy[wb])[0].max()))
        imax=int(abs(dh).argmax())
        rows.append(dict(time_s=t,depth_max_mm=float(abs(dh).max()*1000),
            depth_wet_area_rms_mm=float(np.sqrt(np.average(dh[wet]**2,weights=area[wet]))*1000),
            max_depth_error_xy=xy[imax].tolist(),max_depth_error_cell=imax,
            velocity_both_over_1cm_max_m_s=float(velocity.max()) if velocity.size else None,
            velocity_both_over_1cm_area_rms_m_s=float(np.sqrt(np.average(velocity**2,weights=area[both]))) if velocity.size else None,
            max_velocity_error_cell=iv,max_velocity_error_xy=None if iv is None else xy[iv].tolist(),
            max_velocity_error_depths_m=None if iv is None else [float(qa[0,iv]),float(qb[0,iv])],
            wet_area_a_m2=float(area[wa].sum()),wet_area_b_m2=float(area[wb].sum()),
            wet_disagreement_cells=int(np.sum(wa!=wb)),wet_disagreement_area_m2=float(area[wa!=wb].sum()),
            wet_set_centroid_hausdorff_m=front,
            volume_difference_m3=float(area@dh)))
    return dict(initial_conditions_exact=True,mesh_exact=True,rows=rows,
                worst_depth_mm=max(r['depth_max_mm'] for r in rows),
                worst_wet_rms_mm=max(r['depth_wet_area_rms_mm'] for r in rows),
                wet_front_note='1 mm wet threshold; symmetric nearest-wet-cell centroid distance, not a subcell front reconstruction')


def main():
    OUT.mkdir(parents=True,exist_ok=True);runs={};result=dict(runs={},comparisons={})
    for p in sorted(BASE.glob('*/PROFILE_SUMMARY.json')):
        if not (p.parent/'EXPERIMENT.json').exists():continue
        run=load(p.parent);runs[p.parent.name]=run;result['runs'][p.parent.name]=run['record']
    for size in ('5m','1m'):
        ref=next((size+'_explicit_gpu_cfl'+v for v in ('0.028125','0.05625','0.1125') if size+'_explicit_gpu_cfl'+v in runs),size+'_explicit_gpu_cfl0.1125')
        pairs=[(f'{size}_{method}_{backend}',ref) for method in ('explicit','implicit') for backend in ('cpu','gpu')]
        pairs += [(size+'_explicit_gpu_cfl'+v,ref) for v in ('0.225','0.1125','0.05625') if size+'_explicit_gpu_cfl'+v != ref]
        pairs += [(size+'_explicit_cpu_cfl0.225',ref),
                  (size+'_explicit_cpu_cfl0.225',size+'_explicit_gpu_cfl0.225'),
                  (size+'_explicit_gpu_cfl0.225',size+'_implicit_gpu')]
        pairs += [(size+'_implicit_gpu_tight',ref),
                  (size+'_explicit_cpu',size+'_explicit_gpu'),(size+'_implicit_cpu',size+'_implicit_gpu')]
        for left,right in pairs:
            if left in runs and right in runs:result['comparisons'][left+'__vs__'+right]=comparison(runs[left],runs[right])
        names=[size+'_implicit_gpu',size+'_explicit_gpu_cfl0.225',ref]
        if not all(n in runs for n in names):continue
        a,b,r=[runs[n] for n in names];t=max(r['states']);refq=r['states'][t]['q']
        wet=refq[0]>.001;xy=r['xy'];origin=xy[wet].mean(0);bounds=np.array([xy[wet].min(0),xy[wet].max(0)])-origin
        pick=np.all((xy-origin>bounds[0]-10)&(xy-origin<bounds[1]+10),axis=1)
        polygons=[p-origin for p,k in zip(r['polygons'],pick) if k]
        fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
        vmax=max(float(v['states'][t]['q'][0].max()) for v in (a,b,r))
        maxdiff=max(abs(a['states'][t]['q'][0]-refq[0]).max(),abs(b['states'][t]['q'][0]-refq[0]).max())*1000
        for j,(run,label) in enumerate(zip((a,b,r),('Implicit, installed settings','Explicit, CFL 0.225','Reference CFL '+ref.split('cfl')[1]))):
            q=run['states'][t]['q'];delta=(q[0]-refq[0])*1000
            for i,(values,cmap,limits) in enumerate(((q[0],'Blues',(0,vmax)),(delta,'RdBu_r',(-maxdiff,maxdiff)))):
                ax=axes[i,j];pc=PolyCollection(polygons,array=values[pick],cmap=cmap,clim=limits,edgecolors='0.5',linewidths=.15)
                ax.add_collection(pc);ax.set(xlim=(bounds[0,0]-8,bounds[1,0]+8),ylim=(bounds[0,1]-8,bounds[1,1]+8),aspect='equal',
                    title=label if i==0 else 'Depth minus finer-time reference',xlabel='Local easting (m)',ylabel='Local northing (m)')
                fig.colorbar(pc,ax=ax,label='Depth (m)' if i==0 else 'Difference (mm)',shrink=.7)
        fig.suptitle(f'{size} static terrain mesh, t={t:g} s; raw cells, no spatial smoothing')
        fig.savefig(OUT/f'DEPTH_{size}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for run,label in zip((a,b,r),('Implicit','Explicit CFL .225','Reference CFL '+ref.split('cfl')[1])):
            times=sorted(run['states']);states=[run['states'][t] for t in times];area=run['g']['area']
            axes[0].plot(times,[s['q'][0].max() for s in states],marker='.',label=label)
            axes[1].plot(times,[area[s['q'][0]>.001].sum() for s in states],marker='.',label=label)
            axes[2].plot(times,[area@s['q'][0] for s in states],marker='.',label=label)
        for ax,y in zip(axes,('Peak depth (m)','Wet area, h > 1 mm (m²)','Water volume (m³)')):
            ax.set(xlabel='Time (s)',ylabel=y);ax.legend(fontsize=8)
        fig.suptitle(size+' matched cold start: hydraulic trends')
        fig.savefig(OUT/f'TRENDS_{size}.png',dpi=160);plt.close(fig)
        selected=result['comparisons'][size+'_explicit_gpu_cfl0.225__vs__'+ref]
        velocity_time=max(selected['rows'],key=lambda row:row['velocity_both_over_1cm_max_m_s'] or 0)['time_s']
        qb=b['states'][velocity_time]['q'];qr=r['states'][velocity_time]['q'];both=(qb[0]>.01)&(qr[0]>.01)
        vb=np.divide(qb[1:],qb[0],out=np.zeros_like(qb[1:]),where=qb[0]>1e-6)
        vr=np.divide(qr[1:],qr[0],out=np.zeros_like(qr[1:]),where=qr[0]>1e-6)
        error=np.linalg.norm(vb-vr,axis=0);ids=np.flatnonzero(both);cell=int(ids[error[ids].argmax()])
        fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
        for ax,values,cmap,label in ((axes[0],np.linalg.norm(vr,axis=0),'Reds','Reference speed (m/s)'),
                                     (axes[1],np.where(both,error,np.nan),'magma','Explicit vector-velocity error (m/s), h > 1 cm')):
            pc=PolyCollection(polygons,array=values[pick],cmap=cmap,edgecolors='0.5',linewidths=.15)
            ax.add_collection(pc);ax.set(xlim=(bounds[0,0]-8,bounds[1,0]+8),ylim=(bounds[0,1]-8,bounds[1,1]+8),aspect='equal',
                xlabel='Local easting (m)',ylabel='Local northing (m)',title=label)
            ax.plot(*(xy[cell]-origin),'o',mfc='none',mec='cyan',ms=10)
            fig.colorbar(pc,ax=ax,shrink=.65)
        trace=[]
        for run,label in zip((a,b,r),('Implicit','Explicit CFL .225','Finer-time reference')):
            times=sorted(run['states']);speeds=[float(np.linalg.norm(run['states'][tt]['q'][1:,cell])/max(run['states'][tt]['q'][0,cell],1e-6)) for tt in times]
            depths=[float(run['states'][tt]['q'][0,cell]) for tt in times]
            trace.append(dict(method=label,time_s=times,speed_m_s=speeds,depth_m=depths))
            axes[2].plot(times,speeds,marker='.',label=label)
        axes[2].set(xlabel='Time (s)',ylabel='Speed (m/s)',title=f'Largest velocity-error cell {cell}')
        axes[2].legend(fontsize=8);fig.suptitle(f'{size} velocity audit at t={velocity_time:g} s; cyan circle identifies traced cell')
        fig.savefig(OUT/f'VELOCITY_{size}.png',dpi=160);plt.close(fig)
        result.setdefault('velocity_traces',{})[size]=dict(time_s=velocity_time,cell=cell,xy=xy[cell].tolist(),area_m2=float(r['g']['area'][cell]),trace=trace)
    names=[f'{s}_{m}_{b}'+('_cfl0.225' if m=='explicit' else '') for s in ('5m','1m') for b in ('cpu','gpu') for m in ('implicit','explicit')]
    names=[n for n in names if n in runs]
    result['timing_pairs']={}
    for name in names:
        pair=[runs[n]['record'] for n in (name,name+'_repeat') if n in runs]
        result['timing_pairs'][name]={key:dict(samples=[v[key] for v in pair],median=float(np.median([v[key] for v in pair])))
            for key in ('preparation_s','worker_s','stepping_s','total_prepare_plus_worker_s')}
        if name+'_repeat' in runs:result['comparisons'][name+'__vs__repeat']=comparison(runs[name],runs[name+'_repeat'])
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');x=np.arange(len(names))
    for ax,key,title in zip(axes,('stepping_s','total_prepare_plus_worker_s'),('Hydraulic stepping','Complete preparation + worker')):
        vals=[result['timing_pairs'][n][key]['median'] for n in names];ax.bar(x,vals,color=['#d37634' if '_implicit_' in n else '#2c719f' for n in names])
        ax.set(xticks=x,xticklabels=[n.replace('_cfl0.225','').replace('_','\n') for n in names],ylabel='Wall time (s)',title=title+' (median; candidate CFL .225)')
        for xi,v in zip(x,vals):ax.text(xi,v,f'{v:.2f}',ha='center',va='bottom',fontsize=8)
    fig.savefig(OUT/'RUNTIME.png',dpi=160);plt.close(fig)
    from braids_gpu.atomic_output import write_json_atomic
    write_json_atomic(OUT/'METRICS.json',result)
    for name,row in result['runs'].items():
        print(name,'step',round(row['stepping_s'],3),'total',round(row['total_prepare_plus_worker_s'],2),'status',row['status']['status'])
    for name,row in result['comparisons'].items():print(name,'max/rms mm',round(row['worst_depth_mm'],4),round(row['worst_wet_rms_mm'],4))
    from braids_core.river_worker import source_identity
    from tools.build_unified_qgis import verify
    import hashlib
    frozen=read(BASE/'INPUTS.json');actual=source_identity()
    assert actual==frozen['runtime_id']
    assert all(v['record']['runtime_id']==actual for v in runs.values())
    plugin=Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis')
    manifest=verify(plugin)
    archive=Path(frozen['frozen_release']);release_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert release_sha==frozen['release_sha256']
    write_json_atomic(OUT/'INTEGRITY.json',dict(runtime_id=actual,production_source_unchanged=True,
        installed_version=manifest['version'],installed_manifest_files=len(manifest['files']),installed_manifest_verified=True,
        frozen_release_sha256=release_sha,plugin_replaced=False,
        all_runs_complete=all(v['record']['status']['status']=='complete' for v in runs.values()),
        all_runs_conservative=all(v['record']['conservation_pass'] for v in runs.values()),
        all_runs_nonnegative_finite=all(v['record']['finite'] and v['record']['minimum_depth_m']>=0 for v in runs.values())))


if __name__=='__main__':main()
