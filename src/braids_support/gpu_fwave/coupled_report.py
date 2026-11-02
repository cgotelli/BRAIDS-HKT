"""Scientific CPU/GPU coupled evolution figures and auditable error summaries.

Only existing validation outputs are read. This never edits a case, changes a
solver parameter, or interprets a short transient as equilibrium morphology.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from . import coupled_reference as ref

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_rms(values,area):
    return np.sqrt(np.sum(np.asarray(values)**2*area,axis=-1)/np.sum(area))


def evolution_metrics(arrays,area,dry):
    time=np.asarray(arrays['times']);bed=np.asarray(arrays['bed']);native=np.asarray(arrays['native_bed'])
    q=np.asarray(arrays['q']);nq=np.asarray(arrays['native_q']);area=np.asarray(area)
    if time.ndim!=1 or bed.shape!=native.shape or bed.shape!=(len(time),len(area)) or q.shape!=nq.shape or q.shape!=(len(time),3,len(area)):
        raise ValueError('Incompatible coupled trajectory shapes')
    if not all(np.isfinite(v).all() for v in (time,bed,native,q,nq,area)) or np.any(area<=0):
        raise ValueError('Nonfinite fields or nonpositive cell areas')
    if len(time)<2 or time[0]!=0 or np.any(np.diff(time)<=0):raise ValueError('At least two strictly increasing times from zero required')
    if np.max(abs(bed[0]-native[0]))>1e-12 or np.max(abs(q[0]-nq[0]))>1e-12:
        raise ValueError('CPU/GPU initial conditions differ')
    dz=bed-bed[0];ndz=native-native[0];delta=bed-native
    uv=np.divide(q[:,1:3],q[:,0,None,:],out=np.zeros_like(q[:,1:3]),where=q[:,0,None,:]>dry)
    nuv=np.divide(nq[:,1:3],nq[:,0,None,:],out=np.zeros_like(nq[:,1:3]),where=nq[:,0,None,:]>dry)
    velocity_error=np.sqrt(np.sum((uv-nuv)**2,axis=1))
    speed=np.sqrt(np.sum(uv**2,axis=1));native_speed=np.sqrt(np.sum(nuv**2,axis=1))
    curves=dict(times=time,bed_max_mm=np.max(abs(delta),axis=1)*1000.,bed_rms_mm=weighted_rms(delta,area)*1000.,
                depth_max_mm=np.max(abs(q[:,0]-nq[:,0]),axis=1)*1000.,depth_rms_mm=weighted_rms(q[:,0]-nq[:,0],area)*1000.,
                velocity_max_m_s=np.max(velocity_error,axis=1),velocity_rms_m_s=weighted_rms(velocity_error,area),
                gpu_deposition_m3=np.maximum(dz,0.)@area,gpu_erosion_m3=-np.minimum(dz,0.)@area,
                native_deposition_m3=np.maximum(ndz,0.)@area,native_erosion_m3=-np.minimum(ndz,0.)@area,
                gpu_deposition_max_mm=np.maximum(dz,0.).max(axis=1)*1000.,gpu_erosion_max_mm=-np.minimum(dz,0.).min(axis=1)*1000.,
                native_deposition_max_mm=np.maximum(ndz,0.).max(axis=1)*1000.,native_erosion_max_mm=-np.minimum(ndz,0.).min(axis=1)*1000.,
                wet_dry_disagreement_cells=np.sum((q[:,0]>dry)!=(nq[:,0]>dry),axis=1))
    summary={key:dict(final=float(v[-1]),maximum_over_outputs=float(np.max(v))) for key,v in curves.items() if key!='times'}
    summary.update(final_time_s=float(time[-1]),output_frames=len(time),cells=len(area),domain_area_m2=float(area.sum()),
                   initial_conditions_matched=True,bed_change_sign='positive deposition; negative erosion',
                   volume_basis='bulk bed volume: integral of elevation change, including porosity; not solid grain volume',
                   velocity_error_basis='magnitude of vector difference; depth <= dry tolerance has zero reported velocity')
    return summary,curves,dict(dz=dz,native_dz=ndz,bed_error=delta,speed=speed,native_speed=native_speed,velocity_error=velocity_error)


def polygons(case,data):
    with np.load(Path(case)/'geometry.npz',allow_pickle=False) as geometry:
        offset=geometry['vertex_offsets'];vertices=geometry['vertices_xy'];centres=geometry['cell_scalar'][:,1:3]
        if not np.allclose(centres,data['xy'].T,rtol=0,atol=1e-11):raise ValueError('Geometry cell ordering does not match native states')
        if len(offset)!=data['nc']+1:raise ValueError('Geometry count mismatch')
        if np.any(np.diff(geometry['cell_hole_offsets'])):raise ValueError('Polygon holes need explicit rendering support')
        result=[]
        for c in range(data['nc']):
            polygon=vertices[offset[c]:offset[c+1]].copy()
            # Native compact geometry deliberately omits ordinary Cartesian
            # rectangle vertices. Empty records are not holes or missing cells.
            if not len(polygon):
                width=np.asarray(data['rectangle_width'])[:,c]
                if not geometry['regular_cell'][c] or np.any(width<=0) or not np.isclose(np.prod(width),data['area'][c],rtol=1e-10,atol=1e-13):
                    raise ValueError('Cannot reconstruct an omitted regular-cell polygon')
                xx,yy=centres[c];hx,hy=width/2.
                polygon=np.array([[xx-hx,yy-hy],[xx+hx,yy-hy],[xx+hx,yy+hy],[xx-hx,yy+hy]])
            result.append(polygon)
        return result


def spatial_audit(arrays,data):
    """Comparative outlet and cell-scale diagnostics, not a physical truth test."""
    area=data['area'];xy=data['xy'];nc=len(area);dx,dy=data['dx'],data['dy']
    dz=(arrays['bed'][-1]-arrays['bed'][0])*1000.
    native=(arrays['native_bed'][-1]-arrays['native_bed'][0])*1000.
    neighbours=[[] for _ in range(nc)]
    for c,n in zip(data['owner'],data['neighbour']):
        if n>=0:neighbours[c].append(int(n));neighbours[n].append(int(c))
    core=[];stencils=[]
    for c,near in enumerate(neighbours):
        choices=[]
        for shift in ((dx,0.),(-dx,0.),(0.,dy),(0.,-dy)):
            match=[n for n in near if np.max(abs((xy[:,n]-xy[:,c])-shift))<1e-10]
            if len(match)!=1:break
            choices.append(match[0])
        if len(choices)==4:core.append(c);stencils.append(choices)
    core=np.asarray(core,int);stencils=np.asarray(stencils,int)
    if not len(core):raise ValueError('No certified equal-spacing Cartesian cells for checkerboard diagnostic')
    high=dz[core]-dz[stencils].mean(axis=1);native_high=native[core]-native[stencils].mean(axis=1)
    grid=np.rint((xy[:,core]-xy[:,core[:1]])/np.array([dx,dy])[:,None]).astype(np.int64)
    parity=np.where(np.sum(grid,axis=0)%2,1.,-1.)
    w=area[core];parity=parity-np.sum(parity*w)/w.sum()
    amplitude=lambda z:float(np.sum(z[core]*parity*w)/np.sum(parity**2*w))
    audit=dict(core_cells=len(core),highpass_definition='cell bed change minus mean of four equal-spacing cardinal neighbours',
        highpass_gpu_rms_mm=float(weighted_rms(high,w)),highpass_cpu_rms_mm=float(weighted_rms(native_high,w)),
        highpass_difference_rms_mm=float(weighted_rms(high-native_high,w)),
        checkerboard_projection_gpu_mm=amplitude(dz),checkerboard_projection_cpu_mm=amplitude(native),
        checkerboard_projection_difference_mm=amplitude(dz-native),
        interpretation='These measure grid-scale content and GPU/CPU differences; neither alone proves absence of numerical oscillations.',outlets=[])
    for boundary,mode in enumerate(data['mode']):
        if mode not in (2,3):continue
        owners=np.unique(data['owner'][data['bid']==boundary]);distance=np.full(nc,-1,int);distance[owners]=0
        wave=set(map(int,owners))
        for level in range(1,10):
            new={n for c in wave for n in neighbours[c] if distance[n]<0}
            if not new:break
            distance[list(new)]=level;wave=new
        rows=[]
        for level in range(10):
            mask=distance==level
            if not mask.any():continue
            rows.append(dict(adjacency_ring=level,cells=int(mask.sum()),gpu_mean_mm=float(np.average(dz[mask],weights=area[mask])),
                             cpu_mean_mm=float(np.average(native[mask],weights=area[mask])),
                             gpu_min_mm=float(dz[mask].min()),gpu_max_mm=float(dz[mask].max()),
                             cpu_min_mm=float(native[mask].min()),cpu_max_mm=float(native[mask].max())))
        audit['outlets'].append(dict(boundary=int(boundary),hydraulic_mode=int(mode),rows=rows,
            boundary_max_deposition_mm=float(np.maximum(dz[owners],0.).max()),
            native_boundary_max_deposition_mm=float(np.maximum(native[owners],0.).max()),
            boundary_max_gpu_cpu_difference_mm=float(abs(dz[owners]-native[owners]).max())))
    return audit,dict(core=core,highpass=high,native_highpass=native_high)


def make_report(report_folder,prepared_case=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    report_folder=Path(report_folder);prepared_case=Path(prepared_case or HERE/'runs'/ref.TAG/'bend')
    data=ref.read_snapshot(prepared_case)
    with np.load(report_folder/'trajectory.npz',allow_pickle=False) as archive:
        # Full layer/ledger histories are large and are not needed by these
        # field and already-computed budget diagnostics.
        required=('times','q','native_q','bed','native_bed','water_residual','sediment_residual')
        arrays={name:archive[name] for name in required}
    validation=json.loads((report_folder/'braids_support.json').read_text())
    summary,curves,field=evolution_metrics(arrays,data['area'],data['dry_tolerance'])
    audit,audit_fields=spatial_audit(arrays,data)
    cells=polygons(prepared_case,data);times=curves['times'];mf=float(data['morphological_factor'])
    x=data['xy'][0];y=data['xy'][1];extent=(x.min()-.1,x.max()+.1,y.min()-.1,y.max()+.1)
    duration=f"hydraulic t = {times[-1]:g} s; MorFac {mf:g}"
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'savefig.dpi':165})

    def map_panel(ax,value,title,cmap,vmin,vmax,label):
        poly=PolyCollection(cells,array=value,cmap=cmap,edgecolors='none',linewidth=0,antialiased=False,rasterized=True)
        poly.set_clim(vmin,vmax);ax.add_collection(poly);ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        bar=ax.figure.colorbar(poly,ax=ax,shrink=.85,pad=.025);bar.set_label(label)

    figures=[]
    fig,axes=plt.subplots(1,3,figsize=(17,5.0),layout='constrained')
    scale=max(float(np.max(abs(field['dz'][-1]))),float(np.max(abs(field['native_dz'][-1]))))*1000.
    scale=max(scale,1e-12);difference=max(float(np.max(abs(field['bed_error'][-1])))*1000.,1e-12)
    map_panel(axes[0],field['native_dz'][-1]*1000.,'Frozen CPU: bed change','RdBu_r',-scale,scale,'Bed change (mm)')
    map_panel(axes[1],field['dz'][-1]*1000.,'CUDA: bed change','RdBu_r',-scale,scale,'Bed change (mm)')
    map_panel(axes[2],field['bed_error'][-1]*1000.,'CUDA − CPU: separate error scale','RdBu_r',-difference,difference,'Bed elevation error (mm)')
    fig.suptitle('Coupled MPM bend: erosion (blue) and deposition (red)\n'+duration,fontsize=14)
    name='COUPLED_BED_CPU_GPU.png';fig.savefig(report_folder/name);plt.close(fig);figures.append(name)

    fig,axes=plt.subplots(2,3,figsize=(17,9.0),layout='constrained')
    depthscale=max(float(arrays['q'][-1,0].max()),float(arrays['native_q'][-1,0].max()),1e-15)
    err=arrays['q'][-1,0]-arrays['native_q'][-1,0];errscale=max(float(abs(err).max())*1000.,1e-12)
    map_panel(axes[0,0],arrays['native_q'][-1,0],'Frozen CPU: water depth','Blues',0,depthscale,'Depth (m)')
    map_panel(axes[0,1],arrays['q'][-1,0],'CUDA: water depth','Blues',0,depthscale,'Depth (m)')
    map_panel(axes[0,2],err*1000.,'CUDA − CPU: depth error','RdBu_r',-errscale,errscale,'Depth error (mm)')
    speedscale=max(float(field['speed'][-1].max()),float(field['native_speed'][-1].max()),1e-15)
    velerr=max(float(field['velocity_error'][-1].max()),1e-15)
    map_panel(axes[1,0],field['native_speed'][-1],'Frozen CPU: speed','Reds',0,speedscale,'Speed (m/s)')
    map_panel(axes[1,1],field['speed'][-1],'CUDA: speed','Reds',0,speedscale,'Speed (m/s)')
    map_panel(axes[1,2],field['velocity_error'][-1],'Magnitude of velocity-vector error','magma',0,velerr,'Vector error (m/s)')
    fig.suptitle('Matched coupled hydraulics\n'+duration,fontsize=14)
    name='COUPLED_HYDRAULICS_CPU_GPU.png';fig.savefig(report_folder/name);plt.close(fig);figures.append(name)

    fig,axes=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    for prefix,style,label in [('gpu','-','CUDA'),('native','--','CPU')]:
        axes[0,0].plot(times,curves[prefix+'_erosion_max_mm'],style,color='#2678ad',label=label+' erosion')
        axes[0,0].plot(times,curves[prefix+'_deposition_max_mm'],style,color='#d34d36',label=label+' deposition')
        axes[0,1].plot(times,curves[prefix+'_erosion_m3'],style,color='#2678ad',label=label+' erosion')
        axes[0,1].plot(times,curves[prefix+'_deposition_m3'],style,color='#d34d36',label=label+' deposition')
    axes[0,0].set(title='Resolved erosion/deposition extrema',ylabel='Change magnitude (mm)')
    axes[0,1].set(title='Integrated bed-volume redistribution',ylabel='Bulk bed volume (m³)')
    for ax,kind in [(axes[1,0],'bed'),(axes[1,1],'depth')]:
        ax.plot(times,curves[kind+'_max_mm'],label='Maximum absolute error',color='#a4372d')
        ax.plot(times,curves[kind+'_rms_mm'],label='Area-weighted RMS error',color='#276fa1')
        ax.set(title=('Bed elevation' if kind=='bed' else 'Water depth')+' agreement',ylabel='CUDA − CPU error (mm)')
    axes[2,0].plot(times,arrays['water_residual'],color='#276fa1',label='Water inventory − boundary volume')
    axes[2,0].set(title='GPU water balance',ylabel='Residual (m³)')
    solids=np.asarray(arrays['sediment_residual'])
    if solids.ndim==1:solids=solids[:,None]
    for g in range(solids.shape[1]):axes[2,1].plot(times,solids[:,g],label=f'Label {g+1}')
    axes[2,1].plot(times,solids.sum(axis=1),'k--',label='All labels')
    axes[2,1].set(title='GPU solid-mass balance, including both layers',ylabel='Residual (kg)')
    for row,axrow in enumerate(axes):
        for ax in axrow:
            ax.set_xlabel('Hydraulic time (s)');ax.grid(alpha=.22);ax.legend(fontsize=8)
            if row<2:ax.set_ylim(bottom=0)
            else:ax.axhline(0,color='black',linewidth=.6)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4),useMathText=True)
    fig.suptitle('Coupled evolution and numerical checks\n'+duration+'; short transient, not equilibrium or bar-formation evidence',fontsize=14)
    name='COUPLED_EVOLUTION_AND_BUDGET.png';fig.savefig(report_folder/name);plt.close(fig);figures.append(name)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for outlet in audit['outlets']:
        rows=outlet['rows'];rings=[row['adjacency_ring'] for row in rows]
        axes[0].plot(rings,[row['gpu_mean_mm'] for row in rows],'-o',label=f"CUDA boundary {outlet['boundary']+1}",markersize=4)
        axes[0].plot(rings,[row['cpu_mean_mm'] for row in rows],'--',label=f"CPU boundary {outlet['boundary']+1}")
    axes[0].set(title='Outlet-adjacent bed change',xlabel='Cell-adjacency ring inward (0 = outlet owners)',ylabel='Area-weighted mean bed change (mm)')
    axes[0].axhline(0,color='black',linewidth=.7);axes[0].grid(alpha=.2);axes[0].legend()
    maximum=max(abs(audit_fields['highpass']).max(),abs(audit_fields['native_highpass']).max(),1e-12)
    bins=np.linspace(-maximum,maximum,81)
    axes[1].hist(audit_fields['highpass'],bins=bins,histtype='step',label='CUDA',linewidth=1.5)
    axes[1].hist(audit_fields['native_highpass'],bins=bins,histtype='step',label='CPU',linestyle='--',linewidth=1.5)
    axes[1].set(title='Equal-spacing interior: cell-scale bed-change content',xlabel='Cell minus four-neighbour mean (mm)',ylabel='Cell count',yscale='log')
    axes[1].grid(alpha=.2);axes[1].legend()
    fig.suptitle('Outlet and grid-scale audit: comparison, not proof of physical accuracy\n'+duration,fontsize=13)
    name='COUPLED_OUTLET_AND_GRIDCHECK.png';fig.savefig(report_folder/name);plt.close(fig);figures.append(name)

    timing=braids_support.get('timing')
    timing_caveat=(timing['warning'] if timing else
                   'Timing may include full-field validation downloads and prescribed CPU timestep replay; no isolated speed claim.')
    timing_text=(f"Whole-process screening: CPU {timing['cpu_process_s']:.6g} s / GPU {timing['gpu_process_s']:.6g} s; "
                 f"ratio {timing['cpu_to_gpu_process_ratio']:.6g}. {timing_caveat}" if timing else
                 f"GPU setup: {braids_support.get('setup_s','not recorded')} s; evolution including validation downloads: "
                 f"{braids_support.get('evolution_with_validation_download_s','not recorded')} s.")
    summary.update(validation_passed=bool(braids_support.get('passed',False)),spatial_audit=audit,
                   source_validation=validation,source_trajectory_sha256=sha(report_folder/'trajectory.npz'),
                   source_geometry_sha256=sha(prepared_case/'geometry.npz'),report_driver_sha256=sha(__file__),
                   morfac=mf,morphological_time_s=float(times[-1])*mf,
                   performance_qualified=bool(braids_support.get('performance_qualified',False)),
                   figures=[dict(file=name,sha256=sha(report_folder/name)) for name in figures],
                   limitations=['Short transient comparison, not equilibrium or experimental braids_support.',
                                'A visible bed-change field alone does not establish bars, meandering or braiding.',
                                timing_caveat,
                                'Error maps use their own clearly labelled scales; CPU/GPU physical maps use identical scales.'])
    (report_folder/'EVOLUTION_METRICS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    np.savez_compressed(report_folder/'evolution_metrics.npz',**curves)
    final=lambda name:summary[name]['final']
    text=[f'Coupled CUDA / frozen CPU comparison: {duration}',
          f"Cells: {len(data['area'])}; output frames: {len(times)}; validation passed: {summary['validation_passed']}",
          f"Bed error at final output: max {final('bed_max_mm'):.8g} mm; area-weighted RMS {final('bed_rms_mm'):.8g} mm.",
          f"Depth error at final output: max {final('depth_max_mm'):.8g} mm; area-weighted RMS {final('depth_rms_mm'):.8g} mm.",
          f"Velocity-vector error at final output: max {final('velocity_max_m_s'):.8g} m/s; area-weighted RMS {final('velocity_rms_m_s'):.8g} m/s.",
          f"CUDA erosion/deposition extrema: {final('gpu_erosion_max_mm'):.8g} / {final('gpu_deposition_max_mm'):.8g} mm.",
          f"CPU erosion/deposition extrema: {final('native_erosion_max_mm'):.8g} / {final('native_deposition_max_mm'):.8g} mm.",
          f"CUDA eroded/deposited bulk bed volume: {final('gpu_erosion_m3'):.8g} / {final('gpu_deposition_m3'):.8g} m3.",
          f"CPU eroded/deposited bulk bed volume: {final('native_erosion_m3'):.8g} / {final('native_deposition_m3'):.8g} m3.",
          f"Maximum GPU water/solid balance residual: {np.max(abs(arrays['water_residual'])):.8g} m3 / {np.max(abs(solids)):.8g} kg.",
          timing_text,
          f"Interior cell-scale high-pass RMS CUDA/CPU: {audit['highpass_gpu_rms_mm']:.8g} / {audit['highpass_cpu_rms_mm']:.8g} mm; RMS difference {audit['highpass_difference_rms_mm']:.8g} mm.",
          f"Signed checkerboard projection CUDA/CPU: {audit['checkerboard_projection_gpu_mm']:.8g} / {audit['checkerboard_projection_cpu_mm']:.8g} mm.",
          'No isolated CPU/GPU speedup is claimed by this report.', '', *summary['limitations'], '', 'Figures:',*figures]
    (report_folder/'EVOLUTION_REPORT.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ('final_time_s','bed_max_mm','bed_rms_mm','gpu_erosion_max_mm','gpu_deposition_max_mm','validation_passed')},indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--case',type=Path)
    args=p.parse_args();make_report(HERE/'reports'/args.tag,args.case)
