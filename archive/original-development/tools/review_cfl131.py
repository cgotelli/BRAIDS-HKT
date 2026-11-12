"""Read-only reassessment of archived speed130 CFL trials; no solver execution."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'validation/_speed130/closed2d_v2'
OUT = ROOT / 'validation/_amr131/cfl_review'


def load(name):
    folder = (OUT if name.endswith('_combined_v2') else SOURCE) / name / 'run/frames'
    index = json.loads((folder / 'INDEX.json').read_text())
    for filename, checksum in index['epochs'].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == checksum
    frames = []
    for row in index['frames']:
        assert hashlib.sha256((folder / row['file']).read_bytes()).hexdigest() == row['sha256']
        with np.load(folder / row['file']) as item:
            frames.append({key: item[key].copy() for key in item.files})
    with np.load(folder / index['frames'][0]['epoch']) as mesh:
        area = mesh['area'].copy()
        xy = np.array([mesh['vertices'][mesh['offsets'][i]:mesh['offsets'][i+1]-1].mean(0)
                       for i in range(len(area))])
    return np.array([x['time_s'] for x in index['frames']]), frames, area, xy


def compare(name, refname):
    times, frames, area, xy = load(name)
    rt, refs, ra, rxy = load(refname)
    assert np.array_equal(times, rt) and np.array_equal(area, ra) and np.array_equal(xy, rxy)
    h = np.array([f['q'][0] for f in frames]); rh = np.array([f['q'][0] for f in refs])
    err = h-rh; ae = abs(err)
    t, cell = np.unravel_index(ae.argmax(), ae.shape)
    history = []
    for i in range(len(times)):
        row = dict(time_s=float(times[i]), depth_max_mm=float(ae[i].max()*1000),
                   depth_area_rms_mm=float(np.sqrt(np.average(err[i]**2, weights=area))*1000),
                   area_fraction_error_above1mm=float(area[ae[i]>.001].sum()/area.sum()),
                   area_fraction_error_above5mm=float(area[ae[i]>.005].sum()/area.sum()),
                   maximum_depth_m=float(h[i].max()), reference_maximum_depth_m=float(rh[i].max()),
                   signed_volume_difference_m3=float(err[i]@area),
                   absolute_water_redistribution_m3=float(ae[i]@area/2))
        for threshold in (.001, .01):
            wet=h[i]>threshold; rwet=rh[i]>threshold
            row[f'front_diagonal_difference_{threshold:g}_m'] = float(
                (xy[wet].sum(1).max(initial=0)-xy[rwet].sum(1).max(initial=0))/np.sqrt(2))
            row[f'wet_symmetric_area_{threshold:g}_m2'] = float(area[wet!=rwet].sum())
        for threshold in (.01, .05, .1):
            both=(h[i]>threshold)&(rh[i]>threshold)
            dv=np.linalg.norm(frames[i]['q'][1:,both]/h[i,both]-refs[i]['q'][1:,both]/rh[i,both],axis=0)
            row[f'velocity_max_both_deeper_{threshold:g}_m_s']=float(dv.max(initial=0))
        v=np.divide(frames[i]['q'][1:],h[i],out=np.zeros((2,len(area))),where=h[i]>1e-6)
        rv=np.divide(refs[i]['q'][1:],rh[i],out=np.zeros((2,len(area))),where=rh[i]>1e-6)
        both=(h[i]>.01)&(rh[i]>.01)
        dv=np.linalg.norm(v-rv,axis=0)
        vidx=int(np.where(both,dv,0).argmax())
        row['worst_velocity_cell']=dict(cell=vidx,depths_m=[float(h[i,vidx]),float(rh[i,vidx])],
            candidate_velocity_m_s=v[:,vidx].tolist(),reference_velocity_m_s=rv[:,vidx].tolist())
        row['velocity_water_weighted_rms_both_deeper1cm_m_s']=float(np.sqrt(np.average(dv[both]**2,weights=area[both]*rh[i,both])))
        row['maximum_speed_m_s']=float(np.linalg.norm(v,axis=0).max())
        row['reference_maximum_speed_m_s']=float(np.linalg.norm(rv,axis=0).max())
        ke=float((.5*h[i]*np.sum(v*v,axis=0)*area).sum())
        rke=float((.5*rh[i]*np.sum(rv*rv,axis=0)*area).sum())
        row['kinetic_energy_difference_percent']=100*(ke-rke)/rke if rke else 0.
        history.append(row)
    arrival = {}
    for threshold in (.001, .01):
        wet=h>threshold; rwet=rh>threshold
        initially_dry=rh[0]<=threshold
        both=np.any(wet, axis=0)&np.any(rwet,axis=0)&initially_dry
        at=times[np.argmax(wet,axis=0)]; rat=times[np.argmax(rwet,axis=0)]
        delta=at[both]-rat[both]
        arrival[str(threshold)]=dict(common_newly_wet_cells=int(both.sum()),
            different_first_saved_wet_frame_cells=int(np.count_nonzero(delta)),
            maximum_first_saved_wet_time_difference_s=float(abs(delta).max(initial=0)),
            sample_spacing_s=.2,
            caution='First saved threshold crossing, not resolved physical arrival time')
    return dict(candidate=name, reference=refname, domain_area_m2=float(area.sum()),
        worst=dict(time_s=float(times[t]), cell=int(cell), xy_approx_m=xy[cell].tolist(),
            area_m2=float(area[cell]), candidate_depth_m=float(h[t,cell]),
            reference_depth_m=float(rh[t,cell]), error_mm=float(ae[t,cell]*1000),
            relative_to_reference_depth_percent=float(100*ae[t,cell]/rh[t,cell]),
            initially_wet=bool(rh[0,cell]>.001)),
        history=history, sampled_arrival=arrival)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results={f'{name}__{ref}':compare(name,ref) for name,ref in (
        ('dam_cpu_cfl1','dam_cpu_cfl0.125'),('dam_cpu_cfl1','dam_cpu_cfl0.5'),
        ('dam_gpu_cfl1','dam_cpu_cfl1'),
        ('dam_cpu_cfl1_combined_v2','dam_cpu_cfl0.125'),
        ('dam_gpu_cfl1_combined_v2','dam_cpu_cfl1_combined_v2'))}
    replay = {}
    for backend in ('cpu','gpu'):
        record=json.loads((OUT/f'dam_{backend}_cfl1_combined_v2/RESEARCH.json').read_text())
        replay[backend]=dict(source_runtime=record['source_runtime'],source_unchanged=record['source_unchanged'],
            actual_controls=record['research'],status=record['status'],
            maximum_water_residual_m3=max(abs(r['water_residual_m3']) for r in record['history']),
            maximum_wall_ledger_m3=max(abs(r['wall_integral']) for r in record['history']),
            minimum_depth_m=min(r['minimum_depth_m'] for r in record['history']))
    summary=dict(scope='Archived 2-second diagonal dam review plus current-source combined-clock CPU/GPU CFL1 replay',
        reference='Same spatial discretization at CFL 0.125, not experiment or analytical truth',
        all_frame_and_mesh_checksums_verified=True, comparisons=results, current_replay=replay,
        earlier_failed_attempt='dam_cpu_cfl1_combined rejected before solving because runtime changed during preparation; retained. CPU/GPU v2 used one unchanged source identity.',
        recommendation=dict(hydro_cfl1='Accept practical flood-depth/wet-area tradeoff under the user updated tolerance; may expose up to1 and use1 for fresh simple-hydrodynamics projects after production identity/AMR tests.',
            scope='Existing pure hydrodynamics mode: no active sediment, secondary-flow feedback, diffusivity or viscosity; user-selected CPU or GPU.',
            combined_cfl='Keep explicit0.5 limit until coupled qualification; never silently clamp a saved user value.',
            caution='Impulsive dam-break early shallow-water velocity differs materially (0.249m/s local,22.35% initial kinetic energy deficit), decaying rapidly. Not universally byte-identical or exact-arrival validated.',
            no_dry_threshold_change='Keep1e-6m; higher thresholds provided no speed benefit in previous natural trials.',
            no_blanket_gate='Do not reject CFL1 solely because one maxdepth difference is7.5mm; retain positivity/conservation/hardstage/restart checks and assess physical trend.'))
    (OUT/'REVIEW.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for name,style in [('dam_cpu_cfl0.125','-'),('dam_cpu_cfl0.5','--'),('dam_cpu_cfl1','-.'),
                       ('dam_cpu_cfl1_combined_v2',':')]:
        times,frames,area,xy=load(name)
        h=np.array([f['q'][0] for f in frames])
        axes[0,0].plot(times, h[:,240]*1000,style,label=name.replace('dam_cpu_cfl','CFL ').replace('_combined_v2',' + current clock'))
    axes[0,0].set(title='Depth at the worst-difference wall cell',xlabel='Time (s)',ylabel='Water depth (mm)')
    axes[0,0].set_ylim(bottom=0)
    axes[0,0].legend()
    row=results['dam_cpu_cfl1__dam_cpu_cfl0.125'];hist=row['history'];t=[h['time_s'] for h in hist]
    axes[0,1].plot(t,[h['depth_max_mm'] for h in hist],label='Maximum')
    axes[0,1].plot(t,[h['depth_area_rms_mm'] for h in hist],label='Area-weighted RMS')
    current=results['dam_cpu_cfl1_combined_v2__dam_cpu_cfl0.125']['history']
    axes[0,1].plot(t,[h['depth_max_mm'] for h in current],':',label='Current-clock maximum')
    axes[0,1].set(title='CFL 1 vs fine-time control',xlabel='Time (s)',ylabel='Depth difference (mm)')
    axes[0,1].legend()
    axes[1,0].plot(t,[h['area_fraction_error_above1mm']*100 for h in hist],label='Above 1 mm')
    axes[1,0].plot(t,[h['area_fraction_error_above5mm']*100 for h in hist],label='Above 5 mm')
    axes[1,0].set(title='Spatial extent of the difference',xlabel='Time (s)',ylabel='Domain area (%)')
    axes[1,0].legend()
    axes[1,1].plot(t,[h['wet_symmetric_area_0.001_m2'] for h in hist],label='1 mm wet threshold')
    axes[1,1].plot(t,[h['wet_symmetric_area_0.01_m2'] for h in hist],label='10 mm wet threshold')
    axes[1,1].set(title='Area classified wet by only one solution',xlabel='Time (s)',ylabel='Area (m²)')
    axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.25)
    fig.suptitle('Practical impact of the archived CFL 1 trial: stable, conservative; local depth differences')
    fig.savefig(OUT/'CFL1_IMPACT.png',dpi=170)
    print(str(OUT/'REVIEW.json'))


if __name__=='__main__':main()
