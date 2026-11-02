"""Finite cold-start rerun of the six-hour reference, with automatic figures.

No engine modification, checkpoint continuation, repeated polling or Delft run.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from . import run_case as r
from . import analyze_runs as a
from . import experiment as e
from . import curvature_multilevel as optimized
from . import background_run as background
from . import summarize_regularization as secondary
from . import equilibrium_mf25 as equilibrium
from . import mf25_validation_runtime as mf25

TAG = 'cold_optimized20h_20261001_v1'
REFERENCE = r.HERE/'runs/fine_mf25_equilibrium20h_v1_braids'
HYDRAULIC_REFERENCE = r.HERE/'runs/regularized_cold300_v1_fine_t8'
THREADS = 8
MATCH_KEYS = ('dx','Q','manning','feed_kg_s','porosity','d50_m','d90_m',
    'viscosity_m2_s','secondary_momentum_beta','longitudinal_slope_factor',
    'transverse_slope_factor','morfac','end','output','dt','cfl','scheme','cells')


def read(path):
    return json.loads(path.read_text())


def verify_inputs(folder):
    meta = read(folder/'input_provenance.json')
    for name, expected in meta['files'].items():
        if r.sha256(folder/name) != expected:
            raise ValueError(f'Changed reference input: {folder/name}')
    return meta


def assert_matching(left, right):
    for key in MATCH_KEYS:
        if not np.isclose(left[key], right[key], rtol=1e-12, atol=0):
            raise ValueError(f'Changed run setting: {key}')


def clone_cold_inputs(source, target):
    meta = verify_inputs(source)
    if not meta['hydraulic'] or meta['seed'] != 'None':
        raise ValueError('Warm-up must start from cold inputs, not a saved state')
    target.mkdir(parents=True, exist_ok=False)
    for name in meta['files']:
        shutil.copy2(source/name, target/name)
    meta.update(seed='None', cold_reference_folder=str(source),
        initialization='Exact original cold hydraulic input, no restart file',
        executable_identity='See this run execution.json; old input source hash is input provenance only')
    r.write_json(target/'input_provenance.json', meta)


def warmup_gates(data):
    record = secondary.evidence(data)
    gates = background.hydraulic_gates(record['metrics'])
    gates.update(secondary_budget=record['secondary_inventory_budget_max_residual'] < 1e-8,
        momentum_budget=record['momentum_budget_max_residual'] < 1e-8,
        secondary_active=record['bend_rms_intensity_m_s'] > 1e-4)
    return dict(gates=gates, passed=all(gates.values()), **record)


def compare_series(old, new):
    assert_matching(old['p'],new['p'])
    for key in ('xy','area','th'):
        np.testing.assert_allclose(old[key],new[key],rtol=0,atol=1e-10)
    np.testing.assert_allclose(old['z'][0],new['z'][0],rtol=0,atol=1e-12)
    result = dict(morphological_hours=(old['th']/3600).tolist())
    for label, left, right in (('bed',old['z'],new['z']),
            ('depth',old['h'],new['h']),('speed',old['speed'],new['speed'])):
        diff=right-left
        result[label+'_rms_difference'] = np.sqrt(np.average(diff**2,axis=1,weights=old['area'])).tolist()
        result[label+'_maximum_difference'] = np.max(abs(diff),axis=1).tolist()
    return result


def visuals(old, new, report, *, animation=True, qa_only=False):
    """Raw cell maps on shared scales, not interpolated morphological fields."""
    bed=[d['z']-d['z'][0] for d in (old,new)]
    limits=max(float(np.max(abs(x))) for x in bed)
    limit=max(limits,.001)
    difference=bed[1]-bed[0];difference_limit=max(float(np.max(abs(difference))),1e-6)
    banner='RENDER CHECK ONLY — historical data on both sides\n' if qa_only else ''
    new_label='Historical data repeated (QA)' if qa_only else 'Optimized cold rerun'
    hours=(1,5,10,20)
    ids=[int(np.argmin(abs(old['th']-hour*3600))) for hour in hours]
    fig,axes=plt.subplots(2,4,figsize=(15,6.7),layout='constrained')
    for row,(d,label) in enumerate(((old,'Earlier six-hour run'),(new,new_label))):
        for col,(hour,k) in enumerate(zip(hours,ids)):
            ax=axes[row,col]
            im=PolyCollection(d['polygons'],array=bed[row][k],cmap='RdBu_r',clim=(-limit,limit),edgecolors='none')
            ax.add_collection(im);ax.autoscale();ax.set_aspect('equal')
            ax.set(title=f'{label}\n{hour} morphological h',xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Bed change (m)',shrink=.8)
    fig.suptitle(banner+'Same initial bed, mesh, MF25 and forcing — positive is deposition')
    fig.savefig(report/'EVOLUTION.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,p in zip(axes,e.observations()['profiles']):
        n=np.array([v['relative_from_outer'] for v in p['points']])
        obs=np.array([v['bed_change_m'] for v in p['points']])
        station=np.full(len(n),e.APPROACH+e.RADIUS*np.deg2rad(p['angle_degrees']))
        ax.plot(n,obs,'ko',ms=3,label='Digitized experiment equilibrium')
        for row,(d,label,color) in enumerate(((old,'Earlier run','#777777'),(new,new_label,'#087f9a'))):
            values,_=a.sample(d,bed[row][-1],station,n*e.WIDTH)
            ax.plot(n,values,label=label,color=color)
        ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
        ax.set(title=f"{p['angle_degrees']} degree section",xlabel='Outer (0) to inner (1) wall',ylabel='Bed change (m)')
    axes[0].legend(fontsize=7)
    fig.suptitle(banner+'20 morphological hours — neither simulation is assumed to be equilibrium')
    fig.savefig(report/'FINAL_PROFILES.png',dpi=160);plt.close(fig)

    series=compare_series(old,new)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,key,unit,scale in zip(axes,('bed','depth','speed'),('mm','mm','m/s'),(1000,1000,1)):
        ax.plot(series['morphological_hours'],np.array(series[key+'_rms_difference'])*scale,color='#087f9a')
        ax.set(title=f'{key.capitalize()} difference',xlabel='Morphological hours',ylabel=f'Area-weighted RMS ({unit})',ylim=(0,None))
        ax.grid(alpha=.2)
    fig.suptitle(banner+new_label+' compared with the preserved earlier trajectory')
    fig.savefig(report/'DIFFERENCES.png',dpi=160);plt.close(fig)
    if not animation:
        return
    frame_dir=report/'animation_png';frame_dir.mkdir(exist_ok=qa_only)
    fig,axes=plt.subplots(1,3,figsize=(11,4.3),layout='constrained')
    collections=[]
    for ax,field,cap,title,unit in zip(axes,(bed[0],bed[1],difference*1000),
            (limit,limit,difference_limit*1000),('Earlier run',new_label,'New minus earlier'),('m','m','mm')):
        im=PolyCollection(old['polygons'],array=field[0],cmap='RdBu_r',clim=(-cap,cap),edgecolors='none')
        ax.add_collection(im);ax.autoscale();ax.set_aspect('equal');ax.set(title=title,xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(im,ax=ax,shrink=.65,label=f'Bed change ({unit})');collections.append(im)
    heading=fig.suptitle('')
    def update(k):
        for im,field in zip(collections,(bed[0],bed[1],difference*1000)):
            im.set_array(field[k])
        heading.set_text(banner+f"Bed evolution at MF25 — {old['th'][k]/3600:.2f} morphological hours")
        return [*collections,heading]
    for k in range(len(old['th'])):
        update(k);fig.savefig(frame_dir/f'frame_{k:04d}.png',dpi=110)
    movie=FuncAnimation(fig,update,frames=len(old['th']),interval=150,blit=False)
    movie.save(report/'EVOLUTION.gif',writer=PillowWriter(fps=7),dpi=95);plt.close(fig)


def finish(folder, report, old):
    new=a.load(folder)
    series=compare_series(old,new)
    accounts=secondary.evidence(new)
    stationarity=equilibrium.equilibrium_review(new)
    previous_s=old['execution']['wall_s']; current_s=new['execution']['wall_s']
    result=dict(series=series, new_accounts=accounts, new_stationarity=stationarity,
        old_mobile_wall_s=previous_s,new_mobile_wall_s=current_s,
        mobile_speedup=previous_s/current_s,mobile_wall_reduction_percent=100*(1-current_s/previous_s),
        wall_time_caveat='Historical versus current timings, not alternating paired measurements; warm-up time recorded separately',
        previous_run_stationary=False,experimental_accuracy_validated=False,
        delft_modified=False,production_promoted=False)
    r.write_json(report/'comparison.json',result)
    visuals(old,new,report)
    r.write(report/'RESULTS.md',f'''# Cold rerun compared with the earlier bend simulation

The optimized run completed from a new cold hydraulic warm-up and the original
uneroded bed, then simulated 20 morphological hours at MF25. It uses the same
5,012 cells, 0.10625 m spacing, forcing and ten-minute morphological output cadence.
At this size the hybrid filter uses MIC, not multilevel.

Mobile-bed wall time was {current_s/3600:.3f} h, compared with {previous_s/3600:.3f} h
historically ({previous_s/current_s:.3f} times speed). These are not alternating
paired performance measurements. Warm-up time is separate in status.json.

Final area-weighted bed RMS difference: {series['bed_rms_difference'][-1]*1000:.4f} mm.
New final three-hour stationarity screen: **{stationarity['passed']}**.
The earlier run failed stationarity; neither result is assumed to be truth.

[Evolution maps](EVOLUTION.png), [animation](EVOLUTION.gif),
[PNG animation frames](animation_png/), [final profiles](FINAL_PROFILES.png),
[difference histories](DIFFERENCES.png), [metrics and conservation](comparison.json).

Next: review global erosion/deposition, the differences through time, conservation
and experimental profiles before deciding whether the accumulated optimizations
can be adopted for the experimental bend workflow. No QGIS or Delft3D change.
''')
    return result


def run(tag=TAG):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):
        raise ValueError('Use a simple unique tag')
    report=r.HERE/'reports'/tag
    hydro=r.HERE/'runs'/f'{tag}_hydraulic_braids'
    mobile=r.HERE/'runs'/f'{tag}_mobile_braids'
    if any(x.exists() for x in (report,hydro,mobile)):
        raise FileExistsError('Refusing to overwrite a prior run')
    report.mkdir(parents=True)
    executions={}
    def status(state,phase,**extra):
        record=dict(status=state,phase=phase,supervisor_pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(),executions=executions,**extra)
        r.write_json(report/'status.json',record);print(json.dumps(record),flush=True)
    original_build=r.shared.verified_braids_build;original_writer=r.write_cut_case
    phase='preflight'
    try:
        status('preparing',phase)
        reference_meta=verify_inputs(REFERENCE);old=a.load(REFERENCE)
        exe,identity=optimized.runtime('hybrid')
        r.write_json(report/'build.json',identity)
        r.write_json(report/'reference.json',dict(folder=str(REFERENCE),
            input_provenance_sha256=r.sha256(REFERENCE/'input_provenance.json'),
            execution_sha256=r.sha256(REFERENCE/'execution.json'),
            frames_sha256=r.sha256(r.shared.native_frames_path(REFERENCE)),
            historical_mobile_wall_s=old['execution']['wall_s'],execution=old['execution']))
        r.write_json(report/'plan.json',dict(tag=tag,threads=THREADS,hydraulic_end_s=300,
            hydraulic_output_s=50,morphological_end_s=72000,morfac=25,
            mobile_hydraulic_end_s=2880,output_morphological_s=600,dx_m=.10625,cells=5012,
            initialization='Clone original cold warm-up inputs; rerun all 300 seconds. Seed mobile hydraulics only from this new run. No old checkpoint or evolved bed.',
            hydraulic_folder=str(hydro),mobile_folder=str(mobile),reference_folder=str(REFERENCE),
            solver_branch='MIC because 5012 < 65536; no refinement to force a different algorithm',
            solver_changes='Accumulated hydraulic/auxiliary implementation optimizations and practical MIC tolerance versus historical Jacobi; GeoClaw and sediment reconstruction retained',
            candidate_executable_sha256=identity['executable_sha256'],
            script_sha256=r.sha256(Path(__file__)),timeout_per_phase_s=172800,
            delft_executed=False,delft_modified=False,production_promoted=False,
            automatic_figures=['EVOLUTION.png','FINAL_PROFILES.png','DIFFERENCES.png','EVOLUTION.gif','animation_png/'],
            periodic_polling=False,auto_extension=False))
        r.write(report/'RUN.md','''# Cold rerun of the bend evolution case

This finite background job reruns the earlier 6.27-hour BRAIDS calculation from
the original cold hydraulic inputs. It does not continue an old checkpoint or
reuse an evolved bed. Check [current status](status.json) for the active phase.

First it runs 300 hydraulic seconds over the original fixed bed and checks
steady flow, discharge, depth and conservation. Only if those checks pass does
it start a continuous mobile-bed run: 2880 hydraulic seconds, MF25, representing
20 morphological hours. Initial secondary intensity is reset at the mobile-bed
transition, matching the historical protocol. The hydraulic seed is newly computed.

The mesh remains 5,012 cells at 0.10625 m nominal spacing. Discharge, sediment,
roughness, boundaries, timestep ceiling, reconstruction and output cadence match
the historical case. Eight OpenMP threads are requested. The size-aware filter
selects MIC on this mesh; this tests the accumulated optimizations, not the
large-mesh multilevel branch. Runtime and input identities are in [plan.json](plan.json),
[build.json](build.json) and [reference.json](reference.json).

Results are saved every ten morphological minutes. On successful completion the
job generates evolution maps at 1, 5, 10 and 20 hours, a GIF, all 121 PNG animation
frames, final cross-sections with the digitized experiment, difference histories,
conservation checks and a results report. The historical run did not meet its
stationarity criterion; 20 hours is a comparison duration, not assumed equilibrium.

The reference uses the historical Jacobi filter; the optimized build retains
the same curvature equation with the previously qualified practical MIC residual
target. Long-run results need not be bitwise identical. Warm-up and mobile times
are recorded separately; historical wall-clock comparison is not an alternating
paired benchmark. No new speed or accuracy result is claimed before completion.

No Delft3D source/runtime or QGIS installation is modified. There is no periodic
monitoring loop and no automatic extension beyond 20 morphological hours.
Keep this PC awake while the job runs. The next step is to inspect the completed
evolution, global erosion/deposition, profiles and conservation before adoption.
''')
        r.shared.verified_braids_build=lambda:(optimized.runtime('hybrid')[0],identity)
        r.write_cut_case=mf25.validation_writer(report)
        phase='hydraulic';clone_cold_inputs(HYDRAULIC_REFERENCE,hydro)
        status('running',phase,folder=str(hydro))
        executions[phase]=r.execute_native(hydro,THREADS,timeout_s=172800)
        water=a.load(hydro);check=warmup_gates(water)
        r.write_json(report/'hydraulic_checks.json',check)
        if not check['passed']:
            raise RuntimeError(f"Cold warm-up failed: {check['gates']}")
        np.testing.assert_allclose(water['z'][-1],water['z'][0],rtol=0,atol=1e-12)
        phase='mobile';status('preparing',phase,folder=str(mobile))
        r.prepare_native(mobile,dx=.10625,end=2880,output=24,dt=.0125,
            morfac=25,hydraulic=False,seed=hydro)
        meta=read(mobile/'input_provenance.json');assert_matching(reference_meta,meta)
        for key in ('xy','area','bed'):
            np.testing.assert_allclose(np.load(REFERENCE/'analysis_geometry.npz')[key],
                np.load(mobile/'analysis_geometry.npz')[key],rtol=0,atol=1e-12)
        meta.update(engine_source_modified=True,production_source_modified=False,
            candidate='Frozen size-aware filter and current hydraulic optimizations; same experimental ell=2h closure',
            curvature_regularization_depth_factor=2.,hydraulic_seed_frames_sha256=r.sha256(r.shared.native_frames_path(hydro)),
            comparison_reference=str(REFERENCE),no_evolved_bed_restart=True)
        r.write_json(mobile/'input_provenance.json',meta)
        status('running',phase,folder=str(mobile))
        executions[phase]=r.execute_native(mobile,THREADS,timeout_s=172800)
        phase='analysis';status('analyzing',phase,folder=str(mobile))
        result=finish(mobile,report,old)
        status('complete',phase,report=str(report/'RESULTS.md'),stationary=result['new_stationarity']['passed'])
    except Exception as exc:
        status('failed',phase,error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r.shared.verified_braids_build=original_build;r.write_cut_case=original_writer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',default=TAG)
    run(parser.parse_args().tag)
