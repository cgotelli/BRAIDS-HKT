"""Continuous MF25 equilibrium attempt; unchanged candidate and stock reference.

Finite background run, no status-polling loop. User accepted the measured MF25
error on 2026-09-30; the failed earlier strict gate is not retroactively changed.
"""
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import run_case as r
from . import analyze_runs as a
from . import experiment as e
from . import mf25_pair as control
from . import mf25_validation_runtime as runtime_builder
from . import regularized_mobile as mobile
from . import review_regularized_pair as review

TAG='fine_mf25_equilibrium20h_v1'
MORPH_END=20*3600.
FACTOR=25.


def stationarity(times,bed,areas,profiles,windows=3,tolerance=.001):
    """Require each of the final three hours to pass for bend and every section.

    Maximum saved change relative to each window's start also rejects an
    oscillation that returns to its starting bed at the end of the hour.
    """
    times=np.asarray(times);bed=np.asarray(bed);areas=np.asarray(areas)
    profiles=[np.asarray(p) for p in profiles]
    if not np.isfinite(bed).all() or not all(np.isfinite(p).all() for p in profiles):
        raise ValueError('Nonfinite stationarity input')
    if len(times)!=len(bed) or np.any(np.diff(times)<=0) or np.any(areas<=0):
        raise ValueError('Invalid stationarity history or areas')
    rows=[]
    for k in range(windows):
        end=times[-1]-(windows-1-k)*3600;start=end-3600
        ids=np.flatnonzero((times>=start-1e-7)&(times<=end+1e-7))
        if len(ids)<2 or not np.isclose(times[ids[0]],start,atol=1e-7,rtol=0) or not np.isclose(times[ids[-1]],end,atol=1e-7,rtol=0):
            return dict(passed=False,reason='Missing complete hourly windows',windows=rows)
        bend_rms=np.sqrt(np.average((bed[ids]-bed[ids[0]])**2,weights=areas,axis=1))
        section_max=[float(np.sqrt(np.mean((p[ids]-p[ids[0]])**2,axis=1)).max()) for p in profiles]
        if not section_max:raise ValueError('No measured sections supplied')
        maximum=max(float(bend_rms.max()),*section_max)
        rows.append(dict(start_morphological_s=float(start),end_morphological_s=float(end),
            bend_max_rms_change_m=float(bend_rms.max()),section_max_rms_changes_m=section_max,
            passed=maximum<=tolerance))
    return dict(passed=all(row['passed'] for row in rows),windows=rows,tolerance_m=tolerance,
        criterion='Max sampled RMS change from hourly start <=1 mm in bend and every measured section, for three successive final hours',
        limitation='Stationarity of this accelerated discretized model, not experimental accuracy or MF independence')


def equilibrium_review(d):
    bend=(d['s']>=e.APPROACH)&(d['s']<=e.BEND_END)
    sections=[]
    for p in e.observations()['profiles']:
        coords=np.array([x['relative_from_outer'] for x in p['points']])
        use=np.array([x['primary_interior'] for x in p['points']])
        station=np.full(len(coords),e.APPROACH+e.RADIUS*math.radians(p['angle_degrees']))
        history=[]
        for z in d['z']:
            values,outside=a.sample(d,z,station,coords*e.WIDTH)
            if np.any(outside&use):raise ValueError('Primary profile leaves interpolation domain')
            history.append(values[use])
        sections.append(history)
    return stationarity(d['th'],d['z'][:,bend],d['area'][bend],sections)


def prefix_check(base,long):
    """Changing requested end time must not materially change the first hour."""
    from .finish_milestone import truncate
    prefix=truncate(long,base['th'][-1])
    for key in ('xy','area','th'):
        np.testing.assert_allclose(prefix[key],base[key],atol=1e-7,rtol=0)
    differences={key:float(np.max(abs(prefix[key]-base[key]))) for key in ('h','z','speed','win','wout','sin','sout')}
    # Stock engine floating-point end-time/table interpolation may be non-bitwise.
    for key,delta in differences.items():
        if delta>1e-6:raise ValueError(f'First-hour reproduction mismatch: {key}={delta}')
    return dict(max_absolute_differences=differences,tolerance=1e-6,passed=True)


def final_figures(data,gates,report):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharey=True)
    for j,p in enumerate(e.observations()['profiles']):
        coords=np.array([x['relative_from_outer'] for x in p['points']])
        obs=np.array([x['bed_change_m'] for x in p['points']])
        ss=np.full(len(coords),e.APPROACH+e.RADIUS*np.deg2rad(p['angle_degrees']))
        for i,d in enumerate(data):
            ax=axes[i,j];ax.plot(coords,obs,'ko',ms=3,label='Experimental equilibrium')
            for hour,color in [(1,'#bbbbbb'),(5,'#93bbd1'),(10,'#237ca5'),(20,'#b6412c')]:
                ids=np.flatnonzero(np.isclose(d['th'],hour*3600,atol=1e-7,rtol=0))
                if not len(ids):continue
                vals,_=a.sample(d,d['z'][ids[0]]-d['z'][0],ss,coords*e.WIDTH)
                ax.plot(coords,vals,color=color,label=f'{hour} morphological h')
            ax.axhline(0,color='.5',lw=.6);ax.grid(alpha=.2)
            ax.set(title=f"{d['p']['engine'].upper()}, {p['angle_degrees']} degrees",xlabel='Outer (0) to inner (1) wall',ylabel='Bed change (m)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('MF25 long-run bed profiles: stationarity is assessed separately, not assumed')
    fig.savefig(report/'long_run_profiles.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for d,color in zip(data,['#bf432f','#2469ad']):
        bend=(d['s']>=e.APPROACH)&(d['s']<=e.BEND_END)
        ids=np.flatnonzero(np.isclose(np.mod(d['th'],3600),0,atol=1e-7))
        hourly=[review.weighted_rms(d['z'][k,bend]-d['z'][j,bend],d['area'][bend])*1000 for j,k in zip(ids[:-1],ids[1:])]
        axes[0].plot(d['th'][ids[1:]]/3600,hourly,'o-',ms=3,color=color,label=d['p']['engine'].upper())
        mass=(d['z']-d['z'][0])@d['area']*r.BULK
        axes[1].plot(d['th']/3600,mass,color=color,label=d['p']['engine'].upper())
    axes[0].axhline(1,color='.4',ls='--',label='1 mm hourly threshold')
    axes[0].set(ylabel='Hourly bend RMS bed change (mm)',title='Approach to a stationary bed',ylim=(0,None))
    axes[1].set(ylabel='Net bed mass change (kg)',title='Global sediment storage');axes[1].axhline(0,color='.5',lw=.7)
    for ax in axes:ax.set_xlabel('Morphological hours');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('MF25: BRAIDS stationary='+str(gates[0]['passed'])+', Delft3D stationary='+str(gates[1]['passed']))
    fig.savefig(report/'equilibrium_history.png',dpi=170);plt.close(fig)


def main():
    report=r.HERE/'reports'/TAG;report.mkdir(parents=True,exist_ok=False)
    status=lambda state,**extra:r.write_json(report/'status.json',dict(status=state,updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    status('preparing')
    original_build=r.shared.verified_braids_build;original_writer=r.write_cut_case
    try:
        runtime=r.HERE/'runs'/f'{control.TAG}_runtime'
        exe,identity=mobile.locked_candidate(runtime)
        writer=runtime_builder.validation_writer(report)  # isolated writer, no runtime/source edits
        r.shared.verified_braids_build=lambda:(exe,identity);r.write_cut_case=writer
        baselines={eng:a.load(r.HERE/'runs'/f'{control.TAG}_{eng}') for eng in ('braids','delft')}
        if baselines['braids']['execution']['executable_sha256']!=identity['executable_sha256']:
            raise ValueError('Candidate is not the completed MF25 executable')
        r.write_json(report/'plan.json',dict(tag=TAG,morfac=FACTOR,hydraulic_end_s=MORPH_END/FACTOR,
            morphological_end_s=MORPH_END,output_morphological_s=600,dx_m=.10625,dt_max_s=.0125,
            native_threads=8,delft_ranks=4,physical_coefficients_changed=False,delft_source_modified=False,
            candidate_executable_sha256=identity['executable_sha256'],
            user_decision='Use MF25 despite the recorded 0.594 mm / 3.24% bed difference and 4.96% export difference; proceed to equilibrium and method comparison, no MF10 prerequisite',
            previous_strict_acceleration_screen_passed=False,
            initialization='Continuous run from the same qualified hydraulic seeds as the one-hour control; no periodic restart or secondary-flow reset',
            equilibrium_criterion='Final three hourly windows: max sampled RMS change from each window start <=1 mm in area-weighted bend and every primary measured section',
            maximum_morphological_hours=20,auto_extension=False,
            caution='20 hours is a bounded equilibrium attempt, not a claimed experimental duration. If nonstationary, report that honestly and extend.',
            script_sha256=r.sha256(Path(__file__)),seeds=control.SEEDS,
            expected_native_wall_hours_from_short_run=baselines['braids']['execution']['wall_s']*20/3600))
        r.write_json(report/'braids_build.json',identity)
        folders={}
        for eng in ('braids','delft'):
            folder=r.HERE/'runs'/f'{TAG}_{eng}';folders[eng]=folder
            fn=r.prepare_native if eng=='braids' else r.prepare_delft
            fn(folder,dx=.10625,end=MORPH_END/FACTOR,output=600/FACTOR,dt=.0125,morfac=FACTOR,
               hydraulic=False,seed=r.HERE/'runs'/control.SEEDS[eng])
            p=json.loads((folder/'input_provenance.json').read_text())
            if eng=='braids':p.update(engine_source_modified=True,production_source_modified=False,
                candidate='Unchanged ell=2h + MF25 validation input guard candidate',curvature_regularization_depth_factor=2.)
            p.update(equilibrium_attempt=True,acceleration_error_accepted_by_user=True)
            r.write_json(folder/'input_provenance.json',p)
        mobile.verify_common_configuration(*[json.loads((folders[eng]/'input_provenance.json').read_text()) for eng in ('braids','delft')])
        sys.path.insert(0,str(r.ROOT/'braids_support/sediment/synthetic_expansion'))
        import screen_delft_parallel as mpi
        def one(eng):
            r.write_json(report/f'{eng}_status.json',dict(status='running',folder=str(folders[eng])))
            try:
                execution=r.execute_native(folders[eng],8,timeout_s=172800) if eng=='braids' else mpi.execute(folders[eng],4,timeout=172800)
                d=a.load(folders[eng]);gate=equilibrium_review(d);prefix=prefix_check(baselines[eng],d)
                r.write_json(report/f'{eng}_status.json',dict(status='complete',execution=execution,stationarity=gate,first_hour_reproduction=prefix))
                return d,gate,prefix
            except Exception as exc:
                r.write_json(report/f'{eng}_status.json',dict(status='failed',error=str(exc)));raise
        status('running',morfac=FACTOR,morphological_hours=20)
        with ThreadPoolExecutor(max_workers=2) as pool:finished=list(pool.map(one,('braids','delft')))
        data=[item[0] for item in finished];gates=[item[1] for item in finished]
        a.analyze(TAG,list(folders.values()))
        mobile.artifact_figure(data,report);final_figures(data,gates,report)
        profiles=review.paired_profile_review(data)
        metrics=[a.audit(d) for d in data]
        r.write_json(report/'equilibrium.json',dict(stationarity=gates,metrics=metrics,experimental_profiles=profiles,
            both_stationary=all(g['passed'] for g in gates),experimental_accuracy_validated=False,
            acceleration_independence_proved=False,production_promoted=False,delft_modified=False))
        rows=[]
        for d,g,p,metric in zip(data,gates,profiles['trajectories'],metrics):
            rows.append(f"| {d['p']['engine'].upper()} | {g['passed']} | {1000*p['pooled_interior_rmse_m'][-1]:.2f} | {1000*metric['max_erosion_m']:.2f} | {1000*metric['max_deposition_m']:.2f} | {metric['sediment_budget_residual_kg']:.3g} |")
        r.write(report/'RESULTS.md',f'''# Continuous MF25, 20-morphological-hour equilibrium attempt

Both engines completed 2880 hydraulic seconds = 20 morphological hours, with
unchanged physics, seeds, nominal 0.10625 m spacing, BRAIDS 8 threads and stock
Delft3D 4 MPI ranks. The first morphological hour was checked against the prior
MF25 control. The user explicitly accepted its recorded acceleration differences;
the earlier strict acceleration-screen result remains unchanged.

| Model | Final three hourly stationarity checks pass | Interior experiment RMSE (mm) | Maximum scour (mm) | Maximum deposition (mm) | Sediment residual (kg) |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Stationarity requires <=1 mm maximum sampled RMS change from each hourly start
in both the area-weighted bend and every measured section, for three final hours.
It does not prove experimental agreement or acceleration independence. If a model
fails, its result is still transient and needs extension, not an equilibrium label.
Experimental scores use digitized equilibrium markers, not raw measurements.

[Long-run profiles](long_run_profiles.png), [equilibrium history](equilibrium_history.png),
[raw fields](fields.png), [outlet/cell-scale checks](bed_scale_outlet.png),
[full metrics and gates](equilibrium.json).

Next: inspect equilibrium shape, sediment balance and cell-scale/outlet behaviour;
compare published bedload-bend methods using experiments, then implement and test
the strongest supported alternative in BRAIDS. No Delft3D changes or plugin
promotion occurred. No automatic extension beyond 20 hours was started.
''')
        status('complete',both_stationary=all(g['passed'] for g in gates),equilibrium_report=str(report/'RESULTS.md'))
    except Exception as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:r.shared.verified_braids_build=original_build;r.write_cut_case=original_writer


if __name__=='__main__':main()
