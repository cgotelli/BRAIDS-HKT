"""MF=25 common-time control against the completed fine-grid MF=1 pair.

Run the isolated BRAIDS candidate and locked stock one-layer Delft3D only.
No production or Delft3D source changes, automatic equilibrium extension,
or plugin promotion. A validation-only BRAIDS input guard permits MF=25.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import run_case as r
from . import analyze_runs as a
from . import qualify as q
from . import regularized_mobile as mobile
from . import background_run as background
from .finish_milestone import truncate
from . import mf25_validation_runtime as runtime_builder

TAG='fine_mf25_3600_v2'
FACTOR=25.
MORPH_END=3600.
BASELINES=dict(braids='regularized_mobile3600_v1_mobile_braids',delft='fine3600_m1_v1_mobile_delft')
SEEDS=dict(braids='regularized_cold300_v1_fine_t8',delft='fine3600_m1_v1_hydraulic_delft')


def same_initial_state_and_settings(base,accelerated):
    # Preserve the MF=1 protocol (including secondary reset). No changing
    # physics while testing the consequences of the faster morphological clock.
    for key in ('dx','dt','Q','manning','feed_kg_s','d50_m','d90_m','porosity',
                'viscosity_m2_s','secondary_momentum_beta','longitudinal_slope_factor',
                'transverse_slope_factor','seed'):
        if base['p'][key]!=accelerated['p'][key]:raise ValueError(f'Changed control setting: {key}')
    for key in ('xy','area'):
        np.testing.assert_allclose(base[key],accelerated[key],atol=1e-12,rtol=0)
    for key in ('h','z','speed'):
        np.testing.assert_allclose(base[key][0],accelerated[key][0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(base['th'],accelerated['th'],atol=1e-7,rtol=0)
    if base['p']['morfac']!=1 or accelerated['p']['morfac']!=FACTOR:
        raise ValueError('Wrong acceleration factors')


def compare(base,accelerated):
    same_initial_state_and_settings(base,accelerated)
    rows=[]
    for time in base['th'][1:]:
        row=q.acceleration(truncate(base,time),truncate(accelerated,time))
        row.update(time_morphological_s=float(time),criterion_scope='Same conservative 0.5 mm / 10% bed-field screen used previously; not experimental validation')
        rows.append(row)
    bm,am=a.audit(base),a.audit(accelerated)
    return dict(engine=base['p']['engine'],by_time=rows,baseline_metrics=bm,mf25_metrics=am,
        final_acceleration_screen_passed=rows[-1]['passed'],
        all_saved_times_passed=all(row['passed'] for row in rows),
        sediment_export_relative_difference=abs(am['exported_sediment_kg']-bm['exported_sediment_kg'])/max(abs(bm['exported_sediment_kg']),1e-12),
        baseline_executable=base['execution'].get('executable_sha256',base['execution'].get('runtime')),
        accelerated_executable=accelerated['execution'].get('executable_sha256',accelerated['execution'].get('runtime')))


def figures(base,data,report):
    from . import experiment as e
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained',sharex=True,sharey=True)
    for j,profile in enumerate(e.observations()['profiles']):
        coord=np.array([p['relative_from_outer'] for p in profile['points']])
        obs=np.array([p['bed_change_m'] for p in profile['points']])
        s=np.full(len(coord),e.APPROACH+e.RADIUS*np.deg2rad(profile['angle_degrees']))
        for i,engine in enumerate(('braids','delft')):
            ax=axes[i,j];ax.plot(coord,obs,'ko',ms=3,label='Experimental equilibrium')
            for d,label,color in [(base[engine],'MF=1','#333333'),(data[engine],'MF=25','#cf6125')]:
                values,_=a.sample(d,d['z'][-1]-d['z'][0],s,coord*e.WIDTH)
                ax.plot(coord,values,'o-',ms=2,label=label,color=color)
            ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
            ax.set(title=f'{engine.upper()}, {profile["angle_degrees"]} degree section',
                   xlabel='Outer (0) to inner (1) wall',ylabel='Bed change (m)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Morphological acceleration check at 3600 morphological seconds\nMF=25 uses 144 hydraulic seconds; both trajectories are transient',fontsize=12)
    fig.savefig(report/'mf25_vs_mf1_profiles.png',dpi=160);plt.close(fig)


def main():
    report=r.HERE/'reports'/TAG;report.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        r.write_json(report/'status.json',dict(status=state,updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    status('preparing')
    old=r.shared.verified_braids_build
    old_writer=r.write_cut_case
    try:
        exe,identity,writer=runtime_builder.build(TAG)
        mobile.completed_hydraulics(r.HERE/'reports/regularized_cold300_v1')
        delft_hydro=a.load(r.HERE/'runs'/SEEDS['delft'])
        if not all(background.hydraulic_gates(a.audit(delft_hydro)).values()):
            raise RuntimeError('Delft3D hydraulic seed is unqualified')
        base={engine:a.load(r.HERE/'runs'/name) for engine,name in BASELINES.items()}
        if base['braids']['execution']['executable_sha256']!=identity['original_executable_sha256']:
            raise ValueError('BRAIDS executable differs from MF1')
        r.write_json(report/'plan.json',dict(morfac=FACTOR,hydraulic_seconds=MORPH_END/FACTOR,
            morphological_seconds=MORPH_END,dx_m=.10625,dt_max_s=.0125,
            output_hydraulic_s=600/FACTOR,output_morphological_s=600,
            native_threads=8,delft_ranks=4,baselines=BASELINES,seeds=SEEDS,
            secondary_seed='Zero in both engines, identical to MF1 initialization protocol',
            source_change='BRAIDS constant-MF input guard only: <=20 to <=25 in isolated build/writer; numerical arithmetic unchanged. Stock Delft3D unchanged.',
            physical_parameters_changed=False,delft_source_modified=False,production_promoted=False,
            automatic_equilibrium_extension=False,
            acceleration_criterion='Bed RMS <=0.5 mm AND <=10% relative, separately against each engine MF1',
            caution='MF25 requested by user, not assumed accurate or 25x faster. Fixed timestep retained.',
            runner_sha256=r.sha256(Path(__file__)),
            baseline_receipts={engine:dict(input=r.sha256(d['folder']/'input_provenance.json'),
                execution=r.sha256(d['folder']/'execution.json')) for engine,d in base.items()}))
        r.write_json(report/'braids_build.json',identity)
        # An identical-input MF1 execution must remain identical under the new
        # parser guard; do not assume that source inspection is sufficient.
        check_rows=[]
        for label,(check_exe,check_identity) in [('original',mobile.locked_candidate(r.HERE/'runs/fine_regularized_v1_runtime')),
                                                ('mf25_guard',(exe,identity))]:
            r.shared.verified_braids_build=lambda:(check_exe,check_identity)
            folder=r.HERE/'runs'/f'{TAG}_mf1_guard_{label}'
            r.prepare_native(folder,dx=.10625,end=1.,output=1.,dt=.0125,morfac=1.,
                             hydraulic=False,seed=r.HERE/'runs'/SEEDS['braids'])
            execution=r.execute_native(folder,8)
            frames=r.shared.read_frames(r.shared.native_frames_path(folder))
            check_rows.append((frames[-1],execution))
        state_difference=float(np.max(abs(check_rows[0][0]['q']-check_rows[1][0]['q'])))
        ledger_difference=float(np.max(abs(check_rows[0][0]['ledger']-check_rows[1][0]['ledger'])))
        r.write_json(report/'mf1_guard_compatibility.json',dict(max_state_difference=state_difference,
            max_ledger_difference=ledger_difference,end_s=1.,threads=8,
            executions=[row[1] for row in check_rows]))
        if state_difference!=0 or ledger_difference!=0:raise RuntimeError('Input-limit-only build changed MF1 control')
        r.shared.verified_braids_build=lambda:(exe,identity)
        r.write_cut_case=writer
        folders={}
        for engine in ('braids','delft'):
            folder=r.HERE/'runs'/f'{TAG}_{engine}';folders[engine]=folder
            prepare=r.prepare_native if engine=='braids' else r.prepare_delft
            prepare(folder,dx=.10625,end=MORPH_END/FACTOR,output=600/FACTOR,dt=.0125,
                    morfac=FACTOR,hydraulic=False,seed=r.HERE/'runs'/SEEDS[engine])
            p=json.loads((folder/'input_provenance.json').read_text())
            p.update(acceleration_control=True,mf1_baseline=str(base[engine]['folder']))
            if engine=='braids':p.update(engine_source_modified=True,production_source_modified=False,
                candidate='ell=2h candidate plus isolated constant MF25 input allowance; MF1 control identical',
                curvature_regularization_depth_factor=2.,constant_mf_validation_limit=25)
            r.write_json(folder/'input_provenance.json',p)
        native_p=json.loads((folders['braids']/'input_provenance.json').read_text())
        delft_p=json.loads((folders['delft']/'input_provenance.json').read_text())
        mobile.verify_common_configuration(native_p,delft_p)
        sys.path.insert(0,str(r.ROOT/'braids_support/sediment/synthetic_expansion'))
        import screen_delft_parallel as mpi
        def one(engine):
            folder=folders[engine]
            r.write_json(report/f'{engine}_status.json',dict(status='running',folder=str(folder)))
            try:
                execution=r.execute_native(folder,8,timeout_s=86400) if engine=='braids' else mpi.execute(folder,4,timeout=86400)
                data=a.load(folder);result=compare(base[engine],data)
                r.write_json(report/f'{engine}_comparison.json',result)
                r.write_json(report/f'{engine}_status.json',dict(status='complete',
                    final_acceleration_screen_passed=result['final_acceleration_screen_passed'],execution=execution))
                return engine,data,result
            except Exception as exc:
                r.write_json(report/f'{engine}_status.json',dict(status='failed',error=str(exc)));raise
        status('running',morfac=FACTOR)
        with ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(one,('braids','delft')))
        data={engine:d for engine,d,result in completed};results=[result for engine,d,result in completed]
        a.analyze(TAG,[folders['braids'],folders['delft']])
        mobile.artifact_figure([data['braids'],data['delft']],report)
        figures(base,data,report)
        accepted=all(row['all_saved_times_passed'] for row in results)
        r.write_json(report/'acceleration.json',dict(cases=results,mf25_common_time_screen_passed=accepted,
            equilibrium_extension_started=False,experimental_validation_passed=False))
        rows='\n'.join(f"| {row['engine']} | {1000*row['by_time'][-1]['rms_bed_difference_m']:.3f} | {100*row['by_time'][-1]['relative_rms_difference']:.2f}% | {row['final_acceleration_screen_passed']} |" for row in results)
        r.write(report/'RESULTS.md',f'''# MF25 versus MF1 at one morphological hour

Both runs completed 144 hydraulic seconds = 3600 morphological seconds, nominal
spacing 0.10625 m. Same physical settings and initial-state protocol as MF1.
BRAIDS used 8 OpenMP threads; stock one-layer Delft3D used 4 MPI ranks.

| Model | Final bed-field RMS difference (mm) | Relative RMS difference | Final screen passed |
| --- | ---: | ---: | --- |
{rows}

Both engines pass at every saved common time: **{accepted}**.
The criterion remains <=0.5 mm AND <=10% relative RMS. This numerical-acceleration
test is not an experiment match, equilibrium proof, or universal safe MF limit.
Startup includes secondary-flow reinitialization, exactly as in the MF1 baseline.

[MF25/MF1 profiles](mf25_vs_mf1_profiles.png), [fields](fields.png),
[outlet and cell-scale diagnostics](bed_scale_outlet.png), [full receipts](acceleration.json).

Review before any equilibrium extension. If MF25 fails, separate startup-memory
effects from ongoing flow/bed lag, then lower the factor or qualify a consistently
initialized rerun in both engines. Do not tune physical coefficients to compensate
for acceleration error. Delft3D source/runtime and the installed plugin are unchanged.
''')
        status('complete',mf25_common_time_screen_passed=accepted,equilibrium_extension_started=False)
    except Exception as exc:status('failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r.shared.verified_braids_build=old
        r.write_cut_case=old_writer


if __name__=='__main__':main()
