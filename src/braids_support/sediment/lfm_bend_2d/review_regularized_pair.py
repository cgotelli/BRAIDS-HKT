"""Read-only solver-output review of the completed fine-grid candidate pair.

Writes analysis artifacts only; never launches, changes or promotes an engine.
"""
import json
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from . import run_case as r
from . import experiment as e
from . import analyze_runs as a

TAG='regularized_mobile3600_v1'


def weighted_rms(values,weights):
    return float(np.sqrt(np.average(np.asarray(values)**2,weights=weights)))


def temporal_bed_review(d):
    bend=(d['s']>=e.APPROACH)&(d['s']<=e.BEND_END)
    dz=d['z']-d['z'][0]
    age=np.asarray(d['th']);weights=d['area'][bend]
    start=np.flatnonzero(np.isclose(age,age[-1]-3600,atol=1e-8,rtol=0))
    last_hour=None if not len(start) else weighted_rms(d['z'][-1,bend]-d['z'][start[0],bend],weights)
    return dict(times_s=age.tolist(),
        bend_rms_bed_change_m=[weighted_rms(row[bend],weights) for row in dz],
        interval_s=np.diff(age).tolist(),
        interval_bend_rms_change_m=[weighted_rms(row[bend],weights) for row in np.diff(d['z'],axis=0)],
        last_hour_bend_rms_change_m=last_hour,
        last_hour_bend_gate_passed=bool(last_hour is not None and last_hour<=.001),
        note='Last-hour test uses saved endpoints, not an extrapolation of the last 600 seconds')


def paired_profile_review(data):
    observations=e.observations();rows=[];pooled=[[] for _ in data]
    for profile in observations['profiles']:
        coords=np.array([p['relative_from_outer'] for p in profile['points']])
        observed=np.array([p['bed_change_m'] for p in profile['points']])
        use=np.array([p['primary_interior'] for p in profile['points']])
        station=np.full(len(coords),e.APPROACH+e.RADIUS*math.radians(profile['angle_degrees']))
        series=[]
        for d in data:
            samples=[]
            for dz in d['z']-d['z'][0]:
                values,outside=a.sample(d,dz,station,coords*e.WIDTH)
                samples.append(values);use &= ~outside
            series.append(np.array(samples))
        model_rows=[]
        for i,(d,values) in enumerate(zip(data,series)):
            errors=values[:,use]-observed[use]
            pooled[i].append(errors)
            model_rows.append(dict(engine=d['p']['engine'],
                common_interior_rmse_m=float(np.sqrt(np.mean(errors[-1]**2))),
                primary_relief_m=float(np.ptp(values[-1,use])),
                final_primary_bed_change_m=values[-1,use].tolist(),
                last_600_primary_rms_change_m=float(np.sqrt(np.mean((values[-1,use]-values[-2,use])**2))),
                all_sampled_profile_metrics=e.profile_metrics(coords,values[-1])))
        rows.append(dict(angle=profile['angle_degrees'],common_primary_points=int(use.sum()),
            observed_primary_relief_m=float(np.ptp(observed[use])),
            observed_all_marker_metrics=e.profile_metrics(coords,observed),models=model_rows))
    trajectories=[]
    for d,errors in zip(data,pooled):
        errors=np.concatenate(errors,axis=1)
        trajectories.append(dict(engine=d['p']['engine'],times_s=d['th'].tolist(),
            common_primary_points=errors.shape[1],
            pooled_interior_rmse_m=np.sqrt(np.mean(errors**2,axis=1)).tolist()))
    return dict(sections=rows,trajectories=trajectories,
        comparison='Both models scored on the same primary markers with no nearest-neighbour extrapolation',
        warning='Transient model versus digitized equilibrium; not an accuracy ranking or raw measurement uncertainty')


def spatial_residual_review(d):
    dz=d['z'][-1]-d['z'][0]
    _,neighbors=cKDTree(d['xy']).query(d['xy'],k=9)
    residual=np.empty(len(dz))
    for c,ids in enumerate(neighbors):
        matrix=np.column_stack([np.ones(len(ids)),d['xy'][ids]-d['xy'][c]])
        residual[c]=dz[c]-np.linalg.lstsq(matrix,dz[ids],rcond=None)[0][0]
    bend=(d['s']>=e.APPROACH)&(d['s']<=e.BEND_END)
    bank=(d['n']<2*d['p']['dx'])|(d['n']>e.WIDTH-2*d['p']['dx'])
    result={}
    for name,mask in [('bend',bend),('bank_band',bend&bank),('interior',bend&~bank)]:
        result[name]=dict(cells=int(mask.sum()),rms_m=weighted_rms(residual[mask],d['area'][mask]),
            max_abs_m=float(np.max(abs(residual[mask]))))
    return dict(regions=result,note='Residual includes physical transverse bed curvature and cut-cell geometry; not a pure checkerboard amplitude')


def main():
    report=r.HERE/'reports'/TAG;review=report/'review';review.mkdir(exist_ok=True)
    native=r.HERE/'runs'/f'{TAG}_mobile_braids'
    delft=r.HERE/'runs/fine3600_m1_v1_mobile_delft'
    saved=json.loads((report/'mobile_qualification.json').read_text())
    assert r.sha256(r.shared.native_frames_path(native))==saved['frames_sha256']
    assert r.sha256(native/'input_provenance.json')==saved['input_sha256']
    assert r.sha256(native/'execution.json')==saved['execution_sha256']
    reference=json.loads((report/'delft_reference.json').read_text())
    for name,key in [('input_provenance.json','input_sha256'),('execution.json','execution_sha256'),
                     ('trim-basin.nc','trim_sha256'),('trih-basin.nc','trih_sha256')]:
        assert r.sha256(delft/name)==reference[key],f'Delft3D reference changed: {name}'
    data=[a.load(native),a.load(delft)]
    np.testing.assert_allclose(data[0]['th'],data[1]['th'],atol=1e-8,rtol=0)
    metrics=[a.audit(d) for d in data]
    histories=[dict(engine=d['p']['engine'],**temporal_bed_review(d)) for d in data]
    profiles=paired_profile_review(data)
    residuals=[dict(engine=d['p']['engine'],**spatial_residual_review(d)) for d in data]
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for h,p,color in zip(histories,profiles['trajectories'],['#bf432f','#2469ad']):
        minutes=np.array(h['times_s'])/60
        axes[0].plot(minutes,np.array(h['bend_rms_bed_change_m'])*1000,'o-',color=color,label=h['engine'].upper())
        axes[1].plot(minutes[1:],np.array(h['interval_bend_rms_change_m'])*1000,'o-',color=color)
        axes[2].plot(minutes,np.array(p['pooled_interior_rmse_m'])*1000,'o-',color=color)
    axes[0].set(title='Morphology is developing',ylabel='Bend RMS bed change from initial (mm)')
    axes[1].set(title='Still changing at the end',ylabel='Bend RMS change per 10 minutes (mm)')
    axes[2].set(title='Distance from measured equilibrium',ylabel='Common interior profile RMSE (mm)')
    for ax in axes:ax.set_xlabel('Morphological time (minutes)');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Completed matched-resolution, MF=1 pair: transient evolution, not equilibrium validation',fontsize=12)
    fig.savefig(review/'time_history.png',dpi=160);plt.close(fig)
    result=dict(complete=True,review_date='2026-09-30',input_and_output_hashes_verified=True,
        end_morphological_s=3600,dx_m=.10625,morfac=1,
        stability_and_conservation_milestone_passed=True,experimental_equilibrium_reproduced=False,
        production_promoted=False,new_simulations_started=False,delft_source_modified=False,
        final_metrics=metrics,temporal_bed_review=histories,experimental_profile_review=profiles,
        local_plane_diagnostic=residuals,
        native_wall_hours=data[0]['execution']['wall_s']/3600,
        observations_sha256=r.sha256(e.HERE/'data/figure17_observations.json'),
        review_script_sha256=r.sha256(Path(__file__)))
    r.write_json(review/'review.json',result)
    table=[]
    for m in metrics:
        table.append(f"| {m['engine'].upper()} | {m['max_erosion_m']*1000:.1f} | {m['max_deposition_m']*1000:.1f} | {m['outlet_max_deposition_m']*1000:.2f} | {m['sediment_budget_residual_kg']:.3g} |")
    trend='; '.join(f"{h['engine'].upper()}: {h['interval_bend_rms_change_m'][-1]*1000:.2f} mm" for h in histories)
    gaps='; '.join(f"{p['engine'].upper()}: {p['pooled_interior_rmse_m'][0]*1000:.1f} to {p['pooled_interior_rmse_m'][-1]*1000:.1f} mm" for p in profiles['trajectories'])
    r.write(review/'REVIEW.md',f'''# Reviewed fine-grid one-hour pair — 2026-09-30

**Completed stability/conservation milestone; experimental validation is not complete.**
Both models reached 3600 seconds at MF=1 on matched nominal 0.10625 m spacing.
Output and input hashes match the recorded candidate and preserved stock reference.
BRAIDS took {result['native_wall_hours']:.2f} wall hours using eight OpenMP threads.
No new simulation, production change, plugin installation or Delft3D modification was made during review.

## Global result

Both models erode the outer side and deposit on the inner side of the bend.
BRAIDS has stronger outer scour and less inner accumulation than Delft3D at this time.
Both pass the broad erosion/deposition-location test, but neither reproduces the
much larger experimental equilibrium relief. The final section shapes also differ
from the experiment, not only their amplitudes. More elapsed simulation time is
needed to assess equilibrium, but is not guaranteed to remove those differences.

| Model | Global maximum scour (mm) | Global maximum deposition (mm) | Outlet maximum deposition (mm) | Sediment budget residual (kg) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

BRAIDS accepted {metrics[0]['accepted_feed_kg']:.6f} kg and exported
{metrics[0]['exported_sediment_kg']:.6f} kg; its bed lost {-metrics[0]['bed_inventory_change_kg']:.6f} kg.
Thus exported sediment exceeds feed consistently with net bed erosion, not a hidden outlet trap.
Outlet deposition remains millimetric and the approach profiles are smooth, apart
from a small end-cell upturn also visible in the reference.

The previous large, domain-wide ripple pattern is absent in the reviewed raw maps.
Small bank-adjacent cut-cell roughness remains. Whole-bend local-plane residual RMS
is about 1 mm in both models; that diagnostic also includes physical bed curvature
and cannot certify the absence of every grid artefact. No result smoothing was applied.

## Still transient, and still far from the experiment

Area-weighted RMS bed change in the last 600 seconds: {trend}.
Both fail the predeclared last-hour equilibrium test (<=1 mm), using saved endpoints.
Pooled RMSE against common interior experimental markers decreases over the hour:
{gaps}. Those numbers describe approach to an equilibrium target, not an accuracy
ranking between transient models. The markers are digitized figure data, not raw
laboratory records, and graphical tolerance is not measurement uncertainty.

## Figures

- [Final raw-cell depth and bed change](../fields.png)
- [Final profiles versus experimental equilibrium](../profiles.png)
- [Time histories and distance from equilibrium](time_history.png)
- [Cell-scale and outlet diagnostics](../bed_scale_outlet.png)
- [Full numerical review](review.json)

## Next milestone and immediate step

Next milestone: a numerically qualified equilibrium comparison with the experiment.
Immediate step: use this completed fine-grid MF=1 pair as the baseline to qualify
a faster morphological factor at the same elapsed morphological time, then extend
both models toward the declared equilibrium criterion if that acceleration passes.
Keep the same physical coefficients; do not tune them to the experimental profile.
If a steady bed remains discrepant, audit the documented transport, slope,
wall/secondary-flow and reconstructed-feed assumptions. Stability alone does not
justify releasing this experimental closure in the plugin.
''')
    print(json.dumps(dict(histories=histories,profile_trend=profiles['trajectories'],residuals=residuals),indent=2))


if __name__=='__main__':main()
