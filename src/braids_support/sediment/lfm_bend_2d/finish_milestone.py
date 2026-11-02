"""Create the final milestone report only from complete, audited trajectories."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from . import run_case as r
from . import analyze_runs as a
from . import qualify as q
from . import experiment as e


def truncate(d,time):
    selected=np.flatnonzero(d['th']<=time+1e-8)
    if not len(selected) or not np.isclose(d['th'][selected[-1]],time):
        raise ValueError('No saved frame at the requested morphological time')
    fields={'t','th','h','z','speed','win','wout','sin','sout'}
    return {k:(v[selected] if k in fields else v) for k,v in d.items()}


def evolution(data,report):
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,profile in zip(axes,e.observations()['profiles']):
        coords=np.array([p['relative_from_outer'] for p in profile['points']])
        primary=np.array([p['primary_interior'] for p in profile['points']])
        station=e.APPROACH+e.RADIUS*math.radians(profile['angle_degrees'])
        for d,color in zip(data,['#bf432f','#2469ad']):
            out=[];inside=[]
            for z in d['z']-d['z'][0]:
                pred,_=a.sample(d,z,np.full(len(coords),station),coords*e.WIDTH)
                out.append(np.mean(pred[primary&(coords<=.25)]))
                inside.append(np.mean(pred[primary&(coords>=.75)]))
            ax.plot(d['th']/60,out,color=color,label=d['p']['engine']+' outer')
            ax.plot(d['th']/60,inside,'--',color=color,label=d['p']['engine']+' inner')
        ax.axhline(0,color='.5',lw=.8)
        ax.set(title=f"{profile['angle_degrees']} degree section",xlabel='Morphological minutes')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean bed change in bank-side quarter (m)')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Growth of outer scour and inner deposition — transient, not equilibrium')
    fig.savefig(report/'evolution.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,d in zip(axes,data):
        mask=d['s']>e.LENGTH-3
        for i,color in [(1,'#a1b8c9'),(-1,'#252525')]:
            ax.scatter(e.LENGTH-d['s'][mask],(d['z'][i]-d['z'][0])[mask]*1000,
                s=9,color=color,label=f"{d['th'][i]/60:g} min")
        ax.axhline(0,color='.4',lw=.8);ax.grid(alpha=.2);ax.legend()
        ax.set(title=d['p']['engine']+' — actual outlet cells',xlabel='Distance upstream from outlet (m)',
               ylabel='Bed change (mm)');ax.invert_xaxis()
    fig.suptitle('Outlet audit: all cell values shown, not an averaged or smoothed bed')
    fig.savefig(report/'outlet.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    limit=max(float(np.max(abs(d['z']-d['z'][0]))) for d in data)
    collections=[]
    for ax,d in zip(axes,data):
        im=PolyCollection(d['polygons'],array=d['z'][0]-d['z'][0],cmap='RdBu_r',
                          clim=(-limit,limit),edgecolors='none',antialiased=False)
        collections.append(im);ax.add_collection(im);ax.autoscale();ax.set_aspect('equal')
        ax.set(title=d['p']['engine']+' 2D',xlabel='x (m)',ylabel='y (m)')
    fig.colorbar(collections[0],ax=axes,label='Bed change (m): blue erosion; red deposition',shrink=.8)
    def update(i):
        for im,d in zip(collections,data):im.set_array(d['z'][i]-d['z'][0])
        fig.suptitle(f"{data[0]['th'][i]/60:g} morphological min; factor 5; experimental equilibrium not yet reached")
        return collections
    FuncAnimation(fig,update,frames=len(data[0]['t']),interval=700).save(
        report/'evolution.gif',writer=PillowWriter(fps=2),dpi=105)
    plt.close(fig)


def main():
    report=r.HERE/'reports/first_2d_milestone'
    data,metrics=a.analyze('first_2d_milestone',[r.HERE/'runs'/f'mobile3600_m5_{k}' for k in ('braids','delft')])
    controls=[]
    for d in data:
        base=a.load(r.HERE/'runs'/f"mobile600_m1_{d['p']['engine']}")
        controls.append(dict(engine=d['p']['engine'],acceleration=q.acceleration(base,truncate(d,600)),
                             unaccelerated_audit=a.audit(base)))
    short=json.loads((r.HERE/'reports/qualification_v1/metrics.json').read_text())
    gates=dict(short_operational_pair=short['operational_pair_passed'],
        extended_acceleration_screen=all(x['acceleration']['passed'] for x in controls),
        full_feed=all(abs(x['prescribed_feed_fraction']-1)<.005 for x in metrics),
        bed_mass_balance=all(x['sediment_budget_residual_over_throughput']<.01 for x in metrics),
        positive_depth=all(x['min_depth_m']>0 for x in metrics),
        sediment_export=all(x['exported_sediment_kg']>0 for x in metrics),
        two_dimensional=data[1]['nk']==1,
        same_nominal_spacing=data[0]['p']['dx']==data[1]['p']['dx'])
    evolution(data,report)
    scripts=['run_case.py','analyze_runs.py','qualify.py','finish_milestone.py','experiment.py','CLOSURE_MAP.md']
    r.write_json(report/'completed.json',dict(milestone='First operationally qualified 2D bedload comparison',
        completed=all(gates.values()),gates=gates,extended_controls=controls,
        final_metrics=metrics,engine_sources_modified=False,
        experiments_reproduced=False,equilibrium_established=False,grid_convergence_established=False,
        full_duration_acceleration_equivalence_established=False,
        runtime_identities=[d['execution'] for d in data],
        script_hashes={name:r.sha256(r.HERE/name) for name in scripts},
        run_inputs={d['p']['engine']:r.sha256(d['folder']/'input_provenance.json') for d in data},
        source_observations_sha256=r.sha256(r.HERE/'data/figure17_observations.json')))
    rows=[]
    for m in metrics:
        rows.append(f"| {m['engine']} | {1000*m['bend_outer_mean_change_m']:.2f} | {1000*m['bend_inner_mean_change_m']:.2f} | "
                    f"{1000*m['max_erosion_m']:.2f} | {1000*m['max_deposition_m']:.2f} | {1000*m['outlet_max_deposition_m']:.3f} |")
    budgets=[]
    for m in metrics:
        budgets.append(f"| {m['engine']} | {m['accepted_feed_kg']:.6f} | {m['exported_sediment_kg']:.6f} | "
                       f"{m['bed_inventory_change_kg']:.6f} | {m['sediment_budget_residual_kg']:.3g} |")
    controls_text='; '.join(f"{x['engine']}: {100*x['acceleration']['relative_rms_difference']:.2f}% relative RMS "
                           f"({1000*x['acceleration']['rms_bed_difference_m']:.3f} mm)" for x in controls)
    text=f'''# First 2D bedload comparison — 2026-09-29

**Operational milestone {'passed' if all(gates.values()) else 'NOT passed'}:** both models completed 3,600 morphological seconds on matched 0.2125 m nominal spacing after a 300 s fixed-bed hydraulic warm-up. Stock Delft3D used one vertical layer. BRAIDS used eight OpenMP threads. The primary one-hour pair used morphological factor 5 (720 hydraulic seconds), not factor 10.

This is a **transient comparison**, not a reproduction of experimental equilibrium or an accuracy ranking. The experimental measurements remain the reference; neither model is ground truth. No production solver/plugin or Delft3D source was modified.

## Global pattern at one morphological hour

| Model | Outer-quarter mean bed change (mm) | Inner-quarter mean (mm) | Maximum scour (mm) | Maximum deposition (mm) | Maximum deposition in last two outlet cell lengths (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Bank-side means use cell centres in the corresponding quarter of the bend width and the whole bend; extrema use all cells. These summaries do not conceal the actual-cell maps or replace the three experimental cross-sections. The experiment's equilibrium scour is much deeper (approximately 142–202 mm at the three sampled sections); comparing that equilibrium with an evolving one-hour bed cannot establish which model is more accurate.

Figures: [actual-cell water depth and bed change](fields.png), [experimental cross-section comparison](profiles.png), [time evolution](evolution.png), [all outlet cell changes](outlet.png), [animation](evolution.gif). Blue is erosion and red is deposition in the maps. Profile error bars are graphical reading tolerance, not laboratory measurement uncertainty. Cell-scale local-plane residuals are retained in [metrics.json](metrics.json); these are diagnostics, not a declaration that grid oscillations are absent.

## Flux and inventory audit

| Model | Accepted sediment input (kg) | Export (kg) | Bed inventory change (kg) | Budget residual (kg) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(budgets)}

Both fixed-bed warm-ups settle to the prescribed 0.17 m³/s inflow and outflow. The native and stock execution diagnostics independently reproduce their intended MPM capacities to numerical precision. Using collocated water depth instead of FLOW's face-interpolated roughness changes its capacity by at most 0.17% in the short check. Sediment feed was **reconstructed**, not measured: 0.0193051458 kg/s in both engines. See [closure and unit mapping](../../CLOSURE_MAP.md).

## Acceleration and multicore checks

At 120 morphological seconds, factor 10 changes the bend's bed-change field by 9.96% in BRAIDS and 13.47% in Delft3D relative to factor 1. It therefore fails the chosen common 10% short-screen criterion. Factor 5 gives 5.03% and 6.18% and was retained. The extended unaccelerated controls at 600 morphological seconds give **{controls_text}**. This checks startup/early evolution, **not factor-5 equivalence over the entire hour or at equilibrium**.

The native identical-input 1/2/4/8-thread screen returned identical saved states; eight threads was fastest (4.27 s versus 6.33 s with one thread for the short probe). Stock FLOW's same-input 1/2/4/8-rank screen favoured four ranks (8.86 s versus 12.75 s serial). Two and four ranks stayed within the strict 1e-7 trajectory screen; eight ranks exceeded it slightly in velocity (1.81e-7 m/s), so it was not selected. These are local short-run timings with other independent runs active, not universal scaling claims. The completed primary FLOW pair remains the serial reference; no running input or engine was replaced after the timings.

## What remains / next milestone

The next milestone is **an equilibrium and resolution-qualified comparison with the experiment**, using the already prepared 0.10625 m geometry. Immediate next step: check the finer mesh and a smaller acceleration factor at a shared time before extending to equilibrium, while retaining raw bank-cell and outlet diagnostics. Longer-run acceleration, timestep and boundary-distance sensitivity remain open. Do not tune coefficients to the final experimental profiles or enable lateral erosion before this fixed-wall bedload assessment is complete.

The selected 2D paper uses 2DEF, not Delft3D. Our two solvers use thresholded MPM rather than its smoothed law; assumed grain roughness, reconstructed feed and simplified wall/mixing treatment remain documented limitations. Raw laboratory files and the original numerical deck have not been recovered.

Machine-readable gates, exact run identities and provenance hashes: [completed.json](completed.json). Earlier controls: [qualification](../qualification_v1/metrics.json). Full inputs and reproduction context: [benchmark README](../../README.md).
'''
    r.write(report/'RESULTS.md',text)
    print(json.dumps(dict(gates=gates,extended_controls=controls),indent=2))


if __name__=='__main__':main()
