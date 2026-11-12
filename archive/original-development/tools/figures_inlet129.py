"""Compact, read-only figures from sealed milestone-129 natural-river results.

No solver/runtime imports or numerical reruns. Figures use the existing all-frame
metrics and raw final cell polygons; no smoothing or spatial interpolation.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/inlet129'
METHODS = ('critical', 'normal', 'stage', 'depth')
LABELS = {'critical': 'Critical-depth reference', 'normal': 'Mesh-derived normal reference',
          'stage': 'Water-elevation reference', 'depth': 'Water-depth reference'}
COLORS = dict(zip(METHODS, ('#0072B2', '#009E73', '#D55E00', '#CC79A7')))
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(metrics, res, method, backend='gpu', fine=False):
    group = f'natural{res}_final' + ('_fine' if fine else '')
    return next(r for r in metrics['runs'].values()
                if r['group'] == group and r['case'] == f'{res}_{method}_{backend}')


def pair(metrics, method, category):
    return next(p for p in metrics['pairs'].values()
                if p['category'] == category and
                list(p['configuration_a']['inlet_methods'].values())[0] == f'inlet.{method}_discharge')


def save(fig, name):
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


def accuracy(metrics, res):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained', sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        backend = pair(metrics, method, 'same_method_cpu_gpu')
        time = pair(metrics, method, 'same_method_timestep_sensitivity')
        for p, color, label in ((time, '#D55E00', 'GPU: CFL 0.5 versus 0.125'),
                                (backend, '#0072B2', 'CPU versus GPU: CFL 0.5')):
            rows = p['per_time']
            assert p['skipped_times'] == 0 and p['compared_times'] == 11
            ax.plot([r['time_s'] for r in rows], [1000*r['depth_max_m'] for r in rows],
                    '.-', color=color, label=label, linewidth=1.8)
        ax.text(.03, .96, f"Maximum instantaneous area-RMS:\n"
                f"CFL contrast {1000*time['depth_rms_m']:.4f} mm; "
                f"backend {1000*backend['depth_rms_m']:.4f} mm",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        ax.set(title=LABELS[method], xlabel='Simulation time (s)',
               ylabel='Maximum depth difference (mm)', ylim=(0, .75))
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle(f'{res} natural river | all four inlet references remain below 1 mm\n'
                 '11 matched output times, identical mesh supports; no skipped comparisons', fontsize=14)
    save(fig, f'ACCURACY_{res.upper()}.png')


def boundary_history(metrics, res):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained', sharex=True, sharey=True)
    datum = 464.0
    for ax, method in zip(axes.flat, METHODS):
        gpu = run(metrics, res, method)
        cpu = run(metrics, res, method, 'cpu')
        rows = gpu['boundary_per_time']
        t = [r['time_s'] for r in rows]
        ax.fill_between(t, [r['actual_trace_stage_min_m']-datum for r in rows],
                        [r['actual_trace_stage_max_m']-datum for r in rows],
                        color='#56B4E9', alpha=.25, label='GPU wet-face trace range')
        ax.plot(t, [r['mean_trace_stage_m']-datum for r in rows], '-', color='#0072B2',
                linewidth=2, label='GPU actual trace mean')
        ax.plot(t, [r['mean_trace_stage_m']-datum for r in cpu['boundary_per_time']], 'o',
                color='#000000', fillstyle='none', markersize=3.5, label='CPU actual trace mean')
        ax.plot(t, [r['requested_stage_m']-datum for r in rows], '--', color='#D55E00',
                label='Requested external reference')
        ax.text(.03, .04, f"Net Q = 11 m³/s; max |error| = {gpu['max_Q_error_m3_s']:.2g}\n"
                f"Max local backflow = {gpu['maximum_output_gross_backflow_m3_s']:.5f} m³/s",
                transform=ax.transAxes, va='bottom', fontsize=9,
                bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        ax.set(title=LABELS[method], xlabel='Simulation time (s)',
               ylabel='Water elevation minus 464 m (m)', ylim=(.55, 1.0))
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle(f'{res} inlet: exact net discharge with backwater-compatible traces\n'
                 'The chosen reference is not an imposed actual depth/elevation', fontsize=14)
    save(fig, f'BOUNDARY_HISTORY_{res.upper()}.png')


def final_arrays(summary):
    folder = Path(summary['folder']) / 'run/frames'
    row = read(folder / 'INDEX.json')['frames'][-1]
    with np.load(folder / row['file']) as src:
        q = src['q'].copy()
    with np.load(folder / row['epoch']) as src:
        mesh = {k: src[k].copy() for k in src.files}
    project = read(Path(summary['folder']) / 'PROJECT.json')
    return q, mesh, project


def inlet_maps(metrics, res):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout='constrained', sharex=True, sharey=True)
    for ax, method in zip(axes.flat, METHODS):
        r = run(metrics, res, method)
        q, g, project = final_arrays(r)
        domain = np.asarray(project['settings']['domain']['vertices_xy'])
        inlet_y = domain[:, 1].min()
        bottom = domain[np.isclose(domain[:, 1], inlet_y, rtol=0, atol=1e-8)]
        origin = np.array([bottom[:, 0].mean(), inlet_y])
        all_polys = [g['vertices'][g['offsets'][i]:g['offsets'][i+1]] - origin
                     for i in range(len(g['area']))]
        pick = [i for i, p in enumerate(all_polys) if p[:, 1].min() < 80]
        velocity = np.divide(q[2], q[0], out=np.zeros_like(q[0]), where=q[0] > 1e-6)
        dry = [i for i in pick if q[0, i] <= 1e-6]
        wet = [i for i in pick if q[0, i] > 1e-6]
        ax.add_collection(PolyCollection([all_polys[i] for i in dry], facecolors='#eeeeee',
                                        edgecolors='#888888', linewidths=.2))
        pc = PolyCollection([all_polys[i] for i in wet], array=velocity[wet],
                            cmap='RdBu', clim=(-2.5, 2.5), edgecolors='#777777', linewidths=.2)
        ax.add_collection(pc)
        ax.plot(domain[:, 0]-origin[0], domain[:, 1]-origin[1], color='#444444', linewidth=.9)
        ax.plot(bottom[:, 0]-origin[0], bottom[:, 1]-origin[1], color='#E69F00', linewidth=3)
        ax.set(xlim=(-65, 30), ylim=(-5, 80), aspect='equal', title=LABELS[method],
               xlabel='Easting from inlet midpoint (m)', ylabel='Northing from inlet (m)')
    fig.colorbar(pc, ax=list(axes.flat), label='Owner-cell northward velocity (m/s)', shrink=.7)
    fig.suptitle(f'{res} natural river | inlet and cropped walls at 100 s, GPU\n'
                 'Raw cells; grey = dry, orange = inlet. No spatial smoothing.', fontsize=14)
    save(fig, f'INLET_FLOW_{res.upper()}.png')


def stepping(all_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    x = np.arange(4)
    for ax, res in zip(axes[0], ('2m', '5m')):
        metrics = all_metrics[res]
        cpu = [run(metrics, res, m, 'cpu') for m in METHODS]
        gpu = [run(metrics, res, m) for m in METHODS]
        ax.bar(x-.24, [r['mean_hydraulic_dt_s'] for r in cpu], .24, color='#0072B2', label='CPU mean accepted dt')
        ax.bar(x, [r['mean_hydraulic_dt_s'] for r in gpu], .24, color='#009E73', label='GPU mean accepted dt')
        ax.bar(x+.24, [r['status']['hydraulic']['next_substep_hint_s'] for r in gpu], .24,
               color='#E69F00', label='GPU next-step hint at 100 s')
        ax.set_xticks(x, METHODS)
        ax.set(title=f'{res}: accepted steps, not output-interval size', ylabel='Timestep (s)', ylim=(0, .22))
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    for method in METHODS:
        rows = run(all_metrics['5m'], '5m', method)['boundary_per_time']
        t = [r['time_s'] for r in rows]
        axes[1, 0].plot(t, [r['cells'] for r in rows], '.-', color=COLORS[method], label=method)
        frequency = [r['maximum_frequency_s_inverse'] if r['maximum_frequency_s_inverse'] > 0 else np.nan
                     for r in rows]
        axes[1, 1].plot(t, frequency, '.-', color=COLORS[method], label=method)
    axes[1, 0].set(title='5m wet-front AMR: 252 → 512 cells', xlabel='Simulation time (s)', ylabel='Active cells')
    axes[1, 0].text(.04, .95, 'All four curves coincide.\n2m static mesh stays at 9,732 cells.',
                    transform=axes[1, 0].transAxes, va='top', fontsize=9)
    axes[1, 1].set(title='5m CFL-frequency diagnostic (valid samples only)',
                   xlabel='Simulation time (s)', ylabel='Maximum face-sum frequency (1/s)',
                   ylim=(1.8, 2.6), xlim=(0, 100))
    axes[1, 1].text(.04, .04, 'Gaps at 0, 30, 60, 100 s are uninitialized\nnew-owner diagnostics, not zero wave speed.',
                    transform=axes[1, 1].transAxes, va='bottom', fontsize=9)
    axes[1, 1].legend(fontsize=8, loc='upper left', ncol=2)
    for ax in axes[1]:
        ax.grid(alpha=.2)
    fig.suptitle('Natural-river stepping and mesh evolution | CFL 0.5', fontsize=14)
    save(fig, 'STEPPING_AND_MESH.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {res: read(ROOT / f'validation/_inlet129/analysis{res}_final/METRICS.json')
               for res in ('2m', '5m')}
    for res, data in metrics.items():
        accuracy(data, res)
        boundary_history(data, res)
        inlet_maps(data, res)
    stepping(metrics)
    print('Created seven compact figures in', OUT)


if __name__ == '__main__':
    main()
