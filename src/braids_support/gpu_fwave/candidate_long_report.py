"""Read-only before/after evidence for an isolated native-MIC long bend.

Retains the original long-run gates and frozen optimized CPU reference. This
does not adopt a solver, modify a reference, or establish experimental truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from . import long_bend_analysis as long

ORIGINAL = long.HERE / 'runs/gpu_bend20h_20261001_v1/gpu'


def candidate_identity(folder, manifest):
    folder = Path(folder).resolve()
    audit_folder = folder.with_name(folder.name + '_audit')
    audit = long.read_json(audit_folder / 'audit.json')
    plan_path = Path(audit['plan'])
    if long.sha(plan_path) != audit['plan_sha256']:
        raise ValueError('Candidate plan changed')
    plan = long.read_json(plan_path)
    if Path(plan['output_path']).resolve() != folder:
        raise ValueError('Audit belongs to another run')
    if long.sha(folder / 'source/audit_plan.json') != audit['plan_sha256']:
        raise ValueError('Executed frozen plan differs')
    for path, digest in {**plan['authenticated_files'], **plan['protected_files']}.items():
        if long.sha(path) != digest:
            raise ValueError('Authenticated reference or source changed: ' + path)
    for name, digest in audit['artifacts'].items():
        if Path(name).name != name or long.sha(audit_folder / name) != digest:
            raise ValueError('Candidate diagnostic artifact changed: ' + name)
    for name, digest in plan['baseline_source_sha256'].items():
        if manifest['source_sha256'].get(name) != digest:
            raise ValueError('Original numerical module changed: ' + name)
    for name, digest in plan['candidate_source_sha256'].items():
        if manifest['source_sha256'].get(name) != digest:
            raise ValueError('Candidate source differs from reviewed plan: ' + name)
    if audit['selection'] != plan['selection']:
        raise ValueError('Selection mismatch')
    chosen = plan['selection']
    if chosen['filter'] != 'native-mic' or chosen['backend'] != 'gpu' or chosen['automatic_backend_selection']:
        raise ValueError('Require explicit native-MIC GPU candidate')
    if (chosen['rtol'], chosen['atol'], chosen['maxiter']) != (1e-4, 1e-10, 500):
        raise ValueError('Original filter tolerance was changed')
    if audit.get('all_identities_unchanged') is not True:
        raise ValueError('Candidate provenance verification did not pass')
    if audit['accepted'] != manifest['steps'] or audit['rejected'] != manifest['rejected']:
        raise ValueError('Observed and worker step counts differ')
    return audit, plan, audit_folder


def retained_gates(comparison, audit, old_wall, trace=None):
    """Never treat a short smoke run as full qualification or hide audit cost."""
    seconds = float(audit['audit_total_wall_s'])
    worker = float(audit['worker_total_wall_s'])
    cpu = float(comparison['timing']['fast_cpu_mobile_wall_s'])
    if min(seconds, worker, cpu, old_wall) <= 0 or not np.isfinite([seconds, worker, cpu, old_wall]).all():
        raise ValueError('Finite positive timings required')
    if worker > seconds + 1e-6:
        raise ValueError('Inclusive audit time is smaller than worker time')
    gates = dict(full_original_20h_comparison=bool(comparison['full_direct_gpu_port_qualified']),
                 observer_completed=audit['status'] == 'completed',
                 source_identity_retained=audit.get('all_identities_unchanged') is True,
                 full_trace_verified=bool(trace and trace.get('verified') is True and trace.get('complete_20h') is True),
                 gpu_faster_than_optimized_cpu=seconds < cpu)
    return dict(gates=gates, passed=all(gates.values()),
                candidate_including_diagnostics_s=seconds,
                candidate_worker_only_s=worker,
                original_gpu_s=float(old_wall), optimized_cpu_s=cpu,
                speedup_vs_optimized_cpu=cpu / seconds, speedup_vs_original_gpu=old_wall / seconds,
                note='Measured run wall times, not alternating repeat benchmarks. Instrumentation, initialization, '
                     'output and final provenance hashing included in candidate audit time. '
                     'Input preparation and shared CPU hydraulic spin-up excluded. '
                     'Concurrent workload evidence must be reviewed separately; no automatic claim of exclusivity.')


def same_experiment(candidate, original, plan, manifest, old_manifest):
    if plan['baseline_source_sha256'] != old_manifest['source_sha256']:
        raise ValueError('Candidate baseline is not the preserved original GPU')
    if manifest['input_preparation_sha256'] != old_manifest['input_preparation_sha256']:
        raise ValueError('Candidate and original prepared input identities differ')
    initial_identity = {}
    for key in ('q', 'layers', 'secondary', 'bed', 'ledger_water', 'ledger_bed', 'ledger_secondary', 'ledger_source'):
        np.testing.assert_array_equal(candidate[key][0], original[key][0], err_msg='Changed initial field: ' + key)
        initial_identity[key] = True
    return initial_identity


def check_output_grid(candidate, original, cpu):
    expected = np.arange(121, dtype=float) * 24.
    for name, values in [('candidate', candidate), ('original GPU', original), ('optimized CPU', cpu)]:
        times = np.asarray(values, dtype=float)
        if (times.shape != expected.shape or not np.isfinite(times).all()
                or not np.allclose(times, expected, rtol=0, atol=1e-7)):
            raise ValueError('Require exactly the original 121 outputs: ' + name)
    return True


def render(destination, candidate, original, cpu, candidate_comparison, original_comparison, data, prepared,
           *, title='Filter-algorithm isolation: same tolerances, boundaries, f-wave and sediment reconstruction'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from .coupled_report import polygons
    mf = float(data['morphological_factor'])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for name, value, color in [('Original GPU', original_comparison, '#bd583a'),
                               ('Candidate GPU', candidate_comparison, '#2166ac')]:
        curves = value['curves']; hours = np.asarray(curves['time_s']) * mf / 3600
        maximum = np.asarray(curves['bed_maximum_m']) * 1000
        rms = np.asarray(curves['bed_rms_m']) * 1000
        axes[0, 0].plot(hours, maximum, color=color, label=name + ' max')
        axes[0, 0].plot(hours, rms, '--', color=color, label=name + ' RMS')
    axes[0, 0].axhline(5, color='.5', ls=':', label='Max limit 5 mm')
    axes[0, 0].axhline(3, color='.7', ls=':', label='RMS limit 3 mm')
    axes[0, 0].set(title='Every saved output vs the same optimized CPU', xlabel='Morphological hours',
                   ylabel='Bed difference (mm)', ylim=(0, None))
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=.2)
    for name, fields, color in [('Optimized CPU', cpu, '#333333'), ('Original GPU', original, '#bd583a'),
                                 ('Candidate GPU', candidate, '#2166ac')]:
        axes[0, 1].plot(fields['time'] * mf / 3600,
                        (fields['bed'][:, 3113] - fields['bed'][0, 3113]) * 1000, label=name, color=color)
    axes[0, 1].set(title='Original hotspot: inlet/wall corner cell 3113', xlabel='Morphological hours',
                   ylabel='Bed change from initial (mm)')
    axes[0, 1].axhline(0, color='.5', lw=.6); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=.2)
    peak_time = float(original_comparison['worst_bed_location']['time_s'])
    cells = polygons(prepared, data)
    region = (data['xy'][0] < 1.) & (data['xy'][1] > 0.)
    points = np.concatenate([np.asarray(cell) for cell, use in zip(cells, region) if use])
    xmin, ymin = points.min(axis=0); xmax, ymax = points.max(axis=0)
    reference_index = np.flatnonzero(abs(cpu['time'] - peak_time) < 1e-7)
    if len(reference_index) != 1:
        raise ValueError('Original peak has no exact CPU output')
    errors = []
    for fields in (original, candidate):
        index = np.flatnonzero(abs(fields['time'] - peak_time) < 1e-7)
        if len(index) != 1:
            raise ValueError('Candidate must include original peak time; no interpolation')
        errors.append((fields['bed'][index[0]] - cpu['bed'][reference_index[0]]) * 1000)
    cap = max(1e-6, *(float(abs(error[region]).max()) for error in errors))
    for ax, error, label in zip(axes[1], errors, ('Original GPU - CPU', 'Candidate GPU - CPU')):
        collection = PolyCollection(cells, array=error, cmap='RdBu_r', clim=(-cap, cap), edgecolors='none')
        ax.add_collection(collection); ax.set_aspect('equal')
        ax.set(xlim=(xmin - .03, xmax + .03), ylim=(ymin - .03, ymax + .03), xlabel='x (m)', ylabel='y (m)',
               title=f'{label}: inlet metre at {peak_time*mf/3600:.3f} h')
        fig.colorbar(collection, ax=ax, label='Signed bed difference (mm)', shrink=.85)
    fig.suptitle(title + '\nRaw cell values; common map scale; CPU consistency is not experimental validation')
    path = destination / 'BEFORE_AFTER.png'; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def run(folder, destination, *, animation=True):
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Preserve earlier analysis; choose a fresh report directory')
    candidate, manifest, data, prepared = long.load_worker(folder)
    audit, plan, audit_folder = candidate_identity(folder, manifest)
    if manifest.get('status') != 'completed' or abs(candidate['time'][-1] - 2880.) > 1e-7:
        raise ValueError('Full before/after report requires the completed 20h candidate')
    original, old_manifest, old_data, _ = long.load_worker(ORIGINAL)
    if old_manifest.get('status') != 'completed':
        raise ValueError('Original GPU reference is not complete')
    initial_identity = same_experiment(candidate, original, plan, manifest, old_manifest)
    for key in ('area', 'xy', 'owner', 'neighbour'):
        np.testing.assert_array_equal(data[key], old_data[key], err_msg='Changed geometry: ' + key)
    if plan.get('wrapper_schema') == 4:
        from .adaptive_native_optimized_trace import analyze as analyze_trace
    elif plan.get('wrapper_schema') == 2:
        from .adaptive_trace_analysis_v2 import analyze as analyze_trace
    else:
        from .adaptive_trace_analysis import analyze as analyze_trace
    trace = analyze_trace(folder, require_complete_20h=True)
    cpu = long.load_native(long.FAST_REFERENCE)
    exact_output_grid = check_output_grid(candidate['time'], original['time'], cpu['time'])
    outcome = long.run(folder, destination / 'standard', animation=animation)
    old_comparison, _, _ = long.comparison(original, cpu, data['area'], float(data['dry_tolerance']))
    result = dict(candidate_run=str(folder), candidate_audit_sha256=long.sha(audit_folder / 'audit.json'),
                  candidate_selection=plan['selection'], original_gpu_run=str(ORIGINAL),
                  original_gpu_manifest_sha256=long.sha(ORIGINAL / 'run.json'),
                  standard_comparison_sha256=long.sha(destination / 'standard/comparison.json'),
                  original_gpu_initial_fields_identical=initial_identity,
                  exact_original_121_output_grid=exact_output_grid,
                  original_gpu_comparison=old_comparison, candidate_gpu_comparison=outcome['comparisons']['fast_cpu'],
                  trace_verification=trace,
                  timing_and_gates=retained_gates(outcome, audit, float(old_manifest['total_wall_s']), trace),
                  experimental_truth_established=False, production_adopted=False, delft_modified=False)
    image = render(destination, candidate, original, cpu, outcome['comparisons']['fast_cpu'], old_comparison, data, prepared,
                   title='Native-MIC GPU candidate: same filter tolerances, boundary physics, f-wave and sediment reconstruction')
    result['before_after_figure'] = dict(file=image.name, sha256=long.sha(image))
    result['analysis_source_sha256'] = long.sha(__file__)
    long.write_json(destination / 'comparison.json', long.serializable(result))
    record = outcome['comparisons']['fast_cpu']; timing = result['timing_and_gates']
    text = ['NATIVE-MIC GPU LONG BEND: BEFORE/AFTER',
            f"Numerical and retained-speed gates passed: {timing['passed']}.",
            f"Maximum bed error: original {old_comparison['maximum_bed_difference_m']*1000:.6g} mm; candidate {record['maximum_bed_difference_m']*1000:.6g} mm.",
            f"Maximum area-weighted RMS: original {old_comparison['maximum_bed_rms_m']*1000:.6g} mm; candidate {record['maximum_bed_rms_m']*1000:.6g} mm.",
            f"Candidate inclusive wall time {timing['candidate_including_diagnostics_s']:.3f} s; optimized CPU {timing['optimized_cpu_s']:.3f} s; original GPU {timing['original_gpu_s']:.3f} s.",
            f"Speedup vs optimized CPU {timing['speedup_vs_optimized_cpu']:.4g}x; vs original GPU {timing['speedup_vs_original_gpu']:.4g}x.",
            timing['note'], '',
            'Only the GPU curvature linear-solve implementation changed. The tolerance, incoming secondary-flow rule,',
            'augmented f-wave solver, hydraulic integration and sediment reconstruction remained unchanged.',
            'Explicit executed implementation selection: ' + json.dumps(plan['selection'], sort_keys=True),
            'This is CPU/GPU port consistency, not proof that either model reproduces the experiment.',
            'The historical slower CPU uses different integration controls; its separate comparison remains in standard/REPORT.txt.',
            'No production/plugin adoption. Review the boundary isolation and conditioned routine evidence for causal interpretation.']
    (destination / 'REPORT.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return long.serializable(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--no-animation', action='store_true')
    args = parser.parse_args()
    result = run(args.gpu_output, args.report, animation=not args.no_animation)
    print(json.dumps(result['timing_and_gates'], indent=2))
