"""CPU-only supplement to a completed candidate's existing long-run report.

Checks the case-specific CPU bed decoding, records analysis dependencies, and
reports hydraulic differences without introducing retrospective accuracy gates.
Simulation/reference files are read-only; output must be a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONSTANTS = dict(RHOS=2650., POROSITY=.4, RESERVOIR=.5, BULK=1590.)
BED_INDICES = np.array([[5, 6], [7, 8]], dtype=np.int32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_numeric(actual, expected, label, tolerance=1e-12):
    values = np.asarray(actual, dtype=float)
    require(values.size > 0 and np.isfinite(values).all()
            and np.allclose(values, expected, rtol=0, atol=tolerance),
            'CPU bed reconstruction assumption differs: ' + label)


def check_constants(data, references, live_constants):
    """Assert the fixed two-label/two-layer decoding against retained inputs."""
    for key, value in CONSTANTS.items():
        equal_numeric(live_constants[key], value, key)
    require(int(data['nfrac']) == 2 and int(data['nbed_layers']) == 2,
            'Expected two bed layers and two sediment labels')
    require(np.array_equal(data['bed_indices'], BED_INDICES), 'Native bed state columns differ')
    require(np.shape(data['rhos_fraction']) == np.shape(data['bulk_fraction']) == (2,),
            'Expected two material labels')
    equal_numeric(data['rhos_fraction'], CONSTANTS['RHOS'], 'prepared grain density')
    equal_numeric(data['bulk_fraction'], CONSTANTS['BULK'], 'prepared bulk density')
    inferred_porosity = 1. - np.asarray(data['bulk_fraction']) / np.asarray(data['rhos_fraction'])
    equal_numeric(inferred_porosity, CONSTANTS['POROSITY'], 'prepared inferred porosity')
    nc = len(data['area'])
    require(np.asarray(data['layer_mass']).shape == (2, 2, nc), 'Prepared layer layout differs')
    evidence = {}
    for name, reference in references.items():
        meta, geometry = reference['metadata'], reference['geometry']
        equal_numeric(meta['porosity'], CONSTANTS['POROSITY'], name + ' input porosity')
        equal_numeric(meta['mobile_thickness_m'], CONSTANTS['RESERVOIR'], name + ' input reservoir')
        require(reference['layers'].shape[1:] == (2, 2, nc), name + ' decoded layer layout differs')
        require(reference['q'].shape[1:] == (3, nc), name + ' hydraulic layout differs')
        equal_numeric(geometry['area'], data['area'], name + ' area')
        equal_numeric(geometry['xy'], np.asarray(data['xy']).T, name + ' cell order', 1e-10)
        equal_numeric(geometry['bed'] - data['rock'], CONSTANTS['RESERVOIR'], name + ' bedrock offset')
        thickness = reference['layers'][0].sum(axis=(0, 1)) / CONSTANTS['BULK']
        equal_numeric(thickness, CONSTANTS['RESERVOIR'], name + ' initial inventory thickness')
        decoded = (np.asarray(geometry['bed'])[None] - CONSTANTS['RESERVOIR']
                   + reference['layers'].sum(axis=(1, 2)) / CONSTANTS['BULK'])
        equal_numeric(reference['bed'], decoded, name + ' entire bed decoding')
        evidence[name] = dict(porosity=float(meta['porosity']),
            mobile_thickness_m=float(meta['mobile_thickness_m']),
            initial_inventory_thickness_max_error_m=float(abs(thickness - CONSTANTS['RESERVOIR']).max()),
            metadata_sha256=sha(reference['folder'] / 'input_provenance.json')
            if 'folder' in reference else None)
    return dict(passed=True, constants=CONSTANTS.copy(), zero_based_bed_state_indices=BED_INDICES.tolist(),
                layers=2, labels=2, prepared_rhos_fraction=np.asarray(data['rhos_fraction']).tolist(),
                prepared_bulk_fraction=np.asarray(data['bulk_fraction']).tolist(),
                prepared_inferred_porosity=inferred_porosity.tolist(), references=evidence)


def hydraulic_metrics(candidate, reference, area, dry):
    """Diagnostic differences only; no new numerical acceptance thresholds."""
    from .long_bend_analysis import common_times, velocity, weighted_rms
    gi, ri = common_times(candidate['time'], reference['time'])
    require(len(gi) == len(candidate['time']) == len(reference['time']), 'Hydraulic outputs must all align')
    left, right = candidate['q'][gi], reference['q'][ri]
    area = np.asarray(area, dtype=float)
    require(left.shape == right.shape == (len(gi), 3, len(area)), 'Hydraulic field shape differs')
    require(np.isfinite(left).all() and np.isfinite(right).all()
            and np.isfinite(area).all() and np.all(area > 0), 'Invalid hydraulic fields/area')
    require(np.isfinite(dry) and dry >= 0, 'Invalid dry threshold')
    dh = left[:, 0] - right[:, 0]
    dv = np.linalg.norm(velocity(left, dry) - velocity(right, dry), axis=1)
    mismatch = (left[:, 0] > dry) != (right[:, 0] > dry)
    curves = dict(time_s=np.asarray(candidate['time'])[gi],
        depth_maximum_m=np.max(abs(dh), axis=1), depth_area_rms_m=weighted_rms(dh, area),
        velocity_vector_maximum_m_s=dv.max(axis=1), velocity_vector_area_rms_m_s=weighted_rms(dv, area),
        wet_mask_disagreement_cells=mismatch.sum(axis=1), wet_mask_disagreement_area_m2=mismatch @ area)
    maxima = {key: float(value.max()) for key, value in curves.items() if key != 'time_s'}
    return dict(diagnostic_only=True, accuracy_gate_added=False, accuracy_qualified=None,
        compared_frames=len(gi), temporal_interpolation=False, dry_tolerance_m=float(dry),
        curves=curves, maxima_over_saved_frames=maxima,
        minimum_candidate_depth_m=float(left[:, 0].min()),
        minimum_reference_depth_m=float(right[:, 0].min()),
        note='RMS is area-weighted per frame; maxima are over all saved frames. Velocity is a vector '
             'difference with velocity zeroed where depth <= dry tolerance. Unsaved intermediate states are not assessed.')


def common_depth_upper(fields):
    depths = [np.asarray(field['q'])[-1, 0] for field in fields]
    require(bool(depths) and all(d.size and np.isfinite(d).all() for d in depths), 'Invalid final depth fields')
    return max(.001, *(float(depth.max()) for depth in depths))


def dependency_hashes():
    """Record actual imported repository modules, not guessed loader filenames."""
    result = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        path = Path(path).resolve()
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix == '.py' and relative.parts[0] in ('validation', 'braids_gpu', 'braids'):
            result[str(path)] = sha(path)
    return result


def render_depth(destination, candidate, references, prepared, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from .coupled_report import polygons
    fields = [candidate, references['fast_cpu'], references['historical']]
    upper = common_depth_upper(fields)
    cells = polygons(prepared, data)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout='constrained')
    labels = ('Candidate GPU', 'Optimized CPU (direct port reference)', 'Historical CPU (different controls)')
    for ax, field, label in zip(axes, fields, labels):
        collection = PolyCollection(cells, array=field['q'][-1, 0], cmap='Blues',
                                    clim=(0., upper), edgecolors='none')
        ax.add_collection(collection); ax.autoscale(); ax.set_aspect('equal')
        ax.set(title=label, xlabel='x (m)', ylabel='y (m)')
    fig.colorbar(collection, ax=axes, label='Water depth (m); identical scale', shrink=.8)
    fig.suptitle('Final depth at 20 morphological hours / 2880 hydraulic s, MF25\n'
                 'Shared upper limit spans all three states; diagnostic, not experimental validation')
    path = destination / 'FINAL_DEPTH_COMMON_SCALE.png'
    fig.savefig(path, dpi=150); plt.close(fig)
    return dict(file=path.name, sha256=sha(path), vmin_m=0., vmax_m=upper,
                final_depth_maximum_m={name: float(field['q'][-1, 0].max())
                    for name, field in zip(('candidate', 'fast_cpu', 'historical'), fields)})


def run(candidate_folder, report_folder, destination=None):
    """Run only after the candidate and its normal final report are complete."""
    from . import candidate_long_report as report
    from . import long_bend_analysis as long
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    candidate_folder, report_folder = Path(candidate_folder).resolve(), Path(report_folder).resolve()
    destination = (Path(destination).resolve() if destination else report_folder / 'analysis_review')
    require(not destination.exists(), 'Preserve existing evidence: use a fresh output directory')
    document_path = report_folder / 'comparison.json'
    document = long.read_json(document_path)
    require(Path(document['candidate_run']).resolve() == candidate_folder, 'Final report belongs to another candidate')
    standard = report_folder / 'standard/comparison.json'
    require(sha(standard) == document['standard_comparison_sha256'], 'Standard comparison changed')
    candidate, manifest, data, prepared = long.load_worker(candidate_folder)
    require(manifest.get('status') == 'completed' and abs(candidate['time'][-1] - 2880.) <= 1e-7,
            'Require completed original 20h candidate')
    audit, plan, audit_folder = report.candidate_identity(candidate_folder, manifest)
    require(sha(audit_folder / 'audit.json') == document['candidate_audit_sha256'], 'Final report audit changed')
    references = dict(fast_cpu=long.load_native(long.FAST_REFERENCE), historical=long.load_native(previous.REFERENCE))
    retained = long.read_json(standard)
    require(all(value['identity'] == retained['reference_identities'][key]
                for key, value in references.items()), 'CPU reference differs from completed final report')
    for source in (candidate_folder, prepared, *(value['folder'] for value in references.values())):
        require(not destination.is_relative_to(Path(source).resolve()),
                'Supplement must not be written inside simulation/reference folders')
    expected = np.arange(121, dtype=float) * 24.
    for name, field in [('candidate', candidate), *references.items()]:
        require(field['time'].shape == expected.shape
                and np.allclose(field['time'], expected, rtol=0, atol=1e-7), name + ' must contain original 121 outputs')
    constants = check_constants(data, references, {key: getattr(previous.r, key) for key in CONSTANTS})
    metrics = {key: hydraulic_metrics(candidate, value, data['area'], float(data['dry_tolerance']))
               for key, value in references.items()}
    # Resolve actual decoder identities, including indirect frame loader/writer modules.
    decoder_modules = {label: str(Path(inspect.getfile(obj)).resolve()) for label, obj in
        [('native_loader', long.load_native), ('input_verifier', previous.verify_inputs),
         ('native_frame_decoder', previous.r.shared.read_frames),
         ('native_frame_selector', previous.r.shared.native_frames_path)]}
    before = dependency_hashes()
    destination.mkdir(parents=True, exist_ok=False)
    figure = render_depth(destination, candidate, references, prepared, data)
    after = dependency_hashes()
    require(all(after.get(path) == value for path, value in before.items()), 'Analysis dependency changed during review')
    result = dict(schema=1, candidate_run=str(candidate_folder), final_report=str(report_folder),
        final_report_sha256=sha(document_path), standard_comparison_sha256=sha(standard),
        candidate_plan_sha256=sha(audit['plan']), candidate_manifest_sha256=sha(candidate_folder / 'run.json'),
        constants=constants, hydraulic_diagnostics=metrics, analysis_dependency_sha256=after,
        cpu_decoder_modules=decoder_modules, reference_identities={key: value['identity'] for key, value in references.items()},
        supplementary_depth_figure=figure, original_gate_results_unchanged=True,
        retrospective_hydraulic_gates_added=False, solver_or_reference_modified=False, gpu_executed=False,
        historical_comparison_note='Historical CPU has different numerical controls/initial water; only optimized CPU is the direct port reference.')
    long.write_json(destination / 'analysis_review.json', long.serializable(result))
    lines = ['COMPLETED CANDIDATE: ANALYSIS REVIEW',
             'CPU bed decoding constants/layout verified. No retrospective hydraulic accuracy gates added.',
             'All 121 original output times compared without interpolation. Results are diagnostics, not experimental braids_support.']
    for name, values in metrics.items():
        peak = values['maxima_over_saved_frames']
        lines.append(f"{name}: depth max {peak['depth_maximum_m']:.9g} m; area-RMS max {peak['depth_area_rms_m']:.9g} m; "
                     f"velocity-vector max {peak['velocity_vector_maximum_m_s']:.9g} m/s; area-RMS max {peak['velocity_vector_area_rms_m_s']:.9g} m/s; "
                     f"wet-mask disagreement max {peak['wet_mask_disagreement_cells']:g} cells / {peak['wet_mask_disagreement_area_m2']:.9g} m2.")
    lines += [f"Final depth map common scale: 0 to {figure['vmax_m']:.9g} m.", result['historical_comparison_note']]
    (destination / 'REPORT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return long.serializable(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    outcome = run(args.gpu_output, args.report, args.destination)
    print(json.dumps(dict(constants_verified=outcome['constants']['passed'],
                          retrospective_hydraulic_gates_added=False), indent=2))
