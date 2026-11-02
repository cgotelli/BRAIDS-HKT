"""Three-way, read-only full bend audit of the isolated outlet-bound repair.

Original CPU remains the mandatory agreement reference unless an explicit
user-approval record authorizes Corrected CPU. Both comparisons always remain
visible. This report never promotes a solver or certifies contended timing.
"""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
from . import long_bend_analysis as long
from . import candidate_analysis_review as review
from . import outlet_bounds_cpu20h as cpu

LABELS = dict(original='Original CPU', corrected='Corrected CPU', gpu='GPU')
EXPECTED_LIMITS = dict(bed_rms_m=.003, bed_maximum_m=.005, water_budget_m3=1e-6,
    sediment_budget_kg=1e-5, secondary_budget=1e-8, minimum_inventory_kg_m2=-1e-10)
TIMES = np.arange(121, dtype=float)*24.
require = review.require


def trace_auditor(plan):
    schema=plan.get('wrapper_schema');entrypoint=plan.get('wrapper_entrypoint')
    if schema==6 and entrypoint=='outlet_bounds_gpu_candidate_v2.py':
        from .outlet_bounds_gpu_trace_v2 import analyze
    elif schema==7 and entrypoint=='outlet_bounds_tight_gpu_candidate.py':
        from .outlet_bounds_tight_gpu_trace import analyze
    else:raise ValueError('Only reviewed explicit outlet schema6 or7 is supported')
    return analyze


def check_gpu_controls(manifest):
    require(manifest['status'] == 'completed' and manifest['end_s'] == 2880.
            and np.isfinite(manifest['final_time_s'])
            and abs(manifest['final_time_s']-2880.) <= 1e-7
            and (manifest['dtmax_s'], manifest['output_every_s'], manifest['cfl'], manifest['morfac']) == (.0125,24.,.8,25.),
            'GPU experiment controls/completion differ')
    return True


def analysis_dependencies():
    """Include indirect native decoder, which lives outside the validation package."""
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    hashes = review.dependency_hashes()
    for function in (previous.r.shared.read_frames, previous.verify_inputs, long.load_native,
                     long.load_worker, cpu.read_history, review.check_constants):
        path = Path(inspect.getfile(function)).resolve(); hashes[str(path)] = long.sha(path)
    hashes[str(Path(__file__).resolve())] = long.sha(__file__)
    return hashes


def check_grid(fields):
    for name, value in fields.items():
        times = np.asarray(value['time'])
        require(times.shape == TIMES.shape and np.isfinite(times).all()
                and np.allclose(times, TIMES, rtol=0, atol=1e-7),
                'Require all121 original outputs with no interpolation: '+name)
    return True


def choose_reference(policy, approval, plan_sha, runtime_sha):
    require(policy in ('original', 'corrected'), 'Unknown reference policy')
    if policy == 'original':
        require(approval is None, 'Approval record supplied but corrected policy was not selected')
        return dict(selected='original', original_CPU_mandatory=True, user_approval=None)
    require(isinstance(approval, dict) and approval.get('user_approved') is True,
            'Corrected reference requires explicit user-approval evidence')
    require(approval.get('reference_policy') == 'corrected'
            and approval.get('corrected_plan_sha256') == plan_sha
            and approval.get('corrected_executable_sha256') == runtime_sha,
            'Approval does not identify this corrected experiment')
    require(isinstance(approval.get('user_statement'), str) and approval['user_statement'].strip()
            and isinstance(approval.get('approval_context'), str) and approval['approval_context'].strip(),
            'Record the actual user statement and its context; do not invent approval')
    return dict(selected='corrected', original_CPU_mandatory=False, user_approval=approval,
                original_comparison_retained=True, numerical_limits_unchanged=True)


def evaluate(fields, area, dry, *, reference='original'):
    require(set(fields) == set(LABELS), 'Three named models are required')
    require(long.LIMITS == EXPECTED_LIMITS, 'Original declared accuracy/budget limits changed')
    require(reference in ('original', 'corrected'), 'Unknown agreement reference')
    check_grid(fields)
    initial = {}
    for name in ('corrected', 'gpu'):
        initial[name] = {}
        for key in ('q', 'layers', 'bed', 'secondary'):
            left, right = fields[name][key][0], fields['original'][key][0]
            require(left.dtype == right.dtype == np.float64 and left.shape == right.shape,
                    'Initial FP64 layout differs: '+name+'/'+key)
            same = bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))
            require(same, 'Initial state is not bitwise identical: '+name+'/'+key)
            initial[name][key] = same
    for name, value in fields.items():
        for key in ('ledger_water', 'ledger_bed', 'ledger_secondary', 'ledger_source'):
            require(np.all(np.asarray(value[key])[0] == 0.), 'Initial ledger is not zero: '+name+'/'+key)
    pairs = dict(gpu_vs_original=('gpu', 'original'), gpu_vs_corrected=('gpu', 'corrected'),
                 corrected_vs_original=('corrected', 'original'))
    comparisons = {key: long.comparison(fields[a], fields[b], area, dry)[0]
                   for key, (a, b) in pairs.items()}
    budgets = {name: long.budgets(value, area, dry) for name, value in fields.items()}
    hydraulics = {key: review.hydraulic_metrics(fields[a], fields[b], area, dry)
                  for key, (a, b) in pairs.items()}
    selected = comparisons['gpu_vs_'+reference]
    gates = dict(full121_outputs=True, initial_fields_bitwise=True,
        bed_maximum_every_frame=selected['gates']['every_common_frame_bed_maximum'],
        bed_area_rms_every_frame=selected['gates']['every_common_frame_bed_rms'],
        all_three_models_own_budgets_and_nonnegativity=all(v['passed'] for v in budgets.values()))
    return dict(comparisons=comparisons, conservation=budgets, hydraulic_diagnostics=hydraulics,
                initial_fields_bitwise_equal=initial, selected_reference=reference,
                numerical_gates=gates, numerical_passed=all(gates.values()),
                limits=dict(EXPECTED_LIMITS), temporal_interpolation=False,
                hydraulic_accuracy_gate_added=False)


def verify_cpu(plan_path, expected_sha):
    path = Path(plan_path).resolve()
    require(long.sha(path) == expected_sha, 'Reviewed correctedCPU plan SHA differs')
    plan = long.read_json(path)
    require(plan['tag'] == cpu.TAG and Path(plan['original_reference']).resolve() == long.FAST_REFERENCE.resolve(),
            'Unexpected original/corrected CPU experiment')
    require((plan['endpoint_s'], plan['output_s'], plan['output_frames'], plan['morfac'],
             plan['cfl'], plan['dt_ceiling_s']) == (2880., 24., 121, 25., .8, .0125),
            'Corrected CPU controls differ')
    cpu.verify_files({**plan['authenticated_files'], **plan['protected_files']})
    identity, _ = cpu.verify_runtime()
    require(identity == plan['runtime_identity'] and plan['executable_sha256'] == cpu.EXE_SHA,
            'Corrected CPU runtime changed')
    case = Path(plan['case']); executed = long.read_json(case/'execution.json')
    require(executed.get('status') == 'completed' and executed.get('return_code') == 0
            and executed.get('plan_sha256') == expected_sha and executed.get('executable_sha256') == cpu.EXE_SHA
            and executed.get('source_sha256') == identity['source_sha256']
            and executed.get('threads') == 8 and executed.get('environment') == plan['environment'],
            'Corrected CPU execution incomplete or differs from reviewed controls')
    original_execution = long.read_json(long.FAST_REFERENCE/'execution.json')
    require(original_execution['return_code'] == 0
            and original_execution['executable_sha256'] == cpu.base.EXE_SHA,
            'Frozen original CPU runtime differs')
    return plan, case


def shared_scales(fields):
    dz = [v['bed']-v['bed'][0] for v in fields.values()]
    require(all(np.isfinite(v).all() for v in dz), 'Nonfinite bed-map data')
    return dict(bed_change_symmetric_m=max(1e-9, *(float(abs(v).max()) for v in dz)),
                final_depth_upper_m=review.common_depth_upper(list(fields.values())))


def worst_maps(fields):
    maps = []
    for left,right in (('gpu','original'),('gpu','corrected'),('corrected','original')):
        delta = fields[left]['bed']-fields[right]['bed']
        require(np.isfinite(delta).all(), 'Nonfinite worst-time difference')
        frame,cell = np.unravel_index(np.argmax(abs(delta)),delta.shape)
        maps.append(dict(left=left,right=right,frame=int(frame),cell=int(cell),
            time_s=float(fields[left]['time'][frame]),maximum_m=float(abs(delta[frame,cell])),
            signed_peak_m=float(delta[frame,cell]),values_m=delta[frame]))
    return maps,max(1e-9,*(value['maximum_m'] for value in maps))


def render(destination, fields, result, data, prepared, *, animation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.animation import FuncAnimation, PillowWriter
    from .coupled_report import polygons
    cells = polygons(prepared, data); scales = shared_scales(fields); records = []
    hours = TIMES*25/3600
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained'); axes=axes.ravel()
    for name, value in result['comparisons'].items():
        curve = value['curves']; left, right = name.split('_vs_'); label = LABELS[left]+' vs '+LABELS[right]
        axes[0].plot(hours, np.asarray(curve['bed_maximum_m'])*1000, label=label)
        axes[1].plot(hours, np.asarray(curve['bed_rms_m'])*1000, label=label)
    axes[0].axhline(5, color='.5', ls=':'); axes[1].axhline(3, color='.5', ls=':')
    axes[0].set(title='Maximum cell bed difference', ylabel='mm', ylim=(0, None))
    axes[1].set(title='Area-weighted RMS bed difference', ylabel='mm', ylim=(0, None))
    for ax,cell in zip(axes[2:],(3113,4911)):
        for name, value in fields.items():
            ax.plot(hours, (value['bed'][:,cell]-value['bed'][0,cell])*1000, label=LABELS[name])
        ax.set(title=f'Inlet/wall corner{cell}', ylabel='Bed change (mm)')
    for ax in axes: ax.set_xlabel('Morphological hours'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('Outlet-bound repair: three-way diagnostic; no reference replacement without explicit approval\n'
                 +result['gpu_numerical_controls_label'])
    path = destination/'COMPARISONS.png'; fig.savefig(path, dpi=150); plt.close(fig)
    records.append(dict(file=path.name, sha256=long.sha(path)))
    maps,cap_m=worst_maps(fields)
    fig,axes=plt.subplots(1,3,figsize=(12,5.1),layout='constrained')
    for ax,entry in zip(axes,maps):
        image=PolyCollection(cells,array=entry['values_m']*1000,cmap='RdBu_r',
            clim=(-cap_m*1000,cap_m*1000),edgecolors='none')
        ax.add_collection(image);ax.autoscale();ax.set_aspect('equal')
        x,y=data['xy'][:,entry['cell']]
        ax.scatter([x],[y],s=60,facecolors='none',edgecolors='black',linewidths=1.2,zorder=5)
        ax.annotate(f"cell{entry['cell']}",(x,y),xytext=(5,8),textcoords='offset points',fontsize=7,
            bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=1))
        ax.set(title=f"{LABELS[entry['left']]} − {LABELS[entry['right']]}\n"
            f"{entry['time_s']*25/3600:.3f}h; max|Δz|={entry['maximum_m']*1000:.4g}mm",
            xlabel='x(m)',ylabel='y(m)')
    fig.colorbar(image,ax=axes,label='Signed bed difference(mm); common symmetric scale',shrink=.8)
    fig.suptitle('Each comparison at its own worst saved time; circle marks maximum-error cell\n'
                 +result['gpu_numerical_controls_label']+'; locations do not establish localized causality')
    path=destination/'WORST_TIME_BED_DIFFERENCES.png';fig.savefig(path,dpi=160);plt.close(fig)
    records.append(dict(file=path.name,sha256=long.sha(path),colour_limits_mm=(-cap_m*1000,cap_m*1000),
        panels=[{key:value for key,value in entry.items() if key!='values_m'} for entry in maps]))
    for kind, filename, cmap, limits in (
            ('bed', 'FINAL_BED_CHANGE.png', 'RdBu_r', (-scales['bed_change_symmetric_m']*1000, scales['bed_change_symmetric_m']*1000)),
            ('depth', 'FINAL_DEPTH.png', 'Blues', (0., scales['final_depth_upper_m']))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout='constrained')
        for ax, (name, value) in zip(axes, fields.items()):
            values = (value['bed'][-1]-value['bed'][0])*1000 if kind == 'bed' else value['q'][-1,0]
            image = PolyCollection(cells, array=values, cmap=cmap, clim=limits, edgecolors='none')
            ax.add_collection(image); ax.autoscale(); ax.set_aspect('equal')
            ax.set(title=LABELS[name], xlabel='x (m)', ylabel='y (m)')
        fig.colorbar(image, ax=axes, label='Bed change (mm)' if kind == 'bed' else 'Depth (m)', shrink=.8)
        fig.suptitle('Final state at20 morphological hours; identical colour scale spanning all three models\n'
                     +result['gpu_numerical_controls_label'])
        path = destination/filename; fig.savefig(path, dpi=150); plt.close(fig)
        records.append(dict(file=path.name, sha256=long.sha(path), colour_limits=limits))
    if animation:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), layout='constrained'); images = []
        cap = scales['bed_change_symmetric_m']*1000
        for ax, name in zip(axes, fields):
            image = PolyCollection(cells, array=np.zeros(len(data['area'])), cmap='RdBu_r',
                                   clim=(-cap, cap), edgecolors='none')
            ax.add_collection(image); ax.autoscale(); ax.set_aspect('equal')
            ax.set(title=LABELS[name], xlabel='x (m)', ylabel='y (m)'); images.append(image)
        fig.colorbar(images[-1], ax=axes, label='Bed change (mm); fixed scale for all121frames/models', shrink=.8)
        title = fig.suptitle('')
        def update(index):
            for image, value in zip(images, fields.values()):
                image.set_array((value['bed'][index]-value['bed'][0])*1000)
            title.set_text(f'{hours[index]:.3f} morphological hours / {TIMES[index]:.0f} hydraulic s; MF25\n'
                           +result['gpu_numerical_controls_label'])
            return images+[title]
        movie = FuncAnimation(fig, update, frames=121, interval=125, blit=False)
        path = destination/'BED_EVOLUTION.gif'; movie.save(path, writer=PillowWriter(fps=8), dpi=90); plt.close(fig)
        records.append(dict(file=path.name, sha256=long.sha(path), frames=121, colour_limits_mm=(-cap,cap)))
    return dict(scales=scales, artifacts=records)


def run(gpu_output, destination, cpu_plan, cpu_plan_sha256, *, reference_policy='original',
        reference_approval=None, animation=True):
    from braids_support.sediment.lfm_bend_2d import cold_optimized_comparison as previous
    folder, destination = Path(gpu_output).resolve(), Path(destination).resolve()
    require(not destination.exists(), 'Fresh report directory required')
    gpu_audit_path=folder.with_name(folder.name+'_audit')/'audit.json'
    gpu_audit=long.read_json(gpu_audit_path)
    require(long.sha(gpu_audit['plan'])==gpu_audit['plan_sha256'],'Reviewed GPU plan changed')
    gpu_plan=long.read_json(gpu_audit['plan']);analyze_trace=trace_auditor(gpu_plan)
    dependencies_before = analysis_dependencies()
    plan, corrected_folder = verify_cpu(cpu_plan, cpu_plan_sha256)
    forbidden = (folder, corrected_folder, long.FAST_REFERENCE.resolve(), cpu.RUNTIME.resolve(), cpu.base.RUNTIME.resolve())
    require(not any(destination == p or p in destination.parents for p in forbidden), 'Report must not alter simulation/reference folders')
    approval = long.read_json(reference_approval) if reference_approval else None
    policy = choose_reference(reference_policy, approval, cpu_plan_sha256, cpu.EXE_SHA)
    gpu, manifest, data, prepared = long.load_worker(folder)
    require(not destination.is_relative_to(Path(prepared).resolve()), 'Report must not alter prepared GPU inputs')
    trace = analyze_trace(folder, require_complete_20h=True)
    check_gpu_controls(manifest)
    original_prepared = long.HERE/'runs/gpu_bend20h_20261001_v1/prepared/preparation_receipt.json'
    require(manifest['input_preparation_sha256'] == long.sha(original_prepared),
            'GPU input is not the preserved original cold prepared experiment')
    fields = dict(original=cpu.read_history(long.FAST_REFERENCE), corrected=cpu.read_history(corrected_folder), gpu=gpu)
    observed = {str(folder/'run.json'):long.sha(folder/'run.json')}
    audit_folder = folder.with_name(folder.name+'_audit'); audit = long.read_json(audit_folder/'audit.json')
    observed[str(audit_folder/'audit.json')] = trace['audit_sha256']
    observed[str(Path(audit['plan']))] = trace['plan_sha256']
    observed[str(original_prepared)] = manifest['input_preparation_sha256']
    for name,digest in audit['artifacts'].items(): observed[str(audit_folder/name)] = digest
    for item in manifest['frames']:
        observed[str(folder/item['file'])] = item['sha256']
    for name,digest in manifest['source_sha256'].items(): observed[str(folder/'source'/name)] = digest
    for native_folder in (long.FAST_REFERENCE,corrected_folder):
        for name in ('frames.bin','checkpoint.bin','secondary_flow.txt','secondary_flow_budget.txt','execution.json'):
            observed[str(native_folder/name)] = long.sha(native_folder/name)
    live = {name: getattr(previous.r,name) for name in review.CONSTANTS}
    constants = review.check_constants(data, {key:fields[key] for key in ('original','corrected')}, live)
    result = evaluate(fields, data['area'], float(data['dry_tolerance']), reference=policy['selected'])
    for value in result['comparisons'].values():
        value['worst_bed_location']['xy_m'] = data['xy'][:,value['worst_bed_location']['cell']]
    result.update(reference_policy=policy, CPU_decoding_checks=constants, trace_verification=trace,
        full_accuracy_passed=bool(result['numerical_passed'] and trace['verified'] and trace['complete_20h']),
        model_labels=LABELS, completed_frames=121, final_hydraulic_time_s=manifest['final_time_s'],
        nominal_final_hydraulic_time_s=2880., endpoint_roundoff_tolerance_s=1e-7, morphological_hours=20.,
        GPU_wrapper_schema=gpu_plan['wrapper_schema'],GPU_selection=gpu_plan['selection'],
        GPU_requested_filter_controls=gpu_plan['requested_filter_controls'],
        GPU_effective_filter_controls=gpu_plan['effective_filter_controls'],
        gpu_numerical_controls_label='GPU: outlet extension1; MG rtol='+format(gpu_plan['effective_filter_controls']['rtol'],'.0e')+
            ', atol='+format(gpu_plan['effective_filter_controls']['atol'],'.0e')+'; original CPU filter unchanged',
        experimental_truth_established=False, production_adopted=False, delft_modified=False,
        performance_qualified=False, accuracy_is_not_performance_qualification=True,
        measured_wall_times_s=dict(original_CPU=fields['original']['execution']['wall_s'],
            corrected_CPU=fields['corrected']['execution']['wall_s'], GPU_worker=trace['worker_total_wall_s'],
            GPU_inclusive_audit=trace['audit_total_wall_s']),
        timing_note='Corrected CPU overlaps independent diagnostic jobs. Elapsed times are observations, not an uncontended comparison or speed acceptance.',
        runtime_input_identity=dict(cpu_plan=str(Path(cpu_plan).resolve()), cpu_plan_sha256=cpu_plan_sha256,
            original_cpu=fields['original']['identity'], corrected_cpu=fields['corrected']['identity'],
            original_executable_sha256=cpu.base.EXE_SHA, corrected_executable_sha256=cpu.EXE_SHA,
            gpu_run_sha256=long.sha(folder/'run.json'), GPU_source_sha256=manifest['source_sha256'],
            GPU_input_preparation_sha256=manifest['input_preparation_sha256']),
        observed_output_and_runtime_sha256=observed,
        morphology={name:long.morphology(value, fields['original']['geometry']) for name,value in fields.items()})
    if reference_approval:
        result['reference_approval_evidence'] = dict(file=str(Path(reference_approval).resolve()), sha256=long.sha(reference_approval))
    destination.mkdir(parents=True)
    result['figures'] = render(destination, fields, result, data, prepared, animation=animation)
    result['analysis_dependency_sha256_before'] = dependencies_before
    result['analysis_dependency_sha256'] = analysis_dependencies()
    cpu.verify_files(dependencies_before)
    verify_cpu(cpu_plan,cpu_plan_sha256)
    cpu.verify_files(observed)
    long.write_json(destination/'comparison.json', long.serializable(result))
    lines = ['OUTLET-BOUND REPAIR: ORIGINAL CPU / CORRECTED CPU / GPU',
        f"Mandatory agreement reference: {LABELS[policy['selected']]}. Full accuracy checks passed: {result['full_accuracy_passed']}.",
        result['gpu_numerical_controls_label'],
        'All121 original outputs compared, no interpolation. Bed limits remain5mm maximum and3mm area-RMS at every frame.']
    for name, value in result['comparisons'].items():
        lines.append(f"{name}: max {value['maximum_bed_difference_m']*1000:.9g}mm; maximum area-RMS {value['maximum_bed_rms_m']*1000:.9g}mm; agreement gates {value['passed']}.")
    lines += [result['timing_note'], 'Hydraulic and wet-mask differences are diagnostics; no retrospective hydraulic accuracy gates.',
        'No automatic production adoption, experimental validation, reference replacement or performance qualification.']
    (destination/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return long.serializable(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-output',type=Path,required=True); parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--cpu-plan',type=Path,required=True); parser.add_argument('--cpu-plan-sha256',required=True)
    parser.add_argument('--reference-policy',choices=('original','corrected'),default='original')
    parser.add_argument('--reference-approval',type=Path); parser.add_argument('--no-animation',action='store_true')
    args = parser.parse_args()
    result = run(args.gpu_output,args.report,args.cpu_plan,args.cpu_plan_sha256,
        reference_policy=args.reference_policy,reference_approval=args.reference_approval,animation=not args.no_animation)
    print(json.dumps(dict(reference=result['reference_policy']['selected'],accuracy_passed=result['full_accuracy_passed'],
                          performance_qualified=False),indent=2))
