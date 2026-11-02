"""Read-only schema7 trace/provenance checks with truthful tighter-MG controls."""
from pathlib import Path
import numpy as np
from . import adaptive_trace_analysis as original
from . import adaptive_multilevel_trace as mg
from . import outlet_bounds_tight_gpu_candidate as candidate

require,verify,read_json,child=original.require,original.verify,original.read_json,original.child


def analyze(output_path,*,require_complete_20h=False):
    output=Path(output_path).resolve();diagnostics=output.with_name(output.name+'_audit')
    audit=read_json(diagnostics/'audit.json')
    require(audit.get('all_identities_unchanged') is True and audit.get('status') in ('completed','step_limit','cancelled'),
            'Candidate audit did not complete with intact provenance')
    verify(audit['plan'],audit['plan_sha256']);plan=read_json(audit['plan']);candidate.validate_plan(plan)
    require(plan.get('base_wrapper_archive')=='audit_driver.py' and plan.get('original_numeric_sources_unchanged') is True,
            'Original source/archive controls differ')
    require(Path(plan['output_path']).resolve()==output and plan['selection']==audit['selection'],'Plan/output/selection differs')
    require((plan['physical_start_s'],plan['cfl'],plan['morfac'],plan['dt_max_s'],plan['output_every_s'],plan['accepted_CFL_maximum'])==
            (0.,.8,25.,.0125,24.,.8),'Original cold controls changed')
    require(audit.get('production_modified') is False and audit.get('delft_modified') is False,'Reference modified')
    require(audit['protected_files']==plan['protected_files'],'Protected reference set differs')
    for path,digest in {**plan['authenticated_files'],**plan['protected_files']}.items():verify(path,digest)
    expected_artifacts={'step_trials.jsonl','filter_calls.jsonl','filter_controls.json','outlet_control.json','graph_outlet_control.json','stop_control.json'}
    require(expected_artifacts<=audit['artifacts'].keys(),'Missing executed scalar/control evidence')
    for name,digest in audit['artifacts'].items():verify(child(diagnostics,name),digest)
    source=output/'source';verify(source/'manifest.json',audit['frozen_manifest_sha256']);manifest=read_json(source/'manifest.json')
    ids=[digest for path,digest in plan['authenticated_files'].items() if Path(path).name=='adaptive_filter_candidate.py']
    require(len(ids)==1,'Missing unique base driver')
    require(manifest=={**plan['baseline_source_sha256'],**plan['candidate_source_sha256'],
        'audit_driver.py':ids[0],'audit_plan.json':audit['plan_sha256']},'Archived executed source set differs')
    for name,digest in manifest.items():verify(child(source,name),digest)
    for name in candidate.HELPERS:
        digest=original.digest(candidate.HERE/name)
        require(plan['candidate_source_sha256'].get(name)==digest and
                plan['authenticated_files'].get(str(candidate.HERE/name))==digest,'Unpinned Python ancestry: '+name)
    for name,digest in plan['baseline_source_sha256'].items():verify(child(plan['baseline_source'],name),digest)
    require(plan['active_filter_cuda']=='secondary_curvature.cu' and
            manifest['secondary_curvature.cu']==plan['baseline_source_sha256']['secondary_curvature.cu'],'Original MG shader differs')
    static=read_json(diagnostics/'filter_controls.json');effective=mg.check_controls(plan,static)
    control=read_json(diagnostics/'outlet_control.json')
    require(control.get('requested_input_outlet_extension')==0 and control.get('effective_reconstruction_outlet_extension')==1 and
        control.get('applied_after_unchanged_original_validation') is True and control.get('before_graph_capture') is True and
        control.get('original_multilevel_method_unchanged') is True and control.get('original_multilevel_tolerance_unchanged') is False and
        control.get('requested_filter_controls')==candidate.REQUESTED and control.get('effective_filter_controls')==candidate.EFFECTIVE and
        control.get('reconstruction_cuda_sha256')==plan['baseline_source_sha256']['reconstruction.cu'],'Executed combined controls differ')
    graph=read_json(diagnostics/'graph_outlet_control.json')
    require(graph.get('actual_captured_outlet_extension')==1 and graph.get('guard_scope_expanded') is True and
        graph.get('original_other_graph_constraints_enforced') is True and graph.get('original_control_fingerprint_guard_retained') is True and
        graph.get('captures')==1 and graph.get('original_graph_source_sha256')==plan['baseline_source_sha256']['coupled_graph.py'] and
        graph.get('original_graph_scope_source_sha256')==plan['baseline_source_sha256']['graph_execution.py'],'Actual graph scope/control differs')
    worker=read_json(output/'run.json')
    require(worker['source_sha256']==manifest and worker['status']==audit['status'] and
            worker['transport_graph']['outlet_research_scope']==graph,'Worker identity/status/graph differs')
    for key,plan_key in (('end_s','until_s'),('dtmax_s','dt_max_s'),('output_every_s','output_every_s'),('cfl','cfl'),('morfac','morfac'),('launch_mode','launch_mode')):
        require(worker[key]==plan[plan_key],'Worker control differs: '+key)
    trace=original.summarize_streams(diagnostics/'step_trials.jsonl',diagnostics/'filter_calls.jsonl',start=0.,dtmax=.0125,launch_mode='graph')
    mg.accepted_cfl_gate(trace,.8)
    for key,wkey in (('accepted','steps'),('rejected','rejected')):
        require(trace[key]==audit[key]==worker[wkey]==audit['worker_'+wkey],'Worker/observer counts differ')
    require(trace['trials']==audit['trials']==trace['accepted']+trace['rejected'],'Trial counts differ')
    final=original.finite(worker['final_time_s'],'final time',nonnegative=True)
    require(trace['final_clock_s']==final==audit['final_time_s'],'Accepted-trial final clock differs')
    require(abs(trace['clock_minus_compensated_dt_sum_s'])<=1e-8*max(1.,final/2880.),'Accepted dt sum differs')
    for phase,stats in trace['filters'].items():
        for key in ('calls','total_iterations','maximum_iterations','maximum_true_residual','maximum_recurrence_residual','wall_s','all_passed'):
            require(stats[key]==audit['filter_statistics'][phase][key],'Filter aggregate differs')
    stop=read_json(diagnostics/'stop_control.json')
    require(stop.get('path')==str(candidate.outlet.stop_path(output,plan['stop_file'])) and stop.get('initially_absent') is True and
        stop.get('passed_to_original_worker') is True and stop.get('status')==audit['status'] and stop.get('final_time_s')==final,'Stop delegation differs')
    checkpoint=output/'checkpoint.npz'
    with np.load(checkpoint,allow_pickle=False) as data:
        for key in ('q','layers','secondary','bed_elevation','ledger_water','ledger_bed','ledger_secondary','ledger_source'):
            require(np.isfinite(data[key]).all(),'Nonfinite checkpoint: '+key)
        require(float(data['time'])==final and int(data['steps'])==trace['accepted'] and int(data['rejected'])==trace['rejected'],'Checkpoint counts differ')
        require(data['source_identity'].item()==audit['frozen_manifest_sha256'] and data['input_identity'].item()==worker['input_preparation_sha256'],'Checkpoint identity differs')
    complete=mg.complete_20h(worker['status'],plan['until_s'],final)
    if require_complete_20h:require(complete,'Complete20h required; short smoke is not long qualification')
    require(audit['worker_total_wall_s']==worker['total_wall_s'],'Timing differs')
    for key in ('audit_total_wall_s','worker_total_wall_s'):original.finite(audit[key],key,nonnegative=True)
    return dict(schema=1,wrapper_schema=7,verified=True,output=str(output),status=worker['status'],
        complete_20h=complete,complete_20h_trace=complete,completion_time_tolerance_s=1e-7,
        physical_accuracy_qualified=False,accepted_CFL_gate_passed=True,selection=plan['selection'],
        requested_filter_controls=candidate.REQUESTED,effective_filter_controls=effective,
        CPU_reference_retuned=False,original_numeric_sources_unchanged=True,hydraulic_numerical_operator_changed=True,
        filter_tolerance_changed=True,outlet_control=control,graph_outlet_control=graph,stop_control=stop,
        active_filter_cuda_sha256=manifest['secondary_curvature.cu'],audit_total_wall_s=audit['audit_total_wall_s'],worker_total_wall_s=worker['total_wall_s'],
        audit_sha256=original.digest(diagnostics/'audit.json'),run_sha256=original.digest(output/'run.json'),
        checkpoint_sha256=original.digest(checkpoint),plan_sha256=audit['plan_sha256'],frozen_manifest_sha256=audit['frozen_manifest_sha256'],
        static_filter_controls=static,**trace)
