"""Schema6 read-only outlet graph trace verifier, original sources unmodified."""
from pathlib import Path
from . import adaptive_trace_analysis as original
from . import outlet_bounds_gpu_candidate_v2 as candidate
from .outlet_bounds_gpu_trace import accepted_cfl_gate, complete_20h


def analyze(output_path, *, require_complete_20h=False):
    result=original.analyze(output_path,require_complete_20h=False)
    output=Path(output_path).resolve();diagnostics=output.with_name(output.name+'_audit')
    audit=original.read_json(diagnostics/'audit.json');plan=original.read_json(audit['plan'])
    original.require(plan.get('wrapper_schema')==6 and plan.get('wrapper_entrypoint')=='outlet_bounds_gpu_candidate_v2.py' and
                     plan['selection']==candidate.v1.selection() and plan['launch_mode']=='graph', 'Schema6 explicit selection differs')
    original.require(plan['requested_filter_controls']==candidate.v1.CONTROLS==plan['effective_filter_controls'], 'Original filter controls changed')
    original.require(plan.get('hydraulic_numerical_operator_changed') is True and
                     plan.get('original_numeric_source_files_unchanged') is True and
                     plan.get('requested_input_outlet_extension')==0 and plan.get('effective_reconstruction_outlet_extension')==1,
                     'Declared outlet operator change differs')
    original.require({'outlet_control.json','stop_control.json','graph_outlet_control.json'}<=audit['artifacts'].keys(),'Missing executed outlet/graph/stop controls')
    control=original.read_json(diagnostics/'outlet_control.json')
    original.require(control['requested_input_outlet_extension']==0 and control['effective_reconstruction_outlet_extension']==1 and
                     control.get('applied_after_unchanged_original_validation') is True and control.get('before_graph_capture') is True and
                     control['reconstruction_cuda_sha256']==plan['baseline_source_sha256']['reconstruction.cu'],'Executed outlet selection differs')
    graph=original.read_json(diagnostics/'graph_outlet_control.json')
    original.require(graph['actual_captured_outlet_extension']==1 and graph.get('guard_scope_expanded') is True and
                     graph.get('original_other_graph_constraints_enforced') is True and
                     graph.get('original_control_fingerprint_guard_retained') is True and graph.get('captures')==1 and
                     graph['original_graph_source_sha256']==plan['baseline_source_sha256']['coupled_graph.py'] and
                     graph['original_graph_scope_source_sha256']==plan['baseline_source_sha256']['graph_execution.py'], 'Actual captured extension/control guard differs')
    worker=original.read_json(output/'run.json')
    original.require(worker['transport_graph']['outlet_research_scope']==graph,'Worker graph evidence differs')
    stop=original.read_json(diagnostics/'stop_control.json')
    original.require(stop['path']==str(candidate.v1.stop_path(output,plan['stop_file'])) and stop.get('initially_absent') is True and
                     stop.get('passed_to_original_worker') is True and stop['status']==audit['status'] and
                     stop['final_time_s']==result['final_clock_s'],'Executed stop control differs')
    accepted_cfl_gate(result)
    complete=complete_20h(result['status'],plan['until_s'],result['final_clock_s'])
    if require_complete_20h: original.require(complete,'Complete finite20h required; short smoke is not qualification')
    result.update(wrapper_schema=6,complete_20h=complete,complete_20h_trace=complete,accepted_CFL_gate_passed=True,
                  outlet_control=control,graph_outlet_control=graph,stop_control=stop,
                  hydraulic_numerical_operator_changed=True,physical_accuracy_qualified=False)
    return result
