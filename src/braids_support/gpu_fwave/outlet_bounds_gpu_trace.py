"""Schema5 read-only trace verifier for the explicit hydraulic outlet experiment."""
from pathlib import Path
import numpy as np
from . import adaptive_trace_analysis as original
from .outlet_bounds_gpu_candidate import CONTROLS, selection, stop_path


def accepted_cfl_gate(trace):
    maximum = trace['accepted_cfl']['maximum']
    original.require(maximum is not None and np.isfinite(maximum) and 0 <= maximum <= .8,
                     'Accepted CFL exceeds0.8 or is not finite')


def complete_20h(status, until, final):
    return bool(status == 'completed' and until == 2880. and np.isfinite(final) and abs(final - 2880.) <= 1e-7)


def analyze(output_path, *, require_complete_20h=False):
    result = original.analyze(output_path, require_complete_20h=False)
    output = Path(output_path).resolve(); diagnostics = output.with_name(output.name + '_audit')
    audit = original.read_json(diagnostics / 'audit.json'); plan = original.read_json(audit['plan'])
    original.require(plan.get('wrapper_schema') == 5 and plan.get('wrapper_entrypoint') == 'outlet_bounds_gpu_candidate.py' and
                     plan['selection'] == selection(), 'Schema5 explicit selection differs')
    original.require(plan['requested_filter_controls'] == CONTROLS == plan['effective_filter_controls'], 'Original filter controls changed')
    original.require(plan.get('hydraulic_numerical_operator_changed') is True and
                     plan.get('original_numeric_source_files_unchanged') is True and
                     plan.get('requested_input_outlet_extension') == 0 and plan.get('effective_reconstruction_outlet_extension') == 1,
                     'Declared outlet operator change differs')
    original.require({'outlet_control.json', 'stop_control.json'} <= audit['artifacts'].keys(), 'Missing executed outlet/stop controls')
    control = original.read_json(diagnostics / 'outlet_control.json')
    original.require(control['requested_input_outlet_extension'] == 0 and control['effective_reconstruction_outlet_extension'] == 1 and
                     control.get('applied_after_unchanged_original_validation') is True and control.get('before_graph_capture') is True and
                     control['reconstruction_cuda_sha256'] == plan['baseline_source_sha256']['reconstruction.cu'], 'Executed outlet selection differs')
    stop = original.read_json(diagnostics / 'stop_control.json')
    original.require(stop['path'] == str(stop_path(output, plan['stop_file'])) and stop.get('initially_absent') is True and
                     stop.get('passed_to_original_worker') is True and stop['status'] == audit['status'] and
                     stop['final_time_s'] == result['final_clock_s'], 'Executed stop control differs')
    accepted_cfl_gate(result)
    complete = complete_20h(result['status'], plan['until_s'], result['final_clock_s'])
    if require_complete_20h:
        original.require(complete, 'Complete finite20h required; short smoke is not qualification')
    result.update(wrapper_schema=5, complete_20h=complete, complete_20h_trace=complete,
                  accepted_CFL_gate_passed=True, outlet_control=control, stop_control=stop,
                  hydraulic_numerical_operator_changed=True, physical_accuracy_qualified=False)
    return result
