"""Schema5 isolated SSPRK43 outlet-bound extension using original GPU kernels.

Only the explicit reconstruction outlet-extension control changes0->1 after
normal original input braids_support. The augmented f-wave solver, sediment
reconstruction and original multilevel filter/tolerances remain unchanged.
This is a hydraulic numerical-operator experiment, not an I/O-only change.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json

from . import adaptive_filter_candidate as base

HERE = Path(__file__).resolve().parent
V1_SELECTION = base.selection
CONTROLS = dict(rtol=1e-4, atol=1e-10, maxiter=500, graph_batch=5)
HELPERS = ('outlet_bounds_gpu_candidate.py', 'outlet_bounds_gpu_trace.py', 'adaptive_trace_analysis.py')


def selection():
    chosen = V1_SELECTION('multilevel')
    chosen.update(outlet_extension=1, numerical_change='Existing hydraulic virtual-exterior outlet bounds enabled for SSPRK43')
    return chosen


def enable_extension(solver):
    """Called only after the unchanged original constructor/strict guard."""
    if solver.reconstruction.outlet_extension != 0 or any(solver.data.get(key, 0) != 0 for key in ('outlet_extension', 'cpu_outlet_extension')):
        raise ValueError('Expected unchanged zero-extension prepared input')
    solver.reconstruction.outlet_extension = 1


def stop_path(output, requested=None):
    output = Path(output).resolve()
    target = Path(requested).resolve() if requested is not None else output.with_name(output.name + '.stop')
    if target.parent != output.parent or target in (output, output.with_name(output.name + '_audit')):
        raise ValueError('Stop signal must be a separate file beside output')
    return target


class ExtensionTraces(base.Traces):
    def close(self):
        result = super().close()
        for name in ('outlet_control.json', 'stop_control.json'):
            path = self.destination / name
            if path.exists():
                result['artifacts'][name] = base.sha(path)
        return result


@contextmanager
def adapters(plan=None):
    from . import coupled_solver
    saved = base.selection, base.Traces, coupled_solver.CoupledSolver, base.worker.run_gpu
    def choose(name, arithmetic='native_order', block_size=128):
        if name != 'multilevel':
            raise ValueError('Outlet experiment retains original multilevel filter')
        return selection()
    class ExtensionSolver(saved[2]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            enable_extension(self)
            folder = Path(plan['output_path']).with_name(Path(plan['output_path']).name + '_audit')
            control = dict(requested_input_outlet_extension=0, effective_reconstruction_outlet_extension=1,
                           applied_after_unchanged_original_validation=True, before_graph_capture=True,
                           hydraulic_numerical_operator_changed=True, augmented_fwave_unchanged=True,
                           sediment_reconstruction_unchanged=True, original_multilevel_filter_unchanged=True,
                           reconstruction_cuda_sha256=base.sha(self.folder / 'reconstruction.cu'),
                           solver_cuda_files_unchanged=True)
            if control['reconstruction_cuda_sha256'] != plan['baseline_source_sha256']['reconstruction.cu']:
                raise ValueError('Original reconstruction kernel differs')
            base.write_json(folder / 'outlet_control.json', control)
        def step(self, *args, **kwargs):
            if self.reconstruction.outlet_extension != 1:
                raise ValueError('Effective outlet extension changed during run')
            return super().step(*args, **kwargs)
    def run_gpu(**kwargs):
        signal = stop_path(plan['output_path'], plan['stop_file'])
        if signal.exists() or 'stop_file' in kwargs:
            raise ValueError('Stop signal must initially be absent')
        path = Path(plan['output_path']).with_name(Path(plan['output_path']).name + '_audit') / 'stop_control.json'
        evidence = dict(path=str(signal), initially_absent=True, passed_to_original_worker=True, status='running')
        base.write_json(path, evidence)
        try:
            result = saved[3](**kwargs, stop_file=str(signal))
            evidence.update(status=result['status'], final_time_s=result['final_time_s'])
            return result
        except Exception as exc:
            evidence.update(status='failed', error=str(exc))
            raise
        finally:
            evidence['signal_present_at_return'] = signal.exists()
            base.write_json(path, evidence)
    base.selection, base.Traces = choose, ExtensionTraces
    if plan is not None:
        coupled_solver.CoupledSolver, base.worker.run_gpu = ExtensionSolver, run_gpu
    try:
        yield
    finally:
        base.selection, base.Traces, coupled_solver.CoupledSolver, base.worker.run_gpu = saved


def prepare(plan_path, output_path, *, until=24., max_steps=None, launch_mode='graph', stop_file=None):
    signal = stop_path(output_path, stop_file)
    if signal.exists():
        raise FileExistsError('Stop signal already exists')
    plan_path = Path(plan_path).resolve()
    with adapters():
        result = base.prepare(plan_path, output_path, filter_name='multilevel', until=until,
                              max_steps=max_steps, launch_mode=launch_mode)
    plan = json.loads(plan_path.read_text())
    for name in ('execution.json', 'analysis_geometry.npz', 'input_provenance.json'):
        path = base.CPU_REFERENCE / name
        plan['protected_files'][str(path)] = base.sha(path)
    for name in HELPERS:
        identity = base.sha(HERE / name)
        plan['candidate_source_sha256'][name] = identity
        plan['authenticated_files'][str(HERE / name)] = identity
    plan.update(wrapper_schema=5, wrapper_entrypoint=Path(__file__).name,
                requested_filter_controls=CONTROLS, effective_filter_controls=CONTROLS,
                requested_input_outlet_extension=0, effective_reconstruction_outlet_extension=1,
                hydraulic_numerical_operator_changed=True, original_numeric_source_files_unchanged=True,
                augmented_fwave_unchanged=True, sediment_reconstruction_unchanged=True,
                filter_unchanged=True, CPU_reference_retuned=False,
                accepted_CFL_maximum=.8, stop_file=str(signal), stop_file_initially_absent=True,
                candidate_numerical_scope='Enable existing virtual-exterior hydraulic outlet extrema before slope limiting for SSPRK43; no new kernel formula')
    base.write_json(plan_path, plan)
    return dict(result, sha256=base.sha(plan_path), wrapper_schema=5, selection=selection())


def run(plan_path, approved_sha256):
    plan_path = Path(plan_path).resolve()
    if base.sha(plan_path) != approved_sha256:
        raise ValueError('Reviewed schema5 plan SHA required')
    plan = json.loads(plan_path.read_text())
    if (plan.get('wrapper_schema') != 5 or plan.get('wrapper_entrypoint') != Path(__file__).name or
            plan['selection'] != selection() or plan['requested_filter_controls'] != CONTROLS or
            plan['effective_filter_controls'] != CONTROLS or plan.get('effective_reconstruction_outlet_extension') != 1 or
            plan.get('requested_input_outlet_extension') != 0 or not plan.get('hydraulic_numerical_operator_changed')):
        raise ValueError('Explicit outlet-only candidate controls differ')
    if stop_path(plan['output_path'], plan['stop_file']).exists():
        raise ValueError('Plan-pinned stop signal must be absent')
    for name in HELPERS:
        identity = base.sha(HERE / name)
        if plan['candidate_source_sha256'].get(name) != identity or plan['authenticated_files'].get(str(HERE / name)) != identity:
            raise ValueError('Unpinned outlet dependency: ' + name)
    with adapters(plan):
        return base.run(plan_path, approved_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--until', type=float, default=24.); p.add_argument('--max-steps', type=int)
    p.add_argument('--launch-mode', choices=('graph', 'plain'), default='graph'); p.add_argument('--stop-file', type=Path)
    p = subs.add_parser('run'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--approved-plan-sha256', required=True)
    args = parser.parse_args()
    result = (prepare(args.plan, args.output, until=args.until, max_steps=args.max_steps, launch_mode=args.launch_mode, stop_file=args.stop_file)
              if args.command == 'prepare' else run(args.plan, args.approved_plan_sha256))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
