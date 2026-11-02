"""Schema7 research: outlet extension1 plus explicitly tighter GPU multilevel CG.

No existing source file or CPU reference is changed. A reviewed plan is needed
for execution; preparing a24s smoke plan does not authorize any GPU launch.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import json
from . import adaptive_filter_candidate as base
from . import adaptive_multilevel_candidate as mg
from . import outlet_bounds_gpu_candidate as outlet
from . import outlet_bounds_gpu_candidate_v2 as graph_outlet

HERE=Path(__file__).resolve().parent
REQUESTED=dict(rtol=1e-4,atol=1e-10,maxiter=500,graph_batch=5)
EFFECTIVE=dict(REQUESTED,rtol=1e-8)
HELPERS=tuple(dict.fromkeys(mg.HELPERS+outlet.HELPERS+graph_outlet.HELPERS+
    ('outlet_bounds_tight_gpu_candidate.py','outlet_bounds_tight_gpu_trace.py')))


def selection():
    chosen=mg.selection(1e-8)
    chosen.update(outlet_extension=1,
        numerical_change='Existing hydraulic virtual-exterior outlet bounds enabled for SSPRK43 AND GPU multilevel rtol1e-8')
    return chosen


def outlet_control(solver):
    return dict(requested_input_outlet_extension=0,effective_reconstruction_outlet_extension=1,
        applied_after_unchanged_original_validation=True,before_graph_capture=True,
        hydraulic_numerical_operator_changed=True,augmented_fwave_unchanged=True,sediment_reconstruction_unchanged=True,
        original_multilevel_method_unchanged=True,original_multilevel_tolerance_unchanged=False,
        requested_filter_controls=REQUESTED,effective_filter_controls=EFFECTIVE,
        reconstruction_cuda_sha256=base.sha(solver.folder/'reconstruction.cu'),solver_cuda_files_unchanged=True)


@contextmanager
def adapters(plan=None):
    from . import coupled_solver
    saved=(base.selection,base.ObservedFilter,base.Traces,coupled_solver.CoupledSolver,base.worker.run_gpu)
    def choose(name,arithmetic='native_order',block_size=128):
        if name!='multilevel':raise ValueError('Schema7 requires explicit multilevel GPU')
        return selection()
    base.selection=choose
    base.ObservedFilter=lambda original,trace:mg.ToleranceObservedFilter(original,trace,1e-8)
    base.Traces=graph_outlet.GraphTraces
    if plan is not None:
        class CombinedSolver(saved[3]):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);outlet.enable_extension(self)
                control=outlet_control(self)
                if control['reconstruction_cuda_sha256']!=plan['baseline_source_sha256']['reconstruction.cu']:
                    raise ValueError('Original reconstruction shader differs')
                target=Path(plan['output_path']);audit=target.with_name(target.name+'_audit')
                base.write_json(audit/'outlet_control.json',control)
            def step(self,*args,**kwargs):
                if self.reconstruction.outlet_extension!=1:raise ValueError('Outlet extension changed during execution')
                return super().step(*args,**kwargs)
        def run_gpu(**kwargs):
            signal=outlet.stop_path(plan['output_path'],plan['stop_file'])
            if signal.exists() or 'stop_file' in kwargs:raise ValueError('Stop signal must initially be absent')
            target=Path(plan['output_path']);path=target.with_name(target.name+'_audit')/'stop_control.json'
            evidence=dict(path=str(signal),initially_absent=True,passed_to_original_worker=True,status='running')
            base.write_json(path,evidence)
            try:
                result=saved[4](**kwargs,stop_file=str(signal))
                evidence.update(status=result['status'],final_time_s=result['final_time_s']);return result
            except Exception as error:
                evidence.update(status='failed',error=str(error));raise
            finally:
                evidence['signal_present_at_return']=signal.exists();base.write_json(path,evidence)
        coupled_solver.CoupledSolver=CombinedSolver;base.worker.run_gpu=run_gpu
    try:yield
    finally:
        base.selection,base.ObservedFilter,base.Traces,coupled_solver.CoupledSolver,base.worker.run_gpu=saved


def validate_plan(plan):
    if (plan.get('wrapper_schema')!=7 or plan.get('wrapper_entrypoint')!=Path(__file__).name or
        plan['selection']!=selection() or plan['requested_filter_controls']!=REQUESTED or
        plan['effective_filter_controls']!=EFFECTIVE or plan['launch_mode']!='graph' or
        plan.get('requested_input_outlet_extension')!=0 or plan.get('effective_reconstruction_outlet_extension')!=1 or
        plan.get('actual_captured_outlet_extension_required')!=1 or plan.get('hydraulic_numerical_operator_changed') is not True or
        plan.get('filter_tolerance_changed') is not True or plan.get('CPU_reference_retuned') is not False):
        raise ValueError('Explicit combined outlet1/tight-MG schema7 controls differ')
    return True


def prepare(plan_path,output_path,*,until=24.,max_steps=None,stop_file=None):
    signal=outlet.stop_path(output_path,stop_file)
    if signal.exists():raise FileExistsError('Stop signal already exists')
    path=Path(plan_path).resolve()
    with adapters():
        result=base.prepare(path,output_path,filter_name='multilevel',until=until,max_steps=max_steps,launch_mode='graph')
    plan=json.loads(path.read_text())
    for name in ('execution.json','analysis_geometry.npz','input_provenance.json'):
        source=base.CPU_REFERENCE/name;plan['protected_files'][str(source)]=base.sha(source)
    for name in HELPERS:
        digest=base.sha(HERE/name);plan['candidate_source_sha256'][name]=digest
        plan['authenticated_files'][str(HERE/name)]=digest
    plan.update(wrapper_schema=7,wrapper_entrypoint=Path(__file__).name,base_wrapper_archive='audit_driver.py',
        requested_filter_controls=REQUESTED,effective_filter_controls=EFFECTIVE,
        requested_input_outlet_extension=0,effective_reconstruction_outlet_extension=1,
        actual_captured_outlet_extension_required=1,hydraulic_numerical_operator_changed=True,
        filter_tolerance_changed=True,original_numeric_sources_unchanged=True,CPU_reference_retuned=False,
        augmented_fwave_unchanged=True,sediment_reconstruction_unchanged=True,active_filter_cuda='secondary_curvature.cu',
        accepted_CFL_maximum=.8,stop_file=str(signal),stop_file_initially_absent=True,
        graph_guard_scope_expansion='Only outlet_extension0->1; all remaining graph guards retained',
        interpretation='Two explicit research controls; tighter independent GPU solve is NOT the loose CPU finite-iteration path',
        qualification='Prepared contingency only; short smoke is not full long-run accuracy or speed qualification')
    validate_plan(plan);base.write_json(path,plan)
    return dict(result,sha256=base.sha(path),wrapper_schema=7,selection=selection(),gpu_executed=False)


def run(plan_path,approved_sha256):
    path=Path(plan_path).resolve()
    if base.sha(path)!=approved_sha256:raise ValueError('Reviewed schema7 plan SHA required')
    plan=json.loads(path.read_text());validate_plan(plan)
    for name in HELPERS:
        digest=base.sha(HERE/name)
        if plan['candidate_source_sha256'].get(name)!=digest or plan['authenticated_files'].get(str(HERE/name))!=digest:
            raise ValueError('Unpinned combined candidate dependency: '+name)
    if outlet.stop_path(plan['output_path'],plan['stop_file']).exists():raise ValueError('Stop signal must be absent')
    with adapters(plan),graph_outlet.graph_adapters(plan):
        return base.run(path,approved_sha256)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--until',type=float,default=24.);p.add_argument('--max-steps',type=int);p.add_argument('--stop-file',type=Path)
    p=sub.add_parser('run');p.add_argument('--plan',type=Path,required=True);p.add_argument('--approved-plan-sha256',required=True)
    args=parser.parse_args()
    result=(prepare(args.plan,args.output,until=args.until,max_steps=args.max_steps,stop_file=args.stop_file)
        if args.command=='prepare' else run(args.plan,args.approved_plan_sha256))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
