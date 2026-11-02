"""Prepare-only tight-MG late checkpoint adapter; never launch/build.

Only the completed correctedCPU20h case is accepted. Its exact conserved hI,
physical boundary ledgers, source integral, clock and timestep hint are retained.
The exported GPU snapshot is new and explicitly identifies checkpoint ancestry.
GPU construction/resume hooks are provided for a separately reviewed execution
envelope; this module has no simulation-launch CLI.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import numpy as np
from . import cpu_cold_ulp24 as base
from . import outlet_bounds_cpu20h as cpu
from . import outlet_bounds_gpu_candidate_v2 as graph_scope
from . import coupled_reference as cr
from . import outlet_bounds_late_checkpoint as original
from . import outlet_bounds_tight_gpu_candidate as tight

ORIGINAL_ADAPTER_SHA='44179bf52b4b696e48856a152cace49fd9975a0bfa42d34249b64a2f51d43215'
NATIVE_ENV=dict(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1',BRAIDS_TIME_INTEGRATOR='ssp43')


def candidate_context():
    return dict(wrapper_schema=7,wrapper_entrypoint='outlet_bounds_tight_gpu_candidate.py',
                outlet_extension=1,filter='multilevel',selection=tight.selection(),
                requested_filter_controls=dict(tight.REQUESTED),effective_filter_controls=dict(tight.EFFECTIVE),
                filter_tolerance_changed=True,CPU_reference_retuned=False,
                qualification='Unqualified late-checkpoint adapter only; no simulation or full20h speed qualification')



decode_checkpoint=original.decode_checkpoint
write_records=original.write_records
checkpoint_arrays=original.checkpoint_arrays


def window_controls(start,end=2904.):
    if not np.isfinite([start,end]).all() or abs(start-2880.)>1e-7 or end!=2904.:
        raise ValueError('Require exact nominal2880 restart and absolute2904 endpoint; only restart-clock roundoff allowed')
    return dict(start_s=float(start),end_s=end,duration_s=end-start,GPU_output_every_s=end-start,
                CPU_output_every_s=24.,nominal_duration_s=24.)


def validate_controls(plan):
    control=window_controls(plan['initial_time_s'],plan['end_time_s'])
    if (plan['GPU_output_every_s']!=control['GPU_output_every_s'] or
        plan['duration_s']!=control['duration_s'] or plan['nominal_duration_s']!=24. or
        plan['native_environment']!=NATIVE_ENV or plan['CPU_threads']!=8 or
        plan['controls']!=dict(cfl=.8,dtmax=.0125,morfac=25.,output=24.,integrator=43,outlet_extension=1) or
        plan.get('GPU_candidate_context')!=candidate_context()):
        raise ValueError('Late source, interval, native8thread or numerical controls differ')
    return control


def check_output_times(cpu_times,gpu_times,plan):
    validate_controls(plan)
    left,right=np.asarray(cpu_times),np.asarray(gpu_times)
    for value in (left,right):
        if (value.shape!=(2,) or not np.isfinite(value).all() or
            value[0]!=plan['initial_time_s'] or abs(value[-1]-2904.)>1e-12):
            raise ValueError('Exactly two actual outputs at raw checkpoint time and2904 are required')
    if np.max(abs(left-right))>1e-12:raise ValueError('CPU/GPU actual output clocks differ')
    return True


def worker_arguments(plan,resume,*,stop_file=None):
    """Only returns arguments: no launch. Deliberately omit global max_steps."""
    validate_controls(plan)
    signal=graph_scope.v1.stop_path(plan['GPU_output'],stop_file)
    if signal.exists():raise ValueError('Stop signal must be absent; never remove an old signal')
    return dict(input_path=plan['prepared_input'],output_path=plan['GPU_output'],resume=str(resume),
        until=2904.,output_every=plan['GPU_output_every_s'],dt_max=.0125,launch_mode='graph',stop_file=str(signal))


def checkpoint_snapshot(data,checkpoint,end):
    window_controls(checkpoint['time'],end)
    result=original.checkpoint_snapshot(data,checkpoint,checkpoint['time']+24.)
    result['endtime']=float(end)
    return result


def prepare(completed_case,tag):
    case=Path(completed_case).resolve()
    if Path(tag).name!=tag or not tag.startswith('outlet_bounds_tight_late24_'):
        raise ValueError('New task-specific tag required')
    if base.sha(original.__file__)!=ORIGINAL_ADAPTER_SHA:raise ValueError('Original checkpoint semantics changed')
    execution=json.loads((case/'execution.json').read_text())
    if execution.get('status')!='completed' or execution.get('return_code')!=0 or execution.get('executable_sha256')!=cpu.EXE_SHA:
        raise ValueError('Corrected CPU must finish successfully before checkpoint can be frozen')
    identity,protected=cpu.verify_runtime()
    for name in ('case.data','case.data.bin','case.data.json','checkpoint.bin','execution.json','native.log'):
        protected[str(case/name)]=base.sha(case/name)
    source=graph_scope.base.LONG/'prepared'
    expected_path=base.HERE/'runs/outlet_bounds_cpu24_20261002_v1/control/checkpoint.bin'
    for path in (expected_path,source/'coupled_input.bin',source/'coupled_diffusion.bin',source/'preparation_receipt.json'):
        protected[str(path)]=base.sha(path)
    checkpoint=decode_checkpoint((case/'checkpoint.bin').read_bytes())
    if abs(checkpoint['time']-2880.)>1.e-7:raise ValueError('Require completed20morphological-hour state')
    expected=decode_checkpoint(expected_path.read_bytes())
    if checkpoint['configuration_prefix']!=expected['configuration_prefix']:
        raise ValueError('Original resolved configuration/fingerprint changed')
    original_data=cr.read_records(source/'coupled_input.bin')
    data=checkpoint_snapshot(original_data,checkpoint,2904.)
    run=base.HERE/'runs'/tag;report=base.HERE/'reports'/tag
    if run.exists() or report.exists():raise FileExistsError('Fresh late adapter paths required')
    run.mkdir(parents=True);report.mkdir(parents=True);prepared=run/'prepared';prepared.mkdir()
    shutil.copy2(__file__,report/Path(__file__).name)
    shutil.copy2(case/'checkpoint.bin',run/'native_checkpoint.bin')
    write_records(prepared/'coupled_input.bin',data)
    shutil.copy2(source/'coupled_diffusion.bin',prepared/'coupled_diffusion.bin')
    roundtrip=cr.read_records(prepared/'coupled_input.bin')
    changed=('state','ledger','layer_mass','bed','time','endtime')
    for key,value in original_data.items():
        if key not in changed and not np.array_equal(value,roundtrip[key]):raise ValueError('Changed static record: '+key)
    for key,value in data.items():
        if not np.array_equal(value,roundtrip[key]):raise ValueError('Prepared record roundtrip differs: '+key)
    receipt=dict(schema=1,artifacts={name:base.sha(prepared/name) for name in ('coupled_input.bin','coupled_diffusion.bin')},
                 ancestry='Exact completed correctedCPU checkpoint; not a cold initial state and not printed-I reconstruction',
                 checkpoint_sha256=protected[str(case/'checkpoint.bin')],
                 original_preparation_receipt_sha256=base.sha(source/'preparation_receipt.json'),
                 copied_static_diffusion_sha256=base.sha(source/'coupled_diffusion.bin'),
                 changed_records=list(changed),solver_steps_performed=0,initialization_CPU_only=True,
                 production_modified=False,delft_modified=False)
    base.save_json(prepared/'preparation_receipt.json',receipt)
    initial=checkpoint_arrays(checkpoint,data['bed_indices'])
    np.savez(run/'exact_initial.npz',**initial,time=checkpoint['time'],dt_hint=checkpoint['dt_hint'],steps=checkpoint['steps'],
             native_ancillary=checkpoint['ancillary'])
    cpu_case=run/'cpu';origin=cr.copy_inputs(case/'case.data',cpu_case)
    text=(cpu_case/'case.data').read_text();old='2880 0.012500000000000001 24';new=f'{data["endtime"]:.17g} 0.012500000000000001 24'
    if text.splitlines().count(old)!=1:raise ValueError('Ambiguous native endpoint')
    (cpu_case/'case.data').write_bytes(text.replace(old+'\n',new+'\n').encode())
    plan=dict(schema=2,tag=tag,prepared_only=True,simulation_launched=False,
              adapter_plan_path=str(report/'plan.json'),corrected_CPU_case=str(case),checkpoint_sha256=base.sha(run/'native_checkpoint.bin'),
              exact_initial_sha256=base.sha(run/'exact_initial.npz'),
              prepared_input=str(prepared),preparation_sha256=base.sha(prepared/'preparation_receipt.json'),
              CPU_output=str(cpu_case),GPU_output=str(run/'gpu'),
              native_command=[str(cpu.RUNTIME/'xbraids_candidate.exe'),'case.data',str(run/'native_checkpoint.bin')],
              CPU_input_sha256={str(path):base.sha(path) for path in cpu_case.glob('case.data*')},
              CPU_runtime=identity,protected_files=protected,
              checkpoint_configuration_bitwise_equal=True,conserved_hI_copied_exactly=True,
              all_physical_ledgers_preserved=True,source_integral_preserved=True,
              discarded_native_diagnostics='None onCPU; GPU does not implement the ancillary diagnostic momentum-ledger reports, which are archived in exact_initial.npz',
              initial_time_s=checkpoint['time'],end_time_s=data['endtime'],duration_s=data['endtime']-checkpoint['time'],
              nominal_duration_s=24.,GPU_output_every_s=data['endtime']-checkpoint['time'],
              native_environment=NATIVE_ENV,CPU_threads=8,
              endpoint_alignment='Native absolute24s output grid; GPU output interval=end2904-rawcheckpointtime. No state clock rounding.',
              trace_counter_contract='Original scalar Traces counts are local; worker steps remain native global. Compare workersteps-initialsteps. Rejections reset locally because nativecheckpoint has no priorrejectcount.',
              graph_warmup_clock_contract='Original noncommit warmups use time0 with the exact late state. Constant forcing only; actual accepted evolution retains native absolute time.',
              initial_dt_hint_s=checkpoint['dt_hint'],initial_global_steps=checkpoint['steps'],
              controls=dict(cfl=.8,dtmax=.0125,morfac=25.,output=24.,integrator=43,outlet_extension=1),
              forcing_scope='Unchanged constant forcing only; nhydro_series=0 andnmorfac=0 required; absolute time retained',
              GPU_cache_contract='Fresh geometry/solver caches, originalgraph; inject exact late state/ledgers BEFORE graph warmup; verifybitwiseafter; originalworkerresume then restoresdthint/globalsteps',
              required_before_timing=['Root planSHA review andexclusive window',
                  'Samecheckpoint plain/graph one-step accuracy andnoncommitstate/ledgerbitwise checks',
                  'Initialoutput equals exact checkpoint arrays/ledgers onbothbackends',
                  'Incrementalwater/eachlabel/hI budgets subtract carried initialledgers',
                  'Record trialdt/CFL/rejections; do not reinterpretCPUglobalsteps aswindowsteps',
                  'Disclose this is24s late continuation, not measured20h speed'],
              helper_sha256=base.sha(__file__),GPU_candidate_context=candidate_context(),
              GPU_source_sha256=graph_scope.base.verify_baseline(graph_scope.base.LONG/'gpu/source'),
              adapter_dependency_sha256={str(Path(module.__file__).resolve()):base.sha(module.__file__)
                  for module in (base,cpu,graph_scope,graph_scope.v1,cr,original,tight,tight.mg)},
              GPU_execution_adapter_qualified=False,CPU_restart_executed=False,GPU_calls=0,production_modified=False,delft_modified=False)
    for name in tight.HELPERS:
        plan['adapter_dependency_sha256'][str(base.HERE/name)]=base.sha(base.HERE/name)
    validate_controls(plan);cpu.verify_files(protected);base.save_json(report/'plan.json',plan)
    return dict(plan=str(report/'plan.json'),sha256=base.sha(report/'plan.json'),simulations_launched=False)


def validate_resume_plan(plan,approved_plan_sha256):
    """Host-only fail-closed guard, before any device constructor/import."""
    path=Path(plan['adapter_plan_path'])
    if base.sha(path)!=approved_plan_sha256 or json.loads(path.read_text())!=plan:
        raise ValueError('Exact reviewed late adapter plan required')
    validate_controls(plan)
    if plan.get('helper_sha256')!=base.sha(__file__):raise ValueError('Late adapter source changed')
    prepared=Path(plan['prepared_input']);root=prepared.parent
    for path,key in ((root/'exact_initial.npz','exact_initial_sha256'),
                     (root/'native_checkpoint.bin','checkpoint_sha256'),
                     (prepared/'preparation_receipt.json','preparation_sha256')):
        if base.sha(path)!=plan[key]:raise ValueError('Late adapter artifact changed: '+str(path))
    cpu.verify_files(plan['protected_files']);cpu.verify_files(plan['adapter_dependency_sha256'])
    cpu.verify_files(plan['CPU_input_sha256'])
    current=graph_scope.base.verify_baseline(graph_scope.base.LONG/'gpu/source')
    if current!=plan['GPU_source_sha256']:raise ValueError('Frozen GPU numerical sources differ')
    return True


@contextmanager
def gpu_resume_hooks(plan,*,approved_plan_sha256,traces):
    """Unexecuted implementation hook for the future reviewed GPU envelope.

    Caller must verify/freeze the plan and exact_initial/source dependencies,
    use worker_arguments(plan, returned_path), and retain/close all receipts.
    Traces are LOCAL counters; worker keeps native GLOBAL accepted-step count.
    This is NOT itself an execution/qualification API.
    """
    validate_resume_plan(plan,approved_plan_sha256)
    if type(traces) is not tight.base.Traces or (traces.accepted,traces.rejected,traces.trials)!=(0,0,0):
        raise ValueError('Require fresh original scalar Traces with local zero counters')
    from braids_gpu import runner
    from . import coupled_solver,coupled_graph
    prepared=Path(plan['prepared_input']);data,_=runner.load_prepared(prepared)
    root=prepared.parent;resume=root/'gpu_resume.npz'
    if resume.exists():raise FileExistsError(resume)
    with np.load(root/'exact_initial.npz',allow_pickle=False) as archive:
        initial={k:archive[k].copy() for k in ('q','layers','secondary','ledger_water','ledger_bed','ledger_secondary','ledger_source')}
        current=float(archive['time']);hint=float(archive['dt_hint']);steps=int(archive['steps'])
    if (current,hint,steps)!=(plan['initial_time_s'],plan['initial_dt_hint_s'],plan['initial_global_steps']):
        raise ValueError('Late restart scalar controls differ')
    saved_solver=coupled_solver.CoupledSolver;saved_freeze=runner.freeze_sources
    class LateSolver(saved_solver):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);graph_scope.v1.enable_extension(self)
            self.secondary.filter=tight.mg.ToleranceObservedFilter(self.secondary.filter,traces,1e-8)
            for key,array in self.ledger.items():array.set(initial['ledger_'+key])
            for group in ('state','ledger'):
                for key,array in getattr(self,group).items():
                    expected=initial[key if group=='state' else 'ledger_'+key]
                    if not np.array_equal(array.get().view(np.uint64),expected.view(np.uint64)):
                        raise ValueError('Checkpoint state/ledger did not reach GPU bitwise')
            runner.write_json(traces.destination/'late_control.json',dict(candidate_context(),
                initial_time_s=current,initial_dt_hint_s=hint,initial_global_steps=steps,
                physical_state_and_ledgers_copied_bitwise=True,physical_ledgers_reset=False))
        def step(self,dt,current,*,commit=True):
            if self.reconstruction.outlet_extension!=1:raise ValueError('Late outlet control changed')
            traces.begin_step(dt,current,commit);began=time.perf_counter()
            try:result=super().step(dt,current,commit=commit)
            except Exception as error:
                traces.failed_step(error,time.perf_counter()-began,commit);raise
            traces.end_step(result,time.perf_counter()-began,commit);return result
    def freeze(destination):
        manifest=saved_freeze(destination);destination=Path(destination)
        if manifest!=plan['GPU_source_sha256']:raise ValueError('Numerical source identity changed while freezing')
        extras={Path(key).name:Path(key) for key in plan['adapter_dependency_sha256']}
        extras[Path(__file__).name]=Path(__file__)
        extras['late_adapter_plan.json']=Path(plan['adapter_plan_path'])
        for name,path in extras.items():
            target=destination/name;digest=base.sha(path)
            if target.exists():
                if base.sha(target)!=digest:raise ValueError('Archive name collision: '+name)
            else:shutil.copy2(path,target)
            manifest[name]=digest
        runner.write_json(destination/'manifest.json',manifest)
        # Resume binds the FINAL manifest, including adapter and reviewed plan.
        runner.save_arrays(resume,**initial,time=np.float64(current),dt_hint=np.float64(hint),steps=np.int64(steps),rejected=np.int64(0),
                           input_identity=np.asarray(runner.sha(prepared/'preparation_receipt.json')),
                           source_identity=np.asarray(runner.sha(destination/'manifest.json')))
        return manifest
    coupled_solver.CoupledSolver=LateSolver;runner.freeze_sources=freeze
    try:
        with graph_scope.graph_adapters(None):
            base_graph=coupled_graph.CoupledTransportGraph
            class CheckedGraph(base_graph):
                def __init__(self,solver,**kwargs):
                    before={name:{k:a.get().view(np.uint64).copy() for k,a in getattr(solver,name).items()} for name in ('state','ledger')}
                    super().__init__(solver,**kwargs)
                    for name,group in before.items():
                        for key,value in group.items():
                            if not np.array_equal(getattr(solver,name)[key].get().view(np.uint64),value):
                                raise ValueError('Graph warmup modified exact late accepted state/ledger')
                    self.checkpoint_warmup_preserved=True
                def receipt(self):
                    return dict(super().receipt(),checkpoint_state_and_ledgers_preserved=self.checkpoint_warmup_preserved,
                        source_controls=candidate_context(),native_global_steps_at_resume=steps,
                        local_rejection_count_reset_to_zero=True,physical_ledgers_reset=False)
            coupled_graph.CoupledTransportGraph=CheckedGraph
            try:yield resume
            finally:coupled_graph.CoupledTransportGraph=base_graph
    finally:
        coupled_solver.CoupledSolver=saved_solver;runner.freeze_sources=saved_freeze


def summarize_late_trace(plan,diagnostics,worker):
    """Read-only scalar checks, not a substitute for fields/budget/identity gates."""
    from . import adaptive_trace_analysis as trace
    from . import adaptive_multilevel_trace as mg
    validate_controls(plan);diagnostics=Path(diagnostics)
    static=trace.read_json(diagnostics/'filter_controls.json')
    mg.check_controls(plan['GPU_candidate_context'],static)
    result=trace.summarize_streams(diagnostics/'step_trials.jsonl',diagnostics/'filter_calls.jsonl',
        start=plan['initial_time_s'],dtmax=.0125,launch_mode='graph')
    mg.accepted_cfl_gate(result,.8)
    if (worker['status']!='completed' or worker['steps']-plan['initial_global_steps']!=result['accepted'] or
        worker['rejected']!=result['rejected'] or worker['final_time_s']!=result['final_clock_s'] or
        abs(result['final_clock_s']-2904.)>1e-12 or worker['end_s']!=2904. or
        worker['output_every_s']!=plan['GPU_output_every_s'] or worker['dtmax_s']!=.0125 or
        worker['cfl']!=.8 or worker['morfac']!=25. or worker['launch_mode']!='graph'):
        raise ValueError('Local trace/global worker clocks, counters or controls differ')
    return dict(result,requested_filter_controls=tight.REQUESTED,effective_filter_controls=tight.EFFECTIVE,
        initial_global_steps=plan['initial_global_steps'],physical_accuracy_qualified=False,
        full20h_speed_qualified=False,scope='Absolute late24s local trace only')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--completed-case',type=Path,required=True)
    p.add_argument('--tag',required=True);args=p.parse_args()
    print(json.dumps(prepare(args.completed_case,args.tag),indent=2))

