"""Prepare an exact, common native-checkpoint continuation; never build/run.

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
import struct
import numpy as np
from . import cpu_cold_ulp24 as base
from . import outlet_bounds_cpu20h as cpu
from . import outlet_bounds_gpu_candidate_v2 as graph_scope
from . import coupled_reference as cr


def candidate_context():
    """This draft adapter is schema6 only, not a future tighter-filter claim."""
    return dict(wrapper_schema=6,wrapper_entrypoint='outlet_bounds_gpu_candidate_v2.py',
                outlet_extension=1,filter='multilevel',
                requested_filter_controls=dict(graph_scope.v1.CONTROLS),
                effective_filter_controls=dict(graph_scope.v1.CONTROLS),
                filter_tolerance_changed=False,
                qualification='Unqualified late adapter; candidate selection and timing approval are still required')


def decode_checkpoint(blob):
    """Strict frozen fixed-grid10-state/2label/2layer/SSP43 checkpoint decoder."""
    if len(blob)<100:raise ValueError('Truncated checkpoint')
    fingerprint=blob[:64].decode('ascii')
    if len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('Invalid fingerprint')
    nc,nb,count=struct.unpack_from('<iii',blob,64)
    if nc!=5012 or nb!=2 or not 0<count<1000000:
        raise ValueError('Require the existing5012cell2boundary bend checkpoint')
    pos=76+8*count
    expected=pos+20+8*(10*nc+6*nb+10)+12
    if len(blob)!=expected:raise ValueError('Unexpected checkpoint layout/additional physics')
    signature=np.frombuffer(blob,dtype='<f8',count=count,offset=76).copy()
    current,hint,steps=struct.unpack_from('<ddi',blob,pos);pos+=20
    q=np.frombuffer(blob,dtype='<f8',count=10*nc,offset=pos).reshape(nc,10).T.copy();pos+=80*nc
    ledger=np.frombuffer(blob,dtype='<f8',count=6*nb,offset=pos).reshape(nb,6).T.copy();pos+=48*nb
    ancillary=np.frombuffer(blob,dtype='<f8',count=10,offset=pos).copy();pos+=80
    if blob[pos:pos+8]!=b'BRDINT01' or struct.unpack_from('<i',blob,pos+8)[0]!=43:
        raise ValueError('Checkpoint must retain SSPRK43')
    if not all(np.isfinite(x).all() for x in (signature,q,ledger,ancillary,np.array([current,hint]))):
        raise ValueError('Nonfinite checkpoint')
    if current<0 or not 0<hint<=.0125 or steps<0 or np.any(q[0]<0) or np.any(q[3:9]<0):
        raise ValueError('Invalid physical checkpoint state')
    if np.any(q[3:5]):raise ValueError('Suspended transport is outside this late adapter')
    return dict(fingerprint=fingerprint,signature=signature,configuration_prefix=blob[:76+8*count],
                time=current,dt_hint=hint,steps=steps,state=q,ledger=ledger,ancillary=ancillary,
                secondary_initial_integral=ancillary[0],secondary_source_cumulative=ancillary[1],
                integrator=43)


def write_records(path,records):
    """Preserve native typed/ranked record ABI; no numerical cast beyond FP64/I32."""
    with Path(path).open('xb') as stream:
        stream.write(cr.MAGIC+struct.pack('<i',1))
        for name,value in records.items():
            array=np.asarray(value);kind=1 if array.dtype.kind in 'iub' else 2
            if array.ndim>4 or len(name)>64 or array.dtype.kind not in 'iubf':
                raise ValueError('Unsupported record')
            dims=list(array.shape)+[0]*(4-array.ndim)
            stream.write(name.ljust(64).encode('ascii')+struct.pack('<6i',kind,array.ndim,*dims))
            stream.write(array.astype('<i4' if kind==1 else '<f8',copy=False).tobytes(order='F'))
        stream.write(b'END'.ljust(64,b' ')+struct.pack('<6i',0,0,0,0,0,0))


def checkpoint_snapshot(original,checkpoint,end):
    """Copy physical fields/clock only; retain static geometry, methods and forcing."""
    data=dict(original)
    if (data['nstate'],data['nledger'],data['nfrac'],data['nbed_layers'],data['cpu_time_integrator'])!=(10,6,2,2,43):
        raise ValueError('Unexpected coupled schema')
    if np.any(data['nhydro_series']) or int(data['nmorfac'])!=0:
        raise ValueError('Absolute-time series not qualified by graph continuation; do not reset time or disable forcing')
    if any(data[k] for k in ('amr_enabled','bank_failure','bank_front_tracking','bedload_adaptation','bedform_roughness')):
        raise ValueError('Unsupported stateful physics')
    if (float(data['morphological_factor']),float(data['cfl_target']),float(data['dtmax']))!=(25.,.8,.0125):
        raise ValueError('Original controls differ')
    if not np.isfinite(end) or end!=checkpoint['time']+24.:raise ValueError('Exactly24hydraulic seconds required')
    q=checkpoint['state']
    if q.shape!=data['state'].shape:raise ValueError('Checkpoint geometry/layout differs')
    indices=np.asarray(data['bed_indices'],int);layers=np.ascontiguousarray(q[indices])
    bulk=np.asarray(data['bulk_fraction'],dtype=np.float64)
    if bulk.shape!=(2,) or not np.isfinite(bulk).all() or np.any(bulk<=0):
        raise ValueError('Invalid frozen bulk densities')
    # Mirror st_bed exactly: grain-major active then substrate, divide each
    # contribution before addition. No hardcoded density or reassociated sum.
    bed=np.asarray(data['rock'],dtype=np.float64).copy()
    for grain in range(2):
        bed+=layers[0,grain]/bulk[grain]
        bed+=layers[1,grain]/bulk[grain]
    # Conserved hI is copied, never recovered from h*printed intensity.
    data.update(state=q.copy(),ledger=checkpoint['ledger'].copy(),layer_mass=layers,
                bed=bed,
                time=float(checkpoint['time']),endtime=float(end))
    return data


def checkpoint_arrays(checkpoint,bed_indices):
    q=checkpoint['state'];ledger=checkpoint['ledger']
    return dict(q=np.ascontiguousarray(q[:3]),layers=np.ascontiguousarray(q[np.asarray(bed_indices,int)]),
                secondary=np.ascontiguousarray(q[9]),ledger_water=np.ascontiguousarray(ledger[0]),
                ledger_bed=np.ascontiguousarray(ledger[3:5]),ledger_secondary=np.ascontiguousarray(ledger[5]),
                ledger_source=np.asarray(checkpoint['secondary_source_cumulative'],dtype=np.float64))


def prepare(completed_case,tag):
    case=Path(completed_case).resolve()
    if Path(tag).name!=tag or not tag.startswith('outlet_bounds_late24_'):
        raise ValueError('New task-specific tag required')
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
    original=cr.read_records(source/'coupled_input.bin')
    data=checkpoint_snapshot(original,checkpoint,checkpoint['time']+24.)
    run=base.HERE/'runs'/tag;report=base.HERE/'reports'/tag
    if run.exists() or report.exists():raise FileExistsError('Fresh late adapter paths required')
    run.mkdir(parents=True);report.mkdir(parents=True);prepared=run/'prepared';prepared.mkdir()
    shutil.copy2(__file__,report/Path(__file__).name)
    shutil.copy2(case/'checkpoint.bin',run/'native_checkpoint.bin')
    write_records(prepared/'coupled_input.bin',data)
    shutil.copy2(source/'coupled_diffusion.bin',prepared/'coupled_diffusion.bin')
    roundtrip=cr.read_records(prepared/'coupled_input.bin')
    changed=('state','ledger','layer_mass','bed','time','endtime')
    for key,value in original.items():
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
    plan=dict(schema=1,tag=tag,prepared_only=True,simulation_launched=False,
              corrected_CPU_case=str(case),checkpoint_sha256=base.sha(run/'native_checkpoint.bin'),
              exact_initial_sha256=base.sha(run/'exact_initial.npz'),
              prepared_input=str(prepared),preparation_sha256=base.sha(prepared/'preparation_receipt.json'),
              CPU_output=str(cpu_case),GPU_output=str(run/'gpu'),
              native_command=[str(cpu.RUNTIME/'xbraids_candidate.exe'),'case.data',str(run/'native_checkpoint.bin')],
              CPU_input_sha256={str(path):base.sha(path) for path in cpu_case.glob('case.data*')},
              CPU_runtime=identity,protected_files=protected,
              checkpoint_configuration_bitwise_equal=True,conserved_hI_copied_exactly=True,
              all_physical_ledgers_preserved=True,source_integral_preserved=True,
              discarded_native_diagnostics='None onCPU; GPU does not implement the ancillary diagnostic momentum-ledger reports, which are archived in exact_initial.npz',
              initial_time_s=checkpoint['time'],end_time_s=data['endtime'],duration_s=24.,
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
                  for module in (base,cpu,graph_scope,graph_scope.v1,cr)},
              GPU_execution_adapter_qualified=False,CPU_restart_executed=False,GPU_calls=0,production_modified=False,delft_modified=False)
    cpu.verify_files(protected);base.save_json(report/'plan.json',plan)
    return dict(plan=str(report/'plan.json'),sha256=base.sha(report/'plan.json'),simulations_launched=False)


def validate_resume_plan(plan):
    """Host-only fail-closed guard, before any device constructor/import."""
    if plan.get('GPU_candidate_context')!=candidate_context():
        raise ValueError('Late adapter candidate/filter context differs; do not relabel schema6 as a tighter variant')
    if plan.get('helper_sha256')!=base.sha(__file__):raise ValueError('Late adapter source changed')
    prepared=Path(plan['prepared_input']);root=prepared.parent
    for path,key in ((root/'exact_initial.npz','exact_initial_sha256'),
                     (root/'native_checkpoint.bin','checkpoint_sha256'),
                     (prepared/'preparation_receipt.json','preparation_sha256')):
        if base.sha(path)!=plan[key]:raise ValueError('Late adapter artifact changed: '+str(path))
    cpu.verify_files(plan['protected_files']);cpu.verify_files(plan['adapter_dependency_sha256'])
    current=graph_scope.base.verify_baseline(graph_scope.base.LONG/'gpu/source')
    if current!=plan['GPU_source_sha256']:raise ValueError('Frozen GPU numerical sources differ')
    return True


@contextmanager
def gpu_resume_hooks(plan):
    """Unexecuted implementation hook for the future reviewed GPU envelope.

    Caller must verify/freeze the plan and exact_initial/source dependencies,
    call worker.run_gpu with resume=returned_path, absoluteend, dtmax/output,
    and retain all receipts. This is NOT itself an execution/qualification API.
    """
    validate_resume_plan(plan)
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
            for key,array in self.ledger.items():array.set(initial['ledger_'+key])
            for key,array in self.state.items():
                if not np.array_equal(array.get().view(np.uint64),initial[key].view(np.uint64)):
                    raise ValueError('Checkpoint state did not reach GPU bitwise')
    def freeze(destination):
        manifest=saved_freeze(destination);destination=Path(destination)
        # The original manifest remains numerical sources only. Adapter/plan
        # provenance lives in the enclosing reviewed late protocol.
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
            coupled_graph.CoupledTransportGraph=CheckedGraph
            try:yield resume
            finally:coupled_graph.CoupledTransportGraph=base_graph
    finally:
        coupled_solver.CoupledSolver=saved_solver;runner.freeze_sources=saved_freeze


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--completed-case',type=Path,required=True)
    p.add_argument('--tag',required=True);args=p.parse_args()
    print(json.dumps(prepare(args.completed_case,args.tag),indent=2))
