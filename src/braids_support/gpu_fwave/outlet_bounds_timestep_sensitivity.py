"""Four bounded common-checkpoint late24s timestep jobs; no kernel changes.

Preparation/host tests do not launch solvers. Execute requires root-approved
plan SHA and a released hardware window. This is local timestep sensitivity,
not convergence of the preceding cold20h trajectory and not a speed benchmark.
"""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from . import outlet_bounds_timestep_checkpoint as adapter
from . import long_bend_analysis as analysis

base=adapter.base;HERE=base.HERE;ROOT=base.ROOT
TAG='outlet_bounds_timestep_20261002_v1'
SOURCE=HERE/'runs/outlet_bounds_cpu20h_20261002_v1/control'
GPU_PYTHON=HERE/'.venv_gpu/Scripts/python.exe'
FIELDS=('q','layers','secondary','ledger_water','ledger_bed','ledger_secondary','ledger_source')


def save(path,value):base.save_json(path,analysis.serializable(value))


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint64),b.view(np.uint64))


def derive_bed(layers,data):
    result=np.broadcast_to(data['rock'],layers.shape[:-3]+(len(data['rock']),)).copy()
    for grain in range(2):
        result+=layers[...,0,grain,:]/data['bulk_fraction'][grain]
        result+=layers[...,1,grain,:]/data['bulk_fraction'][grain]
    return result


def prepare():
    root=HERE/'runs'/TAG;report=HERE/'reports'/TAG
    if root.exists() or report.exists():raise FileExistsError('Fresh sensitivity paths required')
    root.mkdir();report.mkdir();(report/'source').mkdir()
    from braids_qgis.core.results import read_frames
    import inspect
    dependencies={str(Path(module.__file__).resolve()):base.sha(module.__file__)
                  for module in (adapter,analysis,base)}
    for path in (Path(__file__).resolve(),Path(inspect.getfile(read_frames)).resolve(),GPU_PYTHON,ROOT/'.venv/Scripts/python.exe'):
        dependencies[str(path)]=base.sha(path)
    for path in (Path(__file__),Path(adapter.__file__)):
        shutil.copy2(path,report/'source'/path.name)
    children={};jobs=[]
    for label,dtmax in (('baseline',.0125),('half',.00625)):
        result=adapter.prepare(SOURCE,TAG+'_'+label,dtmax)
        path=Path(result['plan']);child=json.loads(path.read_text())
        child['adapter_dependency_sha256'][str(Path(__file__).resolve())]=base.sha(__file__)
        save(path,child)
        children[label]=dict(path=str(path),sha256=base.sha(path),dtmax_s=dtmax)
        dependencies.update(child['adapter_dependency_sha256']);dependencies[str(path)]=base.sha(path)
        for backend in ('cpu','gpu'):
            jobs.append(dict(name=backend+'_'+label,backend=backend,label=label,output=child[backend.upper()+'_output'],
                             stop_file=str(adapter.graph_scope.v1.stop_path(child['GPU_output'])) if backend=='gpu' else None))
    plan=dict(schema=1,scope='local_timestep_sensitivity_20_to_20_167_morph_hours',tag=TAG,
        root=str(root),report=str(report),source_checkpoint=str(SOURCE/'checkpoint.bin'),source_checkpoint_sha256=base.sha(SOURCE/'checkpoint.bin'),
        children=children,jobs=jobs,authenticated_files=dependencies,
        controls=dict(CFL=.8,MF=25.,endpoint_s=2904.,outlet_extension=1,CPU_filter_rtol=1e-4,GPU_filter_rtol=1e-8),
        physical_state_and_ledgers_same=True,control_only_checkpoint_rebinding=True,
        checkpoint_changes='Only new input fingerprint and min(original timestep hint, selected dtmax); exact byte ranges in each child plan',
        expected_wall_s=[240,360],numerical_jobs=4,quarter_dt_not_authorized=True,
        interpretation='Compare CPU timestep sensitivity, GPU timestep sensitivity and matched-backend differences on one identical late state. Cannot certify preceding cold20h convergence.',
        immutable_limits=dict(analysis.LIMITS),full20h_qualification=False,performance_claim=False,
        production_modified=False,Delft3D_modified=False)
    validate(plan);save(report/'plan.json',plan)
    return dict(plan=str(report/'plan.json'),sha256=base.sha(report/'plan.json'),jobs=4,solvers_launched=0)


def validate(plan):
    if (plan['schema']!=1 or plan['tag']!=TAG or plan['immutable_limits']!=analysis.LIMITS or
        plan['controls']!=dict(CFL=.8,MF=25.,endpoint_s=2904.,outlet_extension=1,CPU_filter_rtol=1e-4,GPU_filter_rtol=1e-8) or
        [(j['backend'],j['label']) for j in plan['jobs']]!=[('cpu','baseline'),('gpu','baseline'),('cpu','half'),('gpu','half')]):
        raise ValueError('Reviewed four-job scope differs')
    adapter.cpu.verify_files(plan['authenticated_files'])
    if base.sha(plan['source_checkpoint'])!=plan['source_checkpoint_sha256']:raise ValueError('Original checkpoint changed')
    physical=[]
    for label,item in plan['children'].items():
        child=json.loads(Path(item['path']).read_text());adapter.validate_resume_plan(child,item['sha256'])
        if child['dt_max_s']!=item['dtmax_s'] or item['dtmax_s']!=dict(baseline=.0125,half=.00625)[label]:
            raise ValueError('Wrong declared timestep')
        with np.load(Path(child['prepared_input']).parent/'exact_initial.npz',allow_pickle=False) as data:
            physical.append({k:data[k].copy() for k in FIELDS+('time','steps')})
    if len(physical)!=2 or not all(exact(physical[0][k],physical[1][k]) for k in physical[0]):
        raise ValueError('Timestep cases differ in physical initial state, ledgers or clock')
    for job in plan['jobs']:
        child=json.loads(Path(plan['children'][job['label']]['path']).read_text())
        if job['output']!=child[job['backend'].upper()+'_output']:raise ValueError('Output route differs')
        if job['backend']=='gpu' and job['stop_file']!=str(adapter.graph_scope.v1.stop_path(child['GPU_output'])):
            raise ValueError('Stop path differs')
    return True


def gpu_child(plan_path,approved,name):
    path=Path(plan_path)
    if base.sha(path)!=approved:raise ValueError('Exact parent plan SHA required')
    plan=json.loads(path.read_text());validate(plan)
    matches=[j for j in plan['jobs'] if j['name']==name and j['backend']=='gpu']
    if len(matches)!=1:raise ValueError('Unknown GPU case')
    job=matches[0];item=plan['children'][job['label']];child=json.loads(Path(item['path']).read_text())
    from braids_gpu import runner
    output=Path(job['output']);diagnostics=output.with_name(output.name+'_sensitivity_audit')
    if output.exists() or diagnostics.exists():raise FileExistsError('Fresh GPU continuation required')
    diagnostics.mkdir(parents=True);traces=adapter.tight.base.Traces(diagnostics)
    result=dict(status='running',parent_plan_sha256=approved,child_plan_sha256=item['sha256'],worker_sha256=base.sha(__file__))
    try:
        with adapter.gpu_resume_hooks(child,approved_plan_sha256=item['sha256'],traces=traces) as resume:
            worker=runner.run_gpu(**adapter.worker_arguments(child,resume,stop_file=job['stop_file']))
        result.update(status=worker['status'])
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        result['trace']=traces.close()
        if (diagnostics/'late_control.json').exists():result['trace']['artifacts']['late_control.json']=base.sha(diagnostics/'late_control.json')
        save(diagnostics/'execution.json',result)
    if result['status']!='completed':raise ValueError('GPU continuation incomplete')
    return result


def load_cpu(child):
    from braids_qgis.core.results import read_frames
    out=Path(child['CPU_output']);data=adapter.cr.read_records(Path(child['prepared_input'])/'coupled_input.bin')
    initial=adapter.decode_checkpoint((Path(child['prepared_input']).parent/'native_checkpoint.bin').read_bytes())
    final=adapter.decode_checkpoint((out/'checkpoint.bin').read_bytes())
    if final['configuration_prefix']!=initial['configuration_prefix']:raise ValueError('Native restart configuration differs')
    checkpoints=(initial,final);frames=read_frames(out/'frames.bin')
    if len(frames)!=2:raise ValueError('Exactly two native outputs required')
    rows=[adapter.checkpoint_arrays(cp,data['bed_indices']) for cp in checkpoints]
    fields={key:np.stack([r[key] for r in rows]) for key in FIELDS}
    fields['time']=np.array([f['time_s'] for f in frames]);fields['bed']=derive_bed(fields['layers'],data)
    for frame,cp in zip(frames,checkpoints):
        if (frame['time_s']!=cp['time'] or int(frame['steps'])!=cp['steps'] or
            not exact(frame['q'],cp['state'][:9].T) or not exact(frame['ledger'],cp['ledger'][:5].T)):
            raise ValueError('Native output/checkpoint mismatch')
    budget=np.loadtxt(out/'secondary_flow_budget.txt',ndmin=2)
    expected=np.column_stack((fields['time'],fields['secondary']@data['area'],
        [cp['secondary_initial_integral'] for cp in checkpoints],fields['ledger_secondary'].sum(axis=1),fields['ledger_source']))
    if budget.shape!=(2,6) or not np.allclose(budget[:,:5],expected,atol=1e-12,rtol=1e-12):
        raise ValueError('Native independently emitted hI budget differs')
    log=(out/'native.log').read_text()
    if 'wet stage-outlet bound extension=1' not in log:raise ValueError('Corrected native boundary not executed')
    return fields,data,dict(accepted=final['steps']-initial['steps'],global_steps=final['steps'],
        native_per_step_trace_available=False,dtmax_enforced_by_unchanged_native_controller=child['dt_max_s'],
        exact_hI_from_accepted_checkpoints=True,ancillary_budget_max=float(abs(budget[:,5]).max()),
        artifacts={name:base.sha(out/name) for name in ('frames.bin','checkpoint.bin','secondary_flow_budget.txt','native.log')})


def load_gpu(child,job,approved):
    from braids_gpu import runner
    out=Path(job['output']);manifest=json.loads((out/'run.json').read_text())
    data,identity=runner.load_prepared(child['prepared_input']);archive=json.loads((out/'source/manifest.json').read_text())
    expected=dict(child['GPU_source_sha256'])
    expected.update({Path(p).name:d for p,d in child['adapter_dependency_sha256'].items()})
    expected[Path(adapter.__file__).name]=child['helper_sha256']
    expected['late_adapter_plan.json']=base.sha(child['adapter_plan_path'])
    if archive!=expected or manifest['source_sha256']!=archive or manifest['input_preparation_sha256']!=identity:
        raise ValueError('Actual GPU source/input manifest differs')
    adapter.cpu.verify_files({str(out/'source'/name):digest for name,digest in archive.items()})
    records=manifest['frames'];rows=[];times=[]
    if len(records)!=2:raise ValueError('Exactly two GPU outputs required')
    for record in records:
        path=(out/record['file']).resolve()
        if path.parent!=out.resolve() or base.sha(path)!=record['sha256']:raise ValueError('GPU frame identity differs')
        with np.load(path,allow_pickle=False) as frame:
            if float(frame['time'])!=record['time_s']:raise ValueError('GPU frame time differs')
            rows.append({key:frame[key].copy() for key in FIELDS+('bed_elevation',)});times.append(float(frame['time']))
    fields={key:np.stack([r[key] for r in rows]) for key in rows[0]};fields['bed']=fields.pop('bed_elevation');fields['time']=np.asarray(times)
    with np.load(out/'checkpoint.npz',allow_pickle=False) as cp:
        if (float(cp['time'])!=times[-1] or int(cp['steps'])!=manifest['steps'] or int(cp['rejected'])!=manifest['rejected'] or
            cp['source_identity'].item()!=base.sha(out/'source/manifest.json') or cp['input_identity'].item()!=identity or
            any(not exact(cp[k],rows[-1][k]) for k in rows[-1])):raise ValueError('GPU final checkpoint/output differs')
    graph=manifest['transport_graph']
    if graph.get('checkpoint_state_and_ledgers_preserved') is not True or graph.get('source_controls')!=adapter.candidate_context():
        raise ValueError('Late graph state/control proof missing')
    diagnostics=out.with_name(out.name+'_sensitivity_audit');receipt=json.loads((diagnostics/'execution.json').read_text())
    if receipt['status']!='completed' or receipt['parent_plan_sha256']!=approved or receipt['worker_sha256']!=base.sha(__file__):
        raise ValueError('GPU execution identity differs')
    adapter.cpu.verify_files({str(diagnostics/name):digest for name,digest in receipt['trace']['artifacts'].items()})
    control=json.loads((diagnostics/'late_control.json').read_text())
    if not control['physical_state_and_ledgers_copied_bitwise'] or control['physical_ledgers_reset']:
        raise ValueError('Late physical-state/ledger transfer differs')
    return fields,data,adapter.summarize_late_trace(child,diagnostics,manifest)


def execute(path,approved,hardware_released=False):
    path=Path(path)
    if not hardware_released or base.sha(path)!=approved:raise ValueError('Reviewed plan SHA and hardware release required')
    plan=json.loads(path.read_text());validate(plan);report=Path(plan['report'])
    if (report/'execution.json').exists():raise FileExistsError('Preserve earlier attempt')
    result=dict(status='running',plan_sha256=approved,jobs=[],no_performance_claim=True,full20h_qualification=False)
    fields={};save(report/'execution.json',result)
    try:
        for job in plan['jobs']:
            validate(plan);item=plan['children'][job['label']];child=json.loads(Path(item['path']).read_text())
            out=Path(job['output']);is_cpu=job['backend']=='cpu'
            command=child['native_command'] if is_cpu else [str(GPU_PYTHON),'-m','braids_support.gpu_fwave.outlet_bounds_timestep_sensitivity',
                'gpu-child','--plan',str(path),'--approved-plan-sha256',approved,'--name',job['name']]
            env={k:v for k,v in os.environ.items() if not k.upper().startswith(('BRAIDS_','OMP_','OPENBLAS_','MKL_','NUMEXPR_'))}
            env.update(adapter.NATIVE_ENV);env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            log=out/'native.log' if is_cpu else report/(job['name']+'.log')
            with log.open('x') as stream:
                began=time.perf_counter();proc=subprocess.run(command,cwd=out if is_cpu else ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=900,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                wall=time.perf_counter()-began
            record=dict(**job,returncode=proc.returncode,diagnostic_process_wall_s=wall,log_sha256=base.sha(log));result['jobs'].append(record)
            save(report/'execution.json',result);proc.check_returncode()
            values,data,receipt=load_cpu(child) if is_cpu else load_gpu(child,job,approved)
            adapter.check_output_times(values['time'],values['time'],child)
            with np.load(Path(child['prepared_input']).parent/'exact_initial.npz',allow_pickle=False) as initial:
                if not all(exact(values[k][0],initial[k]) for k in FIELDS):raise ValueError('Exact physical initial state/ledger differs')
            budgets=analysis.budgets(values,data['area'],float(data['dry_tolerance']))
            if not budgets['passed']:raise ValueError('Incremental conservation/nonnegativity failed')
            record.update(receipt=receipt,budgets=budgets,passed=True);fields[job['name']]=values
            np.savez(report/(job['name']+'_frames.npz'),**values);save(report/'execution.json',result)
        comparisons={}
        for name,left,right in (('CPU_half_minus_baseline','cpu_half','cpu_baseline'),('GPU_half_minus_baseline','gpu_half','gpu_baseline'),
                                ('GPU_minus_CPU_baseline','gpu_baseline','cpu_baseline'),('GPU_minus_CPU_half','gpu_half','cpu_half')):
            a,b=fields[left],fields[right];comparison,gi,ri=analysis.comparison(a,b,data['area'],float(data['dry_tolerance']))
            if len(gi)!=2 or len(ri)!=2 or abs(a['time']-b['time']).max()>1e-12:raise ValueError('Unmatched output times')
            comparison['focus']={str(i):dict(bed_difference_m=float(a['bed'][-1,i]-b['bed'][-1,i]),
                water_depth_difference_m=float(a['q'][-1,0,i]-b['q'][-1,0,i])) for i in (4911,3113,776,1140)}
            comparison['erosion_volume_difference_m3']=float((np.maximum(a['bed'][0]-a['bed'][-1],0)-np.maximum(b['bed'][0]-b['bed'][-1],0))@data['area'])
            comparison['deposition_volume_difference_m3']=float((np.maximum(a['bed'][-1]-a['bed'][0],0)-np.maximum(b['bed'][-1]-b['bed'][0],0))@data['area'])
            comparisons[name]=comparison
        validate(plan);result.update(status='completed',comparisons=comparisons,
            original_bed_gates_passed=all(x['passed'] for x in comparisons.values()),
            interpretation=plan['interpretation'])
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save(report/'execution.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True);sub.add_parser('prepare')
    for name in ('execute','gpu-child'):
        s=sub.add_parser(name);s.add_argument('--plan',type=Path,required=True);s.add_argument('--approved-plan-sha256',required=True)
        if name=='execute':s.add_argument('--hardware-released',action='store_true')
        else:s.add_argument('--name',required=True)
    args=p.parse_args()
    value=prepare() if args.action=='prepare' else gpu_child(args.plan,args.approved_plan_sha256,args.name) if args.action=='gpu-child' else execute(args.plan,args.approved_plan_sha256,args.hardware_released)
    print(json.dumps(analysis.serializable(value),indent=2,allow_nan=False))
