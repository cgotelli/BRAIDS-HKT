"""Plan-gated corrected CPU cold20h diagnostic, not an adopted reference.

Executes only the already-built isolated outlet-bound candidate. Every input
byte remains the original cold-mobile deck; frozen oldCPU stays immutable.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from . import cpu_cold_ulp24 as base

TAG='outlet_bounds_cpu20h_20261002_v1'
RUNTIME=base.HERE/'runs/outlet_bounds_cpu_candidate_20261002_v1/runtime'
EXE_SHA='a9f9d17b96eb0b06144876bddea97ba796c1c33ea5e65d7f61338738749044d3'
OLD='  if(cpu_time_integrator==6) cpu_outlet_extension=1'
NEW='  if(cpu_time_integrator==6.or.cpu_time_integrator==43) cpu_outlet_extension=1'
TIMES=np.arange(121,dtype=float)*24.


def verify_runtime():
    identity=json.loads((RUNTIME/'build.json').read_text())
    base.require(identity['executable_sha256']==EXE_SHA and base.sha(RUNTIME/'xbraids_candidate.exe')==EXE_SHA,
                 'Corrected candidate executable differs')
    base.require(identity['parent_executable_sha256']==base.EXE_SHA,'Wrong frozen parent')
    files={str(RUNTIME/'build.json'):base.sha(RUNTIME/'build.json'),str(RUNTIME/'xbraids_candidate.exe'):EXE_SHA}
    parents={Path(name.replace('\\','/')).name:digest for name,digest in identity['parent_sources'].items()}
    base.require(set(identity['source_sha256'])==set(parents),'Candidate source set differs')
    for name,digest in identity['source_sha256'].items():
        candidate=RUNTIME/name;original=base.RUNTIME/name
        base.require(base.sha(candidate)==digest and base.sha(original)==parents[name],'Candidate/parent source changed')
        if name=='cutcell_solver.f90':
            left,right=original.read_text(),candidate.read_text()
            base.require(left.count(OLD)==1 and right==left.replace(OLD,NEW),'Unexpected hydraulic/source edit')
        else:base.require(digest==parents[name],'Non-driver numerical source changed')
        files[str(candidate)]=digest;files[str(original)]=parents[name]
    files[str(base.RUNTIME/'xbraids_candidate.exe')]=base.EXE_SHA
    for folder in (RUNTIME,base.RUNTIME):
        for path in folder.glob('*.dll'):files[str(path)]=base.sha(path)
    return identity,files


def verify_files(files):
    for path,digest in files.items():base.require(base.sha(path)==digest,'Authenticated file changed: '+path)


def prepare():
    from .coupled_reference import copy_inputs
    from braids_support.sediment.lfm_bend_2d.cold_optimized_comparison import verify_inputs
    root,report=base.HERE/'runs'/TAG,base.HERE/'reports'/TAG
    base.require(not root.exists() and not report.exists(),'Fresh run/report paths required')
    identity,authenticated=verify_runtime();meta=verify_inputs(base.SOURCE)
    base.require((meta['end'],meta['output'],meta['morfac'],meta['cfl'],meta['dt'])==(2880,24,25,.8,.0125),'Original controls differ')
    case=root/'control';origin=copy_inputs(base.SOURCE/'case.data',case);report.mkdir()
    # Retain all original metadata/dependencies, not only dependencies needed by
    # the executable; record the *actual* candidate separately in this plan.
    for name in meta['files']:
        source=base.SOURCE/name;target=case/name
        if not target.exists():shutil.copy2(source,target)
        base.require(base.sha(target)==meta['files'][name],'Input clone differs')
    shutil.copy2(base.SOURCE/'input_provenance.json',case/'input_provenance.json')
    authenticated.update({str(case/name):digest for name,digest in meta['files'].items()})
    authenticated[str(case/'input_provenance.json')]=base.sha(case/'input_provenance.json')
    for path in (Path(__file__),Path(base.__file__)):
        authenticated[str(path.resolve())]=base.sha(path);shutil.copy2(path,report/path.name)
    protected={str(base.SOURCE/name):digest for name,digest in meta['files'].items()}
    for name in ('input_provenance.json','execution.json','frames.bin','checkpoint.bin','secondary_flow.txt','secondary_flow_budget.txt'):
        protected[str(base.SOURCE/name)]=base.sha(base.SOURCE/name)
    plan=dict(schema=1,tag=TAG,case=str(case),report=str(report),original_reference=str(base.SOURCE),
        executable=str(RUNTIME/'xbraids_candidate.exe'),executable_sha256=EXE_SHA,runtime_identity=identity,
        authenticated_files=authenticated,protected_files=protected,input_copy=origin,
        all_original_input_files_byte_identical=True,endpoint_s=2880.,morphological_hours=20.,
        output_s=24.,output_frames=121,expected_times_s=TIMES.tolist(),morfac=25.,cfl=.8,dt_ceiling_s=.0125,
        environment=dict(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1',BRAIDS_PROGRESS_STEPS='20000'),
        initialization='Original uneroded mobile bed, shared original hydraulic spin-up; no restart, no ULP perturbation.',
        numerical_change='Enable existing virtual-exterior hydraulic outlet bounds for SSPRK43; no other formula/input change.',
        reference_policy='CorrectedCPU is a diagnostic candidate, NOT an approved replacement acceptance reference.',
        timing_policy='Concurrent jobs are not an uncontended benchmark. Record process wall time, do not assert speed qualification.',
        comparison_policy='Compare all121oldCPU outputs and original5mm/3mm diagnostics; do not treat any result as automatic promotion.',
        production_modified=False,delft_modified=False,simulation_launched=False)
    base.save_json(report/'plan.json',plan)
    return dict(plan=str(report/'plan.json'),sha256=base.sha(report/'plan.json'),simulation_launched=False,
                executable_sha256=EXE_SHA,all_original_input_files_byte_identical=True,frames=121)


def read_history(folder):
    from . import long_bend_analysis as long
    result=long.load_native(folder);times=result['time'];nc=len(result['geometry']['area'])
    base.require(times.shape==TIMES.shape and np.allclose(times,TIMES,rtol=0,atol=1e-7),'Require all121original output times')
    intensity=np.full((121,nc),np.nan)
    with (folder/'secondary_flow.txt').open() as stream:
        for line in stream:
            if line.startswith('#') or not line.strip():continue
            row=line.split()
            if int(row[2])!=0:continue
            t=float(row[0]);k=int(round(t/24.))
            base.require(0<=k<121 and abs(t-TIMES[k])<1e-7,'Invalid secondary output time')
            intensity[k,int(row[1])]=float(row[6])
    base.require(np.isfinite(intensity).all() and np.all(intensity[0]==0.),'Invalid signed secondary records')
    budget=np.loadtxt(folder/'secondary_flow_budget.txt',ndmin=2)
    base.require(budget.shape==(121,6) and np.allclose(budget[:,0],times,rtol=0,atol=1e-7),'Secondary budget grid differs')
    ledger=result['ledger'];base.require(ledger.shape==(121,2,5),'Native physical ledger layout differs')
    result.update(secondary=result['q'][:,0]*intensity,ledger_water=ledger[:,:,0],
        ledger_bed=ledger[:,:,3:5].transpose(0,2,1),ledger_secondary=budget[:,3,None],ledger_source=budget[:,4],
        native_secondary_budget=budget)
    return result


def analyze(plan):
    from . import long_bend_analysis as long
    from .candidate_analysis_review import hydraulic_metrics
    case,report=Path(plan['case']),Path(plan['report'])
    candidate,original=read_history(case),read_history(base.SOURCE);area=original['geometry']['area']
    for key in ('xy','area','bed'):
        np.testing.assert_array_equal(candidate['geometry'][key],original['geometry'][key])
    initial={}
    for key in ('q','layers','bed','secondary','ledger'):
        initial[key]=bool(np.array_equal(candidate[key][0].view(np.uint64),original[key][0].view(np.uint64)))
    base.require(all(initial.values()),'Initial candidate/reference state differs')
    comparison,_,_=long.comparison(candidate,original,area,1e-4)
    accounts={name:long.budgets(value,area,1e-4) for name,value in [('candidate',candidate),('original',original)]}
    morphology={name:long.morphology(value,original['geometry']) for name,value in [('candidate',candidate),('original',original)]}
    hydraulic=hydraulic_metrics(candidate,original,area,1e-4)
    for name,value in [('candidate',candidate),('original',original)]:
        np.savez(report/(name+'_frames.npz'),**{key:value[key] for key in ('time','q','layers','bed','secondary','ledger_water','ledger_bed','ledger_secondary','ledger_source')})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hours=candidate['time']*25/3600;fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    curves=comparison['curves']
    axes[0].plot(hours,curves['bed_maximum_m']*1000,label='Maximum');axes[0].plot(hours,curves['bed_rms_m']*1000,label='Area-RMS')
    axes[0].set(title='CorrectedCPU versus originalCPU',ylabel='Bed difference(mm)');axes[0].legend()
    for name,value in [('Corrected',candidate),('Original',original)]:
        axes[1].plot(hours,(value['bed'][:,3113]-value['bed'][0,3113])*1000,label=name)
    axes[1].set(title='Inlet/wall corner3113',ylabel='Bed change(mm)');axes[1].legend()
    axes[2].plot(hours,hydraulic['curves']['depth_area_rms_m']*1000,label='Depth area-RMS')
    axes[2].set(title='Hydraulic diagnostic',ylabel='Depth difference(mm)')
    for ax in axes:ax.set_xlabel('Morphological hours');ax.grid(alpha=.2)
    fig.suptitle('Existing outlet-bound repair enabled: diagnostic, not an adopted acceptance reference')
    figure=report/'COMPARISON.png';fig.savefig(figure,dpi=160);plt.close(fig)
    result=dict(initial_fields_bitwise_equal=initial,all121outputs_aligned=True,comparison_vs_immutable_original=comparison,
        hydraulic_diagnostics=hydraulic,conservation=accounts,morphology=morphology,
        reference_identity=original['identity'],candidate_identity=candidate['identity'],
        figures=[dict(file=figure.name,sha256=base.sha(figure))],
        arrays_sha256={path.name:base.sha(path) for path in report.glob('*_frames.npz')},
        analysis_dependency_sha256={str(Path(path).resolve()):base.sha(path) for path in
            (__file__,base.__file__,long.__file__,__import__('inspect').getfile(hydraulic_metrics))},
        corrected_CPU_approved_as_acceptance_reference=False,production_adopted=False,experimental_truth_established=False,
        original_5mm_gate_relaxed=False,performance_qualified=False,delft_modified=False,
        timing_note=plan['timing_policy'],input_source_provenance_note='Input metadata is intentionally byte-identical to oldCPU; actual executed corrected runtime identity is recorded separately.')
    base.save_json(report/'comparison.json',long.serializable(result))
    (report/'REPORT.txt').write_text('CORRECTED CPU20h DIAGNOSTIC; NOT AN APPROVED REFERENCE\n'
        f"All121outputs compared. Initial fields bitwise equal: {all(initial.values())}.\n"
        f"Versus originalCPU: maximum bed {comparison['maximum_bed_difference_m']*1000:.9g}mm; maximum area-RMS {comparison['maximum_bed_rms_m']*1000:.9g}mm.\n"
        'Hydraulic differences and own budgets are recorded. No automatic GPU qualification, promotion, reference replacement or uncontended speed claim.\n',encoding='utf-8')
    return long.serializable(result)


def execute(plan_path,approved_sha256):
    plan_path=Path(plan_path).resolve();base.require(base.sha(plan_path)==approved_sha256,'Reviewed plan SHA differs')
    plan=json.loads(plan_path.read_text());base.require(plan['tag']==TAG and plan['endpoint_s']==2880.,'Wrong bounded plan')
    verify_files({**plan['authenticated_files'],**plan['protected_files']});verify_runtime()
    case,report=Path(plan['case']),Path(plan['report'])
    base.require(not (case/'execution.json').exists(),'Never restart/overwrite an earlier candidate execution')
    removed={key:value for key,value in os.environ.items() if key.upper().startswith(('BRAIDS_','OMP_'))}
    environment={key:value for key,value in os.environ.items() if key not in removed};environment.update(plan['environment'])
    started=time.perf_counter();record=dict(status='running',removed_inherited_environment=removed,
        environment=plan['environment'],executable_sha256=EXE_SHA,source_sha256=plan['runtime_identity']['source_sha256'],
        threads=8,plan_sha256=approved_sha256,performance_claim=False,concurrency_not_excluded=True)
    with (case/'native.log').open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([plan['executable'],'case.data'],cwd=case,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        record['pid']=process.pid;base.save_json(case/'execution.json',record)
        try:code=process.wait(timeout=14400)
        except subprocess.TimeoutExpired:process.kill();code=process.wait();record['timed_out']=True
    record.update(return_code=code,wall_s=time.perf_counter()-started,status='completed' if code==0 else 'failed',
                  native_log_sha256=base.sha(case/'native.log'))
    base.save_json(case/'execution.json',record);base.require(code==0,'CorrectedCPU process failed; retain diagnostic log')
    verify_files({**plan['authenticated_files'],**plan['protected_files']});result=analyze(plan)
    verify_files({**plan['authenticated_files'],**plan['protected_files']});verify_runtime()
    result['all_authenticated_identities_retained']=True;base.save_json(report/'comparison.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');p=sub.add_parser('execute');p.add_argument('--plan',type=Path,required=True);p.add_argument('--approved-sha256',required=True)
    args=parser.parse_args()
    if args.command=='prepare':print(json.dumps(prepare(),indent=2))
    else:
        result=execute(args.plan,args.approved_sha256)
        print(json.dumps(result['comparison_vs_immutable_original']['gates'],indent=2))
