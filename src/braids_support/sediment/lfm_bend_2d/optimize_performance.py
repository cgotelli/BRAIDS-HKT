"""Build, benchmark and qualify curvature/identical-grain execution improvements.

Delft3D is not opened or executed. Runs and immutable source copies are isolated.
The archived MF25 bend equations and all accuracy settings are retained.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from . import run_case as r
from . import profile_current as audit
from .mf25_validation_runtime import validation_writer

TAG='performance_optimized_20260930_v4'
HERE=r.HERE
PARENT=HERE/'runs/fine_mf25_3600_v2_runtime'
PROFILE=HERE/'runs/performance_20260930_v1/runtime'
FAST=HERE/'native/secondary_curvature_exact.f90'
LIMITS=dict(depth_m=1e-6,velocity_m_s=1e-5,bed_m=1e-6,other_state=1e-6,
            depth_rms_m=1e-8,bed_rms_m=1e-8,secondary_intensity_m_s=1e-6,
            secondary_budget=1e-8,momentum_budget=1e-8,
            boundary_ledger=1e-6,water_increment_budget_m3=1e-8,
            sediment_increment_budget_kg=1e-6,minimum_speedup=1.05)


def build(root,variant='both'):
    runtime=root/'runtime';runtime.mkdir(parents=True,exist_ok=False)
    baseline=PROFILE/'xbraids_profile.exe'
    prior=json.loads((PROFILE/'build.json').read_text())
    assert r.sha256(baseline)==prior['executable_sha256']
    identity=json.loads((PARENT/'build.json').read_text())
    command=[identity['command'][0],'-O3','-flto=8','-fopenmp',
             '-ffree-line-length-none','-ffixed-line-length-none']
    hashes={}
    for name,expected in identity['source_sha256'].items():
        base=Path(name).name
        if base=='cutcell_solver.f90':source=PROFILE/base
        elif base=='secondary_curvature_regularization.f90':source=PARENT/base if variant in ('grain_only','jacobi_tight') else FAST
        elif base=='braids_bedload_mpm_module.f90' and variant in ('filter_only','jacobi_tight'):source=PARENT/base
        else:source=r.ROOT/name
        if base not in ('cutcell_solver.f90','secondary_curvature_regularization.f90','braids_bedload_mpm_module.f90'):
            if r.sha256(source)!=expected:raise ValueError(f'Unrelated source drift: {source}')
        target=runtime/base;shutil.copy2(source,target)
        if variant in ('oracle','jacobi_tight') and base=='secondary_curvature_regularization.f90' and 'tolerance=1.d-10*scale' in target.read_text():
            text=target.read_text()
            target.write_text(audit.replace_once(text,'tolerance=1.d-10*scale+1.d-14*sqrt(sum(area*area))',
                                                'tolerance=1.d-13*scale+1.d-16*sqrt(sum(area*area))'))
        hashes[name]=r.sha256(target);command.append(str(target))
    exe=runtime/'xbraids_candidate.exe';command += ['-o',str(exe)]
    started=time.perf_counter()
    result=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=240)
    (runtime/'build.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    record=dict(source_sha256=hashes,executable_sha256=r.sha256(exe),command=command,
        depth_factor=2.,constant_mf_limit=25,series_mf_limit=20,
        role='Arithmetic-preserving packed Jacobi curvature and identical-grain fast path; validation candidate',
        baseline_executable=str(baseline),baseline_sha256=r.sha256(baseline),
        baseline_build=prior,compile_wall_s=time.perf_counter()-started)
    record['variant']=variant
    r.write_json(runtime/'build.json',record)
    print('BUILD COMPLETE',flush=True)
    return baseline,exe,record


def execute(root,label,exe,source,threads,duration,start,checkpoint=None,timeout=1200,extra_env=None):
    folder=root/label;folder.mkdir(parents=True,exist_ok=False)
    for name in ('case.data.bin','analysis_geometry.npz','input_provenance.json'):
        shutil.copy2(source/name,folder/name)
    lines=(source/'case.data').read_text().splitlines()
    # Header version 37, locate its unique time/dt/output triple from metadata.
    if source==audit.SOURCE:old='2880 0.012500000000000001 24'
    else:
        meta=json.loads((source/'preparation.json').read_text()) if (source/'preparation.json').exists() else {}
        candidates=[line for line in lines if len(line.split())==3 and line.split()[1]=='0.012500000000000001']
        if len(candidates)!=1:raise ValueError('Nonunique time line')
        old=candidates[0]
    new=f'{start+duration:.17g} '+' '.join(old.split()[1:])
    deck=audit.replace_once('\n'.join(lines),old,new)
    (folder/'case.data').write_text(deck+'\n')
    command=[str(exe),'case.data']
    if checkpoint:command.append(str(checkpoint))
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
             BRAIDS_PROFILE='1',BRAIDS_PROGRESS_STEPS='2000')
    env.update(extra_env or {})
    initial_steps=0 if not checkpoint else audit.parse_profile((checkpoint.parent/'native.log').read_text())['steps']
    started=time.perf_counter()
    with (folder/'native.log').open('w') as log:
        result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    wall=time.perf_counter()-started
    result.check_returncode()
    profile=audit.parse_profile((folder/'native.log').read_text())
    row=dict(label=label,wall_s=wall,profile=profile,threads=threads,
             steps=profile['steps']-initial_steps,
             compute_s=profile['profile: measured run wall seconds']-profile['profile: result output wall seconds'],
             executable_sha256=r.sha256(exe),input_deck_sha256=r.sha256(folder/'case.data'),
             source_deck_sha256=r.sha256(source/'case.data'),
             input_checkpoint_sha256=r.sha256(checkpoint) if checkpoint else None,
             checkpoint_sha256=r.sha256(folder/'checkpoint.bin'),
             start_s=start,end_s=start+duration,command=command,
             environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY','BRAIDS_PROFILE',*(extra_env or {}))})
    r.write_json(folder/'execution.json',row)
    print(json.dumps(dict(label=label,wall_s=wall,compute_s=row['compute_s'],steps=row['steps'])),flush=True)
    return row


def compare(left,right):
    lf=r.shared.read_frames(r.shared.native_frames_path(left))
    rf=r.shared.read_frames(r.shared.native_frames_path(right))
    np.testing.assert_array_equal([f['time_s'] for f in lf],[f['time_s'] for f in rf])
    lq=np.array([f['q'] for f in lf]);rq=np.array([f['q'] for f in rf])
    area=np.load(left/'analysis_geometry.npz')['area']
    ledger_l=audit.read_ledger(left);ledger_r=audit.read_ledger(right)
    # q contains h, hu, hv, two suspended masses and two active/substrate pairs.
    lbed=lq[:,:,5:9].sum(axis=2)/r.BULK;rbed=rq[:,:,5:9].sum(axis=2)/r.BULK
    row=dict(depth_m=float(np.max(abs(lq[:,:,0]-rq[:,:,0]))),
        depth_rms_m=float(np.max(np.sqrt(np.sum(area*(lq[:,:,0]-rq[:,:,0])**2,axis=1)/area.sum()))),
        bed_rms_m=float(np.max(np.sqrt(np.sum(area*(lbed-rbed)**2,axis=1)/area.sum()))),
        velocity_m_s=float(np.max(abs(lq[:,:,1:3]/lq[:,:,:1]-rq[:,:,1:3]/rq[:,:,:1]))),
        bed_m=float(np.max(abs(lbed-rbed))),
        # Bed inventories are kg/m2, not metres: their physical difference is
        # already checked above after division by bulk density. Do not compare
        # a mixed-unit raw state maximum to a hydraulic tolerance.
        other_state=float(np.max(abs(lq[:,:,:5]-rq[:,:,:5]))),
        raw_state_maximum=float(np.max(abs(lq-rq))),
        boundary_ledger=float(np.max(abs(ledger_l-ledger_r))),
        minimum_depth_m=float(rq[:,:,0].min()),all_finite=bool(np.isfinite(rq).all()),
        checkpoint_bitwise_equal=r.sha256(left/'checkpoint.bin')==r.sha256(right/'checkpoint.bin'))
    # Native ledger: dimensions, then time + nb*nledger cumulative entries.
    nb,nledger=map(int,ledger_r[:2]);records=ledger_r[2:].reshape(-1,1+nb*nledger)
    delta=records[-1,1:].reshape(nb,nledger)-records[0,1:].reshape(nb,nledger)
    water=float(np.dot(area,rq[-1,:,0]-rq[0,:,0])-delta[:,0].sum())
    sediment=float(np.dot(area,(rq[-1,:,3:9]-rq[0,:,3:9]).sum(axis=1))-delta[:,1:].sum())
    row.update(water_increment_budget_m3=abs(water),sediment_increment_budget_kg=abs(sediment))
    ls=np.loadtxt(left/'secondary_flow.txt');rs=np.loadtxt(right/'secondary_flow.txt')
    row['secondary_intensity_m_s']=float(np.max(abs(ls[:,6]-rs[:,6])))
    row['secondary_budget']=float(np.max(abs(np.loadtxt(right/'secondary_flow_budget.txt',ndmin=2)[:,-1])))
    row['momentum_budget']=float(np.max(abs(np.loadtxt(right/'secondary_momentum_budget.txt',ndmin=2)[:,-2:])))
    row['passed']=all(row[k]<=v for k,v in LIMITS.items() if k!='minimum_speedup') and row['all_finite'] and row['minimum_depth_m']>0
    return row


def run(tag,stage,variant='both'):
    root=HERE/'runs'/tag;report=HERE/'reports'/tag
    if stage=='build':
        root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
        r.write_json(report/'plan.json',dict(limits=LIMITS,
            changes=['Arithmetic-preserving packed Jacobi PCG and geometry/workspace cache','closed-form identical-grain percentile'],
            retained=['operator','true residual check','raw initial guess',
                      'dt/CFL','MF25','SSPRK3','mesh and boundary forcing'],
            filter_relative_tolerance=1e-10,filter_absolute_area_scaled_tolerance=1e-14,
            acceptance_rationale='Final delivery preserves arithmetic and original tolerances; require bitwise-identical checkpoints in fine/large and longer regression in addition to the physical gates. IC/MIC alternatives are not selected.',
            large_mesh_dx=.053125,long_continuation_s=24,plugin_promotion=False,delft_changes=False))
        build(root,variant);return
    record=json.loads((root/'runtime/build.json').read_text())
    baseline=Path(record['baseline_executable']);fast=root/'runtime/xbraids_candidate.exe'
    assert r.sha256(baseline)==record['baseline_sha256'] and r.sha256(fast)==record['executable_sha256']
    if stage=='prepare':
        original=r.shared.verified_braids_build;writer=r.write_cut_case
        try:
            r.shared.verified_braids_build=lambda:(fast,record)
            r.write_cut_case=validation_writer(root/'runtime')
            r.prepare_native(root/'large_input',dx=.053125,end=.4,output=.1,dt=.0125,morfac=25,hydraulic=False)
        finally:r.shared.verified_braids_build=original;r.write_cut_case=writer
        print('LARGE INPUT READY',flush=True);return
    if stage=='screen':
        rows=[];checks={}
        for label,exe in [('baseline',baseline),('optimized',fast)]:
            rows.append(execute(root,f'screen_{label}',exe,audit.SOURCE,8,1,2880,audit.SOURCE/'checkpoint.bin'))
        checks['fine']=compare(root/'screen_baseline',root/'screen_optimized')
        r.write_json(report/'screen.json',dict(rows=rows,checks=checks))
        print(json.dumps(checks),flush=True)
        if not all(x['passed'] and x['checkpoint_bitwise_equal'] for x in checks.values()):raise RuntimeError('Exact accuracy screen failed')
        return
    if stage=='benchmark':
        screen=json.loads((report/'screen.json').read_text());assert screen['checks']['fine']['passed']
        rows=[];checks={}
        for size,source,start,duration,checkpoint,threads in [
            ('fine',audit.SOURCE,2880,2,audit.SOURCE/'checkpoint.bin',8),
            ('large',root/'large_input',0,.4,None,16)]:
            for repeat,order in enumerate((('baseline','optimized'),('optimized','baseline')),1):
                for name in order:
                    rows.append(execute(root,f'{size}_r{repeat}_{name}',baseline if name=='baseline' else fast,
                                        source,threads,duration,start,checkpoint))
                checks[f'{size}_r{repeat}']=compare(root/f'{size}_r{repeat}_baseline',root/f'{size}_r{repeat}_optimized')
                r.write_json(report/'benchmark.json',dict(rows=rows,checks=checks))
                if not (checks[f'{size}_r{repeat}']['passed'] and checks[f'{size}_r{repeat}']['checkpoint_bitwise_equal']):
                    raise RuntimeError('Benchmark exact numerical gate failed')
        return
    if stage=='regression':
        # Reuse the preserved original-algorithm trajectory, rather than rerun
        # an identical expensive reference. This is an accuracy test, not timing.
        shutil.copytree(HERE/'runs/performance_optimized_20260930_v2/long_baseline',root/'long_baseline')
        rows=[json.loads((root/'long_baseline/execution.json').read_text()),
              execute(root,'long_optimized',fast,audit.SOURCE,8,24,2880,audit.SOURCE/'checkpoint.bin')]
        check=compare(root/'long_baseline',root/'long_optimized')
        historical=compare(HERE/'runs/performance_optimized_20260930_v2/long_baseline',root/'long_optimized')
        r.write_json(report/'regression.json',dict(rows=rows,check=check,
            reference_kind='Preserved original Jacobi, unchanged 1e-10 relative / 1e-14 area-scaled tolerance',
            historical_looser_reference_difference=historical,
            timing_caveat='Preserved reference reused for accuracy; not a paired speed benchmark'))
        print(json.dumps(check),flush=True)
        if not check['passed'] or not check['checkpoint_bitwise_equal']:raise RuntimeError('Longer-run exact numerical gate failed')
        # Thread reproducibility and cold-workspace restart at a scheduled output knot.
        rows=[execute(root,'large_t1_optimized',fast,root/'large_input',1,.4,0)]
        rows += [execute(root,'large_split_a',fast,root/'large_input',16,.2,0)]
        rows += [execute(root,'large_split_b',fast,root/'large_input',16,.2,.2,root/'large_split_a/checkpoint.bin')]
        reference=root/'large_r1_optimized/checkpoint.bin'
        flags={name:r.sha256(root/name/'checkpoint.bin')==r.sha256(reference)
               for name in ('large_t1_optimized','large_split_b')}
        r.write_json(report/'reproducibility.json',dict(rows=rows,checks=flags))
        if not all(flags.values()):raise RuntimeError('Thread/restart bitwise gate failed')
        return
    raise ValueError(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build','prepare','screen','benchmark','regression'])
    parser.add_argument('--tag',default=TAG)
    parser.add_argument('--variant',choices=['both','grain_only','filter_only','oracle','jacobi_tight'],default='both')
    args=parser.parse_args();run(args.tag,args.stage,args.variant)
