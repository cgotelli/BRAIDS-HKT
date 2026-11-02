"""Bounded, isolated timestep experiments on the frozen alpha-81 baseline.

No production/plugin changes and no Delft3D writes. Research CFL is a recorded
post-restart override so the exact same checkpoint can seed all methods without
bypassing the original input-fingerprint checks. Research checkpoints must keep
their method/environment manifest; they are not production restarts.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np

from . import run_case as r
from . import profile_current as profile
from . import mf25_validation_runtime as mf25
from . import hydraulic_methods as methods
from . import qualify_hydraulic_cfl as physics
from tools import build_cutcell

TAG = 'larger_timesteps_20261001_v3'
OPTIONS = (('rk3_04','reference',.4),('rk3_08','reference',.8),
           ('rk3_09','reference',.9),('ssp43_08','ssp43',.8))
SOURCES = {
    'developed': r.HERE/'runs/fine_mf25_equilibrium20h_v1_braids',
    'recent': r.HERE/'runs/cold_optimized20h_20261001_v1_mobile_braids',
}
LIMITS = dict(depth_rms_m=.002, velocity_rms_m_s=.02, bed_rms_m=.001,
              bed_max_m=.005, bed_increment_relative_rms=.10,
              water_budget_m3=1e-8, sediment_budget_kg=1e-6,
              secondary_budget=1e-8, minimum_speedup=1.2)


def paths(tag=TAG):
    return r.HERE/'runs'/tag, r.HERE/'reports'/tag


def replace(text, old, new):
    if text.count(old)!=1: raise ValueError(f'Nonunique source anchor: {old[:120]}')
    return text.replace(old,new,1)


def transform(original, variant):
    assert variant in ('reference','ssp43')
    text=mf25.native_source(original)
    # Run settings are deliberately applied AFTER fingerprint braids_support.
    anchor=' ! Qualified same-equation fixed-grid hydraulic workflow. Resolve after restart'
    setting=""" ! Recorded research setting; not a production restart contract.
 call get_environment_variable('BRAIDS_RESEARCH_CFL',profile_setting,status=environment_status)
 if(environment_status==0.and.len_trim(profile_setting)>0) then
  read(profile_setting,*,iostat=ios) cfl_target
  if(ios/=0.or..not.ieee_is_finite(cfl_target).or.cfl_target<=0.or.cfl_target>.9d0) &
   error stop 'research Euler CFL must be within (0,0.9]'
 endif
 write(*,'(a,f8.4)') 'research: Euler-stage CFL target=',cfl_target
"""
    if variant=='ssp43':
        setting+=""" if(amr_enabled.or.bank_failure.or.bank_avalanche.or.general_avalanche.or. &
  bedform_roughness.or.bedload_adaptation.or.exchange_mode/=0.or. &
  nmorfac>0) &
   error stop 'SSP43 research requires fixed mesh, no banks, exchange, settling, adaptation or MF series'
 write(*,'(a)') 'research: SSPRK(4,3); outer-step CFL bound is twice the Euler-stage target'
"""
    text=replace(text,anchor,setting+anchor)
    if variant=='reference':return text
    for name in ('advance','advance_prognostic_secondary'):
        old=methods.routine(text,name);new=old
        if name=='advance':
            new=replace(new,'  integer :: target','  real(8) :: research_momentum(2,2)\n  integer :: target')
            initial='q';momentum='research_momentum'
        else:initial='a';momentum='momentum_a'
        begin=new.index('  step_morfac=morphological_factor_at(time)')
        end=new.index('  call rebalance_active_layer(c)',begin)
        new=new[:begin]+f'''  step_morfac=morphological_factor_at(time)
  call research_ssp43({initial},dt,step_morfac,c,fa,{momentum},ok)
  if(.not.ok) return
'''+new[end:]
        new=replace(new,'  ledger=ledger+(fa+fb)/6.d0+2.d0*fc/3.d0','  ledger=ledger+fa')
        if name=='advance_prognostic_secondary':
            new=replace(new,'  secondary_momentum_boundary_trial=(momentum_a+momentum_b)/6.d0+2.d0*momentum_c/3.d0',
                '  secondary_momentum_boundary_trial=momentum_a')
        text=replace(text,old,new)
    helper=(r.HERE/'native/ssp43_step.inc').read_text()
    text=replace(text,' subroutine profile_report()',helper+'\n subroutine profile_report()')
    # All finite-volume kernels, closures and the controller remain untouched.
    for name in ('euler','wave_flux','source','face_value','secondary_relax_state',
                 'prepare_bedload_capacity','coupled_mpm_face','coupled_stage_characteristic'):
        assert methods.routine(text,name)==methods.routine(original,name),name
    return text


def build(tag):
    root,report=paths(tag);root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    production=json.loads((r.ROOT/'solver/cut_cells/build/build.json').read_text())
    assert r.sha256(r.ROOT/'solver/cut_cells/build/xbraids_cutcell.exe')==production['executable_sha256']
    r.write_json(report/'production_before.json',production)
    r.write_json(report/'plan.json',dict(limits=LIMITS,options=OPTIONS,
        acceptance='Finite/positive states, conserved budgets, modest hydraulic and bed differences; no bitwise requirement',
        scope='Fixed-mesh hydraulics and existing MF25 bend; not general AMR/bank qualification',
        controller='Unchanged Euler CFL check applied to dt for SSP33 and dt/2 for SSP43',
        cfl_definition='dt_stage * sum(face_length*face_wave_speed) / cell_area; not stock GeoClaw directional CFL',
        production_promoted=False,delft_modified=False))
    for variant in ('reference','ssp43'):
        work=root/('runtime_'+variant);work.mkdir();hashes={}
        for name in build_cutcell.SOURCES:
            source=r.ROOT/name;assert r.sha256(source)==production['source_sha256'][name]
            target=work/Path(name).name
            if name==build_cutcell.SOURCES[-1]: target.write_text(transform(source.read_text(),variant))
            else:shutil.copy2(source,target)
            hashes[name]=r.sha256(target)
        exe=work/'xbraids_candidate.exe'
        command=[shutil.which('gfortran') or r'C:\Strawberry\c\bin\gfortran.exe',
            '-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
            *[str(work/Path(s).name) for s in build_cutcell.SOURCES],'-o',str(exe)]
        started=time.perf_counter();result=subprocess.run(command,cwd=work,text=True,capture_output=True,timeout=180)
        (work/'build.log').write_text(result.stdout+result.stderr);result.check_returncode()
        record=dict(variant=variant,source_sha256=hashes,executable_sha256=r.sha256(exe),command=command,
            compile_wall_s=time.perf_counter()-started,parent_executable_sha256=production['executable_sha256'],
            guard_only_mf25=True,research_cfl_override=True,production_promoted=False,delft_modified=False)
        r.write_json(work/'build.json',record);print('BUILT '+variant,flush=True)


def runtime(root,variant):
    work=root/('runtime_'+variant);record=json.loads((work/'build.json').read_text());exe=work/'xbraids_candidate.exe'
    assert r.sha256(exe)==record['executable_sha256']
    for name,value in record['source_sha256'].items():assert r.sha256(work/Path(name).name)==value
    return exe


def execute(folder,exe,cfl,*,checkpoint=None,initial_steps=0,threads=8,timeout=300):
    command=[str(exe),'case.data']+([str(checkpoint)] if checkpoint else [])
    env=dict(os.environ,OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
             BRAIDS_PROFILE='1',BRAIDS_RESEARCH_CFL=str(cfl))
    started=time.perf_counter()
    with (folder/'native.log').open('w') as log:
        result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    elapsed=time.perf_counter()-started;log=(folder/'native.log').read_text();p=profile.parse_profile(log)
    calls=re.search(r'profile: Euler update wall seconds=.* calls=(\d+)',log)
    record=dict(label=folder.name,wall_s=elapsed,return_code=result.returncode,profile=p,
        steps=p.get('steps',initial_steps)-initial_steps,euler_calls=int(calls[1]) if calls else None,
        compute_s=p.get('profile: measured run wall seconds',elapsed)-p.get('profile: result output wall seconds',0),
        command=command,executable_sha256=r.sha256(exe),case_sha256=r.sha256(folder/'case.data'),
        input_checkpoint_sha256=r.sha256(checkpoint) if checkpoint else None,
        environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_DYNAMIC','OMP_WAIT_POLICY','BRAIDS_RESEARCH_CFL')})
    r.write_json(folder/'execution.json',record)
    print(json.dumps({k:record[k] for k in ('label','return_code','wall_s','steps','euler_calls')}),flush=True)
    if result.returncode: raise RuntimeError(log[-2500:])
    return record


def bend_run(root,label,variant,cfl,source,duration,threads=8):
    folder=root/label;folder.mkdir(exist_ok=False)
    for name in ('case.data.bin','analysis_geometry.npz','input_provenance.json'):
        shutil.copy2(source/name,folder/name)
    start,initial_steps=(source/'checkpoint_status.txt').read_text().split();start=float(start);initial_steps=int(initial_steps)
    deck=(source/'case.data').read_text();times=[line for line in deck.splitlines()
        if len(line.split())==3 and line.split()[1]=='0.012500000000000001']
    assert len(times)==1
    deck=replace(deck,times[0],f'{start+duration:.17g} '+' '.join(times[0].split()[1:]))
    (folder/'case.data').write_text(deck)
    meta=json.loads((source/'input_provenance.json').read_text())
    meta.update(end=start+duration,start_s=start,cfl=cfl,
        source_input_provenance_sha256=r.sha256(source/'input_provenance.json'),
        seed_checkpoint=str(source/'checkpoint.bin'),method=variant,
        cfl_override='Research runtime setting after native fingerprint verification')
    meta['files']['case.data']=r.sha256(folder/'case.data')
    r.write_json(folder/'input_provenance.json',meta)
    r.write_json(folder/'input_changes.json',dict(source=str(source),start_s=start,end_s=start+duration,
        changes={'end_time_s':start+duration,'runtime_Euler_CFL':cfl},
        unchanged=['mesh','initial checkpoint','forcing','sediment','MF25','output schedule','dtmax'],
        method=variant,checkpoint_policy='Original fingerprint verified before documented runtime CFL override'))
    return execute(folder,runtime(root,variant),cfl,checkpoint=source/'checkpoint.bin',initial_steps=initial_steps,threads=threads)


def fields(folder):
    frames=r.shared.read_frames(r.shared.native_frames_path(folder));q=np.array([f['q'] for f in frames])
    return frames,q


def compare(left,right):
    lf,lq=fields(left);rf,rq=fields(right)
    np.testing.assert_array_equal([f['time_s'] for f in lf],[f['time_s'] for f in rf])
    np.testing.assert_array_equal(lq[0],rq[0])
    area=np.load(left/'analysis_geometry.npz')['area'];weights=area/area.sum()
    bed_l=lq[:,:,5:9].sum(axis=2)/r.BULK;bed_r=rq[:,:,5:9].sum(axis=2)/r.BULK
    rms=lambda x:float(np.sqrt(np.sum(weights*x*x,axis=-1)).max())
    u_l=lq[:,:,1:3]/np.maximum(lq[:,:,:1],1e-4);u_r=rq[:,:,1:3]/np.maximum(rq[:,:,:1],1e-4)
    inc_l=bed_l[-1]-bed_l[0];inc_r=bed_r[-1]-bed_r[0];active=abs(inc_l)>.0001
    ledger=profile.read_ledger(right);nb,nv=map(int,ledger[:2]);v=ledger[2:].reshape(-1,1+nb*nv)
    delta=(v[-1,1:]-v[0,1:]).reshape(nb,nv)
    check=dict(depth_rms_m=rms(lq[:,:,0]-rq[:,:,0]),velocity_rms_m_s=rms(np.linalg.norm(u_l-u_r,axis=2)),
        bed_rms_m=rms(bed_l-bed_r),bed_max_m=float(abs(bed_l-bed_r).max()),
        bed_increment_relative_rms=rms(inc_l-inc_r)/max(rms(inc_l),1e-6),
        active_pattern_area_m2=float(area[active].sum()),
        erosion_deposition_sign_agreement=float(np.average(np.sign(inc_l[active])==np.sign(inc_r[active]),weights=area[active])) if active.any() else None,
        water_budget_m3=abs(float(area@(rq[-1,:,0]-rq[0,:,0])-delta[:,0].sum())),
        sediment_budget_kg=abs(float(area@(rq[-1,:,3:9]-rq[0,:,3:9]).sum(axis=1)-delta[:,1:].sum())),
        secondary_budget=float(abs(np.loadtxt(right/'secondary_flow_budget.txt',ndmin=2)[:,-1]).max()),
        finite=bool(np.isfinite(rq).all()),min_depth_m=float(rq[:,:,0].min()),min_inventory=float(rq[:,:,3:].min()))
    check['gates']={key:check[key]<=limit for key,limit in LIMITS.items() if key!='minimum_speedup'}
    check['gates']['finite_positive']=check['finite'] and check['min_depth_m']>=0 and check['min_inventory']>=0
    check['passed']=all(check['gates'].values())
    return check


def bend_stage(tag,stage):
    root,report=paths(tag);rows=[];checks={}
    if stage!='screen':
        screen=json.loads((report/'screen.json').read_text())
        if not all(x['passed'] for x in screen['checks'].values()):
            raise RuntimeError('Complete and review the numerical screen first')
    if stage=='screen':cases=[('developed',2),('recent',2)];repeats=1;options=OPTIONS
    elif stage=='benchmark':cases=[('developed',6)];repeats=3;options=[o for o in OPTIONS if o[0]!='rk3_09']
    elif stage=='long':cases=[('developed',24),('recent',24)];repeats=1;options=[o for o in OPTIONS if o[0]!='rk3_09']
    else:raise ValueError(stage)
    for case,duration in cases:
        for repeat in range(repeats):
            order=list(options);order=order[repeat:]+order[:repeat]
            labels={}
            for option,variant,cfl in order:
                label=f'{stage}_{case}_r{repeat+1}_{option}';labels[option]=label
                rows.append(bend_run(root,label,variant,cfl,SOURCES[case],duration))
                r.write_json(report/(stage+'.json'),dict(rows=rows,checks=checks))
            for option,_,_ in options:
                if option=='rk3_04':continue
                checks[f'{case}_r{repeat+1}_{option}']=compare(root/labels['rk3_04'],root/labels[option])
            r.write_json(report/(stage+'.json'),dict(rows=rows,checks=checks))
            print(json.dumps(checks),flush=True)


def physics_stage(tag):
    root,report=paths(tag);rows=[]
    for kind in ('steep_wave','dry_dam_break','cutcell_lake'):
        mesh,q,bed,bc,end,truth,core,scale=physics.fixture(kind);area=np.array([c.area_m2 for c in mesh.cells]);fine=None
        for option,variant,cfl in [('small','reference',.2),*OPTIONS]:
            folder=root/f'physics_{kind}_{option}';folder.mkdir()
            r.write_cut_case(folder/'case.data',mesh,q,bed,bc,end_time_s=end,dt_max_s=1.,output_interval_s=end,
                cfl=.4,scheme=6,bedload=0,exchange=0,settling_velocity_m_s=0,dry_tolerance_m=.0001,manning=0,
                source_sha256={'test':'larger-timesteps-physics'})
            row=execute(folder,runtime(root,variant),cfl,threads=4)
            final=r.shared.read_frames(folder/'frames.txt')[-1];state=final['q']
            if fine is None:fine=state.copy()
            error=float(abs(state[core,0]-truth[core]).sum()/scale)
            diff=float(abs(state[core,0]-fine[core,0]).sum()/scale)
            residual=float(area@(state[:,0]-q[:,0])-final['ledger'][:,0].sum())
            row.update(case=kind,method=option,depth_relative_L1=error,vs_small_step_relative_L1=diff,
                water_residual_m3=residual,minimum_depth=float(state[:,0].min()),
                passed=bool(error<.02 and diff<.01 and abs(residual)<1e-8 and np.isfinite(state).all() and state[:,0].min()>=0))
            if kind=='cutcell_lake':row['passed'] &= bool(abs(state-q).max()<1e-10)
            rows.append(row);r.write_json(report/'physics.json',rows)


def thread_stage(tag):
    """Real bend thread control, kept outside the sequential timing matrix."""
    root,report=paths(tag);rows=[]
    for threads in (1,8):
        rows.append(bend_run(root,f'threads_developed_{threads}','ssp43',.8,
                             SOURCES['developed'],.25,threads=threads))
    one=root/rows[0]['label'];eight=root/rows[1]['label']
    same=r.sha256(one/'checkpoint.bin')==r.sha256(eight/'checkpoint.bin')
    result=dict(rows=rows,checkpoint_bitwise_equal=same,
                checks=compare(one,eight),scope='Numerical thread control, not a scaling benchmark')
    r.write_json(report/'threads.json',result)
    assert same and result['checks']['passed']


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','screen','benchmark','long','physics','threads']);p.add_argument('--tag',default=TAG)
    a=p.parse_args()
    if a.stage=='build':build(a.tag)
    elif a.stage=='physics':physics_stage(a.tag)
    elif a.stage=='threads':thread_stage(a.tag)
    else:bend_stage(a.tag,a.stage)


if __name__=='__main__':main()
