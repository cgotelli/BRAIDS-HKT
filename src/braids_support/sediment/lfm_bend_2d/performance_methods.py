"""Next performance milestone: cached reconstruction and MIC-preconditioned PCG.

All candidate sources are frozen beside each executable. Delft3D is read-only.
The v4 exact-order runtime is the baseline, not the older slower executable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from . import run_case as r
from . import optimize_performance as old
from . import profile_current as audit

TAG = 'performance_methods_20260930_v2'
BASE = r.HERE / 'runs/performance_optimized_20260930_v4'
LIMITS = dict(depth_m=1e-4, bed_m=1e-4, depth_rms_m=1e-5, bed_rms_m=1e-5,
              velocity_m_s=1e-3, secondary_intensity_m_s=1e-4,
              water_increment_budget_m3=1e-8, sediment_increment_budget_kg=1e-6,
              secondary_budget=1e-8, momentum_budget=1e-8)


def routine(text, name):
    start = text.index(f' subroutine {name}(')
    end = text.index('\n end subroutine', start) + len('\n end subroutine')
    return text[start:end]


def cached_source(text):
    current = (r.ROOT / 'solver/cut_cells/cutcell_solver.f90').read_text()
    text = audit.replace_once(text, ' real(8), allocatable :: face_displacement(:,:,:)',
        ' real(8), allocatable :: face_displacement(:,:,:)\n'
        ' real(8), allocatable :: ls_delta(:,:),ls_weight(:,:),ls_matrix(:,:,:)')
    for name in ('gradients', 'cache_face_geometry', 'bedload_capacity_gradients', 'bed_downslopes', 'prepare_wet_dry_traces'):
        text = audit.replace_once(text, routine(text, name), routine(current, name))
    text = audit.replace_once(text, ' subroutine prepare_wet_dry_traces(',
        routine(current, 'face_elevations') + '\n\n subroutine prepare_wet_dry_traces(')
    return audit.replace_once(text, ' subroutine cache_one_face(c,f,side)',
        routine(current, 'cache_reconstruction_geometry') + '\n\n subroutine cache_one_face(c,f,side)')


def diagnostic_source(text):
    """Sparse, read-only CFL samples; excluded from timed builds."""
    text = audit.replace_once(text, ' logical :: profile_enabled=.false.',
        ' logical :: profile_enabled=.false.\n integer :: cfl_diagnostic_unit=0')
    anchor = '  if(stage_failure>0) then\n   if(profile_enabled) profile_euler_limit_s='
    text = audit.replace_once(text, anchor,
        '  if(profile_enabled.and.mod(profile_euler_calls,97)==0) &\n'
        '   call diagnose_cfl(state,speed,dt,evaluation_time)\n' + anchor)
    body = ''' subroutine diagnose_cfl(state,speed,dt,t)
  real(8),intent(in) :: state(nstate,nc),speed(nf),dt,t
  real(8) :: peak,rate,regular_peak,cut_peak,u,h,fr,frac
  integer :: c,k,f,winner,regular_winner
  logical :: cut
  peak=-1;regular_peak=0;cut_peak=0;winner=1;regular_winner=1
  do c=1,nc
   if(.not.active_cell(c).and.amr_step_level<0) cycle
   rate=0
   do k=offset(c),offset(c+1)-1
    f=faces(k);rate=rate+length(f)*speed(f)
   enddo
   rate=rate/area(c);cut=any(rectangle_width(:,c)<=0)
   if(cut) then
    cut_peak=max(cut_peak,rate)
   else if(rate>regular_peak) then
    regular_peak=rate;regular_winner=c
   endif
   if(rate>peak) then
    peak=rate;winner=c
   endif
  enddo
  c=winner;h=state(1,c);u=0;fr=0
  if(h>dry_tolerance) then
   u=sqrt(sum(state(2:3,c)**2))/h;fr=u/sqrt(grav*h)
  endif
  frac=area(c)/(dx*dy)
  if(cfl_diagnostic_unit==0) then
   open(newunit=cfl_diagnostic_unit,file='cfl_controllers.csv',status='replace')
   write(cfl_diagnostic_unit,'(a)') 'time_s,dt_s,cell,x_m,y_m,area_fraction,h_m,speed_m_s,Froude,rate_s_inv,regular_rate_s_inv,cut_rate_s_inv,regular_cell'
  endif
  write(cfl_diagnostic_unit,'(*(g0,:,","))') t,dt,c,xy(:,c),frac,h,u,fr,peak,regular_peak,cut_peak,regular_winner
  flush(cfl_diagnostic_unit)
 end subroutine

'''
    return audit.replace_once(text, ' subroutine wave_flux(', body + ' subroutine wave_flux(')


def build(root, variant):
    runtime = root / ('runtime_' + variant)
    runtime.mkdir(parents=True, exist_ok=False)
    prior = json.loads((BASE / 'runtime/build.json').read_text())
    baseline = BASE / 'runtime/xbraids_candidate.exe'
    assert r.sha256(baseline) == prior['executable_sha256']
    command = [prior['command'][0], '-O3', '-flto=8', '-fopenmp',
               '-ffree-line-length-none', '-ffixed-line-length-none']
    hashes = {}
    for name, expected in prior['source_sha256'].items():
        source = BASE / 'runtime' / Path(name).name
        assert r.sha256(source) == expected, source
        target = runtime / source.name
        text = source.read_text()
        if source.name == 'cutcell_solver.f90':
            if variant in ('geometry', 'combined'): text = cached_source(text)
            if variant == 'diagnostic': text = diagnostic_source(text)
        elif source.name == 'secondary_curvature_regularization.f90' and variant in ('mic', 'combined'):
            text = (r.HERE / 'native/secondary_curvature_fast.f90').read_text()
        target.write_text(text)
        hashes[name] = r.sha256(target)
        command.append(str(target))
    exe = runtime / 'xbraids_candidate.exe'
    command += ['-o', str(exe)]
    begin = time.perf_counter()
    proc = subprocess.run(command, cwd=runtime, capture_output=True, text=True, timeout=240)
    (runtime/'build.log').write_text(proc.stdout+proc.stderr)
    proc.check_returncode()
    record = dict(source_sha256=hashes, executable_sha256=r.sha256(exe), command=command,
        depth_factor=2., constant_mf_limit=25, series_mf_limit=20, variant=variant,
        baseline_executable=str(baseline), baseline_sha256=r.sha256(baseline),
        relative_tolerance=1e-13 if variant in ('mic', 'combined') else 1e-10,
        compile_wall_s=time.perf_counter()-begin, delft_modified=False)
    r.write_json(runtime/'build.json', record)
    print('BUILT '+variant, flush=True)


def executable(root, name):
    runtime = BASE/'runtime' if name == 'baseline' else root/('runtime_'+name)
    record = json.loads((runtime/'build.json').read_text())
    exe = runtime/'xbraids_candidate.exe'
    assert r.sha256(exe) == record['executable_sha256']
    for source, expected in record['source_sha256'].items():
        assert r.sha256(runtime/Path(source).name) == expected
    return exe


def compare(left, right, *, exact=False):
    check = old.compare(left, right)
    check['passed'] = all(check[k] <= limit for k, limit in LIMITS.items())
    check['passed'] &= check['all_finite'] and check['minimum_depth_m'] > 0
    if exact: check['passed'] &= check['checkpoint_bitwise_equal']
    return check


def run(tag, stage):
    root = r.HERE/'runs'/tag
    report = r.HERE/'reports'/tag
    if stage == 'build':
        root.mkdir(parents=True, exist_ok=False); report.mkdir(parents=True, exist_ok=False)
        r.write_json(report/'plan.json', dict(
            limits=LIMITS, baseline=str(BASE), delft_modified=False,
            physical_accuracy_rationale='Bed/depth max 0.1 mm and RMS 0.01 mm over 600 morphological seconds: below 0.78 mm grain size and 0.2 m flow depth; these are bounded-regression gates, not proof of experimental accuracy or long-time equilibrium.',
            retained=['SSPRK3','CFL 0.4','augmented f-wave Riemann solver','WENO5 and cut-cell fallback',
                      'MF25','mesh','boundaries','ell=2h operator','cold filter initial guess'],
            changes=['Static least-squares geometry cache','Only reconstruct eta and bed for wet/dry checks',
                     'MIC(0) preconditioning with a stricter true-residual target'],
            minimum_speedup=1.05, installed_plugin_updated=False))
        for variant in ('geometry','mic','combined','diagnostic'): build(root,variant)
        return
    if stage == 'screen':
        rows=[];checks={}
        for name in ('baseline','geometry','mic','combined','diagnostic'):
            rows.append(old.execute(root,'screen_'+name,executable(root,name),audit.SOURCE,
                                    8,1,2880,audit.SOURCE/'checkpoint.bin'))
            if name!='baseline':
                checks[name]=compare(root/'screen_baseline',root/('screen_'+name),exact=name in ('geometry','diagnostic'))
            r.write_json(report/'screen.json',dict(rows=rows,checks=checks))
        print(json.dumps(checks),flush=True)
        assert all(x['passed'] for x in checks.values())
        return
    if stage == 'prepare':
        runtime=root/'runtime_combined'
        record=json.loads((runtime/'build.json').read_text())
        original=r.shared.verified_braids_build;writer=r.write_cut_case
        try:
            r.shared.verified_braids_build=lambda:(executable(root,'combined'),record)
            r.write_cut_case=old.validation_writer(runtime)
            r.prepare_native(root/'larger_input',dx=.0265625,end=.1,output=.025,dt=.0125,morfac=25,hydraulic=False)
        finally:
            r.shared.verified_braids_build=original;r.write_cut_case=writer
        print('LARGER INPUT READY',flush=True)
        return
    if stage == 'benchmark':
        assert all(x['passed'] for x in json.loads((report/'screen.json').read_text())['checks'].values())
        rows=[];checks={}
        for size,source,start,duration,checkpoint,threads in [
            ('fine',audit.SOURCE,2880,2,audit.SOURCE/'checkpoint.bin',8),
            ('large',BASE/'large_input',0,.4,None,16),
            ('larger',root/'larger_input',0,.1,None,16)]:
            for rep,order in enumerate((('baseline','geometry','combined'),('combined','geometry','baseline')),1):
                for name in order:
                    rows.append(old.execute(root,f'{size}_r{rep}_{name}',executable(root,name),source,
                                            threads,duration,start,checkpoint))
                for name in ('geometry','combined'):
                    checks[f'{size}_r{rep}_{name}']=compare(root/f'{size}_r{rep}_baseline',root/f'{size}_r{rep}_{name}',exact=name=='geometry')
                r.write_json(report/'benchmark.json',dict(rows=rows,checks=checks))
                assert all(x['passed'] for x in checks.values())
        return
    if stage == 'regression':
        row=old.execute(root,'long_combined',executable(root,'combined'),audit.SOURCE,
                        8,24,2880,audit.SOURCE/'checkpoint.bin')
        check=compare(BASE/'long_optimized',root/'long_combined')
        r.write_json(report/'regression.json',dict(row=row,check=check,reference=str(BASE/'long_optimized')))
        print(json.dumps(check),flush=True)
        assert check['passed']
        rows=[]
        for label,threads,duration,start,checkpoint in [
            ('large_t1',1,.4,0,None),('large_split_a',16,.2,0,None),
            ('large_split_b',16,.2,.2,root/'large_split_a/checkpoint.bin')]:
            rows.append(old.execute(root,label,executable(root,'combined'),BASE/'large_input',
                                    threads,duration,start,checkpoint))
        sha=r.sha256(root/'large_r1_combined/checkpoint.bin')
        flags={name:r.sha256(root/name/'checkpoint.bin')==sha for name in ('large_t1','large_split_b')}
        r.write_json(report/'reproducibility.json',dict(rows=rows,checks=flags))
        assert all(flags.values())
        return
    raise ValueError(stage)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['build','prepare','screen','benchmark','regression'])
    p.add_argument('--tag',default=TAG)
    args=p.parse_args();run(args.tag,args.stage)
