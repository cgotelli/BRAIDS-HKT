"""Validation-only SDIRK2/JFNK around the frozen augmented f-wave operator.

All production files and Delft3D are read-only. The runtime contains both the
frozen SSP43 reference and SDIRK2 for a same-executable timing comparison.
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
from . import larger_timesteps as p
from . import hydraulic_performance as h
from . import hybrid_hydraulics as fixtures

TAG = 'implicit_fwave_20261001_v5'
PARENT = p.paths()[0]/'runtime_ssp43'
HERE = p.r.HERE


def paths(tag=TAG):
    return p.paths(tag)


def transform(original):
    text = p.replace(original, ' implicit none\n integer :: nc,nf', ''' implicit none
 integer :: implicit_method=0,implicit_vector_parallel=1,implicit_jacobian_order=1
 integer(kind=8) :: implicit_rhs_calls=0,implicit_newton_iterations=0,implicit_krylov_iterations=0
 integer(kind=8) :: implicit_steps=0,implicit_fallback_steps=0,implicit_failures=0
 real(8) :: implicit_rhs_seconds=0,implicit_linear_seconds=0,implicit_fallback_time=0
 real(8) :: implicit_max_residual=0,implicit_max_cfl=0,implicit_last_rate=0
 real(8) :: implicit_tolerance=1.d-8,implicit_fd_scale=1.d-8
 integer :: nc,nf''')
    # Keep the complete explicit routine unchanged under a different name.
    advance = p.methods.routine(text, 'advance')
    explicit = advance.replace('subroutine advance(', 'subroutine research_explicit_advance(', 1)
    text = p.replace(text, advance, explicit)
    # Extract exactly the frozen preparation/reconstruction/face-flux prefix.
    # Stop BEFORE the dt-dependent draining limiter, CFL check and Euler update.
    euler = p.methods.routine(original, 'euler')
    rhs = euler[:euler.index('  ! Conservative water draining changes')]
    rhs = rhs.replace('subroutine euler(', 'subroutine research_spatial_rates(', 1)
    rhs += (HERE/'native/implicit_rates_tail.inc').read_text()
    helpers = (HERE/'native/implicit_fwave_step.inc').read_text()
    text = p.replace(text, ' subroutine profile_report()', rhs+'\n'+helpers+'\n subroutine profile_report()')
    marker = ' ! Qualified same-equation fixed-grid hydraulic workflow. Resolve after restart'
    settings = p.methods.GUARD + '''
 call get_environment_variable('BRAIDS_IMPLICIT_METHOD',profile_setting,status=environment_status)
 if(environment_status/=0) error stop 'implicit research requires an explicit method selection'
 read(profile_setting,*,iostat=ios) implicit_method
 if(ios/=0) error stop 'invalid implicit method'
 if(implicit_method/=0.and.implicit_method/=1) error stop 'implicit method must be 0 or 1'
 call get_environment_variable('BRAIDS_IMPLICIT_VECTOR_PARALLEL',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) implicit_vector_parallel
  if(ios/=0) error stop 'invalid vector parallel setting'
 endif
 if(implicit_method==1) then
  call get_environment_variable('BRAIDS_IMPLICIT_CFL',profile_setting,status=environment_status)
  if(environment_status/=0) error stop 'implicit research requires a recorded CFL target'
  read(profile_setting,*,iostat=ios) cfl_target
  if(ios/=0) error stop 'invalid implicit CFL'
  if(.not.ieee_is_finite(cfl_target).or.cfl_target<=0.or.cfl_target>32) error stop 'implicit CFL must be in (0,32]'
 endif
 call get_environment_variable('BRAIDS_IMPLICIT_TOL',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) implicit_tolerance
  if(ios/=0) error stop 'invalid Newton tolerance'
  if(.not.ieee_is_finite(implicit_tolerance).or.implicit_tolerance<1.d-12.or.implicit_tolerance>1.d-5) error stop 'Newton tolerance range'
 endif
 call get_environment_variable('BRAIDS_IMPLICIT_FD',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) implicit_fd_scale
  if(ios/=0) error stop 'invalid FD scale'
  if(.not.ieee_is_finite(implicit_fd_scale).or.implicit_fd_scale<1.d-11.or.implicit_fd_scale>1.d-5) error stop 'FD scale range'
 endif
 call get_environment_variable('BRAIDS_IMPLICIT_JACOBIAN_ORDER',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) implicit_jacobian_order
  if(ios/=0) error stop 'invalid Jacobian order'
  if(implicit_jacobian_order/=0.and.implicit_jacobian_order/=1) error stop 'Jacobian order must be 0 or 1'
 endif
 call get_environment_variable('BRAIDS_IMPLICIT_DTMAX',profile_setting,status=environment_status)
 if(environment_status==0) then
  read(profile_setting,*,iostat=ios) dtmax
  if(ios/=0) error stop 'invalid research dtmax'
  if(.not.ieee_is_finite(dtmax).or.dtmax<=0) error stop 'invalid research dtmax'
  step_hint=dtmax
 endif
 write(*,'(a,i0,a,f10.4,a,es12.4)') 'research: implicit method=',implicit_method,' CFL=',cfl_target,' dtmax=',dtmax
'''
    text = p.replace(text, marker, settings+marker)
    report = p.methods.routine(text, 'profile_report')
    added = report.replace('  if(.not.profile_enabled) return', '''  if(.not.profile_enabled) return
  write(*,'(a,i0)') 'implicit: rhs evaluations=',implicit_rhs_calls
  write(*,'(a,i0)') 'implicit: Newton iterations=',implicit_newton_iterations
  write(*,'(a,i0)') 'implicit: Krylov iterations=',implicit_krylov_iterations
  write(*,'(a,i0)') 'implicit: accepted steps=',implicit_steps
  write(*,'(a,i0)') 'implicit: explicit fallback steps=',implicit_fallback_steps
  write(*,'(a,i0)') 'implicit: failed attempts=',implicit_failures
  write(*,'(a,es22.14)') 'implicit: rhs wall seconds=',implicit_rhs_seconds
  write(*,'(a,es22.14)') 'implicit: linear wall seconds=',implicit_linear_seconds
  write(*,'(a,es22.14)') 'implicit: explicit fallback physical seconds=',implicit_fallback_time
  write(*,'(a,es22.14)') 'implicit: maximum scaled stage residual=',implicit_max_residual
  write(*,'(a,es22.14)') 'implicit: maximum accepted CFL=',implicit_max_cfl''')
    text = p.replace(text, report, added)
    for name in ('euler','wave_flux','source','face_value','traces','gradients','prepare_hydraulic_stage'):
        assert p.methods.routine(text,name) == p.methods.routine(original,name), name
    return text


def build(tag):
    root, report = paths(tag)
    root.mkdir(parents=True,exist_ok=False); report.mkdir(parents=True,exist_ok=False)
    parent=json.loads((PARENT/'build.json').read_text()); p.runtime(PARENT.parent,'ssp43')
    production=json.loads((p.r.ROOT/'solver/cut_cells/build/build.json').read_text())
    p.r.write_json(report/'production_before.json',production)
    folder=root/'runtime_switch'; folder.mkdir(); hashes={}
    for name in p.build_cutcell.SOURCES:
        source=PARENT/Path(name).name; target=folder/source.name
        assert p.r.sha256(source)==parent['source_sha256'][name]
        if name==p.build_cutcell.SOURCES[-1]: target.write_text(transform(source.read_text()))
        else: shutil.copy2(source,target)
        hashes[name]=p.r.sha256(target)
    exe=folder/'xbraids_candidate.exe'
    cmd=[shutil.which('gfortran') or r'C:\Strawberry\c\bin\gfortran.exe','-O3','-flto','-fopenmp',
         '-ffree-line-length-none','-ffixed-line-length-none',
         *[str(folder/Path(s).name) for s in p.build_cutcell.SOURCES],'-o',str(exe)]
    start=time.perf_counter(); result=subprocess.run(cmd,cwd=folder,text=True,capture_output=True,timeout=240)
    (folder/'build.log').write_text(result.stdout+result.stderr); result.check_returncode()
    p.r.write_json(folder/'build.json',dict(source_sha256=hashes,executable_sha256=p.r.sha256(exe),command=cmd,
        parent_executable_sha256=parent['executable_sha256'],compile_wall_s=time.perf_counter()-start,
        production_promoted=False,delft_modified=False))
    p.r.write_json(report/'plan.json',dict(method='SDIRK2 gamma=1-1/sqrt(2), Newton + restarted left-Jacobi GMRES',
        flux='Unmodified augmented GeoClaw f-wave and wrapper; frozen WENO and boundary closures',
        source='Manning drag evaluated in implicit stages with original work-depth formula',
        fallback='Moving wet/dry cells use entire frozen explicit SSP43 steps; counted, not implicit qualification',
        limits=dict(depth_rms_m=.002,velocity_rms_m_s=.02,water_budget_m3=1e-8,minimum_speedup=1.2),
        scope='Fixed bed and grid, no sediment, secondary flow, diffusion, banks, AMR',
        production_promoted=False,delft_modified=False))
    print('BUILT '+str(exe),flush=True)


def execute(root,folder,method,cfl=4.,threads=4,dtmax=.1,timeout=180,vector_parallel=1,tolerance=1e-8,fd=1e-8,jacobian_order=1,extra_environment=None):
    exe=p.runtime(root,'switch')
    settings=dict(OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_PROFILE='1',
        BRAIDS_RESEARCH_CFL='.8',BRAIDS_IMPLICIT_METHOD=str(method),BRAIDS_IMPLICIT_CFL=str(cfl),
        BRAIDS_IMPLICIT_DTMAX=str(dtmax),BRAIDS_IMPLICIT_VECTOR_PARALLEL=str(vector_parallel),BRAIDS_PROGRESS_STEPS='10',
        BRAIDS_IMPLICIT_TOL=str(tolerance),BRAIDS_IMPLICIT_FD=str(fd),BRAIDS_IMPLICIT_JACOBIAN_ORDER=str(jacobian_order))
    if extra_environment:
        settings.update({str(k):str(v) for k,v in extra_environment.items()})
    env=dict(os.environ,**settings); start=time.perf_counter(); timed_out=False
    with (folder/'native.log').open('w') as log:
        try: result=subprocess.run([str(exe),'case.data'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        except subprocess.TimeoutExpired: timed_out=True
    wall=time.perf_counter()-start; log=(folder/'native.log').read_text(); prof=p.profile.parse_profile(log)
    counters={key:float(value) for key,value in re.findall(r'^implicit: ([^=]+)=\s*([\d.Ee+\-]+)',log,re.M)}
    record=dict(label=folder.name,method=method,threads=threads,cfl=cfl,wall_s=wall,timed_out=timed_out,
        return_code=None if timed_out else result.returncode,profile=prof,counters=counters,
        steps=prof.get('steps'),compute_s=prof.get('profile: measured run wall seconds',wall)-prof.get('profile: result output wall seconds',0),
        environment=settings,executable_sha256=p.r.sha256(exe),input_sha256=p.r.sha256(folder/'case.data'),
        files={f.name:p.r.sha256(f) for f in folder.glob('case.data*')})
    p.r.write_json(folder/'execution.json',record)
    print(json.dumps({k:record[k] for k in ('label','return_code','wall_s','steps','counters')}),flush=True)
    return record


def fixture(kind):
    if kind!='hydraulic_jump': return fixtures.fixture(kind)
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.boundaries.cut_cells import map_cut_cells
    d=PolygonDomain(kind,[(0,0),(20,0),(20,.2),(0,.2)],'EPSG:2056')
    assignments=[BoundaryAssignment(d.edges[3].edge_id,'inlet','inlet'),BoundaryAssignment(d.edges[1].edge_id,'outlet','outlet')]
    mesh=map_cut_cells(d,assignments,NativeGrid(400,2,0,0,.05,.1))
    x=np.array([c.centroid_xy[0] for c in mesh.cells]); h1=.2; u1=3.; fr=u1/np.sqrt(9.81*h1)
    h2=.5*h1*(np.sqrt(1+8*fr*fr)-1); q=np.zeros((len(x),5)); q[:,0]=np.where(x<10,h1,h2); q[:,1]=h1*u1
    from braids_core.boundaries.cut_cell_io import CutBoundary
    bc=dict(inlet=CutBoundary('prescribed_depth_trace',discharge_m3_s=h1*u1*.2,depth_m=h1),outlet=CutBoundary('stage',stage_m=h2))
    return mesh,q,np.zeros(len(x)),bc,1.,q[:,0].copy(),np.ones(len(x),bool),float(q[:,0].sum())


def physics(tag):
    root,report=paths(tag)
    rows=json.loads((report/'physics.json').read_text()) if (report/'physics.json').exists() else []
    for kind in ('smooth_subcritical','steep_wave','dry_dam_break','cutcell_lake','hydraulic_jump'):
        mesh,q,bed,bc,end,truth,core,scale=fixture(kind); area=np.array([c.area_m2 for c in mesh.cells]); reference=None
        for name,method,cfl in [('reference',0,.8),('implicit2',1,2.),('implicit8',1,8.)]:
            folder=root/f'physics_{kind}_{name}'
            if any(row['label']==folder.name for row in rows):
                if method==0: reference=p.fields(folder)[1][-1].copy()
                continue
            folder.mkdir(exist_ok=True)
            p.r.write_cut_case(folder/'case.data',mesh,q,bed,bc,end_time_s=end,dt_max_s=1.,output_interval_s=end/4,
                cfl=.4,scheme=6,bedload=0,exchange=0,settling_velocity_m_s=0,dry_tolerance_m=.0001,manning=0,
                source_sha256={'test':'implicit-fwave-physics'})
            row=execute(root,folder,method,cfl,threads=1,dtmax=.05,timeout=180)
            if row['return_code']==0:
                frames,state=p.fields(folder); state=state[-1]
                if reference is None: reference=state.copy()
                row.update(analytic_depth_L1=float(abs(state[core,0]-truth[core]).sum()/scale),
                    reference_depth_L1=float(abs(state[core,0]-reference[core,0]).sum()/scale),
                    water_budget_m3=float(area@(state[:,0]-q[:,0])-frames[-1]['ledger'][:,0].sum()),
                    minimum_depth_m=float(state[:,0].min()),finite=bool(np.isfinite(state).all()),
                    maximum_lake_change=float(abs(state-q).max()) if kind=='cutcell_lake' else None)
                row['passed']=bool(row['analytic_depth_L1']<.02 and row['reference_depth_L1']<.01 and
                    abs(row['water_budget_m3'])<1e-8 and row['minimum_depth_m']>=0 and row['finite'])
                if kind=='cutcell_lake': row['passed'] &= row['maximum_lake_change']<1e-10
                np.savez(folder/'analytic.npz',truth=truth,core=core,initial=q,xy=np.array([c.centroid_xy for c in mesh.cells]))
            else: row['passed']=False
            row['case']=kind; rows.append(row); p.r.write_json(report/'physics.json',rows)


def bend(tag):
    root,report=paths(tag); rows=[]; checks={}
    for size,dx,end in [('small',.10625,.2),('large',.053125,.05)]:
        source=root/f'input_{size}'
        h.prepare_native(root,source,dx=dx,end=end,output=end/4,variant='switch')
        for name,method,cfl in [('reference',0,.8),('implicit2',1,2.),('implicit8',1,8.),('implicit16',1,16.)]:
            folder=root/f'bend_{size}_{name}'; h.clone(source,folder)
            row=execute(root,folder,method,cfl,threads=8,dtmax=.1,timeout=60);rows.append(row)
            if method and row['return_code']==0:
                check=h.native_comparison(root/f'bend_{size}_reference',folder)
                check['passed']=bool(check['depth_rms_m']<=.002 and check['velocity_rms_m_s']<=.02 and
                    check['water_residual_max_m3']<=1e-8 and check['minimum_depth_m']>=0)
                checks[f'{size}_{name}']=check
            p.r.write_json(report/'bend.json',dict(rows=rows,checks=checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','physics','bend']);parser.add_argument('--tag',default=TAG)
    args=parser.parse_args();globals()[args.action](args.tag)
