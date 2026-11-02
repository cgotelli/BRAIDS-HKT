"""Frozen full-operator and coupled qualification of multilevel curvature PCG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from . import sediment_workflow as previous
from . import performance_speedfirst as physical

r=previous.r
TAG='curvature_multilevel_20260930_v2'
MODULE=r.HERE/'native/secondary_curvature_multilevel.f90'


def paths():return r.HERE/'runs'/TAG,r.HERE/'reports'/TAG


def setup():
    root,report=paths();root.mkdir(exist_ok=False);report.mkdir(exist_ok=False)
    old,_=previous.paths();previous.identity('reference')
    shutil.copytree(old/'runtime_reference',root/'runtime_reference')
    baseline=json.loads((root/'runtime_reference/build.json').read_text())
    r.write_json(report/'plan.json',dict(baseline=baseline,physical_operator='diag(area) + weighted graph Laplacian, ell=2h',
        relative_tolerance=1e-4,absolute_area_scaled_tolerance=1e-10,cold_initial_guess=True,
        unchanged=['GeoClaw','WENO','RK3','CFL','bedload laws','Exner update','boundary forcing','MF25'],
        physical_limits=physical.LIMITS,require_true_residual=True,require_thread_and_restart_identity=True,
        runtime_scope='Existing experimental curvature closure; no automatic production or plugin promotion',
        external_dependency=False,delft_modified=False,
        method_reference='https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html',
        method_note='Independent native prototype; symmetric Jacobi smoother for a PCG-compatible V cycle, not a hypre build.'))


CAPTURE='''program capture
 use cutcell_solver
 implicit none
 character(2048) :: deck,statefile
 real(8),allocatable :: raw(:)
 call get_command_argument(1,deck);call get_command_argument(2,statefile)
 call load_case(trim(deck))
 ! Curvature needs only depth and the two momentum components. Result frames
 ! omit the separately recorded prognostic secondary-flow state.
 open(91,file=trim(statefile),access='stream',form='unformatted',status='old');read(91) q(1:3,:);close(91)
 allocate(raw(nc));call secondary_flow_curvatures(q,raw)
 open(91,file='operator.bin',access='stream',form='unformatted',status='replace')
 write(91) nc,nf,owner,neighbour,area,xy,normal,length,q(1,:),raw,dry_tolerance
 close(91)
 write(*,*) 'CAPTURED',nc,nf,minval(raw),maxval(raw)
end program
'''

PROBE='''program probe
 use secondary_curvature_regularization
 use omp_lib,only:omp_get_wtime
 implicit none
 integer :: nc,nf,it,total,j,repeats
 integer,allocatable :: owner(:),neighbour(:)
 real(8),allocatable :: area(:),xy(:,:),normal(:,:),length(:),h(:),raw(:),out(:)
 real(8) :: dry,residual,start,cold,elapsed
 logical :: ok
 character(2048) :: input,argument
 call get_command_argument(1,input);call get_command_argument(2,argument);read(argument,*) repeats
 open(91,file=trim(input),access='stream',form='unformatted',status='old');read(91) nc,nf
 allocate(owner(nf),neighbour(nf),area(nc),xy(2,nc),normal(2,nf),length(nf),h(nc),raw(nc),out(nc))
 read(91) owner,neighbour,area,xy,normal,length,h,raw,dry;close(91)
 start=omp_get_wtime()
 call regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,h,dry,2.d0,raw,out,it,residual,ok, &
  relative_tolerance=1.d-4,absolute_tolerance=1.d-10)
 cold=omp_get_wtime()-start
 if(.not.ok) error stop 'Cold solve failed'
 total=0;start=omp_get_wtime()
 do j=1,repeats
  call regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,h,dry,2.d0,raw,out,it,residual,ok, &
   relative_tolerance=1.d-4,absolute_tolerance=1.d-10)
  if(.not.ok) error stop 'Repeated solve failed'
  total=total+it
 enddo
 elapsed=omp_get_wtime()-start
 write(*,*) 'FILTER_RESULT',nc,cold,elapsed,total,residual
 open(91,file='filtered.bin',access='stream',form='unformatted',status='replace');write(91) out;close(91)
end program
'''


def capture():
    root,report=paths();work=root/'capture';work.mkdir(exist_ok=False)
    frozen=root/'runtime_reference';build=json.loads((frozen/'build.json').read_text())
    (work/'module.f90').write_text((frozen/'cutcell_solver.f90').read_text().split('program run_cutcell')[0])
    (work/'driver.f90').write_text(CAPTURE)
    names=[str(frozen/Path(n).name) for n in build['source_sha256'] if Path(n).name!='cutcell_solver.f90']
    executable=work/'capture.exe'
    command=[build['command'][0],'-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
        *names,str(work/'module.f90'),str(work/'driver.f90'),'-o',str(executable)]
    proc=subprocess.run(command,cwd=work,capture_output=True,text=True,timeout=180)
    (work/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    r.write_json(work/'build.json',dict(command=command,executable_sha256=r.sha256(executable),
        parent=build,driver_sha256=r.sha256(work/'driver.f90'),module_sha256=r.sha256(work/'module.f90')))
    old,_=previous.paths();rows=[]
    for case,nt in (('fine',8),('large',16),('larger',16)):
        source=old/f'diffusion_screen_{case}_reference_t{nt}_r1';folder=root/('input_'+case);folder.mkdir()
        frames=r.shared.read_frames(r.shared.native_frames_path(source));q=frames[-1]['q']
        q[:,:3].astype('<f8').tofile(folder/'state.bin');shutil.copy2(source/'case.data.bin',folder/'case.data.bin')
        proc=subprocess.run([str(executable),str(source/'case.data'),str(folder/'state.bin')],cwd=folder,
            capture_output=True,text=True,timeout=90,env=dict(os.environ,OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE'))
        (folder/'capture.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
        row=dict(case=case,state_time_s=frames[-1]['time_s'],source=str(source),state_sha256=r.sha256(folder/'state.bin'),
            operator_sha256=r.sha256(folder/'operator.bin'),source_frames_sha256=r.sha256(r.shared.native_frames_path(source)),
            case_data_sha256=r.sha256(source/'case.data'),case_binary_sha256=r.sha256(source/'case.data.bin'),
            capture_executable_sha256=r.sha256(executable))
        rows.append(row);r.write_json(folder/'provenance.json',row)
    r.write_json(report/'capture.json',rows)


def build_probe(variant):
    root,report=paths();work=root/('probe_'+variant);work.mkdir(exist_ok=False)
    if variant=='reference':source=root/'runtime_reference/secondary_curvature_fast.f90'
    elif variant in ('multilevel','multilevel2'):source=MODULE
    elif variant=='hybrid':source=Path('secondary_curvature_hybrid.f90')
    else:raise ValueError(variant)
    module=work/source.name
    if variant=='hybrid':module.write_text(hybrid_source())
    else:shutil.copy2(source,module)
    (work/'driver.f90').write_text(PROBE)
    reference=json.loads((root/'runtime_reference/build.json').read_text());exe=work/'probe.exe'
    command=[reference['command'][0],'-O3','-flto','-fopenmp','-ffree-line-length-none',str(module),str(work/'driver.f90'),'-o',str(exe)]
    proc=subprocess.run(command,cwd=work,capture_output=True,text=True,timeout=60)
    (work/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    r.write_json(work/'build.json',dict(command=command,module_sha256=r.sha256(module),driver_sha256=r.sha256(work/'driver.f90'),
        executable_sha256=r.sha256(exe),variant=variant,module_name=module.name))
    print('BUILT '+variant,flush=True)


def hybrid_source():
    mic=(r.HERE/'native/secondary_curvature_fast.f90').read_text()
    mg=MODULE.read_text()
    mic=mic.replace('module secondary_curvature_regularization','module curvature_mic_backend',1)
    mg=mg.replace('module secondary_curvature_regularization','module curvature_multilevel_backend',1)
    mg=mg.replace(' use, intrinsic :: iso_fortran_env, only: int64',
        ' use, intrinsic :: iso_fortran_env, only: int64\n use omp_lib, only: omp_get_max_threads')
    mg=mg.replace(' integer,save :: nlevels=0',' integer,save :: nlevels=0,team_threads=1')
    mg=mg.replace('!$omp parallel do schedule(static)', '!$omp parallel do schedule(static) num_threads(team_threads)')
    mg=mg.replace('  call prepare_workspace(nc,nf,owner,neighbour,area,xy,normal,length,ok)',
        '  team_threads=min(omp_get_max_threads(),merge(4,merge(8,16,nc<524288),nc<131072))\n'
        '  call prepare_workspace(nc,nf,owner,neighbour,area,xy,normal,length,ok)')
    return mic+'\n'+mg+'\n'+WRAPPER


WRAPPER='''! Measured cell-count dispatch, independent of OpenMP team count.
! MIC remains faster on the 5k/20k controls. Multilevel starts at 65,536 cells.
! Fine-grid equations and tolerance do not change at this crossover.
module secondary_curvature_regularization
 use curvature_mic_backend,only:mic_solve=>regularize_curvature,mic_reset=>reset_curvature_workspace
 use curvature_multilevel_backend,only:mg_solve=>regularize_curvature,mg_reset=>reset_curvature_workspace
 implicit none
 private
 public :: regularize_curvature,reset_curvature_workspace
 integer,save :: selected=0
contains
 subroutine reset_curvature_workspace()
  call mic_reset();call mg_reset();selected=0
 end subroutine
 subroutine regularize_curvature(nc,nf,owner,neighbour,area,xy,normal,length,h,dry, &
                                 depth_factor,raw,filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  integer,intent(in) :: nc,nf,owner(nf),neighbour(nf)
  real(8),intent(in) :: area(nc),xy(2,nc),normal(2,nf),length(nf),h(nc),dry,depth_factor,raw(nc)
  real(8),intent(out) :: filtered(nc),residual
  integer,intent(out) :: iterations
  logical,intent(out) :: ok
  real(8),intent(in),optional :: relative_tolerance,absolute_tolerance
  integer :: wanted
  wanted=1
  if(nc>=65536) wanted=2
  if(selected/=wanted) then
   call reset_curvature_workspace();selected=wanted
  endif
  if(wanted==1) then
   call mic_solve(nc,nf,owner,neighbour,area,xy,normal,length,h,dry,depth_factor,raw, &
    filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  else
   call mg_solve(nc,nf,owner,neighbour,area,xy,normal,length,h,dry,depth_factor,raw, &
    filtered,iterations,residual,ok,relative_tolerance,absolute_tolerance)
  endif
 end subroutine
end module
'''


def build_runtime(variant):
    root,report=paths();folder=root/('runtime_'+variant);folder.mkdir(exist_ok=False)
    parent=root/'runtime_reference';manifest=json.loads((parent/'build.json').read_text());hashes={}
    for name,expected in manifest['source_sha256'].items():
        src=parent/Path(name).name;assert r.sha256(src)==expected
        dest=folder/src.name
        if src.name=='secondary_curvature_fast.f90':
            dest.write_text(hybrid_source() if variant=='hybrid' else MODULE.read_text())
        else:shutil.copy2(src,dest)
        hashes[name]=r.sha256(dest)
    exe=folder/'xbraids_candidate.exe'
    command=[manifest['command'][0],'-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
        *[str(folder/Path(n).name) for n in hashes],'-o',str(exe)]
    start=time.perf_counter();proc=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=180)
    (folder/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    record=dict(manifest,variant=variant,command=command,source_sha256=hashes,executable_sha256=r.sha256(exe),
        compile_wall_s=time.perf_counter()-start,solver='MIC below 65536 cells; symmetric aggregation V-cycle PCG above',
        multilevel_team_limit='4 below 131072 cells; 8 below 524288; 16 above; respects lower requested thread counts',
        baseline_executable_sha256=manifest['executable_sha256'],production_source_changed=False)
    r.write_json(folder/'build.json',record);print('BUILT RUNTIME '+variant,flush=True)


def runtime(variant):
    root,_=paths();folder=root/('runtime_'+variant);record=json.loads((folder/'build.json').read_text())
    exe=folder/'xbraids_candidate.exe';assert r.sha256(exe)==record['executable_sha256']
    for name,value in record['source_sha256'].items():assert r.sha256(folder/Path(name).name)==value
    return exe,record


def coupled(variants,threads,cases,repeats,label,scale):
    root,report=paths();path=report/(label+'.json');assert not path.exists();rows=[]
    for case,source,start,duration,checkpoint,default_threads in previous.aux.CASES:
        if case not in cases:continue
        for repeat in range(repeats):
            for nt in threads or [default_threads]:
                order=variants if repeat%2==0 else list(reversed(variants))
                for variant in order:
                    exe,_=runtime(variant)
                    row=previous.old.execute(root,f'{label}_{case}_{variant}_t{nt}_r{repeat+1}',exe,source,nt,
                        duration*scale,start,checkpoint)
                    row.update(case=case,variant=variant,repeat=repeat+1)
                    rows.append(row);r.write_json(path,rows)


def reproducibility():
    root,report=paths();source=previous.aux.CASES[2][1];prepared=root/'restart_input';prepared.mkdir(exist_ok=False)
    for name in ('case.data.bin','analysis_geometry.npz','input_provenance.json'):
        shutil.copy2(source/name,prepared/name)
    lines=(source/'case.data').read_text().splitlines()
    old=next(line for line in lines if len(line.split())==3 and line.split()[1]=='0.012500000000000001')
    new='0.05 0.012500000000000001 0.025'
    (prepared/'case.data').write_text(previous.audit.replace_once('\n'.join(lines),old,new)+'\n')
    r.write_json(prepared/'changes.json',dict(source=str(source),old_time_line=old,new_time_line=new,
        reason='Identical scheduled output at the restart split; no off-step forced termination',
        source_sha256=r.sha256(source/'case.data'),input_sha256=r.sha256(prepared/'case.data')))
    exe,_=runtime('hybrid');rows=[]
    for name,nt,duration,start,checkpoint in [('whole_t1',1,.05,0,None),('whole_t16',16,.05,0,None),
            ('whole_t32',32,.05,0,None),('prefix',16,.025,0,None),
            ('resumed',16,.025,.025,root/'reproduce_prefix/checkpoint.bin')]:
        row=previous.old.execute(root,'reproduce_'+name,exe,prepared,nt,duration,start,checkpoint)
        rows.append(row);r.write_json(report/'reproducibility_runs.json',rows)
    reference=root/'reproduce_whole_t1'
    checks={name:r.sha256(root/('reproduce_'+name)/'checkpoint.bin')==r.sha256(reference/'checkpoint.bin')
        for name in ('whole_t16','whole_t32','resumed')}
    r.write_json(report/'reproducibility.json',dict(checks=checks,passed=all(checks.values()),rows=rows))
    assert all(checks.values()),checks


def read_operator(folder):
    with (folder/'operator.bin').open('rb') as f:
        nc,nf=np.fromfile(f,dtype='<i4',count=2)
        owner=np.fromfile(f,dtype='<i4',count=nf)-1;neighbour=np.fromfile(f,dtype='<i4',count=nf)-1
        area=np.fromfile(f,dtype='<f8',count=nc);xy=np.fromfile(f,dtype='<f8',count=2*nc).reshape(nc,2)
        normal=np.fromfile(f,dtype='<f8',count=2*nf).reshape(nf,2);length=np.fromfile(f,dtype='<f8',count=nf)
        h=np.fromfile(f,dtype='<f8',count=nc);raw=np.fromfile(f,dtype='<f8',count=nc);dry=np.fromfile(f,dtype='<f8',count=1)[0]
        assert not f.read(1)
    return dict(nc=int(nc),nf=int(nf),owner=owner,neighbour=neighbour,area=area,xy=xy,normal=normal,length=length,h=h,raw=raw,dry=dry)


def residual(data,result):
    c=data['owner'];n=data['neighbour'];valid=n>=0;c=c[valid];n=n[valid]
    h=data['h'];delta=data['xy'][n]-data['xy'][c]
    a,b=(2*h[c])**2,(2*h[n])**2
    weight=np.zeros(len(c));wet=np.minimum(h[c],h[n])>data['dry']
    weight[wet]=(2*a[wet]*b[wet]/(a[wet]+b[wet])*data['length'][valid][wet]
        *np.maximum(0,np.sum(delta[wet]*data['normal'][valid][wet],axis=1))/np.sum(delta[wet]**2,axis=1))
    rhs=data['area']*data['raw'];value=data['area']*result
    flux=weight*(result[c]-result[n]);np.add.at(value,c,flux);np.add.at(value,n,-flux)
    target=1e-4*np.linalg.norm(rhs)+1e-10*np.linalg.norm(data['area'])
    error=float(np.linalg.norm(rhs-value))
    assert np.isfinite(result).all() and error<=10*target,(error,target)
    return dict(true_residual=error,target=target,ratio=error/target)


def synthetic():
    root,report=paths()
    for tag,nx,ny in (('square64k',256,256),('square256k',512,512),('square1m',1024,1024)):
        folder=root/('input_'+tag);folder.mkdir(exist_ok=False)
        dx=.025;indices=np.arange(nx*ny).reshape(ny,nx)
        xy=np.column_stack([np.tile(np.arange(nx)+.5,ny),np.repeat(np.arange(ny)+.5,nx)])*dx
        c=np.r_[indices[:,:-1].ravel(),indices[:-1,:].ravel()];n=np.r_[indices[:,1:].ravel(),indices[1:,:].ravel()]
        normal=np.zeros((len(c),2));normal[:ny*(nx-1),0]=1;normal[ny*(nx-1):,1]=1
        area=np.full(nx*ny,dx*dx);length=np.full(len(c),dx)
        h=.2+.04*np.sin(.21*xy[:,0])*np.cos(.31*xy[:,1])
        raw=.05*np.sin(.3*xy[:,0])*np.cos(.4*xy[:,1])+.01*np.sin(1.2*xy[:,0])
        with (folder/'operator.bin').open('wb') as out:
            np.array([len(area),len(c)],dtype='<i4').tofile(out)
            for a in (c+1,n+1):a.astype('<i4').tofile(out)
            for a in (area,xy,normal,length,h,raw,[1e-4]):np.asarray(a,dtype='<f8').tofile(out)
        r.write_json(folder/'provenance.json',dict(synthetic=True,shape=[nx,ny],dx_m=dx,domain_m=[nx*dx,ny*dx],
            input_sha256=r.sha256(folder/'operator.bin'),purpose='Independent large-grid scaling controls, not laboratory data'))


def kernels(variants,threads,cases,repeats,label,calls):
    root,report=paths();path=report/(label+'.json');assert not path.exists();rows=[]
    for case in cases:
        data=read_operator(root/('input_'+case))
        for repeat in range(repeats):
            order=variants if repeat%2==0 else list(reversed(variants))
            for variant in order:
                work=root/('probe_'+variant);build=json.loads((work/'build.json').read_text());exe=work/'probe.exe'
                assert r.sha256(exe)==build['executable_sha256'] and r.sha256(work/build['module_name'])==build['module_sha256']
                for nt in (threads if variant!='reference' else [1]):
                    folder=root/f'{label}_{case}_{variant}_t{nt}_r{repeat+1}';folder.mkdir()
                    command=[str(exe),str(root/('input_'+case)/'operator.bin'),str(calls)]
                    env=dict(os.environ,OMP_NUM_THREADS=str(nt),OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
                    start=time.perf_counter();proc=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,timeout=120)
                    elapsed=time.perf_counter()-start;(folder/'native.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
                    fields=next(s.split() for s in proc.stdout.splitlines() if 'FILTER_RESULT' in s)
                    output=np.fromfile(folder/'filtered.bin',dtype='<f8');check=residual(data,output)
                    row=dict(label=folder.name,case=case,variant=variant,threads=nt,repeat=repeat+1,cells=int(fields[1]),
                        cold_s=float(fields[2]),kernel_s=float(fields[3]),iterations=int(fields[4]),reported_residual=float(fields[5]),
                        process_s=elapsed,calls=calls,result_sha256=r.sha256(folder/'filtered.bin'),check=check,
                        executable_sha256=r.sha256(exe),operator_sha256=r.sha256(root/('input_'+case)/'operator.bin'))
                    rows.append(row);r.write_json(folder/'execution.json',row);r.write_json(path,rows)
                    print(json.dumps({k:row[k] for k in ('case','variant','threads','kernel_s','iterations','cold_s')}),flush=True)


def memory():
    """Kernel memory cost, separate from the unmonitored timing campaign."""
    root,report=paths();rows=[];case='square1m'
    for variant,nt in (('reference',1),('hybrid',32)):
        work=root/('probe_'+variant);build=json.loads((work/'build.json').read_text());exe=work/'probe.exe'
        assert r.sha256(exe)==build['executable_sha256']
        assert r.sha256(work/build['module_name'])==build['module_sha256']
        folder=root/('memory_'+variant);folder.mkdir(exist_ok=False)
        command=[str(exe),str(root/('input_'+case)/'operator.bin'),'8']
        env=dict(os.environ,OMP_NUM_THREADS=str(nt),OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE')
        row=previous.hydraulic.measure(command,folder,env)
        check=residual(read_operator(root/('input_'+case)),np.fromfile(folder/'filtered.bin',dtype='<f8'))
        row.update(variant=variant,case=case,requested_threads=nt,command=command,check=check,
            executable_sha256=build['executable_sha256'],operator_sha256=r.sha256(root/('input_'+case)/'operator.bin'))
        rows.append(row);r.write_json(report/'memory.json',rows)
        print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('setup','capture','synthetic','build_probe','build_runtime','kernels','coupled','reproducibility','memory'))
    parser.add_argument('--variants',nargs='+',default=['reference','multilevel'])
    parser.add_argument('--threads',nargs='+',type=int);parser.add_argument('--cases',nargs='+',default=['fine','large','larger'])
    parser.add_argument('--repeats',type=int,default=1);parser.add_argument('--label',default='screen');parser.add_argument('--calls',type=int,default=16)
    parser.add_argument('--duration-scale',type=float,default=2.)
    args=parser.parse_args()
    if args.stage=='setup':setup()
    elif args.stage=='capture':capture()
    elif args.stage=='synthetic':synthetic()
    elif args.stage=='build_probe':
        for variant in args.variants:build_probe(variant)
    elif args.stage=='build_runtime':
        for variant in args.variants:build_runtime(variant)
    elif args.stage=='coupled':coupled(args.variants,args.threads,args.cases,args.repeats,args.label,args.duration_scale)
    elif args.stage=='reproducibility':reproducibility()
    elif args.stage=='memory':memory()
    else:kernels(args.variants,args.threads or [1,8,16,32],args.cases,args.repeats,args.label,args.calls)
