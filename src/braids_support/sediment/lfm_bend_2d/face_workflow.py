"""Frozen same-equation face/setup profiling and implementation experiments."""
from pathlib import Path
import argparse
import difflib
import json
import re
import shutil
import subprocess
import time
from . import workflow_adopted as prior

p=prior.p; h=p.h; m=p.m
TAG='face_workflow_20260930_v3'
PARENT=prior.paths()[0]/'runtime_adopted'


def paths():return p.paths(TAG)


def instrument(original,stride):
    """Sample worker elapsed time, not additive phase wall time.

    Rotate sampled face indices between stages. Instrument only the caller;
    the GeoClaw core and wave_flux wrapper stay byte-for-byte unchanged.
    """
    declarations='''
 real(8) :: detail_active_s=0,detail_primitive_s=0,detail_hydraulic_bc_s=0
 real(8) :: detail_face_total_s=0,detail_face_reconstruct_s=0,detail_face_wave_s=0
 integer(8) :: detail_face_count=0
'''
    text=original.replace('\ncontains\n',declarations+'contains\n',1)
    before=m.routine(text,'euler');after=before
    after=m.replace_once(after,'  ok=.false.;accepted=0;next=state;nwork=nf;ntarget=nc',
        '''  real(8) :: detail_tick,face_begin,face_tick,face_total,face_reconstruct,face_wave
  integer(8) :: face_count
  logical :: sampled_face
  ok=.false.;accepted=0;next=state;nwork=nf;ntarget=nc''')
    marker='  ! Recompute each RK stage: all wet cells, their dry neighbours, and every'
    after=m.replace_once(after,marker,'  detail_tick=omp_get_wtime()\n'+marker)
    marker='  call boundary_sediment_at(evaluation_time,bc_concentration,bc_bed_input)'
    after=m.replace_once(after,marker,'  detail_active_s=detail_active_s+omp_get_wtime()-detail_tick\n'+marker)
    marker='  call primitive(state,v,bed,suspended_active)'
    after=m.replace_once(after,marker,'  detail_tick=omp_get_wtime()\n'+marker+
        '\n  detail_primitive_s=detail_primitive_s+omp_get_wtime()-detail_tick\n  detail_tick=omp_get_wtime()')
    marker='  call boundary_hydraulic_at(evaluation_time,bed,bc_hydraulic,valid)'
    after=m.replace_once(after,marker,marker+'\n  detail_hydraulic_bc_s=detail_hydraulic_bc_s+omp_get_wtime()-detail_tick')
    start=after.index('  !$omp parallel do schedule(static,256) if(parallel_faces)',after.index('  fl=0;fr=0;speed=0'))
    end=after.index('  !$omp end parallel do',start)+len('  !$omp end parallel do')
    loop=after[start:end]
    loop=loop.replace('reduction(max:secondary_invalid)',
        'private(sampled_face,face_begin,face_tick) reduction(max:secondary_invalid) &\n'+
        '  !$omp& reduction(+:face_count,face_total,face_reconstruct,face_wave)',1)
    marker='   left=face_value(c,f,v,grad,vlo,vhi);depth=max(left(1)-left(bed_primitive_index()),0.d0)'
    loop=m.replace_once(loop,marker,f'''   sampled_face=mod(f+17*profile_euler_calls,{stride})==0
   if(sampled_face) then
    face_begin=omp_get_wtime();face_tick=face_begin;face_count=face_count+1
   endif
'''+marker+'\n   if(sampled_face) face_reconstruct=face_reconstruct+omp_get_wtime()-face_tick')
    marker='    right=face_value(n,f,v,grad,vlo,vhi);depth=max(right(1)-right(bed_primitive_index()),0.d0)'
    loop=m.replace_once(loop,marker,'    if(sampled_face) face_tick=omp_get_wtime()\n'+marker+
        '\n    if(sampled_face) face_reconstruct=face_reconstruct+omp_get_wtime()-face_tick')
    pattern=r'(?m)^( +)call wave_flux\([^\n]+&\n[^\n]+\)'
    def clock_wave(match):
        indent=match[1]
        return indent+'if(sampled_face) face_tick=omp_get_wtime()\n'+match[0]+'\n'+indent+'if(sampled_face) face_wave=face_wave+omp_get_wtime()-face_tick'
    loop,count=re.subn(pattern,clock_wave,loop);assert count==2
    marker='  enddo\n  !$omp end parallel do'
    loop=m.replace_once(loop,marker,'   if(sampled_face) face_total=face_total+omp_get_wtime()-face_begin\n'+marker)
    loop='  face_total=0;face_reconstruct=0;face_wave=0;face_count=0\n'+loop+'''
  detail_face_total_s=detail_face_total_s+face_total
  detail_face_reconstruct_s=detail_face_reconstruct_s+face_reconstruct
  detail_face_wave_s=detail_face_wave_s+face_wave
  detail_face_count=detail_face_count+face_count'''
    after=after[:start]+loop+after[end:]
    # This Windows libgomp build exposes millisecond omp_get_wtime ticks. That
    # is adequate for phase totals but not individual sub-microsecond faces.
    after=after.replace('omp_get_wtime()','detail_clock()')
    text=m.replace_once(text,before,after)
    clock='''
 real(8) function detail_clock() result(seconds)
  integer(8) :: tick,rate
  call system_clock(tick,rate)
  if(rate<=0) error stop 'high-resolution diagnostic clock unavailable'
  seconds=real(tick,8)/real(rate,8)
 end function
'''
    text=text.replace('\ncontains\n','\ncontains\n'+clock,1)
    marker="  write(*,'(a,f12.4)') 'profile: Euler face-flux wall seconds=',profile_euler_flux_s"
    records='\n'.join(f"  write(*,'(a,es20.10)') 'detail: {key}=',{value}" for key,value in (
        ('active mask wall seconds','detail_active_s'),('primitive wall seconds','detail_primitive_s'),
        ('hydraulic boundary wall seconds','detail_hydraulic_bc_s'),('sampled face worker seconds','detail_face_total_s'),
        ('sampled reconstruction worker seconds','detail_face_reconstruct_s'),('sampled wave worker seconds','detail_face_wave_s')))
    text=m.replace_once(text,marker,marker+'\n'+records+"\n  write(*,'(a,i0)') 'detail: sampled faces=',detail_face_count")
    return text


def transform(original,variant):
    if variant=='reference':return original
    if variant.startswith('sample'):text=instrument(original,int(variant[6:]))
    elif variant in ('cache','delivery'):
        text=cache_workspace(transform(original,'qualified') if variant=='delivery' else original)
    elif variant in ('prepare','friction','combined','extended','qualified'):
        text=original
        if variant in ('prepare','combined','extended'):text=fuse_inputs(text)
        if variant in ('friction','combined','extended','qualified'):
            before=m.routine(text,'source')
            after=m.replace_once(before,'if(nwork*nfrac>=8192) default(shared)',
                'if(nwork*nfrac>=8192.or.(hydraulic_workflow_fast.and.omp_get_max_threads()<=8)) default(shared)')
            text=m.replace_once(text,before,after)
        if variant=='qualified':
            # Preparation fusion showed no robust independent gain. Keep it
            # out; broaden trace fusion only to explicitly qualified physics.
            before=m.routine(text,'euler')
            after=m.replace_once(before,'''  if(hydraulic_workflow_fast) then
   downslope=0
   call prepare_hydraulic_traces_fused''','''  if(scheme==6.and.nc>=4096.and..not.amr_enabled.and. &
     .not.bank_front_tracking.and..not.bank_failure.and..not.bank_avalanche.and. &
     .not.general_avalanche.and.bank_erosion_fraction==0.d0.and. &
     .not.bedform_roughness.and..not.bedload_adaptation.and.exchange_mode==0.and. &
     (bedlaw==0.or.bedlaw==2)) then
   downslope=0
   if(bed_slope==2) call bed_downslopes(bed,downslope)
   call prepare_hydraulic_traces_fused''')
            text=m.replace_once(text,before,after)
        if variant=='extended':
            # Research expansion: preserve scalar/bed update guards, broaden
            # only the already numerically identical reconstruction preparation.
            before=m.routine(text,'euler')
            after=m.replace_once(before,'''  if(hydraulic_workflow_fast) then
   downslope=0
   call prepare_hydraulic_traces_fused''','''  if(scheme==6.and.nc>=4096.and..not.amr_enabled.and..not.bank_front_tracking) then
   downslope=0
   if(bed_slope==2) call bed_downslopes(bed,downslope)
   call prepare_hydraulic_traces_fused''')
            text=m.replace_once(text,before,after)
    else:raise ValueError(variant)
    protected=['wave_flux','advance','face_value','face_elevations','prepare_hydraulic_traces_fused']
    if variant not in ('friction','combined','extended','qualified','delivery'):protected.append('source')
    for name in protected:
        assert m.routine(text,name)==m.routine(original,name),name
    return text


def cache_workspace(original):
    """Euler is called serially by the stage/AMR drivers, never recursively.

    Only allocation lifetime changes. Shape changes invalidate the whole cache;
    all original per-stage initialization and arithmetic remain in place.
    """
    before=m.routine(original,'euler')
    start=before.index('  real(8) :: v(nprimitive,nc),bed(nc),grad(nprimitive,2,nc),downslope(2,nc)')
    end=before.index('  real(8) :: left(nprimitive)',start)
    declarations=before[start:end]
    arrays=re.findall(r'(\w+)\(([^()]*)\)',declarations.replace('real(8)','real'))
    assert len(arrays)==27,len(arrays)
    cached='  ! Reusable stage workspace: no recursive/concurrent Euler calls are permitted.\n'
    cached+='  ! AMR shape changes invalidate all buffers before any stage reads them.\n'
    for name,dims in arrays:
        cached+='  real(8),allocatable,save :: '+name+'('+','.join(':' for _ in dims.split(','))+')\n'
    cached+='  integer,save :: workspace_shape(6)=[-1,-1,-1,-1,-1,-1]\n'
    after=before[:start]+cached+before[end:]
    allocate=', &\n    '.join(name+'('+dims+')' for name,dims in arrays)
    names=', &\n     '.join(name for name,_ in arrays)
    ensure='''  if(any(workspace_shape/=[nprimitive,nstate,nc,nf,nfrac,nb])) then
   if(allocated(v)) then
    deallocate('''+names+''')
   endif
   allocate('''+allocate+''')
   workspace_shape=[nprimitive,nstate,nc,nf,nfrac,nb]
  endif
'''
    after=m.replace_once(after,'  ok=.false.;accepted=0;next=state;nwork=nf;ntarget=nc',
                         ensure+'  ok=.false.;accepted=0;next=state;nwork=nf;ntarget=nc')
    return m.replace_once(original,before,after)


def fuse_inputs(original):
    before=m.routine(original,'euler')
    start=before.index('  !$omp parallel do schedule(static,256) if(nc>=4096) default(shared) private(c,k,f,n)')
    end=before.index('  !$omp end parallel do',start)+len('  !$omp end parallel do')
    loop=before[start:end]
    body=loop[loop.index('   active_cell(c)='):loop.rindex('  enddo')]
    routine='''
 subroutine prepare_hydraulic_inputs(state,v,bed)
  ! Fixed-grid, one-fraction hydraulics only; selected by hydraulic_workflow_fast.
  ! Same raw-bed/primitive arithmetic; initialize each cell before its mask CYCLE.
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: v(nprimitive,nc),bed(nc)
  integer :: c,k,f,n
  !$omp parallel do schedule(static,256) default(shared) private(c,k,f,n)
  do c=1,nc
   bed(c)=rock(c)
   bed(c)=bed(c)+state(bed_index(1),c)/bulk_fraction(1)
   v(:,c)=0;v(1,c)=state(1,c)+bed(c);v(bed_primitive_index(),c)=bed(c)
   if(state(1,c)>dry_tolerance) v(2:3,c)=state(2:3,c)/state(1,c)
'''+body+'''  enddo
  !$omp end parallel do
 end subroutine
'''
    after=before[:start]+'''  if(hydraulic_workflow_fast) then
   call prepare_hydraulic_inputs(state,v,bed)
  else
'''+loop+'\n  endif'+before[end:]
    after=m.replace_once(after,'  call primitive(state,v,bed,suspended_active)',
        '  if(.not.hydraulic_workflow_fast) call primitive(state,v,bed,suspended_active)')
    text=m.replace_once(original,before,after)
    text=text.replace('\ncontains\n','\ncontains\n'+routine,1)
    return text


def build(variants):
    root,report=paths();root.mkdir(exist_ok=True);report.mkdir(exist_ok=True)
    parent=json.loads((PARENT/'build.json').read_text())
    if not (root/'runtime_reference').exists():shutil.copytree(PARENT,root/'runtime_reference')
    for variant in variants:
        if variant=='reference':continue
        folder=root/('runtime_'+variant);folder.mkdir(exist_ok=False);hashes={}
        for name in h.SOURCES:
            source=PARENT/Path(name).name;target=folder/source.name
            assert h.r.sha256(source)==parent['source_sha256'][name]
            if name==h.SOURCES[-1]:
                old=source.read_text();new=transform(old,variant);target.write_text(new)
                (folder/'candidate.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))))
            else:shutil.copy2(source,target)
            hashes[name]=h.r.sha256(target)
        command=[parent['command'][0],'-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
                 *[str(folder/Path(s).name) for s in h.SOURCES],'-o',str(folder/'xbraids_candidate.exe')]
        start=time.perf_counter();result=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
        (folder/'build.log').write_text(result.stdout+result.stderr);result.check_returncode()
        h.r.write_json(folder/'build.json',dict(command=command,source_sha256=hashes,variant=variant,
            executable_sha256=h.r.sha256(folder/'xbraids_candidate.exe'),parent_executable_sha256=parent['executable_sha256'],
            compile_wall_s=time.perf_counter()-start,builder_sha256=h.r.sha256(Path(__file__))))
        shutil.copy2(__file__,folder/'builder_snapshot.py');print('BUILT '+variant,flush=True)


def run(variants,*,label='diagnostic',repeats=1,cases=('small','large','bend')):
    root,report=paths();old,_=p.p.paths();rows=[]
    sources=dict(small=(old/'input_small_reference',8),large=(old/'input_large_reference',32),
                 bend=(p.p.PARENT/'input40_reference',8))
    for case in cases:
        source,threads=sources[case]
        for repeat in range(repeats):
            for variant in variants[repeat%len(variants):]+variants[:repeat%len(variants)]:
                row=h.execute_native(root,source,f'{label}_{case}_{variant}_r{repeat+1}',threads,variant)
                log=(root/row['label']/'native.log').read_text()
                row['detail']={k:float(v) for k,v in re.findall(r'^detail: (.*?)=\s*(\S+)',log,re.M)}
                rows.append(dict(row,case=case,repeat=repeat+1));h.r.write_json(report/(label+'.json'),rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('build','diagnostic'))
    parser.add_argument('--variants',nargs='+',default=['reference','sample64','sample256']);args=parser.parse_args()
    if args.stage=='build':build(tuple(args.variants))
    else:run(tuple(args.variants))
