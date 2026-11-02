"""Current-baseline, same-equation sediment workflow experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from . import amr_workflow as hydraulic
from . import optimize_performance as old
from . import performance_auxiliary as aux
from . import regularization_candidate as reg
from . import profile_current as audit
from .mf25_validation_runtime import native_source as allow_validation_mf25

r=old.r
TAG='sediment_workflow_20260930_v2'


def paths():return r.HERE/'runs'/TAG,r.HERE/'reports'/TAG


def identity(variant):
    root,_=paths();folder=root/('runtime_'+variant)
    data=json.loads((folder/'build.json').read_text());exe=folder/'xbraids_candidate.exe'
    assert r.sha256(exe)==data['executable_sha256']
    for name,value in data['source_sha256'].items():assert r.sha256(folder/Path(name).name)==value
    return exe,data


def diagnostic(source):
    source=audit.replace_once(source,' logical :: profile_enabled=.false.',
        ' logical :: profile_enabled=.false.\n real(8) :: detail_secondary_s=0,detail_curvature_s=0,detail_filter_s=0\n integer :: detail_filter_calls=0,detail_filter_iterations=0')
    source=audit.replace_once(source,'  real(8) :: filter_raw(nc),filter_residual',
        '  real(8) :: filter_raw(nc),filter_residual,detail_begin,detail_phase')
    source=audit.replace_once(source,'  call secondary_flow_curvatures(state,secondary_curvature)\n  filter_raw=secondary_curvature',
        '  detail_begin=omp_get_wtime();detail_phase=detail_begin\n  call secondary_flow_curvatures(state,secondary_curvature)\n  detail_curvature_s=detail_curvature_s+omp_get_wtime()-detail_phase\n  detail_phase=omp_get_wtime()\n  filter_raw=secondary_curvature')
    source=audit.replace_once(source,"  if(.not.ok) error stop 'secondary curvature regularization failed'",
        "  detail_filter_s=detail_filter_s+omp_get_wtime()-detail_phase\n  detail_filter_calls=detail_filter_calls+1;detail_filter_iterations=detail_filter_iterations+filter_iterations\n  if(.not.ok) error stop 'secondary curvature regularization failed'")
    source=audit.replace_once(source,'   source_integral=source_integral+secondary_source_work(c)\n  enddo\n end subroutine',
        '   source_integral=source_integral+secondary_source_work(c)\n  enddo\n  detail_secondary_s=detail_secondary_s+omp_get_wtime()-detail_begin\n end subroutine')
    anchor="  write(*,'(a,f12.4)') 'profile: authoritative output wall seconds='"
    extra=''
    for key,variable in (('secondary relaxation','secondary'),('raw curvature','curvature'),('curvature filter','filter')):
        extra+=f"  write(*,'(a,f12.6)') 'diagnostic: {key} wall seconds=',detail_{variable}_s\n"
    for key,variable in (('calls','calls'),('iterations','iterations')):
        extra+=f"  write(*,'(a,i0)') 'diagnostic: curvature filter {key}=',detail_filter_{variable}\n"
    return audit.replace_once(source,anchor,extra+anchor)


def build(variant):
    if variant.startswith('production_'):return build_production(variant)
    root,report=paths();root.mkdir(exist_ok=True);report.mkdir(exist_ok=True)
    folder=root/('runtime_'+variant);folder.mkdir(exist_ok=False)
    manifest=json.loads((r.ROOT/'solver/cut_cells/build/build.json').read_text())
    if variant=='reference':
        for name,value in manifest['source_sha256'].items():assert r.sha256(r.ROOT/name)==value
        names=[*hydraulic.h.SOURCES[:-1],str(reg.FAST_MODULE.relative_to(r.ROOT)),hydraulic.h.SOURCES[-1]]
        original=(r.ROOT/hydraulic.h.SOURCES[-1]).read_text()
        driver=allow_validation_mf25(reg.candidate_source(original,2.,filter_rtol=1e-4))
        aux.hydraulic_guard(driver,original)
        compiler=manifest['command'][0]
        r.write_json(report/'baseline.json',dict(development_manifest=manifest,
            curvature_filter='Existing validation-only ell=2h MIC-PCG, rtol=1e-4, atol=1e-10',
            physical_formulas_changed=False,delft_modified=False,installed_plugin_updated=False,
            validation_only_input_allowance='Existing constant MF25 allowance; production and factor-series guard remain <=20',
            retained_v1_failure='First reference lacked the historical MF25 input allowance and stopped at input validation; no trajectory or performance result. Its manifest MF25 declaration was incorrect.'))
    else:
        _,parent=identity('reference');names=list(parent['source_sha256']);compiler=parent['command'][0]
        driver=(root/'runtime_reference/cutcell_solver.f90').read_text()
        if variant=='diagnostic':driver=diagnostic(driver)
        else:driver=transform(driver,variant)
    hashes={}
    for name in names:
        target=folder/Path(name).name
        if target.name=='cutcell_solver.f90':target.write_text(driver)
        else:shutil.copy2(r.ROOT/name if variant=='reference' else root/'runtime_reference'/target.name,target)
        hashes[name]=r.sha256(target)
    exe=folder/'xbraids_candidate.exe'
    command=[compiler,'-O3','-flto','-fopenmp','-ffree-line-length-none','-ffixed-line-length-none',
             *[str(folder/Path(n).name) for n in names],'-o',str(exe)]
    start=time.perf_counter();proc=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
    (folder/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
    r.write_json(folder/'build.json',dict(command=command,source_sha256=hashes,variant=variant,
        executable_sha256=r.sha256(exe),compile_wall_s=time.perf_counter()-start,
        hydraulic_baseline_sha256=manifest['executable_sha256'],depth_factor=2.,curvature_relative_tolerance=1e-4,
        curvature_absolute_area_scaled_tolerance=1e-10,constant_mf_limit=25,series_mf_limit=20))
    shutil.copy2(__file__,folder/'builder_snapshot.py');print('BUILT '+variant,flush=True)


def build_production(variant):
    """No experimental curvature filter or MF25 input override in these builds."""
    root,report=paths();folder=root/('runtime_'+variant);folder.mkdir(exist_ok=False)
    if variant=='production_reference':
        manifest=json.loads((r.ROOT/'solver/cut_cells/build/build.json').read_text())
        for name,value in manifest['source_sha256'].items():
            assert r.sha256(r.ROOT/name)==value
            shutil.copy2(r.ROOT/name,folder/Path(name).name)
        executable=r.ROOT/'solver/cut_cells/build/xbraids_cutcell.exe'
        assert r.sha256(executable)==manifest['executable_sha256']
        shutil.copy2(executable,folder/'xbraids_candidate.exe')
        manifest['frozen_without_recompiling']=True
    elif variant=='production_cached':
        _,manifest=identity('production_reference')
        for name in manifest['source_sha256']:
            shutil.copy2(root/'runtime_production_reference'/Path(name).name,folder/Path(name).name)
        driver=folder/'cutcell_solver.f90';driver.write_text(diffusion_source(driver.read_text()))
        executable=folder/'xbraids_candidate.exe'
        command=[manifest['command'][0],'-O3','-flto','-fopenmp','-ffree-line-length-none',
            '-ffixed-line-length-none',*[str(folder/Path(n).name) for n in manifest['source_sha256']],'-o',str(executable)]
        start=time.perf_counter();proc=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=240)
        (folder/'build.log').write_text(proc.stdout+proc.stderr);proc.check_returncode()
        manifest=dict(manifest,command=command,executable_sha256=r.sha256(executable),
            source_sha256={n:r.sha256(folder/Path(n).name) for n in manifest['source_sha256']},
            source_text_sha256={n:hashlib.sha256((folder/Path(n).name).read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')).hexdigest()
                for n in manifest['source_sha256']},
            compile_wall_s=time.perf_counter()-start,frozen_without_recompiling=False)
    else:raise ValueError(variant)
    manifest.update(variant=variant,experimental_filter=False,constant_mf_limit=20,series_mf_limit=20,
        installed_plugin_updated=False)
    r.write_json(folder/'build.json',manifest);shutil.copy2(__file__,folder/'builder_snapshot.py')
    print('BUILT '+variant,flush=True)


def transform(source,variant):
    if variant=='availability':return availability_source(source)
    if variant=='properties':return properties_source(source)
    if variant=='combined':return properties_source(availability_source(source))
    if variant=='capacity':return capacity_source(source)
    if variant=='all':return capacity_source(availability_source(source))
    if variant in ('diffusion','diffusion_cached'):return diffusion_source(source)
    raise ValueError(variant)


def properties_source(source):
    m=hydraulic.m
    hide=m.routine(source,'hiding_factors');mix=m.routine(source,'mixture_statistics')
    hide_body=hide[hide.index('   do g=1,nfrac'):hide.index('\n  enddo\n  !$omp end parallel do')]
    hide_body=hide_body.replace('factors(:,c)','hiding(:,c)')
    mix_body=mix[mix.index('   ! Bed-capacity closures'):mix.index('\n  enddo\n  !$omp end parallel do')]
    helper=''' subroutine prepare_sediment_properties(state,hiding,dg,dgsd,sand_fraction,d90_mix)
  ! The original hiding and mixture cell bodies share one immutable state and
  ! the same AMR evaluation set. Keep their defaults and arithmetic unchanged.
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: hiding(nfrac,nc),dg(nc),dgsd(nc),sand_fraction(nc),d90_mix(nc)
  real(8) :: active_volume(nfrac),mean_diameter,fraction_mass(nfrac),fraction_volume(nfrac)
  integer :: c,g,work,nwork,region
  logical :: valid
  hiding=1;dg=0;dgsd=1;sand_fraction=0;d90_mix=0;nwork=nc
  if(amr_step_level>=0) nwork=amr_level_eval_cell_count(amr_step_level)
  !$omp parallel do schedule(static,256) if(nwork*nfrac>=8192) default(shared) &
  !$omp& private(work,c,g,region,active_volume,mean_diameter,fraction_mass,fraction_volume,valid)
  do work=1,nwork
   c=work
   if(amr_step_level>=0) c=amr_level_eval_cells(work,amr_step_level)
   if(.not.active_cell(c)) cycle
   if(hiding_exposure/=0) then
    do region=1,1
'''+hide_body+'''
    enddo
   endif
'''+mix_body+'''
  enddo
  !$omp end parallel do
 end subroutine

'''
    source=audit.replace_once(source,' subroutine mixture_statistics(',helper+' subroutine mixture_statistics(')
    before='''   call hiding_factors(state,hiding)
   if(profile_enabled) then
    profile_euler_hiding_s=profile_euler_hiding_s+omp_get_wtime()-profile_detail_start
    profile_detail_start=omp_get_wtime()
   endif
   call mixture_statistics(state,mixture_dg,mixture_gsd,mixture_sand,mixture_d90)
   if(profile_enabled) profile_euler_mixture_s=profile_euler_mixture_s+omp_get_wtime()-profile_detail_start'''
    after='''   call prepare_sediment_properties(state,hiding,mixture_dg,mixture_gsd,mixture_sand,mixture_d90)
   if(profile_enabled) profile_euler_properties_s=profile_euler_properties_s+omp_get_wtime()-profile_detail_start'''
    source=audit.replace_once(source,before,after)
    source=audit.replace_once(source,' real(8) :: profile_euler_hiding_s=0.d0,profile_euler_mixture_s=0.d0',
        ' real(8) :: profile_euler_hiding_s=0.d0,profile_euler_mixture_s=0.d0\n real(8) :: profile_euler_properties_s=0.d0')
    source=audit.replace_once(source,'  profile_euler_hiding_s=0;profile_euler_mixture_s=0',
        '  profile_euler_hiding_s=0;profile_euler_mixture_s=0;profile_euler_properties_s=0')
    source=audit.replace_once(source,"  write(*,'(a,f12.4)') 'profile: Euler cell capacity wall seconds='",
        "  write(*,'(a,f12.4)') 'profile: Euler combined sediment properties wall seconds=',profile_euler_properties_s\n  write(*,'(a,f12.4)') 'profile: Euler cell capacity wall seconds='")
    return source


def capacity_source(source):
    m=hydraulic.m
    props=m.routine(properties_source(source),'prepare_sediment_properties')
    body=props[props.index('   if(hiding_exposure/=0)'):props.index('\n  enddo\n  !$omp end parallel do')]
    for old_name,new_name in (('sand_fraction(c)','mixture_sand(c)'),('d90_mix(c)','mixture_d90(c)'),
                              ('dgsd(c)','mixture_gsd(c)'),('dg(c)','mixture_dg(c)')):
        body=body.replace(old_name,new_name)
    original=m.routine(source,'prepare_bedload_capacity');new=original
    new=audit.replace_once(new,'subroutine prepare_bedload_capacity(', 'subroutine prepare_sediment_capacity(')
    new=audit.replace_once(new,'  real(8), intent(in) :: state(nstate,nc),hiding(nfrac,nc)',
        '  real(8), intent(in) :: state(nstate,nc)\n  real(8), intent(out) :: hiding(nfrac,nc)')
    new=audit.replace_once(new,'  real(8), intent(in) :: mixture_dg(nc)',
        '  real(8), intent(out) :: mixture_dg(nc)')
    new=audit.replace_once(new,'  real(8) :: profile_cell_start',
        '  real(8) :: profile_cell_start\n  real(8) :: active_volume(nfrac),mean_diameter,fraction_mass(nfrac),fraction_volume(nfrac)\n  integer :: region,properties_region')
    new=audit.replace_once(new,'  capacity=0;tau=0;gradient=0',
        '  capacity=0;tau=0;gradient=0\n  hiding=1;mixture_dg=0;mixture_gsd=1;mixture_sand=0;mixture_d90=0')
    new=audit.replace_once(new,'wet_fraction,xi) reduction(max:invalid)',
        'wet_fraction,xi,region,properties_region,active_volume,mean_diameter,fraction_mass,fraction_volume) reduction(max:invalid)')
    new=audit.replace_once(new,'   if(.not.active_cell(c)) cycle',
        '   if(.not.active_cell(c)) cycle\n   do properties_region=1,1\n'+body+'\n   enddo')
    new=new.replace('  ! Evaluate each nonlinear formula once at cell centres, as Delft3D does',
        '  ! Same hiding, mixture and capacity cell kernels in one parallel loop.\n  ! Evaluate each nonlinear formula once at cell centres, as Delft3D does')
    source=audit.replace_once(source,' subroutine prepare_bedload_capacity(',new+'\n\n subroutine prepare_bedload_capacity(')
    before='''   if(profile_enabled) profile_detail_start=omp_get_wtime()
   call hiding_factors(state,hiding)
   if(profile_enabled) then
    profile_euler_hiding_s=profile_euler_hiding_s+omp_get_wtime()-profile_detail_start
    profile_detail_start=omp_get_wtime()
   endif
   call mixture_statistics(state,mixture_dg,mixture_gsd,mixture_sand,mixture_d90)
   if(profile_enabled) profile_euler_mixture_s=profile_euler_mixture_s+omp_get_wtime()-profile_detail_start'''
    source=audit.replace_once(source,before,'''   call prepare_sediment_capacity(state,hiding,mixture_dg,mixture_gsd,mixture_sand,mixture_d90, &
    bc_bed_input,bedq_cell,bedtau_cell,bedq_grad)''')
    source=audit.replace_once(source,'''  if(bedlaw>0) call prepare_bedload_capacity(state,hiding,mixture_dg,mixture_gsd,mixture_sand,mixture_d90, &
   bc_bed_input,bedq_cell,bedtau_cell,bedq_grad)\n''','')
    return source


def diffusion_source(source):
    m=hydraulic.m
    assert '\ncontains\n' in source
    source=source.replace('\ncontains\n',
        '\n integer(8) :: diffusion_geometry_epoch=0\ncontains\n',1)
    original=m.routine(source,'cache_face_geometry')
    modified=audit.replace_once(original,'  call cache_sediment_geometry()',
        '  diffusion_geometry_epoch=diffusion_geometry_epoch+1\n  call cache_sediment_geometry()')
    source=audit.replace_once(source,original,modified)
    old=m.routine(source,'diffuse_momentum_nonorthogonal');new=old
    new=new.replace('  real(8),allocatable :: recovery_','  real(8),allocatable,save :: recovery_')
    new=new.replace('  integer,allocatable :: recovery_','  integer,allocatable,save :: recovery_')
    new=new.replace('  logical,allocatable :: boundary_cell','  logical,allocatable,save :: boundary_cell')
    new=audit.replace_once(new,'attempt,nrecovery,slot,bad_geometry', 'attempt,slot,bad_geometry')
    new=audit.replace_once(new,'  integer :: requested_recovery_faces,active_recovery_faces,active_affine_faces',
        '  integer,save :: nrecovery,requested_recovery_faces,active_recovery_faces,active_affine_faces')
    new=audit.replace_once(new,'  logical :: accepted', '''  logical :: accepted,reuse_geometry
  ! Geometry recovery depends only on mesh identity and the conducting-face
  ! mask, not on the changing positive conductances or transported velocity.
  ! A single native driver owns this workspace; concurrent calls are unsupported.
  logical,allocatable,save :: previous_conducting(:),geometry_valid(:)
  real(8),allocatable,save :: recovery_face_norm(:)
  integer(8),save :: saved_geometry_epoch=-1
  integer :: cache_bound_invalid''')
    new=audit.replace_once(new,'  bad_geometry=0','  bad_geometry=0;cache_bound_invalid=0')
    new=audit.replace_once(new,'private(f,l,r,sl,sr,recovery_bound)',
        'private(f,l,r,sl,sr,recovery_bound) reduction(max:cache_bound_invalid)')
    new=audit.replace_once(new,'''   if(.not.ieee_is_finite(recovery_bound).or.recovery_bound<0) then
    corrected_face(f)=.false.;cycle''', '''   if(.not.ieee_is_finite(recovery_bound).or.recovery_bound<0) then
    cache_bound_invalid=1;corrected_face(f)=.false.;cycle''')
    anchor='  allocate(boundary_cell(nc),corrected_face(nf),correction_cell(nc),recovery_slot(nc))'
    new=audit.replace_once(new,anchor,'''  reuse_geometry=.false.
  if(allocated(previous_conducting)) then
   if(saved_geometry_epoch==diffusion_geometry_epoch.and.size(previous_conducting)==nf.and.size(boundary_cell)==nc) &
    reuse_geometry=all(previous_conducting.eqv.(conductance>0))
  endif
  if(.not.reuse_geometry) then
   if(allocated(previous_conducting)) deallocate(previous_conducting,geometry_valid,recovery_face_norm, &
    boundary_cell,corrected_face,correction_cell,recovery_slot,recovery_cells,recovery_inverse, &
    recovery_polynomial,recovery_affine_inverse,recovery_affine_polynomial,recovery_order,recovery_valid)
   allocate(previous_conducting(nf),recovery_face_norm(nf));recovery_face_norm=0
'''+anchor)
    new=audit.replace_once(new,'   recovery_valid(nrecovery))','   recovery_valid(nrecovery),geometry_valid(nrecovery))')
    new=audit.replace_once(new,'   recovery_bound=.5d0*face_diffusivity(f)*(sl+sr)',
        '   recovery_face_norm(f)=sl+sr\n   recovery_bound=.5d0*face_diffusivity(f)*(sl+sr)')
    anchor='  diffusion_recovered_faces=diffusion_recovered_faces+active_recovery_faces'
    counts=old[old.index('  active_recovery_faces=0;active_affine_faces=0'):old.index(anchor)]
    new=audit.replace_once(new,counts,'')
    new=audit.replace_once(new,anchor,'''   previous_conducting=conductance>0;geometry_valid=recovery_valid
   saved_geometry_epoch=diffusion_geometry_epoch
   if(cache_bound_invalid/=0) saved_geometry_epoch=-1
  else
   ! Numerical failures during a previous velocity stage may disable recovery
   ! locally; restore the geometry validity for each new diffusion call.
   recovery_valid=geometry_valid
   !$omp parallel do schedule(static) if(nf>=4096) default(shared) private(f,recovery_bound) reduction(max:cache_bound_invalid)
   do f=1,nf
    if(conductance(f)<=0) cycle
    face_bound(f)=2*conductance(f)
    if(.not.corrected_face(f)) cycle
    recovery_bound=.5d0*face_diffusivity(f)*recovery_face_norm(f)
    if(.not.ieee_is_finite(recovery_bound).or.recovery_bound<0) then
     ! Match the original call-local fallback without poisoning the cached
     ! envelope: this exceptional state must rebuild on its next call.
     cache_bound_invalid=1;corrected_face(f)=.false.;cycle
    endif
    face_bound(f)=max(face_bound(f),recovery_bound)
   enddo
   !$omp end parallel do
   if(cache_bound_invalid/=0) saved_geometry_epoch=-1
  endif
'''+counts+anchor)
    return audit.replace_once(source,old,new)


def availability_source(source):
    m=hydraulic.m;old_euler=m.routine(source,'euler');new=old_euler
    new=audit.replace_once(new,'  real(8),allocatable,save :: hiding(:,:)',
        '  real(8),allocatable,save :: hiding(:,:)\n  real(8),allocatable,save :: bedweights(:,:)')
    new=audit.replace_once(new,'     hiding, &','     hiding, &\n     bedweights, &')
    new=audit.replace_once(new,'    hiding(nfrac,nc), &','    hiding(nfrac,nc), &\n    bedweights(nfrac,nc), &')
    new=audit.replace_once(new,'  if(bedlaw>0) then\n   if(profile_enabled)',
        '  if(bedlaw>0) then\n   call prepare_bed_availability(state,bedweights)\n   if(profile_enabled)')
    for cell in ('c','donor'):
        new=new.replace(f'bed_availability(state,g,{cell})',f'bedweights(g,{cell})')
    new=audit.replace_once(new,'hiding(:,c),mixture_d90(c),step_morfac,trace(:,f)',
        'hiding(:,c),mixture_d90(c),step_morfac,bedweights(:,c),trace(:,f)')
    new=audit.replace_once(new,'hiding,mixture_d90,step_morfac,fl(:,f),fr(:,f),speed(f))',
        'hiding,mixture_d90,step_morfac,bedweights,fl(:,f),fr(:,f),speed(f))')
    source=audit.replace_once(source,old_euler,new)
    for name in ('coupled_stage_characteristic','coupled_mpm_face'):
        before=m.routine(source,name);after=before
        if name=='coupled_stage_characteristic':
            after=audit.replace_once(after,'d90,morfac,trace,ztrace,valid)', 'd90,morfac,bedweights,trace,ztrace,valid)')
            after=audit.replace_once(after,'  real(8),intent(inout) :: trace(3)',
                '  real(8),intent(in) :: bedweights(nfrac)\n  real(8),intent(inout) :: trace(3)')
            after=after.replace('bed_availability(state,g,c)','bedweights(g)')
        else:
            after=audit.replace_once(after,'d90mix,morfac,fl,fr,speed)','d90mix,morfac,bedweights,fl,fr,speed)')
            after=audit.replace_once(after,'  real(8),intent(inout) :: fl(nstate)',
                '  real(8),intent(in) :: bedweights(nfrac,nc)\n  real(8),intent(inout) :: fl(nstate)')
            for cell in ('c','n'):after=after.replace(f'bed_availability(state,g,{cell})',f'bedweights(g,{cell})')
        source=audit.replace_once(source,before,after)
    helper=''' subroutine prepare_bed_availability(state,weights)
  ! Stage-local cache of the unchanged active-layer fractions and thin-layer
  ! factor. Preserve the scalar thickness sum and each class operation order.
  ! Called outside the face team; all face consumers read immutable weights.
  real(8),intent(in) :: state(nstate,nc)
  real(8),intent(out) :: weights(nfrac,nc)
  real(8) :: thickness,target,value
  integer :: c,g
  weights=1.d0
  if(deck_version<13) return
  !$omp parallel do schedule(static,256) if(nc*nfrac>=8192) default(shared) private(c,g,thickness,target,value)
  do c=1,nc
   thickness=0
   do g=1,nfrac
    thickness=thickness+state(bed_index(g),c)/bulk_fraction(g)
   enddo
   if(thickness<=0) then
    weights(:,c)=0;cycle
   endif
   target=active_layer_target(state,c)
   do g=1,nfrac
    value=(state(bed_index(g),c)/bulk_fraction(g))/thickness
    value=value*min(1.d0,thickness/target)
    weights(g,c)=value
   enddo
  enddo
  !$omp end parallel do
 end subroutine

'''
    return audit.replace_once(source,' subroutine mixture_statistics(',helper+' subroutine mixture_statistics(')


def run(variants,threads,cases,repeats,label,duration_scale=2.):
    root,report=paths();path=report/(label+'.json');assert not path.exists();rows=[]
    for case,source,start,duration,checkpoint,default_threads in aux.CASES:
        if case not in cases:continue
        for repeat in range(repeats):
            for nt in threads or (default_threads,):
                for variant in variants[repeat%len(variants):]+variants[:repeat%len(variants)]:
                    exe,_=identity(variant)
                    row=old.execute(root,f'{label}_{case}_{variant}_t{nt}_r{repeat+1}',exe,source,nt,duration_scale*duration,start,checkpoint)
                    row.update(case=case,variant=variant,repeat=repeat+1)
                    rows.append(row);r.write_json(path,rows)


def compare(left,right):
    result=old.compare(left,right)
    assert result['passed'] and result['raw_state_maximum']==0 and result['boundary_ledger']==0
    assert result['checkpoint_bitwise_equal'] and result['secondary_intensity_m_s']==0
    return result


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,report=paths();checks=[];timings=[];profiles=[]
    for label in ('profile','screen','diffusion_screen','diffusion_repeats','diffusion_long','diffusion_long_repeats'):
        rows=json.loads((report/(label+'.json')).read_text())
        for case in sorted({x['case'] for x in rows}):
            ref=next(x for x in rows if x['case']==case and x['variant']=='reference')
            for row in (x for x in rows if x['case']==case):
                check=compare(root/ref['label'],root/row['label'])
                assert row['steps']==ref['steps'] and row['profile']['rejected']==ref['profile']['rejected']
                checks.append(dict(label=row['label'],**check))
    rows=json.loads((report/'diffusion_repeats.json').read_text())
    for case in ('fine','large','larger'):
        groups={v:[x for x in rows if x['case']==case and x['variant']==v] for v in ('reference','diffusion_cached')}
        medians={}
        for variant,group in groups.items():
            values=dict(compute_s=[x['compute_s'] for x in group],process_s=[x['wall_s'] for x in group],
                engine_s=[x['profile']['profile: measured run wall seconds'] for x in group],
                diffusion_s=[x['profile']['profile: horizontal diffusion wall seconds'] for x in group])
            medians[variant]={key:float(np.median(value)) for key,value in values.items()}
            medians[variant]['compute_range_s']=[min(values['compute_s']),max(values['compute_s'])]
        baseline=medians['reference'];candidate=medians['diffusion_cached']
        timings.append(dict(case=case,threads=groups['reference'][0]['threads'],**medians,
            compute_reduction_percent=100*(1-candidate['compute_s']/baseline['compute_s']),
            diffusion_reduction_percent=100*(1-candidate['diffusion_s']/baseline['diffusion_s'])))
    for row in json.loads((report/'profile.json').read_text()):
        if row['variant']!='diagnostic':continue
        log=(root/row['label']/'native.log').read_text()
        detail={k:float(v) for k,v in re.findall(r'^diagnostic: (.*?)=\s*(\S+)',log,re.M)}
        pp=row['profile'];compute=row['compute_s']
        categories={'Curvature filter':detail['curvature filter wall seconds'],
            'Other secondary flow':detail['secondary relaxation wall seconds']-detail['curvature filter wall seconds'],
            'Bedload preparation':pp['profile: Euler bedload-capacity wall seconds'],
            'Viscosity':pp['profile: horizontal diffusion wall seconds']}
        categories['Remaining coupled work']=compute-sum(categories.values())
        profiles.append(dict(case=row['case'],compute_s=compute,categories=categories,detail=detail))
    longrows=sum([json.loads((report/(label+'.json')).read_text()) for label in ('diffusion_long','diffusion_long_repeats')],[])
    longtimings={v:{key:dict(median=float(np.median(values)),minimum=min(values),maximum=max(values),values=values)
        for key,values in ((key,[x[key] if key in x else x['profile'][key] for x in longrows if x['variant']==v])
            for key in ('compute_s','wall_s','profile: horizontal diffusion wall seconds'))}
        for v in ('reference','diffusion_cached')}
    r.write_json(report/'checks.json',checks);r.write_json(report/'summary.json',dict(timings=timings,profiles=profiles,long_timings=longtimings,checked_runs=len(checks)))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=['5,012 cells\n8 threads','20,088 cells\n16 threads','80,342 cells\n16 threads'];x=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for offset,v,title,color in ((-.18,'reference','Reference','#777c85'),(.18,'diffusion_cached','Geometry cache','#168381')):
        med=np.array([t[v]['compute_s'] for t in timings]);lo=np.array([t[v]['compute_range_s'][0] for t in timings]);hi=np.array([t[v]['compute_range_s'][1] for t in timings])
        axes[0].bar(x+offset,med,.34,color=color,label=title,yerr=np.array([med-lo,hi-med]),capsize=4)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Compute wall time (s)',title='Median and range of three repeats',ylim=(0,10));axes[0].legend()
    for offset,key,title,color in ((-.18,'compute_reduction_percent','Total compute','#168381'),(.18,'diffusion_reduction_percent','Viscosity phase','#ab6a26')):
        values=[t[key] for t in timings];bars=axes[1].bar(x+offset,values,.34,color=color,label=title)
        axes[1].bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3)
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Reduction in wall time (%)',title='A phase gain is not a whole-model gain',ylim=(0,35));axes[1].legend()
    fig.suptitle('Same-equation cache in coupled sediment runs')
    fig.savefig(report/'TIMINGS.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.4,4.6),layout='constrained');bottom=np.zeros(3)
    for category,color in zip(profiles[0]['categories'],('#7043a5','#b995d1','#ce963d','#168381','#b6bac3')):
        values=np.array([100*p['categories'][category]/p['compute_s'] for p in profiles])
        ax.bar(x,values,bottom=bottom,label=category,color=color)
        for j,v in enumerate(values):
            if v>=8:ax.text(j,bottom[j]+v/2,f'{v:.0f}%',ha='center',va='center')
        bottom+=values
    ax.set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Share of compute wall time (%)',title='Where the coupled run spends time')
    ax.legend(loc='center left',bbox_to_anchor=(1, .5))
    fig.savefig(report/'PROFILE.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.4),layout='constrained')
    for j,(v,color) in enumerate((('reference','#777c85'),('diffusion_cached','#168381'))):
        data=longtimings[v]['compute_s'];median=data['median']
        bars=ax.bar(j,median,.6,color=color,yerr=[[median-data['minimum']],[data['maximum']-median]],capsize=5)
        ax.bar_label(bars,labels=[f'{median:.3f} s'],padding=7)
        ax.scatter(np.full(3,j),data['values'],color='black',s=15,zorder=5)
    ax.set(xticks=[0,1],xticklabels=['Reference','Geometry cache'],ylim=(0,60),ylabel='Compute wall time (s)',
        title='Longer bend replay: 12 hydraulic seconds\nThree repeats, 5,012 cells, 8 threads, MF25')
    fig.savefig(report/'LONG_TIMINGS.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(timings=timings,long_timings=longtimings,checked_runs=len(checks)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('build','run','summarize'))
    parser.add_argument('--variants',nargs='+',default=['reference','diagnostic'])
    parser.add_argument('--threads',nargs='+',type=int)
    parser.add_argument('--cases',nargs='+',default=['fine','large','larger'])
    parser.add_argument('--repeats',type=int,default=1);parser.add_argument('--label',default='profile')
    parser.add_argument('--duration-scale',type=float,default=2.)
    args=parser.parse_args()
    if args.stage=='build':
        for variant in args.variants:build(variant)
    elif args.stage=='summarize':summarize()
    else:run(tuple(args.variants),args.threads,tuple(args.cases),args.repeats,args.label,args.duration_scale)
