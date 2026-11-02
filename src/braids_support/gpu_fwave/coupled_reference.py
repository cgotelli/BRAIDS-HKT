"""Practical native-case initialization and frozen coupled CPU diagnostics.

An I/O-only COPY of frozen alpha82 MF25 reads an ordinary case.data and native
sidecars. Production, immutable references and Delft3D are never edited. The
exporter resolves geometry, materials and state once on CPU; no CPU flux oracle
is necessary to run the resulting GPU input. Separate stage/trajectory exports
are validation only, not inputs required by the GPU solver.
"""
from __future__ import annotations
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
from . import resident_update as r

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=ROOT/'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_default_20261001_v1/runtime_mf25'
DEFAULT_INPUT=ROOT/'braids_support/sediment/lfm_bend_2d/runs/fast_cpu_cold20h_20261001_v1_mobile_braids/case.data'
TAG='gpu_coupled_reference_20261004_v5'
MAGIC=b'BRDGPU01'

INTEGER_CONTROLS=('nc nf nb nx ny scheme bedlaw exchange_mode bed_slope deck_version nfrac nstate nprimitive '
    'nledger nbed_layers nsubstrate_layers nphysical_state nphysical_primitive nphysical_ledger '
    'cpu_time_integrator cpu_outlet_extension active_layer_mode hiding_exposure mpm_friction_mode '
    'sediment_roughness_mode settling_mode secondary_flow_model secondary_flow_inflow_mode '
    'horizontal_viscosity_mode horizontal_viscosity_geometry nmorfac bank_toe_erosion_mode').split()
REAL_CONTROLS=('grav dry_tolerance rho speed_limit dx dy time dtmax endtime output_dt cfl_target '
    'mpm_cf mpm_critical bedload_adaptation_length_diameters active_layer_thickness active_layer_depth_fraction active_layer_minimum_thickness '
    'substrate_layer_thickness hiding_parker_alpha hiding_wu_exponent transport_calibration '
    'bed_slope_alfabs bed_slope_alfabn bed_friction_angle_deg secondary_flow_espir secondary_flow_momentum_beta '
    'gaia_skin_roughness_ratio gaia_slope_beta gaia_deviation_beta2 gaia_secondary_alpha '
    'morphological_factor horizontal_viscosity manning_chezy_minimum_depth manning_friction_chezy_minimum_depth '
    'hindered_settling_reference_concentration '
    'bank_erosion_fraction').split()
FLAGS=('amr_enabled secondary_flow secondary_flow_transport_direction bedload_adaptation bedform_roughness '
    'bank_failure bank_front_tracking bank_avalanche general_avalanche production_section_geometry '
    'suspended_transport_active morphological_acceleration').split()


def rec(name,expr=None,dims=(),kind='r'):
    expr=expr or name;shape='['+','.join(dims)+']' if dims else '[integer::]'
    value=f'reshape({expr},[size({expr})])' if dims else f'[{expr}]'
    return f"  call gpu_put_{kind}(gpu_unit,'{name}',{value},{shape})\n"


WRITERS='''
 subroutine gpu_put_header(unit,name,kind,dimensions)
  integer,intent(in) :: unit,kind,dimensions(:)
  character(*),intent(in) :: name
  character(64) :: padded
  integer :: dims(4)
  if(size(dimensions)>4.or.len_trim(name)>64) error stop 'GPU export schema overflow'
  padded=name;dims=0;dims(1:size(dimensions))=dimensions
  write(unit) padded,kind,size(dimensions),dims
 end subroutine
 subroutine gpu_put_r(unit,name,values,dimensions)
  integer,intent(in) :: unit,dimensions(:)
  character(*),intent(in) :: name
  real(8),intent(in) :: values(:)
  call gpu_put_header(unit,name,2,dimensions);write(unit) values
 end subroutine
 subroutine gpu_put_i(unit,name,values,dimensions)
  integer,intent(in) :: unit,dimensions(:),values(:)
  character(*),intent(in) :: name
  call gpu_put_header(unit,name,1,dimensions);write(unit) values
 end subroutine
 subroutine gpu_finish(unit)
  integer,intent(in) :: unit
  call gpu_put_header(unit,'END',0,[integer::]);close(unit)
 end subroutine
 subroutine gpu_open(unit,path)
  integer,intent(out) :: unit
  character(*),intent(in) :: path
  open(newunit=unit,file=path,access='stream',form='unformatted',status='replace')
  write(unit) 'BRDGPU01',1
 end subroutine
'''


def static_export_source():
    text='''
 subroutine gpu_export_input()
  integer :: gpu_unit,c,f,g,l
  integer :: gpu_bed_indices(nbed_layers,nfrac),gpu_suspended_indices(nfrac),gpu_face_mode(nf),gpu_active(nc)
  integer :: gpu_fixed_boundary_bed_mode(nb)
  real(8) :: gpu_bed(nc),gpu_distance(2,nf),gpu_layers(nbed_layers,nfrac,nc),gpu_fixed_boundary_bed(nf)
  real(8) :: gpu_state(nstate,nc)
  if(amr_enabled.or.scheme/=6) error stop 'GPU preparation requires fixed-grid scheme6'
  call effective_bed_elevation(q,gpu_bed)
  do g=1,nfrac
   gpu_suspended_indices(g)=suspended_index(g)-1
   do l=1,nbed_layers
    gpu_bed_indices(l,g)=bed_layer_index(l,g)-1
    gpu_layers(l,g,:)=q(bed_layer_index(l,g),:)
   enddo
  enddo
  gpu_face_mode=0;gpu_distance=0;gpu_active=1
  gpu_fixed_boundary_bed_mode=0;gpu_fixed_boundary_bed=0.d0
  if(bedlaw==9) then
   if(.not.allocated(fixed_boundary_bed_mode).or..not.allocated(fixed_boundary_bed)) &
    error stop 'GPU TELEMAC-GAIA fixed boundary bed is not initialized'
   gpu_fixed_boundary_bed_mode=fixed_boundary_bed_mode
   gpu_fixed_boundary_bed=fixed_boundary_bed
  endif
  do f=1,nf
   if(bid(f)>0) gpu_face_mode(f)=mode(bid(f))
   gpu_distance(1,f)=abs(dot_product(mid(:,f)-xy(:,owner(f)),normal(:,f)))
   if(neighbour(f)>0) gpu_distance(2,f)=abs(dot_product(xy(:,neighbour(f))-mid(:,f),normal(:,f)))
  enddo
  call gpu_open(gpu_unit,'coupled_input.bin')
'''
    for name in INTEGER_CONTROLS:text+=rec(name,kind='i')
    for name in REAL_CONTROLS:text+=rec(name)
    for name in FLAGS:text+=rec(name,f'merge(1,0,{name})',kind='i')
    text+=rec('secondary_index','merge(secondary_state_index()-1,-1,prognostic_secondary_flow())',kind='i')
    text+=rec('secondary_ledger_index','merge(secondary_ledger_index()-1,-1,prognostic_secondary_flow())',kind='i')
    text+=rec('secondary_primitive_index','merge(secondary_primitive_index()-1,-1,prognostic_secondary_flow())',kind='i')
    arrays=[('state','q',('nstate','nc')),('ledger','ledger',('nledger','nb')),
        ('bed','gpu_bed',('nc',)),('layer_mass','gpu_layers',('nbed_layers','nfrac','nc')),
        ('area','area',('nc',)),('rock','rock',('nc',)),('xy','xy',('2','nc')),
        ('normal','normal',('2','nf')),('length','length',('nf',)),('mid','mid',('2','nf')),
        ('distance','gpu_distance',('2','nf')),('forcing','forcing',('6','nb')),('manning_cell','manning_cell',('nc',)),
        ('hydraulic_ls_matrix','hydraulic_ls_matrix',('4','nc')),('hydraulic_ls_weight','hydraulic_ls_weight',('size(faces)',)),
        ('bed_ls_matrix','bed_ls_matrix',('4','nc')),('bed_ls_delta','bed_ls_delta',('2','size(faces)')),
        ('bed_ls_weight','bed_ls_weight',('size(faces)',)),('rectangle_width','rectangle_width',('2','nc')),
        ('face_displacement','face_displacement',('2','2','nf')),('wall_stencil_normal','wall_stencil_normal',('2','5','2','nc')),
        ('wall_box_center','wall_box_center',('2','nc')),('wall_normal','wall_normal',('2','nc')),
        ('wall_mean_distance','wall_mean_distance',('nc',)),('boundary_bed_offset','boundary_bed_offset',('nf',)),
        ('fixed_boundary_bed','gpu_fixed_boundary_bed',('nf',)),
        ('boundary_bed_left_offset','boundary_bed_left_offset',('nf',)),('boundary_bed_right_offset','boundary_bed_right_offset',('nf',)),
        ('fraction_concentration','fraction_concentration',('nfrac','nb')),('fraction_bed_input','fraction_bed_input',('nfrac','nb')),
        ('section_energy_slope','section_energy_slope',('nb',))]
    for name in ('bulk_fraction ws_fraction d50_fraction d90_fraction rhos_fraction z0_fraction sedthr_fraction '
                 'bedload_min_depth_fraction reference_height_fraction dstar_fraction taucr_fraction').split():
        arrays.append((name,name,('nfrac',)))
    for name,expr,dims in arrays:text+=rec(name,expr,dims)
    integer_arrays=[('bed_indices','gpu_bed_indices',('nbed_layers','nfrac')),
        ('suspended_indices','gpu_suspended_indices',('nfrac',)),('owner','owner-1',('nf',)),
        ('neighbour','neighbour-1',('nf',)),('bid','bid-1',('nf',)),('offset','offset-1',('nc+1',)),
        ('faces','faces-1',('size(faces)',)),('signs','signs',('size(faces)',)),('active','gpu_active',('nc',)),
        ('boundary_mode','gpu_face_mode',('nf',)),('boundary_offset','boundary_offset-1',('nb+1',)),
        ('boundary_faces','boundary_faces-1',('size(boundary_faces)',)),('mode','mode',('nb',)),
        ('reversal_policy','reversal_policy',('nb',)),('bed_boundary_mode','bed_boundary_mode',('nb',)),
        ('fixed_boundary_bed_mode','gpu_fixed_boundary_bed_mode',('nb',)),
        ('section_control','section_control',('nb',)),('sampled','sampled_boundary_bed',('nf',)),
        ('viscosity_equilibrated_inlet','merge(1,0,viscosity_equilibrated_inlet)',('nb',)),
        ('sampled_boundary_open','merge(1,0,sampled_boundary_open)',('nf',)),
        ('weno_cells','weno_cells-1',('5','2','nf')),('wall_triangle','merge(1,0,wall_triangle)',('nc',)),
        ('wall_stencil_ids','wall_stencil_ids-1',('5','2','nc')),('wall_stencil_kind','wall_stencil_kind',('5','2','nc')),
        ('wall_slot','wall_slot-1',('2','nf')),('open_boundary_distance','open_boundary_distance',('nc',)),
        ('uncoupled_boundary_distance','uncoupled_boundary_distance',('nc',)),
        ('nseries','nseries',('nb',)),('nhydro_series','nhydro_series',('nb',)),
        ('nfraction_series','nfraction_series',('nfrac','nb')),('sediment_type_fraction','sediment_type_fraction',('nfrac',)),
        ('sediment_roughness_fraction','sediment_roughness_fraction',('nfrac',))]
    for name,expr,dims in integer_arrays:text+=rec(name,expr,dims,'i')
    text+='''  call gpu_finish(gpu_unit)
  if(horizontal_viscosity>0.and.horizontal_viscosity_geometry==1) then
   gpu_state=q;gpu_export_diffusion=.true.
   call diffuse_momentum(gpu_state,1.d-6)
   gpu_export_diffusion=.false.
  endif
 end subroutine
'''
    return text


def diffusion_export_block():
    text="  if(gpu_export_diffusion) then\n   call gpu_open(gpu_unit,'coupled_diffusion.bin')\n"
    for name,dims in [('conductance',('nf',)),('face_diffusivity',('nf',)),('tangent_offset',('nf',)),
        ('normal_asymmetry',('nf',)),('face_bound',('nf',)),('rate',('nc',)),
        ('recovery_inverse',('5','5','nrecovery')),('recovery_affine_inverse',('2','2','nrecovery'))]:text+=rec(name,dims=dims)
    for name,expr,dims in [('corrected_face','merge(1,0,corrected_face)',('nf',)),('recovery_slot','recovery_slot-1',('nc',)),
        ('recovery_cells','recovery_cells-1',('nrecovery',)),('recovery_order','recovery_order',('nrecovery',)),
        ('recovery_valid','merge(1,0,recovery_valid)',('nrecovery',))]:text+=rec(name,expr,dims,'i')
    text+='''   allocate(gpu_stencils(max_diffusion_recovery_stencil,nrecovery),gpu_stencil_counts(nrecovery))
   gpu_stencils=-1
   do slot=1,nrecovery
    call diffusion_recovery_stencil(recovery_cells(slot),conductance,gpu_stencil,gpu_nstencil,gpu_complete)
    if(.not.gpu_complete) error stop 'GPU diffusion recovery stencil incomplete'
    gpu_stencil_counts(slot)=gpu_nstencil
    gpu_stencils(:,slot)=gpu_stencil-1
   enddo
'''
    text+=rec('recovery_stencils','gpu_stencils',('max_diffusion_recovery_stencil','nrecovery'),'i')
    text+=rec('recovery_stencil_counts','gpu_stencil_counts',('nrecovery',),'i')
    text+=rec('maximum_rate')
    text+='   call gpu_finish(gpu_unit)\n   return\n  endif\n'
    return text


def stage_before():
    text="  if(gpu_export_stage) then\n   call gpu_open(gpu_unit,'coupled_stage_pre.bin')\n"
    for name in ('dt','evaluation_time','step_morfac'):text+=rec(name)
    for name,dims in [('state',('nstate','nc')),('bed',('nc',)),('v',('nprimitive','nc')),
        ('grad',('nprimitive','2','nc')),('vlo',('nprimitive','nc')),('vhi',('nprimitive','nc')),
        ('trace',('3','nf')),('bc_hydraulic',('2','nb')),('bc_bed_input',('nfrac','nb')),
        ('downslope',('2','nc')),('hiding',('nfrac','nc')),('mixture_d90',('nc',)),
        ('bedq_cell',('2','nfrac','nc')),('bedq_grad',('2','2','nfrac','nc')),('bedtau_cell',('nfrac','nc')),
        ('fl',('nstate','nf')),('fr',('nstate','nf')),('speed',('nf',))]:text+=rec(name,dims=dims)
    text+=rec('active','merge(1,0,active_cell)',('nc',),'i')
    text+=rec('hydro_constant','merge(1,0,hydro_constant)',('nc',),'i')
    text+='   if(prognostic_secondary_flow()) then\n'
    for name,dims in [('secondary_curvature',('nc',)),('secondary_face_left',('nf',)),('secondary_face_right',('nf',)),
        ('secondary_boundary_intensity',('nf',)),('secondary_low_flux',('nf',)),('secondary_anti_flux',('nf',)),
        ('secondary_stress_face',('2','nf'))]:text+=rec(name,dims=dims)
    text+='   endif\n   call gpu_finish(gpu_unit)\n  endif\n'
    return text


def stage_after():
    text="  if(gpu_export_stage) then\n   call gpu_open(gpu_unit,'coupled_stage_post.bin')\n"
    for name,dims in [('next',('nstate','nc')),('accepted',('nledger','nb')),('fl',('nstate','nf')),('fr',('nstate','nf')),
        ('theta',('nfrac','nc')),('water_theta',('nc',))]:text+=rec(name,dims=dims)
    text+='''   if(prognostic_secondary_flow()) then
'''
    for name,dims in [('secondary_low_state',('nc',)),('secondary_low_flux',('nf',)),('secondary_anti_flux',('nf',))]:text+=rec(name,dims=dims)
    text+='''   endif
   call gpu_finish(gpu_unit)
   stop 'GPU coupled stage diagnostic complete; not a complete simulation'
  endif
'''
    return text


def instrument(source):
    """Add export queries only. Normal evolution formulas/calls stay intact."""
    marker='\ncontains\n';at=source.index(marker)
    source=source[:at]+'''\n logical :: gpu_export_stage=.false.,gpu_export_diffusion=.false.,gpu_source_written=.false.
 logical :: gpu_export_trajectory=.false.
 integer :: gpu_trajectory_unit=-1
 character(32) :: gpu_export_setting
'''+source[at:]
    source=r.replace_once(source,'end module\n\nprogram run_cutcell',WRITERS+static_export_source()+'end module\n\nprogram run_cutcell')
    anchor=' call cpu_time(started)'
    setup=''' call get_environment_variable('BRAIDS_GPU_STAGE_EXPORT',gpu_export_setting)
 gpu_export_stage=trim(gpu_export_setting)=='1'
 call get_environment_variable('BRAIDS_GPU_TRAJECTORY_EXPORT',gpu_export_setting)
 gpu_export_trajectory=trim(gpu_export_setting)=='1'
 call get_environment_variable('BRAIDS_GPU_INITIAL_EXPORT',gpu_export_setting)
 if(trim(gpu_export_setting)=='1') then
  call gpu_export_input()
  stop 'GPU native initialization exported; no solver timestep performed'
 endif
 if(gpu_export_trajectory) then
  open(newunit=gpu_trajectory_unit,file='coupled_trajectory.bin',access='stream',form='unformatted',status='replace')
  write(gpu_trajectory_unit) nc,nstate,nb,nledger
  write(gpu_trajectory_unit) time,0.d0,0.d0,q,ledger
 endif
'''
    viscosity=''' call get_environment_variable('BRAIDS_GPU_VISCOSITY_EXPORT',gpu_export_setting)
 if(trim(gpu_export_setting)=='1') then
  call gpu_export_input()
  call gpu_open(unit,'coupled_viscosity.bin')
'''+rec('state_before','q',('nstate','nc')).replace('gpu_unit','unit')+rec('dt','dtmax').replace('gpu_unit','unit')+'''
  call diffuse_momentum(q,dtmax)
'''+rec('state_after','q',('nstate','nc')).replace('gpu_unit','unit')+rec('diffusion_substeps',kind='i').replace('gpu_unit','unit')+rec('diffusion_minimum_correction_factor').replace('gpu_unit','unit')+'''
  call gpu_finish(unit)
  stop 'GPU isolated native viscosity oracle complete'
 endif
'''
    setup=viscosity+setup
    source=r.replace_once(source,anchor,setup+anchor)
    source=r.replace_once(source,'   if(checkpoint_steps>0) then',
        '   if(gpu_export_trajectory) write(gpu_trajectory_unit) time,dt,trial_cfl_peak,q,ledger\n   if(checkpoint_steps>0) then')
    source=r.replace_once(source,' if(write_ascii_frames) close(unit)',
        ' if(gpu_export_trajectory) close(gpu_trajectory_unit)\n if(write_ascii_frames) close(unit)')
    begin=source.index(' subroutine euler(');end=source.index(' end subroutine',begin);block=source[begin:end]
    block=r.replace_once(block,'  if(any(workspace_shape/=', '  integer :: gpu_unit\n  if(any(workspace_shape/=')
    block=r.replace_once(block,'  morfac=step_morfac',stage_before()+'  morfac=step_morfac')
    block=r.replace_once(block,'  ok=.true.',stage_after()+'  ok=.true.')
    source=source[:begin]+block+source[end:]
    begin=source.index(' subroutine secondary_relax_state(');end=source.index(' end subroutine',begin);block=source[begin:end]
    block=r.replace_once(block,'  source_integral=0.d0;ok=.true.','  integer :: gpu_unit\n  source_integral=0.d0;ok=.true.')
    tail='''  if(gpu_export_stage.and..not.gpu_source_written) then
   call gpu_open(gpu_unit,'coupled_source.bin')
'''+rec('dt_source')+rec('source_integral')+rec('state',dims=('nstate','nc'))+rec('secondary_curvature',dims=('nc',))+'''
   call gpu_finish(gpu_unit);gpu_source_written=.true.
  endif
'''
    block+=tail;source=source[:begin]+block+source[end:]
    begin=source.index(' subroutine diffuse_momentum_nonorthogonal(');end=source.index(' end subroutine',begin);block=source[begin:end]
    block=r.replace_once(block,'  if(horizontal_viscosity<=0.or.dt<=0) return',
        '  integer :: gpu_unit,gpu_stencil(max_diffusion_recovery_stencil),gpu_nstencil\n'
        '  integer,allocatable :: gpu_stencils(:,:),gpu_stencil_counts(:)\n  logical :: gpu_complete\n'
        '  if(horizontal_viscosity<=0.or.dt<=0) return')
    block=r.replace_once(block,'  base_step=dt',diffusion_export_block()+'  base_step=dt')
    return source[:begin]+block+source[end:]


def verify_parent(parent=PARENT):
    parent=Path(parent);identity=json.loads((parent/'build.json').read_text())
    if r.sha(parent/'xbraids_candidate.exe')!=identity['executable_sha256']:raise ValueError('Frozen native executable changed')
    for name,digest in identity['source_sha256'].items():
        if r.sha(parent/Path(name).name)!=digest:raise ValueError('Frozen native source changed: '+name)
    return identity


def build(tag=TAG,parent=PARENT):
    root,_=r.paths(tag);root.mkdir(parents=True,exist_ok=False);runtime=root/'runtime_export';runtime.mkdir()
    identity=verify_parent(parent);sources=[]
    for name in identity['source_sha256']:
        source=Path(parent)/Path(name).name;target=runtime/source.name
        if source.name=='cutcell_solver.f90':target.write_text(instrument(source.read_text()))
        else:shutil.copy2(source,target)
        sources.append(str(target))
    command=[r'C:\Strawberry\c\bin\gfortran.exe','-O3','-flto','-fopenmp','-ffree-line-length-none',
        '-ffixed-line-length-none',*sources,'-o',str(runtime/'xbraids_coupled_export.exe')]
    result=subprocess.run(command,cwd=runtime,capture_output=True,text=True,timeout=300)
    (runtime/'build.log').write_text(result.stdout+result.stderr);result.check_returncode()
    shutil.copy2(__file__,runtime/'coupled_reference.py')
    record=dict(command=command,source_sha256={Path(x).name:r.sha(x) for x in sources},
        executable_sha256=r.sha(runtime/'xbraids_coupled_export.exe'),parent_path=str(parent),
        parent_executable_sha256=identity['executable_sha256'],driver_sha256=r.sha(__file__),
        transformation='Export-only additions; no normal native solver formulas changed',production_modified=False,delft_modified=False)
    r.write(runtime/'build.json',record);return runtime


def runtime_path(tag=TAG):
    root,_=r.paths(tag);runtime=root/'runtime_export';identity=json.loads((runtime/'build.json').read_text())
    if r.sha(runtime/'xbraids_coupled_export.exe')!=identity['executable_sha256']:raise ValueError('Export runtime changed')
    for name,digest in identity['source_sha256'].items():
        if r.sha(runtime/name)!=digest:raise ValueError('Export source changed')
    return runtime,identity


def _check_stop(stop_file):
    if stop_file is not None and Path(stop_file).exists():
        raise InterruptedError('GPU preparation cancelled by stop file: '+str(stop_file))


def _sha_with_stop(path,stop_file):
    if stop_file is None:return r.sha(path)
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):
            _check_stop(stop_file);digest.update(chunk)
    _check_stop(stop_file)
    return digest.hexdigest()


def _copy_with_stop(source,target,stop_file):
    """A cancelled copy remains visibly partial, never a completed sidecar."""
    partial=target.with_name(target.name+'.partial')
    with Path(source).open('rb') as input_stream,partial.open('xb') as output_stream:
        while chunk:=input_stream.read(1024*1024):
            _check_stop(stop_file);output_stream.write(chunk)
    _check_stop(stop_file)
    shutil.copystat(source,partial)
    partial.replace(target)


def copy_inputs(deck,destination,*,stop_file=None):
    """Copy explicit deck dependencies only; never run native output in user's folder."""
    _check_stop(stop_file)
    deck=Path(deck).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    text=deck.read_text();dependencies={deck.name:deck}
    # Native binary envelope identifies sidecar plus SHA on its own line.
    for line in text.splitlines():
        tokens=line.split()
        if len(tokens)==2 and re.fullmatch(r'[0-9a-fA-F]{64}',tokens[1]):
            name=tokens[0];path=Path(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Deck sidecar must be a safe relative path')
            source=deck.parent/path
            if not source.is_file() or _sha_with_stop(source,stop_file)!=tokens[1].lower():raise ValueError('Native sidecar checksum mismatch')
            dependencies[name]=source
    for name in (deck.name+'.json','geometry.npz','analysis_geometry.npz'):
        if (deck.parent/name).is_file():dependencies[name]=deck.parent/name
    for name,source in dependencies.items():
        _check_stop(stop_file)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if stop_file is None:shutil.copy2(source,target)
        else:_copy_with_stop(source,target,stop_file)
    return dict(original_deck=str(deck),files={name:_sha_with_stop(destination/name,stop_file) for name in dependencies})


def _execute(folder,deck_name,runtime,action,*,stop_file=None,native_integrator='ssp43'):
    if native_integrator not in ('ssp43','legacy'):
        raise ValueError('GPU native initializer requires ssp43 or legacy')
    env={k:v for k,v in os.environ.items() if not k.startswith('BRAIDS_')}
    env.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',
               BRAIDS_TIME_INTEGRATOR=native_integrator)
    env['BRAIDS_GPU_'+action+'_EXPORT']='1'
    command=[str(runtime/'xbraids_coupled_export.exe'),deck_name]
    if stop_file is None:
        result=subprocess.run(command,cwd=folder,env=env,capture_output=True,text=True,timeout=240)
        (folder/(action.lower()+'.log')).write_text(result.stdout+result.stderr);result.check_returncode()
        return
    _check_stop(stop_file)
    process=subprocess.Popen(command,cwd=folder,env=env,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True)
    began=time.monotonic();cancelled=False;timed_out=False
    while True:
        if Path(stop_file).exists():
            cancelled=True;break
        remaining=240-(time.monotonic()-began)
        if remaining<=0:
            timed_out=True;break
        try:
            stdout,stderr=process.communicate(timeout=min(.25,remaining))
            break
        except subprocess.TimeoutExpired:
            continue
    if cancelled or timed_out:
        if process.poll() is None:
            process.terminate()
        try:stdout,stderr=process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill();stdout,stderr=process.communicate(timeout=5)
    (folder/(action.lower()+'.log')).write_text(stdout+stderr)
    if cancelled:raise InterruptedError('GPU preparation cancelled by stop file: '+str(stop_file))
    if timed_out:raise subprocess.TimeoutExpired(command,240,output=stdout,stderr=stderr)
    _check_stop(stop_file)
    if process.returncode:raise subprocess.CalledProcessError(process.returncode,command,output=stdout,stderr=stderr)


def prepare(deck,destination,tag=TAG,*,stop_file=None,native_integrator='ssp43'):
    if native_integrator not in ('ssp43','legacy'):
        raise ValueError('GPU preparation requires native integrator ssp43 or legacy')
    if stop_file is None:
        # Preserve the established uninterruptible preparation and byte output.
        runtime,identity=runtime_path(tag);destination=Path(destination)
        origin=copy_inputs(deck,destination);_execute(destination,Path(deck).name,runtime,'INITIAL',
                                                       native_integrator=native_integrator)
        data=read_snapshot(destination)
        record=dict(origin=origin,export_runtime_sha256=identity['executable_sha256'],parent_runtime_sha256=identity['parent_executable_sha256'],
            artifacts={p.name:r.sha(p) for p in destination.glob('coupled_*.bin')},
            schema=1,cells=data['nc'],faces=data['nf'],fractions=data['nfrac'],layers=data['nbed_layers'],
            secondary_model=data['secondary_flow_model'],GPU_solver_requires_no_oracle_outputs=True,
            initialization_CPU_only=True,solver_steps_performed=0,production_modified=False,delft_modified=False)
        if native_integrator!='ssp43':record['native_integrator_requested']=native_integrator
        r.write(destination/'preparation_receipt.json',record)
        np.savez_compressed(destination/'coupled_input.npz',**data)
        return data
    stop_file=Path(stop_file).resolve();destination=Path(destination).resolve()
    _check_stop(stop_file)
    try:
        runtime,identity=runtime_path(tag)
        _check_stop(stop_file)
        origin=copy_inputs(deck,destination,stop_file=stop_file)
        _execute(destination,Path(deck).name,runtime,'INITIAL',stop_file=stop_file,
                 native_integrator=native_integrator)
        _check_stop(stop_file)
        data=read_snapshot(destination)
        _check_stop(stop_file)
        record=dict(origin=origin,export_runtime_sha256=identity['executable_sha256'],parent_runtime_sha256=identity['parent_executable_sha256'],
            artifacts={p.name:_sha_with_stop(p,stop_file) for p in destination.glob('coupled_*.bin')},
            schema=1,cells=data['nc'],faces=data['nf'],fractions=data['nfrac'],layers=data['nbed_layers'],
            secondary_model=data['secondary_flow_model'],GPU_solver_requires_no_oracle_outputs=True,
            initialization_CPU_only=True,solver_steps_performed=0,production_modified=False,delft_modified=False)
        if native_integrator!='ssp43':record['native_integrator_requested']=native_integrator
        np.savez_compressed(destination/'coupled_input.partial.npz',**data)
        _check_stop(stop_file)
        (destination/'coupled_input.partial.npz').replace(destination/'coupled_input.npz')
        receipt=destination/'preparation_receipt.partial.json'
        r.write(receipt,record)
        _check_stop(stop_file)
        receipt.replace(destination/'preparation_receipt.json')
        return data
    except Exception as exc:
        if destination.is_dir():
            try:r.write(destination/'preparation_status.json',dict(
                status='cancelled' if isinstance(exc,InterruptedError) else 'failed',
                error=str(exc),stop_file=str(stop_file)))
            except OSError:pass
        raise


def read_records(path):
    """Strict little-endian self-describing records -> native-shaped C arrays."""
    result={}
    from braids_core.windows_paths import filesystem_path
    with filesystem_path(path).open('rb') as stream:
        if stream.read(8)!=MAGIC:raise ValueError('Wrong coupled export magic')
        version=np.fromfile(stream,dtype='<i4',count=1)
        if version.size!=1 or version[0]!=1:raise ValueError('Unsupported coupled export schema')
        while True:
            raw=stream.read(64)
            if len(raw)!=64:raise ValueError('Truncated coupled record name')
            name=raw.decode('ascii').strip();header=np.fromfile(stream,dtype='<i4',count=6)
            if header.size!=6:raise ValueError('Truncated coupled record header')
            kind,rank,*shape=map(int,header)
            if name=='END':
                if kind!=0 or rank!=0 or stream.read(1):raise ValueError('Invalid coupled export terminator')
                break
            if name in result or kind not in (1,2) or not 0<=rank<=4 or any(x<0 for x in shape[:rank]):
                raise ValueError('Invalid coupled record header')
            shape=tuple(shape[:rank]);count=int(np.prod(shape)) if rank else 1
            if count>100000000:raise ValueError('Implausibly large coupled record')
            values=np.fromfile(stream,dtype='<i4' if kind==1 else '<f8',count=count)
            if len(values)!=count or not np.isfinite(values).all():raise ValueError('Invalid/truncated coupled '+name)
            result[name]=values[0].item() if not rank else np.ascontiguousarray(values.reshape(shape,order='F'))
    return result


def validate_adaptation_record(data):
    """Never infer an enabled deck's relaxation length from a GPU default."""
    flag=np.asarray(data.get('bedload_adaptation'))
    if flag.shape!=() or flag.dtype.kind not in 'iub' or int(flag) not in (0,1):
        raise ValueError('Missing/invalid native bed-load adaptation flag')
    value=data.get('bedload_adaptation_length_diameters')
    if value is None:
        if int(flag):
            raise ValueError('Enabled bed-load adaptation requires exported native length')
        return
    length=np.asarray(value)
    if length.shape!=() or length.dtype.kind!='f' or not np.isfinite(length) or float(length)<=0:
        raise ValueError('Invalid exported native bed-load adaptation length')


def read_snapshot(folder):
    from braids_core.windows_paths import filesystem_path
    folder=filesystem_path(folder);data=read_records(folder/'coupled_input.bin')
    validate_adaptation_record(data)
    nc,nf=data['nc'],data['nf']
    if data['state'].shape!=(data['nstate'],nc) or data['layer_mass'].shape!=(data['nbed_layers'],data['nfrac'],nc):
        raise ValueError('Invalid coupled state layout')
    if data['owner'].shape!=(nf,) or np.any(data['owner']<0) or np.any(data['owner']>=nc):raise ValueError('Invalid coupled owners')
    fixed_keys={'fixed_boundary_bed_mode','fixed_boundary_bed'}
    present=fixed_keys.intersection(data)
    if present and present!=fixed_keys:
        raise ValueError('Incomplete TELEMAC-GAIA fixed boundary-bed export')
    if present:
        fixed_mode=np.asarray(data['fixed_boundary_bed_mode'])
        fixed_bed=np.asarray(data['fixed_boundary_bed'])
        if (fixed_mode.shape!=(data['nb'],) or fixed_mode.dtype.kind not in 'iu' or
                not np.isin(fixed_mode,(0,1)).all() or fixed_bed.shape!=(nf,) or
                fixed_bed.dtype.kind!='f' or not np.isfinite(fixed_bed).all()):
            raise ValueError('Invalid TELEMAC-GAIA fixed boundary-bed export')
        bid=np.asarray(data['bid']);owner=np.asarray(data['owner'])
        face_fixed=np.zeros(nf,dtype=bool)
        exterior=bid>=0
        face_fixed[exterior]=fixed_mode[bid[exterior]]==1
        bed_mode=np.asarray(data['bed_boundary_mode'])
        if (bed_mode.shape!=(data['nb'],) or bed_mode.dtype.kind not in 'iu' or
                not np.isin(bed_mode,(0,1,2)).all()):
            raise ValueError('Invalid native bed boundary-mode export')
        expected_mode=np.isin(bed_mode,(1,2)).astype(fixed_mode.dtype)
        if int(data['bedlaw'])==9:
            if (not np.array_equal(fixed_mode,expected_mode) or
                    np.any(np.asarray(data['mode'])[fixed_mode==1]!=6)):
                raise ValueError('GAIA inlet lacks exact fixed-face bed semantics')
            expected=(np.asarray(data['bed'])[owner[face_fixed]]+
                      np.asarray(data['boundary_bed_offset'])[face_fixed])
            if (not np.array_equal(fixed_bed[face_fixed],expected) or
                    np.any(fixed_bed[~face_fixed]!=0.) or
                    np.any(np.signbit(fixed_bed[~face_fixed]))):
                raise ValueError('GAIA fixed boundary bed differs from the native initial face bed')
        elif np.any(fixed_mode) or np.any(fixed_bed):
            raise ValueError('Fixed GAIA boundary-bed data is inert outside law 9')
        data['gaia_fixed_inlet_contract']='explicit_native_v5'
    else:
        # Authenticated historical v3/v4 snapshots remain inspectable.  Do not
        # manufacture the new physics: current law-9 execution/acceptance gates
        # require the explicit arrays and will reject this marker.
        data['gaia_fixed_inlet_contract']='historical_absent'
    data['q']=np.ascontiguousarray(data['state'][:3]);data['dry']=data['dry_tolerance']
    data['outlet_extension']=data['cpu_outlet_extension'];data['dt']=data['dtmax']
    data['bedoffset']=data['boundary_bed_offset'];data['bc_hydraulic']=np.ascontiguousarray(data['forcing'][:2])
    data['bc_bed_input']=data['fraction_bed_input'].copy()
    if (folder/'coupled_diffusion.bin').exists():
        for key,value in read_records(folder/'coupled_diffusion.bin').items():data['diffusion_'+key]=value
    return data


def change_time_controls(text,end,dt,output):
    """Identify the unique scheme line; the immediately preceding line is time."""
    lines=text.splitlines();candidates=[]
    for index,line in enumerate(lines):
        if re.fullmatch(r'\s*[1-6]\s+[0-9]+\s+[0-9]+\s*',line):
            try:values=[float(x) for x in lines[index-1].split()]
            except ValueError:continue
            if len(values)==3 and values[0]>0 and values[1]>0 and values[2]>0:candidates.append(index-1)
    if len(candidates)!=1:raise ValueError('Ambiguous native time controls')
    position=candidates[0];previous=lines[position];lines[position]=f'{end:.17g} {dt:.17g} {output:.17g}'
    lines[1]=hashlib.sha256(('GPU coupled CPU reference\n'+'\n'.join(lines[2:])).encode()).hexdigest()
    return '\n'.join(lines)+'\n',dict(line_one_based=position+1,previous=previous,current=lines[position])


def stage(folder,tag=TAG):
    runtime,_=runtime_path(tag);folder=Path(folder);receipt=json.loads((folder/'preparation_receipt.json').read_text())
    deck_name=Path(receipt['origin']['original_deck']).name
    _execute(folder,deck_name,runtime,'STAGE')
    return read_records(folder/'coupled_stage_pre.bin'),read_records(folder/'coupled_stage_post.bin')


def reference(deck,destination,tag=TAG,end=.001,dt=.0001,output=.0001):
    runtime,identity=runtime_path(tag);destination=Path(destination);origin=copy_inputs(deck,destination)
    target=destination/Path(deck).name;changed,changes=change_time_controls(target.read_text(),end,dt,output);target.write_text(changed)
    _execute(destination,target.name,runtime,'TRAJECTORY')
    frames=read_trajectory(destination/'coupled_trajectory.bin')
    r.write(destination/'reference_receipt.json',dict(origin=origin,time_changes=changes,
        runtime_sha256=identity['executable_sha256'],parent_runtime_sha256=identity['parent_executable_sha256'],
        trajectory_sha256=r.sha(destination/'coupled_trajectory.bin'),steps=len(frames['times'])-1,
        final_time_s=float(frames['times'][-1]),source_formulas_unchanged=True,
        diagnostics_include='All native state and boundary ledger components AFTER diffusion',timing_claim=False))
    np.savez_compressed(destination/'trajectory.npz',**frames);return frames


def read_trajectory(path):
    with Path(path).open('rb') as stream:
        header=np.fromfile(stream,dtype='<i4',count=4)
        if len(header)!=4 or np.any(header<0):raise ValueError('Invalid coupled trajectory header')
        nc,ns,nb,nl=map(int,header);records=np.fromfile(stream,dtype='<f8')
    size=3+nc*ns+nb*nl
    if len(records)%size:raise ValueError('Truncated coupled trajectory')
    records=records.reshape(-1,size)
    if not len(records) or not np.isfinite(records).all():raise ValueError('Invalid coupled trajectory')
    return dict(times=records[:,0],dt=records[:,1],cfl=records[:,2],
        state=np.ascontiguousarray(records[:,3:3+ns*nc].reshape(-1,nc,ns).transpose(0,2,1)),
        ledger=np.ascontiguousarray(records[:,3+ns*nc:].reshape(-1,nb,nl).transpose(0,2,1)))


def verify_unmodified_reference(reference_folder,destination,parent=PARENT):
    """Prove diagnostic writes did not change physical/auxiliary checkpoints."""
    reference_folder=Path(reference_folder);destination=Path(destination);identity=verify_parent(parent)
    copy_inputs(reference_folder/'case.data',destination)
    env={k:v for k,v in os.environ.items() if not k.startswith('BRAIDS_')}
    env.update(OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OMP_WAIT_POLICY='PASSIVE',BRAIDS_TIME_INTEGRATOR='ssp43')
    result=subprocess.run([str(Path(parent)/'xbraids_candidate.exe'),'case.data'],cwd=destination,env=env,
        capture_output=True,text=True,timeout=240)
    (destination/'native.log').write_text(result.stdout+result.stderr);result.check_returncode()
    record=dict(unmodified_runtime_sha256=identity['executable_sha256'],
        checkpoint_sha256=r.sha(destination/'checkpoint.bin'),
        instrumented_checkpoint_sha256=r.sha(reference_folder/'checkpoint.bin'),
        full_state_ledger_and_auxiliary_checkpoint_bitwise_equal=r.sha(destination/'checkpoint.bin')==r.sha(reference_folder/'checkpoint.bin'),
        input_sha256=r.sha(destination/'case.data'),production_modified=False,delft_modified=False)
    r.write(destination/'equivalence.json',record)
    if not record['full_state_ledger_and_auxiliary_checkpoint_bitwise_equal']:
        raise RuntimeError('Export instrumentation changed full checkpoint')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('build','prepare','stage','reference'))
    parser.add_argument('--tag',default=TAG);parser.add_argument('--deck',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--destination',type=Path);parser.add_argument('--end',type=float,default=.001)
    parser.add_argument('--dt',type=float,default=.0001);parser.add_argument('--output',type=float,default=.0001)
    args=parser.parse_args()
    if args.action=='build':print(build(args.tag))
    else:
        if args.destination is None:parser.error('--destination required')
        if args.action=='prepare':data=prepare(args.deck,args.destination,args.tag);print(dict(cells=data['nc'],states=data['nstate'],fields=len(data)))
        elif args.action=='stage':stage(args.destination,args.tag)
        else:reference(args.deck,args.destination,args.tag,args.end,args.dt,args.output)
