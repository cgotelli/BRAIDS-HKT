"""Build an additive, current-source, initialization-only native exporter.

Generation eighteen reads the additive Q3-H3 classic prognostic-secondary
envelope over an intact Q3-C/v54 or Q3-H1/v55 foundation.  It writes geometry,
untouched initial state, forcing and exact outer identity records, then exits
unconditionally before any solver timestep.  Older initializer generations
and Delft3D stay read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tools.build_cutcell import SOURCES
from validation.gpu_fwave import coupled_reference as reference
from braids_gpu.build_gaia_initializer import normal_metadata_source
from braids_gpu.build_integrated_initializer import extra_export

GENERATION=18
GENERATION_ID=f'native_exporter_v{GENERATION}'
DESTINATION=ROOT/f'braids_gpu/{GENERATION_ID}'
CURRENT_BUILD=ROOT/'solver/cut_cells/build/build.json'
EXECUTABLE='xbraids_delft_export.exe'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source,old,new):
    if source.count(old)!=1:
        raise ValueError('Current initializer source anchor differs: '+old)
    return source.replace(old,new,1)


def transform(source):
    """Reversible insertions only, except the compulsory initialization exit.

    Unlike a diagnostic CPU oracle, this executable has no path that advances
    hydraulics, sediment or secondary flow.  The diffusion query returns after
    constructing coefficients, before the momentum-update part of its routine.
    """
    declaration='\n logical :: gpu_export_diffusion=.false.\n'
    writers=reference.WRITERS
    static=reference.static_export_source()
    static=replace_once(
        static,'  integer :: gpu_unit,c,f,g,l\n',
        '  integer :: gpu_unit,c,f,g,l\n'
        '  character(len=64) :: gpu_prognostic_secondary_method_id, &\n'
        '   gpu_prognostic_secondary_preimage_sha\n')
    static=replace_once(
        static,"  call gpu_open(gpu_unit,'coupled_input.bin')\n",
        "  gpu_prognostic_secondary_method_id='secondary.prognostic'\n"
        "  gpu_prognostic_secondary_preimage_sha=''\n"
        "  if(allocated(prognostic_secondary_execution_preimage)) &\n"
        "   call sha256_ascii(prognostic_secondary_execution_preimage, &\n"
        "    gpu_prognostic_secondary_preimage_sha)\n"
        "  call gpu_open(gpu_unit,'coupled_input.bin')\n")
    forcing_export=''.join((
        reference.rec('engelund_hansen_chezy'),
        reference.rec('native_hydraulic_roughness_mode',
                      'hydraulic_roughness_mode',kind='i'),
        reference.rec('native_white_colebrook_chzk2d',
                      'merge(1,0,white_colebrook_chzk2d)',kind='i'),
        reference.rec('native_sediment_shear_mode',
                      'sediment_shear_mode',kind='i'),
        reference.rec('native_roughness_minimum_work_depth_m',
                      'roughness_minimum_work_depth'),
        reference.rec('native_skin_kinematic_viscosity_m2_s',
                      'skin_kinematic_viscosity'),
        reference.rec('native_skin_sand_roughness_m',
                      'skin_sand_roughness'),
        reference.rec('native_skin_silt_roughness_m',
                      'skin_silt_roughness'),
        reference.rec('native_skin_critical_mud_fraction_low',
                      'skin_critical_mud_fraction_low'),
        reference.rec('native_skin_critical_mud_fraction_high',
                      'skin_critical_mud_fraction_high'),
        reference.rec('native_skin_fluff_fraction','skin_fluff_fraction'),
        reference.rec('native_q3c_forcing_verified',
                      'merge(1,0,capacity_foundation_forcing_identity_declaration)',kind='i'),
        reference.rec('native_q3c_forcing_envelope','54',kind='i'),
        reference.rec('native_q3c_formula_code','bedlaw',kind='i'),
        reference.rec('native_q3c_method_id_ascii',
                      '[(iachar(capacity_route_method_id(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3c_execution_fingerprint_ascii',
                      '[(iachar(capacity_route_execution_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3c_profile_fingerprint_ascii',
                      '[(iachar(capacity_route_profile_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3c_forcing_fingerprint_ascii',
                      '[(iachar(capacity_foundation_forcing_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3c_raw_roughness_sha256_ascii',
                      '[(iachar(capacity_foundation_forcing_raw_sha256(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h1_identity_verified',
                      'merge(1,0,hiding_foundation_identity_declaration)',kind='i'),
        reference.rec('native_q3h1_identity_envelope','55',kind='i'),
        reference.rec('native_q3h1_formula_code','bedlaw',kind='i'),
        reference.rec('native_q3h1_hiding_mode','hiding_exposure',kind='i'),
        reference.rec('native_q3h1_parker_alpha','hiding_parker_alpha'),
        reference.rec('native_q3h1_wu_exponent','hiding_wu_exponent'),
        reference.rec('native_q3h1_execution_fingerprint_ascii',
                      '[(iachar(hiding_foundation_execution_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_secondary_verified',
                      'merge(1,0,prognostic_secondary_identity_declaration)',kind='i'),
        reference.rec('native_q3h3_secondary_envelope','57',kind='i'),
        reference.rec('native_q3h3_model','2',kind='i'),
        reference.rec('native_q3h3_base_envelope',
                      'prognostic_secondary_base_envelope',kind='i'),
        reference.rec('native_q3h3_execution_fingerprint_ascii',
                      '[(iachar(prognostic_secondary_execution_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_capacity_execution_fingerprint_ascii',
                      '[(iachar(capacity_route_execution_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_capacity_profile_fingerprint_ascii',
                      '[(iachar(capacity_route_profile_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_hiding_execution_fingerprint_ascii',
                      '[(iachar(hiding_foundation_execution_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_roughness_fingerprint_ascii',
                      '[(iachar(capacity_foundation_forcing_fingerprint(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_method_id_ascii',
                      '[(iachar(gpu_prognostic_secondary_method_id(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_canonical_preimage_sha256_ascii',
                      '[(iachar(gpu_prognostic_secondary_preimage_sha(c:c)),c=1,64)]',('64',),kind='i'),
        reference.rec('native_q3h3_rmincf','prognostic_secondary_rmincf'),
        reference.rec('native_q3h3_chzmin','prognostic_secondary_chzmin'),
        reference.rec('native_q3h3_espir','prognostic_secondary_espir'),
        reference.rec('native_q3h3_coriolis_constant_s_inverse',
                      'prognostic_secondary_coriolis'),
        reference.rec('native_q3h3_betac','prognostic_secondary_betac'),
        reference.rec('native_q3h3_scalar_diffusivity_m2_s',
                      'prognostic_secondary_scalar_diffusivity'),
        reference.rec('native_q3h3_von_karman_constant',
                      'prognostic_secondary_von_karman'),
        reference.rec('native_q3h3_transport_direction',
                      'merge(1,0,secondary_flow_transport_direction)',kind='i'),
        reference.rec('native_q3h3_inflow_mode',
                      'secondary_flow_inflow_mode',kind='i'),
    ))
    static=replace_once(static,'  call gpu_finish(gpu_unit)\n',
                        forcing_export+'  call gpu_finish(gpu_unit)\n')
    additions=writers+static+normal_metadata_source()+extra_export()
    exit_block=(
                " if(.not.component_native_identity_declaration.or. &\n"
                "    .not.capacity_foundation_runtime_declaration.or. &\n"
                "    .not.capacity_foundation_route_identity_declaration.or. &\n"
                "    .not.capacity_foundation_forcing_identity_declaration.or. &\n"
                "    .not.prognostic_secondary_identity_declaration.or. &\n"
                "    secondary_foundation_identity_declaration.or. &\n"
                "    all(prognostic_secondary_base_envelope/=[54,55]).or. &\n"
                "    all(bedlaw/=[braids_capacity_mpm,braids_capacity_engelund_hansen, &\n"
                "                 braids_capacity_general]).or. &\n"
                "    .not.secondary_flow.or.secondary_flow_model/=2.or. &\n"
                "    .not.secondary_flow_transport_direction.or. &\n"
                "    secondary_flow_inflow_mode/=1.or.bed_slope/=0.or.exchange_mode/=0) &\n"
                "  error stop 'v18 exporter requires an exact Q3-H3 identity over Q3-C/Q3-H1'\n"
                " call gpu_export_input()\n call normal_export_metadata()\n"
                " if(general_avalanche) call integrated_export_avalanche_metadata()\n"
                " stop 'GPU native initialization exported; no solver timestep performed'\n")
    diff_declarations=('  integer :: gpu_unit,gpu_stencil(max_diffusion_recovery_stencil),gpu_nstencil\n'
                       '  integer,allocatable :: gpu_stencils(:,:),gpu_stencil_counts(:)\n'
                       '  logical :: gpu_complete\n')
    diff_export=reference.diffusion_export_block()
    marker='\ncontains\n';at=source.index(marker)
    transformed=source[:at]+declaration+source[at:]
    transformed=replace_once(transformed,'end module\n\nprogram run_cutcell',
                             additions+'end module\n\nprogram run_cutcell')
    transformed=replace_once(transformed,' call cpu_time(started)',
                             exit_block+' call cpu_time(started)')
    begin=transformed.index(' subroutine diffuse_momentum_nonorthogonal(')
    end=transformed.index(' end subroutine',begin)
    block=transformed[begin:end]
    block=replace_once(block,'  if(horizontal_viscosity<=0.or.dt<=0) return',
                       diff_declarations+'  if(horizontal_viscosity<=0.or.dt<=0) return')
    block=replace_once(block,'  base_step=dt',diff_export+'  base_step=dt')
    transformed=transformed[:begin]+block+transformed[end:]
    restored=transformed
    for insertion in (declaration,additions,exit_block,diff_declarations,diff_export):
        if restored.count(insertion)!=1:
            raise ValueError('Initializer inserted block is not uniquely reversible')
        restored=restored.replace(insertion,'',1)
    if restored!=source:
        raise ValueError('Initializer transformation altered native numerical source')
    return transformed


def frozen_identities():
    identities={}
    for generation in range(1,GENERATION):
        folder=ROOT/f'braids_gpu/native_exporter_v{generation}'
        receipt=folder/'build.json'
        if not receipt.is_file():continue
        build=json.loads(receipt.read_text(encoding='utf-8'))
        executables=list(folder.glob('*.exe'))
        identities[folder.name]=dict(build_sha256=sha256(receipt),
            executable_sha256={path.name:sha256(path) for path in executables})
        expected=build.get('executable_sha256')
        if expected and expected not in identities[folder.name]['executable_sha256'].values():
            raise ValueError('Historical initializer executable differs: '+folder.name)
    return identities


def build():
    if DESTINATION.exists():
        raise FileExistsError('Preserve existing '+GENERATION_ID+': '+str(DESTINATION))
    frozen=frozen_identities()
    production=json.loads(CURRENT_BUILD.read_text(encoding='utf-8'))
    hashes={name:sha256(ROOT/name) for name in SOURCES}
    if production.get('source_sha256')!=hashes:
        raise ValueError('Rebuild current production native sources before creating '+GENERATION_ID)
    parent=ROOT/'solver/cut_cells/build/xbraids_cutcell.exe'
    if sha256(parent)!=production.get('executable_sha256'):
        raise ValueError('Current production executable differs from its receipt')
    compiler=shutil.which(production['command'][0])
    if compiler is None:raise FileNotFoundError('Current native compiler is unavailable')
    source=ROOT/'solver/cut_cells/cutcell_solver.f90'
    exported=transform(source.read_text(encoding='utf-8'))
    DESTINATION.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        original=ROOT/name;target=DESTINATION/original.name
        if original.name=='cutcell_solver.f90':
            target.write_text(exported,encoding='utf-8')
        else:shutil.copy2(original,target)
    executable=DESTINATION/EXECUTABLE
    command=[compiler,'-O3','-flto','-fopenmp','-ffree-line-length-none',
             '-ffixed-line-length-none',*[str(DESTINATION/Path(name).name) for name in SOURCES],
             '-o',str(executable)]
    with (DESTINATION/'build.log').open('x',encoding='utf-8') as log:
        subprocess.run(command,cwd=DESTINATION,stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=120)
    if hashes!={name:sha256(ROOT/name) for name in SOURCES}:
        raise RuntimeError(
            'Production sources changed during '+GENERATION_ID+' initializer build')
    if frozen!=frozen_identities():
        raise RuntimeError('Historical initializer identity changed during '+GENERATION_ID+' build')
    helpers=(Path(__file__).resolve(),Path(reference.__file__).resolve(),
             ROOT/'braids_gpu/build_gaia_initializer.py',ROOT/'braids_gpu/build_integrated_initializer.py')
    snapshot={Path(name).name:sha256(DESTINATION/Path(name).name) for name in SOURCES}
    receipt=dict(schema=7,generation=GENERATION_ID,command=command,
        executable=str(executable),executable_sha256=sha256(executable),
        compiler_sha256=sha256(compiler),
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        driver_sha256=sha256(__file__),
        transform_helpers_sha256={path.relative_to(ROOT).as_posix():sha256(path) for path in helpers},
        production_build_sha256=sha256(CURRENT_BUILD),
        production_executable_sha256=production['executable_sha256'],
        parent_executable_sha256=production['executable_sha256'],
        current_source_sha256=hashes,snapshot_source_sha256=snapshot,source_sha256=snapshot,
        numerical_source_unchanged=True,numerical_source_unchanged_by_instrumentation=True,
        instrumentation_reversibility='Removing five uniquely authenticated inserted blocks restores source exactly',
        instrumentation_scope=(
            'Exact authenticated v57 Q3-H3 classic prognostic-secondary with '
            'laws 2/8/11 over an intact v54/v55 foundation; initial physical '
            'state, geometry, forcing, outer identity and diffusion I/O; '
            'unconditional zero-step exit'),
        initializer_scope_extension=(
            'Delft3D hydraulic C/M/W roughness, total/Soulsby-Clarke sediment '
            'shear, authenticated v54 scalar-capacity laws 2/11 with raw-FP64 '
            'forcing, additive v55 hiding, and additive v57 classic prognostic secondary identity'),
        exported_native_controls=['engelund_hansen_chezy',
            'hydraulic_roughness_mode','white_colebrook_chzk2d',
            'sediment_shear_mode','roughness_minimum_work_depth',
            'skin_kinematic_viscosity','skin_sand_roughness',
            'skin_silt_roughness','skin_critical_mud_fraction_low',
            'skin_critical_mud_fraction_high','skin_fluff_fraction',
            'native_q3c_forcing_verified','native_q3c_forcing_envelope',
            'native_q3c_formula_code','native_q3c_method_id_ascii',
            'native_q3c_execution_fingerprint_ascii',
            'native_q3c_profile_fingerprint_ascii',
            'native_q3c_forcing_fingerprint_ascii',
            'native_q3c_raw_roughness_sha256_ascii',
            'native_q3h1_identity_verified','native_q3h1_identity_envelope',
            'native_q3h1_formula_code','native_q3h1_hiding_mode',
            'native_q3h1_parker_alpha','native_q3h1_wu_exponent',
            'native_q3h1_execution_fingerprint_ascii',
            'native_q3h3_secondary_verified','native_q3h3_secondary_envelope',
            'native_q3h3_model','native_q3h3_base_envelope',
            'native_q3h3_execution_fingerprint_ascii',
            'native_q3h3_capacity_execution_fingerprint_ascii',
            'native_q3h3_capacity_profile_fingerprint_ascii',
            'native_q3h3_hiding_execution_fingerprint_ascii',
            'native_q3h3_roughness_fingerprint_ascii',
            'native_q3h3_method_id_ascii',
            'native_q3h3_canonical_preimage_sha256_ascii',
            'native_q3h3_rmincf','native_q3h3_chzmin','native_q3h3_espir',
            'native_q3h3_coriolis_constant_s_inverse','native_q3h3_betac',
            'native_q3h3_scalar_diffusivity_m2_s',
            'native_q3h3_von_karman_constant',
            'native_q3h3_transport_direction','native_q3h3_inflow_mode'],
        native_solver_steps_performed=0,unconditional_initialization_exit=True,
        frozen_initializer_identities=frozen,frozen_initializers_modified=False,
        frozen_qualification_relabelled=False,Delft3D_source_modified=False)
    (DESTINATION/'build.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(executable=str(executable),sha256=receipt['executable_sha256'])))
    return receipt


if __name__=='__main__':build()
