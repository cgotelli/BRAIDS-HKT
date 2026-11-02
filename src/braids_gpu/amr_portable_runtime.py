"""Authenticated package-relative initializer and CUDA bundle for bank AMR.

No compiler, validation input, CPU trajectory, QGIS or GDAL is needed at runtime.
The frozen native generator is unchanged; its schema-1 verifier also accepts the
sanitized build receipt because all executable/source identity fields remain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


VARIANT = 'on_demand_bank_amr_ssp33'
GENERAL_VARIANT = 'on_demand_transport_amr_ssp33'
RUNTIME_RELATIVE = 'braids_gpu/native_amr_v1'
LIVE_RUNTIME_RELATIVE = 'braids_gpu/native_amr_live_v1'
BEDFORM_BANK_RUNTIME_RELATIVE = 'braids_gpu/native_amr_bedform_bank_v3'
BEDFORM_PHYSICAL_TOE_RUNTIME_RELATIVE = \
    'braids_gpu/native_amr_bedform_physical_toe_v1'
GENERAL_RUNTIME_RELATIVE = 'braids_gpu/native_amr_general_v1'
KERNEL_RELATIVE = 'braids_gpu/frozen_sources/on_demand_amr_v9'
GENERAL_KERNEL_RELATIVE = 'braids_gpu/frozen_sources/on_demand_transport_amr_v8'
GENERAL_SOURCE_ARCHIVE_REVISION = dict(
    schema=1,
    archive=GENERAL_KERNEL_RELATIVE,
    parent_archive='braids_gpu/frozen_sources/on_demand_transport_amr_v7',
    parent_runtime_manifest_sha256=
        '9cd1e5bb729bc5fe4ae0e83a3e7aa1d3c36b0c068ef8bfafac18a4c00e658b5d',
    qualified_source_manifest_sha256=
        '549b0154fb2d88685bc3609c0f3a019f781a0e658fc9d062841ab8fd201b6e5c',
    route_receipt=dict(
        path='braids_support/g5_consolidated/runs/regrid_fast_path_v12/RESULT.json',
        sha256='4ef67eae6e626035a63ef5a13de6600cd87cf79a033428e98d0417ff6d467d9f'),
    component_receipt=dict(
        path=('braids_support/sediment/delft3d_current/bedload_direction_slope/'
              'reports/stress_ratio_cpu_gpu.json'),
        sha256='5098ec46dfb043cbf0ff870e1094b17f9dedadeb97f35500a29cf7292bf08594'),
    numerical_source_revisions={
        'sediment_transport.cu':dict(
            parent_sha256=
                'd0fe3cffad597b3eeec2575b35eaba825f818a23775db64e552cb1c8c467faf7',
            sha256=
                '77636e48b5fa52a22e15bee1bbf89be1ea1ca9a8b47ca36a3b914a237936b61a',
            change='MPM transverse-slope attenuation uses tau/tau_cr')})
RECURSIVE_GENERAL_VISCOSITY_CONTRACT = dict(
    schema=1,policy='parent_post_reflux_ssprk2_once_per_macro_v1',
    placement='external_parent_post_reflux_before_avalanche',
    calls_per_macro=1,
    integrator='SSPRK2_subcycled_area_weighted_velocity_L2_limited',
    limited_norm='area_weighted_velocity_L2',
    area_weighted_velocity_l2_nonincrease=True,
    pairwise_velocity_face_flux=True,
    momentum_conservation='variable_depth_not_exact',
    dynamic_wet_geometry=True,general_avalanche_modes=[0,1],
    composition_order='viscosity_then_avalanche',bank_mode=0,
    secondary_flow_model=2,boundary_modes=[9,4],viscosity_mode=1,
    viscosity_geometry=1,formulation='velocity_laplacian',
    checkpoint_schema=6)
RECURSIVE_GENERAL_BEDFORM_CONTRACT = dict(
    schema=1,policy='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,bank_mode=0,
    boundary_mode_counts={'1':1,'2':1},bedform_modes=[0,1],unsampled_zero_offset=True,
    scheduled_forcing=False,bedlaw=2,mpm_friction_mode=1,bed_slope=0,
    hiding_exposure=0,exchange_mode=0,secondary_flow_model=0,
    bedload_adaptation=0,horizontal_viscosity=0.,general_avalanche=0,
    morphological_factor=1.,nsubstrate_layers=1,checkpoint_schema=6)
RECURSIVE_BANK_BEDFORM_AVALANCHE_CONTRACT = dict(
    schema=1,policy='recursive_bedform_mode3_parent_avalanche_v1',
    bedform_order='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    bank_transaction='distributed_mode3_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    calls_per_macro={'bank':1,'avalanche':1},bank_mode=3,
    evolving_banks=True,boundary_mode_counts={'2':1},
    unsampled_zero_offset=True,scheduled_forcing=False,bank_failure=1,
    bedform_roughness=1,general_avalanche=1,bedlaw=2,
    mpm_friction_mode=1,bed_slope=0,hiding_exposure=0,exchange_mode=0,
    secondary_flow_model=0,bedload_adaptation=0,horizontal_viscosity=0.,
    morphological_factor=1.,nsubstrate_layers=1,
    checkpoint_kind='recursive_bank_amr',checkpoint_schema=1,frame_schema=7)
RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_CONTRACT = dict(
    schema=1,policy='recursive_bedform_mode2_physical_toe_parent_avalanche_v1',
    bedform_order='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    accepted_flux_owner='completed_root_exner_register_v1',
    bank_transaction=
        'physical_toe_supply_undercut_collapse_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    mapping_transaction='reconcile_then_evolving_toe_rebuild_before_commit_v1',
    calls_per_macro={'bank':1,'avalanche':1},bank_mode=2,
    evolving_banks=True,toe_geometry_seed_only=True,
    boundary_mode_counts={'2':1},unsampled_zero_offset=True,
    scheduled_forcing=False,bank_failure=1,bedform_roughness=1,
    general_avalanche=1,bedlaw=2,mpm_friction_mode=1,bed_slope=0,
    hiding_exposure=0,exchange_mode=0,secondary_flow_model=0,
    bedload_adaptation=0,horizontal_viscosity=0.,morphological_factor=1.,
    nsubstrate_layers=1,checkpoint_kind='recursive_bank_amr',
    checkpoint_schema=2,frame_schema=8)
EXPORTER_SHA256 = '98806f1cb766ea4b5bdeefcd86d3c343a3bbee1f1fe255cd070c57b73048ef9a'
EXECUTABLE = 'xbraids_amr_geometry.exe'
GENERAL_EXECUTABLE = 'xbraids_amr_general_geometry.exe'
GENERAL_EXPORTER_SHA256 = '9179b7356863652cdcf4754ae2b87ee2fcf54a6706db6b3dd64b8b25056ad356'
INTERFACE_REVISION = 'inventory_bounded_secondary_outlet_pair_v7'
REVISED_KERNEL_ABIS = {
    'toe_supply.cu':'#define BRAIDS_TOE_INTERFACE_ABI 4',
    'bank_undercut.cu':'#define BRAIDS_UNDERCUT_INTERFACE_ABI 3',
    'toe_face_diagnostic.cu':'#define BRAIDS_TOE_DIAGNOSTIC_ABI 2',
    'live_normal_section.cu':'#define BRAIDS_NORMAL_ENDPOINT_ROOT_ABI 2',
    'sediment_transport.cu':'#define BRAIDS_SEDIMENT_STAGE_GUARD_ABI 1',
    'sediment_stratigraphy.cu':'#define BRAIDS_STRATIGRAPHY_EXHUMATION_POLICY 3',
    'secondary_transport.cu':'#define BRAIDS_SECONDARY_OUTLET_ABI 1',
}
PAIRED_PYTHON = ('toe_supply.py','bank_undercut.py','toe_face_diagnostic.py','coupled_solver.py',
    'live_normal_section.py','live_normal_boundary.py','live_critical_section.py',
    'normal_outlet_boundary.py','sediment_transport.py','sediment_stratigraphy.py','secondary_transport.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def interface_source_pair(root, kernels):
    """Authenticate the revised Python/CUDA ABI without importing GPU modules.

    ``kernels`` explicitly maps every revised interface name to its file.
    Call before copying/launching. Original qualified archives are never edited;
    a fresh archive records these intentional changes and their complete hashes.
    This contract establishes source pairing, not numerical AMR qualification.
    """
    root=Path(root).resolve()
    require(set(kernels)==set(REVISED_KERNEL_ABIS),'AMR revised kernel inventory differs')
    kernel_hashes={}
    for name,marker in REVISED_KERNEL_ABIS.items():
        path=Path(kernels[name])
        require(path.is_file() and not path.is_symlink(),
            'Missing or linked AMR interface kernel: '+name)
        lines=path.read_text(encoding='utf-8').splitlines()
        require(sum(line.strip()==marker for line in lines)==1,
            'AMR Python/CUDA interface ABI mismatch: '+name)
        kernel_hashes[name]=sha(path)
    python_hashes={}
    for name in PAIRED_PYTHON:
        relative='braids_support/gpu_fwave/'+name
        path=root/relative
        require(path.is_file() and not path.is_symlink(),
            'Missing or linked AMR interface Python: '+name)
        if name in ('toe_supply.py','bank_undercut.py','toe_face_diagnostic.py',
                    'sediment_transport.py','sediment_stratigraphy.py','secondary_transport.py'):
            require(REVISED_KERNEL_ABIS[name.replace('.py','.cu')] in
                path.read_text(encoding='utf-8'),
                'AMR Python lacks current-interface ABI guard: '+name)
        python_hashes[relative]=sha(path)
    return dict(schema=1,revision=INTERFACE_REVISION,
        required_cuda_abi=dict(REVISED_KERNEL_ABIS),
        python_sha256=python_hashes,cuda_sha256=kernel_hashes)


def verify_interface_source_pair(root,kernels,receipt):
    measured=interface_source_pair(root,kernels)
    require(receipt==measured,'AMR Python/CUDA source pair differs')
    return measured


def current_interface_kernel_sources(root,base):
    """Select declared current revisions over an immutable baseline in memory."""
    root,base=Path(root).resolve(),Path(base).resolve()
    files={path.name:path for path in base.glob('*.cu')}
    require(set(REVISED_KERNEL_ABIS)<=set(files),
        'AMR baseline lacks the complete revised-kernel inventory')
    revisions={name:root/'braids_support/gpu_fwave'/name for name in REVISED_KERNEL_ABIS}
    pair=interface_source_pair(root,revisions)
    files.update(revisions)
    return files,pair


def portable_build_receipt(runtime):
    """Packaging-time conversion; never changes the original build receipt."""
    from .amr_native_geometry import verify_runtime
    runtime = Path(runtime)
    original = verify_runtime(runtime)
    require(original['executable_sha256'] == EXPORTER_SHA256, 'Unqualified AMR initializer')
    fields = ('schema','executable','executable_sha256','parent_executable_sha256',
        'compiler_sha256','generator_sha256','source_sha256','export_only',
        'numerical_routines_unchanged','source_original_modified','Delft3D_source_modified')
    result = {key:original[key] for key in fields}
    result.update(portable=True, original_build_receipt_sha256=sha(runtime/'build.json'),
        historical_compiler_and_parent_runtime_not_bundled=True)
    return result


def portable_live_build_receipt(runtime):
    """Keep the new export identity, without historical machine paths."""
    from .amr_on_demand import verify_runtime
    runtime=Path(runtime)
    original=verify_runtime(runtime)
    fields=('schema','kind','live_boundary_metadata','executable','executable_sha256',
        'parent_executable_sha256','compiler_sha256','generator_sha256',
        'generator_dependencies','source_sha256','export_only','numerical_routines_unchanged',
        'source_original_modified','Delft3D_source_modified')
    result={key:original[key] for key in fields}
    for key in ('toe_geometry_seed_only', 'selected_toe_initialization_policy',
                'evolving_toe_geometry',
                'geometry_initialization_version','bank_metadata_contract','bank_modes',
                'recursive_bank_bedform_avalanche','parser_scope_revision'):
        if key in original:
            result[key] = original[key]
    result.update(portable=True,original_build_receipt_sha256=sha(runtime/'build.json'),
                  historical_compiler_and_parent_runtime_not_bundled=True)
    return result


def portable_general_build_receipt(runtime):
    """Sanitize the versioned nonbank zero-step initializer provenance."""
    from .amr_general_native_geometry import KIND, verify_runtime
    runtime = Path(runtime)
    original = verify_runtime(runtime)
    require(original['kind'] == KIND and
        original['executable'] == GENERAL_EXECUTABLE and
        original['executable_sha256'] == GENERAL_EXPORTER_SHA256 and
        original['export_only'] is True and
        original['numerical_routines_unchanged'] is True and
        original['source_original_modified'] is False and
        original['Delft3D_source_modified'] is False,
        'Unqualified general AMR initializer')
    fields = ('schema', 'kind', 'executable', 'executable_sha256',
        'parent_executable_sha256', 'compiler_sha256', 'generator_sha256',
        'generator_dependencies', 'source_sha256', 'export_only',
        'numerical_routines_unchanged', 'source_original_modified',
        'Delft3D_source_modified')
    result = {key: original[key] for key in fields}
    result.update(portable=True,
        original_build_receipt_sha256=sha(runtime/'build.json'),
        historical_compiler_and_parent_runtime_not_bundled=True)
    return result


def _member(root,name):
    require(isinstance(name,str) and bool(name) and '\\' not in name and ':' not in name,
        'Unsafe packaged AMR member')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.as_posix() == name and
        not any(part in ('.','..','') for part in relative.parts), 'Unsafe packaged AMR path')
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file() and
        not path.is_symlink(), 'Missing or linked packaged AMR member: '+name)
    return path


def verify_packaged_runtime(root=None):
    """Verify both the native dependency closure and frozen CUDA bytes."""
    from .amr_native_geometry import verify_runtime
    from .amr_evolving_runtime import KERNEL_ABIS,verify as verify_evolving_pair
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    package = json.loads((root/'PACKAGE-MANIFEST.json').read_text())
    files = package.get('files')
    scope = package.get('source_worker_scope',{})
    require(package.get('kind') == 'experimental Windows source worker' and isinstance(files,dict)
        and VARIANT in scope.get('supported_gpu_variants',[])
        and scope.get('native_exporters_by_variant',{}).get(VARIANT) == RUNTIME_RELATIVE
        and scope.get('source_archives_by_variant',{}).get(VARIANT) == KERNEL_RELATIVE
        and scope.get('recursive_bank_bedform_avalanche_by_variant') == {
            VARIANT:RECURSIVE_BANK_BEDFORM_AVALANCHE_CONTRACT}
        and scope.get('recursive_bank_bedform_native_exporters_by_variant') == {
            VARIANT:BEDFORM_BANK_RUNTIME_RELATIVE},
        'Packaged AMR capability/runtime contract differs')
    toe_contract=scope.get(
        'recursive_physical_toe_bedform_avalanche_by_variant')
    toe_exporter=scope.get(
        'recursive_physical_toe_bedform_native_exporters_by_variant')
    require((toe_contract is None and toe_exporter is None) or
            (toe_contract=={
                VARIANT:RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_CONTRACT} and
             toe_exporter=={VARIANT:BEDFORM_PHYSICAL_TOE_RUNTIME_RELATIVE}),
            'Packaged physical-toe bedform capability/runtime contract differs')
    for prefix in (RUNTIME_RELATIVE,KERNEL_RELATIVE):
        folder = root/prefix
        require(folder.is_dir() and not folder.is_symlink(), 'Missing packaged AMR payload')
        declared = {name:value for name,value in files.items() if name.startswith(prefix+'/')}
        actual = {path.relative_to(root).as_posix() for path in folder.rglob('*') if path.is_file()}
        require(bool(declared) and set(declared) == actual and
            not any(path.is_symlink() for path in folder.rglob('*')),
            'Packaged AMR payload inventory differs')
        for name,value in declared.items():
            require(sha(_member(root,name)) == value, 'Packaged AMR payload changed: '+name)
    runtime = root/RUNTIME_RELATIVE
    build = verify_runtime(runtime)
    provenance = package.get('provenance',{}).get('on_demand_amr',{})
    require(build.get('portable') is True and build['executable'] == EXECUTABLE and
        build['executable_sha256'] == EXPORTER_SHA256 and
        files.get(RUNTIME_RELATIVE+'/'+EXECUTABLE) == EXPORTER_SHA256,
        'Packaged AMR initializer identity differs')
    expected_dlls = provenance.get('dll_sha256',{})
    require({'libgfortran-5.dll','libgomp-1.dll'} <= set(expected_dlls) and
        {p.name for p in runtime.glob('*.dll')} == set(expected_dlls) and
        all(files.get(RUNTIME_RELATIVE+'/'+name) == value for name,value in expected_dlls.items()),
        'Packaged AMR initializer DLL closure differs')
    kernels = root/KERNEL_RELATIVE
    kernel_manifest = json.loads((kernels/'runtime_manifest.json').read_text())
    require(kernel_manifest.get('schema') == 2 and kernel_manifest.get('variant') == VARIANT and
        kernel_manifest.get('numerical_cuda_unchanged') is False and
        kernel_manifest.get('intentional_cuda_revisions') == sorted(REVISED_KERNEL_ABIS) and
        kernel_manifest.get('added_cuda_kernels') == sorted(KERNEL_ABIS) and
        isinstance(kernel_manifest.get('source_files'),dict), 'Packaged AMR kernel manifest differs')
    declared_kernels = kernel_manifest['source_files']
    parent_kernels = kernel_manifest.get('parent_cuda_sha256')
    require(isinstance(parent_kernels,dict) and set(REVISED_KERNEL_ABIS)<=set(parent_kernels)
        and not set(parent_kernels)&set(KERNEL_ABIS)
        and set(declared_kernels)==set(parent_kernels)|set(KERNEL_ABIS)
        and all(declared_kernels[name]==digest for name,digest in parent_kernels.items()
                if name not in REVISED_KERNEL_ABIS),
        'Packaged AMR baseline/added kernel inventory differs')
    require({p.name for p in kernels.iterdir()} == set(declared_kernels)|{'runtime_manifest.json'} and
        all(Path(name).name == name and name.endswith('.cu') and sha(kernels/name) == value
            for name,value in declared_kernels.items()), 'Packaged AMR kernel bytes differ')
    pair=verify_interface_source_pair(root,
        {name:kernels/name for name in REVISED_KERNEL_ABIS},
        kernel_manifest.get('interface_source_pair'))
    evolving_pair=kernel_manifest.get('evolving_interface_source_pair')
    verify_evolving_pair(root,{name:kernels/name for name in KERNEL_ABIS},evolving_pair)
    result=dict(runtime=runtime, kernels=kernels, build=build,
        interface_source_pair=pair,evolving_interface_source_pair=evolving_pair,
        package_manifest_sha256=sha(root/'PACKAGE-MANIFEST.json'))
    live_policy=scope.get('evolving_native_exporters_by_variant',{})
    if live_policy:
        from .amr_on_demand import verify_runtime as verify_live
        require(live_policy=={VARIANT:LIVE_RUNTIME_RELATIVE},
                'Packaged live AMR initializer policy differs')
        live=root/LIVE_RUNTIME_RELATIVE
        declared={name:value for name,value in files.items()
                  if name.startswith(LIVE_RUNTIME_RELATIVE+'/')}
        actual={p.relative_to(root).as_posix() for p in live.rglob('*') if p.is_file()}
        require(live.is_dir() and not live.is_symlink() and bool(declared) and
                set(declared)==actual and not any(p.is_symlink() for p in live.rglob('*')) and
                all(sha(_member(root,name))==value for name,value in declared.items()),
                'Packaged live AMR payload inventory/bytes differ')
        live_build=verify_live(live)
        provenance=package.get('provenance',{}).get('on_demand_amr_live',{})
        dlls=provenance.get('dll_sha256',{})
        require(live_build.get('portable') is True and
                live_build['executable_sha256']==provenance.get('executable_sha256') and
                {'libgfortran-5.dll','libgomp-1.dll'}<=set(dlls) and
                {p.name for p in live.glob('*.dll')}==set(dlls) and
                all(files.get(LIVE_RUNTIME_RELATIVE+'/'+name)==value for name,value in dlls.items()),
                'Packaged live AMR initializer provenance differs')
        result.update(live_runtime=live,live_build=live_build)
    exact_policy=scope.get('recursive_bank_bedform_native_exporters_by_variant',{})
    if exact_policy:
        from .amr_on_demand import verify_runtime as verify_exact
        require(exact_policy=={VARIANT:BEDFORM_BANK_RUNTIME_RELATIVE},
                'Packaged bedform-bank AMR initializer policy differs')
        exact=root/BEDFORM_BANK_RUNTIME_RELATIVE
        declared={name:value for name,value in files.items()
                  if name.startswith(BEDFORM_BANK_RUNTIME_RELATIVE+'/')}
        actual={p.relative_to(root).as_posix() for p in exact.rglob('*')
                if p.is_file()}
        require(exact.is_dir() and not exact.is_symlink() and bool(declared) and
                set(declared)==actual and
                not any(p.is_symlink() for p in exact.rglob('*')) and
                all(sha(_member(root,name))==value
                    for name,value in declared.items()),
                'Packaged bedform-bank AMR initializer inventory/bytes differ')
        exact_build=verify_exact(exact)
        provenance=package.get('provenance',{}).get(
            'on_demand_amr_bedform_bank',{})
        dlls=provenance.get('dll_sha256',{})
        require(exact_build.get('portable') is True and
                exact_build.get('recursive_bank_bedform_avalanche') is True and
                exact_build.get('bank_modes')==[3] and
                exact_build.get('bank_metadata_contract')==
                    'recursive_bedform_mode3_parent_avalanche_v1' and
                exact_build['executable_sha256']==
                    provenance.get('executable_sha256') and
                {'libgfortran-5.dll','libgomp-1.dll'}<=set(dlls) and
                {p.name for p in exact.glob('*.dll')}==set(dlls) and
                all(files.get(BEDFORM_BANK_RUNTIME_RELATIVE+'/'+name)==value
                    for name,value in dlls.items()),
                'Packaged bedform-bank AMR initializer provenance differs')
        result.update(bedform_bank_runtime=exact,
                      bedform_bank_build=exact_build)
    toe_policy=scope.get(
        'recursive_physical_toe_bedform_native_exporters_by_variant',{})
    if toe_policy:
        from .amr_on_demand import verify_runtime as verify_exact
        require(toe_policy=={VARIANT:BEDFORM_PHYSICAL_TOE_RUNTIME_RELATIVE},
                'Packaged physical-toe bedform AMR initializer policy differs')
        exact=root/BEDFORM_PHYSICAL_TOE_RUNTIME_RELATIVE
        declared={name:value for name,value in files.items()
                  if name.startswith(BEDFORM_PHYSICAL_TOE_RUNTIME_RELATIVE+'/')}
        actual={p.relative_to(root).as_posix() for p in exact.rglob('*')
                if p.is_file()}
        require(exact.is_dir() and not exact.is_symlink() and bool(declared) and
                set(declared)==actual and
                not any(p.is_symlink() for p in exact.rglob('*')) and
                all(sha(_member(root,name))==value
                    for name,value in declared.items()),
                'Packaged physical-toe bedform AMR initializer inventory/bytes differ')
        exact_build=verify_exact(exact)
        provenance=package.get('provenance',{}).get(
            'on_demand_amr_bedform_physical_toe',{})
        dlls=provenance.get('dll_sha256',{})
        require(exact_build.get('portable') is True and
                exact_build.get('recursive_bank_bedform_avalanche') is True and
                exact_build.get('bank_modes')==[2] and
                exact_build.get('toe_geometry_seed_only') is True and
                exact_build.get('bank_metadata_contract')==
                    'recursive_bedform_mode2_physical_toe_parent_avalanche_v1' and
                exact_build['executable_sha256']==
                    provenance.get('executable_sha256') and
                {'libgfortran-5.dll','libgomp-1.dll'}<=set(dlls) and
                {p.name for p in exact.glob('*.dll')}==set(dlls) and
                all(files.get(BEDFORM_PHYSICAL_TOE_RUNTIME_RELATIVE+'/'+name)==value
                    for name,value in dlls.items()),
                'Packaged physical-toe bedform AMR initializer provenance differs')
        result.update(bedform_physical_toe_runtime=exact,
                      bedform_physical_toe_build=exact_build)
    if GENERAL_VARIANT in scope.get('supported_gpu_variants', []):
        from .amr_general_native_geometry import verify_runtime as verify_general
        require(scope.get('native_exporters_by_variant', {}).get(GENERAL_VARIANT)
            == GENERAL_RUNTIME_RELATIVE,
            'Packaged general AMR initializer policy differs')
        require(scope.get('source_archives_by_variant', {}).get(GENERAL_VARIANT)
            == GENERAL_KERNEL_RELATIVE,
            'Packaged general AMR CUDA source policy differs')
        require(scope.get('recursive_general_viscosity_by_variant') == {
            GENERAL_VARIANT:RECURSIVE_GENERAL_VISCOSITY_CONTRACT},
            'Packaged recursive general-viscosity capability differs')
        require(scope.get('recursive_general_bedform_by_variant') == {
            GENERAL_VARIANT:RECURSIVE_GENERAL_BEDFORM_CONTRACT},
            'Packaged recursive general-bedform capability differs')
        general = root/GENERAL_RUNTIME_RELATIVE
        declared = {name:value for name,value in files.items()
                    if name.startswith(GENERAL_RUNTIME_RELATIVE+'/')}
        actual = {p.relative_to(root).as_posix() for p in general.rglob('*') if p.is_file()}
        require(general.is_dir() and not general.is_symlink() and bool(declared)
            and set(declared) == actual
            and not any(p.is_symlink() for p in general.rglob('*'))
            and all(sha(_member(root,name)) == value for name,value in declared.items()),
            'Packaged general AMR initializer inventory/bytes differ')
        general_build = verify_general(general)
        general_provenance = package.get('provenance', {}).get('on_demand_amr_general', {})
        general_dlls = general_provenance.get('dll_sha256', {})
        require(general_build.get('portable') is True and
            general_build['executable'] == GENERAL_EXECUTABLE and
            general_build['executable_sha256'] == GENERAL_EXPORTER_SHA256 and
            general_build['executable_sha256'] == general_provenance.get('executable_sha256') and
            files.get(GENERAL_RUNTIME_RELATIVE+'/'+GENERAL_EXECUTABLE)
                == GENERAL_EXPORTER_SHA256 and
            {'libgfortran-5.dll','libgomp-1.dll'} <= set(general_dlls) and
            {p.name for p in general.glob('*.dll')} == set(general_dlls) and
            all(files.get(GENERAL_RUNTIME_RELATIVE+'/'+name) == value
                for name,value in general_dlls.items()),
            'Packaged general AMR initializer provenance differs')
        general_kernels = root/GENERAL_KERNEL_RELATIVE
        general_declared = {name:value for name,value in files.items()
            if name.startswith(GENERAL_KERNEL_RELATIVE+'/')}
        general_actual = {p.relative_to(root).as_posix()
            for p in general_kernels.rglob('*') if p.is_file()}
        require(general_kernels.is_dir() and not general_kernels.is_symlink() and
            bool(general_declared) and set(general_declared) == general_actual and
            not any(p.is_symlink() for p in general_kernels.rglob('*')) and
            all(sha(_member(root,name)) == value
                for name,value in general_declared.items()),
            'Packaged general AMR CUDA source inventory/bytes differ')
        general_source = json.loads(
            (general_kernels/'runtime_manifest.json').read_text(encoding='utf-8'))
        source_files = general_source.get('source_files')
        python_pins = general_source.get('worker_python_sha256')
        require(general_source.get('schema') == 1 and
            general_source.get('variant') == GENERAL_VARIANT and
            general_source.get('archive_revision') ==
                GENERAL_SOURCE_ARCHIVE_REVISION and
            general_source.get('native_preparation_schema') == 3 and
            general_source.get('generated_live_transport_sources_bundled') is False and
            general_source.get('general_transport_source_generated_per_deck') is True and
            isinstance(source_files,dict) and len(source_files) >= 50 and
            isinstance(python_pins,dict) and len(python_pins) >= 100 and
            {p.name for p in general_kernels.iterdir()} ==
                set(source_files)|{'runtime_manifest.json'} and
            all(name.endswith('.cu') and Path(name).name == name and
                sha(general_kernels/name) == digest for name,digest in source_files.items()) and
            all(files.get(name) == digest and sha(_member(root,name)) == digest
                for name,digest in python_pins.items()),
            'Packaged general AMR source closure differs')
        general_pair = verify_interface_source_pair(root,
            {name:general_kernels/name for name in REVISED_KERNEL_ABIS},
            general_source.get('interface_source_pair'))
        result.update(general_runtime=general,general_build=general_build,
            general_kernels=general_kernels,
            general_interface_source_pair=general_pair)
    return result
