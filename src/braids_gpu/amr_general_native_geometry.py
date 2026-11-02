"""Versioned zero-step selected AMR initializer for nonbank GPU transport.

Only a private copy of the BRAIDS native source is transformed. The selected
initialization branch exports geometry and controls; it performs no timestep.
Existing bank initializers and every Delft3D source remain untouched.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from . import amr_native_geometry as legacy
from . import amr_live_native_geometry as live
from .amr_prepared import _leaves, _read_coupled, _tree_arrays
from .atomic_output import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
KIND = 'selected-general-transport-amr-v1'
DEFAULT_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v10'
HISTORICAL_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v1'
HISTORICAL_BUILD_SHA256 = \
    '524ca4719daeadaf9764f48f218dd67f852250715b488a6c09929c55658c433f'
HISTORICAL_EXECUTABLE_SHA256 = \
    '3387d2a56c37215985d0086feb9e151a0cc78a913a6923770fd4c746d5215592'
PREVIOUS_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v5'
PREVIOUS_BUILD_SHA256 = \
    '2d562e5cabfadf95edcf37aa4ad1be573275224dfb9d1930b57caff0ef33d08e'
PREVIOUS_EXECUTABLE_SHA256 = \
    '8073e9154d6451ef40769233980c25151707016b0fb19e66715bdd6ef450ee5d'
FROZEN_V6_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v6'
FROZEN_V6_BUILD_SHA256 = \
    'c0a303e17c6d85b707514a21265ff7d99441ddacb7388ed010290599d90b9faf'
FROZEN_V6_EXECUTABLE_SHA256 = \
    '750512dbdd7646dcff35e64143ec2ba62df855cf5d1aadf5df67584a99f99f6d'
FROZEN_V7_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v7'
FROZEN_V7_BUILD_SHA256 = \
    '2e7a634eb355fb4ac2c28edfc109b2b7308c4c105d88a78748f321f32442df08'
FROZEN_V7_EXECUTABLE_SHA256 = \
    '08dc10483bafb1dac12824fb049fe75955572b7124b4232c52ba115b0746fa5d'
FROZEN_V8_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v8'
FROZEN_V8_BUILD_SHA256 = \
    'a6ae8067b443bf28d575820ca3d11fd980426c60bb1b606a371ee753310fe8c4'
FROZEN_V8_EXECUTABLE_SHA256 = \
    '7f02dbf33a2f8af38f1530e1e7b854e151b96664061bec9fbed9fda59892b326'
FROZEN_V9_RUNTIME = ROOT / 'braids_gpu/native_amr_general_v9'
FROZEN_V9_BUILD_SHA256 = \
    '2328159e98699c35d84cf6fbdae23efcecc9476616bc72ea1433a2210f166efa'
FROZEN_V9_EXECUTABLE_SHA256 = \
    'ddac217050fe4c2545dff763b3bb8481edd27fbc0139597e884c96d38ae97d89'
CURRENT_CPU_BUILD = ROOT / 'solver/cut_cells/build/build.json'
CURRENT_CPU_EXECUTABLE = ROOT / 'solver/cut_cells/build/xbraids_cutcell.exe'
CURRENT_CPU_BUILD_SHA256 = \
    '4e7440f5c09b2aab1c655c319451c46f3d35a14bb39b30e7b2c9bff29ea5ba09'
CURRENT_CPU_EXECUTABLE_SHA256 = \
    'd5036a060d7d9f4cf3456531197137f08a4de3b341ca65eecf63f8570ac2a055'
CURRENT_CPU_CUTCELL_SOURCE_SHA256 = \
    '4a10d7a5891974030e6335e11e9ac3f6f0fc1cf2bdb5ab4d18276615c9133411'
sha = legacy.sha
require = legacy.require


def _selected_source(base):
    selected = live._routine(base, 'gpu_selected_amr_initialization')
    replacements = (
        ('if(bank_front_tracking.or.bank_toe_erosion_mode/=2.or..not.bank_failure)',
         'if(bank_front_tracking.or.bank_failure.or.bank_toe_erosion_mode/=0)'),
        ("error stop 'Selected GPU initializer requires mode2 bank without persistent front'",
         "error stop 'Selected general GPU initializer requires bank physics OFF'"),
        ('  call build_physical_toe_map(q)\n', ''),
        ('  call i4_export_physical_toe_metadata()\n', ''),
        ('  call g4_export_bank_history_metadata()\n', ''),
        ('  call i4_export_avalanche_metadata()\n',
         '  if(general_avalanche) call i4_export_avalanche_metadata()\n'
         '  call normal_export_metadata()\n'),
    )
    for before, after in replacements:
        selected = live._replace_once(selected, before, after)
    tree_anchor = ("  call gpu_put_i(gpu_unit,'subcycling',"
                   "[merge(1,0,amr_subcycling)],[integer::])\n")
    typed_records = (
        "  call gpu_put_r(gpu_unit,'roughness_parameter',"
        "reshape(at_manning,[size(at_manning)]),[amr_nn])\n"
        "  call gpu_put_i(gpu_unit,'raw_roughness_mode',"
        "[at_roughness_mode],[integer::])\n")
    selected = live._replace_once(selected, tree_anchor,
                                  tree_anchor + typed_records)
    return selected


def _normal_export_v9():
    """Add native values required to authenticate the v49 JSON binding.

    The records are export-only observations of values already parsed
    by the unchanged native initializer.  They close a stale-sidecar gap
    without adding those values to a second hand-maintained deck parser.
    """
    from braids_support.gpu_fwave import coupled_reference as core
    from braids_support.gpu_fwave.normal_coupled_oracle import extra_export

    base = extra_export()
    anchor = '  call gpu_finish(gpu_unit)\n end subroutine\n'
    additions = core.rec(
        'native_suspended_boundary_return_time_s',
        'suspended_boundary_return_time', ('nb',))
    additions += core.rec(
        'native_constitutive_kinematic_viscosity_m2_s',
        'constitutive_kinematic_viscosity')
    additions += core.rec(
        'native_roughness_minimum_work_depth_m',
        'roughness_minimum_work_depth')
    additions += core.rec(
        'native_hydraulic_roughness_mode',
        'hydraulic_roughness_mode', kind='i')
    additions += core.rec(
        'native_white_colebrook_chzk2d',
        'merge(1,0,white_colebrook_chzk2d)', kind='i')
    additions += core.rec('native_sediment_shear_mode',
                          'sediment_shear_mode', kind='i')
    additions += core.rec('native_skin_kinematic_viscosity_m2_s',
                          'skin_kinematic_viscosity')
    additions += core.rec('native_skin_sand_roughness_m',
                          'skin_sand_roughness')
    additions += core.rec('native_skin_silt_roughness_m',
                          'skin_silt_roughness')
    additions += core.rec('native_skin_critical_mud_fraction_low',
                          'skin_critical_mud_fraction_low')
    additions += core.rec('native_skin_critical_mud_fraction_high',
                          'skin_critical_mud_fraction_high')
    additions += core.rec('native_skin_fluff_fraction',
                          'skin_fluff_fraction')
    require(base.count(anchor) == 1,
            'General native metadata export anchor differs')
    return base.replace(anchor, additions + anchor, 1)


def transform(source):
    """Transform only the selected export branch and add a metadata writer."""
    base = legacy.transform(source)
    selected = live._routine(base, 'gpu_selected_amr_initialization')
    changed = live._replace_once(base, selected, _selected_source(base))
    return live._replace_once(changed, 'end module\n\nprogram run_cutcell',
                             _normal_export_v9() +
                             'end module\n\nprogram run_cutcell')


def verify_transform(source, changed):
    require(changed == transform(source), 'General selected initializer transform differs')
    base = legacy.transform(source)
    restored = live._replace_once(changed, _normal_export_v9(), '')
    restored = live._replace_once(restored, _selected_source(base),
                                  live._routine(base, 'gpu_selected_amr_initialization'))
    require(restored == base, 'General initializer does not restore its exact parent')
    legacy.verify_transform(source, restored)
    for name in ('load_case', 'amr_rebuild_mesh', 'amr_allocate_step_workspaces',
                 'gpu_export_input'):
        require(live._routine(changed, name) == live._routine(base, name),
                'General initializer altered native numerical/parser routine: ' + name)
    return True


def _dependencies():
    names = ('braids_gpu/amr_native_geometry.py',
             'braids_gpu/amr_live_native_geometry.py',
             'braids_support/gpu_fwave/normal_coupled_oracle.py',
             'braids_support/gpu_fwave/coupled_reference.py',
             'braids_support/gpu_fwave/g4_bank_active_front_export.py',
             'braids_support/gpu_fwave/cpu_filter_cold20h.py')
    return {name: sha(ROOT / name) for name in names}


def build(runtime=DEFAULT_RUNTIME):
    """Build a write-once initializer from the stable current v49 CPU build."""
    runtime = Path(runtime).resolve()
    require(not runtime.exists(), 'General selected initializer is write-once')
    require(sha(CURRENT_CPU_BUILD) == CURRENT_CPU_BUILD_SHA256 and
            sha(CURRENT_CPU_EXECUTABLE) == CURRENT_CPU_EXECUTABLE_SHA256,
            'Stable current BRAIDS v49 CPU parent changed')
    identity = json.loads(CURRENT_CPU_BUILD.read_text(encoding='utf-8'))
    require(identity.get('executable_sha256') ==
            CURRENT_CPU_EXECUTABLE_SHA256 and
            identity.get('source_sha256', {}).get(
                'solver/cut_cells/cutcell_solver.f90') ==
            CURRENT_CPU_CUTCELL_SOURCE_SHA256,
            'Stable current BRAIDS v49 parent manifest differs')
    source_paths = [Path(value).resolve() for value in identity['command'][6:-2]]
    source_identity = {path.relative_to(ROOT).as_posix(): sha(path)
                       for path in source_paths}
    require(source_identity == identity['source_sha256'] and
            sha(ROOT/'solver/cut_cells/cutcell_solver.f90') ==
            CURRENT_CPU_CUTCELL_SOURCE_SHA256,
            'Stable current BRAIDS v49 parent source inventory changed')
    dependencies = _dependencies()
    runtime.mkdir(parents=True)
    sources = []
    for original in source_paths:
        target = runtime / original.name
        if original.name == 'cutcell_solver.f90':
            source = original.read_text(encoding='utf-8')
            changed = transform(source)
            verify_transform(source, changed)
            target.write_text(changed, encoding='utf-8')
        else:
            shutil.copy2(original, target)
        sources.append(str(target))
    for original in CURRENT_CPU_EXECUTABLE.parent.glob('*.dll'):
        shutil.copy2(original, runtime / original.name)
    compiler = Path(shutil.which(identity['command'][0]) or '')
    require(compiler.is_file(), 'Stable current BRAIDS v49 compiler missing')
    flags = identity['command'][1:6]
    require(flags == ['-O3', '-flto', '-fopenmp', '-ffree-line-length-none',
                      '-ffixed-line-length-none'], 'Frozen BRAIDS compiler flags differ')
    executable = runtime / 'xbraids_amr_general_geometry.exe'
    command = [str(compiler), *flags, *sources, '-o', str(executable)]
    result = subprocess.run(command, cwd=runtime, capture_output=True, text=True,
                            timeout=300,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (runtime / 'build.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    require(result.returncode == 0,
            'General selected initializer build failed: ' + result.stderr[-3000:])
    require(_dependencies() == dependencies and
            sha(CURRENT_CPU_BUILD) == CURRENT_CPU_BUILD_SHA256 and
            sha(CURRENT_CPU_EXECUTABLE) == CURRENT_CPU_EXECUTABLE_SHA256 and
            all(sha(ROOT/name) == digest
                for name, digest in source_identity.items()),
            'Stable v49 BRAIDS parent or generator dependency changed during build')
    record = dict(schema=1, kind=KIND, command=command, executable=executable.name,
        executable_sha256=sha(executable),
        parent_executable_sha256=CURRENT_CPU_EXECUTABLE_SHA256,
        parent_build_manifest_sha256=CURRENT_CPU_BUILD_SHA256,
        parent_cutcell_source_sha256=CURRENT_CPU_CUTCELL_SOURCE_SHA256,
        parent_source_sha256=source_identity,
        parent_deck_versions_including_v49=True,
        compiler_sha256=sha(compiler), generator_sha256=sha(__file__),
        generator_dependencies=dependencies,
        source_sha256={Path(path).name: sha(path) for path in sources},
        parent_protected_files={str(CURRENT_CPU_BUILD):CURRENT_CPU_BUILD_SHA256,
            str(CURRENT_CPU_EXECUTABLE):CURRENT_CPU_EXECUTABLE_SHA256},
        export_only=True,
        numerical_routines_unchanged=True, source_original_modified=False,
        Delft3D_source_modified=False)
    write_json_atomic(runtime / 'build.json', record)
    return record


def verify_runtime(runtime=DEFAULT_RUNTIME):
    runtime = Path(runtime).resolve()
    record = json.loads((runtime / 'build.json').read_text(encoding='utf-8'))
    current = (record.get('generator_sha256') == sha(__file__) and
               record.get('generator_dependencies') == _dependencies())
    historical = (runtime == HISTORICAL_RUNTIME.resolve() and
        sha(runtime / 'build.json') == HISTORICAL_BUILD_SHA256 and
        record.get('executable_sha256') == HISTORICAL_EXECUTABLE_SHA256)
    previous = (runtime == PREVIOUS_RUNTIME.resolve() and
        sha(runtime / 'build.json') == PREVIOUS_BUILD_SHA256 and
        record.get('executable_sha256') == PREVIOUS_EXECUTABLE_SHA256)
    frozen_v6 = (runtime == FROZEN_V6_RUNTIME.resolve() and
        sha(runtime / 'build.json') == FROZEN_V6_BUILD_SHA256 and
        record.get('executable_sha256') == FROZEN_V6_EXECUTABLE_SHA256)
    frozen_v7 = (runtime == FROZEN_V7_RUNTIME.resolve() and
        sha(runtime / 'build.json') == FROZEN_V7_BUILD_SHA256 and
        record.get('executable_sha256') == FROZEN_V7_EXECUTABLE_SHA256)
    frozen_v8 = (runtime == FROZEN_V8_RUNTIME.resolve() and
        sha(runtime / 'build.json') == FROZEN_V8_BUILD_SHA256 and
        record.get('executable_sha256') == FROZEN_V8_EXECUTABLE_SHA256)
    frozen_v9 = (runtime == FROZEN_V9_RUNTIME.resolve() and
        sha(runtime / 'build.json') == FROZEN_V9_BUILD_SHA256 and
        record.get('executable_sha256') == FROZEN_V9_EXECUTABLE_SHA256)
    require(record.get('schema') == 1 and record.get('kind') == KIND and
            record.get('export_only') is True and
            record.get('numerical_routines_unchanged') is True and
            record.get('source_original_modified') is False and
            record.get('Delft3D_source_modified') is False and
            (current or historical or previous or frozen_v6 or frozen_v7 or
             frozen_v8 or frozen_v9) and
            Path(record['executable']).name == record['executable'] and
            sha(runtime / record['executable']) == record['executable_sha256'] and
            all(Path(name).name == name and sha(runtime / name) == digest
                for name, digest in record['source_sha256'].items()),
            'General selected initializer runtime identity differs')
    return record


def export_selected(deck, folder, ids, *, runtime=DEFAULT_RUNTIME,
                    hierarchy_path=None, cancelled=lambda: False):
    """Export one selected operator and return (schema-3 package, native folder)."""
    from braids_support.gpu_fwave import coupled_reference as core
    from .amr_general_prepared import export_general_amr_prepared_v3

    deck, folder, runtime = (Path(path).resolve() for path in (deck, folder, runtime))
    hierarchy_path = Path(hierarchy_path or deck.parent / 'amr_hierarchy.npz').resolve()
    hierarchy = legacy.hierarchy_from_sidecar(hierarchy_path)
    ids = _leaves(np.asarray(ids, dtype=np.int64), _tree_arrays(hierarchy))
    record = verify_runtime(runtime)
    require(not folder.exists(), 'Selected general operator destination is write-once')
    if cancelled():
        raise InterruptedError('General AMR initialization cancelled')
    work = folder.with_name(folder.name + '_native')
    require(not work.exists(), 'Selected general native work directory is write-once')
    origin = core.copy_inputs(deck, work)
    if (work / 'geometry.npz').exists():
        (work / 'geometry.npz').rename(work / 'original_geometry.npz')
        origin['files']['original_geometry.npz'] = origin['files'].pop('geometry.npz')
    shutil.copy2(hierarchy_path, work / 'amr_hierarchy.npz')
    origin['files']['amr_hierarchy.npz'] = sha(work / 'amr_hierarchy.npz')
    np.savetxt(work / 'selected_nodes.txt', ids, fmt='%d',
               header=str(len(ids)), comments='')
    origin['files']['selected_nodes.txt'] = sha(work / 'selected_nodes.txt')
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    environment.update(BRAIDS_GPU_SELECTED_AMR_EXPORT='1', BRAIDS_TIME_INTEGRATOR='legacy',
        OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
    environment['PATH'] = str(runtime) + os.pathsep + environment.get('PATH', '')
    started = time.perf_counter()
    process = subprocess.Popen([str(runtime / record['executable']), deck.name],
        cwd=work, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        while True:
            if cancelled():
                raise InterruptedError('General AMR initialization cancelled')
            if time.perf_counter() - started > 120:
                raise TimeoutError('General AMR initialization exceeded 120 seconds')
            try:
                stdout, stderr = process.communicate(timeout=.05)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        (work / 'initial.log').write_text(stdout + stderr, encoding='utf-8')
        raise
    (work / 'initial.log').write_text(stdout + stderr, encoding='utf-8')
    require(process.returncode == 0,
            'General AMR initializer failed: ' + (stdout + stderr)[-2000:])
    require(not (work / 'frames.txt').exists() and not (work / 'checkpoint.bin').exists(),
            'General geometry preparation advanced a native simulation')
    data = _read_coupled(work / 'coupled_input.bin')
    require(int(data['bank_toe_erosion_mode']) == 0 and int(data['bank_failure']) == 0,
            'General selected initializer exported bank physics')
    names = {'normal_metadata.bin'}
    if int(data['general_avalanche']):
        names.add('avalanche_metadata.bin')
    if float(data['horizontal_viscosity']) > 0.:
        names.add('coupled_diffusion.bin')
    require({p.name for p in work.glob('*metadata.bin')} |
            {p.name for p in work.glob('coupled_diffusion.bin')} == names,
            'General selected sidecar inventory differs')
    selected = _read_coupled(work / 'amr_selected_identity.bin')
    require(set(selected) == {'node_ids', 'face_source_ids'} and
            np.array_equal(selected['node_ids'], ids) and
            np.shape(selected['face_source_ids']) == (int(data['nf']),),
            'General initializer selected different nodes/faces')
    legacy._selected_geometry(hierarchy_path, work / 'geometry.npz', ids)
    receipt = dict(schema=1, status='completed', kind=KIND, origin=origin,
        runtime_build_sha256=sha(runtime / 'build.json'),
        runtime_executable_sha256=record['executable_sha256'],
        native_cpu_parent_sha256=record['parent_executable_sha256'],
        exporter_source_sha256=sha(__file__), initialization_CPU_only=True,
        solver_steps_performed=0, numerical_routines_unchanged=True,
        GPU_solver_executed=False, Delft3D_source_modified=False,
        cells=int(data['nc']), faces=int(data['nf']),
        controls={key: int(data[key]) for key in
                  ('scheme', 'bedlaw', 'exchange_mode', 'bank_toe_erosion_mode',
                   'general_avalanche', 'secondary_flow_model', 'cpu_time_integrator')},
        artifacts={name: sha(work / name) for name in
                   ('coupled_input.bin', *sorted(names))})
    write_json_atomic(work / 'preparation_receipt.json', receipt)
    write_json_atomic(work / 'selection_receipt.json', dict(
        node_ids=ids.tolist(), tree_input_sha256=sha(work / 'amr_tree_input.bin'),
        selection_identity_sha256=sha(work / 'amr_selected_identity.bin'),
        elapsed_s=time.perf_counter() - started, original_input_sha256=sha(deck),
        native_solver_steps=0, full_state_transfer_on_GPU=True))
    if cancelled():
        raise InterruptedError('General AMR initialization cancelled before publication')
    source = dict(
        hierarchy_sha256=sha(hierarchy_path), base_input_sha256=sha(deck),
        runtime_build_sha256=sha(runtime / 'build.json'),
        on_demand_initialization=True, live_boundary_metadata=True)
    metadata_path = work / (deck.name + '.json')
    if metadata_path.is_file() and not metadata_path.is_symlink():
        from .implicit_suspension_amr import (
            PREPARED_BINDING_FIELD, prepared_binding_from_case_metadata)
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        binding = prepared_binding_from_case_metadata(
            metadata, deck_path=work / deck.name)
        if binding is not None:
            source[PREPARED_BINDING_FIELD] = binding
        if ('hydraulic_roughness' in metadata and
                'sediment_shear' in metadata):
            from .forcing_profile import forcing_binding_from_metadata
            from .amr_typed_roughness import (
                BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD,
                build_binding as build_amr_roughness_binding)
            tree_record = _read_coupled(work / 'amr_tree_input.bin')
            normal = _read_coupled(work / 'normal_metadata.bin')
            level = np.asarray(tree_record['node_level'])
            finest = np.flatnonzero(level == np.max(level))
            skin = dict(
                kinematic_viscosity_m2_s=float(normal[
                    'native_skin_kinematic_viscosity_m2_s']),
                sand_roughness_m=float(normal[
                    'native_skin_sand_roughness_m']),
                silt_roughness_m=float(normal[
                    'native_skin_silt_roughness_m']),
                critical_mud_fraction_low=float(normal[
                    'native_skin_critical_mud_fraction_low']),
                critical_mud_fraction_high=float(normal[
                    'native_skin_critical_mud_fraction_high']),
                fixed_fluff_fraction=float(normal[
                    'native_skin_fluff_fraction']))
            local_binding = forcing_binding_from_metadata(
                metadata,
                np.ascontiguousarray(
                    tree_record['roughness_parameter'][finest],
                    dtype=np.float64),
                native_skin_controls=skin)
            typed_tree = dict(parent=np.ascontiguousarray(
                                  tree_record['parent'], dtype=np.int64),
                              node_level=np.ascontiguousarray(
                                  tree_record['node_level'], dtype=np.int64),
                              area=np.ascontiguousarray(
                                  tree_record['area'], dtype=np.float64),
                              roughness_parameter=np.ascontiguousarray(
                                  tree_record['roughness_parameter'],
                                  dtype=np.float64),
                              raw_roughness_mode=np.asarray(
                                  tree_record['raw_roughness_mode']))
            source[AMR_ROUGHNESS_BINDING_FIELD] = \
                build_amr_roughness_binding(local_binding, typed_tree)
    package = export_general_amr_prepared_v3(folder, hierarchy,
        [('leaves_' + hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(),
          ids, work)], runtime_build=runtime / 'build.json', source=source)
    return package, work


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'verify'))
    parser.add_argument('--runtime', type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    result = build(args.runtime) if args.action == 'build' else verify_runtime(args.runtime)
    print(json.dumps(result, indent=2))
