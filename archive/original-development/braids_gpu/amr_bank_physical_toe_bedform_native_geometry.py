"""Exact zero-step initializer for recursive bedform/physical-toe AMR.

This additive BRAIDS runtime admits only the constant-stage, noncohesive
mode-2 physical-toe composition with equilibrium bedform roughness and one
parent-owned general avalanche.  It derives geometry-only toe seeds from the
selected graph and never advances the native solver.  Historical mode-2/3 and
mode-3 bedform initializers and their authenticated runtimes remain untouched.
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

from . import amr_bank_native_geometry as base
from . import amr_native_geometry as legacy
from .amr_prepared import (_read_coupled, _leaves, _tree_arrays,
                           export_bank_amr_prepared_v2)
from .atomic_output import write_json_atomic


sha = legacy.sha
require = legacy.require
KIND = 'live-selected-bank-mode2-bedform-avalanche-amr-v1'
CONTRACT = 'recursive_bedform_mode2_physical_toe_parent_avalanche_v1'
DEFAULT_RUNTIME = (Path(__file__).resolve().parents[1] / 'validation/g4_bank_amr/runs'
                   / 'on_demand_bank_mode2_physical_toe_bedform_runtime_20261013_v6')
GEOMETRY_INITIALIZATION_VERSION = 7
SEED_POLICY = base.SEED_POLICY
SIDECARS = ('bank_toe_metadata.bin', 'avalanche_metadata.bin',
            'bank_history_metadata.bin', 'normal_metadata.bin')
EXECUTABLE = 'xbraids_amr_bedform_physical_toe_geometry.exe'
PARSER_SCOPE_REVISION = 'exact_recursive_mode2_physical_toe_bedform_parent_avalanche_v1'


_INHERITED_EXPORT_GATE = "  if(trim(request)/='1') return\n"
_FAIL_CLOSED_EXPORT_GATE = (
    "  if(trim(request)/='1') &\n"
    "   error stop 'Selected mode2 bedform physical-toe initializer is export-only'\n")


_OLD_BEDFORM_GUARD = (
    "  if(bedform_roughness.and.(secondary_flow.or.bedload_adaptation.or.morphological_acceleration.or. &\n"
    "     bank_erosion_fraction>0.or.bank_avalanche.or.bank_failure.or.general_avalanche)) &\n"
    "   error stop 'bedform roughness is not qualified with another experimental morphology process'\n")


_NEW_BEDFORM_GUARD = (
    "  if(bedform_roughness.and.(secondary_flow.or.bedload_adaptation.or.morphological_acceleration.or. &\n"
    "     bank_erosion_fraction>0.or.bank_avalanche.or. &\n"
    "     ((bank_failure.or.general_avalanche).and..not.(bank_failure.and.general_avalanche.and. &\n"
    "       bank_toe_erosion_mode==2.and..not.bank_front_tracking.and.bedlaw==2.and. &\n"
    "       mpm_friction_mode==1.and.bed_slope==0.and.hiding_exposure==0.and.exchange_mode==0.and. &\n"
    "       secondary_flow_model==0.and.nsubstrate_layers==1.and..not.morphological_acceleration.and. &\n"
    "       nmorfac==0.and.morphological_factor==1.d0.and.horizontal_viscosity==0.d0.and. &\n"
    "       horizontal_viscosity_mode==0.and.horizontal_viscosity_geometry==0)))) &\n"
    "   error stop 'bedform roughness composition is outside exact recursive mode2 physical-toe scope'\n")


def _scope_guard_source():
    return r"""
 subroutine gpu_require_bedform_physical_toe_parent_avalanche_scope()
  integer :: b
  if(.not.bedform_roughness.or..not.amr_enabled.or..not.amr_subcycling.or.bank_toe_erosion_mode/=2.or. &
     .not.bank_failure.or..not.general_avalanche.or.bank_front_tracking.or.bank_avalanche.or. &
     bank_erosion_fraction/=0.d0.or.bank_toe_capacity_fraction<0.d0.or. &
     bank_toe_capacity_fraction>1.d0.or.bank_critical_undercut<=0.d0.or. &
     bedlaw/=2.or.mpm_friction_mode/=1.or.bed_slope/=0.or.hiding_exposure/=0.or.exchange_mode/=0.or. &
     secondary_flow.or.secondary_flow_model/=0.or.bedload_adaptation.or.morphological_acceleration.or. &
     nmorfac/=0.or.morphological_factor/=1.d0.or.horizontal_viscosity/=0.d0.or. &
     horizontal_viscosity_mode/=0.or.horizontal_viscosity_geometry/=0.or. &
     nsubstrate_layers/=1.or.nbed_layers/=2.or.cpu_time_integrator/=33.or. &
     any(sediment_type_fraction/=0).or..not.any(bank_erodible)) &
   error stop 'Selected initializer requires exact recursive mode2 physical-toe bedform parent-avalanche physics'
  if(nb/=1.or.any(mode/=2).or.any(section_control/=0).or.any(nseries/=0).or. &
     any(nhydro_series/=0).or.any(nfraction_series/=0).or.any(sampled_boundary_bed/=0).or. &
     any(boundary_bed_offset/=0.d0).or.any(fraction_concentration/=0.d0).or. &
     any(fraction_bed_input/=0.d0).or.any(series_concentration/=0.d0).or. &
     any(fraction_series_concentration/=0.d0)) &
   error stop 'Selected initializer rejects live sections and requires one unscheduled unsampled constant-stage boundary'
  do b=1,nfrac
   if(any(q(suspended_index(b),:)/=0.d0)) &
    error stop 'Selected initializer requires zero suspended state'
  enddo
 end subroutine
"""


def transform(source):
    """Create the exact export-only source without changing a frozen runtime."""
    previous = base.transform(source)
    changed = base._replace_once(previous, _OLD_BEDFORM_GUARD,
                                 _NEW_BEDFORM_GUARD)
    selected = base._routine(changed, 'gpu_selected_amr_initialization')
    checked = base._replace_once(selected, _INHERITED_EXPORT_GATE,
                                 _FAIL_CLOSED_EXPORT_GATE)
    checked = base._replace_once(
        checked,
        "   error stop 'Selected GPU initializer requires mode2 or3 bank without persistent front'\n"
        "  ! This record is the original complete initial tree, not a CPU trajectory.\n",
        "   error stop 'Selected GPU initializer requires mode2 or3 bank without persistent front'\n"
        "  call gpu_require_bedform_physical_toe_parent_avalanche_scope()\n"
        "  ! This record is the original complete initial tree, not a CPU trajectory.\n")
    changed = base._replace_once(changed, selected, checked)
    marker = 'end module\n\nprogram run_cutcell'
    return base._replace_once(changed, marker, _scope_guard_source() + marker)


def verify_transform(source, changed):
    require(changed == transform(source),
            'Mode2 bedform physical-toe initialization transform differs')
    previous = base.transform(source)
    restored = base._replace_once(changed, _scope_guard_source(), '')
    checked = base._routine(restored, 'gpu_selected_amr_initialization')
    original = base._routine(previous, 'gpu_selected_amr_initialization')
    restored = base._replace_once(restored, checked, original)
    restored = base._replace_once(restored, _NEW_BEDFORM_GUARD,
                                  _OLD_BEDFORM_GUARD)
    require(restored == previous,
            'Mode2 bedform scope changes do not restore the mode23 exporter')
    base.verify_transform(source, restored)
    for name in ('amr_rebuild_mesh', 'amr_allocate_step_workspaces',
                 'process_distributed_bank_failure', 'build_physical_toe_map'):
        require(base._routine(changed, name) == base._routine(source, name),
                'Original native numerical body changed: ' + name)
    load = base._routine(changed, 'load_case')
    require(load.count(_NEW_BEDFORM_GUARD) == 1 and
            _OLD_BEDFORM_GUARD not in load,
            'Exact mode2 bedform parser guard differs')
    selected = base._routine(changed, 'gpu_selected_amr_initialization')
    require(selected.count(
        'call gpu_require_bedform_physical_toe_parent_avalanche_scope()') == 1,
        'Exact mode2 bedform post-load guard is absent')
    require(selected.count(_FAIL_CLOSED_EXPORT_GATE) == 1 and
            _INHERITED_EXPORT_GATE not in selected,
            'Exact mode2 bedform initializer is not fail-closed export-only')
    require('if(bank_toe_erosion_mode==2) call gpu_selected_build_physical_toe_map(q)' in selected and
            'call gpu_selected_export_physical_toe_metadata()' in selected,
            'Exact mode2 initializer lost geometry-only physical-toe export')
    return True


def _dependencies():
    root = Path(__file__).resolve().parents[1]
    names = ('braids_gpu/amr_native_geometry.py',
        'braids_gpu/amr_live_native_geometry.py',
        'braids_gpu/amr_bank_native_geometry.py',
        'validation/gpu_fwave/normal_coupled_oracle.py',
        'validation/gpu_fwave/boundary_schedule_oracle.py',
        'validation/gpu_fwave/coupled_reference.py',
        'validation/gpu_fwave/g4_bank_active_front_export.py',
        'validation/gpu_fwave/cpu_filter_cold20h.py')
    return {name: sha(root / name) for name in names}


def build(runtime=DEFAULT_RUNTIME):
    """Build a new write-once BRAIDS initializer; never rebuild a reference."""
    from validation.gpu_fwave import cpu_filter_cold20h as parent
    runtime = Path(runtime).resolve()
    require(not runtime.exists(),
            'Mode2 bedform physical-toe initialization runtime is write-once')
    identity, protected = parent.runtime_identity()
    dependencies = _dependencies()
    runtime.mkdir(parents=True)
    sources = []
    for name, digest in identity['source_sha256'].items():
        original, target = parent.TIGHT_RUNTIME / name, runtime / name
        require(sha(original) == digest, 'Frozen native source changed')
        if name == 'cutcell_solver.f90':
            source = original.read_text(encoding='utf-8')
            generated = transform(source)
            verify_transform(source, generated)
            target.write_text(generated, encoding='utf-8')
        else:
            shutil.copy2(original, target)
        sources.append(str(target))
    for original in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(original, runtime / original.name)
    compiler = Path(identity['command'][0])
    flags = identity['command'][1:6]
    require(flags == ['-O3', '-flto', '-fopenmp', '-ffree-line-length-none',
                      '-ffixed-line-length-none'], 'Frozen compiler options differ')
    executable = runtime / EXECUTABLE
    command = [str(compiler), *flags, *sources, '-o', str(executable)]
    result = subprocess.run(command, cwd=runtime, capture_output=True, text=True,
        timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (runtime / 'build.log').write_text(result.stdout + result.stderr,
                                       encoding='utf-8')
    require(result.returncode == 0,
            'Mode2 bedform physical-toe initializer build failed: ' +
            result.stderr[-3000:])
    require(_dependencies() == dependencies,
            'Mode2 bedform physical-toe exporter dependency changed during build')
    require(all(sha(path) == digest for path, digest in protected.items()),
            'Frozen reference changed')
    record = dict(schema=1, kind=KIND, live_boundary_metadata=True,
        command=command, bank_metadata_contract=CONTRACT, bank_modes=[2],
        geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION,
        toe_geometry_seed_only=True,
        selected_toe_initialization_policy=SEED_POLICY,
        evolving_toe_geometry=True,
        recursive_bank_bedform_avalanche=True,
        executable=executable.name, executable_sha256=sha(executable),
        parent_executable_sha256=parent.TIGHT_SHA,
        compiler_sha256=sha(compiler), generator_sha256=sha(__file__),
        generator_dependencies=dependencies,
        source_sha256={Path(path).name: sha(path) for path in sources},
        parent_protected_files=protected, export_only=True,
        numerical_routines_unchanged=True,
        parser_scope_revision=PARSER_SCOPE_REVISION,
        source_original_modified=False, Delft3D_source_modified=False)
    write_json_atomic(runtime / 'build.json', record)
    return record


def verify_runtime(runtime=DEFAULT_RUNTIME):
    runtime = Path(runtime).resolve()
    record = json.loads((runtime / 'build.json').read_text(encoding='utf-8'))
    require(record.get('schema') == 1 and record.get('kind') == KIND and
        record.get('live_boundary_metadata') is True and
        record.get('export_only') is True and
        record.get('bank_metadata_contract') == CONTRACT and
        record.get('bank_modes') == [2] and
        record.get('geometry_initialization_version') == GEOMETRY_INITIALIZATION_VERSION and
        record.get('toe_geometry_seed_only') is True and
        record.get('selected_toe_initialization_policy') == SEED_POLICY and
        record.get('evolving_toe_geometry') is True and
        record.get('recursive_bank_bedform_avalanche') is True and
        record.get('parser_scope_revision') == PARSER_SCOPE_REVISION and
        record.get('numerical_routines_unchanged') is True and
        record.get('source_original_modified') is False and
        record.get('Delft3D_source_modified') is False and
        record.get('generator_sha256') == sha(__file__) and
        record.get('generator_dependencies') == _dependencies() and
        record.get('executable') == EXECUTABLE and
        sha(runtime / EXECUTABLE) == record['executable_sha256'] and
        all(Path(name).name == name and sha(runtime / name) == digest
            for name, digest in record['source_sha256'].items()),
        'Mode2 bedform physical-toe initialization runtime identity differs')
    return record


def export_selected(deck, folder, ids, *, runtime=DEFAULT_RUNTIME,
                    hierarchy_path=None, cancelled=lambda: False):
    """Export one exact mode-2 selected operator without native timesteps."""
    from validation.gpu_fwave import coupled_reference as core
    deck, folder, runtime = (Path(value).resolve()
                             for value in (deck, folder, runtime))
    hierarchy_path = Path(hierarchy_path or
                          deck.parent / 'amr_hierarchy.npz').resolve()
    hierarchy = legacy.hierarchy_from_sidecar(hierarchy_path)
    ids = _leaves(np.asarray(ids, dtype=np.int64), _tree_arrays(hierarchy))
    record = verify_runtime(runtime)
    require(not folder.exists(),
            'Mode2 bedform physical-toe operator destination is write-once')
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled')
    work = folder.with_name(folder.name + '_native')
    require(not work.exists(),
            'Mode2 bedform physical-toe native work directory is write-once')
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
    environment.update(BRAIDS_GPU_SELECTED_AMR_EXPORT='1',
        BRAIDS_TIME_INTEGRATOR='legacy', OMP_NUM_THREADS='1',
        OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
    environment['PATH'] = str(runtime) + os.pathsep + environment.get('PATH', '')
    started = time.perf_counter()
    process = subprocess.Popen([str(runtime / EXECUTABLE), deck.name], cwd=work,
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        while True:
            if cancelled():
                raise InterruptedError('AMR geometry preparation cancelled')
            if time.perf_counter() - started > 120:
                raise TimeoutError('AMR geometry initialization exceeded 120 seconds')
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
            'Mode2 bedform physical-toe initializer failed: ' +
            (stdout + stderr)[-2000:])
    require(not (work / 'frames.txt').exists() and
            not (work / 'checkpoint.bin').exists(),
            'Geometry preparation unexpectedly advanced a native simulation')
    data = _read_coupled(work / 'coupled_input.bin')
    require({path.name for path in work.glob('*metadata.bin')} == set(SIDECARS),
            'Mode2 bedform physical-toe metadata inventory differs')
    for filename in SIDECARS:
        extra = _read_coupled(work / filename)
        require(all(key not in data or np.array_equal(data[key], value)
                    for key, value in extra.items()),
                'Initializer sidecar conflicts with native core')
        data.update(extra)
    require(int(data.get('bank_toe_erosion_mode', -1)) == 2 and
            int(data.get('bedform_roughness', 0)) == 1 and
            int(data.get('bank_failure', 0)) == 1 and
            int(data.get('general_avalanche', 0)) == 1 and
            int(data.get('toe_geometry_seed_only', 0)) == 1 and
            np.array_equal(np.asarray(data.get('mode')), [2]) and
            np.count_nonzero(data.get('is_toe_face', 0)) > 0,
            'Initializer did not export exact constant-stage mode2 bedform/toe composition')
    selected = _read_coupled(work / 'amr_selected_identity.bin')
    require(np.array_equal(selected['node_ids'], ids),
            'Native initializer selected different leaves')
    legacy._selected_geometry(hierarchy_path, work / 'geometry.npz', ids)
    receipt = dict(schema=1, status='completed', live_boundary_metadata=True,
        origin=origin, bank_mode=2, bank_metadata_contract=CONTRACT,
        geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION,
        toe_geometry_seed_only=True,
        selected_toe_initialization_policy=SEED_POLICY,
        evolving_toe_geometry=True,
        recursive_bank_bedform_avalanche=True,
        runtime_build_sha256=sha(runtime / 'build.json'),
        runtime_executable_sha256=record['executable_sha256'],
        native_cpu_parent_sha256=record['parent_executable_sha256'],
        exporter_source_sha256=sha(__file__), initialization_CPU_only=True,
        solver_steps_performed=0, numerical_routines_unchanged=True,
        GPU_solver_executed=False, Delft3D_source_modified=False,
        cells=int(data['nc']), faces=int(data['nf']),
        controls={key: int(data[key]) for key in ('scheme', 'bedlaw',
            'exchange_mode', 'bank_toe_erosion_mode', 'general_avalanche',
            'secondary_flow_model', 'cpu_time_integrator', 'bedform_roughness')},
        bank_cells=int(np.count_nonzero(data['bank_material_mask'])),
        toe_faces=int(np.count_nonzero(data['is_toe_face'])),
        artifacts={name: sha(work / name)
                   for name in ('coupled_input.bin', *SIDECARS)})
    write_json_atomic(work / 'preparation_receipt.json', receipt)
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled before publication')
    package = export_bank_amr_prepared_v2(folder, hierarchy,
        [('leaves_' + hashlib.sha256(
            np.asarray(ids, dtype='<i8').tobytes()).hexdigest(), ids, work)],
        runtime_build=runtime / 'build.json', source=dict(
            hierarchy_sha256=sha(hierarchy_path), base_input_sha256=sha(deck),
            runtime_build_sha256=sha(runtime / 'build.json'),
            on_demand_initialization=True, live_boundary_metadata=True,
            toe_geometry_seed_only=True, bank_mode=2,
            bank_metadata_contract=CONTRACT,
            recursive_bank_bedform_avalanche=True,
            geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION))
    write_json_atomic(work / 'selection_receipt.json', dict(
        node_ids=ids.tolist(),
        tree_input_sha256=sha(work / 'amr_tree_input.bin'),
        selection_identity_sha256=sha(work / 'amr_selected_identity.bin'),
        elapsed_s=time.perf_counter() - started,
        original_input_sha256=sha(deck), native_solver_steps=0,
        full_state_transfer_on_GPU=True))
    lineage = []
    for face in np.flatnonzero(data['is_toe_face']):
        lineage.append(dict(segment_id=int(selected['face_source_ids'][face]),
            local_face=int(face), bank_node=int(ids[int(data['candidate_bank'][face])]),
            toe_node=int(ids[int(data['candidate_toe'][face])]),
            midpoint_xy_m=data['mid'][:, face].tolist(),
            length_m=float(data['length'][face])))
    return package, lineage
