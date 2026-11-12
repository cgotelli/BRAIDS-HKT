"""Versioned mode-2/mode-3 selected AMR initialization; no native time stepping.

Old exporters and reference runtimes remain immutable.  Mode 3 exports only
its distributed-bank controls, material mask and histories, never a toe map.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from . import amr_native_geometry as legacy
from . import amr_live_native_geometry as mode2
from .amr_prepared import _read_coupled, _leaves, _tree_arrays, export_bank_amr_prepared_v2
from .atomic_output import write_json_atomic

sha = legacy.sha
require = legacy.require
KIND = 'live-selected-bank-mode23-amr'
BANK_CONTRACT = 'mode_dispatch_v1'
BANK_MODES = [2, 3]
DEFAULT_RUNTIME = Path(__file__).resolve().parents[1] / 'validation/g4_bank_amr/runs/on_demand_bank_mode23_geometry_runtime_20261008_v2'
GEOMETRY_INITIALIZATION_VERSION = 4
SEED_POLICY = 'native_status_1_or_2_geometry_only_evolving_gpu_authority_v1'
def sidecars(bank_mode):
    require(type(bank_mode) is int and bank_mode in BANK_MODES, 'Unsupported selected bank mode')
    return (('bank_toe_metadata.bin' if bank_mode == 2 else 'distributed_bank_metadata.bin'),
            'avalanche_metadata.bin', 'bank_history_metadata.bin', 'normal_metadata.bin')


_routine = mode2._routine
_replace_once = mode2._replace_once


def _distributed_export():
    from validation.gpu_fwave.coupled_reference import rec
    text = "\n subroutine gpu_selected_export_distributed_bank_metadata()\n  integer :: gpu_unit\n"
    text += "  if(bank_toe_erosion_mode/=3.or..not.bank_failure) &\n"
    text += "   error stop 'Selected distributed-bank export requires mode3'\n"
    text += "  call gpu_open(gpu_unit,'distributed_bank_metadata.bin')\n"
    for name in ('nc', 'nf', 'nfrac', 'nbed_layers', 'deck_version', 'bank_toe_erosion_mode'):
        text += rec(name, kind='i')
    text += rec('bank_failure', 'merge(1,0,bank_failure)', kind='i')
    text += rec('bank_distributed_transport_fraction')
    text += rec('bank_distributed_critical_slope')
    text += rec('bank_critical_undercut_m', 'bank_critical_undercut')
    text += rec('bank_material_mask', 'merge(1,0,bank_erodible)', ('nc',), 'i')
    return text + '  call gpu_finish(gpu_unit)\n end subroutine\n'


def _selected_source(base):
    selected = _routine(base, 'gpu_selected_amr_initialization')
    selected = _replace_once(selected,
        'if(bank_front_tracking.or.bank_toe_erosion_mode/=2.or..not.bank_failure)',
        'if(bank_front_tracking.or.(bank_toe_erosion_mode/=2.and.bank_toe_erosion_mode/=3).or..not.bank_failure)')
    selected = _replace_once(selected,
        "error stop 'Selected GPU initializer requires mode2 bank without persistent front'",
        "error stop 'Selected GPU initializer requires mode2 or3 bank without persistent front'")
    selected = _replace_once(selected, '  call gpu_selected_build_physical_toe_map(q)\n',
        '  if(bank_toe_erosion_mode==2) call gpu_selected_build_physical_toe_map(q)\n')
    selected = _replace_once(selected, '  call gpu_selected_export_physical_toe_metadata()\n',
        '  if(bank_toe_erosion_mode==2) then\n'
        '   call gpu_selected_export_physical_toe_metadata()\n'
        '  else\n'
        '   call gpu_selected_export_distributed_bank_metadata()\n'
        '  endif\n')
    return selected


def transform(source):
    base = mode2.transform(source)
    selected = _routine(base, 'gpu_selected_amr_initialization')
    changed = _replace_once(base, selected, _selected_source(base))
    return _replace_once(changed, 'end module\n\nprogram run_cutcell',
                         _distributed_export() + 'end module\n\nprogram run_cutcell')


def verify_transform(source, changed):
    require(changed == transform(source), 'Mode-dispatched selected initialization transform differs')
    base = mode2.transform(source)
    restored = _replace_once(changed, _distributed_export(), '')
    restored = _replace_once(restored, _selected_source(base),
                             _routine(base, 'gpu_selected_amr_initialization'))
    require(restored == base, 'Mode dispatcher does not restore exact mode2 exporter')
    mode2.verify_transform(source, restored)
    for name in ('load_case', 'amr_rebuild_mesh', 'amr_allocate_step_workspaces',
                 'process_distributed_bank_failure', 'build_physical_toe_map'):
        require(_routine(changed, name) == _routine(source, name),
                'Original native numerical/parser body changed: ' + name)
    return True


def _dependencies():
    # Verification is used inside a compiler/GDAL-free package. Do not import
    # build-only fixture modules merely to authenticate their immutable bytes.
    root = Path(__file__).resolve().parents[1]
    names = ('braids_gpu/amr_native_geometry.py',
        'braids_gpu/amr_live_native_geometry.py',
        'validation/gpu_fwave/normal_coupled_oracle.py',
        'validation/gpu_fwave/boundary_schedule_oracle.py',
        'validation/gpu_fwave/coupled_reference.py',
        'validation/gpu_fwave/g4_bank_active_front_export.py',
        'validation/gpu_fwave/cpu_filter_cold20h.py')
    return {name: sha(root / name) for name in names}


def build(runtime):
    """Build a new, write-once BRAIDS initializer; never rebuild a reference."""
    from validation.gpu_fwave import cpu_filter_cold20h as parent
    runtime = Path(runtime).resolve()
    require(not runtime.exists(), 'Live selected initialization runtime is write-once')
    identity, protected = parent.runtime_identity()
    dependencies = _dependencies()
    runtime.mkdir(parents=True)
    sources = []
    for name, digest in identity['source_sha256'].items():
        original, target = parent.TIGHT_RUNTIME / name, runtime / name
        require(sha(original) == digest, 'Frozen native source changed')
        if name == 'cutcell_solver.f90':
            source = original.read_text(encoding='utf-8')
            changed = transform(source)
            verify_transform(source, changed)
            # Generated compilation input; this is never a reference-source patch.
            target.write_text(changed, encoding='utf-8')
        else:
            shutil.copy2(original, target)
        sources.append(str(target))
    for original in parent.TIGHT_RUNTIME.glob('*.dll'):
        shutil.copy2(original, runtime / original.name)
    compiler = Path(identity['command'][0])
    flags = identity['command'][1:6]
    require(flags == ['-O3', '-flto', '-fopenmp', '-ffree-line-length-none', '-ffixed-line-length-none'],
            'Frozen compiler options differ')
    exe = runtime / 'xbraids_amr_live_geometry.exe'
    command = [str(compiler), *flags, *sources, '-o', str(exe)]
    result = subprocess.run(command, cwd=runtime, capture_output=True, text=True, timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (runtime / 'build.log').write_text(result.stdout + result.stderr)
    require(result.returncode == 0, 'Live selected initializer build failed: ' + result.stderr[-3000:])
    require(_dependencies() == dependencies, 'Export generator dependency changed during build')
    require(all(sha(path) == digest for path, digest in protected.items()), 'Frozen reference changed')
    record = dict(schema=1, kind=KIND, live_boundary_metadata=True, command=command,
        bank_metadata_contract=BANK_CONTRACT, bank_modes=BANK_MODES,
        geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION,
        toe_geometry_seed_only=True, selected_toe_initialization_policy=SEED_POLICY,
        executable=exe.name, executable_sha256=sha(exe),
        parent_executable_sha256=parent.TIGHT_SHA, compiler_sha256=sha(compiler),
        generator_sha256=sha(__file__), generator_dependencies=dependencies,
        source_sha256={Path(path).name: sha(path) for path in sources},
        parent_protected_files=protected, export_only=True, numerical_routines_unchanged=True,
        source_original_modified=False, Delft3D_source_modified=False)
    write_json_atomic(runtime / 'build.json', record)
    return record


def verify_runtime(runtime):
    runtime = Path(runtime).resolve()
    record = json.loads((runtime / 'build.json').read_text())
    require(record.get('schema') == 1 and record.get('kind') == KIND and
        record.get('live_boundary_metadata') is True and record.get('export_only') is True and
        record.get('bank_metadata_contract') == BANK_CONTRACT and record.get('bank_modes') == BANK_MODES and
        record.get('geometry_initialization_version') == GEOMETRY_INITIALIZATION_VERSION and
        record.get('toe_geometry_seed_only') is True and
        record.get('selected_toe_initialization_policy') == SEED_POLICY and
        record.get('numerical_routines_unchanged') is True and
        record.get('source_original_modified') is False and record.get('Delft3D_source_modified') is False and
        record.get('generator_sha256') == sha(__file__) and
        record.get('generator_dependencies') == _dependencies() and
        Path(record['executable']).name == record['executable'] and
        sha(runtime / record['executable']) == record['executable_sha256'] and
        all(Path(name).name == name and sha(runtime / name) == digest
            for name, digest in record['source_sha256'].items()),
        'Live selected initialization runtime identity differs')
    return record


def export_selected(deck, folder, ids, *, runtime, hierarchy_path=None, cancelled=lambda: False):
    """Return ``(PreparedAMR, toe_lineage)`` with the legacy export call shape."""
    from validation.gpu_fwave import coupled_reference as core
    deck, folder, runtime = Path(deck).resolve(), Path(folder).resolve(), Path(runtime).resolve()
    hierarchy_path = Path(hierarchy_path or deck.parent / 'amr_hierarchy.npz').resolve()
    hierarchy = legacy.hierarchy_from_sidecar(hierarchy_path)
    ids = _leaves(np.asarray(ids, dtype=np.int64), _tree_arrays(hierarchy))
    record = verify_runtime(runtime)
    require(not folder.exists(), 'Live selected operator destination is write-once')
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled')
    work = folder.with_name(folder.name + '_native')
    require(not work.exists(), 'Live selected native work directory is write-once')
    origin = core.copy_inputs(deck, work)
    if (work / 'geometry.npz').exists():
        (work / 'geometry.npz').rename(work / 'original_geometry.npz')
        origin['files']['original_geometry.npz'] = origin['files'].pop('geometry.npz')
    shutil.copy2(hierarchy_path, work / 'amr_hierarchy.npz')
    origin['files']['amr_hierarchy.npz'] = sha(work / 'amr_hierarchy.npz')
    np.savetxt(work / 'selected_nodes.txt', ids, fmt='%d', header=str(len(ids)), comments='')
    origin['files']['selected_nodes.txt'] = sha(work / 'selected_nodes.txt')
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    environment.update(BRAIDS_GPU_SELECTED_AMR_EXPORT='1', BRAIDS_TIME_INTEGRATOR='legacy',
        OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
    environment['PATH'] = str(runtime) + os.pathsep + environment.get('PATH', '')
    started = time.perf_counter()
    process = subprocess.Popen([str(runtime / record['executable']), deck.name], cwd=work, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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
        (work / 'initial.log').write_text(stdout + stderr)
        raise
    (work / 'initial.log').write_text(stdout + stderr)
    require(process.returncode == 0, 'Live selected initializer failed: ' + (stdout + stderr)[-2000:])
    require(not (work / 'frames.txt').exists() and not (work / 'checkpoint.bin').exists(),
            'Geometry preparation unexpectedly advanced a native simulation')
    data = _read_coupled(work / 'coupled_input.bin')
    bank_mode = data.get('bank_toe_erosion_mode')
    names = sidecars(bank_mode)
    require({path.name for path in work.glob('*metadata.bin')} == set(names),
            'Mode-dispatched selected metadata inventory differs')
    for filename in names:
        extra = _read_coupled(work / filename)
        require(all(key not in data or np.array_equal(data[key], value) for key, value in extra.items()),
                'Initializer sidecar conflicts with native core')
        data.update(extra)
    if bank_mode == 2:
        require(data.get('toe_geometry_seed_only') == 1,
                'Mode2 selected initializer did not declare geometry-only toe seeds')
    else:
        require('toe_geometry_seed_only' not in data and 'is_toe_face' not in data,
                'Mode3 selected initializer fabricated physical toe metadata')
    selected = _read_coupled(work / 'amr_selected_identity.bin')
    require(np.array_equal(selected['node_ids'], ids), 'Native initializer selected different leaves')
    legacy._selected_geometry(hierarchy_path, work / 'geometry.npz', ids)
    receipt = dict(schema=1, status='completed', live_boundary_metadata=True, origin=origin,
        bank_mode=bank_mode, bank_metadata_contract=BANK_CONTRACT,
        geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION,
        toe_geometry_seed_only=bank_mode == 2, selected_toe_initialization_policy=SEED_POLICY if bank_mode == 2 else None,
        runtime_build_sha256=sha(runtime / 'build.json'), runtime_executable_sha256=record['executable_sha256'],
        native_cpu_parent_sha256=record['parent_executable_sha256'], exporter_source_sha256=sha(__file__),
        initialization_CPU_only=True, solver_steps_performed=0, numerical_routines_unchanged=True,
        GPU_solver_executed=False, Delft3D_source_modified=False, cells=int(data['nc']), faces=int(data['nf']),
        controls={key: int(data[key]) for key in ('scheme', 'bedlaw', 'exchange_mode',
            'bank_toe_erosion_mode', 'general_avalanche', 'secondary_flow_model', 'cpu_time_integrator')},
        bank_cells=int(np.count_nonzero(data['bank_material_mask'])),
        artifacts={name: sha(work / name) for name in ('coupled_input.bin', *names)})
    if bank_mode == 2:
        receipt['toe_faces'] = int(np.count_nonzero(data['is_toe_face']))
    write_json_atomic(work / 'preparation_receipt.json', receipt)
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled before publication')
    package = export_bank_amr_prepared_v2(folder, hierarchy,
        [('leaves_' + hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(), ids, work)],
        runtime_build=runtime / 'build.json', source=dict(
            hierarchy_sha256=sha(hierarchy_path), base_input_sha256=sha(deck),
            runtime_build_sha256=sha(runtime / 'build.json'), on_demand_initialization=True,
            live_boundary_metadata=True, toe_geometry_seed_only=bank_mode == 2,
            bank_mode=bank_mode, bank_metadata_contract=BANK_CONTRACT,
            geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION))
    write_json_atomic(work / 'selection_receipt.json', dict(
        node_ids=ids.tolist(), tree_input_sha256=sha(work / 'amr_tree_input.bin'),
        selection_identity_sha256=sha(work / 'amr_selected_identity.bin'),
        elapsed_s=time.perf_counter() - started, original_input_sha256=sha(deck),
        native_solver_steps=0, full_state_transfer_on_GPU=True))
    lineage = []
    for face in (np.flatnonzero(data['is_toe_face']) if bank_mode == 2 else []):
        lineage.append(dict(segment_id=int(selected['face_source_ids'][face]), local_face=int(face),
            bank_node=int(ids[int(data['candidate_bank'][face])]),
            toe_node=int(ids[int(data['candidate_toe'][face])]),
            midpoint_xy_m=data['mid'][:, face].tolist(), length_m=float(data['length'][face])))
    return package, lineage

