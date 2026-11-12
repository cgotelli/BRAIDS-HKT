"""Versioned zero-step selected AMR exporter with authenticated live boundaries.

The legacy selected initializer and all native numerical bodies remain unchanged.
The selected path exports a versioned, geometry-only toe seed.  Equal-distance
native seed choices are permitted there, but not in the unchanged native solver.
The evolving GPU mapper must construct the physical map from accepted state.
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
from .amr_prepared import _read_coupled, _leaves, _tree_arrays, export_bank_amr_prepared_v2
from .atomic_output import write_json_atomic

sha = legacy.sha
require = legacy.require
KIND = 'live-selected-bank-amr'
DEFAULT_RUNTIME = Path(__file__).resolve().parents[1] / 'validation/g4_bank_amr/runs/on_demand_live_geometry_runtime_20261008_v4'
GEOMETRY_INITIALIZATION_VERSION = 3
SEED_POLICY = 'native_status_1_or_2_geometry_only_evolving_gpu_authority_v1'
SIDECARS = ('bank_toe_metadata.bin', 'avalanche_metadata.bin',
            'bank_history_metadata.bin', 'normal_metadata.bin')


def _export_source():
    from validation.gpu_fwave.normal_coupled_oracle import extra_export
    return extra_export()


def _routine(source, name):
    """Extract one complete native routine without changing its source bytes."""
    match = re.search(r'^ subroutine ' + re.escape(name) + r'\([^\n]*\n.*?^ end subroutine\n',
                      source, flags=re.MULTILINE | re.DOTALL)
    require(match is not None and source.count('subroutine ' + name + '(') == 1,
            'Native routine anchor differs: ' + name)
    return match.group(0)


def _replace_once(source, old, new):
    require(source.count(old) == 1, 'Live geometry copy anchor differs: ' + old[:100])
    return source.replace(old, new, 1)


def _seed_sources(base):
    """Only these two private initialization copies differ from legacy export."""
    mapper = _routine(base, 'build_physical_toe_map')
    mapper = _replace_once(mapper, 'subroutine build_physical_toe_map(state)',
                           'subroutine gpu_selected_build_physical_toe_map(state)')
    mapper = _replace_once(mapper, '  integer :: mapping_error\n',
                           '  integer :: mapping_error\n  integer,parameter :: gpu_seed_ambiguous=2\n')
    mapper = _replace_once(mapper,
        '     any(bank_erodible.and.physical_toe_status/=BANK_TOE_MAPPED)) &',
        '     any(bank_erodible.and.physical_toe_status/=BANK_TOE_MAPPED.and. &\n'
        '         physical_toe_status/=gpu_seed_ambiguous)) &')
    mapper = _replace_once(mapper,
        "   error stop 'noncohesive bank material does not map uniquely to a physical toe'",
        "   error stop 'invalid or unreachable geometry-only physical toe seed'")
    exporter = _routine(base, 'i4_export_physical_toe_metadata')
    exporter = _replace_once(exporter, 'subroutine i4_export_physical_toe_metadata()',
                             'subroutine gpu_selected_export_physical_toe_metadata()')
    exporter = _replace_once(exporter, 'call build_physical_toe_map(q)',
                             'call gpu_selected_build_physical_toe_map(q)')
    exporter = _replace_once(exporter,
        "  call gpu_put_i(gpu_unit,'toe_mapping_status',[0],[integer::])\n",
        "  call gpu_put_i(gpu_unit,'toe_mapping_status',[0],[integer::])\n"
        "  call gpu_put_i(gpu_unit,'toe_geometry_seed_only',[1],[integer::])\n")
    return mapper + '\n' + exporter + '\n'


def _selected_source(base, *, private_seed):
    selected = _routine(base, 'gpu_selected_amr_initialization')
    if private_seed:
        selected = _replace_once(selected, 'call build_physical_toe_map(q)',
                                  'call gpu_selected_build_physical_toe_map(q)')
        selected = _replace_once(selected, 'call i4_export_physical_toe_metadata()',
                                  'call gpu_selected_export_physical_toe_metadata()')
        selected = _replace_once(selected, '  call g4_export_bank_history_metadata()\n',
                                  '  call g4_export_bank_history_metadata()\n  call normal_export_metadata()\n')
    return selected


def transform(source):
    base = legacy.transform(source)
    marker = 'end module\n\nprogram run_cutcell'
    # The native selected rebuild does not construct the static physical map;
    # its explicit subsequent call is the only mapping authority changed here.
    for name in ('amr_rebuild_mesh', 'amr_allocate_step_workspaces'):
        require('call build_physical_toe_map(' not in _routine(base, name),
                'Selected rebuild now invokes a strict map internally: ' + name)
    changed = _replace_once(base, _selected_source(base, private_seed=False),
                            _selected_source(base, private_seed=True))
    return _replace_once(changed, marker, _seed_sources(base) + _export_source() + marker)


def verify_transform(source, changed):
    require(changed == transform(source), 'Live selected initialization transform differs')
    base = legacy.transform(source)
    restored = _replace_once(changed, _seed_sources(base) + _export_source(), '')
    restored = _replace_once(restored, _selected_source(base, private_seed=True),
                              _selected_source(base, private_seed=False))
    require(restored == base, 'Live geometry-only copies do not restore the exact legacy exporter')
    legacy.verify_transform(source, restored)
    for name in ('load_case', 'amr_sort_boundary_sections', 'section_nodes',
                 'live_section_stage', 'boundary_hydraulic_at', 'boundary_sediment_at',
                 'build_physical_toe_map', 'amr_rebuild_mesh', 'amr_allocate_step_workspaces'):
        require(_routine(changed, name) == _routine(source, name),
                'Original native numerical/parser body changed: ' + name)
    require(_routine(changed, 'i4_export_physical_toe_metadata') ==
            _routine(base, 'i4_export_physical_toe_metadata'), 'Legacy toe exporter changed')
    selected = _routine(changed, 'gpu_selected_amr_initialization')
    require(selected.index('call amr_rebuild_mesh()') < selected.index('call normal_export_metadata()'),
            'Live section metadata must follow the selected geometry rebuild')
    return True


def _dependencies():
    # Verification is used inside a compiler/GDAL-free package. Do not import
    # build-only fixture modules merely to authenticate their immutable bytes.
    root = Path(__file__).resolve().parents[1]
    names = ('braids_gpu/amr_native_geometry.py',
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
    for filename in SIDECARS:
        extra = _read_coupled(work / filename)
        require(all(key not in data or np.array_equal(data[key], value) for key, value in extra.items()),
                'Initializer sidecar conflicts with native core')
        data.update(extra)
    require(np.asarray(data.get('toe_geometry_seed_only')).shape == () and
            int(data['toe_geometry_seed_only']) == 1,
            'Live selected initializer did not declare geometry-only toe seeds')
    selected = _read_coupled(work / 'amr_selected_identity.bin')
    require(np.array_equal(selected['node_ids'], ids), 'Native initializer selected different leaves')
    legacy._selected_geometry(hierarchy_path, work / 'geometry.npz', ids)
    receipt = dict(schema=1, status='completed', live_boundary_metadata=True, origin=origin,
        geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION,
        toe_geometry_seed_only=True, selected_toe_initialization_policy=SEED_POLICY,
        runtime_build_sha256=sha(runtime / 'build.json'), runtime_executable_sha256=record['executable_sha256'],
        native_cpu_parent_sha256=record['parent_executable_sha256'], exporter_source_sha256=sha(__file__),
        initialization_CPU_only=True, solver_steps_performed=0, numerical_routines_unchanged=True,
        GPU_solver_executed=False, Delft3D_source_modified=False, cells=int(data['nc']), faces=int(data['nf']),
        controls={key: int(data[key]) for key in ('scheme', 'bedlaw', 'exchange_mode',
            'bank_toe_erosion_mode', 'general_avalanche', 'secondary_flow_model', 'cpu_time_integrator')},
        bank_cells=int(np.count_nonzero(data['bank_material_mask'])),
        toe_faces=int(np.count_nonzero(data['is_toe_face'])),
        artifacts={name: sha(work / name) for name in ('coupled_input.bin', *SIDECARS)})
    write_json_atomic(work / 'preparation_receipt.json', receipt)
    if cancelled():
        raise InterruptedError('AMR geometry preparation cancelled before publication')
    package = export_bank_amr_prepared_v2(folder, hierarchy,
        [('leaves_' + hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest(), ids, work)],
        runtime_build=runtime / 'build.json', source=dict(
            hierarchy_sha256=sha(hierarchy_path), base_input_sha256=sha(deck),
            runtime_build_sha256=sha(runtime / 'build.json'), on_demand_initialization=True,
            live_boundary_metadata=True, toe_geometry_seed_only=True,
            geometry_initialization_version=GEOMETRY_INITIALIZATION_VERSION))
    write_json_atomic(work / 'selection_receipt.json', dict(
        node_ids=ids.tolist(), tree_input_sha256=sha(work / 'amr_tree_input.bin'),
        selection_identity_sha256=sha(work / 'amr_selected_identity.bin'),
        elapsed_s=time.perf_counter() - started, original_input_sha256=sha(deck),
        native_solver_steps=0, full_state_transfer_on_GPU=True))
    lineage = []
    for face in np.flatnonzero(data['is_toe_face']):
        lineage.append(dict(segment_id=int(selected['face_source_ids'][face]), local_face=int(face),
            bank_node=int(ids[int(data['candidate_bank'][face])]),
            toe_node=int(ids[int(data['candidate_toe'][face])]),
            midpoint_xy_m=data['mid'][:, face].tolist(), length_m=float(data['length'][face])))
    return package, lineage
