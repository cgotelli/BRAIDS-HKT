"""Fast host packaging for the unchanged general-AMR zero-step exporter.

The native executable, selected faces, controls, and package loader are the
same as :mod:`amr_general_native_geometry`.  Only viewer polygon packaging is
changed: authenticated packed hierarchy arrays are selected directly instead
of expanding every cell into Python dictionaries and packing them again.
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

from . import amr_general_native_geometry as base
from .amr_prepared import _leaves, _read_coupled, _tree_arrays
from .atomic_output import write_json_atomic
from .packed_geometry_selection import write_selected_packed_geometry


def export_selected(deck, folder, ids, *, runtime=base.DEFAULT_RUNTIME,
                    hierarchy_path=None, cancelled=lambda: False):
    """Export one schema-3 operator with byte-identical packed geometry."""
    from validation.gpu_fwave import coupled_reference as core
    from .amr_general_prepared import export_general_amr_prepared_v3

    deck, folder, runtime = (Path(path).resolve() for path in
                             (deck, folder, runtime))
    hierarchy_path = Path(hierarchy_path or
                          deck.parent / 'amr_hierarchy.npz').resolve()
    hierarchy = base.legacy.hierarchy_from_sidecar(hierarchy_path)
    ids = _leaves(np.asarray(ids, dtype=np.int64), _tree_arrays(hierarchy))
    record = base.verify_runtime(runtime)
    base.require(not folder.exists(),
                 'Selected general operator destination is write-once')
    if cancelled():
        raise InterruptedError('General AMR initialization cancelled')
    work = folder.with_name(folder.name + '_native')
    base.require(not work.exists(),
                 'Selected general native work directory is write-once')
    origin = core.copy_inputs(deck, work)
    if (work / 'geometry.npz').exists():
        (work / 'geometry.npz').rename(work / 'original_geometry.npz')
        origin['files']['original_geometry.npz'] = \
            origin['files'].pop('geometry.npz')
    shutil.copy2(hierarchy_path, work / 'amr_hierarchy.npz')
    origin['files']['amr_hierarchy.npz'] = base.sha(
        work / 'amr_hierarchy.npz')
    np.savetxt(work / 'selected_nodes.txt', ids, fmt='%d',
               header=str(len(ids)), comments='')
    origin['files']['selected_nodes.txt'] = base.sha(
        work / 'selected_nodes.txt')
    environment = {key: value for key, value in os.environ.items()
                   if not key.upper().startswith(('BRAIDS_', 'OMP_'))}
    environment.update(BRAIDS_GPU_SELECTED_AMR_EXPORT='1',
        BRAIDS_TIME_INTEGRATOR='legacy', OMP_NUM_THREADS='1',
        OMP_DYNAMIC='FALSE', OMP_WAIT_POLICY='PASSIVE')
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
    base.require(process.returncode == 0,
                 'General AMR initializer failed: ' +
                 (stdout + stderr)[-2000:])
    base.require(not (work / 'frames.txt').exists() and
                 not (work / 'checkpoint.bin').exists(),
                 'General geometry preparation advanced a native simulation')
    data = _read_coupled(work / 'coupled_input.bin')
    base.require(int(data['bank_toe_erosion_mode']) == 0 and
                 int(data['bank_failure']) == 0,
                 'General selected initializer exported bank physics')
    names = {'normal_metadata.bin'}
    if int(data['general_avalanche']):
        names.add('avalanche_metadata.bin')
    if float(data['horizontal_viscosity']) > 0.:
        names.add('coupled_diffusion.bin')
    base.require({p.name for p in work.glob('*metadata.bin')} |
                 {p.name for p in work.glob('coupled_diffusion.bin')} == names,
                 'General selected sidecar inventory differs')
    selected = _read_coupled(work / 'amr_selected_identity.bin')
    base.require(set(selected) == {'node_ids', 'face_source_ids'} and
                 np.array_equal(selected['node_ids'], ids) and
                 np.shape(selected['face_source_ids']) == (int(data['nf']),),
                 'General initializer selected different nodes/faces')
    write_selected_packed_geometry(hierarchy_path, work / 'geometry.npz', ids)
    receipt = dict(schema=1, status='completed', kind=base.KIND, origin=origin,
        runtime_build_sha256=base.sha(runtime / 'build.json'),
        runtime_executable_sha256=record['executable_sha256'],
        native_cpu_parent_sha256=record['parent_executable_sha256'],
        exporter_source_sha256=base.sha(__file__),
        geometry_writer_source_sha256=base.sha(
            Path(write_selected_packed_geometry.__code__.co_filename)),
        geometry_packing='direct_authenticated_packed_selection_v1',
        initialization_CPU_only=True, solver_steps_performed=0,
        numerical_routines_unchanged=True, GPU_solver_executed=False,
        Delft3D_source_modified=False, cells=int(data['nc']),
        faces=int(data['nf']), controls={key: int(data[key]) for key in
            ('scheme', 'bedlaw', 'exchange_mode', 'bank_toe_erosion_mode',
             'general_avalanche', 'secondary_flow_model',
             'cpu_time_integrator')}, artifacts={name: base.sha(work / name)
             for name in ('coupled_input.bin', *sorted(names))})
    write_json_atomic(work / 'preparation_receipt.json', receipt)
    write_json_atomic(work / 'selection_receipt.json', dict(
        node_ids=ids.tolist(),
        tree_input_sha256=base.sha(work / 'amr_tree_input.bin'),
        selection_identity_sha256=base.sha(work / 'amr_selected_identity.bin'),
        elapsed_s=time.perf_counter() - started,
        original_input_sha256=base.sha(deck), native_solver_steps=0,
        full_state_transfer_on_GPU=True,
        geometry_packing='direct_authenticated_packed_selection_v1'))
    if cancelled():
        raise InterruptedError(
            'General AMR initialization cancelled before publication')
    source = dict(
        hierarchy_sha256=base.sha(hierarchy_path),
        base_input_sha256=base.sha(deck),
        runtime_build_sha256=base.sha(runtime / 'build.json'),
        on_demand_initialization=True,
        live_boundary_metadata=True,
        geometry_packing='direct_authenticated_packed_selection_v1')
    # ``GeneralGeometrySession`` deliberately uses this fast exporter.  Keep
    # the authenticated v49 suspension/AMR binding that the base exporter
    # publishes; otherwise the same ordinary deck is silently relabelled as
    # the historical SSPRK33 route when it is packed here.
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
                metadata, np.ascontiguousarray(
                    tree_record['roughness_parameter'][finest],
                    dtype=np.float64), native_skin_controls=skin)
            typed_tree = dict(
                parent=np.ascontiguousarray(
                    tree_record['parent'], dtype=np.int64),
                node_level=np.ascontiguousarray(
                    tree_record['node_level'], dtype=np.int64),
                area=np.ascontiguousarray(
                    tree_record['area'], dtype=np.float64),
                roughness_parameter=np.ascontiguousarray(
                    tree_record['roughness_parameter'], dtype=np.float64),
                raw_roughness_mode=np.asarray(
                    tree_record['raw_roughness_mode']))
            source[AMR_ROUGHNESS_BINDING_FIELD] = \
                build_amr_roughness_binding(local_binding, typed_tree)
    package = export_general_amr_prepared_v3(folder, hierarchy,
        [('leaves_' + hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()
                                     ).hexdigest(), ids, work)],
        runtime_build=runtime / 'build.json', source=source)
    return package, work
