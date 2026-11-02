"""Authenticated schema-3 selected AMR operators for nonbank GPU physics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np

from braids_core.windows_paths import filesystem_path
from .amr_prepared import (AMRTopology, PreparedAMR, _leaves, _tree_arrays,
    _validate_tree, _validate_operator, _solver_aliases, _read_coupled)


KIND = 'braids-gpu-general-amr-prepared'
SCHEMA = 3
TREE = 'tree.npz'
MANIFEST = 'amr_prepared.json'
IDENTITY_FILES = ('amr_tree_input.bin', 'amr_selected_identity.bin',
                  'selection_receipt.json')
CORE_FILES = ('coupled_input.bin', 'geometry.npz', 'preparation_receipt.json')
NORMAL = 'normal_metadata.bin'
AVALANCHE = 'avalanche_metadata.bin'
DIFFUSION = 'coupled_diffusion.bin'
AVALANCHE_CONTROLS = ('general_avalanche_wet_slope',
    'general_avalanche_dry_slope', 'general_avalanche_time',
    'general_avalanche_wet_depth_switch',
    'general_avalanche_relaxation_length')
GLOBAL_CONTROLS = ('nfrac', 'nstate', 'nbed_layers', 'bedlaw', 'exchange_mode',
    'secondary_flow_model', 'morphological_factor', 'general_avalanche',
    'horizontal_viscosity', 'horizontal_viscosity_mode',
    'horizontal_viscosity_geometry', 'mpm_friction_mode', 'mpm_cf',
    'hiding_exposure', 'bed_slope', 'bedload_adaptation',
    'bedload_adaptation_length_diameters', 'bulk_fraction',
    'mode', 'section_control', 'section_energy_slope', 'forcing',
    'fraction_concentration', 'fraction_bed_input', 'nseries',
    'nhydro_series', 'nfraction_series')


def sha(path):
    return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _direct(folder, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative and
            all(piece not in ('', '.', '..') for piece in relative.split('/')) and
            len(relative.split('/')) <= 2, 'Invalid general AMR artifact path')
    path = folder.joinpath(*relative.split('/'))
    require(path.resolve().is_relative_to(folder.resolve()) and path.is_file() and
            not path.is_symlink(), 'Missing/linked general AMR artifact: ' + relative)
    return path


def _sidecars(core):
    expected = [NORMAL]
    if int(core['general_avalanche']):
        expected.append(AVALANCHE)
    if float(core['horizontal_viscosity']) > 0.:
        expected.append(DIFFUSION)
    return tuple(expected)


def _merged(topology_folder, ids, tree, sidecars, source=None):
    core = _read_coupled(topology_folder / 'coupled_input.bin')
    _validate_operator(core, ids, tree)
    require(int(core['bank_failure']) == 0 and
            int(core['bank_toe_erosion_mode']) == 0 and
            int(core['bank_front_tracking']) == 0,
            'General AMR operator contains bank physics')
    require(tuple(sidecars) == _sidecars(core),
            'General AMR optional physics sidecars differ from native controls')
    data = dict(core)
    for filename in sidecars:
        extra = _read_coupled(topology_folder / filename)
        if filename == DIFFUSION:
            for key, value in extra.items():
                require('diffusion_' + key not in data,
                        'Duplicate diffusion control: ' + key)
                data['diffusion_' + key] = value
        else:
            for key, value in extra.items():
                require(key not in data or np.array_equal(data[key], value),
                        'General AMR sidecar/core conflict: ' + key)
                data[key] = value
    if NORMAL in sidecars:
        for name in ('section_seed_node', 'boundary_seed_position'):
            value = np.asarray(data['native_' + name])
            require(value.dtype == np.int32 and value.shape == (int(core['nb']),) and
                    np.all(value >= 0), 'Invalid native general AMR section seed: ' + name)
            data[name] = np.ascontiguousarray(value - 1)
    data = _solver_aliases(data)
    implicit_binding = None
    if isinstance(source, dict):
        from .amr_typed_roughness import (
            BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD,
            apply_amr_forcing_binding)
        roughness_binding = source.get(AMR_ROUGHNESS_BINDING_FIELD)
        if roughness_binding is not None:
            native_tree = _read_coupled(
                topology_folder / 'amr_tree_input.bin')
            typed_tree = dict(
                parent=np.ascontiguousarray(tree['parent'], dtype=np.int64),
                node_level=np.ascontiguousarray(
                    tree['node_level'], dtype=np.int64),
                area=np.ascontiguousarray(tree['area'], dtype=np.float64),
                roughness_parameter=np.ascontiguousarray(
                    native_tree['roughness_parameter'], dtype=np.float64),
                raw_roughness_mode=np.asarray(
                    native_tree['raw_roughness_mode']))
            apply_amr_forcing_binding(data, ids, typed_tree,
                                      roughness_binding)
        from .implicit_suspension_amr import (
            PREPARED_BINDING_FIELD, apply_prepared_binding)
        implicit_binding = source.get(PREPARED_BINDING_FIELD)
        apply_prepared_binding(data, implicit_binding)
    from .general_live_transport import _validate_prepared
    _validate_prepared(data, integrator=(
        'implicit_fwave_midpoint' if implicit_binding is not None else 'ssp33'))
    return data


def _runtime_build(build_path):
    record = json.loads(build_path.read_text(encoding='utf-8'))
    from .amr_general_native_geometry import KIND as RUNTIME_KIND
    require(record.get('schema') == 1 and record.get('kind') == RUNTIME_KIND and
            record.get('export_only') is True and
            record.get('numerical_routines_unchanged') is True and
            record.get('source_original_modified') is False and
            record.get('Delft3D_source_modified') is False and
            isinstance(record.get('executable_sha256'), str) and
            len(record['executable_sha256']) == 64,
            'General AMR runtime build contract differs')
    return record


def _native_receipt(path, core_path, sidecars, runtime_hash, executable_hash,
                    *, check_origin):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    from .amr_general_native_geometry import KIND as RUNTIME_KIND
    require(receipt.get('schema') == 1 and receipt.get('kind') == RUNTIME_KIND and
            receipt.get('status') == 'completed' and
            receipt.get('initialization_CPU_only') is True and
            receipt.get('solver_steps_performed') == 0 and
            receipt.get('GPU_solver_executed') is False and
            receipt.get('runtime_build_sha256') == runtime_hash and
            receipt.get('runtime_executable_sha256') == executable_hash and
            receipt.get('artifacts') == {name: sha(path.parent / name)
                for name in ('coupled_input.bin', *sidecars)} and
            receipt['artifacts']['coupled_input.bin'] == sha(core_path),
            'General AMR native initialization receipt differs')
    if check_origin:
        origin = receipt.get('origin', {}).get('files', {})
        require(isinstance(origin, dict) and bool(origin) and
                all(Path(name).name == name and
                    (path.parent / name).is_file() and
                    sha(path.parent / name) == digest
                    for name, digest in origin.items()),
                'General AMR native original input changed')
    return receipt


def _selection(topology_folder, ids, tree, source):
    selection = _read_coupled(topology_folder / 'amr_selected_identity.bin')
    tree_record = _read_coupled(topology_folder / 'amr_tree_input.bin')
    receipt = json.loads((topology_folder / 'selection_receipt.json').read_text('utf-8'))
    require(set(selection) == {'node_ids', 'face_source_ids'} and
            np.array_equal(selection['node_ids'], ids) and
            np.asarray(selection['face_source_ids']).ndim == 1 and
            np.all(np.asarray(selection['face_source_ids']) > 0) and
            receipt.get('node_ids') == ids.tolist() and
            receipt.get('native_solver_steps') == 0 and
            receipt.get('full_state_transfer_on_GPU') is True and
            receipt.get('tree_input_sha256') == sha(topology_folder / 'amr_tree_input.bin') and
            receipt.get('selection_identity_sha256') == sha(topology_folder / 'amr_selected_identity.bin') and
            receipt.get('original_input_sha256') == source.get('base_input_sha256'),
            'General AMR native selected node/face identity differs')
    for key in ('parent', 'node_level', 'maximum_level', 'area'):
        require(key in tree_record and np.array_equal(tree_record[key], tree[key]),
                'General AMR selected tree differs: ' + key)
    require(np.array_equal(tree_record['xy'].T, tree['xy']),
            'General AMR selected coordinates differ')
    from .amr_typed_roughness import BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD
    if AMR_ROUGHNESS_BINDING_FIELD in source:
        require('roughness_parameter' in tree_record and
                'raw_roughness_mode' in tree_record and
                np.asarray(tree_record['roughness_parameter']).shape ==
                np.asarray(tree_record['area']).shape,
                'General AMR typed roughness tree record differs')
    return selection


def load_general_amr_prepared(folder):
    """Verify all package bytes and return native-layout selected operators."""
    folder = filesystem_path(folder).resolve()
    manifest_path = _direct(folder, MANIFEST)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(set(manifest) == {'kind', 'schema', 'ratio', 'source', 'artifacts',
                              'topologies', 'runtime_build', 'native_solver_steps'} and
            manifest['kind'] == KIND and manifest['schema'] == SCHEMA and
            manifest['ratio'] == 2 and manifest['runtime_build'] == 'runtime_build.json' and
            manifest['native_solver_steps'] == 0 and
            isinstance(manifest['topologies'], list) and manifest['topologies'] and
            isinstance(manifest['source'], dict) and isinstance(manifest['artifacts'], dict),
            'General AMR schema-3 package manifest differs')
    artifacts = manifest['artifacts']
    require(all(isinstance(name, str) and isinstance(digest, str) and
                len(digest) == 64 and sha(_direct(folder, name)) == digest
                for name, digest in artifacts.items()),
            'General AMR package artifact changed')
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} ==
            set(artifacts) | {MANIFEST},
            'General AMR package contains unlisted or missing artifact')
    with np.load(_direct(folder, TREE), allow_pickle=False) as archive:
        require(set(archive.files) == {'parent', 'node_level', 'maximum_level',
                'offsets', 'area', 'xy'}, 'General AMR hierarchy record differs')
        tree = {name: archive[name].copy() for name in archive.files}
    _validate_tree(tree, 2)
    build_path = _direct(folder, 'runtime_build.json')
    build = _runtime_build(build_path)
    require(manifest['source'].get('runtime_build_sha256') == sha(build_path),
            'General AMR package runtime identity differs')
    tops = []
    expected = {TREE, 'runtime_build.json'}
    names = set()
    for index, row in enumerate(manifest['topologies']):
        prefix = f'topology_{index:04d}/'
        required = {'name', 'node_ids', 'operator', 'geometry', 'receipt',
                    'sidecars', 'identity'}
        require(isinstance(row, dict) and set(row) == required and
                isinstance(row['name'], str) and row['name'] and row['name'] not in names and
                row['node_ids'] == prefix + 'node_ids.npy' and
                row['operator'] == prefix + 'coupled_input.bin' and
                row['geometry'] == prefix + 'geometry.npz' and
                row['receipt'] == prefix + 'preparation_receipt.json' and
                isinstance(row['sidecars'], list) and
                isinstance(row['identity'], list) and
                row['identity'] == [prefix + name for name in IDENTITY_FILES],
                'General AMR topology manifest differs')
        names.add(row['name'])
        sidecars = tuple(name.removeprefix(prefix) for name in row['sidecars'])
        require(row['sidecars'] == [prefix + name for name in sidecars] and
                len(set(sidecars)) == len(sidecars),
                'General AMR sidecar path differs')
        files = (row['node_ids'], row['operator'], row['geometry'], row['receipt'],
                 *row['sidecars'], *row['identity'])
        require(all(name in artifacts for name in files),
                'General AMR topology artifact omitted')
        expected.update(files)
        ids = _leaves(np.load(_direct(folder, row['node_ids']), allow_pickle=False), tree)
        where = folder / f'topology_{index:04d}'
        data = _merged(where, ids, tree, sidecars, manifest['source'])
        receipt = _native_receipt(where / 'preparation_receipt.json',
            where / 'coupled_input.bin', sidecars, sha(build_path),
            build['executable_sha256'], check_origin=False)
        require(receipt['cells'] == len(ids) and receipt['faces'] == int(data['nf']),
                'General AMR receipt/operator dimensions differ')
        selected = _selection(where, ids, tree, manifest['source'])
        require(len(selected['face_source_ids']) == int(data['nf']),
                'General AMR source-face identity length differs')
        # Preserve the authenticated native candidate-face lineage in the
        # selected operator.  M2-C Thatcher--Harleman memory is face-owned and
        # may cross a regrid only through an exact one-to-one source ID; local
        # selected face indices are deliberately insufficient.
        data['face_source_ids'] = np.ascontiguousarray(
            selected['face_source_ids'], dtype=np.int64)
        with np.load(_direct(folder, row['geometry']), allow_pickle=False) as geometry:
            cell = geometry['cell_integer']
            require(cell.shape == (len(ids), 6) and
                    np.array_equal(cell[:, 0], np.arange(len(ids))) and
                    np.array_equal(cell[:, 4], tree['node_level'][ids]) and
                    np.array_equal(cell[:, 5], ids),
                    'General AMR polygon geometry differs from leaves')
        tops.append(AMRTopology(row['name'], ids.copy(), data, sha(where / 'coupled_input.bin')))
    require(set(artifacts) == expected,
            'General AMR package artifact inventory differs')
    first = tops[0].data
    for top in tops[1:]:
        require(all(key in first and key in top.data and
                    np.array_equal(first[key], top.data[key]) for key in GLOBAL_CONTROLS),
                'General AMR selected operators change global physical controls')
        require(all((key in first) == (key in top.data) and
                    (key not in first or np.array_equal(first[key], top.data[key]))
                    for key in AVALANCHE_CONTROLS),
                'General AMR selected operators change avalanche controls')
    prepared = PreparedAMR(folder, sha(manifest_path), tree, tuple(tops),
                           dict(manifest['source']), SCHEMA)
    # Native all-face identity is already copied into the package. Check it
    # against selected face geometry before any solver consumes this package.
    from .amr_operator_store import _face_lineage
    for index in range(len(tops)):
        _face_lineage(prepared, folder / f'topology_{index:04d}',
                      sha(folder / f'topology_{index:04d}' / 'amr_tree_input.bin'))
    return prepared


def export_general_amr_prepared_v3(destination, hierarchy, topologies: Sequence,
                                   *, runtime_build, source):
    """Copy one or more zero-step native selected operators into schema 3."""
    destination = filesystem_path(destination).resolve()
    require(not destination.exists(), 'General AMR package is write-once')
    tree = _tree_arrays(hierarchy)
    from .implicit_suspension_amr import PREPARED_BINDING_FIELD
    from .amr_typed_roughness import BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD
    nested = {PREPARED_BINDING_FIELD, AMR_ROUGHNESS_BINDING_FIELD}
    require(isinstance(topologies, Sequence) and topologies and
            isinstance(source, dict) and
            all(isinstance(key, str) and
                (key in nested and isinstance(value, dict) or
                 key not in nested and
                 isinstance(value, (str, int, float, bool)))
                for key, value in source.items()),
            'General AMR source/topology request differs')
    try:
        json.dumps(source, sort_keys=True, separators=(',', ':'),
                   allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('General AMR source is not canonical JSON') from error
    runtime_build = filesystem_path(runtime_build).resolve()
    build = _runtime_build(runtime_build)
    checked = []
    for name, ids, where in topologies:
        ids = _leaves(np.asarray(ids), tree)
        where = filesystem_path(where).resolve()
        core = _read_coupled(where / 'coupled_input.bin')
        sidecars = _sidecars(core)
        expected = {*CORE_FILES, *IDENTITY_FILES, *sidecars}
        require(all((where / filename).is_file() and not (where / filename).is_symlink()
                    for filename in expected) and
                {p.name for p in where.glob('*metadata.bin')} |
                {p.name for p in where.glob('coupled_diffusion.bin')} == set(sidecars),
                'General AMR native selected artifact inventory differs')
        data = _merged(where, ids, tree, sidecars, source)
        _native_receipt(where / 'preparation_receipt.json',
            where / 'coupled_input.bin', sidecars, sha(runtime_build),
            build['executable_sha256'], check_origin=True)
        require(int(data['nc']) == len(ids),
                'General AMR native selected cell count differs')
        checked.append((name, ids, where, sidecars))
    destination.mkdir(parents=True)
    np.savez_compressed(destination / TREE, **tree)
    shutil.copy2(runtime_build, destination / 'runtime_build.json')
    artifacts = {name: sha(destination / name)
                 for name in (TREE, 'runtime_build.json')}
    rows = []
    for index, (name, ids, where, sidecars) in enumerate(checked):
        location = destination / f'topology_{index:04d}'
        location.mkdir()
        np.save(location / 'node_ids.npy', ids)
        for filename in (*CORE_FILES, *sidecars, *IDENTITY_FILES):
            shutil.copy2(where / filename, location / filename)
        prefix = location.name + '/'
        row = dict(name=name, node_ids=prefix + 'node_ids.npy',
            operator=prefix + 'coupled_input.bin', geometry=prefix + 'geometry.npz',
            receipt=prefix + 'preparation_receipt.json',
            sidecars=[prefix + filename for filename in sidecars],
            identity=[prefix + filename for filename in IDENTITY_FILES])
        rows.append(row)
        for path in location.iterdir():
            artifacts[prefix + path.name] = sha(path)
    manifest = dict(kind=KIND, schema=SCHEMA, ratio=2, source=source,
        runtime_build='runtime_build.json', native_solver_steps=0,
        artifacts=artifacts, topologies=rows)
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2,
        sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return load_general_amr_prepared(destination)
