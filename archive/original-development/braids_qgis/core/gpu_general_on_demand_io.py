"""CUDA-free, read-only QGIS admission for nonbank general-transport AMR.

The schema-3 initializer and schema-5 frames are deliberately distinct from
bank AMR.  A run is self-contained for display; only restarting it requires
the original, authenticated preparation/session.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from braids_core.windows_paths import filesystem_path
from braids_core.gpu_live_boundary_metadata import (
    boundary_event_identity, validate_live_boundary_metadata)

from .gpu_amr_prepared_io import _artifact, _selected_nodes, _tree
from .gpu_amr_frame_io import _fail, _method_identity
from .gpu_frame_io import (_archive, _array, _digest, _geometry, _identity,
    _json, _number, _scalar, _source_identity, _string_scalar)
from .gpu_prepared_io import _read_records, read_gpu_prepared
from .gpu_on_demand_prepared import (_digest as _json_digest, _hex, _ids,
    _object, _regular, _selector, _sha)


VARIANT = 'on_demand_transport_amr_ssp33'
PACKAGE_KIND = 'braids-gpu-general-amr-prepared'
SESSION_KIND = 'on-demand-general-transport-amr'
RUNTIME_KIND = 'selected-general-transport-amr-v1'
SLOPE_POLICY = 'local_eta_plane_v1'
SLOPE_SOURCES = frozenset((
    'amr_slope_indicator_v2.py', 'amr_slope_indicator_research.py'))
RECURSIVE_CHECKPOINT = 'recursive_general_amr_v1'
RECURSIVE_STEPPER = 'recursive_ssprk33_fine_replaced_register_v1'
RECURSIVE_AVALANCHE_POLICY = 'parent_post_reflux_once_per_macro_v1'
RECURSIVE_AVALANCHE_METHOD = dict(policy=RECURSIVE_AVALANCHE_POLICY,
    placement='external_parent_post_reflux',calls_per_macro=1,
    conservative_per_fraction=True)
RECURSIVE_VISCOSITY_POLICY = 'parent_post_reflux_ssprk2_once_per_macro_v1'
RECURSIVE_VISCOSITY_METHOD = dict(policy=RECURSIVE_VISCOSITY_POLICY,
    placement='external_parent_post_reflux_before_avalanche',calls_per_macro=1,
    integrator='SSPRK2_subcycled_area_weighted_velocity_L2_limited',
    limited_norm='area_weighted_velocity_L2',
    area_weighted_velocity_l2_nonincrease=True,
    pairwise_velocity_face_flux=True,
    momentum_conservation='variable_depth_not_exact',
    dynamic_wet_geometry=True,general_avalanche_modes=[0,1],
    composition_order='viscosity_then_avalanche')
RECURSIVE_BEDFORM_METHOD = dict(
    policy='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False)
BEDFORM_FIELDS = frozenset(('bedform_effective_manning',
    'bedform_dune_height','bedform_dune_length','bedform_total_ks',
    'bedform_iterations'))
RECURSIVE_FRAME_FIELDS = frozenset((
    'recursive_level_steps','recursive_epoch_start_steps',
    'recursive_epoch_start_time','recursive_epoch_start_euler_calls',
    'recursive_register'))
_IDENTITY_FILES = ('amr_tree_input.bin', 'amr_selected_identity.bin',
                   'selection_receipt.json')
_GLOBAL_CONTROLS = ('nfrac', 'nstate', 'nbed_layers', 'bedlaw', 'exchange_mode',
    'secondary_flow_model', 'morphological_factor', 'general_avalanche',
    'bedform_roughness', 'nsubstrate_layers', 'suspended_transport_active',
    'horizontal_viscosity', 'horizontal_viscosity_mode',
    'horizontal_viscosity_geometry', 'mpm_friction_mode', 'mpm_cf',
    'hiding_exposure', 'bed_slope', 'bedload_adaptation',
    'bedload_adaptation_length_diameters', 'bulk_fraction', 'mode',
    'section_control', 'section_energy_slope', 'forcing',
    'fraction_concentration', 'fraction_bed_input', 'nseries',
    'nhydro_series', 'nfraction_series')
_AVALANCHE_CONTROLS = ('general_avalanche_wet_slope',
    'general_avalanche_dry_slope', 'general_avalanche_time',
    'general_avalanche_wet_depth_switch',
    'general_avalanche_relaxation_length')


def _require(test, message):
    if not test:
        _fail('General GPU AMR: ' + message)


def _session_live_boundaries(value, initial_data):
    """Validate the sole constant-boundary general-AMR session branch."""
    _require(type(value) is bool, 'session live-boundary flag differs')
    if value:
        return True
    from .gpu_on_demand_prepared import _recursive_general_bedform_scope
    _require(_recursive_general_bedform_scope(initial_data),
             'constant-boundary session is outside recursive bedform scope')
    return False


def _policy_fields(record):
    """Validate optional, versioned selector fields without upgrading legacy jobs."""
    policy = record.get('selector_policy')
    sources = record.get('selector_source_sha256')
    if policy is None:
        _require('selector_policy' not in record and
                 'selector_source_sha256' not in record,
                 'legacy selector has unexpected versioned fields')
        return None, None
    _require(policy == SLOPE_POLICY and isinstance(sources, dict) and
             set(sources) == SLOPE_SOURCES and
             all(_hex(value) for value in sources.values()),
             'unsupported selector policy or source hashes')
    return policy, sources


def _slope_selector_contract(legacy):
    """Reconstruct the declared CUDA-free wrapper identity from native controls."""
    baseline = {**legacy,
        'settings': {**legacy['settings'], 'surface_tolerance_m': 1e9},
        'coarsen_stage_span_m': 1e9}
    return dict(rule='eta_plane_residual_resident_forecast_v2',
        legacy_selector=_json_digest(legacy),
        no_eta_baseline=_json_digest(baseline),
        surface_tolerance_m=legacy['settings']['surface_tolerance_m'],
        coarsen_stage_span_m=legacy['coarsen_stage_span_m'],
        physical_state_download=False, forecast_kernel='gi_reach_abi2')


def _files(folder):
    result = {}
    for member in folder.rglob('*'):
        _require(not member.is_symlink() and member.resolve().is_relative_to(folder),
                 'linked or escaping package member')
        if member.is_file():
            result[member.relative_to(folder).as_posix()] = _sha(member)
        else:
            _require(member.is_dir(), 'non-file package member')
    return result


def _read_initial(folder, digest, sidecars, *, allow_constant_bedform=False):
    data = read_gpu_prepared(folder, receipt_sha256=digest)
    _require(int(data['bank_failure']) == 0 and
             int(data['bank_toe_erosion_mode']) == 0 and
             int(data['bank_front_tracking']) == 0,
             'operator enables bank physics')
    for name in sidecars:
        records = _read_records(folder / name)
        if name == 'coupled_diffusion.bin':
            _require(all('diffusion_' + key not in data for key in records),
                     'diffusion sidecar conflicts with operator')
            data.update({'diffusion_' + key: value for key, value in records.items()})
        elif name == 'normal_metadata.bin':
            try:
                data = validate_live_boundary_metadata(data, records)
            except ValueError:
                if not allow_constant_bedform:
                    raise
                # The exact recursive bedform pair carries the native boundary
                # table as an authenticated sidecar, but its constant mode-1/
                # mode-2 controls are deliberately not scheduled/live data.
                candidate = dict(data)
                for key, value in records.items():
                    _require(key not in candidate or
                             np.array_equal(candidate[key], value),
                             'constant boundary sidecar conflicts with operator: ' + key)
                    candidate[key] = value
                from .gpu_on_demand_prepared import _recursive_general_bedform_scope
                _require(_recursive_general_bedform_scope(candidate),
                         'constant boundary sidecar is outside recursive bedform scope')
                data = candidate
        else:
            for key, value in records.items():
                _require(key not in data or np.array_equal(data[key], value),
                         'sidecar conflicts with operator: ' + key)
                data[key] = value
    return data


def _package(folder, identity, *, originals, runtime_sha, tree_sha,
             allow_constant_bedform=False):
    folder = _regular(folder, directory=True)
    manifest_path = _regular(folder / 'amr_prepared.json')
    _require(_hex(identity) and _sha(manifest_path) == identity,
             'operator package identity differs')
    manifest = _object(manifest_path)
    _require(set(manifest) == {'kind', 'schema', 'ratio', 'source', 'artifacts',
        'topologies', 'runtime_build', 'native_solver_steps'} and
        manifest['kind'] == PACKAGE_KIND and manifest['schema'] == 3 and
        manifest['ratio'] == 2 and manifest['runtime_build'] == 'runtime_build.json' and
        manifest['native_solver_steps'] == 0 and
        isinstance(manifest['topologies'], list) and len(manifest['topologies']) == 1,
        'unsupported general operator package')
    source = manifest['source']
    _require(isinstance(source, dict) and source.get('on_demand_initialization') is True and
        source.get('base_input_sha256') == originals['case.data'] and
        source.get('hierarchy_sha256') == originals['amr_hierarchy.npz'] and
        source.get('runtime_build_sha256') == runtime_sha and
        source.get('live_boundary_metadata') is True,
        'operator original-input family differs')
    artifacts = manifest['artifacts']
    _require(isinstance(artifacts, dict) and artifacts and
             artifacts.get('runtime_build.json') == runtime_sha and
             'tree.npz' in artifacts and
             _files(folder) == {**artifacts, 'amr_prepared.json': identity},
             'operator artifact inventory or hashes differ')
    for name in artifacts:
        _artifact(folder, name)
    build = _object(folder / 'runtime_build.json')
    _require(build.get('schema') == 1 and build.get('kind') == RUNTIME_KIND and
             build.get('export_only') is True and
             build.get('numerical_routines_unchanged') is True and
             build.get('Delft3D_source_modified') is False and
             _hex(build.get('executable_sha256')),
             'native initializer provenance differs')
    tree = _tree(folder / 'tree.npz', 2)
    row = manifest['topologies'][0]
    prefix = 'topology_0000/'
    _require(isinstance(row, dict) and set(row) == {'name', 'node_ids',
        'operator', 'geometry', 'receipt', 'sidecars', 'identity'} and
        row['node_ids'] == prefix + 'node_ids.npy' and
        row['operator'] == prefix + 'coupled_input.bin' and
        row['geometry'] == prefix + 'geometry.npz' and
        row['receipt'] == prefix + 'preparation_receipt.json' and
        row['identity'] == [prefix + name for name in _IDENTITY_FILES] and
        isinstance(row['sidecars'], list), 'topology artifact layout differs')
    nodes = _selected_nodes(folder / row['node_ids'], tree)
    _require(row['name'] == 'leaves_' + hashlib.sha256(nodes.astype('<i8').tobytes()).hexdigest(),
             'selected topology name differs')
    topdir = folder / 'topology_0000'
    receipt = _object(topdir / 'preparation_receipt.json')
    sidecars = tuple(item.removeprefix(prefix) for item in row['sidecars'])
    expected_sidecars = ('normal_metadata.bin',) + (
        ('avalanche_metadata.bin',) if receipt.get('controls', {}).get('general_avalanche') else ()) + (
        ('coupled_diffusion.bin',) if 'coupled_diffusion.bin' in sidecars else ())
    _require(sidecars == expected_sidecars and
             row['sidecars'] == [prefix + name for name in sidecars] and
             set(artifacts) == {'runtime_build.json', 'tree.npz',
                row['node_ids'], row['operator'], row['geometry'], row['receipt'],
                *row['sidecars'], *row['identity']},
             'general operator optional sidecars differ')
    origin = receipt.get('origin', {}).get('files')
    _require(isinstance(origin, dict) and
             {key: value for key, value in origin.items() if key != 'selected_nodes.txt'} == originals and
             _hex(origin.get('selected_nodes.txt')) and
             receipt.get('schema') == 1 and receipt.get('kind') == RUNTIME_KIND and
             receipt.get('status') == 'completed' and
             receipt.get('runtime_build_sha256') == runtime_sha and
             receipt.get('runtime_executable_sha256') == build['executable_sha256'] and
             receipt.get('initialization_CPU_only') is True and
             receipt.get('GPU_solver_executed') is False and
             receipt.get('solver_steps_performed') == 0 and
             receipt.get('artifacts') == {name: artifacts[prefix + name]
                for name in ('coupled_input.bin', *sidecars)},
             'native zero-step receipt differs')
    selection = _read_records(topdir / 'amr_selected_identity.bin')
    selection_receipt = _object(topdir / 'selection_receipt.json')
    _require(set(selection) == {'node_ids', 'face_source_ids'} and
             np.array_equal(selection['node_ids'], nodes) and
             selection_receipt.get('node_ids') == nodes.tolist() and
             selection_receipt.get('native_solver_steps') == 0 and
             selection_receipt.get('full_state_transfer_on_GPU') is True and
             selection_receipt.get('tree_input_sha256') == tree_sha and
             selection_receipt.get('selection_identity_sha256') == artifacts[prefix + 'amr_selected_identity.bin'] and
             selection_receipt.get('original_input_sha256') == originals['case.data'] and
             artifacts[prefix + 'amr_tree_input.bin'] == tree_sha,
             'selected face/node identity differs')
    selected_tree = _read_records(topdir / 'amr_tree_input.bin')
    for key in ('parent', 'node_level', 'maximum_level', 'area'):
        _require(np.array_equal(selected_tree[key], tree[key]), 'selected hierarchy differs: ' + key)
    _require(np.array_equal(selected_tree['xy'].T, tree['xy']),
             'selected hierarchy coordinates differ')
    geometry = _geometry(topdir, {'origin': {'files': {'geometry.npz': artifacts[row['geometry']]}}})
    _require(geometry['cell_count'] == len(nodes) and
             np.allclose(geometry['area_m2'], tree['area'][nodes], rtol=1e-11, atol=1e-11) and
             np.allclose(geometry['centroid_xy'], tree['xy'][nodes], rtol=1e-11, atol=1e-11),
             'selected polygons differ from hierarchy')
    data = _read_initial(topdir, artifacts[row['receipt']], sidecars,
                         allow_constant_bedform=allow_constant_bedform)
    _require(int(data['general_avalanche']) == int('avalanche_metadata.bin' in sidecars) and
             (float(data['horizontal_viscosity']) > 0.) == ('coupled_diffusion.bin' in sidecars) and
             all(data.get(key) == value for key, value in receipt.get('controls', {}).items()),
             'operator physical sidecars/controls differ')
    _require(int(data['nc']) == len(nodes) and int(data['nf']) == len(selection['face_source_ids']) and
             receipt.get('cells') == len(nodes) and receipt.get('faces') == int(data['nf']) and
             np.array_equal(data['rock'], selected_tree['rock'][nodes]) and
             np.array_equal(data['state'], selected_tree['state'][:, nodes]) and
             np.allclose(data['area'], tree['area'][nodes], rtol=1e-12, atol=1e-12),
             'operator cold state or dimensions differ')
    levels = tree['node_level'][nodes]
    owner, other = data['owner'], data['neighbour']
    _require(not np.any(np.abs(levels[owner[other >= 0]] - levels[other[other >= 0]]) > 1) and
             np.all(levels[owner[data['boundary_faces']]] == max(tree['node_level'])),
             'operator 2:1 or open-boundary refinement differs')
    topology = dict(name=row['name'], folder=topdir, node_ids=nodes,
        node_levels=tree['node_level'][nodes].copy(),
        geometry=geometry, receipt_sha256=artifacts[row['receipt']],
        operator_sha256=artifacts[row['operator']], package_identity=identity,
        initial_data=data)
    return dict(root=folder, manifest_sha256=identity, tree=tree,
        topologies={row['name']: topology}, manifest=manifest,
        inventory=artifacts, topology=topology, receipt=receipt)


def inspect_prepared(folder, *, expected_receipt_sha256=None):
    """Authenticate a general-AMR cold session without importing a worker."""
    try:
        root = _regular(Path(folder).expanduser(), directory=True).resolve()
        receipt_path = root / 'preparation_receipt.json'
        receipt = _object(receipt_path)
        receipt_sha = _sha(receipt_path)
        _require(expected_receipt_sha256 is None or
                 (_hex(expected_receipt_sha256) and receipt_sha == expected_receipt_sha256),
                 'preparation receipt identity differs')
        _require(receipt.get('schema') == 6 and receipt.get('variant') == VARIANT and
                 receipt.get('native_solver_steps_performed') == 0 and
                 receipt.get('initialization_CPU_only') is True and
                 receipt.get('GPU_solver_executed') is False,
                 'unsupported general AMR preparation receipt')
        selector_policy, selector_sources = _policy_fields(receipt)
        session_root = _regular(root / 'session', directory=True)
        manifest = _object(session_root / 'session.json')
        _require(_sha(session_root / 'session.json') == receipt.get('session_manifest_sha256') and
                 manifest.get('identity') == receipt.get('session_identity') and
                 manifest.get('identity') == _json_digest({k: v for k, v in manifest.items() if k != 'identity'}) and
                 manifest.get('schema') == 3 and manifest.get('kind') == SESSION_KIND and
                 manifest.get('native_solver_steps') == 0 and
                 manifest.get('evolving_banks') is False and
                 type(manifest.get('live_boundaries')) is bool,
                 'session identity or nonbank controls differ')
        _require(_policy_fields(manifest) == (selector_policy, selector_sources),
                 'session selector policy differs from preparation receipt')
        originals = receipt.get('original_input_files')
        _require(isinstance(originals, dict) and
                 {'case.data', 'case.data.json', 'case.data.bin', 'amr_hierarchy.npz'} <= set(originals) and
                 originals == manifest.get('original_input_files'),
                 'original input inventory differs')
        deck = filesystem_path(manifest['deck'])
        _require(deck.is_absolute() and deck.name == 'case.data' and
                 Path(manifest['runtime']).is_absolute(),
                 'original deck/runtime path differs')
        from .gpu_prepared_io import _authenticated_member
        for name, digest in originals.items():
            actual = 'geometry.npz' if name == 'original_geometry.npz' else name
            _authenticated_member(deck.parent.resolve(), actual, digest, nested=False)
        request = _object(root / 'preparation_request.json')
        expected_request = dict(schema=3, variant=VARIANT, general_transport=True,
            deck=str(deck), deck_sha256=originals['case.data'],
            hierarchy_sha256=originals['amr_hierarchy.npz'],
            runtime_build_sha256=receipt['runtime_build_sha256'])
        if selector_policy is not None:
            expected_request.update(selector_policy=selector_policy,
                                    selector_source_sha256=selector_sources)
        _require(request == expected_request,
            'preparation request/session differs')
        bootstrap = _package(session_root / 'bootstrap', manifest['bootstrap_identity'],
            originals=originals, runtime_sha=receipt['runtime_build_sha256'],
            tree_sha=manifest['tree_input_sha256'],
            allow_constant_bedform=manifest['live_boundaries'] is False)
        # The qualified recursive bedform pair has one constant unsampled
        # discharge inlet and one constant stage outlet, so its session
        # honestly has no live/scheduled boundary metadata.  Authenticate the
        # complete initial operator before admitting that sole false branch;
        # historical general routes continue to require ``True``.
        _session_live_boundaries(manifest['live_boundaries'],
                                 bootstrap['topology']['initial_data'])
        native_tree = _read_records(_regular(session_root / 'bootstrap_native' / 'amr_tree_input.bin'))
        _require(_sha(session_root / 'bootstrap_native' / 'amr_tree_input.bin') ==
                 manifest['tree_input_sha256'], 'native hierarchy identity differs')
        settings = _selector(native_tree, bootstrap['tree'], manifest,
            initial_data=bootstrap['topology']['initial_data'],
            contract_transform=(_slope_selector_contract if selector_policy is not None else None))
        initial_ids = _ids(manifest['initial_node_ids'])
        proposal = manifest['initial_proposal']
        _require(proposal.get('catalog_identity') == bootstrap['manifest_sha256'] and
                 proposal.get('selector_contract_identity') == manifest['selector_contract'] and
                 proposal.get('old_operator_identity') == bootstrap['topology']['operator_sha256'] and
                 proposal.get('new_leaves') == len(initial_ids) and
                 proposal.get('leaf_digest') == hashlib.sha256(initial_ids.astype('<i8').tobytes()).hexdigest() and
                 proposal.get('time_s') == 0.,
                 'cold leaf proposal differs')
        store_root = _regular(session_root / 'store', directory=True)
        store = _object(store_root / 'store.json')
        base = dict(schema=3, kind='general-transport-amr-operator-store',
            hierarchy_sha256=originals['amr_hierarchy.npz'],
            base_input_sha256=originals['case.data'],
            runtime_build_sha256=receipt['runtime_build_sha256'],
            tree_input_sha256=manifest['tree_input_sha256'])
        _require(store == {**base, 'identity': _json_digest(base)},
                 'general operator store identity differs')
        entries = []
        initial = None
        seen = set()
        for path in sorted(_regular(store_root / 'operators', directory=True).iterdir()):
            _regular(path, directory=True)
            _require({p.name for p in path.iterdir()} == {'entry.json', 'package'},
                     'operator object inventory differs')
            row = _object(path / 'entry.json')
            nodes = _ids(row.get('selected_node_ids'))
            request_key = _json_digest(dict(store_identity=store['identity'],
                                            selected_node_ids=nodes.tolist()))
            _require(set(row) == {'schema', 'kind', 'store_identity', 'request_key',
                'selected_node_ids', 'package_identity', 'package_files'} and
                row['schema'] == 3 and row['kind'] == 'zero-step-general-amr-operator' and
                row['store_identity'] == store['identity'] and
                row['request_key'] == request_key and request_key not in seen and
                row['package_identity'] == path.name,
                'operator store entry differs')
            seen.add(request_key)
            package = _package(path / 'package', row['package_identity'],
                originals=originals, runtime_sha=receipt['runtime_build_sha256'],
                tree_sha=manifest['tree_input_sha256'],
                allow_constant_bedform=manifest['live_boundaries'] is False)
            if settings['subcycling']:
                from .gpu_on_demand_prepared import _recursive_general_bedform_scope
                _recursive_general_bedform_scope(
                    package['topology']['initial_data'])
            _require(row['package_files'] == _files(path / 'package') and
                     np.array_equal(package['topology']['node_ids'], nodes) and
                     all(np.array_equal(package['tree'][key], value)
                         for key, value in bootstrap['tree'].items()),
                     'cached selected operator differs')
            entries.append(dict(folder=path, request_key=request_key,
                package_identity=path.name, entry_sha256=_sha(path / 'entry.json'),
                selected_ids=nodes))
            if np.array_equal(nodes, initial_ids):
                initial = package
        _require(initial is not None, 'initial selected operator is absent')
        data = initial['topology']['initial_data']
        return dict(root=root, variant=VARIANT, receipt=receipt,
            receipt_sha256=receipt_sha, session_identity=manifest['identity'],
            session_manifest=manifest, session_root=session_root,
            selector_contract=manifest['selector_contract'], amr_settings=settings,
            selector_policy=selector_policy, selector_source_sha256=selector_sources,
            original_deck=deck, runtime_location=Path(manifest['runtime']),
            runtime_build=_object(bootstrap['root'] / 'runtime_build.json'),
            runtime_build_sha256=receipt['runtime_build_sha256'],
            original_input_files=dict(originals), bootstrap_index=bootstrap,
            initial_index=initial, initial_topology=initial['topology'],
            initial_data=data, store_identity=store['identity'], entries=entries,
            evolving_banks=False, general_transport=True, bank_mode=0,
            checkpoint_schema=('recursive_general_amr_v1'
                if settings['subcycling'] else 5),
            nc=int(data['nc']), nfrac=int(data['nfrac']),
            nbed_layers=int(data['nbed_layers']), time_s=float(data['time']),
            end_s=float(data['endtime']), output_every_s=float(data['output_dt']),
            dtmax_s=float(data['dtmax']), cfl=float(data['cfl_target']))
    except (OSError, KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError('Cannot inspect general on-demand GPU preparation: ' + str(exc)) from exc


def _run_preparation(root, manifest, prepared_root):
    expected = _identity(manifest.get('input_preparation_sha256'), 'general AMR preparation')
    _require(manifest.get('preparation_receipt_sha256') == expected and
             _digest(root / 'preparation_receipt.json') == expected,
             'copied preparation receipt differs')
    receipt = _json(root / 'preparation_receipt.json')
    selector_policy, selector_sources = _policy_fields(receipt)
    _require(_policy_fields(manifest['method']) ==
             (selector_policy, selector_sources),
             'run method selector differs from preparation receipt')
    _require(receipt.get('schema') == 6 and receipt.get('variant') == VARIANT and
             receipt.get('native_solver_steps_performed') == 0 and
             receipt.get('initialization_CPU_only') is True and
             receipt.get('GPU_solver_executed') is False and
             manifest.get('input_identity') == receipt.get('session_identity') and
             manifest['method'].get('session_identity') == receipt.get('session_identity'),
             'run preparation/session identity differs')
    for name in ('session_identity', 'session_manifest_sha256', 'runtime_build_sha256'):
        _identity(receipt.get(name), name)
    originals = receipt.get('original_input_files')
    _require(isinstance(originals, dict) and
             {'case.data', 'case.data.json', 'case.data.bin', 'amr_hierarchy.npz'} <= set(originals),
             'run original input family differs')
    for name, digest in originals.items():
        _require(Path(name).name == name, 'unsafe original input name')
        _identity(digest, 'original input ' + name)
    if prepared_root is not None:
        prepared = filesystem_path(Path(prepared_root).expanduser()).resolve()
        _require(_digest(prepared / 'preparation_receipt.json') == expected and
                 _digest(prepared / 'session' / 'session.json') == receipt['session_manifest_sha256'],
                 'supplied preparation/session differs')
    return receipt


def _index(root, manifest, receipt, *, live=False):
    rows = manifest.get('operators')
    if (live and rows == [] and manifest.get('topologies') == [] and
            manifest.get('frames') == [] and manifest.get('status') == 'initializing'):
        return dict(root=root, manifest_sha256=manifest['input_preparation_sha256'],
                    tree=None, topologies={})
    _require(isinstance(rows, list) and rows, 'missing registered operators')
    operator_root = root / 'operators'
    _require(operator_root.is_dir() and not operator_root.is_symlink(),
             'operator folder is missing or linked')
    topologies = {}
    tree = None
    identities = set()
    tree_sha = None
    method = manifest.get('method', {})
    recursive = method.get('checkpoint_schema') == RECURSIVE_CHECKPOINT
    for row in rows:
        _require(isinstance(row, dict) and
                 set(row) == {'name', 'path', 'package_identity', 'operator_identity'},
                 'operator inventory row differs')
        identity = _identity(row['package_identity'], 'general operator package')
        _identity(row['operator_identity'], 'general operator')
        _require(row['path'] == 'operators/' + identity and identity not in identities,
                 'operator path/order differs')
        identities.add(identity)
        package = _package(operator_root / identity, identity,
            originals=receipt['original_input_files'],
            runtime_sha=receipt['runtime_build_sha256'],
            tree_sha=tree_sha or _digest(operator_root / identity / 'topology_0000' / 'amr_tree_input.bin'),
            allow_constant_bedform=recursive)
        topology = package['topology']
        if manifest.get('method', {}).get('checkpoint_schema') == RECURSIVE_CHECKPOINT:
            from .gpu_on_demand_prepared import _recursive_general_bedform_scope
            _recursive_general_bedform_scope(topology['initial_data'])
        _require(row['name'] == topology['name'] and
                 row['operator_identity'] == topology['operator_sha256'] and
                 topology['name'] not in topologies,
                 'operator/topology binding differs')
        if tree is None:
            tree = package['tree']
            tree_sha = _digest(operator_root / identity / 'topology_0000' / 'amr_tree_input.bin')
        else:
            _require(all(np.array_equal(package['tree'][key], value)
                         for key, value in tree.items()) and
                     _digest(operator_root / identity / 'topology_0000' / 'amr_tree_input.bin') == tree_sha,
                     'operator hierarchy changed during run')
            reference = next(iter(topologies.values()))['initial_data']
            current = topology['initial_data']
            _require(all(np.array_equal(reference[key], current[key])
                         for key in _GLOBAL_CONTROLS) and
                     all((key in reference) == (key in current) and
                         (key not in reference or
                          np.array_equal(reference[key], current[key]))
                         for key in _AVALANCHE_CONTROLS) and
                     all(np.array_equal(reference[key], current[key])
                         for key in reference if key.startswith((
                            'series_', 'hydro_series_', 'fraction_series_'))),
                     'operator changed global transport/boundary controls')
        topologies[topology['name']] = topology
    _require(manifest.get('topologies') == list(topologies),
             'registered topology order differs')
    actual = {member.name for member in operator_root.iterdir()}
    if actual != identities:
        from .gpu_on_demand_frame_io import _pending_operator_directory, _ACTIVE_STATUSES
        extra = [member for member in operator_root.iterdir() if member.name not in identities]
        _require(live and manifest.get('status') in _ACTIVE_STATUSES and
                 identities <= actual and all(_pending_operator_directory(p) for p in extra),
                 'unregistered operator directory differs')
    return dict(root=root, manifest_sha256=manifest['input_preparation_sha256'],
                tree=tree, topologies=topologies)


def _method(manifest, index, receipt):
    method = manifest['method']
    _require(_policy_fields(method) == _policy_fields(receipt),
             'general AMR method selector differs')
    recursive=method.get('checkpoint_schema') == RECURSIVE_CHECKPOINT
    _require(method.get('variant') == VARIANT and
             (recursive or method.get('checkpoint_schema') == 5) and
             method.get('subcycling') is recursive and
             not method.get('evolving_banks', False) and 'bank_mode' not in method and
             ((recursive and method.get('level_stepping') == RECURSIVE_STEPPER) or
              (not recursive and 'level_stepping' not in method)) and
             method.get('integrator') == 'SSPRK33' and
             method.get('launch_mode') == 'plain' and
             method.get('precision') == 'FP64',
             'general AMR method/stepper differs')
    if not index['topologies']:
        return
    first = next(iter(index['topologies'].values()))['initial_data']
    from .gpu_on_demand_prepared import _recursive_general_bedform_scope
    bedform_branch = (recursive and
        _recursive_general_bedform_scope(first))
    expected = dict(bedlaw=int(first['bedlaw']),
        exchange_mode=int(first['exchange_mode']),
        secondary_flow_model=int(first['secondary_flow_model']),
        bedload_adaptation=bool(first['bedload_adaptation']),
        general_avalanche=bool(first['general_avalanche']),
        horizontal_viscosity=float(first['horizontal_viscosity']),
        state_transfer=('conservative_all_rows_closed_recursive_epoch_v1'
                        if recursive else
                        'conservative_all_rows_synchronous_v1'))
    if bool(first.get('bedform_roughness', 0)) or bedform_branch:
        expected['bedform_roughness'] = bool(first.get('bedform_roughness', 0))
    _require(method.get('general_transport') == expected,
             'general AMR physical method differs from operators')
    if recursive and bool(first['bedload_adaptation']):
        _require(method.get('bedload_adaptation') == dict(
            policy='full_leaf_steady_v1',
            length_diameters=float(first['bedload_adaptation_length_diameters'])),
            'recursive general AMR adaptation method differs')
    else:
        _require('bedload_adaptation' not in method,
                  'general AMR adaptation-OFF method carries an ON closure')
    if recursive and bool(first['general_avalanche']):
        _require(method.get('recursive_general_avalanche') ==
                 RECURSIVE_AVALANCHE_METHOD,
                 'recursive general avalanche method differs')
    else:
        _require('recursive_general_avalanche' not in method,
                 'general AMR avalanche-OFF method carries an ON closure')
    if recursive and float(first['horizontal_viscosity']) > 0.:
        _require(method.get('recursive_general_viscosity') ==
                 RECURSIVE_VISCOSITY_METHOD,
                 'recursive general viscosity method differs')
    else:
        _require('recursive_general_viscosity' not in method,
                 'general AMR viscosity-OFF method carries an ON closure')
    if recursive and bool(first.get('bedform_roughness', 0)):
        _require(method.get('recursive_general_bedform') ==
                 RECURSIVE_BEDFORM_METHOD,
                 'recursive general bedform method differs')
    else:
        _require('recursive_general_bedform' not in method,
                 'general AMR bedform-OFF method carries an ON closure')
    forcing = method.get('boundary_forcing')
    _require(forcing == dict(schema=1, event_identity=boundary_event_identity(first),
        stage_fractions=[0., 1., .5], clock='forcing_knot_clipped'),
        'general AMR boundary event clock differs')


def _common(run_root, prepared_root=None, *, live=False):
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    manifest = _json(root / 'run.json')
    status = _json(root / 'status.json')
    _require(manifest.get('schema') == 3 and
             manifest.get('format') == 'braids-gpu-amr-run-v2' and
             manifest.get('input_kind') == 'amr-on-demand-general' and
             manifest.get('backend') == 'gpu' and
             manifest.get('backend_selected_by') == 'user' and
             manifest.get('automatic_backend_selection') is False,
             'run is not explicit general GPU AMR')
    _method_identity(manifest)
    _source_identity(root, manifest)
    receipt = _run_preparation(root, manifest, prepared_root)
    index = _index(root, manifest, receipt, live=live)
    _method(manifest, index, receipt)
    from .gpu_on_demand_frame_io import _STATUSES, _ACTIVE_STATUSES, _SnapshotAdvanced
    _require(status.get('backend') == 'gpu' and
             status.get('status') in _STATUSES and manifest.get('status') in _STATUSES,
             'worker status differs')
    if status['status'] != manifest['status']:
        if live and _json(root / 'run.json') != manifest:
            raise _SnapshotAdvanced()
        pending = ((status['status'] == 'initializing' and manifest['status'] != 'initializing') or
                   (status['status'] == 'running' and manifest['status'] not in _ACTIVE_STATUSES))
        _require(live and pending, 'worker status differs from run manifest')
    return root, manifest, status, index


def _frame(path, row, manifest, topology, index):
    archive = _archive(path)
    data = topology['initial_data']
    n, ng, nl, nb = (len(topology['node_ids']), int(data['nfrac']),
                     int(data['nbed_layers']), int(data['nb']))
    recursive=manifest['method']['checkpoint_schema'] == RECURSIVE_CHECKPOINT
    required = {'frame_schema', 'state', 'node_ids', 'topology_name',
        'operator_identity', 'package_identity', 'topology_identity',
        'regrid_mesh_index', 'time_s', 'steps', 'input_identity',
        'source_identity', 'method_identity', 'dt_hint', 'rejected',
        'ledger_water', 'ledger_bed', 'ledger_secondary', 'ledger_source'}
    suspended = int(data['exchange_mode']) != 0
    if suspended:
        required |= {'ledger_suspended', 'ledger_exchange'}
    if recursive:
        required |= RECURSIVE_FRAME_FIELDS
    bedform = bool(data.get('bedform_roughness', 0))
    if bedform:
        required |= BEDFORM_FIELDS
    _require(set(archive) == required and
             int(_scalar(archive, 'frame_schema', 'int64')) == (6 if recursive else 5) and
             _string_scalar(archive, 'topology_name') == topology['name'] and
             _string_scalar(archive, 'operator_identity') == topology['operator_sha256'] and
             _string_scalar(archive, 'topology_identity') == topology['operator_sha256'] and
             _string_scalar(archive, 'package_identity') == topology['package_identity'] and
             int(_scalar(archive, 'regrid_mesh_index', 'int64')) == index,
             'frame topology/schema differs')
    for key, expected in (('input_identity', manifest['input_preparation_sha256']),
                          ('source_identity', manifest['source_identity']),
                          ('method_identity', manifest['method_identity'])):
        _require(_string_scalar(archive, key) == expected,
                 'frame identity differs: ' + key)
    _require(np.array_equal(_array(archive, 'node_ids', (n,), 'int64'), topology['node_ids']),
             'frame selected nodes differ')
    state = _array(archive, 'state', (n, int(data['nstate'])), 'float64')
    bed_indices = np.asarray(data['bed_indices'])
    suspended_indices = np.asarray(data['suspended_indices'])
    physical_width = 3 + ng + nl * ng
    secondary_index = int(data['secondary_index'])
    has_secondary = secondary_index >= 0
    _require(bed_indices.shape == (nl, ng) and suspended_indices.shape == (ng,) and
             secondary_index == (physical_width if has_secondary else -1) and
             int(data['nstate']) == physical_width + int(has_secondary) and
             np.all(state[:, 0] >= 0) and
             np.all(state[:, bed_indices.ravel()] >= 0) and
             np.all(state[:, suspended_indices] >= 0),
             'frame physical state layout or positivity differs')
    dry = state[:, 0] == 0
    _require(not np.any(state[dry, 1:3]) and
             not np.any(state[dry][:, suspended_indices]) and
             (not has_secondary or not np.any(state[dry, secondary_index])),
             'frame dry cells carry momentum or suspended material')
    _array(archive, 'ledger_water', (nb,), 'float64')
    _array(archive, 'ledger_bed', (ng, nb), 'float64')
    _array(archive, 'ledger_secondary', (nb,), 'float64')
    _scalar(archive, 'ledger_source', 'float64')
    if suspended:
        _array(archive, 'ledger_suspended', (ng, nb), 'float64')
        exchange = _array(archive, 'ledger_exchange', (ng, 2), 'float64')
        _require(np.max(np.abs(np.sum(exchange, axis=1))) <= 1e-8,
                 'frame exchange ledger is unbalanced')
    bedform_values = None
    if bedform:
        bedform_values = {key:_array(archive,key,(n,),
            'int32' if key == 'bedform_iterations' else 'float64').copy()
            for key in BEDFORM_FIELDS}
        _require(all(np.all(value >= 0) for value in bedform_values.values()),
                 'frame bedform diagnostics are negative')
    time_s = float(_scalar(archive, 'time_s', 'float64'))
    steps = int(_scalar(archive, 'steps', 'int64'))
    _require(time_s >= 0 and steps >= 0 and
             math.isclose(time_s, row['time_s'], abs_tol=1e-12, rel_tol=1e-12) and
             steps == row['steps'] and
             int(_scalar(archive, 'rejected', 'int64')) >= 0 and
             float(_scalar(archive, 'dt_hint', 'float64')) > 0,
             'frame accepted clock/trial controls differ')
    bulk = np.asarray(data['bulk_fraction'], dtype=np.float64)
    rock = np.asarray(data['rock'], dtype=np.float64)
    _require(bulk.shape == (ng,) and np.all(bulk > 0) and rock.shape == (n,),
             'frame bed material geometry differs')
    bed = rock + np.sum(state[:, bed_indices] / bulk[None, None, :], axis=(1, 2))
    recursive_values=None
    if recursive:
        levels=np.asarray(topology['node_levels'])
        root,finest=int(np.min(levels)),int(np.max(levels))
        level_steps=_array(archive,'recursive_level_steps',(finest+1,),'int64')
        origin=int(_scalar(archive,'recursive_epoch_start_steps','int64'))
        origin_time=float(_scalar(archive,'recursive_epoch_start_time','float64'))
        origin_calls=int(_scalar(archive,'recursive_epoch_start_euler_calls','int64'))
        expected=np.zeros(finest+1,np.int64)
        for level in range(root,finest+1):
            if np.any(levels==level):
                expected[level]=(steps-origin)*(1 << (level-root))
        register=_array(archive,'recursive_register',
                        (int(data['nf']),2,int(data['nstate'])),'float64')
        bed=np.asarray(data['bed_indices'])
        advected=np.r_[np.arange(3),np.asarray(data['suspended_indices']),bed[0],
                       [secondary_index] if has_secondary else []].astype(int)
        inactive=np.setdiff1d(np.arange(int(data['nstate'])),advected)
        symmetric=np.r_[np.asarray(data['suspended_indices']),bed[0],
                        [secondary_index] if has_secondary else []].astype(int)
        _require(0 <= origin <= steps and 0 <= origin_time <= time_s and
                 origin_calls >= 0 and origin_calls % 3 == 0 and
                 (origin == 0) == (origin_time == 0.) == (origin_calls == 0) and
                 np.array_equal(level_steps,expected) and
                 not np.any(register[:,:,inactive]) and
                 np.array_equal(register[:,0][:,symmetric],
                                register[:,1][:,symmetric]) and
                 (steps > origin or not np.any(register)),
                 'recursive frame clock/register differs')
        recursive_values={key:archive[key].copy()
                          for key in RECURSIVE_FRAME_FIELDS}
    return dict(time_s=time_s, steps=steps, mesh_id=index,
        topology_name=topology['name'], node_ids=topology['node_ids'].copy(),
        q=state[:, :physical_width].copy(),
        secondary=(state[:, secondary_index].copy() if has_secondary else np.zeros(n)),
        bed_elevation=bed, nfrac=ng, bed_layers=nl,
        suspended_enabled=suspended, bank_history=None,
        bedform=bedform_values,
        recursive=recursive_values,
        ledger={key: archive[key].copy() for key in sorted(required)
                if key.startswith('ledger_')})


def read_run(run_root, *, prepared_root=None):
    """Authenticate all published schema-5 general-AMR frames for QGIS."""
    from .gpu_on_demand_frame_io import _rows, _status
    root, manifest, status, index = _common(run_root, prepared_root)
    rows, last_time, last_steps = _rows(manifest, index, require_nonempty=True)
    frames = []
    for row in rows:
        path = root / row['file']
        _require(not path.is_symlink() and _digest(path) == row['sha256'],
                 'frame hash differs: ' + row['file'])
        topology = index['topologies'][row['topology']]
        frame = _frame(path, row, manifest, topology,
                       list(index['topologies']).index(row['topology']))
        _require(not frames or (frame['nfrac'], frame['bed_layers']) ==
                 (frames[0]['nfrac'], frames[0]['bed_layers']),
                 'material layout changed across meshes')
        frames.append(frame)
    if manifest['status'] in {'completed', 'step_limit', 'cancelled'}:
        _require({p.name for p in root.glob('frame_*.npz')} ==
                 {row['file'] for row in rows}, 'unlisted result frame')
    _status(status, manifest, last_time, last_steps)
    return dict(manifest=manifest, status=status,
                prepared_index=index, frames=frames)


def read_live_frame(run_root, *, prepared_root=None, after_index=-1):
    """Read only the newest committed general-AMR preview frame."""
    from .gpu_on_demand_frame_io import _rows, _status, _SnapshotAdvanced
    _require(type(after_index) is int and after_index >= -1,
             'invalid live-preview cursor')
    root = filesystem_path(Path(run_root).expanduser()).resolve()
    if not (root / 'run.json').is_file():
        return None
    try:
        root, manifest, status, index = _common(root, prepared_root, live=True)
    except _SnapshotAdvanced:
        return None
    rows, last_time, last_steps = _rows(manifest, index, require_nonempty=False)
    latest = len(rows) - 1
    if latest <= after_index:
        return None
    row = rows[latest]
    path = root / row['file']
    _require(not path.is_symlink() and _digest(path) == row['sha256'],
             'live frame hash differs')
    topology = index['topologies'][row['topology']]
    frame = _frame(path, row, manifest, topology,
                   list(index['topologies']).index(row['topology']))
    if not _status(status, manifest, last_time, last_steps, live=True):
        return None
    return dict(frame_index=latest, frame=frame, manifest=manifest,
                topology=topology, prepared_root=root)


def _fields_digest(fields):
    digest = hashlib.sha256()
    for key in sorted(fields):
        value = np.ascontiguousarray(fields[key])
        digest.update(key.encode() + b'\0')
        digest.update(value.dtype.str.encode() + b'\0')
        digest.update(json.dumps(value.shape).encode() + b'\0')
        digest.update(value.tobytes())
    return digest.hexdigest()


def inspect_resume(run_root, prepared_root):
    """Authenticate a manifest-selected accepted general-AMR restart on the host."""
    from .gpu_resume import (_amr_number, _amr_regular, _CHECKPOINT,
        _CHECKPOINT_FILES, _RESUMABLE)
    from .gpu_on_demand_prepared import inspect_gpu_on_demand_prepared
    root = _amr_regular(Path(run_root).expanduser(), directory=True).resolve()
    manifest = _object(root / 'run.json')
    _require(manifest.get('schema') == 3 and
             manifest.get('input_kind') == 'amr-on-demand-general' and
             manifest.get('status') in _RESUMABLE,
             'selected run is not resumable general AMR')
    descriptor = manifest.get('checkpoint')
    _require(isinstance(descriptor, dict) and set(descriptor) ==
             {'path', 'worker_sha256', 'payload_sha256', 'operators_sha256'},
             'checkpoint pointer differs')
    relative = descriptor['path']
    _require(isinstance(relative, str) and relative.startswith('checkpoints/') and
             relative.count('/') == 1, 'checkpoint path is unsafe')
    match = _CHECKPOINT.fullmatch(relative.split('/', 1)[1])
    _require(match is not None, 'checkpoint path/step revision differs')
    checkpoint = _amr_regular(root / relative, directory=True)
    _require(checkpoint.resolve().is_relative_to(root) and
             {p.name for p in checkpoint.iterdir()} == _CHECKPOINT_FILES,
             'checkpoint file inventory differs')
    for name, key in (('worker.json', 'worker_sha256'),
                      ('state.npz', 'payload_sha256'),
                      ('operators.json', 'operators_sha256')):
        _require(_sha(_regular(checkpoint / name)) == descriptor[key],
                 'checkpoint file hash differs: ' + name)
    prepared = inspect_gpu_on_demand_prepared(prepared_root,
        expected_receipt_sha256=manifest['input_preparation_sha256'])
    run = read_run(root, prepared_root=prepared_root)
    _require(run['manifest'] == manifest, 'run commit moved during inspection')
    status, index = run['status'], run['prepared_index']
    method = manifest['method']
    recursive=method.get('checkpoint_schema') == RECURSIVE_CHECKPOINT
    settings = prepared['amr_settings']
    data = prepared['initial_data']
    from .gpu_on_demand_prepared import _recursive_general_bedform_scope
    bedform_branch = (recursive and
        _recursive_general_bedform_scope(data))
    _require(method.get('variant') == VARIANT and
             method.get('checkpoint_schema') ==
                (RECURSIVE_CHECKPOINT if recursive else 5) and
             method.get('session_identity') == prepared['session_identity'] and
             method.get('selector_contract') == prepared['selector_contract'] and
             _policy_fields(method) ==
             (prepared['selector_policy'], prepared['selector_source_sha256']) and
             method.get('cfl') == prepared['cfl'] and
             method.get('regrid_interval_s') == settings['regrid_interval_s'] and
             method.get('general_transport') == dict(
                bedlaw=int(data['bedlaw']), exchange_mode=int(data['exchange_mode']),
                secondary_flow_model=int(data['secondary_flow_model']),
                bedload_adaptation=bool(data['bedload_adaptation']),
                general_avalanche=bool(data.get('general_avalanche', 0)),
                horizontal_viscosity=float(data['horizontal_viscosity']),
                **({'bedform_roughness':bool(data.get('bedform_roughness', 0))}
                   if bool(data.get('bedform_roughness', 0)) or bedform_branch else {}),
                state_transfer=('conservative_all_rows_closed_recursive_epoch_v1'
                                if recursive else
                                'conservative_all_rows_synchronous_v1')) and
              method.get('boundary_forcing') == dict(schema=1,
                 event_identity=boundary_event_identity(data),
                 stage_fractions=[0., 1., .5], clock='forcing_knot_clipped'),
              'checkpoint method differs from prepared controls')
    if recursive and bool(data.get('general_avalanche', 0)):
        _require(method.get('recursive_general_avalanche') ==
                 RECURSIVE_AVALANCHE_METHOD,
                 'checkpoint recursive general avalanche method differs')
    else:
        _require('recursive_general_avalanche' not in method,
                 'checkpoint avalanche-OFF method carries an ON closure')
    if recursive and float(data['horizontal_viscosity']) > 0.:
        _require(method.get('recursive_general_viscosity') ==
                 RECURSIVE_VISCOSITY_METHOD,
                 'checkpoint recursive general viscosity method differs')
    else:
        _require('recursive_general_viscosity' not in method,
                 'checkpoint viscosity-OFF method carries an ON closure')
    if recursive and bool(data.get('bedform_roughness', 0)):
        _require(method.get('recursive_general_bedform') ==
                 RECURSIVE_BEDFORM_METHOD,
                 'checkpoint recursive general bedform method differs')
    else:
        _require('recursive_general_bedform' not in method,
                 'checkpoint bedform-OFF method carries an ON closure')
    dtmax = _amr_number(method.get('dtmax_s'), 'maximum timestep', positive=True)
    interval = _amr_number(method.get('output_every_s'), 'output interval', positive=True)
    start = _amr_number(manifest.get('start_s'), 'start time')
    end = _amr_number(manifest.get('end_s'), 'end time', positive=True)
    _require(dtmax <= prepared['dtmax_s'] and start == prepared['time_s'] and
             start < end <= prepared['end_s'] and
             manifest.get('dtmax_s') == dtmax and
             manifest.get('output_every_s') == interval,
             'run timestep/clock exceeds prepared input')
    worker = _object(checkpoint / 'worker.json')
    _require(set(worker) == {'schema', 'method_identity', 'input_identity',
        'source_identity', 'time_s', 'steps', 'dt_hint', 'rejected',
        'payload_sha256', 'operators_sha256'} and
        worker['schema'] == 1 and
        worker['method_identity'] == manifest['method_identity'] and
        worker['input_identity'] == prepared['session_identity'] and
        worker['source_identity'] == manifest['source_identity'] and
        all(worker[key] == descriptor[key]
            for key in ('payload_sha256', 'operators_sha256')),
        'checkpoint worker identity differs')
    current = _amr_number(worker['time_s'], 'accepted time')
    steps = _amr_number(worker['steps'], 'accepted steps', integer=True)
    rejected = _amr_number(worker['rejected'], 'rejected trials', integer=True)
    hint = _amr_number(worker['dt_hint'], 'timestep hint', positive=True)
    _require(start <= current <= end + 1e-12 and hint <= dtmax and
             int(match[1]) == steps, 'checkpoint accepted clock differs')
    if match[2] is not None:
        hint_digest = _json_digest(hint)[:16]
        _require(int(match[2]) == rejected and match[3] == hint_digest,
                 'checkpoint trial revision differs')
    for row in (manifest, status):
        _require(_amr_number(row.get('time_s'), 'published time') == current and
                 _amr_number(row.get('steps'), 'published steps', integer=True) == steps and
                 _amr_number(row.get('rejected'), 'published rejected', integer=True) == rejected,
                 'checkpoint differs from published clock')
    _require(status.get('end_s') == end and
             (manifest['status'] != 'completed' or
              math.isclose(current, end, rel_tol=0, abs_tol=1e-12)),
             'completed run/end clock differs')
    receipt = _object(checkpoint / 'checkpoint.json')
    receipt_keys={'schema', 'session_identity', 'catalog_identity',
        'topology_identity', 'method_identity', 'payload_file',
        'payload_file_sha256', 'payload_sha256', 'accepted_steps',
        'time_s', 'event_count'} | ({'kind'} if recursive else set())
    _require(set(receipt) == receipt_keys and
        receipt['schema'] == (6 if recursive else 5) and
        (not recursive or receipt['kind'] == 'recursive_general_amr') and
        receipt['session_identity'] == prepared['session_identity'] and
        receipt['catalog_identity'] == prepared['session_identity'] and
        receipt['method_identity'] == manifest['method_identity'] and
        receipt['payload_file'] == 'state.npz' and
        receipt['payload_file_sha256'] == descriptor['payload_sha256'] and
        receipt['accepted_steps'] == steps and receipt['time_s'] == current,
        'general checkpoint receipt differs')
    inventory = _object(checkpoint / 'operators.json')
    _require(set(inventory) == {'schema', 'session_identity',
        'checkpoint_sha256', 'entries'} and inventory['schema'] == 1 and
        inventory['session_identity'] == prepared['session_identity'] and
        inventory['checkpoint_sha256'] == _sha(checkpoint / 'checkpoint.json') and
        isinstance(inventory['entries'], list) and inventory['entries'],
        'checkpoint operator inventory differs')
    known = {entry['package_identity']: dict(
        selected_ids=entry['selected_ids'].tolist(),
        request_key=entry['request_key'], package_identity=entry['package_identity'],
        entry_sha256=entry['entry_sha256']) for entry in prepared['entries']}
    _require(all(row == known.get(row.get('package_identity'))
                 for row in inventory['entries']) and
             [row['package_identity'] for row in inventory['entries']] ==
             [row['package_identity'] for row in manifest['operators']] and
             inventory['entries'][0]['selected_ids'] == prepared['session_manifest']['initial_node_ids'] and
             all(np.array_equal(index['tree'][key], value)
                 for key, value in prepared['initial_index']['tree'].items()),
             'checkpoint cached operators/hierarchy differ')
    fields = _archive(checkpoint / 'state.npz')
    stored_digest = _string_scalar(fields, 'payload_sha256')
    fields_without = {key: value for key, value in fields.items()
                      if key != 'payload_sha256'}
    _require(stored_digest == receipt['payload_sha256'] and
             _fields_digest(fields_without) == stored_digest,
             'checkpoint field digest differs')
    topology_candidates = [top for top in index['topologies'].values()
        if top['operator_sha256'] == receipt['topology_identity']]
    _require(len(topology_candidates) == 1,
             'checkpoint selected topology is unregistered')
    topology = topology_candidates[0]
    physical = topology['initial_data']
    nn, n, ns = len(index['tree']['parent']), len(topology['node_ids']), int(physical['nstate'])
    ng, nb = int(physical['nfrac']), int(physical['nb'])
    suspended = int(physical['exchange_mode']) != 0
    required = {'schema', 'session_identity', 'catalog_identity',
        'topology_identity', 'method_identity', 'node_ids', 'state',
        'tree_state', 'delay_state', 'time_s', 'accepted_steps',
        'euler_calls', 'maximum_cfl', 'events_json', 'ledger_water',
        'ledger_bed', 'ledger_secondary', 'ledger_source'}
    if suspended:
        required |= {'ledger_suspended', 'ledger_exchange'}
    if settings['refine_sediment']:
        required.add('selector_last_bed')
    recursive_checkpoint_fields={'recursive_level_steps',
        'recursive_epoch_start_steps','recursive_epoch_start_time_s',
        'recursive_epoch_start_euler_calls','recursive_register',
        'recursive_last_source','recursive_last_exchange'}
    if recursive:
        required |= recursive_checkpoint_fields
    _require(set(fields_without) == required and
             int(_scalar(fields, 'schema', 'int64')) == (6 if recursive else 5) and
             _string_scalar(fields, 'session_identity') == prepared['session_identity'] and
             _string_scalar(fields, 'catalog_identity') == prepared['session_identity'] and
             _string_scalar(fields, 'topology_identity') == topology['operator_sha256'] and
             _string_scalar(fields, 'method_identity') == manifest['method_identity'] and
             np.array_equal(_array(fields, 'node_ids', (n,), 'int64'), topology['node_ids']),
             'checkpoint payload identity/layout differs')
    state = _array(fields, 'state', (n, ns), 'float64')
    tree_state = _array(fields, 'tree_state', (nn, ns), 'float64')
    delay = _array(fields, 'delay_state', (nn,), 'int64')
    _require(np.all(delay >= 0) and np.array_equal(tree_state[topology['node_ids']], state),
             'checkpoint accepted tree state differs')
    sediment = np.r_[physical['suspended_indices'].ravel(),
                     physical['bed_indices'].ravel()]
    for value in (state, tree_state):
        dry = value[:, 0] == 0
        _require(np.all(value[:, 0] >= 0) and np.all(value[:, sediment] >= 0) and
                 not np.any(value[dry, 1:3]) and
                 not np.any(value[dry][:, physical['suspended_indices']]) and
                 (int(physical['secondary_index']) < 0 or
                  not np.any(value[dry, int(physical['secondary_index'])])),
                 'checkpoint nonphysical water/sediment inventory')
    if settings['refine_sediment']:
        _array(fields, 'selector_last_bed', (nn,), 'float64')
    for name, shape in (('ledger_water', (nb,)), ('ledger_bed', (ng, nb)),
                        ('ledger_secondary', (nb,)), ('ledger_source', ())):
        _array(fields, name, shape, 'float64')
    if suspended:
        _array(fields, 'ledger_suspended', (ng, nb), 'float64')
        _array(fields, 'ledger_exchange', (ng, 2), 'float64')
    expected_euler=3*steps
    recursive_origin=dict(accepted_steps=0,time_s=0.,euler_calls=0)
    if recursive:
        levels=np.asarray(topology['node_levels'])
        root_level,finest=int(np.min(levels)),int(np.max(levels))
        level_steps=_array(fields,'recursive_level_steps',(finest+1,),'int64')
        recursive_origin=dict(
            accepted_steps=int(_scalar(fields,'recursive_epoch_start_steps','int64')),
            time_s=float(_scalar(fields,'recursive_epoch_start_time_s','float64')),
            euler_calls=int(_scalar(fields,'recursive_epoch_start_euler_calls','int64')))
        expected_levels=np.zeros(finest+1,np.int64)
        for level in range(root_level,finest+1):
            if np.any(levels==level):
                expected_levels[level]=(steps-recursive_origin['accepted_steps'])*(1 << (level-root_level))
        register=_array(fields,'recursive_register',
                        (int(physical['nf']),2,ns),'float64')
        _scalar(fields,'recursive_last_source','float64')
        last_exchange=_array(fields,'recursive_last_exchange',(ng,2),'float64')
        expected_euler=recursive_origin['euler_calls']+3*int(expected_levels.sum())
        _require(0 <= recursive_origin['accepted_steps'] <= steps and
                 0 <= recursive_origin['time_s'] <= current and
                 recursive_origin['euler_calls'] >= 0 and
                 (recursive_origin['accepted_steps']==0) ==
                    (recursive_origin['time_s']==0.) ==
                    (recursive_origin['euler_calls']==0) and
                 np.array_equal(level_steps,expected_levels) and
                 np.max(np.abs(np.sum(last_exchange,axis=1))) <= 1e-8 and
                 (steps > recursive_origin['accepted_steps'] or not np.any(register)),
                 'recursive checkpoint clock/register differs')
    _require(int(_scalar(fields, 'accepted_steps', 'int64')) == steps and
             float(_scalar(fields, 'time_s', 'float64')) == current and
             int(_scalar(fields, 'euler_calls', 'int64')) == expected_euler and
             0 <= float(_scalar(fields, 'maximum_cfl', 'float64')) <= prepared['cfl'] + 1e-12,
             'checkpoint accepted steps/CFL differs')
    events = json.loads(_string_scalar(fields, 'events_json'))
    _require(isinstance(events, list) and len(events) == receipt['event_count'],
             'checkpoint mesh-event inventory differs')
    by_name = index['topologies']
    previous = next(iter(by_name))
    previous_step = 0
    previous_time = 0.
    event_origin=dict(accepted_steps=0,time_s=0.,euler_calls=0)
    for event in events:
        _require(isinstance(event, dict) and event.get('old_topology') == previous and
                 event.get('new_topology') in by_name and
                 event['new_topology'] != previous and
                 type(event.get('step')) is int and
                 previous_step <= event['step'] <= steps and
                 type(event.get('time_s')) in (int, float) and
                 previous_time <= event['time_s'] <= current and
                 event.get('catalog_identity') == prepared['session_identity'] and
                 event.get('old_operator_identity') == by_name[previous]['operator_sha256'] and
                 event.get('new_operator_identity') ==
                     by_name[event['new_topology']]['operator_sha256'],
                 'checkpoint mesh-event chain differs')
        if recursive:
            epoch=event.get('recursive_epoch')
            old_top=by_name[previous]
            old_levels=np.asarray(old_top['node_levels'])
            old_root,old_finest=int(np.min(old_levels)),int(np.max(old_levels))
            expected_levels=np.zeros(old_finest+1,np.int64)
            macros=event['step']-event_origin['accepted_steps']
            for level in range(old_root,old_finest+1):
                if np.any(old_levels==level):
                    expected_levels[level]=macros*(1 << (level-old_root))
            epoch_levels=np.asarray(epoch.get('level_steps')
                                    if isinstance(epoch,dict) else None)
            _require(isinstance(epoch,dict) and set(epoch)=={
                'previous_origin','end_step','end_time_s','euler_calls',
                'level_steps','old_operator_identity','consumed_register_shape',
                'consumed_register_sha256','remapped_as_pending_flux'} and
                epoch['previous_origin']==event_origin and
                epoch['end_step']==event['step'] and
                epoch['end_time_s']==event['time_s'] and
                epoch['euler_calls']==event_origin['euler_calls']+
                    3*int(expected_levels.sum()) and
                epoch_levels.dtype.kind in 'iu' and
                np.array_equal(epoch_levels,expected_levels) and
                epoch['old_operator_identity']==old_top['operator_sha256'] and
                epoch['consumed_register_shape']==[
                    int(old_top['initial_data']['nf']),2,
                    int(old_top['initial_data']['nstate'])] and
                _hex(epoch['consumed_register_sha256']) and
                epoch['remapped_as_pending_flux'] is False,
                'recursive checkpoint mesh epoch differs')
            event_origin=dict(accepted_steps=event['step'],
                time_s=event['time_s'],euler_calls=epoch['euler_calls'])
        previous, previous_step, previous_time = (
            event['new_topology'], event['step'], event['time_s'])
    _require(previous == topology['name'],
             'checkpoint events do not reach selected topology')
    if recursive:
        _require(event_origin==recursive_origin,
                 'recursive checkpoint current epoch origin differs')
    final = run['frames'][-1]
    secondary_index = int(physical['secondary_index'])
    q = state[:, :secondary_index] if secondary_index >= 0 else state
    secondary = (state[:, secondary_index] if secondary_index >= 0
                 else np.zeros(len(state)))
    _require(final['time_s'] == current and final['steps'] == steps and
             final['topology_name'] == topology['name'] and
             np.array_equal(final['q'], q) and
             np.array_equal(final['secondary'], secondary) and
             all(np.array_equal(value, fields[name])
                 for name, value in final['ledger'].items()),
             'checkpoint differs from last published result frame')
    if recursive:
        mapping=dict(recursive_level_steps='recursive_level_steps',
            recursive_epoch_start_steps='recursive_epoch_start_steps',
            recursive_epoch_start_time='recursive_epoch_start_time_s',
            recursive_epoch_start_euler_calls='recursive_epoch_start_euler_calls',
            recursive_register='recursive_register')
        _require(final['recursive'] is not None and all(
            np.array_equal(final['recursive'][frame_name],fields[field_name])
            for frame_name,field_name in mapping.items()),
            'recursive checkpoint differs from last frame diagnostics')
    return dict(checkpoint=checkpoint, run_root=root,
        prepared_root=prepared['root'], time_s=current, steps=steps,
        rejected=rejected, dt_hint=hint, status=manifest['status'],
        method=method, input_identity=prepared['receipt_sha256'],
        preparation_receipt_sha256=prepared['receipt_sha256'],
        session_identity=prepared['session_identity'],
        source_identity=manifest['source_identity'],
        method_identity=manifest['method_identity'],
        output_every_s=interval, dt_max=dtmax, dtmax_s=dtmax,
        start_s=start, end_s=end)
