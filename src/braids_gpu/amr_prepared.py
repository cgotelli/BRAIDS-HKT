"""Versioned, topology-independent prepared inputs for experimental GPU AMR.

This module does not choose refinement or execute a numerical timestep.  It
packages a complete immutable hierarchy and one or more selected-leaf CUDA
operator exports made by the native *initialization-only* path.  No later CPU
solution, reference frame, or CPU numerical fallback is part of this format.

The format deliberately has its own version and artifact identities.  It is
not compatible with the archived 91-to-64-leaf authenticated bridge unless
that case is explicitly exported into this schema and separately qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
from braids_core.windows_paths import filesystem_path

from braids_core.gpu_live_boundary_metadata import TABLE_KEYS, validate_live_boundary_metadata


MAGIC = b'BRDGPU01'
KIND = 'braids-gpu-amr-prepared'
SCHEMA = 1
BANK_SCHEMA = 2
MANIFEST = 'amr_prepared.json'
TREE = 'tree.npz'
_MATERIAL = ('nfrac', 'nstate', 'nbed_layers', 'bedlaw', 'exchange_mode',
             'secondary_flow_model', 'dry_tolerance', 'cfl_target',
             'morphological_factor')
_BANK_SIDECARS = {
    'bank_toe': 'bank_toe_metadata.bin',
    'avalanche': 'avalanche_metadata.bin',
    'bank_history': 'bank_history_metadata.bin',
}
_BANK_IDENTITY = {'nc', 'nf', 'nfrac', 'nbed_layers', 'deck_version'}
_BANK_TOE_KEYS = (_BANK_IDENTITY |
    {'bank_toe_erosion_mode', 'bank_failure', 'toe_mapping_status',
     'bank_toe_capacity_fraction', 'bank_critical_undercut_m',
     'bank_material_mask', 'physical_toe_cell', 'physical_toe_donor',
     'physical_toe_face', 'physical_toe_distance', 'is_toe_cell',
     'is_toe_donor', 'is_toe_face', 'candidate_bank', 'candidate_toe',
     'physical_toe_status'})
_BANK_DISTRIBUTED_KEYS = (_BANK_IDENTITY | {
    'bank_toe_erosion_mode', 'bank_failure', 'bank_distributed_transport_fraction',
    'bank_distributed_critical_slope', 'bank_critical_undercut_m', 'bank_material_mask'})
_MODE_DISPATCH_KIND = 'live-selected-bank-mode23-amr'
_MODE_DISPATCH_CONTRACT = 'mode_dispatch_v1'
_BEDFORM_BANK_KIND = 'live-selected-bank-mode3-bedform-avalanche-amr-v3'
_BEDFORM_BANK_CONTRACT = 'recursive_bedform_mode3_parent_avalanche_v1'
_BEDFORM_PHYSICAL_TOE_KIND = 'live-selected-bank-mode2-bedform-avalanche-amr-v1'
_BEDFORM_PHYSICAL_TOE_CONTRACT = (
    'recursive_bedform_mode2_physical_toe_parent_avalanche_v1')
_EXACT_BEDFORM_BANK_RUNTIMES = {
    _BEDFORM_BANK_KIND: dict(
        bank_mode=3, contract=_BEDFORM_BANK_CONTRACT, toe_geometry_seed_only=False),
    _BEDFORM_PHYSICAL_TOE_KIND: dict(
        bank_mode=2, contract=_BEDFORM_PHYSICAL_TOE_CONTRACT,
        toe_geometry_seed_only=True),
}
_AVALANCHE_CONTROLS = {
    'general_avalanche_wet_slope', 'general_avalanche_dry_slope',
    'general_avalanche_time', 'general_avalanche_wet_depth_switch',
    'general_avalanche_relaxation_length',
}
_AVALANCHE_KEYS = _BANK_IDENTITY | {'general_avalanche'} | _AVALANCHE_CONTROLS
_BANK_HISTORY_FLOATS = (
    'bank_undercut', 'bank_reference_height', 'bank_toe_height',
    'bank_reference_length', 'bank_toe_cumulative',
    'bank_collapse_cumulative',
)
_BANK_HISTORY_KEYS = (_BANK_IDENTITY | {'bank_failure',
    'bank_collapse_count', *_BANK_HISTORY_FLOATS})
_BANK_CLASS_CONTROLS = (
    'd50_fraction','d90_fraction','dstar_fraction','rhos_fraction',
    'sedthr_fraction','ws_fraction','z0_fraction','sediment_type_fraction',
    'sediment_roughness_fraction','reference_height_fraction',
)


def _bank_selected_mode(build, metadata):
    """Authenticate mode dispatch; old generator contracts remain mode2 only."""
    exact = _EXACT_BEDFORM_BANK_RUNTIMES.get(build.get('kind'))
    if exact is not None:
        if (metadata.get('bank_mode') != exact['bank_mode'] or
                metadata.get('bank_metadata_contract') != exact['contract'] or
                metadata.get('recursive_bank_bedform_avalanche') is not True):
            raise ValueError('Missing or invalid selected bedform-bank contract')
        return exact['bank_mode']
    if build.get('kind') != _MODE_DISPATCH_KIND:
        if metadata.get('bank_mode', 2) != 2 or 'bank_metadata_contract' in metadata:
            raise ValueError('Legacy bank initializer cannot declare mode dispatch')
        return 2
    mode = metadata.get('bank_mode')
    if (type(mode) is not int or mode not in (2, 3) or
            metadata.get('bank_metadata_contract') != _MODE_DISPATCH_CONTRACT):
        raise ValueError('Missing or invalid selected bank-mode contract')
    return mode


def _bank_sidecar_inventory(build, bank_mode=2):
    live = build.get('live_boundary_metadata', False)
    dispatched = build.get('kind') == _MODE_DISPATCH_KIND
    exact = _EXACT_BEDFORM_BANK_RUNTIMES.get(build.get('kind'))
    bedform_bank = exact is not None
    if (type(live) is not bool or
            (live and build.get('kind') not in ('live-selected-bank-amr',
                _MODE_DISPATCH_KIND, *_EXACT_BEDFORM_BANK_RUNTIMES)) or
            (dispatched and (not live or build.get('bank_metadata_contract') != _MODE_DISPATCH_CONTRACT or
                             build.get('bank_modes') != [2, 3])) or
            (bedform_bank and (not live or
                build.get('bank_metadata_contract') != exact['contract'] or
                build.get('bank_modes') != [exact['bank_mode']] or
                build.get('recursive_bank_bedform_avalanche') is not True)) or
            type(bank_mode) is not int or
            bank_mode not in ((exact['bank_mode'],) if bedform_bank else
                              (2, 3) if dispatched else (2,))):
        raise ValueError('Invalid live-boundary initializer contract')
    seed_only = build.get('toe_geometry_seed_only', False)
    if (type(seed_only) is not bool or (seed_only and not live) or
            (bedform_bank and seed_only is not exact['toe_geometry_seed_only'])):
        raise ValueError('Invalid geometry-only bank-toe initializer contract')
    base = dict(_BANK_SIDECARS)
    if bank_mode == 3:
        base.pop('bank_toe')
        base['distributed_bank'] = 'distributed_bank_metadata.bin'
    return {**base, **({'normal': 'normal_metadata.bin'} if live else {})}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_artifact(folder: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('Invalid AMR artifact path')
    parts = relative.split('/')
    if any(part in ('', '.', '..') for part in parts) or len(parts) > 2:
        raise ValueError('AMR artifact must be inside the prepared package')
    path = folder.joinpath(*parts)
    if path.resolve() != folder.resolve() and not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError('AMR artifact escapes the prepared package')
    if not path.is_file() or path.is_symlink():
        raise ValueError('Missing or linked AMR artifact: ' + relative)
    return path


def _finite_float(name: str, value, shape=None):
    array = np.asarray(value)
    if array.dtype != np.dtype('float64') or (shape is not None and array.shape != shape) or \
            not np.isfinite(array).all():
        raise ValueError('Invalid finite float64 AMR array: ' + name)
    return array


def _integer(name: str, value, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in 'iu' or (shape is not None and array.shape != shape):
        raise ValueError('Invalid integer AMR array: ' + name)
    return array.astype(np.int64, copy=False)


def _tree_arrays(hierarchy):
    """Extract the topology-independent hierarchy contract at preparation."""
    cells = hierarchy.cells
    arrays = dict(parent=np.asarray(hierarchy.parent, dtype=np.int64),
        node_level=np.asarray(hierarchy.node_level, dtype=np.int64),
        maximum_level=np.asarray(hierarchy.maximum_level, dtype=np.int64),
        offsets=np.asarray(hierarchy.offsets, dtype=np.int64),
        area=np.asarray([cell.area_m2 for cell in cells], dtype=np.float64),
        xy=np.asarray([cell.centroid_xy for cell in cells], dtype=np.float64))
    _validate_tree(arrays, int(hierarchy.refinement_ratio))
    return arrays


def _validate_tree(arrays, ratio: int):
    if ratio != 2:
        raise ValueError('GPU AMR currently requires a 2:1 hierarchy')
    parent = _integer('parent', arrays['parent'])
    n = len(parent)
    if n < 5:
        raise ValueError('AMR hierarchy is too small')
    level = _integer('node_level', arrays['node_level'], (n,))
    maximum = _integer('maximum_level', arrays['maximum_level'], (n,))
    offsets = _integer('offsets', arrays['offsets'])
    area = _finite_float('area', arrays['area'], (n,))
    _finite_float('xy', arrays['xy'], (n, 2))
    if (offsets.ndim != 1 or len(offsets) < 3 or offsets[0] != 0 or
            offsets[-1] != n or np.any(np.diff(offsets) <= 0) or
            not np.array_equal(level, np.searchsorted(offsets[1:],
                                                       np.arange(n), side='right')) or
            np.any(area <= 0) or np.any(parent >= np.arange(n)) or
            np.any(parent < -1) or not np.any(parent == -1) or
            np.any(maximum < level) or np.any(maximum >= len(offsets)-1)):
        raise ValueError('Invalid AMR hierarchy levels, ancestry or areas')
    for node in range(n):
        children = np.flatnonzero(parent == node)
        if children.size not in (0, 4):
            raise ValueError('AMR parent must have four children or none')
        if children.size:
            if np.any(level[children] != level[node]+1) or \
                    not np.isclose(area[node], np.sum(area[children]),
                                   rtol=1e-11, atol=1e-11):
                raise ValueError('AMR child levels/areas do not partition parent')
    if np.any(level[parent >= 0] != level[parent[parent >= 0]]+1):
        raise ValueError('AMR parent level differs by more than one')
    return n


def _leaves(ids, arrays):
    parent = _integer('parent', arrays['parent'])
    level = _integer('node_level', arrays['node_level'])
    area = _finite_float('area', arrays['area'])
    ids = _integer('node_ids', ids)
    if (ids.ndim != 1 or not len(ids) or np.any(ids < 0) or
            np.any(ids >= len(parent)) or np.any(np.diff(ids) <= 0)):
        raise ValueError('Selected AMR leaves must be sorted unique valid nodes')
    active = np.zeros(len(parent), dtype=bool)
    active[ids] = True
    for node in ids:
        ancestor = int(parent[node])
        while ancestor >= 0:
            if active[ancestor]:
                raise ValueError('AMR selected leaf overlaps an ancestor')
            ancestor = int(parent[ancestor])
    for node in np.flatnonzero(level == level.max()):
        ancestor = int(node)
        while ancestor >= 0 and not active[ancestor]:
            ancestor = int(parent[ancestor])
        if ancestor < 0:
            raise ValueError('Selected AMR leaves do not cover every finest cell')
    roots = np.flatnonzero(parent == -1)
    if not np.isclose(np.sum(area[ids]), np.sum(area[roots]),
                      rtol=1e-11, atol=1e-11):
        raise ValueError('Selected AMR leaf areas do not cover the domain')
    return ids


def _read_coupled(path: Path):
    """Read the strict native initialization-only operator record in-process."""
    result = {}
    with path.open('rb') as stream:
        if stream.read(8) != MAGIC:
            raise ValueError('Wrong AMR coupled operator magic')
        version = np.fromfile(stream, dtype='<i4', count=1)
        if version.size != 1 or int(version[0]) != 1:
            raise ValueError('Unsupported AMR coupled operator schema')
        while True:
            raw = stream.read(64)
            if len(raw) != 64:
                raise ValueError('Truncated AMR coupled operator name')
            name = raw.decode('ascii').strip()
            header = np.fromfile(stream, dtype='<i4', count=6)
            if header.size != 6:
                raise ValueError('Truncated AMR coupled operator header')
            kind, rank, *dimensions = map(int, header)
            if name == 'END':
                if kind != 0 or rank != 0 or stream.read(1):
                    raise ValueError('Invalid AMR coupled operator terminator')
                break
            if (not name or name in result or kind not in (1, 2) or
                    not 0 <= rank <= 4 or any(d < 0 for d in dimensions[:rank])):
                raise ValueError('Invalid AMR coupled operator record')
            shape = tuple(dimensions[:rank])
            count = int(np.prod(shape)) if rank else 1
            if count > 100_000_000:
                raise ValueError('Implausibly large AMR coupled operator')
            values = np.fromfile(stream, dtype='<i4' if kind == 1 else '<f8', count=count)
            if values.size != count or not np.isfinite(values).all():
                raise ValueError('Invalid/truncated AMR coupled operator ' + name)
            result[name] = values[0].item() if not rank else np.ascontiguousarray(
                values.reshape(shape, order='F'))
    return result


def _bank_int(name, value):
    if type(value) is not int or not -(1 << 31) <= value < (1 << 31):
        raise ValueError('Invalid native integer bank scalar: ' + name)
    return value


def _bank_real(name, value):
    if type(value) is not float or not np.isfinite(value):
        raise ValueError('Invalid native real bank scalar: ' + name)
    return value


def _bank_array(name, value, shape, dtype):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype != np.dtype(dtype) or
            (array.dtype.kind == 'f' and not np.isfinite(array).all())):
        raise ValueError('Invalid native bank array: ' + name)
    return array


def _validate_bank_sidecars(data, records, ids, tree, *, live_boundaries=False,
                            toe_geometry_seed_only=False, bank_mode=2,
                            physical_toe_bedform=False):
    """Fail-closed schema-2 mode-specific bank, avalanche and history admission."""
    if (type(physical_toe_bedform) is not bool or type(bank_mode) is not int or
            bank_mode not in (2, 3) or
            (bank_mode == 3 and (not live_boundaries or toe_geometry_seed_only)) or
            (physical_toe_bedform and
             (bank_mode != 2 or not live_boundaries or not toe_geometry_seed_only))):
        raise ValueError('Invalid bank-mode sidecar admission')
    bank_key = 'bank_toe' if bank_mode == 2 else 'distributed_bank'
    if set(records) != {bank_key, 'avalanche', 'bank_history'} | ({'normal'} if live_boundaries else set()):
        raise ValueError('Incomplete bank-enabled AMR sidecar set')
    toe, avalanche, history = (records[name] for name in (bank_key, 'avalanche', 'bank_history'))
    for name, values, expected in (
            (bank_key, toe, (_BANK_TOE_KEYS | ({'toe_geometry_seed_only'} if toe_geometry_seed_only else set())) if bank_mode == 2 else _BANK_DISTRIBUTED_KEYS),
            ('avalanche', avalanche, _AVALANCHE_KEYS),
            ('bank-history', history, _BANK_HISTORY_KEYS)):
        if set(values) != expected:
            raise ValueError('Unexpected '+name+' AMR sidecar records')
        for key in _BANK_IDENTITY:
            value = _bank_int(key, values[key])
            if key not in data or _bank_int(key, data[key]) != value:
                raise ValueError('Native bank sidecar/core identity differs: '+key)
        for key, value in values.items():
            if key in data and not np.array_equal(data[key], value):
                raise ValueError('Conflicting native bank sidecar/core record: '+key)
    nc, nf = int(data['nc']), int(data['nf'])
    if toe_geometry_seed_only and _bank_int('toe_geometry_seed_only', toe['toe_geometry_seed_only']) != 1:
        raise ValueError('Geometry-only bank-toe sidecar marker differs')
    if nc != len(ids) or nf < 1:
        raise ValueError('Bank AMR sidecar selected dimensions differ')
    stage_distributed = bank_mode == 3 and np.array_equal(data.get('mode'), [2])
    if stage_distributed and (int(data['secondary_index']) != -1 or
            any(int(data[key]) != 0 for key in
                ('bed_slope', 'hiding_exposure', 'bedload_adaptation'))):
        raise ValueError('Stage-only distributed bank requires secondary-OFF and unmodified sediment controls')
    if (not all(_bank_int(key, toe[key]) == required for key, required in
                (('bank_toe_erosion_mode', bank_mode), ('bank_failure', 1))) or
            (bank_mode == 2 and _bank_int('toe_mapping_status', toe['toe_mapping_status']) != 0) or
            _bank_int('bank_failure', history['bank_failure']) != 1 or
            _bank_int('general_avalanche', avalanche['general_avalanche']) not in ((0, 1) if bank_mode == 3 else (1,)) or
            int(data['scheme']) != 6 or int(data['bedlaw']) != 2 or
            int(data['exchange_mode']) != 0 or
            int(data['secondary_flow_model']) != (
                0 if stage_distributed or physical_toe_bedform else 2) or
            int(data['bank_toe_erosion_mode']) != bank_mode or
            int(data['bank_failure']) != 1 or
            int(data['general_avalanche']) not in ((0, 1) if bank_mode == 3 else (1,)) or
            int(data['amr_enabled']) != 0):
        raise ValueError('Bank AMR native physical controls differ')
    capacity_key = 'bank_toe_capacity_fraction' if bank_mode == 2 else 'bank_distributed_transport_fraction'
    capacity = _bank_real(capacity_key, toe[capacity_key])
    if bank_mode == 3 and _bank_real('bank_distributed_critical_slope', toe['bank_distributed_critical_slope']) <= 0:
        raise ValueError('Invalid distributed-bank critical slope')
    undercut = _bank_real('bank_critical_undercut_m',
                          toe['bank_critical_undercut_m'])
    if not 0 <= capacity <= 1 or undercut <= 0:
        raise ValueError('Invalid AMR physical-toe capacity/undercut')
    controls = {key:_bank_real(key, avalanche[key])
                for key in _AVALANCHE_CONTROLS}
    if (controls['general_avalanche_wet_slope'] <= 0 or
            controls['general_avalanche_dry_slope'] <= 0 or
            controls['general_avalanche_time'] <= 0 or
            controls['general_avalanche_wet_depth_switch'] < 0 or
            controls['general_avalanche_relaxation_length'] <= 0):
        raise ValueError('Invalid AMR general-avalanche controls')
    state = _finite_float('state', data['state'])
    suspended = _integer('suspended_indices', data['suspended_indices'])
    if (suspended.shape != (int(data['nfrac']),) or
            np.any(suspended < 3) or np.any(suspended >= state.shape[0]) or
            np.any(state[suspended] != 0) or
            np.any(np.asarray(data['fraction_concentration']) != 0)):
        raise ValueError('Bank AMR requires zero suspended inventories/forcing')
    material = _bank_array('bank_material_mask', toe['bank_material_mask'], (nc,), '<i4')
    if not np.isin(material, (0, 1)).all() or not np.any(material):
        raise ValueError('Invalid or empty bank material mask')
    owner = _integer('owner', data['owner'], (nf,))
    neighbour = _integer('neighbour', data['neighbour'], (nf,))
    if bank_mode == 3 and any(key in data for key in
            (_BANK_TOE_KEYS - _BANK_DISTRIBUTED_KEYS - {'bank_toe_capacity_fraction'}) |
            {'toe_geometry_seed_only'}):
        raise ValueError('Distributed-bank operator contains physical-toe metadata')
    if bank_mode == 2:
        flags = {key:_bank_array(key, toe[key], (nc if key != 'is_toe_face' else nf,), '<i4')
                 for key in ('bank_material_mask','is_toe_cell','is_toe_donor','is_toe_face')}
        if any(not np.isin(value, (0, 1)).all() for value in flags.values()):
            raise ValueError('Nonbinary physical-toe AMR flags')
        material = flags['bank_material_mask'].astype(bool)
        active = flags['is_toe_face'].astype(bool)
        if not material.any() or not active.any():
            raise ValueError('Empty physical-toe AMR material/face map')
        cell_ids = {key:_bank_array(key, toe[key], (nc,), '<i4')
                    for key in ('physical_toe_cell','physical_toe_donor',
                                'physical_toe_face','physical_toe_status')}
        face_ids = {key:_bank_array(key, toe[key], (nf,), '<i4')
                    for key in ('candidate_bank','candidate_toe')}
        distance = _bank_array('physical_toe_distance',
                               toe['physical_toe_distance'], (nc,), '<f8')
        if (np.any(distance < 0) or
                not np.isin(cell_ids['physical_toe_status'][material],
                            (1, 2) if toe_geometry_seed_only else (1,)).all() or
                np.any(cell_ids['physical_toe_status'][~material] != 0) or
                any(np.any(cell_ids[key][~material] != -1)
                    for key in ('physical_toe_cell','physical_toe_donor',
                                'physical_toe_face')) or
                np.any(face_ids['candidate_bank'][~active] != -1) or
                np.any(face_ids['candidate_toe'][~active] != -1)):
            raise ValueError('Incomplete or ambiguous AMR bank material mapping')
        target = cell_ids['physical_toe_cell'][material]
        donor = cell_ids['physical_toe_donor'][material]
        mapped_face = cell_ids['physical_toe_face'][material]
        if (np.any(target < 0) or np.any(target >= nc) or
                np.any(donor < 0) or np.any(donor >= nc) or
                np.any(mapped_face < 0) or np.any(mapped_face >= nf) or
                np.any(material[target]) or np.any(~material[donor]) or
                np.any(~active[mapped_face]) or
                np.any(face_ids['candidate_bank'][mapped_face] != donor) or
                np.any(face_ids['candidate_toe'][mapped_face] != target)):
            raise ValueError('Invalid AMR mapped bank/toe cell or face ID')
        bank_face = face_ids['candidate_bank'][active]
        toe_face = face_ids['candidate_toe'][active]
        owner = _integer('owner', data['owner'], (nf,))
        neighbour = _integer('neighbour', data['neighbour'], (nf,))
        if (np.any(bank_face < 0) or np.any(bank_face >= nc) or
                np.any(toe_face < 0) or np.any(toe_face >= nc) or
                np.any(~material[bank_face]) or np.any(material[toe_face]) or
                np.any(~(((owner[active] == bank_face) &
                          (neighbour[active] == toe_face)) |
                         ((owner[active] == toe_face) &
                          (neighbour[active] == bank_face)))) or
                not np.array_equal(flags['is_toe_donor'].astype(bool),
                                   np.isin(np.arange(nc), bank_face)) or
                not np.array_equal(flags['is_toe_cell'].astype(bool),
                                   np.isin(np.arange(nc), toe_face))):
            raise ValueError('Physical-toe AMR candidate face graph differs')
    for key in _BANK_HISTORY_FLOATS:
        value = _bank_array(key, history[key], (nc,), '<f8')
        if np.any(value < 0):
            raise ValueError('Negative AMR bank history: '+key)
    count = _bank_array('bank_collapse_count',
                        history['bank_collapse_count'], (nc,), '<i4')
    if np.any(count < 0):
        raise ValueError('Negative AMR bank collapse count')
    boundary = _integer('boundary_faces', data['boundary_faces'])
    mode = _integer('mode', data['mode'])
    if ((not live_boundaries and (mode.shape != (1,) or int(mode[0]) != 2)) or boundary.ndim != 1 or
            not len(boundary) or np.any(boundary < 0) or
            np.any(boundary >= nf) or
            np.any(neighbour[boundary] != -1) or
            np.any(tree['node_level'][ids[owner[boundary]]] !=
                   np.max(tree['node_level']))):
        raise ValueError('Bank AMR open boundaries must remain on finest leaves')
    merged = dict(data)
    for values in records.values():
        merged.update(values)
    if live_boundaries:
        # Checks duplicate core fields before accepting the sidecar, including
        # all padded tables and the rebuilt operator's native seed indices.
        merged = validate_live_boundary_metadata(data, records['normal']) | {
            key: value for values in (toe, avalanche, history) for key, value in values.items()}
    return merged


def _validate_operator(data, ids, tree):
    n = len(ids)
    if (int(data.get('nc', -1)) != n or int(data.get('scheme', -1)) != 6 or
            int(data.get('amr_enabled', -1)) != 0):
        raise ValueError('AMR selected-leaf operator must be fixed scheme-6 initialization')
    state = _finite_float('state', data['state'])
    if state.shape != (int(data['nstate']), n):
        raise ValueError('AMR state shape differs from selected leaves')
    area = _finite_float('operator area', data['area'], (n,))
    xy = _finite_float('operator xy', data['xy'], (2, n))
    if (not np.allclose(area, tree['area'][ids], atol=1e-12, rtol=1e-12) or
            not np.allclose(xy.T, tree['xy'][ids], atol=1e-12, rtol=1e-12)):
        raise ValueError('AMR native operator geometry differs from hierarchy leaves')
    nf = int(data.get('nf', -1))
    owner = _integer('owner', data['owner'], (nf,))
    neighbour = _integer('neighbour', data['neighbour'], (nf,))
    length = _finite_float('length', data['length'], (nf,))
    if (nf < 1 or np.any(owner < 0) or np.any(owner >= n) or
            np.any(neighbour < -1) or np.any(neighbour >= n) or
            np.any(neighbour == owner) or np.any(length <= 0)):
        raise ValueError('Invalid AMR selected-leaf face graph')
    internal = neighbour >= 0
    levels = tree['node_level'][ids]
    if np.any(np.abs(levels[owner[internal]]-levels[neighbour[internal]]) > 1):
        raise ValueError('AMR selected-leaf face exceeds 2:1 balance')
    if np.any(state[0] < -1e-12):
        raise ValueError('Negative AMR water inventory')
    sediment_rows = np.concatenate((np.asarray(data['suspended_indices']).ravel(),
                                    np.asarray(data['bed_indices']).ravel()))
    if (np.any(sediment_rows < 3) or np.any(sediment_rows >= state.shape[0]) or
            np.any(state[sediment_rows] < -1e-12)):
        raise ValueError('Negative or invalid AMR sediment inventory')
    if any(key not in data for key in _MATERIAL):
        raise ValueError('Incomplete AMR material controls')
    return data


def _solver_aliases(data):
    """Match the fixed-grid CUDA constructor's native-record aliases."""
    data['q'] = np.ascontiguousarray(data['state'][:3])
    data['dry'] = data['dry_tolerance']
    data['outlet_extension'] = data['cpu_outlet_extension']
    data['dt'] = data['dtmax']
    data['bedoffset'] = data['boundary_bed_offset']
    data['bc_hydraulic'] = np.ascontiguousarray(data['forcing'][:2])
    data['bc_bed_input'] = data['fraction_bed_input'].copy()
    return data


def _initialization_receipt(path: Path, operator: Path, *, check_origin):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    digest = _sha(operator)
    runtime = receipt.get('export_runtime_sha256')
    if (receipt.get('schema') != 1 or
            receipt.get('initialization_CPU_only') is not True or
            receipt.get('solver_steps_performed') != 0 or
            not isinstance(runtime, str) or len(runtime) != 64 or
            receipt.get('artifacts', {}).get('coupled_input.bin') != digest):
        raise ValueError('AMR operator lacks a zero-step native initialization receipt')
    if check_origin:
        files = receipt.get('origin', {}).get('files', {})
        if not isinstance(files, dict) or not files or \
                any('/' in name or '\\' in name or not (path.parent/name).is_file() or
                    _sha(path.parent/name) != declared for name, declared in files.items()):
            raise ValueError('AMR native initialization origin changed')
    return receipt


def _hex_digest(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(character in '0123456789abcdef' for character in value))


def _bank_runtime_build(path: Path):
    build = json.loads(path.read_text(encoding='utf-8'))
    if (build.get('schema') != 1 or build.get('export_only') is not True or
            build.get('numerical_routines_unchanged') is not True or
            not all(_hex_digest(build.get(key)) for key in
                    ('executable_sha256','parent_executable_sha256',
                     'compiler_sha256','generator_sha256')) or
            not isinstance(build.get('source_sha256'), dict) or
            not build['source_sha256'] or
            not all(_hex_digest(value) for value in
                    build['source_sha256'].values())):
        raise ValueError('Invalid unchanged-native bank AMR runtime build')
    exact = _EXACT_BEDFORM_BANK_RUNTIMES.get(build.get('kind'))
    _bank_sidecar_inventory(build, exact['bank_mode'] if exact is not None else 2)
    return build


def _bank_initialization_receipt(path: Path, build_path: Path, *, check_origin):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    build = _bank_runtime_build(build_path)
    bank_mode = _bank_selected_mode(build, receipt)
    names = {'coupled_input.bin', *_bank_sidecar_inventory(build, bank_mode).values()}
    runtime = receipt.get('runtime_executable_sha256')
    alias = receipt.get('export_runtime_sha256')
    artifacts = receipt.get('artifacts')
    if (receipt.get('schema') != 1 or
            receipt.get('status') != 'completed' or
            receipt.get('initialization_CPU_only') is not True or
            receipt.get('solver_steps_performed') != 0 or
            receipt.get('numerical_routines_unchanged') is not True or
            receipt.get('GPU_solver_executed') is not False or
            receipt.get('Delft3D_source_modified') is not False or
            'validation_receipt_adapter' in receipt or
            not _hex_digest(runtime) or
            runtime != build['executable_sha256'] or
            (alias is not None and alias != runtime) or
            receipt.get('runtime_build_sha256') != _sha(build_path) or
            receipt.get('native_cpu_parent_sha256') !=
                build['parent_executable_sha256'] or
            receipt.get('exporter_source_sha256') !=
                build['generator_sha256'] or
            receipt.get('live_boundary_metadata', False) is not
                build.get('live_boundary_metadata', False) or
            receipt.get('toe_geometry_seed_only', False) is not
                (build.get('toe_geometry_seed_only', False) and bank_mode == 2) or
            not isinstance(artifacts, dict) or set(artifacts) != names or
            any(not _hex_digest(artifacts[name]) or
                artifacts[name] != _sha(_direct_artifact(path.parent, name))
                for name in names)):
        raise ValueError('Incomplete bank AMR native zero-step receipt')
    if check_origin:
        files = receipt.get('origin', {}).get('files', {})
        if (not isinstance(files, dict) or not files or
                any('/' in name or '\\' in name or
                    _sha(_direct_artifact(path.parent, name)) != digest
                    for name, digest in files.items())):
            raise ValueError('Bank AMR native initialization origin changed')
    return receipt


def _bank_receipt_operator_match(receipt, data):
    controls = receipt.get('controls')
    keys = ('scheme','bedlaw','exchange_mode','bank_toe_erosion_mode',
            'general_avalanche','secondary_flow_model','cpu_time_integrator')
    exact = receipt.get('recursive_bank_bedform_avalanche', False)
    if exact is True:
        keys += ('bedform_roughness',)
    if (not isinstance(controls, dict) or set(controls) != set(keys) or
            any(type(controls[key]) is not int or
                controls[key] != _bank_int(key, data[key]) for key in keys) or
            type(exact) is not bool or
            (exact and (controls['bedform_roughness'] != 1 or
                        controls['bank_toe_erosion_mode'] not in (2, 3) or
                        controls['general_avalanche'] != 1)) or
            (not exact and 'recursive_bank_bedform_avalanche' in receipt) or
            receipt.get('cells') != _bank_int('nc', data['nc']) or
            receipt.get('faces') != _bank_int('nf', data['nf']) or
            receipt.get('bank_cells') != int(np.count_nonzero(
                data['bank_material_mask'])) or
            (int(data['bank_toe_erosion_mode']) == 2 and receipt.get('toe_faces') !=
                int(np.count_nonzero(data['is_toe_face']))) or
            (int(data['bank_toe_erosion_mode']) == 3 and 'toe_faces' in receipt) or
            ('bank_mode' in receipt and receipt['bank_mode'] != int(data['bank_toe_erosion_mode']))):
        raise ValueError('Bank AMR original receipt/operator controls differ')


def _bank_same_global_controls(first, second):
    """Require identical global physics, while leaving local face maps local."""
    if set(first) != set(second):
        return False
    for key in first:
        if key not in ('nc','nf') and np.asarray(first[key]).shape == () and \
                not np.array_equal(first[key], second[key]):
            return False
    keys = _BANK_CLASS_CONTROLS
    if 'native_section_seed_node' in first:
        keys += ('mode', 'section_control', 'section_energy_slope', 'forcing',
                 'fraction_concentration', 'fraction_bed_input', 'nseries',
                 'nhydro_series', 'nfraction_series', *TABLE_KEYS)
    return all(np.array_equal(first[key], second[key]) for key in keys)


@dataclass(frozen=True)
class AMRTopology:
    name: str
    node_ids: np.ndarray
    data: dict
    source_identity: str


@dataclass(frozen=True)
class PreparedAMR:
    folder: Path
    identity: str
    tree: dict
    topologies: tuple[AMRTopology, ...]
    source: dict
    schema: int = SCHEMA


def load_amr_prepared(folder) -> PreparedAMR:
    """Authenticate every artifact and topology before returning GPU inputs."""
    folder = filesystem_path(folder).resolve()
    manifest_path = _direct_artifact(folder, MANIFEST)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    schema = manifest.get('schema')
    if (manifest.get('kind') != KIND or schema not in (SCHEMA, BANK_SCHEMA) or
            manifest.get('ratio') != 2 or not isinstance(manifest.get('artifacts'), dict) or
            not isinstance(manifest.get('topologies'), list) or
            not manifest['topologies']):
        raise ValueError('Unsupported or incomplete AMR prepared manifest')
    build_path = None
    if schema == BANK_SCHEMA:
        if (manifest.get('runtime_build') != 'runtime_build.json' or
                set(manifest) != {'kind','schema','ratio','source','artifacts',
                                  'topologies','runtime_build',
                                  'solver_steps_performed_by_exporter',
                                  'CPU_numerical_fallback'} or
                manifest.get('solver_steps_performed_by_exporter') != 0 or
                manifest.get('CPU_numerical_fallback') is not False):
            raise ValueError('Incomplete bank AMR schema-2 manifest')
        build_path = _direct_artifact(folder, 'runtime_build.json')
        build = _bank_runtime_build(build_path)
        bank_mode = _bank_selected_mode(build, manifest.get('source', {}))
        bank_sidecars = _bank_sidecar_inventory(build, bank_mode)
        if manifest.get('source', {}).get('live_boundary_metadata', False) is not \
                build.get('live_boundary_metadata', False):
            raise ValueError('Bank AMR live-boundary source/runtime contract differs')
        if manifest.get('source', {}).get('toe_geometry_seed_only', False) is not \
                (build.get('toe_geometry_seed_only', False) and bank_mode == 2):
            raise ValueError('Bank AMR geometry-only toe source/runtime contract differs')
    artifacts = manifest['artifacts']
    if TREE not in artifacts or any(not isinstance(name, str) or
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest)
            for name, digest in artifacts.items()):
        raise ValueError('Invalid AMR artifact inventory')
    for name, digest in artifacts.items():
        if _sha(_direct_artifact(folder, name)) != digest:
            raise ValueError('Changed AMR prepared artifact: ' + name)
    with np.load(folder/TREE, allow_pickle=False) as archive:
        if set(archive.files) != {'parent', 'node_level', 'maximum_level',
                                 'offsets', 'area', 'xy'}:
            raise ValueError('Unexpected AMR tree array schema')
        tree = {key: archive[key].copy() for key in archive.files}
    _validate_tree(tree, 2)
    topologies = []
    expected = {TREE} | ({'runtime_build.json'} if schema == BANK_SCHEMA else set())
    names = set()
    for index, row in enumerate(manifest['topologies']):
        row_keys = {'name', 'node_ids', 'operator', 'geometry', 'receipt'}
        if schema == BANK_SCHEMA:
            row_keys.add('sidecars')
        if (not isinstance(row, dict) or set(row) != row_keys or
                not isinstance(row['name'], str) or not row['name'] or
                row['name'] in names):
            raise ValueError('Invalid or duplicate AMR topology name')
        names.add(row['name'])
        base = f'topology_{index:04d}/'
        if (row['node_ids'] != base+'node_ids.npy' or
                row['operator'] != base+'coupled_input.bin' or
                row['geometry'] != base+'geometry.npz' or
                row['receipt'] != base+'preparation_receipt.json'):
            raise ValueError('Unexpected AMR topology artifact layout')
        expected.update((row['node_ids'], row['operator'], row['geometry'],
                         row['receipt']))
        ids = _leaves(np.load(folder/row['node_ids'], allow_pickle=False), tree)
        if schema == BANK_SCHEMA:
            sidecars = row['sidecars']
            required = {key:base+filename for key, filename in
                        bank_sidecars.items()}
            if not isinstance(sidecars, dict) or sidecars != required:
                raise ValueError('Incomplete bank AMR schema-2 sidecar row')
            sidecar_folder = folder/f'topology_{index:04d}'
            if {path.name for path in sidecar_folder.glob('*metadata.bin')} != \
                    set(bank_sidecars.values()):
                raise ValueError('Unlisted bank AMR physics sidecar')
            expected.update(sidecars.values())
            receipt = _bank_initialization_receipt(folder/row['receipt'], build_path,
                                                   check_origin=False)
        else:
            _initialization_receipt(folder/row['receipt'], folder/row['operator'],
                                    check_origin=False)
        data = _validate_operator(_read_coupled(folder/row['operator']), ids, tree)
        if schema == BANK_SCHEMA:
            records = {key:_read_coupled(folder/relative)
                       for key, relative in sidecars.items()}
            data = _validate_bank_sidecars(data, records, ids, tree,
                live_boundaries=build.get('live_boundary_metadata', False),
                toe_geometry_seed_only=build.get('toe_geometry_seed_only', False) and bank_mode == 2,
                bank_mode=bank_mode,
                physical_toe_bedform=
                    build.get('kind') == _BEDFORM_PHYSICAL_TOE_KIND)
            _bank_receipt_operator_match(receipt, data)
        data = _solver_aliases(data)
        # The viewer geometry is identity-checked here.  Its full polygon
        # decoding remains the responsibility of the QGIS result materializer.
        with np.load(folder/row['geometry'], allow_pickle=False) as geometry:
            cell = geometry['cell_integer']
            if (cell.shape != (len(ids), 6) or
                    not np.array_equal(cell[:, 0], np.arange(len(ids))) or
                    not np.array_equal(cell[:, 4], tree['node_level'][ids]) or
                    not np.array_equal(cell[:, 5], ids)):
                raise ValueError('AMR polygon geometry differs from selected leaves')
        topologies.append(AMRTopology(row['name'], ids.copy(), data,
                                      artifacts[row['operator']]))
    if expected != set(artifacts):
        raise ValueError('AMR package has unknown or missing numerical artifacts')
    first = topologies[0].data
    bank_material = ('bank_failure','bank_toe_erosion_mode',
        'bank_critical_undercut_m',
        'general_avalanche', *_AVALANCHE_CONTROLS,
        'mpm_friction_mode','mpm_critical','cpu_time_integrator',
        'bulk_fraction','secondary_flow')
    if schema == BANK_SCHEMA:
        bank_material += (('bank_toe_capacity_fraction',) if int(first['bank_toe_erosion_mode']) == 2 else
                          ('bank_distributed_transport_fraction', 'bank_distributed_critical_slope'))
    for topology in topologies[1:]:
        keys = _MATERIAL + (bank_material if schema == BANK_SCHEMA else ())
        if any(key not in first or key not in topology.data or
               not np.array_equal(first[key], topology.data[key])
               for key in keys) or (schema == BANK_SCHEMA and
                    not _bank_same_global_controls(first, topology.data)):
            raise ValueError('AMR topologies use different physical controls')
    identity = _sha(manifest_path)
    return PreparedAMR(folder, identity, tree, tuple(topologies),
                       manifest.get('source', {}), schema)


def export_amr_prepared(destination, hierarchy, topologies: Sequence,
                        *, source: dict | None = None) -> PreparedAMR:
    """Package arbitrary prepared selected-leaf operators, never CPU outputs.

    ``topologies`` is a sequence of ``(name, node_ids, prepared_folder)``.
    Each folder must contain an initialization-only ``coupled_input.bin`` and
    the corresponding selected-mesh ``geometry.npz``.  The caller owns the
    native initialization and records its runtime identity in ``source``.
    """
    destination = filesystem_path(destination).resolve()
    if destination.exists():
        raise FileExistsError('AMR prepared destination must be new')
    tree = _tree_arrays(hierarchy)
    if not isinstance(topologies, Sequence) or not topologies:
        raise ValueError('At least one AMR topology is required')
    if source is not None and (not isinstance(source, dict) or
            not all(isinstance(key, str) and isinstance(value, (str, int, float, bool))
                    for key, value in source.items())):
        raise ValueError('AMR source receipt must be flat JSON metadata')
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/TREE, **tree)
    artifacts = {TREE: _sha(destination/TREE)}
    rows = []
    for index, item in enumerate(topologies):
        if len(item) != 3:
            raise ValueError('AMR topology requires name, node IDs and prepared folder')
        name, ids, prepared = item
        ids = _leaves(ids, tree)
        prepared = filesystem_path(prepared).resolve()
        data = _validate_operator(_read_coupled(prepared/'coupled_input.bin'), ids, tree)
        if data['nc'] != len(ids):
            raise ValueError('AMR native operator does not match requested leaves')
        location = destination/f'topology_{index:04d}'
        location.mkdir()
        np.save(location/'node_ids.npy', ids)
        _initialization_receipt(prepared/'preparation_receipt.json',
                                prepared/'coupled_input.bin', check_origin=True)
        for filename in ('coupled_input.bin', 'geometry.npz',
                         'preparation_receipt.json'):
            if not (prepared/filename).is_file():
                raise ValueError('Missing AMR native prepared artifact: '+filename)
            shutil.copy2(prepared/filename, location/filename)
        prefix = location.name+'/'
        row = dict(name=name, node_ids=prefix+'node_ids.npy',
                   operator=prefix+'coupled_input.bin',
                   geometry=prefix+'geometry.npz',
                   receipt=prefix+'preparation_receipt.json')
        rows.append(row)
        artifacts.update({relative: _sha(destination/relative)
                          for relative in (row['node_ids'], row['operator'],
                                           row['geometry'], row['receipt'])})
    manifest = dict(kind=KIND, schema=SCHEMA, ratio=2, source=source or {},
                    artifacts=artifacts, topologies=rows,
                    solver_steps_performed_by_exporter=0,
                    CPU_numerical_fallback=False)
    (destination/MANIFEST).write_text(json.dumps(manifest, indent=2,
        sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return load_amr_prepared(destination)


def export_bank_amr_prepared_v2(destination, hierarchy, topologies: Sequence,
                                *, runtime_build, source: dict | None = None) -> PreparedAMR:
    """Package unchanged-native, bank-enabled zero-step operators as schema 2.

    This is host-side input authentication only. It does not advance or
    qualify the GPU bank/AMR solver. Schema-1 export remains unchanged.
    """
    destination = filesystem_path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Bank AMR prepared destination must be new')
    tree = _tree_arrays(hierarchy)
    if not isinstance(topologies, Sequence) or not topologies:
        raise ValueError('At least one bank AMR topology is required')
    if source is not None and (not isinstance(source, dict) or
            not all(isinstance(key, str) and
                    isinstance(value, (str, int, float, bool))
                    for key, value in source.items())):
        raise ValueError('Bank AMR source receipt must be flat JSON metadata')
    build_path = Path(runtime_build).resolve()
    if not build_path.is_file() or build_path.is_symlink():
        raise ValueError('Bank AMR native runtime build is missing or linked')
    build = _bank_runtime_build(build_path)
    bank_mode = _bank_selected_mode(build, source or {})
    bank_sidecars = _bank_sidecar_inventory(build, bank_mode)
    if (source or {}).get('live_boundary_metadata', False) is not build.get('live_boundary_metadata', False):
        raise ValueError('Bank AMR live-boundary source/runtime contract differs')
    if (source or {}).get('toe_geometry_seed_only', False) is not (build.get('toe_geometry_seed_only', False) and bank_mode == 2):
        raise ValueError('Bank AMR geometry-only toe source/runtime contract differs')
    checked = []
    names = set()
    for item in topologies:
        if len(item) != 3:
            raise ValueError('Bank AMR topology requires name, IDs and prepared folder')
        name, ids, prepared = item
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Invalid or repeated bank AMR topology name')
        names.add(name)
        ids = _leaves(ids, tree)
        prepared = filesystem_path(prepared).resolve()
        files = {'coupled_input.bin','geometry.npz','preparation_receipt.json',
                 *bank_sidecars.values()}
        if (not prepared.is_dir() or
                any(not (prepared/filename).is_file() or
                    (prepared/filename).is_symlink() for filename in files) or
                {path.name for path in prepared.glob('*metadata.bin')} !=
                    set(bank_sidecars.values())):
            raise ValueError('Incomplete or incompatible bank AMR native inputs')
        data = _validate_operator(_read_coupled(prepared/'coupled_input.bin'),
                                  ids, tree)
        records = {key:_read_coupled(prepared/filename)
                   for key, filename in bank_sidecars.items()}
        data = _validate_bank_sidecars(data, records, ids, tree,
            live_boundaries=build.get('live_boundary_metadata', False),
            toe_geometry_seed_only=build.get('toe_geometry_seed_only', False) and bank_mode == 2,
            bank_mode=bank_mode,
            physical_toe_bedform=
                build.get('kind') == _BEDFORM_PHYSICAL_TOE_KIND)
        receipt = _bank_initialization_receipt(
            prepared/'preparation_receipt.json', build_path, check_origin=True)
        _bank_receipt_operator_match(receipt, data)
        checked.append((name, ids, prepared))
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination/TREE, **tree)
    shutil.copy2(build_path, destination/'runtime_build.json')
    artifacts = {relative:_sha(destination/relative)
                 for relative in (TREE,'runtime_build.json')}
    rows = []
    for index, (name, ids, prepared) in enumerate(checked):
        location = destination/f'topology_{index:04d}'
        location.mkdir()
        np.save(location/'node_ids.npy', ids)
        for filename in ('coupled_input.bin','geometry.npz',
                         'preparation_receipt.json', *bank_sidecars.values()):
            shutil.copy2(prepared/filename, location/filename)
        prefix = location.name+'/'
        row = dict(name=name, node_ids=prefix+'node_ids.npy',
                   operator=prefix+'coupled_input.bin',
                   geometry=prefix+'geometry.npz',
                   receipt=prefix+'preparation_receipt.json',
                   sidecars={key:prefix+filename for key,filename in
                             bank_sidecars.items()})
        rows.append(row)
        for relative in (row['node_ids'], row['operator'], row['geometry'],
                         row['receipt'], *row['sidecars'].values()):
            artifacts[relative] = _sha(destination/relative)
    manifest = dict(kind=KIND, schema=BANK_SCHEMA, ratio=2,
                    runtime_build='runtime_build.json',
                    source=source or {}, artifacts=artifacts,
                    topologies=rows, solver_steps_performed_by_exporter=0,
                    CPU_numerical_fallback=False)
    (destination/MANIFEST).write_text(json.dumps(manifest, indent=2,
        sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return load_amr_prepared(destination)
