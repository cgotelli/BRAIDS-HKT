"""Read-only admission of a schema-6 on-demand GPU AMR preparation.

This module authenticates host artifacts and the current bounded composition.
It imports no worker, validation, QGIS or CUDA code and executes no initializer.
The copied runtime build is provenance; executable/package admission remains
the external worker's responsibility. Original input files must remain present
because future selected meshes are initialized from that exact input family.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from braids_core.windows_paths import filesystem_path

from braids_core.gpu_live_boundary_metadata import (
    TABLE_KEYS, boundary_event_identity, validate_live_boundary_metadata)

from .gpu_amr_prepared_io import _artifact, _selected_nodes, _tree
from .gpu_frame_io import _geometry
from .gpu_prepared_io import _authenticated_member, _read_records, read_gpu_prepared


VARIANT = "on_demand_bank_amr_ssp33"
_MODE_DISPATCH_KIND = 'live-selected-bank-mode23-amr'
_MODE_DISPATCH_CONTRACT = 'mode_dispatch_v1'
_BEDFORM_BANK_KIND = 'live-selected-bank-mode3-bedform-avalanche-amr-v3'
_BEDFORM_BANK_CONTRACT = 'recursive_bedform_mode3_parent_avalanche_v1'
_BEDFORM_PHYSICAL_TOE_KIND = \
    'live-selected-bank-mode2-bedform-avalanche-amr-v1'
_BEDFORM_PHYSICAL_TOE_CONTRACT = \
    'recursive_bedform_mode2_physical_toe_parent_avalanche_v1'
_EXACT_BEDFORM_BANK_RUNTIMES = {
    _BEDFORM_BANK_KIND: dict(
        bank_mode=3, contract=_BEDFORM_BANK_CONTRACT,
        toe_geometry_seed_only=False,
        parser_scope_revision='exact_recursive_mode3_bedform_parent_avalanche_v3',
        geometry_initialization_version=6),
    _BEDFORM_PHYSICAL_TOE_KIND: dict(
        bank_mode=2, contract=_BEDFORM_PHYSICAL_TOE_CONTRACT,
        toe_geometry_seed_only=True,
        parser_scope_revision=
            'exact_recursive_mode2_physical_toe_bedform_parent_avalanche_v1',
        geometry_initialization_version=7),
}


def recursive_boundary_event_identity(data):
    """Return the worker event-clock identity for a recursive operator.

    Current live/scheduled operators carry the complete normal-metadata table
    family and are checked by the shared strict decoder.  Historical and exact
    constant-stage operators can legitimately carry only the three all-zero
    count arrays in their coupled input.  The worker represents that case as
    ``BoundaryEvents([])``; reproduce that identity without weakening the
    scheduled/live path.
    """
    present = tuple(key in data for key in TABLE_KEYS)
    if all(present):
        return boundary_event_identity(data)
    _require(not any(present),
             'recursive boundary event tables are incomplete')
    mode = np.asarray(data.get('mode'))
    nb, ng = data.get('nb'), data.get('nfrac')
    _require(type(nb) is int and nb == 1 and type(ng) is int and ng > 0 and
             mode.shape == (1,) and mode.dtype == np.dtype('int32') and
             np.array_equal(mode, np.asarray([2], dtype=np.int32)),
             'boundary event tables are required for live/scheduled forcing')
    for name, shape in (('nseries', (nb,)), ('nhydro_series', (nb,)),
                        ('nfraction_series', (ng, nb))):
        counts = np.asarray(data.get(name))
        _require(counts.shape == shape and counts.dtype == np.dtype('int32') and
                 not np.any(counts),
                 'constant-stage boundary event counts must be explicit zeros')
    return hashlib.sha256(b'BRAIDS boundary event clock v1\0').hexdigest()


def _recursive_adaptation_scope(data):
    """Validate the opt-in recursive closure without importing worker code."""
    enabled = data.get('bedload_adaptation', 0)
    _require(type(enabled) is int and enabled in (0, 1),
             'recursive bedload adaptation switch must be integer 0 or 1')
    if not enabled:
        return False
    control = np.asarray(data.get('section_control'))
    length = np.asarray(data.get('bedload_adaptation_length_diameters'))
    bank_mode=data.get('bank_toe_erosion_mode')
    _require(np.array_equal(data.get('mode'), [9, 4]) and
             type(bank_mode) is int and bank_mode in (0,3) and
             type(data.get('secondary_flow_model')) is int and data['secondary_flow_model'] == 2 and
             control.shape == (2,) and control.dtype.kind in 'iu' and int(control[0]) in (0, 1, 2) and
             length.shape == () and length.dtype.kind in 'fiu' and np.isfinite(length) and float(length) > 0.,
             'recursive adaptation requires surveyed prognostic-secondary mode0/3 scope and a positive finite length')
    return True


def _recursive_general_avalanche_scope(data):
    """Validate recursive nonbank avalanche admission without worker imports.

    Avalanche is an opt-in parent transaction.  Keeping this check separate
    from the base recursive capability lets historical workers continue to run
    avalanche-OFF preparations while refusing to acquire the new ON closure.
    """
    enabled = data.get('general_avalanche', 0)
    _require(type(enabled) is int and enabled in (0, 1),
             'recursive general avalanche switch must be integer 0 or 1')
    if not enabled:
        return False
    mode = np.asarray(data.get('mode'))
    viscosity = np.asarray(data.get('horizontal_viscosity'))
    viscosity_mode = data.get('horizontal_viscosity_mode')
    viscosity_geometry = data.get('horizontal_viscosity_geometry')
    _require(mode.shape == (2,) and mode.dtype.kind in 'iu' and
             np.array_equal(mode, [9, 4]) and
             type(data.get('bank_toe_erosion_mode')) is int and
             data['bank_toe_erosion_mode'] == 0 and
             type(data.get('bank_failure')) is int and data['bank_failure'] == 0 and
             type(data.get('bank_front_tracking')) is int and
             data['bank_front_tracking'] == 0 and
             type(data.get('bedlaw')) is int and data['bedlaw'] == 2 and
             type(data.get('exchange_mode')) is int and data['exchange_mode'] == 2 and
             type(data.get('secondary_flow_model')) is int and
             data['secondary_flow_model'] == 2 and
             viscosity.shape == () and viscosity.dtype.kind in 'fiu' and
             np.isfinite(viscosity) and float(viscosity) >= 0. and
             (float(viscosity) == 0. or
              (type(viscosity_mode) is int and viscosity_mode == 1 and
               type(viscosity_geometry) is int and viscosity_geometry == 1)) and
             type(data.get('bedform_roughness')) is int and
             data['bedform_roughness'] == 0,
             'recursive general avalanche requires nonbank MPM2, exchange 2, '
             'prognostic secondary flow, supported mode1/geometry1 viscosity '
             'and zero bedform feedback')
    return True


def _recursive_general_viscosity_scope(data):
    """Validate the opt-in positive-viscosity recursive closure.

    Zero remains the historical route and deliberately does not require a new
    package capability.  Positive viscosity is admitted only for the native
    velocity-Laplacian mode/geometry pair whose topology-local sidecar has
    already been authenticated by the general prepared reader.
    """
    viscosity = np.asarray(data.get('horizontal_viscosity'))
    _require(viscosity.shape == () and viscosity.dtype.kind in 'fiu' and
             np.isfinite(viscosity) and float(viscosity) >= 0.,
             'recursive general viscosity must be a finite nonnegative scalar')
    if float(viscosity) == 0.:
        return False
    mode = np.asarray(data.get('mode'))
    avalanche = data.get('general_avalanche')
    _require(mode.shape == (2,) and mode.dtype.kind in 'iu' and
             np.array_equal(mode, [9, 4]) and
             type(data.get('bank_toe_erosion_mode')) is int and
             data['bank_toe_erosion_mode'] == 0 and
             type(data.get('bank_failure')) is int and data['bank_failure'] == 0 and
             type(data.get('bank_front_tracking')) is int and
             data['bank_front_tracking'] == 0 and
             type(data.get('bedlaw')) is int and data['bedlaw'] == 2 and
             type(data.get('exchange_mode')) is int and data['exchange_mode'] == 2 and
             type(data.get('secondary_flow_model')) is int and
             data['secondary_flow_model'] == 2 and
             type(data.get('horizontal_viscosity_mode')) is int and
             data['horizontal_viscosity_mode'] == 1 and
             type(data.get('horizontal_viscosity_geometry')) is int and
             data['horizontal_viscosity_geometry'] == 1 and
             type(avalanche) is int and avalanche in (0, 1) and
             type(data.get('bedform_roughness')) is int and
             data['bedform_roughness'] == 0,
             'recursive general viscosity requires nonbank MPM2, exchange 2, '
             'prognostic secondary flow, native mode1/geometry1, avalanche 0/1 '
             'and zero bedform feedback')
    return True


def _recursive_general_bedform_scope(data):
    """Validate the qualified recursive equilibrium-bedform branch.

    Bedform properties are topology-local derived scratch.  They may be
    reconstructed from the accepted hydraulic/material state, so this host
    check deliberately does not require a new checkpoint field.  Enabling the
    closure is nevertheless an exact opt-in: only the already qualified
    unsuspended, one-substrate, roughness-corrected MPM envelope is admitted.
    """
    enabled = data.get('bedform_roughness', 0)
    _require(type(enabled) is int and enabled in (0, 1),
             'recursive general bedform switch must be integer 0 or 1')
    mode = np.asarray(data.get('mode'))
    bedform_boundary = (mode.shape == (2,) and mode.dtype.kind in 'iu' and
        np.count_nonzero(mode == 1) == 1 and
        np.count_nonzero(mode == 2) == 1)
    # The qualified one-flag control uses the same legacy envelope with the
    # switch OFF.  It still requires the new worker: historical recursive
    # [9,4]/exchange-2 jobs remain compatible, but cannot silently acquire
    # this different boundary/transport branch.
    if not enabled and not bedform_boundary:
        return False
    sampled = np.asarray(data.get('sampled'))
    offset = np.asarray(data.get('boundary_bed_offset',
                                 data.get('bedoffset')))
    control = np.asarray(data.get('section_control'))
    morph = np.asarray(data.get('morfac', data.get('morphological_factor')))
    morph_alias_equal = ('morfac' not in data or
        'morphological_factor' not in data or
        np.array_equal(np.asarray(data['morfac']),
                       np.asarray(data['morphological_factor'])))
    forcing = np.asarray(data.get('forcing'))
    state = np.asarray(data.get('state'))
    suspended = np.asarray(data.get('suspended_indices'))
    schedules = tuple(np.asarray(data.get(key)) for key in
                      ('nseries', 'nhydro_series', 'nfraction_series'))
    forcing_shape_ok = forcing.shape in ((2, 2), (6, 2))
    forcing_rows_ok = (bedform_boundary and forcing_shape_ok and
        np.isfinite(forcing).all() and
        float(forcing[0, np.flatnonzero(mode == 1)[0]]) > 0. and
        float(forcing[0, np.flatnonzero(mode == 2)[0]]) == 0. and
        (forcing.shape[0] == 2 or
         (not np.any(forcing[2:4]) and np.all(forcing[4] > 0.) and
          not np.any(forcing[5]))))
    _require(
        bedform_boundary and
        sampled.ndim == 1 and sampled.dtype.kind in 'iu' and
        offset.shape == sampled.shape and offset.dtype.kind in 'fiu' and
        not np.any(sampled) and not np.any(offset) and
        control.shape == (2,) and control.dtype.kind in 'iu' and
        not np.any(control) and
        all(value.dtype.kind in 'iu' and not np.any(value)
            for value in schedules) and
        forcing_rows_ok and
        type(data.get('bank_toe_erosion_mode')) is int and
        data['bank_toe_erosion_mode'] == 0 and
        type(data.get('bank_failure')) is int and data['bank_failure'] == 0 and
        type(data.get('bank_front_tracking')) is int and
        data['bank_front_tracking'] == 0 and
        type(data.get('bedlaw')) is int and data['bedlaw'] == 2 and
        type(data.get('mpm_friction_mode')) is int and
        data['mpm_friction_mode'] == 1 and
        type(data.get('bed_slope')) is int and data['bed_slope'] == 0 and
        type(data.get('hiding_exposure')) is int and
        data['hiding_exposure'] == 0 and
        type(data.get('exchange_mode')) is int and data['exchange_mode'] == 0 and
        type(data.get('secondary_flow_model')) is int and
        data['secondary_flow_model'] == 0 and
        type(data.get('bedload_adaptation')) is int and
        data['bedload_adaptation'] == 0 and
        type(data.get('general_avalanche')) is int and
        data['general_avalanche'] == 0 and
        type(data.get('nsubstrate_layers')) is int and
        data['nsubstrate_layers'] == 1 and
        type(data.get('suspended_transport_active')) is int and
        data['suspended_transport_active'] == 0 and
        morph_alias_equal and morph.shape == () and morph.dtype.kind in 'fiu' and
        np.isfinite(morph) and float(morph) == 1. and
        np.asarray(data.get('horizontal_viscosity')).shape == () and
        np.asarray(data.get('horizontal_viscosity')).dtype.kind in 'fiu' and
        np.isfinite(data['horizontal_viscosity']) and
        float(data['horizontal_viscosity']) == 0. and
        np.asarray(data.get('fraction_concentration')).dtype.kind in 'fiu' and
        not np.any(data['fraction_concentration']) and
        state.ndim == 2 and suspended.ndim == 1 and
        suspended.dtype.kind in 'iu' and
        np.all((suspended >= 0) & (suspended < state.shape[0])) and
        not np.any(state[suspended]),
        'recursive general bedform requires the qualified unscheduled mode-1/'
        'stage-2, unsuspended one-substrate MPM-friction1 envelope with '
        'slope/hiding/secondary/adaptation/viscosity/avalanche/banks off and MorFac 1')
    return True


def _recursive_bank_bedform_avalanche_scope(data):
    """Admit only the recursive stage/mode-2/3 bedform/avalanche composition.

    Bedform fields remain derived GPU scratch, while bank histories are the
    persistent morphology state.  This predicate is intentionally independent
    of the general mode-1/mode-2 bedform route and admits no OFF control.
    """
    enabled = data.get('bedform_roughness', 0)
    _require(type(enabled) is int and enabled in (0, 1),
             'recursive bank bedform switch must be integer 0 or 1')
    if not enabled:
        return False
    mode = np.asarray(data.get('mode'))
    sampled = np.asarray(data.get('sampled'))
    offset = np.asarray(data.get('boundary_bed_offset', data.get('bedoffset')))
    control = np.asarray(data.get('section_control'))
    morph = np.asarray(data.get('morfac', data.get('morphological_factor')))
    morph_alias_equal = ('morfac' not in data or
        'morphological_factor' not in data or
        np.array_equal(np.asarray(data['morfac']),
                       np.asarray(data['morphological_factor'])))
    state = np.asarray(data.get('state'))
    suspended = np.asarray(data.get('suspended_indices'))
    schedules = tuple(np.asarray(data.get(key)) for key in
                      ('nseries', 'nhydro_series', 'nfraction_series'))
    zero_controls = ('bank_front_tracking','bank_avalanche',
        'bank_erosion_fraction','bedload_adaptation','horizontal_viscosity_mode',
        'horizontal_viscosity_geometry','morphological_acceleration','nmorfac')
    _require(
        mode.shape == (1,) and mode.dtype.kind in 'iu' and int(mode[0]) == 2 and
        type(data.get('nb')) is int and data['nb'] == 1 and
        type(data.get('nf')) is int and data['nf'] > 0 and
        sampled.shape == (data['nf'],) and sampled.dtype.kind in 'iu' and
        not np.any(sampled) and
        offset.shape == sampled.shape and offset.dtype.kind in 'fiu' and not np.any(offset) and
        control.shape == (1,) and control.dtype.kind in 'iu' and not np.any(control) and
        all(value.dtype.kind in 'iu' and not np.any(value) for value in schedules) and
        type(data.get('bank_toe_erosion_mode')) is int and
        data['bank_toe_erosion_mode'] in (2, 3) and
        type(data.get('bank_failure')) is int and data['bank_failure'] == 1 and
        type(data.get('general_avalanche')) is int and data['general_avalanche'] == 1 and
        type(data.get('bedlaw')) is int and data['bedlaw'] == 2 and
        type(data.get('mpm_friction_mode')) is int and data['mpm_friction_mode'] == 1 and
        type(data.get('bed_slope')) is int and data['bed_slope'] == 0 and
        type(data.get('hiding_exposure')) is int and data['hiding_exposure'] == 0 and
        type(data.get('exchange_mode')) is int and data['exchange_mode'] == 0 and
        type(data.get('secondary_flow_model')) is int and data['secondary_flow_model'] == 0 and
        type(data.get('nsubstrate_layers')) is int and data['nsubstrate_layers'] == 1 and
        type(data.get('nbed_layers')) is int and data['nbed_layers'] == 2 and
        type(data.get('cpu_time_integrator')) is int and data['cpu_time_integrator'] == 33 and
        type(data.get('suspended_transport_active')) is int and
        data['suspended_transport_active'] == 0 and
        morph_alias_equal and morph.shape == () and morph.dtype.kind in 'fiu' and
        np.isfinite(morph) and float(morph) == 1. and
        np.asarray(data.get('horizontal_viscosity')).shape == () and
        np.asarray(data.get('horizontal_viscosity')).dtype.kind in 'fiu' and
        np.isfinite(data['horizontal_viscosity']) and
        float(data['horizontal_viscosity']) == 0. and
        all(not np.any(np.asarray(data.get(name, 0))) for name in zero_controls) and
        np.asarray(data.get('fraction_concentration')).dtype.kind in 'fiu' and
        not np.any(data['fraction_concentration']) and
        state.ndim == 2 and suspended.ndim == 1 and suspended.dtype.kind in 'iu' and
        np.all((suspended >= 0) & (suspended < state.shape[0])) and
        not np.any(state[suspended]),
        'recursive bank bedform composition requires one unscheduled constant-stage '
        'boundary, evolving physical-toe mode-2 or distributed mode-3 banks, '
        'parent avalanche, unsuspended '
        'one-substrate MPM-friction1, secondary/adaptation/viscosity off and MorFac 1')
    return True


def _bedform_preparation_request(base, request, *, composed, bank_mode):
    """Return the exact request expected for a committed exact composition.

    Mode 2 was introduced with an explicit bank-mode discriminator and must
    always carry it.  Completed mode-3 archives predate that discriminator;
    accept its omission there, while authenticating it exactly when present.
    """
    expected = dict(base)
    if composed:
        expected['recursive_bank_bedform_avalanche'] = True
        if bank_mode == 2 or 'bank_mode' in request:
            expected['bank_mode'] = bank_mode
        if 'recursive_geometry_adapter' in request:
            expected['recursive_geometry_adapter'] = \
                request['recursive_geometry_adapter']
    return expected


_SIDECARS = {"avalanche": "avalanche_metadata.bin",
             "bank_history": "bank_history_metadata.bin", "bank_toe": "bank_toe_metadata.bin"}
_GLOBAL_LINEAGE = {"segment_id", "bank_node", "toe_node", "midpoint_xy_m", "length_m"}
_GLOBAL_ARRAYS = ("d50_fraction", "d90_fraction", "dstar_fraction", "rhos_fraction",
    "sedthr_fraction", "ws_fraction", "z0_fraction", "sediment_type_fraction",
    "sediment_roughness_fraction", "reference_height_fraction", "bulk_fraction", "secondary_flow",
    "mode", "bed_indices", "suspended_indices", "forcing", "fraction_concentration", "fraction_bed_input",
    "section_energy_slope", "reversal_policy", "bed_boundary_mode", "section_control",
    "nseries", "nhydro_series", "nfraction_series")


def _require(value, message):
    if not value:
        raise ValueError("On-demand GPU preparation: " + message)


def _sha(path):
    with filesystem_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _hex(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(letter in "0123456789abcdef" for letter in value))


def _linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _regular(path, *, directory=False):
    path = filesystem_path(path)
    _require(not _linked(path) and (path.is_dir() if directory else path.is_file()),
             "missing or linked " + str(path))
    return path


def _base_input_digest(originals):
    names=[name for name in originals if name=='case.data' or
           (name.startswith('.br') and name.endswith('.dat'))]
    _require(len(names)==1 and _hex(originals[names[0]]),
             'original-input deck inventory differs')
    return originals[names[0]]


def _verify_recursive_geometry_adapter(source, adapted, record, bank_mode):
    """Independently reproduce the metadata-only v54-to-v38/v40 view."""
    keys={'schema','applied','outer_version','base_version',
          'expected_base_version','stripped_metadata_versions',
          'source_deck_sha256','adapted_deck_name','adapted_deck_sha256',
          'physical_payload_unchanged','identity'}
    expected_base=38 if bank_mode==2 else 40
    _require(isinstance(record,dict) and set(record)==keys and
             record.get('identity')==_digest({key:value for key,value in
                 record.items() if key!='identity'}) and
             record.get('schema')==1 and record.get('applied') is True and
             record.get('outer_version')==54 and
             record.get('base_version')==expected_base and
             record.get('expected_base_version')==expected_base and
             record.get('stripped_metadata_versions')==[54,53,52,51,50] and
             record.get('physical_payload_unchanged') is True and
             record.get('source_deck_sha256')==_sha(source) and
             record.get('adapted_deck_name')==adapted.name and
             record.get('adapted_deck_sha256')==_sha(adapted),
             'recursive geometry adapter receipt differs')
    try:
        rows=source.read_text(encoding='ascii').splitlines()
        _require(len(rows)>10 and len(rows[1])==64 and _hex(rows[1]),
                 'recursive source deck fingerprint differs')
        logical=[rows[0],*rows[2:]]
        cursor=0;versions=[]
        while int(logical[cursor])>41:
            version=int(logical[cursor]);versions.append(version)
            if version==54:
                counts=[int(value) for value in logical[cursor+1].split()]
                _require(len(counts)==5 and counts[0]==1 and
                         all(value>=0 for value in counts[1:]),
                         'recursive v54 identity envelope differs')
                cursor+=2+counts[2]+counts[4]
                _require(int(logical[cursor])==53,
                         'recursive v54 nested deck differs')
            elif version==53:
                count=int(logical[cursor+1]);_require(count>0,
                    'recursive v53 component count differs')
                cursor+=2
                for _ in range(count):
                    header=logical[cursor].split()
                    _require(len(header)==4 and int(header[3])>0,
                             'recursive v53 component header differs')
                    cursor+=1+int(header[3])
                _require(int(logical[cursor])==52,
                         'recursive v53 nested deck differs')
            elif version==52:
                header=[int(value) for value in logical[cursor+1].split()]
                _require(len(header)==4 and header[0]==2 and
                         all(value>0 for value in header[1:]),
                         'recursive v52 composition envelope differs')
                cursor+=2+4+header[1]+1+header[3]
                _require(int(logical[cursor])==2,
                         'recursive execution-profile schema differs')
                size=int(logical[cursor+3]);_require(size>0,
                    'recursive execution-profile size differs')
                cursor+=4+(2*size+127)//128
                _require(int(logical[cursor])==51,
                         'recursive v52 nested deck differs')
            elif version==51:
                cursor+=6;_require(int(logical[cursor])==50,
                    'recursive v51 nested deck differs')
            elif version==50:
                cursor+=8
            else:
                _require(False,'unsupported recursive geometry envelope')
        _require(versions==[54,53,52,51,50] and
                 int(logical[cursor])==expected_base,
                 'recursive physical geometry version differs')
        expected=('\n'.join((logical[cursor],rows[1],
                             *logical[cursor+1:]))+'\n').encode('ascii')
    except (IndexError,TypeError,UnicodeError):
        _require(False,'malformed recursive geometry identity envelope')
    _require(adapted.read_bytes()==expected,
             'recursive geometry adapter changed the physical payload')


def _object(path):
    value = json.loads(_regular(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), Path(path).name + " must contain an object")
    return value


def _ids(value):
    _require(isinstance(value, list) and bool(value) and
             all(type(item) is int and item >= 0 for item in value), "invalid selected node IDs")
    result = np.asarray(value, dtype=np.int64)
    _require(np.all(np.diff(result) > 0), "selected node IDs must be sorted and unique")
    return result


def _lineage(rows, *, local=False):
    _require(isinstance(rows, list) and bool(rows), "missing physical-toe lineage")
    normalized = []
    for row in rows:
        _require(isinstance(row, dict) and set(row) == _GLOBAL_LINEAGE | ({"local_face"} if local else set()),
                 "invalid physical-toe lineage fields")
        for key in ("segment_id", "bank_node", "toe_node") + (("local_face",) if local else ()):
            _require(type(row[key]) is int and row[key] >= 0, "invalid lineage " + key)
        point, length = row["midpoint_xy_m"], row["length_m"]
        _require(isinstance(point, list) and len(point) == 2 and
                 all(type(value) in (int, float) and np.isfinite(value) for value in point) and
                 type(length) in (int, float) and np.isfinite(length) and length > 0,
                 "invalid physical-toe geometry")
        normalized.append({**row, "midpoint_xy_m": list(map(float, point)), "length_m": float(length)})
    normalized.sort(key=lambda row: row["segment_id"])
    _require(len({row["segment_id"] for row in normalized}) == len(normalized) and
             (not local or len({row["local_face"] for row in normalized}) == len(normalized)),
             "duplicate physical-toe lineage")
    return normalized


def _file_inventory(folder):
    folder = _regular(folder, directory=True)
    files = {}
    for path in folder.rglob("*"):
        _require(not _linked(path) and path.resolve().is_relative_to(folder), "linked package member")
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = _sha(path)
        else:
            _require(path.is_dir(), "invalid package member")
    return files


def _all_face_lineage(path, row, package, tree_sha):
    """Host-only mirror of the evolving operator store's native-face contract."""
    folder = _regular(path / "face_identity", directory=True)
    names = {"amr_selected_identity.bin", "amr_tree_input.bin", "selection_receipt.json"}
    files = _file_inventory(folder)
    _require(set(files) == names and row["face_identity_files"] == files and
             files["amr_tree_input.bin"] == tree_sha, "all-face source artifact identity differs")
    content = _digest(dict(package_identity=row["package_identity"], face_identity_files=files))
    _require(row["object_identity"] == content and path.name == content,
             "evolving operator content address differs")
    selected = _read_records(folder / "amr_selected_identity.bin")
    tree = _read_records(folder / "amr_tree_input.bin")
    receipt = _object(folder / "selection_receipt.json")
    top = package["topology"]
    ids, data = top["node_ids"], top["initial_data"]
    nf = int(data["nf"])
    _require(set(selected) == {"node_ids", "face_source_ids"} and
             np.asarray(selected["node_ids"]).dtype == np.dtype("int32") and
             np.array_equal(selected["node_ids"], ids), "native selected global IDs differ")
    faces = np.asarray(selected["face_source_ids"])
    _require(faces.dtype == np.dtype("int32") and faces.shape == (nf,), "native all-face dimensions differ")
    _require(receipt.get("node_ids") == ids.tolist() and receipt.get("tree_input_sha256") == tree_sha and
             receipt.get("selection_identity_sha256") == files["amr_selected_identity.bin"] and
             receipt.get("original_input_sha256") == package["manifest"]["source"]["base_input_sha256"] and
             type(receipt.get("native_solver_steps")) is int and receipt["native_solver_steps"] == 0 and
             receipt.get("full_state_transfer_on_GPU") is True, "native all-face selection receipt differs")
    for key in ("parent", "node_level", "maximum_level", "area"):
        _require(np.array_equal(tree.get(key), package["tree"][key]), "all-face hierarchy differs: " + key)
    _require(np.array_equal(np.asarray(tree.get("xy")).T, package["tree"]["xy"]),
             "all-face hierarchy coordinates differ")
    parent = package["tree"]["parent"]
    nn = len(parent)
    owner, other = np.asarray(tree.get("face_owner")), np.asarray(tree.get("face_neighbour"))
    mid, length = np.asarray(tree.get("face_mid")), np.asarray(tree.get("face_length"))
    fn = owner.size
    _require(owner.dtype == np.dtype("int32") and owner.shape == (fn,) and fn > 0 and
             other.dtype == np.dtype("int32") and other.shape == (fn,) and
             np.all((owner >= 0) & (owner < nn)) and np.all((other >= -1) & (other < nn)) and
             not np.any(owner == other) and mid.dtype == np.float64 and mid.shape == (2, fn) and
             np.isfinite(mid).all() and length.dtype == np.float64 and length.shape == (fn,) and
             np.all(length > 0), "invalid native all-face geometry")
    cover = np.full(nn, -1, dtype=np.int64)
    cover[ids] = ids
    for node, ancestor in enumerate(parent):
        if ancestor >= 0 and cover[ancestor] >= 0:
            cover[node] = cover[ancestor]
    a = cover[owner]
    internal = other >= 0
    b = np.full(fn, -1, dtype=np.int64)
    b[internal] = cover[other[internal]]
    used = (a >= 0) & np.where(internal, (b >= 0) & (a != b) & ((owner == a) | (other == b)), owner == a)
    _require(np.array_equal(faces.astype(np.int64), np.flatnonzero(used) + 1),
             "all-face source IDs differ from selected hierarchy interfaces")
    source = faces.astype(np.int64) - 1
    o, n = np.asarray(data["owner"]), np.asarray(data["neighbour"])
    nodes = np.full(nf, -1, dtype=np.int64)
    nodes[n >= 0] = ids[n[n >= 0]]
    normal = np.asarray(data["normal"])
    _require(np.array_equal(ids[o], a[source]) and np.array_equal(nodes, b[source]) and
             np.array_equal(data["mid"], mid[:, source]) and np.array_equal(data["length"], length[source]) and
             normal.dtype == np.float64 and normal.shape == (2, nf) and np.isfinite(normal).all() and
             np.allclose(np.sum(normal * normal, axis=0), 1., rtol=0., atol=1e-12),
             "all-face operator geometry/orientation differs")
    expected = [dict(local_face=f, source_face_id=int(faces[f]), owner_node=int(ids[o[f]]),
        neighbour_node=int(nodes[f]), midpoint_xy_m=mid[:, source[f]].tolist(),
        length_m=float(length[source[f]]), normal_xy=normal[:, f].tolist()) for f in range(nf)]
    _require(_digest(expected) == _digest(row["lineage"]), "all-face lineage differs from native operator")


def _toe_geometry(data, *, seed_only=False):
    """Validate a complete cold map, even when it is only an evolving-map seed.

    Status 2 preserves an ambiguous *static* assignment for observation; it does
    not qualify that assignment as a runtime map. Only the authenticated
    evolving route may resolve and use these seeds.
    """
    nc, nf = int(data["nc"]), int(data["nf"])
    marker = data.get("toe_geometry_seed_only")
    _require((seed_only and type(marker) is int and marker == 1) or
             (not seed_only and "toe_geometry_seed_only" not in data),
             "geometry-only bank-toe sidecar marker differs")

    def integers(name, size):
        value = np.asarray(data.get(name))
        _require(value.dtype == np.dtype("int32") and value.shape == (size,),
                 "invalid physical-toe integer array " + name)
        return value

    flags = {name: integers(name, size) for name, size in (
        ("bank_material_mask", nc), ("is_toe_cell", nc),
        ("is_toe_donor", nc), ("is_toe_face", nf))}
    _require(all(np.isin(value, (0, 1)).all() for value in flags.values()),
             "nonbinary physical-toe flags")
    material, active = flags["bank_material_mask"].astype(bool), flags["is_toe_face"].astype(bool)
    status = integers("physical_toe_status", nc)
    _require(type(data.get("toe_mapping_status")) is int and data["toe_mapping_status"] == 0 and
             np.any(material) and np.any(active) and
             np.isin(status[material], (1, 2) if seed_only else (1,)).all() and
             not np.any(status[~material]), "incomplete physical-toe mapping")
    ids = {name: integers(name, nc) for name in
           ("physical_toe_cell", "physical_toe_donor", "physical_toe_face")}
    _require(all(np.all(value[~material] == -1) for value in ids.values()),
             "physical-toe IDs outside bank material")
    target, donor, face = (ids[name][material] for name in
                          ("physical_toe_cell", "physical_toe_donor", "physical_toe_face"))
    _require(np.all((target >= 0) & (target < nc)) and
             np.all((donor >= 0) & (donor < nc)) and
             np.all((face >= 0) & (face < nf)), "invalid physical-toe IDs")
    distance = np.asarray(data.get("physical_toe_distance"))
    _require(distance.dtype == np.float64 and distance.shape == (nc,) and
             np.isfinite(distance).all() and np.all(distance >= 0), "invalid physical-toe distance")
    bank, toe = (integers(name, nf) for name in ("candidate_bank", "candidate_toe"))
    _require(np.all(bank[~active] == -1) and np.all(toe[~active] == -1) and
             np.all((bank[active] >= 0) & (bank[active] < nc)) and
             np.all((toe[active] >= 0) & (toe[active] < nc)), "invalid physical-toe candidate IDs")
    _require(not np.any(material[target]) and np.all(material[donor]) and
             np.all(active[face]) and np.array_equal(bank[face], donor) and
             np.array_equal(toe[face], target), "mapped physical-toe face differs")
    owner, other = (integers(name, nf) for name in ("owner", "neighbour"))
    a, b = bank[active], toe[active]
    _require(np.all(material[a]) and not np.any(material[b]) and
             np.all(((owner[active] == a) & (other[active] == b)) |
                    ((owner[active] == b) & (other[active] == a))) and
             np.array_equal(flags["is_toe_donor"].astype(bool), np.isin(np.arange(nc), a)) and
             np.array_equal(flags["is_toe_cell"].astype(bool), np.isin(np.arange(nc), b)),
             "physical-toe candidate face graph differs")


def _scope(data, *, live_boundaries=False, toe_geometry_seed_only=False, bank_mode=2):
    _require(type(bank_mode) is int and bank_mode in (2, 3), "invalid bank mode")
    live = live_boundaries and np.array_equal(data.get("mode"), [9, 4])
    composed = _recursive_bank_bedform_avalanche_scope(data)
    required = dict(scheme=6, bedlaw=2, exchange_mode=0, mpm_friction_mode=1,
        secondary_flow_model=0 if composed or bank_mode == 3 and not live else 2,
        bank_failure=1,
        bank_toe_erosion_mode=bank_mode, amr_enabled=0, cpu_time_integrator=33,
        cpu_outlet_extension=0, suspended_transport_active=0)
    for name, expected in required.items():
        _require(type(data.get(name)) is int and data[name] == expected,
                 "unsupported control " + name)
    _require(live or (np.array_equal(data.get("mode"), [2]) and data.get("nb") == 1),
             "requires a constant stage boundary or authenticated surveyed inlet/critical outlet")
    # The native geometry-only seed contract is admitted only by an evolving
    # session in _package. Broaden exactly its live sediment controls; legacy
    # stage/static-map packages retain the previous zero-control envelope.
    graded = live and (toe_geometry_seed_only or bank_mode == 3)
    for name, options in (("hiding_exposure", (0, 4)), ("bed_slope", (0, 2)),
                          ("bedload_adaptation", (0, 1))):
        _require(type(data.get(name)) is int and data[name] in (options if graded else (0,)),
                 "unsupported graded control " + name)
    if graded:
        length = np.asarray(data.get("bedload_adaptation_length_diameters"))
        _require(length.shape == () and length.dtype.kind in "fiu" and np.isfinite(length) and
                 (not data["bedload_adaptation"] or float(length) > 0.),
                 "bedload adaptation requires a finite positive enabled length")
    if live:
        _require(data.get("nb") == 2, "live boundary count differs")
        count = data["nhydro_series"]
        _require(count[0] >= 2 and count[1] == 0 and
                 data["hydro_series_time"][0, 0] <= data["time"] and
                 data["hydro_series_time"][count[0] - 1, 0] >= data["endtime"] and
                 np.all(data["hydro_series_q"][:count[0], 0] > 0),
                 "live inlet requires positive discharge schedule covering the run")
        for grain, active in enumerate(data["nfraction_series"][:, 0]):
            if active:
                times = data["fraction_series_time"][:active, grain, 0]
                _require(times[0] <= data["time"] and times[-1] >= data["endtime"],
                         "live class feed schedule must cover the run")
    else:
        for name in ("nseries", "nhydro_series", "nfraction_series"):
            _require(name in data and not np.any(data[name]), "unsupported active control " + name)
    _require(not composed or (not live and bank_mode in (2, 3)),
              'recursive bank bedform composition requires constant-stage bank mode 2 or 3')
    for name in ("fraction_concentration",
                 "active_layer_mode", "nmorfac", "bank_front_tracking", "bank_avalanche",
                 "bank_erosion_fraction", "bedform_roughness",
                 "viscosity_equilibrated_inlet", "horizontal_viscosity"):
        _require(name in data and (name == 'bedform_roughness' and composed or
                                  not np.any(data[name])),
                 "unsupported active control " + name)
    _require(data.get("morphological_factor") == 1. and data.get("time") == 0.,
             "requires a cold input and MorFac 1")
    _require(data.get("general_avalanche") in (0, 1), "invalid general avalanche switch")
    _require(not composed or data['general_avalanche'] == 1,
             'recursive bank bedform composition requires parent avalanche ON')
    for name in ("general_avalanche_wet_slope", "general_avalanche_dry_slope", "general_avalanche_time",
                 "general_avalanche_relaxation_length"):
        _require(type(data.get(name)) in (int, float) and data[name] > 0, "invalid avalanche control " + name)
    _require(data.get("general_avalanche_wet_depth_switch", -1) >= 0 and
             (bank_mode == 3 or 0 <= data.get("bank_toe_capacity_fraction", -1) <= 1) and
             data.get("bank_critical_undercut_m", 0) > 0, "invalid avalanche/bank controls")
    for name in ("endtime", "output_dt", "dtmax", "cfl_target"):
        value = data.get(name)
        _require(type(value) in (int, float) and np.isfinite(value) and value > 0,
                 "invalid time control " + name)
    nc, nf, ng, nl = (int(data[key]) for key in ("nc", "nf", "nfrac", "nbed_layers"))
    state = np.asarray(data["state"])
    bed, suspended = np.asarray(data.get("bed_indices")), np.asarray(data.get("suspended_indices"))
    secondary_index = 3 + ng * (1 + nl) if required['secondary_flow_model'] else -1
    _require(np.array_equal(suspended, np.arange(3, 3 + ng)) and
             np.array_equal(bed, np.arange(3 + ng, 3 + ng * (1 + nl)).reshape(nl, ng)) and
             data.get("secondary_index") == secondary_index and
             data.get("nstate") == state.shape[0] == 3 + ng * (1 + nl) + (secondary_index >= 0),
             "invalid material/secondary state indices")
    _require(np.all(state[0] >= 0) and np.all(state[bed] >= 0) and not np.any(state[suspended]),
             "invalid or suspended initial inventory")
    dry = state[0] == 0
    _require(not np.any(state[1:3, dry]) and (secondary_index < 0 or not np.any(state[secondary_index, dry])),
             "dry initial cells carry momentum or secondary flow")
    for name in ("bank_undercut", "bank_reference_height", "bank_toe_height", "bank_reference_length",
                 "bank_toe_cumulative", "bank_collapse_cumulative", "bank_collapse_count"):
        values = np.asarray(data.get(name))
        _require(values.shape == (nc,) and np.isfinite(values).all() and np.all(values >= 0),
                 "invalid bank history " + name)
    if bank_mode == 2:
        _toe_geometry(data, seed_only=toe_geometry_seed_only)
    else:
        _require(not toe_geometry_seed_only and not any(key in data for key in
            ('is_toe_face', 'is_toe_donor', 'candidate_bank', 'candidate_toe', 'mapped_donor',
             'mapped_toe', 'mapped_face', 'toe_geometry_seed_only')), "mode 3 cannot carry physical-toe metadata")
        mask = np.asarray(data.get('bank_material_mask'))
        _require(mask.shape == (nc,) and mask.dtype == np.int32 and np.isin(mask, [0, 1]).all() and
                 np.any(mask), "invalid distributed-bank material mask")
        fraction, slope = data.get('bank_distributed_transport_fraction'), data.get('bank_distributed_critical_slope')
        _require(type(fraction) in (int, float) and np.isfinite(fraction) and 0 <= fraction <= 1 and
                 type(slope) in (int, float) and np.isfinite(slope) and slope > 0,
                 "invalid distributed-bank controls")


def _package(folder, identity, *, originals, runtime_sha, evolving_banks=False):
    folder = _regular(folder, directory=True)
    manifest_path = folder / "amr_prepared.json"
    manifest = _object(manifest_path)
    _require(_hex(identity) and _sha(manifest_path) == identity, "operator package identity differs")
    _require(manifest.get("schema") == 2 and manifest.get("kind") == "braids-gpu-amr-prepared" and
             manifest.get("ratio") == 2 and manifest.get("CPU_numerical_fallback") is False and
             type(manifest.get("solver_steps_performed_by_exporter")) is int and
             manifest["solver_steps_performed_by_exporter"] == 0, "unsupported operator package")
    source = manifest.get("source", {})
    live = source.get("live_boundary_metadata", False)
    _require(type(live) is bool, "invalid live-boundary source contract")
    bank_mode = source.get('bank_mode', 2)
    _require(type(bank_mode) is int and bank_mode in (2, 3) and (bank_mode != 3 or evolving_banks),
             'distributed-bank operator needs explicit evolving session')
    sidecars = {**_SIDECARS, **({"normal": "normal_metadata.bin"} if live else {})}
    if bank_mode == 3:
        sidecars.pop('bank_toe')
        sidecars['distributed_bank'] = 'distributed_bank_metadata.bin'
    _require(source.get("on_demand_initialization") is True and
             source.get("base_input_sha256") == _base_input_digest(originals) and
             source.get("hierarchy_sha256") == originals["amr_hierarchy.npz"] and
             source.get("runtime_build_sha256") == runtime_sha, "operator original-input family differs")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict) and bool(artifacts), "missing operator artifacts")
    inventory = _file_inventory(folder)
    _require(set(inventory) == set(artifacts) | {"amr_prepared.json"}, "unlisted or missing operator file")
    for name, expected in artifacts.items():
        _require(_hex(expected) and _sha(_artifact(folder, name)) == expected, "operator artifact differs: " + name)
    _require(manifest.get("runtime_build") == "runtime_build.json" and
             artifacts.get("runtime_build.json") == runtime_sha and "tree.npz" in artifacts,
             "operator runtime/tree binding differs")
    runtime_build = _object(folder / "runtime_build.json")
    runtime_kind = runtime_build.get('kind')
    exact = _EXACT_BEDFORM_BANK_RUNTIMES.get(runtime_kind)
    bedform_bank_runtime = exact is not None
    dispatched = runtime_kind == _MODE_DISPATCH_KIND or bedform_bank_runtime
    _require(not dispatched or
             (not bedform_bank_runtime and runtime_build.get('bank_modes') == [2, 3] and
              runtime_build.get('bank_metadata_contract') == _MODE_DISPATCH_CONTRACT and
              type(source.get('bank_mode')) is int) or
             (bedform_bank_runtime and
              runtime_build.get('bank_modes') == [exact['bank_mode']] and
              runtime_build.get('bank_metadata_contract') == exact['contract'] and
              runtime_build.get('parser_scope_revision') ==
                exact['parser_scope_revision'] and
               runtime_build.get('geometry_initialization_version') ==
                 exact['geometry_initialization_version'] and
               runtime_build.get('export_only') is True and
               runtime_build.get('recursive_bank_bedform_avalanche') is True and
               (exact['bank_mode'] != 2 or
                (runtime_build.get('evolving_toe_geometry') is True and
                 runtime_build.get('source_original_modified') is False and
                 runtime_build.get('Delft3D_source_modified') is False)) and
               source.get('bank_mode') == exact['bank_mode'] and
               source.get('recursive_bank_bedform_avalanche') is True),
             'invalid native bank dispatch contract')
    _require(bank_mode != 3 or dispatched, 'mode 3 needs the mode-dispatch native initializer')
    _require(runtime_build.get("live_boundary_metadata", False) is live and
              (not live or runtime_kind in ("live-selected-bank-amr",
                 _MODE_DISPATCH_KIND, _BEDFORM_BANK_KIND,
                 _BEDFORM_PHYSICAL_TOE_KIND)),
              "operator live-boundary runtime/source contract differs")
    seed_only = runtime_build.get("toe_geometry_seed_only", False) if bank_mode == 2 else False
    _require(type(evolving_banks) is bool and type(seed_only) is bool and
              source.get("toe_geometry_seed_only", False) is seed_only and
              (not seed_only or (evolving_banks and
               (live or (bedform_bank_runtime and exact['bank_mode'] == 2)))) and
              (not bedform_bank_runtime or
               seed_only is exact['toe_geometry_seed_only']),
              "geometry-only bank-toe initializer requires the evolving-map session")
    rows = manifest.get("topologies")
    _require(isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict),
             "operator must have one selected topology")
    row = rows[0]
    for name in ("node_ids", "geometry", "operator", "receipt"):
        _require(isinstance(row.get(name), str) and row[name] in artifacts,
                 "missing topology artifact " + name)
    tree = _tree(folder / "tree.npz", 2)
    nodes = _selected_nodes(_artifact(folder, row["node_ids"]), tree)
    topology_folder = _artifact(folder, row["node_ids"]).parent
    _require(all(_artifact(folder, row[name]).parent == topology_folder for name in ("geometry", "operator", "receipt")),
             "topology artifacts have different folders")
    _require(row.get("name") == "leaves_" + hashlib.sha256(nodes.astype("<i8").tobytes()).hexdigest(),
             "topology name differs from selected nodes")
    _require(row.get("sidecars") == {kind: topology_folder.name + "/" + name for kind, name in sidecars.items()},
             "physical-toe/avalanche/history sidecar inventory differs")
    topology_receipt = _object(_artifact(folder, row["receipt"]))
    files = topology_receipt.get("origin", {}).get("files", {})
    _require(isinstance(files, dict) and {key: value for key, value in files.items() if key != "selected_nodes.txt"} == originals and
             _hex(files.get("selected_nodes.txt")), "topology original inputs differ")
    _require(topology_receipt.get("schema") == 1 and topology_receipt.get("status") == "completed" and
             topology_receipt.get("runtime_build_sha256") == runtime_sha and
             topology_receipt.get("initialization_CPU_only") is True and
             topology_receipt.get("GPU_solver_executed") is False and
             type(topology_receipt.get("solver_steps_performed")) is int and
             topology_receipt["solver_steps_performed"] == 0, "operator is not zero-step initialization")
    _require(topology_receipt.get("live_boundary_metadata", False) is live,
             "native live-boundary metadata contract differs")
    _require(topology_receipt.get("toe_geometry_seed_only", False) is seed_only,
             "native geometry-only bank-toe contract differs")
    _require(not dispatched or (type(topology_receipt.get('bank_mode')) is int and
             topology_receipt['bank_mode'] == bank_mode), 'native bank mode contract differs')
    _require((topology_receipt.get('recursive_bank_bedform_avalanche') is True) is
             bedform_bank_runtime,
             'native bedform-bank initializer receipt contract differs')
    expected_native = {"coupled_input.bin", *sidecars.values()}
    _require(set(topology_receipt.get("artifacts", {})) == expected_native and
             all(topology_receipt["artifacts"][name] == artifacts.get(topology_folder.name + "/" + name)
                 for name in expected_native), "native receipt/package artifact binding differs")
    geometry = _geometry(topology_folder, {"origin": {"files": {"geometry.npz": artifacts[row["geometry"]]}}})
    _require(geometry["cell_count"] == len(nodes) and
             np.allclose(geometry["area_m2"], tree["area"][nodes], rtol=1e-11, atol=1e-11) and
             np.allclose(geometry["centroid_xy"], tree["xy"][nodes], rtol=1e-11, atol=1e-11),
             "selected geometry differs from hierarchy")
    data = read_gpu_prepared(topology_folder, receipt_sha256=artifacts[row["receipt"]])
    for kind, filename in sidecars.items():
        if kind == 'normal':
            continue
        for key, value in _read_records(topology_folder / filename).items():
            _require(key not in data or np.array_equal(data[key], value), "native sidecar conflicts: " + key)
            data[key] = value
    if live:
        data = validate_live_boundary_metadata(data, _read_records(topology_folder / "normal_metadata.bin"))
    _scope(data, live_boundaries=live, toe_geometry_seed_only=seed_only, bank_mode=bank_mode)
    owner, other = np.asarray(data.get("owner")), np.asarray(data.get("neighbour"))
    _require(owner.shape == (data["nf"],) and other.shape == owner.shape and
             owner.dtype.kind in "iu" and other.dtype.kind in "iu" and
             np.all((owner >= 0) & (owner < len(nodes))) and
             np.all((other >= -1) & (other < len(nodes))) and not np.any(owner == other),
             "invalid selected operator face graph")
    active = other >= 0
    levels = tree["node_level"][nodes]
    _require(not np.any(np.abs(levels[owner[active]] - levels[other[active]]) > 1) and
             np.allclose(data["area"], tree["area"][nodes], rtol=1e-12, atol=1e-12) and
             np.allclose(data["xy"].T, tree["xy"][nodes], rtol=1e-12, atol=1e-12),
             "selected operator geometry or 2:1 balance differs")
    boundary = np.asarray(data.get("boundary_faces"))
    _require(boundary.ndim == 1 and boundary.size > 0 and boundary.dtype.kind in "iu" and
             np.all((boundary >= 0) & (boundary < data["nf"])), "invalid stage outlet face IDs")
    _require(np.all(other[boundary] == -1) and np.all(levels[owner[boundary]] == np.max(tree["node_level"])) and
             (live or len(np.unique(owner[boundary])) == len(boundary)),
             "open boundary owners must be finest leaves (legacy stage owners unique)")
    _require(data["nc"] == len(nodes) and topology_receipt.get("cells") == len(nodes) and
             topology_receipt.get("faces") == data["nf"] and
             topology_receipt.get("bank_cells") == int(np.count_nonzero(data["bank_material_mask"])) and
             (topology_receipt.get("toe_faces") == int(np.count_nonzero(data["is_toe_face"])) if bank_mode == 2
              else 'toe_faces' not in topology_receipt) and
             all(data.get(key) == value for key, value in topology_receipt.get("controls", {}).items()),
             "native receipt dimensions/controls differ")
    topology = dict(name=row["name"], folder=topology_folder, node_ids=nodes, geometry=geometry,
                    receipt_sha256=artifacts[row["receipt"]], operator_sha256=artifacts[row["operator"]],
                    initial_data=data)
    return dict(root=folder, manifest_sha256=identity, tree=tree, topologies={row["name"]: topology},
                manifest=manifest, inventory=inventory, topology=topology, receipt=topology_receipt)


def _selector(tree, packed, manifest, *, initial_data,
              contract_transform=None):
    count = len(packed["parent"])
    for name in ("parent", "node_level", "maximum_level", "area"):
        _require(np.array_equal(tree.get(name), packed[name]), "native hierarchy differs: " + name)
    _require(np.array_equal(np.asarray(tree.get("xy")).T, packed["xy"]), "native hierarchy centroids differ")
    minimum, rock = np.asarray(tree.get("minimum_level")), np.asarray(tree.get("rock"))
    _require(minimum.shape == (count,) and minimum.dtype.kind in "iu" and
             np.all((minimum >= 0) & (minimum <= packed["maximum_level"])) and
             rock.shape == (count,) and rock.dtype == np.float64, "invalid native minimum levels/rock")
    flags = np.asarray(tree.get("flags"))
    _require(flags.shape == (5,) and flags.dtype.kind in "iu" and np.isin(flags, (0, 1)).all(),
             "invalid AMR indicator flags")
    settings = {name: bool(flags[index]) for index, name in enumerate(("refine_wet_dry", "refine_walls",
        "refine_bathymetry", "refine_hydraulics", "refine_sediment"))}
    for target, source in (("levels", "amr_levels"), ("refinement_ratio", "amr_ratio"),
                           ("coarsen_delay", "amr_delay"), ("wet_level", "amr_wet_level"),
                           ("buffer_cells", "amr_buffer")):
        _require(type(tree.get(source)) is int, "invalid AMR integer " + source)
        settings[target] = tree[source]
    for target, source in (("regrid_interval_s", "amr_interval"), ("wet_depth_m", "amr_wet_depth"),
                           ("surface_tolerance_m", "amr_eta_tol"), ("velocity_tolerance_m_s", "amr_u_tol")):
        _require(type(tree.get(source)) in (int, float) and np.isfinite(tree[source]) and tree[source] > 0,
                 "invalid AMR threshold " + source)
        settings[target] = float(tree[source])
    extended = any(settings[key] for key in ("refine_walls", "refine_bathymetry", "refine_sediment"))
    general_session = (manifest.get('schema') == 3 and
                       (manifest.get('kind') == 'on-demand-general-transport-amr' or
                        manifest.get('general_transport') is True) and
                       manifest.get('evolving_banks') is False)
    _require(not extended or (manifest.get("schema") in (2, 3) and
             (manifest.get("evolving_banks") is True or general_session)),
             "extended AMR indicators require the explicit evolving-bank session")
    if extended:
        for target, source in (("bed_tolerance_m", "amr_bed_tol"),
                               ("concentration_tolerance_kg_m3", "amr_c_tol"),
                               ("composition_tolerance", "amr_mix_tol"),
                               ("bed_change_tolerance_m", "amr_dz_tol")):
            value = tree.get(source)
            _require(type(value) in (int, float) and np.isfinite(value) and value > 0 and
                     (target != "composition_tolerance" or value <= 1),
                     "invalid extended AMR threshold " + source)
            settings[target] = float(value)
    _require(type(tree.get("subcycling")) is int and tree["subcycling"] in (0, 1), "invalid AMR subcycling switch")
    settings["subcycling"] = bool(tree["subcycling"])
    if settings['subcycling']:
        bank_mode = int(initial_data['bank_toe_erosion_mode'])
        general = general_session and bank_mode == 0
        physical_toe = (bank_mode == 2 and manifest.get('schema') in (1, 2) and
                        (manifest.get('schema') != 2 or
                         manifest.get('evolving_banks') is True))
        distributed = (bank_mode == 3 and manifest.get('schema') == 2 and
                       manifest.get('evolving_banks') is True and
                       manifest.get('bank_mode') == 3)
        _require(general or physical_toe or distributed,
                 'recursive AMR route/physics contract differs')
    recursive_bedform = False
    if settings['subcycling']:
        _recursive_adaptation_scope(initial_data)
        if general:
            _recursive_general_avalanche_scope(initial_data)
            _recursive_general_viscosity_scope(initial_data)
            recursive_bedform = _recursive_general_bedform_scope(initial_data)
        elif distributed or physical_toe:
            _recursive_bank_bedform_avalanche_scope(initial_data)
    original = manifest.get("original_controls")
    _require(settings["refine_wet_dry"] and settings["refine_hydraulics"] and
             settings["levels"] == len(packed["offsets"]) - 1 and settings["refinement_ratio"] == 2 and
             1 <= settings["wet_level"] < settings["levels"] and settings["coarsen_delay"] >= 1 and
             settings["buffer_cells"] >= 0 and isinstance(original, dict) and set(original) == set(settings) and
             all(type(original[key]) is type(value) and original[key] == value for key, value in settings.items()),
             "unsupported or changed AMR selection controls")
    owner, other = np.asarray(tree.get("face_owner")), np.asarray(tree.get("face_neighbour"))
    _require(owner.ndim == 1 and owner.dtype.kind in "iu" and other.shape == owner.shape and
             other.dtype.kind in "iu" and np.all((owner >= 0) & (owner < count)) and
             np.all((other >= -1) & (other < count)), "invalid native face connectivity")
    level = packed["node_level"]
    active = (other >= 0) & (level[owner] == np.max(level)) & (level[np.maximum(other, 0)] == np.max(level))
    pairs = np.unique(np.sort(np.column_stack((owner[active], other[active])), axis=1), axis=0)
    contract = dict(settings=settings,
        minimum_sha256=hashlib.sha256(minimum.astype("<i8").tobytes()).hexdigest(),
        rock_sha256=hashlib.sha256(rock.astype("<f8").tobytes()).hexdigest(),
        finest_pairs_sha256=hashlib.sha256(pairs.astype("<i8").tobytes()).hexdigest(),
        protected_ids=[], coarsen_stage_span_m=.020)
    if manifest.get("schema") == 2:
        contract.update(evolving_banks=True, event_rule="complete_one_level_material_homogeneous_v1")
        if extended:
            contract["extended_indicator_rule"] = "native_directed_csr_v1"
        if np.array_equal(initial_data["mode"], [9, 4]):
            contract["boundary_forecast"] = "all_open_finest_current_inlet_characteristic_v1"
    elif manifest.get("schema") == 3:
        contract.update(general_transport=True, event_rule="complete_one_level_transport_v1")
        if np.array_equal(initial_data["mode"], [9, 4]):
            contract["boundary_forecast"] = "all_open_finest_current_inlet_characteristic_v1"
        if recursive_bedform:
            contract["recursive_bedform_boundary"] = \
                "unscheduled_unsampled_mode1_mode2_finest_v1"
    if extended:
        contract["extended_indicator_rule"] = "native_directed_csr_v1"
    if contract_transform is not None:
        contract = contract_transform(contract)
    _require(_digest(contract) == manifest.get("selector_contract"), "AMR selector contract differs")
    return settings


def _inspect(folder, expected_receipt_sha256):
    root = _regular(Path(folder).expanduser(), directory=True).resolve()
    receipt_path = root / "preparation_receipt.json"
    receipt = _object(receipt_path)
    receipt_sha = _sha(receipt_path)
    _require(expected_receipt_sha256 is None or
             (_hex(expected_receipt_sha256) and receipt_sha == expected_receipt_sha256), "receipt identity differs")
    _require(receipt.get("schema") == 6 and receipt.get("variant") == VARIANT and
             type(receipt.get("native_solver_steps_performed")) is int and receipt["native_solver_steps_performed"] == 0 and
             receipt.get("initialization_CPU_only") is True and receipt.get("GPU_solver_executed") is False and
             type(receipt.get("portable")) is bool, "unsupported schema-6 receipt")
    for name in ("session_identity", "session_manifest_sha256", "runtime_build_sha256"):
        _require(_hex(receipt.get(name)), "invalid receipt " + name)
    session_root = _regular(root / "session", directory=True)
    manifest = _object(session_root / "session.json")
    _require(_sha(session_root / "session.json") == receipt["session_manifest_sha256"] and
             manifest.get("identity") == receipt["session_identity"] and
             manifest["identity"] == _digest({key: value for key, value in manifest.items() if key != "identity"}) and
             type(manifest.get("schema")) is int and manifest["schema"] in (1, 2) and
             manifest.get("kind") == "on-demand-bank-amr" and
             type(manifest.get("native_solver_steps")) is int and manifest["native_solver_steps"] == 0,
             "session identity or zero-step contract differs")
    evolving = manifest["schema"] == 2
    bank_mode = manifest.get('bank_mode', 2)
    _require(type(bank_mode) is int and bank_mode in (2, 3) and
              (bank_mode != 3 or evolving) and
              ('bank_mode' not in manifest or bank_mode == 3 or
               manifest.get('recursive_bank_bedform_avalanche') is True),
              'invalid session bank mode contract')
    _require((evolving and manifest.get("evolving_banks") is True and "global_toe_lineage" not in manifest) or
             (not evolving and "evolving_banks" not in manifest), "session evolving-interface contract differs")
    composed = manifest.get('recursive_bank_bedform_avalanche', False)
    _require(type(composed) is bool and
              (not composed or (evolving and bank_mode in (2, 3))) and
              (composed or bank_mode != 2 or 'bank_mode' not in manifest),
              'invalid session bedform-bank initializer contract')
    originals = receipt.get("original_input_files")
    _require(isinstance(originals, dict) and "amr_hierarchy.npz" in originals and
             originals == manifest.get("original_input_files"), "original-input inventory differs")
    _base_input_digest(originals)
    _require(isinstance(manifest.get("deck"), str) and Path(manifest["deck"]).is_absolute() and
             isinstance(manifest.get("runtime"), str) and
             Path(manifest["runtime"]).is_absolute(), "invalid original deck/runtime locations")
    deck = _regular(manifest["deck"])
    request = _object(root / "preparation_request.json")
    adapter=request.get('recursive_geometry_adapter')
    if adapter is None:
        _require(deck.name=='case.data' and 'case.data' in originals,
                 'invalid original deck location')
        source_deck=deck
    else:
        _require(composed and receipt.get('recursive_geometry_adapter')==adapter and
                 deck.name==adapter.get('adapted_deck_name') and
                 originals.get(deck.name)==adapter.get('adapted_deck_sha256'),
                 'recursive geometry adapter binding differs')
        source_deck=_regular(request.get('deck'))
        _require(source_deck.name=='case.data' and
                 source_deck.parent.resolve()==deck.parent.resolve() and
                 request.get('deck_sha256')==_sha(source_deck)==
                    adapter.get('source_deck_sha256'),
                 'recursive source-deck identity differs')
        _verify_recursive_geometry_adapter(source_deck,deck,adapter,bank_mode)
    for name, expected in originals.items():
        actual = "geometry.npz" if name == "original_geometry.npz" else name
        _authenticated_member(deck.parent.resolve(), actual, expected, nested=False)
    for line in deck.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) == 2 and _hex(tokens[1].lower()):
            _require(originals.get(tokens[0]) == tokens[1].lower(), "native deck dependency is unauthenticated")
    expected_request = dict(schema=1, variant=VARIANT, deck=str(source_deck), deck_sha256=_sha(source_deck),
        hierarchy_sha256=originals["amr_hierarchy.npz"], runtime_build_sha256=receipt["runtime_build_sha256"])
    if evolving:
        expected_request.update(schema=2, evolving_banks=True)
        expected_request = _bedform_preparation_request(
            expected_request, request, composed=composed, bank_mode=bank_mode)
    _require(request == expected_request,
             "preparation request differs from committed session")
    runtime_sha = receipt["runtime_build_sha256"]
    bootstrap = _package(session_root / "bootstrap", manifest.get("bootstrap_identity"),
                         originals=originals, runtime_sha=runtime_sha, evolving_banks=evolving)
    _require(bootstrap['topology']['initial_data']['bank_toe_erosion_mode'] == bank_mode,
             'session bank mode differs from native controls')
    _require(_recursive_bank_bedform_avalanche_scope(
                bootstrap['topology']['initial_data']) is composed,
             'session bedform-bank initializer differs from native controls')
    build = _object(bootstrap["root"] / "runtime_build.json")
    _require(type(build.get("schema")) is int and build["schema"] == 1 and build.get("export_only") is True and
             build.get("numerical_routines_unchanged") is True and _hex(build.get("executable_sha256")) and
             _hex(build.get("parent_executable_sha256")) and _hex(build.get("generator_sha256")),
             "runtime build provenance differs")
    executable = build.get("executable")
    _require(isinstance(executable, str) and bool(executable) and Path(executable).name == executable and
             not any(character in executable for character in ("/", "\\", ":")) and
             executable not in (".", ".."), "invalid recorded runtime executable name")
    _regular(session_root / "bootstrap_native", directory=True)
    native_tree_path = _authenticated_member(session_root / "bootstrap_native", "amr_tree_input.bin",
                                             manifest.get("tree_input_sha256"), nested=False)
    native_tree = _read_records(native_tree_path)
    settings = _selector(native_tree, bootstrap["tree"], manifest,
                         initial_data=bootstrap["topology"]["initial_data"])
    _require(np.array_equal(bootstrap["topology"]["node_ids"], np.flatnonzero(
        bootstrap["tree"]["node_level"] == settings["levels"] - 1)), "bootstrap is not the original finest mesh")
    tree_state = np.asarray(native_tree.get("state"))
    _require(tree_state.shape == (bootstrap["topology"]["initial_data"]["nstate"], len(bootstrap["tree"]["parent"])) and
             tree_state.dtype == np.float64 and np.all(tree_state[0] >= 0), "invalid original tree state")
    initial_ids = _ids(manifest.get("initial_node_ids"))
    proposal = manifest.get("initial_proposal", {})
    _require(proposal.get("catalog_identity") == bootstrap["manifest_sha256"] and
             proposal.get("selector_contract_identity") == manifest["selector_contract"] and
             proposal.get("old_operator_identity") == bootstrap["topology"]["operator_sha256"] and
             proposal.get("time_s") == 0. and proposal.get("new_leaves") == len(initial_ids) and
             proposal.get("leaf_digest") == hashlib.sha256(initial_ids.astype("<i8").tobytes()).hexdigest(),
             "initial leaf proposal differs")
    lineage = None if evolving else _lineage(manifest.get("global_toe_lineage"))
    store_root = _regular(session_root / "store", directory=True)
    store = _object(store_root / "store.json")
    store_base = dict(schema=2 if evolving else 1, kind="braids-bank-amr-operator-store",
        hierarchy_sha256=originals["amr_hierarchy.npz"], base_input_sha256=_base_input_digest(originals),
        runtime_build_sha256=runtime_sha)
    if evolving:
        store_base.update(tree_input_sha256=manifest["tree_input_sha256"],
                          face_lineage_contract="native-all-faces-owner-oriented-v1")
    else:
        store_base["global_toe_lineage"] = lineage
    _require(type(store.get("schema")) is int and store == dict(store_base, identity_sha256=_digest(store_base)),
             "operator store identity differs")
    entries, initial, seen = [], None, set()
    for path in sorted(_regular(store_root / "operators", directory=True).iterdir()):
        _regular(path, directory=True)
        row = _object(path / "entry.json")
        expected_keys = {"schema", "kind", "store_identity_sha256", "request_key", "selected_node_ids",
                         "package_identity", "package_files", "lineage"}
        if evolving:
            expected_keys |= {"face_identity_files", "object_identity"}
        _require(set(row) == expected_keys and
                 type(row["schema"]) is int and row["schema"] == (2 if evolving else 1) and
                 row["kind"] == "zero-step-bank-amr-operator" and row["store_identity_sha256"] == store["identity_sha256"] and
                 (evolving or row["package_identity"] == path.name) and
                 {item.name for item in path.iterdir()} == {"entry.json", "package"} | ({"face_identity"} if evolving else set()),
                 "invalid operator store entry")
        selected = _ids(row["selected_node_ids"])
        key = _digest(dict(store_identity_sha256=store["identity_sha256"], selected_node_ids=selected.tolist()))
        _require(row["request_key"] == key and key not in seen, "duplicate or changed operator request identity")
        seen.add(key)
        package = _package(path / "package", row["package_identity"], originals=originals,
                           runtime_sha=runtime_sha, evolving_banks=evolving)
        topology = package["topology"]
        _require(row["package_files"] == package["inventory"] and np.array_equal(topology["node_ids"], selected) and
                 all(np.array_equal(package["tree"][name], value) for name, value in bootstrap["tree"].items()),
                 "operator files/selection/hierarchy differ")
        if evolving:
            _all_face_lineage(path, row, package, manifest["tree_input_sha256"])
        else:
            local = _lineage(row["lineage"], local=True)
            _require([{key: value for key, value in item.items() if key != "local_face"} for item in local] == lineage,
                     "operator global physical-toe lineage differs")
        data = topology["initial_data"]
        base = bootstrap["topology"]["initial_data"]
        global_arrays = _GLOBAL_ARRAYS + (TABLE_KEYS if "native_section_seed_node" in base else ())
        _require(set(data) == set(base) and all(np.array_equal(data[key], value) for key, value in base.items()
                 if key not in ("nc", "nf") and np.asarray(value).shape == ()) and
                 all(np.array_equal(data[key], base[key]) for key in global_arrays),
                 "operator changes global physics or boundary controls")
        _require(np.array_equal(data["rock"], native_tree["rock"][selected]) and
                 np.array_equal(data["state"], native_tree["state"][:, selected]),
                 "operator changes original cold tree state")
        if not evolving:
            _require(np.array_equal(np.sort([item["local_face"] for item in local]), np.flatnonzero(data["is_toe_face"])),
                     "operator physical-toe face set differs")
            for item in local:
                face = item["local_face"]
                bank, toe = int(data["candidate_bank"][face]), int(data["candidate_toe"][face])
                _require(0 <= bank < len(selected) and 0 <= toe < len(selected) and
                         int(selected[bank]) == item["bank_node"] and int(selected[toe]) == item["toe_node"] and
                         np.array_equal(data["mid"][:, face], item["midpoint_xy_m"]) and data["length"][face] == item["length_m"],
                         "operator local/global physical-toe geometry differs")
        _require(package["receipt"].get("runtime_executable_sha256") == build["executable_sha256"] and
                 package["receipt"].get("native_cpu_parent_sha256") == build["parent_executable_sha256"] and
                 package["receipt"].get("exporter_source_sha256") == build["generator_sha256"],
                 "operator runtime provenance differs")
        entries.append(dict(folder=path, request_key=key, package_identity=row["package_identity"],
                            entry_sha256=_sha(path / "entry.json"), selected_ids=selected))
        if np.array_equal(selected, initial_ids):
            initial = package
    _require(initial is not None, "initial selected operator is absent from store")
    data = initial["topology"]["initial_data"]
    return dict(root=root, variant=VARIANT, receipt=receipt, receipt_sha256=receipt_sha,
        session_identity=manifest["identity"], session_manifest=manifest, session_root=session_root,
        selector_contract=manifest["selector_contract"], amr_settings=settings,
        original_deck=deck, runtime_location=Path(manifest["runtime"]), runtime_build=build,
        runtime_build_sha256=runtime_sha, original_input_files=dict(originals),
        bootstrap_index=bootstrap, initial_index=initial, initial_topology=initial["topology"], initial_data=data,
        store_identity=store["identity_sha256"], entries=entries,
        evolving_banks=evolving, bank_mode=bank_mode,
        checkpoint_schema=('recursive_bank_amr_v1' if settings['subcycling'] else
                           4 if bank_mode == 3 else (3 if settings["refine_sediment"] else (2 if evolving else 1))),
        nc=int(data["nc"]), nfrac=int(data["nfrac"]), nbed_layers=int(data["nbed_layers"]),
        time_s=float(data["time"]), end_s=float(data["endtime"]), output_every_s=float(data["output_dt"]),
        dtmax_s=float(data["dtmax"]), cfl=float(data["cfl_target"]))


def inspect_gpu_on_demand_prepared(folder: str | Path, *, expected_receipt_sha256: str | None = None) -> dict:
    """Authenticate the cold session/store and return host metadata for launch.

    ``receipt_sha256`` identifies the schema-6 preparation envelope;
    ``session_identity`` is the numerical input/catalog identity in checkpoints.
    No recorded compiler, reference checkout or runtime executable is opened.
    """
    try:
        receipt = _object(filesystem_path(Path(folder).expanduser()) / "preparation_receipt.json")
        if receipt.get("variant") == "on_demand_transport_amr_ssp33":
            from .gpu_general_on_demand_io import inspect_prepared
            return inspect_prepared(folder, expected_receipt_sha256=expected_receipt_sha256)
        return _inspect(folder, expected_receipt_sha256)
    except (OSError, KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError("Cannot inspect on-demand GPU preparation: " + str(exc)) from exc
