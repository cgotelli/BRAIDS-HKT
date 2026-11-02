"""NumPy-only authentication of selected-AMR live boundary metadata.

This validates immutable inputs, not a hydraulic solve. It is shared by the
standalone worker and QGIS without importing either execution environment.
"""
from __future__ import annotations

import hashlib
import numpy as np

FAMILIES = (('nhydro_series', 'hydro_series_', ('time', 'q', 'stage')),
            ('nseries', 'series_', ('time', 'concentration', 'bed_input')),
            ('nfraction_series', 'fraction_series_', ('time', 'concentration', 'bed_input')))
TABLE_KEYS = tuple(prefix + suffix for _, prefix, suffixes in FAMILIES for suffix in suffixes)
METADATA_KEYS = frozenset(('nb', 'nfrac', 'deck_version', 'nhydro_series', 'nseries',
    'mode', 'section_control', 'nfraction_series', 'native_section_seed_node',
    'native_boundary_seed_position', 'section_energy_slope', *TABLE_KEYS))


def _require(value, message):
    if not value:
        raise ValueError('Live AMR boundary metadata: ' + message)


def _array(data, key, shape, dtype):
    value = np.asarray(data[key])
    _require(value.shape == shape and value.dtype == np.dtype(dtype) and
             np.isfinite(value).all(), 'invalid ' + key)
    return value


def _active_knots(data):
    nb, ng = int(data['nb']), int(data['nfrac'])
    result = []
    for count_key, prefix, _ in FAMILIES:
        shape = (ng, nb) if count_key == 'nfraction_series' else (nb,)
        counts = _array(data, count_key, shape, 'int32')
        times = np.asarray(data[prefix + 'time'])
        _require(times.ndim == len(shape) + 1 and times.shape[1:] == shape and
                 times.dtype == np.float64 and np.isfinite(times).all() and
                 np.all((counts == 0) | (counts >= 2)) and np.all(counts <= times.shape[0]),
                 'invalid active table counts: ' + count_key)
        for index in np.ndindex(shape):
            values = times[(slice(0, int(counts[index])), *index)]
            _require(np.all(values >= 0) and np.all(np.diff(values) > 0),
                     'active times must strictly increase: ' + prefix)
            result.extend(values.tolist())
    return np.unique(np.asarray(result, dtype='<f8'))


def boundary_event_identity(data):
    """Exact BoundaryEvents identity, using active knots and ignoring padding."""
    return hashlib.sha256(b'BRAIDS boundary event clock v1\0' +
                          _active_knots(data).astype('<f8').tobytes()).hexdigest()


def validate_live_boundary_metadata(data, metadata=None):
    """Return a detached mapping with native seeds and explicit 0-based aliases.

    Supported input laws are stage outlet 2, critical outlet 4 and surveyed inlet
    9. Numerical bridge admission remains separately responsible for its exact
    law/control combination. No extra physical process is admitted here.
    """
    merged = dict(data)
    if metadata is not None:
        _require(set(metadata) == METADATA_KEYS, 'unexpected normal_metadata.bin records')
        for key, value in metadata.items():
            _require(key not in merged or np.array_equal(merged[key], value),
                     'native core/sidecar conflict: ' + key)
            merged[key] = value
    _require(METADATA_KEYS <= set(merged), 'missing normal_metadata.bin records')
    for key in ('nb', 'nfrac', 'deck_version', 'nc', 'nf'):
        _require(type(merged[key]) is int and merged[key] > 0, 'invalid ' + key)
    nb, ng, nc, nf = (merged[key] for key in ('nb', 'nfrac', 'nc', 'nf'))
    mode = _array(merged, 'mode', (nb,), 'int32')
    control = _array(merged, 'section_control', (nb,), 'int32')
    _require(np.isin(mode, (2, 4, 9)).all() and np.isin(control[mode == 9], (0, 1, 2)).all()
             and not np.any(control[mode != 9]), 'unsupported boundary law/control')
    inlet = mode == 9
    slope = _array(merged, 'section_energy_slope', (nb,), 'float64')
    _require(np.all(slope[(mode == 9) & (control == 1)] > 0), 'normal depth requires positive energy slope')
    forcing = _array(merged, 'forcing', (6, nb), 'float64')
    concentration = _array(merged, 'fraction_concentration', (ng, nb), 'float64')
    feed = _array(merged, 'fraction_bed_input', (ng, nb), 'float64')
    _require(np.all(forcing[0] >= 0) and not np.any(forcing[0, ~inlet]) and
             not np.any(concentration) and np.all(feed >= 0) and not np.any(feed[:, ~inlet]),
             'invalid discharge, unsupported suspended load or outlet sediment feed')
    knots = _active_knots(merged)
    for count_key, prefix, suffixes in FAMILIES:
        count = np.asarray(merged[count_key])
        times = np.asarray(merged[prefix + 'time'])
        allowed = inlet | (mode == 2) if count_key == 'nhydro_series' else inlet
        _require(not np.any(count[..., ~allowed]), 'schedule incompatible with boundary law')
        for suffix in suffixes:
            table = _array(merged, prefix + suffix, times.shape, 'float64')
            if suffix == 'concentration':
                _require(not np.any(table), 'suspended schedules are not supported')
            for index in np.ndindex(count.shape):
                active = table[(slice(0, int(count[index])), *index)]
                if suffix != 'stage':
                    _require(np.all(active >= 0), 'negative active ' + prefix + suffix)
        if count_key == 'nhydro_series':
            for b in np.flatnonzero(mode == 2):
                _require(not np.any(merged['hydro_series_q'][:count[b], b]),
                         'stage outlet schedule cannot prescribe discharge')

    # Piecewise linear forcing is checked at its entire knot union. Between
    # knots nonnegative Q can vanish only on a zero-Q interval or an endpoint.
    checks = np.unique(np.r_[0., float(merged.get('endtime', 0.)), knots])
    def interpolate(prefix, suffix, count, index, default):
        if not count:
            return np.full(checks.shape, default, dtype=np.float64)
        select = (slice(0, int(count)), *index)
        return np.interp(checks, merged[prefix + 'time'][select], merged[prefix + suffix][select])
    for b in np.flatnonzero(inlet):
        discharge = interpolate('hydro_series_', 'q', merged['nhydro_series'][b], (b,), forcing[0, b])
        for g in range(ng):
            values = np.full(checks.shape, feed[g, b])
            if g == 0:
                values = interpolate('series_', 'bed_input', merged['nseries'][b], (b,), feed[g, b])
            count = merged['nfraction_series'][g, b]
            if count:
                values = interpolate('fraction_series_', 'bed_input', count, (g, b), feed[g, b])
            _require(not np.any((discharge <= 0) & (values > 0)), 'positive sediment feed without discharge')

    owner = _array(merged, 'owner', (nf,), 'int32')
    other = _array(merged, 'neighbour', (nf,), 'int32')
    bid = _array(merged, 'bid', (nf,), 'int32')
    faces = np.asarray(merged['boundary_faces'])
    offset = _array(merged, 'boundary_offset', (nb + 1,), 'int32')
    _require(faces.ndim == 1 and faces.dtype == np.int32 and np.all((faces >= 0) & (faces < nf)) and
             len(np.unique(faces)) == len(faces) and offset[0] == 0 and offset[-1] == len(faces) and
             np.all(np.diff(offset) > 0) and np.all((owner >= 0) & (owner < nc)) and
             np.all((bid >= -1) & (bid < nb)) and np.array_equal(np.sort(faces), np.flatnonzero(bid >= 0)),
             'invalid or incomplete boundary CSR')
    sampled = _array(merged, 'sampled', (nf,), 'int32')
    expected_sampled = np.zeros(nf, dtype=np.int32)
    seeds = _array(merged, 'native_section_seed_node', (nb,), 'int32')
    face_seeds = _array(merged, 'native_boundary_seed_position', (nb,), 'int32')
    for b in range(nb):
        ff = faces[offset[b]:offset[b + 1]]
        _require(np.all(bid[ff] == b) and np.all(other[ff] == -1),
                 'boundary faces must belong to the correct exterior section')
        if mode[b] == 9:
            _require(merged.get('production_section_geometry') == 1 and
                     2 <= seeds[b] <= len(ff) and offset[b] < face_seeds[b] <= offset[b + 1],
                     'surveyed section seed is not a selected interior node/face')
            expected_sampled[ff] = 1
        else:
            _require(seeds[b] == 0 and face_seeds[b] == 0, 'unselected section has native seeds')
    _require(np.array_equal(sampled, expected_sampled), 'sampled faces must equal the surveyed inlet faces')
    for raw, alias in (('native_section_seed_node', 'section_seed_node'),
                       ('native_boundary_seed_position', 'boundary_seed_position')):
        value = np.ascontiguousarray(merged[raw] - np.int32(1))
        _require(alias not in merged or np.array_equal(merged[alias], value), 'conflicting converted seed alias')
        merged[alias] = value
    return merged
