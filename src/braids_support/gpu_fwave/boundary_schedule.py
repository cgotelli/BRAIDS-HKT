"""Isolated resident FP64 boundary-table evaluation, not coupled integration.

Four immutable SSPRK43 records are evaluated at t, t+dt/2, t+dt, t+dt/2.
This does not solve live boundary sections, choose a backend, transport suspended
sediment, or relax any existing coupled solver/graph guard. The host evaluator
is a readable comparison implementation, not the independent Fortran oracle.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np

OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false')
STAGES = (0., .5, 1., .5)
INLETS = (1, 6, 7, 8, 9)
NONFINITE = 1
POSITIVE_FEED_WITHOUT_DISCHARGE = 2
INVALID_CLOCK = 4
FAMILIES = (
    ('nhydro_series', 'hydro_series_', ('time', 'q', 'stage')),
    ('nseries', 'series_', ('time', 'concentration', 'bed_input')),
    ('nfraction_series', 'fraction_series_', ('time', 'concentration', 'bed_input')),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _fp64(value, shape, name):
    value = np.array(value, dtype=np.float64, order='C', copy=True)
    require(value.shape == shape and np.isfinite(value).all(), 'Invalid finite shape: '+name)
    return value


def _integer(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in 'iu' and
        np.all((value >= 0) & (value <= np.iinfo(np.int32).max)), 'Invalid int32 '+name)
    return np.array(value, dtype=np.int32, order='C', copy=True)


def validate_tables(data):
    """Return immutable copied arrays with native table dimensions retained.

    Missing zero-count tables become one zero padding row. Active native rows
    alone define forcing; explicit supplied padding must also be finite. This
    validates the schedule component, not the native deck's version/physics.
    """
    mode = np.asarray(data['mode']); require(mode.ndim == 1, 'One-dimensional mode required')
    nb = mode.size; mode = _integer(mode, (nb,), 'mode')
    require(np.isin(mode, np.arange(1, 10)).all(), 'Unknown native boundary mode')
    for key in ('section_control', 'sampled'):
        require(not np.any(data.get(key, 0)), 'Live sections are outside schedule-only scope: '+key)
    forcing = np.asarray(data['forcing'])
    require(forcing.shape in ((2, nb), (6, nb)), 'forcing must be (2,nb) or (6,nb)')
    forcing = _fp64(forcing[:2], (2, nb), 'forcing')
    require(np.all(forcing[0] >= 0), 'Negative prescribed discharge')
    defaults = np.asarray(data['fraction_concentration'])
    require(defaults.ndim == 2 and defaults.shape[1] == nb and defaults.shape[0] > 0,
        'Fraction defaults must be (nfrac,nb), nfrac > 0')
    ng = defaults.shape[0]
    arrays = dict(mode=mode, forcing=forcing)
    for key in ('fraction_concentration', 'fraction_bed_input'):
        arrays[key] = _fp64(data[key], (ng, nb), key)
        require(np.all(arrays[key] >= 0), 'Negative fraction default: '+key)
    inlet = np.isin(mode, INLETS)
    require(np.all(forcing[0, ~inlet] == 0) and
        np.all(arrays['fraction_concentration'][:, ~inlet] == 0) and
        np.all(arrays['fraction_bed_input'][:, ~inlet] == 0), 'Outlet cannot prescribe discharge/sediment')
    for count_key, prefix, suffixes in FAMILIES:
        shape = (ng, nb) if count_key == 'nfraction_series' else (nb,)
        count = _integer(data.get(count_key, np.zeros(shape, dtype=np.int32)), shape, count_key)
        require(not np.any(count == 1), 'Series must have zero or at least two rows: '+count_key)
        allowed = inlet | (mode == 2) if count_key == 'nhydro_series' else inlet
        require(not np.any(count[..., ~allowed]), 'Series is incompatible with boundary mode: '+count_key)
        arrays[count_key] = count
        size = max(1, int(count.max(initial=0)))
        time_key = prefix+'time'
        if time_key in data:
            raw = np.asarray(data[time_key]); require(raw.ndim == len(shape)+1 and raw.shape[1:] == shape
                and raw.shape[0] >= size, 'Table dimensions/count mismatch: '+time_key)
            size = raw.shape[0]
        for suffix in suffixes:
            key = prefix+suffix
            require(key in data or not np.any(count), 'Missing active table: '+key)
            arrays[key] = _fp64(data.get(key, np.zeros((size, *shape))), (size, *shape), key)
        for index in np.ndindex(shape):
            n = int(count[index])
            if not n: continue
            selection = (slice(0, n), *index)
            times = arrays[time_key][selection]
            require(np.all(times >= 0) and np.all(np.diff(times) > 0), 'Active times must strictly increase: '+time_key)
            for suffix in suffixes:
                if suffix == 'stage': continue  # Absolute elevation can be negative.
                require(np.all(arrays[prefix+suffix][selection] >= 0), 'Negative active table: '+prefix+suffix)
        if count_key == 'nhydro_series':
            for b in np.flatnonzero(mode == 2):
                require(np.all(arrays[prefix+'q'][:count[b], b] == 0), 'Stage outlet hydrograph cannot prescribe Q')
    # Immutable bytes backing also prevents callers re-enabling WRITEABLE.
    arrays = {key: np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
        for key, value in arrays.items()}
    return MappingProxyType(arrays)


def _interpolate(times, values, count, current, default):
    if not count: return float(default)
    if current <= times[0]: return float(values[0])
    if current >= times[count-1]: return float(values[count-1])
    k = int(np.searchsorted(times[:count], current, side='left'))
    weight = (current-times[k-1])/(times[k]-times[k-1])
    return float((1.-weight)*values[k-1]+weight*values[k])


def stage_times(current, dt):
    require(np.isscalar(current) and np.isscalar(dt) and np.isfinite([current, dt]).all()
        and dt > 0 and np.isfinite(current+dt) and current+dt > current,
        'Finite advancing clock required')
    return np.asarray([current+fraction*dt for fraction in STAGES], dtype=np.float64)


def evaluate_host(data, current, dt, *, validated=False):
    """Read-only NumPy comparison, independent from CUDA control flow."""
    a = data if validated else validate_tables(data)
    times = stage_times(current, dt); nb = len(a['mode']); ng = a['fraction_bed_input'].shape[0]
    result = dict(stage_time=times, hydraulic=np.empty((4, 2, nb)),
        concentration=np.empty((4, ng, nb)), feed=np.empty((4, ng, nb)), status=np.zeros((4, nb), np.int32))
    for s, time in enumerate(times):
        for b in range(nb):
            for j, suffix in enumerate(('q', 'stage')):
                result['hydraulic'][s, j, b] = _interpolate(a['hydro_series_time'][:, b],
                    a['hydro_series_'+suffix][:, b], a['nhydro_series'][b], time, a['forcing'][j, b])
            for g in range(ng):
                for key, suffix in (('concentration', 'concentration'), ('feed', 'bed_input')):
                    value = a['fraction_'+suffix][g, b]
                    if g == 0:
                        value = _interpolate(a['series_time'][:, b], a['series_'+suffix][:, b], a['nseries'][b], time, value)
                    result[key][s, g, b] = _interpolate(a['fraction_series_time'][:, g, b],
                        a['fraction_series_'+suffix][:, g, b], a['nfraction_series'][g, b], time, value)
            if a['mode'][b] in INLETS and result['hydraulic'][s, 0, b] <= 0 and np.any(result['feed'][s, :, b] > 0):
                result['status'][s, b] |= POSITIVE_FEED_WITHOUT_DISCHARGE
            if not all(np.isfinite(result[key][s, ..., b]).all() for key in ('hydraulic', 'concentration', 'feed')):
                result['status'][s, b] |= NONFINITE
    return result


def array_identity(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


class ResidentBoundarySchedule:
    """Explicit GPU component; tables uploaded once, clocks uploaded per trial.

    evaluate() without arguments only launches kernels and is capture-compatible.
    set_clock() remains outside capture. No field/result transfer is implicit.
    Public device outputs are stable; callers must check status before using them.
    """
    def __init__(self, data, *, source_path=None):
        host = validate_tables(data)
        import cupy as cp
        self.cp = cp; self.host = host; self.nb = len(host['mode']); self.ng = host['fraction_bed_input'].shape[0]
        self._arrays = {key: cp.asarray(value) for key, value in host.items()}
        self.clock = cp.asarray([0., 1.], dtype=cp.float64)
        self.outputs = dict(stage_time=cp.empty(4, dtype=cp.float64), hydraulic=cp.empty((4, 2, self.nb)),
            concentration=cp.empty((4, self.ng, self.nb)), feed=cp.empty((4, self.ng, self.nb)),
            status=cp.empty((4, self.nb), dtype=cp.int32))
        from braids_core.windows_paths import filesystem_path
        path = filesystem_path(source_path) if source_path else filesystem_path(__file__).with_suffix('.cu')
        self._source = path.resolve(); self._source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self._identities = {key: array_identity(value) for key, value in host.items()}
        self.module = cp.RawModule(code=path.read_text(), options=OPTIONS, name_expressions=('bs_evaluate',))
        self.kernel = self.module.get_function('bs_evaluate')
        self._storage = self._storage_identity()

    def _storage_identity(self):
        return tuple((key, value.data.ptr, value.shape, value.dtype.str) for key, value in
            [*self._arrays.items(), ('clock', self.clock), *self.outputs.items()])

    def set_clock(self, current, dt):
        stage_times(current, dt)
        self.clock.set(np.array([current, dt], dtype=np.float64))

    def evaluate(self, current=None, dt=None):
        require((current is None) == (dt is None), 'Provide both clock values or use resident clock')
        require(self._storage_identity() == self._storage, 'Captured schedule storage changed')
        if current is not None: self.set_clock(current, dt)
        a = self._arrays; o = self.outputs
        keys = ('mode', 'forcing', 'fraction_concentration', 'fraction_bed_input',
            'nhydro_series', 'nseries', 'nfraction_series',
            'hydro_series_time', 'hydro_series_q', 'hydro_series_stage',
            'series_time', 'series_concentration', 'series_bed_input',
            'fraction_series_time', 'fraction_series_concentration', 'fraction_series_bed_input')
        count = 4*max(1, self.nb)
        self.kernel(((count+127)//128,), (128,), (np.int32(self.nb), np.int32(self.ng), self.clock,
            *(a[key] for key in keys), o['stage_time'], o['hydraulic'], o['concentration'], o['feed'], o['status']))
        return self.outputs

    def verify_tables(self):
        """Explicit audit transfer, never called by evaluate()."""
        require(all(array_identity(self.cp.asnumpy(value)) == self._identities[key]
            for key, value in self._arrays.items()), 'Immutable device table bytes changed')
        require(hashlib.sha256(self._source.read_bytes()).hexdigest() == self._source_sha, 'CUDA source changed')
        return True

    def receipt(self):
        return dict(backend='gpu', automatic_CPU_fallback=False, precision='FP64',
            source=str(self._source), source_sha256=self._source_sha, requested_cuda_options=list(OPTIONS),
            tables=self._identities, stage_fractions=list(STAGES), tables_uploaded_once=True,
            status_bits=dict(nonfinite=NONFINITE, positive_feed_without_discharge=POSITIVE_FEED_WITHOUT_DISCHARGE,
                invalid_clock=INVALID_CLOCK), live_section_roots=False, coupled_integration_qualified=False,
            secondary_flow_required=False, performance_qualified=False)
