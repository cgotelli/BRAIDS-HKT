"""Isolated FP64 device-controlled critical-INLET section root, not traces.

The SHA-pinned normal metric prefix is reused literally in the compiled source.
The independent critical kernel retains native stationary-knot construction,
increasing-interval/exact-knot candidate order and minimum-specific-energy rule.
No CPU numerical fallback, production guard lift or fixed 4096-face limit.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import live_normal_section as normal
from . import live_section_metrics as geometry_module
from .time_integration import OPTIONS

NORMAL_PY_SHA = 'b3615c811a1d002da4a9bfdecab32ab56e3353e7e737ed31c0a914e10e030b1f'
NORMAL_CUDA_SHA = 'c918345899bcf178a8cafd9beca2e06c856e96383fef7a734167e7333f831dec'
GEOMETRY_SHA = normal.GEOMETRY_SHA
NAMES = ('ln_inputs', 'ln_metrics', 'lc_solve')
STATUS_BITS = dict(invalid_live_input=1, inadmissible_current_profile=2,
                   no_critical_root_found=8, invalid_arithmetic=16, workspace_capacity_failure=32)
freeze, require, identity = normal.freeze, normal.require, normal.identity


def validate_controls(geometry, controls):
    nb = len(geometry['offsets'])-1
    result = {}
    for key in ('mode', 'section_control', 'energy_slope', 'seed_node'):
        value = np.asarray(controls[key])
        dtype = np.float64 if key == 'energy_slope' else np.int32
        require(value.shape == (nb,) and value.dtype == dtype and value.flags.c_contiguous,
                'Canonical control array required: '+key)
        result[key] = freeze(value)
    require(np.all(result['mode'] == 9) and np.all(result['section_control'] == 2),
            'Only mode9 CRITICAL inlet section control2 is implemented')
    require(np.isfinite(result['energy_slope']).all(), 'Finite unused energy slope required; zero is allowed')
    count = np.diff(geometry['offsets'])
    require(np.all((result['seed_node'] >= 1)&(result['seed_node'] < count)),
            'Zero-based initially interior seed_node metadata required')
    return MappingProxyType(result)


def validate_gravity(gravity):
    value = np.asarray(gravity)
    require(value.shape == () and value.dtype == np.float64 and np.isfinite(value) and value > 0,
            'Explicit finite positive FP64 gravity scalar required')
    return float(value)


def workspace_layout(geometry, *, maximum_workspace_bytes=512*1024*1024):
    require(type(maximum_workspace_bytes) is int and maximum_workspace_bytes > 0,
            'Explicit positive integer workspace byte bound required')
    counts = np.diff(geometry['offsets']).astype(np.int64)
    # Match native critical capacities; round levels upward for parallel sort.
    native_levels = 3*counts+6
    capacity = np.asarray([1 << (int(n)-1).bit_length() for n in native_levels], dtype=np.int64)
    root_capacity = 6*counts+12
    level_offsets = np.r_[0, np.cumsum(capacity)]
    root_offsets = np.r_[0, np.cumsum(root_capacity)]
    nbf, nb = len(geometry['faces']), len(counts)
    estimated = 48*int(level_offsets[-1])+24*int(root_offsets[-1])+24*nbf+136*nb+4
    require(max(int(capacity.max()), int(root_capacity.max()), int(level_offsets[-1]), int(root_offsets[-1])) < np.iinfo(np.int32).max,
            'Critical workspace exceeds canonical int32 addressing')
    require(estimated <= maximum_workspace_bytes,
            'Declared GPU workspace byte capacity exceeded; no CPU fallback')
    return dict(level_capacity=freeze(capacity.astype(np.int32)),
        root_capacity=freeze(root_capacity.astype(np.int32)),
        level_offsets=freeze(level_offsets.astype(np.int32)), root_offsets=freeze(root_offsets.astype(np.int32)),
        estimated_dynamic_bytes=estimated, maximum_workspace_bytes=maximum_workspace_bytes)


def metric_prefix(path):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == NORMAL_CUDA_SHA, 'Qualified normal CUDA dependency changed')
    marker = b'extern "C" __global__ void ln_solve('
    require(raw.count(marker) == 1, 'Expected one normal root delimiter after qualified metric prefix')
    prefix = raw.split(marker, 1)[0]
    return prefix, hashlib.sha256(prefix).hexdigest()


class ResidentCriticalSection(normal.ResidentNormalSection):
    """Fixed-pointer critical root with independent diagnostic metrics().

    Values of bed/Q/optional metric_stage may change in-place; all other arrays
    and gravity are immutable. Public outputs/workspaces must not be edited by
    callers. check_storage() is required before an externally captured replay.
    verify_geometry() explicitly audits immutable bytes/source identities, with
    host downloads outside the hot path. No normal constructor is invoked.
    """
    def __init__(self, geometry, controls, bed_device, discharge_device, *, gravity,
                 metric_stage_device=None, source_path=None, metric_source_path=None,
                 maximum_workspace_bytes=512*1024*1024):
        require(hashlib.sha256(Path(normal.__file__).read_bytes()).hexdigest() == NORMAL_PY_SHA,
                'Qualified normal Python helper changed')
        require(hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest() == GEOMETRY_SHA,
                'Qualified immutable geometry helper changed')
        self.host = geometry_module.validate_geometry(geometry)
        self.controls = validate_controls(self.host, controls)
        self.gravity = validate_gravity(gravity)
        self._gravity_identity = self.gravity.hex()
        self.topology = geometry_module.topology(self.host)
        self.layout = workspace_layout(self.host, maximum_workspace_bytes=maximum_workspace_bytes)
        self.nc, self.nb = len(self.host['manning']), len(self.host['offsets'])-1
        self.nbf, self.ng = len(self.host['faces']), len(self.topology['group_manning'])
        self._launch_identity = (self.nc, self.nb, self.nbf, self.ng)
        import cupy as cp
        self.cp, self.device_id = cp, cp.cuda.Device().id
        self.bed, self.discharge, self.metric_stage = bed_device, discharge_device, metric_stage_device
        for name, value, shape in [('bed', self.bed, (self.nc,)), ('discharge', self.discharge, (self.nb,))]+(
            [('metric_stage', self.metric_stage, (self.nb,))] if self.metric_stage is not None else []):
            require(isinstance(value, cp.ndarray) and value.shape == shape and value.dtype == cp.float64
                and value.flags.c_contiguous and value.device.id == self.device_id,
                'Current-device resident contiguous FP64 input required: '+name)
        host = dict(self.host, **self.controls, **self.topology)
        host['face_group'] = freeze(np.repeat(np.arange(self.ng, dtype=np.int32), np.diff(self.topology['group_offsets'])))
        for key in ('level_capacity', 'root_capacity', 'level_offsets', 'root_offsets'):
            host[key] = self.layout[key]
        self._host_arrays = MappingProxyType(host)
        self._arrays = {key: cp.asarray(value) for key, value in host.items()}
        self.outputs = self._metric_outputs()
        self.outputs.update(stage=cp.zeros(self.nb), residual=cp.empty(self.nb),
            level_offsets=self._arrays['level_offsets'], root_offsets=self._arrays['root_offsets'])
        for key in ('cache_hit', 'selected_branch', 'selected_kind', 'selected_index',
                    'nlevels', 'nterrain', 'nroots', 'bisections', 'metric_evaluations'):
            self.outputs[key] = cp.full(self.nb, -1 if key.startswith('selected_') else 0, dtype=cp.int32)
        nl, nr = int(self.layout['level_offsets'][-1]), int(self.layout['root_offsets'][-1])
        for key in ('levels', 'interval_roots', 'interval_lo', 'interval_hi'):
            self.outputs[key] = cp.empty(nl)
        for key in ('candidate_valid', 'knot_valid', 'interval_reason', 'interval_iterations'):
            self.outputs[key] = cp.empty(nl, dtype=cp.int32)
        for key in ('roots', 'energies'):
            self.outputs[key] = cp.empty(nr)
        for key in ('root_kind', 'root_branch'):
            self.outputs[key] = cp.empty(nr, dtype=cp.int32)
        self.metric_outputs = self._metric_outputs()
        self.cache = dict(valid=cp.zeros(self.nb, dtype=cp.int32), bed=cp.empty(self.nbf), discharge=cp.empty(self.nb))
        self.work = dict(global_status=cp.zeros(1, dtype=cp.int32))
        self._source = (Path(source_path) if source_path else Path(__file__).with_suffix('.cu')).resolve()
        self._metric_source = (Path(metric_source_path) if metric_source_path else Path(normal.__file__).with_suffix('.cu')).resolve()
        prefix, self._metric_prefix_sha = metric_prefix(self._metric_source)
        raw = self._source.read_bytes()
        self._source_sha = hashlib.sha256(raw).hexdigest()
        self._compiled_source = prefix+b'\n'+raw
        self._compiled_source_sha = hashlib.sha256(self._compiled_source).hexdigest()
        self._python_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.module = cp.RawModule(code=self._compiled_source.decode('utf-8'), options=OPTIONS, name_expressions=NAMES)
        self.kernels = {key: self.module.get_function(key) for key in NAMES}
        self._array_identity = {key: identity(value) for key, value in host.items()}
        self._storage = self._storage_identity()

    def check_storage(self):
        require(self.cp.cuda.Device().id == self.device_id, 'Current CUDA device differs from captured critical section')
        require(isinstance(self.gravity, float) and self.gravity.hex() == self._gravity_identity,
                'Captured critical gravity changed')
        require((self.nc, self.nb, self.nbf, self.ng) == self._launch_identity,
                'Captured critical launch dimensions changed')
        require(self._storage_identity() == self._storage, 'Captured critical-section storage changed')
        return True

    def solve(self):
        self.check_storage()
        a, o, c = self._arrays, self.outputs, self.cache
        self._validate_live(self.discharge, o, True)
        self.kernels['lc_solve']((self.nb,), (128,),
            (a['owner'], a['faces'], a['offsets'], a['face_group'], a['group_manning'], a['length'],
             a['left_offset'], a['right_offset'], self.bed, self.discharge, self.work['global_status'], np.float64(self.gravity),
             a['level_offsets'], a['level_capacity'], a['root_offsets'], a['root_capacity'], np.int32(self.nb),
             o['nodes'], c['valid'], c['bed'], c['discharge'],
             o['stage'], o['area'], o['top_width'], o['conveyance'], o['residual'],
             o['status'], o['cache_hit'], o['selected_branch'], o['selected_kind'], o['selected_index'],
             o['nlevels'], o['nterrain'], o['nroots'], o['bisections'], o['metric_evaluations'],
             o['levels'], o['interval_roots'], o['interval_lo'], o['interval_hi'], o['roots'], o['energies'],
             o['candidate_valid'], o['knot_valid'], o['interval_reason'], o['interval_iterations'], o['root_kind'], o['root_branch']))
        return o

    def verify_geometry(self):
        self.check_storage()
        require(all(identity(self.cp.asnumpy(value)) == self._array_identity[key]
            for key, value in self._arrays.items()), 'Immutable critical geometry/control bytes changed')
        prefix, digest = metric_prefix(self._metric_source)
        require(digest == self._metric_prefix_sha
            and hashlib.sha256(self._source.read_bytes()).hexdigest() == self._source_sha
            and hashlib.sha256(prefix+b'\n'+self._source.read_bytes()).hexdigest() == self._compiled_source_sha
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == self._python_sha
            and hashlib.sha256(Path(normal.__file__).read_bytes()).hexdigest() == NORMAL_PY_SHA
            and hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest() == GEOMETRY_SHA,
            'Executed critical section source identity changed')
        return True

    def receipt(self):
        return dict(backend='gpu', automatic_CPU_fallback=False, precision='FP64', device_id=self.device_id,
            source=str(self._source), source_sha256=self._source_sha, python_sha256=self._python_sha,
            metric_source=str(self._metric_source), metric_source_sha256=NORMAL_CUDA_SHA,
            metric_prefix_sha256=self._metric_prefix_sha, compiled_source_sha256=self._compiled_source_sha,
            normal_python_sha256=NORMAL_PY_SHA, geometry_helper_sha256=GEOMETRY_SHA,
            requested_cuda_options=list(OPTIONS), immutable_inputs=self._array_identity,
            gravity=self.gravity, critical_inlet_only=True, mode9_trace_implemented=False,
            critical_outlet_implemented=False, production_guard_changed=False, block_size=128,
            root_control='Device cooperative block; no host per-iteration launches',
            metric_helper='Byte-identical qualified normal CUDA prefix; independent metric outputs/cache',
            sorting='Cooperative bitonic padded global workspace; exact duplicate tests',
            maximum_faces_per_section=int(np.diff(self.host['offsets']).max()),
            estimated_dynamic_bytes=self.layout['estimated_dynamic_bytes'], maximum_workspace_bytes=self.layout['maximum_workspace_bytes'],
            exact_cache_keys=['discharge equality', 'every section face owner bed equality'],
            immutable_cache_dependencies=['geometry', 'gravity', 'mode9/control2', 'initial interior seed', 'positive Manning'],
            current_device_guard=True, raw_graph_replay_requires_check_storage=True,
            status_bits=STATUS_BITS, selected_kind=dict(interval=0, exact_knot=1, zero_discharge=2),
            selected_index_zero_based=True, selected_branch_zero_based=True,
            interval_reason=dict(unused=0, excluded_width=1, descending=2, unbracketed=3, candidate=4),
            diagnostic_packing='level_offsets index level/interval/knot fields; root_offsets index compact roots/energies/kinds/branches',
            diagnostic_order='Ascending interval candidates first, then ascending exact upper knots; strict energy< first tie',
            diagnostics_cache_hit='Retained prior candidate arrays/metadata; actual bisections/metric_evaluations reset0 for a hit',
            root_outputs_caller_mutation_prohibited=True, metric_evaluations_include_final_diagnostic=True,
            finite_arithmetic_guard_stricter_than_native=True, performance_qualified=False, physical_validation=False)
