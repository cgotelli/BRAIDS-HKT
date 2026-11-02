"""Isolated device-controlled FP64 NORMAL section root, without mode9 traces.

The exact native ascending terrain-branch search and stopping order execute in
one cooperative block per section. Long sections use tiled global workspaces,
not a4096-face cap. No CPU numerical fallback or production scope changes.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import live_section_metrics as geometry_module
from .time_integration import OPTIONS

GEOMETRY_SHA = '168de3fdd330d2c00e9f41ca09a86384b98f6421bcfe624d0a1c85c06e3ddcfc'
NAMES = ('ln_inputs','ln_metrics','ln_solve')
STATUS_BITS = dict(invalid_live_input=1, inadmissible_current_profile=2,
                   no_normal_root_found=8, invalid_arithmetic=16)
freeze = geometry_module.freeze
require = geometry_module.require
identity = geometry_module.identity


def validate_controls(geometry, controls):
    nb = len(geometry['offsets'])-1; data = {}
    for key in ('mode','section_control','energy_slope','seed_node'):
        value = np.asarray(controls[key]); dtype = np.float64 if key == 'energy_slope' else np.int32
        require(value.shape == (nb,) and value.dtype == dtype and value.flags.c_contiguous,
                'Canonical control array required: '+key)
        data[key] = freeze(value)
    require(np.all(data['mode'] == 9) and np.all(data['section_control'] == 1),
            'Only mode9 NORMAL section control1 is implemented')
    require(np.isfinite(data['energy_slope']).all() and np.all(data['energy_slope'] > 0),
            'Finite strictly positive energy slope required')
    count = np.diff(geometry['offsets'])
    require(np.all((data['seed_node'] >= 1)&(data['seed_node'] < count)),
            'Zero-based initially interior seed_node metadata required')
    return MappingProxyType(data)


def workspace_layout(geometry, *, maximum_workspace_bytes=512*1024*1024):
    require(type(maximum_workspace_bytes) is int and maximum_workspace_bytes > 0,
            'Explicit positive integer workspace byte bound required')
    counts = np.diff(geometry['offsets']).astype(np.int64)
    capacity = np.asarray([1 << int(n).bit_length() for n in counts], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(capacity)]
    nbf = len(geometry['faces']); nb = len(counts)
    # Dynamic arrays including two independent node outputs, sort, cache and
    # diagnostics. Immutable geometry/controls and external inputs are separate.
    estimated = 8*int(offsets[-1]) + 8*(3*nbf+2*nb) + (18*8+14*4)*nb + 8
    require(int(capacity.max()) < np.iinfo(np.int32).max and int(offsets[-1]) < np.iinfo(np.int32).max,
            'Sorting workspace exceeds canonical int32 addressing')
    require(estimated <= maximum_workspace_bytes, 'Declared GPU workspace byte capacity exceeded; no CPU fallback')
    return dict(level_capacity=freeze(capacity.astype(np.int32)),
        level_offsets=freeze(offsets.astype(np.int32)), estimated_dynamic_bytes=estimated,
        maximum_workspace_bytes=maximum_workspace_bytes)


class ResidentNormalSection:
    """Fixed-pointer GPU root and diagnostic metric-only interface.

    bed/discharge/optional metric stage may change VALUES in-place. Every solve
    checks current CUDA device as well as pointer/shape/dtype identities. Call
    check_storage() before externally captured raw graph.launch(). metrics()
    uses the identical cooperative helper but separate outputs and no root-cache
    writes. invalidate_cache() is explicit device-only reset, never a host solve.
    """
    def __init__(self, geometry, controls, bed_device, discharge_device, *,
                 metric_stage_device=None, source_path=None, maximum_workspace_bytes=512*1024*1024):
        require(hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest() == GEOMETRY_SHA,
                'Qualified immutable geometry helper changed')
        self.host = geometry_module.validate_geometry(geometry)
        self.controls = validate_controls(self.host, controls)
        self.topology = geometry_module.topology(self.host)
        self.layout = workspace_layout(self.host, maximum_workspace_bytes=maximum_workspace_bytes)
        self.nc = len(self.host['manning']); self.nb = len(self.host['offsets'])-1
        self.nbf = len(self.host['faces']); self.ng = len(self.topology['group_manning'])
        import cupy as cp
        self.cp = cp; self.device_id = cp.cuda.Device().id
        self.bed = bed_device; self.discharge = discharge_device; self.metric_stage = metric_stage_device
        for name,value,shape in [('bed',self.bed,(self.nc,)),('discharge',self.discharge,(self.nb,))]+(
            [('metric_stage',self.metric_stage,(self.nb,))] if self.metric_stage is not None else []):
            require(isinstance(value,cp.ndarray) and value.shape == shape and value.dtype == cp.float64
                and value.flags.c_contiguous and value.device.id == self.device_id,
                'Current-device resident contiguous FP64 input required: '+name)
        host = dict(self.host, **self.controls, **self.topology)
        host['face_group'] = freeze(np.repeat(np.arange(self.ng,dtype=np.int32), np.diff(self.topology['group_offsets'])))
        host['level_capacity'] = self.layout['level_capacity']; host['level_offsets'] = self.layout['level_offsets']
        self._host_arrays = MappingProxyType(host)
        self._arrays = {key: cp.asarray(value) for key,value in host.items()}
        self.outputs = self._metric_outputs()
        self.outputs.update(stage=cp.zeros(self.nb), residual=cp.empty(self.nb),
            cache_hit=cp.zeros(self.nb,dtype=cp.int32), selected_branch=cp.full(self.nb,-1,dtype=cp.int32),
            selected_kind=cp.full(self.nb,-1,dtype=cp.int32), bisections=cp.zeros(self.nb,dtype=cp.int32),
            expansions=cp.zeros(self.nb,dtype=cp.int32), metric_evaluations=cp.zeros(self.nb,dtype=cp.int32))
        self.metric_outputs = self._metric_outputs()
        self.cache = dict(valid=cp.zeros(self.nb,dtype=cp.int32), bed=cp.empty(self.nbf),
            discharge=cp.empty(self.nb),stage=cp.empty(self.nb),area=cp.empty(self.nb),top_width=cp.empty(self.nb),
            conveyance=cp.empty(self.nb),residual=cp.empty(self.nb),
            selected_branch=cp.empty(self.nb,dtype=cp.int32),selected_kind=cp.empty(self.nb,dtype=cp.int32))
        self.work = dict(levels=cp.empty(int(self.layout['level_offsets'][-1])),
            global_status=cp.zeros(1,dtype=cp.int32))
        self._source = (Path(source_path) if source_path else Path(__file__).with_suffix('.cu')).resolve()
        self._source_sha = hashlib.sha256(self._source.read_bytes()).hexdigest()
        self._python_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.module = cp.RawModule(code=self._source.read_text(), options=OPTIONS, name_expressions=NAMES)
        self.kernels = {key:self.module.get_function(key) for key in NAMES}
        self._array_identity = {key: identity(value) for key,value in host.items()}
        self._storage = self._storage_identity()

    def _metric_outputs(self):
        cp=self.cp
        return dict(nodes=cp.empty(self.nbf+self.nb),node_offsets=self._arrays['node_offsets'],
            area=cp.empty(self.nb),top_width=cp.empty(self.nb),conveyance=cp.empty(self.nb),
            status=cp.zeros(self.nb,dtype=cp.int32))

    def _storage_identity(self):
        values = [('bed',self.bed),('discharge',self.discharge)]
        if self.metric_stage is not None: values.append(('metric_stage',self.metric_stage))
        values += [(prefix+key,value) for prefix,mapping in (
            ('geometry:',self._arrays),('root:',self.outputs),('metric:',self.metric_outputs),
            ('cache:',self.cache),('work:',self.work)) for key,value in mapping.items()]
        return tuple((key,id(value),value.data.ptr,value.shape,value.dtype.str,
            bool(value.flags.c_contiguous),value.device.id) for key,value in values)

    def check_storage(self):
        require(self.cp.cuda.Device().id == self.device_id, 'Current CUDA device differs from captured normal section')
        require(self._storage_identity() == self._storage, 'Captured normal-section storage changed')
        return True

    def invalidate_cache(self):
        self.check_storage(); self.cache['valid'].fill(0)

    def _validate_live(self, value, outputs, nonnegative):
        outputs['status'].fill(0); self.work['global_status'].fill(0)
        self.kernels['ln_inputs'](((max(self.nc,self.nb)+127)//128,), (128,),
            (self.bed,value,np.int32(self.nc),np.int32(self.nb),np.int32(nonnegative),
             self.work['global_status'],outputs['status']))

    def metrics(self):
        self.check_storage(); require(self.metric_stage is not None, 'Bind fixed metric_stage_device at construction')
        a,o=self._arrays,self.metric_outputs
        self._validate_live(self.metric_stage,o,False)
        self.kernels['ln_metrics']((self.nb,),(128,),
            (a['owner'],a['faces'],a['offsets'],a['face_group'],a['group_manning'],a['length'],
             a['left_offset'],a['right_offset'],self.bed,self.metric_stage,self.work['global_status'],
             np.int32(self.nb),o['nodes'],o['area'],o['top_width'],o['conveyance'],o['status']))
        return o

    def solve(self):
        self.check_storage(); a,o,c=self._arrays,self.outputs,self.cache
        self._validate_live(self.discharge,o,True)
        self.kernels['ln_solve']((self.nb,),(128,),
            (a['owner'],a['faces'],a['offsets'],a['face_group'],a['group_manning'],a['length'],
             a['left_offset'],a['right_offset'],a['energy_slope'],self.bed,self.discharge,self.work['global_status'],
             a['level_offsets'],a['level_capacity'],np.int32(self.nb),o['nodes'],self.work['levels'],
             c['valid'],c['bed'],c['discharge'],c['stage'],c['area'],c['top_width'],c['conveyance'],
             c['residual'],c['selected_branch'],c['selected_kind'],
             o['stage'],o['area'],o['top_width'],o['conveyance'],o['residual'],o['status'],o['cache_hit'],
             o['selected_branch'],o['selected_kind'],o['bisections'],o['expansions'],o['metric_evaluations']))
        return o

    def verify_geometry(self):
        self.check_storage()
        require(all(identity(self.cp.asnumpy(value)) == self._array_identity[key]
            for key,value in self._arrays.items()), 'Immutable normal geometry/control bytes changed')
        require(hashlib.sha256(self._source.read_bytes()).hexdigest() == self._source_sha
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == self._python_sha
            and hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest() == GEOMETRY_SHA,
            'Executed normal section source identity changed')
        return True

    def receipt(self):
        return dict(backend='gpu',automatic_CPU_fallback=False,precision='FP64',device_id=self.device_id,
            source=str(self._source),source_sha256=self._source_sha,python_sha256=self._python_sha,
            geometry_helper_sha256=GEOMETRY_SHA,requested_cuda_options=list(OPTIONS),
            immutable_inputs=self._array_identity,normal_only=True,mode9_trace_implemented=False,
            critical_implemented=False,production_guard_changed=False,
            block_size=128,root_control='Device cooperative block; no host per-iteration launches',
            metric_helper='Same tiled segmented helper for root and metrics(), distinct outputs/cache',
            sorting='Block-cooperative bitonic over global power-of-two workspace; exact distinct levels',
            maximum_faces_per_section=int(np.diff(self.host['offsets']).max()),
            estimated_dynamic_bytes=self.layout['estimated_dynamic_bytes'],
            maximum_workspace_bytes=self.layout['maximum_workspace_bytes'],
            exact_cache_keys=['discharge equality','every section face owner bed equality'],
            current_device_guard=True,raw_graph_replay_requires_check_storage=True,
            status_bits=STATUS_BITS,selected_kind=dict(interval=0,terrain_knot=1,zero_discharge=2),
            selected_branch_zero_based=True,metric_evaluations_include_final_diagnostic=True,
            performance_qualified=False,physical_validation=False)
