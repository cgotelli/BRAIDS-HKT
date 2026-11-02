"""Resident FP64 production-section nodes and metrics, without live roots.

Static exact-Manning contiguous groups are prepared once in linear space/time.
Four deterministic kernels parallelize nodes/segments, roughness groups and
boundaries. Strided block reductions permit arbitrary practical section lengths;
no mesh-size dispatch, CPU numerical fallback or solver guard relaxation exists.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from .time_integration import OPTIONS

KEYS = ('owner','faces','offsets','length','left_offset','right_offset','mid_offset','manning')
INT_KEYS = ('owner','faces','offsets')
NAMES = ('lsm_nodes','lsm_segments','lsm_groups','lsm_totals')


def require(condition, message):
    if not condition: raise ValueError(message)


def freeze(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def identity(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
        sha256=hashlib.sha256(value.tobytes()).hexdigest())


def validate_counts(nc, nf, nbf, nb):
    require(min(nc,nf,nbf,nb) > 0 and max(nc,nf,nbf+nb) < np.iinfo(np.int32).max,
            'Nonempty int32-sized section geometry required')


def validate_geometry(geometry):
    flag = np.asarray(geometry.get('production_section_geometry', False))
    require(flag.shape == () and flag.dtype.kind == 'b' and bool(flag),
            'production_section_geometry=True is the only qualified metric policy')
    data = {}
    for key in KEYS:
        value = np.asarray(geometry[key])
        dtype = np.int32 if key in INT_KEYS else np.float64
        require(value.ndim == 1 and value.dtype == dtype, 'Canonical one-dimensional '+str(np.dtype(dtype))+' required: '+key)
        require(value.flags.c_contiguous, 'Contiguous geometry required: '+key)
        if key not in INT_KEYS: require(np.isfinite(value).all(), 'Nonfinite immutable geometry: '+key)
        data[key] = freeze(value)
    owner, faces, offsets, manning = (data[k] for k in ('owner','faces','offsets','manning'))
    nc, nf, nbf, nb = len(manning), len(owner), len(faces), len(offsets)-1
    validate_counts(nc,nf,nbf,nb)
    require(offsets[0] == 0 and offsets[-1] == nbf and np.all(np.diff(offsets.astype(np.int64)) > 0),
            'CSR must contain nonempty canonically ordered boundaries')
    require(np.all((owner >= 0)&(owner < nc)) and np.all((faces >= 0)&(faces < nf)), 'Owner/face index outside geometry')
    require(len(np.unique(faces)) == nbf, 'A physical boundary face may occur only once')
    require(np.all(manning > 0), 'Strictly positive Manning required; zero roughness is outside scope')
    for key in ('length','left_offset','right_offset','mid_offset'):
        require(data[key].shape == (nf,), 'Per-face shape differs: '+key)
    require(np.all(data['length'] > 0), 'Strictly positive face lengths required')
    return MappingProxyType(data)


def topology(host):
    offsets, faces, owner = (host[k] for k in ('offsets','faces','owner'))
    nb = len(offsets)-1; roughness = host['manning'][owner[faces]]
    starts = np.zeros(len(faces), dtype=bool); starts[offsets[:-1]] = True
    starts[1:] |= roughness[1:] != roughness[:-1]
    group_offsets = np.r_[np.flatnonzero(starts), len(faces)].astype(np.int32)
    boundary_group_offsets = np.r_[np.int32(0), np.cumsum(starts, dtype=np.int32)][offsets]
    face_boundary = np.repeat(np.arange(nb, dtype=np.int32), np.diff(offsets))
    return MappingProxyType({key: freeze(value) for key, value in dict(
        node_offsets=offsets+np.arange(nb+1, dtype=np.int32), face_boundary=face_boundary,
        group_offsets=group_offsets, boundary_group_offsets=boundary_group_offsets,
        group_boundary=face_boundary[group_offsets[:-1]], group_manning=roughness[group_offsets[:-1]]).items()})


class ResidentSectionMetrics:
    """Explicit GPU component with fixed device bed/stage pointers.

    Modify bed/stage values in-place between queries. evaluate() neither uploads
    nor downloads fields and is capture-compatible; inputs/outputs cannot be
    replaced. An externally retained raw CUDA graph bypasses Python guards:
    callers MUST call check_storage() before its graph.launch(). Device geometry
    values are immutable by contract; verify_geometry()
    performs an explicit outside-capture byte audit. Check status before using
    metrics: any nonfinite live bed invalidates all boundaries, and bad stage or
    arithmetic invalidates its boundary. Invalid metrics are NaN, never success.
    """
    def __init__(self, geometry, bed_device, stage_device, *, source_path=None):
        self.host = validate_geometry(geometry); self.topology = topology(self.host)
        self.nc = len(self.host['manning']); self.nf = len(self.host['owner'])
        self.nbf = len(self.host['faces']); self.nb = len(self.host['offsets'])-1
        self.ng = len(self.topology['group_manning'])
        import cupy as cp
        self.cp = cp; self.bed = bed_device; self.stage = stage_device
        for key, value, shape in (('bed', self.bed, (self.nc,)), ('stage', self.stage, (self.nb,))):
            require(isinstance(value, cp.ndarray) and value.shape == shape and value.dtype == cp.float64
                and value.flags.c_contiguous and value.device.id == cp.cuda.Device().id,
                'Resident contiguous FP64 input on current device required: '+key)
        self._arrays = {key: cp.asarray(value) for key, value in {**self.host, **self.topology}.items()}
        self.outputs = dict(nodes=cp.empty(self.nbf+self.nb, dtype=cp.float64),
            node_offsets=self._arrays['node_offsets'], area=cp.empty(self.nb, dtype=cp.float64),
            top_width=cp.empty(self.nb, dtype=cp.float64), conveyance=cp.empty(self.nb, dtype=cp.float64),
            status=cp.zeros(self.nb, dtype=cp.int32))
        self.work = dict(segment_area=cp.empty(self.nbf), segment_width=cp.empty(self.nbf),
            segment_perimeter=cp.empty(self.nbf), group_conveyance=cp.empty(self.ng),
            global_status=cp.zeros(1, dtype=cp.int32))
        self._source = (Path(source_path) if source_path else Path(__file__).with_suffix('.cu')).resolve()
        self._source_sha = hashlib.sha256(self._source.read_bytes()).hexdigest()
        self._python_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.module = cp.RawModule(code=self._source.read_text(), options=OPTIONS, name_expressions=NAMES)
        self.kernels = {name: self.module.get_function(name) for name in NAMES}
        self._host_identity = {k: identity(v) for k, v in {**self.host, **self.topology}.items()}
        self._storage = self._storage_identity()

    def _storage_identity(self):
        values = [('bed', self.bed), ('stage', self.stage)]
        values += [(prefix+key, value) for prefix, mapping in
            (('geometry:',self._arrays),('output:',self.outputs),('work:',self.work)) for key, value in mapping.items()]
        return tuple((key, id(value), value.data.ptr, value.shape, value.dtype.str,
                      bool(value.flags.c_contiguous), value.device.id) for key, value in values)

    def check_storage(self):
        require(self._storage_identity() == self._storage, 'Captured metric input/geometry/output storage changed')
        return True

    def evaluate(self):
        self.check_storage()
        a, o, w = self._arrays, self.outputs, self.work
        o['status'].fill(0); w['global_status'].fill(0)
        integer = np.int32
        self.kernels['lsm_nodes'](((max(self.nc,self.nbf,self.nb)+127)//128,), (128,),
            (a['owner'],a['faces'],a['offsets'],a['face_boundary'],a['left_offset'],a['right_offset'],
             self.bed,self.stage,integer(self.nc),integer(self.nbf),integer(self.nb),o['nodes'],o['status'],w['global_status']))
        self.kernels['lsm_segments'](((self.nbf+127)//128,), (128,),
            (a['faces'],a['offsets'],a['face_boundary'],a['length'],o['nodes'],self.stage,integer(self.nbf),
             w['segment_area'],w['segment_width'],w['segment_perimeter'],o['status']))
        self.kernels['lsm_groups']((self.ng,), (128,),
            (a['group_offsets'],a['group_boundary'],a['group_manning'],w['segment_area'],w['segment_perimeter'],
             integer(self.ng),w['group_conveyance'],o['status']))
        self.kernels['lsm_totals']((self.nb,), (128,),
            (a['offsets'],a['boundary_group_offsets'],w['segment_area'],w['segment_width'],w['group_conveyance'],
             integer(self.nb),w['global_status'],o['status'],o['area'],o['top_width'],o['conveyance']))
        return self.outputs

    def verify_geometry(self):
        require(self._storage_identity() == self._storage, 'Metric storage changed')
        require(all(identity(self.cp.asnumpy(value)) == self._host_identity[key]
            for key, value in self._arrays.items()), 'Immutable device section geometry was modified')
        require(hashlib.sha256(self._source.read_bytes()).hexdigest() == self._source_sha
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == self._python_sha,
            'Executed section metric source identity changed')
        return True

    def receipt(self):
        return dict(backend='gpu', automatic_CPU_fallback=False, precision='FP64',
            production_section_geometry=True, all_submerged_pools=True, roots_implemented=False,
            solver_eligibility_changed=False, source=str(self._source), source_sha256=self._source_sha,
            python_sha256=self._python_sha, requested_cuda_options=list(OPTIONS),
            geometry=self._host_identity, cells=self.nc, faces=self.nbf, boundaries=self.nb,
            contiguous_roughness_groups=self.ng, block_size=128, kernels_per_query=4,
            reduction='Deterministic lane-strided partial sums and binary tree; not native serial addition order',
            topology_prepared_once=True, repeated_live_bed_stage_host_transfers=False,
            raw_graph_replay_requires_explicit_check_storage=True,
            status_bits=dict(nonfinite_live_input=1, nonfinite_or_invalid_arithmetic=2),
            performance_qualified=False, physical_validation=False)
