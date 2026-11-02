"""Validation-only GPU production mode9 NORMAL trace/discharge closure.

Bind the qualified root's stage/status and the SAME resident bed/discharge used
by that root. This component never computes or authenticates a stage itself.
Its own unassigned traces are zero; a future coupled adapter must scatter only
assigned faces. Root+closure sequencing is the caller's responsibility. No
production guard is lifted, and traceQ is not an accepted Riemann-flux ledger.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import live_section_metrics as geometry_module
from .time_integration import OPTIONS

GEOMETRY_SHA='168de3fdd330d2c00e9f41ca09a86384b98f6421bcfe624d0a1c85c06e3ddcfc'
STATUS_BITS=dict(invalid_live_input=1,upstream_root_failed=2,positive_Q_without_wet_area=4,
                 speed_limit_exceeded=8,invalid_arithmetic=16)
NAMES=('lnt_inputs','lnt_apply')
require,freeze,identity=geometry_module.require,geometry_module.freeze,geometry_module.identity


def validate_geometry(geometry):
    host=dict(geometry_module.validate_geometry(geometry))
    normal=np.asarray(geometry['normal']);nf=len(host['owner'])
    require(normal.shape==(2,nf) and normal.dtype==np.float64 and normal.flags.c_contiguous,
            'Canonical contiguous FP64 normal(2,nf) required')
    require(np.isfinite(normal).all(),'Finite immutable normals required')
    require(np.all(np.abs(np.sum(normal*normal,axis=0)-1.)<=32*np.finfo(float).eps),
            'Unit outward face normals required; no silent normalization')
    host['normal']=freeze(normal)
    return MappingProxyType(host)


def validate_controls(dry,speed_limit):
    require(np.isscalar(dry) and np.isfinite(dry) and dry>=0.,'Finite nonnegative dry threshold required')
    require(np.isscalar(speed_limit) and np.isfinite(speed_limit) and speed_limit>0.,
            'Finite positive speed limit required')
    return np.float64(dry),np.float64(speed_limit)


class ResidentNormalTrace:
    """Fixed-pointer, current-device, capture-compatible GPU trace component.

    Live input VALUES may change in-place, never their storage. Static geometry
    and owned outputs remain private/immutable by contract. check_storage()
    precedes raw externally captured graph replay; verify_geometry() is the
    explicit, outside-capture content/source audit. Nonfinite hydraulic state
    or negative owner depth is an additional GPU safeguard, not native mode9
    physical dependence: native all-pool mode9 ignores owner momentum.
    """
    def __init__(self,geometry,bed_device,q_device,stage_device,root_status_device,discharge_device,
                 *,dry,speed_limit,source_path=None):
        require(hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest()==GEOMETRY_SHA,
                'Qualified immutable geometry helper changed')
        self.host=validate_geometry(geometry);self.dry,self.speed_limit=validate_controls(dry,speed_limit)
        self.nc=len(self.host['manning']);self.nf=len(self.host['owner']);self.nb=len(self.host['offsets'])-1
        import cupy as cp
        self.cp=cp;self.device_id=cp.cuda.Device().id
        self.bed=bed_device;self.q=q_device;self.stage=stage_device
        self.root_status=root_status_device;self.discharge=discharge_device
        for key,value,shape,dtype in (
            ('bed',self.bed,(self.nc,),cp.float64),('q',self.q,(3,self.nc),cp.float64),
            ('stage',self.stage,(self.nb,),cp.float64),('root_status',self.root_status,(self.nb,),cp.int32),
            ('discharge',self.discharge,(self.nb,),cp.float64)):
            require(isinstance(value,cp.ndarray) and value.shape==shape and value.dtype==dtype
                and value.flags.c_contiguous and value.device.id==self.device_id,
                'Current-device canonical resident input required: '+key)
        self._arrays={key:cp.asarray(value) for key,value in self.host.items()}
        self.outputs=dict(trace=cp.zeros((3,self.nf)),status=cp.zeros(self.nb,dtype=cp.int32),
            face_status=cp.zeros(self.nf,dtype=cp.int32),discrete_wet_area=cp.empty(self.nb),
            speed=cp.empty(self.nb),inward_discharge=cp.empty(self.nb))
        self.work=dict(invalid=cp.zeros(1,dtype=cp.int32))
        self._source=(Path(source_path) if source_path else Path(__file__).with_suffix('.cu')).resolve()
        self._source_sha=hashlib.sha256(self._source.read_bytes()).hexdigest()
        self._python_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.module=cp.RawModule(code=self._source.read_text(),options=OPTIONS,name_expressions=NAMES)
        self.kernels={name:self.module.get_function(name) for name in NAMES}
        self._host_identity={k:identity(v) for k,v in self.host.items()}
        self._storage=self._storage_identity();self._control_identity=(self.dry,self.speed_limit)

    def _storage_identity(self):
        values=[(key,getattr(self,key)) for key in ('bed','q','stage','root_status','discharge')]
        values += [(prefix+key,value) for prefix,mapping in
            (('geometry:',self._arrays),('output:',self.outputs),('work:',self.work)) for key,value in mapping.items()]
        return tuple((key,id(value),value.data.ptr,value.shape,value.dtype.str,
            bool(value.flags.c_contiguous),value.device.id) for key,value in values)

    def check_storage(self):
        require(self.cp.cuda.Device().id==self.device_id,'Current CUDA device differs from captured trace component')
        require(self._storage_identity()==self._storage,'Captured normal trace storage changed')
        require((self.dry,self.speed_limit)==self._control_identity,'Captured trace scalar controls changed')
        return True

    def evaluate(self):
        self.check_storage();a,o=self._arrays,self.outputs
        o['trace'].fill(0);o['face_status'].fill(0);self.work['invalid'].fill(0)
        self.kernels['lnt_inputs'](((self.nc+127)//128,),(128,),
            (self.bed,self.q,np.int32(self.nc),self.work['invalid']))
        self.kernels['lnt_apply']((self.nb,),(128,),
            (a['owner'],a['faces'],a['offsets'],a['length'],a['mid_offset'],a['normal'],
             self.bed,self.stage,self.root_status,self.discharge,self.work['invalid'],
             np.int32(self.nf),np.int32(self.nb),self.dry,self.speed_limit,o['trace'],o['status'],
             o['face_status'],o['discrete_wet_area'],o['speed'],o['inward_discharge']))
        return o

    def verify_geometry(self):
        self.check_storage()
        require(all(identity(self.cp.asnumpy(v))==self._host_identity[k] for k,v in self._arrays.items()),
                'Immutable normal-trace geometry bytes changed')
        require(hashlib.sha256(self._source.read_bytes()).hexdigest()==self._source_sha
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==self._python_sha
            and hashlib.sha256(Path(geometry_module.__file__).read_bytes()).hexdigest()==GEOMETRY_SHA,
            'Executed trace source identity changed')
        return True

    def receipt(self):
        return dict(backend='gpu',automatic_CPU_fallback=False,precision='FP64',device_id=self.device_id,
            source=str(self._source),source_sha256=self._source_sha,python_sha256=self._python_sha,
            geometry_helper_sha256=GEOMETRY_SHA,requested_cuda_options=list(OPTIONS),
            immutable_inputs=self._host_identity,dry=float(self.dry),speed_limit=float(self.speed_limit),
            status_bits=STATUS_BITS,block_size=128,production_section_geometry=True,
            all_submerged_pools=True,tangential_velocity_retained=False,
            unassigned_trace='Zero; future adapter must scatter only assigned faces',
            failure_policy='Each failing boundary trace is zero with explicit status; its integratedQ is NaN',
            native_failure_policy_difference='Native returns globally on first failure; no failed partial trace is compared as usable',
            additional_GPU_safeguards='Reject nonfinite live arrays and negative owner hydraulic depth',
            root_sequencing='Caller binds qualified stage/status and same bed/Q; this component does not verify root freshness',
            discharge_diagnostic='Integral of trace momentum, not accepted augmented-fwave face flux',
            current_device_guard=True,raw_graph_replay_requires_check_storage=True,
            production_guard_changed=False,coupled_implemented=False,performance_qualified=False,physical_validation=False)
