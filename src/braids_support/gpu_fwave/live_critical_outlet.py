"""Validation-only resident mode4 characteristic critical OUTLET.

This is not the mode9 critical-inlet root. Owner h/hu/hv and owner bed define
its active set on every call. No surveyed offsets, prescribed Q or cache is used.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import MappingProxyType
import numpy as np
from .time_integration import OPTIONS
from . import time_integration as options_module

HERE=Path(__file__).resolve().parent
NATIVE_SOURCE=HERE/'runs/cpu_filter_convergence_control_20261002_v1/runtime/cutcell_solver.f90'
NATIVE_SHA='75a5451cce5d11e2aec244330aa91724e514dbe16dfdfef97fb9b50572020340'
STATUS_BITS=dict(invalid_live_input=1,upper_bracket_failure=2,empty_controlled_geometry=4,
    residual_failure=8,controlled_speed_limit=16,invalid_arithmetic=32,active_set_exhausted=64)
REMOVAL_REASONS=dict(retained_or_unassigned=0,initial_dry=1,initial_reverse=2,
    unresolved_shallow=3,active_dry=4,active_inward=5)
NAMES=('lco_inputs','lco_apply')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(array):
    a=np.asarray(array)
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def freeze(array):
    a=np.asarray(array);return np.frombuffer(a.tobytes(order='C'),dtype=a.dtype).reshape(a.shape)


def validate_geometry(geometry):
    nc=np.asarray(geometry['nc'])
    require(nc.shape==() and nc.dtype.kind in 'iu' and 0<int(nc)<2**31-1,'Positive int32-addressable nc required')
    host={}
    for key in ('owner','faces','offsets'):
        a=np.asarray(geometry[key]);require(a.ndim==1 and a.dtype==np.int32 and a.flags.c_contiguous,
            'Canonical contiguous int32 vector required: '+key);host[key]=freeze(a)
    owner,faces,offsets=(host[k] for k in ('owner','faces','offsets'))
    nf=len(owner);nb=len(offsets)-1
    require(0<nf<2**31-1 and 0<nb<2**31-1 and len(faces)<2**31-1,'Int32-addressable face/boundary counts required')
    require(offsets[0]==0 and offsets[-1]==len(faces) and np.all(np.diff(offsets.astype(np.int64))>0),
        'Nonempty canonical boundary CSR required')
    require(np.all((owner>=0)&(owner<int(nc))) and np.all((faces>=0)&(faces<nf)),'Owner/face index outside range')
    require(len(np.unique(faces))==len(faces),'A physical face cannot occur twice')
    for key,shape in (('length',(nf,)),('normal',(2,nf))):
        a=np.asarray(geometry[key]);require(a.shape==shape and a.dtype==np.float64 and a.flags.c_contiguous
            and np.isfinite(a).all(),'Finite canonical FP64 geometry required: '+key);host[key]=freeze(a)
    require(np.all(host['length']>0),'Positive face lengths required')
    require(np.all(abs(np.sum(host['normal']*host['normal'],axis=0)-1.)<=32*np.finfo(float).eps),
        'Outward unit normals required; no silent normalization')
    return MappingProxyType(host),int(nc)


def validate_controls(gravity,dry,speed_limit,block_size,max_workspace_bytes):
    for key,value,positive in (('gravity',gravity,True),('dry',dry,False),('speed_limit',speed_limit,True)):
        require(np.isscalar(value) and np.isfinite(value) and (value>0 if positive else value>=0),
            'Finite physical scalar required: '+key)
    require(type(block_size) is int and block_size in (32,64,128,256),'Power-of-two cooperative block32..256 required')
    require(type(max_workspace_bytes) is int and max_workspace_bytes>0,'Positive explicit workspace byte bound required')
    return float(gravity),float(dry),float(speed_limit),block_size,max_workspace_bytes


class ResidentCriticalOutlet:
    """Fixed pointers/current CUDA device; live values may change in-place.

Call check_storage() before an external captured graph replay. verify_geometry()
is the explicit outside-capture content/source audit. Geometry and owned output
arrays are immutable-by-contract; the cheap pointer guard is not a content hash.
"""
    def __init__(self,geometry,bed_device,q_device,*,gravity,dry,speed_limit,source_path=None,
                 block_size=128,max_workspace_bytes=2**31):
        require(sha(NATIVE_SOURCE)==NATIVE_SHA,'Frozen native mode4 reference changed')
        self.host,self.nc=validate_geometry(geometry)
        self.gravity,self.dry,self.speed_limit,self.block_size,self.max_workspace_bytes=validate_controls(
            gravity,dry,speed_limit,block_size,max_workspace_bytes)
        self.nf=len(self.host['owner']);self.nb=len(self.host['offsets'])-1
        import cupy as cp
        self.cp=cp;self.device_id=cp.cuda.Device().id;self.bed=bed_device;self.q=q_device
        for key,value,shape in (('bed',self.bed,(self.nc,)),('q',self.q,(3,self.nc))):
            require(isinstance(value,cp.ndarray) and value.shape==shape and value.dtype==cp.float64
                and value.flags.c_contiguous and value.device.id==self.device_id,
                'Current-device contiguous FP64 live input required: '+key)
        self._source=(Path(source_path) if source_path else Path(__file__).with_suffix('.cu')).resolve()
        self._source_sha=sha(self._source);self._python_sha=sha(__file__)
        self._options=tuple(OPTIONS);self._options_dependency_sha=sha(options_module.__file__)
        # No face-count cap: workspace is O(nf+nb), with strided cooperative loops.
        floats=dict(trace=(3,self.nf),invariant=(self.nf,),face_speed=(self.nf,),stage=(self.nb,),
            area=(self.nb,),top_width=(self.nb,),characteristic_Q=(self.nb,),critical_Q=(self.nb,),
            residual=(self.nb,),outward_discharge=(self.nb,),lower_residual=(self.nb,),
            upper_residual=(self.nb,),failed_speed=(self.nb,))
        integers=dict(branch=(self.nf,),initial_branch=(self.nf,),removal_pass=(self.nf,),
            removal_reason=(self.nf,),face_status=(self.nf,),status=(self.nb,),active_passes=(self.nb,),
            expansions=(self.nb,),bisections=(self.nb,),metric_evaluations=(self.nb,),failed_face=(self.nb,))
        self.workspace_bytes=sum(a.nbytes for a in self.host.values())+sum(int(np.prod(s))*8 for s in floats.values())+sum(int(np.prod(s))*4 for s in integers.values())+4
        require(self.workspace_bytes<=self.max_workspace_bytes,'Requested outlet workspace exceeds explicit bound')
        self._arrays={k:cp.asarray(v) for k,v in self.host.items()}
        self.outputs={**{k:cp.empty(s,dtype=cp.float64) for k,s in floats.items()},
            **{k:cp.empty(s,dtype=cp.int32) for k,s in integers.items()}}
        self.work=dict(invalid=cp.zeros(1,dtype=cp.int32))
        self.module=cp.RawModule(code=self._source.read_text(),options=self._options,name_expressions=NAMES)
        self.kernels={name:self.module.get_function(name) for name in NAMES}
        self._host_identity={k:identity(v) for k,v in self.host.items()}
        self._controls=(self.gravity,self.dry,self.speed_limit,self.block_size,self.nc,self.nf,self.nb)
        self._storage=self._storage_identity()

    def _storage_identity(self):
        values=[('bed',self.bed),('q',self.q)]
        values += [(prefix+k,v) for prefix,mapping in
            (('geometry:',self._arrays),('output:',self.outputs),('work:',self.work)) for k,v in mapping.items()]
        return tuple((k,id(v),v.data.ptr,v.shape,v.dtype.str,bool(v.flags.c_contiguous),v.device.id) for k,v in values)

    def check_storage(self):
        require(self.cp.cuda.Device().id==self.device_id,'Current CUDA device changed')
        require(self._storage_identity()==self._storage,'Captured critical outlet storage changed')
        require((self.gravity,self.dry,self.speed_limit,self.block_size,self.nc,self.nf,self.nb)==self._controls,
            'Captured critical outlet controls changed')
        require(tuple(OPTIONS)==self._options,'Critical outlet compiler controls changed')
        return True

    def evaluate(self):
        self.check_storage();a,o=self._arrays,self.outputs
        for key in ('trace','invariant','face_speed','branch','initial_branch','removal_pass','removal_reason','face_status'):
            o[key].fill(0)
        self.work['invalid'].fill(0)
        self.kernels['lco_inputs'](((self.nc+127)//128,),(128,),
            (self.bed,self.q,np.int32(self.nc),self.work['invalid']))
        args=(a['owner'],a['faces'],a['offsets'],a['length'],a['normal'],self.bed,self.q,
            self.work['invalid'],np.int32(self.nc),np.int32(self.nf),np.int32(self.nb),
            np.float64(self.gravity),np.float64(self.dry),np.float64(self.speed_limit))
        order=('trace','branch','initial_branch','invariant','removal_pass','removal_reason','face_speed',
            'status','face_status','stage','area','top_width','characteristic_Q','critical_Q','residual',
            'outward_discharge','lower_residual','upper_residual','active_passes','expansions','bisections',
            'metric_evaluations','failed_face','failed_speed')
        self.kernels['lco_apply']((self.nb,),(self.block_size,),args+tuple(o[k] for k in order))
        return o

    def verify_geometry(self):
        self.check_storage()
        require(all(identity(self.cp.asnumpy(v))==self._host_identity[k] for k,v in self._arrays.items()),
            'Immutable outlet geometry values changed')
        require(sha(self._source)==self._source_sha and sha(__file__)==self._python_sha
            and sha(NATIVE_SOURCE)==NATIVE_SHA and sha(options_module.__file__)==self._options_dependency_sha,
            'Critical outlet source identity changed')
        return True

    def receipt(self):
        return dict(backend='gpu',precision='FP64',automatic_CPU_fallback=False,mode=4,
            source=str(self._source),source_sha256=self._source_sha,python_sha256=self._python_sha,
            native_source=str(NATIVE_SOURCE),native_source_sha256=NATIVE_SHA,
            requested_cuda_options=list(self._options),options_dependency_sha256=self._options_dependency_sha,
            block_size=self.block_size,device_id=self.device_id,
            immutable_inputs=self._host_identity,gravity=self.gravity,dry=self.dry,speed_limit=self.speed_limit,
            workspace_bytes=self.workspace_bytes,face_count_cap=None,status_bits=STATUS_BITS,
            removal_reasons=REMOVAL_REASONS,cache=False,
            parallel_scope='One cooperative block per boundary; strided parallel face work and deterministic reductions; device-controlled root/active-set loops',
            root_limits=dict(upper_iterations=60,bisections_per_pass=90,active_passes='assigned face count + 1'),
            speed_scope='Native cap applies only to controlled branch1; reverse-closed and supercritical-released tangential states unchanged',
            stage_scope='stage/A/T/characteristicQ/criticalQ describe branch1 only; total outward trace Q includes released branch2',
            no_controlled_stage='NaN stage, zero controlled metrics for dry/reverse/released-only or unresolved closure',
            failure_policy='Safe zero trace for each failed boundary; outward discharge NaN; classifications and failure diagnostics retained',
            native_failure_policy_difference='Native returns globally on first failing boundary; GPU boundaries report independently; failed partial native trace is not usable',
            additional_GPU_safeguards='Nonfinite live q/bed or negative owner h rejects all boundaries; explicit nonfinite intermediate arithmetic failure',
            current_device_guard=True,raw_graph_replay_requires_check_storage=True,
            unassigned_trace='Always zero',trace_not_accepted_Riemann_flux=True,
            performance_qualified=False,coupled_implemented=False,physical_validation=False,production_modified=False)
