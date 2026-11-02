"""Private current-Euler mode4 outlet bridge; inherited flux/sediment unchanged."""
from __future__ import annotations
import inspect
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import live_critical_outlet as component
from .time_integration import OPTIONS

require,freeze,identity,sha=component.require,component.freeze,component.identity,component.sha
NAMES=('ob_scatter','ob_merge','ob_record_trace','ob_record_float','ob_record_int')
FACE_FLOATS=('invariant','face_speed')
FACE_INTS=('branch','initial_branch','removal_pass','removal_reason','face_status')
BOUNDARY_FLOATS=('stage','area','top_width','characteristic_Q','critical_Q','residual','outward_discharge',
    'lower_residual','upper_residual','failed_speed')
BOUNDARY_INTS=('status','active_passes','expansions','bisections','metric_evaluations','failed_face')


def validate_outlets(data):
    def vector(key,dtype,shape=None):
        value=np.asarray(data[key]);require(value.ndim==1 and value.dtype==dtype and value.flags.c_contiguous,
            'Canonical native vector required: '+key)
        if shape is not None:require(value.shape==shape,'Native vector shape differs: '+key)
        return value
    mode=vector('mode',np.int32);nb=len(mode);owner=vector('owner',np.int32);nf=len(owner)
    require(np.isin(mode,[2,3,4,6,8]).all(),'Only nonsurveyed modes2/3/4/6/8 supported')
    control=vector('section_control',np.int32,(nb,));sampled=vector('sampled',np.int32,(nf,))
    require(not control.any() and not sampled.any(),'Surveyed sections remain unsupported')
    for key in ('section_seed_node','boundary_seed_position'):
        if key in data:require(np.all(vector(key,np.int32,(nb,))==-1),'Surveyed seed metadata remains unsupported')
    neighbour=vector('neighbour',np.int32,(nf,));bid=vector('bid',np.int32,(nf,))
    offsets=vector('boundary_offset',np.int32,(nb+1,));faces=vector('boundary_faces',np.int32)
    require(offsets[0]==0 and offsets[-1]==len(faces) and np.all(np.diff(offsets.astype(np.int64))>=0),
        'Canonical native boundary CSR required')
    require(np.all((faces>=0)&(faces<nf)) and len(np.unique(faces))==len(faces),'Unique physical boundary faces required')
    require(np.all((bid>=-1)&(bid<nb)) and np.array_equal(np.sort(faces),np.flatnonzero(bid>=0)),
        'Boundary CSR must cover every assigned open face exactly once')
    for b in range(nb):
        ff=faces[offsets[b]:offsets[b+1]]
        require(np.all(bid[ff]==b) and np.all(neighbour[ff]<0),'Boundary faces must belong to own exterior boundary')
    selected=np.flatnonzero(mode==4).astype(np.int32);require(len(selected)>0,'Explicit mode4 critical outlet required')
    counts=np.array([offsets[b+1]-offsets[b] for b in selected],dtype=np.int64)
    require(np.all(counts>0) and counts.sum()<2**31-1,'Nonempty int32-addressable critical outlet sections required')
    compact=np.concatenate([faces[offsets[b]:offsets[b+1]] for b in selected]).astype(np.int32)
    raw_nc=np.asarray(data['nc']);require(raw_nc.shape==() and raw_nc.dtype.kind in 'iu','Canonical integer nc required')
    nc=int(raw_nc);require(np.asarray(data['q']).shape==(3,nc),'Hydraulic q/nc mismatch')
    geometry=dict(nc=nc,owner=owner,faces=compact,offsets=np.r_[0,np.cumsum(counts)].astype(np.int32),
        length=data['length'],normal=data['normal'])
    host,count=component.validate_geometry(geometry)
    return dict(geometry=MappingProxyType(dict(host,nc=count)),global_boundary_ids=freeze(selected),
        native_metadata=MappingProxyType({key:freeze(np.asarray(data[key])) for key in
            ('mode','section_control','sampled','boundary_offset','boundary_faces','bid','neighbour')}))


def make_boundary_controls(ParentControls):
    class OutletBoundary(ParentControls):
        def __init__(self,data,inputs,source=None):
            require(source is not None,'Explicit archived parent source required')
            self.outlets=validate_outlets(data);folder=Path(source).resolve().parent
            super().__init__(data,inputs,source=source)
            cp=self.cp;self._outlet_device=cp.cuda.Device().id
            self._outlet_ids=cp.asarray(self.outlets['global_boundary_ids']);self._outlet_n=len(self._outlet_ids)
            self._outlet_faces=cp.asarray(self.outlets['geometry']['faces']);count=len(self._outlet_faces)
            self._outlet_latch=cp.zeros(self._outlet_n,dtype=cp.int32);self._outlet_q=None
            self.critical_outlet=None;self._outlet_ready_storage=None
            self._outlet_source=folder/'outlet_boundary.cu';self._outlet_source_sha=sha(self._outlet_source)
            self._outlet_python_sha=sha(__file__);self._parent_source=Path(source).resolve();self._parent_sha=sha(source)
            self._parent_python=Path(inspect.getfile(ParentControls)).resolve();self._parent_python_sha=sha(self._parent_python)
            self._outlet_module=cp.RawModule(code=self._outlet_source.read_text(),options=OPTIONS,name_expressions=NAMES)
            self._outlet_kernels={name:self._outlet_module.get_function(name) for name in NAMES}
            self.stage_records=dict(bed=cp.empty((4,self.nc)),q=cp.empty((4,3,self.nc)),trace=cp.empty((4,3,count)),
                failure_latch=cp.empty((4,self._outlet_n),dtype=cp.int32),recorded=cp.zeros(4,dtype=cp.int32))
            for keys,n,dtype in ((FACE_FLOATS,count,cp.float64),(FACE_INTS,count,cp.int32),
                (BOUNDARY_FLOATS,self._outlet_n,cp.float64),(BOUNDARY_INTS,self._outlet_n,cp.int32)):
                for key in keys:self.stage_records[key]=cp.empty((4,n),dtype=dtype)
            self._outlet_parent_static={k:identity(cp.asnumpy(v)) for k,v in self.arrays.items() if k!='forcing'}
            self._outlet_host={k:identity(np.asarray(data[k])) for k in data if isinstance(data[k],np.ndarray) and data[k].dtype.kind in 'biuf'}
            self._outlet_static={}
            for key in ('owner','normal','length','sampled'):
                require(key in inputs,'Actual current geometry missing: '+key)
                actual=cp.asnumpy(inputs[key]);expected=np.asarray(data[key])
                require(actual.dtype==expected.dtype and np.array_equal(actual,expected),'Actual/native geometry differs: '+key)
                self._outlet_static[key]=identity(actual)
            actual=cp.asnumpy(inputs['bedoffset']);expected=np.asarray(data['bedoffset'] if 'bedoffset' in data else data['boundary_bed_offset'])
            require(actual.dtype==expected.dtype and np.array_equal(actual,expected),'Actual/native pressure bed offset differs')
            self._outlet_static['bedoffset']=identity(actual)
            require(inputs['trace'].shape==(10,self.nf),'Actual ten-slot reconstruction trace required')
            self._outlet_initial_storage=self._storage_identity()

        def _storage_identity(self):
            values=[('ids',self._outlet_ids),('faces',self._outlet_faces),('latch',self._outlet_latch),
                ('status',self.status),('face_status',self.face_status),('ledger',self.ledger)]
            values += [(prefix+k,v) for prefix,d in (('parent:',self.arrays),('records:',self.stage_records)) for k,v in d.items()]
            values += [('input:'+k,self.inputs[k]) for k in ('bed','normal','length','owner','sampled','bedoffset','trace')]
            if self._outlet_q is not None:values.append(('q',self._outlet_q))
            return tuple((k,id(v),v.data.ptr,v.shape,v.dtype.str,bool(v.flags.c_contiguous),v.device.id) for k,v in values)

        def bind_stage_state(self,q):
            cp=self.cp
            require(not cp.cuda.get_current_stream().is_capturing(),'Bind stable stage-q before capture')
            require(self._outlet_q is None and self._outlet_ready_storage is None,'Outlet stage-q can be bound only once')
            require(self._storage_identity()==self._outlet_initial_storage,'Outlet construction storage changed')
            require(isinstance(q,cp.ndarray) and q.shape==(3,self.nc) and q.dtype==cp.float64 and q.flags.c_contiguous
                and q.device.id==self._outlet_device,'Canonical resident stage-q required')
            self._outlet_q=q
            self.critical_outlet=component.ResidentCriticalOutlet(self.outlets['geometry'],self.inputs['bed'],q,
                gravity=float(self.grav),dry=float(self.dry),speed_limit=float(self.speed),
                source_path=self._outlet_source.parent/'live_critical_outlet_cuda_v2.cu')
            self._outlet_component_id=id(self.critical_outlet);self._outlet_ready_storage=self._storage_identity()
            self._outlet_controls=tuple(float(getattr(self,k)) for k in ('dry','grav','speed','chezy'))

        def check_storage(self):
            require(self.cp.cuda.Device().id==self._outlet_device,'Current CUDA device differs from outlet bridge')
            require(self._outlet_ready_storage is not None,'Bind stable stage-q before outlet use')
            require(self._storage_identity()==self._outlet_ready_storage,'Captured outlet bridge storage changed')
            require(id(self.critical_outlet)==self._outlet_component_id,'Outlet component object changed')
            require(tuple(float(getattr(self,k)) for k in ('dry','grav','speed','chezy'))==self._outlet_controls,
                'Captured outlet controls changed')
            self.critical_outlet.check_storage();return True

        def begin_trial(self):
            self.check_storage();self._outlet_latch.fill(0);self.stage_records['recorded'].fill(0)

        def apply(self,q,trace):
            self.check_storage();require(q is self._outlet_q and trace is self.inputs['trace'],
                'Outlet requires bound stable stage-q and actual reconstruction trace')
            # Existing kernels reset every status and handle modes2/3/6/8 only.
            super().apply(q,trace)
            values=self.critical_outlet.evaluate();count=np.int32(len(self._outlet_faces))
            self._outlet_kernels['ob_scatter'](((int(count)+127)//128,),(128,),
                (self._outlet_faces,count,np.int32(self.nf),values['trace'],values['face_status'],trace,self.face_status))
            self._outlet_kernels['ob_merge'](((self._outlet_n+127)//128,),(128,),
                (self._outlet_ids,np.int32(self._outlet_n),values['status'],self.status,self._outlet_latch))

        def record_stage(self,index):
            self.check_storage();require(type(index) is int and 0<=index<4,'Four actual Euler stage slots required')
            cp=self.cp;s=self.stage_records;o=self.critical_outlet.outputs;n=len(self._outlet_faces);grid=((n+127)//128,)
            cp.copyto(s['bed'][index],self.inputs['bed']);cp.copyto(s['q'][index],self._outlet_q)
            cp.copyto(s['failure_latch'][index],self._outlet_latch)
            self._outlet_kernels['ob_record_trace'](grid,(128,),
                (self._outlet_faces,np.int32(n),np.int32(self.nf),self.inputs['trace'],s['trace'][index]))
            for keys,name in ((FACE_FLOATS,'ob_record_float'),(FACE_INTS,'ob_record_int')):
                for key in keys:self._outlet_kernels[name](grid,(128,),(self._outlet_faces,np.int32(n),o[key],s[key][index]))
            for key in (*BOUNDARY_FLOATS,*BOUNDARY_INTS):cp.copyto(s[key][index],o[key])
            s['recorded'][index:index+1].fill(1)

        def audit(self):
            self.check_storage();self.critical_outlet.verify_geometry()
            require(all(identity(self.cp.asnumpy(self.inputs[k]))==v for k,v in self._outlet_static.items()),'Coupled outlet geometry changed')
            require(all(identity(self.cp.asnumpy(self.arrays[k]))==v for k,v in self._outlet_parent_static.items()),'Parent boundary values changed')
            require(sha(__file__)==self._outlet_python_sha and sha(self._outlet_source)==self._outlet_source_sha
                and sha(self._parent_source)==self._parent_sha and sha(self._parent_python)==self._parent_python_sha,
                'Executed outlet bridge/parent source changed')
            return True

        def receipt(self):
            return dict(backend='gpu',automatic_CPU_fallback=False,mode=4,surveyed_sections=False,
                source_sha256=self._outlet_source_sha,python_sha256=self._outlet_python_sha,
                parent_cuda_sha256=self._parent_sha,parent_python_sha256=self._parent_python_sha,
                global_boundary_ids=self.outlets['global_boundary_ids'].tolist(),native_input_identities=self._outlet_host,
                component=None if self.critical_outlet is None else self.critical_outlet.receipt(),
                single_stable_q_binding=True,current_Euler_owner_bed_and_state=True,
                no_prescribed_stage_or_Q_scatter=True,status_merged_after_parent_reset=True,
                failure_latched_until_begin_trial=True,three_trace_slots_scattered_only=True,
                current_device_guard=True,raw_graph_replay_requires_check_storage=True,
                recorded_fields_are_actual_Euler_not_accepted=True,physical_validation=False,
                coupled_qualification_pending=True,production_modified=False)
    return OutletBoundary
