"""Private mode9 NORMAL adapter over an explicitly scope-extended parent.

No original validators/kernels are edited here. The caller supplies a privately
generated ParentControls whose own validator admits the reviewed mode9 scope.
Bind the ONE stable SSPRK43 stage-q and sediment hydraulic buffer after solver
construction, before warmup/capture. No state-field copy or CPU root fallback.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
from types import MappingProxyType
import numpy as np
from . import live_normal_section as root_component
from . import live_normal_trace as trace_component
from .time_integration import OPTIONS

require=trace_component.require;freeze=trace_component.freeze;identity=trace_component.identity
PINNED={
 'live_normal_section.py':'b3615c811a1d002da4a9bfdecab32ab56e3353e7e737ed31c0a914e10e030b1f',
 'live_normal_section.cu':'c918345899bcf178a8cafd9beca2e06c856e96383fef7a734167e7333f831dec',
 'live_normal_trace.py':'3639789161b491565edebe80b1d87f3624ca08c2f7aa9bb91464a16ed1101d95',
 'live_normal_trace.cu':'4207923ca4bb67611406f4fd5b4ea61a54b76f81212805d941950a7c1e11d474'}
NAMES=('lnb_gather','lnb_stage','lnb_scatter','lnb_merge','lnb_record')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sections(data):
    """Detach exact native arrays, retaining global face/owner/boundary IDs."""
    def vector(key,dtype,shape=None):
        value=np.asarray(data[key])
        require(value.dtype==dtype and value.ndim==1 and value.flags.c_contiguous,
                'Canonical native vector required: '+key)
        if shape is not None:require(value.shape==shape,'Native vector shape differs: '+key)
        return value
    mode=vector('mode',np.int32);nb=len(mode)
    require(np.isin(mode,[2,3,6,8,9]).all(),'Only existing modes2/3/6/8 and NORMAL9 supported')
    flag=np.asarray(data['production_section_geometry'])
    require(flag.shape==() and flag.dtype.kind in 'biu' and int(flag)==1,
            'Explicit production_section_geometry=True required')
    owner=vector('owner',np.int32);nf=len(owner)
    neighbour=vector('neighbour',np.int32,(nf,));bid=vector('bid',np.int32,(nf,))
    offsets=vector('boundary_offset',np.int32,(nb+1,));faces=vector('boundary_faces',np.int32)
    require(offsets[0]==0 and offsets[-1]==len(faces) and np.all(np.diff(offsets.astype(np.int64))>=0),
            'Native boundary CSR offsets invalid')
    require(np.all((faces>=0)&(faces<nf)) and len(np.unique(faces))==len(faces),
            'Assigned physical faces must be valid and unique')
    require(np.all((bid>=-1)&(bid<nb)) and np.array_equal(np.sort(faces),np.flatnonzero(bid>=0)),
            'Boundary CSR must cover every assigned open face once')
    for b in range(nb):
        ff=faces[offsets[b]:offsets[b+1]]
        require(np.all(bid[ff]==b) and np.all(neighbour[ff]<0),'Boundary CSR/owner exterior identity differs')
    control=vector('section_control',np.int32,(nb,));seeds=vector('section_seed_node',np.int32,(nb,))
    face_seeds=vector('boundary_seed_position',np.int32,(nb,))
    slope=vector('section_energy_slope',np.float64,(nb,))
    selected=np.flatnonzero(mode==9).astype(np.int32)
    require(len(selected)>0,'At least one explicit NORMAL mode9 boundary required')
    require(np.all(control[selected]==1) and np.all(control[mode!=9]==0),
            'Only NORMAL sectioncontrol1 on mode9; other modes must have no section controls')
    sampled=vector('sampled',np.int32,(nf,))
    expected=np.zeros(nf,np.int32)
    for b in selected:
        count=int(offsets[b+1]-offsets[b]);ff=faces[offsets[b]:offsets[b+1]];expected[ff]=1
        require(1<=seeds[b]<count,'Zero-based initially interior section_seed_node required')
        require(face_seeds[b]==-1 or offsets[b]<=face_seeds[b]<offsets[b+1],
                'Zero-based global boundary_seed_position outside assigned CSR interval')
    require(np.array_equal(sampled,expected),'Sampled flags must identify exactly every NORMAL face only')
    compact_faces=np.concatenate([faces[offsets[b]:offsets[b+1]] for b in selected]).astype(np.int32)
    compact_offsets=np.r_[0,np.cumsum([offsets[b+1]-offsets[b] for b in selected])].astype(np.int32)
    g=dict(owner=owner,faces=compact_faces,offsets=compact_offsets,length=data['length'],normal=data['normal'],
        left_offset=data['boundary_bed_left_offset'],right_offset=data['boundary_bed_right_offset'],
        mid_offset=data['boundary_bed_offset'],manning=data['manning_cell'],production_section_geometry=True)
    g=trace_component.validate_geometry(g)
    if 'manning' in data:require(np.array_equal(data['manning'],data['manning_cell']),'Hydraulic/native Manning differs')
    if 'bedoffset' in data:require(np.array_equal(data['bedoffset'],data['boundary_bed_offset']),'Pressure/section bed offsets differ')
    controls=root_component.validate_controls(g,dict(mode=mode[selected].copy(),section_control=control[selected].copy(),
        energy_slope=slope[selected].copy(),seed_node=seeds[selected].copy()))
    return dict(geometry=g,controls=controls,global_boundary_ids=freeze(selected),
        native_metadata=MappingProxyType({key:freeze(np.asarray(data[key])) for key in
            ('mode','section_control','section_seed_node','boundary_seed_position','boundary_offset','boundary_faces',
             'bid','neighbour','sampled','section_energy_slope')}))


def make_boundary_controls(ParentControls):
    class LiveNormalBoundary(ParentControls):
        def __init__(self,data,inputs,source=None):
            sections=validate_sections(data)
            require(source is not None,'Explicit archived parent boundary CUDA source required')
            folder=Path(source).resolve().parent
            for name,digest in PINNED.items():
                path=Path(root_component.__file__) if name=='live_normal_section.py' else Path(trace_component.__file__) if name=='live_normal_trace.py' else folder/name
                require(sha(path)==digest,'Qualified normal component changed: '+name)
            super().__init__(data,inputs,source=source)
            cp=self.cp;self._normal_device=cp.cuda.Device().id;self.sections=sections
            self._normal_ids=cp.asarray(sections['global_boundary_ids']);self._normal_n=len(self._normal_ids)
            self._normal_Q=cp.zeros(self._normal_n);self._normal_sediment=None;self._normal_q=None
            self.normal_trace=None;self._normal_ready_storage=None
            self.normal_root=root_component.ResidentNormalSection(
                dict(sections['geometry'],production_section_geometry=True),sections['controls'],inputs['bed'],self._normal_Q,
                source_path=folder/'live_normal_section.cu')
            self._normal_geometry=dict(sections['geometry'],production_section_geometry=True)
            self._normal_latch=cp.zeros(self._normal_n,dtype=cp.int32)
            self._normal_bridge_source=folder/'live_normal_boundary.cu'
            self._normal_bridge_sha=sha(self._normal_bridge_source);self._normal_python_sha=sha(__file__)
            self._normal_parent_source=Path(source).resolve();self._normal_parent_sha=sha(source)
            self._normal_parent_python=Path(inspect.getfile(ParentControls)).resolve()
            self._normal_parent_python_sha=sha(self._normal_parent_python)
            self._normal_module=cp.RawModule(code=self._normal_bridge_source.read_text(),options=OPTIONS,name_expressions=NAMES)
            self._normal_kernels={name:self._normal_module.get_function(name) for name in NAMES}
            nfaces=len(sections['geometry']['faces']);nnodes=nfaces+self._normal_n
            self.stage_records=dict(bed=cp.empty((4,self.nc)),nodes=cp.empty((4,nnodes)),
                stage=cp.empty((4,self._normal_n)),discharge=cp.empty((4,self._normal_n)),
                trace=cp.empty((4,3,nfaces)),root_status=cp.empty((4,self._normal_n),dtype=cp.int32),
                trace_status=cp.empty((4,self._normal_n),dtype=cp.int32),
                failure_latch=cp.empty((4,self._normal_n),dtype=cp.int32),
                cache_hit=cp.empty((4,self._normal_n),dtype=cp.int32),
                bisections=cp.empty((4,self._normal_n),dtype=cp.int32),
                metric_evaluations=cp.empty((4,self._normal_n),dtype=cp.int32),recorded=cp.zeros(4,dtype=cp.int32))
            self._normal_parent_static={k:identity(cp.asnumpy(v)) for k,v in self.arrays.items() if k!='forcing'}
            self._normal_host_inputs={k:identity(np.asarray(data[k])) for k in data
                if isinstance(data[k],np.ndarray) and data[k].dtype.kind in 'biuf'}
            self._normal_static_device={}
            for key,host_key in (('owner','owner'),('normal','normal'),('length','length'),
                                 ('sampled','sampled'),('bedoffset','boundary_bed_offset')):
                require(key in inputs,'Actual coupled geometry buffer missing: '+key)
                actual=cp.asnumpy(inputs[key]);expected=np.asarray(data[host_key])
                require(actual.dtype==expected.dtype and np.array_equal(actual,expected),
                        'Actual coupled geometry differs from native input: '+key)
                self._normal_static_device[key]=identity(actual)
            require('trace' in inputs,'Actual reconstruction trace buffer must be bound at construction')
            # Exact pointer binding is audited before any capture; no field copies in apply.
            self._normal_initial_storage=self._normal_storage_identity()

        def _normal_storage_identity(self):
            values=[('normal_ids',self._normal_ids),('normal_Q',self._normal_Q),('normal_latch',self._normal_latch),
                ('status',self.status),('face_status',self.face_status),('ledger',self.ledger)]
            values += [(prefix+k,v) for prefix,mapping in (('parent:',self.arrays),('records:',self.stage_records)) for k,v in mapping.items()]
            values += [('input:'+k,self.inputs[k]) for k in ('bed','normal','length','owner','sampled','bedoffset','trace')]
            if self._normal_sediment is not None:values.append(('sediment',self._normal_sediment))
            if self._normal_q is not None:values.append(('q',self._normal_q))
            return tuple((k,id(v),v.data.ptr,v.shape,v.dtype.str,bool(v.flags.c_contiguous),v.device.id) for k,v in values)

        def _normal_bind(self,key,value,shape):
            cp=self.cp
            require(not cp.cuda.get_current_stream().is_capturing(),'Bind normal buffers before CUDA capture')
            require(self._normal_ready_storage is None,'Normal buffers already sealed')
            require(tuple(row for row in self._normal_storage_identity() if row[0] not in ('sediment','q'))
                ==self._normal_initial_storage,'Normal construction storage changed before binding')
            require(getattr(self,key) is None,'Normal buffer can be bound only once')
            require(isinstance(value,cp.ndarray) and value.shape==shape and value.dtype==cp.float64
                and value.flags.c_contiguous and value.device.id==self._normal_device,'Canonical resident normal buffer required')
            setattr(self,key,value)

        def bind_sediment_hydraulic(self,device):
            self._normal_bind('_normal_sediment',device,(2,self.nb));self._normal_finish_binding()

        def bind_stage_state(self,q):
            self._normal_bind('_normal_q',q,(3,self.nc));self._normal_finish_binding()

        def _normal_finish_binding(self):
            if self._normal_sediment is None or self._normal_q is None:return
            self.normal_trace=trace_component.ResidentNormalTrace(self._normal_geometry,self.inputs['bed'],self._normal_q,
                self.normal_root.outputs['stage'],self.normal_root.outputs['status'],self._normal_Q,
                dry=float(self.dry),speed_limit=float(self.speed),source_path=self._normal_bridge_source.parent/'live_normal_trace.cu')
            self._normal_ready_storage=self._normal_storage_identity()
            self._normal_scalar_identity=tuple(float(getattr(self,k)) for k in ('dry','grav','speed','chezy'))

        def check_storage(self):
            require(self.cp.cuda.Device().id==self._normal_device,'Current CUDA device differs from normal bridge')
            require(self._normal_ready_storage is not None,'Bind stage-state and sediment hydraulics before normal apply')
            require(self._normal_storage_identity()==self._normal_ready_storage,'Captured normal bridge storage changed')
            require(tuple(float(getattr(self,k)) for k in ('dry','grav','speed','chezy'))==self._normal_scalar_identity,
                    'Captured normal bridge controls changed')
            self.normal_root.check_storage();self.normal_trace.check_storage()
            return True

        def begin_trial(self):
            self.check_storage();self._normal_latch.fill(0);self.stage_records['recorded'].fill(0)

        def apply(self,q,trace):
            self.check_storage()
            require(q is self._normal_q,'Normal bridge requires the bound stable SSPRK43 stage q')
            require(trace is self.inputs['trace'],'Normal bridge requires the bound reconstruction trace')
            require(isinstance(trace,self.cp.ndarray) and trace.shape==(10,self.nf) and trace.dtype==self.cp.float64
                and trace.flags.c_contiguous and trace.device.id==self._normal_device,'Native ten-slot reconstruction trace required')
            n=np.int32(self._normal_n);grid=((self._normal_n+127)//128,)
            self._normal_kernels['lnb_gather'](grid,(128,),(self.arrays['forcing'],self._normal_ids,n,self._normal_Q))
            root=self.normal_root.solve()
            self._normal_kernels['lnb_stage'](grid,(128,),(root['stage'],root['status'],self._normal_ids,n,np.int32(self.nb),
                self.arrays['forcing'],self._normal_sediment))
            # Parent resets statuses and processes only its unchanged existing modes.
            super().apply(q,trace)
            values=self.normal_trace.evaluate();faces=self.normal_root._arrays['faces'];count=np.int32(len(faces))
            self._normal_kernels['lnb_scatter'](((len(faces)+127)//128,),(128,),
                (faces,count,np.int32(self.nf),values['trace'],values['face_status'],trace,self.face_status))
            self._normal_kernels['lnb_merge'](grid,(128,),
                (self._normal_ids,n,root['status'],values['status'],self.status,self._normal_latch))

        def record_stage(self,index):
            self.check_storage();require(type(index) is int and 0<=index<4,'Four SSPRK43 stage-record indices only')
            cp=self.cp;r=self.normal_root.outputs;t=self.normal_trace.outputs;s=self.stage_records
            for key,value in (('bed',self.inputs['bed']),('nodes',r['nodes']),('stage',r['stage']),('discharge',self._normal_Q),
                ('root_status',r['status']),('trace_status',t['status']),('failure_latch',self._normal_latch)):
                cp.copyto(s[key][index],value)
            for key in ('cache_hit','bisections','metric_evaluations'):cp.copyto(s[key][index],r[key])
            faces=self.normal_root._arrays['faces'];self._normal_kernels['lnb_record'](((len(faces)+127)//128,),(128,),
                (faces,np.int32(len(faces)),np.int32(self.nf),self.inputs['trace'],s['trace'][index]))
            s['recorded'][index:index+1].fill(1)

        def audit(self):
            self.check_storage();self.normal_root.verify_geometry();self.normal_trace.verify_geometry()
            require(all(identity(self.cp.asnumpy(self.inputs[k]))==v for k,v in self._normal_static_device.items()),
                    'Static coupled geometry bytes changed')
            require(all(identity(self.cp.asnumpy(self.arrays[k]))==v for k,v in self._normal_parent_static.items()),
                    'Immutable parent boundary arrays changed')
            require(sha(__file__)==self._normal_python_sha and sha(self._normal_bridge_source)==self._normal_bridge_sha
                and sha(self._normal_parent_source)==self._normal_parent_sha and sha(self._normal_parent_python)==self._normal_parent_python_sha,
                'Executed normal bridge/parent source changed')
            return True

        def receipt(self):
            return dict(backend='gpu',automatic_CPU_fallback=False,production_guard_changed=False,
                source_sha256=self._normal_bridge_sha,python_sha256=self._normal_python_sha,
                parent_cuda_sha256=self._normal_parent_sha,parent_python_sha256=self._normal_parent_python_sha,
                global_boundary_ids=self.sections['global_boundary_ids'].tolist(),native_input_identities=self._normal_host_inputs,
                normal_root=self.normal_root.receipt(),normal_trace=self.normal_trace.receipt() if self.normal_trace else None,
                single_stable_q_binding=True,per_stage_state_field_copy=False,
                normal_status_merged_after_parent_reset=True,normal_failure_latched_until_begin_trial=True,
                three_trace_slots_scattered_only=True,sampled_pressure_flux_body_unchanged=True,
                stage_records='Four slots of current bed/nodes/stage/Q/assigned trace/status/cache_hit/bisections/metric_evaluations, not accepted outputs',
                current_device_guard=True,raw_graph_replay_requires_check_storage=True,
                fixed_pointer_audit_required_before_capture=True,integrated_qualification_pending=True)
    return LiveNormalBoundary
