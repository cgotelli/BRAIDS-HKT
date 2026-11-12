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
 'live_normal_section.cu':'999d854b5f262f5f0752d6ba928456d133fa2707e7a58f02c08e7bde71e2dd51',
 'live_normal_trace.py':'3639789161b491565edebe80b1d87f3624ca08c2f7aa9bb91464a16ed1101d95',
 'live_normal_trace.cu':'4207923ca4bb67611406f4fd5b4ea61a54b76f81212805d941950a7c1e11d474'}
NAMES=('lnb_gather','lnb_stage','lnb_scatter','lnb_merge','lnb_record')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_sections(data):
    from .normal_outlet_boundary import validate_sections as validate_actual
    return validate_actual(data)


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
            self.normal_records=dict(bed=cp.empty((4,self.nc)),nodes=cp.empty((4,nnodes)),
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
            values += [(prefix+k,v) for prefix,mapping in (('parent:',self.arrays),('records:',self.normal_records)) for k,v in mapping.items()]
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

        def _bind_stage_state_normal(self,q):
            self._normal_bind('_normal_q',q,(3,self.nc));self._normal_finish_binding()

        def _normal_finish_binding(self):
            if self._normal_sediment is None or self._normal_q is None:return
            self.normal_trace=trace_component.ResidentNormalTrace(self._normal_geometry,self.inputs['bed'],self._normal_q,
                self.normal_root.outputs['stage'],self.normal_root.outputs['status'],self._normal_Q,
                dry=float(self.dry),speed_limit=float(self.speed),source_path=self._normal_bridge_source.parent/'live_normal_trace.cu')
            self._normal_ready_storage=self._normal_storage_identity()
            self._normal_scalar_identity=tuple(float(getattr(self,k)) for k in ('dry','grav','speed','chezy'))

        def _check_storage_normal(self):
            require(self.cp.cuda.Device().id==self._normal_device,'Current CUDA device differs from normal bridge')
            require(self._normal_ready_storage is not None,'Bind stage-state and sediment hydraulics before normal apply')
            require(self._normal_storage_identity()==self._normal_ready_storage,'Captured normal bridge storage changed')
            require(tuple(float(getattr(self,k)) for k in ('dry','grav','speed','chezy'))==self._normal_scalar_identity,
                    'Captured normal bridge controls changed')
            self.normal_root.check_storage();self.normal_trace.check_storage()
            return True

        def _begin_trial_normal(self):
            self._check_storage_normal();self._normal_latch.fill(0);self.normal_records['recorded'].fill(0)

        def _apply_normal(self,q,trace):
            self._check_storage_normal()
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
            self._apply_outlet(q,trace)
            values=self.normal_trace.evaluate();faces=self.normal_root._arrays['faces'];count=np.int32(len(faces))
            self._normal_kernels['lnb_scatter'](((len(faces)+127)//128,),(128,),
                (faces,count,np.int32(self.nf),values['trace'],values['face_status'],trace,self.face_status))
            self._normal_kernels['lnb_merge'](grid,(128,),
                (self._normal_ids,n,root['status'],values['status'],self.status,self._normal_latch))

        def _record_stage_normal(self,index):
            self._check_storage_normal();require(type(index) is int and 0<=index<4,'Four SSPRK43 stage-record indices only')
            cp=self.cp;r=self.normal_root.outputs;t=self.normal_trace.outputs;s=self.normal_records
            for key,value in (('bed',self.inputs['bed']),('nodes',r['nodes']),('stage',r['stage']),('discharge',self._normal_Q),
                ('root_status',r['status']),('trace_status',t['status']),('failure_latch',self._normal_latch)):
                cp.copyto(s[key][index],value)
            for key in ('cache_hit','bisections','metric_evaluations'):cp.copyto(s[key][index],r[key])
            faces=self.normal_root._arrays['faces'];self._normal_kernels['lnb_record'](((len(faces)+127)//128,),(128,),
                (faces,np.int32(len(faces)),np.int32(self.nf),self.inputs['trace'],s['trace'][index]))
            s['recorded'][index:index+1].fill(1)

        def _audit_normal(self):
            self._check_storage_normal();self.normal_root.verify_geometry();self.normal_trace.verify_geometry()
            require(all(identity(self.cp.asnumpy(self.inputs[k]))==v for k,v in self._normal_static_device.items()),
                    'Static coupled geometry bytes changed')
            require(all(identity(self.cp.asnumpy(self.arrays[k]))==v for k,v in self._normal_parent_static.items()),
                    'Immutable parent boundary arrays changed')
            require(sha(__file__)==self._normal_python_sha and sha(self._normal_bridge_source)==self._normal_bridge_sha
                and sha(self._normal_parent_source)==self._normal_parent_sha and sha(self._normal_parent_python)==self._normal_parent_python_sha,
                'Executed normal bridge/parent source changed')
            return True

        def _receipt_normal(self):
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
