"""Isolated scheduled normal inlet + owner-characteristic critical outlet."""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from . import normal_coupled as normal
from . import scheduled_coupled as parent
from . import scheduled_coupled_guarded as guarded
from . import boundary_schedule as schedule
from . import normal_outlet_boundary as bridge

HERE=Path(__file__).resolve().parent
require=schedule.require
NAMES={name:private.replace('_normal_','_normal_outlet_') for name,private in normal.NAMES.items()}
NAMES.update(bridge.PRIVATE_NAMES)
STATIC_KEYS=tuple(dict.fromkeys(('nc',*normal.NORMAL_KEYS,'grav','dry','speed_limit')))
HELPERS=tuple(dict.fromkeys(('normal_outlet_coupled.py','normal_outlet_boundary.py',*normal.HELPERS,
    'outlet_boundary.py','outlet_boundary.cu','live_critical_outlet.py','live_critical_outlet.cu',
    'live_critical_outlet_cuda_v2.cu','time_integration.py')))
CUDA_NAMES=tuple(name for name in HELPERS if name.endswith('.cu'))


def transform(name,source):
    if name in bridge.PRIVATE_NAMES:return bridge.transform_bridge(name,source)
    source=normal.transform(name,source)
    source=source.replace('from .live_normal_boundary import validate_sections',
                          'from .normal_outlet_boundary import validate_sections')
    if name in ('boundaries.py','sediment_transport.py','graph_execution.py'):
        source=normal.replace_once(source,'[2,3,6,8,9]','[2,3,4,6,8,9]')
    if name=='scheduled_coupled.py':
        source=normal.replace_once(source,'[2, 3, 6, 8, 9]','[2, 3, 4, 6, 8, 9]')
    for private in normal.NAMES.values():
        source=source.replace('.'+private+' ','.'+private.replace('_normal_','_normal_outlet_')+' ')
        source=source.replace('import '+private+' ','import '+private.replace('_normal_','_normal_outlet_')+' ')
    return source


def prepare_sources(folder,manifest):
    guarded.verify_parent();bridge.verify_ancestors();folder=Path(folder).resolve()
    parent.qualified.checked_hashes({str(folder/name):value for name,value in manifest.items()})
    targets=[folder/(private+'.py') for private in NAMES.values()];target_receipt=folder/'normal_outlet_sources.json'
    require(not any(path.exists() for path in (*targets,target_receipt)),'Preserve previous combined archive')
    rows={}
    for name,private in NAMES.items():
        original=folder/name if name in manifest else HERE/name;target=folder/(private+'.py')
        target.write_text(transform(name,original.read_text()),encoding='utf-8')
        rows[name]=dict(original_path=str(original),original_sha256=parent.qualified.sha(original),
                       generated_name=target.name,generated_sha256=parent.qualified.sha(target))
    helpers={}
    for name in HELPERS:
        source=HERE/name;target=folder/('normal_outlet_ancestor_'+name)
        require(not target.exists(),'Preserve combined helper snapshot');shutil.copy2(source,target)
        helpers[name]=dict(live_path=str(source),sha256=parent.qualified.sha(source),archive=target.name)
        if name in CUDA_NAMES:
            if (folder/name).exists():
                require(parent.qualified.sha(folder/name)==parent.qualified.sha(source),'Existing CUDA snapshot differs')
            else:shutil.copy2(source,folder/name)
    receipt=dict(schema=1,rows=rows,helpers=helpers,native_controls_retained=True,
        admitted_normal_inlet_control=1,admitted_critical_outlet_mode=4,critical_inlet_control2_admitted=False,
        numeric_CUDA_unchanged=True,hydraulic_and_sediment_bodies_unchanged=True,
        additions='Explicit composition of qualified normal inlet and critical outlet; no surrogate native flags')
    parent.qualified.write_json(target_receipt,receipt)
    return dict(path=str(target_receipt),sha256=parent.qualified.sha(target_receipt),**receipt)


def verify_sources(folder,manifest):
    guarded.verify_parent();bridge.verify_ancestors();folder=Path(folder).resolve()
    r=json.loads((folder/'normal_outlet_sources.json').read_text())
    require(r['schema']==1 and set(r['rows'])==set(NAMES) and set(r['helpers'])==set(HELPERS),'Combined source set differs')
    parent.qualified.checked_hashes({str(folder/name):value for name,value in manifest.items()})
    for name,row in r['rows'].items():
        source=Path(row['original_path']);target=folder/row['generated_name'];ancestor=folder/name if name in manifest else HERE/name
        require(source.resolve()==ancestor.resolve() and row['generated_name']==NAMES[name]+'.py','Noncanonical combined source path')
        require(parent.qualified.sha(source)==row['original_sha256'] and parent.qualified.sha(target)==row['generated_sha256']
            and target.read_text()==transform(name,source.read_text()),'Generated combined module differs: '+name)
    for name,row in r['helpers'].items():
        require(Path(row['live_path']).resolve()==(HERE/name).resolve() and row['archive']=='normal_outlet_ancestor_'+name,
                'Noncanonical combined helper')
        require(parent.qualified.sha(row['live_path'])==row['sha256'] and parent.qualified.sha(folder/row['archive'])==row['sha256'],
                'Combined helper identity differs: '+name)
        if name in CUDA_NAMES:require(parent.qualified.sha(folder/name)==row['sha256'],'Combined CUDA identity differs: '+name)
    return r


def frozen_inputs(data):
    values=guarded.frozen_inputs(data)
    for key in STATIC_KEYS:
        if key not in values:continue
        array=np.asarray(values[key]);require(array.dtype.kind in 'biuf' and np.isfinite(array).all(),'Invalid combined static input '+key)
        values[key]=np.frombuffer(np.ascontiguousarray(array).tobytes(),dtype=array.dtype).reshape(array.shape)
    return values


class CombinedStorageGuard:
    def __init__(self,solver):
        self.solver=solver;self.schedule=guarded.StorageGuard(solver)
        self.manifest={key:schedule.array_identity(np.asarray(solver.data[key])) for key in STATIC_KEYS if key in solver.data}
        self.expected=self.capture()
    def capture(self):
        return dict(boundary=id(self.solver.boundary),
            data={key:guarded._host_identity(self.solver.data[key]) for key in STATIC_KEYS if key in self.solver.data})
    def check(self):
        self.schedule.check();self.solver.boundary.check_storage()
        require(self.capture()==self.expected,'Combined captured input/object changed');return True
    def audit(self):
        self.check();result=self.schedule.audit();boundary=self.solver.boundary.audit()
        require({key:schedule.array_identity(np.asarray(self.solver.data[key])) for key in self.manifest}==self.manifest,
                'Combined retained geometry/control values changed')
        return dict(schedule=result,boundary=boundary,combined_manifest=self.manifest)


@contextmanager
def constructors(folder,manifest,bindings):
    from . import coupled_solver as original_solver
    folder=Path(folder).resolve();receipt=verify_sources(folder,manifest)
    _,expected,live=parent.qualified.source_identity();require(dict(manifest)==expected,'Full qualified source manifest required')
    prefix='braids_support.gpu_fwave.';loaded=[];modules={}
    require(not any(prefix+private in sys.modules for private in NAMES.values()),'Private combined modules already loaded')
    try:
        with parent.qualified.constructors(folder,manifest,bindings):
            for name in ('live_normal_boundary.py','outlet_boundary.py','boundary_schedule.py','boundaries.py',
                         'sediment_transport.py','graph_execution.py','coupled_solver.py','coupled_graph.py','scheduled_coupled.py'):
                key=prefix+NAMES[name];spec=importlib.util.spec_from_file_location(key,folder/(NAMES[name]+'.py'))
                module=importlib.util.module_from_spec(spec);sys.modules[key]=module;loaded.append(key)
                spec.loader.exec_module(module);modules[name]=module
            core=modules['coupled_solver.py'];core.SecondaryTransport=original_solver.SecondaryTransport
            core.BoundaryControls=bridge.make_boundary_controls(modules['boundaries.py'].BoundaryControls,
                modules['live_normal_boundary.py'],modules['outlet_boundary.py'])
            ParentSolver=modules['scheduled_coupled.py']._make_solver(core.CoupledSolver,folder,receipt)
            class Solver(ParentSolver):
                def __init__(self,data,*,source_folder=folder):
                    super().__init__(frozen_inputs(data),source_folder=source_folder);self.data=frozen_inputs(self.data)
                    self.boundary.bind_sediment_hydraulic(self.sediment.geometry['bc_hydraulic'])
                    self.boundary.bind_stage_state(self.stepper.stage['q']);self._combined_guard=CombinedStorageGuard(self)
                def check_storage(self):return self._combined_guard.check()
                def check_normal_storage(self):return self.check_storage()
                def check_outlet_storage(self):return self.check_storage()
                def verify_identity(self):return self._combined_guard.audit()
                def verify_normal_identity(self):return self.verify_identity()
                def verify_outlet_identity(self):return self.verify_identity()
                def step(self,dt,current,*,commit=True):
                    self.check_storage();self.boundary.begin_trial();return super().step(dt,current,commit=commit)
                def _scheduled_transport(self,dt,current,euler,rebalance):
                    self.check_storage();return super()._scheduled_transport(dt,current,euler,rebalance)
                def euler(self,state,dt,current):
                    self.check_storage();return super().euler(state,dt,current)
                def receipt(self):
                    return dict(boundary=self.boundary.receipt(),schedule=self.schedule_receipt(),
                        sources_sha256=parent.qualified.sha(folder/'normal_outlet_sources.json'),
                        combined_metadata=self._combined_guard.manifest,physical_validation=False,coupled_qualification_pending=True)
                def normal_receipt(self):return self.receipt()
                def outlet_receipt(self):return self.receipt()
            ParentGraph=modules['coupled_graph.py'].CoupledTransportGraph
            class Graph(ParentGraph):
                def __init__(self,solver,**kwargs):
                    require(isinstance(solver,Solver),'Combined graph requires matching solver')
                    solver.check_storage();super().__init__(solver,**kwargs);solver.check_storage()
                def replay(self,dt,current,euler,rebalance):
                    s=self.solver;s.check_storage()
                    require(s._schedule_trial_open and s._schedule_clock==(float(current),float(dt)),
                            'Combined replay requires enclosing live scheduled trial')
                    return super().replay(dt,current,euler,rebalance)
                def receipt(self):
                    return dict(super().receipt(),normal_and_outlet_current_stage=True,both_hydraulic_copies_updated=True,
                        combined_storage_guarded_before_replay=True,parent_reset_once=True)
            bindings.append(dict(kind='normal_outlet_coupled_private',executed_modules={k:str(folder/(v+'.py')) for k,v in NAMES.items()},
                source_receipt_sha256=parent.qualified.sha(folder/'normal_outlet_sources.json'),original_controls_preserved=True))
            yield Solver,Graph
    finally:
        for key in reversed(loaded):sys.modules.pop(key,None)
        verify_sources(folder,manifest);parent.qualified.checked_hashes(live)
