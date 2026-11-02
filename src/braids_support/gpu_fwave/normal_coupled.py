"""Private live-normal inlet extension; frozen production/qualified files stay intact.

Exact-anchor copies admit only surveyed normal mode9/control1, alongside the
already supported modes2/3/6/8. The normal root executes after the current Euler
layer-to-bed operation. No physics flag is cleared to construct a surrogate.
"""
from __future__ import annotations
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from . import scheduled_coupled as parent
from . import scheduled_coupled_guarded as guarded
from . import boundary_schedule as schedule
from . import live_normal_boundary as bridge

HERE=Path(__file__).resolve().parent
require=schedule.require
NAMES={name:private.replace('_m2_','_normal_') for name,private in parent.NAMES.items()}
NAMES.update({'boundary_schedule.py':'_normal_schedule','scheduled_coupled.py':'_normal_scheduling'})
NORMAL_KEYS=('owner','neighbour','bid','boundary_offset','boundary_faces','normal','length',
             'bedoffset','boundary_bed_offset','boundary_bed_left_offset','boundary_bed_right_offset',
             'manning_cell','manning','section_energy_slope','section_seed_node','boundary_seed_position',
             'production_section_geometry')
HELPERS=('normal_coupled.py','live_normal_boundary.py','live_normal_boundary.cu',
         'live_normal_section.py','live_normal_section.cu','live_normal_trace.py','live_normal_trace.cu',
         'live_section_metrics.py','scheduled_coupled.py','scheduled_coupled_guarded.py',
         'boundary_schedule.py','boundary_schedule.cu')


def replace_once(source,before,after):
    require(source.count(before)==1 and after not in source,'Absent/ambiguous/already transformed normal anchor: '+before)
    return source.replace(before,after,1)


def transform(name,source):
    if name in parent.NAMES:
        source=parent.transform(name,source)
        if name=='boundaries.py':
            before="    for key in ('section_control','sampled'):\n        if key in data and np.any(data[key]):\n            raise ValueError('GPU boundary port excludes '+key)"
            source=replace_once(source,before,'    validate_sections(data)')
            source=replace_once(source,'np.isin(modes,[2,3,6,8])','np.isin(modes,[2,3,6,8,9])')
            source=replace_once(source,'np.isin(modes,[6,8])','np.isin(modes,[6,8,9])')
            source=replace_once(source,'from pathlib import Path','from pathlib import Path\nfrom .live_normal_boundary import validate_sections')
        elif name=='sediment_transport.py':
            source=replace_once(source,'np.isin(c[\'mode\'],[2,3,6,8])','np.isin(c[\'mode\'],[2,3,6,8,9])')
            source=replace_once(source,'    source=dict(data);source.update(config or {})',
                '    source=dict(data);source.update(config or {})\n    validate_sections(source)')
            source=replace_once(source,'from pathlib import Path','from pathlib import Path\nfrom .live_normal_boundary import validate_sections')
        elif name=='graph_execution.py':
            source=replace_once(source,'np.isin(np.asarray(data["mode"]),[2,3,6,8])','np.isin(np.asarray(data["mode"]),[2,3,6,8,9])')
            source=replace_once(source,'for key in ("nseries","nmorfac","section_control","sampled"):',
                'for key in ("nseries","nmorfac"):')
            source=replace_once(source,'    if not np.isfinite(dt) or dt<=0:',
                '    validate_sections(data)\n    if not np.isfinite(dt) or dt<=0:')
            source=replace_once(source,'import numpy as np','import numpy as np\nfrom .live_normal_boundary import validate_sections')
        for private in parent.NAMES.values():source=source.replace('.'+private+' ','.'+private.replace('_m2_','_normal_')+' ')
    elif name=='boundary_schedule.py':
        before="    for key in ('section_control', 'sampled'):\n        require(not np.any(data.get(key, 0)), 'Live sections are outside schedule-only scope: '+key)"
        source=replace_once(source,before,'    validate_sections(data)')
        source=replace_once(source,'import numpy as np','import numpy as np\nfrom .live_normal_boundary import validate_sections')
    elif name=='scheduled_coupled.py':
        source=replace_once(source,'from . import boundary_schedule as schedule','from . import _normal_schedule as schedule')
        source=replace_once(source,"np.isin(arrays['mode'], [2, 3, 6, 8])","np.isin(arrays['mode'], [2, 3, 6, 8, 9])")
        source=replace_once(source,'            result = super().euler(state, dt, current)',
            '            result = super().euler(state, dt, current)\n            self.boundary.record_stage(stage)')
    else:raise ValueError('Unknown private normal source '+name)
    return source


def prepare_sources(folder,manifest):
    guarded.verify_parent();folder=Path(folder).resolve()
    parent.qualified.checked_hashes({str(folder/name):value for name,value in manifest.items()})
    targets=[folder/(private+'.py') for private in NAMES.values()]
    receipt_path=folder/'normal_sources.json'
    require(not any(path.exists() for path in (*targets,receipt_path)),'Preserve previous normal archive')
    rows={}
    for name,private in NAMES.items():
        original=folder/name if name in manifest else HERE/name
        require(original.exists(),'Missing normal ancestor '+name)
        target=folder/(private+'.py');target.write_text(transform(name,original.read_text()),encoding='utf-8')
        rows[name]=dict(original_path=str(original),original_sha256=parent.qualified.sha(original),
                       generated_name=target.name,generated_sha256=parent.qualified.sha(target))
    helpers={}
    for name in HELPERS:
        source=HERE/name;target=folder/('normal_ancestor_'+name)
        require(not target.exists(),'Preserve helper snapshot '+name);shutil.copy2(source,target)
        helpers[name]=dict(live_path=str(source),sha256=parent.qualified.sha(source),archive=target.name)
        # Numeric CUDA filenames are selected explicitly by bridge constructors.
        if name in ('live_normal_boundary.cu','live_normal_section.cu','live_normal_trace.cu','boundary_schedule.cu'):
            require(not (folder/name).exists(),'Preserve CUDA snapshot '+name);shutil.copy2(source,folder/name)
    receipt=dict(schema=1,rows=rows,helpers=helpers,native_controls_retained=True,
        critical_controls_admitted=False,numerical_core_changed=False,
        additions='Current-stage normal root and mode9 trace, stage scatter to both forcing copies; explicit scope guards and storage guards')
    parent.qualified.write_json(receipt_path,receipt)
    return dict(path=str(receipt_path),sha256=parent.qualified.sha(receipt_path),**receipt)


def verify_sources(folder,manifest):
    guarded.verify_parent();folder=Path(folder).resolve()
    receipt=json.loads((folder/'normal_sources.json').read_text())
    require(receipt['schema']==1 and set(receipt['rows'])==set(NAMES)
        and set(receipt['helpers'])==set(HELPERS),'Normal source set differs')
    parent.qualified.checked_hashes({str(folder/name):value for name,value in manifest.items()})
    for name,row in receipt['rows'].items():
        source=Path(row['original_path']);target=folder/row['generated_name']
        ancestor=folder/name if name in manifest else HERE/name
        require(source.resolve()==ancestor.resolve() and row['generated_name']==NAMES[name]+'.py',
                'Noncanonical normal executed filename/ancestor: '+name)
        require(parent.qualified.sha(source)==row['original_sha256']
            and parent.qualified.sha(target)==row['generated_sha256']
            and target.read_text()==transform(name,source.read_text()),'Normal generated source differs: '+name)
    for name,row in receipt['helpers'].items():
        require(Path(row['live_path']).resolve()==(HERE/name).resolve()
            and row['archive']=='normal_ancestor_'+name,'Noncanonical normal helper selection')
        require(parent.qualified.sha(row['live_path'])==row['sha256']
            and parent.qualified.sha(folder/row['archive'])==row['sha256'],'Normal helper identity differs: '+name)
        if name in ('live_normal_boundary.cu','live_normal_section.cu','live_normal_trace.cu','boundary_schedule.cu'):
            require(parent.qualified.sha(folder/name)==row['sha256'],'Normal CUDA differs: '+name)
    return receipt


def frozen_inputs(data):
    values=guarded.frozen_inputs(data)
    for key in NORMAL_KEYS:
        if key not in values:continue
        array=np.asarray(values[key]);require(array.dtype.kind in 'biuf' and np.isfinite(array).all(),'Invalid normal static input '+key)
        values[key]=np.frombuffer(np.ascontiguousarray(array).tobytes(),dtype=array.dtype).reshape(array.shape)
    return values


class NormalStorageGuard:
    def __init__(self,solver):
        self.solver=solver;self.schedule=guarded.StorageGuard(solver)
        self.manifest={key:schedule.array_identity(np.asarray(solver.data[key])) for key in NORMAL_KEYS if key in solver.data}
        self.expected=self.capture()
    def capture(self):
        return dict(boundary=id(self.solver.boundary),
            data={key:guarded._host_identity(self.solver.data[key]) for key in NORMAL_KEYS if key in self.solver.data})
    def check(self):
        self.schedule.check();self.solver.boundary.check_storage()
        require(self.capture()==self.expected,'Normal captured host input/object identity changed')
        return True
    def audit(self):
        self.check();result=self.schedule.audit();boundary=self.solver.boundary.audit()
        require({key:schedule.array_identity(np.asarray(self.solver.data[key])) for key in self.manifest}==self.manifest,
                'Normal retained section metadata values changed')
        return dict(schedule=result,boundary=boundary,normal_manifest=self.manifest)


@contextmanager
def constructors(folder,manifest,bindings):
    from . import coupled_solver as original_solver
    folder=Path(folder).resolve();receipt=verify_sources(folder,manifest)
    _,expected,live=parent.qualified.source_identity()
    require(dict(manifest)==expected,'Full qualified baseline manifest required')
    prefix='braids_support.gpu_fwave.';loaded=[];modules={}
    require(not any(prefix+private in sys.modules for private in NAMES.values()),'Private normal modules already loaded')
    try:
        with parent.qualified.constructors(folder,manifest,bindings):
            for name in ('boundary_schedule.py','boundaries.py','sediment_transport.py','graph_execution.py',
                         'coupled_solver.py','coupled_graph.py','scheduled_coupled.py'):
                private=NAMES[name];key=prefix+private
                spec=importlib.util.spec_from_file_location(key,folder/(private+'.py'))
                module=importlib.util.module_from_spec(spec);sys.modules[key]=module;loaded.append(key)
                spec.loader.exec_module(module);modules[name]=module
            core=modules['coupled_solver.py']
            core.SecondaryTransport=original_solver.SecondaryTransport
            core.BoundaryControls=bridge.make_boundary_controls(modules['boundaries.py'].BoundaryControls)
            ParentSolver=modules['scheduled_coupled.py']._make_solver(core.CoupledSolver,folder,receipt)
            class NormalSolver(ParentSolver):
                def __init__(self,data,*,source_folder=folder):
                    super().__init__(frozen_inputs(data),source_folder=source_folder)
                    self.data=frozen_inputs(self.data)
                    self.boundary.bind_sediment_hydraulic(self.sediment.geometry['bc_hydraulic'])
                    self.boundary.bind_stage_state(self.stepper.stage['q'])
                    self._normal_guard=NormalStorageGuard(self)
                def check_normal_storage(self):return self._normal_guard.check()
                def verify_normal_identity(self):return self._normal_guard.audit()
                def step(self,dt,current,*,commit=True):
                    self.check_normal_storage();self.boundary.begin_trial()
                    return super().step(dt,current,commit=commit)
                def _scheduled_transport(self,dt,current,euler,rebalance):
                    self.check_normal_storage()
                    return super()._scheduled_transport(dt,current,euler,rebalance)
                def euler(self,state,dt,current):
                    self.check_normal_storage()
                    return super().euler(state,dt,current)
                def normal_receipt(self):
                    return dict(boundary=self.boundary.receipt(),schedule=self.schedule_receipt(),
                                sources_sha256=parent.qualified.sha(folder/'normal_sources.json'),
                                normal_metadata=self._normal_guard.manifest,
                                physical_validation=False,coupled_qualification_pending=True)
            ParentGraph=modules['coupled_graph.py'].CoupledTransportGraph
            class NormalGraph(ParentGraph):
                def __init__(self,solver,**kwargs):
                    require(isinstance(solver,NormalSolver),'Normal graph requires matching solver')
                    solver.check_normal_storage();super().__init__(solver,**kwargs);solver.check_normal_storage()
                def replay(self,dt,current,euler,rebalance):
                    solver=self.solver;solver.check_normal_storage()
                    require(solver._schedule_trial_open and solver._schedule_clock==(float(current),float(dt)),
                            'Normal replay requires enclosing live scheduled trial')
                    return super().replay(dt,current,euler,rebalance)
                def receipt(self):
                    return dict(super().receipt(),normal_current_stage_bed=True,
                        both_hydraulic_copies_updated=True,normal_storage_guarded_before_replay=True)
            bindings.append(dict(kind='normal_coupled_private',executed_modules={k:str(folder/(v+'.py')) for k,v in NAMES.items()},
                source_receipt_sha256=parent.qualified.sha(folder/'normal_sources.json'),original_controls_preserved=True))
            yield NormalSolver,NormalGraph
    finally:
        for key in reversed(loaded):sys.modules.pop(key,None)
        verify_sources(folder,manifest);parent.qualified.checked_hashes(live)
