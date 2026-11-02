"""Isolated current-source normal9/critical4 plus SSPRK43 boundary schedules.

This module is a research constructor, not a selectable runner variant. It
preserves the native controls and transforms only exact private braids_support/
constructor anchors. CUDA numerical bodies are copied byte-for-byte.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from . import a1_combined_live_scheduled_native_case as native
from . import a1_combined_one_step_native as one
from . import normal_outlet_boundary as bridge
from . import scheduled_coupled as scheduled


HERE = Path(__file__).resolve().parent
PRIVATE = {
    'boundary_schedule.py':'_a1_boundary_schedule',
    'boundaries.py':'_a1_boundaries',
    'sediment_transport.py':'_a1_sediment_transport',
    'coupled_solver.py':'_a1_coupled_solver',
    'scheduled_coupled.py':'_a1_scheduled_coupled',
    'live_normal_boundary.py':'_a1_normal_bridge',
    'outlet_boundary.py':'_a1_outlet_bridge',
}


def require(test, message):
    if not test:raise ValueError(message)


def replace_once(source, before, after, name):
    require(source.count(before)==1 and after not in source,
            'Absent/ambiguous current-source anchor '+name+': '+before[:90])
    return source.replace(before, after, 1)


def transform(name, source):
    """Exact selected-admission patch; no Euler, flux, Exner or CUDA body edit."""
    if name in bridge.PRIVATE_NAMES:
        result = bridge.transform_bridge(name,source)
        # The combined bridge validates against the detached native arrays.
        return result
    if name=='boundary_schedule.py':
        source=replace_once(source,
            "    for key in ('section_control', 'sampled'):\n        require(not np.any(data.get(key, 0)), 'Live sections are outside schedule-only scope: '+key)",
            '    validate_boundaries(data)', name)
        source=replace_once(source,'import numpy as np',
            'import numpy as np\nfrom .normal_outlet_boundary import validate_boundaries',name)
    elif name=='boundaries.py':
        source=replace_once(source,
            'supported=[2,3,6,8]+([1] if allow_bedform_mode1 else [])',
            'supported=[2,3,4,6,8,9]+([1] if allow_bedform_mode1 else [])',name)
        source=replace_once(source,
            "    for key in ('nhydro_series','section_control','sampled'):",
            "    validate_boundaries(data)\n    for key in ('nhydro_series',):",name)
        source=replace_once(source,
            'def __init__(self,data,inputs,source=None,*,allow_scheduled=False,allow_bedform_mode1=False):',
            'def __init__(self,data,inputs,source=None,*,allow_scheduled=True,allow_bedform_mode1=False):',name)
        source=replace_once(source,'import numpy as np',
            'import numpy as np\nfrom .normal_outlet_boundary import validate_boundaries',name)
    elif name=='sediment_transport.py':
        source=replace_once(source,
            'allowed_modes=[2,3,6,8]+([1] if allow_bedform else [])',
            'allowed_modes=[2,3,4,6,8,9]+([1] if allow_bedform else [])',name)
        source=replace_once(source,
            '    source=dict(data);source.update(config or {})',
            '    source=dict(data);source.update(config or {})\n    validate_boundaries(source)',name)
        source=replace_once(source,'import numpy as np',
            'import numpy as np\nfrom .normal_outlet_boundary import validate_boundaries',name)
    elif name=='coupled_solver.py':
        for before,after in (
            ('from .boundaries import BoundaryControls',
             'from ._a1_boundaries import BoundaryControls'),
            ('from .sediment_transport import ResidentSedimentTransport',
             'from ._a1_sediment_transport import ResidentSedimentTransport'),
            ("        if scheduled and key in ('nhydro_series','nfraction_series'):continue",
             "        if key in ('nhydro_series','nfraction_series'):continue"),
            ("self.boundary=BoundaryControls(bcdata,self.flux.inputs,source=self.folder/'boundaries.cu',\n                                       allow_scheduled=scheduled,allow_bedform_mode1=bedform_variant)",
             "self.boundary=BoundaryControls(bcdata,self.flux.inputs,source=self.folder/'boundaries.cu')"),
            ("        if scheduled:sediment_kwargs['allow_scheduled']=True",
             "        sediment_kwargs['allow_scheduled']=True"),
        ):
            source=replace_once(source,before,after,name)
    elif name=='scheduled_coupled.py':
        source=replace_once(source,
            'from . import boundary_schedule as schedule',
            'from . import _a1_boundary_schedule as schedule',name)
        source=replace_once(source,
            "np.isin(arrays['mode'], [2, 3, 6, 8]).all()",
            "np.isin(arrays['mode'], [2, 3, 4, 6, 8, 9]).all()",name)
        source=replace_once(source,
            '            result = super().euler(state, dt, current)',
            '            result = super().euler(state, dt, current)\n            self.boundary.record_stage(stage)',name)
    else:raise ValueError('Unknown combined candidate source: '+name)
    return source


def validate_input(data):
    """Fail closed on the exact authenticated engineering fixture before CUDA."""
    one.source_identity()
    bridge.verify_ancestors()
    require(np.asarray(data['q']).shape==(3,128), 'A1 requires exactly 128 cells')
    require(np.array_equal(data['mode'],[9,4]) and
            np.array_equal(data['section_control'],[1,0]) and
            np.array_equal(np.diff(data['boundary_offset']),[8,8]),
            'A1 requires one sampled normal9 inlet and one critical4 outlet')
    require(np.count_nonzero(data['sampled'])==8 and
            int(data['production_section_geometry'])==1,
            'A1 requires genuine surveyed inlet faces')
    require(int(data['nfrac'])==2 and int(data['nbed_layers'])==2 and
            np.array_equal(data['d50_fraction'],native.D50_M),
            'A1 requires the two distinct pinned grain classes and layers')
    require(int(data['bedlaw'])==2 and int(data['exchange_mode'])==0 and
            int(data['secondary_flow_model'])==0 and
            int(data['cpu_time_integrator'])==43 and
            int(data['mpm_friction_mode'])==1 and
            float(data['morphological_factor'])==1.,
            'A1 requires the unchanged MPM, exchange0, secondary OFF SSPRK43 controls')
    require(np.array_equal(data['nhydro_series'],[4,0]) and
            np.array_equal(data['nfraction_series'],[[4,0],[4,0]]) and
            not np.any(data['nseries']),
            'A1 requires genuine four-knot Q and per-class feed tables')
    require(np.array_equal(data['hydro_series_time'][:4,0],native.KNOTS_S) and
            np.array_equal(data['hydro_series_q'][:4,0],native.Q_M3_S) and
            np.array_equal(data['fraction_series_bed_input'][:4,:,0],native.FEED_KG_S.T),
            'A1 schedule table values differ from predeclared native fixture')
    require(not np.any(data['fraction_concentration']) and
            not np.any(data['fraction_series_concentration']) and
            not bool(data.get('suspended_transport_active',False)),
            'A1 excludes suspended inlet/exchange physics')
    for key in ('amr_enabled','bedload_adaptation','bedform_roughness','bank_failure',
                'bank_avalanche','general_avalanche','bank_front_tracking',
                'bank_erosion_fraction','horizontal_viscosity','nmorfac'):
        require(not np.any(data.get(key,0)), 'A1 excludes '+key)
    bridge.validate_boundaries(data)
    return True


def prepare_sources(folder, data):
    """Make a fresh private source archive, including byte-identical CUDA."""
    folder=Path(folder).resolve()
    require(not folder.exists(), 'Preserve earlier combined candidate archive')
    validate_input(data)
    folder.mkdir(parents=True)
    numeric={}
    for path in sorted(HERE.glob('*.cu')):
        dest=folder/path.name
        shutil.copy2(path,dest)
        numeric[path.name]=native.sha(path)
        require(native.sha(dest)==numeric[path.name], 'CUDA snapshot differs: '+path.name)
    rows={}
    for name,private in PRIVATE.items():
        original=HERE/name
        generated=folder/(private+'.py')
        generated.write_text(transform(name,original.read_text(encoding='utf-8')),
                             encoding='utf-8')
        rows[name]=dict(original_sha256=native.sha(original),
                        generated_name=generated.name,
                        generated_sha256=native.sha(generated))
    all_python={path.name:native.sha(path) for path in sorted(HERE.glob('*.py'))}
    receipt=dict(schema=1,kind='isolated_current_source_combined_boundary',
        one_step_gate_sha256=one.GATE_SHA, source_lock_sha256=native.sha(one.SOURCE_LOCK),
        one_step_native_plan_sha256=native.sha(one.ROOT/'plan.json'),
        original_python_sha256=all_python, original_cuda_sha256=numeric,
        generated_python=rows, native_controls_retained=True,
        numerical_cuda_unchanged=True, runner_selectable=False,
        Delft3D_source_modified=False)
    native.save(folder/'candidate_sources.json',receipt)
    return receipt


def verify_sources(folder):
    folder=Path(folder).resolve()
    r=json.loads((folder/'candidate_sources.json').read_text(encoding='utf-8'))
    require(r['schema']==1 and r['kind']=='isolated_current_source_combined_boundary' and
            r['one_step_gate_sha256']==one.GATE_SHA and
            r['source_lock_sha256']==native.sha(one.SOURCE_LOCK) and
            r['one_step_native_plan_sha256']==native.sha(one.ROOT/'plan.json'),
            'Combined source receipt identity differs')
    for name,digest in r['original_python_sha256'].items():
        require(native.sha(HERE/name)==digest, 'Live Python source drift: '+name)
    for name,digest in r['original_cuda_sha256'].items():
        require(native.sha(HERE/name)==digest and native.sha(folder/name)==digest,
                'CUDA source drift: '+name)
    for name,row in r['generated_python'].items():
        path=folder/row['generated_name']
        require(native.sha(HERE/name)==row['original_sha256'] and
                native.sha(path)==row['generated_sha256'] and
                path.read_text(encoding='utf-8')==transform(name,(HERE/name).read_text(encoding='utf-8')),
                'Generated private Python source drift: '+name)
    return r


@contextmanager
def constructors(folder, data):
    """Yield private Solver; never rebind a shared selectable runner alias."""
    from . import coupled_solver as live_solver
    folder=Path(folder).resolve()
    verify_sources(folder)
    validate_input(data)
    prefix='braids_support.gpu_fwave.'
    loaded=[];modules={}
    names=('boundary_schedule.py','boundaries.py','sediment_transport.py',
           'coupled_solver.py','scheduled_coupled.py',
           'live_normal_boundary.py','outlet_boundary.py')
    require(not any(prefix+PRIVATE[name] in sys.modules for name in names),
            'Private A1 modules already loaded')
    try:
        for name in names:
            key=prefix+PRIVATE[name]
            path=folder/(PRIVATE[name]+'.py')
            spec=importlib.util.spec_from_file_location(key,path)
            module=importlib.util.module_from_spec(spec)
            sys.modules[key]=module;loaded.append(key)
            spec.loader.exec_module(module);modules[name]=module
        core=modules['coupled_solver.py']
        core.SecondaryTransport=live_solver.SecondaryTransport
        core.BoundaryControls=bridge.make_boundary_controls(
            modules['boundaries.py'].BoundaryControls,
            modules['live_normal_boundary.py'],modules['outlet_boundary.py'])
        Parent=modules['scheduled_coupled.py']._make_solver(core.CoupledSolver,folder,
            dict(source_receipt_sha256=native.sha(folder/'candidate_sources.json')))
        class A1Solver(Parent):
            def __init__(self, raw, *, source_folder=folder):
                validate_input(raw)
                require(Path(source_folder).resolve()==folder, 'Private A1 source folder differs')
                native_extension=int(raw['cpu_outlet_extension'])
                require(native_extension==1, 'A1 native critical outlet extension must be 1')
                constructor=dict(raw,native_outlet_extension=native_extension,
                                 outlet_extension=0,cpu_outlet_extension=0)
                super().__init__(constructor,source_folder=folder)
                self.reconstruction.outlet_extension=1
                self.boundary.bind_sediment_hydraulic(self.sediment.geometry['bc_hydraulic'])
                self.boundary.bind_stage_state(self.stepper.stage['q'])
                # These buffers are diagnostics, not accepted state.  Giving
                # their pre-first-step value a definition makes rollback
                # deterministic without changing any numerical input.
                for records in (self.boundary.normal_records,self.boundary.outlet_records,
                                self.schedule.outputs):
                    for value in records.values():value.fill(0)
                self._a1_source=folder/'candidate_sources.json'
            def _a1_trial_diagnostics(self):
                return dict(clock=self.schedule.clock.copy(),
                    schedule={key:value.copy() for key,value in self.schedule.outputs.items()},
                    normal={key:value.copy() for key,value in self.boundary.normal_records.items()},
                    outlet={key:value.copy() for key,value in self.boundary.outlet_records.items()},
                    normal_latch=self.boundary._normal_latch.copy(),
                    outlet_latch=self.boundary._outlet_latch.copy())
            def _a1_restore_diagnostics(self, saved, *, where=True):
                cp=self.cp
                cp.copyto(self.schedule.clock,saved['clock'],where=where)
                for key,value in saved['schedule'].items():
                    cp.copyto(self.schedule.outputs[key],value,where=where)
                for key,value in saved['normal'].items():
                    cp.copyto(self.boundary.normal_records[key],value,where=where)
                for key,value in saved['outlet'].items():
                    cp.copyto(self.boundary.outlet_records[key],value,where=where)
                cp.copyto(self.boundary._normal_latch,saved['normal_latch'],where=where)
                cp.copyto(self.boundary._outlet_latch,saved['outlet_latch'],where=where)
            def step(self,dt,current,*,commit=True):
                self.boundary.check_storage()
                saved=self._a1_trial_diagnostics()
                try:
                    self.boundary.begin_trial()
                    result=super().step(dt,current,commit=commit)
                    # Device scalar mask: no D2H status read or Python branch on
                    # every successful production-sized timestep.
                    self._a1_restore_diagnostics(saved,where=(result['status']!=0) if commit else True)
                    return result
                except Exception:
                    self._a1_restore_diagnostics(saved)
                    raise
            def candidate_receipt(self):
                self.boundary.check_storage()
                return dict(source_sha256=native.sha(self._a1_source),
                    boundary=self.boundary.receipt(),schedule=self.schedule_receipt(),
                    integrator=self.integrator,native_controls_retained=True,
                    runner_selectable=False,physical_validation=False)
        yield A1Solver
    finally:
        for key in reversed(loaded):sys.modules.pop(key,None)
        verify_sources(folder)
