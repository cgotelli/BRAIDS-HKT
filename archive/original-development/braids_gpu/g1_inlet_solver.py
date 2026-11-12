"""Coupled GPU SSPRK33 composition for surveyed stage/critical inlets.

The private Python imports extend boundary admission and connect resident
operators. Numerical CUDA sources are archived without modification.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from . import live_inlet_modes as scope

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / 'validation/gpu_fwave'
PRIVATE = {name: '_g1_' + name[:-3] for name in (
    'boundary_schedule.py', 'boundaries.py', 'sediment_transport.py',
    'coupled_solver.py', 'outlet_boundary.py')}
SHA = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(data):
    return scope.validate_scope(data, start=float(data.get('time', 0.)),
        end=float(data['endtime']), interval=float(data['output_dt']),
        dtmax=float(data['dtmax']), launch_mode='plain', integrator='ssp33')


def transform(name, source):
    """Replace constructor/admission anchors without loading native GIS tools."""
    def replace(before, after):
        nonlocal source
        scope.require(source.count(before) == 1,
                      'G1 source anchor absent or ambiguous: ' + name + ': ' + before[:80])
        source = source.replace(before, after, 1)

    if name == 'outlet_boundary.py':
        from validation.gpu_fwave.normal_outlet_boundary import transform_bridge
        source = transform_bridge(name, source)
        replace('from .normal_outlet_boundary import validate_outlets as validate_actual',
                'from braids_gpu.live_inlet_modes import validate_outlets as validate_actual')
    elif name == 'boundary_schedule.py':
        replace("    for key in ('section_control', 'sampled'):\n        require(not np.any(data.get(key, 0)), 'Live sections are outside schedule-only scope: '+key)",
                '    validate_boundaries(data)')
        replace('import numpy as np',
                'import numpy as np\nfrom braids_gpu.live_inlet_modes import validate_boundaries')
    elif name == 'boundaries.py':
        replace('supported=[2,3,6,8]+([1] if allow_bedform_mode1 else [])',
                'supported=[2,3,4,6,8,9]+([1] if allow_bedform_mode1 else [])')
        replace("    for key in ('nhydro_series','section_control','sampled'):",
                "    validate_boundaries(data)\n    for key in ('nhydro_series',):")
        replace('def __init__(self,data,inputs,source=None,*,allow_scheduled=False,allow_bedform_mode1=False):',
                'def __init__(self,data,inputs,source=None,*,allow_scheduled=True,allow_bedform_mode1=False):')
        replace('import numpy as np',
                'import numpy as np\nfrom braids_gpu.live_inlet_modes import validate_boundaries')
    elif name == 'sediment_transport.py':
        replace('allowed_modes=[2,3,6,8]+([1] if allow_bedform else [])',
                'allowed_modes=[2,3,4,6,8,9]+([1] if allow_bedform else [])')
        replace('    source=dict(data);source.update(config or {})',
                '    source=dict(data);source.update(config or {})\n    validate_boundaries(source)')
        replace('import numpy as np',
                'import numpy as np\nfrom braids_gpu.live_inlet_modes import validate_boundaries')
    elif name == 'coupled_solver.py':
        replace('from .boundaries import BoundaryControls',
                'from ._g1_boundaries import BoundaryControls')
        replace('from .sediment_transport import ResidentSedimentTransport',
                'from ._g1_sediment_transport import ResidentSedimentTransport')
        replace("        if scheduled and key in ('nhydro_series','nfraction_series'):continue",
                "        if key in ('nhydro_series','nfraction_series'):continue")
        replace("self.boundary=BoundaryControls(bcdata,self.flux.inputs,source=self.folder/'boundaries.cu',\n                                       allow_scheduled=scheduled,\n                                       allow_bedform_mode1=(self.bedform_variant and\n                                           bool(np.any(np.asarray(data['mode']) == 1))))",
                 "self.boundary=BoundaryControls(bcdata,self.flux.inputs,source=self.folder/'boundaries.cu')")
        replace("        if scheduled:sediment_kwargs['allow_scheduled']=True",
                "        sediment_kwargs['allow_scheduled']=True")
    else:
        raise ValueError('Unknown G1 source: ' + name)
    return source


def prepare_sources(folder, data):
    """Freeze the executed spatial operators and bridge before a GPU trial."""
    validate(data)
    folder = Path(folder).resolve()
    scope.require(not folder.exists(), 'Preserve existing G1 source archive')
    folder.mkdir(parents=True)
    source_map = {}
    for source in sorted(SPATIAL.glob('*.cu')):
        shutil.copy2(source, folder / source.name)
        source_map['validation/gpu_fwave/' + source.name] = SHA(source)
    python_files = list(sorted(SPATIAL.glob('*.py'))) + [
        ROOT / 'braids_gpu' / name for name in
        ('live_inlet_modes.py', 'g1_live_inlet_boundary.py', 'g1_inlet_solver.py',
         'a1_full_array_transaction.py')]
    for source in python_files:
        relative = source.relative_to(ROOT).as_posix()
        target = folder / 'python_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_map[relative] = SHA(source)
    generated = {}
    for name, private in PRIVATE.items():
        target = folder / (private + '.py')
        target.write_text(transform(name, (SPATIAL / name).read_text(encoding='utf-8')),
                          encoding='utf-8')
        generated[name] = dict(name=target.name, sha256=SHA(target))
    gates = {}
    for name in ('GPU_PRE_RUN_GATE_V3.json', 'GPU_CLOCK_AMENDMENT_V4.json'):
        source = ROOT / 'validation/g1_live_inlet_modes' / name
        shutil.copy2(source, folder / name)
        gates[name] = SHA(source)
    receipt = dict(schema=1, variant=scope.VARIANT, sources=source_map,
        generated=generated, gates=gates, numerical_CUDA_unchanged=True,
        integrator='SSPRK33', stage_fractions=[0., 1., .5],
        physical_validation=False, Delft3D_source_modified=False)
    (folder / 'g1_sources.json').write_text(json.dumps(receipt, indent=2) + '\n',
                                          encoding='utf-8')
    verify_sources(folder)
    return receipt


def verify_sources(folder):
    folder = Path(folder).resolve()
    receipt = json.loads((folder / 'g1_sources.json').read_text(encoding='utf-8'))
    scope.require(receipt['schema'] == 1 and receipt['variant'] == scope.VARIANT,
                  'G1 source contract differs')
    for relative, digest in receipt['sources'].items():
        path = ROOT / relative
        saved = (folder / path.name if path.suffix == '.cu' else
                 folder / 'python_snapshot' / relative)
        scope.require(SHA(path) == digest and SHA(saved) == digest,
                      'G1 source drift: ' + relative)
    for name, row in receipt['generated'].items():
        path = folder / row['name']
        scope.require(SHA(path) == row['sha256'] and
                      path.read_text(encoding='utf-8') ==
                      transform(name, (SPATIAL / name).read_text(encoding='utf-8')),
                      'G1 generated bridge drift: ' + name)
    for name, digest in receipt['gates'].items():
        scope.require(SHA(folder / name) == digest and
                      SHA(ROOT / 'validation/g1_live_inlet_modes' / name) == digest,
                      'G1 predeclared gate differs: ' + name)
    return receipt


@contextmanager
def constructors(folder, data):
    """Yield the same coupled f-wave/MPM solver with three live inlet stages."""
    from .g1_live_inlet_boundary import make_boundary_controls
    from .a1_full_array_transaction import wrap_solver
    from validation.gpu_fwave.boundary_events import tolerance

    validate(data)
    folder = Path(folder).resolve()
    verify_sources(folder)
    prefix = 'validation.gpu_fwave.'
    scope.require(not any(prefix + key in sys.modules for key in PRIVATE.values()),
                  'G1 private modules already loaded')
    modules = {}
    loaded = []
    try:
        for name, private in PRIVATE.items():
            key = prefix + private
            spec = importlib.util.spec_from_file_location(key, folder / (private + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            loaded.append(key)
            spec.loader.exec_module(module)
            modules[name] = module
        core = modules['coupled_solver.py']
        core.BoundaryControls = make_boundary_controls(
            modules['boundaries.py'].BoundaryControls, modules['outlet_boundary.py'])
        schedule = modules['boundary_schedule.py']

        class G1InletSolver(core.CoupledSolver):
            def __init__(self, raw, *, source_folder=folder):
                validate(raw)
                scope.require(Path(source_folder).resolve() == folder,
                              'G1 source folder differs')
                super().__init__(raw, source_folder=folder, integrator='ssp33')
                # SSPRK33 evaluates every Euler update in this one stage
                # buffer. Bind it before inventories/checkpoints are created;
                # otherwise the first Euler call changes an array alias.
                self.flux.inputs['q'] = self.stepper.stage['q']
                self.boundary.bind_sediment_hydraulic(
                    self.sediment.geometry['bc_hydraulic'])
                self.boundary.bind_stage_state(self.stepper.stage['q'])
                self.schedule = schedule.ResidentBoundarySchedule(
                    raw, source_path=folder / 'boundary_schedule.cu')
                for value in self.schedule.outputs.values():
                    value.fill(0)
                self._g1_stage = None
                self._g1_times = None

            def euler(self, state, dt, current):
                index = self._g1_stage
                scope.require(type(index) is int and 0 <= index < 3 and
                    abs(float(current) - self._g1_times[index]) <= tolerance(current),
                    'G1 SSPRK33 Euler clock/order differs')
                slot = scope.STAGE_SLOTS[index]
                values = self.schedule.outputs
                self.cp.copyto(self.boundary.arrays['forcing'], values['hydraulic'][slot])
                self.cp.copyto(self.sediment.geometry['bc_hydraulic'], values['hydraulic'][slot])
                self.cp.copyto(self.sediment.geometry['feed'], values['feed'][slot])
                result = super().euler(state, dt, current)
                self.boundary.record_stage(index)
                self._merge(values['status'][slot])
                self._g1_stage += 1
                return result

            # The complete-array wrapper snapshots/restores every reachable
            # device buffer, including schedule, root workspace and diagnostics.
            def _a1_trial_diagnostics(self):
                return {}

            def _a1_restore_diagnostics(self, saved, *, where=True):
                pass

            def step(self, dt, current, *, commit=True):
                scope.require(self._g1_stage is None, 'Concurrent G1 trials are unsupported')
                self.boundary.check_storage()
                saved = self._a1_trial_diagnostics()
                self._g1_stage = 0
                self._g1_times = tuple(float(current + f * dt) for f in (0., 1., .5))
                try:
                    self.boundary.begin_trial()
                    self.schedule.evaluate(current, dt)
                    result = super().step(dt, current, commit=commit)
                    scope.require(self._g1_stage == 3, 'G1 SSPRK33 omitted an Euler stage')
                    self._a1_restore_diagnostics(saved,
                        where=(result['status'] != 0) if commit else True)
                    return result
                except Exception:
                    self._a1_restore_diagnostics(saved)
                    raise
                finally:
                    self._g1_stage = None
                    self._g1_times = None

            def candidate_receipt(self):
                return dict(variant=scope.VARIANT, integrator='SSPRK33',
                    stage_slots=list(scope.STAGE_SLOTS),
                    source_identity=SHA(folder / 'g1_sources.json'),
                    boundary=self.boundary.receipt(), schedule=self.schedule.receipt(),
                    backend='gpu', automatic_CPU_fallback=False,
                    accepted_transaction='segmented_all_reachable_v5',
                    physical_validation=False)

        yield wrap_solver(G1InletSolver)
    finally:
        for key in reversed(loaded):
            sys.modules.pop(key, None)
        verify_sources(folder)
