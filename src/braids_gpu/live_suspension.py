"""Guarded live normal/critical, scheduled suspension SSPRK33 GPU composition.

The spatial CUDA kernels are copied from the current qualified source archive.
Only host admission, three-stage forcing selection and transaction bookkeeping
are new here.  This is an engineering selector, not physical braids_support.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


VARIANT = 'live_normal_critical_suspension_mpm_ssp33'
VR_VARIANT = 'live_normal_critical_van_rijn_mpm_ssp33'
GRADED_VARIANT = 'live_normal_critical_van_rijn_graded_ssp33'
PRIVATE_KEY = 'braids_support.gpu_fwave._g2_live_suspension_adapter'
STAGE_SLOTS = (0, 2, 1)
STAGE_FRACTIONS = (0., 1., .5)
GATE = (Path(__file__).resolve().parents[1] / 'validation' / 'gpu_fwave' /
        'reports' / 'g2_live_suspension_native_20261004_v1' / 'GATE.json')
GATE_SHA = '0fa58245fd53ce8530c626599f04ec9d4d4f6d3d2015d64f8d0450c5fc7a07a2'
VR_GATE = (Path(__file__).resolve().parents[1] / 'validation' / 'gpu_fwave' /
           'reports' / 'g2_live_vr_gpu_20261004_v1' / 'PRE_RUN_GATE.json')
VR_GATE_SHA = 'bfca7e188cb3a0346af783b5fb0e4259ab42c164f97a07f7be9feeb555509f03'
GRADED_GATE = (Path(__file__).resolve().parents[1] / 'validation' / 'gpu_fwave' /
               'reports' / 'g2_live_vr_graded_gpu_20261004_v2' / 'PRE_RUN_GATE.json')
GRADED_GATE_SHA = '278e46488f3a4b33a2b887ac07b581c060d29b81be7c568877b352b01158fa67'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_case(folder):
    from .live_secondary import read_case as read_live
    return read_live(folder)


def gate_for_exchange(exchange_mode, *, graded=False):
    require(exchange_mode in (1, 2), 'Live suspension selector requires exchange 1 or 2')
    if graded:
        require(exchange_mode == 2, 'Graded live suspension requires Van Rijn exchange 2')
        return GRADED_GATE, GRADED_GATE_SHA
    return (GATE, GATE_SHA) if exchange_mode == 1 else (VR_GATE, VR_GATE_SHA)


def verify_vr_reference(input_path):
    """Bind the research selector to one authenticated unchanged-native case."""
    from .runner import sha

    require(VR_GATE.is_file() and sha(VR_GATE) == VR_GATE_SHA,
            'Frozen live Van Rijn GPU gate identity differs')
    gate = json.loads(VR_GATE.read_text(encoding='utf-8'))
    require(gate['status'].startswith('frozen_after_unchanged_native_reference') and
            int(gate['controls']['exchange_mode']) == 2,
            'Live Van Rijn gate controls differ')
    reference = gate['reference']
    root = VR_GATE.parents[2] / 'runs' / 'g2_live_vr_native_20261004_v1'
    native_gate = VR_GATE.parents[2] / 'reports' / 'g2_live_vr_native_20261004_v1' / 'GATE.json'
    source = VR_GATE.parents[2] / 'g2_live_vr_native.py'
    helper = VR_GATE.parents[2] / 'g2_live_vr_preparation.py'
    artifact_paths = dict(native_gate_sha256=native_gate,
        native_plan_sha256=root / 'plan.json',
        native_audit_sha256=root / 'native_audit.json',
        native_driver_sha256=source,
        preparation_helper_sha256=helper,
        native_executable_sha256=None,
        zero_step_exporter_sha256=VR_GATE.parents[2] / 'runs' /
            'g2_live_suspension_export_20261004_v1' / 'runtime' / 'xbraids_normal_export.exe',
        prognostic_secondary_flow_sha256=root / 'prognostic' / 'cpu' / 'secondary_flow.txt',
        prognostic_secondary_budget_sha256=root / 'prognostic' / 'cpu' /
            'secondary_flow_budget.txt')
    plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    artifact_paths['native_executable_sha256'] = Path(plan['native_executable'])
    for key, path in artifact_paths.items():
        require(path.is_file() and sha(path) == reference[key],
                'Live Van Rijn native reference differs: ' + key)
    require(plan['gate_sha256'] == reference['native_gate_sha256'] and
            plan['native_executable_sha256'] == reference['native_executable_sha256'] and
            {row['variant'] for row in plan['cases']} == {'off', 'prognostic'},
            'Live Van Rijn native plan controls differ')
    for name, digest in plan['sources'].items():
        require(Path(name).is_file() and sha(name) == digest,
                'Live Van Rijn frozen native source differs: ' + name)
    for row in plan['cases']:
        for name, digest in row['files'].items():
            require(Path(name).is_file() and sha(name) == digest,
                    'Live Van Rijn native deck differs: ' + name)
    audit = json.loads((root / 'native_audit.json').read_text(encoding='utf-8'))
    require(audit['passed'] is True and audit['plan_sha256'] ==
            reference['native_plan_sha256'] and
            all(audit['variants'][variant]['passed'] for variant in ('off', 'prognostic')),
            'Live Van Rijn native audit did not pass')
    folder = Path(input_path).resolve()
    input_file = folder / 'coupled_input.npz'
    require(input_file.is_file(), 'Live Van Rijn prepared input is missing')
    input_sha = sha(input_file)
    variants = [variant for variant, digest in reference['prepared_input_sha256'].items()
                if input_sha == digest]
    require(len(variants) == 1, 'Live Van Rijn selector requires one frozen prepared input')
    variant = variants[0]
    receipt_path = folder / 'preparation_receipt.json'
    require(receipt_path.is_file() and sha(receipt_path) ==
            reference['preparation_receipt_sha256'][variant],
            'Live Van Rijn zero-step receipt differs')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt['status'] == 'completed' and receipt['variant'] == variant and
            receipt['plan_sha256'] == reference['native_plan_sha256'] and
            receipt['native_audit_sha256'] == reference['native_audit_sha256'] and
            receipt['export_executable_sha256'] == reference['zero_step_exporter_sha256'] and
            receipt['parent_executable_sha256'] == reference['native_executable_sha256'] and
            receipt['solver_steps_performed'] == 0 and
            receipt['GPU_solver_executed'] is False and
            receipt['cpu_time_integrator'] == 33 and
            receipt['exchange_mode'] == 2 and all(receipt['checks'].values()),
            'Live Van Rijn zero-step preparation controls differ')
    for name, digest in receipt['artifacts'].items():
        path = (folder / name).resolve()
        require(path.parent == folder and path.is_file() and sha(path) == digest,
                'Live Van Rijn prepared artifact differs: ' + name)
    for current in ('off', 'prognostic'):
        cpu = root / current / 'cpu'
        require(sha(cpu / 'frames.txt') == reference['native_frames_sha256'][current],
                'Live Van Rijn native frames differ: ' + current)
        execution = json.loads((cpu / 'execution.json').read_text(encoding='utf-8'))
        require(execution['status'] == 'completed' and execution['return_code'] == 0 and
                execution['plan_sha256'] == reference['native_plan_sha256'] and
                execution['native_executable_sha256'] == reference['native_executable_sha256'],
                'Live Van Rijn native execution receipt differs: ' + current)
        for name, digest in execution['files'].items():
            require(Path(name).is_file() and sha(name) == digest,
                    'Live Van Rijn native evolution output differs: ' + name)
    return dict(variant=variant, gate_sha256=VR_GATE_SHA,
        native_plan_sha256=reference['native_plan_sha256'],
        native_audit_sha256=reference['native_audit_sha256'],
        native_executable_sha256=reference['native_executable_sha256'],
        prepared_input_sha256=input_sha,
        preparation_receipt_sha256=reference['preparation_receipt_sha256'][variant],
        archive_bound=True)


def verify_graded_reference(input_path):
    """Pin the seven-class research route to unchanged native and zero-step I/O."""
    from .runner import sha
    from braids_support.gpu_fwave import g2_live_vr_graded_native as native
    from braids_support.gpu_fwave import g2_live_vr_graded_process as audit_helper
    from braids_support.gpu_fwave import g2_live_vr_graded_export_v3 as export_helper

    require(GRADED_GATE.is_file() and sha(GRADED_GATE) == GRADED_GATE_SHA,
            'Frozen graded live Van Rijn gate differs')
    frozen = json.loads(GRADED_GATE.read_text(encoding='utf-8'))
    require(frozen['status'] ==
            'frozen_after_unchanged_native_reference_before_GPU_evolution' and
            frozen['selector'] == GRADED_VARIANT and
            frozen['inherited_numerical_gate_sha256'] ==
                sha(GRADED_GATE.parent.parent / 'g2_live_vr_graded_gpu_20261004_v1' /
                    'PRE_RUN_GATE.json') and
            frozen['controls']['grain_classes'] == 7 and
            frozen['controls']['bed_layers'] == 4,
            'Graded live Van Rijn gate controls differ')
    reference = frozen['reference']
    native_paths = dict(native_gate_sha256=native.GATE,
        native_driver_sha256=native.__file__,
        native_audit_helper_sha256=audit_helper.__file__,
        zero_step_helper_sha256=export_helper.__file__,
        native_plan_sha256=native.ROOT / 'plan.json',
        native_audit_sha256=native.ROOT / 'native_audit.json',
        native_executable_sha256=native.frozen_cpu.TIGHT_EXE)
    for key, path in native_paths.items():
        expected = frozen['native_gate_sha256'] if key == 'native_gate_sha256' else reference[key]
        require(Path(path).is_file() and sha(path) == expected,
                'Graded native reference differs: ' + key)
    plan, plan_sha = native.load_plan()
    audit = json.loads((native.ROOT / 'native_audit.json').read_text(encoding='utf-8'))
    require(plan_sha == reference['native_plan_sha256'] and
            audit['passed'] is True and audit['plan_sha256'] == plan_sha and
            audit['audit_helper_sha256'] == reference['native_audit_helper_sha256'] and
            all(audit['variants'][name]['passed'] for name in ('off', 'prognostic')),
            'Graded native plan/audit identity or acceptance differs')
    for name in ('off', 'prognostic'):
        cpu = native.ROOT / name / 'cpu'
        require(sha(cpu / 'frames.txt') == reference['native_frames_sha256'][name],
                'Graded native frame identity differs: ' + name)
        receipt = json.loads((cpu / 'execution.json').read_text(encoding='utf-8'))
        require(receipt['status'] == 'completed' and receipt['return_code'] == 0 and
                receipt['plan_sha256'] == plan_sha and
                receipt['native_executable_sha256'] ==
                    reference['native_executable_sha256'],
                'Graded native execution receipt differs: ' + name)
        for path, digest in receipt['files'].items():
            require(Path(path).is_file() and sha(path) == digest,
                    'Graded native evolution output differs: ' + path)
    folder = Path(input_path).resolve()
    choices = {name: (native.ROOT / name / 'prepared_gpu_ssp33_v3').resolve()
               for name in ('off', 'prognostic')}
    matches = [name for name, expected in choices.items() if folder == expected]
    require(len(matches) == 1, 'Graded live Van Rijn requires its frozen prepared input')
    variant = matches[0]
    input_file = folder / 'coupled_input.npz'
    receipt_path = folder / 'preparation_receipt.json'
    require(input_file.is_file() and
            sha(input_file) == reference['prepared_input_sha256'][variant] and
            receipt_path.is_file() and
            sha(receipt_path) == reference['preparation_receipt_sha256'][variant],
            'Graded prepared input/receipt differs')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt['schema'] == 1 and receipt['status'] == 'completed' and
            receipt['variant'] == variant and receipt['plan_sha256'] == plan_sha and
            receipt['native_audit_sha256'] == reference['native_audit_sha256'] and
            receipt['gate_sha256'] == frozen['native_gate_sha256'] and
            receipt['parent_executable_sha256'] == reference['native_executable_sha256'] and
            receipt['helper_sha256'] == reference['zero_step_helper_sha256'] and
            receipt['solver_steps_performed'] == 0 and
            receipt['GPU_solver_executed'] is False and
            all(receipt['checks'].values()),
            'Graded zero-step preparation controls differ')
    for name, digest in receipt['artifacts'].items():
        path = (folder / name).resolve()
        require(path.parent == folder and path.is_file() and sha(path) == digest,
                'Graded prepared artifact differs: ' + name)
    return dict(variant=variant, gate_sha256=GRADED_GATE_SHA,
        native_plan_sha256=reference['native_plan_sha256'],
        native_audit_sha256=reference['native_audit_sha256'],
        native_executable_sha256=reference['native_executable_sha256'],
        prepared_input_sha256=reference['prepared_input_sha256'][variant],
        preparation_receipt_sha256=reference['preparation_receipt_sha256'][variant],
        archive_bound=True)


def validate_scope(data, *, start, end, interval, dtmax, launch_mode, integrator,
                   expected_exchange=1, graded=False):
    """Reject unported switches before output creation or CUDA initialization."""
    from braids_support.gpu_fwave.coupled_solver import normalized
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables
    from braids_support.gpu_fwave.normal_outlet_boundary import validate_boundaries
    from braids_support.gpu_fwave.secondary_transport import validate_scope as secondary_scope
    from .runner import sha

    data = normalized(data)
    gate, gate_sha = gate_for_exchange(expected_exchange, graded=graded)
    ng = 7 if graded else 2
    nl = 4 if graded else 2
    require(gate.is_file() and sha(gate) == gate_sha,
            'Frozen G2 pre-GPU gate identity differs')
    require(launch_mode == 'plain' and integrator == 'ssp33',
            'Live suspension requires explicit plain SSPRK33')
    require(int(data['scheme']) == 6 and int(data['cpu_time_integrator']) == 33 and
            int(data['cpu_outlet_extension']) == 0,
            'Live suspension requires native scheme 6 legacy SSPRK33 and outlet extension 0')
    require(int(data['nb']) == 2 and np.array_equal(data['mode'], [9, 4]) and
            np.array_equal(data['section_control'], [1, 0]),
            'Live suspension requires one mode9/control1 inlet and one mode4 outlet')
    offsets = np.asarray(data['boundary_offset'])
    require(offsets.shape == (3,) and offsets[1] - offsets[0] >= 4 and
            offsets[2] - offsets[1] >= 1,
            'Live suspension requires nonempty surveyed sections')
    require(int(data['nfrac']) == ng and int(data['nbed_layers']) == nl and
            int(data['bedlaw']) == 2 and int(data['mpm_friction_mode']) == 1 and
            int(data['exchange_mode']) == expected_exchange and int(data['settling_mode']) == 0,
            f'Live suspension requires {ng}-class/{nl}-layer MPM and noncohesive exchange {expected_exchange}')
    require(np.array_equal(data['sediment_type_fraction'], np.zeros(ng, np.int32)),
            'Live suspension requires noncohesive sand classes')
    require(int(data['nsubstrate_layers']) == nl - 1 and
            int(data['sediment_roughness_mode']) == 0 and
            np.array_equal(data['sediment_roughness_fraction'], np.zeros(ng, np.int32)) and
            int(data['bed_slope']) == (2 if graded else 0) and
            int(data['hiding_exposure']) == (4 if graded else 0),
            'Live suspension requires the frozen MPM stratigraphy, hiding, slope and roughness')
    grains = np.asarray(data['d50_fraction'])
    settling = np.asarray(data['ws_fraction'])
    require(grains.shape == settling.shape == (ng,) and np.isfinite(grains).all() and
            np.isfinite(settling).all() and np.all(grains > 0) and
            np.all(settling > 0) and np.all(np.diff(grains) > 0) and
            float(data['hindered_settling_reference_concentration']) > 0,
            'Live suspension requires ordered distinct positive sand sizes and settling controls')
    model = int(data['secondary_flow_model'])
    require(model in (0, 2), 'Live suspension supports secondary OFF or prognostic model 2')
    base_state = 3 + ng + ng * nl
    require(int(data['secondary_index']) == (base_state if model == 2 else -1) and
            int(data['nstate']) == base_state + int(model == 2) and
            int(data['nledger']) == 1 + 2 * ng + int(model == 2) and
            np.array_equal(data['suspended_indices'], np.arange(3, 3 + ng)),
            'Live suspension native state/ledger layout differs')
    if model == 2:
        require(bool(data['secondary_flow']) and
                bool(data['secondary_flow_transport_direction']) and
                float(data['secondary_flow_momentum_beta']) == 1. and
                float(data['secondary_flow_espir']) == 1.,
                'Live suspension requires prognostic direction, Beta 1 and Espir 1')
    secondary_scope(data)
    if expected_exchange == 2:
        require(int(data['nc']) == 128 and int(data['nf']) >= 16 and
                np.array_equal(np.diff(offsets), [8, 8]),
                'Frozen live Van Rijn gate requires 128 cells and eight faces per boundary')
    state = np.asarray(data['q']); suspended = np.asarray(data['suspended_mass'])
    dry = float(data['dry_tolerance'])
    require(state.shape == (3, int(data['nc'])) and
            suspended.shape == (ng, int(data['nc'])) and
            np.isfinite(state).all() and np.isfinite(suspended).all() and
            np.all(state[0] >= 0) and np.all(suspended >= 0) and np.any(state[0] > dry),
            'Live suspension requires finite nonnegative hydraulic/suspended state')
    initially_dry = state[0] <= dry
    require(not np.any(state[1:, initially_dry]) and
            not np.any(suspended[:, initially_dry]),
            'Live suspension requires zero momentum and suspended mass in dry cells')
    require(float(data['morphological_factor']) == 1. and
            float(data['horizontal_viscosity']) == 0. and
            not np.any(data['nseries']),
            'Live suspension excludes MorFac, viscosity and legacy scalar series')
    for key in ('amr_enabled', 'bedload_adaptation', 'bedform_roughness',
                'bank_failure', 'bank_avalanche', 'general_avalanche',
                'bank_front_tracking', 'bank_erosion_fraction', 'nmorfac',
                'viscosity_equilibrated_inlet'):
        require(not np.any(data.get(key, 0)), 'Live suspension excludes ' + key)
    validate_boundaries(data)
    tables = validate_tables(dict(data, section_control=np.zeros(2, np.int32),
                                  sampled=np.zeros(int(data['nf']), np.int32)))
    counts = np.asarray(data['nfraction_series'])
    count = int(data['nhydro_series'][0])
    require(count >= 2 and np.array_equal(counts[:, 0], np.full(ng, count)) and
            not np.any(counts[:, 1]) and not int(data['nhydro_series'][1]),
            'Live suspension requires common inlet Q/class schedules')
    times = np.asarray(data['hydro_series_time'])[:count, 0]
    require(np.array_equal(data['fraction_series_time'][:count, :, 0],
                           np.repeat(times[:, None], ng, axis=1)) and
            times[0] <= start and times[-1] >= end,
            'Live suspension forcing knots must be shared and cover the run')
    concentration = tables['fraction_series_concentration'][:count, :, 0]
    feed = tables['fraction_series_bed_input'][:count, :, 0]
    require(np.ptp(tables['hydro_series_q'][:count, 0]) > 0 and
            np.all(np.any(concentration > 0, axis=0)) and
            np.all(np.any(feed > 0, axis=0)) and
            np.unique(concentration, axis=1).shape[1] == ng and
            np.unique(feed, axis=1).shape[1] == ng,
            'Live suspension requires varying Q, nonzero distinct class concentrations and feeds')
    events = BoundaryEvents.from_tables(data)
    require(np.array_equal(events.knots, times),
            'Live suspension requires one common active forcing clock')
    require(np.isfinite([start, end, interval, dtmax]).all() and
            start >= 0 and end > start and interval > 0 and dtmax > 0,
            'Live suspension requires finite positive run controls')
    return events


@contextmanager
def isolated_adapter(*, expected_exchange=1, graded=False):
    """Generate the proven live spatial bridge, with G2 admission isolated."""
    from braids_support.gpu_fwave import a1_combined_current_adapter as ancestor
    require(PRIVATE_KEY not in sys.modules, 'G2 live adapter already loaded')
    spec = importlib.util.spec_from_file_location(PRIVATE_KEY, Path(ancestor.__file__))
    require(spec is not None and spec.loader is not None, 'Live spatial bridge source absent')
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_KEY] = module
    try:
        spec.loader.exec_module(module)
        module.validate_input = lambda data: validate_scope(
            data, start=float(data.get('time', 0.)), end=float(data['endtime']),
            interval=float(data['output_dt']), dtmax=float(data['dtmax']),
            launch_mode='plain', integrator='ssp33',
            expected_exchange=expected_exchange, graded=graded) is not None
        yield module
    finally:
        sys.modules.pop(PRIVATE_KEY, None)


def _spatially_validated_scope(actual, stock_scope, *, expected_exchange=1,
                               graded=False, **kwargs):
    """Reuse stock material checks on a boundary-neutral view of real data."""
    validate_scope(actual, start=float(actual.get('time', 0.)),
                   end=float(actual['endtime']), interval=float(actual['output_dt']),
                   dtmax=float(actual['dtmax']), launch_mode='plain', integrator='ssp33',
                   expected_exchange=expected_exchange, graded=graded)
    check = dict(actual, mode=np.array([6, 2], np.int32),
                 section_control=np.zeros(2, np.int32),
                 sampled=np.zeros(int(actual['nf']), np.int32),
                 nhydro_series=np.zeros(2, np.int32),
                 nfraction_series=np.zeros((int(actual['nfrac']), 2), np.int32))
    return stock_scope(check, **kwargs)


@contextmanager
def constructors(folder, data, *, expected_exchange=1, graded=False):
    """Yield a private three-stage solver; restore module admission on exit."""
    from braids_support.gpu_fwave.boundary_events import tolerance
    folder = Path(folder).resolve()
    with isolated_adapter(expected_exchange=expected_exchange, graded=graded) as candidate:
        with candidate.constructors(folder, data) as historical:
            core = historical.__mro__[2]
            require(core.__name__ == 'CoupledSolver', 'Live spatial base class differs')
            globals_ = core.__init__.__globals__
            stock_scope = globals_['validate_scope']
            schedule_module = sys.modules['braids_support.gpu_fwave._a1_boundary_schedule']
            globals_['validate_scope'] = lambda actual, **kwargs: _spatially_validated_scope(
                actual, stock_scope, expected_exchange=expected_exchange,
                graded=graded, **kwargs)
            try:
                class LiveSuspensionSolver(core):
                    def __init__(self, raw, *, source_folder=folder):
                        require(Path(source_folder).resolve() == folder,
                                'G2 private source folder differs')
                        super().__init__(raw, source_folder=folder, integrator='ssp33')
                        self.boundary.bind_sediment_hydraulic(self.sediment.geometry['bc_hydraulic'])
                        self.boundary.bind_stage_state(self.stepper.stage['q'])
                        self.schedule = schedule_module.ResidentBoundarySchedule(
                            raw, source_path=folder / 'boundary_schedule.cu')
                        for records in (self.boundary.normal_records,
                                        self.boundary.outlet_records,self.schedule.outputs):
                            for value in records.values():
                                value.fill(0)
                        for value in self.boundary.normal_root.cache.values():
                            value.fill(0)
                        self._g2_stage = None
                        self._g2_times = None

                    def euler(self, state, dt, current):
                        stage = self._g2_stage
                        require(type(stage) is int and 0 <= stage < 3 and
                                abs(float(current) - self._g2_times[stage]) <=
                                tolerance(current),
                                'Live SSPRK33 Euler stage time/order changed')
                        slot = STAGE_SLOTS[stage]
                        values = self.schedule.outputs
                        cp = self.cp
                        cp.copyto(self.boundary.arrays['forcing'], values['hydraulic'][slot])
                        cp.copyto(self.sediment.geometry['bc_hydraulic'], values['hydraulic'][slot])
                        cp.copyto(self.sediment.geometry['feed'], values['feed'][slot])
                        cp.copyto(self.bc_concentration, values['concentration'][slot])
                        result = super().euler(state, dt, current)
                        self.boundary.record_stage(stage)
                        self._merge(values['status'][slot])
                        self._g2_stage += 1
                        return result

                    def _a1_trial_diagnostics(self):
                        return dict(clock=self.schedule.clock.copy(),
                            schedule={key:value.copy() for key,value in self.schedule.outputs.items()},
                            normal={key:value.copy() for key,value in self.boundary.normal_records.items()},
                            outlet={key:value.copy() for key,value in self.boundary.outlet_records.items()},
                            normal_latch=self.boundary._normal_latch.copy(),
                            outlet_latch=self.boundary._outlet_latch.copy())

                    def _a1_restore_diagnostics(self, saved, *, where=True):
                        cp = self.cp
                        cp.copyto(self.schedule.clock, saved['clock'], where=where)
                        for key,value in saved['schedule'].items():
                            cp.copyto(self.schedule.outputs[key], value, where=where)
                        for key,value in saved['normal'].items():
                            cp.copyto(self.boundary.normal_records[key], value, where=where)
                        for key,value in saved['outlet'].items():
                            cp.copyto(self.boundary.outlet_records[key], value, where=where)
                        cp.copyto(self.boundary._normal_latch, saved['normal_latch'], where=where)
                        cp.copyto(self.boundary._outlet_latch, saved['outlet_latch'], where=where)

                    def step(self, dt, current, *, commit=True):
                        require(self._g2_stage is None, 'Concurrent live suspension trials are unsupported')
                        self.boundary.check_storage()
                        saved = self._a1_trial_diagnostics()
                        self._g2_stage = 0
                        self._g2_times = tuple(float(current + fraction*dt)
                                               for fraction in STAGE_FRACTIONS)
                        try:
                            self.boundary.begin_trial()
                            self.schedule.evaluate(current, dt)
                            result = super().step(dt, current, commit=commit)
                            require(self._g2_stage == 3, 'Live SSPRK33 omitted an Euler record')
                            self._a1_restore_diagnostics(saved,
                                where=(result['status'] != 0) if commit else True)
                            return result
                        except Exception:
                            self._a1_restore_diagnostics(saved)
                            raise
                        finally:
                            self._g2_stage = None
                            self._g2_times = None

                    def candidate_receipt(self):
                        self.boundary.check_storage()
                        return dict(boundary=self.boundary.receipt(),
                            schedule=self.schedule.receipt(), integrator='SSPRK33',
                            euler_slots=list(STAGE_SLOTS),
                            accepted_transaction='segmented_all_reachable_v5',
                            selected_exchange_mode=expected_exchange,
                            selected_gate_sha256=gate_for_exchange(
                                expected_exchange, graded=graded)[1],
                            graded_seven_class_four_layer=graded,
                            native_controls_retained=True, backend='gpu',
                            automatic_CPU_fallback=False, physical_validation=False)

                yield LiveSuspensionSolver
            finally:
                globals_['validate_scope'] = stock_scope
