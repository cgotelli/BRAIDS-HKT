"""Guarded fixed-grid live-boundary/secondary/bank GPU composition.

The only admitted inputs are the four authenticated G3 live-bank zero-step
exports. CUDA numerical kernels are archived unchanged; this adapter joins
the already-ported live boundary and bank/avalanche transactions explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / 'braids_support/g3_live_bank/runs/live_combined_export_20261004_v2'
GATE = ROOT / 'braids_support/g3_live_bank/reports/GPU_ACCEPTANCE_GATE.md'
MANIFEST_SHA = 'db80e7987f98c6ccd9e487124feed5c3ce11d117b39771e6b12a4512bf893291'
GATE_SHA = '7f75985d87116f8c4b059b17d470bb180fed5f4549f09173bc8ebed4a0ecaa82'
PRIVATE_KEY = 'braids_support.gpu_fwave._g3_live_bank_adapter'
VARIANTS = {
    'g3_live_bank_mode2_ssp33': 2,
    'g3_live_bank_mode3_ssp33': 3,
}
STAGE_SLOTS = (0, 2, 1)
STAGE_FRACTIONS = (0., 1., .5)
HISTORY = {
    'undercut': 'bank_undercut',
    'reference_height': 'bank_reference_height',
    'toe_height': 'bank_toe_height',
    'reference_length': 'bank_reference_length',
    'toe_cumulative': 'bank_toe_cumulative',
    'collapse_cumulative': 'bank_collapse_cumulative',
    'collapse_count': 'bank_collapse_count',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    from .runner import sha as digest
    return digest(path)


def _case(folder, variant):
    folder = Path(folder).resolve()
    require(variant in VARIANTS, 'Unknown explicit G3 live-bank GPU variant')
    require(folder.parent == PREPARED.resolve() and
            folder.name in ('mode2_on', 'mode2_off', 'mode3_on', 'mode3_off') and
            folder.name.startswith(f'mode{VARIANTS[variant]}_'),
            'G3 live-bank selector requires its matching frozen prepared folder')
    return folder.name


def read_case(folder, variant):
    """Authenticate all five sidecars and convert native section seeds."""
    from braids_support.g3_live_bank import combined_export as export
    from braids_support.gpu_fwave.coupled_reference import read_records, read_snapshot

    case = _case(folder, variant)
    folder = Path(folder).resolve()
    require(sha(GATE) == GATE_SHA and sha(PREPARED / 'manifest.json') == MANIFEST_SHA,
            'G3 live-bank frozen gate or manifest differs')
    manifest = json.loads((PREPARED / 'manifest.json').read_text(encoding='utf-8'))
    receipt = export.verify_execution(case)
    require(manifest['receipt_sha256'][case] == sha(folder / 'execution.json') and
            manifest['artifact_sha256'][case] == receipt['artifacts_present'] and
            {path.name for path in folder.glob('*.bin')} == set(export.artifacts(case)),
            'G3 live-bank exact binary set or receipt differs')
    data = read_snapshot(folder)
    for name in export.artifacts(case)[1:]:
        extra = read_records(folder / name)
        for key, value in extra.items():
            require(key not in data or np.array_equal(data[key], value),
                    'G3 live-bank duplicate record differs: ' + name + '/' + key)
            data[key] = value
    nb = int(data['nb'])
    for name in ('section_seed_node', 'boundary_seed_position'):
        values = np.asarray(data['native_' + name])
        require(values.dtype == np.int32 and values.shape == (nb,) and
                np.all(values >= 0), 'G3 live-bank native section seed differs: ' + name)
        data[name] = np.ascontiguousarray(values - 1)
    data['g3_live_bank_case'] = case
    return data, sha(folder / 'execution.json')


def validate_scope(data, *, folder, variant, start, end, interval, dtmax,
                   launch_mode, integrator):
    """Narrow admission of the native 128-cell live-bank engineering fixture."""
    from braids_support.g3_live_bank import native_case
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables
    from braids_support.gpu_fwave.coupled_solver import normalized
    from braids_support.gpu_fwave.normal_outlet_boundary import validate_boundaries
    from braids_support.gpu_fwave.secondary_transport import validate_scope as secondary_scope

    case = _case(folder, variant)
    data = normalized(data)
    require(data.get('g3_live_bank_case') == case and
            launch_mode == 'plain' and integrator == 'ssp33',
            'G3 live-bank requires authenticated plain SSPRK33')
    require(np.isfinite((start, end, interval, dtmax)).all() and
            start == 0. and 0. < end <= 1. and interval == .2 and
            dtmax == .002 and float(data['cfl_target']) == .4 and
            float(data['endtime']) == 1. and float(data['output_dt']) == .2 and
            float(data['dtmax']) == .002,
            'G3 live-bank clock/CFL differs from frozen native scope')
    require(int(data['nc']) == 128 and int(data['nf']) == 280 and
            int(data['nb']) == 2 and int(data['nfrac']) == 2 and
            int(data['nbed_layers']) == 2 and int(data['nsubstrate_layers']) == 1 and
            int(data['scheme']) == 6 and int(data['bedlaw']) == 2 and
            int(data['exchange_mode']) == 0 and
            int(data['cpu_time_integrator']) == 33 and
            int(data['cpu_outlet_extension']) == 0 and
            int(data['mpm_friction_mode']) == 1 and
            int(data['bed_slope']) == 0 and int(data['hiding_exposure']) == 0 and
            float(data['morphological_factor']) == 1. and
            float(data['horizontal_viscosity']) == 0. and
            int(data['nstate']) == 10 and int(data['nledger']) == 6 and
            int(data['secondary_index']) == 9 and
            int(data['secondary_ledger_index']) == 5 and
            int(data['secondary_flow_model']) == 2 and
            int(data['secondary_flow']) == 1 and
            int(data['secondary_flow_transport_direction']) == 1 and
            float(data['secondary_flow_momentum_beta']) == 1. and
            float(data['secondary_flow_espir']) == 1. and
            int(data['bank_failure']) == 1 and
            int(data['bank_toe_erosion_mode']) == VARIANTS[variant],
            'G3 live-bank numeric/secondary/bank controls differ')
    require(np.array_equal(data['mode'], [9, 4]) and
            np.array_equal(data['section_control'], [1, 0]) and
            np.array_equal(np.diff(data['boundary_offset']), [8, 8]) and
            int(np.count_nonzero(data['sampled'])) == 8 and
            int(data['production_section_geometry']) == 1 and
            np.array_equal(data['nhydro_series'], [6, 0]) and
            np.array_equal(data['nfraction_series'], [[6, 0], [6, 0]]) and
            not np.any(data['nseries']),
            'G3 live-bank normal/critical section or schedule shape differs')
    require(np.array_equal(data['hydro_series_time'][:6, 0], native_case.KNOTS) and
            np.array_equal(data['hydro_series_q'][:6, 0], native_case.Q_M3_S) and
            np.array_equal(data['fraction_series_bed_input'][:6, 0, 0],
                           native_case.FINE_FEED_KG_S) and
            np.array_equal(data['fraction_series_bed_input'][:6, 1, 0],
                           native_case.COARSE_FEED_KG_S),
            'G3 live-bank authenticated inlet Q/class-feed values differ')
    require(np.count_nonzero(data['q'][0] <= float(data['dry_tolerance'])) == 20 and
            np.all(data['q'][0] >= 0.) and
            not np.any(data['q'][1:, data['q'][0] <= float(data['dry_tolerance'])]) and
            np.count_nonzero(data['bank_material_mask']) == 64,
            'G3 live-bank initial wet/dry or bank-material mask differs')
    require(bool(data['general_avalanche']) == case.endswith('_on') and
            not np.any(data['fraction_concentration']) and
            not np.any(data['fraction_series_concentration']) and
            not np.any(data['suspended_mass']),
            'G3 live-bank avalanche/suspension controls differ')
    for key in ('amr_enabled', 'bedload_adaptation', 'bedform_roughness',
                'bank_front_tracking', 'bank_avalanche', 'bank_erosion_fraction',
                'nmorfac', 'morphological_acceleration', 'viscosity_equilibrated_inlet',
                'suspended_transport_active'):
        require(not np.any(data.get(key, 0)),
                'G3 live-bank excludes unported control: ' + key)
    for accepted, exported in HISTORY.items():
        values = np.asarray(data[exported])
        require(values.shape == (128,) and np.all(values == 0.) and
                values.dtype == (np.int32 if accepted == 'collapse_count'
                                 else np.float64),
                'G3 live-bank initial bank history differs: ' + exported)
    if VARIANTS[variant] == 2:
        require(int(data['toe_mapping_status']) == 0 and
                float(data['bank_toe_capacity_fraction']) == 1. and
                np.any(data['is_toe_face']),
                'G3 live-bank physical-toe mode 2 differs')
    else:
        require(float(data['bank_distributed_transport_fraction']) == 1. and
                float(data['bank_distributed_critical_slope']) == .1,
                'G3 live-bank distributed-slope mode 3 differs')
    validate_boundaries(data)
    secondary_scope(data)
    tables = validate_tables(dict(data, section_control=np.zeros(2, np.int32),
                                  sampled=np.zeros(280, np.int32)))
    require(np.array_equal(tables['hydro_series_time'][:6, 0], native_case.KNOTS),
            'G3 live-bank boundary table normalization differs')
    events = BoundaryEvents.from_tables(data)
    require(np.array_equal(events.knots, native_case.KNOTS),
            'G3 live-bank event clock differs')
    return events


@contextmanager
def isolated_adapter(*, folder, variant):
    """Load only private generated bridge modules, never mutate shared source."""
    from braids_support.gpu_fwave import a1_combined_current_adapter as ancestor

    require(PRIVATE_KEY not in sys.modules,
            'G3 live-bank private adapter already loaded')
    spec = importlib.util.spec_from_file_location(PRIVATE_KEY, Path(ancestor.__file__))
    require(spec is not None and spec.loader is not None,
            'G3 live-bank bridge source absent')
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_KEY] = module
    try:
        spec.loader.exec_module(module)
        module.validate_input = lambda data: validate_scope(
            data, folder=folder, variant=variant,
            start=float(data.get('time', 0.)), end=float(data['endtime']),
            interval=float(data['output_dt']), dtmax=float(data['dtmax']),
            launch_mode='plain', integrator='ssp33') is not None
        yield module
    finally:
        sys.modules.pop(PRIVATE_KEY, None)


@contextmanager
def constructors(source_folder, data, *, input_folder, variant):
    """Compose private live SSPRK33 stages with existing bank/avalanche GPU kernels."""
    from braids_support.gpu_fwave.boundary_events import tolerance
    from . import a1_full_array_transaction as transaction

    source_folder = Path(source_folder).resolve()
    mode = VARIANTS[variant]
    with isolated_adapter(folder=input_folder, variant=variant) as candidate:
        with candidate.constructors(source_folder, data) as historical:
            core = historical.__mro__[2]
            require(core.__name__ == 'CoupledSolver',
                    'G3 live-bank private spatial base class differs')
            globals_ = core.__init__.__globals__
            stock_scope = globals_['validate_scope']
            schedule_module = sys.modules['braids_support.gpu_fwave._a1_boundary_schedule']
            def admitted_scope(actual, **kwargs):
                require(bool(kwargs.get('bank_avalanche_mode2_variant')) == (mode == 2) and
                        bool(kwargs.get('g3_bank_secondary_variant')) == (mode == 2) and
                        bool(kwargs.get('distributed_bank_avalanche_mode3_variant')) == (mode == 3),
                        'G3 live-bank constructor bank-law flags differ')
                return validate_scope(actual, folder=input_folder, variant=variant,
                    start=float(actual.get('time', 0.)), end=float(actual['endtime']),
                    interval=float(actual['output_dt']), dtmax=float(actual['dtmax']),
                    launch_mode='plain', integrator='ssp33') is not None
            globals_['validate_scope'] = admitted_scope
            try:
                class LiveBankSolver(core):
                    def __init__(self, raw, *, source_folder=source_folder):
                        require(Path(source_folder).resolve() == source_folder,
                                'G3 live-bank private source folder differs')
                        super().__init__(raw, source_folder=source_folder,
                            integrator='ssp33',
                            bank_avalanche_mode2_variant=(mode == 2),
                            g3_bank_secondary_variant=(mode == 2),
                            distributed_bank_avalanche_mode3_variant=(mode == 3))
                        self.boundary.bind_sediment_hydraulic(
                            self.sediment.geometry['bc_hydraulic'])
                        self.boundary.bind_stage_state(self.stepper.stage['q'])
                        self.schedule = schedule_module.ResidentBoundarySchedule(
                            raw, source_path=source_folder / 'boundary_schedule.cu')
                        for accepted, exported in HISTORY.items():
                            self.bank_history[accepted].set(np.asarray(raw[exported]))
                        for records in (self.boundary.normal_records,
                                        self.boundary.outlet_records,
                                        self.schedule.outputs):
                            for value in records.values():
                                value.fill(0)
                        for value in self.boundary.normal_root.cache.values():
                            value.fill(0)
                        self._g3_stage = None
                        self._g3_times = None

                    def euler(self, state, dt, current):
                        stage = self._g3_stage
                        require(type(stage) is int and 0 <= stage < 3 and
                                abs(float(current) - self._g3_times[stage]) <=
                                tolerance(current),
                                'G3 live-bank SSPRK33 Euler stage order differs')
                        slot = STAGE_SLOTS[stage]
                        values = self.schedule.outputs
                        cp = self.cp
                        cp.copyto(self.boundary.arrays['forcing'], values['hydraulic'][slot])
                        cp.copyto(self.sediment.geometry['bc_hydraulic'],
                                  values['hydraulic'][slot])
                        cp.copyto(self.sediment.geometry['feed'], values['feed'][slot])
                        result = super().euler(state, dt, current)
                        self.boundary.record_stage(stage)
                        self._merge(values['status'][slot])
                        self._g3_stage += 1
                        return result

                    def _a1_trial_diagnostics(self):
                        return dict(clock=self.schedule.clock.copy(),
                            schedule={key:value.copy() for key, value in
                                      self.schedule.outputs.items()},
                            normal={key:value.copy() for key, value in
                                    self.boundary.normal_records.items()},
                            outlet={key:value.copy() for key, value in
                                    self.boundary.outlet_records.items()},
                            normal_latch=self.boundary._normal_latch.copy(),
                            outlet_latch=self.boundary._outlet_latch.copy())

                    def _a1_restore_diagnostics(self, saved, *, where=True):
                        cp = self.cp
                        cp.copyto(self.schedule.clock, saved['clock'], where=where)
                        for name, value in saved['schedule'].items():
                            cp.copyto(self.schedule.outputs[name], value, where=where)
                        for name, value in saved['normal'].items():
                            cp.copyto(self.boundary.normal_records[name], value, where=where)
                        for name, value in saved['outlet'].items():
                            cp.copyto(self.boundary.outlet_records[name], value, where=where)
                        cp.copyto(self.boundary._normal_latch, saved['normal_latch'], where=where)
                        cp.copyto(self.boundary._outlet_latch, saved['outlet_latch'], where=where)

                    def step(self, dt, current, *, commit=True):
                        require(self._g3_stage is None,
                                'Concurrent G3 live-bank trials are unsupported')
                        self.boundary.check_storage()
                        saved = self._a1_trial_diagnostics()
                        self._g3_stage = 0
                        self._g3_times = tuple(float(current + fraction * dt)
                                               for fraction in STAGE_FRACTIONS)
                        try:
                            self.boundary.begin_trial()
                            self.schedule.evaluate(current, dt)
                            result = super().step(dt, current, commit=commit)
                            require(self._g3_stage == 3,
                                    'G3 live-bank SSPRK33 omitted an Euler record')
                            self._a1_restore_diagnostics(saved,
                                where=(result['status'] != 0) if commit else True)
                            return result
                        except Exception:
                            self._a1_restore_diagnostics(saved)
                            raise
                        finally:
                            self._g3_stage = None
                            self._g3_times = None

                yield transaction.wrap_solver(LiveBankSolver)
            finally:
                globals_['validate_scope'] = stock_scope
