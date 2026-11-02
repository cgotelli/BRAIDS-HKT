"""Guarded live seven-class bedload-only adaptation SSPRK33 composition.

This exact native-reference engineering route reuses the current augmented
f-wave, MPM, stratigraphy and classwise adaptation CUDA kernels unchanged.
It is not a general solver admission, QGIS release or physical braids_support.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

import numpy as np

VARIANT = 'live_normal_critical_graded_adaptation_ssp33'
PRIVATE_KEY = 'braids_support.gpu_fwave._target_b_live_adaptation_adapter'
STAGE_SLOTS = (0, 2, 1)
STAGE_FRACTIONS = (0., 1., .5)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_case(folder):
    from .live_secondary import read_case as read_live
    return read_live(folder)


def verify_reference(input_path):
    from braids_support.g2_target_b import gpu_scope
    return gpu_scope.verify_reference(input_path)


def validate_scope(data, *, start, end, interval, dtmax, launch_mode, integrator):
    from braids_support.g2_target_b import gpu_scope
    return gpu_scope.validate_scope(data, start=start, end=end, interval=interval,
                                    dtmax=dtmax, launch_mode=launch_mode,
                                    integrator=integrator)


GENERAL_VARIANT = 'live_normal_critical_graded_bedload_ssp33'
ADAPTATION_EXPORTER_SHA256 = 'a50f91dd99e5b48af94535761a7ec2f5b933e1e21ca9479370e78d03275eac33'
ADAPTATION_EXPORTER = (Path(__file__).resolve().parents[1] / 'validation' /
    'gpu_fwave' / 'runs' / 'i2_adaptation_export_20261004_v1' /
    'runtime' / 'xbraids_normal_export.exe')


def validate_general_scope(data, *, start, end, interval, dtmax,
                           launch_mode, integrator):
    """Input-contract admission, independent of a validation archive path.

    This is deliberately a bounded physics scope, not a frozen geometry or
    forcing fixture.  A preparation receipt binds the actual binary, section
    sidecar and NPZ; this validator checks their decoded numerical controls.
    """
    from braids_support.gpu_fwave.boundary_events import BoundaryEvents
    from braids_support.gpu_fwave.boundary_schedule import validate_tables
    from braids_support.gpu_fwave.coupled_solver import normalized
    from braids_support.gpu_fwave.normal_outlet_boundary import validate_boundaries
    from braids_support.gpu_fwave.secondary_transport import validate_scope as secondary_scope

    data = normalized(data)
    require(launch_mode == 'plain' and integrator == 'ssp33' and
            np.isfinite([start, end, interval, dtmax]).all() and
            start >= 0 and end > start and interval > 0 and dtmax > 0 and
            (start, end, interval, dtmax) ==
            (float(data.get('time', 0.)), float(data['endtime']),
             float(data['output_dt']), float(data['dtmax'])),
            'General graded bedload requires prepared plain SSPRK33 clock')
    nc = int(data['nc']); nf = int(data['nf']); nb = int(data['nb'])
    sections = np.diff(np.asarray(data['boundary_offset']))
    require(nc > 0 and nf > 0 and nb == 2 and
            np.array_equal(data['mode'], [9, 4]) and
            np.array_equal(data['section_control'], [1, 0]) and
            sections.shape == (2,) and np.all(sections >= 2) and
            int(data['production_section_geometry']) == 1 and
            int(np.count_nonzero(data['sampled'])) == int(sections[0]),
            'General graded bedload requires real normal9/critical4 sections')
    ng = int(data['nfrac']); nlayer = int(data['nbed_layers'])
    require(ng == 7 and nlayer == 4 and
            int(data['nsubstrate_layers']) == 3 and
            int(data['nstate']) == 38 and int(data['nledger']) == 15 and
            int(data['secondary_index']) == -1 and
            np.asarray(data['q']).shape == (3, nc) and
            np.asarray(data['layer_mass']).shape == (4, 7, nc),
            'General graded bedload requires seven classes and four real layers')
    require(int(data['scheme']) == 6 and int(data['bedlaw']) == 2 and
            int(data['mpm_friction_mode']) == 1 and
            int(data['exchange_mode']) == 0 and
            int(data['hiding_exposure']) == 4 and
            int(data['bed_slope']) == 2 and
            float(data['morphological_factor']) == 1. and
            int(data['cpu_time_integrator']) == 33 and
            int(data['cpu_outlet_extension']) == 0 and
            int(data['secondary_flow_model']) == 0 and
            float(data['horizontal_viscosity']) == 0. and
            0. < float(data['cfl_target']) <= .8,
            'General graded bedload numerical controls are unsupported')
    diameters = np.asarray(data['d50_fraction'])
    require(diameters.shape == (7,) and np.isfinite(diameters).all() and
            np.all(diameters > 0.) and np.all(np.diff(diameters) > 0.) and
            not np.any(data['sediment_type_fraction']) and
            int(data['sediment_roughness_mode']) == 1 and
            np.array_equal(data['sediment_roughness_fraction'],
                           np.ones(7, dtype=np.int32)),
            'General graded bedload requires ordered noncohesive classes')
    require(not np.any(data['suspended_mass']) and
            not np.any(data['fraction_concentration']) and
            not np.any(data['fraction_series_concentration']) and
            not bool(data['suspended_transport_active']) and
            not np.any(data['ws_fraction']) and
            not np.any(data['secondary_mass']),
            'General graded bedload excludes suspended/secondary inventory')
    for key in ('amr_enabled', 'bedform_roughness', 'bank_failure',
                'bank_avalanche', 'general_avalanche', 'bank_front_tracking',
                'bank_erosion_fraction', 'nmorfac',
                'viscosity_equilibrated_inlet', 'active_layer_mode'):
        require(not np.any(data.get(key, 0)),
                'General graded bedload excludes unsupported process ' + key)
    flag = int(data['bedload_adaptation'])
    length = float(data.get('bedload_adaptation_length_diameters', 0.))
    require(flag in (0, 1) and np.isfinite(length) and
            (flag == 0 or length > 0.),
            'General graded bedload adaptation switch/length differs')
    q = np.asarray(data['q']); layers = np.asarray(data['layer_mass'])
    dry = float(data['dry_tolerance'])
    require(np.isfinite(q).all() and np.min(q[0]) >= 0. and
            np.any(q[0] <= dry) and np.any(q[0] > dry) and
            not np.any(q[1:, q[0] <= dry]) and
            np.isfinite(layers).all() and np.min(layers) >= 0.,
            'General graded bedload requires finite mixed wet/dry layered state')
    validate_boundaries(data)
    secondary_scope(data)
    tables = validate_tables(dict(data,
        section_control=np.zeros(2, np.int32),
        sampled=np.zeros(nf, np.int32)))
    count = int(np.asarray(data['nhydro_series'])[0])
    knots = np.asarray(tables['hydro_series_time'][:count, 0])
    feed_count = np.asarray(data['nfraction_series'])
    require(count >= 2 and np.array_equal(data['nhydro_series'], [count, 0]) and
            feed_count.shape == (7, 2) and
            np.array_equal(feed_count[:, 0], np.full(7, count)) and
            not np.any(feed_count[:, 1]) and
            not np.any(data['nseries']) and
            knots[0] == start and knots[-1] == end and
            np.all(np.diff(knots) > 0) and
            np.array_equal(tables['fraction_series_time'][:count, :, 0],
                           np.broadcast_to(knots[:, None], (count, 7))) and
            np.isfinite(tables['hydro_series_q'][:count, 0]).all() and
            np.all(tables['hydro_series_q'][:count, 0] >= 0) and
            np.any(tables['hydro_series_q'][:count, 0] > 0) and
            np.isfinite(tables['fraction_series_bed_input'][:count, :, 0]).all() and
            np.all(tables['fraction_series_bed_input'][:count, :, 0] >= 0),
            'General graded bedload requires common live Q/class-feed knots')
    events = BoundaryEvents.from_tables(data)
    require(np.array_equal(events.knots, knots),
            'General graded bedload event schedule differs from prepared tables')
    return events


def verify_general_prepared(input_path, data):
    """Authenticate a regular versioned preparation manifest, without path pins."""
    import json
    from .runner import sha
    from . import graded_native_exporter as exporter
    folder = Path(input_path).resolve()
    receipt_path = folder / 'preparation_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    required = {'coupled_input.bin', 'coupled_input.npz', 'normal_metadata.bin'}
    require(receipt.get('schema') == 3 and
            receipt.get('variant') == GENERAL_VARIANT and
            set(receipt.get('artifacts', {})) == required and
            not any((folder / name).exists() for name in
                ('scheduled_metadata.bin', 'bank_toe_metadata.bin',
                 'distributed_bank_metadata.bin', 'avalanche_metadata.bin',
                 'coupled_diffusion.bin')),
            'General graded bedload requires three authenticated prepared artifacts')
    for name, digest in receipt['artifacts'].items():
        path = (folder / name).resolve()
        require(path.parent == folder and path.is_file() and sha(path) == digest,
                'General graded bedload prepared artifact differs: ' + name)
    worker = Path(__file__).resolve().parents[1]
    if (worker / 'PACKAGE-MANIFEST.json').is_file():
        authenticated = exporter.verify_packaged_exporter()
        executable_sha = authenticated['executable_sha256']
    else:
        require(ADAPTATION_EXPORTER.is_file() and
                sha(ADAPTATION_EXPORTER) == ADAPTATION_EXPORTER_SHA256,
                'General graded bedload live-checkout exporter differs')
        executable_sha = ADAPTATION_EXPORTER_SHA256
    require(executable_sha == exporter.EXPORTER_SHA256 and
            receipt.get('exporter_executable_sha256') == executable_sha and
            receipt.get('initialization_CPU_only') is True,
            'General graded bedload zero-step exporter identity differs')
    origin = receipt.get('origin', {})
    inputs = origin.get('files', {}) if isinstance(origin, dict) else {}
    require(isinstance(inputs, dict) and
            {'case.data', 'case.data.bin', 'geometry.npz'} <= set(inputs),
            'General graded bedload native deck/geometry identity is absent')
    for name, digest in inputs.items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder) and path.is_file() and
                sha(path) == digest,
                'General graded bedload native deck/geometry differs: ' + name)
    require((receipt.get('native_solver_steps_performed',
             receipt.get('solver_steps_performed')) == 0) and
            receipt.get('GPU_solver_executed', False) is False,
            'General graded bedload requires zero-step native preparation')
    with np.load(folder / 'coupled_input.npz', allow_pickle=False) as archive:
        require(set(archive.files) <= set(data) and
                all(np.array_equal(archive[key], data[key]) for key in archive.files),
                'General graded bedload NPZ differs from native binary/sidecar')
    return dict(contract_schema=1, preparation_receipt_sha256=sha(receipt_path),
        prepared_input_sha256=sha(folder / 'coupled_input.npz'),
        normal_metadata_sha256=sha(folder / 'normal_metadata.bin'),
        archive_bound=False, cells=int(data['nc']), faces=int(data['nf']),
        grain_classes=int(data['nfrac']), bed_layers=int(data['nbed_layers']),
        zero_step_native_preparation=True)


@contextmanager
def isolated_adapter(*, scope_validator=None):
    """Generate the private bridge under an explicit bounded host validator."""
    from braids_support.gpu_fwave import a1_combined_current_adapter as ancestor
    validator = validate_scope if scope_validator is None else scope_validator
    require(PRIVATE_KEY not in sys.modules,
            'Target-B live adaptation adapter is already loaded')
    spec = importlib.util.spec_from_file_location(PRIVATE_KEY, Path(ancestor.__file__))
    require(spec is not None and spec.loader is not None,
            'Target-B live spatial bridge source is absent')
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_KEY] = module
    try:
        spec.loader.exec_module(module)
        module.validate_input = lambda data: validator(
            data, start=float(data.get('time', 0.)),
            end=float(data['endtime']), interval=float(data['output_dt']),
            dtmax=float(data['dtmax']), launch_mode='plain',
            integrator='ssp33') is not None
        yield module
    finally:
        sys.modules.pop(PRIVATE_KEY, None)


def _spatially_validated_scope(actual, stock_scope, validator, **kwargs):
    """Apply stock material checks to a validation-only boundary-neutral view."""
    validator(actual, start=float(actual.get('time', 0.)),
        end=float(actual['endtime']), interval=float(actual['output_dt']),
        dtmax=float(actual['dtmax']), launch_mode='plain', integrator='ssp33')
    check = dict(actual,
        mode=np.array([6, 2], np.int32),
        section_control=np.zeros(2, np.int32),
        sampled=np.zeros(int(actual['nf']), np.int32),
        nhydro_series=np.zeros(2, np.int32),
        nfraction_series=np.zeros((int(actual['nfrac']), 2), np.int32),
        bedload_adaptation=np.int32(0))
    neutral_kwargs = dict(kwargs, bedload_adaptation_variant=False)
    return stock_scope(check, **neutral_kwargs)


@contextmanager
def constructors(folder, data, *, scope_validator=None,
                 selected_gate_sha256=None):
    """Private live SSPRK33 solver with optional native classwise adaptation."""
    from braids_support.gpu_fwave.boundary_events import tolerance
    folder = Path(folder).resolve()
    validator = validate_scope if scope_validator is None else scope_validator
    if selected_gate_sha256 is None:
        from braids_support.g2_target_b import gpu_scope
        selected_gate = gpu_scope.GATE_SHA
    else:
        selected_gate = selected_gate_sha256
    if validator is validate_general_scope:
        from . import graded_bedload_prebuilt
        adapter_context = graded_bedload_prebuilt.adapter()
    else:
        adapter_context = isolated_adapter(scope_validator=validator)
    with adapter_context as candidate:
        with candidate.constructors(folder, data) as historical:
            core = historical.__mro__[2]
            require(core.__name__ == 'CoupledSolver',
                    'Target-B live spatial base class differs')
            globals_ = core.__init__.__globals__
            stock_scope = globals_['validate_scope']
            schedule_module = sys.modules['braids_support.gpu_fwave._a1_boundary_schedule']
            globals_['validate_scope'] = lambda actual, **kwargs: (
                _spatially_validated_scope(actual, stock_scope, validator,
                                           **kwargs))
            try:
                class LiveBedloadAdaptationSolver(core):
                    def __init__(self, raw, *, source_folder=folder):
                        require(Path(source_folder).resolve() == folder,
                                'Target-B private source folder differs')
                        enabled = bool(int(raw['bedload_adaptation']))
                        super().__init__(raw, source_folder=folder,
                            integrator='ssp33',
                            bedload_adaptation_variant=enabled)
                        self.boundary.bind_sediment_hydraulic(
                            self.sediment.geometry['bc_hydraulic'])
                        self.boundary.bind_stage_state(self.stepper.stage['q'])
                        self.schedule = schedule_module.ResidentBoundarySchedule(
                            raw, source_path=folder / 'boundary_schedule.cu')
                        for records in (self.boundary.normal_records,
                                        self.boundary.outlet_records,
                                        self.schedule.outputs):
                            for value in records.values():
                                value.fill(0)
                        for value in self.boundary.normal_root.cache.values():
                            value.fill(0)
                        self._target_b_stage = None
                        self._target_b_times = None

                    def euler(self, state, dt, current):
                        stage = self._target_b_stage
                        require(type(stage) is int and 0 <= stage < 3 and
                                abs(float(current)-self._target_b_times[stage]) <=
                                tolerance(current),
                                'Target-B live SSPRK33 Euler stage time/order changed')
                        slot = STAGE_SLOTS[stage]
                        values = self.schedule.outputs
                        cp = self.cp
                        cp.copyto(self.boundary.arrays['forcing'],
                                  values['hydraulic'][slot])
                        cp.copyto(self.sediment.geometry['bc_hydraulic'],
                                  values['hydraulic'][slot])
                        cp.copyto(self.sediment.geometry['feed'],
                                  values['feed'][slot])
                        # Exchange mode 0 has no suspended boundary buffer.
                        # The authenticated concentration schedule is exactly
                        # zero and is still checked in the host stage gate.
                        if self.has_suspended:
                            cp.copyto(self.bc_concentration,
                                      values['concentration'][slot])
                        result = super().euler(state, dt, current)
                        self.boundary.record_stage(stage)
                        self._merge(values['status'][slot])
                        self._target_b_stage += 1
                        return result

                    def _a1_trial_diagnostics(self):
                        return dict(clock=self.schedule.clock.copy(),
                            schedule={key:value.copy() for key,value in
                                      self.schedule.outputs.items()},
                            normal={key:value.copy() for key,value in
                                    self.boundary.normal_records.items()},
                            outlet={key:value.copy() for key,value in
                                    self.boundary.outlet_records.items()},
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
                        cp.copyto(self.boundary._normal_latch,
                                  saved['normal_latch'], where=where)
                        cp.copyto(self.boundary._outlet_latch,
                                  saved['outlet_latch'], where=where)

                    def step(self, dt, current, *, commit=True):
                        require(self._target_b_stage is None,
                                'Concurrent target-B live trials are unsupported')
                        self.boundary.check_storage()
                        saved = self._a1_trial_diagnostics()
                        self._target_b_stage = 0
                        self._target_b_times = tuple(
                            float(current+fraction*dt)
                            for fraction in STAGE_FRACTIONS)
                        try:
                            self.boundary.begin_trial()
                            self.schedule.evaluate(current, dt)
                            result = super().step(dt, current, commit=commit)
                            require(self._target_b_stage == 3,
                                    'Target-B SSPRK33 omitted an Euler record')
                            self._a1_restore_diagnostics(saved,
                                where=(result['status'] != 0) if commit else True)
                            return result
                        except Exception:
                            self._a1_restore_diagnostics(saved)
                            raise
                        finally:
                            self._target_b_stage = None
                            self._target_b_times = None

                    def candidate_receipt(self):
                        self.boundary.check_storage()
                        return dict(boundary=self.boundary.receipt(),
                            schedule=self.schedule.receipt(),
                            integrator='SSPRK33', euler_slots=list(STAGE_SLOTS),
                            selected_gate_sha256=selected_gate,
                            adaptation_enabled=self.bedload_adaptation_variant,
                            grain_classes=7, bed_layers=4,
                            accepted_transaction='segmented_all_reachable_v5',
                            native_controls_retained=True, backend='gpu',
                            automatic_CPU_fallback=False,
                            physical_validation=False)

                yield LiveBedloadAdaptationSolver
            finally:
                globals_['validate_scope'] = stock_scope
