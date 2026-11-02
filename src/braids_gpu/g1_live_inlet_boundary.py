"""G1 mode-9 inlet bridge for one fixed-grid SSPRK33 Euler evaluation.

The caller supplies the archived outlet-only bridge and its scope-extended
parent.  The numerical mode-9 trace and critical section root are existing CUDA
components.  This wrapper retains the real control code (0 or 2), uses the
current Euler bed and stable stage-q storage, and never evaluates a root on CPU.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from braids_support.gpu_fwave import live_critical_inlet as critical_component
from braids_support.gpu_fwave import live_normal_trace as trace_component
from braids_support.gpu_fwave import live_section_metrics as metric_component
from braids_support.gpu_fwave.time_integration import OPTIONS


BRIDGE_CUDA_SHA = '8dec7824ee6ec1b5f46d83be917b6a1a3bce97ac7730731f8836468d7e68fcb5'
TRACE_CUDA_SHA = critical_component.FROZEN['trace_cuda']
CRITICAL_CUDA_SHA = critical_component.FROZEN['root_cuda']
METRIC_CUDA_SHA = critical_component.FROZEN['metric_cuda']
STAGE_METRIC_CUDA_SHA = '71e7e8131c71cdebf525f7dc72df43919cbc054aab7f0105b4db95cacd178c8a'
KERNELS = ('lnb_gather', 'lnb_stage', 'lnb_scatter', 'lnb_merge', 'lnb_record')
require = trace_component.require
identity = trace_component.identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_boundary_controls(ParentControls, outlet_module):
    """Return the composed G1 inlet/mode-4 outlet BoundaryControls class.

    ``outlet_module`` is the private generated outlet bridge whose public
    methods were renamed to ``_..._outlet``.  Its parent applies the unchanged
    existing boundary kernels once.  The caller must bind the stable
    ``stepper.stage['q']`` and sediment ``bc_hydraulic`` buffers after solver
    construction and call ``begin_trial`` before each three-Euler SSPRK33 step.
    """
    Outlet = outlet_module.make_boundary_controls(ParentControls)

    class G1LiveInletBoundary(Outlet):
        def __init__(self, data, inputs, source=None, **kwargs):
            from braids_gpu.live_inlet_modes import validate_boundaries

            sections = validate_boundaries(data)
            inlet = sections['inlet']
            mode = np.asarray(data['mode'])
            control = np.asarray(data['section_control'])
            require(np.array_equal(mode, [9, 4]) and int(control[0]) in (0, 2),
                    'G1 inlet requires native mode9/control0 or control2 and mode4 outlet')
            require(source is not None, 'Explicit archived parent CUDA source required')
            folder = Path(source).resolve().parent
            required = {'live_normal_boundary.cu': BRIDGE_CUDA_SHA,
                        'live_normal_trace.cu': TRACE_CUDA_SHA}
            if int(control[0]) == 2:
                required.update({'live_critical_resolved.cu': CRITICAL_CUDA_SHA,
                                 'live_normal_section.cu': METRIC_CUDA_SHA})
            else:
                required['live_section_metrics.cu'] = STAGE_METRIC_CUDA_SHA
            for name, digest in required.items():
                require((folder / name).is_file() and sha(folder / name) == digest,
                        'Archived G1 numerical source differs: ' + name)
            super().__init__(data, inputs, source=source, **kwargs)

            cp = self.cp
            self.inlet_sections = inlet
            self.inlet_control = int(control[0])
            self._g1_device = cp.cuda.Device().id
            self._g1_ids = cp.asarray(inlet['global_boundary_ids'])
            self._g1_n = len(self._g1_ids)
            require(self._g1_n == 1, 'G1 requires exactly one surveyed inlet')
            self._g1_Q = cp.zeros(self._g1_n, dtype=cp.float64)
            self._g1_stage = cp.zeros(self._g1_n, dtype=cp.float64)
            self._g1_root_status = cp.zeros(self._g1_n, dtype=cp.int32)
            self._g1_latch = cp.zeros(self._g1_n, dtype=cp.int32)
            self._g1_sediment = None
            self._g1_q = None
            self.inlet_component = None
            self.inlet_root = None
            self.inlet_trace = None
            self.inlet_metrics = None
            self._g1_ready_storage = None
            self._g1_next_stage = None
            self._g1_geometry = dict(inlet['geometry'], production_section_geometry=True)
            self._g1_bridge_source = folder / 'live_normal_boundary.cu'
            self._g1_source_sha = sha(self._g1_bridge_source)
            self._g1_python_sha = sha(__file__)
            self._g1_parent_source = Path(source).resolve()
            self._g1_parent_sha = sha(source)
            self._g1_parent_python = Path(inspect.getfile(ParentControls)).resolve()
            self._g1_parent_python_sha = sha(self._g1_parent_python)
            module = cp.RawModule(code=self._g1_bridge_source.read_text(),
                                  options=OPTIONS, name_expressions=KERNELS)
            self._g1_kernels = {name: module.get_function(name) for name in KERNELS}
            nfaces = len(inlet['geometry']['faces'])
            self.inlet_records = dict(
                bed=cp.empty((3, self.nc)),
                nodes=cp.empty((3, nfaces + self._g1_n)),
                stage=cp.empty((3, self._g1_n)),
                discharge=cp.empty((3, self._g1_n)),
                trace=cp.empty((3, 3, nfaces)),
                root_status=cp.empty((3, self._g1_n), dtype=cp.int32),
                trace_status=cp.empty((3, self._g1_n), dtype=cp.int32),
                failure_latch=cp.empty((3, self._g1_n), dtype=cp.int32),
                critical_quality=cp.full((3, self._g1_n), -1, dtype=cp.int32),
                area=cp.empty((3, self._g1_n)),
                top_width=cp.empty((3, self._g1_n)),
                metric_status=cp.empty((3, self._g1_n), dtype=cp.int32),
                discrete_wet_area=cp.empty((3, self._g1_n)),
                trace_speed=cp.empty((3, self._g1_n)),
                inward_discharge=cp.empty((3, self._g1_n)),
                recorded=cp.zeros(3, dtype=cp.int32))
            if self.inlet_control == 2:
                self.inlet_records['residual'] = cp.empty((3, self._g1_n))
            self._g1_static_device = {}
            for key, host_key in (('owner', 'owner'), ('normal', 'normal'),
                                  ('length', 'length'), ('sampled', 'sampled'),
                                  ('bedoffset', 'boundary_bed_offset')):
                require(key in inputs, 'Actual coupled geometry buffer missing: ' + key)
                actual = cp.asnumpy(inputs[key])
                expected = np.asarray(data[host_key])
                require(actual.dtype == expected.dtype and np.array_equal(actual, expected),
                        'Actual coupled geometry differs from G1 input: ' + key)
                self._g1_static_device[key] = identity(actual)
            require('trace' in inputs and inputs['trace'].shape == (10, self.nf),
                    'Actual ten-slot reconstruction trace required')
            self._g1_host_inputs = {key: identity(value) for key, value in data.items()
                                    if isinstance(value, np.ndarray) and value.dtype.kind in 'biuf'}
            self._g1_initial_storage = self._g1_storage_identity()

        def _g1_storage_identity(self):
            values = [('ids', self._g1_ids), ('Q', self._g1_Q),
                      ('stage', self._g1_stage), ('root_status', self._g1_root_status),
                      ('latch', self._g1_latch),
                      ('status', self.status), ('face_status', self.face_status),
                      ('ledger', self.ledger)]
            values += [('parent:' + key, value) for key, value in self.arrays.items()]
            values += [('record:' + key, value) for key, value in self.inlet_records.items()]
            values += [('input:' + key, self.inputs[key]) for key in
                       ('bed', 'normal', 'length', 'owner', 'sampled', 'bedoffset', 'trace')]
            if self._g1_sediment is not None:
                values.append(('sediment', self._g1_sediment))
            if self._g1_q is not None:
                values.append(('q', self._g1_q))
            return tuple((key, id(value), value.data.ptr, value.shape, value.dtype.str,
                          bool(value.flags.c_contiguous), value.device.id) for key, value in values)

        def _g1_bind(self, name, value, shape):
            cp = self.cp
            require(not cp.cuda.get_current_stream().is_capturing(),
                    'Bind G1 buffers before CUDA capture')
            require(self._g1_ready_storage is None and getattr(self, name) is None,
                    'G1 buffers can be bound only once')
            require(tuple(row for row in self._g1_storage_identity()
                          if row[0] not in ('sediment', 'q')) == self._g1_initial_storage,
                    'G1 construction storage changed before binding')
            require(isinstance(value, cp.ndarray) and value.shape == shape and
                    value.dtype == cp.float64 and value.flags.c_contiguous and
                    value.device.id == self._g1_device,
                    'Canonical resident G1 buffer required: ' + name)
            setattr(self, name, value)

        def bind_sediment_hydraulic(self, device):
            self._g1_bind('_g1_sediment', device, (2, self.nb))
            self._g1_finish_binding()

        def bind_stage_state(self, q):
            self._bind_stage_state_outlet(q)
            self._g1_bind('_g1_q', q, (3, self.nc))
            self._g1_finish_binding()

        def _g1_finish_binding(self):
            if self._g1_sediment is None or self._g1_q is None:
                return
            folder = self._g1_bridge_source.parent
            if self.inlet_control == 2:
                self.inlet_component = critical_component.ResidentCriticalInlet(
                    self._g1_geometry, self.inlet_sections['controls'],
                    self.inputs['bed'], self._g1_q, self._g1_Q,
                    gravity=float(self.grav), dry=float(self.dry),
                    speed_limit=float(self.speed),
                    root_source_path=folder / 'live_critical_resolved.cu',
                    metric_source_path=folder / 'live_normal_section.cu',
                    trace_source_path=folder / 'live_normal_trace.cu')
                self.inlet_root = self.inlet_component.root
                self.inlet_trace = self.inlet_component.trace_component
            else:
                self.inlet_metrics = metric_component.ResidentSectionMetrics(
                    self._g1_geometry, self.inputs['bed'], self._g1_stage,
                    source_path=folder / 'live_section_metrics.cu')
                self.inlet_trace = trace_component.ResidentNormalTrace(
                    self._g1_geometry, self.inputs['bed'], self._g1_q,
                    self._g1_stage, self._g1_root_status, self._g1_Q,
                    dry=float(self.dry), speed_limit=float(self.speed),
                    source_path=folder / 'live_normal_trace.cu')
            self._g1_component_ids = tuple(id(item) for item in
                                           (self.inlet_component, self.inlet_root,
                                            self.inlet_trace, self.inlet_metrics))
            for key, value in self.inlet_records.items():
                value.fill(-1 if key == 'critical_quality' else 0)
            for value in self.outlet_records.values():
                value.fill(0)
            if self.inlet_root is not None:
                for value in self.inlet_root.cache.values():
                    value.fill(0)
            self._g1_ready_storage = self._g1_storage_identity()
            self._g1_scalar_identity = tuple(float(getattr(self, key))
                                             for key in ('dry', 'grav', 'speed', 'chezy'))

        def check_storage(self):
            self._check_storage_outlet()
            require(self.cp.cuda.Device().id == self._g1_device,
                    'Current CUDA device differs from G1 inlet')
            require(self._g1_ready_storage is not None and
                    self._g1_storage_identity() == self._g1_ready_storage,
                    'Captured G1 inlet storage changed or is not bound')
            require(tuple(float(getattr(self, key)) for key in
                          ('dry', 'grav', 'speed', 'chezy')) == self._g1_scalar_identity,
                    'Captured G1 scalar controls changed')
            require(tuple(id(item) for item in (self.inlet_component, self.inlet_root,
                                                self.inlet_trace, self.inlet_metrics)) == self._g1_component_ids,
                    'Captured G1 component bindings changed')
            if self.inlet_component is not None:
                self.inlet_component.check_storage()
            else:
                self.inlet_trace.check_storage()
                self.inlet_metrics.check_storage()
            return True

        def begin_trial(self):
            self.check_storage()
            self._begin_trial_outlet()
            self._g1_latch.fill(0)
            self.inlet_records['recorded'].fill(0)
            self._g1_next_stage = 0

        def apply(self, q, trace):
            self.check_storage()
            require(self._g1_next_stage is not None and self._g1_next_stage < 3,
                    'G1 requires begin_trial and exactly three SSPRK33 Euler stages')
            require(q is self._g1_q and trace is self.inputs['trace'] and
                    trace.shape == (10, self.nf) and trace.dtype == self.cp.float64 and
                    trace.flags.c_contiguous and trace.device.id == self._g1_device,
                    'G1 requires bound current Euler q and ten-slot reconstruction trace')
            n = np.int32(self._g1_n)
            grid = ((self._g1_n + 127) // 128,)
            self._g1_kernels['lnb_gather'](
                grid, (128,), (self.arrays['forcing'], self._g1_ids, n, self._g1_Q))
            if self.inlet_control == 2:
                result = self.inlet_component.evaluate()
                root, values = result['root'], result['trace']
            else:
                self._g1_root_status.fill(0)
                self._g1_kernels['lnb_gather'](
                    grid, (128,), (self.arrays['forcing'][1], self._g1_ids, n,
                                   self._g1_stage))
                self.inlet_metrics.evaluate()
                root = {'stage': self._g1_stage, 'status': self._g1_root_status}
                values = self.inlet_trace.evaluate()
            self._g1_kernels['lnb_stage'](
                grid, (128,), (root['stage'], root['status'], self._g1_ids, n,
                               np.int32(self.nb), self.arrays['forcing'],
                               self._g1_sediment))
            # The outlet bridge resets the shared statuses exactly once.
            self._apply_outlet(q, trace)
            faces = self.inlet_trace._arrays['faces']
            count = np.int32(len(faces))
            self._g1_kernels['lnb_scatter'](
                ((int(count) + 127) // 128,), (128,),
                (faces, count, np.int32(self.nf), values['trace'],
                 values['face_status'], trace, self.face_status))
            self._g1_kernels['lnb_merge'](
                grid, (128,), (self._g1_ids, n, root['status'], values['status'],
                               self.status, self._g1_latch))
            self._g1_next_stage += 1

        def record_stage(self, index):
            self.check_storage()
            require(type(index) is int and 0 <= index < 3 and
                    self._g1_next_stage == index + 1,
                    'G1 stage records must follow the three SSPRK33 Euler evaluations')
            self._record_stage_outlet(index)
            cp = self.cp
            s = self.inlet_records
            root = self.inlet_root.outputs if self.inlet_root is not None else {
                'stage': self._g1_stage, 'status': self._g1_root_status}
            trace = self.inlet_trace.outputs
            for key, value in (('bed', self.inputs['bed']), ('stage', root['stage']),
                               ('discharge', self._g1_Q), ('root_status', root['status']),
                               ('trace_status', trace['status']),
                               ('failure_latch', self._g1_latch),
                               ('discrete_wet_area', trace['discrete_wet_area']),
                               ('trace_speed', trace['speed']),
                               ('inward_discharge', trace['inward_discharge'])):
                cp.copyto(s[key][index], value)
            if self.inlet_root is not None:
                for key in ('nodes', 'area', 'top_width', 'residual'):
                    cp.copyto(s[key][index], root[key])
                cp.copyto(s['critical_quality'][index], root['quality'])
                s['metric_status'][index:index + 1].fill(0)
            else:
                for key in ('nodes', 'area', 'top_width'):
                    cp.copyto(s[key][index], self.inlet_metrics.outputs[key])
                cp.copyto(s['metric_status'][index], self.inlet_metrics.outputs['status'])
            faces = self.inlet_trace._arrays['faces']
            self._g1_kernels['lnb_record'](
                ((len(faces) + 127) // 128,), (128,),
                (faces, np.int32(len(faces)), np.int32(self.nf),
                 self.inputs['trace'], s['trace'][index]))
            s['recorded'][index:index + 1].fill(1)

        def audit(self):
            self.check_storage()
            outlet = self._audit_outlet()
            if self.inlet_component is not None:
                self.inlet_component.verify_geometry()
            else:
                self.inlet_trace.verify_geometry()
                self.inlet_metrics.verify_geometry()
            require(all(identity(self.cp.asnumpy(self.inputs[key])) == value
                        for key, value in self._g1_static_device.items()),
                    'Static coupled G1 geometry changed')
            require(sha(__file__) == self._g1_python_sha and
                    sha(self._g1_bridge_source) == self._g1_source_sha and
                    sha(self._g1_parent_source) == self._g1_parent_sha and
                    sha(self._g1_parent_python) == self._g1_parent_python_sha,
                    'Executed G1 bridge or parent source changed')
            return dict(inlet=True, outlet=outlet)

        def receipt(self):
            return dict(backend='gpu', automatic_CPU_fallback=False,
                        control=self.inlet_control, mode=9, outlet_mode=4,
                        source_sha256=self._g1_source_sha,
                        python_sha256=self._g1_python_sha,
                        parent_cuda_sha256=self._g1_parent_sha,
                        parent_python_sha256=self._g1_parent_python_sha,
                        native_input_identities=self._g1_host_inputs,
                        inlet_component=(self.inlet_component.receipt()
                                         if self.inlet_component is not None else
                                         self.inlet_trace.receipt() if self.inlet_trace else None),
                        outlet=self._receipt_outlet(),
                        three_ssprk33_euler_records=True,
                        current_Euler_bed_and_q=True,
                        resolved_critical_method_if_control2=self.inlet_control == 2,
                        prescribed_stage_if_control0=self.inlet_control == 0,
                        stage_and_discharge_from_resident_schedule=True,
                        sediment_hydraulic_stage_scattered=True,
                        accepted_flux_ledger_uses_parent_augmented_fwave=True,
                        inlet_root_and_trace_status_latched=True,
                        coupled_qualification_pending=True)

    return G1LiveInletBoundary
