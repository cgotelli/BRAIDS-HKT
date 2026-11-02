"""Storage/input guards around the frozen scheduled-coupled adapter.

No numerical/source transform is added. Checkpoint input identity is a stable
hash of immutable retained schedule values, not CUDA addresses. Pointer guards
are process-local and recaptured by each fresh restart process. Auditing device
table contents is explicit; ordinary steps do not download or hash large tables.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from . import scheduled_coupled as previous
from . import boundary_schedule as component

PARENT_SHA256 = 'bb77190e36267d2f7e4b6ac8c1f8e7636cc08ad12e3d76ee4fa9e1a6dddba9fd'
COMPONENT_SHA256 = '53035c9459d0f068a7f852e9f4c1803d3365be1ed2c9ee8b2d3f40e4dddb2be0'
KERNEL_SHA256 = 'e44b263653855f37d20d35e2a14fe180d7fe8437441ea019fbf63aba4692894a'
INPUT_KEYS = ('deck_version', 'mode', 'forcing', 'fraction_concentration', 'fraction_bed_input',
    'nseries', 'nhydro_series', 'nfraction_series', 'nmorfac', 'section_control', 'sampled',
    'bed_boundary_mode', 'bc_hydraulic', 'bc_bed_input',
    *tuple(prefix+suffix for _, prefix, suffixes in component.FAMILIES for suffix in suffixes))


def frozen_inputs(data):
    """Detach caller arrays, retaining every value/dtype/shape without coercion."""
    result = dict(data)
    for key in INPUT_KEYS:
        if key not in data: continue
        value = np.asarray(data[key])
        component.require(value.dtype.kind in 'biuf' and np.isfinite(value).all(), 'Invalid native schedule input: '+key)
        result[key] = np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype).reshape(value.shape)
    return result


def input_manifest(data):
    return {key: component.array_identity(np.asarray(data[key])) for key in INPUT_KEYS if key in data}


def manifest_identity(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _host_identity(value):
    array = np.asarray(value)
    return (id(value), int(array.__array_interface__['data'][0]), array.shape, array.dtype.str,
        bool(array.flags.c_contiguous), bool(array.flags.writeable))


def _device_identity(value):
    return (id(value), int(value.data.ptr), value.shape, value.dtype.str, bool(value.flags.c_contiguous))


class StorageGuard:
    """O(number of arrays), never O(table rows), ordinary transaction checks."""
    def __init__(self, solver):
        self.solver = solver; self.inputs = input_manifest(solver.data)
        self.input_identity = manifest_identity(self.inputs)
        self.expected = self.capture()
        self.checks = 0

    def capture(self):
        solver = self.solver; schedule = solver.schedule
        host = {key: _host_identity(solver.data[key]) for key in INPUT_KEYS if key in solver.data}
        component_host = {key: _host_identity(value) for key, value in schedule.host.items()}
        component.require(all(not row[-1] for row in [*host.values(), *component_host.values()]),
            'Retained native schedule inputs must have immutable storage')
        return dict(owner=(id(solver.data), id(schedule), id(schedule.host), id(schedule._arrays), id(schedule.outputs)),
            native_inputs=host, component_host=component_host,
            tables={key: _device_identity(value) for key, value in schedule._arrays.items()},
            outputs={key: _device_identity(value) for key, value in schedule.outputs.items()},
            clock=_device_identity(schedule.clock),
            destinations={
                'boundary.forcing': _device_identity(solver.boundary.arrays['forcing']),
                'sediment.bc_hydraulic': _device_identity(solver.sediment.geometry['bc_hydraulic']),
                'sediment.feed': _device_identity(solver.sediment.geometry['feed'])})

    def check(self):
        try:
            observed = self.capture()
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError('Scheduled captured storage/input schema changed') from exc
        component.require(observed == self.expected, 'Scheduled captured storage/input identity changed')
        self.checks += 1
        return True

    def audit(self):
        self.check()
        component.require(input_manifest(self.solver.data) == self.inputs,
            'Retained native schedule values changed')
        self.solver.schedule.verify_tables()
        return dict(verified=True, native_input_identity=self.input_identity, input_manifest=self.inputs,
            device_table_bytes_verified=True, large_table_transfers_per_ordinary_step=0)

    def receipt(self):
        return dict(native_input_identity=self.input_identity, input_manifest=self.inputs,
            guarded_destinations=list(self.expected['destinations']), guard_checks=self.checks,
            tables_outputs_clock_storage_guarded=True, retained_host_schedule_inputs_immutable=True,
            external_input_arrays_detached=True, pointer_identity_is_process_local=True,
            device_contents_checked_only_by_explicit_audit=True,
            no_per_step_table_download_or_hash=True)


def verify_parent():
    previous.qualified.checked_hashes({str(Path(previous.__file__).resolve()): PARENT_SHA256,
        str(Path(component.__file__).resolve()): COMPONENT_SHA256,
        str(Path(component.__file__).with_suffix('.cu').resolve()): KERNEL_SHA256})


def prepare_sources(folder, manifest):
    verify_parent(); folder = Path(folder).resolve()
    target = folder/'scheduled_coupled_guarded.py'; receipt_path = folder/'scheduled_guard_sources.json'
    component.require(not target.exists() and not receipt_path.exists(), 'Preserve previous guarded source archive')
    parent = previous.prepare_sources(folder, manifest)
    shutil.copy2(__file__, target)
    receipt = dict(schema=1, wrapper=target.name, wrapper_sha256=previous.qualified.sha(target),
        parent_adapter_sha256=PARENT_SHA256, component_sha256=COMPONENT_SHA256, kernel_sha256=KERNEL_SHA256,
        parent_source_receipt_sha256=parent['sha256'], numerical_changes=False,
        original_source_files_modified=False, guard_scope='Captured schedule source/destination storage and immutable native input values')
    previous.qualified.write_json(receipt_path, receipt)
    return dict(path=str(receipt_path), sha256=previous.qualified.sha(receipt_path), parent=parent, **receipt)


def verify_sources(folder, manifest):
    verify_parent(); folder = Path(folder).resolve(); previous.verify_sources(folder, manifest)
    receipt = json.loads((folder/'scheduled_guard_sources.json').read_text())
    component.require(receipt['schema'] == 1 and receipt['wrapper'] == 'scheduled_coupled_guarded.py'
        and receipt['parent_adapter_sha256'] == PARENT_SHA256
        and receipt['component_sha256'] == COMPONENT_SHA256 and receipt['kernel_sha256'] == KERNEL_SHA256
        and receipt['wrapper_sha256'] == previous.qualified.sha(__file__)
        and previous.qualified.sha(folder/receipt['wrapper']) == receipt['wrapper_sha256']
        and previous.qualified.sha(folder/'scheduled_sources.json') == receipt['parent_source_receipt_sha256'],
        'Guarded executed source ancestry differs')
    return receipt


@contextmanager
def constructors(folder, manifest, bindings):
    folder = Path(folder).resolve(); source = verify_sources(folder, manifest)
    try:
        with previous.constructors(folder, manifest, bindings) as (ParentSolver, ParentGraph):
            class GuardedSolver(ParentSolver):
                def __init__(self, data, *, source_folder=folder):
                    detached = frozen_inputs(data)
                    super().__init__(detached, source_folder=source_folder)
                    # Parent installs its own immutable normalized table arrays;
                    # freeze the remaining full forcing/scalars without changing values.
                    self.data = frozen_inputs(self.data)
                    self._captured_schedule_guard = StorageGuard(self)

                def check_schedule_storage(self):
                    return self._captured_schedule_guard.check()

                def verify_schedule_identity(self):
                    return self._captured_schedule_guard.audit()

                def step(self, dt, current, *, commit=True):
                    self.check_schedule_storage()  # Before even uploading the clock.
                    return super().step(dt, current, commit=commit)

                def _scheduled_transport(self, dt, current, euler, rebalance):
                    self.check_schedule_storage()  # Includes capture/plain warmup.
                    return super()._scheduled_transport(dt, current, euler, rebalance)

                def euler(self, state, dt, current):
                    self.check_schedule_storage()
                    return super().euler(state, dt, current)

                def schedule_receipt(self):
                    return dict(super().schedule_receipt(), captured_storage_guard=self._captured_schedule_guard.receipt(),
                        guard_source_receipt_sha256=previous.qualified.sha(folder/'scheduled_guard_sources.json'))

            class GuardedGraph(ParentGraph):
                def __init__(self, solver, **kwargs):
                    component.require(isinstance(solver, GuardedSolver), 'Guarded graph requires guarded solver')
                    solver.check_schedule_storage()  # Before parent warmup/capture.
                    super().__init__(solver, **kwargs)
                    solver.check_schedule_storage()
                def replay(self, dt, current, euler, rebalance):
                    self.solver.check_schedule_storage()  # Before any graph launch.
                    return super().replay(dt, current, euler, rebalance)
                def receipt(self):
                    return dict(super().receipt(), destination_storage_guarded=True,
                        source_storage_guarded=True, retained_native_schedule_identity=self.solver._captured_schedule_guard.input_identity)

            bindings.append(dict(kind='scheduled_coupled_storage_guard', before_graph_capture=True,
                executed_wrapper=str(Path(__file__).resolve()), archived_wrapper=str(folder/source['wrapper']),
                wrapper_sha256=source['wrapper_sha256'], source_receipt_sha256=previous.qualified.sha(folder/'scheduled_guard_sources.json'),
                numerical_changes=False))
            yield GuardedSolver, GuardedGraph
    finally:
        verify_sources(folder, manifest)
