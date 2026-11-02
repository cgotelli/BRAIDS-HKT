"""Private, explicit schedule scope extension of the qualified coupled GPU.

Original files/guards are preserved. Exact-anchor generated Python copies only
admit hydraulic/per-fraction tables; no counts are cleared and no constant-input
surrogate is passed. Euler/SSPRK43 numerical bodies and all CUDA remain unchanged.
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
from . import boundary_schedule as schedule
from . import curved_wetdry_worker as qualified

HERE = Path(__file__).resolve().parent
NAMES = {
    'boundaries.py': '_m2_boundaries', 'sediment_transport.py': '_m2_sediment_transport',
    'graph_execution.py': '_m2_graph_execution', 'coupled_solver.py': '_m2_coupled_solver',
    'coupled_graph.py': '_m2_coupled_graph',
}
EDITS = {
    'boundaries.py': [("for key in ('nhydro_series','section_control','sampled'):",
        "for key in ('section_control','sampled'):")],
    'sediment_transport.py': [("for key in ('nseries','nhydro_series','nfraction_series'):",
        "for key in ('nseries',):")],
    'coupled_solver.py': [
        ("for key in ('nmorfac','nhydro_series','nseries','nfraction_series'):", "for key in ('nmorfac','nseries'):"),
        ('from .boundaries import BoundaryControls', 'from ._m2_boundaries import BoundaryControls'),
        ('from .sediment_transport import ResidentSedimentTransport', 'from ._m2_sediment_transport import ResidentSedimentTransport')],
    'graph_execution.py': [
        ('for key in ("nhydro_series","nseries","nfraction_series","nmorfac","section_control","sampled"):',
         'for key in ("nseries","nmorfac","section_control","sampled"):'),
        ('if outlet_extension!=0:', 'if outlet_extension!=1:'),
        ('CUDA graph SSP43 requires outlet_extension=0', 'Scheduled coupled graph requires qualified outlet_extension=1')],
    'coupled_graph.py': [('from .graph_execution import validate_graph_scope',
        'from ._m2_graph_execution import validate_graph_scope')],
}


def transform(name, source):
    if name not in EDITS: raise ValueError('Unknown private source '+name)
    for before, after in EDITS[name]:
        schedule.require(source.count(before) == 1 and after not in source, 'Absent/ambiguous/already-patched anchor: '+name)
        source = source.replace(before, after, 1)
    return source


def validate_scope(data):
    version = np.asarray(data.get('deck_version', -1))
    schedule.require(version.shape == () and version.dtype.kind in 'iu' and int(version) >= 13,
        'Explicit native deck_version >=13 required for scheduled coupled scope')
    arrays = schedule.validate_tables(data)
    schedule.require(np.isin(arrays['mode'], [2, 3, 6, 8]).all(), 'Only existing GPU hydraulic modes are supported')
    schedule.require(not np.any(arrays['nseries']), 'Legacy scalar sediment series are component-only, not coupled scope')
    schedule.require(not np.any(data.get('nmorfac', 0)), 'Time-dependent MorFac remains unsupported')
    for key in ('fraction_concentration', 'series_concentration', 'fraction_series_concentration'):
        schedule.require(not np.any(arrays[key]), 'Nonzero suspended concentration remains unsupported: '+key)
    forcing = np.asarray(data['forcing'])
    if forcing.shape[0] == 6:
        schedule.require(not np.any(forcing[2]), 'Legacy suspended concentration remains unsupported')
    bed_mode = np.asarray(data.get('bed_boundary_mode', np.zeros(len(arrays['mode']), np.int32)))
    schedule.require(bed_mode.shape == arrays['mode'].shape, 'Bed boundary-mode dimensions differ')
    for b in np.flatnonzero(bed_mode == 1):
        schedule.require(not np.any(arrays['fraction_bed_input'][:, b] > 0) and
            not np.any(arrays['fraction_series_bed_input'][:, :, b] > 0),
            'Equilibrium bed-load inlet conflicts with positive prescribed feed')
    return arrays


def prepare_sources(folder, baseline_manifest):
    """Write only new generated modules/receipt inside a fresh private archive."""
    folder = Path(folder).resolve()
    qualified.checked_hashes({str(folder/name): value for name, value in baseline_manifest.items()})
    receipt_path = folder/'scheduled_sources.json'
    targets = [folder/(private+'.py') for private in NAMES.values()]
    extras = [folder/'scheduled_coupled_adapter.py', folder/'boundary_schedule.py', folder/'boundary_schedule.cu']
    schedule.require(not any(path.exists() for path in [receipt_path, *targets, *extras]), 'Preserve existing scheduled source archive')
    rows = {}
    for name, private in NAMES.items():
        schedule.require(name in baseline_manifest, 'Missing authenticated base source '+name)
        original = folder/name; source = original.read_text(); changed = transform(name, source)
        target = folder/(private+'.py'); target.write_text(changed, encoding='utf-8')
        rows[name] = dict(original_sha256=baseline_manifest[name], generated_name=target.name,
            generated_sha256=qualified.sha(target), replacements=len(EDITS[name]), exact_transform=True)
    for source, destination in zip((Path(__file__), Path(schedule.__file__), Path(schedule.__file__).with_suffix('.cu')), extras):
        shutil.copy2(source, destination)
    receipt = dict(schema=1, rows=rows, helper_files={p.name: qualified.sha(p) for p in extras},
        original_counts_preserved=True, constant_surrogate_metadata=False,
        original_numeric_bodies_unchanged=True, schedule_source_copies_per_stage=3,
        added_schedule_status_merge=True, qualified_outlet_extension=1,
        standalone_stage_functions_only_previously_qualified=True, integrated_qualification_pending=True)
    qualified.write_json(receipt_path, receipt)
    return dict(path=str(receipt_path), sha256=qualified.sha(receipt_path), **receipt)


def verify_sources(folder, baseline_manifest):
    folder = Path(folder).resolve(); path = folder/'scheduled_sources.json'
    receipt = json.loads(path.read_text())
    schedule.require(receipt['schema'] == 1 and set(receipt['rows']) == set(NAMES), 'Private source selection differs')
    qualified.checked_hashes({str(folder/name): value for name, value in baseline_manifest.items()})
    for name, private in NAMES.items():
        row = receipt['rows'][name]; target = folder/(private+'.py')
        schedule.require(row['original_sha256'] == baseline_manifest[name] and row['generated_name'] == target.name
            and qualified.sha(target) == row['generated_sha256']
            and target.read_text() == transform(name, (folder/name).read_text()), 'Private generated source differs: '+name)
    qualified.checked_hashes({str(folder/name): value for name, value in receipt['helper_files'].items()})
    schedule.require(qualified.sha(__file__) == receipt['helper_files']['scheduled_coupled_adapter.py']
        and qualified.sha(schedule.__file__) == receipt['helper_files']['boundary_schedule.py'],
        'Executed adapter/evaluator differs from archived source')
    return receipt


def _make_solver(parent, folder, source_receipt):
    class ScheduledCoupledSolver(parent):
        def __init__(self, data, *, source_folder=folder):
            schedule.require(Path(source_folder).resolve() == folder, 'Private solver source folder differs')
            tables = validate_scope(data)
            original_counts = {key: np.asarray(data[key]).copy() for key in ('nseries', 'nhydro_series', 'nfraction_series')}
            super().__init__(data, source_folder=folder)
            schedule.require(all(np.array_equal(self.data[key], value) for key, value in original_counts.items()),
                'Native schedule counts changed during construction')
            self.schedule = schedule.ResidentBoundarySchedule(data, source_path=folder/'boundary_schedule.cu')
            # Counts/tables are immutable, but their original values are retained.
            for key in tables:
                if key != 'forcing': self.data[key] = tables[key]
            self._plain_transport = self.stepper.transport
            self.stepper.transport = self._scheduled_transport
            self._schedule_stage = None; self._schedule_trial_open = False
            self._schedule_clock = None; self._schedule_plain_stages = []
            self._schedule_source_receipt = source_receipt

        def _scheduled_transport(self, dt, current, euler, rebalance):
            self._schedule_stage = 0; self._schedule_plain_stages = []
            try:
                result = self._plain_transport(dt, current, euler, rebalance)
                schedule.require(self._schedule_stage == 4, 'Expected all four scheduled Euler records')
                return result
            finally:
                self._schedule_stage = None

        def euler(self, state, dt, current):
            stage = self._schedule_stage
            schedule.require(type(stage) is int and 0 <= stage < 4, 'Scheduled Euler must be called by four-stage transport')
            values = self.schedule.outputs
            # Stable destination buffers; these device copies are captured with
            # four distinct immutable source pointers, never Python clock values.
            self.cp.copyto(self.boundary.arrays['forcing'], values['hydraulic'][stage])
            self.cp.copyto(self.sediment.geometry['bc_hydraulic'], values['hydraulic'][stage])
            self.cp.copyto(self.sediment.geometry['feed'], values['feed'][stage])
            result = super().euler(state, dt, current)
            self._merge(values['status'][stage])
            self._schedule_plain_stages.append(float(current)); self._schedule_stage += 1
            return result

        def step(self, dt, current, *, commit=True):
            schedule.require(not self._schedule_trial_open, 'Concurrent/reentrant schedule trials are unsupported')
            self.schedule.evaluate(current, dt)
            self._schedule_clock = (float(current), float(dt)); self._schedule_trial_open = True
            try:
                return super().step(dt, current, commit=commit)
            finally:
                self._schedule_trial_open = False

        def schedule_receipt(self):
            return dict(component=self.schedule.receipt(), source_generation=self._schedule_source_receipt,
                counts={key: self.data[key].tolist() for key in ('nseries', 'nhydro_series', 'nfraction_series')},
                original_counts_preserved=True, constant_surrogate_metadata=False,
                stage_record_indices=[0, 1, 2, 3], stage_fractions=list(schedule.STAGES),
                copies_per_stage=['hydraulic_to_boundary', 'same_hydraulic_to_sediment', 'feed_to_sediment'],
                status_merged_per_stage=True, live_clock_before_every_trial=True,
                last_host_transport_stage_times=self._schedule_plain_stages,
                last_host_transport_is_not_a_replay_trace=True, suspended_transport_supported=False,
                secondary_profile_substituted=False, backend='gpu', automatic_CPU_fallback=False)
    return ScheduledCoupledSolver


@contextmanager
def constructors(folder, baseline_manifest, bindings):
    """Yield explicit Solver/Graph constructors, restoring all aliases afterward."""
    from . import coupled_solver as original_solver
    folder = Path(folder).resolve(); receipt = verify_sources(folder, baseline_manifest)
    _, qualified_manifest, live_dependencies = qualified.source_identity()
    schedule.require(dict(baseline_manifest) == qualified_manifest, 'Full qualified baseline source manifest required')
    prefix = 'braids_support.gpu_fwave.'; loaded = []
    schedule.require(not any(prefix+private in sys.modules for private in NAMES.values()), 'Private schedule modules already loaded')
    try:
        with qualified.constructors(folder, baseline_manifest, bindings):
            modules = {}
            for name in ('boundaries.py', 'sediment_transport.py', 'graph_execution.py', 'coupled_solver.py', 'coupled_graph.py'):
                private = NAMES[name]; key = prefix+private; path = folder/(private+'.py')
                spec = importlib.util.spec_from_file_location(key, path); module = importlib.util.module_from_spec(spec)
                sys.modules[key] = module; loaded.append(key); spec.loader.exec_module(module); modules[name] = module
            # Preserve the already-qualified corrected-secondary constructor;
            # its curvature dependency was bound by qualified.constructors.
            modules['coupled_solver.py'].SecondaryTransport = original_solver.SecondaryTransport
            Solver = _make_solver(modules['coupled_solver.py'].CoupledSolver, folder, receipt)
            ParentGraph = modules['coupled_graph.py'].CoupledTransportGraph
            class ScheduledTransportGraph(ParentGraph):
                def __init__(self, solver, **kwargs):
                    schedule.require(isinstance(solver, Solver), 'Scheduled graph requires matching private solver')
                    validate_scope(solver.data)
                    super().__init__(solver, **kwargs)
                def replay(self, dt, current, euler, rebalance):
                    schedule.require(self.solver._schedule_trial_open and
                        self.solver._schedule_clock == (float(current), float(dt)), 'Replay needs the live enclosing scheduled trial')
                    return super().replay(dt, current, euler, rebalance)
                def receipt(self):
                    return dict(super().receipt(), boundary_schedule=True,
                        stage_record_indices=[0, 1, 2, 3], stage_copies=12,
                        schedule_status_merges=4, current_clock_upload_outside_transport_graph=True,
                        schedule_sources_sha256=qualified.sha(folder/'scheduled_sources.json'))
            bindings.append(dict(kind='scheduled_coupled_private', before_graph_capture=True,
                executed_modules={name: str(folder/(private+'.py')) for name, private in NAMES.items()},
                generated_sha256={name: row['generated_sha256'] for name, row in receipt['rows'].items()},
                source_receipt_sha256=qualified.sha(folder/'scheduled_sources.json'), counts_preserved=True))
            yield Solver, ScheduledTransportGraph
    finally:
        for key in reversed(loaded): sys.modules.pop(key, None)
        verify_sources(folder, baseline_manifest)
        qualified.checked_hashes(live_dependencies)
