"""Read-only-operator execution audit for the qualified public river worker.

Timing wrappers are process-local and call every original operator exactly once.
GPU event spans are sampled without introducing per-call synchronization. Python
wall times therefore include the existing synchronization sites, not necessarily
the device work launched by the labelled function. This is NOT a clean benchmark.
"""
import argparse
import cProfile
from collections import defaultdict
from contextlib import contextmanager
import functools
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Audit:
    def __init__(self, backend, sample):
        self.backend = backend
        self.sample = sample
        self.stack = []
        self.phases = defaultdict(float)
        self.calls = defaultdict(lambda: dict(count=0, inclusive_s=0., exclusive_s=0.))
        self.events = defaultdict(list)
        self.cp = None
        if backend == 'gpu':
            import cupy as cp
            self.cp = cp

    @contextmanager
    def timing(self, label, phase, device=False):
        row = self.calls[label]
        row['count'] += 1
        event = None
        if device and self.cp is not None and (row['count'] - 1) % self.sample == 0:
            event = (self.cp.cuda.Event(), self.cp.cuda.Event())
            event[0].record()
        item = [label, time.perf_counter(), 0.]
        self.stack.append(item)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - item[1]
            self.stack.pop()
            row['inclusive_s'] += elapsed
            row['exclusive_s'] += elapsed - item[2]
            if self.stack:
                self.stack[-1][2] += elapsed
            else:
                self.phases[phase] += elapsed
            if event is not None:
                event[1].record()
                self.events[label].append(event)

    def wrap(self, obj, name, phase, label=None, device=False):
        original = getattr(obj, name)
        # Native DLL Python wrappers can survive owner reconstruction.
        original = getattr(original, '_profile130_original', original)
        @functools.wraps(original)
        def measured(*a, **kw):
            with self.timing(label or name, phase, device):
                return original(*a, **kw)
        measured._profile130_original = original
        setattr(obj, name, measured)

    def collect_events(self):
        if self.cp is None:
            return {}
        self.cp.cuda.get_current_stream().synchronize()
        result = {}
        for label, pairs in self.events.items():
            elapsed = np.array([self.cp.cuda.get_elapsed_time(*p) * .001 for p in pairs])
            result[label] = dict(samples=len(pairs), mean_s=float(elapsed.mean()),
                maximum_s=float(elapsed.max()), measured_sum_s=float(elapsed.sum()),
                estimated_all_calls_s=float(elapsed.mean() * self.calls[label]['count']))
        return result


def compare_frames(run, reference):
    """Compare every common saved numerical field/mesh; metadata is not a field."""
    left = json.loads((run / 'frames/INDEX.json').read_text())
    right = json.loads((reference / 'frames/INDEX.json').read_text())
    rrows = {round(v['time_s'], 10): v for v in right['frames']}
    rows = []
    for row in left['frames']:
        ref = rrows[round(row['time_s'], 10)]
        diffs = []
        for key in ('file', 'epoch'):
            with np.load(run / 'frames' / row[key], allow_pickle=False) as a, np.load(reference / 'frames' / ref[key], allow_pickle=False) as b:
                for name in sorted(set(a.files) | set(b.files)):
                    if name == 'mesh_epoch':
                        continue
                    if name not in a or name not in b:
                        diffs.append(dict(field=name, reason='missing'))
                        continue
                    x, y = a[name], b[name]
                    if x.dtype.kind not in 'biufc' and y.dtype.kind not in 'biufc':
                        continue
                    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                        maximum = float(np.max(np.abs(x - y))) if x.shape == y.shape and x.dtype.kind in 'biufc' else None
                        diffs.append(dict(field=name, maximum=maximum, shape=[list(x.shape), list(y.shape)]))
        rows.append(dict(time_s=row['time_s'], exact=not diffs, differences=diffs))
    return dict(exact=all(v['exact'] for v in rows), frames=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--until', type=float)
    ap.add_argument('--sample', type=int, default=32)
    ap.add_argument('--reference-run', type=Path)
    ap.add_argument('--clean', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = args.source.read_bytes()
    project = json.loads(raw)
    backend = project['model']['request']['execution']['backend']
    from braids_core import river_worker as worker
    from braids_gpu.atomic_output import write_json_atomic
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.river_field_policy import AutomaticRiver, FixedRiver, FieldIndicators
    from braids_core.sediment.river_leaf_controller import RiverLeafController
    from braids_core.sediment.river_leaf_io import PreparedLeafCache, MeshEpochWriter
    from braids_core.sediment.river_explicit import RiverExplicit
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.shoreline_solver import ShorelineSolver, NativeShoreline
    from braids_core.sediment.river_exterior_bed import ExteriorBed
    from braids_core.sediment.river_exterior_inlet import CompatibleInlet
    from braids_core.sediment.river_exterior_section import CriticalSection
    from braids_core.sediment.river_exterior_normal import NormalSection
    from braids_core.sediment.river_section_cache import SectionCache
    from braids_core.sediment.flow2dh_mobile_bed import MobileEngine
    from braids_core.sediment.bed_columns_v101 import host
    from braids_gpu import general_live_transport as transport
    from braids_core.sediment import river_amr_transfer
    from braids_core.sediment import bend_native_hydraulics
    identity = worker.source_identity()
    write_json_atomic(out / 'PROJECT.json', project)
    t = time.perf_counter()
    prepared = worker.prepare(out / 'PROJECT.json', out / 'prepared')
    preparation = time.perf_counter() - t
    audit = Audit(backend, args.sample)
    owners, steps, restrictions = [], [], []
    if not args.clean:
        for obj, method, phase, label in (
            (core, 'load_prepared', 'load', 'load_prepared'),
            (AutomaticRiver, 'assess', 'amr', 'amr.assess'),
            (FieldIndicators, 'evaluate', 'amr', 'amr.indicators'),
            (RiverLeafController, 'plan', 'amr', 'amr.plan'),
            (RiverLeafController, 'adapt', 'amr', 'amr.adapt'),
            (PreparedLeafCache, 'get', 'construction', 'mesh.cache'),
            (MeshEpochWriter, 'write', 'output', 'output.frames'),
            (core, 'fields', 'output', 'output.fields'),
            (core, 'diagnostics', 'output', 'output.diagnostics'),
            (AutomaticRiver, 'checkpoint', 'checkpoint', 'checkpoint.adaptive'),
            (FixedRiver, 'checkpoint', 'checkpoint', 'checkpoint.fixed'),
            (worker, 'publish_checkpoint', 'checkpoint', 'checkpoint.public'),
            (transport, 'verify_sources', 'construction', 'source.verify'),
            (NativeShoreline, '__init__', 'construction', 'native.construct'),
            (ShorelineSolver, '_trial_snapshot', 'step', 'transaction.snapshot'),
            (ShorelineSolver, 'restore', 'step', 'transaction.restore'),
            (RiverExplicit, 'advance', 'step', 'hydraulics.advance'),
            (RiverExplicit, '_single', 'step', 'hydraulics.ssprk22'),
            (RiverMidpoint, 'rate', 'step', 'hydraulics.spatial_rate'),
            (ExteriorBed, 'evaluate', 'step', 'boundary.evaluate'),
            (ExteriorBed, 'apply_gpu_traces', 'step', 'boundary.traces'),
            (ExteriorBed, 'apply_gpu_flux', 'step', 'boundary.apply_flux'),
            (CompatibleInlet, 'evaluate', 'step', 'boundary.exact_Q'),
            (CriticalSection, 'solve', 'step', 'boundary.critical_reference'),
            (NormalSection, 'solve', 'step', 'boundary.normal_reference'),
            (SectionCache, 'restore', 'step', 'boundary.reference_cache'),
        ):
            audit.wrap(obj, method, phase, label, device=phase == 'step')
        oldlaunch = MobileEngine.launch
        def launch(self, name, n, *a):
            with audit.timing('kernel.' + name, 'step', device=True):
                return oldlaunch(self, name, n, *a)
        MobileEngine.launch = launch
        oldfactory = core.factory
        def factory(*a, **kw):
            make = oldfactory(*a, **kw)
            def construct(leaves, mesh):
                with audit.timing('owner.construct', 'construction'):
                    owner = make(leaves, mesh)
                owners.append(dict(cells=len(leaves), native_faces=owner.physics.nf))
                for method in ('begin', 'finish', 'friction', 'commit', 'local', 'effective_inputs'):
                    audit.wrap(owner.physics, method, 'step', 'physics.' + method, device=True)
                if backend == 'cpu':
                    audit.wrap(owner.hydraulics.lib, 'shoreline_native_boundary_rates', 'step', 'native.spatial_rate')
                else:
                    for obj, method, label in (
                        (owner.hydraulics.reconstruction, 'reconstruct', 'cuda.reconstruction'),
                        (owner.hydraulics.boundary, 'apply', 'cuda.boundary_legacy'),
                        (owner.hydraulics.boundary, 'apply_pressure', 'cuda.boundary_pressure'),
                        (owner.hydraulics.secondary, 'reconstruct', 'cuda.secondary_reconstruct'),
                        (owner.hydraulics.secondary, 'add_stress', 'cuda.secondary_stress'),
                    ):
                        audit.wrap(obj, method, 'step', label, device=True)
                return owner
            construct.required_leaves = make.required_leaves
            return construct
        core.factory = factory
        oldstep = ShorelineSolver.step
        def step(self, dt, *a, **kw):
            with audit.timing('step', 'step'):
                result = oldstep(self, dt, *a, **kw)
            steps.append(dict(time_s=self.time, dt=dt, diagnostics=result['diagnostics']))
            return result
        ShorelineSolver.step = step
        oldrate = RiverExplicit.rate
        rate_count = [0]
        def rate(self, q, dt, at):
            with audit.timing('hydraulics.rate_and_cfl', 'step', device=True):
                result = oldrate(self, q, dt, at)
            rate_count[0] += 1
            if rate_count[0] % args.sample == 0:
                with audit.timing('diagnostic.restriction_sample', 'diagnostics'):
                    f = host(self.frequency)
                    c = int(np.argmax(f))
                    a = self.solver
                    d = a.data
                    state = host(q[:, c])
                    linked = np.flatnonzero((d['owner'] == c) | (d['neighbour'] == c))
                    boundary = linked[d['neighbour'][linked] < 0]
                    restrictions.append(dict(time_s=at, proposed_dt=dt, frequency_s_inverse=float(f[c]),
                        cell=c, area_m2=float(d['area'][c]), xy=host(d['xy'][:, c]).tolist(),
                        state=state.tolist(), velocity_m_s=(state[1:] / state[0]).tolist() if state[0] else [0., 0.],
                        boundary_ids=d['bid'][boundary].tolist(), wet_cells=int(np.count_nonzero(host(q[0]) > a.physics.parameters.dry)),
                        cells=a.physics.n))
            return result
        RiverExplicit.rate = rate
    profiler = cProfile.Profile()
    result = None
    t = time.perf_counter()
    if not args.clean:
        profiler.enable()
    try:
        result = worker.run(prepared, out / 'run', until=args.until,
            on_frame=lambda _, row: print(json.dumps(row), flush=True))
    finally:
        profiler.disable()
        elapsed = time.perf_counter() - t
        audit.phases['other'] = elapsed - sum(audit.phases.values())
        events = audit.collect_events()
        equality = compare_frames(out / 'run', args.reference_run.resolve()) if result and args.reference_run else None
        write_json_atomic(out / 'PROFILE.json', dict(backend=backend, clean=args.clean,
            runtime=identity, runtime_after=worker.source_identity(),
            source=str(args.source.resolve()), source_sha256=hashlib.sha256(raw).hexdigest(),
            profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            controls=project['settings']['controls'], preparation_s=preparation, worker_wall_s=elapsed,
            result=result, phases=dict(audit.phases), calls=dict(audit.calls), cuda_event_spans=events,
            cuda_sampling_stride=args.sample, owners=owners, steps=steps, restrictions=restrictions,
            exact_reference=equality,
            note='Instrumented phase timings include Python/cProfile/event overhead. Inclusive calls overlap. CUDA spans contain asynchronous wait/dispatch gaps and cannot be added; no new barrier inside execution. Clean mode leaves all operators untouched.'))
        if result is not None:
            write_json_atomic(out / 'QUALIFICATION.json', dict(status=result, wall_s=elapsed,
                preparation_s=preparation, runtime=identity, steps=steps, history=[],
                scope='Output-array analysis adapter for execution profiling; no extra boundary evaluations or per-step sampling in clean mode'))
        if not args.clean:
            profiler.dump_stats(str(out / 'PROFILE.pstats'))
            with (out / 'PROFILE.txt').open('w') as f:
                pstats.Stats(profiler, stream=f).strip_dirs().sort_stats('cumulative').print_stats(100)
        assert args.source.read_bytes() == raw
        assert identity == worker.source_identity()
        print(json.dumps(dict(result=result, phases=dict(audit.phases), exact_reference=equality)), flush=True)


if __name__ == '__main__':
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--optimizations', choices=('none', 'mesh', 'toolchain', 'both'), default='none')
    options.add_argument('--numerics', choices=('baseline', 'selected'), default='baseline')
    selected, remaining = options.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if selected.optimizations == 'none' and selected.numerics == 'baseline':
        main()
    else:
        # DEM preparation lazily loads GDAL, whose Windows loader prepends its
        # own DLL directory to PATH. Resolve the test's toolchain only after
        # that prerequisite has initialized (production leases start at run).
        from osgeo import gdal
        from contextlib import ExitStack
        from tools.execution_optimizations130 import install
        from braids_gpu.atomic_output import write_json_atomic
        target = Path(remaining[remaining.index('--output') + 1])
        with ExitStack() as stack:
            numerical, clock = None, None
            if selected.numerics == 'selected':
                from tools.research_wet_cfl130 import candidate as cfl_candidate
                from validation.hydraulics.speed130_clock import candidate as clock_candidate
                numerical = stack.enter_context(cfl_candidate(1., 1e-6))
                clock = stack.enter_context(clock_candidate('combined'))
            pinned = stack.enter_context(install(mesh=selected.optimizations in ('mesh', 'both'),
                     toolchain=selected.optimizations in ('toolchain', 'both')))
            try:
                main()
            finally:
                if target.exists():
                    write_json_atomic(target / 'OPTIMIZATIONS.json', dict(
                        variant=selected.optimizations, toolchain=pinned.evidence(),
                        helper_sha256=hashlib.sha256((ROOT / 'tools/execution_optimizations130.py').read_bytes()).hexdigest(),
                        scope='Process-local equivalent implementation prototype; maintained solver and installed plugin unchanged'))
                    if numerical is not None:
                        write_json_atomic(target/'NUMERICAL_EXPERIMENT.json', dict(
                            research_only=True, actual_cfl_cap=1., hydraulic_dry_m=1e-6, clock=clock,
                            actual_owner_settings=[dict(cfl=o.midpoint.cfl, native_dry=o.data['dry'],
                                physical_dry=o.physics.parameters.dry, identity=o.midpoint.identity) for o in numerical['owners']],
                            script_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                                'tools/research_wet_cfl130.py', 'validation/hydraulics/speed130_clock.py')},
                            public_project_note='The project declares public CFL0.5; this explicit experiment sidecar and owner identities declare actual cap1.0 and combined clock; outer interval is explicitly saved in PROJECT'))
