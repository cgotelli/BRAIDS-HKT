"""Whole-process hydraulic audit using an authenticated released runtime.

Clean measurements have no operator wrappers or profiler. Instrumented runs
retain every computation and write and measure non-overlapping parent phases.
No GPU synchronization is inserted inside the execution loop.
"""
import argparse
import cProfile
from collections import defaultdict
import functools
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--memory-candidate', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = args.runtime.resolve(); raw = args.source.read_bytes()
    project = json.loads(raw); backend = project['model']['request']['execution']['backend']
    clocks = {}; counts = defaultdict(lambda: dict(calls=0, array_bytes=0))
    preparation_profile = cProfile.Profile(); execution_profile = cProfile.Profile()
    if not args.clean: preparation_profile.enable()
    stamp = time.perf_counter()
    # Import tools first, then put the selected immutable runtime before checkout.
    from tools.profile_execution130 import Audit, compare_frames
    import numpy as np
    clocks['analysis_import_s'] = time.perf_counter()-stamp
    sys.path.insert(0, str(runtime)); sys.dont_write_bytecode = True
    stamp = time.perf_counter()
    from braids_core.river_launcher import verify
    manifest = verify()
    clocks['runtime_authentication_s'] = time.perf_counter()-stamp
    stamp = time.perf_counter()
    from braids_core.sediment.portable_river_runtime import install
    install(json.loads((runtime/'PRECOMPILED.json').read_text()), runtime)
    from braids_core import river_worker as worker
    from braids_core.sediment import adaptive_river_worker as core
    from braids_gpu import atomic_output
    from braids_core.sediment.river_field_policy import AutomaticRiver, FixedRiver, FieldIndicators
    from braids_core.sediment.river_leaf_controller import RiverLeafController
    from braids_core.sediment.river_leaf_io import PreparedLeafCache, MeshEpochWriter
    from braids_core.sediment.river_explicit import RiverExplicit
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.shoreline_solver import ShorelineSolver, NativeShoreline
    from braids_core.sediment.river_mixtures import RiverMixtureSolver
    from braids_core.sediment.river_exterior_bed import ExteriorBed
    from braids_core.sediment.river_exterior_inlet import CompatibleInlet
    from braids_core.sediment.river_exterior_section import CriticalSection
    from braids_core.sediment.river_exterior_normal import NormalSection
    from braids_core.sediment.river_runtime_lease import RuntimeFileLease
    from braids_core.sediment import river_mesh_records
    from braids_core.sediment.flow2dh_mobile_bed import MobileEngine
    from braids_gpu import general_live_transport as transport
    clocks['solver_import_s'] = time.perf_counter()-stamp
    if not Path(worker.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError('Selected runtime was not imported')
    identity = worker.source_identity()
    candidate = None
    if args.memory_candidate:
        from tools.output_candidate134 import install as install_candidate
        candidate = install_candidate()
    (out/'PROJECT.json').write_bytes(raw)
    audit = Audit(backend, 64)
    owners = []; steps = []
    if not args.clean:
        # Top-level wrappers supply disjoint phase totals. Nested call costs
        # remain inclusive/exclusive and must not be summed as phase totals.
        for obj, method, phase, label in (
            (worker, 'load_prepared', 'preflight', 'public.load_prepared'),
            (core, 'load_prepared', 'load', 'core.load_prepared'),
            (core, 'hierarchy', 'load', 'hierarchy'),
            (core, 'initial_leaves', 'construction', 'initial_leaves'),
            (PreparedLeafCache, 'get', 'construction', 'mesh.cache'),
            (PreparedLeafCache, 'load_payload', 'construction', 'mesh.payload'),
            (river_mesh_records, 'initialize', 'construction', 'native.geometry'),
            (transport, 'verify_sources', 'construction', 'source.verify'),
            (NativeShoreline, '__init__', 'construction', 'native.construct'),
            (AutomaticRiver, 'assess', 'amr', 'amr.assess'),
            (FieldIndicators, 'evaluate', 'amr', 'amr.indicators'),
            (RiverLeafController, 'adapt', 'amr', 'amr.adapt'),
            (core, 'fields', 'output', 'output.host_fields'),
            (MeshEpochWriter, 'write', 'output', 'output.frames'),
            (core, 'diagnostics', 'diagnostics', 'output.budgets'),
            (AutomaticRiver, 'checkpoint', 'checkpoint', 'checkpoint.policy'),
            (FixedRiver, 'checkpoint', 'checkpoint', 'checkpoint.fixed'),
            (RiverLeafController, 'checkpoint', 'checkpoint', 'checkpoint.leaves'),
            (RiverMixtureSolver, 'checkpoint', 'checkpoint', 'checkpoint.columns'),
            (ShorelineSolver, 'checkpoint', 'checkpoint', 'checkpoint.solver'),
            (worker, 'publish_checkpoint', 'checkpoint', 'checkpoint.public'),
            (RuntimeFileLease, 'close', 'finalization', 'runtime_lease.close'),
            (ShorelineSolver, '_trial_snapshot', 'step', 'transaction.snapshot'),
            (ShorelineSolver, 'restore', 'step', 'transaction.restore'),
            (RiverExplicit, 'advance', 'step', 'hydraulics.advance'),
            (RiverExplicit, '_single', 'step', 'hydraulics.ssprk22'),
            (RiverExplicit, 'rate', 'step', 'hydraulics.rate_and_cfl'),
            (RiverMidpoint, 'rate', 'step', 'hydraulics.spatial_rate'),
            (ExteriorBed, 'evaluate', 'step', 'boundary.evaluate'),
            (CompatibleInlet, 'evaluate', 'step', 'boundary.exact_Q'),
            (CriticalSection, 'solve', 'step', 'boundary.critical_reference'),
            (NormalSection, 'solve', 'step', 'boundary.normal_reference'),
        ):
            audit.wrap(obj, method, phase, label)
        for cls in (AutomaticRiver, RiverLeafController, RiverMixtureSolver, ShorelineSolver):
            if hasattr(cls, 'checkpoint_bytes'):
                audit.wrap(cls, 'checkpoint_bytes', 'checkpoint', 'bytes.'+cls.__name__)
        old_factory = core.factory
        def factory(*a, **kw):
            with audit.timing('factory.configure', 'construction'):
                original = old_factory(*a, **kw)
            def construct(leaves, mesh):
                with audit.timing('owner.construct', 'construction'):
                    owner = original(leaves, mesh)
                owners.append(dict(cells=len(leaves), faces=owner.physics.nf))
                for method in ('begin', 'finish', 'local', 'friction', 'commit', 'effective_inputs',
                               'directional_geometry', 'gradients', 'add_momentum'):
                    if hasattr(owner.physics, method):
                        audit.wrap(owner.physics, method, 'step', 'physics.'+method)
                if backend == 'cpu':
                    audit.wrap(owner.hydraulics.lib, 'shoreline_native_boundary_rates', 'step', 'native.spatial_rate')
                return owner
            construct.required_leaves = original.required_leaves
            return construct
        core.factory = factory
        old_step = ShorelineSolver.step
        def step(owner, dt, *a, **kw):
            with audit.timing('step', 'step'):
                result = old_step(owner, dt, *a, **kw)
            steps.append(dict(time_s=owner.time, outer_dt_s=dt, diagnostics=result['diagnostics']))
            return result
        ShorelineSolver.step = step
        old_launch = MobileEngine.launch
        def launch(engine, name, n, *a):
            with audit.timing('kernel.'+name, 'step'):
                return old_launch(engine, name, n, *a)
        MobileEngine.launch = launch
        # Replace existing imported aliases as well as the module owner. The
        # wrappers are read-only instrumentation; all original writes occur.
        for name in ('save_arrays_atomic', 'write_json_atomic', 'write_bytes_atomic'):
            if not hasattr(atomic_output, name): continue
            original = getattr(atomic_output, name)
            def measured(path, *a, _original=original, _name=name, **kw):
                category = ('progress' if str(path).endswith('PROGRESS.json') else
                            'checkpoint' if 'checkpoint' in str(path).lower() else 'other')
                row = counts[_name+'.'+category]; row['calls'] += 1
                if _name == 'save_arrays_atomic':
                    row['array_bytes'] += sum(np.asarray(v).nbytes for v in kw.values())
                elif _name == 'write_bytes_atomic':
                    row['array_bytes'] += len(a[0])
                with audit.timing('io.'+_name+'.'+category, 'progress' if category == 'progress' else 'io'):
                    return _original(path, *a, **kw)
            for module in list(sys.modules.values()):
                if module is not None and getattr(module, name, None) is original:
                    setattr(module, name, measured)
    stamp = time.perf_counter()
    prepared = worker.prepare(out/'PROJECT.json', out/'prepared')
    clocks['preparation_s'] = time.perf_counter()-stamp
    preparation_profile.disable()
    if not args.clean:
        preparation_profile.dump_stats(str(out/'PREPARATION.pstats'))
        with (out/'PREPARATION.txt').open('w') as stream:
            pstats.Stats(preparation_profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(100)
    preparation_calls = dict(audit.calls); preparation_phases = dict(audit.phases)
    audit.calls.clear(); audit.phases.clear(); counts.clear()
    stamp = time.perf_counter()
    if not args.clean: execution_profile.enable()
    result = worker.run(prepared, out/'run')
    execution_profile.disable()
    clocks['worker_s'] = time.perf_counter()-stamp
    clocks['through_run_s'] = time.perf_counter()-START
    audit.phases['unwrapped'] = clocks['worker_s']-sum(audit.phases.values())
    if not args.clean:
        execution_profile.dump_stats(str(out/'EXECUTION.pstats'))
        with (out/'EXECUTION.txt').open('w') as stream:
            pstats.Stats(execution_profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(120)
    matched = compare_frames(out/'run', args.reference_run.resolve()) if args.reference_run else None
    origins = {name: module.__file__ for name, module in sys.modules.items()
               if name.startswith(('braids_core.', 'braids_gpu.')) and getattr(module, '__file__', None)}
    if not all(Path(path).resolve().is_relative_to(runtime) for path in origins.values()):
        raise AssertionError('Checkout solver was imported')
    if verify() != manifest or worker.source_identity() != identity or args.source.read_bytes() != raw:
        raise AssertionError('Input/runtime changed')
    inventory = defaultdict(lambda: dict(files=0, bytes=0))
    for path in out.rglob('*'):
        if path.is_file():
            relative = path.relative_to(out)
            group = '/'.join(relative.parts[:2])
            inventory[group]['files'] += 1; inventory[group]['bytes'] += path.stat().st_size
    report = dict(clean=args.clean, backend=backend, project=str(args.source.resolve()),
        project_sha256=hashlib.sha256(raw).hexdigest(), runtime=str(runtime), runtime_identity=identity,
        runtime_unchanged=True, all_solver_modules_selected=True, clocks=clocks,
        controls=project['settings']['controls'], settings={k:project['settings'][k]
            for k in ('dt_max_s','end_time_s','output_interval_s','checkpoint_interval_s','adaptive') if k in project['settings']},
        phases=dict(audit.phases), calls=dict(audit.calls), preparation_calls=preparation_calls,
        preparation_phases=preparation_phases, write_counts=dict(counts), owners=owners, steps=steps,
        result=result, storage=dict(inventory), exact_reference=matched,
        candidate=None if candidate is None else dict(identity=candidate.identity, stats=candidate.stats),
        note='Instrumented phase times include profiler overhead. Inclusive children overlap; phase totals do not. '
             'No extra CUDA barrier; wall cost includes existing synchronization sites. '
             'Clean mode has no operator instrumentation. Integrity recheck/analysis follows through_run_s.')
    (out/'PROFILE.json').write_text(json.dumps(report, indent=2)+'\n')
    if candidate is not None: candidate.close()
    print(json.dumps(dict(clocks=clocks, phases=dict(audit.phases), result=result,
        exact=None if matched is None else dict(exact=matched['exact'],frames=len(matched['frames'])))), flush=True)


if __name__ == '__main__':
    main()
