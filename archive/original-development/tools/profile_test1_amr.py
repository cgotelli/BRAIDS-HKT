"""Profile an installed or checkout river solver on private Test1 inputs.

Explicit clock overrides are saved in the private project. Instrumentation is
process-local and does not substitute any solver operator.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True, help='Existing prepared project')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    ap.add_argument('--until', type=float, default=8.)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--checkout', action='store_true')
    ap.add_argument('--depth-only', action='store_true', help='Explicit new private project: retain clocks and use depth-only AMR schema')
    ap.add_argument('--dt-max',type=float)
    ap.add_argument('--amr-interval',type=float,help='Explicit interval override for a private comparison project')
    ap.add_argument('--audit-front',action='store_true',help='Qualification only: inspect wet cells before every assessment; not a clean timing')
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    runtime = (Path(__file__).resolve().parents[1] if args.checkout else
        Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime'))
    sys.path.insert(0, str(runtime)); sys.dont_write_bytecode = True
    if not args.checkout:
        from braids_core.river_launcher import verify
        verify()
        from braids_core.sediment.portable_river_runtime import install
        install(json.loads((runtime/'PRECOMPILED.json').read_text()), runtime)
    from braids_core import river_worker as worker
    from braids_core.river_model import resolve_model
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.river_field_policy import AutomaticRiver, FieldIndicators
    from braids_core.sediment.river_leaf_controller import RiverLeafController
    from braids_core.sediment.river_leaf_io import PreparedLeafCache, MeshEpochWriter
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.shoreline_solver import ShorelineSolver
    from braids_gpu.atomic_output import write_json_atomic

    source = args.source.resolve()
    raw = (source/'PROJECT.json').read_bytes()
    doc = json.loads(raw)
    doc['terrain']['dem_path'] = json.loads((source/'core/RECIPE.json').read_text())['dem_path']
    doc['model']['request']['execution']['backend'] = args.backend
    doc['model'] = resolve_model(doc['model']['request']).document()
    if args.depth_only:
        from braids_core.river_project import SCHEMA
        doc['schema']=SCHEMA
        doc['settings']['adaptation']={k:doc['settings']['adaptation'][k] for k in ('interval','coarsen_delay','buffer_cells')}
    if args.dt_max is not None:doc['settings']['dt_max_s']=args.dt_max
    if args.amr_interval is not None:doc['settings']['adaptation']['interval']=args.amr_interval
    write_json_atomic(out/'PROJECT.json', doc)
    began = time.perf_counter()
    prepared = worker.prepare(out/'PROJECT.json', out/'prepared')
    preparation_s = time.perf_counter()-began

    stack = []; phases = defaultdict(float); calls = defaultdict(lambda: [0, 0.])
    singles = []; steps = []; owners = []; stop = out/'STOP'; owner_levels={}
    run_start = [0.]
    @contextmanager
    def timing(name, phase):
        start = time.perf_counter(); outer = not stack
        stack.append(name)
        try: yield
        finally:
            elapsed = time.perf_counter()-start; stack.pop()
            calls[name][0] += 1; calls[name][1] += elapsed
            if outer: phases[phase] += elapsed

    def wrap(obj, name, phase, label=None):
        original = getattr(obj, name)
        @functools.wraps(original)
        def measured(*a, **kw):
            with timing(label or name, phase): return original(*a, **kw)
        setattr(obj, name, measured)

    wrap(core, 'load_prepared', 'load')
    wrap(AutomaticRiver, '__init__', 'initialization', 'policy_init')
    wrap(AutomaticRiver, 'assess', 'amr')
    wrap(FieldIndicators, 'evaluate', 'amr', 'indicators')
    wrap(RiverLeafController, '__init__', 'initialization', 'controller_init')
    wrap(RiverLeafController, 'plan', 'amr')
    wrap(PreparedLeafCache, 'get', 'initialization', 'mesh_cache')
    wrap(MeshEpochWriter, 'write', 'output', 'frames')
    wrap(core, 'fields', 'output')
    wrap(core, 'diagnostics', 'output')
    wrap(AutomaticRiver, 'checkpoint', 'checkpoint')
    wrap(worker, 'publish_checkpoint', 'checkpoint', 'public_checkpoint')
    factory = core.factory
    def measured_factory(*a, **kw):
        make = factory(*a, **kw)
        def measured_make(leaves, mesh):
            with timing('make_owner', 'initialization'):
                owner = make(leaves, mesh)
            owners.append(dict(cells=len(leaves), time_s=owner.time))
            if args.audit_front:owner_levels[id(owner.solver)]=a[2].node_level[list(leaves)].copy()
            return owner
        return measured_make
    core.factory = measured_factory
    original_step = ShorelineSolver.step
    def measured_step(self, dt, *a, **kw):
        with timing('step', 'stepping'):
            result = original_step(self, dt, *a, **kw)
        d = result['diagnostics']
        keys = ('hydraulic_substeps','hydraulic_minimum_dt','nonlinear_iterations',
                'fwave_evaluations','linear_matvecs','hydraulic_preempted_retries')
        steps.append(dict(time_s=self.time,dt=dt,**{k:d[k] for k in keys if k in d}))
        if args.audit_front:
            from braids_core.sediment.bed_columns_v101 import host
            levels=owner_levels[id(self)]
            steps[-1]['coarse_wet_cells']=int(((host(self.q[0])>self.physics.parameters.dry)&(levels<doc['settings']['levels']-1)).sum())
        if time.perf_counter()-run_start[0]>300: stop.touch()
        return result
    ShorelineSolver.step = measured_step
    single = RiverMidpoint._single
    def measured_single(self, base, dt, at):
        started = time.perf_counter(); ev = self.evaluations; error = None
        try: return single(self, base, dt, at)
        except Exception as exc: error = str(exc); raise
        finally: singles.append(dict(time=at,dt=dt,wall_s=time.perf_counter()-started,
                                     evaluations=self.evaluations-ev,error=error))
    RiverMidpoint._single = measured_single
    wrap(RiverMidpoint, 'rate', 'stepping')
    profile = cProfile.Profile(); result = None
    run_start[0] = time.perf_counter()
    if args.profile: profile.enable()
    try:
        result = worker.run(prepared,out/'run',until=args.until,stop_file=stop,
            on_frame=lambda _,row:print(json.dumps(row),flush=True))
    finally:
        profile.disable(); elapsed = time.perf_counter()-run_start[0]
        phases['other'] = elapsed-sum(phases.values())
        write_json_atomic(out/'PROFILE_SUMMARY.json',dict(
            backend=args.backend,cprofile=args.profile,front_audit=args.audit_front,source_prepared=str(source),
            source_project_sha256=hashlib.sha256(raw).hexdigest(),runtime=str(runtime),
            runtime_identity=worker.source_identity(),preparation_s=preparation_s,
            worker_call_s=elapsed,result=result,phases=dict(phases),
            inclusive_calls=dict(calls),steps=steps,singles=singles,owners=owners,
            note='Phases are disjoint. Inclusive calls overlap; do not add them. GPU phases use existing synchronization; no per-kernel barrier inserted.'))
        if args.profile:
            profile.dump_stats(str(out/'PROFILE.pstats'))
            with (out/'PROFILE.txt').open('w') as f:
                pstats.Stats(profile,stream=f).strip_dirs().sort_stats('cumulative').print_stats(100)
        assert (source/'PROJECT.json').read_bytes() == raw
        print(json.dumps(dict(result=result,phases=dict(phases))),flush=True)


if __name__ == '__main__': main()
