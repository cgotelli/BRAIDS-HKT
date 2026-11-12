"""Matched research runs. All production spatial sources remain unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--integration', choices=('implicit', 'explicit'), required=True)
    ap.add_argument('--cfl', type=float, default=.45)
    ap.add_argument('--tolerance-factor', type=float, default=1.)
    extra, remaining = ap.parse_known_args()
    if '--checkout' not in remaining:
        remaining.append('--checkout')
    out = Path(remaining[remaining.index('--output')+1]).resolve()
    if out.exists():
        raise FileExistsError(out)
    from braids_core import river_worker as worker
    from braids_core.sediment import adaptive_river_worker as core
    from braids_core.sediment.bed_columns_v101 import host
    from braids_gpu.atomic_output import write_json_atomic, save_arrays_atomic
    from validation.hydraulics.merged_fwave115 import native, explicit as previous
    from validation.hydraulics.explicit_fwave124 import explicit
    native.install()  # Both methods use the same private spatial binary.
    before = worker.source_identity()
    original_prepare = worker.prepare
    def prepare(project, *args, **kwargs):
        path = Path(project); doc = json.loads(path.read_text())
        assert doc['model']['physics']['sediment'] is None
        assert not doc['settings']['adaptive']
        controls = doc['settings']['controls']
        assert controls['betac'] == controls['viscosity'] == controls['diffusivity'] == 0
        controls['threads'] = 8
        for key in ('temporal_depth_atol', 'temporal_momentum_atol', 'temporal_rtol'):
            controls[key] *= extra.tolerance_factor
        write_json_atomic(path, doc)
        return original_prepare(project, *args, **kwargs)
    worker.prepare = prepare
    original_factory = core.factory
    identities, owners, trials = [], [], []
    def factory(*args, **kwargs):
        make = original_factory(*args, **kwargs)
        def construct(*a, **kw):
            owner = make(*a, **kw)
            if extra.integration == 'explicit':
                explicit.install(owner.solver, extra.cfl)
            identities.append(owner.solver.identity.copy()); owners.append(owner)
            return owner
        if hasattr(make, 'required_leaves'):
            construct.required_leaves = make.required_leaves
        return construct
    core.factory = factory
    original_run = core.run
    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        result = dict(result, integrator=explicit.METHOD if extra.integration == 'explicit' else result['integrator'],
                      scope='isolated_explicit_fwave124_research', QGIS_admitted=False)
        write_json_atomic(Path(args[1])/'STATUS.json', result)
        return result
    core.run = run
    single = explicit.Explicit._single
    def trace(self, q, dt, at):
        start = time.perf_counter(); ev = self.evaluations; error = None
        try:
            return single(self, q, dt, at)
        except Exception as exc:
            error = str(exc); raise
        finally:
            trials.append(dict(time=at, dt=dt, evaluations=self.evaluations-ev,
                               error=error, wall_s=time.perf_counter()-start))
    explicit.Explicit._single = trace
    from tools import profile_test1_amr as profiler
    sys.argv = [profiler.__file__, *remaining]
    try:
        profiler.main()
    finally:
        if out.exists():
            if owners and extra.integration == 'explicit':
                a = owners[-1]; m = a.solver.midpoint
                save_arrays_atomic(out/'CFL_FINAL.npz', speed=host(m.speed), frequency=host(m.frequency),
                    owner=a.data['owner'], neighbour=a.data['neighbour'], normal=a.data['normal'], xy=a.data['xy'])
            paths = [Path(__file__), Path(profiler.__file__), Path(explicit.__file__),
                     Path(previous.__file__), Path(previous.__file__).with_suffix('.h'), Path(native.__file__)]
            write_json_atomic(out/'EXPERIMENT.json', dict(research_only=True, settings=vars(extra),
                actual_integrator=explicit.METHOD if extra.integration == 'explicit' else 'implicit_fwave_midpoint_river_v1',
                input_model_note='PROJECT declares public implicit baseline; this sidecar and STATUS declare the private actual method',
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                runtime_before=before, runtime_after=worker.source_identity(),
                solver_identities=identities, explicit_trials=trials, production_and_plugin_changed=False))


if __name__ == '__main__':
    main()
