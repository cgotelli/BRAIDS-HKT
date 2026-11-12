"""Isolated, explicitly identified numerical experiment on an alpha129 input.

The installed plugin and maintained runtime sources are never edited. Public
project CFL remains its declared value; the experimental effective limit and
wet/dry threshold are recorded in EXPERIMENT.json and controller identities.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    p.add_argument('--method', choices=['normal', 'critical', 'stage', 'depth'], default='critical')
    p.add_argument('--stage', type=float, default=464.9)
    p.add_argument('--until', type=float, default=100.)
    p.add_argument('--outer-dt', type=float, default=.5)
    p.add_argument('--project-cfl', type=float, default=.5)
    p.add_argument('--cfl-cap', type=float, default=.5)
    p.add_argument('--dry', type=float, default=1e-6)
    p.add_argument('--clock', choices=['baseline', 'carry', 'near_limit', 'combined', 'measured_retry', 'combined_measured'], default='baseline')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    from braids_core import river_worker as worker
    from tools.research_wet_cfl130 import candidate
    from tools import qualify_inlet129
    baseline = worker.source_identity()
    source_hash = sha(a.source)
    scripts = [Path(__file__), Path(qualify_inlet129.__file__), ROOT/'tools/research_wet_cfl130.py']
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    error = None
    context = None
    clock_receipt = None
    try:
        with ExitStack() as stack:
            context = stack.enter_context(candidate(a.cfl_cap, a.dry))
            if a.clock != 'baseline':
                from validation.hydraulics.speed130_clock import candidate as clock_candidate
                scripts.append(ROOT/'validation/hydraulics/speed130_clock.py')
                clock_receipt = stack.enter_context(clock_candidate(a.clock))
            sys.argv = [qualify_inlet129.__file__, '--source', str(a.source),
                        '--output', str(a.output), '--backend', a.backend, '--method', a.method,
                        '--until', str(a.until), '--cfl', str(a.project_cfl), '--dt-max', str(a.outer_dt)]
            if a.method in ('stage', 'depth'):
                sys.argv += ['--stage', str(a.stage)]
            qualify_inlet129.main()
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        after = worker.source_identity()
        if a.output.exists():
            result = dict(research_only=True, settings=settings, source_project_sha256=source_hash,
                          runtime_before=baseline, runtime_after=after,
                          scripts={str(s.relative_to(ROOT)):sha(s) for s in scripts}, error=error,
                          public_project_note='PROJECT.json names the base public method. This experiment sidecar and private controller identity declare the actual numerical overrides.',
                          installed_plugin_changed=False, clock_options=clock_receipt,
                          actual_owner_settings=[] if context is None else [
                              dict(native_dry=o.data['dry'], physical_dry=o.physics.parameters.dry,
                                   effective_cfl=o.midpoint.cfl, identity=o.midpoint.identity)
                              for o in context['owners']])
            (a.output/'EXPERIMENT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        assert baseline == after, 'Maintained runtime changed during experiment'
        assert source_hash == sha(a.source), 'Source input changed'


if __name__ == '__main__':
    main()
