"""Profile/benchmark the untouched compiler-free alpha129 runtime in QGIS.

Checkout tools supply instrumentation or explicitly declared research hooks only.
The authenticated installed runtime supplies every maintained solver module.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--runtime', type=Path, default=Path('C:/Users/STREEM/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/braids_qgis/river_runtime'))
    p.add_argument('--optimizations', choices=('none', 'mesh'), default='none')
    p.add_argument('--numerics', choices=('baseline', 'conservative', 'selected'), default='baseline')
    a, remaining = p.parse_known_args()
    from tools import profile_execution130 as profile
    runtime = a.runtime.resolve()
    sys.path.insert(0, str(runtime))
    sys.dont_write_bytecode = True
    from braids_core.river_launcher import verify
    manifest = verify()
    from braids_core.sediment.portable_river_runtime import install as portable_install
    portable_install(json.loads((runtime/'PRECOMPILED.json').read_text()), runtime)
    from braids_core import river_worker
    if not Path(river_worker.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError('Worker did not import from the selected installed runtime')
    out = Path(remaining[remaining.index('--output')+1]).resolve()
    sys.argv = [profile.__file__, *remaining]
    receipts = dict(installed_runtime=str(runtime), installed_manifest_sha256=hashlib.sha256((runtime/'RUNTIME.json').read_bytes()).hexdigest(),
        precompiled_sha256=hashlib.sha256((runtime/'PRECOMPILED.json').read_bytes()).hexdigest(),
        installation_modified=False, worker_origin=river_worker.__file__, optimizations=a.optimizations, numerics=a.numerics)
    with ExitStack() as stack:
        numerical = None
        if a.numerics != 'baseline':
            from tools.research_wet_cfl130 import candidate as cfl_candidate
            from validation.hydraulics.speed130_clock import candidate as clock_candidate
            cap = .5 if a.numerics == 'conservative' else 1.
            numerical = stack.enter_context(cfl_candidate(cap, 1e-6))
            receipts['clock'] = stack.enter_context(clock_candidate('combined'))
            receipts['actual_cfl_cap'] = cap
            receipts['hydraulic_dry_m'] = 1e-6
            receipts['research_hook_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ('tools/research_wet_cfl130.py', 'validation/hydraulics/speed130_clock.py')}
        if a.optimizations == 'mesh':
            from tools.execution_optimizations130 import install as prototype
            stack.enter_context(prototype(mesh=True, toolchain=False))
        try:
            profile.main()
        finally:
            if out.exists():
                receipts['module_origins'] = {name: module.__file__ for name, module in sys.modules.items()
                    if name.startswith(('braids_core.', 'braids_gpu.')) and getattr(module, '__file__', None)}
                receipts['all_solver_modules_installed'] = all(Path(path).resolve().is_relative_to(runtime)
                    for path in receipts['module_origins'].values())
                if numerical is not None:
                    receipts['actual_owner_settings'] = [dict(actual_cfl=o.midpoint.cfl, native_dry=o.data['dry'],
                        physical_dry=o.physics.parameters.dry, identity=o.midpoint.identity) for o in numerical['owners']]
                receipts['manifest_after_sha256'] = hashlib.sha256((runtime/'RUNTIME.json').read_bytes()).hexdigest()
                receipts['source_integrity_after'] = verify() == manifest
                (out/'INSTALLED_RUNTIME.json').write_text(json.dumps(receipts, indent=2)+'\n')
                assert receipts['all_solver_modules_installed']
                assert receipts['source_integrity_after']


if __name__ == '__main__':
    main()
