"""Same-state checks of cross-momentum transport without any solver changes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'desktop/runtime'), str(ROOT), str(ROOT/'desktop')]
from braids_core import river_worker
from braids_core.sediment.portable_river_runtime import install
from braids_core.sediment.bed_columns_v101 import host
from check_bedload161 import save, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    args = parser.parse_args()
    out = ROOT/f'validation/_advection168/probes/{args.backend}'
    out.mkdir(parents=True, exist_ok=False)
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()), ROOT/'desktop/runtime')
    base = np.load(ROOT/'validation/_scour166/v3/mpm_cpu/BRAIDS.npz')
    xy = base['xy']; y = xy[:, 1]-3
    cases = {}
    for slope in (0., .2):
        for sign in (-1, 1):
            label = ('uniform' if slope == 0 else 'shear')+('_in' if sign < 0 else '_out')
            cases[label] = np.array([np.full(len(xy), .5), .5*(1+slope*y), .5*.3*sign*y])
    original = river_worker.core.factory

    def factory(*a, **kw):
        make = original(*a, **kw)

        def build(*a, **kw):
            owner = make(*a, **kw); xp = owner.xp; c = owner.midpoint
            for label, q in cases.items():
                owner.z[:] = 0.
                c.interval_base = xp.asarray(q).copy(); c.minimum_theta = 1.; c.frequency_peak = 0.
                rate, water = c.rate(c.interval_base, .0001, 40.)
                data = dict(q=q, z=np.zeros(len(xy)), xy=xy, rate=host(rate).copy(), water=host(water).copy())
                data.update({k: owner.data[k] for k in ('owner', 'neighbour', 'normal', 'mid', 'length', 'area', 'bid')})
                if args.backend == 'gpu':
                    data.update({k: host(v).copy() for k, v in owner.hydraulics.flux.buffers.items()})
                    data.update({'rec_'+k: host(v).copy() for k, v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/(label+'.npz'), **data)
            owner.z[:] = xp.asarray(base['z'])
            return owner
        build.__dict__.update(make.__dict__)
        return build

    river_worker.core.factory = factory
    try:
        source = ROOT/f'validation/_shoal165/groin/mpm_{args.backend}/project.json'
        prepared = river_worker.prepare(source, out/'prepared')
        status = river_worker.run(prepared, out/'construction_run', until=.001)
        save(out/'AUDIT.json', dict(backend=args.backend, status=status, script_sha256=sha(__file__),
             runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'), solver_modified=False,
             outputs={p.name: sha(p) for p in out.glob('*.npz')}))
    finally:
        river_worker.core.factory = original


if __name__ == '__main__':
    main()
