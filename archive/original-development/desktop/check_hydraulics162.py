"""Bounded, explicitly labelled hydraulic controls; no installed runtime edits.

The tangent contrast is a diagnostic, not a new supported boundary option.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'desktop'))
import check_bedload161 as case
from braids_core import river_worker
from braids_core.sediment import adaptive_river_worker
from braids_core.sediment.bed_columns_v101 import host
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('baseline', 'tangent', 'half_cfl'), required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    args = parser.parse_args()
    case.BASE = ROOT/'validation/_hydraulics162'/args.variant
    case.END = 40.
    case.SPIN = 40.
    out = case.BASE/('braids_mpm_'+args.backend)
    original = river_worker.core.factory
    owners = []
    snapshots = []

    def factory(*a, **kw):
        make = original(*a, **kw)
        def build(*a, **kw):
            owner = make(*a, **kw)
            owners.append(owner)
            e = owner.exterior_bed
            xp = owner.xp
            # Research identity participates in this control's checkpoint.
            owner.identity['diagnostic162'] = dict(variant=args.variant, source_sha256=case.sha(__file__))
            if args.variant == 'half_cfl':
                owner.midpoint.cfl *= .5
                owner.midpoint.identity['cfl'] = owner.midpoint.cfl
            if args.variant == 'tangent':
                evaluate = e.evaluate
                def altered(q, z, **kwargs):
                    flux = evaluate(q, z, **kwargs)
                    g = owner.physics.g
                    mask = (e.hydraulic_mode[g['bid']] == 2) & e.face_selected & (flux[0] < 0)
                    cells = g['owner'][mask]
                    n = g['normal'][:, mask]
                    v = q[1:, cells]/q[0, cells]
                    tangent = v-xp.sum(v*n, axis=0)*n
                    e.trace[1:, mask] += e.trace[0, mask]*tangent
                    flux[1:, mask] += flux[0, mask]*tangent
                    return flux
                e.evaluate = altered
            return owner
        build.__dict__.update(make.__dict__)
        return build
    river_worker.core.factory = factory
    fields = adaptive_river_worker.fields
    def sampled(owner):
        result = fields(owner)
        e = owner.exterior_bed
        e.evaluate(owner.q, owner.z, time=owner.time)
        d = owner.data
        path = out/'audit'
        path.mkdir(exist_ok=True)
        snap = path/f'{owner.time:08.3f}.npz'
        np.savez(snap, time=owner.time, q=host(owner.q), z=host(owner.z),
                 trace=host(e.trace), flux=host(e.flux), bed=host(e.p.exterior_bed),
                 **{k:d[k] for k in ('owner','neighbour','bid','mid','normal','length','area')})
        snapshots.append(dict(time=owner.time, file=str(snap.relative_to(out)), sha256=case.sha(snap)))
        return result
    adaptive_river_worker.fields = sampled
    try:
        case.braids('mpm', args.backend)
        case.save(out/'AUDIT.json', dict(variant=args.variant, backend=args.backend,
            runtime_manifest_sha256=case.sha(case.RUNTIME/'RUNTIME.json'),
            script_sha256=case.sha(__file__), snapshots=snapshots,
            research_only=args.variant != 'baseline', no_runtime_source_modification=True))
    finally:
        river_worker.core.factory = original
        adaptive_river_worker.fields = fields


if __name__ == '__main__':
    main()
