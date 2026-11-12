"""Matched cold-start/restart qualification of the isolated wall-bound fix."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import sha,save
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from wall_bounds163 import install as wall_test


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--formula',default='mpm',choices=['mpm','ashida_michiue_generalized','wang_fredsoe'])
    ap.add_argument('--until',type=float,default=60.)
    ap.add_argument('--restart')
    ap.add_argument('--tag',default='coupled')
    args=ap.parse_args()
    inp=ROOT/f'validation/_hydraulics162/mobile/braids_{args.formula}_{args.backend}/prepared'
    out=ROOT/f'validation/_groin163/{args.tag}/braids_{args.formula}_{args.backend}'
    out.mkdir(parents=True,exist_ok=False)
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    wall_test(args.backend)
    original=river_worker.core.factory
    def factory(*a,**kw):
        make=original(*a,**kw)
        def build(*a,**kw):
            owner=make(*a,**kw)
            owner.identity['groin163_research']='reflected_wall_velocity_limiter_bounds'
            return owner
        build.__dict__.update(make.__dict__)
        return build
    river_worker.core.factory=factory
    try:
        status=river_worker.run(inp,out/'run',until=args.until,restart=args.restart)
        save(out/'STATUS.json',dict(status=status,prepared=str(inp),restart=args.restart,
             script_sha256=sha(__file__),probe_sha256=sha(ROOT/'desktop/wall_bounds163.py'),
             installed_runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),production_modified=False))
        print(status,flush=True)
    finally:river_worker.core.factory=original


if __name__=='__main__':main()
