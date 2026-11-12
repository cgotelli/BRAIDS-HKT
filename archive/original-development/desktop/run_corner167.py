"""Short fresh groin replay with an isolated hydraulic reconstruction candidate."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--variant',choices=['tangent','normal','lswall','normal_ls'],required=True)
    ap.add_argument('--tag');ap.add_argument('--until',type=float,default=60.)
    ap.add_argument('--dt-scale',type=float,default=1.)
    a=ap.parse_args()
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from check_bedload161 import save,sha
    from corner_candidate167 import install as candidate
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    candidate(a.backend,a.variant)
    from corner_candidate167 import configure
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);configure(owner,a.variant);return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    out=ROOT/f'validation/_corner167/runs/{a.tag or a.variant}/{a.backend}'
    out.mkdir(parents=True,exist_ok=False)
    doc=json.loads((ROOT/f'validation/_shoal165/groin/mpm_{a.backend}/project.json').read_text())
    doc['settings']['dt_max_s']*=a.dt_scale
    save(out/'project.json',doc)
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    start=time.perf_counter()
    status=river_worker.run(prepared,out/'run',until=a.until,checkpoint_dir=out/'checkpoints')
    save(out/'STATUS.json',dict(status=status,elapsed_s=time.perf_counter()-start,
         variant=a.variant,backend=a.backend,script_sha256=sha(__file__),candidate_sha256=sha(ROOT/'desktop/corner_candidate167.py')))
    print(status,flush=True)


if __name__=='__main__':main()
