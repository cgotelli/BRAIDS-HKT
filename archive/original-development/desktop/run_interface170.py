"""One short fresh CPU/GPU groin control for exact reflective-wall pressure."""
import argparse
import json
from pathlib import Path
import sys
import shutil
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--tag',default='pressure_v2');a=ap.parse_args()
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from check_bedload161 import save,sha
    from wall_pressure170 import install as candidate,METHOD
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    candidate(a.backend)
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw)
            identity=dict(method=METHOD,script_sha256=sha(ROOT/'desktop/wall_pressure170.py'))
            if a.backend=='gpu':identity['cuda_sha256']=owner.identity['activity_generated_cuda_sha256']
            else:identity['library_sha256']=sha(Path(owner.hydraulics.lib._name))
            owner.identity['isolated_wall_pressure170']=identity
            save(out/'CANDIDATE.json',identity)
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    out=ROOT/f'validation/_interface170/runs/{a.tag}/{a.backend}';out.mkdir(parents=True,exist_ok=False)
    for p in (Path(__file__),ROOT/'desktop/wall_pressure170.py'):shutil.copy2(p,out/p.name)
    doc=json.loads((ROOT/f'validation/_shoal165/groin/mpm_{a.backend}/project.json').read_text())
    save(out/'project.json',doc)
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    start=time.perf_counter();status=river_worker.run(prepared,out/'run',until=60.,checkpoint_dir=out/'checkpoints')
    save(out/'STATUS.json',dict(status=status,elapsed_s=time.perf_counter()-start,backend=a.backend,
        script_sha256=sha(__file__),candidate_sha256=sha(ROOT/'desktop/wall_pressure170.py')))
    print(status,flush=True)


if __name__=='__main__':main()
