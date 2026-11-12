"""Verify exact wall pressure on the final public owner construction route."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True);a=ap.parse_args()
    from braids_core import river_worker
    from braids_core.sediment.portable_river_runtime import install
    from braids_core.sediment.bed_columns_v101 import host
    from wall_pressure170 import install as candidate
    from check_bedload161 import save,sha
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    candidate(a.backend)
    out=ROOT/f'validation/_interface170/probes/{a.backend}';out.mkdir(parents=True,exist_ok=False)
    for path in (Path(__file__),ROOT/'desktop/wall_pressure170.py'):shutil.copy2(path,out/path.name)
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint
            z0=owner.z.copy()
            for name in ('BRAIDS','FLOW_PRIMARY','REST'):
                old=np.load(ROOT/f'validation/_corner167/baseline/gpu/{name}.npz')
                q=old['q'];owner.z[:]=xp.asarray(old['z'])
                c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,.0001,40.)
                data=dict(q=q,z=old['z'],rate=host(rate).copy(),water=host(water).copy(),
                    limiter_theta=host(c.theta).copy())
                if a.backend=='gpu':
                    data.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    data.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/(name+'.npz'),**data)
            owner.z[:]=z0
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    source=ROOT/f'validation/_shoal165/groin/mpm_{a.backend}/project.json'
    prepared=river_worker.prepare(source,out/'prepared')
    status=river_worker.run(prepared,out/'construction_run',until=.001)
    save(out/'AUDIT.json',dict(status=status,script_sha256=sha(__file__),
        candidate_sha256=sha(ROOT/'desktop/wall_pressure170.py'),outputs={p.name:sha(p) for p in out.glob('*.npz')}))


if __name__=='__main__':main()
