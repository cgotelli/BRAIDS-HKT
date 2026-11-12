"""Actual CPU/GPU boundary maps on uniform infinitesimal characteristics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from pulse177 import BASE,H,U,C
from check_jet173 import setup
from stage_wall172 import save,sha


def main(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup();out=BASE/'operators'/backend;out.mkdir(parents=True,exist_ok=False)
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            a=make(*args,**kw);e=a.exterior_bed;xp=a.xp
            save(out/'OWNER.json',a.identity)
            for amplitude in (0.,1e-5,-1e-5):
                for sign in (-1,1):
                    h=np.full(a.q.shape[1],H+amplitude)
                    q=np.vstack([h,h*(U+sign*C/H*amplitude),np.zeros_like(h)])
                    e.evaluate(xp.asarray(q),a.z,time=0.)
                    arrays=dict(q=q,amplitude=amplitude,direction=sign,
                        trace=host(e.trace).copy(),flux=host(e.flux).copy())
                    arrays.update({k:a.data[k] for k in ('owner','neighbour','normal','mid','length','bid')})
                    np.savez(out/f'{amplitude:+.0e}_{sign:+d}.npz',**arrays)
            return a
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    status=river_worker.run(BASE/'base'/backend/'prepared',out/'construction',until=.001)
    save(out/'CHECK.json',dict(status=status,script_sha256=sha(__file__),
        runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),production_changed=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('backend',choices=('cpu','gpu'));main(ap.parse_args().backend)
