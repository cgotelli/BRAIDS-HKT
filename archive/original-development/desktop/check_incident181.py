"""Paired isolated candidate, using existing audited prepared-field harnesses."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from wallstate181 import install,configure,METHOD
from stage_wall172 import save,sha

def setup_factory(original,backend):
    original();install(backend)
    from braids_core import river_worker
    factory0=river_worker.core.factory
    def factory(*args,**kw):
        make=factory0(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);configure(o);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory

def main(backend,mode,half):
    base=ROOT/'validation/_incident181'
    if mode=='walls':
        import control_trace179 as check
        old=check.setup;check.setup=lambda:setup_factory(old,backend);check.install=lambda backend:None
        check.BASE=base;check.main(backend);folder=base/'controls'/backend
    else:
        import check_trace179 as check
        old=check.setup;check.setup=lambda:setup_factory(old,backend);check.BASE=base
        doc=check.document
        def document(*args,**kw):
            d=doc(*args,**kw)
            if half:d['settings']['controls']['cfl']=.5
            return d
        check.document=document;variant='wall_half' if half else 'wall'
        check.main(backend,mode,variant);folder=base/mode/variant/backend
    save(folder/'METHOD181.json',dict(method=METHOD,candidate_sha256=sha(ROOT/'desktop/wallstate181.py'),
        script_sha256=sha(__file__),production_changed=False,Delft3D_changed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'));p.add_argument('mode',choices=('probe','run','walls'))
    p.add_argument('--half',action='store_true');a=p.parse_args();main(a.backend,a.mode,a.half)
