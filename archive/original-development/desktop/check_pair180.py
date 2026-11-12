"""Run matched research180 controls with explicit backend and new initial state."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
import check_trace179 as check
from normal180 import install,configure
from stage_wall172 import save,sha


def main(backend,mode,half):
    from braids_core import river_worker
    oldsetup=check.setup
    def setup():
        oldsetup();install(backend);original=river_worker.core.factory
        def factory(*args,**kw):
            make=original(*args,**kw)
            def build(*args,**kw):
                o=make(*args,**kw);configure(o);return o
            build.__dict__.update(make.__dict__);return build
        river_worker.core.factory=factory
    check.setup=setup;check.BASE=ROOT/'validation/_pair180'
    # Keep check_trace179's prepared saved-field inputs in its original folder.
    doc=check.document
    def document(*args,**kw):
        d=doc(*args,**kw)
        if half:d['settings']['controls']['cfl']=.5
        return d
    check.document=document
    variant='normal_half' if half else 'normal'
    check.main(backend,mode,variant)
    out=check.BASE/mode/variant/backend
    save(out/'METHOD.json',dict(method='normal_reflected180_research',scripts={p.name:sha(p) for p in
         (Path(__file__),ROOT/'desktop/normal180.py',ROOT/'desktop/axis_stencil176.py')},
         public_admitted=False,installed_runtime_changed=False,Delft3D_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'));p.add_argument('mode',choices=('probe','run'))
    p.add_argument('--half',action='store_true');a=p.parse_args();main(a.backend,a.mode,a.half)
