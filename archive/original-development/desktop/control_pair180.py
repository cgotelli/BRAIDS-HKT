"""Wall/restart and wave regression controls for the normal-only candidate."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from normal180 import configure,install
from stage_wall172 import save,sha


def setup_factory(setup):
    setup()
    from braids_core import river_worker
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            o=make(*args,**kw);configure(o);return o
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory


def main(backend,case):
    if case=='walls':
        import control_trace179 as control
        original=control.setup;control.setup=lambda:setup_factory(original)
        control.install=install;control.BASE=ROOT/'validation/_pair180'
        control.main(backend);folder=control.BASE/'controls'/backend
    else:
        import pulse177 as pulse
        original=pulse.setup
        def setup():
            setup_factory(original);install(backend)
        pulse.setup=setup;pulse.BASE=ROOT/'validation/_pair180/pulses'
        pulse.braids(case,backend);folder=pulse.BASE/case/backend
    save(folder/'METHOD180.json',dict(method='normal_reflected180_research',backend=backend,
        script_sha256=sha(__file__),candidate_sha256=sha(ROOT/'desktop/normal180.py'),
        note='Reused harness legacy candidate/source labels do not identify this build; this receipt and configured owner do.',
        production_changed=False,Delft3D_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'))
    p.add_argument('case',choices=('walls','base','left','right'));a=p.parse_args();main(a.backend,a.case)
