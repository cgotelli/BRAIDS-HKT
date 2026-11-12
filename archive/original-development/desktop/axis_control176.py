"""Paired known-answer and zero-viscosity controls for axis-wall stencils."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_jet173 import setup,document
from stage_wall172 import save,sha
from axis_stencil176 import install,configure
BASE=ROOT/'validation/_quadrature176/axis_stencil'


def main(backend,end=None):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup();install(backend)
    out=BASE/('probe' if end is None else f'run{end:g}')/backend;out.mkdir(parents=True,exist_ok=False)
    inputs={'potential':ROOT/'validation/_corner174/potential/gpu/RESIDUAL.npz',
            'regular':ROOT/'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz',
            'affine':ROOT/'validation/_corner174/probes_v2/gpu/affine_+0.5.npz',
            'base40':ROOT/'validation/_zero175/probes_v2/baseline/gpu/base40.npz'}
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);changed=configure(owner);save(out/'OWNER.json',owner.identity)
            if end is not None:return owner
            xp=owner.xp;c=owner.midpoint;oldz=owner.z.copy()
            for name,path in inputs.items():
                a=np.load(path);q=np.ascontiguousarray(a['q']);owner.z[:]=xp.asarray(a['z'])
                c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,1e-6,40.)
                arrays=dict(q=q,z=a['z'],xy=a['xy'],rate=host(rate).copy(),water=host(water).copy())
                if backend=='gpu':
                    arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/(name+'.npz'),**arrays)
            owner.z[:]=oldz;return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory;save(out/'project.json',document(backend,end or 120.))
    if end is None:
        prepared=river_worker.prepare(out/'project.json',out/'prepared')
        status=river_worker.run(prepared,out/'construction',until=.001)
    else:
        from braids_desktop.preparation import prepare
        prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
        status=river_worker.run(out/'prepared',out/'run')
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        sources={p.name:sha(p) for p in (Path(__file__),ROOT/'desktop/axis_stencil176.py')},
        replay_inputs={k:sha(p) for k,p in inputs.items()},production_changed=False,Delft3D_changed=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu');ap.add_argument('--end',type=float)
    a=ap.parse_args();main(a.backend,a.end)
