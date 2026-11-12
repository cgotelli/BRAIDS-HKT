"""Paired operator and developed-state controls for the facewise limiter."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_jet173 import setup,document
from stage_wall172 import save,sha
BASE=ROOT/'validation/_trace179'


def main(backend,mode,variant):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    setup()
    if variant=='facewise':
        from face179 import install
        install(backend)
    out=BASE/mode/variant/backend;out.mkdir(parents=True,exist_ok=False)
    inputs={'bounded':ROOT/'validation/_tip178/probes/gpu/bounded.npz',
        'regular':ROOT/'validation/_quadrature176/regular_v2/gpu/RESIDUAL.npz',
        'potential':ROOT/'validation/_corner174/potential/gpu/RESIDUAL.npz',
        'affine':ROOT/'validation/_corner174/probes_v2/gpu/affine_+0.5.npz',
        'developed':ROOT/'validation/_tip178/probes/gpu/BRAIDS200.npz'}
    if mode=='probe':
        cases={k:dict(np.load(v)) for k,v in inputs.items()}
        original=river_worker.core.factory
        def factory(*args,**kw):
            make=original(*args,**kw)
            def build(*args,**kw):
                o=make(*args,**kw);xp=o.xp;c=o.midpoint;oldz=o.z.copy();save(out/'OWNER.json',o.identity)
                for name,a in cases.items():
                    q=np.ascontiguousarray(a['q']);o.z[:]=xp.asarray(a['z']);c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                    rate,water=c.rate(c.interval_base,1e-6,200.)
                    data=dict(q=q,z=a['z'],xy=a['xy'],rate=host(rate).copy(),water=host(water).copy())
                    if backend=='gpu':
                        data.update({k:host(v).copy() for k,v in o.hydraulics.flux.buffers.items()})
                        data.update({'rec_'+k:host(v).copy() for k,v in o.hydraulics.reconstruction.buffers.items()})
                    np.savez(out/(name+'.npz'),**data)
                o.z[:]=oldz;return o
            build.__dict__.update(make.__dict__);return build
        river_worker.core.factory=factory;d=document(backend,300.)
    else:
        a=np.load(inputs['developed']);d=document(backend,20.);d['settings']['output_interval_s']=.5
        recipe=river_worker.prepare_recipe
        def prepare(*args,**kw):
            previous=kw.get('on_fields')
            def fields(tree,data):
                if previous:previous(tree,data)
                xy=np.array([cell.centroid_xy for cell in tree.finest.cells])
                assert np.array_equal(xy,a['xy'])
                np.testing.assert_allclose(data['rock']+data['thickness'].sum(0),a['z'],rtol=0,atol=2e-15)
                data['q'][:]=a['q']
            kw['on_fields']=fields;return recipe(*args,**kw)
        river_worker.prepare_recipe=prepare
    save(out/'project.json',d);p=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(p,out/'run',until=.001 if mode=='probe' else None)
    save(out/'CHECK.json',dict(status=status,variant=variant,mode=mode,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
        source={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'desktop/face179.py')},
        inputs={k:sha(p) for k,p in inputs.items()},production_modified=False,Delft3D_modified=False,
        scope='Explicit new initial-condition run from saved t=200 s; not reinterpretation of an old checkpoint. '
              'Constant hydraulic forcing; morphology inactive; elapsed 0..20 s corresponds to original 200..220 s.'))
    print('Complete',mode,variant,backend,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'));p.add_argument('mode',choices=('probe','run'))
    p.add_argument('--variant',choices=('facewise','baseline'),default='facewise');a=p.parse_args();main(a.backend,a.mode,a.variant)
