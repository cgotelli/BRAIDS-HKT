"""Short cropped-wall, rest, wetting and restart controls for research179."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_jet173 import setup
from check_wall171 import document,probes
from stage_wall172 import save,sha
from face179 import install
BASE=ROOT/'validation/_trace179'


def main(backend):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    setup();install(backend)
    original=river_worker.core.factory
    for case,angle in [('reflection',0),('reflection',37),('wetting',37),('lake',37)]:
        out=BASE/'controls'/backend/f'{case}_{angle}';out.mkdir(parents=True,exist_ok=False)
        t,n,_=document(out/'inputs',backend,angle,'wetting' if case=='lake' else case)
        path=out/'inputs/project/braids_project.json'
        if case=='lake':
            doc=json.loads(path.read_text());doc['initial']['velocity_m_s']=[0.,0.];save(path,doc)
        prepared=out/'prepared';prepare(path,prepared,asset_cache=out.parent/'asset_cache')
        phase=['probe' if case=='reflection' else 'full']
        def factory(*args,**kw):
            make=original(*args,**kw)
            def build(*args,**kw):
                o=make(*args,**kw);d=o.data
                save(out/'OWNER.json',o.identity)
                np.savez(out/'GEOMETRY.npz',**{k:d[k] for k in ('xy','area','owner','neighbour','normal','length')})
                assert d['area'].min()>=.25-1e-9
                if phase[0]=='probe':
                    p=out/'probes';p.mkdir();probes(o,p,t,n)
                return o
            build.__dict__.update(make.__dict__);return build
        river_worker.core.factory=factory
        if phase[0]=='probe':
            river_worker.run(prepared,out/'probe_run',until=.001);phase[0]='full'
        result=river_worker.run(prepared,out/'full');assert result['status']=='complete'
        s=ResultsStore(out/'full');fs=[s.read(j) for j in range(len(s.times))]
        q=np.array([f.q for f in fs]);z=np.array([f.z for f in fs])
        assert np.isfinite(q).all() and q[:,0].min()>=0
        assert np.array_equal(z,np.broadcast_to(z[0],z.shape))
        rows=json.loads((out/'full/DIAGNOSTICS.json').read_text())
        balance=max(abs(r['water_residual_m3']) for r in rows)
        wall=max(abs(v) for r in rows for v in r['wall_integrals'])
        assert balance<1e-10 and wall==0
        restart={}
        if case=='reflection' and angle==37:
            phase[0]='cancel';stop=out/'STOP'
            def cancel(folder,row):
                if row['time_s']>=2:stop.touch()
            part=river_worker.run(prepared,out/'cancelled',stop_file=stop,on_frame=cancel)
            assert part['status']=='cancelled' and part['time_s']==2
            resumed=river_worker.run(prepared,out/'resumed',restart=out/'cancelled/checkpoint.npz')
            a=ResultsStore(out/'full').read(-1);b=ResultsStore(out/'resumed').read(-1)
            equal={k:bool(np.array_equal(a.arrays[k],b.arrays[k])) for k in ('q','z','mass','water','bed_boundary')}
            assert all(equal.values());restart=dict(cancel=part,resume=resumed,exact=equal)
        rest=float(abs(q-q[0]).max()) if case=='lake' else None
        if rest is not None:assert rest<1e-11
        save(out/'CHECK.json',dict(status=result,backend=backend,case=case,angle=angle,
            minimum_depth=float(q[:,0].min()),water_residual_max_m3=balance,wall_integral_max=wall,
            bed_unchanged=True,rest_change=rest,restart=restart,
            script_sha256=sha(__file__),candidate_sha256=sha(ROOT/'desktop/face179.py')))
        print('Complete control',backend,case,angle,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('backend',choices=('cpu','gpu'));main(p.parse_args().backend)
