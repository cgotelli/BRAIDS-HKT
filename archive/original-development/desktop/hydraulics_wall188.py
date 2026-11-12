"""Public-worker closed oblique walls, lake-at-rest and wet/dry run-up."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TARGET=ROOT/'validation/_wall188/desktop_candidate'
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path[:0]=[str(TARGET),str(ROOT)]
from braids_core import river_worker
from braids_core.river_launcher import verify
from braids_core.sediment.portable_river_runtime import install
from braids_desktop import services
from braids_desktop.preparation import prepare
from braids_desktop.results import ResultsStore
from stage_secondary185 import save,sha
import check_wall171 as retained
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),required=True)
    args=ap.parse_args();verify();install(json.loads((TARGET/'PRECOMPILED.json').read_text()),TARGET)
    for kind,angle in [('lake',0),('lake',37),('lake',63),('wetting',37)]:
        out=TARGET.parent/'hydraulics'/f'{kind}_{angle}_{args.backend}';out.mkdir(parents=True,exist_ok=False)
        t,n,origin=retained.document(out/'inputs',args.backend,angle,'wetting')
        doc=services.load_project_folder(out/'inputs/project')
        if kind=='lake':doc['initial']['velocity_m_s']=[0.,0.]
        services.save_project_folder(out/'inputs/project',doc)
        prep=out/'prepared';prepare(out/'inputs/project/braids_project.json',prep,asset_cache=out/'cache')
        factory=river_worker.core.factory;owners=[]
        def instrument(*aa,**kk):
            maker=factory(*aa,**kk)
            def build(*a,**k):
                owner=maker(*a,**k);owners.append(owner)
                if kind=='wetting':
                    probe=out/'probes';probe.mkdir();retained.probes(owner,probe,t,n)
                return owner
            build.__dict__.update(maker.__dict__);return build
        river_worker.core.factory=instrument
        try:status=river_worker.run(prep,out/'run')
        finally:river_worker.core.factory=factory
        assert status['status']=='complete'
        frames=ResultsStore(out/'run');first,last=frames.read(0),frames.read(-1)
        delta=float(np.max(abs(last.q-first.q)))
        if kind=='lake':assert delta<1e-10,delta
        assert last.q[0].min()>=0 and np.isfinite(last.q).all()
        rows=json.loads((out/'run/DIAGNOSTICS.json').read_text())
        assert all(not any(r['wall_integrals']) for r in rows)
        assert max(abs(r['water_residual_m3']) for r in rows)<1e-9
        owner=owners[-1];count=np.diff(owner.data['offset'])
        save(out/'CHECK.json',dict(status=status,max_q_change=delta,water_residual_m3=max(abs(r['water_residual_m3']) for r in rows),
            minimum_depth=float(last.q[0].min()),wall_water_zero=True,maximum_faces=int(count.max()),
            projection_capacity_fallback_cells=int(np.count_nonzero(count>32)),runtime=sha(TARGET/'RUNTIME.json')))
        print('PASS',kind,angle,args.backend,delta,flush=True)


if __name__=='__main__':main()
