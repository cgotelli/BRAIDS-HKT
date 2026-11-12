"""Known-answer corner flux checks and delayed-bedload controls.

Only validation inputs and diagnostics change. The installed preview27 and
the locked FLOW engine remain immutable.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop'),
             str(ROOT/'validation/sediment/delft3d_current')]
from stage_wall172 import save,sha
from check_jet173 import document,setup
BASE=ROOT/'validation/_corner174'


def project(backend):
    d=document(backend,340.)
    d['settings']['sediment_start_time_s']=300.
    return d


def run(backend,resume=False):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    setup();out=BASE/backend
    if not resume:
        out.mkdir(parents=True,exist_ok=False);save(out/'project.json',project(backend))
        prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
        result=river_worker.run(out/'prepared',out/'warmup',until=300.)
        name='WARMUP.json'
    else:
        gate=json.loads((BASE/'SETTLING.json').read_text())
        assert gate['both_pass'],'Hydraulics not qualified for morphology activation'
        result=river_worker.run(out/'prepared',out/'mobile',restart=out/'warmup/checkpoint.npz')
        name='MOBILE.json'
    save(out/name,dict(status=result,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
                      analysis_sha256=sha(__file__)))
    print(json.dumps(result),flush=True)


def native():
    from check_bedload161 import replace
    import runtime
    parent=ROOT/'validation/_jet173/fixed/delft';out=BASE/'delft'
    out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest
        shutil.copy2(parent/name,out/name)
    p=out/'sim_P057.mdf';p.write_text(replace(p.read_text(),'Tstop','340'))
    code=p.read_text()
    for key,value in dict(Flmap='0 1 340',Flhis='0 1 340').items():code=replace(code,key,value)
    p.write_text(code)
    p=out/'mor.mor';p.write_text(replace(p.read_text(),'MorStt','300'))
    for name in ('flume.bct','bcm.bcm'):
        p=out/name;lines=p.read_text().splitlines()
        for i,line in enumerate(lines):
            tokens=line.split()
            if tokens and tokens[0]=='2':lines[i]=f'{340/60:.14g} '+' '.join(tokens[1:])
        p.write_text('\n'.join(lines)+'\n')
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=dict(end_time_s=340,sediment_start_time_s=300,output_interval_s=1,
                     constant_tables_extended_to_minutes=340/60),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=240)),flush=True)


def probe(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    from braids_desktop.results import ResultsStore
    setup();out=BASE/'probes_v2'/backend;out.mkdir(parents=True,exist_ok=False)
    f=ResultsStore(ROOT/'validation/_jet173/fixed/cpu/run').read(0)
    xy=np.array([p.centroid.coords[0] for p in f.polygons]);X=xy[:,0]-8;Y=xy[:,1]-3
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint
            original_z=owner.z.copy();save(out/'OWNER.json',owner.identity)
            for a in (.1,.5,-.5):
                # Divergence-free affine field u=-aX, v=aY satisfies both
                # faces of the leading slip-wall corner. Exact area-average
                # divergence is [0,-h*a*a*X,-h*a*a*Y]. No drag in this residual.
                analytic=np.vstack([np.full(len(xy),.5),-.5*a*X,.5*a*Y])
                # Do not send the manufactured field through the unrelated
                # real inlet/outlet. A wide local patch encloses every tested
                # stencil, with original resting conditions outside it.
                patch=(xy[:,0]>=5)&(xy[:,0]<=12)
                q=f.q.copy();q[:,patch]=analytic[:,patch]
                z=f.z.copy();z[patch]=0
                owner.z[:]=xp.asarray(z);c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,1e-6,40.)
                arrays=dict(q=q,z=z,xy=xy,rate=host(rate).copy(),water=host(water).copy(),
                    exact=np.vstack([np.zeros(len(xy)),-.5*a*a*X,-.5*a*a*Y]))
                arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
                if backend=='gpu':
                    arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/f'affine_{a:+.1f}.npz',**arrays)
            owner.z[:]=original_z
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    save(out/'project.json',project(backend))
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(prepared,out/'construction_run',until=.001)
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),
                             script_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.glob('*.npz')}))
    for p in sorted(out.glob('*.npz')):
        a=np.load(p);m=(xy[:,0]>=7.25)&(xy[:,0]<=8.75)&(xy[:,1]>=2.25)&(xy[:,1]<=4.25)
        print(p.stem,json.dumps(dict(xy=xy[m].tolist(),difference=(a['rate']-a['exact'])[:,m].tolist())),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('run','resume','native','probe'))
    ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu');a=ap.parse_args()
    if a.action=='native':native()
    elif a.action=='probe':probe(a.backend)
    else:run(a.backend,a.action=='resume')
