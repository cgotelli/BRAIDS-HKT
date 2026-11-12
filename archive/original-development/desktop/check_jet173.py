"""Frozen-state hydraulic attribution and fixed-bed controls; no engine edits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'desktop/runtime'
BASE=ROOT/'validation/_jet173'
sys.path[:0]=[str(RUNTIME),str(ROOT),str(ROOT/'desktop'),str(ROOT/'validation/sediment/delft3d_current')]
from stage_wall172 import save,sha


def document(backend,end=120.):
    path=ROOT/f'validation/_delivery172/package/groin_mpm_{backend}/project.json'
    doc=json.loads(path.read_text())
    # Retain the same physical definitions, but keep morphology off throughout.
    doc['settings']['end_time_s']=end
    doc['settings']['sediment_start_time_s']=end
    doc['settings']['output_interval_s']=1.
    doc['settings']['checkpoint_interval_s']=end
    from braids_core.river_open_boundary import resolve
    doc['boundary_resolution']=resolve(doc['settings']['hydraulics'],doc['boundary_states'],
        dict(Upstream='inlet',Downstream='outlet'),sediment=doc['model']['physics']['sediment'],ng=1,end_time=end)
    return doc


def setup():
    from braids_core.river_launcher import verify
    from braids_core.sediment.portable_river_runtime import install
    verify();install(json.loads((RUNTIME/'PRECOMPILED.json').read_text()),RUNTIME)


def probe(backend):
    from braids_core import river_worker
    from braids_core.sediment.bed_columns_v101 import host
    from review_hydraulics162 import load
    setup()
    out=BASE/'probes'/backend;out.mkdir(parents=True,exist_ok=False)
    data,meta=load(ROOT/'validation/_delivery172/package/groin_mpm_cpu/full',ROOT/'validation/_hydraulics162/mobile/delft_mpm')
    z=data['bz'][0].copy();xy=data['xy'];cases={}
    for time in (20,30,35,40):
        j=int(np.flatnonzero(data['store'].times==time)[0])
        cases[f'BRAIDS{time}']=data['frames'][j].q.copy()
        cases[f'FLOW{time}']=np.vstack([data['nh'][j],data['nh'][j]*data['nu'][j]])
    b,n=cases['BRAIDS40'],cases['FLOW40']
    cases['BRAIDS40_FLOWdepth']=np.vstack([n[0],n[0]*b[1:]/b[0]])
    cases['FLOW40_BRAIDSdepth']=np.vstack([b[0],b[0]*n[1:]/n[0]])
    original=river_worker.core.factory
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);xp=owner.xp;c=owner.midpoint
            save(out/'OWNER.json',owner.identity)
            for label,q in cases.items():
                owner.z[:]=xp.asarray(z);c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
                rate,water=c.rate(c.interval_base,.0001,40.)
                arrays=dict(q=q,z=z,xy=xy,rate=host(rate).copy(),water=host(water).copy())
                arrays.update({k:owner.data[k] for k in ('owner','neighbour','normal','mid','length','area','bid')})
                if backend=='gpu':
                    arrays.update({k:host(v).copy() for k,v in owner.hydraulics.flux.buffers.items()})
                    arrays.update({'rec_'+k:host(v).copy() for k,v in owner.hydraulics.reconstruction.buffers.items()})
                np.savez(out/(label+'.npz'),**arrays)
            owner.z[:]=xp.asarray(z)
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    doc=document(backend);save(out/'project.json',doc)
    prepared=river_worker.prepare(out/'project.json',out/'prepared')
    status=river_worker.run(prepared,out/'construction_run',until=.001)
    save(out/'AUDIT.json',dict(status=status,runtime_sha256=sha(RUNTIME/'RUNTIME.json'),
        script_sha256=sha(__file__),inputs=meta,outputs={p.name:sha(p) for p in out.glob('*.npz')}))
    print('Probes complete',backend,flush=True)


def run_braids(backend):
    from braids_core import river_worker
    from braids_desktop.preparation import prepare
    setup();out=BASE/'fixed'/backend;out.mkdir(parents=True,exist_ok=False)
    doc=document(backend);save(out/'project.json',doc)
    prepare(out/'project.json',out/'prepared',asset_cache=BASE/'assets')
    status=river_worker.run(out/'prepared',out/'run')
    save(out/'CHECK.json',dict(status=status,runtime_sha256=sha(RUNTIME/'RUNTIME.json'),script_sha256=sha(__file__),
        changes=dict(end_time_s=120,sediment_start_time_s=120,output_interval_s=1,checkpoint_interval_s=120)))
    print(status,flush=True)


def run_native():
    from check_bedload161 import replace
    import runtime
    parent=ROOT/'validation/_hydraulics162/mobile/delft_mpm'
    out=BASE/'fixed/delft';out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((parent/'INPUTS.json').read_text())
    for name,digest in receipt['inputs'].items():
        assert sha(parent/name)==digest
        shutil.copy2(parent/name,out/name)
    changes=dict(Tstop='120',Flmap='0 1 120',Flhis='0 1 120',MomTrm='#Y#')
    p=out/'sim_P057.mdf';code=p.read_text()
    for key,value in changes.items():code=replace(code,key,value)
    p.write_text(code)
    p=out/'mor.mor';p.write_text(replace(p.read_text(),'MorStt','120'))
    # Extend only constant forcing table coverage, not its values.
    for name in ('flume.bct','bcm.bcm'):
        p=out/name;lines=p.read_text().splitlines()
        for i,line in enumerate(lines):
            tokens=line.split()
            if tokens and tokens[0]=='1':lines[i]='2 '+' '.join(tokens[1:])
        p.write_text('\n'.join(lines)+'\n')
    save(out/'INPUTS.json',dict(parent=str(parent),parent_manifest_sha256=sha(parent/'INPUTS.json'),
        changes=dict(mdf=changes,MorStt=120,constant_tables_extended_to_minutes=2),
        Delft3D_source_modified=False,Delft3D_runtime_modified=False,
        inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(runtime.run_case(out,configuration='config_flow2d3d.xml',timeout_s=180)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('probe','run','native'))
    ap.add_argument('--backend',choices=('cpu','gpu'),default='cpu');a=ap.parse_args()
    if a.action=='probe':probe(a.backend)
    elif a.action=='run':run_braids(a.backend)
    else:run_native()
