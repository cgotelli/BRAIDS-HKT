"""Complete Desktop wall-pressure qualification; isolated candidate, no release edit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
os.environ['QT_QPA_PLATFORM']='offscreen'
from check_bedload161 import save,sha


def document(folder,backend,angle,case):
    from osgeo import gdal,osr
    from braids_desktop import services
    theta=np.deg2rad(angle);t=np.array([np.cos(theta),np.sin(theta)]);n=np.array([-t[1],t[0]])
    origin=np.array([6.13,5.17]);corners=np.array([[-3,0],[3,0],[3,2],[-3,2]])
    vertices=origin+corners[:,0,None]*t+corners[:,1,None]*n
    folder.mkdir(parents=True)
    dem=folder/'terrain.tif'
    raster=gdal.GetDriverByName('GTiff').Create(str(dem),112,112,1,gdal.GDT_Float64)
    raster.SetGeoTransform((0.,.125,0.,14.,0.,-.125))
    srs=osr.SpatialReference();srs.ImportFromEPSG(2056);raster.SetProjection(srs.ExportToWkt())
    xx,yy=np.meshgrid((np.arange(112)+.5)*.125,14-(np.arange(112)+.5)*.125)
    along=(xx-origin[0])*t[0]+(yy-origin[1])*t[1]
    bed=.1*(along+3) if case=='wetting' else np.zeros_like(xx)
    raster.GetRasterBand(1).WriteArray(bed);raster=None
    doc=services.default_document(dem,vertices.tolist(),'EPSG:2056',backend=backend,
        minimum_size_m=.5,levels=1,end_time_s=4.,name=f'Wall qualification {angle} {case} {backend}')
    doc['initial'].update(stage_m=.3 if case=='wetting' else .5,velocity_m_s=(.8*t).tolist())
    doc['settings'].update(dt_max_s=.05,output_interval_s=.25,checkpoint_interval_s=4.)
    doc['settings']['controls'].update(cfl=1.,threads=4)
    services.save_project_folder(folder/'project',doc)
    return t,n,origin


def probes(owner,out,t,n):
    from braids_core.sediment.bed_columns_v101 import host
    from wall_pressure170 import analytic_pressure
    xp=owner.xp;c=owner.midpoint;d=owner.data;z0=owner.z.copy()
    nc=len(d['area']);wall=d['neighbour']<0
    assert np.all(d['bid'][wall]<0)
    rows=[]
    for depth in (0.,1e-5,.005,.5):
        for fr in (0.,-.75,.75):
            if depth==0 and fr!=0:continue
            uv=np.sqrt(9.81*depth)*(fr*t+.2*n) if fr else np.zeros(2)
            q=np.vstack([np.full(nc,depth),np.broadcast_to((depth*uv)[:,None],(2,nc))])
            owner.z[:]=0.;c.interval_base=xp.asarray(q).copy();c.minimum_theta=1.;c.frequency_peak=0.
            rate,water=c.rate(c.interval_base,1e-9,0.)
            data=dict(q=q,rate=host(rate).copy(),water=host(water).copy(),
                area=d['area'],normal=d['normal'],owner=d['owner'],neighbour=d['neighbour'],length=d['length'])
            r=dict(depth=depth,fr=fr,wall_water_max=float(abs(data['water'][wall]).max()),
                volume_rate=float(np.dot(data['rate'][0],d['area'])),
                rest_rate_max=float(abs(data['rate']).max()) if fr==0 else None)
            if owner.backend=='gpu':
                b=owner.hydraulics.flux.buffers;data.update({k:host(b[k]).copy() for k in ('local','fl','fr','closed')})
                h=data['local'][0,wall];momentum=data['local'][1,wall]
                expected=np.array([analytic_pressure(a,b) for a,b in zip(h,momentum)])
                pressure=np.sum(data['fl'][1:,wall]*d['normal'][:,wall],axis=0)
                r['pressure_error']=float(np.max(abs(pressure-expected)))
                r['wall_tangential_flux']=float(np.max(abs(data['fl'][1:,wall]-expected*d['normal'][:,wall])))
                r['minimum_pressure']=float(pressure.min())
                assert r['pressure_error']<1e-11 and r['wall_tangential_flux']<1e-11,r
                assert r['minimum_pressure']>=-1e-14,r
            assert np.isfinite(data['rate']).all() and r['wall_water_max']==0
            assert abs(r['volume_rate'])<1e-10,r
            if fr==0:assert r['rest_rate_max']<1e-10,r
            name=f'depth{depth:g}_fr{fr:g}'
            np.savez(out/(name+'.npz'),**data);rows.append(dict(name=name,**r))
    owner.z[:]=z0
    save(out/'PROBES.json',rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cpu','gpu'],required=True)
    ap.add_argument('--angle',type=int,required=True);ap.add_argument('--case',choices=['reflection','wetting','groin'],required=True)
    ap.add_argument('--tag',default='v1');ap.add_argument('--restart',action='store_true');a=ap.parse_args()
    from braids_core.sediment.portable_river_runtime import install
    from braids_core import river_worker
    from wall_pressure170 import install as candidate,METHOD
    from braids_desktop.preparation import prepare
    from braids_desktop.results import ResultsStore
    from braids_core.sediment.bed_columns_v101 import host
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime');candidate(a.backend)
    out=ROOT/f'validation/_wall171/{a.tag}/{a.case}_{a.angle}_{a.backend}'
    out.mkdir(parents=True,exist_ok=False)
    for p in (Path(__file__),ROOT/'desktop/wall_pressure170.py'):shutil.copy2(p,out/p.name)
    if a.case=='groin':
        doc=json.loads((ROOT/f'validation/_shoal165/groin/mpm_{a.backend}/project.json').read_text())
        doc['settings'].update(end_time_s=44.,output_interval_s=1.,checkpoint_interval_s=44.)
        save(out/'project.json',doc);prepared=river_worker.prepare(out/'project.json',out/'prepared')
    else:
        t,n,origin=document(out/'inputs',a.backend,a.angle,a.case)
        prepared=out/'prepared';prepare(out/'inputs/project/braids_project.json',prepared,asset_cache=out.parent/'asset_cache')
    original=river_worker.core.factory;phase=['probe' if a.case=='reflection' else 'full'];owners=[]
    def factory(*args,**kw):
        make=original(*args,**kw)
        def build(*args,**kw):
            owner=make(*args,**kw);owners.append(owner)
            identity=dict(method=METHOD,script_sha256=sha(ROOT/'desktop/wall_pressure170.py'))
            if a.backend=='gpu':identity['cuda_sha256']=owner.identity['activity_generated_cuda_sha256']
            else:identity['library_sha256']=sha(Path(owner.hydraulics.lib._name))
            owner.identity['isolated_wall_pressure170']=identity
            save(out/'CANDIDATE.json',identity)
            d=owner.data
            if phase[0]=='probe':
                folder=out/'probes';folder.mkdir(exist_ok=False);probes(owner,folder,t,n)
            np.savez(out/'GEOMETRY.npz',xy=d['xy'],area=d['area'],owner=d['owner'],neighbour=d['neighbour'],normal=d['normal'],length=d['length'])
            assert d['area'].min()>=.25-1e-9
            return owner
        build.__dict__.update(make.__dict__);return build
    river_worker.core.factory=factory
    if phase[0]=='probe':
        river_worker.run(prepared,out/'probe_run',until=.001);phase[0]='full'
    result=river_worker.run(prepared,out/'full',checkpoint_dir=out/'checkpoints/full')
    assert result['status']=='complete'
    checks={}
    if a.restart:
        phase[0]='cancel';stop=out/'STOP'
        split=42. if a.case=='groin' else 2.
        def cancel(folder,row):
            if row['time_s']>=split:stop.touch()
        part=river_worker.run(prepared,out/'cancelled',stop_file=stop,on_frame=cancel,checkpoint_dir=out/'checkpoints/cancelled')
        assert part['status']=='cancelled' and part['time_s']==split,part
        phase[0]='resume'
        resumed=river_worker.run(prepared,out/'resumed',restart=out/'cancelled/checkpoint.npz')
        assert resumed['status']=='complete'
        x=ResultsStore(out/'full').read(-1);y=ResultsStore(out/'resumed').read(-1)
        fields={k:bool(np.array_equal(x.arrays[k],y.arrays[k])) for k in x.arrays if k in y.arrays and x.arrays[k].dtype.kind not in 'OUS'}
        for k in ('q','z','mass','water','bed_boundary'):assert fields[k],(k,a.backend,a.case)
        # Corrupted outer payload must be rejected before creation of an output.
        with np.load(out/'cancelled/checkpoint.npz') as f:identity=f['identity'].copy();payload=f['payload'].copy()
        payload[len(payload)//2]^=1
        np.savez(out/'corrupt_checkpoint.npz',identity=identity,payload=payload)
        try:river_worker.run(prepared,out/'must_not_exist',restart=out/'corrupt_checkpoint.npz')
        except ValueError as exc:corruption=str(exc)
        else:raise AssertionError('Corrupt checkpoint accepted')
        assert not (out/'must_not_exist').exists()
        checks=dict(cancel=part,resumed=resumed,exact_fields=fields,corruption_rejected=corruption)
    save(out/'AUDIT.json',dict(backend=a.backend,angle=a.angle,case=a.case,status=result,restart=checks,
        script_sha256=sha(__file__),candidate_sha256=sha(ROOT/'desktop/wall_pressure170.py')))
    print(json.dumps(dict(case=a.case,angle=a.angle,backend=a.backend,status=result['status'],restart=a.restart)),flush=True)


if __name__=='__main__':main()
