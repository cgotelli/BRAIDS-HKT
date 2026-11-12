"""Replay identical archived states through CPU/CUDA spatial operators.

No state is fed back to a coupled trajectory. Native staggered face values
remain a separate diagnostic from the centre-state projection used here.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import sha,save
from braids_core.sediment.bed_columns_v101 import host
from braids_core.sediment.portable_river_runtime import install
from braids_core import river_worker
from braids_desktop.results import ResultsStore


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','gpu'],required=True)
    args=p.parse_args();base=ROOT/'validation/_bedload161_matched'
    out=ROOT/'validation/_hydraulics162/operator_v3'/args.backend
    out.mkdir(parents=True,exist_ok=False)
    install(json.loads((ROOT/'desktop/runtime/PRECOMPILED.json').read_text()),ROOT/'desktop/runtime')
    store=ResultsStore(base/'braids_mpm_cpu/run')
    xy=np.array([p.centroid.coords[0] for p in store.read(0).polygons])
    ix=np.rint(xy[:,0]/.5+.5).astype(int);iy=np.rint(xy[:,1]/.5+.5).astype(int)
    original=river_worker.core.factory;rows=[]
    def factory(*a,**kw):
        make=original(*a,**kw)
        def build(*a,**kw):
            owner=make(*a,**kw);xp=owner.xp;e=owner.exterior_bed;controller=owner.midpoint
            with Dataset(base/'delft_mpm_v8/trim-sim_P057.nc') as ds:
                for j in (1,2,4,6,8):
                    for source in ('BRAIDS','FLOW'):
                        frame=store.read(j)
                        if source=='BRAIDS':q=frame.q.copy()
                        else:
                            h=np.asarray(ds['S1'][j]+ds['DPS'][j])[ix,iy]
                            q=np.array([h,h*np.asarray(ds['UUU'][j])[ix,iy],h*np.asarray(ds['VVV'][j])[ix,iy]])
                        before=host(owner.q).copy()
                        controller.interval_base=xp.asarray(q).copy()
                        controller.minimum_theta=1.
                        controller.frequency_peak=0.
                        rate,water=controller.rate(xp.asarray(q),.0001,frame.time_s)
                        assert np.array_equal(host(owner.q),before)
                        d=owner.data;extra={}
                        if args.backend=='gpu':
                            extra={k:host(owner.hydraulics.flux.buffers[k]).copy() for k in ('fl','fr')}
                        file=out/f'{source}_{frame.time_s:04.0f}.npz'
                        np.savez(file,q=q,z=frame.z,xy=xy,rate=host(rate),water=host(water),
                            exterior_flux=host(e.flux),exterior_trace=host(e.trace),
                            **{k:d[k] for k in ('owner','neighbour','normal','length','mid','bid','area')},**extra)
                        mask=(d['neighbour']<0)&(d['bid']>=0)
                        wall=(d['neighbour']<0)&(d['bid']<0)
                        residual=float(np.dot(host(rate)[0],d['area'])+np.dot(host(water)[mask],d['length'][mask]))
                        rows.append(dict(source=source,time_s=frame.time_s,file=file.name,sha256=sha(file),
                            water_rate_balance_m3_s=residual,max_wall_water_m2_s=float(abs(host(water)[wall]).max())))
            for seconds in (10,30):
                native=ROOT/'validation/_hydraulics162/native_replay'/str(seconds)/'trim-sim_P057.nc'
                with Dataset(native) as ds:
                    def primary(j):
                        hh=np.asarray(ds['S1'][j]+ds['DPS'][j])[ix,iy]
                        uu=np.asarray(ds['U1'][j,0]);vv=np.asarray(ds['V1'][j,0])
                        uv=np.array([.5*(uu+np.roll(uu,1,axis=0)),.5*(vv+np.roll(vv,1,axis=1))])[:,ix,iy]
                        return np.vstack([hh,hh*uv])
                    initial=primary(0);native_final=primary(-1)
                    q=xp.asarray(initial).copy()
                    for step in range(10):q,_,_=controller._single(q,.001,seconds+step*.001)
                    file=out/f'STEP_FLOW_{seconds:04d}.npz'
                    np.savez(file,initial=initial,braids_final=host(q),native_final=native_final,xy=xy,
                        native_times=np.asarray(ds['time'][:]))
                    rows.append(dict(source='FLOW primary face-interpolated velocity',time_s=seconds,
                        window_s=.01,steps=10,file=file.name,sha256=sha(file)))
            return owner
        build.__dict__.update(make.__dict__)
        return build
    river_worker.core.factory=factory
    try:
        # Fresh, separate micro-run only constructs the authenticated public owner.
        status=river_worker.run(base/('braids_mpm_'+args.backend)/'prepared',out/'construction_run',until=.001)
        save(out/'AUDIT.json',dict(backend=args.backend,rows=rows,status=status,
            runtime_manifest_sha256=sha(ROOT/'desktop/runtime/RUNTIME.json'),script_sha256=sha(__file__)))
    finally:river_worker.core.factory=original


if __name__=='__main__':main()
