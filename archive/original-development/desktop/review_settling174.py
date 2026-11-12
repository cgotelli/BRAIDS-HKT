"""Predeclared hydraulic settling gate for the delayed-bedload comparison."""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from stage_wall172 import save,sha
from braids_desktop.results import ResultsStore
from review_hydraulics162 import verify_native
BASE=ROOT/'validation/_corner174'
LIMITS=dict(eta_temporal_rms_m=.002,eta_cell_std_max_m=.005,
            eta_mean_drift_rms_m=.002,eta_mean_drift_max_m=.005,
            velocity_mean_drift_rms_m_s=.01,velocity_mean_drift_max_m_s=.05,
            storage_rate_fraction_of_inflow=.01)


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))


def gate(t,h,z,uv,end,area,Q=4.):
    before=(t>=end-40)&(t<end-20);after=(t>=end-20)&(t<=end)
    assert before.sum()>=20 and after.sum()>=20
    eta=h+z;demean=eta[after]-eta[after].mean(0)
    drift=eta[after].mean(0)-eta[before].mean(0)
    vd=uv[after].mean(0)-uv[before].mean(0)
    indices=np.flatnonzero(after)
    values=dict(eta_temporal_rms_m=rms(demean),eta_cell_std_max_m=float(np.std(eta[after],axis=0).max()),
        eta_mean_drift_rms_m=rms(drift),eta_mean_drift_max_m=float(abs(drift).max()),
        velocity_mean_drift_rms_m_s=rms(vd),velocity_mean_drift_max_m_s=float(np.linalg.norm(vd,axis=0).max()),
        storage_rate_fraction_of_inflow=float(abs(np.dot(h[indices[-1]]-h[indices[0]],area))/(t[indices[-1]]-t[indices[0]])/Q))
    passed={k:v<=LIMITS[k] for k,v in values.items()}
    return dict(end_s=end,values=values,criteria_pass=passed,passed=all(passed.values()))


def load():
    store=ResultsStore(BASE/'cpu/warmup');frames=[store.read(i) for i in range(len(store.times))]
    xy=np.array([p.centroid.coords[0] for p in frames[0].polygons]);area=frames[0].arrays['area']
    ix,iy=np.rint(xy[:,0]/.5+.5).astype(int),np.rint(xy[:,1]/.5+.5).astype(int)
    b=dict(t=store.times,h=np.array([f.q[0] for f in frames]),z=np.array([f.z for f in frames]),
           uv=np.array([f.q[1:]/f.q[0] for f in frames]))
    with Dataset(BASE/'delft/trim-sim_P057.nc') as d:
        keep=np.asarray(d['time'][:])<=300;z=-np.asarray(d['DPS'][:])[keep][:,ix,iy]
        u,v=[np.asarray(d[k][:])[keep,0] for k in ('U1','V1')]
        uv=np.stack([.5*(u+np.roll(u,1,1)),.5*(v+np.roll(v,1,2))],axis=1)[:,:,ix,iy]
        n=dict(t=np.asarray(d['time'][:])[keep],z=z,h=np.asarray(d['S1'][:])[keep][:,ix,iy]-z,uv=uv)
    assert np.array_equal(b['t'],n['t'])
    assert abs(b['z']-b['z'][0]).max()==abs(n['z']-n['z'][0]).max()==0
    assert abs(b['z'][0]-n['z'][0]).max()<1e-12
    assert abs(b['h'][0]-n['h'][0]).max()<3e-8
    return b,n,xy,area


def main():
    verify_native(BASE/'delft');b,n,xy,area=load()
    results={name:[gate(**data,end=t,area=area) for t in (40,80,120,160,200,240,280,300)]
             for name,data in [('BRAIDS',b),('FLOW',n)]}
    # Require two consecutive final windows, not one convenient snapshot.
    both=all(r['passed'] for rows in results.values() for r in rows[-2:])
    save(BASE/'SETTLING.json',dict(limits=LIMITS,results=results,both_pass=both,
        activation_s=300,consecutive_final_window_ends_s=[280,300],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (BASE/'cpu/project.json',BASE/'delft/INPUTS.json')},
        script_sha256=sha(__file__)))
    print(json.dumps(dict(both_pass=both,last={k:v[-2:] for k,v in results.items()}),indent=2))


if __name__=='__main__':main()
