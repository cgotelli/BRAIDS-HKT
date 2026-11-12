"""Read-only interpretation of retained zero-viscosity groin histories."""
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from braids_desktop.results import ResultsStore
from stage_wall172 import save,sha
from review_hydraulics162 import verify_native
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_pulse177'
PARENT=ROOT/'validation/_corner174'


def main():
    store=ResultsStore(PARENT/'cpu/warmup')
    ids=np.flatnonzero((store.times>=100)&(store.times<=300));t=store.times[ids]
    frames=[store.read(int(i)) for i in ids]
    xy=np.array([p.centroid.coords[0] for p in frames[0].polygons])
    h=np.array([f.q[0] for f in frames]);uv=np.array([f.q[1:]/f.q[0] for f in frames])
    eta=h+np.array([f.z for f in frames]);x=np.unique(xy[:,0])
    b=dict(h=h,uv=uv,eta=eta)
    verify_native(PARENT/'delft')
    with Dataset(PARENT/'delft/trim-sim_P057.nc') as d:
        time=np.asarray(d['time'][:]);idx=np.searchsorted(time,t);assert np.array_equal(time[idx],t)
        ix,iy=np.rint(xy[:,0]/.5+.5).astype(int),np.rint(xy[:,1]/.5+.5).astype(int)
        e=np.asarray(d['S1'][idx])[:,ix,iy];z=-np.asarray(d['DPS'][idx])[:,ix,iy]
        u,v=[np.asarray(d[k][idx,0]) for k in ('U1','V1')]
        n=dict(eta=e,h=e-z,uv=np.stack([.5*(u+np.roll(u,1,1)),.5*(v+np.roll(v,1,2))],1)[:,:,ix,iy])
    result={};fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,(name,d) in enumerate((('BRAIDS',b),('FLOW Flood',n))):
        e=np.stack([d['eta'][:,xy[:,0]==xx].mean(1) for xx in x],1)
        fluct=e-e.mean(0)
        image=axes[row,0].pcolormesh(x,t,1000*fluct,shading='nearest',vmin=-40,vmax=40,cmap='RdBu_r')
        axes[row,0].set(title=name+': section-mean surface anomaly',xlabel='x (m)',ylabel='Time (s)')
        fig.colorbar(image,ax=axes[row,0],label='mm')
        result[name]={}
        for xx in (4.25,20.25,35.25):
            m=xy[:,0]==xx
            hh=d['h'][:,m].mean(1);uu=(d['h'][:,m]*d['uv'][:,0,m]).mean(1)/hh
            H=hh.mean();U=uu.mean();c=np.sqrt(9.81*H)
            p=.5*(hh-H+H/c*(uu-U));q=.5*(hh-H-H/c*(uu-U))
            result[name][str(xx)]=dict(mean_depth_m=float(H),mean_u_m_s=float(U),
                downstream_characteristic_rms_mm=float(np.std(p)*1000),
                upstream_characteristic_rms_mm=float(np.std(q)*1000),
                transverse_velocity_rms_m_s=float(np.sqrt(np.mean(d['uv'][:,1,m]**2))))
        stats=result[name]
        pos=np.arange(3)
        axes[row,1].bar(pos-.16,[stats[str(a)]['downstream_characteristic_rms_mm'] for a in (4.25,20.25,35.25)],.32,label='Downstream travelling')
        axes[row,1].bar(pos+.16,[stats[str(a)]['upstream_characteristic_rms_mm'] for a in (4.25,20.25,35.25)],.32,label='Upstream travelling')
        axes[row,1].set_xticks(pos,['4.25','20.25','35.25'])
        axes[row,1].set(title=name+': approximate wave decomposition',xlabel='Section x (m)',ylabel='Characteristic RMS (mm)',ylim=(0,25))
        axes[row,1].legend();axes[row,1].grid(axis='y',alpha=.2)
    fig.suptitle('Saved groin histories, 100–300 s: waves travel both ways; no new groin run')
    fig.savefig(OUT/'GROIN_WAVES.png',dpi=150);plt.close(fig)
    save(OUT/'GROIN_WAVES.json',dict(metrics=result,script_sha256=sha(__file__),
        source_manifest_sha256=sha(OUT.parent/'desktop_corner174/MANIFEST.json'),
        interpretation='Cross-section linearization about its temporal mean, descriptive only. '
        'The nonuniform two-dimensional groin is not a uniform linear wave experiment; '
        'these amplitudes do not isolate reflection coefficients or prove a feedback instability.'))
    print(result)


if __name__=='__main__':main()
