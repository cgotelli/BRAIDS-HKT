"""Read-only endpoint reconstruction of FLOW's groin V-momentum stencil.

This is analysis, not a modified FLOW engine or a production flux. Map values
do not expose the intermediate ADI states; retain the mismatch with native
momentum diagnostics instead of claiming an exact internal replay.
"""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop')]
from stage_wall172 import save,sha
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_jet173'


def increment(a,b):
    """The limited increment ulim(a,b)*a, not the unrelated slim routine."""
    return .5*np.sign(a)*min(abs(a),abs(b)) if a*b>0 else 0.


def v_stencil(u,v,s,dps,dpu,dpv,ku,kv,i,j,dx=.5):
    # upwhu + Flood: this locked slim.f90 returns zero. All diagnostic cells
    # here are fully wet, single-layer, without weirs or thin dams.
    hu=np.where(u>.001,s,np.where(u<-.001,np.roll(s,-1,0),np.maximum(s,np.roll(s,-1,0))))+dpu
    hv=np.where(v>.001,s,np.where(v<-.001,np.roll(s,-1,1),np.maximum(s,np.roll(s,-1,1))))+dpv
    qu=hu*u*dx*ku;qv=hv*v*dx*kv
    def ub(a,b):
        if not ku[a,b]*ku[a,b+1]:return 0.
        if qu[a,b]+qu[a,b+1]>0:
            d1=(v[a,b]-v[a-1,b])*kv[a,b]*kv[a-1,b]
            d2=(v[a+1,b]-v[a,b])*kv[a+1,b]*kv[a,b]
            return v[a,b]+increment(d1,d2)
        d1=(v[a+2,b]-v[a+1,b])*kv[a+2,b]*kv[a+1,b]
        d2=(v[a+1,b]-v[a,b])*kv[a+1,b]*kv[a,b]
        return v[a+1,b]-increment(d1,d2)
    def ua(a,b):
        d2=(v[a,b]-v[a,b-1])*kv[a,b]*kv[a,b-1]
        if qv[a,b]+qv[a,b-1]>0:
            d1=(v[a,b-1]-v[a,b-2])*kv[a,b-1]*kv[a,b-2]
            return v[a,b-1]+increment(d1,d2)
        d1=(v[a,b+1]-v[a,b])*kv[a,b+1]*kv[a,b]
        return v[a,b]-increment(d1,d2)
    qe=(qu[i,j]+qu[i,j+1])/max(1,ku[i,j]+ku[i,j+1])
    qw=(qu[i-1,j]+qu[i-1,j+1])/max(1,ku[i-1,j]+ku[i-1,j+1])
    qn=(qv[i,j]+qv[i,j+1])/max(1,kv[i,j]+kv[i,j+1])
    qs=(qv[i,j-1]+qv[i,j])/max(1,kv[i,j-1]+kv[i,j])
    depth=.5*(dps[i,j]+s[i,j]+dps[i,j+1]+s[i,j+1])
    be,bw=ub(i,j),ub(i-1,j);an,ass=ua(i,j+1),ua(i,j)
    # Original source's factors are 1 on this fixed, smooth longitudinal bed
    # in the transverse equation; the local transverse bed gradient is zero.
    assert dps[i,j]==dps[i,j+1]
    xraw=-(qe*(be-v[i,j])-qw*(bw-v[i,j]))/(dx*dx*depth)
    yraw=-(qn*(an-v[i,j])-qs*(ass-v[i,j]))/(dx*dx*depth)
    return dict(v=float(v[i,j]),depth=float(depth),
                q_east=float(qe),q_west=float(qw),
                transported_v_east=float(be),transported_v_west=float(bw),
                UA_north=float(an),UA_south=float(ass),
                x_acceleration=float(xraw if be*bw!=0 else 0),
                y_acceleration=float(yraw if an*ass!=0 else 0),
                unguarded_y_acceleration=float(yraw),
                self_advection_suppressed=bool(an*ass==0),
                west_v_stencil=v[i-2:i+2,j].tolist(),
                west_v_mask=kv[i-2:i+2,j].tolist())


def main():
    path=ROOT/'validation/_corner167/native_momentum_fine/trim-sim_P057.nc'
    rows=[]
    with Dataset(path) as f:
        t=int(np.argmin(abs(np.asarray(f['time'][:])-40)))
        u,v=[np.asarray(f[k][t,0],dtype=float) for k in ('U1','V1')]
        s,dps,ku,kv=[np.asarray(f[k][t],dtype=float) for k in ('S1','DPS','KFU','KFV')]
        dpu,dpv=[np.asarray(f[k][:],dtype=float) for k in ('DPU0','DPV0')]
        for x,y in ([4.25,3.5],[7.75,3.5],[8.25,3.5],[8.75,3.5]):
            i=int(round(x/.5+.5));j=int(round(y/.5))
            r=v_stencil(u,v,s,dps,dpu,dpv,ku,kv,i,j)
            r.update(x=x,y=y,native_x=float(np.asarray(f['MOM_UDVDX'][t]).reshape(82,18)[i,j]),
                     native_y=float(np.asarray(f['MOM_VDVDY'][t]).reshape(82,18)[i,j]))
            r['endpoint_replay_minus_native_x']=r['x_acceleration']-r['native_x']
            r['endpoint_replay_minus_native_y']=r['y_acceleration']-r['native_y']
            rows.append(r)
    source=ROOT/'resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/flow2d3d_kernel/src/compute'
    save(OUT/'NATIVE_STENCIL.json',dict(rows=rows,
        scope='Read-only source-informed endpoint reconstruction; not internal ADI state replay',
        input_sha256=sha(path),analysis_sha256=sha(__file__),
        native_sources_read_only={str((source/n).relative_to(ROOT)):sha(source/n)
            for n in ('mom_fls.f90','upwhu.f90','slim.f90','ulim.f90','uzd.f90')}))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
