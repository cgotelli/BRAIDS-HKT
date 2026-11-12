"""Directional same-state momentum accounting at the leading groin corner."""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop'),str(ROOT)]
from stage_wall172 import save,sha
from review_jet173 import OUT,G,ledger


def main():
    a=np.load(ROOT/'validation/_jet173/probes/gpu/FLOW40.npz')
    rows=[]
    with Dataset(ROOT/'validation/_corner167/native_momentum_fine/trim-sim_P057.nc') as f:
        j=int(np.argmin(abs(np.asarray(f['time'][:])-40)))
        for point in ([4.25,3.25],[7.75,3.25],[8.25,3.25],[8.25,3.75],[8.25,4.25],[8.75,3.25]):
            c=int(np.flatnonzero(np.all(a['xy']==point,axis=1))[0]);h=a['q'][0,c];v=a['q'][2,c]/h
            ix,iy=np.rint(np.asarray(point)/.5+.5).astype(int)
            native={}
            for key,name in [('pressure','MOM_VPRESSURE'),('x_advection','MOM_UDVDX'),('y_advection','MOM_VDVDY'),('drag','MOM_VBEDSHEAR'),('total','MOM_DVDT')]:
                field=np.asarray(f[name][j]).reshape(82,18)
                native[key]=float(.5*(field[ix,iy]+field[ix,iy-1]))
            face_rows=[];rates=np.zeros((2,3));advection=0.
            for k in np.flatnonzero((a['owner']==c)|(a['neighbour']==c)):
                sign=1 if a['owner'][k]==c else -1
                flux=a['fl' if sign==1 else 'fr'][:,k]
                r=-sign*a['length'][k]/a['area'][c]*flux
                axis=int(abs(a['normal'][1,k])>.5);rates[axis]+=r
                acc=float((r[2]-v*r[0])/h)
                face_rows.append(dict(mid=a['mid'][:,k].tolist(),normal=a['normal'][:,k].tolist(),
                                      outward_water_flux=float(sign*flux[0]),y_momentum_flux=float(sign*flux[2]),
                                      transverse_velocity_rate=acc,local=a['local'][:,k].tolist()))
                if axis==0:advection+=acc
            b=ledger(a,c)
            rows.append(dict(xy=point,depth=h,velocity_v=v,braids=b,native=native,faces=face_rows,
                             x_transport_acceleration=advection,
                             x_transport_minus_native_x_advection=advection-native['x_advection']))
    save(OUT/'CORNER_TERMS.json',dict(same_input='FLOW depth and primary face-averaged velocity at 40 s',
        scope='Comparison of discrete operators; the projected native field is not a BRAIDS equilibrium',
        rows=rows,source_sha256=sha(__file__)))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels=['x transport','y transport / pressure','resistance','total'];x=np.arange(4)
    for ax,point in zip(axs,([4.25,3.25],[8.25,3.25])):
        r=next(s for s in rows if s['xy']==point);b=r['braids'];n=r['native']
        bv=[b['axis_total'][0][1],b['axis_total'][1][1],b['drag'][1],b['total'][1]]
        nv=[n['x_advection'],n['pressure']+n['y_advection'],n['drag'],n['total']]
        ax.bar(x-.18,bv,.36,label='BRAIDS on FLOW state');ax.bar(x+.18,nv,.36,label='FLOW mapped diagnostic')
        ax.set_xticks(x,labels,rotation=18);ax.set(title=f'Cell at {point} m',ylabel='Transverse acceleration (m/s²)')
        ax.axhline(0,color='gray',lw=.7);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Remaining corner difference is primarily downstream transport of transverse momentum\nMapped native velocity diagnostics are not an exported native conservative flux ledger')
    fig.savefig(OUT/'CORNER_TRANSPORT.png',dpi=155);plt.close(fig)
    print(json.dumps([dict(xy=r['xy'],x_gap=r['x_transport_minus_native_x_advection'],
                         total_gap=r['braids']['total'][1]-r['native']['total']) for r in rows],indent=2))


if __name__=='__main__':main()
