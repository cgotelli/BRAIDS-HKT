"""Offline face-divergence attribution, not an alternative transport method.

Replay FLOW's sign-upwind/no-lateral-average rule on saved cell capacities.
Compare it with the actually executed BRAIDS physical and wave-corrected
fluxes. No time integration, tuning, or changes to the native reference.
"""
import json
from pathlib import Path
import sys
import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'desktop/runtime'),str(ROOT),str(ROOT/'desktop')]
from check_bedload161 import save,sha

BASE=ROOT/'validation/_groin163'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_groin163'
NATIVE=ROOT/'validation/_hydraulics162/mobile/delft_mpm/trim-sim_P057.nc'


def upwind(cap,a):
    o=a['owner'];j=a['neighbour'];normal=a['normal']
    left=np.sum(cap[:,o]*normal,axis=0)
    right=np.sum(cap[:,np.maximum(j,0)]*normal,axis=0)
    flux=np.where((left>0)&(right>0),left,
                  np.where((left<0)&(right<0),right,.5*(left+right)))
    flux[j<0]=0 # Interior audit only: no attempt to recreate open BCs.
    return flux


def main():
    records={}
    for sec in (10,30,40):
        a=np.load(BASE/f'cpu/FLOW_CHARACTERISTIC_{sec:04d}.npz')
        xy=a['xy'];mid=a['mid'];normal=a['normal']
        ix=np.rint(xy[:,0]/.5+.5).astype(int)
        iy=np.rint(xy[:,1]/.5+.5).astype(int)
        vertical=abs(normal[0])>.5
        mx=np.rint(mid[0]/.5+np.where(vertical,0,.5)).astype(int)
        my=np.rint(mid[1]/.5+np.where(vertical,.5,0)).astype(int)
        with Dataset(NATIVE) as ds:
            cap=np.array([ds[k][sec//5,0] for k in ('SBCU','SBCV')])[:,ix,iy]
            nflux=np.where(vertical,np.asarray(ds['SBUU'][sec//5,0])[mx,my]*normal[0],
                                   np.asarray(ds['SBVV'][sec//5,0])[mx,my]*normal[1])
        nearby=(mid[0]>=6)&(mid[0]<=13)&(mid[1]>=2)&(mid[1]<=6)&(a['neighbour']>=0)
        control=upwind(cap,a)
        error=float(abs(control[nearby]-nflux[nearby]).max())
        # This validates the native face rule against its actual output,
        # including the no-lateral-averaging setting in this input deck.
        assert error<2e-9,error
        cases={'FLOW actual':nflux,'FLOW capacity / replayed upwind':control}
        capacities={}
        for label in ('BRAIDS','FLOW_PRIMARY','FLOW_CHARACTERISTIC'):
            b=np.load(BASE/f'cpu/{label}_{sec:04d}.npz')
            bc=b['bedcell'][3:5]/2650
            physical=b['wave_physical_normal']/2650
            physical=physical.copy();physical[b['neighbour']<0]=0
            np.testing.assert_allclose(
                b['class_flux'][0,nearby]-physical[nearby],
                b['wave_diagnostic'][1,nearby]/2650,atol=2e-16,rtol=2e-12)
            cases[label+' / sign-upwind']=upwind(bc,b)
            cases[label+' / physical MUSCL']=physical
            cases[label+' / actual wave flux']=b['class_flux'][0]
            capacities[label]=bc
        selected=[]
        for x in (7.75,8.25,8.75,9.25,9.75):
            i=int(np.argmin(((xy-[x,3.25])**2).sum(axis=1)))
            fs=np.where((a['owner']==i)|(a['neighbour']==i))[0]
            sign=np.where(a['owner'][fs]==i,1.,-1.)
            rates={name:(-sign*flux[fs]*a['length'][fs]/a['area'][i]/.6*1000)
                   for name,flux in cases.items()}
            selected.append(dict(xy=xy[i].tolist(),face_midpoints=mid[:,fs].T.tolist(),
                bed_rate_mm_s={name:float(v.sum()) for name,v in rates.items()},
                per_face_bed_rate_mm_s={name:v.tolist() for name,v in rates.items()},
                cell_capacity_m2_s={'FLOW native':cap[:,i].tolist(),
                    **{name:v[:,i].tolist() for name,v in capacities.items()}}))
        records[str(sec)]=dict(native_face_replay_max_m2_s=error,cells=selected)
    save(OUT/'FACE_CHAIN.json',dict(native_output_sha256=sha(NATIVE),script_sha256=sha(__file__),
        units='instantaneous bed elevation rate mm/s; positive deposition',
        scope='Offline face-assembly contrasts, same initial planar bed and finite stock; not coupled runs or proposed defaults. FLOW velocity substitution does not also impose identical native shear/roughness inputs.',
        records=records))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    cells=records['40']['cells'];x=[r['xy'][0] for r in cells]
    for label,color in [('FLOW actual','#111111'),('BRAIDS / actual wave flux','#2563eb'),
                        ('FLOW_PRIMARY / actual wave flux','#16a34a'),
                        ('FLOW_CHARACTERISTIC / actual wave flux','#b45309')]:
        axes[0].plot(x,[r['bed_rate_mm_s'][label] for r in cells],'o-',color=color,label=label)
    for label,color in [('FLOW actual','#111111'),('FLOW_CHARACTERISTIC / sign-upwind','#9333ea'),
                        ('FLOW_CHARACTERISTIC / physical MUSCL','#b45309'),
                        ('FLOW_CHARACTERISTIC / actual wave flux','#2563eb')]:
        axes[1].plot(x,[r['bed_rate_mm_s'][label] for r in cells],'o-',color=color,label=label)
    axes[0].set_title('Same BRAIDS face operator; different hydraulic inputs')
    axes[1].set_title('Same FLOW-characteristic input; face assembly contrasts')
    for ax in axes:
        ax.axhline(0,color='k',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set(xlabel='x (m), y = 3.25 m',ylabel='Instantaneous bed rate (mm/s), t = 40 s')
    fig.savefig(OUT/'FACE_CHAIN.png',dpi=170);plt.close(fig)
    print(json.dumps(records['40'],indent=2))


if __name__=='__main__':main()
