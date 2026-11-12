"""Read-only source-footprint and raw-profile evidence; no curve smoothing."""
from nearwake60 import *
import numpy as np
from types import SimpleNamespace
from scipy.integrate import quad
from scipy.interpolate import LinearNDInterpolator
import matplotlib;matplotlib.use('Agg')
from matplotlib import pyplot as plt
from braids_core.sediment.river_turbine_flow import CoveredRotor

def audit():
    REPORT.mkdir(exist_ok=True,parents=True)
    old=np.load(PARENT/'nash_gpu/TIME_AVERAGE.npz');xy=old['centers_xy'];speed=np.hypot(old['u'],old['v'])
    zero=np.load(PARENT/'zero_gpu/TIME_AVERAGE.npz');base=np.hypot(zero['u'],zero['v'])
    values=1-speed/base;records={};fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for x,col in [(520.,'#2377ae'),(540.,'#ce7e2c')]:
        mask=(abs(xy[:,0]-x)<=2.5001)&(abs(xy[:,1]-295)<30)
        y=np.unique(xy[mask,1]);d=np.array([values[mask&(xy[:,1]==v)].mean() for v in y])
        axes[0].plot((y-295)/10,100*d,'o-',color=col,label=f'{(x-500)/10:g}D, actual cells')
        records[f'{x:g}']=dict(y_D=((y-295)/10).tolist(),deficit=d.tolist(),central_pair_difference=float(abs(d[np.argmin(abs(y-292.5))]-d[np.argmin(abs(y-297.5))])))
    axes[0].set(xlabel='y / D',ylabel='Velocity deficit (%)',title='Two equal central CVs: no plotting cap');axes[0].legend(fontsize=8)
    c=dict(X=0.,Y=0.,D=10.,normal_xy=[1.,0.],name='audit',hub_elevation_m=9.,minimum_bed_clearance_m=0.,minimum_surface_clearance_m=0.,ct=.9031324031235483,cp=0.)
    beta=np.pi*10**2/4*c['ct']/(10*20);alpha=.5*(1+np.sqrt(1-beta));U=.2*alpha
    one=lambda s:np.exp(1-1/(1-s*s)) if abs(s)<1 else 0.
    exact=25*quad(lambda s:s*s*one(s),-1,1)[0]/quad(one,-1,1)[0]
    footprint={}
    for n,col in [(2,'#236da2'),(4,'#d17a28'),(8,'#388859')]:
        dx=10/n;cells=[]
        for y in np.arange(-10,10,dx):
            for x in np.arange(-10,10,dx):
                cells.append(SimpleNamespace(vertices_xy=[(x,y),(x+dx,y),(x+dx,y+dx),(x,y+dx)],centroid_xy=(x+dx/2,y+dx/2),area_m2=dx*dx))
        s=CoveredRotor(cells,c,'cpu');q=np.tile([20.,20*U,0.],(len(cells),1)).T.copy();z=np.zeros(len(cells));grad=np.zeros((10,len(cells)));rate=np.zeros_like(q)
        diag=s.apply(q,z,grad,rate)[0];w=s.normalized;pos=np.array([a.centroid_xy for a in cells]);var=float(np.sum(w*(pos[:,1]**2+dx*dx/12)))
        force=-rate[1]*s.area;profile=[]
        for y in np.unique(pos[:,1]):profile.append([y,float(force[pos[:,1]==y].sum()/dx)])
        p=np.array(profile);axes[1].step(p[:,0]/10,p[:,1],where='mid',label=f'D/{n}',color=col)
        footprint[str(n)]=dict(force_per_density=diag['force_per_density_m4_s2'],continuous_variance_m2=exact,
            represented_variance_m2=var,variance_relative_error=var/exact-1,coefficient_sum=float(np.sum(w)),
            center_of_force_m=(pos.T@force/force.sum()).tolist())
    axes[1].set(xlabel='y / D',ylabel='Thrust / density / transverse metre',title='Actual CV-averaged transverse force');axes[1].legend()
    axes[2].bar(['D/2','D/4','D/8'],[footprint[str(n)]['represented_variance_m2'] for n in (2,4,8)])
    axes[2].axhline(exact,color='k',ls='--',label='Continuous bump');axes[2].set(ylabel='Transverse force variance (m²)',title='Area integration conserves force, not shape');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(REPORT/'FOOTPRINT_AND_PLATEAU.png',dpi=170);plt.close(fig)
    save(REPORT/'OPERATOR_AUDIT.json',dict(raw_profiles=records,footprints=footprint,
        unchanged_integrated_thrust=max(f['force_per_density'] for f in footprint.values())-min(f['force_per_density'] for f in footprint.values()),
        classification='The two-cell force is uniform over its support, despite exact quadrature of a smooth bump. Raw symmetric centre cells plus linear interpolation create a plateau, not a value clamp.'))
    print(json.dumps(footprint,indent=2))
if __name__=='__main__':audit()
