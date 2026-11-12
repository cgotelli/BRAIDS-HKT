"""Show the conserved velocity field, not an analytical wake overlay."""
from coupled_wake76 import ROOT,TARGET,REPORT,read,save,environment
environment()
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from braids_desktop.results import ResultsStore
from scipy.spatial import cKDTree

HOME=TARGET.parent
def main():
    summary=[];fig,axes=plt.subplots(3,2,figsize=(14,9),layout='constrained')
    power,paxes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for j,case in enumerate(['single','tandem','staggered']):
        root=HOME/(case+'_gpu');doc=read(root/'input.json');done=read(root/'DONE.json')
        f=ResultsStore(root/'run').read(-1);ambient=ResultsStore(ROOT/'desktop/qualification/power_planning75'/(case+'_cpu')/'run').read(-1)
        xy=np.array([[p.centroid.x,p.centroid.y] for p in f.polygons]);axy=np.array([[p.centroid.x,p.centroid.y] for p in ambient.polygons])
        dist,ids=cKDTree(axy).query(xy);assert dist.max()<1e-8
        speed=f.values['velocity_m_s'];baseline=ambient.values['velocity_m_s'][ids]
        deficit=100*(1-speed/baseline);polys=[np.asarray(p.exterior.coords) for p in f.polygons]
        for ax,values,title,cmap,clim in [(axes[j,0],speed,'Resolved velocity (m/s)','viridis',(.6,.86)),(axes[j,1],deficit,'Deficit relative to turbine-free flow (%)','RdBu_r',(-5,20))]:
            pc=PolyCollection(polys,array=values,cmap=cmap,clim=clim,edgecolors='none');
            if cmap=='RdBu_r':pc.set_norm(TwoSlopeNorm(vmin=-5,vcenter=0,vmax=20))
            ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');fig.colorbar(pc,ax=ax,shrink=.75)
            for r in doc['turbines']['turbines']:
                x,y=r['center_xy'];D=r['rotor_diameter_m'];ax.plot([x,x],[y-D/2,y+D/2],color='black',lw=2)
            ax.set(title=case.capitalize()+' - '+title,xlabel='x (m)',ylabel='y (m)')
        rows=read(root/'run/TURBINE_PERFORMANCE.json');rotors=doc['turbines']['turbines'];result=[]
        for r in rotors:
            rr=[v for v in rows if v['turbine_id']==r['id']]
            paxes[j].plot([v['time_s'] for v in rr],[v['useful_power_W'] for v in rr],label=r['id'])
            x,y=r['center_xy'];D=r['rotor_diameter_m'];mask=(xy[:,0]>x+D)&(xy[:,0]<x+3*D)&(abs(xy[:,1]-y)<D/2)
            assert mask.any() and deficit[mask].max()>2.,(case,r['id'])
            result.append(dict(id=r['id'],wake_deficit_1_to_3D_max_percent=float(deficit[mask].max()),power_W=rr[-1]['useful_power_W'],thrust_N=rr[-1]['thrust_N']))
        paxes[j].set(title=case.capitalize(),xlabel='Time (s)',ylabel='Shaft power (W)');paxes[j].set_ylim(bottom=0);paxes[j].legend();paxes[j].grid(alpha=.2)
        summary.append(dict(case=case,rotors=result,water_residual_m3=done['water_residual_m3'],end_s=f.time_s,velocity_is_saved_q=True))
    fig.savefig(REPORT/'RESOLVED_WAKES.png',dpi=170);power.savefig(REPORT/'COUPLED_POWER.png',dpi=170);plt.close('all')
    a=ResultsStore(HOME/'staggered_cpu_short/run').read(-1);b=ResultsStore(HOME/'staggered_gpu_short/run').read(-1)
    gaps=dict(q_max=float(np.max(abs(a.q-b.q))),z_max=float(np.max(abs(a.z-b.z))));assert gaps['q_max']<1e-9
    ar=read(HOME/'staggered_cpu_short/run/TURBINE_PERFORMANCE.json');br=read(HOME/'staggered_gpu_short/run/TURBINE_PERFORMANCE.json')
    gaps['power_W_max']=max(abs(x['useful_power_W']-y['useful_power_W']) for x,y in zip(ar,br));assert gaps['power_W_max']<1e-8
    assert all(read(HOME/('staggered_'+be+'_short')/'DONE.json')['exact_restart'] for be in ['cpu','gpu'])
    save(REPORT/'METRICS.json',dict(passed=True,cases=summary,paired_short_run=gaps,exact_restart_both=True,scope='Fresh 40 s coupled fields; two-second paired restart checks. Not new experimental validation.'))
    print(summary,gaps)
if __name__=='__main__':main()
