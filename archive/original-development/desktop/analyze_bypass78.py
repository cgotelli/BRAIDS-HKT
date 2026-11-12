"""Audit actual saved velocities, matched transparent control and display time."""
from particle_bypass78 import ROOT,TARGET,REPORT,read,save,environment
environment()
from pathlib import Path
import numpy as np
import shapely
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from braids_desktop.results import ResultsStore
from braids_desktop.visual_fields import advect_frozen_particles

USER=Path.home()/'Desktop/BRAIDS_Bern_Turbines77/Bern_data/runs/run_20261101_211340_4f57d1e6'
def main():
    store=ResultsStore(USER);off=ResultsStore(TARGET.parent/'transparent_control/run')
    f=store.read(-1);b=off.read(-1);assert f.time_s==b.time_s==14.
    np.testing.assert_array_equal(f.mesh['vertices'],b.mesh['vertices'])
    rows=store.project['turbines']['turbines'];n=np.array(rows[0]['river_direction_xy']);t=np.array([-n[1],n[0]])
    origin=(np.array(rows[0]['center_xy'])+rows[1]['center_xy'])/2;D=rows[0]['rotor_diameter_m'];basis=np.column_stack((n,t))
    xy=np.array([[p.centroid.x,p.centroid.y] for p in f.polygons]);local=(xy-origin)@basis
    uv=np.column_stack([f.values['velocity_x_m_s'],f.values['velocity_y_m_s']]);uv0=np.column_stack([b.values['velocity_x_m_s'],b.values['velocity_y_m_s']])
    speed=uv@n;diff=(uv-uv0)@n;trans=(uv-uv0)@t
    polygons=[(np.asarray(p.exterior.coords)-origin)@basis/D for p in f.polygons]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,v,title,cmap,lim in zip(axes,[speed,diff,trans],['Actual resolved axial velocity','Turbines on minus transparent: axial','Turbines on minus transparent: lateral'],['viridis','coolwarm','coolwarm'],[(1.2,1.9),(-.25,.25),(-.08,.08)]):
        pc=PolyCollection(polygons,array=v,cmap=cmap,clim=lim);ax.add_collection(pc);ax.set(xlim=(-4,12),ylim=(-5,5),xlabel='Downstream distance / D',ylabel='Transverse distance / D',title=title);ax.set_aspect('equal');fig.colorbar(pc,ax=ax,label='m/s',shrink=.7)
        for r in rows:
            c=(np.array(r['center_xy'])-origin)@basis/D;ax.plot([c[0],c[0]],[c[1]-.5,c[1]+.5],'k-',lw=2)
    fig.suptitle('Modified Bern run at 14 s: same mesh, initial state, boundaries and viscosity; only turbine force removed in control')
    fig.savefig(REPORT/'FLOW_DIFFERENCE.png',dpi=160);plt.close(fig)
    # Display-only interpolated cross sections; quantitative maxima below use raw cells.
    funcs=[LinearNDInterpolator(local,uv@n),LinearNDInterpolator(local,uv0@n)]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained');ys=np.linspace(-4*D,4*D,250)
    for ax,s in zip(axes.flat,[-1,0,1,3]):
        p=np.column_stack((np.full(len(ys),s*D),ys))
        ax.plot(ys/D,funcs[0](p),label='Turbines active');ax.plot(ys/D,funcs[1](p),'--',label='Transparent rotors')
        ax.set(title=f'Cross section x/D = {s}',xlabel='Transverse distance / D',ylabel='Axial velocity (m/s)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(REPORT/'BYPASS_SECTIONS.png',dpi=160);plt.close(fig)
    wet=f.q[0]>1e-6;polys=np.array(f.polygons,dtype=object)[wet];nodes=shapely.point_on_surface(polys)
    points=np.column_stack((shapely.get_x(nodes),shapely.get_y(nodes)));tree=cKDTree(points);v=uv[wet];radii=np.sqrt(shapely.area(polys));geometry=shapely.union_all(polys)
    def velocity(p):
        ds,ids=tree.query(p,k=4);w=1/np.maximum(ds,1e-12)**2;w/=w.sum(axis=1)[:,None]
        return (v[ids]*w[:,:,None]).sum(axis=1)
    seeds=origin+np.column_stack((np.full(13,-3*D),np.linspace(-2.5,2.5,13)*D))@basis.T
    histories=[]
    for corrected in (False,True):
        path=[seeds.copy()]
        for _ in range(25):
            p=path[-1]
            if corrected:q,good=advect_frozen_particles(p,velocity,tree,radii,geometry);assert good.all()
            else:
                vel=velocity(p);_,ids=tree.query(p);dt=np.minimum(.15,.2*radii[ids]/np.maximum(np.linalg.norm(vel,axis=1),1e-12));q=p+dt[:,None]*velocity(p+.5*dt[:,None]*vel)
            path.append(q)
        histories.append((np.asarray(path)-origin)@basis/D)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,history,title in zip(axes,histories,['Old display: cell-size-dependent travel time','Corrected display: equal travel time, substeps']):
        for j in range(len(seeds)):ax.plot(history[:,j,0],history[:,j,1],lw=1);ax.plot(*history[-1,j],'.',color='black')
        for r in rows:
            c=(np.array(r['center_xy'])-origin)@basis/D;ax.plot([c[0],c[0]],[c[1]-.5,c[1]+.5],'k-',lw=3)
        ax.set(xlim=(-3.2,7),ylim=(-3,3),title=title,xlabel='Downstream distance / D',ylabel='Transverse distance / D');ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle('Same saved 14-second velocity field, seeds and 25 display updates; no hydraulic change')
    fig.savefig(REPORT/'PARTICLE_COMPARISON.png',dpi=160);plt.close(fig)
    near=(abs(local[:,1])<3*D)&(local[:,0]>-2*D)&(local[:,0]<3*D)&wet
    sides=near&(abs(local[:,1])>1.5*D)
    all_on=read(USER/'TURBINE_PERFORMANCE.json');all_off=read(TARGET.parent/'transparent_control/run/TURBINE_PERFORMANCE.json')
    samples=[r for r in all_on if r['time_s']==14.]
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');startup=[]
    for ax,r in zip(axes,rows):
        series=[]
        for data,label in [(all_on,'Turbines active'),(all_off,'Transparent rotors')]:
            subset=[s for s in data if s['turbine_id']==r['id']];series.append(subset)
            ax.plot([s['time_s'] for s in subset],[s['local_velocity_m_s'] for s in subset],label=label)
        ax.set(title=r['id'],xlabel='Simulation time (s)',ylabel='Local depth-mean axial speed (m/s)');ax.legend();ax.grid(alpha=.2)
        startup.append(dict(turbine=r['id'],initial_velocity_m_s=series[0][0]['local_velocity_m_s'],final_on_m_s=series[0][-1]['local_velocity_m_s'],final_transparent_m_s=series[1][-1]['local_velocity_m_s']))
    fig.suptitle('Global startup adjustment occurs even without turbine forces')
    fig.savefig(REPORT/'STARTUP.png',dpi=160);plt.close(fig)
    area=sum(p.area for p in f.polygons);window=(abs(local[:,0])<.5*D)&wet
    # Discrete area / strip width estimates the water cross-sectional area.
    section=float(np.sum(np.array([p.area for p in f.polygons])[window]*f.q[0,window]) / D)
    metric=dict(passed=True,user_run=str(USER),time_s=14.,cells=len(f.polygons),samples=samples,
        upstream_row_blockage_estimate=2*np.pi*D**2/4/section,section_area_estimate_m2=section,
        near_axial_difference_min_m_s=float(diff[near].min()),near_axial_difference_max_m_s=float(diff[near].max()),
        outer_side_acceleration_max_m_s=float(diff[sides].max()),near_lateral_change_max_m_s=float(abs(trans[near]).max()),
        side_acceleration_percent_max=float(np.max(diff[sides]/(uv0@n)[sides])*100),
        uniform_velocity_example=dict(velocity_m_s=2.,coarse_cell_m=4.,fine_cell_m=.25,old_coarse_distance_m=.3,old_fine_distance_m=.05,corrected_both_distance_m=.3),
        startup=startup,unchanged_user_project=True,scope='Matched 14-second transient diagnostic; not equilibrium or additional experimental validation')
    save(REPORT/'METRICS.json',metric)
    print({k:v for k,v in metric.items() if k not in ('samples',)},flush=True)
if __name__=='__main__':main()
