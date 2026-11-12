from pathlib import Path
from hub_bern77 import ROOT,TARGET,REPORT,read,save,environment
environment()
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from braids_desktop.results import ResultsStore

def main():
    examples=[('Laboratory staggered',ROOT/'desktop/qualification/coupled_wake76/staggered_gpu/run',(.6,.86),None),
              ('Bern startup check',Path.home()/'Desktop/BRAIDS_Bern_Turbines77/Bern_data/runs/startup_check',(1.3,2.),(2600438,2600466,1198646,1198676))]
    metrics=[]
    for name,path,clim,zoom in examples:
        store=ResultsStore(path);f=store.read(-1);key=store.hub_variables[-1];hub=store.hub_field(f,key)
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        polys=[np.asarray(p.exterior.coords) for p in f.polygons]
        arrays=[f.values['velocity_m_s'],hub['speed_m_s'],hub['speed_m_s']-f.values['velocity_m_s']]
        for ax,values,title,limits,cmap in zip(axes,arrays,['Depth-averaged speed','Approximate hub-plane speed','Hub-plane minus depth mean'],[clim,clim,(-.3,.3)],['Reds','Reds','coolwarm']):
            pc=PolyCollection(polys,array=np.ma.masked_invalid(values),clim=limits,cmap=cmap,edgecolors='none');ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');fig.colorbar(pc,ax=ax,label='m/s',shrink=.8)
            for r in store.project['turbines']['turbines']:
                c=np.array(r['center_xy']);n=np.array(r['river_direction_xy']);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])*r['rotor_diameter_m']/2
                ax.plot([c[0]-t[0],c[0]+t[0]],[c[1]-t[1],c[1]+t[1]],'k-',lw=2)
            if zoom:ax.set_xlim(zoom[:2]);ax.set_ylim(zoom[2:])
            ax.set_title(title);ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(useOffset=False,style='plain')
        fig.suptitle(f"{name}, t={f.time_s:g} s; plane z={hub['elevation_m']:.3f} m\nWhite: plane outside water, or unqualified near wake. Uniform ambient vertical profile; no own-induction added.")
        tag='BERN' if zoom else 'LAB';fig.savefig(REPORT/(tag+'_HUB_COMPARISON.png'),dpi=160);plt.close(fig)
        metrics.append(dict(case=name,time_s=f.time_s,hub_elevation_m=hub['elevation_m'],valid_cells=int(hub['valid'].sum()),near_wake_masked_cells=int(hub['near_wake_unresolved'].sum()),max_speed_departure_m_s=float(np.nanmax(abs(arrays[2])))))
        if zoom:
            areas=np.asarray([p.area for p in f.polygons]);sizes=np.sqrt(areas)
            assert sizes.min()>=.25*(1-1e-7) and sizes.max()==8.
            fig,ax=plt.subplots(figsize=(7,9));pc=PolyCollection(polys,array=sizes,cmap='viridis',norm=LogNorm(.25,8),edgecolors=(0,0,0,.18),linewidths=.15);ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');fig.colorbar(pc,ax=ax,label='Square root of actual cell area (m)')
            for r in store.project['turbines']['turbines']:ax.plot(*r['center_xy'],'rx')
            ax.set(title='Bern: coarse river mesh with local turbine/wake refinement',xlabel='Easting (m)',ylabel='Northing (m)');ax.ticklabel_format(useOffset=False,style='plain');fig.tight_layout();fig.savefig(REPORT/'BERN_MESH.png',dpi=160);plt.close(fig)
            mesh=dict(cells=len(areas),ordinary_cell_size_m=8.,minimum_area_m2=float(areas.min()),ordinary_cells=int((sizes==8.).sum()),ordinary_area_fraction=float(areas[sizes==8.].sum()/areas.sum()),finest_area_fraction=float(areas[sizes<.36].sum()/areas.sum()))
    a=ResultsStore(ROOT/'desktop/qualification/coupled_wake76/staggered_cpu_short/run');b=ResultsStore(ROOT/'desktop/qualification/coupled_wake76/staggered_gpu_short/run')
    av=a.values(a.read(-1),a.hub_variables[-1]);bv=b.values(b.read(-1),b.hub_variables[-1]);np.testing.assert_allclose(av,bv,atol=1e-12,rtol=0,equal_nan=True)
    save(REPORT/'METRICS.json',dict(passed=True,fields=metrics,bern_mesh=mesh,backend_hub_speed_max_difference_m_s=float(np.nanmax(abs(av-bv))),scope='Postprocessing identity/consistency and bounded Bern startup, not a 3-D or site-energy validation.'))
    print(mesh)
if __name__=='__main__':main()
