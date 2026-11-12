"""Raw-cell comparison of the matched synthetic Smart-Jaeggi bend runs."""
import json
import sys
from pathlib import Path
import numpy as np
from check_meanders184 import ROOT,OUT,REPORT,save
from braids_desktop.results import ResultsStore
from shapely.geometry import LineString,Point
from shapely.strtree import STRtree
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation,PillowWriter


def main():
    cfg=json.loads((OUT/'CASE.json').read_text());line=LineString(cfg['xy'])
    centers=np.asarray(cfg['xy']);ss=np.asarray(cfg['s']);angles=np.asarray(cfg['angle']);br=cfg['breaks']
    cases=('advanced_off','advanced_on');data={};metrics={}
    for key in cases:
        warm=ResultsStore(OUT/key/'runs/to_200');morph=ResultsStore(OUT/key/'runs/to_400')
        frames=[warm.read(i) for i in range(len(warm.times))]+[morph.read(i) for i in range(1,len(morph.times))]
        a=frames[0];b=frames[-1];area=b.arrays['area']
        for field in ('q','z','mass','W','water','bed_boundary'):
            np.testing.assert_array_equal(warm.read(-1).arrays[field],morph.read(0).arrays[field])
        for frame in frames:
            assert frame.epoch==a.epoch
            assert np.isfinite(frame.q).all() and frame.q[0].min()>=0
            assert frame.arrays['mass'].min()>=0
        if key==cases[0]:
            xy=np.array([p.centroid.coords[0] for p in a.polygons]);points=shapely.points(xy)
            station=shapely.line_locate_point(line,points)
            foot=shapely.get_coordinates(shapely.line_interpolate_point(line,station))
            angle=np.interp(station,ss,angles);normal=np.array([-np.sin(angle),np.cos(angle)]).T
            cross=np.sum((xy-foot)*normal,axis=1)
            masks=[]
            for start,end,sign in ((br[1],br[2],1),(br[3],br[4],-1)):
                interior=(station>start+2)&(station<end-2)
                masks.append(dict(inner=interior&(cross*sign>.4),outer=interior&(cross*sign<-.4)))
            polygons=[np.asarray(p.exterior.coords) for p in a.polygons]
            tree=STRtree(a.polygons)
        else:
            np.testing.assert_array_equal(a.mesh['vertices'],data[cases[0]]['frames'][0].mesh['vertices'])
            np.testing.assert_array_equal(a.q,data[cases[0]]['frames'][0].q)
            np.testing.assert_array_equal(a.z,data[cases[0]]['frames'][0].z)
            np.testing.assert_array_equal(a.arrays['mass'],data[cases[0]]['frames'][0].arrays['mass'])
        dz=b.z-a.z
        diags=sum((json.loads((OUT/key/f'runs/to_{t}/DIAGNOSTICS.json').read_text()) for t in (200,400)),[])
        water=max(abs(d['water_residual_m3']) for d in diags)
        sediment=max(max(abs(v) for v in d['class_residual_kg']) for d in diags)
        assert water<1e-8 and sediment<1e-7
        assert all(not any(d['wall_integrals']) for d in diags)
        late0=warm.read(17);late1=warm.read(-1)
        dh=late1.q[0]-late0.q[0];dv=late1.values['velocity_m_s']-late0.values['velocity_m_s']
        result=dict(cells=len(area),minimum_area_m2=float(area.min()),
            end_time_s=b.time_s,morphological_duration_s=(b.time_s-cfg['warmup_s'])*cfg['morph_factor'],
            bed_min_mm=float(dz.min()*1000),bed_max_mm=float(dz.max()*1000),
            erosion_bulk_m3=float(np.dot(area,np.maximum(-dz,0))),deposition_bulk_m3=float(np.dot(area,np.maximum(dz,0))),
            water_residual_m3=water,sediment_residual_kg=sediment,
            fixed_bed_change_m=float(abs(late1.z-a.z).max()),
            warmup_depth_change_max_mm=float(abs(dh).max()*1000),
            warmup_depth_change_rms_mm=float(np.sqrt(np.average(dh*dh,weights=area))*1000),
            warmup_speed_change_max_m_s=float(abs(dv).max()),
            secondary_W_max=float(abs(b.arrays['W']).max()),
            surface_fraction_min=float(b.arrays['sv'].min()),
            bends=[])
        for m in masks:
            result['bends'].append({region:dict(mean_dz_mm=float(np.average(dz[mask],weights=area[mask])*1000),
                min_dz_mm=float(dz[mask].min()*1000),max_dz_mm=float(dz[mask].max()*1000),
                area_m2=float(area[mask].sum()),deposited_bulk_m3=float(np.dot(area[mask],np.maximum(dz[mask],0))),
                eroded_bulk_m3=float(np.dot(area[mask],np.maximum(-dz[mask],0)))) for region,mask in m.items()})
        j=b.values['velocity_m_s']**2/(cfg['chezy']**2*b.q[0])
        result['energy_slope_final']=dict(min=float(j.min()),max=float(j.max()),median=float(np.median(j)),
            area_fraction_below_published_lower_bound=float(area[j<.005].sum()/area.sum()))
        metrics[key]=result;data[key]=dict(frames=frames,area=area,dz=dz)
    difference=data['advanced_on']['dz']-data['advanced_off']['dz']
    metrics['difference']=dict(max_abs_mm=float(abs(difference).max()*1000),
        rms_mm=float(np.sqrt(np.average(difference**2,weights=area))*1000))
    save(REPORT/'COMPARISON.json',metrics)
    def mapplot(ax,values,limit=None,label=None):
        pc=PolyCollection(polygons,array=np.asarray(values),cmap='RdBu_r',edgecolors='none',rasterized=True)
        if limit is not None:pc.set_clim(-limit,limit)
        ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)')
        if label:ax.set_title(label)
        return pc
    limit=max(abs(data[k]['dz']).max() for k in cases)*1000
    fig=plt.figure(figsize=(12,14))
    grid=fig.add_gridspec(3,2,width_ratios=(1,.035),left=.075,right=.89,bottom=.055,top=.91,hspace=.33,wspace=.08)
    axes=[fig.add_subplot(grid[i,0]) for i in range(3)]
    color_axes=[fig.add_subplot(grid[i,1]) for i in range(3)]
    for ax,cax,key in zip(axes,color_axes,cases):
        pc=mapplot(ax,data[key]['dz']*1000,limit,'Advanced hydrodynamics '+key.split('_')[1].upper())
        fig.colorbar(pc,cax=cax,label='Bed change (mm) | blue erosion, red deposition')
        # Indicate the interior of each bend independently of the computed bed.
        for n,(start,end,sign) in enumerate(((br[1],br[2],1),(br[3],br[4],-1))):
            s=(start+end)/2;k=np.argmin(abs(ss-s));normal0=np.array([-np.sin(angles[k]),np.cos(angles[k])])
            p=centers[k]+sign*normal0*2.5
            ax.annotate(f'Inner bend {n+1}',xy=centers[k]+sign*normal0*1.3,xytext=p+sign*normal0*3,
                        arrowprops={'arrowstyle':'->'},fontsize=9)
    pc=mapplot(axes[2],difference*1000,max(abs(difference))*1000,'ON minus OFF — same DEM, mesh and initial state')
    fig.colorbar(pc,cax=color_axes[2],label='Difference in bed change (mm)')
    fig.suptitle('Smart & Jäggi | 200 s hydraulic warm-up + 200 s mobile bed at MF5\n1000 s morphological evolution, raw native-cell values')
    fig.savefig(REPORT/'BED_COMPARISON.png',dpi=170);plt.close(fig)
    # Cross-sections are sampled at native cell values, never smoothed to an expected curve.
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    samples={}
    for row,(start,end,sign) in enumerate(((br[1],br[2],1),(br[3],br[4],-1))):
        for col,fraction in enumerate((.25,.5,.75)):
            s=start+(end-start)*fraction;k=np.argmin(abs(ss-s));n=np.array([-np.sin(angles[k]),np.cos(angles[k])])*sign
            positions=np.linspace(-1.95,1.95,161);pts=centers[k]+positions[:,None]*n
            matches=tree.query(shapely.points(pts),predicate='intersects');idx=np.full(len(pts),-1,dtype=int)
            idx[matches[0]]=matches[1];valid=idx>=0
            values={}
            for key,color in zip(cases,('#4c78a8','#e45756')):
                zz=data[key]['dz'][idx[valid]]*1000;values[key]=zz.tolist()
                axes[row,col].plot(positions[valid],zz,label=key.replace('_',' '),color=color,lw=1.6)
            axes[row,col].axhline(0,color='k',lw=.6);axes[row,col].grid(alpha=.2)
            axes[row,col].set_title(f'Bend {row+1}: {fraction*60:g}° into the bend')
            axes[row,col].set_xlabel('Cross-channel (m): outer − / inner +')
            samples[f'bend{row+1}_{fraction}']=dict(cross_m=positions[valid].tolist(),values_mm=values)
        axes[row,0].set_ylabel('Bed change (mm)')
    axes[0,0].legend();fig.suptitle('Native-cell bed change across each bend after 1000 morphological seconds')
    fig.savefig(REPORT/'CROSS_SECTIONS.png',dpi=170);plt.close(fig)
    save(REPORT/'SECTIONS.json',samples)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    evolution={}
    for key,color in zip(cases,('#4c78a8','#e45756')):
        frames=data[key]['frames'];base=frames[0].z;t=np.array([f.time_s for f in frames]);mask=t>=200
        t=(t[mask]-200)*5/60;dzs=np.array([f.z-base for f in frames])[mask]
        entry={}
        for b,m in enumerate(masks):
            for j,region in enumerate(('inner','outer')):
                mm=m[region];curve=np.average(dzs[:,mm],weights=area[mm],axis=1)*1000
                axes[b,j].plot(t,curve,color=color,label=key.replace('_',' '));axes[b,j].axhline(0,color='k',lw=.5)
                axes[b,j].set(title=f'Bend {b+1}: {region} bank zone',xlabel='Morphological minutes',ylabel='Area-mean bed change (mm)')
                axes[b,j].grid(alpha=.2);entry[f'bend{b+1}_{region}']=curve.tolist()
        evolution[key]=dict(morph_minutes=t.tolist(),mean_bed_change_mm=entry)
    axes[0,0].legend();fig.suptitle('Inner deposition and outer scour: direction and development, not only extrema')
    fig.savefig(REPORT/'EVOLUTION.png',dpi=170);plt.close(fig);save(REPORT/'EVOLUTION.json',evolution)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    pcs=[]
    for ax,key in zip(axes,cases):
        pc=mapplot(ax,data[key]['frames'][20].values['velocity_m_s'],label=key.replace('_',' ').upper())
        pc.set_cmap('viridis');pc.set_clim(1.4,1.9);pcs.append(pc)
    fig.colorbar(pcs[0],ax=axes,label='Speed before bedload activation (m/s)')
    fig.suptitle('Settled hydrodynamics at 200 s, initially unchanged bed')
    fig.savefig(REPORT/'HYDRODYNAMICS.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    pcs=[];movie=[f for f in data[cases[0]]['frames'] if f.time_s>=200]
    for ax,key in zip(axes,cases):
        pcs.append(mapplot(ax,np.zeros(len(area)),limit,key.replace('_',' ').upper()))
    fig.colorbar(pcs[0],ax=axes,label='Bed change (mm)');title=fig.suptitle('')
    def animate(i):
        for pc,key in zip(pcs,cases):
            frames=[f for f in data[key]['frames'] if f.time_s>=200]
            pc.set_array((frames[i].z-frames[0].z)*1000)
        title.set_text(f'Smart & Jäggi | {(movie[i].time_s-200)*5/60:.2f} morphological minutes | MF5')
        return pcs+[title]
    animation=FuncAnimation(fig,animate,frames=len(movie),interval=180,blit=False)
    animation.save(REPORT/'EVOLUTION.gif',writer=PillowWriter(fps=5),dpi=100);plt.close(fig)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
