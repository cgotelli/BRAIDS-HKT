"""Compare corrected same-physics temporal references and historical drag."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.analyze_explicit_fwave124 import load,comparison
from braids_gpu.atomic_output import write_json_atomic
BASE=ROOT/'validation/_resistance126'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/resistance126'


def main():
    OUT.mkdir(parents=True,exist_ok=True);metrics={}
    for resolution in ('1m','5m'):
        path=BASE/f'{resolution}_explicit_gpu_cfl0.225'
        finepath=BASE/f'{resolution}_explicit_gpu_cfl{0.05625 if resolution=="1m" else 0.028125}'
        if not (finepath/'EXPERIMENT.json').exists():continue
        a=load(path);b=load(finepath);old=load(ROOT/f'validation/_explicit_fwave124/{resolution}_explicit_gpu_cfl0.225')
        metric=dict(candidate=a['record'],fine=b['record'],temporal=comparison(a,b),change_from_old=comparison(a,old))
        for label,suffix in [('cpu','explicit_cpu_cfl0.225'),('implicit','implicit_gpu'),('implicit_cpu','implicit_cpu')]:
            extra=BASE/f'{resolution}_{suffix}'
            if (extra/'EXPERIMENT.json').exists():
                v=load(extra);metric[label]=dict(record=v['record'],vs_candidate=comparison(a,v))
        metric['shallow_velocity_controls']=[]
        for threshold in (1e-6,.001,.01):
            peak=None
            for t,s in a['states'].items():
                q=s['q'];f=b['states'][t]['q'];wet=(q[0]>threshold)&(f[0]>threshold)
                if not wet.any():continue
                difference=np.linalg.norm(q[1:,wet]/q[0,wet]-f[1:,wet]/f[0,wet],axis=0)
                j=int(np.flatnonzero(wet)[difference.argmax()]);value=float(difference.max())
                if peak is None or value>peak['maximum_difference_m_s']:
                    peak=dict(minimum_depth_m=threshold,maximum_difference_m_s=value,time_s=t,
                        cell=j,xy=a['xy'][j].tolist(),depths_m=[float(q[0,j]),float(f[0,j])])
            metric['shallow_velocity_controls'].append(peak)
        cell=4358 if resolution=='1m' else 127
        fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for data,label,color in [(old,'Previous face-averaged drag','#1f77b4'),(a,'Corrected CFL 0.225','#ff7f0e'),(b,'Corrected finer timestep','#2ca02c')]:
            rows=sorted(data['states'].items());ts=[t for t,s in rows]
            h=np.array([s['q'][0,cell] for t,s in rows]);mom=np.array([s['q'][1:,cell] for t,s in rows]);v=np.linalg.norm(mom,axis=1)/np.maximum(h,1e-30)
            axs[0].plot(ts,h*1000,'o-',color=color,label=label);axs[1].plot(ts,v,'o-',color=color)
            if data is not old:
                axs[2].plot(ts,np.where(h>1e-6,np.maximum(h,1e-6)**(1/6)/.03,np.nan),'o-',color=color,label=label)
        audit=ROOT/'validation/_explicit_qualification125'/('audit_1m_gpu_v3' if resolution=='1m' else 'audit_5m_gpu_v2')/'AUDIT.json'
        samples=[s for s in json.loads(audit.read_text())['samples'] if s['label']=='explicit225']
        axs[2].plot([s['time'] for s in samples],[s['chezy'] for s in samples],'kx',ms=8,label='Old applied C, audited states')
        for ax,y in zip(axs,['Cell depth (mm)','Cell speed (m/s)','Applied Chezy (m$^{1/2}$/s)']):
            ax.set(xlabel='Time (s)',ylabel=y);ax.grid(alpha=.2)
        axs[0].legend(fontsize=8);axs[1].set_ylim(bottom=0)
        axs[2].set_title('Applied hydraulic resistance',fontsize=9);axs[2].legend(fontsize=7)
        fig.suptitle(f'{resolution} river, audited cell {cell}, cold start')
        fig.savefig(OUT/f'HOTSPOT_{resolution}.png',dpi=170);plt.close(fig)
        metrics[resolution]=metric
        # Show the source correction separately from timestep error. Do not
        # label the former as a millimetre-scale numerical perturbation.
        end=max(a['states']);q=a['states'][end]['q'];qo=old['states'][end]['q']
        wet=(q[0]>.001)|(qo[0]>.001);xy=a['xy'];origin=xy[wet].mean(0)
        bounds=np.array([xy[wet].min(0),xy[wet].max(0)])-origin
        pick=np.all((xy-origin>bounds[0]-12)&(xy-origin<bounds[1]+12),axis=1)
        polygons=[p-origin for p,k in zip(a['polygons'],pick) if k]
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        limit=max(q[0].max(),qo[0].max());difference=(q[0]-qo[0])*1000;maximum=abs(difference).max()
        for ax,values,label,cmap,limits in zip(axes,[qo[0],q[0],difference],
            ['Previous face-averaged drag','Cell-local drag','Corrected minus previous (mm)'],
            ['Blues','Blues','RdBu_r'],[(0,limit),(0,limit),(-maximum,maximum)]):
            pc=PolyCollection(polygons,array=values[pick],cmap=cmap,clim=limits,edgecolors='0.6',linewidths=.15)
            ax.add_collection(pc);ax.set(xlim=(bounds[0,0]-8,bounds[1,0]+8),ylim=(bounds[0,1]-8,bounds[1,1]+8),
                aspect='equal',xlabel='Local easting (m)',ylabel='Local northing (m)',title=label)
            fig.colorbar(pc,ax=ax,shrink=.8,label='mm' if cmap=='RdBu_r' else 'Depth (m)')
        fig.suptitle(f'{resolution} cold-start river at 4 s; raw cells, unchanged geometry and inflow')
        fig.savefig(OUT/f'CORRECTION_{resolution}.png',dpi=170);plt.close(fig)
    # Reuse the already-qualified cropped-cell analyzer on the new evidence.
    root=BASE/'pytest_coupled_v2/unified-river0'
    if len(list(root.glob('*/RESEARCH.json')))==20:
        from tools.analyze_explicit125 import final,read
        coupled={}
        for p in root.glob('*/RESEARCH.json'):
            r=read(p);rows=read(p.parent/'run/DIAGNOSTICS.json')
            coupled[p.parent.name]=dict(history=r['records'],status=r['status'],
                water_max_m3=max(abs(s['water_residual_m3']) for s in rows),
                sediment_max_kg=max(abs(v) for s in rows for v in s['class_residual_kg']),
                wall_max=max(abs(v) for s in rows for v in s['wall_integrals']))
        for case in ('bedload_fixed','bedload_amr'):
            fig,axs=plt.subplots(2,3,figsize=(11,6),layout='constrained');states=[]
            for k,(suffix,label) in enumerate([('explicit_cpu','CPU'),('explicit_gpu','GPU'),('explicit_fine_gpu','GPU finer time')]):
                s,g=final(root/(case+'_'+suffix)/'run');states.append(s)
                polygons=[g['vertices'][g['offsets'][j]:g['offsets'][j+1]] for j in range(len(g['area']))]
                for row,values in enumerate([s['q'][0],(s['z']-s['rock']-.22)*1000]):
                    pc=PolyCollection(polygons,array=values,cmap='Blues' if row==0 else 'RdBu_r',edgecolors='gray',linewidths=.15)
                    axs[row,k].add_collection(pc);axs[row,k].autoscale();axs[row,k].set_aspect('equal')
                    axs[row,k].set(xlabel='x (m)',ylabel='y (m)',title=label)
            for row in range(2):
                artists=[ax.collections[0] for ax in axs[row]];lo=min(p.get_array().min() for p in artists);hi=max(p.get_array().max() for p in artists)
                if row:hi=max(abs(lo),abs(hi));lo=-hi
                for p in artists:p.set_clim(lo,hi)
                fig.colorbar(artists[0],ax=list(axs[row]),label='Depth (m)' if row==0 else 'Bed change (mm)',shrink=.8)
            fig.suptitle(case.replace('_',' ')+': corrected Manning drag, 0.5 s, raw cropped cells')
            fig.savefig(OUT/f'{case.upper()}.png',dpi=170);plt.close(fig)
            c,g,f=states
            coupled[case+'_comparison']=dict(cpu_gpu_depth_mm=float(abs(c['q'][0]-g['q'][0]).max()*1000),
                cpu_gpu_bed_mm=float(abs(c['z']-g['z']).max()*1000),fine_depth_mm=float(abs(g['q'][0]-f['q'][0]).max()*1000),
                fine_bed_mm=float(abs(g['z']-f['z']).max()*1000),
                bed_range_mm=[float(((g['z']-g['rock']-.22)*1000).min()),float(((g['z']-g['rock']-.22)*1000).max())])
        write_json_atomic(OUT/'COUPLED_METRICS.json',coupled)
    write_json_atomic(OUT/'METRICS.json',metrics)
    if metrics:
        fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,(res,m) in zip(axs,metrics.items()):
            names=['Explicit CPU','Explicit GPU','Implicit GPU'];values=[m['cpu']['record']['stepping_s'],m['candidate']['stepping_s'],m['implicit']['record']['stepping_s']]
            if 'implicit_cpu' in m:names.insert(0,'Implicit CPU');values.insert(0,m['implicit_cpu']['record']['stepping_s'])
            ax.bar(names,values,color=['#537da0' if 'CPU' in n else '#bb703d' for n in names])
            for j,v in enumerate(values):ax.text(j,v+.015*max(values),f'{v:.3f}',ha='center')
            ax.set(title=res+' static terrain, 4 simulated seconds',ylabel='Stepping wall time (s)',ylim=(0,max(values)*1.17));ax.tick_params(axis='x',labelsize=8)
        fig.suptitle('Matched corrected resistance; sequential measurements, no preparation or I/O')
        fig.savefig(OUT/'STEPPING_TIME.png',dpi=170);plt.close(fig)
    for k,v in metrics.items():
        print(k,json.dumps(dict(depth_max_mm=v['temporal']['worst_depth_mm'],
            velocity_max_m_s=max(r['velocity_both_over_1cm_max_m_s'] or 0 for r in v['temporal']['rows']),
            old_depth_change_mm=v['change_from_old']['worst_depth_mm'],stepping=v['candidate']['stepping_s'])))


if __name__=='__main__':main()
