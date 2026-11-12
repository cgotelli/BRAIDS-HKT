"""Unsmoothened local-drag diagnosis and bounded public-route figures."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_explicit_qualification125'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/explicit_qualification125'


def read(p):return json.loads(p.read_text())


def final(folder):
    fs=folder/'frames';idx=read(fs/'INDEX.json');row=idx['frames'][-1]
    return dict(np.load(fs/row['file'])),dict(np.load(fs/row['epoch']))


def main():
    OUT.mkdir(parents=True,exist_ok=True);metrics={'audits':{},'public':{},'coupled':{}}
    for resolution,folder in [('1m','audit_1m_gpu_v3'),('5m','audit_5m_gpu_v2')]:
        d=read(BASE/folder/'AUDIT.json');e=dict(np.load(BASE/folder/'ENDS.npz'));i=d['cell']
        metrics['audits'][resolution]={'cell':i,'samples':d['samples'],'continuations':{}}
        ref=e['explicit0140625'];wet=(ref[0]>.01)
        cpu=dict(np.load(BASE/f'audit_{resolution}_cpu/ENDS.npz'))
        metrics['audits'][resolution]['cpu_gpu_max_state_difference']={k:float(abs(cpu[k]-e[k]).max()) for k in e}
        for k,q in e.items():
            if k=='initial':continue
            metrics['audits'][resolution]['continuations'][k]=dict(depth_max_mm=float(abs(q[0]-ref[0]).max()*1000),
                cell_depth_mm=float((q[0,i]-ref[0,i])*1000),cell_velocity_m_s=(q[1:,i]/q[0,i]).tolist(),
                velocity_max_m_s=float(np.linalg.norm(q[1:,wet]/q[0,wet]-ref[1:,wet]/ref[0,wet],axis=0).max()))
        fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
        for label in ('explicit225','fine','implicit'):
            rows=[s for s in d['samples'] if s['label']==label]
            times=[s['time'] for s in rows]
            axs[0].plot(times,[1000*s['h'] for s in rows],'o-',label=label)
            axs[1].plot(times,[np.linalg.norm(s['velocity']) for s in rows],'o-')
            axs[2].plot(times,[s['chezy'] for s in rows],'o-')
        for ax,title in zip(axs,['Depth (mm)','Speed (m/s)','Effective Chezy (m$^{1/2}$/s)']):
            ax.set(xlabel='Simulation time (s)',ylabel=title);ax.grid(alpha=.25)
        axs[0].legend();fig.suptitle(f'{resolution} raw cell {i}: velocity sensitivity accompanies resistance changes')
        fig.savefig(OUT/f'LOCAL_{resolution}.png',dpi=170);plt.close(fig)
    s=next(s for s in metrics['audits']['1m']['samples'] if s['label']=='fine' and s['time']==2.5)
    f=next(f for f in s['faces'] if f['face']==8054)
    values=[*f['depths'],f['hydraulic_local'][0],f['hydraulic_local'][4],f['physical_face_depth_from_manning_m']]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.bar(['Cell L','Cell R','Hydraulic face L','Hydraulic face R','Roughness face'],values)
    ax.set(yscale='log',ylabel='Water depth (m; logarithmic axis)',title='Same state, same open interior face 8054 — incompatible depth reconstructions')
    for k,v in enumerate(values):ax.text(k,v*1.25,f'{v:.6g}',ha='center')
    ax.set_ylim(5e-7,1.);ax.grid(axis='y',alpha=.2);fig.savefig(OUT/'FACE_DEPTH_AUDIT.png',dpi=170);plt.close(fig)
    for path in BASE.glob('audit_*_frozen*/AUDIT.json'):
        d=read(path);e=dict(np.load(path.with_name('ENDS.npz')));i=d['cell'];r=e['explicit0140625'];c=e['explicit225']
        metrics['audits'][d['resolution']]['frozen_drag_control']=dict(folder=str(path.parent.relative_to(ROOT)),
            cell_velocity_difference_m_s=float(np.linalg.norm(c[1:,i]/c[0,i]-r[1:,i]/r[0,i])),
            depth_max_mm=float(abs(c[0]-r[0]).max()*1000),not_a_physical_replacement=True)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for ax,res in zip(axs,('1m','5m')):
        v=metrics['audits'][res];rows=v['continuations']
        initial=np.array(rows['explicit225']['cell_velocity_m_s'])-rows['explicit0140625']['cell_velocity_m_s']
        vals=[float(np.linalg.norm(initial)),v['frozen_drag_control']['cell_velocity_difference_m_s']]
        ax.bar(['Dynamic face resistance','Frozen resistance\n(diagnostic only)'],vals,color=['#bb5544','#4878a8'])
        ax.set(title=res+' same-state, 0.5 s continuation',ylabel='Local velocity timestep difference (m/s)',ylim=(0,max(vals)*1.2))
        for j,value in enumerate(vals):ax.text(j,value+max(vals)*.025,f'{value:.5g}',ha='center')
    fig.savefig(OUT/'DRAG_CAUSAL_CONTROL.png',dpi=170);plt.close(fig)
    for path in sorted((BASE/'pytest_public').glob('unified-river*/*/run/DIAGNOSTICS.json')):
        rows=read(path);end,g=final(path.parent)
        metrics['public'][path.parent.parent.name]=dict(water_max_m3=max(abs(r['water_residual_m3']) for r in rows),
            wall_max=max(abs(v) for r in rows for v in r['wall_integrals']),depth_min=float(end['q'][0].min()),
            final_time=rows[-1]['time_s'],cells=rows[-1]['cells'])
    roots=sorted(BASE.glob('pytest_coupled*/unified-river*'))
    root=next(r for r in reversed(roots) if len(list(r.glob('*/RESEARCH.json')))==20)
    for p in sorted(root.glob('*/RESEARCH.json')):
        d=read(p);out=p.parent/'run';s,g=final(out);rows=read(out/'DIAGNOSTICS.json');events=read(out/'ADAPTATION.json')
        # Original traces used sub-maximum area as a count: on mixed meshes
        # that also counted whole fine cells. Derive the true geometry here.
        history=[dict(r) for r in d['records']]
        index=read(out/'frames/INDEX.json')
        for row,frame in zip(history,index['frames']):
            geom=dict(np.load(out/'frames'/frame['epoch']))
            nominal=.25/(4.**geom['levels'])
            row['cropped_cells']=int(np.sum(geom['area']<.99*nominal))
        metrics['coupled'][p.parent.name]=dict(source=str(p.relative_to(ROOT)),method=d['method'],backend=d['backend'],
            status=d['status'],history=history,water_max_m3=max(abs(r['water_residual_m3']) for r in rows),
            sediment_max_kg=max(abs(v) for r in rows for v in r['class_residual_kg']),
            wall_max=max(abs(v) for r in rows for v in r['wall_integrals']),adaptation_events=events)
    for case in ('bedload_fixed','bedload_amr'):
        variants=[('explicit_cpu','Explicit CPU'),('explicit_gpu','Explicit GPU'),('explicit_fine_gpu','Fine-time explicit GPU'),('implicit_gpu','Implicit GPU')]
        states=[];fig,axs=plt.subplots(2,4,figsize=(13,5),layout='constrained')
        for k,(name,label) in enumerate(variants):
            s,g=final(root/(case+'_'+name)/'run');states.append(s)
            polygons=[g['vertices'][g['offsets'][j]:g['offsets'][j+1]] for j in range(len(g['area']))]
            for row,data in [(0,s['q'][0]),(1,(s['z']-s['rock']-.22)*1000)]:
                artist=PolyCollection(polygons,array=data,cmap='Blues' if row==0 else 'RdBu_r',edgecolors='gray',linewidths=.12)
                axs[row,k].add_collection(artist);axs[row,k].autoscale();axs[row,k].set_aspect('equal');axs[row,k].set_title(label)
                if k==0:axs[row,k].set_ylabel('y (m)')
                axs[row,k].set_xlabel('x (m)')
        for row in range(2):
            artists=[ax.collections[0] for ax in axs[row]];lo=min(x.get_array().min() for x in artists);hi=max(x.get_array().max() for x in artists)
            if row:hi=max(abs(lo),abs(hi));lo=-hi
            for a in artists:a.set_clim(lo,hi)
            fig.colorbar(artists[0],ax=axs[row].tolist(),shrink=.8,label='Depth (m)' if row==0 else 'Bed change (mm)')
        fig.suptitle(case.replace('_',' ')+' at 0.5 s — raw cropped cells, shared colour scales')
        fig.savefig(OUT/(case.upper()+'.png'),dpi=170);plt.close(fig)
        c,g,f,imp=states
        metrics['coupled'][case+'_comparisons']=dict(cpu_gpu_depth_max_mm=float(abs(c['q'][0]-g['q'][0]).max()*1000),
            cpu_gpu_bed_max_mm=float(abs(c['z']-g['z']).max()*1000),fine_depth_max_mm=float(abs(g['q'][0]-f['q'][0]).max()*1000),
            fine_bed_max_mm=float(abs(g['z']-f['z']).max()*1000),implicit_depth_max_mm=float(abs(g['q'][0]-imp['q'][0]).max()*1000),
            implicit_bed_max_mm=float(abs(g['z']-imp['z']).max()*1000),bed_change_range_mm=[float(((g['z']-g['rock']-.22)*1000).min()),float(((g['z']-g['rock']-.22)*1000).max())])
    from braids_core import river_worker
    metrics['runtime_identity']=river_worker.source_identity()
    paths=[Path(__file__),ROOT/'tools/audit_explicit125.py',ROOT/'tests/test_explicit_qualification125.py',ROOT/'validation/hydraulics/explicit_qualification125/pytest_candidate.py']
    metrics['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    from braids_gpu.atomic_output import write_json_atomic
    write_json_atomic(OUT/'METRICS.json',metrics)
    print(json.dumps({k:v for k,v in metrics['coupled'].items() if k.endswith('comparisons')},indent=2))


if __name__=='__main__':main()
