"""Area-overlap comparisons, measured timings and figures for isolated trials."""
import json
import hashlib
import zipfile
from pathlib import Path
import sys
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_merged_fwave115'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/merged_fwave115'
read=lambda p:json.loads(p.read_text())

def frames(name):
    root=(BASE/name if not name.startswith('old_') else ROOT/'validation/_test1_amr_interval'/('cpu_installed_1s' if name=='old_cpu' else 'gpu_installed_1s'))/'run/frames'
    cache={};result=[]
    for row in read(root/'INDEX.json')['frames']:
        ep=row['epoch']
        if ep not in cache:
            with np.load(root/ep) as f:
                origin=np.floor(np.min(f['vertices'],axis=0)/100)*100
                polys=[Polygon(f['vertices'][a:b]-origin) for a,b in zip(f['offsets'][:-1],f['offsets'][1:])]
                cache[ep]=(polys,origin)
        with np.load(root/row['file']) as f:data={k:f[k] for k in ('q','z','area')}
        result.append(dict(time=row['time_s'],polys=cache[ep][0],origin=cache[ep][1],**data))
    return result

def overlap(a,b):
    # Project every piecewise-constant control volume onto its exact overlap;
    # do not use nearest-neighbour values for differing AMR/merged meshes.
    ap=np.array(a['polys'],object);bp=np.array(b['polys'],object)
    if np.any(a['origin']!=b['origin']):
        from shapely.affinity import translate
        delta=b['origin']-a['origin'];bp=np.array([translate(p,*delta) for p in bp],object)
    ia,ib=STRtree(bp).query(ap,predicate='intersects')
    area=shapely.area(shapely.intersection(ap[ia],bp[ib]));valid=area>1e-8
    ia,ib,area=ia[valid],ib[valid],area[valid]
    assert abs(area.sum()-a['area'].sum())<1e-5
    return ia,ib,area

def compare(a,b):
    assert abs(a['time']-b['time'])<1e-8
    ia,ib,area=overlap(a,b);q=a['q'][:,ia];r=b['q'][:,ib]
    wet=(q[0]>.001)|(r[0]>.001);both=(q[0]>.01)&(r[0]>.01)
    dh=q[0]-r[0]
    du=np.linalg.norm(q[1:,both]/q[0,both]-r[1:,both]/r[0,both],axis=0)
    rms=lambda x,w:float(np.sqrt(np.sum(x*x*w)/np.sum(w))) if len(w) else 0.
    return dict(time_s=a['time'],h_max_mm=1000*float(abs(dh).max()),h_wet_rms_mm=1000*rms(dh[wet],area[wet]),
        velocity_max_m_s=float(du.max(initial=0)),velocity_wet_rms_m_s=rms(du,area[both]),
        wet_symmetric_difference_m2=float(area[(q[0]>.001)!=(r[0]>.001)].sum()),
        maximum_depth_m=[float(q[0].max()),float(r[0].max())],
        water_volume_m3=[float(np.sum(q[0]*area)),float(np.sum(r[0]*area))],
        bed_max_mm=1000*float(np.max(abs(a['z'][ia]-b['z'][ib]))))

def main():
    OUT.mkdir(parents=True,exist_ok=True);runs={}
    for p in BASE.glob('*/PROFILE_SUMMARY.json'):
        d=read(p)
        if not d.get('result'):continue
        ex=read(p.parent/'EXPERIMENT.json')
        runs[p.parent.name]=dict(phases=d['phases'],preparation_s=d['preparation_s'],
            worker_call_s=d['worker_call_s'],hydraulic=d['result']['hydraulic'],settings=ex['settings'])
    pairs=[('gpu_explicit_fixed','cpu_explicit_fixed'),('cpu_explicit_fixed','cpu_imp_fixed'),
        ('gpu_implicit_fixed','cpu_imp_fixed'),('cpu_explicit_amr_v2','cpu_explicit_fixed'),
        ('gpu_explicit_amr_v2','cpu_explicit_amr_v2'),('cpu_explicit_amr_v2','cpu_implicit_amr_v2'),
        ('cpu_explicit_fixed','old_cpu'),('cpu_imp_fixed','old_cpu'),
        ('cpu_explicit_dt0.05','cpu_explicit_dt0.025'),('cpu_explicit_fixed','cpu_explicit_dt0.025'),
        ('cpu_explicit_dt0.025','cpu_implicit_dt0.025')]
    pairs.extend([('cpu_explicit_original','old_cpu'),('gpu_explicit_original','cpu_explicit_original'),
        ('cpu_explicit_original','cpu_explicit_original_dt0.05'),
        ('cpu_explicit_original_dt0.05','cpu_explicit_original_dt0.025'),
        ('gpu_explicit_bedload','cpu_explicit_bedload'),('cpu_explicit_bedload','cpu_implicit_bedload')])
    pairs.extend([('gpu_explicit_original_bedload','cpu_explicit_original_bedload'),
        ('cpu_explicit_original_bedload','cpu_implicit_original_bedload')])
    cache={};comparisons={}
    for first,second in pairs:
        try:
            for name in (first,second):
                if name not in cache:cache[name]=frames(name)
        except FileNotFoundError:continue
        comparisons[first+' vs '+second]=[compare(a,b) for a,b in zip(cache[first],cache[second])]
    bed=[]
    for name in ('cpu_explicit_original_bedload','gpu_explicit_original_bedload','cpu_implicit_original_bedload'):
        if name not in cache:continue
        v=cache[name]
        # Map initial bed onto each actual output control volume.
        for d in v:
            ia,ib,area=overlap(d,v[0]);initial=np.bincount(ia,weights=v[0]['z'][ib]*area,minlength=len(d['z']))/np.bincount(ia,weights=area,minlength=len(d['z']))
            dz=d['z']-initial;d['dz']=dz
            bed.append(dict(run=name,time_s=d['time'],minimum_dz_mm=float(dz.min())*1000,maximum_dz_mm=float(dz.max())*1000,
                erosion_m3=float(-np.sum(np.minimum(dz,0)*d['area'])),deposition_m3=float(np.sum(np.maximum(dz,0)*d['area']))))
    (OUT/'METRICS.json').write_text(json.dumps(dict(runs=runs,comparisons=comparisons,bed=bed),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,backend in zip(axes,('cpu','gpu')):
        labels=[];rows=[]
        old=read(ROOT/f'validation/_test1_amr_interval/{backend}_installed_1s/PROFILE_SUMMARY.json')
        labels.append('Installed\nimplicit AMR');rows.append(old['phases'])
        for name,label in [(f'{backend}_imp_fixed' if backend=='cpu' else 'gpu_implicit_fixed','Merged\nimplicit fixed'),
            (f'{backend}_explicit_fixed','Merged\nexplicit fixed'),(f'{backend}_explicit_amr_v2','Merged\nexplicit AMR'),
            (f'{backend}_explicit_original','Original\nexplicit AMR')]:
            if name in runs:labels.append(label);rows.append(runs[name]['phases'])
        bottom=np.zeros(len(rows))
        for phase,color in [('stepping','#d95f02'),('amr','#7570b3'),('initialization','#66a61e'),('output','#e6ab02'),('checkpoint','#a6761d'),('other','#999999')]:
            v=np.array([r.get(phase,0) for r in rows]);ax.bar(labels,v,bottom=bottom,label=phase,color=color);bottom+=v
        ax.set_title(backend.upper()+' · 8 simulated seconds');ax.set_ylabel('Wall time (s)');ax.set_ylim(0,max(bottom)*1.15)
        for x,y in enumerate(bottom):ax.text(x,y+.1,f'{y:.2f}',ha='center')
    axes[1].legend(fontsize=8);fig.savefig(OUT/'PERFORMANCE.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,backend in zip(axes,('cpu','gpu')):
        old=read(ROOT/f'validation/_test1_amr_interval/{backend}_installed_1s/PROFILE_SUMMARY.json')
        new=runs[f'{backend}_explicit_original'];rows=[old['phases'],new['phases']];bottom=np.zeros(2)
        for phase,color in [('stepping','#d95f02'),('amr','#7570b3'),('initialization','#66a61e'),('output','#e6ab02'),('checkpoint','#a6761d'),('other','#999999'),('load','#bbbbbb')]:
            values=np.array([r.get(phase,0) for r in rows]);ax.bar(['Installed implicit','Explicit prototype'],values,bottom=bottom,label=phase,color=color);bottom+=values
        for x,y in enumerate(bottom):ax.text(x,y+.12,f'{y:.2f} s',ha='center')
        ax.set_ylim(0,max(bottom)*1.16);ax.set_title(backend.upper()+' · same original mesh and AMR');ax.set_ylabel('Worker wall time for 8 simulated seconds (s)')
    axes[1].legend(fontsize=8);fig.suptitle('Preparation excluded · identical forcing and sediment disabled')
    fig.savefig(OUT/'DECISION_PERFORMANCE.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for key,label in [('cpu_explicit_fixed vs cpu_imp_fixed','Time integrator only'),
        ('cpu_explicit_fixed vs old_cpu','Mesh and time integrator'),
        ('cpu_explicit_amr_v2 vs cpu_explicit_fixed','AMR vs fixed merged'),
        ('cpu_explicit_fixed vs cpu_explicit_dt0.025','Explicit 0.1 vs 0.025 s')]:
        if key not in comparisons:continue
        v=comparisons[key]
        for ax,field in zip(axes,('h_max_mm','h_wet_rms_mm')):ax.plot([x['time_s'] for x in v],[x[field] for x in v],label=label)
    for ax,title in zip(axes,('Maximum depth difference','Wet-area RMS depth difference')):
        ax.set_title(title);ax.set_xlabel('Simulation time (s)');ax.set_ylabel('mm');ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'ACCURACY.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for key,label in [('cpu_explicit_original vs old_cpu','Explicit vs installed implicit'),
        ('cpu_explicit_original vs cpu_explicit_original_dt0.05','Explicit dtmax 0.1 vs 0.05'),
        ('cpu_explicit_original_dt0.05 vs cpu_explicit_original_dt0.025','Explicit dtmax 0.05 vs 0.025')]:
        if key not in comparisons:continue
        v=comparisons[key]
        for ax,field in zip(axes,('h_max_mm','velocity_max_m_s')):ax.plot([x['time_s'] for x in v],[x[field] for x in v],label=label)
    for ax,title,unit in zip(axes,('Maximum depth difference','Maximum velocity difference where both h > 1 cm'),('mm','m/s')):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Simulation time (s)');ax.set_ylabel(unit);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'ORIGINAL_MESH_ACCURACY.png',dpi=160);plt.close(fig)
    a=cache['cpu_explicit_fixed'][-1];b=cache['old_cpu'][-1]
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    wet=np.flatnonzero(a['q'][0]>.001);pts=np.concatenate([np.array(a['polys'][i].exterior.coords) for i in wet]);lo=pts.min(0)-5;hi=pts.max(0)+5
    for ax,d,title in zip(axes,(b,a),('Installed implicit AMR','Merged explicit fixed')):
        polygons=[np.array(p.exterior.coords)+(d['origin']-a['origin']) for p in d['polys']]
        pc=PolyCollection(polygons,array=np.ma.masked_where(d['q'][0]<.001,d['q'][0]),cmap='Blues',edgecolors='#aaaaaa',linewidths=.3)
        pc.set_clim(0,max(a['q'][0].max(),b['q'][0].max()));ax.add_collection(pc);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title(title+' · t = 8 s');ax.set_xlabel('Local east (m)');ax.set_ylabel('Local north (m)')
    fig.colorbar(pc,ax=axes,label='Water depth (m)',shrink=.8);fig.savefig(OUT/'INLET_DEPTH.png',dpi=160);plt.close(fig)
    a=cache['cpu_explicit_original'][-1];b=cache['old_cpu'][-1]
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    wet=np.flatnonzero(a['q'][0]>.001);pts=np.concatenate([np.array(a['polys'][i].exterior.coords) for i in wet]);lo=pts.min(0)-5;hi=pts.max(0)+5
    ia,ib,weights=overlap(a,b);ref=np.bincount(ia,weights=b['q'][0,ib]*weights,minlength=len(a['area']))/np.bincount(ia,weights=weights,minlength=len(a['area']))
    for ax,d,title,val in zip(axes,(b,a,a),('Installed implicit','Explicit prototype','Explicit minus implicit (mm)'),(b['q'][0],a['q'][0],1000*(a['q'][0]-ref))):
        polygons=[np.array(p.exterior.coords)+(d['origin']-a['origin']) for p in d['polys']]
        diff=ax is axes[-1]
        pc=PolyCollection(polygons,array=val if diff else np.ma.masked_where(val<.001,val),cmap='RdBu_r' if diff else 'Blues',edgecolors='#aaaaaa',linewidths=.3)
        pc.set_clim(-.15,.15) if diff else pc.set_clim(0,.35)
        ax.add_collection(pc);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('Local east (m)')
        fig.colorbar(pc,ax=ax,orientation='horizontal',label='mm' if diff else 'Water depth (m)',shrink=.9)
    axes[0].set_ylabel('Local north (m)');fig.suptitle('Test1 inlet after 8 seconds · original mesh retained')
    fig.savefig(OUT/'ORIGINAL_INLET_DEPTH.png',dpi=160);plt.close(fig)
    if bed:
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
        for name,label in [('cpu_explicit_original_bedload','Explicit CPU'),('gpu_explicit_original_bedload','Explicit GPU'),('cpu_implicit_original_bedload','Implicit CPU')]:
            rows=[r for r in bed if r['run']==name]
            if not rows:continue
            for ax,field in zip(axes,('erosion_m3','minimum_dz_mm')):ax.plot([r['time_s'] for r in rows],[r[field] for r in rows],label=label)
        for ax,title,unit in zip(axes,('Bed erosion volume','Largest local erosion'),('m³','Bed change (mm)')):
            ax.set_title(title);ax.set_xlabel('Simulation time (s)');ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend()
        fig.savefig(OUT/'BEDLOAD.png',dpi=160);plt.close(fig)
    for k,v in comparisons.items():print(k,'peak mm',max(x['h_max_mm'] for x in v),'final',v[-1])
    sources=[ROOT/'tools/profile_merged_fwave115.py',Path(__file__),ROOT/'tests/test_merged_fwave115.py',
        *sorted((ROOT/'validation/hydraulics/merged_fwave115').glob('*.py')),
        *sorted((ROOT/'validation/hydraulics/merged_fwave115').glob('*.h'))]
    with zipfile.ZipFile(OUT/'RESEARCH_SOURCES.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:archive.write(path,path.relative_to(ROOT))
    manifest=dict(sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        evidence={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*')
            if p.name in ('EXPERIMENT.json','PROFILE_SUMMARY.json','PROJECT.json','DIAGNOSTICS.json','INDEX.json','tests.xml')},
        hardware=dict(cpu='AMD Ryzen 9 9950X3D',physical_cores=16,logical_processors=32,threads=8,gpu='NVIDIA GeForce RTX 5080',driver='610.88'),
        scope='Research only; production, installed alpha113 and Delft3D untouched')
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
