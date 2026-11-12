"""Figures, exact-field comparisons and source provenance for the small-cell audit."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/_smallcell119'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/smallcell119'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def last(folder):
    idx=read(folder/'run/frames/INDEX.json');row=idx['frames'][-1]
    with np.load(folder/'run/frames'/row['file']) as a:return {k:a[k] for k in a.files}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    geo=dict(np.load(BASE/'GEOMETRY.npz'));saved=read(BASE/'SAVED_AUDIT.json')
    xy=geo['xy'].T;area=geo['area'];fraction=geo['fraction']
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,b,label,mark in zip(axes,('1','0'),('Inlet: wet at 3.75 s','Outlet: still dry at 3.75 s'),(3920,40572)):
        owners=np.array(saved['boundaries'][b]['owners']);origin=xy[owners].mean(0)
        selected=np.where(np.max(abs(xy-origin),axis=1)<25)[0]
        polys=[geo['vertices'][geo['offsets'][i]:geo['offsets'][i+1]]-origin for i in selected]
        pc=PolyCollection(polys,array=fraction[selected],cmap='viridis',norm=LogNorm(1e-4,1),
            edgecolors='.55',linewidths=.3);ax.add_collection(pc)
        wet=selected[geo['q'][0,selected]>.001]
        ax.scatter(*(xy[wet]-origin).T,s=4,c='cyan',label='Water depth > 1 mm')
        ax.scatter(*(xy[mark]-origin),s=75,facecolors='none',edgecolors='red',lw=1.5,zorder=5)
        ax.annotate(f'Cell {mark}\n{area[mark]:.6f} m²',xy[mark]-origin,xytext=(5,15),
            textcoords='offset points',color='darkred',fontsize=9)
        ax.set(xlim=(-23,23),ylim=(-15,25),aspect='equal',title=label,
            xlabel='Easting relative to boundary centre (m)',ylabel='Northing relative to boundary centre (m)')
        if len(wet):ax.legend(loc='lower left',fontsize=8)
    fig.colorbar(pc,ax=axes,label='Actual area / nominal cell area',shrink=.65)
    fig.suptitle('Original 1 m boundary geometry — no cell or bed modification')
    fig.savefig(OUT/'BOUNDARY_CELLS.png',dpi=180);plt.close(fig)
    stats={};fields={}
    for backend in ('cpu','gpu'):
        rows=[]
        for variant in ('baseline','budget_fixed'):
            p=BASE/(backend+'_'+variant);profile=read(p/'PROFILE_SUMMARY.json');trace=read(p/'TRACE.json')
            rows.append(dict(worker_s=profile['worker_call_s'],stepping_s=profile['phases']['stepping'],
                discarded_s=sum(x['wall_s'] for x in trace['outer'] if not x['accepted']),
                attempted_singles=len(trace['singles']),accepted_substeps=profile['result']['hydraulic']['accepted_substeps'],
                minimum_dt=profile['result']['hydraulic']['minimum_substep_s'],
                budget=read(p/'run/DIAGNOSTICS.json')[-1]))
            fields[backend,variant]=last(p)
        before=fields[backend,'baseline'];after=fields[backend,'budget_fixed']
        differences={k:float(np.max(abs(before[k]-after[k]))) for k in ('q','z','mass','water','W','suspended','bedforms')}
        assert all(v==0 for v in differences.values()),differences
        stats[backend]=dict(before=rows[0],after=rows[1],exact_field_differences=differences,
            stepping_reduction_percent=100*(1-rows[1]['stepping_s']/rows[0]['stepping_s']),
            worker_reduction_percent=100*(1-rows[1]['worker_s']/rows[0]['worker_s']))
    stats['backend_maximum_q_difference']=float(np.max(abs(fields['cpu','budget_fixed']['q']-fields['gpu','budget_fixed']['q'])))
    assert stats['backend_maximum_q_difference']<1e-10
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,b in zip(axes,('cpu','gpu')):
        values=[stats[b]['before'],stats[b]['after']]
        wasted=np.array([r['discarded_s'] for r in values]);steps=np.array([r['stepping_s'] for r in values])
        ax.bar(['Before','Budget check'],steps-wasted,label='Accepted work and step overhead')
        ax.bar(['Before','Budget check'],wasted,bottom=steps-wasted,label='Discarded hydraulic work',color='salmon')
        ax.set(title=b.upper()+' — first 0.75 simulated seconds',ylabel='Stepping wall time (s)')
        ax.legend(fontsize=8)
    fig.suptitle('Same accepted timesteps and fields; less work before unavoidable rejection')
    fig.savefig(OUT/'CONTROLLER_WORK.png',dpi=180);plt.close(fig)
    stats['saved_run']=saved
    (OUT/'SUMMARY.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    refs={}
    for folder,repo,commit in [('smallcell_amrex/439e6404','AMReX-Codes/amrex','439e6404dd2c28ffb0dad4647585942a9121b36c'),
        ('smallcell_geoflood/cf9a8234','KYANJO/GeoFlood','cf9a82345e77c614c2ccd93e57cf392af687b763')]:
        root=ROOT/'validation/vendor'/folder
        refs[repo]=dict(commit=commit,files={p.name:sha(p) for p in root.iterdir() if p.is_file()})
    (OUT/'REFERENCES.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
    sources=['braids_core/sediment/river_midpoint.py','tests/test_river_midpoint_retry_cap.py',
        'tools/audit_smallcell119.py','tools/analyze_smallcell119.py']
    with zipfile.ZipFile(OUT/'SOURCES.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    artifacts=[p for p in OUT.iterdir() if p.name!='MANIFEST.json']+[BASE/'BEFORE_CONTROLLER.zip']
    artifacts+=list(BASE.glob('*.xml'))+list(BASE.glob('*.json'))
    for name in ('cpu_baseline','cpu_budget_fixed','gpu_baseline','gpu_budget_fixed','gpu_preflight'):
        folder=BASE/name;artifacts+=list(folder.glob('*.json'))+list((folder/'run/frames').glob('*'))
    (OUT/'MANIFEST.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in artifacts if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in stats.items() if k!='saved_run'},indent=2))

if __name__=='__main__':main()
