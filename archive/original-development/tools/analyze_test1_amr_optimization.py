"""Frozen/current matched Test1 profiles and common finest-grid fields."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.sediment.adaptive_river_worker import hierarchy
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/amr_optimization112'
BASE=ROOT/'validation/_test1_amr_profile';NEW=ROOT/'validation/_test1_amr_optimization'
def read(p):return json.loads(p.read_text())
def frame(folder,row):
    with np.load(folder/row['epoch']) as f:mesh={k:f[k] for k in f.files}
    with np.load(folder/row['file']) as f:values={k:f[k] for k in f.files}
    return mesh,values
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tree=hierarchy(read(NEW/'combined_cpu/prepared/core/prepared/profile.json'))
    def series(root):
        folder=root/'run/frames';result=[]
        for row in read(folder/'INDEX.json')['frames']:
            mesh,v=frame(folder,row);leaves=mesh['leaves'];cover=tree.leaf_cover(leaves)[tree.offsets[-2]:]
            indices=np.searchsorted(leaves,cover)
            assert np.array_equal(leaves[indices],cover)
            result.append((row['time_s'],v['q'][:,indices],leaves))
        return result
    metrics={};states={};bars=[];names=[]
    for backend in ('cpu','gpu'):
        for tag,root in [('original',BASE/(backend+'_clean')),('optimized',NEW/('combined_'+backend))]:
            s=read(root/'PROFILE_SUMMARY.json');failed=[r for r in s['singles'] if r['error']]
            diag=read(root/'run/DIAGNOSTICS.json');events=read(root/'run/ADAPTATION.json')
            m=dict(worker_s=s['worker_call_s'],preparation_s=s['preparation_s'],runtime=s['runtime_identity'],
                phases=s['phases'],failed_attempts=len(failed),failed_s=sum(r['wall_s'] for r in failed),
                fwave_evaluations=sum(r['evaluations'] for r in s['singles']),
                accepted_substeps=sum(not r['error'] for r in s['singles']),
                initial_cells=diag[0]['cells'],final_cells=diag[-1]['cells'],
                mesh_changes=sum(e['changed'] for e in events),
                water_budget_max_m3=max(abs(r['water_residual_m3']) for r in diag),
                wall_max=max(max(r['wall_integrals']) for r in diag))
            key=backend+'_'+tag;metrics[key]=m;states[key]=series(root)
            p=s['phases'];bars.append([m['failed_s'],p['stepping']-m['failed_s'],p['amr'],
                p['load']+p['initialization']+p['other']+p['output']+p['checkpoint']]);names.append(key.replace('_',' '))
    area=np.array([c.area_m2 for c in tree.finest.cells]);comparisons={}
    pairs=[('cpu_original','cpu_optimized'),('gpu_original','gpu_optimized'),('cpu_optimized','gpu_optimized')]
    if (NEW/'half_clock_gpu/run/STATUS.json').is_file():
        states['gpu_half_clock']=series(NEW/'half_clock_gpu');pairs.append(('gpu_optimized','gpu_half_clock'))
    for left,right in pairs:
        rows=[]
        for (t,a,_),(s,b,_) in zip(states[left],states[right],strict=True):
            assert abs(t-s)<1e-10;wet=np.maximum(a[0],b[0])>1e-6;dh=b[0]-a[0]
            rows.append(dict(time_s=t,depth_max_mm=float(abs(dh).max()*1000),
                depth_wet_rms_mm=float(np.sqrt(np.average(dh[wet]**2,weights=area[wet]))*1000) if wet.any() else 0.,
                momentum_max_m2_s=float(abs(b[1:]-a[1:]).max()),wet_cells=int(wet.sum())))
        comparisons[left+'_vs_'+right]=rows
    metrics['comparisons']=comparisons
    portable={}
    for backend in ('cpu','gpu'):
        root=NEW/('portable_'+backend)
        if not (root/'PROFILE_SUMMARY.json').exists():continue
        s=read(root/'PROFILE_SUMMARY.json');got=series(root)
        err=max(float(abs(a[1]-b[1]).max()) for a,b in zip(got,states[backend+'_optimized'],strict=True))
        portable[backend]=dict(worker_s=s['worker_call_s'],preparation_s=s['preparation_s'],
            phases=s['phases'],runtime=s['runtime_identity'],
            speedup=metrics[backend+'_original']['worker_s']/s['worker_call_s'],
            checkout_field_max=err,hydraulic=s['result']['hydraulic'])
    metrics['installed_confirmation']=portable
    if len(portable)==2:
        fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
        labels=['CPU alpha111','CPU alpha112','GPU alpha111','GPU alpha112']
        values=[metrics['cpu_original']['worker_s'],portable['cpu']['worker_s'],
            metrics['gpu_original']['worker_s'],portable['gpu']['worker_s']]
        ax.barh(labels,values,color=['#9ba5bd','#287271','#9ba5bd','#287271'])
        for i,v in enumerate(values):ax.text(v+.4,i,f'{v:.2f} s',va='center')
        ax.set_xlim(0,max(values)*1.16);ax.set_xlabel('Public worker wall seconds; preparation excluded')
        ax.set_title('Installed QGIS runtimes: matched 5 m Test1, 0–8 s')
        fig.savefig(OUT/'INSTALLED_PERFORMANCE.png',dpi=165);plt.close(fig)
    metrics['speedup']={b:metrics[b+'_original']['worker_s']/metrics[b+'_optimized']['worker_s'] for b in ('cpu','gpu')}
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained');left=np.zeros(4);bars=np.array(bars)
    for j,(label,color) in enumerate(zip(['Failed attempts','Other stepping','AMR','Initialization / output / other'],['#bf4e30','#287271','#e9c46a','#9ba5bd'])):
        ax.barh(names,bars[:,j],left=left,label=label,color=color)
        left+=bars[:,j]
    for i,x in enumerate(left):ax.text(x+.3,i,f'{x:.1f} s',va='center')
    ax.set_xlim(0,max(left)*1.15);ax.set_xlabel('Wall seconds, 0–8 simulated seconds (preparation excluded)')
    ax.set_title('Test1: persistent hydraulic substeps + depth-only AMR + section-root reuse')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
    fig.savefig(OUT/'PERFORMANCE.png',dpi=165);plt.close(fig)
    origin=np.min(np.concatenate([c.vertices_xy for c in tree.finest.cells]),axis=0)
    polygons=[np.array(c.vertices_xy)-origin for c in tree.finest.cells]
    old=states['gpu_original'][-1][1];new=states['gpu_optimized'][-1][1]
    fig,axes=plt.subplots(1,3,figsize=(11,6),layout='constrained',sharex=True,sharey=True)
    delta=(new[0]-old[0])*1000;lim=max(1e-10,abs(delta).max());depthlim=max(old[0].max(),new[0].max())
    for ax,val,title,cmap,clim in zip(axes,[old[0],new[0],delta],['Original depth at 8 s','Optimized depth at 8 s','Optimized − original (mm)'],['Blues','Blues','RdBu_r'],[(0,depthlim),(0,depthlim),(-lim,lim)]):
        pc=PolyCollection(polygons,array=val,cmap=cmap,clim=clim,linewidth=0)
        ax.add_collection(pc);ax.autoscale();ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('Relative easting (m)')
        fig.colorbar(pc,ax=ax,fraction=.04,pad=.02)
    axes[0].set_ylabel('Relative northing (m)');fig.savefig(OUT/'WETTING_OVERVIEW.png',dpi=165)
    wet=np.maximum(old[0],new[0])>1e-6
    box=np.concatenate([p for p,w in zip(polygons,wet) if w])
    axes[0].set_xlim(box[:,0].min()-5,box[:,0].max()+5)
    axes[0].set_ylim(box[:,1].min()-5,box[:,1].max()+5)
    fig.savefig(OUT/'WETTING_COMPARISON.png',dpi=165);plt.close(fig)
    probe=read(NEW/'failure_probe_gpu/RESULT.json');fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for direction in ('acoustic','picard'):
        rows=[r for r in probe['probes'] if r['direction']==direction]
        ax.loglog([r['state_change'] for r in rows],[r['norm'] for r in rows],'o-',label=direction)
    ax.axhline(probe['norm'],color='grey',ls='--',label='Unperturbed residual')
    ax.axhline(probe['tolerance'],color='#287271',ls=':',label='Required tolerance')
    ax.set_xlabel('Norm of iterate perturbation');ax.set_ylabel('Nonlinear residual norm')
    ax.set_title('Frozen failure: abrupt wet/dry stencil transition near the inlet')
    ax.legend(fontsize=8);fig.savefig(OUT/'NONLINEAR_FAILURE.png',dpi=165);plt.close(fig)
    (OUT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(dict(speedup=metrics['speedup'],comparisons=comparisons),indent=2))
if __name__=='__main__':main()
