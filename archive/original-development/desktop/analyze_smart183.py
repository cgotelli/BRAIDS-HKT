"""Public result comparison and figures; no smoothing or solver modifications."""
import json,sys
import numpy as np
from stage_smart183 import ROOT,OUT,TARGET,save
sys.path[:0]=[str(TARGET),str(ROOT)]
from braids_desktop.results import ResultsStore

REPORT=ROOT/'validation/sediment/river_morphodynamics/reports/desktop_smart183'

def compare(a,b,keys):
    a,b=ResultsStore(a),ResultsStore(b);assert np.array_equal(a.times,b.times)
    result={k:0. for k in keys}
    for i in range(len(a.times)):
        x,y=a.read(i),b.read(i)
        assert all(p.equals(q) for p,q in zip(x.polygons,y.polygons)) and len(x.polygons)==len(y.polygons)
        for k in keys:result[k]=max(result[k],float(abs(x.arrays[k]-y.arrays[k]).max(initial=0)))
    return result

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    regression={}
    keys=('q','z','mass','water','bed_boundary','bedload_vector_m2_s')
    for backend in ('cpu','gpu'):
        for law in ('mpm','ashida_michiue_generalized','wang_fredsoe'):
            name=law+'_'+backend
            diff=compare(OUT/'regression_baseline'/name/'full',OUT/'regression_candidate'/name/'full',keys)
            assert max(diff.values())==0.,(name,diff)
            regression[name]=diff
    pairs={}
    for tag in ('v3','dry_amr_v2','nonerodible'):
        pairs[tag]=compare(OUT/tag/'smart_jaggi_cpu/full',OUT/tag/'smart_jaggi_gpu/full',keys)
        # User's small-error allowance: 0.1 mm bed, below the earlier few-mm allowance.
        assert pairs[tag]['z']<1e-4 and pairs[tag]['q']<1e-4,(tag,pairs[tag])
    release={}
    for backend in ('cpu','gpu'):
        name='smart_jaggi_'+backend
        release[backend]=compare(OUT/'v3'/name/'full',OUT/'release'/name/'full',keys)
        assert max(release[backend].values())==0.,release
    save(REPORT/'COMPARISON.json',dict(existing_formula_maximum_difference=regression,smart_cpu_gpu=pairs,
        final_validation_guard_trajectory_difference=release,
        bed_acceptance_m=1e-4,initial_exploratory_10_micrometre_screen_passed=pairs['v3']['z']<1e-5))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import matplotlib.colors as colors
    cpu=ResultsStore(OUT/'v3/smart_jaggi_cpu/full');gpu=ResultsStore(OUT/'v3/smart_jaggi_gpu/full')
    zero=cpu.read(0).z;left=cpu.read(-1);right=gpu.read(-1)
    dz=(left.z-zero)*1000.;dg=(right.z-zero)*1000.
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    norm=colors.TwoSlopeNorm(vmin=-max(abs(dz).max(),abs(dg).max()),vcenter=0.,vmax=max(abs(dz).max(),abs(dg).max()))
    for ax,values,label in zip(axes,[dz,dg,dg-dz],['CPU bed change','GPU bed change','GPU minus CPU']):
        p=PolyCollection([np.asarray(p.exterior.coords) for p in left.polygons],array=values,cmap='RdBu_r',
            norm=norm if 'minus' not in label else colors.TwoSlopeNorm(vcenter=0.,vmin=-abs(dg-dz).max(),vmax=abs(dg-dz).max()),edgecolors='none')
        ax.add_collection(p);ax.autoscale_view();ax.set_aspect('equal');ax.set_title(label)
        ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');fig.colorbar(p,ax=ax,shrink=.8,label='mm')
    fig.suptitle('Smart & Jäggi: short steep-channel groin test at 16 s (sediment starts at 4 s)')
    fig.savefig(REPORT/'GROIN_COMPARISON.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label,store,ls in [('CPU',cpu,'-'),('GPU',gpu,'--')]:
        area=np.array([p.area for p in store.read(0).polygons]);scour=[];deposit=[];maximum=[];minimum=[]
        for i in range(len(store.times)):
            dz=store.read(i).z-zero
            scour.append(-np.minimum(dz,0).dot(area));deposit.append(np.maximum(dz,0).dot(area))
            maximum.append(dz.max()*1000);minimum.append(dz.min()*1000)
        axes[0].plot(store.times,scour,ls,label=label+' erosion',color='tab:blue')
        axes[0].plot(store.times,deposit,ls,label=label+' deposition',color='tab:orange')
        axes[1].plot(store.times,minimum,ls,label=label+' minimum',color='tab:blue')
        axes[1].plot(store.times,maximum,ls,label=label+' maximum',color='tab:orange')
    for ax in axes:ax.set_xlabel('Time (s)');ax.axvline(4,color='.5',lw=1);ax.axhline(0,color='.5',lw=.5);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Bulk bed volume (m³)');axes[1].set_ylabel('Bed change (mm)')
    fig.savefig(REPORT/'EVOLUTION.png',dpi=160);plt.close(fig)
    print(json.dumps(pairs,indent=2))
if __name__=='__main__':main()
