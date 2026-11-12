"""Compare AMR clocks on identical Test1 hydraulic inputs and finest cover."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from braids_core.sediment.adaptive_river_worker import hierarchy
from braids_gpu.atomic_output import write_json_atomic

BASE=ROOT/'validation/_test1_amr_optimization'
NEW=ROOT/'validation/_test1_amr_interval'
OUT=ROOT/'validation/sediment/river_morphodynamics/reports/unified_qgis_v1/amr_interval113'
def read(p):return json.loads(p.read_text())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tree=hierarchy(read(BASE/'portable_cpu/prepared/core/prepared/profile.json'))
    area=np.array([c.area_m2 for c in tree.finest.cells]);states={};metrics={}
    paths={b+'_baseline':BASE/('portable_'+b) for b in ('cpu','gpu')}
    paths.update({p.parent.name:p.parent for p in NEW.glob('*/PROFILE_SUMMARY.json')})
    for name,path in paths.items():
        summary=read(path/'PROFILE_SUMMARY.json')
        if summary['result'] is None or summary['result']['status'] not in ('complete','segment_complete'):continue
        folder=path/'run/frames';series=[];coarse_wet=[]
        for row in read(folder/'INDEX.json')['frames']:
            with np.load(folder/row['epoch']) as f:leaves=f['leaves']
            with np.load(folder/row['file']) as f:q=f['q']
            coarse_wet.append(int(np.count_nonzero((q[0]>1e-6)&(tree.node_level[leaves]<tree.finest_level))))
            cover=tree.leaf_cover(leaves)[tree.offsets[-2]:];ids=np.searchsorted(leaves,cover)
            assert np.array_equal(leaves[ids],cover)
            series.append((row['time_s'],q[:,ids]))
        states[name]=series
        events=read(path/'run/ADAPTATION.json');diag=read(path/'run/DIAGNOSTICS.json')
        metrics[name]=dict(worker_s=summary['worker_call_s'],preparation_s=summary['preparation_s'],
            phases=summary['phases'],runtime=summary['runtime_identity'],hydraulic=summary['result']['hydraulic'],
            assessments=len(events),remeshes=sum(r['changed'] for r in events),
            water_residual_max_m3=max(abs(r['water_residual_m3']) for r in diag),
            wall_flux_max=max(max(abs(x) for x in r['wall_integrals']) for r in diag),
            saved_coarse_wet_cells_max=max(coarse_wet),
            accepted_step_coarse_wet_max=max([r['coarse_wet_cells'] for r in summary['steps'] if 'coarse_wet_cells' in r],default=None),
            interval=read(path/'PROJECT.json')['settings']['adaptation']['interval'])
    def compare(left,right):
        rows=[]
        for (t,a),(s,b) in zip(states[left],states[right],strict=True):
            assert abs(t-s)<1e-10
            wet=np.maximum(a[0],b[0])>1e-6;dh=b[0]-a[0]
            rows.append(dict(time_s=t,depth_max_mm=float(abs(dh).max()*1000),
                wet_rms_mm=float(np.sqrt(np.average(dh[wet]**2,weights=area[wet]))*1000) if wet.any() else 0.,
                momentum_max_m2_s=float(abs(b[1:]-a[1:]).max())))
        return rows
    comparisons={}
    for name in states:
        if name.endswith('baseline'):continue
        backend=name.split('_')[0]
        comparisons[name]=compare(backend+'_baseline',name)
        metrics[name]['speedup']=metrics[backend+'_baseline']['worker_s']/metrics[name]['worker_s']
        if backend=='gpu' and name.replace('gpu_','cpu_',1) in states:
            comparisons[name+'_backend']=compare(name.replace('gpu_','cpu_',1),name)
        if name.endswith('installed_1s'):
            comparisons[name+'_vs_checkout']=compare(backend+'_guard2_1s',name)
    for name in ('cpu_guard2_halfdt','cpu_guard2_2s'):
        if name in states:comparisons['corrected_1s_vs_'+name]=compare('cpu_guard2_1s',name)
    write_json_atomic(OUT/'METRICS.json',dict(runs=metrics,comparisons=comparisons))
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
    selected=[n for n in ('cpu_baseline','cpu_installed_1s','gpu_baseline','gpu_installed_1s') if n in metrics]
    if len(selected)<4:selected=[n for n in ('cpu_baseline','cpu_guard2_1s','cpu_guard2_2s','gpu_baseline','gpu_guard2_1s') if n in metrics]
    names=selected;left=np.zeros(len(names))
    labels={'cpu_baseline':'CPU alpha112 (0.05 s)','gpu_baseline':'GPU alpha112 (0.05 s)',
        'cpu_installed_1s':'CPU alpha113 (1 s)','gpu_installed_1s':'GPU alpha113 (1 s)',
        'cpu_guard2_1s':'CPU corrected (1 s)','cpu_guard2_2s':'CPU corrected (2 s)',
        'gpu_guard2_1s':'GPU corrected (1 s)','cpu_guard2_halfdt':'Corrected 1 s, half outer dt',
        'gpu_1s':'1 s without entrance buffer'}
    for key,label,color in [('amr','AMR','#e9c46a'),('stepping','Hydraulics / physics','#287271'),('other_all','Other','#9ba5bd')]:
        vals=[metrics[n]['phases'][key] if key!='other_all' else metrics[n]['worker_s']-metrics[n]['phases']['amr']-metrics[n]['phases']['stepping'] for n in names]
        ax.barh([labels[n] for n in names],vals,left=left,label=label,color=color);left+=vals
    ax.set_xlabel('Worker wall time (s), 0–8 simulated seconds; preparation excluded')
    for i,value in enumerate(left):ax.text(value+.15,i,f'{value:.2f} s',va='center')
    ax.set_xlim(0,max(left)*1.15)
    ax.set_title('Test1: less frequent depth-only AMR');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3)
    fig.savefig(OUT/'PERFORMANCE.png',dpi=160);plt.close(fig)
    for corrected in (False,True):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for name,rows in comparisons.items():
            if name not in ('gpu_1s','gpu_guard2_1s','cpu_guard2_2s','cpu_guard2_halfdt') or (corrected and name=='gpu_1s'):continue
            axes[0].plot([r['time_s'] for r in rows],[r['depth_max_mm'] for r in rows],'.-',label=labels[name])
            axes[1].plot([r['time_s'] for r in rows],[r['wet_rms_mm'] for r in rows],'.-',label=labels[name])
        for ax,label in zip(axes,['Maximum depth difference (mm)','Wet-area RMS depth difference (mm)']):
            ax.set_xlabel('Physical time (s)');ax.set_ylabel(label);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(OUT/('CORRECTED_DEPTH.png' if corrected else 'DEPTH_COMPARISON.png'),dpi=160);plt.close(fig)
    print(json.dumps({n:dict(worker_s=m['worker_s'],assessments=m['assessments'],remeshes=m['remeshes'],
        max_mm=max([r['depth_max_mm'] for r in comparisons.get(n,[])],default=0),
        rms_mm=max([r['wet_rms_mm'] for r in comparisons.get(n,[])],default=0),
        coarse_wet=m['saved_coarse_wet_cells_max']) for n,m in metrics.items()},indent=2))
if __name__=='__main__':main()
