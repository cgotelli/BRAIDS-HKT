"""CPU/GPU Desktop integration figures, budgets and ordinary-route regression."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/hkt4';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/desktop_hkt4'


def frame(folder):
    p=folder/'frames';idx=json.loads((p/'INDEX.json').read_text());row=idx['frames'][-1]
    with np.load(p/row['file']) as d:q={k:d[k] for k in d.files}
    with np.load(p/row['epoch']) as d:mesh={k:d[k] for k in d.files}
    return q,mesh


def main():
    REPORT.mkdir(parents=True,exist_ok=True);metrics={}
    for kind in ['full','remesh']:
        ca=OUT/'public_release_cpu'/kind;ga=OUT/'public_release_gpu'/kind;a,ma=frame(ca);b,mb=frame(ga)
        for key in ['vertices','offsets','area','leaves']:np.testing.assert_array_equal(ma[key],mb[key])
        metrics[kind]=dict(maximum_backend_depth_m=float(np.max(abs(a['q'][0]-b['q'][0]))),maximum_backend_momentum=float(np.max(abs(a['q'][1:]-b['q'][1:]))),cells=len(ma['leaves']),minimum_area_m2=float(np.min(ma['area'])))
        for backend,folder in [('cpu',ca),('gpu',ga)]:
            diag=json.loads((folder/'DIAGNOSTICS.json').read_text());r=json.loads((folder/'TURBINE_PERFORMANCE.json').read_text())
            metrics[kind][backend]=dict(maximum_water_residual_m3=max(abs(z['water_residual_m3']) for z in diag),samples=len(r),finite=all(np.isfinite(z['thrust_N']) and np.isfinite(z['useful_power_W']) for z in r),final_thrust_N=r[-1]['thrust_N'],final_useful_power_W=r[-1]['useful_power_W'])
    a,_=frame(OUT/'ordinary_reference/run');b,_=frame(OUT/'ordinary_release/run');assert np.array_equal(a['q'],b['q'])
    metrics['ordinary_reference']=dict(q_byte_identical=True,maximum_field_difference=float(np.max(abs(a['q']-b['q']))),reference='Unmodified installed preview51 engine; separately constructed reference input')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for backend,color in [('cpu','tab:blue'),('gpu','tab:orange')]:
        rows=json.loads((OUT/('public_release_'+backend)/'full/TURBINE_PERFORMANCE.json').read_text());t=[z['time_s'] for z in rows]
        for ax,key,label in zip(axes.flat,['thrust_coefficient','power_coefficient','thrust_N','useful_power_W'],['Applied Ct','Applied Cp','Thrust (N)','Useful power (W)']):
            ax.plot(t,[z[key] for z in rows],color=color,ls='-' if backend=='cpu' else '--',label=backend.upper());ax.set(xlabel='Hydraulic time (s)',ylabel=label,ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Packaged Desktop integration: matched CPU/GPU, short transient control');fig.tight_layout();fig.savefig(REPORT/'CPU_GPU_PERFORMANCE.png',dpi=170);plt.close(fig)
    q,mesh=frame(OUT/'public_release_gpu/remesh');polys=[mesh['vertices'][a:b] for a,b in zip(mesh['offsets'][:-1],mesh['offsets'][1:])];wet=q['q'][0]>1e-6;u=np.divide(q['q'][1],q['q'][0],out=np.zeros_like(q['q'][0]),where=wet)
    fig,ax=plt.subplots(figsize=(11,5));pc=PolyCollection(polys,array=np.ma.masked_where(~wet,u),cmap='viridis',edgecolors='.4',linewidths=.2,clim=(0,.5));pc.cmap.set_bad('.88');ax.add_collection(pc)
    ax.plot([3.03,3.03],[1.65,2.35],color='red',lw=2.5,label='Rotor');ax.plot([2.68,3.38,3.38,2.68,2.68],[1.65,1.65,2.35,2.35,1.65],color='red',ls='--',lw=1,label='Finest force support')
    ax.set(xlim=(0,8),ylim=(0,4),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Accepted-time remesh: 2042 → 1874 cells; dry banks gray');ax.legend(fontsize=8);fig.colorbar(pc,ax=ax,label='Raw streamwise velocity (m/s)');fig.tight_layout();fig.savefig(REPORT/'REMESH_AND_FLOW.png',dpi=170);plt.close(fig)
    # Placement illustration only, not a claimed bend experiment.
    fig,ax=plt.subplots(figsize=(10,4));x=np.linspace(0,10,401);y=.6*np.sin(.65*x);ax.plot(x,y+1.,color='.6');ax.plot(x,y-1.,color='.6');ax.plot(x,y,'k--',lw=1,label='Local river direction / centreline')
    for x0 in [1.5,4.5,8.]:
        y0=.6*np.sin(.65*x0);n=np.array([1.,.39*np.cos(.65*x0)]);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]]);c=np.array([x0,y0]);a,b=c-.6*t,c+.6*t
        ax.plot([a[0],b[0]],[a[1],b[1]],color='teal',lw=4);ax.plot(*c,'o',color='teal');ax.annotate('',xy=c+.7*n,xytext=c,arrowprops=dict(arrowstyle='->',color='tab:orange',lw=2))
    ax.plot([],[],color='teal',lw=4,label='Rotor perpendicular to the selected local direction');ax.plot([],[],color='tab:orange',label='Second click points downstream')
    ax.set(aspect='equal',xlabel='x (illustrative)',ylabel='y (illustrative)',title='Placement rule in bends: click centre, then point along the river');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.24));fig.tight_layout();fig.savefig(REPORT/'PLACEMENT_IN_BENDS.png',dpi=170,bbox_inches='tight');plt.close(fig)
    for backend in ['cpu','gpu']:
        check=json.loads((OUT/('public_release_'+backend)/'CHECK.json').read_text());parent=OUT/'QUALIFIED_NUMERICAL_RUNTIME.json'
        assert check['runtime']==hashlib.sha256(parent.read_bytes()).hexdigest()
        before=json.loads(parent.read_text())['files'];after=json.loads((OUT/'candidate/RUNTIME.json').read_text())['files']
        assert set(before)==set(after) and [n for n in before if before[n]!=after[n]]==['braids_desktop/app.py']
        metrics['qualification_'+backend]=check
    metrics['ui_only_delivery_update']=dict(runtime=hashlib.sha256((OUT/'candidate/RUNTIME.json').read_bytes()).hexdigest(),
        numerical_qualification_runtime=hashlib.sha256((OUT/'QUALIFIED_NUMERICAL_RUNTIME.json').read_bytes()).hexdigest(),only_changed_payload='braids_desktop/app.py',numerical_payloads_identical=True)
    (REPORT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps({k:v for k,v in metrics.items() if not k.startswith('qualification')},indent=2))


if __name__=='__main__':main()
