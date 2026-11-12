"""Measured hierarchy cost, per-rotor stationarity and backend figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/turbine_array55';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/array_hkt4d'


def read(path):return json.loads(Path(path).read_text())


def frame(folder,time=None):
    index=read(folder/'frames/INDEX.json');row=index['frames'][-1] if time is None else next(r for r in index['frames'] if abs(r['time_s']-time)<1e-10)
    with np.load(folder/'frames'/row['file']) as a:q={k:a[k].copy() for k in a.files}
    with np.load(folder/'frames'/row['epoch']) as a:m={k:a[k].copy() for k in a.files}
    return q,m


def keys(mesh):
    return [tuple(np.round([*(np.asarray(Polygon(mesh['vertices'][a:b]).centroid.coords[0])),area],8)) for a,b,area in zip(mesh['offsets'][:-1],mesh['offsets'][1:],mesh['area'])]


def main():
    REPORT.mkdir(parents=True,exist_ok=True);metrics={}
    baseline=read(OUT/'baseline_LEVELS_CLEAN_PROFILE.json');candidate=read(OUT/'candidate_LEVELS_CLEAN_PROFILE.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));cost={}
    for ax,level in zip(axes,[1,4]):
        names=['Previous off','Previous rotor','Current off','Current rotor'];rows=[baseline[f'level{level}_off'],baseline[f'level{level}_one'],candidate[f'level{level}_off'],candidate[f'level{level}_one']]
        a=np.array([r['geometry_s'] for r in rows]);b=np.array([r['turbine_mapping_s'] for r in rows]);ax.bar(range(4),a,color=['#94a3b8','#f97316','#94a3b8','#2563eb'],label='Geometry');ax.bar(range(4),b,bottom=a,color='#10b981',label='Turbine mapping')
        ax.set(xticks=range(4),xticklabels=names,ylabel='Wall time (s)',title=f'{level} level'+('s' if level>1 else ''),ylim=(0,19));ax.tick_params(axis='x',labelrotation=20);ax.grid(axis='y',alpha=.2)
        for i,total in enumerate(a+b):ax.text(i,total+.2,f'{total:.2f}',ha='center',fontsize=8)
        cost[str(level)]=dict(before_s=rows[1]['total_s'],after_s=rows[3]['total_s'],off_s=rows[2]['total_s'],speedup=rows[1]['total_s']/rows[3]['total_s'],finest_cells=rows[3]['finest_cells'])
    axes[0].legend();fig.suptitle('Clean geometry + mapping; same 0.10 m finest grid (36,000 cells)\nDEM integration, native records, output and hydraulic stepping excluded');fig.tight_layout();fig.savefig(REPORT/'PREPARATION_TIMINGS.png',dpi=160);plt.close(fig);metrics['preparation']=cost
    cpu=read(OUT/'candidate_array_cpu/run/TURBINE_PERFORMANCE.json');gpu=read(OUT/'candidate_array_gpu/run/TURBINE_PERFORMANCE.json');names=list(dict.fromkeys(r['turbine_id'] for r in cpu));steady={}
    fig,axes=plt.subplots(2,2,figsize=(12,8));quantities=[('upstream_velocity_m_s','Incoming reference velocity (m/s)'),('thrust_coefficient','Ct'),('power_coefficient','Cp'),('useful_power_W','Mechanical power (W)')]
    for number,name in enumerate(names):
        rows=[r for r in cpu if r['turbine_id']==name];tail=[r for r in rows if r['time_s']>=50.];comparison=[r for r in gpu if r['turbine_id']==name];stats={}
        for ax,(quantity,label) in zip(axes.flat,quantities):
            ax.plot([r['time_s'] for r in rows],[r[quantity] for r in rows],color=f'C{number}',label=name)
            ax.plot([r['time_s'] for r in comparison],[r[quantity] for r in comparison],ls='none',marker='x',ms=4,color=f'C{number}')
            values=np.array([r[quantity] for r in tail]);mean=float(values.mean());stats[quantity]=dict(mean=mean,peak_to_peak_percent=float(100*np.ptp(values)/abs(mean)),standard_deviation_percent=float(100*values.std()/abs(mean)))
            ax.set(xlabel='Physical time (s)',ylabel=label,ylim=(0,None));ax.grid(alpha=.25)
        force=np.array([r['thrust_N'] for r in tail]);stats['thrust_N']=dict(mean=float(force.mean()),peak_to_peak_percent=float(100*np.ptp(force)/force.mean()))
        steady[name]=stats
    for ax in axes.flat:ax.axvspan(50,60,color='gray',alpha=.12)
    axes[0,0].legend();fig.suptitle('Constant inlet, three independent models\nCPU: 0–60 s lines; GPU: matched startup crosses at 0–6 s; shaded stationarity window: 50–60 s');fig.tight_layout();fig.savefig(REPORT/'CONSTANT_INFLOW.png',dpi=160);plt.close(fig);metrics['stationarity_cpu_50_60_s']=steady
    paired=[]
    for time in sorted({r['time_s'] for r in gpu}):
        a,ma=frame(OUT/'candidate_array_cpu/run',time);b,mb=frame(OUT/'candidate_array_gpu/run',time)
        np.testing.assert_array_equal(ma['leaves'],mb['leaves']);np.testing.assert_array_equal(ma['area'],mb['area'])
        rows_a={r['turbine_id']:r for r in cpu if r['time_s']==time};rows_b={r['turbine_id']:r for r in gpu if r['time_s']==time}
        paired.append(dict(time_s=time,maximum_depth_m=float(np.max(abs(a['q'][0]-b['q'][0]))),maximum_power_relative=max(abs(rows_a[n]['useful_power_W']-rows_b[n]['useful_power_W'])/max(abs(rows_a[n]['useful_power_W']),1e-12) for n in names)))
    metrics['array_backend_startup']=paired
    old,mo=frame(OUT/'baseline_single_gpu/run');new,mn=frame(OUT/'candidate_single_gpu/run');ko=keys(mo);kn=keys(mn);lookup={k:i for i,k in enumerate(kn)};assert set(ko)==set(kn) and len(lookup)==len(kn);order=[lookup[k] for k in ko]
    metrics['frozen_single_regression']=dict(cells=len(ko),maximum_depth_m=float(np.max(abs(old['q'][0]-new['q'][0,order]))),maximum_momentum=float(np.max(abs(old['q'][1:]-new['q'][1:,order]))))
    fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['time_s'] for r in paired],[1e3*r['maximum_depth_m'] for r in paired],marker='o');ax.set(xlabel='Physical time (s)',ylabel='Maximum CPU/GPU depth difference (mm)',title='Three-rotor backend comparison over the shared startup window');ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));ax.grid(alpha=.25);fig.tight_layout();fig.savefig(REPORT/'BACKEND_STARTUP.png',dpi=160);plt.close(fig)
    doc=read(OUT/'candidate_array_cpu/input.json');fig,ax=plt.subplots(figsize=(12,5));x=np.linspace(0,30,300);y=np.linspace(0,12,120);z=500.+.001*(30-x[None,:])+np.minimum(2.5,np.maximum(abs(y[:,None]-6)-2,0)/1.5)
    image=ax.imshow(z,origin='lower',extent=(0,30,0,12),cmap='terrain',aspect='equal');origin=np.array([2600000.,1200000.])
    for row in doc['turbines']['turbines']:
        p=np.array(row['center_xy'])-origin;D=row['rotor_diameter_m'];ax.plot([p[0],p[0]],[p[1]-D/2,p[1]+D/2],lw=3,color='#dc2626');ax.add_patch(Rectangle(p-D/2,D,D,fill=False,ls='--',lw=.8));ax.annotate(row['id'],p,xytext=p+[1.3,1.2],arrowprops=dict(arrowstyle='->'),fontsize=8)
    ax.set(xlabel='Along channel (m)',ylabel='Across channel (m)',title='Constant-inflow array: 1 m rotors; 0.25 m finest cells; four mesh levels');fig.colorbar(image,ax=ax,label='Bed elevation (m)');fig.tight_layout();fig.savefig(REPORT/'ARRAY_LAYOUT.png',dpi=160);plt.close(fig)
    ra,rma=frame(OUT/'remesh_cpu/run');rb,rmb=frame(OUT/'remesh_gpu/run');np.testing.assert_array_equal(rma['leaves'],rmb['leaves'])
    metrics['remesh']=dict(cpu=read(OUT/'remesh_cpu/CHECK.json'),gpu=read(OUT/'remesh_gpu/CHECK.json'),maximum_backend_depth_m=float(np.max(abs(ra['q'][0]-rb['q'][0]))))
    assert metrics['remesh']['maximum_backend_depth_m']<1e-3
    polygons=[rmb['vertices'][a:b] for a,b in zip(rmb['offsets'][:-1],rmb['offsets'][1:])];wet=rb['q'][0]>1e-6
    for x in [3.03,5.03]:
        patch=Polygon([(x-.35,1.65),(x+.35,1.65),(x+.35,2.35),(x-.35,2.35)])
        ids=[i for i,p in enumerate(polygons) if Polygon(p).intersection(patch).area>1e-14]
        assert ids and np.all(rmb['levels'][ids]==1)
    history=read(OUT/'remesh_gpu/run/TURBINE_PERFORMANCE.json')
    for name in {r['turbine_id'] for r in history}:assert len({r['rotor_top_elevation_m'] for r in history if r['turbine_id']==name})==1
    metrics['remesh'].update(all_rotor_patches_finest=True,fixed_hubs_preserved=True)
    fig,ax=plt.subplots(figsize=(11,5));dry=PolyCollection([p for p,w in zip(polygons,wet) if not w],facecolors='#d1d5db',edgecolors='#94a3b8',linewidths=.2);ax.add_collection(dry)
    field=PolyCollection([p for p,w in zip(polygons,wet) if w],array=rb['q'][0,wet],cmap='Blues',edgecolors='#334155',linewidths=.2);ax.add_collection(field)
    for x in [3.03,5.03]:ax.plot([x,x],[1.65,2.35],color='red',lw=3)
    ax.set(xlim=(0,8),ylim=(0,4),aspect='equal',xlabel='Along channel (m)',ylabel='Across channel (m)',title='Two rotors after accepted-time remesh: 2,036 → 1,868 cells; dry banks gray');fig.colorbar(field,ax=ax,label='Raw water depth (m)');fig.tight_layout();fig.savefig(REPORT/'ARRAY_REMESH.png',dpi=160);plt.close(fig)
    for folder in ['candidate_array_cpu','candidate_array_gpu','baseline_single_gpu','candidate_single_gpu']:metrics[folder]=read(OUT/folder/'CHECK.json')
    (REPORT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (REPORT/'TURBINE_PERFORMANCE_EXAMPLE.csv').write_bytes((OUT/'candidate_array_cpu/run/TURBINE_PERFORMANCE.csv').read_bytes())
    assert max(r['maximum_depth_m'] for r in paired)<1e-3 and metrics['frozen_single_regression']['maximum_depth_m']<1e-3
    assert max(s['useful_power_W']['peak_to_peak_percent'] for s in steady.values())<1.
    print(json.dumps(dict(preparation=cost,steady=steady,single=metrics['frozen_single_regression'],maximum_backend_depth_m=max(r['maximum_depth_m'] for r in paired)),indent=2))


if __name__=='__main__':main()
