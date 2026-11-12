"""Raw-cell comparison and bounded curve-driven performance evidence."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'desktop/qualification/decimal_mesh54';REPORT=ROOT/'validation/hydrokinetic_turbines/reports/power_mesh_hkt4c'
sys.path.insert(0,str(OUT/'candidate'))
from braids_core.river_turbine_performance import operating_point


def polygons(path):
    with np.load(path) as a:xy=a['vertices'].copy();offsets=a['offsets'].copy()
    return [xy[i:j] for i,j in zip(offsets[:-1],offsets[1:])]


def counts(rows):
    square=0
    for xy in rows:
        w=np.ptp(xy,axis=0);delta=np.diff(xy,axis=0)
        square+=int(abs(w[0]-w[1])<1e-8 and np.all(np.min(np.abs(delta),axis=1)<1e-8))
    return dict(cells=len(rows),squares=square,non_square=len(rows)-square)


def last(path):
    r=json.loads((path/'frames/INDEX.json').read_text())['frames'][-1]
    with np.load(path/'frames'/r['file']) as a:q={k:a[k].copy() for k in a.files}
    with np.load(path/'frames'/r['epoch']) as a:mesh={k:a[k].copy() for k in a.files}
    return q,mesh


def main():
    REPORT.mkdir(parents=True,exist_ok=True);metrics={};origin=np.array([2600000.,1200000.])
    old=polygons(OUT/'user_snapshot/OLD_MESH_PREVIEW.npz');new=polygons(OUT/'qualified_cpu/off/prepared/MESH_PREVIEW.npz')
    metrics['mesh_before']=counts(old);metrics['mesh_after']=counts(new);assert metrics['mesh_after']['non_square']==0
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,rows,label in zip(axes,[old,new],['Before: false area-triggered unions','After: complete squares retained']):
        lines=[p-origin for p in rows if np.any((p[:,0]>origin[0])&(p[:,0]<origin[0]+3)&(p[:,1]>origin[1]+4.5)&(p[:,1]<origin[1]+7.5))]
        ax.add_collection(LineCollection(lines,colors='#334155',linewidths=.6));ax.set(xlim=(0,3),ylim=(4.5,7.5),aspect='equal',title=label,xlabel='Along channel (m)',ylabel='Across channel (m)')
    fig.suptitle('User channel, 0.10 m finest / 0.20 m coarse; turbines disabled in both');fig.tight_layout();fig.savefig(REPORT/'MESH_COMPARISON.png',dpi=160);plt.close(fig)
    for case in ['off','on','rpm']:
        a,ma=last(OUT/'qualified_cpu'/case/'run');b,mb=last(OUT/'qualified_gpu'/case/'run')
        for k in ['area','vertices','offsets','leaves']:np.testing.assert_array_equal(ma[k],mb[k])
        data=dict(cells=len(ma['area']),minimum_area_m2=float(np.min(ma['area'])),maximum_depth_difference_m=float(np.max(abs(a['q'][0]-b['q'][0]))))
        assert data['minimum_area_m2']>=.01-1e-9 and data['maximum_depth_difference_m']<1e-4
        data['cpu']=json.loads((OUT/'qualified_cpu/CHECK.json').read_text())[case];data['gpu']=json.loads((OUT/'qualified_gpu/CHECK.json').read_text())[case];metrics[case]=data
    doc=json.loads((OUT/'qualified_cpu/rpm/input.json').read_text());control=doc['turbines']['turbines'][0]['control'];vel=np.linspace(.1,3.,220);records=[]
    for speed in vel:
        p=operating_point(control,1.,2.,float(speed));records.append(p)
    active=np.array([p['active'] for p in records]);U=np.array([p.get('upstream_velocity_m_s',v) for p,v in zip(records,vel)])
    ct=np.array([p.get('ct',0) for p in records]);cp=np.array([p.get('cp',0) for p in records]);power=.5*1000*np.pi/4*cp*U**3
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));axes[0].plot(vel,ct,label='Computed Ct');axes[0].plot(vel,cp,label='Computed Cp');axes[0].set(ylabel='Coefficient',ylim=(0,1));axes[0].legend()
    axes[1].plot(vel,power/1000);axes[1].set(ylabel='Mechanical power (kW)',ylim=(0,None))
    for ax in axes:ax.set(xlabel='Local normal model velocity (m/s)');ax.grid(alpha=.25)
    fig.suptitle('Synthetic curve verification: 1 m rotor, 90 RPM, 2 m depth\nDemonstration data; not a real-turbine power forecast');fig.tight_layout();fig.savefig(REPORT/'POWER_MODEL_RESPONSE.png',dpi=160);plt.close(fig)
    # Confirm the earlier binary-width example's hierarchy is byte-for-byte
    # unchanged. Only a false threshold crossing on decimal widths is repaired.
    from braids_core.boundaries import breaklines
    from braids_core.boundaries.domain import PolygonDomain,BoundaryAssignment
    from braids_core.boundaries.inlet_io import NativeGrid
    from braids_core.river_project import worker_profile
    from braids_desktop.preparation import encode_tree
    frozen=ROOT/'desktop/qualification/turbine_clearance53/candidate/braids_core/boundaries/breaklines.py'
    spec=importlib.util.spec_from_file_location('braids_core.boundaries.frozen_breaklines53',frozen);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    baseline=json.loads((ROOT/'desktop/qualification/turbine_clearance53/channel_template/input.json').read_text());p=worker_profile(baseline)
    args=(PolygonDomain(**p['domain']),[BoundaryAssignment(**a) for a in p['assignments']],NativeGrid(**p['grid']),p['breaklines'])
    assert encode_tree(module.build_breakline_hierarchy(*args,levels=p['levels']))==encode_tree(breaklines.build_breakline_hierarchy(*args,levels=p['levels']))
    metrics['original_quarter_metre_hierarchy_exact']=True
    (REPORT/'METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (REPORT/'TURBINE_PERFORMANCE_EXAMPLE.csv').write_bytes((OUT/'qualified_cpu/rpm/run/TURBINE_PERFORMANCE.csv').read_bytes())
    print(json.dumps({k:v for k,v in metrics.items() if k.startswith('mesh') or isinstance(v,bool)},indent=2))


if __name__=='__main__':main()
