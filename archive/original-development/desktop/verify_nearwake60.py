"""Matched CPU/GPU and physical control checks, independent of wake fitting."""
from analyze_nearwake60 import *


def frame(name,time):
    row=read(REPORT/name/'STARTED.json');store=ResultsStore(Path(row['output']))
    ids=np.flatnonzero(abs(store.times-time)<1e-8)
    if len(ids)!=1:raise ValueError(f'{name}: missing {time}s')
    return store.read(int(ids[0]))


def backend():
    a,b=frame('wake4_cpu',100.),frame('wake4_gpu',100.)
    np.testing.assert_array_equal(a.mesh['area'],b.mesh['area'])
    np.testing.assert_array_equal(np.array([p.centroid.coords[0] for p in a.polygons]),
                                  np.array([p.centroid.coords[0] for p in b.polygons]))
    result=dict(time_s=100.,cells=len(a.polygons),
        maximum_depth_m=float(np.max(abs(a.q[0]-b.q[0]))),
        maximum_velocity_m_s=float(np.max(abs(a.q[1:]/a.q[0]-b.q[1:]/b.q[0]))),
        maximum_discharge_m2_s=float(np.max(abs(a.q[1:]-b.q[1:]))),
        scope='Matched refined-wake startup, not a complete 2000s CPU wake reproduction.')
    assert result['maximum_depth_m']<1e-6 and result['maximum_velocity_m_s']<1e-6
    save(REPORT/'CPU_GPU.json',result);print(json.dumps(result,indent=2))


def controls():
    result={};old=cached('zero_gpu');x=np.linspace(-2,25,109);y=np.linspace(-3,3,25)
    xx,yy=np.meshgrid(x,y)
    for name in ['zero4_gpu','zero8_corrected']:
        receipt=read(REPORT/name/'RECEIPT.json');assert receipt['status']=='complete'
        a=model(name,400,500);first=model(name,300,400)
        s=sample(a,xx,yy);s0=sample(first,xx,yy);o=sample(old,xx,yy)
        result[name]=dict(average_times_s=a['times'].tolist(),
            maximum_speed_window_change_m_s=float(np.max(abs(s-s0))),
            maximum_control_effect_on_deficit_pp=float(100*np.max(abs(s-s0))/np.min(s)),
            maximum_speed_difference_old_steady_control_m_s=float(np.max(abs(s-o))),
            maximum_old_control_effect_pp=float(100*np.max(abs(s-o))/np.min(s)))
        result[name]['strict_0p02pp_control_stationarity_pass']=result[name]['maximum_control_effect_on_deficit_pp']<.02
        # Keep the original stringent screen even if it fails. Report sensitivity
        # against the completed historical steady control instead of disguising
        # a small background transient as a fully stationary normalization.
        assert np.isfinite(s).all() and np.min(s)>0
        # No matching field is substituted in the final analysis: this merely
        # bounds the shorter steady zero-thrust normalization window.
    save(REPORT/'CONTROL_CHECK.json',result);print(json.dumps(result,indent=2))


def initial():
    result={}
    for name in ['wake4_gpu','wake8_gpu','wake4_cpu','zero4_gpu','zero8_gpu','zero8_corrected']:
        f=frame(name,0.)
        row=read(REPORT/name/'STARTED.json');project=read(Path(row['output'])/'PROJECT.json')
        area=np.asarray(f.mesh['area']);h=f.q[0];u=f.q[1]/h;v=f.q[2]/h
        result[name]=dict(cells=len(area),area_m2=float(area.sum()),volume_m3=float(np.dot(h,area)),
            maximum_depth_departure_m=float(np.max(abs(h-20.))),maximum_velocity_departure_m_s=float(max(np.max(abs(u-.2)),np.max(abs(v)))),
            hydraulic_boundaries=project['settings']['hydraulics'],rotor=project['turbines']['turbines'][0],
            shared_ambient_viscosity_m2_s=project['turbines']['ambient_viscosity_m2_s'])
        assert abs(result[name]['area_m2']-900000.)<1e-7
        assert result[name]['maximum_depth_departure_m']<1e-12 and result[name]['maximum_velocity_departure_m_s']<1e-12
    save(REPORT/'INITIAL_CONDITIONS.json',result);print('All six cases have the same physical initial water state and domain area.')


def status():
    out={}
    for p in REPORT.glob('*/STARTED.json'):
        row=read(p);progress=read(Path(row['output'])/'PROGRESS.json')
        out[p.parent.name]={k:progress.get(k) for k in ['status','phase','time_s']}
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['backend','controls','initial','status']);a=p.parse_args();globals()[a.action]()
