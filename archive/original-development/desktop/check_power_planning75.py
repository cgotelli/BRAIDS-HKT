"""Independent acceptance controls for the Desktop power-planning route."""
import copy,json,sys
import numpy as np
from power_planning75 import ROOT,TARGET,REPORT,read,save,environment
environment()
from braids_core.river_turbine_power_plan import METHOD,validate,evaluate,finite_amplitude
from braids_core.river_project import validate_project
from braids_core.river_model import resolve_model

CASES={'single':'single_high','tandem':'tandem_high6','staggered':'staggered_high'}
def document(case,backend='cpu'):
    d=read(ROOT/'desktop/qualification/selection74'/case/'input.json');layout=d.pop('turbines');d['turbines']=None
    for r in layout['turbines']:r['control']['wake']['method']='gaussian_shape_power_planning'
    d['power_plan']=dict(method=METHOD,layout=layout);d.pop('editor_state',None)
    d['model']['request']['execution']['backend']=backend;d['model']=resolve_model(d['model']['request']).document()
    d['settings'].update(end_time_s=20.,checkpoint_interval_s=20.,output_interval_s=2.)
    return validate_project(d)

def sample(U,H):
    def at(xy):
        n=len(xy);return np.full(n,H),np.zeros(n),np.full(n,U),np.zeros(n)
    return at

def checks():
    results=[]
    for old in read(ROOT/'validation/hydrokinetic_turbines/reports/selection74/METHODS.json'):
        d=document(old['case']);U=d['initial']['velocity_m_s'][0];H=d['initial']['stage_m'];bed={r['id']:0. for r in d['power_plan']['layout']['turbines']}
        row=evaluate(d,sample(U,H),0,bed)[-1];expected=old['methods']['finite_linear']['power_W']
        np.testing.assert_allclose(row['useful_power_W'],expected,rtol=1e-12,atol=1e-12)
        results.append(dict(case=old['case'],power_W=row['useful_power_W'],oracle_W=expected))
    d=document('staggered_high');bed={r['id']:0. for r in d['power_plan']['layout']['turbines']};H=d['initial']['stage_m']
    a=evaluate(d,sample(.75,H),0,bed);b=evaluate(d,sample(.85,H),1,bed)
    assert all(y['useful_power_W']>x['useful_power_W'] for x,y in zip(a,b))
    assert all(not r['active'] and r['useful_power_W']==0 for r in evaluate(d,sample(.8,.5),2,bed))
    assert all(r['active'] for r in evaluate(d,sample(.812,H),3,bed))
    # Independent finite cross-section quadrature, not another invocation of the helper.
    sig=.238;ct=.89;D=.27;H=.45;W=5.;a0=finite_amplitude(ct,D,sig,H,H/2,W,W/2)
    y,wy=np.polynomial.legendre.leggauss(180);z,wz=np.polynomial.legendre.leggauss(100)
    deficit=a0*np.exp(-.5*(((y[:,None]*W/2)/sig)**2+((z[None,:]*H/2)/sig)**2))
    np.testing.assert_allclose((wy*W/2)@(deficit*(1-deficit))@(wz*H/2),.5*ct*np.pi*D**2/4,rtol=1e-12)
    invalid=[]
    for label,mutate in [('disturbed',lambda x:x.update(turbines=x['power_plan']['layout'])),('yaw',lambda x:x['power_plan']['layout']['turbines'][1].update(river_direction_xy=[1,.2])),('missing_surface',lambda x:x['power_plan']['layout']['turbines'][0].update(control=dict(mode='constant_ct',thrust_coefficient=.8,power_coefficient=.4))),('amr',lambda x:x['settings'].update(adaptive=True))]:
        dd=copy.deepcopy(d);mutate(dd)
        try:validate(dd['power_plan'],dd)
        except ValueError:invalid.append(label)
        else:raise AssertionError(label)
    save(REPORT/'OPERATORS.json',dict(passed=True,oracle_comparisons=results,varying_flow=True,coverage_off_on=True,finite_momentum_integral=True,rejections=invalid))
    print('All 8 archived estimates reproduced; varying flow, coverage and rejection controls passed.')

if __name__=='__main__':checks()
