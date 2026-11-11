"""Research189: existing Gaussian incoming power + explicit device yaw response.

Not an experimental yaw-array validation or a new resolved wake force.
The hydraulic source stays rotor-normal; only shutdown depends on the new
power operating interval. Wake width remains an independent supplied input.
"""
import math
import numpy as np
from . import river_turbine_device_yaw as device
METHOD='power.gaussian_device_yaw.v1'
ASSUMPTIONS=('straight quasi-uniform reach; H/D>=2; recovered-pressure Gaussian far wake; '
 'actual streamwise actuator force sets wake amplitude; unchanged supplied wake growth; '
 'no yaw steering; additive deficits; local incident direction and reconstructed magnitude; '
 'uniform-inflow device yaw ratio applied to local mean inflow; conditional yaw-array prediction')

def validate(setting):
    from .river_turbine_power import validate as validate_wake, METHOD as WAKE
    if set(setting)!={'method','wake','device'} or setting['method'] not in (METHOD,'power.dou2019_device_yaw.v1'):
        raise ValueError('Combined power needs explicit wake parameters and a matching device yaw table.')
    if not isinstance(setting['wake'],dict) or setting['wake'].get('method')!=WAKE:raise ValueError('Select the qualified Gaussian incoming-power method.')
    if setting['device'] is None:
        raise ValueError('Dou array power needs a device yaw-performance table. In Hydrokinetic turbines, import the matching device JSON (diameter, RPM, speed/yaw axes and power ratios). Dou describes the wake; it does not supply an arbitrary rotor power curve.')
    return dict(method=setting['method'],wake=validate_wake(setting['wake']),device=device.validate(setting['device']))

def flow_axis(rotor,normal):
    a=np.asarray(rotor.original.get('river_direction_xy',normal),float)
    if a.shape!=(2,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-10:raise ValueError('Supply a valid ambient river direction for wake propagation.')
    return a/np.linalg.norm(a)

def stream_ct(rotor,s,axis):
    if not s['active']:return 0.
    if rotor.original.get('orientation',{}).get('mode')!='fixed_axis':return s['thrust_coefficient']
    force=-(s['force_x_per_density_m4_s2']*axis[0]+s['force_y_per_density_m4_s2']*axis[1])
    U=s['upstream_velocity_m_s'];area=math.pi*rotor.c['D']**2/4
    ct=force/(.5*area*U**2)
    if ct<0:raise ValueError('Reverse fixed-axis wake loading is unsupported.')
    return ct

def incoming(rotor,h,u,v,n,xp):
    if rotor.original.get('orientation',{}).get('mode')!='fixed_axis':
        from .river_turbine_thetis import cell_terms
        _,_,U,_,_=cell_terms(rotor.control,rotor.c['D'],h,u,v,xp)
        return U,xp.zeros_like(U)
    from .river_turbine_yaw import yaw_terms
    un=u*n[0]+v*n[1];ut=-u*n[1]+v*n[0]
    _,_,U,_,_=yaw_terms(rotor.control,rotor.c['D'],h,un,ut,xp)
    angle=xp.degrees(xp.arctan2(ut,xp.sqrt(xp.maximum(U**2-ut**2,0.))))
    return U,angle

def effective_cp(rotor,U,angle,xp):
    from .river_turbine_power import cp_values
    if not getattr(rotor,'device_yaw',False):return cp_values(rotor.control,U,xp)
    if bool(xp.any(U<rotor.device_setting['speed_m_s'][0])):return None
    from .river_turbine_thetis import curve_rows
    if rotor.control['mode']!='constant_ct' and bool(xp.any(U<curve_rows(rotor.control,'power_coefficient')[0]['speed_m_s'])):return None
    return cp_values(rotor.control,U,xp)*device.lookup(rotor.device_setting,U,angle,xp)

def stop(s,parents):
    s.update(active=False,inactive_reason='below_cut_in_speed',rotational_speed_rpm=0.,
        thrust_coefficient=0.,power_coefficient=0.,force_per_density_m4_s2=0.,
        force_x_per_density_m4_s2=0.,force_y_per_density_m4_s2=0.,
        useful_power_per_density_m5_s3=0.,hydraulic_power_removed_per_density_m5_s3=0.,
        reconstruction_active=True,active_upstream_wakes=parents)
