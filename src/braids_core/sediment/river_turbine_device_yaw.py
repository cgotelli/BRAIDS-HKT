"""Research188: device-specific yaw power ratio, computed offline from BEM.

No generic turbine geometry, controller or yaw response is inferred. The
runtime interpolates a provenance-bearing table; no MATLAB dependency, BEM
solve, empirical cosine multiplier or extra wake deficit in the timestep.
"""
import math
import numpy as np

METHOD='power.device_bem_yaw_ratio.v1'

def validate(value):
    if not isinstance(value,dict) or value.get('method')!=METHOD:
        raise ValueError('Select an explicit device yaw-performance table.')
    for key in ('model_name','provenance','physical_scope'):
        if not isinstance(value.get(key),str) or not value[key].strip():
            raise ValueError(f'Device yaw table requires {key}.')
    for key in ('diameter_m','rpm'):
        if type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) or value[key]<=0:
            raise ValueError(f'Device yaw table requires a positive {key}.')
    s=np.asarray(value.get('speed_m_s'),float);a=np.asarray(value.get('yaw_deg'),float)
    p=np.asarray(value.get('power_ratio'),float)
    if (s.ndim!=1 or a.ndim!=1 or len(s)<2 or len(a)<2 or
        not np.isfinite(s).all() or not np.isfinite(a).all() or
        s[0]<=0 or a[0]!=0 or a[-1]>=90 or
        np.any(np.diff(s)<=0) or np.any(np.diff(a)<=0) or
        p.shape!=(len(s),len(a)) or not np.isfinite(p).all() or
        np.any(p<0) or np.any(p[:,0]!=1)):
        raise ValueError('Device yaw table needs increasing speed and yaw axes, nonnegative finite ratios, and exactly unit ratio at zero yaw.')
    return dict(value)

def lookup(value,speed,yaw_deg,xp=np):
    """Bilinear interpolation; reject uncovered states, never clamp/extrapolate."""
    s=xp.asarray(value['speed_m_s']);a=xp.asarray(value['yaw_deg']);p=xp.asarray(value['power_ratio'])
    angle=xp.abs(yaw_deg)
    if bool(xp.any(~xp.isfinite(speed) | ~xp.isfinite(angle) | (speed<s[0]) | (speed>s[-1]) | (angle>a[-1]+1e-12))):
        raise ValueError(f"Device yaw performance is outside its supplied coverage ({value['speed_m_s'][0]:g}-{value['speed_m_s'][-1]:g} m/s; |yaw| <= {value['yaw_deg'][-1]:g} deg). Actual speed {float(xp.min(speed)):g}-{float(xp.max(speed)):g} m/s; maximum |yaw| {float(xp.max(angle)):g} deg. Supply matching blade/controller data; no extrapolation is allowed.")
    i=xp.minimum(xp.searchsorted(s,speed,side='right')-1,len(s)-2)
    j=xp.minimum(xp.searchsorted(a,angle,side='right')-1,len(a)-2)
    f=(speed-s[i])/(s[i+1]-s[i]);g=(angle-a[j])/(a[j+1]-a[j])
    q=(1-f)*((1-g)*p[i,j]+g*p[i,j+1])+f*((1-g)*p[i+1,j]+g*p[i+1,j+1])
    return xp.where(angle==0,1.,q)

def correct(value,power,incident,ut,cp,xp=np):
    cosine=xp.sqrt(xp.maximum(1-(ut/incident)**2,0.))
    angle=xp.degrees(xp.arctan2(ut,incident*cosine))
    factor=lookup(value,incident,angle,xp)
    # Existing source power uses Cp0 * (U cos gamma)^3. Replace its yaw law
    # once, with Cp0 * U^3 * BEM_yaw/BEM_zero, retaining the same local input.
    return power*factor/cosine**3,cp*factor
