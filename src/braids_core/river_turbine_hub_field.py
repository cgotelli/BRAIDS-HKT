"""Diagnostic horizontal plane of the coupled model, not a 3-D flow solution.

The resolved depth mean is retained. Only the model's zero-column-mean wake
departure is added. Own-induction is an incoming-reference conversion at a
receiver and is deliberately NOT added to the physical velocity field.
"""
import math
import numpy as np
from scipy.special import erf

METHOD='diagnostic.coupled_hub_plane_zero_mean.v1'

def plane_field(xy,bed,depth,velocity,rotors,samples,elevation):
    xy=np.asarray(xy,float);bed=np.asarray(bed,float);depth=np.asarray(depth,float)
    velocity=np.asarray(velocity,float)
    if xy.shape!=(len(bed),2) or velocity.shape!=xy.shape or depth.shape!=bed.shape:
        raise ValueError('Hub plane requires matching cell coordinates, bed, depth and horizontal velocity.')
    if not np.isfinite(elevation):raise ValueError('Finite hub elevation required.')
    by_id={s['turbine_id']:s for s in samples}
    if len(by_id)!=len(samples) or set(by_id)!={r['id'] for r in rotors}:
        raise ValueError('Hub plane needs exactly one simultaneous sample per turbine.')
    valid=(depth>1e-6)&(elevation>bed)&(elevation<bed+depth)
    result=velocity.copy();near=np.zeros(len(bed),bool)
    H=np.maximum(depth,1e-6)
    for r in rotors:
        s=by_id[r['id']];wake=r['control'].get('wake')
        if r['control']['mode']!='fixed_rpm_surface' or not wake or wake['method'] not in ('gaussian_zero_mean_single_wake','gaussian_zero_mean_linear_overlap'):
            raise ValueError('Hub reconstruction requires the coupled device-surface wake model; no wake parameters are invented for constant-Ct controls.')
        if not s['active']:continue
        n=np.asarray(r['river_direction_xy'],float);n/=np.linalg.norm(n);t=np.array([-n[1],n[0]])
        delta=xy-np.asarray(r['center_xy']);D=r['rotor_diameter_m'];xD=delta@n/D;y=delta@t
        sigma=D*(wake['growth']*xD+wake['initial_width'])
        near|=(xD>=0)&(xD<wake['minimum_distance_D'])&(abs(y)<=D/2+4*np.maximum(sigma,0))
        mask=valid&(xD>=wake['minimum_distance_D'])&(sigma>0)&(abs(y)<=4*sigma)
        if not mask.any():continue
        ids=np.flatnonzero(mask);sig=sigma[ids];hub=float(s['rotor_top_elevation_m'])-D/2
        rad=1-float(s['thrust_coefficient'])/(8*(sig/D)**2)
        ok=(rad>0)&(hub>bed[ids])&(hub<bed[ids]+H[ids])
        valid[ids[~ok]]=False;ids=ids[ok];sig=sigma[ids];rad=rad[ok]
        amp=float(s['upstream_velocity_m_s'])*(1-np.sqrt(rad))*np.exp(-.5*(y[ids]/sig)**2)
        rel=hub-bed[ids];mean=sig*np.sqrt(np.pi/2)/H[ids]*(erf((H[ids]-rel)/(np.sqrt(2)*sig))+erf(rel/(np.sqrt(2)*sig)))
        anomaly=amp*(mean-np.exp(-.5*((elevation-hub)/sig)**2))
        result[ids]+=anomaly[:,None]*n
    valid&=~near&np.isfinite(result).all(axis=1)
    result[~valid]=np.nan
    return dict(velocity_xy_m_s=result,speed_m_s=np.linalg.norm(result,axis=1),valid=valid,
                near_wake_unresolved=near,elevation_m=float(elevation),method=METHOD)
