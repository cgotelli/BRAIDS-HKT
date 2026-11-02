"""Explicit ambient-flow power screening. Never mutates conserved SWE fields.

Finite-section Gaussian momentum normalization, linear wake overlap, rotor
moments and device-owned Cp/Ct surfaces. First release: straight rectangular
sections, parallel rotors, one upstream row, hydrodynamics only.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.special import erf
from .river_turbines import validate as validate_layout, frame
from .river_turbine_surface import coefficients
from .sediment.river_turbine_inflow import _quadrature

METHOD='power.finite_section_gaussian_linear.v1'


def section(project, layout):
    from shapely.geometry import Polygon
    n,t=frame(layout['turbines'][0]);points=np.asarray(project['settings']['domain']['vertices_xy'],float)
    x,y=points@n,points@t;bounds=(float(x.min()),float(x.max()),float(y.min()),float(y.max()))
    if project['settings']['solids'] or abs(Polygon(points).area-np.ptp(x)*np.ptp(y))>1e-7*max(1.,Polygon(points).area):
        raise ValueError('Power planning currently requires a straight rectangular channel aligned with the turbines, without solids. Curved/obstructed wakes need separate qualification.')
    return n,t,bounds


def validate(value, project):
    if set(value)!={'method','layout'} or value['method']!=METHOD:raise ValueError('Choose finite-depth Gaussian power planning with an explicit turbine layout.')
    if project['turbines'] is not None:raise ValueError('Power planning requires turbine-free reference hydraulics. Select power planning instead of coupled turbine drag.')
    if project['model']['physics']['sediment'] is not None:raise ValueError('Turn off sediment for power planning; its analytical wakes are not sediment forcing.')
    if project['settings']['adaptive']:raise ValueError('Choose a static mesh for the first power-planning release. Local rotor/wake refinement is retained.')
    layout=value['layout'];rows=layout.get('turbines',[])
    if not 1<=len(rows)<=3:raise ValueError('This release supports one turbine, two in tandem or three staggered; use at most three rotors.')
    # Existing single-device validation is reused; its coupled overlap policy
    # does not apply to this explicitly different analytical estimator.
    for r in rows:
        validate_layout(dict(layout,turbines=[r]),project)
        if r['control']['mode']!='fixed_rpm_surface':raise ValueError('Power planning requires Device speed / TSR surfaces for every turbine; supply actual Cp/Ct data.')
        if r['control']['wake']['method']!='gaussian_shape_power_planning':raise ValueError('Save this layout through the Power planning editor to identify its analytical wake model explicitly.')
    n,t,bounds=section(project,layout);parents={i:[] for i in range(len(rows))}
    for j,r in enumerate(rows):
        if not np.allclose(frame(r)[0],n,atol=1e-10,rtol=0):raise ValueError('Power planning requires parallel, zero-yaw rotors in this release.')
        for i,up in enumerate(rows):
            dx=float((np.array(r['center_xy'])-up['center_xy'])@n)
            if dx<=1e-9:continue
            # Gaussian tails are not truncated at the mesh-refinement length.
            xD=dx/up['rotor_diameter_m']
            if xD<up['control']['wake']['minimum_distance_D']:raise ValueError('Increase downstream spacing to the declared far-wake minimum; no near-wake extrapolation is allowed.')
            parents[j].append(i)
    if any(parents[i] for pp in parents.values() for i in pp):raise ValueError('Only one upstream row is qualified. Cascading three rows needs separate braids_support.')
    return deepcopy(value)


def finite_amplitude(ct,D,sigma,H,hub,W,center_y):
    if not all(np.isfinite(v) for v in [ct,D,sigma,H,hub,W,center_y]) or not(0<=ct<1 and D>0 and sigma>0 and H>0 and W>0 and 0<hub<H and 0<center_y<W):
        raise ValueError('Finite-section wake requires finite in-water geometry and 0 <= Ct < 1.')
    def integral(L,c,s):return s*np.sqrt(np.pi/2)*(erf((L-c)/(np.sqrt(2)*s))+erf(c/(np.sqrt(2)*s)))
    I1=integral(W,center_y,sigma)*integral(H,hub,sigma)
    I2=integral(W,center_y,sigma/np.sqrt(2))*integral(H,hub,sigma/np.sqrt(2))
    force=.5*ct*np.pi*D**2/4;disc=I1**2-4*I2*force
    if disc<=0:raise ValueError('No supported finite-section wake root: blockage/loading is outside this power model.')
    a=2*force/(I1+np.sqrt(disc))
    if a>=1:raise ValueError('Wake predicts reversed flow; this loading is outside the power model.')
    return a


def evaluate(project,sample,time_s,reference_bed):
    """sample(xy) -> (h,bed,u,v) arrays; same frame, no induced velocity input.

    Bed reference is fixed at first saved frame, matching a mounted rotor.
    Ambient velocity is uniform vertically; lateral variations are sampled.
    Changes propagate quasi-steadily, without a wake-advection time delay.
    """
    plan=project['power_plan'];layout=plan['layout'];rows=layout['turbines'];n,t,bb=section(project,layout)
    order=sorted(range(len(rows)),key=lambda i:np.dot(rows[i]['center_xy'],n));resolved={};out=[]
    for j in order:
        r=rows[j];D=r['rotor_diameter_m'];R=D/2;c=np.array(r['center_xy']);ctl=r['control']
        ry,rz,w=_quadrature(R);xy=c+ry[:,None]*t;h,z,u,v=map(np.asarray,sample(xy));velocity=u*n[0]+v*n[1]
        if not np.isfinite([h,z,u,v]).all():raise ValueError('Rotor samples lie outside the saved mesh or contain invalid fields.')
        hub=reference_bed[r['id']]+r['hub_height_m'];height=hub+rz
        bed_margin=float(np.min(height-z));cover=float(np.min(z+h-height))
        # Test the entire rotor extremes, conservatively over the lateral footprint.
        bed_margin=min(bed_margin,float(hub-R-np.max(z)));cover=min(cover,float(np.min(z+h)-hub-R))
        reserve_b=bed_margin-r['minimum_bed_clearance_m'];reserve_s=cover-r['minimum_surface_clearance_m']
        ambient=float(w@velocity);vel=velocity.copy();parents=[]
        for i in order:
            if i not in resolved:continue
            p=resolved[i];dx=float((c-p['center'])@n)
            if dx<=1e-9 or not p['active']:continue
            up=rows[i];wake=up['control']['wake'];sig=wake['growth']*dx+up['rotor_diameter_m']*wake['initial_width']
            a=finite_amplitude(p['Ct'],up['rotor_diameter_m'],sig,p['H'],p['hub']-p['bed'],bb[3]-bb[2],float(p['center']@t-bb[2]))
            dy=(xy-p['center'])@t
            vel-=p['mean']*a*np.exp(-.5*((dy/sig)**2+((height-p['hub'])/sig)**2));parents.append(up['id'])
        mean=float(w@vel);reason='operating'
        if reserve_b < -1e-10 or reserve_s < -1e-10:reason='insufficient_clearance'
        elif ctl['rotational_speed_rpm']==0:reason='parked'
        elif mean<=1e-10:reason='no_forward_flow'
        elif np.any(vel<=0):raise ValueError('Overlapping wakes predict local reversal; outside qualified power-planning range.')
        op=ctl.get('operation')
        if reason=='operating' and op:
            if mean<op['cut_in_m_s']:reason='below_incoming_speed_cut_in'
            elif mean>=op['cut_out_m_s']:reason='at_or_above_incoming_speed_cut_out'
        active=reason=='operating';Ct=Cp=TSR=P=F=0.
        if active:
            if float(w@np.abs(-u*n[1]+v*n[0]))>0.1*max(abs(ambient),1e-10):raise ValueError('Ambient flow is substantially yawed (> 10% transverse velocity). Reorient or use a qualified turning-wake model.')
            TSR=ctl['rotational_speed_rpm']*np.pi/30*R/mean
            try:Ct,Cp=coefficients(ctl,mean,TSR)
            except ValueError as exc:raise ValueError(f'{r["id"]} at {time_s:g} s: incoming speed {mean:.6g} m/s, TSR {TSR:.6g}. {exc} Supply device data covering these operating conditions; no extrapolation was applied.') from exc
            P=.5*1000*np.pi*R**2*Cp*float(w@vel**3);F=.5*1000*np.pi*R**2*Ct*float(w@vel**2)
        record=dict(time_s=float(time_s),turbine_id=r['id'],method=METHOD,model_name=ctl['model_name'],active=active,status=reason,
            thrust_coefficient=float(Ct),power_coefficient=float(Cp),useful_power_W=float(P),electrical_power_W=None if op is None else float(P*op['electrical_efficiency']),thrust_N=float(F),
            upstream_velocity_m_s=mean,local_velocity_m_s=ambient,ambient_velocity_m_s=ambient,tip_speed_ratio=float(TSR),rotational_speed_rpm=ctl['rotational_speed_rpm'],
            hydraulic_work_W=None,minimum_coverage_margin_m=cover,minimum_bed_clearance_m=bed_margin,surface_clearance_reserve_m=reserve_s,bed_clearance_reserve_m=reserve_b,
            upstream_wakes=';'.join(parents),assumptions='uniform vertical ambient velocity; quasi-steady straight wakes; shaft Cp; no hydraulic feedback',
            applicability='Research estimate: higher-turbulence evidence; low-TI and strong blockage not qualified')
        out.append(record);resolved[j]=dict(center=c,active=active,Ct=Ct,mean=mean,H=float(w@h),bed=float(w@z),hub=hub)
    return out
