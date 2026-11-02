"""Device-owned speed/TSR performance surface and rotor-moment induction.

No turbine-specific data, wake constants or spacing corrections live here.
Uniform and nonuniform inflow use the same mechanical power convention.
"""
import math
import numpy as np

MODE='fixed_rpm_surface'


def exposure_parents(centers,normals,diameters,controls,*,names=None):
    """Conservative potential exposure, independent of instantaneous on/off state.

    Gaussian tails are not compact. Use the declared four-sigma lateral screen
    plus the receiving radius; never truncate at the mesh refinement distance.
    """
    parents={j:[] for j in range(len(centers))}
    names=names or [f'Turbine {i+1}' for i in range(len(centers))]
    tolerance=128*np.finfo(float).eps*max(1.,float(np.max(np.abs(centers)))) if len(centers) else 0.
    for j,down in enumerate(centers):
        for i,up in enumerate(centers):
            if i==j:continue
            n=np.asarray(normals[i]);t=np.array([-n[1],n[0]]);delta=np.asarray(down)-up
            xD=float(delta@n/diameters[i]);y=float(delta@t);w=controls[i]['wake']
            if xD*diameters[i]<=tolerance:continue
            sigma=diameters[i]*(w['growth']*xD+w['initial_width'])
            if sigma<=0:raise ValueError('Wake width is nonpositive at another rotor.')
            if abs(y)>4*sigma+diameters[j]/2:continue
            if xD<w['minimum_distance_D']:
                required=w['minimum_distance_D'];D=diameters[i]
                raise ValueError(f'Turbine "{names[j]}" lies in the near wake of "{names[i]}". Downstream spacing is {xD*D:.6g} m ({xD:.4g}D); this wake model requires at least {required*D:.6g} m ({required:g}D, upstream rotor diameter {D:g} m). The receiving rotor intersects the potential lateral wake envelope. Near-wake exclusion means the selected far-wake velocity/power correction is not qualified this close; it is not a physical rotor collision. Move the downstream turbine farther along the upstream river direction, or outside its wake envelope. Do not lower the model limit merely to bypass this check. The 10D mesh-refinement length is a separate setting.')
            if not np.allclose(n,normals[j],atol=1e-10,rtol=0):raise ValueError('Wake turning/yawed exposure is not qualified; provide a validated wake trajectory model.')
            parents[j].append((i,xD,y,sigma))
        if len(parents[j])>1 and controls[j]['wake']['method']=='gaussian_zero_mean_single_wake':raise ValueError('Overlapping upstream wakes require a qualified superposition rule; move turbines to avoid multiple upstream exposures.')
    return parents

def validate(control):
    from .river_turbine_performance import validate as curve_validate
    required={'mode','model_name','rotational_speed_rpm','performance','outside_surface','wake'}
    if not isinstance(control,dict) or set(control) not in (required,required|{'operation'}) or control['mode']!=MODE:
        raise ValueError('Speed surface requires mode, model_name, RPM, performance, outside_surface and explicit wake model.')
    if control['outside_surface']!='error':raise ValueError('Missing performance data must raise an error; no silent extrapolation or zero energy.')
    if 'operation' in control:
        op=control['operation']
        if not isinstance(op,dict) or set(op)!={'method','cut_in_m_s','cut_out_m_s','electrical_efficiency','provenance'} or op['method']!='local_normal_speed_window':
            raise ValueError('Operating limits require the explicit local-normal-speed window, cut-in/out, efficiency and provenance.')
        if any(type(op[k]) not in (int,float) or not math.isfinite(op[k]) for k in ['cut_in_m_s','cut_out_m_s','electrical_efficiency']):raise ValueError('Operating speeds and efficiency must be finite numbers.')
        if not 0<=op['cut_in_m_s']<op['cut_out_m_s'] or not 0<=op['electrical_efficiency']<=1:raise ValueError('Require 0 <= cut-in < cut-out and electrical efficiency between 0 and 1.')
        if not isinstance(op['provenance'],str) or not op['provenance'].strip():raise ValueError('Describe the controller and electrical-efficiency assumptions.')
    tables=control['performance']
    if not isinstance(tables,list) or len(tables)<2:raise ValueError('Provide measured/manufacturer curves at two or more incoming speeds.')
    clean=[]
    for table in tables:
        if not isinstance(table,dict) or set(table)!={'inflow_m_s','curve'}:raise ValueError('Each speed slice needs inflow_m_s and a TSR/Ct/Cp curve.')
        speed=table['inflow_m_s']
        if type(speed) not in (int,float) or not math.isfinite(speed) or speed<=0:raise ValueError('Surface speeds must be finite positive m/s.')
        c=curve_validate(dict(mode='fixed_rpm_table',model_name=control['model_name'],rotational_speed_rpm=control['rotational_speed_rpm'],performance=table['curve'],outside_curve='off'))
        clean.append(dict(inflow_m_s=float(speed),curve=c['performance']))
    clean.sort(key=lambda t:t['inflow_m_s'])
    if any(a['inflow_m_s']==b['inflow_m_s'] for a,b in zip(clean,clean[1:])):raise ValueError('Duplicate surface speeds.')
    wake=control['wake']
    if not isinstance(wake,dict) or set(wake)!={'method','growth','initial_width','minimum_distance_D','provenance'} or wake['method'] not in ('gaussian_zero_mean_single_wake','gaussian_zero_mean_linear_overlap','gaussian_shape_power_planning'):
        raise ValueError('Give an explicit Gaussian wake shape, minimum distance and calibration provenance; no automatic Mycek default.')
    if any(type(wake[k]) not in (int,float) or not math.isfinite(wake[k]) for k in ['growth','initial_width','minimum_distance_D']):raise ValueError('Finite wake parameters required.')
    if wake['growth']<0 or wake['minimum_distance_D']<3 or wake['growth']*wake['minimum_distance_D']+wake['initial_width']<=0:raise ValueError('Invalid far-wake width/range; qualification starts at 3D or farther.')
    if not isinstance(wake['provenance'],str) or not wake['provenance'].strip():raise ValueError('Identify the wake calibration and applicability.')
    return dict(control,model_name=c['model_name'],rotational_speed_rpm=c['rotational_speed_rpm'],performance=clean,wake=dict(wake))

def _line(curve,tsr,field):
    x=np.array([r['tip_speed_ratio'] for r in curve]);y=np.array([r[field] for r in curve])
    if tsr<x[0]-1e-11 or tsr>x[-1]+1e-11:raise ValueError('TSR outside supplied surface; extend the device data.')
    i=min(len(x)-2,max(0,int(np.searchsorted(x,tsr))-1));m=(y[i+1]-y[i])/(x[i+1]-x[i])
    return m,y[i]-m*x[i]

def _abc(control,v,K,field):
    tables=control['performance'];speeds=np.array([t['inflow_m_s'] for t in tables])
    if not speeds[0]-1e-11<=v<=speeds[-1]+1e-11:raise ValueError('Incoming rotor speed outside supplied performance surface; extend the device data.')
    i=min(len(speeds)-2,max(0,int(np.searchsorted(speeds,v))-1));lo,hi=tables[i:i+2]
    ml,bl=_line(lo['curve'],K/v,field);mh,bh=_line(hi['curve'],K/v,field)
    ds=speeds[i+1]-speeds[i];a=(bh-bl)/ds;b=bl+(mh-ml)*K/ds-speeds[i]*a;c=K*(ml-speeds[i]*(mh-ml)/ds)
    return a,b,c

def coefficients(control,speed,tsr):
    return tuple(a*speed+b+c/speed for a,b,c in [_abc(control,speed,tsr*speed,f) for f in ['thrust_coefficient','power_coefficient']])

def resolve_moments(control,diameter,depth,local,mu,var,central3,min_offset):
    """Enumerate exact piecewise quartic roots; reject ambiguous/no solution.

    Mean rotor speed v=U0+mu, Ct(v,K/v)=a*v+b+c/v per table segment.
    U0=local + A*Ct*(v*v+var)/(4*D*H*local). No lagged Ct or trial clamping.
    """
    if control['rotational_speed_rpm']==0:return dict(active=False,reason='parked')
    if local<=1e-10:return dict(active=False,reason='reverse_flow' if local<0 else 'no_incoming_flow')
    if 'operation' in control:
        op=control['operation']
        if local<op['cut_in_m_s']:return dict(active=False,reason='below_local_speed_cut_in')
        if local>=op['cut_out_m_s']:return dict(active=False,reason='at_or_above_local_speed_cut_out')
    if not all(math.isfinite(x) for x in [diameter,depth,local,mu,var,central3,min_offset]) or diameter<=0 or depth<=0 or var<0:raise ValueError('Invalid rotor moment/geometry.')
    K=control['rotational_speed_rpm']*math.pi/30*diameter/2;A=math.pi*diameter**2/4
    lo=max(local+mu,mu-min_offset,control['performance'][0]['inflow_m_s']);hi=min(2*local+mu,control['performance'][-1]['inflow_m_s'])
    cuts=[lo,hi]
    for t in control['performance']:
        cuts.append(t['inflow_m_s']);cuts.extend(K/r['tip_speed_ratio'] for r in t['curve'] if r['tip_speed_ratio']>0)
    cuts=sorted(set(x for x in cuts if lo<=x<=hi));roots=[];B=A/(4*diameter*depth*local)
    # Include isolated envelope endpoints (e.g. Ct=0 at the highest measured
    # speed), where the admissible interval can collapse to one point.
    for v in cuts:
        if v<=0:continue
        try:Ct,Cp=coefficients(control,v,K/v)
        except ValueError:continue
        u0=v-mu;m2=v*v+var;m3=v**3+3*v*var+central3;F=.5*A*Ct*m2;P=.5*A*Cp*m3
        residual=u0-local-F/(2*diameter*depth*local)
        if abs(residual)<1e-12 and u0+min_offset>=-1e-10 and P>=0:
            roots.append(dict(active=True,U0=u0,mean=v,second=m2,third=m3,Ct=Ct,Cp=Cp,TSR=K/v,F=F,P=P,residual=residual))
    for left,right in zip(cuts,cuts[1:]):
        try:a,b,c=_abc(control,(left+right)/2,K,'thrust_coefficient')
        except ValueError:continue
        # (v-local-mu)*v - B*(a*v^2+b*v+c)*(v^2+var) = 0.
        p=[-B*c*var,-local-mu-B*b*var,1-B*(c+a*var),-B*b,-B*a]
        for root in np.polynomial.polynomial.polyroots(np.trim_zeros(p,'b')):
            if abs(root.imag)>1e-8:continue
            v=float(root.real)
            if not left-1e-10<=v<=right+1e-10:continue
            Ct,Cp=coefficients(control,v,K/v);u0=v-mu;m2=v*v+var;m3=v**3+3*v*var+central3
            F=.5*A*Ct*m2;P=.5*A*Cp*m3;residual=u0-local-F/(2*diameter*depth*local)
            if abs(residual)>1e-9 or u0+min_offset < -1e-10 or P<0:continue
            if any(abs(v-r['mean'])<1e-9 for r in roots):continue
            roots.append(dict(active=True,U0=u0,mean=v,second=m2,third=m3,Ct=Ct,Cp=Cp,TSR=K/v,F=F,P=P,residual=residual))
    if len(roots)!=1:raise ValueError(('Multiple operating points' if roots else 'No supported operating point')+f' for {control["model_name"]}: local speed {local:.6g} m/s. Check RPM and supply speed/TSR data covering the induced rotor inflow; no extrapolation performed.')
    return roots[0]

def resolve(control,diameter,depth,local,offsets,weights,xp=np):
    if bool(xp.any(~xp.isfinite(offsets))) or bool(xp.any(weights<0)) or abs(float(xp.sum(weights))-1)>1e-10:raise ValueError('Finite offsets and normalized nonnegative disk weights required.')
    mu=float(xp.sum(weights*offsets));centered=offsets-mu
    return resolve_moments(control,diameter,depth,local,mu,float(xp.sum(weights*centered**2)),float(xp.sum(weights*centered**3)),float(xp.min(offsets)))

def operating_point(control,diameter,depth,local):
    p=resolve_moments(control,diameter,depth,local,0.,0.,0.,0.)
    if not p['active']:return dict(p,rotational_speed_rpm=control['rotational_speed_rpm'])
    return dict(active=True,reason=None,ct=p['Ct'],cp=p['Cp'],tip_speed_ratio=p['TSR'],upstream_velocity_m_s=p['U0'],rotational_speed_rpm=control['rotational_speed_rpm'])

def integrate_energy(records):
    """Output-sampled shaft energy; reject duplicate restart timestamps.

    The caller must verify history completeness and output sampling adequacy.
    Irregular but strictly increasing sample times are supported.
    """
    grouped={}
    for row in records:
        t=float(row['time_s']);p=float(row['useful_power_W'])
        if not math.isfinite(t) or not math.isfinite(p) or p<0:raise ValueError('Finite time and nonnegative shaft power required.')
        e=row.get('electrical_power_W')
        if e is not None and (not math.isfinite(float(e)) or not 0<=e<=p*(1+1e-10)):raise ValueError('Electrical power must be finite and between zero and shaft power.')
        grouped.setdefault(row['turbine_id'],[]).append((t,p,e))
    result=[]
    for name,points in grouped.items():
        points.sort();energy=0.;electrical=0.;last=None
        known=[x[2] is not None for x in points]
        if any(known) and not all(known):raise ValueError('Electrical history is incomplete; do not infer missing efficiency.')
        for t,p,e in points:
            if last is not None:
                if t<=last[0]:raise ValueError('Duplicate turbine output time; stitch restart histories explicitly before energy integration.')
                energy+=.5*(p+last[1])*(t-last[0])
                if e is not None:electrical+=.5*(e+last[2])*(t-last[0])
            result.append(dict(turbine_id=name,time_s=t,shaft_power_W=p,shaft_energy_J=energy,shaft_energy_Wh=energy/3600,electrical_power_W=e,electrical_energy_Wh=None if e is None else electrical/3600))
            last=t,p,e
    return result
