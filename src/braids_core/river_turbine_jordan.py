"""Jordan (2022) paired SWE / analytical array evaluation, non-yawed scope.

Gaussian and turbulence expressions adapted from NREL FLORIS 2.2.0,
commit 5f7e6b0473e145207b5e6a86b3f0ed505ec725e1 (Apache-2.0).
Copyright 2020 NREL; modifications Copyright 2026 BRAIDS contributors.
See licenses/FLORIS-2.2.0.txt. This module never changes a SWE state or source.
It evaluates AUTHENTICATED turbine-free ambient samples, not coupled velocities.
The final hydraulic layout assessment remains a separate turbine-on SWE run.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

METHOD = 'analysis.jordan2022_floris220.v1'


@dataclass(frozen=True)
class WakeParameters:
    # No universal site defaults. Use paper_channel() only for that reconstruction.
    ka: float
    kb: float
    alpha: float
    beta: float
    ambient_ti: float
    added_ti_constant: float
    added_ti_induction_exponent: float
    added_ti_ambient_exponent: float
    added_ti_distance_exponent: float
    provenance: str

    def __post_init__(self):
        nums=[v for v in asdict(self).values() if not isinstance(v,str)]
        if not np.isfinite(nums).all() or not self.provenance.strip():
            raise ValueError('Supply finite wake parameters and calibration provenance.')
        if min(self.ka,self.kb,self.alpha,self.beta,self.added_ti_constant)<0 or not 0<self.ambient_ti<1:
            raise ValueError('Wake parameters must be nonnegative; ambient TI is a fraction between 0 and 1.')
        if self.ka*self.ambient_ti+self.kb<=0 or self.alpha*self.ambient_ti+self.beta<=0:
            raise ValueError('Positive wake growth and near/far transition are required.')

    @classmethod
    def paper_channel(cls):
        return cls(.1087,.006912,.4886,.2496,.20,.5,.8325,.12,-.32,
            'Jordan2022 Tables2/3; FLORIS2.2.0 gauss_legacy/FLS. Initial=.12 is the source exponent, not a second ambient TI. Author input deck unavailable; explicit reconstruction.')


def width(x,D,Ct,TI,p):
    """FLORIS legacy Gaussian, zero yaw/veer. x in metres downstream."""
    if not (D>0 and 0<Ct<1 and 0<TI<1):
        raise ValueError('Gaussian width requires D>0, 0<Ct<1 and 0<TI<1.')
    x=np.asarray(x,float);s=math.sqrt(1-Ct)
    x0=D*(1+s)/(math.sqrt(2)*(4*p.alpha*TI+2*p.beta*(1-s)))
    # At zero yaw sigma_y0=sigma_z0=D/sqrt(8), by exact cancellation.
    sigma0=D/math.sqrt(8);start=.501*D*math.sqrt(Ct/2)
    sigma=np.where(x<x0,start+(sigma0-start)*np.maximum(x,0)/x0,
                   sigma0+(p.ka*TI+p.kb)*(x-x0))
    return sigma,x0


def deficit(x,y,z,D,Ct,TI,ambient,p):
    """Free-stream linear-superposition contribution at each receptor.

    The velocity multiplier is the undisturbed velocity at that receptor,
    following FLORIS u_initial. It is not the already reduced parent speed.
    """
    x,y,z,ambient=np.broadcast_arrays(x,y,z,ambient)
    if Ct==0:return np.zeros_like(ambient,dtype=float)
    sigma,x0=width(x,D,Ct,TI,p)
    amp=1-np.sqrt(1-Ct*D**2/(8*sigma**2))
    ans=ambient*amp*np.exp(-.5*(y*y+z*z)/sigma**2)
    return np.where(x>0,ans,0.)


def depth_mean_deficit(x,y,depth,hub,D,Ct,TI,ambient,p):
    """Analytic vertical integral, not a hub-height/depth-mean substitution."""
    from scipy.special import erf
    if np.any(np.asarray(depth)<=0):raise ValueError('Depth must be positive.')
    if Ct==0:return np.zeros_like(np.broadcast_arrays(x,y,depth,ambient)[0],dtype=float)
    sigma,_=width(x,D,Ct,TI,p)
    vertical=sigma*np.sqrt(np.pi/2)/depth*(erf((depth-hub)/(np.sqrt(2)*sigma))+erf(hub/(np.sqrt(2)*sigma)))
    return deficit(x,y,0.,D,Ct,TI,ambient,p)*vertical


def rotor_grid(D):
    # FLORIS 2.2.0 defaults: 5x5 at +/-0.5R, all 25 points inside disk.
    # Kept explicit for reproducibility; this is not full-disk Gaussian quadrature.
    y,z=np.meshgrid(np.linspace(-D/4,D/4,5),np.linspace(-D/4,D/4,5))
    return y.ravel(),z.ravel()


def device(row,speed,rho):
    """Shared incident-speed curves; interpolate tabulated power as FLORIS220."""
    D=row['rotor_diameter_m'];a=np.pi*D*D/4;control=row['control'];mode=control['mode']
    if mode=='constant_ct':
        ct=float(control['thrust_coefficient']);cp=control.get('power_coefficient')
        if not math.isfinite(ct) or not 0<=ct<1:raise ValueError('Analytical constant Ct must satisfy 0<=Ct<1.')
        if not cp:cp=.5*ct*(1+math.sqrt(1-ct))
        if not math.isfinite(cp) or not 0<=cp<=1:raise ValueError('Supply a finite mechanical Cp between 0 and 1.')
        return ct,float(cp),.5*rho*a*cp*speed**3
    if mode=='incident_speed_curves':
        from .sediment.river_turbine_thetis import validate_control,curve_arrays
        validate_control(control)
        u,ct=curve_arrays(control,'thrust_coefficient',np)
        pu,cp=curve_arrays(control,'power_coefficient',np)
        if speed<u[0] or speed>u[-1] or speed<pu[0] or speed>pu[-1]:
            raise ValueError('Analytical rotor speed is outside the supplied Ct or Cp curve; no extrapolation.')
        # Preserve FLORIS220 power-interpolation convention on the Cp knots.
        return float(np.interp(speed,u,ct)),float(np.interp(speed,pu,cp)),float(np.interp(speed,pu,.5*rho*a*cp*pu**3))
    if mode!='incident_speed_table':
        raise ValueError('Jordan analysis requires an incoming-speed Ct/Cp table or constant coefficients. A local-speed Thetis table is a different convention; supply the original incident-speed curve.')
    table=control['performance'];u=np.array([v['speed_m_s'] for v in table]);ct=np.array([v['thrust_coefficient'] for v in table]);cp=np.array([v['power_coefficient'] for v in table])
    if len(u)<2 or not np.isfinite([u,ct,cp]).all() or np.any(np.diff(u)<=0) or np.any(u<0) or np.any((ct<0)|(ct>=1)) or np.any((cp<0)|(cp>1)):
        raise ValueError('Supply increasing finite nonnegative speed / Ct / Cp data with 0<=Ct<1 and 0<=Cp<=1.')
    if speed<u[0] or speed>u[-1]:raise ValueError('Analytical rotor speed is outside the supplied device curve; no extrapolation.')
    return float(np.interp(speed,u,ct)),float(np.interp(speed,u,cp)),float(np.interp(speed,u,.5*rho*a*cp*u**3))


def evaluate(layout,sample,parameters,*,time_s=0.,rho=1000.,reference_bed=None):
    """sample(xy)->depth,bed,u,v from a turbine-free SWE frame.

    Straight, parallel, non-yawed rotors, quasi-steady frames. Returns analytical
    rows and resolved parent states for field evaluation. Caller authenticates
    origin of sample; no conservation or hydraulic field is overwritten here.
    """
    from .river_turbines import frame
    p=parameters;rows=layout['turbines']
    if not rows or rho<=0 or not np.isfinite(rho):raise ValueError('A layout and positive water density are required.')
    n,t=frame(rows[0]);n=np.asarray(n);t=np.asarray(t)
    if any(not np.allclose(frame(r)[0],n,atol=1e-9,rtol=0) for r in rows):raise ValueError('Jordan non-yawed comparison requires parallel rotors.')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Turbine identifiers must be unique.')
    order=sorted(range(len(rows)),key=lambda i:np.dot(rows[i]['center_xy'],n));states=[];output=[]
    for j in order:
        r=rows[j];D=float(r['rotor_diameter_m']);c=np.asarray(r['center_xy'],float);ry,rz=rotor_grid(D)
        # Include full rotor edges for coverage, although legacy power points are narrower.
        xy=c+np.r_[ry,-D/2,D/2][:,None]*t
        h,b,u,v=map(lambda q:np.asarray(q,float),sample(xy))
        if not np.isfinite([h,b,u,v]).all():raise ValueError(f'{r["id"]}: rotor outside ambient mesh or invalid field.')
        if reference_bed is None:base=float(np.mean(b[:-2]))
        else:base=float(reference_bed[r['id']])
        hub=base+r['hub_height_m'];s=u*n[0]+v*n[1];cross=-u*n[1]+v*n[0]
        active=bool(h.min()>0 and hub-D/2-b.max()>=r.get('minimum_bed_clearance_m',0)-1e-9 and (b+h).min()-hub-D/2>=r.get('minimum_surface_clearance_m',0)-1e-9)
        if active and (np.any(s<0) or np.max(abs(cross))>.1*float(np.mean(s))):
            raise ValueError(f'{r["id"]}: ambient flow reverses or turns; reorient the non-yawed layout for this frame.')
        active=bool(active and np.max(s)>1e-10)
        ambient=s[:-2];vel=ambient.copy();TI=p.ambient_ti;parents=[]
        for up in states:
            dx=float((c-up['center'])@n);dy=(xy[:-2]-up['center'])@t
            if dx<=1e-8 or not up['active']:continue
            if dx/up['D']<1.5:raise ValueError(f'{r["id"]}: downstream spacing is below Jordan comparison region (1.5D).')
            du=deficit(dx,dy,hub+rz-up['hub'],up['D'],up['Ct'],up['TI'],ambient,p)
            vel-=du
            if du.max()>1e-8:parents.append(up['id'])
            if dx<=15*up['D'] and abs(float((c-up['center'])@t))<2*up['D']:
                overlap=float(np.mean(du>.05))  # exact dimensional FLORIS220 support rule
                ai=.5*(1-math.sqrt(1-up['Ct']))
                added=overlap*p.added_ti_constant*ai**p.added_ti_induction_exponent*p.ambient_ti**p.added_ti_ambient_exponent*(dx/up['D'])**p.added_ti_distance_exponent
                TI=max(TI,math.sqrt(p.ambient_ti**2+added**2))
        if vel.min()<0:raise ValueError(f'{r["id"]}: linear wake overlap predicts reverse flow; not supported.')
        speed=float(np.cbrt(np.mean(vel**3)));ct,cp,power=device(r,speed,rho) if active else (0.,0.,0.)
        state=dict(id=r['id'],center=c,D=D,Ct=ct,TI=TI,hub=hub,active=active and ct>0)
        states.append(state)
        output.append(dict(time_s=float(time_s),turbine_id=r['id'],method=METHOD,analytical_power_W=power,
            analytical_U3_m_s=speed,ambient_velocity_m_s=float(np.mean(ambient)),Ct=ct,Cp=cp,local_ti=TI,
            active=active,upstream_wakes=';'.join(parents),status='operating' if active else 'insufficient_clearance',
            source='turbine-free SWE + FLORIS220 Gaussian/FLS; separate from coupled hydraulic power'))
    return output,states
