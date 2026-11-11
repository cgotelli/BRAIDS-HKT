"""Dou et al. 2019, doi:10.1016/j.apenergy.2019.03.164, Eqs1,3,12,14,15.

Hub-plane empirical model, NOT a 3-D momentum closure. The printed Eq12 is
retained (sqrt(2*mu/pi), not mu*sqrt(2/pi)); it is an approximate location
correction, not an exact skew-normal mode. Negative yaw mirrors positive yaw.
BRAIDS power uses an explicitly separate Gaussian vertical completion.
"""
import math
import numpy as np

METHOD='power.dou2019_device_yaw.v1'
ASSUMPTIONS=('Dou2019 empirical hub-plane yaw deficit; reflected signed yaw; '
 'printed Eq12 approximate mode; provenance-bearing aligned width/growth; '
 'Gaussian vertical completion retains aligned width, unvalidated for yaw; '
 'linear sum on first-row reference; each source uses own local hydraulic inflow; '
 'quasi-steady straight reach; conditional multi-row power prediction')

def beta(ct):
    if not math.isfinite(ct) or not 0<=ct<1:raise ValueError('Dou wake needs 0 <= effective zero-yaw Ct < 1; check thrust convention.')
    root=math.sqrt(1-ct)
    return (1+root)/(2*root)

def parameters(D,x,ct0,gamma,k,epsilon=None,rotor_to_tower=0.,xp=np):
    """gamma: signed wake deflection toward +y; radians, |gamma|<=32deg.

    epsilon=None reproduces the paper's sqrt(beta)/5 width. Explicit epsilon
    preserves a separately qualified aligned width, using the paper's ratio
    of yawed to unyawed initial widths. x,y originate at rotor centre in BRAIDS;
    rotor_to_tower is used ONLY to reproduce a tower-origin published plot.
    """
    if not all(math.isfinite(float(v)) for v in (D,gamma,k,rotor_to_tower)) or D<=0 or k<0:
        raise ValueError('Invalid Dou wake geometry/growth.')
    if abs(gamma)>math.radians(32)+1e-12:raise ValueError('Dou yaw profile is restricted to |yaw| <= 32 degrees in this research implementation.')
    if bool(xp.any(~xp.isfinite(x)|(x<=0))):raise ValueError('Dou wake requires positive finite downstream distances.')
    cg=math.cos(gamma);sg=math.sin(gamma);b0=beta(ct0);stream=ct0*cg**3;by=beta(stream)
    e0=math.sqrt(b0)/5 if epsilon is None else float(epsilon)
    if not math.isfinite(e0) or e0<=0:raise ValueError('Positive finite initial wake width required.')
    sigma_y=k*x+e0*math.sqrt(by/b0)*D*cg
    sigma_z=k*x+e0*D
    offset=.603*ct0*math.copysign(abs(ct0*sg)**.75,sg)*cg**1.5*xp.sqrt(x/D)*D+rotor_to_tower*sg
    alpha=5*ct0**4.63*math.tan(gamma)
    a=abs(alpha)
    if a==0:m=0.
    else:
        mu=a/math.sqrt(1+a*a)
        skew=.5*(4-math.pi)*(mu*math.sqrt(2/math.pi))**3/(1-2*mu*mu/math.pi)**1.5
        m=math.copysign(math.sqrt(2*mu/math.pi)-.5*skew*math.sqrt(1-2*mu*mu/math.pi)-.5*math.exp(-2*math.pi/a),alpha)
    rad=1-stream*(D*cg)**2/(8*sigma_y**2)
    if bool(xp.any(rad<=0)):raise ValueError('Dou wake width is too small for its real recovered-pressure amplitude; increase spacing or provide supported width data.')
    return sigma_y,sigma_z,offset,alpha,m,1-xp.sqrt(rad)

def lateral(D,x,y,ct0,gamma,k,epsilon=None,rotor_to_tower=0.,xp=np):
    sy,sz,offset,alpha,m,amplitude=parameters(D,x,ct0,gamma,k,epsilon,rotor_to_tower,xp)
    chi=(y-offset+sy*m)/sy
    erf=__import__('scipy.special',fromlist=['erf']).erf if xp is np else __import__('cupyx.scipy.special',fromlist=['erf']).erf
    shape=xp.exp(-.5*chi**2)*(1+erf(alpha*chi/math.sqrt(2)))
    return amplitude*shape,sz

def source_parameters(rotor,state,stream_ct):
    """Incoming signed angle, not the angle of the already induced velocity.

    Force projection has ALREADY occurred in stream_ct: divide by cos^3 to
    construct the equivalent reference Ct; parameters() applies it ONCE.
    """
    if rotor.original.get('orientation',{}).get('mode')!='fixed_axis':return stream_ct,0.
    U=state['upstream_velocity_m_s'];ut=state['local_tangent_velocity_m_s']
    if U<=0 or abs(ut)>U*(1+1e-12):raise ValueError('Invalid incoming vector for Dou wake.')
    gamma=math.asin(max(-1.,min(1.,ut/U)))
    ct0=stream_ct/math.cos(gamma)**3
    beta(ct0)
    return ct0,gamma
