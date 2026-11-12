"""Standalone flat-bed SWE Riemann oracle; never imported by runtime.

Equations: LeVeque's author-hosted Riemann Problems and Jupyter Solutions,
https://faculty.washington.edu/rjl/riemann_book/Shallow_water.html

This is an independent shock/rarefaction admissibility reference, not a new
BRAIDS boundary method. A prescribed reservoir state does NOT necessarily keep
its original discharge when a backwater shock travels into that reservoir.
The exact-Q boundary must therefore be checked separately for mass closure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class State:
    depth: float
    velocity: float

    @property
    def discharge(self):
        return self.depth * self.velocity

    def flux(self, gravity=9.81):
        return self.discharge, self.depth*self.velocity**2 + .5*gravity*self.depth**2


def wave_curve(h: float, href: float, gravity=9.81):
    """Positive compression (Hugoniot); negative expansion (rarefaction)."""
    if h < 0 or href <= 0:
        raise ValueError('Nonnegative trial depth and positive reference depth required')
    if h > href:
        return (h-href)*math.sqrt(.5*gravity*(1/h+1/href))
    return 2*(math.sqrt(gravity*h)-math.sqrt(gravity*href))


def _fan(left: State, xi: float, gravity: float):
    c=math.sqrt(gravity*left.depth)
    u=(left.velocity+2*c+2*xi)/3
    return State((u-xi)**2/gravity,u)


def _right_fan(right: State, xi: float, gravity: float):
    c=math.sqrt(gravity*right.depth)
    u=(right.velocity-2*c+2*xi)/3
    return State((u-xi)**2/gravity,u)


def solve(left: State, right: State, *, xi=0., gravity=9.81):
    """Exact homogeneous 1-D solution sampled at x/t=xi, inward-positive x."""
    if gravity <= 0 or min(left.depth,right.depth) < 0:
        raise ValueError('Positive gravity and nonnegative depths required')
    if not all(math.isfinite(x) for x in (left.depth,left.velocity,right.depth,right.velocity,xi,gravity)):
        raise ValueError('Finite states required')
    hl,ul=left.depth,left.velocity;hr,ur=right.depth,right.velocity
    cl=math.sqrt(gravity*hl);cr=math.sqrt(gravity*hr)
    if hl==0 and hr==0:
        return dict(state=State(0.,0.),branch='dry',middle=None,speeds=None)
    if hr==0:
        if xi <= ul-cl:state,branch=left,'left'
        elif xi < ul+2*cl:state,branch=_fan(left,xi,gravity),'left_rarefaction'
        else:state,branch=State(0.,0.),'dry'
        return dict(state=state,branch=branch,middle=None,speeds=[ul-cl,ul+2*cl])
    if hl==0:
        if xi >= ur+cr:state,branch=right,'right'
        elif xi > ur-2*cr:state,branch=_right_fan(right,xi,gravity),'right_rarefaction'
        else:state,branch=State(0.,0.),'dry'
        return dict(state=state,branch=branch,middle=None,speeds=[ur-2*cr,ur+cr])
    if ur-ul >= 2*(cl+cr):
        if xi <= ul-cl:state,branch=left,'left'
        elif xi < ul+2*cl:state,branch=_fan(left,xi,gravity),'left_rarefaction'
        elif xi <= ur-2*cr:state,branch=State(0.,0.),'dry'
        elif xi < ur+cr:state,branch=_right_fan(right,xi,gravity),'right_rarefaction'
        else:state,branch=right,'right'
        return dict(state=state,branch=branch,middle=None,speeds=None)

    def residual(h):return wave_curve(h,hl,gravity)+wave_curve(h,hr,gravity)+ur-ul
    low=0.;high=max(hl,hr)
    while residual(high)<0:high*=2
    for _ in range(120):
        mid=.5*(low+high)
        if residual(mid)>0:high=mid
        else:low=mid
    hs=.5*(low+high)
    us=.5*(ul+ur+wave_curve(hs,hr,gravity)-wave_curve(hs,hl,gravity))
    middle=State(hs,us);cs=math.sqrt(gravity*hs)
    if hs>hl:
        sl=ul-math.sqrt(gravity*hs*(hs+hl)/(2*hl))
        speeds_l=[sl,sl]
        if xi<=sl:return dict(state=left,branch='left',middle=middle,speeds=[*speeds_l,None,None])
    else:
        speeds_l=[ul-cl,us-cs]
        if xi<=speeds_l[0]:return dict(state=left,branch='left',middle=middle,speeds=[*speeds_l,None,None])
        if xi<speeds_l[1]:return dict(state=_fan(left,xi,gravity),branch='left_rarefaction',middle=middle,speeds=[*speeds_l,None,None])
    if hs>hr:
        sr=ur+math.sqrt(gravity*hs*(hs+hr)/(2*hr))
        speeds_r=[sr,sr]
        state,branch=(middle,'middle') if xi<sr else (right,'right')
    else:
        speeds_r=[us+cs,ur+cr]
        if xi>=speeds_r[1]:state,branch=right,'right'
        elif xi>speeds_r[0]:state,branch=_right_fan(right,xi,gravity),'right_rarefaction'
        else:state,branch=middle,'middle'
    return dict(state=state,branch=branch,middle=middle,speeds=[*speeds_l,*speeds_r])


def conjugate_depth(depth, discharge, gravity=9.81):
    """Rectangular stationary hydraulic jump: equal mass/momentum fluxes."""
    if depth<=0 or discharge<=0:raise ValueError('Positive depth/discharge required')
    fr2=discharge**2/(gravity*depth**3)
    return .5*depth*(math.sqrt(1+8*fr2)-1)


def exact_discharge_compatible_state(discharge, interior: State, gravity=9.81):
    """Right-wave curve + Q root; no sonic/regime admission silently added.

    A caller must check characteristic/wave-speed admissibility. Unlike the
    production characteristic approximation, compression uses the Hugoniot
    curve. The returned state is not claimed to solve every boundary regime.
    """
    if discharge<=0 or interior.depth<=0:raise ValueError('Positive discharge and wet interior required')
    def residual(h):return h*(interior.velocity+wave_curve(h,interior.depth,gravity))-discharge
    low=0.;high=max(interior.depth,(discharge**2/gravity)**(1/3))
    while residual(high)<0:high*=2
    for _ in range(120):
        mid=.5*(low+high)
        if residual(mid)>0:high=mid
        else:low=mid
    depth=.5*(low+high)
    return State(depth,discharge/depth)


def section_fixed_discharge(reference_depths, interiors, widths, discharge, gravity=9.81):
    """Independent common-reservoir-velocity/exact-Riemann mass-flux root.

    Reference depths come from one external elevation minus the individual
    beds. The returned physical trace can have another elevation. Signed face
    fluxes are retained deliberately: one-way face closure is a further
    boundary assumption and must be tested separately, not hidden here.
    """
    if len(reference_depths)!=len(interiors) or len(interiors)!=len(widths):
        raise ValueError('Equal-length section arrays required')
    if discharge<=0 or any(h<0 for h in reference_depths) or any(w<=0 for w in widths):
        raise ValueError('Positive discharge/widths and nonnegative reference depths required')
    area=sum(h*w for h,w in zip(reference_depths,widths))
    if area<=0:raise ValueError('A wet external reference section is required')

    def evaluate(velocity):
        solutions=[solve(State(h,velocity),inside,gravity=gravity)
            for h,inside in zip(reference_depths,interiors)]
        fluxes=[row['state'].flux(gravity) for row in solutions]
        return sum(w*f[0] for w,f in zip(widths,fluxes)),solutions,fluxes

    scale=max(1.,discharge/area,*[math.sqrt(gravity*h) for h in reference_depths],
        *[abs(row.velocity) for row in interiors])
    lo,hi=-scale,scale
    for _ in range(80):
        if evaluate(lo)[0]<=discharge<=evaluate(hi)[0]:break
        lo*=2;hi*=2
    else:raise ArithmeticError('Could not bracket section Riemann mass flux')
    for _ in range(100):
        mid=.5*(lo+hi)
        if evaluate(mid)[0]>discharge:hi=mid
        else:lo=mid
    velocity=.5*(lo+hi);actual,solutions,fluxes=evaluate(velocity)
    return dict(virtual_reservoir_velocity=velocity,discharge=actual,
        traces=[row['state'] for row in solutions],branches=[row['branch'] for row in solutions],
        fluxes=fluxes,total_momentum_flux=sum(w*f[1] for w,f in zip(widths,fluxes)))
