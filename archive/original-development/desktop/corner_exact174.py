"""Independent integrals for a sharp 270-degree slip-wall corner.

psi=(3A/2) r^(2/3) sin(2 theta/3), theta in [0,3pi/2], gives a
divergence-free potential velocity tangent to both walls. Constant depth is
used for an instantaneous spatial residual, NOT a steady shallow-water
solution. The singular velocity is integrable; no sample hits the corner.
"""
import numpy as np
from scipy.integrate import quad


def velocity(x,y,A=.5):
    r=np.hypot(x,y);theta=np.mod(np.arctan2(y,x),2*np.pi)
    assert np.all(r>0)
    speed=A*r**(-1/3)
    return np.array([speed*np.cos(theta/3),speed*np.sin(theta/3)])


def psi(x,y,A=.5):
    return 1.5*A*np.hypot(x,y)**(2/3)*np.sin(2*np.mod(np.arctan2(y,x),2*np.pi)/3)


def integrate(fun,lo,hi):return quad(fun,lo,hi,epsabs=1e-11,epsrel=1e-11)[0]


def mean_velocity(x,y,dx=.5,A=.5):
    west,east=x-dx/2,x+dx/2;south,north=y-dx/2,y+dx/2
    u=(integrate(lambda t:psi(t,north,A),west,east)-integrate(lambda t:psi(t,south,A),west,east))/dx**2
    v=-(integrate(lambda t:psi(east,t,A),south,north)-integrate(lambda t:psi(west,t,A),south,north))/dx**2
    return np.array([u,v])


def face_flux(x,y,axis,dx=.5,A=.5,h=.5,g=9.81):
    # Fully integrated conservative flux per metre; global positive axis.
    lo=(y if axis==0 else x)-dx/2;hi=lo+dx
    def value(t):
        uv=velocity(x,t,A) if axis==0 else velocity(t,y,A)
        out=h*uv[axis]*np.r_[1.,uv];out[axis+1]+=.5*g*h*h
        return out
    # quad handles integrable axis-endpoint singularities without sampling 0.
    return np.array([integrate(lambda t:float(value(t)[k]),lo,hi)/dx for k in range(3)])


def mean_rate(x,y,dx=.5,A=.5,h=.5):
    return -(face_flux(x+dx/2,y,0,dx,A,h)-face_flux(x-dx/2,y,0,dx,A,h)
             +face_flux(x,y+dx/2,1,dx,A,h)-face_flux(x,y-dx/2,1,dx,A,h))/dx


def point_acceleration(x,y,A=.5):
    # -(u dot grad)u = -grad(|u|^2/2) for this irrotational field.
    return A*A/3*np.array([x,y])*np.hypot(x,y)**(-8/3)


def projected_point_acceleration(x,y,dx=.5,A=.5):
    return np.array([.5*(point_acceleration(x-dx/2,y,A)[0]+point_acceleration(x+dx/2,y,A)[0]),
                     .5*(point_acceleration(x,y-dx/2,A)[1]+point_acceleration(x,y+dx/2,A)[1])])
