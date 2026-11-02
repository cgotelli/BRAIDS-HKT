"""Identified Cartesian reconstruction of JC/de Vriend, NOT author-code parity.

Normalized zeta in (0,1); integrals F1,F2 have VARIABLE upper limit zeta.
Covariance enters momentum with -div(hD), as required by depth integration.
This explicitly differs from the inconsistent printed +div sign in JC2026.
No public menu, wet/dry, restart or AMR admission. Installed Desktop untouched.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .jiang_cao_coupled import JiangCaoCoupled
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='secondary.de_vriend1977_jc2026_cartesian198_v1'
NAMES=('jc_secondary','jc_dispersion_faces','jc_dispersion_divergence')


class SecondaryApplicabilityError(BendTrialRejected):
    """A constitutive envelope rejection, not a hydraulic CFL diagnosis."""
    def __init__(self,index,hk,profile_a,depth):
        self.index,self.hk,self.profile_a,self.depth=index,hk,profile_a,depth
        super().__init__(f'Secondary closure outside qualified mild-curvature/log-profile range at input cell {index}: '
            f'abs(h*k)={abs(hk):.9g} (limit 0.1), a={profile_a:.9g} (require a<0.5), h={depth:.9g} m; '
            'no clipping applied. Inspect local flow curvature and startup/boundary compatibility; '
            'a smaller timestep does not extend this constitutive envelope.')


def profile_basis(zeta):
    """Reference evaluation only, never called in a cell/timestep kernel."""
    from scipy.special import spence
    from scipy.integrate import quad
    z=np.asarray(zeta,dtype=float)
    if np.any((z<=0)|(z>1)|~np.isfinite(z)):raise ValueError('0 < normalized height <= 1 required')
    log=np.log(z);L=1.+log
    f1=spence(1.-z)+np.where(z<1,log*np.log1p(-np.minimum(z,1.-1e-16)),0.)
    # Transform to t=-log(zeta) for smooth integration; expm1 avoids cancellation.
    f2=np.array([-quad(lambda t:t*t/np.expm1(t) if t<700 else 0.,-np.log(v),
                       np.inf,epsabs=1e-12,epsrel=1e-12)[0] for v in z.flat]).reshape(z.shape)
    return L,np.array([2*(f1-1),f2+2-2*L,2*L])


@lru_cache(maxsize=4)
def profile_moments(order=192):
    """Gauss-Laguerre integration: integral_0^1 f(z) dz=int_0^inf f(e^-t)e^-t dt."""
    from scipy.special import roots_laguerre
    t,w=roots_laguerre(order);keep=t<700
    L,G=profile_basis(np.exp(-t[keep]));w=w[keep]
    return np.r_[G@(w*L),((G*w)@G.T).ravel()]


class SecondaryOperator:
    def __init__(self,backend,xp,threads=4):
        self.xp=xp;self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.mom=xp.asarray(profile_moments())
        self.identity=dict(method=METHOD,runtime=self.engine.identity,wrapper_sha256=digest(__file__),
            profile_moments=profile_moments().tolist(),kappa=.4,curvature='positive left; physical Cartesian velocity Jacobian',
            dispersion='negative divergence of full positive covariance; local frame rotated to Cartesian',
            bed_direction='tan(phi_left)=7*h*k-1.5*slope_left*sqrt(theta_c/theta)',
            guard='fully wet; sqrt(g)/(kappa*C)<0.5; abs(h*k)<=0.1 qualification envelope, no clipping',
            source_limit='normalized-height and sign-consistent reconstruction; not verified author implementation')

    def evaluate(self,q,gradient,manning,cap,gravity=9.81,active=True):
        xp=self.xp;n=q.shape[1]
        q=xp.ascontiguousarray(xp.asarray(q,dtype=xp.float64));gradient=xp.ascontiguousarray(xp.asarray(gradient,dtype=xp.float64))
        cap=xp.ascontiguousarray(xp.asarray(cap,dtype=xp.float64))
        manning=xp.ascontiguousarray(xp.broadcast_to(xp.asarray(manning,dtype=xp.float64),(n,)))
        if q.shape!=(3,n) or gradient.shape!=(4,n) or cap.shape!=(23,n):raise ValueError('q(3,n), velocity Jacobian(4,n), capacity(23,n) required')
        if not np.isfinite(gravity) or gravity<=0 or any(not bool(xp.all(xp.isfinite(v))) for v in (q,gradient,manning,cap)):
            raise ValueError('Finite secondary-flow inputs and positive gravity required')
        if bool(xp.any(q[0]<=0)|xp.any(manning<=0)|xp.any(cap[3:5]<0)):raise ValueError('Positive depth/roughness and nonnegative Shields required')
        out=xp.empty((9,n));self.engine.launch('jc_secondary',n,n,float(gravity),int(active),q,gradient,manning,cap,self.mom,out)
        if not bool(xp.all(xp.isfinite(out))):raise BendTrialRejected('Nonfinite secondary closure')
        if active and bool(xp.any(out[2]>=.5)|xp.any(xp.abs(out[1])>.1)):
            index=int(xp.argmax(xp.maximum(xp.abs(out[1])/.1,out[2]/.5)))
            raise SecondaryApplicabilityError(index,float(out[1,index]),float(out[2,index]),float(q[0,index]))
        return out


class JiangCaoSecondary(JiangCaoCoupled):
    def __init__(self,*args,secondary_enabled=True,**kwargs):
        if not isinstance(secondary_enabled,bool):raise ValueError('Explicit boolean secondary switch required')
        super().__init__(*args,**kwargs)
        self.secondary_enabled=secondary_enabled
        self.secondary=SecondaryOperator(self.backend,self.xp,self.a.physics.flow.threads)
        self.identity=dict(self.identity,method='coupled.jc2026_p15_secondary198_v1',
            secondary=self.secondary.identity,secondary_enabled=secondary_enabled,
            limits='fully wet fixed mesh; one grain; MF1; mild curvature; no collapse/restart/AMR')

    def relative_discharge(self,s,cap):
        if not self.secondary_enabled:return super().relative_discharge(s,cap)
        velocity=s[1:3]/s[0];gradient=self.xp.vstack([self.mesh.gradient(v) for v in velocity])
        self.secondary_fields=self.secondary.evaluate(s[:3],gradient,self.manning,cap.values,
            self.exchange.material.capacity_material.gravity_m_s2)
        return self.secondary_fields[7:].copy()

    def momentum_correction(self,s,rate):
        if not self.secondary_enabled:return
        xp=self.xp;g=self.mesh.g;n=self.n;nf=self.nf
        tensor=xp.ascontiguousarray(s[0]*self.secondary_fields[3:6])
        grad=xp.vstack([self.mesh.gradient(t) for t in tensor]);flux=xp.empty((2,nf))
        e=self.secondary.engine
        e.launch('jc_dispersion_faces',nf,n,nf,g['owner'],g['neighbour'],g['kind'],g['length'],
            g['normal'],g['xy'],g['mid'],tensor,grad,flux)
        e.launch('jc_dispersion_divergence',n,n,nf,g['offset'],g['faces'],g['owner'],g['area'],flux,rate)
        self.dispersion_flux=flux
