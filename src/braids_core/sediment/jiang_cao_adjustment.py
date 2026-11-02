"""Named KB86 finite-adjustment extension of the JC reconstructed profiles.

Not the Jiang-Cao author model: I_e=U*h*k; L=h*(1-2a)/(2*kappa^2*a),
T=L/U, D_t I=(I_e-I)/T. No horizontal turbulent diffusion in this bundle.
State carries h*I; flux uses the actual hydraulic discharge and a bounded
second-order scalar reconstruction. No wall transport; incoming open-face I=0.
"""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='secondary.kb86_jc_profiles_adjustment206_v1'


class AdjustmentOperator:
    def __init__(self,owner):
        self.owner=owner;self.xp=owner.xp
        self.engine=WaveEngine(owner.backend,owner.a.physics.flow.threads,
            source=Path(__file__).with_suffix('.h'),names=('jc206_relax','jc206_scalar_faces'))
        self.identity=dict(method=METHOD,runtime=self.engine.identity,source_sha256=digest(__file__),
            state='signed h*I in seventh state row; I has m/s units; initial I=0 explicitly',
            source='I_e=U*h*k; L=h*(1-2*a)/(2*0.4^2*a); lambda=U/L; exact frozen-coefficient relaxation',
            curvature='unique physically wet neighbour LS; shallow hydraulic velocities retained; no resolution weight or curvature clipping',
            momentum='de Vriend covariance with A=I/0.4^2; bed direction 7*I/U plus unchanged slope term',
            boundary='I=0 on all incoming open faces; donor export; zero wall flux',
            shallow='zero equilibrium forcing below h_ref; relaxation coefficients matched at max(h,h_ref); dry intensity dissipates',
            source_coupling='exchange/collapse preserves I; relaxation uses midpoint hydraulic/bed source state',
            transport='limited linear scalar plus shared-face antidiffusive bounds; same SSPRK22 hydraulic flux',
            limits='research extension, not author reproduction; zero horizontal diffusion; unchanged effective abs(I/U)<=0.1 and a<0.5 guards')

    def relax(self,state,dt,coefficient_state=None):
        s=self.owner;xp=self.xp;st=state.copy();q=st if coefficient_state is None else coefficient_state
        wet=q[0]>s.dry;velocity=q[1:3]/xp.where(wet,q[0],1.)
        # Constitutive activation belongs to the receiving cell, not to the
        # hydraulic stencil. A shallow wet neighbour still supplies a physical
        # velocity. Excluding it can leave nearly collinear transverse samples
        # on cropped cells and invent a large streamwise velocity derivative.
        raw=s.direction.evaluate(velocity,wet)
        n=s.shallow.resistance(s.roughness,q[:3])
        hr=s.shallow.reference_submergence*s.roughness.parameters.d90_m
        gravity=s.exchange.material.capacity_material.gravity_m_s2
        hcoef=xp.maximum(q[0],hr);a=xp.sqrt(gravity)*n/(.4*hcoef**(1/6))
        if bool(xp.any(a>=.5)):raise BendTrialRejected('KB86 adjustment requires positive length (a<0.5)')
        self.engine.launch('jc206_relax',s.n,s.n,float(dt),s.dry,hr,q,raw,a,st)
        return st

    def rate(self,st,Q,dh,dt):
        s=self.owner;xp=self.xp;n,nf=s.n,s.nf;g=s.mesh.g
        wet=st[0]>s.dry;I=st[6]/xp.where(wet,st[0],1.)
        # Translate the signed scalar to a positive bounded concentration for
        # the already qualified shared-face antidiffusive limiter. The inverse
        # translation cancels the hydraulic continuity contribution exactly.
        lo=min(0.,float(xp.min(I)));hi=max(0.,float(xp.max(I)))
        if hi==lo:return xp.zeros(n)
        pad=64*np.finfo(float).eps*max(abs(lo),abs(hi));lo-=pad;hi+=pad;width=hi-lo
        c=xp.ascontiguousarray((I-lo)/width);m=st[0]*c
        grad=s.front.gradient(c,wet,limited=True);low=xp.empty(nf);high=xp.empty(nf)
        self.engine.launch('jc206_scalar_faces',nf,n,nf,float(-lo/width),g['owner'],g['neighbour'],
            g['kind'],g['xy'],g['mid'],Q,c,grad,low,high)
        H=xp.empty(n);M=xp.empty(n);plus=xp.empty(n);minus=xp.empty(n);bad=xp.empty(n,dtype=xp.int32)
        e=s.front.engine
        e.launch('jc201_bounds',n,n,float(dt),1.,g['offset'],g['faces'],g['owner'],g['area'],
            st[0],m,Q,low,high,H,M,plus,minus,bad)
        if bool(xp.any(bad)):raise BendTrialRejected('Secondary scalar donor bound/CFL failed')
        flux=xp.empty(nf);theta=xp.empty(nf);end=xp.empty(n)
        e.launch('jc201_correct',nf,nf,g['owner'],g['neighbour'],low,high,plus,minus,flux,theta)
        # Recover the signed physical intensity flux BEFORE divergence to avoid
        # subtracting nearly equal O(h) scalar states at small timesteps.
        flux=width*flux+lo*Q
        s.engine.launch('jc_mobile_divergence',n,n,g['offset'],g['faces'],g['owner'],g['area'],flux,end)
        return end
