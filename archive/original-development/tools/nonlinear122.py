"""Full-residual midpoint direction with a cheap convergent Picard trial."""
import numpy as np
from braids_core.sediment.bend_methods_v1 import BendTrialRejected


def solve(self,base,dt,at):
    xp=self.xp;tau=.5*dt;mid=base.copy();self.interval_base=base;self.minimum_theta=1.
    nrm=lambda v:float(xp.linalg.norm(v));evaluations=0;calls=0;picard=0
    def residual(v):
        nonlocal evaluations
        r,w=self.rate(v,dt,at+tau);evaluations+=1
        return v-base-tau*r,r,w
    defect,rate,water=residual(mid)
    for iteration in range(40):
        size=nrm(defect);tol=max(self.atol*np.sqrt(base.size),self.rtol*max(nrm(mid-base),tau*nrm(rate)))
        if size<=tol:break
        trial=mid-defect;trial[0]=xp.maximum(trial[0],0.)
        pd,pr,pw=residual(trial)
        if nrm(pd)<=max(.25*size,tol):
            mid,defect,rate,water=trial,pd,pr,pw;picard+=1
            continue
        v=[-defect/size];h=xp.zeros((25,24));delta=xp.zeros_like(mid)
        cs=xp.zeros(24);sn=xp.zeros(24);rhs=xp.zeros(25);rhs[0]=size
        for k in range(24):
            calls+=1;self.matvecs+=1
            eps=1e-7*max(1.,nrm(mid))
            trial=mid+eps*v[k];trial[0]=xp.maximum(trial[0],0.)
            jv=(residual(trial)[0]-defect)/eps
            for i in range(k+1):
                h[i,k]=xp.sum(jv*v[i]);jv-=h[i,k]*v[i]
            h[k+1,k]=nrm(jv);beta=h[k+1,k].copy()
            for i in range(k):
                a,b=h[i,k].copy(),h[i+1,k].copy()
                h[i,k]=cs[i]*a+sn[i]*b;h[i+1,k]=-sn[i]*a+cs[i]*b
            den=xp.sqrt(h[k,k]**2+h[k+1,k]**2)
            if float(den)<1e-30:raise BendTrialRejected('Full-residual Krylov breakdown')
            cs[k]=h[k,k]/den;sn[k]=h[k+1,k]/den
            h[k,k]=den;h[k+1,k]=0.
            rhs[k+1]=-sn[k]*rhs[k];rhs[k]=cs[k]*rhs[k]
            if abs(float(rhs[k+1]))<=max(.1*size,.2*tol) or float(beta)<1e-14 or k==23:
                y=xp.zeros(k+1)
                for i in reversed(range(k+1)):y[i]=(rhs[i]-xp.sum(h[i,i+1:k+1]*y[i+1:]))/h[i,i]
                delta=sum(y[i]*v[i] for i in range(k+1));break
            v.append(jv/beta)
        found=False
        for direction in (delta,-defect):
            for k in range(12):
                trial=mid+2.**(-k)*direction;trial[0]=xp.maximum(trial[0],0.)
                d,r,w=residual(trial)
                if nrm(d)<size or nrm(d)<=tol:
                    mid,defect,rate,water=trial,d,r,w;found=True;break
            if found:break
        if not found:raise BendTrialRejected(f'Full-residual line search failed norm={size:g},tol={tol:g},it={iteration}')
    else:raise BendTrialRejected(f'Full-residual iterations exceeded norm={size:g}')
    endpoint=base+dt*rate
    if not bool(xp.all(xp.isfinite(endpoint))) or bool(xp.any(endpoint[0]<0)):
        raise BendTrialRejected('Negative midpoint endpoint')
    endpoint[1:,endpoint[0]<=self.solver.physics.parameters.dry]=0.
    return endpoint,water,dict(integrator='implicit_fwave_midpoint_river_v1',nonlinear_iterations=iteration,
        nonlinear_norm_residual=size,nonlinear_tolerance=tol,linear_matvecs=calls,fwave_evaluations=evaluations,
        picard_directions=picard,explicit_fallback=False,water_donor_factor_min=self.minimum_theta)
