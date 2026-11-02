"""Conservative midpoint step doubling with local, cellwise error control.

The accepted path is two half steps, never extrapolated. The raw discrepancy
is used without /3 because wet/dry inventory switches need not be second order.
These local tolerances are not a guarantee on accumulated global error.
"""
import numpy as np
from .bend_methods_v1 import BendTrialRejected


def error_norm(xp,coarse,fine,depth_atol,relative,momentum_atol):
    scale=xp.maximum(xp.abs(coarse),xp.abs(fine))*relative
    scale[0]+=depth_atol
    scale[1:]+=momentum_atol
    return float(xp.max(xp.abs(coarse-fine)/scale))


def advance(self,base,dt,at):
    hint=self.step_hint or dt;elapsed=0.;q=base;water=None;steps=[];rejected=[];estimates=[]
    cache=None;easy=self.easy_steps;ev=self.evaluations;mv=self.matvecs;trials=0
    while elapsed<dt*(1-1e-14):
        self.solver.check()
        interval=min(hint,dt-elapsed);trials+=1
        try:
            if cache is not None and cache[0]==interval:
                coarse=cache[1];cache=None
            else:coarse,_,_=self._single(q,interval,at+elapsed)
            half,w1,d1=self._single(q,.5*interval,at+elapsed)
            fine,w2,d2=self._single(half,.5*interval,at+elapsed+.5*interval)
            error=error_norm(self.xp,coarse,fine,self.depth_atol,self.temporal_rtol,self.momentum_atol)
            if not np.isfinite(error):raise BendTrialRejected('Nonfinite temporal error')
        except BendTrialRejected as exc:
            rejected.append(dict(time=at+elapsed,dt=interval,kind='nonlinear',reason=str(exc)))
            hint=.5*interval;cache=None;easy=0
        else:
            if error>1.:
                rejected.append(dict(time=at+elapsed,dt=interval,kind='accuracy',error=error))
                hint=.5*interval;cache=(hint,half);easy=0
            else:
                q=fine;integral=.5*interval*(w1+w2)
                water=integral if water is None else water+integral
                steps.extend((dict(dt=.5*interval,**d1),dict(dt=.5*interval,**d2)))
                estimates.append(error);elapsed+=interval
                # Grow only after two comfortably accurate trials. A shortened
                # output/schedule tail must not poison the persistent hint.
                if interval>=.99*hint:
                    easy=easy+1 if error<.15 else 0
                    if easy>=2:hint=2*interval;easy=0
                cache=None
        if hint<dt/65536 or trials>256:
            raise BendTrialRejected('Automatic temporal subdivision budget')
    self.step_hint=float(hint);self.easy_steps=easy
    d=dict(steps[-1]);d.pop('dt')
    d.update(hydraulic_substeps=len(steps),hydraulic_preempted_retries=0,
        hydraulic_minimum_dt=min(r['dt'] for r in steps),hydraulic_rejections=rejected,
        nonlinear_iterations=sum(r['nonlinear_iterations'] for r in steps),
        nonlinear_norm_residual=max(r['nonlinear_norm_residual'] for r in steps),
        nonlinear_residual_ratio=max(r['nonlinear_norm_residual']/r['nonlinear_tolerance'] for r in steps),
        water_donor_factor_min=min(r['water_donor_factor_min'] for r in steps),
        temporal_error_ratio_max=max(estimates),temporal_trials=trials,
        fwave_evaluations=self.evaluations-ev,linear_matvecs=self.matvecs-mv,
        hydraulic_next_substep_hint_s=self.step_hint)
    return q,water/dt,d
