"""Isolated automatic temporal control, before maintained-source adoption.

Full versus two half midpoint/drag steps; accept the conservative half path,
without Richardson extrapolation or assuming second order at wet/dry switches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RECORDS=[]


def error_norm(xp,coarse,fine,depth_atol,relative,momentum_atol):
    scale=xp.maximum(xp.abs(coarse),xp.abs(fine))*relative
    scale[0]+=depth_atol
    scale[1:]+=momentum_atol
    return float(xp.max(xp.abs(coarse-fine)/scale))


def advance(self,base,dt,at):
    from braids_core.sediment.bend_methods_v1 import BendTrialRejected
    hint=self.step_hint or dt;elapsed=0.;q=base;water=None;steps=[];rejected=[];estimates=[]
    cache=None;easy=self.easy_steps;dyadic=getattr(self,'dyadic',False)
    ev=self.evaluations;mv=self.matvecs;trials=0
    while elapsed<dt*(1-1e-14):
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
            hint=.5*interval
            cache=None;easy=0
            RECORDS.append(dict(time=at+elapsed,dt=interval,kind='nonlinear',reason=str(exc)))
        else:
            # Square-root reduction is conservative where the donor limiter
            # lowers the local order. No /3 or extrapolation at a wet front.
            factor=min(2.,max(.2,.85/max(error,1e-12)**.5))
            if error>1.:
                rejected.append(dict(time=at+elapsed,dt=interval,kind='accuracy',error=error))
                hint=.5*interval if dyadic else interval*min(.8,factor)
                cache=(hint,half) if dyadic else None;easy=0
            else:
                q=fine;integral=.5*interval*(w1+w2)
                water=integral if water is None else water+integral
                steps.extend((dict(dt=.5*interval,**d1),dict(dt=.5*interval,**d2)))
                estimates.append(error);elapsed+=interval
                # A short output-clock tail must not shrink the next hint.
                if dyadic:
                    if interval>=.99*hint:
                        easy=easy+1 if error<.15 else 0
                        if easy>=2:hint=2*interval;easy=0
                else:hint=interval*factor if interval>=.99*hint or factor<1 else max(hint,interval*factor)
                cache=None
            RECORDS.append(dict(time=at+elapsed-(interval if error<=1 else 0),dt=interval,
                kind='accepted' if error<=1 else 'accuracy',error=error))
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


def main():
    from braids_core.sediment.current_river_solver import CurrentRiverControls
    if hasattr(CurrentRiverControls,'temporal_depth_atol'):
        raise SystemExit('Historical pre-adoption experiment: replay only with its frozen runtime. '
                         'Use deliver_accuracy122.py bench for the maintained solver; do not patch drag twice.')
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True)
    ap.add_argument('--backend',choices=('cpu','gpu'),default='gpu')
    ap.add_argument('--until',type=float,default=3.75)
    ap.add_argument('--depth-atol',type=float,default=1e-4)
    ap.add_argument('--momentum-atol',type=float,default=1e-3)
    ap.add_argument('--relative',type=float,default=1e-3)
    ap.add_argument('--outer-dt',type=float)
    ap.add_argument('--dyadic',action='store_true')
    ap.add_argument('--picard',action='store_true')
    ap.add_argument('--pytest',nargs='+')
    args=ap.parse_args()
    from tools.wetfront121 import solve
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.shoreline_solver import ShorelineSolver
    oldinit=RiverMidpoint.__init__;oldstep=ShorelineSolver.step
    def initialize(self,*a,**kw):
        oldinit(self,*a,**kw);self.atol=1e-7
        self.depth_atol=args.depth_atol;self.momentum_atol=args.momentum_atol;self.temporal_rtol=args.relative
        self.dyadic=args.dyadic
    def step(self,*a,**kw):
        if not hasattr(self,'_research_drag'):
            self._research_drag=self.physics.friction
            self.physics.friction=lambda q,dt,*extra:self._research_drag(q,0.,*extra)
        return oldstep(self,*a,**kw)
    def single(self,q,dt,at):
        base=q.copy();drag=getattr(self.solver,'_research_drag',None)
        if drag is not None:drag(base,.5*dt)
        if args.picard:
            from tools.nonlinear122 import solve as efficient_solve
            result=efficient_solve(self,base,dt,at)
        else:result=solve(self,base,dt,at,'jfnk')
        if drag is not None:drag(result[0],.5*dt)
        return result
    RiverMidpoint.__init__=initialize;RiverMidpoint.advance=advance;RiverMidpoint._solve_single=single
    ShorelineSolver.step=step
    from tools import profile_test1_amr
    out=ROOT/'validation/_accuracy122'/args.label
    source=Path(__file__).read_bytes()
    sys.argv=[profile_test1_amr.__file__,'--source',str(ROOT/'validation/_axis_boundary120/aligned_prepared'),
        '--output',str(out),'--backend',args.backend,'--until',str(args.until),'--checkout']
    if args.outer_dt:sys.argv+=['--dt-max',str(args.outer_dt)]
    try:
        if args.pytest:
            import pytest
            out.mkdir(parents=True,exist_ok=False)
            status=pytest.main([*args.pytest,'-q','--junitxml='+str(out/'TESTS.xml')])
            if status:raise SystemExit(status)
        else:profile_test1_amr.main()
    finally:
        (out/'EXECUTED_RESEARCH.py').write_bytes(source)
        (out/'TEMPORAL_TRIALS.json').write_text(json.dumps(RECORDS,indent=2))
        (out/'RESEARCH.json').write_text(json.dumps(dict(settings=vars(args),
            source_sha256=hashlib.sha256(source).hexdigest(),
            efficient_nonlinear_source_sha256=hashlib.sha256((ROOT/'tools/nonlinear122.py').read_bytes()).hexdigest(),
            nonlinear_source_sha256=hashlib.sha256((ROOT/'tools/wetfront121.py').read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
