"""Isolated nonlinear replay; never changes installed or maintained solver code."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'validation/_wetfront121'


def solve(self,base,dt,at,variant):
    """Same midpoint equation, positivity and acceptance; alternative directions."""
    from braids_core.sediment.bend_methods_v1 import BendTrialRejected
    xp=self.xp;tau=.5*dt;mid=base.copy();self.interval_base=base;self.minimum_theta=1.
    nrm=lambda v:float(xp.linalg.norm(v));evaluations=0;calls=0
    def residual(v):
        nonlocal evaluations
        r,w=self.rate(v,dt,at+tau);evaluations+=1
        return v-base-tau*r,r,w
    defect,rate,water=residual(mid)
    for iteration in range(40):
        size=nrm(defect);tol=max(self.atol*np.sqrt(base.size),self.rtol*max(nrm(mid-base),tau*nrm(rate)))
        if size<=tol:break
        if variant=='current_depth':
            delta,c=self.linear(-defect,mid,tau,max(.2*tol,1e-15));calls+=c
        else:
            # Restarted, backend-resident Jacobian-free GMRES, used only as
            # nonlinear direction. Exact residual is always the acceptance gate.
            v=[-defect/size];h=xp.zeros((25,24));delta=xp.zeros_like(mid)
            cs=xp.zeros(24);sn=xp.zeros(24);rhs=xp.zeros(25);rhs[0]=size
            for k in range(24):
                calls+=1;self.matvecs+=1
                eps=1e-7*max(1.,nrm(mid))
                trial=mid+eps*v[k]
                trial[0]=xp.maximum(trial[0],0.)
                jv=(residual(trial)[0]-defect)/eps
                for i in range(k+1):
                    h[i,k]=xp.sum(jv*v[i]);jv-=h[i,k]*v[i]
                h[k+1,k]=nrm(jv)
                beta=h[k+1,k].copy()
                for i in range(k):
                    a,b=h[i,k].copy(),h[i+1,k].copy()
                    h[i,k]=cs[i]*a+sn[i]*b;h[i+1,k]=-sn[i]*a+cs[i]*b
                den=xp.sqrt(h[k,k]**2+h[k+1,k]**2)
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
        if not found:raise BendTrialRejected(f'{variant} line search failed norm={size:g},tol={tol:g},it={iteration}')
    else:raise BendTrialRejected(f'{variant} iterations exceeded norm={size:g}')
    endpoint=base+dt*rate
    if not bool(xp.all(xp.isfinite(endpoint))) or bool(xp.any(endpoint[0]<0)):
        raise BendTrialRejected('Negative midpoint endpoint')
    endpoint[1:,endpoint[0]<=self.solver.physics.parameters.dry]=0.
    return endpoint,water,dict(integrator='implicit_fwave_midpoint_river_v1',nonlinear_iterations=iteration,
        nonlinear_norm_residual=size,nonlinear_tolerance=tol,linear_matvecs=calls,fwave_evaluations=evaluations,
        explicit_fallback=False,water_donor_factor_min=self.minimum_theta)


def recover_advance(self,base,dt,at):
    """Research controller: recover within a physical transaction, not after 8."""
    from braids_core.sediment.bend_methods_v1 import BendTrialRejected
    hint=self.step_hint or dt;easy=0;elapsed=0.;q=base;steps=[];rejected=[];water=None
    ev=self.evaluations;mv=self.matvecs
    while elapsed<dt*(1-1e-14):
        interval=min(hint,dt-elapsed)
        try:nextq,w,d=self._single(q,interval,at+elapsed)
        except BendTrialRejected as exc:
            rejected.append(dict(time=at+elapsed,dt=interval,reason=str(exc)))
            hint=.5*interval;easy=0
            if hint<dt/65536:raise
            continue
        q=nextq;water=interval*w if water is None else water+interval*w
        steps.append(dict(dt=interval,**d));elapsed+=interval
        if len(steps)>128:raise BendTrialRejected('Research recovery subdivision budget')
        if interval>=.99*hint and d['nonlinear_iterations']<=8:
            easy+=1
            if easy==4:hint*=2;easy=0
        else:easy=0
    self.step_hint=float(hint);self.easy_steps=0
    d=dict(steps[-1]);d.pop('dt')
    d.update(hydraulic_substeps=len(steps),hydraulic_preempted_retries=0,
        hydraulic_minimum_dt=min(r['dt'] for r in steps),hydraulic_rejections=rejected,
        nonlinear_iterations=sum(r['nonlinear_iterations'] for r in steps),
        nonlinear_norm_residual=max(r['nonlinear_norm_residual'] for r in steps),
        nonlinear_residual_ratio=max(r['nonlinear_norm_residual']/r['nonlinear_tolerance'] for r in steps),
        water_donor_factor_min=min(r['water_donor_factor_min'] for r in steps),
        fwave_evaluations=self.evaluations-ev,linear_matvecs=self.matvecs-mv,
        hydraulic_next_substep_hint_s=self.step_hint)
    return q,water/dt,d


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('cpu','gpu'),default='gpu');ap.add_argument('--label',required=True)
    ap.add_argument('--until',type=float,default=.75);ap.add_argument('--variant',default='baseline')
    ap.add_argument('--cap',type=float);ap.add_argument('--outer-dt',type=float)
    ap.add_argument('--substep-drag',action='store_true')
    ap.add_argument('--unlimited-flux',action='store_true')
    ap.add_argument('--recover',action='store_true')
    ap.add_argument('--atol',type=float)
    ap.add_argument('--pytest',nargs='+',help='Exercise research override on existing tests instead of a river run')
    ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    from braids_core.sediment.river_midpoint import RiverMidpoint
    from braids_core.sediment.bed_columns_v101 import host
    from braids_core.sediment.bend_methods_v1 import BendTrialRejected
    executed_source=Path(__file__).read_bytes()
    original=RiverMidpoint._solve_single;records=[]
    if args.recover:RiverMidpoint.advance=recover_advance
    out=BASE/args.label
    if args.unlimited_flux or args.atol:
        init=RiverMidpoint.__init__
        def initialize(self,*a,**kw):
            init(self,*a,**kw)
            # Diagnostic: retain native f-wave water/momentum and strict
            # endpoint positivity rejection, without old-inventory clipping.
            if args.unlimited_flux:self.positive.launch=lambda *a,**kw:None
            if args.atol:self.atol=args.atol
        RiverMidpoint.__init__=initialize
    if args.substep_drag:
        from braids_core.sediment.shoreline_solver import ShorelineSolver
        oldstep=ShorelineSolver.step
        def step(self,*a,**kw):
            if not hasattr(self,'_research_drag'):
                self._research_drag=self.physics.friction
                # Retain accepted-start predictor and roughness refresh; move
                # only the duration of hydraulic drag to hydraulic substeps.
                self.physics.friction=lambda q,dt,*extra:self._research_drag(q,0.,*extra)
            return oldstep(self,*a,**kw)
        ShorelineSolver.step=step
    def measured(self,base,dt,at):
        if args.cap and dt>args.cap*(1+1e-12):raise BendTrialRejected('Research hydraulic cap')
        if args.substep_drag:
            base=base.copy();self.solver._research_drag(base,.5*dt)
        if args.variant!='baseline':
            result=solve(self,base,dt,at,args.variant)
            if args.substep_drag:self.solver._research_drag(result[0],.5*dt)
            return result
        try:
            result=original(self,base,dt,at)
            if args.substep_drag:self.solver._research_drag(result[0],.5*dt)
            return result
        except BendTrialRejected as exc:
            if not args.replay:raise
            tb=exc.__traceback__;loc=None
            while tb:
                if tb.tb_frame.f_code.co_name=='_solve_single':loc=tb.tb_frame.f_locals.copy()
                tb=tb.tb_next
            row=dict(time=float(at),dt=float(dt),error=str(exc))
            if loc is not None:
                defect=host(loc['defect']);q=host(loc['mid']);idx=np.argsort(abs(defect).max(axis=0))[-8:][::-1]
                d=self.solver.data;area=host(self.solver.physics.g['area'])
                row['cells']=[dict(cell=int(i),base=host(base)[:,i].tolist(),mid=q[:,i].tolist(),
                    defect=defect[:,i].tolist(),theta=float(host(self.theta)[i]),z=float(host(self.solver.z)[i]),
                    xy=d['xy'][:,i].tolist(),area=float(area[i])) for i in idx]
                np.savez(out/f'FAIL_{len(records):02d}.npz',base=host(base),mid=q,defect=defect,
                    rate=host(loc['rate']),water=host(loc['water']),dt=dt,time=at)
            row['trials']=[]
            for variant in ('current_depth','jfnk'):
                start=time.perf_counter();before=self.evaluations
                try:
                    result=solve(self,base,dt,at,variant)
                    trial=dict(variant=variant,ok=True,diagnostics=result[2])
                    np.savez(out/f'REPLAY_{len(records):02d}_{variant}.npz',q=host(result[0]))
                except Exception as err:trial=dict(variant=variant,ok=False,error=str(err))
                trial.update(wall_s=time.perf_counter()-start,evaluations=self.evaluations-before)
                row['trials'].append(trial)
            records.append(row)
            print(json.dumps(row),flush=True)
            raise
    RiverMidpoint._solve_single=measured
    from tools import profile_test1_amr
    sys.argv=[profile_test1_amr.__file__,'--source',str(ROOT/'validation/_axis_boundary120/aligned_prepared'),
        '--output',str(out),'--backend',args.backend,'--until',str(args.until),'--checkout']
    if args.outer_dt:sys.argv+=['--dt-max',str(args.outer_dt)]
    try:
        if args.pytest:
            import pytest
            out.mkdir(parents=True,exist_ok=False)
            result=pytest.main([*args.pytest,'-q','--junitxml='+str(out/'TESTS.xml')])
            if result:raise SystemExit(result)
        else:profile_test1_amr.main()
    finally:
        (out/'EXECUTED_RESEARCH.py').write_bytes(executed_source)
        (out/'RESEARCH.json').write_text(json.dumps(dict(settings=vars(args),replays=records,
            research_source_sha256=hashlib.sha256(executed_source).hexdigest(),
            identity_scope='PROFILE runtime identity identifies unchanged production operators; this source and settings identify process-local research changes.'),indent=2))


if __name__=='__main__':main()
