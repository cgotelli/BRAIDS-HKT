"""One augmented-f-wave midpoint algorithm on NumPy/OpenMP and CuPy/CUDA.

Full-residual Newton directions and a temporal estimator retain the same
hydraulic flux. Physical processes are owned by ShorelineSolver's transaction.
There is no explicit fallback, artificial water film or backend selection.
"""
import ctypes
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected, digest
from . import bend_native_hydraulics as native

METHOD='implicit_fwave_midpoint_river_v1'


_status_reduction=None


def spatial_status_invalid(xp,statuses):
    """Check every resident status, with one GPU-to-host synchronization.

    Do not cache the result: every residual reconstructs the evolving iterate.
    Reductions retain each status array's own shape and integer dtype.
    """
    # One generic resident reduction instead of eight independent reductions,
    # a stack and a ninth reduction. Retain each input's signed integer dtype;
    # all values (including negative failure codes) participate on every call.
    if xp.__name__=='cupy' and len(statuses)==8:
        global _status_reduction
        if _status_reduction is None:
            parameters=['A s0']
            for j in range(1,8):parameters += [f'raw {chr(65+j)} s{j}',f'int64 n{j}']
            expression='s0 != 0' + ''.join(f' || (_J < n{j} && s{j}[_J] != 0)' for j in range(1,8))
            _status_reduction=xp.ReductionKernel(', '.join(parameters),'bool invalid',expression,
                'a || b','invalid = a','false','river_spatial_status')
        ordered=sorted(statuses,key=lambda s:s.size,reverse=True)
        args=[ordered[0].ravel()]
        for s in ordered[1:]:args += [s.ravel(),np.int64(s.size)]
        return bool(_status_reduction(*args))
    return bool(xp.any(xp.stack([xp.any(s) for s in statuses])))


class RiverMidpoint:
    def __init__(self,solver):
        self.solver=solver;self.xp=solver.xp
        # Honour the same saved tolerances as the existing bend midpoint.
        # Artificially tightening them below the piecewise wet/dry spatial
        # operator's resolution causes repeated threshold-crossing failures.
        self.atol=solver.controls.nonlinear_atol
        self.rtol=solver.controls.nonlinear_rtol
        self.maxiter=40;self.linear_maxiter=24
        self.depth_atol=solver.controls.temporal_depth_atol
        self.momentum_atol=solver.controls.temporal_momentum_atol
        self.temporal_rtol=solver.controls.temporal_rtol
        self.evaluations=0;self.matvecs=0
        self.step_hint=None;self.easy_steps=0
        from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine
        p=solver.physics;xp=self.xp
        self.positive=WaveEngine(solver.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),
            names=('midpoint_inventory','midpoint_faces','midpoint_gather'))
        self.theta=xp.ones(p.n);self.correction=xp.zeros((2,p.nf));self.status=xp.zeros(p.nf,dtype=xp.int32)
        self.identity=dict(method=METHOD,controller_sha256=digest(__file__),
            atol=self.atol,rtol=self.rtol,maxiter=self.maxiter,
            linear_maxiter=self.linear_maxiter,explicit_fallback=False,
            substeps='full versus two half implicit steps; accept conservative half path; one outer physical inventory commit',
            wet_dry='accepted-water donor bound INSIDE midpoint residual; reject prescribed-boundary limitation; no water clipping',
            endpoint_flux_limiter=self.positive.identity,
            state_history='accepted substep hint and easy-step count; checkpointed and rolled back; reset on regrid',
            step_growth='double after two accepted temporal ratios below 0.15; halve rejected trials',
            temporal_depth_atol_m=self.depth_atol,temporal_momentum_atol_m2_s=self.momentum_atol,
            temporal_rtol=self.temporal_rtol,temporal_norm='maximum cell/component raw full-half discrepancy; no Richardson extrapolation',
            nonlinear_method='contractive Picard trial then full-residual finite-difference Newton GMRES',
            iteration_sha256=digest(Path(__file__).with_name('river_midpoint_iteration.py')),
            accuracy_sha256=digest(Path(__file__).with_name('river_midpoint_accuracy.py')),
            friction_clock='Strang drag around every actual hydraulic substep')
        if solver.backend=='gpu':
            from braids_support.gpu_fwave.resident_update import OPTIONS
            path=Path(__file__).resolve().parents[2]/'braids_gpu/implicit_midpoint_fwave.cu'
            name='implicit_full_rate_from_faces'
            self.module=self.xp.RawModule(code=path.read_text(),options=OPTIONS,name_expressions=(name,))
            self.gather=self.module.get_function(name)
            self.identity['gather_sha256']=digest(path)

    def state(self):
        return dict(step_hint=self.step_hint,easy_steps=self.easy_steps)

    @staticmethod
    def validate_state(state):
        if not isinstance(state,dict) or set(state)!={'step_hint','easy_steps'}:
            raise ValueError('Midpoint controller state differs')
        hint=state['step_hint'];easy=state['easy_steps']
        if hint is not None and (type(hint) not in (int,float) or not np.isfinite(hint) or hint<=0):
            raise ValueError('Invalid midpoint step hint')
        if type(easy) is not int or not 0<=easy<2:raise ValueError('Invalid midpoint growth history')

    def restore(self,state):
        self.validate_state(state)
        self.step_hint=state['step_hint'];self.easy_steps=state['easy_steps']

    def rate(self,q,dt,time):
        a=self.solver;p=a.physics;xp=self.xp;h=a.hydraulics
        self.evaluations+=1
        a.check()
        if not bool(xp.all(xp.isfinite(q))) or bool(xp.any(q[0]<0)):
            raise BendTrialRejected('Midpoint iterate is not finite/nonnegative')
        exterior=getattr(a,'exterior_bed',None)
        if exterior is not None:
            exterior.evaluate(q,a.z,time=time)
            boundary_budget=getattr(self,'explicit_boundary_budget',None)
            if boundary_budget is not None:boundary_budget.allocate(self.interval_base,dt)
        if a.backend=='cpu':
            # Same corrected native spatial prefix used by the explicit
            # owner, stopping BEFORE draining and its explicit CFL test.
            q=np.ascontiguousarray(q);z=np.ascontiguousarray(a.z)
            rate=np.empty_like(q);water=np.empty(p.nf);code=ctypes.c_int()
            with native._lock:
                if exterior is None:
                    h.lib.bend_native_rates(q.ctypes.data,z.ctypes.data,float(dt),float(time),h.threads,
                        rate.ctypes.data,water.ctypes.data,ctypes.byref(code))
                else:
                    h.lib.shoreline_native_boundary_rates(q.ctypes.data,z.ctypes.data,float(dt),float(time),h.threads,
                        exterior.selected.ctypes.data,exterior.flux.ctypes.data,rate.ctypes.data,water.ctypes.data,ctypes.byref(code))
            h.calls+=1
            if code.value:raise BendTrialRejected('Native midpoint spatial residual rejected')
        else:
            # Boundary sections authenticate a stable resident stage buffer.
            # Copy values into that buffer; never rebind its storage per Newton call.
            h.stepper.stage['q'][...]=q;q=h.stepper.stage['q']
            h.boundary.begin_trial()
            h.reconstruction.geometry['bed'][...]=a.z
            rec=h.reconstruction.reconstruct(q)
            if exterior is None:h.boundary.apply(q,rec['trace'])
            else:exterior.apply_gpu_traces(q,rec['trace'])
            secondary=h.secondary.reconstruct(q,a.z,h.state['secondary'])
            h.flux.inputs['q']=q
            # Execute only the authenticated spatial prefix. dt=0 is NOT a
            # safe way to disable an Euler limiter: a dry donor can produce
            # 0/0 in its inventory ratio. No Euler update/draining is called.
            op=h.flux;g=op.inputs;b=op.buffers
            nc,nf=np.int64(p.n),np.int64(p.nf)
            gravity=np.float64(p.parameters.gravity);dry=np.float64(p.parameters.dry)
            def launch(name,args):op.kernels[name](((p.nf+127)//128,),(128,),args)
            launch('resident_prepare_faces',(g['trace'],q,g['bed'],g['normal'],g['distance'],g['owner'],g['neighbour'],g['kind'],nc,nf,gravity,dry,b['local'],b['closed']))
            launch('augmented_fwave_local',(b['local'],nf,gravity,dry,b['riemann'],b['augmented_status']))
            launch('resident_assemble_flux',(g['trace'],b['local'],b['riemann'],q,g['bed'],g['normal'],
                g['owner'],g['neighbour'],g['kind'],g['boundary_mode'],g['sampled'],g['bedoffset'],g['hydro_constant'],
                b['closed'],nc,nf,gravity,dry,b['fl'],b['fr'],b['speed'],b['face_status']))
            h.boundary.apply_pressure(q,a.z,rec['trace'],b['fl'],b['fr'],b['face_status'])
            if exterior is not None:exterior.apply_gpu_flux(b['fl'],b['fr'],b['face_status'])
            stress=h.secondary.add_stress(g['trace'],g['kind'],b['closed'],b['fl'],b['fr'],b['speed'])['status']
            statuses=(rec['primitive_status'],rec['reconstruction_status'],h.boundary.status,
                h.boundary.face_status,b['augmented_status'],b['face_status'],secondary['status'],stress)
            if spatial_status_invalid(xp,statuses):
                raise BendTrialRejected('CUDA midpoint spatial residual rejected')
            g=h.flux.inputs;rate=xp.empty_like(q)
            self.gather(((p.n+127)//128,),(128,),
                (q,g['area'],g['length'],g['normal'],b['fl'],b['fr'],g['offset'],g['faces'],
                 g['signs'],g['active'],np.int64(p.n),np.int64(p.nf),np.float64(p.parameters.gravity),rate))
            water=b['fl'][0].copy()
        # The implicit residual must limit endpoint inventory, not the water
        # in its midpoint iterate. Correct BOTH sides' advective flux equally;
        # pressure/f-wave bed balance is unchanged. Never clip accepted depth.
        g=p.g;n=p.n;nf=p.nf
        self.positive.launch('midpoint_inventory',n,n,float(dt),self.interval_base,g['area'],g['length'],
            g['offset'],g['faces'],g['signs'],water,self.theta)
        boundary_budget=getattr(self,'explicit_boundary_budget',None)
        saved_boundary=None if boundary_budget is None else boundary_budget.reserve(self.interval_base,dt,water,self.theta)
        self.positive.launch('midpoint_faces',nf,nf,n,p.parameters.dry,q,self.theta,
            g['owner'],g['neighbour'],g['bid'] if boundary_budget is None else boundary_budget.limiter_bid,water,self.correction,self.status)
        if boundary_budget is not None:boundary_budget.restore(water,self.correction,saved_boundary)
        if bool(xp.any(self.status)):raise BendTrialRejected('Prescribed boundary would exceed available water')
        self.positive.launch('midpoint_gather',n,n,nf,g['area'],g['length'],g['offset'],g['faces'],g['signs'],water,self.correction,rate)
        self.minimum_theta=min(self.minimum_theta,float(xp.min(self.theta)))
        p.add_momentum(q,water,rate)
        if not bool(xp.all(xp.isfinite(rate))):raise BendTrialRejected('Nonfinite midpoint rate')
        return rate,water

    def _single(self,base,dt,time):
        # Stable boundary-root caching is scoped to ONE fixed residual time.
        # Friction surrounds every actual hydraulic substep, never the outer
        # morphology transaction. All trial arrays remain owned by rollback.
        exterior=getattr(self.solver,'exterior_bed',None)
        if exterior is not None:exterior.residual_token=object()
        try:
            work=base.copy()
            self.solver.physics.friction(work,.5*dt)
            result=self._solve_single(work,dt,time)
            self.solver.physics.friction(result[0],.5*dt)
            return result
        finally:
            if exterior is not None:exterior.residual_token=None

    def _solve_single(self,base,dt,time):
        from .river_midpoint_iteration import solve
        return solve(self,base,dt,time)

    def advance(self,base,dt,time):
        from .river_midpoint_accuracy import advance
        return advance(self,base,dt,time)


def install(solver):
    """Install before execution; physical wrapper transactions remain owners."""
    if solver.time or solver.steps or solver.physics._open:raise ValueError('Install midpoint before execution')
    controller=RiverMidpoint(solver);solver.midpoint=controller
    solver.identity['integrator']=METHOD
    solver.identity['river_midpoint']=controller.identity
    return solver
