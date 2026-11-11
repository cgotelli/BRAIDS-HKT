"""Shared CPU/GPU SSPRK(2,2) augmented-f-wave hydraulic integration.

Reuses the existing spatial residual and conservative donor limiter. Friction
is Strang-split at each hydraulic substep; sediment/AMR retain their owner clock.
No implicit method, backend substitution or artificial cut-cell area floor.
"""
import ctypes
from pathlib import Path
import numpy as np
from .river_midpoint import RiverMidpoint
from .bend_methods_v1 import BendTrialRejected, digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD = 'explicit_fwave_ssprk22_river_v1'
CFL_SAFETY = .98
MAX_CFL = 1.


class RiverExplicit(RiverMidpoint):
    # The inherited constructor/rate provide spatial buffers, not a time step.
    # Neither the midpoint nonlinear solve nor its temporal estimator is called.
    def __init__(self, solver, cfl=.5):
        super().__init__(solver)
        if isinstance(cfl, bool) or not np.isfinite(cfl) or not 0 < cfl <= MAX_CFL:
            raise ValueError('Explicit hydraulic face-sum CFL must be in (0, 1]')
        self.cfl = float(cfl); xp = self.xp; p = solver.physics
        self.cfl_engine = WaveEngine(solver.backend, p.flow.threads,
            source=Path(__file__).with_suffix('.h'), names=('explicit_frequency',))
        self.frequency = xp.zeros(p.n)
        self.zero_mask = xp.zeros(p.nf, dtype=xp.int32)
        self.zero_trace = xp.zeros((3, p.nf)); self.speed = xp.zeros(p.nf)
        self._speed_library = None
        from .river_secondary_clock import SecondaryClock
        self.secondary = SecondaryClock(solver, self.cfl)
        p._secondary_clock = self.secondary
        exterior=getattr(solver,'exterior_bed',None)
        self.explicit_boundary_budget=None
        if exterior is not None:
            from .river_explicit_boundary_budget import ExplicitBoundaryBudget
            self.explicit_boundary_budget=ExplicitBoundaryBudget(exterior)
        self.identity = dict(method=METHOD, controller_sha256=digest(__file__), cfl=self.cfl,
            cfl_kernel=self.cfl_engine.identity,
            shared_spatial_source=digest(Path(__file__).with_name('river_midpoint.py')),
            flux_limiter=self.positive.identity,
            friction='Strang drag around each hydraulic substep; trial-local',
            boundary_cache='distinct token for each SSP stage time',
            cfl_definition='dt * max_cell(sum_face(length * wave_speed) / actual_area)',
            clock_policy=dict(predictor_safety=CFL_SAFETY, maximum_cfl=MAX_CFL,
                accepted_growth_limit=2., rejected_growth_limit=.5,
                endpoint_remainder='retain unclipped hint, bounded by current stage waves',
                regrid='carry accepted scalar hint only under identical explicit controller identity; recheck every new-mesh stage'),
            explicit_fallback=False)
        self.identity['secondary_transport'] = self.secondary.identity
        self.identity['boundary_budget']=None if self.explicit_boundary_budget is None else self.explicit_boundary_budget.identity

    def bind_boundary_budget(self, exterior):
        # Public exterior owners are installed after the explicit controller.
        from .river_explicit_boundary_budget import ExplicitBoundaryBudget
        if self.explicit_boundary_budget is not None:raise ValueError('Explicit boundary budget already bound')
        self.explicit_boundary_budget=ExplicitBoundaryBudget(exterior)
        self.identity['boundary_budget']=self.explicit_boundary_budget.identity

    def rate(self, q, dt, time):
        a = self.solver; p = a.physics; g = p.g; xp = self.xp
        exterior = getattr(a, 'exterior_bed', None)
        if exterior is not None: exterior.residual_token = object()
        # Native speeds are module-owned: keep their export in the residual lock.
        from . import bend_native_hydraulics as native
        if a.backend == 'cpu':
            with native._lock:
                rate, water = super().rate(q, dt, time)
                # Boundary installation may replace the native library after
                # controller construction. Bind the actual current instance.
                if self._speed_library is not a.hydraulics.lib:
                    self._speed_library = a.hydraulics.lib
                    self._get_speed = self._speed_library.river_get_face_speed
                    self._get_speed.argtypes = [ctypes.c_void_p]
                    self._get_speed.restype = None
                self._get_speed(self.speed.ctypes.data)
        else:
            rate, water = super().rate(q, dt, time)
            self.speed = a.hydraulics.flux.buffers['speed']
        self.cfl_engine.launch('explicit_frequency', p.n, p.n, p.nf,
            p.parameters.gravity, p.parameters.dry, q, self.speed,
            exterior.trace if exterior else self.zero_trace,
            exterior.selected if exterior else self.zero_mask,
            g['owner'], g['normal'], g['area'], g['length'], g['offset'], g['faces'], self.frequency)
        frequency = float(xp.max(self.frequency))
        self.frequency_peak = max(self.frequency_peak, frequency)
        if not np.isfinite(frequency) or dt*frequency > self.cfl*(1+1e-12):
            raise BendTrialRejected(f'Explicit stage CFL {dt*frequency:g} exceeds {self.cfl:g}')
        return rate, water

    def _single(self, base, dt, time):
        p = self.solver.physics; xp = self.xp
        exterior = getattr(self.solver, 'exterior_bed', None)
        work = base.copy()
        self.minimum_theta = 1.; self.frequency_peak = 0.
        try:
            p.friction(work, .5*dt)
            if self.secondary.active:
                p.refresh_secondary_momentum(work,p.secondary_pending)
            self.interval_base = work
            r, w = self.rate(work, dt, time); q1 = work+dt*r
            if bool(xp.any(q1[0] < 0)): raise BendTrialRejected('Explicit first stage negative')
            q1[1:, q1[0] <= p.parameters.dry] = 0.
            if self.secondary.active:
                predictor=self.secondary.trial(work,q1,q1,w,w,dt)
                p.refresh_secondary_momentum(q1,predictor[0])
            self.interval_base = q1
            r2, w2 = self.rate(q1, dt, time+dt); q2 = .5*work+.5*(q1+dt*r2)
            if not bool(xp.all(xp.isfinite(q2))) or bool(xp.any(q2[0] < 0)):
                raise BendTrialRejected('Explicit endpoint invalid')
            q2[1:, q2[0] <= p.parameters.dry] = 0.
            p.friction(q2, .5*dt)
            secondary = self.secondary.trial(work, q1, q2, w, w2, dt)
            return q2, .5*(w+w2), self.frequency_peak, secondary
        finally:
            if exterior is not None: exterior.residual_token = None

    def advance(self, base, dt, time):
        q = base; water = None; used = []; rejected = []; freqs = []
        ev = self.evaluations; elapsed = 0.; floor = dt*2**-20
        hint = dt if self.step_hint is None else self.step_hint
        remainders = 0
        self.secondary.begin()
        # A forcing/activation clock can leave a positive sub-roundoff tail.
        # It still needs one conservative update, never a None flux return.
        while not used or elapsed < dt-1e-13*max(1., dt):
            trial = min(hint, dt-elapsed)
            truncated = trial < hint*(1-1e-12)
            try: nextq, w, frequency, secondary = self._single(q, trial, time+elapsed)
            except BendTrialRejected as exc:
                rejected.append(dict(time=time+elapsed, dt=trial, reason=str(exc),
                    kind='secondary_transport' if str(exc).startswith('Secondary transport') else 'cfl' if str(exc).startswith('Explicit stage CFL') else 'spatial_or_positivity'))
                hint = min(.5*trial, CFL_SAFETY*self.cfl/self.frequency_peak) if self.frequency_peak > 0 else .5*trial
                if hint < floor or len(rejected) > 100: raise
                continue
            self.secondary.accept(secondary,trial)
            q = nextq; water = trial*w if water is None else water+trial*w
            elapsed += trial; used.append(trial); freqs.append(trial*frequency)
            # Output/AMR/outer-clock clipping is not a new stability limit.
            # Retain (but never grow from) the unclipped hint, still bounded
            # by the actual current stage waves. Rejections always reduce it.
            growth = hint if truncated else 2*trial
            hint = min(growth, CFL_SAFETY*self.cfl/frequency) if frequency > 0 else growth
            remainders += int(truncated)
            if len(used) > 4096: raise BendTrialRejected('Explicit substep limit')
        self.step_hint = float(hint); self.easy_steps = 0
        return q, water/dt, dict(integrator=METHOD, hydraulic_substeps=len(used),
            hydraulic_minimum_dt=min(used), hydraulic_maximum_dt=max(used),
            hydraulic_endpoint_remainders=remainders, hydraulic_rejections=rejected, hydraulic_preempted_retries=0,
            fwave_evaluations=self.evaluations-ev, linear_matvecs=0, nonlinear_iterations=0,
            accepted_cfl_max=max(freqs), hydraulic_next_substep_hint_s=self.step_hint, explicit_fallback=False)


def install(solver, cfl=.5):
    if solver.time or solver.steps or solver.physics._open:
        raise ValueError('Install explicit controller on a fresh owner')
    solver.midpoint = RiverExplicit(solver, cfl)  # Private transaction slot, not a method choice.
    solver.identity['integrator'] = METHOD
    solver.identity.pop('river_midpoint', None)
    solver.identity['river_explicit'] = solver.midpoint.identity
    return solver
