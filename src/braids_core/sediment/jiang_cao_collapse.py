"""Named finite-volume repose projection for JC research, not mechanical failure.

Bed-to-bed solid and pore-water transfer; h,hu,hv,mobile remain unchanged.
Actual areas and centroids; shared-face downhill transfers; no wall/open export.
Both pair secants and the capacity owner's full LS gradient must pass repose.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from .jiang_cao_roughness import JiangCaoRoughness
from .jiang_cao_exchange import ExchangeMaterial
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='redistribution.repose_jc2026_cartesian200_v1'
NAMES=('jc_repose_propose','jc_repose_donors','jc_repose_scale','jc_repose_gather')


@dataclass(frozen=True)
class ReposeMaterial:
    """Geometric closure inputs, independent of any transport-capacity law."""
    porosity: float
    repose_angle_rad: float

    def __post_init__(self):
        if not np.isfinite([self.porosity,self.repose_angle_rad]).all() or not 0<=self.porosity<1 or not 0<self.repose_angle_rad<np.pi/2:
            raise ValueError('Require porosity in [0,1) and repose angle in (0,90) degrees')


@dataclass(frozen=True)
class CollapseResult:
    state: object
    solid_face_volume: object
    iterations: int
    max_slope: float
    gross_solid_transfer_m3: float
    solid_residual_m3: float
    stock_constrained: bool = False


class ReposeCollapse:
    def __init__(self,mesh,material,backend,*,threads=4,max_iterations=20000,require_repose=True):
        if not isinstance(material,(ExchangeMaterial,ReposeMaterial)):raise TypeError('Explicit exchange or geometric repose material required')
        if type(require_repose) is not bool:raise TypeError('Explicit repose constraint policy required')
        self.require_repose=require_repose
        if type(max_iterations) is not int or max_iterations<1:raise ValueError('Positive integer collapse iteration limit required')
        self.mesh,self.xp,self.material=mesh,mesh.xp,material
        self.n,self.nf=mesh.n,mesh.nf;self.B=1-material.porosity
        angle=material.capacity_material.repose_angle_rad if isinstance(material,ExchangeMaterial) else material.repose_angle_rad
        self.slope=float(np.tan(angle))
        # Capacity rejects equality. Tiny explicit numerical margin, not angle calibration.
        self.target_slope=self.slope*(1.-1e-8);self.tolerance=self.slope*1e-10
        self.max_iterations=max_iterations
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        xp=self.xp;g=mesh.g;o,j=g['owner'],g['neighbour'];inside=j>=0
        degree=xp.bincount(xp.concatenate((o[inside],j[inside])),minlength=self.n)
        # Sum of Jacobi relaxation weights <= .45 at every cell. Each update
        # is a convex downhill averaging operation before donor stock limiting.
        weight=xp.zeros(self.nf);distance=xp.ones(self.nf)
        weight[inside]=.45/xp.maximum(degree[o[inside]],degree[j[inside]])
        distance[inside]=xp.sqrt(((g['xy'][:,j[inside]]-g['xy'][:,o[inside]])**2).sum(axis=0))
        if bool(xp.any(distance<=0)) or not bool(xp.all(xp.isfinite(distance))):raise ValueError('Positive finite centroid distances required')
        self.weight,self.distance=weight,distance
        self.identity=dict(method=METHOD,public_support=False,runtime=self.engine.identity,wrapper_sha256=digest(__file__),
            geometry_sha256=mesh.geometry_sha256,repose_angle_rad=angle,
            slope=self.slope,target_relative_margin=1e-8,relative_slope_tolerance=1e-10,max_iterations=max_iterations,
            transfer='shared-face downhill solid volume; harmonic pair area; parallel bounded Jacobi proposals; one donor limiter',
            criterion='actual centroid secants AND selected full vector LS gradient; no coordinate-axis repose test',
            water='h,hu,hv,mobile unchanged; saturated bed carries pore water p/(1-p) times solid transfer',
            boundary='no exterior transfer, including inlet/outlet; only internal physical faces',
            source_scope='named numerical realization of local repose rule; not author code, kinetic collapse or cohesive failure')
        if not require_repose:
            self.identity.update(constraint_policy='finite-stock feasible downhill transfers; immobile residual slope reported, never clipped',
                underflow_policy='retain donor stock when stock or area-integrated solid volume is below float64 tiny/epsilon; no mass deletion',
                water='geometric bed-only projection; caller owns physical water and mobile inventory')

    def apply(self,state,*,check=lambda:None):
        xp=self.xp;n=self.n;nf=self.nf;g=self.mesh.g
        s=xp.ascontiguousarray(xp.asarray(state,dtype=xp.float64))
        if s.shape!=(6,n) or not bool(xp.all(xp.isfinite(s))) or bool(xp.any(s[4]<0)):
            raise ValueError('Finite six-field state with nonnegative bed stock required')
        if bool(xp.any(s[0]<0)|xp.any(s[3]<0)|xp.any(s[3]>self.B*s[0])):
            raise ValueError('Invalid water/mobile inventory')
        work=s.copy();ledger=xp.zeros(nf);F=xp.empty(nf);excess=xp.empty(nf);scale=xp.empty(n)
        gross=0.;area=g['area'];initial=float(xp.sum(s[4]*area))
        for iteration in range(self.max_iterations+1):
            check();grad=self.mesh.gradient(work[5])
            self.engine.launch('jc_repose_propose',nf,n,nf,self.B,self.target_slope,g['owner'],g['neighbour'],
                area,self.distance,self.weight,work[5],grad,F,excess)
            full=float(xp.max(xp.hypot(grad[0],grad[1])))
            residual=float(xp.max(excess))
            if full<=self.target_slope+self.tolerance and residual<=self.tolerance:
                closure=float(xp.sum(work[4]*area))-initial
                if abs(closure)>1e-11*max(1.,abs(initial)) or bool(xp.any(work[4]<0)):
                    raise BendTrialRejected('Repose stock or solid budget failed')
                inside=g['neighbour']>=0
                pair=float(xp.max(xp.abs(work[5,g['owner'][inside]]-work[5,g['neighbour'][inside]])/self.distance[inside])) if bool(xp.any(inside)) else 0.
                return CollapseResult(work,ledger,iteration,max(full,pair),gross,closure)
            if iteration==self.max_iterations:break
            self.engine.launch('jc_repose_donors',n,n,g['offset'],g['faces'],g['owner'],area,work[4],F,scale)
            if not self.require_repose:
                # At subnormal magnitudes a relative donor safety margin loses
                # its representable bits. Retain this stock, rather than let
                # rounded face transfers overdraw it or clip a negative bed.
                floor=np.finfo(float).tiny/np.finfo(float).eps
                scale[(work[4]<=floor)|(work[4]*area<=floor)]=0.
            self.engine.launch('jc_repose_scale',nf,nf,g['owner'],g['neighbour'],scale,F,ledger)
            moved=float(xp.sum(xp.abs(F)))
            if moved==0.:
                if self.require_repose:
                    raise BendTrialRejected('Repose cannot be satisfied: immobile substrate/stock or disconnected gradient stencil; no clipping applied')
                closure=float(xp.sum(work[4]*area))-initial
                if abs(closure)>1e-11*max(1.,abs(initial)) or bool(xp.any(work[4]<0)):
                    raise BendTrialRejected('Stock-constrained repose budget failed')
                return CollapseResult(work,ledger,iteration,max(full,self.target_slope+residual),gross,closure,True)
            gross+=moved
            self.engine.launch('jc_repose_gather',n,n,self.B,g['offset'],g['faces'],g['owner'],area,F,work[4],work[5])
            if not bool(xp.all(xp.isfinite(work))) or bool(xp.any(work[4]<0)):
                raise BendTrialRejected('Repose produced invalid stock; transaction rejected')
        raise BendTrialRejected(f'Repose did not converge in {self.max_iterations} sweeps; residual {residual:g}; inspect finite bed stock and geometry')


class JiangCaoCollapse(JiangCaoRoughness):
    def __init__(self,*args,collapse_enabled=True,**kwargs):
        if type(collapse_enabled) is not bool:raise ValueError('Explicit boolean collapse switch required')
        super().__init__(*args,**kwargs)
        self.collapse_enabled=collapse_enabled
        self.collapse=ReposeCollapse(self.mesh,self.exchange.material,self.backend,threads=self.a.physics.flow.threads)
        self.collapse_integral=self.xp.zeros(self.nf);self.collapse_gross=0.
        self._pending=self.xp.zeros(self.nf);self._gross=0.;self._iterations=0;self._virtual_iterations=0
        self.identity=dict(self.identity,method='coupled.jc2026_p15_repose200_v1',collapse=self.collapse.identity,
            collapse_enabled=collapse_enabled,limits='fixed fully wet alluvial bed; one grain MF1; mild curvature; no wetdry/restart/AMR/public admission',
            collapse_splitting='project before/after exchange; virtual midpoint projection is not an accepted physical transfer')

    def _relax(self,s,*,physical):
        r=self.collapse.apply(s,check=self.a.check)
        if physical:
            self._pending+=r.solid_face_volume;self._gross+=r.gross_solid_transfer_m3;self._iterations+=r.iterations
        else:self._virtual_iterations+=r.iterations
        return r.state

    def source(self,s,dt):
        if not self.collapse_enabled:return super().source(s,dt)
        base=self._relax(s,physical=True)
        ce,_=self.target(base)
        predictor=self.exchange.exchange(base,ce,.5*dt).state
        predictor=self._relax(predictor,physical=False)
        midce,_=self.target(predictor)
        end=self.exchange.exchange(base,midce,dt).state
        return self._relax(end,physical=True)

    def step(self,dt,*,inlet_concentration=0.):
        self._pending.fill(0.);self._gross=0.;self._iterations=0;self._virtual_iterations=0
        try:result=super().step(dt,inlet_concentration=inlet_concentration)
        except BaseException:
            self._pending.fill(0.);self._gross=0.;self._iterations=0;self._virtual_iterations=0
            raise
        self.collapse_integral+=self._pending;self.collapse_gross+=self._gross
        self.last.update(collapse_sweeps=self._iterations,virtual_collapse_sweeps=self._virtual_iterations,
            collapse_gross_solid_m3=self._gross,cumulative_collapse_gross_solid_m3=self.collapse_gross)
        return dict(self.last)
