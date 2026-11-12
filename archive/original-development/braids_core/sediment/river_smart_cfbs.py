"""Named, nonpublic Smart-CFBS reconstruction; no current Smart alias.

Juez2013 Phi column, signed geometric slope threshold, characteristic meanDm.
Source conflicts and the discontinuous zero-slope switch remain explicit.
The parabolic frequency is a frozen-coefficient monotonicity diagnostic, not
a proof of stability of the discontinuous nonlinear law or full SWE coupling.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from .bend_methods_v1 import digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD = 'capacity.smart_cfbs_juez2013_phi_thetaS_reconstruction_v1'
NAMES = ('cfbs_probe', 'cfbs_gradient', 'cfbs_limit', 'cfbs_faces', 'cfbs_frequency', 'cfbs_gather', 'cfbs_wave_probe')


@dataclass(frozen=True)
class Material:
    dm_mean_m: float
    d30_m: float
    d90_m: float
    solid_density_kg_m3: float = 2650.
    water_density_kg_m3: float = 1000.
    manning_n: float = .015
    repose_deg: float = 30.
    transport_depth_m: float = .001

    def wire(self):
        values = list(asdict(self).values())
        if any(isinstance(v, (bool, np.bool_)) for v in values) or not np.isfinite(values).all():
            raise ValueError('CFBS requires explicit finite physical material inputs')
        if (min(self.dm_mean_m, self.d30_m, self.d90_m, self.manning_n) <= 0
                or self.d30_m > self.d90_m
                or not self.solid_density_kg_m3 > self.water_density_kg_m3 > 0
                or not 0 < self.repose_deg < 90 or self.transport_depth_m < 0):
            raise ValueError('CFBS requires positive measured meanDm and D30 <= D90, positive Manning resistance/density contrast and valid repose/depth')
        return np.array([*values[:6], np.tan(np.deg2rad(self.repose_deg)), self.transport_depth_m])


class SmartCFBS:
    def __init__(self, backend, material, *, threads=2):
        if backend not in ('cpu', 'gpu'): raise ValueError('Choose CPU or GPU explicitly')
        self.material = material
        if backend == 'gpu':
            import cupy
            self.xp = cupy
        else: self.xp = np
        self.backend = backend; self.wire = self.xp.asarray(material.wire())
        self.engine = WaveEngine(backend, threads, source=Path(__file__).with_suffix('.h'), names=NAMES)
        self.solid_density = material.solid_density_kg_m3
        self.needs_characteristic_shear = self.uses_physical_drag = False
        self.needs_intrinsic_mixture = self.algebraic_suspension = False
        self.is_smart_jaggi = self.is_braids_mpm = False
        self.identity = dict(method=METHOD, material=asdict(material), runtime=self.engine.identity,
                             wrapper_sha256=digest(__file__), registered=False, public_support=False,
                             source_reading='Phi, slope-adjusted .047 threshold, signed geometric projected angle, explicit meanDm',
                             source_conflicts='2013 Sf typo; Phi/K1 threshold; characteristic-grain and angle interpretation',
                             derivatives='exact branchwise d(solid vector flux)/d(h,hu,hv,z_x,z_y), Manning n held fixed',
                             zero_slope='literal discontinuous CFBS branch; no smoothing or slope floor',
                             repose='explicit error when a flowing state exceeds repose; no negative-threshold extrapolation',
                             geometry='unique-neighbour inverse-distance affine LS; true-face nonorthogonal slope correction')

    def array(self, value, shape, kind=None):
        xp = self.xp
        if kind is None: kind = xp.float64
        if not isinstance(value, xp.ndarray) or value.shape != shape or value.dtype != kind or not value.flags.c_contiguous:
            raise ValueError('CFBS requires contiguous backend-resident typed inputs')

    def check(self, status):
        codes = np.unique(self.xp.asnumpy(status) if self.backend == 'gpu' else status)
        bad = codes[codes != 0]
        if len(bad):
            raise ValueError('CFBS rejected forcing, status '+str(bad.tolist())+
                             ' (1 nonfinite/negative depth,2 nonfinite law,3 super-repose needs conservative failure,4 face orientation,5 antidiffusive slope,6 nonhyperbolic first-order symbol,7 unresolved eigenvalue cluster)')

    def probe(self, q, gradient, *, check=True):
        n = q.shape[1]; xp = self.xp
        self.array(q, (3, n)); self.array(gradient, (2, n))
        field, jac, diagnostic, status = xp.zeros((6, n)), xp.empty((10, n)), xp.empty((5, n)), xp.empty(n, xp.int32)
        self.engine.launch('cfbs_probe', n, n, self.wire, q, gradient, field, jac, diagnostic, status)
        if check: self.check(status)
        return field, jac, diagnostic, status

    def gradient(self, geometry, z):
        n = len(z); xp = self.xp; self.array(z, (n,))
        out, rank = xp.empty((2, n)), xp.empty(n, xp.int32)
        self.engine.launch('cfbs_gradient', n, n, geometry['offset'], geometry['faces'], geometry['owner'],
                           geometry['neighbour'], geometry['xy'], z, out, rank)
        return out, rank

    def faces(self, geometry, q, z, bedbc, feed, *, packing=.78):
        n = q.shape[1]; nf = len(geometry['owner']); xp = self.xp
        self.array(q, (3, n)); self.array(z, (n,)); self.array(bedbc, (nf,), xp.int32); self.array(feed, (nf,))
        if not np.isfinite(packing) or not 0 < packing <= 1: raise ValueError('Physical packing required')
        gradient, rank = self.gradient(geometry, z)
        wet=q[0]>self.material.transport_depth_m
        denominator=xp.where(wet,q[0],1.)
        value=xp.vstack((q[0],xp.where(wet,q[1]/denominator,0.),xp.where(wet,q[2]/denominator,0.),z))
        tracegrad=xp.stack([self.gradient(geometry,xp.ascontiguousarray(v))[0] for v in value]).reshape(8,n)
        self.engine.launch('cfbs_limit',n,n,nf,value,geometry['offset'],geometry['faces'],geometry['owner'],
                           geometry['neighbour'],geometry['displacement'],tracegrad)
        flux, conductance = xp.zeros(nf), xp.zeros(nf)
        diagnostic, status, frequency = xp.empty((4, nf)), xp.empty(nf, xp.int32), xp.empty(n)
        self.engine.launch('cfbs_faces', nf, nf, n, self.wire, q, z, gradient, tracegrad, geometry['owner'], geometry['neighbour'],
                           bedbc, geometry['normal'], geometry['xy'], geometry['displacement'],feed, flux, conductance, diagnostic, status)
        self.check(status)
        self.engine.launch('cfbs_frequency', n, n, geometry['offset'], geometry['faces'], geometry['length'],
                           geometry['area'], conductance, float(packing), frequency)
        return dict(flux=flux, gradient=gradient, tracegrad=tracegrad,rank=rank, conductance=conductance,
                    face_diagnostic=diagnostic, frequency=frequency)

    def gather(self, geometry, flux, packing=.78):
        n = len(geometry['area']); xp = self.xp; out = xp.empty(n)
        self.engine.launch('cfbs_gather', n, n, geometry['offset'], geometry['faces'], geometry['signs'],
                           geometry['length'], geometry['area'], flux, float(packing), out)
        return out

    def waves(self, q, gradient, normal, packing=.78, *, check=True):
        n = q.shape[1]; xp = self.xp
        self.array(q, (3, n)); self.array(gradient, (2, n)); self.array(normal, (2, n))
        if not 0 < packing <= 1: raise ValueError('Physical packing required')
        out, status = xp.empty((6, n)), xp.empty(n, xp.int32)
        self.engine.launch('cfbs_wave_probe', n, n, self.wire, float(packing), q, gradient, normal, out, status)
        if check: self.check(status)
        return out, status


def validate_binding(roughness, physics, material, integrator):
    """Reject unsupported composition before replacing any producer."""
    from braids_core.river_model import HLLC_INTEGRATOR
    if integrator!=HLLC_INTEGRATOR:raise ValueError('CFBS coupled research is qualified with HLLC only')
    if (roughness['method']!='roughness.manning' or roughness.get('zones')
            or roughness['parameters']['manning_s_m_third']!=material.manning_n):
        raise ValueError('CFBS requires the same uniform Manning n in hydraulics and capacity; spatial resistance is not qualified')
    h=physics['hydrodynamics'];s=physics['sediment']
    if h['secondary_flow']!='secondary.off' or h['turbulence']!='off':
        raise ValueError('CFBS research coupling requires advanced hydrodynamics off')
    if s is None or s['transport']!='bedload':raise ValueError('CFBS research coupling requires bedload only')
    for key,expected in [('bed_slope','bed_slope.off'),('capacity_hiding','hiding.none'),('slope_hiding','hiding.none'),
                         ('suspension','off'),('banks','off'),('bedforms','off'),('avalanche','off')]:
        if s[key]!=expected:raise ValueError('CFBS owns its slope law; research coupling requires '+key+' off')


def install_research(solver, material):
    """Explicit fixed-MF1, single-bulk, clear-inflow research composition.

    Reuses hydraulic execution, finite-stock columns, budget and transaction
    owners. Replaces their transport producer with a named physical-gradient
    flux; it does not claim the old practical-Smart coupled qualification.
    Only clear inflow/export boundaries are permitted by this bounded adapter.
    """
    a=solver;p=a.physics;xp=a.xp
    if a.time or a.steps or p._open:raise ValueError('Install research CFBS before stepping')
    validate_binding(p.identity['river_roughness'],p.identity['resolved_river_model']['model']['physics'],material,a.identity['integrator'])
    if len(a.capacity)!=1 or p.morfac!=1. or a.methods.bed!='mixed' or a.methods.base_update!=1:
        raise ValueError('CFBS research coupling requires one mixed bulk and MF1')
    if a.methods.solid_density_kg_m3[0]!=material.solid_density_kg_m3 or a.methods.diameter_m[0]!=material.dm_mean_m:
        raise ValueError('CFBS material differs from the explicit finite-stock carrier')
    if a.capacity[0].bed_factor!=1. or p.sediment_depth!=material.transport_depth_m:
        raise ValueError('CFBS requires full bedload factor and matching hydraulic-owner sediment-depth activation')
    if not bool(xp.all(a.feed==0)) or bool(xp.any(p.W!=0)) or bool(xp.any(p.suspended_mass!=0)):
        raise ValueError('CFBS bounded research adapter requires clear feed, secondary flow and suspension off')
    op=SmartCFBS(a.backend,material,threads=p.flow.threads);a.capacity=[op]
    p.exterior_sediment=None
    # The bounded case admits no incoming sediment, regardless of which open
    # section admits water. Preserve export on reversed water at an inlet.
    bc=xp.where((p.g['neighbour']<0)&(p.g['bid']>=0),2,0).astype(xp.int32)
    feed=xp.zeros(p.nf)
    if hasattr(a,'_transport_boundary'):a._transport_boundary=lambda grain,water:None
    stats=dict(calls=0,max_frequency_s_inv=0.,last_frequency_s_inv=0.,maximum_trial_diffusion_number=0.,
               maximum_passing_diffusion_number=0.,diffusion_guard_rejections=0)
    def faces(q,z,hi):
        result=op.faces(p.g,q,z,bc,feed,packing=1.-p.parameters.porosity)
        frequency=float(xp.max(result['frequency']));number=p._dt*frequency*p.morfac
        stats.update(calls=stats['calls']+1,max_frequency_s_inv=max(stats['max_frequency_s_inv'],frequency),
                     last_frequency_s_inv=frequency,maximum_trial_diffusion_number=max(stats['maximum_trial_diffusion_number'],number))
        if number>.4*(1+1e-12):
            stats['diffusion_guard_rejections']+=1
            from .bend_methods_v1 import BendTrialRejected
            raise BendTrialRejected(f'CFBS bed-gradient diffusion number {number:g} exceeds0.4; retry same method')
        stats['maximum_passing_diffusion_number']=max(stats['maximum_passing_diffusion_number'],number)
        p.bedcell[:]=op.probe(q,result['gradient'])[0]
        p.aux[2]=result['flux']
        p.last.update(cfbs_diffusion_frequency_s_inv=frequency,cfbs_diffusion_number=number)
    a._single_faces=faces
    ident=dict(capacity=op.identity,method='research_cfbs_true_face_gradient_MUSCL_clear_stock_mf1_v1',
               physical_direction='source flow direction; no extra transverse slope law',
               face_law='fresh primitive MUSCL traces with covariant local-flow vector limiting, Roe velocity and mean physical depth, corrected centroid-normal bed slope',
               numerical_coupling='explicit hydraulic endpoint/bed splitting; physical gradient flux, no unrelated hydraulic-jump injection',
               timestep='frozen-coefficient normal diffusion number <=0.4 plus existing hydraulic CFL and5% bed-increment guards',
               failure='flowing super-repose forcing is an error; no silently enabled failure or threshold clipping',
               boundary='clear incoming sediment, export-only at each open face; closed wall flux exactly zero',
               public_support=False,experimental_validation=False)
    p.identity['smart_cfbs_research']=ident
    p.identity['river_mixtures']['capacity']=[op.identity]
    a.identity['smart_cfbs_research']=ident
    a.cfbs_operator=op;a.cfbs_statistics=stats
    return a
