"""Exact-Q boundary Riemann solve; the requested water line is a reference."""
from pathlib import Path
from .bend_methods_v1 import BendTrialRejected,digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


class CompatibleInlet:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        methods=('inlet.stage_discharge','inlet.depth_discharge','inlet.critical_discharge','inlet.normal_discharge')
        self.control=xp.asarray([controls[str(b)]['method'] in methods for b in range(e.nb)],dtype=xp.int32)
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=('exterior_compatible_inlet',))
        self.branch=xp.zeros(p.nf,dtype=xp.int32);self.invariant=xp.zeros(p.nf)
        self.stage=xp.zeros(e.nb);self.target_stage=xp.zeros(e.nb)
        self.exterior_velocity=xp.zeros(e.nb);self.iterations=xp.zeros((2,e.nb),dtype=xp.int32)
        self.mass_balance=xp.zeros((3,e.nb))
        self.metrics=xp.zeros((4,e.nb));self.status=xp.zeros(e.nb,dtype=xp.int32)
        self.identity=dict(method='inlet.exact_discharge_boundary_riemann_reference_stage.v3',
            source_sha256=digest(__file__),runtime=self.engine.identity,
            target='selected stage/depth/critical/normal water line is the common external reference, not a strong actual trace elevation',
            compatible='sample exact one-dimensional SWE shock/rarefaction/dry Riemann fluxes; solve common virtual external normal velocity for exact net section Q',
            virtual_velocity='may be negative; it is a boundary datum, not the accepted physical inlet velocity; no velocity cap on this scalar',
            signed_flow='physical local backflow is retained and reported; the imposed positive net section discharge is exact',
            root_budget='relative net-Q tolerance plus explicit double-precision gross-throughflow roundoff budget; no minimum discharge; mass_balance records actual-minus-requested Q, gross flux and acceptance bound',
            closed='zero requested Q closes each face with the exact reflected-owner Riemann pressure; no film or interior overwrite',
            tangent='zero incoming tangent; local outgoing flow advects the adjacent interior tangent',
            diagnostics='target_stage common reference; stage area-weighted actual trace elevation, not a common water line; exterior_velocity; iterations outer/root and maximum inner Riemann; branch 0 closed/dry, 1 subcritical incoming, 2 supercritical incoming, 3 sonic, 4 local outgoing',
            critical_normal_scope='section critical Froude/normal resistance describe the reference, not a second imposed actual wet backwater condition',
            parallel='resident 128-lane section reduction; nested safeguarded Newton with analytical derivatives on OpenMP/CUDA',
            state='derived scratch; no physical state overwrites, film, velocity clip or backend fallback')

    def evaluate(self,q,z):
        e=self.e;p=e.p;g=p.g
        self.engine.launch('exterior_compatible_inlet',128*e.nb,p.nf,p.n,e.nb,p.parameters.gravity,p.parameters.dry,
            float(e.a.data.get('speed_limit',50.)),self.control,e.offset,e.faces,g['owner'],q,z,p.exterior_bed,
            g['normal'],g['length'],e.forcing,self.branch,self.invariant,self.stage,self.target_stage,self.metrics,
            self.exterior_velocity,self.iterations,self.mass_balance,e.trace,e.flux,e.status,self.status)
        if bool(e.xp.any(self.status)):raise BendTrialRejected('Prescribed inlet has no converged boundary-Riemann section discharge root')
