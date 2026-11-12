"""Batched class transport assembly without changing physical fluxes or budgets."""
from pathlib import Path
from .bend_methods_v1 import digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('class_donor','class_scale','class_divergence','class_ledger')

class ClassBedUpdate:
    def __init__(self,physics,classes,solid_density):
        self.p=physics;self.ng=int(classes)
        xp=physics.xp
        self.density=xp.ascontiguousarray(solid_density,dtype=xp.float64)
        if (self.density.shape!=(self.ng,) or not bool(xp.all(xp.isfinite(self.density)))
                or bool(xp.any(self.density<=0))):
            raise ValueError('Explicit positive per-class solid densities required')
        self.engine=WaveEngine(physics.engine.backend,physics.flow.threads,
            source=Path(__file__).with_suffix('.h'),names=NAMES)
        # Private scratch, not accepted state. Every entry is overwritten at
        # every trial; a remesh owns a new shape/topology and a new workspace.
        self.factors=physics.xp.empty((self.ng,physics.n))
        self.identity=dict(runtime=self.engine.identity,wrapper_sha256=digest(__file__),
            solid_density_kg_m3=(self.density.get() if hasattr(self.density,'get') else self.density).tolist(),
            method='conservative_class_batched_donor_scale_divergence_roundoff_reserve',
            depletion='face donor roundoff reserve; retain inventory below sqrt(DBL_MIN) kg/m2; no cell stock clipping',
            budgets='unchanged global class and solid-volume reductions')

    def apply(self,available,dt):
        p=self.p;g=p.g;e=self.engine;n=p.n;nf=p.nf;ng=self.ng
        e.launch('class_donor',ng*n,n,nf,ng,float(dt),self.density,available,p.class_flux,
            g['length'],g['area'],g['offset'],g['faces'],g['signs'],self.factors,p.factors)
        e.launch('class_scale',ng*nf,n,nf,ng,float(dt),g['owner'],g['neighbour'],
            self.factors,p.class_flux,p.class_face_integral,p.aux)
        e.launch('class_divergence',ng*n,n,nf,ng,float(dt),self.density,p.class_flux,
            g['length'],g['area'],g['offset'],g['faces'],g['signs'],p.class_delta,p.rate)
        return float(p.xp.min(self.factors))

    def commit_ledger(self):
        p=self.p;g=p.g
        self.engine.launch('class_ledger',self.ng*p.nf,p.nf,self.ng,g['neighbour'],
            g['bid'],g['length'],p.class_face_integral,p.class_boundary_ledger)
