"""Characteristic critical outlet on the selected CPU/GPU exterior owner."""
from pathlib import Path
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('exterior_critical_outlet',)


class CriticalOutlet:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        self.control=xp.asarray([controls[str(b)]['method']=='outlet.critical' for b in range(e.nb)],dtype=xp.int32)
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.branch=xp.zeros(p.nf,dtype=xp.int32);self.invariant=xp.zeros(p.nf)
        self.stage=xp.zeros(e.nb);self.metrics=xp.zeros((3,e.nb));self.status=xp.zeros(e.nb,dtype=xp.int32)
        self.identity=dict(method='outlet.critical_characteristic_active_set',source_sha256=digest(__file__),
            runtime=self.engine.identity,geometry='actual exterior face depth, area and wet width',
            root='outgoing characteristic discharge equals sqrt(g*A^3/B) on the controlled opening',
            regimes='close local inflow; extrapolate supercritical outflow; recompute after removing locally inward or dry controlled faces',
            unresolved='close the controlled opening when its critical root lies below resolved dry depth',
            tangent='interior tangential velocity retained; no prescribed incoming state',
            parallel='resident 128-lane section reductions; persistent OpenMP team or CUDA block',
            state='derived scratch only; no Python root, backend fallback or physical inventory')

    def evaluate(self,q,z):
        e=self.e;p=e.p;g=p.g
        self.engine.launch('exterior_critical_outlet',128*e.nb,p.nf,p.n,e.nb,p.parameters.gravity,p.parameters.dry,
            float(e.a.data.get('speed_limit',50.)),self.control,e.offset,e.faces,g['owner'],q,z,p.exterior_bed,
            g['normal'],g['length'],self.branch,self.invariant,self.stage,self.metrics,e.trace,e.flux,e.status,self.status)
        if bool(e.xp.any(self.status)):raise BendTrialRejected('Critical outlet has no converged characteristic section root')


class FreeWeir:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        self.control=xp.asarray([controls[str(b)]['method']=='outlet.free_weir' for b in range(e.nb)],dtype=xp.int32)
        keys=('crest_m','coefficient_m_half_s','effective_width_m')
        self.parameters=xp.asarray([[controls[str(b)].get(k,0.) for b in range(e.nb)] for k in keys],dtype=xp.float64)
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=('exterior_free_weir',))
        self.branch=xp.zeros(p.nf,dtype=xp.int32);self.invariant=xp.zeros(p.nf)
        self.stage=xp.zeros(e.nb);self.metrics=xp.zeros((3,e.nb));self.status=xp.zeros(e.nb,dtype=xp.int32)
        self.identity=dict(method='outlet.free_weir_lax_active_outward',source_sha256=digest(__file__),runtime=self.engine.identity,
            controls={b:row.copy() for b,row in controls.items() if row['method']=='outlet.free_weir'},
            root='outgoing shock/rarefaction Lax curve, nonnegative local flux, common subcritical stage; sum Q equals coefficient * effective_width * max(stage-crest,0)^1.5',
            coefficient_units='m^0.5/s; includes the gravity/discharge-coefficient factors, not a dimensionless Cd',
            regimes='free undrowned weir; interior recirculation allowed; upstream-going hydraulic jumps allowed; reject no subcritical root or downstream-only control shock',
            closure='dry section or locally incoming solved trace closes without importing water; momentum pressure retained',
            geometry='actual exterior face bed/depth; section width is not substituted for the explicit effective weir width',
            parallel='resident 128-lane section reductions; persistent OpenMP team or CUDA block',
            state='derived scratch; fixed authenticated parameters, no host root or backend fallback')

    def evaluate(self,q,z):
        e=self.e;p=e.p;g=p.g
        self.engine.launch('exterior_free_weir',128*e.nb,p.nf,p.n,e.nb,p.parameters.gravity,p.parameters.dry,
            float(e.a.data.get('speed_limit',50.)),self.control,self.parameters,e.offset,e.faces,g['owner'],q,z,p.exterior_bed,
            g['normal'],g['length'],self.branch,self.invariant,self.stage,self.metrics,e.trace,e.flux,e.status,self.status)
        if bool(e.xp.any(self.status)):raise BendTrialRejected('Free-weir outlet has no admissible subcritical control: check crest, width and coefficient, or use free outflow for an uncontrolled supercritical outlet')
