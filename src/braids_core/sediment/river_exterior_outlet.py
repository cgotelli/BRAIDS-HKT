"""Characteristic critical outlet on the selected CPU/GPU exterior owner."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('exterior_critical_outlet',)


class CriticalOutlet:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        self.control=xp.asarray([controls[str(b)]['method']=='outlet.critical' for b in range(e.nb)],dtype=xp.int32)
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        # One section root is a small, synchronization-heavy task. Keep its
        # 128 logical lanes and their deterministic reductions, but avoid a
        # full OpenMP team and a barrier for every root iteration on the CPU.
        # Other CPU kernels retain their separately selected thread count.
        if e.a.backend=='cpu':self.engine.threads=1
        self.branch=xp.zeros(p.nf,dtype=xp.int32);self.invariant=xp.zeros(p.nf)
        self.stage=xp.zeros(e.nb);self.metrics=xp.zeros((3,e.nb));self.status=xp.zeros(e.nb,dtype=xp.int32)
        from .river_boundary_components import connected_opening_chains
        self.chains={b:connected_opening_chains(e.a.data,b) for b in range(e.nb)
                     if controls[str(b)]['method']=='outlet.critical'}
        self.identity=dict(method='outlet.critical_characteristic_active_set',source_sha256=digest(__file__),
            runtime=self.engine.identity,geometry='actual exterior face depth, area and wet width',
            root='outgoing characteristic discharge equals sqrt(g*A^3/B) on the controlled opening',
            regimes='close local inflow; extrapolate supercritical outflow; recompute after removing locally inward or dry controlled faces',
            shallow_face='bound downward face-bed extrapolation by cell-average depth below 0.1 mm; smoothly restore actual face geometry by 1 mm',
            unresolved='close the controlled opening when its critical root lies below resolved dry depth',
            tangent='interior tangential velocity retained; no prescribed incoming state',
            parallel='resident 128-lane deterministic reductions; one CPU thread or a CUDA block',
            state='derived scratch only; no Python root, backend fallback or physical inventory')
        self.identity['dry_separation']='independent critical roots on wet face chains separated by dry physical owners; common section otherwise'

    def evaluate(self,q,z):
        e=self.e;p=e.p;g=p.g
        self.engine.launch('exterior_critical_outlet',128*e.nb,p.nf,p.n,e.nb,p.parameters.gravity,p.parameters.dry,
            float(e.a.data.get('speed_limit',50.)),self.control,e.offset,e.faces,g['owner'],q,z,p.exterior_bed,
            g['normal'],g['length'],self.branch,self.invariant,self.stage,self.metrics,e.trace,e.flux,e.status,self.status)
        # A single opening may cross a dry bar. Each disconnected wet branch
        # has its own critical control; an aggregated Fr=1 condition does not
        # impose critical flow in either branch when their states differ.
        # Only gather boundary-face scalars, leaving the section root on the
        # selected native backend. The common all-wet path stays unchanged.
        from .bed_columns_v101 import host
        for b,chains in self.chains.items():
            flat=[f for chain in chains for f in chain]
            fi=e.xp.asarray(flat,dtype=e.xp.int32);owner=g['owner'][fi]
            sample=host(e.xp.stack((q[0,owner],z[owner],p.exterior_bed[fi])))
            h=sample[0];geom=h+sample[1]-sample[2]
            wet=(h>p.parameters.dry)&(geom>p.parameters.dry);at=0;groups=[]
            for chain in chains:
                current=[]
                for f in chain:
                    if wet[at]:current.append(f)
                    elif current:groups.append(current);current=[]
                    at+=1
                if current:groups.append(current)
            if len(groups)<=1:continue
            offset=e.xp.asarray(np.cumsum([0]+[len(x) for x in groups]),dtype=e.xp.int32)
            faces=e.xp.asarray([f for group in groups for f in group],dtype=e.xp.int32)
            count=len(groups);control=e.xp.ones(count,dtype=e.xp.int32)
            stage=e.xp.zeros(count);metrics=e.xp.zeros((3,count));status=e.xp.zeros(count,dtype=e.xp.int32)
            self.engine.launch('exterior_critical_outlet',128*count,p.nf,p.n,count,p.parameters.gravity,p.parameters.dry,
                float(e.a.data.get('speed_limit',50.)),control,offset,faces,g['owner'],q,z,p.exterior_bed,
                g['normal'],g['length'],self.branch,self.invariant,stage,metrics,e.trace,e.flux,e.status,status)
            dry_ids=e.xp.asarray([f for f,is_wet in zip(flat,wet) if not is_wet],dtype=e.xp.int32)
            if len(dry_ids):
                dry_owner=g['owner'][dry_ids];height=e.xp.maximum(q[0,dry_owner],0.)
                pressure=.5*p.parameters.gravity*height*height
                e.trace[:,dry_ids]=0.;e.flux[0,dry_ids]=0.
                e.flux[1,dry_ids]=pressure*g['normal'][0,dry_ids]
                e.flux[2,dry_ids]=pressure*g['normal'][1,dry_ids]
                e.status[dry_ids]=0
            self.status[b]=e.xp.max(status)
            self.metrics[:,b]=e.xp.sum(metrics,axis=1)
            self.stage[b]=e.xp.nan  # No single stage represents separate channels.
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
