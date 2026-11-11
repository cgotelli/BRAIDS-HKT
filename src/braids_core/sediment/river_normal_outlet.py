"""Manual-slope normal-depth outlet on connected physical face chains."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from .river_boundary_components import connected_opening_chains
from .river_roughness import selected
try:  # Packaged BRAIDS keeps the native loader in braids_support.
    from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine
except ModuleNotFoundError:  # Development checkout retains validation sources.
    from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


class NormalOutlet:
    def __init__(self,exterior,controls):
        e=exterior;self.e=e;xp=e.xp;p=e.p
        if e.bedforms_enabled:
            raise ValueError('Normal-depth outlet with dynamic bedform roughness is not yet supported; select a different outlet or disable bedforms')
        chains=[];slopes=[];boundaries=[]
        for b in range(e.nb):
            row=controls[str(b)]
            if row['method']!='outlet.normal_depth':continue
            for chain in connected_opening_chains(e.a.data,b):
                chains.append(chain);slopes.append(row['bed_slope']);boundaries.append(b)
        if not chains:raise ValueError('Normal-depth outlet has no connected physical section')
        self.offset=xp.asarray(np.cumsum([0]+[len(c) for c in chains]),dtype=xp.int32)
        self.faces=xp.asarray([i for c in chains for i in c],dtype=xp.int32)
        self.slopes=xp.asarray(slopes,dtype=xp.float64);self.count=len(chains)
        self.stage=xp.zeros(self.count);self.metrics=xp.zeros((2,self.count));self.status=xp.zeros(self.count,dtype=xp.int32)
        self.transcritical=xp.zeros(self.count,dtype=xp.int32)
        self.law,_=selected(p.identity['river_roughness']['method'],p.identity['river_roughness']['parameters'])
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=('normal_outlet',))
        self.identity=dict(method='normal_outlet.connected_section_manual_slope_v2',source_sha256=digest(__file__),
            runtime=self.engine.identity,section_boundaries=boundaries,section_faces=chains,bed_slopes=slopes,
            section='ordered physical boundary faces, separate disconnected owner-connected chains',
            closure='single stage section characteristic with conveyance and selected resistance; manual bed slope is energy slope proxy',
            supercritical_branch='A supercritical normal-flow relation is accepted as an outgoing characteristic state. Its stage cannot be imposed from downstream, so the exterior trace preserves the interior outgoing state.',
            limitations='no dynamic bedform roughness; outflow only; no explicit reverse-flow forcing')

    def evaluate(self,q,z):
        e=self.e;p=e.p;g=p.g
        self.engine.launch('normal_outlet',self.count,p.nf,p.n,self.count,p.parameters.gravity,p.parameters.dry,
            float(e.a.data.get('speed_limit',50.)),self.law,self.offset,self.faces,g['owner'],q,z,p.exterior_bed,
            g['normal'],g['length'],e.roughness,self.slopes,self.stage,self.metrics,e.trace,e.flux,e.status,self.status,self.transcritical)
        if bool(e.xp.any(self.status)):
            raise BendTrialRejected('Normal-depth outlet has no admissible subcritical section root; check slope, roughness and outlet flow regime')
