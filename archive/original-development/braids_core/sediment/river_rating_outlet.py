"""Native CPU/GPU characteristic outlet for a whole-section Q–H curve."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


class RatingOutlet:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        control=[];offset=[0];q=[];stage=[]
        for b in range(e.nb):
            row=controls[str(b)];active=row['method']=='outlet.rating_curve';control.append(active)
            if active:
                for discharge,elevation in row['rating_curve']:q.append(discharge);stage.append(elevation)
            offset.append(len(q))
        self.control=xp.asarray(control,dtype=xp.int32);self.offset=xp.asarray(offset,dtype=xp.int32)
        self.q=xp.asarray(q,dtype=xp.float64);self.elevation=xp.asarray(stage,dtype=xp.float64)
        self.stage=xp.zeros(e.nb);self.rate=xp.zeros(e.nb);self.status=xp.zeros(e.nb,dtype=xp.int32)
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=('rating_outlet',))
        self.identity=dict(method='outlet.rating_curve.characteristic_whole_section_v1',source_sha256=digest(__file__),
            runtime=self.engine.identity,curves={k:v['rating_curve'] for k,v in controls.items() if v['method']=='outlet.rating_curve'},
            definition='common water elevation and total outward discharge across the named opening; piecewise linear Q(H); no extrapolation',
            regimes='outward subcritical characteristic, dry owner closed, outgoing supercritical extrapolated, local backflow reflected',
            assumption='single-valued stationary Q–elevation curve; user supplies datum and coverage')

    def evaluate(self,q,z):
        e=self.e;p=e.p;g=p.g
        self.engine.launch('rating_outlet',e.nb,p.nf,p.n,e.nb,p.parameters.gravity,p.parameters.dry,
            float(e.a.data.get('speed_limit',50.)),self.control,e.offset,e.faces,g['owner'],q,z,p.exterior_bed,
            g['normal'],g['length'],self.offset,self.q,self.elevation,self.stage,self.rate,e.trace,e.flux,e.status,self.status)
        if bool(e.xp.any(self.status)):
            raise ValueError('Q–H outlet has no admissible subcritical solution within the supplied curve. Extend or correct the discharge/elevation range, or choose a different outlet condition.')
