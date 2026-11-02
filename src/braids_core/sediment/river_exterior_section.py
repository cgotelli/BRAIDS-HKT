"""Critical section roots consistent with executed exterior face areas."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('section_critical_candidates','section_critical_partial','section_critical_total')


class CriticalSection:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        selected=np.array([controls[str(b)]['method']=='inlet.critical_discharge' for b in range(e.nb)],np.int32)
        self.control=xp.asarray(selected)
        from .river_section_cache import SectionCache
        self.cache=SectionCache(e,self.control)
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.candidate=xp.zeros(p.nf);self.energy=xp.zeros(p.nf)
        self.stage_part=xp.zeros(e.nb*e.tiles);self.energy_part=xp.zeros(e.nb*e.tiles)
        self.status=xp.zeros(e.nb,dtype=xp.int32)
        self.identity=dict(method='section.critical_minimum_specific_energy',source_sha256=digest(__file__),
            runtime=self.engine.identity,geometry='actual exterior face bed; piecewise flat; same h > dry face area as executed flux',
            criterion='Q^2 * wet_width / (g * wetted_area^3) = 1; minimum specific energy among admissible branches',
            discharge='selected total Q with uniform section-normal speed; not facewise Froude one on irregular bed',
            zero_discharge='stage at minimum exterior bed; zero wet inlet area',
            unresolved='reject positive Q with no critical root above hydraulic dry depth',
            state='derived scratch only; recompute from current accepted exterior bed and sampled Q',
            parallel='independent wet-width branches; deterministic 64-face tiled minimum')

    def solve(self):
        e=self.e;p=e.p;g=p.g;op=self.engine
        inputs=self.cache.inputs();scalars=(p.parameters.gravity,p.parameters.dry)
        if self.cache.restore(inputs,scalars):return
        op.launch('section_critical_candidates',p.nf,p.nf,e.nb,p.parameters.gravity,p.parameters.dry,
            e.selected,g['bid'],self.control,e.offset,e.faces,p.exterior_bed,g['length'],e.forcing,self.candidate,self.energy)
        op.launch('section_critical_partial',e.nb*e.tiles,e.nb*e.tiles,e.nb,e.tiles,
            e.offset,e.faces,self.candidate,self.energy,self.stage_part,self.energy_part)
        op.launch('section_critical_total',e.nb,e.nb,e.tiles,self.control,self.stage_part,self.energy_part,e.forcing,self.status)
        if bool(e.xp.any(self.status)):
            raise BendTrialRejected('Critical inlet has no resolved section root above hydraulic dry depth')
        self.cache.store(inputs,scalars)
