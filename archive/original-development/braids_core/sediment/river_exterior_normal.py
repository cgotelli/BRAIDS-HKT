"""Normal depth from section force balance on actual exterior face geometry."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from .river_roughness import selected
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('section_normal_candidates','section_normal_metrics')


def neighbours(data,controls):
    """Physical adjacency, not mesh enumeration; gaps terminate wetted banks."""
    nf=len(data['owner']);left=np.full(nf,-1,np.int32);right=left.copy()
    for b in range(len(data['mode'])):
        if controls[str(b)]['method']!='inlet.normal_discharge':continue
        ids=np.flatnonzero((data['neighbour']<0)&(data['bid']==b))
        if not len(ids):raise ValueError('Normal inlet has no physical faces')
        normal=data['normal'][:,ids[0]];relative=data['mid'][:,ids]-data['mid'][:,ids[0],None]
        tolerance=max(1e-10,64*np.spacing(max(1.,float(np.max(abs(data['mid'][:,ids]))))))
        if (not np.allclose(data['normal'][:,ids],normal[:,None],atol=1e-12,rtol=0) or
            np.any(abs(normal@relative)>tolerance)):
            raise ValueError('Normal inlet requires one collinear consistently oriented section')
        station=np.array([-normal[1],normal[0]])@relative;order=np.argsort(station);ids=ids[order];station=station[order]
        widths=data['length'][ids];gaps=np.diff(station)-.5*(widths[:-1]+widths[1:])
        if np.any(gaps < -tolerance):raise ValueError('Normal inlet faces overlap')
        for i in np.flatnonzero(abs(gaps)<=tolerance):left[ids[i+1]]=ids[i];right[ids[i]]=ids[i+1]
    return left,right


class NormalSection:
    def __init__(self,exterior,controls):
        e=exterior;p=e.p;xp=e.xp;self.e=e
        self.control=xp.asarray([controls[str(b)]['method']=='inlet.normal_discharge' for b in range(e.nb)],dtype=xp.int32)
        from .river_section_cache import SectionCache
        self.cache=SectionCache(e,self.control)
        self.slope=xp.asarray([controls[str(b)].get('energy_slope',0.) for b in range(e.nb)],dtype=xp.float64)
        left,right=neighbours(e.a.data,controls);self.left=xp.asarray(left);self.right=xp.asarray(right)
        outside=np.flatnonzero((e.a.data['neighbour']<0)&(e.a.data['bid']>=0));self.ne=len(outside)
        compact=np.full(p.nf,-1,np.int32);compact[outside]=np.arange(self.ne,dtype=np.int32)
        self.compact=xp.asarray(compact);self.empty_state=xp.zeros((9,self.ne))
        roughness=p.identity['river_roughness'];self.law,self.parameter=selected(roughness['method'],roughness['parameters'])
        self.engine=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.selection=WaveEngine(e.a.backend,p.flow.threads,source=Path(__file__).with_name('river_exterior_section.h'),
            names=('section_critical_partial','section_critical_total'))
        self.candidate=xp.zeros(p.nf);self.rank=xp.zeros(p.nf);self.stage_part=xp.zeros(e.nb*e.tiles)
        self.rank_part=xp.zeros(e.nb*e.tiles);self.status=xp.zeros(e.nb,dtype=xp.int32);self.values=xp.zeros((4,e.nb))
        self.identity=dict(method='section.normal_uniform_velocity_force_balance',source_sha256=digest(__file__),
            runtime=self.engine.identity,reduction=self.selection.identity,
            geometry='actual flat exterior faces; wetted vertical steps and terminal sidewalls; no artificial partition walls',
            criterion='Q^2 * sum(P_i/C_i^2) / A^3 = energy_slope; uniform normal Q/A',
            resistance='selected law at section R=A/P; evolving bedform ks retains local-depth cap; perimeter drag aggregation',
            roughness=roughness,energy_slope={str(b):controls[str(b)]['energy_slope'] for b in range(e.nb)
                if controls[str(b)]['method']=='inlet.normal_discharge'},
            root='lowest bracketed root; 32 sub-brackets per wet-width branch, 80 bisections, final rate residual check',
            parallel='independent wet-width branches; deterministic 64-face minimum tiles',
            state='derived scratch only; no host numerical root or backend fallback')

    def _arguments(self):
        e=self.e;p=e.p
        feedback=int(e.bedforms is not None and not e.bedforms.initializing)
        state=p.exterior_bedform_state if feedback else self.empty_state
        return (e.nb,self.ne,p.parameters.gravity,p.parameters.dry,self.law,self.parameter,feedback),state

    def solve(self):
        e=self.e;p=e.p;head,state=self._arguments()
        inputs=self.cache.inputs(self.slope,state)
        if self.cache.restore(inputs,head):return
        self.engine.launch('section_normal_candidates',p.nf,p.nf,*head,e.selected,p.g['bid'],self.control,e.offset,e.faces,
            self.left,self.right,self.compact,p.exterior_bed,p.g['length'],state,self.slope,e.forcing,self.candidate,self.rank)
        self.selection.launch('section_critical_partial',e.nb*e.tiles,e.nb*e.tiles,e.nb,e.tiles,e.offset,e.faces,
            self.candidate,self.rank,self.stage_part,self.rank_part)
        self.selection.launch('section_critical_total',e.nb,e.nb,e.tiles,self.control,self.stage_part,self.rank_part,e.forcing,self.status)
        if bool(e.xp.any(self.status)):raise BendTrialRejected('Normal inlet has no converged resolved section root')
        self.cache.store(inputs,head)

    def metrics(self):
        e=self.e;p=e.p;head,state=self._arguments()
        self.engine.launch('section_normal_metrics',e.nb,*head,self.control,e.offset,e.faces,self.left,self.right,self.compact,
            p.exterior_bed,p.g['length'],state,self.slope,e.forcing,self.values)
        return self.values
