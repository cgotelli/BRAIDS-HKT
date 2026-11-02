"""One accepted-start bedform predictor and live-depth exterior resistance.

Uses the same constitutive bundle as the interior, evaluated on the true-face
hydraulics and exterior surface material. No additional drag/source or solver.
"""
from pathlib import Path
from types import MethodType
import numpy as np
from . import river_bedforms as bedforms
from .river_roughness import selected
from .bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('bdf_grains','bdf_predict','exterior_bdf_weights','exterior_bdf_resistance','exterior_bdf_faces')


class ExteriorBedforms:
    def __init__(self,sediment,folder):
        self.s=sediment;self.e=sediment.exterior;self.a=sediment.a;self.p=sediment.p;self.xp=sediment.xp
        a,p,xp=self.a,self.p,self.xp;ne=sediment.ne
        if a.time or a.steps or p._open or hasattr(p,'exterior_bedforms'):
            raise ValueError('Install exterior bedforms once before execution')
        if not self.e.bedforms_enabled:raise ValueError('Exterior bedforms require selected interior bedforms')
        methods=bedforms.StaticMethodsV102(**p.identity['river_bedforms']['methods'])
        params=bedforms.StaticParametersV102(**p.identity['river_bedforms']['parameters'])
        self.modes=xp.asarray(bedforms._modes(bedforms.resolve_static_methods_v102(methods),params))
        self.wire=xp.asarray(params.wire());self.diam=xp.asarray(a.methods.diameter_m)
        knots,cdf=bedforms.grain_cdf_plan(a.methods.diameter_m);self.nk=len(knots)
        self.knots,self.cdf=xp.asarray(knots),xp.asarray(cdf)
        from braids_core.windows_paths import filesystem_path
        path=filesystem_path(folder)/'exterior_bedforms.h';code=bedforms.source()+'\n'+Path(__file__).with_suffix('.h').read_text()
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_text()!=code:raise ValueError('Exterior bedform runtime source differs')
        if not path.exists():path.write_text(code)
        self.engine=WaveEngine(a.backend,p.flow.threads,source=path,names=NAMES)
        p.exterior_bedform_state=xp.zeros((9,ne));p.exterior_bedform_chezy=xp.empty(ne)
        p.exterior_bedform_updates=xp.zeros(1,dtype=xp.int64)
        self.grains=xp.empty((2,ne));self.status=xp.zeros(ne,dtype=xp.int32);self.values=xp.empty((3,ne))
        self.initializing=True;self.e.bedforms=self;p.exterior_bedforms=self
        self.e.evaluate(a.q,a.z)
        h=xp.maximum(p.parameters.dry,self.e.trace[0,sediment.faces])
        law=p.identity['river_roughness'];mode,value=selected(law['method'],law['parameters'])
        c=(xp.full(ne,value) if mode==1 else h**(1./6.)/value if mode==2 else
           np.sqrt(9.81)/.41*xp.log1p(30.*h/(np.e*value)) if mode==3 else
           18.*xp.log10(12.*xp.maximum(.5,h/value)))
        p.exterior_bedform_chezy[:]=c
        self.predict(a.q,a.z);self.initializing=False
        previous=p.friction
        def friction(owner,q,dt,*args):
            if not p._open:
                self.predict(q,a.z);p.exterior_bedform_updates+=1
            return previous(q,dt,*args)
        p.friction=MethodType(friction,p)
        # Refresh the single resistance gather now that exterior faces exist.
        safe,_=p.safe(a.q);p._effective_physical_q=a.q
        try:p.effective_inputs(safe,a.z)
        finally:del p._effective_physical_q
        self.e.owner.identity['exterior_bedforms']=dict(source_sha256=digest(__file__),runtime=self.engine.identity,
            methods=p.identity['river_bedforms']['methods'].copy(),parameters=p.identity['river_bedforms']['parameters'].copy(),
            geometry='equilibrium predictor at true exterior face using independent exterior material',
            cadence='accepted-start trial only; frozen components with live physical-depth resistance',
            initialization='explicit selected static law; thereafter bedform total resistance',
            assembly='trachytope 105 quadratic components; half-depth cap then calrou 1 mm floor; ROUMET=C',
            consumers=['hydraulic boundary section weights','physical face drag gather','secondary-flow shear','sediment capacity'],
            ownership='zero-area constitutive history in parent transaction and physical-overlap remesh',public_support=False)

    def predict(self,q,z):
        p,s,xp=self.p,self.s,self.xp;ne=s.ne
        self.e.evaluate(q,z)
        trace=xp.ascontiguousarray(self.e.trace[:,s.faces])
        weights=xp.ascontiguousarray(self.a.columns.composition.weights(p.exterior_mass[0]))
        self.engine.launch('bdf_grains',ne,ne,s.ng,self.nk,weights,self.diam,self.knots,self.cdf,self.grains)
        self.engine.launch('bdf_predict',ne,ne,p.sediment_depth,self.modes,self.wire,trace,
            p.exterior_bedform_chezy,self.grains,p.exterior_bedform_state,self.status)
        if bool(xp.any(self.status)):raise ValueError('Invalid exterior equilibrium bedforms')

    def weights(self,q,z):
        p,s,e=self.p,self.s,self.e;g=p.g
        # Non-open entries stay zero; only compact exterior entries are written.
        self.engine.launch('exterior_bdf_weights',s.ne,s.ne,p.nf,p.n,e.nb,p.parameters.dry,
            s.faces,s.owner,g['bid'],e.hydraulic_mode,q,z,p.exterior_bed,g['length'],e.stage,e.forcing,
            p.exterior_bedform_state,e.weights)

    def resistance(self,trace):
        p,s=self.p,self.s
        self.engine.launch('exterior_bdf_resistance',s.ne,s.ne,p.parameters.dry,trace,p.exterior_bedform_state,self.values)
        p.exterior_bedform_chezy[:]=self.values[0]
        return self.values

    def faces(self,q,z):
        p,s,xp=self.p,self.s,self.xp
        self.e.evaluate(q,z);trace=xp.ascontiguousarray(self.e.trace[:,s.faces])
        values=self.resistance(trace)
        self.engine.launch('exterior_bdf_faces',s.ne,s.ne,p.nf,p.n,p.parameters.dry,s.faces,s.owner,
            q,trace,values,p.effective_weights,p.effective_face,p.effective_wet_weights,p.bedform_faces)
