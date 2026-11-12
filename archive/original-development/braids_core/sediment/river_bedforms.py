"""Equilibrium bedform roughness on the maintained class/layer river route.

One explicit, current-only bundle: Van Rijn 1984 height/length/roughness,
FLOW trachytope 105, area average 1 and ROUMET=C. Fixed Cartesian cut cells,
CPU or resident CUDA chosen by the caller. Not QGIS/implicit/AMR admission.
"""
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np
from .bend_methods_v1 import digest
from .static_bedforms_v102 import HEADER, StaticMethodsV102, StaticParametersV102, resolve_static_methods_v102, _modes
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

HERE=Path(__file__).resolve().parent
NAMES=('bdf_grains','bdf_predict','bdf_faces','bdf_gather','flow_inputs','flow_drag')


def grain_cdf_plan(diameters):
    """Immutable knots and class CDF values for the existing log-size mixture.

    Every class has half its probability on each side of log(d). The sorted
    union of endpoints/midpoints makes the mixture linear on every interval,
    even with overlapping, unordered or coincident grain classes.
    """
    diam=np.asarray(diameters,dtype=np.float64)
    if diam.ndim!=1 or not 1<=len(diam)<=32 or not np.isfinite(diam).all() or np.any(diam<=0):
        raise ValueError('Grain CDF requires 1..32 finite positive diameters')
    mid=np.log(diam);lo=mid+1.25*np.log(.75);hi=mid+1.25*np.log(1.5)
    knots=np.unique(np.r_[lo,mid,hi]);x=knots[:,None]
    cdf=np.where(x>=hi,1.,np.where(x<=lo,0.,np.where(x>mid,
        .5+.5*(x-mid)/(hi-mid),.5*(x-lo)/(mid-lo))))
    return np.ascontiguousarray(knots),np.ascontiguousarray(cdf)


def source():
    pre='#ifdef __CUDACC__\n#define BRAIDS_HD __device__\n#else\n#include <cmath>\nusing std::isfinite;\n#endif\n'
    flow=(HERE/'flow2dh_fixed_bed_kernels.h').read_text()
    flow=flow[flow.index('ENTRY void flow_inputs'):flow.index('// Cartesian tensor')]
    flow=flow.replace('double chezy,','const double*chezy,').replace('=chezy;','=chezy[i];').replace('(chezy*chezy*h)','(chezy[i]*chezy[i]*h)')
    return pre+HEADER.read_text()+'\n'+Path(__file__).with_suffix('.h').read_text()+'\n'+flow


def material_parameters(material):
    """FLOW trtrou's current-only relation assumes equal class solid density.

    The common density may differ from quartz; substituting the first class
    for a genuinely mixed-density material would be a different model.
    """
    definition = material.definition
    if len(set(definition.solid_density)) != 1:
        raise ValueError("FLOW bedform relation assumes equal solid density across all fractions")
    return StaticParametersV102(critical_shields=.047,
        relative_density=(definition.solid_density[0]-definition.water_density)/definition.water_density,
        sediment_density_kg_m3=definition.solid_density[0],
        water_density_kg_m3=definition.water_density)


def install_bedforms(a,folder,*,enabled=True):
    """Install before the first step; replace total roughness, never add drag twice.

    Equilibrium geometry is predicted once from the accepted start state. Its
    components are frozen within the hydraulic/Exner trial; local depth still
    updates the face Chezy and z0. All arrays belong to the parent's atomic
    snapshot/restart, so rejected or cancelled trials cannot advance feedback.
    """
    p=a.physics;xp=a.xp
    if a.time or a.steps or p._open:raise ValueError('Install bedforms before execution')
    if hasattr(p,'bedform_state'):raise ValueError('Bedform feedback already installed')
    if type(enabled) is not bool:raise ValueError('Explicit bedform switch required')
    if not enabled:
        p.bedform_state=xp.zeros((9,p.n));p.bedform_chezy=xp.full(p.n,p.chezy)
        p.bedform_updates=xp.zeros(1,dtype=xp.int64)
        p.identity['river_bedforms']=dict(methods={'bedforms':False},parameters={},enabled=False,
            wrapper_sha256=digest(__file__),roughness='prescribed constant Chezy; no bedform feedback')
        a.identity['selected_river_bundle']['bedforms_integrated']=False
        return a
    if not hasattr(a,'columns'):raise ValueError('Bedform feedback requires the finite class/layer owner')
    ng=len(a.methods.diameter_m)
    if not 1<=ng<=32:raise ValueError('Bedform mixture supports 1..32 fractions')
    methods=StaticMethodsV102(capacity='off',bedforms=True,height='van_rijn_1984',length='van_rijn_1984',roughness='van_rijn_1984')
    params=material_parameters(a.material)
    mode=xp.asarray(_modes(resolve_static_methods_v102(methods),params));wire=xp.asarray(params.wire());diam=xp.asarray(a.methods.diameter_m)
    knots,cdf=grain_cdf_plan(a.methods.diameter_m);nk=len(knots)
    grain_knots=xp.asarray(knots);grain_cdf=xp.asarray(cdf)
    from braids_core.windows_paths import filesystem_path
    target=filesystem_path(folder)/'river_bedforms.h';target.parent.mkdir(parents=True,exist_ok=True);code=source()
    if target.exists():assert target.read_text()==code
    else:target.write_text(code)
    engine=WaveEngine(a.backend,p.flow.threads,source=target,names=NAMES)
    p.bedform_state=xp.zeros((9,p.n));p.bedform_grains=xp.zeros((2,p.n));p.bedform_status=xp.zeros(p.n,dtype=xp.int32)
    p.bedform_chezy=xp.full(p.n,p.chezy);p.bedform_faces=xp.zeros((3,p.nf))
    p.bedform_updates=xp.zeros(1,dtype=xp.int64)
    previous_local=p.local;previous_launch=p.flow.launch
    def predict(q):
        engine.launch('bdf_grains',p.n,p.n,ng,nk,a.columns.composition.weights(a.columns.available),
            diam,grain_knots,grain_cdf,p.bedform_grains)
        engine.launch('bdf_predict',p.n,p.n,p.sediment_depth,mode,wire,q,p.bedform_chezy,p.bedform_grains,p.bedform_state,p.bedform_status)
        if bool(xp.any(p.bedform_status)):raise ValueError('Invalid equilibrium bedforms')
    def effective(owner,q,z):
        g=p.g;raw=getattr(p,'_effective_physical_q',q)
        # Compute current-state geometry once. The baseline Chezy/shear kernel
        # used to be run here and immediately overwritten by bedform feedback.
        # Face interpolation, wet weights and the diagnostic depth now share
        # the two bedform face/gather loops, preserving their separate floors.
        p.effective.launch('effective_wet_gradient',p.n,p.n,p.parameters.dry,raw,z,
            g['offset'],g['faces'],g['owner'],g['neighbour'],g['xy'],p.effective_gradient)
        engine.launch('bdf_faces',p.nf,p.nf,p.n,p.parameters.dry,raw,z,p.effective_gradient,p.bedform_state,
            g['owner'],g['neighbour'],g['bid'],g['distance'],g['displacement'],p.effective_weights,
            p.effective_face,p.effective_wet_weights,p.bedform_faces)
        if hasattr(p,'exterior_bedforms'):p.exterior_bedforms.faces(raw,z)
        engine.launch('bdf_gather',p.n,p.n,p.nf,p.chezy,q,p.bedform_faces,p.effective_wet_weights,
            p.effective_face,g['offset'],g['faces'],g['owner'],p.bedform_chezy,p.effective_fields,p.effective_q)
        if not bool(xp.all(xp.isfinite(p.effective_fields))) or not bool(xp.all(p.bedform_chezy>0)):
            raise ValueError('Invalid bedform roughness feedback')
    def refresh(q,z):
        safe,_=p.safe(q);p._effective_physical_q=q
        try:effective(p,safe,z)
        finally:del p._effective_physical_q
    def friction(owner,q,dt,*args):
        if not p._open:predict(q);p.bedform_updates+=1
        refresh(q,a.z if not p._open else p.z0)
        safe,wet=p.safe(q)
        engine.launch('flow_drag',p.n,p.n,float(dt),p.parameters.gravity,p.bedform_chezy,safe)
        q[1:]=xp.where(wet,safe[1:],0.)
    def local(owner,q,z,olddepth,hi):
        refresh(q,z);return previous_local(q,z,olddepth,hi)
    def launch(name,count,*args):
        if name=='flow_inputs':return engine.launch(name,count,args[0],args[1],p.bedform_chezy,*args[3:])
        return previous_launch(name,count,*args)
    p.effective_inputs=MethodType(effective,p);p.friction=MethodType(friction,p);p.local=MethodType(local,p);p.flow.launch=launch
    predict(a.q);refresh(a.q,a.z)
    p.identity['river_bedforms']=dict(enabled=True,wrapper_sha256=digest(__file__),runtime=engine.identity,
        predictor_sha256=digest(HEADER),methods=asdict(methods),parameters=asdict(params),
        density_assumption='FLOW trtrou: equal solid density across fractions; explicit common value',
        assembly='FLOW trtrou 105 quadratic components; calrou area 1; ROUMET=C',
        mapping='true internal/open faces, distance interpolation; face-pyramid weighted drag and z0; walls/dry faces excluded',
        state='equilibrium predictor once per accepted-start trial; dry cells retain state; transactional arrays',
        implementation='direct piecewise-linear log-grain CDF inverse; fused current-state face roughness and depth gather',
        grain_correction='no external correction: MPM tranb2 owns its one ripple factor; EH none',
        consumers=['hydraulic_friction','secondary_flow','sediment_capacity','sediment_wave'],
        dynamic_dune_advection=False,public_support=False)
    a.identity['selected_river_bundle']['bedforms_integrated']=True
    return a
