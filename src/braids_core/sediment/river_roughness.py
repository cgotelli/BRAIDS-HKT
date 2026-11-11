"""One selected resistance law feeding the shared CPU/GPU river physics.

Manning and both FLOW White-Colebrook relations are evaluated at current
physical face depths. Bedforms, when enabled, own total resistance instead;
the selected prescribed law supplies initialization and isolated-cell fallback.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import numpy as np

from . import river_bedforms as bedforms
from .river_capacity import replace_once
from .bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES = (*bedforms.NAMES, "river_roughness_cells")
KEYS = {"roughness.chezy": "chezy_m_half_s", "roughness.manning": "manning_s_m_third",
        "roughness.white_colebrook": "nikuradse_m"}


def selected(method, parameters):
    if method not in KEYS:
        raise ValueError("Unknown prescribed roughness")
    mode = {"roughness.chezy": 1, "roughness.manning": 2, "roughness.white_colebrook": 3}[method]
    if mode == 3:
        relation = parameters["white_colebrook_relation"]
        if relation not in ("flow_log_layer", "classical_2d"):
            raise ValueError("Explicit White-Colebrook relation required")
        if relation == "classical_2d":
            mode = 4
    value = float(parameters[KEYS[method]])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Positive finite roughness coefficient required")
    return mode, value


def source(spatial=False):
    code = bedforms.source()
    local = """
BDF_LOCAL double river_chezy(int method,double depth,double value){
 if(method==1)return value;
 if(method==2)return pow(depth,1./6.)/value;
 if(method==3)return sqrt(9.81)/.41*log1p(30.*depth/(exp(1.)*value));
 return 18.*log10(12.*fmax(.5,depth/value));
}
ENTRY void river_roughness_cells(int n,int method,double value,double dry,const double*q,double*out){
 LOOP(n){out[i]=river_chezy(method,fmax(dry,q[i]),value);}
}
"""
    code = replace_once(code, "ENTRY void bdf_faces", local+"\nENTRY void bdf_faces")
    code = replace_once(code, "const double*disp,const double*weights,double*depth,double*wetweights,double*out){",
        "const double*disp,const double*weights,double*depth,double*wetweights,double*out,int method,double value,int feedback){")
    start = code.index("  double ks2=0.;")
    stop = code.index("  out[i]=c;", start)
    code = code[:start] + """  double c,ks;
  if(feedback){
   double ks2=0.;for(int k=2;k<5;++k){double r=wo*components[k*n+o]+(j>=0?wj*components[k*n+j]:0.);ks2+=r*r;}
   ks=fmax(.001,fmax(1e-6,fmin(sqrt(ks2),.5*h)));
   c=18.*log10(fmax(12.*h/ks,1.0129));
  }else{
   c=river_chezy(method,h,value);
   ks=method>=3?value:12.*h/pow(10.,c/18.);
  }
""" + code[stop:]
    code = replace_once(code, "int nf,double baseline,const double*q", "int nf,const double*baseline,const double*q")
    code = replace_once(code, "sqrt(sum/drag):baseline;", "sqrt(sum/drag):baseline[i];")
    code = replace_once(code, ".41*baseline/sqrt(9.81)", ".41*baseline[i]/sqrt(9.81)")
    if spatial:
        code=replace_once(code,'river_roughness_cells(int n,int method,double value,','river_roughness_cells(int n,int method,const double*values,')
        code=replace_once(code,'LOOP(n){out[i]=river_chezy(method,fmax(dry,q[i]),value);}',
            'LOOP(n){out[i]=river_chezy(method,fmax(dry,q[i]),values[i]);}')
        code=replace_once(code,'double*out,int method,double value,int feedback){','double*out,int method,const double*values,int feedback){')
        code=replace_once(code,'  double c,ks;', '''  double value=values[o];
  if(j>=0){
   if(method==1)value=1./sqrt(wo/(value*value)+wj/(values[j]*values[j]));
   else if(method==2)value=sqrt(wo*value*value+wj*values[j]*values[j]);
   else value=wo*value+wj*values[j];
  }
  double c,ks;''')
    return code


def install(a, folder, *, method, parameters, bedforms_enabled, spatial=None, zones=()):
    mode, value = selected(method, parameters)
    p, xp = a.physics, a.xp
    if a.time or a.steps or p._open or hasattr(p, "bedform_state"):
        raise ValueError("Install selected roughness once before stepping")
    identity = dict(method=method, parameters=deepcopy(parameters), wrapper_sha256=digest(__file__),
        total_resistance_owner="bedform.van_rijn_1984" if bedforms_enabled else method,
        prescribed_role="initialization and isolated-cell fallback" if bedforms_enabled else "total resistance",
        white_colebrook="FLOW taubot ChzK2d=false (log layer) or true (classical 2D)",
        consumers=["hydraulic_friction", "secondary_flow", "sediment_capacity", "sediment_wave"],
        explicit_backend=a.backend, double_drag=False)
    if spatial is not None:
        if np.shape(spatial)!=(p.n,) or not np.isfinite(spatial).all() or np.any(spatial<=0):raise ValueError('Finite positive cell roughness required')
        p.spatial_roughness=xp.asarray(spatial)
        value=p.spatial_roughness
        identity['zones']=deepcopy(zones)
        identity['zone_mapping']='fluid overlap area; Manning n squared / Chezy inverse square / Colebrook ks'
    if mode == 1 and spatial is None:
        if value != p.chezy:
            raise ValueError("Configured Chezy differs from the selected roughness coefficient")
        # Preserve the qualified constant-Chezy and bedform trajectories.
        bedforms.install_bedforms(a, folder, enabled=bedforms_enabled)
        p.identity["river_roughness"] = identity
        return a
    from braids_core.windows_paths import filesystem_path
    path = filesystem_path(folder)/"resolved_roughness.h"
    code = source(spatial is not None)
    if path.exists() and path.read_text() != code:
        raise ValueError("Existing roughness source differs")
    if not path.exists():
        path.write_text(code)
    engine = WaveEngine(a.backend, p.flow.threads, source=path, names=NAMES)
    p.prescribed_chezy = xp.empty(p.n)
    p.bedform_state = xp.zeros((9, p.n))
    p.bedform_chezy = xp.empty(p.n)
    p.bedform_faces = xp.zeros((3, p.nf))
    p.bedform_updates = xp.zeros(1, dtype=xp.int64)
    p.bedform_status = xp.zeros(p.n, dtype=xp.int32)
    methods = bedforms.StaticMethodsV102(capacity="off", bedforms=True,
        height="van_rijn_1984", length="van_rijn_1984", roughness="van_rijn_1984")
    params = (bedforms.material_parameters(a.material) if bedforms_enabled
              else bedforms.StaticParametersV102(critical_shields=.047))
    if bedforms_enabled:
        ng = len(a.methods.diameter_m)
        modes = xp.asarray(bedforms._modes(bedforms.resolve_static_methods_v102(methods), params))
        wire, diam = xp.asarray(params.wire()), xp.asarray(a.methods.diameter_m)
        knots, cdf = bedforms.grain_cdf_plan(a.methods.diameter_m)
        nk = len(knots)
        knots, cdf = xp.asarray(knots), xp.asarray(cdf)
        p.bedform_grains = xp.zeros((2, p.n))
    previous_local, previous_launch = p.local, p.flow.launch
    from .shoreline_solver import ShorelinePhysics
    # Preserve third-party/subclass wrappers rather than bypassing their work.
    share_safe = getattr(previous_local, '__func__', None) is ShorelinePhysics.local

    def prescribed(q):
        engine.launch("river_roughness_cells", p.n, p.n, mode, value, p.parameters.dry, q, p.prescribed_chezy)

    def predict(q):
        if not bedforms_enabled:
            return
        engine.launch("bdf_grains", p.n, p.n, ng, nk, a.columns.composition.weights(a.columns.available),
                      diam, knots, cdf, p.bedform_grains)
        engine.launch("bdf_predict", p.n, p.n, p.sediment_depth, modes, wire, q, p.bedform_chezy,
                      p.bedform_grains, p.bedform_state, p.bedform_status)
        if bool(xp.any(p.bedform_status)):
            raise ValueError("Invalid bedform geometry from selected roughness")

    def effective(owner, q, z):
        g = p.g
        raw = getattr(p, "_effective_physical_q", q)
        prescribed(raw)
        p.effective.launch("effective_wet_gradient", p.n, p.n, p.parameters.dry, raw, z,
            g["offset"], g["faces"], g["owner"], g["neighbour"], g["xy"], p.effective_gradient)
        engine.launch("bdf_faces", p.nf, p.nf, p.n, p.parameters.dry, raw, z, p.effective_gradient,
            p.bedform_state, g["owner"], g["neighbour"], g["bid"], g["distance"], g["displacement"],
            p.effective_weights, p.effective_face, p.effective_wet_weights, p.bedform_faces,
            mode, value, int(bedforms_enabled))
        if hasattr(p,"exterior_bedforms"):p.exterior_bedforms.faces(raw,z)
        engine.launch("bdf_gather", p.n, p.n, p.nf, p.prescribed_chezy, q, p.bedform_faces,
            p.effective_wet_weights, p.effective_face, g["offset"], g["faces"], g["owner"],
            p.bedform_chezy, p.effective_fields, p.effective_q)
        if not bool(xp.all(xp.isfinite(p.effective_fields))) or not bool(xp.all(p.bedform_chezy > 0)):
            raise ValueError("Nonfinite or nonpositive selected roughness")

    def refresh(q, z):
        safe, wet = p.safe(q)
        p._effective_physical_q = q
        try:
            effective(p, safe, z)
        finally:
            del p._effective_physical_q
        return safe, wet

    def friction(owner, q, dt, *args):
        if bedforms_enabled and not p._open:
            predict(q)
            p.bedform_updates += 1
        safe, wet = refresh(q, a.z if not p._open else p.z0)
        engine.launch("flow_drag", p.n, p.n, float(dt), p.parameters.gravity, p.bedform_chezy, safe)
        q[1:] = xp.where(wet, safe[1:], 0.)

    def local(owner, q, z, olddepth, hi):
        safe, wet = refresh(q, z)
        if share_safe:
            return p.local_with_safe(z, olddepth, hi, safe, wet)
        return previous_local(q, z, olddepth, hi)

    def launch(name, count, *args):
        if name == "flow_inputs":
            return engine.launch(name, count, args[0], args[1], p.bedform_chezy, *args[3:])
        return previous_launch(name, count, *args)

    p.effective_inputs, p.friction = MethodType(effective, p), MethodType(friction, p)
    p.local, p.flow.launch = MethodType(local, p), launch
    prescribed(a.q)
    p.bedform_chezy[:] = p.prescribed_chezy
    predict(a.q)
    refresh(a.q, a.z)
    identity.update(runtime=engine.identity,
        mapping="physical face depths; inverse-square face-pyramid drag average; weighted z0; closed/dry faces excluded",
        secondary_minimum_chezy=20., sediment_working_minimum_chezy=1.e-6,
        hydraulic_chezy_floor=False, geometry="reconstructed surface minus higher adjacent bed")
    p.identity["river_roughness"] = identity
    p.identity["river_bedforms"] = dict(enabled=bedforms_enabled,
        methods=asdict(methods) if bedforms_enabled else {"bedforms": False},
        parameters=asdict(params) if bedforms_enabled else {}, runtime=engine.identity,
        wrapper_sha256=digest(__file__), roughness_owner=identity,
        state="accepted-start equilibrium prediction; transactional state; local depth updates resistance",
        public_support=False)
    a.identity["selected_river_bundle"]["bedforms_integrated"] = bedforms_enabled
    return a
