"""Resolved current-only scalar capacity on the shared river owner.

Cells, sediment-wave derivatives and suspended equilibrium use one generated
constitutive function, compiled for OpenMP/CUDA. Hydraulic f-waves, physical
effective velocity, reconstruction and finite-layer ownership are unchanged.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from braids_core.river_model import ModelError
from .flow2dh_eh_operator import ClassicFlowEH
from .bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

HERE = Path(__file__).resolve().parent
METHODS = {"capacity.van_rijn_1993": -1, "capacity.engelund_hansen": 1, "capacity.mpm": 2, "capacity.general": 4,
           "capacity.soulsby_van_rijn": 11, "capacity.soulsby": 12,
           "capacity.van_rijn_1984_modified": 7, "capacity.wang_fredsoe": 13,
           "capacity.ashida_michiue_generalized": 14,
           "capacity.wilcock_crowe": 16, "capacity.wilcock_crowe_modified": 17,
           "capacity.gaeuman": 18, "capacity.smart_jaggi": 101,
           "capacity.ancey_recking_2023": 102, "capacity.recking_2010": 103,
           "capacity.camenen_larson_2005": 104, "capacity.wong_parker_2006": 105}
NAMES = ("split_capacity", "mixture", "scalar_probe", "scalar_forcing_probe", "sus_faces", "sus_limit",
         "sus_gradients", "sus_nonorthogonal",
         "sus_scale", "sus_gather", "sus_exchange", "sus_check", "intrinsic_capacity",
         "intrinsic_forcing_probe", "intrinsic_grains", "vector_forcing_probe", "vector_wave_probe",
         "vr_sus_exchange", "total_capacity", "total_suspended_probe", "total_radial_probe",
         "settling_only_exchange", "skin_stress_probe", "intrinsic_skin_probe", "smart_capacity")
WIRE_SIZE = 49


def parameter_wire(capacity, bed_factor=1., *, solid_density=2650., water_density=1000.,
                   viscosity=1.e-6, reference=None, skin_shear=None, diameter=None):
    method, p = capacity["method"], capacity["parameters"]
    from .grain_diameter_contract import REQUIRED, validate_grain_sizes
    sizes = validate_grain_sizes(capacity, primary=diameter) if method in REQUIRED else {}
    # Immutable measured properties, shared by cell, face-wave and exterior laws.
    bulk = [sizes.get('d90', 0.),
            (sizes['d90']/sizes['d30'])**.2 if method == 'capacity.smart_jaggi' else 1.,
            sizes['d84']/sizes['d50'] if method == 'capacity.recking_2010' else 1.]
    if method in ('capacity.smart_jaggi','capacity.ancey_recking_2023','capacity.recking_2010','capacity.camenen_larson_2005','capacity.wong_parker_2006'):
        from .capacity_parameters import resolve_capacity_parameters
        p = dict(resolve_capacity_parameters(method, p))
    if method not in METHODS:
        raise ModelError("integration_pending", "sediment.capacity", "Scalar river binding is not implemented for "+method)
    # FLOW17/18 have no ACal parameter: the unused wire slot is not a public
    # coefficient and must never be saved as one.
    acal = 1. if METHODS[method] in (-1, 17, 18) else p["ACal"]
    values = dict(LAW=METHODS[method], ACAL=acal, SHARE=p.get("SusFac", 0.),
        POWERB=p['ReachSlope'] if METHODS[method]==103 else p.get("PowerB", 0.), POWERC=p.get("PowerC", 1.5),
        RIPPLE=p.get("RipFac", 1.), CRITICAL=p.get("ThetaC", .047),
        SUSACAL=p.get("SusACal", 0.), SUSPOWERB=p.get("SusPowerB", 0.),
        SUSPOWERC=p.get("SusPowerC", 1.), SUSRIPPLE=p.get("SusRipFac", 1.),
        SUSCRITICAL=p.get("SusThetaC", 0.))
    if (any(isinstance(v, (bool, np.bool_)) for v in (solid_density, water_density))
            or not np.isfinite([solid_density, water_density]).all()
            or not solid_density > water_density > 0.):
        raise ValueError("Sediment must be denser than positive-density water")
    relative = (float(solid_density)-float(water_density))/float(water_density)
    extra = [float(viscosity), .001, 0., 0., 0., 0., 0.]
    if METHODS[method] == 7:
        extra[1:4] = [p["Aks"], p["WSettle"], p.get("BetaM", 0.)]
    elif METHODS[method] in (13,104):
        extra[0] = p["VicMol"]
    elif METHODS[method] == 14:
        extra[4:] = [p["PowerM"], p["PowerP"], p["PowerQ"]]
        if min(extra[4:]) < 0:
            raise ModelError("integration_pending", "sediment.capacity",
                             "Negative Ashida-Michiue powers require a qualified threshold treatment")
        if p["ThetaC"] < 0:
            raise ValueError("Critical Shields stress must be nonnegative")
    if extra[0] <= 0 or extra[1] <= 0 or extra[2] < 0:
        raise ValueError("Positive viscosity/reference height and nonnegative settling required")
    if METHODS[method] == 104:
        if isinstance(diameter,(bool,np.bool_)) or diameter is None or not np.isfinite(diameter) or diameter<=0:
            raise ValueError('Camenen & Larson requires a finite positive D50 in metres')
        # Camenen & Larson CR-07-1 Eq49 (Soulsby & Whitehouse 1997).
        # Grain properties are immutable: do not recompute this in every cell/face.
        dstar=diameter*np.cbrt(9.81*relative/(extra[0]*extra[0]))
        values['CRITICAL']=.30/(1.+1.2*dstar)-.055*np.expm1(-.02*dstar)
    elif METHODS[method] == 105:
        values['CRITICAL']=.0495 # fixed published threshold, not a user coefficient
    intrinsic = [p["ThetaC0"], p["Alpha0"]] if METHODS[method] in (17, 18) else [0., 0.]
    if min(intrinsic) < 0.:
        raise ValueError("Intrinsic mixture coefficients must be nonnegative")
    vector = [p.get("RatioD90D50", 1.), p.get("RouZ0", .0001),
              p.get("RatioD50Z0", 30.), p.get("ModInd", 1), p.get("AlfaUrms", 1.)]
    if (min(vector[:3]) <= 0. or vector[3] not in range(1, 9)
            or isinstance(vector[3], (bool, np.bool_))):
        raise ValueError("Positive vector roughness/ratios and integer ModInd 1..8 required")
    vr = [0.]*14
    if METHODS[method] == -1:
        from .river_vr1993 import PARAMETERS, REFERENCE_PARAMETERS
        if reference is None or set(reference) != set(REFERENCE_PARAMETERS):
            raise ValueError("Explicit van Rijn reference parameters required")
        vr = [p[k] for k in PARAMETERS]+[reference[k] for k in REFERENCE_PARAMETERS]
        if (not np.isfinite(vr).all() or min(vr[9:])<=0. or vr[11]>1.
                or (p["IopKCW"]==0 and p["RDC"]<=0.)):
            raise ValueError("Invalid van Rijn reference parameters")
    skin = [0., 0., 0.]
    if skin_shear is not None:
        from .river_skin_shear import validate_parameters
        validate_parameters(skin_shear)
        if METHODS[method] not in (16, 17, 18):
            raise ValueError("Only intrinsic capacities consume replaceable skin stress")
        skin = [1., skin_shear["sand_roughness_m"], skin_shear["viscosity_m2_s"]]
    wire = np.asarray([*values.values(), float(bed_factor), float(solid_density), relative,
                       *extra, *intrinsic, *vector, *vr, *skin, *bulk], dtype=np.float64)
    if not np.isfinite(wire).all() or bed_factor not in (0., 1.):
        raise ValueError("Invalid scalar formula coefficients")
    return wire


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("The audited river source changed at: "+old[:60])
    return source.replace(old, new)


def capacity_source():
    source = (HERE/"river_suspension.h").read_text()
    # The shared template admits finite positive physical resistance already;
    # a secondary-flow working limit is not a hydraulic/transport admission cap.
    # eqtran computes spiral-adjusted USTAR even if this class has zero
    # bedload. Its stress can still enter a neighbouring face correction.
    source = replace_once(source, "us>0.&&mag>0.?", "us>0.?")
    source = replace_once(source, "ENTRY void split_capacity",
                          (HERE/"river_scalar_law.h").read_text()+"\n"+
                          (HERE/"river_skin_shear.h").read_text()+"\n"+
                          (HERE/"river_vector_law.h").read_text()+"\nENTRY void split_capacity")
    source = replace_once(source, "const double*hi,double*out,int*status){",
                          "const double*hi,const double*d90,const double*hide,const double*law_params,const double*z0,double*out,int*status){")
    source = replace_once(source, "status[i]=valid?0:1;if(!valid)continue;",
        "valid=valid&&isfinite(d90[i])&&isfinite(hide[i])&&d90[i]>0.&&hide[i]>0.;\n"
        "  valid=valid&&isfinite(z0[i])&&z0[i]>0.;\n"
        "  status[i]=valid?0:1;if(!valid)continue;")
    source = replace_once(source,
        "double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*1.65*d),ex=fmax(0.,theta-.047);\n"
        "  double mag=bed_factor*2650.*8.*sqrt(9.81*1.65*d)*d*ex*sqrt(ex);",
        "double speed=hypot(ue,ve),theta=speed*speed/(c[i]*c[i]*law_params[14]*d),bed,sus,derivative;\n"
        "  double shear=hypot(u,v)*.41/log1p(h*double(float(.368))/z0[i]);\n"
        "  theta=shear*shear/(9.81*law_params[14]*d);\n"
        "  if(law_params[0]==11||law_params[0]==12){double gx,gy,own_theta;\n"
        "   double physical_speed=hypot(u,v),factor=physical_speed/(physical_speed+double(1.e-6f));\n"
        "   ue*=factor;ve*=factor;speed=hypot(ue,ve);\n"
        "   river_current_vector_law(law_params,ue,ve,fmax(.01,cq[i]),d,bed,sus,gx,gy,own_theta);\n"
        "   derivative=hypot(gx,gy);theta=shear*shear/(9.81*law_params[14]*d);\n"
        "  }else river_scalar_law(law_params,speed,c[i],fmax(.01,cq[i]),d,d90[i],hide[i],bed,sus,derivative,shear);\n"
        "  if(!isfinite(bed)||!isfinite(sus)||!isfinite(derivative)){status[i]=2;continue;}\n"
        "  double mag=bed_factor*law_params[13]*bed;")
    source = replace_once(source, "double bx=u+alpha*v,by=v-alpha*u,bs=hypot(bx,by);",
        "bool vector_owned=law_params[0]==11||law_params[0]==12;\n"
        "  if(vector_owned)alpha=0.;\n"
        "  double bx=vector_owned?ue:u+alpha*v,by=vector_owned?ve:v-alpha*u,bs=hypot(bx,by);")
    source = replace_once(source, "double*next,double*delta,double*diagnostic){",
        "double*next,double*delta,double*diagnostic,const double*d90,const double*hide,const double*law_params){")
    source = replace_once(source,
        "double cap=sus_factor*2650.*2.5*d*sqrt(9.81*1.65*d)*theta*theta*fmax(.9*theta-.035,0.);",
        "double bed,qs,derivative;\n"
        "   if(law[0]==11||law[0]==12){double gx,gy,own_theta,factor=speed/(speed+double(1.e-6f));\n"
        "    river_current_vector_law(law,h>dry?factor*cq[n+i]/cq[i]:0.,h>dry?factor*cq[2*n+i]/cq[i]:0.,hs,d,bed,qs,gx,gy,own_theta);\n"
        "   }else river_scalar_law(law,ve,c[i],hs,d,d90[i],hide[a],bed,qs,derivative,shear);\n"
        "   double cap=sus_factor*law[13]*qs;")
    source = replace_once(source,
        "for(int k=0;k<ng;++k){int a=k*n+i;double d=diam[k],ws;",
        f"for(int k=0;k<ng;++k){{const double*law=law_params+{WIRE_SIZE}*k;\n"
        "   int a=k*n+i;double d=diam[k],ws;")
    # Only SettleFrm2 and its Shields diagnostic still contain this constant;
    # split_capacity and river_scalar_law above already use explicit properties.
    if source.count("1.65") != 4:
        raise ValueError("The audited settling/Shear density sites changed")
    source = source.replace("1.65", "law[14]")
    mixture = (HERE/"river_mpm.h").read_text()
    source += "\n"+mixture[mixture.index("ENTRY void mixture"):mixture.index("ENTRY void mpm")]
    # Same direction, domain checks and units as the other scalar capacities.
    # Specialize only the physical forcing required by the intrinsic laws.
    cell = source[source.index("ENTRY void split_capacity"):source.index("ENTRY void sus_faces")]
    cell = replace_once(cell, "ENTRY void split_capacity", "ENTRY void intrinsic_capacity")
    cell = replace_once(cell, "const double*z0,double*out", "const double*z0,const double*grain,const double*hydraulic_c,double*out")
    cell = replace_once(cell, "status[i]=valid?0:1;",
        "valid=valid&&isfinite(hydraulic_c[i])&&hydraulic_c[i]>0.&&isfinite(grain[i])&&grain[i]>=0.\n"
        "   &&isfinite(grain[n+i])&&grain[n+i]>=0.&&isfinite(grain[2*n+i])&&grain[2*n+i]>=0.&&grain[2*n+i]<=1.;\n"
        "  status[i]=valid?0:1;")
    cell = replace_once(cell,
        "}else river_scalar_law(law_params,speed,c[i],fmax(.01,cq[i]),d,d90[i],hide[i],bed,sus,derivative,shear);",
        "}else {double tau=law_params[13]/(1.+law_params[14])*9.81*(u*u+v*v)/(hydraulic_c[i]*hydraulic_c[i]);\n"
        "   double elasticity=2.,cdr,cds;\n"
        "   if(law_params[43]&&river_skin_stress(hypot(u,v),fmax(.01,h),law_params[13]/(1.+law_params[14]),\n"
        "     law_params[44],law_params[45],tau,elasticity,cdr,cds)){status[i]=4;continue;}\n"
        "   river_scalar_law(law_params,speed,c[i],fmax(.01,cq[i]),d,d90[i],hide[i],bed,sus,derivative,shear,\n"
        "    grain[i],grain[n+i],grain[2*n+i],tau,elasticity);}")
    cell = replace_once(cell, "double mag=bed_factor*law_params[13]*bed;",
        "// FLOW ust2 for adjust_bedload is characteristic shear, not TAUB.\n"
        "  // TAUB above belongs only to the intrinsic capacity law.\n"
        "  theta=shear*shear/(9.81*law_params[14]*d);\n"
        "  double mag=bed_factor*law_params[13]*bed;")
    source += "\n"+cell
    source += "\n"+(HERE/"river_intrinsic_grains.h").read_text()
    source += """
ENTRY void scalar_probe(int n,double d,const double*speed,const double*c,const double*h,
 const double*d90,const double*hide,const double*law_params,double*out){
 LOOP(n){double b,s,der;river_scalar_law(law_params,speed[i],c[i],h[i],d,d90[i],hide[i],b,s,der,sqrt(9.81)*speed[i]/c[i]);
  out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void scalar_forcing_probe(int n,double d,const double*speed,const double*c,const double*h,
 const double*d90,const double*hide,const double*shear,const double*law_params,double*out){
 LOOP(n){double b,s,der;river_scalar_law(law_params,speed[i],c[i],h[i],d,d90[i],hide[i],b,s,der,shear[i]);
  out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void intrinsic_forcing_probe(int n,double d,const double*speed,const double*c,
 const double*grain,const double*taub,const double*law_params,double*out){
 LOOP(n){double b,s,der;river_scalar_law(law_params,speed[i],c[i],.1,d,d,1.,b,s,der,0.,
  grain[i],grain[n+i],grain[2*n+i],taub[i]);out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void intrinsic_skin_probe(int n,double d,const double*speed,const double*c,
 const double*grain,const double*physical_speed,const double*depth,const double*law_params,double*out){
 LOOP(n){double b,s,der,tau,elasticity,cdr,cds;
  if(river_skin_stress(physical_speed[i],depth[i],law_params[13]/(1.+law_params[14]),
    law_params[44],law_params[45],tau,elasticity,cdr,cds)){
   out[i]=nan("");out[n+i]=nan("");out[2*n+i]=nan("");continue;}
  river_scalar_law(law_params,speed[i],c[i],depth[i],d,d,1.,b,s,der,0.,
   grain[i],grain[n+i],grain[2*n+i],tau,elasticity);out[i]=b;out[n+i]=s;out[2*n+i]=der;}
}
ENTRY void vector_forcing_probe(int n,double d,const double*u,const double*v,const double*h,
 const double*law_params,double*out){
 LOOP(n){double b,s,gx,gy,theta;river_current_vector_law(law_params,u[i],v[i],h[i],d,b,s,gx,gy,theta);
  double speed=hypot(u[i],v[i]);out[i]=speed>0.?b*u[i]/speed:0.;out[n+i]=speed>0.?b*v[i]/speed:0.;
  out[2*n+i]=speed>0.?s*u[i]/speed:0.;out[3*n+i]=speed>0.?s*v[i]/speed:0.;
  out[4*n+i]=gx;out[5*n+i]=gy;out[6*n+i]=theta;}
}
ENTRY void vector_wave_probe(int n,double d,const double*u,const double*v,const double*h,
 const double*angle,const double*k,const double*law_params,double*out){
 LOOP(n){double nx=cos(angle[i]),ny=sin(angle[i]),b,s,gx,gy,t,speed=hypot(u[i],v[i]);
  double kv=k[i]*speed/(speed+double(1.e-6f));
  river_current_vector_law(law_params,kv*(u[i]*nx-v[i]*ny),kv*(u[i]*ny+v[i]*nx),h[i],d,b,s,gx,gy,t);
  double a=0.,c=0.,e=0.;
  if(u[i]*u[i]+v[i]*v[i]>0.)river_vector_bed_row(1.,h[i],u[i],v[i],nx,ny,k[i],b,gx,gy,a,c,e);
  out[i]=a;out[n+i]=c;out[2*n+i]=e;}
}
"""
    from .river_vr1993 import embedded_source
    source = replace_once(source, "ENTRY void split_capacity", embedded_source()+"\nENTRY void split_capacity")
    # Optional native stress output; never put TAURAT in the spiral-angle row.
    for last in ("const double*z0", "const double*hydraulic_c"):
        source = replace_once(source, last+",double*out,int*status){",
                              last+",double*out,int*status,double*taurat){")
    source = source.replace("status[i]=valid?0:1;if(!valid)continue;",
                            "status[i]=valid?0:1;taurat[i]=0.;if(!valid)continue;")
    source = source.replace("if(law_params[0]==11||law_params[0]==12){double gx,gy,own_theta;",
        "if(law_params[0]==-1){double vr[12];\n"
        "   if(river_vr_current(law_params,h,u,v,d,d90[i],z0[i],1.,d,0.,vr)){status[i]=3;continue;}\n"
        "   bed=vr[11]/law_params[13];sus=0.;derivative=vr[10]/law_params[13];\n"
        "   taurat[i]=vr[3];theta=shear*shear/(9.81*law_params[14]*d);ue=u;ve=v;speed=hypot(u,v);\n"
        "  }else if(law_params[0]==11||law_params[0]==12){double gx,gy,own_theta;")
    source = source.replace("bool vector_owned=law_params[0]==11||law_params[0]==12;",
                            "bool vector_owned=law_params[0]==-1||law_params[0]==11||law_params[0]==12;")
    original_exchange = source[source.index("ENTRY void sus_exchange"):source.index("ENTRY void mixture")]
    source = replace_once(source, "const double*d90,const double*hide,const double*law_params){",
        "const double*d90,const double*hide,const double*law_params,const double*dss,double*dss_next){")
    source = replace_once(source, "int a=k*n+i;double d=diam[k],ws;",
                           "int a=k*n+i;double d=diam[k],ws;dss_next[a]=dss[a];\n"
                           "   double transport_d=d; if(law[0]==-1)d=dss[a];")
    source = replace_once(source, "ws*=hinder;double wsl=", "ws*=hinder;d=transport_d;double wsl=")
    source = replace_once(source, "}else river_scalar_law(law,ve,c[i],hs,d,d90[i],hide[a],bed,qs,derivative,shear);",
        "}else if(law[0]==-1){bed=0.;qs=0.;derivative=0.;}\n"
        "   else river_scalar_law(law,ve,c[i],hs,d,d90[i],hide[a],bed,qs,derivative,shear);")
    source = replace_once(source, "if(h>sedthr){if(speed>1.e-6",
        "if(law[0]==-1&&h>sedthr){double vr[12];\n"
        "    int bad=river_vr_current(law,h,q[n+i]/h,q[2*n+i]/h,d,d90[i],z0[i],frac[a],dss[a],ws,vr);\n"
        "    if(bad){next[a]=nan(\"\");delta[a]=nan(\"\");continue;}\n"
        "    ceq=sus_factor*vr[2];dss_next[a]=vr[6];\n"
        "   }\n"
        "   if(h>sedthr){if(speed>1.e-6")
    source = replace_once(source, "ENTRY void sus_exchange", "ENTRY void vr_sus_exchange")
    # Non-VR operators retain their existing exchange call contract and body.
    # A VR wire passed to that entry is an error, never zero concentration.
    original_exchange = replace_once(original_exchange, "int a=k*n+i;double d=diam[k],ws;",
        "int a=k*n+i;if(law[0]==-1){next[a]=nan(\"\");delta[a]=nan(\"\");continue;}double d=diam[k],ws;")
    from .river_total_load import capacity_source as total_source
    # A separate forcing adapter, not a replacement of FLOW effective inputs.
    from .river_smart_jaggi import capacity_entry
    source += "\n" + capacity_entry(source, replace_once)
    return total_source(source + "\n" + original_exchange,replace_once)+"\n"+(
        HERE/"river_settling_only.h").read_text()


def wave_source(*, intrinsic=False, vector=False, vr1993=False, total=False, smart=False, physical_drag=False):
    if (smart or physical_drag) and (intrinsic or vector or vr1993 or total) or (smart and physical_drag):
        raise ValueError("Smart-Jaeggi single-grain bedload cannot share a different capacity adapter")
    if intrinsic and vector:
        raise ValueError("An intrinsic scalar and a vector capacity cannot own the same flux")
    from .river_mpm import wave_source as mpm_wave_source
    source = mpm_wave_source()
    source = replace_once(source, "cwork=fmax(20.,cmean)", "cwork=fmax(1.e-6,cmean)")
    source = replace_once(source, "const double*hide,const double*d90,const double*rough,double diameter){",
        "const double*law_params,const double*packing,const double*hide,const double*d90,const double*rough,double diameter){")
    source = replace_once(source, "int nf,int n,double acceleration,const double*q",
                          "int nf,int n,double acceleration_base,const double*q")
    source = replace_once(source, "diagnostic[k*nf+i]=0.;if(j<0)continue;",
        "diagnostic[k*nf+i]=0.;if(j<0)continue;\n"
        "  // Freeze the accepted active-layer packing at this face, just like\n"
        "  // roughness/hiding. This is not a full mixture/porosity Jacobian.\n"
        "  double acceleration=acceleration_base/(.5*(packing[o]+packing[j]));")
    start = source.index("  double dm90=")
    end = source.index("  if(!(h>0.)", start)
    source = source[:start]+"""  double dm90=.5*(d90[o]+d90[j]);
  double mag=0.,sus=0.,derivative=0.;
  double shear=law_params[0]==13?sqrt(sp2)*.41/log1p(h*double(float(.368))/zrough):0.;
  river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear);
  mag*=law_params[12];derivative*=law_params[12];
"""+source[end:]
    source = replace_once(source,
        "double radial=acceleration*3.*pref*sqrt(excess)*beta*sqrt(sp2)/h;",
        "double radial=acceleration*derivative*velocity_factor/h;")
    source = replace_once(source, "qs[side]=acceleration*(x*nx+y*ny)/2650.;",
                          "qs[side]=acceleration*(x*nx+y*ny)/law_params[13];")
    source = replace_once(source, "double next=(.5*(qs[0]+qs[1]-diss))*2650./acceleration;",
                          "double next=(.5*(qs[0]+qs[1]-diss))*law_params[13]/acceleration;")
    if intrinsic:
        source = replace_once(source, "const double*law_params,const double*packing,",
            "const double*grain,const double*hydraulic_c,const double*law_params,const double*packing,")
        source = replace_once(source,
            "river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear);",
            "double tau=law_params[13]/(1.+law_params[14])*9.81*sp2*\n"
            "   .5*(1./(hydraulic_c[o]*hydraulic_c[o])+1./(hydraulic_c[j]*hydraulic_c[j]));\n"
            "  double elasticity=2.,cdr,cds;\n"
            "  if(law_params[43]&&river_skin_stress(sqrt(sp2),fmax(.01,h),law_params[13]/(1.+law_params[14]),\n"
            "    law_params[44],law_params[45],tau,elasticity,cdr,cds)){diagnostic[2*nf+i]=1.;continue;}\n"
            "  river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear,\n"
            "   .5*(grain[o]+grain[j]),.5*(grain[n+o]+grain[n+j]),.5*(grain[2*n+o]+grain[2*n+j]),tau,elasticity);")
    if vr1993:
        source = replace_once(source,
            "river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear);",
            "double vr[12];\n"
            "  if(river_vr_current(law_params,h,u,v,diameter,dm90,zrough,1.,diameter,0.,vr)){\n"
            "   diagnostic[2*nf+i]=1.;continue;}\n"
            "  mag=vr[11]/law_params[13];derivative=vr[10]/law_params[13];velocity_factor=1.;")
    elif vector:
        source = replace_once(source,
            "river_scalar_law(law_params,ve,cwork,fmax(.01,h),diameter,dm90,.5*(hide[o]+hide[j]),mag,sus,derivative,shear);",
            "double gx,gy,own_theta,vector_factor=velocity_factor*sqrt(sp2)/(sqrt(sp2)+double(1.e-6f));\n"
            "  river_current_vector_law(law_params,vector_factor*(u*nx-v*ny),vector_factor*(u*ny+v*nx),\n"
            "   fmax(.01,h),diameter,mag,sus,gx,gy,own_theta);\n"
            "  gx*=law_params[12];gy*=law_params[12];")
        source = replace_once(source,
            "double radial=acceleration*derivative*velocity_factor/h;\n"
            "  der[0]=-radial*u;der[1]=factor+(radial-factor)*u*u/sp2;\n"
            "  der[2]=(radial-factor)*u*v/sp2;",
            "river_vector_bed_row(acceleration,h,u,v,nx,ny,velocity_factor,mag,gx,gy,der[0],der[1],der[2]);")
    from .river_vr1993 import embedded_source
    source=replace_once(source, "ENTRY void wave_state_gradients",
                        (HERE/"river_scalar_law.h").read_text()+"\n"+
                        (HERE/"river_skin_shear.h").read_text()+"\n"+
                        (HERE/"river_vector_law.h").read_text()+"\n"+
                        embedded_source()+"\nENTRY void wave_state_gradients")
    if total:
        from .river_total_load import wave_source as total_wave
        source=total_wave(source,replace_once,vector=vector,vr1993=vr1993)
    from .bed_wave_flux import source as bed_wave_source
    if smart or physical_drag:
        from .river_smart_jaggi import wave_adapter
        source = wave_adapter(source, replace_once,depth_power=.6 if smart else 0.)
    return bed_wave_source(source)


class ResolvedRiverCapacity(ClassicFlowEH):
    def __init__(self, backend, *, diameter, chezy, capacity, folder, threads=4, bed_factor=1.,
                 solid_density=2650., water_density=1000., viscosity=1.e-6, reference=None, transport="bedload",
                 skin_shear=None):
        if transport not in ("bedload","split","total_load"):
            raise ValueError("Explicit bedload, split or total_load transport required")
        if capacity['method'] in ('capacity.smart_jaggi','capacity.ancey_recking_2023','capacity.recking_2010','capacity.camenen_larson_2005','capacity.wong_parker_2006') and transport!='bedload':
            raise ValueError('Selected published physical-drag law supports bedload only')
        from .river_total_load import ZERO_SUSPENDED
        self.total_load=transport=="total_load"
        self.algebraic_suspension=self.total_load and METHODS[capacity["method"]] not in ZERO_SUSPENDED
        self.surface_fraction=self.suspended_diameter=None
        wire = parameter_wire(capacity, bed_factor, solid_density=solid_density, water_density=water_density,
                              viscosity=viscosity, reference=reference, skin_shear=skin_shear,diameter=diameter)
        super().__init__(backend, diameter=diameter, chezy=chezy, threads=threads)
        self.solid_density, self.water_density = float(solid_density), float(water_density)
        self.bed_factor = float(bed_factor)
        self.d90 = self.hiding = self.z0 = None
        # Every law exports characteristic stress for adjust_bedload, even
        # when its own capacity formula does not consume RP_USTAR.
        self.needs_characteristic_shear = True
        self.needs_intrinsic_mixture = METHODS[capacity["method"]] in (16, 17, 18)
        self.owns_vector_direction = METHODS[capacity["method"]] in (-1, 11, 12)
        self.is_vr1993 = METHODS[capacity["method"]] == -1
        self.is_smart_jaggi = METHODS[capacity["method"]] == 101
        self.is_ancey_recking = METHODS[capacity["method"]] == 102
        self.is_recking_2010 = METHODS[capacity["method"]] == 103
        self.is_camenen_larson = METHODS[capacity["method"]] == 104
        self.is_wong_parker = METHODS[capacity["method"]] == 105
        self.uses_physical_drag = self.is_smart_jaggi or self.is_ancey_recking or self.is_recking_2010 or self.is_camenen_larson or self.is_wong_parker
        self.taurat = None
        self.grain_statistics = self.hydraulic_chezy = None
        self.wire = self.xp.asarray(wire)
        from braids_core.windows_paths import filesystem_path
        path = filesystem_path(folder)/"resolved_capacity.h"
        path.parent.mkdir(parents=True, exist_ok=True)
        code = capacity_source()
        if path.exists() and path.read_text() != code:
            raise ValueError("Capacity source differs in existing owner folder")
        if not path.exists():
            path.write_text(code)
        self.engine = WaveEngine(backend, threads, source=path, names=NAMES)
        self.identity = dict(method=capacity["method"], parameters=deepcopy(capacity["parameters"]),
            grain_sizes_m=deepcopy(capacity.get('grain_sizes_m', {})),
            backend=backend, diameter_m=self.d, chezy=self.c, bed_factor=self.bed_factor,
            solid_density_kg_m3=self.solid_density, water_density_kg_m3=self.water_density,
            relative_density=float(wire[14]),
            capacity_viscosity_m2_s=float(wire[15]),
            sediment_minimum_chezy=1.e-6,
            slope_stress="eqtran USTAR squared; scalar spiral norm applied once at true faces; never skin or formula Shields",
            characteristic_shear=("physical velocity and ZUMOD=0.368f*h log law; slope forcing separate from total drag and floored Chezy"
                                  if self.needs_characteristic_shear else "not consumed by capacity"),
            mud_fraction=0.,
            intrinsic_mixture=("FLOW compdiam Dg/DgSD; compsandfrac d<2mm; stock composition; no external hiding"
                               if self.needs_intrinsic_mixture else "not consumed"),
            intrinsic_stress=("physical hydraulic velocity and total drag; NOT sediment Chezy"
                              if self.needs_intrinsic_mixture else "not consumed"),
            runtime=self.engine.identity, wrapper_sha256=digest(__file__),
            law_sha256=digest(HERE/"river_scalar_law.h"),
            skin_shear_parameters=deepcopy(skin_shear),
            skin_shear_source_sha256=digest(HERE/"river_skin_shear.h"),
            vector_law_sha256=digest(HERE/"river_vector_law.h"),
            direction=("capacity-owned current-only vector; no external spiral rotation"
                       if self.owns_vector_direction else "FLOW scalar external spiral rotation"),
            vector_velocity=("FLOW erosed EFU/EFV: effective magnitude times physical direction with UMOD+1e-6f denominator"
                             if self.owns_vector_direction else "not consumed"),
            channels="separate bed and suspended capacities; source partition retained",
            derivative=("frozen coefficients, global vector magnitude gradient including FLOW12 angular branch"
                        if self.owns_vector_direction else "frozen roughness/hiding, same scalar bed law; physical steering unchanged"),
            depth="FLOW erosed working-depth floor 0.01 m", slope="Koch-Flokstra unless explicitly off")
        if skin_shear is not None:
            self.identity.update(
                intrinsic_stress="FLOW compbsskin current-only noncohesive RP_TAUB; physical speed; explicit sand roughness/viscosity",
                derivative="same skin stress and Reynolds-dependent elasticity; hydraulic drag and slope characteristic shear unchanged")
        if self.uses_physical_drag:
            self.identity.update(source="Smart-Jaeggi; BASEHPC 4.0.2 Eqs 1.29-1.31 (not FLOW)",
                adapter_sha256=digest(HERE/"river_smart_jaggi.py"),
                depth="physical water depth; no 1 cm working-depth floor",
                velocity="physical water velocity, not FLOW sediment effective velocity",
                capacity_stress="physical hydraulic total drag; J=U^2/(C_hyd^2*h)",
                grading="one bulk material; measured Dm, D30 and D90; grading multiplier (D90/D30)^0.2",
                applicability="energy slope 0.005..0.2; unclamped extrapolation elsewhere",
                derivative="analytic physical-speed derivative plus -0.6*qb/h depth partial; frozen hydraulic drag")
        if self.is_ancey_recking:
            self.identity.update(source='Ancey & Recking (2023), DOI 10.1016/j.earscirev.2023.104571, Eq. 42',
                capacity_stress='physical hydraulic total drag; theta=U^2/(C_hyd^2*Delta*d50)',
                grading='one grain: surface d50=diameter; no external hiding or form-drag multiplier',
                applicability='laboratory master curve; natural-river discrepancies described in source section 3.6',
                derivative='analytic physical-speed derivative; zero explicit depth partial; frozen hydraulic drag')
        if self.is_recking_2010:
            self.identity.update(source='Recking (2010), DOI 10.1029/2009WR008007, Eqs. 4 and 12-14',
                capacity_stress='local physical hydraulic stress; does not replace hydraulic resistance with reach Eq.1 inversion',
                grading='one bulk material; measured D84/D50 in threshold and low-transport coefficient; no external hiding',
                reach_slope=capacity['parameters']['ReachSlope'],
                applicability='main surface-based Eq12; source field range S=0.0002..0.08, D84=0.9..558 mm',
                breakpoint='exact Eq12 intersection, Eq13 before rounding; no artificial capacity discontinuity',
                derivative='piecewise analytic physical-speed derivative; zero explicit depth partial; fixed reach slope and drag')
        if self.is_camenen_larson:
            self.identity.update(source='Camenen & Larson (2005), DOI 10.1016/j.ecss.2004.10.019; authors CR-07-1 Eqs49,81',
                capacity_stress='physical hydraulic total drag; theta=U^2/(C_hyd^2*Delta*D50)',
                grading='one grain: median D50; no external hiding or MPM form-drag multiplier',
                critical_shields=float(wire[6]),critical_method='Soulsby & Whitehouse (1997), CR-07-1 Eq49',
                applicability='noncohesive steady-current bedload; no waves, phase lag, mixture or suspension',
                derivative='analytic physical-speed derivative; zero explicit depth partial; frozen hydraulic drag')
        if self.is_wong_parker:
            self.identity.update(source='Wong & Parker (2006), DOI 10.1061/(ASCE)0733-9429(2006)132:11(1159); authors preprint Eq24',
                capacity_stress='local hydraulic bed shear; theta=U^2/(C_hyd^2*Delta*Dm); no MPM form drag',
                grading='one uniform grain: arithmetic mean Dm; no external hiding or inferred percentile',
                critical_shields=.0495,coefficient=3.97,exponent=1.5,
                applicability='lower-regime plane-bed equilibrium uniform sediment; no suspension or mixture',
                derivative='analytic physical-speed derivative; zero explicit depth partial; frozen hydraulic bed drag')
        if self.is_vr1993:
            self.identity.update(reference_parameters=deepcopy(reference),
                vr1993_source_sha256=digest(HERE/"river_vr1993.h"),
                vector_velocity="physical velocity; no effective-magnitude conversion",
                derivative="same physical-speed bed law; frozen depth/roughness/material",
                reference_concentration="native fraction-before-cutoff; mass concentration already fraction-owned")
        self.identity.update(transport=transport,total_load_source_sha256=digest(HERE/"river_total_load.h"),
            total_load_binding_sha256=digest(HERE/"river_total_load.py"))
        if self.total_load:
            self.identity.update(channels="algebraic bed + equilibrium suspended vector; no resident suspension",
                total_suspended_factor=1.,
                direction="bed keeps selected steering; equilibrium suspension uses FLOW EFUMN/EFVMN",
                derivative="sum of frozen bed principal part and same-law symmetric radial suspended derivative")
            if self.is_vr1993:
                self.identity.update(algebraic_profile="FLOW non-suspfrac: zero fall speed, sigmol=1, nominal DSS",
                    inactive_reference_inputs=["molecular_schmidt","suspended_diameter_m","IopSus state evolution"])

    def evaluate(self, q, hi, *, capacity_q=None, chezy=None):
        xp = self.xp
        cq = q if capacity_q is None else capacity_q
        n = q.shape[1]
        c = xp.full(n, self.c) if chezy is None else chezy
        for value, shape in ((q, (3, n)), (cq, (3, n)), (c, (n,)), (hi, (n,)),
                             (self.d90, (n,)), (self.hiding, (n,))):
            if not isinstance(value, xp.ndarray) or value.shape != shape or value.dtype != xp.float64 or not value.flags.c_contiguous:
                raise ValueError("Resident contiguous FP64 scalar-capacity inputs required")
        out, status = xp.zeros((6, n)), xp.empty(n, dtype=xp.int32)
        if self.taurat is None or self.taurat.shape != (n,):
            self.taurat = xp.zeros(n)
        z0 = self.z0
        if z0 is None:
            raise ValueError("Capacity/slope forcing requires explicit characteristic roughness, not floored Chezy")
        elif (not isinstance(z0, xp.ndarray) or z0.shape != (n,) or z0.dtype != xp.float64
              or not z0.flags.c_contiguous):
            raise ValueError("Resident contiguous FP64 characteristic roughness required")
        tail = []
        if self.uses_physical_drag:
            value = self.hydraulic_chezy
            if (not isinstance(value, xp.ndarray) or value.shape != (n,) or value.dtype != xp.float64
                    or not value.flags.c_contiguous or not bool(xp.all(xp.isfinite(value) & (value > 0.)))):
                raise ValueError("Selected physical-drag law requires explicit resident physical hydraulic Chezy")
            tail = [value]
        if self.needs_intrinsic_mixture:
            for value, shape in ((self.grain_statistics, (3, n)), (self.hydraulic_chezy, (n,))):
                if (not isinstance(value, xp.ndarray) or value.shape != shape or value.dtype != xp.float64
                        or not value.flags.c_contiguous):
                    raise ValueError("Intrinsic capacity requires explicit resident mixture and hydraulic drag")
            tail = [self.grain_statistics, self.hydraulic_chezy]
        if self.algebraic_suspension:
            for value in (self.surface_fraction,self.suspended_diameter):
                if (not isinstance(value,xp.ndarray) or value.shape!=(n,) or value.dtype!=xp.float64
                        or not value.flags.c_contiguous):
                    raise ValueError("Algebraic total load requires explicit resident fraction and diameter state")
        self.engine.launch("total_capacity" if self.algebraic_suspension else
                           ("smart_capacity" if self.uses_physical_drag else
                            "intrinsic_capacity" if self.needs_intrinsic_mixture else "split_capacity"),
                           n, n, self.d, self.bed_factor, q, cq, c, hi, self.d90, self.hiding,
                           self.wire, z0, *tail, out, status, self.taurat,
                           *((self.surface_fraction,self.suspended_diameter) if self.algebraic_suspension else ()))
        if bool(xp.any(status)):
            raise ValueError("Invalid scalar capacity forcing or nonfinite formula value")
        return out
