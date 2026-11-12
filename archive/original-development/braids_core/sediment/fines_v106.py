"""Bounded Q9 cohesive/fine-sediment transaction (v106).

The erosion, settling, flocculation and fluff branches below translate the
locked Delft3D4 FLOW 2DH equations.  Consolidation selector/state paths are
bounded, mass-preserving implementations pending focused physical validation.
The transaction owns water/bed/fluff exchange once, while the hydraulic and
horizontal suspended-transport owners remain outside this file.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

SCOPE = "braids-q9-conditional-fines-v106"
FINE_TYPES = (1, 2)                 # FLOW mud-like silt/clay classes
CAPACITY = ("partheniades_krone_ratio", "partheniades_krone_difference")
SETTLING = ("function_conditions", "manning_dyer", "chassagne_safar")
FLOCCULATION = ("none", "manning_dyer", "chassagne_safar", "verney_etal")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FineMethodsV106:
    capacity: str = "partheniades_krone_ratio"
    erodibility_mode: int = 0
    settling: str = "function_conditions"
    flocculation: str = "none"
    consolidation_mode: int = 0
    fluff_type: int = 0
    fluff_burial_mode: int = 0
    neumann_mud: bool = False
    integrator: str = "implicit_fwave_midpoint"

    def __post_init__(self):
        if self.capacity not in CAPACITY or self.settling not in SETTLING:
            raise ValueError("Unknown FLOW fine-sediment capacity/settling method")
        if self.flocculation not in FLOCCULATION:
            raise ValueError("Unknown FLOW flocculation method")
        if self.erodibility_mode not in range(6) or self.consolidation_mode not in range(5):
            raise ValueError("FLOW erodibility/consolidation selector is outside its source range")
        if self.fluff_type not in range(3) or self.fluff_burial_mode not in range(3):
            raise ValueError("FLOW fluff selector is outside its source range")
        if self.fluff_type != 1 and self.fluff_burial_mode:
            raise ValueError("FLOW fluff burial is active only for FluffLayer type 1")
        if self.integrator != "implicit_fwave_midpoint":
            raise ValueError("Q9 GPU composition requires implicit_fwave_midpoint")

    @property
    def method_ids(self):
        return (
            f"capacity.{self.capacity}", f"cohesive_erodibility.mode_{self.erodibility_mode}",
            f"settling.{self.settling}", f"flocculation.{self.flocculation}",
            f"consolidation.mode_{self.consolidation_mode}",
            ("fluff_layer.off" if self.fluff_type == 0 else f"fluff_layer.type_{self.fluff_type}"),
            *(() if not self.fluff_burial_mode else (f"fluff_burial.mode_{self.fluff_burial_mode}",)),
            "suspended_boundary.neumann_mud" if self.neumann_mud else "suspended_boundary.prescribed_mud",
        )

    @property
    def fingerprint(self):
        return digest({"scope": SCOPE, **asdict(self), "method_ids": self.method_ids})


@dataclass(frozen=True)
class FineParametersV106:
    erosion_power: float = 1.0
    pmcrit: float = 0.4
    old_mud_fraction: bool = False
    deposition_efficiency: float = -1.0
    slope_critical_min_pa: float = 0.1
    slope_beta: float = 2.0
    wet_slope: float = 10.0
    # getbedprop defaults in locked bedcomposition_module.f90
    activity_a: float = 2.67
    alpha: float = 0.7
    alpha_me: float = 1.0
    alpha_mix: float = 0.2205
    alpha_lehir: float = 1.0
    alpha_winterwerp: float = 10.0
    alpha1: float = 1.0
    alpha2: float = 2.0
    beta: float = 0.2
    beta_mix: float = 0.9125
    c0: float = 0.07
    d50_supply_m: float = 3.0e-5
    rho_min: float = 200.0
    rho_star: float = 1200.0
    rho_max: float = 1600.0
    taucr_min1_pa: float = 0.2
    taucr_max_pa: float = 1.5
    taucr_min2_pa: float = 0.2
    nf: float = 2.0
    ksigma_pa: float = 1.0
    permeability_m_s: float = 1.0e-9
    ky: float = 1.0
    svfrac0: float = 500.0 / 2650.0
    svmax: float = 0.6
    confac: float = 1.0
    consolidation_time_s: float = 1209600.0
    crmsec: float = 3.0e-4
    min_porosity_mud: float = 0.4
    min_porosity_sand: float = 0.3
    fluff_critical_pa: float = 0.1
    fluff_erosion_zero: float = 1.0e-5
    fluff_erosion_first: float = 1.0e-4
    fluff_dep_fraction: float = 0.5
    fluff_burial_zero: float = 1.0e-6
    fluff_burial_first: float = 1.0e-4
    fluff_acalbur0: float = 1.0
    fluff_permeability_m_s: float = 1.0e-9
    fluff_density_kg_m3: float = 40.0
    water_density_kg_m3: float = 1000.0
    gravity_m_s2: float = 9.81
    viscosity_m2_s: float = 1.0e-6
    flocculation_time_s: float = 300.0
    breakup_time_s: float = 120.0
    micro_diameter_m: float = 2.0e-5
    macro_shear_velocity_m_s: float = 0.01

    def __post_init__(self):
        vals=np.asarray([v for v in asdict(self).values() if not isinstance(v,bool)],float)
        if not np.isfinite(vals).all(): raise ValueError("Finite Q9 controls required")
        if self.erosion_power <= 0 or not 0 <= self.pmcrit <= 1 or self.wet_slope <= 0:
            raise ValueError("Invalid Q9 erosion/mixed-bed controls")
        if not (0 < self.rho_min < self.rho_star < self.rho_max):
            raise ValueError("MUSA density breakpoints must increase")
        if not 0 < self.svmax < 1 or not 0 <= self.fluff_dep_fraction <= 1:
            raise ValueError("Invalid Q9 consolidation/fluff control")
        if self.deposition_efficiency != -1 and not 0 <= self.deposition_efficiency <= 1:
            raise ValueError("Deposition efficiency must be -1 or a fraction")


def resolve_fine_methods_v106(fraction_types, **explicit):
    """Resolve source-compatible auxiliaries without exposing a Cartesian menu."""
    types=tuple(int(v) for v in fraction_types)
    if any(v not in (0,1,2) for v in types):raise ValueError("Unknown fine-fraction type")
    if not any(v in FINE_TYPES for v in types):return None
    choices=dict(explicit)
    floc=choices.get("flocculation","none")
    if "settling" not in choices:
        choices["settling"]={"manning_dyer":"manning_dyer",
            "chassagne_safar":"chassagne_safar"}.get(floc,"function_conditions")
    if choices.get("fluff_burial_mode",0) and "fluff_type" not in choices:choices["fluff_type"]=1
    return FineMethodsV106(**choices)


def _asnumpy(value):
    return value


class FineRuntimeV106:
    """Vector-parallel CPU implementation; CUDA subclasses replace ``xp``."""
    backend = "native_cpu_vector"
    xp = np

    def __init__(self, methods=FineMethodsV106(), parameters=FineParametersV106()):
        self.methods, self.parameters = methods, parameters
        self.identity=digest({"scope":SCOPE,"backend":self.backend,
            "methods":methods.fingerprint,"parameters":asdict(parameters)})

    def _host(self, value): return np.asarray(value)
    def _bool(self, value): return bool(value)

    def _array(self, value, shape, name, *, nonnegative=False, positive=False, dtype=None):
        xp=self.xp; a=xp.asarray(value,dtype=xp.float64 if dtype is None else dtype)
        a=xp.ascontiguousarray(xp.broadcast_to(a,shape).copy())
        if self._bool(xp.any(~xp.isfinite(a))) or (nonnegative and self._bool(xp.any(a<0))) or \
                (positive and self._bool(xp.any(a<=0))): raise ValueError(f"Invalid {name}")
        return a

    def erodibility(self, layers, thickness, solid_fraction, bulk_density, sediment_type,
                    *, water_density=None):
        """FLOW ``getbedprop`` critical stress and erosion multiplier."""
        xp=self.xp;p=self.parameters
        q=xp.asarray(layers,dtype=xp.float64); nl,ng,ny,nx=q.shape
        thick=self._array(thickness,(nl,ny,nx),"layer thickness",nonnegative=True)
        sv=self._array(solid_fraction,(nl,ny,nx),"solid fraction",positive=True)
        rho=self._array(bulk_density,(ng,),"solid density",positive=True)
        typ=xp.asarray(sediment_type,dtype=xp.int32)
        water=self._array(p.water_density_kg_m3 if water_density is None else water_density,(ny,nx),"water density",positive=True)
        top=q[0];safe_thick=xp.maximum(thick[0],1.e-15)
        phi=xp.where(thick[0][None]>0,top/rho[:,None,None]/safe_thick[None],0.)
        fine=(typ>0)[:,None,None]; phi_mud=xp.sum(xp.where(fine,phi,0.),axis=0)
        phi_sand=xp.sum(xp.where(~fine,phi,0.),axis=0)
        dry=xp.where(thick[0]>0,xp.sum(top,axis=0)/safe_thick,0.)
        por=1.-sv[0]; tcr=xp.ones((ny,nx),xp.float64); ero=xp.ones_like(tcr); mode=self.methods.erodibility_mode
        eps=1.e-15
        if mode==1:
            tcr=.0012*xp.maximum(phi_mud*phi_sand,0.)**1.2
        elif mode==2:
            term=xp.maximum(phi_mud,eps)**(por/max(1.-p.svfrac0,eps))
            tcr=p.alpha_lehir*term;ero=p.alpha_me/xp.maximum(term,eps)
        elif mode==3:
            por_ref=xp.maximum(por,eps) # source getporosity; supplied state is authoritative
            term=(phi_mud/max(p.c0,eps))*(por_ref/xp.maximum(por,eps))
            active=phi_mud>p.c0;tcr=xp.where(active,p.alpha_lehir*term,0.)
            ero=xp.where(active,p.alpha_me/xp.maximum(term,eps),0.)
        elif mode==4:
            pi=p.activity_a*xp.maximum(phi_mud-p.c0,0.)*100.
            low=phi_mud<=p.c0
            tcr=xp.where(low,p.alpha_mix*pi**p.beta_mix,p.alpha*pi**p.beta)
            mud_eff=xp.where(low,p.c0,phi_mud);sand_eff=xp.where(low,1.-mud_eff-por,phi_sand)
            nfd=2./(3.-p.nf);cu=p.ky*xp.maximum(mud_eff/xp.maximum(1.-sand_eff,eps),eps)**nfd
            cv=(nfd*p.ksigma_pa*p.permeability_m_s/p.gravity_m_s2)/water
            ero=cv*mud_eff**2*rho[0]/max(p.alpha_winterwerp*p.d50_supply_m,eps)/xp.maximum(cu,eps)
        elif mode==5:
            x1=xp.maximum((dry-p.rho_min)/(p.rho_star-p.rho_min),0.)**p.alpha1
            x2=xp.maximum((p.rho_max-dry)/(p.rho_max-p.rho_star),0.)**p.alpha2
            tcr=xp.where(dry<p.rho_min,p.taucr_min1_pa,
                xp.where(dry<p.rho_star,p.taucr_min1_pa*(1-x1)+p.taucr_max_pa*x1,
                xp.where(dry<p.rho_max,p.taucr_max_pa*x2+p.taucr_min2_pa*(1-x2),p.taucr_min2_pa)))
        return tcr,ero,phi_mud,phi_sand

    def settling_velocity(self, concentration, sediment_type, *, salinity=0., temperature_c=20.,
                          turbulent_shear_pa=0., dissipation_m2_s3=1.e-6, ws0=1.e-4,
                          wsm=1.e-3, salinity_max=10., temperature_coefficient=1.,
                          floc_critical_kg_m3=.01, concentration_power=1., csoil=1600.):
        xp=self.xp;p=self.parameters;c=xp.asarray(concentration,dtype=xp.float64)
        ng,ny,nx=c.shape;typ=xp.asarray(sediment_type,dtype=xp.int32)
        shape=(ny,nx);ctot=xp.sum(c,axis=0);cclay=xp.sum(xp.where((typ==2)[:,None,None],c,0.),axis=0)*1000.
        sal=self._array(salinity,shape,"salinity",nonnegative=True);temp=self._array(temperature_c,shape,"temperature")
        tau=self._array(turbulent_shear_pa,shape,"turbulent shear",nonnegative=True)
        eps=self._array(dissipation_m2_s3,shape,"dissipation",positive=True)
        out=xp.zeros_like(c); mode=self.methods.settling
        if mode=="function_conditions":
            w0=self._array(ws0,(ng,1,1),"WS0",positive=True);wm=self._array(wsm,(ng,1,1),"WSM",positive=True)
            sm=self._array(salinity_max,(ng,1,1),"SalMax",positive=True)
            tc=self._array(temperature_coefficient,(ng,1,1),"TCoefWS",positive=True)
            cf=self._array(floc_critical_kg_m3,(ng,1,1),"CFlocCr",positive=True)
            pw=self._array(concentration_power,(ng,1,1),"CPowWS",positive=True)
            enh=wm/w0; local=xp.where(sal[None]<sm,.5*w0*((1+enh)-(enh-1)*xp.cos(xp.pi*sal[None]/sm)),wm)
            local*=tc**(temp[None]-20.)*xp.minimum((ctot[None]/cf)**pw,1.)
            out=local
        elif mode=="manning_dyer":
            macro=xp.where(tau<.65,.644+.000471*cclay+9.36*tau-13.1*tau**2,
                xp.where(tau<1.45,3.96+.000346*cclay-4.38*tau+1.33*tau**2,
                         1.18+.000302*cclay-.491*tau+.057*tau**2))*.001
            micro=xp.where(tau<.52,.244+3.25*tau-3.71*tau**2,.65*xp.maximum(tau,1.e-15)**(-.541))*.001
            spm=xp.minimum(cclay,100000.);ratio=.815+.00318*spm-1.4e-7*spm**2;frac=ratio/(1+ratio)
            out[:]=micro+frac*(macro-micro)
        else:
            factor1=(eps*p.micro_diameter_m**4/p.viscosity_m2_s**3)**.166
            factor2=(cclay/xp.maximum(p.water_density_kg_m3,1.e-15))**.22044
            macro=.129*p.gravity_m_s2*factor1*factor2*xp.sqrt(p.viscosity_m2_s/eps)*xp.exp(-xp.sqrt(
                p.water_density_kg_m3*p.macro_shear_velocity_m_s**2/xp.maximum(tau,1.e-12))**.463)
            f1=(eps*1.e-5**4/p.viscosity_m2_s**3)**.39
            micro=.594*p.gravity_m_s2*f1*xp.sqrt(p.viscosity_m2_s/eps)*xp.exp(-xp.sqrt(
                p.water_density_kg_m3*.025**2/xp.maximum(tau,1.e-12))**.66)
            frac=xp.where(cclay<=1.,.1,xp.minimum(.1+.221*xp.log10(cclay),1.))
            out[:]=micro+frac*(macro-micro)
        hind=xp.maximum(0.,1.-xp.maximum(ctot,0.)/self._array(csoil,shape,"Csoil",positive=True))**5
        return xp.where((typ>0)[:,None,None],xp.maximum(out,0.)*hind[None],0.)

    def _flocculate(self, suspended, sediment_type, depth, dt_s, wet=None):
        xp=self.xp;m=self.methods.flocculation;typ=xp.asarray(sediment_type)
        transfer=xp.zeros_like(suspended)
        if m in ("none","verney_etal"): return suspended,transfer
        ids=self._host(xp.where(typ==2)[0]).tolist()
        if len(ids)<2 or len(ids)%2: raise ValueError("Active FLOW flocculation requires micro/macro clay pairs")
        total_clay=xp.sum(suspended[ids],axis=0)/depth*1000.
        if m=="manning_dyer":
            spm=xp.minimum(total_clay,100000.);ratio=.815+.00318*spm-1.4e-7*spm**2;macro=ratio/(1+ratio)
        else: macro=xp.where(total_clay<=1.,.1,xp.minimum(.1+.221*xp.log10(total_clay),1.))
        flocdt=1.-xp.exp(-dt_s/self.parameters.flocculation_time_s)
        breakdt=1.-xp.exp(-dt_s/self.parameters.breakup_time_s)
        out=suspended.copy()
        for j in range(0,len(ids),2):
            micro_i,macro_i=ids[j],ids[j+1];pair=out[micro_i]+out[macro_i];target=macro*pair
            fac=xp.where(target>out[macro_i],flocdt,breakdt);delta=fac*(target-out[macro_i])
            if wet is not None: delta=xp.where(wet,delta,0.)
            out[macro_i]+=delta;out[micro_i]-=delta;transfer[macro_i]+=delta;transfer[micro_i]-=delta
        return out,transfer

    def _consolidate(self, layers, solid_fraction, preload, deposit_time, bulk, typ, dt_s, time_s):
        xp=self.xp;p=self.parameters;mode=self.methods.consolidation_mode
        sv=solid_fraction.copy();pre=preload.copy();age=deposit_time.copy()
        if mode in (0,4): return sv,pre,age # mode 4 is a declared but uncalled FLOW branch
        solidvol=xp.sum(layers/bulk[None,:,None,None],axis=1);thick=solidvol/xp.maximum(sv,1.e-15)
        if mode==1:
            target=xp.minimum(p.svmax,xp.maximum(sv,p.svfrac0))
            rate=p.confac*p.permeability_m_s*p.ksigma_pa/xp.maximum(
                p.water_density_kg_m3*p.gravity_m_s2*thick**2,1.e-15)
            sv=xp.minimum(target,sv+(target-sv)*(1.-xp.exp(-rate*dt_s)))
        elif mode==2:
            mudvol=xp.sum(xp.where((typ>0)[None,:,None,None],layers/bulk[None,:,None,None],0.),axis=1)
            sandvol=xp.maximum(solidvol-mudvol,0.);nfd=2./(3.-p.nf)
            eqth=nfd/(nfd-1)*p.ksigma_pa/p.gravity_m_s2/xp.maximum(bulk[0]-p.water_density_kg_m3,1.)
            eqth*=xp.maximum(p.gravity_m_s2*(bulk[0]-p.water_density_kg_m3)*mudvol/p.ksigma_pa,1.e-15)**((nfd-1)/nfd)
            eqth=xp.maximum(sandvol+eqth,sandvol/(1.-p.min_porosity_sand))
            target=xp.minimum(p.svmax,solidvol/xp.maximum(eqth,1.e-15))
            fac=1.-xp.exp(-dt_s/max(p.consolidation_time_s,1.e-15));sv=xp.maximum(sv,sv+fac*(target-sv))
        else:
            load=xp.cumsum(xp.sum(layers,axis=1),axis=0)-.5*xp.sum(layers,axis=1)
            primary=load>pre;finevol=xp.sum(xp.where((typ>0)[None,:,None,None],layers/bulk[None,:,None,None],0.),axis=1)
            finefrac=finevol/xp.maximum(solidvol,1.e-15);critpor=finefrac*p.min_porosity_mud+(1-finefrac)*p.min_porosity_sand
            maxsv=xp.minimum(p.svmax,1.-critpor);increment=xp.maximum(load-pre,0.)
            compliance=.5*(finefrac*.001+(1-finefrac)*.01)
            trial_thick=thick*(1.-compliance*increment*p.gravity_m_s2)
            trial_sv=solidvol/xp.maximum(trial_thick,1.e-15)
            sv=xp.where(primary,xp.minimum(maxsv,xp.maximum(sv,trial_sv)),sv)
            elapsed=xp.maximum(time_s-age,1.);old=xp.maximum(elapsed-dt_s,1.)
            secondary=sv*(1.+p.crmsec*finefrac*xp.maximum(xp.log(elapsed)-xp.log(old),0.))
            sv=xp.where((~primary)&(pre>0),xp.minimum(maxsv,xp.maximum(sv,secondary)),sv)
            age=xp.where(primary,time_s,age);pre=xp.maximum(pre,load)
        return sv,pre,age

    def advance(self, layers, suspended_mass_areal, fluff_mass_areal, solid_fraction,
                preload, deposit_time, area, depth, shear_pa, bulk_density, sediment_type,
                *, dt_s, time_s=0., morfac=1., erosion_calibration=1.e-4,
                critical_erosion_pa=.2, critical_deposition_pa=.1,
                base_erosion_flux=0., fixfac=1., max_bed_slope=0., salinity=0., temperature_c=20.,
                turbulent_shear_pa=0., dissipation_m2_s3=1.e-6, ws0=1.e-4,
                wsm=1.e-3, salinity_max=10., temperature_coefficient=1.,
                floc_critical_kg_m3=.01, concentration_power=1., csoil=1600.):
        xp=self.xp
        if not np.isfinite([dt_s,time_s,morfac]).all() or dt_s<=0 or morfac<0:
            raise ValueError("Invalid Q9 clock")
        q=xp.ascontiguousarray(xp.asarray(layers,dtype=xp.float64));
        if q.ndim!=4 or min(q.shape)<1 or self._bool(xp.any(~xp.isfinite(q))) or self._bool(xp.any(q<0)):
            raise ValueError("Q9 layers require nonnegative (layer,grain,y,x) inventory")
        nl,ng,ny,nx=q.shape;shape=(ny,nx);typ=xp.asarray(sediment_type,dtype=xp.int32)
        if typ.shape!=(ng,) or self._bool(xp.any((typ<0)|(typ>2))): raise ValueError("Sediment type must be sand=0, silt=1 or clay=2")
        water=self._array(suspended_mass_areal,(ng,ny,nx),"suspended inventory",nonnegative=True)
        fluff=self._array(fluff_mass_areal,(ng,ny,nx),"fluff inventory",nonnegative=True)
        sv=self._array(solid_fraction,(nl,ny,nx),"solid fraction",positive=True)
        pre=self._array(preload,(nl,ny,nx),"preload",nonnegative=True);age=self._array(deposit_time,(nl,ny,nx),"deposit time")
        area=self._array(area,shape,"area",positive=True);depth=self._array(depth,shape,"depth",nonnegative=True)
        shear=self._array(shear_pa,shape,"bed shear",nonnegative=True);bulk=self._array(bulk_density,(ng,),"density",positive=True)
        if not self._bool(xp.any(typ>0)):
            return {"layers":q.copy(),"suspended_mass_areal":water.copy(),"fluff_mass_areal":fluff.copy(),
                "solid_fraction":sv.copy(),"preload":pre.copy(),"deposit_time":age.copy(),"kernel_launches":0,
                "method_ids":(),"runtime_identity":self.identity,"dt_s":float(dt_s),"sand_only_identity":True}
        solidvol=xp.sum(q/bulk[None,:,None,None],axis=1);thickness=solidvol/xp.maximum(sv,1.e-15)
        tcr_bed,ero_bed,_,_=self.erodibility(q,thickness,sv,bulk,typ)
        frac=q[0]/xp.maximum(xp.sum(q[0],axis=0),1.e-15)
        mudfrac=xp.sum(xp.where((typ>0)[:,None,None],frac,0.),axis=0)
        tcrit=self._array(critical_erosion_pa,(ng,ny,nx),"critical erosion",positive=True)*tcr_bed[None]
        maxslope=self._array(max_bed_slope,(ny,nx),"maximum bed slope",nonnegative=True)
        if self._bool(xp.any(maxslope>self.parameters.wet_slope)):
            denom=(self.parameters.slope_beta-1.)*self.parameters.wet_slope
            if denom<=0:raise ValueError("Slope erosion correction requires slope_beta > 1")
            fac=(self.parameters.slope_critical_min_pa-tcrit)/denom
            reduced=fac*shear[None]+tcrit-fac*self.parameters.wet_slope
            tcrit=xp.where(maxslope[None]>self.parameters.wet_slope,
                xp.maximum(reduced,self.parameters.slope_critical_min_pa),tcrit)
        coeff=self._array(erosion_calibration,(ng,ny,nx),"erosion calibration",nonnegative=True)*ero_bed[None]
        if self.methods.capacity.endswith("ratio"):
            excess=xp.maximum(shear[None]/tcrit-1.,0.)**self.parameters.erosion_power
        else: excess=xp.maximum(shear[None]-tcrit,0.)
        velocity=coeff*excess+self._array(base_erosion_flux,(ng,ny,nx),"base erosion",nonnegative=True)
        # Exact locked sand_mud transformation on per-fraction erosion rates.
        if self.parameters.pmcrit>=0:
            sand=xp.sum(xp.where((typ==0)[:,None,None],frac*velocity,0.),axis=0)/xp.maximum(1.-mudfrac,1.e-15)
            noncoh=mudfrac<=self.parameters.pmcrit
            velocity=xp.where(((typ>0)[:,None,None]&noncoh[None])&(sand[None]>0),sand[None],velocity)
            weight=(1.-mudfrac)/xp.maximum(1.-self.parameters.pmcrit,1.e-15)
            finev=xp.where((typ>0)[:,None,None]&(~noncoh[None])&(velocity>0),
                velocity*(sand[None]/xp.maximum(velocity,1.e-15))**weight[None],0.)
            favg=xp.sum(frac*finev,axis=0)/xp.maximum(mudfrac,1.e-15)
            velocity=xp.where((typ==0)[:,None,None]&(~noncoh[None]),favg[None],velocity)
            velocity=xp.where((typ>0)[:,None,None]&(~noncoh[None])&(velocity>0),finev,velocity)
        fix=self._array(fixfac,(ng,ny,nx),"fixfac",nonnegative=True)
        erosion_weight=xp.where(((typ>0)&self.parameters.old_mud_fraction)[:,None,None],1.,frac)
        wet=depth>0;erosion_rate=fix*erosion_weight*velocity*morfac*wet[None]
        erosion=xp.minimum(erosion_rate*dt_s,q[0]);q=q.copy();q[0]-=erosion;water+=erosion
        safe_depth=xp.maximum(depth,1.e-15);conc=xp.where(wet[None],water/safe_depth[None],0.)
        ws=self.settling_velocity(conc,typ,salinity=salinity,temperature_c=temperature_c,
            turbulent_shear_pa=turbulent_shear_pa,dissipation_m2_s3=dissipation_m2_s3,
            ws0=ws0,wsm=wsm,salinity_max=salinity_max,temperature_coefficient=temperature_coefficient,
            floc_critical_kg_m3=floc_critical_kg_m3,concentration_power=concentration_power,csoil=csoil)
        tdep=self._array(critical_deposition_pa,(ng,ny,nx),"critical deposition",nonnegative=True)
        if self.parameters.deposition_efficiency == -1:
            krone=xp.where(tdep>0,xp.maximum(0.,1.-shear[None]/xp.maximum(tdep,1.e-15)),0.)
        else:krone=xp.full((ng,ny,nx),self.parameters.deposition_efficiency)
        deposition=xp.minimum(ws*krone*conc*dt_s*morfac,water)*wet[None];water-=deposition
        if self.methods.fluff_type==0:q[0]+=deposition
        elif self.methods.fluff_type==1:fluff+=deposition
        else:
            depfac=self.parameters.fluff_dep_fraction;q[0]+=deposition*depfac;fluff+=deposition*(1-depfac)
        fluff_total=xp.sum(fluff,axis=0);ffrac=fluff/xp.maximum(fluff_total[None],1.e-15)
        ferode=ffrac*xp.minimum(fluff_total[None]*self.parameters.fluff_erosion_first,
            self.parameters.fluff_erosion_zero)*xp.maximum(shear[None]-self.parameters.fluff_critical_pa,0.)*dt_s
        ferode=xp.minimum(ferode,fluff)*wet[None];fluff-=ferode;water+=ferode
        burial=xp.zeros_like(fluff)
        if self.methods.fluff_type==1 and self.methods.fluff_burial_mode:
            rate=self.parameters.fluff_burial_zero
            if self.methods.fluff_burial_mode==2:
                rate=self.parameters.fluff_acalbur0*(bulk[:,None,None]-self.parameters.water_density_kg_m3)/self.parameters.water_density_kg_m3
                rate*=self.parameters.fluff_permeability_m_s*self.parameters.fluff_density_kg_m3**2/bulk[:,None,None]
            burial=xp.minimum(ffrac*xp.minimum(fluff_total[None]*self.parameters.fluff_burial_first,rate)*dt_s*morfac,fluff)
            fluff-=burial;q[0]+=burial
        water,floc_transfer=self._flocculate(water,typ,safe_depth,dt_s,wet=wet)
        sv,pre,age=self._consolidate(q,sv,pre,age,bulk,typ,dt_s,time_s+dt_s)
        return {"layers":q,"suspended_mass_areal":water,"fluff_mass_areal":fluff,
            "solid_fraction":sv,"preload":pre,"deposit_time":age,"critical_erosion_pa":tcrit,
            "erodibility":coeff,"settling_velocity_m_s":ws,"erosion_mass_areal":erosion,
            "deposition_mass_areal":deposition,"fluff_erosion_mass_areal":ferode,
            "fluff_burial_mass_areal":burial,"floc_transfer_mass_areal":floc_transfer,
            "mixed_erosion_rate":velocity,"kernel_launches":1,"method_ids":self.methods.method_ids,
            "runtime_identity":self.identity,"dt_s":float(dt_s),"sand_only_identity":False}

    def resolve_boundary(self, interior, prescribed, sediment_type, *, outflow=False):
        xp=self.xp;inside=xp.asarray(interior,dtype=xp.float64);outside=xp.broadcast_to(
            xp.asarray(prescribed,dtype=xp.float64),inside.shape).copy();typ=xp.asarray(sediment_type)
        if inside.shape[0]!=typ.size:raise ValueError("Boundary fraction layout differs")
        if outflow:return inside.copy()
        if self.methods.neumann_mud:outside=xp.where((typ>0)[:,None],inside,outside)
        return outside


def resolve_mud_boundary_v106(interior, prescribed, sediment_type, *, neumann_mud, outflow=False):
    """Apply locked ``NeuBcMud`` after the common Q4 boundary schedule."""
    interior=np.asarray(interior,float);prescribed=np.broadcast_to(np.asarray(prescribed,float),interior.shape).copy()
    typ=np.asarray(sediment_type,int)
    if interior.shape[0]!=typ.size: raise ValueError("Boundary fraction layout differs")
    if outflow: return interior.copy()
    if neumann_mud: prescribed[typ>0]=interior[typ>0]
    return prescribed


class FineOwnerV106:
    def __init__(self,runtime:FineRuntimeV106,*,time_s=0.,updates=0,**state):
        self.runtime=runtime;self.time_s=float(time_s);self.updates=int(updates);self._trial=None
        self.state={k:runtime.xp.asarray(v).copy() for k,v in state.items()}

    @property
    def state_sha256(self):
        h=hashlib.sha256();h.update(np.asarray([self.time_s,self.updates],np.float64).tobytes())
        for key in sorted(self.state):
            h.update(key.encode());h.update(np.ascontiguousarray(self.runtime._host(self.state[key])).tobytes())
        return h.hexdigest()

    def propose(self,**forcing):
        if self._trial is not None: raise RuntimeError("Q9 trial already pending")
        self._trial=self.runtime.advance(**self.state,**forcing,time_s=self.time_s);return self._trial

    def commit(self,*,cancelled=False,accept=True):
        if self._trial is None: raise RuntimeError("No Q9 trial pending")
        trial=self._trial;self._trial=None
        if cancelled: raise InterruptedError("Q9 transaction cancelled before publication")
        if not accept:return False
        mapping={"layers":"layers","suspended_mass_areal":"suspended_mass_areal","fluff_mass_areal":"fluff_mass_areal",
            "solid_fraction":"solid_fraction","preload":"preload","deposit_time":"deposit_time"}
        for target,source in mapping.items(): self.state[target]=trial[source].copy()
        self.time_s+=trial["dt_s"];self.updates+=1;return True

    def discard(self):self._trial=None

    def checkpoint(self):
        if self._trial is not None:raise RuntimeError("Cannot checkpoint pending Q9 trial")
        return {"schema":"braids.fines.v106","runtime_identity":self.runtime.identity,"time_s":self.time_s,
            "updates":self.updates,"state":{k:self.runtime._host(v).copy() for k,v in self.state.items()},
            "state_sha256":self.state_sha256}

    def restore(self,document:Mapping):
        if document.get("schema")!="braids.fines.v106" or document.get("runtime_identity")!=self.runtime.identity:
            raise ValueError("Q9 checkpoint identity differs")
        candidate=FineOwnerV106(self.runtime,time_s=document["time_s"],updates=document["updates"],**document["state"])
        if candidate.state_sha256!=document.get("state_sha256"):raise ValueError("Q9 checkpoint digest differs")
        self.__dict__.update(candidate.__dict__)


def remap_fine_owner_v106(owner:FineOwnerV106, offsets, sources, weights, target_area):
    """Parent-owned conservative remap of every Q9 mass/history field."""
    xp=owner.runtime.xp;off=np.asarray(offsets,int);src=np.asarray(sources,int);w=np.asarray(weights,float)
    old_area=np.asarray(owner.runtime._host(owner.state["area"])).ravel();ta=np.asarray(target_area,float);nt=ta.size
    if off.shape!=(nt+1,) or off[0]!=0 or off[-1]!=src.size or src.size!=w.size or np.any(src<0) or np.any(src>=old_area.size):
        raise ValueError("Invalid Q9 remap CSR")
    row=np.repeat(np.arange(nt),np.diff(off))
    if not np.allclose(np.bincount(row,weights=w,minlength=nt),1.,atol=1e-13,rtol=0) or not np.allclose(
        np.bincount(src,weights=w*ta.ravel()[row],minlength=old_area.size),old_area,atol=1e-12,rtol=0):
        raise ValueError("Q9 remap is not area conservative")
    shape=ta.shape if ta.ndim==2 else (1,nt);new={}
    mass_fields={"layers","suspended_mass_areal","fluff_mass_areal"}
    for key,val in owner.state.items():
        a=owner.runtime._host(val);spatial=a.shape[-2:]
        if spatial!=owner.state["area"].shape: new[key]=xp.asarray(a);continue
        flat=a.reshape((-1,old_area.size));out=np.empty((flat.shape[0],nt),float)
        for t in range(nt):
            ids=src[off[t]:off[t+1]];ww=w[off[t]:off[t+1]]
            out[:,t]=np.sum(flat[:,ids]*ww,axis=1)
        new[key]=xp.asarray(out.reshape(a.shape[:-2]+shape))
    new["area"]=xp.asarray(ta.reshape(shape))
    result=FineOwnerV106(owner.runtime,time_s=owner.time_s,updates=owner.updates,**new)
    for field in mass_fields:
        before=np.sum(owner.runtime._host(owner.state[field])*old_area,axis=tuple(range(owner.state[field].ndim-2,owner.state[field].ndim)))
        after=np.sum(owner.runtime._host(result.state[field])*ta.reshape(shape),axis=tuple(range(result.state[field].ndim-2,result.state[field].ndim)))
        if not np.allclose(before,after,atol=2e-10,rtol=0):raise ValueError(f"Q9 remap changed {field}")
    return result
