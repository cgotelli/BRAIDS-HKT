"""Transported blade-induced turbulence for actuator-disc research.

The blade source is the source-complete Nishino--Willden low-order model:
``k_b=beta S_U``, a blade length scale, lifetime-conserving mixing, and the
published disc-plane ``S_k``/``S_epsilon`` sources.  Standard depth-averaged
``k-epsilon`` transports those sources downstream and supplies the conservative
horizontal momentum viscosity.  This remains a research component until the
Mycek velocity and turbulence-intensity gates are passed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
from types import MethodType

import numpy as np
from shapely.geometry import Polygon

from .bend_methods_v1 import BendTrialRejected, OPTIONS, digest
from .flow2dh_mobile_bed import ROOT


METHOD = "turbulence.nishino_willden_bit_standard_kepsilon_v1"
REFERENCE = "Nishino and Willden, Low-Order Modelling of Blade-Induced Turbulence for RANS Actuator Disk Computations of Wind and Tidal Turbines"
DOI = "10.1007/978-3-642-54696-9_23"
PREPRINT = "arXiv:1210.3173"


@dataclass(frozen=True)
class TransportedTurbulence:
    ambient_turbulence_intensity: float
    ambient_velocity_m_s: float
    ambient_length_scale_m: float
    source_distribution: str = "outer_annulus"
    beta: float = .30
    blade_length_ratio: float = .10
    c_mu: float = .09
    c_epsilon_1: float = 1.44
    c_epsilon_2: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3
    molecular_viscosity_m2_s: float = 1.e-6
    maximum_substep_s: float = .02

    def validate(self):
        values=asdict(self)
        numeric=[v for k,v in values.items() if k!='source_distribution']
        if not all(math.isfinite(v) and v>0 for v in numeric):
            raise ValueError("Positive finite transported-turbulence parameters required")
        if self.source_distribution not in ("whole_disc","outer_annulus"):
            raise ValueError("Choose whole_disc or outer_annulus blade-source distribution")
        if not 0 < self.ambient_turbulence_intensity < 1 or not 0 < self.c_mu < 1:
            raise ValueError("Invalid turbulence intensity or Cmu")
        return self

    @property
    def ambient_k_m2_s2(self):
        return 1.5*(self.ambient_turbulence_intensity*self.ambient_velocity_m_s)**2

    @property
    def ambient_epsilon_m2_s3(self):
        return (self.c_mu**.75*self.ambient_k_m2_s2**1.5/
                self.ambient_length_scale_m)


def blade_state(force_per_density_m4_s2, rotor_area_m2, local_velocity_m_s,
                diameter_m, config):
    """Published BIT source state per unit physical rotor-disc area."""
    config.validate();area=float(rotor_area_m2);diameter=float(diameter_m)
    if not math.isfinite(area) or area<=0 or not math.isfinite(diameter) or diameter<=0:
        raise ValueError("Positive rotor area and diameter required")
    source_u=abs(float(force_per_density_m4_s2))/area
    kb=config.beta*source_u
    length=config.blade_length_ratio*diameter
    epsb=config.c_mu**.75*kb**1.5/length if kb else 0.
    ka=config.ambient_k_m2_s2;epsa=config.ambient_epsilon_m2_s3
    epsm=(ka+kb)**2/(ka*ka/epsa+kb*kb/epsb) if kb else epsa
    velocity=abs(float(local_velocity_m_s))
    return dict(momentum_loss_m3_s2=source_u, blade_k_m2_s2=kb,
                blade_epsilon_m2_s3=epsb, mixed_epsilon_m2_s3=epsm,
                k_source_m3_s3=velocity*kb,
                epsilon_source_m3_s4=velocity*(epsm-epsa),
                blade_length_scale_m=length)


class TurbulenceEngine:
    NAMES=("hkt_scalar_faces","hkt_momentum_faces")
    def __init__(self,backend,threads):
        source=Path(__file__).with_suffix('.h');self.backend=backend;self.threads=threads
        self.identity=dict(source_sha256=digest(source),backend=backend,threads=threads)
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=source.read_text(),options=OPTIONS,
                                     name_expressions=self.NAMES)
            self.fn={name:self.module.get_function(name) for name in self.NAMES}
            self.identity['options']=list(OPTIONS);return
        if backend!='cpu':raise ValueError("Explicit CPU or GPU backend required")
        compiler=shutil.which('g++')
        if not compiler:raise RuntimeError("OpenMP compiler required")
        version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
        flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc',
               '-static-libstdc++','-ffp-contract=off']
        key=hashlib.sha256((version+str(flags)+source.read_text()).encode()).hexdigest()[:20]
        folder=ROOT/'build/hydrokinetic_turbulence'/key;folder.mkdir(parents=True,exist_ok=True)
        library=folder/'transport.dll'
        if not library.exists():
            subprocess.run([compiler,*flags,str(source),'-o',str(library)],check=True,
                           capture_output=True,timeout=120)
        self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
        self.lib=ctypes.CDLL(str(library));self.fn={n:getattr(self.lib,n) for n in self.NAMES}
        for function in self.fn.values():function.restype=None
        self.identity.update(compiler=version,flags=flags,library_sha256=digest(library))

    def launch(self,name,n,*args):
        if self.backend=='gpu':
            args=tuple(np.int32(a) if isinstance(a,int) else np.float64(a)
                       if isinstance(a,float) else a for a in args)
            self.fn[name](((n+127)//128,),(128,),args);return
        self.lib.hkt_threads(ctypes.c_int(self.threads))
        args=tuple(ctypes.c_int(a) if isinstance(a,int) else ctypes.c_double(a)
                   if isinstance(a,float) else ctypes.c_void_p(a.ctypes.data) for a in args)
        self.fn[name](*args)


class TurbulenceTransport:
    def __init__(self,owner,cells,config):
        self.owner=owner;self.p=owner.physics;self.xp=self.p.xp
        self.config=config.validate();self.engine=TurbulenceEngine(owner.backend,self.p.flow.threads)
        if not hasattr(owner,'turbine_source'):
            raise ValueError("Install hydrokinetic turbines before transported turbulence")
        if float(self.p.nu)!=0.:
            raise ValueError("Transported turbine turbulence replaces constant horizontal viscosity")
        self._allocate(owner.q)
        self.source_weights=self._source_weights(cells)

    def _allocate(self,q):
        p=self.p;xp=self.xp;n=p.n;nf=p.nf;h=q[0]
        ka=self.config.ambient_k_m2_s2;ea=self.config.ambient_epsilon_m2_s3
        p.hkt_k=h*ka;p.hkt_epsilon=h*ea
        p.hkt_k_trial=p.hkt_k.copy();p.hkt_epsilon_trial=p.hkt_epsilon.copy()
        p.hkt_nu=xp.full(n,self.config.c_mu*ka*ka/ea)
        p.hkt_nu_trial=p.hkt_nu.copy();p.hkt_grad=xp.zeros((10,n))
        p.hkt_flux_k=xp.zeros(nf);p.hkt_flux_epsilon=xp.zeros(nf)
        p.hkt_work_k=xp.zeros(n);p.hkt_work_epsilon=xp.zeros(n)
        p.hkt_work2_k=xp.zeros(n);p.hkt_work2_epsilon=xp.zeros(n)
        p.hkt_source_k=xp.zeros(n);p.hkt_source_epsilon=xp.zeros(n)
        p.hkt_momentum_face=xp.zeros((2,nf));p.hkt_momentum_work=xp.zeros((3,n))
        p.hkt_unit=xp.ones(n)
        p.hkt_k_face_integral=xp.zeros(nf);p.hkt_epsilon_face_integral=xp.zeros(nf)
        p.hkt_k_boundary_ledger=xp.zeros(nf);p.hkt_epsilon_boundary_ledger=xp.zeros(nf)
        p.hkt_k_source_ledger=xp.zeros(());p.hkt_epsilon_source_ledger=xp.zeros(())
        p.hkt_diffusion_open=xp.ones(nf)

    def _source_weights(self,cells):
        result=[];xp=self.xp
        for footprint in self.owner.turbine_source.footprints:
            definition=footprint.definition;c=np.asarray(definition.center_xy)
            tangent=np.asarray(definition.tangent_xy);radius=.5*definition.rotor_diameter_m
            inner=.8*radius
            normal=np.asarray(definition.normal_xy)
            half=.5*footprint.support_thickness_m;t=tangent*radius;n=normal*half
            # The published zero-thickness disc creates the mixed state just
            # downstream.  Smear that jump over only the downstream half of
            # the numerical support, not symmetrically upstream/downstream.
            support=Polygon((c-t,c+t,c+t+n,c-t+n))
            raw=[]
            for i in footprint.cell_ids:
                polygon=Polygon(cells[int(i)].vertices_xy,getattr(cells[int(i)],'holes_xy',()))
                piece=polygon.intersection(support)
                if piece.area<=1.e-16:
                    raw.append(0.);continue
                s=abs(float((np.asarray(piece.centroid.coords[0])-c)@tangent))
                outer=2*math.sqrt(max(radius*radius-s*s,0.))
                chord=outer
                if self.config.source_distribution=='outer_annulus':
                    chord-=2*math.sqrt(max(inner*inner-s*s,0.))
                raw.append(float(piece.area)*max(chord,0.)/half)
            raw=np.asarray(raw);total=float(raw.sum())
            if total<=0:raise ValueError(definition.turbine_id+": blade source misses the mesh")
            # Nishino--Willden keeps the full disc-integrated BIT source when
            # concentrating production in the published outer annulus.
            target=(definition.rotor_area_m2 if self.config.source_distribution=='whole_disc'
                    else math.pi*(radius*radius-inner*inner))
            weighted=raw*(target/total)
            density=weighted/footprint.cell_area_m2
            result.append(dict(ids=xp.asarray(footprint.cell_ids),density=xp.asarray(density),
                               integrated_area_m2=float(weighted.sum()),definition=definition))
            xy=np.asarray(self.p.mesh['xy']).T;owner=np.asarray(self.p.mesh['owner'])
            neighbour=np.asarray(self.p.mesh['neighbour']);inside=neighbour>=0
            other=np.maximum(neighbour,0);ro=xy[owner]-c;rj=xy[other]-c
            so=ro@normal;sj=rj@normal;den=sj-so
            fraction=np.divide(-so,den,out=np.full_like(so,np.nan),where=np.abs(den)>1.e-30)
            cross=ro+fraction[:,None]*(rj-ro)
            transverse=np.asarray(definition.tangent_xy)
            tol=1.e-8*max(1.,definition.rotor_diameter_m)
            barrier=inside&(fraction>=-tol)&(fraction<=1+tol)&(np.abs(cross@transverse)<=radius+tol)
            self.p.hkt_diffusion_open[self.xp.asarray(np.flatnonzero(barrier))]=0.
        return tuple(result)

    def _nu(self,q,k_inventory,epsilon_inventory,out):
        xp=self.xp;wet=q[0]>self.p.parameters.dry
        h=xp.where(wet,q[0],1.);k=xp.where(wet,k_inventory/h,0.)
        eps=xp.where(wet,epsilon_inventory/h,1.)
        out[:]=xp.where(wet,self.config.c_mu*k*k/xp.maximum(eps,1.e-30),0.)
        if not bool(xp.all(xp.isfinite(out))) or bool(xp.any(out<0)):
            raise BendTrialRejected("Invalid transported eddy viscosity")

    def momentum(self,q,rate):
        p=self.p;g=p.g
        p._gradients(q,p.hkt_k,p.hkt_grad)
        self.engine.launch('hkt_momentum_faces',p.nf,p.nf,p.n,float(p.parameters.dry),
            q,p.z0,p.hkt_grad,p.hkt_nu,g['owner'],g['neighbour'],g['normal'],
            g['distance'],g['xy'],p.hkt_momentum_face)
        p.hkt_momentum_work.fill(0.)
        p.flow.launch('flow_momentum_gather',p.n,p.n,p.nf,p.hkt_unit,p.hkt_momentum_face,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],p.hkt_momentum_work)
        rate[1:]+=p.hkt_momentum_work[1:]

    def _faces(self,q,inventory,nu,water,sigma,incoming,out):
        p=self.p;g=p.g;p._gradients(q,inventory,p.hkt_grad)
        self.engine.launch('hkt_scalar_faces',p.nf,p.nf,p.n,float(p.parameters.dry),
            float(sigma),float(self.config.molecular_viscosity_m2_s),q,inventory,nu,
            p.hkt_grad,water,float(incoming),g['owner'],g['neighbour'],g['bid'],
            g['distance'],g['displacement'],p.hkt_diffusion_open,out)

    def _sources(self,q,k_inventory,epsilon_inventory,nu):
        p=self.p;xp=self.xp;h=q[0];wet=h>p.parameters.dry;safe=xp.where(wet,h,1.)
        k=xp.where(wet,k_inventory/safe,0.);eps=xp.where(wet,epsilon_inventory/safe,0.)
        p._gradients(q,k_inventory,p.hkt_grad);g=p.hkt_grad
        strain=2*g[0]**2+2*g[3]**2+(g[1]+g[2])**2
        production=nu*strain
        p.hkt_source_k[:]=xp.where(wet,h*(production-eps),0.)
        ratio=xp.where(wet,eps/xp.maximum(k,1.e-30),0.)
        p.hkt_source_epsilon[:]=xp.where(wet,h*(self.config.c_epsilon_1*ratio*production-
            self.config.c_epsilon_2*ratio*eps),0.)
        diagnostics=self.owner.turbine_source.sample(q)
        blade=[]
        for state,weights in zip(diagnostics,self.source_weights):
            value=blade_state(state['force_per_density_m4_s2'],weights['definition'].rotor_area_m2,
                state['local_velocity_m_s'],weights['definition'].rotor_diameter_m,self.config)
            ids=weights['ids'];density=weights['density']
            p.hkt_source_k[ids]+=density*value['k_source_m3_s3']
            p.hkt_source_epsilon[ids]+=density*value['epsilon_source_m3_s4']
            blade.append(dict(turbine_id=weights['definition'].turbine_id,**value))
        return blade

    def _euler(self,q,ki,ei,water,dt,ko,eo,nu,*,base_k=None,base_epsilon=None):
        p=self.p;g=p.g;self._nu(q,ki,ei,nu);self._sources(q,ki,ei,nu)
        self._faces(q,ki,nu,water,self.config.sigma_k,self.config.ambient_k_m2_s2,p.hkt_flux_k)
        self._faces(q,ei,nu,water,self.config.sigma_epsilon,self.config.ambient_epsilon_m2_s3,p.hkt_flux_epsilon)
        base_k=ki if base_k is None else base_k
        base_epsilon=ei if base_epsilon is None else base_epsilon
        p.flow.launch('flow_scalar_gather',p.n,p.n,float(dt),base_k,p.hkt_flux_k,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],ko)
        p.flow.launch('flow_scalar_gather',p.n,p.n,float(dt),base_epsilon,p.hkt_flux_epsilon,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],eo)
        ko+=dt*p.hkt_source_k;eo+=dt*p.hkt_source_epsilon
        return (p.hkt_flux_k.copy(),p.hkt_flux_epsilon.copy(),
                p.hkt_source_k.copy(),p.hkt_source_epsilon.copy())

    def finish(self,qold,qnew,water):
        p=self.p;xp=self.xp;dt=float(p._dt);g=p.g
        wet=p.hkt_k>0
        ratio=xp.where(wet,p.hkt_k/xp.maximum(p.hkt_epsilon,1.e-30),1.e30)
        reaction=.20*float(xp.min(ratio[wet]))/self.config.c_epsilon_2 if bool(xp.any(wet)) else dt
        diffusion=.15*float(xp.min(g['area']))/max(float(xp.max(p.hkt_nu)),1.e-30)
        sub=min(self.config.maximum_substep_s,reaction,diffusion)
        count=max(1,int(math.ceil(dt/sub)))
        if count>10000:raise BendTrialRejected("Turbulence substep count exceeded")
        ds=dt/count;ki=p.hkt_k.copy();ei=p.hkt_epsilon.copy()
        p.hkt_k_face_integral.fill(0.);p.hkt_epsilon_face_integral.fill(0.)
        source_k=source_e=0.;last_blade=[]
        for index in range(count):
            a0=index/count;a1=(index+1)/count
            qa=(1-a0)*qold+a0*qnew;qb=(1-a1)*qold+a1*qnew
            first=self._euler(qa,ki,ei,water,ds,p.hkt_work_k,p.hkt_work_epsilon,p.hkt_nu_trial)
            if bool(xp.any(p.hkt_work_k<0)) or bool(xp.any(p.hkt_work_epsilon<0)):
                raise BendTrialRejected("Turbulence predictor lost positivity")
            second=self._euler(qb,p.hkt_work_k,p.hkt_work_epsilon,water,ds,
                               p.hkt_work2_k,p.hkt_work2_epsilon,p.hkt_nu_trial,
                               base_k=ki,base_epsilon=ei)
            knew=.5*(p.hkt_work_k+p.hkt_work2_k);enew=.5*(p.hkt_work_epsilon+p.hkt_work2_epsilon)
            if (not bool(xp.all(xp.isfinite(knew))) or not bool(xp.all(xp.isfinite(enew)))
                    or bool(xp.any(knew<0)) or bool(xp.any(enew<0))):
                raise BendTrialRejected("Transported turbulence lost positivity")
            p.hkt_k_face_integral+=.5*ds*(first[0]+second[0])
            p.hkt_epsilon_face_integral+=.5*ds*(first[1]+second[1])
            source_k+=.5*ds*float(xp.sum((first[2]+second[2])*g['area']))
            source_e+=.5*ds*float(xp.sum((first[3]+second[3])*g['area']))
            ki=knew.copy();ei=enew.copy();last_blade=self._sources(qb,ki,ei,p.hkt_nu_trial)
        p.hkt_k_trial[:]=ki;p.hkt_epsilon_trial[:]=ei
        self._nu(qnew,ki,ei,p.hkt_nu_trial)
        outside=(g['neighbour']<0)&(g['bid']>=0)
        boundary_k=-float(xp.sum(p.hkt_k_face_integral*g['length']*outside))
        boundary_e=-float(xp.sum(p.hkt_epsilon_face_integral*g['length']*outside))
        change_k=float(xp.sum((ki-p.hkt_k)*g['area']))
        change_e=float(xp.sum((ei-p.hkt_epsilon)*g['area']))
        residual_k=change_k-boundary_k-source_k;residual_e=change_e-boundary_e-source_e
        scale=max(1.,abs(change_k),abs(boundary_k),abs(source_k),abs(change_e),abs(boundary_e),abs(source_e))
        if max(abs(residual_k),abs(residual_e))>2.e-10*scale:
            raise BendTrialRejected("Transported turbulence budget failed: "
                f"k={residual_k:.6e}, epsilon={residual_e:.6e}, scale={scale:.6e}")
        p.last.update(turbulence_substeps=count,turbulence_k_residual=residual_k,
            turbulence_epsilon_residual=residual_e,turbulence_k_boundary=boundary_k,
            turbulence_epsilon_boundary=boundary_e,turbulence_k_source=source_k,
            turbulence_epsilon_source=source_e,turbine_blade_turbulence=last_blade,
            maximum_eddy_viscosity_m2_s=float(xp.max(p.hkt_nu_trial)))

    def commit(self):
        p=self.p;g=p.g;outside=(g['neighbour']<0)&(g['bid']>=0)
        p.hkt_k[:]=p.hkt_k_trial;p.hkt_epsilon[:]=p.hkt_epsilon_trial;p.hkt_nu[:]=p.hkt_nu_trial
        p.hkt_k_boundary_ledger-=p.hkt_k_face_integral*g['length']*outside
        p.hkt_epsilon_boundary_ledger-=p.hkt_epsilon_face_integral*g['length']*outside
        p.hkt_k_source_ledger+=p.last['turbulence_k_source']
        p.hkt_epsilon_source_ledger+=p.last['turbulence_epsilon_source']


def install(owner,cells,config):
    """Compose transported BIT after the actuator and before execution."""
    p=owner.physics
    if owner.time or owner.steps or p._open:
        raise ValueError("Install transported turbine turbulence before execution")
    if owner.morphology:
        raise ValueError("The first transported turbine-turbulence milestone is hydrodynamic")
    transport=TurbulenceTransport(owner,cells,config)
    old_momentum=p.add_momentum;old_finish=p.finish;old_commit=p.commit
    def momentum(self,q,water,rate):
        old_momentum(q,water,rate);transport.momentum(q,rate)
    def finish(self,qold,qnew,water,*,morphology=False):
        result=old_finish(qold,qnew,water,morphology=morphology)
        transport.finish(qold,qnew,water);return result
    def commit(self):
        old_commit();transport.commit()
    p.add_momentum=MethodType(momentum,p);p.finish=MethodType(finish,p);p.commit=MethodType(commit,p)
    identity=dict(schema=1,method=METHOD,reference=REFERENCE,doi=DOI,preprint=PREPRINT,
        configuration=asdict(config),
        ambient_k_m2_s2=config.ambient_k_m2_s2,
        ambient_epsilon_m2_s3=config.ambient_epsilon_m2_s3,
        equations=dict(k_b="beta*S_U",epsilon_b="Cmu^(3/4)*k_b^(3/2)/l_b",
            mixed_epsilon="(ka+kb)^2/(ka^2/epsa+kb^2/epsb)",
            disk_sources="S_k=|Ud|*kb; S_epsilon=|Ud|*(epsilon_m-epsilon_a)"),
        source_total=("full physical rotor area" if config.source_distribution=='whole_disc' else
            "published edge annulus 0.4<r/D<0.5; beta=0 in rotor core"),
        source_support="conservative downstream half-support representation of zero-thickness disc jump",
        transport="depth-integrated standard k-epsilon; Heun; conservative water-face advection",
        momentum="conservative true-polygon variable eddy viscosity",
        runtime=transport.engine.identity,source_sha256=digest(__file__),
        hydrodynamics_only=True,desktop_admission=False)
    p.identity['hydrokinetic_turbulence_transport']=identity
    owner.identity['hydrokinetic_turbulence_transport']=identity
    owner.hydrokinetic_turbulence=transport
    return owner
