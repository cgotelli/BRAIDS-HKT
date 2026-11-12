"""Exterior composition/capacity on the common owner; public admission pending.

No extra physical inventory or second transport solver. Exterior columns are
boundary constitutive history, not finite donor cells. The common AMR transfer
preserves accepted history by physical boundary overlap, not owner-cell IDs.
"""
from copy import copy
import hashlib
from pathlib import Path
from types import MethodType
import numpy as np

from .bend_methods_v1 import digest
from .bed_columns_v101 import host
from .river_corrections import RiverCorrections,SLOPE_MODES
from .river_roughness import selected
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHODS={'composition.free':1,'composition.fixed':2,
         'composition.mass_fraction':3,'composition.reference_volume_fraction':4}
NAMES=('exterior_composition','exterior_sediment_face')


class ExteriorSediment:
    def __init__(self,exterior,*,methods,initial_mass,initial_sv,fractions,folder,composition_schedules=None):
        a=exterior.a;p=a.physics;xp=a.xp
        if a.time or a.steps or p._open or hasattr(p,'exterior_sediment'):
            raise ValueError('Install exterior sediment once before execution')
        if not a.morphology:raise ValueError('Exterior composition requires sediment enabled')
        if set(methods)!=set(range(exterior.nb)) or not set(methods.values())<=set(METHODS):
            raise ValueError('Explicit named composition method for every boundary required')
        self.a,self.p,self.xp,self.exterior=a,p,xp,exterior
        faces=np.flatnonzero(host(exterior.selected));ne=len(faces)
        self.faces=xp.asarray(faces,dtype=xp.int32);self.owner=p.g['owner'][self.faces].copy()
        nl,ng,nc=a.columns.mass.shape;self.ne,self.nl,self.ng=ne,nl,ng
        m=np.asarray(initial_mass,float);sv=np.asarray(initial_sv,float);f=np.asarray(fractions,float)
        if (m.shape!=(nl,ng,ne) or sv.shape!=(nl,ne) or f.shape!=(ng,ne)
                or not np.isfinite(m).all() or not np.isfinite(sv).all() or not np.isfinite(f).all()
                or np.any(m<0) or np.any(sv<=0) or np.any(sv>1) or np.any(f<0)
                or not np.allclose(f.sum(0),1.,atol=1e-12,rtol=0)):
            raise ValueError('Explicit finite exterior columns and normalized fraction fields required')
        self.selection=p.identity['resolved_river_model']['model']['physics']['sediment']
        self.parameters=p.identity['river_corrections']['parameters']
        coarse=self.selection['bed']['coarse_layer']
        mode=np.array([METHODS[methods[int(a.data['bid'][j])]] for j in faces],np.int32)
        if coarse:mode[mode==2]=1 # FLOW bndmorlyr: fixed+coarse copies the full interior column.
        self.mode=xp.asarray(mode);self.fractions=xp.asarray(f).copy()
        from braids_core.river_exterior_schedule import ExteriorForcing
        self.schedule=ExteriorForcing(xp,methods,composition_schedules,float(a.data['endtime']),
            a.data['bid'][faces],f,composition=True)
        self.scheduled_fractions=self.fractions.copy() if self.schedule.tables else None
        self.engine=WaveEngine(a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.op=RiverCorrections(a.backend,folder,threads=p.flow.threads)
        p.exterior_mass=xp.asarray(m).copy();p.exterior_sv=xp.asarray(sv).copy()
        p.exterior_mass_trial=p.exterior_mass.copy();p.exterior_sv_trial=p.exterior_sv.copy()
        self.update(a.columns.mass,a.columns.sv)
        p.exterior_mass[:]=p.exterior_mass_trial;p.exterior_sv[:]=p.exterior_sv_trial
        self.diam=xp.asarray(a.methods.diameter_m);self.mean=xp.empty(ne);self.d90=xp.empty(ne)
        self.hide=xp.ones((ng,ne));self.slope_hide=xp.ones((ng,ne));self.grains=xp.zeros((3,ne))
        self.grain_status=xp.zeros(ne,dtype=xp.int32)
        self.capacity=[copy(c) for c in a.capacity]
        for cap in self.capacity:cap.taurat=None
        self.values=[xp.zeros((6,ne)) for _ in self.capacity]
        self.surface=xp.zeros((ng,ne));self.trace=xp.zeros((3,ne))
        p.exterior_sediment=self
        previous_finish,previous_commit=p.finish,p.commit
        def finish(qold,qnew,water,**kw):
            result=previous_finish(qold,qnew,water,**kw)
            c=a.columns;mass,packing=c.pending[:2] if c.pending is not None else (c.mass,c.sv)
            self.update(mass,packing,start=exterior.owner.time,dt=p._dt)
            return result
        def commit():
            previous_commit();p.exterior_mass[:]=p.exterior_mass_trial;p.exterior_sv[:]=p.exterior_sv_trial
        p.finish,p.commit=finish,commit
        for k,c in enumerate(a.capacity):
            old=c.slope
            def slope(capacity,fields,k=k,old=old):
                result=old(fields)
                self.apply_slope(k,fields,result)
                return result
            c.slope=MethodType(slope,c)
        def sha(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        identity=dict(source_sha256=digest(__file__),runtime=self.engine.identity,
            requested_methods={str(k):v for k,v in methods.items()},
            resolved_methods={str(k):('composition.free' if coarse and v=='composition.fixed' else v) for k,v in methods.items()},
            fixed_coarse_reason='FLOW bndmorlyr copies the complete column for fixed composition with coarse layer',
            initial_mass_sha256=sha(m),initial_sv_sha256=sha(sv),fractions_sha256=sha(f),
            schedule=dict(tables=self.schedule.document,
                source_sha256=digest(Path(__file__).parents[1]/'river_exterior_schedule.py'),
                clock='physical endpoint composition; split at knots',
                state='immutable inputs; accepted exterior feeds capacity until the next committed update'),
            column_reference_density=list(a.material.definition.column_density),
            composition=a.columns.composition.identity,capacity=[c.identity for c in a.capacity],
            ownership='zero-area exterior column; accepted face flux alone changes physical inventory',
            geometry='exterior value at true face; affine secant with actual owner displacement',
            secondary='owner intensity extrapolated to exterior physical depth',
            suspended_size='owner accepted class DSS extrapolation; separate suspended concentration boundary unchanged',
            scope='hydraulic and sediment boundary integration with physical-overlap AMR history; public schema pending',public_support=False)
        exterior.owner.identity['exterior_sediment']=identity
        self.schedule.bind(a,'exterior_composition')
        if exterior.bedforms_enabled:
            from .river_exterior_bedforms import ExteriorBedforms
            ExteriorBedforms(self,folder)

    def update(self,mass,sv,*,start=0.,dt=0.):
        p=self.p
        fractions=self.fractions
        if self.scheduled_fractions is not None:
            self.schedule.sample(start,dt,self.scheduled_fractions);fractions=self.scheduled_fractions
        self.engine.launch('exterior_composition',self.ne,self.ne,p.n,self.nl,self.ng,self.mode,self.owner,
            self.a.material.column,fractions,p.exterior_mass,p.exterior_sv,mass,sv,
            p.exterior_mass_trial,p.exterior_sv_trial)

    def prepare(self,q,z,hi):
        a,p,xp=self.a,self.p,self.xp;ne=self.ne
        self.exterior.evaluate(q,z) # Fresh endpoint state, never the last midpoint iteration's scratch.
        self.trace[:]=self.exterior.trace[:,self.faces]
        safe,wet=p.safe(self.trace,p.sediment_depth)
        h=safe[0];raw=xp.maximum(self.trace[0],p.parameters.dry)
        rm,rv=selected(p.identity['river_roughness']['method'],p.identity['river_roughness']['parameters'])
        if hasattr(p,'spatial_roughness'):rv=p.spatial_roughness[self.owner]
        if self.exterior.bedforms is not None:
            c=self.exterior.bedforms.resistance(self.trace)[0]
        elif rm==1:c=xp.broadcast_to(xp.asarray(rv),(ne,)).copy()
        elif rm==2:c=raw**(1./6.)/rv
        elif rm==3:c=np.sqrt(9.81)/.41*xp.log1p(30.*raw/(np.e*rv))
        else:c=18.*xp.log10(12.*xp.maximum(.5,raw/rv))
        z0=raw/(np.e*xp.expm1(.41*c/np.sqrt(9.81)))
        cs=np.sqrt(9.81)/.41*xp.log1p(xp.maximum(.01,h)/xp.maximum(1e-8,np.e*z0))
        cs=xp.maximum(1e-6,cs)
        factor=.41*cs/(np.sqrt(9.81)*xp.log1p(h*float(np.float32(.368))/z0))
        cq=safe.copy();cq[1:]*=factor
        intensity=xp.where(q[0,self.owner]>p.parameters.dry,hi[self.owner]/xp.maximum(q[0,self.owner],p.parameters.dry),0.)
        eh=xp.where(wet,intensity*h,0.)
        stock=p.exterior_mass[0]
        # Prescribed exterior fractions describe an unlimited external supply,
        # not the finite (possibly empty) adjacent bed. Unit reference bulk
        # thickness is constitutive only: it never enters physical inventory.
        prescribed=(self.mode==3)|(self.mode==4)
        reference=self.fractions*xp.where(self.mode[None,:]==4,a.material.column[:,None],1.)
        stock=xp.where(prescribed[None,:],reference,stock)
        self.surface[:]=a.columns.composition.fractions(stock)
        weights=xp.ascontiguousarray(a.columns.composition.weights(stock))
        engine=self.capacity[0].engine
        engine.launch('mixture',ne,ne,self.ng,0,weights,self.diam,self.hide,self.d90,self.mean)
        measured_d90=self.selection['capacity'].get('grain_sizes_m',{}).get('d90')
        if measured_d90 is not None:self.d90.fill(measured_d90)
        intrinsic=self.capacity[0].needs_intrinsic_mixture
        if intrinsic:
            engine.launch('intrinsic_grains',ne,ne,self.ng,weights,self.diam,self.grains,self.grain_status)
            if bool(xp.any(self.grain_status)):raise ValueError('Invalid exterior intrinsic grain statistics')
        if self.selection['bed']['coarse_layer'] and self.parameters['dependent_layers']['hiding_scope']=='active_coarse':
            weights=xp.ascontiguousarray(a.columns.composition.weights(stock+p.exterior_mass[1]))
        hiding=self.parameters['hiding']
        self.op.hiding(weights,self.diam,'hiding.none' if intrinsic else self.selection['capacity_hiding'],
            hiding['capacity'],self.hide,self.mean)
        self.op.hiding(weights,self.diam,self.selection['slope_hiding'],hiding['slope'],self.slope_hide,self.mean)
        for k,cap in enumerate(self.capacity):
            cap.d90,cap.hiding,cap.z0=self.d90,self.hide[k],z0
            cap.grain_statistics,cap.hydraulic_chezy=self.grains,c
            cap.surface_fraction=self.surface[k]
            cap.suspended_diameter=(xp.ascontiguousarray(p.suspended_diameter[k,self.owner])
                if cap.is_vr1993 else xp.full(ne,cap.d))
            self.values[k][:]=cap.evaluate(safe,eh,capacity_q=cq,chezy=cs)
            self.values[k][:,~wet]=0.
        self.q,self.z=q,z

    def apply_slope(self,k,fields,out):
        p,xp=self.p,self.xp;cap=self.capacity[k];ne=self.ne
        local=xp.ascontiguousarray(fields[:,self.faces]);mode=SLOPE_MODES[self.selection['bed_slope']]
        self.engine.launch('exterior_sediment_face',ne,ne,p.nf,p.n,mode,int(cap.is_vr1993),
            p.parameters.dry,p.sediment_depth,self.faces,self.owner,p.g['normal'],p.g['displacement'],
            self.q,self.z,p.exterior_bed,p.grad,p.bedcell,self.values[k],self.trace,self.mean,
            self.slope_hide[k],p.river_slope_hiding[k],cap.taurat,local)
        corrected=self.op.slope(local,self.selection['bed_slope'],self.parameters['slope'])
        wet=(self.q[0,self.owner]>p.parameters.dry)&(self.trace[0]>p.parameters.dry)
        corrected[:,~wet]=0.
        out[:,self.faces]=corrected
        self.last_inputs=local

    def donor_fraction(self,k,factor):
        p=self.p;f=self.faces
        incoming=(p.aux[2,f]<0.)&(p.bedbc[f]!=1)
        factor[f]=self.xp.where(incoming,self.surface[k],factor[f])
        return factor
