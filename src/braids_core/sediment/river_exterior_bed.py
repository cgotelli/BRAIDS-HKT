"""Exterior-state hydraulic binding; not yet a public sediment boundary option.

The arrays participate in the existing physical owner's transaction/checkpoint.
Composition, dynamic roughness and AMR history use the common owner. Remaining
hydraulic modes and public schema admission are qualified separately.
"""
import ctypes
import hashlib
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine
from braids_core.river_open_boundary import hydraulic_controls

METHODS={'bed.free':1,'bed.fixed':2,'bed.elevation':3,'bed.elevation_rate':4}
STAGE_POLICIES={'stage.outflow_only':0,'stage.subcritical_backflow':1}
NAMES=('exterior_update','exterior_partial','exterior_totals','exterior_stage_terms',
       'exterior_weights','exterior_flux')


def native_source(code):
    """Add a scoped flux override to BRAIDS' native bridge, never Delft3D."""
    marker='\ncontains\n'
    if code.count(marker)!=1:raise ValueError('Native module declaration layout changed')
    code=code.replace(marker,'\n logical :: river_exterior_active=.false.\n'
        ' integer, allocatable :: river_exterior_mask(:)\n'
        ' real(8), allocatable :: river_exterior_flux(:,:)\n'+marker)
    marker='   gaia_fixed=.false.'
    if code.count(marker)!=1:raise ValueError('Native boundary trace layout changed')
    code=code.replace(marker,'   if(river_exterior_active) then\n'
        '    if(all(river_exterior_mask(boundary_faces(boundary_offset(b):boundary_offset(b+1)-1))/=0)) cycle\n'
        '   endif\n'+marker)
    marker='  next=0.d0;accepted=0.d0'
    if code.count(marker)!=1:raise ValueError('Native spatial tail layout changed')
    code=code.replace(marker,'  if(river_exterior_active) then\n'
        '   !$omp parallel do schedule(static)\n'
        '   do f=1,nf\n'
        '    if(river_exterior_mask(f)==0) cycle\n'
        '    fl(1:3,f)=river_exterior_flux(:,f);fr(1:3,f)=river_exterior_flux(:,f)\n'
        '   enddo\n'
        '   !$omp end parallel do\n'
        '  endif\n'+marker)
    return code


class ExteriorBed:
    def __init__(self,a,*,methods,initial,value,stage_policies=None,hydraulics=None,hydraulic_schedules=None,bed_schedules=None):
        if a.time or a.steps or a.physics._open:raise ValueError('Install exterior bed before execution')
        p=a.physics;xp=a.xp;d=a.data;nf=p.nf;nb=len(d['mode'])
        if not nb:raise ValueError('Exterior bed requires an open boundary')
        self.bedforms_enabled=bool(p.identity.get('river_bedforms',{}).get('methods',{}).get('bedforms',False))
        self.bedforms=None
        roughness=p.identity.get('river_roughness')
        if not isinstance(roughness,dict):raise ValueError('Explicit resolved hydraulic roughness required')
        from .river_roughness import selected
        roughness_mode,roughness_value=selected(roughness['method'],roughness['parameters'])
        if hasattr(p,'exterior_bed'):raise ValueError('Exterior bed is already installed')
        if set(methods)!=set(range(nb)):raise ValueError('Explicit method for every open boundary required')
        if not set(methods.values())<=set(METHODS):raise ValueError('Named exterior bed method required')
        if not np.isin(d['mode'],[2,6]).all():raise ValueError('Exterior hydraulic binding pending for this inlet mode')
        hydraulic_mode,forcing,hydraulic_identity=hydraulic_controls(d,hydraulics)
        from braids_core.river_hydraulic_schedule import HydraulicForcing
        schedule=HydraulicForcing(xp,hydraulic_identity,hydraulic_schedules,float(d['endtime']))
        stage_ids=set(np.flatnonzero(hydraulic_mode==2).tolist())
        policies=({b:'stage.outflow_only' for b in stage_ids} if stage_policies is None else dict(stage_policies))
        if set(policies)!=stage_ids or not set(policies.values())<=set(STAGE_POLICIES):
            raise ValueError('Explicit named reversal policy for every stage boundary required')
        z=np.asarray(initial,dtype=float);v=np.asarray(value,dtype=float)
        if z.shape!=(nf,) or v.shape!=(nf,) or not np.isfinite([z,v]).all():
            raise ValueError('Finite per-face initial exterior elevation and forcing required')
        outside=(d['neighbour']<0)&(d['bid']>=0);mode=np.zeros(nf,np.int32)
        for b,method in methods.items():mode[outside&(d['bid']==b)]=METHODS[method]
        if np.any(z[~outside]) or np.any(v[~outside]):raise ValueError('Exterior fields must be zero on internal/wall faces')
        from braids_core.river_exterior_schedule import ExteriorForcing
        self.bed_schedule=ExteriorForcing(xp,methods,bed_schedules,float(d['endtime']),
            np.where(outside,d['bid'],-1),v)
        self.bed_values=xp.asarray(v).copy() if self.bed_schedule.tables else None
        self.a=a;self.p=p;self.xp=xp;self.nb=nb;self.schedule=schedule
        self.engine=WaveEngine(a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        p.exterior_mode=xp.asarray(mode);p.exterior_value=xp.asarray(v).copy()
        p.exterior_bed=xp.asarray(np.where(mode==3,v,z));p.exterior_trial=p.exterior_bed.copy()
        self.selected=xp.asarray(outside.astype(np.int32));self.face_selected=self.selected!=0
        self.face_indices=xp.asarray(np.flatnonzero(outside))
        self.hydraulic_mode=xp.asarray(hydraulic_mode,dtype=xp.int32)
        self.stage_policy=xp.asarray([STAGE_POLICIES[policies[b]] if b in policies else 0 for b in range(nb)],dtype=xp.int32)
        self.offset=xp.asarray(d['boundary_offset'],dtype=xp.int32)
        self.faces=xp.asarray(d['boundary_faces'],dtype=xp.int32)
        self.forcing=xp.asarray(forcing).copy();self.roughness=xp.full(p.n,roughness_value)
        if hasattr(p,'spatial_roughness'):self.roughness=p.spatial_roughness.copy()
        self.tiles=max(1,(int(np.max(np.diff(d['boundary_offset'])))+63)//64)
        self.terms=xp.zeros((2,nf));self.partial=xp.zeros(2*nb*self.tiles);self.stage=xp.zeros(2*nb)
        self.weights=xp.zeros(nf);self.total=xp.zeros(nb);self.trace=xp.zeros((3,nf));self.flux=xp.zeros((3,nf))
        self.status=xp.zeros(nf,dtype=xp.int32)
        self.section=None
        if any(v['method']=='inlet.critical_discharge' for v in hydraulic_identity.values()):
            from .river_exterior_section import CriticalSection
            self.section=CriticalSection(self,hydraulic_identity)
        self.normal_section=None
        if any(v['method']=='inlet.normal_discharge' for v in hydraulic_identity.values()):
            from .river_exterior_normal import NormalSection
            self.normal_section=NormalSection(self,hydraulic_identity)
        self.inlet_compatibility=None
        if np.any(hydraulic_mode==9):
            from .river_exterior_inlet import CompatibleInlet
            self.inlet_compatibility=CompatibleInlet(self,hydraulic_identity)
        self.critical_outlet=None
        if any(v['method']=='outlet.critical' for v in hydraulic_identity.values()):
            from .river_exterior_outlet import CriticalOutlet
            self.critical_outlet=CriticalOutlet(self,hydraulic_identity)
        self.free_weir=None
        if any(v['method']=='outlet.free_weir' for v in hydraulic_identity.values()):
            from .river_exterior_outlet import FreeWeir
            self.free_weir=FreeWeir(self,hydraulic_identity)
        self.normal_outlet=None
        if any(v['method']=='outlet.normal_depth' for v in hydraulic_identity.values()):
            from .river_normal_outlet import NormalOutlet
            self.normal_outlet=NormalOutlet(self,hydraulic_identity)
        self.rating_outlet=None
        if any(v['method']=='outlet.rating_curve' for v in hydraulic_identity.values()):
            from .river_rating_outlet import RatingOutlet
            self.rating_outlet=RatingOutlet(self,hydraulic_identity)
        # The inert geometry carrier's Manning coefficient is not the user's
        # selected law. Use the same resolved relation as hydraulic friction.
        self.roughness_mode=min(roughness_mode,3)
        self.floor=float(p.parameters.dry);self.waqua=int(roughness_mode==4)
        target=a
        while 'solver' in vars(target):target=target.solver
        self.owner=target
        target.exterior_bed=self
        previous_finish,previous_commit=p.finish,p.commit
        def finish(qold,qnew,water,**kw):
            result=previous_finish(qold,qnew,water,**kw)
            values=p.exterior_value
            if self.bed_values is not None:
                self.bed_schedule.sample(target.time,p._dt,self.bed_values);values=self.bed_values
            self.engine.launch('exterior_update',nf,nf,float(p._dt*p.morfac),p.exterior_mode,p.g['owner'],
                p.exterior_bed,p.z0,p.ztrial,water,values,p.exterior_trial)
            if not bool(xp.all(xp.isfinite(p.exterior_trial))):raise BendTrialRejected('Nonfinite exterior bed trial')
            return result
        def commit():
            previous_commit();p.exterior_bed[:]=p.exterior_trial
        p.finish=finish;p.commit=commit
        identity=dict(methods={str(k):v for k,v in methods.items()},source_sha256=digest(__file__),runtime=self.engine.identity,
            definition_sha256=digest(Path(__file__).parents[1]/'river_open_boundary.py'),
            hydraulics=hydraulic_identity,
            section_control=None if self.section is None else self.section.identity,
            normal_section_control=None if self.normal_section is None else self.normal_section.identity,
            inlet_compatibility=None if self.inlet_compatibility is None else self.inlet_compatibility.identity,
            critical_outlet_control=None if self.critical_outlet is None else self.critical_outlet.identity,
            free_weir_control=None if self.free_weir is None else self.free_weir.identity,
            hydraulic_schedule=dict(tables=schedule.document,
                source_sha256=digest(Path(__file__).parents[1]/'river_hydraulic_schedule.py'),
                clock_sha256=digest(Path(__file__).parents[1]/'river_schedule.py'),
                clock='physical seconds; actual midpoint residual time; accepted-start predictor and endpoint friction/capacity',
                state='immutable input tables; no cursor; resample on every evaluation'),
            bed_schedule=dict(tables=self.bed_schedule.document,
                source_sha256=digest(Path(__file__).parents[1]/'river_exterior_schedule.py'),
                clock='physical endpoint elevation; outer midpoint rate times morphological dt; split at knots',
                state='immutable inputs; accepted exterior used during the decoupled physical step'),
            prescribed_section='selected external reference water line; exact net Q from boundary Riemann fluxes with common virtual exterior velocity; actual backwater/sonic/shock trace resolved, incoming tangent zero and outgoing tangent advected',
            stage_policies={str(k):v for k,v in policies.items()},
            stage_wave='outward_entropy_1wave_hugoniot_characteristic_v1; incoming finite bore admitted across supercritical outflow; sonic rarefaction sampling',
            incoming_stage_tangent='zero for explicitly permitted subcritical backflow',
            stage_inundation='backflow policy rejects a wet prescribed stage over a dry exterior trace; separate inundation binding pending',
            initial_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
            forcing_sha256=hashlib.sha256(v.tobytes()).hexdigest(),
            roughness=dict(method=roughness['method'],parameters=dict(roughness['parameters']),
                implementation_sha256=digest(Path(__file__).with_name('river_roughness.py'))),
            scope='hydraulic exterior-state integration with physical-overlap AMR history; public admission pending',
            ownership='zero-area exterior state; no interior bed clamp or direct material source',
            outward='free increment at accepted local outward water flux',public_support=False)
        if self.normal_outlet is not None:identity['normal_outlet_control']=self.normal_outlet.identity
        if self.rating_outlet is not None:identity['rating_outlet_control']=self.rating_outlet.identity
        target.identity['exterior_bed']=identity
        schedule.bind(a)
        self.bed_schedule.bind(a,'exterior_bed')
        if a.backend=='cpu':
            h=a.hydraulics;ptr=ctypes.c_void_p
            h.lib.shoreline_native_boundary_rates.argtypes=[ptr,ptr,ctypes.c_double,ctypes.c_double,
                ctypes.c_int,ptr,ptr,ptr,ptr,ctypes.POINTER(ctypes.c_int)]
            h.lib.shoreline_native_boundary_rates.restype=None

    def reduce(self,values,fields,result):
        p=self.p;n=self.nb*fields
        self.engine.launch('exterior_partial',n*self.tiles,n*self.tiles,p.nf,self.nb,self.tiles,fields,
            self.offset,self.faces,values,self.partial)
        self.engine.launch('exterior_totals',n,n,self.tiles,self.partial,result)

    def evaluate(self,q,z,*,time=None):
        p=self.p;g=p.g;nf=p.nf;n=p.n;e=self.engine;dry=p.parameters.dry;grav=p.parameters.gravity
        if self.bedforms_enabled and self.bedforms is None:
            raise ValueError('Exterior bedform roughness requires the exterior material owner')
        if self.schedule.tables:
            # Before begin the predictor/friction uses accepted start time;
            # open physical endpoint consumers use the outer interval end.
            # Midpoint residuals supply their own (possibly bisected) clock.
            at=self.owner.time+(p._dt if p._open else 0.) if time is None else time
            self.schedule.sample(float(at),self.forcing)
        if self.section is not None:self.section.solve()
        if self.normal_section is not None:self.normal_section.solve()
        e.launch('exterior_stage_terms',nf,nf,n,dry,g['owner'],self.selected,q,z,g['length'],self.terms)
        self.reduce(self.terms,2,self.stage)
        if self.bedforms is not None and not self.bedforms.initializing:
            self.bedforms.weights(q,z)
        else:
            e.launch('exterior_weights',nf,nf,n,self.nb,dry,grav,self.roughness_mode,self.floor,self.waqua,
                g['owner'],g['bid'],self.selected,self.hydraulic_mode,q,z,p.exterior_bed,g['length'],
                self.roughness,self.stage,self.forcing,self.weights)
        self.reduce(self.weights,1,self.total)
        e.launch('exterior_flux',nf,nf,n,self.nb,grav,dry,float(self.a.data.get('speed_limit',50.)),
            g['owner'],g['bid'],self.selected,self.hydraulic_mode,self.stage_policy,q,z,p.exterior_bed,g['normal'],g['length'],
            self.forcing,self.weights,self.total,self.trace,self.flux,self.status)
        if self.inlet_compatibility is not None:self.inlet_compatibility.evaluate(q,z)
        if self.critical_outlet is not None:self.critical_outlet.evaluate(q,z)
        if self.free_weir is not None:self.free_weir.evaluate(q,z)
        if self.normal_outlet is not None:self.normal_outlet.evaluate(q,z)
        if self.rating_outlet is not None:self.rating_outlet.evaluate(q,z)
        if bool(self.xp.any(self.status)):raise BendTrialRejected('Exterior hydraulic boundary rejected')
        return self.flux

    def apply_gpu_traces(self,q,trace):
        # The old closure must not reject an unrelated interior-bed section.
        # All open boundaries belong to this explicit exterior-state owner.
        h=self.a.hydraulics;b=h.boundary;saved=b.arrays['mode'].copy()
        try:
            b.arrays['mode'][:]=0;b.apply(q,trace)
        finally:b.arrays['mode'][:]=saved
        trace[:3,self.face_indices]=self.trace[:,self.face_indices]

    def apply_gpu_flux(self,fl,fr,status):
        fl[:,self.face_indices]=self.flux[:,self.face_indices]
        fr[:,self.face_indices]=self.flux[:,self.face_indices]
        status[self.face_indices]=self.status[self.face_indices]
