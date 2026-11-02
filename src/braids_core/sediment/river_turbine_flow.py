"""Covered-rotor hydraulic extension derived from qualified HKT-3l."""
from contextlib import contextmanager
import ctypes,hashlib,json,math,subprocess
from pathlib import Path
from types import MethodType
import numpy as np
from shapely.geometry import Polygon,LineString
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment.river_hydrokinetic_stress import FullStressEngine
from . import river_turbine_geometry as polygon

HERE=Path(__file__).resolve().parent


def instrumented_native(original):
    library,parent,compiler=original();code=(library.parent/'shoreline.f90').read_text();marker='\ncontains\n'
    assert code.count(marker)==1;code=code.replace(marker,'\n real(8),allocatable::hkt_surface_trace(:,:)\n'+marker)
    start=code.index(' subroutine bend_native_spatial_rates(');end=code.index('\nend subroutine',start);section=code[start:end]
    marker='  if(any(workspace_shape/=';assert section.count(marker)==1
    section=section.replace(marker,'''  if(.not.allocated(hkt_surface_trace)) allocate(hkt_surface_trace(10,nf))
  if(size(hkt_surface_trace,2)/=nf) then
   deallocate(hkt_surface_trace);allocate(hkt_surface_trace(10,nf))
  endif
  hkt_surface_trace=0.d0
'''+marker)
    marker='   ! Constant-surface wet/dry traces integrate pressure up to the shoreline:';assert section.count(marker)==1
    section=section.replace(marker,'''   if(n>0) then
    hkt_surface_trace(1,f)=left(1);hkt_surface_trace(2,f)=right(1)
    if(l(1)>0.d0) then
     hkt_surface_trace(3,f)=dot_product(l(2:3),normal(:,f))/l(1)
     hkt_surface_trace(5,f)=sqrt(sum(l(2:3)**2))/l(1)
    endif
    if(r(1)>0.d0) then
     hkt_surface_trace(4,f)=dot_product(r(2:3),normal(:,f))/r(1)
     hkt_surface_trace(6,f)=sqrt(sum(r(2:3)**2))/r(1)
    endif
    hkt_surface_trace(7,f)=left(bed_primitive_index());hkt_surface_trace(8,f)=right(bed_primitive_index())
    hkt_surface_trace(9,f)=merge(1.d0,0.d0,hydro_constant(c))
    hkt_surface_trace(10,f)=merge(1.d0,0.d0,hydro_constant(n))
   endif
'''+marker)
    code=code[:start]+section+code[end:];index=code.rindex('\nend module')
    code=code[:index]+'''
subroutine hkt_get_surface_trace(out) bind(C)
 use iso_c_binding
 real(c_double),intent(out)::out(10*nf)
 integer::k
 do k=1,10
  out((k-1)*nf+1:k*nf)=hkt_surface_trace(k,:)
 enddo
end subroutine
'''+code[index:]
    key=hashlib.sha256((digest(library)+code).encode()).hexdigest()[:20];folder=HERE.parents[1]/'build/hkt_surface_inspection'/key;folder.mkdir(parents=True,exist_ok=True)
    target=folder/'hydraulics.dll';generated=folder/'shoreline.f90'
    if not target.exists():
        from tools.build_cutcell import SOURCES
        from braids_core.sediment.wall_bed_hydraulics import build
        parent_library,_,_=build();deps=[parent_library.parent/'source'/Path(name).name for name in SOURCES[:-1]];generated.write_text(code)
        command=[compiler,'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',*map(str,deps),str(generated),'-o',str(target),'-Wl,--export-all-symbols']
        r=subprocess.run(command,cwd=folder,text=True,capture_output=True,timeout=180);(folder/'build.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError(str(folder/'build.log'))
        (folder/'identity.json').write_text(json.dumps(dict(parent=parent,generated_source_sha256=digest(generated),library_sha256=digest(target),change='Read-only exported free-surface traces; physical native residual unchanged'),indent=2))
    identity=json.loads((folder/'identity.json').read_text());assert digest(target)==identity['library_sha256']
    return target,identity,compiler


@contextmanager
def native_context(backend):
    if backend=='gpu':yield;return
    from braids_core.sediment import shoreline_solver as shoreline,bend_native_hydraulics as native
    with native._lock:
        old=shoreline.build_native
        try:shoreline.build_native=lambda:instrumented_native(old);yield
        finally:shoreline.build_native=old


class Engine(FullStressEngine):
    SOURCE=Path(__file__).with_suffix('.h')
    NAMES=('hkt_wet_gradients','hkt_surface_pack','hkt_balanced_pressure')


class CoveredRotor:
    def __init__(self,cells,c,backend,dry=1e-6,*,geometry=None):
        if backend not in ['cpu','gpu']:raise ValueError('Explicit CPU/GPU required')
        self.xp=np if backend=='cpu' else __import__('cupy');xp=self.xp;self.c=dict(c);self.dry=dry
        if 'hub_elevation_m' not in c or not math.isfinite(c['hub_elevation_m']):raise ValueError('Explicit fixed hub elevation required')
        for name in ['minimum_bed_clearance_m','minimum_surface_clearance_m']:
            if name not in c or isinstance(c[name],bool) or not math.isfinite(c[name]) or c[name]<0:raise ValueError('Explicit nonnegative turbine clearance limits required')
        self.minimum_bed_clearance=c['minimum_bed_clearance_m'];self.minimum_surface_clearance=c['minimum_surface_clearance_m']
        self.top=c['hub_elevation_m']+c['D']/2;self.bottom=c['hub_elevation_m']-c['D']/2
        geometry=geometry or polygon.GeometryIndex(cells)
        weights,self.geometry=polygon.polygon_integrals(cells,c,geometry=geometry);self.weights=xp.asarray(weights);self.normalized=self.weights/weights.sum();self.area=xp.asarray(geometry.area);self.ids=xp.asarray(np.flatnonzero(weights>0))
        self.normal=np.array(c.get('normal_xy',[1.,0.]));tangent=np.array([-self.normal[1],self.normal[0]])
        for i in np.flatnonzero(weights>0):
            if np.ptp(np.asarray(cells[i].vertices_xy)@tangent)>c['D']*(1+1e-9):raise ValueError('Merged turbine footprint is unresolved across the rotor; reduce the smallest mesh size or move the turbine away from nearby constraints.')
        self.xy=geometry.xy;center=np.array([c['X'],c['Y']]);line=LineString([center-c['D']/2*tangent,center+c['D']/2*tangent])
        rows=[];displacements=[]
        for i in geometry.query(line):
            part=geometry.polygons[i].intersection(line)
            parts=[part] if part.geom_type=='LineString' else list(getattr(part,'geoms',()))
            for p in parts:
                if p.geom_type=='LineString' and p.length>1e-12:
                    for point in [p.coords[0],p.coords[-1]]:rows.append(i);displacements.append(np.array(point)-self.xy[i])
        if not rows:raise ValueError('Rotor plane is not represented')
        self.plane=xp.asarray(rows,dtype=xp.int32);self.delta=xp.asarray(np.array(displacements).T)
        self.tolerance=max(1e-9,64*np.finfo(float).eps*max(1.,abs(self.top),abs(self.bottom)))
        self.identity=dict(method='actuator.otf_actual_depth_clearance_rotor_v2',physical=c,
            hub='fixed absolute elevation, never moved with water or bed',activation='minimum actual surface and bed gaps meet their explicit limits; fully covered, no bed contact, whole numerical support wet',
            sampling='minimum of plane endpoint LS surfaces and incident CV means; maximum of reconstructed and mean bed',
            off='zero actuator thrust/useful power/work; automatic reactivation when covered, no hysteresis or remembered trial state',
            tolerance_m=self.tolerance,depth='current bump-weighted physical depth in Kramer-Piggott correction',
            geometry_parent=digest(polygon.__file__),wrapper=digest(__file__),parked_blades_and_support_drag=False)

    def sample(self,q,z,grad):
        xp=self.xp;c=self.c;ids=self.ids;h=q[0,ids];row=self.plane;eta=q[0,row]+z[row]
        surface=eta+grad[4,row]*self.delta[0]+grad[5,row]*self.delta[1]
        bed=z[row]+grad[6,row]*self.delta[0]+grad[7,row]*self.delta[1]
        margin=float(xp.min(xp.minimum(surface,eta)))-self.top
        clearance=self.bottom-float(xp.max(xp.maximum(bed,z[row])))
        reason='covered'
        if margin<self.tolerance:reason='insufficient_depth'
        elif clearance<self.tolerance:reason='bed_intersection'
        elif margin+self.tolerance<self.minimum_surface_clearance:reason='insufficient_surface_clearance'
        elif clearance+self.tolerance<self.minimum_bed_clearance:reason='insufficient_bed_clearance'
        elif bool(xp.any(h<=self.dry)):reason='dry_numerical_support'
        active=reason=='covered';u=xp.where(h>self.dry,q[1,ids]/xp.maximum(h,self.dry),0.);v=xp.where(h>self.dry,q[2,ids]/xp.maximum(h,self.dry),0.)
        w=self.normalized[ids];un=u*self.normal[0]+v*self.normal[1];u1=float(xp.sum(w*un));depth=float(xp.sum(w*h));A=np.pi*c['D']**2/4
        # This is a directed, zero-yaw rotor for every control mode. Previously
        # only the RPM resolver shut down at reversed/absent approach flow.
        if active and u1<=1e-10:
            active=False;reason='reverse_flow' if u1<0 else 'no_incoming_flow'
        control=c.get('control',{});ct=c['ct'];cp=c['cp'];tsr=control.get('tip_speed_ratio');rpm=control.get('rotational_speed_rpm')
        if active and control.get('mode') in ('fixed_rpm_table','fixed_rpm_surface'):
            if control['mode']=='fixed_rpm_surface':
                from braids_core.river_turbine_surface import operating_point
            else:
                from braids_core.river_turbine_performance import operating_point
            point=operating_point(control,c['D'],depth,u1);active=point['active']
            if active:ct=point['ct'];cp=point['cp'];tsr=point['tip_speed_ratio']
            else:reason=point['reason']
        coefficient=xp.zeros_like(h);u0=u1;F=P=work=0.
        if active:
            beta=A*ct/(c['D']*depth)
            if not 0<=beta<1:raise ValueError('Covered rotor has invalid numerical blockage')
            alpha=.5*(1+math.sqrt(1-beta));u0=u1/alpha
            coefficient=(.5*A*ct/alpha**2)*w/self.area[ids]
            speed=xp.hypot(u,v);F=float(xp.sum(coefficient*self.area[ids]*speed*un));work=float(xp.sum(coefficient*self.area[ids]*speed*(u*u+v*v)))
            P=F*.5*(1+math.sqrt(1-ct))*u0 if cp is None else .5*A*cp*abs(u0)**3
            if control.get('mode')=='prescribed_tsr_table':rpm=tsr*abs(u0)/(c['D']/2)*30/math.pi
        diagnostic=dict(turbine_id=c['name'],active=active,inactive_reason=None if active else reason,
            control_mode=control.get('mode','constant_ct'),tip_speed_ratio=tsr,rotational_speed_rpm=rpm if active else (0. if rpm is not None else None),
            commanded_rotational_speed_rpm=control.get('rotational_speed_rpm'),
            performance_model=control.get('model_name','User prescribed coefficients'),power_basis='rotor mechanical shaft power; generator losses excluded',
            minimum_coverage_margin_m=margin,minimum_bed_clearance_m=clearance,rotor_top_elevation_m=self.top,
            required_surface_clearance_m=self.minimum_surface_clearance,required_bed_clearance_m=self.minimum_bed_clearance,
            surface_clearance_reserve_m=margin-self.minimum_surface_clearance,bed_clearance_reserve_m=clearance-self.minimum_bed_clearance,
            local_velocity_m_s=u1,upstream_velocity_m_s=u0,mean_depth_m=depth,requested_Ct=c['ct'],requested_Cp=c['cp'],
            thrust_coefficient=ct if active else 0.,power_coefficient=cp if active else 0.,
            force_per_density_m4_s2=F,useful_power_per_density_m5_s3=P,hydraulic_power_removed_per_density_m5_s3=work)
        return diagnostic,coefficient,u,v

    def apply(self,q,z,grad,rate):
        s,k,u,v=self.sample(q,z,grad);speed=self.xp.hypot(u,v)
        rate[1,self.ids]-=k*speed*u;rate[2,self.ids]-=k*speed*v
        return [s]


class RotorArray:
    def __init__(self,cells,rows,backend):
        geometry=polygon.GeometryIndex(cells)
        self.rotors=tuple(CoveredRotor(cells,c,backend,geometry=geometry) for c in rows)
        self.identity=dict(method='hkt.covered_rotor_array_v1',rotors=[r.identity for r in self.rotors],
            coupling='simultaneous source superposition on the same hydraulic state; ambient stress and pressure correction applied once')

    def apply(self,q,z,grad,rate):
        return [d for rotor in self.rotors for d in rotor.apply(q,z,grad,rate)]

    def sample(self,q,z,grad):return [r.sample(q,z,grad)[0] for r in self.rotors]


def install(a,cells,rows,root):
    p=a.physics;xp=a.xp;g=p.g;root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if a.time or a.steps or p._open or a.morphology:raise ValueError('Fresh hydraulic owner required')
    if not rows or len({r['nu'] for r in rows})!=1:raise ValueError('A turbine array requires one explicitly shared ambient viscosity.')
    from .river_turbine_inflow import make_array_class
    source_class=make_array_class(RotorArray) if any(r['control'].get('mode')=='fixed_rpm_surface' for r in rows) else RotorArray
    e=Engine(a.backend,p.flow.threads);stress=FullStressEngine(a.backend,p.flow.threads);source=source_class(cells,rows,a.backend);a.turbine_source=source
    grad=xp.zeros((10,p.n));raw=xp.zeros((3,p.n));trace=xp.zeros((10,p.nf));face=xp.zeros((2,p.nf));work=xp.zeros((3,p.n));status=xp.zeros(p.nf,dtype=xp.int32);ones=xp.ones(p.n);nu=xp.full(p.n,rows[0]['nu'])
    if a.backend=='cpu':getter=a.hydraulics.lib.hkt_get_surface_trace;getter.argtypes=[ctypes.c_void_p];getter.restype=None
    previous=p.add_momentum
    def gradients(q,z):
        e.launch('hkt_wet_gradients',p.n,p.n,float(p.parameters.dry),q,z,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],grad)
    def gather(rate):
        work.fill(0.);p.flow.launch('flow_momentum_gather',p.n,p.n,p.nf,ones,face,g['offset'],g['faces'],g['signs'],g['length'],g['area'],work);rate[1:]+=work[1:]
    def add(self,q,water,rate):
        previous(q,water,rate);gradients(q,self.z0)
        stress.launch('hkt_full_strain_faces',p.nf,p.nf,p.n,float(p.parameters.dry),q,self.z0,grad,nu,g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],face);gather(rate)
        raw.fill(0.);self.turbine_diagnostics=source.apply(q,self.z0,grad,raw);rate[1:]+=raw[1:]
        if a.backend=='cpu':getter(trace.ctypes.data)
        else:
            f=a.hydraulics.flux;e.launch('hkt_surface_pack',p.nf,p.nf,p.n,f.buffers['local'],g['owner'],g['neighbour'],f.inputs['hydro_constant'],trace)
        e.launch('hkt_balanced_pressure',p.nf,p.nf,p.n,float(p.parameters.gravity),float(p.parameters.dry),q,self.z0,trace,grad,raw,g['owner'],g['neighbour'],g['normal'],g['distance'],g['displacement'],face,status);gather(rate)
    p.add_momentum=MethodType(add,p)
    def diagnose():gradients(a.q,a.z);return source.sample(a.q,a.z,grad)
    a.hkt_diagnose=diagnose;a.hkt_gradient=grad;a.hkt_status=status
    identity=dict(method='hkt.common_bed_surface_perturbation_hkt3l_v1',source=source.identity,engine=e.identity,stress=stress.identity,
        nu_m2_s=rows[0]['nu'],scope='zero-yaw rotor array, fixed individual orientations; dry regions and nonflat fixed bed; no sediment',
        balance='correction vanishes at constant free surface and zero flow/source; original augmented f-wave bed balance retained',
        shore='wet-only LS gradients; no viscosity across dry faces; pressure correction zero outside wet/common-depth small-surface-jump scope',
        base='same augmented f-wave, mass/advective flux, within-cell bed quadrature and SSPRK22 on every face',wrapper=digest(__file__),desktop_admission='hkt4_hydrodynamics_only')
    a.identity['hkt_terrain']=identity;p.identity['hkt_terrain']=identity
    return a
