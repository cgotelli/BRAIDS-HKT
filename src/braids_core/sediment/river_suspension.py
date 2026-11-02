"""Continuous noncohesive suspension on the finite-layer river owner.

Resolved capacity with separate bed/suspended channels, SettleFrm=2 and
explicit Galappatti or deposition-only ownership on the shared explicit
worker. Scalar donor-cell transport is deliberately distinct from the
unchanged second-order bed reconstruction. Public admission is checked by
the strict river-project resolver.
"""
from pathlib import Path
from types import MethodType
import numpy as np
from .river_mpm import RiverMPM, install_mpm, wave_source
from .bend_methods_v1 import BendTrialRejected, digest
from .bed_columns_v101 import host
from .river_morfac import physical_stock, accelerate_wet_exchange
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('split_capacity','sus_faces','sus_gradients','sus_nonorthogonal','sus_limit','sus_scale','sus_gather','sus_exchange','sus_check')


class SplitCapacity(RiverMPM):
    def __init__(self,*args,bed_factor=1.,**kwargs):
        super().__init__(*args,**kwargs)
        self.bed_factor=float(bed_factor)
        self.engine=WaveEngine(self.backend,self.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.identity.update(method='FLOW_tranb4_general_split',bed_factor=self.bed_factor,
            bed_parameters=[8.,0.,1.5,1.,.047],suspended_parameters=[2.5,2.,1.,.9,.035],
            runtime=self.engine.identity,wrapper_sha256=digest(__file__),hiding='none',shear='total_Shields')

    def evaluate(self,q,hi,*,capacity_q=None,chezy=None):
        xp=self.xp;cq=q if capacity_q is None else capacity_q;n=q.shape[1]
        c=xp.full(n,self.c) if chezy is None else chezy
        for v,shape in ((q,(3,n)),(cq,(3,n)),(c,(n,)),(hi,(n,))):
            if v.shape!=shape or v.dtype!=xp.float64 or not v.flags.c_contiguous:
                raise ValueError('Finite resident FP64 split transport inputs required')
        out=xp.zeros((6,n))
        status=xp.empty(n,dtype=xp.int32)
        # All original finite/domain checks are performed in the parallel
        # capacity kernel, before division. One status synchronization replaces
        # seven independent scan/reduction/synchronization sequences on CUDA.
        self.engine.launch('split_capacity',n,n,self.d,self.bed_factor,q,cq,c,hi,out,status)
        if bool(xp.any(status)):
            raise ValueError('Invalid or nonfinite split transport depth/Chezy/inputs')
        return out


def install_suspension(a,folder,*,initial_concentration,boundary_concentration,
                       bed_factor=1.,sus_factor=1.,diffusivity=0.,temperature=20.,cref=1600.,enabled=True,
                       capacity=None,hiding='none',corrections=None):
    """Compose before banks/collapse, with one accepted column transaction.

    Positive bed exchange deposits material. Its opposite enters suspended inventory
    on hydraulic time, and the extra MF bed change is separately ledgered.
    Dry stranding is a physical conservative transfer, not MF entrainment.
    As in FLOW's Sus factor, sus_factor=0 suppresses entrainment, not settling
    or transport of an existing suspended inventory. It never adds that
    capacity to bedload. With zero initial/incoming concentration this is the
    corresponding bedload-only control of the same physical bundle.
    """
    p=a.physics;xp=a.xp;c=a.columns;ng=len(a.methods.diameter_m);n=p.n;nf=p.nf
    if a.time or a.steps or p._open:raise ValueError('Install suspension before execution')
    if hasattr(p,'suspended_mass'):raise ValueError('Suspension already installed')
    if hasattr(a,'mpm_installed'):raise ValueError('Install split bundle on EH carrier, not another capacity bundle')
    if not 1<=ng<=32 or p.morfac<=0:raise ValueError('Positive MF and 1..32 noncohesive classes required')
    if (not np.isfinite([bed_factor,sus_factor,diffusivity,temperature,cref]).all()
        or bed_factor not in (0.,1.) or sus_factor not in (0.,1.) or diffusivity<0 or temperature<=-20 or cref<=0):
        raise ValueError('Invalid split suspension controls')
    ci=np.asarray(initial_concentration,float);bc=np.asarray(boundary_concentration,float)
    if ci.shape!=(ng,n) or bc.shape!=(ng,nf) or not np.isfinite(ci).all() or not np.isfinite(bc).all() or np.any(ci<0) or np.any(bc<0):
        raise ValueError('Explicit nonnegative class concentrations required')
    if type(enabled) is not bool or (not enabled and (sus_factor or diffusivity or np.any(ci) or np.any(bc))):
        raise ValueError('Disabled suspension requires zero concentration, entrainment and diffusion')
    settling_only=corrections is not None and corrections[0]["suspension"]=="suspension.settling_only"
    supplied_params=None
    if settling_only:
        from .river_supplied_suspension import validate as validate_supplied
        supplied_params=dict(validate_supplied(corrections[1]['supplied_suspension']))
        if ng!=1:raise ValueError('Supplied suspension requires one bulk material')
    gp=corrections is not None and corrections[0]['suspension']=='suspension.garcia_parker_1991'
    dl=corrections is not None and corrections[0]['suspension']=='suspension.de_leeuw_2020'
    from .river_classic_suspension import METHODS,ClassicSuspension
    classic=corrections is not None and corrections[0]['suspension'] in METHODS
    classic_operator=None
    if classic:
        if ng!=1 or not enabled or sus_factor!=1.:
            raise ValueError('S6 requires one bulk material and active entrainment')
        classic_operator=ClassicSuspension(a.backend,p.flow.threads,corrections[1]['classic_suspension'])
    gp_operator=None
    if gp:
        from .river_garcia_parker import GarciaParker
        if ng!=1 or not enabled or sus_factor!=1.:
            raise ValueError('GP1991 requires one bulk material and active entrainment')
        gp_operator=GarciaParker(a.backend,p.flow.threads,corrections[1]['garcia_parker'])
    dl_operator=None
    if dl:
        from .river_de_leeuw import DeLeeuw
        if ng!=1 or not enabled or sus_factor!=1.:
            raise ValueError('de Leeuw 2020 requires one bulk material and active entrainment')
        dl_operator=DeLeeuw(a.backend,p.flow.threads,corrections[1]['de_leeuw'])
    if settling_only and (capacity is None or not enabled or sus_factor):
        raise ValueError('Settling-only requires an explicit capacity, active suspension and zero entrainment')
    h=host(a.q[0])
    if np.any(ci[:,h<=p.parameters.dry]):raise ValueError('Dry initial cells cannot contain suspended concentration')
    # Reuse the already qualified threshold derivative, but replace its physical
    # ripple factor by general-formula RipFac=1 in BOTH capacity and Jacobian.
    previous_faces=a.bed_faces
    install_mpm(a,folder,hiding='none')
    if capacity is None:
        a.capacity=[SplitCapacity(a.backend,diameter=d,chezy=p.chezy,threads=p.flow.threads,bed_factor=bed_factor) for d in a.methods.diameter_m]
    else:
        from .river_capacity import ResolvedRiverCapacity
        a.capacity=[ResolvedRiverCapacity(a.backend,diameter=d,chezy=p.chezy,threads=p.flow.threads,
            bed_factor=bed_factor,capacity=capacity,folder=folder,solid_density=rho,
            water_density=a.material.definition.water_density,viscosity=4.e-5/(20.+temperature),
            reference=corrections[1]["van_rijn_reference"] if capacity["method"]=="capacity.van_rijn_1993" else None,
            skin_shear=(corrections[1]["skin_shear"]
                        if corrections[0]["shear"]=="shear.soulsby_clarke_skin" else None),
            transport='bedload' if gp or dl or classic or settling_only else corrections[0]["transport"])
            for d,rho in zip(a.methods.diameter_m,a.material.definition.solid_density)]
    # General RipFac=1 has no D90/ripple calculation. Reuse the derivative's
    # call interface without running the unrelated mixture-percentile solve.
    a.bed_faces=previous_faces;p.mpm_d90.fill(.001)
    for k,cap in enumerate(a.capacity):cap.d90=p.mpm_d90;cap.hiding=p.mpm_hiding[k]
    if capacity is None:
        code=wave_source().replace('double mu=fmin(1.,pow(cwork/grain,1.5));','double mu=1.;')
        code=code.replace('double pref=8.*sqrt(9.81*1.65*diameter)*diameter;',f'double pref={bed_factor:.1f}*8.*sqrt(9.81*1.65*diameter)*diameter;')
    else:
        from .river_capacity import wave_source as selected_wave_source
        code=selected_wave_source(intrinsic=a.capacity[0].needs_intrinsic_mixture,
                                  vector=a.capacity[0].owns_vector_direction,
                                  vr1993=a.capacity[0].is_vr1993,total=a.capacity[0].algebraic_suspension,
                                  smart=a.capacity[0].is_smart_jaggi,
                                  physical_drag=a.capacity[0].uses_physical_drag and not a.capacity[0].is_smart_jaggi)
        if corrections is None:
            raise ValueError('Selected capacity requires explicit slope/hiding binding')
    from . import class_wave_cache
    cached=hasattr(p.wave_engine,'_class_shared')
    if cached:code=class_wave_cache.source(code)
    from braids_core.windows_paths import filesystem_path
    target=filesystem_path(folder)/'split_wave.h';target.write_text(code)
    wave=WaveEngine(a.backend,p.flow.threads,source=target,
        names=class_wave_cache.NAMES if cached else ('wave_state_gradients','matrix_bed_faces'))
    p.wave_engine.fn=wave.fn;p.wave_engine.identity=wave.identity
    if cached:class_wave_cache.bind(p.wave_engine)
    if a.backend=='cpu':p.wave_engine.lib=wave.lib;p.wave_engine.dirs=wave.dirs
    else:p.wave_engine.module=wave.module
    if capacity is not None:
        prior_wave_launch=p.wave_engine.launch
        def selected_launch(name,count,*args):
            if name=='matrix_bed_faces':
                if p.eh.needs_intrinsic_mixture:
                    args=(*args,p.eh.grain_statistics,p.eh.hydraulic_chezy)
                elif p.eh.uses_physical_drag:
                    args=(*args,p.eh.hydraulic_chezy)
                args=(*args,p.eh.wire,a.columns.sv[0])
                if p.eh.algebraic_suspension:
                    args=(*args,p.eh.surface_fraction,p.eh.suspended_diameter)
            return prior_wave_launch(name,count,*args)
        p.wave_engine.launch=selected_launch
    engine=a.capacity[0].engine;diam=xp.asarray(a.methods.diameter_m)
    supplied_diam=xp.asarray([supplied_params['suspended_diameter_m']]) if settling_only else None
    # Immutable class-major parameter rows. Passing the first class's wire to
    # every exchange calculation would give all grains its relative density.
    exchange_wires=(xp.stack([cap.wire for cap in a.capacity]) if capacity is not None else None)
    # Topology is immutable for this owner. Remesh constructs a new owner;
    # only this boundary mask is reused, never an evolving physical field.
    outside=(p.g['neighbour']<0)&(p.g['bid']>=0)
    check_status=xp.empty(n,xp.int32)
    concentration_gradient=xp.empty((ng,2,n)) if enabled and diffusivity else None
    p.suspended_mass=xp.asarray(ci)*a.q[0];p.suspended_trial=p.suspended_mass.copy()
    vr1993=capacity is not None and capacity["method"]=="capacity.van_rijn_1993"
    if vr1993:
        ds=supplied_params['suspended_diameter_m'] if settling_only else corrections[1]["van_rijn_reference"]["suspended_diameter_m"]
        p.suspended_diameter=xp.full((ng,n),ds)
        p.suspended_diameter_trial=p.suspended_diameter.copy()
    if capacity is not None:
        for k,cap in enumerate(a.capacity):
            if cap.algebraic_suspension:
                cap.surface_fraction=c.surface_fraction[k]
                cap.suspended_diameter=p.suspended_diameter[k] if vr1993 else xp.full(n,cap.d)
    p.suspended_advected=xp.zeros((ng,n));p.suspended_scale=xp.ones((ng,n))
    p.suspended_flux=xp.zeros((ng,nf));p.suspended_boundary=xp.asarray(bc)
    p.suspended_boundary_ledger=xp.zeros((ng,nf));p.suspended_exchange_trial=xp.zeros((ng,n))
    p.suspended_exchange_cumulative=xp.zeros((ng,n));p.suspended_acceleration=xp.zeros((ng,n))
    p.suspended_acceleration_trial=xp.zeros((ng,n));p.suspended_diagnostic=xp.zeros((3,ng,n))
    p.suspended_initial=(c.mass.sum(axis=0)+p.suspended_mass)*p.g['area']
    previous_commit=p.commit
    def commit(owner):
        previous_commit()
        owner.suspended_mass[:]=owner.suspended_trial
        if vr1993:owner.suspended_diameter[:]=owner.suspended_diameter_trial
        owner.suspended_boundary_ledger-=p._dt*p.suspended_flux*p.g['length']*outside
        owner.suspended_exchange_cumulative+=owner.suspended_exchange_trial*p.g['area']
        owner.suspended_acceleration+=owner.suspended_acceleration_trial*p.g['area']
    p.commit=MethodType(commit,p)
    def exchange(owner,qold,qnew,water):
        if not enabled:
            # No advection, settling, entrainment or bed exchange. Retain inert
            # zero inventory arrays for the common restart/remesh contract.
            return p.suspended_exchange_trial
        g=p.g;dt=p._dt
        engine.launch('sus_faces',nf,nf,n,ng,float(diffusivity),p.parameters.dry,g['owner'],g['neighbour'],g['bid'],
            g['length'],g['distance'],qold[0],p.suspended_mass,water,p.suspended_boundary,p.suspended_flux)
        if concentration_gradient is not None:
            engine.launch('sus_gradients',n,n,ng,p.parameters.dry,g['offset'],g['faces'],g['owner'],g['neighbour'],
                g['xy'],qold[0],p.suspended_mass,concentration_gradient)
            engine.launch('sus_nonorthogonal',nf,nf,n,ng,float(diffusivity),p.parameters.dry,g['owner'],g['neighbour'],
                g['xy'],g['normal'],g['distance'],qold[0],concentration_gradient,p.suspended_flux)
        engine.launch('sus_limit',n,n,nf,ng,dt,g['offset'],g['faces'],g['owner'],g['length'],g['area'],p.suspended_mass,p.suspended_flux,p.suspended_scale)
        engine.launch('sus_scale',nf,nf,n,ng,g['owner'],g['neighbour'],p.suspended_scale,p.suspended_flux)
        engine.launch('sus_gather',n,n,nf,ng,dt,g['offset'],g['faces'],g['owner'],g['length'],g['area'],p.suspended_mass,p.suspended_flux,p.suspended_advected)
        if float(xp.min(p.suspended_advected)) < -1e-12:raise BendTrialRejected('Suspended donor budget failed')
        # Avoid silently clipping a material scalar deficit. Only FP roundoff.
        p.suspended_advected[:]=xp.maximum(p.suspended_advected,0.)
        if gp or dl or classic:
            stock=physical_stock(xp,c.available,p.class_delta,p.morfac)
        if gp:
            rho=float(a.material.definition.solid_density[0])
            gp_operator.engine.launch('gp_exchange',n,n,dt,p.parameters.dry,
                float(gp_operator.parameters['suspended_diameter_m']),rho,
                rho/float(a.material.definition.water_density)-1.,float(temperature),
                qnew,p.effective_fields[2],p.suspended_advected,stock,gp_operator.quadrature,
                p.suspended_trial,p.suspended_exchange_trial,p.suspended_diagnostic)
            if vr1993:p.suspended_diameter_trial[:]=p.suspended_diameter
        elif dl:
            rho=float(a.material.definition.solid_density[0])
            dl_operator.engine.launch('dl_exchange',n,n,dt,p.parameters.dry,
                float(dl_operator.parameters['suspended_diameter_m']),
                float(dl_operator.parameters['bed_d84_m']),rho,
                rho/float(a.material.definition.water_density)-1.,float(temperature),
                qnew,p.effective_fields[2],p.suspended_advected,stock,dl_operator.quadrature,
                p.suspended_trial,p.suspended_exchange_trial,p.suspended_diagnostic)
            if vr1993:p.suspended_diameter_trial[:]=p.suspended_diameter
        elif classic:
            rho=float(a.material.definition.solid_density[0])
            classic_operator.exchange(n,dt,p.parameters.dry,rho,
                rho/float(a.material.definition.water_density)-1.,float(temperature),
                qnew,p.effective_fields[2],p.suspended_advected,stock,
                p.suspended_trial,p.suspended_exchange_trial,p.suspended_diagnostic)
            if vr1993:p.suspended_diameter_trial[:]=p.suspended_diameter
        elif settling_only:
            engine.launch('settling_only_exchange',n,n,ng,exchange_wires.shape[1],dt,p.morfac,
                p.parameters.dry,float('inf'),float(temperature),supplied_diam,qnew,p.suspended_advected,
                exchange_wires,p.suspended_trial,p.suspended_exchange_trial,p.suspended_diagnostic)
            if vr1993:p.suspended_diameter_trial[:]=p.suspended_diameter
        else:
            stock=physical_stock(xp,c.available,p.class_delta,p.morfac)
            # Use the same robust stock reservation and bed-clock adapter as
            # the independent closures. The native FLOW source uses physical
            # dt and MF1 here; it must not apply the bed factor a second time.
            engine.launch('vr_sus_exchange' if vr1993 else 'sus_exchange',n,n,ng,dt,1.,p.parameters.dry,p.sediment_depth,float(cref),float(temperature),float(sus_factor),
                diam,qold[0],qnew,p.effective_q,p.effective_fields[0],p.effective_fields[2],p.suspended_advected,stock,c.surface_fraction,
                p.suspended_trial,p.suspended_exchange_trial,p.suspended_diagnostic,
                *((p.mpm_d90,p.mpm_hiding,exchange_wires) if capacity is not None else ()),
                *((p.suspended_diameter,p.suspended_diameter_trial) if vr1993 else ()))
        if (not gp and not dl and not classic and not settling_only and capacity is not None and capacity['method']=='capacity.wang_fredsoe'
                and bool(xp.any(p.suspended_diagnostic[1]<0.))):
            raise ValueError('Wang-Fredsoe produced negative suspended equilibrium for these inputs; '
                             'it cannot be used as a nonnegative concentration source without changing the law')
        if not settling_only:
            accelerate_wet_exchange(xp,p.suspended_exchange_trial,qnew[0],p.parameters.dry,p.morfac)
        p.class_delta+=p.suspended_exchange_trial
        p.suspended_acceleration_trial[:]=p.suspended_exchange_trial+p.suspended_trial-p.suspended_advected
        residual=xp.sum((p.suspended_advected-p.suspended_mass)*g['area'],axis=1)+dt*xp.sum(p.suspended_flux*g['length']*outside,axis=1)
        engine.launch('sus_check',n,n,ng,p.suspended_trial,c.available,p.class_delta,check_status)
        # One host synchronization instead of serial scalar transfers. Keep
        # the original global budget reduction and its acceptance threshold.
        residual_max,invalid,donor_min=host(xp.stack((xp.max(xp.abs(residual)),xp.any(check_status),xp.min(p.suspended_scale))))
        if residual_max>1e-10 or invalid:
            raise BendTrialRejected('Suspended transport/exchange budget failed')
        p.last.update(suspended_transport_residual_kg=float(residual_max),
            suspended_donor_factor_min=float(donor_min),suspended_exchange=True)
        return p.suspended_exchange_trial
    a.process_bed_delta=MethodType(exchange,a)
    previous_finish=a.finish
    def finish(owner,*args,**kwargs):
        if enabled and not kwargs.get('morphology',True):raise ValueError('This integrated suspension route requires morphology ON')
        return previous_finish(*args,**kwargs)
    a.finish=MethodType(finish,a)
    p.identity.pop('river_mpm')
    p.identity['river_mixtures']['capacity']=[cc.identity for cc in a.capacity]
    p.identity['river_mixtures']['hiding']=hiding if capacity is not None else 'none'
    p.identity['river_suspension']=dict(wrapper_sha256=digest(__file__),runtime=engine.identity,wave=wave.identity,
        formula=capacity if capacity is not None else 'FLOW reference selector 4 / TRA_COMBINE; fixed explicit general parameters',
        bed_factor=bed_factor,sus_factor=sus_factor,diffusivity=diffusivity,temperature=temperature,cref=cref,
        initial_sha256=__import__('hashlib').sha256(ci.tobytes()).hexdigest(),boundary_sha256=__import__('hashlib').sha256(bc.tobytes()).hexdigest(),
        settling='SettleFrm2, FacDSS1, explicit per-class density' if capacity is not None else
                 'SettleFrm2, FacDSS1, fixed fresh water density1000, class rho2650',
        material=a.material.identity,
        exchange_shear='erosed UMOD*kappa/log1p(ZUMOD/z0rou); physical ZUMOD=0.368f*h; not floored capacity Chezy',
        transport='accepted-water-flux donor cell; wet-LS nonorthogonal internal diffusion; finite donor scaling; actual areas',
        exchange='Galappatti implicit sink, finite active stock/MF, one column commit before banks/collapse',
        morphology='MF bed exchange; physical scalar time; explicit acceleration ledger; unaccelerated dry stranding',
        enabled=enabled,scope='explicit_hydraulics_split_transport_Cartesian_cut_cells',public_support=False)
    if capacity is not None:
        from .bed_wave_flux import METHOD as bed_wave_method
        p.identity['river_suspension']['bed_numerical_flux'] = bed_wave_method
    a.identity['selected_river_bundle'].update(suspension_integrated=enabled,scope='explicit_split_transport_layers_banks')
    if vr1993:
        p.identity['river_suspension'].update(
            reference_parameters=corrections[1]["van_rijn_reference"],
            settling="SettleFrm2 uses accepted class DSS; IopSus trial updates publish only at commit",
            dss_remesh="positive intensive area mean; local constitutive memory, not transported sediment mass",
            vr1993_source_sha256=digest(Path(__file__).with_name("river_vr1993.h")))
    if settling_only:
        p.identity['river_suspension'].update(
            exchange_method="suspension.settling_only",
            exchange="BRAIDS exponential deposition-only extension; no equilibrium or entrainment",
            settling="SettleFrm2 clear-water on explicitly measured Ds; no Galappatti floor",
            exchange_shear="not consumed by deposition",
            dss_update="inactive; measured Ds retained",reference_parameters=dict(supplied_params),
            supplied_wrapper_sha256=digest(Path(__file__).with_name("river_supplied_suspension.py")),cref="inactive; clear water",
            source_integration="exact exponential with frozen physical depth and clear-water settling",
            shallow="all wet depths; no sediment-depth gate; dry stranding once without MF",
            settling_only_source_sha256=digest(Path(__file__).with_name("river_settling_only.h")))
    if gp:
        p.identity['river_suspension'].update(
            exchange_method='suspension.garcia_parker_1991',
            reference_parameters=dict(gp_operator.parameters),gp_runtime=gp_operator.engine.identity,
            gp_wrapper_sha256=digest(Path(__file__).with_name('river_garcia_parker.py')),
            exchange='GP1991 independent reference entrainment; exponential frozen-depth source/sink; finite stock',
            exchange_shear='physical hydraulic total shear; bounded EH1967 skin partition for entrainment only',
            settling='SettleFrm2 clear-water on measured Ds; no hindering or minimum settling floor',
            profile='Rouse with total shear, a/h=.05; constant below reference; depth mean, not transport weighted',
            shallow='h<20Ds: no entrainment, well-mixed physical settling; dry stranding',
            morphology='physical source/sink followed by the named bed-clock adapter',
            dss_update='inactive; measured Ds retained',cref='inactive for GP1991')
    if dl:
        p.identity['river_suspension'].update(
            exchange_method='suspension.de_leeuw_2020',
            reference_parameters=dict(dl_operator.parameters),dl_runtime=dl_operator.engine.identity,
            dl_wrapper_sha256=digest(Path(__file__).with_name('river_de_leeuw.py')),
            exchange='de Leeuw Eq26a; exponential frozen-depth source/sink; finite stock',
            exchange_shear='Manning-Strickler Hsk solve with 3D84, local physical friction slope; skin bounded by total',
            settling='Ferguson-Church2004 natural grains C1=18 C2=1 on measured Ds; clear water',
            profile='Section3.3 P=(ws/ut)^.45; a/h=.1, below a:1.66Ca; whole-depth mass average',
            shallow='h<max(10Ds,3D84): no entrainment, well-mixed settling; dry stranding',
            morphology='physical source/sink followed by the named bed-clock adapter',
            dss_update='inactive; measured Ds retained',cref='inactive for de Leeuw 2020')
    if classic:
        params=classic_operator.parameters
        p.identity['river_suspension'].update(
            exchange_method=params['method'],reference_parameters=dict(params),
            classic_runtime=classic_operator.engine.identity,
            classic_wrapper_sha256=digest(Path(__file__).with_name('river_classic_suspension.py')),
            exchange=params['exchange'],exchange_shear=params['skin_shear'],
            settling=params['settling'],profile=params['profile'],shallow=params['shallow'],
            morphology='physical source/sink followed by the named bed-clock adapter',
            dss_update='inactive; measured Ds retained',cref='inactive for S6')
    from .river_morfac import METHOD as MORFAC_METHOD, DESCRIPTION
    p.identity['river_suspension'].update(
        morphology=DESCRIPTION, morphology_method=MORFAC_METHOD, morfac=float(p.morfac),
        morphology_source_sha256=digest(Path(__file__).with_name('river_morfac.py')))
    if capacity is not None and a.capacity[0].total_load:
        p.identity['river_suspension'].update(
            algebraic_total_load=True,resident_inventory="inert zero arrays only; no advection/exchange",
            algebraic_factor=1.,dss_update="inactive; FLOW non-suspfrac uses nominal diameter",
            algebraic_profile="VR93 zero fall speed and Schmidt=1; other formulas retain native suspended law")
    if capacity is not None:
        from .river_corrections import install
        install(a,folder,*corrections)
    return a
