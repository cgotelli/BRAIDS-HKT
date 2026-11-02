"""Isolated hydraulic/sediment face activity experiment. No public promotion.

Retain the existing dry depth. Resolve subtraction roundoff at that depth,
then use the HR common-bed opening for sediment connectivity. Not a FLOW
KFU clone: native Flood uses a staggered upwind surface above a cell crest.
"""
from pathlib import Path
from types import MethodType
from .sediment_wave import WaveEngine
from . import hydrostatic_step_probe as hr
from braids_core.sediment.bend_methods_v1 import digest
from braids_core.sediment import reflecting_wall


NATIVE = '''
  tol=16.d0*epsilon(1.d0)*max(dry_tolerance,abs(top),abs(left(1)+bl),abs(right(1)+br))
  if(hl<=dry_tolerance+tol) hl=0.d0
  if(hr<=dry_tolerance+tol) hr=0.d0
'''


def native_transform(code):
    code=hr.native_transform(code)
    code=code.replace('top,hl,hr\n','top,hl,hr,tol\n')
    needle='  ll=0.d0;rr=0.d0;ll(1)=hl;rr(1)=hr'
    assert code.count(needle)==1
    return reflecting_wall.native_source(code.replace(needle,NATIVE+needle))


def cuda_transform(code):
    code=hr.cuda_transform(code)
    needle='const double top=fmax(bL,bR), hl=fmax(0.,hL+bL-top), hr=fmax(0.,hR+bR-top);'
    assert code.count(needle)==1
    return reflecting_wall.cuda_source(code.replace(needle,'''const double top=fmax(bL,bR);
        const double tol=16.*2.2204460492503131e-16*fmax(dry,fmax(fabs(top),fmax(fabs(hL+bL),fabs(hR+bR))));
        double hl=fmax(0.,hL+bL-top), hr=fmax(0.,hR+bR-top);
        if(hl<=dry+tol)hl=0.;if(hr<=dry+tol)hr=0.;'''))


def native_hydraulics(deck,data,threads):
    """Construct only the final identical library, not two discarded carriers."""
    from braids_core.sediment import bend_native_hydraulics as native
    from braids_core.sediment import shoreline_solver as shoreline
    with native._lock:
        original=hr.probe.native_code
        try:
            hr.probe.native_code=lambda code,mode:native_transform(original(code,mode))
            result=hr.probe.build('positive_bounded')
        finally:hr.probe.native_code=original
        previous=shoreline.build_native
        try:
            shoreline.build_native=lambda:result
            hydraulics=shoreline.NativeShoreline(deck,data,threads)
        finally:shoreline.build_native=previous
    hydraulics._river_activity_ready=True
    return hydraulics


def install_hydraulic(a):
    """Install before the fresh sediment trace adapter captures the library."""
    if a.backend=='cpu':
        if getattr(a.hydraulics,'_river_activity_ready',False):
            a.identity['hydraulic_probe']=a.hydraulics.identity
        else:
            original=hr.probe.native_code
            try:
                hr.probe.native_code=lambda code,mode:native_transform(original(code,mode))
                hr.probe.install(a,'positive_bounded')
            finally:hr.probe.native_code=original
    else:
        import hashlib
        import cupy as cp
        from braids_support.gpu_fwave.resident_update import OPTIONS
        from braids_core.windows_paths import filesystem_path
        folder=filesystem_path(a._shoreline_source_folder)
        code=(folder/'augmented_fwave.cu').read_text()+'\n'+cuda_transform((folder/'resident_update.cu').read_text())
        op=a.hydraulics.flux;names=tuple(op.kernels)
        op.module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
        op.kernels={name:op.module.get_function(name) for name in names}
        a.identity['activity_generated_cuda_sha256']=hashlib.sha256(code.encode()).hexdigest()
    a.identity['hydrostatic_step_probe']=dict(driver_sha256=digest(hr.__file__),production=False,
        augmented_fwave_unchanged=True,sediment_changed=False)
    a.identity['hr_activity_roundoff']=dict(source_sha256=digest(__file__),
        dry_depth='unchanged; 16 epsilon times elevation scale only at threshold',public_support=False)
    a.identity['reflecting_wall']=dict(method=reflecting_wall.METHOD,
        source_sha256=digest(reflecting_wall.__file__),interior_augmented_fwave_unchanged=True)


def install_sediment(a):
    p=a.physics;xp=p.xp
    if p.time or p.steps or p._open:raise ValueError('Install before execution')
    if not hasattr(p,'hydraulic_wave_trace'):raise ValueError('Requires fresh HR traces')
    if 'hydrostatic_step_probe' not in a.identity:raise ValueError('Requires HR hydraulics')
    if hasattr(p,'hydraulic_face_active'):raise ValueError('Activity already installed')
    p.activity_engine=WaveEngine(a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=('hr_face_activity',))
    p.hydraulic_face_active=xp.zeros(p.nf,dtype=xp.int32)
    p.hydraulic_face_opening=xp.zeros(p.nf)
    old_launch=p.shallow_engine.launch
    def launch(name,count,*args):
        if name=='normal_cell_vector':
            g=p.g
            batch=getattr(p,'_river_hydraulic_batch',None)
            if batch is None or not batch.get('face_activity',False):
                p.activity_engine.launch('hr_face_activity',p.nf,p.nf,p.parameters.dry,
                    g['neighbour'],g['bid'],p.hydraulic_wave_trace,p.hydraulic_face_active,p.hydraulic_face_opening)
                if batch is not None:batch['face_activity']=True
            values=list(args);values[-2]=values[-2]*p.hydraulic_face_active
            return old_launch(name,count,*values)
        return old_launch(name,count,*args)
    p.shallow_engine.launch=launch
    old_rec=p.reconstruct_capacity
    def reconstruct(owner):
        old_rec()
        owner.bedface[:2]*=owner.hydraulic_face_active
        owner.wave_diagnostic*=owner.hydraulic_face_active
    p.reconstruct_capacity=MethodType(reconstruct,p)
    old_bed=p.bed_faces
    def faces(owner,q,z,hi):
        old_bed(q,z,hi)
        owner.aux[2]*=owner.hydraulic_face_active
    p.bed_faces=MethodType(faces,p)
    p.identity['hydraulic_sediment_activity']=dict(wrapper_sha256=digest(__file__),runtime=p.activity_engine.identity,
        before_tangent='hydraulic-open normal capacities only',before_slope='hydraulic-open vector only',
        after_slope='hydraulic-open final normal flux only',public_support=False,
        dry_threshold_m=p.parameters.dry,physical_laws_changed=False,stock_limiter_changed=False)
    return a
