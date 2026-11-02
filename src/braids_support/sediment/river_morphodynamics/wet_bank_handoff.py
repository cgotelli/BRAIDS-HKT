"""Paired research-only geometric depth for the existing bank recipient/donor rule.

No sediment face weighting, no second bank transfer, no hydraulic geometry change.
The native hard activity cutoff is retained on wet-portion mean depth. This does
not constitute a continuous subcell erosion model or an experimental braids_support.
"""
from pathlib import Path
from .sediment_wave import WaveEngine
from .wet_exposure import engine as geometry_engine
from braids_core.sediment.bend_methods_v1 import digest


def engine(backend,threads=4):
    return WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=('handoff_depth',))


def install(a):
    p=a.physics;xp=a.xp;g=p.g
    if p.time or p.steps or p._open or hasattr(p,'handoff_engine'):
        raise ValueError('Install handoff once before execution')
    if not hasattr(p,'bank_engine'):raise ValueError('Existing single bank owner required')
    if hasattr(p,'exposure_engine'):raise ValueError('Rejected face weighting must not be composed')
    settings=p.identity['dry_bank']
    if settings['hmaxth']!=0:raise ValueError('HmaxTH depth taper not qualified for this research rule')
    dx=float(a.data['dx'])
    if not bool(xp.all(xp.abs(g['area']-dx*dx)<1e-12)):raise ValueError('Square full-cell research only')
    p.handoff_geometry_engine=geometry_engine(a.backend,p.flow.threads)
    p.handoff_engine=engine(a.backend,p.flow.threads)
    p.handoff_gradient=xp.zeros((2,p.n));p.handoff_geometry=xp.zeros((4,p.n))
    p.handoff_depth=xp.zeros(p.n);p.handoff_active=xp.zeros(p.n,dtype=xp.int32)
    original=p.bank_engine.apply
    def apply(bed,depth,area,bulk,wet,erodible,erosion,fraction,fixfac,initial,*,theta,hmaxth,sedthr):
        if hmaxth!=0 or sedthr!=settings['sedthr']:raise ValueError('Bank controls changed after research installation')
        h=xp.ascontiguousarray(depth.ravel()[p.bank_back]);z=p.z0;e=p.handoff_geometry_engine
        e.launch('exposure_gradient',p.n,p.n,z,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],p.handoff_gradient)
        e.launch('exposure_cells',p.n,p.n,dx,dx,sedthr,h,z,p.handoff_gradient,p.handoff_geometry)
        p.handoff_engine.launch('handoff_depth',p.n,p.n,float(p.parameters.dry),sedthr,h,p.handoff_geometry,p.handoff_depth,p.handoff_active)
        active=p.handoff_active[p.bank_order].reshape(depth.shape)
        # Exactly one original bank owner applies the same conservative proposal,
        # shared-donor stock limit and gather. Both roles use this SAME flag.
        return original(bed,depth,area,bulk,active,erodible,erosion,fraction,fixfac,initial,
                        theta=theta,hmaxth=hmaxth,sedthr=sedthr)
    p.bank_engine.apply=apply
    p.identity['wet_bank_handoff_research']=dict(runtime=p.handoff_engine.identity,geometry_runtime=p.handoff_geometry_engine.identity,
        wrapper_sha256=digest(__file__),method='native_bank_rule_with_wet_portion_mean_depth_activity',
        physical_threshold_m=settings['sedthr'],hydraulic_operator_changed=False,transport_flux_changed=False,
        physical_capacity_changed=False,one_bank_owner=True,public_support=False,
        limitation='binary cutoff retained; affine geometry diagnostic, not full subcell hydraulics or a continuous subcell bank model')
    return a
