"""One conservative slope-transfer owner for the explicit shoreline research path.

The existing paired planform kernels are spatial operators, not evidence of
implicit time integration. This adapter records the actual enclosing solver.
Not a new QGIS admission or a mechanical undercut/block-failure model.
"""
from types import MethodType
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from .planform_v105 import NativePlanformRuntimeV105,PlanformMethodsV105,PlanformParametersV105


def install_shoreline_relaxation(solver,xy,*,method='classic_scour_bundle',slope=.5,
                                transport=True,avalanche_time_s=1.,relaxation_length_m=.25):
    p=solver.physics;xp=p.xp
    if p.time or p.steps or p._open or hasattr(p,'relaxation_engine'):
        raise ValueError('Install only one relaxation owner before execution')
    if method not in ('off','classic_scour_bundle','braids_conservative_wet_dry_avalanche'):
        raise ValueError('Unsupported shoreline relaxation method')
    xy=np.asarray(xy);x=np.unique(xy[:,0]);y=np.unique(xy[:,1]);nx=len(x);ny=len(y)
    if min(nx,ny)<2 or nx*ny!=p.n:raise ValueError('Full Cartesian patch required')
    dx=np.diff(x);dy=np.diff(y);order=np.lexsort((xy[:,0],xy[:,1]));back=np.argsort(order)
    target=np.array(np.meshgrid(x,y)).reshape(2,-1).T
    if not (np.allclose(dx,dx[0]) and np.allclose(dy,dy[0]) and np.allclose(xy[order],target)
            and np.allclose(solver.host(p.g['area']),dx[0]*dy[0])):
        raise ValueError('Uniform uncut Cartesian geometry required')
    o=solver.host(p.g['owner']);j=solver.host(p.g['neighbour'])
    edges={tuple(sorted((int(back[a]),int(back[b])))) for a,b in zip(o,j) if b>=0}
    expected={(c,c+1) for c in range(p.n) if c%nx+1<nx}|{(c,c+nx) for c in range(p.n) if c//nx+1<ny}
    if edges!=expected:raise ValueError('Internal walls or holes not admitted')
    parameters=PlanformParametersV105(scour_slope=slope,wet_slope=slope,dry_slope=slope,
        avalanche_time_s=avalanche_time_s,relaxation_length_m=relaxation_length_m)
    methods=PlanformMethodsV105(method)
    if solver.backend=='cpu':engine=NativePlanformRuntimeV105(methods,parameters,threads=solver.controls.threads)
    else:
        from braids_gpu.planform_v105 import ResidentPlanformRuntimeV105
        engine=ResidentPlanformRuntimeV105(methods,parameters)
    p.relaxation_engine=engine;p.relaxation_order=xp.asarray(order);p.relaxation_back=xp.asarray(back)
    p.relaxation_cumulative_mass=xp.zeros(p.n);p.relaxation_trial_mass=xp.zeros(p.n)
    base_finish=p.finish;base_commit=p.commit
    def finish(owner,qold,qnew,water,*,morphology=True):
        result=base_finish(qold,qnew,water,morphology=morphology and transport)
        owner.relaxation_trial_mass.fill(0.)
        if not morphology or method=='off':return result
        before=owner.ztrial.copy();order=owner.relaxation_order
        field=lambda a:xp.ascontiguousarray(a[order].reshape(ny,nx))
        layers=field(1590.*(before-owner.rock))[None,None]
        out=engine.advance(layers,field(owner.rock),field(qnew[0]),field(owner.g['area']),xp.array([1590.]),
            dt_s=owner._dt*owner.morfac,dx_m=float(dx[0]),dy_m=float(dy[0]))
        # No transferred inventory means no bed update. Reconstructing all
        # cells as rock + stock/bulk introduces roundoff even when Scour is
        # inactive; repeated false perturbations can seed shoreline feedback.
        # Exact inventory comparison, not a tolerance that filters real moves.
        changed=xp.any(out['layers']!=layers,axis=(0,1))
        bed=xp.where(changed,out['bed_elevation_m'],field(before)).ravel()[owner.relaxation_back]
        moved=(bed-before)*1590.*owner.g['area']
        if (not bool(xp.all(xp.isfinite(bed))) or bool(xp.any(bed<owner.rock-1e-13))
                or abs(float(xp.sum(moved)))>1e-9):
            raise BendTrialRejected('Slope relaxation inventory or conservation failure')
        owner.relaxation_trial_mass[:]=moved;owner.ztrial[:]=bed
        owner.last.update(morphology=True,relaxation_mass_kg=float(xp.sum(xp.maximum(moved,0.))),
            relaxation_internal_residual_kg=float(xp.sum(moved)))
        return owner.last
    def commit(owner):
        base_commit();owner.relaxation_cumulative_mass+=owner.relaxation_trial_mass
    p.finish=MethodType(finish,p);p.commit=MethodType(commit,p)
    p.identity['slope_relaxation']=dict(method=method,runtime=engine.identity,source_sha256=digest(__file__),
        slope=slope,transport=bool(transport),shape=[ny,nx],single_owner=True,
        integrator=solver.identity['integrator'],kernel_is_not_integrator=True,
        mechanical_block_failure=False,implicit_wetdry=False,
        unchanged_inventory='preserve_bed_exactly')
    return solver
