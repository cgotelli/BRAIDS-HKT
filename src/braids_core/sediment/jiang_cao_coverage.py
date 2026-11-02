"""Explicit initial-repose mesh protection; research only, not a new AMR default.

One preparation-time projection on an already verified finest owner locates the
initial dry-bank redistribution footprint. Its unchanged physical state is NOT
replaced by the projection. The caller explicitly pins returned node IDs in its
adaptive identity. This static envelope does not discover later remote failures.
"""
import numpy as np
from .jiang_cao_remesh import validate
from .jiang_cao_checkpoint import array_hash
from .bend_methods_v1 import digest


def initial_repose_patch(hierarchy,owner,*,halo=1,check=lambda:None):
    if type(halo) is not int or halo<1:raise ValueError('At least one explicit stencil halo required')
    if not owner.collapse_enabled:raise ValueError('Initial repose protection requires active collapse physics')
    g=owner.mesh.g;xp=owner.xp;base=hierarchy.offsets[-2]
    xy=np.array([c.centroid_xy for c in hierarchy.finest.cells]).T
    if (owner.n!=len(xy.T) or not np.allclose(owner.a.host(g['xy']),xy,rtol=0,atol=1e-12) or
        any(not np.array_equal(poly,np.asarray(cell.vertices_xy)) for poly,cell in zip(owner.polygons,hierarchy.finest.cells))):
        raise ValueError('Verified finest owner in hierarchy order required')
    validate(owner);check()
    result=owner.collapse.apply(owner.state[:6],check=check)
    o,j=g['owner'],g['neighbour'];inside=j>=0
    moved=result.solid_face_volume!=0
    if bool(xp.any(moved&~inside)):raise ValueError('Exterior repose transfer')
    # Keep all accepted nonzero redistribution endpoints, including tiny tails.
    active=xp.zeros(owner.n,dtype=xp.bool_)
    active[o[moved]]=True;active[j[moved]]=True
    affected=owner.a.host(active).copy()
    for _ in range(halo):
        grown=active.copy()
        linked=inside&(active[o]|active[xp.maximum(j,0)])
        grown[o[linked]]=True;grown[j[linked]]=True;active=grown
    ids=tuple(int(i+base) for i in np.flatnonzero(owner.a.host(active)))
    check();validate(owner)
    return dict(method='explicit.initial_repose_patch205_v1',source_sha256=digest(__file__),
        protected_finest=list(ids),affected_finest=(np.flatnonzero(affected)+base).tolist(),halo=halo,
        initial_state_sha256=array_hash(owner.a.host(owner.state)),geometry_sha256=owner.mesh.geometry_sha256,
        gross_projection_solid_m3=result.gross_solid_transfer_m3,projection_sweeps=result.iterations,
        projection_residual_m3=result.solid_residual_m3,projection_max_slope=result.max_slope,
        scope='Preparation-time initial collapse footprint plus face-neighbour halo; not dynamic new-bank discovery',
        physical_state_changed=False)
