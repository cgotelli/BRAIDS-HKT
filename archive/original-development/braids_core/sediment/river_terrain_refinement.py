"""Static finest-DEM slope-break protection, independent of wet/dry state.

Fit planes using actual clipped-cell centres and face neighbours, not row/column
indices or fabricated samples outside walls. Compare directional slopes across
faces; require both angular change and resolved vertical detail. A constant
plane is not a feature. One finest-cell halo protects the crest/toe stencil.
The initial terrain mask is computed once per policy, never per timestep.
"""
from pathlib import Path
import hashlib
import numpy as np
from .bed_columns_v101 import host, array_sha
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine


def terrain_features(tree, bed, settings, *, backend='cpu', threads=4):
    n=len(tree.finest.cells);bed=np.asarray(bed,dtype=float)
    if bed.shape!=(n,) or not np.isfinite(bed).all():
        raise ValueError('Finite initial bed on every finest cell required for terrain refinement')
    if backend not in ('cpu','gpu'):raise ValueError('Explicit terrain backend required')
    xp=np
    if backend=='gpu':
        import cupy as xp
    angle=np.zeros(n);detail=np.zeros(n);tag=np.zeros(n,np.int32);pinned=tag.copy()
    runtime=None
    if settings.terrain_enabled and tree.finest_level>0:
        rows=[set() for _ in range(n)]
        for face in tree.finest.faces:
            if face.neighbour>=0:
                rows[face.owner].add(face.neighbour);rows[face.neighbour].add(face.owner)
        off=xp.asarray(np.r_[0,np.cumsum([len(r) for r in rows])].astype(np.int32))
        adj=xp.asarray(np.array([j for r in rows for j in sorted(r)],np.int32))
        xy=xp.asarray(np.array([c.centroid_xy for c in tree.finest.cells],float))
        z=xp.asarray(bed);gradient=xp.empty((n,2));angle=xp.empty(n);detail=xp.empty(n)
        tag=xp.empty(n,dtype=xp.int32);pinned=xp.empty_like(tag)
        engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),
            names=('terrain_gradient','terrain_tags','terrain_halo'))
        engine.launch('terrain_gradient',n,n,off,adj,xy,z,gradient)
        engine.launch('terrain_tags',n,n,float(np.deg2rad(settings.terrain_angle_deg)),
            float(settings.terrain_detail_m),off,adj,xy,z,gradient,angle,detail,tag)
        engine.launch('terrain_halo',n,n,off,adj,tag,pinned)
        runtime=engine.identity
        angle=host(angle);detail=host(detail);tag=host(tag);pinned=host(pinned)
    border=np.zeros(n,dtype=bool)
    if settings.domain_border_enabled:
        for face in tree.finest.faces:
            if face.neighbour<0 and face.boundary_role!='solid_obstacle':border[face.owner]=True
        pinned=np.maximum(pinned,border)
    nodes=np.flatnonzero(pinned)+tree.offsets[-2]
    identity=dict(method='initial_finest_directional_slope_break_v1',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_bed_sha256=array_sha(bed),pinned_sha256=array_sha(nodes),
        enabled=settings.terrain_enabled,domain_border_enabled=settings.domain_border_enabled,
        domain_border_cells=int(border.sum()),angle_deg=settings.terrain_angle_deg,
        detail_m=settings.terrain_detail_m,halo_finest_rings=1,
        detected_cells=int(np.count_nonzero(tag)),pinned_cells=len(nodes),runtime=runtime)
    return dict(nodes=nodes,tag=tag.astype(bool),angle_deg=np.rad2deg(angle),
        detail_m=detail,identity=identity)
