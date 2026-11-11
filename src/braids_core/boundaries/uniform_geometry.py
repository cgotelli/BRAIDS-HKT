"""Batch exact rectangles through GEOS; retain general cut-cell predicates."""
from pathlib import Path
import numpy as np
from shapely import (polygons as make_polygons,linearrings,get_coordinates,
                     get_num_coordinates,get_num_interior_rings)
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine


def rectangle_flags(cells,threads=4):
    """Recognize, never approximate, the rings already produced by clipping."""
    if type(threads) is not int or not 1<=threads<=256:raise ValueError('Positive preparation thread count <=256 required')
    if not cells:raise ValueError('Nonempty cell geometry required')
    counts=np.fromiter((len(c.vertices_xy) for c in cells),dtype=np.int64,count=len(cells))
    if counts.sum()>np.iinfo(np.int32).max:raise ValueError('Too many geometry vertices')
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int32)
    xy=np.ascontiguousarray(np.concatenate([c.vertices_xy for c in cells]),dtype=np.float64)
    holes=np.fromiter((bool(c.holes_xy) for c in cells),dtype=np.int32,count=len(cells))
    result=_rectangle_flags(offsets,xy,holes,threads)
    return offsets,xy,result


def _rectangle_flags(offsets,xy,holes,threads):
    if type(threads) is not int or not 1<=threads<=256:raise ValueError('Positive preparation thread count <=256 required')
    result=np.empty(len(holes),dtype=np.int32)
    engine=WaveEngine('cpu',threads,source=Path(__file__).with_suffix('.h'),names=('mesh_rectangles',))
    engine.launch('mesh_rectangles',len(holes),len(holes),offsets,xy,holes,result)
    return result


def polygon_rectangle_flags(polys,threads=4):
    """Same compiled proof on existing local polygons, without new CV records."""
    counts=np.asarray(get_num_coordinates(polys),dtype=np.int64)
    if counts.sum()>np.iinfo(np.int32).max:raise ValueError('Too many geometry vertices')
    offsets=np.r_[0,np.cumsum(counts)].astype(np.int32)
    xy=np.ascontiguousarray(get_coordinates(polys),dtype=np.float64)
    holes=np.asarray(get_num_interior_rings(polys),dtype=np.int32)
    return _rectangle_flags(offsets,xy,holes,threads)


def cell_polygons(cells,origin=None,stop=lambda:None):
    """Batch identical ring construction; retain vertex order and hole rings."""
    result=[];origin=None if origin is None else np.asarray(origin)
    for start in range(0,len(cells),256):
        stop();batch=cells[start:start+256]
        counts=np.fromiter((len(c.vertices_xy) for c in batch),dtype=np.int64,count=len(batch))
        xy=np.asarray(np.concatenate([c.vertices_xy for c in batch]),dtype=np.float64)
        if origin is not None:xy-=origin
        rings=linearrings(xy,indices=np.repeat(np.arange(len(batch)),counts))
        polygons=make_polygons(rings)
        for j,c in enumerate(batch):
            cell_holes=getattr(c,'holes_xy',())
            if cell_holes:
                from shapely.geometry import Polygon
                shell=np.asarray(c.vertices_xy);holes=[np.asarray(r) for r in cell_holes]
                if origin is not None:shell=shell-origin;holes=[r-origin for r in holes]
                polygons[j]=Polygon(shell,holes)
        result.extend(polygons)
    stop();return result


def simple_seeds(template,origin,threads,stop):
    """CCW rings identical to simplify(0)+orient for exact rectangles only."""
    offsets,xy,flags=rectangle_flags(template.cells,threads)
    xy-=origin;ids=np.flatnonzero(flags);result=[None]*len(flags)
    # Bounded batches retain prompt cancellation and avoid one Python GEOS
    # constructor, simplify, orient and generic star test for every square.
    for start in range(0,len(ids),4096):
        stop();batch=ids[start:start+4096];rings=xy[offsets[batch,None]+np.arange(5)]
        negative=flags[batch]<0;rings[negative]=rings[negative,::-1]
        for i,p in zip(batch,make_polygons(rings)):result[int(i)]=p
    return result,flags
