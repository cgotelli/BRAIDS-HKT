"""Sparse, permanent subcells inside the ordinary finest AMR level.

Only intersected background tiles are subdivided. A buffered 2:1 staircase
returns to the ordinary mesh; no domain-wide fine grid is ever allocated.
"""
import math
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def split_cells(template, regions, origin, stop):
    base=min(template.grid.dx_m,template.grid.dy_m)
    plans=[]
    for r in regions:
        size=float(r['maximum_size_m'])
        if not math.isfinite(size) or size<=0:raise ValueError('Positive local refinement size required')
        depth=max(0,math.ceil(math.log2(base/size)))
        if depth>16:raise ValueError('Turbine refinement exceeds 16 local subdivisions; check rotor diameter and mesh units.')
        if depth:
            shape=Polygon(np.asarray(r['points_xy'],float)-origin)
            if not shape.is_valid or shape.area<=0:raise ValueError('Invalid local refinement polygon')
            plans.append((depth,shape))
    if not plans:return None
    bands={}
    for level in range(1,max(d for d,s in plans)+1):
        # Several complete rows between subcell sizes also isolate this local
        # patch from the coarser dynamically selected AMR parents.
        bands[level]=unary_union([s.buffer(3*base/2**(level-1),join_style=2) for d,s in plans if d>=level])
    result=[];areas=[]
    def descend(poly,i,j,level):
        stop();dx=template.grid.dx_m/2**level;dy=template.grid.dy_m/2**level
        if level+1 not in bands or not poly.intersects(bands[level+1]):
            result.append(poly);areas.append(dx*dy);return
        for dj in (0,1):
            for di in (0,1):
                ii,jj=2*i+di,2*j+dj
                part=poly.intersection(box(ii*dx/2,jj*dy/2,(ii+1)*dx/2,(jj+1)*dy/2))
                parts=[part] if part.geom_type=='Polygon' else list(getattr(part,'geoms',()))
                for p in parts:
                    if p.geom_type=='Polygon' and p.area>0:descend(p,ii,jj,level+1)
    for cell in template.cells:
        poly=Polygon(np.asarray(cell.vertices_xy)-origin,[np.asarray(h)-origin for h in cell.holes_xy])
        descend(poly,cell.i,cell.j,0)
    return result,np.asarray(areas)
