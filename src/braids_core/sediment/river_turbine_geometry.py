"""Actual-polygon actuator quadrature; rotated turbine-local coordinates."""
import math
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from shapely.geometry import Polygon,box
from shapely import constrained_delaunay_triangles
from shapely.strtree import STRtree


class GeometryIndex:
    """One immutable world-coordinate lookup shared by all rotors on a mesh."""
    def __init__(self,cells,*,cancelled=None):
        self.cells=cells
        from braids_core.boundaries.uniform_geometry import cell_polygons
        def stop():
            if cancelled and cancelled():raise InterruptedError('Turbine geometry setup cancelled')
        polygons=cell_polygons(cells,stop=stop)
        self.polygons=tuple(polygons)
        self.index=STRtree(self.polygons)
        self.area=np.array([c.area_m2 for c in cells]);self.xy=np.array([c.centroid_xy for c in cells])

    def query(self,shape):return sorted(map(int,self.index.query(shape)))


@lru_cache(maxsize=4)
def _quadrature(order):
    x,w=leggauss(order);s=(x+1)/2;w=w/2;s.setflags(write=False);w.setflags(write=False)
    return s,w


def triangle_bump(vertices,c,order):
    s,w=_quadrature(order)
    a,b,d=np.asarray(vertices)[:3];v=b-a;t=d-a
    xy=a+s[:,None,None]*v+(1-s[:,None,None])*s[None,:,None]*t
    z=2*(xy-np.array([c['X'],c['Y']]))/c['D'];inside=np.all(abs(z)<1,axis=2)
    value=np.zeros(inside.shape);zz=z[inside];value[inside]=np.exp(2-1/(1-zz[:,0]**2)-1/(1-zz[:,1]**2))
    return abs(np.linalg.det(np.array([v,t])))*np.sum(value*(1-s)[:,None]*w[:,None]*w[None,:])


def polygon_integrals(cells,c,*,geometry=None):
    # Integrate in turbine-local coordinates. Global GIS arithmetic can lose
    # several ulps in every Duffy point, preventing quadrature convergence.
    # Preserve original authoritative CV areas; translate only the integral.
    origin=np.array([c['X'],c['Y']]);local=dict(c,X=0.,Y=0.)
    normal=np.array(c.get('normal_xy',[1.,0.]));frame=np.array([normal,[-normal[1],normal[0]]]).T
    footprint=box(-c['D']/2,-c['D']/2,c['D']/2,c['D']/2)
    geometry=geometry or GeometryIndex(cells)
    world_footprint=Polygon(np.asarray(footprint.exterior.coords)@frame.T+origin)
    values=np.zeros(len(cells));area=0.;error=0.
    for i in geometry.query(world_footprint):
        a=cells[i];poly=Polygon((np.asarray(a.vertices_xy)-origin)@frame,[(np.asarray(ring)-origin)@frame for ring in getattr(a,'holes_xy',())])
        clipped=poly.intersection(footprint)
        if clipped.is_empty or clipped.area<=0:continue
        area+=clipped.area;triangles=constrained_delaunay_triangles(clipped)
        vertices=[np.asarray(t.exterior.coords)[:3] for t in triangles.geoms]
        previous=sum(triangle_bump(v,local,16) for v in vertices)
        for order in [32,64,128]:
            value=sum(triangle_bump(v,local,order) for v in vertices)
            if abs(value-previous)<=2e-12*c['D']**2:break
            previous=value
        else:raise ValueError('Polygon bump quadrature did not converge')
        values[i]=value;error+=abs(value-previous)
    if not math.isclose(area,footprint.area,rel_tol=1e-10,abs_tol=1e-13):
        raise ValueError('Turbine footprint intersects a wall or is not covered exactly')
    one=quad(lambda s:math.exp(1-1/(1-s*s)),-1.,1.,epsabs=1e-13,epsrel=1e-13)[0]*c['D']/2
    expected=one*one
    if abs(values.sum()-expected)>1e-9*expected:raise ValueError('Polygon partition changes the physical bump integral')
    return values,dict(covered_area_m2=area,integral_m2=float(values.sum()),independent_integral_m2=expected,
        relative_integral_error=float(values.sum()/expected-1),quadrature_error_bound_estimate_m2=error)

