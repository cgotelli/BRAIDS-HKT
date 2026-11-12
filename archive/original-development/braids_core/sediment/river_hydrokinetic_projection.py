"""Circular rotor projection onto depth-integrated polygonal control volumes.

For a uniformly loaded circular disc the transverse force density is
proportional to its vertical chord, 2 sqrt(R**2-s**2).  A uniform plan-view
strip loses that geometry.  This research owner retains the integrated
Kramer--Piggott load but uses exact chord integrals over the actual polygons.
The local velocity is force-weighted so reported hydraulic power equals the
work of the distributed source, including in a nonuniform wake.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from braids_core.hydrokinetic_turbines import TurbineMomentumSource
from .bend_methods_v1 import digest


METHOD = 'depth_averaged_actuator_disc.kramer_piggott_circular_projection_v1'


def _primitive(s, radius):
    s = min(radius, max(-radius, float(s)))
    root = math.sqrt(max(0., radius*radius-s*s))
    return (s*root+radius*radius*math.asin(s/radius), -2.*root**3/3.)


def _ring_integral(ring, center, normal, tangent, radius):
    """Green's theorem: integral c(s) dn ds = contour integral n c(s) ds."""
    xy = np.asarray(ring.coords)-center
    ns = np.column_stack((xy@normal, xy@tangent))
    value = 0.
    for (n0,s0),(n1,s1) in zip(ns,ns[1:]):
        if abs(s1-s0) <= 1.e-15*radius:
            continue
        slope = (n1-n0)/(s1-s0)
        a0,b0 = _primitive(s0,radius);a1,b1 = _primitive(s1,radius)
        value += (n0-slope*s0)*(a1-a0)+slope*(b1-b0)
    return abs(value)


def chord_polygon_integral(polygon, definition):
    """Return integral of rotor chord over a polygon, in m3.

    The caller clips to the physical diameter-wide support first. Holes and
    disconnected pieces are included; no centroid sampling is used.
    """
    if polygon.is_empty or polygon.area <= 0:
        return 0.
    if polygon.geom_type != 'Polygon':
        return sum(chord_polygon_integral(p,definition) for p in polygon.geoms
                   if p.geom_type in ('Polygon','MultiPolygon','GeometryCollection'))
    center=np.asarray(definition.center_xy);normal=np.asarray(definition.normal_xy)
    tangent=np.asarray(definition.tangent_xy);radius=.5*definition.rotor_diameter_m
    args=(center,normal,tangent,radius)
    value=_ring_integral(polygon.exterior,*args)
    value-=sum(_ring_integral(ring,*args) for ring in polygon.interiors)
    return max(0.,value)


class CircularRotorSource(TurbineMomentumSource):
    def __init__(self,footprints,cells,*,backend='cpu',dry_depth_m=1.e-6):
        super().__init__(footprints,backend=backend,dry_depth_m=dry_depth_m)
        self.projected_rotor_areas=[]
        for footprint,device in zip(self.footprints,self.device):
            definition=footprint.definition;c=np.asarray(definition.center_xy)
            t=np.asarray(definition.tangent_xy)*(.5*definition.rotor_diameter_m)
            n=np.asarray(definition.normal_xy)*(.5*footprint.support_thickness_m)
            physical=Polygon((c-t-n,c+t-n,c+t+n,c-t+n))
            areas=[]
            for cell_id in footprint.cell_ids:
                cell=cells[int(cell_id)]
                polygon=Polygon(cell.vertices_xy,getattr(cell,'holes_xy',()))
                areas.append(chord_polygon_integral(polygon.intersection(physical),definition)/
                             footprint.support_thickness_m)
            areas=np.asarray(areas)
            if not math.isclose(float(areas.sum()),definition.rotor_area_m2,
                                rel_tol=1.e-10,abs_tol=1.e-13):
                raise ValueError(definition.turbine_id+': circular rotor projection is incomplete')
            areas*=definition.rotor_area_m2/areas.sum()
            drag=(definition.thrust_coefficient*areas+
                  definition.support_drag_coefficient*definition.support_frontal_area_m2*
                  footprint.overlap_area_m2/footprint.support_area_m2)
            total=definition.combined_drag_area_m2
            weights=(drag/total if total>0 else areas/definition.rotor_area_m2)
            # The inherited force and sample use this common normalized measure.
            device['overlap']=self.xp.asarray(weights*footprint.support_area_m2)
            device['fractions']=self.xp.asarray(weights*footprint.support_area_m2/
                                               footprint.cell_area_m2)
            self.projected_rotor_areas.append(areas)
        self.identity.update(method=METHOD,source_sha256=digest(__file__),
            distribution='analytic circular-rotor chord / actual polygon area; uniform support drag',
            velocity='force-weighted local depth-averaged normal velocity',
            depth='force-weighted physical water depth',
            force_geometry_reference='Nash et al. 2015 section 2.2: proportional swept area on fine cells',
            correction_scope='Kramer-Piggott footprint-mean correction; not a 3D rotor velocity reconstruction')
