"""Grid-independent DEM strips on open inlet edges; control volumes unchanged."""
from dataclasses import replace
import math

import numpy as np

from braids_core.boundaries.domain import boundary_polyline
from braids_core.boundaries.gis import _RasterSampler
from braids_core.boundaries.section import Section


def simplify_section_profile(stations, elevations, vertical_tolerance_m):
    """Return an endpoint-preserving profile with bounded vertical error.

    The iterative Ramer-Douglas-Peucker form avoids recursion depth limits and
    measures error vertically, which is the quantity used by section depth and
    pressure. A zero tolerance preserves every surveyed node exactly.
    """
    x=np.asarray(stations,dtype=float);z=np.asarray(elevations,dtype=float)
    tolerance=float(vertical_tolerance_m)
    if (x.ndim!=1 or z.shape!=x.shape or len(x)<2 or
            not np.isfinite(x).all() or not np.isfinite(z).all() or
            np.any(np.diff(x)<=0) or not math.isfinite(tolerance) or tolerance<0):
        raise ValueError('Section simplification needs finite increasing samples and a nonnegative tolerance')
    if tolerance==0:
        return x.copy(),z.copy(),0.
    keep=np.zeros(len(x),dtype=bool);keep[[0,-1]]=True;stack=[(0,len(x)-1)]
    while stack:
        left,right=stack.pop()
        if right<=left+1:continue
        interior=x[left+1:right]
        predicted=z[left]+(z[right]-z[left])*(interior-x[left])/(x[right]-x[left])
        residual=np.abs(z[left+1:right]-predicted)
        local=int(np.argmax(residual))
        if residual[local]>tolerance:
            split=left+1+local;keep[split]=True
            stack.append((left,split));stack.append((split,right))
    indices=np.flatnonzero(keep)
    maximum_error=float(np.max(np.abs(z-np.interp(x,x[indices],z[indices]))))
    return x[indices],z[indices],maximum_error


def section_with_cell_roughness(section, coordinates, mesh, manning_by_cell):
    """Attach the same whole-cell Manning field used by native friction."""
    values=np.asarray(manning_by_cell,dtype=float)
    if (values.shape!=(len(mesh.cells),) or not np.isfinite(values).all() or
            np.any(values<0)):
        raise ValueError('Section roughness requires one nonnegative Manning value per retained cell')
    origin=np.array(min(coordinates[0],coordinates[-1]),dtype=float)
    end=np.array(max(coordinates[0],coordinates[-1]),dtype=float)
    length=float(np.linalg.norm(end-origin));direction=(end-origin)/length
    intervals=[]
    for face in mesh.faces:
        if face.neighbour>=0 or face.bc_id!=section.bc_id:continue
        a=float((np.array(face.start_xy)-origin)@direction)
        b=float((np.array(face.end_xy)-origin)@direction)
        intervals.append((min(a,b),max(a,b),face.owner))
    coefficients=[]
    for left,right in zip(section.stations_m,section.stations_m[1:]):
        station=.5*(left+right)
        owners=[owner for lo,hi,owner in intervals if lo-1.e-8<=station<=hi+1.e-8]
        if not owners:
            raise ValueError(
                f'Cannot map section roughness to inlet {section.bc_id!r} at station {station:g} m')
        coefficient=float(max(values[owner] for owner in owners))
        if coefficient<=0:
            raise ValueError(
                f'Normal-depth section {section.bc_id!r} has zero Manning n at station {station:g} m')
        coefficients.append(coefficient)
    return replace(section,segment_manning_n=tuple(coefficients))


def terrain_inward_slope(dem_path, coordinates, inward_normal):
    """Return the representative positive bed slope into an inlet.

    The two sampling strips are centred half and one-and-a-half minimum DEM
    pixel scales inside the domain. Their separation is therefore one minimum
    pixel scale. A median over the face prevents one bank pixel or local spike
    from setting the normal-depth control for the whole section.
    """
    sampler=_RasterSampler(dem_path,allow_undeclared_metres=True)
    points=[tuple(map(float,point)) for point in coordinates]
    nx,ny=map(float,inward_normal)
    near_distance=sampler.max_spacing
    far_distance=3*sampler.max_spacing
    separation=far_distance-near_distance
    slopes=[]
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        length=math.hypot(x1-x0,y1-y0)
        count=max(1,math.ceil(length/sampler.max_spacing))
        for index in range(count):
            fraction=(index+.5)/count
            x=x0+fraction*(x1-x0);y=y0+fraction*(y1-y0)
            near=sampler.sample(x+nx*near_distance,y+ny*near_distance)
            far=sampler.sample(x+nx*far_distance,y+ny*far_distance)
            slopes.append((near-far)/separation)
    if not slopes:
        raise ValueError('Normal-depth inlet needs a nonzero boundary face')
    slope=float(np.median(slopes))
    if not math.isfinite(slope) or slope<=0:
        raise ValueError(
            f'Normal-depth inlet terrain does not fall into the domain '
            f'(computed DEM slope {slope:.6g}); choose another inlet or boundary mode')
    return slope,{
        'method':'Median DEM bed fall between parallel inward strips',
        'near_strip_distance_m':near_distance,
        'far_strip_distance_m':far_distance,
        'sample_count':len(slopes),
        'energy_slope':slope,
        'vertical_units':('metres' if sampler.band.GetUnitType() else
                          'metres assumed from the BRAIDS project terrain convention')}


def sample_inlet_terrain(mesh, domain, assignments, boundaries, dem_path,
                         manning_by_cell=None,section_vertical_tolerance_m=0.):
    """Split only flux-integration faces at common, DEM-scale sample stations.

Each strip retains one bilinear midpoint elevation, shared by every mesh that
intersects it. The native solver sums its wet area/pressure, rather than treating
an average of a downhill interior cell as the elevation of the domain edge.
No cells, cell areas, Manning values or cell-centre topography are changed.
"""
    sampler=_RasterSampler(dem_path,allow_undeclared_metres=True)
    tolerance=float(section_vertical_tolerance_m)
    if not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Section vertical tolerance must be finite and nonnegative')
    if manning_by_cell is not None:
        manning_by_cell=np.asarray(manning_by_cell,dtype=float)
        if (manning_by_cell.shape!=(len(mesh.cells),) or
                not np.isfinite(manning_by_cell).all() or
                np.any(manning_by_cell<0)):
            raise ValueError('Section roughness requires one nonnegative Manning value per retained cell')
    spacing=sampler.max_spacing/2  # survey source at quarter-pixel spacing
    profiles={};sections={};faces=[];simplification={}
    for name,spec in boundaries.items():
        if spec.mode not in ('discharge','inlet_stage','inlet_depth','inlet_normal','inlet_critical'):
            continue
        coordinates,normal=boundary_polyline(domain,assignments,name)
        origin=np.array(min(coordinates[0],coordinates[-1]));end=np.array(max(coordinates[0],coordinates[-1]))
        length=float(np.linalg.norm(end-origin));direction=(end-origin)/length
        # write_cut_case also enforces this contract; do not flatten a bend.
        if max(abs(direction[0]*(point[1]-origin[1])-direction[1]*(point[0]-origin[0])) for point in coordinates)>1.e-8:
            raise ValueError('Noncollinear inlet needs an explicit allocation/control rule')
        source_stations=np.linspace(0.,length,math.ceil(length/spacing)+1)
        source_xy=origin+source_stations[:,None]*direction
        source_nodes=np.array([sampler.sample(*point) for point in source_xy])
        stations,nodes,maximum_error=simplify_section_profile(
            source_stations,source_nodes,tolerance
        )
        centers=.5*(stations[:-1]+stations[1:]);xy=origin+centers[:,None]*direction
        elevations=np.array([sampler.sample(*point) for point in xy])
        source_faces=[face for face in mesh.faces if face.neighbour<0 and face.bc_id==name]
        source_intervals=[]
        for face in source_faces:
            left=float((np.array(face.start_xy)-origin)@direction)
            right=float((np.array(face.end_xy)-origin)@direction)
            source_intervals.append((min(left,right),max(left,right),face.owner))
        profiles[name]={'origin':origin,'direction':direction,'stations':stations,'bed':elevations,
                        'source_intervals':source_intervals}
        simplification[name]={'source_nodes':len(source_stations),'retained_nodes':len(stations),
            'vertical_tolerance_m':tolerance,'maximum_vertical_error_m':maximum_error}
        if spec.needs_section:
            section=Section(name,tuple(stations),tuple(nodes),float(nodes.min()),tuple(normal))
            if manning_by_cell is not None and spec.mode=='inlet_normal':
                section=section_with_cell_roughness(section,coordinates,mesh,manning_by_cell)
            sections[name]=section
    for face in mesh.faces:
        profile=profiles.get(face.bc_id) if face.neighbour<0 else None
        if profile is None:
            faces.append(replace(face,face_id=len(faces)));continue
        origin=profile['origin'];direction=profile['direction'];stations=profile['stations']
        a=float((np.array(face.start_xy)-origin)@direction);b=float((np.array(face.end_xy)-origin)@direction)
        lo,hi=sorted((a,b))
        cuts=np.r_[lo,stations[(stations>lo+1.e-8)&(stations<hi-1.e-8)],hi]
        if b<a:cuts=cuts[::-1]
        for left,right in zip(cuts,cuts[1:]):
            index=min(len(profile['bed'])-1,max(0,int(np.searchsorted(stations,.5*(left+right),side='right')-1)))
            fraction0=(left-a)/(b-a);fraction1=(right-a)/(b-a)
            p0=np.array(face.start_xy)+fraction0*(np.array(face.end_xy)-face.start_xy)
            p1=np.array(face.start_xy)+fraction1*(np.array(face.end_xy)-face.start_xy)
            edge0=face.edge_start_m;edge1=face.edge_end_m
            faces.append(replace(face,face_id=len(faces),start_xy=tuple(p0),end_xy=tuple(p1),
                length_m=face.length_m*abs(fraction1-fraction0),bed_elevation_m=float(profile['bed'][index]),
                section_bed_start_m=float(sampler.sample(*p0)),
                section_bed_end_m=float(sampler.sample(*p1)),
                edge_start_m=None if edge0 is None else edge0+fraction0*(edge1-edge0),
                edge_end_m=None if edge1 is None else edge0+fraction1*(edge1-edge0)))
    audit={'method':('Error-bounded piecewise-linear simplification of quarter-pixel DEM samples'
                     if tolerance else 'Common quarter-pixel bilinear midpoint strips on selected inlet edges'),
        'source_sample_spacing_m':spacing,
        'maximum_strip_length_m':max((float(np.diff(value['stations']).max())
                                      for value in profiles.values()),default=0.),
        'section_simplification':simplification,
        'inlet_ids':list(profiles),'cells_unchanged':True,
        'horizontal_units':'metres from the projected DEM/domain CRS',
        'vertical_units':'metres','declared_vertical_units':sampler.declared_vertical_units,
        'vertical_units_source':sampler.vertical_units_source,
        'vertical_datum':'source DEM elevations retained; no independent datum shift',
        'original_face_count':len(mesh.faces),'integration_face_count':len(faces),
        'bed_evolves_with_owner_increment':True,
        'wetting_component':'connected to the current lowest sampled section point',
        'section_profile':'piecewise-linear DEM profile with explicit segment endpoints',
        'section_roughness':('same whole-cell Manning field used by native friction'
                             if manning_by_cell is not None else 'not supplied')}
    return replace(mesh,faces=tuple(faces)),sections,audit
