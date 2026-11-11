"""Result shoreline to permeable mesh breaklines, and GIS line import."""
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import LineString,Polygon


def bank_lines(frame,project,*,wet_depth_m=.005):
    from .visual_fields import terrain_wet_edges
    from braids_core.boundaries.domain import PolygonDomain
    wet=np.isfinite(frame.q[0])&(frame.q[0]>wet_depth_m)
    if not np.any(wet):raise ValueError('No flooded channel at the selected result time.')
    cells=np.asarray(frame.polygons,dtype=object)
    geometry=shapely.union_all(cells[wet])
    rows,wedges=terrain_wet_edges(frame,dry=wet_depth_m)
    if len(rows):geometry=geometry.difference(shapely.union_all(cells[rows])).union(shapely.union_all(wedges))
    scale=float(np.sqrt(np.median(shapely.area(cells[wet]))))
    # A small inward margin avoids the outward dilation in the old domain
    # exporter. These are mesh constraints; neither side becomes a wall.
    tight=geometry.buffer(-.15*scale,quad_segs=4)
    if tight.is_empty:tight=geometry
    tight=tight.simplify(.5*scale,preserve_topology=True)
    domain=PolygonDomain(**project['settings']['domain'])
    tight=tight.intersection(Polygon(domain.vertices_xy))
    open_ids={a['edge_id'] for a in project['settings']['assignments'] if a['kind'] in ('inlet','outlet')}
    caps=[LineString([e.start_xy,e.end_xy]) for e in domain.edges if e.edge_id in open_ids]
    lines=tight.boundary
    if caps:lines=lines.difference(shapely.union_all(caps).buffer(2*scale))
    parts=[lines] if lines.geom_type=='LineString' else list(shapely.get_parts(lines))
    result=[p for p in parts if p.geom_type=='LineString' and p.length>=2*scale]
    if not result:raise ValueError('No bank lines remain away from the inlet/outlet sections.')
    audit=dict(schema='braids.result-bank-lines.v1',time_s=float(frame.time_s),wet_depth_threshold_m=wet_depth_m,
               cell_scale_m=scale,inward_margin_m=.15*scale,simplification_m=.5*scale,
               lines=len(result),vertices=sum(len(line.coords) for line in result),
               length_m=sum(line.length for line in result),open_sections_excluded=len(caps))
    return result,audit


def export_bank_lines(frame,project,path,*,crs_wkt=''):
    from osgeo import ogr
    from ._avac.geo import spatial_reference
    path=Path(path).with_suffix('.geojson')
    if path.exists():raise FileExistsError('Choose a new filename; existing breaklines are not overwritten.')
    lines,audit=bank_lines(frame,project)
    ds=ogr.GetDriverByName('GeoJSON').CreateDataSource(str(path))
    layer=ds.CreateLayer('River banks',srs=spatial_reference(crs_wkt),geom_type=ogr.wkbLineString)
    layer.CreateField(ogr.FieldDefn('name',ogr.OFTString))
    for i,line in enumerate(lines,1):
        feature=ogr.Feature(layer.GetLayerDefn());feature.SetField('name',f'River bank {i}')
        feature.SetGeometry(ogr.CreateGeometryFromWkb(line.wkb))
        if layer.CreateFeature(feature)!=0:raise ValueError('Unable to write river-bank breakline.')
    feature=None;ds=None
    audit_path=path.with_suffix('.geojson.audit.json');audit_path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return path,audit_path,audit


def read_breaklines(path,target_wkt):
    from osgeo import ogr
    from ._avac.geo import geometry_transform
    from braids_core.boundaries.breaklines import validate_breaklines
    ds=ogr.Open(str(path),0)
    if ds is None or ds.GetLayerCount()!=1:raise ValueError('Select one GIS line layer for breakline import.')
    layer=ds.GetLayer(0);transform=geometry_transform(layer.GetSpatialRef(),target_wkt);rows=[]
    for feature in layer:
        shape=feature.GetGeometryRef()
        if shape is None or shape.IsEmpty():continue
        shape=shape.Clone()
        if transform:shape.Transform(transform)
        kind=ogr.GT_Flatten(shape.GetGeometryType())
        if kind==ogr.wkbLineString:parts=[shape]
        elif kind==ogr.wkbMultiLineString:parts=[shape.GetGeometryRef(i) for i in range(shape.GetGeometryCount())]
        else:raise ValueError('Breakline input must contain LineString or MultiLineString features.')
        for line in parts:
            xy=np.asarray(line.GetPoints(),float)[:,:2];closed=bool(len(xy)>2 and np.array_equal(xy[0],xy[-1]))
            if closed:xy=xy[:-1]
            rows.append(dict(id=f'Imported bank {len(rows)+1}',closed=closed,points_xy=xy.tolist()))
    if not rows:raise ValueError('No breaklines found in the selected file.')
    validate_breaklines(rows)
    return rows
