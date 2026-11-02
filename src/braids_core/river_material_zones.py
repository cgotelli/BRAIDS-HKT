"""Explicit polygon overrides; conservative area integration, no mesh snapping."""
import math
import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def validate(zones):
    if not isinstance(zones,dict) or set(zones)!={'roughness','erodible'}:
        raise ValueError('material_zones requires roughness and erodible polygon lists')
    for kind,rows in zones.items():
        if not isinstance(rows,list):raise ValueError(kind+': polygon list required')
        names=set();polys=[]
        for row in rows:
            if not isinstance(row,dict) or set(row)!={'id','points_xy','value'}:raise ValueError(kind+': id, points_xy and value required')
            if not isinstance(row['id'],str) or not row['id'] or row['id'] in names:raise ValueError(kind+': unique polygon name required')
            names.add(row['id']);v=row['value'];points=np.asarray(row['points_xy'],float)
            if type(v) not in (float,int) or not math.isfinite(v) or v<0 or (kind=='roughness' and v==0):
                raise ValueError(kind+': positive roughness / nonnegative erodible depth required')
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():raise ValueError(kind+': finite polygon coordinates required')
            poly=Polygon(points)
            if not poly.is_valid or poly.area<=0:raise ValueError(kind+': simple positive-area polygon required')
            if any(poly.intersection(old).area>1e-10 for old in polys):raise ValueError(kind+': polygons must not overlap; edit or remove the overlap')
            polys.append(poly)
    return zones


def cell_values(mesh,rows,default,*,method=None):
    """Area-mean stock; area-mean drag coefficient for Manning/Chézy.

    Colebrook uses area-mean ks (nonlinear depth-dependent law evaluated later).
    Overrides may cross the exterior; only their fluid-cell intersection counts.
    """
    values=np.full(len(mesh.cells),float(default))
    if not rows:return values
    polys=[Polygon(row['points_xy']) for row in rows];tree=STRtree(polys)
    power=-2 if method=='roughness.chezy' else 2 if method=='roughness.manning' else 1
    base=float(default)**power
    for i,cell in enumerate(mesh.cells):
        poly=Polygon(cell.vertices_xy);v=base
        for j in tree.query(poly):
            area=poly.intersection(polys[int(j)]).area
            v+=(float(rows[int(j)]['value'])**power-base)*area/poly.area
        values[i]=max(0.,v)**(1./power)
    return values


def apply_initial(project,tree,fields):
    if project['model']['physics']['sediment'] is None:return
    rows=project['material_zones']['erodible']
    if not rows:return
    if len(fields['thickness'])!=1:raise ValueError('Spatial erodible depth currently requires one mixed bed')
    z=fields['rock']+fields['thickness'].sum(0)
    fields['thickness'][0]=cell_values(tree.finest,rows,project['initial']['layer_thickness_m'][0])
    fields['rock']=z-fields['thickness'].sum(0)
