"""Named impermeable exclusions; the saved polygon is the physical boundary."""
import numpy as np
from shapely.geometry import Polygon
from .cut_cells import SolidObstacle


def validate_solids(rows, domain=None):
    if not isinstance(rows, list):
        raise ValueError('Solids must be a list of named polygons')
    obstacles=[]; names=set(); shapes=[]
    origin=np.asarray(domain.vertices_xy[0] if domain is not None else (0.,0.))
    outer=None if domain is None else Polygon(np.asarray(domain.vertices_xy)-origin)
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'id','points_xy'}:
            raise ValueError('Each solid needs id and points_xy')
        obstacle=SolidObstacle(row['id'],row['points_xy'])
        if obstacle.obstacle_id in names:raise ValueError('Solid names must be unique')
        names.add(obstacle.obstacle_id)
        shape=Polygon(np.asarray(obstacle.vertices_xy)-origin)
        if outer is not None and (not outer.contains(shape) or outer.boundary.intersects(shape)):
            raise ValueError(row['id']+': solid must lie strictly inside the domain without touching its boundary')
        if any(shape.intersects(other) for other in shapes):
            raise ValueError(row['id']+': solids must not overlap or touch')
        obstacles.append(obstacle);shapes.append(shape)
    return tuple(obstacles)
