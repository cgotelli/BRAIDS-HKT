"""User-drawn or imported minimum AMR level polygons."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import ogr

from braids_core.boundaries.amr import AMR_LEVELS,AMR_MAX_LEVELS

from .bank_inputs import BankZone,create_bank_dataset,load_bank_zones,map_bank_zones_to_cells

REFINEMENT_LAYER_NAME='braids_refinement_zones'


@dataclass(frozen=True)
class RefinementZone:
    zone_id: str
    minimum_level: int
    geometry_wkb: bytes

    def __post_init__(self):
        if (isinstance(self.minimum_level,bool) or not isinstance(self.minimum_level,int) or
                not 0<=self.minimum_level<AMR_MAX_LEVELS):
            raise ValueError(f'Minimum refinement level must be from L0 to L{AMR_MAX_LEVELS-1}')


def create_refinement_dataset(path,crs_wkt):
    path=create_bank_dataset(path,crs_wkt)
    ds=ogr.Open(str(path),1);layer=ds.GetLayer(0)
    layer.Rename(REFINEMENT_LAYER_NAME)
    if layer.CreateField(ogr.FieldDefn('min_level',ogr.OFTInteger))!=ogr.OGRERR_NONE:
        raise ValueError('Could not create minimum refinement level field')
    ds=None
    return path


def load_refinement_zones(path,crs_wkt,layer_name=REFINEMENT_LAYER_NAME,
                          maximum_level=AMR_LEVELS-1):
    if (isinstance(maximum_level,bool) or not isinstance(maximum_level,int) or
            not 0<=maximum_level<AMR_MAX_LEVELS):
        raise ValueError(f'Maximum refinement level must be from L0 to L{AMR_MAX_LEVELS-1}')
    geometry_rows=load_bank_zones(path,crs_wkt,layer_name)
    ds=ogr.Open(str(path));layer=ds.GetLayerByName(layer_name)
    if layer.GetLayerDefn().GetFieldIndex('min_level')<0:
        raise ValueError(f'Refinement polygons require a min_level field with values 0-{maximum_level}')
    levels={}
    for feature in layer:
        value=feature['min_level'];identifier=str(feature['zone_id']).strip()
        if value is None or float(value)!=int(value) or not 0<=int(value)<=maximum_level:
            raise ValueError(f'Refinement zone {identifier!r} requires an integer min_level from 0 to {maximum_level}')
        levels[identifier]=int(value)
    ds=None
    return tuple(RefinementZone(row.zone_id,levels[row.zone_id],row.geometry_wkb) for row in geometry_rows)


def map_refinement_zones(hierarchy,zones):
    minimum=np.zeros(hierarchy.count,dtype=int);counts={}
    for zone in zones:
        mask,audit=map_bank_zones_to_cells(hierarchy.finest,[BankZone(zone.zone_id,zone.geometry_wkb)])
        tail=minimum[hierarchy.offsets[-2]:]
        tail[mask]=np.maximum(tail[mask],zone.minimum_level)
        counts[zone.zone_id]=int(mask.sum())
    return minimum,{'zone_count':len(zones),'finest_cells_touched':counts,
                    'overlap_policy':'highest requested minimum level',
                    'cell_rule':'positive-area contact; whole cell refinement'}
