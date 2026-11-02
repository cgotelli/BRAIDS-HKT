"""Fail-closed prepared spatial material/roughness consistency checks.

The native input's bed elevation is a derived value: rock plus all grain
inventories divided by their dry bulk densities. This check prevents an
otherwise validly hashed but inconsistent prepared record from running with
different hydraulic and Exner beds. It does not infer or invent a raster.
"""
from __future__ import annotations

import numpy as np


def validate_prepared_spatial_material(data):
    """Validate exact per-cell bed/roughness layout before GPU output creation.

    Accepts bedlaw 0 and 2; strict-positive Manning is required only where
    native MPM roughness mode 1 or secondary-flow physics needs it.
    """
    for name in ('bed', 'rock', 'layer_mass', 'bulk_fraction', 'manning_cell',
                 'bedlaw', 'mpm_friction_mode', 'secondary_flow_model'):
        if name not in data:
            raise ValueError('Missing prepared spatial material field '+name)
    bed=np.asarray(data['bed']);rock=np.asarray(data['rock'])
    layers=np.asarray(data['layer_mass']);bulk=np.asarray(data['bulk_fraction'])
    manning=np.asarray(data['manning_cell'])
    if (bed.ndim!=1 or bed.size<1 or rock.shape!=bed.shape or
            layers.ndim!=3 or layers.shape[0]<1 or layers.shape[1]<1 or
            layers.shape[2]!=bed.size or bulk.shape!=(layers.shape[1],) or
            manning.shape!=bed.shape):
        raise ValueError('Invalid prepared spatial bed/material/roughness shapes')
    if any(not np.issubdtype(array.dtype,np.number) or
           not np.isfinite(array).all() for array in
           (bed,rock,layers,bulk,manning)):
        raise ValueError('Nonfinite or nonnumeric prepared spatial material')
    if np.any(layers<0) or np.any(bulk<=0) or np.any(manning<0):
        raise ValueError('Negative inventory, invalid bulk density or roughness')
    for name in ('bedlaw','mpm_friction_mode','secondary_flow_model'):
        value=np.asarray(data[name])
        if value.shape!=() or value.dtype.kind not in 'iu':
            raise ValueError('Invalid prepared spatial material control '+name)
    if ((int(data['bedlaw'])==2 and int(data['mpm_friction_mode'])==1) or
            int(data['secondary_flow_model'])!=0) and np.any(manning<=0):
        raise ValueError('Positive per-cell Manning required for active roughness/secondary physics')
    reconstructed=rock+np.sum(layers/bulk[None,:,None],axis=(0,1))
    scale=np.maximum.reduce((np.ones_like(bed),np.abs(bed),np.abs(rock)))
    tolerance=1e-10+64*np.finfo(np.float64).eps*scale
    bad=np.flatnonzero(np.abs(bed-reconstructed)>tolerance)
    if bad.size:
        cell=int(bad[0])
        raise ValueError(f'Prepared bedrock/layer thickness disagrees with bed at cell {cell}')
    return dict(cells=int(bed.size),classes=int(bulk.size),layers=int(layers.shape[0]),
                rock_only_cells=int(np.count_nonzero(np.sum(layers,axis=(0,1))==0)),
                min_manning=float(manning.min()),max_manning=float(manning.max()),
                min_mobile_thickness_m=float(np.min(np.sum(layers/bulk[None,:,None],axis=(0,1)))),
                max_mobile_thickness_m=float(np.max(np.sum(layers/bulk[None,:,None],axis=(0,1)))),
                max_bed_reconstruction_error_m=float(np.max(np.abs(bed-reconstructed))))
