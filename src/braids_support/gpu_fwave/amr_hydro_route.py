"""Explicit GPU-AMR wet-front scheduling guard for no-inflow hydro cases.

Native synchronizes all leaves when an internal face separates wet and dry
cells. This reproduces that predicate on device, returning only a control
scalar to Python. It does not copy a field to CPU or run CPU numerics.
Prescribed dry inlet detection is outside this bounded no-inflow gate.
"""
from __future__ import annotations

import numpy as np


def requires_synchronous_wet_front(q, owner, neighbour, wet_depth=1.e-6):
    import cupy as cp
    if not all(isinstance(value,cp.ndarray) for value in (q,owner,neighbour)):
        raise TypeError('Resident GPU state and face topology required')
    if (q.ndim!=2 or q.shape[0]!=3 or owner.ndim!=1 or
            neighbour.shape!=owner.shape or owner.dtype.kind not in 'iu' or
            neighbour.dtype.kind not in 'iu' or not np.isfinite(wet_depth) or wet_depth<=0):
        raise ValueError('Invalid GPU AMR wet-front geometry or threshold')
    internal=neighbour>=0
    left=q[0,owner[internal]]>wet_depth
    right=q[0,neighbour[internal]]>wet_depth
    return bool(cp.any(left!=right).item())
