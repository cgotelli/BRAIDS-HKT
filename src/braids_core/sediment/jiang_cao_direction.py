"""Explicit research-only direction-resolution experiment; not a paper closure.

Unique physical neighbour cells define the velocity fit. If the local velocity
is smaller than its RMS neighbour increment, its direction is under-resolved.
A C1 resolution weight regularizes curvature, not water/sediment inventories.
There is no dimensional speed cutoff and no relaxation of the profile guard.
"""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD = 'secondary.direction_resolved206_v1'


class DirectionOperator:
    def __init__(self, mesh, backend, *, threads=4):
        if backend not in ('cpu', 'gpu') or mesh.engine.backend != backend:
            raise ValueError('Matching explicit geometry/backend required')
        self.mesh, self.xp = mesh, mesh.xp
        self.engine = WaveEngine(backend, threads, source=Path(__file__).with_suffix('.h'),
                                 names=('jc206_direction',))
        # Immutable geometry preprocessing only. Evolved fits and resolution
        # weights are evaluated in parallel on the selected backend each stage.
        host = lambda v: v.get() if hasattr(v, 'get') else np.asarray(v)
        g = mesh.g; owner, other = host(g['owner']), host(g['neighbour'])
        cells = [set() for _ in range(mesh.n)]
        for i, j in zip(owner, other):
            if j >= 0:
                cells[int(i)].add(int(j)); cells[int(j)].add(int(i))
        offsets = np.r_[0, np.cumsum([len(c) for c in cells])].astype(np.int32)
        neighbours = np.array([j for c in cells for j in sorted(c)], dtype=np.int32)
        self.offsets = self.xp.asarray(offsets); self.neighbours = self.xp.asarray(neighbours)
        self.identity = dict(method=METHOD, source_sha256=digest(__file__),
            runtime=self.engine.identity, geometry=mesh.geometry_sha256,
            fit='unique active physical-centroid neighbours; inverse-distance-squared LS',
            resolution='sigma = RMS velocity increment to unique active neighbours; r=min(1,U/sigma); w=r*r*(3-2*r)',
            action='curvature Jacobian multiplied by w; published covariance and drift evaluated with that curvature',
            limits='research numerical regularization, not author implementation or experimental validation; unchanged mild-curvature/log-profile guard')

    def evaluate(self, velocity, active):
        xp = self.xp; n = self.mesh.n
        v = xp.ascontiguousarray(xp.asarray(velocity, dtype=xp.float64))
        a = xp.ascontiguousarray(xp.asarray(active, dtype=xp.int32))
        if (v.shape != (2,n) or a.shape != (n,) or not bool(xp.all(xp.isfinite(v)))
                or bool(xp.any((a != 0)&(a != 1)))):
            raise ValueError('Finite velocity (2,n) and activity (n) required')
        out = xp.empty((7,n))
        self.engine.launch('jc206_direction', n, n, self.offsets, self.neighbours,
                           self.mesh.g['xy'], a, v, out)
        # Four raw gradient rows, then sigma, weight, raw signed curvature.
        return out
