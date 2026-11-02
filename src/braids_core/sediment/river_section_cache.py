"""Exact-input memoization of prescribed inlet roots, on the chosen backend.

Only derived scratch is retained. No interior-flow boundary (stage, critical
outlet, supercritical or weir) can use this cache. Mesh geometry is immutable
within one owner; a remesh constructs a fresh cache. Bed, forcing, resistance
and bedform changes are compared by value, including rollback/restart.
"""


class SectionCache:
    def __init__(self, exterior, control):
        self.e = exterior
        self.ids = exterior.xp.flatnonzero(control)
        d = exterior.a.data
        # Static compact indices avoid boolean GPU indexing/synchronization
        # on each nonlinear residual.
        import numpy as np
        selected = np.isin(d['bid'], exterior.xp.asnumpy(self.ids)
                          if exterior.a.backend == 'gpu' else self.ids)
        self.faces = exterior.xp.asarray(np.flatnonzero(selected & (d['neighbour'] < 0)))
        self.saved = None
        self.token = None
        self.hits = self.misses = 0

    def inputs(self, *extra):
        e = self.e
        token=getattr(e,'residual_token',None)
        if token is not None and token is self.token:return self.saved
        return (e.p.exterior_bed[self.faces], e.forcing[0, self.ids], *extra)

    def restore(self, arrays, scalars):
        e = self.e; xp = e.xp
        if self.saved is None or scalars != self.scalars:
            return False
        token=getattr(e,'residual_token',None)
        if token is not None and token is self.token:
            e.forcing[1,self.ids]=self.stage
            self.hits+=1
            return True
        if any(a.shape != b.shape for a, b in zip(arrays, self.saved)):
            return False
        # One host decision for all mutable inputs; no rounding/hash tolerance.
        if bool(xp.any(xp.stack([xp.any(a != b) for a, b in zip(arrays, self.saved)]))):
            return False
        e.forcing[1, self.ids] = self.stage
        self.token=token
        self.hits += 1
        return True

    def store(self, arrays, scalars):
        self.saved = tuple(a.copy() for a in arrays)
        self.scalars = scalars
        self.stage = self.e.forcing[1, self.ids].copy()
        self.token=getattr(self.e,'residual_token',None)
        self.misses += 1
