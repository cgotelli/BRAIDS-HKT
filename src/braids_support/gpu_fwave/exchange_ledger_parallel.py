"""Resident deterministic CUDA reduction for coupled suspension exchange.

The two signed ledger columns are gains in suspended and active-bed mass.
The coupled solver uses this only for active noncohesive exchange. Output and
scratch arrays are allocated once and reused; callers must consume them
before the next call and retain trial/commit ownership themselves.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .suspension_resident import OPTIONS


HERE = Path(__file__).resolve().parent
TILE = 256
DEFAULT_DIRECT_LIMIT = 2048  # Measured RTX 5080 crossover lies between 2,048/4,096.


def method_for_cells(nc, *, direct_limit=DEFAULT_DIRECT_LIMIT):
    """Frozen host-only dispatch; never selects or changes the CPU/GPU backend."""
    if (not isinstance(nc, (int, np.integer)) or isinstance(nc, (bool, np.bool_)) or
            nc < 1 or nc > np.iinfo(np.int32).max or
            not isinstance(direct_limit, (int, np.integer)) or
            isinstance(direct_limit, (bool, np.bool_)) or direct_limit < 1):
        raise ValueError('nc/direct_limit must be positive supported integers')
    return 'direct' if nc <= direct_limit else 'tiled'


class ResidentExchangeLedger:
    """One GPU-only, graph-capturable, signed class-exchange reduction.

    Dispatch is fixed at construction from the cell count, never from the
    CPU/GPU backend choice. `step` performs no allocation, host transfer, or
    synchronization. Inputs and outputs must not overlap. Failure status is
    per grain and reproduces the legacy nonfinite/negative/conservation gate.
    """

    def __init__(self, area, ng, *, source_path=None,
                 direct_limit=DEFAULT_DIRECT_LIMIT):
        import cupy as cp

        if not isinstance(ng, (int, np.integer)) or isinstance(ng, (bool, np.bool_)) or ng < 1:
            raise ValueError('ng must be a positive integer')
        if (not isinstance(area, cp.ndarray) or area.ndim != 1 or
                area.dtype != cp.float64 or not area.flags.c_contiguous or
                area.size < 1):
            raise ValueError('area must be a nonempty contiguous device float64 vector')

        self.cp = cp
        self.area = area
        self.nc = int(area.size)
        self.ng = int(ng)
        if self.ng*self.nc > np.iinfo(np.int32).max:
            raise ValueError('Exchange-ledger indexing exceeds CUDA int32 range')
        self.nt = (self.nc + TILE - 1)//TILE
        self.method = method_for_cells(self.nc, direct_limit=direct_limit)
        self.source_path = Path(source_path or HERE/'exchange_ledger_parallel.cu')
        source = self.source_path.read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        names = ('sr_exchange_direct', 'sr_exchange_partial', 'sr_exchange_finish')
        module = cp.RawModule(code=source.decode('utf-8'), options=OPTIONS,
                              name_expressions=names)
        self.k = {name: module.get_function(name) for name in names}

        self.ledger = cp.empty((self.ng, 2), dtype=cp.float64)
        self.status = cp.empty(self.ng, dtype=cp.int32)
        if self.method == 'tiled':
            self.scratch = tuple(cp.empty((self.ng, self.nt), dtype=cp.float64)
                                 for _ in range(3)) + (
                cp.empty((self.ng, self.nt), dtype=cp.int32),)
        else:
            self.scratch = ()

    def _input(self, value, label):
        if (not isinstance(value, self.cp.ndarray) or
                value.shape != (self.ng, self.nc) or
                value.dtype != self.cp.float64 or
                not value.flags.c_contiguous):
            raise ValueError(
                f'{label} must be contiguous device float64 ({self.ng}, {self.nc})')
        if self.cp.may_share_memory(value, self.ledger) or any(
                self.cp.may_share_memory(value, scratch) for scratch in self.scratch):
            raise ValueError(f'{label} must not overlap ledger or scratch storage')

    def step(self, before_s, before_b, after_s, after_b):
        """Return reused `{ledger, status}` buffers (no implicit sync)."""
        for label, value in (('before_s', before_s), ('before_b', before_b),
                             ('after_s', after_s), ('after_b', after_b)):
            self._input(value, label)
        common = (np.int32(self.nc), np.int32(self.ng), self.area,
                  before_s, before_b, after_s, after_b)
        if self.method == 'direct':
            self.k['sr_exchange_direct']((self.ng,), (TILE,),
                                         (*common, self.ledger, self.status))
        else:
            self.k['sr_exchange_partial']((self.nt, self.ng), (TILE,),
                    (np.int32(self.nc), np.int32(self.ng), np.int32(self.nt),
                     self.area, before_s, before_b, after_s, after_b,
                     *self.scratch))
            self.k['sr_exchange_finish']((self.ng,), (TILE,),
                    (np.int32(self.nt), np.int32(self.ng),
                     *self.scratch, self.ledger, self.status))
        return {'ledger': self.ledger, 'status': self.status}
