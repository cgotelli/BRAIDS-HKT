"""Actual CuPy/CUDA backend for the Q9 fines transaction; no CPU fallback."""
from __future__ import annotations

import cupy as cp

from braids_core.sediment.fines_v106 import FineRuntimeV106


class ResidentFineRuntimeV106(FineRuntimeV106):
    backend = "implicit_fwave_midpoint_gpu"
    xp = cp

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force context creation here so backend failure is explicit at selection.
        self.device_id=int(cp.cuda.Device().id)
        self.identity=self.identity[:-8]+f"{self.device_id:08x}"

    def _host(self,value): return cp.asnumpy(value)
    def _bool(self,value): return bool(value.get() if hasattr(value,"get") else value)
