"""Resident CUDA C/M/W roughness operator; never selects or falls back to CPU."""
from __future__ import annotations

from pathlib import Path

import numpy as np


CUDA_OPTIONS = ("--std=c++11", "--fmad=false", "--prec-div=true",
                "--prec-sqrt=true", "--ftz=false")
KERNEL_NAMES = ("hydraulic_roughness_state", "hydraulic_roughness_shear",
                "hydraulic_roughness_friction_rate")
ROUGHNESS_MODES = {"chezy": 1, "manning": 2, "white_colebrook": 3}


class HydraulicRoughnessCUDA:
    """Compile and execute the bounded pointwise roughness kernels."""

    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        self.module = cp.RawModule(code=source, options=CUDA_OPTIONS,
                                   name_expressions=KERNEL_NAMES)
        self.kernels = {name: self.module.get_function(name) for name in KERNEL_NAMES}

    @staticmethod
    def _mode(mode: str | int) -> int:
        value = ROUGHNESS_MODES.get(mode, mode)
        if isinstance(value, bool) or value not in (1, 2, 3):
            raise ValueError("Roughness mode must be chezy, manning or white_colebrook")
        return int(value)

    def _inputs(self, depth, parameter):
        cp = self.cp
        if not isinstance(depth, cp.ndarray) or not isinstance(parameter, cp.ndarray):
            raise TypeError("CUDA roughness inputs must already be resident CuPy arrays")
        if (depth.dtype != cp.float64 or parameter.dtype != cp.float64 or
                depth.ndim != 1 or parameter.shape != depth.shape or
                not depth.flags.c_contiguous or not parameter.flags.c_contiguous):
            raise ValueError("CUDA roughness inputs must be same-sized contiguous FP64 vectors")
        return depth, parameter

    def _finish(self, outputs, status):
        # Keep fields and the per-cell status resident.  The only device/host
        # transfer is a one-scalar fail-closed reduction.
        if status.size and bool(self.cp.any(status).item()):
            raise ValueError("Invalid CUDA hydraulic roughness state")
        return (*outputs, status)

    def state(self, depth, parameter, *, mode, gravity=9.81,
              minimum_work_depth=0.0, waqua_white_colebrook=False):
        cp = self.cp
        depth_d, parameter_d = self._inputs(depth, parameter)
        count = int(depth_d.size)
        outputs = tuple(cp.empty(count, dtype=cp.float64) for _ in range(3))
        status = cp.empty(count, dtype=cp.int32)
        if count:
            self.kernels["hydraulic_roughness_state"](
                ((count + 255) // 256,), (256,),
                (np.int64(count), np.int32(self._mode(mode)), depth_d, parameter_d,
                 np.float64(gravity), np.float64(minimum_work_depth),
                 np.int32(bool(waqua_white_colebrook)), *outputs, status))
        return self._finish(outputs, status)

    def shear(self, depth, speed, parameter, *, density=1000.0, mode,
              gravity=9.81, minimum_work_depth=0.0,
              waqua_white_colebrook=False):
        cp = self.cp
        depth_d, parameter_d = self._inputs(depth, parameter)
        if (not isinstance(speed, cp.ndarray) or speed.dtype != cp.float64 or
                speed.shape != depth_d.shape or not speed.flags.c_contiguous):
            raise ValueError("CUDA speed must be a same-sized contiguous resident FP64 vector")
        count = int(depth_d.size)
        outputs = tuple(cp.empty(count, dtype=cp.float64) for _ in range(3))
        status = cp.empty(count, dtype=cp.int32)
        if count:
            self.kernels["hydraulic_roughness_shear"](
                ((count + 255) // 256,), (256,),
                (np.int64(count), np.int32(self._mode(mode)), depth_d, speed,
                 parameter_d, np.float64(density), np.float64(gravity),
                 np.float64(minimum_work_depth),
                 np.int32(bool(waqua_white_colebrook)), *outputs, status))
        return self._finish(outputs, status)

    def friction_rate(self, depth, speed, parameter, *, mode, gravity=9.81,
                      minimum_work_depth=0.0,
                      waqua_white_colebrook=False):
        cp = self.cp
        depth_d, parameter_d = self._inputs(depth, parameter)
        if (not isinstance(speed, cp.ndarray) or speed.dtype != cp.float64 or
                speed.shape != depth_d.shape or not speed.flags.c_contiguous):
            raise ValueError("CUDA speed must be a same-sized contiguous resident FP64 vector")
        count = int(depth_d.size)
        rate = cp.empty(count, dtype=cp.float64)
        status = cp.empty(count, dtype=cp.int32)
        if count:
            self.kernels["hydraulic_roughness_friction_rate"](
                ((count + 255) // 256,), (256,),
                (np.int64(count), np.int32(self._mode(mode)), depth_d, speed,
                 parameter_d, np.float64(gravity), np.float64(minimum_work_depth),
                 np.int32(bool(waqua_white_colebrook)), rate, status))
        return self._finish((rate,), status)
