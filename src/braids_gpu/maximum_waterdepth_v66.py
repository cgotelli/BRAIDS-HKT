"""Resident CUDA wrapper for the Q3 v66 ``MaximumWaterdepth`` operator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)


@dataclass(frozen=True)
class ResidentMaximumWaterdepthV66:
    observation_depth: object
    characteristic_u: object
    characteristic_v: object
    characteristic_speed: object
    observation_height: object
    status: object


class MaximumWaterdepthCUDAv66:
    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        source = Path(source_path or Path(__file__).with_suffix(".cu")).read_text(
            encoding="utf-8"
        )
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source,
                options=CUDA_OPTIONS,
                name_expressions=("maximum_waterdepth_v66",),
            )
            self.kernel = self.module.get_function("maximum_waterdepth_v66")

    def _array(self, name: str, value: object, *, dtype, ndim: int,
               size: int | None = None):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be a resident contiguous {dtype} array")
        if int(value.device.id) != self.device_id:
            raise ValueError(f"{name} is resident on a different CUDA device")
        if size is not None and int(value.size) != size:
            raise ValueError(f"{name} must contain {size} values")
        return value

    def apply(self, *, cell_depth, transport_u, transport_v, active,
              support_offset, support_depth, support_active, enabled: bool,
              direction_epsilon: float = 1.0e-12,
              reject_invalid: bool = True) -> ResidentMaximumWaterdepthV66:
        cp = self.cp
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError("enabled must be Boolean")
        epsilon = float(direction_epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("direction_epsilon must be finite and positive")
        depth = self._array("cell_depth", cell_depth, dtype=cp.float64, ndim=1)
        ncell = int(depth.size)
        numerator_u = self._array(
            "transport_u", transport_u, dtype=cp.float64, ndim=1, size=ncell)
        numerator_v = self._array(
            "transport_v", transport_v, dtype=cp.float64, ndim=1, size=ncell)
        active = self._array("active", active, dtype=cp.int32, ndim=1, size=ncell)
        offset = self._array(
            "support_offset", support_offset, dtype=cp.int64, ndim=1,
            size=ncell + 1)
        links = self._array(
            "support_depth", support_depth, dtype=cp.float64, ndim=1)
        link_active = self._array(
            "support_active", support_active, dtype=cp.int32, ndim=1,
            size=int(links.size))
        with cp.cuda.Device(self.device_id):
            observation = cp.empty(ncell, dtype=cp.float64)
            u = cp.empty(ncell, dtype=cp.float64)
            v = cp.empty(ncell, dtype=cp.float64)
            speed = cp.empty(ncell, dtype=cp.float64)
            height = cp.empty(ncell, dtype=cp.float64)
            status = cp.empty(ncell, dtype=cp.int32)
            if ncell:
                self.kernel(
                    ((ncell + 255) // 256,), (256,),
                    (np.int64(ncell), np.int64(int(links.size)), depth,
                     numerator_u, numerator_v, active, offset, links,
                     link_active, np.int32(bool(enabled)), np.float64(epsilon),
                     observation, u, v, speed, height, status),
                )
            if reject_invalid and ncell:
                count = int(cp.count_nonzero(status).item())
                if count:
                    raise ValueError(
                        f"MaximumWaterdepth CUDA rejected {count} cell(s)")
        return ResidentMaximumWaterdepthV66(
            observation, u, v, speed, height, status)


__all__ = ["MaximumWaterdepthCUDAv66", "ResidentMaximumWaterdepthV66"]
