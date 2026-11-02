"""Isolated resident-CUDA Delft3D4 FLOW bed-slope direction operator.

The kernel mirrors the locked ``adjust_bedload.f90`` ``ISlope=2/3/4``
constitutive branches.  It does not select a backend, copy host arrays, fall
back to CPU, or participate in public profile dispatch.  Integration into the
accepted face path is deliberately a later Q3-H gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator
from pathlib import Path

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAME = "flow_bed_slope_direction"
SUPPORTED_MODES = frozenset((2, 3, 4))

STATUS_OK = 0
STATUS_INVALID_MODE = 1
STATUS_INVALID_INPUT = 2
STATUS_SINGULAR = 3
STATUS_INVALID_OUTPUT = 4


@dataclass(frozen=True)
class ResidentFlowBedSlopeResult:
    """Corrected resident transport vectors and per-state status."""

    mode: int
    transport_x: object
    transport_y: object
    status: object


class FlowBedSlopeCUDA:
    """Compile and execute the isolated FP64 FLOW direction operator."""

    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=path.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS,
                name_expressions=(KERNEL_NAME,),
            )
            self.kernel = self.module.get_function(KERNEL_NAME)

    @staticmethod
    def _mode(mode: int) -> int:
        if isinstance(mode, (bool, np.bool_)):
            raise ValueError("FLOW bed-slope mode must be 2, 3 or 4")
        try:
            selected = operator.index(mode)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("FLOW bed-slope mode must be 2, 3 or 4") from exc
        if selected not in SUPPORTED_MODES:
            raise ValueError("FLOW bed-slope mode must be 2, 3 or 4")
        return selected

    @staticmethod
    def _finite_scalar(name: str, value: float) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite scalar")
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if not np.isfinite(result):
            raise ValueError(f"{name} must be a finite scalar")
        return result

    def _array(self, name: str, value: object, shape=None):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.ndim != 1 or not value.flags.c_contiguous):
            raise TypeError(
                f"{name} must be a resident contiguous 1-D FP64 CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        return value

    def evaluate(
        self,
        transport_x,
        transport_y,
        downhill_x,
        downhill_y,
        stress,
        depth,
        d50,
        mean_diameter,
        hiding_left,
        hiding_right,
        *,
        mode: int,
        alfabs: float = 1.0,
        alfabn: float = 1.5,
        ashld: float = 0.85,
        bshld: float = 0.5,
        cshld: float = 0.0,
        dshld: float = 0.0,
        alfpa: float = 0.4,
        thcrpa: float = 0.047,
        reject_invalid: bool = True,
    ) -> ResidentFlowBedSlopeResult:
        """Correct one vector per resident state.

        ``stress`` is FLOW's ``taurat`` for mode 2 and absolute Shields stress
        for modes 3/4.  Modes 3/4 consume the geometric inputs documented by
        their equations; unused arrays remain shape-checked but are not read.
        """
        selected = self._mode(mode)
        controls = {
            name: self._finite_scalar(name, value)
            for name, value in (
                ("alfabs", alfabs), ("alfabn", alfabn),
                ("ashld", ashld), ("bshld", bshld),
                ("cshld", cshld), ("dshld", dshld),
                ("alfpa", alfpa), ("thcrpa", thcrpa),
            )
        }
        if not isinstance(reject_invalid, (bool, np.bool_)):
            raise TypeError("reject_invalid must be boolean")

        transport_x = self._array("transport_x", transport_x)
        shape = tuple(transport_x.shape)
        arrays = [transport_x]
        for name, value in (
            ("transport_y", transport_y), ("downhill_x", downhill_x),
            ("downhill_y", downhill_y), ("stress", stress),
            ("depth", depth), ("d50", d50),
            ("mean_diameter", mean_diameter),
            ("hiding_left", hiding_left), ("hiding_right", hiding_right),
        ):
            arrays.append(self._array(name, value, shape))

        cp = self.cp
        count = int(shape[0])
        with cp.cuda.Device(self.device_id):
            corrected_x = cp.empty_like(transport_x)
            corrected_y = cp.empty_like(transport_x)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernel(
                    ((count + 255) // 256,),
                    (256,),
                    (
                        np.int64(count), np.int32(selected), *arrays,
                        *[np.float64(controls[name]) for name in (
                            "alfabs", "alfabn", "ashld", "bshld",
                            "cshld", "dshld", "alfpa", "thcrpa",
                        )],
                        corrected_x, corrected_y, status,
                    ),
                )
            if bool(reject_invalid) and count:
                invalid_count = int(cp.count_nonzero(status).item())
                if invalid_count:
                    raise ValueError(
                        "CUDA FLOW bed-slope operator rejected "
                        f"{invalid_count} invalid state(s)"
                    )
        return ResidentFlowBedSlopeResult(
            selected, corrected_x, corrected_y, status,
        )


__all__ = [
    "CUDA_OPTIONS",
    "FlowBedSlopeCUDA",
    "KERNEL_NAME",
    "ResidentFlowBedSlopeResult",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_MODE",
    "STATUS_INVALID_OUTPUT",
    "STATUS_OK",
    "STATUS_SINGULAR",
    "SUPPORTED_MODES",
]
