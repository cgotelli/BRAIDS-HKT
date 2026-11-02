"""Resident CUDA hiding/exposure factors for graded noncohesive sediment.

This operator is the GPU counterpart of
``solver/wave/braids_hiding_exposure_module.f90`` and the locked Delft3D4
``comphidexp``/``compdiam`` arithmetic used by that module.  A state is one
active-layer mixture; arrays use the existing BRAIDS fraction-major layout
``(nclass, nstate)``.

The implementation is deliberately isolated from public profile dispatch.
It accepts only already-resident FP64 CuPy arrays, never selects a backend and
has no CPU fallback.  Invalid states retain neutral factors (one), a zero mean
diameter and a nonzero resident status.  Outputs from a nonzero-status state
must not be consumed.
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
KERNEL_NAME = "hiding_exposure_factors"
SUPPORTED_MODES = frozenset((0, 2, 3, 4, 5))

# Per-state device status.  Shape/type/mode errors are rejected by the host
# before launch because no well-defined output shape or kernel route exists.
STATUS_OK = 0
STATUS_INVALID_INPUT = 1
STATUS_DEPLETED = 2
STATUS_SINGULAR_FORMULA = 3
STATUS_INVALID_OUTPUT = 4


@dataclass(frozen=True)
class ResidentHidingExposureResult:
    """Fraction factors, arithmetic mean diameter and per-state status."""

    mode: int
    factors: object
    mean_diameter: object
    status: object


class HidingExposureCUDA:
    """Compile and execute Delft3D-compatible hiding/exposure on one GPU."""

    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source,
                options=CUDA_OPTIONS,
                name_expressions=(KERNEL_NAME,),
            )
            self.kernel = self.module.get_function(KERNEL_NAME)

    @staticmethod
    def _mode(mode: int) -> int:
        if isinstance(mode, (bool, np.bool_)):
            raise ValueError("Hiding/exposure mode must be one of 0, 2, 3, 4 or 5")
        try:
            value = operator.index(mode)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Hiding/exposure mode must be one of 0, 2, 3, 4 or 5"
            ) from exc
        if value not in SUPPORTED_MODES:
            raise ValueError("Hiding/exposure mode must be one of 0, 2, 3, 4 or 5")
        return value

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

    def _resident_array(self, name: str, value: object, ndim: int):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.ndim != ndim or not value.flags.c_contiguous):
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D FP64 CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        return value

    def evaluate(
        self,
        active_volume,
        d50,
        *,
        mode: int,
        parker_alpha: float = 0.6,
        wu_exponent: float = -0.6,
        reject_invalid: bool = True,
    ) -> ResidentHidingExposureResult:
        """Evaluate one column per mixture, keeping every field resident.

        ``active_volume`` has shape ``(nclass, nstate)`` and ``d50`` has shape
        ``(nclass,)``.  As in the native routine, even a mode-0 state must have
        positive total active volume and valid finite controls.  By default a
        one-scalar device reduction rejects any failed state; diagnostics may
        request ``reject_invalid=False`` and inspect the resident status array.
        """
        selected_mode = self._mode(mode)
        alpha = self._finite_scalar("parker_alpha", parker_alpha)
        exponent = self._finite_scalar("wu_exponent", wu_exponent)
        if not isinstance(reject_invalid, (bool, np.bool_)):
            raise TypeError("reject_invalid must be boolean")

        active_volume = self._resident_array("active_volume", active_volume, 2)
        d50 = self._resident_array("d50", d50, 1)
        nclass, nstate = map(int, active_volume.shape)
        if nclass < 1:
            raise ValueError("At least one sediment class is required")
        if tuple(d50.shape) != (nclass,):
            raise ValueError(
                f"d50 must have shape ({nclass},), got {tuple(d50.shape)}"
            )

        cp = self.cp
        with cp.cuda.Device(self.device_id):
            factors = cp.empty_like(active_volume)
            mean_diameter = cp.empty(nstate, dtype=cp.float64)
            status = cp.empty(nstate, dtype=cp.int32)
            if nstate:
                self.kernel(
                    ((nstate + 255) // 256,),
                    (256,),
                    (
                        np.int64(nstate),
                        np.int32(nclass),
                        np.int32(selected_mode),
                        active_volume,
                        d50,
                        np.float64(alpha),
                        np.float64(exponent),
                        factors,
                        mean_diameter,
                        status,
                    ),
                )
            if bool(reject_invalid) and nstate:
                invalid_count = int(cp.count_nonzero(status).item())
                if invalid_count:
                    raise ValueError(
                        "CUDA hiding/exposure rejected "
                        f"{invalid_count} invalid state(s)"
                    )

        return ResidentHidingExposureResult(
            selected_mode, factors, mean_diameter, status,
        )


__all__ = [
    "CUDA_OPTIONS",
    "HidingExposureCUDA",
    "KERNEL_NAME",
    "ResidentHidingExposureResult",
    "STATUS_DEPLETED",
    "STATUS_INVALID_INPUT",
    "STATUS_INVALID_OUTPUT",
    "STATUS_OK",
    "STATUS_SINGULAR_FORMULA",
    "SUPPORTED_MODES",
]
