"""Isolated resident CUDA operators for Delft3D TraFrm 12 and 14.

These point operators are qualification scaffolding, not public coupled GPU
dispatch.  Inputs and outputs stay resident on one CUDA device, and malformed
states fail closed through a per-point status vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)

FORMULA_FIELDS = {
    12: (
        "u", "v", "depth", "d50", "solid_density", "water_density", "gravity",
        "kinematic_viscosity", "acal", "model_index", "ratio_d50_z0", "von_karman",
    ),
    14: (
        "u", "v", "d50", "solid_density", "water_density", "gravity", "chezy",
        "hiding_factor", "acal", "critical_shields", "power_m", "power_p", "power_q",
    ),
}
FORMULA_NAMES = {12: "soulsby", 14: "ashida_michiue"}
FORMULA_IDS = {name: code for code, name in FORMULA_NAMES.items()}
KERNEL_NAMES = {
    12: "delft_soulsby_parameterized",
    14: "delft_ashida_michiue_parameterized",
}


@dataclass(frozen=True)
class ResidentBedloadResult:
    """Resident mass-flux x/y and stress-ratio rows plus point status."""

    formula: int
    values: object
    status: object

    @property
    def mass_flux(self):
        return self.values[0:2]

    @property
    def stress_ratio(self):
        return self.values[2]


class DelftParameterizedBedloadCUDA:
    """Compile and evaluate exact current-only point operators on CUDA."""

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
                name_expressions=tuple(KERNEL_NAMES.values()),
            )
            self.kernels = {
                formula: self.module.get_function(name)
                for formula, name in KERNEL_NAMES.items()
            }

    @staticmethod
    def formula_id(formula: int | str) -> int:
        if isinstance(formula, str):
            try:
                return FORMULA_IDS[formula]
            except KeyError as exc:
                raise ValueError(f"Unsupported parameterized formula: {formula!r}") from exc
        if isinstance(formula, (bool, np.bool_)) or formula not in FORMULA_FIELDS:
            raise ValueError(f"Unsupported parameterized formula: {formula!r}")
        return int(formula)

    def _resident_inputs(self, formula: int, fields: Mapping[str, object]):
        cp = self.cp
        required = FORMULA_FIELDS[formula]
        missing = tuple(name for name in required if name not in fields)
        extra = tuple(name for name in fields if name not in required)
        if missing or extra:
            raise ValueError(
                f"Formula {formula} input fields differ; missing={missing}, extra={extra}"
            )
        arrays = []
        count = None
        for name in required:
            value = fields[name]
            dtype = cp.int32 if name == "model_index" else cp.float64
            if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                    value.ndim != 1 or not value.flags.c_contiguous):
                label = "int32" if name == "model_index" else "FP64"
                raise TypeError(
                    f"{name} must be a resident contiguous one-dimensional {label} CuPy array"
                )
            if int(value.device.id) != self.device_id:
                raise ValueError(f"{name} is not resident on CUDA device {self.device_id}")
            if count is None:
                count = int(value.size)
            elif value.size != count:
                raise ValueError("All resident parameter arrays must have identical lengths")
            arrays.append(value)
        return int(count or 0), tuple(arrays)

    def evaluate(self, formula: int | str, *, reject_invalid: bool = True,
                 **fields) -> ResidentBedloadResult:
        """Evaluate one formula without host-side physical computation."""
        code = self.formula_id(formula)
        count, arrays = self._resident_inputs(code, fields)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            values = cp.empty((3, count), dtype=cp.float64)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernels[code](
                    ((count + 255) // 256,), (256,),
                    (np.int64(count), *arrays, values, status),
                )
            if reject_invalid and count:
                invalid = int(cp.count_nonzero(status).item())
                if invalid:
                    raise ValueError(
                        f"CUDA Delft formula {code} rejected {invalid} invalid state(s)"
                    )
        return ResidentBedloadResult(code, values, status)
