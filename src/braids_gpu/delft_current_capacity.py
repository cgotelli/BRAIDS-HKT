"""Experimental resident CUDA operators for four Delft3D current capacities.

This module is intentionally isolated from BRAIDS' production method catalogue
and coupled sediment runtime.  It accepts only same-device, contiguous FP64
CuPy vectors and never evaluates a constitutive equation on the CPU.
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
    4: (
        "u", "v", "d50", "solid_density", "water_density", "gravity", "chezy",
        "hiding_factor", "bed_calibration", "bed_shields_power", "bed_excess_power",
        "bed_ripple_factor", "bed_critical_shields", "sus_calibration",
        "sus_shields_power", "sus_excess_power", "sus_ripple_factor",
        "sus_critical_shields",
    ),
    7: (
        "depth", "u", "v", "d50", "d90", "solid_density", "water_density",
        "gravity", "kinematic_viscosity", "calibration", "reference_height",
        "settling_velocity", "mud_power", "mud_fraction",
    ),
    11: (
        "depth", "u", "v", "d50", "solid_density", "water_density", "gravity",
        "kinematic_viscosity", "calibration", "d90_d50", "roughness_z0",
    ),
    13: (
        "depth", "u", "v", "d50", "d90", "solid_density", "water_density",
        "gravity", "kinematic_viscosity", "chezy", "current_shear_velocity",
        "calibration",
    ),
}

FORMULA_NAMES = {
    4: "general",
    7: "modified_van_rijn_1984",
    11: "soulsby_van_rijn",
    13: "wang_fredsoe",
}
FORMULA_IDS = {name: code for code, name in FORMULA_NAMES.items()}
KERNEL_NAMES = {
    4: "delft_general_capacity",
    7: "delft_modified_van_rijn_1984",
    11: "delft_soulsby_van_rijn",
    13: "delft_wang_fredsoe_capacity",
}


@dataclass(frozen=True)
class ResidentCapacityResult:
    """Seven resident FP64 result rows plus a resident per-point status."""

    formula: int
    values: object
    status: object

    @property
    def bed_mass(self):
        return self.values[0:2]

    @property
    def suspended_mass(self):
        return self.values[2:4]

    @property
    def shields(self):
        return self.values[4]

    @property
    def auxiliary(self):
        """Formula-specific rows: critical Shields and stage/speed/total mass."""
        return self.values[5:7]


class DelftCurrentCapacityCUDA:
    """Compile and run isolated pointwise operators on resident CUDA arrays."""

    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        names = tuple(KERNEL_NAMES.values())
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=names,
            )
            self.kernels = {
                code: self.module.get_function(name)
                for code, name in KERNEL_NAMES.items()
            }

    @staticmethod
    def formula_id(formula: int | str) -> int:
        if isinstance(formula, str):
            try:
                return FORMULA_IDS[formula]
            except KeyError as exc:
                raise ValueError(f"Unsupported Delft current capacity formula: {formula!r}") from exc
        if isinstance(formula, (bool, np.bool_)) or formula not in FORMULA_FIELDS:
            raise ValueError(f"Unsupported Delft current capacity formula: {formula!r}")
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
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                    value.ndim != 1 or not value.flags.c_contiguous):
                raise TypeError(
                    f"{name} must be a resident contiguous one-dimensional FP64 CuPy array"
                )
            if int(value.device.id) != self.device_id:
                raise ValueError(f"{name} is not resident on CUDA device {self.device_id}")
            if count is None:
                count = int(value.size)
            elif value.size != count:
                raise ValueError("All resident capacity inputs must have identical lengths")
            arrays.append(value)
        return int(count or 0), tuple(arrays)

    def evaluate(self, formula: int | str, *, reject_invalid: bool = True,
                 **fields) -> ResidentCapacityResult:
        """Evaluate one formula without copying physical inputs or results to CPU.

        A nonzero device status raises by default.  ``reject_invalid=False`` is
        provided only for bounded diagnostics; failed entries remain zero and
        their resident status is one.
        """
        code = self.formula_id(formula)
        count, arrays = self._resident_inputs(code, fields)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            values = cp.empty((7, count), dtype=cp.float64)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernels[code](
                    ((count + 255) // 256,), (256,),
                    (np.int64(count), *arrays, values, status),
                )
            if reject_invalid and count:
                invalid_count = int(cp.count_nonzero(status).item())
                if invalid_count:
                    raise ValueError(
                        f"CUDA Delft formula {code} rejected {invalid_count} invalid state(s)"
                    )
        return ResidentCapacityResult(code, values, status)

