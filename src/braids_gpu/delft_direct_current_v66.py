"""Resident CUDA producers for the Q3-H6 direct-current formula family.

This module is deliberately isolated from the public dispatcher.  It closes
the constitutive CPU/CUDA foundation for locked Delft3D4 FLOW TraFrm 5, 19 and
20 without pretending that their different suspended-output ownership is
interchangeable:

* Bijker returns explicit bed and suspended mass vectors.  Its concentration
  row is diagnostic only.
* Van Thiel/Van Rijn 2008 and XBeach Soulsby/Van Rijn return a bed mass vector
  and an equilibrium mass concentration; their explicit suspended rows are
  exactly zero.

All physical arrays stay resident.  Wave activity, classic-FLOW unavailable
state, and formula-internal options that require the D-Flow FM route fail with
distinct device statuses instead of silently falling back.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)

STATUS_OK = 0
STATUS_INVALID = 1
STATUS_WAVE_UNSUPPORTED = 2
STATUS_CLASSIC_STATE_UNSUPPORTED = 3
STATUS_OPTION_UNSUPPORTED = 4
STATUS_NAMES = MappingProxyType({
    STATUS_OK: "ok",
    STATUS_INVALID: "invalid_state",
    STATUS_WAVE_UNSUPPORTED: "wave_forcing_unsupported",
    STATUS_CLASSIC_STATE_UNSUPPORTED: "classic_flow_state_must_be_zero",
    STATUS_OPTION_UNSUPPORTED: "fm_derived_option_unsupported",
})

_WAVE_FIELDS = (
    "wave_height", "wave_period", "wavelength", "wave_angle",
    "orbital_velocity", "breaker_turbulence",
)
_CLASSIC_ZERO_FIELDS = ("bed_change_rate", "bed_slope_x", "bed_slope_y")
_COMMON_1920 = (
    "depth", "u", "v", "d15", "d50", "d90", "solid_density",
    "water_density", "gravity", "kinematic_viscosity", "chezy",
    "facua", "facas", "facsk", "waveform", "sws", "lws", "dilatancy",
    "rheea", "pormax", "bedslpini", "smax", "reposeangle", "cmax",
)

FORMULA_FIELDS = MappingProxyType({
    5: (
        "depth", "u", "v", "d50", "d90", "solid_density", "water_density",
        "gravity", "chezy", "settling_velocity", "cal_bs", "cal_bd",
        "crit_cs", "crit_cd", "roughness_ks", "formula_settling_velocity",
        "porosity", "formula_wave_period", "fac_a", "fac_u", "eps_sl",
        *_WAVE_FIELDS, "bed_slope_x", "bed_slope_y",
    ),
    19: (*_COMMON_1920, "alfad50", *_WAVE_FIELDS, *_CLASSIC_ZERO_FIELDS),
    20: (*_COMMON_1920, "z0", "alfad50", *_WAVE_FIELDS,
         *_CLASSIC_ZERO_FIELDS),
})
INTEGER_FIELDS = MappingProxyType({
    5: frozenset(),
    19: frozenset(("waveform", "sws", "lws", "dilatancy", "bedslpini")),
    20: frozenset(("waveform", "sws", "lws", "dilatancy", "bedslpini")),
})
FORMULA_NAMES = MappingProxyType({
    5: "bijker_current",
    19: "van_thiel_van_rijn_2008_current",
    20: "xbeach_soulsby_van_rijn_current",
})
FORMULA_IDS = MappingProxyType({value: key for key, value in FORMULA_NAMES.items()})
KERNEL_NAMES = MappingProxyType({
    5: "delft_bijker_current_v66",
    19: "delft_van_thiel_van_rijn_current_v66",
    20: "delft_xbeach_soulsby_van_rijn_current_v66",
})
OUTPUT_OWNERSHIP = MappingProxyType({
    5: "explicit_bed_and_suspended_vectors",
    19: "bed_vector_and_equilibrium_concentration",
    20: "bed_vector_and_equilibrium_concentration",
})


@dataclass(frozen=True)
class ResidentDirectCurrentResult:
    """Eight resident FP64 result rows and one resident status vector."""

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
    def concentration_mass(self):
        return self.values[4]

    @property
    def auxiliary(self):
        """Bijker: z, bed scale, speed; 19/20: Ucrb, Ucrs, Ueff."""
        return self.values[5:8]

    @property
    def output_ownership(self) -> str:
        return OUTPUT_OWNERSHIP[self.formula]


class DelftDirectCurrentV66CUDA:
    """Compile and run the isolated direct-current producers on one GPU."""

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
            except KeyError as error:
                raise ValueError(
                    f"Unsupported direct-current formula: {formula!r}"
                ) from error
        if isinstance(formula, (bool, np.bool_)) or type(formula) is not int:
            raise ValueError(f"Unsupported direct-current formula: {formula!r}")
        if formula not in FORMULA_FIELDS:
            raise ValueError(f"Unsupported direct-current formula: {formula!r}")
        return formula

    def _resident_inputs(self, formula: int, fields: dict[str, object]):
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
            integer = name in INTEGER_FIELDS[formula]
            expected = cp.int32 if integer else cp.float64
            if (not isinstance(value, cp.ndarray) or value.dtype != expected or
                    value.ndim != 1 or not value.flags.c_contiguous):
                label = "int32" if integer else "FP64"
                raise TypeError(
                    f"{name} must be a resident contiguous one-dimensional "
                    f"{label} CuPy array"
                )
            if int(value.device.id) != self.device_id:
                raise ValueError(
                    f"{name} is not resident on CUDA device {self.device_id}"
                )
            if count is None:
                count = int(value.size)
            elif int(value.size) != count:
                raise ValueError("All direct-current inputs must have identical lengths")
            arrays.append(value)
        return int(count or 0), arrays

    def evaluate(self, formula: int | str, *, reject_invalid: bool = True,
                 **fields) -> ResidentDirectCurrentResult:
        code = self.formula_id(formula)
        count, arrays = self._resident_inputs(code, fields)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            # The CUDA 19/20 kernels share one ABI.  TraFrm=19 has no z0
            # keyword; insert a resident inert value at the exact kernel
            # position and on the operator's authenticated device.
            if code == 19:
                z0_position = _COMMON_1920.index("cmax") + 1
                # Kernel order is common through cmax, followed by z0 then
                # alfad50.
                arrays.insert(
                    z0_position, cp.full(count, 0.006, dtype=cp.float64),
                )
            values = cp.empty((8, count), dtype=cp.float64)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernels[code](
                    ((count + 255)//256,), (256,),
                    (np.int64(count), *arrays, values, status),
                )
            if reject_invalid and count:
                bad = status != STATUS_OK
                bad_count = int(cp.count_nonzero(bad).item())
                if bad_count:
                    unique, counts = cp.unique(status[bad], return_counts=True)
                    detail = ", ".join(
                        f"{STATUS_NAMES.get(int(key), f'status_{int(key)}')}={int(value)}"
                        for key, value in zip(cp.asnumpy(unique), cp.asnumpy(counts))
                    )
                    raise ValueError(
                        f"CUDA direct-current formula {code} rejected "
                        f"{bad_count} state(s): {detail}"
                    )
        return ResidentDirectCurrentResult(code, values, status)


__all__ = [
    "CUDA_OPTIONS", "DelftDirectCurrentV66CUDA", "FORMULA_FIELDS",
    "FORMULA_IDS", "FORMULA_NAMES", "INTEGER_FIELDS", "KERNEL_NAMES",
    "OUTPUT_OWNERSHIP", "ResidentDirectCurrentResult", "STATUS_CLASSIC_STATE_UNSUPPORTED",
    "STATUS_INVALID", "STATUS_NAMES", "STATUS_OK", "STATUS_OPTION_UNSUPPORTED",
    "STATUS_WAVE_UNSUPPORTED",
]
