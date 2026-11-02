"""One bounded resident-CUDA point ABI for six scalar FLOW laws.

This additive qualification adapter unifies TraFrm 7, 13, 14 and 16--18
without changing the live GPU dispatcher.  It reuses the already qualified
TraFrm 7/13 and 14 kernels and adds only the missing explicit-input
Wilcock-family point kernel.  Every physical array and numerical result stays
resident on one CUDA device; the default fail-closed check transfers only a
single status reduction.

The operator consumes FLOW's *resolved point inputs*.  In particular, D90 is
``RP_D90MX`` for TraFrm 7/13, ``bed_shear`` is ``RP_TAUB`` for TraFrm 16--18,
and Dg/DgSD/sand fraction have already been constructed by the active-layer
composition operator.  This module does not claim that upstream preprocessing
or live coupling has been qualified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)

FORMULA_FIELDS = MappingProxyType({
    7: (
        "depth", "u", "v", "d50", "d90", "solid_density",
        "water_density", "gravity", "kinematic_viscosity", "acal",
        "reference_height", "settling_velocity", "mud_power", "mud_fraction",
    ),
    13: (
        "depth", "u", "v", "d50", "d90", "solid_density",
        "water_density", "gravity", "vicmol", "chezy",
        "current_shear_velocity", "acal",
    ),
    14: (
        "u", "v", "d50", "solid_density", "water_density", "gravity",
        "chezy", "hiding_factor", "acal", "critical_shields", "power_m",
        "power_p", "power_q",
    ),
    16: (
        "u", "v", "d50", "solid_density", "water_density", "gravity",
        "chezy", "bed_shear", "dg", "sand_fraction", "acal",
    ),
    17: (
        "u", "v", "d50", "solid_density", "water_density", "gravity",
        "chezy", "bed_shear", "dg", "dgsd", "sand_fraction", "theta_c0",
        "alpha0",
    ),
    18: (
        "u", "v", "d50", "solid_density", "water_density", "gravity",
        "chezy", "bed_shear", "dg", "dgsd", "sand_fraction", "theta_c0",
        "alpha0",
    ),
})

FORMULA_NAMES = MappingProxyType({
    7: "van_rijn_1984_modified",
    13: "wang_fredsoe",
    14: "ashida_michiue_generalized",
    16: "wilcock_crowe",
    17: "wilcock_crowe_modified",
    18: "gaeuman",
})
FORMULA_IDS = MappingProxyType({name: code for code, name in FORMULA_NAMES.items()})


@dataclass(frozen=True)
class ResidentScalarCapacityV63Result:
    """Resident rows: bed x/y, suspended x/y, theta, theta_ref, theta ratio."""

    selector: int
    values: object
    status: object

    @property
    def bed_mass(self):
        return self.values[0:2]

    @property
    def suspended_mass(self):
        return self.values[2:4]

    @property
    def absolute_shields(self):
        return self.values[4]

    @property
    def reference_shields(self):
        return self.values[5]

    @property
    def stress_ratio(self):
        return self.values[6]


class DelftScalarCapacityV63CUDA:
    """Evaluate the six bounded formulas through one strict resident ABI."""

    def __init__(self, source_path: str | Path | None = None, *,
                 current_source_path: str | Path | None = None,
                 parameterized_source_path: str | Path | None = None):
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
                name_expressions=("delft_wilcock_scalar_v63",),
            )
            self.wilcock_kernel = self.module.get_function(
                "delft_wilcock_scalar_v63"
            )
        self._current_source_path = current_source_path
        self._parameterized_source_path = parameterized_source_path
        self._current = None
        self._parameterized = None

    @staticmethod
    def formula_id(formula: int | str) -> int:
        if isinstance(formula, str):
            try:
                return FORMULA_IDS[formula]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported scalar FLOW formula: {formula!r}"
                ) from error
        if isinstance(formula, (bool, np.bool_)) or formula not in FORMULA_FIELDS:
            raise ValueError(f"Unsupported scalar FLOW formula: {formula!r}")
        return int(formula)

    def _operator_current(self):
        if self._current is None:
            from .delft_current_capacity import DelftCurrentCapacityCUDA
            self._current = DelftCurrentCapacityCUDA(self._current_source_path)
            if self._current.device_id != self.device_id:  # pragma: no cover
                raise RuntimeError("CUDA dependency selected a different device")
        return self._current

    def _operator_parameterized(self):
        if self._parameterized is None:
            from .delft_parameterized_bedload import DelftParameterizedBedloadCUDA
            self._parameterized = DelftParameterizedBedloadCUDA(
                self._parameterized_source_path
            )
            if self._parameterized.device_id != self.device_id:  # pragma: no cover
                raise RuntimeError("CUDA dependency selected a different device")
        return self._parameterized

    def _resident_inputs(self, selector: int, fields: Mapping[str, object]):
        cp = self.cp
        required = FORMULA_FIELDS[selector]
        missing = tuple(name for name in required if name not in fields)
        extra = tuple(name for name in fields if name not in required)
        if missing or extra:
            raise ValueError(
                f"TraFrm {selector} input fields differ; "
                f"missing={missing}, extra={extra}"
            )
        arrays = {}
        count = None
        for name in required:
            value = fields[name]
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                    value.ndim != 1 or not value.flags.c_contiguous):
                raise TypeError(
                    f"{name} must be a resident contiguous one-dimensional FP64 "
                    "CuPy array"
                )
            if int(value.device.id) != self.device_id:
                raise ValueError(
                    f"{name} is not resident on CUDA device {self.device_id}"
                )
            if count is None:
                count = int(value.size)
            elif int(value.size) != count:
                raise ValueError("All resident scalar-capacity inputs must have identical lengths")
            arrays[name] = value
        return int(count or 0), MappingProxyType(arrays)

    def evaluate(self, formula: int | str, *, reject_invalid: bool = True,
                 **fields) -> ResidentScalarCapacityV63Result:
        selector = self.formula_id(formula)
        count, value = self._resident_inputs(selector, fields)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            if selector == 7:
                raw = self._operator_current().evaluate(
                    7, reject_invalid=False,
                    depth=value["depth"], u=value["u"], v=value["v"],
                    d50=value["d50"], d90=value["d90"],
                    solid_density=value["solid_density"],
                    water_density=value["water_density"], gravity=value["gravity"],
                    kinematic_viscosity=value["kinematic_viscosity"],
                    calibration=value["acal"],
                    reference_height=value["reference_height"],
                    settling_velocity=value["settling_velocity"],
                    mud_power=value["mud_power"], mud_fraction=value["mud_fraction"],
                )
                values = raw.values.copy()
                status = raw.status.copy()
                self._replace_ratio(values, status)
            elif selector == 13:
                raw = self._operator_current().evaluate(
                    13, reject_invalid=False,
                    depth=value["depth"], u=value["u"], v=value["v"],
                    d50=value["d50"], d90=value["d90"],
                    solid_density=value["solid_density"],
                    water_density=value["water_density"], gravity=value["gravity"],
                    kinematic_viscosity=value["vicmol"], chezy=value["chezy"],
                    current_shear_velocity=value["current_shear_velocity"],
                    calibration=value["acal"],
                )
                values = raw.values.copy()
                status = raw.status.copy()
                self._replace_ratio(values, status)
            elif selector == 14:
                raw = self._operator_parameterized().evaluate(
                    14, reject_invalid=False, **value
                )
                values = cp.zeros((7, count), dtype=cp.float64)
                values[0:2] = raw.mass_flux
                status = raw.status.copy()
                relative = value["solid_density"]/cp.where(
                    value["water_density"] != 0.0,
                    value["water_density"], 1.0,
                )-1.0
                denominator = value["chezy"]**2*relative*value["d50"]
                theta = cp.where(
                    denominator > 0.0,
                    (value["u"]**2+value["v"]**2)/cp.where(
                        denominator > 0.0, denominator, 1.0
                    ),
                    0.0,
                )
                values[4] = theta
                values[5] = value["hiding_factor"]*value["critical_shields"]
                values[6] = raw.stress_ratio
                valid = status == 0
                values[:] = cp.where(valid[None, :], values, 0.0)
            else:
                values = cp.empty((7, count), dtype=cp.float64)
                status = cp.empty(count, dtype=cp.int32)
                zeros = cp.zeros(count, dtype=cp.float64)
                acal = value["acal"] if selector == 16 else zeros
                theta_c0 = value["theta_c0"] if selector != 16 else zeros
                alpha0 = value["alpha0"] if selector != 16 else zeros
                # Locked trabwc (TraFrm 16) has no DgSD argument.  Keep the
                # shared CUDA kernel pointer private rather than exposing an
                # inert physical input through the public point ABI.
                dgsd = zeros if selector == 16 else value["dgsd"]
                if count:
                    self.wilcock_kernel(
                        ((count+255)//256,), (256,),
                        (np.int64(count), np.int32(selector), value["u"],
                         value["v"], value["d50"], value["solid_density"],
                         value["water_density"], value["gravity"],
                         value["chezy"], value["bed_shear"], value["dg"],
                         dgsd, value["sand_fraction"], acal,
                         theta_c0, alpha0, values, status),
                    )
            if reject_invalid and count:
                rejected = int(cp.count_nonzero(status).item())
                if rejected:
                    raise ValueError(
                        f"CUDA TraFrm {selector} rejected {rejected} invalid point(s)"
                    )
        return ResidentScalarCapacityV63Result(selector, values, status)

    def _replace_ratio(self, values, status) -> None:
        """Replace formula-specific row 6 with the common theta ratio."""
        cp = self.cp
        theta = values[4]
        reference = values[5]
        ratio = cp.where(
            reference > 0.0,
            theta/cp.where(reference > 0.0, reference, 1.0),
            cp.where(theta > 0.0, np.finfo(np.float64).max, 0.0),
        )
        values[6] = cp.where(status == 0, ratio, 0.0)


__all__ = [
    "CUDA_OPTIONS", "FORMULA_FIELDS", "FORMULA_NAMES",
    "ResidentScalarCapacityV63Result", "DelftScalarCapacityV63CUDA",
]
