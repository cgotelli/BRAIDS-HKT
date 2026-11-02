"""Isolated resident-CUDA Soulsby/Clarke sediment skin-shear operator.

The operator mirrors the current-only branch of locked Delft3D4
``compbsskin`` and the BRAIDS CPU transcription.  Its first result row is
strictly the sediment ``RP_TAUB`` substitute.  The API deliberately exposes no
hydraulic Ch\N{LATIN SMALL LETTER E WITH ACUTE}zy, hydraulic roughness, friction, effective velocity, or
momentum state, so selecting skin shear cannot mutate the hydraulic chain.

Every physical input and output remains a contiguous FP64 CuPy vector on one
CUDA device.  There is no CPU constitutive fallback and this research operator
is not connected to public/runtime/QGIS dispatch.
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
KERNEL_NAME = "sediment_soulsby_clarke_skin_shear"
INPUT_FIELDS = (
    "depth",
    "speed",
    "kinematic_viscosity",
    "water_density",
    "ks_sand",
    "ks_silt",
    "mud_factor",
    "critical_mud_1",
    "critical_mud_2",
    "fluff_fraction",
)


@dataclass(frozen=True)
class ResidentSkinShearResult:
    """Resident sediment-only shear result and diagnostic coefficients."""

    rp_taub: object
    z0_skin: object
    cd_rough: object
    cd_smooth: object
    status: object


class SedimentSkinShearCUDA:
    """Compile and run the isolated pointwise skin-shear operator."""

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

    def _resident_inputs(self, fields: Mapping[str, object]):
        missing = tuple(name for name in INPUT_FIELDS if name not in fields)
        extra = tuple(name for name in fields if name not in INPUT_FIELDS)
        if missing or extra:
            raise ValueError(
                f"Skin-shear input fields differ; missing={missing}, extra={extra}"
            )

        cp = self.cp
        arrays = []
        count = None
        for name in INPUT_FIELDS:
            value = fields[name]
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                    value.ndim != 1 or not value.flags.c_contiguous):
                raise TypeError(
                    f"{name} must be a resident contiguous one-dimensional "
                    "FP64 CuPy array"
                )
            if int(value.device.id) != self.device_id:
                raise ValueError(
                    f"{name} is not resident on CUDA device {self.device_id}"
                )
            if count is None:
                count = int(value.size)
            elif int(value.size) != count:
                raise ValueError(
                    "All resident skin-shear inputs must have identical lengths"
                )
            arrays.append(value)
        return int(count or 0), tuple(arrays)

    def evaluate(self, *, reject_invalid: bool = True,
                 **fields) -> ResidentSkinShearResult:
        """Return a resident sediment ``RP_TAUB`` substitute.

        Invalid entries are zeroed and marked on device.  They raise by
        default; ``reject_invalid=False`` is intended only for bounded
        diagnostics.  The status reduction is the only default host transfer.
        """
        count, arrays = self._resident_inputs(fields)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            output = cp.empty((4, count), dtype=cp.float64)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernel(
                    ((count + 255) // 256,), (256,),
                    (np.int64(count), *arrays, output, status),
                )
            if reject_invalid and count:
                invalid_count = int(cp.count_nonzero(status).item())
                if invalid_count:
                    raise ValueError(
                        "CUDA sediment skin shear rejected "
                        f"{invalid_count} invalid state(s)"
                    )
        return ResidentSkinShearResult(
            output[0], output[1], output[2], output[3], status,
        )
