"""Isolated resident-CUDA operators for public BRAIDS river capacities.

This research module covers the CPU ``bedlaw`` values that are not part of
the existing MPM/current-law CUDA paths: Van Rijn 1993 (1), Wilcock--Crowe
(5), modified Wilcock--Crowe (6), and Gaeuman (7).  It is deliberately not
connected to the method catalogue, QGIS, or a coupled runtime.

All physical inputs and numerical outputs are contiguous FP64 CuPy arrays on
one CUDA device.  The only host transfer performed by the default fail-closed
interface is the reduction of the device status vector.  No constitutive
equation has a host implementation or CPU fallback here.

For laws 5--7, ``bed_shear`` is Delft3D's ``RP_TAUB`` input.  It may be total
hydraulic shear or an explicitly supplied sediment skin shear.  The transport
scale remains based on the total hydraulic Chezy chain, as in ``trabwc``,
``trabwc2``, and ``trabg``; it must not be silently replaced by
``sqrt(bed_shear / water_density)``.  :meth:`total_shear` supplies the ordinary
total-shear chain entirely on device, while callers with a qualified skin
closure can pass that closure's resident output explicitly.

The formula parameters follow locked ``rdtrafrm``: law 5 accepts ``acal``;
laws 6 and 7 require explicit ``theta_c0`` and ``alpha0`` arrays.  Their Delft
defaults are configuration defaults, not constants in this operator.  The
current CPU public wrapper still hardcodes those two pairs, so non-default
6/7 parameters remain a CPU/runtime integration gap and are tested here only
against the locked scalar formula sources.
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

LAW_NAMES = {
    1: "van_rijn_1993",
    5: "wilcock_crowe",
    6: "modified_wilcock_crowe",
    7: "gaeuman",
}
LAW_IDS = {name: code for code, name in LAW_NAMES.items()}
WILCOCK_LAWS = frozenset((5, 6, 7))

LEGACY_CPU_PARAMETER_GAP = (
    "braids_current_bedload hardcodes law-6 ThetaC0/Alpha0=0.021/0.33 and "
    "law-7 ThetaC0/Alpha0=0.03/0.30; runtime persistence and dispatch of "
    "non-default values are not implemented by this isolated CUDA operator"
)

TOTAL_SHEAR_FIELDS = (
    "depth", "u", "v", "manning_n", "water_density", "gravity",
)
VAN_RIJN_CELL_FIELDS = (
    "depth", "u", "v", "observation_depth", "water_density", "gravity",
    "kinematic_viscosity",
)
VAN_RIJN_CLASS_FIELDS = (
    "d50", "d90", "solid_density", "minimum_depth",
)
VAN_RIJN_MATRIX_FIELDS = ("roughness_z0",)
WILCOCK_COMMON_CELL_FIELDS = (
    "depth", "u", "v", "manning_n", "bed_shear", "water_density",
    "gravity",
)
WILCOCK_LAW_FIELDS = {
    5: ("acal",),
    6: ("theta_c0", "alpha0"),
    7: ("theta_c0", "alpha0"),
}
WILCOCK_CLASS_FIELDS = ("d50", "solid_density", "bulk_density")
WILCOCK_MATRIX_FIELDS = ("surface_mass",)
WILCOCK_INTEGER_FIELDS = ("sediment_type",)

KERNEL_NAMES = (
    "river_total_shear",
    "river_vr1993_capacity",
    "river_wilcock_capacity",
)


@dataclass(frozen=True)
class ResidentTotalShearResult:
    """Resident total hydraulic shear and Chezy vectors."""

    bed_shear: object
    chezy: object
    status: object


@dataclass(frozen=True)
class ResidentRiverCapacityResult:
    """Resident per-class capacity and diagnostics.

    ``capacity`` has shape ``(2, nclass, ncell)``.  Diagnostic row meanings
    are law-specific:

    * law 1: stress ratio, total-current shear velocity, ripple/skin factor,
      dimensionless grain diameter, and critical shear;
    * laws 5--7: mobility ratio, total-current shear velocity, dimensionless
      transport rate, class reference shear, and mixture reference shear.

    ``mixture`` is ``None`` for law 1.  For laws 5--7 its three rows are
    geometric mean diameter, geometric standard deviation, and surface sand
    fraction, all evaluated from non-mud volume shares.
    """

    law: int
    capacity: object
    diagnostics: object
    status: object
    mixture: object | None = None

    @property
    def tau_ratio(self):
        return self.diagnostics[0]


class RiverCapacityCUDA:
    """Compile and launch the isolated river-capacity CUDA operators."""

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
                name_expressions=KERNEL_NAMES,
            )
            self.kernels = {
                name: self.module.get_function(name) for name in KERNEL_NAMES
            }

    @staticmethod
    def law_id(law: int | str) -> int:
        if isinstance(law, str):
            try:
                return LAW_IDS[law]
            except KeyError as exc:
                raise ValueError(f"Unsupported river capacity law: {law!r}") from exc
        if isinstance(law, (bool, np.bool_)) or law not in LAW_NAMES:
            raise ValueError(f"Unsupported river capacity law: {law!r}")
        return int(law)

    def _array(self, name: str, value: object, *, ndim: int,
               dtype: object | None = None):
        cp = self.cp
        expected = cp.float64 if dtype is None else dtype
        if (not isinstance(value, cp.ndarray) or value.dtype != expected or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "FP64" if expected == cp.float64 else "int32"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(f"{name} is not resident on CUDA device {self.device_id}")
        return value

    @staticmethod
    def _exact_fields(fields: Mapping[str, object], required: tuple[str, ...],
                      context: str) -> None:
        missing = tuple(name for name in required if name not in fields)
        extra = tuple(name for name in fields if name not in required)
        if missing or extra:
            raise ValueError(
                f"{context} input fields differ; missing={missing}, extra={extra}"
            )

    def _raise_device_status(self, status, context: str) -> None:
        invalid_count = int(self.cp.count_nonzero(status).item())
        if invalid_count:
            raise ValueError(f"CUDA {context} rejected {invalid_count} invalid state(s)")

    def total_shear(self, *, reject_invalid: bool = True,
                    **fields) -> ResidentTotalShearResult:
        """Evaluate the ordinary total hydraulic ``RP_TAUB`` chain on device.

        Zero depth or zero Manning resistance is a valid inactive state and
        returns zero shear/Ch\N{LATIN SMALL LETTER E WITH ACUTE}zy.  Negative or non-finite states fail closed.
        """
        self._exact_fields(fields, TOTAL_SHEAR_FIELDS, "total shear")
        arrays = [self._array(name, fields[name], ndim=1)
                  for name in TOTAL_SHEAR_FIELDS]
        count = int(arrays[0].size)
        if any(int(value.size) != count for value in arrays[1:]):
            raise ValueError("All resident total-shear inputs must have identical lengths")
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            values = cp.empty((2, count), dtype=cp.float64)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernels["river_total_shear"](
                    ((count + 255) // 256,), (256,),
                    (np.int64(count), *arrays, values, status),
                )
            if reject_invalid and count:
                self._raise_device_status(status, "total shear")
        return ResidentTotalShearResult(values[0], values[1], status)

    def evaluate(self, law: int | str, *, reject_invalid: bool = True,
                 **fields) -> ResidentRiverCapacityResult:
        """Evaluate one public CPU river law using only resident CUDA arrays."""
        code = self.law_id(law)
        if code == 1:
            return self._evaluate_van_rijn(reject_invalid=reject_invalid, **fields)
        return self._evaluate_wilcock(
            code, reject_invalid=reject_invalid, **fields,
        )

    def _evaluate_van_rijn(self, *, reject_invalid: bool,
                           **fields) -> ResidentRiverCapacityResult:
        required = VAN_RIJN_CELL_FIELDS + VAN_RIJN_CLASS_FIELDS + VAN_RIJN_MATRIX_FIELDS
        self._exact_fields(fields, required, "Van Rijn 1993")
        cells = [self._array(name, fields[name], ndim=1)
                 for name in VAN_RIJN_CELL_FIELDS]
        classes = [self._array(name, fields[name], ndim=1)
                   for name in VAN_RIJN_CLASS_FIELDS]
        matrices = [self._array(name, fields[name], ndim=2)
                    for name in VAN_RIJN_MATRIX_FIELDS]
        ncell = int(cells[0].size)
        nclass = int(classes[0].size)
        if nclass <= 0:
            raise ValueError("Van Rijn 1993 requires at least one sediment class")
        if any(int(value.size) != ncell for value in cells[1:]):
            raise ValueError("All resident Van Rijn cell inputs must have identical lengths")
        if any(int(value.size) != nclass for value in classes[1:]):
            raise ValueError("All resident Van Rijn class inputs must have identical lengths")
        if any(tuple(value.shape) != (nclass, ncell) for value in matrices):
            raise ValueError(
                "Van Rijn roughness_z0 must have shape (nclass, ncell)"
            )
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            capacity = cp.empty((2, nclass, ncell), dtype=cp.float64)
            diagnostics = cp.empty((5, nclass, ncell), dtype=cp.float64)
            status = cp.empty(ncell, dtype=cp.int32)
            if ncell:
                self.kernels["river_vr1993_capacity"](
                    ((ncell + 255) // 256,), (256,),
                    (np.int64(ncell), np.int32(nclass), *cells, *classes,
                     *matrices, capacity, diagnostics, status),
                )
            if reject_invalid and ncell:
                self._raise_device_status(status, "Van Rijn 1993 capacity")
        return ResidentRiverCapacityResult(
            1, capacity, diagnostics, status, None,
        )

    def _evaluate_wilcock(self, law: int, *, reject_invalid: bool,
                          **fields) -> ResidentRiverCapacityResult:
        cell_names = WILCOCK_COMMON_CELL_FIELDS + WILCOCK_LAW_FIELDS[law]
        required = (cell_names + WILCOCK_CLASS_FIELDS +
                    WILCOCK_MATRIX_FIELDS + WILCOCK_INTEGER_FIELDS)
        self._exact_fields(fields, required, LAW_NAMES[law])
        cells = [self._array(name, fields[name], ndim=1)
                 for name in cell_names]
        classes = [self._array(name, fields[name], ndim=1)
                   for name in WILCOCK_CLASS_FIELDS]
        matrices = [self._array(name, fields[name], ndim=2)
                    for name in WILCOCK_MATRIX_FIELDS]
        integers = [self._array(name, fields[name], ndim=1, dtype=self.cp.int32)
                    for name in WILCOCK_INTEGER_FIELDS]
        ncell = int(cells[0].size)
        nclass = int(classes[0].size)
        if nclass <= 0:
            raise ValueError("Wilcock-family capacity requires at least one sediment class")
        if any(int(value.size) != ncell for value in cells[1:]):
            raise ValueError("All resident Wilcock cell inputs must have identical lengths")
        if any(int(value.size) != nclass for value in (*classes[1:], *integers)):
            raise ValueError("All resident Wilcock class inputs must have identical lengths")
        if any(tuple(value.shape) != (nclass, ncell) for value in matrices):
            raise ValueError(
                "Wilcock surface_mass must have shape (nclass, ncell)"
            )
        cp = self.cp
        # The unified device kernel has two formula-parameter slots.  Reusing
        # a resident input for an unused slot avoids inventing a host constant
        # or staging any physical data through CPU memory.
        if law == 5:
            parameter_1 = parameter_2 = cells[-1]
            common_cells = cells[:-1]
        else:
            parameter_1, parameter_2 = cells[-2:]
            common_cells = cells[:-2]
        with cp.cuda.Device(self.device_id):
            capacity = cp.empty((2, nclass, ncell), dtype=cp.float64)
            diagnostics = cp.empty((5, nclass, ncell), dtype=cp.float64)
            mixture = cp.empty((3, ncell), dtype=cp.float64)
            status = cp.empty(ncell, dtype=cp.int32)
            if ncell:
                self.kernels["river_wilcock_capacity"](
                    ((ncell + 255) // 256,), (256,),
                    (np.int64(ncell), np.int32(nclass), np.int32(law),
                     *common_cells, parameter_1, parameter_2,
                     *classes, *matrices, *integers,
                     capacity, diagnostics, mixture, status),
                )
            if reject_invalid and ncell:
                self._raise_device_status(status, f"{LAW_NAMES[law]} capacity")
        return ResidentRiverCapacityResult(
            law, capacity, diagnostics, status, mixture,
        )


# Descriptive alias retained for callers that prefer the longer research name.
ResidentRiverCapacityCUDA = RiverCapacityCUDA
