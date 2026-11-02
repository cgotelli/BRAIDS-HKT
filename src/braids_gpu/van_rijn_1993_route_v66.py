"""Resident CUDA Van Rijn 1993 current-only Q3 route (v66).

This isolated producer preserves the unusual Delft3D4 FLOW ``TraFrm=-1``
ownership contract instead of pretending that equilibrium suspension is a
second transport vector:

* the formula returns a direct bed-mass vector and ``conc2d`` mass
  concentration, so neither receives another solid-density conversion;
* ``bedload`` folds the suspended-equivalent vector into bed transport before
  applying the surface fraction once;
* ``combine`` applies the fraction once to the bed vector and passes
  ``fraction * Sus * conc2d`` to the Q4 water-column owner;
* no external secondary-flow rotation is accepted for this direct-vector law.

The module is deliberately not product/catalogue admission.  It supports only
current-only, algebraic-turbulence 2DH production on BRAIDS' fixed 20-layer
integration column.  Wave physics and the Q4 suspended update remain explicit
later milestones and cannot silently fall back to CPU here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAME = "vr1993_route_v66"
ROUTE_BEDLOAD = "bedload"
ROUTE_COMBINE = "combine"
ROUTE_CODES = {ROUTE_BEDLOAD: 1, ROUTE_COMBINE: 2}

CELL_FIELDS = (
    "depth",
    "velocity",
    "observation_height",
    "water_density",
    "gravity",
    "kinematic_viscosity",
    "additional_shear_stress",
)
CLASS_FIELDS = (
    "d50",
    "d90",
    "solid_density",
    "molecular_schmidt",
    "aks_factor",
    "maximum_volume_concentration",
    "minimum_depth",
    "bed_factor",
    "suspended_factor",
)
POINT_FIELDS = (
    "settling_velocity",
    "roughness_z0",
    "dstar",
    "critical_shear_stress",
    "surface_fraction",
)


@dataclass(frozen=True)
class ResidentVanRijn1993RouteV66Result:
    """Device-resident formula and route outputs.

    ``raw_bed_mass`` is ``(2, class, cell)`` in kg/(m s), while
    ``raw_conc2d`` and ``equilibrium_concentration`` are ``(class, cell)`` in
    kg/m3.  ``bed_mass`` is the fully composed Q3 bed channel after Bed/Sus
    calibration and one surface-fraction application.  For ``combine`` the Q4
    owner consumes ``equilibrium_concentration``; for ``bedload`` that array is
    exactly zero because the suspended-equivalent transport is already folded
    into ``bed_mass``.
    """

    route: str
    raw_bed_mass: object
    raw_conc2d: object
    bed_mass: object
    equilibrium_concentration: object
    diagnostics: object
    status: object

    @property
    def shear_velocity(self):
        return self.diagnostics[0]

    @property
    def efficiency_factor(self):
        return self.diagnostics[1]

    @property
    def shear_ratio(self):
        return self.diagnostics[2]

    @property
    def reference_concentration(self):
        return self.diagnostics[3]

    @property
    def reference_height(self):
        return self.diagnostics[4]

    @property
    def current_shear(self):
        return self.diagnostics[5]


class ResidentVanRijn1993RouteV66:
    """Compile and evaluate the fail-closed resident v66 producer."""

    def __init__(self, route: str, source_path: str | Path | None = None):
        if route not in ROUTE_CODES:
            raise ValueError(
                "Van Rijn 1993 current-only route must be 'bedload' or 'combine'"
            )

        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.route = route
        self.route_code = np.int32(ROUTE_CODES[route])
        source = Path(source_path or Path(__file__).with_suffix(".cu")).read_text(
            encoding="utf-8"
        )
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source,
                options=CUDA_OPTIONS,
                name_expressions=(KERNEL_NAME,),
            )
            self.kernel = module.get_function(KERNEL_NAME)

    def _array(self, name: str, value: object, shape: tuple[int, ...]):
        cp = self.cp
        if (
            not isinstance(value, cp.ndarray)
            or value.dtype != cp.float64
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise TypeError(
                f"{name} must be a resident contiguous FP64 CuPy array with shape {shape}"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        return value

    def evaluate(
        self,
        *,
        reject_invalid: bool = True,
        **fields,
    ) -> ResidentVanRijn1993RouteV66Result:
        """Evaluate all class/cell points without a CPU numerical fallback."""

        required = CELL_FIELDS + CLASS_FIELDS + POINT_FIELDS
        missing = tuple(name for name in required if name not in fields)
        extra = tuple(name for name in fields if name not in required)
        if missing or extra:
            raise ValueError(
                "Van Rijn 1993 v66 fields differ; "
                f"missing={missing}, extra={extra}"
            )

        cp = self.cp
        if not isinstance(fields["depth"], cp.ndarray) or fields["depth"].ndim != 1:
            raise TypeError("depth must be a resident contiguous one-dimensional FP64 array")
        if not isinstance(fields["d50"], cp.ndarray) or fields["d50"].ndim != 1:
            raise TypeError("d50 must be a resident contiguous one-dimensional FP64 array")
        ncell = int(fields["depth"].size)
        nclass = int(fields["d50"].size)
        if ncell <= 0 or nclass <= 0:
            raise ValueError("Van Rijn 1993 v66 requires at least one cell and fraction")

        cells = []
        for name in CELL_FIELDS:
            shape = (2, ncell) if name == "velocity" else (ncell,)
            cells.append(self._array(name, fields[name], shape))
        classes = [
            self._array(name, fields[name], (nclass,)) for name in CLASS_FIELDS
        ]
        points = [
            self._array(name, fields[name], (nclass, ncell))
            for name in POINT_FIELDS
        ]

        with cp.cuda.Device(self.device_id):
            raw_bed = cp.empty((2, nclass, ncell), dtype=cp.float64)
            raw_conc2d = cp.empty((nclass, ncell), dtype=cp.float64)
            bed_mass = cp.empty((2, nclass, ncell), dtype=cp.float64)
            equilibrium = cp.empty((nclass, ncell), dtype=cp.float64)
            diagnostics = cp.empty((6, nclass, ncell), dtype=cp.float64)
            status = cp.empty((nclass, ncell), dtype=cp.int32)
            count = nclass * ncell
            self.kernel(
                ((count + 127) // 128,),
                (128,),
                (
                    np.int64(ncell),
                    np.int32(nclass),
                    self.route_code,
                    *cells,
                    *classes,
                    *points,
                    raw_bed,
                    raw_conc2d,
                    bed_mass,
                    equilibrium,
                    diagnostics,
                    status,
                ),
            )
            if reject_invalid and int(cp.count_nonzero(status).item()):
                raise ValueError("CUDA Van Rijn 1993 v66 rejected an invalid state")

        return ResidentVanRijn1993RouteV66Result(
            self.route,
            raw_bed,
            raw_conc2d,
            bed_mass,
            equilibrium,
            diagnostics,
            status,
        )


__all__ = [
    "CELL_FIELDS",
    "CLASS_FIELDS",
    "POINT_FIELDS",
    "ROUTE_BEDLOAD",
    "ROUTE_COMBINE",
    "ResidentVanRijn1993RouteV66",
    "ResidentVanRijn1993RouteV66Result",
]
