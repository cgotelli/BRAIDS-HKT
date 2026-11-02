"""Resident CUDA route composition for direct-vector Delft3D outputs.

The wrapper accepts stable method/route strings as control metadata and maps
them to dedicated output-kind/route enums.  It never accepts historical
numeric formula identifiers.  All physical fields and all result fields stay
resident on the selected CUDA device; there is no CPU fallback.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from braids_core.sediment.direct_vector_route_composer_v66 import (
    DirectVectorOutputKind,
    METHOD_OUTPUT_KINDS,
    ROUTE_TRA_BEDLOAD,
    ROUTE_TRA_COMBINE,
    resolve_direct_vector_output_kind,
)


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)

ROUTE_CODES = MappingProxyType({
    ROUTE_TRA_BEDLOAD: 0,
    ROUTE_TRA_COMBINE: 1,
})

STATUS_OK = 0
STATUS_INVALID_METHOD = 1
STATUS_INVALID_ROUTE = 2
STATUS_NONFINITE = 3
STATUS_INVALID_ARGUMENT = 4
STATUS_SOURCE_KIND_MISMATCH = 5
STATUS_DRY_SOURCE_NONZERO = 6
STATUS_ARITHMETIC = 7
STATUS_NAMES = MappingProxyType({
    STATUS_OK: "ok",
    STATUS_INVALID_METHOD: "invalid_method",
    STATUS_INVALID_ROUTE: "invalid_route",
    STATUS_NONFINITE: "nonfinite",
    STATUS_INVALID_ARGUMENT: "invalid_argument",
    STATUS_SOURCE_KIND_MISMATCH: "source_kind_mismatch",
    STATUS_DRY_SOURCE_NONZERO: "dry_source_nonzero",
    STATUS_ARITHMETIC: "arithmetic_overflow",
})


@dataclass(frozen=True)
class ResidentDirectVectorRouteResultV66:
    """Resident route outputs plus immutable host control identities."""

    methods: tuple[str, ...]
    routes: tuple[str, ...]
    output_kind: object
    values: object
    status: object

    @property
    def bed_mass(self):
        return self.values[0:2]

    @property
    def q4_equilibrium_mass_concentration(self):
        return self.values[2]


class DirectVectorRouteComposerV66CUDA:
    """Compile and execute the generic route composer on one CUDA device."""

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
                name_expressions=("compose_direct_vector_route_v66",),
            )
            self.kernel = self.module.get_function(
                "compose_direct_vector_route_v66"
            )

    @staticmethod
    def _exact_string_sequence(
        name: str,
        value: str | Sequence[str],
        count: int,
    ) -> tuple[str, ...]:
        if type(value) is str:
            return (value,)*count
        if (not isinstance(value, Sequence) or
                isinstance(value, (str, bytes, bytearray))):
            raise TypeError(
                f"{name} must be an exact string or ordered string sequence"
            )
        resolved = tuple(value)
        if len(resolved) != count or any(type(item) is not str for item in resolved):
            raise TypeError(
                f"{name} must contain {count} exact string value(s)"
            )
        return resolved

    def _vector(self, name: str, value, count: int | None = None):
        cp = self.cp
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
        if count is not None and int(value.size) != count:
            raise ValueError(f"{name} length differs from the route state")
        return value

    def _pair(self, name: str, value, count: int):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.ndim != 2 or value.shape != (2, count) or
                not value.flags.c_contiguous):
            raise TypeError(
                f"{name} must be a resident contiguous (2, {count}) FP64 "
                "CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        return value

    def compose(
        self,
        *,
        method: str | Sequence[str],
        route: str | Sequence[str],
        depth,
        velocity,
        raw_bed_mass,
        raw_suspended_mass,
        raw_equilibrium_mass_concentration,
        bed_factor,
        suspended_factor,
        active_fraction,
        speed_epsilon,
        reject_invalid: bool = True,
    ) -> ResidentDirectVectorRouteResultV66:
        """Compose resident physical arrays without a host/CPU fallback."""
        depth = self._vector("depth", depth)
        count = int(depth.size)
        velocity = self._pair("velocity", velocity, count)
        raw_bed_mass = self._pair("raw_bed_mass", raw_bed_mass, count)
        raw_suspended_mass = self._pair(
            "raw_suspended_mass", raw_suspended_mass, count,
        )
        raw_equilibrium_mass_concentration = self._vector(
            "raw_equilibrium_mass_concentration",
            raw_equilibrium_mass_concentration, count,
        )
        bed_factor = self._vector("bed_factor", bed_factor, count)
        suspended_factor = self._vector(
            "suspended_factor", suspended_factor, count,
        )
        active_fraction = self._vector(
            "active_fraction", active_fraction, count,
        )
        speed_epsilon = self._vector(
            "speed_epsilon", speed_epsilon, count,
        )

        methods = self._exact_string_sequence("method", method, count)
        routes = self._exact_string_sequence("route", route, count)
        # Resolve the method string through the shared source-of-truth mapping.
        # No legacy numeric formula selector is admitted by this API.
        output_codes = tuple(
            int(resolve_direct_vector_output_kind(item)) for item in methods
        )
        try:
            route_codes = tuple(ROUTE_CODES[item] for item in routes)
        except KeyError as error:
            raise ValueError(
                f"unsupported direct-vector route: {error.args[0]!r}"
            ) from error

        cp = self.cp
        with cp.cuda.Device(self.device_id):
            output_kind = cp.asarray(output_codes, dtype=cp.int32)
            route_device = cp.asarray(route_codes, dtype=cp.int32)
            values = cp.empty((3, count), dtype=cp.float64)
            status = cp.empty(count, dtype=cp.int32)
            if count:
                self.kernel(
                    ((count + 255)//256,), (256,),
                    (
                        np.int64(count), output_kind, route_device, depth,
                        velocity[0], velocity[1],
                        raw_bed_mass[0], raw_bed_mass[1],
                        raw_suspended_mass[0], raw_suspended_mass[1],
                        raw_equilibrium_mass_concentration,
                        bed_factor, suspended_factor, active_fraction,
                        speed_epsilon, values, status,
                    ),
                )
            if reject_invalid and count:
                bad = status != STATUS_OK
                bad_count = int(cp.count_nonzero(bad).item())
                if bad_count:
                    unique, counts = cp.unique(status[bad], return_counts=True)
                    detail = ", ".join(
                        f"{STATUS_NAMES.get(int(key), f'status_{int(key)}')}="
                        f"{int(value)}"
                        for key, value in zip(
                            cp.asnumpy(unique), cp.asnumpy(counts), strict=True,
                        )
                    )
                    raise ValueError(
                        "CUDA direct-vector route composer rejected "
                        f"{bad_count} state(s): {detail}"
                    )
        return ResidentDirectVectorRouteResultV66(
            methods, routes, output_kind, values, status,
        )


__all__ = [
    "CUDA_OPTIONS",
    "DirectVectorRouteComposerV66CUDA",
    "METHOD_OUTPUT_KINDS",
    "ROUTE_CODES",
    "ResidentDirectVectorRouteResultV66",
    "STATUS_ARITHMETIC",
    "STATUS_DRY_SOURCE_NONZERO",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_INVALID_METHOD",
    "STATUS_INVALID_ROUTE",
    "STATUS_NAMES",
    "STATUS_NONFINITE",
    "STATUS_OK",
    "STATUS_SOURCE_KIND_MISMATCH",
]
