"""Resident CUDA companion for Q3/v69 surface identity semantics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from braids_core.sediment.scalar_flow_cpu_producer_v64 import load_mixture_profile
from braids_core.sediment.surface_identity_live_v69 import (
    BACKEND_GPU,
    SurfaceIdentityV69Error,
    load_surface_identity_execution_v69,
)


KERNELS = (
    "surface_resolve_v69", "surface_availability_v69",
    "surface_statistics_v69",
)
CUDA_OPTIONS = ("--std=c++14", "--fmad=false")


@dataclass(frozen=True)
class ResidentSurfaceResolutionV69:
    identity: object
    has_stock: object
    status: object


@dataclass(frozen=True)
class ResidentSurfaceStatisticsV69:
    values: object
    has_exposed_nonmud: object
    status: object

    @property
    def dm_m(self):
        return self.values[0]

    @property
    def dg_m(self):
        return self.values[1]

    @property
    def d10_m(self):
        return self.values[2]

    @property
    def d50_m(self):
        return self.values[3]

    @property
    def d90_m(self):
        return self.values[4]

    @property
    def dgsd(self):
        return self.values[5]

    @property
    def sand_fraction(self):
        return self.values[6]

    @property
    def mud_fraction(self):
        return self.values[7]


class SurfaceIdentityLiveV69CUDA:
    """Explicit ``implicit_fwave_midpoint`` resident surface owner."""

    def __init__(
        self, *, execution: Mapping[str, object],
        parent_execution: Mapping[str, object],
        mixture_profile: Mapping[str, object], device_id: int | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        try:
            import cupy as cp
        except Exception as error:  # pragma: no cover - environment dependent
            raise RuntimeError("CuPy is required for Q3/v69 CUDA") from error
        self.cp = cp
        self.device_id = (int(cp.cuda.Device().id) if device_id is None
                          else int(device_id))
        self.execution = load_surface_identity_execution_v69(
            execution, parent_execution=parent_execution,
            mixture_profile=mixture_profile)
        if self.execution["backend"] != BACKEND_GPU:
            raise SurfaceIdentityV69Error(
                "Q3/v69 CUDA requires the explicit implicit_fwave_midpoint GPU parent")
        self.mixture = load_mixture_profile(mixture_profile)
        rows = self.mixture["ordered_fractions"]
        self.nclass = len(rows)
        maxpoint = max(1, max(len(row["percentile_table"]) for row in rows))
        count = np.asarray(
            [len(row["percentile_table"]) for row in rows], dtype=np.int32)
        probability = np.zeros((self.nclass, maxpoint), dtype=np.float64)
        logdiameter = np.zeros_like(probability)
        for grain, row in enumerate(rows):
            for point, value in enumerate(row["percentile_table"]):
                probability[grain, point] = value["probability"]
                logdiameter[grain, point] = np.log(value["diameter_m"])
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=KERNELS)
            self.kernels = {name: module.get_function(name) for name in KERNELS}
            self.mean = cp.ascontiguousarray(cp.asarray(
                [row["mean_diameter_m"] for row in rows], dtype=cp.float64))
            self.median = cp.ascontiguousarray(cp.asarray(
                [row["median_diameter_m"] for row in rows], dtype=cp.float64))
            self.kind = cp.ascontiguousarray(cp.asarray(
                [row["sediment_type"] for row in rows], dtype=cp.int32))
            self.count = cp.ascontiguousarray(cp.asarray(count))
            self.probability = cp.ascontiguousarray(cp.asarray(probability))
            self.logdiameter = cp.ascontiguousarray(cp.asarray(logdiameter))
        self.maxpoint = maxpoint

    def _array(self, name: str, value: object, ndim: int,
               shape: tuple[int, ...] | None = None):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.ndim != ndim or not value.flags.c_contiguous or
                int(value.device.id) != self.device_id):
            raise TypeError(
                f"{name} must be resident contiguous CUDA FP64 with {ndim} dimensions")
        if shape is not None and tuple(value.shape) != shape:
            raise SurfaceIdentityV69Error(
                f"{name} shape {tuple(value.shape)} differs from {shape}")
        return value

    def _check_status(self, status: object, label: str) -> None:
        if bool(self.cp.any(status != 0).item()):
            raise SurfaceIdentityV69Error(f"Q3/v69 CUDA {label} rejected input")

    def resolve(
        self, *, mass: object, density: object,
        retained_identity: object | None,
    ) -> ResidentSurfaceResolutionV69:
        cp = self.cp
        mass = self._array("mass", mass, 2)
        if mass.shape[0] != self.nclass or mass.shape[1] < 1:
            raise SurfaceIdentityV69Error("Q3/v69 CUDA mass layout differs")
        ncell = int(mass.shape[1])
        density = self._array("density", density, 1, (self.nclass,))
        if retained_identity is None:
            retained = cp.zeros_like(mass)
            supplied = 0
        else:
            retained = self._array(
                "retained_identity", retained_identity, 2,
                (self.nclass, ncell))
            supplied = 1
        identity = cp.empty_like(mass)
        stock = cp.empty(ncell, dtype=cp.int32)
        status = cp.empty(ncell, dtype=cp.int32)
        if ncell:
            self.kernels["surface_resolve_v69"](
                ((ncell+255)//256,), (256,),
                (np.int64(ncell), np.int32(self.nclass), mass, density,
                 retained, np.int32(supplied), identity, stock, status))
        self._check_status(status, "surface resolution")
        return ResidentSurfaceResolutionV69(
            cp.ascontiguousarray(identity), cp.ascontiguousarray(stock),
            cp.ascontiguousarray(status))

    def after_exner(
        self, *, candidate_mass: object, density: object,
        parent_identity: object,
    ) -> ResidentSurfaceResolutionV69:
        return self.resolve(
            mass=candidate_mass, density=density,
            retained_identity=parent_identity)

    def availability(
        self, *, mass: object, identity: object, outgoing_demand: object,
    ) -> tuple[object, object]:
        cp = self.cp
        mass = self._array("mass", mass, 2)
        shape = tuple(mass.shape)
        if shape[0] != self.nclass or shape[1] < 1:
            raise SurfaceIdentityV69Error("Q3/v69 CUDA availability layout differs")
        identity = self._array("identity", identity, 2, shape)
        demand = self._array("outgoing_demand", outgoing_demand, 2, shape)
        count = int(mass.size)
        fill = cp.empty_like(mass)
        availability = cp.empty_like(mass)
        status = cp.empty(count, dtype=cp.int32)
        if count:
            self.kernels["surface_availability_v69"](
                ((count+255)//256,), (256,),
                (np.int64(count), mass, identity, demand, fill,
                 availability, status))
        self._check_status(status, "availability")
        return cp.ascontiguousarray(fill), cp.ascontiguousarray(availability)

    def statistics(self, identity: object) -> ResidentSurfaceStatisticsV69:
        cp = self.cp
        identity = self._array("identity", identity, 2)
        if identity.shape[0] != self.nclass or identity.shape[1] < 1:
            raise SurfaceIdentityV69Error("Q3/v69 CUDA statistics layout differs")
        ncell = int(identity.shape[1])
        values = cp.empty((8, ncell), dtype=cp.float64)
        exposed = cp.empty(ncell, dtype=cp.int32)
        status = cp.empty(ncell, dtype=cp.int32)
        if ncell:
            self.kernels["surface_statistics_v69"](
                ((ncell+255)//256,), (256,),
                (np.int64(ncell), np.int32(self.nclass),
                 np.int32(self.maxpoint), identity, self.mean, self.median,
                 self.kind, np.int32(self.mixture["max_mud_sediment_type"]),
                 np.int32(self.mixture["min_dxx_sediment_type"]), self.count,
                 self.probability, self.logdiameter, values, exposed, status))
        self._check_status(status, "statistics")
        return ResidentSurfaceStatisticsV69(
            cp.ascontiguousarray(values), cp.ascontiguousarray(exposed),
            cp.ascontiguousarray(status))


__all__ = [
    "ResidentSurfaceResolutionV69", "ResidentSurfaceStatisticsV69",
    "SurfaceIdentityLiveV69CUDA",
]
