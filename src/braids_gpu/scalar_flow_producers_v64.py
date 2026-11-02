"""Resident CUDA producer closure for six bounded scalar FLOW laws.

This module is deliberately one promotion step short of the live solver.  It
connects the accepted ``implicit_fwave_midpoint`` resident state to FLOW-like
mixture, roughness, effective-velocity, shear and v63 point-capacity inputs,
but it does not own reconstructed faces, boundaries or Exner.  Runtime
physical arrays never leave their CUDA device and no CPU constitutive fallback
exists.

The non-spatial mixture scope is the exact classic ``erosed`` call site:
``compdiam`` and ``compsandfrac`` both receive ``sedd50`` as their class mean
diameter.  Broad within-class CDFs are supported; degenerate/Dirac CDFs are
rejected pending a separately qualified implementation of FLOW's hard-coded
``stage=5`` branch.  The spatial-D50 scope is explicit: one negative class-D50
sentinel plus one resident positive D50 field, with FLOW's intentional
``DgSD=0`` result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .delft_scalar_capacity_v63 import DelftScalarCapacityV63CUDA
from .hydraulic_roughness import HydraulicRoughnessCUDA
from .sediment_skin_shear import SedimentSkinShearCUDA


CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
KERNEL_NAMES = ("scalar_flow_mixture_v64", "scalar_flow_hydraulic_v64")
MAX_CLASSES = 32
SELECTORS = (7, 13, 14, 16, 17, 18)
SHEAR_METHODS = ("shear.total_roughness", "shear.soulsby_clarke_skin")
MIXTURE_ROWS = MappingProxyType({
    "dm": 0, "dg": 1, "dgsd": 2, "d90": 3,
    "sand_fraction": 4, "mud_fraction": 5,
})
HYDRAULIC_ROWS = MappingProxyType({
    "raw_depth": 0, "depth": 1,
    "characteristic_u": 2, "characteristic_v": 3,
    "characteristic_speed": 4, "observation_height": 5,
    "effective_u": 6, "effective_v": 7, "rp_ustar": 8,
    "total_current_shear": 9, "total_current_ust2_pre_secondary": 10,
    "total_current_ust2_after_secondary": 11,
    "secondary_bed_direction_x": 12, "secondary_bed_direction_y": 13,
    "sediment_chezy": 14,
})
FORMULA_INPUTS = MappingProxyType({
    7: ("acal", "reference_height", "settling_velocity", "mud_power"),
    13: ("acal",),
    14: ("hiding_factor", "acal", "critical_shields", "power_m",
         "power_p", "power_q"),
    16: ("acal",),
    17: ("theta_c0", "alpha0"),
    18: ("theta_c0", "alpha0"),
})


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _float_rows(values: np.ndarray) -> list[str]:
    return [float(value).hex() for value in values.reshape(-1)]


@dataclass(frozen=True)
class ResidentMixtureV64:
    fraction: object
    resolved_d50: object
    values: object
    status: object

    @property
    def dm(self):
        return self.values[MIXTURE_ROWS["dm"]]

    @property
    def dg(self):
        return self.values[MIXTURE_ROWS["dg"]]

    @property
    def dgsd(self):
        return self.values[MIXTURE_ROWS["dgsd"]]

    @property
    def d90(self):
        return self.values[MIXTURE_ROWS["d90"]]

    @property
    def sand_fraction(self):
        return self.values[MIXTURE_ROWS["sand_fraction"]]

    @property
    def mud_fraction(self):
        return self.values[MIXTURE_ROWS["mud_fraction"]]


@dataclass(frozen=True)
class ResidentHydraulicV64:
    values: object
    hydraulic_chezy: object
    z0: object
    drag: object
    status: object

    def _row(self, name: str):
        return self.values[HYDRAULIC_ROWS[name]]

    @property
    def depth(self):
        return self._row("depth")

    @property
    def raw_depth(self):
        return self._row("raw_depth")

    @property
    def chezy(self):
        return self._row("sediment_chezy")

    @property
    def characteristic_u(self):
        return self._row("characteristic_u")

    @property
    def characteristic_v(self):
        return self._row("characteristic_v")

    @property
    def characteristic_speed(self):
        return self._row("characteristic_speed")

    @property
    def observation_height(self):
        return self._row("observation_height")

    @property
    def effective_u(self):
        return self._row("effective_u")

    @property
    def effective_v(self):
        return self._row("effective_v")

    @property
    def rp_ustar(self):
        return self._row("rp_ustar")

    @property
    def total_current_shear(self):
        return self._row("total_current_shear")

    @property
    def total_current_ust2_pre_secondary(self):
        return self._row("total_current_ust2_pre_secondary")

    @property
    def total_current_ust2_after_secondary(self):
        return self._row("total_current_ust2_after_secondary")

    @property
    def secondary_bed_direction(self):
        return self.values[12:14]


@dataclass(frozen=True)
class ResidentScalarFlowInputsV64:
    mixture: ResidentMixtureV64
    hydraulic: ResidentHydraulicV64
    kinematic_viscosity: object
    solid_density: object
    rp_taub: object
    slope_taurat_zero: object
    status: object
    active: object
    water_density: float
    gravity: float
    sediment_shear_method_id: str
    secondary_enabled: bool
    distribution_fingerprint: str
    slope_taurat_semantics: str = (
        "positive_scalar_eqtran_unassigned_erosed_zero_pending_locked_probe")


@dataclass(frozen=True)
class ResidentScalarFlowCapacityV64:
    selector: int
    raw_values: object
    bed_mass_pre_secondary: object
    bed_mass: object
    suspended_mass: object
    status: object
    active_fraction: object
    slope_total_current_ust2: object
    slope_taurat_zero: object
    active_fraction_applied: bool = False

    @property
    def absolute_shields(self):
        return self.raw_values[4]

    @property
    def reference_shields(self):
        return self.raw_values[5]

    @property
    def stress_ratio_diagnostic(self):
        return self.raw_values[6]


class ScalarFlowProducersV64CUDA:
    """Produce and connect the bounded scalar-law inputs on one CUDA device."""

    def __init__(self, *, class_d50: Sequence[float], is_mud: Sequence[bool],
                 distribution_cdf: Sequence[Sequence[float]] | None = None,
                 distribution_diameter: Sequence[Sequence[float]] | None = None,
                 spatial_d50: bool = False,
                 log_sigma: Sequence[float] | None = None,
                 source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.spatial_d50 = bool(spatial_d50)
        d50 = np.asarray(class_d50, dtype=np.float64)
        mud = np.asarray(is_mud)
        if d50.ndim != 1 or d50.size == 0 or d50.size > MAX_CLASSES:
            raise ValueError("Scalar v64 requires 1..32 class D50 values")
        if mud.shape != d50.shape or mud.dtype.kind not in "biu":
            raise ValueError("Scalar v64 mud mask must match the class inventory")
        mud = mud.astype(np.int32)
        if np.any((mud != 0) & (mud != 1)):
            raise ValueError("Scalar v64 mud mask must contain only zero or one")
        self.nclass = int(d50.size)

        if self.spatial_d50:
            if self.nclass != 1 or not np.isfinite(d50[0]) or d50[0] >= 0.0:
                raise ValueError(
                    "Spatial D50 requires one negative FLOW class-D50 sentinel")
            sigma = np.asarray(log_sigma, dtype=np.float64)
            if sigma.shape != (1,) or not np.isfinite(sigma[0]) or sigma[0] < 0.0:
                raise ValueError("Spatial D50 requires one finite nonnegative log sigma")
            if distribution_cdf is not None or distribution_diameter is not None:
                raise ValueError("Spatial D50 does not accept static class CDFs")
            counts = np.zeros(1, dtype=np.int32)
            max_points = 2
            cdf = np.zeros((1, max_points), dtype=np.float64)
            log_diameter = np.zeros_like(cdf)
        else:
            if np.any(~np.isfinite(d50)) or np.any(d50 <= 0.0):
                raise ValueError("Non-spatial class D50 values must be finite positive")
            if log_sigma is not None:
                raise ValueError("Non-spatial broad CDFs, not log_sigma, own distributions")
            sigma = np.zeros(self.nclass, dtype=np.float64)
            if distribution_cdf is None or distribution_diameter is None or \
                    len(distribution_cdf) != self.nclass or \
                    len(distribution_diameter) != self.nclass:
                raise ValueError("Every non-spatial class requires a distribution entry")
            parsed: list[tuple[np.ndarray, np.ndarray]] = []
            counts = np.empty(self.nclass, dtype=np.int32)
            for grain, (cdf_values, diameter_values) in enumerate(zip(
                    distribution_cdf, distribution_diameter)):
                cdf_row = np.asarray(cdf_values, dtype=np.float64)
                diameter_row = np.asarray(diameter_values, dtype=np.float64)
                if mud[grain]:
                    if cdf_row.size or diameter_row.size:
                        raise ValueError("Mud classes must have zero distribution points")
                    parsed.append((cdf_row, diameter_row))
                    counts[grain] = 0
                    continue
                is_dirac = cdf_row.size == 1
                broad_invalid = (
                    cdf_row.size >= 2 and (
                        cdf_row[0] != 0.0 or cdf_row[-1] != 1.0 or
                        np.any(np.diff(cdf_row) <= 0.0) or
                        np.any(np.diff(diameter_row) <= 0.0)))
                if (cdf_row.ndim != 1 or diameter_row.shape != cdf_row.shape or
                        cdf_row.size < 1 or cdf_row.size > 101 or
                        np.any(~np.isfinite(cdf_row)) or
                        np.any(~np.isfinite(diameter_row)) or
                        np.any(diameter_row <= 0.0) or
                        np.any((cdf_row < 0.0) | (cdf_row > 1.0)) or
                        (not is_dirac and broad_invalid)):
                    raise ValueError(
                        "Non-mud CDFs require either one exact/Dirac diameter "
                        "or 2..101 strictly increasing broad points spanning "
                        "exactly [0,1]")
                parsed.append((cdf_row, diameter_row))
                counts[grain] = int(cdf_row.size)
            max_points = max(2, int(max(counts, default=0)))
            cdf = np.zeros((self.nclass, max_points), dtype=np.float64)
            log_diameter = np.zeros_like(cdf)
            for grain, (cdf_row, diameter_row) in enumerate(parsed):
                count = int(counts[grain])
                if count:
                    cdf[grain, :count] = cdf_row
                    log_diameter[grain, :count] = np.log(diameter_row)

        payload = {
            "schema": 1,
            "mode": "spatial_d50_sentinel" if self.spatial_d50 else
                    "classic_erosed_seddm_equals_sedd50",
            "class_d50_hex": _float_rows(d50),
            "is_mud": mud.tolist(),
            "log_sigma_hex": _float_rows(sigma),
            "distribution_count": counts.tolist(),
            "distribution_cdf_hex": _float_rows(cdf),
            "distribution_log_diameter_hex": _float_rows(log_diameter),
            "dirac_distribution_supported": True,
        }
        self.distribution_document = MappingProxyType({
            **payload,
            "fingerprint": hashlib.sha256(_canonical(payload)).hexdigest(),
        })
        self.distribution_fingerprint = self.distribution_document["fingerprint"]
        self.max_points = max_points

        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            self.module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.kernels = {
                name: self.module.get_function(name) for name in KERNEL_NAMES}
            self.class_d50 = cp.ascontiguousarray(cp.asarray(d50))
            self.is_mud = cp.ascontiguousarray(cp.asarray(mud))
            self.log_sigma = cp.ascontiguousarray(cp.asarray(sigma))
            self.distribution_count = cp.ascontiguousarray(cp.asarray(counts))
            self.distribution_cdf = cp.ascontiguousarray(cp.asarray(cdf))
            self.distribution_log_diameter = cp.ascontiguousarray(
                cp.asarray(log_diameter))
        self.roughness = HydraulicRoughnessCUDA()
        self.point = DelftScalarCapacityV63CUDA()
        if (self.roughness.cp.cuda.Device().id != self.device_id or
                self.point.device_id != self.device_id):  # pragma: no cover
            raise RuntimeError("Scalar v64 dependencies selected different CUDA devices")
        self._skin = None

    def _array(self, name: str, value, *, shape: tuple[int, ...], dtype=None):
        cp = self.cp
        expected = cp.float64 if dtype is None else dtype
        if (not isinstance(value, cp.ndarray) or value.dtype != expected or
                value.shape != shape or not value.flags.c_contiguous):
            raise TypeError(
                f"{name} must be a resident contiguous {expected} CuPy array "
                f"with shape {shape}")
        if int(value.device.id) != self.device_id:
            raise ValueError(f"{name} is not resident on CUDA device {self.device_id}")
        return value

    @staticmethod
    def _positive_scalar(name: str, value: float) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be finite positive")
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite positive")
        return result

    def _skin_operator(self):
        if self._skin is None:
            self._skin = SedimentSkinShearCUDA()
            if self._skin.device_id != self.device_id:  # pragma: no cover
                raise RuntimeError("Skin-shear dependency selected a different device")
        return self._skin

    def produce(self, *, q, active, surface_mass, bulk_density,
                solid_density, roughness_parameter, kinematic_viscosity,
                roughness_mode: str | int, water_density: float = 1000.0,
                gravity: float = 9.81, minimum_work_depth: float = 0.0,
                waqua_white_colebrook: bool = False,
                sediment_shear_method_id: str = "shear.total_roughness",
                skin_fields: Mapping[str, object] | None = None,
                spatial_d50_field=None, spiral_intensity=None,
                retained_surface_fraction=None,
                secondary_enabled: bool = False, espir: float = 0.0,
                direction_epsilon: float = 1.0e-12,
                reject_invalid: bool = True) -> ResidentScalarFlowInputsV64:
        """Produce all shared v63 inputs from resident midpoint state."""
        cp = self.cp
        if not isinstance(q, cp.ndarray) or q.dtype != cp.float64 or \
                q.ndim != 2 or q.shape[0] != 3 or not q.flags.c_contiguous:
            raise TypeError("q must be a resident contiguous (3,ncell) FP64 array")
        if int(q.device.id) != self.device_id:
            raise ValueError("q is resident on a different CUDA device")
        ncell = int(q.shape[1])
        active = self._array("active", active, shape=(ncell,), dtype=cp.int32)
        surface_mass = self._array(
            "surface_mass", surface_mass, shape=(self.nclass, ncell))
        if retained_surface_fraction is None:
            retained_fraction = cp.zeros(
                (self.nclass, ncell), dtype=cp.float64)
            retain_depleted = 0
        else:
            retained_fraction = self._array(
                "retained_surface_fraction", retained_surface_fraction,
                shape=(self.nclass, ncell))
            retain_depleted = 1
        bulk_density = self._array(
            "bulk_density", bulk_density, shape=(self.nclass,))
        solid_density = self._array(
            "solid_density", solid_density, shape=(self.nclass,))
        roughness_parameter = self._array(
            "roughness_parameter", roughness_parameter, shape=(ncell,))
        kinematic_viscosity = self._array(
            "kinematic_viscosity", kinematic_viscosity, shape=(ncell,))
        rho = self._positive_scalar("water_density", water_density)
        grav = self._positive_scalar("gravity", gravity)
        epsilon = self._positive_scalar("direction_epsilon", direction_epsilon)
        if (not np.isfinite(float(minimum_work_depth)) or
                float(minimum_work_depth) < 0.0):
            raise ValueError("minimum_work_depth must be finite nonnegative")
        if not np.isfinite(float(espir)):
            raise ValueError("espir must be finite")
        if not isinstance(secondary_enabled, (bool, np.bool_)):
            raise ValueError("secondary_enabled must be Boolean")
        if sediment_shear_method_id not in SHEAR_METHODS:
            raise ValueError("Scalar v64 sediment shear method is unsupported")

        if self.spatial_d50:
            spatial = self._array(
                "spatial_d50_field", spatial_d50_field, shape=(ncell,))
        else:
            if spatial_d50_field is not None:
                raise ValueError("Non-spatial mixture rejects a spatial D50 field")
            spatial = cp.zeros(ncell, dtype=cp.float64)
        if secondary_enabled:
            spiral = self._array(
                "spiral_intensity", spiral_intensity, shape=(ncell,))
        else:
            if spiral_intensity is not None:
                raise ValueError("Secondary-off producer rejects inert spiral intensity")
            spiral = cp.zeros(ncell, dtype=cp.float64)

        with cp.cuda.Device(self.device_id):
            fraction = cp.empty((self.nclass, ncell), dtype=cp.float64)
            resolved_d50 = cp.empty_like(fraction)
            mixture_values = cp.empty((6, ncell), dtype=cp.float64)
            mixture_status = cp.empty(ncell, dtype=cp.int32)
            if ncell:
                self.kernels["scalar_flow_mixture_v64"](
                    ((ncell+255)//256,), (256,),
                    (np.int64(ncell), np.int32(self.nclass),
                     np.int32(self.max_points), np.int32(self.spatial_d50),
                     surface_mass, bulk_density, retained_fraction,
                     np.int32(retain_depleted), self.class_d50,
                     self.log_sigma, self.is_mud, self.distribution_count,
                     self.distribution_cdf, self.distribution_log_diameter,
                     spatial, fraction, resolved_d50, mixture_values,
                     mixture_status))

            # Accepted hydraulic roughness remains at its own raw/work-depth
            # policy.  The CUDA producer independently recomputes erosed's
            # sediment Chezy from this z0 and max(0.01, raw depth).
            safe_depth = cp.ascontiguousarray(cp.where(active == 1, q[0], 0.0))
            hydraulic_chezy, z0, drag, roughness_status = self.roughness.state(
                safe_depth, roughness_parameter, mode=roughness_mode,
                gravity=grav, minimum_work_depth=float(minimum_work_depth),
                waqua_white_colebrook=bool(waqua_white_colebrook))
            hydraulic_values = cp.empty((15, ncell), dtype=cp.float64)
            hydraulic_status = cp.empty(ncell, dtype=cp.int32)
            if ncell:
                self.kernels["scalar_flow_hydraulic_v64"](
                    ((ncell+255)//256,), (256,),
                    (np.int64(ncell), q, active, hydraulic_chezy, z0, drag, spiral,
                     np.int32(secondary_enabled), np.float64(espir),
                     np.float64(rho), np.float64(grav), np.float64(epsilon),
                     hydraulic_values, hydraulic_status))

            rp_taub = hydraulic_values[HYDRAULIC_ROWS["total_current_shear"]]
            skin_status = cp.zeros(ncell, dtype=cp.int32)
            if sediment_shear_method_id == "shear.soulsby_clarke_skin":
                if not isinstance(skin_fields, Mapping):
                    raise ValueError("Skin shear requires its resident control fields")
                names = ("ks_sand", "ks_silt", "critical_mud_1",
                         "critical_mud_2", "fluff_fraction")
                missing = tuple(name for name in names if name not in skin_fields)
                extra = tuple(name for name in skin_fields if name not in names)
                if missing or extra:
                    raise ValueError(
                        f"Skin fields differ; missing={missing}, extra={extra}")
                controls = {name: self._array(
                    f"skin_fields[{name!r}]", skin_fields[name], shape=(ncell,))
                    for name in names}
                skin_depth = cp.ascontiguousarray(cp.where(
                    active == 1, q[0], max(float(minimum_work_depth), 1.0e-12)))
                skin = self._skin_operator().evaluate(
                    reject_invalid=False, depth=skin_depth,
                    speed=hydraulic_values[HYDRAULIC_ROWS["characteristic_speed"]],
                    kinematic_viscosity=kinematic_viscosity,
                    water_density=cp.full(ncell, rho, dtype=cp.float64),
                    mud_factor=mixture_values[MIXTURE_ROWS["mud_fraction"]],
                    **controls)
                rp_taub = cp.ascontiguousarray(cp.where(
                    active == 1, skin.rp_taub, 0.0))
                skin_status = cp.ascontiguousarray(cp.where(
                    active == 1, skin.status, 0))
            elif skin_fields is not None:
                raise ValueError("Total-roughness shear rejects inert skin fields")

            # FLOW getfrac can retain composition outside this bounded input,
            # but Q5 has not yet authenticated that lifecycle.  Never invent
            # a replacement fraction for a genuinely empty active surface.
            surface_volume = cp.sum(surface_mass/bulk_density[:, None], axis=0)
            depleted_status = cp.where(
                (active == 1) & (surface_volume <= 0.0) &
                (np.int32(retain_depleted) == 0), cp.int32(2), cp.int32(0))
            status = cp.maximum(depleted_status, cp.maximum(
                mixture_status, cp.maximum(
                roughness_status, cp.maximum(hydraulic_status, skin_status))))
            slope_taurat_zero = cp.zeros(
                (self.nclass, ncell), dtype=cp.float64)
            if reject_invalid and ncell:
                count = int(cp.count_nonzero(status).item())
                if count:
                    raise ValueError(
                        f"Scalar v64 CUDA producers rejected {count} cell(s)")

        return ResidentScalarFlowInputsV64(
            ResidentMixtureV64(fraction, resolved_d50, mixture_values,
                               mixture_status),
            ResidentHydraulicV64(hydraulic_values, hydraulic_chezy, z0, drag,
                                 hydraulic_status),
            kinematic_viscosity, solid_density, rp_taub, slope_taurat_zero,
            status, active, rho, grav, sediment_shear_method_id,
            bool(secondary_enabled), self.distribution_fingerprint)

    def _point_matrix(self, name: str, value, ncell: int):
        cp = self.cp
        if not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or \
                not value.flags.c_contiguous or value.shape not in (
                    (self.nclass,), (self.nclass, ncell)):
            raise TypeError(
                f"{name} must be a resident contiguous FP64 class vector or "
                "(nclass,ncell) matrix")
        if int(value.device.id) != self.device_id:
            raise ValueError(f"{name} is resident on a different CUDA device")
        if value.ndim == 1:
            value = cp.broadcast_to(value[:, None], (self.nclass, ncell))
        return cp.ascontiguousarray(value).reshape(-1)

    def _class_points(self, value, ncell: int):
        return self.cp.ascontiguousarray(self.cp.broadcast_to(
            value[:, None], (self.nclass, ncell))).reshape(-1)

    def _cell_points(self, value, ncell: int):
        return self.cp.ascontiguousarray(self.cp.broadcast_to(
            value[None, :], (self.nclass, ncell))).reshape(-1)

    def evaluate(self, selector: int, produced: ResidentScalarFlowInputsV64,
                 *, formula_inputs: Mapping[str, object],
                 reject_invalid: bool = True) -> ResidentScalarFlowCapacityV64:
        """Connect one selector to v63 without claiming the downstream fraction owner.

        FLOW applies composition at different stages for bedload and
        advection-diffusion classes.  Until the live channel/face owner is
        bound, both raw channels and the active fraction therefore remain
        separate resident arrays.
        """
        if isinstance(selector, (bool, np.bool_)) or selector not in SELECTORS:
            raise ValueError("Scalar v64 selector must be 7/13/14/16/17/18")
        if (not isinstance(produced, ResidentScalarFlowInputsV64) or
                produced.distribution_fingerprint != self.distribution_fingerprint):
            raise ValueError("Scalar v64 produced state has a different identity")
        required = FORMULA_INPUTS[selector]
        if not isinstance(formula_inputs, Mapping):
            raise TypeError("formula_inputs must be a mapping of resident arrays")
        missing = tuple(name for name in required if name not in formula_inputs)
        extra = tuple(name for name in formula_inputs if name not in required)
        if missing or extra:
            raise ValueError(
                f"TraFrm {selector} formula inputs differ; missing={missing}, "
                f"extra={extra}")
        cp = self.cp
        ncell = int(produced.active.size)
        point_extra = {name: self._point_matrix(
            name, formula_inputs[name], ncell) for name in required}
        h = produced.hydraulic
        m = produced.mixture
        depth = self._cell_points(h.depth, ncell)
        u = self._cell_points(h.effective_u, ncell)
        v = self._cell_points(h.effective_v, ncell)
        d50 = cp.ascontiguousarray(m.resolved_d50).reshape(-1)
        d90 = self._cell_points(m.d90, ncell)
        solid_density = self._class_points(produced.solid_density, ncell)
        water_density = cp.full(self.nclass*ncell, produced.water_density,
                                dtype=cp.float64)
        gravity = cp.full(self.nclass*ncell, produced.gravity, dtype=cp.float64)
        chezy = self._cell_points(h.chezy, ncell)
        fields = {
            7: dict(depth=depth, u=u, v=v, d50=d50, d90=d90,
                    solid_density=solid_density, water_density=water_density,
                    gravity=gravity,
                    kinematic_viscosity=self._cell_points(
                        produced.kinematic_viscosity, ncell),
                    acal=point_extra.get("acal"),
                    reference_height=point_extra.get("reference_height"),
                    settling_velocity=point_extra.get("settling_velocity"),
                    mud_power=point_extra.get("mud_power"),
                    mud_fraction=self._cell_points(m.mud_fraction, ncell)),
            13: dict(depth=depth, u=u, v=v, d50=d50, d90=d90,
                     solid_density=solid_density, water_density=water_density,
                     gravity=gravity,
                     vicmol=self._cell_points(
                         produced.kinematic_viscosity, ncell),
                     chezy=chezy,
                     current_shear_velocity=self._cell_points(h.rp_ustar, ncell),
                     acal=point_extra.get("acal")),
            14: dict(u=u, v=v, d50=d50, solid_density=solid_density,
                     water_density=water_density, gravity=gravity, chezy=chezy,
                     **point_extra),
            16: dict(u=u, v=v, d50=d50, solid_density=solid_density,
                     water_density=water_density, gravity=gravity, chezy=chezy,
                     bed_shear=self._cell_points(produced.rp_taub, ncell),
                     dg=self._cell_points(m.dg, ncell),
                     sand_fraction=self._cell_points(m.sand_fraction, ncell),
                     acal=point_extra.get("acal")),
            17: dict(u=u, v=v, d50=d50, solid_density=solid_density,
                     water_density=water_density, gravity=gravity, chezy=chezy,
                     bed_shear=self._cell_points(produced.rp_taub, ncell),
                     dg=self._cell_points(m.dg, ncell),
                     dgsd=self._cell_points(m.dgsd, ncell),
                     sand_fraction=self._cell_points(m.sand_fraction, ncell),
                     **point_extra),
            18: dict(u=u, v=v, d50=d50, solid_density=solid_density,
                     water_density=water_density, gravity=gravity, chezy=chezy,
                     bed_shear=self._cell_points(produced.rp_taub, ncell),
                     dg=self._cell_points(m.dg, ncell),
                     dgsd=self._cell_points(m.dgsd, ncell),
                     sand_fraction=self._cell_points(m.sand_fraction, ncell),
                     **point_extra),
        }[selector]
        result = self.point.evaluate(selector, reject_invalid=False, **fields)
        raw = result.values.reshape(7, self.nclass, ncell)
        point_status = result.status.reshape(self.nclass, ncell)
        active = produced.active == 1
        status = cp.maximum(point_status, produced.status[None, :])
        status = cp.where(active[None, :], status, 0).astype(cp.int32)
        raw = cp.ascontiguousarray(cp.where(active[None, None, :], raw, 0.0))

        pre_secondary = raw[0:2]
        magnitude = cp.sqrt(cp.sum(pre_secondary*pre_secondary, axis=0))
        direction = h.secondary_bed_direction
        bed = cp.ascontiguousarray(direction[:, None, :]*magnitude[None, :, :])
        suspended = raw[2:4]
        valid = status == 0
        bed = cp.where(valid[None, :, :], bed, 0.0)
        suspended = cp.where(valid[None, :, :], suspended, 0.0)
        pre_secondary = cp.where(valid[None, :, :], pre_secondary, 0.0)
        if reject_invalid and ncell:
            count = int(cp.count_nonzero(status).item())
            if count:
                raise ValueError(
                    f"CUDA TraFrm {selector} producer seam rejected "
                    f"{count} class-point(s)")
        return ResidentScalarFlowCapacityV64(
            selector, raw, pre_secondary, bed, suspended, status, m.fraction,
            h.total_current_ust2_after_secondary,
            produced.slope_taurat_zero)


__all__ = [
    "CUDA_OPTIONS", "KERNEL_NAMES", "MAX_CLASSES", "SELECTORS",
    "SHEAR_METHODS", "MIXTURE_ROWS", "HYDRAULIC_ROWS", "FORMULA_INPUTS",
    "ResidentMixtureV64", "ResidentHydraulicV64",
    "ResidentScalarFlowInputsV64", "ResidentScalarFlowCapacityV64",
    "ScalarFlowProducersV64CUDA",
]
