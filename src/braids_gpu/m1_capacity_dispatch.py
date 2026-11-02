"""Common resident-GPU dispatch for Delft3D-current M1 capacities.

This is a bounded integration prototype, not yet the coupled face/Exner
runtime.  It composes the independently qualified C/M/W roughness,
Soulsby/Clarke skin shear, and current-capacity CUDA operators behind one
strict class-major ABI.  Physical cell/class arrays never visit the host.

The API deliberately rejects three common sources of silent physics loss:

* a formula's suspended output is always retained in a separate array;
* direct-vector formulae cannot receive external secondary-flow steering, and
  scalar steering rotates only bed transport while suspended transport keeps
  Delft's depth-averaged flow direction;
* Van Rijn 1993 is listed but rejected because the existing CUDA prototype
  supplies only its bed vector, not the required equilibrium suspended route.

Formula parameters use the exact locked ``rdtrafrm`` keywords and must already
be resolved to one resident vector per sediment class.  Defaults therefore
belong to project/deck resolution, not to this numerical dispatch layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from braids_core.sediment.capacity_parameters import formula_contract
from .hydraulic_roughness import HydraulicRoughnessCUDA


CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAMES = (
    "m1_basic_capacity",
    "m1_wilcock_capacity_chezy",
    "m1_rotate_scalar_capacity",
)


@dataclass(frozen=True)
class CapacityDispatchContract:
    law: int
    method_id: str
    direction: str
    output: str
    suspended: str
    generic_hiding: bool
    skin_shear: bool
    complete: bool = True
    incomplete_reason: str | None = None

    @property
    def external_direction(self) -> bool:
        return self.direction == "scalar_direction"

    @property
    def bed_direction(self) -> str:
        """Stable bed-channel direction metadata (legacy ``direction`` alias)."""
        return self.direction

    @property
    def suspended_direction(self) -> str:
        """Delft suspended direction; never external secondary steering."""
        if self.suspended == "zero":
            return "not_applicable"
        if self.direction == "direct_vector":
            return "direct_vector"
        return "depth_averaged_flow_direction"


def _contract(law: int, method_id: str, direction: str, output: str,
              suspended: str, hiding: bool = False, skin: bool = False,
              *, complete: bool = True,
              reason: str | None = None) -> CapacityDispatchContract:
    return CapacityDispatchContract(
        law, method_id, direction, output, suspended, hiding, skin,
        complete, reason,
    )


_CONTRACTS = (
    _contract(0, "capacity.off", "none", "off", "zero"),
    _contract(
        1, "capacity.van_rijn_1993", "direct_vector",
        "split_bed_suspended_vectors", "equilibrium_concentration_route",
        complete=False,
        reason=(
            "Van Rijn 1993 CUDA currently provides only the bed vector; the "
            "required 2DH equilibrium-concentration/suspended route is not "
            "part of the M1 resident ABI"
        ),
    ),
    _contract(2, "capacity.mpm", "scalar_direction", "scalar_split", "zero",
              hiding=True),
    _contract(3, "capacity.ashida_michiue_generalized", "scalar_direction",
              "scalar_split", "zero", hiding=True),
    _contract(4, "capacity.soulsby", "direct_vector", "split_vectors", "zero"),
    _contract(5, "capacity.wilcock_crowe", "scalar_direction",
              "intrinsic_mixture_scalar", "zero", skin=True),
    _contract(6, "capacity.wilcock_crowe_modified", "scalar_direction",
              "intrinsic_mixture_scalar", "zero", skin=True),
    _contract(7, "capacity.gaeuman", "scalar_direction",
              "intrinsic_mixture_scalar", "zero", skin=True),
    _contract(8, "capacity.engelund_hansen", "scalar_direction",
              "scalar_split", "explicit"),
    _contract(11, "capacity.general", "scalar_direction", "scalar_split",
              "explicit", hiding=True),
    _contract(12, "capacity.van_rijn_1984_modified", "scalar_direction",
              "scalar_split", "explicit"),
    _contract(13, "capacity.soulsby_van_rijn", "direct_vector",
              "split_vectors", "explicit"),
    _contract(14, "capacity.wang_fredsoe", "scalar_direction",
              "scalar_split_signed_suspended", "explicit_signed"),
)
DISPATCH_CONTRACTS = MappingProxyType({row.law: row for row in _CONTRACTS})


class IncompleteCapacityContractError(ValueError):
    """Raised before numerical work when an output channel is unavailable."""


@dataclass(frozen=True)
class ResidentM1CapacityResult:
    """One complete resident result.

    All transport and diagnostic arrays use ``(component, class, cell)`` or
    ``(class, cell)`` layout.  ``status`` is per cell while ``point_status``
    preserves the class-level rejection source.  ``formula_auxiliary`` keeps
    formula-specific rows that cannot safely be overloaded into Shields
    diagnostics.
    """

    contract: CapacityDispatchContract
    bed_mass: object
    suspended_mass: object
    absolute_shields: object
    critical_shields: object
    stress_ratio: object
    formula_auxiliary: object
    chezy: object
    hydraulic_z0: object
    hydraulic_drag: object
    total_shear: object
    sediment_shear: object
    mixture: object | None
    point_status: object
    status: object
    roughness_status: object
    skin_status: object | None
    direction_applied: bool


class ResidentM1CapacityDispatch:
    """Route supported non-bedform Delft3D-current laws on one CUDA device."""

    def __init__(self, source_path: str | Path | None = None, *,
                 roughness_source_path: str | Path | None = None,
                 delft_source_path: str | Path | None = None,
                 parameterized_source_path: str | Path | None = None,
                 skin_source_path: str | Path | None = None):
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
            self.roughness = HydraulicRoughnessCUDA(
                source_path=roughness_source_path)
        self._dependency_sources = dict(
            delft=delft_source_path,
            parameterized=parameterized_source_path,
            skin=skin_source_path,
        )
        self._delft = None
        self._parameterized = None
        self._skin = None

    @staticmethod
    def contract(law: int) -> CapacityDispatchContract:
        if isinstance(law, (bool, np.bool_)):
            raise ValueError(f"Unsupported M1 capacity law: {law!r}")
        try:
            return DISPATCH_CONTRACTS[int(law)]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Unsupported M1 capacity law: {law!r}") from error

    def _lazy_delft(self):
        if self._delft is None:
            from .delft_current_capacity import DelftCurrentCapacityCUDA
            self._delft = DelftCurrentCapacityCUDA(
                source_path=self._dependency_sources["delft"])
        return self._delft

    def _lazy_parameterized(self):
        if self._parameterized is None:
            from .delft_parameterized_bedload import DelftParameterizedBedloadCUDA
            self._parameterized = DelftParameterizedBedloadCUDA(
                source_path=self._dependency_sources["parameterized"])
        return self._parameterized

    def _lazy_skin(self):
        if self._skin is None:
            from .sediment_skin_shear import SedimentSkinShearCUDA
            self._skin = SedimentSkinShearCUDA(
                source_path=self._dependency_sources["skin"])
        return self._skin

    def _array(self, name: str, value: object, *, ndim: int,
               shape: tuple[int, ...] | None = None,
               integer: bool = False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(f"{name} is not resident on CUDA device {self.device_id}")
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value

    def _class_to_points(self, value, ncell: int):
        cp = self.cp
        return cp.ascontiguousarray(cp.broadcast_to(value[:, None],
                                                     (value.size, ncell))).reshape(-1)

    def _cell_to_points(self, value, nclass: int):
        cp = self.cp
        return cp.ascontiguousarray(cp.broadcast_to(value[None, :],
                                                     (nclass, value.size))).reshape(-1)

    def _matrix_to_points(self, value):
        return value.reshape(-1)

    def _parameters(self, law: int, supplied: Mapping[str, object],
                    nclass: int):
        cp = self.cp
        if not isinstance(supplied, Mapping):
            raise TypeError("Resolved capacity parameters must be a mapping")
        specification = formula_contract(law)
        expected = tuple(row.keyword for row in specification.parameters)
        missing = tuple(name for name in expected if name not in supplied)
        extra = tuple(name for name in supplied if name not in expected)
        if missing or extra:
            raise ValueError(
                f"Law {law} resolved parameter fields differ; "
                f"missing={missing}, extra={extra}"
            )
        result = {}
        invalid = cp.zeros(nclass, dtype=cp.int32)
        for row in specification.parameters:
            integer = row.value_kind == "integer"
            value = self._array(
                f"parameters[{row.keyword!r}]", supplied[row.keyword], ndim=1,
                shape=(nclass,), integer=integer,
            )
            if integer:
                if row.allowed:
                    choices = cp.asarray(row.allowed, dtype=cp.int32)
                    invalid = cp.maximum(
                        invalid, (~cp.isin(value, choices)).astype(cp.int32),
                    )
            else:
                invalid = cp.maximum(
                    invalid, (~cp.isfinite(value)).astype(cp.int32),
                )
            result[row.keyword] = value
        return MappingProxyType(result), invalid

    @staticmethod
    def _positive_scalar(name: str, value: float) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite positive scalar")
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be a finite positive scalar")
        return result

    def _shields_curve(self, dstar):
        cp = self.cp
        return cp.where(
            dstar <= 4.0, 0.240/dstar,
            cp.where(
                dstar <= 10.0, 0.140/cp.power(dstar, 0.64),
                cp.where(
                    dstar <= 20.0, 0.040/cp.power(dstar, 0.10),
                    cp.where(dstar <= 150.0, 0.013*cp.power(dstar, 0.29), 0.055),
                ),
            ),
        )

    def evaluate(
        self, law: int, *,
        depth, u, v, roughness_parameter, kinematic_viscosity,
        d50, d90, solid_density, parameters: Mapping[str, object] | None = None,
        roughness_mode: str | int = "manning", water_density: float = 1000.0,
        gravity: float = 9.81, minimum_work_depth: float = 0.0,
        waqua_white_colebrook: bool = False, hiding=None,
        mixture_d90=None,
        skin_fields: Mapping[str, object] | None = None,
        transport_direction=None, observation_depth=None, roughness_z0=None,
        surface_mass=None, bulk_density=None, sediment_type=None,
        mud_fraction=None, reject_invalid: bool = True,
    ) -> ResidentM1CapacityResult:
        """Evaluate one complete configured formula on resident arrays.

        ``parameters`` contains every resolved source keyword as a class
        vector.  ``transport_direction`` is an optional ``(2,ncell)`` target
        bed direction and is admitted only for scalar-direction contracts.
        It never rotates the suspended channel, which retains Delft's
        depth-averaged flow direction.
        ``skin_fields`` contains only the six Soulsby/Clarke bed descriptors;
        depth, velocity, viscosity and density are taken from the common state.
        """
        contract = self.contract(law)
        if not contract.complete:
            raise IncompleteCapacityContractError(
                f"Law {law} has an incomplete resident output contract: "
                f"{contract.incomplete_reason}"
            )
        cp = self.cp
        depth = self._array("depth", depth, ndim=1)
        ncell = int(depth.size)
        u = self._array("u", u, ndim=1, shape=(ncell,))
        v = self._array("v", v, ndim=1, shape=(ncell,))
        roughness_parameter = self._array(
            "roughness_parameter", roughness_parameter, ndim=1, shape=(ncell,),
        )
        kinematic_viscosity = self._array(
            "kinematic_viscosity", kinematic_viscosity, ndim=1, shape=(ncell,),
        )
        d50 = self._array("d50", d50, ndim=1)
        nclass = int(d50.size)
        if nclass <= 0:
            raise ValueError("M1 capacity dispatch requires at least one sediment class")
        d90 = self._array("d90", d90, ndim=1, shape=(nclass,))
        solid_density = self._array(
            "solid_density", solid_density, ndim=1, shape=(nclass,),
        )
        water_density = self._positive_scalar("water_density", water_density)
        gravity = self._positive_scalar("gravity", gravity)
        if (not np.isfinite(float(minimum_work_depth)) or
                float(minimum_work_depth) < 0.0):
            raise ValueError("minimum_work_depth must be finite and nonnegative")
        if law in (2, 12, 14):
            if mixture_d90 is None:
                raise ValueError(
                    f"Law {law} requires resident local mixture_d90; "
                    "per-fraction d90 is not a valid fallback"
                )
            mixture_d90 = self._array(
                "mixture_d90", mixture_d90, ndim=1, shape=(ncell,),
            )
        elif mixture_d90 is not None:
            # Keep the common ABI honest: this value is FLOW's local
            # RP_D90MX state and must not be accepted as an inert decoration
            # for formulae that do not consume it.
            raise ValueError(f"Law {law} does not consume mixture_d90")

        if law == 0:
            if parameters is not None and len(parameters) != 0:
                raise ValueError("Capacity-off accepts no formula parameters")
            parameter_invalid = cp.zeros(nclass, dtype=cp.int32)
            resolved = MappingProxyType({})
        else:
            resolved, parameter_invalid = self._parameters(
                law, parameters if parameters is not None else {}, nclass,
            )

        if hiding is not None and not contract.generic_hiding:
            raise ValueError(
                f"Law {law} does not admit a generic hiding factor; "
                "omit it rather than silently applying or discarding it"
            )
        if hiding is None:
            hiding_matrix = cp.ones((nclass, ncell), dtype=cp.float64)
        else:
            hiding_matrix = self._array(
                "hiding", hiding, ndim=2, shape=(nclass, ncell),
            )
        if transport_direction is not None and not contract.external_direction:
            raise ValueError(
                f"Law {law} returns {contract.direction}; external direction "
                "steering is forbidden"
            )
        if skin_fields is not None and not contract.skin_shear:
            raise ValueError(
                f"Law {law} does not consume RP_TAUB skin shear; omit skin_fields"
            )

        speed = cp.sqrt(u*u+v*v)
        common_invalid = (
            (~cp.isfinite(u)) | (~cp.isfinite(v)) |
            (~cp.isfinite(kinematic_viscosity)) |
            (kinematic_viscosity <= 0.0) |
            ((depth == 0.0) & (speed != 0.0))
        ).astype(cp.int32)
        with cp.cuda.Device(self.device_id):
            chezy, hydraulic_z0, hydraulic_drag, roughness_status = (
                self.roughness.state(
                    depth, roughness_parameter, mode=roughness_mode,
                    gravity=gravity,
                    minimum_work_depth=float(minimum_work_depth),
                    waqua_white_colebrook=bool(waqua_white_colebrook),
                )
            )
            total_shear = water_density*hydraulic_drag*speed*speed
            sediment_shear = total_shear
            skin_status = None
            if skin_fields is not None:
                required = (
                    "ks_sand", "ks_silt", "mud_factor", "critical_mud_1",
                    "critical_mud_2", "fluff_fraction",
                )
                missing = tuple(name for name in required if name not in skin_fields)
                extra = tuple(name for name in skin_fields if name not in required)
                if missing or extra:
                    raise ValueError(
                        f"Skin fields differ; missing={missing}, extra={extra}"
                    )
                skin = {
                    name: self._array(
                        f"skin_fields[{name!r}]", skin_fields[name], ndim=1,
                        shape=(ncell,),
                    ) for name in required
                }
                rho_cell = cp.full(ncell, water_density, dtype=cp.float64)
                # compbsskin is called only for wet FLOW points.  Give dry,
                # zero-speed points a harmless positive evaluation depth and
                # mask their sediment shear back to zero.  A dry point with
                # nonzero velocity is rejected by ``common_invalid``.
                skin_depth = cp.where(
                    depth > 0.0, depth,
                    max(float(minimum_work_depth), 1.0e-12),
                )
                skin_result = self._lazy_skin().evaluate(
                    depth=skin_depth, speed=speed,
                    kinematic_viscosity=kinematic_viscosity,
                    water_density=rho_cell, reject_invalid=False, **skin,
                )
                sediment_shear = cp.where(
                    depth > 0.0, skin_result.rp_taub, 0.0,
                )
                skin_status = skin_result.status

            points = nclass*ncell
            formula_auxiliary = cp.zeros((2, nclass, ncell), dtype=cp.float64)
            mixture = None

            if law == 0:
                bed = cp.zeros((2, nclass, ncell), dtype=cp.float64)
                suspended = cp.zeros_like(bed)
                absolute = cp.zeros((nclass, ncell), dtype=cp.float64)
                critical = cp.zeros_like(absolute)
                ratio = cp.zeros_like(absolute)
                point_status = cp.zeros((nclass, ncell), dtype=cp.int32)
            elif law in (2, 8):
                bed = cp.empty((2, nclass, ncell), dtype=cp.float64)
                suspended = cp.empty_like(bed)
                absolute = cp.empty((nclass, ncell), dtype=cp.float64)
                critical = cp.empty_like(absolute)
                ratio = cp.empty_like(absolute)
                point_status = cp.empty((nclass, ncell), dtype=cp.int32)
                acal = self._class_to_points(resolved["ACal"], ncell)
                if law == 8:
                    susfac = self._class_to_points(resolved["SusFac"], ncell)
                else:
                    susfac = cp.zeros(points, dtype=cp.float64)
                # ``m1_basic_capacity`` consumes this pointer only for MPM.
                # Keeping a cell-shaped inert pointer for EH avoids a host
                # scalar or a second kernel ABI while preserving the fact that
                # EH has no D90 input.
                basic_mixture_d90 = mixture_d90 if law == 2 else depth
                self.kernels["m1_basic_capacity"](
                    ((points+255)//256,), (256,),
                    (np.int64(ncell), np.int32(nclass), np.int32(law),
                     depth, u, v, chezy, d50, basic_mixture_d90, solid_density,
                     np.float64(water_density), np.float64(gravity), acal,
                     susfac, self._matrix_to_points(hiding_matrix), bed,
                     suspended, absolute, critical, ratio, point_status),
                )
            elif law in (5, 6, 7):
                bulk_density = self._array(
                    "bulk_density", bulk_density, ndim=1, shape=(nclass,),
                )
                sediment_type = self._array(
                    "sediment_type", sediment_type, ndim=1,
                    shape=(nclass,), integer=True,
                )
                surface_mass = self._array(
                    "surface_mass", surface_mass, ndim=2,
                    shape=(nclass, ncell),
                )
                bed = cp.empty((2, nclass, ncell), dtype=cp.float64)
                suspended = cp.empty_like(bed)
                absolute = cp.empty((nclass, ncell), dtype=cp.float64)
                critical = cp.empty_like(absolute)
                ratio = cp.empty_like(absolute)
                cell_formula_status = cp.empty(ncell, dtype=cp.int32)
                mixture = cp.empty((3, ncell), dtype=cp.float64)
                if law == 5:
                    parameter_1 = self._class_to_points(resolved["ACal"], ncell)
                    parameter_2 = cp.zeros(points, dtype=cp.float64)
                else:
                    parameter_1 = self._class_to_points(
                        resolved["ThetaC0"], ncell,
                    )
                    parameter_2 = self._class_to_points(
                        resolved["Alpha0"], ncell,
                    )
                self.kernels["m1_wilcock_capacity_chezy"](
                    ((ncell+255)//256,), (256,),
                    (np.int64(ncell), np.int32(nclass), np.int32(law),
                     depth, u, v, chezy, sediment_shear,
                     np.float64(water_density), np.float64(gravity), d50,
                     solid_density, bulk_density, surface_mass, sediment_type,
                     parameter_1, parameter_2, bed, suspended, absolute,
                     critical, ratio, mixture, cell_formula_status),
                )
                point_status = cp.ascontiguousarray(cp.broadcast_to(
                    cell_formula_status[None, :], (nclass, ncell),
                ))
            elif law in (3, 4):
                point_u = self._cell_to_points(u, nclass)
                point_v = self._cell_to_points(v, nclass)
                point_depth = self._cell_to_points(depth, nclass)
                point_d50 = self._class_to_points(d50, ncell)
                point_rhos = self._class_to_points(solid_density, ncell)
                point_rho = cp.full(points, water_density, dtype=cp.float64)
                point_g = cp.full(points, gravity, dtype=cp.float64)
                if law == 3:
                    result = self._lazy_parameterized().evaluate(
                        14, reject_invalid=False, u=point_u, v=point_v,
                d50=point_d50, solid_density=point_rhos,
                        water_density=point_rho, gravity=point_g,
                        chezy=self._cell_to_points(chezy, nclass),
                        hiding_factor=self._matrix_to_points(hiding_matrix),
                        acal=self._class_to_points(resolved["ACal"], ncell),
                        critical_shields=self._class_to_points(
                            resolved["ThetaC"], ncell,
                        ),
                        power_m=self._class_to_points(resolved["PowerM"], ncell),
                        power_p=self._class_to_points(resolved["PowerP"], ncell),
                        power_q=self._class_to_points(resolved["PowerQ"], ncell),
                    )
                    bed = result.mass_flux.reshape(2, nclass, ncell)
                    suspended = cp.zeros_like(bed)
                    ratio = result.stress_ratio.reshape(nclass, ncell)
                    relative = point_rhos/point_rho-1.0
                    point_chezy = self._cell_to_points(chezy, nclass)
                    denominator = point_chezy*point_chezy*relative*point_d50
                    positive_denominator = denominator > 0.0
                    absolute_flat = cp.where(
                        positive_denominator,
                        (point_u*point_u+point_v*point_v)/cp.where(
                            positive_denominator, denominator, 1.0,
                        ),
                        0.0,
                    )
                    absolute = absolute_flat.reshape(nclass, ncell)
                    critical = (self._matrix_to_points(hiding_matrix)*
                                self._class_to_points(
                                    resolved["ThetaC"], ncell,
                                )).reshape(nclass, ncell)
                else:
                    result = self._lazy_parameterized().evaluate(
                        12, reject_invalid=False, u=point_u, v=point_v,
                        depth=point_depth, d50=point_d50,
                        solid_density=point_rhos, water_density=point_rho,
                        gravity=point_g,
                        kinematic_viscosity=self._cell_to_points(
                            kinematic_viscosity, nclass,
                        ),
                        acal=self._class_to_points(resolved["ACal"], ncell),
                        model_index=self._class_to_points(
                            resolved["ModInd"], ncell,
                        ),
                        ratio_d50_z0=self._class_to_points(
                            resolved["RatioD50Z0"], ncell,
                        ),
                        von_karman=cp.full(points, 0.41, dtype=cp.float64),
                    )
                    bed = result.mass_flux.reshape(2, nclass, ncell)
                    suspended = cp.zeros_like(bed)
                    ratio = result.stress_ratio.reshape(nclass, ncell)
                    relative = point_rhos/point_rho-1.0
                    dstar = (cp.power(
                        gravity*relative/
                        self._cell_to_points(kinematic_viscosity, nclass)**2,
                        1.0/3.0,
                    )*point_d50)
                    critical = self._shields_curve(dstar).reshape(nclass, ncell)
                    absolute = ratio*critical
                point_status = result.status.reshape(nclass, ncell)
            else:
                # Exact Delft scalar/vector point operators (TraFrm 4, 7,
                # 11 and 13) are expanded on-device from cell/class storage.
                point_u = self._cell_to_points(u, nclass)
                point_v = self._cell_to_points(v, nclass)
                point_depth = self._cell_to_points(depth, nclass)
                point_d50 = self._class_to_points(d50, ncell)
                point_d90 = self._class_to_points(d90, ncell)
                if law in (12, 14):
                    point_d90 = self._cell_to_points(mixture_d90, nclass)
                point_rhos = self._class_to_points(solid_density, ncell)
                point_rho = cp.full(points, water_density, dtype=cp.float64)
                point_g = cp.full(points, gravity, dtype=cp.float64)
                operator = self._lazy_delft()
                if law == 11:
                    safe_chezy = cp.where(depth > 0.0, chezy, 1.0)
                    result = operator.evaluate(
                        4, reject_invalid=False, u=point_u, v=point_v,
                        d50=point_d50, solid_density=point_rhos,
                        water_density=point_rho, gravity=point_g,
                        chezy=self._cell_to_points(safe_chezy, nclass),
                        hiding_factor=self._matrix_to_points(hiding_matrix),
                        bed_calibration=self._class_to_points(
                            resolved["ACal"], ncell,
                        ),
                        bed_shields_power=self._class_to_points(
                            resolved["PowerB"], ncell,
                        ),
                        bed_excess_power=self._class_to_points(
                            resolved["PowerC"], ncell,
                        ),
                        bed_ripple_factor=self._class_to_points(
                            resolved["RipFac"], ncell,
                        ),
                        bed_critical_shields=self._class_to_points(
                            resolved["ThetaC"], ncell,
                        ),
                        sus_calibration=self._class_to_points(
                            resolved["SusACal"], ncell,
                        ),
                        sus_shields_power=self._class_to_points(
                            resolved["SusPowerB"], ncell,
                        ),
                        sus_excess_power=self._class_to_points(
                            resolved["SusPowerC"], ncell,
                        ),
                        sus_ripple_factor=self._class_to_points(
                            resolved["SusRipFac"], ncell,
                        ),
                        sus_critical_shields=self._class_to_points(
                            resolved["SusThetaC"], ncell,
                        ),
                    )
                elif law == 12:
                    mud_fraction = self._array(
                        "mud_fraction", mud_fraction, ndim=2,
                        shape=(nclass, ncell),
                    )
                    result = operator.evaluate(
                        7, reject_invalid=False, depth=point_depth,
                        u=point_u, v=point_v, d50=point_d50, d90=point_d90,
                        solid_density=point_rhos, water_density=point_rho,
                        gravity=point_g,
                        kinematic_viscosity=self._cell_to_points(
                            kinematic_viscosity, nclass,
                        ),
                        calibration=self._class_to_points(
                            resolved["ACal"], ncell,
                        ),
                        reference_height=self._class_to_points(
                            resolved["Aks"], ncell,
                        ),
                        settling_velocity=self._class_to_points(
                            resolved["WSettle"], ncell,
                        ),
                        mud_power=self._class_to_points(
                            resolved["BetaM"], ncell,
                        ),
                        mud_fraction=self._matrix_to_points(mud_fraction),
                    )
                elif law == 13:
                    result = operator.evaluate(
                        11, reject_invalid=False, depth=point_depth,
                        u=point_u, v=point_v, d50=point_d50,
                        solid_density=point_rhos, water_density=point_rho,
                        gravity=point_g,
                        kinematic_viscosity=self._cell_to_points(
                            kinematic_viscosity, nclass,
                        ),
                        calibration=self._class_to_points(
                            resolved["ACal"], ncell,
                        ),
                        d90_d50=self._class_to_points(
                            resolved["RatioD90D50"], ncell,
                        ),
                        roughness_z0=self._class_to_points(
                            resolved["RouZ0"], ncell,
                        ),
                    )
                elif law == 14:
                    safe_depth = cp.where(depth > 0.0, depth, 0.01)
                    result = operator.evaluate(
                        13, reject_invalid=False,
                        depth=self._cell_to_points(safe_depth, nclass),
                        u=point_u, v=point_v, d50=point_d50, d90=point_d90,
                        solid_density=point_rhos, water_density=point_rho,
                        gravity=point_g,
                        kinematic_viscosity=self._class_to_points(
                            resolved["VicMol"], ncell,
                        ),
                        chezy=self._cell_to_points(chezy, nclass),
                        current_shear_velocity=self._cell_to_points(
                            cp.sqrt(total_shear/water_density), nclass,
                        ),
                        calibration=self._class_to_points(
                            resolved["ACal"], ncell,
                        ),
                    )
                else:  # pragma: no cover - guarded by the immutable table.
                    raise AssertionError(f"Missing M1 dispatch for law {law}")
                values = result.values.reshape(7, nclass, ncell)
                bed = values[0:2]
                suspended = values[2:4]
                absolute = values[4]
                critical = values[5]
                formula_auxiliary = values[5:7].copy()
                if law == 11:
                    critical = (self._matrix_to_points(hiding_matrix)*
                                self._class_to_points(
                                    resolved["ThetaC"], ncell,
                                )).reshape(nclass, ncell)
                    ratio = cp.where(
                        critical>0.0, absolute/critical,
                        cp.where(absolute>0.0,
                                 np.finfo(np.float64).max, 0.0),
                    )
                elif law == 12:
                    ratio = cp.where(
                        critical>0.0, absolute/critical,
                        cp.where(absolute>0.0,
                                 np.finfo(np.float64).max, 0.0),
                    )
                else:
                    ratio = cp.where(
                        critical>0.0, absolute/critical,
                        cp.where(absolute>0.0,
                                 np.finfo(np.float64).max, 0.0),
                    )
                point_status = result.status.reshape(nclass, ncell)

            point_status = cp.maximum(
                cp.ascontiguousarray(point_status), parameter_invalid[:, None],
            ).astype(cp.int32, copy=False)
            point_status = cp.maximum(
                point_status, common_invalid[None, :],
            ).astype(cp.int32, copy=False)
            if skin_status is not None:
                point_status = cp.maximum(
                    point_status, skin_status[None, :],
                ).astype(cp.int32, copy=False)

            direction_applied = transport_direction is not None
            if transport_direction is not None:
                transport_direction = self._array(
                    "transport_direction", transport_direction, ndim=2,
                    shape=(2, ncell),
                )
                self.kernels["m1_rotate_scalar_capacity"](
                    ((points+255)//256,), (256,),
                    (np.int64(ncell), np.int32(nclass), u, v,
                     transport_direction[0], transport_direction[1], bed,
                     point_status),
                )

            status = cp.max(point_status, axis=0).astype(cp.int32, copy=False)
            # ``reject_invalid=False`` is a diagnostic mode, not permission
            # to expose transport from a malformed configuration.  Kernels
            # already zero their own invalid entries; apply the same device-
            # resident fail-closed rule to dormant-parameter and composed
            # state errors discovered outside a formula kernel.
            valid_points = point_status == 0
            bed = cp.where(valid_points[None, :, :], bed, 0.0)
            suspended = cp.where(valid_points[None, :, :], suspended, 0.0)
            absolute = cp.where(valid_points, absolute, 0.0)
            critical = cp.where(valid_points, critical, 0.0)
            ratio = cp.where(valid_points, ratio, 0.0)
            formula_auxiliary = cp.where(
                valid_points[None, :, :], formula_auxiliary, 0.0,
            )
            if mixture is not None:
                mixture = cp.where(status[None, :] == 0, mixture, 0.0)
            if reject_invalid and ncell:
                invalid_count = int(cp.count_nonzero(status).item())
                if invalid_count:
                    raise ValueError(
                        f"Resident M1 law {law} rejected "
                        f"{invalid_count} invalid cell(s)"
                    )

        return ResidentM1CapacityResult(
            contract=contract, bed_mass=bed, suspended_mass=suspended,
            absolute_shields=absolute, critical_shields=critical,
            stress_ratio=ratio, formula_auxiliary=formula_auxiliary,
            chezy=chezy, hydraulic_z0=hydraulic_z0,
            hydraulic_drag=hydraulic_drag, total_shear=total_shear,
            sediment_shear=sediment_shear, mixture=mixture,
            point_status=point_status, status=status,
            roughness_status=roughness_status, skin_status=skin_status,
            direction_applied=direction_applied,
        )


# Longer descriptive alias for callers and reports.
ResidentDelftCurrentCapacityDispatch = ResidentM1CapacityDispatch
