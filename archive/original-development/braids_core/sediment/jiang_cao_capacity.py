"""Paired slope-aware MPM component; NOT an admitted river execution option.

Jiang-Cao (2026) Eqs. 5, 10, 12, with exponent 3/2 explicitly selected by the
user on 2026-10-27. Shared CPU/CUDA kernel; fixed physical coefficients; one D50.
Caller supplies physical Manning roughness and Cartesian bed gradients.
No mobile-bedload transport, settling, mixture coupling, collapse or secondary
momentum closure is implied. Never attach this as an equilibrium Exner alias.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path

import numpy as np

METHOD = "capacity.mpm_jc2026_p15_reconstruction_v1"
FIELDS = (
    "qb", "qx", "qy", "theta", "critical_shields", "k_longitudinal", "k_transverse",
    "slope_along", "slope_left", "chezy", "ustar_squared", "tau_mixture",
    "dqb_dspeed", "dqb_dh", "dqb_dhu", "dqb_dhv", "dqx_dh", "dqx_dhu", "dqx_dhv",
    "dqy_dh", "dqy_dhu", "dqy_dhv", "slope_deflection_rad",
)
STATUSES = {
    0: "valid", 1: "dry: zero bedload", 2: "stationary: zero bedload",
    3: "invalid input: require finite nonnegative depth, positive Manning and 0<=c<1",
    4: "repose redistribution required before capacity: |grad(z)| must be below tan(repose)",
    5: "nonfinite or out-of-domain derived value; no numerical floor or fallback applied",
}


@dataclass(frozen=True)
class JiangCaoMaterial:
    d50_m: float
    solid_density_kg_m3: float
    water_density_kg_m3: float
    repose_angle_rad: float
    gravity_m_s2: float = 9.81
    dry_depth_m: float = 1.e-6
    diameter_definition: str = "d50"

    def __post_init__(self):
        if self.diameter_definition != "d50":
            raise ValueError("Jiang-Cao single-grain reconstruction requires D50, not another diameter basis")
        values = list(asdict(self).values())[:-1]
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not np.isfinite(v)
               for v in values):
            raise ValueError("Finite real material and numerical controls required")
        if not (self.d50_m > 0 and self.solid_density_kg_m3 > self.water_density_kg_m3 > 0
                and 0 < self.repose_angle_rad < np.pi/2 and self.gravity_m_s2 > 0
                and self.dry_depth_m >= 0):
            raise ValueError("Positive grain, densities, gravity, repose in (0,pi/2), and nonnegative dry depth required")
        # Verify immutable derived quantities once, outside the cell loop.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            wire = self.wire()
        if not np.isfinite(wire).all() or min(wire[1], wire[6], wire[7]) <= 0:
            raise ValueError("Material outside finite constitutive range")

    def wire(self):
        relative = (self.solid_density_kg_m3-self.water_density_kg_m3)/self.water_density_kg_m3
        scale = 8*np.sqrt(relative*self.gravity_m_s2*self.d50_m)*self.d50_m
        return np.asarray([self.d50_m, relative, self.solid_density_kg_m3,
                           self.water_density_kg_m3, self.gravity_m_s2, self.dry_depth_m,
                           np.tan(self.repose_angle_rad), scale], dtype=np.float64)


@dataclass(frozen=True)
class CapacityEvaluation:
    values: object
    status: object

    def field(self, name):
        return self.values[FIELDS.index(name)]


class JiangCaoCapacity:
    """Stateless component. All physical cell arithmetic executes on the chosen backend.

    Inputs q=(h,hu,hv), grad_z=(dz/dx,dz/dy), Manning and optional volume
    concentration. Concentration affects diagnostic mixture traction only;
    its density cancels from Shields. Outputs are solid m2/s, not kg/(m s).
    Derivatives hold Manning and Cartesian grad_z fixed. They are NOT the
    Jacobian of a future dynamic bedform-roughness model.
    """

    def __init__(self, backend, material: JiangCaoMaterial, *, threads=4):
        if backend not in ("cpu", "gpu"):
            raise ValueError("Choose cpu or gpu explicitly; no fallback")
        if not isinstance(material, JiangCaoMaterial):
            raise TypeError("Explicit JiangCaoMaterial required")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("Positive integer CPU thread count required")
        from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
        self.backend, self.material = backend, material
        if backend == "gpu":
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np
        path = Path(__file__).with_suffix(".h")
        self.engine = WaveEngine(backend, threads, source=path, names=("jc_capacity",))
        self._wire = self.xp.asarray(material.wire())
        runtime = dict(self.engine.identity)
        if backend == "gpu":
            from .bend_methods_v1 import OPTIONS
            props = self.xp.cuda.runtime.getDeviceProperties(self.xp.cuda.runtime.getDevice())
            name = props["name"]
            runtime.update(options=list(OPTIONS), cupy_version=self.xp.__version__,
                           driver_version=self.xp.cuda.runtime.driverGetVersion(),
                           device_name=name.decode() if isinstance(name, bytes) else name,
                           compute_capability=[props["major"], props["minor"]])
        self.identity = dict(
            method=METHOD, public_support=False, scope="capacity_shear_slope_component_only",
            coefficient=8., critical_shields_flat=.047, exponent=1.5,
            exponent_authority="user-selected 2026-10-27; authors' executed exponent unconfirmed",
            transverse_slope_coefficient=1.5, curvature_coefficient=0.,
            material=asdict(material), forcing="physical_Manning_no_FLOW_mu_or_extra_Bagnold",
            gradient="Cartesian_bed_elevation; streamwise_uphill_and_left_uphill",
            derivative="full_q=(h,hu,hv)_Jacobian_fixed_Manning_and_Cartesian_gradient",
            repose_policy="reject_wet_at_or_above_total_repose; no_slope_clipping",
            runtime=runtime,
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        self.identity["sha256"] = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def evaluate(self, q, grad_z, manning, *, concentration=0., strict=True):
        xp = self.xp
        q = xp.ascontiguousarray(xp.asarray(q, dtype=xp.float64))
        grad_z = xp.ascontiguousarray(xp.asarray(grad_z, dtype=xp.float64))
        if q.ndim != 2 or q.shape[0] != 3 or q.shape[1] > np.iinfo(np.int32).max:
            raise ValueError("q must have shape (3,n), with n fitting the native cell count")
        n = q.shape[1]
        if grad_z.shape != (2, n):
            raise ValueError("Cartesian bed gradient must have shape (2,n)")
        rough = xp.ascontiguousarray(xp.broadcast_to(xp.asarray(manning, dtype=xp.float64), (n,)))
        c = xp.ascontiguousarray(xp.broadcast_to(xp.asarray(concentration, dtype=xp.float64), (n,)))
        # Caller-owned outputs are never aliases of reused/private state.
        out, status = xp.empty((len(FIELDS), n)), xp.empty(n, dtype=xp.int32)
        if n:
            self.engine.launch("jc_capacity", n, n, self._wire, q, grad_z, rough, c, out, status)
        if strict and bool(xp.any(status >= 3)):
            index = int(xp.flatnonzero(status >= 3)[0])
            code = int(status[index])
            raise ValueError(f"Jiang-Cao capacity cell {index}: {STATUSES[code]}")
        return CapacityEvaluation(out, status)
