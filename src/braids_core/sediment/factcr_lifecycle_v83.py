"""Route-aware Delft3D4 FLOW ``FactCr`` composition (Q4/v83).

The locked FLOW source computes, for every noncohesive fraction,

``taucr0 = FactCr * (rho_s-rho_w) * g * D50 * theta_cr``.

That value is *not* consumed by every transport formula.  ``eqtran`` passes it
only to the negative Van Rijn ``tram1``/``tram2`` routes and those routines use
it only when their private ``iTauCr`` selector equals one.  V83 therefore
admits one honest coupled slice: current-only ``TraFrm=-1`` (Van Rijn 1993),
pure sand, ``iTauCr=1``, fixed grid, two or three fractions and at most four
accepted ``implicit_fwave_midpoint`` updates.  ``TraFrm=-2/-4`` critical-shear
formation is equation-audited but their constitutive producers are not claimed.

The accepted v66 producer and v78 lifecycle are frozen.  This additive owner
forms the threshold exactly once, evaluates the constitutive producer, bridges
its raw equilibrium concentration to the v78 pre-fraction capacity, and binds
all evidence to one immutable transaction.  Bed capacity is retained as
diagnostic Q3 evidence; an Exner update is deliberately outside this milestone.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .multifraction_suspended_lifecycle_v78 import (
    MAX_ACCEPTED_UPDATES,
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    MultifractionSuspendedExecutionV78,
    MultifractionSuspendedStateV78,
    MultifractionSuspendedTrialV78,
    begin_multifraction_cpu_oracle_trial_v78,
    commit_multifraction_trial_v78,
    execution_document_v78,
    load_multifraction_checkpoint_v78,
    multifraction_checkpoint_v78,
    resolve_multifraction_suspended_state_v78,
    rollback_multifraction_trial_v78,
    validate_multifraction_suspended_state_v78,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-flow2dh-factcr-lifecycle-v83-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
METHOD_ID = "capacity.van_rijn_1993"
FORMULA_CODE = -1
ITAUCR = 1
FORMATION_UNIFORM_D50 = "uniform_d50_initialization"
FORMATION_SPATIAL_D50 = "spatial_d50_recalculation"
SPEED_EPSILON = 1.0e-6
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

SIGMA = np.asarray([
    -.0874, -.2472, -.3797, -.4897, -.5809, -.6565, -.7193, -.7713,
    -.8145, -.8503, -.8800, -.9046, -.9250, -.9419, -.9560, -.9676,
    -.9773, -.9854, -.9920, -.9975,
], dtype=np.float64)
THICKNESS = np.asarray([
    .1747, .1449, .1202, .0997, .0827, .0686, .0569, .0472, .0391,
    .0325, .0269, .0223, .0185, .0154, .0127, .0106, .0088, .0073,
    .0060, .0050,
], dtype=np.float64)

# This is an effect inventory, not a claim that all selectors are implemented
# in BRAIDS.  It follows the actual ``eqtran`` call signatures.
FACTCR_ROUTE_EFFECT = {
    -4: "consumed_only_when_itaucr_1_via_tram2",
    -2: "consumed_only_when_itaucr_1_via_tram2",
    -1: "consumed_only_when_itaucr_1_via_tram1",
    1: "not_consumed_by_tranb1",
    2: "not_consumed_by_tranb2",
    3: "disabled_ackers_white_before_any_transport_call",
    4: "not_consumed_by_tranb4",
    5: "not_consumed_by_tranb5",
    6: "disabled_bailard_before_any_transport_call",
    7: "not_consumed_by_tranb7",
    8: "disabled_van_rijn_ribberink_1994_before_any_transport_call",
    9: "disabled_original_silt_module_before_any_transport_call",
    10: "disabled_ashida_michiue_before_any_transport_call",
    11: "not_consumed_by_trab11",
    12: "not_consumed_by_trab12",
    13: "not_consumed_by_tran9t",
    14: "not_consumed_by_trab14",
    15: "user_dll_receives_no_taucr0_framework_argument_RP_TAUCR_is_taucrb",
    16: "not_consumed_by_trabwc",
    17: "not_consumed_by_trabwc2",
    18: "not_consumed_by_trabg",
    19: "not_consumed_by_trab19",
    20: "not_consumed_by_trab20",
    22: "not_consumed_by_asmita",
}

LOCKED_SOURCES = {
    "morphology_data": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_data/src/morphology_data_module.f90",
        "dd9ca54571498a067a118a19626d1779cd62c5cb558c391e9b8dd627c4e381ac",
    ),
    "rdmor": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_io/src/rdmor.f90",
        "ab4e3a1bdf1290e790412375ea5a91cddf9039670948307169c38194314e0d49",
    ),
    "initsedtra": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/initsedtra.f90",
        "43a3e6da535105d3ef28325c50fff665102bf08ffbcf0af7b99b23dd446428b4",
    ),
    "erosed": (
        "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
        "flow2d3d_kernel/src/compute_sediment/erosed.f90",
        "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
    ),
    "eqtran": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/eqtran.f90",
        "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
    ),
    "tram1": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tram1.f90",
        "33f5f60439c8b98facc7d5d1431e8f4a06fc37fe7b14d531a8fa9d9be6f74246",
    ),
    "tram2": (
        "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
        "morphology_kernel/src/tram2.f90",
        "527f2c01065e1da343100b10e2d7b791d32dfcbd801de70a4e9b71429fb3133a",
    ),
}


class FactCrV83Error(ValueError):
    """Fail-closed v83 identity, physics, transaction or ledger error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise FactCrV83Error("v83 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array(value: object, shape: tuple[int, ...], name: str, *,
           positive: bool = False, nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise FactCrV83Error(f"v83 {name} must be numeric with shape {shape}")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FactCrV83Error(f"v83 {name} must be finite")
    if positive and np.any(result <= 0.0):
        raise FactCrV83Error(f"v83 {name} must be positive")
    if nonnegative and np.any(result < 0.0):
        raise FactCrV83Error(f"v83 {name} must be nonnegative")
    result.setflags(write=False)
    return result


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise FactCrV83Error(f"v83 {name} must be finite numeric data")
    result = float(value)
    if positive and result <= 0.0:
        raise FactCrV83Error(f"v83 {name} must be positive")
    return result


def critical_shear_fields_v83(
    *, d50_m: object, solid_density_kg_m3: object,
    water_density_kg_m3: object, gravity_m_s2: object,
    kinematic_viscosity_m2_s: object, factcr: float,
    formula_code: int = FORMULA_CODE,
    formation_mode: str = FORMATION_UNIFORM_D50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``dstar, theta_cr, unfactored_taucr, effective_taucr``.

    The uniform-D50 initializer applies the alternate ``Dstar <= 4`` branch
    only to ``-2``; the spatial-D50 recalculation applies it to ``-2/-4``.
    This source asymmetry is retained explicitly.  Only ``formula_code=-1``
    is admitted to the coupled lifecycle below.
    """
    d50_raw = np.asarray(d50_m)
    rho_s_raw = np.asarray(solid_density_kg_m3)
    rho_raw = np.asarray(water_density_kg_m3)
    grav_raw = np.asarray(gravity_m_s2)
    nu_raw = np.asarray(kinematic_viscosity_m2_s)
    if d50_raw.ndim != 1 or rho_s_raw.shape != d50_raw.shape:
        raise FactCrV83Error("v83 grain arrays must have one identical dimension")
    if rho_raw.ndim != 1 or grav_raw.shape != rho_raw.shape or nu_raw.shape != rho_raw.shape:
        raise FactCrV83Error("v83 hydraulic arrays must have one identical dimension")
    ng, nc = d50_raw.size, rho_raw.size
    d50 = _array(d50_raw, (ng,), "D50", positive=True)
    rho_s = _array(rho_s_raw, (ng,), "solid density", positive=True)
    rho = _array(rho_raw, (nc,), "water density", positive=True)
    gravity = _array(grav_raw, (nc,), "gravity", positive=True)
    viscosity = _array(nu_raw, (nc,), "kinematic viscosity", positive=True)
    factor = _finite(factcr, "FactCr", positive=True)
    if type(formula_code) is not int or formula_code not in (-4, -2, -1):
        raise FactCrV83Error("v83 critical-shear equation supports only -1/-2/-4")
    if formation_mode not in (FORMATION_UNIFORM_D50, FORMATION_SPATIAL_D50):
        raise FactCrV83Error("v83 critical-shear formation mode differs")
    if np.any(rho_s[:, None] <= rho[None, :]):
        raise FactCrV83Error("v83 solid density must exceed water density")
    relative = (rho_s[:, None]-rho[None, :])/rho[None, :]
    dstar = d50[:, None]*(relative*gravity[None, :]/viscosity[None, :]**2)**0.3333
    theta = np.empty_like(dstar)
    first = dstar <= 4.0
    alternate_low_dstar = (
        formula_code == -2 or
        (formula_code == -4 and formation_mode == FORMATION_SPATIAL_D50))
    if alternate_low_dstar:
        theta[first] = 0.115/np.sqrt(dstar[first])
    else:
        theta[first] = 0.24/dstar[first]
    mask = (dstar > 4.0) & (dstar <= 10.0)
    theta[mask] = 0.14/dstar[mask]**0.64
    mask = (dstar > 10.0) & (dstar <= 20.0)
    theta[mask] = 0.04/dstar[mask]**0.10
    mask = (dstar > 20.0) & (dstar <= 150.0)
    theta[mask] = 0.013*dstar[mask]**0.29
    theta[dstar > 150.0] = 0.055
    base = ((rho_s[:, None]-rho[None, :])*gravity[None, :]*
            d50[:, None]*theta)
    effective = factor*base
    return tuple(np.ascontiguousarray(value) for value in
                 (dstar, theta, base, effective))


def verify_locked_sources_v83(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result: dict[str, str] = {}
    for symbol, (relative, expected) in LOCKED_SOURCES.items():
        path = root/relative
        if not path.is_file():
            raise FactCrV83Error("v83 locked source is absent: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise FactCrV83Error("v83 locked source hash differs: " + relative)
        result[symbol] = actual
    return result


@dataclass(frozen=True)
class FactCrExecutionV83:
    lifecycle_execution: MultifractionSuspendedExecutionV78
    factcr: float
    semantic_id: str
    fingerprint: str

    @property
    def role(self) -> str:
        return self.lifecycle_execution.role

    @property
    def fraction_count(self) -> int:
        return self.lifecycle_execution.fraction_count

    @property
    def cell_count(self) -> int:
        return self.lifecycle_execution.cell_count


def _execution_payload(value: FactCrExecutionV83) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "lifecycle_execution_v78": execution_document_v78(
            value.lifecycle_execution),
        "method_id": METHOD_ID,
        "formula_code": FORMULA_CODE,
        "itaucr": ITAUCR,
        "factcr": value.factcr,
        "critical_shear_formation_mode": FORMATION_UNIFORM_D50,
        "critical_shear_equation": (
            "FactCr*(rho_s-rho_w)*g*D50*theta_cr"),
        "factcr_application_count": 1,
        "route_effect_inventory": {
            str(key): FACTCR_ROUTE_EFFECT[key]
            for key in sorted(FACTCR_ROUTE_EFFECT)
        },
        "integrator": INTEGRATOR,
        "backend_role": value.role,
        "current_only": True,
        "pure_sand_mud_correction": 1.0,
        "fixed_grid": True,
        "automatic_cpu_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "bed_capacity_role": "diagnostic_not_an_exner_update",
        "producer_settling_velocity": (
            "explicit_authenticated_input_not_a_settling_law_qualification"),
        "exchange_shear_owner": (
            "accepted_hydraulic_cell_shear_input_not_fractional_producer_diagnostic"),
        "product_admission": False,
    }


def resolve_factcr_execution_v83(
    lifecycle_execution: MultifractionSuspendedExecutionV78, *, factcr: float,
) -> FactCrExecutionV83:
    execution_document_v78(lifecycle_execution)
    if lifecycle_execution.role not in (ROLE_CPU_ORACLE, ROLE_CUDA):
        raise FactCrV83Error("v83 requires an explicit CPU-oracle or CUDA role")
    factor = _finite(factcr, "FactCr", positive=True)
    placeholder = FactCrExecutionV83(
        lifecycle_execution, factor, SEMANTIC_ID, "")
    return FactCrExecutionV83(
        lifecycle_execution, factor, SEMANTIC_ID,
        _fingerprint(_execution_payload(placeholder)))


def execution_document_v83(value: FactCrExecutionV83) -> dict[str, object]:
    if not isinstance(value, FactCrExecutionV83):
        raise FactCrV83Error("v83 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise FactCrV83Error("v83 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


@dataclass(frozen=True)
class FactCrStateV83:
    execution_fingerprint: str
    lifecycle: MultifractionSuspendedStateV78
    accepted_factcr_chain_sha256: str
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.lifecycle.accepted_updates

    @property
    def time_s(self) -> float:
        return self.lifecycle.time_s


def _state_token(value: FactCrStateV83) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "lifecycle_state_sha256": value.lifecycle.state_sha256,
        "accepted_factcr_chain_sha256": value.accepted_factcr_chain_sha256,
    })


def _wrap_state(execution: FactCrExecutionV83,
                lifecycle: MultifractionSuspendedStateV78,
                chain: str) -> FactCrStateV83:
    execution_document_v83(execution)
    validate_multifraction_suspended_state_v78(
        lifecycle, execution.lifecycle_execution)
    if not isinstance(chain, str) or _SHA256.fullmatch(chain) is None:
        raise FactCrV83Error("v83 accepted FactCr chain digest differs")
    placeholder = FactCrStateV83(execution.fingerprint, lifecycle, chain, "")
    return FactCrStateV83(
        placeholder.execution_fingerprint, placeholder.lifecycle,
        placeholder.accepted_factcr_chain_sha256, _state_token(placeholder))


def resolve_factcr_state_v83(execution: FactCrExecutionV83,
                             **state_fields) -> FactCrStateV83:
    try:
        lifecycle = resolve_multifraction_suspended_state_v78(
            execution.lifecycle_execution, **state_fields)
    except (TypeError, ValueError) as error:
        raise FactCrV83Error("v83 initial lifecycle state is invalid") from error
    if lifecycle.accepted_updates != 0:
        raise FactCrV83Error("v83 fresh state must begin at update zero")
    chain = _fingerprint({
        "scope_id": SCOPE_ID,
        "execution_fingerprint": execution.fingerprint,
        "initial_lifecycle_state_sha256": lifecycle.state_sha256,
    })
    return _wrap_state(execution, lifecycle, chain)


def validate_factcr_state_v83(state: FactCrStateV83,
                              execution: FactCrExecutionV83) -> None:
    if (not isinstance(state, FactCrStateV83) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise FactCrV83Error("v83 state is stale, mutated, or execution-mismatched")
    _wrap_state(execution, state.lifecycle, state.accepted_factcr_chain_sha256)


def _vr1993_cpu(
    *, depth: np.ndarray, velocity: np.ndarray,
    observation_height: np.ndarray, water_density: np.ndarray,
    gravity: np.ndarray, kinematic_viscosity: np.ndarray,
    additional_shear_stress: np.ndarray, d50: np.ndarray, d90: np.ndarray,
    solid_density: np.ndarray, molecular_schmidt: np.ndarray,
    aks_factor: np.ndarray, maximum_volume_concentration: np.ndarray,
    minimum_depth: np.ndarray, settling_velocity: np.ndarray,
    roughness_z0: np.ndarray, dstar: np.ndarray,
    critical_shear_stress: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar CPU oracle matching the frozen v66 current-only producer."""
    ng, nc = dstar.shape
    raw_bed = np.zeros((2, ng, nc), dtype=np.float64)
    raw_conc = np.zeros((ng, nc), dtype=np.float64)
    diagnostics = np.zeros((6, ng, nc), dtype=np.float64)
    for grain in range(ng):
        for cell in range(nc):
            h = depth[cell]
            if h <= minimum_depth[grain]:
                continue
            ux, uy = velocity[:, cell]
            speed = math.hypot(ux, uy)
            z0 = roughness_z0[grain, cell]
            rc = 30.0*z0
            aks = min(max(aks_factor[grain]*rc, 0.01*h), 0.2*h)
            if observation_height[cell] <= 0.0 or 12.0*h <= max(rc, 3.0*d90[grain]):
                raise FactCrV83Error("v83 CPU Van Rijn geometry is invalid")
            ustar = speed*0.41/math.log(1.0+observation_height[cell]/z0)
            shear = water_density[cell]*ustar*ustar
            tauadd = additional_shear_stress[cell]
            if tauadd > 0.0:
                shear = math.hypot(shear, tauadd)
                ustar = math.sqrt(shear/water_density[cell])
            f1c = 0.24*math.log10(12.0*h/(3.0*d90[grain]))**-2.0
            fc = 0.24*math.log10(12.0*h/rc)**-2.0
            muc = f1c/fc
            ratio = muc*shear/critical_shear_stress[grain, cell]
            stage = max(0.0, ratio-1.0)
            caks = 0.0
            if stage > 1.0e-6:
                caks = min(
                    0.015*d50[grain]*stage**1.5/
                    (aks*dstar[grain, cell]**0.3),
                    maximum_volume_concentration[grain])
            beta = 1.0
            ws = settling_velocity[grain, cell]
            if ustar > 1.0e-6:
                beta = min(max(1.0+2.0*(ws/ustar)**2.0, 1.0), 1.5)
            epsmxc = 0.25*0.41*beta*ustar*h
            seddif = np.empty(20, dtype=np.float64)
            profile = np.zeros(20, dtype=np.float64)
            for k in range(20):
                zi = (1.0+SIGMA[k]-THICKNESS[k]/2.0)*h
                seddif[k] = (epsmxc if zi >= 0.5*h else
                              epsmxc*(1.0-(1.0-2.0*zi/h)**2.0))
            kmax = 0
            for k in range(18, -1, -1):
                zi = (1.0+SIGMA[k]-THICKNESS[k]/2.0)*h
                if zi >= aks:
                    kmax = k
                    break
            bakdif = kinematic_viscosity[cell]/molecular_schmidt[grain]
            if caks > 1.0e-6:
                dz = h*(1.0+SIGMA[kmax])-aks
                diffbt = max(seddif[kmax]+bakdif, 0.1*ws*dz)
                profile[kmax] = caks*solid_density[grain]/(1.0+dz*ws/diffbt)
                for k in range(kmax-1, -1, -1):
                    dz = h*(SIGMA[k]-SIGMA[k+1])
                    diffbt = max(seddif[k]+bakdif, 0.1*ws*dz)
                    profile[k] = profile[k+1]/(1.0+dz*ws/diffbt)
                for k in range(kmax+1, 20):
                    profile[k] = profile[k-1]
            avgu = 0.0
            avgcu = 0.0
            for k in range(20):
                z = (1.0+SIGMA[k])*h
                ulog = math.log(1.0+z/z0)
                avgu += ulog*THICKNESS[k]
                avgcu += ulog*profile[k]*THICKNESS[k]
            conc = avgcu/max(avgu, 1.0e-6)
            bed_stage = max(1.0e-10, ratio-1.0)
            skin_ustar = ustar*math.sqrt(muc)
            magnitude = (0.5*solid_density[grain]*d50[grain]*skin_ustar*
                         dstar[grain, cell]**-0.3*bed_stage)
            if speed > 1.0e-6:
                raw_bed[0, grain, cell] = magnitude*ux/speed
                raw_bed[1, grain, cell] = magnitude*uy/speed
            raw_conc[grain, cell] = conc
            diagnostics[:, grain, cell] = (
                ustar, muc, ratio, caks, aks, shear)
    return raw_bed, raw_conc, diagnostics


@dataclass(frozen=True)
class FactCrTrialV83:
    execution_fingerprint: str
    parent_state_sha256: str
    accepted_step_receipt_sha256: str
    candidate: FactCrStateV83
    lifecycle_trial: MultifractionSuspendedTrialV78
    dstar: np.ndarray
    critical_shields: np.ndarray
    unfactored_critical_shear_pa: np.ndarray
    effective_critical_shear_pa: np.ndarray
    raw_bed_capacity_kg_m_s: np.ndarray
    raw_equilibrium_concentration_kg_m3: np.ndarray
    bed_capacity_kg_m_s: np.ndarray
    lifecycle_capacity_kg_m_s: np.ndarray
    equilibrium_target_kg_m3: np.ndarray
    producer_diagnostics: np.ndarray
    factcr_application_count: np.ndarray
    equilibrium_reconstruction_max_abs_kg_m3: float
    backend: str
    trial_sha256: str


def _trial_token(value: FactCrTrialV83) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "accepted_step_receipt_sha256": value.accepted_step_receipt_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "lifecycle_trial_sha256": value.lifecycle_trial.trial_sha256,
        "arrays_sha256": _digest(
            value.dstar, value.critical_shields,
            value.unfactored_critical_shear_pa,
            value.effective_critical_shear_pa,
            value.raw_bed_capacity_kg_m_s,
            value.raw_equilibrium_concentration_kg_m3,
            value.bed_capacity_kg_m_s, value.lifecycle_capacity_kg_m_s,
            value.equilibrium_target_kg_m3, value.producer_diagnostics,
            value.factcr_application_count),
        "equilibrium_reconstruction_max_abs_kg_m3":
            value.equilibrium_reconstruction_max_abs_kg_m3,
        "backend": value.backend,
    })


def _build_trial(
    execution: FactCrExecutionV83, parent: FactCrStateV83,
    lifecycle_trial: MultifractionSuspendedTrialV78,
    accepted_step_receipt: Mapping[str, object], *, dstar: object,
    critical_shields: object, unfactored_taucr: object,
    effective_taucr: object, raw_bed: object, raw_conc: object,
    bed_capacity: object, lifecycle_capacity: object, target: object,
    diagnostics: object, backend: str,
) -> FactCrTrialV83:
    ng, nc = execution.fraction_count, execution.cell_count
    scalar_shape = (ng, nc)
    arrays = (
        _array(dstar, scalar_shape, "Dstar", positive=True),
        _array(critical_shields, scalar_shape, "critical Shields", positive=True),
        _array(unfactored_taucr, scalar_shape, "unfactored critical shear", positive=True),
        _array(effective_taucr, scalar_shape, "effective critical shear", positive=True),
        _array(raw_bed, (2, ng, nc), "raw bed capacity"),
        _array(raw_conc, scalar_shape, "raw equilibrium concentration", nonnegative=True),
        _array(bed_capacity, (2, ng, nc), "bed capacity"),
        _array(lifecycle_capacity, scalar_shape, "lifecycle capacity", nonnegative=True),
        _array(target, scalar_shape, "equilibrium target", nonnegative=True),
        _array(diagnostics, (6, ng, nc), "producer diagnostics", nonnegative=True),
    )
    expected_taucr = execution.factcr*arrays[2]
    if not np.array_equal(arrays[3], expected_taucr):
        raise FactCrV83Error("v83 FactCr was not applied exactly once")
    count = np.ones(scalar_shape, dtype=np.int32)
    count.setflags(write=False)
    error = float(np.max(np.abs(
        lifecycle_trial.equilibrium_concentration_kg_m3-arrays[8])))
    scale = max(1.0, float(np.max(np.abs(arrays[8]))))
    if error > 128.0*np.finfo(np.float64).eps*scale:
        raise FactCrV83Error("v83 lifecycle equilibrium differs from producer target")
    chain = _fingerprint({
        "previous": parent.accepted_factcr_chain_sha256,
        "accepted_step_receipt": accepted_step_receipt["fingerprint"],
        "lifecycle_trial_sha256": lifecycle_trial.trial_sha256,
        "factcr": execution.factcr,
        "effective_taucr_sha256": _digest(arrays[3]),
        "factcr_application_count_sha256": _digest(count),
    })
    candidate = _wrap_state(execution, lifecycle_trial.candidate, chain)
    placeholder = FactCrTrialV83(
        execution.fingerprint, parent.state_sha256,
        str(accepted_step_receipt["fingerprint"]), candidate,
        lifecycle_trial, *arrays, count, error, backend, "")
    return FactCrTrialV83(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _producer_inputs_cpu(execution: FactCrExecutionV83, *,
                         depth_after_m: object,
                         endpoint_momentum_m2_s: object,
                         observation_height_m: object,
                         water_density_kg_m3: object,
                         gravity_m_s2: object,
                         kinematic_viscosity_m2_s: object,
                         additional_shear_stress_pa: object,
                         sediment_d50_m: object, sediment_d90_m: object,
                         sediment_density_kg_m3: object,
                         molecular_schmidt: object, aks_factor: object,
                         maximum_volume_concentration: object,
                         minimum_depth_m: object, bed_factor: object,
                         suspended_factor: object,
                         settling_velocity_m_s: object,
                         roughness_z0_m: object,
                         surface_fraction: object,
                         frozen_uniform_threshold: object | None = None,
                         ) -> tuple:
    ng, nc = execution.fraction_count, execution.cell_count
    h = _array(depth_after_m, (nc,), "depth after", nonnegative=True)
    momentum = _array(endpoint_momentum_m2_s, (2, nc), "endpoint momentum")
    zvel = _array(observation_height_m, (nc,), "observation height", positive=True)
    rho = _array(water_density_kg_m3, (nc,), "water density", positive=True)
    gravity = _array(gravity_m_s2, (nc,), "gravity", positive=True)
    viscosity = _array(kinematic_viscosity_m2_s, (nc,), "viscosity", positive=True)
    tauadd = _array(additional_shear_stress_pa, (nc,), "additional shear", nonnegative=True)
    d50 = _array(sediment_d50_m, (ng,), "D50", positive=True)
    d90 = _array(sediment_d90_m, (ng,), "D90", positive=True)
    rho_s = _array(sediment_density_kg_m3, (ng,), "solid density", positive=True)
    schmidt = _array(molecular_schmidt, (ng,), "molecular Schmidt", positive=True)
    aks = _array(aks_factor, (ng,), "Aks factor", positive=True)
    camax = _array(maximum_volume_concentration, (ng,), "maximum volume concentration", positive=True)
    hmin = _array(minimum_depth_m, (ng,), "minimum depth", nonnegative=True)
    bedfac = _array(bed_factor, (ng,), "bed factor", nonnegative=True)
    susfac = _array(suspended_factor, (ng,), "suspended factor", nonnegative=True)
    ws = _array(settling_velocity_m_s, (ng, nc), "settling velocity", nonnegative=True)
    z0 = _array(roughness_z0_m, (ng, nc), "roughness z0", positive=True)
    fraction = _array(surface_fraction, (ng, nc), "surface fraction", nonnegative=True)
    if np.any(d90 < d50) or np.any(camax > 1.0) or np.any(fraction > 1.0):
        raise FactCrV83Error("v83 grain ordering, concentration, or fraction is invalid")
    profile = execution.lifecycle_execution.base_execution.profile
    # FLOW's transported temperature/salinity path supplies the current
    # cellwise water density to erosed/tram1.  D50, solid density and gravity
    # remain profile identities; water density is a live authenticated input.
    if (not np.array_equal(d50, np.asarray(profile.sediment_d50_m)) or
            not np.array_equal(rho_s,
                               np.asarray(profile.sediment_density_kg_m3)) or
            not np.all(gravity == profile.gravity_m_s2)):
        raise FactCrV83Error(
            "v83 producer grains, solid density, or gravity differ from the "
            "lifecycle")
    if not np.allclose(np.sum(fraction, axis=0), 1.0, rtol=0.0,
                       atol=32.0*np.finfo(np.float64).eps):
        raise FactCrV83Error("v83 surface fractions must sum to one per cell")
    velocity = np.zeros((2, nc), dtype=np.float64)
    wet = h > execution.lifecycle_execution.base_execution.dry_tolerance_m
    velocity[:, wet] = momentum[:, wet]/h[None, wet]
    if frozen_uniform_threshold is None:
        dstar, theta, base_taucr, effective_taucr = \
            critical_shear_fields_v83(
                d50_m=d50, solid_density_kg_m3=rho_s,
                water_density_kg_m3=rho, gravity_m_s2=gravity,
                kinematic_viscosity_m2_s=viscosity,
                factcr=execution.factcr)
    else:
        if (not isinstance(frozen_uniform_threshold, (tuple, list)) or
                len(frozen_uniform_threshold) != 4):
            raise FactCrV83Error(
                "v83 frozen uniform threshold inventory differs")
        frozen = tuple(
            _array(item, (ng,), name, positive=True)
            for item, name in zip(frozen_uniform_threshold, (
                "frozen Dstar", "frozen critical Shields",
                "frozen unfactored critical shear",
                "frozen effective critical shear")))
        if not np.array_equal(frozen[3], execution.factcr*frozen[2]):
            raise FactCrV83Error(
                "v83 frozen uniform threshold FactCr identity differs")
        dstar, theta, base_taucr, effective_taucr = tuple(
            np.ascontiguousarray(np.broadcast_to(
                item[:, None], (ng, nc))) for item in frozen)
    raw_bed, raw_conc, diagnostics = _vr1993_cpu(
        depth=h, velocity=velocity, observation_height=zvel,
        water_density=rho, gravity=gravity,
        kinematic_viscosity=viscosity,
        additional_shear_stress=tauadd, d50=d50, d90=d90,
        solid_density=rho_s, molecular_schmidt=schmidt, aks_factor=aks,
        maximum_volume_concentration=camax, minimum_depth=hmin,
        settling_velocity=ws, roughness_z0=z0, dstar=dstar,
        critical_shear_stress=effective_taucr)
    bed_capacity = fraction[None, :, :]*bedfac[None, :, None]*raw_bed
    base = execution.lifecycle_execution.base_execution
    sediment_active = wet & (h > base.profile.sediment_threshold_m)
    exchange_depth = np.where(
        sediment_active, np.maximum(base.profile.reference_depth_floor_m, h), 1.0)
    speed = np.hypot(velocity[0], velocity[1])
    capacity = susfac[:, None]*raw_conc*(speed[None, :]+SPEED_EPSILON)*exchange_depth[None, :]
    capacity[:, ~sediment_active] = 0.0
    target = fraction*susfac[:, None]*raw_conc
    target[:, ~sediment_active] = 0.0
    return (h, momentum, fraction, dstar, theta, base_taucr,
            effective_taucr, raw_bed, raw_conc, bed_capacity, capacity,
            target, diagnostics)


@dataclass(frozen=True)
class FactCrProducerReceiptV83:
    """Pure v83 constitutive output, before any v78 scalar lifecycle.

    V94 must share these rows with the already authoritative v92 scalar owner
    and the v93 Exner owner.  Keeping the pure producer in v83 avoids either
    duplicating Van Rijn 1993 or invoking v83's historical v78 child a second
    time.
    """

    execution_fingerprint: str
    accepted_endpoint_sha256: str
    active_fraction_sha256: str
    incoming_settling_velocity_sha256: str
    background_kinematic_viscosity_sha256: str
    dstar: np.ndarray
    critical_shields: np.ndarray
    unfactored_critical_shear_pa: np.ndarray
    effective_critical_shear_pa: np.ndarray
    raw_bed_capacity_kg_m_s: np.ndarray
    raw_equilibrium_concentration_kg_m3: np.ndarray
    bed_capacity_kg_m_s: np.ndarray
    lifecycle_capacity_kg_m_s: np.ndarray
    equilibrium_target_kg_m3: np.ndarray
    producer_diagnostics: np.ndarray
    factcr_application_count: np.ndarray
    scalar_lifecycle_application_count: np.ndarray
    backend: str
    receipt_sha256: str


def _pure_producer_token_v83(value: FactCrProducerReceiptV83) -> str:
    return _fingerprint({
        "scope_id": SCOPE_ID + ":pure-producer",
        "execution_fingerprint": value.execution_fingerprint,
        "accepted_endpoint_sha256": value.accepted_endpoint_sha256,
        "active_fraction_sha256": value.active_fraction_sha256,
        "incoming_settling_velocity_sha256":
            value.incoming_settling_velocity_sha256,
        "background_kinematic_viscosity_sha256":
            value.background_kinematic_viscosity_sha256,
        "arrays_sha256": _digest(
            value.dstar, value.critical_shields,
            value.unfactored_critical_shear_pa,
            value.effective_critical_shear_pa,
            value.raw_bed_capacity_kg_m_s,
            value.raw_equilibrium_concentration_kg_m3,
            value.bed_capacity_kg_m_s,
            value.lifecycle_capacity_kg_m_s,
            value.equilibrium_target_kg_m3,
            value.producer_diagnostics,
            value.factcr_application_count,
            value.scalar_lifecycle_application_count),
        "backend": value.backend,
        "scalar_lifecycle_owner": "external_v92_exactly_once",
    })


def make_factcr_producer_receipt_v83(
    execution: FactCrExecutionV83, *, accepted_endpoint: object,
    active_fraction: object, incoming_settling_velocity_m_s: object,
    background_kinematic_viscosity_m2_s: object,
    dstar: object, critical_shields: object,
    unfactored_critical_shear_pa: object,
    effective_critical_shear_pa: object,
    raw_bed_capacity_kg_m_s: object,
    raw_equilibrium_concentration_kg_m3: object,
    bed_capacity_kg_m_s: object,
    lifecycle_capacity_kg_m_s: object,
    equilibrium_target_kg_m3: object,
    producer_diagnostics: object, backend: str,
) -> FactCrProducerReceiptV83:
    """Seal pure CPU or actual-CUDA v83 rows for the v94 shared owner.

    Numerical production stays backend-specific.  This helper authenticates
    the common receipt ABI and requires its provenance to match the resolved
    v83 role; it is not a CPU numerical fallback or a loose-row lifecycle.
    """
    execution_document_v83(execution)
    if backend != execution.role or backend not in (ROLE_CPU_ORACLE, ROLE_CUDA):
        raise FactCrV83Error(
            "v83 pure producer backend differs from its resolved role")
    ng, nc = execution.fraction_count, execution.cell_count
    endpoint = _array(accepted_endpoint, (3, nc), "accepted endpoint")
    fraction = _array(
        active_fraction, (ng, nc), "active fraction", nonnegative=True)
    settling = _array(
        incoming_settling_velocity_m_s, (ng, nc),
        "settling velocity", nonnegative=True)
    background_viscosity = _array(
        background_kinematic_viscosity_m2_s, (nc,),
        "background kinematic viscosity", positive=True)
    arrays = (
        _array(dstar, (ng, nc), "Dstar", positive=True),
        _array(critical_shields, (ng, nc), "critical Shields", positive=True),
        _array(unfactored_critical_shear_pa, (ng, nc),
               "unfactored critical shear", positive=True),
        _array(effective_critical_shear_pa, (ng, nc),
               "effective critical shear", positive=True),
        _array(raw_bed_capacity_kg_m_s, (2, ng, nc), "raw bed capacity"),
        _array(raw_equilibrium_concentration_kg_m3, (ng, nc),
               "raw equilibrium concentration", nonnegative=True),
        _array(bed_capacity_kg_m_s, (2, ng, nc), "bed capacity"),
        _array(lifecycle_capacity_kg_m_s, (ng, nc),
               "lifecycle capacity", nonnegative=True),
        _array(equilibrium_target_kg_m3, (ng, nc),
               "equilibrium target", nonnegative=True),
        _array(producer_diagnostics, (6, ng, nc),
               "producer diagnostics", nonnegative=True),
    )
    if (np.any(fraction > 1.0) or
            not np.allclose(np.sum(fraction, axis=0), 1.0, rtol=0.0,
                            atol=32.0*np.finfo(np.float64).eps) or
            not np.array_equal(arrays[3], execution.factcr*arrays[2])):
        raise FactCrV83Error(
            "v83 pure producer fraction or FactCr identity differs")
    ones = np.ones((ng, nc), dtype=np.int32)
    zeros = np.zeros((ng, nc), dtype=np.int32)
    ones.setflags(write=False)
    zeros.setflags(write=False)
    placeholder = FactCrProducerReceiptV83(
        execution.fingerprint, _digest(endpoint), _digest(fraction),
        _digest(settling), _digest(background_viscosity),
        *arrays, ones, zeros, backend, "")
    return FactCrProducerReceiptV83(
        **{**placeholder.__dict__,
           "receipt_sha256": _pure_producer_token_v83(placeholder)})


def produce_factcr_rows_cpu_v83(
    execution: FactCrExecutionV83, *,
    depth_after_m: object, endpoint_momentum_m2_s: object,
    observation_height_m: object, water_density_kg_m3: object,
    gravity_m_s2: object, kinematic_viscosity_m2_s: object,
    additional_shear_stress_pa: object, sediment_d50_m: object,
    sediment_d90_m: object, sediment_density_kg_m3: object,
    molecular_schmidt: object, aks_factor: object,
    maximum_volume_concentration: object, minimum_depth_m: object,
    bed_factor: object, suspended_factor: object,
    settling_velocity_m_s: object, roughness_z0_m: object,
    surface_fraction: object,
    frozen_uniform_threshold: object | None = None,
) -> FactCrProducerReceiptV83:
    """Run the accepted v83 CPU constitutive producer without scalar work."""
    execution_document_v83(execution)
    if execution.role != ROLE_CPU_ORACLE:
        raise FactCrV83Error("v83 pure CPU producer requires isolated_cpu_oracle")
    produced = _producer_inputs_cpu(
        execution, depth_after_m=depth_after_m,
        endpoint_momentum_m2_s=endpoint_momentum_m2_s,
        observation_height_m=observation_height_m,
        water_density_kg_m3=water_density_kg_m3,
        gravity_m_s2=gravity_m_s2,
        kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
        additional_shear_stress_pa=additional_shear_stress_pa,
        sediment_d50_m=sediment_d50_m,
        sediment_d90_m=sediment_d90_m,
        sediment_density_kg_m3=sediment_density_kg_m3,
        molecular_schmidt=molecular_schmidt, aks_factor=aks_factor,
        maximum_volume_concentration=maximum_volume_concentration,
        minimum_depth_m=minimum_depth_m, bed_factor=bed_factor,
        suspended_factor=suspended_factor,
        settling_velocity_m_s=settling_velocity_m_s,
        roughness_z0_m=roughness_z0_m, surface_fraction=surface_fraction,
        frozen_uniform_threshold=frozen_uniform_threshold)
    (depth, momentum, fraction, dstar, theta, base_taucr,
     effective_taucr, raw_bed, raw_conc, bed_capacity, capacity,
     target, diagnostics) = produced
    return make_factcr_producer_receipt_v83(
        execution, accepted_endpoint=np.vstack((depth, momentum)),
        active_fraction=fraction,
        incoming_settling_velocity_m_s=settling_velocity_m_s,
        background_kinematic_viscosity_m2_s=
            kinematic_viscosity_m2_s,
        dstar=dstar, critical_shields=theta,
        unfactored_critical_shear_pa=base_taucr,
        effective_critical_shear_pa=effective_taucr,
        raw_bed_capacity_kg_m_s=raw_bed,
        raw_equilibrium_concentration_kg_m3=raw_conc,
        bed_capacity_kg_m_s=bed_capacity,
        lifecycle_capacity_kg_m_s=capacity,
        equilibrium_target_kg_m3=target,
        producer_diagnostics=diagnostics, backend=ROLE_CPU_ORACLE)


def validate_factcr_producer_receipt_v83(
    value: FactCrProducerReceiptV83, execution: FactCrExecutionV83,
) -> None:
    execution_document_v83(execution)
    if (not isinstance(value, FactCrProducerReceiptV83) or
            value.execution_fingerprint != execution.fingerprint or
            value.backend != execution.role or
            value.backend not in (ROLE_CPU_ORACLE, ROLE_CUDA) or
            value.receipt_sha256 != _pure_producer_token_v83(value) or
            np.any(value.factcr_application_count != 1) or
            np.any(value.scalar_lifecycle_application_count != 0)):
        raise FactCrV83Error(
            "v83 pure producer receipt is stale, mutated, or duplicated")


def begin_factcr_cpu_trial_v83(
    execution: FactCrExecutionV83, parent: FactCrStateV83, *,
    accepted_step_receipt: Mapping[str, object], depth_before_m: object,
    depth_after_m: object, accepted_water_flux_m2_s: object,
    reconstructed_left_concentration_kg_m3: object,
    reconstructed_right_concentration_kg_m3: object,
    prescribed_boundary_concentration_kg_m3: object,
    accepted_face_exchange_area_m2: object, transport_face_active: object,
    endpoint_momentum_m2_s: object, shear_velocity_m_s: object,
    hydro_constant: object,
    observation_height_m: object,
    water_density_kg_m3: object, gravity_m_s2: object,
    kinematic_viscosity_m2_s: object,
    additional_shear_stress_pa: object, sediment_d50_m: object,
    sediment_d90_m: object, sediment_density_kg_m3: object,
    molecular_schmidt: object, aks_factor: object,
    maximum_volume_concentration: object, minimum_depth_m: object,
    bed_factor: object, suspended_factor: object,
    settling_velocity_m_s: object, roughness_z0_m: object,
    surface_fraction: object, dt_s: float, cancelled: bool = False,
) -> FactCrTrialV83:
    if execution.role != ROLE_CPU_ORACLE:
        raise FactCrV83Error("v83 CPU path requires isolated_cpu_oracle")
    validate_factcr_state_v83(parent, execution)
    if cancelled:
        raise FactCrV83Error("v83 update cancelled before candidate publication")
    produced = _producer_inputs_cpu(
        execution, depth_after_m=depth_after_m,
        endpoint_momentum_m2_s=endpoint_momentum_m2_s,
        observation_height_m=observation_height_m,
        water_density_kg_m3=water_density_kg_m3,
        gravity_m_s2=gravity_m_s2,
        kinematic_viscosity_m2_s=kinematic_viscosity_m2_s,
        additional_shear_stress_pa=additional_shear_stress_pa,
        sediment_d50_m=sediment_d50_m, sediment_d90_m=sediment_d90_m,
        sediment_density_kg_m3=sediment_density_kg_m3,
        molecular_schmidt=molecular_schmidt, aks_factor=aks_factor,
        maximum_volume_concentration=maximum_volume_concentration,
        minimum_depth_m=minimum_depth_m, bed_factor=bed_factor,
        suspended_factor=suspended_factor,
        settling_velocity_m_s=settling_velocity_m_s,
        roughness_z0_m=roughness_z0_m, surface_fraction=surface_fraction)
    (h1, momentum, fraction, dstar, theta, base_taucr, effective_taucr,
     raw_bed, raw_conc, bed_capacity, capacity, target, diagnostics) = produced
    cell_shear = _array(
        shear_velocity_m_s, (execution.cell_count,),
        "accepted hydraulic cell shear velocity", nonnegative=True)
    try:
        lifecycle_trial = begin_multifraction_cpu_oracle_trial_v78(
            execution.lifecycle_execution, parent.lifecycle,
            accepted_step_receipt=accepted_step_receipt,
            depth_before_m=depth_before_m, depth_after_m=h1,
            accepted_water_flux_m2_s=accepted_water_flux_m2_s,
            reconstructed_left_concentration_kg_m3=
                reconstructed_left_concentration_kg_m3,
            reconstructed_right_concentration_kg_m3=
                reconstructed_right_concentration_kg_m3,
            prescribed_boundary_concentration_kg_m3=
                prescribed_boundary_concentration_kg_m3,
            accepted_face_exchange_area_m2=accepted_face_exchange_area_m2,
            transport_face_active=transport_face_active,
            endpoint_momentum_m2_s=momentum,
            shear_velocity_m_s=cell_shear,
            suspended_transport_capacity_kg_m_s=capacity,
            surface_fraction=fraction, hydro_constant=hydro_constant,
            dt_s=dt_s)
    except (TypeError, ValueError) as error:
        raise FactCrV83Error("v83 CPU lifecycle rejected the trial") from error
    return _build_trial(
        execution, parent, lifecycle_trial, accepted_step_receipt,
        dstar=dstar, critical_shields=theta,
        unfactored_taucr=base_taucr, effective_taucr=effective_taucr,
        raw_bed=raw_bed, raw_conc=raw_conc, bed_capacity=bed_capacity,
        lifecycle_capacity=capacity, target=target,
        diagnostics=diagnostics, backend=ROLE_CPU_ORACLE)


def _validate_trial(execution: FactCrExecutionV83, parent: FactCrStateV83,
                    trial: FactCrTrialV83) -> None:
    validate_factcr_state_v83(parent, execution)
    if (not isinstance(trial, FactCrTrialV83) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise FactCrV83Error("v83 trial is stale, mutated, or parent-mismatched")
    validate_factcr_state_v83(trial.candidate, execution)


def commit_factcr_trial_v83(execution: FactCrExecutionV83,
                            parent: FactCrStateV83,
                            trial: FactCrTrialV83) -> FactCrStateV83:
    _validate_trial(execution, parent, trial)
    committed = commit_multifraction_trial_v78(
        execution.lifecycle_execution, parent.lifecycle,
        trial.lifecycle_trial)
    if committed.state_sha256 != trial.candidate.lifecycle.state_sha256:
        raise FactCrV83Error("v83 committed lifecycle differs from candidate")
    return trial.candidate


def rollback_factcr_trial_v83(execution: FactCrExecutionV83,
                              parent: FactCrStateV83,
                              trial: FactCrTrialV83) -> FactCrStateV83:
    _validate_trial(execution, parent, trial)
    base = rollback_multifraction_trial_v78(
        execution.lifecycle_execution, parent.lifecycle,
        trial.lifecycle_trial)
    if base is not parent.lifecycle:
        raise FactCrV83Error("v83 rollback changed the accepted lifecycle")
    return parent


def cancel_factcr_trial_v83(execution: FactCrExecutionV83,
                            parent: FactCrStateV83,
                            trial: FactCrTrialV83) -> FactCrStateV83:
    return rollback_factcr_trial_v83(execution, parent, trial)


def factcr_checkpoint_v83(execution: FactCrExecutionV83,
                          state: FactCrStateV83) -> dict[str, object]:
    validate_factcr_state_v83(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "method_id": METHOD_ID,
        "formula_code": FORMULA_CODE,
        "itaucr": ITAUCR,
        "factcr": execution.factcr,
        "integrator": INTEGRATOR,
        "accepted_factcr_chain_sha256": state.accepted_factcr_chain_sha256,
        "lifecycle_checkpoint_v78": multifraction_checkpoint_v78(
            execution.lifecycle_execution, state.lifecycle),
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_factcr_checkpoint_v83(execution: FactCrExecutionV83,
                               document: Mapping[str, object]) -> FactCrStateV83:
    expected = {
        "schema", "scope_id", "semantic_id", "execution_fingerprint",
        "method_id", "formula_code", "itaucr", "factcr", "integrator",
        "accepted_factcr_chain_sha256", "lifecycle_checkpoint_v78",
        "state_sha256", "fingerprint",
    }
    if not isinstance(document, Mapping) or set(document) != expected:
        raise FactCrV83Error("v83 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["method_id"] != METHOD_ID or
            document["formula_code"] != FORMULA_CODE or
            document["itaucr"] != ITAUCR or
            document["factcr"] != execution.factcr or
            document["integrator"] != INTEGRATOR):
        raise FactCrV83Error("v83 checkpoint identity differs")
    try:
        lifecycle = load_multifraction_checkpoint_v78(
            execution.lifecycle_execution,
            document["lifecycle_checkpoint_v78"])
        state = _wrap_state(
            execution, lifecycle,
            str(document["accepted_factcr_chain_sha256"]))
    except (TypeError, ValueError) as error:
        raise FactCrV83Error("v83 checkpoint lifecycle differs") from error
    if state.state_sha256 != document["state_sha256"]:
        raise FactCrV83Error("v83 checkpoint state digest differs")
    return state


def method_identity_v83() -> dict[str, object]:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "integrator": INTEGRATOR,
        "method_id": METHOD_ID,
        "formula_code": FORMULA_CODE,
        "itaucr": ITAUCR,
        "factcr_exactly_once": True,
        "actual_route_effect_inventory": dict(FACTCR_ROUTE_EFFECT),
        "coupled_scope": "current_only_pure_sand_TraFrm_minus_1_fixed_grid",
        "equation_only_scope": "TraFrm_minus_2_and_minus_4",
        "equation_formation_modes": (
            FORMATION_UNIFORM_D50, FORMATION_SPATIAL_D50),
        "automatic_cpu_fallback": False,
        "accepted_update_limit": MAX_ACCEPTED_UPDATES,
        "bed_capacity_role": "diagnostic_not_exner",
        "long_run_claimed": False,
        "product_admission": False,
        "delft3d_source_modified": False,
    }


__all__ = [
    "CHECKPOINT_SCOPE_ID", "FACTCR_ROUTE_EFFECT", "FORMULA_CODE",
    "FORMATION_SPATIAL_D50", "FORMATION_UNIFORM_D50", "ITAUCR",
    "FactCrExecutionV83", "FactCrProducerReceiptV83", "FactCrStateV83",
    "FactCrTrialV83",
    "FactCrV83Error", "INTEGRATOR", "LOCKED_SOURCES", "METHOD_ID",
    "SCHEMA", "SCOPE_ID", "SEMANTIC_ID", "SPEED_EPSILON",
    "begin_factcr_cpu_trial_v83", "cancel_factcr_trial_v83",
    "commit_factcr_trial_v83", "critical_shear_fields_v83",
    "execution_document_v83", "factcr_checkpoint_v83",
    "load_factcr_checkpoint_v83", "method_identity_v83",
    "make_factcr_producer_receipt_v83", "produce_factcr_rows_cpu_v83",
    "resolve_factcr_execution_v83", "resolve_factcr_state_v83",
    "rollback_factcr_trial_v83", "validate_factcr_producer_receipt_v83",
    "validate_factcr_state_v83",
    "verify_locked_sources_v83",
]
