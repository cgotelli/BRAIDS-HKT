"""Bounded Q4/v88 dynamic fluid properties for noncohesive settling.

This additive component follows the locked Delft3D4 FLOW 2DH *fluid-property*
call order: ``fallve`` uses the *accepted* temperature, salinity and water
density, then ``dens`` rebuilds water density from the newly accepted
constituent fields for the following update.  With neither constituent active
FLOW retains ``Rhow``;
with either constituent active it evaluates the selected Eckart, UNESCO or
NaCl equation of state using the local value and the other constituent's
equilibrium constant.  ``eqsettle`` then uses the local
``4e-5/(20+temperature)`` molecular viscosity.

The caller remains responsible for transporting temperature and salinity.
V88 transaction-hashes caller-supplied endpoint cell fields and owns the one accepted
SettleFrm=2/Galappatti result layered on the v82 FCT/diffusion lifecycle.  It
does not yet bind those fields to a typed scalar-transport receipt and does
not claim a production scalar-transport or hydraulic-density coupling.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .flow2dh_exchange import advance_flow2dh_equilibrium_exchange
from .flow2dh_settling import settlefrm2_noncohesive
from .multifraction_suspended_lifecycle_v78 import (
    ROLE_CPU_ORACLE,
    ROLE_CUDA,
    build_multifraction_trial_v78,
)
from .suspended_lifecycle_live_v76 import resolve_suspended_lifecycle_state_v76
from .suspended_support_closures_v82 import (
    MAX_UPDATES,
    SuspendedSupportExecutionV82,
    SuspendedSupportStateV82,
    SuspendedSupportTrialV82,
    _advected_mass_from_flux,
    _wrap_trial as _wrap_trial_v82,
    begin_suspended_support_cpu_trial_v82,
    commit_suspended_support_trial_v82,
    execution_document_v82,
    load_suspended_support_checkpoint_v82,
    resolve_suspended_support_state_v82,
    rollback_suspended_support_trial_v82,
    suspended_support_checkpoint_v82,
    validate_suspended_support_state_v82,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-dynamic-fluid-settling-v88-v1"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
SEMANTIC_ID = hashlib.sha256(SCOPE_ID.encode("ascii")).hexdigest()
INTEGRATOR = "implicit_fwave_midpoint"
MODE_CONSTANT = "equilibrium_constant"
MODE_CELL = "accepted_cell_field"
MODES = (MODE_CONSTANT, MODE_CELL)
EOS_ECKART = "eckart"
EOS_UNESCO = "unesco"
EOS_NACL = "nacl"
EOS_CHOICES = (EOS_ECKART, EOS_UNESCO, EOS_NACL)
EOS_CODE = {EOS_ECKART: 0, EOS_UNESCO: 1, EOS_NACL: 2}
SETTLING_VARIANT = "dss"


class DynamicFluidSettlingV88Error(ValueError):
    """Fail-closed v88 configuration, field, transaction or receipt error."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise DynamicFluidSettlingV88Error(
            "v88 value is not finite canonical JSON") from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_scalar(value: object, name: str, *, lower: float | None = None,
                   positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value))):
        raise DynamicFluidSettlingV88Error(f"v88 {name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise DynamicFluidSettlingV88Error(f"v88 {name} must be positive")
    if lower is not None and result < lower:
        raise DynamicFluidSettlingV88Error(
            f"v88 {name} must be at least {lower}")
    return result


def _frozen_field(value: object, size: int, name: str, *,
                  nonnegative: bool = False,
                  temperature: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "fiu":
        raise DynamicFluidSettlingV88Error(
            f"v88 {name} must be a numeric cell field with shape ({size},)")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise DynamicFluidSettlingV88Error(f"v88 {name} must be finite")
    if nonnegative and np.any(result < 0.0):
        raise DynamicFluidSettlingV88Error(
            f"v88 {name} must be nonnegative")
    if temperature and np.any(result <= -20.0):
        raise DynamicFluidSettlingV88Error(
            "v88 temperature must exceed -20 C for eqsettle viscosity")
    result.setflags(write=False)
    return result


def water_density_eckart_v88(temperature_c: object,
                             salinity_ppt: object) -> np.ndarray:
    """Locked ``dens_eck.f90`` density value (derivatives omitted)."""
    t, s = np.broadcast_arrays(np.asarray(temperature_c, dtype=np.float64),
                               np.asarray(salinity_ppt, dtype=np.float64))
    cp0 = 5890.0 + 38.0*t - 0.3750*t*t
    clam = 1779.5 + 11.25*t - 0.0745*t*t
    clam0 = 3.8 + 0.01*t
    cp1 = cp0 + 3.0*s
    return 1000.0*cp1/(0.698*cp1 + clam - clam0*s)


def water_density_unesco_v88(temperature_c: object,
                             salinity_ppt: object) -> np.ndarray:
    """Locked ``dens_unes.f90`` density value (derivatives omitted)."""
    t, s = np.broadcast_arrays(np.asarray(temperature_c, dtype=np.float64),
                               np.asarray(salinity_ppt, dtype=np.float64))
    sq = np.sqrt(np.maximum(0.0, s))
    rhwa = (999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 +
            1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5)
    asal = (8.24493e-1 - 4.0899e-3*t + 7.6438e-5*t**2 -
            8.2467e-7*t**3 + 5.3875e-9*t**4)
    bsal = -5.72466e-3 + 1.0227e-4*t - 1.6546e-6*t**2
    return rhwa + (asal + bsal*sq + 4.8314e-4*s)*s


def water_density_nacl_v88(temperature_c: object,
                           salinity_ppt: object) -> np.ndarray:
    """Locked ``dens_nacl.f90`` density value (derivatives omitted)."""
    t, s = np.broadcast_arrays(np.asarray(temperature_c, dtype=np.float64),
                               np.asarray(salinity_ppt, dtype=np.float64))
    return (999.904 + 4.8292e-2*t - 7.2312e-3*t**2 +
            2.9963e-5*t**3 + 7.6427e-1*s - 3.1490e-3*s*t +
            3.1273e-5*s*t**2)


def resolve_water_density_v88(
    temperature_c: object, salinity_ppt: object, *, eos: str,
    constituents_active: bool, reference_water_density_kg_m3: float,
) -> np.ndarray:
    """Resolve the exact 2DH ``dens`` branch admitted by v88.

    FLOW does *not* evaluate an EOS when both salinity and temperature are
    equilibrium constants: it assigns ``rhow``.  If either constituent is
    active, the selected EOS sees local/equilibrium values for both.
    """
    t, s = np.broadcast_arrays(np.asarray(temperature_c, dtype=np.float64),
                               np.asarray(salinity_ppt, dtype=np.float64))
    if (not np.isfinite(t).all() or not np.isfinite(s).all() or
            np.any(t <= -20.0) or np.any(s < 0.0)):
        raise DynamicFluidSettlingV88Error(
            "v88 temperature/salinity fields are outside admitted bounds")
    reference = _finite_scalar(reference_water_density_kg_m3,
                               "reference water density", positive=True)
    if not isinstance(constituents_active, bool):
        raise DynamicFluidSettlingV88Error(
            "v88 constituent-active flag must be Boolean")
    if eos not in EOS_CHOICES:
        raise DynamicFluidSettlingV88Error(
            "v88 EOS must be eckart, unesco, or nacl")
    if not constituents_active:
        result = np.full(t.shape, reference, dtype=np.float64)
    elif eos == EOS_ECKART:
        result = water_density_eckart_v88(t, s)
    elif eos == EOS_UNESCO:
        result = water_density_unesco_v88(t, s)
    else:
        result = water_density_nacl_v88(t, s)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise DynamicFluidSettlingV88Error(
            "v88 selected EOS produced invalid water density")
    return np.ascontiguousarray(result, dtype=np.float64)


@dataclass(frozen=True)
class DynamicFluidExecutionV88:
    base_execution: SuspendedSupportExecutionV82
    temperature_mode: str
    salinity_mode: str
    equilibrium_temperature_c: float
    equilibrium_salinity_ppt: float
    density_eos: str
    settling_variant: str
    semantic_id: str
    fingerprint: str

    @property
    def role(self) -> str:
        return self.base_execution.role

    @property
    def cell_count(self) -> int:
        return self.base_execution.cell_count

    @property
    def fraction_count(self) -> int:
        return self.base_execution.fraction_count

    @property
    def face_count(self) -> int:
        return self.base_execution.face_count

    @property
    def boundary_count(self) -> int:
        return self.base_execution.boundary_count

    @property
    def constituents_active(self) -> bool:
        return (self.temperature_mode == MODE_CELL or
                self.salinity_mode == MODE_CELL)


def _execution_payload(value: DynamicFluidExecutionV88) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": value.semantic_id,
        "base_v82_execution": execution_document_v82(value.base_execution),
        "integrator": INTEGRATOR,
        "kmax": 1,
        "temperature_mode": value.temperature_mode,
        "salinity_mode": value.salinity_mode,
        "equilibrium_temperature_c": value.equilibrium_temperature_c,
        "equilibrium_salinity_ppt": value.equilibrium_salinity_ppt,
        "constituents_active": value.constituents_active,
        "density_eos": value.density_eos,
        "constant_density_branch": "rhow_when_no_T_or_S_constituent",
        "settling_variant": value.settling_variant,
        "molecular_viscosity": "4e-5/(20+local_temperature_c)",
        "bounded_input_ranges": {
            "temperature_c": "greater_than_-20_for_positive_viscosity",
            "salinity_ppt": "nonnegative",
            "relation_to_locked_FLOW":
                "deliberately_stricter_component_validation_not_range_parity",
        },
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "accepted_owner_count": {
            "fct": 1, "horizontal_diffusion": 1,
            "settling_exchange": 1, "wetdry_cleanup": 1,
        },
        "endpoint_field_provenance":
            "caller_supplied_transaction_hashed; typed_scalar_transport_"
            "receipt_open",
        "hindering_concentration_timing":
            "BRAIDS_post_FCT_diffusion_split_state; locked_FLOW_uses_parent_"
            "r0_before_tritra",
        "locked_delft3d_source_read_only": True,
    }


def resolve_dynamic_fluid_execution_v88(
    base_execution: SuspendedSupportExecutionV82, *,
    temperature_mode: str = MODE_CONSTANT,
    salinity_mode: str = MODE_CONSTANT,
    equilibrium_temperature_c: float = 15.0,
    equilibrium_salinity_ppt: float = 0.0,
    density_eos: str = EOS_UNESCO,
    settling_variant: str = SETTLING_VARIANT,
    kmax: int = 1,
) -> DynamicFluidExecutionV88:
    execution_document_v82(base_execution)
    if temperature_mode not in MODES or salinity_mode not in MODES:
        raise DynamicFluidSettlingV88Error(
            "v88 field modes must be equilibrium_constant or accepted_cell_field")
    temperature = _finite_scalar(
        equilibrium_temperature_c, "equilibrium temperature")
    if temperature <= -20.0:
        raise DynamicFluidSettlingV88Error(
            "v88 equilibrium temperature must exceed -20 C")
    salinity = _finite_scalar(
        equilibrium_salinity_ppt, "equilibrium salinity", lower=0.0)
    if density_eos not in EOS_CHOICES:
        raise DynamicFluidSettlingV88Error(
            "v88 EOS must be eckart, unesco, or nacl")
    if settling_variant != SETTLING_VARIANT:
        raise DynamicFluidSettlingV88Error(
            "v88 admits ordinary noncohesive DSS/SettleFrm=2 only; "
            "DSS_2004, SalTemCon, floc and DLL/fines branches are unsupported")
    if isinstance(kmax, bool) or type(kmax) is not int or kmax != 1:
        raise DynamicFluidSettlingV88Error(
            "v88 is FLOW 2DH KMAX=1; 3D/vertical branches are unsupported")
    profile = base_execution.base_execution.base_execution.profile
    if any(kind != "sand" for kind in profile.sediment_fraction_classes):
        raise DynamicFluidSettlingV88Error(
            "v88 admits noncohesive sand/gravel classes only")
    placeholder = DynamicFluidExecutionV88(
        base_execution, temperature_mode, salinity_mode, temperature,
        salinity, density_eos, settling_variant, SEMANTIC_ID, "")
    return DynamicFluidExecutionV88(
        **{**placeholder.__dict__,
           "fingerprint": _fingerprint(_execution_payload(placeholder))})


def execution_document_v88(value: DynamicFluidExecutionV88) -> dict:
    if not isinstance(value, DynamicFluidExecutionV88):
        raise DynamicFluidSettlingV88Error("v88 execution object differs")
    payload = _execution_payload(value)
    if value.fingerprint != _fingerprint(payload):
        raise DynamicFluidSettlingV88Error(
            "v88 execution fingerprint differs")
    return {**payload, "fingerprint": value.fingerprint}


def _resolved_field(execution: DynamicFluidExecutionV88, value: object | None,
                    *, temperature: bool) -> np.ndarray:
    mode = (execution.temperature_mode if temperature else
            execution.salinity_mode)
    constant = (execution.equilibrium_temperature_c if temperature else
                execution.equilibrium_salinity_ppt)
    name = "temperature" if temperature else "salinity"
    if mode == MODE_CONSTANT:
        if value is not None:
            raise DynamicFluidSettlingV88Error(
                f"v88 {name} cell field supplied for equilibrium_constant mode")
        result = np.full(execution.cell_count, constant, dtype=np.float64)
        result.setflags(write=False)
        return result
    if value is None:
        raise DynamicFluidSettlingV88Error(
            f"v88 accepted {name} cell field is required")
    return _frozen_field(value, execution.cell_count, name,
                         nonnegative=not temperature,
                         temperature=temperature)


def _properties(execution: DynamicFluidExecutionV88,
                temperature: np.ndarray,
                salinity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = execution.base_execution.base_execution.base_execution.profile
    density = resolve_water_density_v88(
        temperature, salinity, eos=execution.density_eos,
        constituents_active=execution.constituents_active,
        reference_water_density_kg_m3=profile.water_density_kg_m3)
    viscosity = np.ascontiguousarray(4.0e-5/(20.0+temperature),
                                     dtype=np.float64)
    if not np.isfinite(viscosity).all() or np.any(viscosity <= 0.0):
        raise DynamicFluidSettlingV88Error(
            "v88 eqsettle viscosity is invalid")
    density.setflags(write=False)
    viscosity.setflags(write=False)
    return density, viscosity


@dataclass(frozen=True)
class DynamicFluidStateV88:
    execution_fingerprint: str
    base_state: SuspendedSupportStateV82
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    water_density_cell_kg_m3: np.ndarray
    molecular_viscosity_cell_m2_s: np.ndarray
    state_sha256: str

    @property
    def accepted_updates(self) -> int:
        return self.base_state.accepted_updates

    @property
    def time_s(self) -> float:
        return self.base_state.time_s

    @property
    def suspended_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.suspended_mass_kg_m2

    @property
    def bed_mass_kg_m2(self) -> np.ndarray:
        return self.base_state.bed_mass_kg_m2


def _state_token(value: DynamicFluidStateV88) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "base_state_sha256": value.base_state.state_sha256,
        "temperature_sha256": _array_digest(value.temperature_cell_c),
        "salinity_sha256": _array_digest(value.salinity_cell_ppt),
        "water_density_sha256": _array_digest(
            value.water_density_cell_kg_m3),
        "molecular_viscosity_sha256": _array_digest(
            value.molecular_viscosity_cell_m2_s),
    })


def _wrap_state(execution: DynamicFluidExecutionV88,
                base_state: SuspendedSupportStateV82,
                temperature: object, salinity: object,
                density: object | None = None,
                viscosity: object | None = None) -> DynamicFluidStateV88:
    validate_suspended_support_state_v82(base_state, execution.base_execution)
    t = _frozen_field(temperature, execution.cell_count, "temperature",
                      temperature=True)
    s = _frozen_field(salinity, execution.cell_count, "salinity",
                      nonnegative=True)
    expected_rho, expected_nu = _properties(execution, t, s)
    rho = expected_rho if density is None else _frozen_field(
        density, execution.cell_count, "water density")
    nu = expected_nu if viscosity is None else _frozen_field(
        viscosity, execution.cell_count, "molecular viscosity")
    if (not np.array_equal(rho, expected_rho) or
            not np.array_equal(nu, expected_nu)):
        raise DynamicFluidSettlingV88Error(
            "v88 supplied property fields differ from the selected FLOW branch")
    placeholder = DynamicFluidStateV88(
        execution.fingerprint, base_state, t, s, rho, nu, "")
    return DynamicFluidStateV88(
        **{**placeholder.__dict__, "state_sha256": _state_token(placeholder)})


def resolve_dynamic_fluid_state_v88(
    execution: DynamicFluidExecutionV88, *,
    temperature_cell_c: object | None = None,
    salinity_cell_ppt: object | None = None,
    **base_state_fields,
) -> DynamicFluidStateV88:
    base = resolve_suspended_support_state_v82(
        execution.base_execution, **base_state_fields)
    t = _resolved_field(execution, temperature_cell_c, temperature=True)
    s = _resolved_field(execution, salinity_cell_ppt, temperature=False)
    return _wrap_state(execution, base, t, s)


def validate_dynamic_fluid_state_v88(
    state: DynamicFluidStateV88,
    execution: DynamicFluidExecutionV88,
) -> None:
    if (not isinstance(state, DynamicFluidStateV88) or
            state.execution_fingerprint != execution.fingerprint or
            state.state_sha256 != _state_token(state)):
        raise DynamicFluidSettlingV88Error(
            "v88 state is stale, mutated, or execution-mismatched")
    validate_suspended_support_state_v82(state.base_state,
                                         execution.base_execution)
    expected_rho, expected_nu = _properties(
        execution, state.temperature_cell_c, state.salinity_cell_ppt)
    if (not np.array_equal(state.water_density_cell_kg_m3, expected_rho) or
            not np.array_equal(state.molecular_viscosity_cell_m2_s,
                               expected_nu)):
        raise DynamicFluidSettlingV88Error(
            "v88 state fluid properties differ from accepted T/S fields")


def _apply_face_mass(execution: SuspendedSupportExecutionV82,
                     parent_mass: np.ndarray,
                     face_mass: np.ndarray) -> np.ndarray:
    base = execution.base_execution.base_execution
    result = np.array(parent_mass, dtype=np.float64, copy=True)
    for cell in range(base.cell_count):
        for slot in range(base.cell_face_offset[cell],
                          base.cell_face_offset[cell + 1]):
            face = int(base.cell_faces[slot])
            result[:, cell] -= (base.cell_face_sign[slot]*face_mass[:, face]/
                                base.cell_area_m2[cell])
    tolerance = 8192.0*np.finfo(float).eps*np.maximum(1.0, np.abs(parent_mass))
    if not np.isfinite(result).all() or np.any(result < -tolerance):
        raise DynamicFluidSettlingV88Error(
            "v88 accepted v82 diffusion reconstruction is invalid")
    return np.maximum(result, 0.0)


@dataclass(frozen=True)
class DynamicFluidTrialV88:
    execution_fingerprint: str
    parent_state_sha256: str
    base_trial: SuspendedSupportTrialV82
    candidate: DynamicFluidStateV88
    settling_water_density_kg_m3: np.ndarray
    settling_molecular_viscosity_m2_s: np.ndarray
    trial_sha256: str


def _trial_token(value: DynamicFluidTrialV88) -> str:
    return _fingerprint({
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "base_trial_sha256": value.base_trial.trial_sha256,
        "candidate_state_sha256": value.candidate.state_sha256,
        "settling_water_density_sha256": _array_digest(
            value.settling_water_density_kg_m3),
        "settling_viscosity_sha256": _array_digest(
            value.settling_molecular_viscosity_m2_s),
    })


def _wrap_trial(execution: DynamicFluidExecutionV88,
                parent: DynamicFluidStateV88,
                base_trial: SuspendedSupportTrialV82,
                candidate: DynamicFluidStateV88,
                settling_density: object,
                settling_viscosity: object) -> DynamicFluidTrialV88:
    rho = _frozen_field(settling_density, execution.cell_count,
                        "settling water density")
    nu = _frozen_field(settling_viscosity, execution.cell_count,
                       "settling molecular viscosity")
    placeholder = DynamicFluidTrialV88(
        execution.fingerprint, parent.state_sha256, base_trial, candidate,
        rho, nu, "")
    return DynamicFluidTrialV88(
        **{**placeholder.__dict__, "trial_sha256": _trial_token(placeholder)})


def _next_fields(execution: DynamicFluidExecutionV88, *,
                 accepted_endpoint_temperature_c: object | None,
                 accepted_endpoint_salinity_ppt: object | None
                 ) -> tuple[np.ndarray, np.ndarray]:
    return (
        _resolved_field(execution, accepted_endpoint_temperature_c,
                        temperature=True),
        _resolved_field(execution, accepted_endpoint_salinity_ppt,
                        temperature=False),
    )


def begin_dynamic_fluid_cpu_trial_v88(
    execution: DynamicFluidExecutionV88,
    parent: DynamicFluidStateV88, *, accepted_step_receipt,
    accepted_endpoint_temperature_c: object | None = None,
    accepted_endpoint_salinity_ppt: object | None = None,
    cancelled: bool = False, **inputs,
) -> DynamicFluidTrialV88:
    """Evaluate one CPU v88 trial; accepted T/S becomes next-step state."""
    if execution.role != ROLE_CPU_ORACLE:
        raise DynamicFluidSettlingV88Error(
            "v88 CPU trial requires isolated_cpu_oracle role")
    validate_dynamic_fluid_state_v88(parent, execution)
    if cancelled:
        raise DynamicFluidSettlingV88Error(
            "v88 update cancelled before candidate publication")
    if parent.accepted_updates >= MAX_UPDATES:
        raise DynamicFluidSettlingV88Error(
            "v88 accepted update limit was reached")
    next_t, next_s = _next_fields(
        execution,
        accepted_endpoint_temperature_c=accepted_endpoint_temperature_c,
        accepted_endpoint_salinity_ppt=accepted_endpoint_salinity_ppt)
    try:
        static_trial = begin_suspended_support_cpu_trial_v82(
            execution.base_execution, parent.base_state,
            accepted_step_receipt=accepted_step_receipt, **inputs)
    except (TypeError, ValueError) as error:
        raise DynamicFluidSettlingV88Error(
            "v88 accepted v82 CPU lifecycle rejected the trial") from error

    base_v82 = execution.base_execution
    base_v76 = base_v82.base_execution.base_execution
    dt = float(inputs["dt_s"])
    h0 = np.asarray(inputs["depth_before_m"], dtype=np.float64)
    h1 = np.asarray(inputs["depth_after_m"], dtype=np.float64)
    momentum = np.asarray(inputs["endpoint_momentum_m2_s"], dtype=np.float64)
    shear = np.asarray(inputs["shear_velocity_m_s"], dtype=np.float64)
    capacity = np.asarray(
        inputs["suspended_transport_capacity_kg_m_s"], dtype=np.float64)
    surface_fraction = np.asarray(inputs["surface_fraction"], dtype=np.float64)

    advected = _advected_mass_from_flux(
        base_v82, parent.suspended_mass_kg_m2,
        static_trial.base_trial.advective_face_flux_kg_m_s, dt)
    diffused = _apply_face_mass(
        base_v82, advected,
        static_trial.base_trial.diffusive_face_mass_owner_out_kg)
    wet = h1 > base_v76.dry_tolerance_m
    speed = np.zeros(execution.cell_count, dtype=np.float64)
    speed[wet] = np.hypot(momentum[0, wet], momentum[1, wet])/h1[wet]
    concentration = np.zeros(
        (execution.cell_count, execution.fraction_count), dtype=np.float64)
    concentration[wet] = (diffused[:, wet]/h1[wet]).T
    settling = settlefrm2_noncohesive(
        concentration, base_v76.profile.sediment_d50_m,
        base_v76.profile.settling_fac_dss,
        base_v76.profile.sediment_density_kg_m3,
        parent.water_density_cell_kg_m3,
        parent.temperature_cell_c,
        settling_reference_concentration_kg_m3=
            base_v76.profile.settling_reference_concentration_kg_m3,
        gravity_m_s2=base_v76.profile.gravity_m_s2, wet_mask=wet)
    if not np.array_equal(settling.molecular_viscosity_m2_s,
                          parent.molecular_viscosity_cell_m2_s):
        raise DynamicFluidSettlingV88Error(
            "v88 settling viscosity differs from accepted fluid state")
    exchange = advance_flow2dh_equilibrium_exchange(
        diffused, parent.bed_mass_kg_m2, capacity, surface_fraction,
        speed, shear, h0, h1, settling.settling_velocity_m_s.T,
        base_v76.cell_area_m2, dt_s=dt, fac_tsd=base_v76.profile.fac_tsd,
        sediment_threshold_m=base_v76.profile.sediment_threshold_m,
        minimum_reference_depth_m=base_v76.profile.reference_depth_floor_m,
        flow_wet_mask=wet)
    cleanup = np.where(wet[None, :], 0.0,
                       exchange.suspended_mass_kg_m2)
    final_suspended = exchange.suspended_mass_kg_m2-cleanup
    final_bed = exchange.bed_mass_kg_m2+cleanup
    area = base_v76.cell_area_m2[None, :]
    cleanup_water = -np.sum(cleanup*area, axis=1)
    cleanup_bed = -cleanup_water
    old = parent.base_state.base_state.lifecycle
    static = static_trial.candidate.base_state.lifecycle
    candidate_lifecycle = resolve_suspended_lifecycle_state_v76(
        base_v76,
        suspended_mass_kg_m2=final_suspended,
        bed_mass_kg_m2=final_bed,
        boundary_last_outflow_concentration_kg_m3=
            static.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=static.boundary_return_clock_s,
        advective_boundary_ledger_kg=static.advective_boundary_ledger_kg,
        diffusive_boundary_ledger_kg=static.diffusive_boundary_ledger_kg,
        water_exchange_ledger_kg=(old.water_exchange_ledger_kg+
                                  exchange.suspended_exchange_ledger_kg),
        bed_exchange_ledger_kg=(old.bed_exchange_ledger_kg+
                                exchange.bed_exchange_ledger_kg),
        wetdry_water_ledger_kg=old.wetdry_water_ledger_kg+cleanup_water,
        wetdry_bed_ledger_kg=old.wetdry_bed_ledger_kg+cleanup_bed,
        reference_water_mass_kg=old.reference_water_mass_kg,
        reference_bed_mass_kg=old.reference_bed_mass_kg,
        accepted_updates=parent.accepted_updates+1,
        time_s=parent.time_s+dt)
    base_v78_trial = build_multifraction_trial_v78(
        base_v82.base_execution, parent.base_state.base_state,
        accepted_step_receipt, candidate_lifecycle,
        boundary_concentration_kg_m3=
            static_trial.base_trial.boundary_concentration_kg_m3,
        advective_face_flux_kg_m_s=
            static_trial.base_trial.advective_face_flux_kg_m_s,
        diffusive_face_mass_owner_out_kg=
            static_trial.base_trial.diffusive_face_mass_owner_out_kg,
        settling_velocity_m_s=settling.settling_velocity_m_s.T,
        equilibrium_concentration_kg_m3=
            exchange.equilibrium_concentration_kg_m3,
        transferred_to_suspension_kg_m2=
            exchange.transferred_to_suspension_kg_m2,
        wetdry_transfer_to_bed_kg_m2=cleanup,
        observed_parallel_workers=1, backend=ROLE_CPU_ORACLE)
    base_trial = _wrap_trial_v82(
        base_v82, parent.base_state, base_v78_trial,
        static_trial.effective_horizontal_diffusivity_m2_s)
    candidate_base = base_trial.candidate
    candidate = _wrap_state(execution, candidate_base, next_t, next_s)
    return _wrap_trial(
        execution, parent, base_trial, candidate,
        parent.water_density_cell_kg_m3,
        parent.molecular_viscosity_cell_m2_s)


def _validate_trial(execution: DynamicFluidExecutionV88,
                    parent: DynamicFluidStateV88,
                    trial: DynamicFluidTrialV88) -> None:
    validate_dynamic_fluid_state_v88(parent, execution)
    if (not isinstance(trial, DynamicFluidTrialV88) or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.trial_sha256 != _trial_token(trial)):
        raise DynamicFluidSettlingV88Error(
            "v88 trial is stale, mutated, or belongs to another parent")
    validate_dynamic_fluid_state_v88(trial.candidate, execution)


def commit_dynamic_fluid_trial_v88(
    execution: DynamicFluidExecutionV88, parent: DynamicFluidStateV88,
    trial: DynamicFluidTrialV88,
) -> DynamicFluidStateV88:
    _validate_trial(execution, parent, trial)
    committed = commit_suspended_support_trial_v82(
        execution.base_execution, parent.base_state, trial.base_trial)
    if committed.state_sha256 != trial.candidate.base_state.state_sha256:
        raise DynamicFluidSettlingV88Error(
            "v88/base committed candidate differs")
    return trial.candidate


def rollback_dynamic_fluid_trial_v88(
    execution: DynamicFluidExecutionV88, parent: DynamicFluidStateV88,
    trial: DynamicFluidTrialV88,
) -> DynamicFluidStateV88:
    _validate_trial(execution, parent, trial)
    rolled = rollback_suspended_support_trial_v82(
        execution.base_execution, parent.base_state, trial.base_trial)
    if rolled is not parent.base_state:
        raise DynamicFluidSettlingV88Error("v88/base rollback differs")
    return parent


def cancel_dynamic_fluid_trial_v88(*args, **kwargs) -> DynamicFluidStateV88:
    return rollback_dynamic_fluid_trial_v88(*args, **kwargs)


def dynamic_fluid_checkpoint_v88(
    execution: DynamicFluidExecutionV88,
    state: DynamicFluidStateV88,
) -> dict:
    validate_dynamic_fluid_state_v88(state, execution)
    body = {
        "schema": SCHEMA,
        "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "execution_fingerprint": execution.fingerprint,
        "integrator": INTEGRATOR,
        "base_v82_checkpoint": suspended_support_checkpoint_v82(
            execution.base_execution, state.base_state),
        "accepted_fluid_state": {
            "temperature_cell_c": state.temperature_cell_c.tolist(),
            "salinity_cell_ppt": state.salinity_cell_ppt.tolist(),
            "water_density_cell_kg_m3":
                state.water_density_cell_kg_m3.tolist(),
            "molecular_viscosity_cell_m2_s":
                state.molecular_viscosity_cell_m2_s.tolist(),
        },
        "state_sha256": state.state_sha256,
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_dynamic_fluid_checkpoint_v88(
    execution: DynamicFluidExecutionV88,
    document: Mapping[str, object],
) -> DynamicFluidStateV88:
    expected = {"schema", "scope_id", "semantic_id",
                "execution_fingerprint", "integrator",
                "base_v82_checkpoint", "accepted_fluid_state",
                "state_sha256", "fingerprint"}
    if not isinstance(document, Mapping) or set(document) != expected:
        raise DynamicFluidSettlingV88Error(
            "v88 checkpoint inventory differs")
    body = {key: document[key] for key in document if key != "fingerprint"}
    if (document["fingerprint"] != _fingerprint(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["execution_fingerprint"] != execution.fingerprint or
            document["integrator"] != INTEGRATOR):
        raise DynamicFluidSettlingV88Error(
            "v88 checkpoint identity differs")
    try:
        base = load_suspended_support_checkpoint_v82(
            execution.base_execution, document["base_v82_checkpoint"])
        fields = document["accepted_fluid_state"]
        if not isinstance(fields, Mapping) or set(fields) != {
                "temperature_cell_c", "salinity_cell_ppt",
                "water_density_cell_kg_m3",
                "molecular_viscosity_cell_m2_s"}:
            raise DynamicFluidSettlingV88Error(
                "v88 checkpoint fluid-state inventory differs")
        state = _wrap_state(
            execution, base, fields["temperature_cell_c"],
            fields["salinity_cell_ppt"],
            fields["water_density_cell_kg_m3"],
            fields["molecular_viscosity_cell_m2_s"])
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, DynamicFluidSettlingV88Error):
            raise
        raise DynamicFluidSettlingV88Error(
            "v88 checkpoint fluid state is invalid") from error
    if document["state_sha256"] != state.state_sha256:
        raise DynamicFluidSettlingV88Error(
            "v88 checkpoint state digest differs")
    return state


def method_identity_v88() -> dict:
    return {
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "builds_additively_on":
            "braids-q4-flow2dh-suspended-support-closures-v82-v1",
        "integrator": INTEGRATOR,
        "backend_roles": [ROLE_CPU_ORACLE, ROLE_CUDA],
        "eos_choices": list(EOS_CHOICES),
        "constant_density_branch": "rhow_if_no_active_T_or_S",
        "local_eqsettle_viscosity": "4e-5/(20+T)",
        "call_order": "accepted(parent)_fluid -> settling/exchange -> "
                      "accepted(endpoint)_fluid_for_next_update",
        "hindering_concentration_timing":
            "BRAIDS_post_FCT_diffusion_split_state; locked_FLOW_uses_parent_"
            "r0_before_tritra",
        "scalar_transport_owner": "external_not_claimed",
        "endpoint_field_provenance":
            "caller_supplied_transaction_hashed; typed_scalar_transport_"
            "receipt_open",
        "bounded_input_ranges": {
            "temperature_c": "greater_than_-20_for_positive_viscosity",
            "salinity_ppt": "nonnegative",
            "relation_to_locked_FLOW":
                "deliberately_stricter_component_validation_not_range_parity",
        },
        "accepted_update_limit": MAX_UPDATES,
        "automatic_cpu_fallback": False,
        "unsupported_fail_closed": [
            "KMAX>1/3D", "DSS_2004 fine-flocculation branch",
            "SalTemCon", "Manning-Dyer flocs", "Chassagne-Safar flocs",
            "user DLL settling", "cohesive fractions",
        ],
        "component_only_not_product_integration": True,
        "delft3d_source_modified": False,
        "long_run_claimed": False,
    }


FROZEN_PREDECESSOR_SHA256 = {
    "braids_core/sediment/suspended_support_closures_v82.py":
        "aad9844d3ee66a7c98f2995e9f187229dbf9a3d62498056261aba113cde8fd24",
    "braids_gpu/suspended_support_closures_v82.py":
        "41656dfcf196a1018c8e8cf1e43adbf4276f002bba8b020c32ef98e6586f5271",
    "braids_core/sediment/flow2dh_settling.py":
        "ab1abf406a9a08b4a193de0ed93c32956a505b3ba7fba225c05f00aaca745616",
    "braids_gpu/law11_soursin_exchange.cu":
        "26fd25cee3bbdfdced65690e9367e34daf8fa50ed260dd425b7a3b2248c2ff6d",
}


def verify_frozen_predecessors_v88(root: str | Path) -> dict[str, str]:
    root = Path(root)
    result = {}
    for relative, expected in FROZEN_PREDECESSOR_SHA256.items():
        actual = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise DynamicFluidSettlingV88Error(
                "v88 frozen predecessor hash differs: " + relative)
        result[relative] = actual
    return result


__all__ = [
    "CHECKPOINT_SCOPE_ID", "DynamicFluidExecutionV88",
    "DynamicFluidSettlingV88Error", "DynamicFluidStateV88",
    "DynamicFluidTrialV88", "EOS_CHOICES", "EOS_ECKART", "EOS_NACL",
    "EOS_UNESCO", "INTEGRATOR", "MAX_UPDATES", "MODE_CELL",
    "MODE_CONSTANT", "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "begin_dynamic_fluid_cpu_trial_v88",
    "cancel_dynamic_fluid_trial_v88", "commit_dynamic_fluid_trial_v88",
    "dynamic_fluid_checkpoint_v88", "execution_document_v88",
    "load_dynamic_fluid_checkpoint_v88", "method_identity_v88",
    "resolve_dynamic_fluid_execution_v88",
    "resolve_dynamic_fluid_state_v88", "resolve_water_density_v88",
    "rollback_dynamic_fluid_trial_v88",
    "validate_dynamic_fluid_state_v88", "verify_frozen_predecessors_v88",
    "water_density_eckart_v88", "water_density_nacl_v88",
    "water_density_unesco_v88",
]
