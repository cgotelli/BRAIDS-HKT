"""Typed CPU face-time integrals for the first real coupled AMR gate.

This module is deliberately additive.  It reads a still-open, fully validated
``CoupledWetDryCPUTrialV97`` and exposes the conserved interface integrals that
v98 must reflux.  It does not build a shadow state, publish a clock, mutate a
Q3 provider, or claim that the v98 AMR owner is complete.

Rates are integrated exactly once here.  The v78/v92 horizontal diffusive face
arrays already contain accepted per-step mass and therefore are *not*
multiplied by ``dt`` a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np
from .amr_suspended_bed_selection_v100 import mode_v100

from .active_suspended_control_v94 import CPUQ3V83ActiveFaceTrialV94
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    CoupledWetDryCPUTrialV97,
    _validate_trial as _validate_v97_trial,
)
from .coupled_amr_register_lifecycle_v98 import (
    CoupledAMRTwoPhaseRegisterV98,
    PHASE_HYDRAULIC,
    PHASE_SCALAR_BED,
    compose_two_phase_register_v98,
    resolve_scheduled_register_phase_v98,
    validate_two_phase_register_v98,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr_register.cpu.v98"
BACKEND = "native_cpu"


class CoupledAMRRegisterCPUV98Error(ValueError):
    """Fail-closed v98 CPU register error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(ch not in "0123456789abcdef" for ch in value)):
        raise CoupledAMRRegisterCPUV98Error(
            f"v98 {name} must be a lowercase SHA-256 digest")
    return value


def _frozen(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu" or
            not np.isfinite(array).all()):
        raise CoupledAMRRegisterCPUV98Error(
            f"v98 {name} must be finite with shape {shape}")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _grid(execution: CoupledWetDryCPUExecutionV97):
    return (execution.active_execution.base_execution.base_execution.
            base_execution.base_execution.base_execution)


@dataclass(frozen=True)
class CoupledFaceTimeIntegralsCPUV98:
    """All conserved face-time integrals from one actual v97 CPU trial."""

    execution_fingerprint: str
    parent_state_sha256: str
    candidate_state_sha256: str
    source_trial_sha256: str
    hydraulic_rate_result_sha256: str
    q3_face_trial_sha256: str
    parent_accepted_updates: int
    candidate_accepted_updates: int
    time_start_s: float
    time_end_s: float
    dt_s: float
    morphological_factor: float
    face_length_m: np.ndarray
    midpoint_depth_m: np.ndarray
    water_volume_m3: np.ndarray
    augmented_left_face_integral: np.ndarray
    augmented_right_face_integral: np.ndarray
    owner_hydrostatic_momentum_integral_m4_s: np.ndarray
    neighbour_hydrostatic_momentum_integral_m4_s: np.ndarray
    augmented_momentum_jump_integral_m4_s: np.ndarray
    viscous_momentum_integral_m4_s: np.ndarray
    suspended_advective_mass_kg: np.ndarray
    suspended_diffusive_mass_kg: np.ndarray
    fluid_advective_integral: np.ndarray
    fluid_diffusive_integral: np.ndarray
    bedload_mass_kg: np.ndarray
    backend: str
    receipt_sha256: str
    suspended_bed_divergence_selected: bool = False

    @property
    def selected_suspended_bed_mass_kg(self):
        return (self.suspended_advective_mass_kg+self.suspended_diffusive_mass_kg
                if self.suspended_bed_divergence_selected else
                np.zeros_like(self.suspended_advective_mass_kg))


def _integral_payload(value: CoupledFaceTimeIntegralsCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":face-time-integrals",
        "suspended_bed_divergence_selected": value.suspended_bed_divergence_selected,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "candidate_state_sha256": value.candidate_state_sha256,
        "source_trial_sha256": value.source_trial_sha256,
        "hydraulic_rate_result_sha256":
            value.hydraulic_rate_result_sha256,
        "q3_face_trial_sha256": value.q3_face_trial_sha256,
        "parent_accepted_updates": value.parent_accepted_updates,
        "candidate_accepted_updates": value.candidate_accepted_updates,
        "time_start_s": value.time_start_s,
        "time_end_s": value.time_end_s,
        "dt_s": value.dt_s,
        "morphological_factor": value.morphological_factor,
        "arrays_sha256": _digest(
            value.face_length_m, value.midpoint_depth_m,
            value.water_volume_m3,
            value.augmented_left_face_integral,
            value.augmented_right_face_integral,
            value.owner_hydrostatic_momentum_integral_m4_s,
            value.neighbour_hydrostatic_momentum_integral_m4_s,
            value.augmented_momentum_jump_integral_m4_s,
            value.viscous_momentum_integral_m4_s,
            value.suspended_advective_mass_kg,
            value.suspended_diffusive_mass_kg,
            value.fluid_advective_integral,
            value.fluid_diffusive_integral,
            value.bedload_mass_kg),
        "rate_integration": "face_length_times_accepted_dt_once",
        "viscous_integration":
            "already_face_length_integrated_times_accepted_dt_once",
        "hydraulic_momentum":
            "augmented_left_right_sides_plus_local_hydrostatic_ledger",
        "diffusive_integration": "already_integrated_no_second_dt",
        "backend": value.backend,
        "state_or_clock_publication_count": 0,
    }


def validate_coupled_face_time_integrals_cpu_v98(
    value: CoupledFaceTimeIntegralsCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
) -> None:
    if type(value) is not CoupledFaceTimeIntegralsCPUV98:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 face-time integral receipt type differs")
    nf, ng = execution.face_count, execution.fraction_count
    arrays = (
        (value.face_length_m, (nf,), "face length"),
        (value.midpoint_depth_m, (execution.cell_count,), "midpoint depth"),
        (value.water_volume_m3, (nf,), "water integral"),
        (value.augmented_left_face_integral, (3, nf),
         "augmented left integral"),
        (value.augmented_right_face_integral, (3, nf),
         "augmented right integral"),
        (value.owner_hydrostatic_momentum_integral_m4_s, (2, nf),
         "owner hydrostatic momentum integral"),
        (value.neighbour_hydrostatic_momentum_integral_m4_s, (2, nf),
         "neighbour hydrostatic momentum integral"),
        (value.augmented_momentum_jump_integral_m4_s, (2, nf),
         "augmented momentum jump integral"),
        (value.viscous_momentum_integral_m4_s, (2, nf),
         "viscous momentum integral"),
        (value.suspended_advective_mass_kg, (ng, nf),
         "suspended advective integral"),
        (value.suspended_diffusive_mass_kg, (ng, nf),
         "suspended diffusive integral"),
        (value.fluid_advective_integral, (2, nf),
         "fluid advective integral"),
        (value.fluid_diffusive_integral, (2, nf),
         "fluid diffusive integral"),
        (value.bedload_mass_kg, (ng, nf), "bedload integral"),
    )
    for array, shape, name in arrays:
        checked = _frozen(array, shape, name)
        if (not np.array_equal(checked, array) or array.flags.writeable or
                not array.flags.c_contiguous):
            raise CoupledAMRRegisterCPUV98Error(
                f"v98 {name} changed during validation")
    if not np.array_equal(value.face_length_m, _grid(execution).face_length_m):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 face-time integral geometry differs from its execution")
    if (value.suspended_bed_divergence_selected !=
            (mode_v100(execution.active_execution.base_execution) == 1) or
            value.execution_fingerprint != execution.fingerprint or
            value.backend != BACKEND or
            value.parent_accepted_updates < 0 or
            value.candidate_accepted_updates !=
                value.parent_accepted_updates + 1 or
            not math.isfinite(value.time_start_s) or
            not math.isfinite(value.time_end_s) or
            not math.isfinite(value.dt_s) or value.dt_s <= 0.0 or
            not math.isfinite(value.morphological_factor) or
            value.morphological_factor <= 0.0 or
            value.time_end_s != value.time_start_s + value.dt_s or
            value.receipt_sha256 != _hash(_integral_payload(value))):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 face-time integral identity, clock, or backend differs")
    for raw, name in (
            (value.parent_state_sha256, "parent state"),
            (value.candidate_state_sha256, "candidate state"),
            (value.source_trial_sha256, "source trial"),
            (value.hydraulic_rate_result_sha256,
             "hydraulic rate result"),
            (value.q3_face_trial_sha256, "Q3 face trial")):
        _sha(raw, name)
    grid = _grid(execution)
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal, dtype=np.float64)
    length_dt = value.face_length_m*value.dt_s
    if (not np.array_equal(value.augmented_left_face_integral[0],
                           value.water_volume_m3) or
            not np.array_equal(value.augmented_right_face_integral[0],
                               value.water_volume_m3) or
            not np.array_equal(
                value.augmented_momentum_jump_integral_m4_s,
                value.augmented_right_face_integral[1:]-
                value.augmented_left_face_integral[1:])):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 augmented side/water/source-jump identity differs")
    owner_pressure = (
        0.5*grid.profile.gravity_m_s2*
        value.midpoint_depth_m[owner]**2)[None, :]*normal*length_dt
    neighbour_pressure = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    neighbour_pressure[:, internal] = (
        0.5*grid.profile.gravity_m_s2*
        value.midpoint_depth_m[neighbour[internal]]**2
    )[None, :]*normal[:, internal]*length_dt[internal]
    if (not np.array_equal(
            owner_pressure,
            value.owner_hydrostatic_momentum_integral_m4_s) or
            not np.array_equal(
                neighbour_pressure,
                value.neighbour_hydrostatic_momentum_integral_m4_s)):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 local hydrostatic-pressure ledger differs")


def extract_coupled_face_time_integrals_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    q3_provider: object,
    trial: CoupledWetDryCPUTrialV97,
) -> CoupledFaceTimeIntegralsCPUV98:
    """Extract actual conserved interface integrals from one open v97 trial."""
    if (type(execution) is not CoupledWetDryCPUExecutionV97 or
            type(parent) is not CoupledWetDryCPUStateV97 or
            type(trial) is not CoupledWetDryCPUTrialV97):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 extraction requires exact CPU v97 execution/state/trial")
    _validate_v97_trial(execution, parent, q3_provider, trial)
    grid = _grid(execution)
    nf, ng = execution.face_count, execution.fraction_count
    length = _frozen(grid.face_length_m, (nf,), "face length")
    hydraulic = trial.hydraulic_evidence.wetdry_trial
    v93 = trial.active_trial.base_v93_trial
    v92 = v93.base_v92_trial
    suspended = v92.suspended_trial.base_trial
    fluid = v92.fluid_transport
    open_trial = v93.q3_open_trial
    face_trial = getattr(open_trial, "face_trial", None)
    if type(face_trial) is not CPUQ3V83ActiveFaceTrialV94:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 requires the exact accepted v97 Q3 face trial")
    dt = float(hydraulic.dt_s)
    if (not math.isfinite(dt) or dt <= 0.0 or
            float(v92.hydraulic_receipt.dt_s) != dt or
            float(face_trial.dt_s) != dt):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 nested hydraulic/scalar/Q3 timesteps differ")
    shape_checks = (
        (hydraulic.accepted_limited_face_water_flux_m2_s, (nf,),
         "accepted water flux"),
        (hydraulic.native_rate_result.limited_left_face_flux, (3, nf),
         "accepted augmented left flux"),
        (hydraulic.native_rate_result.limited_right_face_flux, (3, nf),
         "accepted augmented right flux"),
        (v92.momentum_face_flux_m4_s2, (2, nf), "momentum flux"),
        (suspended.advective_face_flux_kg_m_s, (ng, nf),
         "suspended advective flux"),
        (suspended.diffusive_face_mass_owner_out_kg, (ng, nf),
         "suspended diffusive mass"),
        (fluid.advective_face_flux, (2, nf), "fluid advective flux"),
        (fluid.diffusive_face_amount, (2, nf), "fluid diffusive amount"),
        (face_trial.face_flux_kg_m_s, (ng, nf), "bedload face flux"),
    )
    checked = [_frozen(array, shape, name)
               for array, shape, name in shape_checks]
    water_flux, augmented_left, augmented_right, viscous_momentum_flux, \
        suspended_advective, suspended_diffusive, fluid_advective, \
        fluid_diffusive, bedload_flux = checked
    if (not np.array_equal(augmented_left[0], water_flux) or
            not np.array_equal(augmented_right[0], water_flux)):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 accepted water flux differs from augmented side fluxes")
    midpoint_depth = np.where(
        np.asarray(hydraulic.frozen_wet_mask, dtype=bool),
        np.asarray(hydraulic.midpoint_q[0]), 0.0)
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal, dtype=np.float64)
    gravity = float(grid.profile.gravity_m_s2)
    owner_pressure = (
        0.5*gravity*midpoint_depth[owner]**2)[None, :]*normal
    neighbour_pressure = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    neighbour_pressure[:, internal] = (
        0.5*gravity*midpoint_depth[neighbour[internal]]**2
    )[None, :]*normal[:, internal]
    length_dt = length*dt
    augmented_left_integral = augmented_left*length_dt[None, :]
    augmented_right_integral = augmented_right*length_dt[None, :]
    arrays = (
        length,
        midpoint_depth,
        water_flux*length_dt,
        augmented_left_integral,
        augmented_right_integral,
        owner_pressure*length_dt[None, :],
        neighbour_pressure*length_dt[None, :],
        augmented_right_integral[1:]-augmented_left_integral[1:],
        viscous_momentum_flux*dt,
        suspended_advective*length[None, :]*dt,
        suspended_diffusive,
        fluid_advective*length[None, :]*dt,
        fluid_diffusive,
        bedload_flux*length[None, :]*dt*float(face_trial.morphological_factor),
    )
    frozen = []
    shapes = ((nf,), (execution.cell_count,), (nf,),
              (3, nf), (3, nf), (2, nf), (2, nf),
              (2, nf), (2, nf), (ng, nf), (ng, nf),
              (2, nf), (2, nf), (ng, nf))
    names = ("face length", "midpoint depth", "water volume",
             "augmented left integral",
             "augmented right integral", "owner hydrostatic integral",
             "neighbour hydrostatic integral", "augmented momentum jump",
             "viscous momentum integral",
             "suspended advective mass", "suspended diffusive mass",
             "fluid advective integral", "fluid diffusive integral",
             "bedload mass")
    for array, shape, name in zip(arrays, shapes, names):
        frozen.append(_frozen(array, shape, name))
    placeholder = CoupledFaceTimeIntegralsCPUV98(
        execution.fingerprint, parent.state_sha256,
        trial.candidate.state_sha256, trial.trial_sha256,
        hydraulic.native_rate_result.result_sha256,
        face_trial.trial_sha256, parent.accepted_updates,
        trial.candidate.accepted_updates, parent.time_s,
        trial.candidate.time_s, dt, float(face_trial.morphological_factor),
        *frozen, BACKEND, "",
        mode_v100(execution.active_execution.base_execution) == 1)
    result = CoupledFaceTimeIntegralsCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_integral_payload(placeholder))})
    validate_coupled_face_time_integrals_cpu_v98(result, execution)
    return result


def extract_coupled_fine_face_time_integrals_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    source_provider: object,
    fine_state: object,
    private_provider: object,
    hydraulic_trial: object,
    hydraulic_runtime: object,
    scalar_trace: object,
    q3_trace: object,
) -> CoupledFaceTimeIntegralsCPUV98:
    """Extract one real v98 fine-substep register image.

    A fine AMR substep is deliberately *not* published as a historical v97
    trial: its scalar and Q3 internal-interface ledgers are outside the v97
    physical-boundary-only state semantics.  This extractor therefore reads
    the authenticated v98 private fine state directly.  It integrates the
    same quantities as :func:`extract_coupled_face_time_integrals_cpu_v98`
    exactly once, while keeping the v98 operator-chain receipt as the source
    identity.
    """
    # Local imports avoid a module cycle: the fine owner consumes the register
    # algebra only through its outer hierarchy owner.
    from .coupled_amr_fine_substep_cpu_v98 import (
        CoupledAMRFineSubstepStateCPUV98,
        validate_coupled_amr_fine_substep_state_v98,
    )
    from .coupled_amr_fwave_native_cpu_v98 import (
        NativeAMRInterfaceFwaveRuntimeV98,
    )
    from .coupled_amr_wetdry_cpu_v98 import (
        WetDryAMRInterfaceTrialV98,
        validate_wetdry_amr_interface_trial_v98,
    )
    from .coupled_wetdry_owner_cpu_v97 import _accepted_density

    if (type(fine_state) is not CoupledAMRFineSubstepStateCPUV98 or
            type(hydraulic_trial) is not WetDryAMRInterfaceTrialV98 or
            type(hydraulic_runtime) is not
                NativeAMRInterfaceFwaveRuntimeV98):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 fine extraction requires exact fine-state/hydraulic types")
    validate_coupled_amr_fine_substep_state_v98(
        fine_state, execution, parent, source_provider,
        hydraulic_runtime.trace, scalar_trace, q3_trace,
        hydraulic_runtime.turbulence_trace,
        private_provider)
    water, mixture = _accepted_density(execution, parent)
    validate_wetdry_amr_interface_trial_v98(
        hydraulic_trial, execution.hydraulic_execution,
        runtime=hydraulic_runtime, parent_q=parent.q,
        bed_elevation_m=parent.bed_elevation_m,
        accepted_water_density_kg_m3=water,
        accepted_mixture_density_kg_m3=mixture,
        dt_s=fine_state.dt_s)
    if (fine_state.hydraulic_trial_sha256 !=
            hydraulic_trial.trial_sha256 or
            fine_state.substep_index != hydraulic_trial.substep_index):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 fine state and hydraulic trial differ")

    grid = _grid(execution)
    nf, ng = execution.face_count, execution.fraction_count
    length = _frozen(grid.face_length_m, (nf,), "fine face length")
    hydraulic = hydraulic_trial.wetdry_trial
    scalar = fine_state.scalar_stage
    q3 = fine_state.q3_interface_result
    dt = float(fine_state.dt_s)
    water_flux = _frozen(
        hydraulic.accepted_limited_face_water_flux_m2_s,
        (nf,), "fine accepted water flux")
    augmented_left = _frozen(
        hydraulic.native_rate_result.limited_left_face_flux,
        (3, nf), "fine augmented left flux")
    augmented_right = _frozen(
        hydraulic.native_rate_result.limited_right_face_flux,
        (3, nf), "fine augmented right flux")
    internal_viscous = _frozen(
        hydraulic_trial.accepted_internal_turbulence_result.
        face_flux_m4_s2,
        (2, nf), "fine internal viscous flux")
    trace_viscous = _frozen(
        hydraulic_trial.accepted_trace_turbulence_result[
            "face_flux_m4_s2"],
        (2, nf), "fine trace viscous flux")
    if (not np.array_equal(augmented_left[0], water_flux) or
            not np.array_equal(augmented_right[0], water_flux)):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 fine augmented and accepted water flux differ")

    suspended_advective = _frozen(
        scalar.advective_face_flux_kg_m_s,
        (ng, nf), "fine suspended advective flux")
    suspended_diffusive = _frozen(
        scalar.diffusive_face_mass_owner_out_kg,
        (ng, nf), "fine suspended diffusive mass")
    fluid_advective = _frozen(
        scalar.fluid_result.advective_face_flux,
        (2, nf), "fine fluid advective flux")
    fluid_diffusive = _frozen(
        scalar.fluid_result.diffusive_face_amount,
        (2, nf), "fine fluid diffusive amount")
    bedload_flux = _frozen(
        q3.face_flux_kg_m_s,
        (ng, nf), "fine bedload face flux")
    midpoint_depth = np.where(
        np.asarray(hydraulic.frozen_wet_mask, dtype=bool),
        np.asarray(hydraulic.midpoint_q[0]), 0.0)
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal, dtype=np.float64)
    gravity = float(grid.profile.gravity_m_s2)
    owner_pressure = (
        0.5*gravity*midpoint_depth[owner]**2)[None, :]*normal
    neighbour_pressure = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    neighbour_pressure[:, internal] = (
        0.5*gravity*midpoint_depth[neighbour[internal]]**2
    )[None, :]*normal[:, internal]
    length_dt = length*dt
    left_integral = augmented_left*length_dt[None, :]
    right_integral = augmented_right*length_dt[None, :]
    morphological_factor = float(
        execution.active_execution.base_execution.morphological_factor)
    arrays = (
        length,
        midpoint_depth,
        water_flux*length_dt,
        left_integral,
        right_integral,
        owner_pressure*length_dt[None, :],
        neighbour_pressure*length_dt[None, :],
        right_integral[1:]-left_integral[1:],
        (internal_viscous+trace_viscous)*dt,
        suspended_advective*length[None, :]*dt,
        suspended_diffusive,
        fluid_advective*length[None, :]*dt,
        fluid_diffusive,
        bedload_flux*length[None, :]*dt*morphological_factor,
    )
    shapes = ((nf,), (execution.cell_count,), (nf,),
              (3, nf), (3, nf), (2, nf), (2, nf),
              (2, nf), (2, nf), (ng, nf), (ng, nf),
              (2, nf), (2, nf), (ng, nf))
    names = ("fine face length", "fine midpoint depth",
             "fine water volume", "fine augmented left integral",
             "fine augmented right integral",
             "fine owner hydrostatic integral",
             "fine neighbour hydrostatic integral",
             "fine augmented momentum jump",
             "fine viscous momentum integral",
             "fine suspended advective mass",
             "fine suspended diffusive mass",
             "fine fluid advective integral",
             "fine fluid diffusive integral", "fine bedload mass")
    frozen = tuple(_frozen(array, shape, name)
                   for array, shape, name in zip(arrays, shapes, names))
    placeholder = CoupledFaceTimeIntegralsCPUV98(
        execution.fingerprint, parent.state_sha256,
        fine_state.numerical_parent.state_sha256,
        fine_state.state_sha256,
        hydraulic.native_rate_result.result_sha256,
        q3.output_sha256,
        parent.accepted_updates,
        fine_state.numerical_parent.accepted_updates,
        parent.time_s, fine_state.numerical_parent.time_s,
        dt, morphological_factor, *frozen, BACKEND, "",
        mode_v100(execution.active_execution.base_execution) == 1)
    result = CoupledFaceTimeIntegralsCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_integral_payload(placeholder))})
    validate_coupled_face_time_integrals_cpu_v98(result, execution)
    return result


@dataclass(frozen=True)
class CoupledAMRInterfaceMapCPUV98:
    """One real coarse/fine face partition in a pair of v97 executions."""

    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    coarse_leaf_cell: int
    coarse_face: int
    fine_interface_cells: tuple[int, int]
    fine_faces: tuple[int, int]
    coarse_to_canonical: int
    fine_to_canonical: tuple[int, int]
    coarse_global_face_id: int
    fine_global_face_ids: tuple[int, int]
    canonical_normal: np.ndarray
    coarse_cell_area_m2: float
    fingerprint: str


def _interface_payload(value: CoupledAMRInterfaceMapCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":one-interface-map",
        "coarse_execution": value.coarse_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "coarse_leaf_cell": value.coarse_leaf_cell,
        "coarse_face": value.coarse_face,
        "fine_interface_cells": list(value.fine_interface_cells),
        "fine_faces": list(value.fine_faces),
        "coarse_to_canonical": value.coarse_to_canonical,
        "fine_to_canonical": list(value.fine_to_canonical),
        "coarse_global_face_id": value.coarse_global_face_id,
        "fine_global_face_ids": list(value.fine_global_face_ids),
        "canonical_normal_sha256": _digest(value.canonical_normal),
        "coarse_cell_area_m2": value.coarse_cell_area_m2,
        "refinement_ratio": 2,
        "fine_interface_kind": "typed_internal_coarse_fine_not_physical_open",
    }


def resolve_coupled_amr_interface_map_cpu_v98(
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97, *,
    coarse_leaf_cell: int,
    coarse_face: int,
    fine_interface_cells: Sequence[int],
    fine_faces: Sequence[int],
    coarse_global_face_id: int,
    fine_global_face_ids: Sequence[int],
) -> CoupledAMRInterfaceMapCPUV98:
    """Resolve the minimum 2:1 interface from actual CPU v97 topologies."""
    if (type(coarse_execution) is not CoupledWetDryCPUExecutionV97 or
            type(fine_execution) is not CoupledWetDryCPUExecutionV97):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 interface requires exact CPU v97 executions")
    cg, fg = _grid(coarse_execution), _grid(fine_execution)
    if (coarse_execution.fraction_count != fine_execution.fraction_count or
            cg.profile.fingerprint != fg.profile.fingerprint):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse/fine sediment profiles or fraction counts differ")
    values = (coarse_leaf_cell, coarse_face, coarse_global_face_id)
    if any(type(value) is not int for value in values):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse interface indices must be integers")
    fine_cells = tuple(fine_interface_cells)
    faces = tuple(fine_faces)
    global_ids = tuple(fine_global_face_ids)
    if (len(fine_cells) != 2 or len(faces) != 2 or len(global_ids) != 2 or
            len(set(fine_cells)) != 2 or len(set(faces)) != 2 or
            len(set(global_ids)) != 2 or coarse_global_face_id < 0 or
            any(value < 0 for value in global_ids) or
            coarse_global_face_id in global_ids or
            any(type(value) is not int for value in (*fine_cells, *faces,
                                                     *global_ids))):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 requires two distinct fine interface faces and stable IDs")
    nc, nf = coarse_execution.cell_count, coarse_execution.face_count
    if (not 0 <= coarse_leaf_cell < nc or not 0 <= coarse_face < nf):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse interface indices are outside the topology")
    owner = int(cg.owner[coarse_face])
    neighbour = int(cg.neighbour[coarse_face])
    if (owner != coarse_leaf_cell or neighbour != -1 or
            int(cg.boundary_index[coarse_face]) >= 0):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse interface must be an internal negative-neighbour "
            "trace owned by the uncovered leaf")
    coarse_sign = 1
    cn = np.asarray(cg.normal)[:, coarse_face]
    cnorm = float(np.linalg.norm(cn))
    if not math.isfinite(cnorm) or abs(cnorm-1.0) > 1e-12:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse canonical normal is invalid")
    fine_signs = []
    fine_length = 0.0
    for cell, face in zip(fine_cells, faces):
        if (not 0 <= cell < fine_execution.cell_count or
                not 0 <= face < fine_execution.face_count or
                int(fg.owner[face]) != cell or
                int(fg.neighbour[face]) >= 0 or
                int(fg.boundary_index[face]) >= 0):
            raise CoupledAMRRegisterCPUV98Error(
                "v98 fine interface must be an internal-coupling trace, not a physical boundary")
        normal = np.asarray(fg.normal)[:, face]
        # The fine owner outward normal points toward the coarse leaf, hence
        # opposite the canonical coarse-to-fine direction.
        if not np.allclose(normal, -cn, rtol=0.0, atol=1e-12):
            raise CoupledAMRRegisterCPUV98Error(
                "v98 coarse/fine interface normals or orientation differ")
        fine_signs.append(-1)
        fine_length += float(fg.face_length_m[face])
    coarse_length = float(cg.face_length_m[coarse_face])
    tolerance = 256.0*np.finfo(np.float64).eps*max(
        1.0, abs(coarse_length), abs(fine_length))
    if abs(coarse_length-fine_length) > tolerance:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 two fine face lengths do not partition the coarse face")
    area = float(cg.cell_area_m2[coarse_leaf_cell])
    if not math.isfinite(area) or area <= 0.0:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse leaf area is invalid")
    normal = _frozen(cn, (2,), "canonical normal")
    placeholder = CoupledAMRInterfaceMapCPUV98(
        coarse_execution.fingerprint, fine_execution.fingerprint,
        coarse_leaf_cell, coarse_face, fine_cells,
        faces, coarse_sign, tuple(fine_signs), coarse_global_face_id,
        global_ids, normal, area, "")
    return CoupledAMRInterfaceMapCPUV98(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_interface_payload(placeholder))})


def validate_coupled_amr_interface_map_cpu_v98(
    value: CoupledAMRInterfaceMapCPUV98,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRInterfaceMapCPUV98 or
            value.coarse_execution_fingerprint !=
                coarse_execution.fingerprint or
            value.fine_execution_fingerprint != fine_execution.fingerprint or
            value.fingerprint != _hash(_interface_payload(value))):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 interface map is stale or cross-execution")
    rebuilt = resolve_coupled_amr_interface_map_cpu_v98(
        coarse_execution, fine_execution,
        coarse_leaf_cell=value.coarse_leaf_cell,
        coarse_face=value.coarse_face,
        fine_interface_cells=value.fine_interface_cells,
        fine_faces=value.fine_faces,
        coarse_global_face_id=value.coarse_global_face_id,
        fine_global_face_ids=value.fine_global_face_ids)
    if rebuilt.fingerprint != value.fingerprint:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 rebuilt interface map differs")


@dataclass(frozen=True)
class CoupledAMRInternalInterfaceLedgerV98:
    """Signed non-physical interface amounts applied to one coarse leaf.

    These lanes are intentionally separate from every legacy named physical
    boundary ledger.  Positive values are the additive corrections to the
    provisional coarse conserved *amounts* before division by leaf area.
    """

    interface_fingerprint: str
    scheduled_register_sha256: str
    coarse_integrals_sha256: str
    fine_integrals_sha256: tuple[str, str]
    water_volume_m3: float
    momentum_m4_s: np.ndarray
    suspended_mass_kg: np.ndarray
    fluid_integral: np.ndarray
    bed_mass_kg: np.ndarray
    physical_boundary_ledger_count: int
    application_count: int
    receipt_sha256: str


def _internal_ledger_payload(
    value: CoupledAMRInternalInterfaceLedgerV98,
) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":internal-interface-ledger",
        "interface": value.interface_fingerprint,
        "scheduled_register": value.scheduled_register_sha256,
        "coarse_integrals": value.coarse_integrals_sha256,
        "fine_integrals": list(value.fine_integrals_sha256),
        "arrays_sha256": _digest(
            np.asarray(value.water_volume_m3), value.momentum_m4_s,
            value.suspended_mass_kg, value.fluid_integral,
            value.bed_mass_kg),
        "physical_boundary_ledger_count":
            value.physical_boundary_ledger_count,
        "application_count": value.application_count,
        "semantic": "real_internal_family_not_physical_boundary",
    }


@dataclass(frozen=True)
class CoupledAMRCoarseCorrectionCPUV98:
    """One coarse-cell correction replacing its interface integral once."""

    interface_fingerprint: str
    coarse_integrals_sha256: str
    fine_integrals_sha256: tuple[str, str]
    coarse_parent_state_sha256: str
    fine_parent_state_sha256: str
    fine_final_state_sha256: str
    scheduled_register: CoupledAMRTwoPhaseRegisterV98
    internal_interface_ledger: CoupledAMRInternalInterfaceLedgerV98
    water_depth_delta_m: float
    momentum_delta_m2_s: np.ndarray
    suspended_delta_kg_m2: np.ndarray
    fluid_delta: np.ndarray
    bed_delta_kg_m2: np.ndarray
    corrected_face_water_flux_m2_s: np.ndarray
    corrected_face_bedload_flux_kg_m_s: np.ndarray
    coarse_replacement_count: int
    fine_state_correction_count: int
    density_reflux_count: int
    receipt_sha256: str


def _correction_payload(value: CoupledAMRCoarseCorrectionCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":coarse-correction",
        "interface": value.interface_fingerprint,
        "coarse_integrals": value.coarse_integrals_sha256,
        "fine_integrals": list(value.fine_integrals_sha256),
        "coarse_parent": value.coarse_parent_state_sha256,
        "fine_parent": value.fine_parent_state_sha256,
        "fine_final": value.fine_final_state_sha256,
        "scheduled_register": value.scheduled_register.receipt_sha256,
        "internal_interface_ledger":
            value.internal_interface_ledger.receipt_sha256,
        "arrays_sha256": _digest(
            np.asarray(value.water_depth_delta_m),
            value.momentum_delta_m2_s,
            value.suspended_delta_kg_m2,
            value.fluid_delta, value.bed_delta_kg_m2,
            value.corrected_face_water_flux_m2_s,
            value.corrected_face_bedload_flux_kg_m_s),
        "coarse_replacement_count": value.coarse_replacement_count,
        "fine_state_correction_count": value.fine_state_correction_count,
        "density_reflux_count": value.density_reflux_count,
        "split_policy": "retain_coarse_local_sources_rebuild_derived_fields",
    }


def build_coupled_amr_coarse_correction_cpu_v98(
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse: CoupledFaceTimeIntegralsCPUV98,
    fine_steps: Sequence[CoupledFaceTimeIntegralsCPUV98], *,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> CoupledAMRCoarseCorrectionCPUV98:
    """Replace one coarse integral by two fine-face/two-substep integrals."""
    validate_coupled_amr_interface_map_cpu_v98(
        interface, coarse_execution, fine_execution)
    fine = tuple(fine_steps)
    _validate_reflux_sequence(
        coarse, fine, coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    scheduled = _build_scheduled_register(interface, coarse, fine)
    water_raw, momentum_raw, suspended_raw, fluid_raw, bed_raw = \
        _expected_correction_arrays(interface, coarse, fine)
    area = interface.coarse_cell_area_m2
    water = float(water_raw/area)
    momentum = _frozen(
        momentum_raw/area,
        (2,), "momentum correction")
    suspended = _frozen(
        suspended_raw/area,
        (coarse_execution.fraction_count,), "suspended correction")
    fluid = _frozen(
        fluid_raw/area,
        (2,), "fluid correction")
    bed = _frozen(bed_raw/area,
                  (coarse_execution.fraction_count,), "bed correction")
    ledger0 = CoupledAMRInternalInterfaceLedgerV98(
        interface.fingerprint, scheduled.receipt_sha256,
        coarse.receipt_sha256,
        (fine[0].receipt_sha256, fine[1].receipt_sha256),
        float(water_raw),
        _frozen(momentum_raw, (2,), "internal momentum ledger"),
        _frozen(suspended_raw, (coarse_execution.fraction_count,),
                "internal suspended ledger"),
        _frozen(fluid_raw, (2,), "internal fluid ledger"),
        _frozen(bed_raw, (coarse_execution.fraction_count,),
                "internal bed ledger"), 0, 1, "")
    ledger = CoupledAMRInternalInterfaceLedgerV98(
        **{**ledger0.__dict__, "receipt_sha256": _hash(
            _internal_ledger_payload(ledger0))})
    corrected_water_flux = np.array(
        coarse.water_volume_m3/coarse.dt_s/coarse.face_length_m,
        dtype=np.float64, order="C", copy=True)
    # Use the same canonical summation as validation (Python 3.12+ sum
    # compensates floats; a sequential += differs by an ulp in DT).
    fine_outward = sum(float(step.water_volume_m3[face])
                       for step in fine for face in interface.fine_faces)
    corrected_water_flux[interface.coarse_face] = (
        -fine_outward/(coarse.dt_s*
                       coarse.face_length_m[interface.coarse_face]))
    corrected_water_flux = _frozen(
        corrected_water_flux, (coarse_execution.face_count,),
        "corrected coarse water flux")
    corrected_bedload_flux = np.array(
        coarse.bedload_mass_kg/
        (coarse.dt_s*coarse.face_length_m[None, :]*
         coarse.morphological_factor),
        dtype=np.float64, order="C", copy=True)
    fine_bed_outward = np.zeros(
        coarse_execution.fraction_count, dtype=np.float64)
    for step in fine:
        for face in interface.fine_faces:
            fine_bed_outward += step.bedload_mass_kg[:, face]
    corrected_bedload_flux[:, interface.coarse_face] = (
        -fine_bed_outward/
        (coarse.dt_s*coarse.face_length_m[interface.coarse_face]*
         coarse.morphological_factor))
    corrected_bedload_flux = _frozen(
        corrected_bedload_flux,
        (coarse_execution.fraction_count, coarse_execution.face_count),
        "corrected coarse bedload flux")
    placeholder = CoupledAMRCoarseCorrectionCPUV98(
        interface.fingerprint, coarse.receipt_sha256,
        (fine[0].receipt_sha256, fine[1].receipt_sha256),
        coarse.parent_state_sha256, fine[0].parent_state_sha256,
        fine[1].candidate_state_sha256, scheduled, ledger,
        water, momentum, suspended,
        fluid, bed, corrected_water_flux, corrected_bedload_flux,
        1, 0, 0, "")
    result = CoupledAMRCoarseCorrectionCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_correction_payload(placeholder))})
    validate_coupled_amr_coarse_correction_cpu_v98(
        result, interface, coarse, fine,
        coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    return result


def _expected_correction_arrays(
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse: CoupledFaceTimeIntegralsCPUV98,
    fine: Sequence[CoupledFaceTimeIntegralsCPUV98],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct coarse-minus-fine interface amounts independently."""
    cf = interface.coarse_face
    ff = interface.fine_faces
    cs = interface.coarse_to_canonical
    fs = interface.fine_to_canonical

    def hydraulic_side(value, face: int, sign: int) -> np.ndarray:
        if sign not in (-1, 1):
            raise CoupledAMRRegisterCPUV98Error(
                "v98 interface orientation must be signed unit")
        # Both coarse and fine trace cells are the owner of their respective
        # negative-neighbour faces.  ``sign`` only rotates that owner-outward
        # contribution into the common coarse-to-fine orientation; it never
        # selects the nonexistent exterior side.
        augmented = value.augmented_left_face_integral[:, face]
        pressure = value.owner_hydrostatic_momentum_integral_m4_s[:, face]
        # The v91 face value is already length-integrated and its cell rate
        # uses +div(flux).  The equivalent outward conservative contribution
        # is therefore augmented-minus-pressure-minus-viscous.
        return np.ascontiguousarray(np.concatenate((
            np.atleast_1d(augmented[0]),
            augmented[1:]-pressure-
                value.viscous_momentum_integral_m4_s[:, face],
        ))*sign)

    def selected(name: str) -> np.ndarray:
        coarse_value = np.asarray(getattr(coarse, name))[..., cf]*cs
        fine_value = np.zeros_like(coarse_value, dtype=np.float64)
        for step in fine:
            for face, sign in zip(ff, fs):
                fine_value = fine_value + \
                    np.asarray(getattr(step, name))[..., face]*sign
        return np.asarray(coarse_value-fine_value, dtype=np.float64)

    coarse_hydraulic = hydraulic_side(coarse, cf, cs)
    fine_hydraulic = np.zeros(3, dtype=np.float64)
    for step in fine:
        for face, sign in zip(ff, fs):
            fine_hydraulic += hydraulic_side(step, face, sign)

    return (
        coarse_hydraulic[0]-fine_hydraulic[0],
        coarse_hydraulic[1:]-fine_hydraulic[1:],
        selected("suspended_advective_mass_kg")+
            selected("suspended_diffusive_mass_kg"),
        selected("fluid_advective_integral")+
            selected("fluid_diffusive_integral"),
        selected("bedload_mass_kg")+selected("selected_suspended_bed_mass_kg"),
    )


def _build_scheduled_register(
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse: CoupledFaceTimeIntegralsCPUV98,
    fine: Sequence[CoupledFaceTimeIntegralsCPUV98],
) -> CoupledAMRTwoPhaseRegisterV98:
    """Bind actual trial arrays to the backend-neutral two-phase schedule."""
    cf = interface.coarse_face
    ff = interface.fine_faces
    cs = interface.coarse_to_canonical
    fs = interface.fine_to_canonical

    def hydraulic_vector(value, face: int, sign: int) -> np.ndarray:
        if sign not in (-1, 1):
            raise CoupledAMRRegisterCPUV98Error(
                "v98 interface orientation must be signed unit")
        augmented = value.augmented_left_face_integral[:, face]
        pressure = value.owner_hydrostatic_momentum_integral_m4_s[:, face]
        return np.ascontiguousarray(np.concatenate((
            np.atleast_1d(augmented[0]),
            augmented[1:]-pressure-
                value.viscous_momentum_integral_m4_s[:, face],
        ))*sign)

    hydraulic_lanes = ("water_volume_m3", "momentum_x_m4_s",
                       "momentum_y_m4_s")
    coarse_hydraulic = hydraulic_vector(coarse, cf, cs)
    fine_hydraulic = np.empty((2, 2, len(hydraulic_lanes)))
    for step in range(2):
        for slot, (face, sign) in enumerate(zip(ff, fs)):
            fine_hydraulic[step, slot] = hydraulic_vector(
                fine[step], face, sign)

    def scalar_vector(value, face, sign):
        suspended = (
            np.asarray(value.suspended_advective_mass_kg)[..., face]+
            np.asarray(value.suspended_diffusive_mass_kg)[..., face])*sign
        fluid = (
            np.asarray(value.fluid_advective_integral)[..., face]+
            np.asarray(value.fluid_diffusive_integral)[..., face])*sign
        bed = np.asarray(value.bedload_mass_kg)[..., face]*sign
        return np.ascontiguousarray(np.concatenate((suspended, fluid, bed)))

    ng = coarse.suspended_advective_mass_kg.shape[0]
    scalar_lanes = tuple(
        [f"suspended_{index}_kg" for index in range(ng)]+[
            "hT_integral", "hS_integral"]+[
            f"bed_{index}_kg" for index in range(ng)])
    coarse_scalar = scalar_vector(coarse, cf, cs)
    fine_scalar = np.empty((2, 2, len(scalar_lanes)))
    for step in range(2):
        for slot, (face, sign) in enumerate(zip(ff, fs)):
            fine_scalar[step, slot] = scalar_vector(
                fine[step], face, sign)
    common = dict(
        interface_fingerprint=interface.fingerprint,
        fine_source_sha256=(fine[0].receipt_sha256,
                            fine[1].receipt_sha256),
        time_start_s=coarse.time_start_s, coarse_dt_s=coarse.dt_s,
        fine_dt_s=(fine[0].dt_s, fine[1].dt_s),
        coarse_cell_area_m2=interface.coarse_cell_area_m2)
    hydraulic = resolve_scheduled_register_phase_v98(
        phase=PHASE_HYDRAULIC, lane_ids=hydraulic_lanes,
        coarse_source_sha256=coarse.receipt_sha256,
        coarse_integral=coarse_hydraulic,
        fine_integral_slots=fine_hydraulic, **common)
    scalar_bed = resolve_scheduled_register_phase_v98(
        phase=PHASE_SCALAR_BED, lane_ids=scalar_lanes,
        coarse_source_sha256=coarse.receipt_sha256,
        coarse_integral=coarse_scalar,
        fine_integral_slots=fine_scalar, **common)
    return compose_two_phase_register_v98(
        coarse.parent_state_sha256, hydraulic, scalar_bed)


def _validate_reflux_sequence(
    coarse: CoupledFaceTimeIntegralsCPUV98,
    fine: Sequence[CoupledFaceTimeIntegralsCPUV98], *,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    validate_coupled_face_time_integrals_cpu_v98(coarse, coarse_execution)
    if len(fine) != 2:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 requires exactly two fine substep integral receipts")
    for value in fine:
        validate_coupled_face_time_integrals_cpu_v98(value, fine_execution)
    if (coarse.dt_s != 2.0*fine[0].dt_s or
            fine[1].dt_s != fine[0].dt_s or
            fine[0].morphological_factor != coarse.morphological_factor or
            fine[1].morphological_factor != coarse.morphological_factor or
            fine[0].candidate_state_sha256 != fine[1].parent_state_sha256 or
            fine[0].candidate_accepted_updates !=
                fine[1].parent_accepted_updates or
            fine[0].time_end_s != fine[1].time_start_s or
            coarse.time_start_s != fine[0].time_start_s or
            coarse.time_end_s != fine[1].time_end_s):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse/fine trial clocks or private fine staging differ")


def validate_coupled_amr_coarse_correction_cpu_v98(
    value: CoupledAMRCoarseCorrectionCPUV98,
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse: CoupledFaceTimeIntegralsCPUV98,
    fine_steps: Sequence[CoupledFaceTimeIntegralsCPUV98], *,
    coarse_execution: CoupledWetDryCPUExecutionV97,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    if (type(value) is not CoupledAMRCoarseCorrectionCPUV98 or
            value.interface_fingerprint != interface.fingerprint or
            value.coarse_integrals_sha256 != coarse.receipt_sha256 or
            value.fine_integrals_sha256 !=
                tuple(item.receipt_sha256 for item in fine_steps) or
            value.coarse_replacement_count != 1 or
            value.fine_state_correction_count != 0 or
            value.density_reflux_count != 0 or
            value.receipt_sha256 != _hash(_correction_payload(value))):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse correction identity or owner counts differ")
    fine = tuple(fine_steps)
    validate_coupled_amr_interface_map_cpu_v98(
        interface, coarse_execution, fine_execution)
    _validate_reflux_sequence(
        coarse, fine, coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    validate_two_phase_register_v98(value.scheduled_register)
    scheduled = _build_scheduled_register(interface, coarse, fine)
    if value.scheduled_register.receipt_sha256 != scheduled.receipt_sha256:
        raise CoupledAMRRegisterCPUV98Error(
            "v98 CPU correction differs from the scheduled register")
    raw = _expected_correction_arrays(interface, coarse, fine)
    ledger = value.internal_interface_ledger
    if (type(ledger) is not CoupledAMRInternalInterfaceLedgerV98 or
            ledger.interface_fingerprint != interface.fingerprint or
            ledger.scheduled_register_sha256 != scheduled.receipt_sha256 or
            ledger.coarse_integrals_sha256 != coarse.receipt_sha256 or
            ledger.fine_integrals_sha256 !=
                tuple(item.receipt_sha256 for item in fine) or
            ledger.physical_boundary_ledger_count != 0 or
            ledger.application_count != 1 or
            ledger.receipt_sha256 != _hash(_internal_ledger_payload(ledger)) or
            float(ledger.water_volume_m3) != float(raw[0]) or
            any(not np.array_equal(actual, expected) for actual, expected in (
                (ledger.momentum_m4_s, raw[1]),
                (ledger.suspended_mass_kg, raw[2]),
                (ledger.fluid_integral, raw[3]),
                (ledger.bed_mass_kg, raw[4])))):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 internal-interface ledger identity differs")
    area = interface.coarse_cell_area_m2
    expected_arrays = (raw[1]/area, raw[2]/area, raw[3]/area, raw[4]/area)
    if (float(value.water_depth_delta_m) != float(raw[0]/area) or
            any(not np.array_equal(actual, expected)
                for actual, expected in zip((
                    value.momentum_delta_m2_s,
                    value.suspended_delta_kg_m2,
                    value.fluid_delta, value.bed_delta_kg_m2),
                    expected_arrays))):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 coarse correction differs from source face integrals")
    scheduled_hydraulic = (
        value.scheduled_register.hydraulic.coarse_cell_correction)
    scheduled_scalar = (
        value.scheduled_register.scalar_bed.coarse_cell_correction)
    joined = np.concatenate((
        np.atleast_1d(value.water_depth_delta_m),
        value.momentum_delta_m2_s))
    joined_scalar = np.concatenate((
        value.suspended_delta_kg_m2, value.fluid_delta,
        value.bed_delta_kg_m2))
    # The scheduled BED lane is Q3-only. DT reuses the already-owned SUS
    # register correction as an additional suspended-bed contribution.
    expected_scalar = np.array(scheduled_scalar, copy=True)
    if coarse.suspended_bed_divergence_selected:
        expected_scalar[-coarse_execution.fraction_count:] += scheduled_scalar[:coarse_execution.fraction_count]
    if (not np.array_equal(joined, scheduled_hydraulic) or
            not np.allclose(joined_scalar, expected_scalar, rtol=0., atol=1e-13)):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 typed correction and scheduled lane order differ")
    expected_face_flux = np.array(
        coarse.water_volume_m3/coarse.dt_s/coarse.face_length_m,
        dtype=np.float64, order="C", copy=True)
    fine_outward = sum(
        float(step.water_volume_m3[face])
        for step in fine for face in interface.fine_faces)
    expected_face_flux[interface.coarse_face] = (
        -fine_outward/(coarse.dt_s*
                       coarse.face_length_m[interface.coarse_face]))
    if (value.corrected_face_water_flux_m2_s.shape !=
            (coarse_execution.face_count,) or
            value.corrected_face_water_flux_m2_s.flags.writeable or
            not np.array_equal(value.corrected_face_water_flux_m2_s,
                               expected_face_flux)):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 corrected accepted water face flux differs; "
            f"linf={np.max(np.abs(value.corrected_face_water_flux_m2_s-expected_face_flux)):.17g}; "
            f"mutable={value.corrected_face_water_flux_m2_s.flags.writeable}")
    expected_bed_flux = np.array(
        coarse.bedload_mass_kg/
        (coarse.dt_s*coarse.face_length_m[None, :]*
         coarse.morphological_factor),
        dtype=np.float64, order="C", copy=True)
    fine_bed_outward = sum(
        (step.bedload_mass_kg[:, face]
         for step in fine for face in interface.fine_faces),
        start=np.zeros(coarse_execution.fraction_count, dtype=np.float64))
    expected_bed_flux[:, interface.coarse_face] = (
        -fine_bed_outward/
        (coarse.dt_s*coarse.face_length_m[interface.coarse_face]*
         coarse.morphological_factor))
    if (value.corrected_face_bedload_flux_kg_m_s.shape !=
            (coarse_execution.fraction_count,
             coarse_execution.face_count) or
            value.corrected_face_bedload_flux_kg_m_s.flags.writeable or
            not np.array_equal(value.corrected_face_bedload_flux_kg_m_s,
                               expected_bed_flux)):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 corrected accepted bedload face flux differs")
    arrays = (
        (value.momentum_delta_m2_s, (2,), "momentum correction"),
        (value.suspended_delta_kg_m2,
         (coarse_execution.fraction_count,), "suspended correction"),
        (value.fluid_delta, (2,), "fluid correction"),
        (value.bed_delta_kg_m2,
         (coarse_execution.fraction_count,), "bed correction"),
    )
    if not math.isfinite(value.water_depth_delta_m):
        raise CoupledAMRRegisterCPUV98Error(
            "v98 water correction is non-finite")
    for array, shape, name in arrays:
        checked = _frozen(array, shape, name)
        if not np.array_equal(checked, array):
            raise CoupledAMRRegisterCPUV98Error(
                f"v98 {name} changed during validation")


__all__ = [
    "BACKEND", "CoupledAMRCoarseCorrectionCPUV98",
    "CoupledAMRInternalInterfaceLedgerV98",
    "CoupledAMRInterfaceMapCPUV98", "CoupledAMRRegisterCPUV98Error",
    "CoupledFaceTimeIntegralsCPUV98", "SCHEMA", "SCOPE_ID",
    "build_coupled_amr_coarse_correction_cpu_v98",
    "extract_coupled_face_time_integrals_cpu_v98",
    "resolve_coupled_amr_interface_map_cpu_v98",
    "validate_coupled_amr_coarse_correction_cpu_v98",
    "validate_coupled_amr_interface_map_cpu_v98",
    "validate_coupled_face_time_integrals_cpu_v98",
]
