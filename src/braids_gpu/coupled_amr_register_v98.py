"""Actual-CUDA face-time integral evidence for the first coupled AMR gate.

This additive v98 adapter consumes one still-open, fully validated v97
transaction.  It reads the exact resident augmented-f-wave side images left
by the final nonlinear residual and the exact resident v92/Q3 face images.
It does not run a second Riemann solve, correct a state, publish a clock, or
claim that a coarse/fine trace owner exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from braids_core.sediment.amr_suspended_bed_selection_v100 import mode_v100

from braids_core.sediment.coupled_amr_register_lifecycle_v98 import (
    CoupledAMRTwoPhaseRegisterV98,
    PHASE_HYDRAULIC,
    PHASE_SCALAR_BED,
    compose_two_phase_register_v98,
    resolve_scheduled_register_phase_v98,
    validate_two_phase_register_v98,
)

from braids_core.sediment.transported_fluid_settling_v92 import (
    FLUID_SCALAR_COUNT,
    FLUID_TEMPERATURE,
    TEMPERATURE_FLOOR_C,
)
from braids_gpu.active_suspended_control_v94 import (
    ResidentQ3V83ActiveFaceTrialV94,
)
from braids_gpu.coupled_amr_geometry_v98 import (
    CoupledAMRGeometryBindingGPUV98,
    validate_coupled_amr_geometry_binding_gpu_v98,
)
from braids_gpu.coupled_wetdry_v97 import (
    CoupledWetDryGPUExecutionV97,
    CoupledWetDryGPUStateV97,
    CoupledWetDryGPUTrialV97,
    ResidentCoupledWetDryLifecycleV97,
    _grid,
)


SCHEMA = 1
SCOPE_ID = "braids.coupled_amr_register.actual_cuda.v98"
BACKEND = "actual_cuda"
PRIVATE_BACKEND = "actual_cuda_private_amr"
SOURCE_OPEN_V97 = "open_v97_transaction"
SOURCE_PRIVATE_FINE = "private_v98_fine_substep"


class CoupledAMRRegisterGPUV98Error(ValueError):
    """Fail-closed actual-CUDA v98 integral-evidence error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(str(array.dtype).encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _sha(value: object, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise CoupledAMRRegisterGPUV98Error(
            f"v98 CUDA {name} must be a lowercase SHA-256 digest")
    return value


def _frozen(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all()):
        raise CoupledAMRRegisterGPUV98Error(
            f"v98 CUDA {name} must be finite with shape {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _source_sha256() -> str:
    path = Path(__file__).resolve()
    if not path.is_file() or path.is_symlink():
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA extractor source identity differs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_fine_source_sha256() -> str:
    path = Path(__file__).resolve().with_name(
        "coupled_amr_fine_substep_v98.py")
    if not path.is_file() or path.is_symlink():
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA private-fine source identity differs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class CoupledFaceTimeIntegralsGPUV98:
    """Authenticated conserved face-time images from one actual v97 trial."""

    execution_fingerprint: str
    parent_state_sha256: str
    candidate_state_sha256: str
    source_trial_sha256: str
    hydraulic_trial_sha256: str
    hydraulic_runtime_receipt_sha256: str
    scalar_runtime_receipt_sha256: str
    q3_face_trial_sha256: str
    extractor_source_sha256: str
    source_kind: str
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
    hydraulic_side_evaluation_count: int
    discarded_provisional_side_count: int
    accepted_side_replacement_count: int
    resident_hydraulic_readback_count: int
    resident_scalar_readback_count: int
    resident_q3_readback_count: int
    backend: str
    receipt_sha256: str
    suspended_bed_divergence_selected: bool = False

    @property
    def selected_suspended_bed_mass_kg(self):
        return (self.suspended_advective_mass_kg+self.suspended_diffusive_mass_kg
                if self.suspended_bed_divergence_selected else
                np.zeros_like(self.suspended_advective_mass_kg))


def _payload(value: CoupledFaceTimeIntegralsGPUV98) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":face-time-integrals",
        "suspended_bed_divergence_selected": value.suspended_bed_divergence_selected,
        "execution_fingerprint": value.execution_fingerprint,
        "parent_state_sha256": value.parent_state_sha256,
        "candidate_state_sha256": value.candidate_state_sha256,
        "source_trial_sha256": value.source_trial_sha256,
        "hydraulic_trial_sha256": value.hydraulic_trial_sha256,
        "hydraulic_runtime_receipt_sha256":
            value.hydraulic_runtime_receipt_sha256,
        "scalar_runtime_receipt_sha256":
            value.scalar_runtime_receipt_sha256,
        "q3_face_trial_sha256": value.q3_face_trial_sha256,
        "extractor_source_sha256": value.extractor_source_sha256,
        "source_kind": value.source_kind,
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
        "hydraulic_side_evaluation_count":
            value.hydraulic_side_evaluation_count,
        "discarded_provisional_side_count":
            value.discarded_provisional_side_count,
        "accepted_side_replacement_count":
            value.accepted_side_replacement_count,
        "resident_hydraulic_readback_count":
            value.resident_hydraulic_readback_count,
        "resident_scalar_readback_count":
            value.resident_scalar_readback_count,
        "resident_q3_readback_count": value.resident_q3_readback_count,
        "rate_integration": "face_length_times_accepted_dt_once",
        "viscous_integration":
            "already_face_length_integrated_times_accepted_dt_once",
        "diffusive_integration": "already_integrated_no_second_dt",
        "hydraulic_momentum":
            "resident_augmented_left_right_plus_local_pressure_ledger",
        "state_or_clock_publication_count": 0,
        "automatic_cpu_fallback": False,
        "backend": value.backend,
    }


def validate_coupled_face_time_integrals_gpu_v98(
    value: CoupledFaceTimeIntegralsGPUV98,
    execution: CoupledWetDryGPUExecutionV97,
) -> None:
    if type(value) is not CoupledFaceTimeIntegralsGPUV98:
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA face-time integral receipt type differs")
    nf, ng, nc = (execution.face_count, execution.fraction_count,
                  execution.cell_count)
    arrays = (
        (value.face_length_m, (nf,), "face length"),
        (value.midpoint_depth_m, (nc,), "midpoint depth"),
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
        (value.fluid_advective_integral, (FLUID_SCALAR_COUNT, nf),
         "fluid advective integral"),
        (value.fluid_diffusive_integral, (FLUID_SCALAR_COUNT, nf),
         "fluid diffusive integral"),
        (value.bedload_mass_kg, (ng, nf), "bedload integral"),
    )
    for array, shape, name in arrays:
        checked = _frozen(array, shape, name)
        if (not np.array_equal(checked, array) or array.flags.writeable or
                not array.flags.c_contiguous):
            raise CoupledAMRRegisterGPUV98Error(
                f"v98 CUDA {name} changed during validation")
    grid = _grid(execution.active_execution)
    if value.source_kind == SOURCE_OPEN_V97:
        expected_backend = BACKEND
        expected_source = _source_sha256()
    elif value.source_kind == SOURCE_PRIVATE_FINE:
        expected_backend = PRIVATE_BACKEND
        expected_source = _private_fine_source_sha256()
    else:
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA integral source kind differs")
    if (value.suspended_bed_divergence_selected !=
            (mode_v100(execution.active_execution.base_execution) == 1) or
            not np.array_equal(value.face_length_m, grid.face_length_m) or
            value.execution_fingerprint != execution.fingerprint or
            value.parent_accepted_updates < 0 or
            value.candidate_accepted_updates !=
                value.parent_accepted_updates+1 or
            not math.isfinite(value.time_start_s) or
            not math.isfinite(value.time_end_s) or
            not math.isfinite(value.dt_s) or value.dt_s <= 0.0 or
            not math.isfinite(value.morphological_factor) or
            value.morphological_factor <= 0.0 or
            value.time_end_s != value.time_start_s+value.dt_s or
            value.hydraulic_side_evaluation_count < 1 or
            value.discarded_provisional_side_count !=
                value.hydraulic_side_evaluation_count-1 or
            value.accepted_side_replacement_count != 1 or
            value.resident_hydraulic_readback_count != 1 or
            value.resident_scalar_readback_count != 1 or
            value.resident_q3_readback_count != 1 or
            value.backend != expected_backend or
            value.extractor_source_sha256 != expected_source or
            value.receipt_sha256 != _hash(_payload(value))):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA integral identity, clock, count, or backend differs")
    for raw, name in (
            (value.parent_state_sha256, "parent state"),
            (value.candidate_state_sha256, "candidate state"),
            (value.source_trial_sha256, "source trial"),
            (value.hydraulic_trial_sha256, "hydraulic trial"),
            (value.hydraulic_runtime_receipt_sha256,
             "hydraulic runtime receipt"),
            (value.scalar_runtime_receipt_sha256,
             "scalar runtime receipt"),
            (value.q3_face_trial_sha256, "Q3 face trial")):
        _sha(raw, name)
    if (not np.array_equal(value.augmented_left_face_integral[0],
                           value.water_volume_m3) or
            not np.array_equal(value.augmented_right_face_integral[0],
                               value.water_volume_m3) or
            not np.array_equal(
                value.augmented_momentum_jump_integral_m4_s,
                value.augmented_right_face_integral[1:]-
                value.augmented_left_face_integral[1:])):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA augmented side/water/source-jump identity differs")
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal, dtype=np.float64)
    length_dt = value.face_length_m*value.dt_s
    owner_pressure = (
        (0.5*grid.profile.gravity_m_s2*
         value.midpoint_depth_m[owner]**2)[None, :]*normal*
        length_dt[None, :])
    neighbour_pressure = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    neighbour_pressure[:, internal] = (
        0.5*grid.profile.gravity_m_s2*
        value.midpoint_depth_m[neighbour[internal]]**2
    )[None, :]*normal[:, internal]*length_dt[internal]
    if (not np.array_equal(
            value.owner_hydrostatic_momentum_integral_m4_s,
            owner_pressure) or
            not np.array_equal(
                value.neighbour_hydrostatic_momentum_integral_m4_s,
                neighbour_pressure)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA local hydrostatic-pressure ledger differs")


def _resident_scalar_face_images(runtime, trial):
    """Read each final v92 resident face family once and authenticate it."""
    cp = runtime.cp
    combined = runtime.active.inner.combined
    v92 = trial.active_trial.base_v93_trial.base_v92_trial
    suspended = v92.suspended_trial.base_trial
    fluid = v92.fluid_transport
    buffers = combined.advection.buffers
    nsed, nf = combined.nsed, combined.nf
    suspended_advective = cp.asnumpy(buffers["flux"][:nsed])
    suspended_diffusive = cp.asnumpy(
        combined.diffusive_face_mass[:nsed])
    if (not np.array_equal(suspended_advective,
                           suspended.advective_face_flux_kg_m_s) or
            not np.array_equal(suspended_diffusive,
                               suspended.diffusive_face_mass_owner_out_kg)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA resident suspended face image differs")
    fluid_advective = np.zeros((FLUID_SCALAR_COUNT, nf), dtype=np.float64)
    fluid_diffusive = np.zeros_like(fluid_advective)
    water = trial.hydraulic_trial.accepted_limited_face_water_flux_m2_s
    for local, lane in enumerate(combined.active_fluid):
        row = nsed+local
        fluid_advective[lane] = cp.asnumpy(buffers["flux"][row])
        fluid_diffusive[lane] = cp.asnumpy(
            combined.diffusive_face_mass[row])
        if lane == FLUID_TEMPERATURE:
            fluid_advective[lane] += TEMPERATURE_FLOOR_C*water
    if (not np.array_equal(fluid_advective,
                           fluid.advective_face_flux) or
            not np.array_equal(fluid_diffusive,
                               fluid.diffusive_face_amount)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA resident transported-fluid face image differs")
    return tuple(np.ascontiguousarray(value) for value in (
        suspended_advective, suspended_diffusive,
        fluid_advective, fluid_diffusive))


def extract_coupled_face_time_integrals_gpu_v98(
    runtime: ResidentCoupledWetDryLifecycleV97,
    execution: CoupledWetDryGPUExecutionV97,
    parent: CoupledWetDryGPUStateV97,
    trial: CoupledWetDryGPUTrialV97,
) -> CoupledFaceTimeIntegralsGPUV98:
    """Extract actual face integrals without a second numerical operation."""
    if (type(runtime) is not ResidentCoupledWetDryLifecycleV97 or
            type(execution) is not CoupledWetDryGPUExecutionV97 or
            type(parent) is not CoupledWetDryGPUStateV97 or
            type(trial) is not CoupledWetDryGPUTrialV97 or
            runtime.execution is not execution or
            runtime._parent is not parent or runtime._trial is not trial):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA extraction requires the exact open v97 transaction")
    runtime._validate_trial(parent, trial)
    hydraulic_runtime = runtime.hydraulic
    hydraulic_runtime._validate_runtime_identity()
    hydraulic = trial.hydraulic_trial
    if (hydraulic_runtime._v98_hydraulic_trial_sha256 !=
            hydraulic.trial_sha256 or
            hydraulic_runtime._v98_hydraulic_side_evaluation_count !=
            hydraulic.true_residual_evaluation_count):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA resident hydraulic side image is stale")
    cp = runtime.cp
    left = cp.asnumpy(hydraulic_runtime.v98_final_augmented_left_flux)
    right = cp.asnumpy(hydraulic_runtime.v98_final_augmented_right_flux)
    if (not np.array_equal(left[0],
                           hydraulic.accepted_limited_face_water_flux_m2_s) or
            not np.array_equal(right[0],
                               hydraulic.accepted_limited_face_water_flux_m2_s)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA accepted water/augmented side images differ")
    suspended_advective, suspended_diffusive, fluid_advective, \
        fluid_diffusive = _resident_scalar_face_images(runtime, trial)
    v93 = trial.active_trial.base_v93_trial
    face = getattr(v93.q3_open_trial, "face_trial", None)
    if type(face) is not ResidentQ3V83ActiveFaceTrialV94:
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA Q3 face trial concrete type differs")
    bedload = cp.asnumpy(face.face_flux_kg_m_s)
    grid = _grid(execution.active_execution)
    nf, ng = execution.face_count, execution.fraction_count
    dt = float(hydraulic.dt_s)
    if (float(v93.base_v92_trial.hydraulic_receipt.dt_s) != dt or
            float(face.dt_s) != dt):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA nested hydraulic/scalar/Q3 timesteps differ")
    length = _frozen(grid.face_length_m, (nf,), "face length")
    midpoint_depth = np.where(
        np.asarray(hydraulic.frozen_wet_mask, dtype=bool),
        np.asarray(hydraulic.midpoint_q[0]), 0.0)
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal, dtype=np.float64)
    owner_pressure = (0.5*grid.profile.gravity_m_s2*
                      midpoint_depth[owner]**2)[None, :]*normal
    neighbour_pressure = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    neighbour_pressure[:, internal] = (
        0.5*grid.profile.gravity_m_s2*
        midpoint_depth[neighbour[internal]]**2
    )[None, :]*normal[:, internal]
    length_dt = length*dt
    left_integral = left*length_dt[None, :]
    right_integral = right*length_dt[None, :]
    arrays = (
        length,
        midpoint_depth,
        hydraulic.accepted_limited_face_water_flux_m2_s*length_dt,
        left_integral,
        right_integral,
        owner_pressure*length_dt[None, :],
        neighbour_pressure*length_dt[None, :],
        right_integral[1:]-left_integral[1:],
        v93.base_v92_trial.momentum_face_flux_m4_s2*dt,
        suspended_advective*length[None, :]*dt,
        suspended_diffusive,
        fluid_advective*length[None, :]*dt,
        fluid_diffusive,
        bedload*length[None, :]*dt*float(face.morphological_factor),
    )
    shapes = ((nf,), (execution.cell_count,), (nf,),
              (3, nf), (3, nf), (2, nf), (2, nf), (2, nf),
              (2, nf), (ng, nf), (ng, nf),
              (FLUID_SCALAR_COUNT, nf), (FLUID_SCALAR_COUNT, nf),
              (ng, nf))
    names = ("face length", "midpoint depth", "water volume",
             "augmented left integral", "augmented right integral",
             "owner hydrostatic integral", "neighbour hydrostatic integral",
             "augmented momentum jump", "viscous momentum integral",
             "suspended advective mass", "suspended diffusive mass",
             "fluid advective integral", "fluid diffusive integral",
             "bedload mass")
    frozen = tuple(_frozen(array, shape, name)
                   for array, shape, name in zip(arrays, shapes, names))
    hydraulic_receipt = hydraulic_runtime.runtime_receipt["receipt_sha256"]
    scalar_receipt = _hash(runtime.active.inner.runtime_receipt)
    placeholder = CoupledFaceTimeIntegralsGPUV98(
        execution_fingerprint=execution.fingerprint,
        parent_state_sha256=parent.state_sha256,
        candidate_state_sha256=trial.candidate.state_sha256,
        source_trial_sha256=trial.trial_sha256,
        hydraulic_trial_sha256=hydraulic.trial_sha256,
        hydraulic_runtime_receipt_sha256=hydraulic_receipt,
        scalar_runtime_receipt_sha256=scalar_receipt,
        q3_face_trial_sha256=face.trial_sha256,
        extractor_source_sha256=_source_sha256(),
        source_kind=SOURCE_OPEN_V97,
        parent_accepted_updates=parent.accepted_updates,
        candidate_accepted_updates=trial.candidate.accepted_updates,
        time_start_s=parent.time_s,
        time_end_s=trial.candidate.time_s,
        dt_s=dt,
        morphological_factor=float(face.morphological_factor),
        face_length_m=frozen[0],
        midpoint_depth_m=frozen[1],
        water_volume_m3=frozen[2],
        augmented_left_face_integral=frozen[3],
        augmented_right_face_integral=frozen[4],
        owner_hydrostatic_momentum_integral_m4_s=frozen[5],
        neighbour_hydrostatic_momentum_integral_m4_s=frozen[6],
        augmented_momentum_jump_integral_m4_s=frozen[7],
        viscous_momentum_integral_m4_s=frozen[8],
        suspended_advective_mass_kg=frozen[9],
        suspended_diffusive_mass_kg=frozen[10],
        fluid_advective_integral=frozen[11],
        fluid_diffusive_integral=frozen[12],
        bedload_mass_kg=frozen[13],
        hydraulic_side_evaluation_count=
            hydraulic.true_residual_evaluation_count,
        discarded_provisional_side_count=
            hydraulic.true_residual_evaluation_count-1,
        accepted_side_replacement_count=1,
        resident_hydraulic_readback_count=1,
        resident_scalar_readback_count=1,
        resident_q3_readback_count=1,
        backend=BACKEND,
        receipt_sha256="",
        suspended_bed_divergence_selected=mode_v100(execution.active_execution.base_execution) == 1)
    result = CoupledFaceTimeIntegralsGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})
    validate_coupled_face_time_integrals_gpu_v98(result, execution)
    return result


def make_private_fine_face_time_integrals_gpu_v98(
    execution: CoupledWetDryGPUExecutionV97,
    parent: CoupledWetDryGPUStateV97,
    hydraulic_runtime: object,
    hydraulic_trial: object,
    scalar_runtime: object,
    scalar_result: object,
    q3_runtime: object,
    q3_result: object,
    candidate: CoupledWetDryGPUStateV97,
) -> CoupledFaceTimeIntegralsGPUV98:
    """Build the common register image from one authenticated private step.

    The inputs are the already-produced actual-CUDA hydraulic, scalar, and
    Q3 results.  This function performs no numerical launch and never opens a
    legacy v97 transaction merely to satisfy the historical extractor ABI.
    """
    # Lazy imports keep the register algebra independent of the fine-state
    # owner that consumes it, while still requiring exact reviewed classes.
    from braids_gpu.coupled_amr_fwave_v98 import (
        ResidentAMRInterfaceFwaveGPUV98,
        WetDryAMRInterfaceGPUTrialV98,
        validate_amr_interface_residual_audit_gpu_v98,
    )
    from braids_gpu.coupled_amr_q3_face_v98 import (
        AMRQ3FaceResultGPUV98,
        ResidentAMRQ3FaceGPUV98,
        validate_amr_q3_face_result_gpu_v98,
    )
    from braids_gpu.coupled_amr_scalar_exchange_v98 import (
        AMRScalarExchangeResultGPUV98,
        ResidentAMRScalarExchangeGPUV98,
        validate_amr_scalar_exchange_result_gpu_v98,
    )

    if (type(execution) is not CoupledWetDryGPUExecutionV97 or
            type(parent) is not CoupledWetDryGPUStateV97 or
            type(candidate) is not CoupledWetDryGPUStateV97 or
            type(hydraulic_runtime) is not
                ResidentAMRInterfaceFwaveGPUV98 or
            type(hydraulic_trial) is not
                WetDryAMRInterfaceGPUTrialV98 or
            type(scalar_runtime) is not
                ResidentAMRScalarExchangeGPUV98 or
            type(scalar_result) is not AMRScalarExchangeResultGPUV98 or
            type(q3_runtime) is not ResidentAMRQ3FaceGPUV98 or
            type(q3_result) is not AMRQ3FaceResultGPUV98 or
            hydraulic_runtime.coupled_execution is not execution or
            scalar_runtime.execution is not execution or
            q3_runtime.execution is not execution):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA private-fine extraction binding differs")
    hydraulic_runtime._validate_runtime()
    validate_amr_interface_residual_audit_gpu_v98(
        hydraulic_trial.residual_audit, hydraulic_runtime,
        hydraulic_trial.wetdry_trial)
    validate_amr_scalar_exchange_result_gpu_v98(
        scalar_result, scalar_runtime)
    validate_amr_q3_face_result_gpu_v98(q3_result, q3_runtime)
    trial = hydraulic_trial.wetdry_trial
    audit = hydraulic_trial.residual_audit
    if (hydraulic_trial.execution_fingerprint !=
            execution.hydraulic_execution.fingerprint or
            scalar_result.execution_fingerprint != execution.fingerprint or
            q3_result.execution_fingerprint != execution.fingerprint or
            candidate.accepted_updates != parent.accepted_updates+1 or
            candidate.time_s != parent.time_s+trial.dt_s or
            float(scalar_result.dt_s) != float(trial.dt_s) or
            float(q3_result.dt_s) != float(trial.dt_s)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA private-fine trial chain differs")

    grid = _grid(execution.active_execution)
    nf, ng = execution.face_count, execution.fraction_count
    left = np.asarray(audit.final_full_limited_left_flux)
    right = np.asarray(audit.final_full_limited_right_flux)
    water = np.asarray(trial.accepted_limited_face_water_flux_m2_s)
    if (left.shape != (3, nf) or right.shape != (3, nf) or
            not np.array_equal(left[0], water) or
            not np.array_equal(right[0], water)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA private-fine resident augmented flux differs")
    dt = float(trial.dt_s)
    length = np.asarray(grid.face_length_m)
    length_dt = length*dt
    midpoint_depth = np.where(
        np.asarray(trial.frozen_wet_mask, dtype=bool),
        np.asarray(trial.midpoint_q[0]), 0.0)
    owner = np.asarray(grid.owner, dtype=np.int64)
    neighbour = np.asarray(grid.neighbour, dtype=np.int64)
    normal = np.asarray(grid.normal, dtype=np.float64)
    pressure_owner = (
        0.5*grid.profile.gravity_m_s2*midpoint_depth[owner]**2
    )[None, :]*normal
    pressure_neighbour = np.zeros((2, nf), dtype=np.float64)
    internal = neighbour >= 0
    pressure_neighbour[:, internal] = (
        0.5*grid.profile.gravity_m_s2*
        midpoint_depth[neighbour[internal]]**2
    )[None, :]*normal[:, internal]
    left_integral = left*length_dt[None, :]
    right_integral = right*length_dt[None, :]
    morphological_factor = float(
        execution.active_execution.base_execution.morphological_factor)
    arrays = (
        length,
        midpoint_depth,
        water*length_dt,
        left_integral,
        right_integral,
        pressure_owner*length_dt[None, :],
        pressure_neighbour*length_dt[None, :],
        right_integral[1:]-left_integral[1:],
        np.asarray(trial.shared_turbulence_face_flux)*dt,
        np.asarray(scalar_result.suspended_advective_face_flux_kg_m_s)*
            length[None, :]*dt,
        np.asarray(
            scalar_result.suspended_diffusive_face_mass_owner_out_kg),
        np.asarray(scalar_result.fluid_result.advective_face_flux)*
            length[None, :]*dt,
        np.asarray(scalar_result.fluid_result.diffusive_face_amount),
        np.asarray(q3_result.face_flux_kg_m_s)*length[None, :]*dt*
            morphological_factor,
    )
    shapes = ((nf,), (execution.cell_count,), (nf,), (3, nf), (3, nf),
              (2, nf), (2, nf), (2, nf), (2, nf), (ng, nf),
              (ng, nf), (FLUID_SCALAR_COUNT, nf),
              (FLUID_SCALAR_COUNT, nf), (ng, nf))
    names = ("face length", "midpoint depth", "water volume",
             "augmented left integral", "augmented right integral",
             "owner hydrostatic integral",
             "neighbour hydrostatic integral",
             "augmented momentum jump", "viscous momentum integral",
             "suspended advective mass", "suspended diffusive mass",
             "fluid advective integral", "fluid diffusive integral",
             "bedload mass")
    frozen = tuple(_frozen(array, shape, name)
                   for array, shape, name in zip(arrays, shapes, names))
    source = _hash({
        "scope_id": SCOPE_ID+":private-fine-source",
        "hydraulic_trial_sha256": hydraulic_trial.trial_sha256,
        "hydraulic_audit_sha256": audit.audit_sha256,
        "scalar_result_sha256": scalar_result.result_sha256,
        "q3_result_sha256": q3_result.result_sha256,
        "candidate_state_sha256": candidate.state_sha256,
        "source_kind": SOURCE_PRIVATE_FINE,
    })
    placeholder = CoupledFaceTimeIntegralsGPUV98(
        execution_fingerprint=execution.fingerprint,
        parent_state_sha256=parent.state_sha256,
        candidate_state_sha256=candidate.state_sha256,
        source_trial_sha256=source,
        hydraulic_trial_sha256=hydraulic_trial.trial_sha256,
        hydraulic_runtime_receipt_sha256=
            hydraulic_runtime.runtime_receipt["receipt_sha256"],
        scalar_runtime_receipt_sha256=
            scalar_runtime.runtime_receipt["receipt_sha256"],
        q3_face_trial_sha256=q3_result.result_sha256,
        extractor_source_sha256=_private_fine_source_sha256(),
        source_kind=SOURCE_PRIVATE_FINE,
        parent_accepted_updates=parent.accepted_updates,
        candidate_accepted_updates=candidate.accepted_updates,
        time_start_s=parent.time_s,
        time_end_s=candidate.time_s,
        dt_s=dt,
        morphological_factor=morphological_factor,
        face_length_m=frozen[0], midpoint_depth_m=frozen[1],
        water_volume_m3=frozen[2],
        augmented_left_face_integral=frozen[3],
        augmented_right_face_integral=frozen[4],
        owner_hydrostatic_momentum_integral_m4_s=frozen[5],
        neighbour_hydrostatic_momentum_integral_m4_s=frozen[6],
        augmented_momentum_jump_integral_m4_s=frozen[7],
        viscous_momentum_integral_m4_s=frozen[8],
        suspended_advective_mass_kg=frozen[9],
        suspended_diffusive_mass_kg=frozen[10],
        fluid_advective_integral=frozen[11],
        fluid_diffusive_integral=frozen[12],
        bedload_mass_kg=frozen[13],
        hydraulic_side_evaluation_count=audit.evaluation_count,
        discarded_provisional_side_count=audit.evaluation_count-1,
        accepted_side_replacement_count=1,
        resident_hydraulic_readback_count=1,
        resident_scalar_readback_count=1,
        resident_q3_readback_count=1,
        backend=PRIVATE_BACKEND,
        receipt_sha256="",
        suspended_bed_divergence_selected=mode_v100(execution.active_execution.base_execution) == 1)
    result = CoupledFaceTimeIntegralsGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_payload(placeholder))})
    validate_coupled_face_time_integrals_gpu_v98(result, execution)
    return result


@dataclass(frozen=True)
class CoupledAMRInterfaceMapGPUV98:
    coarse_execution_fingerprint: str
    fine_execution_fingerprint: str
    geometry_binding_receipt_sha256: str
    joined_geometry_receipt_sha256: str
    joined_interface_fingerprint: str
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


def _interface_payload(value: CoupledAMRInterfaceMapGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":joined-interface-map",
        "coarse_execution": value.coarse_execution_fingerprint,
        "fine_execution": value.fine_execution_fingerprint,
        "geometry_binding": value.geometry_binding_receipt_sha256,
        "joined_geometry": value.joined_geometry_receipt_sha256,
        "joined_interface": value.joined_interface_fingerprint,
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
        "fine_interface_kind":
            "typed_internal_coarse_fine_not_physical_open",
        "backend": BACKEND,
    }


def resolve_coupled_amr_interface_map_gpu_v98(
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    geometry_binding: CoupledAMRGeometryBindingGPUV98,
) -> CoupledAMRInterfaceMapGPUV98:
    if (type(coarse_execution) is not CoupledWetDryGPUExecutionV97 or
            type(fine_execution) is not CoupledWetDryGPUExecutionV97 or
            type(geometry_binding) is not
                CoupledAMRGeometryBindingGPUV98):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA interface requires exact execution/geometry types")
    validate_coupled_amr_geometry_binding_gpu_v98(geometry_binding)
    cg = _grid(coarse_execution.active_execution)
    fg = _grid(fine_execution.active_execution)
    joined = geometry_binding.joined_geometry
    if (geometry_binding.coarse_execution_fingerprint !=
            coarse_execution.fingerprint or
            geometry_binding.fine_execution_fingerprint !=
            fine_execution.fingerprint or
            geometry_binding.coarse_grid.fingerprint != cg.fingerprint or
            geometry_binding.fine_grid.fingerprint != fg.fingerprint or
            coarse_execution.fraction_count !=
                fine_execution.fraction_count or
            cg.profile.fingerprint != fg.profile.fingerprint):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA interface geometry or transported physics differ")
    coarse_face = int(joined.coarse_face)
    fine_faces = tuple(map(int, joined.fine_faces))
    coarse_cell = int(joined.coarse_owner_cell)
    fine_cells = tuple(map(int, joined.fine_owner_cells))
    normal = _frozen(
        np.asarray(cg.normal)[:, coarse_face], (2,),
        "canonical interface normal")
    if (int(cg.owner[coarse_face]) != coarse_cell or
            int(cg.neighbour[coarse_face]) != -1 or
            int(cg.boundary_index[coarse_face]) != -1 or
            any(int(fg.owner[face]) != cell or
                int(fg.neighbour[face]) != -1 or
                int(fg.boundary_index[face]) != -1
                for cell, face in zip(fine_cells, fine_faces)) or
            not np.all(
                np.asarray(fg.normal)[:, list(fine_faces)] ==
                -normal[:, None])):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA joined interface topology differs")
    coarse_length = float(cg.face_length_m[coarse_face])
    fine_length = sum(float(fg.face_length_m[face])
                      for face in fine_faces)
    tolerance = 256.0*np.finfo(float).eps*max(
        1.0, abs(coarse_length), abs(fine_length))
    area = float(cg.cell_area_m2[coarse_cell])
    if (abs(coarse_length-fine_length) > tolerance or
            not math.isfinite(area) or area <= 0.0):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA joined interface measure differs")
    coarse_global = int(
        geometry_binding.coarse_patch.face_global_ids[coarse_face])
    fine_global = tuple(int(
        geometry_binding.fine_patch.face_global_ids[face])
        for face in fine_faces)
    placeholder = CoupledAMRInterfaceMapGPUV98(
        coarse_execution.fingerprint, fine_execution.fingerprint,
        geometry_binding.receipt_sha256, joined.receipt_sha256,
        joined.interface_fingerprint, coarse_cell, coarse_face,
        fine_cells, fine_faces, 1, (-1, -1), coarse_global, fine_global,
        normal, area, "")
    return CoupledAMRInterfaceMapGPUV98(
        **{**placeholder.__dict__,
           "fingerprint": _hash(_interface_payload(placeholder))})


def validate_coupled_amr_interface_map_gpu_v98(
    value: CoupledAMRInterfaceMapGPUV98,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    geometry_binding: CoupledAMRGeometryBindingGPUV98,
) -> None:
    if (type(value) is not CoupledAMRInterfaceMapGPUV98 or
            value.fingerprint != _hash(_interface_payload(value))):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA joined interface map identity differs")
    rebuilt = resolve_coupled_amr_interface_map_gpu_v98(
        coarse_execution, fine_execution, geometry_binding)
    if rebuilt.fingerprint != value.fingerprint:
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA rebuilt joined interface map differs")


@dataclass(frozen=True)
class CoupledAMRInternalInterfaceLedgerGPUV98:
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
    value: CoupledAMRInternalInterfaceLedgerGPUV98,
) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":internal-interface-ledger",
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
        "backend": BACKEND,
    }


@dataclass(frozen=True)
class CoupledAMRCoarseCorrectionGPUV98:
    interface_fingerprint: str
    geometry_binding_receipt_sha256: str
    coarse_integrals_sha256: str
    fine_integrals_sha256: tuple[str, str]
    coarse_parent_state_sha256: str
    fine_parent_state_sha256: str
    fine_final_state_sha256: str
    scheduled_register: CoupledAMRTwoPhaseRegisterV98
    internal_interface_ledger: CoupledAMRInternalInterfaceLedgerGPUV98
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


def _correction_payload(value: CoupledAMRCoarseCorrectionGPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID+":coarse-correction",
        "interface": value.interface_fingerprint,
        "geometry_binding": value.geometry_binding_receipt_sha256,
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
            value.momentum_delta_m2_s, value.suspended_delta_kg_m2,
            value.fluid_delta, value.bed_delta_kg_m2,
            value.corrected_face_water_flux_m2_s,
            value.corrected_face_bedload_flux_kg_m_s),
        "coarse_replacement_count": value.coarse_replacement_count,
        "fine_state_correction_count": value.fine_state_correction_count,
        "density_reflux_count": value.density_reflux_count,
        "split_policy":
            "retain_coarse_local_sources_rebuild_derived_fields",
        "backend": BACKEND,
    }


def _validate_reflux_sequence_gpu_v98(
    coarse: CoupledFaceTimeIntegralsGPUV98,
    fine: Sequence[CoupledFaceTimeIntegralsGPUV98], *,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
) -> tuple[CoupledFaceTimeIntegralsGPUV98,
           CoupledFaceTimeIntegralsGPUV98]:
    validate_coupled_face_time_integrals_gpu_v98(
        coarse, coarse_execution)
    values = tuple(fine)
    if len(values) != 2:
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA reflux requires two private fine substeps")
    for item in values:
        validate_coupled_face_time_integrals_gpu_v98(item, fine_execution)
    if (coarse.source_kind != SOURCE_OPEN_V97 or
            any(item.source_kind != SOURCE_PRIVATE_FINE for item in values) or
            coarse.dt_s != 2.0*values[0].dt_s or
            values[1].dt_s != values[0].dt_s or
            values[0].morphological_factor !=
                coarse.morphological_factor or
            values[1].morphological_factor !=
                coarse.morphological_factor or
            values[0].candidate_state_sha256 !=
                values[1].parent_state_sha256 or
            values[0].candidate_accepted_updates !=
                values[1].parent_accepted_updates or
            values[0].time_end_s != values[1].time_start_s or
            coarse.time_start_s != values[0].time_start_s or
            coarse.time_end_s != values[1].time_end_s):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA coarse/fine source kinds or private clocks differ")
    return values


def _hydraulic_vector_gpu_v98(
    value: CoupledFaceTimeIntegralsGPUV98,
    face: int,
    sign: int,
) -> np.ndarray:
    if sign not in (-1, 1):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA interface orientation differs")
    augmented = value.augmented_left_face_integral[:, face]
    pressure = value.owner_hydrostatic_momentum_integral_m4_s[:, face]
    return np.ascontiguousarray(np.concatenate((
        np.atleast_1d(augmented[0]),
        augmented[1:]-pressure-
            value.viscous_momentum_integral_m4_s[:, face]))*sign)


def _expected_correction_arrays_gpu_v98(
    interface: CoupledAMRInterfaceMapGPUV98,
    coarse: CoupledFaceTimeIntegralsGPUV98,
    fine: Sequence[CoupledFaceTimeIntegralsGPUV98],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cf, ff = interface.coarse_face, interface.fine_faces
    cs, fs = interface.coarse_to_canonical, interface.fine_to_canonical

    def selected(name: str) -> np.ndarray:
        coarse_value = np.asarray(getattr(coarse, name))[..., cf]*cs
        fine_value = np.zeros_like(coarse_value, dtype=np.float64)
        for step in fine:
            for face, sign in zip(ff, fs):
                fine_value = fine_value + \
                    np.asarray(getattr(step, name))[..., face]*sign
        return np.asarray(coarse_value-fine_value, dtype=np.float64)

    def selected_sum(first: str, second: str) -> np.ndarray:
        """Use the exact per-face addition order recorded by the register."""
        coarse_value = (
            np.asarray(getattr(coarse, first))[..., cf]+
            np.asarray(getattr(coarse, second))[..., cf])*cs
        fine_value = np.zeros_like(coarse_value, dtype=np.float64)
        for step in fine:
            for face, sign in zip(ff, fs):
                fine_value = fine_value + (
                    np.asarray(getattr(step, first))[..., face]+
                    np.asarray(getattr(step, second))[..., face])*sign
        return np.asarray(coarse_value-fine_value, dtype=np.float64)

    coarse_hydraulic = _hydraulic_vector_gpu_v98(coarse, cf, cs)
    fine_hydraulic = np.zeros(3, dtype=np.float64)
    for step in fine:
        for face, sign in zip(ff, fs):
            fine_hydraulic += _hydraulic_vector_gpu_v98(step, face, sign)
    return (
        coarse_hydraulic[0]-fine_hydraulic[0],
        coarse_hydraulic[1:]-fine_hydraulic[1:],
        selected_sum("suspended_advective_mass_kg",
                     "suspended_diffusive_mass_kg"),
        selected_sum("fluid_advective_integral",
                     "fluid_diffusive_integral"),
        selected("bedload_mass_kg")+selected("selected_suspended_bed_mass_kg"),
    )


def _build_scheduled_register_gpu_v98(
    interface: CoupledAMRInterfaceMapGPUV98,
    coarse: CoupledFaceTimeIntegralsGPUV98,
    fine: Sequence[CoupledFaceTimeIntegralsGPUV98],
) -> CoupledAMRTwoPhaseRegisterV98:
    cf, ff = interface.coarse_face, interface.fine_faces
    cs, fs = interface.coarse_to_canonical, interface.fine_to_canonical
    hydraulic_lanes = ("water_volume_m3", "momentum_x_m4_s",
                       "momentum_y_m4_s")
    coarse_hydraulic = _hydraulic_vector_gpu_v98(coarse, cf, cs)
    fine_hydraulic = np.empty((2, 2, len(hydraulic_lanes)))
    for step in range(2):
        for slot, (face, sign) in enumerate(zip(ff, fs)):
            fine_hydraulic[step, slot] = _hydraulic_vector_gpu_v98(
                fine[step], face, sign)

    def scalar_vector(value, face, sign):
        suspended = (
            value.suspended_advective_mass_kg[..., face]+
            value.suspended_diffusive_mass_kg[..., face])*sign
        fluid = (
            value.fluid_advective_integral[..., face]+
            value.fluid_diffusive_integral[..., face])*sign
        bed = value.bedload_mass_kg[..., face]*sign
        return np.ascontiguousarray(np.concatenate((
            suspended, fluid, bed)))

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
        time_start_s=coarse.time_start_s,
        coarse_dt_s=coarse.dt_s,
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


def build_coupled_amr_coarse_correction_gpu_v98(
    interface: CoupledAMRInterfaceMapGPUV98,
    coarse: CoupledFaceTimeIntegralsGPUV98,
    fine_steps: Sequence[CoupledFaceTimeIntegralsGPUV98], *,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    geometry_binding: CoupledAMRGeometryBindingGPUV98,
) -> CoupledAMRCoarseCorrectionGPUV98:
    validate_coupled_amr_interface_map_gpu_v98(
        interface, coarse_execution, fine_execution, geometry_binding)
    fine = _validate_reflux_sequence_gpu_v98(
        coarse, fine_steps, coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    scheduled = _build_scheduled_register_gpu_v98(
        interface, coarse, fine)
    raw = _expected_correction_arrays_gpu_v98(
        interface, coarse, fine)
    area = interface.coarse_cell_area_m2
    ng = coarse_execution.fraction_count
    ledger0 = CoupledAMRInternalInterfaceLedgerGPUV98(
        interface.fingerprint, scheduled.receipt_sha256,
        coarse.receipt_sha256,
        (fine[0].receipt_sha256, fine[1].receipt_sha256),
        float(raw[0]), _frozen(raw[1], (2,), "internal momentum ledger"),
        _frozen(raw[2], (ng,), "internal suspended ledger"),
        _frozen(raw[3], (FLUID_SCALAR_COUNT,), "internal fluid ledger"),
        _frozen(raw[4], (ng,), "internal bed ledger"), 0, 1, "")
    ledger = CoupledAMRInternalInterfaceLedgerGPUV98(
        **{**ledger0.__dict__, "receipt_sha256": _hash(
            _internal_ledger_payload(ledger0))})
    corrected_water = np.array(
        coarse.water_volume_m3/coarse.dt_s/coarse.face_length_m,
        dtype=np.float64, order="C", copy=True)
    fine_water = sum(float(step.water_volume_m3[face])
                     for step in fine for face in interface.fine_faces)
    corrected_water[interface.coarse_face] = (
        -fine_water/(coarse.dt_s*
                     coarse.face_length_m[interface.coarse_face]))
    corrected_water = _frozen(
        corrected_water, (coarse_execution.face_count,),
        "corrected water face flux")
    corrected_bed = np.array(
        coarse.bedload_mass_kg/
        (coarse.dt_s*coarse.face_length_m[None, :]*
         coarse.morphological_factor),
        dtype=np.float64, order="C", copy=True)
    fine_bed = sum(
        (step.bedload_mass_kg[:, face]
         for step in fine for face in interface.fine_faces),
        start=np.zeros(ng, dtype=np.float64))
    corrected_bed[:, interface.coarse_face] = (
        -fine_bed/(coarse.dt_s*
                   coarse.face_length_m[interface.coarse_face]*
                   coarse.morphological_factor))
    corrected_bed = _frozen(
        corrected_bed, (ng, coarse_execution.face_count),
        "corrected bedload face flux")
    placeholder = CoupledAMRCoarseCorrectionGPUV98(
        interface.fingerprint, geometry_binding.receipt_sha256,
        coarse.receipt_sha256,
        (fine[0].receipt_sha256, fine[1].receipt_sha256),
        coarse.parent_state_sha256, fine[0].parent_state_sha256,
        fine[1].candidate_state_sha256, scheduled, ledger,
        float(raw[0]/area),
        _frozen(raw[1]/area, (2,), "momentum correction"),
        _frozen(raw[2]/area, (ng,), "suspended correction"),
        _frozen(raw[3]/area, (FLUID_SCALAR_COUNT,), "fluid correction"),
        _frozen(raw[4]/area, (ng,), "bed correction"),
        corrected_water, corrected_bed, 1, 0, 0, "")
    result = CoupledAMRCoarseCorrectionGPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_correction_payload(placeholder))})
    validate_coupled_amr_coarse_correction_gpu_v98(
        result, interface, coarse, fine,
        coarse_execution=coarse_execution,
        fine_execution=fine_execution,
        geometry_binding=geometry_binding)
    return result


def validate_coupled_amr_coarse_correction_gpu_v98(
    value: CoupledAMRCoarseCorrectionGPUV98,
    interface: CoupledAMRInterfaceMapGPUV98,
    coarse: CoupledFaceTimeIntegralsGPUV98,
    fine_steps: Sequence[CoupledFaceTimeIntegralsGPUV98], *,
    coarse_execution: CoupledWetDryGPUExecutionV97,
    fine_execution: CoupledWetDryGPUExecutionV97,
    geometry_binding: CoupledAMRGeometryBindingGPUV98,
) -> None:
    fine = _validate_reflux_sequence_gpu_v98(
        coarse, fine_steps, coarse_execution=coarse_execution,
        fine_execution=fine_execution)
    validate_coupled_amr_interface_map_gpu_v98(
        interface, coarse_execution, fine_execution, geometry_binding)
    if (type(value) is not CoupledAMRCoarseCorrectionGPUV98 or
            value.interface_fingerprint != interface.fingerprint or
            value.geometry_binding_receipt_sha256 !=
                geometry_binding.receipt_sha256 or
            value.coarse_integrals_sha256 != coarse.receipt_sha256 or
            value.fine_integrals_sha256 !=
                tuple(item.receipt_sha256 for item in fine) or
            value.coarse_replacement_count != 1 or
            value.fine_state_correction_count != 0 or
            value.density_reflux_count != 0 or
            value.receipt_sha256 != _hash(_correction_payload(value))):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA coarse correction identity differs")
    validate_two_phase_register_v98(value.scheduled_register)
    scheduled = _build_scheduled_register_gpu_v98(
        interface, coarse, fine)
    raw = _expected_correction_arrays_gpu_v98(
        interface, coarse, fine)
    ledger = value.internal_interface_ledger
    ng = coarse_execution.fraction_count
    if (value.scheduled_register.receipt_sha256 !=
            scheduled.receipt_sha256 or
            type(ledger) is not CoupledAMRInternalInterfaceLedgerGPUV98 or
            ledger.interface_fingerprint != interface.fingerprint or
            ledger.scheduled_register_sha256 != scheduled.receipt_sha256 or
            ledger.coarse_integrals_sha256 != coarse.receipt_sha256 or
            ledger.fine_integrals_sha256 !=
                tuple(item.receipt_sha256 for item in fine) or
            ledger.physical_boundary_ledger_count != 0 or
            ledger.application_count != 1 or
            ledger.receipt_sha256 !=
                _hash(_internal_ledger_payload(ledger)) or
            float(ledger.water_volume_m3) != float(raw[0]) or
            any(not np.array_equal(actual, expected)
                for actual, expected in (
                    (ledger.momentum_m4_s, raw[1]),
                    (ledger.suspended_mass_kg, raw[2]),
                    (ledger.fluid_integral, raw[3]),
                    (ledger.bed_mass_kg, raw[4])))):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA internal interface ledger differs")
    area = interface.coarse_cell_area_m2
    if (float(value.water_depth_delta_m) != float(raw[0]/area) or
            any(not np.array_equal(actual, expected)
                for actual, expected in (
                    (value.momentum_delta_m2_s, raw[1]/area),
                    (value.suspended_delta_kg_m2, raw[2]/area),
                    (value.fluid_delta, raw[3]/area),
                    (value.bed_delta_kg_m2, raw[4]/area)))):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA coarse correction values differ")
    hydraulic = np.concatenate((
        np.atleast_1d(value.water_depth_delta_m),
        value.momentum_delta_m2_s))
    scalar = np.concatenate((
        value.suspended_delta_kg_m2, value.fluid_delta,
        value.bed_delta_kg_m2))
    expected_scalar = np.array(
        value.scheduled_register.scalar_bed.coarse_cell_correction, copy=True)
    if coarse.suspended_bed_divergence_selected:
        expected_scalar[-ng:] += expected_scalar[:ng]
    if (not np.array_equal(
            hydraulic,
            value.scheduled_register.hydraulic.coarse_cell_correction) or
            not np.allclose(scalar, expected_scalar, rtol=0., atol=1e-13)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA scheduled register lane order differs")
    expected_water = np.array(
        coarse.water_volume_m3/coarse.dt_s/coarse.face_length_m,
        dtype=np.float64, order="C", copy=True)
    fine_water = sum(float(step.water_volume_m3[face])
                     for step in fine for face in interface.fine_faces)
    expected_water[interface.coarse_face] = (
        -fine_water/(coarse.dt_s*
                     coarse.face_length_m[interface.coarse_face]))
    expected_bed = np.array(
        coarse.bedload_mass_kg/
        (coarse.dt_s*coarse.face_length_m[None, :]*
         coarse.morphological_factor),
        dtype=np.float64, order="C", copy=True)
    fine_bed = sum(
        (step.bedload_mass_kg[:, face]
         for step in fine for face in interface.fine_faces),
        start=np.zeros(ng, dtype=np.float64))
    expected_bed[:, interface.coarse_face] = (
        -fine_bed/(coarse.dt_s*
                   coarse.face_length_m[interface.coarse_face]*
                   coarse.morphological_factor))
    if (value.corrected_face_water_flux_m2_s.flags.writeable or
            value.corrected_face_bedload_flux_kg_m_s.flags.writeable or
            not np.array_equal(
                value.corrected_face_water_flux_m2_s, expected_water) or
            not np.array_equal(
                value.corrected_face_bedload_flux_kg_m_s, expected_bed)):
        raise CoupledAMRRegisterGPUV98Error(
            "v98 CUDA corrected interface face flux differs")


__all__ = [
    "BACKEND", "PRIVATE_BACKEND", "SCOPE_ID", "SOURCE_OPEN_V97",
    "SOURCE_PRIVATE_FINE", "CoupledAMRRegisterGPUV98Error",
    "CoupledAMRCoarseCorrectionGPUV98",
    "CoupledAMRInterfaceMapGPUV98",
    "CoupledAMRInternalInterfaceLedgerGPUV98",
    "CoupledFaceTimeIntegralsGPUV98",
    "build_coupled_amr_coarse_correction_gpu_v98",
    "extract_coupled_face_time_integrals_gpu_v98",
    "make_private_fine_face_time_integrals_gpu_v98",
    "resolve_coupled_amr_interface_map_gpu_v98",
    "validate_coupled_amr_coarse_correction_gpu_v98",
    "validate_coupled_amr_interface_map_gpu_v98",
    "validate_coupled_face_time_integrals_gpu_v98",
]
