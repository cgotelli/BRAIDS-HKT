"""Correct one provisional coarse v97 leaf with accepted fine integrals.

The corrected image has an additive v98 identity.  It is deliberately *not*
wrapped as :class:`CoupledWetDryCPUStateV97`: the latter owns only physical
boundary ledgers, whereas this module owns one real internal coarse/fine
interface ledger.  Local scalar, exchange, settling and Q3 constitutive work
is not replayed.  The already accepted v94 candidate DSS is carried forward,
bed/fluid density are derived from the corrected conserved image, and HLES is
evaluated once from the corrected endpoint and corrected water-face flux.

The Q3 child remains a two-phase transaction.  ``begin`` creates an immutable
corrected rebound without changing the live provider.  ``close`` replaces the
unpublished provisional rebound, closes the child, but does not publish it.
``publish`` is the sole provider publication.  This lets a later v98 macro
owner close every coarse/fine child before publishing any accepted state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Sequence

import numpy as np

from . import hles_native_cpu_v97 as _native_hles
from .coupled_amr_register_cpu_v98 import (
    CoupledAMRCoarseCorrectionCPUV98,
    CoupledAMRInterfaceMapCPUV98,
    CoupledFaceTimeIntegralsCPUV98,
    validate_coupled_amr_coarse_correction_cpu_v98,
)
from .coupled_amr_corrected_images_native_v98 import (
    KEYS as _NATIVE_CORRECTED_IMAGE_KEYS,
    NativeAMRCorrectedImagesRuntimeV98,
    _digest as _native_image_digest,
    _hash as _native_image_hash,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    CoupledWetDryCPUTrialV97,
    _bed_from_mass,
    _grid_execution,
    _hles_inputs,
    _v92_execution,
    _v92_state,
    _validate_trial as _validate_v97_trial,
    coupled_wetdry_cpu_checkpoint_v97,
    execution_document_coupled_wetdry_v97,
    load_coupled_wetdry_cpu_checkpoint_v97,
    rollback_coupled_wetdry_cpu_trial_v97,
    validate_coupled_wetdry_cpu_state_v97,
)
from .densin_fwave_v85 import mixture_density_kg_m3_v85
from .hles_native_cpu_v97 import NativeHlesCandidateV97, NativeHlesRuntimeV97
from .hles_producer_v90 import (
    HlesProducerStateV90,
    HlesProducerTrialV90,
    commit_hles_trial_v90,
    hles_checkpoint_v90,
    load_hles_checkpoint_v90,
    make_hles_input_receipt_v90,
    rollback_hles_trial_v90,
    validate_hles_producer_state_v90,
)
from .q3_mixed_route_exner_adapter_v93 import (
    CPUQ3V84MixedPreparedCommitV93,
    CPUQ3V84MixedStateV93,
    _array_digest as _q3_array_digest,
    _prepared_token as _q3_prepared_token,
    resolve_cpu_q3_v84_mixed_state_v93,
    validate_cpu_q3_v84_mixed_state_v93,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .transported_fluid_settling_v92 import (
    MODE_TRANSPORTED,
    TEMPERATURE_FLOOR_C,
    _properties as _fluid_properties,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-amr-corrected-coarse-cpu-v98-v1"
SEMANTIC_ID = SCOPE_ID + ":internal-interface-ledger-corrected-leaf"
CHECKPOINT_SCOPE_ID = SCOPE_ID + ":checkpoint"
BACKEND = "native_cpu_openmp_plus_internal_reflux"
MAX_ACCEPTED_UPDATES = 4
NATIVE_ORACLE_RTOL = 3.0e-13
NATIVE_ORACLE_ATOL = 4.0e-15
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class CoupledAMRCorrectedCoarseCPUV98Error(ValueError):
    """A corrected leaf, source receipt, or two-phase handle differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected leaf value is not finite canonical JSON") \
            from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        result.update(array.dtype.str.encode("ascii"))
        result.update(_canonical(list(array.shape)))
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            f"v98 corrected {name} is not a SHA-256 digest")
    return value


def _array(value: object, shape: tuple[int, ...], name: str, *,
           nonnegative: bool = False, integer: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or raw.dtype.kind not in "fiu" or
            not np.isfinite(raw).all() or
            (nonnegative and np.any(raw < 0.0)) or
            (integer and np.any(raw != np.floor(raw)))):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            f"v98 corrected {name} differs")
    result = np.array(
        raw, dtype=np.int32 if integer else np.float64,
        order="C", copy=True)
    result.setflags(write=False)
    return result


def _cancelled(value: bool | Callable[[], bool]) -> bool:
    if type(value) is bool:
        return value
    if not callable(value):
        raise TypeError("cancelled must be Boolean or callable")
    return bool(value())


@dataclass(frozen=True)
class CoupledAMRCorrectedLeafCountsCPUV98:
    internal_interface_ledger_application_count: int = 1
    physical_boundary_ledger_application_count: int = 0
    local_hydraulic_recompute_count: int = 0
    local_scalar_recompute_count: int = 0
    local_exchange_recompute_count: int = 0
    local_q3_recompute_count: int = 0
    dss_recompute_count: int = 0
    reflux_dry_cleanup_owner_count: int = 1
    bed_geometry_derivation_count: int = 1
    fluid_eos_derivation_count: int = 1
    mixture_eos_derivation_count: int = 1
    q3_rebound_owner_count: int = 1
    native_hles_candidate_owner_count: int = 1
    hles_clock_owner_count: int = 1
    native_corrected_image_owner_count: int = 1
    host_corrected_image_oracle_count: int = 1
    python_corrected_image_numerical_owner_count: int = 0
    duplicate_owner_count: int = 0


@dataclass(frozen=True)
class NativeCorrectedImagesEvidenceCPUV98:
    """Authenticated native image plus validation-only host comparison."""

    runtime_sha256: str
    input_sha256: str
    output_sha256: str
    controls_sha256: str
    source_state_sha256: str
    correction_receipt_sha256: str
    native_result_receipt_sha256: str
    native_threads: int
    host_oracle_keys: tuple[str, ...]
    host_oracle_max_abs: tuple[float, ...]
    exact_conserved_cleanup_sha256: str
    native_owner_count: int
    host_oracle_validation_count: int
    python_numerical_owner_count: int
    state_or_clock_publication_count: int
    evidence_sha256: str


def _native_evidence_payload(value: NativeCorrectedImagesEvidenceCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":native-corrected-images",
        "runtime": value.runtime_sha256,
        "input": value.input_sha256,
        "output": value.output_sha256,
        "controls": value.controls_sha256,
        "source_state": value.source_state_sha256,
        "correction": value.correction_receipt_sha256,
        "native_result_receipt": value.native_result_receipt_sha256,
        "native_threads": value.native_threads,
        "host_oracle_keys": list(value.host_oracle_keys),
        "host_oracle_max_abs": list(value.host_oracle_max_abs),
        "exact_conserved_cleanup": value.exact_conserved_cleanup_sha256,
        "native_owner_count": value.native_owner_count,
        "host_oracle_validation_count": value.host_oracle_validation_count,
        "python_numerical_owner_count": value.python_numerical_owner_count,
        "state_or_clock_publication_count":
            value.state_or_clock_publication_count,
        "host_oracle_is_validation_only": True,
    }


@dataclass(frozen=True)
class CoupledAMRCorrectedLeafStateCPUV98:
    """Accepted v98 leaf image plus immutable provisional provenance."""

    execution_fingerprint: str
    provisional_source_state: CoupledWetDryCPUStateV97
    correction_receipt_sha256: str
    internal_interface_ledger_sha256: str
    q: np.ndarray
    suspended_mass_kg_m2: np.ndarray
    bed_mass_kg_m2: np.ndarray
    h_temperature_c_m: np.ndarray
    h_salinity_ppt_m: np.ndarray
    temperature_cell_c: np.ndarray
    salinity_cell_ppt: np.ndarray
    water_density_cell_kg_m3: np.ndarray
    mixture_density_cell_kg_m3: np.ndarray
    molecular_viscosity_cell_m2_s: np.ndarray
    bed_elevation_m: np.ndarray
    free_surface_elevation_m: np.ndarray
    wet_mask: np.ndarray
    suspended_diameter_m: np.ndarray
    reflux_dry_transfer_to_bed_kg_m2: np.ndarray
    reflux_dry_momentum_cleanup_m2_s: np.ndarray
    reflux_dry_fluid_cleanup: np.ndarray
    corrected_face_water_flux_m2_s: np.ndarray
    corrected_face_bedload_flux_kg_m_s: np.ndarray
    suspended_concentration_kg_m3: np.ndarray
    native_corrected_images_evidence: NativeCorrectedImagesEvidenceCPUV98
    provisional_q3_state: CPUQ3V84MixedStateV93
    q3_state: CPUQ3V84MixedStateV93
    hles_state: HlesProducerStateV90
    accepted_updates: int
    time_s: float
    accepted_chain_sha256: str
    counts: CoupledAMRCorrectedLeafCountsCPUV98
    semantic_id: str
    backend: str
    state_sha256: str


def _state_payload(value: CoupledAMRCorrectedLeafStateCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":state",
        "semantic_id": value.semantic_id,
        "backend": value.backend,
        "execution": value.execution_fingerprint,
        "provisional_source_state":
            value.provisional_source_state.state_sha256,
        "correction": value.correction_receipt_sha256,
        "internal_interface_ledger":
            value.internal_interface_ledger_sha256,
        "conserved_and_derived_arrays_sha256": _digest(
            value.q, value.suspended_mass_kg_m2, value.bed_mass_kg_m2,
            value.h_temperature_c_m, value.h_salinity_ppt_m,
            value.temperature_cell_c, value.salinity_cell_ppt,
            value.water_density_cell_kg_m3,
            value.mixture_density_cell_kg_m3,
            value.molecular_viscosity_cell_m2_s,
            value.bed_elevation_m, value.free_surface_elevation_m,
            value.wet_mask, value.suspended_diameter_m,
            value.reflux_dry_transfer_to_bed_kg_m2,
            value.reflux_dry_momentum_cleanup_m2_s,
            value.reflux_dry_fluid_cleanup,
            value.corrected_face_water_flux_m2_s,
            value.corrected_face_bedload_flux_kg_m_s,
            value.suspended_concentration_kg_m3),
        "native_corrected_images":
            value.native_corrected_images_evidence.evidence_sha256,
        "provisional_q3_state": value.provisional_q3_state.state_sha256,
        "q3_state": value.q3_state.state_sha256,
        "hles_state": value.hles_state.state_sha256,
        "accepted_updates": value.accepted_updates,
        "time_s": value.time_s,
        "accepted_chain": value.accepted_chain_sha256,
        "counts": value.counts.__dict__,
        "provisional_source_is_evidence_not_accepted_image": True,
        "internal_interface_is_not_physical_boundary": True,
    }


@dataclass(frozen=True)
class CoupledAMRCorrectedLeafTrialCPUV98:
    execution_fingerprint: str
    parent_state_sha256: str
    provisional_trial_sha256: str
    interface_fingerprint: str
    coarse_integrals_sha256: str
    fine_integrals_sha256: tuple[str, str]
    correction: CoupledAMRCoarseCorrectionCPUV98
    corrected_q3_prepared: CPUQ3V84MixedPreparedCommitV93
    corrected_hles_trial: HlesProducerTrialV90
    native_hles_evidence: NativeHlesCandidateV97
    hles_inputs: tuple[np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray]
    counts: CoupledAMRCorrectedLeafCountsCPUV98
    candidate: CoupledAMRCorrectedLeafStateCPUV98
    provider_mutation_count: int
    accepted_state_publication_count: int
    backend: str
    trial_sha256: str


def _trial_payload(value: CoupledAMRCorrectedLeafTrialCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":trial",
        "execution": value.execution_fingerprint,
        "parent": value.parent_state_sha256,
        "provisional_trial": value.provisional_trial_sha256,
        "interface": value.interface_fingerprint,
        "coarse_integrals": value.coarse_integrals_sha256,
        "fine_integrals": list(value.fine_integrals_sha256),
        "correction": value.correction.receipt_sha256,
        "corrected_q3_prepared":
            value.corrected_q3_prepared.preparation_sha256,
        "corrected_hles_trial": value.corrected_hles_trial.trial_sha256,
        "native_hles": value.native_hles_evidence.result_sha256,
        "hles_inputs": _digest(*value.hles_inputs),
        "counts": value.counts.__dict__,
        "candidate": value.candidate.state_sha256,
        "provider_mutation_count": value.provider_mutation_count,
        "accepted_state_publication_count":
            value.accepted_state_publication_count,
        "backend": value.backend,
    }


@dataclass(frozen=True)
class CoupledAMRCorrectedLeafClosedCPUV98:
    trial_sha256: str
    corrected_q3_preparation_sha256: str
    candidate_state_sha256: str
    provider_state_before_sha256: str
    provider_state_after_sha256: str
    q3_child_close_count: int
    provider_publication_count: int
    handle_sha256: str


def _closed_payload(value: CoupledAMRCorrectedLeafClosedCPUV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":closed",
        "trial": value.trial_sha256,
        "corrected_q3_preparation":
            value.corrected_q3_preparation_sha256,
        "candidate": value.candidate_state_sha256,
        "provider_before": value.provider_state_before_sha256,
        "provider_after": value.provider_state_after_sha256,
        "q3_child_close_count": value.q3_child_close_count,
        "provider_publication_count": value.provider_publication_count,
    }


def _derive_corrected_images(
    execution: CoupledWetDryCPUExecutionV97,
    provisional: CoupledWetDryCPUStateV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> dict[str, np.ndarray]:
    """Apply one signed internal-interface correction and derive fields."""
    ng, nc, nf = (execution.fraction_count, execution.cell_count,
                  execution.face_count)
    cell = interface.coarse_leaf_cell
    v92 = _v92_state(provisional.active_state)
    q = np.array(v92.q, dtype=np.float64, order="C", copy=True)
    suspended = np.array(
        v92.suspended_state.suspended_mass_kg_m2,
        dtype=np.float64, order="C", copy=True)
    bed = np.array(
        provisional.active_state.base_state.bed_mass_kg_m2,
        dtype=np.float64, order="C", copy=True)
    ht = np.array(v92.h_temperature_c_m, dtype=np.float64,
                  order="C", copy=True)
    hs = np.array(v92.h_salinity_ppt_m, dtype=np.float64,
                  order="C", copy=True)
    q[0, cell] += correction.water_depth_delta_m
    q[1:, cell] += correction.momentum_delta_m2_s
    suspended[:, cell] += correction.suspended_delta_kg_m2
    bed[:, cell] += correction.bed_delta_kg_m2
    ht[cell] += correction.fluid_delta[0]
    hs[cell] += correction.fluid_delta[1]
    tolerance = 32768.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.maximum(np.abs(suspended), np.abs(bed)))
    suspended[np.abs(suspended) <= tolerance] = np.maximum(
        suspended[np.abs(suspended) <= tolerance], 0.0)
    bed[np.abs(bed) <= tolerance] = np.maximum(
        bed[np.abs(bed) <= tolerance], 0.0)
    if (not np.isfinite(q).all() or q[0, cell] < 0.0 or
            not np.isfinite(suspended).all() or np.any(suspended < 0.0) or
            not np.isfinite(bed).all() or np.any(bed < 0.0) or
            not np.isfinite(ht).all() or not np.isfinite(hs).all() or
            np.any(hs < 0.0)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 refluxed conserved image is outside its domain")
    grid = _grid_execution(execution.active_execution)
    dry_tolerance = float(grid.dry_tolerance_m)
    wet = np.ascontiguousarray(q[0] > dry_tolerance)
    dry_transfer = np.zeros((ng, nc), dtype=np.float64)
    momentum_cleanup = np.zeros((2, nc), dtype=np.float64)
    fluid_cleanup = np.zeros((2, nc), dtype=np.float64)
    dry = ~wet
    if np.any(dry):
        dry_transfer[:, dry] = suspended[:, dry]
        momentum_cleanup[:, dry] = -q[1:, dry]
        fluid_cleanup[0, dry] = -ht[dry]
        fluid_cleanup[1, dry] = -hs[dry]
        bed[:, dry] += suspended[:, dry]
        suspended[:, dry] = 0.0
        q[1:, dry] += momentum_cleanup[:, dry]
        ht[dry] += fluid_cleanup[0, dry]
        hs[dry] += fluid_cleanup[1, dry]
    v92_execution = _v92_execution(execution.active_execution)
    temperature = np.full(
        nc, v92_execution.equilibrium_temperature_c, dtype=np.float64)
    salinity = np.full(
        nc, v92_execution.equilibrium_salinity_ppt, dtype=np.float64)
    if v92_execution.temperature_mode == MODE_TRANSPORTED:
        temperature[wet] = ht[wet]/q[0, wet]
    if v92_execution.salinity_mode == MODE_TRANSPORTED:
        salinity[wet] = hs[wet]/q[0, wet]
    salinity_tolerance = 4096.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.abs(salinity))
    if (np.any(temperature <= TEMPERATURE_FLOOR_C) or
            np.any(salinity < -salinity_tolerance)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 refluxed T/S is outside the accepted EOS domain")
    salinity = np.maximum(salinity, 0.0)
    water_density, viscosity = _fluid_properties(
        v92_execution, np.ascontiguousarray(temperature),
        np.ascontiguousarray(salinity))
    concentration = np.zeros((ng, nc), dtype=np.float64)
    concentration[:, wet] = suspended[:, wet]/q[0, wet]
    mixture = mixture_density_kg_m3_v85(
        concentration, water_density,
        execution.hydraulic_execution.base_execution.base_execution.
        solid_density_kg_m3)
    elevation = _bed_from_mass(execution, bed)
    eta = np.ascontiguousarray(q[0]+elevation)
    corrected_water_flux = _array(
        correction.corrected_face_water_flux_m2_s, (nf,),
        "corrected water face flux")
    corrected_bed_flux = _array(
        correction.corrected_face_bedload_flux_kg_m_s, (ng, nf),
        "corrected bedload face flux")
    values = dict(
        q=q, suspended=suspended, bed=bed, ht=ht, hs=hs,
        temperature=temperature, salinity=salinity,
        water_density=water_density, mixture_density=mixture,
        viscosity=viscosity, bed_elevation=elevation, eta=eta,
        wet_mask=wet.astype(np.int32), dry_transfer=dry_transfer,
        momentum_cleanup=momentum_cleanup,
        fluid_cleanup=fluid_cleanup,
        corrected_water_flux=corrected_water_flux,
        corrected_bed_flux=corrected_bed_flux)
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.ascontiguousarray(value)
        array.setflags(write=False)
        result[name] = array
    return result


def _native_input_arrays(
    execution: CoupledWetDryCPUExecutionV97,
    provisional: CoupledWetDryCPUStateV97,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> tuple[np.ndarray, ...]:
    v92 = _v92_state(provisional.active_state)
    return (
        np.ascontiguousarray(v92.q),
        np.ascontiguousarray(
            v92.suspended_state.suspended_mass_kg_m2),
        np.ascontiguousarray(
            provisional.active_state.base_state.bed_mass_kg_m2),
        np.ascontiguousarray(np.vstack((
            v92.h_temperature_c_m, v92.h_salinity_ppt_m))),
        np.ascontiguousarray(np.concatenate((
            np.atleast_1d(correction.water_depth_delta_m),
            correction.momentum_delta_m2_s))),
        np.ascontiguousarray(correction.suspended_delta_kg_m2),
        np.ascontiguousarray(correction.bed_delta_kg_m2),
        np.ascontiguousarray(correction.fluid_delta),
    )


def _native_controls(
    execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
) -> dict[str, object]:
    v92_execution = _v92_execution(execution.active_execution)
    grid = _grid_execution(execution.active_execution)
    return {
        "cell": interface.coarse_leaf_cell,
        "dry_tolerance_m": float(grid.dry_tolerance_m),
        "temperature_mode": v92_execution.temperature_mode,
        "salinity_mode": v92_execution.salinity_mode,
        "equilibrium_temperature_c":
            float(v92_execution.equilibrium_temperature_c),
        "equilibrium_salinity_ppt":
            float(v92_execution.equilibrium_salinity_ppt),
        "density_eos": v92_execution.density_eos,
        "reference_water_density_kg_m3":
            float(grid.profile.water_density_kg_m3),
    }


def _host_as_native_images(
    host: dict[str, np.ndarray], ng: int, nc: int,
) -> dict[str, np.ndarray]:
    concentration = np.zeros((ng, nc), dtype=np.float64)
    wet = np.asarray(host["wet_mask"], dtype=np.bool_)
    concentration[:, wet] = (
        host["suspended"][:, wet]/host["q"][0, wet])
    return {
        "q": host["q"], "suspended": host["suspended"],
        "bed": host["bed"],
        "fluid": np.ascontiguousarray(np.vstack((host["ht"], host["hs"]))),
        "concentration": np.ascontiguousarray(concentration),
        "temperature": host["temperature"],
        "salinity": host["salinity"],
        "bed_elevation": host["bed_elevation"], "eta": host["eta"],
        "wet_mask": host["wet_mask"],
        "dry_transfer": host["dry_transfer"],
        "momentum_cleanup": host["momentum_cleanup"],
        "fluid_cleanup": host["fluid_cleanup"],
        "water_density": host["water_density"],
        "viscosity": host["viscosity"],
        "mixture_density": host["mixture_density"],
    }


def _native_state_output_images(
    value: CoupledAMRCorrectedLeafStateCPUV98,
) -> dict[str, np.ndarray]:
    return {
        "q": value.q, "suspended": value.suspended_mass_kg_m2,
        "bed": value.bed_mass_kg_m2,
        "fluid": np.ascontiguousarray(np.vstack((
            value.h_temperature_c_m, value.h_salinity_ppt_m))),
        "concentration": value.suspended_concentration_kg_m3,
        "temperature": value.temperature_cell_c,
        "salinity": value.salinity_cell_ppt,
        "bed_elevation": value.bed_elevation_m,
        "eta": value.free_surface_elevation_m,
        "wet_mask": value.wet_mask,
        "dry_transfer": value.reflux_dry_transfer_to_bed_kg_m2,
        "momentum_cleanup": value.reflux_dry_momentum_cleanup_m2_s,
        "fluid_cleanup": value.reflux_dry_fluid_cleanup,
        "water_density": value.water_density_cell_kg_m3,
        "viscosity": value.molecular_viscosity_cell_m2_s,
        "mixture_density": value.mixture_density_cell_kg_m3,
    }


def _native_result_metadata(
    evidence: NativeCorrectedImagesEvidenceCPUV98,
) -> dict[str, object]:
    return {
        "runtime_sha256": evidence.runtime_sha256,
        "backend": "native_cpu",
        "native_threads": evidence.native_threads,
        "input_sha256": evidence.input_sha256,
        "output_sha256": evidence.output_sha256,
        "correction_receipt_sha256": evidence.correction_receipt_sha256,
        "source_state_sha256": evidence.source_state_sha256,
        "controls_sha256": evidence.controls_sha256,
        "reflux_correction_application_count": 1,
        "reflux_dry_cleanup_owner_count": 1,
        "water_geometry_source_count": 0,
        "settling_exchange_q3_replay_count": 0,
        "hles_evaluation_count": 0,
        "dss_recompute_count": 0,
        "numerical_python_cell_face_loops": 0,
        "state_or_clock_publication_count": 0,
        "production_admission": False,
    }


def _validate_native_runtime_geometry(
    execution: CoupledWetDryCPUExecutionV97,
    runtime: NativeAMRCorrectedImagesRuntimeV98,
) -> None:
    grid = _grid_execution(execution.active_execution)
    if (type(runtime) is not NativeAMRCorrectedImagesRuntimeV98 or
            runtime.ng != execution.fraction_count or
            runtime.nc != execution.cell_count or
            len(runtime.geometry_host) != 4 or
            not np.array_equal(runtime.geometry_host[0],
                               execution.initial_bed_elevation_m) or
            not np.array_equal(runtime.geometry_host[1],
                               execution.reference_bed_mass_kg_m2) or
            not np.array_equal(runtime.geometry_host[2],
                               execution.dry_bulk_density_kg_m3) or
            not np.array_equal(runtime.geometry_host[3],
                               grid.profile.sediment_density_kg_m3)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected-image geometry/runtime differs")


def _evaluate_native_corrected_images(
    execution: CoupledWetDryCPUExecutionV97,
    provisional: CoupledWetDryCPUStateV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
    runtime: NativeAMRCorrectedImagesRuntimeV98, *,
    cancelled: bool | Callable[[], bool] = False,
) -> tuple[dict[str, np.ndarray], NativeCorrectedImagesEvidenceCPUV98]:
    """Use native images as candidate; Python remains an independent oracle."""
    _validate_native_runtime_geometry(execution, runtime)
    inputs = _native_input_arrays(execution, provisional, correction)
    controls = _native_controls(execution, interface)
    native_threads = execution.hydraulic_execution.native_threads
    result = runtime.evaluate(
        *inputs, **controls,
        correction_receipt_sha256=correction.receipt_sha256,
        source_state_sha256=provisional.state_sha256,
        native_threads=native_threads, cancelled=cancelled)
    if (not isinstance(result, dict) or
            set(result) != {"images", "metadata", "receipt_sha256"} or
            not isinstance(result["images"], dict) or
            tuple(result["images"]) != _NATIVE_CORRECTED_IMAGE_KEYS or
            not isinstance(result["metadata"], dict)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected-image result inventory differs")
    native = result["images"]
    host = _derive_corrected_images(
        execution, provisional, interface, correction)
    host_native = _host_as_native_images(
        host, execution.fraction_count, execution.cell_count)
    exact_keys = (
        "q", "suspended", "bed", "fluid", "wet_mask",
        "dry_transfer", "momentum_cleanup", "fluid_cleanup")
    if any(not np.array_equal(native[key], host_native[key])
           for key in exact_keys):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected conserved/cleanup image differs from oracle")
    errors = tuple(float(np.max(np.abs(
        np.asarray(native[key], dtype=np.float64)-
        np.asarray(host_native[key], dtype=np.float64))))
        for key in _NATIVE_CORRECTED_IMAGE_KEYS)
    for key in _NATIVE_CORRECTED_IMAGE_KEYS:
        if key in exact_keys:
            continue
        if not np.allclose(native[key], host_native[key],
                           rtol=NATIVE_ORACLE_RTOL,
                           atol=NATIVE_ORACLE_ATOL):
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                f"v98 native corrected derived {key} differs from oracle")
    expected_input = _native_image_digest(*inputs)
    expected_output = _native_image_digest(
        *(native[key] for key in _NATIVE_CORRECTED_IMAGE_KEYS))
    expected_controls = _native_image_hash(controls)
    metadata = result["metadata"]
    if (metadata.get("runtime_sha256") !=
            runtime.receipt["receipt_sha256"] or
            metadata.get("input_sha256") != expected_input or
            metadata.get("output_sha256") != expected_output or
            metadata.get("controls_sha256") != expected_controls or
            metadata.get("source_state_sha256") != provisional.state_sha256 or
            metadata.get("correction_receipt_sha256") !=
                correction.receipt_sha256 or
            metadata.get("native_threads") != native_threads or
            result["receipt_sha256"] != _native_image_hash(metadata)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected-image metadata differs")
    exact_digest = _digest(*(native[key] for key in exact_keys))
    placeholder = NativeCorrectedImagesEvidenceCPUV98(
        runtime_sha256=runtime.receipt["receipt_sha256"],
        input_sha256=expected_input, output_sha256=expected_output,
        controls_sha256=expected_controls,
        source_state_sha256=provisional.state_sha256,
        correction_receipt_sha256=correction.receipt_sha256,
        native_result_receipt_sha256=result["receipt_sha256"],
        native_threads=native_threads,
        host_oracle_keys=tuple(_NATIVE_CORRECTED_IMAGE_KEYS),
        host_oracle_max_abs=errors,
        exact_conserved_cleanup_sha256=exact_digest,
        native_owner_count=1, host_oracle_validation_count=1,
        python_numerical_owner_count=0,
        state_or_clock_publication_count=0, evidence_sha256="")
    evidence = NativeCorrectedImagesEvidenceCPUV98(
        **{**placeholder.__dict__,
           "evidence_sha256": _hash(_native_evidence_payload(placeholder))})
    images = {
        "q": native["q"], "suspended": native["suspended"],
        "bed": native["bed"], "ht": native["fluid"][0],
        "hs": native["fluid"][1],
        "concentration": native["concentration"],
        "temperature": native["temperature"],
        "salinity": native["salinity"],
        "water_density": native["water_density"],
        "mixture_density": native["mixture_density"],
        "viscosity": native["viscosity"],
        "bed_elevation": native["bed_elevation"], "eta": native["eta"],
        "wet_mask": native["wet_mask"],
        "dry_transfer": native["dry_transfer"],
        "momentum_cleanup": native["momentum_cleanup"],
        "fluid_cleanup": native["fluid_cleanup"],
        "corrected_water_flux": host["corrected_water_flux"],
        "corrected_bed_flux": host["corrected_bed_flux"],
    }
    for key, value in tuple(images.items()):
        array = np.ascontiguousarray(value)
        array.setflags(write=False)
        images[key] = array
    return images, evidence


def _validate_native_corrected_images_evidence(
    value: CoupledAMRCorrectedLeafStateCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
    host: dict[str, np.ndarray],
) -> None:
    evidence = value.native_corrected_images_evidence
    if (type(evidence) is not NativeCorrectedImagesEvidenceCPUV98 or
            evidence.evidence_sha256 !=
                _hash(_native_evidence_payload(evidence)) or
            evidence.source_state_sha256 !=
                value.provisional_source_state.state_sha256 or
            evidence.correction_receipt_sha256 !=
                correction.receipt_sha256 or
            evidence.native_threads !=
                execution.hydraulic_execution.native_threads or
            evidence.host_oracle_keys !=
                tuple(_NATIVE_CORRECTED_IMAGE_KEYS) or
            len(evidence.host_oracle_max_abs) !=
                len(_NATIVE_CORRECTED_IMAGE_KEYS) or
            evidence.native_owner_count != 1 or
            evidence.host_oracle_validation_count != 1 or
            evidence.python_numerical_owner_count != 0 or
            evidence.state_or_clock_publication_count != 0):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected-image evidence identity differs")
    inputs = _native_input_arrays(
        execution, value.provisional_source_state, correction)
    controls = _native_controls(execution, interface)
    native = _native_state_output_images(value)
    host_native = _host_as_native_images(
        host, execution.fraction_count, execution.cell_count)
    exact_keys = (
        "q", "suspended", "bed", "fluid", "wet_mask",
        "dry_transfer", "momentum_cleanup", "fluid_cleanup")
    errors = tuple(float(np.max(np.abs(
        np.asarray(native[key], dtype=np.float64)-
        np.asarray(host_native[key], dtype=np.float64))))
        for key in _NATIVE_CORRECTED_IMAGE_KEYS)
    if (evidence.input_sha256 != _native_image_digest(*inputs) or
            evidence.output_sha256 != _native_image_digest(
                *(native[key] for key in _NATIVE_CORRECTED_IMAGE_KEYS)) or
            evidence.controls_sha256 != _native_image_hash(controls) or
            evidence.host_oracle_max_abs != errors or
            evidence.exact_conserved_cleanup_sha256 !=
                _digest(*(native[key] for key in exact_keys)) or
            any(not np.array_equal(native[key], host_native[key])
                for key in exact_keys)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected-image inputs, outputs, or exact oracle "
            "comparison differ")
    for key in _NATIVE_CORRECTED_IMAGE_KEYS:
        if key not in exact_keys and not np.allclose(
                native[key], host_native[key], rtol=NATIVE_ORACLE_RTOL,
                atol=NATIVE_ORACLE_ATOL):
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                f"v98 native corrected {key} no longer matches host oracle")
    metadata = _native_result_metadata(evidence)
    if evidence.native_result_receipt_sha256 != _native_image_hash(metadata):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 native corrected-image result receipt differs")


def _corrected_q3_state(
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_prepared: CPUQ3V84MixedPreparedCommitV93,
    corrected_bed: np.ndarray,
    corrected_face_bedload_flux: np.ndarray,
) -> CPUQ3V84MixedStateV93:
    source = provisional_prepared.rebound_state
    result = resolve_cpu_q3_v84_mixed_state_v93(
        execution_fingerprint=provider.execution_fingerprint,
        binding=provider.binding,
        authoritative_bed_mass_kg_m2=corrected_bed,
        bulk_density_kg_m3=source.bulk_density_kg_m3,
        retained_fraction=source.active_fraction,
        accepted_face_flux_kg_m_s=corrected_face_bedload_flux,
        accepted_cell_availability=source.accepted_cell_availability,
        named_boundary_positive_in_kg=
            source.named_boundary_positive_in_kg,
        named_boundary_absolute_kg=
            source.named_boundary_absolute_kg,
        depletion_roundoff_positive_in_kg=
            source.depletion_roundoff_positive_in_kg,
        accepted_updates=source.accepted_updates,
        time_s=source.time_s)
    validate_cpu_q3_v84_mixed_state_v93(
        result, execution_fingerprint=provider.execution_fingerprint,
        binding=provider.binding)
    return result


def _corrected_prepared(
    provisional: CPUQ3V84MixedPreparedCommitV93,
    corrected_q3: CPUQ3V84MixedStateV93,
) -> CPUQ3V84MixedPreparedCommitV93:
    placeholder = CPUQ3V84MixedPreparedCommitV93(
        provisional.open_trial, corrected_q3,
        _q3_array_digest(corrected_q3.authoritative_bed_mass_kg_m2), "")
    return CPUQ3V84MixedPreparedCommitV93(
        placeholder.open_trial, placeholder.rebound_state,
        placeholder.authoritative_bed_sha256,
        _q3_prepared_token(placeholder))


def _wrap_state(
    execution: CoupledWetDryCPUExecutionV97, *,
    provisional: CoupledWetDryCPUStateV97,
    correction: CoupledAMRCoarseCorrectionCPUV98,
    images: dict[str, np.ndarray],
    provisional_q3_state: CPUQ3V84MixedStateV93,
    q3_state: CPUQ3V84MixedStateV93,
    hles_state: HlesProducerStateV90,
    native_corrected_images_evidence: NativeCorrectedImagesEvidenceCPUV98,
    accepted_chain_sha256: str,
) -> CoupledAMRCorrectedLeafStateCPUV98:
    _sha(accepted_chain_sha256, "accepted chain")
    counts = CoupledAMRCorrectedLeafCountsCPUV98()
    candidate_dss = _array(
        provisional.active_state.suspended_diameter_m,
        (execution.fraction_count, execution.cell_count),
        "carried candidate DSS", nonnegative=True)
    if np.any(candidate_dss <= 0.0):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 carried DSS is nonpositive")
    placeholder = CoupledAMRCorrectedLeafStateCPUV98(
        execution_fingerprint=execution.fingerprint,
        provisional_source_state=provisional,
        correction_receipt_sha256=correction.receipt_sha256,
        internal_interface_ledger_sha256=
            correction.internal_interface_ledger.receipt_sha256,
        q=images["q"], suspended_mass_kg_m2=images["suspended"],
        bed_mass_kg_m2=images["bed"],
        h_temperature_c_m=images["ht"], h_salinity_ppt_m=images["hs"],
        temperature_cell_c=images["temperature"],
        salinity_cell_ppt=images["salinity"],
        water_density_cell_kg_m3=images["water_density"],
        mixture_density_cell_kg_m3=images["mixture_density"],
        molecular_viscosity_cell_m2_s=images["viscosity"],
        bed_elevation_m=images["bed_elevation"],
        free_surface_elevation_m=images["eta"],
        wet_mask=images["wet_mask"], suspended_diameter_m=candidate_dss,
        reflux_dry_transfer_to_bed_kg_m2=images["dry_transfer"],
        reflux_dry_momentum_cleanup_m2_s=images["momentum_cleanup"],
        reflux_dry_fluid_cleanup=images["fluid_cleanup"],
        corrected_face_water_flux_m2_s=images["corrected_water_flux"],
        corrected_face_bedload_flux_kg_m_s=images["corrected_bed_flux"],
        suspended_concentration_kg_m3=images["concentration"],
        native_corrected_images_evidence=native_corrected_images_evidence,
        provisional_q3_state=provisional_q3_state,
        q3_state=q3_state, hles_state=hles_state,
        accepted_updates=provisional.accepted_updates,
        time_s=provisional.time_s,
        accepted_chain_sha256=accepted_chain_sha256, counts=counts,
        semantic_id=SEMANTIC_ID, backend=BACKEND, state_sha256="")
    return CoupledAMRCorrectedLeafStateCPUV98(
        **{**placeholder.__dict__,
           "state_sha256": _hash(_state_payload(placeholder))})


def begin_coupled_amr_corrected_coarse_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_trial: CoupledWetDryCPUTrialV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse_integrals: CoupledFaceTimeIntegralsCPUV98,
    fine_integrals: Sequence[CoupledFaceTimeIntegralsCPUV98],
    correction: CoupledAMRCoarseCorrectionCPUV98, *,
    fine_execution: CoupledWetDryCPUExecutionV97,
    hles_runtime: NativeHlesRuntimeV97,
    native_corrected_images_runtime: NativeAMRCorrectedImagesRuntimeV98,
    closure_chezy_mhalf_s: object,
    cancelled: bool | Callable[[], bool] = False,
) -> CoupledAMRCorrectedLeafTrialCPUV98:
    """Build the corrected leaf without mutating the live Q3 provider."""
    execution_document_coupled_wetdry_v97(execution)
    _validate_v97_trial(execution, parent, provider, provisional_trial)
    fine = tuple(fine_integrals)
    validate_coupled_amr_coarse_correction_cpu_v98(
        correction, interface, coarse_integrals, fine,
        coarse_execution=execution, fine_execution=fine_execution)
    if (type(provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            coarse_integrals.source_trial_sha256 !=
                provisional_trial.trial_sha256 or
            coarse_integrals.candidate_state_sha256 !=
                provisional_trial.candidate.state_sha256 or
            correction.coarse_parent_state_sha256 != parent.state_sha256 or
            correction.fine_final_state_sha256 !=
                fine[-1].candidate_state_sha256 or
            provider._open is not
                provisional_trial.active_trial.base_v93_trial.q3_open_trial or
            provider._prepared is not
                provisional_trial.active_trial.base_v93_trial.
                q3_prepared_commit or
            provider._closed is not None):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 correction source trial/provider identity differs")
    if _cancelled(cancelled):
        raise InterruptedError("v98 corrected leaf cancelled before reflux")
    images, native_images_evidence = _evaluate_native_corrected_images(
        execution, provisional_trial.candidate, interface, correction,
        native_corrected_images_runtime, cancelled=cancelled)
    provisional_prepared = (
        provisional_trial.active_trial.base_v93_trial.q3_prepared_commit)
    q3_state = _corrected_q3_state(
        provider, provisional_prepared, images["bed"],
        images["corrected_bed_flux"])
    q3_prepared = _corrected_prepared(provisional_prepared, q3_state)
    if _cancelled(cancelled):
        raise InterruptedError(
            "v98 corrected leaf cancelled before corrected HLES")
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    if (type(hles_runtime) is not NativeHlesRuntimeV97 or
            hles_runtime.receipt["receipt_sha256"] !=
                execution.hles_runtime_sha256):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected HLES runtime identity differs")
    hles_arrays = _hles_inputs(
        execution, images["q"], images["corrected_water_flux"],
        closure_chezy_mhalf_s)
    provisional_v92_trial = (
        provisional_trial.active_trial.base_v93_trial.base_v92_trial)
    accepted_step = provisional_v92_trial.hydraulic_receipt.base_step_receipt
    dt = float(provisional_v92_trial.hydraulic_receipt.dt_s)
    hles_input = make_hles_input_receipt_v90(
        hles_execution, parent.hles_state,
        accepted_step_receipt=accepted_step,
        current_handoff=provisional_trial.current_handoff,
        closure_interval_s=dt,
        accepted_face_normal_velocity_m_s=hles_arrays[0],
        active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
        closure_chezy_mhalf_s=hles_arrays[3], active_cell=hles_arrays[4])
    hles_trial, hles_evidence = hles_runtime.begin_trial(
        hles_execution, parent.hles_state,
        accepted_step_receipt=accepted_step,
        hles_input_receipt=hles_input,
        current_handoff=provisional_trial.current_handoff,
        accepted_face_normal_velocity_m_s=hles_arrays[0],
        active_face=hles_arrays[1], closure_depth_m=hles_arrays[2],
        closure_chezy_mhalf_s=hles_arrays[3], active_cell=hles_arrays[4],
        native_threads=execution.hydraulic_execution.native_threads,
        cancelled=cancelled)
    chain = _hash({
        "previous": parent.accepted_chain_sha256,
        "provisional_trial": provisional_trial.trial_sha256,
        "correction": correction.receipt_sha256,
        "internal_interface_ledger":
            correction.internal_interface_ledger.receipt_sha256,
        "native_corrected_images":
            native_images_evidence.evidence_sha256,
        "corrected_q3": q3_state.state_sha256,
        "carried_dss": _digest(
            provisional_trial.candidate.active_state.suspended_diameter_m),
        "native_hles": hles_evidence.result_sha256,
        "hles": hles_trial.trial_sha256,
    })
    candidate = _wrap_state(
        execution, provisional=provisional_trial.candidate,
        correction=correction, images=images,
        provisional_q3_state=provisional_prepared.rebound_state,
        q3_state=q3_state,
        hles_state=hles_trial.candidate,
        native_corrected_images_evidence=native_images_evidence,
        accepted_chain_sha256=chain)
    counts = CoupledAMRCorrectedLeafCountsCPUV98()
    placeholder = CoupledAMRCorrectedLeafTrialCPUV98(
        execution_fingerprint=execution.fingerprint,
        parent_state_sha256=parent.state_sha256,
        provisional_trial_sha256=provisional_trial.trial_sha256,
        interface_fingerprint=interface.fingerprint,
        coarse_integrals_sha256=coarse_integrals.receipt_sha256,
        fine_integrals_sha256=tuple(
            item.receipt_sha256 for item in fine),
        correction=correction, corrected_q3_prepared=q3_prepared,
        corrected_hles_trial=hles_trial,
        native_hles_evidence=hles_evidence, hles_inputs=hles_arrays,
        counts=counts, candidate=candidate, provider_mutation_count=0,
        accepted_state_publication_count=0, backend=BACKEND,
        trial_sha256="")
    trial = CoupledAMRCorrectedLeafTrialCPUV98(
        **{**placeholder.__dict__,
           "trial_sha256": _hash(_trial_payload(placeholder))})
    validate_coupled_amr_corrected_coarse_trial_v98(
        execution, parent, provider, provisional_trial, interface,
        coarse_integrals, fine, trial, fine_execution=fine_execution)
    return trial


def _validate_native_hles(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    trial: CoupledAMRCorrectedLeafTrialCPUV98,
) -> None:
    evidence = trial.native_hles_evidence
    hles = trial.corrected_hles_trial
    if (type(evidence) is not NativeHlesCandidateV97 or
            evidence.native_runtime_sha256 != execution.hles_runtime_sha256 or
            evidence.parent_state_sha256 != parent.hles_state.state_sha256 or
            evidence.execution_fingerprint !=
                _v92_execution(execution.active_execution).base_execution.
                hles_execution.fingerprint or
            evidence.native_threads !=
                execution.hydraulic_execution.native_threads or
            evidence.filter_owner_count != 1 or
            evidence.reconstruction_owner_count != 1 or
            evidence.protke_owner_count != 1 or
            evidence.detvic_owner_count != 1 or
            evidence.python_numerical_cell_face_loop_count != 0 or
            evidence.native_candidate_launch_count != 1 or
            evidence.state_or_clock_publication_count != 0 or
            hles.candidate.state_sha256 !=
                trial.candidate.hles_state.state_sha256):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected native HLES evidence differs")
    inputs = trial.hles_inputs
    interval = trial.corrected_hles_trial.accepted_dt_s
    expected_input = _native_hles._hash(dict(
        execution=evidence.execution_fingerprint,
        parent=parent.hles_state.state_sha256,
        fields=_native_hles._digest(*inputs), interval=interval))
    values = evidence.values
    if not isinstance(values, tuple) or len(values) != 13:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected native HLES output inventory differs")
    expected_result = _native_hles._hash(dict(
        runtime=evidence.native_runtime_sha256, inputs=expected_input,
        coefficient=float(values[0]),
        outputs=_native_hles._digest(*values[1:]),
        native_threads=evidence.native_threads))
    comparable = (
        hles.filter_coefficient,
        hles.filtered_face_normal_velocity_m_s,
        hles.fluctuating_face_normal_velocity_m_s,
        hles.reconstructed_mean_velocity_m_s,
        hles.reconstructed_fluctuating_velocity_m_s,
        hles.reconstruction_complete, hles.boundary_cell,
        hles.detvic_active_cell, hles.ptke_s2, hles.hles_vicuv_m2_s,
        hles.elder_vicuv_m2_s,
        hles.output_receipt.candidate_dynamic_vicuv_m2_s,
        hles.output_receipt.candidate_dynamic_dicuv_m2_s)
    if (evidence.inputs_sha256 != expected_input or
            evidence.result_sha256 != expected_result or
            float(comparable[0]) != float(values[0]) or
            any(not np.array_equal(left, right)
                for left, right in zip(comparable[1:], values[1:]))):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected HLES arrays differ from native evidence")


def _validate_conservation(
    execution: CoupledWetDryCPUExecutionV97,
    provisional: CoupledWetDryCPUStateV97,
    candidate: CoupledAMRCorrectedLeafStateCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> None:
    area = np.asarray(_grid_execution(
        execution.active_execution).cell_area_m2)
    source = _v92_state(provisional.active_state)
    water_delta = float(np.sum((candidate.q[0]-source.q[0])*area))
    momentum_delta = np.sum(
        (candidate.q[1:]-source.q[1:])*area[None, :], axis=1)
    fluid_delta = np.array((
        np.sum((candidate.h_temperature_c_m-
                source.h_temperature_c_m)*area),
        np.sum((candidate.h_salinity_ppt_m-
                source.h_salinity_ppt_m)*area)))
    source_bed = provisional.active_state.base_state.bed_mass_kg_m2
    total_class_delta = np.sum(
        (candidate.suspended_mass_kg_m2-
         source.suspended_state.suspended_mass_kg_m2+
         candidate.bed_mass_kg_m2-source_bed)*area[None, :], axis=1)
    ledger = correction.internal_interface_ledger
    actual = np.concatenate((
        np.atleast_1d(water_delta), momentum_delta, fluid_delta,
        total_class_delta))
    expected = np.concatenate((
        np.atleast_1d(ledger.water_volume_m3), ledger.momentum_m4_s,
        ledger.fluid_integral, ledger.suspended_mass_kg+ledger.bed_mass_kg))
    expected[1:3] += np.sum(
        candidate.reflux_dry_momentum_cleanup_m2_s*area[None, :], axis=1)
    expected[3:5] += np.array((
        np.sum(candidate.reflux_dry_fluid_cleanup[0]*area),
        np.sum(candidate.reflux_dry_fluid_cleanup[1]*area)))
    scale = np.maximum(1.0, np.maximum(np.abs(actual), np.abs(expected)))
    tolerance = 131072.0*np.finfo(np.float64).eps*scale
    if np.any(np.abs(actual-expected) > tolerance):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected internal-interface conservation differs")


def validate_coupled_amr_corrected_coarse_state_v98(
    value: CoupledAMRCorrectedLeafStateCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
    *, provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
) -> None:
    if (type(value) is not CoupledAMRCorrectedLeafStateCPUV98 or
            value.execution_fingerprint != execution.fingerprint or
            value.correction_receipt_sha256 != correction.receipt_sha256 or
            value.internal_interface_ledger_sha256 !=
                correction.internal_interface_ledger.receipt_sha256 or
            value.counts != CoupledAMRCorrectedLeafCountsCPUV98() or
            value.semantic_id != SEMANTIC_ID or value.backend != BACKEND or
            value.accepted_updates !=
                value.provisional_source_state.accepted_updates or
            value.time_s != value.provisional_source_state.time_s or
            value.state_sha256 != _hash(_state_payload(value))):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected leaf state identity differs")
    validate_coupled_wetdry_cpu_state_v97(
        value.provisional_source_state, execution)
    images = _derive_corrected_images(
        execution, value.provisional_source_state, interface, correction)
    actual_by_name = {
        "q": value.q, "suspended": value.suspended_mass_kg_m2,
        "bed": value.bed_mass_kg_m2,
        "ht": value.h_temperature_c_m, "hs": value.h_salinity_ppt_m,
        "temperature": value.temperature_cell_c,
        "salinity": value.salinity_cell_ppt,
        "water_density": value.water_density_cell_kg_m3,
        "mixture_density": value.mixture_density_cell_kg_m3,
        "viscosity": value.molecular_viscosity_cell_m2_s,
        "bed_elevation": value.bed_elevation_m,
        "eta": value.free_surface_elevation_m, "wet_mask": value.wet_mask,
        "dry_transfer": value.reflux_dry_transfer_to_bed_kg_m2,
        "momentum_cleanup": value.reflux_dry_momentum_cleanup_m2_s,
        "fluid_cleanup": value.reflux_dry_fluid_cleanup,
        "corrected_water_flux": value.corrected_face_water_flux_m2_s,
        "corrected_bed_flux": value.corrected_face_bedload_flux_kg_m_s,
    }
    exact_names = (
        "q", "suspended", "bed", "ht", "hs", "wet_mask",
        "dry_transfer", "momentum_cleanup", "fluid_cleanup",
        "corrected_water_flux", "corrected_bed_flux")
    if any(not np.array_equal(actual_by_name[name], images[name])
           for name in exact_names):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected leaf conserved, cleanup, or flux image differs")
    for name in (
            "temperature", "salinity", "water_density", "mixture_density",
            "viscosity", "bed_elevation", "eta"):
        if not np.allclose(actual_by_name[name], images[name],
                           rtol=NATIVE_ORACLE_RTOL,
                           atol=NATIVE_ORACLE_ATOL):
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                f"v98 corrected leaf derived {name} differs")
    actual = tuple(actual_by_name.values()) + (
        value.suspended_concentration_kg_m3,)
    _validate_native_corrected_images_evidence(
        value, execution, interface, correction, images)
    for array in (*actual, value.suspended_diameter_m):
        if array.flags.writeable or not array.flags.c_contiguous:
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                "v98 corrected leaf array ownership differs")
    if (not np.array_equal(
            value.suspended_diameter_m,
            value.provisional_source_state.active_state.
            suspended_diameter_m) or
            np.any(value.q[1:, value.wet_mask == 0] != 0.0) or
            np.any(value.suspended_mass_kg_m2[:, value.wet_mask == 0] !=
                   0.0) or
            np.any(value.h_temperature_c_m[value.wet_mask == 0] != 0.0) or
            np.any(value.h_salinity_ppt_m[value.wet_mask == 0] != 0.0)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 carried DSS or accepted dry image differs")
    if type(provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected Q3 provider type differs")
    validate_cpu_q3_v84_mixed_state_v93(
        value.provisional_q3_state,
        execution_fingerprint=provider.execution_fingerprint,
        binding=provider.binding)
    validate_cpu_q3_v84_mixed_state_v93(
        value.q3_state,
        execution_fingerprint=provider.execution_fingerprint,
        binding=provider.binding)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    validate_hles_producer_state_v90(value.hles_state, hles_execution)
    outside = np.ones(execution.cell_count, dtype=np.bool_)
    outside[interface.coarse_leaf_cell] = False
    source_v93 = value.provisional_source_state.active_state.base_state
    if (value.provisional_q3_state.state_sha256 !=
            source_v93.q3_provider_state_sha256 or
            not np.array_equal(
                value.provisional_q3_state.authoritative_bed_mass_kg_m2,
                source_v93.bed_mass_kg_m2) or
            value.provisional_q3_state.accepted_updates !=
                value.accepted_updates or
            value.provisional_q3_state.time_s != value.time_s or
            value.q3_state.accepted_updates != value.accepted_updates or
            value.q3_state.time_s != value.time_s or
            value.hles_state.accepted_updates != value.accepted_updates or
            value.hles_state.time_s != value.time_s or
            not np.array_equal(
                value.q3_state.authoritative_bed_mass_kg_m2,
                value.bed_mass_kg_m2) or
            not np.array_equal(
                value.q3_state.accepted_face_flux_kg_m_s,
                value.corrected_face_bedload_flux_kg_m_s) or
            np.any(value.reflux_dry_transfer_to_bed_kg_m2[:, outside] !=
                   0.0) or
            np.any(value.reflux_dry_momentum_cleanup_m2_s[:, outside] !=
                   0.0) or
            np.any(value.reflux_dry_fluid_cleanup[:, outside] != 0.0)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected provider, clock, or leaf cleanup differs")
    _validate_conservation(
        execution, value.provisional_source_state, value, correction)


def validate_coupled_amr_corrected_coarse_trial_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_trial: CoupledWetDryCPUTrialV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse_integrals: CoupledFaceTimeIntegralsCPUV98,
    fine_integrals: Sequence[CoupledFaceTimeIntegralsCPUV98],
    trial: CoupledAMRCorrectedLeafTrialCPUV98, *,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> None:
    _validate_v97_trial(execution, parent, provider, provisional_trial)
    fine = tuple(fine_integrals)
    validate_coupled_amr_coarse_correction_cpu_v98(
        trial.correction, interface, coarse_integrals, fine,
        coarse_execution=execution, fine_execution=fine_execution)
    if (type(trial) is not CoupledAMRCorrectedLeafTrialCPUV98 or
            trial.execution_fingerprint != execution.fingerprint or
            trial.parent_state_sha256 != parent.state_sha256 or
            trial.provisional_trial_sha256 !=
                provisional_trial.trial_sha256 or
            trial.interface_fingerprint != interface.fingerprint or
            trial.coarse_integrals_sha256 !=
                coarse_integrals.receipt_sha256 or
            trial.fine_integrals_sha256 !=
                tuple(item.receipt_sha256 for item in fine) or
            trial.counts != CoupledAMRCorrectedLeafCountsCPUV98() or
            trial.provider_mutation_count != 0 or
            trial.accepted_state_publication_count != 0 or
            trial.backend != BACKEND or
            trial.trial_sha256 != _hash(_trial_payload(trial))):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected leaf trial identity differs")
    provisional_prepared = (
        provisional_trial.active_trial.base_v93_trial.q3_prepared_commit)
    prepared = trial.corrected_q3_prepared
    if (type(prepared) is not CPUQ3V84MixedPreparedCommitV93 or
            prepared.open_trial is not provisional_prepared.open_trial or
            prepared.preparation_sha256 != _q3_prepared_token(prepared) or
            prepared.authoritative_bed_sha256 !=
                _q3_array_digest(trial.candidate.bed_mass_kg_m2) or
            prepared.rebound_state is not trial.candidate.q3_state or
            provider._open is not provisional_prepared.open_trial or
            provider._prepared is not provisional_prepared or
            provider._closed is not None):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected Q3 rebound or live provider phase differs")
    validate_cpu_q3_v84_mixed_state_v93(
        prepared.rebound_state,
        execution_fingerprint=provider.execution_fingerprint,
        binding=provider.binding)
    if (not np.array_equal(
            prepared.rebound_state.authoritative_bed_mass_kg_m2,
            trial.candidate.bed_mass_kg_m2) or
            not np.array_equal(
                prepared.rebound_state.accepted_face_flux_kg_m_s,
                trial.correction.corrected_face_bedload_flux_kg_m_s)):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected Q3 bed or accepted face flux differs")
    validate_hles_producer_state_v90(
        trial.candidate.hles_state,
        _v92_execution(execution.active_execution).base_execution.
        hles_execution)
    if rollback_hles_trial_v90(
            _v92_execution(execution.active_execution).base_execution.
            hles_execution, parent.hles_state,
            trial.corrected_hles_trial) is not parent.hles_state:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected HLES rollback changed its parent")
    _validate_native_hles(execution, parent, trial)
    validate_coupled_amr_corrected_coarse_state_v98(
        trial.candidate, execution, interface, trial.correction,
        provider=provider)


def close_coupled_amr_corrected_coarse_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_trial: CoupledWetDryCPUTrialV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse_integrals: CoupledFaceTimeIntegralsCPUV98,
    fine_integrals: Sequence[CoupledFaceTimeIntegralsCPUV98],
    trial: CoupledAMRCorrectedLeafTrialCPUV98, *,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> CoupledAMRCorrectedLeafClosedCPUV98:
    """Close the corrected Q3 child without publishing provider state."""
    validate_coupled_amr_corrected_coarse_trial_v98(
        execution, parent, provider, provisional_trial, interface,
        coarse_integrals, fine_integrals, trial,
        fine_execution=fine_execution)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    committed_hles = commit_hles_trial_v90(
        hles_execution, parent.hles_state, trial.corrected_hles_trial)
    if committed_hles is not trial.candidate.hles_state:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected HLES commit substituted its candidate")
    provisional_prepared = (
        provisional_trial.active_trial.base_v93_trial.q3_prepared_commit)
    try:
        provider._prepared = trial.corrected_q3_prepared
        closed_state = provider.close_commit(trial.corrected_q3_prepared)
        if closed_state is not trial.candidate.q3_state:
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                "v98 corrected Q3 close substituted its rebound state")
        placeholder = CoupledAMRCorrectedLeafClosedCPUV98(
            trial_sha256=trial.trial_sha256,
            corrected_q3_preparation_sha256=
                trial.corrected_q3_prepared.preparation_sha256,
            candidate_state_sha256=trial.candidate.state_sha256,
            provider_state_before_sha256=
                provisional_prepared.open_trial.parent_state.state_sha256,
            provider_state_after_sha256=
                trial.candidate.q3_state.state_sha256,
            q3_child_close_count=1, provider_publication_count=0,
            handle_sha256="")
        return CoupledAMRCorrectedLeafClosedCPUV98(
            **{**placeholder.__dict__,
               "handle_sha256": _hash(_closed_payload(placeholder))})
    except Exception:
        if provider._closed is trial.corrected_q3_prepared:
            provider.abort_closed_commit(trial.corrected_q3_prepared)
        provider._open = provisional_prepared.open_trial
        provider._prepared = provisional_prepared
        provider._closed = None
        raise


def publish_coupled_amr_corrected_coarse_cpu_v98(
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    trial: CoupledAMRCorrectedLeafTrialCPUV98,
    closed: CoupledAMRCorrectedLeafClosedCPUV98,
) -> CoupledAMRCorrectedLeafStateCPUV98:
    """Publish the already closed Q3 rebound; no fallible work follows."""
    if (type(provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            type(trial) is not CoupledAMRCorrectedLeafTrialCPUV98 or
            type(closed) is not CoupledAMRCorrectedLeafClosedCPUV98 or
            closed.handle_sha256 != _hash(_closed_payload(closed)) or
            closed.trial_sha256 != trial.trial_sha256 or
            closed.corrected_q3_preparation_sha256 !=
                trial.corrected_q3_prepared.preparation_sha256 or
            closed.candidate_state_sha256 != trial.candidate.state_sha256 or
            closed.provider_state_after_sha256 !=
                trial.candidate.q3_state.state_sha256 or
            closed.q3_child_close_count != 1 or
            closed.provider_publication_count != 0 or
            provider._closed is not trial.corrected_q3_prepared):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected closed handle differs")
    published = provider.publish_commit(trial.corrected_q3_prepared)
    if published is not trial.candidate.q3_state:
        raise RuntimeError("v98 corrected Q3 publication substituted state")
    return trial.candidate


def abort_closed_coupled_amr_corrected_coarse_cpu_v98(
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_trial: CoupledWetDryCPUTrialV97,
    trial: CoupledAMRCorrectedLeafTrialCPUV98,
    closed: CoupledAMRCorrectedLeafClosedCPUV98,
) -> None:
    """Undo an unpublished close and restore the provisional open phase."""
    provisional = (
        provisional_trial.active_trial.base_v93_trial.q3_prepared_commit)
    if (type(provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            type(closed) is not CoupledAMRCorrectedLeafClosedCPUV98 or
            closed.handle_sha256 != _hash(_closed_payload(closed)) or
            closed.trial_sha256 != trial.trial_sha256 or
            provider._closed is not trial.corrected_q3_prepared):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected abort handle differs")
    provider.abort_closed_commit(trial.corrected_q3_prepared)
    provider._open = provisional.open_trial
    provider._prepared = provisional
    provider._closed = None


def rollback_coupled_amr_corrected_coarse_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_trial: CoupledWetDryCPUTrialV97,
) -> CoupledWetDryCPUStateV97:
    """Rollback the still-open provisional v97 transaction exactly once."""
    return rollback_coupled_wetdry_cpu_trial_v97(
        execution, parent, provider, provisional_trial)


def commit_coupled_amr_corrected_coarse_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    parent: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    provisional_trial: CoupledWetDryCPUTrialV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    coarse_integrals: CoupledFaceTimeIntegralsCPUV98,
    fine_integrals: Sequence[CoupledFaceTimeIntegralsCPUV98],
    trial: CoupledAMRCorrectedLeafTrialCPUV98, *,
    fine_execution: CoupledWetDryCPUExecutionV97,
) -> CoupledAMRCorrectedLeafStateCPUV98:
    """Focused convenience commit; macro owners should use close/publish."""
    closed = close_coupled_amr_corrected_coarse_cpu_v98(
        execution, parent, provider, provisional_trial, interface,
        coarse_integrals, fine_integrals, trial,
        fine_execution=fine_execution)
    return publish_coupled_amr_corrected_coarse_cpu_v98(
        provider, trial, closed)


def _provider_clone(
    execution: CoupledWetDryCPUExecutionV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    state: CPUQ3V84MixedStateV93,
) -> CPUQ3V83ActiveWetDryExnerAdapterV97:
    clone = CPUQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=execution.active_execution,
        binding=provider.binding, state=state,
        native_factcr_runtime=provider.native_factcr_runtime,
        native_q3_face_runtime=provider.native_q3_face_runtime,
        native_threads=provider.native_factcr_threads)
    if (clone.state is not state or clone._open is not None or
            clone._prepared is not None or clone._closed is not None):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected provenance provider clone differs")
    return clone


def checkpoint_coupled_amr_corrected_coarse_cpu_v98(
    execution: CoupledWetDryCPUExecutionV97,
    state: CoupledAMRCorrectedLeafStateCPUV98,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> dict[str, object]:
    """Serialize one committed v98 leaf without a stale v97-state claim."""
    validate_coupled_amr_corrected_coarse_state_v98(
        state, execution, interface, correction, provider=provider)
    if (provider.state is not state.q3_state or provider._open is not None or
            provider._prepared is not None or provider._closed is not None):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected checkpoint requires its published Q3 state")
    source_provider = _provider_clone(
        execution, provider, state.provisional_q3_state)
    hles_execution = _v92_execution(
        execution.active_execution).base_execution.hles_execution
    body = {
        "schema": SCHEMA, "scope_id": CHECKPOINT_SCOPE_ID,
        "semantic_id": SEMANTIC_ID, "backend": BACKEND,
        "execution_fingerprint": execution.fingerprint,
        "interface_fingerprint": interface.fingerprint,
        "correction_receipt_sha256": correction.receipt_sha256,
        "internal_interface_ledger_sha256":
            correction.internal_interface_ledger.receipt_sha256,
        "native_corrected_images_runtime_sha256":
            state.native_corrected_images_evidence.runtime_sha256,
        "native_corrected_images_evidence_sha256":
            state.native_corrected_images_evidence.evidence_sha256,
        "provisional_source_checkpoint":
            coupled_wetdry_cpu_checkpoint_v97(
                execution, state.provisional_source_state,
                source_provider),
        "corrected_q3_checkpoint": provider.checkpoint(),
        "corrected_hles_checkpoint": hles_checkpoint_v90(
            hles_execution, state.hles_state),
        "accepted_chain_sha256": state.accepted_chain_sha256,
        "state_sha256": state.state_sha256,
        "counts": state.counts.__dict__,
        "corrected_image_rebuilt_not_serialized_as_v97": True,
    }
    return {**body, "fingerprint": _hash(body)}


def load_coupled_amr_corrected_coarse_checkpoint_v98(
    execution: CoupledWetDryCPUExecutionV97,
    document: object,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
    *, native_corrected_images_runtime: NativeAMRCorrectedImagesRuntimeV98,
) -> CoupledAMRCorrectedLeafStateCPUV98:
    """Atomically load a corrected leaf and its authoritative Q3 provider."""
    keys = {
        "schema", "scope_id", "semantic_id", "backend",
        "execution_fingerprint", "interface_fingerprint",
        "correction_receipt_sha256",
        "internal_interface_ledger_sha256",
        "native_corrected_images_runtime_sha256",
        "native_corrected_images_evidence_sha256",
        "provisional_source_checkpoint", "corrected_q3_checkpoint",
        "corrected_hles_checkpoint", "accepted_chain_sha256",
        "state_sha256", "counts",
        "corrected_image_rebuilt_not_serialized_as_v97", "fingerprint",
    }
    if not isinstance(document, dict) or set(document) != keys:
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected checkpoint inventory differs")
    _validate_native_runtime_geometry(
        execution, native_corrected_images_runtime)
    body = {key: value for key, value in document.items()
            if key != "fingerprint"}
    if (document["fingerprint"] != _hash(body) or
            document["schema"] != SCHEMA or
            document["scope_id"] != CHECKPOINT_SCOPE_ID or
            document["semantic_id"] != SEMANTIC_ID or
            document["backend"] != BACKEND or
            document["execution_fingerprint"] != execution.fingerprint or
            document["interface_fingerprint"] != interface.fingerprint or
            document["correction_receipt_sha256"] !=
                correction.receipt_sha256 or
            document["internal_interface_ledger_sha256"] !=
                correction.internal_interface_ledger.receipt_sha256 or
            document["native_corrected_images_runtime_sha256"] !=
                native_corrected_images_runtime.receipt["receipt_sha256"] or
            _SHA.fullmatch(str(document[
                "native_corrected_images_evidence_sha256"])) is None or
            document["counts"] !=
                CoupledAMRCorrectedLeafCountsCPUV98().__dict__ or
            document["corrected_image_rebuilt_not_serialized_as_v97"] is not
                True):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected checkpoint identity differs")
    if (type(provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            provider._open is not None or provider._prepared is not None or
            provider._closed is not None):
        raise CoupledAMRCorrectedCoarseCPUV98Error(
            "v98 corrected restart provider phase differs")
    before = provider.checkpoint()
    try:
        # This clone is only the exact-v97 provisional provenance owner.  The
        # public provider below is restored to the corrected accepted state.
        source_provider = _provider_clone(execution, provider, provider.state)
        source = load_coupled_wetdry_cpu_checkpoint_v97(
            execution, document["provisional_source_checkpoint"],
            source_provider)
        corrected_q3 = provider.restore_checkpoint(
            document["corrected_q3_checkpoint"])
        hles_execution = _v92_execution(
            execution.active_execution).base_execution.hles_execution
        corrected_hles = load_hles_checkpoint_v90(
            hles_execution, document["corrected_hles_checkpoint"])
        images, native_evidence = _evaluate_native_corrected_images(
            execution, source, interface, correction,
            native_corrected_images_runtime)
        if native_evidence.evidence_sha256 != \
                document["native_corrected_images_evidence_sha256"]:
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                "v98 corrected checkpoint native evidence differs")
        state = _wrap_state(
            execution, provisional=source, correction=correction,
            images=images, provisional_q3_state=source_provider.state,
            q3_state=corrected_q3, hles_state=corrected_hles,
            native_corrected_images_evidence=native_evidence,
            accepted_chain_sha256=document["accepted_chain_sha256"])
        validate_coupled_amr_corrected_coarse_state_v98(
            state, execution, interface, correction, provider=provider)
        if state.state_sha256 != document["state_sha256"]:
            raise CoupledAMRCorrectedCoarseCPUV98Error(
                "v98 corrected checkpoint state differs")
        return state
    except Exception as original_error:
        try:
            provider.restore_checkpoint(before)
        except Exception as cleanup_error:
            raise RuntimeError(
                "v98 corrected restart failed and provider restore also "
                "failed") from cleanup_error
        raise original_error


def method_identity_coupled_amr_corrected_coarse_cpu_v98() -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "backend": BACKEND,
        "state_semantic": "v98_corrected_leaf_not_exact_v97_state",
        "interface_ledger": "typed_internal_not_physical_boundary",
        "local_scalar_q3_exchange_replay_count": 0,
        "dss_recompute_count": 0,
        "derived_after_reflux": [
            "bed_elevation", "free_surface", "temperature", "salinity",
            "water_density", "mixture_density", "molecular_viscosity",
            "hles"],
        "q3_provider_commit": "two_phase_close_then_publish",
        "production_admission": False,
    }


__all__ = [
    "BACKEND", "CHECKPOINT_SCOPE_ID",
    "CoupledAMRCorrectedCoarseCPUV98Error",
    "CoupledAMRCorrectedLeafClosedCPUV98",
    "CoupledAMRCorrectedLeafCountsCPUV98",
    "CoupledAMRCorrectedLeafStateCPUV98",
    "CoupledAMRCorrectedLeafTrialCPUV98",
    "NativeCorrectedImagesEvidenceCPUV98", "NATIVE_ORACLE_ATOL",
    "NATIVE_ORACLE_RTOL", "SCHEMA", "SEMANTIC_ID",
    "SCOPE_ID", "abort_closed_coupled_amr_corrected_coarse_cpu_v98",
    "begin_coupled_amr_corrected_coarse_cpu_v98",
    "checkpoint_coupled_amr_corrected_coarse_cpu_v98",
    "close_coupled_amr_corrected_coarse_cpu_v98",
    "commit_coupled_amr_corrected_coarse_cpu_v98",
    "load_coupled_amr_corrected_coarse_checkpoint_v98",
    "method_identity_coupled_amr_corrected_coarse_cpu_v98",
    "publish_coupled_amr_corrected_coarse_cpu_v98",
    "rollback_coupled_amr_corrected_coarse_cpu_v98",
    "validate_coupled_amr_corrected_coarse_state_v98",
    "validate_coupled_amr_corrected_coarse_trial_v98",
]
