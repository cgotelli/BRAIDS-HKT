"""Private exact-v97 numerical parent rebuilt from a corrected v98 leaf.

The legacy v97 state schema owns physical-boundary ledgers only.  A refluxed
v98 leaf additionally owns a real internal coarse/fine interface ledger and
therefore cannot be relabelled as a v97 accepted state.  This module provides
the narrow compatibility bridge needed by the *next* private v98 macro step:

* every numerical image (``q``, suspended/bed mass, T/S, EOS, DSS and HLES)
  comes from the corrected v98 leaf;
* boundary return memories are carried unchanged;
* legacy local ledger windows are rebased to the corrected images, without
  applying either a physical-boundary or an internal-interface transfer;
* no clock, constitutive operator, provider, or public state is advanced.

The returned v97 object is consequently a numerical input only.  Its reset
legacy ledgers are not scientific history.  The enclosing v98 hierarchy state
and its typed internal-interface ledgers remain authoritative across restart.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from .active_suspended_control_v94 import _wrap_state as _wrap_v94_state
from .coupled_amr_corrected_coarse_cpu_v98 import (
    CoupledAMRCorrectedLeafStateCPUV98,
    validate_coupled_amr_corrected_coarse_state_v98,
)
from .coupled_amr_register_cpu_v98 import (
    CoupledAMRCoarseCorrectionCPUV98,
    CoupledAMRInterfaceMapCPUV98,
)
from .coupled_wetdry_owner_cpu_v97 import (
    CoupledWetDryCPUExecutionV97,
    CoupledWetDryCPUStateV97,
    _accepted_density,
    _v92_execution,
    _v92_state,
    _wrap_state as _wrap_v97_state,
    validate_coupled_wetdry_cpu_state_v97,
)
from .multifraction_suspended_lifecycle_v78 import (
    resolve_multifraction_suspended_state_v78,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)
from .shared_turbulence_consumers_v91 import (
    _wrap_state as _wrap_v91_state,
)
from .suspended_exner_owner_v93 import (
    _wrap_state as _wrap_v93_state,
)
from .transported_fluid_settling_v92 import (
    _wrap_state as _wrap_v92_state,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-coupled-amr-private-numerical-parent-cpu-v98-v1"
SEMANTIC_ID = SCOPE_ID + ":corrected-images-local-ledger-window-rebase"
NATIVE_TO_V92_CANONICAL_RTOL = 3.0e-13
NATIVE_TO_V92_CANONICAL_ATOL = 4.0e-15
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class AMRNumericalParentCPUV98Error(ValueError):
    """The corrected leaf, private state, or rebase receipt differs."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise AMRNumericalParentCPUV98Error(
            "v98 numerical-parent value is not finite canonical JSON") \
            from error


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(*values: object) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise AMRNumericalParentCPUV98Error(
            f"v98 numerical-parent {name} is not a SHA-256 digest")
    return value


@dataclass(frozen=True)
class AMRRebasedNumericalParentReceiptCPUV98:
    """Authentication for one zero-time, private compatibility rebase."""

    execution_fingerprint: str
    corrected_leaf_state_sha256: str
    correction_receipt_sha256: str
    internal_interface_ledger_sha256: str
    rebased_parent_state_sha256: str
    rebased_active_state_sha256: str
    rebased_v93_state_sha256: str
    rebased_v92_state_sha256: str
    rebased_v91_state_sha256: str
    rebased_v78_state_sha256: str
    rebased_v76_state_sha256: str
    corrected_q3_state_sha256: str
    corrected_hles_state_sha256: str
    corrected_numerical_images_sha256: str
    native_corrected_images_evidence_sha256: str
    canonical_v92_derived_images_sha256: str
    preserved_suspended_boundary_memory_sha256: str
    preserved_fluid_boundary_memory_sha256: str
    local_legacy_ledger_window_rebase_count: int
    native_to_v92_canonicalization_count: int
    boundary_memory_preservation_count: int
    physical_boundary_ledger_application_count: int
    internal_interface_ledger_application_count: int
    accepted_clock_increment_count: int
    settling_recompute_count: int
    exchange_recompute_count: int
    q3_recompute_count: int
    hles_recompute_count: int
    provider_mutation_count: int
    public_state_publication_count: int
    outer_v98_global_ledgers_authoritative: bool
    numerical_compatibility_parent_only: bool
    semantic_id: str
    receipt_sha256: str


def _receipt_payload(
    value: AMRRebasedNumericalParentReceiptCPUV98,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID + ":receipt",
        "semantic_id": value.semantic_id,
        "execution": value.execution_fingerprint,
        "corrected_leaf": value.corrected_leaf_state_sha256,
        "correction": value.correction_receipt_sha256,
        "internal_interface_ledger":
            value.internal_interface_ledger_sha256,
        "rebased_parent": value.rebased_parent_state_sha256,
        "nested_state_sha256": {
            "v94": value.rebased_active_state_sha256,
            "v93": value.rebased_v93_state_sha256,
            "v92": value.rebased_v92_state_sha256,
            "v91": value.rebased_v91_state_sha256,
            "v78": value.rebased_v78_state_sha256,
            "v76": value.rebased_v76_state_sha256,
        },
        "corrected_q3": value.corrected_q3_state_sha256,
        "corrected_hles": value.corrected_hles_state_sha256,
        "corrected_numerical_images":
            value.corrected_numerical_images_sha256,
        "native_corrected_images_evidence":
            value.native_corrected_images_evidence_sha256,
        "canonical_v92_derived_images":
            value.canonical_v92_derived_images_sha256,
        "preserved_boundary_memory_sha256": {
            "suspended":
                value.preserved_suspended_boundary_memory_sha256,
            "fluid": value.preserved_fluid_boundary_memory_sha256,
        },
        "counts": {
            "local_legacy_ledger_window_rebase":
                value.local_legacy_ledger_window_rebase_count,
            "native_to_v92_canonicalization":
                value.native_to_v92_canonicalization_count,
            "boundary_memory_preservation":
                value.boundary_memory_preservation_count,
            "physical_boundary_ledger_application":
                value.physical_boundary_ledger_application_count,
            "internal_interface_ledger_application":
                value.internal_interface_ledger_application_count,
            "accepted_clock_increment":
                value.accepted_clock_increment_count,
            "settling_recompute": value.settling_recompute_count,
            "exchange_recompute": value.exchange_recompute_count,
            "q3_recompute": value.q3_recompute_count,
            "hles_recompute": value.hles_recompute_count,
            "provider_mutation": value.provider_mutation_count,
            "public_state_publication":
                value.public_state_publication_count,
        },
        "outer_v98_global_ledgers_authoritative":
            value.outer_v98_global_ledgers_authoritative,
        "numerical_compatibility_parent_only":
            value.numerical_compatibility_parent_only,
        "native_to_v92_canonical_tolerance": {
            "rtol": NATIVE_TO_V92_CANONICAL_RTOL,
            "atol": NATIVE_TO_V92_CANONICAL_ATOL,
        },
    }


def _provider_closed_on_corrected_state(
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    corrected: CoupledAMRCorrectedLeafStateCPUV98,
) -> bool:
    return (
        type(provider) is CPUQ3V83ActiveWetDryExnerAdapterV97 and
        provider.state is corrected.q3_state and
        provider._open is None and provider._prepared is None and
        provider._closed is None)


def _build_rebased_parent(
    execution: CoupledWetDryCPUExecutionV97,
    corrected: CoupledAMRCorrectedLeafStateCPUV98,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
) -> CoupledWetDryCPUStateV97:
    """Build the private exact-v97 image without touching ``provider``."""
    if not _provider_closed_on_corrected_state(provider, corrected):
        raise AMRNumericalParentCPUV98Error(
            "v98 numerical-parent provider is not on the corrected Q3 state")
    v92_execution = _v92_execution(execution.active_execution)
    v91_execution = v92_execution.base_execution
    v78_execution = v91_execution.base_execution
    source_v92 = _v92_state(corrected.provisional_source_state.active_state)
    source_v76 = source_v92.suspended_state.lifecycle
    area = np.asarray(v78_execution.base_execution.cell_area_m2)
    water_reference = np.sum(
        corrected.suspended_mass_kg_m2*area[None, :], axis=1)
    bed_reference = np.sum(
        corrected.bed_mass_kg_m2*area[None, :], axis=1)

    # V76/V78 is a local numerical window.  Its reference is the corrected
    # image and every transfer/boundary ledger starts at zero; the outer v98
    # internal-interface ledger remains the scientific history.
    v78 = resolve_multifraction_suspended_state_v78(
        v78_execution,
        suspended_mass_kg_m2=corrected.suspended_mass_kg_m2,
        bed_mass_kg_m2=corrected.bed_mass_kg_m2,
        boundary_last_outflow_concentration_kg_m3=
            source_v76.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=source_v76.boundary_return_clock_s,
        advective_boundary_ledger_kg=np.zeros(
            (execution.fraction_count,
             v78_execution.base_execution.boundary_count)),
        diffusive_boundary_ledger_kg=np.zeros(
            (execution.fraction_count,
             v78_execution.base_execution.boundary_count)),
        water_exchange_ledger_kg=np.zeros(execution.fraction_count),
        bed_exchange_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_water_ledger_kg=np.zeros(execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        reference_water_mass_kg=water_reference,
        reference_bed_mass_kg=bed_reference,
        accepted_updates=corrected.accepted_updates,
        time_s=corrected.time_s)
    v91_chain = _hash({
        "scope_id": SCOPE_ID, "level": "v91",
        "corrected_leaf": corrected.state_sha256,
        "v78": v78.state_sha256,
        "q": _digest(corrected.q),
        "zero_time_rebase": True,
    })
    v91 = _wrap_v91_state(
        v91_execution, q=corrected.q, suspended_state=v78,
        receipt_chain=v91_chain)

    scalar_reference = np.array((
        np.sum(corrected.h_temperature_c_m*area, dtype=np.float64),
        np.sum(corrected.h_salinity_ppt_m*area, dtype=np.float64)))
    v92_chain = _hash({
        "scope_id": SCOPE_ID, "level": "v92",
        "corrected_leaf": corrected.state_sha256,
        "v91": v91.state_sha256,
        "scalar_reference": scalar_reference.tolist(),
        "zero_time_rebase": True,
    })
    v92 = _wrap_v92_state(
        v92_execution, v91,
        h_temperature_c_m=corrected.h_temperature_c_m,
        h_salinity_ppt_m=corrected.h_salinity_ppt_m,
        boundary_last_outflow_fluid=
            source_v92.boundary_last_outflow_fluid,
        boundary_return_clock_s=source_v92.boundary_return_clock_s,
        reference_scalar_integral=scalar_reference,
        advective_boundary_ledger=np.zeros(
            (2, v92_execution.boundary_count)),
        diffusive_boundary_ledger=np.zeros(
            (2, v92_execution.boundary_count)),
        wetdry_scalar_ledger=np.zeros(2),
        accepted_receipt_chain_sha256=v92_chain)

    v93_execution = execution.active_execution.base_execution
    q3_boundary_count = len(v93_execution.q3_boundary_names)
    v93_chain = _hash({
        "scope_id": SCOPE_ID, "level": "v93",
        "corrected_leaf": corrected.state_sha256,
        "v92": v92.state_sha256,
        "q3": corrected.q3_state.state_sha256,
        "zero_time_rebase": True,
    })
    v93 = _wrap_v93_state(
        v93_execution, v92,
        q3_provider_kind=provider.provider_kind,
        q3_provider_state_sha256=corrected.q3_state.state_sha256,
        bed_mass_kg_m2=corrected.bed_mass_kg_m2,
        reference_bed_integral_kg=bed_reference,
        suspended_exchange_ledger_kg=np.zeros(execution.fraction_count),
        suspended_boundary_ledger_kg=np.zeros(
            (execution.fraction_count, v92_execution.boundary_count)),
        excluded_water_exchange_kg=np.zeros(execution.fraction_count),
        q3_named_boundary_positive_in_kg=np.zeros(
            (execution.fraction_count, q3_boundary_count)),
        q3_named_boundary_absolute_kg=np.zeros(
            (execution.fraction_count, q3_boundary_count)),
        q3_depletion_roundoff_positive_in_kg=np.zeros(
            execution.fraction_count),
        wetdry_bed_ledger_kg=np.zeros(execution.fraction_count),
        accepted_exner_chain_sha256=v93_chain)
    v94_chain = _hash({
        "scope_id": SCOPE_ID, "level": "v94",
        "corrected_leaf": corrected.state_sha256,
        "v93": v93.state_sha256,
        "carried_dss": _digest(corrected.suspended_diameter_m),
        "dss_recompute_count": 0,
    })
    active = _wrap_v94_state(
        execution.active_execution, v93,
        suspended_diameter_m=corrected.suspended_diameter_m,
        accepted_iopsus_chain_sha256=v94_chain)
    outer_chain = _hash({
        "scope_id": SCOPE_ID, "level": "v97_private_parent",
        "corrected_leaf": corrected.state_sha256,
        "active": active.state_sha256,
        "hles": corrected.hles_state.state_sha256,
        "correction": corrected.correction_receipt_sha256,
        "internal_interface_ledger":
            corrected.internal_interface_ledger_sha256,
        "outer_v98_global_ledgers_authoritative": True,
    })
    result = _wrap_v97_state(
        execution, active_state=active, hles_state=corrected.hles_state,
        accepted_chain_sha256=outer_chain, q3_provider=provider)
    validate_coupled_wetdry_cpu_state_v97(result, execution, provider)
    return result


def _make_receipt(
    execution: CoupledWetDryCPUExecutionV97,
    corrected: CoupledAMRCorrectedLeafStateCPUV98,
    rebased: CoupledWetDryCPUStateV97,
) -> AMRRebasedNumericalParentReceiptCPUV98:
    v92 = _v92_state(rebased.active_state)
    v76 = v92.suspended_state.lifecycle
    images = _digest(
        corrected.q, corrected.suspended_mass_kg_m2,
        corrected.bed_mass_kg_m2, corrected.h_temperature_c_m,
        corrected.h_salinity_ppt_m, corrected.temperature_cell_c,
        corrected.salinity_cell_ppt,
        corrected.water_density_cell_kg_m3,
        corrected.mixture_density_cell_kg_m3,
        corrected.molecular_viscosity_cell_m2_s,
        corrected.bed_elevation_m, corrected.free_surface_elevation_m,
        corrected.suspended_diameter_m,
        corrected.hles_state.accepted_filtered_face_normal_velocity_m_s,
        corrected.hles_state.accepted_dynamic_vicuv_m2_s,
        corrected.hles_state.accepted_dynamic_dicuv_m2_s)
    suspended_memory = _digest(
        v76.boundary_last_outflow_concentration_kg_m3,
        v76.boundary_return_clock_s)
    fluid_memory = _digest(
        v92.boundary_last_outflow_fluid, v92.boundary_return_clock_s)
    canonical_water, canonical_mixture = _accepted_density(
        execution, rebased)
    canonical_derived = _digest(
        v92.temperature_cell_c, v92.salinity_cell_ppt,
        canonical_water, v92.molecular_viscosity_cell_m2_s,
        canonical_mixture, rebased.bed_elevation_m,
        rebased.free_surface_elevation_m)
    placeholder = AMRRebasedNumericalParentReceiptCPUV98(
        execution_fingerprint=execution.fingerprint,
        corrected_leaf_state_sha256=corrected.state_sha256,
        correction_receipt_sha256=corrected.correction_receipt_sha256,
        internal_interface_ledger_sha256=
            corrected.internal_interface_ledger_sha256,
        rebased_parent_state_sha256=rebased.state_sha256,
        rebased_active_state_sha256=rebased.active_state.state_sha256,
        rebased_v93_state_sha256=rebased.active_state.base_state.state_sha256,
        rebased_v92_state_sha256=v92.state_sha256,
        rebased_v91_state_sha256=v92.base_state.state_sha256,
        rebased_v78_state_sha256=v92.suspended_state.state_sha256,
        rebased_v76_state_sha256=v76.state_sha256,
        corrected_q3_state_sha256=corrected.q3_state.state_sha256,
        corrected_hles_state_sha256=corrected.hles_state.state_sha256,
        corrected_numerical_images_sha256=images,
        native_corrected_images_evidence_sha256=
            corrected.native_corrected_images_evidence.evidence_sha256,
        canonical_v92_derived_images_sha256=canonical_derived,
        preserved_suspended_boundary_memory_sha256=suspended_memory,
        preserved_fluid_boundary_memory_sha256=fluid_memory,
        local_legacy_ledger_window_rebase_count=3,
        native_to_v92_canonicalization_count=1,
        boundary_memory_preservation_count=2,
        physical_boundary_ledger_application_count=0,
        internal_interface_ledger_application_count=0,
        accepted_clock_increment_count=0,
        settling_recompute_count=0, exchange_recompute_count=0,
        q3_recompute_count=0, hles_recompute_count=0,
        provider_mutation_count=0, public_state_publication_count=0,
        outer_v98_global_ledgers_authoritative=True,
        numerical_compatibility_parent_only=True,
        semantic_id=SEMANTIC_ID, receipt_sha256="")
    return AMRRebasedNumericalParentReceiptCPUV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _hash(_receipt_payload(placeholder))})


def rebase_corrected_leaf_to_private_v97_parent_v98(
    execution: CoupledWetDryCPUExecutionV97,
    corrected: CoupledAMRCorrectedLeafStateCPUV98,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> tuple[CoupledWetDryCPUStateV97,
           AMRRebasedNumericalParentReceiptCPUV98]:
    """Return a zero-time private numerical parent for the next macro step."""
    validate_coupled_amr_corrected_coarse_state_v98(
        corrected, execution, interface, correction, provider=provider)
    before = provider.checkpoint()
    rebased = _build_rebased_parent(execution, corrected, provider)
    after = provider.checkpoint()
    if after != before:
        raise AMRNumericalParentCPUV98Error(
            "v98 numerical-parent rebase mutated the Q3 provider")
    receipt = _make_receipt(execution, corrected, rebased)
    validate_amr_rebased_numerical_parent_cpu_v98(
        receipt, execution, corrected, rebased, provider, interface,
        correction)
    return rebased, receipt


def validate_amr_rebased_numerical_parent_cpu_v98(
    receipt: AMRRebasedNumericalParentReceiptCPUV98,
    execution: CoupledWetDryCPUExecutionV97,
    corrected: CoupledAMRCorrectedLeafStateCPUV98,
    rebased: CoupledWetDryCPUStateV97,
    provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    interface: CoupledAMRInterfaceMapCPUV98,
    correction: CoupledAMRCoarseCorrectionCPUV98,
) -> None:
    """Rebuild and validate the private parent and every zero-owner claim."""
    validate_coupled_amr_corrected_coarse_state_v98(
        corrected, execution, interface, correction, provider=provider)
    if (type(receipt) is not AMRRebasedNumericalParentReceiptCPUV98 or
            receipt.receipt_sha256 != _hash(_receipt_payload(receipt)) or
            receipt.semantic_id != SEMANTIC_ID or
            receipt.execution_fingerprint != execution.fingerprint or
            receipt.corrected_leaf_state_sha256 != corrected.state_sha256 or
            receipt.correction_receipt_sha256 !=
                corrected.correction_receipt_sha256 or
            receipt.internal_interface_ledger_sha256 !=
                corrected.internal_interface_ledger_sha256 or
            receipt.native_corrected_images_evidence_sha256 !=
                corrected.native_corrected_images_evidence.evidence_sha256 or
            receipt.local_legacy_ledger_window_rebase_count != 3 or
            receipt.native_to_v92_canonicalization_count != 1 or
            receipt.boundary_memory_preservation_count != 2 or
            any(value != 0 for value in (
                receipt.physical_boundary_ledger_application_count,
                receipt.internal_interface_ledger_application_count,
                receipt.accepted_clock_increment_count,
                receipt.settling_recompute_count,
                receipt.exchange_recompute_count,
                receipt.q3_recompute_count,
                receipt.hles_recompute_count,
                receipt.provider_mutation_count,
                receipt.public_state_publication_count)) or
            receipt.outer_v98_global_ledgers_authoritative is not True or
            receipt.numerical_compatibility_parent_only is not True):
        raise AMRNumericalParentCPUV98Error(
            "v98 numerical-parent receipt identity or owner counts differ")
    expected = _build_rebased_parent(execution, corrected, provider)
    expected_receipt = _make_receipt(execution, corrected, expected)
    if (type(rebased) is not CoupledWetDryCPUStateV97 or
            rebased.state_sha256 != expected.state_sha256 or
            receipt != expected_receipt or
            rebased.accepted_updates != corrected.accepted_updates or
            rebased.time_s != corrected.time_s or
            not np.array_equal(rebased.q, corrected.q) or
            rebased.active_state.base_state.q3_provider_state_sha256 !=
                corrected.q3_state.state_sha256 or
            rebased.hles_state is not corrected.hles_state):
        raise AMRNumericalParentCPUV98Error(
            "v98 numerical-parent state differs from corrected images")
    v92 = _v92_state(rebased.active_state)
    v76 = v92.suspended_state.lifecycle
    source_v92 = _v92_state(corrected.provisional_source_state.active_state)
    source_v76 = source_v92.suspended_state.lifecycle
    rebased_water, rebased_mixture = _accepted_density(execution, rebased)
    zero_arrays = (
        v76.advective_boundary_ledger_kg,
        v76.diffusive_boundary_ledger_kg,
        v76.water_exchange_ledger_kg,
        v76.bed_exchange_ledger_kg,
        v76.wetdry_water_ledger_kg,
        v76.wetdry_bed_ledger_kg,
        v92.advective_boundary_ledger,
        v92.diffusive_boundary_ledger,
        v92.wetdry_scalar_ledger,
        rebased.active_state.base_state.suspended_exchange_ledger_kg,
        rebased.active_state.base_state.suspended_boundary_ledger_kg,
        rebased.active_state.base_state.excluded_water_exchange_kg,
        rebased.active_state.base_state.q3_named_boundary_positive_in_kg,
        rebased.active_state.base_state.q3_named_boundary_absolute_kg,
        rebased.active_state.base_state.
            q3_depletion_roundoff_positive_in_kg,
        rebased.active_state.base_state.wetdry_bed_ledger_kg,
    )
    if (any(np.any(value != 0.0) for value in zero_arrays) or
            not np.array_equal(
                v92.suspended_state.suspended_mass_kg_m2,
                corrected.suspended_mass_kg_m2) or
            not np.array_equal(
                v92.suspended_state.bed_mass_kg_m2,
                corrected.bed_mass_kg_m2) or
            not np.array_equal(v92.h_temperature_c_m,
                               corrected.h_temperature_c_m) or
            not np.array_equal(v92.h_salinity_ppt_m,
                               corrected.h_salinity_ppt_m) or
            not np.allclose(
                v92.temperature_cell_c, corrected.temperature_cell_c,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                v92.salinity_cell_ppt, corrected.salinity_cell_ppt,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                v92.water_density_cell_kg_m3,
                corrected.water_density_cell_kg_m3,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                v92.molecular_viscosity_cell_m2_s,
                corrected.molecular_viscosity_cell_m2_s,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                rebased_water, corrected.water_density_cell_kg_m3,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                rebased_mixture, corrected.mixture_density_cell_kg_m3,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                rebased.bed_elevation_m, corrected.bed_elevation_m,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.allclose(
                rebased.free_surface_elevation_m,
                corrected.free_surface_elevation_m,
                rtol=NATIVE_TO_V92_CANONICAL_RTOL,
                atol=NATIVE_TO_V92_CANONICAL_ATOL) or
            not np.array_equal(rebased.active_state.suspended_diameter_m,
                               corrected.suspended_diameter_m) or
            not np.array_equal(
                v76.boundary_last_outflow_concentration_kg_m3,
                source_v76.boundary_last_outflow_concentration_kg_m3) or
            not np.array_equal(v76.boundary_return_clock_s,
                               source_v76.boundary_return_clock_s) or
            not np.array_equal(v92.boundary_last_outflow_fluid,
                               source_v92.boundary_last_outflow_fluid) or
            not np.array_equal(v92.boundary_return_clock_s,
                               source_v92.boundary_return_clock_s)):
        raise AMRNumericalParentCPUV98Error(
            "v98 numerical-parent arrays, reset ledgers, or memory differ")
    _sha(receipt.receipt_sha256, "receipt")


def method_identity_amr_numerical_parent_cpu_v98() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "semantic_id": SEMANTIC_ID,
        "role": "private_next_macro_numerical_parent_only",
        "source_images": "accepted_v98_corrected_leaf",
        "preserved": [
            "q", "suspended_mass", "bed_mass", "temperature",
            "salinity", "water_eos", "DSS", "HLES",
            "boundary_return_memory", "accepted_clock"],
        "legacy_local_ledger_windows": "reset_with_current_references",
        "native_to_v92_canonicalization_count": 1,
        "native_to_v92_canonicalization": {
            "source": "authenticated_native_corrected_images_evidence",
            "target": "frozen_v92_derived_properties",
            "rtol": NATIVE_TO_V92_CANONICAL_RTOL,
            "atol": NATIVE_TO_V92_CANONICAL_ATOL,
            "physical_operator_count": 0,
        },
        "outer_v98_global_ledgers": "authoritative_and_not_reapplied",
        "physical_boundary_ledger_application_count": 0,
        "internal_interface_ledger_application_count": 0,
        "constitutive_recompute_count": 0,
        "provider_mutation_count": 0,
        "public_state_publication_count": 0,
        "product_admission": False,
    }


__all__ = [
    "AMRNumericalParentCPUV98Error",
    "AMRRebasedNumericalParentReceiptCPUV98",
    "NATIVE_TO_V92_CANONICAL_ATOL", "NATIVE_TO_V92_CANONICAL_RTOL",
    "SCHEMA", "SCOPE_ID", "SEMANTIC_ID",
    "method_identity_amr_numerical_parent_cpu_v98",
    "rebase_corrected_leaf_to_private_v97_parent_v98",
    "validate_amr_rebased_numerical_parent_cpu_v98",
]
