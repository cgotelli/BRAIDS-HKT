"""Validated v92 projection onto the sole v93 bed inventory.

v92 legitimately computes water-column exchange and therefore proposes an
exchange-only nested bed.  The v93 Exner owner replaces that provisional bed
with its one combined suspended/Q3 result.  Every water-column field and
ledger remains bitwise unchanged; only the nested bed and its derived
reference bookkeeping are rebuilt for the next accepted update.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .multifraction_suspended_lifecycle_v78 import _wrap_state as _wrap_v78
from .shared_turbulence_consumers_v91 import _wrap_state as _wrap_v91
from .suspended_lifecycle_live_v76 import resolve_suspended_lifecycle_state_v76
from .transported_fluid_settling_v92 import (
    TransportedFluidExecutionV92,
    TransportedFluidStateV92,
    _validate_state as _validate_v92,
    _wrap_state as _wrap_v92,
)


SCOPE_ID = "braids-q4-v92-authoritative-bed-projection-v93-v1"


class V92BedProjectionV93Error(ValueError):
    """The v92 state cannot be projected without altering water state."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(*values: np.ndarray) -> str:
    result = hashlib.sha256()
    for raw in values:
        value = np.ascontiguousarray(raw)
        result.update(value.dtype.str.encode("ascii"))
        result.update(_canonical(list(value.shape)))
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


@dataclass(frozen=True)
class V92BedProjectionReceiptV93:
    source_state_sha256: str
    projected_state_sha256: str
    authoritative_bed_sha256: str
    water_state_before_sha256: str
    water_state_after_sha256: str
    source_reference_bed_sha256: str
    projected_reference_bed_sha256: str
    accepted_chain_sha256: str
    bed_projection_count: int
    receipt_sha256: str


def _water_digest(state: TransportedFluidStateV92) -> str:
    lifecycle = state.suspended_state.lifecycle
    return _digest(
        lifecycle.suspended_mass_kg_m2,
        lifecycle.boundary_last_outflow_concentration_kg_m3,
        lifecycle.boundary_return_clock_s,
        lifecycle.advective_boundary_ledger_kg,
        lifecycle.diffusive_boundary_ledger_kg,
        lifecycle.water_exchange_ledger_kg,
        lifecycle.wetdry_water_ledger_kg,
        lifecycle.reference_water_mass_kg,
        state.base_state.q,
        state.h_temperature_c_m,
        state.h_salinity_ppt_m,
        state.temperature_cell_c,
        state.salinity_cell_ppt,
        state.water_density_cell_kg_m3,
        state.molecular_viscosity_cell_m2_s,
        state.boundary_last_outflow_fluid,
        state.boundary_return_clock_s,
        state.reference_scalar_integral,
        state.advective_boundary_ledger,
        state.diffusive_boundary_ledger,
        state.wetdry_scalar_ledger,
    )


def _receipt_token(value: V92BedProjectionReceiptV93) -> str:
    return hashlib.sha256(_canonical({
        "scope_id": SCOPE_ID,
        "source_state_sha256": value.source_state_sha256,
        "projected_state_sha256": value.projected_state_sha256,
        "authoritative_bed_sha256": value.authoritative_bed_sha256,
        "water_state_before_sha256": value.water_state_before_sha256,
        "water_state_after_sha256": value.water_state_after_sha256,
        "source_reference_bed_sha256": value.source_reference_bed_sha256,
        "projected_reference_bed_sha256":
            value.projected_reference_bed_sha256,
        "accepted_chain_sha256": value.accepted_chain_sha256,
        "bed_projection_count": value.bed_projection_count,
    })).hexdigest()


def project_v92_authoritative_bed_v93(
    execution: TransportedFluidExecutionV92,
    source: TransportedFluidStateV92, *, authoritative_bed_mass_kg_m2: object,
) -> tuple[TransportedFluidStateV92, V92BedProjectionReceiptV93]:
    """Rebuild the nested accepted state around one authoritative bed.

    The v76 bed exchange and wet/dry ledgers remain the water-exchange
    counterpart.  The reference-bed term is re-anchored so that v76/v78 can
    validate the projected next-step state without becoming a second history
    owner.  v93 retains the true suspended/Q3 bed-ledger decomposition.
    """
    _validate_v92(source, execution)
    v91e = execution.base_execution
    v78e = v91e.base_execution
    v76e = v78e.base_execution
    old91 = source.base_state
    old78 = old91.suspended_state
    old76 = old78.lifecycle
    raw = np.asarray(authoritative_bed_mass_kg_m2)
    shape = old76.bed_mass_kg_m2.shape
    if raw.shape != shape or raw.dtype.kind not in "fiu":
        raise V92BedProjectionV93Error(
            f"v92 projected bed must be numeric with shape {shape}")
    bed = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(bed).all() or np.any(bed < 0.0):
        raise V92BedProjectionV93Error(
            "v92 projected bed must be finite and nonnegative")
    reference_bed = (
        np.sum(bed*v76e.cell_area_m2[None, :], axis=1)-
        old76.bed_exchange_ledger_kg-old76.wetdry_bed_ledger_kg)
    tolerance = 131072.0*np.finfo(np.float64).eps*np.maximum(
        1.0, np.abs(reference_bed))
    if np.any(reference_bed < -tolerance):
        raise V92BedProjectionV93Error(
            "v92 projected reference bed would be negative")
    reference_bed = np.maximum(reference_bed, 0.0)
    before = _water_digest(source)
    lifecycle = resolve_suspended_lifecycle_state_v76(
        v76e,
        suspended_mass_kg_m2=old76.suspended_mass_kg_m2,
        bed_mass_kg_m2=bed,
        boundary_last_outflow_concentration_kg_m3=
            old76.boundary_last_outflow_concentration_kg_m3,
        boundary_return_clock_s=old76.boundary_return_clock_s,
        advective_boundary_ledger_kg=old76.advective_boundary_ledger_kg,
        diffusive_boundary_ledger_kg=old76.diffusive_boundary_ledger_kg,
        water_exchange_ledger_kg=old76.water_exchange_ledger_kg,
        bed_exchange_ledger_kg=old76.bed_exchange_ledger_kg,
        wetdry_water_ledger_kg=old76.wetdry_water_ledger_kg,
        wetdry_bed_ledger_kg=old76.wetdry_bed_ledger_kg,
        reference_water_mass_kg=old76.reference_water_mass_kg,
        reference_bed_mass_kg=reference_bed,
        accepted_updates=old76.accepted_updates,
        time_s=old76.time_s,
    )
    scalar = _wrap_v78(v78e, lifecycle)
    shared = _wrap_v91(
        v91e, q=old91.q, suspended_state=scalar,
        receipt_chain=old91.accepted_receipt_chain_sha256)
    projected = _wrap_v92(
        execution, shared,
        h_temperature_c_m=source.h_temperature_c_m,
        h_salinity_ppt_m=source.h_salinity_ppt_m,
        boundary_last_outflow_fluid=source.boundary_last_outflow_fluid,
        boundary_return_clock_s=source.boundary_return_clock_s,
        reference_scalar_integral=source.reference_scalar_integral,
        advective_boundary_ledger=source.advective_boundary_ledger,
        diffusive_boundary_ledger=source.diffusive_boundary_ledger,
        wetdry_scalar_ledger=source.wetdry_scalar_ledger,
        accepted_receipt_chain_sha256=source.accepted_receipt_chain_sha256,
        temperature_cell_c=source.temperature_cell_c,
        salinity_cell_ppt=source.salinity_cell_ppt,
        water_density_cell_kg_m3=source.water_density_cell_kg_m3,
        molecular_viscosity_cell_m2_s=source.molecular_viscosity_cell_m2_s,
    )
    after = _water_digest(projected)
    if before != after:
        raise V92BedProjectionV93Error(
            "v92 authoritative-bed projection changed water state")
    if (projected.accepted_updates != source.accepted_updates or
            projected.time_s != source.time_s or
            projected.base_state.accepted_receipt_chain_sha256 !=
                source.base_state.accepted_receipt_chain_sha256 or
            projected.accepted_receipt_chain_sha256 !=
                source.accepted_receipt_chain_sha256):
        raise V92BedProjectionV93Error(
            "v92 authoritative-bed projection changed accepted clocks/chains")
    placeholder = V92BedProjectionReceiptV93(
        source.state_sha256, projected.state_sha256, _digest(bed), before,
        after, _digest(old76.reference_bed_mass_kg),
        _digest(lifecycle.reference_bed_mass_kg),
        hashlib.sha256(_canonical({
            "v91": source.base_state.accepted_receipt_chain_sha256,
            "v92": source.accepted_receipt_chain_sha256,
        })).hexdigest(), 1, "")
    receipt = V92BedProjectionReceiptV93(
        **{**placeholder.__dict__, "receipt_sha256": _receipt_token(placeholder)})
    return projected, receipt


def validate_v92_bed_projection_receipt_v93(
    receipt: V92BedProjectionReceiptV93, *,
    execution: TransportedFluidExecutionV92,
    source: TransportedFluidStateV92,
    projected: TransportedFluidStateV92,
    authoritative_bed_mass_kg_m2: object,
) -> None:
    if not isinstance(receipt, V92BedProjectionReceiptV93):
        raise V92BedProjectionV93Error(
            "v92 authoritative-bed projection receipt differs")
    rebuilt, expected = project_v92_authoritative_bed_v93(
        execution, source,
        authoritative_bed_mass_kg_m2=authoritative_bed_mass_kg_m2)
    if (projected.state_sha256 != rebuilt.state_sha256 or
            receipt != expected or
            receipt.bed_projection_count != 1 or
            receipt.water_state_before_sha256 !=
                receipt.water_state_after_sha256 or
            receipt.receipt_sha256 != _receipt_token(receipt)):
        raise V92BedProjectionV93Error(
            "v92 authoritative-bed projection receipt differs")


__all__ = [
    "SCOPE_ID", "V92BedProjectionReceiptV93", "V92BedProjectionV93Error",
    "project_v92_authoritative_bed_v93",
    "validate_v92_bed_projection_receipt_v93",
]
