"""Authenticated private Q3-provider clone for one v98 fine substep.

The public/coarse v97 provider remains open while an AMR owner stages fine
substep two.  This bridge clones only the already authenticated prepared
rebound state; it neither closes nor publishes the source provider and owns no
clock, scalar, face, or Exner work.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from .q3_mixed_route_exner_adapter_v93 import (
    CPUQ3V84MixedPreparedCommitV93,
)
from .q3_wetdry_adapter_cpu_v97 import (
    CPUQ3V83ActiveWetDryExnerAdapterV97,
)


SCHEMA = 1
SCOPE_ID = "braids-q4-private-q3-substep-bridge-cpu-v98-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class PrivateQ3SubstepBridgeV98Error(ValueError):
    """The prepared rebound or exact provider identity differs."""


def _token(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class PrivateQ3SubstepCloneReceiptV98:
    source_execution_fingerprint: str
    source_binding_sha256: str
    source_parent_state_sha256: str
    source_open_handle_sha256: str
    source_preparation_sha256: str
    rebound_state_sha256: str
    factcr_runtime_sha256: str
    q3_face_runtime_sha256: str
    native_threads: int
    source_provider_mutation_count: int
    accepted_state_publication_count: int
    receipt_sha256: str


def _payload(value: PrivateQ3SubstepCloneReceiptV98) -> dict:
    return {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "source_execution_fingerprint":
            value.source_execution_fingerprint,
        "source_binding_sha256": value.source_binding_sha256,
        "source_parent_state_sha256": value.source_parent_state_sha256,
        "source_open_handle_sha256": value.source_open_handle_sha256,
        "source_preparation_sha256": value.source_preparation_sha256,
        "rebound_state_sha256": value.rebound_state_sha256,
        "factcr_runtime_sha256": value.factcr_runtime_sha256,
        "q3_face_runtime_sha256": value.q3_face_runtime_sha256,
        "native_threads": value.native_threads,
        "source_provider_mutation_count":
            value.source_provider_mutation_count,
        "accepted_state_publication_count":
            value.accepted_state_publication_count,
        "private_staging_only": True,
    }


def clone_from_prepared_for_private_substep_v98(
    source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    prepared: CPUQ3V84MixedPreparedCommitV93, *,
    expected_rebound_state_sha256: str,
) -> tuple[CPUQ3V83ActiveWetDryExnerAdapterV97,
           PrivateQ3SubstepCloneReceiptV98]:
    """Clone one exact prepared v97 Q3 state without publishing fine step 1."""
    if (type(source_provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            type(prepared) is not CPUQ3V84MixedPreparedCommitV93 or
            _SHA.fullmatch(expected_rebound_state_sha256) is None):
        raise PrivateQ3SubstepBridgeV98Error(
            "v98 private Q3 source/prepared/digest type differs")
    if (source_provider._open is not prepared.open_trial or
            source_provider._prepared is not prepared or
            source_provider._closed is not None or
            prepared.rebound_state.state_sha256 !=
                expected_rebound_state_sha256):
        raise PrivateQ3SubstepBridgeV98Error(
            "v98 private Q3 prepared rebound is stale or cross-provider")
    source_provider.validate_prepared(
        prepared,
        authoritative_bed_mass_kg_m2=
            prepared.rebound_state.authoritative_bed_mass_kg_m2)
    clone = CPUQ3V83ActiveWetDryExnerAdapterV97(
        outer_execution=source_provider.control_execution,
        binding=source_provider.binding,
        state=prepared.rebound_state,
        native_factcr_runtime=source_provider.native_factcr_runtime,
        native_q3_face_runtime=source_provider.native_q3_face_runtime,
        native_threads=source_provider.native_factcr_threads)
    if (clone._state is not prepared.rebound_state or
            clone._open is not None or clone._prepared is not None or
            clone._closed is not None or clone._armed is not None):
        raise PrivateQ3SubstepBridgeV98Error(
            "v98 private Q3 clone did not start at the rebound commit point")
    placeholder = PrivateQ3SubstepCloneReceiptV98(
        source_provider.control_execution.fingerprint,
        source_provider.binding.binding_sha256,
        prepared.open_trial.parent_state.state_sha256,
        prepared.open_trial.handle_sha256,
        prepared.preparation_sha256,
        prepared.rebound_state.state_sha256,
        source_provider.native_factcr_runtime_sha256,
        source_provider.native_q3_face_runtime_sha256,
        source_provider.native_factcr_threads, 0, 0, "")
    receipt = PrivateQ3SubstepCloneReceiptV98(
        **{**placeholder.__dict__,
           "receipt_sha256": _token(_payload(placeholder))})
    validate_private_q3_substep_clone_v98(
        source_provider, prepared, clone, receipt)
    return clone, receipt


def validate_private_q3_substep_clone_v98(
    source_provider: CPUQ3V83ActiveWetDryExnerAdapterV97,
    prepared: CPUQ3V84MixedPreparedCommitV93,
    clone: CPUQ3V83ActiveWetDryExnerAdapterV97,
    receipt: PrivateQ3SubstepCloneReceiptV98,
) -> None:
    if (type(source_provider) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            type(prepared) is not CPUQ3V84MixedPreparedCommitV93 or
            type(clone) is not CPUQ3V83ActiveWetDryExnerAdapterV97 or
            type(receipt) is not PrivateQ3SubstepCloneReceiptV98 or
            receipt.receipt_sha256 != _token(_payload(receipt)) or
            source_provider._open is not prepared.open_trial or
            source_provider._prepared is not prepared or
            source_provider._closed is not None or
            clone._state is not prepared.rebound_state or
            clone._open is not None or clone._prepared is not None or
            clone._closed is not None or clone._armed is not None or
            receipt.source_execution_fingerprint !=
                source_provider.control_execution.fingerprint or
            receipt.source_binding_sha256 !=
                source_provider.binding.binding_sha256 or
            receipt.source_parent_state_sha256 !=
                prepared.open_trial.parent_state.state_sha256 or
            receipt.source_open_handle_sha256 !=
                prepared.open_trial.handle_sha256 or
            receipt.source_preparation_sha256 !=
                prepared.preparation_sha256 or
            receipt.rebound_state_sha256 !=
                prepared.rebound_state.state_sha256 or
            receipt.factcr_runtime_sha256 !=
                source_provider.native_factcr_runtime_sha256 or
            receipt.q3_face_runtime_sha256 !=
                source_provider.native_q3_face_runtime_sha256 or
            receipt.native_threads != source_provider.native_factcr_threads or
            receipt.source_provider_mutation_count != 0 or
            receipt.accepted_state_publication_count != 0):
        raise PrivateQ3SubstepBridgeV98Error(
            "v98 private Q3 clone receipt is stale, tampered or cross-owner")


__all__ = [
    "PrivateQ3SubstepBridgeV98Error",
    "PrivateQ3SubstepCloneReceiptV98",
    "clone_from_prepared_for_private_substep_v98",
    "validate_private_q3_substep_clone_v98",
]
