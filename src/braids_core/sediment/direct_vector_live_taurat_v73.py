"""Authenticated live-``taurat`` extension for Van Rijn 1993.

Classic Delft3D4 FLOW obtains ``taurat = taubcw/taucr1`` inside
``bedbc1993`` and later projects that *producer output* to the U/V face used
by ``adjust_bedload``.  The value is therefore neither a generic hydraulic
shear diagnostic nor something that may be reconstructed downstream.

This additive v73 envelope binds the existing v67 direct-vector family to
the accepted ISlope=2 operator.  Positive ``TraFrm`` rows retain the v69
exact-zero binding; ``TraFrm=-1`` must carry the live value returned by the
same constitutive invocation.  ``TraFrm=-2`` and ``-4`` remain owned by Q7
and are deliberately outside this receipt.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from .bed_slope_execution import validate_bed_slope_execution_document
from .classic_flow_taurat_v69 import (
    DIRECT_VECTOR_METHOD_SELECTORS,
    TAURAT_POSITIVE_ZERO,
    TAURAT_TRAM_LIVE,
    direct_vector_taurat_semantics,
    verify_locked_sources_v69,
)
from .direct_vector_family_live_v67 import (
    METHOD_VAN_RIJN_1993_CURRENT,
    load_direct_vector_family_execution_v67,
    verify_direct_vector_family_sources_v67,
)


SCHEMA = 1
ENVELOPE_VERSION = 73
SCOPE_ID = "braids-q3h7-vr1993-live-taurat-v73-v1"
METHOD_ID = "bed_slope.delft_bagnold_ikeda"
STRESS_SEMANTICS = "classic_flow_selector_owned_taurat"
MAX_ACCEPTED_STEPS = 4

_COMMON_IMPLEMENTATION_PATHS = (
    "braids_core/sediment/classic_flow_taurat_v69.py",
    "braids_core/sediment/direct_vector_live_taurat_v73.py",
)
_BACKEND_IMPLEMENTATION_PATHS = {
    "cpu": (
        "solver/wave/braids_vr1993_route_v66_module.f90",
        "solver/wave/braids_direct_producer_adapter_v67_module.f90",
        "solver/wave/braids_direct_vector_live_taurat_v73_module.f90",
    ),
    "implicit_fwave_midpoint_gpu": (
        "braids_gpu/van_rijn_1993_route_v66.cu",
        "braids_gpu/van_rijn_1993_route_v66.py",
        "braids_gpu/direct_producer_adapter_v67.py",
        "braids_gpu/direct_vector_live_taurat_v73.py",
    ),
}


class DirectVectorLiveTauratV73Error(ValueError):
    """The v73 source, method, backend, or transaction identity is invalid."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 identity is not canonical JSON"
        ) from error


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _implementation_sources(root: Path, backend: str) -> list[dict[str, str]]:
    try:
        relatives = _COMMON_IMPLEMENTATION_PATHS + \
            _BACKEND_IMPLEMENTATION_PATHS[backend]
    except KeyError as error:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 supports only native CPU/OpenMP and the explicitly "
            "selected implicit_fwave_midpoint GPU backend"
        ) from error
    rows: list[dict[str, str]] = []
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise DirectVectorLiveTauratV73Error(
                f"Q3/v73 implementation source is missing: {relative}"
            )
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return rows


def direct_vector_live_taurat_execution_document_v73(
    *,
    parent_execution: Mapping[str, object],
    bed_slope_execution: Mapping[str, object],
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Bind one v67 family to selector-owned classic-FLOW ISlope=2 stress."""
    try:
        parent = load_direct_vector_family_execution_v67(parent_execution)
        slope = validate_bed_slope_execution_document(bed_slope_execution)
    except (TypeError, ValueError) as error:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 parent or bed-slope execution is invalid"
        ) from error
    if slope["method_id"] != METHOD_ID or slope["runtime_mode"] != 2:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 requires classic FLOW ISlope=2"
        )
    physical = parent["physical_profile"]
    if slope["ordered_fraction_ids"] != physical["fraction_ids"]:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 slope fraction order differs from the producer family"
        )
    upstream = {
        row["stage"]: row["fingerprint"]
        for row in slope["upstream_identities"]
    }
    if upstream.get("capacity_execution") != parent["fingerprint"]:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 slope capacity parent differs from the v67 execution"
        )

    rows = []
    live_count = 0
    for row in physical["fraction_methods"]:
        method = row["method_id"]
        selector = row["reference_selector"]
        if DIRECT_VECTOR_METHOD_SELECTORS.get(method) != selector:
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 method/TraFrm identity differs from the locked audit"
            )
        semantics = direct_vector_taurat_semantics(method)
        if semantics == TAURAT_TRAM_LIVE:
            if method != METHOD_VAN_RIJN_1993_CURRENT or selector != -1:
                raise DirectVectorLiveTauratV73Error(
                    "Q3/v73 admits only TraFrm=-1 among negative tram routes"
                )
            live_count += 1
        elif semantics != TAURAT_POSITIVE_ZERO:
            raise DirectVectorLiveTauratV73Error(
                "Q3/v73 contains an unaudited taurat semantic"
            )
        rows.append({
            "fraction_index": row["fraction_index"],
            "fraction_id": row["fraction_id"],
            "method_id": method,
            "reference_selector": selector,
            "taurat_semantics": semantics,
        })
    if live_count == 0:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 requires at least one Van Rijn 1993 TraFrm=-1 fraction"
        )

    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    source_contract = verify_locked_sources_v69(root)
    sources = _implementation_sources(root, parent["backend"])
    body: dict[str, object] = {
        "schema": SCHEMA,
        "scope_id": SCOPE_ID,
        "envelope_version": ENVELOPE_VERSION,
        "parent_execution": parent,
        "parent_execution_fingerprint": parent["fingerprint"],
        "bed_slope_execution": slope,
        "bed_slope_execution_fingerprint": slope["fingerprint"],
        "taurat_source_contract": source_contract,
        "taurat_source_contract_fingerprint": source_contract["fingerprint"],
        "backend": parent["backend"],
        "integrator": parent["integrator"],
        "method_id": METHOD_ID,
        "reference_islope": 2,
        "stress_semantics": STRESS_SEMANTICS,
        "fraction_methods": rows,
        "live_fraction_count": live_count,
        "producer_output_channel": "bedbc1993_taubcw_over_taucr1",
        "face_projection": {
            "internal": "arithmetic_average_of_adjacent_cell_outputs",
            "open_boundary": "interior_owner_cell_output",
            "closed": "exact_zero_inert",
        },
        "unrelated_shear_diagnostic_substitution_allowed": False,
        "excluded_negative_tram": {
            "TraFrm=-2": "Q7 Van Rijn 2007 owner",
            "TraFrm=-4": "Q7 SANTOSS owner",
        },
        "composition_order": (
            "same_invocation_fraction_free_producer_and_taurat_then_"
            "face_projection_then_one_islope2_then_inventory_then_exner"
        ),
        "accepted_step_limit": MAX_ACCEPTED_STEPS,
        "automatic_backend_fallback": False,
        "normal_product_admission": False,
        "implementation_sources": sources,
        "implementation_source_fingerprint": _fingerprint(sources),
    }
    return {**body, "fingerprint": _fingerprint(body)}


def load_direct_vector_live_taurat_execution_v73(
    value: Mapping[str, object], *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 execution must be an object"
        )
    try:
        rebuilt = direct_vector_live_taurat_execution_document_v73(
            parent_execution=value["parent_execution"],
            bed_slope_execution=value["bed_slope_execution"],
            repository_root=repository_root,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 execution identity is invalid"
        ) from error
    if dict(value) != rebuilt or _canonical(value) != _canonical(rebuilt):
        raise DirectVectorLiveTauratV73Error(
            "Q3/v73 execution identity is stale or forged"
        )
    return rebuilt


def verify_direct_vector_live_taurat_sources_v73(
    execution: Mapping[str, object], *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    resolved = load_direct_vector_live_taurat_execution_v73(
        execution, repository_root=repository_root,
    )
    root = (Path(repository_root) if repository_root is not None else
            Path(__file__).resolve().parents[2])
    for row in resolved["implementation_sources"]:
        path = root / row["path"]
        if (not path.is_file() or
                hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise DirectVectorLiveTauratV73Error(
                f"Q3/v73 implementation source differs: {row['path']}"
            )
    parent_receipt = verify_direct_vector_family_sources_v67(
        resolved["parent_execution"], repository_root=root,
    )
    return {
        "scope_id": SCOPE_ID,
        "execution_fingerprint": resolved["fingerprint"],
        "taurat_source_contract_fingerprint":
            resolved["taurat_source_contract_fingerprint"],
        "implementation_source_fingerprint":
            resolved["implementation_source_fingerprint"],
        "parent_source_receipt_fingerprint": parent_receipt["fingerprint"],
        "verified": True,
    }


__all__ = [
    "DirectVectorLiveTauratV73Error", "ENVELOPE_VERSION",
    "MAX_ACCEPTED_STEPS", "METHOD_ID", "SCHEMA", "SCOPE_ID",
    "STRESS_SEMANTICS", "direct_vector_live_taurat_execution_document_v73",
    "load_direct_vector_live_taurat_execution_v73",
    "verify_direct_vector_live_taurat_sources_v73",
]
