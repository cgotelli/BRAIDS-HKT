"""Locked classic-FLOW ``taurat`` ownership for bed-slope ISlope=2.

The classic FLOW call chain initializes ``erosed``'s cell/class ``taurat``
array to zero and passes one element to ``eqtran``.  Every enabled positive
``TraFrm`` branch leaves the ``intent(out)`` argument unassigned.  That is a
Fortran defect: the returned value is not standard-defined.  A probe compiling
the exact locked ``eqtran.f90`` with gfortran 13.2 observed byte preservation
at ``-O0/-O2/-O3``; the locked ifx 2024.2.1 executable does not expose those
bytes.  BRAIDS therefore binds positive routes explicitly to FLOW's caller-zero
intent and records the compiler limitation, rather than claiming portable
language-level equivalence.  The negative Van Rijn branches are different: their
``bedbc1993``/``bedbc2004`` calls return ``taubcw/taucr1`` through ``taurat``.

This module is an additive, backend-neutral source contract.  It prevents a
capacity diagnostic, RP_TAUB, or RP_USTAR-derived Shields value from being
silently substituted for positive-formula ``taurat``.  It also owns the exact
cell-to-face projection used by the bounded BRAIDS CPU/CUDA implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np


SCHEMA = 1
ENVELOPE_VERSION = 69
SCOPE_ID = "braids-q3-positive-trafrm-islope2-taurat-v69-v1"
LOCKED_REFERENCE_COMMIT = "c05ed672f0a7fe81c5b9867ef354cfdd53c25379"

TAURAT_POSITIVE_ZERO = "classic_flow_positive_trafrm_erosed_zero"
TAURAT_TRAM_LIVE = "classic_flow_negative_tram_taubcw_over_taucr1"

RUNTIME_PROBE_EVIDENCE = MappingProxyType({
    "probe_scope": "exact_locked_eqtran_source_positive_iform12_link_stub_ABI",
    "compiler": "GNU Fortran 13.2.0 MinGW-W64 x86_64-ucrt-posix-seh",
    "optimization_levels": ("O0", "O2", "O3"),
    "caller_zero_output_bits": "0000000000000000",
    "caller_sentinel_input_bits": "3FF3C083126E978D",
    "caller_sentinel_output_bits": "3FF3C083126E978D",
    "observed_behavior": "callee_preserved_caller_bytes",
    "locked_reference_compiler": "Intel ifx 2024.2.1",
    "locked_reference_output_bytes_observable": False,
    "fortran_standard_status": "undefined_unassigned_intent_out",
})

# Enabled/implemented positive branches in the locked eqtran selector chain.
# TraFrm 15 is an external DLL ABI and 22 is ASMITA; neither is a Q3 BRAIDS
# route, but both share the same positive-selector non-assignment semantics.
POSITIVE_EQTRAN_SELECTORS = frozenset(
    (1, 2, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22)
)
NEGATIVE_TRAM_SELECTORS = frozenset((-1, -2, -4))
DISABLED_POSITIVE_SELECTORS = frozenset((3, 6, 8, 9, 10))

# The six-method direct-vector family contains five positive formulae and the
# negative Van Rijn 1993 tram route.  The latter must supply its producer-owned
# live ratio; it may never be coerced to the positive-family zero.
DIRECT_VECTOR_METHOD_SELECTORS = MappingProxyType({
    "capacity.soulsby": 12,
    "capacity.soulsby_van_rijn": 11,
    "capacity.van_rijn_1993": -1,
    "capacity.bijker_current": 5,
    "capacity.van_thiel_van_rijn_2008_current": 19,
    "capacity.xbeach_soulsby_van_rijn_current": 20,
})

SOURCE_ANCHORS = MappingProxyType({
    "erosed": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/erosed.f90"
        ),
        "sha256": "4571fd7912233db03f597161af1a18702ee974483676004cda8186ccf24308c0",
        "anchors": ("taurat = 0.0_fp", "call eqtran", "call adjust_bedload"),
    },
    "eqtran": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/eqtran.f90"
        ),
        "sha256": "d2e8601cff68884f51ee3dcb0cd2fb52d9f0d676b1baaa3283c68913d4778388",
        "anchors": ("call tram1", "call tram2", "intent(out) :: taurat"),
    },
    "bedbc1993": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/bedbc1993.f90"
        ),
        "sha256": "7fd7b619a111e3ee99ad6625097882a0bf3baa0a0f6f6d39f939671390fd73d5",
        "anchors": ("taurat = taubcw/taucr1",),
    },
    "bedbc2004": {
        "path": (
            "resources/Delft3D-main/src/utils_gpl/morphology/packages/"
            "morphology_kernel/src/bedbc2004.f90"
        ),
        "sha256": "25f227bd47dc716c806df1e29d70106092e6dc0d4ed8012a4dde7fb241068654",
        "anchors": ("taurat  = taubcw / taucr1",),
    },
    "adjust_bedload": {
        "path": (
            "resources/Delft3D-main/src/engines_gpl/flow2d3d/packages/"
            "flow2d3d_kernel/src/compute_sediment/adjust_bedload.f90"
        ),
        "sha256": "a3dae8226082977ec1ee78748e0d439f0f7356d267a9f33cba58636a61afdcba",
        "anchors": (
            "tratio = (taurat(nm, l) + taurat(nm2, l)) / 2.0",
            "if (tratio >= 1.0) then",
        ),
    },
})

FACE_INTERNAL = 0
FACE_TRACE_INLET = 1
FACE_TRACE_OUTLET = 2
FACE_OWNER_OUTLET = 3
FACE_CLOSED = 4
FACE_ROLES = frozenset((
    FACE_INTERNAL, FACE_TRACE_INLET, FACE_TRACE_OUTLET,
    FACE_OWNER_OUTLET, FACE_CLOSED,
))


class ClassicFlowTauratV69Error(ValueError):
    """Fail-closed source, selector, topology, or stress-ratio error."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def taurat_semantics(reference_selector: int) -> str:
    """Return the only locked-source semantic admitted for ``TraFrm``."""
    if isinstance(reference_selector, bool) or not isinstance(
            reference_selector, (int, np.integer)):
        raise ClassicFlowTauratV69Error("TraFrm selector must be an integer")
    selector = int(reference_selector)
    if selector in POSITIVE_EQTRAN_SELECTORS:
        return TAURAT_POSITIVE_ZERO
    if selector in NEGATIVE_TRAM_SELECTORS:
        return TAURAT_TRAM_LIVE
    if selector in DISABLED_POSITIVE_SELECTORS:
        raise ClassicFlowTauratV69Error(
            f"TraFrm {selector} is disabled before a usable taurat is returned"
        )
    raise ClassicFlowTauratV69Error(
        f"TraFrm {selector} has no audited classic-FLOW taurat route"
    )


def direct_vector_taurat_semantics(method_id: str) -> str:
    try:
        selector = DIRECT_VECTOR_METHOD_SELECTORS[method_id]
    except (KeyError, TypeError) as error:
        raise ClassicFlowTauratV69Error(
            "direct-vector method identity is not in the six-method family"
        ) from error
    return taurat_semantics(selector)


def resolve_cell_taurat(
    reference_selectors: Sequence[int] | np.ndarray,
    *,
    ncell: int,
    live_stress_ratio: object | None = None,
) -> np.ndarray:
    """Resolve exact cell/class ``taurat`` without diagnostic substitution.

    The returned shape is ``(nfraction,ncell)``.  Positive rows are filled by
    assignment with exact FP64 zero.  A negative tram row requires its live
    producer output in ``live_stress_ratio``; the caller cannot infer it from
    another shear diagnostic.
    """
    raw = np.asarray(reference_selectors)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
        raise ClassicFlowTauratV69Error(
            "reference_selectors must be a nonempty integer vector"
        )
    if isinstance(ncell, bool) or not isinstance(ncell, int) or ncell <= 0:
        raise ClassicFlowTauratV69Error("ncell must be a positive integer")
    selectors = np.asarray(raw, dtype=np.int64)
    semantics = tuple(taurat_semantics(int(value)) for value in selectors)
    needs_live = np.asarray(
        [value == TAURAT_TRAM_LIVE for value in semantics], dtype=bool
    )
    if live_stress_ratio is None:
        if np.any(needs_live):
            raise ClassicFlowTauratV69Error(
                "negative tram TraFrm requires its producer-owned live taurat"
            )
        live = np.zeros((selectors.size, ncell), dtype=np.float64)
    else:
        live = np.asarray(live_stress_ratio, dtype=np.float64)
        if live.shape != (selectors.size, ncell):
            raise ClassicFlowTauratV69Error(
                "live_stress_ratio must have shape (nfraction,ncell)"
            )
        if not np.all(np.isfinite(live)) or np.any(live < 0.0):
            raise ClassicFlowTauratV69Error(
                "live tram taurat must be finite and nonnegative"
            )
        # Values in positive rows are forbidden, even if a caller labels them
        # diagnostic-only.  This catches the exact accidental substitution the
        # source audit was opened to prevent.
        if np.any(live[~needs_live] != 0.0):
            raise ClassicFlowTauratV69Error(
                "positive TraFrm taurat must be exact zero"
            )
    result = np.zeros((selectors.size, ncell), dtype=np.float64)
    if np.any(needs_live):
        result[needs_live] = live[needs_live]
    result.setflags(write=False)
    return result


def project_cell_taurat_to_faces(
    cell_taurat: object,
    *,
    owner: object,
    neighbour: object,
    role: object,
) -> np.ndarray:
    """Apply classic FLOW's internal-average/boundary-interior face rule."""
    cell = np.asarray(cell_taurat, dtype=np.float64)
    if cell.ndim != 2 or min(cell.shape) <= 0 or \
            not np.all(np.isfinite(cell)) or np.any(cell < 0.0):
        raise ClassicFlowTauratV69Error(
            "cell_taurat must be finite nonnegative (nfraction,ncell)"
        )
    own_raw = np.asarray(owner)
    other_raw = np.asarray(neighbour)
    roles_raw = np.asarray(role)
    nface = own_raw.size
    if (own_raw.ndim != 1 or other_raw.shape != (nface,) or
            roles_raw.shape != (nface,) or own_raw.dtype.kind not in "iu" or
            other_raw.dtype.kind not in "iu" or roles_raw.dtype.kind not in "iu"):
        raise ClassicFlowTauratV69Error(
            "face topology must be three equal-length integer vectors"
        )
    own = np.asarray(own_raw, dtype=np.int64)
    other = np.asarray(other_raw, dtype=np.int64)
    roles = np.asarray(roles_raw, dtype=np.int64)
    internal = roles == FACE_INTERNAL
    boundary = ~internal
    if (nface <= 0 or np.any(~np.isin(roles, tuple(FACE_ROLES))) or
            np.any(own < 0) or np.any(own >= cell.shape[1]) or
            np.any(other[internal] < 0) or
            np.any(other[internal] >= cell.shape[1]) or
            np.any(other[internal] == own[internal]) or
            np.any(other[boundary] != -1)):
        raise ClassicFlowTauratV69Error(
            "face roles and owner/neighbour topology are inconsistent"
        )
    result = np.asarray(cell[:, own], dtype=np.float64).copy(order="C")
    if np.any(internal):
        result[:, internal] = 0.5*(
            cell[:, own[internal]] + cell[:, other[internal]]
        )
    # Closed faces are inert.  Keeping exact zero prevents stale values from
    # entering a later role remap or a device kernel.
    result[:, roles == FACE_CLOSED] = 0.0
    result.setflags(write=False)
    return result


def source_contract_document_v69() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": SCHEMA,
        "envelope_version": ENVELOPE_VERSION,
        "scope_id": SCOPE_ID,
        "locked_reference_commit": LOCKED_REFERENCE_COMMIT,
        "positive_eqtran_selectors": sorted(POSITIVE_EQTRAN_SELECTORS),
        "disabled_positive_selectors": sorted(DISABLED_POSITIVE_SELECTORS),
        "negative_tram_selectors": sorted(NEGATIVE_TRAM_SELECTORS),
        "positive_semantics": TAURAT_POSITIVE_ZERO,
        "negative_semantics": TAURAT_TRAM_LIVE,
        "positive_language_semantics": "undefined_unassigned_intent_out",
        "braids_positive_binding": (
            "exact_zero_from_FLOW_erosed_caller_intent_with_gfortran_"
            "byte_preservation_evidence"
        ),
        "runtime_probe_evidence": dict(RUNTIME_PROBE_EVIDENCE),
        "positive_diagnostic_substitution_allowed": False,
        "face_projection": {
            "internal": "arithmetic_average_of_two_cell_values",
            "open_boundary": "interior_owner_cell_value",
            "closed": "exact_zero_inert",
        },
        "direct_vector_family": [
            {
                "method_id": method,
                "reference_selector": selector,
                "taurat_semantics": taurat_semantics(selector),
            }
            for method, selector in DIRECT_VECTOR_METHOD_SELECTORS.items()
        ],
        "source_anchors": {
            name: {
                "path": row["path"],
                "sha256": row["sha256"],
                "anchors": list(row["anchors"]),
            }
            for name, row in SOURCE_ANCHORS.items()
        },
    }
    return {**body, "fingerprint": _fingerprint(body)}


def verify_locked_sources_v69(root: str | Path = ".") -> dict[str, object]:
    """Hash and inspect the locked read-only source without modifying it."""
    base = Path(root)
    for name, row in SOURCE_ANCHORS.items():
        path = base / str(row["path"])
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ClassicFlowTauratV69Error(
                f"locked {name} source is unavailable"
            ) from error
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ClassicFlowTauratV69Error(
                f"locked {name} source hash differs"
            )
        source = payload.decode("utf-8")
        for anchor in row["anchors"]:
            if str(anchor).lower() not in source.lower():
                raise ClassicFlowTauratV69Error(
                    f"locked {name} source anchor differs"
                )
    return source_contract_document_v69()


__all__ = [
    "ClassicFlowTauratV69Error", "DIRECT_VECTOR_METHOD_SELECTORS",
    "DISABLED_POSITIVE_SELECTORS", "ENVELOPE_VERSION", "FACE_CLOSED",
    "FACE_INTERNAL", "FACE_OWNER_OUTLET", "FACE_TRACE_INLET",
    "FACE_TRACE_OUTLET", "LOCKED_REFERENCE_COMMIT",
    "NEGATIVE_TRAM_SELECTORS", "POSITIVE_EQTRAN_SELECTORS", "SCHEMA",
    "RUNTIME_PROBE_EVIDENCE", "SCOPE_ID", "SOURCE_ANCHORS", "TAURAT_POSITIVE_ZERO",
    "TAURAT_TRAM_LIVE", "direct_vector_taurat_semantics",
    "project_cell_taurat_to_faces", "resolve_cell_taurat",
    "source_contract_document_v69", "taurat_semantics",
    "verify_locked_sources_v69",
]
