"""Fail-closed audit seam for the native Q3-H3/v59 CPU AMR route.

The current native AMR controller is implemented inside the cut-cell Fortran
program.  Its hierarchy cache (``at_q``), selected leaves (``at_leaf``),
physical-face lineage (``at_face_source``), retry transaction, and regrid
publication are internal program state.  Consequently an out-of-process
Python adapter cannot safely install the detached v59 operators after launch.

This module records that fact as an executable source audit.  It deliberately
does not admit a solver: the accepted production runtime is v57, the frozen
v58 runtime is fixed-grid, and the v59 native module is a bounded component.
The first future live route must be a new additive native envelope which calls
v59 from inside the existing AMR transaction.  No frozen parent is edited.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


SCOPE_ID = "q3h3-v59-native-cpu-amr-composition-seam-v1"
STATUS = "blocked_not_live_admission"
REQUIRED_INTERFACES = (
    "new_additive_native_envelope",
    "internal_selected_mesh_projection",
    "external_macrostep_hi_operator_ownership",
    "corrected_depth_weighted_betac_override",
    "atomic_regrid_handoff",
    "checkpoint_and_receipt_upgrade",
)


class NativeCPUAMRCompositionUnavailable(RuntimeError):
    """Raised before a component can be mistaken for a live CPU route."""

    def __init__(self, report: Mapping[str, object]):
        self.report = dict(report)
        super().__init__(
            "Q3-H3/v59 native CPU AMR composition is unavailable: "
            + "; ".join(str(value) for value in report.get("blockers", ()))
        )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def required_interface_patch() -> dict[str, object]:
    """Return the exact additive code seams required by a live CPU route."""
    body = {
        "schema": 1,
        "scope_id": SCOPE_ID,
        "status": STATUS,
        "operator_ownership": {
            "accepted_step_retry_and_clock": "existing_native_amr_controller",
            "hierarchy_cache_and_selected_leaf_publication":
                "existing_native_amr_controller",
            "signed_hi_transfer_and_composite_register": "v59_exactly_once",
            "legacy_v58_betac": "disabled_not_composed",
            "corrected_depth_weighted_betac": "v59_exactly_once",
            "coriolis": "v59_exactly_once",
            "other_conserved_rows": "existing_native_amr_controller",
        },
        "interfaces": [
            {
                "id": "new_additive_native_envelope",
                "requirement": (
                    "Create a new deck/runtime envelope above the frozen v57/v58 "
                    "parents; compile the v59 module into that new runtime and bind "
                    "its source and corrected Betac amendment in the build receipt."
                ),
            },
            {
                "id": "internal_selected_mesh_projection",
                "requirement": (
                    "Inside the Fortran program, project at_leaf, at_level, area, "
                    "xy, owner/neighbour and at_face_source to one ordered, "
                    "non-overlapping, 2:1 physical-segment catalog. External Python "
                    "cannot access or atomically mutate this internal state."
                ),
            },
            {
                "id": "external_macrostep_hi_operator_ownership",
                "requirement": (
                    "Give v59 sole ownership of signed-hI advection/diffusion for "
                    "the synchronous fully-wet step. Mask the inherited H3 hI flux "
                    "and ledger in that route so v57/v58 and v59 never advance it twice."
                ),
            },
            {
                "id": "corrected_depth_weighted_betac_override",
                "requirement": (
                    "Replace, rather than supplement, the frozen v58 Betac residual "
                    "with v59 owner=-h_owner*K and neighbour=+h_neighbour*K updates."
                ),
            },
            {
                "id": "atomic_regrid_handoff",
                "requirement": (
                    "Call v59 signed-hI restriction/prolongation within amr_regrid "
                    "before amr_rebuild_mesh publishes at_leaf/q; keep all other rows "
                    "under the existing transfer owner and roll back the whole trial "
                    "on cancellation or rejection."
                ),
            },
            {
                "id": "checkpoint_and_receipt_upgrade",
                "requirement": (
                    "Add a new checkpoint/runtime identity containing the v59 "
                    "execution identity, topology/face fingerprints, operator-owner "
                    "flags and pending-register invariant. Do not reinterpret schema 9."
                ),
            },
        ],
        "first_slice": {
            "time_coupling": "synchronous_one_clock",
            "subcycling": False,
            "fully_wet": True,
            "open_boundaries": False,
            "restart": False,
        },
        "automatic_fallback": False,
        "Delft3D_source_modified": False,
    }
    return {
        **body,
        "fingerprint": hashlib.sha256(_canonical(body)).hexdigest(),
    }


@dataclass(frozen=True)
class SourceAudit:
    report: dict[str, object]

    @property
    def admitted(self) -> bool:
        return self.report.get("status") == "admitted"


def audit_current_sources(root: str | Path) -> SourceAudit:
    """Inspect the authoritative current sources without compiling or running."""
    root = Path(root).resolve()
    paths = {
        "production_solver": root / "solver/cut_cells/cutcell_solver.f90",
        "production_builder": root / "tools/build_cutcell.py",
        "deck_writer": root / "braids_core/boundaries/cut_cell_io.py",
        "qgis_project": root / "braids_qgis/core/project.py",
        "frozen_v58_solver": root / (
            "validation/sediment/delft3d_current/q3h_integration/h3_nonzero/"
            "native_v58/cutcell_solver_v58.f90"
        ),
        "v59_component": root / (
            "validation/sediment/delft3d_current/q3h_integration/h3_amr/"
            "native_v59/braids_flow2dh_prognostic_secondary_amr_v59_module.f90"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError("Native CPU AMR source audit is missing: " + ", ".join(missing))
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    facts = {
        "production_build_stops_at_v57": (
            '"supported_case_envelopes": list(range(1, 58))'
            in text["production_builder"]
        ),
        "production_build_omits_v59_module": (
            "braids_flow2dh_prognostic_secondary_amr_v59_module.f90"
            not in text["production_builder"]
        ),
        "deck_writer_rejects_h3_amr": (
            "amr_hierarchy is not None or sediment_shear_mode!='total'"
            in text["deck_writer"]
        ),
        "qgis_authenticates_cpu_v57_only": (
            "_require_cpu_v57_native_backend" in text["qgis_project"]
        ),
        "production_h3_validator_rejects_amr": (
            "if(exchange_mode/=0.or.bed_slope/=0.or.amr_enabled.or."
            in text["production_solver"]
        ),
        "production_owns_real_tree_cache": all(token in text["production_solver"] for token in (
            "at_q(nstate,amr_nn)", "subroutine amr_regrid(initial)",
            "subroutine amr_rebuild_mesh()", "at_face_source",
        )),
        "production_has_existing_signed_hi_transfer": (
            "subroutine amr_signed_water_transfer" in text["production_solver"]
        ),
        "frozen_v58_rejects_amr": (
            "if(exchange_mode/=0.or.bed_slope/=0.or.amr_enabled.or."
            in text["frozen_v58_solver"] and
            "if(amr_enabled.or.any(state(1,:)<=dry_tolerance)"
            in text["frozen_v58_solver"]
        ),
        "v59_is_component_module": all(token in text["v59_component"] for token in (
            "module braids_flow2dh_prognostic_secondary_amr_v59_module",
            "braids_amr_composite_hi_step_v59",
            "braids_amr_betac_segments_v59",
        )),
    }
    if not all(facts.values()):
        failed = [name for name, value in facts.items() if not value]
        raise ValueError(
            "Native CPU AMR source layout changed; re-audit required: "
            + ", ".join(failed)
        )

    blockers = [
        "the production/QGIS native runtime authenticates envelopes only through v57",
        "the deck writer and native H3 validator explicitly reject AMR",
        "the frozen v58 nonzero route is fixed-grid and rejects AMR",
        "v59 is not compiled into the production runtime",
        "the real AMR tree and retry transaction are internal to the Fortran program",
        "hI and corrected Betac ownership must be replaced inside that transaction to avoid duplicate physics",
    ]
    patch = required_interface_patch()
    report = {
        "schema": 1,
        "scope_id": SCOPE_ID,
        "status": "blocked",
        "live_route_admitted": False,
        "automatic_fallback": False,
        "facts": facts,
        "blockers": blockers,
        "required_interface_patch_fingerprint": patch["fingerprint"],
        "source_sha256": {
            name: _sha(path) for name, path in paths.items()
        },
        "Delft3D_source_modified": False,
    }
    return SourceAudit(report)


def require_live_route(root: str | Path) -> None:
    """Fail before launch; automatic use of v57/v58 is forbidden."""
    audit = audit_current_sources(root)
    if not audit.admitted:
        raise NativeCPUAMRCompositionUnavailable(audit.report)


__all__ = [
    "NativeCPUAMRCompositionUnavailable",
    "REQUIRED_INTERFACES",
    "SCOPE_ID",
    "SourceAudit",
    "audit_current_sources",
    "require_live_route",
    "required_interface_patch",
]
