"""Resolve existing sediment defaults without changing numerical methods.

This resolver describes the current public law-0--8 project configuration.
It does not admit a native/GPU route, validate all physical inputs, or infer
CPU/GPU choice.  Those checks remain with the project and runtime validators.
Source identifiers describe constitutive provenance, not solver equivalence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from braids_core.sediment.capacity_execution_identity import (
    EH_DELFT_METHOD_ID,
    EH_LEGACY_METHOD_ID,
    MPM_DELFT_METHOD_ID,
    MPM_LEGACY_METHOD_ID,
)
from braids_core.sediment_method_catalog import (
    CATALOG_SCHEMA,
    CATALOG_SCOPE_ID,
    capacity_source_by_legacy_code,
    legacy_code_map,
    semantic_catalog_id,
    slope_source_by_legacy_code,
)


_CAPACITY_SOURCES = capacity_source_by_legacy_code()
_SLOPE_SOURCES = slope_source_by_legacy_code()


class SedimentMethodSelectionError(ValueError):
    """A configuration conflict, labelled separately from physical validity."""

    def __init__(self, message: str, *, category: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class SedimentMethodSelection:
    bedload_law: int
    bed_slope: int
    selection_origin: str
    capacity_method_id: str
    capacity_source_id: str
    slope_source_id: str
    exchange_mode: int | None
    mpm_friction_mode: int | None
    messages: tuple[str, ...]
    restrictions: tuple[str, ...]
    catalog_schema: int
    catalog_scope_id: str
    semantic_catalog_id: str
    selection_components: tuple[str, ...]

    def as_dict(self) -> dict:
        """Return JSON-compatible metadata without claiming backend support."""
        result = asdict(self)
        result["messages"] = list(self.messages)
        result["restrictions"] = list(self.restrictions)
        result["selection_components"] = list(self.selection_components)
        return result


def _integer_choice(value: object, name: str, choices: tuple[int, ...]) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value not in choices:
        raise SedimentMethodSelectionError(
            f"{name} must be an integer in {choices}",
            category="invalid_configuration",
        )


def resolve_sediment_methods(
    bedload_law: int,
    *,
    slope_enabled: bool = True,
    slope_method: int | None = None,
    exchange_mode: int | None = None,
    mpm_friction_mode: int | None = None,
    capacity_method_id: str | None = None,
) -> SedimentMethodSelection:
    """Choose compatible existing defaults, preserving explicit saved choices.

    ``slope_method=None`` means automatic: active laws 1--7 select mode 2,
    law 8 selects mode 3, and a disabled slope switch/law 0 selects mode 0.
    Supplying a numeric method makes it explicit; mode 0 remains off even if
    the switch is enabled.  Conflicting explicit choices raise an error rather
    than being rewritten.  Law 9 retains its separate GAIA research contract.

    Optional exchange and MPM shear values are retained and checked against
    the existing project scope.  Omitting either value makes no new choice.
    In particular, capacity/slope resolution does not silently disable a saved
    suspension setting or replace a fixed-Cf MPM closure with Delft3D shear.
    """
    if isinstance(bedload_law, int) and not isinstance(bedload_law, bool) and bedload_law == 9:
        raise SedimentMethodSelectionError(
            "TELEMAC-GAIA law 9 has a separate research configuration; "
            "it is outside the Delft3D automatic-selection scope",
            category="scope_restriction",
        )
    _integer_choice(bedload_law, "Bed-load law", tuple(range(9)))
    if not isinstance(slope_enabled, bool):
        raise SedimentMethodSelectionError(
            "slope_enabled must be boolean", category="invalid_configuration"
        )
    if slope_method is not None:
        _integer_choice(slope_method, "Slope method", (0, 2, 3, 5))
        if slope_method != 0 and not slope_enabled:
            raise SedimentMethodSelectionError(
                "An explicit active slope method conflicts with the disabled slope switch",
                category="invalid_configuration",
            )
        selected_slope = slope_method
        origin = "explicit"
    else:
        selected_slope = (
            (3 if bedload_law == 8 else 2)
            if slope_enabled and bedload_law != 0 else 0
        )
        origin = "automatic"

    if selected_slope and bedload_law == 0:
        raise SedimentMethodSelectionError(
            "Slope correction requires an active bed-load formulation",
            category="invalid_configuration",
        )
    if selected_slope in (3,5) and bedload_law not in (2,8):
        raise SedimentMethodSelectionError(
            "The current Delft3D FLOW slope input contract is implemented for "
            "MPM law 2 and Engelund-Hansen law 8; other laws need their own "
            "Shields/stress contract. This is an implementation restriction, "
            "not a general physical incompatibility",
            category="implementation_restriction",
        )
    if exchange_mode is not None:
        _integer_choice(exchange_mode, "Suspended exchange mode", (0, 1, 2, 3))
        if bedload_law == 8 and exchange_mode != 0:
            raise SedimentMethodSelectionError(
                "Engelund-Hansen law 8 is currently qualified with suspended "
                "exchange off (mode 0); the explicit exchange choice was not changed",
                category="implementation_restriction",
            )
    if mpm_friction_mode is not None:
        _integer_choice(mpm_friction_mode, "MPM friction mode", (0, 1))
        if mpm_friction_mode == 1 and bedload_law != 2:
            raise SedimentMethodSelectionError(
                "Roughness-corrected MPM shear requires MPM bed-load law 2",
                category="implementation_restriction",
            )

    base_capacity_method_id = {
        code: row.method_id
        for code, row in legacy_code_map("capacity").items()
    }[bedload_law]
    resolved_capacity_method_id = base_capacity_method_id
    capacity_source = _CAPACITY_SOURCES[bedload_law]
    restrictions = [
        "Project, material, CPU/GPU route and validation restrictions still apply; "
        "this selection does not grant runtime support."
    ]
    if bedload_law == 2:
        resolved_capacity_method_id = (
            MPM_DELFT_METHOD_ID if mpm_friction_mode == 1
            else MPM_LEGACY_METHOD_ID
        )
        if capacity_method_id is not None and capacity_method_id != resolved_capacity_method_id:
            raise SedimentMethodSelectionError(
                "The explicit MPM capacity identity conflicts with the saved "
                "friction-mode closure",
                category="identity_mismatch",
            )
        capacity_source = (
            "d3d4_mpm_tranb2_resolved_hydraulic_roughness"
            if resolved_capacity_method_id == MPM_DELFT_METHOD_ID else
            "braids_zhu_mpm_specified_friction"
        )
        if resolved_capacity_method_id != MPM_DELFT_METHOD_ID:
            restrictions.append(
                "MPM capacity provenance depends on shear closure; the fixed-Cf "
                "benchmark is not the Delft3D tranb2 roughness-corrected operator."
            )
    elif bedload_law == 8:
        if capacity_method_id is None:
            # Every existing integer-only law-8 project had an independent
            # constant Chezy.  Absence of the new identity must preserve it.
            resolved_capacity_method_id = EH_LEGACY_METHOD_ID
        elif capacity_method_id in (EH_LEGACY_METHOD_ID, EH_DELFT_METHOD_ID):
            resolved_capacity_method_id = capacity_method_id
        else:
            raise SedimentMethodSelectionError(
                "The explicit Engelund-Hansen capacity identity conflicts with law 8",
                category="identity_mismatch",
            )
        capacity_source = (
            "d3d4_engelund_hansen_resolved_hydraulic_chezy"
            if resolved_capacity_method_id == EH_DELFT_METHOD_ID else
            "braids_engelund_hansen_independent_constant_chezy"
        )
    elif capacity_method_id is not None and capacity_method_id != base_capacity_method_id:
        raise SedimentMethodSelectionError(
            "The explicit capacity identity conflicts with the saved bed-load law",
            category="identity_mismatch",
        )
    if selected_slope == 3:
        restrictions.append(
            "Koch-Flokstra currently specializes CShld=DShld=0; its third "
            "stored parameter is BShld, not a friction angle."
        )
    if selected_slope == 2:
        restrictions.append(
            "Legacy mode 2 retains the BRAIDS capacity-stress-ratio input; "
            "use explicit mode 5 for the locked FLOW ISlope=2 contract."
        )
    if selected_slope == 5:
        restrictions.append(
            "BRAIDS mode 5 is Delft3D FLOW ISlope=2, not Delft3D ISlope=5. "
            "For positive EH/MPM formulas, FLOW leaves taurat zero; preserve "
            "that source contract separately from legacy BRAIDS mode 2."
        )
    if bedload_law == 8:
        restrictions.append((
            "Historical Engelund-Hansen uses its explicit independent constant "
            "Chezy coefficient. The Delft identity consumes the Chezy value "
            "resolved from typed hydraulic roughness; neither route may be "
            "silently substituted for the other."
        ))
    messages = (
        (f"Automatically resolved bed-slope method {selected_slope} for bed-load law {bedload_law}.",)
        if origin == "automatic" else
        (f"Preserved explicit bed-slope method {selected_slope} for bed-load law {bedload_law}.",)
    )
    return SedimentMethodSelection(
        bedload_law=bedload_law,
        bed_slope=selected_slope,
        selection_origin=origin,
        capacity_method_id=resolved_capacity_method_id,
        capacity_source_id=capacity_source,
        slope_source_id=_SLOPE_SOURCES[selected_slope],
        exchange_mode=exchange_mode,
        mpm_friction_mode=mpm_friction_mode,
        messages=messages,
        restrictions=tuple(restrictions),
        catalog_schema=CATALOG_SCHEMA,
        catalog_scope_id=CATALOG_SCOPE_ID,
        semantic_catalog_id=semantic_catalog_id(),
        selection_components=("capacity", "bed_slope", "suspension", "shear_closure"),
    )
