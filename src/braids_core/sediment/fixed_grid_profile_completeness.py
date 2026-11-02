"""Conditional execution-profile completeness for the fixed-grid GPU slice.

This is intentionally narrower than the complete FLOW selector ledger.  Many
ledger families apply only to particular boundary, layer, cohesive-sediment or
bedform configurations.  The fixed-grid runtime always consumes the seven
groups below and conditionally consumes settling for combined suspended load.
"""
from __future__ import annotations

from collections.abc import Mapping

from braids_core.sediment_method_catalog import method

from .capacity_profile import capacity_profile_document, load_capacity_profile
from .channel_composition import (
    channel_composition_profile_document, load_channel_composition_profile)
from .execution_profile_v2 import (
    ExecutionProfileError, Flow2DHExecutionProfile, bind_component_profile,
    load_selector_ledger)
from .suspension_profile import (
    load_suspension_profile, suspension_profile_document)


_CORE_FAMILIES = frozenset((
    "capacity", "transport_route", "suspension", "sediment_shear",
    "hiding", "bed_slope", "secondary_flow", "hydraulic_roughness",
))
_OPTIONAL_FAMILIES = frozenset(("settling",))
_Q3C_FIXED_GRID_CAPACITY_CODES = {
    "capacity.mpm": 2,
    "capacity.engelund_hansen": 8,
    "capacity.general": 11,
}


def _reject(message: str) -> None:
    raise ExecutionProfileError(message, category="missing_scope")


def _prepared_integer(data: Mapping[str, object], key: str) -> int:
    try:
        raw = data[key]
        value = int(raw)  # Accept NumPy integer scalars without importing NumPy.
        if isinstance(raw, bool) or raw != value:
            raise ValueError(key)
        return value
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ExecutionProfileError(
            "Prepared state lacks exact execution-profile field: " + key,
            category="identity_mismatch",
        ) from error


def _one_entry(by_family: Mapping[str, list], family: str) -> str:
    rows = by_family[family]
    values = {row.entry_id for row in rows}
    if len(values) != 1:
        _reject(
            "Fixed-grid numeric fields cannot represent mixed " + family +
            " selectors")
    return next(iter(values))


def _canonical_component_documents(
        profile: Flow2DHExecutionProfile,
        documents: Mapping[str, object]) -> dict[str, dict]:
    expected = {row.component_id: row for row in profile.component_profiles}
    if set(documents) != set(expected):
        _reject("Execution-profile component document inventory differs")
    loaders = {
        "capacity": (load_capacity_profile, capacity_profile_document),
        "channel_composition": (
            load_channel_composition_profile,
            channel_composition_profile_document,
        ),
        "suspension": (load_suspension_profile, suspension_profile_document),
    }
    result: dict[str, dict] = {}
    for component_id in sorted(documents):
        document = documents[component_id]
        if not isinstance(document, Mapping) or component_id not in loaders:
            _reject("Unsupported fixed-grid component document: " + component_id)
        loader, writer = loaders[component_id]
        try:
            canonical = writer(loader(document))
        except (TypeError, ValueError) as error:
            raise ExecutionProfileError(
                "Non-canonical fixed-grid component document: " + component_id,
                category="identity_mismatch",
            ) from error
        if canonical != document or bind_component_profile(
                component_id, canonical) != expected[component_id]:
            raise ExecutionProfileError(
                "Execution-profile component document differs: " + component_id,
                category="identity_mismatch",
            )
        result[component_id] = canonical
    return result


def validate_fixed_grid_profile_completeness(
        profile: Flow2DHExecutionProfile, *,
        component_documents: Mapping[str, object] | None = None,
        prepared_data: Mapping[str, object] | None = None,
        ) -> Flow2DHExecutionProfile:
    """Require the exact Q3-S fixed-grid selector/component/data slice.

    Boundary, bed-layer and other globally conditional ledger families are not
    claimed by this helper.  Supplying one here is rejected because this GPU
    slice does not consume it.  The native deck remains authoritative for
    those separately authenticated families.
    """
    entries = {row["entry_id"]: row for row in load_selector_ledger()["entries"]}
    selected = [(row, entries[row.entry_id]["family"])
                for row in profile.method_selections]
    fractions = {row.fraction_id for row in profile.fraction_layout}
    by_family: dict[str, list] = {}
    for row, family in selected:
        by_family.setdefault(family, []).append(row)

    unexpected = set(by_family) - _CORE_FAMILIES - _OPTIONAL_FAMILIES
    if unexpected:
        _reject(
            "Fixed-grid execution profile contains an unconsumed method family: " +
            sorted(unexpected)[0])

    for family in ("capacity", "transport_route", "suspension"):
        rows = by_family.get(family, ())
        scoped = [dict(row.scope).get("fraction_id") for row in rows]
        if len(rows) != len(fractions) or set(scoped) != fractions:
            _reject(
                "Execution profile lacks complete per-fraction " +
                family + " selectors")

    for family in ("sediment_shear", "hiding", "bed_slope", "secondary_flow"):
        rows = by_family.get(family, ())
        if len(rows) != 1 or dict(rows[0].scope) != {"model_id": "model"}:
            _reject(
                "Execution profile lacks one model-scoped " +
                family + " selector")

    # Q3-C binds the hydraulic C/n/ks method because all three scalar laws
    # consume its resolved Chezy value.  Preserve the historical seven-method
    # Q3-S/P law-11 receipt, but never let laws 2/8 or a profile that declares
    # roughness proceed with zero/multiple selectors.
    roughness_rows = by_family.get("hydraulic_roughness", ())
    capacity_methods = {row.entry_id for row in by_family.get("capacity", ())}
    legacy_q3sp = capacity_methods == {"capacity.general"} and not roughness_rows
    if not legacy_q3sp:
        if (len(roughness_rows) != 1 or
                dict(roughness_rows[0].scope) != {"model_id": "model"}):
            _reject(
                "Q3-C execution profile lacks one model-scoped "
                "hydraulic_roughness selector")

    capacity = {dict(row.scope)["fraction_id"]: row.entry_id
                for row in by_family["capacity"]}
    route = {dict(row.scope)["fraction_id"]: row.entry_id
             for row in by_family["transport_route"]}
    suspension = {dict(row.scope)["fraction_id"]: row.entry_id
                  for row in by_family["suspension"]}
    transport_class = {row.fraction_id: row.transport_class
                       for row in profile.fraction_layout}
    combined: set[str] = set()
    active: set[str] = set()
    for fraction_id in fractions:
        selected_route = route[fraction_id]
        capacity_off = capacity[fraction_id] == "capacity.off"
        suspension_off = suspension[fraction_id] == "suspension.off"
        if selected_route == "transport_route.off":
            # The fixed-grid wire has one canonical inactive class.  Permitting
            # class 2/3 here would create distinct identities for identical
            # numerical behaviour.
            valid = (capacity_off and suspension_off and
                     transport_class[fraction_id] == 1)
        elif selected_route == "transport_route.total":
            # Algebraic total load follows TRA_BEDLOAD ownership: capacity is
            # active, while no independently transported suspended reservoir
            # or settling selector is present.
            valid = (not capacity_off and suspension_off and
                     transport_class[fraction_id] == 1)
            active.add(fraction_id)
        elif selected_route == "transport_route.combine":
            valid = (not capacity_off and not suspension_off and
                     transport_class[fraction_id] == 3)
            combined.add(fraction_id)
            active.add(fraction_id)
        else:  # The installed ledger currently defines exactly these routes.
            valid = False
        if not valid:
            _reject(
                "Transport route/capacity/suspension selector combination differs")

    settling_rows = by_family.get("settling", ())
    settling = [dict(row.scope).get("fraction_id") for row in settling_rows]
    if len(settling_rows) != len(combined) or set(settling) != combined:
        _reject(
            "Only combined suspended transport may carry one settling selector")

    components = {row.component_id for row in profile.component_profiles}
    expected_components = (
        {"capacity", "channel_composition", "suspension"}
        if combined else
        {"capacity", "channel_composition"} if active else set()
    )
    if components != expected_components:
        _reject(
            "Fixed-grid execution-profile component inventory differs")

    # Dormant alternatives are not independent identities on an all-OFF
    # fixed-grid state.  Reject them instead of hashing selectors that no
    # numerical operator consumes.
    if not active and (
            _one_entry(by_family, "bed_slope") != "bed_slope.off" or
            _one_entry(by_family, "secondary_flow") != "secondary.off"):
        _reject("All-OFF execution profile selects a dormant active operator")

    canonical_documents = None
    if component_documents is not None:
        canonical_documents = _canonical_component_documents(
            profile, component_documents)
        layout = [{"fraction_id": row.fraction_id,
                   "transport_class": row.transport_class}
                  for row in profile.fraction_layout]
        if active:
            channel = canonical_documents["channel_composition"]
            if channel.get("fractions") != layout:
                raise ExecutionProfileError(
                    "Channel component differs from execution-profile layout",
                    category="identity_mismatch",
                )
            capacity_document = canonical_documents["capacity"]
            capacity_rows = capacity_document.get("fractions")
            if (not isinstance(capacity_rows, list) or
                    [row.get("fraction_id") for row in capacity_rows] !=
                    [row["fraction_id"] for row in layout]):
                raise ExecutionProfileError(
                    "Capacity component differs from execution-profile fractions",
                    category="identity_mismatch",
                )
            if combined:
                suspension_document = canonical_documents["suspension"]
                if suspension_document.get("capacity_profile") != capacity_document:
                    raise ExecutionProfileError(
                        "Suspension and capacity component identities differ",
                        category="identity_mismatch",
                    )

    if prepared_data is not None:
        if component_documents is None:
            raise ExecutionProfileError(
                "Prepared-state validation requires component documents",
                category="identity_mismatch",
            )
        if _prepared_integer(prepared_data, "nfrac") != len(fractions):
            raise ExecutionProfileError(
                "Prepared fraction count differs from execution profile",
                category="identity_mismatch",
            )
        capacity_id = _one_entry(by_family, "capacity")
        # Corrected Q3-C methods deliberately do not use the catalogue's
        # ``legacy_code`` field: codes 2 and 8 there identify distinct frozen
        # BRAIDS formulas.  The authenticated Q3-C profile nevertheless maps
        # to the native fixed-grid dispatch integers carried by its capacity
        # document.  Keep that mapping local to this exact profile gate.
        capacity_code = _Q3C_FIXED_GRID_CAPACITY_CODES.get(
            capacity_id, method(capacity_id).legacy_code)
        if type(capacity_code) is not int:
            _reject("Selected capacity has no fixed-grid numeric code")
        exact = {
            "bedlaw": capacity_code,
            "exchange_mode": (
                4 if _one_entry(by_family, "suspension") ==
                "suspension.van_rijn_exchange" else 0),
            "native_sediment_shear_mode": (
                0 if _one_entry(by_family, "sediment_shear") ==
                "shear.total_roughness" else 1),
            "hiding_exposure": method(
                _one_entry(by_family, "hiding")).legacy_code,
            "bed_slope": method(
                _one_entry(by_family, "bed_slope")).legacy_code,
            "secondary_flow_model": method(
                _one_entry(by_family, "secondary_flow")).legacy_code,
        }
        for key, expected_value in exact.items():
            if type(expected_value) is not int or \
                    _prepared_integer(prepared_data, key) != expected_value:
                raise ExecutionProfileError(
                    "Prepared numeric operator differs from execution profile: " + key,
                    category="identity_mismatch",
                )
        if roughness_rows:
            expected_roughness = roughness_rows[0].entry_id
            if prepared_data.get("capacity_roughness_method_id") != expected_roughness:
                raise ExecutionProfileError(
                    "Prepared typed roughness differs from execution profile",
                    category="identity_mismatch",
                )
        if active:
            capacity_rows = canonical_documents["capacity"]["fractions"]
            if any(row.get("formula_code") != capacity_code
                   for row in capacity_rows):
                raise ExecutionProfileError(
                    "Capacity component formula differs from prepared bedlaw",
                    category="identity_mismatch",
                )
    return profile


__all__ = ["validate_fixed_grid_profile_completeness"]
