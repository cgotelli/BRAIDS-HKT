"""Typed post-capacity channel composition for Delft3D FLOW 2DH.

The capacity operators used by BRAIDS return solid-mass transport per unit
width.  This module therefore starts *after* the one and only density
conversion.  It records the distinction between the formula's own split
(for example Engelund--Hansen ``SusFac``) and FLOW's later, independent
``BedFac``/``SusFac`` multipliers.

The reference behavior is the current-only ordering in ``eqtran.f90``:
bed-only steering, global multipliers, and then either folding the signed
suspended formula channel into a bedload-only fraction or retaining it for an
advection--diffusion fraction.  It is deliberately backend neutral and is not
yet connected to a production route.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence


CHANNEL_COMPOSITION_SCHEMA = 1
CHANNEL_COMPOSITION_SCOPE_ID = (
    "delft3d4-flow-2dh-post-density-channel-composition-v1"
)

# V52 is the first native-verifiable wire for this schema.  These are wire
# bounds, not physical limits: v51 remains available as the explicitly
# host-attested compatibility envelope for identifiers outside this bounded
# native representation.
CHANNEL_COMPOSITION_NATIVE_WIRE_SCHEMA = 2
CHANNEL_COMPOSITION_NATIVE_MAX_PREIMAGE_ASCII = 32768
CHANNEL_COMPOSITION_NATIVE_MAX_IDENTIFIER_TOKEN_ASCII = 1536
CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK = 128

# Locked sediment_basics_module transport bit mask.
TRA_NONE = 0
TRA_BEDLOAD = 1
TRA_ADVDIFF = 2
TRA_COMBINE = 3


class ChannelCompositionError(ValueError):
    """Raised before a channel can be silently reinterpreted."""


@dataclass(frozen=True)
class ChannelCompositionFraction:
    fraction_id: str
    transport_class: int


@dataclass(frozen=True)
class ChannelCompositionProfile:
    schema: int
    scope_id: str
    semantic_id: str
    bed_factor: float
    suspended_factor: float
    epsilon: float
    fractions: tuple[ChannelCompositionFraction, ...]
    fingerprint: str


@dataclass(frozen=True)
class ChannelCompositionResult:
    composed_bed_x: float
    composed_bed_y: float
    retained_suspended_x: float
    retained_suspended_y: float
    signed_suspended_rate: float
    equilibrium_concentration: float


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def channel_composition_semantic_document() -> dict:
    """Return the static ordering and unit contract hashed into every profile."""
    return {
        "schema": CHANNEL_COMPOSITION_SCHEMA,
        "scope_id": CHANNEL_COMPOSITION_SCOPE_ID,
        "input_units": "solid_mass_per_time_per_width",
        "density_conversion": "already_applied_exactly_once",
        "order": [
            "formula_capacity_and_formula_partition",
            "density_conversion_before_this_contract",
            "external_secondary_rotation_of_bed_only",
            "independent_global_bed_and_suspended_factors",
            "transport_class_fold_or_retention",
            "face_reconstruction_slope_availability_and_exner_after_this_contract",
        ],
        "transport_classes": {
            str(TRA_NONE): "no_transport_not_admitted_here",
            str(TRA_BEDLOAD): "fold_signed_suspended_channel_into_bed",
            str(TRA_ADVDIFF): "advdiff_only_owned_by_suspended_exchange_path",
            str(TRA_COMBINE): "retain_signed_suspended_channel_for_advdiff",
        },
        "direction": {
            "bed": "already_rotated_by_external_secondary_flow_when_selected",
            "suspended": "signed_projection_on_unrotated_depth_averaged_flow",
        },
        "equilibrium_concentration": (
            "mass concentration before the downstream active-bed fraction multiplier"
        ),
        "factor_names": {
            "bed_factor": "FLOW model-global Bed factor after formula ACal",
            "suspended_factor": (
                "FLOW model-global Sus factor after any formula-specific "
                "partition parameter"
            ),
        },
        "epsilon": {
            "role": "FLOW utot denominator regularization",
            "locked_reference_value": 1.0e-6,
            "profile_binding": "positive finite runtime value",
        },
        "signed_channels": True,
        "supported_in_this_composer": [TRA_BEDLOAD, TRA_COMBINE],
    }


def channel_composition_semantic_id() -> str:
    return hashlib.sha256(
        _canonical_json(channel_composition_semantic_document()).encode("ascii")
    ).hexdigest()


def _finite_scalar(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ChannelCompositionError(f"{name} must be finite numeric data")
    return float(value)


def resolve_channel_composition_profile(
    fraction_ids: Sequence[str],
    transport_classes: Sequence[int],
    *,
    bed_factor: float = 1.0,
    suspended_factor: float = 1.0,
    epsilon: float = 1.0e-6,
) -> ChannelCompositionProfile:
    """Resolve an ordered, restart-relevant composition profile.

    ``TRA_ADVDIFF`` is a real Delft3D transport class but it bypasses the
    algebraic-capacity composer.  It remains defined in the semantic document
    and fails closed here until the Q4 suspended-only path owns it.
    """
    if not isinstance(fraction_ids, Sequence) or isinstance(
        fraction_ids, (str, bytes, bytearray)
    ):
        raise ChannelCompositionError("fraction_ids must be an ordered sequence")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ChannelCompositionError("epsilon must be positive finite numeric data")
    epsilon = float(epsilon)
    ids = tuple(fraction_ids)
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ChannelCompositionError("fraction IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ChannelCompositionError("fraction IDs must be unique")
    if not isinstance(transport_classes, Sequence) or isinstance(
        transport_classes, (str, bytes, bytearray)
    ) or len(transport_classes) != len(ids):
        raise ChannelCompositionError(
            "transport_classes must contain one exact integer per fraction"
        )
    classes: list[int] = []
    for value in transport_classes:
        if type(value) is not int:
            raise ChannelCompositionError("transport classes are exact integer wire values")
        if value not in (TRA_BEDLOAD, TRA_COMBINE):
            if value == TRA_ADVDIFF:
                raise ChannelCompositionError(
                    "TRA_ADVDIFF bypasses algebraic channel composition and belongs to Q4"
                )
            raise ChannelCompositionError(f"unsupported transport class: {value}")
        classes.append(value)
    bed_factor = _finite_scalar("bed_factor", bed_factor)
    suspended_factor = _finite_scalar("suspended_factor", suspended_factor)
    fractions = tuple(
        ChannelCompositionFraction(identifier, transport_class)
        for identifier, transport_class in zip(ids, classes, strict=True)
    )
    payload = {
        "schema": CHANNEL_COMPOSITION_SCHEMA,
        "scope_id": CHANNEL_COMPOSITION_SCOPE_ID,
        "semantic_id": channel_composition_semantic_id(),
        "bed_factor": bed_factor,
        "suspended_factor": suspended_factor,
        "epsilon": epsilon,
        "fractions": [asdict(value) for value in fractions],
    }
    fingerprint = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return ChannelCompositionProfile(
        CHANNEL_COMPOSITION_SCHEMA,
        CHANNEL_COMPOSITION_SCOPE_ID,
        channel_composition_semantic_id(),
        bed_factor,
        suspended_factor,
        epsilon,
        fractions,
        fingerprint,
    )


def channel_composition_profile_document(profile: ChannelCompositionProfile) -> dict:
    if not isinstance(profile, ChannelCompositionProfile):
        raise ChannelCompositionError("expected a ChannelCompositionProfile")
    if type(profile.schema) is not int or profile.schema != CHANNEL_COMPOSITION_SCHEMA:
        raise ChannelCompositionError("channel composition schema must be the exact integer wire value")
    rebuilt = resolve_channel_composition_profile(
        tuple(value.fraction_id for value in profile.fractions),
        tuple(value.transport_class for value in profile.fractions),
        bed_factor=profile.bed_factor,
        suspended_factor=profile.suspended_factor,
        epsilon=profile.epsilon,
    )
    if profile != rebuilt:
        raise ChannelCompositionError("channel composition profile identity is stale or forged")
    return {
        "schema": profile.schema,
        "scope_id": profile.scope_id,
        "semantic_id": profile.semantic_id,
        "bed_factor": profile.bed_factor,
        "suspended_factor": profile.suspended_factor,
        "epsilon": profile.epsilon,
        "fractions": [asdict(value) for value in profile.fractions],
        "fingerprint": profile.fingerprint,
    }


def channel_composition_profile_preimage(profile: ChannelCompositionProfile) -> str:
    """Return the exact canonical ASCII JSON bytes hashed by ``fingerprint``.

    The fingerprint is deliberately excluded from its own preimage.  This
    public helper is also the single managed-host source for the v52 payload;
    native code receives these exact bytes, independently hashes them, and
    reconstructs them from the separately serialized operational fields.
    """
    document = channel_composition_profile_document(profile)
    payload = {key: value for key, value in document.items() if key != "fingerprint"}
    preimage = _canonical_json(payload)
    if not preimage.isascii():  # ``ensure_ascii=True`` makes this an invariant.
        raise ChannelCompositionError("channel composition preimage must be ASCII")
    if hashlib.sha256(preimage.encode("ascii")).hexdigest() != profile.fingerprint:
        raise ChannelCompositionError("channel composition preimage fingerprint differs")
    return preimage


def channel_composition_native_wire_document(
    profile: ChannelCompositionProfile,
) -> dict:
    """Build the bounded, self-verifying v52 native identity payload.

    Fraction identifiers are carried as their exact canonical JSON string
    tokens.  That keeps arbitrary Unicode identifiers representable in an
    ASCII-only Fortran deck while preserving both order and byte identity.
    """
    preimage = channel_composition_profile_preimage(profile)
    identifier_tokens = tuple(
        _canonical_json(value.fraction_id) for value in profile.fractions
    )
    if len(preimage) > CHANNEL_COMPOSITION_NATIVE_MAX_PREIMAGE_ASCII:
        raise ChannelCompositionError(
            "channel composition canonical preimage exceeds the native v52 bound"
        )
    if any(
        len(token) > CHANNEL_COMPOSITION_NATIVE_MAX_IDENTIFIER_TOKEN_ASCII
        for token in identifier_tokens
    ):
        raise ChannelCompositionError(
            "channel composition fraction identifier exceeds the native v52 bound"
        )
    return {
        "wire_schema": CHANNEL_COMPOSITION_NATIVE_WIRE_SCHEMA,
        "preimage_ascii": preimage,
        "preimage_sha256": profile.fingerprint,
        "numeric_tokens": {
            "bed_factor": _canonical_json(profile.bed_factor),
            "suspended_factor": _canonical_json(profile.suspended_factor),
            "epsilon": _canonical_json(profile.epsilon),
        },
        "ordered_fraction_identifier_tokens": identifier_tokens,
        "ordered_transport_classes": tuple(
            value.transport_class for value in profile.fractions
        ),
    }


def load_channel_composition_profile(document: Mapping[str, object]) -> ChannelCompositionProfile:
    if not isinstance(document, Mapping):
        raise ChannelCompositionError("channel composition profile must be a mapping")
    expected = {
        "schema", "scope_id", "semantic_id", "bed_factor",
        "suspended_factor", "epsilon", "fractions", "fingerprint"
    }
    if set(document) != expected:
        raise ChannelCompositionError("channel composition profile fields differ")
    if (
        type(document["schema"]) is not int
        or document["schema"] != CHANNEL_COMPOSITION_SCHEMA
    ):
        raise ChannelCompositionError(
            "channel composition schema must be the exact integer wire value"
        )
    rows = document["fractions"]
    if not isinstance(rows, list) or not rows:
        raise ChannelCompositionError("channel composition fractions must be a nonempty list")
    if any(not isinstance(row, Mapping) or set(row) != {
        "fraction_id", "transport_class"
    } for row in rows):
        raise ChannelCompositionError("invalid channel composition fraction row")
    rebuilt = resolve_channel_composition_profile(
        tuple(row["fraction_id"] for row in rows),
        tuple(row["transport_class"] for row in rows),
        bed_factor=document["bed_factor"],
        suspended_factor=document["suspended_factor"],
        epsilon=document["epsilon"],
    )
    if channel_composition_profile_document(rebuilt) != dict(document):
        raise ChannelCompositionError("channel composition profile identity differs")
    return rebuilt


def compose_mass_channels(
    *,
    depth: float,
    velocity_x: float,
    velocity_y: float,
    raw_bed_x: float,
    raw_bed_y: float,
    raw_suspended_x: float,
    raw_suspended_y: float,
    transport_class: int,
    bed_factor: float = 1.0,
    suspended_factor: float = 1.0,
    epsilon: float = 1.0e-6,
    alignment_rtol: float = 1.0e-12,
    alignment_atol: float = 1.0e-15,
) -> ChannelCompositionResult:
    """Compose one already-mass-valued scalar-capacity channel pair.

    The suspended vector must be collinear with the unrotated depth-averaged
    velocity.  Its dot product, not its norm, carries the physical sign.
    """
    if type(transport_class) is not int or transport_class not in (
        TRA_BEDLOAD, TRA_COMBINE
    ):
        if transport_class == TRA_ADVDIFF:
            raise ChannelCompositionError(
                "TRA_ADVDIFF bypasses this algebraic-capacity composer"
            )
        raise ChannelCompositionError(f"unsupported transport class: {transport_class!r}")
    values = (
        depth, velocity_x, velocity_y, raw_bed_x, raw_bed_y,
        raw_suspended_x, raw_suspended_y, bed_factor, suspended_factor,
        epsilon, alignment_rtol, alignment_atol,
    )
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in values) or not all(math.isfinite(float(value)) for value in values):
        raise ChannelCompositionError("channel composition inputs must be finite numeric values")
    if epsilon <= 0.0 or min(alignment_rtol, alignment_atol) < 0.0:
        raise ChannelCompositionError("invalid epsilon or alignment tolerance")
    speed = math.hypot(velocity_x, velocity_y)
    if transport_class == TRA_COMBINE and depth <= 0.0:
        raise ChannelCompositionError(
            "combined suspended transport requires positive water depth"
        )
    if speed == 0.0:
        if raw_suspended_x != 0.0 or raw_suspended_y != 0.0:
            raise ChannelCompositionError(
                "zero-flow state cannot direct a nonzero scalar suspended channel"
            )
        return ChannelCompositionResult(
            bed_factor * raw_bed_x,
            bed_factor * raw_bed_y,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    unit_x = velocity_x / speed
    unit_y = velocity_y / speed
    signed_raw_suspended = raw_suspended_x * unit_x + raw_suspended_y * unit_y
    cross_x = raw_suspended_x - signed_raw_suspended * unit_x
    cross_y = raw_suspended_y - signed_raw_suspended * unit_y
    suspended_norm = math.hypot(raw_suspended_x, raw_suspended_y)
    if math.hypot(cross_x, cross_y) > alignment_atol + alignment_rtol * suspended_norm:
        raise ChannelCompositionError(
            "scalar suspended channel is not aligned with depth-averaged flow"
        )

    bed_x = bed_factor * raw_bed_x
    bed_y = bed_factor * raw_bed_y
    signed_suspended = suspended_factor * signed_raw_suspended
    suspended_x = signed_suspended * unit_x
    suspended_y = signed_suspended * unit_y
    if transport_class == TRA_BEDLOAD:
        result = ChannelCompositionResult(
            bed_x + suspended_x,
            bed_y + suspended_y,
            0.0,
            0.0,
            signed_suspended,
            0.0,
        )
    else:
        result = ChannelCompositionResult(
            bed_x,
            bed_y,
            suspended_x,
            suspended_y,
            signed_suspended,
            signed_suspended / (speed + epsilon) / depth,
        )
    if not all(math.isfinite(value) for value in asdict(result).values()):
        raise ChannelCompositionError("channel composition produced a nonfinite result")
    return result


__all__ = [
    "CHANNEL_COMPOSITION_SCHEMA",
    "CHANNEL_COMPOSITION_SCOPE_ID",
    "CHANNEL_COMPOSITION_NATIVE_WIRE_SCHEMA",
    "CHANNEL_COMPOSITION_NATIVE_MAX_PREIMAGE_ASCII",
    "CHANNEL_COMPOSITION_NATIVE_MAX_IDENTIFIER_TOKEN_ASCII",
    "CHANNEL_COMPOSITION_NATIVE_HEX_CHUNK",
    "TRA_NONE",
    "TRA_BEDLOAD",
    "TRA_ADVDIFF",
    "TRA_COMBINE",
    "ChannelCompositionError",
    "ChannelCompositionFraction",
    "ChannelCompositionProfile",
    "ChannelCompositionResult",
    "channel_composition_semantic_document",
    "channel_composition_semantic_id",
    "resolve_channel_composition_profile",
    "channel_composition_profile_document",
    "channel_composition_profile_preimage",
    "channel_composition_native_wire_document",
    "load_channel_composition_profile",
    "compose_mass_channels",
]
