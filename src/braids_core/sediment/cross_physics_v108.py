"""Q9I cross-physics transaction, timestep and output ownership closure.

This module does not introduce another sediment formula.  It composes the
already accepted component owners and makes their publication atomic.  The
same orchestration is used by the native CPU and resident CUDA routes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .cohesive_banks_v107 import remap_cohesive_bank_owner_v107
from .dynamic_bedforms_v104 import remap_dynamic_bedform_owner_v104


SCOPE = "braids-q9i-cross-physics-v108"
INTEGRATOR = "implicit_fwave_midpoint"
LIMITER_ORDER = (
    "hydraulic_cfl", "scalar_fct", "explicit_diffusion", "exner_morfac",
    "bedform_cfl", "avalanche", "bank", "fine_exchange",
)

COMPONENT_OWNERSHIP = {
    "boundary": (
        "boundary_bed_change", "boundary_composition", "point_load_mass",
        "source_sink_ledger", "boundary_event_cursor",
    ),
    "bedform": (
        "bedform_height", "bedform_length", "bedform_clock",
        "hydraulic_roughness_feedback", "transport_roughness_feedback",
    ),
    "bank": (
        "cohesive_toe_erosion", "bank_undercut", "bank_collapse",
        "bank_release", "bed_layers", "suspended_fines", "fluff",
        "consolidation", "porosity", "parent_owned_avalanche",
    ),
    "outputs": ("result_snapshot", "output_checkpoint"),
}

AMR_OWNERSHIP = {
    "water_and_momentum": "inherited_q4_v98_parent_reflux_owner",
    "suspended_classes": "inherited_q4_v98_parent_reflux_owner",
    "active_bed_and_exner": "inherited_q4_q5_parent_reflux_owner",
    "secondary_flow": "inherited_q3h_signed_parent_reflux_owner",
    "bed_layers_bank_and_fines": "q8b_v107_parent_conservative_remap",
    "dynamic_bedforms": "q7b_v104_parent_conservative_remap",
    "sediment_boundary": "fixed_grid_only",
}

BUNDLES = {
    "layered_sorted_dynamic_bedforms": (
        "columns", "dependent_layers", "sorting", "dynamic_bedforms"),
    "dune_scaled_layer_bedform_clock": (
        "columns", "dune_scaled_layer", "dynamic_bedforms", "morfac_clock"),
    "santoss_static_q4_global_slope": (
        "santoss", "static_roughness", "suspension_q4", "global_slope"),
    "cohesive_consolidation_porosity_fluff": (
        "cohesive_inventory", "consolidation", "porosity", "fluff"),
    "boundary_fraction_reversal_restart": (
        "sediment_boundary", "fraction_basis", "reversal", "restart"),
    "bank_layers_suspended_fines": (
        "cohesive_bank", "bed_layers", "suspended_fines", "avalanche"),
}


def _host(value):
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _normal(value):
    if isinstance(value, Mapping):
        return {str(k): _normal(value[k]) for k in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normal(v) for v in value]
    if isinstance(value, np.ndarray) or hasattr(value, "get"):
        a = np.asarray(_host(value))
        return {"dtype": str(a.dtype), "shape": list(a.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(_normal(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class CrossPhysicsProfileV108:
    bundle: str
    backend: str
    integrator: str = INTEGRATOR

    def __post_init__(self):
        if self.bundle not in BUNDLES:
            raise ValueError("Unknown Q9I cross-physics bundle")
        if self.backend not in ("cpu", "implicit_fwave_midpoint_gpu"):
            raise ValueError("Backend must be the explicit user CPU/GPU choice")
        if self.integrator != INTEGRATOR:
            raise ValueError("Q9I GPU composition requires implicit_fwave_midpoint")

    @property
    def components(self):
        return BUNDLES[self.bundle]

    @property
    def fingerprint(self):
        return digest({"scope": SCOPE, **asdict(self), "components": self.components})


def resolve_cross_physics_profile_v108(bundle, *, backend, integrator=INTEGRATOR):
    """Resolve one named compatible combination; no independent free-form menu."""
    return CrossPhysicsProfileV108(str(bundle), str(backend), str(integrator))


@dataclass(frozen=True)
class TimestepLimitsV108:
    hydraulic_cfl: float = math.inf
    scalar_fct: float = math.inf
    explicit_diffusion: float = math.inf
    exner_morfac: float = math.inf
    bedform_cfl: float = math.inf
    avalanche: float = math.inf
    bank: float = math.inf
    fine_exchange: float = math.inf

    def __post_init__(self):
        for name in LIMITER_ORDER:
            value = float(getattr(self, name))
            if math.isnan(value) or value <= 0:
                raise ValueError(f"Invalid {name} timestep bound")

    def rows(self):
        return tuple((name, float(getattr(self, name))) for name in LIMITER_ORDER)


class TimestepArbiterV108:
    """Deterministic global reduction; ``xp`` is NumPy or resident CuPy."""

    def __init__(self, xp=np, *, backend="cpu"):
        self.xp = xp
        self.backend = backend
        if backend not in ("cpu", "implicit_fwave_midpoint_gpu"):
            raise ValueError("Unknown timestep-arbiter backend")
        if backend == "implicit_fwave_midpoint_gpu" and xp is np:
            raise ValueError("GPU timestep arbitration may not fall back to NumPy")
        self.identity = digest({"scope": SCOPE, "role": "timestep_arbiter",
                                "backend": backend, "order": LIMITER_ORDER})

    def arbitrate(self, requested_dt, limits, *, time_s, end_time_s,
                  next_event_s=math.inf):
        values = [float(requested_dt), float(end_time_s)-float(time_s)]
        names = ["requested", "end_time"]
        if math.isfinite(next_event_s):
            values.append(float(next_event_s)-float(time_s)); names.append("schedule_event")
        values.extend(value for _, value in limits.rows())
        names.extend(name for name, _ in limits.rows())
        if not math.isfinite(float(requested_dt)) or requested_dt <= 0 or \
                not math.isfinite(float(time_s)) or not math.isfinite(float(end_time_s)) or \
                not 0 <= time_s < end_time_s or any(math.isnan(v) or v <= 0 for v in values):
            raise ValueError("Invalid Q9I timestep arbitration interval")
        device_values = self.xp.asarray(values, dtype=self.xp.float64)
        index = int(_host(self.xp.argmin(device_values)))
        selected = float(_host(device_values[index]))
        receipt = {
            "schema": "braids.timestep_arbiter.v108", "arbiter": self.identity,
            "backend": self.backend, "dt_s": selected, "active_limiter": names[index],
            "candidates": [{"name": n, "dt_s": v, "active": i == index}
                           for i, (n, v) in enumerate(zip(names, values))],
        }
        receipt["sha256"] = digest(receipt)
        return receipt


def _output_entries():
    path = Path(__file__).with_name("flow2dh_scope_overlay_v1.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in data["added_entries"] if row.get("family") == "morphology_output"]
    if len(rows) != 69 or len({row["entry_id"] for row in rows}) != 69:
        raise RuntimeError("Q9I requires the complete 69-selector output inventory")
    return tuple(sorted(rows, key=lambda row: row["entry_id"]))


OUTPUT_EXCLUSIONS = {
    "output.bed_transp_due_to_waves_at_flux": "wave-current physics is outside the 2DH river profile",
    "output.bed_transp_due_to_waves_at_zeta": "wave-current physics is outside the 2DH river profile",
    "output.orbital_vel": "wave-current physics is outside the 2DH river profile",
    "output.susp_transp_due_to_waves_at_flux": "wave-current physics is outside the 2DH river profile",
    "output.susp_transp_due_to_waves_at_zeta": "wave-current physics is outside the 2DH river profile",
    "output.mor_stats_output_interval": "D-Flow FM-only selector is not a Delft3D4 FLOW output",
    "output.main_channel_averaged_bed_level": "classic FLOW has no output consumer/writer",
}


def output_registry_v108():
    mappings = []
    for row in _output_entries():
        entry = row["entry_id"]
        supported = entry not in OUTPUT_EXCLUSIONS
        mappings.append({
            "entry_id": entry, "keyword": row["selector"]["keyword"],
            "result_field_id": row["result_field_id"],
            "checkpoint_field_id": row["checkpoint_field_id"],
            "producer_stage": row["owner"]["milestone"],
            "destination": row["value_schema"]["destination"],
            "source_default": row["value_schema"]["source_default"],
            "value_type": row["value_schema"]["type"],
            "allowed": row["value_schema"].get("allowed"),
            "source_default_effect": row["value_schema"].get("source_default_effect"),
            "alias_of": row.get("applicability", {}).get("alias_of"),
            "supported": supported,
            "disposition": "wired" if supported else "explicitly_incompatible",
            "reason": None if supported else OUTPUT_EXCLUSIONS[entry],
        })
    document = {"schema": "braids.morphology_output_registry.v108",
                "scope": SCOPE, "mapping_count": len(mappings),
                "logical_destination_count": len({row["destination"] for row in mappings}),
                "mappings": mappings}
    document["identity"] = digest(document)
    return document


def _validated_output_value(row, value):
    kind = row["value_type"]
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{row['keyword']} requires a boolean")
        return value
    if kind == "integer_enum":
        if type(value) is not int or value not in row["allowed"]:
            raise ValueError(f"{row['keyword']} requires one of {row['allowed']}")
        return value
    if kind == "number_vector":
        a = np.asarray(value, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError(f"{row['keyword']} requires three finite values")
        return a.tolist()
    if kind == "string_selector":
        if type(value) is not str:
            raise ValueError(f"{row['keyword']} requires a string selector")
        return value
    if kind == "boolean_or_percentile_list":
        if type(value) is bool:
            return value
        a = np.asarray(value, dtype=float)
        if a.ndim != 1 or a.size < 1 or not np.isfinite(a).all() or \
                np.any((a < 0) | (a > 100)) or np.any(np.diff(a) <= 0):
            raise ValueError(f"{row['keyword']} requires increasing percentiles in [0,100]")
        return a.tolist()
    raise ValueError(f"Unknown output value type for {row['keyword']}")


def _output_value_is_active(row, value):
    if row["value_type"] in ("boolean", "boolean_or_percentile_list") and type(value) is bool:
        return value
    if row["value_type"] == "string_selector":
        # FLOW initializes MorstatsWeightFactor numerically even though the
        # optional parser override is a string selector.
        return bool(str(value).strip())
    return value is not None


def resolve_output_selection_v108(selection, *, apply_source_defaults=False):
    """Resolve typed FLOW output settings and reject incompatibilities early."""
    registry = output_registry_v108()
    by_id = {row["entry_id"]: row for row in registry["mappings"]}
    by_keyword = {row["keyword"].casefold(): row for row in registry["mappings"]}
    if type(apply_source_defaults) is not bool:
        raise ValueError("apply_source_defaults must be an explicit boolean")
    explicit = []
    for key, value in selection.items():
        row = by_id.get(str(key), by_keyword.get(str(key).casefold()))
        if row is None:
            raise ValueError(f"Unknown morphology output selector: {key}")
        explicit.append((row, _validated_output_value(row, value)))
    meta = [value for row, value in explicit if row["entry_id"] in
            ("output.default", "output.out_default")]
    if len(set(meta)) > 1:
        raise ValueError("Default and OutDefault bulk settings conflict")
    configured = {}
    if apply_source_defaults:
        for row in registry["mappings"]:
            if row["supported"] and row["entry_id"] not in ("output.default", "output.out_default"):
                configured.setdefault(row["destination"], (row, row["source_default"]))
    if meta:
        # Profile-scoped bulk override: explicit incompatibilities remain
        # unavailable but do not make the useful FLOW default control unusable.
        for row in registry["mappings"]:
            if row["supported"] and row["value_type"] == "boolean" and \
                    row["entry_id"] not in ("output.default", "output.out_default"):
                configured[row["destination"]] = (row, meta[0])
    settings = []
    explicit_destinations = {}
    for row, value in explicit:
        settings.append({"entry_id": row["entry_id"], "value": value})
        if row["entry_id"] in ("output.default", "output.out_default"):
            continue
        previous = explicit_destinations.get(row["destination"])
        if previous is not None and previous[0]["entry_id"] != row["entry_id"] and previous[1] != value:
            raise ValueError(f"Conflicting aliases for {row['destination']}")
        explicit_destinations[row["destination"]] = (row, value)
        configured[row["destination"]] = (row, value)
    resolved = []
    for row, value in configured.values():
        if not _output_value_is_active(row, value):
            continue
        if not row["supported"]:
            raise ValueError(f"{row['keyword']}: {row['reason']}")
        resolved.append({**row, "configured_value": value})
    # Meta selectors stay in settings/provenance and do not invent result fields.
    unique = {row["entry_id"]: row for row in resolved}
    document = {"schema": "braids.morphology_output_selection.v108",
                "registry": registry["identity"],
                "apply_source_defaults": apply_source_defaults,
                "settings": sorted(settings, key=lambda row: row["entry_id"]),
                "mappings": [unique[k] for k in sorted(unique)]}
    document["identity"] = digest(document)
    return document


class MorphologyOutputOwnerV108:
    def __init__(self, selection):
        self.selection = selection
        self.values = {}
        self.updates = 0
        self.pending = None
        self.identity = digest({"scope": SCOPE, "role": "outputs",
                                "selection": selection["identity"]})

    def propose(self, fields):
        if self.pending is not None:
            raise RuntimeError("Output proposal already pending")
        missing = [row["result_field_id"] for row in self.selection["mappings"]
                   if row["result_field_id"] not in fields]
        if missing:
            raise ValueError("Missing selected result fields: " + ", ".join(missing))
        # Result/checkpoint ids can intentionally alias (for example the two
        # preload spellings).  Entry ids preserve all requested selectors;
        # the selection document retains both canonical destinations.
        self.pending = {row["entry_id"]: np.asarray(
                            _host(fields[row["result_field_id"]])).copy()
                        for row in self.selection["mappings"]}
        return self.pending

    def commit(self, *, cancelled=False):
        if self.pending is None:
            raise RuntimeError("No output proposal")
        trial = self.pending; self.pending = None
        if cancelled:
            raise InterruptedError("Output publication cancelled")
        self.values = trial; self.updates += 1

    def rollback(self):
        self.pending = None

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("Cannot checkpoint pending output")
        body = {"schema": "braids.morphology_output.v108", "identity": self.identity,
                "updates": self.updates, "values": {k: v.copy() for k, v in self.values.items()}}
        body["sha256"] = digest(body)
        return body

    def restore(self, document):
        raw = {k: v for k, v in document.items() if k != "sha256"}
        if digest(raw) != document.get("sha256") or document.get("identity") != self.identity:
            raise ValueError("Output checkpoint identity differs")
        self.values = {k: np.asarray(_host(v)).copy() for k, v in document["values"].items()}
        self.updates = int(document["updates"]); self.pending = None


def _component_backend(owner):
    backend = getattr(getattr(owner, "runtime", None), "backend", "")
    return "implicit_fwave_midpoint_gpu" if backend == "implicit_fwave_midpoint_gpu" else "cpu"


class CrossPhysicsOwnerV108:
    """Atomic publisher for boundary, bedform, bank and output owners."""

    def __init__(self, profile, *, boundary=None, bedform=None, bank=None, outputs=None,
                 time_s=0., updates=0, arbiter=None):
        self.profile = profile
        self.components = {k: v for k, v in (("boundary", boundary), ("bedform", bedform),
                                              ("bank", bank), ("outputs", outputs)) if v is not None}
        if not self.components:
            raise ValueError("Cross-physics transaction has no component owners")
        fields = [field for name in self.components for field in COMPONENT_OWNERSHIP[name]]
        if len(fields) != len(set(fields)):
            raise ValueError("Cross-physics fields have duplicate owners")
        for name, owner in self.components.items():
            if name != "outputs" and _component_backend(owner) != profile.backend:
                raise ValueError(f"{name} owner backend differs from explicit profile choice")
        self.arbiter = arbiter or TimestepArbiterV108(backend=profile.backend)
        if self.arbiter.backend != profile.backend:
            raise ValueError("Arbiter backend differs from explicit profile choice")
        self.time_s = float(time_s); self.updates = int(updates); self.pending = None
        for name, owner in self.components.items():
            if name != "outputs" and abs(float(owner.time_s)-self.time_s) > 1e-13:
                raise ValueError(f"{name} accepted clock differs from cross-physics clock")
        self.identity = digest({"scope": SCOPE, "profile": profile.fingerprint,
            "components": {k: getattr(getattr(v, "runtime", v), "identity", getattr(v, "identity", ""))
                           for k, v in self.components.items()}, "ownership": COMPONENT_OWNERSHIP})

    @staticmethod
    def _cancelled(cancelled):
        return bool(cancelled() if callable(cancelled) else cancelled)

    def _rollback_trials(self):
        for name, owner in self.components.items():
            if name == "bank": owner.discard()
            else: owner.rollback()

    def begin(self, *, requested_dt, end_time_s, limits=TimestepLimitsV108(),
              forcings=None, output_fields=None, next_event_s=math.inf, cancelled=False):
        if self.pending is not None:
            raise RuntimeError("Cross-physics proposal already pending")
        if self.updates >= 20:
            raise ValueError("Q9I bounded transaction permits at most 20 accepted updates")
        for name, owner in self.components.items():
            if name != "outputs" and abs(float(owner.time_s)-self.time_s) > 1e-13:
                raise ValueError(f"{name} accepted clock differs from cross-physics clock")
        receipt = self.arbiter.arbitrate(requested_dt, limits, time_s=self.time_s,
                                         end_time_s=end_time_s, next_event_s=next_event_s)
        dt = receipt["dt_s"]; forcings = {} if forcings is None else dict(forcings)
        pre = {name: owner.checkpoint() for name, owner in self.components.items()}
        proposed = {}
        try:
            for name in ("boundary", "bedform", "bank"):
                if name not in self.components:
                    continue
                if self._cancelled(cancelled):
                    raise InterruptedError(f"Q9I cancelled before {name}")
                args = dict(forcings.get(name, {}))
                if "dt" in args or "dt_s" in args:
                    raise ValueError(f"Q9I owns the {name} timestep")
                if name == "boundary":
                    proposed[name] = self.components[name].begin(dt, **args)
                else:
                    proposed[name] = self.components[name].propose(dt_s=dt, **args)
            if "outputs" in self.components:
                if output_fields is None:
                    raise ValueError("Selected outputs require canonical result fields")
                proposed["outputs"] = self.components["outputs"].propose(output_fields)
            if self._cancelled(cancelled):
                raise InterruptedError("Q9I cancelled after proposals")
        except BaseException:
            self._rollback_trials()
            raise
        self.pending = {"receipt": receipt, "pre": pre, "proposed": proposed}
        self.pending["sha256"] = digest({"receipt": receipt,
            "pre": {k: v.get("sha256", v.get("checkpoint_sha256", v.get("state_sha256")))
                    for k, v in pre.items()}, "components": tuple(proposed)})
        return receipt

    def commit(self, *, cancelled=False):
        if self.pending is None:
            raise RuntimeError("No cross-physics proposal")
        pending = self.pending
        if self._cancelled(cancelled):
            self._rollback_trials(); self.pending = None
            raise InterruptedError("Q9I cancelled before atomic publication")
        committed = []
        try:
            for name in ("boundary", "bedform", "bank", "outputs"):
                if name not in self.components:
                    continue
                self.components[name].commit()
                committed.append(name)
        except BaseException:
            for name, owner in self.components.items():
                try:
                    if name not in committed:
                        if name == "bank": owner.discard()
                        else: owner.rollback()
                    owner.restore(pending["pre"][name])
                except BaseException:
                    pass
            self.pending = None
            raise
        self.time_s += pending["receipt"]["dt_s"]; self.updates += 1; self.pending = None
        return pending["receipt"]

    def rollback(self):
        if self.pending is not None:
            self._rollback_trials(); self.pending = None

    def checkpoint(self):
        if self.pending is not None:
            raise RuntimeError("Cannot checkpoint pending cross-physics state")
        body = {"schema": "braids.cross_physics.v108", "identity": self.identity,
                "profile": self.profile.fingerprint, "time_s": self.time_s,
                "updates": self.updates,
                "components": {name: owner.checkpoint() for name, owner in self.components.items()}}
        body["sha256"] = digest(body)
        return body

    def restore(self, document):
        raw = {k: v for k, v in document.items() if k != "sha256"}
        if digest(raw) != document.get("sha256") or document.get("identity") != self.identity or \
                set(document.get("components", {})) != set(self.components):
            raise ValueError("Cross-physics checkpoint identity differs")
        old = self.checkpoint()
        try:
            for name, owner in self.components.items():
                owner.restore(document["components"][name])
            self.time_s = float(document["time_s"]); self.updates = int(document["updates"])
            if not 0 <= self.updates <= 20 or not math.isfinite(self.time_s) or self.time_s < 0:
                raise ValueError("Invalid cross-physics restart clock")
            for name, owner in self.components.items():
                if name != "outputs" and abs(float(owner.time_s)-self.time_s) > 1e-13:
                    raise ValueError(f"{name} restart clock differs from cross-physics clock")
        except BaseException:
            for name, owner in self.components.items():
                owner.restore(old["components"][name])
            self.time_s = old["time_s"]; self.updates = old["updates"]
            raise


@dataclass
class CrossPhysicsPatchV108:
    patch_id: str
    level: int
    bank: object
    bedform: object | None = None
    boundary: object | None = None

    def __post_init__(self):
        if not self.patch_id or type(self.level) is not int or self.level < 1:
            raise ValueError("Invalid Q9I patch identity")
        if self.bedform is not None and _component_backend(self.bank) != _component_backend(self.bedform):
            raise ValueError("Patch components must share a backend")


class CrossPhysicsHierarchyV108:
    """Parent-owned, bounded whole-patch remap; boundary state is fixed-grid-only."""

    def __init__(self, patches):
        patches = tuple(patches)
        self.patches = {patch.patch_id: patch for patch in patches}
        if len(self.patches) != len(patches) or len(self.patches) < 2:
            raise ValueError("Q9I hierarchy requires multiple uniquely owned patches")
        self.remaps = 0

    def remap(self, patch_id, *, target_level, bank_runtime, bedform_runtime=None,
              offsets, sources, weights, target_area, bank_mask, toe_target,
              bank_height_m, bank_face_length_m):
        if self.remaps >= 8:
            raise ValueError("Q9I bounded hierarchy permits at most eight remaps")
        old = self.patches[patch_id]
        if old.boundary is not None:
            raise ValueError("Sediment-boundary ownership is fixed-grid-only")
        if bank_runtime.identity != old.bank.runtime.identity:
            raise ValueError("Bank AMR may not change the accepted runtime identity")
        bank = remap_cohesive_bank_owner_v107(old.bank, offsets, sources, weights,
            target_area, bank_mask=bank_mask, toe_target=toe_target,
            bank_height_m=bank_height_m, bank_face_length_m=bank_face_length_m)
        bedform = None
        if old.bedform is not None:
            if bedform_runtime is None:
                raise ValueError("Dynamic-bedform AMR requires its target runtime")
            bedform = remap_dynamic_bedform_owner_v104(old.bedform, bedform_runtime,
                offsets, sources, weights, target_area)
        new = CrossPhysicsPatchV108(old.patch_id, int(target_level), bank, bedform)
        self.patches[patch_id] = new; self.remaps += 1
        return new


def cross_physics_catalogue_v108():
    output = output_registry_v108()
    document = {"schema": "braids.cross_physics.catalogue.v108", "scope": SCOPE,
        "integrator": INTEGRATOR, "bundles": {k: list(v) for k, v in BUNDLES.items()},
        "limiter_order": list(LIMITER_ORDER), "ownership": COMPONENT_OWNERSHIP,
        "output_registry": output["identity"], "output_mapping_count": output["mapping_count"],
        "amr": AMR_OWNERSHIP, "qgis_admission": "deferred_to_q10"}
    document["identity"] = digest(document)
    return document


__all__ = [
    "AMR_OWNERSHIP", "BUNDLES", "COMPONENT_OWNERSHIP", "CrossPhysicsHierarchyV108",
    "CrossPhysicsOwnerV108", "CrossPhysicsPatchV108", "CrossPhysicsProfileV108",
    "INTEGRATOR", "LIMITER_ORDER", "MorphologyOutputOwnerV108",
    "TimestepArbiterV108", "TimestepLimitsV108", "cross_physics_catalogue_v108",
    "output_registry_v108", "resolve_cross_physics_profile_v108",
    "resolve_output_selection_v108",
]
