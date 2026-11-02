"""Per-ID suspended-sediment boundary data and conservative face fluxes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .passive import TransportError


@dataclass(frozen=True)
class SedimentBoundarySpec:
    """A concentration or total dry-mass-load series for one hydraulic ID."""

    boundary_id: str
    mode: str
    times_s: tuple[float, ...]
    values: tuple[float, ...]
    endpoint: str = "hold"

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id.strip():
            raise TransportError("sediment boundary_id must be a nonempty string")
        if self.mode not in {"concentration", "mass_load"}:
            raise TransportError("sediment mode must be concentration or mass_load")
        if self.endpoint not in {"hold", "zero", "error"}:
            raise TransportError("sediment endpoint must be hold, zero, or error")
        times = np.asarray(self.times_s, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if times.ndim != 1 or len(times) < 1 or values.shape != times.shape:
            raise TransportError("sediment times and values must be same-sized nonempty series")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise TransportError("sediment boundary series must be finite")
        if np.any(np.diff(times) <= 0.0) or np.any(values < 0.0):
            raise TransportError("sediment times must increase and values must be non-negative")

    def value_at(self, time_s: float) -> float:
        if not math.isfinite(time_s):
            raise TransportError("boundary evaluation time must be finite")
        times = np.asarray(self.times_s)
        values = np.asarray(self.values)
        if time_s < times[0] or time_s > times[-1]:
            if self.endpoint == "zero":
                return 0.0
            if self.endpoint == "error":
                raise TransportError("time outside sediment boundary series")
            return float(values[0] if time_s < times[0] else values[-1])
        return float(np.interp(time_s, times, values))


def sediment_face_mass_rates(
    spec: SedimentBoundarySpec,
    inward_water_discharge_m3_s,
    interior_concentration_kg_m3,
    *,
    time_s: float,
) -> np.ndarray:
    """Return signed inward dry-mass rates on all faces of one boundary ID.

    Incoming faces use the prescribed data. Outgoing faces always carry the
    adjacent interior concentration. A prescribed total mass load is divided
    among incoming faces in proportion to their accepted water discharge.
    """

    water = np.asarray(inward_water_discharge_m3_s, dtype=float)
    interior = np.asarray(interior_concentration_kg_m3, dtype=float)
    if water.ndim != 1 or interior.shape != water.shape:
        raise TransportError("boundary water flux and interior concentration must be same-sized vectors")
    if not np.all(np.isfinite(water)) or not np.all(np.isfinite(interior)) or np.any(interior < 0.0):
        raise TransportError("boundary face data must be finite with non-negative concentration")
    value = spec.value_at(time_s)
    incoming_total = float(np.maximum(water, 0.0).sum())
    if spec.mode == "mass_load" and value > 0.0 and incoming_total <= 0.0:
        raise TransportError("positive sediment mass load requires inward water flow")
    incoming_concentration = value if spec.mode == "concentration" else (
        value / incoming_total if incoming_total > 0.0 else 0.0
    )
    face_concentration = np.where(water >= 0.0, incoming_concentration, interior)
    return water * face_concentration


def write_native_sediment_boundaries(
    path: str | Path,
    specs: Iterable[SedimentBoundarySpec],
    numeric_boundary_ids: Mapping[str, int],
) -> dict:
    """Write ``braids_sediment_boundary.data`` format version 1."""

    records = tuple(specs)
    if len({item.boundary_id for item in records}) != len(records):
        raise TransportError("sediment boundary IDs must be unique")
    missing = {item.boundary_id for item in records} - set(numeric_boundary_ids)
    if missing:
        raise TransportError(f"sediment boundary has no hydraulic ID: {sorted(missing)}")
    mode = {"concentration": 1, "mass_load": 2}
    endpoint = {"hold": 1, "zero": 2, "error": 3}
    lines = ["1", str(len(records))]
    for item in sorted(records, key=lambda record: numeric_boundary_ids[record.boundary_id]):
        lines.append(
            f"{numeric_boundary_ids[item.boundary_id]} {mode[item.mode]} "
            f"{endpoint[item.endpoint]} {len(item.times_s)}"
        )
        lines.extend(f"{time:.17g} {value:.17g}" for time, value in zip(item.times_s, item.values))
    destination = Path(path)
    destination.write_text("\n".join(lines) + "\n")
    return {
        "format_version": 1,
        "boundaries": [
            {"boundary_id": item.boundary_id, "numeric_id": numeric_boundary_ids[item.boundary_id],
             "mode": item.mode, "endpoint": item.endpoint,
             "times_s": list(item.times_s), "values": list(item.values)}
            for item in records
        ],
    }
