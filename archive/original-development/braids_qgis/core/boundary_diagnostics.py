"""Requested-versus-accepted boundary diagnostics for completed QGIS runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from braids_core.boundaries.cut_cell_io import CutBoundary


QUANTITIES = (
    ("water", "water_volume_m3", "Water", "m³", "m³/s"),
    ("suspended", "suspended_mass_kg", "Suspended sediment", "kg", "kg/s"),
    ("bedload", "bedload_mass_kg", "Bed load", "kg", "kg/s"),
)


def _linear_value(times: np.ndarray, values: np.ndarray, at: float) -> float:
    return float(np.interp(at, times, values, left=values[0], right=values[-1]))


def _integral_linear(times: Sequence[float], values: Sequence[float], stop: float) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if stop <= 0:
        return 0.0
    total = min(stop, float(times[0])) * float(values[0]) if times[0] > 0 else 0.0
    for left, right, first, last in zip(times[:-1], times[1:], values[:-1], values[1:]):
        if stop <= left:
            break
        width = min(stop, float(right)) - float(left)
        if width > 0:
            slope = (last - first) / (right - left)
            total += first * width + 0.5 * slope * width * width
        if stop <= right:
            return float(total)
    if stop > times[-1]:
        total += (stop - times[-1]) * values[-1]
    return float(total)


def _integral_product(times_a: Sequence[float], values_a: Sequence[float],
                      times_b: Sequence[float], values_b: Sequence[float], stop: float) -> float:
    if stop <= 0:
        return 0.0
    knots = sorted({0.0, float(stop), *[float(value) for value in times_a if 0 < value < stop],
                    *[float(value) for value in times_b if 0 < value < stop]})
    ta, va = np.asarray(times_a, dtype=float), np.asarray(values_a, dtype=float)
    tb, vb = np.asarray(times_b, dtype=float), np.asarray(values_b, dtype=float)
    total = 0.0
    for left, right in zip(knots[:-1], knots[1:]):
        width = right - left
        a0, a1 = _linear_value(ta, va, left), _linear_value(ta, va, right)
        b0, b1 = _linear_value(tb, vb, left), _linear_value(tb, vb, right)
        da, db = a1 - a0, b1 - b0
        total += width * (a0 * b0 + 0.5 * (a0 * db + b0 * da) + da * db / 3.0)
    return float(total)


def _load_run(run_root: str | Path) -> tuple[Path, dict, dict, dict]:
    run_root = Path(run_root).expanduser().resolve()
    project_path = run_root / "project.json"
    case_path = run_root / "case.data.json"
    manifest_path = run_root / "qgis_results/results.json"
    for path in (project_path, case_path, manifest_path):
        if not path.is_file():
            raise ValueError(f"Completed result artifact is missing: {path}")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    case = json.loads(case_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return run_root, project, case, manifest


def available_boundary_ids(run_root: str | Path) -> list[str]:
    """Return stable logical IDs in native numeric order."""
    _, _, case, _ = _load_run(run_root)
    return [name for name, _ in sorted(case["string_to_numeric_id"].items(), key=lambda item: item[1])]


def _budget_rows(run_root: Path, manifest: dict, case: dict, bc_id: str) -> tuple[np.ndarray, np.ndarray]:
    name = manifest.get("boundary_budgets", "boundary_budgets.csv")
    path = run_root / "qgis_results" / name
    if not path.is_file() and ((run_root / "frames.txt").is_file() or
                               (run_root / "boundary_ledger.txt").is_file()):
        # Alpha runs whose raster products came from the fast live-preview VRT
        # path predate boundary-budget materialization. Recover the native
        # ledger on demand so completed runs do not need to be executed again.
        from .results import _write_boundary_budgets
        name = _write_boundary_budgets(run_root, path.parent, case)
        path = path.parent / name
    if not path.is_file():
        raise ValueError(f"Boundary-budget table is missing: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["bc_id"] == bc_id:
                fraction_ids=case.get("fraction_ids",[])
                if fraction_ids:
                    suspended=sum(float(row[f"suspended_{name}_kg"]) for name in fraction_ids)
                    bedload=sum(float(row[f"bedload_{name}_kg"]) for name in fraction_ids)
                    rows.append((float(row["time_s"]),float(row["water_volume_m3"]),suspended,bedload))
                else:
                    rows.append((float(row["time_s"]), *[float(row[field]) for _, field, *_ in QUANTITIES]))
    if not rows:
        raise ValueError(f"Boundary ID {bc_id!r} has no accepted ledger rows")
    data = np.asarray(rows, dtype=float)
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"Boundary ID {bc_id!r} has invalid or unordered ledger rows")
    return data[:, 0], data[:, 1:]


def _constant_series(value: float, end_time_s: float) -> tuple[tuple[float, float], tuple[float, float]]:
    return (0.0, end_time_s), (float(value), float(value))


def _sum_fraction_series(boundary: CutBoundary, field: str, end_time_s: float):
    knots={0.0,float(end_time_s)}
    for series in boundary.fraction_sediment_series.values():knots.update(series.times_s)
    times=tuple(sorted(knots))
    constants=(boundary.fraction_concentration_kg_m3 if field=="concentration_kg_m3" else
               boundary.fraction_bed_input_kg_s)
    values=[]
    for at in times:
        total=sum(constants.values())
        for series in boundary.fraction_sediment_series.values():
            total+=_linear_value(np.asarray(series.times_s),np.asarray(getattr(series,field)),at)
        values.append(total)
    return times,tuple(values)


def _requested_series(boundary: CutBoundary, case: dict, bc_id: str,
                      end_time_s: float) -> dict[str, tuple[Sequence[float], Sequence[float]] | None] | None:
    if not boundary.is_inlet:
        return None
    if boundary.hydraulic_series is None:
        q_times, q_values = _constant_series(boundary.discharge_m3_s, end_time_s)
    else:
        q_times = boundary.hydraulic_series.times_s
        q_values = boundary.hydraulic_series.discharge_m3_s
    if boundary.sediment_series is None:
        if boundary.fraction_sediment_series:
            c_times,c_values=_sum_fraction_series(boundary,"concentration_kg_m3",end_time_s)
        else:
            total_concentration=(sum(boundary.fraction_concentration_kg_m3.values())
                                 if case.get("fraction_ids") else boundary.concentration_kg_m3)
            c_times, c_values = _constant_series(total_concentration, end_time_s)
        if boundary.bedload_boundary_mode == "equilibrium_capacity":
            bed_series = None
        elif boundary.fraction_sediment_series:
            bed_series=_sum_fraction_series(boundary,"bed_input_kg_s",end_time_s)
        else:
            total_bed=(sum(row["bed_input_kg_s"] for row in case["resolved_fraction_input"][bc_id].values())
                       if case.get("fraction_ids") else case["resolved_bed_input"][bc_id]["total_kg_s"])
            bed_series = _constant_series(total_bed, end_time_s)
    else:
        c_times = boundary.sediment_series.times_s
        c_values = boundary.sediment_series.concentration_kg_m3
        bed_series = (boundary.sediment_series.times_s, boundary.sediment_series.bed_input_kg_s)
    return {
        "q": (q_times, q_values),
        "concentration": (c_times, c_values),
        "bedload": bed_series,
    }


def _requested_cumulative(series: dict | None, at: float) -> tuple[float, float, float | None] | None:
    if series is None:
        return None
    q_times, q_values = series["q"]
    c_times, c_values = series["concentration"]
    bed_series = series["bedload"]
    return (
        _integral_linear(q_times, q_values, at),
        _integral_product(q_times, q_values, c_times, c_values, at),
        None if bed_series is None else _integral_linear(*bed_series, at),
    )


def boundary_diagnostic(run_root: str | Path, bc_id: str) -> dict:
    """Build plot-ready requested and accepted cumulative/interval records.

    Requested fluxes exist for inlet IDs. Stage/critical/weir/supercritical
    outlets are state/rating controls, so their accepted export is reported
    without inventing a requested discharge or sediment flux.
    """
    run_root, project, case, manifest = _load_run(run_root)
    if bc_id not in case["string_to_numeric_id"] or bc_id not in project["boundaries"]:
        raise ValueError(f"Unknown boundary ID: {bc_id}")
    times, accepted = _budget_rows(run_root, manifest, case, bc_id)
    boundary = CutBoundary(**project["boundaries"][bc_id])
    end_time_s = float(project["numerics"]["end_time_s"])
    requested_series = _requested_series(boundary, case, bc_id, end_time_s)
    requested_columns = None
    if requested_series is not None:
        cumulative = [_requested_cumulative(requested_series, at) for at in times]
        requested_columns = [
            None if cumulative[0][column] is None else np.asarray([row[column] for row in cumulative], dtype=float)
            for column in range(len(QUANTITIES))
        ]
    interval_starts, interval_ends = times[:-1], times[1:]
    widths = interval_ends - interval_starts
    # The native ledger retains its conservative signed orientation.  Results
    # expose magnitudes instead: the boundary definition already says whether
    # this is inflow or outflow, and a negative plotted discharge is otherwise
    # easy to misread as a failed outlet.
    accepted_magnitudes=np.abs(accepted)
    accepted_rates=np.abs(np.diff(accepted, axis=0)/widths[:,None])
    quantities = {}
    for column, (key, _, title, cumulative_unit, rate_unit) in enumerate(QUANTITIES):
        requested = None if requested_columns is None else requested_columns[column]
        quantities[key] = {
            "title": title,
            "cumulative_unit": cumulative_unit,
            "rate_unit": rate_unit,
            "accepted_cumulative": accepted_magnitudes[:, column].tolist(),
            "requested_cumulative": None if requested is None else np.abs(requested).tolist(),
            "accepted_interval_average": accepted_rates[:, column].tolist(),
            "requested_interval_average": None if requested is None else np.abs(np.diff(requested)/widths).tolist(),
        }
    return {
        "bc_id": bc_id,
        "mode": boundary.mode,
        "direction": "inflow" if boundary.is_inlet else "outflow",
        "flux_comparison_available": requested_columns is not None,
        "accepted_sign_convention": "nonnegative magnitude; boundary role identifies inflow or outflow",
        "note": (("Water and suspended requested inlet fluxes are exact schedule integrals. Equilibrium bed load has no prescribed target; its accepted native ledger is shown."
                  if boundary.bedload_boundary_mode == "equilibrium_capacity" else
                  "Requested inlet fluxes are exact schedule integrals. Accepted values are native committed-step ledgers.")
                 if requested_columns is not None else
                 "This outlet prescribes a state or rating control, not a target flux; its accepted export magnitude is shown."),
        "times_s": times.tolist(),
        "interval_midpoints_s": ((interval_starts + interval_ends) / 2.0).tolist(),
        "interval_start_s": interval_starts.tolist(),
        "interval_end_s": interval_ends.tolist(),
        "quantities": quantities,
    }
