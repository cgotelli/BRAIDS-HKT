"""Strict, NumPy-only reader for the native GAIA model-3 sidecar."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from braids_core.windows_paths import filesystem_path


SCHEMA = "BRAIDS_GAIA_SECONDARY_STRESS_V1"
COLUMNS = (
    "time_s", "cell_id", "fraction_id", "x_m", "y_m", "active",
    "eta_grad_x", "eta_grad_y", "uphill_bed_grad_x", "uphill_bed_grad_y",
    "tau_b_Pa", "slope_stress_x_Pa", "slope_stress_y_Pa",
    "secondary_delta_x_Pa", "secondary_delta_y_Pa", "total_stress_x_Pa",
    "total_stress_y_Pa", "secondary_turn_rad", "total_turn_from_flow_rad",
    "base_capacity_x_kg_m_s", "base_capacity_y_kg_m_s",
    "corrected_capacity_x_kg_m_s", "corrected_capacity_y_kg_m_s",
    "slope_magnitude_factor",
)
_FLAGS = re.compile(
    r"# v44_process_flags interior_slope_talmon interior_secondary_stress "
    r"open_boundary_secondary_stress ([01]) ([01]) ([01])\Z"
)


def read_gaia_secondary_stress(path: str | Path) -> dict:
    """Read and validate a version-1 accepted-frame CPU stress diagnostic.

    The returned matrix retains the explicit on-disk column order.  No values
    are inferred from the legacy ``secondary_flow.txt`` columns.
    """
    path = filesystem_path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            header = [stream.readline().rstrip("\r\n") for _ in range(4)]
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read GAIA secondary-stress diagnostic: {exc}") from exc
    if header[0] != "# " + SCHEMA:
        raise ValueError("Unsupported GAIA secondary-stress diagnostic schema")
    if header[1] != "# " + " ".join(COLUMNS):
        raise ValueError("GAIA secondary-stress columns differ from schema 1")
    if not header[2].startswith("# eta_grad=grad(z+h); uphill_bed_grad=grad(z);"):
        raise ValueError("GAIA secondary-stress conventions are missing")
    match = _FLAGS.fullmatch(header[3])
    if match is None:
        raise ValueError("GAIA secondary-stress process flags are invalid")
    try:
        values = np.loadtxt(path, comments="#", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid GAIA secondary-stress numeric table: {exc}") from exc
    if values.ndim != 2 or values.shape[1] != len(COLUMNS) or not np.isfinite(values).all():
        raise ValueError("Invalid GAIA secondary-stress numeric layout")
    if (np.any(values[:, 0] < 0.) or np.any(np.diff(values[:, 0]) < 0.) or
            np.any(values[:, 1:3] < 0.) or
            not np.equal(values[:, 1:3], np.floor(values[:, 1:3])).all() or
            not np.isin(values[:, 5], (0., 1.)).all() or np.any(values[:, 10] < 0.) or
            np.any(np.abs(values[:, 17:19]) > np.pi + 32*np.finfo(float).eps)):
        raise ValueError("Invalid GAIA secondary-stress identifiers, clock, or values")
    scale = np.maximum(1., np.maximum(np.abs(values[:, 15:17]),
                                      np.abs(values[:, 11:13])))
    if np.any(np.abs(values[:, 15:17] -
                     (values[:, 11:13] + values[:, 13:15])) >
              512*np.finfo(float).eps*scale):
        raise ValueError("GAIA secondary-stress decomposition differs")
    active = values[:, 5].astype(bool)
    if np.any(active):
        if np.any(values[active, 10] <= 0.):
            raise ValueError("Active GAIA secondary-stress rows require positive shear")
        magnitude = np.hypot(values[active, 19], values[active, 20])
        expected = magnitude[:, None]*values[active, 15:17]/values[active, 10, None]
        actual = values[active, 21:23]
        scale = np.maximum(1.e-20, np.maximum(np.abs(expected), np.abs(actual)))
        if np.any(np.abs(expected-actual) > 512*np.finfo(float).eps*scale):
            raise ValueError("GAIA secondary-stress capacity relation differs")
    names = ("interior_slope_talmon", "interior_secondary_stress",
             "open_boundary_secondary_stress")
    return {
        "schema": SCHEMA,
        "columns": COLUMNS,
        "process_flags": dict(zip(names, (bool(int(value)) for value in match.groups()))),
        "values": values,
    }
