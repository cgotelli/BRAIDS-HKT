"""Geometry and observation access for the fixed-wall, depth-averaged LFM test.

No engine is run or modified by this module. In particular, Delft3D source is
read-only. Figure-derived observations must not be called raw measurements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "specification.json").read_text(encoding="utf-8"))
G = SPEC["geometry"]
P = SPEC["published_2023"]
WIDTH, RADIUS = G["width_m"], G["centerline_radius_m"]
APPROACH, EXIT = G["upstream_straight_m"], G["downstream_straight_m"]
ANGLE = math.radians(G["bend_angle_degrees"])
BEND_END = APPROACH + RADIUS * ANGLE
LENGTH = BEND_END + EXIT
Q, DEPTH = P["discharge_m3_s"], P["downstream_water_depth_m"]
SLOPE, MANNING = P["initial_longitudinal_slope"], 1 / P["strickler_m13_s"]


def xy(s, n):
    """Cartesian location; n is physical distance inward from the OUTER wall."""
    s, n = np.broadcast_arrays(np.asarray(s, float), np.asarray(n, float))
    theta = np.clip((s - APPROACH) / RADIUS, 0, ANGLE)
    radius = RADIUS + WIDTH / 2 - n
    x, y = APPROACH + radius * np.sin(theta), radius * np.cos(theta)
    x = np.where(s < APPROACH, s, x)
    x = np.where(s > BEND_END, APPROACH - (s - BEND_END), x)
    return x, y


def sn(x, y):
    """Inverse of xy for channel points, including both straight reaches."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    theta = np.clip(np.arctan2(x - APPROACH, y), 0, ANGLE)
    choices = np.stack((np.clip(x, 0, APPROACH), APPROACH + RADIUS * theta,
                        np.clip(BEND_END + APPROACH - x, BEND_END, LENGTH)))
    cx, cy = xy(choices, WIDTH / 2)
    index = np.argmin((cx - x) ** 2 + (cy - y) ** 2, axis=0)
    s = np.take_along_axis(choices, index[None], axis=0)[0]
    theta = np.clip((s - APPROACH) / RADIUS, 0, ANGLE)
    cx, cy = xy(s, WIDTH / 2)
    n = WIDTH / 2 - (x - cx) * np.sin(theta) - (y - cy) * np.cos(theta)
    return s, n


def initial_bed(s):
    """Longitudinally sloping, transversely flat reconstruction; outlet z=0."""
    return SLOPE * (LENGTH - np.asarray(s))


def grid_edges(dx):
    """Curvilinear grid whose largest along-wall edge is at most native dx.

    Matching only centerline arc length makes outer Delft3D cells 20% longer.
    The outer radius therefore determines the angular subdivision instead.
    """
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be positive and finite")
    nu, nd = math.ceil(APPROACH / dx), math.ceil(EXIT / dx)
    nb = math.ceil((RADIUS + WIDTH / 2) * ANGLE / dx)
    nw = math.ceil(WIDTH / dx)
    s = np.r_[np.linspace(0, APPROACH, nu + 1),
              np.linspace(APPROACH, BEND_END, nb + 1)[1:],
              np.linspace(BEND_END, LENGTH, nd + 1)[1:]]
    # Positive Jacobian for FLOW: the second grid axis points left of flow.
    n = np.linspace(WIDTH, 0, nw + 1)
    return s, n


def observations():
    value = json.loads((HERE / "data" / "figure17_observations.json").read_text(encoding="utf-8"))
    if value["kind"] != "digitized_experimental_markers_not_raw_data":
        raise ValueError("Not an experimental observation dataset")
    if value["source_figure_sha256"] != value["calibration"]["source_sha256"]:
        raise ValueError("Observation/calibration source mismatch")
    manifest = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    pinned = next(item["sha256"] for item in manifest["files"]
                  if item["name"] == "lazzarin2023_fig17.jpg")
    if value["source_figure_sha256"] != pinned:
        raise ValueError("Observation source differs from the pinned publisher figure")
    return value


def profile_metrics(relative_coordinate, bed_change):
    """Global trends on one observed section; no imposed monotonicity/smoothing."""
    r, z = np.asarray(relative_coordinate), np.asarray(bed_change)
    if r.ndim != 1 or z.shape != r.shape or len(r) < 2 or not np.all(np.isfinite(z)):
        raise ValueError("Finite, paired one-dimensional profile required")
    if not np.all(np.isfinite(r)) or not np.all(np.diff(r) > 0):
        raise ValueError("Coordinates must increase strictly")
    outer, inner = z[r <= .25], z[r >= .75]
    if not len(outer) or not len(inner):
        raise ValueError("Both bank regions must be represented")
    crossings = []
    for i in range(len(z) - 1):
        if z[i] == 0:
            crossings.append(float(r[i]))
        elif z[i] * z[i + 1] < 0:
            crossings.append(float(r[i] - z[i] * (r[i + 1] - r[i]) / (z[i + 1] - z[i])))
    if z[-1] == 0:
        crossings.append(float(r[-1]))
    return dict(min_bed_change_m=float(z.min()), max_bed_change_m=float(z.max()),
                relief_m=float(z.max() - z.min()), outer_mean_m=float(outer.mean()),
                inner_mean_m=float(inner.mean()), zero_crossings=crossings)
