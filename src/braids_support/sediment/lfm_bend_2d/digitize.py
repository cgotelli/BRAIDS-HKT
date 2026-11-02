"""Reproducibly extract RED experimental dots, never black simulated curves.

Pixel coordinates refer to the publisher's original 3333 x 948 Figure 17 JPEG.
Calibration uses all labelled grid lines, not an assumed equal dot spacing.
Scientific image analysis only; the downloaded source is not edited.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PANELS = (
    dict(angle=45, crop=[158, 1020, 50, 665], expected_count=21,
         x_grid=[165, 334, 506, 672, 843, 1013], y_grid=[64, 149, 230, 316, 397, 481, 565, 649]),
    dict(angle=90, crop=[1315, 2175, 50, 665], expected_count=20,
         x_grid=[1321, 1492, 1661, 1830, 2000, 2166], y_grid=[65, 150, 234, 320, 404, 490, 573, 656]),
    dict(angle=135, crop=[2470, 3333, 45, 665], expected_count=20,
         x_grid=[2477, 2646, 2817, 2985, 3153, 3323], y_grid=[58, 145, 228, 314, 401, 486, 571, 655]),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(path, *, red_min=180, other_max=90):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=int)
    if rgb.shape != (948, 3333, 3):
        raise ValueError("Use original-resolution publisher Figure 17, not a screenshot")
    red = (rgb[:, :, 0] > red_min) & (rgb[:, :, 1] < other_max) & (rgb[:, :, 2] < other_max)
    profiles = []
    calibrations = []
    for panel in PANELS:
        x0, x1, y0, y1 = panel["crop"]
        labels, _ = ndimage.label(red[y0:y1, x0:x1])
        xc = np.polyfit(panel["x_grid"], np.linspace(0, 1, 6), 1)
        yc = np.polyfit(panel["y_grid"], np.linspace(.4, -.3, 8), 1)
        dots = []
        for component, sl in enumerate(ndimage.find_objects(labels), 1):
            if sl is None:
                continue
            yy, xx = np.where(labels[sl] == component)
            if len(xx) < 200:
                continue
            if not (12 <= sl[1].stop - sl[1].start <= 28 and 15 <= sl[0].stop - sl[0].start <= 28):
                raise ValueError("Red component is not an expected experimental marker")
            px, py = float(xx.mean() + sl[1].start + x0), float(yy.mean() + sl[0].start + y0)
            r, z = float(np.polyval(xc, px)), float(np.polyval(yc, py))
            clipped = (sl[1].start + x0 <= panel["x_grid"][0] + 5 or
                       sl[1].stop + x0 >= panel["x_grid"][-1] - 5)
            dots.append(dict(pixel_x=px, pixel_y=py, marker_pixels=len(xx),
                             relative_from_outer=r, bed_change_m=z,
                             clipped_by_plot_boundary=clipped,
                             primary_interior=.05 <= r <= .95 and not clipped))
        dots.sort(key=lambda dot: dot["relative_from_outer"])
        if len(dots) != panel["expected_count"]:
            raise ValueError(f"{panel['angle']} deg: found {len(dots)} dots, expected {panel['expected_count']}")
        profiles.append(dict(angle_degrees=panel["angle"], points=dots))
        calibrations.append(dict(**panel, pixel_to_relative=xc.tolist(), pixel_to_metres=yc.tolist(),
                                 max_x_grid_residual=float(np.max(abs(np.polyval(xc, panel["x_grid"]) - np.linspace(0, 1, 6)))),
                                 max_z_grid_residual_m=float(np.max(abs(np.polyval(yc, panel["y_grid"]) - np.linspace(.4, -.3, 8))))))
    digest = sha256(path)
    return dict(schema_version=1, kind="digitized_experimental_markers_not_raw_data",
                source="Lazzarin and Viero (2023), Fig. 17; experimental series attributed to Koch and Flokstra (1980)",
                doi="10.1016/j.advwatres.2022.104355",
                source_url="https://ars.els-cdn.com/content/image/1-s2.0-S0309170822002184-gr17_lrg.jpg",
                source_figure_sha256=digest, count=sum(len(p["points"]) for p in profiles),
                independent_measurement_count=None,
                note="61 plotted markers; independence and original sampling/interpolation are unknown. Never infer 61 independent raw measurements.",
                coordinate="0 outer wall, 1 inner wall; positive delta-z deposition, negative erosion",
                digitization_tolerance=dict(relative_coordinate=.01, bed_change_m=.005,
                    meaning="Conservative graphical reading tolerance, NOT experimental uncertainty or a confidence interval"),
                calibration=dict(source_sha256=digest, image_size=[3333, 948], panels=calibrations),
                profiles=profiles)
