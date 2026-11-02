"""Build checked observations, physical meshes, and figures; never run an engine.

Run from repository root with .venv/Scripts/python.exe -m
braids_support.sediment.lfm_bend_2d.build_package --source-dir tmp/pdfs
or --fetch to obtain the pinned public source files in ignored braids_support/vendor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

from . import experiment as e
from .digitize import extract, sha256
from braids_core.boundaries.agglomeration import agglomerate_small_cells
from braids_core.boundaries.cut_cells import map_cut_cells
from braids_core.boundaries.domain import BoundaryAssignment, PolygonDomain
from braids_core.boundaries.inlet_io import NativeGrid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT = HERE / "reports" / "benchmark_ready"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sources(source_dir, fetch=False):
    manifest = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    source_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        path = source_dir / item["name"]
        if not path.exists() and fetch:
            with urllib.request.urlopen(item["url"], timeout=30) as response:
                payload = response.read()
            import hashlib
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded source changed: {item['name']}")
            path.write_bytes(payload)
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing/changed source: {path}; restore the pinned source before digitizing")
    return manifest


def native_mesh(dx):
    stations = np.unique(np.r_[0, np.linspace(e.APPROACH, e.BEND_END, 721), e.LENGTH])
    xo, yo = e.xy(stations, 0)
    xi, yi = e.xy(stations, e.WIDTH)
    ring = tuple(zip(xo, yo)) + tuple(zip(xi[::-1], yi[::-1]))
    domain = PolygonDomain("LFM fixed walls 2D", ring, "EPSG:2056")
    inlet = set(zip(*e.xy([0, 0], [0, e.WIDTH])))
    outlet = set(zip(*e.xy([e.LENGTH, e.LENGTH], [0, e.WIDTH])))
    assigned = []
    for edge in domain.edges:
        ends = {edge.start_xy, edge.end_xy}
        if ends == inlet:
            assigned.append(BoundaryAssignment(edge.edge_id, "inlet", "inlet"))
        if ends == outlet:
            assigned.append(BoundaryAssignment(edge.edge_id, "outlet", "outlet"))
    if len(assigned) != 2:
        raise ValueError("Physical inlet/outlet edge identification failed")
    points = np.array(ring)
    lower = np.floor(points.min(axis=0) / dx) * dx
    nx, ny = np.ceil((points.max(axis=0) - lower) / dx).astype(int)
    grid = NativeGrid(int(nx), int(ny), *lower, dx, dx)
    raw = map_cut_cells(domain, assigned, grid)
    mesh = agglomerate_small_cells(raw, .5)[0]
    return mesh, len(raw.cells)


def delft_geometry(dx):
    s, n = e.grid_edges(dx)
    ss, nn = np.meshgrid(s, n)
    x, y = e.xy(ss, nn)
    polygons = np.stack((np.stack((x[:-1, :-1], y[:-1, :-1]), -1),
                         np.stack((x[:-1, 1:], y[:-1, 1:]), -1),
                         np.stack((x[1:, 1:], y[1:, 1:]), -1),
                         np.stack((x[1:, :-1], y[1:, :-1]), -1)), -2).reshape(-1, 4, 2)
    a, b = polygons, np.roll(polygons, -1, axis=1)
    signed_area = .5 * np.sum(a[:, :, 0] * b[:, :, 1] - b[:, :, 0] * a[:, :, 1], axis=1)
    if np.any(signed_area <= 0):
        raise ValueError("Non-positive FLOW cell Jacobian")
    return dict(s=s, n=n, x=x, y=y, polygons=polygons, area=signed_area)


def geometry_record(dx, mesh, raw_count, delft, elapsed):
    area = np.array([cell.area_m2 for cell in mesh.cells])
    flux_width = {name: math.fsum(f.length_m for f in mesh.faces if f.bc_id == name)
                  for name in ("inlet", "outlet")}
    circumference = np.zeros(len(mesh.cells))
    for f in mesh.faces:
        circumference[f.owner] += f.length_m
        if f.neighbour >= 0:
            circumference[f.neighbour] += f.length_m
    edge_length = np.linalg.norm(delft["polygons"] - np.roll(delft["polygons"], -1, axis=1), axis=2)
    exact = e.LENGTH * e.WIDTH
    if not all(abs(v - e.WIDTH) < 1e-9 for v in flux_width.values()):
        raise ValueError("Inlet/outlet does not span the physical width")
    if edge_length.max() > dx * (1 + 1e-10):
        raise ValueError("FLOW grid is coarser than the requested native spacing")
    if abs(area.sum() / exact - 1) > 1e-4 or abs(delft["area"].sum() / exact - 1) > 5e-4:
        raise ValueError("Mesh area differs excessively from the analytical channel")
    return dict(nominal_spacing_m=dx, native_cells=len(area), native_raw_cells=raw_count,
                native_merged_cell_count=sum(len(c.member_indices) > 1 for c in mesh.cells),
                native_area_m2=float(area.sum()), analytical_area_m2=exact,
                native_cell_area_quantiles_m2=np.quantile(area, [0, .1, .5, .9, 1]).tolist(),
                native_min_area_over_perimeter_m=float(np.min(area / circumference)),
                native_boundary_width_m=flux_width,
                delft_cells=int(len(delft["area"])), delft_cross_channel_cells=len(delft["n"]) - 1,
                delft_streamwise_cells=len(delft["s"]) - 1, delft_vertical_layers=1,
                delft_max_horizontal_edge_m=float(edge_length.max()),
                delft_area_m2=float(delft["area"].sum()),
                delft_cell_area_quantiles_m2=np.quantile(delft["area"], [0, .1, .5, .9, 1]).tolist(),
                preparation_wall_s=elapsed,
                caveat="Matched background spacing and FLOW max edge, not identical topology. Native cut cells can be agglomerated near walls; area distributions are retained.")


def observation_figures(data, source_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, profile in zip(axes, data["profiles"]):
        r = np.array([p["relative_from_outer"] for p in profile["points"]])
        z = np.array([p["bed_change_m"] for p in profile["points"]])
        ax.axhline(0, color=".25", linewidth=1)
        ax.fill_between(r, 0, z, where=z < 0, color="#4387bf", alpha=.20, interpolate=True)
        ax.fill_between(r, 0, z, where=z > 0, color="#c87135", alpha=.20, interpolate=True)
        ax.errorbar(r, z, yerr=.005, fmt="o", markersize=3.4, color="black", capsize=2,
                    label="Published experimental markers")
        ax.set(title=f"Bend section {profile['angle_degrees']}°", xlabel="Outer wall  0  →  1  Inner wall",
               xlim=(-.02, 1.02), ylim=(-.225, .155))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Bed change from initial bed (m)")
    fig.suptitle("LFM experimental target — no BRAIDS or Delft3D results\n"
                 "Digitized from Lazzarin & Viero (2023), Fig. 17; ±5 mm is graphical tolerance, not measurement uncertainty", fontsize=11)
    fig.savefig(REPORT / "experimental_profiles.png", dpi=180)
    plt.close(fig)

    # Internal QA overlay, in ignored prepared/, not a republished source figure.
    image = plt.imread(source_dir / "lazzarin2023_fig17.jpg")
    fig, ax = plt.subplots(figsize=(17, 5), layout="constrained")
    ax.imshow(image)
    for profile in data["profiles"]:
        ax.scatter([p["pixel_x"] for p in profile["points"]], [p["pixel_y"] for p in profile["points"]],
                   s=80, facecolors="none", edgecolors="#00aaa0", linewidths=1)
    ax.set_axis_off()
    fig.savefig(HERE / "prepared" / "digitization_overlay.png", dpi=200)
    plt.close(fig)


def mesh_figure(pairs, data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, (dx, mesh, delft) in enumerate(pairs):
        for ax, polygons, name in ((axes[row, 0], [c.vertices_xy for c in mesh.cells], "BRAIDS cut cells"),
                                   (axes[row, 1], delft["polygons"], "Delft3D 2D curvilinear")):
            ax.add_collection(PolyCollection(polygons, facecolors="#f1f4f7", edgecolors="#a9b4bc", linewidths=.25))
            for profile in data["profiles"]:
                angle = profile["angle_degrees"]
                s = e.APPROACH + e.RADIUS * math.radians(angle)
                x, y = e.xy(s, np.array([0, e.WIDTH]))
                ax.plot(x, y, color="#bd4e2b", linewidth=1.4)
                xx, yy = e.xy(s, -0.25)
                ax.text(xx, yy, f"{angle}°", fontsize=8, ha="center")
            ax.autoscale()
            ax.set_aspect("equal")
            ax.set(title=f"{name}: {dx:g} m, {len(polygons):,} cells", xlabel="x (m)", ylabel="y (m)")
    fig.suptitle("Same physical bend and two matched resolution levels\n"
                 "Fixed walls; 10 m straight reaches are a documented reconstruction; geometry only, not simulation results", fontsize=11)
    fig.savefig(REPORT / "matched_meshes.png", dpi=180)
    plt.close(fig)


def build(source_dir, fetch=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / "completed.json", dict(benchmark_package_complete=False, status="building", engines_executed=False))
    sources(source_dir, fetch)
    (HERE / "prepared").mkdir(exist_ok=True)
    data = extract(source_dir / "lazzarin2023_fig17.jpg")
    # Robustness to JPEG threshold choice. This is not experimental uncertainty.
    alternative = extract(source_dir / "lazzarin2023_fig17.jpg", red_min=150, other_max=120)
    dz, dr = [], []
    metrics = {}
    for a, b in zip(data["profiles"], alternative["profiles"]):
        dz.extend(abs(x["bed_change_m"] - y["bed_change_m"]) for x, y in zip(a["points"], b["points"]))
        dr.extend(abs(x["relative_from_outer"] - y["relative_from_outer"]) for x, y in zip(a["points"], b["points"]))
        metrics[str(a["angle_degrees"])] = e.profile_metrics(
            [p["relative_from_outer"] for p in a["points"]], [p["bed_change_m"] for p in a["points"]])
    data["threshold_sensitivity"] = dict(max_bed_change_difference_m=max(dz), max_relative_coordinate_difference=max(dr))
    if max(dz) > .001 or max(dr) > .002:
        raise ValueError("Marker extraction is excessively threshold-sensitive")
    write_json(HERE / "data" / "figure17_observations.json", data)
    observation_figures(data, source_dir)
    pairs, mesh_records = [], []
    for dx in (e.SPEC["comparison_design"]["primary_dx_m"], e.SPEC["comparison_design"]["refined_dx_m"]):
        start = time.perf_counter()
        mesh, raw_count = native_mesh(dx)
        delft = delft_geometry(dx)
        record = geometry_record(dx, mesh, raw_count, delft, time.perf_counter() - start)
        mesh_records.append(record)
        pairs.append((dx, mesh, delft))
        tag = str(dx).replace(".", "p")
        mesh.write_geometry(HERE / "prepared" / f"braids_geometry_{tag}.npz")
        np.savez_compressed(HERE / "prepared" / f"delft_geometry_{tag}.npz",
                            **{k: v for k, v in delft.items() if k != "polygons"})
        print(f"Prepared {dx:g} m: BRAIDS {len(mesh.cells)} cells, Delft3D {len(delft['area'])} cells", flush=True)
    mesh_figure(pairs, data)
    lock_path = ROOT / "braids_support/sediment/delft3d_current/runtime.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    build_path = ROOT / "solver/cut_cells/build/build.json"
    build = json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists() else None
    native_path = ROOT / "solver/cut_cells/build/xbraids_cutcell.exe"
    summary = dict(milestone="Experimental data and 2D geometry package",
                   benchmark_package_complete=True, engines_executed=False, engine_source_modified=False,
                   exact_paper_replication=False, mobile_run_ready=False,
                   observations=data["count"], experimental_profile_metrics=metrics,
                   digitization_threshold_sensitivity=data["threshold_sensitivity"],
                   numerical_geometry_checks=mesh_records,
                   unresolved_before_mobile_run=e.SPEC["unresolved_before_mobile_run"],
                   runtimes=dict(delft_recorded_lock=lock,
                       note="Recorded identities only; engine execution/2D activation must be verified in the next milestone.",
                       braids_executable_sha256=sha256(native_path) if native_path.exists() else None,
                       braids_build_record_sha256=sha256(build_path) if build else None),
                   inputs_sha256={p.name: sha256(p) for p in [HERE / "specification.json", HERE / "sources.json",
                       HERE / "experiment.py", HERE / "digitize.py", HERE / "build_package.py", HERE / "data/figure17_observations.json"]},
                   prepared_geometry_sha256={p.name: sha256(p) for p in sorted((HERE / "prepared").glob("*geometry*.npz"))})
    write_json(REPORT / "completed.json", summary)
    print(json.dumps(dict(observations=data["count"], metrics=metrics,
                          extraction_sensitivity=data["threshold_sensitivity"]), indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "braids_support/vendor/lfm_bend_2d")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    build(args.source_dir, args.fetch)
