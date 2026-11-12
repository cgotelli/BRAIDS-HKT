"""Read-only audit of the actual Iber deck and retained BRAIDS natural meshes.

No preparation, solver construction, simulation, or project mutation is used.
All generated analysis is confined to validation/_project133/iber_compare.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
IBER = Path("C:/Users/STREEM/Desktop/DEM-project-CIVIL410/test.gid")
AAA = Path("C:/Users/STREEM/Desktop/aaa/runs")
DOCS = Path("C:/Users/STREEM/Documents/BRAIDS")
OUT = ROOT / "validation/_project133/iber_compare"
INPUTS = {}


def identify(path):
    path = Path(path).resolve()
    INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(identify(path).read_text(encoding="utf-8-sig"))


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), minimum=float(a.min()), p01=float(np.quantile(a, .01)),
                p05=float(np.quantile(a, .05)), median=float(np.median(a)),
                mean=float(a.mean()), p95=float(np.quantile(a, .95)),
                p99=float(np.quantile(a, .99)), maximum=float(a.max()))


def geometry(polys):
    area = np.array([p.area for p in polys])
    perim = np.array([p.length for p in polys])
    sides = [np.linalg.norm(np.diff(np.array(p.exterior.coords), axis=0), axis=1)
             for p in polys]
    lengths = np.concatenate(sides)
    lengths = lengths[lengths > 1e-10]
    xy = np.vstack([p.exterior.coords for p in polys])
    return dict(cells=len(polys), total_area_m2=float(area.sum()),
                extent_xy_m=[*xy.min(axis=0).tolist(), *xy.max(axis=0).tolist()],
                extent_width_height_m=np.ptp(xy, axis=0).tolist(),
                cell_area_m2=stats(area), sqrt_area_m=stats(np.sqrt(area)),
                twice_area_over_perimeter_m=stats(2*area/perim),
                cell_side_length_m=stats(lengths),
                minimum_side_per_cell_m=stats([x[x > 1e-10].min() for x in sides]),
                cell_side_count=dict(Counter(len(x[x > 1e-10]) for x in sides)),
                note="2A/P is a geometric scale, not a claim about Iber's internal CFL formula. Side lengths are cell sides, shared sides counted twice; collinear coarse-fine face fragments are not reconstructed.")


def iber_data():
    lines = identify(IBER / "Iber2D.dat").read_text().splitlines()
    n = int(lines[3]); records = [s.split() for s in lines[4:4+n]]
    assert lines[4+n] == "VERTEX"
    nv = int(lines[5+n])
    verts = {int(s.split()[3]): np.array(list(map(float, s.split()[:3])))
             for s in lines[6+n:6+n+nv]}
    nodes = [list(map(int, c[:3])) for c in records]
    polygons = [Polygon([verts[v][:2] for v in c]) for c in nodes]
    incidence = Counter(tuple(sorted((c[k], c[(k+1) % 3])))
                        for c in nodes for k in range(3))
    bounds = {"inlet": {}, "outlet": {}}
    for line in lines[lines.index("CC: CONDICIONS CONTORN")+1:]:
        kind = "inlet" if "CriticoLento" in line else "outlet" if "Critico/Rapido" in line else None
        if not kind:
            continue
        values = line.split(); cell = int(values[0])-1; side = int(values[1])-1
        pair = (nodes[cell][side], nodes[cell][(side+1) % 3])
        assert incidence[tuple(sorted(pair))] == 1, (cell, side, pair)
        bounds[kind][(cell, side)] = [verts[v][:2].tolist() for v in pair]
    logtext = identify(IBER / "proceso.rep").read_text()
    rows = []
    for line in logtext.splitlines():
        m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+(\d+):(\d+):(\d+):(\d+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            t, dt, h, mi, s, cs, qi, qo = map(float, m.groups())
            rows.append(dict(time_s=t, reported_dt_s=dt, clock_s=3600*h+60*mi+s+.01*cs,
                             qin_m3_s=qi, qout_m3_s=qo, source_line=line.strip()))
    for row in rows:
        row["logged_elapsed_since_t0_s"] = row["clock_s"]-rows[0]["clock_s"]
    prb = identify(IBER / "test.prb").read_text()
    questions = dict(re.findall(r"QUESTION: ([^\r\n]+)[\r\n]+VALUE: ([^\r\n]*)", prb))
    header = lines[1].split()
    pd = identify(IBER / "Iber_Problemdata.dat").read_text()
    cpu = identify(IBER / "Iber_CPUSolver.dat").read_text().strip()
    selections = ("Numerical_Scheme", "CFL", "Max_time_increment_[s]", "Wet-Dry_limit_[m]",
                  "Number_of_Threads", "Strict_Courant_Condition", "Wall_Manning", "Local_Time_Step")
    bdata = {}
    for kind, faces in bounds.items():
        ids = sorted(set(k[0] for k in faces))
        bdata[kind] = dict(faces=len(faces), cells=len(ids),
                           width_m=sum(LineString(v).length for v in faces.values()),
                           lines_xy=list(faces.values()), cell_geometry=geometry([polygons[i] for i in ids]))
    early = [r for r in rows if 0 < r["time_s"] <= 27.139 + 1e-7]
    checkpoints = {}
    for t in (27, 60, 100, 1000):
        row = min((r for r in rows if r["time_s"] >= t), key=lambda r: r["time_s"])
        checkpoints[str(t)] = row
    data = dict(source=str(IBER), version="3.4.0", mesh=geometry(polygons), nodes=nv,
                boundary=bdata, executed=dict(backend=cpu, threads=1, cfl=float(header[4]),
                tvd=int(header[3]), max_dt_s=float(header[0]), output_interval_s=float(header[2]),
                dry_depth_m=float(header[5]), bed_manning=sorted(set(float(c[4]) for c in records)),
                wall_manning=.015, sediment=False, turbulence=False,
                initial_volume_m3=float(re.search(r"Initial volume:\s*([\d.]+)", logtext).group(1)),
                inlet="Total discharge Critical/Subcritical (-41), depth not prescribed, 11 m3/s",
                outlet="Supercritical/Critical (-40)", problemdata=pd),
                saved_gui={k:v for k,v in questions.items() if any(k.split("#")[0] == s for s in selections)},
                reported_dt_first27s=stats([r["reported_dt_s"] for r in early]),
                log_checkpoints=checkpoints, log=rows,
                total_computational_seconds_reported=int(re.search(r"Total computational time\s*=\s*(\d+)", logtext).group(1)),
                log_note="Rows are approximately one simulated second apart, not every step. Fourth clock field is hundredths. dt is the separate Time step column. Clock difference is reported run wall elapsed, not matched BRAIDS worker timing.")
    return data, polygons


def project_settings(path):
    doc = read(path); s = doc["settings"]
    boundary = []
    # Read-only geometry object: no mesh construction or simulation.
    sys.path.insert(0, str(ROOT))
    from braids_core.boundaries.domain import PolygonDomain, BoundaryAssignment
    domain = PolygonDomain(**s["domain"])
    for b in domain.boundary_segments([BoundaryAssignment(**a) for a in s["assignments"]]):
        if b.kind != "wall":
            boundary.append(dict(kind=b.kind, bc_id=b.bc_id, width_m=b.length_m,
                                 line_xy=[b.start_xy, b.end_xy]))
    grid = s["grid"]; ratio = 2**(s["levels"]-1)
    return dict(path=str(path), schema=doc["schema"], execution=doc["model"]["request"]["execution"],
                controls=s["controls"], roughness=doc.get("roughness_parameters"),
                initial=doc.get("initial"), hydraulics=s["hydraulics"], boundary=boundary,
                domain_area_m2=domain.area_m2, domain_xy=domain.vertices_xy,
                grid=grid, levels=s["levels"], nominal_finest_dx_m=grid["dx_m"]/ratio,
                adaptive=s.get("adaptive"), adaptation=s.get("adaptation"),
                dt_max_s=s["dt_max_s"], output_interval_s=s["output_interval_s"],
                end_time_s=s["end_time_s"])


def run_data(runpath):
    context = read(runpath / "RUN_CONTEXT.json")
    index = read(runpath / "frames/INDEX.json")
    epochs = {}; polygons = {}
    for name in dict.fromkeys(r["epoch"] for r in index["frames"]):
        with np.load(identify(runpath / "frames" / name), allow_pickle=False) as f:
            polys = [Polygon(f["vertices"][i:j]) for i, j in zip(f["offsets"][:-1], f["offsets"][1:])]
            assert np.allclose([p.area for p in polys], f["area"], rtol=1e-9, atol=1e-6)
            polygons[name] = polys
            epochs[name] = geometry(polys)
    frames = []
    for row in index["frames"]:
        with np.load(identify(runpath / "frames" / row["file"]), allow_pickle=False) as f:
            h = f["q"][0]
            areas = np.array([p.area for p in polygons[row["epoch"]]])
            frames.append(dict(time_s=row["time_s"], mesh=row["epoch"], cells=len(h),
                wet_cells_1mm=int((h > .001).sum()), volume_m3=float(h @ areas),
                wet_cells_1micrometre=int((h > 1e-6).sum())))
    status = read(runpath / "STATUS.json")
    adapt = read(runpath / "ADAPTATION.json") if (runpath / "ADAPTATION.json").exists() else []
    active = [r.get("to_cells", len(r.get("leaves", []))) for r in adapt] or [r["cells"] for r in frames]
    n = status.get("hydraulic", {}).get("accepted_substeps")
    return dict(path=str(runpath), runtime_id=context.get("runtime_id"), project_id=context.get("project_id"),
                status=status, mean_accepted_substep_s=status["time_s"]/n if n else None,
                active_cell_min=min(active), active_cell_max=max(active), frames=frames, mesh_epochs=epochs), polygons[index["frames"][-1]["epoch"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    identify(Path(__file__))
    identify(ROOT/"validation/sediment/river_morphodynamics/reports/unified_qgis_v1/speed130/REPORT.md")
    iber, ip = iber_data(); idomain = unary_union(ip)
    result = dict(scope="Read-only audit, no solver or input edits, no new simulation", iber=iber, braids={})
    plots = {"Iber triangles": ip}
    plot_bounds = {"Iber triangles": [dict(kind=k, line_xy=line)
                   for k,b in iber["boundary"].items() for line in b["lines_xy"]]}
    cases = [
        ("alpha132_cpu5m", ROOT/"validation/_amr132/clean_final/cpu5m_PROJECT.json", ROOT/"validation/_amr132/clean_final/cpu5m_1_candidate/run"),
        ("alpha132_gpu5m", ROOT/"validation/_amr132/clean_final/gpu5m_PROJECT.json", ROOT/"validation/_amr132/clean_final/gpu5m_1_candidate/run"),
        ("alpha132_gpu2m", ROOT/"validation/_amr132/clean_final/gpu2m_PROJECT.json", ROOT/"validation/_amr132/clean_final/gpu2m_1_candidate/run"),
        ("aaa_alpha127_cpu5m", AAA/"prepare_20261022_204710_d0ea8c.project.json", AAA/"run_20261022_204720_f34397"),
        ("aaa_alpha127_gpu2m", AAA/"prepare_20261022_204320_bba003.project.json", AAA/"run_20261022_204331_367153"),
        ("saved_1m_implicit_gpu", DOCS/"1m.json", DOCS/"runs/run_20261022_114753_046840"),
    ]
    for key, project, run in cases:
        p = project_settings(project); r, polys = run_data(run)
        dom = Polygon(p["domain_xy"])
        result["braids"][key] = dict(project=p, run=r, domain_vs_iber=dict(
            intersection_m2=dom.intersection(idomain).area,
            symmetric_difference_m2=dom.symmetric_difference(idomain).area,
            iber_fraction_shared=dom.intersection(idomain).area/idomain.area))
        if key in ("alpha132_cpu5m", "alpha132_gpu2m", "saved_1m_implicit_gpu"):
            title = {"alpha132_cpu5m":"BRAIDS 5 m AMR (60 s)", "alpha132_gpu2m":"BRAIDS 2 m static", "saved_1m_implicit_gpu":"Saved 1 m implicit (3.75 s)"}[key]
            plots[title] = polys
            plot_bounds[title] = p["boundary"]
        if key == "saved_1m_implicit_gpu":
            retained = read(DOCS/"runs/prepare_20261022_114735_df4b84.project.json")
            saved = read(project)
            result["braids"][key]["saved_project_settings_exact_match_retained_run"] = saved["settings"] == retained["settings"]
            result["braids"][key]["scope_note"] = "Historical implicit project/run, cancelled at 3.75 s; not alpha132 or an admitted current solver option."
    result["alpha132_clean_results"] = read(ROOT/"validation/_amr132/RESULTS.json")
    profile = read(ROOT/"validation/_amr132/profile_after/cpu5m_0_candidate/PROFILE.json")
    simple = []
    for row in profile["steps"]:
        d = row["diagnostics"]
        simple.append(dict(time_s=row["time_s"], outer_dt_s=row["dt"], substeps=d["hydraulic_substeps"],
                           minimum_dt_s=d["hydraulic_minimum_dt"], maximum_dt_s=d["hydraulic_maximum_dt"],
                           next_hint_s=d["hydraulic_next_substep_hint_s"]))
    early = [r for r in simple if r["time_s"] <= 27]
    result["alpha132_cpu_profile"] = dict(steps=simple, restrictions=profile["restrictions"],
        phases=profile["phases"], worker_s=profile["worker_wall_s"],
        first27_substeps=sum(r["substeps"] for r in early),
        first27_mean_dt_s=27/sum(r["substeps"] for r in early),
        note="Profiling timings have instrumentation overhead; clean samples govern speed claims. Per-transaction min/max are not individual substep traces.")
    result["hardware_scope"] = "Retained speed130 identifies Ryzen 9950X3D / RTX 5080. Current alpha132 CPU inputs request eight threads; Iber requests one CPU thread. Iber output itself does not identify processor hardware. No matched same-mesh cross-solver benchmark exists."

    # Whole-domain mesh comparison keeps the same extent and coordinate origin.
    origin = np.array([2571200., 1106650.])
    fig, axes = plt.subplots(1, len(plots), figsize=(15.5, 6.3), layout="constrained")
    colors = ["#5b6470", "#197d96", "#ad663b", "#805ca6"]
    for ax, (name, polys), color in zip(axes, plots.items(), colors):
        ax.add_collection(PolyCollection([np.array(p.exterior.coords)-origin for p in polys],
                          facecolors="none", edgecolors=color, linewidths=.25))
        for boundary in plot_bounds[name]:
            xy = np.asarray(boundary["line_xy"])-origin
            ax.plot(*xy.T, color="#e04040" if boundary["kind"] == "inlet" else "#1844bf", lw=2)
        ax.set(xlim=(20,430), ylim=(0,930), aspect="equal", title=f"{name}\n{len(polys):,} active cells",
               xlabel="East from 2,571,200 m")
        ax.set_ylabel("North from 1,106,650 m")
    fig.suptitle("Actual saved meshes: common map extent; red = inlet, blue = outlet")
    fig.savefig(OUT/"ACTUAL_MESHES.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for (name, polys), color in zip(plots.items(), colors):
        area = np.array([p.area for p in polys]); per = np.array([p.length for p in polys])
        for ax, values in zip(axes[:2], (np.sqrt(area), 2*area/per)):
            x = np.sort(values); y = np.arange(1,len(x)+1)/len(x)
            ax.plot(x, y, label=name, color=color)
    axes[0].set(xscale="log", xlabel="Square root of cell area (m)", ylabel="Fraction of active cells", title="Cell size distribution")
    axes[1].set(xscale="log", xlabel="2A / perimeter (m)", ylabel="Fraction of active cells", title="Geometric CFL scale (not a formula match)")
    for ax in axes[:2]: ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, loc="lower right")
    rows = [r for r in iber["log"] if 0 < r["time_s"] < 60.5]
    axes[2].plot([r["time_s"] for r in rows], [r["reported_dt_s"] for r in rows], ".-", color=colors[0], lw=1, label="Iber: reported dt, CFL .45")
    axes[2].fill_between([r["time_s"] for r in simple], [r["minimum_dt_s"] for r in simple],
                        [r["maximum_dt_s"] for r in simple], color=colors[1], alpha=.2, label="BRAIDS 5 m: min/max within 1 s")
    axes[2].plot([r["time_s"] for r in simple], [r["next_hint_s"] for r in simple], color=colors[1], label="BRAIDS next hint, face-sum CFL 1")
    axes[2].set(xlabel="Simulated time (s)", ylabel="Hydraulic timestep (s)", title="Current 5 m timestep is comparable")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=.2)
    fig.suptitle("Read-only comparison; differing domains, cells and CFL definitions")
    fig.savefig(OUT/"CELL_SIZE_AND_TIMESTEP.png", dpi=180); plt.close(fig)
    result["source_sha256"] = INPUTS.copy()
    result["source_integrity_after"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in INPUTS.items())
    assert result["source_integrity_after"]
    (OUT/"AUDIT.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(OUT), source_integrity_after=result["source_integrity_after"],
        first27=iber["log_checkpoints"]["27"],
        braids={k:dict(cells=[v["run"]["active_cell_min"],v["run"]["active_cell_max"]],
                     mean_dt=v["run"]["mean_accepted_substep_s"],runtime=v["run"]["runtime_id"])
                for k,v in result["braids"].items()}, first27_cpu=result["alpha132_cpu_profile"]["first27_mean_dt_s"]),indent=2))


if __name__ == "__main__":
    main()
