"""Represented native inlet openings, independent of the source-DEM preview.

User-facing section controls use Q / sum(L * max(eta-z_face, 0)) and retain
that resolved horizontal water line across all submerged portions of the inlet.
Only legacy sampled mode-1 decks select a connected minimum. The older mode-1 discharge trace may
instead retain an outgoing characteristic after wetting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from braids_core.boundaries.cut_cell_io import NATIVE_DRY_TOLERANCE_M

# Mirrors solver/cut_cells/geoclaw_parameters.f90; no configurable second cap.
SPEED_LIMIT_M_S = 50.
DRY_TOLERANCE_M = 1.e-8
DIAGNOSTICS_FILE = "inlet_diagnostics.json"


class InletPreparationError(ValueError):
    def __init__(self, message, run_root):
        super().__init__(message)
        self.run_root = Path(run_root)


def wetted_depths(fragments, stage_m, *, connected=False):
    depth = np.maximum(stage_m - np.array([f["bed_m"] for f in fragments]), 0.)
    if connected and len(depth):
        seed=int(np.argmin([f["bed_m"] for f in fragments]))
        if depth[seed]<=0:return np.zeros_like(depth)
        left=right=seed
        while left>0 and depth[left-1]>0:left-=1
        while right+1<len(depth) and depth[right+1]>0:right+=1
        depth[:left]=0.;depth[right+1:]=0.
    return depth


def opening_at(fragments, stage_m, discharge_m3_s, *, connected=False, dry_tolerance_m=0.):
    depth = wetted_depths(fragments,stage_m,connected=connected)
    depth[depth<=dry_tolerance_m]=0.
    length = np.array([f["length_m"] for f in fragments])
    area = float(np.dot(depth, length))
    return {"stage_m": float(stage_m), "discharge_m3_s": float(discharge_m3_s),
            "area_m2": area, "wet_length_m": float(length[depth > 0].sum()),
            "maximum_depth_m": float(depth.max(initial=0.)),
            "implied_velocity_m_s": float(discharge_m3_s / area) if area > 0 else
                (0. if discharge_m3_s == 0 else None),
            "positive_discharge_without_opening": bool(discharge_m3_s > 0 and area <= 0)}


def represented_critical_outlets(mesh, bed, initial, boundaries, metadata):
    """Resolve each initially wet critical outlet exactly as the native trace does."""
    entries={};errors=[];gravity=9.81;machine=np.finfo(float).eps
    for name,spec in boundaries.items():
        if spec.mode!="critical":continue
        selected=sorted((face for face in mesh.faces
            if face.neighbour<0 and face.bc_id==name),key=lambda face:face.face_id)
        entry={"numeric_id":metadata["string_to_numeric_id"].get(name),
            "requested_mode":spec.mode,"face_count":len(selected),
            "selected_length_m":float(sum(face.length_m for face in selected))}
        entries[name]=entry
        if not selected:
            entry["status"]="rejected_missing_faces";errors.append(
                f"Critical-depth outlet {name!r} has no represented polygon face");continue
        owners=np.array([face.owner for face in selected],dtype=int)
        boundary_bed=np.asarray(bed,dtype=float)[owners]
        depth=np.asarray(initial,dtype=float)[owners,0]
        momentum=np.asarray(initial,dtype=float)[owners,1:3]
        normals=np.asarray([face.outward_normal for face in selected],dtype=float)
        lengths=np.asarray([face.length_m for face in selected],dtype=float)
        wet=depth>DRY_TOLERANCE_M
        entry["initial_wet_face_count"]=int(np.count_nonzero(wet))
        entry["maximum_initial_depth_m"]=float(depth.max(initial=0.))
        if not wet.any():
            entry.update(status="initially_dry_closed",critical_stage_m=None,
                critical_discharge_m3_s=0.,maximum_trace_speed_m_s=0.)
            continue
        normal_momentum=np.einsum("ij,ij->i",momentum,normals)
        tolerance=128*machine
        if np.any(normal_momentum[wet] < -tolerance*np.maximum(1.,np.abs(normal_momentum[wet]))):
            entry["status"]="rejected_initial_inflow"
            errors.append(f"Critical-depth outlet {name!r} has inward flow at t = 0 s. "
                "A critical outlet can only control outward flow");continue
        inward_velocity=np.zeros(len(selected));inward_velocity[wet]=-normal_momentum[wet]/depth[wet]
        wave_speed=np.sqrt(gravity*np.maximum(depth,0.))
        if np.any(inward_velocity[wet] < -wave_speed[wet]*(1.+1.e-10)):
            entry["status"]="rejected_supercritical_outflow"
            errors.append(f"Critical-depth outlet {name!r} is already supercritical at t = 0 s. "
                "Use the supercritical outlet behavior for this face");continue
        invariant=np.zeros(len(selected));invariant[wet]=inward_velocity[wet]-2*wave_speed[wet]

        def state_at(stage):
            candidate_depth=np.maximum(stage-boundary_bed,0.)
            candidate_velocity=invariant+2*np.sqrt(gravity*candidate_depth)
            discharge=float(-np.dot(candidate_depth*candidate_velocity,lengths))
            area=float(np.dot(candidate_depth,lengths));top_width=float(lengths[candidate_depth>0].sum())
            critical=0. if top_width<=0 else float(np.sqrt(gravity*area**3/top_width))
            return discharge-critical,discharge,critical,candidate_depth,candidate_velocity

        lower=float(boundary_bed[wet].min())
        lower+=max(2*DRY_TOLERANCE_M,128*machine*max(1.,abs(lower)))
        upper=float(np.max(boundary_bed[wet]+depth[wet]))
        residual_lower=state_at(lower)[0];residual_upper=state_at(upper)[0]
        if residual_lower<0:
            entry["status"]="rejected_no_critical_root"
            errors.append(f"Critical-depth outlet {name!r} has no admissible initial critical-flow root");continue
        for _ in range(60):
            if residual_upper<=0:break
            upper=lower+2*(upper-lower);residual_upper=state_at(upper)[0]
        else:
            entry["status"]="rejected_no_critical_root"
            errors.append(f"Critical-depth outlet {name!r} has no bounded initial critical-flow root");continue
        for _ in range(90):
            stage=.5*(lower+upper)
            if state_at(stage)[0]>0:lower=stage
            else:upper=stage
        stage=.5*(lower+upper);_,discharge,_,candidate_depth,candidate_velocity=state_at(stage)
        candidate_wet=candidate_depth>DRY_TOLERANCE_M
        if np.any(candidate_velocity[candidate_wet]>tolerance):
            entry["status"]="rejected_reverse_trace"
            errors.append(f"Critical-depth outlet {name!r} produces inward flow in its initial trace");continue
        tangential=np.zeros_like(momentum)
        velocity=np.zeros_like(momentum);velocity[wet]=momentum[wet]/depth[wet,None]
        outward_velocity=np.einsum("ij,ij->i",velocity,normals)
        tangential[wet]=velocity[wet]-outward_velocity[wet,None]*normals[wet]
        speed=np.sqrt(candidate_velocity**2+np.einsum("ij,ij->i",tangential,tangential))
        maximum=float(speed[candidate_wet].max(initial=0.))
        entry.update(critical_stage_m=float(stage),critical_discharge_m3_s=discharge,
            maximum_trace_speed_m_s=maximum)
        if maximum>SPEED_LIMIT_M_S:
            entry["status"]="rejected_trace_speed"
            errors.append(
                f"Critical-depth outlet {name!r} at t = 0 s requires a {maximum:.6g} m/s local "
                f"boundary trace, above the native {SPEED_LIMIT_M_S:g} m/s safety limit. Its "
                f"represented {entry['selected_length_m']:.6g} m section would release "
                f"{discharge:.6g} m³/s at critical flow. Select only the physical outlet opening, "
                "use a prescribed water-level outlet, or delete the outlet assignment when the "
                "reservoir has no outlet. Reducing the timestep cannot correct this boundary state")
        else:entry["status"]="initial_screen_passed"
    return entries,errors


def represented_inlets(mesh, bed, initial, boundaries, metadata, end_time_s):
    """Keep each logical ID's Q once, across all its polygon face fragments.

Inspect schedule endpoints/knots, bed wetting thresholds and their interval
midpoints on the initial terrain. Between thresholds both Q and A are linear;
midpoints also expose the finite one-sided Q/A limit beside a 0/0 endpoint.
This is an input diagnostic, not a prediction of the evolving hydraulic trace.
"""
    entries = {}
    errors = []
    for name, spec in boundaries.items():
        if not spec.is_inlet:
            continue
        target = metadata["resolved_boundary_targets"][name]
        entry = {"numeric_id": metadata["string_to_numeric_id"][name],
                 "requested_mode": spec.mode, "native_mode": target["native_mode"],
                 "fragments": [], "samples": [], "warnings": []}
        entries[name] = entry
        distance = 0.
        all_incoming = True
        selected = sorted((f for f in mesh.faces if f.neighbour < 0 and f.bc_id == name),
                          key=lambda f: (f.edge_id, f.edge_start_m or 0., f.face_id))
        sampled=bool(selected) and all(getattr(f,"bed_elevation_m",None) is not None for f in selected)
        production=bool(metadata.get("boundary_terrain",{}).get("live_controls",False)) or any(
            getattr(f,"section_bed_start_m",None) is not None for f in selected)
        connected=sampled and not production and target["native_mode"]!=9
        entry["connected_from_lowest"]=connected
        entry["wetting_policy"]="connected" if connected else "all_submerged"
        if sampled:
            normal=selected[0].outward_normal;tangent=np.array([-normal[1],normal[0]])
            if tangent[0]<0 or (tangent[0]==0 and tangent[1]<0):tangent=-tangent
            selected.sort(key=lambda f:np.dot(np.array(f.start_xy)+f.end_xy,tangent))
        for face in selected:
            h = float(initial[face.owner, 0])
            velocity = -float(np.dot(initial[face.owner, 1:3], face.outward_normal)) / h if h > DRY_TOLERANCE_M else 0.
            if h > DRY_TOLERANCE_M and getattr(face,"bed_elevation_m",None) is not None:
                h=max(h+bed[face.owner]-face.bed_elevation_m,0.)
            incoming = h <= DRY_TOLERANCE_M or velocity >= np.sqrt(9.81*h)*(1.-1.e-12)
            all_incoming = all_incoming and incoming
            entry["fragments"].append({"face_id": face.face_id, "owner": face.owner,
                "edge_id": face.edge_id, "start_xy": list(face.start_xy), "end_xy": list(face.end_xy),
                "length_m": face.length_m, "distance_start_m": distance,
                "distance_end_m": distance+face.length_m,
                "bed_m": float(bed[face.owner] if getattr(face,"bed_elevation_m",None) is None else face.bed_elevation_m),
                "owner_bed_m":float(bed[face.owner]),
                "initial_incoming_branch": bool(incoming)})
            distance += face.length_m
        entry["selected_length_m"] = distance
        entry["all_initial_faces_incoming"] = bool(all_incoming)
        if target["native_mode"] not in (1,9):
            entry["status"] = "native_runtime_check_required"
            entry["warnings"].append("This mode uses a different velocity distribution; Q/A screening is not applicable.")
            continue
        series = metadata["resolved_hydraulic_series"][name]
        knots = series["knots"] if series else [{"time_s": 0.,
            "discharge_m3_s": spec.discharge_m3_s, "target_stage_m": target["target_stage_m"]}]
        times = [k["time_s"] for k in knots]
        stages = [k["target_stage_m"] for k in knots]
        flows = [k["discharge_m3_s"] for k in knots]
        inlet_dry_tolerance=NATIVE_DRY_TOLERANCE_M if target["native_mode"]==9 else 0.
        entry["represented_dry_tolerance_m"]=inlet_dry_tolerance
        unresolved_stages = [stage is None for stage in stages]
        if any(unresolved_stages):
            if not all(unresolved_stages):
                entry["status"] = "rejected_inconsistent_live_section_schedule"
                errors.append(
                    f"Inlet {name!r} mixes prescribed and live-resolved stages in one schedule"
                )
                continue
            # Native mode 9 normal/critical controls deliberately carry no
            # prescribed stage.  The solver reconstructs the opening, stage,
            # velocity and Froude state from the current bed section at every
            # boundary update.  A static Q/A calculation here would duplicate
            # that nonlinear solve and, more importantly, would become stale as
            # morphology changes the inlet.
            sample_times = sorted({
                0., float(end_time_s),
                *(float(t) for t in times if 0 <= t <= end_time_s),
            })
            for time in sample_times:
                entry["samples"].append({
                    "time_s": time,
                    "discharge_m3_s": float(np.interp(time, times, flows)),
                    "stage_m": None,
                    "area_m2": None,
                    "wet_length_m": None,
                    "maximum_depth_m": None,
                    "implied_velocity_m_s": None,
                    "positive_discharge_without_opening": None,
                    "above_speed_limit": None,
                    "resolved_by_native_live_section": True,
                })
            if not entry["fragments"] and any(flow > 0 for flow in flows):
                entry["status"] = "rejected_missing_faces"
                errors.append(f"Inlet {name!r} has positive discharge but no represented polygon face")
            else:
                entry["status"] = "native_live_section_check_required"
                entry["warnings"].append(
                    "Stage, opening, velocity and Froude state are resolved by the native solver "
                    "from the evolving inlet section; the static preparation screen records only "
                    "the requested discharge schedule."
                )
            continue
        sample_times = {0., float(end_time_s), *(float(t) for t in times if 0 <= t <= end_time_s)}
        for left, right in zip(knots, knots[1:]):
            dz = right["target_stage_m"] - left["target_stage_m"]
            if dz == 0:
                continue
            for fragment in entry["fragments"]:
                fraction = (fragment["bed_m"]+inlet_dry_tolerance - left["target_stage_m"]) / dz
                time = left["time_s"] + fraction*(right["time_s"]-left["time_s"])
                if 0 < fraction < 1 and 0 <= time <= end_time_s:
                    sample_times.add(float(time))
        breaks = sorted(sample_times)
        sample_times.update((a+b)/2 for a, b in zip(breaks, breaks[1:]))
        for time in sorted(sample_times):
            row = opening_at(entry["fragments"], np.interp(time, times, stages), np.interp(time, times, flows),
                             connected=connected,dry_tolerance_m=inlet_dry_tolerance)
            row["time_s"] = time
            row["above_speed_limit"] = row["implied_velocity_m_s"] is not None and row["implied_velocity_m_s"] > SPEED_LIMIT_M_S
            entry["samples"].append(row)
        first = entry["samples"][0]
        strong_section=target["native_mode"]==9
        invalid = first["positive_discharge_without_opening"] or ((strong_section or all_incoming) and first["above_speed_limit"])
        entry["status"] = "rejected_initial_inlet" if invalid else "initial_screen_passed"
        if invalid:
            speed = first["implied_velocity_m_s"]
            detail = "no wetted opening" if speed is None else f"Q/A = {speed:.6g} m/s exceeds {SPEED_LIMIT_M_S:g} m/s"
            errors.append(f"Inlet {name!r} at t = 0 s: {detail} (Q = {first['discharge_m3_s']:.6g} m³/s, "
                          f"represented area = {first['area_m2']:.6g} m², stage = {first['stage_m']:.12g} m). "
                          "Review the inlet profile, level and mesh; discharge and level were not changed.")
        if any(row["above_speed_limit"] or row["positive_discharge_without_opening"] for row in entry["samples"]):
            entry["warnings"].append("Some scheduled targets exceed the speed limit or have no opening on the initial bed. "
                "The native solver checks the resolved boundary state at every stage.")
    outlets,outlet_errors=represented_critical_outlets(mesh,bed,initial,boundaries,metadata)
    errors.extend(outlet_errors)
    return {"format": "braids-boundary-preflight-v2", "speed_limit_m_s": SPEED_LIMIT_M_S,
            "scope": "Prepared boundary terrain (DEM strips when supplied, otherwise owner-cell bed); Q/A is the executed velocity for user-facing section controls and the incoming target for legacy characteristic mode. "
                     "Cumulative opening length joins the selected fragments for display, not a surveyed transect. "
                     "An initial screen pass does not qualify future speeds or model accuracy.",
            "inlets": entries,"outlets":outlets,"errors":errors}


def load_inlet_diagnostics(run_root):
    return json.loads((Path(run_root) / DIAGNOSTICS_FILE).read_text(encoding="utf-8"))
