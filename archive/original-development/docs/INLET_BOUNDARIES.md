# G3 inlet API and native boundary format

This document describes the preserved G3 interface. The current reservoir-filling implementation adds the isolated [G4 boundary overlay and exporter](../validation/hydraulics/channels/README.md), including exact interface discharge, explicit dry-aperture walls and the conservative fixed-grid positivity correction. Its [focused filling report](../validation/hydraulics/channels/reports/FOCUSED_RESERVOIR_FILLING.md) is the current demonstration; it is still a headless workflow, not the QGIS polygon-edge interface.

BRAIDS can construct inlet targets in Python and export them to the fixed-grid native WAVE boundary module. Every inlet has a stable string ID and a nonnegative discharge hydrograph. A target water elevation, the compatible boundary state, and the numerical flux accepted by the solver are distinct quantities.

The implementation is in [inlet.py](../braids_core/boundaries/inlet.py), [inlet_io.py](../braids_core/boundaries/inlet_io.py), and [braids_boundary_module.f90](../solver/wave/braids_boundary_module.f90). Cross-section geometry and GIS extraction are described in [SECTION_SERVICE.md](SECTION_SERVICE.md).

## Current scope and evidence

The native interface supports Cartesian, axis-aligned exterior boundaries on fixed AMR level 1, including a domain divided into several patches. It requires three conservative shallow-water state variables, the original water fluctuation path, and matching metre coordinates. The Python exporter currently requires gravity 9.81 m/s²; configure the native solver with that same value.

Each inlet occupies one side and a contiguous sequence of whole native faces. Its exported active profile must be one convex basin. Native face beds must be unimodal and at or above the continuous profile elevation at each face midpoint, within 1e-9 m. The actual native adjacent-cell bed must match the declared face bed. Cell-average topography may therefore differ from the continuous surveyed profile. The exported profile is clipped at the section's declared maximum stage, which must be enclosed by surveyed banks.

Arbitrary oblique lines, cut faces, AMR refinement/reflux, compound native sections, and per-face native Manning values are outside this interface. G2 can load GIS sections; converting arbitrary QGIS polylines into these native face maps remains separate work. Inlet coefficients do not configure spatial solver friction. Outlets and sediment transport are separate features.

The focused suite has **213 passing tests**: 108 pure target tests, 53 exporter tests, and 52 tests that compile the actual Fortran module against small AMR storage fixtures. These verify target roots, nonzero datums, outgoing characteristics, regime policies, signed local backflow, ownership, and lifecycle isolation. The separate [native matrix](../validation/hydraulics/inlets/reports/native_results.json) adds 29 passing full-solver runs, including accepted fluxes, checkpoint continuation and thread equivalence. The [combined verification report](../validation/hydraulics/inlets/reports/G3_VERIFICATION_REPORT.md) binds the tests and native evidence to current source/build/input hashes. Qualification applies only to the fixed-grid scope above; longer channel and reservoir accuracy remains G4/G5 work.

```sh
.venv/bin/python -m pytest -q tests/test_inlet.py tests/test_inlet_io.py tests/test_native_inlet_boundary.py
```

## Python requests and targets

The main request, target and native export types are available from `braids_core.boundaries` and their individual modules. Request, hydrograph, face, and target records are immutable.

| Type or function | Contract |
| --- | --- |
| `Hydrograph(times_s, discharges_m3_s, endpoint_policy="hold")` | Matching, nonempty finite sequences; strictly increasing times; finite nonnegative Q. One knot is allowed. |
| `Hydrograph.discharge(time_s)` | Piecewise-linear interpolation. Exact endpoints retain their knot values. Outside the interval, `hold` keeps the nearest endpoint Q, `zero` returns zero, and `error` rejects the query. |
| `InletSpec(bc_id, mode, hydrograph, value_m=None, manning_n=.03, energy_slope=.001, gravity_m_s2=9.81, allocation="uniform")` | Nonempty ID without whitespace; an explicit hydrograph for every mode. Physical coefficients must be finite and positive. |
| `Face(global_key, station_start_m, station_end_m, bed_elevation_m, inward_normal, length_m=None, manning_n=None)` | One whole physical face. Increasing station span must equal native length. Explicit inward normal is one of the four axis unit vectors. Optional local n affects Python conveyance allocation only. |
| `build_inlet_target(spec, section, faces, time_s)` | Returns the requested elevation, per-face `h, hu, hv`, requested and integrated state Q, continuous/discrete areas, connectivity diagnostics, and an optional hydraulic root result. It does not change solver cells or a water budget. |
| `build_inlet_targets(specs, sections, faces_by_id, time_s)` | Checks unique inlet IDs and global face ownership, then returns targets keyed by ID. |
| `register_face_ownership(faces_by_id)` | Returns physical key → ID and rejects duplicate ownership. Native export additionally checks side/index ownership, catching aliases with different human keys. |

The four target modes are:

| Mode | Target elevation |
| --- | --- |
| `stage` | `value_m`, an absolute elevation in the same vertical reference as the bed. |
| `depth` | `Section.datum_m + value_m`; the nonnegative value is depth above that fixed datum. |
| `normal` | The Section Manning root at Q, using `manning_n` and positive `energy_slope`. Omit `value_m`. |
| `critical` | The Section root of `Q² T / (g A³) = 1`. Omit `value_m`. |

Depth is not independently added above each face bed. There is one horizontal target elevation, and local water depth follows from that elevation and the represented bed. Choose an explicit datum when a raster's sampled minimum differs from the physical section invert.

Faces must tile the full section interval without gaps, overlaps, or endpoint cuts. Input face order may be reversed; canonical station order and explicit normals determine the result. The caller must preserve the physical station-to-face correspondence.

At the target elevation, the pure mapper selects faces whose midpoints belong to the connected wet interval. Dry or disconnected faces receive zero depth and momentum. `connected_fraction` records how much of each face intersects that interval; it does not silently shorten the native face. A partially wet face is represented by its midpoint eligibility, constant native bed, and full native length. Grid refinement reduces this explicit geometry approximation.

The continuous area `A` comes from the sampled section. The actual represented area is `A_d = Σ h_i L_i`. Uniform allocation uses `u_i = Q / A_d`. Conveyance allocation uses `u_i ∝ h_i^(2/3) / n_i`, normalized with `Σ h_i u_i L_i = Q`. This local wide-strip allocation is not a compound-channel roughness model. Positive Q without represented wet area fails; no artificial water film is introduced.

For Q=0 the pure mapper assigns zero momentum without division by Q or dry area. Stage/depth modes retain their requested target geometry; normal/critical targets use the section minimum.

## Compatible native states and explicit policies

For a wet subcritical inlet face, let positive `u_n` point into the domain. The native solver retains the outgoing interior invariant

```text
R−_i = u_n,interior,i − 2 sqrt(g h_interior,i)
u_n,i(eta) = R−_i + 2 sqrt(g max(eta − z_i, 0))
```

It gathers every face of an ID across all patches and solves one compatible elevation so that `Σ h_i(eta) u_n,i(eta) L_i = Q`. Consequently, a drowned normal-depth or prescribed-stage target can differ from the compatible elevation. The two values are reported separately. Prescribing both Q and the target elevation independently on wet subcritical faces would discard outgoing information.

The native Q is the exact piecewise-linear hydrograph average over the proposed interval, split at knots and endpoint discontinuities. Normal/critical target elevations are recomputed from that Q; the solver does not interpolate precomputed nonlinear stage roots. Uniform/conveyance target allocation controls explicitly admitted incoming target states. Wet subcritical velocities follow outgoing compatibility.

The exporter accepts each policy as a string for all IDs or as a dictionary keyed by ID:

| Keyword | Default | Explicit alternative and limits |
| --- | --- | --- |
| `dry_policy` | `error` | `target` permits the requested incoming state to wet a dry face. Dry incoming states are an explicit policy; they have no wet interior invariant to preserve. |
| `supercritical_policy` | `error` | `target` permits an incoming target branch. A wet target branch must remain supercritical in its final state. Wet supercritical-to-subcritical transitions, including a promoted subcritical face whose target becomes subcritical, are rejected. |
| `backflow_policy` | `error` | `allow` permits signed subcritical local face flow while retaining the prescribed nonnegative section total. Negative-flow faces retain interior tangential velocity and outgoing `R−`; positive-flow faces use zero tangential target velocity. Supercritical outward flow remains rejected. |

These policies do not bypass surveyed-bank limits, unsupported regimes, native speed limits, or missing discrete wet area. Local backflow does not authorize a negative inlet hydrograph.

For Q=0, the native boundary closes normal transport and derives each wet face's closed trace from outgoing information. Those local depths can differ from the Python target and need not share a common water elevation. The diagnostic `compatible_eta_m` then contains the midpoint of the actual wet-elevation range (or the target when all faces are dry); use `wet_eta_min_m` and `wet_eta_max_m` to inspect the full range.

The native lifecycle separates global preparation from an accepted update: `braids_boundary_prepare(level,time,dt,nvar,naux)`, then `braids_boundary_begin_step(time,dt)`, patch ghost/flux operations, and finally `braids_boundary_commit(time,dt,restarting)`. Preparation uses physical state/auxiliary counts, not output-component counts. Numerical flux arrays remain unchanged while unarmed. Commit clears the prepared trace, so later output or initialization fills cannot reuse it. The full solver integration owns this lifecycle.

## Native data and diagnostics

```python
write_native_inlets(
    path, grid, specs, sections, native_faces_by_id,
    dry_policy="error", supercritical_policy="error", backflow_policy="error",
)
```

`NativeGrid(nx, ny, xlower_m, ylower_m, dx_m, dy_m)` declares the whole fixed domain. `NativeFace(face, side, global_index)` binds a whole face to that grid. Sides west/east/south/north use codes 1/2/3/4 and inward normals `(1,0)`, `(-1,0)`, `(0,1)`, `(0,-1)`. Zero-based transverse indices increase with y on west/east and x on south/north. Each ID uses contiguous increasing indices. Grid offsets and indices refer to physical locations, independently of the polyline's digitization direction.

The writer returns metadata and writes the numeric file plus `<path>.json`, for example `braids_boundary.data.json`. Sorted string IDs receive deterministic numeric IDs 1…N. The sidecar preserves both ID mappings, specifications, exported profiles, native face records, knot target diagnostics, policy codes, and the numeric file's SHA-256.

The current **pre-release format v1** has the following layout. Its boundary header includes the new backflow field; regenerate earlier development decks with the exporter.

```text
version
nx ny xlower ylower dx dy
number_of_boundaries number_of_faces

# Repeat for every boundary:
numeric_id mode allocation endpoint dry_policy supercritical_policy backflow_policy np nk value_m datum_m manning_n slope
# np profile rows:
station_m bed_elevation_m
# nk hydrograph rows:
time_s discharge_m3_s

# After all boundary blocks, number_of_faces rows:
numeric_id side global_index native_bed_elevation_m whole_face_length_m
```

| Field | Codes or meaning |
| --- | --- |
| `mode` | normal=1, critical=2, depth=3, stage=4 |
| `allocation` | uniform=1, conveyance=2 |
| `endpoint` | hold=1, zero=2, error=3 |
| `dry_policy`, `supercritical_policy` | error=0, target=1 |
| `backflow_policy` | error=0, allow=1 |
| `np`, `nk` | Number of profile points and discharge knots. |
| `value_m`, `datum_m` | Mode value and fixed vertical datum; unused normal/critical value is zero. |
| `manning_n`, `slope` | Scalar inlet roughness and normal-depth energy slope; not a spatial friction layer. |

The native `braids_boundary_diagnostics.csv` records one row per committed ID/interval: times, numeric ID, requested and state-integrated Q, target/compatible elevations, actual wet elevation range, branch counts, outgoing-invariant error, Q residuals, and `local_backflow_faces`. Invariant residuals cover the subcritical characteristic branch; admitted dry/supercritical target branches have different information requirements. These records do not replace the accepted numerical face-flux and water-volume ledgers.

## Reproducible exporter example

This V-shaped section has invert datum 100 m and requested depth 1.5 m. Its declared native beds represent a coarser cell-average geometry. They must match the actual native topography in a solver run.

```python
from pathlib import Path
from braids_core.boundaries.section import Section
from braids_core.boundaries.inlet import Face, Hydrograph, InletSpec, build_inlet_target
from braids_core.boundaries.inlet_io import NativeFace, NativeGrid, write_native_inlets

section = Section("river_inlet_1", [0., 2., 4.], [102., 100., 102.], 100., (1., 0.))
graph = Hydrograph((0., 10.), (.4, .8), endpoint_policy="hold")
spec = InletSpec("river_inlet_1", "depth", graph, value_m=1.5, allocation="uniform")
beds = [101.5, 100.625, 100.625, 101.5]
faces = [Face(f"west:{i}", float(i), float(i + 1), z, (1., 0.))
         for i, z in enumerate(beds)]
target = build_inlet_target(spec, section, faces, time_s=0.)

metadata = write_native_inlets(
    Path("braids_boundary.data"), NativeGrid(4, 4, 0., 0., 1., 1.),
    [spec], {spec.bc_id: section},
    {spec.bc_id: [NativeFace(face, 1, i) for i, face in enumerate(faces)]},
    dry_policy="target", supercritical_policy="error", backflow_policy="error",
)
```

Executing this example gives target elevation **101.5 m**, continuous area **2.25 m²**, discrete area **1.75 m²**, and integrated requested-state discharge **0.4 m³/s** at t=0. The generated numeric file is:

```text
1
4 4 0 0 1 1
1 4
1 3 1 1 1 0 0 3 2 1.5 100 0.029999999999999999 0.001
0 102
2 100
4 102
0 0.40000000000000002
10 0.80000000000000004
1 1 0 101.5 1
1 1 1 100.625 1
1 1 2 100.625 1
1 1 3 101.5 1
```

The example generates an inlet table and its metadata. A complete simulation also needs matching native grid/topography, user boundary code 0 on the configured side, initial conditions, and the standard solver input files.
