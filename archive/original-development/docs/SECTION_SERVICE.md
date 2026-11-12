# Sampled hydraulic sections

The production implementation is [section.py](../braids_core/boundaries/section.py); [gis.py](../braids_core/boundaries/gis.py) adapts DEMs and polylines. Both are independent of the QGIS UI. The service returns a hydraulic target for later boundary-condition construction.

## Geometry and datum

Each immutable `Section` has a `bc_id`, strictly increasing station coordinates, sampled bed elevations, a fixed depth datum, and an explicit unit normal pointing into the domain. The registry and GIS loader enforce ID uniqueness across sections. Every coordinate and elevation is in metres. Stage is an absolute water elevation; `depth_m = stage_m - datum_m`. Local depth is `max(stage_m - bed_elevation_m, 0)` and varies across the section. Set the datum to the physical invert when the returned depth must mean invert depth; do not substitute a grid-dependent sampled minimum for a known survey datum.

The bed is single-valued and piecewise linear. Partial wet segments contribute their actual horizontal width, triangular/trapezoidal area, and sloping length to wetted perimeter. This excludes a grid staircase perimeter. Exact vertical walls cannot be represented by duplicate stations; use a separate future wall-aware representation when required. Default maximum stage is the lower endpoint elevation. A user may lower this limit; stages above it are rejected instead of extrapolated.

Wetting begins at the lowest basin minimum. Separate equal minima require an explicit `seed_station_m`. An optional seed may select another local basin minimum, including an enclosed flat bottom, but cannot select a draining bench. Only the water component connected to that seed contributes to section hydraulics. Other geometrically low pockets remain excluded until stage overtops the connecting crest; `excluded_wet_area_m2` records their area. At an exactly touched sill, the pockets remain separate. This static section connectivity rule does not simulate transient lateral filling or trapped water.

`geometry(stage_m)` returns area, wetted perimeter, top width, hydraulic radius, and wet station intervals. At dry stages these actual wetted quantities are zero. For zero Q, both root methods return the selected basin minimum and zero velocity. Invalid units, normals, identifiers, slopes, roughness, gravity, and non-finite values raise `SectionError`. `register_sections` prevents silent overwriting of duplicate IDs.

## Hydraulic targets

For a scalar whole-section Manning coefficient and positive energy slope, `normal_stage(Q, n, energy_slope)` solves the SI relation

```text
Q = A (A/P)^(2/3) sqrt(Sf) / n
```

For spatial roughness, construct the section with one `segment_manning_n` value
per terrain segment and call `normal_stage(Q, None, energy_slope)`. Contiguous
segments with equal n form conveyance subsections and the solver uses
`Q=sqrt(Sf) sum[A_j (A_j/P_j)^(2/3)/n_j]`. Only physical terrain lengths enter
`P_j`; subsection divisions add no artificial vertical perimeter. Passing an
explicit scalar retains the original whole-section relation.

`critical_stage(Q, g=9.81)` uses a section energy coefficient of one and solves

```text
Q² T = g A³
E = stage - datum + Q² / (2 g A²)
```

These methods accept nonnegative discharge magnitudes. Direction comes from the inward normal; flow reversal requires the later regime policy. The solver brackets continuous branches at terrain elevations and conveyance extrema, so it does not invent a root across a basin-connectivity jump. It reports all admissible candidate stages, selecting the lowest normal root or the lowest-energy critical minimum. No admissible root, iteration exhaustion or unattainable floating-point tolerance raises `NoHydraulicSolution`. Defaults require a stage bracket of `1e-10 m` and a relative discharge residual of `1e-10`; final residuals are evaluated at the returned absolute stage.

`HydraulicSolution` includes stage, depth, requested Q, `mean_velocity_m_s = Q/A`, geometry, residual, iterations, candidate stages, roughness mode and subsection conveyances. A scalar n uses whole-section hydraulic radius; a spatial field uses the subsection relation above. GeoClaw's local Manning source uses cell depth, so matching the section target alone does not establish a friction-consistent transverse steady flow. The section energy criterion likewise does not imply that every local boundary face has Froude number one. The detailed conventions and source references are in [roadmap Section 12](../BRAIDS_ROADMAP.md#12-analytical-geometry-and-initial-reference-cases).

## GIS inputs

`load_boundary_sections(dem_path, gpkg_path, layer_name="boundaries", sample_spacing_m=None, datum_m=None)` returns an ID-to-section dictionary. Each feature must be a simple open LineString. Multipart, closed, self-intersecting, duplicate-vertex and zero-length geometry is rejected.

| GeoPackage field | Meaning |
|---|---|
| `bc_id` | Required unique string without whitespace |
| `normal_x`, `normal_y` | Required numeric components of an explicit inward unit normal |
| `datum_m` | Optional depth datum; the loader argument overrides it |
| `max_stage_m` | Optional maximum admitted absolute stage |
| `seed_station_m` | Optional wetting minimum in canonical station coordinates |

Normals are independent of digitization order. A single vector is stored per section; computing local normals and solver-face intersections along a bent line belongs to G3. The adapter does not infer the model domain's interior from a DEM.

Both DEM and line CRS must match and be projected in metres. This standalone
loader requires the DEM elevation band to declare metre units. The QGIS project
workflow also accepts a blank band unit under its explicit metre-elevation
convention and records whether metres were declared or assumed. A declared
non-metre unit remains an error. GDAL scale/offset are applied, the source datum
is retained without a hidden shift, and missing/invalid pixels or samples
outside the raster footprint raise errors.

The lexicographically smaller endpoint establishes station zero. Every vertex is retained; each line segment is subdivided at spacing no greater than half the smallest pixel scale (using the affine transform's smallest singular value). Reversing the line produces the same stations and terrain. Bilinear interpolation uses pixel centres, with constant extension over the outer half pixel to the raster footprint. On a sloping synthetic channel this gives an edge-elevation offset of half a pixel times bed slope; no exact edge value is inferred from the analytical fixture. More sampling points cannot recover terrain detail absent from the DEM.

`section_from_polyline` provides the same behavior for XY coordinate sequences with explicit `spatial_reference`, `bc_id`, and `inward_normal` arguments. Optional datum, sampling spacing, seed, and maximum stage have the same meanings.

## Verification and limits

[G2_SECTION_REPORT.md](../validation/hydraulics/sections/reports/G2_SECTION_REPORT.md) separates precision on the represented bed from smooth-curve and raster error. Independent Gaussian quadrature and an Illinois root supply expected results; the production module imports no validation code. GIS tests load both boundary features from all 24 G1 fixtures and test units, registration, affine rotation, reversal and invalid inputs. Edge-case tests cover seed connectivity, bank truncation and failed roots.

The existing uniformly sampled dense semicircle misses an additional `1e-4` near-bank top-width diagnostic. This result is retained in the report; angle-spaced analytical samples resolve the steep bank in a separate test. The specified ordinary-depth raster tolerances pass. Curved sections remain limited by the survey/DEM representation.

The [production-section qualification](../validation/hydraulics/production_sections/reports/PRODUCTION_SECTIONS.md)
extends that geometry evidence through the QGIS/native path. Deck v21 stores
sampled endpoint elevations and the normal/critical control type. Normal flow
uses the same whole-cell Manning field as native friction, and both control
levels and the connected component around the current lowest section point are
recomputed from the evolving boundary bed at each SSPRK stage. A
compact asymmetric case passes dry-start/default-wall conservation, exact
restart and byte-identical one/eight-thread checks. Noncollinear/disconnected
external allocation, optional raster Manning and internal structures remain
outside this service.
