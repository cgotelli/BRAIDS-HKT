# Polygon domains and explicitly selected boundaries

The confirmed workflow is to **keep `profile1.shp` as an example section and define the inlet on a selected domain-polygon edge**. The point remains an example seed/location. The polygon is not trimmed to the profile, and the profile is not interpreted as an internal discharge source. Every unassigned perimeter segment is a wall; an outlet exists only after the user explicitly selects and assigns it.

The [whole-exterior native adapter](POLYGON_NATIVE_BOUNDARIES.md) now maps subdivided rectangular polygon edges to grouped inlet/outlet faces and verifies their accepted water/sediment budgets. It rejects unsupported oblique/internal/partial faces explicitly.

The new [domain data model](../braids_core/boundaries/domain.py) implements geometry, stable edge/segment identities, and assignments. It does **not** implement arbitrary-polygon boundaries in the native WAVE solver. The currently qualified native inlet mapping remains restricted to whole axis-aligned external faces on one Cartesian level; masking cells outside a polygon or turning oblique edges into solver faces needs a separate implementation and conservation/geometry qualification.

## Supplied data: read-only inspection

Inspected on 2026-09-07 with GDAL/OGR. All supplied inputs were preserved.

| Layer | Components and fields | Observed geometry |
| --- | --- | --- |
| [domain.shp](../resources/domain.shp) | `.shp`, `.shx`, `.dbf`, `.prj` present; one Integer64 field `id=1` | One valid simple Polygon, no holes; area 2,746,524.262 m² |
| [profile1.shp](../resources/profile1.shp) | Same required components and `id=1` field | One valid simple two-point LineString, length 258.625901 m |
| [point.shp](../resources/point.shp) | Same required components and `id=1` field | One valid Point at E 2,670,330.678864, N 1,171,573.285331 |

The actual section filename is `profile1.shp`; `profile.shp` was not present. None of these shapefiles has a `.cpg` file. Their only attribute is numeric, so no existing text attribute has an unresolved encoding. Their numeric `id` fields do not yet establish distinct persistent domain/boundary IDs or hydraulic types.

All three layers declare **EPSG:2056, CH1903+ / LV95**, with metre horizontal coordinates, matching [topo_cut_dam_1m.tif](../resources/topo_cut_dam_1m.tif). Their horizontal relationships are:

- The whole polygon, profile, and point lie inside the DEM footprint.
- The profile lies entirely **inside** the polygon; its minimum distance to the perimeter is 374.632604 m. It has no intersection with that perimeter and is not an external-boundary segment.
- The point lies inside the polygon, 376.508845 m from its perimeter and 1,452.675975 m from the profile.
- The polygon contains 2,746,524 DEM pixel centres. All those sampled raster values are finite; the band has no declared NoData value.

The polygon ring contains ten coordinate records including its closure, with eight distinct vertices. Source vertex index **8** (zero-based) repeats index 7 exactly at `(2669687.3861683575, 1171383.1554980078)`, creating a zero-length edge. OGR still reports the polygon as valid, but this is unsuitable for assigning an edge normal or ID. The new loader rejects it with that source index and an actionable message. Resolve it explicitly in a derived geometry while retaining the original-to-derived vertex mapping; the loader does not silently repair the supplied file.

The DEM is 5,787 × 3,628 pixels at 1 m spacing. Its footprint is E `[2667311,2673098]`, N `[1169558,1173186]`. All 20,995,236 band values are finite, but **the elevation unit metadata is blank**; no scale or offset is declared. Metre horizontal coordinates do not prove metre elevation values. The strict [section GIS adapter](SECTION_SERVICE.md) therefore requires an explicitly justified metre declaration in a derived DEM before hydraulic section extraction. The domain loader's optional DEM check validates horizontal CRS and footprint only; it does not qualify elevations, vertical datum, or terrain coverage between sampled centres.

## Data contract

`PolygonDomain(domain_id, vertices_xy, spatial_reference)` represents one finite 2D simple polygon without holes. It accepts concave polygons, rejects self-intersections and zero-length edges, and requires a projected metre CRS. The perimeter is stored counterclockwise from the lexicographically smallest vertex; either input orientation and any cyclic starting vertex produce the same result. Each edge has its length and a unit normal pointing locally into the polygon.

IDs have separate purposes:

- `domain_id`: an explicit persistent string identifying the domain.
- `edge_id`: derived from `domain_id` and the exact endpoint coordinates, independent of ring order or feature numbers. Unchanged edges retain their IDs after other edges are edited. Moving an endpoint or inserting a vertex into an edge changes its identity; assignments using a stale ID are rejected and must be reviewed. Reprojection likewise requires a new geometry/assignment review.
- `segment_id`: identifies an edge interval by that edge's identity and its exact start/end stations. Changing hydraulic type alone does not change its geometric identity.
- `bc_id`: identifies a logical hydraulic boundary. Several nonoverlapping pieces may share it when they have the same type and intentionally belong to the same boundary. Distinct physical inlets/outlets should receive distinct IDs.

`BoundaryAssignment(edge_id, bc_id, kind, start_m=0, end_m=None)` assigns a whole edge or an interval. `kind` is `wall`, `inlet`, or `outlet`; `end_m=None` means the edge end. Stations increase in the canonical counterclockwise direction. Adjacent assignments may meet at an endpoint, but overlaps, zero-length/outside intervals, unknown edges, and mixed types under one `bc_id` are errors. There is no implicit snapping, clipping, or transfer from an interior profile.

`domain.boundary_segments(assignments)` partitions the entire perimeter and fills every gap with `wall`, using the default ID `<domain_id>:wall`. These geometric intervals are not native cut faces. Hydraulic mode, hydrograph, section sampling, stage/depth datum, roughness and outlet-control parameters must be attached through their respective services before a future solver mapping can consume them.

## Example and next integration

This synthetic rectangle illustrates selection; it does not choose an edge of the supplied reservoir for the user:

```python
from braids_core.boundaries.domain import (
    PolygonDomain, BoundaryAssignment, load_polygon_domain,
)

domain = PolygonDomain(
    "reservoir_01", [(0, 0), (100, 0), (100, 50), (0, 50)], "EPSG:2056"
)
selected = domain.edges[0]  # Example only: the UI returns the selected edge_id.
assignments = [BoundaryAssignment(selected.edge_id, "river_01", "inlet", 20, 80)]
segments = domain.boundary_segments(assignments)
assert all(s.kind != "outlet" for s in segments)

# Only an explicit additional selection would open an outlet:
# assignments.append(BoundaryAssignment(outlet_edge_id, "outlet_01", "outlet"))

# After explicit duplicate-vertex resolution in a derived layer:
# domain = load_polygon_domain(
#     "derived/domain.gpkg", domain_id="reservoir_01",
#     dem_path="resources/topo_cut_dam_1m.tif",
# )
```

`load_polygon_domain` reads a single polygon feature, with an explicit string `domain_id` or a string ID field of that name. An explicit layer name is required for multilayer datasets. It supports read-only OGR vector inputs, including Shapefile and GeoPackage; it never derives a persistent ID from an FID or silently converts a numeric field. The supplied polygon currently requires the duplicate-vertex correction described above.

The pending UI integration is a QGIS layer/selection interface that persists these assignments in a derived project dataset, displays default walls and selected boundaries, and passes a selected perimeter section to the existing hydraulic section service. Mapping the resulting polygon to the Cartesian solver—including oblique boundaries, cell ownership, partial-face areas, wall momentum, and accepted-flux conservation—remains open. The original geometry-only audit did not run a reservoir. The later whole-exterior native qualification is linked above; arbitrary-polygon support remains open.

Run the focused geometry and I/O tests from the repository root:

```sh
.venv/bin/python -m pytest tests/test_domain.py -q
```

## Multiple polygon-edge inlets and outlets (confirmed 8 September 2026)

A boundary may explicitly own one or several polygon edges/intervals. Multiple
inlet and outlet IDs, including several of the same type, must coexist. A
shared inlet ID carries one total discharge, allocated once over all mapped
openings; it does not prescribe that total separately on every selected edge.
Every unassigned edge remains a wall. Shared outlet IDs need a coherent control
and explicit backflow/sediment policies. Disconnected sections require an
explicit allocation rule. The existing assignment model supports the geometry;
the whole-exterior rectangle map is now qualified, while arbitrary-edge native geometry and QGIS multi-selection remain open.
See the [delivery checklist](BRAIDS_DELIVERY_CHECKLIST.md) for implementation and
bounded acceptance cases.
