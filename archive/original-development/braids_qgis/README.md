# BRAIDS QGIS plugin

BRAIDS provides a polygon-to-native workflow for the Fortran/OpenMP shallow-water and sediment-morphology solver. The dock is organized as **General, BC & IC, Mesh, Sediment, Run and Results**. The current vertical slice can:

- draw and save a styled calculation-domain polygon with QGIS's normal map-capture tool;
- create one persistent editable feature for every polygon boundary face;
- assign exactly one selected edge to an inlet or one or several contiguous collinear edges to an outlet;
- retain every unselected edge as a wall;
- set a domain-default Manning coefficient and override whole cells with editable polygon zones;
- configure sediment densities, porosity, grain sizes, z0, constant or concentration-dependent hindered settling, exchange, bed-load formula parameters, slope correction and spatial erodible inventory/bedrock inputs;
- enable qualified domain-wide wet/dry repose redistribution for mobile bars and channels, independently of bank zones;
- draw internal erodible-bank zones in Sediment and configure conservative dry-bank erosion supply and bank-local slope failure;
- draw free-polygon, circular or elliptical objects, or load polygon Shapefiles, as fixed impermeable reflective cut-cell obstacles;
- save the complete domain, grid, constant-level initial water body, per-ID hydraulic/sediment forcing, material and numerical contract;
- prepare exact cut-cell terrain and polygon-based initial-water conditions in a QGIS background task;
- run or restart the native solver without blocking QGIS; and
- display the exact initial cut-cell mesh, including small-cell agglomeration; and
- load time-indexed depth, velocity, water-level, suspended-concentration, bed-elevation and bed-change GeoTIFF rasters plus per-ID budgets, overlay the computational mesh, and animate their saved times in the dock.

The packaged Phase 2 and Phase 3 qualification manifests remain authenticated
release evidence, but those development labels are no longer shown as modelling
menus in the **Sediment** tab. They qualify the solver release, not the currently
open project. See
the full [Phase 2 report](../validation/sediment/delft3d_current/phase2_qualification/reports/PHASE2_QUALIFICATION.md)
and [Phase 3 report](../validation/sediment/delft3d_current/phase3_qualification/reports/PHASE3_QUALIFICATION.md).
The first Phase 4 manifest adds 28 paired wet/dry repose-avalanche gates against
the locked current-source D-Flow FM engine; it qualifies only the optional
domain-wide slope redistribution control, not bank-collapse physics.
The Phase 4 adaptation-length manifest adds a default-off non-equilibrium
bed-load control. Each grain fraction uses `L_g = C_L D50_g`; 100 diameters is
the editable measured starting value. The production solver passes analytic
exponential, Delft3D equilibrium-limit, per-fraction, conservation, wet/dry,
AMR, restart and OpenMP gates. Suspended-load adaptation is not implied.
The morphological-acceleration manifest adds a second default-off control.
MorFac 1--20 is qualified for the quasi-steady subcritical bedform, while the
paired wet/dry transient requires MorFac 1. The plugin therefore labels this
as a quasi-steady shortcut and rejects suspended exchange, hydraulic forcing
series and either bank-erosion model when it is enabled.
The bedform-roughness manifest adds a blank-by-default Fredsoe-MPM and Van Rijn
1984 feedback. Against its own `ks=3 D90` grain-only control, the matched 300 s
flume changes mean water level by 0.3221 m in locked Delft3D4 and 0.2858 m in
BRAIDS. It is restricted to roughness-corrected MPM, noncohesive material and
no suspended exchange or other experimental morphology process.

## Alpha installation

The current installable build is
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.104-QGIS-windows-x86_64.zip`. It supports QGIS 3.40 or newer
on 64-bit Windows. In QGIS, open **Plugins → Manage and Install Plugins →
Install from ZIP**, select the archive, install it, then enable **BRAIDS**. The
archive includes `braids_core`, the native OpenMP solver and its relocatable
runtime libraries, and the BRAIDS logo used by QGIS Plugin Manager and the
toolbar; a compiler is not required to run the Windows package.

See [the alpha release notes](ALPHA_RELEASE_NOTES.md) for the short manual test
route and current limits.

Alpha 97 binds the integrated GPU product to the authenticated
`implicit_fwave_midpoint` worker explicitly. It also generates the method
manual and all 69 typed morphology-output choices from the executable runtime
catalogue. The selected backend, methods, automatic dependencies, output
settings and catalogue identities persist through preparation, results and
restart; unsupported combinations reject before mesh or CUDA allocation.

The plugin includes a **GPU Run/Results UI and host reader**. The
GPU numerical worker, CPython 3.13/CuPy environment and NVIDIA driver are
separate; none is bundled in the QGIS ZIP. Select the extracted Windows source
worker and its external `python.exe` in the General tab, or use **Detect Windows
GPU setup** to fill likely paths for this PC, then press **Verify** for
the pinned package, Python and FP64 CUDA capability before choosing GPU. The
backend is always selected by the user and never silently falls back to CPU.
Verified paths are remembered, but identity pins are session-only, so Verify is
repeated after a QGIS restart. This is not an offline plug-and-play GPU installation.
Engineering gates cover integrated fixed-grid transport/bank, nonbank dynamic
AMR transport and hydraulic-only runs, accepted-step Stop/restart, live preview
and QGIS temporal/mesh readback. These are not physical validation of a bend
or braided-river experiment. Unsupported controls fail closed. See the
[GPU implementation-first plan](../validation/gpu_fwave/GPU_IMPLEMENTATION_FIRST_PLAN.md)
for the remaining qualification work.

New nonbank GPU AMR runs may select the slope-aware free-surface refinement rule
in the General tab after verifying the compatible worker. They may also request
recursive 2:1 level stepping for the admitted MPM/exchange-2/prognostic-secondary
scope, including the full-leaf bedload-adaptation closure, optional positive
mode-1/geometry-1 velocity-Laplacian viscosity and optional domain-wide general
avalanche. Viscosity runs once on the synchronized selected-leaf graph after
root-clock reflux and before avalanche; it is not applied once per level. Its
limiter controls area-weighted velocity L2, while variable-depth integrated
momentum is not claimed to be exact. Avalanche then runs once before acceptance.
The same shared
recursive scheduler supports static/evolving physical-toe bank mode 2 and
evolving distributed-bank mode 3. It
avoids refining a smooth river slope solely because neighboring water levels
differ. Old prepared jobs keep their legacy rule and exact worker/source
identity; a restarted run cannot change its selector. The v25 worker also
admits a separate paired equilibrium-bedform branch with one constant
unsampled total-discharge inlet, one constant stage outlet, MPM bed load,
exchange/secondary/viscosity/avalanche/banks off, MorFac 1 and one substrate
layer. With feedback ON it recomputes capacity at all three level-local SSPRK
stages and prepares friction once per level-local step; OFF is the matched
control. It additionally admits the exact recursive composition of bedform
feedback, evolving distributed-bank mode 3 and one parent-owned avalanche, and
a second exact composition with surveyed physical-toe bank mode 2. The mode-2
route derives its initial toe/donor/face map from the surveyed bank geometry,
rebuilds it after topology changes, and applies toe supply, undercut, collapse
and exactly one parent-owned avalanche. These banked branches require one
constant unsampled stage boundary and the finite control envelopes in the
reports below. Their initializers refuse direct solver execution without the
export flag. Other bedform compositions, including live hydraulic boundary
forcing with the mode-2 composition, reject before launch. This is a short-run
engineering qualification, not long-term morphological validation. The
exact viscosity scope and recovery checks are recorded in the
[recursive positive-viscosity gate](../validation/g4_bank_amr/runs/recursive_general_viscosity_cli_gate_20261006_v3/REPORT.md).
The bedform scope and package/recovery evidence are recorded in the
[recursive bedform integration report](../validation/g4_bank_amr/RECURSIVE_GENERAL_BEDFORM_INTEGRATION_REPORT.md).
The composed route and public/QGIS recovery evidence are recorded in the
[recursive bedform/bank/avalanche report](../validation/g4_bank_amr/RECURSIVE_BANK_BEDFORM_AVALANCHE_INTEGRATION_REPORT.md).
The physical-toe composition, portable-worker recovery and installed-QGIS
readback are recorded in the
[recursive physical-toe/bedform/avalanche report](../validation/g4_bank_amr/RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_INTEGRATION_REPORT.md).

The v26 general-AMR payload uses the additive
`braids_gpu/frozen_sources/on_demand_transport_amr_v8` namespace. Its corrected
MPM transverse-slope kernel is pinned to the retained CPU/GPU/Delft3D formula
receipt and the passing dynamic-regrid/restart route receipt; historical v7
workers and prepared jobs remain unchanged.

For eligible fixed-grid nonbank MPM runs with suspended exchange, bedform
feedback, adaptation and general avalanche off, Integrated + Plain may use
explicit SSPRK(4,3). The method is retained through preparation and restart and
is authenticated independently of the historical transport schema name. Other
combinations remain on plain SSPRK(3,3) and reject an incompatible choice.

The v26 worker and alpha 96 host also expose **Implicit f-wave midpoint
(experimental)** as an explicit choice for the plain common fixed-grid
transport route. It keeps the augmented f-wave spatial residual and weakly
coupled MPM/Exner update. The admitted envelope is fully wet, low-Froude,
constant forcing, nonnegative Manning, MorFac 1, and static bed or MPM law 2
with 1--32 fractions, at least two bed layers, slope off and hiding off.
Wet/dry, AMR, suspension, secondary flow, viscosity, bank processes,
avalanche, bedforms, adaptation and schedules reject for this method. The user
chooses both backend and time step; there is no automatic method or CPU
fallback. The small 128-cell production gate measured 22.276x warmed
trajectory speedup for a 20x time-step increase, with 0.012315 mm maximum depth
difference and 0.000006763 mm maximum bed difference from its fine SSPRK(4,3)
reference. This does not establish a reach-scale speedup.

Alpha 81 bundles the optimized native development solver. The tested 2h
curvature filter is now automatic when prognostic secondary flow is selected;
the solver chooses MIC or parallel multilevel by mesh size. Existing physics
checkbox defaults, CFL and production MorFac limits are unchanged. Prepare a
new run to select this runtime; old prepared/custom runtimes are not replaced.

## Development installation

First run `python3 tools/build_cutcell.py` from the repository root. The dock
uses `solver/cut_cells/build/xbraids_cutcell` as its current development runtime.
Saved projects retain their selected executable; rebuild older custom runtimes
before using v19 preserved-stratigraphy settings. The builder records native source and
executable hashes and keeps Fortran/OpenMP execution. On 64-bit Windows, run
`python tools/build_qgis_windows.py` to create the verified installable ZIP.
The release builder requires the pinned WinLibs GCC 13.2/MinGW-w64 toolchain
used by the recorded runtime, with `gfortran` and `objdump` on `PATH`, plus
network access to download and hash-check the pinned license/source notices.
Close QGIS before running `tools/install_qgis_windows_release.py`; the installer
also requires `tasklist.exe`, write access to the profile and an explicit
`--destination` when installing to a non-default profile.

Symlink or copy the `braids_qgis` directory into the active QGIS profile's `python/plugins` directory, then enable **BRAIDS** in the QGIS Plugin Manager. This source-checkout version finds `braids_core` beside the plugin. Run `python3 tools/build_qgis_alpha.py` to rebuild the Apple Silicon alpha ZIP after rebuilding the native solver.

The first General accordion exposes one **BRAIDS project folder**. BRAIDS writes
`braids_project.json`, generated GIS layers, prepared runs and converted results
under that folder. Loaded DEM and vector source layers remain referenced at
their existing locations. The native executable is selected automatically from
the installed plugin or development build and is not an editable project
setting.

The groups inside each tab are accordions. In **General**, **Draw Calculation Domain** is beside its layer selector and starts QGIS's polygon-capture tool: click the vertices and right-click to close it. BRAIDS saves the feature in a blue semitransparent GeoPackage layer named **Calculation Domain**. It remains editable with the normal QGIS tools. A loaded single-feature polygon layer, including a Shapefile, can instead be selected directly. Redundant consecutive ring vertices are ignored in the derived solver geometry without modifying the source layer. In **BC & IC → Boundary conditions**, **Create from Domain** creates one red line feature per polygon face, initially as a wall. Select exactly one face for each inlet; walls and outlets may use several selected faces. Multiple faces belonging to one hydraulic ID must be contiguous and collinear so they define one cross-section. **Apply to Selected Face(s)** validates this before editing the layer. Reusing an inlet/outlet ID replaces its former faces, returning them to walls, while a failed assignment leaves the prior valid assignment unchanged. Each active inlet/outlet ID has a structured accordion card with **Delete BC**, which deletes the complete logical condition and returns all of its assigned faces to walls. Inlets prescribe total discharge plus water level, water depth above the section datum, computed normal depth, or computed critical depth. The native run retains that horizontal section level and integrates the uniform normal velocity to the requested total discharge as boundary cells wet. Normal-depth slope is computed internally from the DEM. Its conveyance uses the same whole-cell Manning default and polygon-zone field as native momentum friction, grouped into contiguous roughness subsections. Neither slope nor Manning is an editable boundary field. The sampled piecewise-linear terrain and normal/critical level are updated when morphology changes the boundary bed. An empty DEM band-unit label is treated as metres in this QGIS workflow; a non-metre declared unit remains an error, and the source DEM datum is retained without a hidden shift. Outlets support prescribed level, computed critical depth, free-weir flow, or supercritical release. **Highlight Assigned Face(s)** selects and zooms to existing assignments without changing them. The other buttons plot forcing and sample the DEM to preview the cross-section, resolved horizontal water line and wetted boundary section. The wetted section is the connected portion of terrain below the resolved horizontal water level through which boundary flow can pass. The solver records the resolved target, live section geometry and accepted numerical flux during a run. Reverse flow is rejected explicitly.

Initial water controls are grouped under **BC & IC → Initial conditions**. Water can be drawn or loaded as a polygon and assigned the constant **Initial water level** immediately below it. Alternatively, enter the level, press **Create from DEM Point**, and click inside the intended basin. BRAIDS follows four-neighbour DEM connectivity at or below that horizontal level, rejects an open contour at the DEM edge, and saves the resulting editable polygon together with its clicked seed and selected-cell count. Cells with positive-area polygon contact are initialized to that level where it lies above the bed; cells outside remain dry. With no polygon, a blank level means an explicitly dry initial domain; an entered level applies to every below-level cell in the calculation domain.

**BC & IC → Internal conditions** currently contains only fixed solid reflective walls. Use **★ Polygon**, **○ Circle** or **⬭ Ellipse** to draw an editable object, or select a polygon Shapefile or GeoPackage. Circles use a centre and radius point; oriented ellipses use a centre, major-axis point and perpendicular minor-radius point. **Select**, **Delete Selected** and **Cancel** provide explicit editing controls. These footprints also represent bridge pillars. The solver removes each object at its exact cut-cell perimeter and reflects shallow-water waves from its complete wall.

The **Mesh** tab uses **Minimum cell size** as the finest AMR resolution. Choose from two to eight nested levels; adjacent levels always use the qualified fixed 2:1 ratio, and the displayed L0–L\(N-1\) sizes are derived automatically. The default remains four levels (20, 10, 5 and 2.5 m). Every inlet and outlet is pinned to L\(N-1\), so it always retains the minimum size through all regrids. Manning remains in its own accordion. Reconstruction is selected internally: the solver always uses the qualified bounded WENO-Z general mode, with low-diffusion WENO-Z on supported regular strips and bounded linear reconstruction on irregular cut cells. The boundary-cell policy is also automatic: small cut cells are merged at a 0.5 retained-area threshold to avoid needless timestep restrictions. Set the domain default, enter a new-zone value, then press **Add New Manning Zone** and draw it. BRAIDS assigns its name from the coefficient, styles it semitransparently and labels it `Manning = x`. Vertex-and-segment snapping to all vector layers is enabled by default. A zone assigns its full value to every retained cell with positive-area overlap; cells are never split or area weighted. If snapped zones share a node or edge and both touch one solver cell, that cell receives the highest Manning value. Polygon interiors may not overlap. Disconnected zones may not occupy the same whole cell; both cases report the affected IDs and stop preparation.

Each hydraulic card supports constant forcing or a piecewise-linear hydrograph. Level/depth inlets use `time, total Q, control`; normal/critical inlets use `time, total Q` and resolve the sampled section at every knot; prescribed-stage outlets use `time, water level`. **Plot Time Series** shows total Q and the absolute level that the native solver will execute. Native v7 evaluates both at SSPRK stage times, stops at every forcing knot and holds endpoint values outside the declared range.

The **Sediment** tab is opt-in. New projects start with the master switch off, and suspended load, bed load/Exner, secondary flow, bedforms, adaptation, acceleration, avalanching and both bank routes are all unchecked. Each process must be enabled deliberately; loaded projects restore their saved physics. In **Bed load and morphology**, optional **Non-equilibrium bed load** delays each class transport vector behind local capacity along resolved streamlines; its length is entered in that class's D50 diameters and defaults to 100. The qualified **Domain-wide repose avalanching** accordion applies conservative wet/dry slope redistribution to mobile bars and channels; it is optional and requires multiple fractions plus a positive active layer. Its controls are mutually exclusive with bank-local avalanching. The separate **Erodible banks and slope failure** accordion holds the bank-zone selector, dry-bank erosion supply and bank-local slope-failure controls. Per-fraction transfers remain conservative, stop at bedrock and occur only across internal faces. Lowered cells can wet through the hydraulic solver. [Adaptation-length qualification](../validation/sediment/delft3d_current/adaptation_length/reports/REPORT.md) · [general avalanche qualification](../validation/sediment/delft3d_current/general_avalanche/reports/REPORT.md) · [bank verification](../validation/sediment/bank_morphology/reports/BANK_MORPHOLOGY_VERIFICATION.md) · [stratigraphy verification](../validation/sediment/stratigraphy/reports/STRATIGRAPHY_VERIFICATION.md).

The blank-by-default **Bedform roughness feedback** control solves the Fredsoe-MPM/Van Rijn 1984 local equilibrium at every hydraulic stage and writes `bedform_roughness.txt`. It requires roughness-corrected MPM, noncohesive fractions and suspended exchange off. Normal-depth inlet control and other experimental morphology switches fail closed. See the [paired qualification](../validation/sediment/delft3d_current/bedform_roughness_assessment/reports/REPORT.md).

**Save Project** is in the first General accordion. In **Run**, the **Show live preview** control and variable selector sit directly above **Prepare Run**. Preparation reports mesh/input setup and terrain work on one percentage bar, and **Stop** cancels either preparation or a running solver. An unchanged project and native executable reuse the latest prepared input deck and immutable sidecars directly, avoiding a repeated mesh/terrain build. Fixed-grid and AMR runs write native Float32 preview grids at each saved time while the solver is running. QGIS reads only the completed preview frame through VRT bands, reloads it once per saved time, and centres the first preview; it never opens the solver's growing authoritative frame file. These previews use the finest native grid and omit optional cut-cell display antialiasing. Completed results can overlay the final computational mesh by default. In **Results → Animation**, **Animate adaptive mesh** is an empty, explicit opt-in with its own build progress bar; the generated mesh stays cached on disk when hidden. The same accordion contains PNG animation export. **Boundary flow** reports positive magnitudes for the active boundary IDs stored with the selected run. Results can also sample editable point-probe layers and integrate editable flow-transect layers for plotting or CSV export. **Show execution log** displays output also retained as `native.log`.

Preparation screens the initial native trace of every critical-depth outlet. If a selected face and initial level imply a trace above the solver's 50 m/s safety limit, preparation identifies the outlet, represented section length, critical discharge and speed and stops before execution. For a reservoir with no outlet, delete the outlet assignment so the face returns to its default wall behavior. A real outlet should cover only its physical opening and use the hydraulic control appropriate to that opening.

The resource-reservoir example and proof are in [`validation/qgis/vertical_slice`](../validation/qgis/vertical_slice/).
The native spatial-roughness proof is in [`validation/hydraulics/spatial_manning`](../validation/hydraulics/spatial_manning/reports/SPATIAL_MANNING.md).
The sediment-settings and tab-layout proof is in [`validation/qgis/sediment_settings`](../validation/qgis/sediment_settings/reports/SEDIMENT_SETTINGS_UI.md).
The short hydraulic-mode qualification is in [`validation/hydraulics/cut_cell_boundary_modes`](../validation/hydraulics/cut_cell_boundary_modes/reports/BOUNDARY_MODE_QUALIFICATION.md).
The hydraulic-hydrograph mass-balance proof is in [`validation/hydraulics/hydrographs`](../validation/hydraulics/hydrographs/reports/HYDROGRAPH_VERIFICATION.md).
The supplied-reservoir spatial-bed preparation is in [`validation/qgis/spatial_bed_inputs`](../validation/qgis/spatial_bed_inputs/SPATIAL_BED_INPUTS.md).

## Morphological acceleration

The default-off **Morphological acceleration** control accepts a constant
MorFac from 1 to 20 or a Delft3D-compatible block schedule written as
`time_s:factor` knots. It accelerates conservative bed-load/Exner evolution,
not transport capacity. Each run records hydraulic time, integrated
morphological time and instantaneous factor in
`morphological_acceleration.txt`.

The paired qualification permits factors up to 20 only for the quasi-steady
subcritical bedform; MorFac 1 is required for wet/dry hydraulic transients.
Suspended exchange, hydraulic forcing series and bank erosion/failure fail
closed while acceleration is enabled. See the
[qualification report](../validation/sediment/delft3d_current/morphological_acceleration/reports/REPORT.md).
