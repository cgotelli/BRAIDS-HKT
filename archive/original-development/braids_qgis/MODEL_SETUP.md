# Model-configuration tools

The unified interface retains domain and boundary map editing. Alpha134 requires
a new v20 project with explicit CFL and independent checkpoint controls. Both backends use
the same explicit SSPRK22 augmented-f-wave method. Physical transport formulas and
the Results tab are unchanged. Sediment remains OFF when opening a new dock.

## Start with hydrodynamics only

Choose **Hydrodynamics only** in General, then choose CPU or GPU. Sediment,
Bed & banks and sediment boundary controls stay hidden while transport is off.
The physical choices are not silently changed when changing backend.
Advanced hydraulic/runtime settings and the resolved technical summary are
collapsed; they can still be inspected when needed.

The **Project folder / Choose folder** row is always visible above the tabs.
Choose a new project directory, configure the model, then use **Save project
folder**. **Open project folder** reopens the model, saved run history, results
and compatible preparation. Opening never starts or resumes a calculation.
The history menu can restore a saved parameter revision without overwriting
the old revision or running the solver. To duplicate a project, choose a new
empty folder and Save; this copies the complete history, not just its settings.

The folder contains `braids_project.json` (current model and editor choices),
`inputs/` (a copied DEM and required sidecars), `history/` (parameter revisions,
worker events and console logs), `prepared/` (immutable prepared models), and
`runs/` (results, diagnostics and accepted checkpoints). Drawings are stored in
`model_layers/`; their simulation geometry is also embedded in the model.
Every run retains its own `PROJECT.json`, so changing current settings cannot
rewrite the parameters associated with earlier results.

The folder can be copied or moved as a whole. Runtime identity and backend
checks still apply to restart: a different solver release requires explicit
repreparation, not silent conversion of an old checkpoint. Python, QGIS and
CUDA are installed software, not project assets. The saved worker interpreter
path must be selected again if that installation differs on another PC.
Unrelated QGIS layers and the QGIS `.qgz` session remain under QGIS control.

Create a new current-format project, select the DEM and domain, assign the open
boundaries, review their grid alignment, then Prepare and Run. Do not open old
alpha123/127 project files as current inputs. An initial stage below the DEM
starts the domain dry; a short inflow run is not an equilibrium solution.

## Computation and save intervals

- **Maximum outer step** limits an outer solver transaction. The actual hydraulic
  substeps can be shorter because the explicit method checks stage CFL and
  positivity. It is not a promised hydraulic timestep.
- **Solution output interval** controls the frames used for maps, plots and
  animation. Output times are hit exactly, so changing this interval can also
  change the partition of numerical steps.
- **Restart checkpoint interval** controls full restart-state saves independently.
  New projects select **Use simulation duration**, so the interval follows End
  time. Uncheck it to enter a custom periodic interval; that custom value is
  retained when switching back and forth and when saving/reopening the project.
  Existing v20 projects keep their explicitly saved interval as custom unless
  they already record the automatic choice. The first accepted post-assessment state at
  or after the deadline is saved without inserting a new timestep boundary.
  Initial, final and cancellation states are saved regardless of this interval.
  A crash can lose the work since the last checkpoint, whereas Stop saves the
  current accepted state.
- **AMR interval** is a separate mesh-assessment clock and applies only to dynamic
  meshes. A static mesh incurs no periodic AMR assessment.

Binary NPZ results and checkpoints stay in the project folder. Opening results
does not restart a run. A solution frame alone is not a restart checkpoint.
Full-file authentication remains enabled for result loading and restart.

## Choose an inlet condition

All four depth-reference choices prescribe an inward discharge and zero incoming
tangential momentum. Normal depth computes its reference using the selected
resistance and a signed inward bed-slope estimate on the initial prepared finest
mesh. The width-weighted median of longitudinal strip fits avoids letting a few
steep levee margins dominate the channel slope. The estimate and its spread are
saved in `NORMAL_SLOPES.json` and frozen across AMR/restart. This assumes uniform
flow (energy slope equals bed slope); it is not a measured unsteady energy slope.
A flat/adverse estimate is rejected. Critical depth solves the actual section
condition. Elevation is an absolute water level; depth is relative to the explicit
fixed datum, not a different horizontal level above each face's bed.

These are incoming reference states. A common virtual exterior velocity is solved
so the sum of boundary Riemann mass fluxes is exactly the prescribed discharge.
Each face uses the matching momentum flux, including shock/rarefaction and dry-bed
behavior. Actual face depths and elevations can differ from the reference and
from one another. Backwater and signed local backflow are retained; the specified
discharge is the **net** inward discharge. The virtual velocity is a boundary
root variable, not an imposed interior velocity or a timestep criterion.

Imposing both actual depth and discharge independently would over-constrain a
subcritical boundary. The reference critical section Froude number is one; this
is not a promise of unit Froude number at every wet face or in backwater. Normal
depth is retained in compatible uniform flow, not imposed against backwater.

## Align inlet and outlet sections only

After assigning the inlet and outlet, choose Mesh / **Review axis-aligned
inlets/outlets…**. The proposed polygon appears as a separate map layer.
Review each section's new width and maximum endpoint shift, and expand the
details to inspect the area change. No is the default answer. Cancel leaves
the selected domain and assignments unchanged. Yes selects the new polygon,
retains the original domain layer, and recreates the assigned faces under the
same names. Hydraulic settings are retained. Save as a new JSON project; a
`.geometry-review.json` sidecar records the accepted change.

Only vertices belonging to inlet/outlet sections are snapped. Other wall
vertices stay unchanged; adjoining wall segments reconnect to moved section
ends. Walls are not converted to stair steps, and cropped wall cells are
allowed, including near a section endpoint. Check the changed sections and
DEM coverage before accepting. Changing the grid requires another section
review; it must not trigger wall rasterization.

## Choose hydraulic accuracy

Run / **Hydraulic timestep settings** exposes the face-sum CFL. Fresh QGIS
pure-hydrodynamic projects default to **1.0**, with a range above zero through
1.0. Sediment, secondary-flow or turbulence combinations require **0.5 or less**;
set this explicitly when enabling them. Incompatible values are rejected, never
silently reduced. The timestep criterion is
`dt * max(sum(face length * wave speed) / actual cell area) <= CFL`.
This is not numerically interchangeable with a cell-length CFL from another
solver. Cropped wall cells retain their actual area. The maximum timestep is
an upper bound, not a forced timestep, and is separate from the AMR interval.
The fresh-project maximum is **2 s**. Saved CFL and maximum-timestep values are
loaded unchanged. Each SSP stage still checks its actual CFL; after AMR the
previous accepted estimate is carried only as a predictor, not a stability
guarantee for the new mesh. Output and forcing endpoints remain exact.
The 2 s outer limit has bounded hydraulic and single-grain bedload evidence;
other coupled processes may require a shorter outer timestep for accuracy.
The solver uses the same explicit method for wet and dry regions; it does not
switch to implicit integration or select another backend. Shorter timesteps
can improve temporal accuracy; CFL is not a bound on accumulated global error.
Hydraulic resistance uses each cell's physical depth, not adjacent wet-face
averaging. Sediment shear and physical roughness formulas are unchanged.
Older project/prepared/checkpoint formats are rejected, not converted.

## Choose a mesh strategy

In Mesh / Mesh strategy, choose one of:

- **Dynamic AMR — water depth only**: refine wet cells and a neighbour buffer;
  assess at the configured interval. No terrain criterion is applied.
- **Static mesh — terrain slope breaks only**: use the initial finest DEM to
  refine levee crests, toes and other resolved slope changes once. There are no
  runtime assessments or regrids. It does not follow water fronts or later bed
  changes; wet interior cells can remain coarse. Only terrain controls are active.

Both modes keep inlets/outlets and explicitly surveyed bank corridors
finest, and enforce 2:1 level balance. The smallest cell size is the nominal
Cartesian spacing; cropping at a wall can still produce smaller fragments.
Terrain detection requires both the slope-angle and vertical-detail thresholds
on a neighbouring pair, plus a one-finest-cell halo. A constant plane is not a
slope break. The defaults are 5 degrees and 0.10 m; features narrower than the
selected finest spacing cannot be guaranteed resolved.

## Dynamic mesh update frequency

The Mesh tab's **Assessment interval (s)** defaults to **1 physical second**.
It is independent of the hydraulic timestep. You may set 2 s or more;
increase the **Wet-front buffer** if the advancing water can traverse the
buffer between assessments. Open boundaries retain that buffer before water
arrives, including possible flow reversal. Only inlet/outlet cells are forced
finest by boundary type. Walls follow the selected indicators and 2:1 grading,
unless an explicit surveyed-bank corridor protects them. Dry coarsening delay
counts consecutive assessments, not seconds.

Loading a current v17 project keeps its interval. Older project formats cannot
run. Existing prepared/checkpoint folders remain bound to their own runtime and
settings; prepare a fresh run after updating the plugin.

## Create the domain and boundary conditions

1. In **General**, load/select a projected-metre DEM. Load a polygon or choose
   **Draw polygon**. Left-click vertices, right-click to save; Backspace removes
   a vertex and Escape cancels. A single, valid polygon without holes is required.
2. For an existing layer with multiple polygons, use **Select** to choose exactly
   one feature. Use **Edit vertices** for QGIS vertex editing. Right-click or
   **Finish edits** saves and exits; Escape or **Cancel tool** rolls back that
   BRAIDS-owned edit session. Unrelated existing QGIS edits are never committed
   by these buttons.
3. Optional **inlet guides** are drafting lines, not boundary conditions. They
   can be loaded, drawn and edited, and are hidden when a domain is saved or
   boundary faces are generated. They do not enter the numerical input.
4. In **BC & IC**, choose **Create / replace boundary faces**, then **Select
   faces** and **Apply to selected faces**. Reusing a boundary name replaces its
   assignment. Invalid/noncollinear sections are rejected before replacement.
   Clicking a table row selects that boundary on the map. **Clear named boundary**
   returns its faces to walls, without deleting the domain.
5. After changing the domain, recreate its boundary faces and reassign conditions.
   Old generated faces are replaced; a prepared mesh cannot survive a changed
   domain or a changed selected feature. Do not edit generated face geometry
   independently of the domain.

New drawing layers are unique GeoPackages under `<project folder>/model_layers`.
Existing files are not overwritten. A cancelled first drawing can leave an empty
GeoPackage; it does not replace the selected calculation domain. Use QGIS's
project save to retain the complete layer tree and drafting guides. The BRAIDS
project saves its domain and boundary geometries as explicit simulation input.
Drawing uses the DEM CRS, or the QGIS project CRS when no DEM is selected; map
canvas coordinates are transformed into that CRS. Temporary snapping settings
are restored when drawing/editing ends.

## Surveyed bank and physical-toe geometry

For an admitted surveyed-bank method in **Bed & banks**, use **New + draw** for
the first polygon/line, then **Add geometry** for further IDs. Enter matching,
unique IDs for each polygon and its toe polyline. Select, edit and zoom are
available for both source layers. These controls do not enable bank physics.

**Use selected survey layers (snapshot)** explicitly copies the source geometry
into the simulation input. Editing a source layer does not silently change that
snapshot: reimport it. After reopening a BRAIDS project, **Create editable layers
from saved survey** writes new source copies that can be edited and reimported.
It preserves the saved input until that explicit reimport.

## Comparison with the older configuration interface

| Earlier configuration tool | Unified interface status |
|---|---|
| Domain polygon selection/drawing; inlet-guide lines | Restored, with explicit selection, vertex editing, zoom, finish/cancel and persistent files |
| DEM and project-folder selection | Layer selectors retained; direct file/folder browse restored |
| Generate boundary edges; select/reassign/delete named conditions | Available; generated faces replaced, named conditions cleared back to walls |
| Boundary forcing and hydraulic methods | Existing current-schema editor retained; no legacy method IDs imported |
| Initial stage and velocity | Existing global fields retained |
| Initial-water polygons and DEM-connected-basin tool | **Missing from the unified input route**; must bind a spatial initial-water definition before exposing these buttons |
| Solid obstacles: polygon/circle/ellipse, select/delete | **Missing from the unified input route**; internal cut walls need preparation, boundary and restart connections, not merely drawings |
| Minimum-refinement polygons | **Missing**; current AMR criteria exist, but arbitrary user zone constraints are not mapped into the current preparation schema |
| Spatial roughness polygons and selected-zone values/priorities | **Missing**; current global roughness is available; spatial parameter mapping is not |
| Erodible-thickness polygons | **Missing**; current finite bed layers are available, but spatial stock masks are not exposed |
| Bank zones | Current surveyed-bank polygon/toe drawing and editing added; not a silent translation of legacy bank physics |
| Mesh size/levels, AMR criteria | Existing controls retained; legacy subcycling is not restored because the current solver uses a common clock |
| Separate initial mesh construction | Not restored: preparation must remain the mesh owner, avoiding the former duplicate construction |
| Python/checkpoint browse | Added alongside the existing discovery/checkpoint fields |
| Old projects, cohesive physics, separate execution workspace | Deliberately excluded, not restoration targets |

This is not full parity with the old configuration interface. The next setup
work is to restore the spatial input mappings above, with CPU/GPU preparation
and save/reload checks. No placeholder button should suggest that an unused
polygon changes a simulation. Results are outside this audit.
