# BRAIDS first unified release: noncohesive rivers

The release scope is frozen. Cohesive sediment, mud, flocculation,
consolidation and cohesive-bank strength are excluded. No additional physical
feature is a prerequisite for this release. Historical prototypes are not
public capabilities.

Preparation uses exact compiled cropped-cell terrain integration and bounded
parallel source-archive I/O. The source archive still contains independent copies
and verifies their full contents against the runtime. Stop is checked while
archiving; no copying task survives cancellation or failure. These are setup
optimizations, not different hydraulic or sediment methods. Reprepare under a
new release rather than reusing an older runtime's prepared-state identity.

Solution output and restart checkpoint output are independent. New projects
save viewable solution frames every 1 simulated second. **Use simulation
duration** is checked for the restart interval, which follows End time. Uncheck
it for periodic recovery checkpoints; the custom interval and editor choice are
saved explicitly. Existing v20 projects retain their saved explicit interval
unless they already record the automatic choice. Both resolved intervals remain
visible and saved; no numerical timestep or checkpoint algorithm changes.
Checkpoint publication uses the first accepted state at or after its deadline;
it does not shorten hydraulic timesteps. Initial, final and cancelled runs still
save their accepted restart state. An abrupt process or PC failure can lose work
since the last checkpoint; with the new default this can mean the entire run
after its initial state. Use a shorter custom interval when that trade-off matters.
The solution and restart formats remain uncompressed binary NPZ, with full
checksums and closed-file atomic publication. Nested checkpoint serialization
uses memory instead of intermediate temporary files. Numerical methods, physical
formulas and CFL limits are unchanged. New v20 projects are required; earlier
projects/checkpoints are retained as evidence, not migrated or executed.

Alpha133 added self-contained project folders: copied terrain inputs, model
drawings, saved parameter revisions, preparations, runs, checkpoints and logs.
Open a project folder to restore its configuration and results; restoring an
older parameter revision is an explicit action and never starts a simulation.
Saving an existing project to another folder copies its complete history.
Each run retains its own authenticated parameter snapshot. Moving a complete
folder does not rewrite authenticated preparations or checkpoints. QGIS and
the external worker environment remain installed dependencies; another PC
may require a different interpreter path. Unrelated QGIS layers still belong
in the user's QGIS project file.

The numerical solver is unchanged from alpha132. It retains CFL 1 as the fresh
QGIS default for pure hydrodynamics, with a 2 s maximum outer timestep. Saved
settings are preserved. Sediment, secondary-flow and turbulence combinations
require CFL 0.5 or less; incompatible values produce an explicit error, not a
silent change. CFL 1 trades some impulsive wetting-front velocity accuracy for
speed. The dry threshold and physical formulas are unchanged.

AMR reuses exact immutable cell geometry, omits bank/collapse operators when
those features are disabled, and avoids redundant source-inventory construction
and short-path native input copying. Geometry, source integrity and conservation
checks remain. Accepted timestep estimates survive remeshing, existing output
mesh epochs are reused. Native geometry initialization now returns its exact
records through in-process memory callbacks rather than launching an exporter
and writing/rereading native record binaries. Prepared records and initial fields
are privately cached, with detached working copies and authenticated restart
archives. The CPU still reads its input deck when constructing a hydraulic owner;
this is not a claim of completely disk-free remeshing.

The preceding alpha129 delivered one explicit SSPRK22 augmented-f-wave solver on CPU and GPU,
with cell-local hydraulic resistance and CFL 0.225 visible in Run. Sediment
remains unchecked by default. That release required v19 projects; older projects,
prepared runs and checkpoints cannot be used with the new numerical method.
The alpha123 implicit archive remains a frozen historical reference, not a
second supported solver. Earlier release notes below describe those releases.

The four discharge inlet targets are normal depth (automatic signed mesh-slope
estimate), critical depth, water elevation and depth above an explicit fixed
datum. Their incoming reference is not a second imposed constraint in wet
subcritical flow. A shared virtual exterior velocity is solved so the boundary
Riemann mass flux integrates to the requested discharge; its matching momentum
flux accounts for shocks, rarefactions and backwater. The actual face water levels
can differ from the reference and from each other. Signed local backflow is
retained and reported; the total inward discharge remains exact. Zero discharge
uses a reflected-wall pressure response. No interior velocity is clipped to
hide a boundary error. Normal-depth preparation records its uniform-flow bed-slope
assumption and rejects a flat or adverse estimate rather than taking its absolute
value. Inlets and outlets must lie on actual finest grid lines; walls may remain
cropped. Changing the grid invalidates the geometry review.

AMR construction reuses authenticated immutable runtime material and constructs
the final native hydraulic operator once per owner. Output/outer-clock remainders
no longer artificially reduce the following timestep hint. The actual stage CFL,
positivity checks, physical formulas and user-selected CPU/GPU backend remain.

## What braided-river morphodynamics needs

| Component | Why it matters | Current release scope |
|---|---|---|
| Depth-averaged flow, wetting/drying, conservative water fluxes | Channels divide, bars emerge and floodplains flood | Shared explicit SSPRK22 augmented-f-wave CPU/GPU solver |
| Bedload capacity and conservative Exner update | Erosion, deposition and bar growth | Implemented; explicit compatible capacity choices |
| Longitudinal/transverse bed-slope effects | Gravity redistributes sediment on bars and banks | Implemented, optional and method-dependent |
| Secondary currents | Bend transport direction and momentum redistribution | Implemented; depth-averaged closure, not resolved 3D vortices |
| Erodible banks/floodplain and slope collapse | Channel widening and planform evolution | Existing dry-bank sharing, surveyed toe/collapse and conservative slope redistribution; only admitted combinations |
| Grain mixtures, hiding/exposure and active/subsurface layers | Sorting, armouring and sediment supply | Implemented, 1–32 noncohesive fractions subject to formula compatibility |
| Finite erodible stock and non-erodible substrate | Prevent erosion below available material | Implemented; layer/base method is explicit |
| Hydraulic and sediment boundary conditions | Discharge, stage, sediment feed and outflow control the sediment budget | Implemented, including supported time tables and exterior-state definitions |
| Roughness and turbulence | Hydraulic resistance and lateral momentum mixing | Implemented; dynamic bedform feedback is optional |
| Suspended sand | Important where sand is not primarily bedload | Implemented for admitted noncohesive transport/settling combinations; not compulsory for a gravel-bed case |
| Conservative AMR and cropped-wall treatment | Resolve channels/banks efficiently without spurious wall flux | Implemented and tested; 2:1 levels, user-defined finest size |
| Restart, cancellation, provenance and results | Make the solver usable and reproducible | Existing unified route; installed-release checks required |

Not every process must be enabled. Start with hydrodynamics only or one-grain
bedload, then enable the processes justified by the river. MPM is a capacity
formula, not an exclusive whole-model choice: the resolver determines compatible
slope, secondary-flow, mixture and layer settings and records their identities.

Implementation and short CPU/GPU regression evidence do not establish that
every combination reproduces a measured braided river. Existing bend, bank,
wall, wet/dry and component comparisons remain evidence within their tested
scope. A complete measured braided-river validation remains later work, not a
reason to add more features before release.

## Historical alpha123 interface and limits

Alpha123 offers exactly two Mesh strategies: **Dynamic AMR — water depth
only**, and **Static mesh — terrain slope breaks only**. Dynamic mode protects
wet cells, their buffer and dry open-boundary entrances, without terrain tags.
Static mode selects refinement from the initial DEM and keeps those same cells
throughout the run, with no runtime mesh assessment or regrid. Both retain
finest inlet/outlet cells and any explicitly surveyed bank corridor. Walls do
not force refinement: they follow depth/terrain criteria and 2:1 grading.
Static interior cells can stay coarse when flooded; changing bed features do
not trigger new refinement. Terrain defaults are a 5-degree slope change and
0.10 m vertical-detail threshold, with one finest-cell halo. A planar slope
alone does not trigger refinement. Cropped boundary fragments can be smaller
than the nominal selected cell size.

Project schema v17 is required; older projects and checkpoints are rejected,
not migrated. Both backends use the same augmented-f-wave midpoint residual,
now solved with full-residual Newton directions and a cheap contractive trial.
Friction follows each hydraulic substep. Automatic temporal control compares
one full step with two half steps and accepts the conservative half-step path.
The saved local tolerances are 0.0004 m depth, 0.004 m²/s momentum and 0.004
relative; nonlinear absolute tolerance is 1e-7. These are local error controls,
not a guarantee of millimetre accuracy over an arbitrary simulation duration.
No case-specific hydraulic timestep cap is embedded in the solver.

Mesh includes **Review axis-aligned inlets/outlets…**. This proposes a new
polygon and displays changed boundary widths and endpoint positions. Acceptance
replaces generated faces but retains the original polygon and saved project.
Save the accepted geometry as a new project; its geometry-review sidecar records
the changes. Only inlet/outlet vertices move; wall-only vertices remain unchanged.
Wall segments sharing open-section endpoints reconnect to the moved endpoints.
Walls are not rasterized into stair steps and do not impose finest refinement.
Preparation never silently snaps an existing polygon. Cropped wall cells remain
possible, including beside open-section corners, and can restrict the timestep.

The experimental small-cell rate-redistribution treatment is NOT included:
it conserves water but changes Test1 depths too much. No arbitrary timestep
increase or explicit fallback is hidden in either mesh mode. Sediment remains
unchecked by default. The alpha113/112 notes below are historical release notes,
not an instruction to load their old project formats.

The preceding alpha118 release fixed rollback after a sediment retry when an output clock
supplies a NumPy timestep scalar. The saved hint is normalized to a Python
float before storage; equations, tolerances and timestep values are unchanged.
Alpha117 was superseded during installed-interface qualification.

Alpha113 makes the Mesh assessment interval **1 second by default**, with
larger values available. This is a physical-time remesh clock, not the
hydraulic timestep. Wet cells retain a configurable buffer; open boundaries
also retain a dry entrance buffer before water arrives. Increase the buffer
for faster wet fronts or longer intervals. Dry coarsening delay is counted
in assessments, so its physical duration increases with the interval.
Conservative refinement now preserves already-existing sub-dry water traces
instead of failing or deleting them. No artificial water is added. Saved v13
project settings are preserved: set Mesh / Assessment interval to 1 s or
use the supplied alpha113 Test1 copy and prepare a new run. Prepared runs and
checkpoints from another runtime remain rejected. Sediment stays off by default.

Alpha112 adds persistent, rollback-safe hydraulic substep control and depth-only
AMR with a wet-front neighbour buffer. Simple dry-start hydraulics initializes
dry interiors coarse; all physical boundaries remain finest. Project format
v13 is required: old projects and prepared runs are not migrated automatically.
The Mesh tab has only the assessment interval, buffer and dry-coarsening delay.
Internal nonlinear retries and the next timestep target appear in run progress.
The wet/dry spatial operator remains piecewise: semi-implicit integration does
not guarantee arbitrary timesteps or zero nonlinear retries. The short 5 m
Test1 comparison gives 1.64x CPU and 2.32x GPU worker speedups, with maximum
saved depth difference 0.765 mm, not a universal performance/accuracy bound.

Alpha111 corrects a discontinuous cropped-cell hydraulic reconstruction that
could stall a dry-start inlet. Normal and critical inlet roots reuse exact
unchanged inputs; characteristic outlets still evaluate the current flow.
Conservative AMR maps use the validated hierarchy instead of repeating polygon
intersections. CPU and GPU retain the same semi-implicit augmented-f-wave
method, solver tolerances and explicit backend choice. Prepare a new run after
updating; checkpoints from an earlier runtime are deliberately rejected.

One interface: General, BC & IC, Mesh, Run and Results for hydrodynamics only.
Sediment and Bed & banks tabs appear when sediment transport is enabled.
Choose CPU or NVIDIA CUDA GPU explicitly in General. Both use the same
explicit SSPRK22 method. Sediment and optional physical effects start OFF.
No automatic backend change and no separate execution workspace.
The project-folder chooser stays visible above the tabs; new preparation/run
folders are placed beneath its `runs` subfolder.

Model-map tools are documented in [MODEL_SETUP.md](MODEL_SETUP.md). Domain and
guide creation/editing, boundary selection/reassignment, and surveyed bank/toe
geometry tools are available. Some spatial configuration inputs from the old
interface still need connections to the unified route; the comparison table
lists them explicitly. A first solver delivery is not full old-UI parity.

The Windows package includes the BRAIDS native DLLs; no Fortran/C++ compiler
is required at runtime. An external Python environment with NumPy, SciPy,
Shapely and GDAL is still required. GPU also needs compatible CuPy/CUDA and an
NVIDIA driver. The first delivery is checked on this Windows/QGIS installation;
it is not yet an all-dependencies installer for an arbitrary fresh PC.

Create a new project. Old project/prepared/checkpoint formats are rejected;
there is no automatic conversion. Existing files are not deleted.

The current Results tab displays depth, speed, water surface and bed elevation,
plays saved frames and displays static or adaptive mesh epochs. The old
interface's PNG-sequence export, probe/transect time-series tools and boundary
flow plots are not yet carried into this tab. Those are explicitly deferred
usability work, not grounds to postpone a usable first solver release or retain
the old execution workspace.

Surveyed toe/block-bank methods require a user-defined bank/toe survey and a
protected finest-mesh corridor. They must not be described as an arbitrary
automatic migrating-bank detector. Planform evolution occurs through bed/bank
elevation change inside an erodible domain, not by moving the exterior mesh.

## Release acceptance only

1. Freeze the noncohesive source and compile/package its native dependencies.
2. Verify the packaged, checkout-independent CPU/GPU paths for hydrodynamics
   and simple bedload, plus representative existing options, AMR and restart.
3. Verify QGIS Prepare, Run, Stop, reload and basic Results with sediment OFF
   in a fresh dock; install in the default profile with a recoverable backup.
4. Publish the ZIP, exact manifest, checks and known limitations.

No cohesive integration, new formula, mesh-sensitivity study or long comparison
campaign belongs to this release checklist.
