# Four-level coupled AMR implementation

Status: implemented and qualified on the small numerical and QGIS cases below.
The scheduled-inlet reflux-tail failure is corrected and covered by regression
tests for conservation, rejection of resolved deficits, thread and restart identity.
[Results, limits and snapshots](../validation/amr/AMR_VERIFICATION.md).
A representative reservoir comparison remains the next application step.

The user chooses the base cell size. There are exactly four spatial levels,
L0–L3, with sizes `base`, `base/2`, `base/4`, and `base/8`. Water, suspended
fractions, active bed and every substrate layer share the same leaf mesh.
Hydraulic and sediment criteria independently request levels; their maximum
determines the required resolution. Dry remote terrain can remain at L0.

- [x] Exact nested polygon/cut-cell hierarchy, disconnected components and solid
  holes, stable node IDs, shared coarse/fine faces and 2:1 balance.
- [x] Small-fragment stabilization compatible with parent/child relationships.
- [x] Conservative native restriction and positive, lake-at-rest-preserving
  prolongation of water, momentum and every sediment inventory.
- [x] Dynamic native regridding with buffers, hysteresis and retained bed detail.
- [x] Wet-region minimum L1 and finest wet/dry front including dry bank buffer.
- [x] Geometry criteria: open BCs, wetted walls/obstacles and terrain details.
- [x] Hydraulic criteria: free-surface and velocity gradients, bores and jumps.
- [x] Sediment criteria: concentration, bed change/bedload, bank activity and
  grain-composition fronts; categorical material/roughness boundaries.
- [x] User minimum-refinement polygons and clear threshold controls.
- [x] Multilevel time integration and conservative synchronization of water,
  momentum, suspended and bedload fluxes; no duplicated boundary-ID forcing.
- [x] Independent discharge/concentration schedules match analytical per-ID
  integrals. Negligible coarse/fine scalar tails are repaired conservatively
  using only a neighbouring inventory of the same class/layer; resolved
  negativity still rejects the timestep.
- [x] Full adaptive checkpoint/restart identity, including hierarchy and timers.
- [x] QGIS preparation, initial preview, results/raster conversion and animated
  mesh use the actual hierarchy at each saved time.
- [x] Small cases separately exercise every refinement criterion. Coupled cases
  cross refinement interfaces, regrid, conserve mass and quantify added diffusion
  against a uniform finest-grid reference.
- [x] Same accepted states, hierarchy and ledgers with one/eight OpenMP threads.
- [x] Bounded performance measurement, packaged alpha and installation smoke.

Implementation sequence: geometry/state contract; native synchronized adaptive
stepping; time subcycling/flux synchronization; all refinement criteria and
coupled sediment; QGIS integration; focused qualification and alpha delivery.
Synchronized stepping is an intermediate correctness milestone, not completion
of the full requested AMR implementation. No real reservoir run is required
until the small cases pass.

Geometry and immutable material sampling are prepared once. Adaptation, state
transfer and evolution must run in compiled native code. Covered parents never
participate in physical budgets. Regrids occur at accepted synchronization
times. Face fluxes are computed once, then gathered in a deterministic order.
Byte identity is required across thread counts on the same executable/platform;
AMR versus a uniform mesh is an accuracy comparison, not byte equality.

Qualification: 15 criterion cases, five uniform-grid comparisons, actual 5,000+
cell one/eight-thread coupled execution and exact restart, scheduled-inlet
analytical budgets, 2,125 regressions,
and 93 extracted-package QGIS checks pass. Candidate geometry and fine terrain
are currently prebuilt: active mesh reduction is not a claim of sparse preparation
memory. Unmergeable tiny cuts retain their true areas and CFL restriction.
