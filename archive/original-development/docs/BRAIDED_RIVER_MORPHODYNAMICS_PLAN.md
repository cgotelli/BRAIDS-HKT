# BRAIDS braided-river morphodynamics plan

## Goal and governing rule

The goal is a conservative, parallel two-dimensional model able to reproduce
steep gravel-bed flow, sediment sorting, bar formation, channel splitting and
merging, bank retreat and the morphological evolution of a laboratory braided
river.

Implementation follows evidence, not menu growth. Every physical element must
first pass a small Delft3D comparison designed to isolate that element. BRAIDS
and Delft3D use different grids and numerical schemes, so equality of every cell
is neither expected nor desirable. They must agree on the governing trend,
integral conservation, equilibrium state, propagation speed and statistically
meaningful spatial pattern within limits fixed before running the comparison.

## Definition of done for every validation

No feature is marked qualified unless its case produces both a Delft3D result
and a BRAIDS result. Each case must also provide:

1. an exact Delft3D source lock and executable/container digest;
2. one solver-neutral physical specification with units and sign conventions;
3. generated model-specific inputs, never two manually drifting specifications;
4. raw-output checksums and canonical CSV/NetCDF extracts on common stations or
   a common analysis grid;
5. water and per-fraction sediment mass budgets, including boundary and bed
   exchange terms;
6. time-series, profiles, difference plots and machine-readable metrics;
7. acceptance limits declared before the qualification run;
8. at least three spatial or temporal resolutions for convergence-sensitive
   tests, with equal physical resolution where practical;
9. an independent analytic or manufactured reference whenever one exists; and
10. deterministic restart and thread-count checks for the BRAIDS production
    executable.

Formula-oracle comparisons should approach floating-point agreement when the
same algebra and inputs are used. Local operators should normally agree within
0.1–1%. Short integrated cases should target sediment-budget differences below
5%, bed/load profile correlation above 0.95 and feature position within one
common analysis cell. Long morphological cases use envelopes and statistical
metrics rather than arbitrary cellwise tolerances. Every case may tighten or
relax these values only with a written physical and numerical justification.

## Validation repository layout

New work belongs under `validation/sediment/delft3d_current/<case>` and does not
overwrite historical reports. A case contains `specification.json`, generators
under `delft3d/` and `braids/`, `run_delft3d.*`, `run_braids.*`, `compare.py`, a
raw manifest, `metrics.json`, figures and `REPORT.md`. Source and executables
remain untracked. A preflight stops before computation if a hash, unit, coordinate
orientation, formula identifier or executable digest differs from the case lock.

## Phase 0 — make comparison trustworthy

### Progress on 2026-09-22

- The exact audited source commit, `c05ed672f0a7fe81c5b9867ef354cfdd53c25379`,
  now has a locally built Delft3D4 Release runtime. It was built with the
  official `doc/delft3d.Dockerfile`, the `d3d4-suite` configuration, and Intel
  oneAPI ifx 2024.2.1 in Docker Engine under WSL Ubuntu 24.04.
- Preflight now verifies all ten locked sediment source files, the supplied
  source archive, the immutable Docker image ID, the source commit, the
  executable SHA-256, and the runtime banner. The complete gate passes.
- A repeatable F34 2DH smoke run completes 300 steps with zero errors and zero
  warnings and inventories all four NetCDF outputs. The report is retained at
  `validation/sediment/delft3d_current/reports/runtime_smoke.json`.
- Windows-safe observer-ledger replacement now preserves and rolls back the old
  ledger, recovers an interrupted replacement before any restart read, and is
  covered by an interruption regression test.
- The focused sediment regression set is green.
- The Phase 0 paired exit case now passes. It directly compiles current-source
  Delft3D Van Rijn 1993 and MPM kernels for 20 formula states and runs both
  native solvers on one shared widening-flume specification at 0.5, 0.25 and
  0.125 m resolution. It retains immutable runtime/source identity, hydraulic
  and Exner comparisons, native equilibrium-inlet loads, and independent
  BRAIDS/Delft3D sediment budgets. The one-command runner and report live under
  `validation/sediment/delft3d_current/bedload_formula_matrix`.

Deliver this phase before changing production sediment physics.

- Build or obtain Delft3D FLOW/FM from the exact locked commit and record the
  compiler, options, runtime dependencies and executable SHA-256. If that exact
  build is impractical, lock a released binary and obtain its matching source;
  never mix versions in one qualification claim.
- Add a source/binary preflight for the new comparison line. Retain old reports
  unchanged and label their provenance explicitly.
- Add a `validation` dependency set with `xarray`, `netCDF4`, `shapely`, plotting
  and table packages. Make report readers lazy so missing optional packages do
  not prevent unrelated unit tests from collecting.
- Fix Windows-safe observer/restart output replacement and make the complete
  sediment suite green on Windows. Repeat on Linux for Delft3D automation.
- Record which executable path every test exercises: production cut-cell,
  standalone kernel, or legacy WAVE.

Exit gate: a one-command smoke case runs both executables, captures provenance,
extracts results and compares budgets without manual editing.

**Exit status: complete on 2026-09-22.** The next work remains in Phase 1; this
does not qualify suspended load, slope effects, mixtures, banks, or the final
braided-river experiment.

## Phase 1 — qualify bed-load capacity and direction

Use tiny uniform-flow and rotated-flow domains. Each item gets an exact-source
formula oracle and a full two-dimensional Delft3D/BRAIDS run.

1. **Hydraulic forcing and roughness.** Match depth, velocity, shear velocity,
   Chezy/Manning conversion and grain roughness before enabling morphology.
2. **Formula matrix.** Sweep water depth, velocity, slope and grain size around
   threshold for Van Rijn 1993, MPM, Ashida-Michiue, Soulsby, Wilcock-Crowe,
   modified Wilcock-Crowe and Gaeuman. Verify zero/subthreshold behavior,
   magnitude, units and monotonicity.
3. **Transport direction.** Rotate the same domain through 0°, 30°, 45°, 60° and
   90°. Compare vector magnitude and angle, not grid components alone.
4. **Bed-slope correction.** Test longitudinal, transverse and combined planar
   slopes below and near the angle of repose.
5. **Mixtures and hiding/exposure.** Use two- and three-fraction uniform beds.
   First qualify Egiazaroff; then implement and separately qualify the Delft3D
   alternatives that matter for gravel mixtures.
6. **Boundary load.** Compare equilibrium and prescribed total/per-fraction
   supply at an inlet, including oblique boundaries and a restart.

### Progress on 2026-09-22

- All seven selectable bedload laws pass formula magnitude, threshold behavior,
  monotonicity and vector direction against routines compiled directly from the
  locked Delft3D source (70/70 states; maximum relative error 3.00e-11, from
  harmless operation-order rounding in Soulsby's density terms).
- Their native equilibrium-capacity/Exner cases pass at three resolutions. The
  Wilcock-Crowe, modified Wilcock-Crowe and Gaeuman cases use three distinct
  surface fractions and independently gate each class budget, inlet load and
  initial composition. Fine-grid mixture inlet-load ratios span 0.9588-0.9647
  in total and 0.9485-0.9678 by class; bed-profile correlations span
  0.9496-0.9643. Both solvers satisfy the frozen 1% total and class budget gate.
- Rotated full-domain transport direction and explicit `ISlope=2` bed-slope
  corrections are current-source qualified. The 25-state equation oracle is
  exact, and five paired native planar channels at 0, 30, 45, 60 and 90 degrees
  pass vector, rotation-invariance, Exner and budget gates. Corrected native
  transport differs by 0.57-0.59% in magnitude and 0.164 degrees in direction.
- All four Delft3D hiding/exposure relations are now equation-exact against the
  locked source and pass paired three-fraction native sorting, armoring, Exner
  and class-budget checks.
- Sediment boundaries are current-source qualified. Native equilibrium inlet
  loads agree to 0.65% in total and 0.57-0.74% by fraction; the bed-change
  profile correlation is 0.957. Because `EqmBc`/`.bcc` has no independent
  prescribed bedload-rate field, its measured `SBTRC` class loads are replayed
  through the BRAIDS prescribed boundary and are accepted to machine precision.
  The separate Delft3D `.mor`/`.bcm` morphology transport boundary is qualified
  in Phase 2 step 3. A separate
  30-degree polygon contract passes two inlet IDs, a grouped outlet, exact
  per-class budgets, OpenMP equality and restart equivalence.

**Phase 1 exit status: complete on 2026-09-23.** All selectable bedload laws,
direction and slope correction, graded-bed hiding/exposure, equilibrium and
prescribed boundary supply now pass their declared current-source or exact
contract gates. The next milestone is Phase 2 conservative Exner evolution,
starting with the closed/uniform zero-change cases and the manufactured affine
capacity case.

Exit gate: all selectable bed-load laws either pass their declared operating
range or are visibly marked experimental/disabled in release builds.

## Phase 2 — qualify conservative Exner evolution

Run at morphological factor 1. Start with one fraction and fixed hydraulics,
then enable hydrodynamic coupling.

1. Closed-domain zero-flux conservation and a uniform-equilibrium domain with
   exactly zero bed change.
2. A manufactured affine-capacity field with an analytic constant Exner rate.
3. An inlet supply step producing separately measurable aggradation and
   degradation fronts.
4. A translating bedform/Zhu-style case compared to the analytic solution,
   Delft3D and BRAIDS over the whole event, not only a favorable late window.
5. Finite active-layer exhaustion, non-erodible rock exposure and substrate
   entrainment.
6. Wetting/drying over an emerging bar, including positivity, no transport
   through dry closed faces and complete sediment accounting.
7. The same oblique front on Cartesian, cut-cell and rotated geometries.
8. Fixed fine grid versus AMR, with reflux conservation and refinement-history
   independence; compare both physical outcomes with Delft3D.
9. Time-step, grid and thread-count convergence, plus bitwise or tolerance-based
   restart equivalence.

**Phase 2 progress (2026-09-23): steps 1-9 complete.** The locked current
Delft3D4 runtime and production BRAIDS executable both retain exactly zero bed
change and zero inventory change in the closed and sub-threshold uniform
controls. The affine-capacity manufactured solution passes all BRAIDS scheme
5/6 and 1.0/0.5 m cases with maximum relative rate error below `8.7e-14` and
conservative residual below `8.9e-16 kg/s`. The pinned Delft3D `bott3d.f90`
finite-volume specialization matches the same analytic mass update within
`6.6e-18 kg/m2`. The complete result is in
`validation/sediment/delft3d_current/exner_foundations/reports/REPORT.md`.
The paired inlet-supply qualification then imposes a shared physical load with
Delft3D's `IBedCond=4` morphology boundary and BRAIDS's prescribed boundary.
Equilibrium capacities agree within 0.57%; 150% and 50% supply steps produce
the expected aggradation and degradation fronts. Final profile correlations
exceed 0.9999 and peak magnitudes agree within 1.4%, with all 28 gates passing.
The complete result is in
`validation/sediment/delft3d_current/inlet_supply_fronts/reports/REPORT.md`.
The complete 300 s translating-bedform event now passes all 38 gates on 1.0
and 0.5 m grids. Against the independent pre-shock characteristic solution,
the fine-grid final normalized profile errors are 6.66% for BRAIDS and 7.25%
for Delft3D; their direct profile correlation is 0.999679, centroid positions
differ by 0.265 m and peak amplitudes differ by less than 1%. Both solvers
improve with refinement, retain the bedform volume within 1.32%, close their
sediment budgets, and keep depth/velocity errors below 1.8% throughout every
saved noninitial time. The complete result is in
`validation/sediment/delft3d_current/translating_bedform/reports/REPORT.md`.
Steps 5-7 now qualify finite active-layer/substrate exchange and rock exposure,
wetting/drying over an emerging mobile bar, and aligned/45-degree supply-front
geometry. Their portable reports retain native BRAIDS/current-source Delft3D
comparisons, per-fraction or differential Exner budgets, and OpenMP identity
where applicable.

Step 8 now passes all 23 frozen fixed-grid/AMR gates. On the same translating
bedform, final fixed/AMR bed-profile L1 is 0.762%, correlation is 0.999879 and
centroid difference is 3.1 mm. AMR averages 74.5% of the uniform leaf count,
uses five saved mesh configurations, closes its sediment budget to `7.44e-11`
relative residual, and reproduces the binary frames exactly in a clean repeat.
The complete result is in
`validation/sediment/delft3d_current/fixed_amr_equivalence/reports/REPORT.md`.

Step 9 passes all 36 frozen convergence/restart gates. Spatial characteristic
errors decrease from 7.701% to 6.994% to 6.661% on 2.0/1.0/0.5 m grids, and
successive solution distances contract by a factor of 0.2446. The
0.05/0.025/0.0125 s sequence has a normalized Cauchy envelope of `7.935e-06`;
its non-monotonic formal-order ratio is retained explicitly because spatial
error dominates. One/eight-thread frames are binary-identical. A one-thread
150 s checkpoint resumed on eight threads matches every common state and
ledger exactly and produces the identical final checkpoint. The complete
result is in
`validation/sediment/delft3d_current/convergence_restart/reports/REPORT.md`.

Phase 2 delivery is complete. The fail-closed integrated builder covers all
nine steps and all 222 acceptance gates, hashes every milestone evidence
package, and publishes the same compact manifest in the QGIS plugin. The
human-readable result is in
`validation/sediment/delft3d_current/phase2_qualification/reports/PHASE2_QUALIFICATION.md`.

Exit gate: sediment budgets close, fronts converge, bed change has the correct
sign and speed, and AMR does not introduce a material morphological bias.

## Phase 3 — graded beds, armoring and stratigraphy

### Progress on 2026-09-23

The first Phase 3 milestone is complete. The retained current-source
three-fraction expanding-flume result was freshly rerun, then a new zero-feed
two-fraction channel isolated selective surface armoring at 0.50 m and 0.25 m
resolution. All 29 frozen gates pass. On the fine grid the final coarse surface
fractions are 0.31286 for BRAIDS and 0.31284 for Delft3D, surface and bed-change
correlations are 0.9958, every eroding cell armors, and the BRAIDS sediment
budget residual is `6.33e-13`. D10, D50, D90, Dg, active-layer thickness,
first-substrate composition, and per-fraction budgets are retained. The report
is in
`validation/sediment/delft3d_current/graded_bed_baseline/reports/REPORT.md`.

The second Phase 3 milestone is also complete. An identical four-layer finite
bed was run in both native solvers with a 90% fine-gravel pulse buried beneath
a coarse cap. All 19 frozen gates pass. BRAIDS/Delft3D erode 14.216/15.077 mm,
recover the pulse at the surface at 51/57 s, correlate at 0.9971 in final bed
change, and close sediment budgets to `1.98e-12`/`1.37e-6`. The audit also
established the correct mode mapping: BRAIDS' three substrate layers correspond
to Delft3D `NEuLyr=2` plus its variable base, with `UpdBaseLyr=1` required for
finite conservative inventory. The report is in
`validation/sediment/delft3d_current/stratigraphy_pulse/reports/REPORT.md`.

The third Phase 3 milestone is complete. Constant 2, 4 and 8 mm active layers
and `max(0.008 h, 0.002 m)` water-depth scaling were paired in both native
solvers. All 26 gates pass. The scaled final active-layer profiles differ by
0.001312 mm on average with 0.999953 correlation, bed-change profiles correlate
at 0.997086, and BRAIDS is byte-identical with one and eight threads. A direct
Delft3D `IDiffusion=1` run instead failed after four steps with negative layer
masses and hydraulic blow-up. The decision is therefore to retain constant
thickness as the default, expose qualified depth scaling, keep ordered layers
and a finite updated base, and exclude vertical diffusion, an infinite
constant-composition base, and dune-height scaling from the steep-gravel target.
The report is in
`validation/sediment/delft3d_current/active_layer_sensitivity/reports/REPORT.md`.

Phase 3 integration is also complete. The fail-closed builder covers all three
milestones and all 74 gates, verifies complete D10/D50/D90/Dg diagnostics,
hashes every evidence package, and publishes a byte-identical compact manifest
to the QGIS plugin. The Sediment tab displays its sorting, stratigraphy,
active-layer, budget, determinism, rejected-mode and limitation diagnostics.
The full result is in
`validation/sediment/delft3d_current/phase3_qualification/reports/PHASE3_QUALIFICATION.md`.

Phase 3 delivery is complete in Windows alpha 61. The reproducible packager
rejects stale native sources, unexpected DLLs, unpinned licenses, failed
qualification manifests, raw runs and Delft3D source. The resulting archive
verified all 74 files; its packaged deck-v26 executable reproduced the retained
dynamic-layer frame hash exactly and the same verified stage was installed in
the default QGIS profile. The release record is in
`docs/WINDOWS_ALPHA_61_RELEASE.md`.

**Phase 3 exit status: complete on 2026-09-23.** The next scientific milestone
is Phase 4, beginning with a separately gated general bar/dune avalanching
operator rather than extending the qualified active-layer model implicitly.

- Re-run the expansion flume and a two-fraction sorting/armoring channel with the
  current reference.
- Add substrate exposure, buried pulse recovery and multi-layer exchange cases.
- Compare surface and subsurface grain-size distributions, not bed elevation
  alone: D10, D50, D90, active-layer thickness and per-fraction budgets.
- Test variable active-layer thickness and physically justified mixing. Add
  variable porosity only with its own compaction-free conservation case.
- Decide, from sensitivity evidence, which Delft3D bed-composition modes are
  necessary for steep gravel braiding rather than copying every coastal option.

Exit gate: BRAIDS reproduces the direction, rate and equilibrium of armoring and
the exposure of a buried layer without creating or losing any fraction.

## Phase 4 — processes required for braided rivers

Implement one item at a time, behind an experimental flag until its gate passes.

1. **General repose avalanching.** Extend beyond bank-zone diffusion so bar and
   channel faces relax conservatively. Compare an over-steep dry/wet cone and an
   evolving bar with Delft3D `duneaval`.
2. **Secondary flow and transverse transport.** Implement spiral-flow influence
   on transport direction. Use a mild bend and a symmetric bifurcation to test
   transverse sorting and branch load partition.
3. **Lateral erosion and bank failure.** Separate toe erosion, undercutting,
   cantilever collapse and repose redistribution. Begin with a straight eroding
   bank, then a bend and bifurcation. Compare retreat rate and volume budget to
   Delft3D where equivalent; also require laboratory evidence because models
   may encode different bank physics.
4. **Non-equilibrium/adaptation length.** Add only if steep-gradient tests show
   local equilibrium is inadequate. Validate a capacity step and rapidly varied
   flow over a crest.
5. **Morphological acceleration.** Implement constant, then time-varying MorFac.
   Compare MorFac 2, 5, 10 and 20 against MorFac 1 in both codes. Publish the
   maximum acceptable factor for each flow regime; speed is not evidence.
6. **Bedform roughness feedback.** Add if flume sensitivity shows it changes bar
   wavelength, transport partition or water level materially.

Phase 4 progress on 2026-09-23:

- Milestone 1 is qualified and released in Windows alpha 62. The general
  wet/dry repose operator passes 28 frozen Delft3D/reference gates, per-fraction
  conservation, AMR, restart and OpenMP identity checks.
- Milestone 2 is qualified for equilibrium planar secondary flow and is
  integrated in Windows alpha 63. The implementation reproduces the locked
  D-Flow FM curvature and spiral equations and passes all 15 frozen mild-bend
  and symmetric-bifurcation gates. Across both grain fractions, Exner-effect
  correlations are 0.9516--0.9596 and mean transport-direction differences are
  4.38--6.58 degrees. It preserves scalar bed-load magnitude, applies no
  `BetaSpiral` momentum force, and remains bitwise exact across OpenMP, restart
  and changing AMR meshes.
- Milestone 3 is qualified and integrated in Windows alpha 64. The v29 operator
  separates excess-shear toe erosion, persistent undercut history, discrete
  threshold block failure and conservative per-fraction deposition. All 13
  analytic, conservation, laboratory-direction, current-source Delft3D,
  restart, AMR and OpenMP gates pass. Post-collapse repose uses the separately
  qualified general avalanche operator. The older dry-bank replenishment is
  retained as a clearly labelled mutually exclusive legacy comparison.
- Milestone 4 is qualified. Compiled pinned Delft3D `tranb2` and BRAIDS MPM
  capacities agree to `2.94e-15` in their common local-equilibrium limit, but
  the measured capacity-step response and rapidly varied gravel crest exceed
  the frozen materiality gates. Native deck v30 therefore adds an optional,
  default-off steady first-order streamline relaxation with one class length
  `L_g = C_L D50_g`; the measured moderate-transport mean `C_L=100` is the
  editable default, not a universal calibration. Exact exponential, zero-
  length equilibrium, per-fraction, conservation, wet/dry, AMR, restart and
  one/eight-thread gates pass. The qualified control is exposed in QGIS.
- Milestone 5 is qualified and integrated in Windows alpha 66. Native deck v31
  applies constant or Delft3D-compatible block-interpolated MorFac to the
  conservative bed-load/Exner flux while leaving capacity unchanged. All 22
  retained source, clock, budget and regime-classification gates pass against
  the locked current-source Delft3D runtime. MorFac 20 is the published maximum
  for the quasi-steady subcritical translating bedform; the wet/dry hydraulic
  transient requires MorFac 1 in both models. AMR reflux, multi-fraction
  budgets, restart and one/eight-thread identity are retained. Suspended
  exchange, time-varying hydraulic boundaries and bank erosion/failure fail
  closed.
- Milestone 6 is qualified and integrated in Windows alpha 67. A calibrated
  `ks=3 D90` control shows that Fredsoe-MPM/Van Rijn 1984 feedback changes mean
  water level by 0.3221 m in locked current-source Delft3D4 and 0.2858 m in
  BRAIDS (effect ratio 0.8874), well above the frozen materiality threshold.
  Native v32 is blank by default, has no hidden prognostic state, preserves
  conservation/wet-dry AMR/restart/OpenMP identity, and emits a cell-resolved
  diagnostic. Unsupported process combinations fail closed.
- The next milestone is the integrated experimental braided-river simulation:
  freeze geometry, forcing, grain mixture and observation metrics; establish a
  MorFac-1, grid-converged reference; then enable only the qualified Phase 4
  processes required by that experiment.

Waves, dredging and marine fluff are not on the critical path for the first
inland gravel-bed braid model. They remain explicit non-goals until the core
river milestones pass.

## Phase 5 — suspended and cohesive sediment

Only after bed load and Exner are stable:

- qualify settling and advection without exchange;
- qualify erosion/deposition exchange in a zero-dimensional column and a short
  basin;
- qualify combined suspended and bed load with one shared per-fraction budget;
- add hindered settling and stratification effects where supported by evidence;
- add flocculation, fluff, consolidation and burial as separate optional models,
  each with Delft3D and independent laboratory cases.

Operator splitting must be measured against a smaller-step coupled reference.
If splitting controls the error, source terms move into a coupled higher-order
stage rather than being compensated by calibration.

## Phase 6 — integrated synthetic braided channel

Use a wide, initially shallow gravel channel with a small reproducible bed
perturbation and enough length for bar growth. Run Delft3D and BRAIDS from one
specification at three resolutions and at MorFac 1 before acceleration.

Compare discharge partition, water-surface slope, sediment budget, active width,
number of channels, braiding index, bar wavelength and migration speed, erosion/
deposition distributions, grain-size sorting and bifurcation stability. Repeat
with perturbation seeds to distinguish physical statistics from one realization.

Exit gate: both codes occupy overlapping convergence envelopes for integral and
pattern metrics and BRAIDS remains conservative with AMR enabled.

## Phase 7 — experimental braided-river simulation

Select a laboratory dataset only if it provides DEM time series, hydrograph,
sediment feed and gradation, outlet condition and uncertainty. Prefer a steep
gravel-bed experiment with repeat runs and withheld observation times. Archive
the dataset citation, license and immutable checksum.

1. Build one shared geometry and forcing specification for Delft3D and BRAIDS.
2. Calibrate only parameters that correspond to measured uncertainty, using the
   first part of the experiment. Do not tune numerical resolution separately to
   make either model look better.
3. Freeze parameters and predict withheld DEMs and time series.
4. Report water level/discharge, total and fractional sediment yield, DEM of
   difference, active width, braiding index, bar statistics, channel migration,
   bank retreat and mass closure with observational uncertainty bands.
5. Run grid/time/MorFac sensitivity and at least one alternative perturbation.
6. Publish successful and failed comparisons for both codes.

The final claim is not that BRAIDS reproduces Delft3D cell by cell. It is that
BRAIDS independently conserves the governing quantities, passes analytic and
isolated-physics tests, matches Delft3D within declared numerical envelopes,
and predicts withheld laboratory braid evolution with defensible uncertainty.

### Selected experimental benchmark

The first integrated target is Johannot et al. (2026) run `UC1_C1_LF_01`, a
10 m by 1.5 m, 1.16% slope gravel-bed flume with a 12-degree lateral inlet.
The source lock and executable contract live in
`validation/sediment/johannot2026_braided_flume/`.

- H2 is the calibration observation; H3 and H4 are withheld predictions. The
  checksummed H1 GeoTIFF is excluded because its scale, extent and elevation
  range are incompatible with the physical flume and the other rasters.
- Fixed-grid qualification precedes AMR at 0.05, 0.025 and 0.0125 m, with
  morphology factor 1.
- The measured hydrograph and sediment feed are conservatively normalized from
  four 3,612-second acquisition blocks to four model hours without changing
  either integral.
- Production and scaling runs use 1, 8, 16 and 32 OpenMP threads, with exact
  same-executable output checks before any speed result is accepted.
- The full initial/input grain-size distribution and particle density remain a
  hard material-closure gate; `D84` alone will not be expanded into an invented
  distribution.
- The 0.05/0.025 m screen passes its timestep and physical gates but rejects
  0.05 m: native-grid integrated absolute bed-change volume differs by 10.95%,
  above the frozen 10% limit. The 0.0125 m confirmation exposed and now
  regresses a live normal-depth endpoint transition: adjoining reflective
  polygon edges are vertical section sidewalls, not open spill edges. Long
  runs checkpoint at every completed output frame. The repaired 32-thread run
  passes all budgets and the 0.025/0.0125 m grid gate (0.0907 mm bed MAE and
  3.15% absolute bed-change-volume difference), accepting 0.025 m for H2
  calibration.
- The six-member H2 material envelope now runs all six candidates concurrently
  with five OpenMP threads each (30 assigned logical processors). Recoverable
  1,000-step checkpoints preserve exact restart state, eliminate the slow
  second batch, and leave two logical processors for Windows, I/O and the two
  throttled Delft3D containers. Candidate selection remains H2-only.
- Hydraulic and morphodynamic planform metrics are frozen separately: wetted
  width/channel/discharge-partition metrics require hydraulic fields, while a
  2 mm uncertainty-based DoD mask defines active width, threads and bar
  geometry. This prevents DEM activity from being mislabeled as wet channels.
- The locked Delft3D4 H2 exporter and native startup smoke pass on the shared
  278 by 58 grid with seven fractions, four bed layers, MPM, Parker hiding,
  slope correction and MorFac 1. Its fixed initial-bed Q-H outlet is a declared
  approximation to BRAIDS' evolving critical outlet and requires a production
  sensitivity run.
- The outlet sensitivity is now frozen and passes the locked native startup.
  It uses the measured H2 final outlet section for the alternative fixed Q-H
  curve; the paired full-duration decks differ only in `basin.bcq`. The shared
  reducer verifies that isolation before calculating bed, planform, load and
  fraction-budget sensitivity. The full runs remain gated on the H2-only
  material selection.
- The post-selection intercomparison pipeline now resolves the winning
  candidate's recorded scheduler-assigned folder, prepares the isolated outlet
  pair, executes both locked serial Delft3D references concurrently, and feeds
  both to the shared reducer. It fails closed while calibration is unfinished.
- H2 analysis now machine-evaluates the frozen DEM MAE/correlation and
  morphology gates. Even a clean pass remains `incomplete` until hydraulic
  observations, bar migration, outlet-mass uncertainty and a sorting-error
  threshold supply the remaining required evidence; none is inferred.
- Both published H2 LSPIV videos now have reproducible multicore screening
  products in local coordinates and QGIS-readable GeoTIFFs. The GCP direction
  was audited, one/eight-process outputs are byte-identical, and eight processes
  give the best measured throughput. Bank-marker parallax at the water plane is
  still unresolved, so these fields remain diagnostics rather than an invented
  hydraulic acceptance gate.
- The LSPIV timing audit finds no defensible mapping from movie-header time to
  experiment time. Only the final 14,400 s saved state meets the frozen
  2.00 +/- 0.05 L/s discharge match, so the 7,200 s H2 calibration cannot be
  substituted. The common BRAIDS/Delft3D sampler rejects incomplete runs and
  will report every matched state after the selected candidate is continued
  through the full four-hour experiment.
- The full-experiment driver now stages that continuation without rerunning the
  selected H2 prefix. All six actual 7,200/14,400 s deck pairs retain identical
  restart fingerprints. On the 32-logical-processor host, BRAIDS uses 30
  OpenMP workers while the two serial Delft3D outlet cases execute concurrently.
  The H2 reducer, locked withheld-H3 reducer and common three-model LSPIV
  screening comparison follow. The driver remains gated on a passing H2
  selection, and H3 cannot rank or retune the candidate.
- The integrated Phase 4 ablation is now frozen ahead of H3. It reuses the
  material baseline and runs general avalanche, avalanche plus secondary flow,
  and avalanche plus secondary flow plus adaptation at 8/8/16 threads in one
  32-worker batch. The first profile passing all five observable H2 gates is
  selected. Bedform roughness is excluded by its live-normal-inlet scope, bank
  failure by the rigid flume walls, and morphological acceleration by MorFac 1.

## First execution sequence

The next work should proceed in this exact order:

1. ~~finish Phase 0 and obtain a matching Delft3D executable~~ (complete);
2. ~~create `bedload_formula_matrix`, `bedload_rotated_plane` and
   `bedload_boundary_supply` cases~~ (complete);
3. ~~create the uniform, supply-front and translating-bedform Exner cases~~
   (complete, expanded through all nine Phase 2 steps);
4. ~~repair failures until the Phase 2 cases and sediment regression suite are
   green~~ (complete: 222/222 retained gates);
5. ~~expose pass/fail status, budgets and convergence in plugin diagnostics~~
   (complete in alpha 60);
6. begin the Phase 3 graded-bed, armoring and stratigraphy qualification.

This sequence deliberately tests the foundations before adding more sediment
features. A failure at a later phase returns to the smallest isolating case; it
is not fixed by calibrating the final braided-river experiment.
