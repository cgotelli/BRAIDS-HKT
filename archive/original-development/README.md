# BRAIDS

## BRAIDS Desktop preview

The current UI development target is **BRAIDS 0.1.0-preview15**, a
standalone Windows interface adapted from AVAC Desktop. Launch it with
**Launch BRAIDS Desktop.cmd**. It retains the qualified explicit
CPU/OpenMP and NVIDIA GPU/CUDA solver, with sediment disabled by default and
one workflow for configuration, execution, results and project history.
Verified polygon geometry and the initial leaf mesh are reused during startup; numerical kernels remain
unchanged from alpha136. Prepare again after updating the runtime.
Mesh now supports permeable polyline breaklines with same-side cell merging and
a strict minimum area equal to the finest square. The v21 project schema requires
explicit breakline geometry (an empty list when unused); older projects and
checkpoints are rejected, not migrated. See the
[breakline qualification](validation/sediment/river_morphodynamics/reports/desktop_breaklines147/REPORT.md).
The [polyline UI correction](validation/sediment/river_morphodynamics/reports/desktop_polylines148/REPORT.md)
removes automatic polygon closure from Add breakline.
The [bend-merge repair](validation/sediment/river_morphodynamics/reports/desktop_merge149/REPORT.md)
handles stalled small cells with valid joint unions, prevents preview of failed
preparation folders and puts Generate mesh below the breakline controls.
The [dynamic AMR repair](validation/sediment/river_morphodynamics/reports/desktop_amr150/REPORT.md)
prevents map-coordinate rounding from falsely identifying a changed wall during
refinement/coarsening, while retaining exact inlet/outlet history and coverage checks.
QGIS is not required. Existing QGIS installations and historical results remain
untouched; no further QGIS UI development is part of this transition.

Start with **Help > Open demo**, then **Run**, or open a current project folder. See the
[desktop quick start](desktop/README.md) and
[interface qualification](validation/sediment/river_morphodynamics/reports/desktop_v1/REPORT.md) and
[geometry performance report](validation/sediment/river_morphodynamics/reports/desktop_geometry137/REPORT.md)
and [initial mesh handoff report](validation/sediment/river_morphodynamics/reports/desktop_mesh138/REPORT.md).
The [latest UI revision](validation/sediment/river_morphodynamics/reports/desktop_ui142/REPORT.md)
implements the updated `comments.md`: PNG/MP4 export, smooth wet-only results,
paused-flow particle tracks, switchable map decorations, single-edge selection,
resizable subtle window borders and verified preparation-asset reuse. The demo
copies the requested natural-river setup. The existing CMD shortcut launches
without a persistent console. Solver kernels are unchanged.
The [window correction](validation/sediment/river_morphodynamics/reports/desktop_chrome143/REPORT.md)
keeps light as the default and removes the native thick/rounded frame: the
window has only a square 1 px neutral-gray border and remains resizable.
Paused-flow particles now have bright heads and longer tapered, fading tails;
see the [particle presentation update](validation/sediment/river_morphodynamics/reports/desktop_particles144/REPORT.md).
The latest [layer-appearance update](validation/sediment/river_morphodynamics/reports/desktop_appearance145/REPORT.md)
removes particle halos and the terrain-opacity slider, adds result colorbar and
opacity controls to the existing appearance button, and names saved plots by field.
The [map-tools update](validation/sediment/river_morphodynamics/reports/desktop_tools146/REPORT.md)
adds cursor/hand icons, a temporary multi-point distance ruler and automatic
display of an already-generated mesh in Model & Run.
This is a locally qualified preview using the existing scientific Python
installation, not yet a self-contained installer for a clean Windows PC.

## Archived QGIS alpha127 introduction

The following release description is retained as historical evidence. The last
installed QGIS delivery is alpha136; it is not replaced by a new plugin here.

BRAIDS originated as a QGIS plugin for two-dimensional river hydraulics and noncohesive
morphodynamics. The archived unified release below is **0.1.0-alpha.127**. CPU/OpenMP
and NVIDIA GPU/CUDA use the same explicit SSPRK22 augmented-f-wave solver, selected
explicitly by the user. Sediment transport starts disabled. There is one
tab-based interface and no separate execution workspace.

Alpha127 exposes face-sum CFL (default 0.225) in Run, uses cell-local hydraulic
resistance and preserves exact open-boundary histories when walls regrid.
It retains inlet/outlet-only alignment and forces only their cells to the
finest level. Walls retain cropped polygonal cells and follow depth/terrain
criteria and 2:1 grading, with explicit surveyed-bank corridors still protected.
The geometry review requires approval; existing polygons are never silently
snapped. Dynamic depth-only AMR and static terrain-only refinement remain the
two mesh strategies. There is no implicit fallback or hidden backend selection.
See the [release report and comparison figures](validation/sediment/river_morphodynamics/reports/unified_qgis_v1/explicit_delivery127/REPORT.md),
[release scope](braids_qgis/FIRST_RELEASE.md), and
[current feature status](validation/RIVER_MORPHODYNAMICS_STATUS.md).
The package is [BRAIDS alpha127 for Windows QGIS](dist/unified-alpha127/BRAIDS-0.1.0-alpha.127-QGIS-windows-x86_64.zip).
Use a new v18 project and prepare again after updating; old projects and
old-runtime checkpoints are not migrated.

## Historical development notes

The following records describe earlier, separately bounded development paths;
they are not the current plugin's solver selectors or release specification.

Earlier bend development consolidated one corrected CPU/GPU implementation:
[`Flow2DHCurrentBed`](braids_core/sediment/flow2dh_current_bed.py).
It combines the qualified reconstruction, face-geometry and sediment-shear
corrections without optional numerical-variant switches. Earlier variants are
frozen comparison/restart references. See the
[consolidation results and comparison figures](validation/sediment/struiksma_t2_2d/reports/classic_mobile_matched_inlet_v3/current_bend_v1/REPORT.md).
This fully wet EH bend route has not yet replaced the installed QGIS worker;
the release status below remains separate from the current development baseline.

The historical **0.1.0-alpha.104** Windows release carried the generated Q10
product contract and the M20 desktop integration: explicit CPU or implicit-f-wave GPU ownership, automatic
compatible auxiliary-method resolution, typed morphology-output selection and
one fingerprint through project, preparation, result and restart. Component-only
routes remain visible but fail closed instead of being presented as supported.
The installed release is paired with the authenticated QGIS product worker
`dist/gpu-worker/BRAIDS-GPU-0.1.0-q10.8-windows-x86_64.zip`, including its
qualified recursive physical-toe/distributed-bank AMR routes. GPU execution
still uses a separately installed CPython/CuPy/CUDA environment.
Its guarded CPU default selects
SSPMS6 for eligible fixed-grid hydraulics, SSPRK(4,3) for eligible fixed-grid
MPM/secondary-flow runs, and the legacy method elsewhere. GeoClaw's augmented
f-wave solver, sediment reconstruction, existing CFL settings and the production
MorFac limit remain unchanged. The external GPU route now also admits explicit
plain SSPRK(4,3) for its qualified fixed-grid, nonbank MPM/exchange-off scope.
See [release details](braids_qgis/ALPHA_RELEASE_NOTES.md).
The previous cancelled comparison is preserved. A new cold-start bend comparison
against the completed 6.27-hour reference is defined by
[the finite runner](validation/sediment/lfm_bend_2d/fast_cpu_cold_comparison.py),
including automatic figures and 3 mm RMS / 5 mm maximum bed-difference limits.
That CPU trajectory has now completed. The maximum all-frame bed RMS difference
is 1.815 mm (within the 3 mm gate), but the largest local difference is 30.007 mm
(outside the 5 mm gate). Conservation checks pass; full long-run numerical
agreement has therefore **not** passed. See the
[completed CPU comparison](validation/sediment/lfm_bend_2d/reports/fast_cpu_cold20h_20261001_v1/RESULTS.txt).

The active GPU development work order is the
[implementation-first plan](validation/gpu_fwave/GPU_IMPLEMENTATION_FIRST_PLAN.md):
complete the required GPU physics, AMR and QGIS/Windows integration before we
jointly select focused observation-based experiments; the measured braided
experiment is last. The GPU narrative below describes earlier dated worker
snapshots and is not the current feature-availability table. Consult the
[current status](validation/gpu_fwave/GPU_IMPLEMENTATION_STATUS.txt) and
[control menu](validation/gpu_fwave/GPU_I0_CONTROL_MENU.txt) for that distinction.

The isolated [GPU hydraulic pipeline](validation/gpu_fwave/hydraulic_pipeline.py)
now keeps scheme-6 reconstruction, the augmented f-wave fluxes, supported constant
boundaries, conservative draining/gather, SSPRK(4,3) and Manning friction on the
GPU in double precision. Short evolving bend and wet/dry CPU comparisons and
their figures are recorded in the
[hydraulic qualification](validation/gpu_fwave/reports/gpu_hydraulic_20261001_v6/HYDRAULIC_REPORT.txt).
Optional [CUDA graph replay](validation/gpu_fwave/graph_execution.py) reduces
kernel-launch overhead for fixed timesteps without changing the arithmetic.
The new [adaptive-timestep GPU qualification](validation/gpu_fwave/reports/gpu_adaptive_hydraulics_20261001_review2/REPORT.txt)
reuses one graph across changing timesteps and extends the bend comparison to
20 physical seconds. Maximum depth difference is 0.00267 mm; the original strict
round-off-oriented bend gates failed, while separately documented physical
screens pass. Lake and wet/dry dam fields match the CPU bit-for-bit.

The experimental [coupled GPU worker](braids_gpu/__main__.py) now advances a
fixed-grid MPM morphodynamic bend: augmented f-wave hydraulics, unchanged sediment
reconstruction, conservative Exner/layer updates, prognostic secondary flow,
Manning friction and horizontal momentum diffusion are computed on the GPU in
double precision. Preparation, timestep/convergence decisions and file I/O remain
on the host. The backend is always a user choice: `--backend gpu` or `--backend cpu`.
There is no mesh-size selector and no silent CPU fallback.

The [coupled qualification report](validation/gpu_fwave/reports/gpu_coupled_worker_20261002_v1/REPORT.txt)
records a 20-second, MF25 bend and a separate 20,088-cell, MF20 cold-start check.
The latter took 5.15 s for the GPU process versus 25.58 s for the eight-thread
native CPU process, with a maximum bed difference of 0.00332 mm. This provisional
4.97x speedup includes startup/output but was measured under background CPU load;
it is not an isolated benchmark, a Delft3D timing comparison, or a long-run
morphological validation. Comparison figures are linked in that report.

Use a separate Python/CuPy environment, never QGIS's Python environment:

```powershell
validation\gpu_fwave\.venv_gpu\Scripts\python.exe -m braids_gpu capabilities
validation\gpu_fwave\.venv_gpu\Scripts\python.exe -m braids_gpu prepare --input PATH\case.data --output PATH\new_prepared
validation\gpu_fwave\.venv_gpu\Scripts\python.exe -m braids_gpu run --backend gpu --input PATH\new_prepared --output PATH\new_results --until 20 --output-every 1
```

The worker provides JSON progress, immutable NPZ frames, checkpoints, `--resume`
and `--stop-file`; Ctrl+C stops at an accepted-step boundary. It requires an
NVIDIA CUDA device and the pinned optional
[GPU dependencies](validation/gpu_fwave/requirements-gpu.txt). Input initialization
uses a bundled, authenticated native initializer, not CPU evaluation of time steps.
The integrated worker now connects fixed-grid hydraulics, bed load, noncohesive
suspended exchange, layered mixtures, secondary flow, supported viscosity,
adaptation, avalanche and bank modes 2/3. Separate authenticated routes provide
synchronous broad-physics AMR plus shared recursive stepping for admitted
nonbank transport, static/evolving physical-toe bank mode 2 and evolving
distributed-bank mode 3. The nonbank recursive route now admits its domain-wide
general avalanche and positive mode-1/geometry-1 velocity-Laplacian viscosity,
plus separate exact recursive equilibrium-bedform branches. The new composed
branches combine bedform feedback with either evolving distributed-bank mode 3
or evolving surveyed physical-toe mode 2 and one parent-owned avalanche after
the synchronized bank transaction. Their native initializers are export-only;
the physical-toe route derives its initial toe/donor/face map from the surveyed
bank geometry and rebuilds that map after topology changes.
Viscosity runs once on the synchronized full-leaf graph after fine/coarse
reflux and before optional avalanche. Its limiter controls area-weighted
velocity L2; variable-depth integrated momentum is not claimed to be exact.
Unsupported combinations fail before execution. The nonbank bedform branch
retains its documented unscheduled discharge/stage, exchange-off,
secondary-off, viscosity-off, avalanche-off and bank-off envelope. The new
banked branches are restricted to one constant unsampled stage boundary, MPM
friction mode 1, MorFac 1, evolving mode-2 or mode-3 banks as authenticated,
general avalanche and the other exact controls recorded in their integration
reports. Live hydraulic boundary forcing is not composed with the new mode-2
route. Cohesive sediment and complete experimental validation remain future
milestones. Alpha 96 supplies the
authenticated external-GPU Prepare/Run/Results
handoff, while the user still chooses CPU or GPU explicitly and there is no
fallback. Pair it with the v26 source worker under
`validation/g2_general_transport/runs/portable_integrated_worker_20261008_v26`.
That worker includes the native initializers and runtime DLLs; Python/CuPy and the
NVIDIA driver remain external requirements. Its general-AMR v8 archive pins the
corrected MPM stress-ratio slope kernel to component and dynamic-regrid/restart
receipts while leaving the historical v7 archive immutable. Its QUICKSTART
documents exact scope, stop/restart behavior and the eligible SSPRK(4,3) command.

Alpha 96 additionally exposes GPU **Implicit f-wave midpoint** as an explicit
choice for one narrow production envelope: plain fixed-grid, safely fully wet
low-Froude flow, constant forcing, nonnegative Manning, MorFac 1, and static
bed or weakly coupled MPM law 2 with 1--32 fractions and at least two bed
layers, with slope and hiding disabled. It retains the augmented f-wave
spatial residual and does not couple sediment into the nonlinear hydraulic
solve. Unsupported wet/dry, AMR, suspension, secondary-flow, viscosity, bank,
avalanche, bedform, adaptation and scheduled-forcing combinations reject; no
explicit-integrator or CPU fallback occurs. A 128-cell two-fraction gate with
a user-selected 20x larger time step measured a 22.276x warmed trajectory
speedup, 0.012315 mm maximum depth difference and 0.000006763 mm maximum bed
difference from fine SSPRK(4,3). This small test is not a reach-scale speed
claim. Schema-2 restart is byte-identical in the six-step split-run gate.

The short [recursive general-avalanche engineering report](validation/g4_bank_amr/RECURSIVE_GENERAL_AVALANCHE_INTEGRATION_REPORT.md)
records the bounded public full/cut/fresh-resume gate, conservation and wet/dry
checks, complete rollback after an injected post-avalanche rejection and exact
recovery across an actual topology change. This is implementation evidence, not
physical validation, calibration, a long-duration morphology result or a
reach-scale performance claim.

The [recursive positive-viscosity integration report](validation/g4_bank_amr/RECURSIVE_GENERAL_VISCOSITY_INTEGRATION_REPORT.md)
records the active wet coarse/fine viscous faces, a dry-interface regrid epoch,
viscosity-before-avalanche ordering, rollback, byte-exact fresh-process
recovery, authenticated v14 worker and installed alpha 89 readback. This
remains engineering evidence rather than physical validation.

The [recursive equilibrium-bedform integration report](validation/g4_bank_amr/RECURSIVE_GENERAL_BEDFORM_INTEGRATION_REPORT.md)
records the paired OFF/ON three-stage capacity and pre-friction feedback, two
automatic regrids, byte-exact fresh-process recovery, authenticated v15 worker,
and alpha 90 installed readback. This is engineering evidence; bend/braided
physical validation and reach-scale timing remain later gates.

The [recursive bedform/bank/avalanche integration report](validation/g4_bank_amr/RECURSIVE_BANK_BEDFORM_AVALANCHE_INTEGRATION_REPORT.md)
records the ordinary-deck preparation, export-only initializer guard, active
distributed-bank and bedform response, actual regrid, conservative parent
avalanche, byte-exact fresh-process recovery, five QGIS bedform rasters and
installed-QGIS readback. Its
[six-panel engineering figure](validation/g4_bank_amr/runs/recursive_bank_bedform_public_20261006_v7/GATE.png)
shows the accepted final state. This remains engineering evidence; the surveyed
physical-toe mode-2 composition is delivered separately below and physical
validation follows later.

The [recursive physical-toe/bedform/avalanche integration report](validation/g4_bank_amr/RECURSIVE_PHYSICAL_TOE_BEDFORM_AVALANCHE_INTEGRATION_REPORT.md)
records ordinary-deck preparation, the export-only mode-2 initializer guard,
toe supply, undercut and collapse, active parent avalanche, an actual
toe-intersecting topology change, exact fresh-process recovery, portable-worker
execution and installed-QGIS readback. Its
[six-panel engineering figure](validation/g4_bank_amr/runs/recursive_physical_toe_bedform_public_20261007_v3/GATE.png)
shows the accepted final state and histories. This is short engineering
integration evidence, not physical validation or reach-scale performance.

The [finite long GPU bend runner](validation/gpu_fwave/long_bend_run.py) advances
the original unmorphed bed for 2,880 hydraulic seconds at MF25 (20 morphological
hours), using the verified CPU hydraulic spin-up. It is not a GPU-only cold-water
warm-up. An exclusive copy of the qualified worker and the exact native input
are frozen before execution. Automatic final analysis compares both the direct
optimized CPU reference and the historical 6.27-hour result, retaining the
existing 3 mm RMS / 5 mm maximum gates, budgets and raw-cell figures/animation.
This separates GPU-port discrepancies from the pre-existing CPU-method difference.
The current qualification's [frozen plan](validation/gpu_fwave/reports/gpu_bend20h_20261001_v1/plan.json)
and [worker progress](validation/gpu_fwave/runs/gpu_bend20h_20261001_v1/gpu/status.json)
record the finite run. Its supervisor writes the final comparison and animation
automatically; there is no automatic extension or plugin installation.

The long GPU bend has completed: 39.9 minutes versus 73.6 minutes for the
optimized CPU reference. Its 6.05 mm maximum all-frame bed RMS and 80.1 mm
maximum local difference exceed the unchanged long-run agreement gates.
The subsequent [routine-by-routine audit and figure](validation/gpu_fwave/reports/gpu_routine_audit_20261001_v2/summary/REPORT.txt)
finds the first sampled local discrepancy in the secondary-flow curvature filter:
CPU and GPU satisfy the same residual tolerance but stop at different iterates.
Sixteen conditional Euler stages pass, forty helper checks match bitwise, and
actual graph replay matches plain GPU execution through variable timesteps and
rejections. Tightening the filter on **both** sides in private diagnostics reduces
complete-step bed differences to roundoff. This does **not** yet establish the
cause of the long-run drift or qualify a new production setting.
The [short evolving-window comparison and figures](validation/gpu_fwave/reports/gpu_filter_window_20261001_v1/comparison_reviewed/REPORT.txt)
now covers 3,200 common timesteps from the recovered 8-hour state, with three
sequential repeats per configuration. Tightening both filters reduces maximum
CPU/GPU bed disagreement from 3.35e-7 m to 2.46e-12 m; all conservation checks
pass. The default difference is already tiny over this short window. Stricter
filtering costs about 25% more GPU advancement time (31.33 to 39.17 seconds), so
it remains a diagnostic setting, not an adopted default. The subsequent
[boundary-location audit](validation/gpu_fwave/reports/gpu_boundary_location_20261001_v1/REPORT.txt)
places the long-run 80.1 mm hotspot in the inlet/wall corner. A
[boundary-only CPU/GPU A/B and figures](validation/gpu_fwave/reports/gpu_boundary_window_20261001_v1/CONCLUSION.txt)
now compares existing equilibrium secondary inflow with zero incoming intensity,
keeping original filtering and all interior physics unchanged. Four common-state
runs from 8 to 9 morphological hours pass conservation checks. Zero inflow does
not suppress corner deposition and does not improve the small CPU/GPU difference
(0.071 versus 0.082 mm). The historical large divergence is not reproduced in this
window, so neither a boundary fix nor tighter filtering is adopted. Reproducing
the later amplification period remains an unresolved historical audit item,
not the active implementation milestone. The CPU default, plugin and
experimental GPU package remain unchanged.

The [larger-timestep research and figures](validation/sediment/lfm_bend_2d/reports/larger_timesteps_20261001_v3/REPORT.md)
test higher CFL and a four-stage SSPRK(4,3) driver on frozen alpha-81 sources.
The isolated fixed-mesh candidate reduces the bend benchmark's timestep count
by 74% and computation time by about 61%, with measured numerical differences
rather than a bitwise-equivalence requirement. GeoClaw's Riemann solver and
sediment reconstruction remain unchanged. Alpha 82 adopts this method only
inside its tested fixed-grid scope; AMR retains the legacy method. A full
cold-start morphology trajectory is being checked separately.

The subsequent [hydraulic flux experiment](validation/sediment/lfm_bend_2d/reports/hybrid_hydraulics_20261001_v2/REPORT.md)
tests HLLE and hybrid f-wave alternatives. The f-wave retains the tested bed
patterns; HLLE fails the coupled bed-change screen. Separate builds show
12-19% compute savings, but the same-executable coupled control saves only
2.2%, below the incremental speed target. Neither method is promoted. The
report retains both the favorable timings and this important control.

The milestone notes below describe their original delivery dates. Statements
that the plugin was not updated refer to those earlier milestones; alpha 82
now includes the accumulated development implementation and its documented
limitations, not renewed experimental validation of every process.

The [parallel curvature solver milestone and figures](validation/sediment/lfm_bend_2d/reports/curvature_multilevel_20260930_v2/REPORT.md)
add an opt-in multilevel solver for the existing experimental secondary-flow
filter. In the longer repeated 80,342-cell bend replay, computation time falls
8.7% and whole-process time falls 7.0%, with very small result differences.
All 51 focused tests pass; thread and restart results are exact. Smaller meshes
retain MIC because it is faster there. GeoClaw and sediment reconstruction are
unchanged. This is **not a production-physics or QGIS release change**; developed
large-mesh morphology and multilevel AMR performance still need qualification.

The preceding [resumed sediment optimization screen and figures](validation/sediment/lfm_bend_2d/reports/sediment_workflow_20260930_v2/REPORT.md)
profile the current coupled solver and test six same-equation variants. The
secondary-flow curvature filter accounts for 16–36% of compute time on these
bend workloads. A viscosity cache saves phase time but is 1.8% slower overall
in the longer repeated test, so **none of these candidates is promoted**.
All 54 timed trajectories retain identical saved states and restart files;
production source, development runtime and installed plugin remain unchanged.

The [adaptive hydraulic milestone and figures](validation/sediment/lfm_bend_2d/reports/amr_workflow_20260930_v1/REPORT.md)
extends combined preparation to eligible AMR while preserving each level's
different cell subsets. It also parallelizes short subcycled hydraulic face
loops without a team-count cap, tested through 32 threads. Four- and six-level
pulse controls take about 24% less time at both eight and 32 threads; short
wet/dry timings and fixed-grid regression controls are reported separately.
All 222 timed trajectories preserve saved results; 24 targeted and 24
deployed/source-contract checks pass. This updates source and the exact tested
development executable, **not the
installed QGIS alpha-80 package**. GeoClaw and numerical formulas are unchanged.

The preceding [hydraulic-overhead milestone and figures](validation/sediment/lfm_bend_2d/reports/face_workflow_20260930_v3/REPORT.md)
reuses Euler work arrays and applies measured parallel-loop/trace-preparation
improvements. Compared with the already optimized previous milestone, hydraulic
engine time falls another 26.7% (5,120-cell rectangle), 40.3% (81,920 cells) and
34.5% (bend); active MPM controls improve 19–23%. All 90 timed trajectories retain
identical saved states and ledgers, and 83 targeted tests pass, including AMR
resizing and coupled physics. AMR performance was deferred to the newer milestone above. GeoClaw,
WENO/RK3 and sediment formulas remain unchanged. The exact tested executable is
now the repository development runtime; **the installed QGIS alpha-80 package
is not updated**.

The preceding [2026-09-30 hydraulic workflow audit and figures](validation/sediment/lfm_bend_2d/reports/workflow_adopted_20260930_v1/REPORT.md)
compare the active BRAIDS and stock Delft3D steps. The rebuilt development solver
reduces measured hydraulic engine time by 15.5% (5,120-cell rectangle), 20.2%
(81,920-cell rectangle), and 11.1% (bend), with bitwise-identical results and
33 passing focused tests. It fuses compatible preparation passes and skips
inactive sediment bookkeeping; GeoClaw, WENO and RK3 are unchanged. The optimized
path at that milestone was restricted to eligible fixed-grid, single-fraction
hydraulics with no active auxiliary physics. The newer milestone above broadens
trace preparation conservatively; the inactive-sediment bypass remains restricted.
This updates source/development runtime, **not the installed QGIS alpha-80 package**.
The report retains an initial small-run startup anomaly separately from engine time.

The current macOS Apple Silicon test package is
`dist/BRAIDS-0.1.0-alpha.37-QGIS-macos-arm64.zip`. It contains the QGIS plugin,
the `braids_core` package, `xbraids_cutcell`, and relocatable GNU Fortran/OpenMP
runtime libraries. See [the alpha release notes](braids_qgis/ALPHA_RELEASE_NOTES.md)
for installation and a manual acceptance route.

The current 64-bit Windows package is
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.92-QGIS-windows-x86_64.zip`, with its
matching GNU Fortran/OpenMP runtime DLLs. Alpha 92 supersedes alpha 91 in this
PC's default QGIS profile; its archive, installed-source and recursive
physical-toe/bedform/avalanche readback gates are recorded in the
[integration report](validation/g4_bank_amr/RECURSIVE_PHYSICAL_TOE_BEDFORM_INTEGRATION_REPORT.md).
Existing
prepared runtimes and results are never rewritten; exact-source GPU restarts
must retain their original worker.

The alpha92 ZIP SHA256 is
`58448ace5bdd1ae8183cf9d77d5cd82ef2b6069c5cbcc6133b276475a10f4e68`.
Its [recoverable installation receipt](dist/windows-x86_64/BRAIDS-0.1.0-alpha.92-INSTALL.json)
has SHA256 `2bcded289e5ee502e27802cfdd16360d046457268b054bcca412d64c5ba4b7e3`.
The [installed-profile readback receipt](validation/g4_bank_amr/runs/recursive_physical_toe_bedform_public_20261007_v3/QGIS_READBACK_ALPHA92.json)
has SHA256 `9fdec4b9f28b2bb519fe3c5b81fe812ff751b0fbd032e60ddcee220fc06cc3fc`;
that gate removes the checkout from Python's import path, authenticates the
actual installed manifest and reads the canonical GPU full/cut/resume results
without importing the worker, CuPy or CUDA modules.
The separate [mode-2/mode-3 compatibility receipt](validation/g4_bank_amr/runs/recursive_physical_toe_bedform_public_20261007_v3/QGIS_READBACK_ALPHA92_MODE2_MODE3_COMPATIBILITY.json)
has SHA256 `c8e4cc273f77d7ca23f7d6bc4c2b97ccb81fdb913138619168dd50242ee915ef`.
Using installed modules only, it authenticates canonical mode-2 public v3 and
the immutable alpha91 mode-3 public v7 full/cut/resume checkpoints without any
numerical or native-initializer execution.

Rebuilding the Windows release requires the pinned WinLibs GCC 13.2/MinGW-w64
toolchain with `gfortran` and `objdump` on `PATH`, and network access for the
builder's hash-pinned license/source downloads. QGIS must be closed before the
recoverable installer runs; pass an explicit destination for any non-default
QGIS profile.

An earlier physical-validation thread is the [reconstructed single laboratory bend](validation/sediment/jamieson2013_bend/README.md).
The latest development work corrects the [bank/bed/flow coupling](docs/BANK_BED_COUPLING.md)
after the [persistent-front audit](validation/sediment/jamieson2013_bend/reports/current_model_difference_audit/RESULTS.md).
The [completed 120 s paired comparison and figures](validation/sediment/jamieson2013_bend/reports/bed_coupling_milestone_v1/RESULTS.md)
records 89 passing regressions and restored bed/flow feedback. Both engines
still show crest-to-slope redistribution, not developed channel-floor scour or
a bar. Mobile artificial margins and an inactive floor prevent laboratory
qualification. This does not update the installed alpha-80 package.
The [original experimental duration](validation/sediment/jamieson2013_bend/TIMESCALE_NOTE.md)
also clarifies why startup screens cannot be treated as the final TR3 bed.

The latest [horizontal momentum diffusion milestone and figures](validation/sediment/jamieson2013_bend/reports/horizontal_viscosity_v1/RESULTS.md)
implements optional native viscosity, default zero. Small analytical/stock-Delft3D
controls and 133 targeted tests pass (one missing-fixture skip). The fresh 120 s
viscous bend now passes its original steadiness gates: final velocity variation
0.894%, versus 28.1% in the prior zero-viscosity run. Diffusion consumed 1.85% of
runtime on eight threads. Near-bank velocity profiles still differ; variable-depth
operator differences and a pre-existing rotated cut-cell hydraulic error remain
explicitly unqualified. This is not mobile-bed validation or a new plugin release.
The [previous hydraulic assessment](validation/sediment/jamieson2013_bend/reports/hydraulic_qualification_v1/RESULTS.md)
and [qualification checklist](validation/sediment/jamieson2013_bend/HORIZONTAL_VISCOSITY_NEXT.md)
retain the evidence and remaining gates. Next: audit near-bank hydraulics, then
repeat paired short bank/Exner controls from the stabilized state.
The [bank-correction milestone](validation/sediment/jamieson2013_bend/BANK_CORRECTION_MILESTONE.md)
includes stock-Delft3D microcases, conservative native bank fixes, an inward-critical
stage-outlet correction, and completed 10 s bend controls with figures. The native
development runtime is rebuilt; these changes are not installed in the alpha 80
plugin. The remaining bank hotspot and a pre-existing MPM outlet regression are
explicitly unqualified. No Delft3D source modifications are permitted.

The earlier [published Schuurman et al. (2013) braiding case](validation/sediment/schuurman2013/README.md)
had the following scope:
reconstruct Run 3 in stock Delft3D using the paper's geometry, grid, sediment
formula, roughness and MorFac, then show the bar/channel evolution. This
numerical experiment was selected by the user instead of direct laboratory
replication. It supersedes the bank/toe qualification workflow below; that
historical audit is not a prerequisite for running the published demonstration.
[Live/final results](validation/sediment/schuurman2013/reports/run3_25months_v1/RESULTS.md)
distinguish partial output from a completed simulation. Unreported inputs and
the newer locked runtime are explicitly documented; Delft3D source is unchanged.

The earlier [Delft3D-first incised floodplain trial](validation/sediment/synthetic_meander/reports/SUMMARY.md)
used a 6 m channel with a
very mild sinusoidal centerline, erodible banks and a straight-channel control.
Both reference runs reach 3,600 solver seconds at MorFac 10. Figures, animation
and QGIS rasters show bank erosion, widening and downstream pattern migration,
but growing meanders have not yet passed the observation/numerical gates.
BRAIDS has not been run for this case; Delft3D source remains unchanged.

The follow-up [stable-bank numerical audit](validation/sediment/synthetic_meander/reports/milestone_2/SUMMARY.md)
tests 2H:1V banks, finer meshes and smaller timesteps. It finds strong bank-region
mesh sensitivity and a timestep sequence that does not converge; cell-scale
bed variations are explicitly plotted rather than treated as physical bars.
The 0.125 m fixed-bed preparation is retained, but further mobile refinement
and long extensions were deferred pending the bank-process qualification in the
[next-milestone plan](validation/sediment/synthetic_meander/NEXT_MILESTONE.md).
This validation work does not change the installed plugin or Delft3D engine.

The subsequent [compact bank/toe process audit](validation/sediment/compact_bank/README.md)
separates fixed-bed hydraulics, dry-bank redistribution and wet-slope relaxation,
with timestep, mesh, sediment-budget and serial/MPI checks. It identifies
timestep-dependent apparent bank retreat and mesh-dependent relaxation strength.
The bank-retreat qualification has not passed; neither a longer river extension
nor a BRAIDS meandering comparison is claimed. All reference source remains
unchanged; supported input sensitivities are explicitly recorded.

The [completed 3,600-second exploratory expansion comparison](validation/sediment/synthetic_expansion/reports/exploratory_expansion_1m_3600_v1/SUMMARY.md)
provides matching 1 m BRAIDS/Delft3D maps and an animation at MorFac 10.
Both develop branching submerged deposits in the widening, but the wide
kilometre remains immobile and no exposed depositional bars appear at the
endpoint. This is not numerical qualification: outlet-local irregularities,
model differences and a late native timestep-retry burst remain explicit.
Delft3D source and both solver binaries were left unchanged.

The [matched expansion status](validation/sediment/synthetic_expansion/reports/controlled_expansion_milestone_v1/STATUS.md)
is **not qualified**: both models fail the 2-to-1 m spatial-convergence gate
and develop shallow-bank outlet scour. Pairwise agreement alone is not
convergence. Delft3D source and its locked executable remain unchanged.

The [short trapezoidal-bank control now passes its numerical gates](validation/sediment/synthetic_expansion/reports/bank_outlet_resolved_timestep_v1/RESULT.md),
with 1/0.5/0.25 m controls, sediment-flux budgets, an outlet extension and timestep
checks. At 0.25 m the model-pair signed-volume difference is 2.963% and common-bin
bed RMSE is 0.184 mm. Smaller reference timesteps resolve the original inlet
timestep failure, which is retained in the audit. The full expansion is still
not qualified. The [completed 0.5 m paired spatial screen](validation/sediment/synthetic_expansion/reports/full_expansion_spatial_halfmetre_v1/RESULT.md)
keeps sampled outlet indicators below 0.297 mm (BRAIDS) and 0.113 mm (Delft3D),
but 1-to-0.5 m pattern changes are 11.152% and 14.123%, above the unchanged
10% spatial limit. The [completed timestep controls](validation/sediment/synthetic_expansion/reports/full_expansion_timestep_halfmetre_v1/RESULT.md)
pass their declared aggregate/outlet/budget gates, but retain localized native
differences up to 32.5 mm; they do not override the spatial failures.
The [full four-rank reference comparison](validation/sediment/synthetic_expansion/reports/full_halfmetre_mobile_parallel_v1/RESULT.md)
passes strict serial equivalence and was 2.64 times faster under the observed
background load. No new solver/plugin release has
been installed for this validation work.
The [full audit and figures](validation/sediment/synthetic_expansion/reports/full_expansion_audit_20260926/RESULT.md)
retain the failures, parallel equivalence and the newly isolated near-dry
sediment export in the pre-coupling capacity/slope face flux.
The [additional cellwise audit](validation/sediment/synthetic_expansion/reports/bank_cellwise_limitations_v1/RESULT.md)
also records substantial localized inlet differences: the short outlet/aggregate
gate pass is not pointwise whole-case agreement.
The [full-case inlet-state audit](validation/sediment/synthetic_expansion/reports/full_inlet_hydraulic_states_v1/RESULT.md)
now verifies the same per-face discharge but different supercritical hydraulic
states at the free-stage inlets. These decks do not impose critical inlet depth.
Matching the initial surface and velocities reduces, but does not eliminate,
the early cellwise bed discrepancy; no diagnostic setting has been promoted.

Alpha 79 makes local-equilibrium MPM stage-outlet sediment export use the same
hydraulic boundary state as the water flux. This removes the artificial
centimetric outlet deposit while preserving physical reflection and
conservation. See the [consistent-export audit and figures](validation/sediment/synthetic_expansion/reports/stage_capacity_refinement_v1/AUDIT.md)
for matched-grid Delft3D controls, continuation/refinement errors and precise
scope. Full BRAIDS/Delft3D mobile-bed equivalence remains unproven.

The subsequent [controlled straight-reach milestone](validation/sediment/synthetic_expansion/reports/uniform_straight_pair_v1/AUDIT.md)
passes matched 1/0.5/0.25 m grid, bed-wave, hydraulic and conservation checks:
fine-grid bed-change RMSE is 0.284 mm and mean export differs by 0.30%.
This applies to explicitly matched per-face inlet discharge, not the original
depth-weighted total-Q boundary or the expansion. Delft3D source is unchanged;
only supported reference inputs were adjusted. Four figures and rejected
controls are retained. This historical comparison used alpha79; alpha80 is
the installed Windows release.

Alpha 78 removes an unstable midpoint-only sediment-flux buffer beside
subcritical stage outlets. The coupled MPM water/Exner flux now remains active
there; prescribed-flux, critical/weir and supercritical-release controls are
retained. The [outlet-wave and morphology-control audit](validation/sediment/synthetic_expansion/reports/outlet_buffer_v2/AUDIT.md)
includes independent reflection/convergence checks, paired 1 m/0.5 m native
runs, timestep/MorFac controls, and figures. The outlet-wave fix is verified;
full BRAIDS/Delft3D mobile-bed equivalence is still **not** established.

The subsequent [downstream-extension audit](validation/sediment/synthetic_expansion/reports/downstream_extension_v1/AUDIT.md)
shows that BRAIDS' aggradation follows the relocated outlet. The original
reach is nearly unchanged between 200 m and 300 m domains. A slope-only
boundary correction fails to remove the deposit and is not shipped. The
subsequent alpha79 boundary-state correction is documented above.

Alpha 77 corrects the sloping subcritical stage-outlet reconstruction.
The [outlet correction and 1 m refinement audit](validation/sediment/synthetic_expansion/reports/outlet_corrected_v1/AUDIT.md)
includes analytical hydraulic checks, matched-discharge Delft3D comparisons,
compiled transport-kernel checks and figures. Mobile-bed equivalence remains
an open validation milestone.

Alpha 76 fixes near-critical cell-alternating bed changes in local-equilibrium
MPM transport with a coupled water/Exner face flux. See the
[synthetic expansion audit](validation/sediment/synthetic_expansion/reports/coupled_fix_v1/AUDIT.md)
for the before/after Delft3D comparison, accuracy checks and remaining scope limits.

The subsequent [straight-reach isolation audit](validation/sediment/synthetic_expansion/reports/straight_reach_v4/AUDIT.md)
includes direct internal sediment-flux measurements and comparison figures.
It identifies remaining outlet-hydraulic and transport-state discrepancies;
the expansion morphology is not yet validated as equivalent to Delft3D.

The latest release adds a restart-identical QGIS benchmark preparation path,
correct dry-start mobile-bed initialization, live-section-aware inlet
diagnostics, and compact cell-average terrain input. The Johannot 0.025 m
project defaults to its measured 16-core optimum; the larger fully wet 64,496-
cell state remains 4.31x faster end-to-end on 32 cores, with byte-identical
outputs. A predictive fixed-grid CFL controller removes the former grow,
reject, and half-step cycle; on an active five-thread H2 state it cut accepted
steps by 22.6% and wall time by 22.1% (1.28x speedup), with state and boundary-
ledger relative differences below `6e-8`. Hoisting invariant grain-size
logarithms out of the 60-iteration, seven-fraction D90 solve adds a measured
1.58x hot-state speedup with byte-identical state and ledger output. A repeated
hot-state temporal screen then qualified CFL 0.9 for the Johannot benchmark:
it is 1.98x faster than CFL 0.45, repeats byte-identically, and differs from the
CFL 0.225 reference by only 0.0033 mm mean water depth and
`1.84e-5 kg/m²` mean active-bed mass. Large
fixed-grid result states now use a compact versioned binary
stream that loads transparently through the same QGIS reader; the exact
16,124-cell qualification reduced authoritative storage by 69.3% and total
output time by about 17x without changing a decoded value. Adaptive meshes and
small compatibility cases retain ASCII output. The retained numerical evidence
is linked below. The final integrated audit byte-freezes the experimental
specification, source-data manifest, material calibration envelope, and DEM
registration/exclusion decision so a changed benchmark cannot silently inherit
the qualification. Its LSPIV velocity evidence is processed with a measured
eight-process default and byte-identical products; because observation timing
and surface-to-depth conversion remain unresolved, the final run reports every
discharge-matched state as screening evidence and never selects a best fit.

The active plan and numerical evidence are recorded in the
[GPU implementation-first plan](validation/gpu_fwave/GPU_IMPLEMENTATION_FIRST_PLAN.md),
[GPU feature roadmap](validation/gpu_fwave/GPU_FEATURE_COMPLETION_ROADMAP.md), and
[remaining-work inventory](docs/BRAIDS_REMAINING_WORK.md). The current
sediment implementation audit and the validation-first route to a braided-river
model are in
[SEDIMENT_IMPLEMENTATION_AUDIT_2026-09-22.md](docs/SEDIMENT_IMPLEMENTATION_AUDIT_2026-09-22.md)
and
[BRAIDED_RIVER_MORPHODYNAMICS_PLAN.md](docs/BRAIDED_RIVER_MORPHODYNAMICS_PLAN.md).

## Development setup

Requirements:

- Python 3.10 or newer
- NumPy 2.x and the packages in `requirements-dev.txt`
- GDAL Python bindings matching the installed GDAL library
- GNU Fortran with OpenMP support
- QGIS 3.40 or newer for plugin tests and use

Create a development environment and install the core package:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
```

Build the native solver and run the regression suite with eight threads:

```sh
.venv/bin/python tools/build_cutcell.py
OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE .venv/bin/python -m pytest -q tests
```

Build the installable QGIS package:

```sh
.venv/bin/python tools/build_qgis_alpha.py
```

On 64-bit Windows, use `.venv\Scripts\python.exe tools\build_qgis_windows.py`.
The Windows builder rejects stale native binaries, unexpected runtime DLLs,
failed packaged qualifications, and any Delft3D source or raw-run path.

The build output and all generated validation runs are ignored by Git. Large
upstream comparison repositories are also excluded. Restore their pinned
working copies when needed:

```sh
.venv/bin/python tools/fetch_reference_sources.py
```

See the pinned [reference-source fetcher](tools/fetch_reference_sources.py) and
[third-party notices](braids_qgis/THIRD_PARTY_NOTICES.md) for their revisions,
sources and purpose. The Clawpack 5.14.0 sources required by the native BRAIDS build remain
under `solver/clawpack-v5.14.0`.

## Core section example

The headless package does not import QGIS or GDAL for analytic section work:

```python
from braids_core.boundaries import Section

section = Section(
    bc_id="river_inlet_1",
    stations_m=[-6.5, -2.0, 2.0, 6.5],
    bed_elevations_m=[103.0, 100.0, 100.0, 103.0],
    datum_m=100.0,
    inward_normal=(1.0, 0.0),
)
geometry = section.geometry(stage_m=101.0)
normal = section.normal_stage(discharge_m3_s=1.0, n=0.03, energy_slope=0.001)
critical = section.critical_stage(discharge_m3_s=1.0)
print(geometry.area_m2, normal.stage_m, critical.stage_m)
```

For projected metre DEMs and GeoPackage polylines:

```python
from braids_core.boundaries.gis import load_boundary_sections

sections = load_boundary_sections("terrain.tif", "boundaries.gpkg", datum_m=100.0)
target = sections["river_inlet_1"].normal_stage(1.0, 0.03, 0.001)
```

The [section API](docs/SECTION_SERVICE.md) documents the geometry and unit
conventions. Inlet construction normalizes discharge on the actual discrete
wet faces. For subcritical inflow, the compatible water elevation follows the
outgoing characteristic; supplied normal, critical or water-level values act
as reference targets.

## Repository contents

- `braids_core/`: GIS-independent geometry, boundary and sediment services
- `braids_qgis/`: QGIS user interface, project preparation, execution and results
- `solver/`: Fortran/OpenMP solver and the retained Clawpack build dependency
- `tests/`: unit, native, restart and parallel regression tests
- `validation/`: reproducible study scripts, compact specifications and reports
- `resources/`: BRAIDS logo and compact GIS fixtures
- `tools/`: native/package builders, upstream fetcher and repository cleanup

The source checkout intentionally excludes old run folders, compiled objects,
temporary profiles and complete copies of upstream repositories. Running a
validation script recreates its numerical products locally.
