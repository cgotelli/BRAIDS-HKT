# Native cut-cell executable

The alpha 81 default includes the optimized execution paths below and the
tested curvature regularization for enabled prognostic secondary flow.
`braids_curvature_regularization_module.f90` applies length 2h, relative
tolerance 1e-4 and absolute tolerance 1e-10. It dispatches to MIC below 65,536
cells and multilevel PCG above that threshold. This regularizes secondary-flow
production, not the bed elevation. Small-mesh serial MIC is retained because
the tested parallel alternatives were slower; the large branch uses measured
OpenMP team sizes. Cached geometry is invalidated on mesh identity changes;
depth-dependent coefficients are rebuilt each call. Saved workspace is not
reentrant. Failure to converge is reported, not silently ignored.

See [release scope and remaining qualification](../../docs/WINDOWS_ALPHA_81_RELEASE.md).
No GeoClaw, reconstruction, integration, CFL or MorFac limit change accompanies
this promotion. Secondary flow remains opt-in.

`xbraids_cutcell` is the native stepping path used by the BRAIDS QGIS workflow for clipped polygon control volumes. It supports fixed meshes and four-level coupled AMR with native regridding, time subcycling, conservative interface synchronization and adaptive restart. It is separate from the legacy rectangular WAVE/AMR executable. It compiles the unchanged WAVE GeoClaw Riemann sources and the sediment kernels described below. `geoclaw_parameters.f90` supplies Cartesian parameter bindings for those sources (no atmosphere or spherical metric); it does not replace their numerical routines.

See the [current AMR qualification, snapshots and limits](../../validation/amr/AMR_VERIFICATION.md). The format and validation history below includes earlier fixed-mesh milestones; their historical deferrals do not describe the current alpha.

The development runtime now uses [continuous bed-coupled bank evolution](../../docs/BANK_BED_COUPLING.md).
The experimental v41 persistent-front overlay is retired; prepare a fresh v40
distributed-bank run instead of restarting a v41 checkpoint.

From the repository root, build the runtime and run the short AMR criteria checks:

```sh
.venv/bin/python tools/build_cutcell.py
.venv/bin/python validation/amr/run_criteria.py --threads 8
```

The executable and compiler/source manifest are in `solver/cut_cells/build/`. GNU Fortran and OpenMP are required. Individual criterion runs have a 60 s cap. Run an exported case from its working directory:

The [hydraulic workflow optimization](../../validation/sediment/lfm_bend_2d/reports/workflow_adopted_20260930_v1/REPORT.md)
and [reusable-workspace follow-up](../../validation/sediment/lfm_bend_2d/reports/face_workflow_20260930_v3/REPORT.md)
keep GeoClaw, WENO/RK3 and flux formulas unchanged. Euler now retains 27 work
arrays between calls, reallocating whenever its cell/face/field/boundary dimensions
change. All original stage initialization remains. Euler is not reentrant;
concurrent patch drivers would need independent workspaces. AMR correctness,
including actual shape changes, is tested. The newer
[adaptive performance milestone](../../validation/sediment/lfm_bend_2d/reports/amr_workflow_20260930_v1/REPORT.md)
also measures real regridding, level subcycling, thread scaling and memory.

For fixed grids with at least 4,096 cells and scheme 6, the driver can fuse the
same gradient/wall/wet-dry cell kernels. This includes bedlaw zero or MPM, multiple
fractions, passive suspended transport, viscosity and secondary flow, but excludes
AMR, bank/avalanche processes, bedform roughness, bedload adaptation and nonzero
exchange mode. Other cases retain the original preparation. Separately, only
eligible single-fraction hydraulics with no active auxiliary or sediment transport
bypass zero sediment-flux sweeps; full boundary series and resumed inventories
are checked. Medium eligible hydraulic friction loops also use OpenMP when at
most eight threads are configured; the larger-work threshold is unchanged.

`BRAIDS_PROFILE=1` reports `optimized hydraulic workflow calls`: it counts fused
trace-preparation calls, not permission to skip active sediment work. Zero means
original trace preparation was used, even though Euler workspace is still reused.
These are execution optimizations, not new numerical-method or deck options.

Eligible adaptive hydraulic runs now use `prepare_adaptive_traces_fused`.
Two membership bits cached with each mesh epoch preserve the original gradient
face-cell subset and wider wet/dry evaluation subset; wall preparation still
covers all cells. All default initialization remains, including outside those
subsets. Eligibility requires scheme 6, at least 4,096 active leaves, one
fraction/five states, no bedload/exchange, secondary flow, bank/avalanche process,
bedform roughness or bedload adaptation. Passive suspended transport and viscosity
are checked. Other adaptive configurations retain original preparation. Boundary
refinement constraints and the synchronized wet/dry-front fallback are unchanged.

For the qualified subcycled hydraulic configuration, every nonempty
participating-face loop uses OpenMP. This was measured through 32 threads; the
initial eight-thread cap was removed after an additional 9% gain at 32 threads.
Fixed grids, synchronized stepping and excluded physics retain the prior
work-size threshold. The configured maximum
thread count is not overridden. The profile's combined-preparation call counter
can therefore also be nonzero for AMR. The fixed-grid numerical path is unchanged;
the measured fixed-grid runtime variation is retained in the report.

```sh
OMP_NUM_THREADS=8 /absolute/path/to/xbraids_cutcell case.data
OMP_NUM_THREADS=8 /absolute/path/to/xbraids_cutcell case.data /absolute/path/to/checkpoint.bin
```

Use `braids_core.boundaries.cut_cell_io.write_cut_case` with a `CutCellMesh`, initial cell-average fields, bedrock, explicit `CutBoundary` settings and the build's `source_sha256`. A legacy single-fraction case has five fields; a version-12-or-newer case with `N` fractions has `3 + 2N` fields. This produces a separate versioned deck and metadata with common hydraulic/sediment IDs. It does not convert the legacy `braids_boundary.data` format. End time may increase across restart; the other resolved inputs and native-source fingerprint must match. Checkpoints also retain the next timestep hint and accumulated boundary ledgers. Restart input validation precedes output replacement.

Optional horizontal viscosity has two explicit formulations: the default
depth-conservative momentum operator and a plain `velocity_laplacian` compatible
with Delft3D's smooth-wall continuum form. Their equations, invariants, deck
versions and cut-cell limitations are documented in
[HORIZONTAL_MOMENTUM_DIFFUSION.md](../../docs/HORIZONTAL_MOMENTUM_DIFFUSION.md).
The velocity-Laplacian mode leaves fully wet conductances unchanged but closes
a perched shared face when its centre-stage water column is at or below half
the dry tolerance, with linear activation up to the dry tolerance. The current
bed includes every active/substrate layer. This symmetric, stateless collocated
mask is restart/AMR safe and is only an analogue of Delft3D's staggered
hysteretic wet-face masks, not a source port or exact clone.

New velocity-Laplacian decks use the v35
`boundary_quadratic_lsq_face_gradient_v35` discretization. Orthogonal,
symmetric interior faces retain a two-point fast path. Skew, unequal-distance,
and no-flux-adjacent faces use a rotation-orthonormal two-ring quadratic fit
with boundary-fragment-weighted homogeneous-Neumann rows, evaluated at the
physical face. One shared flux conserves area-integrated velocity; physical
momentum is conserved at constant depth only. A global dissipativity envelope,
conservative coefficient bound, and exact stage-energy retry protect the
multipoint update. V35 does not claim a componentwise maximum principle.
Archived v34 mode-1 decks remain on their original two-point arithmetic.

Hydraulic wetting now applies a separate shared-face aperture test whenever at
least one adjacent cell centre is dry. The face stage is interpolated with the
owner/neighbor centroid distances projected on the face normal (an arithmetic
mean on an equal Cartesian pair), and is compared with the higher current bed.
A sub-threshold face blocks water and bedload transport but retains the
side-specific hydrostatic wall force and gravity-wave CFL speed, so a lake at
rest remains exact. An incoming reflected Riemann wave can still open the face
when its resolved run-up overtops the higher bed; the threshold does not turn a
physical bore into a wall. The decision is symmetric and stateless, hence
trial-retry, OpenMP, restart and AMR safe. Mirrored unequal-centroid cut-cell,
stationary-lake, dynamic-overtopping and conservation regressions are in
`tests/test_internal_face_wetting.py` and `tests/test_bedload_dry_faces.py`.

The legacy fields are `h, hu, hv, S, Mbed`; `S` and `Mbed` are kg per square metre of retained cell area. Version 12 stores `h, hu, hv`, then all suspended inventories `S_i`, then all bed inventories `Mbed_i`. Bed elevation is bedrock plus the sum of `Mbed_i / bulk_i`. Geometry and sources use retained cell areas. Every interior face has one shared water flux and independent suspended/bed fluxes for every fraction; momentum retains GeoClaw's two-sided bathymetric source contribution. The accepted ledger contains signed inward water volume, then suspended mass for every fraction, then bedload mass for every fraction, per logical boundary ID. `frames.txt` writes time, step number, all cell fields and cumulative ledgers; `checkpoint.bin` is a local compiler-native checkpoint, not a portable exchange format.

The transport operator uses SSPRK3. WENO-Z uses verified uniform five-cell rectangular stencils, including clipped strips uniform along the reconstruction direction; genuinely irregular stencils use bounded linear least-squares reconstruction in physical coordinates. Bed is reconstructed alongside stage/velocity, with the within-cell pressure/source correction. Cell momentum residuals subtract constant cell pressure before metric multiplication; its closed-contour integral is zero, and this avoids roundoff-seeded transverse motion. Aperture/area is formed before scaling shared fluxes. Suspended FCT bounds use updated water volumes. Native donor limiting enforces available water and bed inventories; failed stages are retried without accepting fluxes or changing prescribed boundary Q. Native stepping never deletes or merges cells. Optional preprocessing can join tiny boundary fragments to face neighbours, preserving physical boundary geometry and IDs; the native CFL still uses their actual areas. Manning and exchange are split sources; full third-order multiphysics accuracy is not claimed.

Hydraulic least-squares neighbour equations are weighted by aperture length
over squared centroid distance. This preserves affine exactness while making
the reconstruction invariant to splitting one physical aperture into multiple
face records that connect the same two cells.

The hydraulic weights/normal matrices are cached with their original arithmetic
and face accumulation order, and rebuilt with every AMR mesh epoch. Gradient
work uses OpenMP from 4,096 cells even for hydraulic-only five-state decks.
Wet/dry predicates reconstruct only surface and terrain, with the same formulas
as the complete face trace. These implementation changes preserve paired states
and boundary ledgers bit-for-bit; see the
[hydraulics-only performance report](../../validation/sediment/lfm_bend_2d/reports/hydraulics_only_20260930_v1/REPORT.md).

The accepted finite-volume CFL limit is configurable up to 0.9 and is included in restart identity. The established default remains 0.45; decks without the optional trailing CFL value retain 0.45 for backward compatibility. Values above 0.45 can reduce runtime but require case-specific timestep sensitivity checks.

The current shoreline reconstruction extends a wet cell's surface into higher
dry stencil points, preserving a stationary lake beside steep land. Fully wet
rectangular cells retain reconstruction even when the bed excursion exceeds
their depth. Partly dry reconstructions and unsupported irregular geometry
retain the constant-state guard. Bed-source quadrature uses the actual cell and
face water columns, including sampled inlets; a squared fixed-cell-surface
reference produced an erroneous curvature force. Independent stationary-lake,
curved-bed acceleration and steep simple-wave checks are documented in the
[source correction report](../../validation/sediment/reservoir_polygon/geometry_resolution/reports/HYDRAULIC_SOURCE_CORRECTIONS.md).
These fixes preserve native OpenMP execution and scalar FCT; they do not by
themselves qualify reservoir spatial resolution.

Sampled inlets determine fully incoming owner characteristics from the actual
cell depth before extending its water column hydrostatically to a lower edge.
For a prescribed supercritical exterior jet, the existing augmented GeoClaw
Riemann fan can also establish that all information enters over the bed step.
These checks prevent false outgoing-characteristic constraints on thin wetting
streams. Q, section level, conservation and speed safeguards are unchanged.
The [inlet and small-cell report](../../validation/sediment/reservoir_polygon/geometry_resolution/reports/INLET_REGIME_AND_SMALL_CELLS.md)
documents independent references, the optional Mesh setting, and its limits.

Independent water and sediment update loops use OpenMP with fixed within-cell
face-summation order. Parallel stage failures reject the complete trial after
the loop. Bounded startup/evolving-bed checks preserve states, accepted ledgers
and steps exactly; the fine startup benchmark takes 26.8% less time. This does
not imply general platform scaling or reservoir accuracy.

Set `BRAIDS_PROGRESS_STEPS` to a positive integer to print time, accepted step
count, timestep and rejected-trial count at that cadence. It is disabled by
default and changes neither the numerical state nor the checkpoint format.

`write_cut_case(..., scheme=6)` opts into reflected-wall WENO; the default remains `scheme=5` (and `scheme=1` remains first order). Mode 6 admits square-grid half triangles only when the actual areas, centroids, shared Cartesian legs and default 45° wall match the reflected-square geometry. Ghost squares must reflect to actual full squares across a finite, uninterrupted wall. Both axis stencils must be complete and wet for a triangle; unsupported geometry and dry stencils retain the existing reconstruction. This is not a general arbitrary-angle high-order reconstruction.

The triangle trace reconstructs tangential velocity in the wall frame. An affine normal-discharge polynomial retains the actual conserved cell mean and vanishes at the wall; its face value is divided by reconstructed depth. Cartesian velocity clipping after that repair would destroy the mean and is not applied. Existing shared flux, donor bounds, FCT and timestep rejection still apply. Conserved normal-discharge consistency and tangential primitive consistency are separate claims; a conserved tangential-momentum reconstruction has not been established. Wall pressure and interior face integration retain their previous treatment.

Geometry and WENO stencil indices are cached once; trace preparation and timesteps remain Fortran/OpenMP. The GeoClaw wrapper computes only the interface it consumes, using the unchanged Riemann kernel. These arithmetic-preserving optimizations reduce a measured 15 s worker from 5.45 to 4.40 s and pass [156 exact-output checks](../../validation/hydraulics/cut_cells/performance/native_regression/NATIVE_CACHE_REGRESSION.md). Mode 6 also passes [61 compact native checks](../../validation/sediment/zhu2018_oblique/wall_weno/native_regression/SCHEME6_NATIVE_REGRESSION.md), covering actual 45° wet/dry cells, nonzero normal discharge, Van Rijn exchange/bedload, thread/restart equality and unchanged scheme 5 output. Scheme and native-source changes enter restart identity; old checkpoints require the matching build and mode.

Implemented inlet controls combine one total discharge per logical ID with prescribed level, prescribed depth above the section datum, section normal depth or section critical depth. These user-facing section controls retain the resolved horizontal stage and a uniform normal velocity whose face integral is the requested total discharge as boundary cells wet. Legacy mode 1 remains the characteristic discharge trace. Outlet controls are prescribed stage, computed critical depth, free-weir rating, or supercritical release. Every unassigned face is a wall. Multiple IDs are supported; one ID may own multiple selected faces. Shared inlet pieces must be connected and collinear. Reversal and wetting from a dry stage outlet are rejected. The compact native multi-ID fixtures exercise two inlets and two outlets, including shared collinear inlet edges, a noncollinear outlet group and default walls.

The free-stage total-discharge compatibility mode uses the stock FLOW
allocation `B*h_section^(3/2)*max(h_velocity,0.01 m)^(1/6)/n`, normalized over
the currently wet section and divided by the physical face depth to obtain
velocity. The 0.01 m floor belongs only to the Manning-Chezy factor; it does not
replace the physical depth in section geometry or discharge conversion.

Format version 3 adds a total bedload supply per logical inlet ID and explicit bed-slope correction. Total supply is distributed over currently incoming fragments in proportion to accepted water flux; its accepted ledger therefore integrates the requested kg/s once even when an inlet contains several polygon-cut faces. `ISlope=2` applies Bagnold longitudinal scaling and Ikeda/Van Rijn transverse rotation with configurable `AlfaBs`, `AlfaBn` and friction angle. `ISlope=3` is the Van Bendegom/Koch--Flokstra direction law used by the Struiksma T2 case and is admitted only with Engelund--Hansen (`bedlaw=8`): the same three deck fields then mean `AlfaBs`, `AShld`, and `BShld`. That first implementation represents the published `CShld=DShld=0` specialization only; nonzero values are not silently approximated. Bed gradients use Delft3D's positive-downhill convention. `bedtau_cell` remains the critical-stress ratio for existing laws and mode 2, but carries absolute Shields stress for law 8 and mode 3. `CutSediment.minimum_bedload_depth_m` is independent of `minimum_exchange_depth_m`; version 1/2 decks preserve their former shared-threshold behavior. [Formula verification and resource snapshots](../../validation/sediment/reservoir_polygon/bed_slope_comparison/BED_SLOPE_RESERVOIR_COMPARISON.md) document the earlier bounded current-only, single-fraction scope.

Noncohesive bedload capacity uses a C1 wetting transition between the hydraulic dry tolerance and `minimum_bedload_depth_m`. The selected formula is evaluated at its shallowest valid depth while retaining resolved velocity, then multiplied by `xi^2*(3-2*xi)`. Capacity is zero at the dry tolerance and the activation factor is one above the configured minimum depth. This removes the grid-dependent Exner jump produced by a hard capacity switch without smoothing the bed or capping prescribed boundary input. [Compact exact-edge qualification](../../validation/sediment/near_inlet_compact/production_correction/REPORT.md) records conservation, inlet-spike, hydraulic, fully wet, OpenMP and restart checks.

Van Rijn bedload additionally evaluates its finite-depth constitutive terms at
`H=max(h_capacity, 0.01 m)`, rescaling momentum only inside that temporary
state so resolved velocity is preserved. Both fixed and Manning-derived
roughness use `H`. Bed shear retains the physical/activation velocity
observation height `h_capacity/e`, matching FLOW's ordering in which `dwnvel`
precedes the `erosed` depth floor. The separate C1 activation factor still
uses physical depth. The 1 cm value is an internal validity/work depth, not a
wet/dry threshold, calibration parameter, or hydraulic friction setting. The
[operator ablation](../../validation/sediment/jamieson2013_bend/reports/transport_activation_ablation_v3/RESULTS.md)
separates this correction from hard cell/face gates and bed-slope effects.

Format version 4 adds `CutSedimentBoundarySeries` per inlet ID. Suspended concentration and total bedload kg/s are piecewise linear; values outside the declared range explicitly hold the nearest endpoint. Every SSPRK Euler evaluation uses its physical stage time, and the outer timestep stops at every forcing knot. Constant Q remains one total discharge per ID. Reversal policy is serialized and currently accepts only `reject`: a locally outward inlet or inward outlet causes the trial to fail and, if it persists, the run stops without accepting a step. [The 40-check native proof](../../validation/sediment/polygon_boundary_timeseries/reports/POLYGON_SEDIMENT_TIME_SERIES.md) verifies analytic schedules for two inlet IDs, one-edge and multi-edge outlet IDs, conservation, morphology, thread equality and restart.

Format version 5 appends `manning_n` to every cell row and retains the version-4 boundary-series layout. QGIS supplies a domain default and optional editable polygon zones. A polygon assigns its complete value to every retained cell it intersects with positive area; cells are not split or area weighted. Snapped zones that share nodes/edges use the highest value in a jointly touched cell; interior overlaps and disconnected co-occupancy are rejected. The native friction loop uses the per-cell array inside its existing OpenMP cell loop. [The focused proof](../../validation/hydraulics/spatial_manning/reports/SPATIAL_MANNING.md) confirms the contact rule, bitwise equality with legacy scalar roughness for a constant field, the independent friction update, one/four-thread equality and exact restart.

Format version 6 adds the full inlet/outlet mode set and section-resolved controls. Native mode 9 identifies the four user-facing inlet controls and imposes their resolved stage and total Q without a wet-state branch change. Format version 7 adds `CutHydraulicBoundarySeries` per supported ID. Inlet level/depth schedules store time, total Q and control; normal/critical schedules store time and total Q and resolve the target level from the sampled section at each knot; stage outlets store time and level. The deck contains the resulting total-Q/absolute-level knots. Native SSPRK stages use piecewise-linear interpolation, hold endpoint values outside the declared range and stop accepted timesteps at every hydraulic or sediment knot. [The short proof](../../validation/hydraulics/hydrographs/reports/HYDROGRAPH_VERIFICATION.md) closes analytic water/sediment integrals and storage, checks level use in the native trace, and preserves bitwise OpenMP/restart results.

Format version 8 adds an independent bed-load boundary mode per logical ID. `equilibrium_capacity` evaluates the selected native transport formula on each incoming face's resolved hydraulic trace at every SSPRK stage, applies slope correction when enabled, and records the accepted inward load in the existing conservative ledger. The prescribed total, per-width and time-series contracts remain unchanged. [The paired native control](../../validation/sediment/delft3d_intercomparison/bedload_basin/reports/BEDLOAD_BASIN_INTERCOMPARISON.md) finds a 0.474% mean inlet-load difference relative to the initial reference and a 0.0074 mm maximum final bed-profile difference; one/four-thread BRAIDS frames are byte-identical.

The [report and actual snapshots](../../validation/hydraulics/cut_cells/stepping/reports/NATIVE_CUT_CELL_STEPPING.md) cover variable-bed rest, wet/dry filling, constant settling, the Van Rijn suspended branch, a translating pulse, threads and restart. The [corrected Zhu late-window check](../../validation/sediment/zhu2018_cut_cells/pressure_centered/reports/CUT_CELL_ZHU_CORRECTION.md) passes 37/37 frozen gates and 46 compact regression/thread/restart checks. Both unequal strips retain zero transverse bed spread. Fine-grid RMSE/max error are 2.185/23.081 mm; this 69–70 h initialized window does not qualify a full paper event or genuinely oblique sharp-bed transport. Those scope limits, actual-resource terrain testing, AMR/MPI, legacy patch distribution, production performance and QGIS integration remain open.

## MPM and configurable sediment properties

Format version 10 adds `CutSediment.reference_height_factor` (positive, default 1) and `roughness_mode` (`fixed`, the default, or `manning`). In Manning mode the native suspended source derives current z0 from physical local depth, while Van Rijn bed capacity derives z0 from its 1 cm calculation work depth through `H / (e * (exp(0.41*H^(1/6)/(sqrt(g)*n))-1))`; the shear observation height remains physical. Every selected Manning coefficient must be positive. The source factor is passed to the existing reference-height formula. These choices are exposed through the input API, native deck and QGIS Sediment tab.

The v10 deck retains the v9 layout and appends `reference_height_factor roughness_mode_code` after boundary data, with roughness codes 0/1 for fixed/Manning. Both settings enter the resolved restart signature. Legacy defaults retain the old deck selection and numerical behavior. [Matched settings, fixed-bed profiles and native checks](../../validation/sediment/delft3d_intercomparison/reservoir_scale/matched_hydraulics/reports/MATCHED_HYDRAULICS.md) verify active-source parameter use, conservation, one/four-thread equality, exact restart, and byte-identical historical default fields. This uses the same physical roughness convention as FLOW, whose staggered wet-face averaging remains a distinct spatial discretization.

`write_cut_case(..., bedload=2, sediment=CutSediment(porosity=.64, d50_m=.0001, d90_m=.00015), mpm_cf=.01, mpm_critical_shields=0.)` selects the existing native MPM kernel for the Zhu case. `CutSediment` also exposes grain/water densities, d90, z0, minimum exchange depth and minimum bedload depth; its default material preserves the earlier Van Rijn examples. These single-fraction material and formula settings are exposed in the QGIS Sediment tab and recorded in each run's `resolved_inputs.json`.

Bedload modes are 0 (off), 1 (Van Rijn 1993), 2 (MPM), 3 (Ashida--Michiue), 4 (Soulsby), 5 (Wilcock--Crowe), 6 (modified Wilcock--Crowe), 7 (Gaeuman et al.), and 8 (Engelund--Hansen current-only total load). MPM decks use format version 2, adding `mpm_cf mpm_critical_shields` immediately after the scheme/law/exchange row. Old version 1 decks retain their layout for modes 0/1 and may carry the single-material Engelund--Hansen closure; mode 2 requires explicit version 2 parameters and mode-3 slope correction requires version 3. Both versions carry the configured solid bulk density `rho_s * (1 - porosity)` in the physical row. Volume-transport closures are converted once to mass flux using `rho_s`. All material and transport parameters enter restart identity; a change requires a new run. Checkpoints built before this extension are not interchangeable with current-source checkpoints.

The focused MPM pilot reproduces complete fields and ledgers byte for byte with one/four threads. Existing wet filling, Van Rijn exchange/bedload and variable-bed rest regressions also pass. The corrected Zhu window passes its transverse-invariance gate. Residual front error remains, and this axis-aligned strip result is not sufficient to qualify arbitrary cut-cell morphology. The subnormal FCT allowance reserve limits shared antidiffusive fluxes instead of clipping states; see the correction report for the retained failure and passing Van Rijn regression.

## Bounded oblique morphology check

The [45° steep-bed check](../../validation/sediment/zhu2018_oblique/reports/OBLIQUE_BED_VERIFICATION.md) passes 35 frozen gates using the unchanged corrected executable. It advances a Zhu-derived Bernoulli initial profile for 300 s with actual polygon-average references. The fine run uses 1,480 cells (160 triangular half cells), takes 43.35 s and has 7.48% change-normalized L1 error overall / 3.50% in partial cells. Refinement reduces bed L1 by 45.03%; maximum change error remains 15.01%. This is a limited continuous-profile verification, not full-event, arbitrary-angle, tiny-cell or complete Delft3D qualification. The native multi-ID oblique water/suspended/bedload check now passes; the later 5 m diagnostic below leaves the accuracy gate open before the actual reservoir polygon. The geometry exporter now uses identical indexed grid planes for neighboring cells with nonbinary spacing.

## Current multi-ID result and local-error gate

The [aligned 5/2.5 m initial-operator check](../../validation/sediment/zhu2018_oblique/orientation/spatial_resolution/results/reports/SPATIAL_RESOLUTION.md) finds that refinement alone does not improve the raw error at these spacings: maximum rate error grows 84.86%, with opposite-signed errors beside the shared face at 577.5 m. Combining fine rates over 5 m cells improves the apparent maximum through cancellation. Two probes take 0.596 s; native/Euler consistency and conservation checks pass, with no production edits. The evolved-profile reassessment below now sets the next development step; this operator result does not change the separate 300 s qualification status.

The [matched orientation diagnostic](../../validation/sediment/zhu2018_oblique/orientation/results/reports/ORIENTATION_COMPARISON.md) uses the unchanged executable in two genuinely 2D channels, with identical 5 m spacing and 1,500 steps over 15 s. Aligned/45° maximum bed-change errors are 8.14%/8.93%, while L1 is 4.32%/3.51%. Both conserve water/solids and complete in 19.75 s combined. Thus diagonal flow is not the entire explanation; rotation still increases transverse velocity. Actual polygon averaging and shared continuous-reference normalization are both reported. The subsequent spatial-resolution check and evolved-profile reassessment are recorded here; the 300 s accuracy gate remains open.

[Multiple oblique boundaries](../../validation/hydraulics/cut_cells/multi_id/reports/MULTI_ID_OBLIQUE_BOUNDARIES.md) pass the original 73 checks and 35 later coverage additions. Boundary traces now use an ascending per-ID face index built at load; this preserves hydraulic arithmetic and avoids repeated all-face filtering. [132 native comparison checks](../../validation/hydraulics/cut_cells/multi_id/boundary_indexed/BOUNDARY_INDEXING.md) pass, including bit-identical pre/post states, accepted ledgers, steps, threads/restart, and a closed `nb=0` case. This index is native and immutable during OpenMP evolution; checkpoint source identity changes as for any source revision.

[The earlier 10/5 m diagnosis](../../validation/sediment/zhu2018_oblique/refinement/reports/REFINEMENT_DIAGNOSIS.md) showed that refinement alone did not fix the local error. The subsequent [scheme 6 correction and snapshots](../../validation/sediment/zhu2018_oblique/wall_weno/qualification/reports/WALL_WENO_VERIFICATION.md) complete 300 s at 5 m in 93.08 s with four threads. At matched 150 s, L1/max bed-change errors improve from 5.09%/16.12% to 3.98%/11.61%. At 300 s, integrated error passes 5% at 4.44%, but maximum error fails at 11.69% (1.880 mm): 27/28 checks pass. The independent operator matches the native implementation to 1.04e-15 m/s. Interior reconstruction/face integration still needs correction before the actual resource polygon. The 5% limits and conservation safeguards remain unchanged; this is a short initialized Zhu-derived window, not a full HydroSedFoam or Delft3D qualification.

The [evolved-profile reassessment](../../validation/sediment/profile_reassessment/reports/PROFILE_REASSESSMENT.md) preserves the original aligned qualification: 0.144% peak loss and 0.168% variance error over 70 h. All saved final profiles meet the original limits, but even a bed held unchanged passes them over the short oblique 300 s window. Its actual maximum bed error is 1.880 mm, and the 11.69% maximum-change error still fails the newer 5% target. Next is actual resource-polygon input preparation, terrain/boundary preview, output-budget planning and small-cell timestep estimation with the existing native method. The accuracy prerequisite for enabling that sediment event remains open; no solver change or native rerun was made for this review.

## Actual resource-polygon startup

[Prepared inputs and actual snapshots](../../validation/sediment/reservoir_polygon/ready/PREPARATION.md) use the actual unrotated polygon, selected edge interval, default walls and no outlet. A hydrostatic source correction formerly created momentum on completely dry slopes during intermediate RK stages. It now uses the same wet/dry decision as the Riemann traces: dry cells have no bed-pressure column. A dry-terrain Euler check changes the spurious momentum from 0.457674 m²/s to exactly zero; [61 compact native checks](../../validation/sediment/reservoir_polygon/dry_source_regression/SCHEME6_NATIVE_REGRESSION.md) pass, including thread/restart equality. Wet sediment reconstruction is unchanged.

The [subsequent startup correction and reservoir snapshots](../../validation/sediment/reservoir_polygon/startup_fix/reports/STARTUP_AND_RESERVOIR.md) resolve the terminal rejection: zero-water-flux faces no longer reject because their cell drains through another face. Cells whose reconstructed bed excursion exceeds their water depth, or whose reconstructed face becomes dry, use cell-average hydraulic traces on all faces. This prevents an unresolved thin film from generating a fictitious deep pressure column. Concentration reconstruction and FCT are retained; hydraulic order is locally reduced, so general low-diffusion qualification is not inferred. Both changes remain native Fortran/OpenMP. The 0.02 s startup completes in two steps; 61 compact native checks and actual-resource thread/restart checks pass.

The later timestep diagnosis found a second omission: unlike the main WAVE path, the cut-cell RK stages did not clear dry-cell momentum or apply GeoClaw's 50 m/s speed limit. An intermediate full cell just above `dry_tolerance` therefore generated transient velocities above 3,000 m/s. The cut-cell solver now performs the same preparation before every stage. This changes momentum only; water and sediment inventories are untouched.

An inlet and an outlet have separate speed-limit failure diagnostics. For a
critical-depth outlet, reason 7 prints the numeric boundary ID, attempted trace
speed and 50 m/s limit. The QGIS preparation path evaluates the same initial
critical trace and rejects an unsafe setup before starting the executable. This
usually indicates that a long/deep polygon face was assigned as the outlet when
only a physical opening was intended; reducing the timestep does not alter that
boundary state.

The [completed source-consistent resource event](../../validation/sediment/reservoir_polygon/startup_fix/speed_limit/reports/RESOURCE_RESERVOIR_750.md) reaches 600 s at the 0.05 s timestep cap with zero rejected trials and extends to 750 s. Sediment reaches the initial lake, deposits locally by 600 s, and produces 9.94 t positive net lake bed gain at 750 s. Budgets, threads and restart checks pass. The independent 0.025 s run and [current-source pulse check](../../validation/sediment/reservoir_polygon/startup_fix/speed_limit/diffusion/LOW_DIFFUSION.md) quantify timestep sensitivity and resolved scalar diffusion. The locally constant hydraulic fallback at unresolved wetting fronts remains first order and is not claimed to be generally nondiffusive. Complete output frames are flushed and off-schedule restart output is preserved. Source changes require a new build/run identity. The diagonal-accuracy investigation remains explicitly postponed by the user.

## Current QGIS development runtime

Run `python3 tools/build_cutcell.py` from the repository root to compile the
current Fortran/OpenMP solver into `solver/cut_cells/build/xbraids_cutcell`.
The QGIS dock uses this executable by default when it exists; saved projects
retain their explicitly selected executable. The build record includes source
and executable hashes. Set `FC` to the GNU Fortran executable if needed.

Reference-height factors other than 1 and Manning-derived sediment roughness
require a current executable supporting input format v10. Rebuild an older
custom executable before using those settings. This development build is
separate from distribution packaging.

### Physical inlet terrain (native deck v11)

The QGIS preparation path samples the DEM **on the selected inlet edge** at
common quarter-pixel stations. It subdivides only boundary flux-integration
faces; control volumes, Manning assignments and cell-average terrain are
unchanged. Each sample keeps its initial elevation offset from its owner cell.
The native solver adds the owner's bed increment as morphology evolves.

Section-controlled inlet areas and traces use those physical edge elevations;
legacy mode-1 characteristics use them as well. A horizontal section fills from its initial minimum;
disconnected low pockets stay closed until the dividing ridge is submerged.
The boundary pressure has the same cell-to-face hydrostatic source correction
as the interior reconstruction. Water and both sediment fluxes retain one total
supply per logical boundary ID, irrespective of the number of integration faces.
All timestep work remains Fortran/OpenMP. Versions 1–10 retain their previous
owner-bed representation; v11 requires a rebuilt native runtime. Other inlet
modes and explicit outlet modes retain their existing treatment.

The [geometry and reservoir study](../../validation/sediment/reservoir_polygon/geometry_resolution/PROGRESS.md)
records the real DEM issue, native still-water/filling/conservation/thread/restart
checks, QGIS screenshots, and the active reservoir resolution assessment. A
correct inlet opening does not by itself establish adequate interior resolution.

### Multiple fractions, active composition and schedules (native decks v12–v15)

`CutSedimentFraction` supplies a stable ID, initial volume fraction and the
material properties used by suspended exchange, settling and bedload. A v12
case contains two to 32 fractions. Initial erodible thickness is partitioned by
solid volume, while the native solver evolves a separate suspended inventory,
bed inventory, boundary concentration, total bedload supply and accepted ledger
for every fraction. Hydraulics are shared. Each fraction uses the selected
transport law with its own grain and settling properties.

The QGIS Sediment tab creates and persists the fraction table, edits constant
or piecewise-linear per-fraction inlet supplies, plots those schedules, and
exports aggregate plus per-fraction result fields and budgets. The legacy
blank-table route retains the single-fraction deck and its existing output
layout.

Setting a positive active-layer thickness selects deck v13. It adds a separate
active and substrate inventory for every fraction. Surplus active material is
buried with the active composition; erosion deficits expose the substrate
composition. Upwind active volume fraction and thin-layer availability limit
equilibrium bedload and suspended entrainment. Existing aggregate bed results
are retained, with per-fraction active inventory, substrate inventory and
active volume fraction added for v13. Zero active-layer thickness retains v12.

[The multiple-fraction foundation proof](../../validation/sediment/multifraction/reports/MULTIFRACTION_FOUNDATION.md)
records independent class transport and QGIS round trips. [The active-layer
proof](../../validation/sediment/active_layer/reports/ACTIVE_LAYER_VERIFICATION.md)
records independent burial recurrence, selective exposure, class closure,
OpenMP equality and exact restart. The current-source qualification below adds
all remaining hiding/exposure choices and physically matched armoring; preserved
multiple substrate layers are also implemented.

Deck v14 adds hiding/exposure selection. Egiazaroff and Ashida-Michiue use the current
active-layer volume-weighted arithmetic D50 and multiplies each fraction's MPM
critical Shields threshold. It is accepted only for MPM with a positive active
layer; unsupported combinations fail during preparation and native loading.
Pure and equal-grain limits, selective transport, per-class budgets, OpenMP and
restart are covered by the historical [focused verification](../../validation/sediment/hiding_exposure/reports/HIDING_EXPOSURE_VERIFICATION.md).
The production solver also implements Parker-Klingeman-McLean and Wu-Wang-Jia;
deck v25 wraps established layouts with their explicit calibration exponents.
All four relations and paired three-fraction native sorting are covered by the
[current-source qualification](../../validation/sediment/delft3d_current/hiding_exposure_sorting/reports/REPORT.md).

Deck v26 is a configuration envelope for a water-depth-scaled active layer.
The target thickness is `max(alpha * h, minimum)` in every retained cell and is
rebalanced conservatively as depth changes; the existing constant-thickness
mode remains the default. Preparation requires the initial active inventory to
match the per-cell target exactly. Paired current-source Delft3D `TTLForm=2`
evidence, per-fraction budgets, sensitivity and OpenMP determinism are retained
in the [active-layer sensitivity report](../../validation/sediment/delft3d_current/active_layer_sensitivity/reports/REPORT.md).
The same audit rejects vertical bed diffusion after the locked Delft3D runtime
failed nonphysically, so v26 does not add an unqualified mixing operator.

Deck v15 retains the v13 active/substrate layout and writes a hiding/exposure
mode explicitly. It adds a knot count and contiguous
`time, concentration, total bedload` rows for every boundary/fraction pair.
Values are held outside their range and interpolated at SSPRK stage times;
timesteps stop at all hydraulic and sediment knots. The compact proof covers
different knot grids, analytic class integrals, threads and restart. The
[resource-reservoir event](../../validation/sediment/reservoir_multifraction/reports/MULTIFRACTION_RESERVOIR.md)
uses this path through QGIS preparation and native result materialization.

### Conservative bank morphology (native deck v17)

Deck v17 adds a per-cell erodible-bank mask, a configurable fraction of wet-cell
erosion supplied from adjacent higher dry banks, and optional wet/dry slope
failure. Bank polygons mark whole retained cells on positive-area contact; their
union is exported without splitting cells. Every transfer crosses one internal
mesh face, uses the donor active-layer composition, and is limited by the
available per-fraction inventory and bedrock. Faces without a retained neighbour
never participate, so the calculation-domain boundary remains geometrically
fixed. A lowered dry bank is wetted later by the ordinary native hydraulic flux.

The slope operator uses the supplied Delft3D D-Flow FM wet/dry slope selection,
harmonic cell-area link factor and relaxation-time flux. Default wet slope,
dry slope, relaxation time and wet/dry depth switch are 10, 1, 86400 s and
0.1 m. Parallel loops retain a deterministic within-cell gather order. The
[focused proof](../../validation/sediment/bank_morphology/reports/BANK_MORPHOLOGY_VERIFICATION.md)
matches the source-derived SSPRK3 operator to roundoff, conserves every class,
checks bedrock, hydraulic wetting, exact restart and byte-identical one/eight-
thread output.

### Roughness-corrected graded MPM (native deck v18)

Deck v18 adds an explicit MPM shear-closure code. Code 0 retains the specified
friction coefficient used by the Zhu benchmark. Code 1 follows the supplied
FLOW `tranb2.f90` current-only closure: local depth and cell Manning define total
Chezy resistance, active non-mud volume shares define the mixture D90, the
critical Shields value is fixed to the source value 0.047, and the grain-related
Chezy ratio supplies the bounded ripple correction. The mixture
D90 uses the backward-compatible piecewise-loguniform fraction distributions
assembled by `compdiam.f90`. It may be smaller than a rare coarse fraction's
D50, as allowed by the source formula.

The roughness-corrected mode requires MPM, at least two fractions and a positive
active layer. The choice enters the restart signature and is available in the
QGIS Sediment tab. The direct compiled-source test matches `tranb2` to roundoff;
the short compatible combination matrix keeps active hydraulics unchanged and
matches equilibrium inlet bed load within 6.91% across all eleven pairs.
[Matrix and budgets](../../validation/sediment/delft3d_intercomparison/combination_matrix/reports/SEDIMENT_COMBINATION_MATRIX.md).

### Concentration-dependent hindered settling (native deck v20)

Deck v20 wraps the required existing state layout and adds an explicit settling
treatment plus total suspended reference concentration. This envelope supports
the legacy single-material and all multi-fraction/layer layouts without changing
their state columns. Older executables reject v20 instead of silently ignoring
the new closure.

The optional Richardson–Zaki/Mehta treatment evaluates
`max(0, 1 - Ctotal / Creference)^5` once per wet cell and source step, where
`Ctotal` sums all suspended dry-mass concentrations. It multiplies every
fraction's configured fall velocity before settling-only, Van Rijn, or cohesive
deposition. Existing finite bed/suspension bounds conserve each class, and the
new settings enter restart identity. The default reference concentration is
10,000 kg/m³. [Production-path proof](../../validation/sediment/hindered_settling/reports/HINDERED_SETTLING_VERIFICATION.md)
checks the dilute and concentrated limits, independent nonlinear recurrences,
unchanged hydraulics, exact restart, and byte-identical one/eight-thread frames.

### Continuous elevation preparation

DEM and bedrock elevations now use exact cell integrals of the bilinear surface
through source pixel centres, consistent with inlet sampling and GeoClaw's
Cartesian terrain integral. This avoids artificial bed steps from fractional
pixel alignment. It changes prepared elevation/initial-state arrays, requiring
a fresh case; existing pinned decks retain their original inputs. Integration
patches do not split computational cells. Raster-valued material thickness
keeps its pixel-average treatment. Native stepping and OpenMP are unchanged.
[Independent proof and scope](../../validation/sediment/reservoir_polygon/geometry_resolution/reports/CONTINUOUS_TERRAIN.md).

### Bedload at dry internal steps

Water-driven bedload now uses GeoClaw's reflected-wave classification of a dry
bed step. A hydraulically closed face passes zero bedload before the finite-bed
donor limit is calculated. Lower dry beds and dynamically overtopped steps stay
open. This retains the transport formulas, physical thresholds and native OpenMP
loops. Existing checkpoints require their matching build.
[Independent closed-step test, actual-reservoir audit and low-diffusion checks](../../validation/sediment/reservoir_polygon/geometry_resolution/reports/CLOSED_DRY_FACE_BEDLOAD.md).


### Four-level adaptive production meshes (alpha 29)

Envelope v22 adds a fixed L0–L3 candidate tree to the existing native material
and boundary deck. The active leaves refine/coarsen at synchronization times.
Every conserved hydraulic/sediment inventory, active layer and substrate layer
shares the mesh. Area-weighted restriction, positive water/sediment prolongation,
2:1 face balance, deterministic coarse/fine flux synchronization and subcycled
SSPRK3 evolution are implemented. Checkpoints retain leaf IDs, cached child
state, coarsening history, next adaptation time and integration counters.

Adaptive frames begin with `-22 nb nfrac nbedlayers`; each frame has
`time steps leaf_count mesh_id`, then `node_id q...` rows and the usual per-ID
ledger. `amr_hierarchy.json` provides the exact geometry for result/preview cells.
`--initial-mesh` stops after the identical initial native selection used by a run.
Existing fixed-grid decks and results remain supported. QGIS API 16 prevents
stale embedded imports on upgrade.

[Qualification and limitations](../../validation/amr/AMR_VERIFICATION.md).

### Non-equilibrium bed-load adaptation (native deck v30)

Deck v30 wraps the existing v29-and-earlier physics with an optional steady
streamline relaxation, `L_g dq_g/ds = q_eq,g - q_g`. Each sediment fraction
uses `L_g = C_L D50_g`; the default `C_L=100` comes from a measured moderate-
transport mean and remains user calibration. The switch is off by default.

The native solver first evaluates the existing local equilibrium capacity, then
solves an exponential upwind graph relation between cell centroids (and inlet
face centroids) at every Runge--Kutta stage. Exner still consumes one shared
face flux and finite donor inventory. Because adapted capacity is derived from
the current state, no new checkpoint history is required; AMR recomputes it on
the current leaf graph. Equilibrium-capacity inlet conditions retain their
constitutive local-capacity meaning, while prescribed inlet bed load supplies
the upstream relaxation value.

The [qualification](../../validation/sediment/delft3d_current/adaptation_length/reports/REPORT.md)
passes the exact exponential profile and local-equilibrium limit, independent
per-fraction lengths, conservation, wet/dry, changing AMR, bitwise restart and
one/eight-thread identity. The compiled pinned Delft3D `tranb2` kernel supplies
the common equilibrium comparator; no Delft3D adaptation feature is claimed.

### Morphological acceleration (native deck v31)

Deck v31 adds a default-off conservative bed-load/Exner clock. A constant
MorFac from 1 to 20 or a block-interpolated time schedule multiplies the bed
flux before donor limiting, boundary accounting and AMR reflux. Transport
capacity is not multiplied. Timesteps stop at schedule knots and every SSPRK
stage within the step uses the factor held at the step start, matching the
current-source Delft3D `dtmor = dts * MorFac` contract.

`morphological_acceleration.txt` records hydraulic time, the exact integrated
morphological time and the instantaneous factor. The clock is reconstructed
from the immutable forcing schedule, so checkpoint restart adds no hidden
history. The schedule is included in the restart fingerprint.

The [paired qualification](../../validation/sediment/delft3d_current/morphological_acceleration/reports/REPORT.md)
publishes MorFac 20 for a quasi-steady subcritical translating bedform and
MorFac 1 for a wet/dry hydraulic transient. Multi-fraction conservation, AMR
reflux, exact restart and one/eight-thread identity pass. Suspended exchange,
time-varying hydraulic boundaries and both bank-erosion models are rejected
while acceleration is active. General avalanche relaxation remains on the
hydraulic clock, following Delft3D's explicit cancellation of MorFac there.
