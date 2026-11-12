# BRAIDS remaining work and sediment-orientation review

Updated 15 September 2026. This is the current, ordered checklist; older
roadmap sections retain historical failures and milestones. An unchecked item
means further work is needed, not that its outcome or a particular correction
has already been established.

**Latest completed delivery step — BRAIDS alpha 37 and development repository:**
the maintained source, tests, compact validation fixtures and required build
inputs are now in the private `cgotelli/BRAIDS` GitHub repository. Generated
runs, caches, copied upstream repositories and superseded packages were removed;
the pinned optional references can be restored with
`tools/fetch_reference_sources.py`. The product tree and packaged plugin contain
only BRAIDS paths and identifiers. The Apple Silicon QGIS package
passes all 120 installed-profile smoke checks with eight OpenMP threads. The
clean source checkout passes 2,101 tests and explicitly skips 88 optional
external-reference checks. Alpha 37 is ready for manual QGIS acceptance.

The AMR inlet mesh-preview preparation error is fixed in alpha 32:
the hierarchy now uses the same DEM-sampled boundary faces as the case.
[Verification](../validation/qgis/alpha_build/alpha32/AMR_INLET_PREVIEW_FIX.md).
The user confirmed that the corrected mesh preview works. The current-source
sediment investigation below and its localized inlet correction are complete.
Alpha 33 applies a consistent bounded bedload closure over the boundary stencil
radius. The resolved Zhu-derived 300 s evolved-change check is complete. The
compact rotated wet/dry multi-fraction case exposed the Cartesian slope defect
and the mesh-scaled bank-relaxation rate. Both are now corrected. Its integrated
response passed rotation and grid-convergence gates in alpha 35. The new
development wet/dry face reconstruction passes its analytical oracle. The
isolated bank toe now converges against a resolved reference in both orientations;
the finer coupled checks exposed stage-outlet activation/reversal rejections,
now corrected. Sediment orientation passes at 1.75 cm; the remaining numerical
limitation was a shallow inlet transient. The hydraulic face correction below
improves the steady shoreline control; the subsequent paired experiment
identifies incompatible initial/incoming velocity profiles as the main trigger
of the remaining inlet peaks. Its matched coupled control passes the retained
speed limit. The original transient stress case's failed watchdog is preserved.

**Latest completed step — AMR trigger diagnosis and selectivity correction:**
opt-in native masks show that the common characteristic forecast was applied to
static bathymetry and walls, adding about 12,569 m² of buffer-only coverage on
average. Static triggers now keep the configured geometric stencil while travel
distance is limited to moving wet/dry, hydraulic, sediment and evolving-bed
fronts. A prospective 300 s qualification reduces active leaves from 5,097 to
4,428 and finest-area coverage from 80.0% to 67.9%. Maximum uniform-reference
errors remain 0.0127% depth, 0.4396%/0.6332% fine/coarse suspension, 0.5686%
bed change and 4.7739% velocity, within the unchanged limits. Conservation,
inlet forcing, diagnostics-on/off, one/eight-thread and an exact full repeat all
pass. Runtime is 445.3 s versus 559.0 s fixed (1.255×); instrumented solver CPU
falls 14.0%. All 98 focused and 2,189 repository tests pass. Packaging remains
deferred in that historical run; it is now included in alpha 37.
[Report and figures](../validation/amr/bounded_reservoir_deposition/amr_selectivity/REPORT.md).

**Preceding completed step — common-state reservoir AMR qualification:** uniform
2.5 m and AMR 20→2.5 m trajectories start from exact common initial fields and
run through 300 s with the corrected inlet and a 0.0125 s timestep cap. A fixed
0.025 s control confirms temporal resolution. Maximum AMR differences are
0.0094% depth, 0.4906%/0.7053% fine/coarse suspension, 0.5918% bed change and
3.9272% velocity, all within the retained limits. Conservation, prescribed
per-ID forcing, positivity and identical-deck one/eight-thread identity pass.
AMR takes 529.7 s versus 559.0 s fixed, only a 1.055× speedup, and 96.3% of its
final leaves remain at 2.5 m. Peak memory is also higher. The gate therefore
passes, but refinement selectivity is the next optimization target.
[Report and figures](../validation/amr/bounded_reservoir_deposition/corrected_amr/REPORT.md).

**Preceding completed step — production multi-channel inlet correction:** native
stage/depth/normal/critical inlets, Python sections, preparation and QGIS previews
now use every submerged portion of the assigned section at one horizontal
level. Dry faces remain closed to prescribed inflow; initial-lake connectivity
is unchanged. A related critical-root defect is fixed by finding stationary
discharge rates before selecting the least-energy level. Native/Python geometry
and dry-zone roughness grouping agree. The 480–540 s reservoir gate passes:
maximum history differences are 1.49% coarse suspension and 4.30% velocity;
the timestep comparison gives 0.163% and 0.289%. All 604 focused tests, four
QGIS preview modes, final-source one/eight-thread identity and cross-thread
restart pass. All ten flume replays, including seven bedload laws, retain
byte-identical frames/checkpoints and the original inlet-spike limits. Interior
hydraulic/transport routines are unchanged. Development is rebuilt; alpha 36
remains installed and packaging deferred.
[Correction, checks and figures](../validation/amr/reservoir_window/production/REPORT.md).

**Preceding completed step — short reservoir inlet diagnosis:** the 480–540 s
saved-state window reproduces the failure on the same fixed 6,138-cell mesh
and fine timestep. Common-state timestep refinement passes (maximum 0.36%
coarse suspension and 0.49% velocity), but two slightly different saved
histories diverge by 15.52% and 25.03%. The lowest-connected-section rule
removes supply from a still-submerged branch when a small intervening crest
crosses the imposed stage. A separate diagnostic supplies every submerged
portion of each assigned inlet and reduces those maxima to 1.49% and 4.30%.
The supplied Delft3D discharge-weighting routine uses individual wet boundary
points without this lowest-component selection. Budgets, independent geometry,
observer identity, connected-rule one/eight-thread identity and checkpoint
joins pass. The longest run takes 188.1 s; no production or package change
was made. Next correct the boundary selection consistently and qualify it;
full reservoir AMR accuracy remains open. Packaging remains deferred.
[Diagnosis, retained comparisons and figures](../validation/amr/reservoir_window/REPORT.md).

**Preceding completed step — compact evolving inlet crest:** an 8 × 4 m, 512-cell
two-channel case reproduces closure and, with combined transport, reopening.
Accepted RK-stage section geometry and actual face fluxes satisfy an independent
section integral and exact Q/Q·C ledgers. Nine timestep-refinement runs retain
all 2/8/15/10% field limits. Between the finest combined pair, maximum
coarse-suspension and velocity differences are 0.01284% and 0.01582%.
The combined error is not strictly monotonic at coarser steps because the
crest can cross the stage within different RK stages. No unsupported formal
order or full-reservoir convergence claim is made. Observer/native, restart
and one/eight-thread identities pass, including an 8,192-cell prefix that
enters the actual OpenMP loops. All 10 focused section/inlet tests pass.
The longest run takes 112.8 s. This established faithful implementation of the
chosen connected-section rule. The reservoir control above subsequently
exposed why that rule is too restrictive for a multi-channel inlet.
[Report and evolution/accuracy figures](../validation/sediment/inlet_crest/REPORT.md).

**Preceding completed step — reservoir inlet/timestep audit:** the late failure
also occurs with all 6,138 cells held at the finest level: the final coarse-
suspension error is 13.93%. Matching the uniform timestep cap and output-aligned
regrid schedule reproduces the 600 s reference to roundoff (maximum scaled
state difference 6.56e-52), with identical inlet ledgers. Coarsening is not a
necessary cause, so the earlier coarse-history diagnosis was too strong. In
the original AMR run an evolving inlet crest splits its connected supplied
span from 116.5 m to 67.75 m at the saved 420 s frame, redistributing the same Q.
Two independent native defects are fixed: prospective-parent bed detail now
limits coarsening, and fully wet finest-only startup allocates subcycling work
arrays even when the mesh does not change. The 8 × 4 m saddle check reduces
lost bed detail from 28.125 to 3.125 mm under the unchanged 20 mm limit, using
164 versus 512 uniform cells. Focused AMR/inlet checks, compact one/eight-thread
identity and restart pass. Reservoir accuracy remains open; no thresholds or
hydraulic/transport formulas were changed. Packaging remains deferred.
[Report, retained failures and figures](../validation/amr/reservoir_routing/REPORT.md).

**Preceding completed step:** the exact 300 s uniform and AMR checkpoints were
continued to 600 s on eight threads. Sediment now reaches beyond 60 m: the
uniform reference contains 5.721 kg suspended sediment and 22.612 kg gross
positive bed gain there. Restart identity, prescribed forcing, water and class
budgets, nonnegativity, depth and bed-change gates pass. The retained 8%
suspended-field gate fails from 360 s onward, reaching 27.90%/46.09% for the
fine/coarse fractions at 600 s. Integrated reservoir suspended mass differs by
only 0.46%, but velocity-vector L1 rises to 54.68% at 420 s, peaks at 122.11%
at 480 s and remains 88.64% at 600 s. By 420 s the AMR mesh is already almost
entirely finest-level. The inferred coarse-history cause was not established
by that study; the later controls above also fail without coarsening. This
AMR configuration remains unqualified for reservoir sediment transport
and provides only a 1.063× measured 0–600 s speedup.
[Failed qualification, diagnostics and figures](../validation/amr/bounded_reservoir_deposition/continuation_600/REPORT.md).

**Preceding completed step:** the optimized solver passes the first 300 s of
the same 240 × 160 m real-DEM sensitivity. Uniform 2.5 m and AMR 20→2.5 m use
identical inputs. Beyond the first 20 m, the uniform result retains 144.28 kg
suspended sediment and 689.54 kg gross positive bed gain. Final AMR differences
are 0.0087% depth, 2.30%/3.76% fine/coarse suspension and 1.28% bed change. AMR
takes 293.1 s versus 319.8 s uniform (1.091× faster). All 14 predeclared checks
and two inherited read-only inlet audits pass. Only 0.126 kg suspension and
0.047 kg deposition reach beyond 60 m at that time.
[Report and figures](../validation/amr/bounded_reservoir_deposition/REPORT.md).

**Preceding completed step:** deterministic level-indexed AMR cell, face,
reconstruction-halo, reflux and subtree lists are now built with each mesh, and
Runge–Kutta, dense-interpolation and flux-register storage is reused. On the
unchanged compact 8 × 6 m fully wet two-fraction gate, eight-thread subcycling
falls from 41.51 to 35.25 s (15.1%) and peak RSS falls from 177.1 to 124.6 MiB
(29.6%). Pre/post frames are byte-identical. One/eight-thread frames and
checkpoints are byte-identical, the midpoint restart is exact, and water/two-
fraction budgets close to roundoff. A faster inactive-cell shortcut changed the
frame hash and was rejected rather than included. Because this microbenchmark
is nearly refined to the finest level, subcycling still takes 35.25 s versus
25.68 s for synchronous AMR; coarse-cell savings must be assessed in the next
reservoir case.
[Report and figure](../validation/amr/fully_wet_performance/REPORT.md).

**Preceding completed step:** full-resource 10→1.25 m AMR preparation is now
compact and single-pass. Preparation falls from 794.85 to 661.69 s (16.8%),
and post-terrain work falls from 396.86 to 317.83 s (19.9%). Typed sparse
geometry/hierarchy sidecars reduce those artifacts by 86.9% and all retained
preparation outputs by 67.4%. The qualified 791.5 MB native deck and initial
fields remain byte-identical; eight-thread startup retains 61,427 active leaves.
Legacy JSON remains readable and large QGIS results load only selected active
AMR geometry. Peak RSS is not improved and hierarchy construction remains the
memory target. All 2,415 repository tests pass.
[Report and figure](../validation/amr/full_reservoir/preparation_optimized_benchmark/REPORT.md).

**Preceding completed step:** the corrected full-resource 10→1.25 m same-finest
AMR gate passes. With eight OpenMP threads, all ten retained water, solids,
forcing, positivity, depth and near-inlet morphology checks pass. Depth L1
against uniform 1.25 m is 0.0379%; the inlet bed-rise difference is 0.00492 mm
against the unchanged 0.1 mm limit. AMR keeps about 3.5% of the uniform cells
and its native solve is 4.01× faster (89.92 versus 360.45 s). Preparation now
dominates: 794.85 s AMR versus 464.27 s uniform, making this one-second AMR
workflow 7.28% slower end to end. Sediment has not reached the lake and the
young sediment maps remain resolution-sensitive, so reservoir deposition is
not qualified. [Report and figures](../validation/amr/full_reservoir/same_finest_optimized/REPORT.md).

**Preceding completed step:** both remaining compact AMR gates are closed. The
recursive flux register now retains descendant boundary integrals and reconciles
each prescribed water, suspended and bedload schedule at synchronization. A
fully wet two-inlet proof exercises asynchronous L1/L2/L3 updates at 20/40/80
steps and matches every analytical per-ID integral to roundoff. Moving wet/dry
fronts use a synchronized SSPRK step because interpolated coarse ghost states
are not a conservative wet/dry Riemann state; fully wet flow still subcycles.
Against maximum-admissible realizations of the same hierarchy, maximum depth L1
is 0.1991% combined, 0.1876% bedload-only and 0.1804% suspended-only, below the
unchanged 2% limit. The combined final inlet-peak difference is 0.0000185 mm.
One/eight-thread and restart results are exact, 38 focused tests and all 2,414
repository tests pass. The independent uniform mesh remains recorded as a
cut-cell topology sensitivity because it agglomerates across AMR parent
boundaries. [Report and figures](../validation/amr/near_inlet_wetfront/REPORT.md).

**Earlier same-finest development step:** the optimized full-resource 10→1.25 m same-finest
AMR comparison now completes. Preparation fell from 3,185.73 s to 641.35 s.
A fine-substep suspended/active-bed reflux defect that caused repeated rejected
steps was corrected by preserving the local grain-class total while restoring
a nonnegative phase split. Water, solids, forcing, positivity and depth pass;
depth L1 is 0.0613%. The retained morphology gate remains failed: near-inlet
bed rise is 0.2212 mm under AMR versus 0.0326 mm on the uniform 1.25 m grid.
The peak is a regular finest cell at the advancing wet/refinement front. Native
subcycling is now the main cost: AMR takes 1,078.48 s versus 382.67 s uniform
despite using 3.37–4.11% as many active cells. After compacting the evidence,
another 6.53 GB of raw files was removed. Alpha 36 remains the pre-optimization
manual-test baseline. [Same-finest report and figures](../validation/amr/full_reservoir/same_finest_optimized/REPORT.md).

**Preceding completed step:** the corrected inlet passes a complete-resource-domain
uniform 2.5/1.25 m startup comparison and representative 20→2.5 m AMR check at
1 s. All 15 numerical gates pass. Peaks remain 0.0193–0.0326 mm, uniform depth
L1 is 0.279%, representative AMR depth L1 is 0.0435%, and AMR is 1.74× faster
than uniform 2.5 m in the native solve. Sediment-map L1 differences remain
large at this very young front and the lake receives no sediment. The intended
10→1.25 m AMR worker timed out at 20 minutes; exact hierarchy preparation took
53.1 minutes. The successful fallback workflow is estimated at 56.4 minutes,
while the attempted same-finest campaign misses one hour.
[Report and figures](../validation/amr/full_reservoir/corrected_2s/REPORT.md).
The optimized same-finest follow-up and its remaining AMR morphology failure
are recorded above.

**Preceding completed step:** the compact exact-edge diagnosis led to a production
correction of the hard shallow-wetting bedload-capacity switch. A C1 capacity
ramp acts only between the hydraulic dry tolerance and the configured minimum
transport depth; it does not smooth the bed, cap prescribed input, change
hydrodynamics or alter fully wet formulas. The combined near-inlet peak is now
0.0561/0.0660/0.0906 mm at 2.5/1.25/0.625 m, down 78.4–96.5%. Exact boundary
input, roundoff budgets, hydraulic identity, fully wet controls, one/eight-thread
identity, restart identity and 105 focused tests pass. The rejected bounded
inlet allocator remains isolated because it violated exact prescribed loading.
Integrated near-zone deposition is still resolution-sensitive.
[Correction and figures](../validation/sediment/near_inlet_compact/production_correction/REPORT.md).
The short complete-domain follow-up is completed above.

**Preceding completed step:** a **25 s complete-resource-domain** comparison at
uniform 10/5/2.5 m and AMR 20→2.5 m finishes within the requested hour. All
16 conservation/forcing/nonnegativity checks pass. Hydraulic depth sensitivity
improves from 4.744% to 3.067% under uniform refinement; AMR is within 1.533%
of uniform 2.5 m and is 1.77× faster, but uses 1.59× more peak memory. Sediment
has not reached the lake footprint, and whole-domain sediment differences fail
the retained diagnostic limits, so reservoir-sedimentation convergence remains
unqualified. [Report and figures](../validation/amr/full_reservoir/representative_25s/REPORT.md).

**Preceding completed step:** the **240 × 160 m real-DEM reservoir window** passes
**33/33 numerical and offscreen QGIS checks** through 120 s on eight threads.
Two inlet IDs feed suspended sediment and bedload into a closed crop; both
fractions reach the reservoir entrance and deposit. Maximum added AMR errors
are 0.0101% depth, 0.0556% suspended inventory and 0.0299% evolving bed change.
Water/class budgets, exact restart/checkpoint identity, preview, raster/mesh
animation, multiple run layers, deleted-layer recovery and QGIS reopening pass.
Uniform/AMR main runtimes are 84/113 s; this mostly refined window does not
show an AMR speedup. Source operators and installed alpha 35 are unchanged.
[Report, snapshots and saved QGIS project](../validation/qgis/compact_reservoir/COMPACT_RESERVOIR_QGIS.md).

**Preceding completed step:** the production QGIS section inlet and coupled sediment
AMR pass **37/37 compact checks** in a **2 × 2 m** aligned/45° flume, with
2 cm finest cells and a 0.3 s duration. Maximum AMR-versus-uniform differences
over saved times are 0.073% in depth, 1.39% in either suspended inventory and
1.06% in evolving bed change. Exact accepted inlet volume, water/class budgets,
dynamic leaf changes and one/eight-thread restart identity pass. The complete
one-thread trajectory uses a checkpoint at 0.15 s to bound each worker.
No production source changes; installed alpha 35 is unchanged.
[Report and evolution figures](../validation/amr/compact_section/COMPACT_SECTION_AMR.md).
The bounded reservoir/QGIS follow-up is now completed above.

## Final objective and current scope

A practical QGIS shallow-water and morphology solver for reservoir deposition
and mobile river channels: conservative water, suspended sediment, bedload and
bed-layer evolution; low numerical diffusion; editable polygon inputs;
multiple inlet/outlet IDs; and reproducible native OpenMP execution.

Already implemented: seven current-driven bedload laws, the current suspended
exchange options, cohesive/mixed-bed exchange, hindered settling, multiple grain
fractions and preserved layers, bank erosion/slope failure, exact reflective
polygon obstacles, and four shared AMR levels. These are not all missing work.
The alpha includes QGIS drawing, run/restart/log controls, raster results and
animated mesh overlays. Each inlet uses one polygon face, outlets may use one
or several, and unassigned faces remain walls.

The existing tests establish specific capabilities, not universal accuracy or
complete equivalence to Delft3D. Full three-dimensional stratification and
vertical sediment profiles are outside a depth-averaged model's present state.

## 1. Resolve the sediment-orientation question first

**Current-source investigation completed:** [report, metrics and snapshots](../validation/sediment/orientation_current/results/reports/CURRENT_ORIENTATION_REPORT.md).
The fine smooth-pulse L1 errors are 0.049% aligned, 0.299% at 45°, and 0.291% at
30°. The main rotated error is near cut walls and decreases with refinement.
Resolving the steep Zhu-derived front reduces its initial-rate maximum error
to 0.0525% aligned and 1.0737% diagonal at 0.3125 m in the local front window.

**The separate inlet reconstruction defect is corrected in alpha 33:** an
affine bedload capacity now gives the exact constant Exner rate through the
inlet cells (relative error below 9e-14), replacing the 50%/150% pair. The
bounded midpoint closure covers the complete five-cell stencil radius.
Hydraulic equations and hydraulic BCs are unchanged. Pure hydraulic and
suspended-only replay frames/checkpoints are byte-identical to alpha 32.
[Correction, frozen inlet-spike controls and snapshots](../validation/sediment/inlet_reconstruction/corrected/reports/INLET_RECONSTRUCTION.md).
The correction passes 201 numerical controls, 20 focused tests and 117 packaged
QGIS checks. A 5,760-cell diagonal case has byte-identical one/eight-thread
frames/checkpoints and exact restart. That correction remains included in the
installed alpha 35.

The earlier 15 s, 5 m coupled run has maximum bed-change errors of 7.68% aligned
and 18.21% diagonal; the steep transition is underresolved.
All 178 integrity controls pass, including actual coupled one/eight-thread and
restart identity; those historical controls did not override the then-failed inlet
accuracy test. The separate alpha 33 correction report records its closure;
local rotated wet/dry bank evolution and adaptive orientation tests remain open.

**Compact wet/dry multi-fraction check: current development result is partial.**
[Report, metrics and figures](../validation/sediment/wetdry_multifraction_bank/reports/WETDRY_MULTIFRACTION_BANK.md).
The same 15.556 × 5.657 m case keeps two suspended fractions, bedload, exposed
and substrate composition, dry-bank supply and slope failure. The raw 15%
orientation criterion, 2% integrated refinement criterion and all inputs remain
unchanged. Alpha 35 runs/source are preserved under the case's history folder.

The development bank operator now reconstructs subcell wetness from each cell's
conserved water volume and integrates the wet/dry repose activation along each
physical face. This changes only bank morphology; the hydraulic equations and
BCs were untouched by that bank change. With banks disabled, full hydraulic/suspended/bedload frames
and checkpoints are byte-identical to alpha 35. Separate face-rate/conductance
arrays and checking the true transfer donor also remove two unsafe work-array
and mask paths.

The independent native planar wet/dry face-flux oracle agrees to **7.60e-16
m³/s** in aligned and 45° cases. However, the evolved raw bank-change discrepancy
still fails: **47.85% at 0.14 m and 25.99% at 0.07 m**, versus alpha 35's
58.76% and 29.14%. The coarse 0.28 m discrepancy worsens from 74.47% to 87.67%.
The 0.07 m integrated orientation difference is 0.181%, below 3%, but aligned
0.14-to-0.07 m refinement changes the integrated response by **2.362%**, above
the retained 2% limit; rotated refinement changes it by 1.696%.

The native initial-source audit locates **99.86% of the diagonal interior
absolute rate error near the sharp bank toe**, where the flat bed meets the
slope. Away from that toe, the wet-front initial-rate L1 error is 0.203%.
The previous attribution of the entire evolved error to wet/dry placement was
too strong. A raw comparison also includes different cell-average
representations; no rigorous representation-error floor is claimed here.

**Bank-toe and Delft3D follow-up completed:**
[source comparison, independent reference and figures](../validation/sediment/wetdry_multifraction_bank/reports/BANK_TOE_AND_DELFT3D.md).
The actual supplied FM dry-bank kernel matches BRAIDS to 2.31e-14 kg/m² per
fraction in four small matched controls. FM separates dry-bank replenishment
from slope failure; its link-slope construction also uses neighbouring bed
values. This is kernel evidence, not a complete Delft3D simulation or universal
equivalence on unequal cut cells.

The unchanged native bank source, isolated in a 4 × 5.657 m strip, converges
against an independently resolved 1D reference. Bed-change L1 errors are
23.68%/22.19% aligned/45° at 0.14 m, 5.11%/7.94% at 0.07 m, and
**1.88%/1.78% at 0.035 m**. Reference refinement changes the comparison by
less than 0.05%. The isolated fine raw orientation difference is **13.98%**,
also below 15%; even the same reference projected onto those meshes differs
by 13.89%. All eight source/kernel checks pass, including class conservation
and actual one/eight-thread identity on 18,940 cells. These results support
underresolution of the source at the coarser spacing, without another native
bank formula change or smoothing.

The original coupled failures above remain recorded. The subsequent
[dry stage-outlet correction and compact replay](../validation/sediment/wetdry_multifraction_bank/dry_outlet_transition/DRY_OUTLET_TRANSITION.md)
fix the timestep collapse when a bank cell becomes wet above the imposed
tailwater. Resolved drainage uses the outgoing-characteristic critical trace;
unresolved shallow controls close locally. Admissible wet-stage behavior and
the interior SWE are preserved. Both previous wet-outlet coupled controls
retain identical frames/checkpoints. The exact failing 0.035 m decks now
complete with banks enabled and disabled in 8.00/6.62 s on eight threads.

The preceding 4 m coupled strip batch passed **15/17 checks and 56 focused tests**.
Integrated refinement now passes: 0.566% aligned and 0.371% rotated, against
2%; integrated fine orientation differs by 0.00226%, against 3%. Raw local
bed-change disagreement is **15.719%**, still above 15%, and maximum transverse
velocity is **0.08615 m/s**, above 0.05 m/s, in a 2.494 mm-deep outlet cell.
Water/class budgets close to 3.78e-16/2.59e-17 relative, with exact one/eight-thread
and restart identity. No threshold was relaxed.

**The preceding shallow-flow follow-up passed 16/17 coupled checks and 132 focused tests.**
[Diagnosis, correction and figures](../validation/sediment/wetdry_multifraction_bank/shallow_outlet_audit/SHALLOW_OUTLET_AUDIT.md).
Raw bank-change disagreement falls to **13.469% at 2.5 cm and 10.205% at
1.75 cm**, below 15%. Integrated refinement is 0.0664%/0.0572% aligned/45°,
below 2%. The matched fixed-bed steady control is preserved to roundoff when
aligned but develops shallow transverse motion at 45°. Actual native face
fluxes attribute that motion to Cartesian wet/dry interfaces; the physical
outlet contributes zero transverse acceleration to roundoff.

Two further stage-outlet stalls are fixed: dry owner cells remain inactive
while the bed erodes below the target level, and local subcritical reverse
traces close with zero imported water/sediment. This is an explicit one-way
outlet, following the critical-outlet active set; admissible outward traces
and the interior SWE are unchanged. Previous wet controls and the 3.5 cm
coupled frames remain identical. A finest-grid 8-thread prefix followed by
1/8-thread continuations from 0.02 to 0.03 s matches the uninterrupted final
state, ledger and steps exactly.

The **0.05 m/s transverse-speed watchdog remains failed**: 0.05713 m/s in a
1.100 mm-deep rotated outlet cell, and 0.05483 m/s at the aligned inlet.
Halving the timestep changes the rotated value to 0.05700 m/s and bed evolution
by only 0.0471% L1. Blind refinement or additional sediment diffusion is not
justified. That result motivated the hydraulic face correction below.

**Hydraulic wet/dry reconstruction corrected; 81 focused tests pass.**
[Change, retained checks and figures](../validation/sediment/wetdry_multifraction_bank/hydraulic_wet_faces/HYDRAULIC_WET_FACES.md).
The fallback retains the reconstructed bed and integrates pressure only to
the dry face. Matched steady diagonal crossflow falls from 0.077675 to
0.005083 m/s; aligned steady flow and still water remain balanced to roundoff.
The original hydraulic/bedload flume frames and checkpoints are byte-identical.
The entire 36,708-cell coupled run is identical with one/eight threads and
restarts exactly. Water and every class remain conservative and nonnegative.

The updated batch passes 19/20 checks. Fine raw bank-change discrepancy is
11.803% at 1.75 cm, below 15%, but 15.652% at 2.5 cm; both are larger than
the preceding build's 10.205%/13.469%. Integrated refinement remains below
0.067%. The 0.05 m/s watchdog is still failed: 0.05704/0.05373 m/s
aligned/rotated, now near the inlet. Freezing bed evolution retains inlet
peaks (0.04734/0.06895 m/s), so this cannot be attributed solely to bank motion.
That result motivated the inlet-profile experiment below; the original failed
limit is retained rather than suppressing legitimate lateral flow.

**Inlet startup isolated; all 14 diagnostic checks pass.**
[Paired inputs, native trace oracle and figures](../validation/sediment/wetdry_multifraction_bank/inlet_startup/INLET_STARTUP.md).
The initial velocity is 0.6 m/s everywhere, but mode 6 initially imposes
0.01745–0.64212 m/s across the rotated section. Changing only the incoming
distribution, keeping the same total Q and serialized initial fields, reduces
fixed-bed maxima from 0.04734/0.06895 to 0.000000304/0.003901 m/s aligned/45°.
Matched coupled maxima are 0.04080/0.04351 m/s, below 0.05; fine raw bed-change
orientation is 11.8003%, below 15%. Native face discharges match the independent
sectional oracle to 5.20e-18 m³/s. Original frames/checkpoints replay exactly,
and matched one/eight-thread continuations and restart are exact.

This supports startup-profile incompatibility as the main trigger; it does
not establish the original transient's exact amplitude. No further production
boundary patch is supported by this check. The uniform profile exists only
in an isolated diagnostic binary; production/QGIS are unchanged. QGIS already
uses the different mode-9 section inlet with a common velocity. That compatible
production-inlet and compact AMR comparison is now complete:
[37/37 checks and figures](../validation/amr/compact_section/COMPACT_SECTION_AMR.md).
The new 2 m flume has a rounded bank toe and is an AMR added-error check;
it does not replace the earlier sharp-toe or transient results. Fixed-bed
startup maxima are 1.13e-14/0.01672 m/s aligned/45°, below 0.05. Maximum
integrated absolute bed-change differences remain below 0.875% (limit 3%).
Turning off subcycling reduces diagonal bed-field differences to 0.003%,
supporting a small timestep contribution rather than a large transfer error.
Main eight-thread uniform/adaptive native runs take 10–37 s; this small,
mostly refined case provides no AMR speedup claim.

**The preceding subcell batch passed 15/17 coupled checks and 108 focused tests.** Conservation, positivity,
suspended/composition orientation, one/eight-thread identity and restart are
retained. Main/finer/half-step runs close water and class budgets to relative
residuals of 1.957e-16 and 8.584e-17. Main execution uses eight threads.

The installed plugin remains **BRAIDS alpha 35**, whose package passed 120/120
QGIS checks and 2,372 repository tests. Its
[installation receipt](../validation/qgis/alpha_build/alpha35/installation.json)
still identifies the installed archive. **The development subcell build is not
packaged or installed as a newly qualified alpha.**

**Resolved evolved Zhu check completed:** [profiles, snapshots and error histories](../validation/sediment/zhu_evolved_resolved/results/reports/EVOLVED_ZHU_VERIFICATION.md).
The unchanged alpha 33 solver passes the retained 5% integrated and maximum
bed-change limits at every saved nonzero time through 300 s on nominal 0.625 m
meshes. At 300 s, integrated/maximum errors are 0.1526%/0.7532% aligned and
0.2259%/1.4542% at 45°; absolute maxima are 0.1718/0.3316 mm. The 1.25 m
diagonal grid fails the maximum limit at saved times 15–165 s, although it
passes at 300 s. Refinement improves both error measures in each orientation.
All 38 qualification/integrity checks pass, including budgets and a short
one/eight-thread plus restart identity control through 30 s. Main runs request
eight threads; the fine diagonal case has 4,360 cells and enters parallel loops.

This qualifies a fully wet, bedload-only, narrow quasi-1D adaptation retaining
the entire hump, initialized at 69 h and evolved for 300 s. The reference is
the Bernoulli characteristic approximation. It does not qualify the historical
wide-domain run, a full 0–70 h event, arbitrary angles, drying banks, suspended
transport or AMR. Absolute errors grow over this short window even though
relative diagonal errors decrease. No native code or hydraulic setting was
changed, and no bed smoothing, fitted shift or relaxed limit was used.

The following evidence predates this current-source batch:

The most useful retained evidence is:

| Evidence | What it shows | What it does not show |
|---|---|---|
| [Matched 15 s aligned/45° runs](../validation/sediment/zhu2018_oblique/orientation/results/reports/ORIENTATION_COMPARISON.md) | Maximum bed-change error was 8.14% aligned and 8.93% diagonal; integrated error was 4.32% and 3.51%. Diagonal transverse speed reached 0.003458 m/s, while aligned transverse speed was roundoff. | A diagonal-only explanation or a failed inlet mass balance. Both cases conserved accepted water and solids. |
| [300 s reflected-wall experiment](../validation/sediment/zhu2018_oblique/wall_weno/qualification/reports/WALL_WENO_VERIFICATION.md) | Integrated bed-change error was 4.44%; maximum error was 11.69%, or 1.880 mm. The maximum failed the retained 5% target. Absolute maximum error grew from 0.452 mm at 75 s to 1.880 mm at 300 s; the relative maximum stayed near 11–12%. | Accuracy at arbitrary angles, an unbounded instability, or an acceptable local error merely because the full profile looks similar. |
| [Aligned 5/2.5 m initial-rate probe](../validation/sediment/zhu2018_oblique/orientation/spatial_resolution/results/reports/SPATIAL_RESOLUTION.md) | Raw maximum rate error increased by 84.86%; its absolute integral increased by 1.90%. Aggregating fine cells hid opposite-signed errors at an internal face. The issue also existed in the aligned interior. | Asymptotic convergence or proof that no finer grid could work. The central 40–60% front width was only 1.85 m, below both cell sizes. |
| [Reassessment of evolved profiles](../validation/sediment/profile_reassessment/reports/PROFILE_REASSESSMENT.md) | The original aligned long run still met its whole-profile limits, with 0.144% peak loss. However, an unchanged-bed control also passed the broad whole-profile limits for the short oblique window. | That the short oblique test established low diffusion, or that runs of 70 h, 1 h and 300 s are directly comparable. |
| [Reference residual audit](../validation/sediment/zhu2018_oblique/diagnostics_audit/REFERENCE_RESIDUAL.md) | Short-window estimates of the Bernoulli-reference inconsistency were much smaller than the retained numerical bed error. | An exact unsteady SWE reference or a rigorously established error floor. |

These diagnostics concern earlier code, principally optional scheme 6. The
current QGIS default is scheme 5, and alpha 18 changed bedload transport to
reconstruct **cell-centred capacity** instead of evaluating nonlinear capacity
from reconstructed hydraulic traces. Subsequent suspended-limiter and AMR
changes also matter. Current implementation:
[bedload capacity reconstruction](../solver/cut_cells/cutcell_solver.f90),
[production scheme selection](../braids_qgis/gui/dock.py).
The [later inlet/expansion correction](../validation/sediment/delft3d_intercomparison/expansion_flume/diagnostics/inlet_audit/INLET_AND_AGGRADATION_AUDIT.md)
and aligned Zhu reruns do not close arbitrary-angle accuracy.

**Thus the old 11.69% is a historical failure, not a measured error for alpha 31.**
The 5% limit is our frozen acceptance target, not a mathematical definition of
a coding bug. A small absolute error can still fail a demanding local-change
target; conversely, a good-looking full-bed profile can hide inaccurate transport.
The earlier review itself contained no new runs. The linked current-source
batch now separates unresolved steep fronts, wall-local transport error and
the confirmed inlet reconstruction problem.

### First bounded diagnostic batch

- [x] Freeze the current executable/source hashes, production scheme 5,
  hydraulic state, formula parameters, boundary traces and reference definition.
  Retain the old results separately. Disable AMR initially to isolate orientation.
- [x] Reuse the inexpensive initial Exner-rate probe on current code at the
  identified steep face and on a smooth, adequately resolved profile. Record
  cell averages, reconstructed capacity, accepted face flux and flux divergence.
  Do not reconstruct the old operator only in Python and label it current-native proof.
- [x] Run small matched 0° and 45° cases; include a non-special angle such as
  30° in the compact follow-up. Keep physical length, width, forcing and time
  fixed. Integrate references over actual retained polygons; report along-flow
  resolution as well as nominal Cartesian cell size.
- [x] Start with fixed hydraulic fields to isolate sediment transport. Use a
  translating smooth suspended pulse with a known reference, and bedload/Exner
  with independently integrated face capacities. Then allow hydraulic feedback
  in the same cases. Do not alter the hydrodynamic equations to fit sediment.
- [x] Check constant concentration carried by the accepted water flux, dry/wet
  positivity and class budgets. Audit units, porosity exactly once, vector
  rotation, normal signs, face length/area scaling, and shared-face cancellation.
  **The compact rotated case now closes wet/dry positivity and both material
  classes. Constant-concentration advection remains covered by the earlier
  isolated transport oracle rather than this nonuniform two-pulse case.**
- [x] Separate interior full cells, wall-adjacent full cells and actual cut
  cells. Retain all cells in the overall results. The previous worst errors
  were not confined to cuts or inlet cells.
- [x] Measure concentration L1/L∞, peak loss, centroid shift, spreading,
  overshoot/undershoot, and transverse transport. For the bed, report absolute
  error, error in **bed change**, phase error and spurious erosion/deposition.
  Bed-profile error alone is not a measurement of numerical diffusivity.
  **Pulse norms and raw bed-change/rate errors are recorded. The resolved
  300 s Zhu batch adds peak, centroid, variance and absolute-error histories;
  it does not establish full-event or wet/dry diffusion.**
- [x] Use two resolutions first, then a third only if the trend is ambiguous.
  Halve the timestep for the worst case once to separate time and space errors.
  Report raw fine-cell errors alongside common-volume comparisons; do not
  smooth, shift or average away the local defect.
  **Two grids plus a bounded sub-metre steep-front window; exact polygon
  references and physical integrals retain raw local errors. The smaller
  actual timestep changes final pulse L1 by only 0.005504%. The resolved Zhu
  batch now includes raw fine-cell and common-parent-volume comparisons at
  300 s. The compact wet/dry case adds 0.28/0.14/0.07 m grids and a half-timestep
  0.14 m pair, retaining the failed raw local bank comparison.**
- [x] Run routine tests with eight threads. Repeat the selected coupled case
  with one thread and a restart; require identical states/ledgers on the same
  executable/platform. Put a short wall-time limit on each worker and report
  partial results honestly instead of extending runs for hours.

### Decision and correction gate

- [x] Classify the result explicitly: implementation defect; bounded orientation
  error at adequate resolution; underresolution; reference mismatch; or unresolved.
  A sign/unit/conservation error is a defect. A repeatable failure to converge
  in the resolved smooth test is a numerical problem requiring explanation.
  Two underresolved grids and one local maximum cannot settle that diagnosis.
- [x] Correct the localized inlet capacity reconstruction, recover the affine
  Exner rate at two resolutions with schemes 5/6, retain prescribed supply,
  equilibrium inlet and owner-capacity outlet contracts, and keep the previous
  seven-formula inlet-spike/variation gates unchanged. The fix spans the full
  five-cell boundary stencil radius; hydraulic equations and BCs are unchanged.
  See the alpha 33 correction report linked above.
- [x] Assess whether another numerical change is needed for the resolved
  fully wet Zhu case: both fine orientations pass without another native
  change. Future demonstrated failures may warrant
  consistent multidimensional reconstruction/face integration and suitable
  bounded treatment of smooth extrema. Retain positive inventories, shared
  conservative fluxes and low diffusion; do not switch globally to first order
  or smooth the bed to conceal a spike.
- [x] Rerun the retained Zhu-derived 300 s local-change gate on the current
  implementation at adequate resolution. Both fine orientations pass with
  the original 5% integrated and maximum-change limits; retain coarse failures
  and reference uncertainty separately. Scope and figures are linked above.
- [x] Run a compact rotated wet/dry multi-fraction transport and
  bank-evolution case. Separate suspended advection, bedload and bank exchange
  only as needed to locate a discrepancy; retain per-class conservation,
  positivity and one/eight-thread identity with several snapshots.
- [x] Replace the mesh-scaled face-local bank relaxation with conservative
  isotropic nonlinear diffusion using `D=L²/(2T)`. Preserve the corrected
  two-dimensional slope magnitude, per-class donor limits, bedrock limit,
  one-dimensional oracle, positivity and exact thread/restart behavior. The
  alpha 35 integrated response passes rotation and grid-convergence gates.
- [x] Reconstruct bank wetness inside each control volume using its conserved
  water volume and integrate repose activation along the physical face. Verify
  planar wet/dry face rates analytically, preserve class budgets and exact
  one/eight-thread/restart behavior, and repeat the unchanged compact gate.
  Development result: 47.85% raw discrepancy at 0.14 m, 25.99% at 0.07 m;
  the raw 15% and aligned integrated-refinement 2% gates remain failed.
- [x] Assess the bank toe against an independently resolved source reference
  and inspect/execute the supplied Delft3D lateral-erosion kernel. Source
  accuracy converges to 1.88%/1.78%, with 13.98% raw orientation discrepancy
  at 0.035 m; no additional native bank correction is supported by this test.
- [x] Correct the dry portion of a prescribed-stage outlet, retain admissible
  wet-outlet behavior and the interior SWE equations, and repeat the compact
  coupled test with eight threads, restart and one/eight-thread identity.
  The 2% integrated-refinement gate passes at 0.035 m in the separate 4 m strip.
- [x] Isolate the shallow transverse response with still-water and matched
  steady moving-flow controls plus actual native face-flux attribution.
- [x] Correct dry-owner and local-reversal stage-outlet stalls. Close local
  bank orientation at 1.75 cm (10.205% vs 15%), retain integrated refinement,
  water/class conservation, exact restart and one/eight-thread continuation.
- [x] Correct hydraulic wet/dry face reconstruction and pressure balance;
  steady diagonal crossflow decreases by 93.5%, with still-water, fully wet
  flume, inlet-spike, conservation and full one/eight-thread/restart checks.
- [x] Isolate the inlet startup trigger with identical initial fields and
  original/matched-profile controls; verify actual native sectional weights.
  The matched coupled case passes 0.05 m/s and fine orientation. Keep the
  original stress-input failure and the unverified transient amplitude explicit.
- [x] Establish compatible initial conditions with the existing
  QGIS section inlet, then add AMR to a compact rotated sediment/bed front.
  Compare against a uniform grid at the same finest spacing. Isolate extra
  error from refinement/coarsening, coarse/fine synchronization and subcycling.
  The 2 × 2 m case passes all 37 checks at 2 cm finest spacing; no native
  change was needed. Its erroneous exploratory rotated edge assignments are
  retained and explicitly excluded from solver evidence.
- [x] Complete a bounded reservoir/QGIS acceptance window with
  DEM-sampled inlet faces and hydraulic, sediment and adaptive-mesh snapshots.
  The 240 × 160 m window passes 33/33 checks at 2.5 m finest spacing through
  120 s; no production patch was needed. Full-reservoir convergence remains open.
- [x] Measure bounded complete-resource-domain resolution sensitivity and
  representative AMR performance within one hour. The 25 s comparison passes
  all budgets; AMR depth passes at 1.533% and runs 1.77× faster. Sediment has
  not reached the lake, while whole-domain sediment diagnostics remain
  resolution-sensitive, so longer reservoir sedimentation is not qualified.
- [x] Diagnose the near-inlet prescribed-bedload peak on the exact source edge
  with a compact four-grid case, hydraulic-only/passive controls, cutoff
  controls and five isolated allocation candidates. The hard cell-wise
  transport-depth switch is the local numerical defect.
- [x] Replace that switch with the qualified C1 shallow-wetting capacity
  transition. Retain exact prescribed loading, conservation, hydraulic and
  fully wet identity, inlet-spike, one/eight-thread and restart gates. A local
  capacity cap was rejected because it changed the prescribed sediment input.
- [x] Repeat short full-resource-domain resolution and representative AMR
  sensitivity with the corrected production solver. The inlet correction passes
  at 2.5/1.25 m and on 20→2.5 m AMR; hydraulics pass, young sediment maps remain
  resolution-sensitive and no sediment reaches the lake at 1 s.
- [x] Reduce exact finest-hierarchy preparation enough to run the 10→1.25 m
  same-finest comparison, diagnose and correct the fine-substep phase-reflux
  rejection loop, and measure the remaining cost. The case now completes;
  native subcycling is the dominant runtime cost.
- [x] Isolate the AMR-specific near-inlet morphology peak in a compact wet/dry
  case. Retain admissible level-2 parents where a finest cut fragment cannot be
  merged inside its AMR partition; the corrected combined peak passes by a
  factor of 86 against the 0.1 mm limit.
- [x] Integrate prescribed inlet water, suspended load and bedload exactly when
  a boundary spans asynchronously subcycled AMR levels. Synchronize moving
  wet/dry-front steps and pass the unchanged 2% gate against the
  maximum-admissible realization of the same hierarchy; retain independent cut-cell topology as
  a separate resolution sensitivity.
- [x] Repeat the full 10→1.25 m same-finest reservoir gate with corrected
  wet-front integration. All retained gates pass; AMR is 4.01× faster in the
  native solve, while preparation makes the one-second workflow 7.28% slower.
- [x] Stream and compact the multi-gigabyte geometry/hierarchy products while
  preserving exact preview, run and provenance behavior. Preparation falls
  16.8%, retained output falls 67.4%, and native input remains byte-identical.
  [Evidence](../validation/amr/full_reservoir/preparation_optimized_benchmark/REPORT.md).
- [x] Measure a representative fully wet AMR workload before deciding whether
  level-indexed recursive work lists and reusable Runge–Kutta storage are
  worthwhile. The compact coupled gate demonstrates 1.78× level-scan
  amplification, 51.0% subcycling slowdown versus synchronous AMR and 3.87×
  peak memory; both optimizations are justified.
- [x] Implement deterministic level-indexed cell/face work lists and reusable
  recursion-depth Runge–Kutta, flux-register and dense-interpolation storage.
  Repeat the same compact gate without changing equations, update order or
  tolerances; retain exact thread/restart identity and conservation. Eight-
  thread time falls 15.1%, peak RSS falls 29.6%, and pre/post frames are
  byte-identical.
- [x] Run a longer bounded reservoir-deposition sensitivity with inlet-corridor
  and reservoir metrics separated. At 300 s, AMR passes the unchanged depth,
  suspended and bed-change limits and is 1.091× faster than uniform 2.5 m.
- [x] Continue the exact uniform and AMR states to 600 s. Sediment reaches and
  deposits beyond 60 m, but the unchanged 8% suspended-field gate fails from
  360 s; final fine/coarse errors are 27.90%/46.09%. Conservation and inlet
  forcing remain exact, so this is retained as an AMR flow-routing failure.
- [ ] Correct the reservoir-scale AMR refinement/coarsening policy before
  expanding the domain. Add velocity-vector accuracy to the gate, resolve the
  approach flow before sediment arrives, and repeat the unchanged 600 s case.
  Require both the original accuracy limits and useful mesh/runtime reduction.
- [x] Resolve finest AMR-parent-corner cut slivers by retaining an admissible
  level-2 parent; do not merge across parent partitions. Lower-level physical
  boundary fragments remain subject to normal cut-cell handling.
- [ ] Extend deposition far enough into the complete reservoir to qualify lake
  storage, sorting and capacity change. Preserve separate inlet, reservoir and
  deep-reservoir metrics and avoid the abandoned 19–21 hour campaign.
- [x] Package the qualified development build as BRAIDS alpha 37, verify the
  installed-profile workflow with eight threads, and publish it for manual
  alpha acceptance.

**Exit:** a current-source report with actual profiles/maps at several times,
conservation, thread/restart identity, a stated resolution range, and a defensible
classification. If a limit remains failed, leave it failed and state its scope.
The fully wet evolved Zhu portion meets this exit condition in its stated scope.
The compact wet/dry case meets it for suspended transport, composition,
conservation, local bank orientation and integrated refinement at the stated
fine spacing. The original transient stress input still fails the shallow
speed watchdog; its inlet-profile mismatch is now isolated. The matched
diagnostic passes that limit, without changing production code. The subsequent
controlled AMR comparison now passes within the new small flume's scope;
  the original transient and large reservoir are not thereby fully qualified.

## 2. Finish alpha workflow acceptance

- [x] Add a line drawing button beside domain creation. Save editable inlet
  guides in the project folder, give each guide an ID, and allow snapping.
- [x] Hide/deselect guides after successful domain creation, when entering
  BC & IC, when creating boundary faces and when selecting BC faces. Exclude
  guides from the boundary-face menu; keep their geometry available in QGIS.
- [x] Restore retryable actions after input/task/launch failures (alpha 30).
- [ ] User acceptance of guidance drawing, cancellation, shared-node snapping,
  imported domains and project reopening with the actual working DEM.
- [ ] Complete a short end-to-end run with multiple inlet IDs and an outlet
  spanning selected faces, then save/reopen/restart and animate raster and mesh.
  Verify that hidden guides never become hydraulic boundaries.
  The closed reservoir window now passes the two-inlet, restart, reopen and
  raster/mesh portions in actual offscreen QGIS; the grouped outlet remains open.
- [ ] Exercise the actual workstation workflow after deleting/reloading result
  layers and selecting different completed runs. Fix reported failures without
  changing numerical settings merely to bypass them.
  Automated current-source QGIS checks now pass deleted-raster recovery and
  coexistence of results from two runs; installed-alpha manual acceptance remains.

## 3. Qualify the real reservoir and channel-migration use cases

The [retained 600 s fixed-grid reservoir result](../validation/sediment/reservoir_polygon/geometry_resolution/closed_face_reservoir/reports/RESERVOIR_RESULT.md)
passes inlet and conservation checks but remains spatially unqualified. Between
5 and 2.5 m, lake suspension differs by 9.53%, positive bed change by 16.61%, net
bed gain by 24.25% and the positive-deposit map by 26.65%. These are historical
measurements, not a new alpha 31 AMR comparison. They explain why local deposit
distribution and terrain resolution remain on this list even though water and
sediment already reach the lake successfully.

- [ ] Prepare a bounded real-reservoir window with the same DEM, domain,
  water-body level, Manning, material fields and forcing in all comparisons.
  Start with water only, then suspended-only, bedload-only and combined load
  where needed to isolate a demonstrated discrepancy, not an exhaustive matrix.
- [ ] Compare AMR with a uniform grid at the same finest spacing; add one finer
  comparison if resolution remains inadequate. Confirm connected stage–storage,
  wall/escape-crest heights and inlet-section representation across meshes.
- [ ] Check cumulative accepted per-ID water and sediment loads against domain
  storage; separate gross deposition, erosion and net accumulation per fraction.
  Confirm sediment arrival and deposition next to the reservoir wall.
- [ ] Report local deposit thickness/maps, downstream fines, coarse delta
  advance, grain sorting and capacity loss. Stable total sediment mass alone
  does not establish a correct deposit location or a resolved local maximum.
- [ ] Repeat one short, physically matched Delft3D reservoir comparison at
  adequate resolution. Keep the older 32.1% deposit discrepancy and failed
  limits as historical evidence; matching Delft3D is not an independent truth test.
- [ ] Verify lateral bank retreat in a compact mobile-channel case with
  wet/dry refinement and slope failure. Check conservation of eroded bank
  material, evolving grain composition and sensitivity to grid angle/spacing.
  Do not infer a correct braiding rate from one plausible-looking planform.
- [ ] Document/qualify grouped outlet restrictions, especially disconnected or
  noncollinear faces requiring a section-control definition. Use explicit IDs
  where conditions are physically independent; retain one face per inlet.

## 4. Finish the applicable sediment capability inventory

The [source inventory](../DELFT3D_SEDIMENT_INVENTORY.md) is a historical formula
audit; its old implementation-status columns need reconciliation with the later
seven-law, composition, stratigraphy and AMR releases. No new formula is needed
merely to investigate the current orientation error.

- [ ] Update that inventory by enabled dispatch branch, required state, units,
  applicability, implementation, formula oracle, coupled test and intercomparison.
  Keep total-load formulas distinct from separate suspended/bedload closures
  to prevent counting the same transported sediment twice.
- [ ] Add the missing applicable current/total-load options in a declared order:
  Engelund–Hansen, modified Van Rijn 1984, configurable general relations, and
  the remaining applicable Soulsby–Van Rijn/Van Rijn 2004 branches. Audit each
  branch's inputs before exposing it; do not invent missing wave or mixing state.
- [ ] Assess remaining hiding/exposure and mixed-bed options for the intended
  materials. Existing Wilcock mixture statistics and Egiazaroff support are
  implemented; additional laws need their own applicability and coupling checks.
- [ ] Add specialist cohesive processes with explicit state and timescales:
  flocculation, near-bed fluff storage/burial, consolidation and age-dependent
  erosion resistance where required. Preserve per-class mass and bed volume.
- [ ] Keep wave-current/SANTOSS and related wave-dependent branches as separate
  capabilities requiring wave forcing. Distinguish depth-averaged transport
  from mechanisms needing vertical resolution. Wind forcing is a later input
  feature; turbulence work remains paused.
- [ ] For each added capability, supply QGIS controls, saved/restart identity,
  conservative limiting, a small independent formula/coupling test and one/eight
  thread qualification. Use representative compatible combinations; do not
  claim every possible combination was tested by the existing eleven-case matrix.

## 5. Improve AMR cost and qualify performance

- [x] Profile a bounded complete-polygon four-level preparation/startup case
  and remove avoidable Cartesian DEM integration, geometry-intersection,
  repeated-metric, serialization and checkpoint-state costs. Preparation falls
  45.15%, geometry output 42.64%, checkpoint size 89.44% and startup RSS 61.37%,
  with equivalent geometry and exact native/thread/restart identity.
- [x] Measure preparation time, peak memory, active/candidate cell counts,
  native solve time, output cost and timestep restrictions separately on a
  representative reservoir. The 10→1.25 m comparison identifies recursive
  native subcycling as the main wall-time cost and records the limiting cut
  fragment. A representative mobile-channel performance measurement remains
  part of later channel qualification.
- [x] Measure a compact fully wet coupled workload with genuine L1/L2/L3
  subcycling. Eight threads give 2.14× speedup with byte-identical output;
  synchronous AMR is 1.20× faster than uniform, while recursive subcycling is
  51.0% slower than synchronous and uses 3.87× its peak memory. This qualifies
  the need for indexed level work and reusable stage/flux storage.
- [x] Implement and qualify deterministic level-indexed work lists plus reused
  recursive stage/flux storage. Eight-thread subcycling is 15.1% faster and
  uses 29.6% less peak memory on the unchanged gate. Pre/post frames, one/eight-
  thread results and restart trajectories remain exact; all budgets pass.
- [ ] Address the current preparation-memory limitation: the hierarchy prebuilds
  finest candidate geometry and retains fine terrain/bed detail. Investigate
  lazy/local candidate construction and conservative storage without discarding
  shoreline/obstacle detail or buried sediment when cells coarsen.
- [ ] Profile the cells/faces limiting the timestep, especially unmergeable
  clipped fragments at obstacles. Change stabilization only with conserved
  geometry/fluxes and numerical checks; do not enlarge their area artificially.
- [ ] Measure useful one/eight-thread speedup on workloads large enough to enter
  parallel loops. Retain deterministic ordering, exact restart and thread
  equality after optimizations. State when AMR is slower than a fixed grid.
- [ ] Qualify refinement thresholds and buffering for small waves, rapidly moving
  wet/dry fronts and morphology; test that coarsening does not erase bank/bed
  detail. Include actual animated mesh snapshots in the result record.
- [x] Protect prospective-parent bed detail during coarsening, with a planar-
  slope/hidden-relief oracle and an 8 × 4 m saddle before/after comparison.
- [x] Fix unallocated subcycling workspaces when initial AMR selection retains
  the entire finest mesh; check one/eight-thread and restart identity.
- [x] Reproduce an evolving inlet crest in a small two-channel case; record
  accepted RK-stage geometry and face fluxes, verify section integration,
  conservation, temporal accuracy, exact restart and one/eight-thread identity.
  The chosen connected-section implementation passes this compact test; its
  suitability for multi-channel inlets is addressed by the next control.
- [x] Diagnose the reservoir sensitivity in a short 480–540 s saved-state
  window. Same-state timestep refinement passes, but different saved histories
  reproduce the failure. Supplying every submerged inlet portion reduces
  maximum coarse-suspension/velocity differences from 15.52%/25.03% to
  1.49%/4.30%. Retain both comparisons and the supplied-source boundary review.
- [x] Correct native and Python inlet wet-portion selection consistently,
  including stage/depth/normal/critical section calculations and QGIS previews.
  Keep horizontal stage, local dry closure, per-ID flux conservation and initial
  lake-polygon connectivity distinct. Analytical section, inlet-spike,
  thread/restart and short reservoir controls pass. The stationary critical-rate
  search also removes dependence of the selected energy minimum on collinear
  section sampling. See the production correction report above.
- [x] Requalify bounded reservoir AMR accuracy and useful runtime savings with
  the corrected inlet and a temporally resolved reference. The original late
  AMR failure remains retained as historical evidence. The common-state 300 s
  case passes every field, conservation, forcing and thread gate; speedup is a
  marginal 1.055× because 96.3% of final leaves remain finest-level.
- [x] Instrument each AMR refinement trigger and remove unnecessary finest-level
  coverage while retaining the qualified reservoir fields, wet/dry banks,
  solid-wall resolution and abrupt submerged-bed-change detection. Static
  features keep a geometric stencil but no longer receive advective travel
  distance. The prospective common-state gate passes with 67.9% finest-area
  coverage, 4,428 active leaves and 1.255× speedup over fixed 2.5 m.

## 6. Complete results, portability and distribution

- [ ] Add user-facing reservoir stage–storage/capacity-loss products, trapping
  efficiency, retained/exported sediment per fraction and delta-position metrics.
  Define dry-cell masks and gross versus net deposition consistently.
- [ ] Make complete BRAIDS projects portable: relative references or an explicit
  copy-data operation for DEM, polygons, guides, forcing and material inputs.
  Detect missing files and support relinking after the folder is moved.
- [ ] Complete manual acceptance of mesh/raster overlays around curved obstacles,
  multi-run animation, units and colour scales. Display pixels must not imply
  additional numerical resolution beyond the native control volumes.
- [ ] Package and qualify other selected operating systems/architectures, with
  bundled native dependencies, QGIS compatibility and short installation/run tests.

## 7. Experimental validation comes last

- [ ] Select documented sediment experiments covering reservoir settling,
  expansion/deposition, sorting and bank migration; record measured inputs and
  uncertainties before comparing results.
- [ ] Freeze mass, thickness-map, profile, delta/bank-position and grain-size
  metrics before running. Separate numerical convergence, uncertain physical
  parameters and measurement error; report calibration and independent tests
  separately.
- [ ] Compare both BRAIDS and Delft3D with those observations where applicable.
  Agreement between the two solvers is intercomparison, not experimental
  validation. Zhu/theoretical checks remain verification during development.

Fixed reflective solid obstacles, including bridge-pillar footprints, remain
the internal-structure scope. No additional bridge-flow model is scheduled here.

## Immediate sequence

1. Add user-facing reservoir capacity, trapping-efficiency, retained/exported
   sediment per fraction and delta-position products, with explicit dry masks
   and gross/net deposition definitions.
2. When packaging resumes, package the qualified development solver as the next
   alpha and complete the manual guide, snapping, imported-domain, grouped-outlet
   and reopen workflow.
3. Make projects portable, then qualify additional operating-system packages.
4. Complete applicable specialist sediment processes and mobile-channel
   qualification; measured experimental validation remains the final stage.
