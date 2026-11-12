# BRAIDS Desktop local preview

## Unified local BRAIDS

The normal BRAIDS desktop launcher and the BRAIDS shortcut now open the same
qualified installed runtime. Hydrodynamics, the hydrokinetic-turbine extension
and sediment transport are available in the same application, subject to the
existing method and physics compatibility checks. The installed directory
retains its historical `BRAIDS-HKT` name so its existing Windows uninstaller
continues to target the correct files; it is not a second runtime. The separate
source checkout and past results are retained. No installer or public release
was created. See [unified152](../validation/sediment/bedload_validation/reports/unified152/REPORT.md).

## Local Dou193: aligned and yawed turbine power

Hydrokinetic turbines now offers **Dou (2019): aligned / yawed array power**
under Shared power estimator. Import matching device yaw-performance JSON,
provide zero-yaw Ct/Cp curves and independently supported wake width/growth,
then use Fixed rotor axis and the yaw control for fixed turbines. The imported
table is embedded in the project. Existing turbine projects keep their original
curve convention; turbine momentum forces remain active in the flow.

Desktop/BRAIDS_Dou193 contains two example projects with short saved results.
Prepare before a new run. Experimental yawed hydrokinetic array-power validation
is deferred; these are numerical demonstrations. See the
[delivery report](../validation/hydrokinetic_turbines/reports/delivery193/REPORT.md).
This is a local app update, without a new installer or public release.

## Local Espir138: secondary-flow sediment direction

The Advanced hydrodynamics tab now saves a FLOW sediment-direction factor
Espir. Its default is 1; the Mezied bar case uses 0.5. CPU/GPU source,
coupled, restart and portable-runtime checks are recorded in
[Espir138](../validation/sediment/bedload_validation/reports/espir138/REPORT.md).
Capacity-owned direction models explicitly reject a nondefault value rather
than ignoring it. The schema is now v41; older projects are rejected under the
existing no-backward-compatibility policy. Ashida bar validation remains open.
The local app is updated; no installer or GitHub release was created.

## Local121: Recking (2010) mathematical verification

Recking2010 Eq.12 is admitted with bounded HLLC bedload on CPU/GPU. Independent
80-digit arithmetic verifies2232 inputs and both one-sided branch derivatives;
maximum capacity/derivative relative error is below1.4e-14. Current-cell
resistance binding is repaired and the scalar equation remains unchanged.
Enter D84 and D50 independently and a representative reach slope in m/m.
The UI identifies the local-stress adaptation and derivative kink explicitly.
See [Recking121 and figures](../validation/sediment/bedload_validation/reports/recking121/REPORT.md).
Local update only: no installer or GitHub publication. Next: Camenen-Larson2005.

## Local120: Ancey & Recking mathematical verification

Ancey & Recking2023 Eq.42 is admitted with bounded HLLC bedload on CPU/GPU.
Independent80-digit arithmetic matches the native capacity and derivative to
8e-14 across1211 stress values. The current-cell resistance binding is repaired;
the published scalar law is unchanged. Explicit surfaceD50, static finite bulk
bed and MF1 are required; extra physics remain unavailable in this HLLC route.
The user requested mathematical verification without experimental revalidation.
See [Ancey120](../validation/sediment/bedload_validation/reports/ancey120/REPORT.md).
This is a local app update, with no new installer or GitHub publication.

## Local119: accepted Smart changes installed

The installed app and development runtime contain the qualified dry-owner
outlet correction and explicit Smart-Jaggi0.047 HLLC option. The0.05 default
remains unchanged. Installed/development CPU and GPU runs match the qualified
states exactly; the installed UI passes launch and parameter-editor checks.
No installer or GitHub upload was made.

The user accepts the current Smart behavior; its experimental bias remains
recorded. Continue with Ancey-Recking2023, Recking2010, Camenen-Larson2005 and
FLOW Van Rijn1993. See the [remaining validation list](../validation/sediment/bedload_validation/REMAINING_VALIDATION.md)
and [delivery report](../validation/sediment/bedload_validation/reports/delivery119/REPORT.md).

## Smart116: experimental rates and author calculation traced

The Schwindt non-constricted flume is selected. Source replay reproduces the
saved/thesis curves, while exposing distinct grain, bank and normalization
choices. Actual CPU/GPU rates agree at0.05 and the source-justified0.047.
The latter improves conditional experimental bias but does not close the gap.
Figures and precise scope are in
`validation/sediment/bedload_validation/reports/smart116/REPORT.md`.
Next qualify0.047 coupling and the forward2D traction mapping. Installed
local111 and its0.05 preset are unchanged; CFBS remains deferred.

## Smart115: search for the existing Smart-Jaeggi law

The next work seeks experimental validation of the public graded4/.05 law.
CFBS/B-1 work is deferred and preserved as research. The Smart115 source audit
identifies Schwindt's open data as a candidate, with threshold/geometry differences
still to resolve before an exact-match claim. Installed local111 is unchanged.
See `validation/sediment/bedload_validation/reports/smart115/REPORT.md`.

## Smart114: named CFBS research implementation, Desktop unchanged

CFBS now has paired native operators and a short16s HLLC comparison.65 checks,
conservation and exact restart/cancellation/rollback pass. The early crest
erodes faster than the experiment; physical reproduction remains open. This
is not a selectable Desktop formula or validation of practical Smart4/.05.
Figures, numerical controls and the next crest-face audit are in
`validation/sediment/bedload_validation/reports/smart114/REPORT.md`.
Local111 remains installed; no installer or publication.


## Smart113: evolving-bed benchmark locked, runtime unchanged

The B-1 overtopping-dike specification, grain descriptors and published2014
experimental/model curves are locked. Smart-CFBS differs from current Smart
and needs a distinct implementation; no new solver option or validation pass
is claimed.23 source/data/scope checks pass. Figures and coupling gaps:
`validation/sediment/bedload_validation/reports/smart113/REPORT.md`.
Local111 remains installed; no installer or public release.


## Smart112: later experimental rate evidence, runtime unchanged

Five later Smart-Jaeggi measurements give a median prediction/feed ratio1.0405,
but retain a41.1% excess in one run. General physical validation remains open;
no transport coefficient is fitted. Source/data classification, fixed-input
native CPU/GPU calculations and unchanged external2D operator checks are in
`validation/sediment/bedload_validation/reports/smart112/REPORT.md`.
Local111 remains installed; no new installer or public release.


## Local111: Smart & Jäggi with HLLC

The fixed graded4/.05 Smart & Jäggi preset is qualified on CPU/GPU for a
static finite mixed bed, one bulk material with explicit Dm/D30/D90, and MF1.
Current cell-depth physical resistance is shared with hydraulic drag. Exact
restart/cancellation, finite stock, wet/dry and cropped-face controls pass;
independent experimental morphology remains unqualified. Other feature
combinations retain explicit integration errors.

Open Desktop/BRAIDS_SmartJaggi_HLLC111 for the self-contained numerical example
and its completed2s preview. Comparison figures and retained limitations are
in `validation/sediment/bedload_validation/reports/smart111/REPORT.md`.
Local installed/development update only; no new installer or public release.


## Smart110: validation evidence, no Desktop update

Smart & Jäggi's measured Dm/D30/D90 source-data audit is complete. The current
implementation agrees with the exact graded BASEHPC4.2 law. Source-data fit
and remaining independent/coupled-validation gaps are reported with figures in
`validation/sediment/bedload_validation/reports/smart110/REPORT.md`.
The installed runtime remains local109; no new HLLC option or release.

## Local109: stage boundary admits incoming backwater bores

The common CPU/GPU stage condition now admits a finite incoming hydraulic jump
even beside supercritical outflow. The correction preserves hydraulic cores
and sediment formulas. Boundary, coupled regression and exact restart/cancel
checks pass. Figures and scope are in
`validation/sediment/bedload_validation/reports/outlet109/REPORT.md`.
No new installer or release is created.

## Local108: corrected numerical bed jump

The existing bedload path now reconstructs the numerical bed jump at the true
shared face on CPU/CUDA. Physical transport formulas and hydraulic methods
are unchanged. Stock, wet/dry, conservation and restart/cancellation controls
pass. This fixes an independently demonstrated planar-bed error; the corrected
10h experimental prediction still has a12.97mm bed RMS error and is not a
complete physical-validation pass. See the comparison figures and retained
negative results in `validation/sediment/bedload_validation/reports/native108`.

The near-critical degradation case also exposes a pre-existing HLLC/f-wave
outlet contrast before sediment starts. Its physical comparison uses HLLC;
the short same-state hydraulic audit is next. Exact checkpoints from the old
numerical runtime are rejected. No installer or public release is created.

## Local104: MPM — BRAIDS standard

Fresh sediment setups select `capacity.mpm_braids_5_055`: fixed5/.055/1.5,
one measured D50 and physical cell-depth hydraulic resistance. Sediment is
still unchecked initially. FLOW MPM and Ashida keep their original identities.
The bounded current admission is static, finite mixed bed, MF1, bedload only;
other feature combinations produce explicit qualification errors.

The public model reproduces the matched Saiedi control. Its independent gravel
rate check improves the median prediction/feed ratio from2.264 to1.387 but
retains22–67% overprediction without refitting. See the reports and figures in
`validation/sediment/bedload_validation/reports/standard104` and `gravel105`.
Local example: Desktop/BRAIDS_Standard_Bedload104. No new installer or release.

## Matched103: explain the deposition discrepancy

With the same calibrated MPM-like law and physical Manning shear, BRAIDS and
unmodified BASEHPC4.2 closely reproduce the short Saiedi deposition control:
front2.865 versus2.897m, peak39.324 versus39.361mm. Matching the constitutive
setup removes95% of the earlier cross-law front gap. No production correction
is justified by this result. The private driver is not a public formula or
Desktop admission; installed local100 remains unchanged. Figures, scope and
the remaining independent physical-validation work are recorded in
`validation/sediment/bedload_validation/reports/matched103/REPORT.md`.

## Forcing102: source audit and coarse deposition diagnostic

The complete uniform FLOW forcing chain reproduces the installed native
CPU/GPU capacities; it does not explain the remaining experimental rate bias.
A new30-minute Saiedi2mm deposition test with default Ashida advances too far
against the experiment and the published calibrated MPM reference. Constant
Manning follows the reference's setup. This is a cross-closure diagnostic,
not a physical pass or an admission change. See
`validation/sediment/bedload_validation/reports/forcing102/REPORT.md` for figures,
raw polygons, parameter identities and the next matched-law control.
Installed local100 remains unchanged.

## Physical validation status: scope101

Numerical HLLC admission is not complete physical validation of a transport
formula. The current formula-specific inventory and active plan are in
`validation/sediment/bedload_validation/PLAN.md` and `MATRIX.json`. Soni is
retained as a fine-sand calibrated regression. New formulas are deferred while
appropriate gravel/sand rate and same-closure2D physical evidence is established.
The first uncalibrated gravel capacity screen and its limitations are in
`validation/sediment/bedload_validation/reports/scope101/REPORT.md`.
That report includes an actual unmodified BASEMENT4.2 capacity replay;
its standard MPM differs from the FLOW variant and this is not a complete
physical flume reproduction.
This validation work changes no solver, installation or method admission.

## Local100: Wang-Fredsoe audit, HLLC admission withheld

The literal FLOW legacy bedload kernel passes CPU/GPU source, shear, viscosity,
grain and derivative checks. Its coupled Soni comparison fails the retained
HLLC/f-wave screen, so it remains unavailable with HLLC. MPM, Wong-Parker and
the qualified FLOW Ashida preset remain the public HLLC bedload choices.

The existing f-wave option now identifies the legacy variant, exposes physical
water viscosity and rejects invalid coarse shallow grain friction. Required
D50/D90 are explicit; its signed native suspended residual is not transported
in bedload-only runs.399 public tests pass; native libraries are unchanged.
Open `Desktop/BRAIDS_Wang_Audit100` for the completed negative experimental
comparison and figures. Report: validation/hydrokinetic_turbines/reports/
wang100/REPORT.md. Next: Smart & Jaggi source and steep-flume HLLC audit.
Local update only, no new installer or release.

## Local99: Ashida-Michiue (FLOW) with HLLC

Fast HLLC now admits the active FLOW Ashida-Michiue preset beside MPM and
Wong-Parker on CPU/GPU. Enter measured D50. ACal is the entire coefficient A,
standard17, with critical Shields.05 and powers1.5/1/1. Resolved setup prints
the actual coefficient and effective-shear convention. Static finite mixed
bulk bed, MF1 and other sediment/advanced-flow features off remain required.
Native formula and hydraulic libraries are unchanged.

354 regression tests,13 final UI checks, lifecycle and frozen controls pass.
The Soni experimental project includes both standard A17 and an explicitly
initial-rate-calibrated case. Their bed RMSEs are5.62mm and0.63mm; this is not
a universal predictive-validation claim. Open
`Desktop/BRAIDS_Ashida_HLLC_Validation99` and read its comparisons. Full report:
`validation/hydrokinetic_turbines/reports/ashida99/REPORT.md`.
Local runtime delivery only; no installer or release. Next: Wang-Fredsoe
bedload-only source and coupled qualification.

## Local98: corrected open-boundary bed continuation

MPM and Wong-Parker retain their physical formulas on CPU/GPU. The boundary
correction removes an artificial planar-bed jump from the sediment-wave trace
and evaluates the prepared exterior bed at the actual physical face. It also
fixes reference-density conversion when importing compatible bed stock.
325 tests, exact restart/cancellation checks and17 coupled regressions pass.
The Soni experimental profile remains within0.274mm of local97. Small residual
boundary adjustments are documented; long-time equilibrium is not claimed.

Open `BRAIDS_Bedload_Boundary_Validation98` on the Desktop for completed results,
root checkpoints and comparison figures. See
`validation/hydrokinetic_turbines/reports/boundary98/REPORT.md` for the source
trace, measured-feed contrast, longer-buffer tests and current scope. Existing
projects need fresh preparation under the new runtime. Local delivery only.
The next formula is Ashida-Michiue after its source and coupled audit.

## Local97: Wong-Parker bedload with HLLC

Wong & Parker (2006) now runs beside MPM with Fast HLLC on CPU or GPU. It
requires the explicit arithmetic mean Dm and uses local bed shear without
MPM form-drag reduction. The bounded scope remains one finite bulk material,
static mesh and MF1, with other sediment and advanced-flow features off.
Spatial reconstruction is second order; hydraulic/bed splitting is first
order in time. Incompatible choices give explicit preparation errors.

272 checks, short coupled controls and the original plane-bed rate comparison
are recorded in `validation/hydrokinetic_turbines/reports/wong97/REPORT.md`.
The Desktop project `BRAIDS_WongParker_HLLC_Validation97` is a documented
shear-matched realization, not a full narrow-flume reproduction. Read its
comparison report for experimental scatter and endpoint behavior. Delivery
is a local runtime update; no release or installer is created.

## Local MPM bedload with HLLC update 96

Fast HLLC now supports MPM bedload on CPU and GPU: one bulk material, a static
mesh, finite mixed bed, MF=1 and other sediment/advanced features off. The
existing conservative bed owner is retained. Hydraulic/bed time splitting is
first order, with second-order spatial reconstruction; Resolved setup explains
this. Other formulas and combinations still require f-wave.

253 targeted checks pass, with exact restart/cancellation and unchanged frozen
hydraulic, turbine and f-wave MPM controls. A reconstructed Soni aggradation
comparison gives 1.05 mm bed RMS error after a steady hydraulic spin-up, using
an explicitly calibrated initial transport rate. ACal=1 remains the software
default; both coefficient choices and input uncertainties are documented.
This is local delivery, without a new release, installer, upload or QGIS work.
[Audit, experimental comparison and figures](../validation/hydrokinetic_turbines/reports/bedload96/REPORT.md).

## Local HLLC advanced hydrodynamics update 95

Fast HLLC supports the current prognostic secondary flow, its momentum feedback
and constant horizontal viscosity on CPU and GPU, with optional turbines.
Choose the hydraulic method in Run, enable the desired Advanced hydrodynamics
controls and prepare again. With turbines, select one positive ambient mixing
input across the Advanced and Turbine panels; conflicting inputs give an error.
F-wave remains the default. HLLC sediment is not yet admitted.

A viscosity-only activation defect is corrected: the conservative viscosity
owner's coefficient is used even though its legacy coefficient is zero.
118 tests, 27 public controls and eight installed/development checks pass.
The transient bend matches f-wave within 2.17 mm maximum / 0.39 mm RMS depth;
matched advanced hydraulic loops are 1.24x faster on CPU and 1.56x on GPU.
Terrain settings remain unchanged. Local update only, without a new release.
[Report and comparison figures](../validation/hydrokinetic_turbines/reports/advanced95/REPORT.md).

## Local topographic preparation update 94

Fresh multilevel preparation is 3-11% faster than local93 in the tested
manual/topographic controls. The largest changes from 31.16 to 27.59 seconds;
the combined levee, breakline and turbine control changes from 11.27 to 10.39
seconds. Shared parent weights, exact rectangle checks and vectorized final
grading retain identical meshes, DEM integration, initial fields and results.
Cached preparation shows no consistent speed improvement.

195 tests pass, with one optional upstream raster oracle skipped. CPU/GPU,
f-wave/HLLC, MPM, turbine and restart controls preserve the frozen results.
Re-prepare after updating. Local delivery only; no installer or release.
[Audit and figures](../validation/hydrokinetic_turbines/reports/topography94/REPORT.md).
The audit also identifies a pre-existing grid-scale sensitivity of the terrain
detail threshold. Its defaults remain unchanged; qualifying a resolution-
consistent levee criterion is the next bounded task.

## Local turbine mesh preparation update 93

Turbine/permanent-wake preparation is 9–19% faster than local92 in the tested
single, tandem and staggered arrays at one and three ordinary mesh levels.
The largest staggered control changes from 13.29 to 10.73 seconds. River mesh
size, local D/2 resolution, 10D wakes, minimum areas and graded transitions are
unchanged. Batched geometry construction and reuse of exact rectangle proofs
reduce setup work without changing turbine flow or power calculations.

168 targeted tests pass. Prepared arrays/mappings and frozen/current flow,
power, thrust, Ct and Cp records are identical. Prepare again after updating.
This is local delivery only, without a release, installer or QGIS change.
[Audit and comparison figures](../validation/hydrokinetic_turbines/reports/turbine_mesh93/REPORT.md).
Next: profile and optimize topography-based preparation/refinement.

## Local breakline preparation update 92

Fresh preparation with oblique/crossing breaklines is 24–40% faster than local91
in the tested cases. At 34,932 cells it changes from 12.31 to 7.57 seconds; at
142,770 cells, from 45.57 to 27.55 seconds. The three-level control improves
from 16.05 to 10.83 seconds. Cached asset reuse is essentially unchanged.

Exact intersection filtering avoids unnecessary polygon cuts. Native queries
run in parallel; rectangle checks, face metadata and immutable area reuse reduce
repeated work. Complete prepared hierarchies and inputs are identical, and
frozen/current hydraulic and bedload final fields match exactly per backend.
Prepare again after updating. This is local delivery only, without a release
or installer. See [the audit and figures](../validation/hydrokinetic_turbines/reports/breaklines92/REPORT.md).
The next task is turbine/permanent-wake preparation at several mesh levels.

## Local preparation update 91

One-level preparation, without optional refinement, is faster with identical
geometry and initial fields. Fresh preparation takes 3.15 → 2.17 seconds at
8,837 cells, 10.19 → 5.64 seconds at 35,892 cells, and 42.53 → 22.68 seconds at
144,694 cells on the tested machine. This uses an exact-rectangle OpenMP check,
batched geometry and cheaper tree serialization. Irregular/cropped cells keep
their checks. Hydraulic methods, CFL and physical settings are unchanged.

Startup also verifies runtime-local hard-linked DLL aliases without repeatedly
hashing their identical bytes; all manifest entries are still checked. Prepare
your project again after updating. This is a local update, without a release,
installer or QGIS change. Figures, timings and next breakline milestone:
[preparation91](../validation/hydrokinetic_turbines/reports/preparation91/REPORT.md).

## Local hydraulic audit update 90

The CPU augmented Riemann files match stock GeoClaw 5.14.0; the CUDA port and
both full hydraulic backends pass matched-state checks. The complete BRAIDS
polygon update is a GeoClaw derivative. Fast HLLC retains a different spatial
and time scheme; local wet-bank differences are documented in the audit.

Critical/weir roots now stop at floating-point precision, and GPU f-wave batches
its stage status check. Additional paired hydraulic-loop speedups over local89
are 1.08x CPU / 1.11x GPU f-wave and 1.25x CPU / 1.33x GPU HLLC. CFL, physical
formulas and the f-wave default remain unchanged. Re-prepare before running.
See [the audit and comparison figures](../validation/hydrokinetic_turbines/reports/fwave_audit90/REPORT.md).
This is a local update, with no new release or installer.

## Local turbine sampling update 89

Both **Fast HLLC and augmented f-wave** now use the qualified GPU batched
turbine sampler. Coverage, thrust/work, rotor moments and source application
use fresh stage values, sparse interpolation and deterministic GPU reductions.
Physical curves, wake models, clearance rules and CFL settings are unchanged.

The 130,190-cell HLLC hydraulic benchmark is **2.97× faster than local88**.
The one/two/three-rotor GPU controls show 3.15–4.66× HLLC and 1.75–2.78×
f-wave gains. These exclude construction/output; CPU gains are modest.
Re-prepare your project; there is no new optimization switch. The installed
CPU/GPU launchers pass with both hydraulic methods. No release or installer.
See [sampling89](../validation/hydrokinetic_turbines/reports/sampling89/REPORT.md)
for the profiles, comparisons, scope and next milestone.

## Local hydraulic update 88

**Run → Hydraulic method → Fast HLLC (2nd order)** now supports the existing
hydrokinetic turbine extension on CPU and GPU. It retains the actuator,
clearance guards, speed/TSR curves, array inflow correction and turbine CSVs.
Its fully wet low-Froude pressure correction preserves the wake; sediment and
secondary flow still use the default augmented f-wave method.

In the 7,200-cell tandem benchmark, the optimized hydraulic loop is about
4.0× faster on CPU and 1.6× on GPU than the matched current f-wave loop.
The 30 s comparison differs by 0.124 mm maximum depth and 0.124% downstream
power. These are bounded hydraulic timings and comparisons, excluding mesh
generation and output. Re-prepare before running. Figures and scope:
[hllc_turbines88](../validation/hydrokinetic_turbines/reports/hllc_turbines88/REPORT.md).
This is a local update; no new installer or publication.

## Local hydraulic correction 87

Fast HLLC now reserves prescribed boundary withdrawals before limiting
internal fluxes. This removes false positivity rejections at a shallow inlet
in the user's natural-river test, without changing its CFL or discharge.
The copied run reaches 280 s with zero spatial or outer retries; shallow-front
differences and all checks are documented in
[wetting87](../validation/hydrokinetic_turbines/reports/wetting87/REPORT.md).
Re-prepare before running with the updated runtime. No public release was made.

## Local mesh update 86

**Mesh → Refinement controls → Refine inlets/outlets to smallest size** is now
optional. Checked preserves the previous behavior; unchecked lets open cells
follow the selected static or depth-AMR mesh strategy. Other explicit geometry
protection remains active. The f-wave route was checked with advanced
hydrodynamics, all current bedload/suspended choices and Jiang–Cao; Fast HLLC
was checked separately for hydrodynamics. See
[boundary_mesh86](../validation/hydrokinetic_turbines/reports/boundary_mesh86/REPORT.md)
for raw meshes, comparisons and the bounded qualification scope. Local update
only; no installer or publication.

## Local hydraulic update 85

The installed BRAIDS-HKT and local Desktop runtimes offer **Run → Hydraulic
method → Fast HLLC (2nd order)**, a MUSCL–Hancock hydraulic alternative on CPU
and GPU. The matched flowing-hump check measures 2.90× CPU / 6.03× GPU hydraulic
speedup at the same CFL, with 1.072 mm maximum depth difference after 10 s.
This initial option admits hydrodynamics and optional constant viscosity;
active turbines, secondary flow and sediment still require the default
GeoClaw augmented f-wave method. Unsupported combinations produce an error.
The saved numerical-method identity remains explicit. No new installer or
GitHub release accompanies this local update. Figures, scope and verification:
[fast_hydraulics85](../validation/hydrokinetic_turbines/reports/fast_hydraulics85/REPORT.md).

BRAIDS 0.1.0-preview63 is the standalone interface. It uses the qualified
explicit CPU/GPU solver with geometry setup reuse and the current bedload fixes,
without loading QGIS. The central map is adapted from
[AVAC Desktop](https://github.com/cgotelli/AVAC-Desktop/); its scientific engines
are not used. Sediment transport is an optional extension, unchecked by default.
Light is the default, explicitly applied to Qt independently of the Windows
color scheme. Dark remains an explicit Settings choice. The window has square
corners and only a 1 px neutral-gray border; invisible client-side edge/corner
handles preserve resizing without Windows' native thick frame.

## Power planning in preview63

Open one of the three `Open project.cmd` shortcuts in
`C:/Users/STREEM/Desktop/BRAIDS_Turbine_Power75`: single, tandem and staggered.
Each folder includes a DEM, editable project, saved ambient-flow run and power
estimates. **Hydrokinetic turbines → Power planning** selects finite-depth
Gaussian wakes, linear overlap, rotor-area moments and device Cp/Ct surfaces.
Run computes turbine-free hydraulics. In **Results → Turbines**, press
**Calculate / refresh power estimates**, then plot or export individual/all
turbines and energy. The hydraulic map remains the ambient flow.

First-release scope: straight rectangular channels, one upstream row, at most
three parallel rotors, static mesh, fresh hydraulic-only runs. Uniform vertical
ambient profile and quasi-steady wakes are explicit assumptions. Low-turbulence
power remains uncertain. Electrical power requires supplied efficiency. Existing
coupled actuator calculations remain separate; no analytical deficits are added
to an already disturbed velocity field. See the
[delivery report](../validation/hydrokinetic_turbines/reports/power_planning75/REPORT.md).

## Coupled device performance and energy from preview62

Open **Open BRAIDS Power68.cmd** on the Desktop. The editable laboratory example
is in `C:/Users/STREEM/Desktop/BRAIDS_Power68`, including results, performance CSVs
and a warm state for rerunning. Its README explains the restart selection.

In Hydrokinetic turbines, **Device speed / TSR surface + rotor inflow** accepts
`U_m_s,TSR,Ct,Cp` CSV tables and explicit wake parameters/provenance. Optional
local-speed shutdown limits and constant electrical efficiency are named
assumptions; they do not implement a manufacturer's free-stream controller or
rated-power regulation. Missing operating data stops the run. The Turbines results
tab plots and exports shaft/electrical energy from saved samples, including during
a run. Without supplied efficiency, electrical output is unavailable.

This new surface composition requires a fixed mesh and at most one upstream wake
per exposed rotor. Static local rotor/wake refinement is supported. Overlap is
checked before preparation; turning/yawed wakes remain unqualified. Low-turbulence
6D power is still overpredicted; laboratory settings are not universal river data.
Earlier simpler controls retain their previous scope, but do not reconstruct
vertical wakes for exposed-rotor power. Older editor drafts are not migrated.
See [the report](../validation/hydrokinetic_turbines/reports/power_workflow68/REPORT.md)
and [required inputs](../validation/hydrokinetic_turbines/GENERAL_POWER_INPUTS.md).

## Automatic local turbine mesh in preview61

Keep the river's chosen minimum cell size. Each active rotor and its enabled
wake automatically receive cells no larger than half its diameter, using local
2:1 subdivisions only where needed, with graded transitions. An already finer
river mesh is retained. For a 0.7 m rotor in a 2 m river mesh, the local size is
0.25 m. Different turbines may receive different local sizes.

Prepare again to use the new mesh builder. Static and dynamic mesh modes are
supported. Cropped pieces merge against their local square-area floor. Wake
length remains optional and defaults to 10 rotor diameters. Two cells per
diameter is a minimum, not a validated power-accuracy guarantee. The independent
wake-exposed power discrepancy remains open.

See [the mesh report](../validation/hydrokinetic_turbines/reports/local_mesh63/REPORT.md)
and the short editable projects in `C:/Users/STREEM/Desktop/BRAIDS_Turbine_LocalMesh63`.

## Wake refinement, profiles and saved states in preview60

Open **Open BRAIDS Wake and Restart.cmd** on the Desktop. Editable examples,
completed short results and instructions are in `C:\Users\STREEM\Desktop\BRAIDS_Workflow61`.

Each turbine now offers optional wake refinement, default **10 rotor diameters**
downstream, using the selected smallest cell size in a 5D-wide corridor. Spatial
profile plots default to bounded linear interpolation; raw cell steps remain
available and solids remain gaps.

Run defaults to **Load useful fields (changed setup allowed)**. Browse a saved
checkpoint, Prepare the intended setup, then **Check saved state against prepared
mesh**. Matching cells copy directly. If the mesh differs, explicitly press
**Map saved state to prepared mesh**, then Run. The physical domain/solids and CRS
must stay unchanged; the saved time is retained, so End time must be later.
Useful compatible fields load automatically; newly enabled physics uses target
initial settings. The summary lists what was loaded or initialized. Exact
continuation remains the separate full-state mode.

Project v40 and checkpoint v5 are explicit new identities. Old projects do not
execute. Previous public v4 checkpoints may be read as data from their original
authenticated project folder; new v5 checkpoints are self-contained for import.
83 focused tests, eight CPU/GPU initialization/AMR/restart controls and actual
Desktop workflows pass. Figures and limits are in
[the workflow report](../validation/hydrokinetic_turbines/reports/desktop_workflow61/REPORT.md).
Turbine–bedload coupling remains the next separate milestone.

## Near-wake correction and preview59

Open **Open BRAIDS Near Wake Validation.cmd** on the Desktop. The projects in
`C:\Users\STREEM\Desktop\BRAIDS_Turbine_NearWake60` retain the original turbine
law and loading, with explicit rotor-and-wake refinement. D/4 is the practical
starting point; D/8 is the finer confirmation. Refining the rotor alone failed
to improve the near wake and is preserved as a negative control.

The D/8 published-model RMS deficit differences are
2.262 pp at 2.5–4D and
1.542 pp at ≥4D. Experimental near-wake
RMS remains 6.556 pp: this
target is not closed. The apparent capped peak came from two unresolved central
cells, not a velocity limit. Raw profiles, maps and time histories are retained in
[the detailed audit](../validation/hydrokinetic_turbines/reports/nearwake60/REPORT.md).

Preview59 repairs only the cumulative floating-point water-budget diagnostic.
The per-step conservation guard and native physics remain unchanged. Sixty-six
focused checks and four exact-field CPU/GPU hydro/MPM AMR/restart checks pass.
Old results remain viewable; old preparations/checkpoints cannot execute under
the new runtime. Prepare explicitly. Project format v39 is unchanged.

## Direct experimental wake comparison and arithmetic repair in preview58

Use **Open BRAIDS Direct Wake Validation.cmd** on the Desktop. Four editable
projects are in `C:\Users\STREEM\Desktop\BRAIDS_Turbine_Validation59`: full 2000 s
turbine/control GPU runs and bounded CPU companions, verified against GPU at 100 s.
The reference uses Nash et al.'s already depth-averaged experimental curve;
no new hub-plane projection is applied. The steady/PML-input reconstruction
limits remain explicit. From ≥4D, experimental RMS deficit error is 1.465 pp;
the 6.217 pp maximum at 4D narrowly fails the original 6 pp screen. The near
wake remains too weak/narrow, so full experimental and bedload qualification
are not claimed. Figures and all screens are retained in
[the direct-comparison report](../validation/hydrokinetic_turbines/reports/desktop_validation59/REPORT.md).

The installed common CPU/GPU wrapper now uses the retained scale-aware FP64
water increment bound instead of an absolute 1e-10 m³ at every domain size.
The small-domain floor remains; no flux, physical law, CFL or native kernel
changes. Both full runs have zero outer rejections. All 66 focused checks,
four analysis checks and four exact-reference CPU/GPU hydro/MPM AMR/restart
controls pass. Old prepared inputs/checkpoints are rejected after the runtime
change; prepare explicitly. Project format v39 remains unchanged.

## Published turbine validation projects

Five completed current-format projects are in
`C:\Users\STREEM\Desktop\BRAIDS_Turbine_Validation58`. Use the Desktop shortcut
**Open BRAIDS Turbine Validation.cmd**. Each retains its DEM, settings, references,
prepared mesh, results, checkpoints and history; comparison sections are preloaded.
Results → Turbines provides histories and CSV export. Run starts fresh unless a
checkpoint is selected explicitly.

The installed preview57 reproduces the published OpenTidalFarm 2D wake with
0.240 percentage-point RMS deficit difference. Mycek's second-speed loading
check differs by +1.88% Ct / +0.16% Cp; its input performance curve is retained
from the first speed. CPU/GPU matched fields agree to roundoff. Experimental
wake agreement is conditional on an explicitly assumed depth projection, since
the measurements are at hub height. Direct measured depth-mean validation and
turbine–bedload coupling remain open. No solver changes were made for these runs.

Figures, identities and limits:
[turbine validation report](../validation/hydrokinetic_turbines/reports/desktop_validation58/REPORT.md).

## Faster breaklines and refinement bands in preview57

Mesh → Refinement controls → **Transition rows per refinement level** sets
1–32 face-neighbour rows around refinement requests, at each level's own cell
size. It works in both static and dynamic modes and is separate from the wet-front
buffer. Default 1 preserves the previous selection; larger values add cells.
The 2:1 ratio and finest-cell minimum area are unchanged. Constraints/domain
edges can overlap or truncate bands.

Batched face searches reduce clean geometry time from 5.56 to 3.86 s in the
11,912-cell four-level, two-oblique-breakline benchmark (30.5% less time).
The benchmark hierarchy is identical. This excludes DEM/native-record work.
CPU/GPU water and MPM regrid/restart controls pass; physical kernels are unchanged.

The project format is v39 with an explicit saved transition setting; older
formats are rejected rather than migrated. A fresh editable example is
`desktop/examples/refinement57/Refinement.braids.json`.
Figures and evidence: `validation/hydrokinetic_turbines/reports/mesh_grading57/REPORT.md`.

## Live turbine results in preview56

With **Live results** enabled before Run, **Results → Turbines** is available
throughout execution, alongside Profiles and Boundary flow. It appears before
the first saved frame; plotting and CSV export become available when the first
complete turbine sample arrives. The status shows the latest saved time.
Plots and exports use complete verified snapshots at the result-output interval;
reopen a plot to include newly saved samples. All-turbine and selected-turbine
CSV export also work while the worker is running. Solver calculations and the
preview55 mesh implementation are unchanged.

Report: `validation/hydrokinetic_turbines/reports/live_hkt4e/REPORT.md`.

## Multiple turbines and faster multilevel preparation in preview55

**Add turbine** appends a rotor using the selected settings and a unique name.
Select a turbine to edit its individual model, diameter, mounting and clearances;
Reposition and Remove affect only that turbine. The ambient viscosity is shared.
Results → Turbines selects one history and exports either selected or all rows.

Exact grid-coincident rotor lines reuse the existing face ledger. A shared
spatial index replaces repeated whole-mesh scans. In the 36,000-finest-cell
four-level straight-channel benchmark, geometry plus turbine mapping decreases
from 17.02 to 8.24 s; the off control is 7.61 s. Oblique constraints retain their
conforming split/merge route. DEM/native-record/output time is excluded here.

A three-rotor constant-inlet CPU run reaches near-steady values at 60 s: final
10 s power variation is below 0.4%. Matched GPU startup, independent shutdown,
actual two-rotor remeshing and exact restart checks pass. This remains a
hydrodynamics-only extension and does not add array experimental validation.

Open **Open BRAIDS Turbine Array.cmd** on the Desktop. Its separate editable
folder `BRAIDS_Turbine_Array55` includes all inputs, a reference CSV and figures.
Original projects remain unchanged; prepare them again under the new runtime.
Report: `validation/hydrokinetic_turbines/reports/array_hkt4d/REPORT.md`.

## Velocity-driven turbine power and decimal mesh correction

Select **Turbine model + fixed RPM** to estimate shaft power from an identified
TSR/Ct/Cp curve and the velocity facing the rotor. The solver resolves TSR,
coefficients and the incoming-velocity correction together on CPU/GPU. Import
the actual turbine curve with CSV headers `TSR,Ct,Cp`; diameter alone does not
predict blade performance. Out-of-range, parked, reversed and insufficiently
covered rotors switch off. Electrical conversion losses are excluded.

Results → Turbines → **Export all turbine quantities to CSV** includes time,
turbine/model ID, velocities, RPM, TSR, Ct, Cp, thrust, shaft power and operating
status/clearance margins. The run also writes `TURBINE_PERFORMANCE.csv` directly.
Each turbine keeps its own model and operating record.

Open **Open BRAIDS Turbine Power Demo.cmd** on the Desktop for a separate
editable 60 s hydrograph example in `BRAIDS_Turbine_Power_Demo54`. Its synthetic
curve verifies the workflow; replace it before estimating a real turbine.

Complete 0.10 m Cartesian cells are now preserved at projected coordinates.
Coordinate roundoff no longer triggers false area merging. Genuine cropped
cells continue to merge. Reprepare existing projects to rebuild their mesh;
older runtime checkpoints are not reinterpreted. The previous 0.25 m example's
hierarchy is unchanged. Report and comparison figures:
`validation/hydrokinetic_turbines/reports/power_mesh_hkt4c/REPORT.md`.

## Turbine clearance limits and 1 m channel

Open **Open 1 m Turbine Channel.cmd** on the Desktop. The editable project and
standalone GeoTIFF are in `C:/Users/STREEM/Desktop/BRAIDS_1m_Turbine_Channel`.
The synthetic channel is 30 m long, has a 4 m bottom, 1.5H:1V banks 2.5 m high,
a 0.1% bed slope, and 0.1 m DEM pixels. The reference water depth is 2 m.
Its 1 m rotor is mounted with the hub 0.85 m above the reference bed: nominal
bottom gap 0.35 m and top cover 0.65 m. Both minimum operating gaps are 0.25 m.
The inlet uses automatic mesh slope for its normal-depth discharge reference.

In the turbine step, **Hub height** sets mounting position. **Minimum rotor–bed
clearance** and **Minimum water cover above rotor** are independent operating
limits. If either fails, force and power are zero while water keeps flowing;
the rotor restarts when both limits and full coverage/bed clearance are met.
It does not move to follow the bed or surface. Positive limits are inclusive
within roundoff; setting a limit to zero still forbids exposure/contact.
Actual gaps and reserves above the required limits are available in plots/CSV.

Current project schema is **v38** and turbine editor format is 2. Older
projects/checkpoints are rejected, not migrated. Use the supplied fresh project.
The ordinary solver and native bindings are unchanged. Report:
`validation/hydrokinetic_turbines/reports/clearance_hkt4b/REPORT.md`.

## Hydrokinetic turbine workflow

Enable **Advanced → Hydrokinetic turbines**, then open the new workflow step.
Press **Add turbine** and click the rotor centre, then a second point downstream
along the local river. The second click sets direction, not diameter: BRAIDS
draws the rotor perpendicular to that direction, including in bends. Its width
comes from Rotor diameter. **Reposition / orient turbine** repeats the two
clicks; **Remove turbine** clears it. The Turbines layer can be hidden.
Orientation is fixed after placement; this is not an automatic flow-tracking
yaw controller or a yaw-force model.

Set diameter, hub height above the prepared bed and either constant Ct/Cp or
a prescribed TSR with an imported `TSR,Ct,Cp` CSV. The hub is converted to a
fixed absolute elevation during preparation. The method and its explicit
ambient viscosity are shown. Preparation checks the one-rotor hydraulic-only
scope and mesh resolution, and protects the entire numerical force support at
finest resolution. Rotor lines remain permeable; the minimum cell-area rule
still applies. Do not enable independent Advanced hydrodynamics mixing or
sediment with this first turbine delivery.

An exposed rotor produces zero thrust and power while water keeps flowing.
It restarts only when fully covered, with bed clearance and wet force support.
Results → Turbines plots applied Ct/Cp, thrust, useful power, hydraulic work,
coverage and activity. Export all quantities to CSV there; the run directory
also contains `TURBINE_PERFORMANCE.csv` and its checked publication index.
Ct/Cp are performance inputs, not predicted blade coefficients. Prescribed
TSR assumes that operating point is maintained; no rotor inertia is modeled.

Use **Open BRAIDS Turbine Example.cmd** on the Desktop; it now opens the same
current 1 m channel. The earlier `desktop/examples/turbine_hkt4` is a sealed
preview52 reference template, not a current-format editable project. The current case is a fresh
60 s functional flume project; it is not an additional experimental validation.
The usual launch shortcut also opens the new version. Current project schema
is **v38**; older project/checkpoint formats are rejected rather than migrated.
Older example/version sections below document retained historical releases.
Independent long-running sediment projects and their frozen runtimes are
unchanged. Report and figures:
`validation/hydrokinetic_turbines/reports/desktop_hkt4/REPORT.md`.

## Bank erosion and collapse in preview48

Morpho evolution → General exposes three bank methods: local FLOW-style dry-bank
sharing, surveyed toe supply, and surveyed toe supply with retained block release.
Either conservative equalization or Jiang–Cao-criterion repose may follow.
All 71 ordinary bedload/suspension combinations retain their measured-diameter
and paired-formula restrictions. Bank supply uses accepted net erosion once;
MF is not multiplied again. One finite bulk bed, sediment start=0. The mapped
toe corridor does not automatically migrate. Resolved setup explains these limits.
The full coupled Jiang–Cao transport bundle still uses MF1 and its own repose;
ordinary Exner bank supply is not admitted to that separate inventory coupling.
Sediment, banks and collapse are OFF by default. Project v36 rejects old formats.

Open the CMD shortcuts in `desktop/examples/banks224` for matched short examples,
results and checkpoints. Earlier examples remain historical evidence. Report:
`validation/sediment/river_morphodynamics/reports/desktop_banks224/REPORT.md`.
Next: a short matched FLOW 2DH channel-widening comparison, first at MF1.

## Morphological factor and suspended transport

The sediment workflow has General, Bedload, Suspended and Resolved setup tabs.
Bedload and suspension are separate selections. The diameter table asks for
the union of the actual measured descriptors; none are inferred. Resolved setup
records the selected equations, auxiliaries, assumptions and limitations.

| Suspended formula | Required measured diameters |
| --- | --- |
| Van Rijn 1984, original | D50, D90, Ds |
| Zyserman & Fredsøe 1994 | D50, Ds |
| Camenen & Larson 2008, current only | D50, Ds |
| García & Parker 1991 | Ds |
| de Leeuw et al. 2020 | D84, Ds |
| Van Rijn 1993 — FLOW 2DH bundle | D50, D90, Ds |
| Supplied suspension / settling only | Ds |

The first five run alone or with any of the nine ordinary bedload formulas;
their additional bedload descriptors are also required. The FLOW 1993 bundle
runs alone or with its matching Van Rijn 1993 bedload formula. Jiang–Cao
cannot add independent suspension. Supplied suspension/settling only runs alone
or with any of the nine ordinary bedload formulas. It uses clear-water settling
and a well-mixed profile, with no bed entrainment or hindering correction.
No alternative formula is silently substituted.

For a first test, create a **new v36 project**, enable the sediment extension
and suspended transport, and select one of these formulas in Suspended.
Turn bedload off or select a compatible formula. Enter the measured sizes in
General, choose a **positive morphological factor and sediment start 0**. Bank supply and collapse are optional in Morpho evolution; review their finite-stock requirements in Resolved setup. Leave sorting, bedform and slope extras off. Ds is the suspended representative
diameter, not a bed percentile or another transported class. Set initial/inlet
concentrations in Sediment transport BC, review Resolved setup, then prepare.
CPU/GPU selection remains yours; sediment is unchecked in a new project.

MF defaults to **1**. All seven suspended choices, alone or in their existing
compatible bedload combinations, now support MF different from 1. MF scales
bedload divergence and wet bed/suspension exchange. Water, suspended advection
and concentration relaxation stay on hydraulic time; dry stranding is unscaled.
Available bed mass limits entrainment after accounting for MF. The additional
bed mass appears explicitly in the acceleration ledger and diagnostics.

MF is an approximation for slow morphology under near-steady or repeated
forcing, not exact compression of a flood or finite sediment pulse. Check its
sensitivity for your application; it need not yield an MF-fold runtime gain.
The Jiang–Cao coupled-phase model still requires MF1. Ordinary bank supply and both collapse choices are qualified with positive MF within the one-material finite-bed scope described above.
Old projects and checkpoints are rejected; there is no automatic migration.

The earlier `desktop/examples/morfac221` MF examples are retained v33 evidence,
not current inputs. Use the new `desktop/examples/banks224` projects.
The earlier MF report remains at
`validation/sediment/river_morphodynamics/reports/desktop_morfac221/REPORT.md`.

The previous preview44 combined qualification recorded **1,158 distinct passing packaged checks**,
including conservation, cropped walls, wet/dry AMR, finite-bed exhaustion,
restart and retained models. The new 74 supplied-suspension tests passed again
after validation-source recovery. The interrupted original receipt is retained;
three earlier FLOW replay passes could not be rerun because their input files
were lost. The report identifies rebuilt fixtures and this reproducibility gap.
S6 adds the original VR1984 profile and settling branches, ZF1994's
boundary-layer skin-friction solve and Rubey settling, and the current-only
CL2008 exponential profile. The CL threshold-ratio discrepancy in the supplied
paper is explicitly documented and resolved using the authors' other sources.
All shallow-front and local 2DH adaptations are named and saved. Original
VR1984 is not the same formula as the retained FLOW 1993 bundle.

These are equation and coupled numerical checks, **not yet experimental river
validation** or a universal steep-gravel suspension claim. One bulk material,
finite fully mixed bed and dilute noncohesive use are the current scope.
Bank/toe erosion and both collapse choices are available within the qualified scope above; physical channel-widening comparison is next.

The sediment editor is format 13; project/model contracts are v36/v3. Older
projects/checkpoints are rejected, not migrated; their files are preserved.
The usual CMD shortcut opens preview48. The older `desktop/examples/supplied220`
files remain historical v32 evidence, not current inputs. See the
[earlier report and comparison figures](../validation/sediment/river_morphodynamics/reports/desktop_supplied220/REPORT.md)
and [recovery record](qualification/supplied220/RECOVERY.json).
The earlier validation/report directory was lost; its historical reports are
not silently recreated by the new source/test recovery.

## Independent grain sizes in preview38

The selected formula sets the editable measured-diameter rows. MPM/Wang and
Jiang–Cao require D50 and D90; Smart & Jäggi requires Dm, D30 and D90; Recking
2010 requires D84 and D50. The other formulas request their single defined
diameter. No percentile is inferred. These describe one bulk sediment material,
not separately transported classes. Jiang–Cao automatically selects its named
Sun/van Rijn resistance in Domain; compatible prescribed resistance is retained
for the other formulas. Invalid combinations stop before meshing.
[Input list and 405 packaged checks](../validation/sediment/river_morphodynamics/reports/desktop_bulk213/REPORT.md).

Create a new v36 project: old projects/checkpoints are deliberately rejected,
not migrated. Their files are preserved. Sediment remains unchecked by default.

## Jiang–Cao research bundle in preview36

An explicit **Jiang–Cao 3/2 (research bundle)** choice now uses the seven-field
mobile-bedload composition with its own roughness, phase exchange and
finite-adjustment secondary flow. It is not ordinary MPM/Exner. The current
release takes independently measured D50 and D90. Use MF1, sediment start 0, constant qs/q supply and the compatible
settings shown in Resolved setup. Hydraulic Q denotes **mixture** volume flux;
water Q = mixture Q/(1 + qs/q). Invalid combinations stop before meshing.

Morpho evolution offers the existing BRAIDS conservative wet/dry collapse and
the new Jiang–Cao repose projection as distinct, mutually exclusive choices
with the ordinary-formula extension described above. No sediment or collapse is activated
by default. The eight earlier formulas retain exact previous-release results.

Preview36 delivers the corrected inlet curvature stencil, separate shallow
bedload/profile activation and finite-stock exchange. Dry oblique-wall AMR
controls now pass on both backends and restart exactly. All 519 packaged
tests pass, and the eight earlier laws retain exact previous-release fields.

The delivered project is `C:/Users/STREEM/Desktop/BRAIDS_JiangCao_MF1`.
Open its `braids_project.json` or local CMD launcher. It includes a 20 s GPU
flume startup, DEM, settings, results and checkpoints; extension to 21 s has
been checked. MF1 only. Bed changes begin, but the wet front reaches only
4.8 m of the 25 m flume. Depths remain below the secondary-profile activation
range, so this is not established braiding or experimental validation.
Choose a different backend before preparing a new run; never reinterpret a
checkpoint. Older examples and checkpoints remain historical evidence.
[Report, figures and limits](../validation/sediment/river_morphodynamics/reports/desktop_delivery211/REPORT.md).

## Wong and Parker alongside MPM in preview34

The Bedload menu offers **MPM** and **Wong & Parker (2006)** as separate
CPU/GPU choices. MPM is unchanged and remains the initial formula. Wong-Parker
uses the fixed 3.97/0.0495/1.5 relation and a transport multiplier of 1.
Enter **Dm (arithmetic mean)**, not D50; changing the diameter basis clears
the size so it cannot be silently reused. Mixtures and suspension remain deferred.

All 329 regression tests pass on the installed release. The seven earlier laws
retain exact preview33 fields in the paired controls. Wet/dry groins pass
conservation, inlet-capacity checks and exact checkpoint continuation. Maximum
CPU/GPU bed differences are 0.0010 mm (wet) and 0.227 mm (dry AMR), with dry
final RMS 0.0156 mm. These are numerical checks; experimental validation is next.

Please test using the existing CMD shortcut and the
[Wong-Parker groin example](examples/bedload193/wong_parker/braids_project.json).
It enables bedload and selects GPU; choose CPU before preparing if desired.
Prepare fresh: old-runtime checkpoints are rejected.
[Equations, figures and release evidence](../validation/sediment/river_morphodynamics/reports/desktop_wong193/REPORT.md).

## Camenen and Larson bedload in preview33

The Bedload menu now includes **Camenen & Larson (2005)** for one-grain,
steady-current bedload only. Enter **D50** and molecular water viscosity
(default 1e-6 m²/s, not eddy viscosity); the transport multiplier defaults to 1.
Critical Shields is calculated from D50, density and viscosity. The published
smooth low-stress tail is retained, without an MPM hard cutoff or form-drag
reduction. Mixtures, suspension and waves remain deferred.

All 296 regression tests and the installed wet/dry groin checks pass, including
exact checkpoint continuation. The six previous laws retain their preview32
final results. The largest new-law CPU/GPU bed difference is 0.714 mm in the
dry-start AMR test and falls to 0.103 mm by the end.

Use the existing CMD shortcut to open the
[self-contained Camenen groin example](examples/bedload192/camenen_larson/braids_project.json).
It enables bedload and selects GPU; CPU is available before preparation.
Prepare a fresh run: old-runtime checkpoints are rejected.
[Equations, comparison figures and qualification](../validation/sediment/river_morphodynamics/reports/desktop_camenen192/REPORT.md).

## Explicit grain diameters in preview32

The Grain fractions table presets **Percentile / basis** from the formula:
D50 for MPM, Ashida-Michiue, Wang-Fredsoe and Ancey-Recking; Dm (mean, not
a percentile) for Smart & Jaggi; D84 for Recking (2010). Changing to a different
basis clears the size: enter the correct measured value in mm before Prepare.
Wrong or missing definitions in active inputs are rejected before meshing.
The visible notes explain auxiliary diameter assumptions, including D50=D84
in the current one-grain Recking adapter. This is not a graded-mixture model.

Create a fresh project: old unlabeled formats/checkpoints are not migrated.
The six-law CPU/GPU groin checks retain exact preview31 fields and restart;
82 new contract tests and 180 existing regressions pass. See the
[grain diameter audit](../validation/sediment/river_morphodynamics/reports/desktop_grains191/REPORT.md).

## Published bedload laws and sediment stepping in preview31

The Bedload menu now also offers **Ancey & Recking (2023)**, Equation 42, and
**Recking (2010)**, the single-grain limit of main Equation 12. Both use physical
hydraulic shear, with no hidden MPM form-drag reduction. Recking requires an
explicit representative reach slope in m/m; 0.01 means 1%. One grain means
D84 = D50. Neither option adds suspended transport or a mixture/armor model.

A retained accepted-step estimate avoids repeating oversized sediment trials.
The existing 5% bed-increment guard, physical formulas and hydraulic CFL are
unchanged. The rejection-heavy dry-AMR Wang test improves from 110.3 to 27.6 s
on CPU and 250.2 to 71.4 s on GPU. Those single-run timings are for this small
test, not a universal speed guarantee. All 180 regressions pass; old-law wet
groins remain byte-identical. The two new laws pass coupled wet/dry AMR, budgets
and restart checks, with maximum CPU/GPU bed differences below 0.663 mm.

Use the existing CMD shortcut and open either self-contained short example:

- [Ancey & Recking groin](examples/bedload190/ancey_recking_2023/braids_project.json)
- [Recking 2010 groin](examples/bedload190/recking_2010/braids_project.json)

These examples explicitly select GPU and enable bedload; new blank projects
still have sediment unchecked. You may explicitly choose CPU before preparing.
Start a fresh run: old-runtime checkpoints are rejected. Historical results
and all existing example/run folders are preserved.
[Equations, parameter distinctions, comparison figures and qualification](../validation/sediment/river_morphodynamics/reports/desktop_transport190/REPORT.md).

## Direction-aware reconstruction retained from preview30

The matched advanced-ON/OFF comparison and the audit of all four currently
then-available bedload laws are in
[Desktop189: figures and parameter audit](../validation/sediment/river_morphodynamics/reports/desktop_bend189/REPORT.md).
Both backends agree within 0.045 mm bed change in the bounded bend comparison.
Secondary-current steering, rather than viscosity alone, produces the strong
inner-bank deposition. These are early narrow deposits, not validated equilibrium
point bars. The run covers 200 s warm-up plus 20 s mobile bed at MF5; the full
400 s example is not claimed complete. This was qualified before preview31.

The approved direction-aware water-surface/bed limiter is installed on CPU and
GPU, together with the secondary-flow clock and incoming-current corrections.
It retains explicit SSPRK22 augmented f-wave, the physical sediment laws and
sediment reconstruction. Sediment remains unchecked by default.

Two-meander cold starts finish with maximum CPU/GPU bed differences of 0.044 mm
(wet start) and 0.063 mm (dry start). Restart/cancellation is exact; a separate
moving-AMR groin exercises seven mesh geometries. These bounded checks are not
an experimental or equilibrium validation. Compared with the old limiter,
localized bed changes reach 36 mm although bulk erosion/deposition changes <0.8%;
the user approved this numerical correction before delivery.

Use the existing CMD shortcut. For testing, open the fresh
[advanced-ON project](examples/two_meanders188/advanced_on/advanced_on.braids.json)
or [matched advanced-OFF project](examples/two_meanders188/advanced_off/advanced_off.braids.json),
then prepare and start a fresh run. Their requested 400 s runs have not yet been
performed. The examples select GPU; backend choice remains yours. Both leave
bed-slope transport off. In BC & IC, incoming secondary-current intensity is
explicit and defaults to zero for an uncurved upstream approach.

Old project/checkpoint identities are intentionally rejected; there is no
automatic migration. Historical results remain readable.
[Qualification, raw figures and next milestone](../validation/sediment/river_morphodynamics/reports/desktop_wall188/REPORT.md).

## Smart and Jäggi bedload retained from preview29

Enable the Sediment transport extension and bedload, then choose **Smart & Jäggi**
under Sediment transport > Bedload. This is the single-grain energy-slope
formulation: coefficient A defaults to 4 and critical Shields stress to 0.05.
Both are dimensionless and editable. The grain size is entered in mm in General;
the solver uses metres. The grading ratio is exactly one for the single grain.
Energy slope comes from physical velocity, water depth and hydraulic resistance,
not directly from the DEM slope. The formula note and saved resolved setup explain
the applicable slope range and the distinction from the FLOW formulas.

CPU and GPU support the new law, including capacity-fed inlets, finite erodible
stock, dynamic wet/dry AMR and exact same-backend restart. The other three
bedload laws and the hydraulic solver are unchanged. MPM remains the initial
formula and the sediment extension remains unchecked for new projects.
[Implementation, checks and figures](../validation/sediment/river_morphodynamics/reports/desktop_smart183/REPORT.md).

## Interface and profiles retained from preview28

The right-panel sections start collapsed except the first visible section on
each page or subtab. The project JSON editor now opens a searchable settings
tree, with a separate formatted, syntax-highlighted source editor.

Mesh > Refinement controls includes **Refine domain border to smallest size**,
checked for new projects. It protects outer boundary cells during preparation
and dynamic AMR, including dry cells. Clearing it allows ordinary domain walls
to coarsen; inlet/outlet, solid and breakline refinement rules remain in force.
Existing minimum-area merging is unchanged, so cropped/merged cells need not
have the exact area of a finest square.

View > Show result minimum / maximum marks the visible result extrema and
coordinates. They follow the selected field and frame, using stored cell values
rather than display smoothing. Results > Profiles now distinguishes a spatial
profile of the displayed map from the existing flow/mean time series. Spatial
profiles can follow map animation or remain frozen, and export PNG/CSV with
the variable and frame time in the filename. Stop is beside Execution log and
turns red when cancellation is available.

The hydraulic and sediment solver binaries are unchanged. New project files
explicitly store `settings.adaptation.domain_border_enabled`; older projects
missing this required field are not automatically migrated. Old runtime
checkpoints remain rejected. Saved results are still readable. The bundled
natural-river example has a fresh current configuration.

[UI changes, test evidence and screenshots](../validation/sediment/river_morphodynamics/reports/desktop_ui182/REPORT.md).

## Exact reflecting-wall pressure in preview27

The qualified exact wall Riemann pressure is now the single maintained CPU/GPU
implementation. It fixes the pressure approximation for flow moving toward or
away from a solid wall, including clipped and oblique wall faces. Interior
augmented f-wave fluxes, explicit SSPRK22, reconstruction and sediment laws
are unchanged. There is no new numerical-method selector or compiler requirement.

The staged package passed 517 regression/evidence checks before installation.
All 16 packaged wall/AMR/three-law groin controls reproduce the isolated qualified
correction exactly, including cancellation/restart. The hydraulic boundary
inventory passed 32 CPU/GPU checks. Sediment remains unchecked by default.

Use the existing launcher. **Reprepare and start a fresh run**: earlier runtime
checkpoints are intentionally rejected, while saved results remain readable.
The groin jet/scour discrepancy against Delft3D remains open; this release is
a justified wall correction, not a claim that the two models now agree.

[Release report and figures](../validation/sediment/river_morphodynamics/reports/desktop_delivery172/REPORT.md).

## Wall and bed-flux corrections retained from preview26

The qualified reflected wall-velocity bounds and split Exner flux correction
are now installed on CPU and GPU. The latter retains physical second-order
bedload capacity and bed-jump viscosity, without adding hydraulic mass/momentum
jumps a second time to the independently evolved sediment flux. No physical
formula, integrator, backend selection or UI option was changed.

Short wet/dry transcritical shoals pass with MPM, Ashida–Michiue and Wang–Fredsøe:
maximum CPU/GPU bed difference is 0.0092 mm. MPM's shorter-step bed difference
is 0.0011 mm over saved frames; Ashida's is 0.0132 mm. Both backends restart
exactly, conserve water/sediment and retain nonnegative stock. The six packaged
groin runs reproduce milestone164 exactly. This is bounded numerical release
qualification, not experimental validation or complete Delft3D equivalence.

Use the existing launcher. **Reprepare and start a fresh run**: old runtime
checkpoints are intentionally rejected. Existing results are unchanged.
Sediment remains off by default; the three single-grain bedload choices remain
the only public sediment laws. No QGIS update or clean-PC installer is included.

[Report and comparison figures](../validation/sediment/river_morphodynamics/reports/desktop_shoal165/REPORT.md).

## Weir, bedload choices and snapshot correction retained from preview25

The free-weir boundary now solves a subcritical rating-law state using both
shock and rarefaction branches. Local recirculation cannot import water or
abort an otherwise valid outlet. Blocked flow retains reflected pressure.
Dry startup and initially supercritical approaches were tested on CPU/GPU.
It is still a **free, undrowned weir**, not a submerged structure model; an
inadmissible control is reported rather than forcing velocity to a chosen Fr.

The bedload menu now offers **MPM**, **Ashida–Michiue (FLOW)** and
**Wang–Fredsøe (bedload)**. Only one grain and one finite mixed bed are exposed.
Source coefficients are shown in the method note and stored in JSON; suspension
is never silently added. Sediment remains disabled by default.

The large isolated bedload snapshot spikes were caused by floating-point
output-clock tail steps, not old results. The preceding meaningful step now
lands on the output time. No visual smoothing or transport-law change was used.
Old result files remain unchanged: **reprepare and start a new run** to see the
correction. Old runtime checkpoints are not converted.

[Report and figures](../validation/sediment/river_morphodynamics/reports/desktop_boundaries160/REPORT.md).

## Hydraulic boundary inventory

Four inlet choices remain: discharge only (already-wet inlet), discharge with
an elevation reference, critical-depth reference, or automatic normal-depth
reference. Every inlet enforces discharge. A reference depth is not a second
strong condition when backwater changes the boundary state.

Three outlet choices remain: water elevation, free outflow (critical control
with supercritical export), and a free undrowned weir. Stage outlets retain the
explicit choice between outflow-only and permitted subcritical backflow.
The weir is a structure-specific condition, **not a general river outlet**.
The preview24 local-reversal failure is corrected in preview25.

Removed the duplicate depth-plus-datum inlet (equivalent to elevation) and
standalone supercritical outlet (already included in free outflow). Saved
projects selecting them are rejected with an explanation, not converted.
Reprepare before running with this runtime; old checkpoints are not retagged.
See the earlier
[groin tests, comparison figures and open issue](../validation/sediment/river_morphodynamics/reports/desktop_boundaries159/REPORT.md).

## Open it on this PC

Double-click **Launch BRAIDS Desktop.cmd** on the Windows Desktop. The same
launcher is available in the BRAIDS repository root. It starts the graphical
application with `pythonw`, so no CMD window stays open. Startup errors are
written to `desktop/evidence/STARTUP.log`.

## Sediment workflow (single-grain bedload)

Enable **Advanced > Sediment transport** to reveal three steps. Opening the
extension does not activate transport. Check **Enable bedload transport** in 04 / General.
CPU and GPU use the same saved physical choices and warm-up clock.

- **04 Sediment transport**: General has the activation checkbox, start time, morphological factor and
  one grain's size/solid density/dry bulk density. Bedload has three named
  formulas and one visible scaling coefficient, plus the explicit resolved setup. Before the start
  time, hydraulics runs over a fixed bed with no sediment boundary flux.
- **05 Morpho evolution**: default initial erodible thickness above a fixed base.
  Zero means a bare non-erodible base that can receive and subsequently re-erode
  deposits. Conservative Exner updates start with bedload. MF defaults to 1.
- **06 Sediment transport BC**: inlets only — Capacity (selected formula), Total sediment
  (solid m³/s), or Sediment/water ratio (qs/q, solid volume per water volume).
  Prescribed feed can be constant or a physical-time table. Total feed is
  distributed by inward water discharge. Outlets export only, with no reentry.
  Exterior bed/composition are automatic, explicit in JSON, and not UI controls.
- **Domain**: roughness default plus polygons (Manning, Chézy or Colebrook), and
  initial erodible-depth polygons. Draw/edit polygons and edit values in their
  table. Overlapping polygons of the same kind are rejected. Cropped cells use
  actual fluid-overlap areas; roughness is remapped from the polygons after AMR.
  Changing the roughness law does not convert polygon values: review/re-enter
  them in the units shown for the new law before preparing.

Suspension, bank erosion, collapse, hiding, layers/sorting, bedform feedback and
capacity formulas beyond the three listed above are not public choices in this release. Their underlying
source/history is retained, not deleted or claimed as newly qualified. Hidden
active settings are rejected rather than silently switched off. Advanced
hydrodynamic settings remain separate and explicit.

**Prepare, Run, and Generate and Visualize mesh** all check configuration before
mesh construction. Invalid selections produce one message with the first problem,
its location, and a suggested correction **in the execution console, without a
popup**. Save preserves unfinished drafts quietly. Fix the first error and press
Prepare again. The resolved setup is written to the log before mesh construction.
CFL is never changed automatically. Hydraulic face-sum CFL can be set up to 1,
also with bedload, secondary flow and turbulence. Positivity, diffusion and
relative bed-change guards remain active. MF scales sediment flux/bed evolution,
not physical hydraulic time; large MF can require smaller morphological steps.

Use a new project: schema v24 explicitly stores material polygons and rejects
older project/checkpoint identities. Historical files are preserved. MPM and
hydraulic integration are unchanged; the depletion guard retains negligible
floating-point residual stock rather than clipping negative cell masses.
See the [bare-bed groin figures and qualification](../validation/sediment/river_morphodynamics/reports/desktop_sediment158/REPORT.md).

### Continue a sediment run

Checkpoints are in the project's data folder under `checkpoints/<run name>/`.
For 120 → 240 s: change total simulation time to 240, select the `t_120` checkpoint,
then press Run. No new mesh preparation is needed. Only end time and checkpoint
cadence may change; physics, backend, geometry, inputs and runtime must match.
Time-dependent forcing must already cover the requested extended horizon.
Each continuation creates a new results folder and preserves old results.
Move the whole project/data folder together, including checkpoint JSON sidecars.

Profiles can plot bedload flow and mean sediment fields, and export them to CSV.
Available suspended/combined fields are shown only for results containing that
physics; this MPM-only release does not activate suspension. Bedload profiles use
the last accepted limited face flux reconstructed to cell vectors, excluding MF;
they are not exact boundary ledgers. Use Boundary flow for exact budgets.
The collapsible Map layers list can hide profiles, probes and material polygons
without deleting them. The map's **i** tool reports the stored value at a click.

The supplied **BRAIDS bare-bed groin** Desktop folder contains CPU/GPU projects,
0–120 s results, their continuations to 240 s, checkpoints and comparison figures.
It is a short functional/conservation example, not experimental validation.

Drag a DEM from Explorer onto the map to import it. Hold the middle mouse button
to pan without losing an unfinished polygon/polyline. Release it to continue
drawing. CRS labels use an authority code or a short custom-CRS label; the original
WKT remains the computational reference. Splitter panels cannot collapse.

Choose **Help > Open demo** for a separate copy of the natural-river setup from
`C:\Users\STREEM\Desktop\01010101`. The DEM, domain, boundary conditions and
backend/settings are retained; old outputs/checkpoints are not copied. Each
opening creates its own folder under `desktop/test_projects`, leaving the
original untouched. Opening does not start a simulation. Select **Run** when
ready; preparation starts automatically if needed. This is a usability demo,
not a newly validated natural-river experiment.

For your own case:

The authenticated runtime includes the qualified breakline geometry and oblique
adjacency changes. Its numerical kernels and physical methods are unchanged.
Save and prepare after changing configuration; fresh runs start dry, including below sea level.
Completed results and their saved input parameters remain in their original folders.

1. Import a projected, metre-based DEM in Domain. Draw or import one simple
   calculation-domain polygon. Choose a `.braids.json` project filename on the first import;
   the imported DEM, vectors and their sidecars are copied there immediately.
   A virtual raster must first be exported to a self-contained raster.
2. Choose the smallest mesh size and refinement levels. Static mesh, the default,
   uses terrain slope breaks. Dynamic AMR uses water depth. **Generate and Visualize mesh**, at the end of **BC & IC**,
   prepares and displays the actual initial cells; Run reuses that preparation
   while the saved project remains unchanged. After small parameter changes,
   Prepare reuses verified mesh geometry and DEM integrals when their actual
   dependencies are unchanged. It still regenerates current initial fields,
   forcing and native run records. Changed geometry/grid invalidates geometry;
   changed DEM contents invalidate terrain integrals. The mesh preview reads
   the prepared display artifact rather than reconstructing polygon topology.
   Under **Mesh strategy**, **Add breakline** draws a polyline, not a polygon.
   Click two or more points; Enter or double-click finishes without joining the
   last point to the first. Right-click finishes and exits drawing. Select its name and **Edit breakline** to
   drag vertices; right-click ends editing. **Remove breakline** removes that
   constraint. **Breaklines** in the layer list controls visibility only.
   The Mesh order is Add/Edit → breakline list → Remove → Refinement controls.
   These lines are permeable mesh constraints, not walls or elevation changes.
   Cells touching either side stay at the finest level in static or dynamic mode.
   In every mesh, including one without breaklines or solids, all cells have area at least the smallest
   mesh size squared. Cropped pieces merge without crossing the breaklines or
   moving the domain. At bends, several neighbors can merge together when no
   valid single-neighbor union is possible. Shapes must also satisfy the solver's polygon-validity
   requirements. If the constraints cannot be retained with this area floor,
   preparation reports the problem location; reduce the size or edit the geometry.
   Breakline edits invalidate the mesh and its preparation cache. More constraints
   can increase mesh size and preparation time; the area rule alone does not
   guarantee a particular timestep.
3. **Select edge** selects one domain edge at a time, including with Ctrl. Assign an
   inlet/outlet name, then double-click it in the list or map to configure its
   hydraulic method. Reapplying an existing
   name replaces its previous edges. Unassigned exterior edges are walls.
4. Use **Align inlet/outlet sections**, immediately below the boundary table,
   if required. Applying conditions never aligns geometry automatically. The proposed
   geometry and section-width changes are shown before acceptance. Walls can
   remain cropped. Domain geometry edits invalidate old boundary assignments.
   **Internal conditions** below it lets you draw/edit/remove polygon or ellipse
   solids. Their outlines are impermeable walls; their interiors are cropped out.
   Fluid cells touching solids stay at the finest level in static and dynamic
   modes. Ordinary exterior walls are not automatically pinned to that level.
   The **Generate and Visualize mesh** button is at the bottom of this BC & IC tab.
5. Set backend, duration, output interval and CFL in Run. Fresh runs start dry
   with zero initial velocity; hydraulic roughness defaults to Manning.
6. Use File > Save project to choose a filename. For `River.braids.json`, BRAIDS
   creates `River_data` alongside it. The project name appears in the title bar.
   Prepare, Run and Stop appear only in Run. Prepare displays the generated mesh.
   The execution log stays closed unless you open it; its context menu has Clear log.
   Stop requests safe
   cancellation during preparation or calculation. Initial, final and cancelled
   checkpoints are retained; the default checkpoint interval follows duration.

Preview17 repairs merged-cell face adjacency and the Windows project-watcher
sharing error after Save/Prepare/Run. On the retained `01obstacle` geometry,
clean hierarchy construction is 2.16 times faster (30.8 to 14.2 seconds), by
reusing existing faces whenever no sub-cell splitting is needed. Restart Desktop
and regenerate the mesh before a fresh run; previous-runtime preparations and
checkpoints are not reinterpreted. See the
[qualification report](../validation/sediment/river_morphodynamics/reports/desktop_walls152/REPORT.md).

Preview19 changes **View > Smooth results** to join nearby wet-cell tips with
straight segments, replacing preview18's rounded Gaussian shoreline. The local
AMR cell scale limits the shortcuts; isolated puddles and narrow channels remain.
Water follows the actual solid outline (polygon or discretized ellipse): these
physical walls are pinned, not smoothed or replaced by a square outline. Colors
use bounded piecewise-linear interpolation instead of cubic interpolation and
broad neighbor averaging. Interpolation cannot bridge a solid or dry gap.
Only display-pixel edge antialiasing remains, not a wide opacity fade.
Disable smoothing to see the raw wet-cell footprint. This is display-only:
solver fields, probes, profiles and CSV measurements are unchanged; the apparent
shoreline is not a new computed inundation boundary. Restart Desktop and view
existing saved results without rebuilding the mesh or rerunning.
[Comparison figure and checks](../validation/sediment/river_morphodynamics/reports/desktop_shoreline154/REPORT.md).

Preview20 adds **zoom-dependent terrain tiles**. Zooming in reveals the native
DEM pixels instead of magnifying a fixed 1,600-pixel overview. Only visible
256-pixel tiles are colored, and a bounded cache reuses them while panning.
The `01obstacle` 0.5 m DEM now displays at 0.5 m when zoomed in, rather than
remaining at the former 2.5 m display spacing. Hillshade uses neighboring-tile
samples to avoid seams, with correct X/Y spacing for non-square DEM pixels.
Palette/range changes invalidate the color cache; opacity and visibility remain
view-only controls. High-DPI screens and image/video exports select detail from
their actual render scale. Source DEM values, meshes, solver inputs and saved
results are unchanged. Restart Desktop; no preparation or simulation is needed.
[Before/after and qualification](../validation/sediment/river_morphodynamics/reports/desktop_terrain155/REPORT.md).

Right-click leaves map drawing/editing; Enter or double-click finishes a drawing.
Domain drawing/editing tools are on the map; zoom to domain is above the plus
button and in **View > Fit domain**. **Draw a guide line** creates an editable
guide. Lines follow the cursor while drawing, and polygons have no fill.
Select uses a cursor-arrow icon and Pan uses a hand. The ruler immediately
below Pan measures a temporary multi-segment path: click successive points to
label each segment in metres; the label beside the mouse shows the total,
including the current live segment. Right-click, Escape or switching tools
clears it. It creates no layer or saved geometry and is omitted from exports.
Bank polygons and matching toe lines can be imported, drawn and edited when that optional method
is enabled. Incomplete bank pairs are not executable survey snapshots.

## Workflow and results

Use the top **Model & Run / Results** buttons to switch modes. Model navigation
follows Domain, Mesh, BC & IC and Run. **Advanced > Advanced hydrodynamics**
adds optional step **03a Advanced hydrodynamics** without renumbering sediment.
**Advanced > Sediment transport** adds steps 04, 05 and 06 described above. The right
panel shows the selected step. Returning from Results restores the last model step.
The shared resolver shows method identities and rejects incompatible choices;
the interface does not silently enable physics or change the CFL. The resolved
combination is available in 04 / General and logged before preparing the mesh.
Morphological factor is in 05 / General, not Run. CPU threads are disabled for GPU.

**Settings > Preferences** provides language, light/dark theme, decimal separator,
geometry/mesh colors and background choices. Light is the default; polygons/lines
default to red and mesh outlines to RGB green `#00ff00`. OpenStreetMap and Satellite
are also available above the map and under View. Toolbar zoom uses the visible
map center; the restored line tool draws editable domain guides.
Snap while editing and Animate adaptive mesh are in Settings and default to on.
Probes/profiles default to yellow; Settings changes that color. Selected features
are orange. The **Layer appearance** button (◐) on the map controls terrain and
loaded result variables. Choose a layer there to adjust its palette, automatic
or manual colorbar minimum/maximum, opacity and legend visibility. There is no
terrain-opacity slider in the layer list. Appearance choices are stored with
the project when saved, without changing simulation inputs. Backgrounds remain
behind the terrain without automatically making it transparent.
In Model & Run, checking **Mesh** also restores an already-generated initial
mesh from retained preparation/results, including after reopening a project.
Toggling or switching modes reuses that display rather than rebuilding it.
Unrelated timestep/probe edits do not hide it; changed mesh inputs hide a stale
preview until a corresponding mesh is generated. Loading a retained mesh while
the solver runs does not change its execution/backend or progress.
Online backgrounds need network access. Language applies on restart; other
presentation preferences apply immediately. Numeric controls show at most three
decimal places and retain full stored precision; very small numbers use scientific
notation.

Choose **Show results while running** in Run before starting. Its choice is
locked while preparation/calculation runs. If unchecked, live Results controls
remain disabled; completed results become available after the run finishes.

Results use the saved native cropped polygons and mesh epochs. **View > Smooth
results** defaults on and interpolates map colors within the retained wet
footprint. Turning it off shows raw cell colors. Smoothing does not change
solver fields, probe/profile plots or CSV data. Water-surface and other hydraulic
maps exclude dry cells; terrain remains a separate layer you can hide. Bed
elevation and bed change appear only for a loaded sediment-enabled run.

**View > Flow particle tracks** defaults off. It animates short particle trails
through the current frame's frozen velocity field while result playback is
paused. Each particle has a crisp bright head without a halo and a longer, tapered
tail that fades toward older positions like a shooting star. Playback or leaving Results stops the trails. These are visual flow
indicators, not a sediment/particle-transport calculation. Settings controls
their count (default 1000).

Depth uses Blues
with exactly zero transparent. All result variables now use this zero-only rule;
nonzero depths below 0.001 m remain colored. Velocity uses Reds. Mesh is a layer
in both modes; Results uses adaptive mesh by default, configurable in Settings.
**Export** above the map opens a dialog for one selected time frame as PNG or
an MP4 video, all frames by default or an inclusive frame range. Choose FPS and
resolution. Video uses a fixed color range and equal frame durations. PNG/MP4
exports omit the separate paused-field particle trails. Exports use the
current map extent and visible layers. **Scale, North arrow, Legends and Time**
have separate checkboxes at bottom left; turn them off for clean exports.
Existing export files are not overwritten, and cancellation leaves no partial
published output. Manual result ranges, palettes and opacity also apply to media
exports. Plot windows include a floppy-disk **Save plot as PNG** button. PNG and
plot CSV filenames include the probe/profile, project and actual plotted field
(for example `Probe_1_River_Water_depth.png`). They retain that field name even
if the map variable is changed after opening the plot.

Multiple named probes and profiles
can be selected, edited with draggable map handles and deleted. Right-click ends
editing. CSV exports include all nine displayed variables; profiles also include
integrated discharge, with the other variables length-averaged along the profile.
Default CSV names include the measurement and project names. Plots use red lines
and a zero reference with 10% upper headroom; signed variables retain negatives.
Boundary flow uses accepted numerical
flux ledgers; the retained prepared inputs are needed to identify those faces.
A cell-centered transect estimate is never substituted for a boundary ledger.
Results navigation contains **Probes**, **Profiles** and **Boundary flow**.
Variable selection, frame arrows, ▶/⏸, the frame slider and Export
are in the bar above the map, alongside Background. There is no Load Results
button: opening the project recovers its saved/latest result output.

## Project file and data folder

Inputs, current parameters, parameter revisions, preparations, runs, logs and
checkpoints remain under the data folder referenced by the `.braids.json` file.
Use File > Open project or Open recent to resume work and load results. Current
`braids_project.json` files inside existing data folders can also be opened.
Analysis points/lines and geometry
reviews are stored with the session when saved. Save project as saves a new
setup and its inputs; it does not move or erase the original results/history.
Active file references are relative to the project folder. Imported originals
and sidecars are retained in `model_layers/imports`; runnable DEM assets are
authenticated by the existing folder format.

**Save also works with incomplete inputs or invalid settings.** Editable values,
including invalid text, are stored as a versioned draft under
`model_layers/desktop/EDITOR.json`. Opening the project file restores that draft.
Prepare/Run stay strict: fix the errors and Save successfully before execution.
A saved invalid draft cannot execute the previous valid project accidentally.

Move **both** the `.braids.json` file and its data folder together. The JSON
contains `schema`, `data_folder` and the editable `project` configuration.
File > Edit project JSON opens the built-in editor. External saves to that file
also reload the UI automatically when clean; unsaved UI changes require confirmation.
Change `project.settings` for run parameters and `project.model.request` for
physics/backend choices. Derived model identities and boundary resolution are
recomputed with the shared resolver; invalid combinations are not admitted.
Save after reloading changed JSON, then Prepare/Run as usual. Do not hand-edit
authenticated files inside the data directory.

**Checkpoint to resume is always empty when opening a project.** Select one
explicitly only to resume matching settings/backend/runtime. A mismatched
selection is rejected before launch with guidance to clear it for a fresh run.
The worker still authenticates the complete checkpoint payload.

Only the current strict v22 project and authenticated checkpoint formats are
accepted. Importing an older project does not migrate it. Opening a project does
not start a simulation. Create a new project or open the updated bundled demo to
test preview16; older files remain untouched. v21 projects/preparations and old
checkpoints are rejected, not silently migrated. The new required `settings.solids`
list is empty for projects without obstacles.

Preview16 adds **Internal conditions** under the alignment button in **BC & IC**.
Choose Polygon or Ellipse, then **Add solid**. For an ellipse, click two opposite
corners; its saved wall geometry is a 64-sided polygon. Choose a solid in the list
to edit its vertices or remove it. Right-click exits editing. Solids must be
strictly inside the domain and cannot overlap or touch each other. Their interiors
are excluded from every mesh level; their outlines are impermeable walls for
water and sediment, not erodible banks. Fine cells obey the same minimum-area and
valid-shape requirements as breakline meshes. Impossible constraints are rejected.
The **Solids** layer checkbox changes visibility only. Preparation and mesh-preview
cache identities include the solids, so editing one requires new preparation.

Zoom to domain, pan/zoom, measuring, appearance and layer visibility remain
available during preparation and execution. Existing result layers can still be
viewed with live output disabled; this does not enable new live output. Model
geometry/parameters remain locked during a process.
See the [solid-obstacle and UI report](../validation/sediment/river_morphodynamics/reports/desktop_solids151/REPORT.md).

Preview15 fixes dynamic-AMR wall-history matching at large map coordinates.
Coordinate-roundoff handling covers both supporting lines and shared endpoints;
real boundary changes are still rejected. The reported natural-river case passes
20 actual mesh updates on CPU/GPU, with a matching restart and closed-wall budgets.
See the [repair report](../validation/sediment/river_morphodynamics/reports/desktop_amr150/REPORT.md).

## Local environment and packaging scope

This preview runs with `desktop/.venv`, using the existing scientific Python
installation on this PC and an independent copy of the qualified runtime at
`desktop/runtime`. It does not need a QGIS process, QGIS Python or a compiler.
The NVIDIA GPU route needs compatible NVIDIA drivers and the installed CuPy/CUDA
environment. CPU execution does not require an NVIDIA GPU.

This is **not yet a self-contained installer for an arbitrary clean Windows PC**.
After the interface feedback, distribution needs a bundled Python/GDAL/native
runtime, dependency/license inventory and tests on a clean Windows machine.
The existing QGIS installation is preserved but is not the target of further UI
development.

For development, `setup.ps1 -Python <scientific-python.exe>` creates an isolated
Qt environment that shares the selected installation's scientific packages and
stages the qualified runtime identified by `PROVENANCE.json`. `check_preview.py` exercises the assembled
application with its own synthetic inputs, saving screenshots and a result
receipt under `desktop/evidence`. Do not replace the runtime with an unqualified
checkout or mix its source namespace with another release.

## Upstream attribution

The generic map, terrain presentation, theme, styling and related helpers in
`braids_desktop/_avac` derive from cgotelli/AVAC-Desktop commit
`d6ac7127e41d62739ce3cafc6b1eea7635c2a6da` under AGPL-3.0-or-later. Its license is
retained in `licenses/AVAC-Desktop-AGPL-3.0.txt`. Adaptations replace scientific
imports, independent preferences, domain terminology and rectangular raster
display handling. BRAIDS-specific project, workflow and solver connections live
outside those generic helpers. Preserve the source and license when distributing
this derived interface. Native solver notices remain in the authenticated runtime.

## Next milestone

Run a short matched open-channel widening comparison against locked unmodified
FLOW 2DH. Compare bank sharing OFF/ON, raw bed sections, bank/wet outlines and
budgets; keep surveyed toe and retained-block extensions separately named.

## Immediate next step

Freeze identical initial bed, water and finite sediment stocks for MPM MF1;
compare a few accepted transfers before a bounded widening run and MF25 contrast.
For local testing now, use `desktop/examples/banks224` and its CMD launchers.
A clean-Windows installer remains separate work.
