# Alpha113: less frequent AMR and protected dry entrances

New projects assess AMR every 1 physical second; the Mesh field accepts larger
values. Hydraulic substeps and forcing/output clocks remain independent and
still enforce the existing convergence and conservation checks. Saved project
intervals are not silently overwritten.

Open boundaries retain the configured neighbour buffer even while dry, so
water has fine cells available before the first scheduled assessment. Terrain
refinement preserves existing sub-dry water traces with zero momentum and
transported inventory. It does not add a water film or allow unsafe coarsening.
CPU and GPU use the same policy and native transfer arithmetic.

See the alpha113 interval report for matched Test1 comparisons, failed naive
clock-only trials, restart and QGIS checks. The short test is not a guarantee
that an arbitrary long interval is safe for every flood-front speed.

# Alpha112: depth-only AMR and persistent hydraulic timestep control

The shared CPU/GPU semi-implicit solver remembers accepted hydraulic substep
sizes, grows cautiously, and checkpoints/rolls back that history. Internal
retry counts and the next hydraulic timestep target are visible in run progress.
Prescribed normal/critical section roots reuse immutable dependencies during
each nonlinear solve; interior-dependent outlet conditions still update.

AMR now refines wet cells plus a neighbour-ring buffer. All physical boundaries
remain finest and dry coarsening is delayed. Simple hydrodynamics starts dry
interiors coarse. No hydraulic acceptance tolerance or sediment formula changed.
Project format v13 is required; old projects/preparations/checkpoints are rejected.
Sediment remains off by default. See the alpha112 optimization report for short
Test1 timing, accuracy, remaining wet/dry retry limitations and release checks.

# BRAIDS 0.1.0 alpha 106

Fixes one-level preparation in the combined CPU/GPU workspace. A single fixed
level now uses the same cut-cell geometry as its unrefined input, without
passing an invalid one-level request to enabled AMR settings. Numerical kernels
and sediment-default switches are unchanged. Two-level/adaptive regression,
one-level regular/cropped walls and restart remain checked.

Re-prepare inputs after updating; old worker checkpoints deliberately retain
their source-identity restriction. Alpha 105 remains available for old runs.
The short low-freeboard bank comparison is development evidence, not validation
of fully developed braided rivers or equivalence at all wetting fronts.

# BRAIDS 0.1.0 alpha 105

Alpha 105 packages the combined adaptive CPU/GPU river worker and exposes an
explicit opt-in QGIS workspace with DEM preparation, live mesh-epoch results,
animation, cooperative cancellation and authenticated checkpoint continuation.
The precompiled dependency closure needs no compiler or checkout. External Python
with NumPy/GDAL (and CuPy for GPU) is still required.

Sediment transport and its ordinary feature switches start unchecked. The new
workspace also starts disabled. Existing project settings are preserved.
The workspace uses explicit augmented-f-wave SSPRK2, a common clock and MF1;
it does not replace the fixed-grid implicit route. Read
[ADAPTIVE_RIVER.md](ADAPTIVE_RIVER.md) for setup, scope and Windows requirements.

The current native executable was rebuilt from the existing current sources.
Delft3D sources and reference binaries remain unchanged and are not packaged.

# BRAIDS 0.1.0 alpha 104

Alpha 104 pairs QGIS with the q10.8 external GPU worker for the first bounded
desktop acceptance of the recursive bank products. Current v54 QGIS decks keep
their complete method and execution identities; the worker now exposes their
unchanged physical v38/v40 geometry payload to the frozen, authenticated
mode-2/mode-3 zero-step initializers through a fail-closed, content-addressed
orchestration adapter. The original and adapted deck hashes are both recorded,
and the durable adapter is reused for later AMR operators and restart. Compact
private bootstrap names keep this path below legacy Windows limits without
changing any public file or identity. No native initializer, CUDA physics
kernel, or Delft3D source was changed.

Install
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.104-QGIS-windows-x86_64.zip` and use
the authenticated q10.8 external GPU worker.

# BRAIDS 0.1.0 alpha 103

Alpha 103 separates the schema-v2 sediment-method identity from the hydraulic
time-integrator contract during GPU preparation. The strict public fixed-grid
profile continues to require Plain `implicit_fwave_midpoint`; the independently
qualified recursive bank-AMR route now correctly retains Plain SSPRK(3,3).
This removes a contradictory preflight that made the recursive route impossible
to prepare through the normal QGIS button after project save.

Install
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.103-QGIS-windows-x86_64.zip` and use
the unchanged authenticated q10.4 external GPU worker.

# BRAIDS 0.1.0 alpha 102

Alpha 102 preserves an explicitly disabled bed-slope correction as an explicit
saved method identity. Loading and resaving an active bed-load project with the
slope switch off no longer relabels that user choice as an automatically
resolved method. This closes the remaining numerical-metadata round-trip found
by the M21 installed-QGIS recursive-bank acceptance.

Alpha 102 was an intermediate build: its clean installed-QGIS test exposed the
schema-v2/integrator coupling repaired by alpha 103.

Install
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.102-QGIS-windows-x86_64.zip` and use
the unchanged authenticated q10.4 external GPU worker.

# BRAIDS 0.1.0 alpha 101

Alpha 101 completes the saved recursive-project repair. In addition to
restoring the sediment master switch before saved process flags, external GPU
worker verification now rebuilds its route menu atomically. The temporary
empty menu can no longer clear a saved bank/bedform composition while the
worker's authenticated variants are being populated.

Alpha 101 was an intermediate build: its clean installed-QGIS test exposed the
explicit-off provenance round-trip repaired by alpha 102.

Install
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.101-QGIS-windows-x86_64.zip` and use
the unchanged authenticated q10.4 external GPU worker.

# BRAIDS 0.1.0 alpha 100

Alpha 100 fixes saved sediment-project restoration after the alpha 99
default-off UI change. New projects still start with sediment transport and
every optional sediment process disabled, but opening a saved project now
restores the sediment master switch before its bank, avalanche, secondary-flow
and bedform selections. This prevents valid recursive bank/bedform GPU projects
from being silently rewritten with their saved physics disabled.

Alpha 100 was an intermediate build: it repaired initial project loading, but
the M21 desktop gate found a second transient route-menu reset during external
worker verification. Alpha 101 supersedes it.

# BRAIDS 0.1.0 alpha 99

Alpha 99 completes M20 clean QGIS packaging around the v118 recursive-worker
contract. The Compute backend and external-worker controls are now in General.
Sediment transport and every optional sediment process start disabled; users
explicitly enable the master switch and the required processes. The old Phase 2
and Phase 3 development panels were removed from the modelling UI while their
authenticated manifests remain packaged release evidence. Accordion headings
now use 11-point text.

On Windows, BRAIDS searches bounded standard locations for the newest extracted
worker and a likely external CUDA Python, remembers verified paths, and provides
**Detect Windows GPU setup** to repeat discovery. Detection only fills paths: it
never selects GPU, bypasses package authentication, or suppresses the FP64 CUDA
probe. This PC's clean installed-profile smoke passed with QGIS 3.40.11, CuPy
14.2.0 and an NVIDIA GeForce RTX 5080.

Install
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.99-QGIS-windows-x86_64.zip` and extract
the companion worker
`dist/gpu-worker/BRAIDS-GPU-0.1.0-q10.4-windows-x86_64.zip` before verification.

# BRAIDS 0.1.0 alpha 98

Alpha 98 completes the Q10 desktop delivery. It retains the alpha 97 product
contract and pairs it with the authenticated product-only GPU worker at
`dist/gpu-worker/BRAIDS-GPU-0.1.0-q10.2-windows-x86_64.zip`. That worker admits
only `integrated`, `live_normal_critical_transport_ssp33` and the `auto` route
resolver. Research bank and AMR variants remain explicit fail-closed entries;
the package does not weaken or replace their stale frozen evidence.

On this PC, the extracted worker passed package authentication and an external
FP64 CUDA preflight using Miniconda Python, CuPy 14.2.0 and the NVIDIA GeForce
RTX 5080. Install
`dist/windows-x86_64/BRAIDS-0.1.0-alpha.98-QGIS-windows-x86_64.zip`.

# BRAIDS 0.1.0 alpha 97

Alpha 97 packages the Q10 portable sediment-product contract. The QGIS project
now records an explicitly selected CPU or `implicit_fwave_midpoint_gpu` backend,
the exact physical and automatically resolved auxiliary methods, provenance,
typed morphology-output settings and the identities of all generating
catalogues. The same fingerprint is authenticated in preparation, GPU worker
capabilities, results and restart. There is no mesh-size backend choice and no
silent CPU or SSPRK fallback.

The Sediment tab generates 69 FLOW-compatible result selectors from the runtime
registry. Sixty-two are supported; seven wave/FM/classic-FLOW incompatibilities
stay visible and disabled with their reason. Managed Fildad operations reject
before allocation, and TELEMAC-GAIA research methods stay labelled outside
Delft3D4 coverage. A component kernel is still not public product admission:
unconnected methods fail closed with a stable error category.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.97-QGIS-windows-x86_64.zip`.
The external GPU worker remains a separate authenticated source package with a
separate CPython/CuPy/CUDA environment.

# BRAIDS 0.1.0 alpha 96

This release installs the experimental GPU **Implicit f-wave midpoint** as an
explicit user choice. It retains BRAIDS' augmented f-wave spatial residual and
advances the unchanged MPM/Exner operator once from the hydraulic midpoint;
sediment therefore remains weakly coupled rather than entering a monolithic
implicit solve. The selected method, schema-1 identity and schema-2 predictor
state are authenticated through preparation, launch, checkpoints and restart.
BRAIDS never chooses it from mesh size and never falls back to an explicit
integrator or to CPU.

The admitted scope is deliberately narrow: the common plain fixed-grid
transport route, safely fully wet low-Froude flow, constant boundary forcing,
nonnegative Manning friction, MorFac 1, and either a static bed or weakly
coupled MPM law 2 with 1--32 fractions and at least two bed layers. Bed-slope
and hiding corrections are off. Wet/dry, AMR, suspension, secondary flow,
viscosity, banks, avalanching, bedforms, adaptation and time-varying forcing
reject before launch for this integrator. The time step remains the user's
choice.

The production-path adoption gate passes on an RTX 5080. In its authentic
128-cell, two-fraction MPM trajectory, a 20-times larger time step used 2
steps instead of 40, measured a 22.276-times warmed trajectory speedup, and
differed from SSPRK(4,3) at 0.002 s by at most 0.012315 mm in water depth and
0.000006763 mm in bed elevation. This is a small-case measurement, not a
reach-scale speed promise. Six constant-boundary cases cover normal-depth,
depth, critical, stage, total-Q and legacy-discharge controls with Manning
0.03; maximum water-budget residual was 4.743e-18 m3 and maximum class-budget
residual was 1.468e-13 in native inventory units. Rejected trials,
`commit=False` and cancellation preserve all 361 reachable GPU arrays, while
an uninterrupted six-step trajectory is byte-identical to a three-step plus
schema-2 restart.

Pair alpha 96 with the immutable v26 source worker at
`validation/g2_general_transport/runs/portable_integrated_worker_20261008_v26`.
The separately installed worker does not include a Python environment, CUDA
wheel, validation input/output or Delft3D source. Its ZIP SHA256 is
`c5de6310cc6fcc63fad37cee668ebee0b1d592fef4968c8b4dc9f077aa42db8d`;
its package-manifest SHA256 is
`0037d8b60844ee5ee4c52c8c37a8bae43d9c1fcd4560c19b35edcc107e35d3a0`.
The adjacent package receipt binds both identities. Delft3D source and all
frozen reference runtimes remain unchanged.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.96-QGIS-windows-x86_64.zip`
through **Plugins â†’ Manage and Install Plugins â†’ Install from ZIP** after
closing every QGIS process that could hold the existing plugin files.

# BRAIDS 0.1.0 alpha 95

This release adds the explicit law-9 fixed-face inlet-bed contract used by the
Yen--Lee bend integration.  A new immutable native exporter v5 records both
`fixed_boundary_bed_mode` and `fixed_boundary_bed`; current law-9 execution
requires and authenticates those arrays.  Historical v3/v4 snapshots remain
readable as historical evidence, but cannot be routed as a current accepted
law-9 run.  Native exporter v4 and every older prepared/run artifact remain
unchanged.

The paired 18,000 s CPU and GPU v2 runs completed with matching accepted-step
history through all comparison gates.  Their final scientific score is
reported honestly as **FAIL**: CPU/GPU parity, conservation and the relative
TELEMAC RMSE gate pass, while the preregistered pooled and 90-degree absolute
morphology RMSE gates fail.  This release therefore closes the engineering and
provenance integration; it does not claim successful physical validation of
the bend response.

The zero-step AMR initializers were rebuilt under new immutable runtime names
against the current shared dependency closure.  This is a dependency
re-freeze, not a numerical-method or physics change.  The current general-AMR
exporter executable SHA256 is
`2ed65134b3891ed5b279a86fb308582fb20cd8331cb562c5f7c1c2f77a76bb5d`;
its build-receipt SHA256 is
`66847199f0d2b17fd2d0e1fac0ead79f5c978c0518707d93694575497c1afb71`.
The older validation-only G4 v4 host-catalog route is not wired into QGIS Run;
because its receipt predates this dependency closure, it now rejects explicitly
instead of being treated as a current supported initializer.

Pair alpha 95 with the immutable v25 source worker at
`validation/g2_general_transport/runs/portable_integrated_worker_20261007_v25`.
Its ZIP SHA256 is
`ea8fdbd2147b261d6725322b55958dfa92bec136ecd9dd34d28cd512d10771fc`;
its package-manifest SHA256 is
`8d549875c3935d68e4fd0b406a8a480b0fb8e69393453630dd35829eb1034ca4`,
and its adjacent receipt SHA256 is
`72e9a70edc77c24be578a429724be82f4e16ce8c0efce1c430433092e7a4be19`.
The worker contains authenticated legacy exporter v3 and current exporter v5;
it deliberately excludes v4, reference-model source trees, validation run
products, Python environments and CUDA wheels.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.95-QGIS-windows-x86_64.zip`
through **Plugins → Manage and Install Plugins → Install from ZIP** after
closing every QGIS process that could hold the existing plugin files.

# BRAIDS 0.1.0 alpha 94

This release corrects stress-dependent transverse bed-slope transport for the
Meyer-Peter--Mueller path. The integrated CPU and CUDA solvers now carry the
computed Shields stress ratio into the Delft3D-compatible `ISlope=2`
correction, including the optimized multi-fraction CPU loop and CUDA boundary
traces. Strongly mobile sediment therefore receives the required
`1/sqrt(tau/tau_cr)` attenuation instead of the former unscaled transverse
steering.

The retained 16-state CPU/GPU/source-equation qualification spans stress ratios
1.02--64. All gates pass; maximum corrected-vector relative errors are
`3.753e-16` on CPU and `1.093e-14` on GPU. The older behavior reaches a
`15.746%` vector error in the sweep. The five-orientation native BRAIDS/Delft3D
bed-slope case was also rebuilt and passes all gates. Delft3D source and frozen
historical builds remain unchanged.

Pair alpha 94 with the immutable v24 source worker at
`validation/g2_general_transport/runs/portable_integrated_worker_20261007_v24`.
Its ZIP SHA256 is
`9cda651150e28f9c85ff82309709602c970fec3c488ff2e83cc4fc72c121bed0`;
its package-manifest SHA256 is
`282963b73060178903e733ebd12a10cec254567e515be6b002da82c774bc5165`.
The additive general-AMR v8 namespace leaves v7 unchanged and pins the
corrected CUDA source to both the component receipt and the passing
11,949-to-25,197-cell dynamic-regrid/restart qualification. This is
engineering qualification, not physical-experiment validation.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.94-QGIS-windows-x86_64.zip`
through **Plugins → Manage and Install Plugins → Install from ZIP** after
closing every QGIS process that could hold the existing plugin files.

# BRAIDS 0.1.0 alpha 93

This engineering release closes the public Windows GPU workflow around the
existing numerical methods. The QGIS path now exercises explicit CPU/GPU
selection, cancellable preparation, Run, cooperative Stop, authenticated
checkpoint Resume and Results creation. It reports CUDA FP64 support plus
current free/total device memory; it does not estimate per-case VRAM demand.
Unsupported physics still rejects before launch and there is no CPU fallback.

Long Windows run folders use authenticated short-lived source staging while
retaining the complete published source snapshot in the run. Checkpoints and
result paths use extended-path I/O. Result sidecars and GeoTIFFs use unique
temporary names plus bounded retry for transient Windows sharing violations,
so a failed promotion preserves the last complete artifact.

The paired source worker is the immutable v23 package at
`validation/g2_general_transport/runs/portable_integrated_worker_20261007_v23`.
Its ZIP SHA256 is
`42906ccfecf8ceb7c702e80f73d27506a1d1656077c2e05a67359958da311ec3`;
its package-manifest SHA256 is
`fb057e1a6dd597bc8eb9e6a04bde0fd5067f8573445ffc157b2ab5939c4243a2`.
The fail-closed AMR source pin is the exact production-route v11 receipt,
SHA256 `d0e9f84458a8060770dac958e78c27bcc5bd36460ada38463d42ae3882985c53`.
That qualification retained byte-identical final state/node identities and a
`7.46e-11` transfer residual. The fixed-grid long-path cut/resume test resumed
from step 5 and completed step 67 at `0.067 s` on the selected GPU.

This is engineering integration and recovery evidence from a tiny three-cell
case, not physical validation, reach-scale performance or proof that available
VRAM is sufficient for an arbitrary project. The QGIS ZIP does not contain the
worker, CPython, CuPy, NVIDIA drivers, Delft3D source, validation resources or
Python environments. Delft3D and all frozen references remain unchanged.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.93-QGIS-windows-x86_64.zip`
through **Plugins → Manage and Install Plugins → Install from ZIP** after
closing every QGIS process that could hold the existing plugin files.

# BRAIDS 0.1.0 alpha 92

This release adds the exact recursive GPU-AMR composition of equilibrium
bedform roughness, evolving surveyed physical-toe bank mode 2 and one
parent-owned general avalanche. The native zero-step initializer derives the
initial toe, donor and face mapping from the surveyed bank geometry. During
each accepted macrostep, bedform feedback is evaluated at all three
level-local SSPRK(3,3) capacity stages and once before friction; conservative
fine/coarse replacement and reflux close before toe supply, undercut and
collapse; avalanche then runs exactly once on the synchronized parent state.
The evolving toe map is rebuilt after topology changes and all state is
committed atomically.

The exact route is
`recursive_bedform_mode2_physical_toe_parent_avalanche_v1`. It requires MPM
bedlaw 2/friction mode 1, MorFac 1, one active and one substrate layer,
bedform feedback ON, physical-toe bank mode 2, general avalanche ON,
secondary flow OFF, one constant unsampled zero-offset stage boundary and
recursive 2:1 subcycling. Suspended exchange, adaptation, slope/hiding
corrections, viscosity, persistent bank fronts, schedules and live hydraulic
boundary forcing are excluded. Unlisted combinations reject before output;
backend selection remains the user's choice and there is no CPU fallback.

The immutable initializer kind is
`live-selected-bank-mode2-bedform-avalanche-amr-v1`. It is export-only and
cannot advance native timesteps. Its executable SHA256 is
`0714bad1c8ff2539318e62b56637b8c62ada7e07f44d9ed2b5e062ab56510e57`.
QGIS authenticates the bank-mode discriminator, schema-8 frames, source and
method identity before materializing the animated adaptive mesh, five bedform
rasters and four bank-history rasters. Both prepared and saved-run readers
also require the exact evolving-toe marker and false source/Delft3D modification
provenance flags; altered or missing values fail closed.
The historical seven-adapter checkpoint-receipt exception is pinned to the two
immutable alpha91 distributed-bank mode-3 identity pairs (runtime v2/public v6
and runtime v3/public v7); it does not by itself admit an older preparation.
The installed end-to-end compatibility gate exercises runtime v3/public v7.
A new identity or any missing, added or changed adapter fails closed.

The ordinary-deck public gate completed 20 macrosteps and 21 frames with one
real 359-to-341-cell topology change that retired 24 physical-toe faces. Toe
supply, undercut, 40 collapses, bedform feedback and parent avalanche were all
active. Maximum dune height was `0.03615118309687022 m`; maximum water/class
budget residual magnitude was `8.731149137020111e-11`. All 83 final checkpoint
fields and every overlapping cut/resume frame recovered byte-for-byte.

| Final release receipt | Recorded value |
| --- | --- |
| Public gate result | `validation/g4_bank_amr/runs/recursive_physical_toe_bedform_public_20261007_v3/RESULT.json` |
| Public result / figure SHA256 | `d239e4f6632f38985bd25c4adc2554401424ecbe87a37580c7dcab5a8ac67f87` / `648ea0511fe291db43dbe8a1b6219960b47726eb0f12ca28c7cb0eda979c5026` |
| Public source / method identity | `edf9da009e3d0b9cfe38a7f1d01450d96131618ca651461070d2014596eb9ce4` / `87ae0f281618d2398a6a2d212c0d66ee37de6f8ac73db577db474664ab777944` |
| v20 worker ZIP / manifest SHA256 | `cb1b4827899237b0477a6a6f1dcb095e87b87fa51043dec43ecb05e0458d49b0` / `ef0ac1d0516d3a1d1e9cd5cbdfe4d4e3028d5106074b11619099f0fb129fa572` |
| v20 package receipt SHA256 / payload | `25cfd494918c6721fe7ec03c1366399b59e23942f9a61a803493383b947cb13e` / `775 files`, `22,864,719 bytes` |
| Portable v20/v2 receipt / result / report SHA256 | `fa306372449837cf927b541310009d7fd2d57f80deeac3d69d824fb4e52dbff2` / `9ff64b1c0eafe02c22851043368d0bcf1ce8687acdc20ea97e4e269bdcf8f97e` / `14b98e1d08fea8ea0b9cb27b4a45b10be877c67a8f40c6f45ad756f830ffc5cd` |
| Alpha 92 archive integrity | External sibling `.zip.sha256` and recoverable installation receipt; self-hashes are not embedded in packaged notes |
| Installed-profile gate | External post-build alpha92 identity/readback receipt; not embedded in this archive |

Pair alpha 92 with the authenticated flat-layout v20 source worker at
`validation/g2_general_transport/runs/portable_integrated_worker_20261007_v20`.
The isolated portable v2 gate recovered all 83 final checkpoint fields and 22
overlapping frames byte-for-byte; all 286 recorded run-source entries were
package-backed. Checkout-QGIS host readback also passed before release build.
The QGIS ZIP does not bundle that worker, CPython, CuPy or NVIDIA drivers. This
is engineering integration and recovery evidence, not physical validation,
calibration, long-duration morphology or reach-scale performance. Delft3D
source and frozen references remain read-only, unchanged and unbundled.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.92-QGIS-windows-x86_64.zip`
through **Plugins → Manage and Install Plugins → Install from ZIP** after
closing every QGIS process that could hold the existing plugin files.

# BRAIDS 0.1.0 alpha 91

This release composes recursive equilibrium-bedform roughness, evolving
distributed-bank mode 3 and parent-owned general avalanche in one guarded GPU
AMR solver path. Bed-load capacity uses the updated bedform roughness at all
three level-local SSPRK(3,3) stages and friction receives one prepared field
per local step. After fine/coarse replacement and reflux, distributed bank
erosion runs once on the synchronized parent state and avalanche runs once
after the bank transaction. Neither operation is multiplied by AMR levels or
Runge--Kutta stages.

The exact route requires MPM bedlaw 2/friction mode 1, MorFac 1, one active and
one substrate layer, bedform feedback ON, evolving distributed bank mode 3,
general avalanche ON, one constant unsampled zero-offset stage boundary, and
recursive 2:1 subcycling. Suspended exchange/state, secondary flow, adaptation,
slope/hiding corrections, viscosity, persistent bank fronts and schedules are
excluded. Unlisted compositions reject before output; backend selection remains
the user's choice and there is no CPU fallback.

The new native initializer is immutable kind
`live-selected-bank-mode3-bedform-avalanche-amr-v3`. It is export-only and
fails if invoked without its selected-AMR export flag. The public negative
gate returns nonzero with no frame, checkpoint or numerical operator output.
Historical authenticated v2 results remain readable, but v2 cannot prepare or
start a new run.

The ordinary-deck public gate completed 20 macrosteps and 21 frames with one
real 359-to-341-cell topology change. Integrated toe removal was
`5.071749847804835e-06 m3`, maximum dune height was
`0.03615118309924965 m`, and maximum bedform iterations were 56. Water and
two class residuals were at most `1.7763568394002505e-15 m3`,
`3.637978807091713e-11 kg` and `8.731149137020111e-11 kg`. All 42 final
checkpoint fields and every overlapping 40-field cut/resume frame recovered
byte-for-byte. The QGIS host path materialized the animated adaptive mesh plus effective
Manning, dune-height, dune-length, total-roughness and iteration rasters.

| Final release receipt | Recorded value |
| --- | --- |
| Public gate result | `validation/g4_bank_amr/runs/recursive_bank_bedform_public_20261006_v7/RESULT.json` |
| Public result / QGIS-runtime host receipt SHA256 | `d80622e361c2c9cd694a16d2728649288292aad6b9bc1cd6a77a7e2bd0d00d82` / `8a068097716482e60af5efc757bd4994db4f96569272b39cc42337758ef0d55a` |
| Public source / method identity | `7eff401f72a979d079f126bc09f77339cbd1a9afdcb58d19fe9b3a22a11b9f23` / `c8bc2096cd34747f3c4260bcdd271b65116b80868a7ac7e512c4fa89b985aeb0` |
| Export-only native initializer SHA256 | `0cdb01acf5ee49a99db99ff6ec8e47d9410bd3c245a73d2952a5db265416173b` |
| v17 worker ZIP / manifest SHA256 | `cbc7a0d7be3339413fdbc7bb65ad6569c35b017d080417d9591ac8fad3a475c3` / `288db0fb94ba727877666a2c35b39d723527a6a28d33ec13df0843da3ffa2c19` |
| Alpha 91 archive integrity | External sibling `.zip.sha256` and external installation receipt; self-hashes are not embedded in packaged notes |
| Installed-profile gate | External post-build alpha91 identity/readback receipt; not embedded in this archive |

Pair alpha 91 with the authenticated flat-layout v17 source worker at
`validation/g2_general_transport/runs/portable_integrated_worker_20261006_v17`.
The QGIS ZIP does not bundle that worker, CPython, CuPy or NVIDIA drivers. This
is engineering integration and recovery evidence, not physical validation,
calibration, long-duration morphology or reach-scale performance. The next
implementation milestone is recursive surveyed physical-toe mode 2. Delft3D
source and frozen references remain read-only, unchanged and unbundled.

Install `dist/windows-x86_64/BRAIDS-0.1.0-alpha.91-QGIS-windows-x86_64.zip`
through **Plugins → Manage and Install Plugins → Install from ZIP** after
closing any second QGIS process that could hold the existing plugin files.

# BRAIDS 0.1.0 alpha 90

This release adds recursive equilibrium-bedform roughness feedback to the
bounded nonbank GPU-AMR route. The active branch recomputes bed-load capacity
at all three level-local SSPRK(3,3) stages and prepares one updated friction
field before each level-local step. Accepted-frame dune height, dune length,
total roughness, effective Manning coefficient and iteration diagnostics are
reconstructed lazily; they are derived values and are not added to checkpoint
schema 6.

The exact admitted pair uses bed-load law 2, MPM friction mode 1, slope and
hiding modes 0, suspended exchange off, secondary flow off, adaptation off,
zero horizontal viscosity, general avalanche off, all bank modes off, MorFac
1, one active plus one substrate layer, one unsampled zero-offset mode-1 total
discharge inlet and one constant mode-2 stage outlet. The bedform flag may be
OFF for the matched control or ON for feedback. Schedules and all unlisted
compositions fail before execution; the backend remains an explicit user
choice and there is no CPU fallback.

The source-tree gate completed 20 macrosteps and 21 frames with two automatic
topology changes, no rejected step, 240 active bedform evaluations and exact
26/26 final and admitted checkpoint recovery. Maximum water and class
conservation residuals were `6.661338147750939e-16 m3` and
`1.3642420526593924e-12 kg`. The paired feedback changed the state by up to
`0.08308718320961361`, depth by `0.0010550842024846796 m`, and momentum by
`0.0015796877222612193 m2/s`; maximum reconstructed dune height was
`0.05257670797561877 m`.

| Final release receipt | Recorded value |
| --- | --- |
| Public gate result | `validation/g4_bank_amr/runs/recursive_bedform_cli_20261006_v3/RESULT.json` |
| Public result / report / figure SHA256 | `ccfa63980196f1f2db8d13cc594c968fd7ae15d8fed974d5d4b801ec912227a4` / `b2a205a8a687974d7a37acef06fa71fbe85532337ad9adcca368200aed09204e` / `d678fa024eb28a1c76555918166a7e626471b75c6e14d2e5bbf4a996098d4508` |
| Public ON source / method identity | `9a3237d6330cbda16de59e182406578a750122785c48fadf11f911099dff3d27` / `ffa046acd1b0547c09e81a5649189b1a8d9526dacf2a046dcbad98748c2adbd4` |
| v15 worker ZIP / manifest SHA256 | `f7c1e24d6061ffad6abf681783c018c8729048e55a23be9419a1c51a98f5dcc6` / `672254c6483f64f52a13340def51c861ff398517be456d26803722d8fc68e4bd` |
| Portable gate / report SHA256 | `0a3ba2bea5904337a8c3103b1bcb8be50d113f5cec3a1c0fb2e50081e2ae7992` / `592bb1c7ff84f88e392f982f48910c915827e040f7cc372c32a92785dea900df` |
| Portable ON source / method identity | `7c64ddb8cd14a24b68bd7a16317c23d55ac3b2452aa240fbf439788f4089f349` / `46b09ca93ac18cd4b607459b89e0ac3c9acb62a85e26de37df9cbe4ca8869b21` |
| Alpha 90 archive integrity | External sibling `.zip.sha256` and external validation receipt; self-hashes are not embedded in packaged notes |
| Installed-profile gate | External post-build installation/readback receipt; not embedded in the archive |

Pair alpha 90 with the authenticated flat-layout v15 source worker at
`validation/g2_general_transport/runs/portable_integrated_worker_20261006_v15`.
The QGIS ZIP does not bundle that worker, CPython, CuPy or NVIDIA drivers. This
is engineering integration, packaging and recovery evidence—not physical
validation, calibration, long-duration morphology or reach-scale performance.
Recursive bedform-plus-viscosity/avalanche/bank compositions, surveyed
physical-toe combinations, cohesive sediment and unlisted controls remain
outside this release. Delft3D source and frozen references remain read-only,
unchanged and unbundled.

# BRAIDS 0.1.0 alpha 89

This release adds positive native mode-1/geometry-1 velocity-Laplacian
viscosity to the bounded nonbank recursive GPU-AMR route. The root transaction
applies the existing resident SSPRK2 diffusion operator once to the complete
synchronized leaf graph after fine/coarse reflux and before optional general
avalanche. It is never repeated per level or per hydraulic Runge--Kutta stage.
A rejected post-viscosity trial restores physical arrays, diffusion-cache
bindings and diagnostics before an exact retry.

The exact admitted scope is MPM bed-load law 2, Van Rijn exchange mode 2,
prognostic secondary flow, normal-depth inlet, critical outlet, banks off,
bedform feedback off, and general avalanche either off or on. Positive
viscosity requires the velocity-Laplacian formulation and the authenticated
v14 worker's explicit v6 capability. Historical zero-viscosity preparations
remain bound to their original compatible workers; physics and backend are
never silently changed.

The native diffusion law uses pairwise velocity-face flux divergence and a
monotonically limited area-weighted velocity-L2 functional. For uniform depth,
the closed focused test conserves integrated momentum to numerical tolerance.
At variable depth the local depth multiplier means integrated momentum is not
exactly conserved, and this release makes no broader momentum or physical-energy
claim.

The public seven-class/four-layer full/cut/fresh-resume gate completed 20
macrosteps, 21 frames and two real topology changes with one viscosity call per
macrostep, active wet coarse/fine viscous faces in the initial topology, and
correctly inactive dry coarse/fine faces in the refined topology. All 28 final
and admitted-cut checkpoint fields and every overlapping 26-field frame match
byte-for-byte. Maximum water, class, secondary and exchange residuals are
`8.882e-16 m3`, `1.364e-12 kg`, `4.235e-22` and zero.

| Final release receipt | Recorded value |
| --- | --- |
| Public gate result | `validation/g4_bank_amr/runs/recursive_general_viscosity_cli_gate_20261006_v3/RESULT.json` |
| Gate `RESULT.json` SHA256 | `016325ee372eb474354c4e1a7b40a655d42543742a055aa0473d385294fa3a5e` |
| Runtime source / method identity | `496feecde93ff959de2d36be79f6a692a913b7b559654a220ad79f06962cb20b` / `90baea3761787a59691d77040c8118b952b5edd5be319579e53d04497622f42d` |
| v14 worker ZIP / manifest SHA256 | `d98d6edcade0e201f0cb06d45b7ca1de100dde1ab4688ef09727d9a89128f0a5` / `76c33e99eb582ebede289cb2cf08d15b2a9181356c800320b9fb73e258ca8274` |
| Alpha 89 archive integrity | External sibling `.zip.sha256` and external validation receipt; self-hashes are not embedded in packaged notes |
| Installed-profile gate | External post-build installation/readback receipt; not embedded in the archive |

Pair alpha 89 with the authenticated v14 source worker under
`validation/g2_general_transport/runs/portable_integrated_worker_20261006_v14`.
The QGIS ZIP does not bundle that worker, CPython, CuPy or NVIDIA drivers. This
is engineering integration/recovery evidence, not physical validation,
calibration, long-duration morphology or reach-scale performance. Recursive
bedform feedback, bank/avalanche composition, surveyed physical-toe mode 2,
cohesive sediment and unlisted combinations remain outside this scope.
Delft3D source and frozen references remain read-only, unchanged and unbundled.

# BRAIDS 0.1.0 alpha 88

This release adds opt-in domain-wide general avalanche to the qualified
nonbank recursive GPU-AMR route. Avalanche is owned by the root transaction:
it runs once on the synchronized selected-leaf state after all fine/coarse
clocks close and conservative reflux completes, and before budgets and the
accepted state are installed. It is not repeated independently on every AMR
level. A post-avalanche rejection restores the complete pre-step state,
registers, clocks, ledgers and avalanche history before an exact fresh retry.

The admitted combination remains deliberately narrow: MPM bed-load law 2,
Van Rijn suspended exchange mode 2, prognostic secondary flow, full-leaf
bedload adaptation, a normal-depth inlet and critical outlet, no banks, zero
horizontal viscosity and no bedform feedback. Avalanche-OFF recursive jobs
remain compatible with historical workers. Avalanche-ON jobs require the
authenticated v13 capability and fail before launch with an older or altered
worker. No backend, integrator or physical process is silently substituted.

The public cold full/cut/fresh-resume gate uses seven sediment classes, four
bed layers, genuinely mixed balanced 2:1 meshes and an automatic topology
change. Its final receipts are summarized in the
[recursive general-avalanche integration report](../validation/g4_bank_amr/RECURSIVE_GENERAL_AVALANCHE_INTEGRATION_REPORT.md).
The sealed numerical and worker receipts are recorded below. Alpha 88 archive
and installed-profile identities are deferred to external receipts because a
packaged release note cannot contain its own archive or manifest hash:

| Final release receipt | Recorded value |
| --- | --- |
| Public gate result | `validation/g4_bank_amr/runs/recursive_general_avalanche_cli_gate_20261006_v1/RESULT.json` |
| Accepted macrosteps / end time | `20` / `0.04000000000000002 s` |
| Mesh changes / authenticated operators | `2` / `2` |
| Avalanche substeps, minimum / total | `1` / `20` |
| Maximum water / class residual | `7.771561172376096e-16 m3` / `1.3642420526593924e-12 kg` |
| Exact final / admitted checkpoint fields | `28` / `28` |
| Gate `RESULT.json` SHA256 | `43ff44492f296d1a571f82f0fcff004b39275b3df4eb34970962f2d1eae8862f` |
| v13 worker ZIP / manifest SHA256 | `10deecbafcd2b8b1c680f336368959bde306a63934bd2354bcb3bc3135ebebb9` / `9b646653f022e508122c31705b7a3a32ac4095e9e581eb20529d339d85fe12cc` |
| Alpha 88 archive integrity | External sibling `.zip.sha256` and external validation receipt; self-hashes are not embedded in packaged notes |
| Installed-profile gate | External post-build installation/readback receipt; not embedded in the archive |

These gates establish engineering integration, conservative accounting,
wet/dry-state finiteness and deterministic recovery for the named short scope.
They are not calibration, experimental agreement, long-duration morphology,
a reach-scale performance result or physical validation of a bend or braided
river. Positive-viscosity, bedform-feedback, bank/avalanche composition,
surveyed physical-toe recursion, cohesive sediment and unlisted combinations
remain outside this recursive scope.

Pair alpha 88 with the authenticated v13 source worker under
`validation/g2_general_transport/runs/portable_integrated_worker_20261006_v13`.
The QGIS ZIP does not bundle that worker, CPython, CuPy or NVIDIA drivers. The
user still selects CPU or GPU explicitly; unsupported GPU work is rejected and
never falls back to CPU. Delft3D source and frozen references remain unchanged,
read-only and unbundled.

# BRAIDS 0.1.0 alpha 87

This release connects the shared conservative recursive GPU-AMR engine to the
ordinary QGIS Prepare/Run/Results workflow. The user's **Advance finer levels
with smaller timesteps** choice now selects recursive stepping explicitly; it
never selects CPU/GPU automatically and never drops unsupported physics.

The authenticated v12 source worker admits three recursive routes: nonbank
MPM bed load with Van Rijn exchange mode 2, prognostic secondary flow and the
full-leaf bedload-adaptation closure; static or evolving physical-toe bank mode
2; and evolving distributed-bank mode 3. The scheduler supports arbitrary
balanced 2:1 depth, including empty intermediate levels, with fine-replaced
face registers and one conservative reflux per parent clock. Nonbank avalanche,
positive viscosity and bedform feedback retain their broader synchronous AMR
routes but remain explicit recursive exclusions. Surveyed recursive bank mode
2 is also still excluded.

Recursive nonbank checkpoints use schema 6. Physical-toe checkpoints use outer
schema 2 with engine schema 7 before a topology change and mode-bound schema 8
after one; distributed-bank checkpoints retain historical outer schema 1 and
mode-bound engine schema 6 after a topology change. QGIS authenticates level
clocks, registers, toe diagnostics/maps, retired face epochs and final frames
without importing or executing worker code. Historical source workers and
their exact prepared jobs remain readable.

The cold ordinary recursive transport gate completed ten macrosteps with one
automatic 165-to-189-cell regrid. A fresh process reproduced all 28 checkpoint
fields exactly and all 12 overlapping frames matched in all 26 fields. Maximum
water, grain-class, secondary and exchange conservation residuals were
`6.66e-16 m3`, `5.68e-13 kg`, `5.29e-23` and zero. These are engineering and
recovery checks, not bend or braided-river physical validation.

Pair alpha 87 with the authenticated v12 source worker in
`validation/g2_general_transport/runs/portable_integrated_worker_20261006_v12`.
The QGIS ZIP still does not bundle Python, CuPy, NVIDIA drivers or the source
worker. The backend remains an explicit user choice, there is no silent CPU
fallback, and Delft3D source remains unchanged and unbundled.

# BRAIDS 0.1.0 alpha 86

This release connects the qualified four-stage SSPRK(4,3) transaction to the
ordinary fixed-grid GPU Prepare/Run/Results path. It is selectable only for
nonbank MPM bedlaw-2 runs with exchange, bedform feedback, bedload adaptation,
general avalanche and AMR off. Other physical combinations retain plain
SSPRK(3,3), and an incompatible selection fails explicitly instead of disabling
physics or changing backend. The historical
`live_normal_critical_transport_ssp33` value remains a preparation-schema name;
the run manifest and checkpoint bind the actual integrator and stage map.

The authenticated RTX 5080 engineering gate used the same prepared two-grain,
two-layer MPM/secondary-flow input for both methods. SSPRK(4,3) reduced accepted
steps from 61 to 31 and total Euler evaluations, including one deterministic
startup rejection, from 186 to 128. Balanced repeated timing measured a 1.46x
evolution speedup. Maximum bed difference was 0.000267 mm; depth, velocity,
wet-mask and conservation screens passed. A fresh-process resume reproduced all
19 checkpoint fields exactly, while a different-integrator resume was rejected.
This is numerical engineering qualification, not experimental bend or braid
validation.

Pair alpha 86 with the authenticated v9 source worker. The QGIS gate covers
explicit SSPRK(4,3) selection, preparation, the external CUDA process and
completed raster/mesh handoff. The previous v8/alpha85 pair remains recoverable
for exact-source restarts. Python, CuPy and the NVIDIA driver remain external;
there is no automatic CPU/GPU selection or fallback. Delft3D source is unchanged
and is not packaged.

# BRAIDS 0.1.0 alpha 85

This maintenance release closes the recursive GPU AMR result-source contract.
The v8 worker now publishes top-level authenticated copies of every Python
module named by its live-bank source receipt, including `windows_paths.py` and
`gpu_capabilities.py`. QGIS therefore verifies the completed recursive
checkpoint against the exact code that produced it instead of rejecting the
result after a numerically successful run. The check remains fail-closed: a
missing or altered source is still rejected, and there is no compatibility
fallback to unauthenticated code.

Pair this plugin with the separately extracted v8 source worker. Alpha 84 and
v7 remain a recoverable historical pair for their existing prepared jobs, but
v7's recursive bank-AMR result cannot complete the automatic QGIS Results
handoff. The numerical method, augmented f-wave hydraulics, GPU-only execution
selection and Delft3D reference sources are unchanged by this closure fix.

# BRAIDS 0.1.0 alpha 84

The experimental GPU Run tab now offers a versioned slope-aware refinement
selector for synchronous, nonbank general AMR. It requires the authenticated
v7 source worker; a missing capability fails before preparation. New compatible
GPU projects can select it explicitly, while old prepared jobs retain their
legacy selector and exact restart identity. CPU selection remains the user's
choice, with no automatic GPU-to-CPU fallback. This does not change the native
CPU solver or the augmented f-wave hydraulic kernel.

In a short, full-physics seven-grain/four-layer 5% reach, the new selector
avoided a 11,949-to-25,197-to-11,949-cell mesh cycle. At 0.04 s its maximum
bed/depth differences from the legacy trajectory were 0.125/0.107 mm; water
and per-class budgets closed to roundoff. The measured short cold route was
3.76 times faster and peak device use was 0.89 GiB lower on the test GPU.
A separate coupled pulse did refine through three topologies and reproduced
the cold final state and ledgers bitwise after fresh-process restart. These
are engineering tests, not physical or long-duration morphology validation.
Parsed operator-package caching also reduces repeated authenticated lookup
time on the measured reach; it does not change solver physics.

The QGIS ZIP still does not bundle Python, CuPy, CUDA drivers or the GPU
worker. Pair it with the separately extracted v7 worker and a compatible
external GPU Python, then press Verify in QGIS. Keep the v6 worker with old
prepared runs: source and selector identities must match on restart. Delft3D
source was not modified or packaged.

# BRAIDS 0.1.0 alpha 82

The faster CPU time integrators are now selected automatically within their
tested scope: second-order SSPMS6 for eligible fixed-grid hydraulic-only runs,
and SSPRK(4,3) for eligible fixed-grid MPM runs, retaining secondary flow and
viscosity. AMR, bank processes, suspended exchange, adaptation and other
unqualified combinations retain the legacy SSPRK(3,3) method. The execution
log records the actual selection and any fallback. GeoClaw's augmented f-wave
solver and sediment reconstruction remain unchanged.

Existing project CFL values and the new-project CFL default 0.45 are preserved;
the integrator does not silently override them. The matched bend rerun explicitly
uses Euler-stage CFL 0.8. The earlier paired short bend qualification measured
about 61% less computation time with SSPRK(4,3) at CFL 0.8 versus SSPRK(3,3) at
0.4; that is not a claim about the pending full cold-start trajectory.

Checkpoints now record the resolved method and reject a different integrator on
resume. SSPMS6 restarts rebuild their history using SSPRK(4,3), so their restarted
trajectory is not promised byte-identical. Old checkpoints without method metadata
are legacy-only and still require the original matching build/input fingerprint.
For diagnostic CLI runs, BRAIDS_TIME_INTEGRATOR accepts auto, legacy, ssp43 or
multistep6; unsupported forced methods fail rather than dropping physical terms.

The new 20-morphological-hour bend comparison starts with a fresh 300-second
hydraulic warm-up on the original bed, then MF25 mobile-bed evolution. It checks
3 mm bed RMS and 5 mm maximum bed difference at every saved time against the
completed 6.27-hour historical CPU run, plus conservation and spatial patterns.
Until that finishes, full cold-start agreement is not established. MF25 remains
a validation-only input allowance; the plugin keeps its existing MF1--20 guard.

GPU work remains an isolated FP64 research implementation, not a shipped coupled
GPU solver. Delft3D source and runtime remain unchanged and are not packaged.
Restart QGIS and prepare a new run to use this release; old prepared/custom
runtimes and existing results are not overwritten. Earlier qualification manifests
remain historical evidence, not renewed certification of every physical feature.

# BRAIDS 0.1.0 alpha 81

The optimized development implementation is now the default Windows runtime.
It includes reusable Euler workspace, compatible fused preparation, measured
OpenMP loop improvements and cached sediment geometry from the recent milestones.
GeoClaw's Riemann routines, WENO/limited sediment reconstruction and SSPRK3 are
unchanged. Numerical defaults, CFL and the production MorFac 1--20 guard remain
unchanged; the laboratory MF25 input allowance is not shipped.

When prognostic secondary flow is enabled, curvature production now uses the
tested depth-scaled filter with length 2h and relative residual target 1e-4.
This is a model regularization compared with the older unfiltered release,
not just a faster linear solver. It is the same regularization used in the
recent bend comparisons. Small meshes use MIC-preconditioned CG; meshes with
at least 65,536 cells use parallel multilevel-preconditioned CG. Secondary flow
is not automatically enabled. Algebraic secondary flow is unchanged.

The paired 80,342-cell replay previously measured 8.7% less computation time
and 7.0% less whole-process time for the multilevel change alone. These are not
universal speedups. The long cold-start comparison was stopped at the user's
request, preserving its last checkpoint at 840 hydraulic seconds (5 h 50 min
of morphology at MF25). It did not complete the 20-hour target. Developed
large-mesh morphology and multilevel AMR performance remain unqualified.

See `docs/WINDOWS_ALPHA_81_RELEASE.md` and the linked performance audit for
verification, inherited development changes, limitations and runtime identities.
Release checks: 137 passes, one missing historical-fixture skip and one inherited
wet-tailwater outlet assertion failure. The failing case is bitwise-identical
in installed alpha 80, pre-promotion development and alpha 81. It is retained
as an unresolved boundary/oracle discrepancy, not hidden or newly certified.
Prior process qualification manifests are historical evidence, not renewed
certification of every combination in this release. Delft3D source is unchanged
and is not packaged. Restart QGIS and prepare a new run; saved results and
previously prepared/custom runtimes are not rewritten.

# BRAIDS 0.1.0 alpha 80

Wet, local-equilibrium MPM stage outlets now impose the prescribed water surface
through the coupled water/Exner incoming characteristic. The outgoing modal
content and the matching water/sediment boundary trace are retained. Adjacent
equal rectangular cells use bounded inward quadratic capacity reconstruction;
uncertified stencils retain the bounded fallback. No bed smoothing, sediment
deletion, supply retuning or transport-coefficient change is introduced.

The refined 0.125 m straight-channel control passes the predeclared 0.1 mm local
outlet limits: pulse contamination 0.052 mm and curvature 0.042 mm. Coarser-grid
failures remain recorded; this is not a promise of grid-independent accuracy.
The 0.25 m / 300 s Delft3D comparison gives 0.262 mm bed-change RMSE and 0.43%
mean-export difference. Analytical reflection tolerances are unchanged.

Fixed-bed and non-MPM stage handling, critical/weir modes, supercritical-owner
release, adaptation, secondary flow and bedform-roughness closures are unchanged.
An MPM stage boundary is no longer checked against a water-only invariant:
its state is independently verified using a numerical coupled Jacobian.

Delft3D source is unchanged. Expansion duration/grid/timestep qualification is
separate; older process manifests are not renewed by this scoped correction.
See `validation/sediment/synthetic_expansion/reports/outlet_characteristic_final_v1/AUDIT.md`.
Restart QGIS and prepare a new run. Existing results and prepared runtimes are
not modified.

# BRAIDS 0.1.0 alpha 79

Stage outlets now evaluate local-equilibrium MPM sediment capacity using the
same boundary depth and velocity as the water flux. Using the adjacent cell
state instead could under-export sediment and create an artificial outlet
deposit. Fraction availability, conservative inventory limits, MorFac,
wetting controls and physical fixed-stage wave reflection are preserved.
Supercritical release, critical/weir modes, other transport laws, adaptation,
secondary flow and bedform-roughness configurations are unchanged.

In the 1 m straight-channel control, the last-40-m bed-volume change moves
from +7.785 to -0.142 m3. Mean sediment export is within 0.73% of the same
section inside a longer reference channel. Physical reflection error at
MorFac 10 decreases from 2.85% at 0.5 m to 0.556% at 0.25 m and 0.167% at
0.125 m. No bed smoothing or transport-coefficient tuning is used.

The audit includes matched-grid Delft3D runs, refinement, nonzero moving-
gravel formula checks, equilibrium, wet/dry, four flow directions, restart,
OpenMP identity and conservation. Delft3D's alternating morphology remains
grid-sensitive: full model equivalence and braided-river qualification are
not claimed, and older qualification manifests are not renewed by this fix.

See `validation/sediment/synthetic_expansion/reports/stage_capacity_refinement_v1/AUDIT.md`
for the acceptance gates, figures, exact scope and next milestone. Restart
QGIS and prepare a new run; existing prepared runtimes/results are unchanged.

# BRAIDS 0.1.0 alpha 78

Removes the midpoint-only local-equilibrium MPM flux buffer beside subcritical
fixed-stage outlets. That buffer bypassed the coupled water/Exner waves and
could amplify a small bed disturbance at MorFac 10. Bounded boundary capacity
reconstruction remains; the coupled interior flux now stays active. No saved
bed smoothing, physical transport coefficient, or prescribed boundary feed
was changed. Non-stage buffers and supercritical-release behavior are retained.

An independent fixed-stage reflection test, rather than a zero-reflection
assumption, verifies the correction. At 0.25 m, the old peak reached 1323 times
the seed; the corrected peak is 0.393 times the seed. MorFac-10 relative L1
reflection errors decrease from 35.9% at 0.5 m to 13.1% at 0.25 m and 5.8% at
0.125 m. The physical reflected peak is about 0.356 times the incident peak.

The paired mobile-bed straight reach is still grid-sensitive, and Delft3D's
cell-scale oscillations grow under refinement. Turning off Delft3D's bedload
upwinding makes that diagnostic worse; it is not a recommended setting change.
This release does **not** qualify full braided-river morphology or renew older
qualification manifests for the changed solver. Prepare a new run to use it.

See `validation/sediment/synthetic_expansion/reports/outlet_buffer_v2/AUDIT.md`
for figures, numerical evidence, limitations and the next milestone.

# BRAIDS 0.1.0 alpha 77

Corrects the sloping, subcritical stage-outlet reconstruction. The outgoing
characteristic and imposed stage now use the reconstructed face bed, with the
matching half-cell gravity contribution. End-cell slope limiting no longer
flattens that boundary. Flat-bed, dry-face and supercritical-release behaviour
retain their existing controls; critical-depth and weir outlet modes are unchanged.

The exact uniform-flow test reduces the final-cell mean depth error from
40.45 to 1.86 mm at 2 m spacing, and from 27.80 to 0.56 mm at 1 m spacing.
Normal-flow tests in all four directions, wet/partly dry lake-at-rest tests,
one/eight-thread identity and existing boundary/morphodynamic regressions pass.
The read-only flux observer is bitwise identical to the production run.

The paired Delft3D audit now matches actual initial staggered-face discharge,
not just velocity inputs. Identical-input compiled MPM kernels agree to machine
precision. Mobile-bed outlet flux and spatial oscillations still need further
validation: this release does **not** qualify the synthetic braided-river case
or establish full Delft3D morphology equivalence.

See `validation/sediment/synthetic_expansion/reports/outlet_corrected_v1/AUDIT.md`
and its figures. Prepare a new run to use the correction; saved results do not
change. The Windows ZIP excludes Delft3D source and raw validation runs.

## Previous alpha 76 correction

Fixes the nonphysical alternating scour/deposition pattern in the steep
synthetic widening case. Local-equilibrium MPM now uses coupled shallow-water
and Exner characteristic face fluxes, including MorFac in the wave speeds.
This replaces sediment-flow-only upwinding in wet interior cells; it does not
smooth the saved bed or change the physical MPM/slope formula.

- Critical/supercritical small-amplitude bed waves stay bounded and converge
  under refinement, for both flow directions. Analytic capacity derivatives
  are independently checked against the production MPM closure.
- The 900 s, 4 m, MorFac 10 expansion changes from +1.035/-1.274 m to
  +0.136/-0.178 m; the matched-resolution Delft3D reference is +0.169/-0.160 m.
  Strong adjacent opposite-sign bed-change pairs fall from 45.4% to zero.
- Sediment inventory limits, physical slope corrections and the established
  inlet reconstruction remain in place. One/eight-thread and restart states
  and boundary ledgers are bitwise identical in the retained expansion test.
- This fix is not a new qualification of the entire braided-river model.
  Other transport formulas, non-equilibrium adaptation, secondary-flow and
  dynamic bedform-roughness paths retain their existing numerical flux.
  Broad graded-mixture and adaptive-mesh equivalence need separate validation.
- Existing qualification manifests describe their original validation scope;
  they must not be read as fresh certification of this changed solver.

# BRAIDS 0.1.0 alpha 75

This build is the gated Windows delivery for the integrated Johannot braided-
river qualification and its reproducible QGIS benchmark preset.

## Alpha 75 integrated benchmark delivery

- Final preset verification resolves the selected H2 material and Phase 4
  profile from their authoritative reports; fixed development defaults cannot
  accidentally certify a different model.
- H3 reporting explicitly compares BRAIDS and locked-source Delft3D wetted
  width, braiding, discharge partition, dominant-channel share, centerline
  movement, avulsion, bar geometry and H2-to-H3 hydraulic evolution.
- Phase 4, the 14,400 s full experiment and same-finest AMR qualification use
  sequential fail-closed handoffs and measured multicore schedules without
  oversubscribing the 32-logical-processor reference workstation.
- AMR screen, scaling and H2 subruns retain atomic execution records and resume
  only from revalidated checkpoints and exact output hashes.
- The Windows archive audit rejects Delft3D source trees, validation run
  directories, NetCDF maps, checkpoints, frame streams and boundary ledgers.
- Final installation is staged, manifest-verified and recoverable; the prior
  QGIS plugin is preserved before the directory swap.
- A final integrated audit fails closed unless fixed-grid convergence,
  deterministic 1/8/16/32-core execution, the blind H2/Delft3D comparison,
  the withheld H3 prediction, Phase 4, same-finest AMR, QGIS equivalence,
  archive exclusions and installed-file hashes all preserve one selected
  material/process lineage.

# BRAIDS 0.1.0 alpha 74

This build removes formatted-text result output as the fixed-grid bottleneck
while preserving exact state, live display, restart and adaptive-mesh behavior.

## Alpha 74 compact fixed-grid results

- Fixed grids above 4,096 cells write the complete double-precision state and
  accepted boundary ledger to a versioned binary stream. The small
  `frames.txt` marker keeps existing completion checks compatible.
- QGIS and the shared validation reader transparently load new binary streams
  or historical ASCII frames. Small test cases and variable-topology AMR runs
  retain their existing ASCII layout.
- The exact 0.025 m, 16,124-cell, 38-state-variable benchmark reduces two
  authoritative frames from 31.9 MB to 9.8 MB and loads them 84.7x faster.
- Total output time falls from about 0.88 s to 0.05 s. Across three repetitions
  at 1, 8, 16 and 32 threads, output is only 0.28--0.32% of runtime.
- Every decoded value, boundary record and live-preview raster is identical to
  the retained ASCII reference across all twelve runs.
- Native performance logging now reports complete output, authoritative state,
  preview packing, preview I/O and checkpoint time separately.
- Preview packing uses OpenMP only above 262,144 native pixels, avoiding team
  overhead on small grids while allowing large domains to use multiple cores.

# BRAIDS 0.1.0 alpha 73

This build makes the experimental benchmark reproducible through the normal
QGIS preparation path while preserving the measured multicore optimum.

## Alpha 73 QGIS benchmark identity and live-inlet preparation

- Physically dry projects now retain a mobile sediment bed when the minimum
  erodible-depth threshold is zero; dry water no longer implies zero sediment
  inventory.
- Live normal/critical inlet preflight records the requested discharge and
  defers stage, opening, velocity and Froude checks to the native evolving-bed
  control instead of attempting arithmetic on an intentionally unresolved
  stage.
- Projects may use a separate high-resolution inlet-section DEM while a compact
  aligned raster supplies model-cell elevations.
- Derived cell-average rasters have an explicit interpretation and a direct
  aligned-cell path that preserves exact elevation bits while avoiding overlap
  calculations.
- Projects may explicitly request compact binary native sidecars. The Johannot
  preset uses them to match the validation deck and reduce loading overhead.
- The Johannot QGIS exporter defaults to 16 OpenMP threads on hosts with at
  least 16 logical processors—the measured optimum for its 16,124-cell grid—
  and fails closed until H2 material and Phase 4 process selection pass.
- Independent verification proves exact initial state, bedrock, cell area,
  process switches and native restart fingerprint across direct and QGIS
  preparation.

# BRAIDS 0.1.0 alpha 72

This build accelerates the fully wet multi-fraction morphodynamic workload and
qualifies exact 32-core execution on the production-size experimental state.

## Alpha 72 exact multicore morphodynamics

- The roughness-corrected MPM closure evaluates common hydraulics once per cell
  for all noncohesive fractions while preserving scalar arithmetic order.
- Mixture statistics are evaluated only when the selected transport closure
  consumes them and dry cells no longer solve an unused D90 percentile.
- Double-precision D90 bisection stops after representable convergence rather
  than repeating unchanged endpoints.
- Windows launchers keep GNU OpenMP workers active between short RK kernels and
  disable unsupported affinity binding. The selected thread count remains
  explicit and user-controlled.
- The portable Windows build uses whole-program optimisation without unsafe
  floating-point reassociation or host-specific CPU instructions.
- On a fully wet 64,496-cell, seven-fraction continuation, 32 cores reduce the
  complete wall time from 30.404 to 7.063 s (4.31x), Euler work from 27.483 to
  4.307 s (6.38x), and bedload capacity from 19.283 to 1.318 s (14.63x).
- Frames, boundary ledgers and checkpoints are byte-identical at 1, 8, 16 and
  32 threads. The smaller 16,124-cell startup can remain faster on fewer cores
  because its kernels fall below the Windows OpenMP crossover.

# BRAIDS 0.1.0 alpha 71

This build makes long morphodynamic runs recoverable, fixes the evolving
normal-depth inlet at reflective section sidewalls, and removes unnecessary
normal-section work without changing established solutions.

## Alpha 71 evolving sections, checkpoints, and fine-grid continuation

- Live production sections now treat their junctions with neighbouring
  reflective domain edges as vertical sidewalls. An evolving endpoint can no
  longer turn into an artificial spill edge and eliminate an otherwise valid
  normal-depth solution.
- A retained 64,496-cell state that failed deterministically at
  `t=40.299284894 s` now crosses the same transition and reaches 40.5 s. The
  new above-endpoint regression satisfies the requested Manning discharge and
  all earlier production-section tests remain exact.
- Every completed output frame now writes a restart checkpoint. Stopping or a
  later hydraulic failure therefore retains the last displayed model state
  instead of discarding the whole run.
- The normal-depth search visits elevation branches in ascending order and
  stops at the lowest admissible root. This preserves the general irregular
  section solution while avoiding speculative work above the selected branch.
- Bedload-only runs resolve the suspended-transport switch once from the full
  forcing schedule and initial/restart inventory instead of scanning every
  suspended field at every SSPRK stage. Any active suspended process still
  restores the complete path.
- The 64,496-cell, seven-fraction run now reaches 60 s on 32 threads and closes
  water to `1.08e-15 m3`; every fraction passes its budget tolerance. Against
  0.025 m, the 0.0125 m result has 0.0907 mm bed MAE and a 3.15% difference in
  natively integrated absolute bed-change volume, so 0.025 m passes the frozen
  grid gate for H2 calibration.
- The exact root optimization is 2.25x faster for the complete one-core pilot
  and 15.26x faster in boundary control. The compact 4,031-cell pilot is fastest
  at eight threads (1.18x versus one); the large confirmation used all 32
  workers and averaged about 14.5 effective cores.

# BRAIDS 0.1.0 alpha 70

This build qualifies the physical dry startup, rejects the coarse experimental
grid, and accelerates the exact bedload-only multicore path.

## Alpha 70 startup, convergence, and native runtime

- The production benchmark starts exactly dry and uses the measured discharge
  through a live normal-depth section; the earlier 2 mm film is historical
  evidence only.
- Error-bounded inlet simplification retains 755 of 8,173 survey nodes with a
  0.25 mm hard vertical bound. Normal stage changes by at most 0.071 mm and the
  two-second native result by at most 2.12 micrometres.
- Complete 14,453-knot preparation is 24.44x faster, with identical ASCII and
  binary decks and no redundant Python solution of live section roots.
- The 60-second timestep check passes. The 0.05 m grid is rejected because its
  natively integrated absolute bed-change volume differs from 0.025 m by 10.95%, above the frozen
  10% gate; all conservation and physical checks still pass.
- Exact-zero suspended work and remote dry-cell bedload work are bypassed,
  section terrain sorting is `O(n log n)`, and independent live-section root
  intervals run in deterministic OpenMP parallel order.
- The paired 0.025 m, 16-thread pilot improves from 37.21 to 13.14 seconds
  (2.83x). Frames and boundary ledgers are byte-identical to alpha 69 and at
  1, 8, 16, and 32 threads.
- Native profiling now separates boundary control, bedload capacity,
  reconstruction, traces, face flux, limiting/update and finalization.

# BRAIDS 0.1.0 alpha 69

This build closes the experimental material inputs as a declared calibration
envelope, registers the usable flume observations, and runs the first real-DEM,
seven-fraction MorFac-1 bed-load/Exner integration pilot.

## Alpha 69 morphodynamic integration pilot

- The seven-class material envelope preserves the published 2.8 mm D84 and
  clearly labels unpublished feed gradation, particle density, and porosity as
  bounded calibration quantities rather than measurements.
- Initial, H2, H3, and H4 DEMs are reproducibly registered to a common 5 mm
  grid. H4 is excluded because its implied retained volume is 5.81 times the
  measured mass balance and it has an undocumented longitudinal datum tilt;
  H3 remains withheld validation.
- Multi-fraction native kernels now choose deterministic OpenMP execution from
  their actual cells-by-fractions/state workload. The 4,031-cell, seven-class
  pilot is 6.24x faster on 16 cores than on one core; 1/8/16/32-core outputs
  are byte-identical.
- The 60-second MorFac-1 pilot injects 1.82726 kg of graded sediment, conserves
  every fraction to at most 5.97e-11 kg, remains finite/nonnegative, and
  resolves bed evolution on the measured initial terrain.
- Local engineering coordinate systems with metre units are now accepted for
  laboratory domains, while geographic and non-metre inputs remain rejected.

# BRAIDS 0.1.0 alpha 68

This build starts the integrated experimental braided-river qualification and
improves multicore execution without changing the accepted numerical result.

## Alpha 68 experimental benchmark and multicore runtime

- The Johannot et al. (2026) UC1 large-flood experiment is locked by source
  identifiers and checksums. A conservative 14,453-knot forcing schedule
  preserves 34.070062181 m3 of corrected water and 220.5702 kg of sediment.
- The unusable published H1 GeoTIFF is retained but explicitly excluded; H2 is
  calibration and H3/H4 are withheld predictions. Missing complete input-bed
  gradation and particle density remain a hard sediment-qualification gate.
- Native time-series interpolation now uses logarithmic interval lookup, and
  additional deterministic OpenMP regions cover the measured fine-grid serial
  hotspots. The 65,608-cell pilot reaches 4.04x at 32 threads.
- Checkpoints, ledgers, depth, stage, velocity, and discharge outputs are
  byte-identical at 1, 8, 16, and 32 threads.
- The QGIS UI defaults to detected physical cores (16 on the reference PC),
  keeps all logical processors selectable, and binds native threads to cores.

# BRAIDS 0.1.0 alpha 67

This is the source-clean installable test build of the BRAIDS QGIS plugin. The
current package targets QGIS 3.40 or newer on 64-bit Windows and includes the
Fortran/OpenMP solver and its runtime libraries.

## Alpha 67 qualified bedform roughness feedback

- A new blank-by-default **Bedform roughness feedback** control couples the
  Fredsoe-MPM equilibrium dune-height predictor to Van Rijn 1984 dune length
  and Nikuradse roughness, including `3 D90` grain roughness.
- Against that calibrated grain-only limit, the retained 300 s flume raises
  mean water level by 0.3221 m in locked current-source Delft3D4 and 0.2858 m
  in BRAIDS; the effect ratio is 0.8874.
- A deterministic local fixed point avoids hidden bedform state, preserving
  AMR and restart identity. `bedform_roughness.txt` records effective Manning,
  dune geometry, Nikuradse roughness and iteration count.
- Qualification covers source equations, paired hydraulic response, mass,
  wet/dry AMR, restart, one/eight-thread identity and default-off deck identity.
- It is limited to roughness-corrected MPM and noncohesive material. Suspended
  exchange, normal-depth inlet control and other experimental processes fail closed.

## Alpha 66 qualified morphological acceleration

- A new default-off **Morphological acceleration** control accepts a constant
  MorFac from 1 to 20 or an optional Delft3D-compatible block schedule.
- It scales the conservative bed-load/Exner update, not transport capacity.
  Donor limiting, boundary ledgers and AMR reflux therefore all see the same
  accelerated flux.
- All 22 retained paired gates pass against the locked current-source Delft3D
  runtime. The published maximum is MorFac 20 for the quasi-steady subcritical
  translating bedform, but MorFac 1 for wet/dry hydraulic transients.
- Each run writes `morphological_acceleration.txt` with hydraulic time,
  integrated morphological time and instantaneous factor. Timesteps stop at
  schedule knots; restart and one/eight-thread outputs are exact.
- Suspended exchange, time-varying hydraulic boundaries, legacy bank erosion
  and process-separated bank failure fail closed. General repose relaxation
  remains on hydraulic time, matching the Delft3D source behavior.

## Alpha 65 qualified bed-load adaptation length

- A new default-off **Non-equilibrium bed load** control applies steady
  first-order relaxation along resolved streamlines. Each sediment fraction
  uses `L_g = C_L D50_g`; the editable default is 100 grain diameters.
- The default retains the established local-equilibrium solver. Very short
  lengths recover that limit exactly, and equilibrium-capacity inlet
  conditions keep their constitutive meaning.
- A frozen adequacy assessment shows that local equilibrium materially misses
  a measured gravel capacity step and a rapidly varied crest response. The
  pinned current-source Delft3D `tranb2` kernel supplies the common equilibrium
  comparator and agrees with BRAIDS MPM to `2.94e-15` relative difference.
- Qualification covers the exact exponential profile, independent lengths for
  multiple grain fractions, a mixed-level AMR analytic profile, wet/dry
  regridding, conservation, exact checkpoint restart, and bitwise one/eight-
  thread behavior including the parallel kernel.
- The 100-D50 value is a measured starting point, not universal calibration.
  Suspended-load adaptation and a spatially varying adaptation-length field are
  outside this milestone.

## Alpha 64 qualified toe erosion and bank failure

- A new default-off **Toe erosion, undercutting and block failure** option
  separates continuous excess-shear toe retreat, persistent undercut memory,
  discrete block release and conservative per-fraction deposition.
- The QGIS project exposes erodibility `kd`, critical toe shear `tau_c`, the
  excess-shear exponent and critical undercut. The displayed literature
  defaults are starting values and require site or laboratory calibration.
- The process works on fixed and adaptive meshes, preserves its history across
  regridding and restart, stops at finite bed inventory, and is byte-identical
  with one and eight OpenMP threads. The qualified domain-wide avalanche option
  provides post-collapse repose redistribution.
- All 13 qualification gates pass: analytic toe and collapse volumes, residual
  undercut, event count, critical-shear cutoff, fraction conservation,
  laboratory depth-response direction, current-source Delft3D dry-bank kernel
  equivalence, source identity/scope and native regression.
- The legacy Delft3D-style dry-bank supply remains available for comparison but
  is clearly labelled and cannot be enabled with the qualified bank model.
- Honest non-goals are predictive geotechnical factor of safety, pore pressure,
  seepage, thermal erosion and a moving vertical overhang mesh.

## Alpha 63 qualified secondary-flow sediment steering

- A new default-off **Secondary-flow transport** option in the Sediment tab
  rotates equilibrium bed load using the D-Flow FM streamline-curvature and
  spiral-intensity equations. `Espir` is explicit and defaults to 1.0.
- The control is limited to magnitude-based bed-load laws: MPM,
  Ashida-Michiue, Wilcock-Crowe, modified Wilcock-Crowe and Gaeuman. Positive
  Manning roughness is required. Scalar transport magnitude is preserved.
- All 15 frozen gates pass in paired locked current-source D-Flow FM mild-bend
  and symmetric-bifurcation cases. Curvature/intensity correlations exceed
  0.95, per-fraction Exner-effect correlations exceed 0.95, and mean direction
  differences are below 6.6 degrees.
- Fine and coarse sediment both move in the correct transverse direction, and
  the symmetric bifurcation partitions each BRAIDS fraction exactly equally.
  Closed-domain fraction residuals are below `1e-8 kg`.
- Native regressions cover scalar-magnitude preservation, one/eight-thread
  binary identity, exact restart, conservation and an actively changing AMR
  hierarchy. The Windows packager now requires the 15-gate manifest.
- Scope is deliberately narrow: `BetaSpiral=0`, so there is no secondary-flow
  momentum forcing, and Coriolis is not part of the planar Cartesian model.

## Alpha 62 qualified general wet/dry repose avalanching

- A new optional **Domain-wide repose avalanching** panel redistributes the
  mobile bed across bars and channels without requiring erodible-bank zones.
- Wet and dry/shallow limiting slopes, relaxation time, physical relaxation
  length and wet/dry transition depth are saved in the project and prepared
  native deck. Domain and obstacle boundaries remain fixed.
- The production operator passes all 28 frozen gates against the locked
  current-source D-Flow FM `duneaval` implementation and an independent
  conserved-volume repose reference. Normalized paired profile error is below
  0.243%, with bed-volume closure near `1e-12 m3`.
- BRAIDS conserves both grain fractions, is bitwise identical with one and
  eight OpenMP threads and across a one-thread to eight-thread restart, and has
  a dedicated AMR hierarchy conservation/identity regression.
- Domain-wide and bank-zone avalanching are mutually exclusive so slope
  redistribution is never applied twice. Bank erosion supply can remain active
  as a separate process.
- The Windows package now fails closed unless the packaged 28-gate avalanche
  qualification manifest passes. Delft3D source, binaries and raw runs remain
  excluded.

## Alpha 61 Phase 3 graded-bed and stratigraphy qualification

- The integrated Phase 3 qualification passes all 74 gates across graded-bed
  sorting/armoring, multilayer buried-pulse recovery, and active-layer
  thickness/mixing sensitivity.
- Surface sorting correlations are at least 0.995793, buried-pulse and dynamic-
  layer bed-change correlations are 0.997062 and 0.997086, and the maximum
  retained BRAIDS total sediment-budget residual is `1.407e-11`.
- The new **Phase 3 graded-bed qualification** panel exposes evidence hashes,
  surface/subsurface D10/D50/D90, armoring and stratigraphy diagnostics,
  budgets, determinism, retained modes, rejected modes, and limitations.
- The Sediment tab now offers a qualified water-depth-scaled active layer,
  `max(depth fraction × local water depth, minimum thickness)`. Constant
  thickness remains the default and existing projects remain compatible.
- Delft3D vertical bed diffusion is deliberately excluded: the locked native
  runtime failed after four steps with negative layer mass and hydraulic
  blow-up. Partial failed output is not accepted as evidence.
- A reproducible Windows packager verifies current native source identities,
  the complete pinned runtime-DLL closure, license/source-access files, every
  archived file hash, and the packaged Phase 3 manifest. Raw validation runs
  and the Delft3D source tree are rejected from the archive.

## Alpha 60 Phase 2 morphodynamics qualification

- The complete nine-step Phase 2 bedload/Exner qualification now passes all
  222 frozen gates against the locked current-source Delft3D runtime and the
  applicable independent analytic references.
- The fixed-grid/AMR bed-profile difference is 0.762%, with 0.999879
  correlation, 74.5% average leaf count, and a `7.44e-11` relative sediment
  budget residual.
- Grid refinement contracts the solution distance by a factor of 0.2446. The
  timestep Cauchy envelope is `7.935e-06`; no unsupported formal time-order
  claim is made after the solution reaches its spatial-error floor.
- One- and eight-thread state frames are binary identical. A one-thread
  checkpoint resumed with eight threads reproduces every common state, ledger,
  and final checkpoint exactly.
- The expanded **Phase 2 qualification** panel in Sediment shows the packaged
  manifest status, gate count, conservation, AMR, convergence, determinism,
  evidence hashes, scope, and limitations. It describes the qualified release;
  it does not certify the user's current project or inputs.
- The report builder fails closed if evidence is absent, any gate fails,
  milestone coverage is incomplete, or an evidence digest is malformed.

## Alpha 59 graded-bed hiding/exposure qualification

- The MPM active-layer path now offers every hiding/exposure relation in the
  pinned Delft3D source: Egiazaroff, Ashida-Michiue,
  Parker-Klingeman-McLean and Wu-Wang-Jia.
- Parker alpha and the Wu exponent are explicit sediment controls. Their UI
  starting values are 0.6 and -0.6, respectively; the tooltips make clear that
  these are calibration inputs, not universal constants.
- The new deck-v25 envelope carries those exponents without changing existing
  state layouts, fixed-grid/AMR data, compact binary input or old-deck support.
- Thirty three-fraction equation states agree with the compiled current
  Delft3D `comphidexp.f90` oracle. Paired native runs for no correction and all
  four models pass hydraulics, class budgets, equilibrium inlet capacity,
  Exner bed pattern, sorting direction and erosional armoring gates. Surface-D50
  correlations are 0.915–0.956 and bed-change correlations are 0.923–0.959.
- The QGIS project file, input summary, prepared-run metadata and reload path
  preserve the selected relation and both exponents.

## Alpha 58 native large-mesh input and lower-copy construction

- Large fixed and adaptive cases now keep controls and boundary forcing in the
  inspectable text deck but transfer bulk cell, face and hierarchy arrays in a
  versioned little-endian native stream. The Fortran solver reads each array in
  one operation instead of parsing millions of ASCII values. Historical text
  decks remain supported.
- The binary stream is written atomically, has a recorded SHA-256 identity and
  is included in prepared-run cache reuse. Text and binary adaptive runs have
  byte-identical state frames, including live normal/critical inlet sections.
- Hierarchy construction may now consume its private pre-agglomeration mesh,
  avoiding a second immutable copy of every retained cell and face. Public
  agglomeration remains non-mutating by default.
- On the current domain at six levels and 1 m minimum size (3,249,480 hierarchy
  cells and 6,515,228 faces), clean construction fell from 98.55 s in alpha 57
  to 72.06 s. The complete 1.38 GB native stream writes in 13.21 s.
- Whole-level Python threading is intentionally not enabled: the measured
  four-level kernel slowed from 13.08 s to 13.95 s because object creation is
  GIL-bound. Native solver OpenMP threading and numerical results are unchanged.

## Alpha 57 faster AMR preparation

- Mesh clipping now visits only cells intersecting the calculation domain,
  builds regular Cartesian geometry without temporary NumPy objects, and
  validates face closure and moments in vectorized batches.
- Small-cell agglomeration remaps immutable cells and faces directly instead
  of performing reflective dataclass copies for every hierarchy object.
- On the 22 September four-level project (462,359 hierarchy cells and 930,841
  faces), clean hierarchy construction fell from 29.57 s to 13.34--13.48 s,
  a 54% reduction. Cell indices, parent links, scalar geometry, polygons and
  source-member arrays match the previous run artifact exactly.
- AMR construction now reports work-weighted progress inside **Prepare Run**.
  The finest level receives the appropriate share of the progress interval,
  and cancellation checks remain active during clipping.

## Alpha 56 configurable AMR level count

- **Minimum cell size** now defines the finest requested resolution. Choose
  two to eight mesh levels and BRAIDS derives every coarser size using the
  fixed, qualified 2:1 refinement ratio. The default is unchanged: four levels
  at 20, 10, 5 and 2.5 m.
- The project, packed hierarchy, initial-mesh preview and native solver all
  carry the selected level count. Minimum-refinement zone choices follow it.
- Inlet and outlet cells are forced to the finest configured level at initial
  selection and every later regrid. Native tests cover two, three and five
  levels and verify persistent finest-level boundary ownership.

## Alpha 55 results animation and boundary-flow cleanup

- **Animate adaptive mesh** and **Export Animation PNG Frames…** now live in
  the same **Animation** accordion.
- The first adaptive-mesh build reports progress while it reads topology and
  writes the GeoPackage. The completed product is cached beside the run and is
  not deleted when animation is unchecked; later loads reuse it immediately.
- **Boundary flow audit** is now **Boundary flow**. Plots and CSV exports use
  nonnegative magnitudes and state whether the selected ID is inflow or
  outflow, while the conservative native ledger remains signed internally.
- Reloading or re-preparing results clears the boundary selector before adding
  the active IDs saved in that run, preventing repeated entries.

## Alpha 54 atomic boundary-face reassignment

- **Apply to Selected Face(s)** now validates that every multi-face hydraulic
  boundary is contiguous and collinear before it edits the boundary layer.
  The error therefore appears at assignment time instead of being deferred to
  **Preview Section**.
- A failed validation leaves the previous valid face assignment untouched.
- Reusing an existing inlet/outlet ID now replaces that ID's prior faces and
  returns the old faces to walls. This also lets projects created with an older
  build repair a persisted invalid assignment by selecting the intended faces
  and applying the same ID again.

## Alpha 53 boundary-flow audit

- The Results tab now plots and exports the native accepted face-flux ledger
  for every inlet and outlet. Positive values enter the domain and outlet
  exports are negative. This is distinct from a raster transect, which samples
  evolving cell-centred flow inside the domain and is not an imposed boundary
  measurement.
- Fast live-preview results now materialize boundary budgets instead of
  omitting them. Existing completed adaptive runs repair the budget on demand,
  without rerunning the solver.
- Every new run writes a compact cumulative boundary ledger, including large
  fixed-grid runs that intentionally skip large ASCII state archives.
- The native executable creates and verifies its own `live_preview` directory,
  so standalone cases no longer depend on QGIS preparation having created it.
- The Windows boundary qualification covers all four user inlet controls and
  prescribed-stage, critical, free-weir and supercritical outlets. The matrix
  passes its discharge, conservation, wet/dry and OpenMP-repeatability checks.

## Install

1. In QGIS, open **Plugins → Manage and Install Plugins → Install from ZIP**.
2. Select `BRAIDS-0.1.0-alpha.91-QGIS-windows-x86_64.zip` and install it.
3. Enable **BRAIDS** in **Installed**, then open it from the BRAIDS toolbar button
   or **Plugins → BRAIDS**.

Existing source-checkout installations should be removed from the active QGIS
profile before installing the ZIP, so QGIS sees only one `braids_qgis` plugin.

## Suggested manual smoke test

1. Start a new QGIS project in a projected CRS whose units are metres and add a
   DEM covering the intended domain.
2. Open BRAIDS and confirm the tabs are **General, BC & IC, Mesh, Sediment, Run, Results**.
   The bundled native solver is selected internally and has no editable field.
3. Choose one writable **BRAIDS project folder**. The project definition,
   generated layers, prepared runs and results are stored there.
4. Optionally press the line button beside **Draw Calculation Domain** to draw inlet guidance polylines (right-click to finish). These are saved in `braids_inlet_guides.gpkg` in the project folder. They remain visible for drawing/snapping, then hide automatically when the domain is saved or BC assignment begins. They are excluded from boundary-face choices and can be shown again from the QGIS Layers panel.
   Press **Draw Calculation Domain**, click the polygon vertices, and right-click
   to close it. A loaded one-feature polygon Shapefile can be selected instead.
5. In **BC & IC → Boundary conditions**, press **Create from Domain** beside the boundary-face selector. Every new red face starts as a wall. Press **Select** to activate the QGIS selection tool before assigning faces.
   Assign exactly one selected face to
   each inlet. Outlets and walls may use several selected faces. Each inlet or
   outlet card has **Delete BC**, which removes that ID and returns all of its
   assigned faces to walls.
6. In **BC & IC → Initial conditions**, leave **Initial water level** blank for a
   dry start. To initialize water, draw or select an initial-water polygon and
   enter its constant level immediately below it. You can instead enter the
   level, press **Create from DEM Point**, and click inside the intended basin;
   BRAIDS creates its connected below-level contour as an editable polygon.
7. In **Mesh**, choose the minimum cell size and number of levels. AMR derives
   the coarser levels at a fixed 2:1 ratio; inlet/outlet cells always use the
   minimum size. **Visualize Initial Mesh** uses the terrain, initial
   water, boundaries and refinement criteria. Existing saved fixed-grid projects
   stay fixed until AMR is enabled. Adjust criteria or draw minimum-refinement
   polygons when needed. Set the default Manning value. Optionally press **Add New Manning
   Zone**, draw a polygon and right-click to close it. The zone name is assigned
   automatically from the coefficient. Press **Visualize Initial Mesh** to load
   the exact initial cut-cell mesh, including retained partial and agglomerated cells.
   To add a fixed object, open **BC & IC → Internal conditions** and use **★
   Polygon**, **○ Circle** or **⬭ Ellipse**. Circles use two clicks; oriented
   ellipses use three. You may select an existing polygon Shapefile or
   GeoPackage instead. Use **Select** and **Delete Selected** to remove a saved
   object, or **Cancel** to leave an unfinished drawing. Preview the mesh to
   inspect the reflective cut perimeter.
8. In **Sediment**, use **Enable sediment transport** to select a morphology or
   hydraulics-only run. When enabled, set suspended load, bed load, material,
   active-layer and inlet-supply values. **Settling treatment** selects a
   constant velocity or concentration-dependent hindered settling; the latter
   uses the displayed total-concentration reference. Use **Plot Time Series** on scheduled boundaries.
   To model mobile banks, define at least two grain fractions and a positive
   active-layer thickness, then open **Erodible banks and slope failure** in
   Sediment, enable bank morphology and draw one or more bank zones. Domain walls
   remain fixed; only marked internal cells can donate bank material.
9. Use **Save Project** in the first General accordion. In **Run**, choose the CFL limit and OpenMP thread count, then press
   **Prepare Run** to build the mesh, terrain, initial fields and native input
   deck, then run it. **Show execution log** reveals the live output; the same
   register is saved as `native.log` in the run folder.
10. In **Results**, choose any completed run in the project folder, prepare the raster products, select a variable and saved
    time, or load its complete time series and use the Animation player. Enable
    **Show computational mesh** to inspect cell size over the raster. The overlay
    follows the actual adaptive mesh at each saved time.
    Load a second run to compare its rasters on the same map.

Alpha 36 removes the artificial hard bed-load capacity jump as an inlet first
wets. Capacity now follows a continuously differentiable shallow-wetting ramp
between the dry tolerance and the existing minimum bed-load depth. Fully wet
transport formulas, prescribed inlet loads, hydrodynamics, WENO-Z interior
reconstruction and the earlier inlet-spike protection are unchanged. The
compact near-inlet suite passes all 105 correction, conservation, restart and
one/eight-thread checks. A short full resource-domain sensitivity also closes
water and sediment ledgers to roundoff; its young local sediment front remains
resolution-sensitive and does not yet qualify reservoir deposition.

Alpha 35 replaces the mesh-scaled bank link relaxation with a conservative
isotropic nonlinear diffusion using a physical relaxation length and time. A
continuous wet/dry repose transition and cut-cell centroid correction are
included. The compact aligned/45-degree test passes its integrated response,
grid convergence, conservation, restart and one/eight-thread checks. Its raw
cell-scale bank-change comparison still fails the retained rotation limit, so
local bank-front placement remains explicitly unqualified.

Alpha 34 corrects the two-dimensional slope magnitude used by erodible-bank
slope failure. The repose limit is applied to the reconstructed physical slope,
then its excess is projected onto each face; diagonal banks are no longer
artificially stabilized by treating Cartesian components independently. The
one-dimensional bank oracle, conservative class transfers and deterministic
execution remain intact. A compact aligned/45-degree two-fraction test closes
water and sediment balances and shows convergent integrated bank redistribution,
but its raw local bed-change rotation limit still fails. Alpha 35 supersedes
this mesh-scaled relaxation while retaining the local failed gate.

Alpha 33 corrects inlet bedload reconstruction: the bounded capacity treatment now spans the full boundary stencil radius, removing the alternating affine Exner-rate error while retaining the inlet-spike controls. Hydraulic equations and hydraulic boundary conditions are unchanged. Existing WENO-Z reconstruction remains in the interior.

Alpha 32 fixes AMR mesh preview and run preparation with DEM-sampled inlets. The hierarchy and case now share the same subdivided boundary integration faces and terrain samples; computational cells, hydraulic equations and the native executable are unchanged.

Alpha 31 adds persistent inlet guidance polylines beside domain creation, with automatic hiding and deselection when the domain is complete or boundary conditions are selected. Guides are drafting aids; hydraulic conditions still belong to polygon faces.

Alpha 30 restores preparation and run buttons after input, background-task, task-submission or executable-launch errors. Mesh follows BC & IC in the tab order.

Alpha 29 adds native four-level adaptive refinement and coarsening. Water,
all suspended fractions and every bed layer share the same conservative leaf
mesh. Subcycled timesteps synchronize actual hydraulic, suspended and bedload
fluxes; checkpoints retain the mesh and adaptation history. Refinement follows
wet/dry fronts, wetted walls and obstacles, unresolved terrain, hydraulic
changes, sediment fronts, bank activity, roughness and optional minimum-level
polygons. The initial preview and animated mesh overlay show actual solver cells.
Independent discharge and sediment schedules are qualified on subcycled levels,
including a conservative correction for negligible negative sediment tails at
coarse/fine interfaces. One/eight-thread results and restart remain identical.

The compact criteria suite and uniform-grid comparisons are numerical checks,
not experimental validation. Choose the base size so `base / 8` resolves the
smallest important physical feature. AMR cannot recover information missing
from the DEM. Large regions of fine cells can remove its speed advantage, and
unmergeable cut fragments still constrain the timestep. This release prebuilds
the candidate geometry and keeps fine terrain/layer detail: fewer active cells
do not yet imply a corresponding reduction in preparation memory.

Alpha 28 renames the plugin to BRAIDS and installs the supplied BRAIDS logo.
Only the first accordion in each tab opens initially, section typography is
larger, long grain-table headers wrap, and the BC & IC selection, deletion and
cancel actions are explicit. Result GeoTIFFs now include a fluid-area alpha
band: each edge pixel uses the exact cut-fragment coverage, so raster display no
longer turns a polygon cut into a hard whole-pixel mask. The solver still keeps
one conservative state per true cut control volume.

Alpha 27 introduced the four-times-finer result display grid, cached unchanged
normal/critical section roots, circle and oriented-ellipse obstacle capture,
and Sediment-tab bank controls. Internal hydraulic conditions currently consist
only of fixed reflective solid walls; bridge-pillar footprints use the same objects.

Alpha 26 added arbitrary fixed polygon objects under **BC & IC → Internal
conditions**. Draw a managed object or select an existing polygon Shapefile or
GeoPackage. Convex, concave and objects smaller than one background cell are
cut exactly from the fluid geometry. Their complete perimeters use the native
impermeable reflective shallow-water wall flux and zero suspended or bed-load
flux. Small remaining fluid fragments use the existing conservative
agglomeration policy. Still water remains at rest, a travelling wave reflects
from an irregular object while conserving water to roundoff, and one/eight
OpenMP outputs are byte-identical. Embedded core API 15 reloads the new project,
mesh and QGIS controls during an in-place upgrade.

[Solid-object numerical proof](../validation/hydraulics/solid_obstacles/reports/SOLID_OBSTACLES.md).

Alpha 25 carries the full sampled DEM profile into native normal- and
critical-depth controls. Normal flow uses contiguous conveyance subsections
from the same whole-cell Manning field used by solver friction, and both
controls recompute their horizontal level as the boundary bed evolves. The
run folder records these values in `boundary_sections.txt`. Asymmetric and
compound geometry, partial wetting, a dry default-wall fill, conservative
bed feedback, exact restart, and byte-identical one/eight-thread results pass
the compact production qualification. Embedded core API 14 reloads the new
case writer and preview behavior during an in-place upgrade.

[Production-section numerical proof](../validation/hydraulics/production_sections/reports/PRODUCTION_SECTIONS.md).

Alpha 24 added optional concentration-dependent hindered settling to every
suspended exchange path. It multiplies each fraction's fall velocity by
`max(0, 1 - Ctotal / Creference)^5`, where `Ctotal` sums all suspended
fractions. The default reference is 10,000 kg/m³. Single-material,
multi-fraction, Van Rijn and cohesive paths match independent recurrences;
class inventories close to roundoff, concentrated limiting makes no negative
fall velocity, hydrodynamics remain unchanged, restart is exact, and one/eight
OpenMP frames are byte-identical. Embedded core API 13 reloads the new QGIS
controls and project/native fields during an in-place upgrade.

Alpha 23 fixes time-varying **total discharge** inlets. The native deck now
writes a neutral stage value for this free-stage mode while retaining `None` in
the project diagnostics, so a Q-only hydrograph prepares and executes without
requiring an artificial water-level series. The compact 1/0.5/0.25 m graded
reservoir qualification accepts its water and both fraction loads, closes water
and class budgets near roundoff, resolves deposition and downstream sorting,
and changes its medium/fine positive deposit by 0.62%. Its production runs use
eight OpenMP threads. Embedded core API 12 forces an in-place QGIS upgrade to
reload the corrected writer.

Alpha 22 added configurable preserved substrate stratigraphy below the active
layer. Choose the number of substrate layers and their vertical resolution in
**Sediment → Bed load and morphology**. The upper substrate layers retain that
fixed thickness and composition order; the deepest layer stores the remaining
erodible bed. Burial and later exhumation conserve every grain class, remain
restart exact and produce byte-identical one/eight-thread results. Result
rasters expose both aggregate substrate inventories and each saved layer's
inventory and volume fraction. Native deck v19 and embedded core API 11 carry
the layer count through preparation, execution, checkpoints and results.

Alpha 21 completed the compatible suspended/bed-load formula matrix and added an
MPM shear-closure choice. **Specified friction coefficient (Zhu)** preserves the
validated Zhu benchmark exactly. **Manning and grain roughness corrected** uses
local depth, the Mesh Manning field, active-mixture D90 and the fixed source
critical Shields value 0.047 for the source-matched
graded-bed closure. The latter writes native deck v18 and requires MPM, multiple
fractions and a positive active layer. Eleven short BRAIDS/reference pairs retain
unchanged hydraulics, pass the profile and class-budget gates, and differ by at
most 6.91% in equilibrium inlet bed load. The embedded core API marker advances
to 10 so an in-place upgrade reloads that UI, project field and native writer.

Alpha 20 added opt-in lateral bank erosion and wet/dry bank avalanching. The
QGIS internal-condition controls create editable erodible-bank polygons and
map every positive-area contact to one complete solver cell. The native
version-17 deck conserves each sediment fraction, limits removal by the active
layer and substrate above bedrock, excludes calculation-domain boundary faces,
and lets lowered bank cells wet through the normal shallow-water flux. Its
defaults match the supplied Delft3D morphology source: wet slope 10:1, dry
slope 1:1, avalanche time 86400 s and wet-depth switch 0.1 m. A compact
one/eight-thread verification is byte-identical and matches the source-derived
link operator to `2.78e-17 m`. The embedded core API marker advances to 9 so an
Alpha 20 in-place upgrades reload the bank inputs and version-17 case writer.

Alpha 19 corrects the Wilcock-family surface statistics to use active-layer volume
shares consistently, including classes with different bulk densities. The matched
Delft3D flume now uses explicit two-class composition and gives -3.68% inlet-load
difference for Wilcock–Crowe. Hydrodynamics and transport coefficients are unchanged.

Alpha 18 evaluates nonlinear bedload capacity at cell centres and transfers it
to shared faces with bounded WENO-Z or bounded cut-cell least-squares traces.
The equilibrium-capacity inlet uses a monotone closure. This removes the
rejected inlet-adjacent bed spike and lowers inlet-region total variation below
Delft3D without changing hydrodynamic fluxes or
adding bed smoothing. The corrected Zhu window passes 37/37 frozen gates, and
the straight-basin one/eight-thread outputs are byte-identical.

Alpha 17 added current-driven Ashida-Michiue, Soulsby, Wilcock-Crowe, modified
Wilcock-Crowe and Gaeuman bed-load choices. Sediment fractions can now be sand
or mud. Mud carries explicit critical erosion/deposition shear stresses,
erosion parameter and exponent; the native solver applies conservative
Partheniades-Krone exchange and the active-surface sand-mud erosion switch.
The new production deck remains native Fortran/OpenMP and its focused mixed-bed
test runs with eight threads. The embedded core API marker advances to 8 so an
in-place Alpha 16 upgrade reloads the new sediment fields and case writer.
Wave-dependent formulas, flocculation, fluff layers and consolidation are not
presented as available.

Alpha 16 adds an initial mesh preview to the Mesh tab and stores a computational
mesh GeoPackage beside each run's raster results. The Results overlay is tied to
the selected saved time. Current runs have one fixed mesh; the manifest and UI
also accept multiple mesh snapshots and change the visible snapshot during
animation when future bathymetry-driven AMR writes them. The embedded core API
marker advanced to 7 so an in-place Alpha 15 upgrade reloaded these services.

Alpha 15 fixes the long runtime and later critical-outlet stalls reproduced
from `braids_20260912_001152`. The native wet/dry threshold now matches the
standard WAVE/GeoClaw `0.001 m` value, preventing negligible water films from
creating extreme wave speeds and collapsing the global time step. QGIS reads
native progress from stdout instead of repeatedly loading the growing frame
file. Output-time alignment no longer lowers the next physical time-step
estimate. A multi-face critical outlet now closes locally reversing or
unresolved shallow faces, recomputes its common critical section, and lets
outward-supercritical faces leave unchanged.

The reported 61,458-cell case advances with `dt = 0.05 s` and no rejected
trials. The original needed 47,713 steps and about 42 minutes to reach 450 s;
the repaired build used 9,000 steps there, passed both outlet transitions, and
completed the verified 450–600 s continuation in 3,000 steps and 190 seconds.
Its continuation water balance closes to `3.3e-13 m3`. Reinstall and press
**Prepare Run** again because the dry threshold is written into `case.data`.

Raw native output remains under `<project>/braids_runs/<run>/`. **Prepare Raster
Results** writes GeoTIFFs, their manifest and the boundary budget table under
that run's `qgis_results/` folder. Results now displays this exact path.
The embedded core API marker ensures an in-place QGIS upgrade reloads the
current case writer.

Alpha 14 fixes the cut-cell failure reported when a section-controlled inlet
wets its initially dry boundary cells. Water-level, water-depth, normal-depth
and critical-depth inlets now retain the resolved horizontal water line and
the requested total discharge throughout the run. Previously, the native
solver changed these controls to an interior characteristic after a cell
became barely wet; the resulting local regime conflict could not be repaired
by timestep halving. Reinstall this build and press **Prepare Run** again so
the new inlet mode is written into the native case.

Alpha 13 moves boundary deletion into the inlet/outlet cards. **Delete BC**
removes the complete logical condition and returns every polygon face assigned
to its ID to the default wall behavior. The former selection-based deletion
button has been removed.

Alpha 12 restores the map-first initial-water workflow from AVAC4QGIS. Enter a
horizontal water level, press **Create from DEM Point**, and click inside a DEM
basin. Only the four-connected DEM cells at or below that level are selected,
so a disconnected depression at the same elevation is excluded. A contour that
reaches the available DEM edge is rejected because it cannot form a closed
water body. The generated polygon stores the clicked seed, selected-cell count
and water level and remains editable with QGIS's normal tools.

Preparation now reproduces the initial native critical-depth outlet trace. It
stops before execution when the selected outlet face would exceed the native
50 m/s speed limit and reports the boundary ID, represented length, critical
discharge and trace speed. The native solver has the same dedicated outlet
failure message as a second line of defense. For a reservoir with no outlet,
delete the outlet assignment and its face returns to the default wall behavior.

Alpha 11 masks the water-level raster by the wet domain at every saved time.
Cells at or below the run's native dry tolerance are written as NoData and
therefore remain transparent in QGIS. If several cut control volumes contribute
to one background pixel, only wet contributions enter its displayed water level.

Alpha 10 fixes reloading a result after its raster layer has been removed from
the QGIS project. The Results cache now detects deleted QGIS C++ layer objects,
discards their stale Python wrappers, and creates a fresh raster layer. The
animation loop performs the same check if a loaded layer is removed during or
between playback actions.

Alpha 9 fixes preparation after an in-place plugin update by advancing the
embedded `braids_core` API, preventing an older cached `write_cut_case` function
from rejecting the CFL input. Reconstruction is no longer exposed: every QGIS
run uses bounded WENO-Z automatically. Cell size and the future AMR placeholder
are together under **Mesh resolution and future AMR**. CFL still defaults to
0.45 and can now be set through 0.9.

Alpha 8 adds the **BC & IC** tab, with separate boundary, initial and reserved
internal-condition accordions. It makes bounded WENO-Z the clearly labelled
general default, exposes the native CFL limit, and sets the OpenMP range from
the machine CPU count with a four-thread default where available. The Results
tab selects among completed runs in one project folder, keeps rasters from
multiple runs loaded, gives every variable a palette whose zero is transparent,
and suppresses sediment variables for hydraulics-only runs. Requested-versus-
accepted boundary plots remain available to validation code but are removed
from the user interface.

Alpha 7 replaced per-frame result polygons with compressed multiband GeoTIFF
rasters. Each band is one solver output time. The animation slider synchronizes
all loaded BRAIDS result rasters. Cut control volumes are mapped to the Cartesian
output cells by exact area weighting. Boundary previews now call the submerged
part of a cross-section the **wetted boundary section** and explain that this is
the portion through which boundary flow can pass.

Alpha 7 presents one project folder and stores `braids_project.json` there
automatically. It removes the native executable control, moves project saving
to the project accordion, moves run preparation to Run, and reorganizes initial
water and boundary assignment controls. Loaded polygon rings are normalized in
memory when they contain redundant consecutive vertices; the source layer is
not edited. One inlet face is enforced in the UI, multi-face outlets remain
available, and unassigned or deleted faces are walls.

The earlier root-folder correction remains: the default project folder is
writable and the filesystem root is rejected. Loaded one-feature polygon
Shapefiles and one-part MultiPolygon provider representations are accepted.

The supplied BRAIDS logo is used in QGIS Plugin Manager, the plugin menu and the
toolbar action.

Normal-depth boundary inputs are automatic. Slope is sampled from the DEM, and
Manning comes from a zone that shares the inlet edge or from the Mesh default.
The slope and section-Manning fields, prepared-inlet diagnostic accordion,
boundary-cell selector and optional Manning-zone ID have been removed. The
internal small-cell policy uses the qualified 0.5 area threshold. Preview and
forcing plots accept DEMs with an undeclared band unit and are protected from
an older `braids_core` module already loaded in QGIS.

The Sediment tab now starts with a master switch. Turning it off disables every
sediment section and prepares a native run with sediment boundary supply,
suspended exchange, bed load and morphology corrections inactive.

An empty initial-water level now saves and prepares an explicitly dry domain.
It no longer blocks section preview or **Prepare Run**. Outlet preview shows the
terrain section without inventing a water line when the initial domain is dry,
and outlet cards no longer expose a forcing-plot button.

Initial water is defined with an optional polygon and the constant **Initial
water level** immediately below its polygon selector in **BC & IC**. Select any
file-backed polygon layer, press **Draw Initial Water Body**, or press **Create
from DEM Point** and click inside the basin after entering its level. Every
retained solver cell with positive-area polygon contact is initialized to
`level − bed elevation` when that value is positive; cells outside the polygon
remain dry. Without a polygon, all cells below the level are initialized. Old
projects containing the earlier point seed remain readable.

## Current alpha scope

- The bundled native CPU solver and guarded experimental external GPU routes
  support their documented fixed-grid and adaptive configurations. GPU AMR is
  admitted only for exact authenticated process combinations; unsupported
  combinations reject before execution.
- Project and source-layer paths are absolute. Keep the project with its input
  data or reselect moved layers.
- The only internal hydraulic condition in this alpha is a fixed reflective
  solid wall. Bridge pillars can be represented by their footprints.
- Quantitative reservoir deposition is not yet spatially converged. The alpha
  is intended for workflow and defect testing before experimental validation.
- The current installable package is Windows x86-64. The experimental GPU
  worker, compatible CPython/CuPy environment and NVIDIA driver remain separate
  authenticated dependencies; no current Intel macOS or Linux package is
  published by this release.

