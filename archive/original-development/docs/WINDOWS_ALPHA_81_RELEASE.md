# Windows alpha 81 optimized default solver

The optimized development implementation is promoted into the default code
build and Windows QGIS package. The requested long calculation was cancelled;
its saved results and last complete checkpoint are retained. It is not a
completed 20-hour or equilibrium qualification.

Package: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.81-QGIS-windows-x86_64.zip`.
Native SHA256: `f8c9fa06eeee0912d4e351d001cffc14c82b6755f5b9d5c6eb2c7f972c83c7d5`.
QGIS minimum remains 3.40 on 64-bit Windows. No compiler or MPI installation is
required to use the bundled Fortran/OpenMP executable and its runtime DLLs.

## What becomes standard

The package now includes the accumulated development changes since alpha 80:
reusable Euler workspace, compatible fused hydraulic preparation, measured
fixed/adaptive OpenMP improvements, cached sediment geometry and the existing
bank/bed and optional diffusion/secondary-flow developments. This is a promotion
of the current development version, not a claim that every intervening physical
feature is newly qualified. Bank-model and experimental-bend limitations remain
as described in the associated development reports.

For enabled prognostic secondary flow, the previously tested depth-scaled
curvature filter now uses length 2h and residual targets 1e-4 relative and 1e-10
absolute. This changes the unfiltered production closure used by older plugin
versions; it matches the closure in the recent bend comparisons. MIC is retained
below 65,536 cells because tested parallel alternatives were slower there.
The parallel multilevel backend is selected automatically above that threshold.
Secondary flow itself remains opt-in. No bed smoothing is introduced.

GeoClaw's source, WENO/limited sediment reconstruction, SSPRK3, CFL settings,
bedload formula selection and the production MorFac 1--20 guards are unchanged.
The MF25 laboratory override is not promoted. Delft3D source/runtime are
untouched and neither Delft3D source nor raw validation runs is packaged.

## Verification and limitations

The production filter module matches the frozen tested hybrid implementation.
All 22 preexisting native dependencies other than the main integration module
are unchanged from the pre-promotion source snapshot. The main module differs
only by the tested curvature integration point. Protected hydraulic routines
are checked directly. The new build records 24 native source identities.

[Release tests](../validation/sediment/lfm_bend_2d/reports/optimized_default_20261001_v1/release_tests.xml)
cover the independent filter oracle, stability modes, parallel repeatability,
coupled bed waves, outlet export, adaptive secondary-flow restart and package
contracts. Additional
[integration tests](../validation/sediment/lfm_bend_2d/reports/optimized_default_20261001_v1/integration_tests.xml)
exercise prognostic state, AMR, wetting and result-output paths. The generated
[audit record](../validation/sediment/lfm_bend_2d/reports/optimized_default_20261001_v1/audit.json)
retains exact test counts and runtime identities.

The focused suite passes 76 tests. The broader suite has 61 passes, one missing
historical-fixture skip and one inherited outlet assertion failure at 0.02 m
tailwater. That case enables coupled MPM but asserts a water-only critical
discharge of 0.078327 m3/s; all three tested runtimes (installed alpha 80,
pre-promotion development and alpha 81) return 0.166414 m3/s with bitwise-equal
saved states and ledgers. This establishes that the optimization did not cause
the mismatch, not that the boundary treatment is correct. The original failure
is retained in the test report; no expectation or boundary physics was changed.

All nine installed-native smoke checks also pass with PATH restricted to
Windows System32, exercising the bundled runtime libraries without a compiler
directory on PATH. Archive and installation verification covers 81 files,
including 62 source files; the installed native hash matches the development
build. An interactive QGIS GUI session was not opened for this release check.

Previous multilevel paired tests measured 8.7% less compute time on an 80,342-cell
bend replay. This is not a universal runtime prediction. Developed high-resolution
morphology and large multilevel AMR performance remain unqualified. Existing
packaged process manifests are historical qualification evidence, not renewed
certification of every process combination in this release.

## Installation and recovery

The [installation record](../validation/sediment/lfm_bend_2d/reports/optimized_default_20261001_v1/installation.json)
records archive, native and installed-file verification plus the previous-plugin
backup. The installer only replaces the BRAIDS plugin directory in the default
QGIS profile, using a recoverable directory swap. Projects, saved results,
prepared runs and custom runtimes are not rewritten. Restart QGIS and prepare
a new run to use the new default. Previously prepared or explicitly selected
custom executables keep their original identity.

The [performance audit](../validation/sediment/lfm_bend_2d/reports/optimized_default_20261001_v1/PERFORMANCE_AUDIT.md)
explains the remaining Delft3D gap and the next bounded optimization milestone.
