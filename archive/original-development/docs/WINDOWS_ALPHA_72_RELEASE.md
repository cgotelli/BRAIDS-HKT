# BRAIDS Windows alpha 72 release

BRAIDS `0.1.0-alpha.72` accelerates exact multi-fraction morphodynamics on
Windows and qualifies the production-size experimental state across 1, 8, 16,
and 32 OpenMP threads.

The roughness-corrected MPM path shares cell hydraulics across fractions,
avoids unused mixture descriptors, skips dry-cell percentile work, and removes
20 bisection iterations that occur after double-precision convergence. The
native build also enables portable whole-program optimisation.

The QGIS and synchronous launchers now use an active GNU OpenMP worker pool and
disable unsupported Windows affinity binding. On the fully wet 64,496-cell,
seven-fraction continuation, 32 cores are 4.31x faster end-to-end, 6.38x faster
in Euler work, 14.63x faster in bedload capacity, and 4.74x faster in face flux
than one core. Every frame, boundary ledger and checkpoint is byte-identical
across 1/8/16/32 threads.

The small 16,124-cell dry-start case remains below the strong-scaling
crossover, so using every logical processor is not automatically faster for a
small or mostly dry domain. BRAIDS continues to honor the chosen thread count;
the 32-core path is qualified for large, wet states.

The Windows package contains neither `resources/Delft3D-main` nor raw
validation run directories.

## Verified Windows artifact

- Archive: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.72-QGIS-windows-x86_64.zip`
- SHA-256: `3e6ecb7964c691342325c8fdbef256156471fdb615e3daa65bdd8df6beb0e8be`
- Size: `3,799,125` bytes
- Verified packaged files: `81`
- Native executable SHA-256:
  `7e5707632b713850a89ab760c43c309a185999566d2e7aaf91ef6595ac4a0117`

The focused solver/QGIS regression passes 46 tests. A broader available-source
run passes 933 tests and skips 5; its remaining collection/errors are confined
to intentionally absent legacy hydraulic fixtures, macOS-only sparse-file
tests, and unrelated removed WAVE source paths. The package manifest verifies
all 81 files.

The staged and installed manifests are identical. Alpha 72 is installed in the
default QGIS profile; alpha 71 remains recoverable at
`braids_qgis.backup-0.1.0-alpha.71-20260924-multicore`.
