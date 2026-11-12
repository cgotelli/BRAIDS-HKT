# BRAIDS Windows alpha 68 release

BRAIDS `0.1.0-alpha.68` starts the integrated experimental braided-river
qualification with a frozen Johannot et al. (2026) flume benchmark and a
measured multicore hydraulic baseline.

The selected experiment is `UC1_C1_LF_01`. Its source files are locked by
Dataverse identifier, byte count, and MD5. The conservative forcing conversion
preserves 34.070062181 m3 of inlet water and 220.5702 kg of sediment over the
14,400 s experiment. The supplied H1 raster failed spatial plausibility checks
and is excluded rather than silently repaired; H2 is the calibration surface,
while H3 and H4 remain withheld validation surfaces. Sediment qualification is
also fail-closed until the missing full grain-size distribution and particle
density are documented.

Native deck v32 now uses binary lookup for long hydraulic and sediment forcing
series. Large primitive-state, water-draining, SSPRK blend, and accepted-state
loops use deterministic OpenMP parallelism. Native QGIS processes disable
dynamic OpenMP teams and bind threads to cores. The UI defaults to detected
physical cores (16 on the release machine), while permitting all 32 logical
processors for sufficiently large grids.

Measured native scaling was:

- 65,608-cell pilot: 20.367 s on 1 thread and 5.037 s on 32 threads, a 4.04x
  speedup;
- 16,402-cell pilot: 12.287 s on 1 thread and 3.546 s on 16 threads, a 3.47x
  speedup;
- checkpoints, boundary ledgers, and all final hydraulic fields are
  byte-identical at 1, 8, 16, and 32 threads.

Release artifact:

- `dist/windows-x86_64/BRAIDS-0.1.0-alpha.68-QGIS-windows-x86_64.zip`
- SHA-256: `daaa250c36e94b55d828666450c0821b9e78d3bdcc27394ace3b3a6fba9dba2b`
- Size: 3,794,326 bytes
- Native executable SHA-256:
  `43f17bd1005e0525f5393bddfc07e7df1aa1c40840beef4f6f418a49e06003d4`
- Verified plugin files: 82
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13, adaptation length 7/7,
  morphological acceleration 22/22, and bedform roughness 9/9

The packaged 82-file plugin is installed byte-for-byte in the default QGIS
profile. Alpha 67 is recoverable from
`braids_qgis.backup-0.1.0-alpha.67-20260923-braided-benchmark`. The installed
tree contains neither `resources/Delft3D-main` nor raw validation downloads.
QGIS was not running during installation, so its next launch will load alpha
68.
