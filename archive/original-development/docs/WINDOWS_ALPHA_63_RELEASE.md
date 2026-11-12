# BRAIDS Windows alpha 63 release

BRAIDS `0.1.0-alpha.63` qualifies equilibrium secondary-flow steering of
magnitude-based bed-load transport and exposes it as a default-off QGIS
control with an explicit `Espir` coefficient.

## Scientific qualification

The retained mild-bend and symmetric-bifurcation comparison uses the locked
current-source D-Flow FM commit
`c05ed672f0a7fe81c5b9867ef354cfdd53c25379`. All 15 frozen gates pass:

- curvature correlation: 0.9549 to 0.9789;
- equilibrium spiral-intensity correlation: 0.9772 to 0.9913;
- per-fraction secondary-on/off Exner correlation: 0.9516 to 0.9596;
- mean transport-direction difference: 4.38 to 6.58 degrees;
- closed-domain per-fraction residual magnitude below `1e-8 kg`;
- correct fine/coarse transverse sorting and symmetric branch partition;
- branch-load scale within the frozen 0.4 to 1.2 cross-model envelope.

The native regression additionally proves scalar transport-magnitude
preservation, exact one/eight-thread identity, exact checkpoint restart,
per-fraction conservation and identical results while AMR changes the active
mesh. The model changes sediment direction only: `BetaSpiral=0`, no secondary-
flow momentum force, and no Coriolis term.

## Windows artifact

- Archive: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.63-QGIS-windows-x86_64.zip`
- SHA-256: `f32609d465c328cb2ea0a97d529d0d179f6bc449e628577f3eee5b05014228cd`
- Size: 3,755,367 bytes
- Native executable SHA-256:
  `58814862d7dd78d6dcb063bc81127326d3fd2fe6476644a1700df76dc351de98`
- Verified archive files: 76
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15

The builder verified the current native source hashes, pinned runtime DLLs,
license files, every archive member, and all three packaged qualification
manifests. It rejected Delft3D source and raw validation-run paths.

## Installed profile

The verified stage was copied to the default QGIS 3 profile. Its installed
manifest hash exactly matches the staged manifest. Alpha 62 remains recoverable
at `braids_qgis.backup-0.1.0-alpha.62-20260923-phase4-secondary`. Because QGIS
was running during installation, restart QGIS before testing alpha 63.

The release-relevant regression set passed 114 tests with one environment-
guarded skip. A complete historical-suite collection was not possible in this
trimmed checkout because legacy hydraulics fixture packages and optional
`shapely`, `xarray`, and `mpmath` dependencies are absent; none is used by the
alpha 63 implementation or its retained qualification.
