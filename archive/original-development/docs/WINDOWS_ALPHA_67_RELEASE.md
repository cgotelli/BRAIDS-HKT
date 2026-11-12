# BRAIDS Windows alpha 67 release

BRAIDS `0.1.0-alpha.67` completes morphodynamics Phase 4 with qualified,
blank-by-default equilibrium bedform-roughness feedback. It retains all alpha
66 and earlier qualified capabilities.

Native deck v32 couples the Fredsoe MPM equilibrium dune-height relation to
Van Rijn 1984 dune length and roughness. Grain roughness is `3 D90`, combined
quadratically with dune roughness and capped at half the local depth. A
deterministic instantaneous fixed point adds no hidden prognostic state.

The locked current-source Delft3D4 sensitivity and the matched BRAIDS run both
show a material hydraulic effect relative to the calibrated grain-only control:

- Delft3D mean water-level change: 0.3220528 m (45.22% of control depth);
- BRAIDS mean water-level change: 0.2857909 m (38.14% of control depth);
- BRAIDS/Delft3D effect ratio: 0.887404;
- paired scientific gates: 5/5; complete packaged qualification gates: 9/9.

The feature is qualified only for noncohesive, roughness-corrected MPM with
suspended exchange disabled. Normal-depth inlet control and other experimental
morphology processes fail closed. Each enabled run writes
`bedform_roughness.txt` with effective Manning, dune geometry, Nikuradse
roughness and fixed-point iterations.

Release artifact:

- `dist/windows-x86_64/BRAIDS-0.1.0-alpha.67-QGIS-windows-x86_64.zip`
- SHA-256: `12817f6be481474fe233a1b27fe409428be36f24d0751a444868838bca8f5562`
- Size: 3,790,899 bytes
- Native executable SHA-256:
  `643e588e187314a113a610c81c4bab3aefa8d1ca3feda0965ce912270754a2bd`
- Verified archive files: 80
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13, adaptation length 7/7,
  morphological acceleration 22/22, and bedform roughness 9/9

The integrated Phase 4 native regression passes 85 tests with one optional
skip. The complementary writer, QGIS-core, boundary and packager regression
passes 120 tests. Ten Delft3D runtime/source-preflight tests also pass, and the
retained bedform assessment passes again in analysis-only mode.

The verified 81-file stage is installed byte-for-byte in the default QGIS
profile. Alpha 66 is recoverable from
`BRAIDS_plugin_backups/braids_qgis_alpha66_before_alpha67`. The installed tree
contains neither `resources/Delft3D-main` nor raw validation runs. QGIS was not
running during installation, so its next launch will load alpha 67.
