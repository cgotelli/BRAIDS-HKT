# BRAIDS Windows alpha 64 release

BRAIDS `0.1.0-alpha.64` qualifies process-separated toe erosion, retained
undercut history and discrete bank-block failure. The model is exposed as a
default-off QGIS control and is mutually exclusive with the older Delft3D-style
dry-bank replenishment option.

## Scientific qualification

All 13 frozen gates pass:

- exact analytic toe-eroded volume, failed-block volume, residual undercut and
  collapse count;
- zero toe erosion below the configured critical shear;
- exact two-fraction sediment conservation in the manufactured case;
- correct depth-response direction and undercut-then-collapse process order
  against Lotsari et al. (2025);
- current-source D-Flow FM dry-bank transfer agreement within
  `2.31e-14 kg/m2`, plus source identity and scope checks;
- native fixed-grid/AMR, one/eight-thread and checkpoint-restart regression.

The production v29 operator derives near-bank shear from local depth, velocity
and Manning roughness; applies the user-configured excess-shear law; preserves
undercut memory through AMR and restart; releases the remaining overhang as a
discrete conservative block; and can couple to the separately qualified
domain-wide repose operator.

The qualified envelope does not include predictive geotechnical factor of
safety, pore pressure, seepage, thermal erosion or a moving vertical overhang
mesh. Erodibility, critical shear and critical undercut require project-specific
calibration.

## Windows artifact

- Archive: `dist/windows-x86_64/BRAIDS-0.1.0-alpha.64-QGIS-windows-x86_64.zip`
- SHA-256: `7c0d7639e3b8f31a1dc03398b471f9b5b236b78fb3927501ea97bf7703ccefca`
- Size: 3,770,410 bytes
- Native executable SHA-256:
  `252dfcd1e082a83f2c88764eb70f86178a815ff535708f79c4bb7df379036bca`
- Verified archive files: 77
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13

The builder verified the current native source hashes, pinned runtime DLLs,
license files, every archive member and all four packaged qualification
manifests. It rejected Delft3D source and raw validation-run paths.

## Installed profile

The verified stage was copied to the default QGIS 3 profile. Its installed
manifest and native executable hashes exactly match the stage. Alpha 63 remains
recoverable at
`braids_qgis.backup-0.1.0-alpha.63-20260923-phase4-bank`. QGIS was running
during installation, so it must be restarted before alpha 64 is loaded.

The release-relevant suite passes 119 tests with one environment-guarded skip.
The complete historical-suite collection remains unavailable in this trimmed
checkout because legacy hydraulics validation packages and optional `xarray`
and `mpmath` dependencies are absent.
