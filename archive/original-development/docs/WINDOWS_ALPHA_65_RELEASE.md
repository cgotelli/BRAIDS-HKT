# BRAIDS Windows alpha 65 release

BRAIDS `0.1.0-alpha.65` qualifies optional non-equilibrium bed-load adaptation
for the Windows QGIS plugin. It retains alpha 64 bank failure, alpha 63
secondary flow, alpha 62 general avalanching, and all Phase 2/3 capabilities.

The new Sediment control is blank by default. When enabled, every grain class
uses the steady relation `L_g dq_g/ds = q_eq,g - q_g`, with
`L_g = C_L D50_g`. The editable default `C_L=100` is a laboratory-supported
starting value and must be calibrated for the application.

Qualification includes:

- pinned current-source Delft3D `tranb2` versus BRAIDS local-equilibrium MPM;
- measured gravel capacity-step and rapidly varied crest materiality decisions;
- exact exponential fixed- and mixed-level-AMR profiles;
- independent per-fraction grain-scaled lengths;
- conservation and nonnegative wet/dry evolution;
- bitwise checkpoint restart and one/eight-thread identity; and
- execution of the actual parallel native relaxation loop.

The installer excludes `resources/Delft3D-main`, raw validation runs and all
compiler source dependencies. QGIS 3.40 or newer on 64-bit Windows is required.
Restart QGIS after installation if an earlier BRAIDS DLL is loaded.

Release artifact:

- `dist/windows-x86_64/BRAIDS-0.1.0-alpha.65-QGIS-windows-x86_64.zip`
- SHA-256: `87485d4804315b1d314f3266e730e02e0190007496702c99eb2c48219703c860`
- Size: 3,779,638 bytes
- Native executable SHA-256:
  `164a60e56940c235f6efd0572ed6d384e2dad2bcbfcf2630c0fb7466a1a56fba`
- Verified archive files: 78
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13, adaptation length 7/7

The release-relevant regression suite passes 136 tests with one optional skip.
The complete historical-suite collection remains unavailable in this trimmed
checkout because legacy hydraulics validation packages and optional `xarray`
and `mpmath` dependencies are absent.

The verified stage is installed in the default QGIS profile. The prior alpha
64 directory is recoverable at
`BRAIDS_plugin_backups/braids_qgis_alpha64_before_alpha65`. QGIS was closed
during installation; start it to load alpha 65.
