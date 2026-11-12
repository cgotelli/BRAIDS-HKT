# BRAIDS Windows alpha 66 release

BRAIDS `0.1.0-alpha.66` qualifies constant and time-varying morphological
acceleration for the Windows QGIS plugin. It retains all alpha 65 and earlier
qualified morphodynamics capabilities.

The new Sediment control is blank by default. When enabled, native deck v31
uses Delft3D's `dtmor = hydraulic dt * MorFac` semantics: the conservative
bed-load/Exner flux is accelerated, while transport capacity is unchanged.
Constant factors 1--20 and block-interpolated schedules are supported.

The paired locked-source qualification establishes deliberately different
regime limits:

- quasi-steady subcritical translating bedform: maximum MorFac 20;
- wet/dry hydraulic transient: maximum MorFac 1.

All 22 retained source, morphological-clock, budget and classification gates
pass. Multi-fraction conservation, AMR reflux, exact restart and one/eight-
thread identity are covered. Suspended exchange, time-varying hydraulic
boundaries and both bank-erosion models fail closed. Every accelerated run
writes `morphological_acceleration.txt` with hydraulic time, integrated
morphological time and instantaneous factor.

Release artifact:

- `dist/windows-x86_64/BRAIDS-0.1.0-alpha.66-QGIS-windows-x86_64.zip`
- SHA-256: `f29a37f9f9a56501e2dc7068b3d40790820fa2503e79da1caab18d7851fb08cb`
- Size: 3,785,786 bytes
- Native executable SHA-256:
  `ed1cb4be58c387abbe762f2102688986e93da283c3ec6a5572a99e86115836c2`
- Verified archive files: 79
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13, adaptation length 7/7, and
  morphological acceleration 22/22

The integrated morphodynamics regression passes 171 tests with one optional
skip; the overlapping API/QGIS-core/packager selection passes 103 tests. The
archive and installed tree contain neither `resources/Delft3D-main` nor raw
validation runs.

The verified stage is installed in the default QGIS profile. Alpha 65 is
recoverable from
`BRAIDS_plugin_backups/braids_qgis_alpha65_before_alpha66`. QGIS had an
unsaved project open during installation, so it was deliberately left running.
Restart QGIS when convenient to load alpha 66.
