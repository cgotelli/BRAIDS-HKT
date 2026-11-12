# BRAIDS Windows alpha 69 release

BRAIDS `0.1.0-alpha.69` closes the experimental flume material definition as a
declared calibration envelope, registers the usable observations, and passes
the first real-DEM, seven-fraction MorFac-1 bed-load/Exner integration pilot.

The reference seven-class material preserves the published 2.8 mm D84. The
unpublished feed/initial distribution, particle density, and porosity remain
clearly labelled bounded calibration quantities. H2 is the sole calibration
surface and H3 is withheld. H4 is now excluded: on the locked common grid its
raw partial-domain deposition is 5.81 times the measured retained-volume
balance and it contains an undocumented longitudinal datum tilt. No correction
was inferred.

Native multi-fraction kernels now base their deterministic OpenMP crossover on
actual work dimensions rather than cell count alone. For the 4,031-cell,
seven-fraction pilot, measured native scaling was:

- 1 thread: 45.042 s;
- 8 threads: 9.249 s (4.87x);
- 16 threads: 7.219 s (6.24x);
- 32 threads: 7.922 s (5.69x).

Checkpoint, ledger, and frame files are byte-identical at all four counts. The
60-second production pilot used 16 physical cores and completed in 330.12 s.
It injected 1.82726 kg of graded sediment, conserved water to 5.55e-17 m3 and
every sediment class to at most 5.97e-11 kg, remained finite/nonnegative, and
resolved -7.15 to +31.25 mm of bed evolution with MorFac exactly 1.

Laboratory domains may now use an explicit local engineering CRS with metre
units. Geographic and non-metre coordinate systems remain rejected.

Release artifact:

- `dist/windows-x86_64/BRAIDS-0.1.0-alpha.69-QGIS-windows-x86_64.zip`
- SHA-256: `835aac120e17481e58bc7aaf570776961c8a3b9d213a14e9479897975df6aaf1`
- Size: 3,794,976 bytes
- Native executable SHA-256:
  `6347d099cfa21f8b20a0ea5fffca4658eb3ae19f45d5df56d512b1d16a9e6890`
- Verified staged/installed plugin files: 82/82, with zero differences
- Changed-code verification: 73 tests passed
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13, adaptation length 7/7,
  morphological acceleration 22/22, and bedform roughness 9/9

The installed default-profile plugin contains neither `resources/Delft3D-main`
nor downloaded experimental source data. Alpha 68 is recoverable from
`braids_qgis.backup-0.1.0-alpha.68-20260924-morfac1`. QGIS was not running
during installation, so its next launch will load alpha 69.
