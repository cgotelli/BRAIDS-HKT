# BRAIDS Windows alpha 70 release

BRAIDS `0.1.0-alpha.70` qualifies the physical dry-start workflow, completes
the first fixed-grid/timestep screen, and substantially accelerates the exact
multicore bedload-only path.

The 8,173-node inlet survey is represented by 755 nodes with a hard 0.25 mm
vertical-error bound. Across the measured discharge range, normal stage changes
by at most 0.071 mm. Complete 14,453-knot preparation is 24.44 times faster
with identical serialized decks.

The 60-second timestep comparison passes. The 0.05 m grid is rejected because
its natively integrated absolute bed-change volume differs from 0.025 m by
10.95%, above the frozen
10% gate. Bed MAE is 0.108 mm, spatial correlation is 0.819, and all water,
fraction-conservation, finite-state and nonnegative-state checks pass. The next
qualification run is the 0.0125 m comparison.

The optimized native path bypasses only exact-zero suspended work and remote
dry-cell bedload work. Live normal/critical section terrain uses deterministic
`O(n log n)` sorting, allocation-free repeated metrics, and OpenMP-parallel
root intervals. Any active suspended process restores the complete path.

On the 0.025 m, 16,124-cell pilot, 16-thread runtime fell from 37.209 to 13.142
seconds (2.83x). Frames and boundary ledgers are byte-identical to alpha 69.
Fresh 1/8/16/32-thread outputs are also byte-identical. Native profiling now
reports boundary control, bedload capacity, reconstruction, boundary traces,
face flux, limiting/update and finalization separately.

Release artifact:

- `dist/windows-x86_64/BRAIDS-0.1.0-alpha.70-QGIS-windows-x86_64.zip`
- SHA-256: `f69012ed31cbcd606f75ac0473895a5581131de7e85a0ddd4111a7a465b52def`
- Size: 3,801,277 bytes
- Native executable SHA-256:
  `2ce1487f9bf781cfd194154fcba87b609bbaa6bdb47504425139fd2fd8a916ac`
- Verified staged/installed plugin files: 82/82, with zero differences
- Changed-code verification: 120 tests passed
- Packaged qualification gates: Phase 3 74/74, general avalanche 28/28,
  secondary flow 15/15, bank failure 13/13, adaptation length 7/7,
  morphological acceleration 22/22, and bedform roughness 9/9

The installed default-profile plugin contains neither `resources/Delft3D-main`
nor raw validation run directories. Alpha 69 is recoverable from
`braids_qgis.backup-0.1.0-alpha.69-20260924-startup-runtime`. QGIS was not
running during installation, so its next launch will load alpha 70.
