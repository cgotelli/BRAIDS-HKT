# BRAIDS repository cleanup — 15 September 2026

The development workspace was reduced from **9.1 GiB to 296 MiB** before the
first Git commit. The cleanup removed 8.21 GiB in its main pass and then pruned
data-only validation and upstream example trees. The retained working tree is
about **229 MiB**. Its remaining size consists of
source, tests, documentation, compact fixtures and validation evidence.

Removed material included:

- generated solver runs, checkpoints, frames, temporary profiles and case logs;
- native build intermediates, Python caches and virtual environments;
- copied upstream repositories and their Git histories;
- superseded plugin packages and old installed-plugin backups;
- archived diagnostic case trees that can be regenerated from retained scripts.

Retained material includes:

- `braids_core`, `braids_qgis`, the Fortran/OpenMP solver and build tools;
- the Clawpack 5.14.0 source required to compile the native solver;
- unit/native/restart/parallel tests and compact test fixtures;
- validation scripts, specifications, final reports and representative figures;
- the reservoir DEM, losslessly DEFLATE-compressed from 84,003,066 to
  38,500,656 bytes without changing its raster values, geotransform or CRS;
- the verified `BRAIDS-0.1.0-alpha.37-QGIS-macos-arm64.zip` package locally
  under `dist/` (ignored by Git).

The large comparison repositories and their exact revisions are recorded in
`THIRD_PARTY_SOURCES.md` and restored with
`python tools/fetch_reference_sources.py`. The repeatable cleanup command is
`python tools/clean_repository.py --apply`.

Qualification after cleanup:

- native BRAIDS solver rebuilt successfully with GNU Fortran and OpenMP;
- compact-checkout suite: **2,101 passed, 88 skipped** with eight OpenMP threads;
- the skips are explicitly gated historical upstream/G3 oracles and optional
  paper/source inputs;
- packaged QGIS 3.44.12 installed-profile smoke: **120/120 passed**;
- no retained path, source identifier or package entry contains the former
  project name.
