# Original development sources

`development-sources.json` inventories the source-only union of both halves of
the development workspace during the interrupted C: to D: move on 12 November
2026. It records 3,185 files with SHA-256 checksums. Identical files point into
`src/`; 2,463 additional or different files are retained under
`original-development/` (about 47 MB).

This preserves earlier solver implementations, GPU kernels, geometry exporters,
QGIS integration, maintenance scripts and design notes alongside the maintained
Desktop source. Historical scripts can contain original machine paths and
references to external experiment inputs.

The maintained application runs from the `src/` snapshot using `tools/develop.py`.
To recover the earlier directory layout for reference, run:

```powershell
python tools/restore_archived_sources.py D:\BRAIDS-original-sources
```

The destination must not already exist. The restoration verifies every source
checksum before copying. Generated libraries, environments, simulations, personal
projects, experimental data and the historical test tree are outside this archive.
