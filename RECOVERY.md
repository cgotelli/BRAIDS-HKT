# Development recovery verified — 12 November 2026

Recovery tag: **`recovery-2026-11-12`**.

The repository contains the maintained BRAIDS source, the original development
source archive, native build inputs, dependency specifications and a tested
development entry point. See [DEVELOPMENT.md](DEVELOPMENT.md) to restore it on
another Windows PC.

## Checks performed

The verification used a fresh clone downloaded from GitHub, followed by a new
conda environment created from the repository specification. The final checks
used **Python 3.13.18**, NumPy 2.3.3, SciPy 1.16.2, GDAL 3.10.1, Rasterio 1.5.1,
Shapely 2.1.2, PySide6 6.8.3 and CuPy 14.2.0. GNU C++/Fortran 13.2.0 supplied
the native builds. The application's installed Python/runtime was not used for
the final clean-environment runs.

| Recovery check | Result |
|---|---|
| Captured installed source/resource checksums | 1,297 verified |
| Original development source inventory | 3,185 verified |
| Maintained Python and recovery-tool syntax | 771 files passed |
| Frozen geometry and native dependency input inventory | Complete for the maintained build route |
| Original development directory restoration | 3,185 files restored and checksum verified |
| Source-built GUI construction and event processing | Passed in the new environment |
| f-wave and HLLC: hydraulics only | Passed on CPU and GPU |
| f-wave and HLLC: 20° yawed turbine + MPM + prognostic secondary flow | Passed on CPU and GPU in the fresh clone/new environment |
| f-wave and HLLC: supplied suspended mass and settling, explicit Ds = 0.2 mm | Passed on CPU in the fresh clone/new environment |

The synthetic hydraulic flume has 32 cells. The turbine's local mesh refinement
produces 1,672 cells. Each solver reaches the requested 0.05 s end time with
finite water/bed state and nonnegative depths. Native libraries were built from
the checkout, and a fresh CUDA kernel cache was used for the GPU checks. No
installed solver library or old simulation input is required by these checks.

These are short software-recovery checks, not additional experimental validation
or long-duration stability tests. The existing experimental conclusions are
unchanged.

The tested environment is recorded in **105 explicit conda package builds** and
**12 CPU wheel pins**, with the optional GPU packages pinned separately. The
original working installation's environment and runtime manifests are also
retained as provenance.

## What needs a separate personal backup

The public repository does not contain personal projects, DEMs, measured
datasets, paper PDFs, historical test trees, generated simulation results or
checkpoints. To retain those, keep the selected projects with their data/input
folders. The original Desktop backup shortlist identifies the useful Bern and
Zin projects, laboratory generator, turbine validation examples and source data.

The original installed application and its first release remain unchanged.
This recovery adds a source-built development route and a named source tag;
it does not create a new installer or release.
