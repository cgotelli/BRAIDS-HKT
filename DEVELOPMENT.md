# Recovering development on another Windows PC

The maintained application is **BRAIDS**. The repository and the original
installer retain the name **BRAIDS-HKT**. The source snapshot on `main` is newer
than the first installer in Releases; installing that release does not restore
the latest development state.

## What is preserved

- `src/`: the authenticated installed Python, C++, CUDA and Fortran sources,
  UI resources, formula catalogues, translations and frozen geometry sources.
- `src/solver/` and `src/tools/build_cutcell.py`: native solver dependencies,
  source order and compiler flags. The small wall reconstruction helper needed
  when rebuilding is included under `src/braids_support/`.
- `packaging/source-snapshot.json`: checksums of 1,297 captured source/resources.
- `packaging/installed-runtime.reference.json` and
  `packaging/native-bindings.reference.json`: installed provenance and native
  binding records. They describe the old installed payload, including compiled
  files; they are reference records, not a runnable runtime manifest.
- `archive/`: additional original development sources and design notes from
  the split workspace, with a checksum inventory and restoration command.
- `packaging/environment.reference.json`: the actual Python and 42 package
  versions found in the working installation, plus the compiler version.
- `packaging/`: the existing installer definition, launcher policy, licensing
  and original release metadata. No new installer is published by this recovery.

Projects, DEMs, measured data, simulation outputs and historical validation
evidence require a separate personal backup. They are not needed to start the
application or generate a synthetic project, but cannot be recovered from this
public repository. The old test tree was excluded from the public source upload;
the recovery commands below supply independent synthetic acceptance checks.

## Environment

Use Windows x64. The recovery checks used Python 3.13.5, GNU C++ and Fortran
13.2.0 (MinGW-w64, UCRT, POSIX threads, SEH, winlibs r8), with OpenMP. Install
matching `g++.exe` and `gfortran.exe`; both must be in the same compiler `bin`
directory. The [upstream winlibs release](https://github.com/brechtsanders/winlibs_mingw/releases/tag/13.2.0posix-18.1.5-11.0.1-ucrt-r8)
provides that compiler generation; choose its `x86_64`, POSIX, SEH, UCRT archive.
Keep that distribution's runtime DLLs alongside the compilers. A QGIS Python
environment is not required for the standalone application.

From Miniforge or another conda installation:

```powershell
git clone https://github.com/cgotelli/BRAIDS-HKT.git
cd BRAIDS-HKT
conda env create -f environment-cpu.yml
conda activate braids-dev
python tools/audit_checkout.py
python tools/develop.py --compiler-bin C:\toolchains\mingw64\bin --smoke
python tools/develop.py --compiler-bin C:\toolchains\mingw64\bin
```

`environment-cpu.yml` specifies the core numerical and GUI dependencies. It is
an environment specification, not a byte-identical conda lock. The exact original
package inventory is retained separately. GDAL comes from conda-forge; Rasterio
and Shapely use their wheels, which supply independent GDAL/GEOS libraries. This
matches the original mixed environment and avoids incompatible conda GEOS pins.
Additional historical utilities may
use optional packages listed in that inventory (for example BASEmesh or fluidfoam).

For GPU operation, install a compatible NVIDIA driver and then:

```powershell
python -m pip install -r requirements-gpu.txt
```

The GPU requirements record the working CUDA 12 Python runtime/NVRTC components.
Source CUDA kernels remain in the repository and CuPy compiles them as needed.
GPU execution is selected explicitly in the application.

## Source-built runtime and checks

```powershell
python tools/check_recovery.py --compiler-bin C:\toolchains\mingw64\bin
python tools/check_recovery.py --compiler-bin C:\toolchains\mingw64\bin --features combined
python tools/check_recovery.py --compiler-bin C:\toolchains\mingw64\bin --backend gpu
python tools/check_recovery.py --compiler-bin C:\toolchains\mingw64\bin --backend gpu --features combined
```

The check creates its own small DEM and project, prepares a mesh and runs both
f-wave and HLLC. `combined` adds a fixed-axis, 20-degree yawed rotor, MPM bedload
and prognostic secondary flow. It verifies completion, finite hydraulic/bed
states and nonnegative depths. These short checks establish software recovery;
they do not repeat the experimental validation programme.

`tools/develop.py` stages all maintained sources in
`.development/<source-identity>/runtime`. Original numerical constructors compile
the CPU libraries there. The development stage adapts only the two bootstraps
and the child process compiler search path; the numerical source is unchanged.
Every staged source is still authenticated before a worker runs. It never
reads installed solver libraries or the former C:/Users/STREEM/Desktop workspace.
The first native build is substantially slower than subsequent runs.

Run the staging command again after editing source. A new source identity
produces a separate stage; a running simulation retains its original source.
Preparation/checkpoints remain bound to that runtime identity. Generate a new
preparation after code changes; use the application's saved-state initialization
workflow when transferring a physical state between different configurations.

Do not copy `packaging/sitecustomize.py` into a development environment: it is an
installed-product policy that deliberately removes compilers from the search
path. The compiler-free release route uses authenticated `PRECOMPILED.json`
bindings; its implementation and original reference bindings are preserved in
`portable_river_runtime.py` and `packaging/`. A future installer must freeze and
qualify its newly built libraries and regenerate its runtime manifest before
using the Inno Setup definition. The source-built development route above is
available immediately and is separate from publishing another release.

## Before formatting

Fetch the recovery tag named in `RECOVERY.md` from a second machine or disk and
run `tools/audit_checkout.py`. Keep any personal project with its associated
`*_data` folder and referenced input files. A Git checkout does not back up those
inputs. The local backup shortlist identifies the small set of examples and
reference material worth retaining; no bulk move is required.
