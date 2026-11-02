# Third-party notices

BRAIDS-HKT includes original BRAIDS developments and adapted components. Original copyright and license notices in source files remain applicable.

| Component | Origin | License |
|---|---|---|
| GeoClaw / Clawpack numerical foundation | https://github.com/clawpack | BSD-3-Clause; see `LICENSES/` |
| AVAC Desktop interface components | https://github.com/cgotelli/AVAC-Desktop | AGPL-3.0-or-later |
| Python | https://www.python.org | Python Software Foundation license |
| PySide6 / Qt | https://www.qt.io/qt-for-python | LGPL/GPL and component-specific licenses, supplied with the installer |
| NumPy, SciPy, Matplotlib, GDAL, Rasterio, Shapely, CuPy and supporting libraries | Respective upstream projects | Their individual notices are retained in the installed Python packages |
| NVIDIA CUDA runtime libraries | https://developer.nvidia.com/cuda-toolkit | NVIDIA CUDA redistributable terms; notices included in installed packages |
| GNU compiler runtime libraries | https://gcc.gnu.org | GPL with GCC Runtime Library Exception; see `LICENSES/` |

The Windows installer uses dynamically linked third-party libraries. Their versions are listed in `packaging/dependencies.json`; their notices accompany the installed distributions. The installer also includes these project notices. Scientific methods cited inside the implementation remain attributed to their authors. No third-party reference solver is distributed as an application backend.
