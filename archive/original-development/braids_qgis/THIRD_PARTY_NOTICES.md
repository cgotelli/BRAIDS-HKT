# Third-party notices

The bundled native BRAIDS executable incorporates GeoClaw/Clawpack Riemann
solver source distributed under the Clawpack BSD license. The archive includes
that license as `licenses/Clawpack-LICENSE.txt`.

The platform packages also bundle GNU Fortran, GNU OpenMP, Quadmath and GCC
support runtime libraries so the solver does not depend on a separate compiler
installation. The Windows package additionally carries MinGW-w64 winpthreads
and dlfcn-win32. Their license texts, runtime exception and pinned source-access
references are in the archive's `licenses` directory.
