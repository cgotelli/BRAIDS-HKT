"""Release compiler search registrations after eager native-library loading.

Windows AddDllDirectory cookies are not closed by Python garbage collection.
The loaded libraries and their function pointers remain owned by the solver;
only redundant search-path registrations are released. No OS monkey patch,
library unload, numerical change or modification of reference runtimes.
"""
import os
from pathlib import Path
import shutil


def release_loaded_search_directories(owner):
    if os.name!='nt':return 0
    allowed={str(Path(p).parent.resolve()).casefold() for name in ('g++','gfortran') if (p:=shutil.which(name))}
    seen=set();closed=0
    def visit(value):
        nonlocal closed
        if id(value) in seen:return
        seen.add(id(value));cls=type(value)
        if cls.__module__=='os' and cls.__name__=='_AddedDllDirectory':
            if value.path and str(Path(value.path).resolve()).casefold() in allowed:
                value.close();closed+=1
        elif isinstance(value,dict):
            for key,item in value.items():
                if key not in ('identity','data','g','initial','parameters','methods'):visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        elif cls.__module__.startswith(('braids_core.sediment.','braids_support.sediment.river_morphodynamics.')):
            visit(vars(value))
    visit(owner)
    # These two eager loaders retain their search cookies in explicit process
    # caches instead of on the solver instance. Keep the cached DLLs intact.
    from . import bend_native_hydraulics as native
    with native._lock:
        for _,directories,_ in native._libraries.values():visit(directories)
    from .bed_columns_v101 import native_runtime_v101
    if native_runtime_v101.cache_info().currsize:visit(native_runtime_v101()[2])
    return closed
