"""Process-local reuse of immutable compiled operators, never physical arrays.

Source and binary content hashes remain the callers' responsibility. A native
library is shared only under its content identity. CUDA modules are scoped to
the selected device and exact options/entry points. Threads are set per launch.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import threading

_lock=threading.RLock()
_libraries={}
_modules={}


@lru_cache(maxsize=16)
def _compiler_version(path, signature):
    return subprocess.check_output([path,'--version'],text=True).splitlines()[0]


def compiler_version(path):
    p=Path(path).resolve();s=p.stat()
    return _compiler_version(str(p),(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))


def native_library(path, sha256):
    key=(str(Path(path).resolve()),sha256)
    with _lock:
        if any(k[0]==key[0] and k[1]!=sha256 for k in _libraries):
            raise ValueError('Loaded native library changed on disk; start a fresh worker')
        if key not in _libraries:_libraries[key]=ctypes.CDLL(str(path))
        return _libraries[key]


def cuda_module(cp, code, options, names):
    key=(cp.cuda.runtime.getDevice(),code,tuple(options),tuple(names))
    with _lock:
        if key not in _modules:
            module=cp.RawModule(code=code,options=options,name_expressions=names)
            _modules[key]=(module,{name:module.get_function(name) for name in names})
        module,functions=_modules[key]
        return module,functions.copy()
