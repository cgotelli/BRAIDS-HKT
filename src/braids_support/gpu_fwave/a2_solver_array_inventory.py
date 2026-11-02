"""Enumerate GPU arrays reachable from the private A2 composed solver.

This deliberately traverses BRAIDS-owned Python objects only. It does not
walk CuPy modules, CUDA graph internals, memory pools, or Python callables.
Within that explicit scope, every reachable CuPy ndarray is recorded by
object identity and backing pointer, including aliases in state/workspaces.
"""
from __future__ import annotations

from collections.abc import Mapping


OWNED_PREFIXES=('braids_gpu','braids_support.gpu_fwave')


def gpu_arrays(solver):
    cp=solver.cp
    seen=set();found={}

    def walk(value,path):
        if isinstance(value,cp.ndarray):
            found[path]=value
            return
        if isinstance(value,(str,bytes,int,float,bool,type(None))):return
        if callable(value):return
        identity=id(value)
        if identity in seen:return
        seen.add(identity)
        if isinstance(value,Mapping):
            for key,child in value.items():walk(child,path+'['+repr(key)+']')
        elif isinstance(value,(list,tuple)):
            for index,child in enumerate(value):walk(child,path+'['+str(index)+']')
        elif type(value).__module__.startswith(OWNED_PREFIXES):
            for key,child in vars(value).items():walk(child,path+'.'+key)

    walk(solver,'solver')
    return found


def unique_storage(arrays):
    """Keep one handle for each same-pointer/same-shape/same-type alias."""
    unique={};aliases={}
    for path,array in arrays.items():
        key=(int(array.data.ptr),array.shape,array.dtype.str)
        if key not in unique:unique[key]=(path,array)
        aliases.setdefault(key,[]).append(path)
    return {path:array for path,array in unique.values()},aliases


def snapshot_host(arrays,cp):
    """Diagnostic only: retain bytes and identities, never used in a solver."""
    return {name:(array.shape,array.dtype.str,cp.asnumpy(array).tobytes())
            for name,array in arrays.items()}


def changed(before,after):
    return sorted(name for name in before.keys() & after.keys()
                  if before[name]!=after[name])
