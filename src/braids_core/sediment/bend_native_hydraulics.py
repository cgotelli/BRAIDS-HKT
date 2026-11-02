"""Isolated native reconstructed f-wave library. Does NOT rebuild production.

Only the Euler routine's dt-independent prefix is exposed; the augmented solver,
WENO/reconstruction, traces and boundary closures are not edited. One immutable
case per library instance, serialized because Fortran has module workspaces.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import numpy as np
from .bend_methods_v1 import ROOT,HERE,digest,BendTrialRejected
from braids_core.windows_paths import filesystem_path
from braids_core.windows_native_io import native_directory

_libraries={}
_lock=threading.RLock()


def _publish_instance(library,instance):
    """Publish complete bytes atomically across independently launched runs.

    The thread lock cannot protect another Python process. A direct copy to
    the shared instance path previously exposed partial bytes to that process.
    Hard-link publication within one directory is atomic and never overwrites
    a concurrently loaded DLL. Keep the checksum failure for genuine damage.
    """
    expected=digest(library)
    if not instance.exists():
        fd,name=tempfile.mkstemp(prefix='hydraulic_instance_',suffix='.tmp',dir=instance.parent)
        os.close(fd);temporary=Path(name)
        try:
            shutil.copy2(library,temporary)
            if digest(temporary)!=expected:raise ValueError('Native source changed during instance copy')
            try:os.link(temporary,instance)
            except FileExistsError:pass
        finally:temporary.unlink(missing_ok=True)
    if digest(instance)!=expected:raise ValueError('Native instance checksum differs')


def build():
    from tools.build_cutcell import SOURCES
    compiler=shutil.which('gfortran')
    if not compiler: raise RuntimeError('Native Fortran/OpenMP compiler required')
    version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    parts=[ROOT/name for name in SOURCES]
    extras=[Path(__file__),HERE/'bend_native_bridge.f90',HERE/'bend_native_rates_tail.inc']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in parts+extras}
    key=hashlib.sha256(json.dumps([hashes,version],sort_keys=True).encode()).hexdigest()[:20]
    folder=ROOT/'build/bend_native_hydraulics'/key;folder.mkdir(parents=True,exist_ok=True)
    library=folder/('hydraulics.dll' if os.name=='nt' else 'hydraulics.so')
    if not library.exists():
        original=parts[-1].read_text()
        begin=original.index(' subroutine euler(')
        stop=original.index('  ! Conservative water draining changes',begin)
        prefix=original[begin:stop].replace('subroutine euler(','subroutine bend_native_spatial_rates(',1)
        module_end=original.index('\nend module',stop)
        transformed=original[:module_end]+'\n'+prefix+extras[2].read_text()+'\n'+extras[1].read_text()+'\nend module\n'
        (folder/'cutcell_bend_isolated.f90').write_text(transformed)
        # Preserve exact dependency bytes next to the measured binary.
        frozen=folder/'source';frozen.mkdir(exist_ok=True)
        for p in parts+extras:shutil.copy2(p,frozen/p.name)
        command=[compiler,'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',
                 *[str(frozen/p.name) for p in parts[:-1]],str(folder/'cutcell_bend_isolated.f90'),'-o',str(library)]
        if os.name=='nt':command+=['-Wl,--export-all-symbols']
        result=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=180)
        (folder/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Native bridge compilation failed: '+str(folder/'build.log'))
        if hashes!={str(p.relative_to(ROOT)):digest(p) for p in parts+extras}:raise RuntimeError('Sources changed during build')
        (folder/'identity.json').write_text(json.dumps(dict(source_sha256=hashes,compiler=version,
            command=command,library_sha256=digest(library),prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
            production_modified=False,delft_modified=False),indent=2))
    identity=json.loads((folder/'identity.json').read_text())
    if digest(library)!=identity['library_sha256']:raise ValueError('Native library checksum differs')
    return library,identity,compiler


class NativeHydraulics:
    def __init__(self,deck,data,threads=4):
        if not isinstance(threads,int) or threads<1:raise ValueError('Positive thread count required')
        self.nc,self.nf,self.threads=int(data['nc']),int(data['nf']),threads
        self.deck=filesystem_path(deck).resolve()
        if not self.deck.is_file():raise FileNotFoundError(self.deck)
        self.initial={k:np.asarray(data[k]).copy() for k in ('area','owner','neighbour','state','rock')}
        files=sorted(self.deck.parent.glob(self.deck.name+'*'))
        self.input_hashes={p.name:digest(p) for p in files if p.is_file()}
        library,identity,compiler=build()
        casekey=hashlib.sha256(json.dumps(self.input_hashes,sort_keys=True).encode()).hexdigest()[:20]
        # Each case uses a separate OS library instance, not a reload of allocated globals.
        instance=library.parent/casekey/library.name;instance.parent.mkdir(exist_ok=True)
        with _lock:
            if str(instance) not in _libraries:
                _publish_instance(library,instance)
                dll_dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
                lib=ctypes.CDLL(str(instance))
                lib.bend_native_load.argtypes=[ctypes.c_char_p,*([ctypes.c_int]*4),ctypes.POINTER(ctypes.c_int)]
                ptr=ctypes.c_void_p
                lib.bend_native_rates.argtypes=[ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_int,ptr,ptr,ctypes.POINTER(ctypes.c_int)]
                lib.bend_native_geometry.argtypes=[ptr]*9
                lib.bend_native_load.restype=lib.bend_native_rates.restype=lib.bend_native_geometry.restype=None
                code=ctypes.c_int();path=self.deck.name.encode('ascii')
                with native_directory(self.deck.parent, self.input_hashes) as native_folder:
                    previous=Path.cwd()
                    try:
                        os.chdir(native_folder)
                        lib.bend_native_load(path,len(path),self.nc,self.nf,threads,ctypes.byref(code))
                    finally:os.chdir(previous)
                if code.value:raise ValueError('Native case not a hydraulic-only single-grain carrier')
                native=dict(state=np.empty((5,self.nc)),rock=np.empty(self.nc),xy=np.empty((2,self.nc)),area=np.empty(self.nc),
                    owner=np.empty(self.nf,np.int32),neighbour=np.empty(self.nf,np.int32),bid=np.empty(self.nf,np.int32),
                    normal=np.empty((2,self.nf)),length=np.empty(self.nf))
                lib.bend_native_geometry(*(a.ctypes.data for a in native.values()))
                _libraries[str(instance)]=(lib,dll_dirs,native)
            self.lib=_libraries[str(instance)][0]
            for name,a in _libraries[str(instance)][2].items():
                if not np.array_equal(a,data[name]):raise ValueError('Native prepared input differs: '+name)
        self.identity=dict(runtime=identity,inputs=self.input_hashes,threads=threads,
                           hydraulic_spatial_operator='native_production_reconstructed_augmented_fwave')
        self.calls=0

    def rates(self,q,z,dt,time):
        q=np.ascontiguousarray(q,dtype=np.float64);z=np.ascontiguousarray(z,dtype=np.float64)
        if (q.shape!=(3,self.nc) or z.shape!=(self.nc,) or not np.isfinite(q).all() or
                not np.isfinite(z).all() or np.any(q[0]<=0) or not np.isfinite([dt,time]).all() or dt<=0):
            raise BendTrialRejected('Native hydraulic input shape/domain invalid')
        rate=np.empty_like(q);water=np.empty(self.nf);code=ctypes.c_int()
        with _lock:
            self.lib.bend_native_rates(q.ctypes.data,z.ctypes.data,float(dt),float(time),self.threads,
                                      rate.ctypes.data,water.ctypes.data,ctypes.byref(code))
        self.calls+=1
        if code.value:raise BendTrialRejected('Native reconstructed residual rejected')
        return rate,water
