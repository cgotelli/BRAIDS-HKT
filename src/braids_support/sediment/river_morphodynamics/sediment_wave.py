"""Research-only sediment-wave direction; no public method or plugin change.

Mahdizadeh & Sharifi, doi:10.1016/j.ijsrc.2020.05.001, Eqs 56--59:
use the capacity/bed secant to determine sediment wave direction. This is a
2D face-normal adaptation with the existing limited second-order traces. It
is NOT a reproduction of their complete 1D solver or its time ordering.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from types import MethodType

from braids_core.sediment.bend_methods_v1 import digest, OPTIONS
from braids_core.sediment.flow2dh_mobile_bed import MobileEngine, ROOT
from braids_core.windows_paths import filesystem_path
from braids_core.sediment.river_kernel_cache import compiler_version,native_library,cuda_module


class WaveEngine(MobileEngine):
    def __init__(self,backend,threads,*,matrix=False,source=None,names=None):
        self.backend=backend;self.threads=threads
        path=filesystem_path(source) if source else (Path(__file__).with_name('matrix_wave.h') if matrix else Path(__file__).with_suffix('.h'))
        names=names or (('wave_state_gradients','matrix_bed_faces') if matrix else ('sediment_wave_faces',))
        self.identity=dict(backend=backend,threads=threads,source_sha256=digest(path))
        if backend=='gpu':
            import cupy as cp
            self.module,self.fn=cuda_module(cp,path.read_text(),OPTIONS,names)
        else:
            compiler=shutil.which('g++')
            if not compiler:raise RuntimeError('OpenMP compiler required')
            version=compiler_version(compiler)
            flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off']
            key=hashlib.sha256((version+str(flags)+path.read_text()).encode()).hexdigest()[:20]
            folder=ROOT/'build/shoreline_sediment_wave'/key;folder.mkdir(parents=True,exist_ok=True)
            lib=folder/'wave.dll'
            # Compile byte-identical source from the content-addressed cache,
            # never a long/Unicode user workspace path passed to narrow GCC.
            frozen=folder/'source.h'
            if not frozen.exists():shutil.copyfile(path,frozen)
            if digest(frozen)!=digest(path):raise ValueError('Native wave source copy differs')
            if not lib.exists():subprocess.run([compiler,*flags,frozen.name,'-o',lib.name],cwd=folder,
                check=True,capture_output=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
            try:
                library_sha256=digest(lib)
                self.lib=native_library(lib,library_sha256);self.fn={name:getattr(self.lib,name) for name in names}
                for fn in self.fn.values():fn.restype=None
            finally:
                # CDLL eagerly loads its dependencies. Search registrations are
                # temporary; os._AddedDllDirectory has no garbage-collection
                # cleanup. Repeated regrid/transfer engines otherwise exhaust
                # Windows' directory list even after their Python objects die.
                for directory in self.dirs:directory.close()
                self.dirs=[]
            self.identity.update(compiler=version,flags=flags,library_sha256=library_sha256)


def install(physics,*,matrix=False):
    """Attach only to an explicitly created validation owner, before stepping."""
    if physics.time or physics.steps or physics._open:raise ValueError('Install before execution')
    if hasattr(physics,'wave_engine'):raise ValueError('Wave correction already installed')
    physics.wave_engine=WaveEngine(physics.engine.backend,physics.flow.threads,matrix=matrix)
    physics.wave_diagnostic=physics.xp.zeros((3 if matrix else 2,physics.nf))
    if matrix:physics.wave_state_gradient=physics.xp.zeros((8,physics.n));physics.wave_q=physics.xp.zeros((3,physics.n))
    previous_bed_faces=physics.bed_faces
    def bed_faces(p,q,z,hi):
        if matrix:p.wave_q[:]=q
        previous_bed_faces(q,z,hi)
    physics.bed_faces=MethodType(bed_faces,physics)
    old=physics.reconstruct_capacity
    physics.wave_base_reconstruction=old
    def reconstructed(p):
        old();g=p.g
        # The shallow transverse assembly needs the physical MUSCL normals,
        # before the wave term. Preserve them here instead of reconstructing
        # every class a second time after the wave calculation.
        if hasattr(p,'wave_physical_normal'):
            p.wave_physical_normal[:]=p.xp.sum(p.bedface[:2]*g['normal'],axis=0)
        if matrix:
            # Wave gradients require hydraulic states, including neighbours
            # of active sediment cells. A 1 mm division-safe sediment scratch
            # is not physical water and contaminated those stencils even when
            # the final inactive sediment face was closed. Keep the hydraulic
            # dry floor and zero truly dry momentum instead, on both backends.
            safe=p.wave_q.copy();safe[0]=p.xp.maximum(safe[0],p.parameters.dry)
            safe[1:,p.wave_q[0]<=p.parameters.dry]=0.
            p.wave_engine.launch('wave_state_gradients',p.n,p.n,p.nf,safe,p.z0,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],g['displacement'],p.wave_state_gradient)
            p.wave_engine.launch('matrix_bed_faces',p.nf,p.nf,p.n,float(p.morfac/(1-p.parameters.porosity)),
                safe,p.z0,p.bedcell,p.capacity_gradient,p.wave_state_gradient,g['owner'],g['neighbour'],g['normal'],g['displacement'],p.bedface,p.wave_diagnostic)
            o=g['owner'];j=g['neighbour'];active=(p.wave_q[0,o]>.001)&(p.wave_q[0,p.xp.maximum(j,0)]>.001)&(j>=0)
            # Inactive sediment faces are closed by ShorelinePhysics; their
            # division-safe scratch is not a physical state to qualify.
            p.wave_diagnostic[:,~active]=0.
            if bool(p.xp.any(p.wave_diagnostic[2])):raise ValueError('Research wave Jacobian rejected; no silent fallback')
        else:
            p.wave_engine.launch('sediment_wave_faces',p.nf,p.nf,p.n,float(p.morfac/(1-p.parameters.porosity)),
                p.z0,p.bedcell,p.capacity_gradient,g['owner'],g['neighbour'],g['normal'],g['displacement'],p.bedface,p.wave_diagnostic)
    physics.reconstruct_capacity=MethodType(reconstructed,physics)
    physics.identity.update(sediment_wave_research=dict(runtime=physics.wave_engine.identity,
        wrapper_sha256=digest(__file__),method='coupled_Jacobian_bed_row_fwave' if matrix else 'capacity_bed_secant_direction_with_existing_MUSCL',
        reference='BRAIDS coupled_exner_flux matrix-sign splitting, bed row only' if matrix else '10.1016/j.ijsrc.2020.05.001 Eqs 56-59',public_support=False,
        wave_state='hydraulic depth and dry momentum; no sediment scratch in gradients' if matrix else None,
        caveat='Frozen local capacity coefficients; augmented hydraulic flux retained; not a full coupled Riemann solver' if matrix else '2D adaptation; original hydraulic-first splitting retained; no claim of paper replication'))
    return physics
