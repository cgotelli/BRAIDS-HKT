"""Maintained wall-aware hydraulic reconstruction for the fixed-grid bend owner.

Separate content-addressed native libraries preserve active historical runs.
The augmented Riemann implementation, integration and sediment reconstruction
are unchanged. Wider worker/AMR routing is not silently inferred from this hook.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from . import bend_native_hydraulics as native
from . import wall_velocity_bounds
from . import directional_hydraulic_limiter as directional
from . import hydraulic_rank
from .bend_methods_v1 import ROOT,HERE,digest

METHOD=directional.METHOD


def native_source(code):
    code=hydraulic_rank.native_source(code)
    needle='   do m=1,nprimitive\n    grad(m,:,c)=grad(m,:,c)*max(0.d0,theta(m))'
    count=code.count(needle)
    if count!=3:raise ValueError(f'Native reconstruction layout changed: {count}')
    code=code.replace(needle,'   theta(bed_primitive_index())=braids_wall_bed_factor(c,v,grad,theta)\n'+needle)
    at=code.index('\nend module')
    return directional.native_source(wall_velocity_bounds.native_source(code[:at]+'\n'+(HERE/'wall_bed_reconstruction.f90.inc').read_text()+'\n'+code[at:]))


def cuda_source(code):
    code=hydraulic_rank.cuda_source(code)
    needle='            for(int m=0;m<4;++m) {gx[m]*=fmax(0.,theta[m]);gy[m]*=fmax(0.,theta[m]);}'
    if code.count(needle)!=1:raise ValueError('CUDA reconstruction layout changed')
    return directional.cuda_source(wall_velocity_bounds.cuda_source(code.replace(needle,(HERE/'wall_bed_reconstruction.cu.inc').read_text()+'\n'+needle)))


def build():
    from tools.build_cutcell import SOURCES
    compiler=shutil.which('gfortran');version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
    parts=[ROOT/name for name in SOURCES]
    bridge=HERE/'bend_native_bridge.f90';tail=HERE/'bend_native_rates_tail.inc'
    probe=ROOT/'braids_support/sediment/struiksma_t2_2d/wall_reconstruction_probe.f90'
    extras=[bridge,tail,probe,Path(__file__),HERE/'wall_bed_reconstruction.f90.inc',HERE/'wall_velocity_bounds.py',HERE/'hydraulic_rank.py']
    extras += [HERE/('directional_hydraulic_limiter'+suffix) for suffix in ('.py','.f90.inc','.cu.inc')]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in parts+extras}
    key=hashlib.sha256(json.dumps([METHOD,hashes,version],sort_keys=True).encode()).hexdigest()[:20]
    folder=ROOT/'build/wall_bed_hydraulics'/key;folder.mkdir(parents=True,exist_ok=True)
    library=folder/('hydraulics.dll' if os.name=='nt' else 'hydraulics.so')
    if not library.exists():
        code=native_source(parts[-1].read_text())
        begin=code.index(' subroutine euler(');stop=code.index('  ! Conservative water draining changes',begin)
        prefix=code[begin:stop].replace('subroutine euler(','subroutine bend_native_spatial_rates(',1)
        end=code.index('\nend module',stop)
        code=code[:end]+'\n'+prefix+tail.read_text()+'\n'+bridge.read_text()+'\n'+probe.read_text()+'\nend module\n'
        source=folder/'cutcell_wall_bed.f90';source.write_text(code)
        frozen=folder/'source';frozen.mkdir(exist_ok=True)
        for p in parts+extras:shutil.copy2(p,frozen/p.name)
        command=[compiler,'-O3','-fopenmp','-shared','-ffree-line-length-none','-ffixed-line-length-none',
            *[str(frozen/p.name) for p in parts[:-1]],str(source),'-o',str(library)]
        if os.name=='nt':command+=['-Wl,--export-all-symbols']
        result=subprocess.run(command,cwd=folder,capture_output=True,text=True,timeout=180)
        (folder/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(str(folder/'build.log'))
        (folder/'identity.json').write_text(json.dumps(dict(method=METHOD,source_sha256=hashes,compiler=version,
            command=command,library_sha256=digest(library),generated_sha256=digest(source),delft_modified=False),indent=2))
    identity=json.loads((folder/'identity.json').read_text())
    if digest(library)!=identity['library_sha256']:raise ValueError('Wall library checksum differs')
    return library,identity,compiler


class WallNativeHydraulics(native.NativeHydraulics):
    def __init__(self,deck,data,threads=4):
        if data.get('amr_enabled',False):raise ValueError('Wall bend library requires fixed grid')
        # The legacy loader is serialized. The build override is restored even
        # after failure; no caller or historical process changes its defaults.
        with native._lock:
            previous=native.build
            try:
                native.build=build
                super().__init__(deck,data,threads)
            finally:native.build=previous
        self.identity['hydraulic_spatial_operator']=METHOD


def install(solver,owner,backend,deck,data,source_folder=None):
    if backend=='cpu':
        solver.hydraulics=WallNativeHydraulics(deck,data,solver.hydraulics.threads)
        owner.identity['hydraulics']=solver.hydraulics.identity
        return solver.hydraulics.identity
    if backend!='gpu':raise ValueError('Explicit CPU/GPU backend required')
    import cupy as cp
    from braids_support.gpu_fwave.reconstruction import OPTIONS
    if data.get('amr_enabled',False):raise ValueError('Wall bend reconstruction requires fixed grid')
    from braids_core.windows_paths import filesystem_path
    source=filesystem_path(source_folder)/'reconstruction.cu'
    code=cuda_source(source.read_text())
    names=('reconstruction_primitive','reconstruction_cells','reconstruction_faces')
    module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
    solver.reconstruction.module=module
    solver.reconstruction.kernels={name:module.get_function(name) for name in names}
    identity=dict(method=METHOD,hydraulic_gradient=hydraulic_rank.METHOD,hydraulic_rank_sha256=digest(HERE/'hydraulic_rank.py'),base_sha256=digest(source),generated_sha256=hashlib.sha256(code.encode()).hexdigest(),
        wrapper_sha256=digest(__file__),kernel_sha256=digest(HERE/'wall_bed_reconstruction.cu.inc'),
        velocity_bounds_sha256=digest(HERE/'wall_velocity_bounds.py'),
        directional_wrapper_sha256=digest(HERE/'directional_hydraulic_limiter.py'),
        directional_kernel_sha256=digest(HERE/'directional_hydraulic_limiter.cu.inc'))
    owner.identity['wall_bed_reconstruction']=identity
    return identity
