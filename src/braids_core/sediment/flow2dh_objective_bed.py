"""Explicit research variant: local-transport-frame capacity limiter only."""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from .bend_methods_v1 import OPTIONS,digest
from .flow2dh_mobile_bed import MobileEngine,ROOT
from .flow2dh_reconstructed_bed import Flow2DHReconstructedBed


class ObjectiveEngine(MobileEngine):
    def __init__(self,backend,threads,*,gradient_kernel='flow2dh_objective_capacity.h'):
        self.backend=backend;self.threads=threads
        if gradient_kernel not in ('flow2dh_objective_capacity.h','flow2dh_component_capacity.h'):
            raise ValueError('Unknown objective capacity kernel')
        paths=[Path(__file__).with_name(name) for name in
            ('flow2dh_reconstruction_kernels.h',gradient_kernel)]
        code='\n'.join(p.read_text() for p in paths)
        names=('capacity_objective_gradients','capacity_reconstructed_faces')
        self.identity=dict(backend=backend,threads=threads,source_sha256={p.name:digest(p) for p in paths})
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=code,options=OPTIONS,name_expressions=names)
            self.fn={k:self.module.get_function(k) for k in names}
        elif backend=='cpu':
            compiler=shutil.which('g++')
            if not compiler:raise RuntimeError('OpenMP compiler required')
            version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
            flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off']
            key=hashlib.sha256((version+str(flags)+code).encode()).hexdigest()[:20]
            folder=ROOT/'build/flow2dh_objective'/key;folder.mkdir(parents=True,exist_ok=True);lib=folder/'objective.dll'
            if not lib.exists():subprocess.run([compiler,*flags,'-','-o',str(lib)],input=code,text=True,check=True,capture_output=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
            self.lib=ctypes.CDLL(str(lib));self.fn={k:getattr(self.lib,k) for k in names}
            for f in self.fn.values():f.restype=None
            self.identity.update(compiler=version,flags=flags,library_sha256=digest(lib))
        else:raise ValueError('Explicit CPU or GPU required')
        self.fn['capacity_gradients']=self.fn['capacity_objective_gradients']


class Flow2DHObjectiveBed(Flow2DHReconstructedBed):
    def __init__(self,*args,correction='objective',**kwargs):
        if correction!='objective':raise ValueError('Unknown objective correction')
        super().__init__(*args,**kwargs)
        self.reconstruction=ObjectiveEngine(self.engine.backend,self.flow.threads)
        self.identity=dict(self.identity,method='classic_flow2dh_eh_mobile_objective_capacity_v1',
            objective_runtime=self.reconstruction.identity,objective_wrapper_sha256=digest(__file__),
            bed_face='same limited LS and sign-upwind normal; common limiter in local transport frame; zero vector uses zero reconstruction',
            historical_parent='classic_flow2dh_eh_mobile_capacity_muscl_v1',public_support=False)
