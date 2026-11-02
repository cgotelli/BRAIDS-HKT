"""Explicit research opt-in: reconstructed capacity, unchanged FLOW physics.

The old owner/source hashes are retained for strict historical restarts. A run
must explicitly change owner identity to use this scheme; no plugin promotion.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from .bend_methods_v1 import digest, OPTIONS
from .flow2dh_mobile_bed import Flow2DHMobileBed, MobileEngine, ROOT

NAMES=('capacity_gradients','capacity_reconstructed_faces')


class ReconstructionEngine(MobileEngine):
    def __init__(self,backend,threads):
        self.backend=backend;self.threads=threads
        path=Path(__file__).with_name('flow2dh_reconstruction_kernels.h')
        self.identity=dict(source_sha256=digest(path),backend=backend,threads=threads)
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=path.read_text(),options=OPTIONS,name_expressions=NAMES)
            self.fn={k:self.module.get_function(k) for k in NAMES}
        elif backend=='cpu':
            compiler=shutil.which('g++')
            if not compiler:raise RuntimeError('OpenMP compiler required')
            version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
            flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off']
            key=hashlib.sha256((version+str(flags)+path.read_text()).encode()).hexdigest()[:20]
            folder=ROOT/'build/flow2dh_reconstruction'/key;folder.mkdir(parents=True,exist_ok=True)
            lib=folder/'capacity.dll'
            if not lib.exists():subprocess.run([compiler,*flags,str(path),'-o',str(lib)],capture_output=True,check=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
            try:
                self.lib=ctypes.CDLL(str(lib));self.fn={k:getattr(self.lib,k) for k in NAMES}
                for f in self.fn.values():f.restype=None
            finally:
                # Dependencies are eagerly loaded; subclasses can replace this
                # engine, so a search-directory cookie must not outlive loading.
                for directory in self.dirs:directory.close()
                self.dirs=[]
            self.identity.update(compiler=version,flags=flags,library_sha256=digest(lib))
        else:raise ValueError('Explicit cpu or gpu required')


class Flow2DHReconstructedBed(Flow2DHMobileBed):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reconstruction=ReconstructionEngine(self.engine.backend,self.flow.threads)
        self.capacity_gradient=self.xp.zeros((4,self.n))
        self.capacity_limiter=self.xp.ones(self.n)
        self.identity=dict(self.identity,schema=3,method='classic_flow2dh_eh_mobile_capacity_muscl_v1',
            reconstruction_runtime=self.reconstruction.identity,reconstruction_wrapper_sha256=digest(__file__),
            bed_face='limited_LS_capacity_to_actual_face; common_vector_Barth_Jespersen; sign_upwind_normal; reconstructed_tangent; unchanged_secant_bed_slope',
            historical_parent='classic_flow2dh_eh_mobile_v2',public_support=False)

    def reconstruct_capacity(self):
        r=self.reconstruction;g=self.g;n=self.n;nf=self.nf
        r.launch('capacity_gradients',n,n,nf,self.bedcell,g['offset'],g['faces'],g['owner'],g['neighbour'],
            g['dxdy'],g['inverse'],g['displacement'],self.capacity_gradient,self.capacity_limiter)
        r.launch('capacity_reconstructed_faces',nf,nf,n,self.bedcell,self.capacity_gradient,
            g['owner'],g['neighbour'],g['normal'],g['distance'],g['displacement'],self.bedface)

    def bed_faces(self,q,z,hi):
        g=self.g;e=self.engine;n=self.n;nf=self.nf
        e.launch('gradients',n,n,q,z,hi,self.M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.grad)
        self.bedcell[:]=self.eh.evaluate(q,hi)
        self.mobile.launch('flow_bed_face_inputs',nf,nf,n,self.parameters.diameter,q,z,self.bedcell,self.grad,
            g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],self.bedface)
        self.reconstruct_capacity()
        self.bedcorrected[:]=self.eh.slope(self.bedface)
        self.mobile.launch('flow_bed_flux',nf,nf,2650.,self.bedcorrected,g['neighbour'],self.bedbc,g['normal'],self.feed,self.aux)
