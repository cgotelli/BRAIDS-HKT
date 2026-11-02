"""Isolated face-derived FLOW effective sediment shear, paired OpenMP/CUDA."""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .bend_methods_v1 import OPTIONS,digest
from .flow2dh_mobile_bed import MobileEngine,ROOT
from .flow2dh_reconstructed_bed import Flow2DHReconstructedBed


class EffectiveShearEngine(MobileEngine):
    def __init__(self,backend,threads):
        self.backend=backend;self.threads=threads
        path=Path(__file__).with_name('flow2dh_effective_shear.h');names=('effective_face_depth','effective_cell_shear','effective_wet_gradient','effective_wet_weights')
        self.identity=dict(backend=backend,threads=threads,source_sha256=digest(path))
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=path.read_text(),options=OPTIONS,name_expressions=names)
            self.fn={k:self.module.get_function(k) for k in names}
        elif backend=='cpu':
            compiler=shutil.which('g++')
            if not compiler:raise RuntimeError('OpenMP compiler required')
            version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
            flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off']
            key=hashlib.sha256((version+str(flags)+path.read_text()).encode()).hexdigest()[:20]
            folder=ROOT/'build/flow2dh_effective_shear'/key;folder.mkdir(parents=True,exist_ok=True);lib=folder/'effective.dll'
            if not lib.exists():subprocess.run([compiler,*flags,str(path),'-o',str(lib)],check=True,capture_output=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
            self.lib=ctypes.CDLL(str(lib));self.fn={k:getattr(self.lib,k) for k in names}
            for f in self.fn.values():f.restype=None
            self.identity.update(compiler=version,flags=flags,library_sha256=digest(lib))
        else:raise ValueError('Explicit CPU or GPU required')


class Flow2DHEffectiveBed(Flow2DHReconstructedBed):
    def __init__(self,*args,correction='effective_shear',**kwargs):
        if correction!='effective_shear':raise ValueError('Unknown effective shear correction')
        super().__init__(*args,**kwargs)
        self.effective=EffectiveShearEngine(self.engine.backend,self.flow.threads)
        xp=self.xp;g=self.mesh
        self.effective_weights=self.engine.array(np.ascontiguousarray(g['length'][None,:]*g['distance']))
        self.effective_face=xp.zeros(self.nf);self.effective_fields=xp.zeros((4,self.n))
        self.effective_q=xp.zeros((3,self.n));self.effective_eta=xp.zeros(self.n)
        self.effective_gradient=xp.zeros((10,self.n))
        self.identity=dict(self.identity,method='classic_flow2dh_eh_mobile_effective_shear_v1',
            effective_runtime=self.effective.identity,effective_wrapper_sha256=digest(__file__),
            effective_mapping='MOR/MIN face depth; affine surface at true midpoint; face-pyramid length*distance weights; closed walls excluded; open owner depth',
            effective_scope='sediment capacity, Shields and spiral direction only; hydraulic Chezy unchanged',
            characteristic_height='dwnvel: actual_depth*float32(0.368); 1cm floor only in erosed roughness working depth',
            historical_parent='classic_flow2dh_eh_mobile_capacity_muscl_v1',public_support=False)

    def effective_inputs(self,q,z):
        g=self.g;e=self.engine;n=self.n;nf=self.nf
        self.effective_eta[:]=q[0]*(q[0]+z)
        e.launch('gradients',n,n,q,z,self.effective_eta,self.M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.effective_gradient)
        self.effective.launch('effective_face_depth',nf,nf,n,q,z,self.effective_gradient,g['owner'],g['neighbour'],g['bid'],g['distance'],g['displacement'],self.effective_face)
        self.effective.launch('effective_cell_shear',n,n,nf,self.chezy,q,self.effective_face,self.effective_weights,
            g['offset'],g['faces'],g['owner'],g['neighbour'],g['bid'],self.effective_fields,self.effective_q)
        if not bool(self.xp.all(self.xp.isfinite(self.effective_fields))):raise ValueError('Invalid effective shear')

    def bed_faces(self,q,z,hi):
        g=self.g;e=self.engine;n=self.n;nf=self.nf
        # This batch exists only inside one class loop with identical q,z,hi,M.
        # Capacity reconstruction has its own buffer and cannot overwrite grad.
        batch=getattr(self,'_river_hydraulic_batch',None)
        if batch is None or not batch.get('gradients',False):
            e.launch('gradients',n,n,q,z,hi,self.M,g['offset'],g['faces'],g['owner'],g['neighbour'],g['dxdy'],g['inverse'],self.grad)
            if batch is not None:batch['gradients']=True
        self.effective_inputs(q,z)
        self.bedcell[:]=self.eh.evaluate(q,hi,capacity_q=self.effective_q,chezy=self.effective_fields[0])
        self.mobile.launch('flow_bed_face_inputs',nf,nf,n,self.parameters.diameter,q,z,self.bedcell,self.grad,
            g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],self.bedface)
        self.reconstruct_capacity()
        self.bedcorrected[:]=self.eh.slope(self.bedface)
        self.mobile.launch('flow_bed_flux',nf,nf,self.eh.solid_density,self.bedcorrected,g['neighbour'],self.bedbc,g['normal'],self.feed,self.aux)
