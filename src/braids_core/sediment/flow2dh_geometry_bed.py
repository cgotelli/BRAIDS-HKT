"""Paired coupled geometry correction with explicit, restart-locked selection.

This is an additive research owner. Accepted reconstructed references and QGIS
defaults remain unchanged. Constitutive physics and augmented f-wave are inherited.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

from .bend_methods_v1 import OPTIONS,digest
from .flow2dh_mobile_bed import MobileEngine,ROOT
from .flow2dh_reconstructed_bed import Flow2DHReconstructedBed

CORRECTIONS={'stress':(True,False,False),'viscosity':(False,True,False),
             'diffusion':(False,False,True),'all':(True,True,True)}


class GeometryEngine(MobileEngine):
    def __init__(self,backend,threads):
        self.backend=backend;self.threads=threads
        source=Path(__file__).with_name('flow2dh_geometry_kernels.h')
        names=('geometry_scalar_correction','geometry_momentum_faces')
        self.identity=dict(source_sha256=digest(source),backend=backend,threads=threads)
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=source.read_text(),options=OPTIONS,name_expressions=names)
            self.fn={k:self.module.get_function(k) for k in names}
        elif backend=='cpu':
            compiler=shutil.which('g++')
            if not compiler:raise RuntimeError('OpenMP compiler required')
            version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
            flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off']
            key=hashlib.sha256((version+str(flags)+source.read_text()).encode()).hexdigest()[:20]
            folder=ROOT/'build/flow2dh_geometry'/key;folder.mkdir(parents=True,exist_ok=True);lib=folder/'geometry.dll'
            if not lib.exists():subprocess.run([compiler,*flags,str(source),'-o',str(lib)],check=True,capture_output=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
            self.lib=ctypes.CDLL(str(lib));self.fn={k:getattr(self.lib,k) for k in names}
            for fn in self.fn.values():fn.restype=None
            self.identity.update(compiler=version,flags=flags,library_sha256=digest(lib))
        else:raise ValueError('Explicit CPU or GPU selection required')


class Flow2DHGeometryBed(Flow2DHReconstructedBed):
    def __init__(self,*args,correction='all',**kwargs):
        if correction not in CORRECTIONS:raise ValueError('Unknown auxiliary geometry correction')
        super().__init__(*args,**kwargs)
        self.correction=correction
        self.correct_stress,self.correct_viscosity,self.correct_diffusion=CORRECTIONS[correction]
        self.geometry=GeometryEngine(self.engine.backend,self.flow.threads)
        xp=self.xp
        self.geometry_grad=xp.zeros((10,self.n))
        self.geometry_Bgrad=xp.zeros((10,self.n))
        self.geometry_Bstate=xp.zeros(self.n)
        self.identity=dict(self.identity,method='classic_flow2dh_eh_mobile_geometry_v1',
            geometry_correction=correction,geometry_runtime=self.geometry.identity,
            geometry_wrapper_sha256=digest(__file__),
            geometry_parent='classic_flow2dh_eh_mobile_capacity_muscl_v1',
            geometry_scope='interior_faces_only; source_physics_boundaries_and_sediment_reconstruction_unchanged',
            public_support=False)

    def _gradients(self,q,scalar,out):
        g=self.g;n=self.n
        self.engine.launch('gradients',n,n,q,self.z0,scalar,self.M,g['offset'],g['faces'],
            g['owner'],g['neighbour'],g['dxdy'],g['inverse'],out)

    def begin(self,q,z,dt,time):
        super().begin(q,z,dt,time)
        if self.correct_stress:
            self.geometry_Bstate[:]=q[0]*self.directional_B
            self._gradients(q,self.geometry_Bstate,self.geometry_Bgrad)

    def add_momentum(self,q,water,rate):
        if not (self.correct_stress or self.correct_viscosity):
            return super().add_momentum(q,water,rate)
        if self.correct_viscosity:self._gradients(q,self.W,self.geometry_grad)
        g=self.g;n=self.n;nf=self.nf
        self.geometry.launch('geometry_momentum_faces',nf,nf,n,int(self.correct_stress),int(self.correct_viscosity),
            self.nu,q,self.directional_B,self.geometry_grad,self.geometry_Bgrad,g['owner'],g['neighbour'],
            g['normal'],g['distance'],g['xy'],g['displacement'],self.momentumflux)
        self.flow.launch('flow_momentum_gather',n,n,nf,self.hold,self.momentumflux,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],rate)
        rate[1:]+=self.hold*self.directional_source

    def scalar_faces(self,q,hi,water):
        super().scalar_faces(q,hi,water)
        if not self.correct_diffusion or self.diff==0.:return
        g=self.g
        self.geometry.launch('geometry_scalar_correction',self.nf,self.nf,self.n,self.diff,q,hi,
            self.scalar_grad,g['owner'],g['neighbour'],g['normal'],g['distance'],g['xy'],self.scalarflux)
