"""Consistent 2D symmetric stress for variable turbine eddy viscosity.

Nash et al. (2015), Eq. 2/9, uses divergence of h nu (grad u + grad u.T).
The previous component Laplacian agrees only under constant viscosity and
incompressibility. The transported k-epsilon production already uses symmetric
strain; this owner gives momentum the matching tensor on true polygon faces.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from types import MethodType

from .bend_methods_v1 import OPTIONS,digest
from .flow2dh_mobile_bed import ROOT
from .river_hydrokinetic_transport import TurbulenceEngine


class FullStressEngine(TurbulenceEngine):
    NAMES=('hkt_full_strain_faces',)
    SOURCE=Path(__file__).with_suffix('.h')
    def __init__(self,backend,threads):
        source=self.SOURCE;self.backend=backend;self.threads=threads
        self.identity=dict(source_sha256=digest(source),backend=backend,threads=threads)
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=source.read_text(),options=OPTIONS,name_expressions=self.NAMES)
            self.fn={name:self.module.get_function(name) for name in self.NAMES}
            self.identity['options']=list(OPTIONS);return
        if backend!='cpu':raise ValueError('Explicit CPU or GPU backend required')
        compiler=shutil.which('g++')
        if not compiler:raise RuntimeError('OpenMP compiler required')
        version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
        flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc',
               '-static-libstdc++','-ffp-contract=off']
        key=hashlib.sha256((version+str(flags)+source.read_text()).encode()).hexdigest()[:20]
        folder=ROOT/'build/hydrokinetic_stress'/key;folder.mkdir(parents=True,exist_ok=True)
        library=folder/'stress.dll'
        if not library.exists():subprocess.run([compiler,*flags,str(source),'-o',str(library)],
            check=True,capture_output=True,timeout=120)
        self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
        self.lib=ctypes.CDLL(str(library));self.fn={name:getattr(self.lib,name) for name in self.NAMES}
        for function in self.fn.values():function.restype=None
        # Inherit the argument conversion/launch path under the expected name.
        self.lib.hkt_threads=self.lib.hkt_stress_threads
        self.identity.update(compiler=version,flags=flags,library_sha256=digest(library))


def install(owner):
    if owner.time or owner.steps or owner.physics._open:
        raise ValueError('Install symmetric turbulent stress before execution')
    transport=owner.hydrokinetic_turbulence;p=owner.physics
    engine=FullStressEngine(owner.backend,p.flow.threads)
    def momentum(self,q,rate):
        g=p.g;p._gradients(q,p.hkt_k,p.hkt_grad)
        engine.launch('hkt_full_strain_faces',p.nf,p.nf,p.n,float(p.parameters.dry),
            q,p.z0,p.hkt_grad,p.hkt_nu,g['owner'],g['neighbour'],g['normal'],
            g['distance'],g['xy'],p.hkt_momentum_face)
        p.hkt_momentum_work.fill(0.)
        p.flow.launch('flow_momentum_gather',p.n,p.n,p.nf,p.hkt_unit,p.hkt_momentum_face,
            g['offset'],g['faces'],g['signs'],g['length'],g['area'],p.hkt_momentum_work)
        rate[1:]+=p.hkt_momentum_work[1:]
    transport.momentum=MethodType(momentum,transport)
    identity=dict(owner.identity['hydrokinetic_turbulence_transport'],
        method='turbulence.nishino_willden_bit_symmetric_kepsilon_v2',
        momentum='divergence of h nu (grad u + grad u.T), nonorthogonal face correction',
        stress_reference='Nash et al. 2015 Eq. 2/9; doi:10.3390/en81212380',
        stress_source_sha256=digest(__file__),stress_runtime=engine.identity)
    owner.identity['hydrokinetic_turbulence_transport']=identity
    p.identity['hydrokinetic_turbulence_transport']=identity
    return owner
