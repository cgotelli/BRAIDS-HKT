"""Conservative wet-bank viscosity component; explicit qualification only.

This component does not change the accepted fully-wet bend or QGIS defaults.
The shared depth bounds exchange by both columns and their common aperture.
On orthogonal meshes internal exchange conserves momentum and dissipates
kinetic energy. Nonorthogonal gradients retain the existing geometric rule;
their nonlinear shoreline behavior requires separate qualification.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from types import MethodType
from .bend_methods_v1 import digest,OPTIONS
from .flow2dh_mobile_bed import MobileEngine,ROOT


class ViscosityEngine(MobileEngine):
    def __init__(self,backend,threads):
        self.backend=backend;self.threads=threads
        source=Path(__file__).with_suffix('.h');name='shoreline_viscous_faces'
        self.identity=dict(source_sha256=digest(source),backend=backend,threads=threads)
        if backend=='gpu':
            import cupy as cp
            self.module=cp.RawModule(code=source.read_text(),options=OPTIONS,name_expressions=(name,))
            self.fn={name:self.module.get_function(name)}
            self.identity['options']=list(OPTIONS)
        elif backend=='cpu':
            compiler=shutil.which('g++')
            if not compiler:raise RuntimeError('OpenMP compiler required')
            version=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]
            flags=['-x','c++','-std=c++11','-O3','-fopenmp','-shared','-static-libgcc','-static-libstdc++','-ffp-contract=off']
            key=hashlib.sha256((version+str(flags)+source.read_text()).encode()).hexdigest()[:20]
            folder=ROOT/'build/shoreline_viscosity'/key;folder.mkdir(parents=True,exist_ok=True)
            lib=folder/'viscosity.dll'
            if not lib.exists():subprocess.run([compiler,*flags,str(source),'-o',str(lib)],check=True,capture_output=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(compiler).parent))] if os.name=='nt' else []
            self.lib=ctypes.CDLL(str(lib));self.fn={name:getattr(self.lib,name)}
            self.fn[name].restype=None
            self.identity.update(compiler=version,flags=flags,library_sha256=digest(lib))
        else:raise ValueError('Explicit CPU/GPU choice required')


def install(solver):
    """One explicit owner, installed before execution, restart-locked.

    No threshold-based backend selection and no alternate physical viscosity
    coefficient. The old viscous term is disabled exactly once; its secondary
    stress and geometric source remain unchanged.
    """
    p=solver.physics;xp=p.xp
    if p.time or p.steps or p._open or hasattr(p,'viscous_engine'):
        raise ValueError('Install one viscous owner before execution')
    p.viscous_engine=ViscosityEngine(solver.backend,p.flow.threads)
    nu=float(p.nu);p.nu=0.
    p.viscous_face=xp.zeros((2,p.nf));p.viscous_unit=xp.ones(p.n)
    p.viscous_work=xp.zeros((3,p.n));old=p.add_momentum
    def momentum(owner,q,water,rate):
        old(q,water,rate)
        if not nu:return
        safe,wet=owner.safe(q) if hasattr(owner,'safe') else (q,q[0]>owner.parameters.dry)
        owner._gradients(safe,owner.W,owner.geometry_grad)
        g=owner.g
        owner.viscous_engine.launch('shoreline_viscous_faces',owner.nf,owner.nf,owner.n,
            nu,float(owner.parameters.dry),q,owner.z0,owner.geometry_grad,g['owner'],g['neighbour'],
            g['normal'],g['distance'],g['xy'],owner.viscous_face)
        owner.viscous_work.fill(0.)
        owner.flow.launch('flow_momentum_gather',owner.n,owner.n,owner.nf,owner.viscous_unit,
            owner.viscous_face,g['offset'],g['faces'],g['signs'],g['length'],g['area'],owner.viscous_work)
        rate[1:]+=owner.viscous_work[1:]
    p.add_momentum=MethodType(momentum,p)
    p.identity['conservative_viscosity']=dict(wrapper_sha256=digest(__file__),runtime=p.viscous_engine.identity,
        coefficient_m2_s=nu,face_depth='min(columns,nonnegative_mean_stage_above_higher_bed)',
        scope='fixed_grid_explicit_shoreline_candidate',public_support=False)
    return solver
