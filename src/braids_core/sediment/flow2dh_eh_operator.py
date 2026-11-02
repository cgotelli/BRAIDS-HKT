"""Native/CUDA composition of existing EH and classic FLOW spiral operators.

Fully wet, one uniform grain, density 2650/1000, g=9.81, ACal=Espir=1.
Returns mass transport kg/(m s); not by itself a coupled solver or public profile.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from .bend_methods_v1 import digest

ROOT=Path(__file__).resolve().parents[2]


class ClassicFlowEH:
    def __init__(self,backend,*,diameter,chezy,threads=4):
        if backend not in ('cpu','gpu'):raise ValueError('Choose CPU or GPU explicitly')
        if not np.isfinite([diameter,chezy]).all() or diameter<=0 or chezy<1.e-6:
            raise ValueError('Invalid EH coefficients')
        if not isinstance(threads,int) or threads<1:raise ValueError('Positive thread count required')
        self.backend=backend;self.d=float(diameter);self.c=float(chezy);self.threads=threads
        self.solid_density=2650.
        self.identity=dict(backend=backend,diameter_m=self.d,chezy=self.c,threads=threads,
            source_sha256={},scope='fully_wet_single_grain_shared_input_operator_only',
            output='raw mass x/y, absolute Shields, rotated mass x/y, spiral alpha')
        if backend=='cpu':
            self.xp=np;fc=shutil.which('gfortran')
            if not fc:raise RuntimeError('Native Fortran compiler required')
            sources=[ROOT/'solver/wave'/f for f in ('braids_bedload_current_module.f90',
                'braids_flow2dh_secondary_operator_module.f90','braids_flow_bed_slope_operator_module.f90',
                'braids_flow_eh_c_api.f90')]
            self.identity['source_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sources}
            version=subprocess.check_output([fc,'--version'],text=True).splitlines()[0]
            flags=['-O3','-fopenmp','-shared','-ffp-contract=off','-ffree-line-length-none']
            key=hashlib.sha256((str(self.identity['source_sha256'])+version+str(flags)).encode()).hexdigest()[:20]
            folder=ROOT/'build/flow2dh_eh'/key;folder.mkdir(parents=True,exist_ok=True)
            lib=folder/'flow_eh.dll'
            if not lib.exists():subprocess.run([fc,*flags,*map(str,sources),'-J',str(folder),'-o',str(lib)],
                cwd=folder,capture_output=True,check=True,timeout=120)
            self.dirs=[os.add_dll_directory(str(Path(fc).parent))] if os.name=='nt' else []
            try:
                self.lib=ctypes.CDLL(str(lib));self.lib.braids_flow_eh_cells.restype=None
                self.lib.braids_flow_eh_slope.restype=None
            finally:
                # Mixed transport bundles can replace their capacity engines.
                # Keep the loaded library, not an uncollectable search cookie.
                for directory in self.dirs:directory.close()
                self.dirs=[]
            self.identity.update(compiler=version,flags=flags,library_sha256=digest(lib))
        else:
            import cupy as cp
            from braids_gpu.flow2dh_secondary_operator import Flow2DHSecondaryCUDA
            from braids_gpu.delft_current_capacity import DelftCurrentCapacityCUDA
            from braids_gpu.flow_bed_slope_runtime import FlowBedSlopeCUDA
            self.xp=cp;self.spiral=Flow2DHSecondaryCUDA();self.capacity=DelftCurrentCapacityCUDA()
            self.slope_operator=FlowBedSlopeCUDA()
            sources=[ROOT/'braids_gpu'/f for f in ('delft_current_capacity.cu','flow2dh_secondary_operator.cu','flow_bed_slope_runtime.cu')]
            self.identity['source_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sources}
            self.identity['device']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            self.identity['capacity']='existing general CUDA specialization: A=.05*C^2/g, PowerB=0, PowerC=2.5, ThetaC=0, SusACal=0; algebraically EH'

    def evaluate(self,q,hi,*,capacity_q=None,chezy=None):
        """Allow FLOW's effective capacity velocity separately from steering U.

        Omitted arguments use q and the constructor C. This distinction is
        required for a shared-input erosed/eqtran replay, not a fitted correction.
        """
        xp=self.xp
        if not isinstance(q,xp.ndarray) or q.dtype!=xp.float64 or q.ndim!=2 or q.shape[0]!=3 or not q.flags.c_contiguous:
            raise ValueError('Contiguous FP64 q=(h,hu,hv) on selected backend required')
        n=q.shape[1]
        if not isinstance(hi,xp.ndarray) or hi.shape!=(n,) or hi.dtype!=xp.float64 or not hi.flags.c_contiguous:
            raise ValueError('Contiguous FP64 signed hI required')
        if not bool(xp.all(xp.isfinite(q))) or not bool(xp.all(xp.isfinite(hi))) or bool(xp.any(q[0]<=.001)):
            raise ValueError('EH bridge requires finite fully wet states')
        cq=q if capacity_q is None else capacity_q
        c=xp.full(n,self.c) if chezy is None else chezy
        if (not isinstance(cq,xp.ndarray) or cq.dtype!=xp.float64 or cq.shape!=q.shape or not cq.flags.c_contiguous
            or not isinstance(c,xp.ndarray) or c.dtype!=xp.float64 or c.shape!=(n,) or not c.flags.c_contiguous):
            raise ValueError('Resident contiguous capacity state and Chezy required')
        if not bool(xp.all(xp.isfinite(cq))) or not bool(xp.all(xp.isfinite(c))) or bool(xp.any(cq[0]<=.001)) or bool(xp.any(c<20)):
            raise ValueError('Invalid effective transport forcing')
        out=xp.zeros((6,n));status=xp.zeros(n,dtype=xp.int32)
        if self.backend=='cpu':
            ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
            self.lib.braids_flow_eh_cells(ctypes.c_int(n),ctypes.c_int(self.threads),ctypes.c_double(self.d),
                ptr(c),ptr(q),ptr(cq),ptr(hi),ptr(out),ptr(status))
        else:
            if not n:return out
            full=lambda v:xp.full(n,v,dtype=xp.float64)
            result=self.capacity.evaluate(4,u=cq[1]/cq[0],v=cq[2]/cq[0],d50=full(self.d),
                solid_density=full(2650.),water_density=full(1000.),gravity=full(9.81),chezy=c,
                hiding_factor=full(1.),bed_calibration=.05*c*c/9.81,bed_shields_power=full(0.),
                bed_excess_power=full(2.5),bed_ripple_factor=full(1.),bed_critical_shields=full(0.),
                sus_calibration=full(0.),sus_shields_power=full(3.),sus_excess_power=full(0.),
                sus_ripple_factor=full(1.),sus_critical_shields=full(0.))
            out[:2]=result.bed_mass;out[2]=result.shields
            u=q[1]/q[0];v=q[2]/q[0]
            r=self.spiral.evaluate(xp.ones(n,dtype=xp.uint8),q[0],q[0],u,v,full(0.),full(1.),full(1.),
                full(0.),c,hi/q[0],out[0],out[1],dryflc=1e-6,rmincf=2.5,chzmin=20.,
                gravity=9.81,von_karman=.41,espir=1.,betac=0.,epsilon_value=np.finfo(float).eps)
            out[3]=r.rotated_x;out[4]=r.rotated_y;out[5]=r.spiral_alpha
            if bool(xp.any(r.status)):raise ValueError('CUDA FLOW secondary operator rejected state')
        if bool(xp.any(status)) or not bool(xp.all(xp.isfinite(out))):raise ValueError('EH operator rejected state')
        return out

    def slope(self,fields):
        """T2 Koch--Flokstra controls, mass units, positive DOWNHILL gradients."""
        xp=self.xp
        if not isinstance(fields,xp.ndarray) or fields.dtype!=xp.float64 or fields.ndim!=2 or fields.shape[0]!=10 or not fields.flags.c_contiguous:
            raise ValueError('Expected ten contiguous resident FP64 face-input rows')
        n=fields.shape[1]
        if self.backend=='gpu':
            result=self.slope_operator.evaluate(*list(fields),mode=3,alfabs=0.,alfabn=0.,ashld=.85,
                bshld=.5,cshld=0.,dshld=0.,alfpa=1.,thcrpa=.05)
            return xp.stack([result.transport_x,result.transport_y])
        out=xp.zeros((2,n));status=xp.zeros(n,dtype=xp.int32)
        ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
        self.lib.braids_flow_eh_slope(ctypes.c_int(n),ctypes.c_int(self.threads),ptr(fields),ptr(out),ptr(status))
        if np.any(status):raise ValueError('Koch--Flokstra rejected face input')
        return out
