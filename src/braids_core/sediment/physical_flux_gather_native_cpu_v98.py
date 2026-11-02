"""Native candidate for physical replacement/draining/gather; no source owner."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .density_active_projection_v97 import _array,_digest,_hash,_sha
from .density_trace_projection_v98 import _hex
from .densin_fwave_v85 import verify_locked_sources_v85

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"solver/wave/braids_physical_flux_gather_v98_module.f90"
KEYS=("raw_left_face_flux","raw_right_face_flux","limited_left_face_flux","limited_right_face_flux",
      "draining_factor","base_rate","accepted_limited_face_water_flux_m2_s")


class NativePhysicalFluxGatherRuntimeV98:
    def __init__(self,receipt,library,dependency,path):
        self.receipt,self.library,self.dependency,self.path=receipt,library,dependency,Path(path)
        self.function=library.braids_physical_flux_gather_v98
        fa=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        ia=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self.function.argtypes=[ctypes.c_int]*4+[ctypes.c_double]*3+[fa,ia]+[ia]*5+[fa]*6+[ia]+[fa]*7+[
            ia,ctypes.POINTER(ctypes.c_int)]
        self.function.restype=None

    def evaluate(self,q,wet_mask,raw_left,raw_right,physical_flux,physical_mask,*,owner,neighbour,
        cell_face_offset,cell_faces,cell_face_sign,face_length_m,cell_area_m2,normal,
        gravity_m_s2,dry_tolerance_m,dt_s,physical_trace_receipt_sha256,reflected_result_sha256,
        native_threads=4,cancelled=False):
        cancel=cancelled if callable(cancelled) else lambda:cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v98 physical gather cancellation must be bool/callable")
        if cancel():raise InterruptedError("v98 physical gather cancelled before launch")
        if (type(native_threads) is not int or not 1<=native_threads<=256 or any(
            isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v)
            for v in (gravity_m_s2,dry_tolerance_m,dt_s)) or gravity_m_s2<=0 or dry_tolerance_m<0 or dt_s<=0):
            raise ValueError("v98 physical gather controls differ")
        if (self.receipt["receipt_sha256"]!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=self.receipt["source_sha256"] or _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(self.path)!=self.receipt["library_sha256"]):
            raise ValueError("v98 physical gather runtime/source identity differs")
        provenance=dict(physical_trace_receipt_sha256=_hex(physical_trace_receipt_sha256,"physical trace"),
            reflected_result_sha256=_hex(reflected_result_sha256,"reflected result"))
        qr=np.asarray(q);lr=np.asarray(raw_left)
        if qr.ndim!=2 or qr.shape[0]!=3 or lr.ndim!=2 or lr.shape[0]!=3 or qr.shape[1]<1 or lr.shape[1]<1:
            raise ValueError("v98 physical gather state/face shapes differ")
        nc,nf=qr.shape[1],lr.shape[1]
        q=_array(q,(3,nc),"q");wet=_array(wet_mask,(nc,),"wet mask",integer=True)
        left=_array(raw_left,(3,nf),"raw left");right=_array(raw_right,(3,nf),"raw right")
        physical=_array(physical_flux,(3,nf),"physical flux");mask=_array(physical_mask,(nf,),"physical mask",integer=True)
        own=_array(owner,(nf,),"owner",integer=True);nei=_array(neighbour,(nf,),"neighbour",integer=True)
        off=_array(cell_face_offset,(nc+1,),"offset",integer=True)
        nadj=int(off[-1]);adj=_array(cell_faces,(nadj,),"faces",integer=True)
        signs=_array(cell_face_sign,(nadj,),"signs",integer=True)
        length=_array(face_length_m,(nf,),"length");area=_array(cell_area_m2,(nc,),"area")
        n=_array(normal,(2,nf),"normal")
        if (np.any(q[0]<0) or np.any((wet!=0)&(wet!=1)) or np.any((mask!=0)&(mask!=1)) or
            np.any(own<0) or np.any(own>=nc) or np.any(nei<-1) or np.any(nei>=nc) or
            off[0]!=0 or np.any(np.diff(off)<0) or nadj<1 or np.any(adj<0) or np.any(adj>=nf) or
            np.any((signs!=-1)&(signs!=1)) or np.any(length<=0) or np.any(area<=0) or
            np.any((mask!=0)&(nei>=0))):raise ValueError("v98 physical gather topology/domain differs")
        # Authenticate the exact incidence used for each parallel cell reduction.
        cells=np.repeat(np.arange(nc),np.diff(off));pair=np.stack((cells,adj,signs),axis=1)
        internal=np.flatnonzero(nei>=0)
        expected=np.concatenate((np.column_stack((own,np.arange(nf),np.ones(nf,dtype=int))),
            np.column_stack((nei[internal],internal,-np.ones(internal.size,dtype=int)))),axis=0)
        order=lambda a:a[np.lexsort((a[:,2],a[:,1],a[:,0]))]
        if not np.array_equal(order(pair),order(expected)):
            raise ValueError("v98 physical gather CSR incidence differs")
        out={key:np.empty(shape) for key,shape in zip(KEYS,((3,nf),)*4+((nc,),(3,nc),(nf,)))}
        counts=np.empty(2,np.int32);status=ctypes.c_int(-1)
        inputs=(q,wet,own,nei,off,adj,signs,length,area,n,left,right,physical,mask)
        self.function(nc,nf,nadj,native_threads,float(gravity_m_s2),float(dry_tolerance_m),float(dt_s),
            *inputs,*(out[key] for key in KEYS),counts,ctypes.byref(status))
        if status.value or any(not np.isfinite(a).all() for a in out.values()):
            raise ValueError("v98 physical gather candidate is outside finite/shared-water domain")
        if cancel():raise InterruptedError("v98 physical gather cancelled after launch")
        for a in out.values():a.setflags(write=False)
        metadata=dict(runtime_sha256=self.receipt["receipt_sha256"],native_threads=native_threads,
            input_sha256=_digest(*inputs),output_sha256=_digest(*(out[key] for key in KEYS)),**provenance,
            physical_flux_consumption_count=int(counts[0]),limited_face_count=int(counts[1]),
            draining_limiter_owner_count=1,pressure_gather_owner_count=1,
            riemann_evaluation_count=0,turbulence_evaluation_count=0,density_evaluation_count=0,
            numerical_python_cell_face_loops=0,state_or_clock_publication_count=0,production_admission=False)
        return dict(**out,metadata=metadata,receipt_sha256=_hash(metadata))


def build_native_physical_flux_gather_v98(build_directory,*,compiler=None):
    chosen=compiler or shutil.which("gfortran")
    if chosen is None and os.name=="nt":chosen=Path(r"C:\Strawberry\c\bin\gfortran.exe")
    if chosen is None or not Path(chosen).is_file():raise ValueError("v98 physical gather requires gfortran")
    chosen=Path(chosen).resolve();root=Path(build_directory).resolve();root.mkdir(parents=True,exist_ok=True)
    build=Path(tempfile.mkdtemp(prefix="physical_flux_gather_v98_",dir=root))
    path=build/("physical_flux_gather_v98.dll" if os.name=="nt" else "libphysical_flux_gather_v98.so")
    partial=path.with_suffix(path.suffix+".partial")
    flags=["-O3","-fopenmp","-shared","-ffree-line-length-none","-static-libgfortran","-static-libgcc"]
    source_sha=_sha(SOURCE)
    proc=subprocess.run([str(chosen),*flags,str(SOURCE),"-o",str(partial)],cwd=build,capture_output=True,text=True,timeout=120)
    if proc.returncode or not partial.is_file():raise ValueError("v98 native physical gather build failed: "+proc.stdout+proc.stderr)
    if _sha(SOURCE)!=source_sha:raise ValueError("v98 physical gather source changed during build")
    os.replace(partial,path)
    receipt=dict(source_sha256=source_sha,wrapper_sha256=_sha(Path(__file__)),library_sha256=_sha(path),
        compiler_sha256=_sha(chosen),compile_flags=flags,locked_source_sha256=verify_locked_sources_v85(ROOT),
        thread_policy="per_cell_per_face_OMP_no_nested_team",numerical_python_cell_face_loops=0,production_admission=False)
    receipt["receipt_sha256"]=_hash(receipt)
    dependency=os.add_dll_directory(str(chosen.parent)) if os.name=="nt" and hasattr(os,"add_dll_directory") else None
    return NativePhysicalFluxGatherRuntimeV98(receipt,ctypes.CDLL(str(path)),dependency,path)
