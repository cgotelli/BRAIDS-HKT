"""Pure OpenMP corrected conserved images/EOS; the outer leaf owns publication."""
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
from .transported_fluid_settling_v92 import MODE_CONSTANT,MODE_TRANSPORTED

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"solver/wave/braids_amr_corrected_images_v98_module.f90"
EOS_CODES={"eckart":0,"unesco":1,"nacl":2}
KEYS=("q","suspended","bed","fluid","concentration","temperature","salinity",
      "bed_elevation","eta","wet_mask","dry_transfer","momentum_cleanup","fluid_cleanup",
      "water_density","viscosity","mixture_density")


class NativeAMRCorrectedImagesRuntimeV98:
    def __init__(self,receipt,library,dependency,path,geometry):
        self.receipt,self.library,self.dependency,self.path=receipt,library,dependency,Path(path)
        self.geometry_host=geometry;self.ng,self.nc=geometry[1].shape
        self.function=library.braids_amr_corrected_images_v98
        fa=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
        ia=np.ctypeslib.ndpointer(dtype=np.int32,flags="C_CONTIGUOUS")
        self.function.argtypes=[ctypes.c_int]*4+[ctypes.c_double,ctypes.c_int,ctypes.c_int]+[
            ctypes.c_double]*3+[ctypes.c_int]+[fa]*21+[ia]+[fa]*6+[ia]
        self.function.restype=None

    def evaluate(self,q,suspended,bed,fluid,delta_q,delta_suspended,delta_bed,delta_fluid,*,
        cell,dry_tolerance_m,temperature_mode,salinity_mode,equilibrium_temperature_c,
        equilibrium_salinity_ppt,density_eos,reference_water_density_kg_m3,
        correction_receipt_sha256,source_state_sha256,native_threads=4,cancelled=False):
        cancel=cancelled if callable(cancelled) else lambda:cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v98 native corrected image cancellation must be bool/callable")
        if cancel():raise InterruptedError("v98 native corrected images cancelled before launch")
        if (type(cell) is not int or not 0<=cell<self.nc or type(native_threads) is not int or not 1<=native_threads<=256 or
            temperature_mode not in (MODE_CONSTANT,MODE_TRANSPORTED) or salinity_mode not in (MODE_CONSTANT,MODE_TRANSPORTED) or
            density_eos not in EOS_CODES or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v)
                for v in (dry_tolerance_m,equilibrium_temperature_c,equilibrium_salinity_ppt,reference_water_density_kg_m3)) or
            dry_tolerance_m<0 or equilibrium_temperature_c<=-20 or equilibrium_salinity_ppt<0 or reference_water_density_kg_m3<=0):
            raise ValueError("v98 native corrected image controls differ")
        if (self.receipt["receipt_sha256"]!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=self.receipt["source_sha256"] or _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(self.path)!=self.receipt["library_sha256"] or _digest(*self.geometry_host)!=self.receipt["geometry_sha256"]):
            raise ValueError("v98 native corrected image runtime/source/geometry identity differs")
        provenance=dict(correction_receipt_sha256=_hex(correction_receipt_sha256,"correction receipt"),
            source_state_sha256=_hex(source_state_sha256,"source state"))
        nc,ng=self.nc,self.ng
        shapes=((3,nc),(ng,nc),(ng,nc),(2,nc),(3,),(ng,),(ng,),(2,))
        arrays=tuple(_array(value,shape,"native corrected image input") for value,shape in zip(
            (q,suspended,bed,fluid,delta_q,delta_suspended,delta_bed,delta_fluid),shapes))
        outputs={name:np.empty(shape,dtype=np.int32 if name=="wet_mask" else np.float64) for name,shape in (
            ("q",(3,nc)),("suspended",(ng,nc)),("bed",(ng,nc)),("fluid",(2,nc)),
            ("concentration",(ng,nc)),("temperature",(nc,)),("salinity",(nc,)),("bed_elevation",(nc,)),
            ("eta",(nc,)),("wet_mask",(nc,)),("dry_transfer",(ng,nc)),("momentum_cleanup",(2,nc)),
            ("fluid_cleanup",(2,nc)),("water_density",(nc,)),("viscosity",(nc,)),("mixture_density",(nc,)))}
        status=np.empty(nc,np.int32)
        self.function(nc,ng,cell,native_threads,float(dry_tolerance_m),int(temperature_mode==MODE_TRANSPORTED),
            int(salinity_mode==MODE_TRANSPORTED),float(equilibrium_temperature_c),float(equilibrium_salinity_ppt),
            float(reference_water_density_kg_m3),EOS_CODES[density_eos],*arrays,*self.geometry_host,
            *(outputs[key] for key in KEYS),status)
        if np.any(status):raise ValueError("v98 native corrected image is outside conserved/EOS domain")
        if cancel():raise InterruptedError("v98 native corrected images cancelled after launch")
        for a in outputs.values():a.setflags(write=False)
        metadata=dict(runtime_sha256=self.receipt["receipt_sha256"],backend="native_cpu",native_threads=native_threads,
            input_sha256=_digest(*arrays),output_sha256=_digest(*(outputs[key] for key in KEYS)),**provenance,
            controls_sha256=_hash(dict(cell=cell,dry_tolerance_m=float(dry_tolerance_m),temperature_mode=temperature_mode,
                salinity_mode=salinity_mode,equilibrium_temperature_c=float(equilibrium_temperature_c),
                equilibrium_salinity_ppt=float(equilibrium_salinity_ppt),density_eos=density_eos,
                reference_water_density_kg_m3=float(reference_water_density_kg_m3))),
            reflux_correction_application_count=1,reflux_dry_cleanup_owner_count=1,water_geometry_source_count=0,
            settling_exchange_q3_replay_count=0,hles_evaluation_count=0,dss_recompute_count=0,
            numerical_python_cell_face_loops=0,state_or_clock_publication_count=0,production_admission=False)
        return dict(images=outputs,metadata=metadata,receipt_sha256=_hash(metadata))


def build_native_amr_corrected_images_v98(build_directory,*,initial_bed_elevation_m,reference_bed_mass_kg_m2,
    dry_bulk_density_kg_m3,solid_density_kg_m3,compiler=None):
    raw=np.asarray(reference_bed_mass_kg_m2)
    if raw.ndim!=2 or min(raw.shape)<1:raise ValueError("v98 native bed reference shape differs")
    ng,nc=raw.shape
    geometry=tuple(np.array(_array(value,shape,"native corrected geometry"),copy=True,order="C") for value,shape in (
        (initial_bed_elevation_m,(nc,)),(raw,(ng,nc)),(dry_bulk_density_kg_m3,(ng,)),(solid_density_kg_m3,(ng,))))
    if np.any(geometry[1]<0) or np.any(geometry[2]<=0) or np.any(geometry[3]<=0):
        raise ValueError("v98 native reference/density domain differs")
    for a in geometry:a.setflags(write=False)
    chosen=compiler or shutil.which("gfortran")
    if chosen is None and os.name=="nt":chosen=Path(r"C:\Strawberry\c\bin\gfortran.exe")
    if chosen is None or not Path(chosen).is_file():raise ValueError("v98 corrected images require gfortran")
    chosen=Path(chosen).resolve();root=Path(build_directory).resolve();root.mkdir(parents=True,exist_ok=True)
    build=Path(tempfile.mkdtemp(prefix="amr_corrected_images_v98_",dir=root))
    path=build/("amr_corrected_images_v98.dll" if os.name=="nt" else "libamr_corrected_images_v98.so")
    partial=path.with_suffix(path.suffix+".partial")
    flags=["-O3","-fopenmp","-shared","-ffree-line-length-none","-static-libgfortran","-static-libgcc"]
    source_sha=_sha(SOURCE)
    proc=subprocess.run([str(chosen),*flags,str(SOURCE),"-o",str(partial)],cwd=build,capture_output=True,text=True,timeout=120)
    if proc.returncode or not partial.is_file():raise ValueError("v98 native corrected image build failed: "+proc.stdout+proc.stderr)
    if _sha(SOURCE)!=source_sha:raise ValueError("v98 native corrected source changed during build")
    os.replace(partial,path)
    receipt=dict(source_sha256=source_sha,wrapper_sha256=_sha(Path(__file__)),library_sha256=_sha(path),
        compiler_sha256=_sha(chosen),compile_flags=flags,geometry_sha256=_digest(*geometry),
        locked_source_sha256=verify_locked_sources_v85(ROOT),cell_count=nc,fraction_count=ng,
        thread_policy="per_cell_OMP_no_nested_team",numerical_python_cell_face_loops=0,production_admission=False)
    receipt["receipt_sha256"]=_hash(receipt)
    dependency=os.add_dll_directory(str(chosen.parent)) if os.name=="nt" and hasattr(os,"add_dll_directory") else None
    return NativeAMRCorrectedImagesRuntimeV98(receipt,ctypes.CDLL(str(path)),dependency,path,geometry)
