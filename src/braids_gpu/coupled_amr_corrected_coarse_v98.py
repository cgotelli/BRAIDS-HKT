"""Actual CUDA reflux-corrected conserved images and derived EOS, not an owner.

The caller authenticates the register correction and retains its authoritative
interface ledgers. This helper performs neither HLES nor any sediment source
again. Frozen v92 EOS/viscosity and v85 mixture kernels are reused unchanged.
"""
from pathlib import Path

import numpy as np

from braids_core.sediment.density_active_projection_v97 import _array,_digest,_hash,_sha
from braids_core.sediment.density_trace_projection_v98 import _hex
from braids_core.sediment.dynamic_fluid_settling_v88 import (
    EOS_ECKART,EOS_UNESCO,EOS_NACL,
)
from braids_core.sediment.densin_fwave_v85 import verify_locked_sources_v85
from braids_core.sediment.transported_fluid_settling_v92 import MODE_TRANSPORTED,MODE_CONSTANT

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).with_suffix(".cu")
EOS_SOURCE=ROOT/"braids_gpu/transported_fluid_settling_v92.cu"
MIXTURE_SOURCE=ROOT/"braids_gpu/densin_fwave_v85.cu"
OPTIONS=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")
EOS_CODES={EOS_ECKART:0,EOS_UNESCO:1,EOS_NACL:2}
KEYS=("q","suspended","bed","fluid","concentration","temperature","salinity",
      "bed_elevation","eta","wet_mask","dry_transfer","momentum_cleanup",
      "fluid_cleanup","water_density","viscosity","mixture_density")


def controls_v98(*,cell,nc,dry_tolerance_m,temperature_mode,salinity_mode,
    equilibrium_temperature_c,equilibrium_salinity_ppt,density_eos,reference_water_density_kg_m3):
    if (type(cell) is not int or not 0<=cell<nc or temperature_mode not in (MODE_TRANSPORTED,MODE_CONSTANT) or
        salinity_mode not in (MODE_TRANSPORTED,MODE_CONSTANT) or density_eos not in EOS_CODES or
        any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) for v in (
            dry_tolerance_m,equilibrium_temperature_c,equilibrium_salinity_ppt,reference_water_density_kg_m3)) or
        dry_tolerance_m<0 or equilibrium_temperature_c<=-20 or equilibrium_salinity_ppt<0 or
        reference_water_density_kg_m3<=0):
        raise ValueError("v98 corrected CUDA image controls differ")
    return dict(cell=cell,dry_tolerance_m=float(dry_tolerance_m),temperature_mode=temperature_mode,
        salinity_mode=salinity_mode,equilibrium_temperature_c=float(equilibrium_temperature_c),
        equilibrium_salinity_ppt=float(equilibrium_salinity_ppt),density_eos=density_eos,
        reference_water_density_kg_m3=float(reference_water_density_kg_m3))


class ResidentAMRCorrectedCoarseImagesV98:
    def __init__(self,*,backend,initial_bed_elevation_m,reference_bed_mass_kg_m2,
                 dry_bulk_density_kg_m3,solid_density_kg_m3):
        if backend!="implicit_fwave_midpoint_gpu":
            raise ValueError("v98 corrected images require explicitly selected implicit GPU")
        raw=np.asarray(reference_bed_mass_kg_m2)
        if raw.ndim!=2 or min(raw.shape)<1:raise ValueError("v98 corrected bed reference shape differs")
        self.ng,self.nc=raw.shape
        z=_array(initial_bed_elevation_m,(self.nc,),"initial bed")
        ref=_array(raw,(self.ng,self.nc),"reference bed")
        bulk=_array(dry_bulk_density_kg_m3,(self.ng,),"bulk density")
        solid=_array(solid_density_kg_m3,(self.ng,),"solid density")
        if np.any(ref<0) or np.any(bulk<=0) or np.any(solid<=0):
            raise ValueError("v98 corrected reference/density domain differs")
        # Immutable independent copies; callers cannot mutate captured geometry.
        self.geometry_host=tuple(np.array(a,copy=True,order="C") for a in (z,ref,bulk,solid))
        for a in self.geometry_host:a.setflags(write=False)
        import cupy as cp
        self.cp=cp;self.device_id=int(cp.cuda.runtime.getDevice())
        self.geometry_device=tuple(cp.asarray(a) for a in self.geometry_host)
        self.module=cp.RawModule(code=SOURCE.read_text(encoding="utf-8")+"\n"+
            EOS_SOURCE.read_text(encoding="utf-8")+"\n"+MIXTURE_SOURCE.read_text(encoding="utf-8"),
            options=OPTIONS,name_expressions=("amr_corrected_images_v98","v92_fluid_properties","densin_mixture_v85"))
        self.correct_kernel=self.module.get_function("amr_corrected_images_v98")
        self.eos_kernel=self.module.get_function("v92_fluid_properties")
        self.mixture_kernel=self.module.get_function("densin_mixture_v85")
        self.receipt=dict(backend=backend,actual_cuda=True,automatic_cpu_fallback=False,
            source_sha256=_sha(SOURCE),wrapper_sha256=_sha(Path(__file__)),
            eos_source_sha256=_sha(EOS_SOURCE),mixture_source_sha256=_sha(MIXTURE_SOURCE),
            geometry_sha256=_digest(*self.geometry_host),device_id=self.device_id,
            cell_count=self.nc,fraction_count=self.ng,cuda_options=list(OPTIONS),
            locked_source_sha256=verify_locked_sources_v85(ROOT),
            source_or_constitutive_replay_count=0,state_or_clock_publication_count=0,
            production_admission=False)
        self.receipt["receipt_sha256"]=_hash(self.receipt)
        self.last_resident_result=None

    def _validate_runtime(self):
        if (self.receipt["receipt_sha256"]!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=self.receipt["source_sha256"] or _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(EOS_SOURCE)!=self.receipt["eos_source_sha256"] or _sha(MIXTURE_SOURCE)!=self.receipt["mixture_source_sha256"] or
            _digest(*self.geometry_host)!=self.receipt["geometry_sha256"]):
            raise ValueError("v98 corrected CUDA runtime/source/geometry identity differs")
        if any(not np.array_equal(self.cp.asnumpy(a),b) for a,b in zip(self.geometry_device,self.geometry_host)):
            raise ValueError("v98 corrected CUDA resident geometry differs")

    def launch_resident(self,q,suspended,bed,fluid,delta_q,delta_suspended,delta_bed,delta_fluid,**kwargs):
        """Pure candidate kernels; caller authenticates runtime and register first."""
        controls=controls_v98(nc=self.nc,**kwargs)
        cp,nc,ng=self.cp,self.nc,self.ng
        inputs=(q,suspended,bed,fluid,delta_q,delta_suspended,delta_bed,delta_fluid)
        shapes=((3,nc),(ng,nc),(ng,nc),(2,nc),(3,),(ng,),(ng,),(2,))
        if any(not isinstance(a,cp.ndarray) or a.device.id!=self.device_id or not a.flags.c_contiguous or
            a.dtype!=cp.float64 or a.shape!=shape for a,shape in zip(inputs,shapes)):
            raise ValueError("v98 corrected images require contiguous selected-device resident arrays")
        out={name:cp.empty(shape,dtype=cp.int32 if name=="wet_mask" else cp.float64) for name,shape in (
            ("q",(3,nc)),("suspended",(ng,nc)),("bed",(ng,nc)),("fluid",(2,nc)),
            ("concentration",(ng,nc)),("temperature",(nc,)),("salinity",(nc,)),
            ("bed_elevation",(nc,)),("eta",(nc,)),("wet_mask",(nc,)),
            ("dry_transfer",(ng,nc)),("momentum_cleanup",(2,nc)),("fluid_cleanup",(2,nc)),
            ("water_density",(nc,)),("viscosity",(nc,)),("mixture_density",(nc,)))}
        out.update(image_status=cp.empty(nc,cp.int32),eos_status=cp.empty(nc,cp.int32),mixture_status=cp.empty(nc,cp.int32))
        geometry=self.geometry_device
        tc=int(controls["temperature_mode"]==MODE_TRANSPORTED);sc=int(controls["salinity_mode"]==MODE_TRANSPORTED)
        grid=((nc+127)//128,);block=(128,)
        self.correct_kernel(grid,block,(np.int32(nc),np.int32(ng),np.int32(controls["cell"]),
            np.float64(controls["dry_tolerance_m"]),np.int32(tc),np.int32(sc),
            np.float64(controls["equilibrium_temperature_c"]),np.float64(controls["equilibrium_salinity_ppt"]),
            *inputs,*geometry[:3],*(out[name] for name in KEYS[:13]),out["image_status"]))
        self.eos_kernel(grid,block,(np.int32(nc),out["temperature"],out["salinity"],
            np.int32(EOS_CODES[controls["density_eos"]]),np.int32(tc or sc),
            np.float64(controls["reference_water_density_kg_m3"]),out["water_density"],out["viscosity"],out["eos_status"]))
        self.mixture_kernel(grid,block,(out["concentration"],out["water_density"],geometry[3],
            np.int64(ng),np.int64(nc),out["mixture_density"],out["mixture_status"]))
        return out

    def evaluate(self,q,suspended,bed,fluid,delta_q,delta_suspended,delta_bed,delta_fluid,*,
                 correction_receipt_sha256,source_state_sha256,cancelled=False,**kwargs):
        self.last_resident_result=None
        cancel=cancelled if callable(cancelled) else lambda:cancelled
        if not callable(cancelled) and type(cancelled) is not bool:
            raise ValueError("v98 corrected image cancellation must be bool/callable")
        if cancel():raise InterruptedError("v98 corrected images cancelled before launch")
        controls=controls_v98(nc=self.nc,**kwargs)
        provenance=dict(correction_receipt_sha256=_hex(correction_receipt_sha256,"correction receipt"),
            source_state_sha256=_hex(source_state_sha256,"source state"))
        nc,ng=self.nc,self.ng
        shapes=((3,nc),(ng,nc),(ng,nc),(2,nc),(3,),(ng,),(ng,),(2,))
        arrays=tuple(_array(value,shape,"corrected image input") for value,shape in zip(
            (q,suspended,bed,fluid,delta_q,delta_suspended,delta_bed,delta_fluid),shapes))
        cp=self.cp
        with cp.cuda.Device(self.device_id):
            self._validate_runtime()
            result=self.launch_resident(*(cp.asarray(a) for a in arrays),**controls)
            if any(np.any(cp.asnumpy(result[key])) for key in ("image_status","eos_status","mixture_status")):
                raise ValueError("v98 corrected GPU image is outside conserved/EOS domain")
            values={key:cp.asnumpy(result[key]) for key in KEYS}
        if cancel():raise InterruptedError("v98 corrected images cancelled after launch")
        for value in values.values():value.setflags(write=False)
        metadata=dict(runtime_sha256=self.receipt["receipt_sha256"],backend="actual_cuda",
            input_sha256=_digest(*arrays),output_sha256=_digest(*(values[key] for key in KEYS)),
            controls_sha256=_hash(controls),**provenance,
            reflux_correction_application_count=1,reflux_dry_cleanup_owner_count=1,
            water_geometry_source_count=0,settling_exchange_q3_replay_count=0,
            hles_evaluation_count=0,dss_recompute_count=0,state_or_clock_publication_count=0,
            automatic_cpu_fallback=False,production_admission=False)
        self.last_resident_result=result
        return dict(images=values,metadata=metadata,receipt_sha256=_hash(metadata))
