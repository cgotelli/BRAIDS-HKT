"""Resident CUDA Q6b boundaries and point loads; no CPU fallback."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from braids_core.sediment.sediment_boundaries_v103 import (
    HEADER, PARSED_UNUSED_BED_MODES, SCOPE, _digest,
)


class ResidentSedimentBoundaryRuntimeV103:
    backend="implicit_fwave_midpoint_gpu"
    def __init__(self,fraction_ids,solid_density):
        import cupy as cp
        self.cp=cp; self.fraction_ids=tuple(fraction_ids)
        rho=np.ascontiguousarray(solid_density,dtype=float)
        if (not self.fraction_ids or len(set(self.fraction_ids))!=len(self.fraction_ids) or
                rho.shape!=(len(self.fraction_ids),) or not np.isfinite(rho).all() or np.any(rho<=0)):
            raise ValueError("Invalid Q6b CUDA fractions or density")
        self.rho=cp.asarray(rho)
        path=Path(__file__).with_suffix(".cu")
        source="#define BRAIDS_HD __device__\n"+HEADER.read_text()+path.read_text()
        options=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")
        names=("braids_q6b_bed_v103","braids_q6b_composition_v103","braids_q6b_load_v103")
        self.module=cp.RawModule(code=source,options=options,name_expressions=names)
        self.bed_kernel=self.module.get_function(names[0]); self.comp_kernel=self.module.get_function(names[1]); self.load_kernel=self.module.get_function(names[2])
        self.receipt={"equations":hashlib.sha256(HEADER.read_bytes()).hexdigest(),
            "entry":hashlib.sha256(path.read_bytes()).hexdigest(),"options":options,
            "device":cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode(),
            "cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),"cupy":cp.__version__}
        self.identity=_digest({"scope":SCOPE,"backend":self.backend,"fractions":self.fraction_ids,
                               "rho":rho.tolist(),"runtime":self.receipt})

    def _resident(self,name,value,shape,dtype=None):
        cp=self.cp; dtype=cp.float64 if dtype is None else dtype
        if (not isinstance(value,cp.ndarray) or value.dtype!=dtype or value.shape!=shape or
                not value.flags.c_contiguous or not bool(cp.isfinite(value).all())):
            raise ValueError(f"{name} must be resident contiguous {dtype} {shape}")
        return value

    def bed_change(self,mode,inward,dt,computed,interior,depth,prescribed):
        cp=self.cp; nc=int(computed.size)
        arrays=[self._resident(n,x,(nc,)) for n,x in zip(("computed","interior","depth","prescribed"),(computed,interior,depth,prescribed))]
        if not np.isfinite(dt) or dt<0: raise ValueError("Invalid bed-boundary interval")
        out=cp.empty(nc,cp.float64); status=cp.empty(nc,cp.int32)
        self.bed_kernel(((nc+127)//128,),(128,),(np.int32(nc),np.int32(mode),np.int32(bool(inward)),np.float64(dt),*arrays,out,status))
        if bool((status!=0).any()):
            if mode in PARSED_UNUSED_BED_MODES: raise NotImplementedError("IBedCond 6/7 is parsed but unused in locked classic FLOW")
            raise ValueError("CUDA bed-boundary update failed")
        return out

    def composition(self,mode,boundary,interior,values,*,coarse_layer=False):
        cp=self.cp; nl,ng,nc=boundary.shape
        boundary=self._resident("boundary",boundary,(nl,ng,nc)); inside=self._resident("interior",interior,(nl,ng,nc)); values=self._resident("values",values,(ng,nc))
        if ng!=len(self.fraction_ids): raise ValueError("CUDA boundary fraction count differs")
        out=cp.ascontiguousarray(boundary.copy()); status=cp.empty(nc,cp.int32)
        self.comp_kernel(((nc+127)//128,),(128,),(np.int32(nc),np.int32(ng),np.int32(nl),np.int32(mode),
            np.int32(bool(coarse_layer)),self.rho,inside,values,out,status))
        if bool((status!=0).any()): raise ValueError("CUDA composition-boundary update failed")
        return out

    def point_load(self,mass,dt,discharge,concentration,initial_volume,current_volume,area,wet):
        cp=self.cp; ng,nc=mass.shape
        out=cp.ascontiguousarray(self._resident("mass",mass,(ng,nc)).copy())
        q=self._resident("discharge",discharge,(nc,)); cin=self._resident("concentration",concentration,(ng,nc))
        v0=self._resident("initial_volume",initial_volume,(nc,)); v=self._resident("current_volume",current_volume,(nc,)); a=self._resident("area",area,(nc,))
        wet=self._resident("wet",wet,(nc,),cp.int32)
        if ng!=len(self.fraction_ids) or not np.isfinite(dt) or dt<0: raise ValueError("CUDA point-load contract differs")
        delta=cp.empty_like(out); status=cp.empty(nc,cp.int32)
        self.load_kernel(((nc+127)//128,),(128,),(np.int32(nc),np.int32(ng),np.float64(dt),q,cin,v0,v,a,wet,out,delta,status))
        if bool((status==2).any()): raise ValueError("Positive suspended source has zero initial cell volume")
        if bool((status!=0).any()): raise ValueError("CUDA point suspended load failed")
        return out,delta


__all__=["ResidentSedimentBoundaryRuntimeV103"]
