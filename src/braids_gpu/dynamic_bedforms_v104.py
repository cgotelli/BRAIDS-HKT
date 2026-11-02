"""Resident CUDA Q7b dynamic-bedform route; no CPU fallback."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import numpy as np

from braids_core.sediment.dynamic_bedforms_v104 import (
    HEADER, DynamicBedformMethodsV104, DynamicBedformParametersV104,
    _boundary_codes, _modes, digest, resolve_dynamic_bedforms_v104,
)


class ResidentDynamicBedformRuntimeV104:
    backend="implicit_fwave_midpoint_gpu"

    def __init__(self,methods=DynamicBedformMethodsV104(),parameters=DynamicBedformParametersV104()):
        import cupy as cp
        self.cp=cp;self.xp=cp;self.methods=methods;self.parameters=parameters
        self.resolved=resolve_dynamic_bedforms_v104(methods)
        self.mode_host=_modes(methods);self.params_host=parameters.wire()
        self.mode=cp.asarray(self.mode_host);self.params=cp.asarray(self.params_host)
        path=Path(__file__).with_suffix(".cu")
        source="#define BRAIDS_HD __device__\n"+HEADER.read_text()+path.read_text()
        options=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")
        self.module=cp.RawModule(code=source,options=options,name_expressions=(
            "braids_dynamic_diagnostics_v104","braids_dynamic_roughness_relax_v104",
            "braids_dynamic_step_v104"))
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.receipt={"equations_sha256":hashlib.sha256(HEADER.read_bytes()).hexdigest(),
                      "entry_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                      "device":props["name"].decode(),"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
                      "cupy":cp.__version__,"options":options}
        self.identity=digest({"backend":self.backend,"methods":self.resolved.fingerprint,
                              "parameters":asdict(parameters),"runtime":self.receipt})

    def _field(self,value,shape,name,*,positive=False,nonnegative=False,dtype=None):
        cp=self.cp;dtype=cp.float64 if dtype is None else dtype
        if isinstance(value,cp.ndarray):
            a=value
            if a.shape==():a=cp.full(shape,a,dtype=dtype)
            elif a.shape!=shape:raise ValueError(f"Invalid CUDA {name} shape")
            if a.dtype!=dtype or not a.flags.c_contiguous:a=cp.ascontiguousarray(a,dtype=dtype)
        else:a=cp.asarray(np.broadcast_to(np.asarray(value),shape).copy(),dtype=dtype)
        if dtype==cp.float64 and (not bool(cp.isfinite(a).all()) or (positive and bool((a<=0).any())) or (nonnegative and bool((a<0).any()))):
            raise ValueError(f"Invalid CUDA {name}")
        return a

    def _diagnostics(self,shape,fields):
        cp=self.cp;nc=int(np.prod(shape));out=cp.empty((5,nc),cp.float64);status=cp.empty(nc,cp.int32)
        order=(3,4,5,6,7,0,1,2,8,9,10)
        self.module.get_function("braids_dynamic_diagnostics_v104")(((nc+127)//128,),(128,),
            (np.int32(nc),self.mode,self.params,*(fields[i].ravel() for i in order),out,status))
        if bool((status!=0).any()):raise ValueError("Nonphysical CUDA Q7b cell")
        return out

    def _roughness_relax(self,shape,current,target,dt_s):
        cp=self.cp;nc=int(np.prod(shape));out=cp.empty((3,nc),cp.float64);status=cp.empty(nc,cp.int32)
        self.module.get_function("braids_dynamic_roughness_relax_v104")(((nc+127)//128,),(128,),
            (np.int32(nc),self.params,np.float64(dt_s),*(x.ravel() for x in current+target),out,status))
        if bool((status!=0).any()):raise ValueError("Invalid CUDA Van Rijn 2007 roughness relaxation state")
        return [out[k].reshape(shape) for k in range(3)]

    def advance(self,height,equilibrium_height,length,depth,u,v,*,bedload_x=0.,bedload_y=0.,
                ripple_roughness=0.,mega_roughness=0.,dune_roughness=0.,static_roughness_target=None,
                wet=None,dt_s,dx_m,dy_m,morfac=1.,
                morphology_started=True,boundary=("open","open","open","open")):
        cp=self.cp;shape=height.shape
        if len(shape)!=2 or min(shape)<1:raise ValueError("Q7b requires a 2-D structured CUDA patch")
        ny,nx=shape;nc=nx*ny
        if not np.isfinite([dt_s,dx_m,dy_m,morfac]).all() or dt_s<=0 or dx_m<=0 or dy_m<=0 or morfac<0:
            raise ValueError("Invalid CUDA Q7b time/grid scale")
        names=("height","equilibrium_height","length","depth","u","v","bedload_x","bedload_y","ripple","mega")
        vals=(height,equilibrium_height,length,depth,u,v,bedload_x,bedload_y,
              ripple_roughness,mega_roughness,dune_roughness)
        names=names+("dune_roughness",)
        fields=[self._field(x,shape,n,positive=n=="length",nonnegative=n in ("height","equilibrium_height","depth","ripple","mega","dune_roughness")) for x,n in zip(vals,names)]
        wet=self._field(fields[3]>self.parameters.dry_depth_m if wet is None else wet,shape,"wet",dtype=cp.int32)
        if bool(((wet!=0)&(wet!=1)).any()):raise ValueError("CUDA Q7b wet mask must be binary")
        diag=self._diagnostics(shape,fields)
        effective_dt=dt_s*(morfac if self.methods.clock=="morphological" else 1.)
        if not morphology_started or (self.methods.clock=="morphological" and morfac==0):effective_dt=0.
        if self.methods.roughness=="van_rijn_2007_static":
            if not isinstance(static_roughness_target,dict) or set(static_roughness_target)!={"ripple","mega_ripple","dune"}:
                raise ValueError("CUDA Van Rijn 2007 static roughness requires all three Q7a target components")
            target=[self._field(static_roughness_target[k],shape,f"target_{k}",nonnegative=True)
                    for k in ("ripple","mega_ripple","dune")]
            fields[8:11]=self._roughness_relax(shape,fields[8:11],target,dt_s)
        elif static_roughness_target is not None:raise ValueError("Static roughness target is only valid for Van Rijn 2007")
        if effective_dt and self.methods.relaxation != "equilibrium":
            diag[2]=cp.where(diag[2]>0.,cp.maximum(diag[2],effective_dt),0.)
        cfl=float((effective_dt*cp.max(cp.abs(diag[0])/dx_m+cp.abs(diag[1])/dy_m)).item()) if effective_dt else 0.
        if self.methods.advection and self.methods.cfl_control:nsub=max(1,int(math.ceil(cfl/self.parameters.cfl_target)))
        else:
            nsub=1
            if self.methods.advection and cfl>1.+1e-12:raise ValueError("CUDA bedform CFL exceeds one while control is disabled")
        current=fields[0].copy().ravel();status=cp.zeros(nc,cp.int32)
        if effective_dt:
            bcode=cp.asarray(_boundary_codes(boundary));ds=effective_dt/nsub
            kernel=self.module.get_function("braids_dynamic_step_v104")
            for _ in range(nsub):
                following=cp.empty_like(current)
                kernel(((nc+127)//128,),(128,),(np.int32(nx),np.int32(ny),self.mode,bcode,np.float64(ds),
                    np.float64(dx_m),np.float64(dy_m),current,fields[1].ravel(),diag[0],diag[1],diag[2],
                    wet.ravel(),following,status))
                current=following
            if bool((status!=0).any()):raise ValueError("Unstable CUDA Q7b update")
        final_fields=list(fields);final_fields[0]=current.reshape(shape);final_diag=self._diagnostics(shape,final_fields)
        return {"height_m":current.reshape(shape),"equilibrium_height_m":fields[1],"length_m":fields[2],
                "celerity_x_m_s":diag[0].reshape(shape),"celerity_y_m_s":diag[1].reshape(shape),
                "relaxation_time_s":diag[2].reshape(shape),"dune_roughness_m":final_diag[3].reshape(shape),
                "roughness_m":final_diag[4].reshape(shape),"effective_dt_s":effective_dt,
                "ripple_roughness_m":fields[8],"mega_roughness_m":fields[9],
                "static_dune_roughness_m":fields[10],
                "hydraulic_dt_s":dt_s,"maximum_courant":cfl,"substeps":nsub,"wet":wet}


__all__=["ResidentDynamicBedformRuntimeV104"]
