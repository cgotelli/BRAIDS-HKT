"""Resident CUDA Q8a planform transaction; no CPU fallback."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from braids_core.sediment.planform_v105 import (
    HEADER, PlanformMethodsV105, PlanformParametersV105, ScourMapV105,
    SCOPE, digest, transfer_token_v105,
)


class ResidentPlanformRuntimeV105:
    backend = "implicit_fwave_midpoint_gpu"

    def __init__(self, methods=PlanformMethodsV105(), parameters=PlanformParametersV105()):
        import cupy as cp
        if methods.dry_bank and parameters.thetsd <= 0:
            raise ValueError("redistribution.dry_bank requires positive ThetSD")
        self.cp=cp;self.xp=cp;self.methods=methods;self.parameters=parameters
        path=Path(__file__).with_suffix(".cu")
        source="#define BRAIDS_HD __device__ inline\n"+HEADER.read_text()+path.read_text()
        names=("braids_q8a_slopes_v105","braids_q8a_shear_map_v105",
            "braids_q8a_classic_flux_v105","braids_q8a_scour_rows_v105",
            "braids_q8a_scour_columns_v105","braids_q8a_dry_proposal_v105",
            "braids_q8a_dry_scale_v105","braids_q8a_dry_gather_v105",
            "braids_q8a_bed_v105","braids_q8a_repose_proposal_v105",
            "braids_q8a_repose_scale_v105","braids_q8a_repose_gather_v105")
        options=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")
        self.module=cp.RawModule(code=source,options=options,name_expressions=names)
        props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.receipt={"equations_sha256":hashlib.sha256(HEADER.read_bytes()).hexdigest(),
            "entry_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "device":props["name"].decode(),"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),
            "cupy":cp.__version__,"options":options}
        self.identity=digest({"scope":SCOPE,"backend":self.backend,"methods":methods.fingerprint,
                              "parameters":asdict(parameters),"runtime":self.receipt})

    def _launch(self,name,count,args,threads=128):
        self.module.get_function(name)(((count+threads-1)//threads,),(threads,),args)

    def _real(self,value,shape,name,*,positive=False,nonnegative=False):
        cp=self.cp
        if isinstance(value,cp.ndarray):
            a=value
            if a.shape==():a=cp.full(shape,a,dtype=cp.float64)
            elif a.shape!=shape:a=cp.broadcast_to(a,shape).copy()
            if a.dtype!=cp.float64 or not a.flags.c_contiguous:a=cp.ascontiguousarray(a,dtype=cp.float64)
        else:a=cp.asarray(np.broadcast_to(np.asarray(value),shape).copy(),dtype=cp.float64)
        if not bool(cp.isfinite(a).all()) or (positive and bool((a<=0).any())) or (nonnegative and bool((a<0).any())):
            raise ValueError(f"Invalid CUDA {name}")
        return a

    def _binary(self,value,shape,name):
        cp=self.cp;a=cp.asarray(value,dtype=cp.int32)
        if a.shape==():a=cp.full(shape,a,dtype=cp.int32)
        elif a.shape!=shape:a=cp.broadcast_to(a,shape).copy()
        a=cp.ascontiguousarray(a)
        if bool(((a!=0)&(a!=1)).any()):raise ValueError(f"{name} must be binary")
        return a

    def slopes(self,bed,*,dx_m,dy_m):
        cp=self.cp;a=cp.ascontiguousarray(bed,dtype=cp.float64);shape=a.shape
        if len(shape)!=2:raise ValueError("Q8a requires a structured 2-D CUDA patch")
        nc=a.size;out=cp.empty((3,nc),cp.float64);status=cp.empty(nc,cp.int32)
        self._launch("braids_q8a_slopes_v105",nc,(np.int32(shape[1]),np.int32(shape[0]),
            np.float64(dx_m),np.float64(dy_m),a,out[0],out[1],out[2],status))
        if bool((status!=0).any()):raise ValueError("Invalid CUDA Q8a bed slope")
        return {"dzduu":out[0].reshape(shape),"dzdvv":out[1].reshape(shape),
                "magnitude":out[2].reshape(shape),"output_id":"output.bedslope"}

    def map_scour_shear(self,shear,mapping:ScourMapV105|None):
        cp=self.cp;a=cp.ascontiguousarray(shear,dtype=cp.float64);out=cp.empty_like(a);status=cp.empty(a.size,cp.int32)
        if mapping is None:
            ref=app=cp.empty(0,cp.int32);factor=cp.empty(0,cp.float64)
        else:
            ref=cp.asarray(mapping.reference,cp.int32);app=cp.asarray(mapping.application,cp.int32);factor=cp.asarray(mapping.factor,cp.float64)
            if ref.size and (int(ref.min())<0 or int(app.min())<0 or int(ref.max())>=a.size or int(app.max())>=a.size):
                raise ValueError("Scour mapping index outside CUDA shear state")
        self._launch("braids_q8a_shear_map_v105",a.size,(np.int32(a.size),np.int32(ref.size),a.ravel(),ref,app,factor,out.ravel(),status))
        if bool((status!=0).any()):raise ValueError("Invalid CUDA Scour shear mapping")
        return out

    def _validated(self,layers,rock,depth,area,bulk,erosion,fraction,fixfac,wet,erodible):
        cp=self.cp;q=cp.ascontiguousarray(cp.asarray(layers,dtype=cp.float64))
        if q.ndim!=4 or min(q.shape)<1:raise ValueError("CUDA layers require (layer,grain,y,x)")
        nl,ng,ny,nx=q.shape;shape=(ny,nx)
        bulk=self._real(bulk,(ng,),"bulk density",positive=True);rock=self._real(rock,shape,"bedrock")
        depth=self._real(depth,shape,"depth",nonnegative=True);area=self._real(area,shape,"area",positive=True)
        wet=self._binary(depth>0 if wet is None else wet,shape,"wet mask")
        erodible=self._binary(cp.ones(shape) if erodible is None else erodible,shape,"erodible mask")
        erosion=self._real(0. if erosion is None else erosion,(ng,ny,nx),"erosion")
        fraction=self._real(1./ng if fraction is None else fraction,(ng,ny,nx),"fraction",nonnegative=True)
        total=fraction.sum(axis=0)
        if bool((total<=0).any()):raise ValueError("Positive CUDA surface-fraction sum required")
        fraction=cp.ascontiguousarray(fraction/total)
        fixfac=self._real(1. if fixfac is None else fixfac,(ng,ny,nx),"fixfac",nonnegative=True)
        if not bool(cp.isfinite(q).all()) or bool((q<0).any()):raise ValueError("Invalid CUDA inventory")
        return q,rock,depth,area,bulk,erosion,fraction,fixfac,wet,erodible

    def advance(self,layers,rock,depth,area,bulk_density,*,dt_s,dx_m,dy_m,
                erosion_mass_areal=None,surface_fraction=None,fixfac=None,wet=None,
                erodible=None,scour_map=None,shear=None,toe_face_delta_mass=None,
                toe_history=None,transfer_tokens=()):
        cp=self.cp
        if not np.isfinite([dt_s,dx_m,dy_m]).all() or min(dt_s,dx_m,dy_m)<=0:raise ValueError("Invalid CUDA Q8a scales")
        q,rock,depth,area,bulk,erosion,fraction,fixfac,wet,erodible=self._validated(
            layers,rock,depth,area,bulk_density,erosion_mass_areal,surface_fraction,fixfac,wet,erodible)
        nl,ng,ny,nx=q.shape;nc=nx*ny;trial=q.copy();status=cp.zeros(nc,cp.int32)
        base=rock.ravel().copy()
        if nl>1:base+=cp.sum(trial[1:]/bulk[None,:,None,None],axis=(0,1)).ravel()
        bed=base+cp.sum(trial[0]/bulk[:,None,None],axis=0).ravel()
        slopes=self.slopes(bed.reshape(ny,nx),dx_m=dx_m,dy_m=dy_m)
        qx=cp.zeros((ny,nx),cp.float64);qy=cp.zeros_like(qx);substeps=0
        if self.methods.relaxation=="classic_wet_slope_flux":
            self._launch("braids_q8a_classic_flux_v105",nc,(np.int32(nx),np.int32(ny),np.float64(dx_m),np.float64(dy_m),
                np.float64(self.parameters.wet_slope),np.float64(self.parameters.avalanche_time_s),area.ravel(),bed,
                slopes["dzduu"].ravel(),slopes["dzdvv"].ravel(),slopes["magnitude"].ravel(),qx.ravel(),qy.ravel(),status))
        elif self.methods.relaxation=="classic_scour_bundle":
            slope=self.parameters.scour_slope if scour_map is None else scour_map.slope
            active=cp.ascontiguousarray(trial[0].reshape(ng,nc).copy())
            self._launch("braids_q8a_scour_rows_v105",ny,(np.int32(nx),np.int32(ny),np.int32(ng),np.float64(dx_m),np.float64(slope),area.ravel(),bulk,base,active))
            self._launch("braids_q8a_scour_columns_v105",nx,(np.int32(nx),np.int32(ny),np.int32(ng),np.float64(dy_m),np.float64(slope),area.ravel(),bulk,base,active))
            trial[0]=active.reshape((ng,ny,nx))
        elif self.methods.relaxation=="braids_conservative_wet_dry_avalanche":
            maximum=cp.zeros(nc,cp.float64);conduct=self.parameters.diffusivity_m2_s
            # Geometry-only stability bound; no per-substep device synchronization.
            count=cp.full((ny,nx),2*(conduct*dy_m/dx_m+conduct*dx_m/dy_m));count[:,0]-=conduct*dy_m/dx_m;count[:,-1]-=conduct*dy_m/dx_m
            count[0]-=conduct*dx_m/dy_m;count[-1]-=conduct*dx_m/dy_m
            maxrate=float(cp.max(count/area).item());ds=dt_s if maxrate<=0 else min(dt_s,.45/maxrate)
            substeps=max(1,int(math.ceil(dt_s/ds)));ds=dt_s/substeps
            active=cp.ascontiguousarray(trial[0].reshape(ng,nc).copy());next_active=cp.empty_like(active)
            aqx=cp.empty_like(active);aqy=cp.empty_like(active);scale=cp.empty_like(active);abed=cp.empty(nc)
            for _ in range(substeps):
                self._launch("braids_q8a_bed_v105",nc,(np.int32(nc),np.int32(ng),bulk,base,active,abed))
                self._launch("braids_q8a_repose_proposal_v105",nc,(np.int32(nx),np.int32(ny),np.int32(ng),np.float64(dx_m),np.float64(dy_m),np.float64(ds),np.float64(conduct),np.float64(self.parameters.wet_slope),np.float64(self.parameters.dry_slope),np.float64(self.parameters.wet_depth_switch_m),bulk,abed,depth.ravel(),active,aqx,aqy))
                self._launch("braids_q8a_repose_scale_v105",nc*ng,(np.int32(nx),np.int32(ny),np.int32(ng),area.ravel(),active,aqx,aqy,scale))
                status.fill(0);self._launch("braids_q8a_repose_gather_v105",nc*ng,(np.int32(nx),np.int32(ny),np.int32(ng),area.ravel(),active,aqx,aqy,scale,next_active,status))
                active,next_active=next_active,active
            trial[0]=active.reshape((ng,ny,nx))
        if bool((status!=0).any()):raise ValueError("CUDA Q8a primary redistribution failed")
        moved=cp.zeros((ng,ny,nx),cp.float64)
        if self.methods.dry_bank:
            bed=base+cp.sum(trial[0]/bulk[:,None,None],axis=0).ravel();proposal=cp.empty((nc*4*ng),cp.float64)
            scale=cp.empty((ng,nc),cp.float64);result=cp.empty((ng,nc),cp.float64);delta=cp.empty_like(result);status.fill(0)
            self._launch("braids_q8a_dry_proposal_v105",nc,(np.int32(nx),np.int32(ny),np.int32(ng),np.float64(self.parameters.thetsd),np.float64(self.parameters.hmaxth_m),np.float64(self.parameters.sedthr_m),area.ravel(),bulk,bed,depth.ravel(),wet.ravel(),erodible.ravel(),erosion.reshape(ng,nc),fraction.reshape(ng,nc),fixfac.reshape(ng,nc),trial[0].reshape(ng,nc),proposal))
            self._launch("braids_q8a_dry_scale_v105",nc*ng,(np.int32(nx),np.int32(ny),np.int32(ng),area.ravel(),wet.ravel(),trial[0].reshape(ng,nc),proposal,scale))
            self._launch("braids_q8a_dry_gather_v105",nc*ng,(np.int32(nx),np.int32(ny),np.int32(ng),area.ravel(),wet.ravel(),trial[0].reshape(ng,nc),proposal,scale,result,delta,status))
            if bool((status!=0).any()):raise ValueError("CUDA Q8a dry-bank transfer failed")
            trial[0]=result.reshape((ng,ny,nx));moved+=delta.reshape((ng,ny,nx))
        tokens=list(transfer_tokens)
        if toe_face_delta_mass is not None:
            if not self.methods.toe_collapse:raise ValueError("Toe proposal supplied without selected CUDA route")
            raw_input=cp.asarray(toe_face_delta_mass,dtype=cp.float64)
            if raw_input.shape==(ng,ny,nx):
                delta=cp.zeros_like(trial);delta[0]=self._real(raw_input,(ng,ny,nx),"toe/collapse mass")
            elif raw_input.shape==trial.shape:delta=self._real(raw_input,trial.shape,"toe/collapse layer mass")
            else:raise ValueError("CUDA toe/collapse proposal must resolve active or all bed layers")
            raw=cp.asnumpy(raw_input);token=transfer_token_v105("redistribution.braids_toe_collapse",raw)
            if token in tokens:raise ValueError("Toe/collapse material transfer applied twice")
            if bool((cp.abs(cp.sum(delta,axis=(0,2,3)))>1.e-9).any()):raise ValueError("CUDA toe proposal must conserve each class")
            next_layers=trial+delta/area
            if bool((next_layers<-1.e-11).any()):raise ValueError("CUDA toe proposal exceeds donor inventory")
            trial=cp.maximum(next_layers,0.);moved+=cp.sum(delta,axis=0);tokens.append(token)
        elif self.methods.toe_collapse:raise ValueError("Selected CUDA toe route requires a conservative proposal")
        bed=rock.copy()
        for l in range(nl):bed+=cp.sum(trial[l]/bulk[:,None,None],axis=0)
        return {"layers":trial,"bed_elevation_m":bed,"bedslope":self.slopes(bed,dx_m=dx_m,dy_m=dy_m),
            "classic_flux":{"qx_m3_s":qx,"qy_m3_s":qy},"mapped_shear":None if shear is None else self.map_scour_shear(shear,scour_map),
            "dry_bank_delta_mass_kg":moved,"substeps":substeps,"transfer_tokens":tuple(tokens),
            "method_ids":self.methods.method_ids,"runtime_identity":self.identity,"dt_s":float(dt_s),
            "toe_history":None if toe_history is None else {k:cp.asarray(v).copy() for k,v in toe_history.items()}}


class ResidentPlanformOwnerV105:
    def __init__(self,runtime,layers,rock,depth,area,bulk_density,*,time_s=0.,updates=0,auxiliary_state=None):
        self.runtime=runtime;cp=runtime.cp
        q,rock,depth,area,bulk,*_=runtime._validated(layers,rock,depth,area,bulk_density,None,None,None,None,None)
        self.layers=q;self.rock=rock;self.depth=depth;self.area=area;self.bulk=bulk
        self.time_s=float(time_s);self.updates=int(updates);self._trial=None
        self.auxiliary={} if auxiliary_state is None else {k:cp.asarray(v).copy() for k,v in auxiliary_state.items()}

    @property
    def state_sha256(self):
        cp=self.runtime.cp;h=hashlib.sha256()
        for x in (self.layers,self.rock,self.depth,self.area,self.bulk):h.update(cp.asnumpy(x).tobytes())
        h.update(np.asarray([self.time_s,self.updates],np.float64).tobytes())
        for key in sorted(self.auxiliary):h.update(key.encode());h.update(cp.asnumpy(self.auxiliary[key]).tobytes())
        return h.hexdigest()

    def propose(self,**kwargs):
        if self._trial is not None:raise RuntimeError("CUDA Q8a trial already pending")
        self._trial=self.runtime.advance(self.layers,self.rock,self.depth,self.area,self.bulk,**kwargs);return self._trial

    def commit(self,*,cancelled=False,accept=True):
        if self._trial is None:raise RuntimeError("No CUDA Q8a trial pending")
        trial=self._trial;self._trial=None
        if cancelled:raise InterruptedError("CUDA Q8a transaction cancelled before publication")
        if not accept:return False
        self.layers=trial["layers"].copy();self.time_s+=trial["dt_s"]
        if trial.get("toe_history") is not None:self.auxiliary=trial["toe_history"]
        self.updates+=1;return True

    def discard(self):self._trial=None

    def checkpoint(self):
        if self._trial is not None:raise RuntimeError("Cannot checkpoint pending CUDA Q8a trial")
        cp=self.runtime.cp
        return {"schema":"braids.planform.gpu.v105","runtime_identity":self.runtime.identity,
            "time_s":self.time_s,"updates":self.updates,"layers":cp.asnumpy(self.layers),
            "rock":cp.asnumpy(self.rock),"depth":cp.asnumpy(self.depth),"area":cp.asnumpy(self.area),
            "bulk":cp.asnumpy(self.bulk),"auxiliary":{k:cp.asnumpy(v) for k,v in self.auxiliary.items()},
            "state_sha256":self.state_sha256}

    def restore(self,document:Mapping):
        if document.get("schema")!="braids.planform.gpu.v105" or document.get("runtime_identity")!=self.runtime.identity:
            raise ValueError("CUDA Q8a checkpoint identity differs")
        candidate=ResidentPlanformOwnerV105(self.runtime,document["layers"],document["rock"],document["depth"],
            document["area"],document["bulk"],time_s=document["time_s"],updates=document["updates"])
        candidate.auxiliary={k:self.runtime.cp.asarray(v) for k,v in document.get("auxiliary",{}).items()}
        if candidate.state_sha256!=document.get("state_sha256"):raise ValueError("CUDA Q8a checkpoint digest differs")
        self.__dict__.update(candidate.__dict__)


def remap_resident_planform_owner_v105(owner,offsets,sources,weights,target_area):
    """GPU-resident counterpart of the parent-owned Q8a conservative remap."""
    cp=owner.runtime.cp
    offsets=np.asarray(offsets,np.int64);sources=np.asarray(sources,np.int64);weights=np.asarray(weights,float)
    area_host=cp.asnumpy(owner.area).ravel();target=cp.asarray(target_area,dtype=cp.float64);nt=target.size
    if (offsets.shape!=(nt+1,) or offsets[0]!=0 or offsets[-1]!=len(sources) or
            len(sources)!=len(weights) or np.any(sources<0) or np.any(sources>=area_host.size) or
            not np.isfinite(weights).all() or np.any(weights<0) or bool((target<=0).any())):
        raise ValueError("Invalid CUDA Q8a remap CSR")
    row=np.repeat(np.arange(nt),np.diff(offsets));target_host=cp.asnumpy(target).ravel()
    if (not np.allclose(np.bincount(row,weights=weights,minlength=nt),1.,rtol=0,atol=1e-13) or
        not np.allclose(np.bincount(sources,weights=weights*target_host[row],minlength=area_host.size),
                        area_host,rtol=0,atol=1e-12)):
        raise ValueError("CUDA Q8a remap weights do not conserve area")
    shape=target.shape if target.ndim==2 else (1,nt);nl,ng=owner.layers.shape[:2]
    def map_float(value):
        flat=value.reshape((-1,area_host.size));out=cp.empty((flat.shape[0],nt),value.dtype)
        for t in range(nt):
            idx=cp.asarray(sources[offsets[t]:offsets[t+1]]);w=cp.asarray(weights[offsets[t]:offsets[t+1]])
            out[:,t]=cp.sum(flat[:,idx]*w,axis=1)
        return out
    layers=map_float(owner.layers).reshape((nl,ng)+shape)
    rock=map_float(owner.rock).reshape(shape);depth=map_float(owner.depth).reshape(shape)
    before=cp.sum(owner.layers*owner.area,axis=(2,3));after=cp.sum(layers*target.reshape(shape),axis=(2,3))
    if not bool(cp.allclose(before,after,rtol=0,atol=2e-10)):raise ValueError("CUDA Q8a remap does not conserve inventory")
    auxiliary={}
    for key,value in owner.auxiliary.items():
        a=cp.asarray(value);leading=a.shape[:-2]
        if a.shape[-2:]!=owner.area.shape:raise ValueError(f"CUDA Q8a auxiliary {key} is not cell-owned")
        flat=a.reshape((-1,area_host.size));out=cp.empty((flat.shape[0],nt),a.dtype)
        for t in range(nt):
            idx=cp.asarray(sources[offsets[t]:offsets[t+1]]);w=cp.asarray(weights[offsets[t]:offsets[t+1]])
            out[:,t]=cp.max(flat[:,idx],axis=1) if a.dtype.kind in "iu" else cp.sum(flat[:,idx]*w,axis=1)
        auxiliary[key]=out.reshape(leading+shape)
    return ResidentPlanformOwnerV105(owner.runtime,layers,rock,depth,target.reshape(shape),owner.bulk,
        time_s=owner.time_s,updates=owner.updates,auxiliary_state=auxiliary)

