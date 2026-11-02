"""One resident combined scalar diffusion limiter with internal trace ledgers."""
from pathlib import Path

import numpy as np

from braids_core.sediment.scalar_diffusion_trace_v98 import (
    _array,_hash,_sha,_cancel,validate_diffusion_trace_v98,
)
from braids_core.sediment.density_active_projection_v97 import _digest
from braids_core.sediment.shared_turbulence_consumers_v91 import verify_locked_sources_v91

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).with_suffix(".cu")
BASE_SOURCE=ROOT/"braids_gpu/flow2dh_boundary_diffusion.cu"
OPTIONS=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")
NAMES=("m2b_diffusion_faces","m2b_diffusion_theta","m2b_diffusion_limit",
       "m2b_diffusion_update","m2b_diffusion_ledger","diffusion_trace_faces_v98","diffusion_trace_ledger_v98")


def topology_token(top):
    return _hash(dict(arrays=_digest(*(getattr(top,name) for name in (
        "owner","neighbour","boundary_index","cell_face_offset","cell_faces",
        "cell_face_sign","cell_area_m2","centre_distance_m"))),
        counts=[top.cell_count,top.face_count,top.boundary_count],
        dry=float(top.profile.diffusion_dry_threshold_m)))


class ResidentScalarDiffusionTraceV98:
    def __init__(self,topology,*,backend):
        if backend!="implicit_fwave_midpoint_gpu":
            raise ValueError("v98 diffusion trace requires explicitly selected implicit GPU")
        import cupy as cp
        self.cp,self.topology=cp,topology
        self.device_id=int(cp.cuda.runtime.getDevice())
        self.geometry_sha=topology_token(topology)
        self.module=cp.RawModule(code=BASE_SOURCE.read_text(encoding="utf-8")+"\n"+SOURCE.read_text(encoding="utf-8"),
            options=OPTIONS,name_expressions=NAMES)
        self.kernels={name:self.module.get_function(name) for name in NAMES}
        self.host=tuple(np.ascontiguousarray(getattr(topology,name),dtype=np.int32) for name in (
            "owner","neighbour","boundary_index","cell_face_offset","cell_faces","cell_face_sign"))
        self.device=tuple(cp.asarray(a) for a in self.host)
        bid=self.host[2];exterior=np.flatnonzero(bid>=0)
        bfaces=np.ascontiguousarray(exterior[np.argsort(bid[exterior],kind="stable")],dtype=np.int32)
        boffset=np.ascontiguousarray(np.concatenate(([0],np.cumsum(np.bincount(bid[exterior],
            minlength=topology.boundary_count)))),dtype=np.int32)
        self.boundary_device=(cp.asarray(boffset),cp.asarray(bfaces))
        self.boundary_host=(boffset,bfaces)
        self.receipt=dict(backend=backend,actual_cuda=True,automatic_cpu_fallback=False,
            source_sha256=_sha(SOURCE),base_source_sha256=_sha(BASE_SOURCE),wrapper_sha256=_sha(Path(__file__)),
            core_wrapper_sha256=_sha(ROOT/"braids_core/sediment/scalar_diffusion_trace_v98.py"),
            topology_sha256=self.geometry_sha,device_id=self.device_id,cuda_options=list(OPTIONS),
            locked_source_sha256=verify_locked_sources_v91(ROOT),
            combined_diffusion_owner_count=1,inventory_limiter_owner_count=1,production_admission=False)
        self.receipt["receipt_sha256"]=_hash(self.receipt)

    def _validate_runtime(self):
        if (self.receipt["receipt_sha256"]!=_hash({k:v for k,v in self.receipt.items() if k!="receipt_sha256"}) or
            _sha(SOURCE)!=self.receipt["source_sha256"] or _sha(BASE_SOURCE)!=self.receipt["base_source_sha256"] or
            _sha(Path(__file__))!=self.receipt["wrapper_sha256"] or
            _sha(ROOT/"braids_core/sediment/scalar_diffusion_trace_v98.py")!=self.receipt["core_wrapper_sha256"] or
            topology_token(self.topology)!=self.geometry_sha):
            raise ValueError("v98 diffusion trace CUDA runtime/source/topology identity differs")
        if any(not np.array_equal(self.cp.asnumpy(a),b) for a,b in zip(
            self.device+self.boundary_device,self.host+self.boundary_host)):
            raise ValueError("v98 diffusion trace resident topology differs")

    def launch_resident(self,inputs,trace_inputs,*,dt_s):
        """Pure kernels; caller authenticates geometry, arrays and receipts first."""
        cp=self.cp;top=self.topology;nc,nf,nb=top.cell_count,top.face_count,top.boundary_count
        dic,mass,h,boundary,exchange,active,area,distance=inputs
        tf,th,tc,td,tw,lookup=trace_inputs;nl,nt=mass.shape[0],tf.size
        arrays=(*inputs,*trace_inputs)
        expected=((nc,),(nl,nc),(nc,),(nl,nf),(nf,),(nf,),(nc,),(nf,),
                  (nt,),(nt,),(nl,nt),(nt,),(nt,),(nf,))
        integers={5,8,12,13}
        if any(not isinstance(a,cp.ndarray) or a.device.id!=self.device_id or not a.flags.c_contiguous or
            a.shape!=shape or a.dtype!=(cp.int32 if i in integers else cp.float64)
            for i,(a,shape) in enumerate(zip(arrays,expected))):
            raise ValueError("v98 diffusion trace requires contiguous selected-device resident inputs")
        own,nei,bid,off,adj,sign=self.device;boff,bfaces=self.boundary_device
        effective,amount=cp.empty((nl,nf)),cp.empty((nl,nf))
        active_out,fs=cp.empty((nl,nf),cp.int32),cp.empty((nl,nf),cp.int32)
        theta,ts=cp.empty((nl,nc)),cp.empty((nl,nc),cp.int32)
        result,cs=cp.empty((nl,nc)),cp.empty((nl,nc),cp.int32)
        ledger,bs=cp.empty((nl,nb)),cp.empty((nl,nb),cp.int32)
        internal,internal_status=cp.empty(nl),cp.empty(nl,cp.int32)
        def launch(name,size,args):
            if size:self.kernels[name](((size+127)//128,),(128,),args)
        nci,nfi,nli=np.int32(nc),np.int32(nf),np.int32(nl)
        dry,dt=np.float64(top.profile.diffusion_dry_threshold_m),np.float64(dt_s)
        launch("m2b_diffusion_faces",nl*nf,(nci,nfi,nli,dry,dt,h,mass,boundary,dic,exchange,distance,
            active,own,nei,bid,effective,amount,active_out,fs))
        launch("diffusion_trace_faces_v98",nl*nf,(nci,nfi,nli,np.int32(nt),dry,dt,h,mass,dic,exchange,distance,
            active,own,lookup,th,tc,td,tw,effective,amount,active_out,fs))
        launch("m2b_diffusion_theta",nl*nc,(nci,nfi,nli,mass,amount,area,off,adj,sign,fs,theta,ts))
        launch("m2b_diffusion_limit",nl*nf,(nci,nfi,nli,own,nei,theta,ts,amount,fs))
        launch("m2b_diffusion_update",nl*nc,(nci,nfi,nli,mass,amount,area,off,adj,sign,fs,result,cs))
        launch("m2b_diffusion_ledger",nl*nb,(nfi,nli,np.int32(nb),boff,bfaces,amount,fs,ledger,bs))
        launch("diffusion_trace_ledger_v98",nl,(nfi,nli,lookup,amount,fs,internal,internal_status))
        return dict(mass=result,amount=amount,ledger=ledger,internal=internal,effective=effective,
            face_status=fs,cell_status=cs,theta_status=ts,boundary_status=bs,internal_status=internal_status)

    def evaluate(self,topology,dicuv_cell_m2_s,mass,depth,boundary,exchange_area,face_active,*,dt_s,
        trace_faces,trace_depth,trace_concentration,trace_dicuv,trace_wet_mask,trace_receipt_sha256,
        trace_source_sha256,trace_runtime_sha256,cap_receipt_sha256,trace_cap_receipt_sha256,
        evaluation_index,native_threads=4,cancelled=False):
        cancel=_cancel(cancelled)
        if cancel():raise InterruptedError("v98 diffusion trace cancelled before launch")
        if isinstance(dt_s,bool) or not np.isfinite(dt_s) or dt_s<=0 or topology_token(topology)!=self.geometry_sha:
            raise ValueError("v98 diffusion trace timestep/topology identity differs")
        nc,nf=topology.cell_count,topology.face_count;raw=np.asarray(mass)
        if raw.ndim!=2 or raw.shape[0]<1:raise ValueError("v98 diffusion trace requires combined lanes")
        nl=raw.shape[0]
        dic=_array(dicuv_cell_m2_s,(nc,),"dicuv",nonnegative=True)
        m=_array(mass,(nl,nc),"mass",nonnegative=True);h=_array(depth,(nc,),"depth",nonnegative=True)
        b=_array(boundary,(nl,nf),"boundary",nonnegative=True)
        ex=_array(exchange_area,(nf,),"exchange",nonnegative=True)
        active=_array(face_active,(nf,),"active",integer=True)
        area=_array(topology.cell_area_m2,(nc,),"area");distance=_array(topology.centre_distance_m,(nf,),"distance")
        trace=validate_diffusion_trace_v98(topology,m,h,trace_faces,trace_depth,trace_concentration,trace_dicuv,
            trace_wet_mask,trace_receipt_sha256,trace_source_sha256,trace_runtime_sha256,
            cap_receipt_sha256,trace_cap_receipt_sha256,evaluation_index)
        tf,th,tc,td,tw,lookup,identity=trace
        if np.any((active!=0)&(active!=1)) or np.any(area<=0) or np.any(distance<=0):
            raise ValueError("v98 diffusion trace active mask/geometry differs")
        closed=(np.asarray(topology.neighbour)<0)&(np.asarray(topology.boundary_index)<0)&(lookup<0)
        if np.any(active[closed]):raise ValueError("v98 diffusion trace unowned closed face is active")
        cp=self.cp
        with cp.cuda.Device(self.device_id):
            self._validate_runtime()
            result=self.launch_resident(tuple(cp.asarray(a) for a in (dic,m,h,b,ex,active,area,distance)),
                tuple(cp.asarray(a) for a in (tf,th,tc,td,tw,lookup)),dt_s=dt_s)
            if any(np.any(cp.asnumpy(value)) for key,value in result.items() if key.endswith("status")):
                raise ValueError("v98 diffusion trace CUDA face/limiter/update failed")
            values={key:cp.asnumpy(result[key]) for key in ("mass","amount","ledger","internal","effective")}
        if cancel():raise InterruptedError("v98 diffusion trace cancelled after CUDA launch")
        for value in values.values():value.setflags(write=False)
        return dict(mass_kg_m2=values["mass"],diffusive_face_mass_owner_out_kg=values["amount"],
            diffusive_boundary_ledger_kg=values["ledger"],internal_interface_mass_in_kg=values["internal"],
            effective_diffusivity_m2_s=values["effective"][0],trace_provenance=identity,
            runtime_sha256=self.receipt["receipt_sha256"],actual_cuda=True,automatic_cpu_fallback=False,
            scalar_diffusion_owner_count=1,inventory_limiter_owner_count=1,
            numerical_python_cell_face_loops=0,state_or_clock_publication_count=0,
            physical_boundary_surrogate_count=0,turbulence_cap_evaluation_count=0,production_admission=False)
