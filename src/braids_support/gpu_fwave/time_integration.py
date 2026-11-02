"""Isolated FP64 GPU SSPRK(4,3), not a hydraulic or coupled solver.

The external Euler callback owns spatial preparation, boundaries and fluxes.
It must return device arrays: next (3,nc), ledger (nb,), status (int32, nonzero
means reject) and cfl (float64, nonnegative stage metric). Ledger values are
already integrated over the supplied half-step, not instantaneous rates.
No cell-sized array is transferred to the CPU during step(). The caller may
read returned scalar status/cfl for retry and timestep control. All returned
buffers are reusable and invalidated by the next step() call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OPTIONS=("--std=c++11","--fmad=false","--prec-div=true","--prec-sqrt=true","--ftz=false")
STAGE_TIMES=(0.0,0.5,1.0,0.5)
LEDGER_DIVISORS=(3.0,3.0,3.0,1.0)
SCOPE="GPU SSPRK43 stage combination and hydraulic-only split Manning; external Euler callback, not a full GPU solver"


def validate_controls(nc,nb,grav,dry,chezy_min,speed_limit):
    if not isinstance(nc,(int,np.integer)) or nc<1 or not isinstance(nb,(int,np.integer)) or nb<0:
        raise ValueError("Positive cell count and nonnegative boundary count required")
    values=np.array([grav,dry,chezy_min,speed_limit],dtype=np.float64)
    if not np.isfinite(values).all() or grav<=0 or dry<0 or chezy_min<0 or speed_limit<=0:
        raise ValueError("Invalid hydraulic controls")


def manning_reference(q,roughness,dt,grav=9.81,dry_tolerance=1.e-4,chezy_minimum_depth=0.):
    """Small analytic/CPU oracle only; never called by the GPU stepper."""
    result=np.array(q,dtype=np.float64,copy=True)
    for c in range(result.shape[1]):
        h=result[0,c]
        if h<=dry_tolerance:
            result[1:,c]=0
        else:
            speed=np.sqrt(result[1,c]**2+result[2,c]**2)/h
            denominator=h*chezy_minimum_depth**(1./3.) if 0<h<chezy_minimum_depth else h**(4./3.)
            result[1:,c]/=1.+dt*grav*roughness[c]**2*speed/denominator
    return result


class SSPRK43:
    """Persistent device buffers, transactional accepted state and ledger.

    Use one CUDA stream for this object and its callbacks. A callback may
    prepare/mutate its private input, but must not mutate external accepted
    arrays. Optional prepare_callback mutates a private state in-place and
    may return a device int32 status array; native dry/speed preparation runs
    first. No sediment, turbulence, AMR, secondary flow or dynamic roughness
    is implemented here. The bed is external and unchanged.
    """
    def __init__(self,nc,nb,manning_device,*,grav=9.81,dry_tolerance=1.e-4,
                 chezy_minimum_depth=0.,speed_limit=1.e100,source_path=None):
        import cupy as cp
        validate_controls(nc,nb,grav,dry_tolerance,chezy_minimum_depth,speed_limit)
        self.cp=cp;self.nc=int(nc);self.nb=int(nb)
        self.grav=np.float64(grav);self.dry=np.float64(dry_tolerance)
        self.chezy_min=np.float64(chezy_minimum_depth);self.speed_limit=np.float64(speed_limit)
        self._array(manning_device,"manning",(self.nc,),cp.float64)
        self.manning=manning_device
        from braids_core.windows_paths import filesystem_path
        source=filesystem_path(source_path or HERE/"time_integration.cu").read_text(encoding="utf-8")
        names=("ti_merge_flag","ti_merge_cfl","ti_prepare","ti_capture","ti_combine","ti_ledger",
               "ti_check_ledger","ti_check_ledger_sum","ti_manning","ti_commit_q","ti_commit_ledger")
        self.module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.kernels={name:self.module.get_function(name) for name in names}
        self.base=cp.empty((3,self.nc),dtype=cp.float64)
        self.stage=cp.empty_like(self.base);self.next=cp.empty_like(self.base)
        self.ledger=cp.empty(self.nb,dtype=cp.float64)
        self.cell_status=cp.empty(self.nc,dtype=cp.int32)
        self.ledger_status=cp.empty(self.nb,dtype=cp.int32)
        self.status=cp.empty((),dtype=cp.int32);self.tmp_status=cp.empty_like(self.status)
        self.cfl=cp.empty((),dtype=cp.float64);self.tmp_cfl=cp.empty_like(self.cfl)

    def _array(self,value,name,shape,dtype):
        cp=self.cp
        if not isinstance(value,cp.ndarray) or value.dtype!=dtype or value.shape!=shape or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be a C-contiguous device {dtype} array with shape {shape}")

    def _launch(self,name,size,args):
        if size:self.kernels[name](((size+127)//128,),(128,),args)

    def _merge_status(self,status):
        cp=self.cp
        if not isinstance(status,cp.ndarray) or status.dtype!=cp.int32 or not status.size:
            raise ValueError("Callback status must be a nonempty device int32 array")
        # Normalize arbitrary external codes, including negative INT_MIN.
        # Public status is zero/nonzero, not a preservation of callback codes.
        cp.max(cp.where(status!=0,cp.int32(1),cp.int32(0)),out=self.tmp_status)
        self._launch("ti_merge_flag",1,(self.tmp_status,self.status))

    def _prepare(self,q,callback):
        self._launch("ti_prepare",self.nc,(q,np.int64(self.nc),self.dry,self.speed_limit,self.cell_status))
        self._merge_status(self.cell_status)
        if callback is not None:
            extra=callback(q)
            if extra is not None:self._merge_status(extra)

    def step(self,accepted_q,accepted_ledger,dt,time,euler_callback,prepare_callback=None,commit=True):
        cp=self.cp
        self._array(accepted_q,"accepted_q",(3,self.nc),cp.float64)
        self._array(accepted_ledger,"accepted_ledger",(self.nb,),cp.float64)
        if not np.isfinite([dt,time]).all() or dt<=0:raise ValueError("Finite positive dt and finite time required")
        if accepted_q.data.ptr in (self.base.data.ptr,self.stage.data.ptr,self.next.data.ptr):
            raise ValueError("Accepted state cannot alias reusable trial buffers")
        if self.nb and accepted_ledger.data.ptr==self.ledger.data.ptr:raise ValueError("Ledger cannot alias trial ledger")
        self.status.fill(0);self.cfl.fill(0);self.ledger.fill(0)
        cp.copyto(self.base,accepted_q)
        self._prepare(self.base,prepare_callback)
        if self.nb:
            self._launch("ti_check_ledger",self.nb,(accepted_ledger,np.int64(self.nb),self.ledger_status))
            self._merge_status(self.ledger_status)
        cp.copyto(self.stage,self.base)
        half=np.float64(.5*dt)
        for index,fraction in enumerate(STAGE_TIMES):
            if index:self._prepare(self.stage,prepare_callback)
            result=euler_callback(self.stage,half,float(time+fraction*dt))
            self._array(result["next"],"callback next",(3,self.nc),cp.float64)
            self._array(result["ledger"],"callback ledger",(self.nb,),cp.float64)
            self._merge_status(result["status"])
            metric=result["cfl"]
            if not isinstance(metric,cp.ndarray) or metric.dtype!=cp.float64 or not metric.size:
                raise ValueError("Callback cfl must be a nonempty device float64 array")
            cp.max(cp.where(cp.isfinite(metric)&(metric>=0),metric,cp.float64("nan")),out=self.tmp_cfl)
            self._launch("ti_merge_cfl",1,(self.tmp_cfl,self.cfl,self.status))
            self._launch("ti_capture",self.nc,(result["next"],self.base,self.status,np.int64(self.nc),self.next,self.cell_status))
            self._merge_status(self.cell_status)
            if self.nb:
                self._launch("ti_ledger",self.nb,(result["ledger"],self.ledger,np.int64(self.nb),np.int32(index),self.ledger_status))
                self._merge_status(self.ledger_status)
            if index==2:
                self._launch("ti_combine",3*self.nc,(self.base,self.next,np.int64(self.nc),self.stage))
            elif index<3:cp.copyto(self.stage,self.next)
        self._prepare(self.next,prepare_callback)
        self._launch("ti_manning",self.nc,(self.next,self.manning,np.int64(self.nc),np.float64(dt),
            self.grav,self.dry,self.chezy_min,self.cell_status))
        self._merge_status(self.cell_status)
        if self.nb:
            self._launch("ti_check_ledger_sum",self.nb,(accepted_ledger,self.ledger,np.int64(self.nb),self.ledger_status))
            self._merge_status(self.ledger_status)
        if commit:
            self._launch("ti_commit_q",3*self.nc,(self.next,accepted_q,np.int64(self.nc),self.status))
            self._launch("ti_commit_ledger",self.nb,(self.ledger,accepted_ledger,np.int64(self.nb),self.status))
        return dict(next=self.next,ledger=self.ledger,status=self.status,cfl=self.cfl)


def validate(output):
    """Small GPU correctness suite; no timing and no hydrodynamic speed claim."""
    import cupy as cp
    rows=[]
    def record(name,passed,**values):
        row=dict(case=name,passed=bool(passed),**values);rows.append(row);print(json.dumps(row),flush=True)
    def callback_factory(rhs,nb=1,fail_stage=None,negative_stage=None,nan_stage=None):
        calls=[];next_q=cp.empty((3,1),dtype=cp.float64);ledger=cp.zeros(nb,dtype=cp.float64)
        status=cp.zeros((),dtype=cp.int32);cfl=cp.zeros((),dtype=cp.float64)
        def euler(q,dt,time):
            k=len(calls);calls.append((float(dt),time))
            cp.copyto(next_q,q+dt*rhs(q,time));ledger.fill(dt*(1.+time+time*time))
            status.fill(7 if k==fail_stage else 0);cfl.fill(.1*(k+1))
            if k==negative_stage:next_q[0,0]=-1
            if k==nan_stage:next_q[1,0]=cp.nan
            return dict(next=next_q,ledger=ledger,status=status,cfl=cfl)
        return euler,calls
    # Autonomous stage polynomial R(z)=1+z+z²/2+z³/6+z⁴/48.
    q=cp.asarray([[1.],[.3],[.2]],dtype=cp.float64);ledger=cp.zeros(1,dtype=cp.float64)
    initial=cp.asnumpy(q);stepper=SSPRK43(1,1,cp.zeros(1,dtype=cp.float64))
    callback,calls=callback_factory(lambda q,t:-q)
    result=stepper.step(q,ledger,.2,1.,callback)
    z=-.2;polynomial=1+z+z*z/2+z**3/6+z**4/48
    error=float(np.abs(cp.asnumpy(q)-initial*polynomial).max())
    # Integral from 1 to 1.2 of 1+t+t²; stage times are not monotonic.
    expected_ledger=(1.2+1.2**2/2+1.2**3/3)-(1+1/2+1/3)
    ledger_error=abs(float(ledger.item())-expected_ledger)
    record("stage_polynomial_and_ledger",error<3.e-16 and ledger_error<8.e-16 and int(result["status"].item())==0,
           max_error=error,ledger_error=ledger_error,callback_times=[v[1] for v in calls],cfl=float(result["cfl"].item()))
    errors=[]
    for count in (10,20,40):
        q=cp.asarray([[1.],[0.],[0.]],dtype=cp.float64);ledger=cp.zeros(1,dtype=cp.float64)
        callback,_=callback_factory(lambda q,t:-q)
        for i in range(count):stepper.step(q,ledger,1./count,i/count,callback)
        errors.append(abs(float(q[0,0].item())-np.exp(-1)))
    orders=[float(np.log2(errors[i]/errors[i+1])) for i in range(2)]
    record("third_order_homogeneous_ode",min(orders)>2.9,errors=errors,orders=orders)
    host=np.array([[2.,.005,0.,1.e-4],[3.,.006,.9,.7],[4.,-.008,-.3,.2]])
    rough=np.array([.03,.025,.03,.02]);q=cp.asarray(host);ledger=cp.zeros(0,dtype=cp.float64)
    stepper_m=SSPRK43(4,0,cp.asarray(rough),chezy_minimum_depth=.01)
    def identity(q,dt,t):return dict(next=q,ledger=ledger,status=cp.zeros((),dtype=cp.int32),cfl=cp.zeros((),dtype=cp.float64))
    result=stepper_m.step(q,ledger,3.,0.,identity)
    expected=manning_reference(host,rough,3.,chezy_minimum_depth=.01)
    error=float(np.abs(cp.asnumpy(q)-expected).max())
    record("uniform_split_manning_workdepth_and_dry",error<3.e-15 and int(result["status"].item())==0,max_error=error)
    for key in ("fail_stage","negative_stage","nan_stage"):
        q=cp.asarray(initial);ledger=cp.asarray([2.]);callback,_=callback_factory(lambda q,t:-q,**{key:2})
        result=stepper.step(q,ledger,.2,0.,callback)
        record(key+"_transactional_rejection",int(result["status"].item())!=0 and np.array_equal(cp.asnumpy(q),initial)
               and float(ledger.item())==2.,status=int(result["status"].item()))
    q=cp.asarray([[1.e-5],[8.],[9.]],dtype=cp.float64);original=cp.asnumpy(q);ledger=cp.asarray([2.])
    callback,_=callback_factory(lambda q,t:q*0,fail_stage=1)
    result=stepper.step(q,ledger,.2,0.,callback)
    record("initial_dry_preparation_private_on_rejection",np.array_equal(cp.asnumpy(q),original) and float(ledger.item())==2.)
    q=cp.asarray(initial);ledger=cp.asarray([2.]);callback,_=callback_factory(lambda q,t:-q)
    result=stepper.step(q,ledger,.2,0.,callback,commit=False)
    record("uncommitted_success_keeps_accepted_arrays",int(result["status"].item())==0 and np.array_equal(cp.asnumpy(q),initial)
           and float(ledger.item())==2.)
    # Reusing an Euler output buffer must not lose earlier ledger increments,
    # and even extreme external error codes must reject the complete trial.
    for name,flag,metric,stage_flux,accepted_flux in (
        ("negative_int_min_callback_status",np.iinfo(np.int32).min,0.,1.,2.),
        ("nonfinite_stage_cfl",0,np.nan,1.,2.),
        ("negative_stage_cfl",0,-.1,1.,2.),
        ("nonfinite_boundary_ledger",0,0.,np.nan,2.),
        ("accepted_ledger_overflow",0,0.,8.e307,8.e307)):
        q=cp.asarray(initial);ledger=cp.asarray([accepted_flux])
        def rejected(q,dt,t):
            return dict(next=q,ledger=cp.full(1,stage_flux,dtype=cp.float64),
                        status=cp.asarray(flag,dtype=cp.int32),cfl=cp.asarray(metric,dtype=cp.float64))
        result=stepper.step(q,ledger,.2,0.,rejected)
        record(name,int(result["status"].item())!=0 and np.array_equal(cp.asnumpy(q),initial)
               and float(ledger.item())==accepted_flux)
    reference=HERE/"runs/gpu_resident_update_20261001_v2/runtime_export/cutcell_solver.f90"
    payload=dict(all_passed=all(row["passed"] for row in rows),rows=rows,scope=SCOPE,cuda_options=OPTIONS,
                 cuda_source_sha256=hashlib.sha256((HERE/"time_integration.cu").read_bytes()).hexdigest(),
                 driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),precision="float64",timing_claim=None)
    if reference.exists():
        payload["frozen_fortran_contract"]=dict(path=str(reference),sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            routines=["research_ssp43","prepare_hydraulic_stage","source"],
            qualification="Analytic GPU stage/source tests; not a new full Fortran trajectory comparison")
    destination=Path(output);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8") as stream:json.dump(payload,stream,indent=2)
    if not payload["all_passed"]:raise RuntimeError("GPU time-integration qualification failed")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("validate",))
    parser.add_argument("--output",default=str(HERE/"reports/time_integration_20261001_v2/braids_support.json"))
    args=parser.parse_args();validate(args.output)
