"""Optional fixed-dt CUDA-graph replay of the unchanged resident SSPRK43 step.

Research only. Constant supported boundary controls, immutable fixed geometry,
hydraulic-only physics and outlet_extension=0 are required. All four stages,
Manning and transactional commit/rejection remain the existing kernels. Each
replay reads only status and CFL scalars; no state field transfers occur.

Capture uses an exclusive retained memory pool so temporary reduction pointers
cannot be recycled by unrelated CuPy allocations while the graph exists. The
pipeline must not be used concurrently or have its device geometry/control
arrays modified after capture. Construct a new graph for another dt/config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

HERE=Path(__file__).resolve().parent
TAG="gpu_graph_execution_20261001_v2"
SOURCES=("https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html",
         "https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Graph.html")


def validate_graph_scope(data,dt,outlet_extension=0):
    if not np.isfinite(dt) or dt<=0:
        raise ValueError("CUDA graph requires finite fixed positive dt")
    if outlet_extension!=0:
        raise ValueError("CUDA graph SSP43 requires outlet_extension=0")
    if not np.isin(np.asarray(data["mode"]),[2,3,6,8]).all():
        raise ValueError("CUDA graph requires supported constant boundary modes")
    for key in ("nhydro_series","nseries","nfraction_series","nmorfac","section_control","sampled"):
        if np.any(data.get(key,0)):
            raise ValueError("CUDA graph excludes time-dependent/unported control "+key)
    forcing=np.asarray(data["forcing"])
    if forcing.ndim!=2 or forcing.shape[1]!=len(data["mode"]) or forcing.shape[0]<2 or not np.isfinite(forcing).all():
        raise ValueError("CUDA graph requires finite constant forcing")


def validate_replay_request(fixed_dt,dt,fixed_commit,commit,time_value):
    if dt is not None and float(dt)!=fixed_dt:
        raise ValueError("CUDA graph dt is fixed; construct another graph")
    if commit is not None and bool(commit)!=fixed_commit:
        raise ValueError("CUDA graph commit policy is fixed")
    if time_value is not None and not np.isfinite(time_value):
        raise ValueError("CUDA graph time must be finite")


class FixedStepGraph:
    """Replay a complete, already-qualified constant-control GPU SSP43 step."""
    def __init__(self,pipeline,dt,*,commit=True,warmups=2):
        import cupy as cp
        validate_graph_scope(pipeline.data,dt,pipeline.reconstruction.outlet_extension)
        if not isinstance(warmups,int) or warmups<1:raise ValueError("Warm before capture")
        self.cp=cp;self.pipeline=pipeline;self.dt=float(dt);self.commit=bool(commit)
        self.stream=cp.cuda.Stream(non_blocking=True)
        self.pool=cp.cuda.MemoryPool()
        self.replays=0
        # Existing arrays were created on the caller's stream. Establish their
        # availability once; replay then owns its nonblocking stream.
        cp.cuda.get_current_stream().synchronize()
        started=time.perf_counter()
        counter=pipeline.euler_calls
        with self.stream,cp.cuda.using_allocator(self.pool.malloc):
            for _ in range(warmups):pipeline.step(self.dt,0.,commit=False)
        self.stream.synchronize()
        try:
            with self.stream,cp.cuda.using_allocator(self.pool.malloc):
                self.stream.begin_capture()
                self.result=pipeline.step(self.dt,0.,commit=self.commit)
                self.graph=self.stream.end_capture()
        except Exception:
            if self.stream.is_capturing():
                try:self.stream.end_capture()
                except Exception:pass
            raise
        finally:
            pipeline.euler_calls=counter
        self.graph.upload(stream=self.stream)
        self.stream.synchronize()
        self.setup_seconds=time.perf_counter()-started
        self._identity=self._configuration_identity()
        self._pointers=(pipeline.q.data.ptr,pipeline.ledger.data.ptr)

    def _configuration_identity(self):
        p=self.pipeline
        return (p.reconstruction.outlet_extension,
            tuple(float(p.reconstruction.controls[k]) for k in ("dry","grav","dx")),
            tuple(float(p.flux.controls[k]) for k in ("dt","grav","dry","cfl_target")),
            tuple(float(getattr(p.boundary,k)) for k in ("dry","grav","speed","chezy")),
            tuple(float(getattr(p.stepper,k)) for k in ("grav","dry","chezy_min","speed_limit")))

    def step(self,dt=None,time=None,*,commit=None):
        """Launch one step and read exactly two scalar controls.

        Time may advance because supported controls are autonomous. The fixed
        captured dt cannot change. Rejected trials leave accepted state/ledger
        unchanged through the original device-side conditional commit kernels.
        """
        validate_replay_request(self.dt,dt,self.commit,commit,time)
        p=self.pipeline
        if (p.q.data.ptr,p.ledger.data.ptr)!=self._pointers:
            raise ValueError("CUDA graph accepted array pointers changed")
        if self._configuration_identity()!=self._identity:
            raise ValueError("CUDA graph scalar controls changed; capture again")
        with self.stream:
            self.graph.launch(stream=self.stream)
            code=int(self.result["status"].item())
            peak=float(self.result["cfl"].item())
        self.replays+=1;p.euler_calls+=4
        return dict(self.result,status_code=code,cfl_value=peak)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:json.dump(value,stream,indent=2)


def _read_plain(pipeline,dt,current,commit):
    value=pipeline.step(dt,current,commit=commit)
    return int(value["status"].item()),float(value["cfl"].item())


def _measure(pipeline,graph,dt,*,batches=5,steps=10):
    """Repeated noncommitting trials, with two scalar reads in BOTH modes."""
    cp=pipeline.cp;rows=[]
    for batch in range(batches):
        # Alternate order to reduce simple warming/order bias. These are two
        # launch strategies on the same GPU, not a CPU/GPU comparison.
        for mode in (("plain","graph") if batch%2==0 else ("graph","plain")):
            stream=graph.stream if mode=="graph" else cp.cuda.Stream.null
            start,end=cp.cuda.Event(),cp.cuda.Event()
            wall=time.perf_counter()
            with stream:
                start.record()
                for _ in range(steps):
                    if mode=="graph":
                        result=graph.step();status=result["status_code"]
                    else:status,_=_read_plain(pipeline,dt,0.,False)
                    if status:raise RuntimeError("Timing trial rejected")
                end.record()
            end.synchronize()
            rows.append(dict(batch=batch,mode=mode,steps=steps,
                wall_ms_per_step=1000.*(time.perf_counter()-wall)/steps,
                event_ms_per_step=float(cp.cuda.get_elapsed_time(start,end))/steps))
    summary={mode:{metric:float(np.median([r[metric] for r in rows if r["mode"]==mode]))
                   for metric in ("wall_ms_per_step","event_ms_per_step")} for mode in ("plain","graph")}
    summary["wall_ratio_plain_over_graph"]=summary["plain"]["wall_ms_per_step"]/summary["graph"]["wall_ms_per_step"]
    return dict(samples=rows,medians=summary,small_scalar_reads_per_step=2,
        same_fixed_state_noncommitting_trials=True,includes_jit=False,includes_capture=False,
        includes_output_transfers=False,concurrent_cpu_run_expected=True,cpu_speed_comparison=False)


def validate(tag=TAG,pipeline_tag="gpu_hydraulic_20261001_v6",oracle_tag="gpu_reconstruction_20261001_v1"):
    from .hydraulic_pipeline import HydraulicPipeline,verify_frozen,load_snapshot
    for value in (tag,pipeline_tag,oracle_tag):
        if Path(value).name!=value or value in (".",".."):raise ValueError("Single directory tags required")
    root,report=HERE/"runs"/tag,HERE/"reports"/tag
    root.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),root/"graph_execution.py")
    source=HERE/"runs"/pipeline_tag/"source";verify_frozen(source)
    identity=dict(driver_sha256=_sha(__file__),pipeline_manifest_sha256=_sha(source/"pipeline_build.json"),
        current_hydraulic_driver_sha256=_sha(HERE/"hydraulic_pipeline.py"),pipeline_tag=pipeline_tag,
        scope="Fixed-dt, constant-control complete hydraulic SSP43 graph replay; no coupled physics",
        docs=list(SOURCES),precision="FP64",production_modified=False,delft_modified=False,
        concurrent_cpu_run_expected=True,cpu_comparison=False)
    _write(root/"build.json",identity)
    rows=[]
    for case in ("bend_small","bend_large"):
        data=load_snapshot(HERE/"runs"/oracle_tag/case)
        plain=HydraulicPipeline(data,source,outlet_extension=0)
        captured=HydraulicPipeline(data,source,outlet_extension=0)
        # A conservative fixed dt is used for correctness and timing, not a new
        # claim about the largest stable physical timestep.
        dt=min(2.e-4,2.*float(data["dt"]))
        graph=FixedStepGraph(captured,dt)
        cp=plain.cp;maximum=0.;passed=True;statuses=[]
        for index in range(10):
            code,cfl=_read_plain(plain,dt,index*dt,True)
            result=graph.step(time=index*dt)
            q1,l1=plain.fields();q2,l2=captured.fields()
            maximum=max(maximum,float(np.max(np.abs(q1-q2))))
            equal=np.array_equal(q1,q2) and np.array_equal(l1,l2)
            passed &= equal and code==0 and result["status_code"]==0 and cfl==result["cfl_value"]
            statuses.append(dict(index=index,state_and_ledger_bitwise=equal,
                plain_status=code,graph_status=result["status_code"],cfl_equal=cfl==result["cfl_value"]))
        row=dict(case=case,cells=plain.nc,faces=plain.nf,dt=dt,steps=10,passed=bool(passed),
            max_state_difference=maximum,checks=statuses,capture_setup_s=graph.setup_seconds,
            retained_temporary_pool_bytes=graph.pool.total_bytes())
        # Separately capture a noncommitting graph for repeatable timings.
        trial=HydraulicPipeline(data,source,outlet_extension=0)
        trial_graph=FixedStepGraph(trial,dt,commit=False)
        row["timing"]=_measure(trial,trial_graph,dt)
        # A deliberately unacceptable trial must preserve accepted arrays.
        reject=HydraulicPipeline(data,source,outlet_extension=0)
        reject_graph=FixedStepGraph(reject,10.)
        before_q,before_l=reject.fields();out=reject_graph.step();after_q,after_l=reject.fields()
        row["rejection"]=dict(status=out["status_code"],cfl=out["cfl_value"],
            unchanged=bool(np.array_equal(before_q,after_q) and np.array_equal(before_l,after_l)))
        row["passed"] &= out["status_code"]!=0 and row["rejection"]["unchanged"]
        rows.append(row);print(json.dumps(row),flush=True)
    record=dict(identity=identity,rows=rows,all_passed=all(r["passed"] for r in rows))
    _write(report/"braids_support.json",record)
    with (report/"REPORT.txt").open("x",encoding="utf-8") as stream:
        stream.write("GPU FIXED-STEP CUDA GRAPH REPLAY\n\n")
        stream.write("All complete-step comparisons passed: "+str(record["all_passed"])+"\n")
        stream.write("Ten evolving SSP43 steps; both accepted state and boundary ledger checked every step.\n")
        stream.write("Two scalar reads per replay. Rejection preserves accepted state and ledger.\n")
        stream.write("Timings are same-GPU launch-strategy comparisons with background CPU activity, not CPU/GPU speedups.\n")
        for row in rows:
            med=row["timing"]["medians"]
            stream.write(f"\n{row['case']} ({row['cells']} cells): bitwise={all(c['state_and_ledger_bitwise'] for c in row['checks'])}\n")
            stream.write(f"Plain {med['plain']['wall_ms_per_step']:.4f} ms; graph {med['graph']['wall_ms_per_step']:.4f} ms; launch-strategy ratio {med['wall_ratio_plain_over_graph']:.3f}x\n")
    if not record["all_passed"]:raise RuntimeError("Graph execution validation failed")
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag",default=TAG);parser.add_argument("--pipeline-tag",default="gpu_hydraulic_20261001_v6")
    parser.add_argument("--oracle-tag",default="gpu_reconstruction_20261001_v1")
    args=parser.parse_args();os.environ.setdefault("CUPY_CACHE_DIR",str(HERE/".cupy_cache"))
    validate(args.tag,args.pipeline_tag,args.oracle_tag)
