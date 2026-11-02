"""One captured four-stage transport graph for the explicit coupled GPU solver.

Only timestep argument transport changes: nine kernels read the resident full
step and use half of it, exactly as native SSP43. Their numerical bodies remain
unchanged. Both secondary PCG/source solves, Manning, diffusion and the complete
transactional acceptance gate remain outside this graph. Backend choice is the
user's, never an automatic mesh-size policy.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
from .adaptive_hydraulics import transform_dt_arguments,_DynamicArgument
from .graph_execution import validate_graph_scope
from .time_integration import OPTIONS


DT_KERNELS={
    'resident_update.cu':{'resident_draining':(10,.5),'resident_gather_update':(14,.5)},
    'boundaries.cu':{'gpu_boundary_ledger':(5,.5)},
    'exner_update.cu':{'bed_donors':(11,.5),'bed_gather':(14,.5),'bed_ledger':(7,.5)},
    'secondary_transport.cu':{'st_fct_bounds':(18,.5),'st_fct_update':(11,.5),'st_ledger':(5,.5)},
}


class CoupledTransportGraph:
    """Install exact four-Euler replay on a privately owned coupled solver.

    Construct once before running, retain this object for the run, and do not
    externally replace its geometry/control arrays. Graph capture uses a held
    exclusive allocation pool; replay uses the caller's current CUDA stream so
    source/transport/diffusion ordering has no cross-stream race. Warmup never
    commits accepted fields or ledgers. No per-step field transfer is introduced.
    """
    def __init__(self,solver,*,warm_dt=1.e-5):
        import cupy as cp
        if getattr(solver,'has_suspended',False):
            raise ValueError('Suspended transport is not qualified for adaptive CUDA graph replay')
        validate_graph_scope(solver.data,warm_dt,solver.reconstruction.outlet_extension)
        if getattr(solver.stepper,'transport_graph',None) is not None:
            raise ValueError('A coupled transport graph is already installed')
        self.cp=cp;self.solver=solver;self.stepper=solver.stepper
        self.plain_transport=self.stepper.transport
        self.parameter=cp.asarray([warm_dt],dtype=cp.float64)
        self.pool=cp.cuda.MemoryPool();self.stream=cp.cuda.Stream(non_blocking=True)
        self.modules=[];self.generated_sha256={};self.original_sha256={};self.replays=0
        self.transformed_kernel_count=0;self.inactive_secondary_dt_kernels=[]
        self.original_kernels=[]
        folder=Path(solver.folder)
        for filename,description in DT_KERNELS.items():
            if filename=='resident_update.cu':
                target=solver.flux.kernels;key=lambda name:name
            elif filename=='boundaries.cu':target=solver.boundary.kernels;key=lambda name:name
            elif filename=='exner_update.cu':target=solver.exner.k;key=lambda name:name
            else:target=solver.secondary.kernels;key=lambda name:name.removeprefix('st_')
            present=[name for name in description if key(name) in target]
            # OFF/equilibrium have no transported hI, hence no secondary FCT or
            # secondary boundary ledger. Partial presence is a broken ABI.
            if len(present)!=len(description):
                if filename!='secondary_transport.cu' or present:
                    raise ValueError('Incomplete coupled graph dt-kernel set: '+filename)
                self.inactive_secondary_dt_kernels=list(description)
                continue
            original=(folder/filename).read_text()
            code=transform_dt_arguments(original,description)
            self.original_sha256[filename]=hashlib.sha256(original.encode()).hexdigest()
            self.generated_sha256[filename]=hashlib.sha256(code.encode()).hexdigest()
            if filename=='resident_update.cu':
                code=(folder/'augmented_fwave.cu').read_text()+'\n'+code
            module=cp.RawModule(code=code,options=OPTIONS,name_expressions=tuple(description))
            for name,(position,_) in description.items():
                self.original_kernels.append((target,key(name),target[key(name)]))
                target[key(name)]=_DynamicArgument(module.get_function(name),position,self.parameter)
            self.modules.append(module)
            self.transformed_kernel_count+=len(description)
        cp.cuda.get_current_stream().synchronize();started=time.perf_counter();calls=solver.euler_calls
        try:
            with self.stream,cp.cuda.using_allocator(self.pool.malloc):
                for _ in range(2):solver.step(warm_dt,0.,commit=False)
            self.stream.synchronize()
            with self.stream,cp.cuda.using_allocator(self.pool.malloc):
                self.stream.begin_capture()
                self.plain_transport(warm_dt,0.,solver.euler,solver.sediment.rebalance)
                self.graph=self.stream.end_capture()
            self.graph.upload(stream=self.stream);self.stream.synchronize()
        except Exception:
            if self.stream.is_capturing():
                try:self.stream.end_capture()
                except Exception:pass
            for target,name,original in self.original_kernels:target[name]=original
            raise
        finally:solver.euler_calls=calls
        self.setup_s=time.perf_counter()-started
        self.identities=self._identities()
        self.controls=self._controls()
        self.stepper.transport=self.replay
        self.stepper.transport_graph=self

    def _identities(self):
        p=self.solver;t=self.stepper
        # Geometry, forcing and reusable temporary arrays are immutable by
        # contract. These state/ledger buffers are the externally exposed ones.
        return tuple((k,v.data.ptr,v.shape,v.dtype.str) for owner in
            (p.state,p.ledger,t.base,t.stage,t.next,t.ledger) for k,v in owner.items())

    def _controls(self):
        p=self.solver
        return (float(p.data['morfac']),tuple(float(p.flux.controls[k]) for k in ('grav','dry','cfl_target')),
            tuple(float(p.reconstruction.controls[k]) for k in ('dry','grav','dx')),
            tuple(float(getattr(p.boundary,k)) for k in ('dry','grav','speed','chezy')),
            tuple(float(getattr(p.secondary,k)) for k in ('grav','dry','espir','beta')),
            tuple(float(getattr(self.stepper.hydro,k)) for k in ('dry','speed_limit')),
            p.reconstruction.outlet_extension)

    def replay(self,dt,current,euler,rebalance):
        if not np.isfinite([dt,current]).all() or dt<=0:
            raise ValueError('Finite positive coupled graph timestep required')
        if euler!=self.solver.euler or rebalance!=self.solver.sediment.rebalance:
            raise ValueError('Captured coupled callbacks cannot change')
        if self._identities()!=self.identities or self._controls()!=self.controls:
            raise ValueError('Captured coupled storage or scalar controls changed')
        stream=self.cp.cuda.get_current_stream()
        self.parameter.fill(float(dt));self.graph.launch(stream=stream)
        self.replays+=1;self.solver.euler_calls+=4

    def receipt(self):
        return dict(captures=1,replays=self.replays,setup_s=self.setup_s,
            transformed_kernel_count=self.transformed_kernel_count,
            inactive_secondary_dt_kernels=self.inactive_secondary_dt_kernels,
            source_sha256=self.original_sha256,generated_sha256=self.generated_sha256,
            transform='Only scalar double dt -> resident full step, multiplied by 0.5; mathematical bodies unchanged',
            source_diffusion_outside_graph=True,backend_automatic_selection=False)
