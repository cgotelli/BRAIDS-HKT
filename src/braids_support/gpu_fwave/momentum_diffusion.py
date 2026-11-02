"""Resident v35 velocity diffusion with exact native recovery geometry.

Fixed-grid mode1/geometry1 only. A changed wet/perched aperture triggers a GPU
geometry rebuild, with the same native quadratic/affine recovery and rank fallback.
There is no CPU numerical fallback. Native energy limiter and SSPRK2 subcycling
are retained. State remains on GPU; only control scalars cross. The operator's
wet/dry parity is not a qualification of full coupled wet/dry river evolution.
"""
from pathlib import Path
import numpy as np
from .resident_update import OPTIONS


class DiffusionGeometryChanged(NotImplementedError):
    """The old cache cannot represent the current wet/perched geometry."""


def validate_scope(data):
    if data['horizontal_viscosity']<=0:raise ValueError('Positive viscosity required for diffusion operator')
    if data['horizontal_viscosity_mode']!=1 or data['horizontal_viscosity_geometry']!=1:
        raise ValueError('GPU viscosity supports native mode1 geometry1 only')
    if data.get('amr_enabled',0):raise ValueError('GPU viscosity requires fixed geometry')
    required=('conductance','face_diffusivity','face_bound','rate','recovery_inverse','recovery_affine_inverse',
        'corrected_face','recovery_slot','recovery_cells','recovery_order','recovery_valid','recovery_stencils',
        'recovery_stencil_counts','maximum_rate')
    for key in required:
        if 'diffusion_'+key not in data:raise ValueError('Missing native diffusion cache '+key)
    nf=len(data['owner']);nr=len(data['diffusion_recovery_cells']);nc=len(data['area'])
    if data['diffusion_recovery_inverse'].shape!=(5,5,nr) or data['diffusion_recovery_affine_inverse'].shape!=(2,2,nr):
        raise ValueError('Wrong diffusion inverse shape')
    if np.any((data['diffusion_recovery_cells']<0)|(data['diffusion_recovery_cells']>=nc)):
        raise ValueError('Invalid recovery cells')
    for k in range(nr):
        count=int(data['diffusion_recovery_stencil_counts'][k]);stencil=data['diffusion_recovery_stencils'][:count,k]
        if count<0 or count>256 or np.any((stencil<0)|(stencil>=nc)):raise ValueError('Invalid recovery stencil')
    if data['diffusion_conductance'].shape!=(nf,):raise ValueError('Wrong conductance shape')


class ResidentMomentumDiffusion:
    def __init__(self,data,source_path=None,*,execution='graph',dynamic_geometry=True):
        import cupy as cp
        validate_scope(data);self.cp=cp;self.nc=len(data['area']);self.nf=len(data['owner'])
        self.nr=len(data['diffusion_recovery_cells']);self.dry=float(data['dry']);self.nu=float(data['horizontal_viscosity'])
        self.maximum_rate=float(data['diffusion_maximum_rate']);self.data=data
        self.a={k:cp.asarray(data[k]) for k in ('area','xy','mid','normal','length','owner','neighbour','offset','faces','signs')}
        for name,value in data.items():
            if name.startswith('diffusion_') and isinstance(value,np.ndarray):self.a[name[10:]]=cp.asarray(value)
        if execution not in ('graph','ordinary'):raise ValueError('Diffusion execution must be graph or ordinary')
        self.execution=execution
        self.allow_dynamic_geometry=bool(dynamic_geometry);self.dynamic_geometry=None;self.geometry_updates=0
        self.source_path=Path(source_path or Path(__file__).with_suffix('.cu'))
        names=('md_mask','md_velocity','md_recover','md_flux','md_factor','md_gather','md_energy',
            'md_gather_dynamic','md_trial_begin','md_trial_operator_check','md_trial_energy_check')
        module=cp.RawModule(code=self.source_path.read_text(),options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names};self.module=module
        self.working=cp.empty((2,self.nc));self.initial=cp.empty_like(self.working);self.stage=cp.empty_like(self.working)
        self.candidate=cp.empty_like(self.working);self.velocity=cp.empty_like(self.working)
        self.polynomial=cp.empty((2,5,self.nr));self.primary=cp.empty((2,self.nf));self.correction=cp.empty_like(self.primary)
        self.d0=cp.empty(self.nf);self.dc=cp.empty(self.nf);self.energy=cp.empty(self.nc)
        self.d0sum=cp.empty(());self.dcsum=cp.empty(());self.factor=cp.empty(())
        self.face_status=cp.empty(self.nf,dtype=cp.int32);self.cell_status=cp.empty(self.nc,dtype=cp.int32)
        self.recovery_status=cp.empty(self.nr,dtype=cp.int32);self.scalar_status=cp.empty((),dtype=cp.int32)
        self.status=cp.zeros((),dtype=cp.int32);self.last_result={}
        self.substeps=0;self.minimum_correction=1.;self.rejected_energy_trials=0
        self.graph=None;self.graph_pointers=None;self.graph_references=None;self.graph_captures=0
        self.graph_pool=cp.cuda.MemoryPool();self.trial_dt=cp.empty(())
        self.trial_summary=cp.empty(4);self.reduced_status=cp.zeros(4,dtype=cp.int32)
        self.energy_before=cp.empty(());self.energy_after=cp.empty(())

    def trial_metadata(self):
        """Snapshot host bindings that an accepted-array transaction cannot copy.

        A first wet-geometry rebuild replaces cache allocations and may enlarge
        the recovery workspace. Keep the original objects alive so rejection
        can restore their bindings before the external byte snapshot is copied
        back. Immutable host input data and CUDA modules are not copied.
        """
        names=('nr','maximum_rate','polynomial','recovery_status','dynamic_geometry',
            'geometry_updates','substeps','minimum_correction','rejected_energy_trials',
            'graph','graph_pointers','graph_references','graph_captures')
        geometry=self.dynamic_geometry
        return dict(owner=self,arrays=dict(self.a),attributes={name:getattr(self,name) for name in names},
            last_result=dict(self.last_result),geometry_state=None if geometry is None else
            dict(ready=geometry.ready,rebuilds=geometry.rebuilds))

    def restore_trial_metadata(self,saved):
        """Restore bindings/scalars only; the caller then restores array bytes."""
        if saved.get('owner') is not self:raise ValueError('Diffusion trial metadata belongs to another operator')
        self.a.clear();self.a.update(saved['arrays'])
        for name,value in saved['attributes'].items():setattr(self,name,value)
        self.last_result=dict(saved['last_result'])
        if self.dynamic_geometry is not None:
            self.dynamic_geometry.ready=saved['geometry_state']['ready']
            self.dynamic_geometry.rebuilds=saved['geometry_state']['rebuilds']

    def _status_reduction(self,index):
        # An ephemeral view retains the same reduction pointer but does not
        # introduce overlapping persistent storage into complete transactions.
        return self.reduced_status[index:index+1].reshape(())

    def launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def _energy(self,momentum,depth):
        self.launch('md_energy',self.nc,(momentum,depth,self.a['area'],np.int64(self.nc),np.float64(self.dry),self.energy))
        return float(self.cp.sum(self.energy).item())

    def _energy_device(self,momentum,depth,out):
        self.launch('md_energy',self.nc,(momentum,depth,self.a['area'],np.int64(self.nc),np.float64(self.dry),self.energy))
        self.cp.sum(self.energy,out=out)

    def _enqueue_trial(self,q,bed):
        """Entire conservative SSPRK2 trial; no host reads, including failures."""
        cp=self.cp;a=self.a;nc,nf,nr=map(np.int64,(self.nc,self.nf,self.nr));dry=np.float64(self.dry);depth=q[0]
        self.launch('md_mask',self.nf,(q,bed,a['xy'],a['normal'],a['length'],a['owner'],a['neighbour'],
            a['conductance'],nc,nf,dry,np.float64(self.nu),self.face_status))
        cp.max(self.face_status,out=self._status_reduction(0))
        self.launch('md_trial_begin',1,(self.reduced_status[0:1],self.trial_summary))
        cp.copyto(self.initial,self.working);cp.copyto(self.stage,self.initial)
        for _ in range(2):
            self.launch('md_velocity',self.nc,(self.stage,depth,nc,dry,self.velocity,self.cell_status))
            self.launch('md_recover',self.nr,(self.velocity,a['xy'],a['area'],a['recovery_cells'],a['recovery_order'],a['recovery_valid'],
                a['recovery_stencils'],a['recovery_stencil_counts'],a['recovery_inverse'],a['recovery_affine_inverse'],nc,nr,self.polynomial,self.recovery_status))
            self.launch('md_flux',self.nf,(self.velocity,a['area'],a['xy'],a['mid'],a['normal'],a['owner'],a['neighbour'],
                a['conductance'],a['face_diffusivity'],a['corrected_face'],a['recovery_slot'],a['recovery_order'],a['recovery_valid'],
                self.polynomial,nc,nf,nr,self.primary,self.correction,self.d0,self.dc,self.face_status))
            cp.sum(self.d0,out=self.d0sum);cp.sum(self.dc,out=self.dcsum)
            self.launch('md_factor',1,(self.d0sum,self.dcsum,self.factor,self.scalar_status))
            cp.max(self.cell_status,out=self._status_reduction(0));cp.max(self.face_status,out=self._status_reduction(1))
            if self.nr:cp.max(self.recovery_status,out=self._status_reduction(2))
            else:self.reduced_status[2:3].fill(0)
            self.launch('md_trial_operator_check',1,(self.reduced_status[0:1],self.reduced_status[1:2],self.reduced_status[2:3],
                self.scalar_status,self.factor,self.trial_summary))
            self.launch('md_gather_dynamic',self.nc,(self.stage,depth,a['area'],self.primary,self.correction,self.factor,
                a['offset'],a['faces'],a['signs'],nc,nf,dry,self.trial_dt,self.candidate))
            self._energy_device(self.stage,depth,self.energy_before);self._energy_device(self.candidate,depth,self.energy_after)
            self.launch('md_trial_energy_check',1,(self.energy_before,self.energy_after,self.trial_summary))
            cp.copyto(self.stage,self.candidate)
        cp.add(self.initial,self.stage,out=self.candidate);self.candidate*=.5
        self._energy_device(self.initial,depth,self.energy_before);self._energy_device(self.candidate,depth,self.energy_after)
        self.launch('md_trial_energy_check',1,(self.energy_before,self.energy_after,self.trial_summary))

    def _graph_trial(self,q,bed,ds):
        cp=self.cp;pointers=(q.data.ptr,bed.data.ptr)
        if self.graph is None or pointers!=self.graph_pointers:
            cp.cuda.get_current_stream().synchronize();stream=cp.cuda.Stream(non_blocking=True)
            with stream,cp.cuda.using_allocator(self.graph_pool.malloc):
                self.trial_dt.fill(ds)
                for _ in range(2):self._enqueue_trial(q,bed)
            stream.synchronize()
            try:
                with stream,cp.cuda.using_allocator(self.graph_pool.malloc):
                    stream.begin_capture();self._enqueue_trial(q,bed);self.graph=stream.end_capture()
            except Exception:
                if stream.is_capturing():stream.end_capture()
                raise
            self.graph.upload(stream=stream);stream.synchronize()
            self.graph_pointers=pointers;self.graph_references=(q,bed);self.graph_captures+=1
        self.trial_dt.fill(ds);self.graph.launch(stream=cp.cuda.get_current_stream())
        return self.trial_summary.get()

    def _validate_state(self,q,bed,dt):
        cp=self.cp
        if not isinstance(q,cp.ndarray) or q.dtype!=cp.float64 or q.shape!=(3,self.nc) or not q.flags.c_contiguous:
            raise ValueError('Diffusion requires resident FP64 hydraulic3xnc state')
        if not isinstance(bed,cp.ndarray) or bed.dtype!=cp.float64 or bed.shape!=(self.nc,) or not bed.flags.c_contiguous:
            raise ValueError('Diffusion requires current resident bed')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Finite positive diffusion dt required')
        if not np.isfinite(self.maximum_rate) or dt*self.maximum_rate>1e8:raise ValueError('Excessive diffusion step')

    def apply(self,q,bed,dt):
        operation=self.apply_ordinary if self.execution=='ordinary' else self.apply_graph
        try:result=operation(q,bed,dt)
        except DiffusionGeometryChanged:
            if not self.allow_dynamic_geometry:raise
            if self.dynamic_geometry is None:
                self.dynamic_geometry=ResidentDiffusionGeometry(self.data,source_path=self.source_path.with_name('momentum_geometry.cu'))
            self.dynamic_geometry.update(q,bed);self.dynamic_geometry.install(self);self.geometry_updates+=1
            result=operation(q,bed,dt)
        result['geometry_updates']=self.geometry_updates
        result['geometry_rebuilds']=0 if self.dynamic_geometry is None else self.dynamic_geometry.rebuilds
        return result

    def apply_graph(self,q,bed,dt):
        self._validate_state(q,bed,dt);cp=self.cp
        cp.copyto(self.working,q[1:]);remaining=float(dt);base=min(dt,.9/self.maximum_rate) if self.maximum_rate>0 else dt
        count=0;minfactor=1.;rejects=0
        while remaining>0:
            ds=min(remaining,base);attempt=0
            while True:
                attempt+=1
                if attempt>60 or ds<=np.finfo(float).eps*max(dt,np.finfo(float).tiny):raise RuntimeError('Diffusion energy step reached machine precision')
                mask,bad,energy,factor=self._graph_trial(q,bed,ds)
                if mask:raise DiffusionGeometryChanged('Wet/perched mask differs from cached diffusion geometry')
                if bad:raise RuntimeError('Invalid resident viscosity operator')
                minfactor=min(minfactor,float(factor))
                if not energy:break
                ds*=.5;rejects+=1
            cp.copyto(self.working,self.candidate);remaining=max(0.,remaining-ds)
            if remaining<=64*np.finfo(float).eps*max(dt,np.finfo(float).tiny):remaining=0.
            count+=1
        cp.copyto(q[1:],self.working);self.substeps+=count;self.minimum_correction=min(self.minimum_correction,minfactor);self.rejected_energy_trials+=rejects
        self.last_result=dict(substeps=count,minimum_correction=minfactor,energy_retries=rejects,fields_resident=True,
            execution='graph',graph_captures=self.graph_captures,host_trial_reads=count+rejects)
        return self.last_result

    def apply_ordinary(self,q,bed,dt):
        cp=self.cp;a=self.a;nc,nf,nr=map(np.int64,(self.nc,self.nf,self.nr));dry=np.float64(self.dry)
        if not isinstance(q,cp.ndarray) or q.dtype!=cp.float64 or q.shape!=(3,self.nc) or not q.flags.c_contiguous:
            raise ValueError('Diffusion requires resident FP64 hydraulic3xnc state')
        if not isinstance(bed,cp.ndarray) or bed.dtype!=cp.float64 or bed.shape!=(self.nc,) or not bed.flags.c_contiguous:
            raise ValueError('Diffusion requires current resident bed')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Finite positive diffusion dt required')
        self.launch('md_mask',self.nf,(q,bed,a['xy'],a['normal'],a['length'],a['owner'],a['neighbour'],
            a['conductance'],nc,nf,dry,np.float64(self.nu),self.face_status))
        if int(cp.max(self.face_status).item()):raise DiffusionGeometryChanged('Wet/perched mask differs from cached diffusion geometry')
        if not np.isfinite(self.maximum_rate) or dt*self.maximum_rate>1e8:raise ValueError('Excessive diffusion step')
        cp.copyto(self.working,q[1:]);depth=q[0];remaining=float(dt);base=min(dt,.9/self.maximum_rate) if self.maximum_rate>0 else dt
        count=0;minfactor=1.;rejects=0
        while remaining>0:
            ds=min(remaining,base);attempt=0
            while True:
                attempt+=1
                if attempt>60 or ds<=np.finfo(float).eps*max(dt,np.finfo(float).tiny):raise RuntimeError('Diffusion energy step reached machine precision')
                cp.copyto(self.initial,self.working);cp.copyto(self.stage,self.initial);accepted=True
                for stage_index in range(2):
                    self.launch('md_velocity',self.nc,(self.stage,depth,nc,dry,self.velocity,self.cell_status))
                    self.launch('md_recover',self.nr,(self.velocity,a['xy'],a['area'],a['recovery_cells'],a['recovery_order'],a['recovery_valid'],
                        a['recovery_stencils'],a['recovery_stencil_counts'],a['recovery_inverse'],a['recovery_affine_inverse'],nc,nr,self.polynomial,self.recovery_status))
                    self.launch('md_flux',self.nf,(self.velocity,a['area'],a['xy'],a['mid'],a['normal'],a['owner'],a['neighbour'],
                        a['conductance'],a['face_diffusivity'],a['corrected_face'],a['recovery_slot'],a['recovery_order'],a['recovery_valid'],
                        self.polynomial,nc,nf,nr,self.primary,self.correction,self.d0,self.dc,self.face_status))
                    cp.sum(self.d0,out=self.d0sum);cp.sum(self.dc,out=self.dcsum)
                    self.launch('md_factor',1,(self.d0sum,self.dcsum,self.factor,self.scalar_status))
                    bad=int(cp.max(self.cell_status).item())|int(cp.max(self.face_status).item())|int(self.scalar_status.item())
                    if self.nr:bad|=int(cp.max(self.recovery_status).item())
                    if bad:raise RuntimeError('Invalid resident viscosity operator')
                    minfactor=min(minfactor,float(self.factor.item()))
                    self.launch('md_gather',self.nc,(self.stage,depth,a['area'],self.primary,self.correction,self.factor,
                        a['offset'],a['faces'],a['signs'],nc,nf,dry,np.float64(ds),self.candidate))
                    before=self._energy(self.stage,depth);after=self._energy(self.candidate,depth)
                    if not np.isfinite([before,after]).all() or after>before+128*np.finfo(float).eps*max(before,np.finfo(float).tiny):accepted=False;break
                    cp.copyto(self.stage,self.candidate)
                if accepted:
                    cp.add(self.initial,self.stage,out=self.candidate);self.candidate*=.5
                    before=self._energy(self.initial,depth);after=self._energy(self.candidate,depth)
                    accepted=bool(np.isfinite([before,after]).all() and after<=before+128*np.finfo(float).eps*max(before,np.finfo(float).tiny))
                if accepted:break
                ds*=.5;rejects+=1
            cp.copyto(self.working,self.candidate);remaining=max(0.,remaining-ds)
            if remaining<=64*np.finfo(float).eps*max(dt,np.finfo(float).tiny):remaining=0.
            count+=1
        cp.copyto(q[1:],self.working);self.substeps+=count;self.minimum_correction=min(self.minimum_correction,minfactor);self.rejected_energy_trials+=rejects
        self.last_result=dict(substeps=count,minimum_correction=minfactor,energy_retries=rejects,fields_resident=True)
        return self.last_result

    def step(self,q,bed,dt):
        self.apply(q,bed,dt);self.status.fill(0)
        return self.status


# Pin the implementation when the solver module is imported. Allocation remains
# lazy, but later working-tree edits cannot change a running solver's rebuild
# implementation at its first wet/dry transition.
from .momentum_geometry import ResidentDiffusionGeometry
