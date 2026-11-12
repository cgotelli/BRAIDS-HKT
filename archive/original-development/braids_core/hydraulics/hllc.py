"""Shared second-order hydraulic prototype; no public admission implied.

Unsplit MUSCL-Hancock, HLLC/HLLE-dry flux, hydrostatic reconstruction and
cell-referenced pressure quadrature on actual polygon areas/faces.
"""
from pathlib import Path
import numpy as np
from braids_core.sediment.bend_methods_v1 import BendTrialRejected, digest
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='hllc_muscl_hancock_polygon_v1'
NAMES=('hllc_reconstruct','hllc_predict_source','hllc_faces','hllc_budget','hllc_limit','hllc_update')


class HLLCHydraulics:
    def __init__(self,mesh,*,backend,threads=4,gravity=9.81,dry=1e-6,cfl=.9,
                 exterior=None,friction=None,check=None,integration='hancock',momentum=None,face_correction=None,advanced=None,boundary_correction=None):
        if backend not in ('cpu','gpu'):raise ValueError('Explicit CPU or GPU backend required')
        if not 0<cfl<=1:raise ValueError('Hydraulic face-sum CFL must be in (0,1]')
        if integration not in ('hancock','ssprk22'):raise ValueError('Unknown HLLC integration')
        if advanced is not None and integration!='hancock':raise ValueError('Advanced coupling requires the qualified Hancock clock')
        self.backend=backend;self.xp=np if backend=='cpu' else __import__('cupy');xp=self.xp
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.n=len(mesh.cells);self.nf=len(mesh.faces);n=self.n;nf=self.nf
        xy=np.array([c.centroid_xy for c in mesh.cells]).T.copy()
        mid=np.array([(np.array(f.start_xy)+f.end_xy)/2 for f in mesh.faces]).T.copy()
        normal=np.array([f.outward_normal for f in mesh.faces]).T.copy()
        owner=np.array([f.owner for f in mesh.faces],np.int32);neighbour=np.array([f.neighbour for f in mesh.faces],np.int32)
        lengths=np.array([f.length_m for f in mesh.faces]);area=np.array([c.area_m2 for c in mesh.cells])
        rows=[[] for _ in range(n)]
        for f,(i,j) in enumerate(zip(owner,neighbour)):
            rows[i].append((f,1))
            if j>=0:rows[j].append((f,-1))
        off=np.r_[0,np.cumsum([len(r) for r in rows])].astype(np.int32)
        faces=np.array([f for row in rows for f,s in row],np.int32);signs=np.array([s for row in rows for f,s in row],np.int32)
        inverse=np.zeros((3,n))
        for i,row in enumerate(rows):
            matrix=np.zeros((2,2))
            for f,s in row:
                j=neighbour[f] if s>0 else owner[f]
                if j<0 and mesh.faces[f].kind in ('inlet','outlet'):continue
                delta=xy[:,j]-xy[:,i] if j>=0 else 2*np.dot(mid[:,f]-xy[:,i],normal[:,f])*normal[:,f]
                norm=np.dot(delta,delta)
                if norm>0:matrix+=np.outer(delta,delta)/norm
            inv=np.linalg.pinv(matrix,rcond=1e-12);inverse[:,i]=[inv[0,0],inv[0,1],inv[1,1]]
        self.g={k:xp.asarray(v) for k,v in dict(xy=xy,mid=mid,normal=normal,owner=owner,
                    neighbour=neighbour,length=lengths,area=area,off=off,faces=faces,signs=signs,inverse=inverse).items()}
        self.selected=xp.asarray(np.array([int(f.kind in ('inlet','outlet')) for f in mesh.faces],np.int32))
        if exterior is None and bool(np.any(np.array([f.kind in ('inlet','outlet') for f in mesh.faces]))):
            raise ValueError('Open boundaries require an explicit physical flux provider')
        self.external=xp.zeros((3,nf));self.external_trace=xp.zeros((3,nf))
        self.exterior=exterior;self.friction=friction;self.check=check
        self.integration=integration;self.momentum=momentum;self.face_correction=face_correction;self.advanced=advanced;self.boundary_correction=boundary_correction;self.momentum_rate=xp.zeros((3,n))
        self.zero_water=xp.zeros(nf)
        self.v=xp.zeros((4,n));self.grad=xp.zeros((8,n));self.half=xp.zeros((4,n));self.qhalf=xp.zeros((3,n))
        self.predict_status=xp.zeros(n,dtype=xp.int32);self.face_status=xp.zeros(nf,dtype=xp.int32)
        self.fl=xp.zeros((3,nf));self.fr=xp.zeros((3,nf));self.speed=xp.zeros(nf)
        self.theta=xp.ones(n);self.frequency=xp.zeros(n);self.next=xp.zeros((3,n));self.status=xp.zeros(n)
        self.gravity=float(gravity);self.dry=float(dry);self.cfl=float(cfl);self.step_hint=None;self.evaluations=0
        self.identity=dict(method=METHOD if integration=='hancock' else 'hllc_muscl_ssprk22_polygon_v1',wrapper_sha256=digest(__file__),runtime=self.engine.identity,
            order='piecewise-linear limited reconstruction; '+integration,
            pressure='hydrostatic interface reconstruction; cell-reference pressure quadrature',
            positivity='stock-feasible exact-discharge inlet allocation; reserve other exterior debit; conservative internal-face donor limiter; no film or depth clipping',
            riemann='HLLC; HLLE for dry faces or inadmissible contact states',
            cfl='dt max_cell(sum_face(length*signal_speed)/actual_area)',max_cfl=1.,
            public_support=False)

    def state(self):return dict(step_hint=self.step_hint)

    def restore(self,state):
        if set(state)!={'step_hint'}:raise ValueError('HLLC clock state differs')
        hint=state['step_hint']
        if hint is not None and (not np.isfinite(hint) or hint<=0):raise ValueError('Invalid HLLC step hint')
        self.step_hint=hint

    def _spatial_step(self,q,z,dt,time,*,predictor=.5,drag=True):
        xp=self.xp;e=self.engine;g=self.g;n=self.n;nf=self.nf
        if (not isinstance(q,xp.ndarray) or not isinstance(z,xp.ndarray) or
            q.shape!=(3,n) or z.shape!=(n,) or q.dtype!=np.float64 or z.dtype!=np.float64 or
            not q.flags.c_contiguous or not z.flags.c_contiguous):
            raise ValueError('Contiguous float64 hydraulic fields on the selected backend required')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite hydraulic interval required')
        self.frequency_peak=0.
        self.evaluations+=1
        if self.check is not None:self.check()
        work=q.copy()
        if drag and self.friction is not None:self.friction(work,.5*dt)
        if self.advanced is not None:self.advanced.start(work)
        e.launch('hllc_reconstruct',n,n,nf,self.gravity,self.dry,float(2*predictor*dt),work,z,
                 g['xy'],g['mid'],g['normal'],g['owner'],g['neighbour'],self.selected,
                 g['off'],g['faces'],g['inverse'],self.grad,self.half,self.qhalf,self.predict_status)
        if predictor and self.momentum is not None:
            self.momentum_rate.fill(0.);self.momentum(work,self.zero_water,self.momentum_rate)
            e.launch('hllc_predict_source',n,n,self.dry,float(dt),self.momentum_rate,self.half,self.qhalf)
        if self.exterior is not None:
            boundary=self.exterior(self.qhalf,z,time+predictor*dt)
            if not isinstance(boundary,tuple) or len(boundary)!=2:
                raise ValueError('Boundary provider must return physical flux and hydraulic trace for CFL control')
            if self.boundary_correction is not None:self.boundary_correction(work,dt,boundary[0],boundary[1])
            self.external[...]=boundary[0];self.external_trace[...]=boundary[1]
        e.launch('hllc_faces',nf,n,nf,self.gravity,self.dry,self.half,self.grad,g['xy'],g['mid'],g['normal'],g['owner'],g['neighbour'],self.selected,self.external,self.external_trace,self.fl,self.fr,self.speed)
        if self.face_correction is not None:self.face_correction(self)
        e.launch('hllc_budget',n,n,nf,float(dt),work,g['area'],g['length'],g['off'],g['faces'],g['signs'],self.fl,self.speed,self.selected,self.predict_status,self.theta,self.frequency)
        if self.advanced is not None:self.advanced.add_frequency(work,self.frequency,self.qhalf)
        frequency=float(xp.max(self.frequency));self.frequency_peak=frequency
        if not np.isfinite(frequency) or dt*frequency>self.cfl*(1+1e-12):raise BendTrialRejected('HLLC stage CFL exceeds selected limit')
        e.launch('hllc_limit',nf,n,nf,self.dry,work,self.theta,g['owner'],g['neighbour'],self.selected,self.fl,self.fr,self.face_status)
        e.launch('hllc_update',n,n,nf,float(dt),self.dry,work,g['area'],g['length'],g['off'],g['faces'],g['signs'],self.fl,self.fr,self.face_status,self.next,self.status)
        if bool(xp.any(self.status)):raise BendTrialRejected('HLLC endpoint/boundary positivity rejected')
        result=self.next.copy()
        if self.advanced is not None:self.advanced.half(work,result,self.qhalf,self.fl[0],dt)
        if self.momentum is not None:
            self.momentum_rate.fill(0.);self.momentum(self.qhalf if predictor else work,self.fl[0],self.momentum_rate)
            result+=dt*self.momentum_rate
            if not bool(xp.all(xp.isfinite(result))):raise BendTrialRejected('HLLC momentum source rejected')
        if drag and self.friction is not None:self.friction(result,.5*dt)
        if self.advanced is not None:self.advanced.end(work,result,self.fl[0],dt)
        return result,self.fl[0].copy(),frequency

    def single(self,q,z,dt,time):
        if self.integration=='hancock':return self._spatial_step(q,z,dt,time)
        work=q.copy()
        if self.friction is not None:self.friction(work,.5*dt)
        q1,w1,f1=self._spatial_step(work,z,dt,time,predictor=0.,drag=False)
        q2,w2,f2=self._spatial_step(q1,z,dt,time+dt,predictor=0.,drag=False)
        result=.5*(work+q2)
        if self.friction is not None:self.friction(result,.5*dt)
        self.frequency_peak=max(f1,f2)
        return result,.5*(w1+w2),max(f1,f2)

    def advance(self,q,z,dt,time=0.):
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite hydraulic interval required')
        xp=self.xp;elapsed=0.;hint=self.step_hint or dt;water=xp.zeros(self.nf);steps=[];retries=0;rejected=[];freqs=[];evaluations=self.evaluations
        if self.advanced is not None:self.advanced.begin()
        while not steps or elapsed<dt-1e-13*max(1.,dt):
            trial=min(hint,dt-elapsed);truncated=trial<hint*(1-1e-12)
            try:result,flux,frequency=self.single(q,z,trial,time+elapsed)
            except BendTrialRejected as exc:
                rejected.append(dict(time=time+elapsed,dt=trial,reason=str(exc),kind='cfl' if 'CFL' in str(exc) else 'spatial_or_positivity'))
                retries+=1;hint=min(.5*trial,.98*self.cfl/self.frequency_peak) if self.frequency_peak>0 else .5*trial
                if retries>100 or hint<dt*2**-20:raise
                continue
            q=result;water+=trial*flux;elapsed+=trial;steps.append(trial);freqs.append(trial*frequency)
            growth=hint if truncated else 2*trial
            hint=min(growth,.98*self.cfl/frequency) if frequency>0 else growth
        self.step_hint=float(hint)
        return q,water/dt,dict(method=self.identity['method'],hydraulic_substeps=len(steps),hydraulic_retries=retries,
                     minimum_dt_s=min(steps),maximum_dt_s=max(steps),next_dt_s=hint,rejections=rejected,
                     spatial_evaluations=self.evaluations-evaluations,accepted_cfl_max=max(freqs))
