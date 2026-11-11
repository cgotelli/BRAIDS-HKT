"""Private GPU sampling workspace; geometry is immutable, fields are fresh.

One block owns each deterministic rotor reduction. Cell-owned source gathering
preserves rotor order even when disjoint physical supports touch the same CV.
No device field is evaluated on CPU; only the existing scalar controls are.
"""
from pathlib import Path
import math
import numpy as np
from .bend_methods_v1 import digest


class RotorSampling:
    def __init__(self,rotors):
        import cupy as cp
        from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine
        self.xp=cp;self.rotors=rotors;self.nr=len(rotors);self.n=len(rotors[0].area)
        path=Path(__file__).with_suffix('.h');names=('rotor_stats','rotor_force','rotor_apply','rotor_scale','rotor_moments')
        self.engine=WaveEngine('gpu',4,source=path,names=names)
        def host(x):return cp.asnumpy(x)
        off=np.r_[0,np.cumsum([len(r.ids) for r in rotors])].astype(np.int32)
        ids=np.concatenate([host(r.ids) for r in rotors]).astype(np.int32);w=np.concatenate([host(r.normalized[r.ids]) for r in rotors])
        poff=np.r_[0,np.cumsum([len(r.plane) for r in rotors])].astype(np.int32)
        plane=np.concatenate([host(r.plane) for r in rotors]);delta=np.concatenate([host(r.delta) for r in rotors],axis=1)
        unique=np.unique(ids);rows=[np.flatnonzero(ids==i) for i in unique]
        self.off=cp.asarray(off);self.ids=cp.asarray(ids);self.w=cp.asarray(w);self.normal=cp.asarray([r.normal for r in rotors])
        self.dry=cp.asarray([r.dry for r in rotors]);self.area=rotors[0].area;self.poff=cp.asarray(poff);self.plane=cp.asarray(plane);self.delta=cp.asarray(delta)
        self.cells=cp.asarray(unique);self.cell_off=cp.asarray(np.r_[0,np.cumsum([len(r) for r in rows])].astype(np.int32));self.support=cp.asarray(np.concatenate(rows).astype(np.int32))
        self.ns=len(ids);self.bounds=off;self.uv=cp.empty((3,self.ns));self.k=cp.empty(self.ns)
        self.stats=cp.empty((self.nr,6));self.forces=cp.empty((self.nr,2));self.scale=cp.empty(self.nr);self.factor=cp.empty(self.nr)
        self.identity=dict(method='hkt.gpu_batched_sparse_sampling_v1',engine=self.engine.identity,wrapper_sha256=digest(__file__),
            state='recompute at every stage; deterministic block reductions and rotor-ordered cell source; no lagged fields or atomic sums')

    def _fields(self,q,z,grad):
        cp=self.xp
        if any(not isinstance(x,cp.ndarray) or x.dtype!=np.float64 or not x.flags.c_contiguous for x in [q,z,grad]) or q.shape!=(3,self.n) or z.shape!=(self.n,) or grad.ndim!=2 or grad.shape[1]!=self.n or grad.shape[0]<8:
            raise ValueError('Contiguous float64 turbine fields on the selected GPU required')

    def sample(self,q,z,grad,*,neutral=False):
        self._fields(q,z,grad);e=self.engine;nr=self.nr
        e.launch('rotor_stats',nr*128,self.n,nr,q,z,grad,self.off,self.ids,self.w,self.normal,self.dry,self.poff,self.plane,self.delta,self.uv,self.stats)
        self.host_stats=self.stats.get();states=[];scales=[]
        for r,stats in zip(self.rotors,self.host_stats):
            config=dict(r.c,ct=.5,cp=0.,control=dict(mode='constant_ct')) if neutral else None
            s,scale,power=r._operating_state(stats,config);states.append((s,power));scales.append(scale)
        self.scale.set(np.asarray(scales,dtype=np.float64))
        e.launch('rotor_force',nr*128,nr,self.off,self.ids,self.w,self.area,self.normal,self.scale,self.uv,self.k,self.forces)
        forces=self.forces.get();result=[]
        for j,(r,(s,power),force) in enumerate(zip(self.rotors,states,forces)):
            a,b=self.bounds[j:j+2];result.append((r._finish(s,float(force[0]),float(force[1]),power),self.k[a:b],self.uv[0,a:b],self.uv[1,a:b]))
        return result

    def apply(self,samples,rate):
        self.engine.launch('rotor_apply',len(self.cells),len(self.cells),self.ns,self.n,self.cells,self.cell_off,self.support,self.k,self.uv,rate)

    def scale_sources(self,factors):
        self.factor.set(np.asarray(factors,dtype=np.float64));self.engine.launch('rotor_scale',self.nr*128,self.nr,self.off,self.factor,self.k)

    def prepare_moments(self,geometry,parents,order):
        cp=self.xp;nodes=[];bary=[];weights=[];ey=[];ez=[];poff=[0];self.parents=parents;self.parent_rows=[]
        for j,r in enumerate(self.rotors):
            nd,ba,y,z,w=geometry[j];nodes.append(nd);bary.append(ba);weights.append(w)
            for i,xD,lateral,sigma in parents[j]:
                self.parent_rows.append((j,i,sigma));ey.append(cp.exp(-.5*((y+lateral)/sigma)**2));ez.append(cp.exp(-.5*((z+r.c['hub_elevation_m']-self.rotors[i].c['hub_elevation_m'])/sigma)**2))
            poff.append(len(ey))
        self.nodes=cp.stack(nodes);self.bary=cp.stack(bary);self.qw=cp.stack(weights);self.nq=len(weights[0]);self.parent_off=cp.asarray(poff,dtype=cp.int32)
        self.ey=cp.stack(ey) if ey else cp.empty((0,self.nq));self.ez=cp.stack(ez) if ez else cp.empty((0,self.nq))
        self.parent_params=cp.zeros((2,len(ey)));self.amp=self.parent_params[0];self.mean=self.parent_params[1]
        self.offsets=cp.empty((self.nr,self.nq));self.moments=cp.empty((self.nr,9));self._selections={}
        self.layers=[];done=set();pending=set(order)
        while pending:
            layer=sorted(j for j in pending if all(p[0] in done for p in parents[j]));self.layers.append(layer);done.update(layer);pending.difference_update(layer)
            if not layer:raise ValueError('Cyclic wake dependence is unsupported.')

    def moment_rows(self,q,samples,selected):
        from .river_turbine_inflow import _mean
        amp=[];mean=[];counts=[0]*self.nr
        for j,i,sigma in self.parent_rows:
            r=self.rotors[j];up=self.rotors[i];us=samples[i][0];s=samples[j][0]
            if j in selected and us['active']:
                H=s['mean_depth_m'];hub=up.c['hub_elevation_m']-self.host_stats[j,5]
                if not 0<hub<H:raise ValueError('Wake axis outside receiving water column')
                rad=1-us['thrust_coefficient']/(8*(sigma/up.c['D'])**2)
                if rad<=0:raise ValueError('Gaussian wake outside real momentum domain')
                amp.append(us['upstream_velocity_m_s']*(1-math.sqrt(rad)));mean.append(_mean(H,hub,sigma));counts[j]+=1
            else:amp.append(0.);mean.append(0.)
        if any(self.parents[j] for j in selected):self.parent_params.set(np.asarray([amp,mean],dtype=np.float64))
        key=tuple(selected)
        if key not in self._selections:
            if len(self._selections)>=32:self._selections.pop(next(iter(self._selections)))
            self._selections[key]=self.xp.asarray(selected,dtype=self.xp.int32)
        ids=self._selections[key]
        self.engine.launch('rotor_moments',len(selected)*128,self.n,self.nr,len(selected),self.nq,ids,q,self.nodes,self.bary,self.qw,self.normal,self.dry,self.stats,self.parent_off,self.amp,self.mean,self.ey,self.ez,self.offsets,self.moments)
        rows=self.moments.get()
        for j in selected:
            row=rows[j]
            if row[7]:raise ValueError('Rotor quadrature spans dry cells; refine or relocate the turbine.')
            if row[4] or row[5] or abs(row[6]-1)>1e-10:raise ValueError('Finite offsets and normalized nonnegative disk weights required.')
        return rows,counts
