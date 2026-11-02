"""Named research coupling, fixed fully-wet mesh, one grain, MF=1.

Uses the maintained augmented-f-wave residual and its accepted SSP stage flux.
Does NOT invoke the carrier's capacity-Exner owner. No public registration,
secondary flow, collapse, restart or AMR admission follows from this module.
"""
from __future__ import annotations
from pathlib import Path
from types import MethodType
import hashlib
import numpy as np
from .bend_methods_v1 import BendTrialRejected, digest
from .jiang_cao_exchange import JiangCaoExchange
from .river_explicit import RiverExplicit
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='coupled.jc2026_p15_fixed_wet_ssprk22_v2'
NAMES=('jc_gradient','jc_capacity_gradient','jc_capacity_average','jc_capacity_state','jc_manning','jc_density_source','jc_relative',
       'jc_mobile_faces','jc_mobile_divergence','jc_mobile_bound','jc_mobile_limit')


class MobileGeometry:
    """Immutable actual-centroid/face geometry, computed once, backend resident."""
    def __init__(self,engine,xp,xy,mid,area,owner,neighbour,length,normal,kind):
        xy,mid=np.asarray(xy,float),np.asarray(mid,float)
        n,nf=xy.shape[1],len(owner)
        from .jiang_cao_exchange import InventoryMesh
        class Proxy:pass
        proxy=Proxy();proxy.xp=xp
        graph=InventoryMesh(proxy,area,owner,neighbour,length,kind)
        if xy.shape!=(2,n) or mid.shape!=(2,nf) or np.shape(normal)!=(2,nf):
            raise ValueError('Cartesian cell centroids and face midpoints/normals required')
        if not np.isfinite(xy).all() or not np.isfinite(mid).all() or not np.isfinite(normal).all():
            raise ValueError('Finite Cartesian geometry required')
        if not np.allclose(np.linalg.norm(normal,axis=0),1.,rtol=0,atol=1e-10):
            raise ValueError('Unit normals required')
        self.engine,self.xp,self.n,self.nf=engine,xp,n,nf
        self.g=graph._g;self.geometry_sha256=hashlib.sha256((graph.geometry_sha256).encode()+
            b''.join(np.ascontiguousarray(v,dtype=np.float64).tobytes() for v in (xy,mid,normal))).hexdigest()
        # One weighted LS stencil per cell. Exterior faces supply no fictitious
        # zero concentrations and do not suppress tangential wall gradients.
        lists=[[] for _ in range(n)]
        for f,(o,j) in enumerate(zip(owner,neighbour)):
            if j>=0:lists[o].append((int(j),f));lists[j].append((int(o),f))
        off=[0];adj=[];faces=[];weights=[];dist=[]
        for i,items in enumerate(lists):
            d=np.array([xy[:,j]-xy[:,i] for j,f in items]).reshape(-1,2)
            if not len(d):raise ValueError('Each cell needs an internal neighbour')
            r2=np.sum(d*d,axis=1)
            if np.any(r2<=0):raise ValueError('Distinct cell centroids required')
            weighted=d/r2[:,None];inv=np.linalg.pinv(d.T@weighted,rcond=1e-13)
            w=weighted@inv
            weights.extend(w);adj.extend(j for j,f in items);faces.extend(f for j,f in items)
            dist.extend(mid[:,f]-xy[:,i] for j,f in items);off.append(len(adj))
        self.ls={k:xp.asarray(v,dtype=xp.int32 if k in ('offset','adj','face') else xp.float64)
                 for k,v in dict(offset=off,adj=adj,face=faces,weights=weights,delta=dist).items()}
        self.g.update(xy=xp.asarray(xy).copy(),mid=xp.asarray(mid).copy(),normal=xp.asarray(normal).copy())

    def gradient(self,value,limited=False):
        a=self.ls;out=self.xp.empty((2,self.n))
        self.engine.launch('jc_gradient',self.n,self.n,int(limited),a['offset'],a['adj'],
                           a['weights'],a['delta'],value,out)
        return out

    def capacity_gradient(self,value,q):
        """Affine-exact bed gradient with an odd/even-visible flow derivative."""
        xp=self.xp
        v=xp.ascontiguousarray(xp.asarray(value,dtype=xp.float64))
        state=xp.ascontiguousarray(xp.asarray(q,dtype=xp.float64))
        if v.shape!=(self.n,) or state.shape!=(3,self.n):
            raise ValueError('Capacity gradient requires scalar bed and q(3,n)')
        if not bool(xp.all(xp.isfinite(v))) or not bool(xp.all(xp.isfinite(state))):
            raise ValueError('Finite bed and hydraulic state required')
        base=self.gradient(v);out=xp.empty_like(base);a=self.ls
        self.engine.launch('jc_capacity_gradient',self.n,self.n,a['offset'],a['adj'],
            self.g['xy'],v,state,base,out)
        return out

    def capacity_state(self,q):
        """Affine-exact flow-face state used only by sediment capacity.

        The hydraulic state itself is not filtered.  Two compact shared-face
        passes act first along and then across the local flow direction.
        Reconstructed values are translated back to the owner centroid with
        their own dynamic-location LS gradient.  Uniform and affine states are
        exact; grid-scale modes in either local direction are not fed directly
        to the capacity/exchange source.
        """
        xp=self.xp;state=xp.ascontiguousarray(xp.asarray(q,dtype=xp.float64))
        if state.shape!=(3,self.n) or not bool(xp.all(xp.isfinite(state))):
            raise ValueError('Finite q(3,n) required for capacity state')
        a=self.ls;g=self.g
        def compact(value,transverse):
            average=xp.empty_like(state);location=xp.empty((2,self.n));out=xp.empty_like(state)
            self.engine.launch('jc_capacity_average',self.n,self.n,self.nf,int(transverse),
                a['offset'],a['adj'],a['face'],g['length'],g['normal'],g['xy'],
                state,value,average,location)
            self.engine.launch('jc_capacity_state',self.n,self.n,a['offset'],a['adj'],g['xy'],
                average,location,value,out)
            return out
        return compact(compact(state,False),True)


class JiangCaoCoupled:
    def __init__(self,hydraulics,material,*,mobile_solid,bed_solid,manning,boundary_kind):
        a=hydraulics;p=a.physics;xp=a.xp
        if not isinstance(a.midpoint,RiverExplicit) or a.time or a.steps or p._open:
            raise ValueError('Fresh shared explicit f-wave hydraulic owner required')
        if a.controls.morfac!=1. or a.controls.betac or a.controls.viscosity or a.controls.diffusivity:
            raise ValueError('This qualification requires MF1; secondary flow and viscosity off')
        cp=material.capacity_material
        if (p.parameters.gravity!=cp.gravity_m_s2 or p.parameters.dry!=cp.dry_depth_m
            or p.parameters.diameter!=cp.d50_m):raise ValueError('Hydraulic/material identities differ')
        self.a,self.xp,self.backend=a,xp,a.backend
        self.exchange=JiangCaoExchange(a.backend,material,threads=p.flow.threads)
        self.engine=WaveEngine(a.backend,p.flow.threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        host=a.host;g=p.g
        self.mesh=MobileGeometry(self.engine,xp,host(g['xy']),a.data['mid'],host(g['area']),
            host(g['owner']),host(g['neighbour']),host(g['length']),host(g['normal']),boundary_kind)
        self.n,self.nf=p.n,p.nf;self.B=1-material.porosity
        self.manning=xp.ascontiguousarray(xp.broadcast_to(xp.asarray(manning,dtype=xp.float64),(p.n,))).copy()
        if bool(xp.any(~xp.isfinite(self.manning)|(self.manning<=0))):raise ValueError('Positive finite Manning required')
        self.state=xp.vstack((a.q,xp.broadcast_to(xp.asarray(mobile_solid),(p.n,)),
                             xp.broadcast_to(xp.asarray(bed_solid),(p.n,)),a.z)).astype(xp.float64)
        self._validate(self.state)
        outside=host(g['neighbour'])<0;wall=host(g['bid'])<0
        k=np.asarray(boundary_kind)
        if np.any((k==1)!=(outside&wall)) or np.any((k>=2)!=(outside&~wall)):
            raise ValueError('Sediment boundary partition must match hydraulic walls/open faces')
        from .river_simple_physics import install
        install(a,'hydrodynamics')
        def friction(owner,q,dt,*args):
            self.engine.launch('jc_manning',p.n,p.n,float(dt),cp.gravity_m_s2,self.manning,q)
        p.friction=MethodType(friction,p)
        a.midpoint.secondary.active=False
        self.time=0.;self.steps=0;self.solid_integral=xp.zeros(p.nf);self.water_integral=xp.zeros(p.nf)
        self.initial_solid,self.initial_water=self.totals(self.state)
        self.last={}
        self.identity=dict(method=METHOD,public_support=False,backend=a.backend,
            hydraulic=a.identity,exchange=self.exchange.identity,runtime=self.engine.identity,
            wrapper_sha256=digest(__file__),geometry=self.mesh.geometry_sha256,
            initial_state_sha256=hashlib.sha256(host(self.state).tobytes()).hexdigest(),
            manning_sha256=hashlib.sha256(host(self.manning).tobytes()).hexdigest(),
            splitting='Strang exchange/friction; SSPRK22 hydraulic+mobile+density source',
            exchange_target='midpoint predictor ce; fresh gradient and capacity at each source evaluation',
            capacity_gradient='aligned one-sided shared-face derivative; forward-preferred/backward-at-corner; affine-exact with centered transverse LS',
            capacity_state='two-pass along/across-flow shared-face reconstruction; affine-exact; sediment forcing only',
            transport='limited linear concentration at true faces; accepted mixture flux plus relative slope drift',
            roughness='prescribed Manning; same value for hydraulic drag and capacity; no bedforms',
            boundary='prescribed incoming concentration; export-only outlet; impermeable walls',
            limits='fully wet fixed mesh; one grain; MF1; no secondary/collapse/restart/AMR')

    def _validate(self,s):
        xp=self.xp;dry=self.exchange.material.capacity_material.dry_depth_m
        if not bool(xp.all(xp.isfinite(s))) or bool(xp.any((s[0]<=dry)|(s[3]<0)|(s[4]<0)|(s[3]>=self.B*s[0]))):
            raise BendTrialRejected(f'Jiang-Cao fixed-wet phase bounds violated: min h/m/b={float(xp.min(s[0])):g}/{float(xp.min(s[3])):g}/{float(xp.min(s[4])):g}; reduce timestep or use an admitted wet/dry route')

    def totals(self,s):
        area=self.mesh.g['area'];p=self.exchange.material.porosity;xp=self.xp
        return float(xp.sum((s[3]+s[4])*area)),float(xp.sum((s[0]-s[3]+p*s[4]/self.B)*area))

    def target(self,s):
        q=self.mesh.capacity_state(s[:3])
        return self.exchange.capacity_target(s,self.mesh.capacity_gradient(s[5],s[:3]),self.manning,
                                             capacity_q=q)

    def source(self,s,dt):
        ce,_=self.target(s)
        predictor=self.exchange.exchange(s,ce,.5*dt).state
        midpoint_ce,_=self.target(predictor)
        return self.exchange.exchange(s,midpoint_ce,dt).state.copy()

    def relative_discharge(self,s,cap):
        delta=self.xp.empty((2,self.n))
        self.engine.launch('jc_relative',self.n,self.n,s[:3],cap.values,delta)
        return delta

    def momentum_correction(self,s,rate):
        """Named secondary bundles override this; the v1 base has none."""
        return

    def rate(self,s,dt,time,inlet_concentration):
        self._validate(s);a=self.a;xp=self.xp;g=self.mesh.g;n=self.n;nf=self.nf
        a.check();a.z[...]=s[5];a.physics.z0[...]=s[5]
        a.midpoint.interval_base=s[:3]
        r,Q=a.midpoint.rate(s[:3],dt,time);Q=Q*g['length']
        c=s[3]/s[0];grad=self.mesh.gradient(c,limited=True)
        cp=self.exchange.material.capacity_material
        self.engine.launch('jc_density_source',n,n,cp.gravity_m_s2,cp.water_density_kg_m3,
                           cp.solid_density_kg_m3,s[:3],c,grad,r)
        _,cap=self.target(s);delta=self.relative_discharge(s,cap)
        self.momentum_correction(s,r)
        F=xp.empty(nf);T=xp.empty(nf);dm=xp.empty(n)
        dgrad=xp.vstack((self.mesh.gradient(delta[0]),self.mesh.gradient(delta[1])))
        cin=inlet_concentration(time) if callable(inlet_concentration) else inlet_concentration
        cin=xp.ascontiguousarray(xp.broadcast_to(xp.asarray(cin,dtype=xp.float64),(nf,)))
        if bool(xp.any(~xp.isfinite(cin)|(cin<0)|(cin>=self.B))):raise ValueError('Inlet concentration must be finite and 0<=c<1-p')
        self.engine.launch('jc_mobile_faces',nf,n,nf,g['owner'],g['neighbour'],g['kind'],
            g['length'],g['normal'],g['xy'],g['mid'],Q,c,grad,delta,dgrad,cin,F,T)
        factors=xp.empty(n);frequency=xp.empty(n)
        self.engine.launch('jc_mobile_bound',n,n,float(dt),g['offset'],g['faces'],g['owner'],
            g['area'],s[0],s[3],F,T,factors,frequency)
        if dt*float(xp.max(frequency))>1.+1e-12:raise BendTrialRejected('Mobile advection CFL exceeds 1')
        self.engine.launch('jc_mobile_limit',nf,nf,g['owner'],g['neighbour'],factors,F)
        self.engine.launch('jc_mobile_divergence',n,n,g['offset'],g['faces'],g['owner'],g['area'],F,dm)
        rate=xp.zeros_like(s);rate[:3]=r;rate[3]=dm
        return rate,Q,F

    def step(self,dt,*,inlet_concentration=0.):
        dt=self.exchange._dt(dt)
        if dt<=0:raise ValueError('Positive coupled timestep required')
        a=self.a;xp=self.xp;base=self.state;oldq=a.q.copy();oldz=a.z.copy();saved=a.snapshot()
        a.check();a.midpoint.minimum_theta=1.;a.midpoint.frequency_peak=0.
        try:
            s=self.source(base,.5*dt);a.physics.friction(s[:3],.5*dt)
            r,Q0,F0=self.rate(s,dt,self.time,inlet_concentration);s1=s+dt*r
            r,Q1,F1=self.rate(s1,dt,self.time+dt,inlet_concentration)
            end=.5*s+.5*(s1+dt*r);self._validate(end)
            a.physics.friction(end[:3],.5*dt);end=self.source(end,.5*dt);self._validate(end)
            Q=.5*(Q0+Q1);F=.5*(F0+F1);outside=self.mesh.g['neighbour']<0
            si=self.solid_integral+dt*F*outside;wi=self.water_integral+dt*(Q-F)*outside
            solid,water=self.totals(end)
            sr=solid+float(xp.sum(si))-self.initial_solid
            wr=water+float(xp.sum(wi))-self.initial_water
            scale=max(1.,abs(self.initial_solid),abs(self.initial_water))
            if max(abs(sr),abs(wr))>1e-10*scale:raise BendTrialRejected('Coupled phase conservation failed')
            a.check()
            self.state=end;self.time+=dt;self.steps+=1;self.solid_integral=si;self.water_integral=wi
            a.q[...]=end[:3];a.z[...]=end[5];a.physics.z0[...]=end[5]
            self.last=dict(time_s=self.time,solid_residual_m3=sr,water_plus_pore_residual_m3=wr,
                           hydraulic_cfl=dt*a.midpoint.frequency_peak,dt_s=float(dt))
            return dict(self.last)
        except BaseException:
            a.q[...]=oldq;a.z[...]=oldz;a.restore(saved)
            raise
