"""Standalone single-grain Jiang-Cao exchange and conservative inventory transport.

Not a public execution path. No implicit activation, capacity-Exner update,
suspended load, MF, hydraulic stepping, checkpoint or AMR integration.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from pathlib import Path
import numpy as np
from .jiang_cao_capacity import JiangCaoCapacity, JiangCaoMaterial

METHOD = "transport.non_equilibrium_bedload_jc2026_component_v1"
STATE_FIELDS = ("h", "hu", "hv", "mobile_solid", "bed_solid", "bed_z")
EXCHANGE_FIELDS = STATE_FIELDS + ("net_erosion_solid", "alpha")


@dataclass(frozen=True)
class ExchangeMaterial:
    capacity_material: JiangCaoMaterial
    porosity: float
    kinematic_viscosity_m2_s: float

    def __post_init__(self):
        if not isinstance(self.capacity_material, JiangCaoMaterial):
            raise TypeError("Explicit JiangCaoMaterial required")
        vals=(self.porosity,self.kinematic_viscosity_m2_s)
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) or not np.isfinite(v) for v in vals):
            raise ValueError("Finite real porosity and viscosity required")
        if not 0<=self.porosity<1 or self.kinematic_viscosity_m2_s<=0:
            raise ValueError("Require 0<=porosity<1 and positive water kinematic viscosity")
        if not np.isfinite(self.settling_velocity) or self.settling_velocity<=0:
            raise ValueError("Material outside finite settling range")

    @property
    def settling_velocity(self):
        p=self.capacity_material
        a=13.95*self.kinematic_viscosity_m2_s/p.d50_m
        b=1.09*(p.solid_density_kg_m3/p.water_density_kg_m3-1)*p.gravity_m_s2*p.d50_m
        # Rationalized Zhang (1961), verified in Chen et al. (2015), Eq. 8.
        # Avoid cancellation for fine grains; not hindered/cohesive settling.
        return b/(np.hypot(a,np.sqrt(b))+a)


@dataclass(frozen=True)
class ExchangeResult:
    values: object
    status: object

    @property
    def state(self):
        return self.values[:6]


@dataclass(frozen=True)
class TransportResult:
    depth: object
    mobile_solid: object
    solid_face_rate: object
    water_face_rate: object
    maximum_cfl: float


class JiangCaoExchange:
    def __init__(self,backend,material: ExchangeMaterial,*,threads=4):
        if not isinstance(material,ExchangeMaterial):
            raise TypeError("Explicit ExchangeMaterial required")
        self.capacity=JiangCaoCapacity(backend,material.capacity_material,threads=threads)
        self.backend,self.xp,self.material=backend,self.capacity.xp,material
        from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
        path=Path(__file__).with_suffix('.h')
        self.engine=WaveEngine(backend,threads,source=path,
            names=('jc_exchange','jc_inventory_faces','jc_inventory_cells'))
        p=material.capacity_material
        self._wire=self.xp.asarray([1-material.porosity,material.settling_velocity,
                                  p.water_density_kg_m3,p.solid_density_kg_m3])
        self.identity=dict(method=METHOD,public_support=False,material=asdict(material),
            capacity=self.capacity.identity,runtime=self.engine.identity,
            settling='zhang1961_13.95_1.09_clear_water_rationalized',
            exchange='exact_frozen_ce_finite_stock_all_targets210_v1; variable_h_piecewise_alpha',
            momentum='local_Sc_exchange_only; rho*hu_and_rho*hv_invariant',
            transport='shared_face_upwind_mobile_inventory; SSPRK22; prescribed_mixture_flux',
            scope='fixed_geometry_component; NOT hydraulic_solver_or_public_model',
            wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.identity['sha256']=hashlib.sha256(json.dumps(self.identity,sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _dt(dt):
        if isinstance(dt,(bool,np.bool_)) or not isinstance(dt,Real) or not np.isfinite(dt) or dt<0:
            raise ValueError("Finite nonnegative dt required")
        return float(dt)

    def exchange(self,state,capacity_concentration,dt):
        """Exact frozen-target source flow, not a frozen-h exponential.

        m=hc and bed_solid are solid volume/area (m); z and h are bulk lengths.
        Locally conserve m+bed_solid, h+z and h-m+p*bed_solid/(1-p).
        The paper's local momentum exchange conserves rho*(hu,hv).
        ce is a finite nonnegative forcing target, not the actual concentration.
        Actual c stays <1-p; finite stock bounds erosion even when ce>=1-p.
        """
        dt=self._dt(dt);xp=self.xp
        state=xp.ascontiguousarray(xp.asarray(state,dtype=xp.float64))
        if state.ndim!=2 or state.shape[0]!=6 or state.shape[1]>np.iinfo(np.int32).max:
            raise ValueError("state must be (6,n): h,hu,hv,mobile_solid,bed_solid,z")
        n=state.shape[1]
        ce=xp.ascontiguousarray(xp.broadcast_to(xp.asarray(capacity_concentration,dtype=xp.float64),(n,)))
        out=xp.empty((8,n));status=xp.empty(n,dtype=xp.int32)
        if n:self.engine.launch('jc_exchange',n,n,self._wire,dt,state,ce,out,status)
        if bool(xp.any(status>=3)):
            cell=int(xp.flatnonzero(status>=3)[0]);code=int(status[cell])
            raise ValueError(f"Exchange cell {cell}, status {code}: require finite state, nonnegative stock and ce, actual 0<=c<1-p; dry cells need zero mobile, momentum and capacity")
        return ExchangeResult(out,status)

    def capacity_target(self,state,gradient,manning,*,capacity_q=None):
        """Convert capacity to ce=qb/(h*V), without 0/0 or a hidden concentration cap."""
        xp=self.xp;s=xp.asarray(state,dtype=xp.float64)
        if s.ndim!=2 or s.shape[0]!=6:raise ValueError('Six-row exchange state required')
        h=s[0];m=s[3];B=1-self.material.porosity
        if bool(xp.any(~xp.isfinite(s))) or bool(xp.any((h<0)|(m<0)|((h>0)&(m>=B*h))|((h==0)&(m!=0)))):
            raise ValueError('Invalid mixture inventory')
        c=xp.where(h>0,m/xp.where(h>0,h,1.),0.)
        q=s[:3] if capacity_q is None else xp.ascontiguousarray(xp.asarray(capacity_q,dtype=xp.float64))
        if q.shape!=(3,s.shape[1]) or bool(xp.any(~xp.isfinite(q))) or bool(xp.any(q[0]<0)):
            raise ValueError('Capacity hydraulic state must be finite q(3,n) with nonnegative depth')
        result=self.capacity.evaluate(q,gradient,manning,concentration=c)
        discharge=xp.hypot(q[1],q[2])
        ce=xp.where(discharge>0,result.field('qb')/xp.where(discharge>0,discharge,1.),0.)
        if bool(xp.any(~xp.isfinite(ce)|(ce<0))):
            raise ValueError('Capacity target must be finite and nonnegative; no transport-law clipping')
        return ce,result

    def transport_mesh(self,areas,owner,neighbour,length,kind):
        return InventoryMesh(self,areas,owner,neighbour,length,kind)


class InventoryMesh:
    """Fixed verified face graph; lengths/areas are actual polygon measures.

    kind: 0 internal, 1 impermeable wall, 2 prescribed inlet, 3 export-only outlet.
    No face is inferred from an axis or a boundary's name. No geometry ownership
    or public mesh/restart integration is claimed by this component.
    """
    def __init__(self,operator,areas,owner,neighbour,length,kind):
        self.op=operator;xp=operator.xp
        area=np.asarray(areas,dtype=float);length=np.asarray(length,dtype=float)
        o=np.asarray(owner);j=np.asarray(neighbour);k=np.asarray(kind)
        n=len(area) if area.ndim==1 else 0;nf=len(o) if o.ndim==1 else 0
        if not n or not nf or max(n,nf)>np.iinfo(np.int32).max or area.shape!=(n,) or length.shape!=(nf,) or j.shape!=(nf,) or k.shape!=(nf,):
            raise ValueError('Nonempty finite-volume cell/face arrays required')
        if any(not np.issubdtype(a.dtype,np.integer) for a in (o,j,k)):
            raise ValueError('Integer topology required; no truncation')
        if (not np.isfinite(area).all() or not np.isfinite(length).all() or min(area.min(),length.min())<=0
            or np.any((o<0)|(o>=n)|(j< -1)|(j>=n)|(o==j)) or np.any(~np.isin(k,[0,1,2,3]))
            or np.any((j>=0)!=(k==0))):
            raise ValueError('Invalid actual geometry or explicit boundary kind')
        cells=[[] for _ in range(n)]
        for f,(a,b) in enumerate(zip(o,j)):
            cells[a].append(f)
            if b>=0:cells[b].append(f)
        if any(not a for a in cells):raise ValueError('Every cell needs at least one face')
        offsets=np.r_[0,np.cumsum([len(a) for a in cells])]
        if offsets[-1]>np.iinfo(np.int32).max:raise ValueError('Adjacency too large')
        self.n,self.nf=n,nf
        self._g={name:xp.asarray(a,dtype=xp.float64 if name in ('area','length') else xp.int32).copy()
                 for name,a in dict(area=area,length=length,owner=o,neighbour=j,kind=k,
                                   offset=offsets,faces=np.concatenate(cells)).items()}
        self.geometry_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in (area,o,j,length,k))).hexdigest()

    def advance(self,depth,mobile_solid,dt,*,sediment_normal_velocity,mixture_face_rate,inlet_solid_rate=0.):
        """SSPRK22 inventory transport, first-order spatial reference operator.

        vn in m/s, total mixture and inlet solid rates in m3/s (include width).
        Mixture hydraulics are prescribed here; water rate=mixture-solid.
        This establishes phase accounting, NOT the hydraulic momentum solve.
        """
        dt=self.op._dt(dt);xp=self.op.xp;g=self._g;B=1-self.op.material.porosity
        def arr(a,n):return xp.ascontiguousarray(xp.broadcast_to(xp.asarray(a,dtype=xp.float64),(n,)))
        h,m=arr(depth,self.n),arr(mobile_solid,self.n)
        vn,Q,feed=(arr(a,self.nf) for a in (sediment_normal_velocity,mixture_face_rate,inlet_solid_rate))
        if any(bool(xp.any(~xp.isfinite(a))) for a in (h,m,vn,Q,feed)):
            raise ValueError('Finite transport inputs required')
        if bool(xp.any((h<0)|(m<0)|((h>0)&(m>=B*h))|((h==0)&(m!=0)))):
            raise ValueError('Transport requires 0<=c<1-p and nonnegative depth')
        if bool(xp.any((feed<0)|((g['kind']!=2)&(feed!=0))|((feed>0)&(Q>=0))|((g['kind']==1)&(Q!=0)))):
            raise ValueError('Feed only at inward-flow inlet; wall mixture flux must be zero')
        if bool(xp.any((g['kind']==2)&(Q<0)&(feed>=-B*Q))):
            raise ValueError('Inlet feed concentration must remain below bed packing')
        def euler(h0,m0):
            F=xp.empty(self.nf);out=xp.empty((3,self.n));status=xp.empty(self.n,dtype=xp.int32)
            self.op.engine.launch('jc_inventory_faces',self.nf,self.nf,self.n,g['owner'],g['neighbour'],
                g['kind'],g['length'],vn,feed,Q,m0,F)
            self.op.engine.launch('jc_inventory_cells',self.n,self.n,dt,float(B),g['area'],g['offset'],
                g['faces'],g['owner'],g['neighbour'],g['kind'],g['length'],vn,Q,F,h0,m0,out,status)
            if bool(xp.any(status)):
                cell=int(xp.flatnonzero(status)[0]);code=int(status[cell])
                raise ValueError(f'Transport cell {cell}: status {code}; reduce timestep (CFL<=1) or correct inconsistent phase fluxes')
            return out,F
        a,Fa=euler(h,m);b,Fb=euler(a[0],a[1])
        F=.5*(Fa+Fb)
        return TransportResult(.5*(h+b[0]),.5*(m+b[1]),F,Q-F,
                               float(xp.maximum(xp.max(a[2]),xp.max(b[2]))))
