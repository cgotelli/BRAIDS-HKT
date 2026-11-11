"""Conservative stock-aware allocation for prescribed-discharge inlets.

Boundary Riemann fluxes are unchanged unless their local outward water debit
exceeds the owner's available water. The resulting adjustment is balanced
within the same section, and the depth-independent pressure contribution is
retained. No cells or hydraulic states are clipped.
"""
from pathlib import Path
import numpy as np
from braids_core.sediment.bend_methods_v1 import digest


class InletFeasibility:
    def __init__(self, exterior):
        e=exterior; self.e=e; self.xp=xp=e.xp
        data=e.a.data
        face=np.asarray(data['boundary_faces'],dtype=np.int32)
        owner=np.asarray(data['owner'],dtype=np.int32)[face]
        bid=np.asarray(data['bid'],dtype=np.int32)[face]
        if np.any(owner<0) or np.any(bid<0):raise ValueError('Invalid inlet feasibility boundary topology')
        unique,inverse=np.unique(owner,return_inverse=True)
        self.faces=xp.asarray(face);self.owner=xp.asarray(owner);self.unique=xp.asarray(unique)
        self.inverse=xp.asarray(inverse,dtype=xp.int32);self.bid=xp.asarray(bid)
        self.nowner=len(unique);self.nb=e.nb
        self.length=e.p.g['length'][self.faces];self.area=e.p.g['area'][self.unique]
        self.is_inlet=e.inlet_compatibility.control[self.bid]!=0
        self.identity=dict(method='hllc.prescribed_inlet_stock_allocation.v1',source_sha256=digest(__file__),
            scope='HLLC exact-discharge inlets only; other open-boundary methods unchanged',
            stock='reserve other prescribed outward boundary debits, then limit each owner\'s outward inlet debit to its start-of-substep water volume',
            section='reduce incoming flux proportionally within that inlet to preserve its prescribed net discharge; reject if no feasible allocation',
            momentum='scale normal and tangential trace velocity and advective momentum; retain Riemann hydrostatic pressure',
            state='substep scratch; no film, clipping, backend fallback or state overwrite')

    def __call__(self,q,dt,flux,trace):
        xp=self.xp;f=self.faces;idx=self.inverse;b=self.bid;inside=self.is_inlet
        water=flux[0,f].copy(); rate=xp.maximum(water,0.)*self.length
        debit=xp.where(inside,rate,0.)
        outside=xp.where(~inside,rate,0.)
        total=xp.bincount(idx,weights=debit,minlength=self.nowner)
        reserved=xp.bincount(idx,weights=outside,minlength=self.nowner)
        available=xp.maximum(0.,q[0,self.unique]*self.area/dt-reserved)
        ratio=xp.minimum(1.,available/xp.maximum(total,1e-300))*(1.-32.*np.finfo(float).eps)
        owner_scale=xp.where(total>available,ratio,1.)
        removed=xp.where(inside & (water>0.),debit*(1.-owner_scale[idx]),0.)
        removed_b=xp.bincount(b,weights=removed,minlength=self.nb)
        incoming=xp.where(inside,xp.maximum(-water,0.)*self.length,0.)
        incoming_b=xp.bincount(b,weights=incoming,minlength=self.nb)
        feasible=removed_b<=incoming_b
        in_scale=xp.where(incoming_b>0.,1.-removed_b/xp.maximum(incoming_b,1e-300),1.)
        # An infeasible section retains its original flux and the ordinary
        # HLLC positivity rejection; no discharge or water is invented.
        scale=xp.where(inside & feasible[b] & (water>0.),owner_scale[idx],
              xp.where(inside & feasible[b] & (water<0.),in_scale[b],1.))
        depth=trace[0,f]; denom=xp.maximum(depth,1e-300)
        ux=xp.where(depth>0.,trace[1,f]/denom,0.);uy=xp.where(depth>0.,trace[2,f]/denom,0.)
        flux[1,f]+=(scale*scale-1.)*water*ux;flux[2,f]+=(scale*scale-1.)*water*uy
        flux[0,f]=water*scale;trace[1,f]*=scale;trace[2,f]*=scale
