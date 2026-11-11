"""Conservative start-of-stage stock bound for water leaving outlets."""
from pathlib import Path
import numpy as np
from braids_core.sediment.bend_methods_v1 import digest


class OutletFeasibility:
    def __init__(self,exterior):
        e=exterior;self.xp=xp=e.xp;d=e.a.data
        faces=np.asarray(d['boundary_faces'],dtype=np.int32)
        owner=np.asarray(d['owner'],dtype=np.int32)[faces];bid=np.asarray(d['bid'],dtype=np.int32)[faces]
        if np.any(owner<0) or np.any(bid<0):raise ValueError('Invalid outlet feasibility boundary topology')
        unique,inverse=np.unique(owner,return_inverse=True)
        self.faces=xp.asarray(faces);self.cells=xp.asarray(unique);self.inverse=xp.asarray(inverse,dtype=xp.int32)
        self.ncells=len(unique);self.length=e.p.g['length'][self.faces];self.area=e.p.g['area'][self.cells]
        modes=e.hydraulic_mode[xp.asarray(bid)]
        self.outlet=(modes==2)|(modes==3)|(modes==4)|(modes==5)|(modes==10)|(modes==11)
        self.identity=dict(method='outlet.stock_limited_stage_flux.v1',source_sha256=digest(__file__),
            scope='stage, free/critical, free-weir, normal-depth and Q–H outlets; outward flux only',
            budget='reserve other boundary debits and cap all outward outlet faces sharing an owner by its start-of-stage water volume',
            semantics='limited stage-average actual outflow; unchanged boundary rating/trace when sufficient stock; no imposed discharge',
            momentum='scale trace velocity and advective momentum, preserve hydrostatic pressure',
            state='per-stage scratch; no water film, depth clip or backend fallback')

    def __call__(self,q,dt,flux,trace):
        xp=self.xp;f=self.faces;idx=self.inverse;out=self.outlet
        water=flux[0,f].copy();debit=xp.maximum(water,0.)*self.length
        outlet_debit=xp.where(out,debit,0.);other=xp.where(~out,debit,0.)
        total=xp.bincount(idx,weights=outlet_debit,minlength=self.ncells)
        reserved=xp.bincount(idx,weights=other,minlength=self.ncells)
        available=xp.maximum(0.,q[0,self.cells]*self.area/dt-reserved)
        ratio=xp.minimum(1.,available/xp.maximum(total,1e-300))*(1.-32.*np.finfo(float).eps)
        factor=xp.where(total>available,ratio,1.)
        scale=xp.where(out & (water>0.),factor[idx],1.)
        depth=trace[0,f];denom=xp.maximum(depth,1e-300)
        ux=xp.where(depth>0.,trace[1,f]/denom,0.);uy=xp.where(depth>0.,trace[2,f]/denom,0.)
        flux[1,f]+=(scale*scale-1.)*water*ux;flux[2,f]+=(scale*scale-1.)*water*uy
        flux[0,f]=water*scale;trace[1,f]*=scale;trace[2,f]*=scale
