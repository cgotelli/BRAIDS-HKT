"""Stock-aware prescribed-boundary budget for explicit f-wave stages."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected, digest
from braids_core.hydraulics.inlet_feasibility import InletFeasibility
from braids_core.hydraulics.outlet_feasibility import OutletFeasibility


class ExplicitBoundaryBudget:
    def __init__(self, exterior):
        self.exterior=exterior;self.xp=xp=exterior.xp;p=exterior.p;g=p.g
        self.inlet=InletFeasibility(exterior) if exterior.inlet_compatibility is not None else None
        self.outlet=OutletFeasibility(exterior)
        host=lambda a:np.asarray(a if xp is np else xp.asnumpy(a))
        self.open_faces=xp.asarray(np.asarray(exterior.a.data['boundary_faces'],dtype=np.int32))
        owner=np.asarray(exterior.a.data['owner'],dtype=np.int32)[host(self.open_faces)]
        unique=np.unique(owner);self.cells=xp.asarray(unique);self.ncells=len(unique)
        off,faces,signs=map(host,(g['offset'],g['faces'],g['signs']))
        pieces=[np.arange(off[i],off[i+1],dtype=np.int32) for i in unique]
        indices=np.concatenate(pieces) if pieces else np.empty(0,dtype=np.int32)
        group=np.repeat(np.arange(len(unique),dtype=np.int32),[len(x) for x in pieces])
        f=faces[indices];self.faces=xp.asarray(f,dtype=xp.int32);self.signs=xp.asarray(signs[indices])
        self.group=xp.asarray(group);self.length=g['length'][self.faces]
        self.boundary=(g['neighbour'][self.faces]<0)&(g['bid'][self.faces]>=0)
        self.area=g['area'][self.cells]
        # midpoint_faces must limit internal faces while retaining the already
        # budgeted boundary flux. It sees all boundary bids as unprescribed;
        # those fluxes and their zero momentum corrections are restored below.
        self.limiter_bid=g['bid'].copy();self.limiter_bid[self.open_faces]=-1
        self.identity=dict(method='fwave.reserved_exterior_stock.v1',source_sha256=digest(__file__),
          inlet_allocation=None if self.inlet is None else digest(Path(__file__).parents[1]/'hydraulics/inlet_feasibility.py'),
          outlet_allocation=digest(Path(__file__).parents[1]/'hydraulics/outlet_feasibility.py'),
          budget='reserve actual outward prescribed face flux before limiting internal donor flux',
          mass='preserve exact net section discharge and shared-face conservation; reject infeasible exterior debit',
          state='per-stage scratch, no film or water-depth clipping')

    def allocate(self,base,dt):
        self.outlet(base,dt,self.exterior.flux,self.exterior.trace)
        if self.inlet is not None:self.inlet(base,dt,self.exterior.flux,self.exterior.trace)

    def reserve(self,base,dt,water,theta):
        xp=self.xp
        debit=xp.maximum(self.signs*water[self.faces],0.)*self.length
        ext=xp.bincount(self.group,weights=xp.where(self.boundary,debit,0.),minlength=self.ncells)
        interior=xp.bincount(self.group,weights=xp.where(self.boundary,0.,debit),minlength=self.ncells)
        available=base[0,self.cells]*self.area/dt
        if bool(xp.any(ext>available)):raise BendTrialRejected('Prescribed boundary would exceed available water')
        factor=xp.minimum(1.,(available-ext)/xp.maximum(interior,1e-300))*(1.-32.*np.finfo(float).eps)
        theta[self.cells]=xp.where((interior>0.) & (interior>=available-ext),factor,1.)
        return water[self.open_faces].copy()

    def restore(self,water,correction,saved):
        water[self.open_faces]=saved
        correction[:,self.open_faces]=0.
