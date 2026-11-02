"""Jiang-Cao repose criterion on the ordinary, single-bulk Exner bed owner.

Same shared CPU/CUDA geometric kernels, distinct inventory coupling. No
hydraulic, entrainment, capacity, friction or secondary-flow substitution.
"""
import hashlib
from pathlib import Path
import numpy as np
from .bed_columns_v101 import host
from .bend_methods_v1 import digest,BendTrialRejected
from .jiang_cao_collapse import ReposeCollapse,ReposeMaterial
from .jiang_cao_coupled import NAMES
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

METHOD='redistribution.jiang_cao_repose_exner223_v1'


class ReposeMesh:
    """True physical faces; unique-neighbour vector LS at actual centroids."""
    def __init__(self,a):
        self.xp=a.xp;self.n=a.physics.n;self.nf=a.physics.nf;self.g=a.physics.g
        self.engine=WaveEngine(a.backend,a.physics.flow.threads,
            source=Path(__file__).with_name('jiang_cao_coupled.h'),names=NAMES)
        xy=host(self.g['xy']).T;o=host(self.g['owner']);j=host(self.g['neighbour'])
        neighbours=[set() for _ in range(self.n)]
        for left,right in zip(o,j):
            if right>=0:neighbours[left].add(int(right));neighbours[right].add(int(left))
        offsets=[0];adj=[];weights=[]
        for i,row in enumerate(neighbours):
            ids=sorted(row);d=xy[ids]-xy[i]
            if ids:
                r2=np.sum(d*d,axis=1)
                if np.any(r2<=0):raise ValueError('Distinct physical centroids required')
                weighted=d/r2[:,None]
                weights.extend(weighted@np.linalg.pinv(d.T@weighted,rcond=1e-13))
                adj.extend(ids)
            offsets.append(len(adj))
        self.ls=dict(offset=self.xp.asarray(offsets,dtype=self.xp.int32),
            adj=self.xp.asarray(adj,dtype=self.xp.int32),
            weights=self.xp.asarray(np.asarray(weights).reshape(-1,2)),
            delta=self.xp.zeros((len(adj),2)))
        h=hashlib.sha256(b'unique-neighbour-physical-repose-geometry223')
        for key in ('xy','owner','neighbour','area','offset','faces'):
            h.update(np.ascontiguousarray(host(self.g[key])).tobytes())
        self.geometry_sha256=h.hexdigest()

    def gradient(self,value):
        xp=self.xp;g=self.ls;out=xp.empty((2,self.n))
        self.engine.launch('jc_gradient',self.n,self.n,0,g['offset'],g['adj'],g['weights'],g['delta'],value,out)
        return out


class ExnerRepose:
    def __init__(self,a,angle_deg):
        if len(a.material.solid)!=1 or a.methods.bed!='mixed' or a.methods.porosity!=0 or a.methods.base_update!=1:
            raise ValueError('Exner repose requires one bulk material, fixed dry-bulk packing and a finite fully mixed bed')
        self.a=a;self.xp=a.xp;self.mesh=ReposeMesh(a)
        self.solid=float(a.material.solid[0]);self.bulk=float(a.material.column[0])
        self.material=ReposeMaterial(1.-self.bulk/self.solid,float(np.deg2rad(angle_deg)))
        self.op=ReposeCollapse(self.mesh,self.material,a.backend,threads=a.physics.flow.threads,require_repose=False)
        self.identity=dict(method=METHOD,projection=self.op.identity,source_sha256=digest(__file__),
            coupling='after transport/exchange; one finite Exner inventory; no extra entrainment; no MF multiplier',
            water='h and momentum conserved by redistribution; no coupled-phase or pore-water source',
            grain='one bulk density and porosity; repose angle is independent of transport diameter descriptors',
            stencil='unique internal neighbours; full vector LS plus physical centroid secants',
            limits='single prescribed repose angle for wet and dry cells; noncohesive geometric slope relaxation; no kinetic block failure; immobile residual slope allowed and reported')

    def apply(self,mass,sv,rock):
        a=self.a;xp=self.xp
        if mass.shape[0:2]!=(1,1) or not bool(xp.all(sv==1.)):
            raise ValueError('Exner repose requires the admitted one-layer dry-bulk packing')
        state=xp.zeros((6,self.mesh.n));state[4]=mass[0,0]/self.solid
        state[5]=a.material.bed(mass,sv,rock)
        result=self.op.apply(state,check=a.check)
        # Preserve a no-op exactly; active transfers come from the conservative
        # solid-volume projection, never differences of near-equal elevations.
        work=mass.copy() if result.gross_solid_transfer_m3==0. else result.state[4][None,None,:]*self.solid
        move=(work-mass).sum(0)*a.physics.g['area']
        if abs(float(move.sum()))>1e-9:raise BendTrialRejected('Exner repose internal mass ledger failed')
        a.physics.last.update(repose_stock_constrained=result.stock_constrained,
            repose_maximum_slope=result.max_slope,repose_iterations=result.iterations,
            repose_angle_deg=float(np.rad2deg(self.material.repose_angle_rad)))
        return work,sv.copy(),move


def install(a,angle_deg):
    if not a.collapse:raise ValueError('Explicit collapse switch required for Exner repose')
    a._repose_projection=ExnerRepose(a,angle_deg)
    a.physics.identity['exner_repose']=a._repose_projection.identity
    a.physics.identity['slope_relaxation']=dict(method=METHOD,repose_angle_deg=float(angle_deg))
