"""Research wet/dry geometry and conservative phase transport; no public route."""
from pathlib import Path
import numpy as np
from .bend_methods_v1 import BendTrialRejected,digest
from braids_support.sediment.river_morphodynamics.sediment_wave import WaveEngine

NAMES=('jc201_gradient','jc201_low','jc201_bounds','jc201_correct','jc201_endpoint','jc201_manning')


class ShorelineOperators:
    def __init__(self,mesh,backend,*,threads=4):
        if backend not in ('cpu','gpu') or mesh.engine.backend!=backend:
            raise ValueError('Geometry and explicitly selected backend must match')
        self.mesh,self.xp,self.n,self.nf=mesh,mesh.xp,mesh.n,mesh.nf
        self.engine=WaveEngine(backend,threads,source=Path(__file__).with_suffix('.h'),names=NAMES)
        self.identity=dict(method='geometry.phase_bounds_jc201_v1',runtime=self.engine.identity,
            wrapper_sha256=digest(__file__),geometry=mesh.geometry_sha256,public_support=False,
            gradient='active-neighbor weighted physical-centroid LS; objective rank-one pseudoinverse',
            flux='actual hydraulic mixture flux; low-order donor reference; shared antidiffusive bounds on m and B*h-m')

    def gradient(self,value,active,*,limited=False):
        xp=self.xp;g=self.mesh.g;n=self.n;nf=self.nf
        v=xp.ascontiguousarray(xp.asarray(value,dtype=xp.float64))
        a=xp.ascontiguousarray(xp.asarray(active,dtype=xp.int32))
        if v.shape!=(n,) or a.shape!=(n,) or not bool(xp.all(xp.isfinite(v))) or bool(xp.any((a!=0)&(a!=1))):
            raise ValueError('Finite scalar field and boolean activity mask required')
        out=xp.empty((2,n))
        self.engine.launch('jc201_gradient',n,n,nf,int(limited),g['offset'],g['faces'],g['owner'],
            g['neighbour'],g['xy'],g['mid'],a,v,out)
        return out

    def flux(self,h,m,Q,high,dt,*,packing,inlet_concentration=0.):
        xp=self.xp;g=self.mesh.g;n=self.n;nf=self.nf
        if not np.isfinite(dt) or dt<=0 or not 0<packing<=1:raise ValueError('Positive dt and valid packing required')
        def array(v,count):return xp.ascontiguousarray(xp.broadcast_to(xp.asarray(v,dtype=xp.float64),(count,)))
        h,m,Q,high,cin=array(h,n),array(m,n),array(Q,nf),array(high,nf),array(inlet_concentration,nf)
        if any(not bool(xp.all(xp.isfinite(v))) for v in (h,m,Q,high,cin)):
            raise ValueError('Finite phase fields and rates required')
        if bool(xp.any((h<0)|(m<0)|(m>packing*h))) or bool(xp.any((cin<0)|(cin>=packing))):
            raise ValueError('Phase inventory or inlet packing bound violated')
        if bool(xp.any(Q[g['kind']==1]!=0)):raise ValueError('Impermeable wall has nonzero mixture flux')
        low=xp.empty(nf);H=xp.empty(n);M=xp.empty(n);plus=xp.empty(n);minus=xp.empty(n);bad=xp.empty(n,dtype=xp.int32)
        self.engine.launch('jc201_low',nf,n,nf,g['owner'],g['neighbour'],g['kind'],h,m,Q,cin,low)
        # Exterior flux follows the prescribed donor/feed rule, never a relative
        # bed drift into walls or a silently altered prescribed concentration.
        high=xp.where(g['neighbour']<0,low,high)
        self.engine.launch('jc201_bounds',n,n,float(dt),float(packing),g['offset'],g['faces'],g['owner'],
            g['area'],h,m,Q,low,high,H,M,plus,minus,bad)
        if bool(xp.any(bad)):
            i=int(xp.flatnonzero(bad)[0]);raise BendTrialRejected(f'Low-order phase bound at cell {i}, code {int(bad[i])}; hydraulic donor flux or timestep inconsistent')
        F=xp.empty(nf);theta=xp.empty(nf);end=xp.empty(n)
        self.engine.launch('jc201_correct',nf,nf,g['owner'],g['neighbour'],low,high,plus,minus,F,theta)
        self.engine.launch('jc201_endpoint',n,n,float(dt),g['offset'],g['faces'],g['owner'],g['area'],m,F,end)
        if not bool(xp.all(xp.isfinite(end))) or bool(xp.any((end<0)|(end>packing*H))):
            raise BendTrialRejected('Corrected phase endpoint out of bounds; no state clipping')
        return H,end,F,theta
