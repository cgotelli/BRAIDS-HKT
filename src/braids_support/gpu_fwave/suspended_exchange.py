"""Isolated resident noncohesive bed-water source exchange (native modes 1/2).

Inputs are the accepted Euler-stage depth, post-friction momentum, suspended
mass and active-bed mass. This port ends immediately before native active-layer
rebalance; the coupled caller must run that operation and own RK/rollback.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentNoncohesiveExchange:
    def __init__(self,data,*,source_path=None):
        import cupy as cp
        if int(data.get('exchange_mode',-1)) not in (1,2):
            raise ValueError('Only noncohesive exchange modes 1 and 2 are ported')
        if int(data.get('settling_mode',-1)) not in (0,1):
            raise ValueError('Only constant and hindered settling modes are ported')
        if bool(data.get('amr_enabled',False)) or int(data.get('scheme',6))!=6:
            raise ValueError('Isolated exchange requires fixed-grid scheme 6')
        if bool(data.get('bedform_roughness',False)) or float(data.get('bank_erosion_fraction',0.))>0.:
            raise ValueError('Bedform/bank source composition is not ported')
        self.cp=cp
        nc=len(data['manning_cell']);ng=len(data['bulk_fraction'])
        if not 1<=nc or not 1<=ng<=32:raise ValueError('Invalid cell/grain count')
        self.nc=nc;self.ng=ng
        self.exchange_mode=int(data['exchange_mode']);self.settling_mode=int(data['settling_mode'])
        self.deck_version=int(data['deck_version']);self.active_mode=int(data.get('active_layer_mode',0))
        if self.deck_version<12 or self.active_mode not in (0,1):
            raise ValueError('Invalid noncohesive active-layer controls')
        sediment_type=np.asarray(data.get('sediment_type_fraction',np.zeros(ng)))
        if sediment_type.shape!=(ng,) or np.any(sediment_type!=0):
            raise ValueError('Cohesive fractions require unported exchange mode 3')
        self.parameters={}
        for name in ('manning_cell','bulk_fraction','ws_fraction','d50_fraction','d90_fraction',
                     'rhos_fraction','z0_fraction','sedthr_fraction','reference_height_fraction',
                     'dstar_fraction','taucr_fraction'):
            values=np.asarray(data[name],dtype=np.float64)
            if values.shape!=((nc,) if name=='manning_cell' else (ng,)) or not np.isfinite(values).all():
                raise ValueError('Invalid exchange parameter '+name)
            self.parameters[name]=cp.asarray(np.ascontiguousarray(values))
        rough=np.asarray(data['sediment_roughness_fraction'])
        if rough.shape!=(ng,) or rough.dtype.kind not in 'iu' or np.any((rough<0)|(rough>1)):
            raise ValueError('Invalid sediment roughness mode')
        self.roughness=cp.asarray(np.ascontiguousarray(rough,dtype=np.int32))
        p={name:cp.asnumpy(value) for name,value in self.parameters.items()}
        for name in ('bulk_fraction','d50_fraction','d90_fraction','rhos_fraction','z0_fraction',
                     'reference_height_fraction','dstar_fraction','taucr_fraction'):
            if np.any(p[name]<=0):raise ValueError('Positive exchange parameter required: '+name)
        for name in ('ws_fraction','sedthr_fraction','manning_cell'):
            if np.any(p[name]<0):raise ValueError('Nonnegative exchange parameter required: '+name)
        if np.any(p['d90_fraction']<p['d50_fraction']):raise ValueError('D90 below D50')
        self.rho=float(data['rho']);self.grav=float(data['grav']);self.dry=float(data['dry_tolerance'])
        self.target=float(data['active_layer_thickness'])
        self.depth_fraction=float(data.get('active_layer_depth_fraction',0.))
        self.minimum_target=float(data.get('active_layer_minimum_thickness',1.e-6))
        self.hindered_reference=float(data.get('hindered_settling_reference_concentration',1.e4))
        scalars=(self.rho,self.grav,self.dry,self.target,self.depth_fraction,self.minimum_target,self.hindered_reference)
        if (not np.isfinite(scalars).all() or
            min(self.rho,self.grav,self.dry,self.target,self.hindered_reference)<=0 or
            self.depth_fraction<0 or self.minimum_target<0 or
            (self.active_mode==1 and (self.depth_fraction<=0 or self.minimum_target<=0))):
            raise ValueError('Invalid fluid/depth/active-layer/settling controls')
        if np.any(p['rhos_fraction']<=self.rho):raise ValueError('Grain density must exceed water density')
        source=Path(source_path or HERE/'suspended_exchange.cu').read_text()
        module=cp.RawModule(code=source,options=OPTIONS,name_expressions=('sx_exchange',))
        self.kernel=module.get_function('sx_exchange')
        self.next_suspended=cp.empty((ng,nc),cp.float64)
        self.next_active=cp.empty((ng,nc),cp.float64)
        self.delta=cp.empty((ng,nc),cp.float64)
        self.status=cp.empty(nc,cp.int32)

    def _array(self,value,shape):
        if not isinstance(value,self.cp.ndarray) or value.shape!=shape or value.dtype!=self.cp.float64 or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous device FP64 array '+str(shape))

    def step(self,depth,post_friction_momentum,suspended,active,dt):
        """Return trial suspended/active mass, water-column change, and status."""
        nc,ng=self.nc,self.ng
        self._array(depth,(nc,));self._array(post_friction_momentum,(2,nc))
        self._array(suspended,(ng,nc));self._array(active,(ng,nc))
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite timestep required')
        p=self.parameters
        self.kernel(((nc+127)//128,),(128,),(
            np.int32(nc),np.int32(ng),np.int32(self.exchange_mode),np.int32(self.settling_mode),
            np.int32(self.deck_version),np.int32(self.active_mode),depth,post_friction_momentum,
            suspended,active,p['manning_cell'],p['bulk_fraction'],p['ws_fraction'],p['d50_fraction'],
            p['d90_fraction'],p['rhos_fraction'],p['z0_fraction'],p['sedthr_fraction'],
            p['reference_height_fraction'],p['dstar_fraction'],p['taucr_fraction'],self.roughness,
            np.float64(self.rho),np.float64(self.grav),np.float64(self.dry),np.float64(self.target),
            np.float64(self.depth_fraction),np.float64(self.minimum_target),
            np.float64(self.hindered_reference),np.float64(dt),self.next_suspended,self.next_active,
            self.delta,self.status))
        return {'suspended':self.next_suspended,'active':self.next_active,'delta':self.delta,'status':self.status}
