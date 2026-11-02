"""Resident fixed-grid suspended FCT plus post-RK noncohesive exchange.

The native cut-cell solver advects suspended classes in each Euler leg of its
four-leg SSPRK(4,3) method, then performs exchange once after RK combination,
friction, and active-layer rebalance.  This component follows that ordering;
the caller owns hydraulic states, RK combinations, and final rebalancing.

`begin_trial`/`finish_trial` make accepted fields and cumulative ledgers
transactional.  Euler outputs must be copied into caller-owned stage storage:
the lower-level FCT's buffers are deliberately reusable.  The signed exchange
ledger records transfers between water and active bed, not external mass.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .suspended_advection import ResidentSuspendedAdvection
from .suspended_exchange import ResidentNoncohesiveExchange

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentSuspensionComposition:
    """One accepted/trial suspension transaction; no host field round trips."""

    def __init__(self,geometry,exchange_data,ng,*,source_path=None,
                 advection_source_path=None,exchange_source_path=None):
        import cupy as cp
        self.advection=ResidentSuspendedAdvection(geometry,ng,source_path=advection_source_path)
        self.exchange=ResidentNoncohesiveExchange(exchange_data,source_path=exchange_source_path)
        if (self.advection.nc,self.advection.ng)!=(self.exchange.nc,self.exchange.ng):
            raise ValueError('Suspended advection/exchange cell and grain dimensions differ')
        self.cp=cp;self.nc=self.advection.nc;self.ng=self.advection.ng;self.nb=self.advection.nb
        self.area=self.advection.geometry['area']
        names=('sr_stage_ledger','sr_exchange_ledger','sr_commit_field','sr_commit_ledger')
        module=cp.RawModule(code=Path(source_path or HERE/'suspension_resident.cu').read_text(),
                            options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names}
        self.trial_suspended=cp.empty((self.ng,self.nc),cp.float64)
        self.trial_active=cp.empty_like(self.trial_suspended)
        self.trial_boundary=cp.zeros((self.ng,self.nb),cp.float64)
        self.trial_exchange=cp.zeros((self.ng,2),cp.float64)
        self.failure=cp.zeros((),cp.int32)
        self._temp=cp.zeros_like(self.failure)
        self.stage_ledger_status=cp.empty((self.ng,self.nb),cp.int32)
        self.exchange_ledger_status=cp.empty(self.ng,cp.int32)
        self._accepted=None
        self._phase='idle';self._stage=0

    def _array(self,value,shape,dtype):
        if (not isinstance(value,self.cp.ndarray) or value.shape!=shape or
            value.dtype!=dtype or not value.flags.c_contiguous):
            raise ValueError('Expected contiguous device '+str(dtype)+' array '+str(shape))

    def _launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def _merge(self,status):
        if isinstance(status,(tuple,list)):
            for item in status:self._merge(item)
            return
        if status.size:
            self.cp.max(self.cp.where(status!=0,self.cp.int32(1),self.cp.int32(0)),out=self._temp)
            self.cp.maximum(self.failure,self._temp,out=self.failure)

    def begin_trial(self,accepted_suspended,accepted_active,
                    accepted_boundary_ledger,accepted_exchange_ledger):
        """Bind accepted storage and clear a private trial; never mutate it yet."""
        if self._phase=='trial':raise RuntimeError('Previous suspension trial is unfinished')
        for value,shape in ((accepted_suspended,(self.ng,self.nc)),
                            (accepted_active,(self.ng,self.nc)),
                            (accepted_boundary_ledger,(self.ng,self.nb)),
                            (accepted_exchange_ledger,(self.ng,2))):
            self._array(value,shape,self.cp.float64)
        values=(accepted_suspended,accepted_active,accepted_boundary_ledger,accepted_exchange_ledger)
        if len({v.data.ptr for v in values})!=4 or any(v.data.ptr in (
           self.trial_suspended.data.ptr,self.trial_active.data.ptr,
           self.trial_boundary.data.ptr,self.trial_exchange.data.ptr) for v in values):
            raise ValueError('Accepted fields/ledgers must not alias trial storage')
        self._accepted=values
        self._identities=tuple((v.data.ptr,v.shape) for v in values)
        self.cp.copyto(self.trial_suspended,accepted_suspended)
        self.cp.copyto(self.trial_active,accepted_active)
        self.trial_boundary.fill(0.);self.trial_exchange.fill(0.);self.failure.fill(0)
        self._stage=0;self._phase='trial'

    def advection_stage(self,depth,mass,next_depth,accepted_water_flux,left,right,
                        boundary_concentration,hydro_constant,half_dt,out_mass):
        """One Euler FCT leg; stage order and weighted boundary ledger are native."""
        if self._phase!='trial' or self._stage>=4:
            raise RuntimeError('Suspended Euler stage requires an open four-leg trial')
        self._array(out_mass,(self.ng,self.nc),self.cp.float64)
        if out_mass.data.ptr in {v.data.ptr for v in self._accepted}:
            raise ValueError('Euler output must not alias accepted storage')
        result=self.advection.step(depth,mass,next_depth,accepted_water_flux,left,right,
                                   boundary_concentration,hydro_constant,half_dt)
        self._merge(result['status'])
        self.cp.copyto(out_mass,result['next_mass'])
        n=self.ng*self.nb
        self._launch('sr_stage_ledger',n,(result['ledger'],self.trial_boundary,
                     np.int64(n),np.int32(self._stage),self.stage_ledger_status))
        self._merge(self.stage_ledger_status)
        self._stage+=1
        return {'next_suspended':out_mass,'face_flux':result['flux'],
                'boundary_ledger':result['ledger'],'status':result['status'],
                'trial_status':self.failure}

    def post_rk_exchange(self,depth,post_friction_momentum,final_suspended,
                         final_active,dt):
        """Native source order: call once after RK/friction, before rebalance."""
        if self._phase!='trial' or self._stage!=4:
            raise RuntimeError('Exchange requires four completed Euler stages')
        result=self.exchange.step(depth,post_friction_momentum,final_suspended,
                                  final_active,dt)
        self.cp.copyto(self.trial_suspended,result['suspended'])
        self.cp.copyto(self.trial_active,result['active'])
        self._merge(result['status'])
        self._launch('sr_exchange_ledger',self.ng,(
            np.int32(self.nc),np.int32(self.ng),self.area,final_suspended,final_active,
            self.trial_suspended,self.trial_active,self.trial_exchange,self.exchange_ledger_status))
        self._merge(self.exchange_ledger_status)
        self._phase='post_source'
        return {'suspended':self.trial_suspended,'active':self.trial_active,
                'exchange_ledger':self.trial_exchange,'boundary_ledger':self.trial_boundary,
                'status':(result['status'],self.exchange_ledger_status,self.failure)}

    def finish_trial(self,*,global_status=None,commit=True):
        """Commit all fields and ledgers only if every device status is zero."""
        if self._phase!='post_source':raise RuntimeError('Suspension source phase is incomplete')
        if global_status is not None:
            self._array(global_status,(),self.cp.int32)
            self._merge(global_status)
        if tuple((v.data.ptr,v.shape) for v in self._accepted)!=self._identities:
            raise ValueError('Accepted suspension storage changed during trial')
        if commit:
            for trial,accepted in ((self.trial_suspended,self._accepted[0]),
                                   (self.trial_active,self._accepted[1])):
                self._launch('sr_commit_field',trial.size,
                             (trial,accepted,np.int64(trial.size),self.failure))
            for trial,accepted in ((self.trial_boundary,self._accepted[2]),
                                   (self.trial_exchange,self._accepted[3])):
                self._launch('sr_commit_ledger',trial.size,
                             (trial,accepted,np.int64(trial.size),self.failure))
        self._phase='idle'
        return {'suspended':self.trial_suspended,'active':self.trial_active,
                'boundary_ledger':self.trial_boundary,'exchange_ledger':self.trial_exchange,
                'status':self.failure}

    def abandon_trial(self):
        """Discard a trial without touching accepted storage (e.g. CFL retry)."""
        if self._phase not in ('trial','post_source'):
            raise RuntimeError('No suspension trial to abandon')
        self._phase='idle'
