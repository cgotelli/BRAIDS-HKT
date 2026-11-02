"""Transactional FP64 coupled SSP43; no mesh-dependent backend selection.

All field arithmetic takes place on CUDA. Python controls four stage launches;
the caller reads only status/CFL scalars after a trial. Rejected trials never
modify accepted fields or boundary/source ledgers. This is reusable integration
machinery, not a claim that every physical process is implemented.
"""
from pathlib import Path
import numpy as np
from .time_integration import SSPRK43, STAGE_TIMES, OPTIONS

HERE=Path(__file__).resolve().parent


class CoupledRK43:
    def __init__(self, state, ledger, manning, *, grav, dry, speed_limit,
                 chezy_minimum_depth=0., source_folder=HERE, _extra_names=()):
        import cupy as cp
        self.cp=cp
        self.has_suspended='suspended' in state
        state_keys={'q','layers','secondary'}|({'suspended'} if self.has_suspended else set())
        ledger_keys={'water','bed','secondary','source'}|(
            {'suspended','exchange'} if self.has_suspended else set())
        if set(state)!=state_keys or set(ledger)!=ledger_keys:
            raise ValueError('Explicit coupled state/ledger schema required')
        self.nc=state['q'].shape[1];self.nb=ledger['water'].size
        expected={'q':(3,self.nc),'secondary':(self.nc,)}
        if state['layers'].ndim!=3 or state['layers'].shape[-1]!=self.nc:
            raise ValueError('Layers must have shape (layer,fraction,cell)')
        for key,shape in expected.items():
            if state[key].shape!=shape:raise ValueError('Invalid '+key+' shape')
        ng=state['layers'].shape[1]
        if ledger['bed'].shape!=(ng,self.nb) or ledger['secondary'].shape!=(self.nb,) or ledger['source'].shape!=():
            raise ValueError('Invalid coupled ledger shape')
        if self.has_suspended and (state['suspended'].shape!=(ng,self.nc) or
                                   ledger['suspended'].shape!=(ng,self.nb) or
                                   ledger['exchange'].shape!=(ng,2)):
            raise ValueError('Invalid suspended state/ledger shape')
        for value in (*state.values(),*ledger.values()):
            if not isinstance(value,cp.ndarray) or value.dtype!=cp.float64 or not value.flags.c_contiguous:
                raise ValueError('Coupled arrays must be contiguous FP64 device arrays')
        self.accepted=state;self.accepted_ledger=ledger
        self.base={k:cp.empty_like(v) for k,v in state.items()}
        self.stage={k:cp.empty_like(v) for k,v in state.items()}
        self.next={k:cp.empty_like(v) for k,v in state.items()}
        self.ledger={k:cp.zeros_like(v) for k,v in ledger.items()}
        self.source_first=cp.zeros((),dtype=cp.float64);self.source_second=cp.zeros_like(self.source_first)
        self.hydro=SSPRK43(self.nc,self.nb,manning,grav=grav,dry_tolerance=dry,
            chezy_minimum_depth=chezy_minimum_depth,speed_limit=speed_limit,
            source_path=Path(source_folder)/'time_integration.cu')
        self.status=self.hydro.status;self.cfl=self.hydro.cfl
        self.array_status=cp.empty(max(v.size for v in (*state.values(),*ledger.values())),dtype=cp.int32)
        names=('coupled_capture','coupled_combine','coupled_commit','coupled_sum_source')+tuple(_extra_names)
        from braids_core.windows_paths import filesystem_path
        module=cp.RawModule(code=(filesystem_path(source_folder)/'coupled_time.cu').read_text(),options=OPTIONS,name_expressions=names)
        self.kernels={name:module.get_function(name) for name in names}
        self.identities={k:(v.data.ptr,v.shape) for k,v in state.items()}

    def launch(self,name,n,args):
        if n:self.kernels[name](((n+127)//128,),(128,),args)

    def merge(self,status):
        if status is not None:self.hydro._merge_status(status)

    def _copy(self,destination,source):
        for key in destination:self.cp.copyto(destination[key],source[key])

    def _capture(self,result):
        if set(result)!=set(self.base):raise ValueError('Euler must return every coupled state field exactly once')
        for key,value in result.items():
            if value.shape!=self.base[key].shape or value.dtype!=self.cp.float64 or not value.flags.c_contiguous:
                raise ValueError('Invalid Euler result '+key)
            status=self.array_status[:value.size]
            self.launch('coupled_capture',value.size,(value,self.base[key],self.status,
                np.int64(value.size),np.int32(key in ('layers','suspended')),self.next[key],status))
            self.merge(status)
        # Do not prepare momentum here: native stage 3 combines before prepare.
        # Depth must be nonnegative; signed momentum and hI need not be.
        self.merge(self.cp.where(self.next['q'][0]>=0,self.cp.int32(0),self.cp.int32(1)))

    def _ledger(self,incoming,index):
        keys=('water','bed','secondary','suspended') if self.has_suspended else ('water','bed','secondary')
        if set(incoming)!=set(keys):raise ValueError('Euler must return all transport ledgers')
        for key in keys:
            value=incoming[key];target=self.ledger[key]
            if value.shape!=target.shape or value.dtype!=self.cp.float64 or not value.flags.c_contiguous:
                raise ValueError('Invalid Euler ledger '+key)
            if value.size:
                status=self.array_status[:value.size]
                self.hydro._launch('ti_ledger',value.size,(value,target,np.int64(value.size),np.int32(index),status))
                self.merge(status)

    def transport(self,dt,time,euler,rebalance):
        """Four native Euler legs; kept separate for exact CUDA graph capture.

        Inputs base/stage, latched status and trial ledgers have already been
        initialized by step(). This routine neither relaxes nor commits state.
        """
        cp=self.cp;h=self.hydro
        for index,fraction in enumerate(STAGE_TIMES):
            if index:h._prepare(self.stage['q'],None)
            result=euler(self.stage,.5*dt,float(time+fraction*dt))
            self.merge(result['status'])
            metric=result['cfl']
            cp.max(cp.where(cp.isfinite(metric)&(metric>=0),metric,cp.float64('nan')),out=h.tmp_cfl)
            h._launch('ti_merge_cfl',1,(h.tmp_cfl,self.cfl,self.status))
            self._capture(result['state']);self._ledger(result['ledger'],index)
            if index==2:
                for key in self.base:
                    n=self.base[key].size
                    self.launch('coupled_combine',n,(self.base[key],self.next[key],np.int64(n),self.stage[key]))
                self.merge(rebalance(self.stage['layers']))
            elif index<3:self._copy(self.stage,self.next)

    def step(self,dt,time,euler,rebalance,relax,*,exchange=None,post_source=None,
             pre_commit=None,commit_aux=None,friction_roughness=None,commit=True):
        """Callbacks operate on private device fields; relax returns integral/status.

        euler(state,half_dt,time) -> state, ledger, status, cfl.
        rebalance(layers) -> optional device status.
        relax(state,half_dt) -> (device scalar source integral, device status).
        exchange(state,dt) -> (device (fraction,2) bed/water integral, status).
            This is applied once after RK and Manning, before final rebalance.
        friction_roughness(state) -> (resident Manning array, device status).
            An optional equilibrium closure evaluated on the prepared,
            pre-friction trial state.  No accepted roughness history is stored.
        post_source(state,dt) applies supported diffusion AFTER the second relax.
        pre_commit(state,dt) may transform only private trial fields after all
            sources and before final validity checks; it returns device status.
        commit_aux(status) commits non-RK histories/diagnostics, conditional on
            the same global status as state and ledgers. It is called only when
            commit=True, after all validity checks have been merged.
        """
        cp=self.cp;h=self.hydro
        if not np.isfinite([dt,time]).all() or dt<=0:raise ValueError('Finite positive dt/time required')
        if any((v.data.ptr,v.shape)!=self.identities[k] for k,v in self.accepted.items()):
            raise ValueError('Accepted state storage changed')
        self.status.fill(0);self.cfl.fill(0)
        for value in self.ledger.values():value.fill(0)
        self._copy(self.base,self.accepted);h._prepare(self.base['q'],None)
        source,status=relax(self.base,.5*dt);cp.copyto(self.source_first,source);self.merge(status)
        self._copy(self.stage,self.base)
        self.transport(dt,time,euler,rebalance)
        self.merge(rebalance(self.next['layers']));h._prepare(self.next['q'],None)
        roughness=h.manning
        if friction_roughness is not None:
            roughness,status=friction_roughness(self.next)
            if (not isinstance(roughness,cp.ndarray) or roughness.shape!=(self.nc,) or
                    roughness.dtype!=cp.float64 or not roughness.flags.c_contiguous):
                raise ValueError('Invalid resident pre-friction Manning array')
            self.merge(status)
        h._launch('ti_manning',self.nc,(self.next['q'],roughness,np.int64(self.nc),np.float64(dt),
            h.grav,h.dry,h.chezy_min,h.cell_status));self.merge(h.cell_status)
        if exchange is not None:
            if not self.has_suspended:raise ValueError('Exchange requires suspended coupled state')
            integral,status=exchange(self.next,dt)
            if (not isinstance(integral,cp.ndarray) or integral.shape!=self.ledger['exchange'].shape or
                integral.dtype!=cp.float64 or not integral.flags.c_contiguous):
                raise ValueError('Invalid suspended exchange ledger')
            cp.copyto(self.ledger['exchange'],integral);self.merge(status)
        self.merge(rebalance(self.next['layers']))
        source,status=relax(self.next,.5*dt);cp.copyto(self.source_second,source);self.merge(status)
        self.launch('coupled_sum_source',1,(self.source_first,self.source_second,self.ledger['source'],self.status))
        if post_source is not None:self.merge(post_source(self.next,dt))
        if pre_commit is not None:self.merge(pre_commit(self.next,dt))
        for key,value in self.next.items():
            if key in ('layers','suspended'):
                self.merge(cp.where(cp.isfinite(value)&(value>=0),cp.int32(0),cp.int32(1)))
            else:self.merge(cp.where(cp.isfinite(value),cp.int32(0),cp.int32(1)))
        self.merge(cp.where(self.next['q'][0]>=0,cp.int32(0),cp.int32(1)))
        for key,trial in self.ledger.items():
            if trial.size:
                status=self.array_status[:trial.size]
                h._launch('ti_check_ledger_sum',trial.size,(self.accepted_ledger[key],trial,np.int64(trial.size),status))
                self.merge(status)
        if commit:
            for key,value in self.next.items():
                self.launch('coupled_commit',value.size,(value,self.accepted[key],np.int64(value.size),self.status))
            for key,value in self.ledger.items():
                h._launch('ti_commit_ledger',value.size,(value,self.accepted_ledger[key],np.int64(value.size),self.status))
            if commit_aux is not None:commit_aux(self.status)
        return dict(state=self.next,ledger=self.ledger,status=self.status,cfl=self.cfl)


class CoupledRK33(CoupledRK43):
    """Explicitly selected, native-order three-stage coupled SSPRK(3,3).

    The Euler operator receives the *full* trial dt at stage times 0, 1 and
    1/2. Secondary relaxation and post-transport source ordering are inherited
    from the transactional coupled step. CUDA graph replay is not yet qualified.
    """

    def __init__(self, state, ledger, manning, *, grav, dry, speed_limit,
                 chezy_minimum_depth=0., source_folder=HERE):
        super().__init__(state, ledger, manning, grav=grav, dry=dry,
            speed_limit=speed_limit, chezy_minimum_depth=chezy_minimum_depth,
            source_folder=source_folder, _extra_names=(
                'coupled_ssp33_combine_second','coupled_ssp33_combine_final',
                'coupled_ssp33_ledger'))

    def _ledger(self,incoming,index):
        keys=('water','bed','secondary','suspended') if self.has_suspended else ('water','bed','secondary')
        if set(incoming)!=set(keys):raise ValueError('Euler must return all transport ledgers')
        for key in keys:
            value=incoming[key];target=self.ledger[key]
            if value.shape!=target.shape or value.dtype!=self.cp.float64 or not value.flags.c_contiguous:
                raise ValueError('Invalid Euler ledger '+key)
            if value.size:
                status=self.array_status[:value.size]
                self.launch('coupled_ssp33_ledger',value.size,(
                    value,target,np.int64(value.size),np.int32(index),status))
                self.merge(status)

    def transport(self,dt,time,euler,rebalance):
        """Native SSP33 transport; stage buffers remain private until commit."""
        cp=self.cp;h=self.hydro
        for index,fraction in enumerate((0.,1.,.5)):
            if index:h._prepare(self.stage['q'],None)
            result=euler(self.stage,dt,float(time+fraction*dt))
            self.merge(result['status'])
            metric=result['cfl']
            cp.max(cp.where(cp.isfinite(metric)&(metric>=0),metric,cp.float64('nan')),out=h.tmp_cfl)
            h._launch('ti_merge_cfl',1,(h.tmp_cfl,self.cfl,self.status))
            self._capture(result['state']);self._ledger(result['ledger'],index)
            if index==1:
                for key in self.base:
                    n=self.base[key].size
                    self.launch('coupled_ssp33_combine_second',n,(
                        self.base[key],self.next[key],np.int64(n),self.stage[key]))
                self.merge(rebalance(self.stage['layers']))
            elif index==0:self._copy(self.stage,self.next)
        for key in self.base:
            n=self.base[key].size
            self.launch('coupled_ssp33_combine_final',n,(
                self.base[key],self.next[key],np.int64(n),self.stage[key]))
        self._copy(self.next,self.stage)
