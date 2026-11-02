"""Continuous river lifecycle for the qualified v101 column arithmetic.

The historical 20-update owner and its checkpoints remain unchanged. This
owner has a distinct restart scope, not a silent widening of that contract.
"""
import numpy as np
from .bed_columns_v101 import ColumnOwnerV101, host, digest, array_sha

SCOPE = 'river-continuous-columns-v1'
_BIT_KERNELS = {}


class RiverColumnOwner(ColumnOwnerV101):
    def __init__(self, runtime, *args, **kwargs):
        from .river_material import RiverComposition
        self.composition = RiverComposition(runtime.methods, getattr(runtime, "cp", np))
        super().__init__(runtime, *args, **kwargs)

    @property
    def surface_fraction(self):
        return self.composition.fractions(self.available)

    def _resident_meta(self):
        return digest(dict(runtime=self.runtime.identity,updates=self.updates,time=self.time_s,
            ledgers=array_sha(*(host(getattr(self,k)) for k in ('reference','external','basal')))))

    def _seal(self):
        if self.xp is np:
            for a in (self.mass,self.sv,self.rock,self.area):a.setflags(write=False)
        images=getattr(self,'_commit_images',None)
        if images is None:
            self._sealed_arrays=tuple(a.copy() for a in (self.mass,self.sv,self.rock,self.area))
        else:
            # commit() already validated BOTH accepted and pending states.
            # Transfer immutable pending images; never overwrite an image
            # retained by the outer solver's rollback dictionary.
            self._sealed_arrays=images[0][:2]+images[1][2:]
        self._sealed_meta=self._resident_meta()
        self._state_sha_cache=None
        self.__dict__.pop('_state_sha_override',None)

    def _accepted_sha(self):
        if self._state_sha_cache is None:self._state_sha_cache=self._token()
        return self._state_sha_cache

    @property
    def state_sha256(self):
        # Canonical durable identity is unchanged. Materialize on publication,
        # checkpoint or explicit request, not every accepted computational step.
        self.validate()
        return self._accepted_sha()

    @state_sha256.setter
    def state_sha256(self,value):self._state_sha_override=value

    def _trial_sha(self):
        if self._pending_sha_cache is None:self._pending_sha_cache=self._pending_token()
        return self._pending_sha_cache

    @property
    def pending_sha256(self):
        if self.pending is None:return None
        self.validate_pending()
        return self._trial_sha()

    @pending_sha256.setter
    def pending_sha256(self,value):self._pending_sha_override=value

    def _same_resident_bits(self,arrays,sealed):
        xp=self.xp
        for a,b in zip(arrays,sealed):
            if not isinstance(a,xp.ndarray) or a.shape!=b.shape or a.dtype!=b.dtype:
                return False
        if xp is np:
            return all(np.array_equal(a.view(np.uint64),b.view(np.uint64)) for a,b in zip(arrays,sealed))
        # Bit equality also detects signed-zero changes, unlike floating-point
        # array_equal. One host status transfer for all fields, no weaker hash.
        count=len(arrays)
        if count not in _BIT_KERNELS:
            parameters=', '.join(f'raw uint64 a{k}, raw uint64 b{k}, int64 n{k}' for k in range(count))
            operation='same=true;'+''.join(f'if(i<n{k}) same = same && (a{k}[i]==b{k}[i]);' for k in range(count))
            _BIT_KERNELS[count]=xp.ElementwiseKernel(parameters,'bool same',operation,f'river_column_bits_{count}')
        args=[]
        for a,b in zip(arrays,sealed):args.extend((a.view(xp.uint64).ravel(),b.view(xp.uint64).ravel(),np.int64(a.size)))
        equal=_BIT_KERNELS[count](*args,size=max(a.size for a in arrays))
        return bool(equal.all())

    def validate(self):
        if (self._resident_meta()!=self._sealed_meta
                or not self._same_resident_bits((self.mass,self.sv,self.rock,self.area),self._sealed_arrays)):
            raise ValueError('Accepted column was mutated')
        if '_state_sha_override' in self.__dict__ and self._state_sha_override!=self._accepted_sha():
            raise ValueError('Accepted column was mutated')

    def seal_pending(self):
        self._sealed_pending_arrays=tuple(a.copy() for a in (*self.pending[:2], self.pending_basal))
        self._sealed_pending_meta=self._pending_meta()
        self._pending_sha_cache=None
        self.__dict__.pop('_pending_sha_override',None)

    def _pending_meta(self):
        return digest(dict(ledgers=array_sha(host(self.pending[2]),host(self.pending[3])),
                           dt=self.pending[4],residual=host(self.pending[5]).tolist()))

    def validate_pending(self):
        if self.pending is None:raise ValueError('No column trial')
        if (self._pending_meta()!=self._sealed_pending_meta
                or not self._same_resident_bits((*self.pending[:2], self.pending_basal),self._sealed_pending_arrays)):
            raise ValueError('Column trial was mutated')
        if '_pending_sha_override' in self.__dict__ and self._pending_sha_override!=self._trial_sha():
            raise ValueError('Column trial was mutated')

    def commit(self,*,cancelled=False):
        if self.pending is None:return super().commit(cancelled=cancelled)
        self._commit_images=(self._sealed_pending_arrays,self._sealed_arrays)
        try:return super().commit(cancelled=cancelled)
        finally:
            self.__dict__.pop('_commit_images',None)
            if self.pending is None:self.pending_basal=None

    def rollback(self):
        super().rollback()
        self.pending_basal=None

    def begin(self, delta, depth, dt, *, cancelled=False, **process_fields):
        self.validate()
        if self.pending is not None: raise ValueError('An uncommitted column trial already exists')
        if cancelled: raise InterruptedError('Column update cancelled before launch')
        if not np.isfinite(dt) or dt <= 0: raise ValueError('Positive column timestep required')
        targets = self.runtime.targets(depth)
        out, sv, basal = self.runtime.evaluate(self.mass, self.sv, delta, targets, dt, **process_fields)
        before = host((self.mass*self.area[None,None,:]).sum(axis=(0,2)))
        after = host((out*self.area[None,None,:]).sum(axis=(0,2)))
        ext = host((delta*self.area[None,:]).sum(axis=1))
        base = host((basal*self.area[None,:]).sum(axis=1))
        residual = after-before-ext-base
        if np.max(np.abs(residual)) > 2e-11*max(1., np.max(before)):
            raise ValueError('Classwise column budget does not close')
        self.pending = (out, sv, ext, base, float(dt), residual)
        # Resident administrative source, not a face flux or bank transfer.
        # Seal the local field with the trial; commit keeps its global ledger.
        self.pending_basal = basal
        self.seal_pending()
        return self.pending

    def checkpoint(self):
        d = super().checkpoint()
        d.pop('checkpoint_sha256')
        d['scope'] = SCOPE
        return dict(d, checkpoint_sha256=digest(d))

    def restore(self, d):
        if self.pending is not None: raise ValueError('Restore requires no open trial')
        raw = {k:v for k,v in d.items() if k != 'checkpoint_sha256'}
        if (digest(raw) != d['checkpoint_sha256'] or d['scope'] != SCOPE or d['schema'] != 1
                or d['runtime'] != self.runtime.identity or d['methods'] != self.runtime.methods.fingerprint):
            raise ValueError('River column restart identity differs')
        c = RiverColumnOwner(self.runtime,d['mass'],d['sv'],d['rock'],d['area'],time_s=d['time_s'])
        if type(d['updates']) is not int or d['updates'] < 0: raise ValueError('Invalid update count')
        c.updates = d['updates']
        for k in ('reference','external','basal'):
            v = np.asarray(d[k],float)
            if v.shape != c.reference.shape or not np.isfinite(v).all(): raise ValueError('Invalid ledger')
            setattr(c,k,v.copy())
        c._seal()
        if c.state_sha256 != d['state_sha256']: raise ValueError('River column restart state differs')
        # Replace stale lazy-token overrides and private trial caches as well
        # as physical fields; an authenticated restore is a complete owner.
        self.__dict__.clear();self.__dict__.update(c.__dict__)
