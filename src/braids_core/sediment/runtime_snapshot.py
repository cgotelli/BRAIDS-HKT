"""Exact batched copies for the private GPU trial rollback snapshot.

All physics arrays remain in the snapshot. Unsupported layouts use the
original copier. The metadata cache only holds addresses for the current owner
and is rebuilt if any array, address, shape or dtype changes.
"""
import copy
import warnings
import numpy as np


def make_copier(original):
    state = {'kernel': None, 'disabled': False}

    def snapshot(owner):
        if owner.backend != 'gpu' or state['disabled']:
            return original(owner)
        if getattr(owner, '_trial_snapshot_in_use', False):
            raise RuntimeError('A shoreline trial is already active')
        xp = owner.xp
        p = owner.physics
        values = [(k, v) for k, v in vars(p).items() if isinstance(v, xp.ndarray)]
        if (len(values) > 65535 or any(not v.flags.c_contiguous or v.data.ptr % 8 for _, v in values)):
            return original(owner)
        pool = getattr(owner, '_trial_snapshot_arrays', {})
        arrays = {}
        for name, value in values:
            saved = pool.get(name)
            if saved is None or saved.shape != value.shape or saved.dtype != value.dtype:
                saved = xp.empty_like(value)
            if not saved.flags.c_contiguous or saved.data.ptr % 8:
                return original(owner)
            arrays[name] = saved
        signature = tuple((k, v.data.ptr, arrays[k].data.ptr, v.nbytes, v.shape, v.dtype.str)
                          for k, v in values)
        cached = getattr(owner, '_batched_trial_copy', None)
        if cached is None or cached[0] != signature:
            sources = xp.asarray(np.asarray([v.data.ptr for _, v in values], dtype=np.uint64))
            targets = xp.asarray(np.asarray([arrays[k].data.ptr for k, _ in values], dtype=np.uint64))
            sizes = xp.asarray(np.asarray([v.nbytes for _, v in values], dtype=np.int64))
            cached = (signature, sources, targets, sizes)
            owner._batched_trial_copy = cached
        if values:
            try:
                if state['kernel'] is None:
                    state['kernel'] = xp.RawKernel(r'''
                    extern "C" __global__ void braids_batch_trial_copy(
                        const unsigned long long* src, const unsigned long long* dst,
                        const long long* sizes) {
                      int a=blockIdx.y;
                      const unsigned long long* s=(const unsigned long long*)src[a];
                      unsigned long long* d=(unsigned long long*)dst[a];
                      long long words=sizes[a]/8;
                      for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
                          i<words; i+=(long long)gridDim.x*blockDim.x) d[i]=s[i];
                      if(blockIdx.x==0 && threadIdx.x<sizes[a]%8) {
                        long long k=words*8+threadIdx.x;
                        ((unsigned char*)d)[k]=((const unsigned char*)s)[k];
                      }
                    }''', 'braids_batch_trial_copy', options=('--std=c++11',))
                state['kernel']((16, len(values)), (256,), cached[1:])
            except Exception as exc:
                if not type(exc).__module__.startswith('cupy.cuda.compiler'):
                    raise
                state['disabled'] = True
                warnings.warn('Batched rollback copies unavailable; using the existing copier: '+str(exc), RuntimeWarning)
                return original(owner)
        meta = copy.deepcopy({k: getattr(p, k) for k in ('time', 'steps', 'last', '_open', '_dt') if hasattr(p, k)})
        meta['midpoint_state'] = owner.midpoint.state()
        owner._trial_snapshot_arrays = arrays
        owner._trial_snapshot_in_use = True
        return arrays, meta

    return snapshot
