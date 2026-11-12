"""Complete accepted-step transaction for the authenticated A1 GPU fixtures.

The numerical solver is unchanged.  Every reachable BRAIDS-owned CuPy array is
packed by the previously qualified segmented-copy kernel before a trial and is
restored only on rejection.  The copy plan is external to the solver so the
inventory cannot recursively include its own snapshot buffer.
"""
from __future__ import annotations

from weakref import WeakKeyDictionary

import numpy as np

from validation.gpu_fwave import a2_segmented_array_copy as segmented
from validation.gpu_fwave import a2_solver_array_inventory as inventory


_PLANS=WeakKeyDictionary()


def _unique_arrays(solver):
    # Preserve different strided views of one pointer.  They are checked for
    # physical overlap below instead of being silently collapsed.
    unique={}
    for path,array in inventory.gpu_arrays(solver).items():
        key=(int(array.data.ptr),array.shape,array.dtype.str,array.strides)
        unique.setdefault(key,(path,array))
    return {path:array for path,array in unique.values()}


def _byte_addresses(array):
    pointer=np.int64(int(array.data.ptr))
    if array.flags.c_contiguous:
        return pointer+np.arange(array.nbytes,dtype=np.int64)
    if array.ndim==0:
        return pointer+np.arange(array.dtype.itemsize,dtype=np.int64)
    coordinates=np.indices(array.shape,dtype=np.int64).reshape(array.ndim,-1)
    starts=pointer+np.asarray(array.strides,dtype=np.int64)@coordinates
    return (starts[:,None]+np.arange(array.dtype.itemsize,dtype=np.int64)[None,:]).ravel()


def _require_disjoint_storage(arrays):
    """Disallow concurrent restore writes to an aliased physical byte.

    A contiguous array occupies one closed-open address interval, so expanding
    every byte into an ``int64`` array is unnecessary.  Sort those intervals
    directly.  Strided views retain the exact byte-address fallback, including
    checks against every contiguous interval.  This preserves the original
    aliasing contract while avoiding O(total bytes) host memory and sorting for
    the normal all-contiguous solver inventory.
    """
    contiguous=[]
    strided=[]
    for array in arrays.values():
        if not array.nbytes:
            continue
        if array.flags.c_contiguous:
            start=int(array.data.ptr)
            contiguous.append((start,start+int(array.nbytes)))
        else:
            strided.append(_byte_addresses(array))

    contiguous.sort()
    if any(right[0] < left[1]
           for left,right in zip(contiguous,contiguous[1:])):
        raise ValueError('A1 complete transaction has overlapping GPU array storage')

    if not strided:
        return
    addresses=np.concatenate(strided)
    addresses.sort()
    if np.any(addresses[1:]==addresses[:-1]):
        raise ValueError('A1 complete transaction has overlapping GPU array storage')
    if contiguous:
        starts=np.asarray([item[0] for item in contiguous],dtype=np.int64)
        ends=np.asarray([item[1] for item in contiguous],dtype=np.int64)
        owners=np.searchsorted(starts,addresses,side='right')-1
        possible=owners>=0
        if np.any(addresses[possible] < ends[owners[possible]]):
            raise ValueError('A1 complete transaction has overlapping GPU array storage')


def wrap_solver(Parent):
    class CompleteA1TransactionSolver(Parent):
        def _a1_trial_diagnostics(self):
            saved=super()._a1_trial_diagnostics()
            diffusion=getattr(self,'diffusion',None)
            if diffusion is not None:
                if not hasattr(diffusion,'trial_metadata'):
                    raise ValueError('Positive viscosity requires transaction-aware diffusion metadata')
                saved['diffusion_metadata']=diffusion.trial_metadata()
            arrays=_unique_arrays(self)
            current=segmented.signature(arrays)
            plan=_PLANS.get(self)
            if plan is None or plan.signature!=current:
                _require_disjoint_storage(arrays)
                plan=segmented.SegmentedCopyPlan(self.cp,arrays)
                _PLANS[self]=plan
            plan.snapshot()
            saved['all_gpu_plan']=plan
            return saved

        def _a1_restore_diagnostics(self,saved,*,where=True):
            rollback=bool(where.item()) if isinstance(where,self.cp.ndarray) else bool(where)
            if rollback:
                # The status/CFL outputs alias scratch that full restoration
                # must put back.  Retain detached values for the caller.
                self._a1_trial_result=(self.stepper.status.copy(),self.stepper.cfl.copy())
                super()._a1_restore_diagnostics(saved,where=True)
                if 'diffusion_metadata' in saved:
                    self.diffusion.restore_trial_metadata(saved['diffusion_metadata'])
                saved['all_gpu_plan'].restore()
            else:
                self._a1_trial_result=(self.stepper.status,self.stepper.cfl)

        def step(self,dt,current,*,commit=True):
            try:
                result=super().step(dt,current,commit=commit)
                result['status'],result['cfl']=self._a1_trial_result
                return result
            finally:
                if hasattr(self,'_a1_trial_result'):del self._a1_trial_result

    return CompleteA1TransactionSolver
