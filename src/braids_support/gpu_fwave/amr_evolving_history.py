"""Private resident AMR history transfer for evolving ordinary-cell banks.

This is a new transfer rule, not the legacy native-max/area-history rule.
Physical layer masses are transferred elsewhere. The two cumulative histories
are removed-volume densities; they are audit counters, not extra sediment.
Geometric width memory is interface-length weighted, including dry support.
No fractional front or cavity-volume state is introduced by this component.
The planner must authenticate ancestry-compatible collinear support overlaps;
numeric conservation alone cannot establish their physical correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np

ABI = 1
RULE = 'support_length_weighted_width_v1'
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')
STATUS = {1: 'invalid input values', 2: 'invalid CSR association',
          4: 'cell overlap partition or duplicate association',
          8: 'lost or duplicated geometric support',
          16: 'integer event allocation or overflow',
          32: 'mixed collapse-trigger merge', 64: 'arithmetic overflow',
          128: 'global conservation failure'}


@dataclass(frozen=True)
class HistoryCandidate:
    history: object
    count: object
    status: object
    receipt: dict


class ResidentAMREvolvingHistory:
    """Reusable compiled kernels; each proposal owns fresh private outputs."""

    def __init__(self, *, source_path=None):
        import cupy as cp
        self.cp = cp
        self.source_path = Path(source_path or Path(__file__).with_suffix('.cu'))
        source = self.source_path.read_bytes()
        self.source_sha256 = hashlib.sha256(source).hexdigest()
        self.module = cp.RawModule(code=source.decode('utf-8'), options=OPTIONS)
        self.kernels = {name: self.module.get_function(name) for name in
                        ('eh_validate', 'eh_old_partition', 'eh_new_rows')}
        # CuPy exposes ufunc.reduce, but bitwise_or.reduce is unimplemented.
        self.status_or = cp.ReductionKernel('int32 x','int32 y','x','a | b',
                                            'y = a','0','eh_status_or')

    def _array(self, value, shape, dtype, name):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.shape != shape or
                value.dtype != dtype or not value.flags.c_contiguous or
                value.device.id != cp.cuda.Device().id):
            raise ValueError(f'{name}: resident contiguous {dtype} {shape} required')
        return value

    def _launch(self, name, size, args):
        if size:
            self.kernels[name](((size + 127) // 128,), (128,), args)

    def _inverse(self, offset, old, old_count):
        """Deterministic inverse CSR, including repeated support associations."""
        cp = self.cp
        rows = cp.repeat(cp.arange(len(offset)-1, dtype=cp.int32), cp.diff(offset))
        edge = cp.arange(len(old), dtype=cp.int32)
        order = cp.lexsort(cp.stack((edge, rows, old))).astype(cp.int32)
        counts = cp.bincount(old, minlength=old_count).astype(cp.int32)
        inverse = cp.empty(old_count+1, dtype=cp.int32)
        inverse[0] = 0
        inverse[1:] = cp.cumsum(counts, dtype=cp.int32)
        return inverse, order, rows

    def propose_history(self, old_history, old_count, old_area, new_area,
                        cell_offsets, cell_old, cell_weights,
                        support_offsets, support_oldroot, support_fraction,
                        old_live_length, *, critical_width,
                        old_collapse_exposure=None, new_collapse_exposure=None):
        """Return private ``HistoryCandidate``; caller alone commits globally.

        All arrays must be resident, contiguous, float64 or int32 as named.
        Cell weights are nonnegative geometric *overlap areas*, not fractions.
        Each nonempty association must have positive weight; duplicate old/new
        cell pairs are rejected. Support fractions distribute each old root's
        stored L over authentic face overlap. Repeated support pairs are legal.
        Old/new exposure, if supplied together, only reports the change in
        current geometric collapse potential; it is not a hidden acceptance
        tolerance. Mixed below/above-critical roots may not merge.
        """
        cp = self.cp
        if not isinstance(old_history, cp.ndarray) or old_history.ndim != 2:
            raise ValueError('Resident old history matrix required')
        no = old_history.shape[0]
        if not isinstance(new_area, cp.ndarray) or new_area.ndim != 1:
            raise ValueError('Resident new area vector required')
        nn = len(new_area)
        if not 0 < no < 2**31 or not 0 < nn < 2**31:
            raise ValueError('Nonempty int32-addressable old/new leaf sets required')
        if not np.isfinite(critical_width) or float(critical_width) <= 0:
            raise ValueError('Positive finite critical_width required')
        for value, shape, dtype, name in (
            (old_history,(no,6),cp.float64,'old_history'),
            (old_count,(no,),cp.int32,'old_count'),
            (old_area,(no,),cp.float64,'old_area'),
            (new_area,(nn,),cp.float64,'new_area'),
            (old_live_length,(no,),cp.float64,'old_live_length'),
            (cell_offsets,(nn+1,),cp.int32,'cell_offsets'),
            (support_offsets,(nn+1,),cp.int32,'support_offsets')):
            self._array(value,shape,dtype,name)
        for ids, weights, label in ((cell_old,cell_weights,'cell'),
                                    (support_oldroot,support_fraction,'support')):
            if not isinstance(ids,cp.ndarray) or ids.ndim != 1 or len(ids)>=2**31:
                raise ValueError(f'{label}: resident int32 association vector required')
            self._array(ids,(len(ids),),cp.int32,label+'_old')
            self._array(weights,(len(ids),),cp.float64,label+'_weights')
        has_exposure = old_collapse_exposure is not None
        if has_exposure != (new_collapse_exposure is not None):
            raise ValueError('Both old/new collapse-exposure arrays are required')
        if has_exposure:
            self._array(old_collapse_exposure,(no,),cp.float64,'old_collapse_exposure')
            self._array(new_collapse_exposure,(nn,),cp.float64,'new_collapse_exposure')
        else:
            old_collapse_exposure = cp.zeros(no,dtype=cp.float64)
            new_collapse_exposure = cp.zeros(nn,dtype=cp.float64)
        history = cp.zeros((nn,6),dtype=cp.float64)
        counts = cp.zeros(nn,dtype=cp.int32)
        status = cp.zeros((),dtype=cp.int32)
        ne, ns = len(cell_old), len(support_oldroot)
        size = max(no,nn+1,ne,ns)
        validity = cp.empty(size,dtype=cp.int32)
        self._launch('eh_validate',size,(np.int32(no),np.int32(nn),np.int32(ne),
            np.int32(ns),old_history,old_count,old_area,new_area,cell_offsets,
            cell_old,cell_weights,support_offsets,support_oldroot,support_fraction,
            old_live_length,np.int32(has_exposure),old_collapse_exposure,
            new_collapse_exposure,validity))
        status[...] = self.status_or(validity)
        # Do not construct repeat/sort operations from unvalidated CSR offsets.
        if int(status.item()) != 0:
            return HistoryCandidate(history,counts,status,self._receipt(status))
        coff, cedge, cnew = self._inverse(cell_offsets,cell_old,no)
        soff, sedge, _ = self._inverse(support_offsets,support_oldroot,no)
        assigned = cp.zeros(ne,dtype=cp.int64)
        old_status = cp.empty(no,dtype=cp.int32)
        self._launch('eh_old_partition',no,(np.int32(no),old_history,old_count,
            old_area,coff,cedge,cnew,cell_weights,soff,sedge,support_fraction,
            assigned,old_status))
        status[...] = self.status_or(old_status)
        new_status = cp.empty(nn,dtype=cp.int32)
        self._launch('eh_new_rows',nn,(np.int32(nn),old_history,old_area,new_area,
            cell_offsets,cell_old,cell_weights,assigned,support_offsets,
            support_oldroot,support_fraction,np.float64(critical_width),status,
            history,counts,new_status))
        status[...] = cp.bitwise_or(status,self.status_or(new_status))
        # Rejected rows intentionally have no candidate history. Do not turn a
        # known geometric/trigger veto into a spurious conservation failure by
        # comparing those zero placeholders with accepted inventories.
        if int(status.item()) != 0:
            return HistoryCandidate(history,counts,status,self._receipt(status))
        before = cp.stack((cp.sum(old_area*old_history[:,4]),
            cp.sum(old_area*old_history[:,5]),cp.sum(old_history[:,3]),
            cp.sum(old_history[:,3]*old_history[:,0])))
        after = cp.stack((cp.sum(new_area*history[:,4]),
            cp.sum(new_area*history[:,5]),cp.sum(history[:,3]),
            cp.sum(history[:,3]*history[:,0])))
        residual = after-before
        tolerance = 1024*np.finfo(float).eps*cp.maximum(cp.abs(before),cp.abs(after))
        bad = (cp.any(~cp.isfinite(before))|cp.any(~cp.isfinite(after))|
               cp.any(cp.abs(residual)>tolerance))
        old_events = cp.sum(old_count,dtype=cp.int64)
        new_events = cp.sum(counts,dtype=cp.int64)
        bad |= old_events != new_events
        status[...] = cp.bitwise_or(status,cp.where(bad,np.int32(128),np.int32(0)))
        receipt = self._receipt(status)
        names = ('toe_removed_volume','collapse_removed_volume',
                 'reference_length','length_weighted_undercut')
        receipt['conservation'] = {name: dict(before=self._number(a),
            after=self._number(b),residual=self._number(r))
            for name,a,b,r in zip(names,cp.asnumpy(before),cp.asnumpy(after),
                                  cp.asnumpy(residual))}
        receipt['event_count'] = dict(before=int(old_events.item()),
                                     after=int(new_events.item()))
        if has_exposure:
            old_potential = cp.sum(old_history[:,0]*old_history[:,3]*old_collapse_exposure)
            new_potential = cp.sum(history[:,0]*history[:,3]*new_collapse_exposure)
            if not bool((cp.isfinite(old_potential)&cp.isfinite(new_potential)).item()):
                status[...] = cp.bitwise_or(status,np.int32(64))
                receipt.update(self._receipt(status))
            receipt['collapse_potential'] = dict(
                before=self._number(old_potential.item()),
                after=self._number(new_potential.item()),
                change=self._number((new_potential-old_potential).item()),
                acceptance_limit=None)
        return HistoryCandidate(history,counts,status,receipt)

    @staticmethod
    def _number(value):
        return float(value) if np.isfinite(value) else None

    def _receipt(self,status):
        code = int(status.item())
        return dict(abi=ABI,rule=RULE,source_sha256=self.source_sha256,
            passed=code==0,status=code,
            failures=[text for bit,text in STATUS.items() if code&bit],
            candidate_only_no_accepted_mutation=True,
            arithmetic_and_count_allocation_on_CUDA=True,
            count_rule='per_old_largest_remainder_tie_new_index',
            physical_support_correspondence='caller_authenticated_overlap_required',
            native_history_equivalence_claim=False)


def propose_history(*args,source_path=None,**kwargs):
    """One-shot convenience; retain the class instance for repeated regrids."""
    return ResidentAMREvolvingHistory(source_path=source_path).propose_history(*args,**kwargs)
