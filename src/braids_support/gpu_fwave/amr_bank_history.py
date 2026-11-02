"""GPU-resident, exact-native bank-history restriction transaction.

This component matches ``amr_sync_bank_failure_state`` for a four-child
parent with persistent-front tracking disabled. It does not select topology,
expose newly refined leaves, or advance bank erosion on an AMR mesh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')


class ResidentAMRBankHistory:
    def __init__(self, weights, accepted_parent, accepted_count,
                 *, source_path=None):
        import cupy as cp

        weight = np.asarray(weights, dtype=np.float64)
        if (weight.ndim != 2 or weight.shape[1] != 4 or
                not np.isfinite(weight).all() or np.any(weight <= 0)):
            raise ValueError('Finite positive four-child areas required')
        self.parents = len(weight)
        if self.parents < 1:
            raise ValueError('At least one parent required')
        parent = cp.asarray(accepted_parent, dtype=cp.float64)
        raw_counts = cp.asarray(accepted_count)
        if (raw_counts.dtype.kind not in 'iu' or
                bool(cp.any(raw_counts > np.iinfo(np.int32).max))):
            raise ValueError('Accepted bank collapse counts must be integers')
        counts = cp.asarray(raw_counts, dtype=cp.int32)
        if (parent.shape != (self.parents, 6) or
                counts.shape != (self.parents,) or
                not bool(cp.isfinite(parent).all()) or
                bool(cp.any(parent < 0)) or bool(cp.any(counts < 0))):
            raise ValueError('Valid accepted seven-field bank history required')
        self.cp = cp
        self.weight = cp.asarray(np.ascontiguousarray(weight))
        self.accepted = cp.array(parent, copy=True, order='C')
        self.accepted_count = cp.array(counts, copy=True, order='C')
        self.trial = cp.empty_like(self.accepted)
        self.trial_count = cp.empty_like(self.accepted_count)
        self.status = cp.empty(self.parents, dtype=cp.int32)
        self.pending = False
        self.source_path = Path(source_path or HERE / 'amr_bank_history.cu')
        module = cp.RawModule(code=self.source_path.read_text(encoding='utf-8'),
                              options=OPTIONS,
                              name_expressions=('amr_restrict_bank_history',))
        self.kernel = module.get_function('amr_restrict_bank_history')

    def propose(self, children, counts):
        cp = self.cp
        if self.pending:
            raise ValueError('Resolve previous bank-history proposal first')
        if (not isinstance(children, cp.ndarray) or
                children.shape != (self.parents, 4, 6) or
                children.dtype != cp.float64 or not children.flags.c_contiguous or
                not isinstance(counts, cp.ndarray) or
                counts.shape != (self.parents, 4) or
                counts.dtype != cp.int32 or not counts.flags.c_contiguous):
            raise ValueError('Resident contiguous four-child bank histories required')
        cp.copyto(self.trial, self.accepted)
        cp.copyto(self.trial_count, self.accepted_count)
        self.kernel(((self.parents + 127) // 128,), (128,),
                    (children, counts, self.weight, np.int32(self.parents),
                     self.trial, self.trial_count, self.status))
        self.pending = True
        return self.trial, self.trial_count, self.status

    def commit(self):
        if not self.pending:
            raise ValueError('No pending bank-history proposal')
        if bool(self.cp.any(self.status != 0)):
            raise ValueError('Invalid bank-history proposal')
        self.accepted, self.trial = self.trial, self.accepted
        self.accepted_count, self.trial_count = self.trial_count, self.accepted_count
        self.pending = False
        return self.accepted, self.accepted_count

    def rollback(self):
        if not self.pending:
            raise ValueError('No pending bank-history proposal')
        self.pending = False
        return self.accepted, self.accepted_count
