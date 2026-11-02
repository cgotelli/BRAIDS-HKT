"""Transactional GPU inheritance of seven bank histories at an AMR split.

Matches the exact native old-covering-leaf copy rule for a selected four-child
refinement. This component does not choose topology or advance bank physics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')


class ResidentAMRBankHistoryProlong:
    def __init__(self, accepted_history, accepted_count, old_leaf,
                 *, source_path=None):
        import cupy as cp

        history = cp.asarray(accepted_history, dtype=cp.float64, order='C')
        raw_count = cp.asarray(accepted_count)
        old_leaf = np.asarray(old_leaf)
        if (history.ndim != 2 or history.shape[1] != 6 or
                len(history) < 5 or raw_count.shape != (len(history),) or
                raw_count.dtype.kind not in 'iu' or
                bool(cp.any(raw_count > np.iinfo(np.int32).max)) or
                bool(cp.any(raw_count < 0)) or
                not bool(cp.isfinite(history).all()) or
                bool(cp.any(history < 0))):
            raise ValueError('Finite nonnegative seven-field accepted bank history required')
        if (old_leaf.shape != (len(history),) or
                old_leaf.dtype.kind not in 'biu' or
                not np.isin(old_leaf, (0, 1)).all() or
                not np.any(old_leaf)):
            raise ValueError('Explicit native old-leaf map required')
        self.cp = cp
        self.nodes = len(history)
        self.old_leaf_host = np.ascontiguousarray(old_leaf, dtype=np.int32)
        self.old_leaf = cp.asarray(self.old_leaf_host)
        self.accepted = cp.array(history, copy=True, order='C')
        self.accepted_count = cp.asarray(raw_count, dtype=cp.int32).copy()
        self.trial = cp.empty_like(self.accepted)
        self.trial_count = cp.empty_like(self.accepted_count)
        self.status = cp.empty(self.nodes, dtype=cp.int32)
        self.pending = False
        self.selected = cp.empty(self.nodes, dtype=cp.int32)
        self.cover = cp.empty(self.nodes, dtype=cp.int32)
        self.source_path = Path(source_path or HERE / 'amr_bank_history_prolong.cu')
        module = cp.RawModule(code=self.source_path.read_text(encoding='utf-8'),
                              options=OPTIONS,
                              name_expressions=('amr_prolong_bank_history',))
        self.kernel = module.get_function('amr_prolong_bank_history')

    def propose(self, selected, cover):
        if self.pending:
            raise ValueError('Resolve previous bank-history proposal first')
        selected = np.asarray(selected)
        cover = np.asarray(cover)
        if (selected.shape != (self.nodes,) or
                selected.dtype.kind not in 'biu' or
                not np.isin(selected, (0, 1)).all() or
                cover.shape != (self.nodes,) or cover.dtype.kind not in 'iu' or
                np.any(cover > np.iinfo(np.int32).max) or
                np.any(cover < -1)):
            raise ValueError('Explicit finite selected/old-cover maps required')
        selected = np.ascontiguousarray(selected, dtype=np.int32)
        cover = np.ascontiguousarray(cover, dtype=np.int32)
        self.cp.copyto(self.selected, self.cp.asarray(selected))
        self.cp.copyto(self.cover, self.cp.asarray(cover))
        self.cp.copyto(self.trial, self.accepted)
        self.cp.copyto(self.trial_count, self.accepted_count)
        self.kernel(((self.nodes + 127) // 128,), (128,),
                    (self.accepted, self.accepted_count, self.selected,
                     self.cover, self.old_leaf, np.int32(self.nodes),
                     self.trial, self.trial_count, self.status))
        self.pending = True
        return self.trial, self.trial_count, self.status

    def commit(self):
        if not self.pending:
            raise ValueError('No pending bank-history proposal')
        if bool(self.cp.any(self.status != 0)):
            raise ValueError('Invalid bank-history inheritance proposal')
        self.accepted, self.trial = self.trial, self.accepted
        self.accepted_count, self.trial_count = self.trial_count, self.accepted_count
        self.pending = False
        return self.accepted, self.accepted_count

    def rollback(self):
        if not self.pending:
            raise ValueError('No pending bank-history proposal')
        self.pending = False
        return self.accepted, self.accepted_count

    def snapshot(self):
        """Read accepted persistent state for braids_support/checkpoint, never trial."""
        return (self.cp.asnumpy(self.accepted).copy(),
                self.cp.asnumpy(self.accepted_count).copy())
