"""Bounded Q4 accepted increment -> authoritative Q5a layered inventory bridge.

Q4 supplies a single already-integrated bed increment. There is no second
capacity, exchange, reconstruction or Exner evaluation. Q5b later replaces
every constitutive surface-mixture view; this bridge does not admit those routes.
"""
import numpy as np
from .bed_columns_v101 import host, digest


class CoupledColumnProjectionV101:
    def __init__(self, columns, parent, *, begin_transport, commit_transport,
                 rollback_transport, checkpoint_transport, restore_transport,
                 resident_delta=None):
        m = columns.runtime.methods
        if m.porosity not in (0, 5) or m.base_update != 1:
            raise ValueError('Q4 projection requires density-embedded porosity and finite independent base')
        if not np.allclose(host(columns.total), parent.active_state.base_state.bed_mass_kg_m2, atol=1e-12, rtol=0):
            raise ValueError('Layer total and Q4 authoritative stock differ')
        if not np.allclose(host(columns.bed), parent.bed_elevation_m, atol=1e-12, rtol=0):
            raise ValueError('Layer and hydraulic bed datums differ')
        if resident_delta is None and columns.runtime.backend.endswith('gpu'):
            raise ValueError('GPU layer projection requires original resident transport increment')
        self.columns, self.parent = columns, parent
        self.begin_transport, self.commit_transport = begin_transport, commit_transport
        self.rollback_transport, self.checkpoint_transport = rollback_transport, checkpoint_transport
        self.restore_transport, self.resident_delta = restore_transport, resident_delta
        self.pending = None

    def begin(self, *, cancelled=False):
        if self.pending is not None: raise ValueError('Coupled column projection is already open')
        if cancelled: raise InterruptedError('Coupled column update cancelled')
        before = self.checkpoint()
        trial = None
        try:
            trial = self.begin_transport(self.parent)
            inner = trial.active_trial.base_v93_trial
            delta = (inner.authoritative_bed_delta_kg_m2 if self.resident_delta is None
                     else self.resident_delta(trial))
            if self.resident_delta is not None:
                np.testing.assert_array_equal(host(delta), inner.authoritative_bed_delta_kg_m2)
            dt = trial.candidate.time_s-self.parent.time_s
            self.columns.begin(delta, trial.candidate.q[0], dt)
            out, sv = self.columns.pending[:2]
            xp = self.columns.xp
            bed = self.columns.rock+((out/xp.asarray(self.columns.runtime.rho)[None, :, None]).sum(axis=1)/sv).sum(axis=0)
            if not np.allclose(host(bed), trial.candidate.bed_elevation_m, atol=1e-12, rtol=0):
                raise ValueError('Layer bed feedback and Q4 hydraulic geometry differ')
            if not np.allclose(host(out.sum(axis=0)), trial.candidate.active_state.base_state.bed_mass_kg_m2, atol=1e-12, rtol=0):
                raise ValueError('Q4 increment was duplicated or stock projection differs')
            self.pending = (trial, before)
            return trial
        except BaseException:
            self.columns.rollback()
            if trial is not None: self.rollback_transport(self.parent, trial)
            raise

    def commit(self, *, cancelled=False, after_transport_publish=None):
        if self.pending is None: raise ValueError('No coupled column trial')
        if cancelled:
            self.rollback(); raise InterruptedError('Coupled column publication cancelled')
        trial, before = self.pending
        try:
            self.columns.validate(); self.columns.validate_pending()
            candidate = self.commit_transport(self.parent, trial)
            if after_transport_publish is not None: after_transport_publish()
            self.columns.commit(); self.parent = candidate; self.pending = None
        except BaseException:
            self.columns.rollback(); self.pending = None
            self.columns.restore(before['columns'])
            self.parent = self.restore_transport(before['transport'])
            raise
        return candidate

    def rollback(self):
        if self.pending is None: raise ValueError('No coupled column trial')
        trial, before = self.pending
        self.rollback_transport(self.parent, trial); self.columns.rollback(); self.pending = None

    def checkpoint(self):
        if self.pending is not None: raise ValueError('Checkpoint requires accepted coupled columns')
        d = {'scope': 'braids-q4-q5a-projection-v101', 'columns': self.columns.checkpoint(),
             'transport': self.checkpoint_transport(self.parent)}
        return {**d, 'sha256': digest(d)}

    def restore(self, d):
        if self.pending is not None: raise ValueError('Restore requires accepted coupled columns')
        raw = {k: v for k, v in d.items() if k != 'sha256'}
        if digest(raw) != d['sha256'] or d['scope'] != 'braids-q4-q5a-projection-v101':
            raise ValueError('Coupled column checkpoint identity differs')
        old = self.checkpoint()
        try:
            self.columns.restore(d['columns']); parent = self.restore_transport(d['transport'])
            if (not np.allclose(host(self.columns.total), parent.active_state.base_state.bed_mass_kg_m2, atol=1e-12, rtol=0)
                    or self.columns.time_s != parent.time_s):
                raise ValueError('Restart projected stock or clocks differ')
            self.parent = parent
        except BaseException:
            self.columns.restore(old['columns']); self.parent = self.restore_transport(old['transport']); raise
