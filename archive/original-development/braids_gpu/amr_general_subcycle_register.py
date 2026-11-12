"""Research-only, schema-driven CUDA face registers for general nonbank AMR.

This is the conservative interface between level-local Euler stages and a
recursive 2:1 clock.  It does not choose a time integrator or silently omit a
physical process.  The native state layout, including every substrate layer,
is taken from an authenticated selected operator rather than a fixed fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .amr_bank_subcycle import level_topology


HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')


def require(ok, message):
    if not ok:
        raise ValueError(message)


class GeneralStateLayout:
    """Native row mapping for water, classes, bed strata and signed hI."""

    def __init__(self, data):
        self.nc = int(data['nc'])
        self.nf = int(data['nf'])
        self.ns = int(data['nstate'])
        self.ng = int(data['nfrac'])
        self.nl = int(data['nbed_layers'])
        self.suspended = np.asarray(data['suspended_indices'], dtype=np.int32).copy()
        self.bed = np.asarray(data['bed_indices'], dtype=np.int32).copy()
        self.secondary = int(data['secondary_index'])
        require(self.nc > 0 and self.nf > 0 and self.ng > 0 and self.nl > 0 and
                self.suspended.shape == (self.ng,) and
                self.bed.shape == (self.nl, self.ng) and
                self.secondary in (-1, self.ns-1),
                'Invalid general AMR native state dimensions')
        indices = np.concatenate((np.arange(3, dtype=np.int32), self.suspended,
                                  self.bed.ravel(),
                                  np.asarray([self.secondary], np.int32)
                                  if self.secondary >= 0 else np.empty(0, np.int32)))
        require(self.ns == len(indices) and
                np.array_equal(np.sort(indices), np.arange(self.ns)),
                'General AMR state rows must partition the entire native state')

    def class_inventory(self, rows):
        return rows[:, self.suspended] + rows[:, self.bed].sum(axis=1)


class GeneralFaceRegister:
    """CUDA-resident all-face integral and transactional coarse-cell reflux.

    ``integral`` is ``(native face, owner/neighbour trace, native state row)``.
    Interface registers are never reduced on the host.  Substrate layers have
    zero advective face flux, but remain part of the corrected state/validation.
    A proposed reflux stays private until ``commit``; ``rollback`` discards it.
    """

    def __init__(self, data, node_levels, *, source_path=None):
        import cupy as cp

        self.cp = cp
        self.schema = GeneralStateLayout(data)
        self.levels = np.asarray(node_levels, dtype=np.int32).copy()
        require(self.levels.shape == (self.schema.nc,) and
                np.all(self.levels >= 0), 'Invalid selected AMR cell levels')
        self.layout = level_topology(data['owner'], data['neighbour'], self.levels)
        self.device_layout = [{name: cp.asarray(value) for name, value in row.items()}
                              for row in self.layout]
        self.area = cp.asarray(data['area'], dtype=cp.float64)
        self.length = cp.asarray(data['length'], dtype=cp.float64)
        require(bool(cp.all(cp.isfinite(self.area))) and
                bool(cp.all(self.area > 0)) and
                bool(cp.all(cp.isfinite(self.length))) and
                bool(cp.all(self.length > 0)), 'Invalid AMR face/cell measures')
        self.source_path = Path(source_path or HERE/'amr_bank_subcycle.cu').resolve()
        code = self.source_path.read_text('utf-8')
        require('#define BRAIDS_BANK_SUBCYCLE_REGISTER_ABI 1' in code,
                'General AMR reflux CUDA source ABI differs')
        module = cp.RawModule(code=code, options=OPTIONS, name_expressions=('bs_reflux',))
        self.kernel = module.get_function('bs_reflux')
        self._pending = None

    @property
    def pending(self):
        return self._pending is not None

    def empty(self):
        s = self.schema
        return self.cp.zeros((s.nf, 2, s.ns), dtype=self.cp.float64)

    def face_integral(self, solver, dt):
        """Map accepted CUDA stage face fluxes into all 39 (or other) rows."""
        cp, s = self.cp, self.schema
        require(np.isfinite(dt) and dt > 0 and
                solver.nc == s.nc and solver.nf == s.nf and solver.ng == s.ng and
                int(solver.data['nstate']) == s.ns,
                'General AMR Euler flux and selected operator differ')
        result = self.empty()
        factor = cp.float64(dt)*self.length
        fl, fr = solver.flux.buffers['fl'], solver.flux.buffers['fr']
        result[:, 0, :3] = factor[:, None]*fl.T
        result[:, 1, :3] = factor[:, None]*fr.T
        if solver.bedload_active:
            bed = factor[:, None]*solver.exner.buffers['flux'].T
            result[:, :, s.bed[0]] = bed[:, None, :]
        if solver.has_suspended:
            suspended = factor[:, None]*solver.suspended_advection.buffers['flux'].T
            result[:, :, s.suspended] = suspended[:, None, :]
        if s.secondary >= 0:
            secondary = factor*solver.secondary.buffers['flux']
            result[:, :, s.secondary] = secondary[:, None]
        return result

    @staticmethod
    def rk33(first, second, third):
        require(first.shape == second.shape == third.shape and
                first.dtype == second.dtype == third.dtype,
                'General AMR SSPRK33 face-integral shapes differ')
        return (first+second)/6. + (2./3.)*third

    def propose(self, rows, coarse, fine, level):
        """Correct current coarse cells by fine-minus-coarse flux history."""
        cp, s = self.cp, self.schema
        require(not self.pending and type(level) is int and
                0 <= level < len(self.layout)-1 and
                all(isinstance(value, cp.ndarray) and
                    value.dtype == cp.float64 and value.flags.c_contiguous
                    for value in (rows, coarse, fine)) and
                rows.shape == (s.nc, s.ns) and
                coarse.shape == fine.shape == (s.nf, 2, s.ns),
                'Invalid pending general AMR reflux input')
        trial = rows.copy()
        schedule = self.device_layout[level]
        groups = int(schedule['coarse_cells'].size)
        if groups:
            self.kernel(((groups*s.ns+127)//128,), (128,),
                        (trial, coarse, fine, schedule['coarse_cells'],
                         schedule['offset'], schedule['faces'], schedule['sides'],
                         self.area, np.int32(groups), np.int32(s.ns)))
        positive = cp.asarray(np.concatenate((s.suspended, s.bed.ravel())))
        valid = (bool(cp.all(cp.isfinite(trial))) and
                 bool(cp.all(trial[:, 0] >= -1e-12)) and
                 bool(cp.all(trial[:, positive] >= -1e-12)))
        if not valid:
            raise RuntimeError('General AMR reflux needs conservative positivity repair')
        self._pending = trial
        return trial

    def commit(self):
        require(self.pending, 'No pending general AMR reflux proposal')
        value, self._pending = self._pending, None
        return value

    def rollback(self):
        require(self.pending, 'No pending general AMR reflux proposal')
        self._pending = None
