"""Repair-free, two-substep, GPU-resident native AMR flux-register transaction.

Exact native scope: ``amr_subcycle_level`` accumulates physical fine SSPRK
integrals, replaces the coarse face integral on both traces, and corrects
only the coarse cell. This class takes already-accepted per-level integrals
and a post-level/pre-reflux state. It deliberately rejects any state needing
native water, grain, exchange, active-layer or prescribed-boundary repair.
It does not generate RK fluxes, topology, substep timing or boundary ledgers.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')


class ResidentAMRReflux:
    """One transaction for native 2:1 level faces and classwise flux, no exchange.

    Accepted/trial state: ``(cell,state)`` in native row order: h, hu, hv,
    classwise suspended rows, active/substrate grain masses, and optionally
    one signed prognostic-secondary water inventory. The connected
    interface face list is a stable ascending native face order. A face side
    is 0 when coarse is owner, 1 when coarse is neighbour. Flux arrays retain
    both independent native owner/neighbour traces; the appropriate one alone
    corrects the coarse cell. The physical time integral is supplied by the
    caller (not recalculated from an instantaneous flux).
    """

    def __init__(self, post_level_state, area, coarse_cells, face_offset,
                 reflux_faces, coarse_side, face_count, *, grains=7,
                 prognostic_secondary=False, source_path=None):
        import cupy as cp
        resident_input = isinstance(post_level_state, cp.ndarray)
        state = post_level_state if resident_input else np.asarray(post_level_state)
        area = np.asarray(area, dtype=np.float64)
        groups = np.asarray(coarse_cells, dtype=np.int32)
        offset = np.asarray(face_offset, dtype=np.int32)
        face = np.asarray(reflux_faces, dtype=np.int32)
        side = np.asarray(coarse_side, dtype=np.int32)
        self.prognostic_secondary = bool(prognostic_secondary)
        physical_states = state.shape[1] - int(self.prognostic_secondary) if state.ndim == 2 else 0
        if (state.ndim != 2 or state.dtype != np.float64 or
                not state.flags.c_contiguous or not 1 <= grains <= 32 or
                (physical_states-3) % grains or
                not 1 <= (physical_states-3)//grains-1 <= 32):
            raise ValueError('Contiguous native FP64, 1-32-grain layered state required')
        self.cells, self.states = state.shape
        self.physical_states = physical_states
        self.grains = int(grains)
        self.layers = (self.physical_states-3)//self.grains-1
        self.faces = int(face_count)
        self.groups = len(groups)
        if self.cells < 3 or self.faces < 2 or self.groups < 1:
            raise ValueError('Nonempty two-level four-child interface fixture required')
        if area.shape != (self.cells,) or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError('Finite positive active-cell areas required')
        if offset.shape != (self.groups+1,) or offset[0] != 0 or offset[-1] != len(face) or np.any(np.diff(offset) <= 0):
            raise ValueError('Each coarse cell needs a nonempty ordered reflux-face group')
        if len(face) != len(side) or len(face) != self.faces or len(set(face.tolist())) != len(face):
            raise ValueError('Every interface face must appear exactly once')
        if (np.any(face < 0) or np.any(face >= self.faces) or np.any((side != 0) & (side != 1)) or
                np.any(groups < 0) or np.any(groups >= self.cells) or len(set(groups.tolist())) != len(groups)):
            raise ValueError('Invalid coarse-cell, face or side index')
        for p in range(self.groups):
            if np.any(np.diff(face[offset[p]:offset[p+1]]) <= 0):
                raise ValueError('Reflux faces must retain native ascending face order')
        xp = cp if resident_input else np
        if (not bool(xp.isfinite(state).all()) or bool(xp.any(state[:, 0] < 0)) or
                bool(xp.any(state[:, 3:self.physical_states] < 0)) or
                bool(xp.any((state[:, 0] == 0) &
                            xp.any(state[:, 3:3+self.grains] != 0, axis=1))) or
                (self.prognostic_secondary and
                 bool(xp.any((state[:, 0] == 0) &
                             (state[:, self.physical_states] != 0))))):
            raise ValueError('Invalid or dry-cell suspended input stock')
        self.cp = cp
        self.area = cp.asarray(np.ascontiguousarray(area))
        self.coarse_cells = cp.asarray(np.ascontiguousarray(groups))
        self.offset = cp.asarray(np.ascontiguousarray(offset))
        self.face = cp.asarray(np.ascontiguousarray(face))
        self.side = cp.asarray(np.ascontiguousarray(side))
        self.accepted = cp.array(state,copy=True,order='C')
        self.trial = cp.empty_like(self.accepted)
        self.accepted_register = cp.zeros((self.faces, 2, self.states),dtype=cp.float64)
        self.trial_register = cp.empty_like(self.accepted_register)
        self.status = cp.empty(max(self.cells, self.faces), dtype=cp.int32)
        self.pending = False
        self.source_path = Path(source_path or HERE/'amr_reflux.cu')
        module = cp.RawModule(code=self.source_path.read_text(encoding='utf-8'),
            options=OPTIONS,
            name_expressions=('amr_fine_register','amr_reflux_coarse','amr_validate_reflux'))
        self.register_kernel = module.get_function('amr_fine_register')
        self.reflux_kernel = module.get_function('amr_reflux_coarse')
        self.validate_kernel = module.get_function('amr_validate_reflux')

    @staticmethod
    def _launch(kernel, size, arguments):
        kernel(((size+127)//128,), (128,), arguments)

    def propose(self, coarse_integral, fine_substep_integrals):
        cp = self.cp
        if self.pending:
            raise ValueError('Resolve previous reflux transaction first')
        if (not isinstance(coarse_integral, cp.ndarray) or
                coarse_integral.shape != (self.faces,2,self.states) or
                coarse_integral.dtype != cp.float64 or not coarse_integral.flags.c_contiguous):
            raise ValueError('Resident contiguous native coarse integrated face flux required')
        if (not isinstance(fine_substep_integrals, cp.ndarray) or
                fine_substep_integrals.shape != (2,self.faces,2,self.states) or
                fine_substep_integrals.dtype != cp.float64 or not fine_substep_integrals.flags.c_contiguous):
            raise ValueError('Exactly two resident integrated fine substeps required')
        cp.copyto(self.trial,self.accepted)
        self._launch(self.register_kernel,self.faces*2*self.states,
                     (fine_substep_integrals,np.int32(self.faces),np.int32(self.states),self.trial_register))
        self._launch(self.reflux_kernel,self.groups*self.states,
                     (coarse_integral,self.trial_register,self.coarse_cells,self.offset,
                      self.face,self.side,self.area,np.int32(self.groups),
                      np.int32(self.states),self.trial))
        self._launch(self.validate_kernel,max(self.cells,self.faces),
                     (self.trial,self.trial_register,coarse_integral,fine_substep_integrals,
                      np.int32(self.cells),np.int32(self.faces),
                      np.int32(self.states),np.int32(self.grains),
                      np.int32(self.physical_states),
                      np.int32(self.prognostic_secondary),self.status))
        self.pending = True
        return self.trial,self.trial_register,self.status

    def commit(self):
        if not self.pending:
            raise ValueError('No pending reflux proposal')
        if bool(self.cp.any(self.status != 0)):
            raise ValueError('Native repair would be required; reject this bounded GPU reflux')
        self.accepted,self.trial=self.trial,self.accepted
        self.accepted_register,self.trial_register=self.trial_register,self.accepted_register
        self.pending=False
        return self.accepted,self.accepted_register

    def rollback(self):
        if not self.pending:
            raise ValueError('No pending reflux proposal')
        self.pending=False
        return self.accepted,self.accepted_register
