"""Transactional, resident physical bank-toe Exner face-flux diagnostic.

This is distinct from lateral bank-material supply.  Its signed step-net is
the SSPRK33-weighted donor-limited Exner mass crossing each mapped physical
face, oriented bank-to-channel.  The gross accumulator adds the absolute value
of each *accepted step net*, matching the unchanged native implementation.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')
SSP33_WEIGHTS = (1. / 6., 1. / 6., 2. / 3.)


class ResidentToeFaceDiagnostic:
    def __init__(self, owner, neighbour, length, is_toe_face,
                 candidate_bank, candidate_toe, nfrac, *, source_path=None,
                 owner_orientation=False):
        import cupy as cp
        owner = np.asarray(owner)
        neighbour = np.asarray(neighbour)
        length = np.asarray(length, dtype=np.float64)
        flag = np.asarray(is_toe_face)
        bank = np.asarray(candidate_bank)
        toe = np.asarray(candidate_toe)
        nf = len(owner)
        if (nf < 1 or not isinstance(nfrac, (int, np.integer)) or not 1 <= nfrac <= 32 or
                any(v.shape != (nf,) for v in (neighbour, length, flag, bank, toe)) or
                any(v.dtype.kind not in 'iu' for v in (owner, neighbour, flag, bank, toe)) or
                not np.isin(flag, (0, 1)).all() or
                not np.isfinite(length).all() or np.any(length <= 0)):
            raise ValueError('Invalid physical bank-toe face diagnostic geometry')
        for f in range(nf):
            if flag[f]:
                if (bank[f] < 0 or toe[f] < 0 or bank[f] == toe[f] or
                        owner[f] < 0 or neighbour[f] < 0 or
                        not ((owner[f] == bank[f] and neighbour[f] == toe[f]) or
                        (neighbour[f] == bank[f] and owner[f] == toe[f]))):
                    raise ValueError('Physical bank-toe face orientation is invalid')
            elif bank[f] != -1 or toe[f] != -1:
                raise ValueError('Non-toe face has a physical mapping')
        self.cp = cp
        self.nf = nf
        self.ng = int(nfrac)
        if type(owner_orientation) is not bool:
            raise ValueError('Explicit boolean toe face orientation policy required')
        self.owner_orientation = owner_orientation
        self.owner = cp.asarray(np.ascontiguousarray(owner, dtype=np.int32))
        self.neighbour = cp.asarray(np.ascontiguousarray(neighbour, dtype=np.int32))
        self.length = cp.asarray(np.ascontiguousarray(length))
        self.is_toe_face = cp.asarray(np.ascontiguousarray(flag, dtype=np.int32))
        self.candidate_bank = cp.asarray(np.ascontiguousarray(bank, dtype=np.int32))
        self.candidate_toe = cp.asarray(np.ascontiguousarray(toe, dtype=np.int32))
        self.step_net = cp.zeros((self.ng, nf), dtype=cp.float64)
        self.signed = cp.zeros_like(self.step_net)
        self.gross = cp.zeros_like(self.step_net)
        self.failure = cp.zeros((), dtype=cp.int32)
        source = Path(source_path or HERE / 'toe_face_diagnostic.cu').read_text()
        if '#define BRAIDS_TOE_DIAGNOSTIC_ABI 2' not in source:
            raise ValueError('Toe diagnostic Python/CUDA ABI mismatch')
        names = ('toe_face_capture', 'toe_face_validate', 'toe_face_commit')
        module = cp.RawModule(code=source, options=OPTIONS, name_expressions=names)
        self.kernels = {name: module.get_function(name) for name in names}

    def _launch(self, name, args):
        count = self.ng * self.nf
        self.kernels[name](((count + 127) // 128,), (128,), args)

    def begin_trial(self):
        self.step_net.fill(0.)
        self.failure.fill(0)

    def capture(self, limited_flux, dt, stage):
        """Capture one Euler leg before the shared Exner buffer is overwritten."""
        if not isinstance(limited_flux, self.cp.ndarray) or \
                limited_flux.shape != self.step_net.shape or \
                limited_flux.dtype != self.cp.float64 or \
                not limited_flux.flags.c_contiguous:
            raise ValueError('Expected resident donor-limited grain/face flux')
        if stage not in range(3) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Native SSPRK33 diagnostic requires three positive-time stages')
        nf = np.int64(self.nf)
        count = np.int64(self.ng * self.nf)
        self._launch('toe_face_capture', (limited_flux, self.length, self.owner,
            self.neighbour, self.is_toe_face, self.candidate_bank,
            self.candidate_toe, nf, count, np.int32(self.owner_orientation), np.float64(dt),
            np.float64(SSP33_WEIGHTS[stage]), self.step_net, self.failure))
        return self.failure

    def set_accepted_face_integral(self, integral):
        """Install one completed root Exner register as the trial step net.

        ``integral`` is already SSPRK-weighted, time-integrated mass with the
        owner-to-neighbour sign convention.  Its two conservative traces must
        agree exactly.  This handoff deliberately applies no further time,
        face-length, or stage-weight factor.
        """
        cp = self.cp
        if (not isinstance(integral, cp.ndarray) or
                integral.shape != (self.ng, 2, self.nf) or
                integral.dtype != cp.float64 or
                not integral.flags.c_contiguous):
            raise ValueError('Expected contiguous accepted toe-face integral')
        if integral.device.id != self.step_net.device.id:
            raise ValueError('Accepted toe-face integral belongs to another CUDA device')

        lo, hi = integral.data.ptr, integral.data.ptr + integral.nbytes

        def arrays(value):
            if isinstance(value, cp.ndarray):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from arrays(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from arrays(item)

        for value in vars(self).values():
            for array in arrays(value):
                begin = array.data.ptr
                end = begin + sum((size - 1) * abs(stride)
                                  for size, stride in zip(array.shape, array.strides)) \
                    + array.dtype.itemsize
                if array.size and lo < end and begin < hi:
                    raise ValueError('Accepted toe-face integral overlaps private diagnostic storage')

        active = self.is_toe_face != 0
        forward = ((self.owner == self.candidate_bank) &
                   (self.neighbour == self.candidate_toe))
        reverse = ((self.neighbour == self.candidate_bank) &
                   (self.owner == self.candidate_toe))
        if (not bool(cp.all(cp.isfinite(integral)).item()) or
                not bool(cp.all(integral[:, 0, :] == integral[:, 1, :]).item())):
            raise ValueError('Accepted toe-face integral must be finite with equal signed traces')
        if not bool(cp.all((~active) | forward | reverse).item()):
            raise ValueError('Accepted toe-face integral has invalid physical orientation')

        sign = cp.where(self.owner_orientation | forward, 1., -1.)
        cp.copyto(self.step_net, cp.where(active[None, :],
                                         integral[:, 0, :] * sign[None, :], 0.))
        self.failure.fill(0)

    def validate_trial(self):
        self._launch('toe_face_validate', (self.step_net, self.signed,
            self.gross, np.int64(self.ng * self.nf), self.failure))
        return self.failure

    def commit(self, global_status):
        if not isinstance(global_status, self.cp.ndarray) or global_status.shape != () or \
                global_status.dtype != self.cp.int32:
            raise ValueError('Expected scalar resident transaction status')
        self._launch('toe_face_commit', (self.step_net, self.signed, self.gross,
            np.int64(self.ng * self.nf), global_status))
