"""Resident, stage-local GPU solve for native classwise bed-load adaptation.

The CPU closure solves L_g dq_g/ds = q_equilibrium,g - q_g by Jacobi at every
Euler/RK stage. Its solution is derived from the current hydraulic state and
current boundary feed, so neither capacity nor Jacobi work is accepted state.
The caller supplies the cell-centred equilibrium capacity after any secondary
direction rotation, then reconstructs gradients from the relaxed result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')
INLET_MODES = (1, 6, 7, 8, 9)


def _scalar(data, name):
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind not in 'fiu' or not np.isfinite(value):
        raise ValueError('Invalid adaptation control ' + name)
    return float(value)


class ResidentBedloadAdaptation:
    """Fixed-grid GPU Jacobi closure, with no persistent/restart state."""

    def __init__(self, data, *, source_path=None, max_iterations=4096):
        import cupy as cp

        self.cp = cp
        self.nc = int(np.asarray(data['q']).shape[1])
        self.ng = len(np.asarray(data['d50_fraction']))
        self.nf = len(np.asarray(data['owner']))
        self.na = len(np.asarray(data['faces']))
        self.nb = len(np.asarray(data['mode']))
        self.length_diameters = _scalar(data, 'bedload_adaptation_length_diameters')
        self.dry = _scalar(data, 'dry_tolerance')
        self.dx = _scalar(data, 'dx')
        self.dy = _scalar(data, 'dy')
        if (not bool(data.get('bedload_adaptation', False)) or
                int(np.asarray(data['bedlaw']).item()) != 2 or
                self.length_diameters <= 0 or self.dry <= 0 or
                self.dx <= 0 or self.dy <= 0):
            raise ValueError('Adaptation requires active MPM and positive controls')
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError('Positive integer adaptation iteration limit required')
        self.max_iterations = max_iterations

        float_shapes = dict(xy=(2, self.nc), mid=(2, self.nf),
                            normal=(2, self.nf), length=(self.nf,),
                            area=(self.nc,), d50_fraction=(self.ng,))
        int_shapes = dict(owner=(self.nf,), neighbour=(self.nf,), bid=(self.nf,),
                          offset=(self.nc+1,), faces=(self.na,), signs=(self.na,),
                          mode=(self.nb,), bed_boundary_mode=(self.nb,))
        host = {}
        for name, shape in {**float_shapes, **int_shapes}.items():
            value = np.asarray(data[name])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid adaptation geometry ' + name)
            if name in int_shapes:
                if value.dtype.kind not in 'iu' or np.any(value > np.iinfo(np.int32).max):
                    raise ValueError('Integer adaptation topology required: ' + name)
                host[name] = np.ascontiguousarray(value, dtype=np.int32)
            else:
                host[name] = np.ascontiguousarray(value, dtype=np.float64)
        if (np.any(host['d50_fraction'] <= 0) or np.any(host['area'] <= 0) or
                np.any(host['length'] <= 0) or
                np.any((host['owner'] < 0) | (host['owner'] >= self.nc)) or
                np.any((host['neighbour'] < -1) | (host['neighbour'] >= self.nc)) or
                np.any((host['bid'] < -1) | (host['bid'] >= self.nb)) or
                np.any((host['bid'] >= 0) & (host['neighbour'] >= 0)) or
                not np.isin(host['signs'], (-1, 1)).all() or
                np.any((host['faces'] < 0) | (host['faces'] >= self.nf)) or
                host['offset'][0] != 0 or host['offset'][-1] != self.na or
                np.any(np.diff(host['offset']) < 0) or
                not np.isin(host['bed_boundary_mode'], (0, 1)).all()):
            raise ValueError('Invalid adaptation geometry/topology')
        for c in range(self.nc):
            ff = host['faces'][host['offset'][c]:host['offset'][c+1]]
            ss = host['signs'][host['offset'][c]:host['offset'][c+1]]
            if np.any(np.where(ss > 0, host['owner'][ff], host['neighbour'][ff]) != c):
                raise ValueError('Adaptation adjacency belongs to another cell')
        span = np.bincount(host['bid'][host['bid'] >= 0],
                           weights=host['length'][host['bid'] >= 0],
                           minlength=self.nb).astype(np.float64)
        self.geometry = {name: cp.asarray(value) for name, value in host.items()}
        self.geometry['boundary_span'] = cp.asarray(span)
        self.work = dict(equilibrium=cp.empty((2, self.ng, self.nc), dtype=cp.float64),
                         first=cp.empty((2, self.ng, self.nc), dtype=cp.float64),
                         second=cp.empty((2, self.ng, self.nc), dtype=cp.float64),
                         face_weight=cp.empty(self.na, dtype=cp.float64),
                         total_weight=cp.empty(self.nc, dtype=cp.float64),
                         distance=cp.empty(self.nc, dtype=cp.float64),
                         change=cp.empty((self.ng, self.nc), dtype=cp.float64))
        code = Path(source_path or HERE/'bedload_adaptation.cu').read_text(encoding='utf-8')
        self.module = cp.RawModule(code=code, options=OPTIONS,
                                   name_expressions=('adapt_prepare', 'adapt_jacobi'))
        self.prepare_kernel = self.module.get_function('adapt_prepare')
        self.jacobi_kernel = self.module.get_function('adapt_jacobi')
        self.iterations = 0
        self.residual = float('nan')

    def _array(self, value, shape):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float64 or
                value.shape != shape or not value.flags.c_contiguous):
            raise ValueError('Expected contiguous GPU adaptation array ' + str(shape))

    def relax(self, q, capacity, feed):
        """Replace stage capacity in place only after a converged solve.

        Inputs are device arrays q(3,nc), capacity(2,ng,nc), feed(ng,nb).
        On failure, capacity is unchanged. The returned diagnostics are for
        this stage only and are not checkpoint values.
        """
        cp, g, w = self.cp, self.geometry, self.work
        self._array(q, (3, self.nc))
        self._array(capacity, (2, self.ng, self.nc))
        self._array(feed, (self.ng, self.nb))
        cp.copyto(w['equilibrium'], capacity)
        self.iterations = 0
        self.residual = float('nan')
        scale = max(float(cp.max(cp.abs(w['equilibrium'])).item()), 1.e-20)
        tolerance = 1.e-11*scale
        self.prepare_kernel(((self.nc+127)//128,), (128,),
            (np.int32(self.nc), np.int32(self.nf), q, g['xy'], g['mid'],
             g['normal'], g['length'], g['area'], g['owner'], g['neighbour'],
             g['bid'], g['offset'], g['faces'], g['signs'], g['mode'],
             np.float64(self.dry), np.float64(self.dx), np.float64(self.dy),
             w['face_weight'], w['total_weight'], w['distance']))
        previous, following = w['equilibrium'], w['first']
        for iteration in range(1, self.max_iterations+1):
            self.jacobi_kernel(((self.nc*self.ng+127)//128,), (128,),
                (np.int32(self.nc), np.int32(self.ng), np.int32(self.nf),
                 np.int32(self.nb), previous, w['equilibrium'], g['d50_fraction'],
                 g['normal'], g['owner'], g['neighbour'], g['bid'], g['offset'],
                 g['faces'], g['signs'], g['bed_boundary_mode'],
                 g['boundary_span'], feed, w['face_weight'], w['total_weight'],
                 w['distance'], np.float64(self.length_diameters), following,
                 w['change']))
            residual = float(cp.max(w['change']).item())
            if not np.isfinite(residual):
                raise RuntimeError('Invalid GPU adapted bed-load capacity')
            self.iterations, self.residual = iteration, residual
            if residual <= tolerance:
                cp.copyto(capacity, following)
                return {'iterations': iteration, 'residual': residual,
                        'tolerance': tolerance}
            previous, following = following, (w['second'] if following is w['first'] else w['first'])
        raise RuntimeError('GPU bed-load adaptation solve did not converge')
