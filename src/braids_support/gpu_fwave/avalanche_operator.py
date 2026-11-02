"""Transactional resident GPU general-avalanche substep composition.

This composes the native-order face proposal, per-grain donor limiter/gather,
and ordered active-layer rebalance. The fixed-grid coupled solver invokes it
at its qualified integration point; recursive nonbank AMR invokes the same
operator once on the synchronized full-leaf state after parent reflux.
Hydraulic state and the supplied accepted bed layers are never changed by
``step``.
"""
from __future__ import annotations

import numpy as np


class ResidentGeneralAvalanche:
    """Host-controlled diffusion substeps over GPU-resident mass and geometry.

    Only the scalar substep duration and failure latch cross to the host. This
    variable-substep loop is deliberately outside CUDA graph capture; a future
    fixed graph path requires separate qualification. A nonzero status means
    the caller must discard the returned trial and retry or stop.
    """

    def __init__(self, proposal, transfer, stratigraphy, rock, nlayer,
                 *, maximum_substeps=100000):
        cp = proposal.cp
        if proposal.cp is not transfer.cp or proposal.cp is not stratigraphy.cp:
            raise ValueError('All avalanche components must use the same GPU runtime')
        if proposal.nc != transfer.nc or proposal.ng != transfer.ng or \
                proposal.nc != stratigraphy.nc or proposal.ng != stratigraphy.ng:
            raise ValueError('Avalanche component geometry/class dimensions differ')
        if not isinstance(nlayer, (int, np.integer)) or nlayer < 2 or nlayer > 33:
            raise ValueError('Expected active plus 1-32 substrate layers')
        if getattr(stratigraphy, 'ns', nlayer - 1) != nlayer - 1:
            raise ValueError('Stratigraphy layer count differs')
        if not isinstance(maximum_substeps, (int, np.integer)) or maximum_substeps < 1:
            raise ValueError('Positive avalanche substep limit required')
        if not isinstance(rock, cp.ndarray) or rock.shape != (proposal.nc,) or \
                rock.dtype != cp.float64 or not rock.flags.c_contiguous:
            raise ValueError('Contiguous FP64 device bedrock required')
        self.cp = cp; self.nc = proposal.nc; self.ng = proposal.ng
        self.nl = int(nlayer); self.maximum_substeps = int(maximum_substeps)
        self.proposal = proposal; self.transfer = transfer
        self.stratigraphy = stratigraphy; self.rock = rock
        self.layers = cp.empty((self.nl, self.ng, self.nc), cp.float64)
        self.bed = cp.empty(self.nc, cp.float64)
        self.bed_status = cp.empty(self.nc, cp.int32)
        self.status = cp.zeros((), cp.int32)
        self._identity = tuple((array.data.ptr, array.shape) for array in
                               (self.layers, self.bed, self.bed_status, self.status, self.rock))

    def _bad(self, *arrays):
        cp = self.cp
        bad = cp.zeros((), cp.int32)
        for array in arrays:
            cp.maximum(bad, cp.max(array), out=bad)
        return bool(bad.item())

    def _bed(self):
        result = self.stratigraphy.bed_elevation(self.layers, self.rock, out=self.bed)
        if isinstance(result, dict):
            return result['status']
        if isinstance(result, self.cp.ndarray):
            # The qualified one-substrate transport kernel returns its bed
            # directly; unlike the multilayer kernel it has no status row.
            self.cp.copyto(self.bed_status,
                self.cp.where(self.cp.isfinite(result),self.cp.int32(0),self.cp.int32(1)))
            return self.bed_status
        status = getattr(self.stratigraphy, 'bed_status', None)
        if status is None and hasattr(self.stratigraphy, 'buffers'):
            status = self.stratigraphy.buffers.get('bed_status')
        if status is None:
            raise ValueError('Bed-elevation operator must expose a device status')
        return status

    def step(self, accepted_layers, depth, dt, *, front_active=None):
        cp = self.cp
        if (not isinstance(accepted_layers, cp.ndarray) or
                accepted_layers.shape != self.layers.shape or
                accepted_layers.dtype != cp.float64 or
                not accepted_layers.flags.c_contiguous or
                not isinstance(depth, cp.ndarray) or depth.shape != (self.nc,) or
                depth.dtype != cp.float64 or not depth.flags.c_contiguous):
            raise ValueError('Contiguous resident avalanche layer/depth state required')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Finite positive avalanche duration required')
        if tuple((array.data.ptr, array.shape) for array in
                 (self.layers, self.bed, self.bed_status, self.status, self.rock)) != self._identity:
            raise ValueError('Resident avalanche storage changed')
        cp.copyto(self.layers, accepted_layers)
        self.status.fill(0)
        remaining = float(dt); substeps = 0
        threshold = 64 * np.finfo(float).eps * max(float(dt), 1.)
        while remaining > threshold:
            if substeps >= self.maximum_substeps:
                self.status.fill(1)
                break
            bed_status = self._bed()
            if self._bad(bed_status):
                self.status.fill(1)
                break
            proposal = self.proposal.step(self.bed, depth, self.layers[0],
                                          remaining, front_active=front_active)
            if self._bad(*proposal['status']):
                self.status.fill(1)
                break
            subdt = float(proposal['subdt'].item())
            if subdt == 0.:
                break  # Native operator stops when the maximum cell rate is zero.
            if not np.isfinite(subdt) or subdt <= 0. or subdt > remaining:
                self.status.fill(1)
                break
            moved = self.transfer.step(self.layers[0], proposal['proposed'])
            if self._bad(moved['cell_status'], moved['face_status']):
                self.status.fill(1)
                break
            cp.copyto(self.layers[0], moved['next'])
            rebalance_status = self.stratigraphy.rebalance(self.layers)
            if self._bad(rebalance_status):
                self.status.fill(1)
                break
            remaining = max(0., remaining - subdt)
            substeps += 1
        if not int(self.status.item()):
            if self._bad(self._bed()):
                self.status.fill(1)
        return dict(layers=self.layers, bed=self.bed, status=self.status,
                    substeps=substeps, remaining_s=remaining,
                    accepted_unchanged=True)
