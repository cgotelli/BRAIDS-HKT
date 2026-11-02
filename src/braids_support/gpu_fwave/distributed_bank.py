"""Resident mode-3 accepted-flux slope-bank transfer and local collapse.

The face mass is the signed, donor-limited Exner flux from all three SSPRK33
Euler legs. Host orchestration reads only two integer wave-progress counters;
face/cell physics and donor scheduling remain resident parallel CUDA work.
Accepted fields and histories are never changed by a rejected trial.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .surface_transfer import ResidentSurfaceTransfer, validate_geometry

HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')
SSP33_WEIGHTS = (1. / 6., 1. / 6., 2. / 3.)
FLOAT_HISTORY = ('undercut', 'reference_height', 'toe_height',
                 'reference_length', 'toe_cumulative', 'collapse_cumulative')
LOCAL_FACE_SLOPE = 'accepted_flux_distributed_slope'
PUBLISHED_NET_GRADIENT = 'accepted_flux_published_net_gradient'


class ResidentDistributedBank:
    """Fixed-grid, no-persistent-front implementation of native mode 3.

    ``step`` produces a private candidate. A caller must reject on nonzero
    status and commit the seven histories only with its global transaction.
    """

    def __init__(self, area, owner, neighbour, offset, faces, signs,
                 xy, normal, length, bulk, bank_material_mask, nlayer,
                 *, transport_fraction, critical_slope, critical_undercut,
                 dry_tolerance, slope_method=LOCAL_FACE_SLOPE,
                 source_path=None, transfer_source=None):
        import cupy as cp

        topology, area = validate_geometry(area, owner, neighbour, offset, faces, signs)
        owner, neighbour, offset, faces, signs = topology
        nc, nf = len(area), len(owner)
        xy = np.asarray(xy, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        length = np.asarray(length, dtype=np.float64)
        bulk = np.asarray(bulk, dtype=np.float64)
        mask = np.asarray(bank_material_mask)
        if (xy.shape != (2, nc) or normal.shape != (2, nf) or
                length.shape != (nf,) or bulk.ndim != 1 or
                not 1 <= len(bulk) <= 32 or mask.shape != (nc,) or
                mask.dtype.kind not in 'biu' or not np.isin(mask, (0, 1)).all() or
                not np.isfinite(xy).all() or not np.isfinite(normal).all() or
                not np.isfinite(length).all() or np.any(length <= 0) or
                not np.isfinite(bulk).all() or np.any(bulk <= 0) or
                not np.isfinite([transport_fraction, critical_slope,
                                 critical_undercut, dry_tolerance]).all() or
                not 0 <= transport_fraction <= 1 or critical_slope <= 0 or
                critical_undercut <= 0 or dry_tolerance < 0 or
                slope_method not in (LOCAL_FACE_SLOPE, PUBLISHED_NET_GRADIENT) or
                not isinstance(nlayer, (int, np.integer)) or not 2 <= nlayer <= 33):
            raise ValueError('Invalid distributed-bank geometry or controls')
        if np.any(abs(np.sum(normal * normal, axis=0) - 1.) > 1.e-12):
            raise ValueError('Accepted-flux reconstruction requires unit normals')
        for f in range(nf):
            n = neighbour[f]
            if n >= 0 and np.linalg.norm(xy[:, owner[f]] - xy[:, n]) <= 0:
                raise ValueError('Coincident interior cell centres')
        self.cp = cp
        self.nc, self.nf, self.ng, self.nl = nc, nf, len(bulk), int(nlayer)
        self.alpha = np.float64(transport_fraction)
        self.slope = np.float64(critical_slope)
        self.slope_method = slope_method
        self.slope_mode = np.int32(slope_method == PUBLISHED_NET_GRADIENT)
        self.critical = np.float64(critical_undercut)
        self.dry = np.float64(dry_tolerance)
        self.area = cp.asarray(area)
        self.owner = cp.asarray(owner)
        self.neighbour = cp.asarray(neighbour)
        self.offset = cp.asarray(offset)
        self.faces = cp.asarray(faces)
        self.signs = cp.asarray(signs)
        self.xy = cp.asarray(np.ascontiguousarray(xy))
        self.normal = cp.asarray(np.ascontiguousarray(normal))
        self.length = cp.asarray(np.ascontiguousarray(length))
        self.bulk = cp.asarray(np.ascontiguousarray(bulk))
        self.mask = cp.asarray(np.ascontiguousarray(mask, dtype=np.int32))
        # Native face loops are ascending by face ID. Build the same donor
        # incidence order once; reconstruction uses the ordinary adjacency.
        donor_faces = [[] for _ in range(nc)]
        recipient_sources = [[] for _ in range(nc)]
        for f in range(nf):
            n = int(neighbour[f])
            if n < 0:
                continue
            o = int(owner[f])
            donor_faces[o].append(f)
            donor_faces[n].append(f)
            recipient_sources[o].append(n)
            recipient_sources[n].append(o)
        def csr(groups):
            groups = [sorted(set(group)) for group in groups]
            offsets = np.r_[0, np.cumsum([len(group) for group in groups])].astype(np.int32)
            values = np.asarray([v for group in groups for v in group], dtype=np.int32)
            return cp.asarray(offsets), cp.asarray(values)
        self.donor_offset, self.donor_faces = csr(donor_faces)
        self.recipient_offset, self.recipient_sources = csr(recipient_sources)
        self.transfer = ResidentSurfaceTransfer(area, owner, neighbour,
            offset, faces, signs, self.ng, source_path=transfer_source or HERE/'surface_transfer.cu')
        source = Path(source_path or HERE/'distributed_bank.cu').read_text()
        names = ('db_capture', 'db_reconstruct', 'db_propose', 'db_history',
                 'db_bed_after_transfer', 'db_plan_collapse', 'db_wave_mark',
                 'db_collapse_wave', 'db_deposit_wave', 'db_budget_partial',
                 'db_budget', 'db_validate')
        if self.slope_mode:
            names += ('db_net_slope',)
        module = cp.RawModule(code=source, options=OPTIONS, name_expressions=names)
        self.k = {name: module.get_function(name) for name in names}
        self.face_mass = cp.zeros((self.ng, nf), dtype=cp.float64)
        self.cell_bedload = cp.zeros((2, self.ng, nc), dtype=cp.float64)
        self.rank_deficient = cp.empty(nc, dtype=cp.int32)
        self.proposed = cp.zeros((self.ng, nf), dtype=cp.float64)
        self.eligible = cp.empty(nf, dtype=cp.int32)
        self.donor = cp.empty(nf, dtype=cp.int32)
        self.recipient = cp.empty(nf, dtype=cp.int32)
        self.local_slope = cp.empty(nf, dtype=cp.float64)
        self.net_slope = cp.empty(nc, dtype=cp.float64)
        self.tangential = cp.empty(nf, dtype=cp.float64)
        self.bed_after = cp.empty(nc, dtype=cp.float64)
        self.trial_layers = cp.empty((self.nl, self.ng, nc), dtype=cp.float64)
        self.trial_history = {key: cp.empty(nc, dtype=cp.float64) for key in FLOAT_HISTORY}
        self.trial_history['collapse_count'] = cp.empty(nc, dtype=cp.int32)
        self.removed = cp.empty((self.ng, nc), dtype=cp.float64)
        self.removed_volume = cp.empty(nc, dtype=cp.float64)
        self.recipient_gain = cp.empty((self.ng, nc), dtype=cp.float64)
        self.collapse_target = cp.empty(nc, dtype=cp.int32)
        self.collapse_candidate = cp.empty(nc, dtype=cp.int32)
        self.collapse_pending = cp.empty(nc, dtype=cp.int32)
        self.collapse_ready = cp.empty(nc, dtype=cp.int32)
        self.wave_counters = cp.empty(2, dtype=cp.int32)
        self.collapse_waves = 0
        self.collapse_budget = cp.empty((self.ng, 2), dtype=cp.float64)
        self.budget_chunks = (nc+255)//256
        self.partial_budget = cp.empty((self.ng,self.budget_chunks,2),dtype=cp.float64)
        self.capture_failure = cp.zeros((), dtype=cp.int32)
        self.status = {name: cp.empty(size, dtype=cp.int32) for name, size in
                       (('reconstruct', nc), ('propose', nf), ('history', nc),
                        ('net_slope', nc),
                        ('bed', nc), ('collapse', nc), ('conflicts', nc),
                        ('deposit', nc), ('budget', self.ng), ('validate', nc))}
        self.failure = cp.zeros((), dtype=cp.int32)
        self._tmp = cp.zeros((), dtype=cp.int32)

    def _collapse_ordered(self):
        """Run independent donors in parallel, preserving native ID order.

        A candidate recipient is fixed before the collapse loop, as in native.
        Each wave contains only donors with no earlier pending donor sharing a
        recipient or touching its donor cell. Deposits and bed updates finish
        before choosing the next wave. A scalar progress check fails closed.
        """
        nc, nf, ng, nl = self.nc, self.nf, self.ng, self.nl
        h = self.trial_history
        self.removed.fill(0.)
        self.removed_volume.fill(0.)
        self.recipient_gain.fill(0.)
        self.collapse_target.fill(-1)
        self.wave_counters.fill(0)
        for key in ('collapse', 'conflicts', 'deposit'):
            self.status[key].fill(0)
        self._launch('db_plan_collapse', nc, (np.int32(nc), self.length,
            self.donor_offset, self.donor_faces, self.eligible, self.donor,
            self.recipient, self.tangential, h['undercut'], self.critical,
            self.collapse_candidate, self.collapse_pending,
            self.wave_counters, self.status['conflicts']))
        # Most accepted steps have no stored wall trigger. A single scalar
        # progress read is cheaper than three empty wave kernels in that case.
        if int(self.wave_counters[0].item()) == 0:
            self.collapse_waves = 0
            return
        waves = 0
        while True:
            self.wave_counters[1:2].fill(0)
            self._launch('db_wave_mark', nc, (np.int32(nc),
                self.recipient_offset, self.recipient_sources,
                self.collapse_candidate, self.collapse_pending,
                self.collapse_ready, self.wave_counters,
                self.status['conflicts']))
            self._launch('db_collapse_wave', nc, (np.int32(nc), np.int32(nf),
                np.int32(ng), np.int32(nl), self.area, self.bulk,
                self.collapse_candidate, self.collapse_ready,
                self.collapse_pending, self.wave_counters, self.bed_after,
                self.trial_layers, self.critical, h['undercut'],
                h['reference_height'], h['toe_height'], h['reference_length'],
                h['collapse_cumulative'], h['collapse_count'], self.removed,
                self.removed_volume, self.collapse_target,
                self.status['collapse']))
            self._launch('db_deposit_wave', nc, (np.int32(nc), np.int32(nf),
                np.int32(ng), self.area, self.recipient_offset,
                self.recipient_sources, self.collapse_ready,
                self.collapse_target, self.removed, self.removed_volume,
                self.bed_after, self.trial_layers, self.recipient_gain,
                self.status['deposit']))
            remaining, ready = map(int, self.cp.asnumpy(self.wave_counters))
            waves += 1
            if remaining == 0:
                break
            if ready == 0 or remaining < 0 or waves >= nc:
                self.status['conflicts'].fill(1)
                break
        self.collapse_waves = waves

    def _array(self, value, shape, dtype):
        if (not isinstance(value, self.cp.ndarray) or value.shape != shape or
                value.dtype != dtype or not value.flags.c_contiguous):
            raise ValueError('Expected contiguous device array ' + str(shape))

    def _launch(self, name, count, args):
        if count:
            self.k[name](((count + 127) // 128,), (128,), args)

    def begin_trial(self):
        self.face_mass.fill(0.)
        self.capture_failure.fill(0)

    def capture(self, limited_flux, dt, stage):
        """Accumulate accepted Exner kg; no state/history is committed."""
        self._array(limited_flux, (self.ng, self.nf), self.cp.float64)
        if stage not in range(3) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Mode-3 face capture requires three SSPRK33 stages')
        count = self.ng * self.nf
        self._launch('db_capture', count, (limited_flux, self.length,
            np.int64(self.nf), np.int64(count), np.float64(dt),
            np.float64(SSP33_WEIGHTS[stage]), self.face_mass, self.capture_failure))
        return self.capture_failure

    def set_accepted_face_integral(self, integral):
        """Install one completed AMR register, already kg per physical face.

        Both traces use the owner-to-neighbour normal and must agree exactly
        for conservative Exner transport. No time, length or MorFac factor is
        applied here. The caller owns synchronization and the macro transaction.
        """
        cp = self.cp
        self._array(integral, (self.ng, 2, self.nf), cp.float64)
        if integral.device.id != self.face_mass.device.id:
            raise ValueError('Accepted bank register belongs to another CUDA device')
        lo, hi = integral.data.ptr, integral.data.ptr + integral.nbytes
        def arrays(value):
            if isinstance(value, cp.ndarray):
                yield value
            elif isinstance(value, dict):
                for item in value.values():
                    yield from arrays(item)
            elif isinstance(value, (list,tuple)):
                for item in value:
                    yield from arrays(item)
        for obj in (self, self.transfer):
            for value in vars(obj).values():
                for array in arrays(value):
                    begin = array.data.ptr
                    # All owned arrays are contiguous allocations or bounded
                    # contiguous views; include their full byte span.
                    end = begin + sum((size-1)*abs(stride) for size,stride in
                                      zip(array.shape,array.strides)) + array.dtype.itemsize
                    if array.size and lo < end and begin < hi:
                        raise ValueError('Accepted bank register overlaps private operator storage')
        if (not bool(cp.all(cp.isfinite(integral)).item()) or
                not bool(cp.all(integral[:,0,:] == integral[:,1,:]).item())):
            raise ValueError('Accepted Exner register must be finite with equal signed traces')
        cp.copyto(self.face_mass, integral[:,0,:])
        self.capture_failure.fill(0)

    def step(self, layers, depth, bed, accepted_history, dt):
        """Accepted-step mode-3 proposal on private, resident CUDA arrays."""
        cp = self.cp
        nc, nf, ng, nl = self.nc, self.nf, self.ng, self.nl
        self._array(layers, (nl, ng, nc), cp.float64)
        self._array(depth, (nc,), cp.float64)
        self._array(bed, (nc,), cp.float64)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive accepted duration required')
        if set(accepted_history) != set(self.trial_history):
            raise ValueError('Complete distributed-bank history required')
        for key, trial in self.trial_history.items():
            accepted = accepted_history[key]
            self._array(accepted, (nc,), trial.dtype)
            if accepted.data.ptr == trial.data.ptr:
                raise ValueError('Accepted history may not alias trial')
            cp.copyto(trial, accepted)
        cp.copyto(self.trial_layers, layers)
        self.failure.fill(0)
        dims = (np.int32(nc), np.int32(nf), np.int32(ng))
        self._launch('db_reconstruct', nc, (*dims, self.owner, self.neighbour,
            self.offset, self.faces, self.normal, self.length, self.face_mass,
            np.float64(dt), self.cell_bedload, self.rank_deficient,
            self.status['reconstruct']))
        if self.slope_mode:
            self._launch('db_net_slope', nc, (np.int32(nc), np.int32(nf),
                self.owner, self.neighbour, self.offset, self.faces, self.xy,
                bed, self.net_slope, self.status['net_slope']))
        else:
            self.net_slope.fill(0.)
            self.status['net_slope'].fill(0)
        self._launch('db_propose', nf, (*dims, self.owner, self.neighbour,
            self.xy, self.normal, self.length, bed, self.mask,
            self.cell_bedload, layers[0], self.net_slope, self.slope_mode,
            self.alpha, self.slope,
            np.float64(dt), self.proposed, self.eligible, self.donor,
            self.recipient, self.local_slope, self.tangential,
            self.status['propose']))
        transfer = self.transfer.step(layers[0], self.proposed)
        cp.copyto(self.trial_layers[0], transfer['next'])
        h = self.trial_history
        self._launch('db_history', nc, (*dims, self.area, self.length,
            self.bulk, self.donor_offset, self.donor_faces, self.eligible,
            self.donor, self.recipient, bed, depth, transfer['limited'],
            self.dry, h['undercut'], h['reference_height'], h['toe_height'],
            h['reference_length'], h['toe_cumulative'], self.status['history']))
        self._launch('db_bed_after_transfer', nc, (*dims, bed, self.bulk,
            layers[0], self.trial_layers[0], self.bed_after,
            self.status['bed']))
        self._collapse_ordered()
        self.k['db_budget_partial']((self.budget_chunks,ng),(256,),(
            np.int32(nc),np.int32(ng),np.int32(self.budget_chunks),self.removed,
            self.recipient_gain,self.partial_budget))
        self._launch('db_budget', ng, (np.int32(ng),
            np.int32(self.budget_chunks),self.partial_budget,
            self.collapse_budget,self.status['budget']))
        self._launch('db_validate', nc, (np.int32(nc), np.int32(ng),
            np.int32(nl), self.trial_layers, h['undercut'],
            h['reference_height'], h['toe_height'], h['reference_length'],
            h['toe_cumulative'], h['collapse_cumulative'],
            h['collapse_count'], self.status['validate']))
        cp.maximum(self.failure, self.capture_failure, out=self.failure)
        for item in (*self.status.values(), transfer['cell_status'], transfer['face_status']):
            cp.max(cp.where(item != 0, cp.int32(1), cp.int32(0)), out=self._tmp)
            cp.maximum(self.failure, self._tmp, out=self.failure)
        return dict(trial_layers=self.trial_layers,
                    trial_history=self.trial_history,
                    face_mass=self.face_mass,
                    cell_bedload=self.cell_bedload,
                    rank_deficient=self.rank_deficient,
                    proposed=self.proposed,
                    limited=transfer['limited'],
                    eligible=self.eligible,
                    donor=self.donor,
                    recipient=self.recipient,
                    local_slope=self.local_slope,
                    net_slope=self.net_slope,
                    tangential=self.tangential,
                    removed=self.removed,
                    recipient_gain=self.recipient_gain,
                    collapse_target=self.collapse_target,
                    collapse_budget=self.collapse_budget,
                    status=self.failure,
                    detail_status=self.status)
