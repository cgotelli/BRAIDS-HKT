"""Resident secondary-flow operations for explicit level-local AMR stepping.

This adapter reuses the existing secondary constitutive law, full-graph
curvature filter, reconstruction and signed FCT kernels. It changes only face
participation and which physical cell columns/source entries are published.
It does not select a backend, advance RK clocks, mutate accepted hI, or qualify
a complete recursive solver. The caller owns global rollback and acceptance.

Native ordering (amr_level_rk): after coarse ghosts and hydraulic preparation,
relax(dt/2); three SSPRK33 Euler legs; final preparation/friction; relax(dt/2).
Curvature is refreshed over the full graph at each half-source and retained
through all three Euler legs. Equilibrium mode uses raw stage curvature and
diagnostic hI for bedload rotation, not a transported conserved state.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ABI = '#define BRAIDS_AMR_LEVEL_SECONDARY_ABI 2'
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false')


def require(value, message):
    if not value:
        raise ValueError(message)


def schedules(data, levels):
    """CPU topology-only setup; no solution/state arrays are read on the host."""
    area, length = (np.asarray(data[k]) for k in ('area', 'length'))
    owner, other, levels = map(np.asarray, (data['owner'], data['neighbour'], levels))
    nc, nf = len(area), len(owner)
    require(nc > 0 and area.shape == (nc,) and length.shape == (nf,) and
            np.isfinite(area).all() and np.all(area > 0) and
            np.isfinite(length).all() and np.all(length >= 0), 'Invalid secondary level geometry')
    require(owner.shape == other.shape == (nf,) and owner.dtype.kind in 'iu' and
            other.dtype.kind in 'iu' and np.all((owner >= 0) & (owner < nc)) and
            np.all((other >= -1) & (other < nc)) and np.all(owner != other),
            'Invalid secondary level face endpoints')
    require(levels.shape == (nc,) and levels.dtype.kind in 'iu' and
            np.all((levels >= 0) & (levels <= 30)), 'Invalid secondary cell levels')
    internal = other >= 0
    require(np.all(abs(levels[owner[internal]].astype(np.int64)-
                       levels[other[internal]].astype(np.int64)) <= 1),
            'Secondary levels require balanced 2:1 neighbours')
    rows = []
    for level in range(int(levels.max())+1):
        cells = levels == level
        faces = cells[owner] | (internal & cells[np.maximum(other, 0)])
        rows.append(dict(cells=cells.astype(np.int32), faces=faces.astype(np.int32)))
    return rows


class ResidentLevelSecondary:
    """Wrap a qualified SecondaryTransport instance on the identical geometry.

    ``relax`` returns a private full hI array whose off-level columns are exact
    copies, plus a signed level-only source integral. ``advect`` returns private
    hI, signed all-face flux (zero on nonincident faces), and boundary ledger.
    Status is a device int32 scalar; zero alone admits a proposal. Retain every
    helper/operator array and ``operator.last_filter_receipt`` in outer rollback.
    """
    def __init__(self, operator, data, levels, *, source_path=None):
        self.operator = operator
        self.cp = cp = operator.cp
        self.model = int(operator.model)
        require(self.model in (0, 1, 2), 'Unsupported level secondary model')
        self.layout = schedules(data, levels)
        self.nc, self.nf, self.nb = operator.nc, operator.nf, operator.nb
        require(self.nc == len(data['area']) and self.nf == len(data['owner']) and self.nb >= 0,
                'Secondary operator geometry size differs')
        require(int(np.asarray(data['secondary_flow_model']).item()) == self.model,
                'Secondary operator model differs')
        self.source_path = Path(source_path or HERE/'amr_level_secondary.cu').resolve()
        code = self.source_path.read_bytes()
        require(sum(line.strip() == ABI for line in code.decode('utf-8').splitlines()) == 1,
                'Level secondary CUDA ABI differs')
        self.source_sha256 = hashlib.sha256(code).hexdigest()
        self.masks = [{k: cp.asarray(v) for k, v in row.items()} for row in self.layout]
        # Bound to the existing operator, never rebuild a differently configured
        # filter or source kernel behind the coupled solver's method identity.
        if self.model == 2:
            require(all(k in operator.inputs for k in ('area','owner','neighbour','offset','faces','signs',
                'length','bid','boundary_offset','boundary_faces')), 'Incomplete secondary operator topology')
            for key in ('area', 'length', 'owner', 'neighbour', 'offset', 'faces', 'signs', 'bid',
                        'boundary_offset', 'boundary_faces'):
                actual = operator.inputs[key]
                expected = np.asarray(data[key])
                require(actual.shape == expected.shape and
                        bool(cp.all(actual == cp.asarray(expected)).item()),
                        'Secondary operator topology differs: '+key)
        self.buffers = {
            'source_candidate': cp.zeros(self.nc), 'source_next': cp.zeros(self.nc),
            'source_work': cp.zeros(self.nc), 'source_integral': cp.zeros(()),
            'source_status': cp.zeros(self.nc, cp.int32),
            'validation_status': cp.zeros(self.nc, cp.int32),
            'next_validation_status': cp.zeros(self.nc, cp.int32),
            'reconstruction_cell_status': cp.zeros(self.nc, cp.int32),
            'reconstruction_face_status': cp.zeros(self.nf, cp.int32),
            'selected_active': cp.zeros(self.nc, cp.int32),
            'all_cells': cp.ones(self.nc, cp.int32),
            'water': cp.zeros(self.nf), 'neutral_next': cp.zeros(self.nc),
            'bound_next_q': cp.zeros((3,self.nc)), 'bound_depth_status': cp.zeros(self.nc,cp.int32),
            'neutral_flux': cp.zeros(self.nf), 'neutral_ledger': cp.zeros(self.nb),
            'status': cp.zeros((), cp.int32), 'temporary_status': cp.zeros((), cp.int32),
        }
        names = ('als_validate', 'als_active', 'als_mask_flux', 'als_mask_low', 'als_bound_depth', 'als_source_publish',
                 'als_scoped_status', 'als_total_status')
        self.module = cp.RawModule(code=code.decode('utf-8'), options=OPTIONS, name_expressions=names)
        self.kernels = {name: self.module.get_function(name) for name in names}
        self.filter_reserved = False

    def mutable_arrays(self):
        """All persistent mutable numeric storage, for caller rollback snapshots.

        Geometry arrays/masks are immutable. CUDA graph storage is allocated by
        reserve_filter before the first trial, and its handle thereafter stays
        valid; it contains no separate accepted physical state.
        """
        result = dict(self.buffers)
        result.update({'operator_'+k: v for k, v in self.operator.buffers.items()})
        filt = self.operator.filter
        if filt is not None:
            result.update({'filter_'+k: v for k, v in filt.buffers.items()})
            result.update({'filter_scalar_'+k: v for k, v in filt.scalar.items()})
            result.update({'filter_'+k: getattr(filt,k) for k in ('factor','status','done','count')})
            for i, row in enumerate(filt.levels):
                result.update({f'filter_level{i}_{k}': row[k]
                               for k in ('diagonal','b','x','work','res','w')})
        return result

    def reserve_filter(self, q, mass):
        """Construction-time reservation; never call inside an accepted trial.

        Initialize every mutable numerical array and capture the existing PCG
        iteration graph even when the cold curvature solve converges immediately.
        Input state is read-only; synthetic capture vectors do not enter physics.
        A second invocation is a no-op so it cannot erase a later trial's cache.
        """
        self._array(q, (3,self.nc)); self._array(mass,(self.nc,))
        self._inputs(q,mass)
        cp, op = self.cp, self.operator
        require(bool(cp.all(cp.isfinite(q)).item()) and bool(cp.all(q[0]>=0).item()) and
                bool(cp.all(cp.isfinite(mass)).item()) and
                (self.model==2 or not bool(cp.any(mass!=0).item())),
                'Invalid secondary reservation state')
        if self.filter_reserved:
            return
        arrays = self.mutable_arrays()
        def clear():
            for value in arrays.values():value.fill(0)
            self.buffers['all_cells'].fill(1)
            op.last_filter_receipt = None
        clear()
        try:
            if self.model==2:
                cp.copyto(self.buffers['source_candidate'],mass)
                op.relax(q,self.buffers['source_candidate'],0.)
                filt=op.filter
                if filt.graph is None:
                    # The regular cold solve may need zero/one iteration. Warm
                    # and capture the same iteration using done=1 and zero
                    # residual, while the finite factorization is still valid.
                    with filt.graph_stream,cp.cuda.using_allocator(filt.graph_pool.malloc):
                        for k in ('p','r','rhs','ap','product'):filt.buffers[k].fill(0)
                        for k in filt.scalar:filt.scalar[k].fill(1)
                        filt.done.fill(1);filt.status.fill(0)
                        filt.graph_iteration()
                        filt.graph_stream.synchronize()
                        try:
                            filt.graph_stream.begin_capture()
                            filt.graph_iteration()
                            filt.graph=filt.graph_stream.end_capture()
                        except Exception:
                            if filt.graph_stream.is_capturing():filt.graph_stream.end_capture()
                            raise
                        filt.graph.upload(stream=filt.graph_stream)
                    filt.graph_stream.synchronize()
                require(filt.graph is not None,'Secondary curvature graph reservation failed')
            self.filter_reserved=True
        finally:
            # No warm-up source/filter result is an accepted initial condition.
            clear()

    def _launch(self, name, size, args):
        if size:
            self.kernels[name](((size+127)//128,), (128,), args)

    def _array(self, value, shape, *, integer=False):
        require(isinstance(value, self.cp.ndarray) and value.shape == shape and
                value.dtype == (self.cp.int32 if integer else self.cp.float64) and
                value.flags.c_contiguous, 'Expected contiguous resident secondary array '+str(shape))

    @staticmethod
    def _overlaps(a, b):
        return a.nbytes and b.nbytes and a.data.ptr < b.data.ptr+b.nbytes and b.data.ptr < a.data.ptr+a.nbytes

    def _inputs(self, *arrays):
        private_arrays = self.mutable_arrays()
        for value in arrays:
            require(not any(self._overlaps(value, private) for private in private_arrays.values()),
                    'Secondary input aliases private/operator/filter workspace')
        for i, value in enumerate(arrays):
            require(not any(self._overlaps(value, other) for other in arrays[i+1:]),
                    'Secondary physical input arrays overlap')

    def _level(self, level):
        require(type(level) is int and 0 <= level < len(self.masks) and
                np.any(self.layout[level]['cells']), 'Secondary operation requires a populated integer level')
        return self.masks[level]

    def _dt(self, dt, *, positive):
        value = np.asarray(dt)
        require(value.shape == () and value.dtype.kind in 'fiu' and np.isfinite(value).item() and
                (float(value) > 0 if positive else float(value) >= 0), 'Invalid secondary timestep')
        return np.float64(value)

    def _merge(self, value):
        b = self.buffers
        self.cp.max(value, out=b['temporary_status'])
        self.cp.maximum(b['status'], b['temporary_status'], out=b['status'])

    def _validate(self, q, mass, out):
        self._launch('als_validate', self.nc, (np.int32(self.nc), np.int32(self.model), q, mass, out))
        self._merge(out)

    def cleanup_dry(self, q, mass):
        """Private accepted-boundary dry hI cleanup and its signed source.

        This explicitly extends native ordering: native's dry branch runs in
        each level half-source, before later reflux can deplete a recipient.
        Use its same ``h <= dry`` rule here without another curvature solve or
        wet relaxation. The caller must add ``source_integral`` once (no dt or
        RK weight) before acceptance, and include this workspace in rollback.
        Input arrays are read-only; a nonzero device status rejects the proposal.
        """
        self._array(q, (3,self.nc)); self._array(mass, (self.nc,))
        self._inputs(q, mass)
        cp, b, op = self.cp, self.buffers, self.operator
        device = int(b['source_next'].device.id)
        require(int(q.device.id)==int(mass.device.id)==device==int(cp.cuda.runtime.getDevice()),
                'Secondary cleanup inputs must share the active operator device')
        b['status'].fill(0)
        self._validate(q, mass, b['validation_status'])
        cp.copyto(b['source_next'], mass)
        if self.model == 2:
            cp.copyto(b['source_next'], 0., where=q[0]<=op.dry)
            cp.subtract(b['source_next'], mass, out=b['source_work'])
            cp.multiply(b['source_work'], op.inputs['area'], out=b['source_work'])
            b['source_status'][...] = (~cp.isfinite(b['source_next']) |
                                       ~cp.isfinite(b['source_work'])).astype(cp.int32)
            self._merge(b['source_status'])
            cp.sum(b['source_work'], out=b['source_integral'])
            self._launch('als_total_status', 1, (b['source_integral'], b['status']))
        else:
            b['source_work'].fill(0); b['source_status'].fill(0); b['source_integral'].fill(0)
        return dict(next=b['source_next'], source_work=b['source_work'],
                    source_integral=b['source_integral'], status=b['status'],
                    cell_status=b['source_status'])

    def relax(self, q, mass, dt_source, level):
        """Full-graph curvature solve; publish only this level's half-source.

        Caller supplies the half-step duration explicitly. The wrapped source
        works on private copied mass, so failed source/filter work cannot alter
        the input. No off-level nonlinear source result enters the ledger.
        """
        self._array(q, (3, self.nc)); self._array(mass, (self.nc,))
        self._inputs(q, mass)
        mask = self._level(level); dt = self._dt(dt_source, positive=False)
        cp, b, op = self.cp, self.buffers, self.operator
        b['status'].fill(0)
        self._validate(q, mass, b['validation_status'])
        cp.copyto(b['source_candidate'], mass)
        result = op.relax(q, b['source_candidate'], dt)
        if self.model == 2:
            self._launch('als_source_publish', self.nc, (np.int32(self.nc), mask['cells'], mass,
                b['source_candidate'], op.buffers['source_work'], op.buffers['source_status'],
                b['source_next'], b['source_work'], b['source_status']))
            self._merge(b['source_status'])
            cp.sum(b['source_work'], out=b['source_integral'])
            self._launch('als_total_status', 1, (b['source_integral'], b['status']))
        else:
            cp.copyto(b['source_next'], mass)
            b['source_work'].fill(0); b['source_status'].fill(0); b['source_integral'].fill(0)
        return dict(next=b['source_next'], source_work=b['source_work'], source_integral=b['source_integral'],
            status=b['status'], cell_status=b['source_status'], curvature=result['curvature'],
            filter_receipt=result['filter_receipt'])

    def reconstruct(self, q, bed, mass, level, *, hydraulic_buffers=None):
        """Reuse unchanged stage reconstruction/diagnostic equilibrium physics."""
        self._array(q, (3, self.nc)); self._array(bed, (self.nc,)); self._array(mass, (self.nc,))
        self._inputs(q, bed, mass)
        mask = self._level(level); b, op = self.buffers, self.operator
        b['status'].fill(0)
        self._validate(q, mass, b['validation_status'])
        result = op.reconstruct(q, bed, mass, hydraulic_buffers=hydraulic_buffers)
        if self.model == 2:
            self._merge(op.buffers['primitive_status'])
            self._launch('als_scoped_status', self.nf, (np.int32(self.nf), mask['faces'],
                op.buffers['reconstruction_status'], b['reconstruction_face_status']))
            self._merge(b['reconstruction_face_status'])
        else:
            self._merge(result['status'])
        return dict(result, status=b['status'])

    def advect(self, q, next_q, mass, water_flux, dt, level, *, hydraulic_buffers=None):
        """Signed hI FCT on accepted water, including coarse ghost donor bounds.

        All-cell bounds include ghost donors. Their private bound depth follows
        the same incident accepted water flux as low-order hI; using stale ghost
        q depth spuriously rejects valid constant-intensity transport. Selected
        cells use actual next_q depth. Physical update still uses next_q, never
        this private limiter volume. The shared returned flux must be refluxed.
        """
        self._array(q, (3, self.nc)); self._array(next_q, (3, self.nc)); self._array(mass, (self.nc,))
        if isinstance(water_flux, self.cp.ndarray) and water_flux.shape == (3, self.nf):
            water_flux = water_flux[0]
        self._array(water_flux, (self.nf,))
        self._inputs(q, next_q, mass, water_flux)
        mask = self._level(level); step = self._dt(dt, positive=True)
        cp, b, op = self.cp, self.buffers, self.operator
        b['status'].fill(0)
        self._validate(q, mass, b['validation_status'])
        self._validate(next_q, mass, b['next_validation_status'])
        if self.model != 2:
            cp.copyto(b['neutral_next'], mass)
            return dict(next=b['neutral_next'], flux=b['neutral_flux'], ledger=b['neutral_ledger'], status=b['status'])
        hydro = op._hydro(hydraulic_buffers)
        self._array(hydro['active'], (self.nc,), integer=True)
        a, s = op.inputs, op.buffers
        nc, nf = np.int32(self.nc), np.int32(self.nf)
        self._launch('als_mask_flux', self.nf, (nf, mask['faces'], water_flux, b['water']))
        self._launch('als_active', self.nc, (nc, mask['cells'], hydro['active'], b['selected_active']))
        op.launch('low_flux', self.nf, (nc, nf, q, s['intensity'], s['face_intensity'], s['curvature'],
            a['owner'], a['neighbour'], b['water'], op.dry, s['low_flux'], s['anti_flux'], s['boundary_intensity'], s['flux_status']))
        # Native AMR does not evaluate excluded faces. In particular a stale
        # excluded trace must not leak 0*NaN into a coarse ghost's FCT bound.
        self._launch('als_mask_low', self.nf, (nf, mask['faces'], s['low_flux'], s['anti_flux'],
            s['boundary_intensity'], s['flux_status']))
        cp.copyto(b['bound_next_q'],next_q)
        self._launch('als_bound_depth',self.nc,(nc,mask['cells'],q,next_q,a['area'],a['length'],
            a['offset'],a['faces'],a['signs'],b['water'],step,b['bound_next_q'],b['bound_depth_status']))
        self._merge(b['bound_depth_status'])
        op.launch('fct_bounds', self.nc, (nc, nf, s['intensity'], mass, b['bound_next_q'], a['area'], a['length'],
            a['owner'], a['neighbour'], a['bid'], a['offset'], a['faces'], a['signs'], b['all_cells'],
            b['water'], s['low_flux'], s['anti_flux'], s['boundary_intensity'], step,
            s['lowstate'], s['plus'], s['minus'], s['fct_status']))
        op.launch('fct_faces', self.nf, (nf, a['owner'], a['neighbour'], s['low_flux'],
            s['plus'], s['minus'], s['anti_flux'], s['flux']))
        op.launch('fct_update', self.nc, (nc, next_q, mass, s['lowstate'], a['area'], a['length'],
            a['offset'], a['faces'], a['signs'], b['selected_active'], s['anti_flux'], step, s['next'], s['fct_status']))
        if self.nb:
            op.kernels['ledger']((self.nb,), (128,), (np.int32(self.nb), a['boundary_offset'],
                a['boundary_faces'], a['length'], s['flux'], step, s['ledger']))
        self._merge(s['flux_status']); self._merge(s['fct_status'])
        return dict(next=s['next'], flux=s['flux'], ledger=s['ledger'], status=b['status'],
                    cell_status=s['fct_status'], face_status=s['flux_status'],bound_depth=b['bound_next_q'][0])
