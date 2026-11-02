"""Resident, conservative per-grain surface transfer for GPU bank/repose work.

The sign convention and donor limiter follow BRAIDS's
``limit_surface_transfer``/``apply_limited_surface_transfer`` in
``solver/cut_cells/cutcell_solver.f90``. This is a reusable I4 component, not
yet a coupled bank or avalanche model. All arrays remain on the GPU during
``step``; the input inventory and proposed transfers are never mutated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')


def validate_geometry(area, owner, neighbour, offset, faces, signs):
    """Validate zero-based finite-volume adjacency once before device setup."""
    area = np.asarray(area, dtype=np.float64)
    owner = np.asarray(owner)
    neighbour = np.asarray(neighbour)
    offset = np.asarray(offset)
    faces = np.asarray(faces)
    signs = np.asarray(signs)
    if area.ndim != 1 or len(area) < 1 or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Positive finite cell areas required')
    nc = len(area)
    if owner.ndim != 1 or len(owner) < 1 or neighbour.shape != owner.shape:
        raise ValueError('Invalid face owner/neighbour arrays')
    nf = len(owner)
    for name, values in (('owner', owner), ('neighbour', neighbour), ('offset', offset),
                         ('faces', faces), ('signs', signs)):
        if values.dtype.kind not in 'iu' or np.any(values > np.iinfo(np.int32).max):
            raise ValueError('Integer32 topology required: ' + name)
    if np.any(owner < 0) or np.any(owner >= nc) or np.any(neighbour < -1) or \
            np.any(neighbour >= nc) or np.any(owner == neighbour):
        raise ValueError('Invalid zero-based face ownership')
    if offset.shape != (nc + 1,) or offset[0] != 0 or offset[-1] != len(faces) or \
            np.any(np.diff(offset) < 0) or faces.shape != signs.shape or \
            np.any(faces < 0) or np.any(faces >= nf) or not np.isin(signs, (-1, 1)).all():
        raise ValueError('Invalid cell-face adjacency')
    count = np.zeros(nf, dtype=np.int32)
    for c in range(nc):
        for k in range(int(offset[c]), int(offset[c + 1])):
            f = int(faces[k]); sign = int(signs[k])
            if (sign == 1 and owner[f] != c) or (sign == -1 and neighbour[f] != c):
                raise ValueError('Face sign/owner adjacency mismatch')
            count[f] += 1
    expected = np.where(neighbour < 0, 1, 2)
    if not np.array_equal(count, expected):
        raise ValueError('Missing or repeated face incidence')
    return tuple(np.ascontiguousarray(v, dtype=np.int32)
                 for v in (owner, neighbour, offset, faces, signs)), \
        np.ascontiguousarray(area)


class ResidentSurfaceTransfer:
    """Limit proposed total face mass, then gather into active mass per area.

    ``active`` has shape ``(grain, cell)`` in kg/m2. ``proposed`` has shape
    ``(grain, face)`` in kg for the current accepted candidate. Positive mass
    goes owner to neighbour. Nonzero exterior-face proposals fail closed.
    Output buffers are reused by the next call; copy only at a caller-owned
    transaction boundary. A nonzero device status requires trial rejection.
    """

    def __init__(self, area, owner, neighbour, offset, faces, signs, nfrac,
                 *, source_path=None):
        import cupy as cp

        topology, area = validate_geometry(area, owner, neighbour, offset, faces, signs)
        if not isinstance(nfrac, (int, np.integer)) or not 1 <= nfrac <= 32:
            raise ValueError('Expected 1-32 sediment classes')
        self.cp = cp
        self.nc = len(area); self.nf = len(topology[0]); self.ng = int(nfrac)
        self.area = cp.asarray(area)
        self.owner, self.neighbour, self.offset, self.faces, self.signs = \
            (cp.asarray(v) for v in topology)
        source = Path(source_path or HERE / 'surface_transfer.cu').read_text()
        names = ('surface_donor_scales', 'surface_limit_faces', 'surface_gather')
        module = cp.RawModule(code=source, options=OPTIONS, name_expressions=names)
        self.k = {name: module.get_function(name) for name in names}
        self.scale = cp.empty((self.ng, self.nc), cp.float64)
        self.limited = cp.empty((self.ng, self.nf), cp.float64)
        self.next = cp.empty((self.ng, self.nc), cp.float64)
        self.cell_status = cp.empty((self.ng, self.nc), cp.int32)
        self.face_status = cp.empty((self.ng, self.nf), cp.int32)
        self._resident_identity = {
            name: (getattr(self, name).data.ptr, getattr(self, name).shape,
                   getattr(self, name).dtype)
            for name in ('area', 'owner', 'neighbour', 'offset', 'faces', 'signs',
                         'scale', 'limited', 'next', 'cell_status', 'face_status')}

    def _check_array(self, value, shape):
        if not isinstance(value, self.cp.ndarray) or value.shape != shape or \
                value.dtype != self.cp.float64 or not value.flags.c_contiguous:
            raise ValueError('Expected contiguous FP64 device array ' + str(shape))

    def step(self, active, proposed):
        self._check_array(active, (self.ng, self.nc))
        self._check_array(proposed, (self.ng, self.nf))
        for name, identity in self._resident_identity.items():
            value = getattr(self, name)
            if (value.data.ptr, value.shape, value.dtype) != identity:
                raise ValueError('Resident surface-transfer storage changed: ' + name)
        dims = (np.int32(self.nc), np.int32(self.nf), np.int32(self.ng))
        def launch(name, size, arguments):
            self.k[name](((size + 127) // 128,), (128,), arguments)
        launch('surface_donor_scales', self.nc * self.ng,
               (*dims, active, proposed, self.area, self.offset, self.faces,
                self.signs, self.scale, self.cell_status))
        launch('surface_limit_faces', self.nf * self.ng,
               (*dims, proposed, self.owner, self.neighbour, self.scale,
                self.limited, self.face_status))
        launch('surface_gather', self.nc * self.ng,
               (*dims, active, self.area, self.offset, self.faces, self.signs,
                self.limited, self.cell_status, self.next))
        return dict(next=self.next, limited=self.limited, scale=self.scale,
                    cell_status=self.cell_status, face_status=self.face_status)
