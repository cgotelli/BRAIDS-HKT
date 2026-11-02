"""Isolated GPU-resident four-child AMR restriction/prolongation transaction.

This ports the explicitly bounded transfer dependencies from native
``amr_sync_state`` and ``amr_prolong_node``, including nonzero classwise
suspended and optional signed secondary inventory. It is NOT a GPU AMR solver:
selection, face rebuilding, level stepping, suspended exchange and bank
fronts remain unsupported and must fail closed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OPTIONS = ('--std=c++11', '--fmad=false', '--prec-div=true',
           '--prec-sqrt=true', '--ftz=false')


def host_geometry(areas, parent_areas, rock, positions, bulk, *, dry):
    weights = np.asarray(areas, dtype=np.float64)
    pa = np.asarray(parent_areas, dtype=np.float64)
    bedrock = np.asarray(rock, dtype=np.float64)
    offsets = np.asarray(positions, dtype=np.float64)
    density = np.asarray(bulk, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 4 or len(weights) < 1:
        raise ValueError('Only nonempty 2:1 groups with exactly four children are supported')
    count = len(weights)
    if pa.shape != (count,) or bedrock.shape != (count, 4) or offsets.shape != (count, 4, 2):
        raise ValueError('AMR parent/child geometry shape differs')
    if density.ndim != 1 or not 1 <= len(density) <= 32:
        raise ValueError('One to 32 explicit grain bulk densities required')
    if (not all(np.isfinite(array).all() for array in (weights, pa, bedrock, offsets, density))
            or not np.isfinite(dry) or dry <= 0 or
            np.any(weights <= 0) or np.any(pa <= 0) or np.any(density <= 0)):
        raise ValueError('Nonfinite/nonpositive AMR geometry, bulk or dry cutoff')
    if not np.allclose(weights.sum(axis=1), pa, atol=1e-13, rtol=1e-12):
        raise ValueError('Four child areas must partition the parent')
    centroid = (weights[:, :, None] * offsets).sum(axis=1) / pa[:, None]
    if not np.allclose(centroid, 0., atol=1e-12, rtol=0):
        raise ValueError('Child centroids must preserve the parent centroid')
    return tuple(np.ascontiguousarray(array) for array in (weights, pa, bedrock, offsets, density))


class ResidentAMRTransfer:
    """The accepted children stay untouched until a successful explicit commit.

    Array layouts: accepted/trial ``(parent, child, state)``; parent state
    ``(parent, state)``; gradient ``(parent, primitive, xy)``. State rows are
    native [h, hu, hv, suspended per grain, active/substrate per grain,
    optional signed secondary inventory].
    The caller supplies the native primitive gradient, cached child states,
    immutable child rock and the parent state after coarse evolution. This
    component never approximates them from a smaller field list.
    """

    def __init__(self, cached_children, areas, parent_areas, rock, positions,
                 bulk, *, dry, prognostic_secondary=False, source_path=None):
        self.areas_host, self.parent_areas_host, rock_host, pos_host, bulk_host = host_geometry(
            areas, parent_areas, rock, positions, bulk, dry=dry)
        self.parents = len(self.areas_host)
        self.grains = len(bulk_host)
        initial = np.asarray(cached_children)
        if (initial.ndim != 3 or initial.shape[:2] != (self.parents, 4) or
                initial.dtype != np.float64 or not initial.flags.c_contiguous):
            raise ValueError('Contiguous FP64 four-child native state required')
        self.states = initial.shape[2]
        self.prognostic_secondary = bool(prognostic_secondary)
        self.physical_states = self.states - int(self.prognostic_secondary)
        if ((self.physical_states - 3) % self.grains or
                self.physical_states < 3 + 2*self.grains):
            raise ValueError('Native layered state layout required')
        self.layers = (self.physical_states - 3)//self.grains - 1
        if not 1 <= self.layers <= 33:
            raise ValueError('One active layer with zero to 32 substrate layers required')
        self.primitives = 4 + self.grains + int(self.prognostic_secondary)
        if (not np.isfinite(initial).all() or np.any(initial[:, :, 0] < 0) or
                np.any(initial[:, :, 3:self.physical_states] < 0)):
            raise ValueError('Finite nonnegative child inventory required')
        dry_child = initial[:, :, 0] <= float(dry)
        if (np.any(dry_child & (initial[:, :, 1:3] != 0).any(axis=2)) or
                np.any(dry_child & (initial[:, :, 3:3+self.grains] != 0).any(axis=2)) or
                (self.prognostic_secondary and
                 np.any(dry_child & (initial[:, :, self.physical_states] != 0)))):
            raise ValueError('Dry children cannot carry momentum, suspended or secondary mass')
        import cupy as cp
        self.cp = cp
        self.weight = cp.asarray(self.areas_host)
        self.parent_area = cp.asarray(self.parent_areas_host)
        self.rock = cp.asarray(rock_host)
        self.position = cp.asarray(pos_host)
        self.bulk = cp.asarray(bulk_host)
        self.dry = np.float64(dry)
        self.accepted = cp.asarray(initial)
        self.trial = cp.empty_like(self.accepted)
        self.restricted = cp.empty((self.parents, self.states), dtype=cp.float64)
        self.status = cp.empty(self.parents, dtype=cp.int32)
        self.pending = False
        self.source_path = Path(source_path or HERE/'amr_transfer.cu')
        module = cp.RawModule(code=self.source_path.read_text(encoding='utf-8'),
            options=OPTIONS,
            name_expressions=('amr_restrict_native_rows', 'amr_prolong_native_dry_bed',
                              'amr_prolong_signed_secondary'))
        self.restrict_kernel = module.get_function('amr_restrict_native_rows')
        self.prolong_kernel = module.get_function('amr_prolong_native_dry_bed')
        self.secondary_kernel = module.get_function('amr_prolong_signed_secondary')

    def _launch(self, kernel, arguments):
        kernel(((self.parents + 127)//128,), (128,), arguments)

    def restrict(self):
        """Area-restrict every conserved native state row on the GPU."""
        self._launch(self.restrict_kernel, (self.accepted, self.weight,
            np.int32(self.parents), np.int32(self.states), self.restricted, self.status))
        if bool(self.cp.any(self.status != 0)):
            raise ValueError('Invalid resident AMR restriction status')
        return self.restricted

    def propose(self, evolved_parent, gradient):
        """Compute a new child state without mutating accepted state."""
        cp = self.cp
        if self.pending:
            raise ValueError('Resolve the previous AMR transaction first')
        if (not isinstance(evolved_parent, cp.ndarray) or
                evolved_parent.shape != (self.parents, self.states) or
                evolved_parent.dtype != cp.float64 or not evolved_parent.flags.c_contiguous):
            raise ValueError('Resident contiguous FP64 parent state required')
        if (not isinstance(gradient, cp.ndarray) or
                gradient.shape != (self.parents, self.primitives, 2) or
                gradient.dtype != cp.float64 or not gradient.flags.c_contiguous):
            raise ValueError('Resident native primitive gradient required')
        cp.copyto(self.trial, self.accepted)
        self._launch(self.prolong_kernel, (evolved_parent, self.accepted,
            self.weight, self.parent_area, self.rock, self.position, gradient,
            self.bulk, np.int32(self.parents), np.int32(self.grains),
            np.int32(self.layers), np.int32(self.states), np.int32(self.primitives),
            np.int32(self.prognostic_secondary), self.dry, self.trial,
            self.status))
        if self.prognostic_secondary:
            self._launch(self.secondary_kernel, (evolved_parent, self.weight,
                self.position, gradient, np.int32(self.parents),
                np.int32(self.states), np.int32(self.primitives),
                np.int32(self.grains), self.dry, self.trial, self.status))
        self.pending = True
        return self.trial, self.status

    def commit(self):
        if not self.pending:
            raise ValueError('No pending AMR transfer')
        if bool(self.cp.any(self.status != 0)):
            raise ValueError('Reject invalid AMR transfer; accepted state is intact')
        self.accepted, self.trial = self.trial, self.accepted
        self.pending = False
        return self.accepted

    def rollback(self):
        """Discard a rejected proposal while keeping all accepted device bytes."""
        if not self.pending:
            raise ValueError('No pending AMR transfer')
        self.pending = False
        return self.accepted
