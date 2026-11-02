"""Geometric AMR face correspondence and deterministic resident ledger transfer.

Faces match only when both global-node endpoint lineages agree and their
straight physical segments overlap. Local face numbering and current bank
roles are not identities. Signed values use owner-to-neighbour orientation;
gross values never acquire a sign. Disappearing portions remain explicit
archived diagnostics, not a zeroed or redistributed transport history.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


_EPS = 512. * np.finfo(np.float64).eps


@dataclass(frozen=True)
class FaceCorrespondence:
    old_face: np.ndarray
    new_face: np.ndarray
    old_fraction: np.ndarray
    orientation: np.ndarray
    unmapped_old: np.ndarray
    unmapped_fraction: np.ndarray
    new_offsets: np.ndarray
    old_nf: int
    new_nf: int


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _frozen(values, dtype):
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _tree_intervals(prepared):
    parent = np.asarray(prepared.tree['parent'])
    _require(parent.ndim == 1 and parent.dtype.kind in 'iu' and len(parent) > 0,
             'AMR face correspondence needs an integer parent tree')
    nn = len(parent)
    _require(np.all((parent >= -1) & (parent < nn)) and
             not np.any(parent == np.arange(nn)), 'Invalid AMR parent IDs')
    children = [[] for _ in parent]
    roots = []
    for node, ancestor in enumerate(parent):
        (roots if ancestor < 0 else children[int(ancestor)]).append(node)
    enter = np.full(nn, -1, np.int64)
    leave = np.full(nn, -1, np.int64)
    clock = 0
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, returning = stack.pop()
            if returning:
                leave[node] = clock
                continue
            _require(enter[node] == -1, 'Cyclic or repeated AMR parent node')
            enter[node] = clock
            clock += 1
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(children[node]))
    _require(np.all(enter >= 0), 'AMR parent tree has an unreachable cycle')
    return parent.astype(np.int64, copy=False), enter, leave


def _geometry(topology, enter, leave):
    ids = np.asarray(topology.node_ids)
    data = topology.data
    _require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(ids) > 0 and
             np.all((ids >= 0) & (ids < len(enter))) and len(np.unique(ids)) == len(ids),
             'Invalid selected global AMR node IDs')
    ids = ids.astype(np.int64, copy=False)
    sequence = ids[np.argsort(enter[ids], kind='stable')]
    _require(len(ids) == 1 or np.all(leave[sequence[:-1]] <= enter[sequence[1:]]),
             'Selected AMR nodes overlap an ancestor/descendant')
    owner, neighbour = (np.asarray(data[key]) for key in ('owner', 'neighbour'))
    _require(owner.ndim == 1 and owner.dtype.kind in 'iu' and
             neighbour.shape == owner.shape and neighbour.dtype.kind in 'iu',
             'Invalid AMR face endpoint arrays')
    nf, nc = len(owner), len(ids)
    _require(np.all((owner >= 0) & (owner < nc)) and
             np.all((neighbour >= -1) & (neighbour < nc)) and
             not np.any(owner == neighbour), 'Invalid AMR local face endpoints')
    result = dict(ids=ids, owner=owner.astype(np.int64, copy=False),
                  neighbour=neighbour.astype(np.int64, copy=False), nf=nf)
    for name, shape in (('xy', (2, nc)), ('mid', (2, nf)),
                         ('normal', (2, nf)), ('length', (nf,))):
        value = np.asarray(data[name])
        _require(value.shape == shape and value.dtype == np.float64 and
                 np.isfinite(value).all(), 'Invalid AMR face geometry '+name)
        result[name] = value
    _require(np.all(result['length'] > 0), 'Nonpositive AMR face length')
    norms = np.linalg.norm(result['normal'], axis=0)
    _require(np.all(np.abs(norms - 1.) <= _EPS), 'AMR face normals must be unit vectors')
    # Normalize only floating-point unit-length roundoff for projections.
    result['normal'] = result['normal'] / norms
    bid = np.asarray(data.get('bid', np.full(nf, -1, np.int64)))
    _require(bid.shape == (nf,) and bid.dtype.kind in 'iu', 'Invalid AMR face boundary IDs')
    result['bid'] = bid
    return result


def plan_face_correspondence(prepared, old, new):
    """Plan ancestry-compatible collinear overlap on authenticated topologies.

    Edges are sorted by ``(new_face, old_face)`` and ``new_offsets`` is their
    destination CSR. ``unmapped_old`` includes partially disappearing faces;
    the archived share is ``unmapped_fraction``. Coordinate comparisons use
    floating-point geometry roundoff, not a physical erosion/depth threshold.
    """
    _require(any(old is top for top in prepared.topologies) and
             any(new is top for top in prepared.topologies),
             'AMR correspondence topology is outside the authenticated family')
    parent, enter, leave = _tree_intervals(prepared)
    a, b = _geometry(old, enter, leave), _geometry(new, enter, leave)

    def related(first, second):
        return (enter[first] <= enter[second] < leave[first] or
                enter[second] <= enter[first] < leave[second])

    incident = [[] for _ in b['ids']]
    for face, (owner, neighbour) in enumerate(zip(b['owner'], b['neighbour'])):
        incident[int(owner)].append(face)
        if neighbour >= 0:
            incident[int(neighbour)].append(face)
    selected = {int(node): local for local, node in enumerate(b['ids'])}
    ordered = np.argsort(enter[b['ids']], kind='stable')
    starts = enter[b['ids'][ordered]]

    def compatible_cells(node):
        lo = np.searchsorted(starts, enter[node], side='left')
        hi = np.searchsorted(starts, leave[node], side='left')
        if hi > lo:
            return ordered[lo:hi]
        ancestor = int(parent[node])
        while ancestor >= 0:
            if ancestor in selected:
                return (selected[ancestor],)
            ancestor = int(parent[ancestor])
        return ()

    edges = []
    old_intervals = [[] for _ in range(a['nf'])]
    new_intervals = [[] for _ in range(b['nf'])]
    for old_face in range(a['nf']):
        old_owner = int(a['ids'][a['owner'][old_face]])
        old_neighbour = int(a['neighbour'][old_face])
        if old_neighbour >= 0:
            old_neighbour = int(a['ids'][old_neighbour])
        candidates = sorted({face for cell in compatible_cells(old_owner)
                             for face in incident[int(cell)]})
        normal = a['normal'][:, old_face]
        tangent = np.array([-normal[1], normal[0]])
        midpoint = a['mid'][:, old_face]
        half = .5 * float(a['length'][old_face])
        for new_face in candidates:
            new_owner = int(b['ids'][b['owner'][new_face]])
            new_neighbour = int(b['neighbour'][new_face])
            if new_neighbour >= 0:
                new_neighbour = int(b['ids'][new_neighbour])
            if (old_neighbour < 0) != (new_neighbour < 0):
                continue
            if old_neighbour < 0:
                if (not related(old_owner, new_owner) or
                        a['bid'][old_face] != b['bid'][new_face]):
                    continue
                orientation = 1
            elif related(old_owner, new_owner) and related(old_neighbour, new_neighbour):
                orientation = 1
            elif related(old_owner, new_neighbour) and related(old_neighbour, new_owner):
                orientation = -1
            else:
                continue
            other_normal = b['normal'][:, new_face]
            cross = normal[0]*other_normal[1] - normal[1]*other_normal[0]
            if abs(cross) > _EPS:
                continue
            other_mid = b['mid'][:, new_face]
            tolerance = _EPS * max(1., float(np.max(np.abs(midpoint))),
                                   float(np.max(np.abs(other_mid))), 2.*half,
                                   float(b['length'][new_face]))
            delta = other_mid - midpoint
            if abs(float(np.dot(normal, delta))) > tolerance:
                continue
            center = float(np.dot(tangent, delta))
            other_half = .5 * float(b['length'][new_face])
            lower, upper = max(-half, center-other_half), min(half, center+other_half)
            overlap = upper-lower
            if overlap <= tolerance:
                continue
            _require(orientation * float(np.dot(normal, other_normal)) > 0.,
                     'Overlapping face normal contradicts global endpoint orientation')
            index = len(edges)
            edges.append([old_face, new_face, overlap/(2.*half), orientation])
            old_intervals[old_face].append((lower, upper, tolerance, index))
            # Canonical old-tangent projection suffices for an individual pair;
            # use the new tangent for overlap checks among multiple old faces.
            other_tangent = np.array([-other_normal[1], other_normal[0]])
            points = (midpoint+tangent*lower-other_mid, midpoint+tangent*upper-other_mid)
            positions = sorted(float(np.dot(other_tangent, point)) for point in points)
            new_intervals[new_face].append((*positions, tolerance, index))

    for intervals in (*old_intervals, *new_intervals):
        ordered_intervals = sorted(intervals)
        for left, right in zip(ordered_intervals, ordered_intervals[1:]):
            _require(right[0] >= left[1]-max(left[2], right[2]),
                     'Ambiguous overlapping AMR face coverage')
    remaining = np.ones(a['nf'], np.float64)
    for face, intervals in enumerate(old_intervals):
        if not intervals:
            continue
        total = math.fsum(edges[row[3]][2] for row in intervals)
        tolerance = max(row[2] for row in intervals)/a['length'][face]
        _require(total <= 1.+tolerance, 'AMR face overlap exceeds source length')
        if abs(total-1.) <= tolerance:
            # Close only geometric roundoff; never renormalize a real gap.
            last = intervals[-1][3]
            edges[last][2] = 1.-math.fsum(edges[row[3]][2] for row in intervals[:-1])
            _require(edges[last][2] > 0., 'Invalid roundoff-closed face fraction')
        else:
            remaining[face] = 1.-total
            continue
        remaining[face] = 0.
    edges.sort(key=lambda row: (row[1], row[0]))
    old_face = np.asarray([row[0] for row in edges], np.int64)
    new_face = np.asarray([row[1] for row in edges], np.int64)
    offsets = np.r_[0, np.cumsum(np.bincount(new_face, minlength=b['nf']), dtype=np.int64)]
    unmapped = np.flatnonzero(remaining > 0.)
    return FaceCorrespondence(
        _frozen(old_face, np.int64), _frozen(new_face, np.int64),
        _frozen([row[2] for row in edges], np.float64),
        _frozen([row[3] for row in edges], np.int32),
        _frozen(unmapped, np.int64), _frozen(remaining[unmapped], np.float64),
        _frozen(offsets, np.int64), a['nf'], b['nf'])


_GATHER = r'''
extern "C" __global__ void amr_face_ledger_gather(
    const long long* offset, const long long* oldface, const double* fraction,
    const int* orientation, const double* old_signed, const double* old_gross,
    long long old_nf, long long new_nf, long long total,
    double* signed_out, double* gross_out) {
  long long item=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(item>=total)return;
  long long g=item/new_nf, face=item%new_nf;
  double net=0., gross=0.;
  for(long long k=offset[face];k<offset[face+1];++k) {
    long long source=g*old_nf+oldface[k];
    net+=orientation[k]*fraction[k]*old_signed[source];
    gross+=fraction[k]*old_gross[source];
  }
  signed_out[item]=net; gross_out[item]=gross;
}
'''


def apply_face_ledgers(cp, plan, old_signed, old_gross, new_nf):
    """Gather resident classwise ledgers; return mapped and archived portions.

    One GPU thread owns each class/destination face and reduces source faces
    in fixed CSR order. No floating-point atomic sums or solution host copies
    occur. Archived values retain their old owner-to-neighbour orientation.
    The caller publishes the archive together with its old topology identity.
    """
    _require(isinstance(plan, FaceCorrespondence) and type(new_nf) is int and
             new_nf == plan.new_nf, 'AMR face ledger plan/target differs')
    _require(isinstance(old_signed, cp.ndarray) and isinstance(old_gross, cp.ndarray) and
             old_signed.ndim == 2 and old_signed.shape == old_gross.shape and
             old_signed.shape[0] > 0 and old_signed.shape[1] == plan.old_nf and
             old_signed.dtype == cp.float64 and old_gross.dtype == cp.float64 and
             old_signed.flags.c_contiguous and old_gross.flags.c_contiguous,
             'AMR face ledgers must be resident contiguous class/face FP64 arrays')
    _require(bool(cp.all(cp.isfinite(old_signed)).item()) and
             bool(cp.all(cp.isfinite(old_gross)).item()) and
             bool(cp.all(old_gross >= 0.).item()) and
             bool(cp.all(old_gross + 1.e-12*cp.maximum(1., cp.abs(old_signed)) >=
                         cp.abs(old_signed)).item()), 'Invalid AMR signed/gross face ledger')
    shape = (old_signed.shape[0], new_nf)
    signed, gross = cp.zeros(shape, cp.float64), cp.zeros(shape, cp.float64)
    if signed.size:
        kernel = cp.RawKernel(_GATHER, 'amr_face_ledger_gather', options=(
            '--std=c++11', '--fmad=false', '--prec-div=true', '--prec-sqrt=true', '--ftz=false'))
        kernel(((signed.size+127)//128,), (128,), (
            cp.asarray(plan.new_offsets), cp.asarray(plan.old_face),
            cp.asarray(plan.old_fraction), cp.asarray(plan.orientation),
            old_signed, old_gross, np.int64(plan.old_nf), np.int64(new_nf),
            np.int64(signed.size), signed, gross))
    indices, fraction = cp.asarray(plan.unmapped_old), cp.asarray(plan.unmapped_fraction)
    archived_signed=old_signed[:, indices]*fraction[None, :]
    archived_gross=old_gross[:, indices]*fraction[None, :]
    _require(all(bool(cp.all(cp.isfinite(value)).item()) for value in
                 (signed,gross,archived_signed,archived_gross)),
             'AMR face ledger output overflow/nonfinite value')
    before=cp.sum(old_gross,axis=1)
    after=cp.sum(gross,axis=1)+cp.sum(archived_gross,axis=1)
    tolerance=2048*np.finfo(float).eps*cp.maximum(cp.abs(before),cp.abs(after))
    _require(bool(cp.all(cp.isfinite(before)&cp.isfinite(after)&
                        (cp.abs(after-before)<=tolerance)).item()),
             'AMR face gross ledger lost inventory or overflowed')
    _require(bool(cp.all(gross+1e-12*cp.maximum(1.,cp.abs(signed))>=cp.abs(signed)).item()),
             'AMR face signed/gross output inconsistent')
    return dict(signed=signed, gross=gross, archived_old_face=indices,
                archived_fraction=fraction,
                archived_signed=archived_signed,archived_gross=archived_gross)
