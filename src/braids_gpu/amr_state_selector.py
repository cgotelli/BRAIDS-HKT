"""Host-side accepted-state leaf proposal for the resolved G4 bank-AMR catalog.

This bounded selector follows the enabled indicators in the frozen native
CPU21b7 ``amr_regrid``. It never reads a future native frame or chooses a
nearby catalog mesh. The caller owns accepted full-tree child caches, rock,
and delay counters; this module does not advance or install a GPU state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq

import numpy as np

from .amr_prepared import AMRTopology, PreparedAMR


_CONTROLS = dict(levels=3, refinement_ratio=2, regrid_interval_s=.04,
    coarsen_delay=1, wet_level=1, buffer_cells=2, wet_depth_m=1e-6,
    surface_tolerance_m=.02, velocity_tolerance_m_s=.2,
    refine_wet_dry=True, refine_walls=False, refine_bathymetry=False,
    refine_hydraulics=True, refine_sediment=False, subcycling=False)


@dataclass(frozen=True)
class LeafProposal:
    node_ids: tuple[int, ...]
    topology: AMRTopology | None
    delay_state: np.ndarray
    receipt: dict


class CatalogLayoutMiss(ValueError):
    """The state-based proposal is valid but absent from the frozen catalog."""

    def __init__(self, proposal: LeafProposal):
        self.proposal = proposal
        super().__init__('Accepted-state AMR leaf set is absent from authenticated catalog: '
                         +proposal.receipt['leaf_digest'])


def leaf_digest(ids) -> str:
    values = np.asarray(sorted(map(int,ids)),dtype='<i8')
    return hashlib.sha256(values.tobytes()).hexdigest()


def _host(value, shape, name, *, integer=False):
    # An explicit device-to-host copy is permitted for this host selector.
    if hasattr(value,'get'):
        value = value.get()
    array = np.asarray(value)
    if (array.shape != shape or
            (integer and array.dtype.kind not in 'iu') or
            (not integer and (array.dtype != np.float64 or
                              not np.isfinite(array).all()))):
        raise ValueError('Invalid accepted AMR selector '+name)
    return array.copy()


class ResolvedAMRStateSelector:
    """Native-style tags, forecast, delay, and 2:1 balance for one frozen scope."""

    def __init__(self, prepared: PreparedAMR, *, settings: dict,
                 minimum_levels, tree_rock):
        if prepared.schema != 2 or len(prepared.topologies) < 2 or not all(
                settings.get(key) == value for key,value in _CONTROLS.items()):
            raise ValueError('Resolved bank-AMR selector controls differ')
        self.prepared = prepared
        tree = prepared.tree
        self.parent = np.asarray(tree['parent'],dtype=np.int64)
        self.level = np.asarray(tree['node_level'],dtype=np.int64)
        self.maximum = np.asarray(tree['maximum_level'],dtype=np.int64)
        self.xy = np.asarray(tree['xy'],dtype=np.float64)
        self.area = np.asarray(tree['area'],dtype=np.float64)
        self.nn = len(self.parent)
        self.minimum = _host(minimum_levels,(self.nn,),
                             'minimum levels',integer=True).astype(np.int64)
        self.rock = _host(tree_rock,(self.nn,),'tree rock')
        required = np.where((self.xy[:,1]>=2.5)|(self.xy[:,0]>=15.5),2,0)
        if (not np.array_equal(self.minimum,required) or
                np.any(self.minimum>self.maximum) or
                np.any(self.level>self.maximum)):
            raise ValueError('Resolved bank/outlet minimum-level geometry differs')
        self.children = tuple(tuple(map(int,np.flatnonzero(self.parent == node)))
                              for node in range(self.nn))
        for topology in prepared.topologies:
            if not np.array_equal(self.rock[topology.node_ids],topology.data['rock']):
                raise ValueError('Resolved AMR tree rock differs from catalog')
        self.by_digest = {}
        outlet = toe = bank = None
        for topology in prepared.topologies:
            digest = leaf_digest(topology.node_ids)
            if digest in self.by_digest:
                raise ValueError('Duplicate AMR catalog leaf set')
            self.by_digest[digest] = topology
            data,ids = topology.data,topology.node_ids
            bfaces = np.asarray(data['boundary_faces'],dtype=np.int64)
            outlet_now = set(map(int,ids[np.asarray(data['owner'])[bfaces]]))
            faces = np.flatnonzero(np.asarray(data['is_toe_face']))
            toe_now = (set(map(int,ids[np.asarray(data['candidate_bank'])[faces]])) |
                       set(map(int,ids[np.asarray(data['candidate_toe'])[faces]])))
            bank_now = set(map(int,ids[np.flatnonzero(data['bank_material_mask'])]))
            if outlet is None:
                outlet,toe,bank = outlet_now,toe_now,bank_now
            elif outlet_now != outlet or toe_now != toe or bank_now != bank:
                raise ValueError('Resolved AMR protected bank/toe/outlet nodes differ')
        self.protected = frozenset(outlet | toe | bank)
        if (len(outlet) != 16 or len(toe) != 40 or
                not np.all(self.level[list(self.protected)] == 2)):
            raise ValueError('Resolved AMR protected finest-node map differs')
        self.interval = float(settings['regrid_interval_s'])
        self.wet_depth = float(settings['wet_depth_m'])
        self.eta_tol = float(settings['surface_tolerance_m'])
        self.velocity_tol = float(settings['velocity_tolerance_m_s'])

    def _bed(self, rows, nodes, topology):
        data = topology.data
        layers = np.asarray(data['bed_indices'],dtype=np.int64)
        bulk = np.asarray(data['bulk_fraction'],dtype=np.float64)
        return self.rock[nodes]+np.sum(rows[:,layers]/bulk[None,None,:],axis=(1,2))

    def _sync_tree(self, topology, accepted, cached):
        tree_rows = cached.copy()
        ids = topology.node_ids
        tree_rows[ids] = accepted
        selected = set(map(int,ids))
        covered = np.full(self.nn,-1,dtype=np.int64)
        for node in range(self.nn):
            parent = self.parent[node]
            covered[node] = node if node in selected else (
                covered[parent] if parent >= 0 else -1)
        for node in range(self.nn-1,-1,-1):
            group = self.children[node]
            if covered[node] < 0 and group:
                weights = self.area[list(group)]
                tree_rows[node] = np.dot(weights,tree_rows[list(group)])/sum(weights)
        return tree_rows

    def _face_pairs(self, ids):
        """Rectangular selected leaves share a face if boxes touch with overlap."""
        xy = self.xy[np.asarray(ids,dtype=np.int64)]
        half = .5*np.sqrt(self.area[np.asarray(ids,dtype=np.int64)])
        dx = np.abs(xy[:,None,0]-xy[None,:,0])
        dy = np.abs(xy[:,None,1]-xy[None,:,1])
        edge = half[:,None]+half[None,:]
        horizontal = (np.abs(dx-edge)<1e-10) & (dy<edge-1e-10)
        vertical = (np.abs(dy-edge)<1e-10) & (dx<edge-1e-10)
        return zip(*np.nonzero(np.triu(horizontal|vertical,1)))

    def _select(self, wanted):
        selected = []

        def descend(node):
            group = self.children[node]
            if self.level[node] < wanted[node] and group:
                for child in group:
                    descend(child)
            else:
                selected.append(node)

        for node in np.flatnonzero(self.parent < 0):
            descend(int(node))
        for _ in range(self.nn):
            to_refine = set()
            for ai,bi in self._face_pairs(selected):
                a,b = selected[ai],selected[bi]
                if abs(self.level[a]-self.level[b]) <= 1:
                    continue
                coarse = a if self.level[a] < self.level[b] else b
                if self.level[coarse]>=self.maximum[coarse] or not self.children[coarse]:
                    raise ValueError('Resolved AMR 2:1 safety cap cannot be balanced')
                to_refine.add(coarse)
            if not to_refine:
                return tuple(sorted(selected))
            selected = [node for node in selected if node not in to_refine]
            for node in sorted(to_refine):
                selected.extend(self.children[node])
        raise RuntimeError('Resolved AMR 2:1 balance did not converge')

    def propose(self, topology: AMRTopology, accepted_state, *,
                cached_tree_state, delay_state, time_s: float,
                initial: bool = False) -> LeafProposal:
        """Propose from current state; raise CatalogLayoutMiss on exact-set miss."""
        if not any(top is topology for top in self.prepared.topologies):
            raise ValueError('Current AMR topology is outside authenticated catalog')
        n = len(topology.node_ids)
        nstate = int(topology.data['nstate'])
        state = _host(accepted_state,(n,nstate),'selected state')
        cache = _host(cached_tree_state,(self.nn,nstate),'full-tree cache')
        delay = _host(delay_state,(self.nn,),'delay state',integer=True).astype(np.int64)
        if (np.any(delay<0) or not np.isfinite(time_s) or time_s<0 or
                abs(time_s/self.interval-round(time_s/self.interval))>1e-9):
            raise ValueError('Accepted AMR regrid clock/delay differs')
        if (np.any(state[:,0]<-1e-12) or
                np.any(state[:,np.asarray(topology.data['bed_indices']).ravel()]<-1e-12)):
            raise ValueError('Invalid accepted AMR water/bed state')
        ids = np.asarray(topology.node_ids,dtype=np.int64)
        data = topology.data
        tree_rows = self._sync_tree(topology,state,cache)
        bed = self._bed(state,ids,topology)
        depth = state[:,0]
        stage = depth+bed
        wet = depth>self.wet_depth
        dry = float(data['dry_tolerance'])
        velocity = np.zeros((n,2),dtype=np.float64)
        moving = depth>dry
        velocity[moving] = state[moving,1:3]/depth[moving,None]
        raw = self.minimum.copy()
        cause = np.zeros(n,dtype=np.int32)
        raw[ids[wet]] = np.maximum(raw[ids[wet]],1)
        cause[wet] |= 1<<1
        local = {int(node):i for i,node in enumerate(ids)}
        wet_front = np.zeros(n,dtype=bool)
        moving_source = np.zeros(n,dtype=bool)
        adjacency = [[] for _ in range(n)]
        owner = np.asarray(data['owner'],dtype=np.int64)
        neighbour = np.asarray(data['neighbour'],dtype=np.int64)
        for a,b in zip(owner,neighbour):
            if b<0:
                continue
            adjacency[a].append(int(b));adjacency[b].append(int(a))
            if wet[a] != wet[b]:
                raw[ids[[a,b]]] = 2
                cause[[a,b]] |= 1<<2
                wet_front[[a,b]] |= wet[[a,b]]
                moving_source[[a,b]] = True
            if wet[a] and wet[b] and (abs(stage[a]-stage[b])>self.eta_tol or
                    np.linalg.norm(velocity[a]-velocity[b])>self.velocity_tol):
                raw[ids[[a,b]]] = 2
                cause[[a,b]] |= 1<<3
                moving_source[[a,b]] = True
            if (wet[a] or wet[b]) and (
                    data['manning_cell'][a] != data['manning_cell'][b] or
                    bool(data['bank_material_mask'][a]) !=
                    bool(data['bank_material_mask'][b])):
                raw[ids[[a,b]]] = np.maximum(raw[ids[[a,b]]],2)
                cause[[a,b]] |= 1<<7
        for face in np.flatnonzero(neighbour<0):
            if data['bid'][face]>=0:
                a = owner[face]
                raw[ids[a]]=2;cause[a] |= 1<<8
        for i,node in enumerate(ids):
            if not wet[i] or not self.children[int(node)]:
                continue
            child = np.asarray(self.children[int(node)],dtype=np.int64)
            child_bed = self._bed(tree_rows[child],child,topology)
            if np.any(child_bed>=stage[i]-self.wet_depth):
                raw[node]=2;cause[i] |= 1<<12;moving_source[i]=True
        raw[list(self.protected)] = 2

        reach = np.full(n,-np.inf,dtype=np.float64)
        buffer = 2*max(float(data['dx']),float(data['dy']))
        heap = []
        for i,node in enumerate(ids):
            if raw[node] != 2:
                continue
            speed = (np.linalg.norm(velocity[i])+
                     (2 if wet_front[i] else 1)*np.sqrt(float(data['grav'])*max(depth[i],0))) \
                    if moving_source[i] and wet[i] else 0.
            reach[i] = buffer+speed*self.interval
            heapq.heappush(heap,(-reach[i],i))
        while heap:
            negative,i = heapq.heappop(heap)
            if -negative<reach[i]-1e-12:
                continue
            for j in adjacency[i]:
                distance = np.linalg.norm(self.xy[ids[i]]-self.xy[ids[j]])
                candidate = reach[i]-distance
                if candidate>=0 and candidate>reach[j]+1e-12:
                    reach[j]=candidate
                    heapq.heappush(heap,(-candidate,j))
        buffered = reach>=0
        raw[ids[buffered]]=2
        cause[buffered] |= 1<<14

        vetoed = []
        for parent in range(self.nn):
            group = self.children[parent]
            if len(group)!=4 or not all(child in local for child in group):
                continue
            pos = np.asarray([local[child] for child in group])
            solver_wet = depth[pos]>dry
            mixed = bool(np.any(solver_wet) and np.any(~solver_wet))
            span = float(np.ptp(stage[pos][solver_wet])) if np.any(solver_wet) else 0.
            if mixed or span>.020:
                raw[list(group)]=2
                vetoed.append(dict(parent_id=parent,mixed_wet_dry=mixed,
                                   wet_stage_span_m=span))
        if not initial:
            for node in ids:
                if raw[node]>=self.level[node]:
                    delay[node]=0
                else:
                    delay[node]+=1
                    if delay[node]<_CONTROLS['coarsen_delay']:
                        raw[node]=self.level[node]
        wanted = raw.copy()
        for node in range(self.nn):
            parent = self.parent[node]
            if parent>=0:
                wanted[node]=max(wanted[node],wanted[parent])
        for node in range(self.nn-1,-1,-1):
            parent = self.parent[node]
            if parent>=0:
                wanted[parent]=max(wanted[parent],wanted[node])
        wanted = np.minimum(wanted,self.maximum)
        selected = self._select(wanted)
        if not self.protected.issubset(selected):
            raise ValueError('Resolved AMR proposal lost a finest bank/toe/outlet node')
        digest = leaf_digest(selected)
        matched = self.by_digest.get(digest)
        receipt = dict(catalog_identity=self.prepared.identity,
            old_operator_identity=topology.source_identity,
            proposed_operator_identity=(None if matched is None else matched.source_identity),
            leaf_digest=digest,old_leaves=n,new_leaves=len(selected),
            time_s=float(time_s),state_based=True,native_mesh_schedule_replayed=False,
            host_selector=True,forecast_cells=int(np.count_nonzero(buffered)),
            wet_dry_trigger_cells=int(np.count_nonzero(cause&(1<<2))),
            hydraulic_trigger_cells=int(np.count_nonzero(cause&(1<<3))),
            wet_child_bed_trigger_cells=int(np.count_nonzero(cause&(1<<12))),
            unsafe_coarsen_veto=vetoed,
            protected_finest_nodes=len(self.protected),
            catalog_hit=matched is not None)
        result = LeafProposal(selected,matched,delay,receipt)
        if matched is None:
            raise CatalogLayoutMiss(result)
        return result
