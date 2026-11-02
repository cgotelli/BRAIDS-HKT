"""Research-only slope-aware hydraulic AMR tagging; legacy selector unchanged.

Local least-squares eta planes remove a *linear* background water-surface
slope.  Face residuals and four-child plane residuals detect localized bends,
pulses and jumps.  CPU and CUDA use the same immutable two-ring geometry and
algebra.  A wrapper delegates all other tags (wet/dry, velocity, sediment,
wall, open boundaries, buffering) to the existing selector under an explicit
versioned contract; it never changes a legacy restart identity.
"""
from __future__ import annotations

import hashlib
import heapq
import json

import numpy as np

from .amr_general_selector import (GeneralAMRStateSelector,
    GeneralLeafProposal, leaf_digest)


RULE = 'research_local_eta_plane_residual_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


class SlopeAwareHydraulicGeometry:
    """Immutable two-ring least-squares geometry for one selected topology."""

    def __init__(self, selector, topology):
        ids = np.asarray(topology.node_ids, dtype=np.int64)
        n = len(ids)
        data = topology.data
        xy = np.asarray(selector.xy[ids], np.float64)
        owner = np.asarray(data['owner'], np.int32)
        neighbour = np.asarray(data['neighbour'], np.int32)
        internal = neighbour >= 0
        self.n = n
        self.xy = xy
        self.owner = owner[internal]
        self.neighbour = neighbour[internal]
        self.dx_face = xy[self.neighbour]-xy[self.owner]
        adjacent = [set() for _ in range(n)]
        for a, b in zip(self.owner, self.neighbour):
            adjacent[int(a)].add(int(b))
            adjacent[int(b)].add(int(a))
        center, other = [], []
        for i, ring in enumerate(adjacent):
            two = set(ring)
            for j in ring:
                two.update(adjacent[j])
            two.discard(i)
            for j in sorted(two):
                center.append(i)
                other.append(j)
        self.center = np.asarray(center, np.int32)
        self.other = np.asarray(other, np.int32)
        self.dx = xy[self.other]-xy[self.center]
        norm2 = np.sum(self.dx*self.dx, axis=1)
        require(np.all(norm2 > 0) and np.isfinite(norm2).all(),
                'Slope-aware AMR has duplicate/invalid selected cell centers')
        self.weight = 1./norm2
        local = {int(node):i for i,node in enumerate(ids)}
        parents = [node for node,children in enumerate(selector.children)
                   if len(children) == 4 and all(c in local for c in children)]
        self.parents = np.asarray(parents, np.int64)
        self.groups = np.asarray([[local[c] for c in selector.children[p]]
                                  for p in parents], np.int32).reshape((-1,4))
        self._device = None

    def _arrays(self, xp):
        if xp is np:
            return self.center,self.other,self.dx,self.weight,self.owner,self.neighbour,self.dx_face,self.groups
        if self._device is None:
            self._device = tuple(xp.asarray(value) for value in
                (self.center,self.other,self.dx,self.weight,self.owner,
                 self.neighbour,self.dx_face,self.groups))
        return self._device

    def residuals(self, eta, wet, *, backend='cpu', tolerance_m,
                  stage_span_m):
        """Return all selected-face and four-child residuals without state mutation."""
        if backend == 'cpu':
            xp = np
        elif backend == 'cuda':
            import cupy as xp
        else:
            raise ValueError('Explicit CPU or CUDA slope-indicator backend required')
        require(isinstance(eta,xp.ndarray) and eta.shape == (self.n,) and
                eta.dtype == xp.float64 and isinstance(wet,xp.ndarray) and
                wet.shape == (self.n,) and
                np.isfinite(tolerance_m) and tolerance_m > 0 and
                np.isfinite(stage_span_m) and stage_span_m > 0,
                'Invalid slope-aware AMR eta/wet state or thresholds')
        c,j,dx,w,a,b,face_dx,groups = self._arrays(xp)
        delta = eta[j]-eta[c]
        dx0,dx1 = dx[:,0],dx[:,1]
        bincount = lambda values: xp.bincount(c, weights=values, minlength=self.n)
        axx = bincount(w*dx0*dx0)
        axy = bincount(w*dx0*dx1)
        ayy = bincount(w*dx1*dx1)
        bx = bincount(w*dx0*delta)
        by = bincount(w*dx1*delta)
        det = axx*ayy-axy*axy
        trace = axx+ayy
        full = det > 1e-13*trace*trace
        safe_det = xp.where(full,det,1.)
        safe_trace = xp.where(trace>0,trace,1.)
        gx = xp.where(full,(ayy*bx-axy*by)/safe_det,bx/safe_trace)
        gy = xp.where(full,(axx*by-axy*bx)/safe_det,by/safe_trace)
        gradient = xp.stack((gx,gy),axis=1)
        face_residual = eta[b]-eta[a]-xp.sum(
            .5*(gradient[a]+gradient[b])*face_dx,axis=1)
        face_trigger = (wet[a] & wet[b] &
                        (xp.abs(face_residual)>tolerance_m))
        if len(self.groups):
            coordinates = xp.asarray(self.xy)[groups]
            centered = coordinates-xp.mean(coordinates,axis=1,keepdims=True)
            values = eta[groups]
            centered_values = values-xp.mean(values,axis=1,keepdims=True)
            x,y = centered[:,:,0],centered[:,:,1]
            aa = xp.sum(x*x,axis=1); ab = xp.sum(x*y,axis=1)
            bb = xp.sum(y*y,axis=1)
            qx = xp.sum(x*centered_values,axis=1)
            qy = xp.sum(y*centered_values,axis=1)
            determinant = aa*bb-ab*ab
            valid = determinant>1e-13*(aa+bb)**2
            denominator = xp.where(valid,determinant,1.)
            pgx = xp.where(valid,(bb*qx-ab*qy)/denominator,0.)
            pgy = xp.where(valid,(aa*qy-ab*qx)/denominator,0.)
            plane_residual = xp.max(xp.abs(centered_values-
                (pgx[:,None]*x+pgy[:,None]*y)),axis=1)
            mixed_wet = xp.any(wet[groups],axis=1) & xp.any(~wet[groups],axis=1)
            veto = (~mixed_wet) & (plane_residual>stage_span_m)
        else:
            plane_residual = xp.empty(0,xp.float64)
            veto = xp.empty(0,xp.bool_)
        require(bool(xp.all(xp.isfinite(gradient))) and
                bool(xp.all(xp.isfinite(face_residual))) and
                bool(xp.all(xp.isfinite(plane_residual))),
                'Nonfinite slope-aware AMR residual')
        return dict(gradient=gradient,face_residual=face_residual,
            face_trigger=face_trigger,parent_residual=plane_residual,
            parent_veto=veto)


class ResearchSlopeAwareSelector:
    """Versioned candidate selector retaining every non-eta legacy trigger."""

    def __init__(self, original: GeneralAMRStateSelector):
        require(isinstance(original, GeneralAMRStateSelector),
                'Slope-aware research selector needs an authenticated original')
        self.original = original
        settings = dict(original.settings)
        settings['surface_tolerance_m'] = 1e9
        self.base = GeneralAMRStateSelector(original.prepared,
            settings=settings,minimum_levels=original.minimum,
            tree_rock=original.rock,finest_face_pairs=original.finest_pairs,
            protected_finest_ids=original.explicit_protected,
            coarsen_stage_span_m=1e9,
            evolving_banks=original.evolving_banks,
            indicator_source_path=original.indicator_source_path)
        contract = dict(rule=RULE,old_contract=original.contract_identity,
            no_eta_base_contract=self.base.contract_identity,
            surface_tolerance_m=original.eta_tolerance,
            coarsen_stage_span_m=original.stage_span)
        self.contract_identity = hashlib.sha256(json.dumps(contract,
            sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.geometry = {}

    def _geometry(self, topology):
        key = id(topology)
        if key not in self.geometry:
            self.geometry[key] = SlopeAwareHydraulicGeometry(self.original,
                                                             topology)
        return self.geometry[key]

    def _eta_wet(self, topology, rows, backend):
        original, data = self.original, topology.data
        if backend == 'cpu':
            xp = np
        else:
            import cupy as xp
        indices = xp.asarray(data['bed_indices'])
        bulk = xp.asarray(data['bulk_fraction'])
        bed = (xp.asarray(original.rock[topology.node_ids])+
               xp.sum(rows[:,indices]/bulk[None,None,:],axis=(1,2)))
        eta = xp.ascontiguousarray(bed+rows[:,0])
        wet = rows[:,0] > original.wet_depth
        return eta,wet

    def _buffer(self, topology, rows, wet, seeds):
        """Same physical reach rule as legacy, for newly added eta sources."""
        s = self.original
        data,ids = topology.data,topology.node_ids
        xy = s.xy[ids]
        depth = rows[:,0]
        speed = np.zeros(len(ids),np.float64)
        valid = wet & (depth > float(data['dry_tolerance']))
        speed[valid] = np.linalg.norm(rows[valid,1:3]/depth[valid,None],axis=1)+\
            np.sqrt(float(data['grav'])*depth[valid])
        owner,other = np.asarray(data['owner']),np.asarray(data['neighbour'])
        adjacent=[[] for _ in ids]
        for a,b in zip(owner,other):
            if b>=0:
                distance = float(np.linalg.norm(xy[a]-xy[b]))
                adjacent[a].append((int(b),distance))
                adjacent[b].append((int(a),distance))
        reach=np.full(len(ids),-np.inf)
        buffer=s.settings['buffer_cells']*max(float(data['dx']),float(data['dy']))
        heap=[]
        for i in np.flatnonzero(seeds):
            reach[i]=buffer+speed[i]*s.interval
            heapq.heappush(heap,(-reach[i],int(i)))
        while heap:
            negative,i=heapq.heappop(heap)
            if -negative<reach[i]-1e-12:
                continue
            for j,distance in adjacent[i]:
                candidate=reach[i]-distance
                if candidate>=0 and candidate>reach[j]+1e-12:
                    reach[j]=candidate
                    heapq.heappush(heap,(-candidate,j))
        return reach>=0

    def propose(self, topology, accepted_state, *, cached_tree_state,
                delay_state, time_s, initial=False,
                selector_last_bed=None, bed_gradient=None):
        xp = np if isinstance(accepted_state,np.ndarray) else None
        if xp is None:
            import cupy as cp
            require(isinstance(accepted_state,cp.ndarray),
                    'Slope-aware accepted state requires explicit CPU or CUDA array')
            xp,backend=cp,'cuda'
        else:
            backend='cpu'
        base = self.base.propose(topology,accepted_state,
            cached_tree_state=cached_tree_state,delay_state=delay_state,
            time_s=time_s,initial=initial,
            selector_last_bed=selector_last_bed,bed_gradient=bed_gradient)
        eta,wet = self._eta_wet(topology,accepted_state,backend)
        geometry = self._geometry(topology)
        residual = geometry.residuals(eta,wet,backend=backend,
            tolerance_m=self.original.eta_tolerance,
            stage_span_m=self.original.stage_span)
        face = xp.asnumpy(residual['face_trigger']) if backend=='cuda' else residual['face_trigger']
        veto = xp.asnumpy(residual['parent_veto']) if backend=='cuda' else residual['parent_veto']
        seeds = np.zeros(len(topology.node_ids),dtype=bool)
        seeds[geometry.owner[face]] = True
        seeds[geometry.neighbour[face]] = True
        if np.any(veto):
            seeds[geometry.groups[veto].ravel()] = True
        host_rows = (xp.asnumpy(accepted_state) if backend=='cuda' else accepted_state)
        host_wet = (xp.asnumpy(wet) if backend=='cuda' else wet)
        buffered = self._buffer(topology,host_rows,host_wet,seeds)
        s = self.base
        wanted = s.minimum.copy()
        for node in base.node_ids:
            ancestor = int(node)
            while ancestor>=0:
                wanted[ancestor] = max(wanted[ancestor],int(s.level[node]))
                ancestor = int(s.parent[ancestor])
        tagged = np.flatnonzero(buffered)
        for i in tagged:
            ancestor = int(topology.node_ids[i])
            while ancestor>=0:
                wanted[ancestor] = max(wanted[ancestor],s.levels-1)
                ancestor = int(s.parent[ancestor])
        wanted = np.minimum(wanted,s.maximum)
        selected = s._selected(wanted,
            current=topology.node_ids if s.dynamic_layout else None)
        protected=s._protected(topology)
        require(protected.issubset(selected),
                'Slope-aware research candidate lost protected open boundary')
        delay = base.delay_state.copy()
        delay[topology.node_ids[tagged]] = 0
        digest=leaf_digest(selected)
        match=s.by_digest.get(digest)
        receipt=dict(base.receipt)
        receipt.update(selector_contract_identity=self.contract_identity,
            indicator_rule=RULE, numerical_indicator_backend=backend,
            leaf_digest=digest,new_leaves=len(selected),catalog_hit=match is not None,
            proposed_operator_identity=None if match is None else match.source_identity,
            slope_face_trigger_faces=int(np.count_nonzero(face)),
            slope_face_trigger_cells=int(np.count_nonzero(seeds)),
            slope_parent_vetoes=int(np.count_nonzero(veto)),
            slope_forecast_buffer_cells=int(np.count_nonzero(buffered)),
            old_absolute_eta_trigger_suppressed=True,
            old_absolute_stage_span_veto_suppressed=True)
        return GeneralLeafProposal(selected,match,delay,receipt)
