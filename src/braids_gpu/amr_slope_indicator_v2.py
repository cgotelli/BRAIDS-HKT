"""Versioned CUDA-resident slope-aware AMR research selector.

The accepted hydraulic/sediment arrays stay on device.  The existing resident
forecast kernel propagates new slope-feature seeds; only leaf/delay metadata
and compact trigger masks cross to the host.  This does not alter the legacy
selector or any existing restart contract.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .amr_general_selector import GeneralLeafProposal, leaf_digest
from .amr_slope_indicator_research import ResearchSlopeAwareSelector, require


RULE = 'eta_plane_residual_resident_forecast_v2'


class VersionedSlopeAwareSelector(ResearchSlopeAwareSelector):
    """Explicit alternative to the legacy absolute stage-jump selector."""

    def __init__(self, original):
        super().__init__(original)
        contract=dict(rule=RULE, legacy_selector=original.contract_identity,
            no_eta_baseline=self.base.contract_identity,
            surface_tolerance_m=original.eta_tolerance,
            coarsen_stage_span_m=original.stage_span,
            physical_state_download=False,
            forecast_kernel='gi_reach_abi2')
        self.contract_identity=hashlib.sha256(json.dumps(contract,
            sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def __getattr__(self, name):
        # The on-demand route reads immutable settings and the tree-bed helper.
        return getattr(self.original,name)

    def _buffer_device(self, topology, rows, wet, seeds):
        import cupy as cp

        s=self.original
        resident=self.base._resident_indicators
        require(resident is not None and
                isinstance(rows,cp.ndarray) and
                isinstance(wet,cp.ndarray) and
                isinstance(seeds,cp.ndarray),
                'CUDA slope forecast requires resident accepted state and baseline geometry')
        n=len(topology.node_ids)
        g=resident._geometry(topology)
        depth=rows[:,0]
        moving=depth>float(topology.data['dry_tolerance'])
        safe=cp.where(moving,depth,1.)
        u=cp.where(moving,rows[:,1]/safe,0.)
        v=cp.where(moving,rows[:,2]/safe,0.)
        speed=cp.where(seeds & wet,cp.sqrt(u*u+v*v)+cp.sqrt(
            float(topology.data['grav'])*cp.maximum(depth,0.)),0.)
        buffer=s.settings['buffer_cells']*max(float(topology.data['dx']),
                                              float(topology.data['dy']))
        reach=cp.where(seeds,buffer+speed*s.interval,-cp.inf)
        following=cp.empty_like(reach)
        changed=cp.zeros((),cp.int32)
        for _ in range(n):
            changed.fill(0)
            resident._launch('gi_reach',n,(np.int32(n),g['offset'],
                g['adjacent'],g['distance'],reach,following,changed))
            reach,following=following,reach
            if int(changed.item())==0:
                break
        else:
            raise RuntimeError('CUDA slope-aware forecast did not converge')
        require(bool(cp.all(cp.isfinite(speed))),
                'Nonfinite slope-aware hydraulic forecast speed')
        return reach>=0

    def propose(self, topology, accepted_state, *, cached_tree_state,
                delay_state, time_s, initial=False,
                selector_last_bed=None, bed_gradient=None):
        if isinstance(accepted_state,np.ndarray):
            xp,backend=np,'cpu'
        else:
            import cupy as cp
            require(isinstance(accepted_state,cp.ndarray),
                    'Slope-aware accepted state requires explicit CPU or CUDA array')
            xp,backend=cp,'cuda'
        base=self.base.propose(topology,accepted_state,
            cached_tree_state=cached_tree_state,delay_state=delay_state,
            time_s=time_s,initial=initial,
            selector_last_bed=selector_last_bed,bed_gradient=bed_gradient)
        eta,wet=self._eta_wet(topology,accepted_state,backend)
        geometry=self._geometry(topology)
        residual=geometry.residuals(eta,wet,backend=backend,
            tolerance_m=self.original.eta_tolerance,
            stage_span_m=self.original.stage_span)
        face=residual['face_trigger']
        veto=residual['parent_veto']
        seeds=xp.zeros(len(topology.node_ids),dtype=xp.bool_)
        if backend=='cuda':
            owner,neighbour,groups=(xp.asarray(value) for value in
                (geometry.owner,geometry.neighbour,geometry.groups))
        else:
            owner,neighbour,groups=(geometry.owner,geometry.neighbour,
                                    geometry.groups)
        seeds[owner[face]]=True
        seeds[neighbour[face]]=True
        if len(geometry.groups):
            seeds[groups[veto].ravel()]=True
        if backend=='cuda':
            buffered=self._buffer_device(topology,accepted_state,wet,seeds)
            feature=xp.asnumpy(buffered)
            face_count=int(xp.count_nonzero(face).item())
            seed_count=int(xp.count_nonzero(seeds).item())
            veto_count=int(xp.count_nonzero(veto).item())
        else:
            feature=self._buffer(topology,accepted_state,wet,seeds)
            face_count=int(np.count_nonzero(face))
            seed_count=int(np.count_nonzero(seeds))
            veto_count=int(np.count_nonzero(veto))
        s=self.base
        wanted=s.minimum.copy()
        for node in base.node_ids:
            ancestor=int(node)
            while ancestor>=0:
                wanted[ancestor]=max(wanted[ancestor],int(s.level[node]))
                ancestor=int(s.parent[ancestor])
        tagged=np.flatnonzero(feature)
        for i in tagged:
            ancestor=int(topology.node_ids[i])
            while ancestor>=0:
                wanted[ancestor]=max(wanted[ancestor],s.levels-1)
                ancestor=int(s.parent[ancestor])
        wanted=np.minimum(wanted,s.maximum)
        selected=s._selected(wanted,current=topology.node_ids
                             if s.dynamic_layout else None)
        protected=s._protected(topology)
        require(protected.issubset(selected),
                'Slope-aware candidate lost protected open boundary')
        delay=base.delay_state.copy()
        delay[topology.node_ids[tagged]]=0
        digest=leaf_digest(selected)
        match=s.by_digest.get(digest)
        receipt=dict(base.receipt)
        receipt.update(selector_contract_identity=self.contract_identity,
            indicator_rule=RULE,numerical_indicator_backend=backend,
            physical_state_download=False if backend=='cuda' else None,
            leaf_digest=digest,new_leaves=len(selected),
            catalog_hit=match is not None,
            proposed_operator_identity=None if match is None else match.source_identity,
            slope_face_trigger_faces=face_count,
            slope_face_trigger_cells=seed_count,
            slope_parent_vetoes=veto_count,
            slope_forecast_buffer_cells=int(np.count_nonzero(feature)),
            old_absolute_eta_trigger_suppressed=True,
            old_absolute_stage_span_veto_suppressed=True)
        return GeneralLeafProposal(selected,match,delay,receipt)
