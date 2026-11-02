"""Accepted-state AMR leaf proposal using an immutable hierarchy.

The legacy indicators are wet/dry, hydraulic surface and velocity jumps,
wet-child bed emergence, material edges, forecast buffering, coarsen delay,
and 2:1 balance. Versioned extensions add walls, native bathymetric detail and
sediment fronts/evolution with explicit accepted observation inputs.
A catalog miss returns the exact proposed
global leaf IDs for an on-demand operator builder. No future native state or
recorded mesh sequence participates in selection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json

import numpy as np

from .amr_prepared import AMRTopology, PreparedAMR
from .general_live_transport import recursive_bedform_envelope


INDICATOR_STATS = dict(concentration_trigger_cells=4, bank_relief_trigger_cells=5,
    composition_trigger_cells=6, wall_trigger_cells=9, bathymetry_trigger_cells=10,
    temporal_bed_trigger_cells=11, ancestor_bathymetry_trigger_cells=13)
EXTENDED_INDICATOR_RULE = 'native_directed_csr_v1'


@dataclass(frozen=True)
class GeneralLeafProposal:
    node_ids: tuple[int, ...]
    topology: AMRTopology | None
    delay_state: np.ndarray
    receipt: dict


def leaf_digest(ids) -> str:
    return hashlib.sha256(np.asarray(sorted(map(int,ids)),
                                     dtype='<i8').tobytes()).hexdigest()


def _host(value, shape, name, *, integer=False):
    if hasattr(value,'get'):
        value = value.get()
    array = np.asarray(value)
    if (array.shape != shape or
            (integer and array.dtype.kind not in 'iu') or
            (not integer and (array.dtype != np.float64 or
                              not np.isfinite(array).all()))):
        raise ValueError('Invalid AMR selector '+name)
    return array.copy()


class GeneralAMRStateSelector:
    """Propose native-style leaves from current accepted state and tree cache."""

    INDICATOR_STATS = INDICATOR_STATS

    def __init__(self, prepared: PreparedAMR, *, settings: dict,
                 minimum_levels, tree_rock, finest_face_pairs,
                 protected_finest_ids=(), coarsen_stage_span_m=.020,
                 evolving_banks=False, indicator_source_path=None):
        if prepared.schema not in (2, 3) or not prepared.topologies:
            raise ValueError('General AMR selector requires authenticated bank or transport geometry')
        # Mesh selection reads synchronized accepted states. Time integration is
        # the route's responsibility, not a reason to alter these indicators.
        if type(settings.get('subcycling')) is not bool:
            raise ValueError('AMR subcycling must be an explicit boolean')
        extra_flags = ('refine_bathymetry','refine_sediment','refine_walls')
        if any(type(settings.get(key)) is not bool for key in extra_flags):
            raise ValueError('Extended AMR indicator flags must be explicit booleans')
        self.extended_indicators = any(settings[key] for key in extra_flags)
        if not settings.get('refine_wet_dry') or not settings.get('refine_hydraulics'):
            raise ValueError('General selector currently requires wet/dry and hydraulic indicators')
        levels = settings.get('levels')
        if (type(levels) is not int or levels < 2 or
                settings.get('refinement_ratio') != 2 or
                type(settings.get('coarsen_delay')) is not int or
                settings['coarsen_delay'] < 1 or
                type(settings.get('buffer_cells')) is not int or
                settings['buffer_cells'] < 0 or
                type(settings.get('wet_level')) is not int or
                not 1<=settings['wet_level']<levels):
            raise ValueError('Invalid AMR hierarchy/selection controls')
        for name in ('regrid_interval_s','wet_depth_m','surface_tolerance_m',
                     'velocity_tolerance_m_s'):
            value = settings.get(name)
            if not isinstance(value,(float,int)) or not np.isfinite(value) or value<=0:
                raise ValueError('Invalid AMR selector control: '+name)
        controls = dict(bed_tolerance=('bed_tolerance_m',settings['refine_bathymetry']),
            c_tolerance=('concentration_tolerance_kg_m3',settings['refine_sediment']),
            mix_tolerance=('composition_tolerance',settings['refine_sediment']),
            dz_tolerance=('bed_change_tolerance_m',settings['refine_sediment']))
        for attribute,(name,enabled) in controls.items():
            value=settings.get(name)
            if self.extended_indicators and (type(value) not in (float,int) or not np.isfinite(value) or value<=0):
                raise ValueError('Invalid AMR selector control: '+name)
            if self.extended_indicators and name=='composition_tolerance' and value>1:
                raise ValueError('Invalid AMR selector control: '+name)
            setattr(self,attribute,float(value) if self.extended_indicators else None)
        if not np.isfinite(coarsen_stage_span_m) or coarsen_stage_span_m<=0:
            raise ValueError('Invalid resolved wet-stage coarsen guard')
        self.prepared = prepared
        if type(evolving_banks) is not bool:
            raise ValueError('evolving_banks must be an explicit boolean')
        self.evolving_banks = evolving_banks
        self.general_transport = prepared.schema == 3
        self.dynamic_layout = evolving_banks or self.general_transport
        if self.general_transport and evolving_banks:
            raise ValueError('General transport AMR cannot claim evolving banks')
        self.indicator_source_path = indicator_source_path
        self._resident_indicators = None
        self.settings = dict(settings)
        self.levels = levels
        self.interval = float(settings['regrid_interval_s'])
        self.wet_depth = float(settings['wet_depth_m'])
        self.eta_tolerance = float(settings['surface_tolerance_m'])
        self.velocity_tolerance = float(settings['velocity_tolerance_m_s'])
        self.stage_span = float(coarsen_stage_span_m)
        tree = prepared.tree
        self.parent = np.asarray(tree['parent'],dtype=np.int64)
        self.level = np.asarray(tree['node_level'],dtype=np.int64)
        self.maximum = np.asarray(tree['maximum_level'],dtype=np.int64)
        self.xy = np.asarray(tree['xy'],dtype=np.float64)
        self.area = np.asarray(tree['area'],dtype=np.float64)
        self.nn = len(self.parent)
        if (np.max(self.level) != levels-1 or
                np.any(self.parent>=np.arange(self.nn))):
            raise ValueError('AMR hierarchy level/parent ordering differs')
        self.minimum = _host(minimum_levels,(self.nn,),
                             'minimum levels',integer=True).astype(np.int64)
        self.rock = _host(tree_rock,(self.nn,),'tree rock')
        if (np.any(self.minimum<0) or np.any(self.minimum>=levels) or
                np.any(self.minimum>self.maximum)):
            raise ValueError('AMR minimum levels exceed hierarchy limits')
        children = [[] for _ in range(self.nn)]
        for node,parent in enumerate(self.parent):
            if parent>=0:
                children[parent].append(node)
        self.children = tuple(tuple(group) for group in children)
        for node,group in enumerate(self.children):
            if group and (len(group)!=4 or
                    any(self.level[child] != self.level[node]+1 for child in group)):
                raise ValueError('AMR hierarchy is not complete 2:1 refinement')
        finest = np.flatnonzero(self.level==levels-1)
        self.finest = frozenset(map(int,finest))
        pairs = np.asarray(finest_face_pairs)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or
                pairs.dtype.kind not in 'iu' or
                np.any(pairs<0) or np.any(pairs>=self.nn) or
                not np.isin(pairs,finest).all() or
                np.any(pairs[:,0]==pairs[:,1])):
            raise ValueError('Immutable finest-face connectivity differs')
        pairs = np.sort(pairs.astype(np.int64),axis=1)
        if len(np.unique(pairs,axis=0)) != len(pairs):
            raise ValueError('Duplicate immutable finest-face pair')
        self.finest_pairs = pairs
        explicit = frozenset(map(int,protected_finest_ids))
        if not explicit.issubset(self.finest):
            raise ValueError('Protected AMR nodes must be finest hierarchy leaves')
        self.explicit_protected = explicit
        self.by_digest = {}
        self.live_sections=False
        bedform_sections=[]
        for topology in prepared.topologies:
            if not np.array_equal(self.rock[topology.node_ids],topology.data['rock']):
                raise ValueError('AMR tree rock differs from prepared operator')
            digest = leaf_digest(topology.node_ids)
            if digest in self.by_digest:
                raise ValueError('Duplicate AMR catalog leaf set')
            self.by_digest[digest] = topology
            self.live_sections |= bool(self.dynamic_layout and
                np.array_equal(topology.data['mode'],[9,4]))
            bedform_sections.append(bool(self.dynamic_layout and
                recursive_bedform_envelope(topology.data)))
        if any(bedform_sections) and not all(bedform_sections):
            raise ValueError('AMR catalog mixes recursive bedform boundary envelopes')
        self.recursive_bedform_sections = bool(bedform_sections and
                                               all(bedform_sections))
        contract = dict(settings=self.settings,
            minimum_sha256=hashlib.sha256(self.minimum.astype('<i8').tobytes()).hexdigest(),
            rock_sha256=hashlib.sha256(self.rock.astype('<f8').tobytes()).hexdigest(),
            finest_pairs_sha256=hashlib.sha256(self.finest_pairs.astype('<i8').tobytes()).hexdigest(),
            protected_ids=sorted(self.explicit_protected),
            coarsen_stage_span_m=self.stage_span)
        # Preserve the legacy default identity; the new support lifecycle is
        # explicit and cannot silently reuse a static-bank selector restart.
        if self.evolving_banks:
            contract.update(evolving_banks=True,
                event_rule='complete_one_level_material_homogeneous_v1')
            if self.live_sections:
                contract['boundary_forecast']='all_open_finest_current_inlet_characteristic_v1'
        if self.general_transport:
            contract.update(general_transport=True,
                event_rule='complete_one_level_transport_v1')
            if self.live_sections:
                contract['boundary_forecast']='all_open_finest_current_inlet_characteristic_v1'
            if self.recursive_bedform_sections:
                contract['recursive_bedform_boundary'] = \
                    'unscheduled_unsampled_mode1_mode2_finest_v1'
        if self.extended_indicators:
            contract['extended_indicator_rule']=EXTENDED_INDICATOR_RULE
        self.contract_identity = hashlib.sha256(json.dumps(contract,
            sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def _bed(self, rows, ids, data):
        indices = np.asarray(data['bed_indices'],dtype=np.int64)
        bulk = np.asarray(data['bulk_fraction'],dtype=np.float64)
        return self.rock[ids]+np.sum(rows[:,indices]/bulk[None,None,:],axis=(1,2))

    def _sync_tree(self, selected_ids, accepted, cached):
        rows = cached.copy()
        rows[selected_ids] = accepted
        cover = np.full(self.nn,-1,dtype=np.int64)
        selected = set(map(int,selected_ids))
        for node in range(self.nn):
            parent = self.parent[node]
            cover[node] = node if node in selected else (
                cover[parent] if parent>=0 else -1)
        for node in range(self.nn-1,-1,-1):
            group = self.children[node]
            if cover[node]<0 and group:
                weights = self.area[list(group)]
                rows[node] = np.dot(weights,rows[list(group)])/np.sum(weights)
        return rows

    def _cover(self, selected):
        cover = np.full(self.nn,-1,dtype=np.int64)
        selected_set = set(selected)
        for node in range(self.nn):
            parent = self.parent[node]
            cover[node] = node if node in selected_set else (
                cover[parent] if parent>=0 else -1)
        if np.any(cover[list(self.finest)]<0):
            raise ValueError('AMR selected leaves do not cover the finest domain')
        return cover

    def _selected(self, wanted, current=None):
        selected = []
        current_set = None if current is None else set(map(int,current))
        allowed = None
        cover_current = None
        if current_set is not None:
            cover_current = self._cover(current_set)
            allowed = current_set | {node for node,group in enumerate(self.children)
                if len(group)==4 and all(child in current_set for child in group)}

        def visit(node):
            group = self.children[node]
            refine = self.level[node]<wanted[node]
            if cover_current is not None and cover_current[node]>=0:
                refine = refine and self.level[node]<self.level[cover_current[node]]+1
            must_descend = allowed is not None and node not in allowed and cover_current[node]<0
            if group and (refine or must_descend):
                for child in group:
                    visit(child)
            else:
                selected.append(node)

        for node in np.flatnonzero(self.parent<0):
            visit(int(node))
        for _ in range(self.nn):
            cover = self._cover(selected)
            refine = set()
            for fa,fb in self.finest_pairs:
                a,b = int(cover[fa]),int(cover[fb])
                if a==b or abs(self.level[a]-self.level[b])<=1:
                    continue
                coarse = a if self.level[a]<self.level[b] else b
                if self.level[coarse]>=self.maximum[coarse] or not self.children[coarse]:
                    raise ValueError('AMR 2:1 balance exceeds hierarchy safety cap')
                if (cover_current is not None and cover_current[coarse]>=0 and
                    self.level[coarse]>=self.level[cover_current[coarse]]+1):
                    raise ValueError('AMR balance would require a multi-level event')
                refine.add(coarse)
            if not refine:
                return tuple(sorted(selected))
            selected = [node for node in selected if node not in refine]
            for node in sorted(refine):
                selected.extend(self.children[node])
        raise RuntimeError('AMR 2:1 balance did not converge')

    def _protected(self, topology):
        ids = topology.node_ids
        data = topology.data
        boundary = np.asarray(data['boundary_faces'],dtype=np.int64)
        protected = set(self.explicit_protected)
        protected.update(map(int,ids[np.asarray(data['owner'])[boundary]]))
        if not self.dynamic_layout:
            active = np.flatnonzero(np.asarray(data['is_toe_face']))
            for key in ('candidate_bank','candidate_toe'):
                protected.update(map(int,ids[np.asarray(data[key])[active]]))
            protected.update(map(int,ids[np.flatnonzero(data['bank_material_mask'])]))
        if not protected.issubset(self.finest):
            raise ValueError('AMR bank/toe/outlet support is not finest level')
        return protected

    def _inlet_owners(self,topology):
        """Local prescribed-inlet owners; topology metadata only, no state copy."""
        if not (self.dynamic_layout and np.array_equal(topology.data['mode'],[9,4])):
            return np.empty(0,np.int64)
        data=topology.data
        faces=np.asarray(data['boundary_faces'])
        selected=faces[np.asarray(data['bid'])[faces]==0]
        return np.unique(np.asarray(data['owner'])[selected]).astype(np.int64)

    def propose(self, topology: AMRTopology, accepted_state, *,
                cached_tree_state, delay_state, time_s: float,
                initial=False, selector_last_bed=None, bed_gradient=None) -> GeneralLeafProposal:
        if not any(top is topology for top in self.prepared.topologies):
            raise ValueError('Current topology is outside authenticated catalog')
        data,ids = topology.data,np.asarray(topology.node_ids,dtype=np.int64)
        modes = np.asarray(data['mode'])
        live=self.dynamic_layout and np.array_equal(modes,[9,4])
        bedform = self.dynamic_layout and recursive_bedform_envelope(data)
        closed_m2c = (self.dynamic_layout and modes.shape == (0,) and
                      'suspended_amr_contract' in data)
        if closed_m2c:
            from .implicit_suspension_amr import contract_from_data
            contract_from_data(data)
        if (int(data['bank_front_tracking'])!=0 or
                not (np.array_equal(modes,[2]) or live or bedform or
                     closed_m2c)):
            raise ValueError('Unported persistent front or boundary forecast')
        if live:
            from .amr_catalog_bank import outlet_owners
            outlet_owners(self.prepared,topology)
        n,nstate = len(ids),int(data['nstate'])
        if hasattr(accepted_state,'__cuda_array_interface__') or hasattr(
                cached_tree_state,'__cuda_array_interface__'):
            if self._resident_indicators is None:
                from braids_support.gpu_fwave.amr_general_indicators import ResidentGeneralIndicators
                self._resident_indicators = ResidentGeneralIndicators(self,
                    source_path=self.indicator_source_path)
            extra=dict(selector_last_bed=selector_last_bed,bed_gradient=bed_gradient) if self.extended_indicators else {}
            result = self._resident_indicators.propose(topology,accepted_state,
                cached_tree_state,delay_state,time_s=time_s,initial=initial,**extra)
            return self._finish(topology,ids,time_s,**result,device=True)
        accepted = _host(accepted_state,(n,nstate),'accepted selected rows')
        cached = _host(cached_tree_state,(self.nn,nstate),'accepted tree cache')
        for enabled,value,shape,name in (
            (self.settings['refine_sediment'],selector_last_bed,(self.nn,),'last bed reference'),
            (self.settings['refine_bathymetry'],bed_gradient,(n,2),'bed gradient')):
            if enabled and (not isinstance(value,np.ndarray) or value.shape!=shape or
                    value.dtype!=np.float64 or not np.isfinite(value).all()):
                raise ValueError('Invalid AMR selector '+name+'; finite FP64 NumPy array required')
        delay = _host(delay_state,(self.nn,),'delay counters',integer=True).astype(np.int64)
        if (np.any(delay<0) or not np.isfinite(time_s) or time_s<0 or
                abs(time_s/self.interval-round(time_s/self.interval))>1e-9):
            raise ValueError('AMR accepted clock/delay differs')
        sediment = np.r_[np.asarray(data['suspended_indices']).ravel(),
                         np.asarray(data['bed_indices']).ravel()]
        if np.any(accepted[:,0]<-1e-12) or np.any(accepted[:,sediment]<-1e-12):
            raise ValueError('Invalid AMR accepted water/sediment')
        tree_rows = self._sync_tree(ids,accepted,cached)
        bed = self._bed(accepted,ids,data)
        depth = accepted[:,0]
        eta = bed+depth
        wet = depth>self.wet_depth
        velocity = np.zeros((n,2),dtype=np.float64)
        moving = depth>float(data['dry_tolerance'])
        velocity[moving] = accepted[moving,1:3]/depth[moving,None]
        if not np.isfinite(bed).all() or not np.isfinite(eta).all() or not np.isfinite(velocity).all():
            raise ValueError('Nonfinite AMR hydraulic indicator')
        raw = self.minimum.copy()
        raw[ids[wet]] = np.maximum(raw[ids[wet]],self.settings['wet_level'])
        cause = np.zeros(n,dtype=np.int32)
        cause[wet] |= 1<<1
        wet_front = np.zeros(n,dtype=bool)
        moving_source = np.zeros(n,dtype=bool)
        adjacent = [[] for _ in range(n)]
        owner = np.asarray(data['owner'],dtype=np.int64)
        neighbour = np.asarray(data['neighbour'],dtype=np.int64)
        for a,b in zip(owner,neighbour):
            if b<0:
                continue
            adjacent[a].append(int(b));adjacent[b].append(int(a))
            if wet[a]!=wet[b]:
                raw[ids[[a,b]]] = self.levels-1
                cause[[a,b]] |= 1<<2
                wet_front[[a,b]] |= wet[[a,b]]
                moving_source[[a,b]] = True
            if wet[a] and wet[b] and (abs(eta[a]-eta[b])>self.eta_tolerance or
                    np.linalg.norm(velocity[a]-velocity[b])>self.velocity_tolerance):
                raw[ids[[a,b]]] = self.levels-1
                cause[[a,b]] |= 1<<3
                moving_source[[a,b]] = True
            bank_material=data.get('bank_material_mask')
            if (wet[a] or wet[b]) and (
                    data['manning_cell'][a]!=data['manning_cell'][b] or
                    (bank_material is not None and
                     bool(bank_material[a])!=bool(bank_material[b]))):
                raw[ids[[a,b]]] = np.maximum(raw[ids[[a,b]]],min(2,self.levels-1))
                cause[[a,b]] |= 1<<7
        for face in np.flatnonzero(neighbour<0):
            # Native bid>0 is exported as bid-1: the first open boundary is 0,
            # while solid walls are -1 in the authenticated GPU operator.
            if data['bid'][face]>=0:
                a = owner[face]
                raw[ids[a]]=self.levels-1;cause[a] |= 1<<8
        if self.extended_indicators:
            from .amr_indicator_reference import extended_tags
            extended_tags(self,data,ids,accepted,tree_rows,bed,wet,raw,cause,moving_source,
                          selector_last_bed=selector_last_bed,bed_gradient=bed_gradient)
        for i,node in enumerate(ids):
            group = self.children[int(node)]
            if not wet[i] or not group:
                continue
            child = np.asarray(group,dtype=np.int64)
            child_bed = self._bed(tree_rows[child],child,data)
            if np.any(child_bed>=eta[i]-self.wet_depth):
                raw[node]=self.levels-1
                cause[i] |= 1<<12
                moving_source[i]=True
        protected = self._protected(topology)
        raw[list(protected)] = self.levels-1
        # Current accepted inlet characteristics are moving forecast sources
        # even in an otherwise uniform reach. This is not a bound on a future
        # discharge jump; the runner separately clips at forcing-table knots.
        moving_source[self._inlet_owners(topology)]=True

        reach = np.full(n,-np.inf,dtype=np.float64)
        buffer = self.settings['buffer_cells']*max(float(data['dx']),float(data['dy']))
        heap = []
        for i,node in enumerate(ids):
            if raw[node]!=self.levels-1:
                continue
            speed = 0.
            if moving_source[i] and wet[i]:
                speed = (np.linalg.norm(velocity[i])+
                         (2 if wet_front[i] else 1)*
                         np.sqrt(float(data['grav'])*depth[i]))
            reach[i] = buffer+speed*self.interval
            heapq.heappush(heap,(-reach[i],i))
        while heap:
            neg,i = heapq.heappop(heap)
            if -neg<reach[i]-1e-12:
                continue
            for j in adjacent[i]:
                distance = np.linalg.norm(self.xy[ids[i]]-self.xy[ids[j]])
                value = reach[i]-distance
                if value>=0 and value>reach[j]+1e-12:
                    reach[j]=value
                    heapq.heappush(heap,(-value,j))
        buffered = reach>=0
        raw[ids[buffered]]=self.levels-1
        cause[buffered] |= 1<<14

        local = {int(node):i for i,node in enumerate(ids)}
        vetoed = []
        dry = float(data['dry_tolerance'])
        for parent,group in enumerate(self.children):
            if len(group)!=4 or not all(child in local for child in group):
                continue
            ix = np.asarray([local[child] for child in group])
            solver_wet = depth[ix]>dry
            mixed = bool(np.any(solver_wet) and np.any(~solver_wet))
            span = float(np.ptp(eta[ix][solver_wet])) if np.any(solver_wet) else 0.
            material = bool(self.evolving_banks and np.any(
                np.asarray(data['bank_material_mask'])[ix]!=data['bank_material_mask'][ix[0]]))
            if mixed or span>self.stage_span or material:
                raw[list(group)] = self.levels-1 if mixed or span>self.stage_span else np.maximum(
                    raw[list(group)],self.level[list(group)])
                entry=dict(parent_id=parent,mixed_solver_wet_dry=mixed,
                           wet_child_stage_span_m=span)
                if self.evolving_banks:entry['mixed_material']=material
                vetoed.append(entry)
        if not initial:
            for node in ids:
                if raw[node]>=self.level[node]:
                    delay[node]=0
                else:
                    if delay[node]==np.iinfo(np.int64).max:
                        raise ValueError('AMR coarsen delay overflow')
                    delay[node]+=1
                    if delay[node]<self.settings['coarsen_delay']:
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
        stats=dict(wet_dry_trigger_cells=int(np.count_nonzero(cause&(1<<2))),
            hydraulic_trigger_cells=int(np.count_nonzero(cause&(1<<3))),
            wet_child_bed_trigger_cells=int(np.count_nonzero(cause&(1<<12))),
            open_boundary_trigger_cells=int(np.count_nonzero(cause&(1<<8))),
            forecast_buffer_cells=int(np.count_nonzero(buffered)))
        if self.extended_indicators:
            stats.update({name:int(np.count_nonzero(cause&(1<<bit)))
                          for name,bit in INDICATOR_STATS.items()})
        return self._finish(topology,ids,time_s,wanted=wanted,delay=delay,
            protected=protected,vetoed=vetoed,stats=stats,device=False)

    def _finish(self,topology,ids,time_s,*,wanted,delay,protected,vetoed,stats,device):
        selected = self._selected(wanted,current=ids if self.dynamic_layout else None)
        if not protected.issubset(selected):
            raise ValueError('AMR candidate lost finest bank/toe/outlet support')
        digest = leaf_digest(selected)
        match = self.by_digest.get(digest)
        receipt = dict(catalog_identity=self.prepared.identity,
            selector_contract_identity=self.contract_identity,
            old_operator_identity=topology.source_identity,
            proposed_operator_identity=(None if match is None else match.source_identity),
            leaf_digest=digest,old_leaves=len(ids),new_leaves=len(selected),
            catalog_hit=match is not None,host_selector=not device,
            native_mesh_schedule_replayed=False,time_s=float(time_s),
            **stats,
            protected_finest_nodes=len(protected),
            unsafe_coarsen_veto=vetoed)
        if self.evolving_banks:
            receipt.update(numerical_indicator_backend='cuda' if device else 'numpy',
                           evolving_banks=True,complete_one_level_events=True)
        if self.general_transport:
            receipt.update(numerical_indicator_backend='cuda' if device else 'numpy',
                           general_transport=True,complete_one_level_events=True)
        return GeneralLeafProposal(selected,match,delay,receipt)
