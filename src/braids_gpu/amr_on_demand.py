"""State-selected bank AMR with initialization-only, on-demand geometry.

The immutable operator store contains no evolving solution. New meshes use
the original BRAIDS initializer; accepted solution/history transfer and all
coupled timesteps use the existing CUDA route. This synchronous route does
not yet support subcycling, persistent bank fronts or suspended exchange.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from braids_core.windows_paths import filesystem_path

from .amr_native_geometry import sha, hierarchy_from_sidecar
from .amr_prepared import (load_amr_prepared, _read_coupled,
    _bank_same_global_controls, _EXACT_BEDFORM_BANK_RUNTIMES)
from .amr_operator_store import AMROperatorStore
from .amr_general_selector import GeneralAMRStateSelector,leaf_digest
from .amr_catalog_bank import CatalogBankRoute, validate_lineages, outlet_owners, require
from .atomic_output import write_json_atomic


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def runtime_api(runtime):
    """Dispatch an authenticated exporter version, never reinterpret an old one."""
    build=json.loads((Path(runtime)/'build.json').read_text())
    if build.get('kind') == 'selected-general-transport-amr-v1':
        from . import amr_general_native_geometry as api
    elif build.get('kind') == 'live-selected-bank-mode3-bedform-avalanche-amr-v3':
        from . import amr_bank_bedform_native_geometry as api
    elif build.get('kind') == 'live-selected-bank-mode2-bedform-avalanche-amr-v1':
        from . import amr_bank_physical_toe_bedform_native_geometry as api
    elif build.get('kind') == 'live-selected-bank-mode23-amr':
        from . import amr_bank_native_geometry as api
    elif build.get('live_boundary_metadata') is True:
        from . import amr_live_native_geometry as api
    else:
        from . import amr_native_geometry as api
    api.verify_runtime(runtime)
    return api


def verify_runtime(runtime):
    return runtime_api(runtime).verify_runtime(runtime)


def export_selected(deck,folder,ids,*,runtime,**kwargs):
    return runtime_api(runtime).export_selected(deck,folder,ids,runtime=runtime,**kwargs)


def input_files(package):
    manifest=json.loads((package.folder/'amr_prepared.json').read_text())
    receipt=json.loads((package.folder/manifest['topologies'][0]['receipt']).read_text())
    return {key:value for key,value in receipt['origin']['files'].items()
            if key!='selected_nodes.txt'}


def _compact_private_layout(folder):
    """Keep frozen native bootstrap paths below legacy Windows limits."""
    folder=filesystem_path(folder)
    projected=folder/'initializations'/'request-xxxxxxxx'/'prepared_native'
    return os.name=='nt' and len(str(projected))>=230


def selector_inputs(tree):
    flags=np.asarray(tree['flags'])
    require(flags.shape==(5,), 'AMR native indicator flag shape differs')
    settings={name:bool(flags[i]) for i,name in enumerate((
        'refine_wet_dry','refine_walls','refine_bathymetry',
        'refine_hydraulics','refine_sediment'))}
    for target,key in (('levels','amr_levels'),('refinement_ratio','amr_ratio'),
                       ('coarsen_delay','amr_delay'),('wet_level','amr_wet_level'),
                       ('buffer_cells','amr_buffer')):
        settings[target]=int(tree[key])
    for target,key in (('regrid_interval_s','amr_interval'),
                       ('wet_depth_m','amr_wet_depth'),
                       ('surface_tolerance_m','amr_eta_tol'),
                       ('velocity_tolerance_m_s','amr_u_tol')):
        settings[target]=float(tree[key])
    settings['subcycling']=bool(tree['subcycling'])
    # Keep old flags-off preparation identities readable. New criteria bind
    # every numerical threshold explicitly, rather than inheriting UI defaults.
    if any(settings[key] for key in ('refine_walls','refine_bathymetry','refine_sediment')):
        for target,key in (('bed_tolerance_m','amr_bed_tol'),
                           ('concentration_tolerance_kg_m3','amr_c_tol'),
                           ('composition_tolerance','amr_mix_tol'),
                           ('bed_change_tolerance_m','amr_dz_tol')):
            settings[target]=float(tree[key])
    owner,other=tree['face_owner'],tree['face_neighbour']
    finest=np.max(tree['node_level'])
    active=(other>=0)&(tree['node_level'][owner]==finest)&(
        tree['node_level'][np.maximum(other,0)]==finest)
    pairs=np.unique(np.sort(np.column_stack((owner[active],other[active])),axis=1),axis=0)
    return dict(settings=settings,minimum_levels=tree['minimum_level'],
                tree_rock=tree['rock'],finest_face_pairs=pairs)


def cold_indicator_state(selector,topology,tree_state):
    """Host-only original-state data for preparation, never a CUDA fallback."""
    extra={}
    if selector.settings['refine_sediment']:
        extra['selector_last_bed']=selector._bed(tree_state,np.arange(selector.nn),topology.data)
    if selector.settings['refine_bathymetry']:
        from .amr_indicator_reference import cold_bed_gradient
        extra['bed_gradient']=cold_bed_gradient(topology.data,topology.data['state'].T)
    return extra


class GeometrySession:
    """Durable original inputs and append-only operator cache for one run family."""

    @classmethod
    def create(cls,folder,deck,*,runtime,cancelled=lambda:False,progress=None,
               evolving_banks=False):
        """Build a cold session privately, then publish the complete tree once.

        ``progress(fraction, message)`` is optional.  The destination never
        exists until bootstrap, original-input receipt and initial operator
        have all passed their normal host authentication.
        """
        folder,deck,runtime=(filesystem_path(p).resolve() for p in (folder,deck,runtime))
        require(not folder.exists(),'On-demand AMR session is write-once')
        require(progress is None or callable(progress),
                'On-demand AMR progress callback must be callable')
        require(type(evolving_banks) is bool,'Evolving-bank selection must be explicit boolean')

        def update(fraction,message):
            if cancelled():
                raise InterruptedError('On-demand AMR session creation cancelled')
            if progress is not None:
                progress(fraction,message)
            if cancelled():
                raise InterruptedError('On-demand AMR session creation cancelled')

        update(0.,'Checking original AMR inputs')
        hierarchy=hierarchy_from_sidecar(deck.parent/'amr_hierarchy.npz')
        finest=np.flatnonzero(hierarchy.node_level==np.max(hierarchy.node_level))
        folder.parent.mkdir(parents=True,exist_ok=True)
        # The temporary path must leave room for content-addressed operator
        # directories below it on Windows before the shorter final rename.
        compact=_compact_private_layout(folder)
        staging_prefix='.a-' if compact else '.amr-'
        staging=filesystem_path(tempfile.mkdtemp(prefix=staging_prefix,dir=folder.parent))
        try:
            update(.10,'Exporting initial zero-step geometry')
            package,lineage=export_selected(deck,staging/'bootstrap',finest,
                runtime=runtime,cancelled=cancelled)
            update(.40,'Selecting initial accepted leaf set')
            tree_path=staging/'bootstrap_native/amr_tree_input.bin'
            tree=_read_coupled(tree_path)
            bank_mode=int(package.topologies[0].data['bank_toe_erosion_mode'])
            require(bank_mode in (2,3) and (bank_mode==2 or evolving_banks),
                    'Distributed bank AMR requires the explicit dynamic-bank workflow')
            arguments=selector_inputs(tree)
            require(evolving_banks or not any(arguments['settings'][key] for key in
                ('refine_walls','refine_bathymetry','refine_sediment')),
                'Extended AMR indicators require the explicit evolving-bank route')
            if evolving_banks:
                arguments['evolving_banks']=True
            selector=GeneralAMRStateSelector(package,**arguments)
            delay=np.zeros(len(tree['parent']),dtype=np.int64)
            proposal=selector.propose(package.topologies[0],package.topologies[0].data['state'].T,
                cached_tree_state=tree['state'].T,delay_state=delay,time_s=0.,initial=True,
                **cold_indicator_state(selector,package.topologies[0],tree['state'].T))
            source=dict(schema=1,kind='on-demand-bank-amr',deck=str(deck),runtime=str(runtime),
                original_input_files=input_files(package),bootstrap_identity=package.identity,
                tree_input_sha256=sha(tree_path),global_toe_lineage=[
                    {key:value for key,value in row.items() if key!='local_face'} for row in lineage],
                initial_node_ids=list(proposal.node_ids),selector_contract=selector.contract_identity,
                initial_proposal=proposal.receipt,original_controls=selector.settings,
                native_solver_steps=0)
            runtime_kind=json.loads((runtime/'build.json').read_text()).get('kind')
            exact_bedform_bank=_EXACT_BEDFORM_BANK_RUNTIMES.get(runtime_kind)
            bedform_bank=exact_bedform_bank is not None
            if evolving_banks:
                source.update(schema=2,evolving_banks=True)
                # Every selected operator authenticates its own full-face
                # source IDs. Original toe membership is not a lifetime rule.
                del source['global_toe_lineage']
            if bank_mode==3 or bedform_bank:
                source['bank_mode']=bank_mode
            if bedform_bank:
                require(evolving_banks and
                        bank_mode==exact_bedform_bank['bank_mode'],
                        'Bedform-bank initializer mode differs from selected geometry')
                source['recursive_bank_bedform_avalanche']=True
            source['identity']=digest(source)
            write_json_atomic(staging/'session.json',source)
            update(.65,'Authenticating session and first operator')
            session=cls(staging,cancelled=cancelled)
            session.get(proposal.node_ids)
            update(.90,'Publishing complete AMR session')
            for attempt in range(12):
                if cancelled():
                    raise InterruptedError('On-demand AMR session creation cancelled')
                require(not folder.exists(),'On-demand AMR session is write-once')
                try:
                    os.rename(staging,folder)
                    break
                except OSError as error:
                    require(not folder.exists(),'On-demand AMR session is write-once')
                    if (os.name!='nt' or getattr(error,'winerror',None) not in (5,32,33)
                            or attempt==11):
                        raise
                    time.sleep(min(.05*(attempt+1),.5))
            ready=cls(folder,cancelled=cancelled)
            ready.requests=session.requests
            # A progress callback failure after atomic publication must not
            # turn a complete, valid session into a reported failed creation.
            if progress is not None:
                try:
                    progress(1.,'AMR session ready')
                except Exception:
                    pass
            return ready
        finally:
            if staging.exists():
                require(staging.resolve().is_relative_to(folder.parent.resolve()) and
                        staging.name.startswith(staging_prefix) and
                        not staging.is_symlink(),
                        'Refusing to clean unexpected AMR session staging directory')
                try:
                    shutil.rmtree(staging)
                except OSError:
                    # Preserve the original preparation/cancellation failure.
                    # A remaining private stage is not a published session.
                    pass

    def __init__(self,folder,*,cancelled=lambda:False):
        self.folder=filesystem_path(folder).resolve()
        self.manifest=json.loads((self.folder/'session.json').read_text())
        m=self.manifest
        require(m['identity']==digest({k:v for k,v in m.items() if k!='identity'}) and
                m['schema'] in (1,2) and m['kind']=='on-demand-bank-amr' and m['native_solver_steps']==0,
                'On-demand session identity differs')
        self.evolving_banks=m['schema']==2
        require((self.evolving_banks and m.get('evolving_banks') is True and
                 'global_toe_lineage' not in m) or
                (not self.evolving_banks and 'evolving_banks' not in m),
                'On-demand bank mapping contract differs')
        runtime_kind=json.loads(
            (filesystem_path(m['runtime'])/'build.json').read_text()).get('kind')
        exact_bedform_bank=_EXACT_BEDFORM_BANK_RUNTIMES.get(runtime_kind)
        self.bedform_bank_composition=exact_bedform_bank is not None
        self.bedform_bank_mode=(None if exact_bedform_bank is None else
                                exact_bedform_bank['bank_mode'])
        require((self.bedform_bank_composition and
                 m.get('recursive_bank_bedform_avalanche') is True) or
                (not self.bedform_bank_composition and
                 'recursive_bank_bedform_avalanche' not in m),
                'On-demand bedform-bank initializer contract differs')
        self.identity=m['identity'];self.cancelled=cancelled
        self.deck,self.runtime=filesystem_path(m['deck']),filesystem_path(m['runtime'])
        self.verify_originals()
        self.bootstrap=load_amr_prepared(self.folder/'bootstrap')
        self.bank_mode=int(self.bootstrap.topologies[0].data['bank_toe_erosion_mode'])
        require((self.bedform_bank_composition and self.evolving_banks and
                 type(m.get('bank_mode')) is int and
                 m['bank_mode']==self.bank_mode==self.bedform_bank_mode) or
                (not self.bedform_bank_composition and self.bank_mode==2 and
                 'bank_mode' not in m) or
                (not self.bedform_bank_composition and self.bank_mode==3 and
                 self.evolving_banks and type(m.get('bank_mode')) is int and
                 m['bank_mode']==3),
                'On-demand bank mode differs from authenticated geometry')
        require(self.evolving_banks or not any('toe_geometry_seed_only' in top.data
                for top in self.bootstrap.topologies),
                'Geometry-only toe seeds cannot initialize a static bank map')
        tree_path=self.folder/'bootstrap_native/amr_tree_input.bin'
        require(self.bootstrap.identity==m['bootstrap_identity'] and
                sha(tree_path)==m['tree_input_sha256'] and
                input_files(self.bootstrap)==m['original_input_files'],
                'On-demand original tree/geometry changed')
        self.tree=_read_coupled(tree_path)
        self.initial_tree_state=np.ascontiguousarray(self.tree['state'].T)
        self.selector_arguments=selector_inputs(self.tree)
        require(self.evolving_banks or not any(self.selector_arguments['settings'][key]
                for key in ('refine_walls','refine_bathymetry','refine_sediment')),
                'Extended AMR indicators require the explicit evolving-bank route')
        self.checkpoint_schema=(4 if self.bank_mode==3 else
                                3 if self.selector_arguments['settings']['refine_sediment']
                                else 2 if self.evolving_banks else 1)
        if self.evolving_banks:
            self.selector_arguments['evolving_banks']=True
        selector=GeneralAMRStateSelector(self.bootstrap,**self.selector_arguments)
        require(selector.contract_identity==m['selector_contract'],
                'On-demand selector controls changed')
        store_arguments=(dict(evolving_toe_variant=True,tree_input_sha256=m['tree_input_sha256'])
                         if self.evolving_banks else dict(global_toe_lineage=m['global_toe_lineage']))
        self.store=AMROperatorStore(self.folder/'store',
            hierarchy_sha256=self.bootstrap.source['hierarchy_sha256'],
            base_input_sha256=self.bootstrap.source['base_input_sha256'],
            runtime_build_sha256=self.bootstrap.source['runtime_build_sha256'],
            **store_arguments)
        self.requests=[]

    def verify_originals(self):
        if (Path(__file__).resolve().parents[1]/'PACKAGE-MANIFEST.json').exists():
            from .amr_portable_runtime import verify_packaged_runtime
            packaged=verify_packaged_runtime()
            require(self.runtime.resolve() in {packaged['runtime'],
                    packaged.get('live_runtime'),packaged.get('bedform_bank_runtime'),
                    packaged.get('bedform_physical_toe_runtime')},
                    'Prepared AMR initializer is outside the selected worker package')
        verify_runtime(self.runtime)
        for name,value in self.manifest['original_input_files'].items():
            actual='geometry.npz' if name=='original_geometry.npz' else name
            require(Path(actual).name==actual and sha(self.deck.parent/actual)==value,
                    'Original AMR input changed: '+name)

    def initial_selection(self):
        """Recover the cold proposal, not a second adaptation of its result.

        Preparation already selected the first mesh from original finest
        geometry. Repeating selection on that selected mesh changes one-level
        coarsening and is not an idempotent initialization operation.
        """
        selector=GeneralAMRStateSelector(self.bootstrap,**self.selector_arguments)
        proposal=selector.propose(self.bootstrap.topologies[0],
            self.bootstrap.topologies[0].data['state'].T,
            cached_tree_state=self.initial_tree_state,
            delay_state=np.zeros(len(self.tree['parent']),dtype=np.int64),time_s=0.,initial=True,
            **cold_indicator_state(selector,self.bootstrap.topologies[0],self.initial_tree_state))
        require(list(proposal.node_ids)==self.manifest['initial_node_ids'] and
                proposal.receipt==self.manifest['initial_proposal'],
                'Original cold AMR proposal differs from prepared session')
        return proposal

    def _produce(self,folder,ids):
        self.verify_originals()
        compact=_compact_private_layout(self.folder)
        jobs=self.folder/('i' if compact else 'initializations')
        jobs.mkdir(exist_ok=True)
        job=filesystem_path(tempfile.mkdtemp(prefix='r-' if compact else 'request-',dir=jobs))
        package_name='p' if compact else 'prepared'
        package,lineage=export_selected(self.deck,job/package_name,ids,
            runtime=self.runtime,cancelled=self.cancelled)
        require(input_files(package)==self.manifest['original_input_files'] and
                sha(job/(package_name+'_native')/'amr_tree_input.bin')==
                    self.manifest['tree_input_sha256'],
                'Selected initializer changed original material/state controls')
        self.verify_originals()
        shutil.copytree(package.folder,folder)
        return load_amr_prepared(folder),(job/(package_name+'_native')
                                         if self.evolving_banks else lineage)

    def get(self,ids):
        if self.cancelled():
            raise InterruptedError('On-demand mesh request cancelled')
        self.verify_originals()
        started=time.perf_counter()
        hit=self.store.get(ids)
        result=hit if hit is not None else self.store.get_or_create(ids,self._produce,
                                                                   cancelled=self.cancelled)
        require(input_files(result.package)==self.manifest['original_input_files'],
                'Cached AMR operator has different original input dependencies')
        self.requests.append(dict(node_count=len(ids),request_key=result.request_key,
            cache_hit=hit is not None,elapsed_s=time.perf_counter()-started,
            package_identity=result.package_identity,native_solver_steps=0))
        return result

    def view(self,entries):
        require(bool(entries),'Empty on-demand selected-operator family')
        first=entries[0].package
        tops=[];lineages={}
        for entry in entries:
            p=entry.package;top=p.topologies[0]
            require(input_files(p)==self.manifest['original_input_files'] and
                    set(p.tree)==set(first.tree) and all(np.array_equal(p.tree[k],first.tree[k])
                    for k in first.tree) and _bank_same_global_controls(first.topologies[0].data,top.data),
                    'New AMR operator changes hierarchy or physics')
            for key in ('bulk_fraction','secondary_flow','mode','bed_indices','suspended_indices',
                        'forcing','fraction_concentration','fraction_bed_input','section_energy_slope',
                        'reversal_policy','bed_boundary_mode','section_control','nseries','nhydro_series',
                        'nfraction_series','bc_hydraulic','bc_bed_input'):
                require(np.array_equal(first.topologies[0].data[key],top.data[key]),
                        'New AMR operator changes global control '+key)
            # Section seeds/local face IDs vary with operator numbering; real
            # forcing tables and section laws must not vary with the mesh.
            base=first.topologies[0].data
            table_keys={key for key in base if key.startswith(('series_',
                'hydro_series_','fraction_series_'))}
            require(table_keys=={key for key in top.data if key.startswith(('series_',
                'hydro_series_','fraction_series_'))} and
                all(np.array_equal(base[key],top.data[key]) for key in table_keys),
                'New AMR operator changes live forcing tables')
            require(top.name not in lineages,'Duplicate on-demand operator')
            tops.append(top)
            lineages[top.name]=([] if self.evolving_banks else [dict(candidate_source_face=row['segment_id'],
                local_face=row['local_face'],bank_node=row['bank_node'],toe_node=row['toe_node'])
                for row in entry.lineage])
        # Identity binds immutable inputs and policy; each selected operator has
        # its own authenticated content identity, retained in events/checkpoints.
        prepared=replace(first,folder=self.folder,identity=self.identity,topologies=tuple(tops),
                         source=dict(session_identity=self.identity))
        if not self.evolving_banks:
            validate_lineages(prepared,lineages)
        return prepared,lineages


class OnDemandBankRoute(CatalogBankRoute):
    """Existing resident coupled route extended with a lazy operator provider."""

    def __init__(self,session,*,source_folder):
        self.session=session
        self.entries=[session.get(session.manifest['initial_node_ids'])]
        prepared,lineages=session.view(self.entries)
        super().__init__(prepared,source_folder=source_folder,
            initial_tree_state=session.initial_tree_state,lineages=lineages,
            evolving_toe_variant=session.evolving_banks and session.bank_mode==2)
        self.selector=self._selector(prepared)
        self.selector_last_bed=None
        if self.selector.settings['refine_sediment']:
            # Original whole-tree references persist for dormant nodes. Publish
            # actual selected-leaf beds without repeating initial selection.
            self.selector_last_bed=self.cp.asarray(self.selector._bed(
                session.initial_tree_state,np.arange(self.selector.nn),self.model.topology.data))
            self._observe_selector_bed()

    def _selector_state(self):
        extra={}
        if getattr(self,'selector_last_bed',None) is not None:
            extra['selector_last_bed']=self.selector_last_bed
        if getattr(self.selector,'settings',{}).get('refine_bathymetry',False):
            extra['bed_gradient']=self.model.accepted_bed_gradient()
        return extra

    def _observe_selector_bed(self):
        """Commit only current leaves after a successful regrid observation.

        This includes unchanged topology and declared safe retained-mesh vetoes.
        It never runs on ordinary intermediate timesteps or rejected trials.
        """
        previous=getattr(self,'selector_last_bed',None)
        if previous is None:
            return
        cp=self.cp
        bed=self.model.solver.bed()
        require(bed.shape==(self.model.n,) and bed.dtype==cp.float64 and
                bool(cp.all(cp.isfinite(bed))), 'Invalid accepted AMR reference bed')
        candidate=previous.copy()
        candidate[cp.asarray(self.model.ids)]=bed
        self.selector_last_bed=candidate

    def _selector(self,prepared):
        arguments=dict(self.session.selector_arguments)
        if self.session.evolving_banks:
            arguments['indicator_source_path']=self.source/'amr_general_indicators.cu'
        return GeneralAMRStateSelector(prepared,**arguments)

    def _regrid(self,selected_ids,*,reject_after_candidate=False):
        key=tuple(map(int,selected_ids))
        saved=(self.entries,self.prepared,self.lineages,self.by_leaves.copy(),
               self.model.prepared,self.selector)
        physical=(self.model,self.tree_state,list(self.events),list(self.face_epochs),
                  getattr(self,'selector_last_bed',None))
        try:
            if key not in self.by_leaves:
                entry=self.session.get(selected_ids)
                entries=[*self.entries,entry]
                prepared,records=self.session.view(entries)
                top=prepared.topologies[-1]
                require(np.array_equal(outlet_owners(prepared,top),self.model.outlet_ids),
                        'On-demand mesh changes protected outlet owners')
                selector=self._selector(prepared)
                # Publishing geometry does not commit any evolving physical state.
                self.entries,self.prepared=entries,prepared
                self.lineages=({} if self.session.evolving_banks else validate_lineages(prepared,records))
                self.by_leaves[key]=top
                self.model.prepared=prepared
                self.selector=selector
            event=super()._regrid(selected_ids,reject_after_candidate=reject_after_candidate)
            self._observe_selector_bed()
            return event
        except BaseException:
            # An immutable cached operator may survive a rejected trial, but
            # accepted route membership, selection policy and checkpoint state
            # must be exactly those of the pre-trial model.
            (self.model,self.tree_state,self.events,self.face_epochs,self.selector_last_bed)=physical
            (self.entries,self.prepared,self.lineages,self.by_leaves,
             self.model.prepared,self.selector)=saved
            raise

    def advance(self,dt,*,delay_state,regrid=False,reject_after_candidate=False):
        if not (self.evolving_toe_variant or getattr(self,'bank_mode',2)==3):
            return super().advance(dt,selector=self.selector,delay_state=delay_state,regrid=regrid,
                                   reject_after_candidate=reject_after_candidate)
        from .amr_evolving_bank import BankRegridVeto
        old=self.model
        saved=old.snapshot()
        try:
            step=old.step(dt,snapshot=saved)
            if not regrid:
                return step,None,None
            selection_started=time.perf_counter()
            proposal=self.selector.propose(old.topology,old.full(),
                cached_tree_state=self.synchronized_tree(),delay_state=delay_state,time_s=old.time_s,
                **self._selector_state())
            if getattr(self.selector,'extended_indicators',False):
                step['amr_selection_wall_s']=time.perf_counter()-selection_started
            # This separate snapshot must not overwrite the pre-step rollback
            # buffer: a genuine numerical/IO error still rejects the full step.
            accepted=old.snapshot(independent=True)
            regrid_started=time.perf_counter()
            try:
                event=self._regrid(np.asarray(proposal.node_ids,dtype=np.int64),
                                   reject_after_candidate=reject_after_candidate)
            except BankRegridVeto as error:
                old.rollback(accepted)
                diagnostic=dict(code=error.code,reason=str(error),
                    requested_node_ids=list(proposal.node_ids),retained_topology=old.topology.name,
                    time_s=old.time_s,step=old.accepted_steps)
                step['amr_regrid_veto']=diagnostic
                proposal=self._retained_proposal(proposal,diagnostic)
                event=None
                self._observe_selector_bed()
            if getattr(self.selector,'extended_indicators',False):
                step['amr_regrid_wall_s']=time.perf_counter()-regrid_started
            return step,event,proposal
        except BaseException:
            old.rollback(saved)
            raise

    def _retained_proposal(self,proposal,diagnostic):
        """A declined mesh is not a declined physical timestep or mesh event."""
        current=set(map(int,self.model.ids));wanted=set(proposal.node_ids)
        affected=current.symmetric_difference(wanted)
        parent=self.prepared.tree['parent']
        for node in tuple(affected):
            ancestor=int(parent[node])
            while ancestor>=0:
                affected.add(ancestor);ancestor=int(parent[ancestor])
        delay=proposal.delay_state.copy()
        delay[list(affected)]=0
        receipt={**proposal.receipt,'requested_selection':dict(proposal.receipt),
                 'regrid_veto':diagnostic,'leaf_digest':leaf_digest(self.model.ids),
                 'new_leaves':len(self.model.ids),'proposed_operator_identity':self.model.topology.source_identity,
                 'catalog_hit':True}
        return replace(proposal,node_ids=tuple(map(int,self.model.ids)),topology=self.model.topology,
                       delay_state=delay,receipt=receipt)

    def write_checkpoint(self,folder,delay_state,*,method_identity):
        from . import amr_catalog_bank_checkpoint as checkpoint
        folder=filesystem_path(folder).resolve()
        if folder.exists():
            raise FileExistsError('On-demand checkpoint is write-once')
        folder.parent.mkdir(parents=True,exist_ok=True)
        pending=filesystem_path(tempfile.mkdtemp(prefix='.bank-amr-checkpoint-',dir=folder.parent))
        complete=pending/'complete'
        try:
            checkpoint.write(complete,self,delay_state,method_identity=method_identity,
                **(dict(requires_selector_last_bed=self.selector.settings['refine_sediment'])
                   if self.session.bank_mode==3 else {}))
            inventory=dict(schema=1,session_identity=self.session.identity,
                checkpoint_sha256=sha(complete/'checkpoint.json'),entries=[dict(
                    selected_ids=entry.selected_ids.tolist(),request_key=entry.request_key,
                    package_identity=entry.package_identity,entry_sha256=entry.entry_sha256)
                    for entry in self.entries])
            write_json_atomic(complete/'operators.json',inventory)
            # A visible checkpoint must contain all three closed artifacts.
            for attempt in range(6):
                if folder.exists():
                    raise FileExistsError('Concurrent checkpoint already exists')
                try:
                    os.rename(complete,folder)
                    break
                except OSError as error:
                    if getattr(error,'winerror',None) not in (5,32,33) or attempt==5:
                        raise
                    time.sleep(.02*2**attempt)
        finally:
            require(pending.parent==folder.parent and pending.name.startswith('.bank-amr-checkpoint-')
                    and not pending.is_symlink(),'Invalid owned checkpoint staging path')
            if pending.exists():
                try:
                    shutil.rmtree(pending)
                except OSError:
                    # Preserve the original publication failure if a scanner
                    # also holds its unpublished staging files.
                    pass

    @classmethod
    def restore(cls,session,folder,*,source_folder,method_identity):
        from . import amr_catalog_bank_checkpoint as checkpoint
        folder=filesystem_path(folder)
        inventory=json.loads((folder/'operators.json').read_text())
        require(set(inventory)=={'schema','session_identity','checkpoint_sha256','entries'} and
                inventory['schema']==1 and inventory['session_identity']==session.identity and
                inventory['checkpoint_sha256']==sha(folder/'checkpoint.json'),
                'On-demand checkpoint operator inventory differs')
        entries=[]
        for row in inventory['entries']:
            entry=session.store.get(row['selected_ids'])
            require(entry is not None and row==dict(selected_ids=entry.selected_ids.tolist(),
                request_key=entry.request_key,package_identity=entry.package_identity,
                entry_sha256=entry.entry_sha256),'Checkpoint operator content differs')
            entries.append(entry)
        require(entries and entries[0].selected_ids.tolist()==session.manifest['initial_node_ids'],
                'Checkpoint initial mesh differs')
        prepared,lineages=session.view(entries)
        require(json.loads((folder/'checkpoint.json').read_text())['schema']==
                session.checkpoint_schema,
                'Checkpoint bank history schema differs from session contract')
        restored,delay=checkpoint.restore(folder,prepared,source_folder=source_folder,
            initial_tree_state=session.initial_tree_state,method_identity=method_identity,lineages=lineages,
            expected_schema=session.checkpoint_schema,
            **(dict(requires_selector_last_bed=session.selector_arguments['settings']['refine_sediment'])
               if session.bank_mode==3 else {}))
        route=cls.__new__(cls)
        route.__dict__.update(restored.__dict__)
        route.session,route.entries=session,entries
        route.selector=route._selector(prepared)
        return route,delay
