"""Recursive bank stepping through the existing conservative AMR handoff.

One macrostep closes all level registers, bank work and ledgers before a mesh
change. A register belongs to its old face graph; only physical state and bank
history are remapped. Geometry-dependent RK/filter work is rebuilt privately.
The ordinary synchronous route remains unchanged unless this route is selected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .amr_bank_subcycle import (BEDFORM_BANK_METHOD,
    BEDFORM_PHYSICAL_TOE_METHOD,
    SubcycledDistributedBankStepper, require)
from .amr_catalog_bank import CatalogBankRoute, SynchronousBankStepper, outlet_owners
from .atomic_output import save_arrays_atomic, write_json_atomic


class RecursiveBankStepper(SubcycledDistributedBankStepper):
    """Adapt the level-local engine to the shared accepted-state AMR protocol."""

    def __init__(self, prepared, topology, *, source_folder, evolving_toe_variant=False):
        require(prepared.schema == 2 and any(topology is item for item in prepared.topologies),
                'Recursive AMR requires an authenticated schema-2 topology')
        bank_mode = int(topology.data['bank_toe_erosion_mode'])
        require(bank_mode in (2, 3) and not (bank_mode == 3 and evolving_toe_variant),
                'Recursive AMR requires physical-toe mode 2 or distributed-bank mode 3')
        self.prepared, self.topology = prepared, topology
        self.ids = topology.node_ids.copy()
        self.outlet_ids = outlet_owners(prepared, topology)
        self.source = Path(source_folder).resolve()
        self.evolving_toe_variant = bool(evolving_toe_variant)
        self.bank_mode = bank_mode
        self.live_boundary = False  # no synchronous stage-clock ownership
        data = topology.data
        scheduled = any(np.any(data.get(key, 0)) for key in
                        ('nhydro_series', 'nfraction_series', 'fraction_bed_input')) or not np.all(data['mode'] == 2)
        super().__init__(data, prepared.tree['node_level'][self.ids], source_folder=self.source,
            euler_mode='level_local', reflux_mode='conservative', wet_front_mode='recursive',
            boundary_mode=('surveyed' if np.array_equal(data['mode'],[9,4]) else
                           'scheduled' if scheduled else 'constant_stage'),
            secondary_mode='native' if int(data['secondary_flow_model']) else 'off',
            evolving_toe_variant=self.evolving_toe_variant)
        self.n = self.nc

    install = SynchronousBankStepper.install
    accepted_gradients = SynchronousBankStepper.accepted_gradients
    accepted_bed_gradient = SynchronousBankStepper.accepted_bed_gradient
    totals = SynchronousBankStepper.totals
    ledger_totals = SynchronousBankStepper.ledger_totals

    def snapshot(self, *, independent=False):
        return self._snapshot(independent=independent)

    def rollback(self, saved):
        self._rollback(saved)

    def sync_accepted_clock(self):
        require(self.solver._distributed_external_clock is None and
                self.solver._distributed_external_phase is None and
                (self.forcing is None or self.forcing.clock is None),
                'Recursive AMR handoff requires a closed macrostep')

    def step(self, dt, *, snapshot=None, reject_after_active_toe=False):
        result = super().step(dt, snapshot=snapshot,
            reject_after='bank' if reject_after_active_toe else None)
        result['step'] = self.accepted_steps
        result['route'] = ((BEDFORM_PHYSICAL_TOE_METHOD
                            if self.bedform_physical_toe_composition else
                            BEDFORM_BANK_METHOD)
                           if self.bedform_bank_composition else
                           'recursive_physical_toe_bank_amr' if self.bank_mode == 2 else
                           'recursive_distributed_bank_amr')
        return result

    def frame(self):
        fields = SynchronousBankStepper.frame(self)
        if self.bedform_bank_composition:
            accepted = self.solver.fields()
            names = ('bedform_effective_manning','bedform_dune_height',
                     'bedform_dune_length','bedform_total_ks','bedform_iterations')
            require(all(name in accepted for name in names),
                    'Recursive bank bedform accepted diagnostics are incomplete')
            fields.update({name:accepted[name] for name in names})
        fields.update(recursive_level_steps=self.level_steps.copy(),
            recursive_epoch_start_steps=np.int64(self.epoch_origin['accepted_steps']),
            recursive_epoch_start_time=np.float64(self.epoch_origin['time_s']),
            recursive_epoch_start_euler_calls=np.int64(self.epoch_origin['euler_calls']),
            recursive_register=self.cp.asnumpy(self.accepted_integral),
            recursive_repair_totals=self.cp.asnumpy(self.repair_totals),
            recursive_preparation_impulse=self.cp.asnumpy(self.wet_preparation.impulse))
        return fields


class RecursiveCatalogBankRoute(CatalogBankRoute):
    """Reuse conservative tree transfer; replace only the macrostep engine."""

    def _make_stepper(self, topology):
        return RecursiveBankStepper(self.prepared, topology, source_folder=self.source,
                                    evolving_toe_variant=self.evolving_toe_variant)

    def _handoff_stepper(self, old, new):
        old.sync_accepted_clock()
        new.begin_topology_epoch(old)
        # The old register has already driven bank erosion exactly once. It is
        # a diagnostic of the retired face graph, never a new pending integral.
        register = old.cp.asnumpy(old.accepted_integral)
        epoch = dict(
            previous_origin=dict(old.epoch_origin),
            end_step=old.accepted_steps, end_time_s=old.time_s,
            euler_calls=old.solver.euler_calls,
            level_steps=old.level_steps.tolist(),
            old_operator_identity=old.topology.source_identity,
            consumed_register_shape=list(register.shape),
            consumed_register_sha256=hashlib.sha256(register.tobytes()).hexdigest(),
            consumed_by_bank=True, remapped_as_pending_flux=False)
        if old.bank_mode == 2:
            epoch.update(bank_mode=2,
                         physical_toe_integral_consumed=True,
                         evolving_toe_variant=old.evolving_toe_variant)
        return dict(recursive_epoch=epoch)

    def write_checkpoint(self, folder, delay_state, *, method_identity):
        """Write one synchronized topology epoch, including dormant tree memory."""
        folder = Path(folder)
        require(not folder.exists(), 'Recursive AMR checkpoint is write-once')
        delay = np.asarray(delay_state)
        nn = len(self.prepared.tree['parent'])
        require(delay.shape == (nn,) and delay.dtype.kind in 'iu' and np.all(delay >= 0),
                'Invalid recursive AMR selector delay')
        fields = {'engine_' + key: value for key, value in self.model.persistent().items()}
        fields.update(tree_state=self.cp.asnumpy(self.tree_state), delay_state=delay.astype(np.int64),
            events_json=np.asarray(json.dumps(self.events, sort_keys=True, allow_nan=False)),
            topology_identity=np.asarray(self.model.topology.source_identity),
            node_ids=self.model.ids.copy(), time_s=np.float64(self.model.time_s),
            accepted_steps=np.int64(self.model.accepted_steps),
            catalog_identity=np.asarray(self.prepared.identity),method_identity=np.asarray(method_identity))
        if getattr(self, 'selector_last_bed', None) is not None:
            fields['selector_last_bed'] = self.cp.asnumpy(self.selector_last_bed)
        schema = 2 if self.bank_mode == 2 else 1
        if schema == 2:
            fields.update(_physical_toe_checkpoint_fields(self))
        folder.mkdir(parents=True)
        save_arrays_atomic(folder/'state.npz', **fields)
        write_json_atomic(folder/'checkpoint.json', dict(schema=schema, kind='recursive_bank_amr',
            catalog_identity=self.prepared.identity, method_identity=method_identity,
            source_identity=self.model.source_identity, adapter_sources=_adapter_sources(),
            payload_sha256=_sha(folder/'state.npz')))

    @classmethod
    def restore(cls, folder, prepared, *, source_folder, initial_tree_state, method_identity,
                lineages=None, evolving_toe_variant=None):
        folder = Path(folder)
        manifest = json.loads((folder/'checkpoint.json').read_text())
        schema = manifest.get('schema')
        require(type(schema) is int and schema in (1, 2) and
                manifest == dict(schema=schema, kind='recursive_bank_amr',
            catalog_identity=prepared.identity, method_identity=method_identity,
            source_identity=manifest.get('source_identity'), adapter_sources=_adapter_sources(),
            payload_sha256=_sha(folder/'state.npz')), 'Recursive AMR checkpoint identity differs')
        with np.load(folder/'state.npz', allow_pickle=False) as archive:
            fields = {key: archive[key].copy() for key in archive.files}
        engine = {key.removeprefix('engine_'): value for key, value in fields.items()
                  if key.startswith('engine_')}
        required={'tree_state','delay_state','events_json','topology_identity','node_ids',
                  'time_s','accepted_steps','catalog_identity','method_identity'}
        outer = set(fields)-{'engine_'+key for key in engine}
        optional = {'selector_last_bed'} if 'selector_last_bed' in fields else set()
        bank_modes = {int(top.data['bank_toe_erosion_mode']) for top in prepared.topologies}
        require(len(bank_modes) == 1 and bank_modes <= {2, 3} and
                schema == (2 if 2 in bank_modes else 1),
                'Recursive checkpoint schema/catalog bank mode differs')
        if schema == 1:
            require(outer == required|optional,
                    'Recursive checkpoint payload field inventory differs')
            require(evolving_toe_variant in (None, False),
                    'Distributed recursive checkpoint cannot use a physical-toe map')
            evolving = False
        else:
            evolving, epoch_fields = _validate_physical_toe_checkpoint_fields(
                fields, prepared, required|optional)
            require(evolving_toe_variant is None or
                    bool(evolving_toe_variant) == evolving,
                    'Recursive checkpoint evolving-toe method differs')
            require(evolving or lineages is not None,
                    'Static recursive physical-toe restart requires authenticated lineages')
            require(outer == required|optional|epoch_fields,
                    'Recursive physical-toe checkpoint payload field inventory differs')
        require(fields['time_s'].shape==() and fields['time_s'].dtype==np.float64 and
                fields['accepted_steps'].shape==() and fields['accepted_steps'].dtype==np.int64 and
                np.array_equal(fields['time_s'],engine['time_s']) and
                np.array_equal(fields['accepted_steps'],engine['accepted_steps']) and
                str(fields['catalog_identity'])==prepared.identity and
                str(fields['method_identity'])==method_identity,
                'Recursive checkpoint worker clock/identity differs')
        topology = str(fields['topology_identity'])
        indices = [i for i, top in enumerate(prepared.topologies) if top.source_identity == topology]
        require(len(indices) == 1, 'Recursive checkpoint topology is unavailable')
        top = prepared.topologies[indices[0]]
        nn, ns = len(prepared.tree['parent']), int(top.data['nstate'])
        require(np.array_equal(fields['node_ids'], top.node_ids) and
                fields['tree_state'].shape == (nn, ns) and fields['tree_state'].dtype == np.float64 and
                np.isfinite(fields['tree_state']).all() and fields['delay_state'].shape == (nn,) and
                fields['delay_state'].dtype == np.int64 and np.all(fields['delay_state'] >= 0),
                'Invalid recursive checkpoint tree or selector state')
        if 'selector_last_bed' in fields:
            previous=fields['selector_last_bed']
            require(previous.shape==(nn,) and previous.dtype==np.float64 and
                    np.isfinite(previous).all(),'Invalid recursive checkpoint temporal bed reference')
        try:
            events = json.loads(str(fields['events_json']))
        except (TypeError, ValueError) as error:
            raise ValueError('Invalid recursive checkpoint event JSON') from error
        _validate_events(events, prepared, topology, engine, bank_mode=next(iter(bank_modes)))
        archived = (_decode_physical_toe_face_epochs(fields, prepared, events, evolving)
                    if schema == 2 else [])
        route = cls(prepared, source_folder=source_folder, initial_tree_state=initial_tree_state,
                    lineages=lineages, initial_topology=indices[0],
                    evolving_toe_variant=evolving)
        require(route.model.source_identity == manifest['source_identity'],
                'Recursive checkpoint numerical source differs')
        route.model.restore(engine)
        route.tree_state = route.cp.asarray(fields['tree_state'])
        route.events = events
        route.face_epochs = [{key:(route.cp.asarray(value) if key in
            ('old_face','fraction','signed','gross') else value)
            for key,value in epoch.items()} for epoch in archived]
        route.selector_last_bed = (route.cp.asarray(fields['selector_last_bed'])
                                  if 'selector_last_bed' in fields else None)
        return route, fields['delay_state'].copy()


_FACE_EPOCH_KEYS = ('operator_identity', 'node_ids', 'old_face', 'fraction',
                    'signed', 'gross')
_PHYSICAL_TOE_META = {'recursive_bank_mode', 'recursive_evolving_toe_variant',
                      'face_epoch_count'}


def _physical_toe_checkpoint_fields(route):
    """Serialize route-owned physical-face history outside the cell engine.

    The engine checkpoint owns the current topology's resident map and face
    totals.  Closed face epochs belong to the AMR route because they refer to
    retired graphs and must never be installed as a pending Exner register.
    """
    require(route.bank_mode == 2 and type(route.evolving_toe_variant) is bool,
            'Physical-toe recursive checkpoint route differs')
    epochs = list(route.face_epochs)
    require(route.evolving_toe_variant or not epochs,
            'Static physical-toe route cannot own retired face epochs')
    fields = dict(recursive_bank_mode=np.int64(2),
                  recursive_evolving_toe_variant=np.int64(route.evolving_toe_variant),
                  face_epoch_count=np.int64(len(epochs)))
    for index, epoch in enumerate(epochs):
        require(set(epoch) == set(_FACE_EPOCH_KEYS),
                'Recursive retired face epoch field inventory differs')
        prefix = f'face_epoch_{index}_'
        fields[prefix+'operator_identity'] = np.asarray(epoch['operator_identity'])
        fields[prefix+'node_ids'] = np.asarray(epoch['node_ids']).copy()
        for key in ('old_face', 'fraction', 'signed', 'gross'):
            value = epoch[key]
            fields[prefix+key] = (route.cp.asnumpy(value) if
                                  isinstance(value, route.cp.ndarray) else
                                  np.asarray(value).copy())
    return fields


def _validate_physical_toe_checkpoint_fields(fields, prepared, base_fields):
    """Host-only inventory/type preflight for recursive mode-2 metadata."""
    del base_fields  # documents that engine/base fields are owned elsewhere
    for name in _PHYSICAL_TOE_META:
        value = np.asarray(fields.get(name))
        require(value.shape == () and value.dtype == np.dtype(np.int64),
                'Invalid recursive physical-toe checkpoint field '+name)
    mode = int(fields['recursive_bank_mode'])
    evolving_value = int(fields['recursive_evolving_toe_variant'])
    count = int(fields['face_epoch_count'])
    require(mode == 2 and evolving_value in (0, 1) and count >= 0 and
            count <= len(fields)//len(_FACE_EPOCH_KEYS) and
            all(int(top.data['bank_toe_erosion_mode']) == 2
                for top in prepared.topologies),
            'Recursive physical-toe checkpoint metadata differs')
    evolving = bool(evolving_value)
    require(evolving or count == 0,
            'Static recursive physical-toe checkpoint has retired face epochs')
    names = set(_PHYSICAL_TOE_META)
    names.update(f'face_epoch_{index}_{key}' for index in range(count)
                 for key in _FACE_EPOCH_KEYS)
    return evolving, names


def _decode_physical_toe_face_epochs(fields, prepared, events, evolving):
    """Validate archived physical-face ledgers before constructing CUDA."""
    count = int(fields['face_epoch_count'])
    require((evolving and count == len(events)) or (not evolving and count == 0),
            'Recursive face epoch/event count differs')
    if not evolving:
        return []
    by_name = {top.name:top for top in prepared.topologies}
    ng = int(prepared.topologies[0].data['nfrac'])
    result = []
    for index, event in enumerate(events):
        prior = by_name[event['old_topology']]
        prefix = f'face_epoch_{index}_'
        identity = np.asarray(fields[prefix+'operator_identity'])
        node_ids = np.asarray(fields[prefix+'node_ids'])
        faces = np.asarray(fields[prefix+'old_face'])
        fraction = np.asarray(fields[prefix+'fraction'])
        signed = np.asarray(fields[prefix+'signed'])
        gross = np.asarray(fields[prefix+'gross'])
        require(identity.shape == () and identity.dtype.kind == 'U' and
                str(identity) == prior.source_identity and
                node_ids.dtype == np.dtype(np.int64) and
                np.array_equal(node_ids, prior.node_ids) and
                faces.ndim == 1 and faces.dtype == np.dtype(np.int64) and
                np.all((faces >= 0) & (faces < len(prior.data['owner']))) and
                np.all(np.diff(faces) > 0) and
                fraction.shape == faces.shape and fraction.dtype == np.dtype(np.float64) and
                np.isfinite(fraction).all() and np.all((fraction > 0.) & (fraction <= 1.)) and
                signed.shape == (ng, len(faces)) and signed.dtype == np.dtype(np.float64) and
                gross.shape == signed.shape and gross.dtype == np.dtype(np.float64) and
                np.isfinite(signed).all() and np.isfinite(gross).all(),
                'Invalid recursive retired physical-face epoch')
        tolerance = 1.e-12*np.maximum(1., np.abs(signed))
        boundary = np.asarray(prior.data['neighbour'])[faces] < 0
        require(np.all(gross >= 0.) and np.all(gross+tolerance >= np.abs(signed)) and
                not np.any(signed[:, boundary]) and not np.any(gross[:, boundary]),
                'Invalid recursive retired physical-face ledger')
        result.append(dict(operator_identity=str(identity), node_ids=node_ids.copy(),
            old_face=faces.copy(), fraction=fraction.copy(), signed=signed.copy(),
            gross=gross.copy()))
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _adapter_sources():
    root = Path(__file__).resolve().parent
    names = ('amr_recursive_route.py', 'amr_catalog_bank.py', 'amr_catalog_mixed_transition.py',
             'amr_bank_wetfront.py', 'amr_prepared.py', 'amr_on_demand.py',
             'amr_general_selector.py', 'amr_evolving_bank.py',
             'amr_face_correspondence.py')
    return {name: _sha(root/name) for name in names}


def _validate_events(events, prepared, topology, engine, *, bank_mode=3):
    require(isinstance(events, list), 'Invalid recursive AMR event list')
    require(bank_mode in (2, 3), 'Invalid recursive checkpoint bank mode')
    by_name = {top.name: top for top in prepared.topologies}
    previous = prepared.topologies[0]
    last_step, last_time, last_euler = 0, 0., 0
    for event in events:
        require(isinstance(event, dict) and event.get('old_topology') == previous.name and
                event.get('new_topology') in by_name and
                event.get('new_topology') != previous.name and 'recursive_epoch' in event,
                'Recursive checkpoint mesh-event chain differs')
        epoch = event['recursive_epoch']
        require(event['old_operator_identity'] == previous.source_identity and
                event['new_operator_identity'] == by_name[event['new_topology']].source_identity and
                last_step <= epoch['end_step'] <= int(engine['accepted_steps']) and
                last_time <= epoch['end_time_s'] <= float(engine['time_s']) and
                epoch['consumed_by_bank'] is True and epoch['remapped_as_pending_flux'] is False and
                epoch['previous_origin'] == dict(accepted_steps=last_step, time_s=last_time,
                                                 euler_calls=last_euler),
                'Recursive checkpoint topology-epoch counters differ')
        if bank_mode == 2:
            require(epoch.get('bank_mode') == 2 and
                    epoch.get('physical_toe_integral_consumed') is True and
                    type(epoch.get('evolving_toe_variant')) is bool,
                    'Recursive physical-toe epoch transaction differs')
        last_step, last_time, last_euler = epoch['end_step'], epoch['end_time_s'], epoch['euler_calls']
        previous = by_name[event['new_topology']]
    require(previous.source_identity == topology, 'Recursive events do not reach checkpoint mesh')
    if events:
        require((int(engine['epoch_start_steps']), float(engine['epoch_start_time_s']),
                 int(engine['epoch_start_euler_calls'])) == (last_step, last_time, last_euler),
                'Recursive engine origin differs from retired topology epochs')


from .amr_on_demand import OnDemandBankRoute, digest, sha


class RecursiveOnDemandBankRoute(OnDemandBankRoute, RecursiveCatalogBankRoute):
    """The same lazy geometry/selector lifecycle with recursive CUDA physics."""

    def write_checkpoint(self, folder, delay_state, *, method_identity):
        folder=Path(folder)
        RecursiveCatalogBankRoute.write_checkpoint(self,folder,delay_state,
                                                   method_identity=method_identity)
        write_json_atomic(folder/'operators.json',dict(schema=1,session_identity=self.session.identity,
            checkpoint_sha256=sha(folder/'checkpoint.json'),entries=[dict(
                selected_ids=entry.selected_ids.tolist(),request_key=entry.request_key,
                package_identity=entry.package_identity,entry_sha256=entry.entry_sha256)
                for entry in self.entries]))

    @classmethod
    def restore(cls,session,folder,*,source_folder,method_identity):
        folder=Path(folder)
        inventory=json.loads((folder/'operators.json').read_text())
        require(set(inventory)=={'schema','session_identity','checkpoint_sha256','entries'} and
                inventory['schema']==1 and inventory['session_identity']==session.identity and
                inventory['checkpoint_sha256']==sha(folder/'checkpoint.json'),
                'Recursive on-demand checkpoint operator inventory differs')
        entries=[]
        for row in inventory['entries']:
            entry=session.store.get(row['selected_ids'])
            require(entry is not None and row==dict(selected_ids=entry.selected_ids.tolist(),
                request_key=entry.request_key,package_identity=entry.package_identity,
                entry_sha256=entry.entry_sha256),'Recursive checkpoint cached geometry differs')
            entries.append(entry)
        require(entries and entries[0].selected_ids.tolist()==session.manifest['initial_node_ids'],
                'Recursive checkpoint initial mesh differs')
        prepared,lineages=session.view(entries)
        restored,delay=RecursiveCatalogBankRoute.restore(folder,prepared,source_folder=source_folder,
            initial_tree_state=session.initial_tree_state,method_identity=method_identity,
            lineages=lineages,
            evolving_toe_variant=session.evolving_banks and session.bank_mode==2)
        route=cls.__new__(cls)
        route.__dict__.update(restored.__dict__)
        route.session,route.entries=session,entries
        route.selector=route._selector(prepared)
        require((route.selector_last_bed is not None)==route.selector.settings['refine_sediment'],
                'Recursive checkpoint temporal bed reference differs from selector')
        return route,delay
