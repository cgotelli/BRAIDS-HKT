"""Research-only general AMR route with an explicit slope-aware selector."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .amr_general_route import GeneralOnDemandRoute
from .amr_slope_indicator_v2 import VersionedSlopeAwareSelector
from .amr_general_stepper import require
from .amr_general_native_geometry import sha


METHOD = 'research_general_amr_slope_aware_v2'


class VersionedSlopeAwareRoute(GeneralOnDemandRoute):
    """Shares physics and topology transfer; changes only versioned tagging."""

    def _selector(self, prepared):
        original = super()._selector(prepared)
        return VersionedSlopeAwareSelector(original)

    def method_identity(self):
        transport = self.transport_source/'general_live_transport_sources.json'
        require(transport.is_file(),
                'Slope-aware method requires a frozen live transport source manifest')
        contract = dict(method=METHOD,
            selector_contract=self.selector.contract_identity,
            live_transport_manifest_sha256=sha(transport),
            route_source_sha256=sha(Path(__file__)),
            slope_selector_source_sha256=sha(
                Path(__file__).with_name('amr_slope_indicator_v2.py')),
            slope_geometry_source_sha256=sha(
                Path(__file__).with_name('amr_slope_indicator_research.py')),
            indicator_source_sha256=sha(Path(self.selector.original.indicator_source_path)
                if self.selector.original.indicator_source_path else
                Path(__file__).resolve().parents[1]/
                'braids_support/gpu_fwave/amr_general_indicators.cu'))
        return hashlib.sha256(json.dumps(contract,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()

    def write_versioned_checkpoint(self, folder, delay_state):
        return self.write_checkpoint(folder,delay_state,
                                     method_identity=self.method_identity())

    @classmethod
    def restore(cls, session, folder, *, source_folder,
                transport_source_folder, method_identity=None):
        from .amr_general_checkpoint import read

        route = cls(session,source_folder=source_folder,
                    transport_source_folder=transport_source_folder)
        expected = route.method_identity()
        if method_identity is not None:
            require(method_identity == expected,
                    'Slope-aware checkpoint method identity differs')
        prepared, top, entries, fields, events, delay = read(
            folder,session,method_identity=expected)
        route.entries = entries
        route.prepared = prepared
        route.by_leaves = {tuple(map(int,item.node_ids)):item
                           for item in prepared.topologies}
        route.selector = route._selector(prepared)
        route.model = route._make_stepper(top)
        route.model.install(route.cp.asarray(fields['state']))
        for name, value in route.model.solver.ledger.items():
            route.cp.copyto(value,route.cp.asarray(fields['ledger_'+name]))
        route.tree_state = route.cp.asarray(fields['tree_state'])
        route.model.time_s = float(fields['time_s'])
        route.model.accepted_steps = int(fields['accepted_steps'])
        route.model.maximum_cfl = float(fields['maximum_cfl'])
        route.model.solver.euler_calls = int(fields['euler_calls'])
        route.model.sync_accepted_clock()
        route.events = events
        route.selector_last_bed = (route.cp.asarray(fields['selector_last_bed'])
            if 'selector_last_bed' in fields else None)
        require(route.method_identity() == expected and
                np.array_equal(route.model.ids,top.node_ids),
                'Restored slope-aware selector or topology differs')
        return route,delay
