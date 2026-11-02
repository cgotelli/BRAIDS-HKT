"""Recursive fixed-topology epochs for on-demand general-transport AMR.

This research route reuses the accepted-state selector and conservative mesh
transition owned by :mod:`amr_general_route`, but each topology epoch advances
with arbitrary-depth 2:1 subcycling.  A completed root register is diagnostic
of the retired face graph and is never remapped as pending flux.
"""
from __future__ import annotations

import hashlib

from .amr_general_route import GeneralOnDemandRoute
from .amr_general_subcycle_stepper import RecursiveGeneralStepper


class RecursiveGeneralOnDemandRoute(GeneralOnDemandRoute):
    """General on-demand AMR with one recursive CUDA stepper per topology."""

    checkpoint_schema = 6

    def _make_stepper(self, top):
        return RecursiveGeneralStepper(self.prepared, top,
                                       source_folder=self.transport_source)

    def _handoff_stepper(self, old, new):
        old.sync_accepted_clock()
        new.begin_topology_epoch(old)
        register = old.cp.asnumpy(old.accepted_integral)
        return dict(recursive_epoch=dict(
            previous_origin=dict(old.epoch_origin),
            end_step=old.accepted_steps, end_time_s=old.time_s,
            euler_calls=old.solver.euler_calls,
            level_steps=old.level_steps.tolist(),
            old_operator_identity=old.topology.source_identity,
            consumed_register_shape=list(register.shape),
            consumed_register_sha256=hashlib.sha256(register.tobytes()).hexdigest(),
            remapped_as_pending_flux=False))

    def write_checkpoint(self, folder, delay_state, *, method_identity):
        from .amr_general_checkpoint import write_recursive
        return write_recursive(folder, self, delay_state,
                               method_identity=method_identity)

    @classmethod
    def restore(cls, session, folder, *, source_folder,
                transport_source_folder, method_identity):
        from .amr_general_checkpoint import restore_recursive
        return restore_recursive(folder, session, source_folder=source_folder,
            transport_source_folder=transport_source_folder,
            method_identity=method_identity)
