"""Additive Q3-H3/v59 selected-mesh transaction route.

The authenticated :mod:`amr_general_route` remains the frozen M2-C/typed-
roughness parent.  This subclass adds only the live prognostic-secondary v59
handoff, cancellation boundaries and fail-closed restart behavior required by
the closed, synchronous implicit-midpoint slice.
"""
from __future__ import annotations

import numpy as np

from .amr_catalog_mixed_transition import (
    plan_mixed_transition,
    propose_mixed_transition,
)
from .amr_general_route import (
    GeneralOnDemandRoute,
    validate_general_boundary_handoff,
)
from .amr_general_stepper import require
from .general_live_transport import static_bed_enabled


class LiveV59GeneralOnDemandRoute(GeneralOnDemandRoute):
    """Closed synchronous selected route with one live v59 owner."""

    def __init__(self, *args, integrator="implicit_fwave_midpoint", **kwargs):
        require(integrator == "implicit_fwave_midpoint",
                "Q3-H3/v59 selected route requires implicit_fwave_midpoint")
        super().__init__(*args, integrator=integrator, **kwargs)
        require(self.model.prognostic_secondary_amr_live_v59 is not None,
                "Q3-H3/v59 selected route lacks its live controller")

    def _handoff_stepper(self, old, new):
        """Transfer accepted clocks and each composed physical ledger once."""
        new.accepted_steps = old.accepted_steps
        new.time_s = old.time_s
        new.maximum_cfl = old.maximum_cfl
        new.solver.euler_calls = old.solver.euler_calls
        require(self.integrator == "implicit_fwave_midpoint" and
                old.solver.implicit_midpoint is not None and
                new.solver.implicit_midpoint is not None,
                "Q3-H3/v59 topology handoff lost the implicit controller")
        old_controller = old.solver.implicit_midpoint
        new_controller = new.solver.implicit_midpoint
        old_base = getattr(old_controller, "base", old_controller)
        new_base = getattr(new_controller, "base", new_controller)
        suspended = bool(old_base.suspended_composition)
        require(suspended == bool(new_base.suspended_composition),
                "Implicit topology handoff changed suspension ownership")
        memory = None
        if suspended:
            require(old.solver.flow2dh_boundary_diffusion is not None and
                    new.solver.flow2dh_boundary_diffusion is not None,
                    "M2-C topology handoff requires law-11/mode-4 suspension")
            memory = old.solver.flow2dh_boundary_diffusion.\
                boundary_memory_fields()
            new.solver.flow2dh_boundary_diffusion.\
                restore_boundary_memory(memory)
        else:
            require(old.solver.flow2dh_boundary_diffusion is None and
                    new.solver.flow2dh_boundary_diffusion is None,
                    "Non-suspended implicit handoff acquired boundary memory")

        # The predictor is a topology-local rate.  Interpolating it would
        # invent an AMR law, so the new topology starts Newton from zero.
        new_base.predictor_rate.fill(0.0)
        new_controller.have_predictor = False
        new_controller.accepted_steps = old_controller.accepted_steps
        new_controller.rejected_steps = old_controller.rejected_steps
        new_controller.fwave_evaluations = old_controller.fwave_evaluations
        new_controller.linear_matvecs = old_controller.linear_matvecs
        new_controller.last_diagnostics = {}
        new_base.equilibrium_candidate = False
        new_base._refresh_equilibrium_scales()
        receipt = dict(
            implicit_predictor_handoff=
                "invalidated_zero_topology_local_initial_guess_v1",
            implicit_suspended_composition=suspended,
            automatic_CPU_fallback=False,
        )
        if suspended:
            new.wetdry_phase_ledger += old.wetdry_phase_ledger
            receipt.update(
                flow2dh_boundary_memory_handoff=
                    "ordered_unchanged_finest_physical_faces_bitwise_v1",
                flow2dh_boundary_memory_fields=sorted(memory),
                suspended_amr_contract_semantic_id=
                    new.suspended_amr_contract.semantic_id,
                suspended_amr_contract_fingerprint=
                    new.suspended_amr_contract.fingerprint,
                wetdry_phase_ledger_kg=
                    self.cp.asnumpy(new.wetdry_phase_ledger).tolist(),
            )
        else:
            receipt.update(
                flow2dh_boundary_memory_handoff="not_applicable",
                flow2dh_boundary_memory_fields=[],
            )

        old_v59 = old.prognostic_secondary_amr_live_v59
        new_v59 = new.prognostic_secondary_amr_live_v59
        require(old_v59 is not None and new_v59 is not None,
                "Q3-H3/v59 topology handoff lost live route ownership")
        # ``new.install`` changed public accepted arrays after construction.
        # Synchronize private snapshots, transfer physical cumulative ledgers,
        # and consume/invalidate topology-local face registers exactly once.
        new_v59.synchronize_installed_state(new)
        receipt.update(old_v59.transfer_private_accepted_state(new_v59))
        receipt.update(
            v59_live_selected_handoff=True,
            v59_corrected_betac_override=
                "receiving_cell_accepted_depth_exactly_once_v1",
            v59_duplicate_within_step_physics=False,
        )
        new.sync_accepted_clock()
        return receipt

    def _regrid(self, selected_ids, *, reject_after_candidate=False):
        """Publish state, physical ledgers and topology as one transaction."""
        ids = np.asarray(selected_ids)
        require(ids.ndim == 1 and ids.dtype.kind in "iu" and
                np.all(np.diff(ids) > 0),
                "General AMR proposal needs unique sorted global IDs")
        key = tuple(map(int, ids))
        registered = (self.entries, self.prepared, self.by_leaves.copy(),
                      self.selector, self.model.prepared)
        physical = (self.model, self.tree_state, list(self.events),
                    self.selector_last_bed)
        try:
            if self.cancelled():
                raise InterruptedError(
                    "Q3-H3/v59 selected regrid cancelled before proposal")
            if key not in self.by_leaves:
                entry = self.session.get(ids)
                entries = [*self.entries, entry]
                prepared = self.session.view(entries)
                target = prepared.topologies[-1]
                validate_general_boundary_handoff(
                    prepared, self.model.topology, target)
                selector = self._selector(prepared)
                self.entries, self.prepared = entries, prepared
                self.by_leaves[key] = target
                self.model.prepared = prepared
                self.selector = selector
            target = self.by_leaves[key]
            old = self.model
            if target is old.topology:
                self._observe_selector_bed()
                return None
            validate_general_boundary_handoff(
                self.prepared, old.topology, target)
            plan = plan_mixed_transition(self.prepared, old.topology, target)

            roughness_candidate = None
            roughness_receipt = {}
            from .amr_typed_roughness import (
                BINDING_FIELD as AMR_ROUGHNESS_BINDING_FIELD,
                propose_resident_transfer,
            )
            roughness_binding = old.topology.data.get(
                AMR_ROUGHNESS_BINDING_FIELD)
            if roughness_binding is not None:
                typed_tree = dict(
                    parent=np.ascontiguousarray(
                        self.prepared.tree["parent"], dtype=np.int64),
                    node_level=np.ascontiguousarray(
                        self.prepared.tree["node_level"], dtype=np.int64),
                    area=np.ascontiguousarray(
                        self.prepared.tree["area"], dtype=np.float64),
                    roughness_parameter=np.ascontiguousarray(
                        self.session.tree["roughness_parameter"],
                        dtype=np.float64),
                    raw_roughness_mode=np.asarray(
                        self.session.tree["raw_roughness_mode"]),
                )
                roughness_candidate, roughness_receipt = \
                    propose_resident_transfer(
                        roughness_binding, typed_tree, old.ids,
                        target.node_ids,
                        old.solver.sediment.geometry["manning_cell"],
                        backend="cuda")

            tree = self.synchronized_tree()
            gradients = old.accepted_gradients() if plan.refine_parents else None
            old_map = {int(node): i for i, node in enumerate(old.ids)}
            inputs = dict(
                refinement_children={
                    parent: self.cp.ascontiguousarray(
                        tree[self.cp.asarray(group)])
                    for parent, group in zip(
                        plan.refine_parents, plan.refine_children)
                },
                primitive_gradients={
                    parent: self.cp.ascontiguousarray(gradients[old_map[parent]])
                    for parent in plan.refine_parents
                },
                reject_after_candidate=reject_after_candidate,
            )
            if static_bed_enabled(old.topology.data):
                from .amr_general_static_transfer import (
                    propose_static_transition,
                )
                proposal = propose_static_transition(
                    self.prepared, plan, old.full(), **inputs)
            else:
                proposal = propose_mixed_transition(
                    self.prepared, plan, old.full(),
                    source_folder=self.source,
                    suspended_wetdry_source=(
                        self.transport_source / "implicit_suspension_amr.cu"
                        if self.integrator == "implicit_fwave_midpoint"
                        else None),
                    transfer_bank_history=False,
                    **inputs,
                )
            if self.cancelled():
                raise InterruptedError(
                    "Q3-H3/v59 selected regrid cancelled after transfer")

            new = self._make_stepper(target)
            new.install(proposal.state)
            if roughness_candidate is not None:
                require(self.cp.array_equal(
                    new.solver.sediment.geometry["manning_cell"],
                    roughness_candidate),
                    "General AMR CUDA raw roughness handoff differs")
            for name, value in old.solver.ledger.items():
                require(name in new.solver.ledger and
                        new.solver.ledger[name].shape == value.shape and
                        new.solver.ledger[name].dtype == value.dtype,
                        "General AMR boundary ledger shape changes: " + name)
                self.cp.copyto(new.solver.ledger[name], value)

            old_v59 = old.prognostic_secondary_amr_live_v59
            new_v59 = new.prognostic_secondary_amr_live_v59
            require(old_v59 is not None and new_v59 is not None,
                    "Q3-H3/v59 candidate changed live route ownership")
            v59_transfer = old_v59.audit_transition(
                new_v59, old, new, plan, proposal,
                inputs["primitive_gradients"])
            handoff = self._handoff_stepper(old, new)

            if new.suspended_amr_contract is not None:
                # The mixed transition already moves dry suspended material;
                # book its accepted phase ledger once, after successful audit.
                phase = np.asarray(proposal.receipt.get(
                    "wetdry_suspended_phase_change_kg"))
                require(phase.shape == (new.ng,) and
                        phase.dtype.kind == "f" and
                        np.isfinite(phase).all() and np.all(phase <= 0.0),
                        "M2-C regrid suspended phase ledger differs")
                device_phase = self.cp.asarray(phase, dtype=self.cp.float64)
                new.wetdry_phase_ledger[:, 0] += device_phase
                new.wetdry_phase_ledger[:, 1] -= device_phase
                new.last_wetdry_phase_change += device_phase
                handoff["accepted_regrid_suspended_phase_change_kg"] = \
                    phase.tolist()
                handoff["wetdry_phase_ledger_kg"] = self.cp.asnumpy(
                    new.wetdry_phase_ledger).tolist()

            tree[self.cp.asarray(new.ids)] = new.full()
            if self.cancelled():
                raise InterruptedError(
                    "Q3-H3/v59 selected regrid cancelled before commit")
            event = dict(
                time_s=old.time_s,
                step=old.accepted_steps,
                old_topology=old.topology.name,
                new_topology=target.name,
                **proposal.receipt,
                **v59_transfer,
                **handoff,
            )
            if roughness_receipt:
                event["typed_roughness_transfer"] = roughness_receipt
            self.tree_state, self.model = tree, new
            self.events.append(event)
            self._observe_selector_bed()
            return event
        except BaseException:
            (self.model, self.tree_state, self.events,
             self.selector_last_bed) = physical
            (self.entries, self.prepared, self.by_leaves,
             self.selector, self.model.prepared) = registered
            raise

    def frame(self):
        values = super().frame()
        binding = self.model.prognostic_secondary_amr_live_v59
        require(binding is not None,
                "Q3-H3/v59 frame lost live route ownership")
        proxy = binding.proxy
        values["prognostic_secondary_amr_live_scope_id"] = np.asarray(
            proxy.diagnostic()["scope_id"])
        values["prognostic_secondary_amr_live_corrected_betac"] = np.int32(1)
        values["prognostic_secondary_amr_signed_hi_face_register"] = \
            self.cp.asnumpy(proxy.accepted_signed_hi_register)
        values["prognostic_secondary_amr_betac_velocity_register"] = \
            self.cp.asnumpy(proxy.accepted_velocity_register)
        values["prognostic_secondary_amr_register_generation"] = np.int64(
            proxy.accepted_register_generation)
        values["prognostic_secondary_amr_topology_generation"] = np.int64(
            proxy.topology_generation)
        return values

    def write_checkpoint(self, folder, delay_state, *, method_identity):
        raise ValueError("Q3-H3/v59 live selected restart is not implemented")

    @classmethod
    def restore(cls, *args, **kwargs):
        raise ValueError("Q3-H3/v59 live selected restart is not implemented")


__all__ = ["LiveV59GeneralOnDemandRoute"]
