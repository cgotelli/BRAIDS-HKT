"""Additive live selected-mesh Q3-H3/v59 GPU composition.

The accepted v58 implicit controller remains the only macrostep owner.  This
module replaces only its legacy ``Betac`` residual injection, authenticates the
general selected mesh, and audits the already GPU-resident conservative AMR
state transfer.  It deliberately does not run another advection, diffusion,
Coriolis, sediment or Exner pass.

The first live slice is fully wet and closed-wall.  Numerical open boundaries,
recursive subcycling and restart stay fail-closed until their separate owners
are composed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
from types import MethodType

import numpy as np

from braids_core.sediment.prognostic_secondary_nonzero_execution import (
    EXECUTION_KEY as NONZERO_EXECUTION_KEY,
)


SCOPE_ID = "q3h3-v59-live-synchronous-selected-mesh-composition-seam-v1"
INTEGRATOR = "implicit_fwave_midpoint"
CUDA_SOURCE_NAME = "flow2dh_prognostic_amr_live_v59.cu"
SOURCE_MANIFEST_NAME = "q3h3_amr_live_v59_sources.json"
LIVE_CAPABILITIES = (
    "general_selected_topology",
    "full_tree_active_leaf_projection",
    "signed_hi_accepted_boundary_handoff",
    "transactional_refine_coarsen",
    "external_v58_macrostep_ownership",
    "corrected_depth_weighted_betac_override",
    "stable_finest_open_boundary_catalog_handoff",
)
CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
KERNEL_NAMES = (
    "v59_live_signed_hi_register",
    "v59_live_add_depth_weighted_betac",
    "v59_live_betac_register",
    "v59_live_gather_betac_increment",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_LIVE_BODY = {
    "schema": 1,
    "scope_id": SCOPE_ID,
    "capabilities": list(LIVE_CAPABILITIES),
    "automatic_cpu_fallback": False,
    "external_v58_macrostep_owner": True,
}
LIVE_SELECTED_INTERFACE = {
    **_LIVE_BODY, "fingerprint": _digest(_LIVE_BODY),
}


class CorrectedBetacV59:
    """Proxy a v58 component while replacing its Betac residual exactly once."""

    def __init__(self, wrapped, *, source_path: str | Path):
        self.wrapped = wrapped
        self.cp = wrapped.cp
        self.device_id = wrapped.device_id
        self.nc, self.nf = wrapped.nc, wrapped.nf
        self.source_path = Path(source_path).resolve()
        if (not self.source_path.is_file() or self.source_path.is_symlink()):
            raise ValueError("Q3-H3/v59 live Betac CUDA source is missing or linked")
        self.source_sha256 = _sha(self.source_path)
        with self.cp.cuda.Device(self.device_id):
            module = self.cp.RawModule(
                code=self.source_path.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.kernels = {
                name: module.get_function(name) for name in KERNEL_NAMES}
            self.face_status = self.cp.zeros(self.nf, dtype=self.cp.int32)
            self.cell_status = self.cp.zeros(self.nc, dtype=self.cp.int32)
            self.trial_velocity_register = self.cp.zeros(
                (2, self.nf), dtype=self.cp.float64)
            self.trial_cell_momentum_increment = self.cp.zeros(
                (2, self.nc), dtype=self.cp.float64)
            self.accepted_velocity_register = self.cp.zeros_like(
                self.trial_velocity_register)
            self.accepted_cell_momentum_increment = self.cp.zeros_like(
                self.trial_cell_momentum_increment)
            self.trial_signed_hi_register = self.cp.zeros(
                (2, self.nf), dtype=self.cp.float64)
            self.accepted_signed_hi_register = self.cp.zeros_like(
                self.trial_signed_hi_register)
        self.accepted_register_generation = 0
        self.topology_generation = 0
        self._register_ready = False
        self._signed_hi_register_ready = False
        self._legacy_injection_calls = 0
        self._corrected_injection_calls = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def _launch(self, name, count, arguments):
        if count:
            with self.cp.cuda.Device(self.device_id):
                self.kernels[name](((count + 255)//256,), (256,), arguments)

    def _status(self) -> int:
        face = int(self.cp.max(self.face_status).item()) if self.nf else 0
        cell = int(self.cp.max(self.cell_status).item()) if self.nc else 0
        return max(face, cell)

    def begin_trial(self, accepted_q, accepted_hi):
        self.trial_velocity_register.fill(0.0)
        self.trial_cell_momentum_increment.fill(0.0)
        self.trial_signed_hi_register.fill(0.0)
        self.face_status.fill(0); self.cell_status.fill(0)
        self._register_ready = False
        self._signed_hi_register_ready = False
        return self.wrapped.begin_trial(accepted_q, accepted_hi)

    def capture_signed_hi_register(self, dt, advective_flux):
        """Capture existing v58 fluxes; never recompute signed-hI transport."""
        self.wrapped._require_trial()
        if self._signed_hi_register_ready:
            raise RuntimeError(
                "Q3-H3/v59 signed-hI register is already captured")
        self.wrapped._fp64(
            "accepted signed-hI face flux", advective_flux, (self.nf,))
        self.face_status.fill(0)
        self.trial_signed_hi_register.fill(0.0)
        self._launch("v59_live_signed_hi_register", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.wrapped.owner,
            self.wrapped.neighbour, self.wrapped.length, advective_flux,
            self.wrapped.diffusion_conductance, self.wrapped.x,
            np.float64(dt), self.trial_signed_hi_register,
            self.face_status))
        if int(self.cp.max(self.face_status).item()) != 0:
            raise RuntimeError("Q3-H3/v59 signed-hI register rejected")
        self._signed_hi_register_ready = True
        return self.trial_signed_hi_register

    def freeze_betac(self, *args, **kwargs):
        # O1 has not yet supplied a numerical open-face law.  Do not inherit
        # v58's open implementation by accident merely because it exists.
        if bool(self.cp.any(self.wrapped.open_face != 0).item()):
            raise ValueError(
                "Q3-H3/v59 live Betac currently requires a closed selected mesh")
        return self.wrapped.freeze_betac(*args, **kwargs)

    def add_frozen_stress(self, flux_left, flux_right):
        """Replace, rather than augment, the legacy equal-depth injection."""
        self.wrapped._require_trial()
        if not self.wrapped._stress_frozen:
            raise RuntimeError("Q3-H3/v59 Betac field has not been frozen")
        self.wrapped._fp64("left f-wave flux", flux_left, (3, self.nf))
        self.wrapped._fp64("right f-wave flux", flux_right, (3, self.nf))
        self.face_status.fill(0)
        self._launch("v59_live_add_depth_weighted_betac", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.wrapped.owner,
            self.wrapped.neighbour, self.wrapped.accepted_q_snapshot,
            self.wrapped.frozen_stress_flux, flux_left, flux_right,
            self.face_status))
        if int(self.cp.max(self.face_status).item()) != 0:
            raise RuntimeError("Q3-H3/v59 corrected Betac residual rejected")
        self._corrected_injection_calls += 1
        return self.wrapped.frozen_stress_flux

    def accept_stress_ledger(self, dt):
        # Preserve the v58 boundary transaction (it is exactly zero in this
        # admitted closed slice), then create the v59 face/cell audit register.
        result = self.wrapped.accept_stress_ledger(dt)
        self.face_status.fill(0); self.cell_status.fill(0)
        self.trial_velocity_register.fill(0.0)
        self.trial_cell_momentum_increment.fill(0.0)
        self._launch("v59_live_betac_register", self.nf, (
            np.int32(self.nc), np.int32(self.nf), self.wrapped.owner,
            self.wrapped.neighbour, self.wrapped.length,
            self.wrapped.frozen_stress_flux, np.float64(dt),
            self.trial_velocity_register, self.face_status))
        if not int(self.cp.max(self.face_status).item()):
            self._launch("v59_live_gather_betac_increment", self.nc, (
                np.int32(self.nc), np.int32(self.nf), self.wrapped.offset,
                self.wrapped.faces, self.wrapped.owner,
                self.wrapped.neighbour, self.wrapped.accepted_q_snapshot,
                self.trial_velocity_register,
                self.trial_cell_momentum_increment, self.cell_status))
        if self._status() != 0:
            raise RuntimeError("Q3-H3/v59 corrected Betac register rejected")
        self._register_ready = True
        return result

    def finish_trial(self, accepted_q, accepted_hi, *, commit):
        if commit and (not self._register_ready or
                       not self._signed_hi_register_ready):
            raise RuntimeError(
                "Q3-H3/v59 cannot commit without both accepted registers")
        result = self.wrapped.finish_trial(
            accepted_q, accepted_hi, commit=commit)
        if commit:
            self.cp.copyto(
                self.accepted_velocity_register,
                self.trial_velocity_register)
            self.cp.copyto(
                self.accepted_cell_momentum_increment,
                self.trial_cell_momentum_increment)
            self.cp.copyto(
                self.accepted_signed_hi_register,
                self.trial_signed_hi_register)
            self.accepted_register_generation += 1
        self._register_ready = False
        self._signed_hi_register_ready = False
        return result

    def abort_trial(self):
        self._register_ready = False
        self._signed_hi_register_ready = False
        return self.wrapped.abort_trial()

    def invalidate_topology_local_register(self):
        self.accepted_velocity_register.fill(0.0)
        self.accepted_cell_momentum_increment.fill(0.0)
        self.trial_velocity_register.fill(0.0)
        self.trial_cell_momentum_increment.fill(0.0)
        self.accepted_signed_hi_register.fill(0.0)
        self.trial_signed_hi_register.fill(0.0)
        self._register_ready = False
        self._signed_hi_register_ready = False
        self.topology_generation += 1

    def diagnostic(self):
        value = dict(self.wrapped.diagnostic())
        value.update(
            scope_id=SCOPE_ID,
            corrected_depth_weighted_betac=True,
            legacy_v58_betac_injection_calls=self._legacy_injection_calls,
            corrected_betac_injection_calls=self._corrected_injection_calls,
            accepted_betac_register_generation=
                self.accepted_register_generation,
            accepted_signed_hi_register_owner=
                "existing_v58_converged_face_flux_and_BE_diffusion_v1",
            topology_generation=self.topology_generation,
            live_v59_cuda_source_sha256=self.source_sha256,
            automatic_cpu_fallback=False,
        )
        return value

    def checkpoint_fields(self):
        raise ValueError("Q3-H3/v59 live selected restart is not implemented")

    def restore_checkpoint(self, *_args, **_kwargs):
        raise ValueError("Q3-H3/v59 live selected restart is not implemented")


class LiveSelectedBindingV59:
    """Topology contract and accepted-boundary handoff owner for one stepper."""

    def __init__(self, solver, prepared, topology, proxy: CorrectedBetacV59):
        from .amr_prognostic_nonzero_v59_route import selected_mesh_contract
        self.solver = solver
        # A selected AMR transaction may legitimately pass through an
        # all-coarse or all-fine cover.  Completeness, hierarchy projection
        # and 2:1 adjacency remain mandatory; mixed levels are not a physical
        # precondition for the live controller itself.
        self.contract = selected_mesh_contract(
            prepared, topology, require_mixed=False)
        if self.contract.open_face_indices.size:
            raise ValueError(
                "Q3-H3/v59 numerical open-boundary operator is not implemented")
        self.controller = solver.implicit_midpoint
        self.proxy = proxy
        self.last_handoff_receipt = None

    @staticmethod
    def _require_idle(stepper):
        controller = stepper.solver.implicit_midpoint
        if (getattr(controller.component, "_trial_open", False) or
                getattr(controller.owner, "_trial_open", False) or
                stepper.solver._glt_stage is not None):
            raise RuntimeError(
                "Q3-H3/v59 regrid requires an accepted macro boundary")

    def audit_transition(self, new_binding, old_stepper, new_stepper, plan,
                         proposal, primitive_gradients):
        """Audit the generalized signed-hI row produced by the CUDA transfer."""
        from .amr_prognostic_nonzero_v59_route import (
            validate_stable_open_boundary_handoff,
        )
        self._require_idle(old_stepper); self._require_idle(new_stepper)
        validate_stable_open_boundary_handoff(
            self.contract, new_binding.contract)
        cp = old_stepper.cp
        old_rows = old_stepper.full()
        new_rows = proposal.state
        old_index = int(old_stepper.topology.data["secondary_index"])
        new_index = int(new_stepper.topology.data["secondary_index"])
        if old_index != new_index:
            raise ValueError("Q3-H3/v59 signed-hI state row changed")
        old_area = cp.asarray(old_stepper.topology.data["area"])
        new_area = cp.asarray(new_stepper.topology.data["area"])
        before = float(cp.sum(old_area*old_rows[:, old_index]).item())
        after = float(cp.sum(new_area*new_rows[:, new_index]).item())
        scale = max(1.0, abs(before), abs(after))
        if not np.isfinite((before, after)).all() or abs(after-before) > 1e-11*scale:
            raise RuntimeError("Q3-H3/v59 signed-hI handoff is not conservative")

        old_map = {int(node): index for index, node in
                   enumerate(old_stepper.ids)}
        new_map = {int(node): index for index, node in
                   enumerate(new_stepper.ids)}
        for node in plan.retained_ids:
            if not bool(cp.array_equal(
                    old_rows[old_map[node], old_index],
                    new_rows[new_map[node], new_index])):
                raise RuntimeError(
                    "Q3-H3/v59 retained signed-hI row changed")

        tree = old_stepper.prepared.tree
        refinement_bounds = []
        primitive_row = 3 + old_stepper.ng
        for parent, children in zip(
                plan.refine_parents, plan.refine_children):
            parent_row = old_rows[old_map[parent]]
            child_local = cp.asarray([new_map[item] for item in children])
            child = new_rows[child_local]
            depth = cp.asnumpy(child[:, 0])
            signed_hi = cp.asnumpy(child[:, new_index])
            if np.any(depth <= 0.0):
                raise RuntimeError(
                    "Q3-H3/v59 live refinement left fully wet scope")
            parent_depth = float(parent_row[0].item())
            parent_hi = float(parent_row[old_index].item())
            gradient = cp.asnumpy(primitive_gradients[parent])[primitive_row]
            offsets = np.asarray(tree["xy"])[np.asarray(children)] - \
                np.asarray(tree["xy"])[parent]
            raw = parent_hi/parent_depth + offsets @ gradient
            lower = min(float(parent_hi/parent_depth), float(np.min(raw)))
            upper = max(float(parent_hi/parent_depth), float(np.max(raw)))
            intensity = signed_hi/depth
            tolerance = 1e-11*max(1.0, abs(lower), abs(upper))
            if (np.min(intensity) < lower-tolerance or
                    np.max(intensity) > upper+tolerance):
                raise RuntimeError(
                    "Q3-H3/v59 refined signed intensity exceeds directional bounds")
            target = float(np.asarray(tree["area"])[parent])*parent_hi
            inventory = float(np.sum(
                np.asarray(tree["area"])[np.asarray(children)]*signed_hi))
            if abs(inventory-target) > 1e-11*max(1.0, abs(target)):
                raise RuntimeError(
                    "Q3-H3/v59 refined signed-hI family does not close")
            refinement_bounds.append({
                "parent_id": int(parent), "lower": lower, "upper": upper,
                "minimum_child_intensity": float(np.min(intensity)),
                "maximum_child_intensity": float(np.max(intensity)),
                "inventory_residual_m4_s": inventory-target,
            })

        receipt = {
            "schema": 1,
            "scope_id": SCOPE_ID,
            "old_topology_fingerprint": self.contract.fingerprints[
                "topology_fingerprint"],
            "new_topology_fingerprint": new_binding.contract.fingerprints[
                "topology_fingerprint"],
            "old_face_catalog_fingerprint": self.contract.fingerprints[
                "face_catalog_fingerprint"],
            "new_face_catalog_fingerprint": new_binding.contract.fingerprints[
                "face_catalog_fingerprint"],
            "signed_hi_inventory_before_m4_s": before,
            "signed_hi_inventory_after_m4_s": after,
            "signed_hi_inventory_residual_m4_s": after-before,
            "refinement_directional_bounds": refinement_bounds,
            "signed_hi_transfer_owner": "v59_general_selected_handoff_v1",
            "other_state_rows_owner": "existing_selected_cuda_transfer",
            "pending_face_register_at_regrid": False,
            "automatic_cpu_fallback": False,
        }
        self.last_handoff_receipt = receipt
        return receipt

    def synchronize_installed_state(self, stepper):
        """Bind v58 private accepted snapshots to the installed candidate."""
        component = self.controller.component
        component.begin_trial(
            stepper.solver.state["q"], stepper.solver.state["secondary"])
        component.finish_trial(
            stepper.solver.state["q"], stepper.solver.state["secondary"],
            commit=False)

    def transfer_private_accepted_state(self, new_binding):
        """Transfer physical cumulatives; invalidate topology-local caches."""
        old = self.controller
        new = new_binding.controller
        cp = self.proxy.cp
        names = (
            "accepted_stress_boundary_cumulative",
            "accepted_diffusion_boundary_cumulative",
            "accepted_coriolis_cumulative",
        )
        for name in names:
            source = getattr(self.proxy.wrapped, name)
            target = getattr(new_binding.proxy.wrapped, name)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    "Q3-H3/v59 private accepted ledger shape changed: " + name)
            cp.copyto(target, source)
        new._accepted_injections = old._accepted_injections
        new._rotation_calls = old._rotation_calls
        new._last_conservation = old._last_conservation
        new_binding.proxy.wrapped.freeze_generation = \
            self.proxy.wrapped.freeze_generation
        new_binding.proxy.accepted_register_generation = \
            self.proxy.accepted_register_generation
        new_binding.proxy.topology_generation = self.proxy.topology_generation
        signed_register = cp.asnumpy(
            self.proxy.accepted_signed_hi_register)
        betac_register = cp.asnumpy(
            self.proxy.accepted_velocity_register)
        new_binding.proxy.invalidate_topology_local_register()
        return {
            "v59_private_accepted_ledgers": list(names),
            "v59_consumed_signed_hi_register_shape":
                list(signed_register.shape),
            "v59_consumed_signed_hi_register_sha256":
                hashlib.sha256(signed_register.tobytes()).hexdigest(),
            "v59_consumed_betac_register_shape":
                list(betac_register.shape),
            "v59_consumed_betac_register_sha256":
                hashlib.sha256(betac_register.tobytes()).hexdigest(),
            "v59_consumed_register_generation":
                self.proxy.accepted_register_generation,
            "v59_topology_local_face_register":
                "invalidated_zero_not_reinterpreted",
            "v59_legacy_betac_injection_calls":
                new_binding.proxy._legacy_injection_calls,
        }


def _patch_commit_rollback(controller, proxy: CorrectedBetacV59) -> None:
    """Extend the inner v58 publication rollback with v59 accepted arrays."""
    original_snapshot = controller._snapshot_commit_targets
    original_restore = controller._restore_commit_targets

    def snapshot(instance):
        base = original_snapshot()
        return (
            "q3h3-v59", base,
            proxy.accepted_velocity_register.copy(),
            proxy.accepted_cell_momentum_increment.copy(),
            proxy.accepted_signed_hi_register.copy(),
            proxy.accepted_register_generation,
            proxy.topology_generation,
        )

    def restore(instance, saved):
        if (not isinstance(saved, tuple) or len(saved) != 7 or
                saved[0] != "q3h3-v59"):
            raise ValueError("Q3-H3/v59 commit rollback token differs")
        original_restore(saved[1])
        proxy.cp.copyto(proxy.accepted_velocity_register, saved[2])
        proxy.cp.copyto(proxy.accepted_cell_momentum_increment, saved[3])
        proxy.cp.copyto(proxy.accepted_signed_hi_register, saved[4])
        proxy.accepted_register_generation = saved[5]
        proxy.topology_generation = saved[6]

    controller._snapshot_commit_targets = MethodType(snapshot, controller)
    controller._restore_commit_targets = MethodType(restore, controller)


def attach_live_selected_v59(solver, prepared, topology, *, source_path):
    """Attach the additive live v59 owner to one actual selected v58 solver."""
    controller = getattr(solver, "prognostic_secondary_nonzero_v58", None)
    if (controller is None or solver.implicit_midpoint is not controller or
            NONZERO_EXECUTION_KEY not in topology.data):
        raise ValueError(
            "Q3-H3/v59 requires the authenticated nonzero v58 controller")
    proxy = CorrectedBetacV59(controller.component, source_path=source_path)
    controller.component = proxy
    controller.flux_proxy.component = proxy
    original_downstream = controller._downstream

    def downstream(instance, sediment_callback, dt, midpoint_time,
                   weak_state):
        del instance
        result = original_downstream(
            sediment_callback, dt, midpoint_time, weak_state)
        # ``owner.advect`` has already consumed ``base.midpoint_water`` and
        # produced its accepted FCT face flux.  Capture that exact result and
        # the converged BE diffusion flux; do not perform another operator.
        proxy.capture_signed_hi_register(
            dt, controller.owner.transport.buffers["flux"])
        return result

    controller._downstream = MethodType(downstream, controller)
    _patch_commit_rollback(controller, proxy)
    binding = LiveSelectedBindingV59(solver, prepared, topology, proxy)
    solver.prognostic_secondary_amr_live_v59 = binding
    original_receipt = solver.candidate_receipt

    def receipt(instance):
        value = dict(original_receipt())
        value.update(
            prognostic_secondary_amr_live_scope_id=SCOPE_ID,
            prognostic_secondary_amr_live_interface_fingerprint=
                LIVE_SELECTED_INTERFACE["fingerprint"],
            prognostic_secondary_amr_live_cuda_source_sha256=
                proxy.source_sha256,
            corrected_depth_weighted_betac=True,
            legacy_v58_betac_injection=False,
            automatic_cpu_fallback=False,
        )
        return value

    solver.candidate_receipt = MethodType(receipt, solver)
    return solver


def prepare_sources(folder: str | Path, data: dict, *,
                    integrator: str = INTEGRATOR) -> dict[str, object]:
    """Archive the v58 foundation plus the additive live v59 CUDA source."""
    if integrator != INTEGRATOR:
        raise ValueError(
            "Q3-H3/v59 source archive requires implicit_fwave_midpoint")
    from .implicit_midpoint_fwave_nonzero_v58 import prepare_sources as parent
    from .flow2dh_prognostic_nonzero_v58 import validate_scope
    identity = validate_scope(data)
    receipt = parent(folder, data)
    folder = Path(folder).resolve()
    source = Path(__file__).with_name(CUDA_SOURCE_NAME).resolve()
    target = folder / CUDA_SOURCE_NAME
    shutil.copy2(source, target)
    body = {
        "schema": 1,
        "scope_id": SCOPE_ID,
        "integrator": INTEGRATOR,
        "execution_fingerprint": identity["fingerprint"],
        "live_interface_fingerprint": LIVE_SELECTED_INTERFACE["fingerprint"],
        "cuda_source": CUDA_SOURCE_NAME,
        "cuda_source_sha256": _sha(target),
        "automatic_cpu_fallback": False,
    }
    manifest = {**body, "fingerprint": _digest(body)}
    (folder / SOURCE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**receipt, "q3h3_amr_live_v59": manifest}


@contextmanager
def constructors(folder: str | Path, prepared, topology):
    """Yield one selected v59 solver constructor; never fall back to v58."""
    from .amr_prognostic_nonzero_v59_route import composition_readiness
    from .implicit_midpoint_fwave_nonzero_v58 import constructors as parent
    report = composition_readiness(prepared, topology)
    if not report["live_route_admitted"]:
        from .amr_prognostic_nonzero_v59_route import (
            LiveAMRCompositionUnavailable,
        )
        raise LiveAMRCompositionUnavailable(report)
    folder = Path(folder).resolve()
    manifest_path = folder / SOURCE_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Q3-H3/v59 live source manifest is missing") from error
    source = folder / CUDA_SOURCE_NAME
    body = {key: value for key, value in manifest.items()
            if key != "fingerprint"}
    identity = report["live_interface"]
    if (set(manifest) != {
            "schema", "scope_id", "integrator", "execution_fingerprint",
            "live_interface_fingerprint", "cuda_source",
            "cuda_source_sha256", "automatic_cpu_fallback", "fingerprint"} or
            manifest.get("schema") != 1 or
            manifest.get("scope_id") != SCOPE_ID or
            manifest.get("integrator") != INTEGRATOR or
            manifest.get("live_interface_fingerprint") != identity["fingerprint"] or
            manifest.get("cuda_source") != CUDA_SOURCE_NAME or
            not source.is_file() or source.is_symlink() or
            _sha(source) != manifest.get("cuda_source_sha256") or
            manifest.get("automatic_cpu_fallback") is not False or
            manifest.get("fingerprint") != _digest(body)):
        raise ValueError("Q3-H3/v59 archived source identity differs")
    with parent(folder, topology.data) as Parent:
        def Solver(actual, *, source_folder=folder):
            if (actual is not topology.data or
                    Path(source_folder).resolve() != folder):
                raise ValueError("Q3-H3/v59 selected constructor deck differs")
            solver = Parent(actual)
            return attach_live_selected_v59(
                solver, prepared, topology, source_path=source)
        yield Solver


__all__ = [
    "CUDA_SOURCE_NAME", "CorrectedBetacV59", "INTEGRATOR",
    "KERNEL_NAMES", "LIVE_CAPABILITIES", "LIVE_SELECTED_INTERFACE",
    "LiveSelectedBindingV59", "SCOPE_ID", "SOURCE_MANIFEST_NAME",
    "attach_live_selected_v59", "constructors", "prepare_sources",
]
