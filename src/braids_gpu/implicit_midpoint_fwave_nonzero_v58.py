"""Additive Q3-H3/v58 composition for the accepted implicit f-wave solver.

The accepted v57 controller and CUDA sources remain immutable.  This module
constructs that controller from a zero-operator foundation view, then wraps it
with one authenticated v58 transaction which owns nonzero Coriolis, lagged
``Betac`` momentum feedback and backward-Euler signed-hI diffusion.

This is a bounded fixed-Cartesian, fully-wet route.  It does not silently fall
back to CPU and it does not alter the QGIS/production dispatcher.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
from types import MethodType

import numpy as np

from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY,
    METHOD_ID,
    MODEL,
    ORDERED_ROUTE_KEY,
    prognostic_secondary_execution_document,
    validate_prognostic_secondary_execution_document,
)
from braids_core.sediment.prognostic_secondary_nonzero_execution import (
    CHECKPOINT_SCHEMA,
    EXECUTION_KEY as NONZERO_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as NONZERO_ORDERED_ROUTE_KEY,
    TRANSACTION_ORDER,
    validate_prognostic_secondary_nonzero_execution_document,
    validate_prognostic_secondary_nonzero_ordered_route,
)
from .flow2dh_prognostic_nonzero_v58 import (
    INTEGRATOR,
    STATUS_OK,
    ResidentClassicFlow2DHPrognosticNonzeroV58,
    validate_scope,
)
from .implicit_midpoint_fwave import (
    ImplicitStepCancelled,
    ImplicitStepRejected,
)


SCOPE_ID = "q3h3-classic-flow2dh-prognostic-nonzero-live-gpu-v1"
SOURCE_NAME = "flow2dh_prognostic_nonzero_v58.cu"
CHECKPOINT_PREFIX = "q3h3_v58_"


def _neutral_identity(parent: dict[str, object]) -> dict[str, object]:
    """Rebuild only the accepted zero-operator v57 construction identity."""
    value = validate_prognostic_secondary_execution_document(parent)
    return prognostic_secondary_execution_document(
        method_id=METHOD_ID,
        model=MODEL,
        capacity_law=int(value["capacity_law"]),
        capacity_execution_fingerprint=value[
            "capacity_execution_fingerprint"],
        capacity_profile_fingerprint=value["capacity_profile_fingerprint"],
        hiding_execution_fingerprint=value["hiding_execution_fingerprint"],
        roughness_method_id=value["roughness_method_id"],
        roughness_fingerprint=value["roughness_fingerprint"],
        fraction_ids=tuple(value["ordered_fraction_ids"]),
        bed_layer_count=int(value["runtime_layout"]["bed_layer_count"]),
        transport_direction=bool(value["transport_direction"]),
        inflow_mode=int(value["inflow_mode"]),
        coriolis_disposition="zero",
        coriolis_constant_s_inverse=0.0,
        coriolis_source="constant_zero_planar",
        rmincf=float(value["rmincf"]),
        chzmin=float(value["chzmin"]),
        espir=float(value["espir"]),
        betac=0.0,
        scalar_diffusivity_m2_s=0.0,
        von_karman_constant=float(value["von_karman_constant"]),
        gpu_checkpoint_schema=int(value["gpu_checkpoint_schema"]),
    )


def neutral_foundation_view(data: dict) -> dict:
    """Return an explicit v57 constructor view; never mutate the v58 deck."""
    validate_scope(data)
    parent = validate_prognostic_secondary_execution_document(
        data[EXECUTION_KEY])
    neutral_identity = _neutral_identity(parent)
    neutral = dict(data)
    neutral[EXECUTION_KEY] = neutral_identity
    neutral[ORDERED_ROUTE_KEY] = {
        "capacity_execution_fingerprint":
            neutral_identity["capacity_execution_fingerprint"],
        "capacity_profile_fingerprint":
            neutral_identity["capacity_profile_fingerprint"],
        "hiding_execution_fingerprint":
            neutral_identity["hiding_execution_fingerprint"],
        "prognostic_secondary_execution_fingerprint":
            neutral_identity["fingerprint"],
    }
    neutral["secondary_flow_momentum_beta"] = np.float64(0.0)
    neutral.pop(NONZERO_EXECUTION_KEY, None)
    neutral.pop(NONZERO_ORDERED_ROUTE_KEY, None)
    return neutral


def _same_prepared_value(left, right):
    """Exact recursive equality for the host-side prepared deck binding."""
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                set(left) == set(right) and
                all(_same_prepared_value(left[key], right[key])
                    for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_same_prepared_value(a, b)
                    for a, b in zip(left, right)))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        return (a.shape == b.shape and a.dtype == b.dtype and
                np.array_equal(a, b, equal_nan=False))
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        a, b = np.asarray(left), np.asarray(right)
        return (a.shape == b.shape and a.dtype == b.dtype and
                bool(a == b))
    return type(left) is type(right) and left == right


class _FrozenBetacFluxProxy:
    """Inject one immutable accepted-start face field into every residual."""

    def __init__(self, wrapped, component):
        self.wrapped = wrapped
        self.component = component
        self.active = False
        self.injections = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def step(self, *args, **kwargs):
        result = self.wrapped.step(*args, **kwargs)
        if self.active:
            flux, _ = result
            self.component.add_frozen_stress(flux["fl"], flux["fr"])
            self.injections += 1
        return result


class IntegratedNonzeroV58Controller:
    """Compose all v58 state, physics and ledgers in one macro transaction."""

    def __init__(self, solver, actual_data, *, source_path=None):
        self.solver = solver
        self.base = solver.implicit_midpoint
        if self.base is None or not self.base.prognostic_secondary_composition:
            raise ValueError(
                "Q3-H3/v58 requires the accepted prognostic implicit owner")
        self.cp = self.base.cp
        self.identity = validate_scope(actual_data)
        self.parent_identity = validate_prognostic_secondary_execution_document(
            actual_data[EXECUTION_KEY])
        validate_prognostic_secondary_nonzero_execution_document(
            actual_data[NONZERO_EXECUTION_KEY], parent=self.parent_identity)
        validate_prognostic_secondary_nonzero_ordered_route(
            actual_data[NONZERO_ORDERED_ROUTE_KEY], self.identity)
        expected_neutral = _neutral_identity(self.parent_identity)
        actual_neutral = validate_prognostic_secondary_execution_document(
            solver.data[EXECUTION_KEY])
        if actual_neutral != expected_neutral:
            raise ValueError("Q3-H3/v58 neutral v57 construction view differs")
        expected_deck = neutral_foundation_view(actual_data)
        normalized_extras = set(solver.data) - set(expected_deck)
        if (set(expected_deck) - set(solver.data) or
                normalized_extras != {
                    "morfac", "secondary_mass", "suspended_mass"} or
                any(not _same_prepared_value(
                    solver.data[key], expected_deck[key])
                    for key in expected_deck) or
                not _same_prepared_value(
                    solver.data["morfac"],
                    expected_deck["morphological_factor"]) or
                not _same_prepared_value(
                    solver.data["secondary_mass"],
                    np.ascontiguousarray(expected_deck["state"][
                        int(expected_deck["secondary_index"])])) or
                not _same_prepared_value(
                    solver.data["suspended_mass"],
                    np.ascontiguousarray(expected_deck["state"][
                        np.asarray(expected_deck[
                            "suspended_indices"]).astype(int)]))):
            raise ValueError(
                "Q3-H3/v58 hydraulic and additive prepared decks differ")
        self.component = ResidentClassicFlow2DHPrognosticNonzeroV58(
            actual_data, source_path=source_path)
        if (self.component.device_id != int(self.cp.cuda.Device().id) or
                self.component.nc != solver.nc or
                self.component.nf != solver.nf or
                self.component.nb != solver.nb):
            raise ValueError("Q3-H3/v58 resident owner topology differs")
        self.owner = solver.prognostic_secondary
        if (self.owner is None or self.owner.cp is not self.cp or
                self.owner.device_id != self.component.device_id):
            raise ValueError("Q3-H3/v58 accepted signed-hI owner differs")
        self.zero_branch = bool(
            float(self.identity["coriolis_constant_s_inverse"]) == 0.0 and
            float(self.identity["betac"]) == 0.0 and
            float(self.identity["scalar_diffusivity_m2_s"]) == 0.0)

        # The accepted v57 object was legally constructed with f=0.  Only the
        # additive v58 wrapper owns and records this mutation of its resident
        # point-input field; accepted v57 source and identity stay unchanged.
        self.owner.classic.coriolis.fill(
            np.float64(self.identity["coriolis_constant_s_inverse"]))
        self.owner.classic.coriolis_value = float(
            self.identity["coriolis_constant_s_inverse"])
        self.flux_proxy = _FrozenBetacFluxProxy(solver.flux, self.component)
        self.combined_secondary_ledger = self.cp.zeros_like(
            solver.ledger["secondary"])
        self.betac_boundary_trace = self.cp.empty(
            (3, solver.nf), dtype=self.cp.float64)
        self.diffusion_boundary_depth = self.cp.empty(
            solver.nf, dtype=self.cp.float64)
        self.source_integral = self.cp.zeros((), dtype=self.cp.float64)
        self._sediment_result = None
        self._post_friction_applied = False
        self._rotation_calls = 0
        self._accepted_injections = 0
        self._last_conservation = 0.0
        self._backup_q = self.cp.empty_like(solver.state["q"])
        self._backup_layers = self.cp.empty_like(solver.state["layers"])
        self._backup_secondary = self.cp.empty_like(
            solver.state["secondary"])
        self._backup_ledgers = {
            key: self.cp.empty_like(value)
            for key, value in solver.ledger.items()}
        self._backup_predictor = self.cp.empty_like(
            self.base.predictor_rate)
        self._backup_component_q = self.cp.empty_like(
            self.component.accepted_q_snapshot)
        self._backup_component_hi = self.cp.empty_like(
            self.component.accepted_hi_snapshot)
        self._backup_component_stress = self.cp.empty_like(
            self.component.accepted_stress_boundary_cumulative)
        self._backup_component_coriolis = self.cp.empty_like(
            self.component.accepted_coriolis_cumulative)
        self._backup_component_diffusion = self.cp.empty_like(
            self.component.accepted_diffusion_boundary_cumulative)

        # Establish a checkpointable committed baseline without changing the
        # accepted arrays or ledgers.
        self.component.begin_trial(
            solver.state["q"], solver.state["secondary"])
        self.component.finish_trial(
            solver.state["q"], solver.state["secondary"], commit=False)

    @property
    def have_predictor(self):
        return self.base.have_predictor

    @have_predictor.setter
    def have_predictor(self, value):
        self.base.have_predictor = value

    @property
    def last_diagnostics(self):
        return self.base.last_diagnostics

    @last_diagnostics.setter
    def last_diagnostics(self, value):
        self.base.last_diagnostics = value

    @property
    def accepted_steps(self):
        return self.base.accepted_steps

    @accepted_steps.setter
    def accepted_steps(self, value):
        self.base.accepted_steps = value

    @property
    def rejected_steps(self):
        return self.base.rejected_steps

    @rejected_steps.setter
    def rejected_steps(self, value):
        self.base.rejected_steps = value

    @property
    def fwave_evaluations(self):
        return self.base.fwave_evaluations

    @fwave_evaluations.setter
    def fwave_evaluations(self, value):
        self.base.fwave_evaluations = value

    @property
    def linear_matvecs(self):
        return self.base.linear_matvecs

    @linear_matvecs.setter
    def linear_matvecs(self, value):
        self.base.linear_matvecs = value

    def bind_cancelled(self, callback):
        return self.base.bind_cancelled(callback)

    def _all_clear(self, *values):
        return all(not bool(self.cp.any(value != 0).item())
                   for value in values if value is not None and value.size)

    def _commit_array(self, name, value, reference):
        cp = self.cp
        if (not isinstance(value, cp.ndarray) or
                value.dtype != reference.dtype or
                value.shape != reference.shape or
                not value.flags.c_contiguous or
                int(value.device.id) != self.component.device_id or
                not bool(cp.isfinite(value).all().item())):
            raise ImplicitStepRejected(
                "Q3-H3/v58 malformed commit output: " + name)
        return value

    def _snapshot_commit_targets(self):
        cp, s, c, base = self.cp, self.solver, self.component, self.base
        cp.copyto(self._backup_q, s.state["q"])
        cp.copyto(self._backup_layers, s.state["layers"])
        cp.copyto(self._backup_secondary, s.state["secondary"])
        for key, value in s.ledger.items():
            cp.copyto(self._backup_ledgers[key], value)
        cp.copyto(self._backup_predictor, base.predictor_rate)
        cp.copyto(self._backup_component_q, c.accepted_q_snapshot)
        cp.copyto(self._backup_component_hi, c.accepted_hi_snapshot)
        cp.copyto(self._backup_component_stress,
                  c.accepted_stress_boundary_cumulative)
        cp.copyto(self._backup_component_coriolis,
                  c.accepted_coriolis_cumulative)
        cp.copyto(self._backup_component_diffusion,
                  c.accepted_diffusion_boundary_cumulative)
        return (base.have_predictor, base.accepted_steps,
                c._have_accepted_state, self._accepted_injections)

    def _restore_commit_targets(self, scalars):
        cp, s, c, base = self.cp, self.solver, self.component, self.base
        cp.copyto(s.state["q"], self._backup_q)
        cp.copyto(s.state["layers"], self._backup_layers)
        cp.copyto(s.state["secondary"], self._backup_secondary)
        for key, value in s.ledger.items():
            cp.copyto(value, self._backup_ledgers[key])
        cp.copyto(base.predictor_rate, self._backup_predictor)
        cp.copyto(c.accepted_q_snapshot, self._backup_component_q)
        cp.copyto(c.accepted_hi_snapshot, self._backup_component_hi)
        cp.copyto(c.accepted_stress_boundary_cumulative,
                  self._backup_component_stress)
        cp.copyto(c.accepted_coriolis_cumulative,
                  self._backup_component_coriolis)
        cp.copyto(c.accepted_diffusion_boundary_cumulative,
                  self._backup_component_diffusion)
        (base.have_predictor, base.accepted_steps,
         c._have_accepted_state, self._accepted_injections) = scalars

    def _freeze_accepted_betac(self):
        s, cp = self.solver, self.cp
        # The accepted boundary implementation authenticates the exact stage
        # allocation by pointer.  Use that private allocation as a byte copy
        # of q^n for the freeze; the component already owns the independent
        # accepted snapshot used by its tensor kernels.
        cp.copyto(self.base.midpoint, s.state["q"])
        reconstructed = s.reconstruction.reconstruct(self.base.midpoint)
        s.boundary.apply(self.base.midpoint, reconstructed["trace"])
        active = reconstructed["active"]
        classic = self.owner.classic
        classic.refresh(self.base.midpoint, active)
        if classic.nf:
            with cp.cuda.Device(self.component.device_id):
                classic.roughness.kernels["hydraulic_roughness_state"](
                    ((classic.nf + 255) // 256,), (256,), (
                        np.int64(classic.nf),
                        np.int32(classic.roughness_mode),
                        reconstructed["trace"][0],
                        classic.boundary_roughness_parameter,
                        np.float64(classic.gravity),
                        np.float64(classic.roughness_work_depth),
                        np.int32(classic.roughness_waqua),
                        classic.boundary_raw_chezy,
                        classic.boundary_roughness_z0,
                        classic.boundary_roughness_drag,
                        classic.boundary_roughness_status))
        statuses = (
            reconstructed["primitive_status"],
            reconstructed["reconstruction_status"],
            s.boundary.status, s.boundary.face_status, classic.status,
            classic.boundary_roughness_status,
        )
        if not self._all_clear(*statuses):
            raise ImplicitStepRejected(
                "Q3-H3/v58 accepted-start Betac inputs are invalid")
        cp.copyto(self.betac_boundary_trace, reconstructed["trace"][:3])
        frozen = self.component.freeze_betac(
            classic.raw_chezy, self.betac_boundary_trace,
            classic.result.boundary_equilibrium,
            classic.boundary_raw_chezy, active)
        if frozen["status"] != STATUS_OK:
            raise ImplicitStepRejected(
                "Q3-H3/v58 accepted-start Betac freeze failed")

    def _post_friction_once(self, original, q, half_dt, phase):
        # v58 binds friction after the hydraulic flux trial.  Suppress the
        # accepted controller's pre-half and execute one full endpoint solve
        # from _downstream; suppress its later generic post-half duplicate.
        if phase == "Pre-midpoint":
            return None
        if phase == "Post-midpoint" and self._post_friction_applied:
            return None
        result = original(q, half_dt, phase)
        if phase == "Post-midpoint":
            self._post_friction_applied = True
        return result

    def _downstream(self, ignored_midpoint, dt, midpoint_time,
                    sediment_callback):
        """Run the declared endpoint secondary chain, then one Exner leg."""
        del ignored_midpoint
        cp, s, base = self.cp, self.solver, self.base
        # This explicit invocation is intercepted by _post_friction_once; the
        # accepted controller's later generic call becomes an exact no-op.
        base._friction_half_step(base.qnew, dt, "Post-midpoint")
        if float(cp.min(base.qnew[0]).item()) <= base.wet_floor:
            raise ImplicitStepRejected(
                "Q3-H3/v58 post-midpoint friction wet-floor failed")
        self.component.load_hydraulic_endpoint(base.qnew)
        rotated = self.component.apply_coriolis(dt)
        if rotated["status"] != STATUS_OK:
            raise ImplicitStepRejected("Q3-H3/v58 Coriolis rotation failed")
        cp.copyto(base.qnew, self.component.q_trial)

        accepted_bed = s.bed(s.state["layers"])
        accepted_reconstruction = s.reconstruction.reconstruct(
            self.component.accepted_q_snapshot)
        advanced = self.owner.advect(
            self.component.accepted_q_snapshot, self.component.q_trial,
            accepted_bed, base.midpoint_water, dt,
            accepted_reconstruction["active"],
            hydraulic_buffers=accepted_reconstruction)
        if int(advanced["status"].item()) != 0:
            raise ImplicitStepRejected(
                "Q3-H3/v58 signed-hI advection failed")
        cp.copyto(self.component.hi_trial, self.owner.trial)

        # FLOW's open-face scalar conductance owns hydraulic trace depth.  The
        # accepted boundary object authenticates the exact stage allocation,
        # so use that private allocation for an endpoint copy and reset its
        # record cursor afterwards for the one sediment/Exner evaluation.
        cp.copyto(base.midpoint, self.component.q_trial)
        endpoint_reconstruction = s.reconstruction.reconstruct(base.midpoint)
        s.boundary.apply(base.midpoint, endpoint_reconstruction["trace"])
        if not self._all_clear(
                endpoint_reconstruction["primitive_status"],
                endpoint_reconstruction["reconstruction_status"],
                s.boundary.status, s.boundary.face_status):
            raise ImplicitStepRejected(
                "Q3-H3/v58 endpoint hydraulic trace failed")
        cp.copyto(self.diffusion_boundary_depth,
                  endpoint_reconstruction["trace"][0])
        s.boundary.begin_trial()
        diffused = self.component.diffuse(
            self.component.q_trial[0],
            self.owner.classic.result.boundary_equilibrium, dt,
            boundary_depth=self.diffusion_boundary_depth)
        if diffused["status"] != STATUS_OK:
            raise ImplicitStepRejected(
                "Q3-H3/v58 signed-hI diffusion failed")
        cp.copyto(self.owner.trial, self.component.hi_trial)
        reacted = self.owner._reaction(
            self.component.accepted_q_snapshot, self.component.q_trial,
            dt, accepted_reconstruction["active"],
            refresh_coefficients=False)
        if int(reacted["status"].item()) != 0:
            raise ImplicitStepRejected(
                "Q3-H3/v58 signed-hI reaction failed")
        cp.copyto(self.component.hi_trial, self.owner.trial)
        self.component.accept_stress_ledger(dt)

        cp.add(self.owner.boundary_ledger,
               self.component.diffusion_boundary_work,
               out=self.combined_secondary_ledger)
        cp.sum(self.owner.source_ledger_cells, out=self.source_integral)
        before = cp.sum(
            self.component.area * self.component.accepted_hi_snapshot)
        after = cp.sum(self.component.area * self.owner.trial)
        boundary = cp.sum(self.combined_secondary_ledger)
        residual = float((after - before - boundary -
                          self.source_integral).item())
        scale = max(1.0, abs(float(before.item())),
                    abs(float(after.item())), abs(float(boundary.item())),
                    abs(float(self.source_integral.item())))
        if (not np.isfinite(residual) or
                abs(residual) > max(1.0e-12, 1.0e-11 * scale)):
            raise ImplicitStepRejected(
                "Q3-H3/v58 combined signed-hI ledger is not conservative")
        self._last_conservation = residual
        base._check_cancelled()

        # The accepted endpoint—not the pre-Coriolis hydraulic midpoint—is
        # the capacity state. The v57 owner phase enforces exactly one
        # transported-I rotation if a sediment/Exner callback is present.
        self.flux_proxy.active = False
        if sediment_callback is None:
            self._sediment_result = None
            return None
        before_phase = self.owner._phase
        sediment = sediment_callback(
            self.component.q_trial, dt, midpoint_time,
            secondary=self.owner.trial)
        if before_phase != "reacted" or self.owner._phase != "rotated":
            raise ImplicitStepRejected(
                "Q3-H3/v58 did not execute exactly one bed rotation")
        self._sediment_result = sediment
        return sediment

    def checkpoint_fields(self):
        with self.cp.cuda.Device(self.component.device_id):
            return self._checkpoint_fields_on_selected_device()

    def _checkpoint_fields_on_selected_device(self):
        result = dict(self.base.checkpoint_fields())
        component = self.component.checkpoint_fields()
        for key, value in component.items():
            result[CHECKPOINT_PREFIX + key] = value
        return result

    def restore_checkpoint(self, fields):
        with self.cp.cuda.Device(self.component.device_id):
            return self._restore_checkpoint_on_selected_device(fields)

    def _restore_checkpoint_on_selected_device(self, fields):
        component_names = (
            "schema", "scope_id", "execution_fingerprint",
            "cuda_source_sha256", "mesh_fingerprint", "q", "h_i",
            "diffusion_boundary_cumulative_m4_s",
            "betac_momentum_boundary_cumulative_m4_s",
            "coriolis_increment_cumulative_m4_s",
        )
        try:
            component = {
                key: fields[CHECKPOINT_PREFIX + key]
                for key in component_names
            }
            rate = np.asarray(fields["implicit_predictor_rate"])
            flag = np.asarray(fields["implicit_have_predictor"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Q3-H3/v58 checkpoint inventory differs") from error
        if (rate.shape != (3, self.component.nc) or
                rate.dtype != np.float64 or not np.isfinite(rate).all() or
                flag.shape != () or flag.dtype != np.int32 or
                int(flag) not in (0, 1)):
            raise ValueError("Q3-H3/v58 implicit checkpoint differs")
        host = self.component.validate_checkpoint_fields(component)
        # q/hI are also owned by the outer segmented checkpoint. Require exact
        # equality before any mutation; never let the component silently
        # overwrite a mismatched main-state restore.
        if (not np.array_equal(
                self.cp.asnumpy(self.solver.state["q"]), host["q"]) or
                not np.array_equal(
                    self.cp.asnumpy(self.solver.state["secondary"]),
                    host["h_i"])):
            raise ValueError(
                "Q3-H3/v58 duplicate checkpoint q/hI differ")
        # Every field is now authenticated. These two restores cannot fail on
        # semantic input after the validation barrier above.
        self.base.restore_checkpoint(fields)
        self.component.restore_checkpoint(
            component, self.solver.state["q"],
            self.solver.state["secondary"])

    def step(self, dt, time, *, sediment_callback=None, commit=True):
        with self.cp.cuda.Device(self.component.device_id):
            return self._step_on_selected_device(
                dt, time, sediment_callback=sediment_callback,
                commit=commit)

    def _step_on_selected_device(
            self, dt, time, *, sediment_callback=None, commit=True):
        cp, s, base = self.cp, self.solver, self.base
        if (not np.isfinite((dt, time)).all() or dt <= 0.0 or
                not isinstance(commit, (bool, np.bool_))):
            raise ValueError("Q3-H3/v58 timestep/commit control differs")
        # The all-zero outer declaration is an exact dispatch to the accepted
        # v57 controller.  No proxy, source split, reordered capacity state or
        # additional arithmetic is reachable on this parity branch.
        if self.zero_branch:
            result = base.step(
                dt, time, sediment_callback=sediment_callback, commit=commit)
            if commit and int(result["status"].item()) == 0:
                self.component.begin_trial(
                    s.state["q"], s.state["secondary"])
                self.component.finish_trial(
                    s.state["q"], s.state["secondary"], commit=False)
            return result
        component_open = owner_open = False
        published_targets = None
        original_flux = s.flux
        original_friction = base._friction_half_step
        original_composition = base.prognostic_secondary_composition
        initial_injections = self.flux_proxy.injections
        initial_fwave = base.fwave_evaluations
        self._sediment_result = None
        self._post_friction_applied = False
        try:
            base._check_cancelled()
            self.component.begin_trial(
                s.state["q"], s.state["secondary"])
            component_open = True
            self.owner.begin_trial(s.state["secondary"])
            owner_open = True
            self._freeze_accepted_betac()
            self.flux_proxy.active = True
            s.flux = self.flux_proxy
            base.prognostic_secondary_composition = False
            base._friction_half_step = MethodType(
                lambda _ignored, q, half_dt, phase:
                    self._post_friction_once(
                        original_friction, q, half_dt, phase),
                base)

            result = base.step(
                dt, time,
                sediment_callback=lambda midpoint, macro_dt, midpoint_time:
                    self._downstream(
                        midpoint, macro_dt, midpoint_time,
                        sediment_callback),
                commit=False)
            self.flux_proxy.active = False
            if int(result["status"].item()) != 0:
                self.component.abort_trial(); component_open = False
                self.owner.abort_trial(); owner_open = False
                return result
            injections = self.flux_proxy.injections - initial_injections
            evaluations = base.fwave_evaluations - initial_fwave
            if injections != evaluations or injections < 1:
                raise ImplicitStepRejected(
                    "Q3-H3/v58 Betac feedback missed an implicit residual")
            base._check_cancelled()

            sediment = self._sediment_result
            try:
                if (sediment is not None and
                        int(sediment["status"].item()) != 0):
                    raise ImplicitStepRejected(
                        "Q3-H3/v58 downstream sediment/Exner leg failed")
                trial_layers = (
                    sediment["state"]["layers"]
                    if sediment is not None else s.state["layers"])
                bed_ledger = (
                    sediment["ledger"]["bed"]
                    if sediment is not None else s.zero_bed_ledger)
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                raise ImplicitStepRejected(
                    "Q3-H3/v58 malformed downstream transaction") from error
            self._commit_array("hydraulics", self.component.q_trial,
                               s.state["q"])
            self._commit_array("signed hI", self.component.hi_trial,
                               s.state["secondary"])
            self._commit_array("layers", trial_layers, s.state["layers"])
            self._commit_array("water ledger", base.water_ledger,
                               s.ledger["water"])
            self._commit_array("bed ledger", bed_ledger,
                               s.ledger["bed"])
            self._commit_array("secondary ledger",
                               self.combined_secondary_ledger,
                               s.ledger["secondary"])
            self._commit_array("source ledger", self.source_integral,
                               s.ledger["source"])
            self._commit_array("implicit predictor rate", base.rate,
                               base.predictor_rate)
            if commit:
                saved_targets = self._snapshot_commit_targets()
                published_targets = saved_targets
                try:
                    self.component.finish_trial(
                        s.state["q"], s.state["secondary"], commit=True)
                    component_open = False
                    self.owner.finish_trial(
                        s.state["secondary"], commit=False)
                    owner_open = False
                    cp.copyto(s.state["layers"], trial_layers)
                    s.ledger["water"] += base.water_ledger
                    s.ledger["bed"] += bed_ledger
                    s.ledger["secondary"] += self.combined_secondary_ledger
                    s.ledger["source"] += self.source_integral
                    cp.copyto(base.predictor_rate, base.rate)
                    base.have_predictor = True
                    base.accepted_steps += 1
                    self._accepted_injections += injections
                except Exception:
                    self._restore_commit_targets(saved_targets)
                    published_targets = None
                    self.component.abort_trial()
                    component_open = False
                    self.owner.abort_trial()
                    owner_open = False
                    raise
            else:
                self.component.finish_trial(
                    s.state["q"], s.state["secondary"], commit=False)
                component_open = False
                self.owner.finish_trial(
                    s.state["secondary"], commit=False)
                owner_open = False
            diagnostics = dict(result.get("diagnostics", {}))
            diagnostics.update(
                component_coupling_schema=12,
                prognostic_secondary_nonzero_checkpoint_schema=
                    CHECKPOINT_SCHEMA,
                prognostic_secondary_nonzero_execution_fingerprint=
                    self.identity["fingerprint"],
                prognostic_secondary_nonzero_transaction_order=list(
                    TRANSACTION_ORDER),
                betac_residual_injections=injections,
                bed_rotation_calls=(1 if sediment is not None else 0),
                secondary_advective_boundary_integral=float(
                    cp.sum(self.owner.boundary_ledger).item()),
                secondary_diffusive_boundary_integral=float(
                    cp.sum(self.component.diffusion_boundary_work).item()),
                secondary_source_integral=float(self.source_integral.item()),
                secondary_inventory_residual=self._last_conservation,
                automatic_cpu_fallback=False,
                actual_cuda_device_id=self.component.device_id,
            )
            base.last_diagnostics = diagnostics
            result["diagnostics"] = diagnostics
            # Publication is now complete, including every operation that can
            # still raise while constructing the accepted-step diagnostics.
            if commit and sediment is not None:
                self._rotation_calls += 1
            published_targets = None
            return result
        except (ImplicitStepCancelled, ImplicitStepRejected) as error:
            if published_targets is not None:
                self._restore_commit_targets(published_targets)
                published_targets = None
            if component_open:
                self.component.abort_trial(); component_open = False
            if owner_open:
                self.owner.abort_trial(); owner_open = False
            base.status.fill(1)
            base.metric.fill(max(float(s.data["cfl_target"]), 1.0))
            base.rejected_steps += 1
            base.last_diagnostics = {
                "cancelled": isinstance(error, ImplicitStepCancelled),
                "error": str(error),
                "component_coupling_schema": 12,
            }
            return {"status": base.status, "cfl": base.metric,
                    "diagnostics": base.last_diagnostics}
        except Exception:
            if published_targets is not None:
                self._restore_commit_targets(published_targets)
                published_targets = None
            if component_open:
                self.component.abort_trial()
            if owner_open:
                self.owner.abort_trial()
            raise
        finally:
            self.flux_proxy.active = False
            s.flux = original_flux
            base._friction_half_step = original_friction
            base.prognostic_secondary_composition = original_composition

    def diagnostic(self):
        return {
            "scope_id": SCOPE_ID,
            "execution_fingerprint": self.identity["fingerprint"],
            "integrator": INTEGRATOR,
            "automatic_cpu_fallback": False,
            "device_id": self.component.device_id,
            "accepted_residual_injections": self._accepted_injections,
            "bed_rotation_calls": self._rotation_calls,
            "component": self.component.diagnostic(),
        }


def attach_nonzero_v58(solver, actual_data, *, source_path=None):
    """Replace only the live instance controller with the additive wrapper."""
    controller = IntegratedNonzeroV58Controller(
        solver, actual_data, source_path=source_path)
    solver.implicit_midpoint = controller
    solver.prognostic_secondary_nonzero_v58 = controller
    original_step = solver.step
    original_receipt = solver.candidate_receipt

    def selected_device_step(instance, *args, **kwargs):
        del instance
        with controller.cp.cuda.Device(controller.component.device_id):
            return original_step(*args, **kwargs)

    def receipt(instance):
        value = dict(original_receipt())
        value.update(
            prognostic_secondary_nonzero_execution_fingerprint=
                controller.identity["fingerprint"],
            prognostic_secondary_nonzero_checkpoint_schema=CHECKPOINT_SCHEMA,
            prognostic_secondary_nonzero_transaction_order=list(
                TRANSACTION_ORDER),
            prognostic_secondary_nonzero_backend="gpu",
            prognostic_secondary_nonzero_integrator=INTEGRATOR,
            prognostic_secondary_nonzero_automatic_cpu_fallback=False,
            prognostic_secondary_nonzero_cuda_source_sha256=
                controller.component.source_sha256,
        )
        return value

    solver.candidate_receipt = MethodType(receipt, solver)
    solver.step = MethodType(selected_device_step, solver)
    return solver


def prepare_sources(folder, data):
    """Archive the accepted foundation plus the additive v58 CUDA source."""
    from . import general_live_transport

    actual_identity = validate_scope(data)
    neutral = neutral_foundation_view(data)
    receipt = general_live_transport.prepare_sources(
        folder, neutral, integrator=INTEGRATOR)
    folder = Path(folder).resolve()
    source = Path(__file__).with_name(SOURCE_NAME).resolve()
    target = folder / SOURCE_NAME
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    additive = {
        "schema": 1,
        "scope_id": SCOPE_ID,
        "execution_fingerprint": actual_identity["fingerprint"],
        "integrator": INTEGRATOR,
        "automatic_cpu_fallback": False,
        "cuda_source": SOURCE_NAME,
        "cuda_source_sha256": digest,
        "foundation_receipt_sha256": hashlib.sha256(
            (folder / "general_live_transport_sources.json").read_bytes()
        ).hexdigest(),
    }
    (folder / "q3h3_nonzero_v58_sources.json").write_text(
        json.dumps(additive, indent=2) + "\n", encoding="utf-8")
    return {**receipt, "q3h3_nonzero_v58": additive}


@contextmanager
def constructors(folder, data):
    """Yield one actual-v58 constructor without changing v57 factories."""
    from . import general_live_transport

    identity = validate_scope(data)
    neutral = neutral_foundation_view(data)
    folder = Path(folder).resolve()
    manifest = json.loads(
        (folder / "q3h3_nonzero_v58_sources.json").read_text(
            encoding="utf-8"))
    source = folder / SOURCE_NAME
    if (manifest.get("schema") != 1 or
            manifest.get("scope_id") != SCOPE_ID or
            manifest.get("execution_fingerprint") != identity["fingerprint"] or
            manifest.get("integrator") != INTEGRATOR or
            manifest.get("automatic_cpu_fallback") is not False or
            manifest.get("cuda_source") != SOURCE_NAME or
            hashlib.sha256(source.read_bytes()).hexdigest() !=
                manifest.get("cuda_source_sha256")):
        raise ValueError("Q3-H3/v58 archived source identity differs")
    with general_live_transport.constructors(
            folder, neutral, integrator=INTEGRATOR) as Foundation:
        def Solver(actual):
            if validate_scope(actual)["fingerprint"] != identity["fingerprint"]:
                raise ValueError("Q3-H3/v58 constructor deck differs")
            instance = Foundation(neutral_foundation_view(actual))
            return attach_nonzero_v58(
                instance, actual, source_path=source)
        yield Solver


__all__ = [
    "CHECKPOINT_PREFIX", "IntegratedNonzeroV58Controller", "SCOPE_ID",
    "attach_nonzero_v58", "constructors", "neutral_foundation_view",
    "prepare_sources",
]
