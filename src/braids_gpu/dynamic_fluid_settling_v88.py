"""Actual-CUDA Q4/v88 dynamic T/S/rho settling lifecycle.

The v82 CUDA owner retains FCT, boundary memory and spatial ``dicuv``.  Its
static exchange adapter is replaced (never launched) by the resident v88 EOS
and SettleFrm=2 kernel.  Backend selection is explicit and there is no CPU
fallback.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from braids_core.sediment.dynamic_fluid_settling_v88 import (
    EOS_CODE,
    MODE_CELL,
    ROLE_CUDA,
    DynamicFluidExecutionV88,
    DynamicFluidSettlingV88Error,
    DynamicFluidStateV88,
    DynamicFluidTrialV88,
    _validate_trial,
    _wrap_state,
    _wrap_trial,
    dynamic_fluid_checkpoint_v88,
    load_dynamic_fluid_checkpoint_v88,
    validate_dynamic_fluid_state_v88,
)
from braids_gpu.multifraction_suspended_lifecycle_v78 import (
    CUDA_OPTIONS,
    CUDA_THREADS_PER_BLOCK,
    _ResidentMultifractionExchangeV78,
)
from braids_gpu.suspended_support_closures_v82 import (
    ResidentSuspendedSupportLifecycleV82,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT/"braids_gpu/dynamic_fluid_settling_v88.cu"
PROPERTY_KERNEL = "v88_fluid_properties"
EXCHANGE_KERNEL = "v88_dynamic_fluid_exchange"
CUDA_COMPOSITION_ID = "resident_dynamic_fluid_settlefrm2_v88_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DynamicFluidSettlingV88Error(message)


class _ResidentDynamicFluidExchangeV88(_ResidentMultifractionExchangeV78):
    """Replace the frozen scalar-property exchange with cell fields."""

    def __init__(self, base_execution, execution: DynamicFluidExecutionV88):
        # Reuse the accepted allocation contract.  The compiled v78 function
        # is immediately replaced and is never launched by this owner.
        super().__init__(base_execution)
        self.execution_v88 = execution
        cp = self.cp
        try:
            module = cp.RawModule(
                code=CUDA_SOURCE.read_text(encoding="utf-8"),
                options=CUDA_OPTIONS,
                name_expressions=(PROPERTY_KERNEL, EXCHANGE_KERNEL))
            self.property_kernel = module.get_function(PROPERTY_KERNEL)
            self.kernel = module.get_function(EXCHANGE_KERNEL)
        except Exception as error:
            raise DynamicFluidSettlingV88Error(
                "v88 CUDA dynamic-fluid kernel compilation failed") from error
        self.parent_temperature = None
        self.parent_salinity = None
        self.parent_density = cp.empty(self.nc, dtype=cp.float64)
        self.parent_viscosity = cp.empty(self.nc, dtype=cp.float64)
        self.parent_property_status = cp.empty(self.nc, dtype=cp.int32)
        self.endpoint_density = cp.empty(self.nc, dtype=cp.float64)
        self.endpoint_viscosity = cp.empty(self.nc, dtype=cp.float64)
        self.endpoint_property_status = cp.empty(self.nc, dtype=cp.int32)
        self.endpoint_temperature = None
        self.endpoint_salinity = None
        self.eos_code = np.int32(EOS_CODE[execution.density_eos])
        self.constituents_active = np.int32(execution.constituents_active)
        self.reference_rho = np.float64(base_execution.profile.
                                        water_density_kg_m3)

    def _cell_field(self, value, name):
        cp = self.cp
        _require(isinstance(value, cp.ndarray) and
                 value.shape == (self.nc,) and value.dtype == cp.float64 and
                 value.flags.c_contiguous,
                 f"v88 CUDA {name} must be contiguous device float64 "
                 f"({self.nc},)")
        return value

    def _properties(self, temperature, salinity, density, viscosity, status):
        temperature = self._cell_field(temperature, "temperature")
        salinity = self._cell_field(salinity, "salinity")
        self.property_kernel(
            ((self.nc+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,),
            (CUDA_THREADS_PER_BLOCK,),
            (np.int32(self.nc), temperature, salinity, self.eos_code,
             self.constituents_active, self.reference_rho, density,
             viscosity, status))
        _require(not bool(self.cp.any(status != 0).item()),
                 "v88 CUDA fluid-property kernel rejected the fields")

    def prepare_parent(self, temperature, salinity) -> None:
        self.parent_temperature = self._cell_field(
            temperature, "accepted parent temperature")
        self.parent_salinity = self._cell_field(
            salinity, "accepted parent salinity")
        self._properties(self.parent_temperature, self.parent_salinity,
                         self.parent_density, self.parent_viscosity,
                         self.parent_property_status)

    def prepare_endpoint(self, temperature, salinity) -> None:
        self.endpoint_temperature = self._cell_field(
            temperature, "accepted endpoint temperature")
        self.endpoint_salinity = self._cell_field(
            salinity, "accepted endpoint salinity")
        self._properties(self.endpoint_temperature, self.endpoint_salinity,
                         self.endpoint_density, self.endpoint_viscosity,
                         self.endpoint_property_status)

    def step(self, depth_before, depth_after, endpoint_momentum, suspended,
             active_bed, equilibrium_capacity, surface_fraction, total_shear,
             wet_mask, dt):
        cp = self.cp
        _require(self.parent_temperature is not None and
                 self.parent_salinity is not None,
                 "v88 CUDA accepted parent fluid properties were not prepared")
        for value, shape, name in (
            (depth_before, (self.nc,), "depth before"),
            (depth_after, (self.nc,), "depth after"),
            (endpoint_momentum, (2, self.nc), "endpoint momentum"),
            (suspended, (self.ng, self.nc), "suspended mass"),
            (active_bed, (self.ng, self.nc), "active bed"),
            (equilibrium_capacity, (self.ng, self.nc),
             "equilibrium capacity"),
            (surface_fraction, (self.ng, self.nc), "surface fraction"),
            (total_shear, (self.nc,), "reference total shear"),
        ):
            self._array(value, shape, name)
        _require(isinstance(wet_mask, cp.ndarray) and
                 wet_mask.shape == (self.nc,) and
                 wet_mask.dtype == cp.int32 and wet_mask.flags.c_contiguous,
                 "v88 CUDA wet mask must be contiguous device int32")
        self.kernel(
            ((self.nc+CUDA_THREADS_PER_BLOCK-1)//CUDA_THREADS_PER_BLOCK,),
            (CUDA_THREADS_PER_BLOCK,),
            (np.int32(self.nc), np.int32(self.ng), depth_before, depth_after,
             endpoint_momentum, suspended, active_bed, equilibrium_capacity,
             surface_fraction, total_shear, wet_mask, self.sediment_d50,
             self.settling_fac_dss, self.sediment_density,
             self.sediment_threshold, self.parent_density,
             self.parent_temperature, self.reference_rho,
             self.reference_concentration, self.capacity_speed_epsilon,
             self.gravity, self.dry, self.factsd,
             self.reference_depth_floor, np.float64(dt),
             self.next_suspended, self.next_active,
             self.equilibrium_concentration, self.adaptation_velocity,
             self.dimensionless_timescale,
             self.clear_water_settling_velocity, self.settling_velocity,
             self.hindered_factor, self.delta, self.limiter_correction,
             self.sediment_active, self.status))
        return {
            "suspended": self.next_suspended,
            "active": self.next_active,
            "equilibrium_concentration": self.equilibrium_concentration,
            "settling_velocity": self.settling_velocity,
            "delta": self.delta,
            "status": self.status,
        }


class ResidentDynamicFluidSettlingLifecycleV88:
    """One explicit-device, no-fallback v88 transaction owner."""

    def __init__(self, execution: DynamicFluidExecutionV88):
        _require(isinstance(execution, DynamicFluidExecutionV88) and
                 execution.role == ROLE_CUDA,
                 "v88 CUDA owner requires an explicit actual_cuda execution")
        self.execution = execution
        self.base = ResidentSuspendedSupportLifecycleV82(
            execution.base_execution)
        self.cp = self.base.cp
        v76_execution = execution.base_execution.base_execution.base_execution
        self.exchange = _ResidentDynamicFluidExchangeV88(
            v76_execution, execution)
        # Replace, rather than compose with, the static v78 exchange owner.
        self.base.base.exchange = self.exchange
        self.device_id = self.base.base.device_id
        source_paths = (
            ROOT/"braids_core/sediment/dynamic_fluid_settling_v88.py",
            ROOT/"braids_gpu/dynamic_fluid_settling_v88.py", CUDA_SOURCE,
        )
        self.runtime_receipt = {
            "cuda_composition_id": CUDA_COMPOSITION_ID,
            "execution_fingerprint": execution.fingerprint,
            "integrator": "implicit_fwave_midpoint",
            "backend": "actual_cuda",
            "fixed_grid": True,
            "fraction_count": execution.fraction_count,
            "automatic_cpu_fallback": False,
            "actual_cuda_kernels": True,
            "device_id": self.device_id,
            "device_name": self.base.base.runtime_receipt["device_name"],
            "threads_per_block": CUDA_THREADS_PER_BLOCK,
            "eos": execution.density_eos,
            "temperature_mode": execution.temperature_mode,
            "salinity_mode": execution.salinity_mode,
            "static_v78_exchange_replaced_not_launched": True,
            "accepted_exchange_owner_count": 1,
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
            "host_readback_scope":
                "candidate authentication and checkpoint evidence only",
        }
        self._open_parent = None
        self._open_trial = None

    def _device_field(self, raw, *, temperature: bool):
        cp = self.cp
        mode = (self.execution.temperature_mode if temperature else
                self.execution.salinity_mode)
        constant = (self.execution.equilibrium_temperature_c if temperature
                    else self.execution.equilibrium_salinity_ppt)
        name = "temperature" if temperature else "salinity"
        if mode != MODE_CELL:
            _require(raw is None,
                     f"v88 CUDA {name} field supplied for constant mode")
            return cp.full(self.execution.cell_count, constant,
                           dtype=cp.float64)
        _require(isinstance(raw, cp.ndarray) and
                 raw.shape == (self.execution.cell_count,) and
                 raw.dtype == cp.float64 and raw.flags.c_contiguous and
                 int(raw.device.id) == self.device_id,
                 f"v88 CUDA accepted endpoint {name} must be contiguous "
                 "resident float64")
        return raw

    def _readback(self, value):
        """Single injectable readback seam used by the rollback gate."""
        return self.cp.asnumpy(value)

    def begin_trial(
        self, parent: DynamicFluidStateV88, *, accepted_step_receipt,
        accepted_endpoint_temperature_c=None,
        accepted_endpoint_salinity_ppt=None,
        cancelled: bool = False, **inputs,
    ) -> DynamicFluidTrialV88:
        validate_dynamic_fluid_state_v88(parent, self.execution)
        _require(self._open_trial is None,
                 "v88 CUDA previous trial is unfinished")
        if cancelled:
            raise DynamicFluidSettlingV88Error(
                "v88 CUDA update cancelled before candidate publication")
        parent_t = self.cp.ascontiguousarray(self.cp.asarray(
            parent.temperature_cell_c, dtype=self.cp.float64))
        parent_s = self.cp.ascontiguousarray(self.cp.asarray(
            parent.salinity_cell_ppt, dtype=self.cp.float64))
        next_t = self._device_field(
            accepted_endpoint_temperature_c, temperature=True)
        next_s = self._device_field(
            accepted_endpoint_salinity_ppt, temperature=False)
        base_trial = None
        try:
            self.exchange.prepare_parent(parent_t, parent_s)
            base_trial = self.base.begin_trial(
                parent.base_state,
                accepted_step_receipt=accepted_step_receipt,
                cancelled=False, **inputs)
            self.exchange.prepare_endpoint(next_t, next_s)
            # All host publication happens inside the rollback guard.
            parent_rho = self._readback(self.exchange.parent_density)
            parent_nu = self._readback(self.exchange.parent_viscosity)
            endpoint_t = self._readback(next_t)
            endpoint_s = self._readback(next_s)
            endpoint_rho = self._readback(self.exchange.endpoint_density)
            endpoint_nu = self._readback(self.exchange.endpoint_viscosity)
            if (not np.array_equal(parent_rho,
                                   parent.water_density_cell_kg_m3) or
                    not np.array_equal(parent_nu,
                                       parent.molecular_viscosity_cell_m2_s)):
                raise DynamicFluidSettlingV88Error(
                    "v88 CUDA parent properties differ from accepted state")
            candidate = _wrap_state(
                self.execution, base_trial.candidate, endpoint_t, endpoint_s,
                endpoint_rho, endpoint_nu)
            trial = _wrap_trial(
                self.execution, parent, base_trial, candidate,
                parent_rho, parent_nu)
        except Exception as error:
            if base_trial is not None:
                try:
                    self.base.rollback(parent.base_state, base_trial)
                except Exception as rollback_error:
                    raise DynamicFluidSettlingV88Error(
                        "v88 CUDA readback failed and resident rollback failed") \
                        from rollback_error
            if isinstance(error, DynamicFluidSettlingV88Error):
                raise
            raise DynamicFluidSettlingV88Error(
                "v88 CUDA trial failed before candidate publication") from error
        self._open_parent = parent.state_sha256
        self._open_trial = trial.trial_sha256
        return trial

    def _finish(self, parent: DynamicFluidStateV88,
                trial: DynamicFluidTrialV88, *, commit: bool
                ) -> DynamicFluidStateV88:
        _require(self._open_parent == parent.state_sha256 and
                 self._open_trial == trial.trial_sha256,
                 "v88 CUDA trial is not the resident open transaction")
        _validate_trial(self.execution, parent, trial)
        base_result = (self.base.commit(parent.base_state, trial.base_trial)
                       if commit else
                       self.base.rollback(parent.base_state, trial.base_trial))
        self._open_parent = None
        self._open_trial = None
        if commit:
            _require(base_result.state_sha256 ==
                     trial.candidate.base_state.state_sha256,
                     "v88 CUDA committed base candidate differs")
            return trial.candidate
        _require(base_result is parent.base_state,
                 "v88 CUDA rollback changed the parent")
        return parent

    def commit(self, parent, trial) -> DynamicFluidStateV88:
        return self._finish(parent, trial, commit=True)

    def rollback(self, parent, trial) -> DynamicFluidStateV88:
        return self._finish(parent, trial, commit=False)

    def cancel(self, parent, trial) -> DynamicFluidStateV88:
        return self.rollback(parent, trial)

    def checkpoint(self, state: DynamicFluidStateV88) -> dict:
        _require(self._open_trial is None,
                 "v88 CUDA checkpoint requires a committed point")
        base_document = self.base.checkpoint(state.base_state)
        document = dynamic_fluid_checkpoint_v88(self.execution, state)
        _require(document["base_v82_checkpoint"] == base_document,
                 "v88 CUDA/base checkpoint differs")
        return document

    def restore_checkpoint(self, document) -> DynamicFluidStateV88:
        _require(self._open_trial is None,
                 "v88 CUDA restart cannot replace an open trial")
        state = load_dynamic_fluid_checkpoint_v88(self.execution, document)
        resident = self.base.restore_checkpoint(
            document["base_v82_checkpoint"])
        _require(resident.state_sha256 == state.base_state.state_sha256,
                 "v88 CUDA/base restored state differs")
        return state


def method_identity_v88_cuda() -> dict:
    return {
        "cuda_composition_id": CUDA_COMPOSITION_ID,
        "integrator": "implicit_fwave_midpoint",
        "threads_per_block": CUDA_THREADS_PER_BLOCK,
        "automatic_cpu_fallback": False,
        "actual_cuda_kernels": True,
        "resident_operators": [
            "Eckart/UNESCO/NaCl-or-Rhow fluid properties",
            "local eqsettle viscosity", "SettleFrm=2/Galappatti exchange",
        ],
        "static_v78_exchange_replaced_not_composed": True,
        "accepted_exchange_owner_count": 1,
    }


__all__ = [
    "CUDA_COMPOSITION_ID", "ResidentDynamicFluidSettlingLifecycleV88",
    "method_identity_v88_cuda",
]
