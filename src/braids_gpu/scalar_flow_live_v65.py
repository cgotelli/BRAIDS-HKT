"""Resident actual-CUDA mirror of the bounded scalar FLOW v65 transaction.

The route is intentionally narrow and explicit.  It accepts only the
``implicit_fwave_midpoint_gpu`` execution identity, keeps physical state on
one CUDA device, consumes the resident v64 producer and channel composer, and
forms an all-or-nothing conservative Exner candidate.  It never substitutes
CPU physics or silently changes backend.

Class-1 fractions can enter the live bed transaction.  Class 3 remains a
resident channel diagnostic until Q4 owns water-column mass and exchange;
class 2 is rejected by the canonical profile.  Wet/dry, AMR, wave/scour and
MaximumWaterdepth remain fail-closed in this envelope.  Exact/Dirac
within-class distributions are live, and a zero-stock cell may carry only an
explicit normalized retained surface identity; its inventory availability is
still exactly zero.  This is a provisional Q3 identity, not Q5 stratigraphy.
Positive-TraFrm ISlope 2 uses the exact zero ``taurat`` inherited from locked
classic FLOW's erosed/eqtran chain.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping

import numpy as np

from braids_core.sediment.channel_composition import (
    TRA_ADVDIFF,
    TRA_BEDLOAD,
    TRA_COMBINE,
    ChannelCompositionProfile,
    channel_composition_profile_document,
    load_channel_composition_profile,
)
from braids_core.sediment.scalar_flow_gpu_producer_v64 import (
    load_scalar_flow_gpu_producer_document,
)
from braids_core.sediment.scalar_flow_live_v65 import (
    DONOR_NEIGHBOUR,
    DONOR_NONE,
    DONOR_OWNER,
    DONOR_TRACE,
    ENVELOPE_VERSION,
    FACE_CLOSED,
    FACE_INTERNAL,
    FACE_OWNER_OUTLET,
    FACE_TRACE_INLET,
    FACE_TRACE_OUTLET,
    MAX_ACCEPTED_STEPS,
    SCHEMA,
    SCOPE_ID,
    SEMANTIC_ID,
    SLOPE_BAGNOLD_IKEDA,
    SLOPE_KOCH_FLOKSTRA,
    SLOPE_OFF,
    SLOPE_PARKER_ANDREWS,
    ScalarFlowLiveV65Error,
    ScalarFlowV65LateralStencil,
    load_state_checkpoint_v65 as load_cpu_state_checkpoint_v65,
    resolve_lateral_stencil_v65 as resolve_cpu_lateral_stencil_v65,
    resolve_state_v65 as resolve_cpu_state_v65,
    state_checkpoint_document_v65 as cpu_state_checkpoint_document_v65,
)
from braids_core.sediment.classic_flow_taurat_v69 import TAURAT_POSITIVE_ZERO

from .channel_composition import ResidentChannelComposer
from .flow_bed_slope_composition import ResidentFlowBedSlopeFaceComposer
from .scalar_flow_producers_v64 import (
    ResidentScalarFlowCapacityV64,
    ResidentScalarFlowInputsV64,
    ScalarFlowProducersV64CUDA,
)


BACKEND = "implicit_fwave_midpoint_gpu"
INTEGRATOR = "implicit_fwave_midpoint"
CUDA_OPTIONS = (
    "--std=c++11", "--fmad=false", "--prec-div=true",
    "--prec-sqrt=true", "--ftz=false",
)
KERNEL_NAMES = (
    "scalar_v65_lateral_421",
    "scalar_v65_face_candidates",
    "scalar_v65_face_potential",
    "scalar_v65_inventory_demand",
    "scalar_v65_inventory_fill",
    "scalar_v65_atomic_exner",
    "scalar_v65_digest_fp64",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: object, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ScalarFlowLiveV65Error(f"Q3-H5/v65 GPU {name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 GPU {name} must be numeric") from error
    if (not math.isfinite(result) or (positive and result <= 0.0) or
            (nonnegative and result < 0.0)):
        raise ScalarFlowLiveV65Error(
            f"Q3-H5/v65 GPU {name} is outside its finite domain")
    return 0.0 if result == 0.0 else result


def _runtime_execution(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU execution identity is missing")
    if (value.get("schema") != SCHEMA or value.get("scope_id") != SCOPE_ID or
            value.get("semantic_id") != SEMANTIC_ID or
            value.get("envelope_version") != ENVELOPE_VERSION):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU execution header differs")
    body = {key: item for key, item in value.items() if key != "fingerprint"}
    if value.get("fingerprint") != _fingerprint(body):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU execution fingerprint differs")
    if (value.get("backend") != BACKEND or
            value.get("automatic_backend_fallback") is not False):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU requires the explicit implicit_fwave_midpoint_gpu "
            "backend with fallback disabled")
    if (value.get("mesh_scope") != "fixed_grid" or
            value.get("wet_scope") != "fully_wet_positive_depth" or
            value.get("forcing_scope") != "current_only_wave_false_scour_false"):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU rejects AMR, wet/dry, wave, or scour scope")
    return dict(value)


def _runtime_profile(
    execution: Mapping[str, object],
    value: ChannelCompositionProfile | Mapping[str, object],
) -> ChannelCompositionProfile:
    profile = (value if isinstance(value, ChannelCompositionProfile)
               else load_channel_composition_profile(value))
    document = channel_composition_profile_document(profile)
    if document["fingerprint"] != execution["dependencies"]["channel_profile"]:
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU channel profile differs from execution")
    if ([row.fraction_id for row in profile.fractions] !=
            execution["ordered_fraction_ids"] or
            [row.transport_class for row in profile.fractions] !=
            execution["transport_classes"]):
        raise ScalarFlowLiveV65Error(
            "Q3-H5/v65 GPU channel layout differs from execution")
    return profile


@dataclass(frozen=True)
class ResidentScalarFlowV65LateralStencil:
    """Canonical locked upwbed 4-2-1 stencil uploaded to one CUDA device."""

    ncell: int
    x_offsets: object
    x_indices: object
    x_weights: object
    y_offsets: object
    y_indices: object
    y_weights: object
    geometry_fingerprint: str
    fingerprint: str
    resident_digest: str
    device_id: int


@dataclass(frozen=True)
class ResidentScalarFlowV65State:
    execution_fingerprint: str
    bed_mass_kg_m2: object
    bulk_density_kg_m3: object
    active_fraction: object
    boundary_signed_kg: object
    boundary_absolute_kg: object
    accepted_step: int
    time_s: float
    state_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentScalarFlowV65CellChannels:
    execution_fingerprint: str
    producer_execution_fingerprint: str
    composed_bed_mass: object
    retained_suspended_mass: object
    signed_suspended_mass_rate: object
    equilibrium_concentration: object
    total_transport_direction: object
    digest: str
    device_id: int


@dataclass(frozen=True)
class ResidentScalarFlowV65SlopeFields:
    downhill: object
    total_current_ust2: object
    ust2_provenance_fingerprint: str
    stress_semantics: str
    stress: object
    depth: object
    d50: object
    mean_diameter: object
    hiding: object
    source_production_sha256: str | None
    digest: str
    device_id: int


@dataclass(frozen=True)
class ResidentScalarFlowV65FaceBatch:
    owner: object
    neighbour: object
    role: object
    normal: object
    length_m: object
    trace_steered_bed_mass: object
    trace_raw_suspended_mass: object
    trace_base_velocity: object
    trace_depth_m: object
    slope: ResidentScalarFlowV65SlopeFields | None
    digest: str
    device_id: int


@dataclass(frozen=True)
class ResidentScalarFlowV65Trial:
    execution_fingerprint: str
    parent_state_sha256: str
    dt_s: float
    morphological_factor: float
    face_raw_candidate: object
    face_corrected_candidate: object
    raw_donor: object
    donor: object
    active_fraction_factor: object
    inventory_fill: object
    availability: object
    face_flux_kg_m_s: object
    cell_delta_kg_m2: object
    candidate: ResidentScalarFlowV65State
    boundary_transfer_kg: object
    depletion_roundoff_transfer_kg: object
    conservation_residual_kg: object
    active_fraction_application_count: int
    inventory_fill_application_count: int
    trial_sha256: str
    device_id: int


@dataclass(frozen=True)
class ResidentScalarFlowV65Production:
    """One authenticated v64 producer result and its v65 cell channels."""

    producer_inputs: ResidentScalarFlowInputsV64
    capacity: ResidentScalarFlowCapacityV64
    channels: ResidentScalarFlowV65CellChannels
    producer_execution_fingerprint: str
    production_sha256: str
    device_id: int


class ResidentScalarFlowLiveV65:
    """One-device resident implementation of the canonical v65 contract."""

    backend = BACKEND
    integrator = INTEGRATOR

    def __init__(
        self,
        *,
        execution: Mapping[str, object],
        producer_execution: Mapping[str, object],
        channel_profile: ChannelCompositionProfile | Mapping[str, object],
        producer: ScalarFlowProducersV64CUDA,
        channel_composer: ResidentChannelComposer | None = None,
        slope_composer: ResidentFlowBedSlopeFaceComposer | None = None,
        source_path: str | Path | None = None,
    ):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.execution = _runtime_execution(execution)
        self.profile = _runtime_profile(self.execution, channel_profile)
        if not isinstance(producer, ScalarFlowProducersV64CUDA):
            raise TypeError("producer must be a resident ScalarFlowProducersV64CUDA")
        self.producer = producer
        if producer.device_id != self.device_id:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU producer selected a different CUDA device")
        producer_identity = load_scalar_flow_gpu_producer_document(
            producer_execution)
        dependencies = self.execution["dependencies"]
        if (producer_identity["backend"] != BACKEND or
                producer_identity["integrator"] != INTEGRATOR or
                producer_identity["automatic_cpu_fallback"] is not False or
                producer_identity["fingerprint"] !=
                dependencies["backend_producer_execution"] or
                producer_identity["scope_id"] !=
                dependencies["backend_producer_scope_id"] or
                producer_identity["dependencies"]["distribution_fingerprint"] !=
                dependencies["backend_distribution_fingerprint"] or
                producer.distribution_fingerprint !=
                dependencies["backend_distribution_fingerprint"] or
                producer_identity["reference_selector"] !=
                self.execution["reference_selector"] or
                producer_identity["fraction_ids"] !=
                self.execution["ordered_fraction_ids"] or
                producer.nclass != len(self.profile.fractions)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU live producer or distribution identity differs")
        self.producer_execution = producer_identity
        self.channel_composer = channel_composer or ResidentChannelComposer()
        if self.channel_composer.device_id != self.device_id:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU channel composer selected a different device")
        self.slope_composer = slope_composer
        slope_method = self.execution["bed_slope_method_id"]
        if slope_method == SLOPE_OFF:
            if slope_composer is not None:
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU slope-off rejects an inert slope composer")
        else:
            if not isinstance(slope_composer, ResidentFlowBedSlopeFaceComposer):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU active slope requires the resident H4 composer")
            identity = getattr(slope_composer, "identity", {})
            if (slope_composer.device_id != self.device_id or
                    identity.get("fingerprint") !=
                    dependencies["bed_slope_execution"] or
                    slope_composer.method_id != slope_method):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU slope composer identity differs")
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source, options=CUDA_OPTIONS, name_expressions=KERNEL_NAMES)
            self.kernels = {
                name: module.get_function(name) for name in KERNEL_NAMES}
            self.transport_class = cp.ascontiguousarray(cp.asarray(
                [row.transport_class for row in self.profile.fractions],
                dtype=cp.int32))
            self.bulk_density = cp.ascontiguousarray(cp.asarray(
                self.execution["bulk_density_kg_m3"], dtype=cp.float64))

    def _array(self, name: str, value: object, ndim: int,
               shape: tuple[int, ...] | None = None, *, integer: bool = False,
               copy: bool = False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} CuPy array")
        if int(value.device.id) != self.device_id:
            raise ScalarFlowLiveV65Error(
                f"{name} is not resident on CUDA device {self.device_id}")
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ScalarFlowLiveV65Error(
                f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value.copy(order="C") if copy else value

    @staticmethod
    def _flag(value: object) -> bool:
        return bool(value.item() if hasattr(value, "item") else value)

    def _all(self, value: object) -> bool:
        return self._flag(self.cp.all(value))

    def _digest(self, *arrays: object, label: str) -> str:
        """Compute a deterministic resident content token; copy only 128 bits."""
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            digest = cp.zeros(2, dtype=cp.uint64)
            inventory = []
            for ordinal, raw in enumerate(arrays):
                if not isinstance(raw, cp.ndarray) or int(raw.device.id) != self.device_id:
                    raise ScalarFlowLiveV65Error(
                        "Q3-H5/v65 GPU digest received a nonresident array")
                value = (raw if raw.dtype == cp.float64 else
                         cp.ascontiguousarray(raw.astype(cp.float64)))
                value = cp.ascontiguousarray(value.reshape(-1))
                count = int(value.size)
                if count:
                    self.kernels["scalar_v65_digest_fp64"](
                        ((count+255)//256,), (256,),
                        (np.int64(count),
                         np.uint64(0x9E3779B97F4A7C15 ^ (ordinal+1)),
                         value, digest))
                inventory.append({
                    "dtype": str(raw.dtype), "shape": list(raw.shape),
                    "ordinal": ordinal,
                })
            words = (int(digest[0].item()), int(digest[1].item()))
        return _fingerprint({
            "backend": BACKEND, "label": label,
            "inventory": inventory, "resident_digest_u64": list(words),
        })

    def _validate_state(self, state: ResidentScalarFlowV65State) -> None:
        if (not isinstance(state, ResidentScalarFlowV65State) or
                state.device_id != self.device_id or
                state.execution_fingerprint != self.execution["fingerprint"]):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU accepted state differs from execution/device")
        expected = self._state_token(
            state.bed_mass_kg_m2, state.bulk_density_kg_m3,
            state.active_fraction, state.boundary_signed_kg,
            state.boundary_absolute_kg, state.accepted_step, state.time_s)
        if expected != state.state_sha256:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU accepted state is stale or mutated")

    def _state_token(self, mass, density, fraction, signed, absolute,
                     step: int, time_s: float) -> str:
        array_token = self._digest(
            mass, density, fraction, signed, absolute, label="accepted_state")
        return _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "accepted_step": step, "time_s": time_s,
            "resident_arrays_sha256": array_token,
        })

    def resolve_lateral_stencil(
        self, *, ncell: int,
        x_offsets: object, x_indices: object, x_weights: object,
        y_offsets: object, y_indices: object, y_weights: object,
        geometry_fingerprint: str,
    ) -> ResidentScalarFlowV65LateralStencil:
        """Authenticate the canonical CPU geometry metadata, then upload it."""
        portable = resolve_cpu_lateral_stencil_v65(
            ncell=ncell, x_offsets=x_offsets, x_indices=x_indices,
            x_weights=x_weights, y_offsets=y_offsets, y_indices=y_indices,
            y_weights=y_weights, geometry_fingerprint=geometry_fingerprint)
        return self._upload_lateral_stencil(portable)

    def _upload_lateral_stencil(
        self, stencil: ScalarFlowV65LateralStencil,
    ) -> ResidentScalarFlowV65LateralStencil:
        if not isinstance(stencil, ScalarFlowV65LateralStencil):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU expected a canonical lateral stencil")
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            values = [cp.ascontiguousarray(cp.asarray(value)) for value in (
                stencil.x_offsets, stencil.x_indices, stencil.x_weights,
                stencil.y_offsets, stencil.y_indices, stencil.y_weights)]
        resident_digest = self._digest(*values, label="lateral_stencil")
        return ResidentScalarFlowV65LateralStencil(
            stencil.ncell, *values, stencil.geometry_fingerprint,
            stencil.fingerprint, resident_digest, self.device_id)

    def apply_lateral_stencil(
        self, bed_mass: object, stencil: ResidentScalarFlowV65LateralStencil,
    ):
        cp = self.cp
        if (not isinstance(stencil, ResidentScalarFlowV65LateralStencil) or
                stencil.device_id != self.device_id or
                stencil.resident_digest != self._digest(
                    stencil.x_offsets, stencil.x_indices, stencil.x_weights,
                    stencil.y_offsets, stencil.y_indices, stencil.y_weights,
                    label="lateral_stencil")):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU lateral stencil differs from device or receipt")
        bed = self._array("bed_mass", bed_mass, 3)
        if bed.shape[0] != 2 or bed.shape[2] != stencil.ncell:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU lateral bed field shape differs")
        nfrac = int(bed.shape[1])
        count = nfrac*stencil.ncell
        output = cp.empty_like(bed)
        if count:
            self.kernels["scalar_v65_lateral_421"](
                ((count+255)//256,), (256,),
                (np.int64(stencil.ncell), np.int32(nfrac),
                 stencil.x_offsets, stencil.x_indices, stencil.x_weights,
                 stencil.y_offsets, stencil.y_indices, stencil.y_weights,
                 bed, output))
        if not self._all(cp.isfinite(output)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU lateral averaging produced nonfinite transport")
        return cp.ascontiguousarray(output)

    def resolve_state(
        self, *, bed_mass_kg_m2: object, active_fraction: object | None = None,
        boundary_signed_kg: object | None = None,
        boundary_absolute_kg: object | None = None,
        accepted_step: int = 0, time_s: float = 0.0,
    ) -> ResidentScalarFlowV65State:
        cp = self.cp
        mass = self._array("bed_mass_kg_m2", bed_mass_kg_m2, 2, copy=True)
        nfrac, ncell = map(int, mass.shape)
        if nfrac != len(self.profile.fractions) or ncell <= 0:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU bed mass layout differs from execution")
        if not self._all(cp.isfinite(mass) & (mass >= 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU bed mass must be finite nonnegative")
        density = self.bulk_density.copy()
        volume = mass/density[:, None]
        total = cp.sum(volume, axis=0, keepdims=True)
        if not self._all(cp.isfinite(total) & (total >= 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU accepted solid volume is invalid")
        positive_stock = total[0] > 0.0
        derived = cp.ascontiguousarray(cp.where(
            positive_stock[None, :],
            volume/cp.where(total > 0.0, total, 1.0), 0.0))
        if active_fraction is None:
            if self._flag(cp.any(~positive_stock)):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU depleted cells require an explicit "
                    "retained surface identity")
            fraction = derived
        else:
            supplied = self._array(
                "active_fraction", active_fraction, 2, (nfrac, ncell), copy=True)
            tolerance = 512.0*np.finfo(np.float64).eps
            if not self._all(
                    cp.isfinite(supplied) & (supplied >= 0.0)) or not self._flag(
                    cp.allclose(cp.sum(supplied, axis=0), 1.0,
                                rtol=0.0, atol=tolerance*nfrac)) or not self._flag(
                    cp.allclose(supplied[:, positive_stock],
                                derived[:, positive_stock],
                                rtol=tolerance, atol=tolerance)):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU active fractions differ from positive "
                    "accepted stock or are not normalized retained identities")
            fraction = supplied
        if boundary_signed_kg is None:
            signed = cp.zeros(nfrac, dtype=cp.float64)
        else:
            signed = self._array(
                "boundary_signed_kg", boundary_signed_kg, 1, (nfrac,), copy=True)
        if boundary_absolute_kg is None:
            absolute = cp.zeros(nfrac, dtype=cp.float64)
        else:
            absolute = self._array(
                "boundary_absolute_kg", boundary_absolute_kg, 1,
                (nfrac,), copy=True)
        if (not self._all(cp.isfinite(signed)) or
                not self._all(cp.isfinite(absolute) & (absolute >= 0.0))):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU boundary ledgers are invalid")
        if (isinstance(accepted_step, (bool, np.bool_)) or
                not isinstance(accepted_step, int) or accepted_step < 0 or
                accepted_step > MAX_ACCEPTED_STEPS):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU accepted step is invalid")
        accepted_time = _finite(time_s, "accepted time", nonnegative=True)
        token = self._state_token(
            mass, density, fraction, signed, absolute,
            accepted_step, accepted_time)
        return ResidentScalarFlowV65State(
            self.execution["fingerprint"], mass, density, fraction, signed,
            absolute, accepted_step, accepted_time, token, self.device_id)

    def compose_cell_channels(
        self, *, steered_bed_mass: object, raw_suspended_mass: object,
        base_velocity: object, depth_m: object,
    ) -> ResidentScalarFlowV65CellChannels:
        cp = self.cp
        nfrac = len(self.profile.fractions)
        raw_bed = self._array("steered_bed_mass", steered_bed_mass, 3)
        if raw_bed.shape[0] != 2 or raw_bed.shape[1] != nfrac:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU steered bed shape differs")
        ncell = int(raw_bed.shape[2])
        raw_suspended = self._array(
            "raw_suspended_mass", raw_suspended_mass, 3,
            (2, nfrac, ncell))
        velocity = self._array(
            "base_velocity", base_velocity, 2, (2, ncell))
        depth = self._array("depth_m", depth_m, 1, (ncell,))
        if not self._all(cp.isfinite(depth) & (depth > 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU is fully wet only")
        result = self.channel_composer.compose(
            raw_bed_mass=raw_bed,
            raw_suspended_mass=raw_suspended,
            base_velocity=velocity,
            depth=depth,
            transport_class=self.transport_class,
            bed_factor=self.profile.bed_factor,
            suspended_factor=self.profile.suspended_factor,
            speed_epsilon=self.profile.epsilon,
            reject_invalid=True)
        total = cp.ascontiguousarray(
            result.composed_bed_mass+result.retained_suspended_mass)
        arrays = (
            result.composed_bed_mass, result.retained_suspended_mass,
            result.signed_suspended_mass_rate,
            result.equilibrium_suspended_mass_concentration, total)
        producer_fingerprint = str(
            self.execution["dependencies"]["backend_producer_execution"])
        return ResidentScalarFlowV65CellChannels(
            str(self.execution["fingerprint"]), producer_fingerprint,
            *arrays, self._digest(*arrays, label="cell_channels"),
            self.device_id)

    def produce_cell_channels(
        self, *, formula_inputs: Mapping[str, object], **producer_inputs,
    ) -> ResidentScalarFlowV65Production:
        """Run the selected v64 producer and v65 composer without host physics."""
        cp = self.cp
        produced = self.producer.produce(**producer_inputs)
        if (not self._all(produced.active == 1) or
                not self._all(produced.hydraulic.raw_depth > 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU producer result is not fully wet/fixed-active")
        capacity = self.producer.evaluate(
            int(self.execution["reference_selector"]), produced,
            formula_inputs=formula_inputs)
        if capacity.active_fraction_applied is not False:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU v64 producer pre-applied active fraction")
        base_velocity = cp.ascontiguousarray(cp.stack((
            produced.hydraulic.characteristic_u,
            produced.hydraulic.characteristic_v)))
        channels = self.compose_cell_channels(
            steered_bed_mass=capacity.bed_mass,
            raw_suspended_mass=capacity.suspended_mass,
            base_velocity=base_velocity,
            depth_m=produced.hydraulic.raw_depth)
        production_token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "producer_execution_fingerprint": self.producer_execution["fingerprint"],
            "selector": int(capacity.selector),
            "producer_arrays_sha256": self._digest(
                produced.hydraulic.values, produced.mixture.values,
                produced.mixture.fraction, produced.mixture.resolved_d50,
                produced.kinematic_viscosity, produced.solid_density,
                produced.rp_taub, produced.active, capacity.raw_values,
                capacity.bed_mass_pre_secondary, capacity.bed_mass,
                capacity.suspended_mass, capacity.slope_total_current_ust2,
                label="v64_production"),
            "channel_receipt_sha256": channels.digest,
        })
        return ResidentScalarFlowV65Production(
            produced, capacity, channels,
            self.producer_execution["fingerprint"], production_token,
            self.device_id)

    def produce_cell_channels_for_state(
        self, *, state: ResidentScalarFlowV65State,
        formula_inputs: Mapping[str, object], **producer_inputs,
    ) -> ResidentScalarFlowV65Production:
        """Produce from one accepted state, including retained zero-stock CDFs.

        The retained surface fractions are metadata only.  The producer sees
        the accepted mass unchanged, so a depleted class/cell still has zero
        finite availability in the downstream Exner owner.
        """
        self._validate_state(state)
        supplied_mass = producer_inputs.get("surface_mass")
        if supplied_mass is None or not self._flag(self.cp.array_equal(
                supplied_mass, state.bed_mass_kg_m2)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU producer surface mass differs from accepted state")
        if "retained_surface_fraction" in producer_inputs:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU retained identity is owned by accepted state")
        return self.produce_cell_channels(
            formula_inputs=formula_inputs,
            retained_surface_fraction=state.active_fraction,
            **producer_inputs,
        )

    def _resolve_slope_fields_impl(
        self, *, nfrac: int, ncell: int, nface: int,
        downhill: object, total_current_ust2: object,
        total_current_ust2_provenance_fingerprint: str,
        gravity_m_s2: float, solid_density_kg_m3: object,
        water_density_kg_m3: object, depth: object, d50: object,
        mean_diameter: object, hiding: object,
        source_production_fingerprint: str | None = None,
    ) -> ResidentScalarFlowV65SlopeFields:
        cp = self.cp
        if nfrac != len(self.profile.fractions) or min(ncell, nface) <= 0:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope dimensions differ")
        down = self._array("slope downhill", downhill, 2, (2, nface), copy=True)
        ust2 = self._array(
            "total_current_ust2", total_current_ust2, 1, (nface,), copy=True)
        solid = self._array(
            "solid_density_kg_m3", solid_density_kg_m3, 1,
            (nfrac,), copy=True)
        water = self._array(
            "water_density_kg_m3", water_density_kg_m3, 1,
            (nface,), copy=True)
        water_depth = self._array(
            "slope depth", depth, 2, (nfrac, nface), copy=True)
        diameter = self._array(
            "slope d50", d50, 2, (nfrac, nface), copy=True)
        mean = self._array(
            "slope mean diameter", mean_diameter, 2,
            (nfrac, nface), copy=True)
        hide = self._array(
            "slope hiding", hiding, 2, (nfrac, ncell), copy=True)
        gravity = _finite(gravity_m_s2, "slope gravity", positive=True)
        provenance = str(total_current_ust2_provenance_fingerprint)
        if provenance != self.execution["dependencies"]["backend_producer_execution"]:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope USTAR-squared provenance differs")
        if source_production_fingerprint is not None and (
                not isinstance(source_production_fingerprint, str) or
                len(source_production_fingerprint) != 64):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope production receipt is invalid")
        valid = (
            self._all(cp.isfinite(down)) and
            self._all(cp.isfinite(ust2) & (ust2 >= 0.0)) and
            self._all(cp.isfinite(solid) & (solid > 0.0)) and
            self._all(cp.isfinite(water) & (water > 0.0)) and
            self._all(solid[:, None] > water[None, :]) and
            self._all(cp.isfinite(water_depth) & (water_depth > 0.0)) and
            self._all(cp.isfinite(diameter) & (diameter > 0.0)) and
            self._all(cp.isfinite(mean) & (mean > 0.0)) and
            self._all(cp.isfinite(hide) & (hide >= 0.0)))
        if not valid:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope fields are outside their physical domain")
        slope_method = self.execution["bed_slope_method_id"]
        if slope_method == SLOPE_BAGNOLD_IKEDA:
            stress_semantics = TAURAT_POSITIVE_ZERO
            stress = cp.zeros((nfrac, nface), dtype=cp.float64)
        elif slope_method in (SLOPE_KOCH_FLOKSTRA, SLOPE_PARKER_ANDREWS):
            stress_semantics = (
                "absolute_shields_from_total_current_RP_USTAR_squared")
            relative = (solid[:, None]-water[None, :])/water[None, :]
            stress = cp.ascontiguousarray(
                ust2[None, :]/gravity/relative/diameter)
        else:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope fields require live ISlope 2, 3 or 4")
        if not self._all(cp.isfinite(stress) & (stress >= 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU derived Shields is invalid")
        digest = _fingerprint({
            "producer_execution_fingerprint": provenance,
            "source_production_sha256": source_production_fingerprint,
            "stress_semantics": stress_semantics,
            "resident_arrays_sha256": self._digest(
                down, ust2, stress, water_depth, diameter, mean, hide,
                label="slope_fields"),
        })
        return ResidentScalarFlowV65SlopeFields(
            down, ust2, provenance, stress_semantics, stress, water_depth,
            diameter, mean, hide, source_production_fingerprint, digest,
            self.device_id)

    def resolve_slope_fields(
        self, *, nfrac: int, ncell: int, nface: int,
        downhill: object, total_current_ust2: object,
        total_current_ust2_provenance_fingerprint: str,
        gravity_m_s2: float, solid_density_kg_m3: object,
        water_density_kg_m3: object, depth: object, d50: object,
        mean_diameter: object, hiding: object,
    ) -> ResidentScalarFlowV65SlopeFields:
        """Build component-only slope fields without a live production receipt."""
        return self._resolve_slope_fields_impl(
            nfrac=nfrac, ncell=ncell, nface=nface, downhill=downhill,
            total_current_ust2=total_current_ust2,
            total_current_ust2_provenance_fingerprint=(
                total_current_ust2_provenance_fingerprint),
            gravity_m_s2=gravity_m_s2,
            solid_density_kg_m3=solid_density_kg_m3,
            water_density_kg_m3=water_density_kg_m3, depth=depth, d50=d50,
            mean_diameter=mean_diameter, hiding=hiding,
            source_production_fingerprint=None)

    def resolve_slope_fields_from_v64(
        self, *, production: ResidentScalarFlowV65Production,
        owner: object, neighbour: object, role: object,
        downhill: object, gravity_m_s2: float,
        solid_density_kg_m3: object, water_density_kg_m3: object,
        depth: object, d50: object, mean_diameter: object, hiding: object,
    ) -> ResidentScalarFlowV65SlopeFields:
        """Derive face USTAR2 from the authenticated resident v64 result.

        Internal faces use the owner/neighbour arithmetic average; physical
        boundaries use the owner value.  This adapter prevents a caller from
        attaching the right provenance string to an unrelated stress field.
        """
        cp = self.cp
        if not isinstance(production, ResidentScalarFlowV65Production):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope requires its authenticated v64 production")
        nfrac = len(self.profile.fractions)
        try:
            ncell = int(production.channels.composed_bed_mass.shape[2])
        except (AttributeError, IndexError, TypeError) as error:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope production layout differs") from error
        self._validate_production(production, nfrac, ncell)
        ust2_cell = production.capacity.slope_total_current_ust2
        if (not isinstance(ust2_cell, cp.ndarray) or
                ust2_cell.dtype != cp.float64 or ust2_cell.ndim != 1 or
                not ust2_cell.flags.c_contiguous or
                int(ust2_cell.device.id) != self.device_id):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU v64 total-current USTAR squared is not resident")
        if int(ust2_cell.size) != ncell:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU production USTAR-squared layout differs")
        own = self._array("slope owner", owner, 1, integer=True)
        nface = int(own.size)
        other = self._array(
            "slope neighbour", neighbour, 1, (nface,), integer=True)
        roles = self._array("slope role", role, 1, (nface,), integer=True)
        internal = roles == FACE_INTERNAL
        valid_roles = (internal | (roles == FACE_TRACE_INLET) |
                       (roles == FACE_TRACE_OUTLET) |
                       (roles == FACE_OWNER_OUTLET) |
                       (roles == FACE_CLOSED))
        topology = ((own >= 0) & (own < ncell) & valid_roles &
                    cp.where(internal,
                             (other >= 0) & (other < ncell) & (other != own),
                             other == -1))
        if not self._all(topology):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU USTAR-squared face topology differs")
        safe_other = cp.maximum(other, 0)
        face_ust2 = cp.ascontiguousarray(cp.where(
            internal, 0.5*(ust2_cell[own]+ust2_cell[safe_other]),
            ust2_cell[own]))
        return self._resolve_slope_fields_impl(
            nfrac=nfrac, ncell=ncell, nface=nface,
            downhill=downhill, total_current_ust2=face_ust2,
            total_current_ust2_provenance_fingerprint=
            production.producer_execution_fingerprint,
            gravity_m_s2=gravity_m_s2,
            solid_density_kg_m3=solid_density_kg_m3,
            water_density_kg_m3=water_density_kg_m3, depth=depth, d50=d50,
            mean_diameter=mean_diameter, hiding=hiding,
            source_production_fingerprint=production.production_sha256)

    def resolve_face_batch(
        self, *, nfrac: int, ncell: int,
        owner: object, neighbour: object, role: object,
        normal: object, length_m: object,
        trace_steered_bed_mass: object | None = None,
        trace_raw_suspended_mass: object | None = None,
        trace_base_velocity: object | None = None,
        trace_depth_m: object | None = None,
        slope: ResidentScalarFlowV65SlopeFields | None = None,
    ) -> ResidentScalarFlowV65FaceBatch:
        cp = self.cp
        if nfrac != len(self.profile.fractions) or ncell <= 0:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU face dimensions differ")
        own = self._array("owner", owner, 1, integer=True, copy=True)
        nface = int(own.size)
        if nface <= 0:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU face catalogue may not be empty")
        other = self._array(
            "neighbour", neighbour, 1, (nface,), integer=True, copy=True)
        roles = self._array(
            "role", role, 1, (nface,), integer=True, copy=True)
        normals = self._array(
            "normal", normal, 2, (2, nface), copy=True)
        length = self._array(
            "length_m", length_m, 1, (nface,), copy=True)
        valid_roles = ((roles == FACE_INTERNAL) |
                       (roles == FACE_TRACE_INLET) |
                       (roles == FACE_TRACE_OUTLET) |
                       (roles == FACE_OWNER_OUTLET) |
                       (roles == FACE_CLOSED))
        internal = roles == FACE_INTERNAL
        topology = ((own >= 0) & (own < ncell) & valid_roles &
                    cp.where(internal,
                             (other >= 0) & (other < ncell) & (other != own),
                             other == -1))
        tolerance = 1024.0*np.finfo(np.float64).eps
        if (not self._all(topology) or
                not self._all(cp.isfinite(normals)) or
                not self._flag(cp.allclose(
                    cp.sum(normals*normals, axis=0), 1.0,
                    rtol=0.0, atol=tolerance)) or
                not self._all(cp.isfinite(length) & (length > 0.0))):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU face topology/geometry differs")
        vector_shape = (2, nfrac, nface)
        trace_bed = (cp.zeros(vector_shape, dtype=cp.float64)
                     if trace_steered_bed_mass is None else
                     self._array("trace_steered_bed_mass",
                                 trace_steered_bed_mass, 3, vector_shape,
                                 copy=True))
        trace_suspended = (cp.zeros(vector_shape, dtype=cp.float64)
                           if trace_raw_suspended_mass is None else
                           self._array("trace_raw_suspended_mass",
                                       trace_raw_suspended_mass, 3,
                                       vector_shape, copy=True))
        trace_velocity = (cp.zeros((2, nface), dtype=cp.float64)
                          if trace_base_velocity is None else
                          self._array("trace_base_velocity",
                                      trace_base_velocity, 2, (2, nface),
                                      copy=True))
        trace_depth = (cp.zeros(nface, dtype=cp.float64)
                       if trace_depth_m is None else
                       self._array("trace_depth_m", trace_depth_m, 1,
                                   (nface,), copy=True))
        trace_role = ((roles == FACE_TRACE_INLET) |
                      (roles == FACE_TRACE_OUTLET))
        nontrace = ~trace_role
        if (not self._all(cp.isfinite(trace_bed)) or
                not self._all(cp.isfinite(trace_suspended)) or
                not self._all(cp.isfinite(trace_velocity)) or
                not self._all(cp.isfinite(trace_depth)) or
                self._flag(cp.any(trace_bed[:, :, nontrace] != 0.0)) or
                self._flag(cp.any(trace_suspended[:, :, nontrace] != 0.0)) or
                self._flag(cp.any(trace_velocity[:, nontrace] != 0.0)) or
                self._flag(cp.any(trace_depth[nontrace] != 0.0)) or
                self._flag(cp.any(trace_depth[trace_role] <= 0.0))):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU trace fields are stale, nonfinite, or dry")
        if slope is not None and (
                not isinstance(slope, ResidentScalarFlowV65SlopeFields) or
                slope.device_id != self.device_id or
                slope.downhill.shape != (2, nface) or
                slope.stress.shape != (nfrac, nface) or
                slope.hiding.shape != (nfrac, ncell)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU slope field shapes differ from faces")
        digest_arrays = [
            own, other, roles, normals, length, trace_bed, trace_suspended,
            trace_velocity, trace_depth]
        if slope is not None:
            digest_arrays.extend([
                slope.downhill, slope.total_current_ust2, slope.stress,
                slope.depth, slope.d50, slope.mean_diameter, slope.hiding])
        digest = _fingerprint({
            "resident_arrays_sha256": self._digest(
                *digest_arrays, label="face_batch"),
            "slope_receipt_sha256": None if slope is None else slope.digest,
        })
        return ResidentScalarFlowV65FaceBatch(
            own, other, roles, normals, length, trace_bed, trace_suspended,
            trace_velocity, trace_depth, slope, digest, self.device_id)

    def _validate_faces(
        self, faces: ResidentScalarFlowV65FaceBatch,
        nfrac: int, ncell: int,
    ) -> None:
        cp = self.cp
        if (not isinstance(faces, ResidentScalarFlowV65FaceBatch) or
                faces.device_id != self.device_id):
            raise ScalarFlowLiveV65Error("Q3-H5/v65 GPU face batch differs")
        nface = int(faces.owner.size)
        arrays = (
            ("owner", faces.owner, 1, (nface,), True),
            ("neighbour", faces.neighbour, 1, (nface,), True),
            ("role", faces.role, 1, (nface,), True),
            ("normal", faces.normal, 2, (2, nface), False),
            ("length_m", faces.length_m, 1, (nface,), False),
            ("trace_steered_bed_mass", faces.trace_steered_bed_mass, 3,
             (2, nfrac, nface), False),
            ("trace_raw_suspended_mass", faces.trace_raw_suspended_mass, 3,
             (2, nfrac, nface), False),
            ("trace_base_velocity", faces.trace_base_velocity, 2,
             (2, nface), False),
            ("trace_depth_m", faces.trace_depth_m, 1, (nface,), False),
        )
        for name, value, ndim, shape, integer in arrays:
            self._array(name, value, ndim, shape, integer=integer)
        roles = faces.role
        internal = roles == FACE_INTERNAL
        valid_roles = (internal | (roles == FACE_TRACE_INLET) |
                       (roles == FACE_TRACE_OUTLET) |
                       (roles == FACE_OWNER_OUTLET) |
                       (roles == FACE_CLOSED))
        topology = ((faces.owner >= 0) & (faces.owner < ncell) & valid_roles &
                    cp.where(internal,
                             (faces.neighbour >= 0) &
                             (faces.neighbour < ncell) &
                             (faces.neighbour != faces.owner),
                             faces.neighbour == -1))
        trace_role = ((roles == FACE_TRACE_INLET) |
                      (roles == FACE_TRACE_OUTLET))
        nontrace = ~trace_role
        tolerance = 1024.0*np.finfo(np.float64).eps
        if (not self._all(topology) or
                not self._all(cp.isfinite(faces.normal)) or
                not self._flag(cp.allclose(
                    cp.sum(faces.normal*faces.normal, axis=0), 1.0,
                    rtol=0.0, atol=tolerance)) or
                not self._all(cp.isfinite(faces.length_m) &
                              (faces.length_m > 0.0)) or
                not self._all(cp.isfinite(faces.trace_steered_bed_mass)) or
                not self._all(cp.isfinite(faces.trace_raw_suspended_mass)) or
                not self._all(cp.isfinite(faces.trace_base_velocity)) or
                not self._all(cp.isfinite(faces.trace_depth_m)) or
                self._flag(cp.any(
                    faces.trace_steered_bed_mass[:, :, nontrace] != 0.0)) or
                self._flag(cp.any(
                    faces.trace_raw_suspended_mass[:, :, nontrace] != 0.0)) or
                self._flag(cp.any(faces.trace_base_velocity[:, nontrace] != 0.0)) or
                self._flag(cp.any(faces.trace_depth_m[nontrace] != 0.0)) or
                self._flag(cp.any(faces.trace_depth_m[trace_role] <= 0.0))):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU face topology, geometry, or trace receipt differs")
        digest_arrays = [
            faces.owner, faces.neighbour, roles, faces.normal, faces.length_m,
            faces.trace_steered_bed_mass, faces.trace_raw_suspended_mass,
            faces.trace_base_velocity, faces.trace_depth_m]
        if faces.slope is not None:
            slope = faces.slope
            if (not isinstance(slope, ResidentScalarFlowV65SlopeFields) or
                    slope.device_id != self.device_id or
                    slope.downhill.shape != (2, nface) or
                    slope.stress.shape != (nfrac, nface) or
                    slope.hiding.shape != (nfrac, ncell)):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU slope field shapes differ from faces")
            slope_arrays = [
                slope.downhill, slope.total_current_ust2, slope.stress,
                slope.depth, slope.d50, slope.mean_diameter, slope.hiding]
            slope_digest = _fingerprint({
                "producer_execution_fingerprint": (
                    slope.ust2_provenance_fingerprint),
                "source_production_sha256": slope.source_production_sha256,
                "stress_semantics": slope.stress_semantics,
                "resident_arrays_sha256": self._digest(
                    *slope_arrays, label="slope_fields"),
            })
            if slope.digest != slope_digest:
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU slope fields are stale or mutated")
            digest_arrays.extend(slope_arrays)
        expected_face_digest = _fingerprint({
            "resident_arrays_sha256": self._digest(
                *digest_arrays, label="face_batch"),
            "slope_receipt_sha256": (
                None if faces.slope is None else faces.slope.digest),
        })
        if faces.digest != expected_face_digest:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU face batch is stale or mutated")

    def _compose_trace(self, faces: ResidentScalarFlowV65FaceBatch):
        cp = self.cp
        trace_role = ((faces.role == FACE_TRACE_INLET) |
                      (faces.role == FACE_TRACE_OUTLET))
        safe_depth = cp.ascontiguousarray(cp.where(
            trace_role, faces.trace_depth_m, 1.0))
        result = self.channel_composer.compose(
            raw_bed_mass=faces.trace_steered_bed_mass,
            raw_suspended_mass=faces.trace_raw_suspended_mass,
            base_velocity=faces.trace_base_velocity,
            depth=safe_depth,
            transport_class=self.transport_class,
            bed_factor=self.profile.bed_factor,
            suspended_factor=self.profile.suspended_factor,
            speed_epsilon=self.profile.epsilon,
            reject_invalid=True)
        return result.composed_bed_mass, result.retained_suspended_mass

    def _validate_channels(self, channels: ResidentScalarFlowV65CellChannels,
                           nfrac: int, ncell: int) -> None:
        shape = (2, nfrac, ncell)
        if (not isinstance(channels, ResidentScalarFlowV65CellChannels) or
                channels.device_id != self.device_id or
                channels.execution_fingerprint != self.execution["fingerprint"] or
                channels.producer_execution_fingerprint !=
                self.producer_execution["fingerprint"] or
                channels.composed_bed_mass.shape != shape or
                channels.retained_suspended_mass.shape != shape or
                channels.signed_suspended_mass_rate.shape != (nfrac, ncell) or
                channels.equilibrium_concentration.shape != (nfrac, ncell) or
                channels.total_transport_direction.shape != shape):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU cell channel shapes differ")
        expected = self._digest(
            channels.composed_bed_mass, channels.retained_suspended_mass,
            channels.signed_suspended_mass_rate,
            channels.equilibrium_concentration,
            channels.total_transport_direction, label="cell_channels")
        if expected != channels.digest:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU cell channels are stale or mutated")

    def _validate_production(
        self, production: ResidentScalarFlowV65Production,
        nfrac: int, ncell: int,
    ) -> None:
        cp = self.cp
        if (not isinstance(production, ResidentScalarFlowV65Production) or
                production.device_id != self.device_id or
                production.producer_execution_fingerprint !=
                self.producer_execution["fingerprint"] or
                not isinstance(production.producer_inputs,
                               ResidentScalarFlowInputsV64) or
                not isinstance(production.capacity,
                               ResidentScalarFlowCapacityV64) or
                production.capacity.selector !=
                int(self.execution["reference_selector"]) or
                production.capacity.active_fraction_applied is not False):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU live proposal requires its v64 production receipt")
        self._validate_channels(production.channels, nfrac, ncell)
        produced = production.producer_inputs
        capacity = production.capacity
        if (not self._all(produced.active == 1) or
                not self._all(produced.hydraulic.raw_depth > 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU production is not fully wet/fixed-active")
        token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "producer_execution_fingerprint": self.producer_execution["fingerprint"],
            "selector": int(capacity.selector),
            "producer_arrays_sha256": self._digest(
                produced.hydraulic.values, produced.mixture.values,
                produced.mixture.fraction, produced.mixture.resolved_d50,
                produced.kinematic_viscosity, produced.solid_density,
                produced.rp_taub, produced.active, capacity.raw_values,
                capacity.bed_mass_pre_secondary, capacity.bed_mass,
                capacity.suspended_mass, capacity.slope_total_current_ust2,
                label="v64_production"),
            "channel_receipt_sha256": production.channels.digest,
        })
        if token != production.production_sha256:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU v64 production receipt is stale or mutated")

    def _begin_bed_exner_trial_impl(
        self, *, state: ResidentScalarFlowV65State,
        channels: ResidentScalarFlowV65CellChannels,
        faces: ResidentScalarFlowV65FaceBatch,
        cell_area_m2: object, dt_s: float,
        morphological_factor: float = 1.0,
        lateral_stencil: ResidentScalarFlowV65LateralStencil | None = None,
        cancelled=lambda: False,
        source_production_sha256: str | None = None,
    ) -> ResidentScalarFlowV65Trial:
        cp = self.cp
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if cancelled():
            raise InterruptedError("Q3-H5/v65 GPU transaction cancelled before launch")
        self._validate_state(state)
        if state.accepted_step >= int(self.execution["accepted_step_limit"]):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU bounded accepted-step limit has been reached")
        classes = [row.transport_class for row in self.profile.fractions]
        if any(value == TRA_COMBINE for value in classes):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU class 3 is diagnostic-only until Q4 owns "
                "water-column mass and exchange")
        if any(value == TRA_ADVDIFF for value in classes) or any(
                value != TRA_BEDLOAD for value in classes):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU live transaction accepts class 1 only")
        nfrac, ncell = map(int, state.bed_mass_kg_m2.shape)
        self._validate_channels(channels, nfrac, ncell)
        self._validate_faces(faces, nfrac, ncell)
        nface = int(faces.owner.size)
        area = self._array("cell_area_m2", cell_area_m2, 1, (ncell,))
        if not self._all(cp.isfinite(area) & (area > 0.0)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU cell area must be finite positive")
        dt = _finite(dt_s, "trial dt", positive=True)
        morfac = _finite(
            morphological_factor, "morphological factor", nonnegative=True)

        if self.execution["laterally_averaged_bedload"]:
            if (not isinstance(lateral_stencil,
                               ResidentScalarFlowV65LateralStencil) or
                    lateral_stencil.ncell != ncell or
                    lateral_stencil.fingerprint !=
                    self.execution["dependencies"]["lateral_stencil"]):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU requires its authenticated 4-2-1 stencil")
            cell_bed = self.apply_lateral_stencil(
                channels.composed_bed_mass, lateral_stencil)
        else:
            if lateral_stencil is not None:
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU lateral stencil supplied while control is off")
            cell_bed = channels.composed_bed_mass

        trace_bed, trace_suspended = self._compose_trace(faces)
        # A nonzero retained trace channel is legitimate component evidence,
        # but class 3 was rejected above before a bed candidate can be formed.
        del trace_suspended
        face_shape = (2, nfrac, nface)
        scalar_face_shape = (nfrac, nface)
        raw = cp.zeros(face_shape, dtype=cp.float64)
        raw_donor = cp.zeros(scalar_face_shape, dtype=cp.int32)
        count = nfrac*nface
        self.kernels["scalar_v65_face_candidates"](
            ((count+255)//256,), (256,),
            (np.int64(ncell), np.int32(nfrac), np.int64(nface),
             np.int32(bool(self.execution["upwind_bedload"])),
             faces.owner, faces.neighbour, faces.role, faces.normal,
             cell_bed, channels.total_transport_direction, trace_bed,
             raw, raw_donor))

        slope_method = self.execution["bed_slope_method_id"]
        if slope_method == SLOPE_OFF:
            if faces.slope is not None:
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU slope-off faces may not carry slope fields")
            corrected = raw.copy()
        else:
            fields = faces.slope
            if (not isinstance(fields, ResidentScalarFlowV65SlopeFields) or
                    fields.ust2_provenance_fingerprint !=
                    self.execution["dependencies"]["backend_producer_execution"]):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU active slope lacks v64 USTAR-squared provenance")
            if slope_method == SLOPE_BAGNOLD_IKEDA:
                if (fields.stress_semantics != TAURAT_POSITIVE_ZERO or
                        self._flag(cp.any(fields.stress != 0.0))):
                    raise ScalarFlowLiveV65Error(
                        "Q3-H5/v65 GPU positive-TraFrm ISlope 2 taurat differs")
            elif fields.stress_semantics != (
                    "absolute_shields_from_total_current_RP_USTAR_squared"):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU ISlope 3/4 absolute-Shields receipt differs")
            if (source_production_sha256 is not None and
                    fields.source_production_sha256 !=
                    source_production_sha256):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU live slope is not derived from this production")
            assert self.slope_composer is not None
            # The accepted H4 face composer currently exposes class D50, not a
            # face-varying D50 ABI.  Reject that implementation gap explicitly.
            if not self._flag(cp.all(fields.d50 == fields.d50[:, :1])):
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU active slope currently requires face-constant "
                    "class D50 in ResidentFlowBedSlopeFaceComposer")
            h4_role = cp.ascontiguousarray(cp.where(
                faces.role == FACE_CLOSED, FACE_INTERNAL, faces.role).astype(
                    cp.int32, copy=False))
            ones = cp.ones((nfrac, ncell), dtype=cp.float64)
            slope_result = self.slope_composer.compose(
                left_bed=raw, right_bed=raw, owner_bed=raw, trace_bed=raw,
                owner=faces.owner, neighbour=faces.neighbour, role=h4_role,
                normal=faces.normal, availability=ones,
                downhill=fields.downhill, stress=fields.stress,
                depth=fields.depth,
                d50=cp.ascontiguousarray(fields.d50[:, 0]),
                mean_diameter=fields.mean_diameter, hiding=fields.hiding,
                reject_invalid=True)
            if slope_result.backend != "implicit_fwave_midpoint_gpu_component":
                raise ScalarFlowLiveV65Error(
                    "Q3-H5/v65 GPU slope component changed backend")
            corrected = cp.ascontiguousarray(slope_result.corrected_transport)

        potential = cp.zeros(scalar_face_shape, dtype=cp.float64)
        donor = cp.zeros(scalar_face_shape, dtype=cp.int32)
        donor_cell = cp.full(scalar_face_shape, -1, dtype=cp.int32)
        self.kernels["scalar_v65_face_potential"](
            ((count+255)//256,), (256,),
            (np.int32(nfrac), np.int64(nface), faces.owner,
             faces.neighbour, faces.role, faces.normal, corrected,
             potential, donor, donor_cell))
        if not self._all(cp.isfinite(raw) & cp.isfinite(corrected)) or \
                not self._all(cp.isfinite(potential)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU face reconstruction produced nonfinite transport")

        outgoing = cp.zeros((nfrac, ncell), dtype=cp.float64)
        self.kernels["scalar_v65_inventory_demand"](
            ((count+255)//256,), (256,),
            (np.int32(nfrac), np.int64(ncell), np.int64(nface),
             np.float64(dt), np.float64(morfac), faces.role, donor_cell,
             potential, state.active_fraction, faces.length_m, area, outgoing))
        fill = cp.empty_like(outgoing)
        availability = cp.empty_like(outgoing)
        cell_count = nfrac*ncell
        self.kernels["scalar_v65_inventory_fill"](
            ((cell_count+255)//256,), (256,),
            (np.int64(cell_count), state.bed_mass_kg_m2,
             state.active_fraction, outgoing, fill, availability))

        flux = cp.zeros(scalar_face_shape, dtype=cp.float64)
        delta = cp.zeros((nfrac, ncell), dtype=cp.float64)
        boundary = cp.zeros(nfrac, dtype=cp.float64)
        boundary_absolute = cp.zeros(nfrac, dtype=cp.float64)
        self.kernels["scalar_v65_atomic_exner"](
            ((count+255)//256,), (256,),
            (np.int32(nfrac), np.int64(ncell), np.int64(nface),
             np.float64(dt), np.float64(morfac), faces.owner,
             faces.neighbour, faces.role, donor_cell, potential,
             availability, faces.length_m, area, flux, delta, boundary,
             boundary_absolute))
        candidate_raw = state.bed_mass_kg_m2+delta
        tolerance = (2048.0*np.finfo(np.float64).eps*
                     cp.maximum(1.0, cp.abs(state.bed_mass_kg_m2)))
        if self._flag(cp.any(candidate_raw < -tolerance)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU Exner candidate exceeds finite inventory")
        candidate_mass = cp.ascontiguousarray(cp.maximum(candidate_raw, 0.0))
        clamped_mass = candidate_mass.copy()
        numerically_depleted = (
            cp.sum(candidate_mass, axis=0) <= cp.sum(tolerance, axis=0))
        candidate_mass[:, numerically_depleted] = 0.0
        depletion_roundoff = cp.ascontiguousarray(cp.sum(
            (clamped_mass-candidate_mass)*area[None, :], axis=1))
        delta = cp.ascontiguousarray(delta)
        delta[:, numerically_depleted] = (
            candidate_mass[:, numerically_depleted]-
            state.bed_mass_kg_m2[:, numerically_depleted])
        candidate_volume = candidate_mass/state.bulk_density_kg_m3[:, None]
        candidate_total = cp.sum(candidate_volume, axis=0)
        candidate_positive = candidate_total > 0.0
        candidate_fraction = cp.ascontiguousarray(cp.where(
            candidate_positive[None, :],
            candidate_volume/cp.where(
                candidate_total[None, :] > 0.0,
                candidate_total[None, :], 1.0),
            state.active_fraction))
        residual = cp.ascontiguousarray(
            cp.sum(delta*area[None, :], axis=1)+boundary+
            depletion_roundoff)
        residual_tolerance = (4096.0*np.finfo(np.float64).eps*cp.maximum(
            1.0, cp.sum(cp.abs(delta*area[None, :]), axis=1)+cp.abs(boundary)))
        if self._flag(cp.any(cp.abs(residual) > residual_tolerance)):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU conservative Exner ledger does not close")

        candidate = self.resolve_state(
            bed_mass_kg_m2=candidate_mass,
            active_fraction=candidate_fraction,
            boundary_signed_kg=cp.ascontiguousarray(
                state.boundary_signed_kg+boundary+depletion_roundoff),
            boundary_absolute_kg=cp.ascontiguousarray(
                state.boundary_absolute_kg+boundary_absolute+
                depletion_roundoff),
            accepted_step=state.accepted_step+1,
            time_s=state.time_s+dt)
        active_factor = state.active_fraction.copy()
        arrays = (
            raw, corrected, raw_donor, donor, active_factor, fill,
            availability, flux, delta, boundary, depletion_roundoff, residual)
        trial_token = _fingerprint({
            "execution_fingerprint": self.execution["fingerprint"],
            "parent_state_sha256": state.state_sha256,
            "candidate_state_sha256": candidate.state_sha256,
            "dt_s": dt, "morphological_factor": morfac,
            "resident_arrays_sha256": self._digest(
                *arrays, label="bed_exner_trial"),
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })
        trial = ResidentScalarFlowV65Trial(
            self.execution["fingerprint"], state.state_sha256, dt, morfac,
            raw, corrected, raw_donor, donor, active_factor, fill,
            availability, flux, delta, candidate, boundary,
            depletion_roundoff, residual,
            1, 1, trial_token, self.device_id)
        if cancelled():
            raise InterruptedError(
                "Q3-H5/v65 GPU transaction cancelled before publication")
        return trial

    def begin_component_bed_exner_trial(
        self, *, state: ResidentScalarFlowV65State,
        channels: ResidentScalarFlowV65CellChannels,
        faces: ResidentScalarFlowV65FaceBatch,
        cell_area_m2: object, dt_s: float,
        morphological_factor: float = 1.0,
        lateral_stencil: ResidentScalarFlowV65LateralStencil | None = None,
        cancelled=lambda: False,
    ) -> ResidentScalarFlowV65Trial:
        """Raw-channel proposal reserved for explicit component/oracle tests."""
        return self._begin_bed_exner_trial_impl(
            state=state, channels=channels, faces=faces,
            cell_area_m2=cell_area_m2, dt_s=dt_s,
            morphological_factor=morphological_factor,
            lateral_stencil=lateral_stencil, cancelled=cancelled,
            source_production_sha256=None)

    def begin_bed_exner_trial(
        self, *, state: ResidentScalarFlowV65State,
        production: ResidentScalarFlowV65Production,
        faces: ResidentScalarFlowV65FaceBatch,
        cell_area_m2: object, dt_s: float,
        morphological_factor: float = 1.0,
        lateral_stencil: ResidentScalarFlowV65LateralStencil | None = None,
        cancelled=lambda: False,
    ) -> ResidentScalarFlowV65Trial:
        """Public live proposal requiring the selected resident v64 result."""
        self._validate_state(state)
        nfrac, ncell = map(int, state.bed_mass_kg_m2.shape)
        self._validate_production(production, nfrac, ncell)
        return self._begin_bed_exner_trial_impl(
            state=state, channels=production.channels, faces=faces,
            cell_area_m2=cell_area_m2, dt_s=dt_s,
            morphological_factor=morphological_factor,
            lateral_stencil=lateral_stencil, cancelled=cancelled,
            source_production_sha256=production.production_sha256)

    def _validate_trial(self, state: ResidentScalarFlowV65State,
                        trial: ResidentScalarFlowV65Trial) -> None:
        self._validate_state(state)
        if (not isinstance(trial, ResidentScalarFlowV65Trial) or
                trial.device_id != self.device_id or
                trial.parent_state_sha256 != state.state_sha256 or
                trial.execution_fingerprint != state.execution_fingerprint or
                trial.active_fraction_application_count != 1 or
                trial.inventory_fill_application_count != 1):
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU trial parent or once-only counters differ")
        arrays = (
            trial.face_raw_candidate, trial.face_corrected_candidate,
            trial.raw_donor, trial.donor, trial.active_fraction_factor,
            trial.inventory_fill, trial.availability,
            trial.face_flux_kg_m_s, trial.cell_delta_kg_m2,
            trial.boundary_transfer_kg,
            trial.depletion_roundoff_transfer_kg,
            trial.conservation_residual_kg)
        token = _fingerprint({
            "execution_fingerprint": trial.execution_fingerprint,
            "parent_state_sha256": trial.parent_state_sha256,
            "candidate_state_sha256": trial.candidate.state_sha256,
            "dt_s": trial.dt_s,
            "morphological_factor": trial.morphological_factor,
            "resident_arrays_sha256": self._digest(
                *arrays, label="bed_exner_trial"),
            "active_fraction_application_count": 1,
            "inventory_fill_application_count": 1,
        })
        if token != trial.trial_sha256:
            raise ScalarFlowLiveV65Error(
                "Q3-H5/v65 GPU trial is stale or mutated")
        self._validate_state(trial.candidate)

    def commit_trial(self, state: ResidentScalarFlowV65State,
                     trial: ResidentScalarFlowV65Trial) -> ResidentScalarFlowV65State:
        self._validate_trial(state, trial)
        return trial.candidate

    def rollback_trial(self, state: ResidentScalarFlowV65State,
                       trial: ResidentScalarFlowV65Trial) -> ResidentScalarFlowV65State:
        self._validate_trial(state, trial)
        return state

    def cancel_trial(self, state: ResidentScalarFlowV65State,
                     trial: ResidentScalarFlowV65Trial) -> ResidentScalarFlowV65State:
        self._validate_trial(state, trial)
        return state

    def state_checkpoint_document(
        self, state: ResidentScalarFlowV65State,
    ) -> dict[str, object]:
        """Create the canonical portable checkpoint (the allowed host boundary)."""
        self._validate_state(state)
        cp = self.cp
        portable = resolve_cpu_state_v65(
            execution=self.execution,
            bed_mass_kg_m2=cp.asnumpy(state.bed_mass_kg_m2),
            active_fraction=cp.asnumpy(state.active_fraction),
            boundary_signed_kg=cp.asnumpy(state.boundary_signed_kg),
            boundary_absolute_kg=cp.asnumpy(state.boundary_absolute_kg),
            accepted_step=state.accepted_step, time_s=state.time_s)
        return cpu_state_checkpoint_document_v65(portable)

    def load_state_checkpoint(
        self, value: Mapping[str, object],
    ) -> ResidentScalarFlowV65State:
        """Validate a canonical checkpoint on host, then restore resident state."""
        cp = self.cp
        portable = load_cpu_state_checkpoint_v65(
            value, execution=self.execution)
        with cp.cuda.Device(self.device_id):
            return self.resolve_state(
                bed_mass_kg_m2=cp.ascontiguousarray(cp.asarray(
                    portable.bed_mass_kg_m2, dtype=cp.float64)),
                active_fraction=cp.ascontiguousarray(cp.asarray(
                    portable.active_fraction, dtype=cp.float64)),
                boundary_signed_kg=cp.ascontiguousarray(cp.asarray(
                    portable.boundary_signed_kg, dtype=cp.float64)),
                boundary_absolute_kg=cp.ascontiguousarray(cp.asarray(
                    portable.boundary_absolute_kg, dtype=cp.float64)),
                accepted_step=portable.accepted_step,
                time_s=portable.time_s)


# Thin functional mirrors keep call sites close to the canonical CPU API while
# making the explicitly selected resident runtime impossible to omit.
def resolve_lateral_stencil_v65(*, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.resolve_lateral_stencil(**values)


def apply_lateral_stencil_v65(bed_mass, stencil, *,
                              runtime: ResidentScalarFlowLiveV65):
    return runtime.apply_lateral_stencil(bed_mass, stencil)


def resolve_state_v65(*, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.resolve_state(**values)


def compose_cell_channels_v65(*, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.compose_cell_channels(**values)


def resolve_slope_fields_v65(*, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.resolve_slope_fields(**values)


def resolve_slope_fields_from_v64_v65(
        *, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.resolve_slope_fields_from_v64(**values)


def resolve_face_batch_v65(*, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.resolve_face_batch(**values)


def begin_bed_exner_trial_v65(*, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.begin_bed_exner_trial(**values)


def begin_component_bed_exner_trial_v65(
        *, runtime: ResidentScalarFlowLiveV65, **values):
    return runtime.begin_component_bed_exner_trial(**values)


def commit_trial_v65(state, trial, *, runtime: ResidentScalarFlowLiveV65):
    return runtime.commit_trial(state, trial)


def rollback_trial_v65(state, trial, *, runtime: ResidentScalarFlowLiveV65):
    return runtime.rollback_trial(state, trial)


def cancel_trial_v65(state, trial, *, runtime: ResidentScalarFlowLiveV65):
    return runtime.cancel_trial(state, trial)


def state_checkpoint_document_v65(state, *, runtime: ResidentScalarFlowLiveV65):
    return runtime.state_checkpoint_document(state)


def load_state_checkpoint_v65(value, *, runtime: ResidentScalarFlowLiveV65):
    return runtime.load_state_checkpoint(value)


__all__ = [
    "BACKEND", "INTEGRATOR", "ResidentScalarFlowLiveV65",
    "ResidentScalarFlowV65CellChannels", "ResidentScalarFlowV65FaceBatch",
    "ResidentScalarFlowV65LateralStencil", "ResidentScalarFlowV65Production",
    "ResidentScalarFlowV65SlopeFields", "ResidentScalarFlowV65State",
    "ResidentScalarFlowV65Trial", "apply_lateral_stencil_v65",
    "begin_bed_exner_trial_v65", "begin_component_bed_exner_trial_v65",
    "cancel_trial_v65", "commit_trial_v65",
    "compose_cell_channels_v65", "load_state_checkpoint_v65",
    "resolve_face_batch_v65", "resolve_lateral_stencil_v65",
    "resolve_slope_fields_from_v64_v65", "resolve_slope_fields_v65",
    "resolve_state_v65", "rollback_trial_v65", "state_checkpoint_document_v65",
]
