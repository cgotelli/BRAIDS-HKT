"""Resident CUDA adapter for four current-only Delft direct producers.

The adapter invokes the established v66 constitutive kernels, normalizes
their heterogeneous outputs to one five-channel ABI, and delegates all
Bed/Sus/fraction ownership to the generic resident route composer.  Physical
arrays never leave the selected device and there is no CPU fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from braids_core.sediment.direct_producer_adapter_v67 import (
    FRACTION_OWNER_DOWNSTREAM_FACE_EXNER,
    FRACTION_OWNER_ROUTE_COMPOSER,
    METHOD_VAN_RIJN_1993_CURRENT,
    DirectProducerContractV67,
    direct_producer_contract_v67,
)
from braids_core.sediment.direct_vector_route_composer_v66 import (
    ROUTE_TRA_BEDLOAD,
    ROUTE_TRA_COMBINE,
)
from .delft_direct_current_v66 import DelftDirectCurrentV66CUDA
from .direct_vector_route_composer_v66 import DirectVectorRouteComposerV66CUDA
from .van_rijn_1993_route_v66 import (
    CELL_FIELDS as VR_CELL_FIELDS,
    CLASS_FIELDS as VR_CLASS_FIELDS,
    POINT_FIELDS as VR_POINT_FIELDS,
    ROUTE_BEDLOAD as VR_ROUTE_BEDLOAD,
    ROUTE_COMBINE as VR_ROUTE_COMBINE,
    ResidentVanRijn1993RouteV66,
)


ROUTE_TO_VR = {
    ROUTE_TRA_BEDLOAD: VR_ROUTE_BEDLOAD,
    ROUTE_TRA_COMBINE: VR_ROUTE_COMBINE,
}
_VR_OWNED_ROUTE_FIELDS = frozenset((
    "bed_factor", "suspended_factor", "surface_fraction",
))
VR_PRODUCER_FIELDS = tuple(
    name for name in VR_CELL_FIELDS + VR_CLASS_FIELDS + VR_POINT_FIELDS
    if name not in _VR_OWNED_ROUTE_FIELDS
)


@dataclass(frozen=True)
class ResidentDirectProducerAdapterResultV67:
    """A device-resident normalized producer and composed route result."""

    method_id: str
    internal_wire_id: int
    route: str
    point_shape: tuple[int, ...]
    raw_values: object
    composed_values: object
    producer_auxiliary: object
    producer_status: object
    route_status: object
    surface_fraction_application_count: int
    surface_fraction_owner: str

    @property
    def raw_bed_mass(self):
        return self.raw_values[0:2]

    @property
    def raw_suspended_mass(self):
        return self.raw_values[2:4]

    @property
    def raw_equilibrium_mass_concentration(self):
        return self.raw_values[4]

    @property
    def bed_mass(self):
        return self.composed_values[0:2]

    @property
    def q4_equilibrium_mass_concentration(self):
        return self.composed_values[2]

    def require_deferred_surface_fraction(self) -> None:
        """Fail if these channels were fractioned before face/Exner use."""
        if (
            self.surface_fraction_application_count != 0
            or self.surface_fraction_owner
            != FRACTION_OWNER_DOWNSTREAM_FACE_EXNER
        ):
            raise ValueError(
                "face/Exner lifecycle requires an authenticated deferred "
                "surface fraction"
            )


class DirectProducerAdapterV67CUDA:
    """Normalize one explicitly selected current-only producer on CUDA."""

    def __init__(self, method: str | int, route: str):
        if type(route) is not str or route not in ROUTE_TO_VR:
            raise ValueError(
                "direct-producer v67 route must be 'tra_bedload' or "
                "'tra_combine'"
            )
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        self.contract: DirectProducerContractV67 = direct_producer_contract_v67(
            method
        )
        self.route = route
        self.composer = DirectVectorRouteComposerV66CUDA()
        if self.contract.method_id == METHOD_VAN_RIJN_1993_CURRENT:
            self.vr1993 = ResidentVanRijn1993RouteV66(ROUTE_TO_VR[route])
            self.direct = None
        else:
            self.vr1993 = None
            self.direct = DelftDirectCurrentV66CUDA()
        for component in (self.composer, self.vr1993, self.direct):
            if component is not None and component.device_id != self.device_id:
                raise RuntimeError("v67 CUDA components selected different devices")

    def _array(self, name: str, value, shape: tuple[int, ...]):
        cp = self.cp
        if (
            not isinstance(value, cp.ndarray)
            or value.dtype != cp.float64
            or tuple(value.shape) != shape
            or not value.flags.c_contiguous
        ):
            raise TypeError(
                f"{name} must be a resident contiguous FP64 CuPy array "
                f"with shape {shape}"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        return value

    @staticmethod
    def _mapping(fields: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(fields, Mapping):
            raise TypeError("producer_fields must be a mapping")
        if any(type(name) is not str for name in fields):
            raise TypeError("producer field names must be exact strings")
        return dict(fields)

    def _raise_if_invalid(self, producer_status, route_status) -> None:
        cp = self.cp
        producer_bad = int(cp.count_nonzero(producer_status).item())
        route_bad = int(cp.count_nonzero(route_status).item())
        if producer_bad or route_bad:
            raise ValueError(
                f"CUDA direct-producer v67 rejected state(s): "
                f"producer={producer_bad}, route={route_bad}"
            )

    def _evaluate_direct(
        self,
        fields: dict[str, object],
        bed_factor,
        suspended_factor,
        active_fraction,
        speed_epsilon,
    ) -> ResidentDirectProducerAdapterResultV67:
        cp = self.cp
        assert self.direct is not None
        selector = self.contract.reference_selector
        produced = self.direct.evaluate(selector, reject_invalid=False, **fields)
        count = int(produced.status.size)
        bed_factor = self._array("bed_factor", bed_factor, (count,))
        suspended_factor = self._array(
            "suspended_factor", suspended_factor, (count,)
        )
        active_fraction = self._array(
            "active_fraction", active_fraction, (count,)
        )
        speed_epsilon = self._array("speed_epsilon", speed_epsilon, (count,))

        # Bijker's concentration row is diagnostic.  Formulae 19/20 own an
        # equilibrium concentration and their explicit suspended rows are
        # exactly zero.  The normalized ABI cannot expose both owners.
        if selector == 5:
            raw_suspended = produced.suspended_mass
            raw_equilibrium = cp.zeros(count, dtype=cp.float64)
        else:
            raw_suspended = produced.suspended_mass
            raw_equilibrium = produced.concentration_mass
        raw_values = cp.concatenate(
            (produced.bed_mass, raw_suspended,
             raw_equilibrium.reshape(1, count)), axis=0,
        )
        velocity = cp.stack((fields["u"], fields["v"]), axis=0)
        composed = self.composer.compose(
            method=self.contract.composer_method,
            route=self.route,
            depth=fields["depth"],
            velocity=velocity,
            raw_bed_mass=raw_values[0:2],
            raw_suspended_mass=raw_values[2:4],
            raw_equilibrium_mass_concentration=raw_values[4],
            bed_factor=bed_factor,
            suspended_factor=suspended_factor,
            active_fraction=active_fraction,
            speed_epsilon=speed_epsilon,
            reject_invalid=False,
        )
        return ResidentDirectProducerAdapterResultV67(
            self.contract.method_id,
            self.contract.internal_wire_id,
            self.route,
            (count,),
            raw_values,
            composed.values,
            produced.auxiliary,
            produced.status,
            composed.status,
            1,
            FRACTION_OWNER_ROUTE_COMPOSER,
        )

    def _evaluate_vr1993(
        self,
        fields: dict[str, object],
        bed_factor,
        suspended_factor,
        active_fraction,
        speed_epsilon,
    ) -> ResidentDirectProducerAdapterResultV67:
        cp = self.cp
        assert self.vr1993 is not None
        expected = set(VR_PRODUCER_FIELDS)
        missing = tuple(sorted(expected-set(fields)))
        extra = tuple(sorted(set(fields)-expected))
        if missing or extra:
            raise ValueError(
                "Van Rijn 1993 producer fields differ; "
                f"missing={missing}, extra={extra}"
            )
        depth = fields["depth"]
        d50 = fields["d50"]
        if not isinstance(depth, cp.ndarray) or depth.ndim != 1:
            raise TypeError("depth must be a resident one-dimensional array")
        if not isinstance(d50, cp.ndarray) or d50.ndim != 1:
            raise TypeError("d50 must be a resident one-dimensional array")
        ncell = int(depth.size)
        nclass = int(d50.size)
        bed_factor = self._array("bed_factor", bed_factor, (nclass,))
        suspended_factor = self._array(
            "suspended_factor", suspended_factor, (nclass,)
        )
        active_fraction = self._array(
            "active_fraction", active_fraction, (nclass, ncell)
        )
        speed_epsilon = self._array(
            "speed_epsilon", speed_epsilon, (nclass, ncell)
        )
        producer_inputs = {
            **fields,
            "bed_factor": bed_factor,
            "suspended_factor": suspended_factor,
            "surface_fraction": active_fraction,
        }
        produced = self.vr1993.evaluate(
            reject_invalid=False, **producer_inputs
        )
        count = nclass*ncell
        raw_bed = produced.raw_bed_mass.reshape(2, count)
        raw_suspended = cp.zeros((2, count), dtype=cp.float64)
        raw_equilibrium = produced.raw_conc2d.reshape(count)
        raw_values = cp.concatenate(
            (raw_bed, raw_suspended, raw_equilibrium.reshape(1, count)),
            axis=0,
        )
        velocity = cp.ascontiguousarray(
            cp.broadcast_to(
                fields["velocity"][:, None, :], (2, nclass, ncell)
            ).reshape(2, count)
        )
        depth_points = cp.ascontiguousarray(
            cp.broadcast_to(depth[None, :], (nclass, ncell)).reshape(count)
        )
        bed_points = cp.ascontiguousarray(
            cp.broadcast_to(bed_factor[:, None], (nclass, ncell)).reshape(count)
        )
        suspended_points = cp.ascontiguousarray(
            cp.broadcast_to(
                suspended_factor[:, None], (nclass, ncell)
            ).reshape(count)
        )
        composed = self.composer.compose(
            method=self.contract.composer_method,
            route=self.route,
            depth=depth_points,
            velocity=velocity,
            raw_bed_mass=raw_values[0:2],
            raw_suspended_mass=raw_values[2:4],
            raw_equilibrium_mass_concentration=raw_values[4],
            bed_factor=bed_points,
            suspended_factor=suspended_points,
            active_fraction=active_fraction.reshape(count),
            speed_epsilon=speed_epsilon.reshape(count),
            reject_invalid=False,
        )
        return ResidentDirectProducerAdapterResultV67(
            self.contract.method_id,
            self.contract.internal_wire_id,
            self.route,
            (nclass, ncell),
            raw_values,
            composed.values,
            produced.diagnostics.reshape(6, count),
            produced.status.reshape(count),
            composed.status,
            1,
            FRACTION_OWNER_ROUTE_COMPOSER,
        )

    def evaluate(
        self,
        *,
        producer_fields: Mapping[str, object],
        bed_factor,
        suspended_factor,
        active_fraction,
        speed_epsilon,
        reject_invalid: bool = True,
    ) -> ResidentDirectProducerAdapterResultV67:
        """Run the selected producer and the shared route composer."""
        fields = self._mapping(producer_fields)
        if self.contract.method_id == METHOD_VAN_RIJN_1993_CURRENT:
            result = self._evaluate_vr1993(
                fields, bed_factor, suspended_factor,
                active_fraction, speed_epsilon,
            )
        else:
            result = self._evaluate_direct(
                fields, bed_factor, suspended_factor,
                active_fraction, speed_epsilon,
            )
        if reject_invalid:
            self._raise_if_invalid(result.producer_status, result.route_status)
        return result

    def evaluate_for_face_lifecycle(
        self,
        *,
        producer_fields: Mapping[str, object],
        bed_factor,
        suspended_factor,
        speed_epsilon,
        reject_invalid: bool = True,
    ) -> ResidentDirectProducerAdapterResultV67:
        """Return unfractioned channels owned by the downstream face update.

        The API has no caller-provided active-fraction argument.  Exact ones
        are allocated internally on the authenticated CUDA device, preventing
        a caller from accidentally pre-fractioning channels that the face and
        Exner lifecycle must fraction exactly once later.
        """
        fields = self._mapping(producer_fields)
        cp = self.cp
        with cp.cuda.Device(self.device_id):
            if self.contract.method_id == METHOD_VAN_RIJN_1993_CURRENT:
                depth = fields.get("depth")
                d50 = fields.get("d50")
                if not isinstance(depth, cp.ndarray) or depth.ndim != 1:
                    raise TypeError(
                        "depth must be a resident one-dimensional array"
                    )
                if not isinstance(d50, cp.ndarray) or d50.ndim != 1:
                    raise TypeError(
                        "d50 must be a resident one-dimensional array"
                    )
                active_fraction = cp.ones(
                    (int(d50.size), int(depth.size)), dtype=cp.float64
                )
            else:
                depth = fields.get("depth")
                if not isinstance(depth, cp.ndarray) or depth.ndim != 1:
                    raise TypeError(
                        "depth must be a resident one-dimensional array"
                    )
                active_fraction = cp.ones(int(depth.size), dtype=cp.float64)
        result = self.evaluate(
            producer_fields=fields,
            bed_factor=bed_factor,
            suspended_factor=suspended_factor,
            active_fraction=active_fraction,
            speed_epsilon=speed_epsilon,
            reject_invalid=reject_invalid,
        )
        return replace(
            result,
            surface_fraction_application_count=0,
            surface_fraction_owner=FRACTION_OWNER_DOWNSTREAM_FACE_EXNER,
        )


__all__ = [
    "DirectProducerAdapterV67CUDA", "ROUTE_TO_VR",
    "ResidentDirectProducerAdapterResultV67", "VR_PRODUCER_FIELDS",
]
