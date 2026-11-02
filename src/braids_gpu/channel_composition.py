"""Resident CUDA composition of already mass-valued transport channels.

This is a bounded post-capacity component, not a live transport route.  It
matches the relevant locked Delft3D4 ``eqtran`` ordering:

* density conversion has already happened upstream;
* the global bed and suspended factors are each applied exactly once;
* any secondary-flow steering has already rotated the bed channel only;
* suspended transport is signed along the unrotated base-flow direction;
* class 1 folds suspended transport into bed transport, while class 3 retains
  it as a pre-active-fraction equilibrium mass concentration for
  advection--diffusion.

Locked transport class 2 is deliberately rejected.  FLOW routes that class
through ``erosilt`` and exits the fraction loop before calling ``eqtran``, so
there is no honest raw-channel composition to emulate here.  There is no CPU
fallback and physical arrays stay resident on the selected CUDA device.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from braids_core.sediment.channel_composition import (
    TRA_ADVDIFF,
    TRA_BEDLOAD,
    TRA_COMBINE,
)

STATUS_OK = 0
STATUS_NONFINITE = 1
STATUS_UNSUPPORTED_CLASS = 2
STATUS_ADVDIFF_REQUIRES_EROSILT = 3
STATUS_NONPOSITIVE_COMBINED_DEPTH = 4
STATUS_SUSPENDED_MISALIGNED = 5
STATUS_ZERO_FLOW_WITH_SUSPENDED = 6
STATUS_ARITHMETIC_OVERFLOW = 7

ALIGNMENT_RELATIVE_TOLERANCE = 1.0e-12
CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
)
KERNEL_NAME = "compose_post_density_channels"


class AdvDiffOnlyCompositionError(ValueError):
    """Raised when FLOW's erosion/deposition-only class reaches this slice."""


@dataclass(frozen=True)
class ResidentChannelCompositionResult:
    """Resident class-major outputs from one bounded composition launch."""

    composed_bed_mass: object
    retained_suspended_mass: object
    equilibrium_suspended_mass_concentration: object
    signed_suspended_mass_rate: object
    status: object

    @property
    def bed_mass(self):
        """Short alias for downstream face-reconstruction adapters."""
        return self.composed_bed_mass


class ResidentChannelComposer:
    """Compose post-density capacity channels without a host-array fallback."""

    def __init__(self, source_path: str | Path | None = None):
        import cupy as cp

        self.cp = cp
        self.device_id = int(cp.cuda.Device().id)
        path = Path(source_path or Path(__file__).with_suffix(".cu"))
        source = path.read_text(encoding="utf-8")
        with cp.cuda.Device(self.device_id):
            module = cp.RawModule(
                code=source,
                options=CUDA_OPTIONS,
                name_expressions=(KERNEL_NAME,),
            )
            self.kernel = module.get_function(KERNEL_NAME)

    def _array(self, name: str, value: object, *, ndim: int,
               shape: tuple[int, ...] | None = None, integer: bool = False):
        cp = self.cp
        dtype = cp.int32 if integer else cp.float64
        if (not isinstance(value, cp.ndarray) or value.dtype != dtype or
                value.ndim != ndim or not value.flags.c_contiguous):
            label = "int32" if integer else "FP64"
            raise TypeError(
                f"{name} must be a resident contiguous {ndim}-D {label} CuPy array"
            )
        if int(value.device.id) != self.device_id:
            raise ValueError(
                f"{name} is not resident on CUDA device {self.device_id}"
            )
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _finite_scalar(name: str, value: float, *, positive: bool = False):
        if isinstance(value, (bool, np.bool_)):
            qualifier = " positive" if positive else ""
            raise ValueError(f"{name} must be a finite{qualifier} scalar")
        result = float(value)
        if not np.isfinite(result) or (positive and result <= 0.0):
            qualifier = " positive" if positive else ""
            raise ValueError(f"{name} must be a finite{qualifier} scalar")
        return result

    def compose(
        self,
        *,
        raw_bed_mass,
        raw_suspended_mass,
        base_velocity,
        depth,
        transport_class,
        bed_factor: float = 1.0,
        suspended_factor: float = 1.0,
        speed_epsilon: float = 1.0e-6,
        alignment_rtol: float = ALIGNMENT_RELATIVE_TOLERANCE,
        alignment_atol: float = 1.0e-15,
        reject_invalid: bool = True,
    ) -> ResidentChannelCompositionResult:
        """Apply locked post-density channel ordering on resident arrays.

        Channel arrays have shape ``(2, nclass, ncell)``; ``base_velocity``
        has shape ``(2, ncell)`` and deliberately remains unrotated;
        ``transport_class`` is resident contiguous ``int32`` with one entry
        per fraction.  The raw bed channel must already include any selected
        bed-only secondary-flow rotation.  Raw arrays are mass-valued, so this
        API intentionally has no solid-density argument.

        Class 2 and unknown classes always fail before the numerical launch,
        including when ``reject_invalid`` is false.  That option only exposes
        per-point physical validation statuses for classes 1 and 3.
        """
        cp = self.cp
        raw_bed_mass = self._array("raw_bed_mass", raw_bed_mass, ndim=3)
        if raw_bed_mass.shape[0] != 2:
            raise ValueError("raw_bed_mass first dimension must contain x/y")
        nclass = int(raw_bed_mass.shape[1])
        ncell = int(raw_bed_mass.shape[2])
        if nclass <= 0 or ncell <= 0:
            raise ValueError("channel composition requires classes and cells")
        raw_suspended_mass = self._array(
            "raw_suspended_mass", raw_suspended_mass, ndim=3,
            shape=(2, nclass, ncell),
        )
        base_velocity = self._array(
            "base_velocity", base_velocity, ndim=2, shape=(2, ncell),
        )
        depth = self._array("depth", depth, ndim=1, shape=(ncell,))
        transport_class = self._array(
            "transport_class", transport_class, ndim=1, shape=(nclass,),
            integer=True,
        )
        bed_factor = self._finite_scalar("bed_factor", bed_factor)
        suspended_factor = self._finite_scalar(
            "suspended_factor", suspended_factor,
        )
        speed_epsilon = self._finite_scalar(
            "speed_epsilon", speed_epsilon, positive=True,
        )
        alignment_rtol = self._finite_scalar(
            "alignment_rtol", alignment_rtol,
        )
        alignment_atol = self._finite_scalar(
            "alignment_atol", alignment_atol,
        )
        if alignment_rtol < 0.0 or alignment_atol < 0.0:
            raise ValueError("alignment tolerances must be nonnegative")
        if not isinstance(reject_invalid, (bool, np.bool_)):
            raise TypeError("reject_invalid must be Boolean")

        with cp.cuda.Device(self.device_id):
            # Metadata is resident too.  Only scalar reductions cross the
            # device boundary; physical channels never do.  Class 2 is checked
            # separately so its erosilt provenance cannot blur into "unknown".
            if bool(cp.any(transport_class == TRA_ADVDIFF).item()):
                raise AdvDiffOnlyCompositionError(
                    "Transport class 2 is erosion/deposition-only in locked "
                    "FLOW: erosed routes it through erosilt and exits before "
                    "eqtran; post-density formula channels cannot be composed"
                )
            if bool(cp.any(
                    (transport_class != TRA_BEDLOAD) &
                    (transport_class != TRA_COMBINE)).item()):
                raise ValueError(
                    "channel composition supports only transport classes 1 and 3"
                )

            composed = cp.zeros_like(raw_bed_mass)
            retained = cp.zeros_like(raw_suspended_mass)
            equilibrium = cp.zeros((nclass, ncell), dtype=cp.float64)
            signed = cp.zeros_like(equilibrium)
            status = cp.zeros((nclass, ncell), dtype=cp.int32)
            count = nclass * ncell
            self.kernel(
                ((count + 127) // 128,),
                (128,),
                (
                    np.int32(ncell), np.int32(nclass),
                    np.float64(bed_factor), np.float64(suspended_factor),
                    np.float64(speed_epsilon), np.float64(alignment_rtol),
                    np.float64(alignment_atol), raw_bed_mass,
                    raw_suspended_mass, base_velocity, depth, transport_class,
                    composed, retained, equilibrium, signed, status,
                ),
            )
            result = ResidentChannelCompositionResult(
                composed_bed_mass=composed,
                retained_suspended_mass=retained,
                equilibrium_suspended_mass_concentration=equilibrium,
                signed_suspended_mass_rate=signed,
                status=status,
            )
            if reject_invalid:
                invalid = int(cp.count_nonzero(status).item())
                if invalid:
                    raise ValueError(
                        f"channel composition rejected {invalid} invalid point"
                        f"{'s' if invalid != 1 else ''}"
                    )
        return result


__all__ = [
    "ALIGNMENT_RELATIVE_TOLERANCE",
    "AdvDiffOnlyCompositionError",
    "ResidentChannelComposer",
    "ResidentChannelCompositionResult",
    "STATUS_ADVDIFF_REQUIRES_EROSILT",
    "STATUS_ARITHMETIC_OVERFLOW",
    "STATUS_NONFINITE",
    "STATUS_NONPOSITIVE_COMBINED_DEPTH",
    "STATUS_OK",
    "STATUS_SUSPENDED_MISALIGNED",
    "STATUS_UNSUPPORTED_CLASS",
    "STATUS_ZERO_FLOW_WITH_SUSPENDED",
    "TRA_ADVDIFF",
    "TRA_BEDLOAD",
    "TRA_COMBINE",
]
