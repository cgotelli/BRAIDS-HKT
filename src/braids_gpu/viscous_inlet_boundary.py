"""Composition bridge for the resident typed viscous-inlet CUDA operator."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .forcing_profile import BINDING_FIELD, forcing_identity
from .viscous_inlet import ViscosityEquilibratedInletCUDA


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_viscous_inlet_scope(data, *, integrator):
    """Authenticate the exact fixed-grid mode-6 dependency bundle."""
    _require(integrator == "implicit_fwave_midpoint",
             "Viscosity-equilibrated inlet requires implicit_fwave_midpoint")
    nb = int(data["nb"])
    active = np.asarray(data.get("viscosity_equilibrated_inlet"))
    modes = np.asarray(data.get("mode"))
    viscosity = np.asarray(data.get("horizontal_viscosity"))
    _require(active.shape == (nb,) and active.dtype.kind in "iu" and
             np.isin(active, (0, 1)).all() and np.any(active),
             "Viscosity-equilibrated inlet flags are absent or invalid")
    _require(modes.shape == (nb,) and modes.dtype.kind in "iu" and
             np.all(modes[active != 0] == 6),
             "Viscosity-equilibrated inlet requires mode 6 total discharge")
    _require(viscosity.shape == () and viscosity.dtype.kind in "fiu" and
             np.isfinite(viscosity) and float(viscosity) > 0.0 and
             int(np.asarray(data.get("horizontal_viscosity_mode")).item()) == 1 and
             int(np.asarray(data.get("horizontal_viscosity_geometry")).item()) == 1,
             "Viscosity-equilibrated inlet requires positive resident velocity-Laplacian viscosity")
    _require(not bool(np.any(data.get("amr_enabled", 0))) and
             int(np.asarray(data.get("amr_levels", 1)).item()) == 1,
             "Viscosity-equilibrated inlet is not yet qualified on AMR")
    forcing = np.asarray(data.get("forcing"))
    _require(forcing.ndim == 2 and forcing.shape[1] == nb and
             forcing.shape[0] >= 2 and forcing.dtype.kind in "fiu" and
             np.isfinite(forcing).all() and np.all(forcing[0, active != 0] >= 0.0),
             "Viscosity-equilibrated inlet forcing is invalid")
    _require(BINDING_FIELD in data,
             "Typed forcing identity is required before viscous-inlet admission")
    forcing_receipt = forcing_identity(data)
    _require(forcing_receipt["automatic_CPU_fallback"] is False,
             "Typed forcing identity permits an automatic CPU fallback")
    for key in ("mid", "normal", "length", "owner", "bid",
                "boundary_offset", "boundary_faces"):
        _require(key in data, "Viscous-inlet native geometry is missing: " + key)
    return dict(
        schema=1,
        backend="cuda",
        integrator="implicit_fwave_midpoint",
        selected_boundary_ids=np.flatnonzero(active).astype(int).tolist(),
        hydraulic_roughness_method_id=forcing_receipt[
            "hydraulic_roughness_method_id"],
        roughness_parameter_name=forcing_receipt["roughness_parameter_name"],
        roughness_parameter_units=forcing_receipt["roughness_parameter_units"],
        horizontal_viscosity_m2_s=float(viscosity),
        numerical_fallback="none",
        automatic_CPU_fallback=False,
    )


def wrap_boundary(Parent, actual_data, *, source_folder=None, integrator):
    """Return a boundary class that overwrites only authenticated inlet faces.

    The inherited stock boundary is given an inert copy of the two controls it
    historically rejects.  The captured, already authenticated values remain
    authoritative and are executed by ``ViscosityEquilibratedInletCUDA``.
    """
    receipt = validate_viscous_inlet_scope(actual_data, integrator=integrator)
    active_host = np.ascontiguousarray(
        np.asarray(actual_data["viscosity_equilibrated_inlet"], dtype=np.int32)
    )
    source_folder = None if source_folder is None else Path(source_folder).resolve()

    class ViscosityEquilibratedBoundary(Parent):
        def __init__(self, data, inputs, source=None, **kwargs):
            runtime_active = np.asarray(data.get("viscosity_equilibrated_inlet"))
            _require(np.array_equal(runtime_active, active_host),
                     "Runtime viscous-inlet selection differs from authenticated preparation")
            parent_data = dict(data)
            parent_data["horizontal_viscosity"] = np.float64(0.0)
            parent_data["viscosity_equilibrated_inlet"] = np.zeros_like(active_host)
            super().__init__(parent_data, inputs, source=source, **kwargs)
            folder = source_folder
            if folder is None:
                _require(source is not None,
                         "Archived viscous-inlet source folder is required")
                folder = Path(source).resolve().parent
            self.viscous_inlet = ViscosityEquilibratedInletCUDA.from_native(
                actual_data,
                source_path=folder / "viscous_inlet.cu",
                roughness_source_path=folder / "hydraulic_roughness.cu",
            )
            self.viscous_active = self.cp.asarray(active_host)
            self.viscous_mode = actual_data[BINDING_FIELD]["hydraulic_roughness_mode"]
            self.viscous_waqua = bool(
                actual_data[BINDING_FIELD]["white_colebrook_chzk2d"])
            self.viscous_work_depth = float(
                actual_data[BINDING_FIELD]["effective_minimum_work_depth_m"])
            self.viscous_value = float(actual_data["horizontal_viscosity"])
            self.viscous_receipt = dict(receipt,
                                        operator=dict(self.viscous_inlet.identity))

        def apply(self, q, trace):
            result = super().apply(q, trace)
            profile = self.viscous_inlet.solve(
                q, self.arrays["manning"], self.viscous_active,
                self.arrays["forcing"][0], mode=self.viscous_mode,
                viscosity=self.viscous_value, gravity=float(self.grav),
                dry_tolerance=float(self.dry),
                minimum_work_depth=self.viscous_work_depth,
                waqua_white_colebrook=self.viscous_waqua,
            )
            _require(profile["automatic_CPU_fallback"] is False,
                     "Viscous inlet selected a CPU fallback")
            self.viscous_inlet.apply_traces(
                q, trace, normal=self.inputs["normal"],
                owner=self.inputs["owner"], bid=self.arrays["bid"],
                active=self.viscous_active, speed_limit=float(self.speed),
                face_status=self.face_status, boundary_status=self.status,
                dry_tolerance=float(self.dry),
            )
            return result

        def receipt(self):
            parent = super().receipt() if hasattr(super(), "receipt") else {}
            return {**dict(parent), "viscosity_equilibrated_inlet":
                    dict(self.viscous_receipt)}

    ViscosityEquilibratedBoundary.__name__ = (
        "ViscosityEquilibrated" + Parent.__name__
    )
    return ViscosityEquilibratedBoundary


__all__ = ["validate_viscous_inlet_scope", "wrap_boundary"]
