"""Explicit parameters for the current-only, noncohesive FLOW skin closure."""
import math

# FLOW initializes KsSand to zero (smooth bed), not a guessed multiple of D50.
# Viscosity is separately supplied, never silently taken from settling controls.
PARAMETERS = dict(sand_roughness_m=0., viscosity_m2_s=1.e-6)


def validate_parameters(value):
    if not isinstance(value, dict) or set(value) != set(PARAMETERS):
        raise ValueError("Explicit sand roughness and molecular viscosity required for skin shear")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value.values()):
        raise ValueError("Finite numeric skin shear parameters required")
    if value["sand_roughness_m"] < 0. or value["viscosity_m2_s"] <= 0.:
        raise ValueError("Nonnegative sand roughness and positive skin shear viscosity required")
