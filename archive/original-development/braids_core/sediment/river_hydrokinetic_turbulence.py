"""Published background-mixing rules for hydrokinetic-turbine research.

This module deliberately does not invent a rotor-local viscosity halo or a
turbulent-kinetic-energy source.  It identifies the domain-wide conservative
horizontal eddy viscosity already executed by BRAIDS and resolves the
velocity-dependent constant-viscosity scenario of Abolghasemi, Piggott and
Kramer (2019).  A dimensionless similarity transfer is provided solely for
cross-scale assessment against the Mycek flume; it is explicitly distinguished
from the paper's dimensional rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .bend_methods_v1 import digest


METHOD = "turbulence.abolghasemi2019_velocity_scaled_constant_eddy_viscosity"
DOI = "10.48550/arXiv.1810.09827"
REFERENCE_DIAMETER_M = 18.75


@dataclass(frozen=True)
class PublishedMixing:
    inlet_velocity_m_s: float
    rotor_diameter_m: float
    viscosity_m2_s: float
    application: str
    reference_inlet_velocity_m_s: float
    reference_viscosity_m2_s: float

    @property
    def dimensionless_viscosity(self):
        return self.viscosity_m2_s / (self.inlet_velocity_m_s * self.rotor_diameter_m)

    def document(self):
        return dict(
            method=METHOD, publication="arXiv:1810.09827",
            publication_doi=DOI, application=self.application,
            inlet_velocity_m_s=self.inlet_velocity_m_s,
            rotor_diameter_m=self.rotor_diameter_m,
            viscosity_m2_s=self.viscosity_m2_s,
            dimensionless_viscosity=self.dimensionless_viscosity,
            reference_inlet_velocity_m_s=self.reference_inlet_velocity_m_s,
            reference_rotor_diameter_m=REFERENCE_DIAMETER_M,
            reference_viscosity_m2_s=self.reference_viscosity_m2_s,
            published_rule="nu = 0.15 m2/s + (0.15 m) U_in",
            operator="conservative depth-averaged horizontal momentum diffusion",
            ambient_background_mixing=True,
            turbine_generated_tke_source=False,
            source_sha256=digest(__file__),
        )


def published_viscosity(inlet_velocity_m_s):
    """Equation 22 of Abolghasemi et al. for its 18.75 m turbine case."""
    velocity=float(inlet_velocity_m_s)
    if not math.isfinite(velocity) or velocity<=0:
        raise ValueError("Positive finite inlet velocity required")
    return .15 + .15*velocity


def resolve_published(inlet_velocity_m_s):
    velocity=float(inlet_velocity_m_s)
    viscosity=published_viscosity(velocity)
    return PublishedMixing(velocity,REFERENCE_DIAMETER_M,viscosity,
        "published_dimensional_rule",velocity,viscosity)


def resolve_similarity(*, inlet_velocity_m_s, rotor_diameter_m,
                       reference_inlet_velocity_m_s=2.9):
    """Transfer the paper's ``nu/(U D)`` at one published operating point.

    This is a transparent cross-scale diagnostic, not a universal law and not
    a fit to the target experiment.
    """
    velocity=float(inlet_velocity_m_s);diameter=float(rotor_diameter_m)
    reference_velocity=float(reference_inlet_velocity_m_s)
    if not all(math.isfinite(v) and v>0 for v in (velocity,diameter,reference_velocity)):
        raise ValueError("Positive finite target and reference scales required")
    reference_viscosity=published_viscosity(reference_velocity)
    ratio=reference_viscosity/(reference_velocity*REFERENCE_DIAMETER_M)
    return PublishedMixing(velocity,diameter,ratio*velocity*diameter,
        "dimensionless_cross_scale_assessment",reference_velocity,
        reference_viscosity)


def bind(owner, resolved):
    """Bind a resolved rule to the already installed conservative operator."""
    if owner.time or owner.steps or owner.physics._open:
        raise ValueError("Bind turbine mixing before execution")
    if not isinstance(resolved,PublishedMixing):
        raise TypeError("Resolved published turbine mixing required")
    existing=owner.physics.identity.get("conservative_viscosity")
    if not existing:
        raise ValueError("Conservative horizontal viscosity is not installed")
    actual=float(existing.get("coefficient_m2_s",math.nan))
    if not math.isclose(actual,resolved.viscosity_m2_s,rel_tol=0.,abs_tol=1e-14):
        raise ValueError("Installed viscosity differs from the resolved published rule")
    identity=resolved.document()
    identity["conservative_operator_identity"]=dict(existing)
    owner.physics.identity["hydrokinetic_turbulence"]=identity
    owner.identity["hydrokinetic_turbulence"]=identity
    return owner
