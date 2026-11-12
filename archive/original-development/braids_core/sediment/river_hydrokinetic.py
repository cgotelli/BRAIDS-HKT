"""Optional hydrodynamic turbine source on the common river owner.

This research installer is intentionally not routed by the Desktop project
schema yet.  It composes one stateless momentum source after the selected
hydrodynamic momentum terms, on the user's explicit CPU or GPU backend.  It
does not change water mass, f-wave fluxes, friction, turbulence or sediment.
"""
from __future__ import annotations

from types import MethodType

from braids_core.hydrokinetic_turbines import (
    METHOD, TurbineMomentumSource, build_footprints, validate_turbines,
)
from .bend_methods_v1 import digest


def install(owner, cells, rows, *, support_thickness_m, source_distribution='uniform_strip'):
    if owner.time or owner.steps or owner.physics._open:
        raise ValueError("Install hydrokinetic turbines before execution")
    if owner.morphology:
        raise ValueError("The first turbine milestone supports hydrodynamics only")
    definitions = validate_turbines(rows)
    if not definitions:
        raise ValueError("At least one turbine is required")
    footprints = build_footprints(cells, definitions, support_thickness_m)
    if source_distribution == 'uniform_strip':
        source = TurbineMomentumSource(footprints, backend=owner.backend,
                                       dry_depth_m=owner.physics.parameters.dry)
    elif source_distribution == 'circular_projection':
        from .river_hydrokinetic_projection import CircularRotorSource
        source = CircularRotorSource(footprints, cells, backend=owner.backend,
                                     dry_depth_m=owner.physics.parameters.dry)
    else:
        raise ValueError('Unknown research turbine momentum distribution')
    physics = owner.physics
    previous = physics.add_momentum

    def add_momentum(self, q, water, rate):
        previous(q, water, rate)
        self.turbine_diagnostics = source.apply(q, rate)

    physics.add_momentum = MethodType(add_momentum, physics)
    physics.turbine_diagnostics = []
    identity = dict(method=source.identity['method'], source_sha256=digest(__file__),
                    operator=source.identity,
                    turbines=[dict(id=f.definition.turbine_id,
                                   center_xy=list(f.definition.center_xy),
                                   tangent_xy=list(f.definition.tangent_xy),
                                   rotor_diameter_m=f.definition.rotor_diameter_m,
                                   hub_height_m=f.definition.hub_height_m,
                                   thrust_coefficient=f.definition.thrust_coefficient,
                                   power_coefficient=f.definition.power_coefficient,
                                   support_area_m2=f.support_area_m2,
                                   support_thickness_m=f.support_thickness_m,
                                   numerical_cross_stream_width_m=f.numerical_cross_stream_width_m,
                                   cell_count=len(f.cell_ids)) for f in footprints],
                    hydrodynamics_only=True, turbulence_source=False,
                    sediment_coupling=False, desktop_admission=False)
    physics.identity["hydrokinetic_turbines"] = identity
    owner.identity["hydrokinetic_turbines"] = identity
    owner.turbine_source = source
    return owner

