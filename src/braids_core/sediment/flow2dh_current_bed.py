"""Canonical classic FLOW bend development implementation, CPU and CUDA.

One fixed correction bundle, not a menu of numerical experiments. Historical
owners remain immutable restart/validation references. This fully wet EH route
does not yet replace the general QGIS worker's wider physics contracts.
"""
from .bend_methods_v1 import digest
from .flow2dh_geometry_bed import Flow2DHGeometryBed
from .flow2dh_effective_bed import Flow2DHEffectiveBed
from .flow2dh_objective_bed import ObjectiveEngine

METHOD = 'classic_flow2dh_eh_current_v2'


class Flow2DHCurrentBed(Flow2DHGeometryBed, Flow2DHEffectiveBed):
    """Compose face geometry, effective shear and objective reconstruction once.

    The cooperative MRO initializes the effective inputs before geometry, and
    uses the effective bed_faces with the objective reconstruction engine. Each
    auxiliary face contribution is therefore applied once, not summed twice.
    """
    def __init__(self, *args, correction='current', **kwargs):
        if correction != 'current':
            raise ValueError('The current FLOW bend implementation has one fixed correction bundle')
        super().__init__(*args, correction='all', **kwargs)
        self.reconstruction = ObjectiveEngine(self.engine.backend, self.flow.threads,
            gradient_kernel='flow2dh_component_capacity.h')
        self.identity = dict(self.identity, method=METHOD,
            current_wrapper_sha256=digest(__file__),
            maintenance_status='canonical_development_implementation',
            objective_runtime=self.reconstruction.identity,
            capacity_forcing='face_derived_effective_Chezy_and_dwnvel_velocity',
            bed_face='limited_LS_capacity_at_true_face; independent_local_transport_component_bounds; sign_upwind_normal; affine_exact_secant_slope',
            capacity_limiter_diagnostic='minimum_of_independent_local_component_factors',
            enabled_corrections=['limited_capacity_reconstruction',
                'objective_component_capacity_limiter','secondary_stress_true_midpoint',
                'nonorthogonal_viscosity','nonorthogonal_secondary_diffusion',
                'face_derived_effective_sediment_shear'],
            correction_selection='fixed_bundle_no_runtime_alternatives',
            public_support=False,
            public_support_reason='General QGIS routing, wet/dry, mixtures and AMR require separate integration qualification')
