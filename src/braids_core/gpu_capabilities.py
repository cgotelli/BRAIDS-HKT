"""Pure routing after explicit GPU selection; never a backend/physics fallback.

Operator admission remains authoritative. This module imports no numerical
worker, NumPy, QGIS or CUDA code.
"""
INTEGRATED = 'integrated'
GRADED = 'live_normal_critical_graded_bedload_ssp33'
INLET = 'live_mode9_section_choice_ssp33'
BANK = 'live_normal_critical_bank_ssp33'
TRANSPORT = 'live_normal_critical_transport_ssp33'
AMR = 'on_demand_bank_amr_ssp33'
AMR_GENERAL = 'on_demand_transport_amr_ssp33'
_PREPARATIONS = {3: frozenset((GRADED,)), 4: frozenset((INLET,)),
                 5: frozenset((BANK,)), 6: frozenset((TRANSPORT, AMR, AMR_GENERAL)),
                 7: frozenset((TRANSPORT,))}

# Public GPU time-method identities are shared by the standalone worker and
# the QGIS host.  Keep the user-facing selector token separate from the exact
# method label written into manifests and checkpoints.  Unknown tokens must
# fail instead of being interpreted as one of the explicit Runge--Kutta
# methods.
GPU_INTEGRATOR_LABELS = {
    'ssp43': 'SSPRK43',
    'ssp33': 'SSPRK33',
    'implicit_fwave_midpoint': 'IMPLICIT-F-WAVE-MIDPOINT',
}


def gpu_integrator_label(token):
    try:
        return GPU_INTEGRATOR_LABELS[token]
    except (KeyError, TypeError) as error:
        raise ValueError('Unsupported explicit GPU time integrator') from error


def has_suspended_transport(exchange_mode, initial_mass=(), boundary_concentration=(),
                            scheduled_concentration=(), *, explicit_active=False):
    """Match the resident core's state-allocation predicate on host inputs.

    This is not physical admission: callers still validate shapes, finiteness,
    nonnegative concentrations and legal integer switches. Arrays may retain
    native zero-valued suspended slots without allocating a transported state.
    """
    def nonzero(value):
        if isinstance(value, (str, bytes, dict)):
            raise TypeError('Suspended transport activation requires numeric values')
        try:
            entries = iter(value)
        except TypeError:
            return bool(value != 0)
        return any(nonzero(entry) for entry in entries)

    return (bool(exchange_mode != 0) or bool(explicit_active) or nonzero(initial_mass) or
            nonzero(boundary_concentration) or nonzero(scheduled_concentration))


def route_for_preparation_schema(schema, variant):
    """Dispatch known ordinary receipts; caller must authenticate their files."""
    if type(schema) is not int or variant not in _PREPARATIONS.get(schema, ()):
        raise ValueError('GPU preparation is not connected to the integrated configuration')
    return variant


def effective_project_transport(project):
    """Match the native writer's explicit sediment-off switch, without edits."""
    numerics = project['numerics']
    enabled = numerics.get('sediment_enabled', True)
    if type(enabled) is not bool:
        raise ValueError('GPU sediment_enabled must be boolean')
    if not enabled:
        internal = project.get('internal_conditions', {})
        for name in ('bank_failure', 'bank_morphology', 'general_avalanche',
                     'secondary_flow', 'bedload_adaptation', 'bedform_roughness',
                     'morphological_acceleration'):
            if internal.get(name, {}).get('enabled', False):
                raise ValueError('Sediment-disabled GPU project cannot enable '+name)
        return 0, 0
    return numerics.get('bedload', 2), numerics.get('exchange', 0)


def _has_public_law11_total_gpu_binding(project):
    """Recognize only the admitted complete schema-v2 law-11 profile.

    Integrated routing must not infer support from the legacy integer alone.
    Re-resolution against the installed v2 ledger authenticates the binding
    and backend admission; the explicit checks below keep this route bounded
    if additional v2 profiles are admitted in the future.
    """
    from braids_core.sediment.execution_profile_binding import (
        EXECUTION_PROFILE_BINDING_KEY, runtime_execution_profile)
    from braids_core.sediment.execution_profile_v2 import BACKEND_GPU
    from braids_core.sediment.fixed_grid_profile_completeness import (
        validate_fixed_grid_profile_completeness)

    binding = project.get(EXECUTION_PROFILE_BINDING_KEY)
    if not isinstance(binding, dict) or binding.get('state') != 'canonical':
        return False
    documents = binding.get('component_documents')
    if not isinstance(documents, dict):
        return False
    try:
        profile, checked = runtime_execution_profile(
            binding, backend=BACKEND_GPU, component_documents=documents)
        validate_fixed_grid_profile_completeness(
            profile, component_documents=checked)
    except (KeyError, TypeError, ValueError):
        return False
    if (profile.backend != BACKEND_GPU or len(profile.fraction_layout) != 1 or
            profile.fraction_layout[0].transport_class != 1):
        return False
    fraction_id = profile.fraction_layout[0].fraction_id
    selected = {(row.entry_id, tuple(row.scope))
                for row in profile.method_selections}
    expected = {
        ('capacity.general', (('fraction_id', fraction_id),)),
        ('transport_route.total', (('fraction_id', fraction_id),)),
        ('suspension.off', (('fraction_id', fraction_id),)),
        ('shear.total_roughness', (('model_id', 'model'),)),
        ('hiding.none', (('model_id', 'model'),)),
        ('bed_slope.off', (('model_id', 'model'),)),
        ('secondary.off', (('model_id', 'model'),)),
    }
    capacity = checked.get('capacity')
    return bool(
        selected == expected and set(checked) == {
            'capacity', 'channel_composition'} and
        isinstance(capacity, dict) and capacity.get('backend') is None and
        len(capacity.get('fractions', ())) == 1 and
        capacity['fractions'][0].get('fraction_id') == fraction_id and
        capacity['fractions'][0].get('method_id') == 'capacity.general' and
        capacity['fractions'][0].get('formula_code') == 11)


def resolve_project_route(project):
    """Choose a connected internal route without modifying the project."""
    if not isinstance(project, dict) or not isinstance(project.get('numerics'), dict):
        raise ValueError('GPU integrated configuration needs explicit project numerics')
    numerics = project['numerics']
    internal = project.get('internal_conditions', {})
    amr = numerics.get('amr', {}).get('enabled', False)
    bank = internal.get('bank_failure', {}).get('enabled', False)
    if type(amr) is not bool or type(bank) is not bool:
        raise ValueError('GPU integrated configuration switches must be boolean')
    exchange_bedload = effective_project_transport(project)
    if amr: return AMR if bank else AMR_GENERAL
    if bank: return BANK
    bedload, exchange = exchange_bedload
    if type(exchange) is not int or type(bedload) is not int:
        raise ValueError('GPU integrated transport choices must be integer controls')
    # Route selection never overrides an explicitly selected historical driver.
    # The ordinary general route performs the actual physics/boundary admission.
    if exchange in (0, 1, 2) and bedload in (0, 2): return TRANSPORT
    if exchange == 0 and bedload == 8: return TRANSPORT
    if (exchange == 0 and bedload == 11 and
            _has_public_law11_total_gpu_binding(project)):
        return TRANSPORT
    raise ValueError('These GPU physics/boundaries are not yet connected to the integrated configuration; '
                     'no backend or numerical fallback was selected')
