"""Bind one accepted coarsened parent to a new closed one-cell execution.

This bounded validation bridge is not an arbitrary hierarchy regridder. The
caller supplies an authenticated v98 coarsened image and a fresh execution
whose geometry and bed datum represent that parent. No time, constitutive
law, dry cleanup, or sediment transfer is evaluated here. Reference windows
are rebased, while physical time, accepted-update count, conserved rows,
lagged DSS and mapped HLES memory are carried into the real v97 solver.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .active_suspended_control_v94 import _wrap_state as wrap_active
from .hles_producer_v90 import _wrap_state as wrap_hles
from .multifraction_suspended_lifecycle_v78 import resolve_multifraction_suspended_state_v78
from .shared_turbulence_consumers_v91 import _wrap_state as wrap_shared
from .transported_fluid_settling_v92 import _wrap_state as wrap_fluid
from .suspended_exner_owner_v93 import _wrap_state as wrap_exner


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind_coarsened_parent_continuation_v100(execution, fresh_state, provider,
        image, *, native_cpu, source_event_sha256):
    """Return a closed, clock-preserving CPU/GPU continuation parent/provider.

    The target perimeter is explicitly reflective. It is an isolated parent
    gate, not preservation of a surrounding joined hierarchy's forcing.
    """
    if native_cpu:
        from .coupled_wetdry_owner_cpu_v97 import _wrap_state, validate_coupled_wetdry_cpu_state_v97 as validate
        from .q3_mixed_route_exner_adapter_v93 import resolve_cpu_q3_v84_mixed_state_v93 as resolve_q3
        from .q3_wetdry_adapter_cpu_v97 import CPUQ3V83ActiveWetDryExnerAdapterV97 as Provider
    else:
        from braids_gpu.coupled_wetdry_v97 import _wrap_state, validate_coupled_wetdry_gpu_state_v97 as validate
        from braids_gpu.q3_exner_adapter_v93 import resolve_resident_q3_v84_mixed_state_v93 as resolve_q3
        from braids_gpu.active_suspended_wetdry_v97 import ResidentQ3V83ActiveWetDryExnerAdapterV97 as Provider
    active_e = execution.active_execution
    exner_e = active_e.base_execution
    fluid_e = exner_e.base_execution
    shared_e = fluid_e.base_execution
    grid = shared_e.base_execution.base_execution
    if (image.layout != 'coarsened_virtual_parent' or execution.cell_count != 1
            or image.q.shape != (3, 1) or image.accepted_updates < 1
            or image.time_s <= 0. or len(source_event_sha256) != 64
            or np.any(grid.neighbour >= 0)
            or not np.array_equal(image.bed_mass_kg_m2, execution.reference_bed_mass_kg_m2)
            or not np.array_equal(image.bed_elevation_m, execution.initial_bed_elevation_m)):
        raise ValueError('v100 continuation requires one accepted coarsened closed parent')
    # Immutable mapped history belongs to the same accepted clock, not a
    # cold HLES prime or a second closure calculation.
    side_normals = ((-1., 0.), (1., 0.), (0., -1.), (0., 1.))
    face_order = [side_normals.index(tuple(grid.normal[:, f]))
                  for f in range(grid.face_count)]
    chain = _hash(dict(scope='coarsened_parent_continuation_v100',
        source_event_sha256=source_event_sha256, image_sha256=image.receipt_sha256,
        target_execution=execution.fingerprint, native_cpu=native_cpu))
    hles = wrap_hles(shared_e.hles_execution,
        filtered=image.hles_filtered_face_normal_velocity_m_s[face_order],
        dynamic_vicuv=image.hles_dynamic_vicuv_m2_s,
        dynamic_dicuv=image.hles_dynamic_dicuv_m2_s,
        accepted_updates=image.accepted_updates, time_s=image.time_s,
        receipt_chain=chain, primed=True,
        prime_receipt_sha256=fresh_state.hles_state.prime_receipt_sha256)
    ng, nb = execution.fraction_count, fluid_e.boundary_count
    water_ref = np.sum(image.suspended_mass_kg_m2*grid.cell_area_m2, axis=1)
    bed_ref = np.sum(image.bed_mass_kg_m2*grid.cell_area_m2, axis=1)
    suspended = resolve_multifraction_suspended_state_v78(shared_e.base_execution,
        suspended_mass_kg_m2=image.suspended_mass_kg_m2,
        bed_mass_kg_m2=image.bed_mass_kg_m2,
        accepted_updates=image.accepted_updates, time_s=image.time_s,
        reference_water_mass_kg=water_ref, reference_bed_mass_kg=bed_ref)
    shared = wrap_shared(shared_e, q=image.q, suspended_state=suspended, receipt_chain=chain)
    fluid = wrap_fluid(fluid_e, shared,
        h_temperature_c_m=image.h_temperature_c_m,
        h_salinity_ppt_m=image.h_salinity_ppt_m,
        boundary_last_outflow_fluid=np.zeros((2, fluid_e.face_count)),
        boundary_return_clock_s=np.zeros((2, fluid_e.face_count)),
        reference_scalar_integral=np.array((
            np.sum(image.h_temperature_c_m*grid.cell_area_m2),
            np.sum(image.h_salinity_ppt_m*grid.cell_area_m2))),
        advective_boundary_ledger=np.zeros((2, nb)),
        diffusive_boundary_ledger=np.zeros((2, nb)),
        wetdry_scalar_ledger=np.zeros(2), accepted_receipt_chain_sha256=chain,
        temperature_cell_c=image.temperature_cell_c,
        salinity_cell_ppt=image.salinity_cell_ppt,
        water_density_cell_kg_m3=image.water_density_cell_kg_m3,
        molecular_viscosity_cell_m2_s=image.molecular_viscosity_cell_m2_s)
    old_q3 = provider.state if native_cpu else provider.state.core
    q3 = resolve_q3(execution_fingerprint=active_e.factcr_execution.fingerprint,
        binding=provider.binding, authoritative_bed_mass_kg_m2=image.bed_mass_kg_m2,
        bulk_density_kg_m3=old_q3.bulk_density_kg_m3,
        retained_fraction=old_q3.active_fraction,
        accepted_updates=image.accepted_updates, time_s=image.time_s)
    kwargs = dict(outer_execution=active_e, binding=provider.binding, state=q3)
    if native_cpu:
        kwargs.update(native_factcr_runtime=provider.native_factcr_runtime,
            native_q3_face_runtime=provider.native_q3_face_runtime,
            native_threads=provider.native_factcr_threads)
    continued_provider = Provider(**kwargs)
    exner = wrap_exner(exner_e, fluid,
        q3_provider_kind=continued_provider.provider_kind,
        q3_provider_state_sha256=q3.state_sha256, bed_mass_kg_m2=image.bed_mass_kg_m2,
        reference_bed_integral_kg=bed_ref,
        suspended_exchange_ledger_kg=np.zeros(ng),
        suspended_boundary_ledger_kg=np.zeros((ng, nb)),
        excluded_water_exchange_kg=np.zeros(ng),
        q3_named_boundary_positive_in_kg=np.zeros((ng, len(exner_e.q3_boundary_names))),
        q3_named_boundary_absolute_kg=np.zeros((ng, len(exner_e.q3_boundary_names))),
        q3_depletion_roundoff_positive_in_kg=np.zeros(ng),
        wetdry_bed_ledger_kg=np.zeros(ng), accepted_exner_chain_sha256=chain)
    active = wrap_active(active_e, exner,
        suspended_diameter_m=image.suspended_diameter_m,
        accepted_iopsus_chain_sha256=chain)
    kwargs = dict(active_state=active, hles_state=hles, accepted_chain_sha256=chain)
    if native_cpu:
        kwargs['q3_provider'] = continued_provider
    state = _wrap_state(execution, **kwargs)
    validate(state, execution, continued_provider)
    np.testing.assert_array_equal(state.q, image.q)
    np.testing.assert_array_equal(state.active_state.suspended_diameter_m, image.suspended_diameter_m)
    return state, continued_provider
