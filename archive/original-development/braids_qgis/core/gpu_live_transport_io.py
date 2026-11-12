"""NumPy-only schema-6 live suspended-transport host contract.

This module authenticates native records and accepted physical arrays. It
never imports the numerical worker, initializes CUDA or provides a fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from braids_core.gpu_capabilities import (effective_project_transport,
    gpu_integrator_label, has_suspended_transport)
from braids_core.sediment_method_catalog import (
    CATALOG_SCHEMA, CATALOG_SCOPE_ID, admitted_legacy_codes,
    semantic_catalog_id)
from braids_core.sediment.execution_profile_v2 import (
    BACKEND_GPU, SELECTOR_LEDGER_FILE, bind_component_profile,
    execution_profile_document, is_frozen_q3sp_execution_profile,
    load_execution_profile, load_selector_ledger, resolve_execution_profile)
from braids_core.sediment.execution_profile_binding import (
    EXECUTION_PROFILE_BINDING_KEY, runtime_execution_profile)
from braids_core.sediment.fixed_grid_profile_completeness import (
    validate_fixed_grid_profile_completeness)
from braids_core.sediment.capacity_foundation_execution import (
    capacity_foundation_execution_document,
    load_capacity_foundation_execution)
from braids_core.sediment.hiding_foundation_execution import (
    EXECUTION_KEY as HIDING_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as HIDING_ORDERED_ROUTE_KEY,
    validate_hiding_foundation_execution_document)
from braids_core.sediment.secondary_foundation_execution import (
    EXECUTION_KEY as SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as SECONDARY_ORDERED_ROUTE_KEY,
    validate_secondary_foundation_execution_document)
from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY as PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document)
from braids_core.sediment.capacity_profile import load_capacity_profile
from braids_core.sediment.channel_composition import (
    load_channel_composition_profile)
from braids_core.sediment.forcing_profile import (
    BINDING_FIELD, apply_forcing_binding, forcing_identity)
VARIANT = 'live_normal_critical_transport_ssp33'
EXPORTER_SHA256 = 'a50f91dd99e5b48af94535761a7ec2f5b933e1e21ca9479370e78d03275eac33'
AVALANCHE_EXPORTER_SHA256 = '44b50de40f6d7851957f435a76a16f0995947012bd1ecfd7fcd8776c3915e226'
EH_EXPORTER_SHA256 = '898f467518789453ee2be202095eef2a81725c4e262f0c1c22b0625be56970fd'
DELFT_EXPORTER_V7_SHA256 = '78d961f9905013590883b2c489ed1599e2f62e385e29687e16dc34e76c7920b5'
DELFT_EXPORTER_V12_SHA256 = 'c6ebef0805fad6b4d3893c9deae7e52d5fccda77059309c66441e4a88e2d33c9'
DELFT_EXPORTER_V13_SHA256 = '1828a447ea08e5311828128cca005581f44d14b88dc409a63ff9f48faacbacb8'
DELFT_EXPORTER_V14_SHA256 = 'd803c052cfc3db3de71ef387dc7e77f1458c1d11a691a1b44b429af4ba7f699f'
DELFT_EXPORTER_V15_SHA256 = '9ccdd9cb457140eda9ad418a2843047666c40ea1d8fe6031d52a02a4da88fbd8'
DELFT_EXPORTER_V16_SHA256 = '4e76d84516ba9fc24cf66154de35b548d8214932fa785cc365a818692e4cb266'
DELFT_EXPORTER_V17_SHA256 = '75faae530e2d8a19144b62aa8e18aff7620b5f0b158c0e60a19e769770a2ae7d'
# Host-side receipt validation must remain importable in a clean QGIS plugin
# without the external CUDA worker on Python's module path.  These are protocol
# identities, not executable code; the companion worker independently pins and
# verifies the corresponding zero-step initializer.
DELFT_EXPORTER_V18_SHA256 = 'fb6c9778e1b21400ed6a8bbce0acce4fcecbf3babda55f1dd17262b322f8f467'
DELFT_EXPORTER_V18_GENERATION = 'native_exporter_v18'
# Preserve the unqualified standalone-Q3-H1 alias.  Repaired v17 is selected
# explicitly and only for an authenticated v56 outer envelope; v16 remains a
# verifiable historical snapshot.
DELFT_EXPORTER_SHA256 = DELFT_EXPORTER_V15_SHA256
DELFT_EXPORTER_GENERATION = 'native_exporter_v15'
DELFT_EXPORTER_GENERATIONS = {
    DELFT_EXPORTER_V7_SHA256: 'native_exporter_v7',
    DELFT_EXPORTER_V12_SHA256: 'native_exporter_v12',
    DELFT_EXPORTER_V13_SHA256: 'native_exporter_v13',
    DELFT_EXPORTER_V14_SHA256: 'native_exporter_v14',
    DELFT_EXPORTER_V15_SHA256: DELFT_EXPORTER_GENERATION,
    DELFT_EXPORTER_V16_SHA256: 'native_exporter_v16',
    DELFT_EXPORTER_V17_SHA256: 'native_exporter_v17',
    DELFT_EXPORTER_V18_SHA256: DELFT_EXPORTER_V18_GENERATION,
}
_DELFT_FORCING_EXPORTERS = frozenset((
    DELFT_EXPORTER_V12_SHA256, DELFT_EXPORTER_V13_SHA256,
    DELFT_EXPORTER_V14_SHA256, DELFT_EXPORTER_V15_SHA256,
    DELFT_EXPORTER_V16_SHA256, DELFT_EXPORTER_V17_SHA256,
    DELFT_EXPORTER_V18_SHA256,
))
GAIA_EXPORTER_V4_SHA256 = '90eeb6a5740ba0fa7aee3613855380e6185724209a636d4603ff77a5db7c203f'
GAIA_EXPORTER_V5_SHA256 = '970dc63b032a08de59ca18506c9eb804c7c0d3efdd646f698000dbf981018eb5'
GAIA_EXPORTER_V6_SHA256 = '1c1c8503b7e297e6df07fa53685c41635937ea982254f6719bfe205d253be838'
GAIA_EXPORTER_SHA256 = GAIA_EXPORTER_V6_SHA256
GAIA_EXPORTER_GENERATIONS = {
    GAIA_EXPORTER_V4_SHA256: 'native_exporter_v4',
    GAIA_EXPORTER_V5_SHA256: 'native_exporter_v5',
    GAIA_EXPORTER_V6_SHA256: 'native_exporter_v6',
}
ARTIFACTS = frozenset(('coupled_input.bin', 'coupled_input.npz', 'normal_metadata.bin'))
DIFFUSION_ARTIFACT = 'coupled_diffusion.bin'
AVALANCHE_ARTIFACT = 'avalanche_metadata.bin'
FORCING_ARTIFACT = 'delft_forcing_binding.json'
EXECUTION_PROFILE_ARTIFACT = 'execution_profile_v2.json'
EXECUTION_PROFILE_SIDECAR_SCHEMA = 1
EXECUTION_PROFILE_IDENTITY_FIELDS = (
    'execution_profile_v2_sidecar_sha256',
    'execution_profile_v2_semantic_id',
    'execution_profile_v2_ledger_semantic_id',
    'execution_profile_v2_fingerprint',
    'execution_profile_v2_components_sha256',
    'execution_profile_v2_admission',
    'execution_profile_v2_backend',
    'execution_profile_v2_profile_backend',
)
_EXECUTION_PROFILE_DATA_KEY = '_execution_profile_v2_contract'
_M1_PUBLIC_ADMISSION_KEY = 'm1_profile_public_admission'
_M1_PUBLIC_ADMISSION_ID = 'q3sp.law11_total_single_fraction_v1'
_Q3C_PUBLIC_ADMISSION_ID = 'q3c.scalar_capacity_total_v1'
_Q3H1_PUBLIC_ADMISSION_ID = 'q3h1.scalar_capacity_hiding_v1'
_Q3H2_DISPLAY_ADMISSION_ID = (
    'q3h2.classic_flow_equilibrium_secondary_composition_v1')
_Q3H3_DISPLAY_ADMISSION_ID = (
    'q3h3.classic_flow_prognostic_secondary_composition_v1')
_Q3H3_PROJECT_BINDING_KEY = 'prognostic_secondary_binding'
_Q3H3_WORKER_SOURCE_KEY = 'q3h3_prognostic_secondary_sources'
_Q3H3_WORKER_SOURCE_CONTRACT = {
    'schema': 1,
    'execution_scope_id':
        'braids-q3h3-flow2dh-prognostic-secondary-composition-v1',
    'method_id': 'secondary.prognostic',
    'coupling':
        'weak_midpoint_q3h3_classic_flow_prognostic_secondary_composition_v1',
    'component_coupling_schema': 11,
    'checkpoint_schema': 8,
    'python_source': 'braids_gpu/flow2dh_prognostic_secondary.py',
    'python_sha256':
        '5637a5513ef615f74bc0823bda8826c0aa14c1f3648279b6cdfa579e4658f2ac',
    'cuda_source': 'braids_gpu/flow2dh_prognostic_secondary.cu',
    'cuda_sha256':
        '90e0b16986048250b74edcd29a1aa43c6e7a3a8dbfa166b20cd315c64db8a082',
}
_Q3H3_RESULT_IDENTITIES = (
    'prognostic_secondary_execution_fingerprint',
    'prognostic_secondary_state_semantics',
    'prognostic_secondary_integrated_ledger_units',
    'prognostic_secondary_bed_boundary_rotation_policy',
    'prognostic_secondary_composition_order',
)
_Q3C_METHODS = {2: 'capacity.mpm', 8: 'capacity.engelund_hansen',
                11: 'capacity.general'}
_ROUGHNESS_METHODS = frozenset((
    'roughness.chezy', 'roughness.manning', 'roughness.white_colebrook',
    'roughness.white_colebrook_waqua'))
OPTIONAL_ARTIFACTS = frozenset((
    DIFFUSION_ARTIFACT, AVALANCHE_ARTIFACT, FORCING_ARTIFACT,
    EXECUTION_PROFILE_ARTIFACT))
_EXPLICIT_COMPACT = frozenset(('transport_checkpoint_schema',
                               'transport_payload_sha256'))
_IMPLICIT_COMPACT = frozenset(('implicit_predictor_rate',
                               'implicit_have_predictor'))
# ``gpu_frame_io`` uses this union only as an allow-list.  ``validate_frame``
# below chooses the exact inventory from the authenticated method identity, so
# historical schema-1 checkpoints do not silently acquire schema-2 fields.
COMPACT = (_EXPLICIT_COMPACT | _IMPLICIT_COMPACT |
           frozenset(EXECUTION_PROFILE_IDENTITY_FIELDS))
ADAPTATION = frozenset(('bedload_adaptation_equilibrium', 'bedload_adaptation_capacity',
                       'bedload_adaptation_iterations', 'bedload_adaptation_residual'))
SUSPENDED = frozenset(('suspended', 'ledger_suspended', 'ledger_exchange'))
BEDFORM = frozenset(('bedform_effective_manning', 'bedform_dune_height',
                     'bedform_dune_length', 'bedform_total_ks', 'bedform_iterations'))
GAIA_DIAGNOSTIC_FIELDS = (
    'active','eta_grad_x','eta_grad_y','uphill_bed_grad_x','uphill_bed_grad_y',
    'tau_b_Pa','slope_stress_x_Pa','slope_stress_y_Pa','secondary_delta_x_Pa',
    'secondary_delta_y_Pa','total_stress_x_Pa','total_stress_y_Pa',
    'secondary_turn_rad','total_turn_from_flow_rad','base_capacity_x_kg_m_s',
    'base_capacity_y_kg_m_s','corrected_capacity_x_kg_m_s',
    'corrected_capacity_y_kg_m_s','slope_magnitude_factor')
GAIA_DIAGNOSTICS = frozenset(('gaia_secondary_diagnostic_schema',
                              'gaia_secondary_diagnostic'))

_IMPLICIT_TOKEN = 'implicit_fwave_midpoint'
_IMPLICIT_LABEL = gpu_integrator_label(_IMPLICIT_TOKEN)
_IMPLICIT_VARIANT = VARIANT
_IMPLICIT_BEDLOAD_LAWS = frozenset(
    admitted_legacy_codes('capacity', 'implicit_gpu'))
_IMPLICIT_SCHEMA_V2_BEDLOAD_LAWS = frozenset((2, 8, 11))
_IMPLICIT_BED_SLOPE_MODES = admitted_legacy_codes('bed_slope', 'implicit_gpu')
_IMPLICIT_SECONDARY_MODELS = admitted_legacy_codes('secondary_flow', 'implicit_gpu')
_IMPLICIT_CAPABILITY = {
    'label': _IMPLICIT_LABEL,
    'schema': 1,
    'variants': [_IMPLICIT_VARIANT],
    'launch_modes': ['plain'],
    'checkpoint_schema': 2,
    'checkpoint_schemas': [2, 3, 4, 5, 6, 7, 8],
    'component_coupling_schemas': [1, 2, 8, 9, 10, 11],
    'bedload_laws': sorted(_IMPLICIT_BEDLOAD_LAWS),
    'execution_profile_v2_bedload_laws': sorted(
        _IMPLICIT_SCHEMA_V2_BEDLOAD_LAWS),
    'bed_slope_modes': list(_IMPLICIT_BED_SLOPE_MODES),
    'secondary_flow_models': list(_IMPLICIT_SECONDARY_MODELS),
}
_IMPLICIT_COMPONENT_METHOD = {
    'component_coupling_schema': 2,
    'secondary_source_split': 'symmetric_private_half_steps_v2',
    'secondary_advection': 'converged_midpoint_water_faces_v2',
    'momentum_diffusion': 'resident_post_source_velocity_laplacian_v2',
}


def implicit_coupling_identity(data):
    """CPU-only mirror; preserve historical checkpoint semantics."""
    if int(data['bedlaw']) == 0:
        return 'hydraulic_only_v1'
    marker = data.get(_M1_PUBLIC_ADMISSION_KEY)
    if PROGNOSTIC_SECONDARY_EXECUTION_KEY in data:
        return 'weak_midpoint_q3h3_classic_flow_prognostic_secondary_composition_v1'
    if SECONDARY_EXECUTION_KEY in data:
        return 'weak_midpoint_q3h2_classic_flow_equilibrium_secondary_composition_v1'
    if (isinstance(marker, dict) and
            marker.get('admission_id') == _Q3H1_PUBLIC_ADMISSION_ID and
            int(data['bedlaw']) in (2, 11)):
        return 'weak_midpoint_q3h1_hiding_capacity_composition_v1'
    if (isinstance(marker, dict) and
            marker.get('admission_id') == _Q3C_PUBLIC_ADMISSION_ID and
            int(data['bedlaw']) in (2, 8, 11)):
        return 'weak_midpoint_q3c_scalar_capacity_composition_v1'
    m1_schema = int(data.get('m1_profile_runtime_schema', 1))
    if int(data['bedlaw']) == 11 and int(data.get('exchange_mode', 0)) == 4:
        return ('weak_midpoint_m2_law11_flow2dh_channel_composition_v2'
                if m1_schema == 2 else
                'weak_midpoint_m2_law11_flow2dh_v1')
    if int(data['bedlaw']) == 11:
        return ('weak_midpoint_m1_general_channel_composition_v2'
                if m1_schema == 2 else 'weak_midpoint_m1_general_v1')
    if int(data['bedlaw']) in (5, 6, 7):
        return 'weak_midpoint_m3_rp_taub_v1'
    if (int(data['bedlaw']) == 2 and int(data['bed_slope']) == 0 and
            int(data['secondary_flow_model']) == 0 and
            float(data.get('horizontal_viscosity', 0.)) == 0.):
        return 'weak_midpoint_bedload_v1'
    return 'weak_midpoint_delft_components_v2'


def implicit_checkpoint_schema(data):
    """Mirror the worker ABI without importing or initializing CUDA."""
    coupling = implicit_coupling_identity(data)
    if coupling == 'weak_midpoint_q3h3_classic_flow_prognostic_secondary_composition_v1':
        return 8
    if coupling == 'weak_midpoint_q3h2_classic_flow_equilibrium_secondary_composition_v1':
        return 7
    if coupling == 'weak_midpoint_q3h1_hiding_capacity_composition_v1':
        return 6
    if coupling == 'weak_midpoint_q3c_scalar_capacity_composition_v1':
        return 5
    if coupling in (
            'weak_midpoint_m1_general_channel_composition_v2',
            'weak_midpoint_m2_law11_flow2dh_channel_composition_v2'):
        return 4
    if coupling == 'weak_midpoint_m2_law11_flow2dh_v1':
        return 3
    return 2


_Q3C_COMPONENT_METHOD = {
    'component_coupling_schema': 8,
    'capacity_dispatch': 'resident_q3c_scalar_capacity_laws_2_8_11_v1',
    'capacity_raw_channels': 'resident_bed_and_suspended_diagnostics_v1',
    'capacity_composition': 'resident_delft_channel_composer_before_faces_v1',
    'bed_capacity_channel': 'resident_reconstructed_face_exner_v1',
    'suspended_capacity_channel': 'composed_once_into_total_bed_route_v1',
    'exner_transaction': 'whole_candidate_accept_or_rollback_v1',
}
_Q3H1_COMPONENT_METHOD = {
    **_Q3C_COMPONENT_METHOD,
    'component_coupling_schema': 9,
    'hiding_dispatch': 'resident_q3h1_hiding_exposure_modes_2_3_4_5_v1',
    'hiding_stage_owner':
        'transactional_trial_surface_once_per_stage_commit_on_accept_v1',
    'hiding_boundary_owner':
        'owner_indexed_factors_for_trace_capacity_v1',
}
_Q3H2_COMPONENT_METHOD = {
    **_Q3C_COMPONENT_METHOD,
    'component_coupling_schema': 10,
    'secondary_dispatch':
        'resident_q3h2_classic_flow_equilibrium_secondary_v1',
    'secondary_stage_owner':
        'every_transactional_trial_stage_before_hiding_capacity_v1',
    'secondary_cell_rotation':
        'raw_bed_after_capacity_hiding_before_composer_v1',
    'secondary_boundary_rotation':
        'roles12_trace_owner_equilibrium_role3_rotated_owner_v1',
    'secondary_suspended_rotation': False,
    'secondary_legacy_rotation_disabled': True,
}
_Q3H3_COMPONENT_METHOD = {
    **_Q3C_COMPONENT_METHOD,
    'component_coupling_schema': 11,
    'secondary_dispatch':
        'resident_q3h3_classic_flow_prognostic_secondary_v1',
    'secondary_transaction':(
        'accepted_qold_then_post_hydraulic_qnew_then_fct_'
        'advection_then_full_classic_BE_reaction_v1'),
    'secondary_depth_pair':
        'accepted_old_depth_post_hydraulic_new_depth_v1',
    'secondary_source_forcing':
        'new_velocity_raw_curvature_typed_chezy_v1',
    'secondary_boundary':(
        'adjacent_interior_classic_equilibrium_all_open_flow_directions_v1'),
    'secondary_stage_owner':
        'one_transaction_after_hydraulics_before_capacity_v1',
    'secondary_cell_rotation':(
        'final_transported_hI_raw_bed_after_capacity_hiding_'
        'before_composer_once_v1'),
    'secondary_cell_rotation_forcing':
        'weak_midpoint_depth_velocity_and_capacity_typed_chezy_v1',
    'secondary_source_split':'single_full_step_BE_after_advection_v1',
    'secondary_generic_model2_prepost_source_disabled':True,
    'secondary_generic_model2_filter_disabled':True,
    'secondary_suspended_rotation': False,
    'secondary_legacy_rotation_disabled': True,
}
_IMPLICIT_METHOD = {
    'implicit_midpoint_schema': 1,
    'nonlinear_rtol': 5.e-2,
    'nonlinear_atol': 1.e-14,
    'nonlinear_maxiter': 5,
    'inner_tolerance_fraction': .2,
    'linear_maxiter': 400,
    'advective_cfl_limit': .8,
    'froude_limit': .65,
    'manning_split': 'symmetric_half_step_resident_v1',
    'open_boundary_jacobian': 'frozen_trace_full_residual_v1',
    'full_residual': 'existing_augmented_fwave_faces_v1',
    'approximate_jacobian': 'frozen_acoustic_upwind_v1',
    'timestep_controller': 'advective_froude_envelope_v1',
    'automatic_explicit_fallback': False,
    'automatic_CPU_fallback': False,
}


def _gaia_diagnostic_method():
    return dict(schema=1, fields=list(GAIA_DIAGNOSTIC_FIELDS),
        clock='accepted_output_frame', eta_gradient='grad(z+h)',
        bed_gradient='grad(z)_positive_uphill',
        decomposition='slope_stress+secondary_delta=total_stress',
        angle='signed_counter_clockwise_radians_minus_pi_to_pi',
        capacity='cell_centred_equilibrium_not_accepted_face_flux',
        checkpointed=False)


def suspended_enabled(data):
    return has_suspended_transport(data['exchange_mode'],
        np.asarray(data['state'])[np.asarray(data['suspended_indices'])],
        data['fraction_concentration'], data.get('fraction_series_concentration', ()),
        explicit_active=data.get('suspended_transport_active', False))


def require(ok, message):
    if not ok: raise ValueError('GPU live transport: '+message)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_object(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result,
                    'duplicate execution-profile sidecar key: '+str(key))
            result[key] = value
        return result
    try:
        value = json.loads(path.read_bytes().decode('ascii'),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError('non-finite execution-profile JSON: '+token)))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError('GPU live transport: invalid execution-profile sidecar JSON') from error
    require(isinstance(value, dict), 'execution-profile sidecar must be an object')
    return value


def _v55_hiding_identity(metadata, deck_lines, *, nested=False):
    """Authenticate Q3-H1's outer wire without reinterpreting Q3-C v54."""
    raw = metadata.get('hiding_foundation_identity')
    require(isinstance(raw, dict) and set(raw) == {
        'envelope_version', 'canonical_preimage_ascii',
        'canonical_preimage_sha256', HIDING_EXECUTION_KEY,
        HIDING_ORDERED_ROUTE_KEY},
        'v55 hiding-foundation metadata differs')
    identity = validate_hiding_foundation_execution_document(
        raw[HIDING_EXECUTION_KEY])
    preimage = raw['canonical_preimage_ascii']
    require(raw['envelope_version'] == 55 and isinstance(preimage, str) and
            preimage.isascii() and _canonical_json(identity) == preimage and
            hashlib.sha256(preimage.encode('ascii')).hexdigest() ==
                raw['canonical_preimage_sha256'],
            'v55 hiding-foundation preimage differs')
    expected_route = {
        'ordered_parts': [
            'capacity_execution_fingerprint',
            'capacity_profile_fingerprint',
            'hiding_execution_fingerprint'],
        'capacity_execution_fingerprint': identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint': identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint': identity['fingerprint'],
    }
    base = metadata.get('capacity_route_identity')
    require(raw[HIDING_ORDERED_ROUTE_KEY] == expected_route and
            isinstance(base, dict) and base.get('envelope_version') == 54 and
            base.get('capacity_execution_fingerprint') ==
                identity['capacity_execution_fingerprint'] and
            base.get('capacity_profile_fingerprint') ==
                identity['capacity_profile_fingerprint'] and
            metadata.get('bedload_law') == identity['capacity_law'],
            'v55 ordered hiding/Q3-C route differs')
    header_index = 1 if nested else 2
    require(len(deck_lines) > header_index + 1 and
            deck_lines[0].strip() == '55',
            'v55 deck envelope is incomplete')
    try:
        schema, byte_count, chunk_count = (
            int(token) for token in deck_lines[header_index].split())
    except (TypeError, ValueError) as error:
        raise ValueError('GPU live transport: v55 hiding header differs') from error
    require(schema == 1 and byte_count == len(preimage) and chunk_count > 0 and
            len(deck_lines) > header_index + 1 + chunk_count,
            'v55 hiding header differs')
    chunks = deck_lines[header_index + 1:header_index + 1 + chunk_count]
    require(all(re.fullmatch(r'[0-9a-f]+', chunk) is not None and
                len(chunk) % 2 == 0 for chunk in chunks),
            'v55 hiding payload is not lowercase hexadecimal')
    try:
        decoded = bytes.fromhex(''.join(chunks)).decode('ascii')
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError('GPU live transport: v55 hiding payload is invalid') from error
    require(len(decoded) == byte_count and decoded == preimage and
            deck_lines[header_index + 1 + chunk_count].strip() == '54',
            'v55 does not wrap its exact v54 foundation')
    return identity, expected_route


def _v56_secondary_identity(metadata, deck_lines):
    """Authenticate v56 and its exact v54 or v55 nested foundation."""
    raw=metadata.get('secondary_foundation_identity')
    require(isinstance(raw,dict) and set(raw)=={
        'envelope_version','canonical_preimage_ascii',
        'canonical_preimage_sha256',SECONDARY_EXECUTION_KEY,
        SECONDARY_ORDERED_ROUTE_KEY},
        'v56 secondary-foundation metadata differs')
    identity=validate_secondary_foundation_execution_document(
        raw[SECONDARY_EXECUTION_KEY])
    preimage=raw['canonical_preimage_ascii']
    require(raw['envelope_version']==56 and isinstance(preimage,str) and
            preimage.isascii() and _canonical_json(identity)==preimage and
            hashlib.sha256(preimage.encode('ascii')).hexdigest()==
                raw['canonical_preimage_sha256'],
            'v56 secondary-foundation preimage differs')
    route={
        'ordered_parts':['capacity_execution_fingerprint',
                         'capacity_profile_fingerprint',
                         'hiding_execution_fingerprint',
                         'secondary_execution_fingerprint'],
        'capacity_execution_fingerprint':identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint':identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint':identity[
            'hiding_execution_fingerprint'],
        'secondary_execution_fingerprint':identity['fingerprint'],
    }
    base=metadata.get('capacity_route_identity')
    require(raw[SECONDARY_ORDERED_ROUTE_KEY]==route and
            isinstance(base,dict) and base.get('envelope_version')==54 and
            base.get('capacity_execution_fingerprint')==
                identity['capacity_execution_fingerprint'] and
            base.get('capacity_profile_fingerprint')==
                identity['capacity_profile_fingerprint'] and
            metadata.get('bedload_law')==identity['capacity_law'],
            'v56 ordered secondary/foundation route differs')
    require(len(deck_lines)>=4 and deck_lines[0].strip()=='56',
            'v56 deck envelope is incomplete')
    try:
        schema,byte_count,chunk_count=(int(token)
                                      for token in deck_lines[2].split())
    except (TypeError,ValueError) as error:
        raise ValueError(
            'GPU live transport: v56 secondary header differs') from error
    require(schema==1 and byte_count==len(preimage) and chunk_count>0 and
            len(deck_lines)>3+chunk_count,
            'v56 secondary header differs')
    chunks=deck_lines[3:3+chunk_count]
    require(all(re.fullmatch(r'[0-9a-f]+',chunk) is not None and
                len(chunk)%2==0 for chunk in chunks),
            'v56 secondary payload is not lowercase hexadecimal')
    try:decoded=bytes.fromhex(''.join(chunks)).decode('ascii')
    except (UnicodeDecodeError,ValueError) as error:
        raise ValueError(
            'GPU live transport: v56 secondary payload is invalid') from error
    base_index=3+chunk_count
    expected_base=str(identity['base_envelope_version'])
    require(len(decoded)==byte_count and decoded==preimage and
            deck_lines[base_index].strip()==expected_base,
            'v56 does not wrap its declared v54/v55 foundation')
    hiding_identity=hiding_route=None
    if identity['base_envelope_version']==55:
        hiding_identity,hiding_route=_v55_hiding_identity(
            metadata,deck_lines[base_index:],nested=True)
        require(hiding_identity['fingerprint']==
                    identity['hiding_execution_fingerprint'],
                'v56 hiding fingerprint differs from its nested v55 foundation')
    else:
        require(metadata.get('hiding_foundation_identity') is None and
                identity['hiding_execution_fingerprint'] is None,
                'v56/v54 foundation unexpectedly carries Q3-H1 hiding')
    return identity,route,hiding_identity,hiding_route


def _v57_prognostic_secondary_identity(metadata,deck_lines):
    """Authenticate v57 directly over one intact v54/v55 foundation."""
    raw=metadata.get('prognostic_secondary_identity')
    require(isinstance(raw,dict) and set(raw)=={
        'envelope_version','canonical_preimage_ascii',
        'canonical_preimage_sha256',
        'prognostic_secondary_execution_fingerprint',
        'prognostic_secondary_bed_boundary_rotation_policy',
        'prognostic_secondary_state_units',
        'prognostic_secondary_integrated_ledger_units',
        PROGNOSTIC_SECONDARY_EXECUTION_KEY,
        PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY},
        'v57 prognostic-secondary metadata differs')
    identity=validate_prognostic_secondary_execution_document(
        raw[PROGNOSTIC_SECONDARY_EXECUTION_KEY])
    preimage=raw['canonical_preimage_ascii']
    require(raw['envelope_version']==57 and isinstance(preimage,str) and
            preimage.isascii() and _canonical_json(identity)==preimage and
            raw['prognostic_secondary_execution_fingerprint']==
                identity['fingerprint'] and
            raw['prognostic_secondary_bed_boundary_rotation_policy']==
                identity['bed_boundary_rotation_policy'] and
            raw['prognostic_secondary_state_units']=='m2_s' and
            raw['prognostic_secondary_integrated_ledger_units']==
                identity['integrated_ledger_units'] and
            hashlib.sha256(preimage.encode('ascii')).hexdigest()==
                raw['canonical_preimage_sha256'],
            'v57 prognostic-secondary preimage/visible receipt differs')
    route={
        'capacity_execution_fingerprint':identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint':identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint':identity[
            'hiding_execution_fingerprint'],
        'prognostic_secondary_execution_fingerprint':identity['fingerprint'],
    }
    base=metadata.get('capacity_route_identity')
    require(raw[PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY]==route and
            metadata.get('secondary_foundation_identity') is None and
            isinstance(base,dict) and base.get('envelope_version')==54 and
            base.get('capacity_execution_fingerprint')==
                identity['capacity_execution_fingerprint'] and
            base.get('capacity_profile_fingerprint')==
                identity['capacity_profile_fingerprint'] and
            metadata.get('bedload_law')==identity['capacity_law'],
            'v57 ordered prognostic/foundation route differs')
    require(len(deck_lines)>=4 and deck_lines[0].strip()=='57',
            'v57 deck envelope is incomplete')
    try:
        schema,byte_count,chunk_count=(
            int(token) for token in deck_lines[2].split())
    except (TypeError,ValueError) as error:
        raise ValueError(
            'GPU live transport: v57 prognostic header differs') from error
    require(schema==1 and byte_count==len(preimage) and chunk_count>0 and
            len(deck_lines)>3+chunk_count,
            'v57 prognostic header differs')
    chunks=deck_lines[3:3+chunk_count]
    require(all(re.fullmatch(r'[0-9a-f]+',chunk) is not None and
                len(chunk)%2==0 for chunk in chunks),
            'v57 prognostic payload is not lowercase hexadecimal')
    try:
        decoded=bytes.fromhex(''.join(chunks)).decode('ascii')
    except (UnicodeDecodeError,ValueError) as error:
        raise ValueError(
            'GPU live transport: v57 prognostic payload is invalid') from error
    base_index=3+chunk_count
    require(len(decoded)==byte_count and decoded==preimage and
            deck_lines[base_index].strip()==
                str(identity['base_envelope_version']),
            'v57 does not directly wrap its declared v54/v55 foundation')
    hiding_identity=hiding_route=None
    if identity['base_envelope_version']==55:
        hiding_identity,hiding_route=_v55_hiding_identity(
            metadata,deck_lines[base_index:],nested=True)
        require(hiding_identity['fingerprint']==
                    identity['hiding_execution_fingerprint'],
                'v57 hiding fingerprint differs from nested v55')
    else:
        require(metadata.get('hiding_foundation_identity') is None and
                identity['hiding_execution_fingerprint'] is None,
                'v57/v54 foundation unexpectedly carries hiding')
    return identity,route,hiding_identity,hiding_route


def _native_identity_ascii(data, key):
    value = np.asarray(data.get(key))
    require(value.shape == (64,) and value.dtype.kind in 'iu' and
            np.all((value >= 32) & (value <= 126)),
            'native Q3-H1 identity record is absent or malformed: '+key)
    return bytes(int(item) for item in value).decode('ascii').rstrip()


def _validate_native_v55_hiding(data, identity):
    """Bind the native parser's authenticated v55 result to host metadata."""
    scalar = lambda key: np.asarray(data.get(key))
    for key in ('native_q3h1_identity_verified',
                'native_q3h1_identity_envelope',
                'native_q3h1_formula_code', 'native_q3h1_hiding_mode'):
        value = scalar(key)
        require(value.shape == () and value.dtype.kind in 'iu',
                'native Q3-H1 scalar record differs: '+key)
    for key in ('native_q3h1_parker_alpha', 'native_q3h1_wu_exponent'):
        value = scalar(key)
        require(value.shape == () and value.dtype.kind == 'f' and
                np.isfinite(value.item()),
                'native Q3-H1 scalar record differs: '+key)
    require(int(scalar('native_q3h1_identity_verified')) == 1 and
            int(scalar('native_q3h1_identity_envelope')) == 55 and
            int(scalar('native_q3h1_formula_code')) == identity['capacity_law'] and
            int(scalar('native_q3h1_hiding_mode')) == identity['mode'] and
            float(scalar('native_q3h1_parker_alpha')) == identity['parker_alpha'] and
            float(scalar('native_q3h1_wu_exponent')) == identity['wu_exponent'] and
            _native_identity_ascii(
                data, 'native_q3h1_execution_fingerprint_ascii') ==
                identity['fingerprint'] and
            _native_identity_ascii(
                data, 'native_q3c_execution_fingerprint_ascii') ==
                identity['capacity_execution_fingerprint'] and
            _native_identity_ascii(
                data, 'native_q3c_profile_fingerprint_ascii') ==
                identity['capacity_profile_fingerprint'],
            'native v55 hiding identity differs from authenticated metadata')


def _validate_native_v56_secondary(data,identity):
    """Cross-check the v17 zero-step parser records against canonical v56."""
    scalar=lambda key:np.asarray(data.get(key))
    for key in ('native_q3h2_secondary_verified',
                'native_q3h2_secondary_envelope','native_q3h2_model'):
        value=scalar(key)
        require(value.shape==() and value.dtype.kind in 'iu',
                'native Q3-H2 scalar record differs: '+key)
    require(int(scalar('native_q3h2_secondary_verified'))==1 and
            int(scalar('native_q3h2_secondary_envelope'))==56 and
            int(scalar('native_q3h2_model'))==1 and
            _native_identity_ascii(data,
                'native_q3h2_execution_fingerprint_ascii')==
                    identity['fingerprint'] and
            _native_identity_ascii(data,
                'native_q3h2_capacity_execution_fingerprint_ascii')==
                    identity['capacity_execution_fingerprint'] and
            _native_identity_ascii(data,
                'native_q3h2_capacity_profile_fingerprint_ascii')==
                    identity['capacity_profile_fingerprint'] and
            _native_identity_ascii(data,
                'native_q3h2_hiding_execution_fingerprint_ascii')==
                    (identity['hiding_execution_fingerprint'] or '') and
            _native_identity_ascii(data,
                'native_q3h2_roughness_fingerprint_ascii')==
                    identity['roughness_fingerprint'] and
            _native_identity_ascii(data,'native_q3h2_method_id_ascii')==
                    identity['method_id'] and
            _native_identity_ascii(data,
                'native_q3h2_canonical_preimage_sha256_ascii')==
                    hashlib.sha256(_canonical_json(identity).encode(
                        'ascii')).hexdigest(),
            'native v56 secondary identity differs from authenticated metadata')


def _native_q3h3_ascii(data,key):
    value=np.asarray(data.get(key))
    require(value.shape==(64,) and value.dtype.kind in 'iu' and
            np.all((value>=32)&(value<=126)),
            'native Q3-H3 identity record is absent or malformed: '+key)
    return bytes(int(item) for item in value).decode('ascii').rstrip()


def _validate_native_v57_prognostic_secondary(data,identity):
    """Bind v18 zero-step native records to the canonical v57 identity."""
    identity=validate_prognostic_secondary_execution_document(identity)
    scalar=lambda key:np.asarray(data.get(key))
    integer_keys=(
        'native_q3h3_secondary_verified','native_q3h3_secondary_envelope',
        'native_q3h3_model','native_q3h3_base_envelope',
        'native_q3h3_transport_direction','native_q3h3_inflow_mode')
    real_keys=(
        'native_q3h3_rmincf','native_q3h3_chzmin','native_q3h3_espir',
        'native_q3h3_coriolis_constant_s_inverse','native_q3h3_betac',
        'native_q3h3_scalar_diffusivity_m2_s',
        'native_q3h3_von_karman_constant')
    for key in integer_keys:
        value=scalar(key)
        require(value.shape==() and value.dtype.kind in 'iu',
                'native Q3-H3 scalar differs: '+key)
    for key in real_keys:
        value=scalar(key)
        require(value.shape==() and value.dtype.kind=='f' and
                np.isfinite(value.item()),
                'native Q3-H3 scalar differs: '+key)
    hiding=identity['hiding_execution_fingerprint'] or ''
    require(
        int(scalar('native_q3h3_secondary_verified'))==1 and
        int(scalar('native_q3h3_secondary_envelope'))==57 and
        int(scalar('native_q3h3_model'))==2 and
        int(scalar('native_q3h3_base_envelope'))==
            identity['base_envelope_version'] and
        _native_q3h3_ascii(data,'native_q3h3_execution_fingerprint_ascii')==
            identity['fingerprint'] and
        _native_q3h3_ascii(
            data,'native_q3h3_capacity_execution_fingerprint_ascii')==
            identity['capacity_execution_fingerprint'] and
        _native_q3h3_ascii(
            data,'native_q3h3_capacity_profile_fingerprint_ascii')==
            identity['capacity_profile_fingerprint'] and
        _native_q3h3_ascii(
            data,'native_q3h3_hiding_execution_fingerprint_ascii')==hiding and
        _native_q3h3_ascii(data,'native_q3h3_roughness_fingerprint_ascii')==
            identity['roughness_fingerprint'] and
        _native_q3h3_ascii(data,'native_q3h3_method_id_ascii')==
            identity['method_id'] and
        _native_q3h3_ascii(
            data,'native_q3h3_canonical_preimage_sha256_ascii')==
            hashlib.sha256(_canonical_json(identity).encode('ascii')).hexdigest() and
        float(scalar('native_q3h3_rmincf'))==identity['rmincf'] and
        float(scalar('native_q3h3_chzmin'))==identity['chzmin'] and
        float(scalar('native_q3h3_espir'))==identity['espir'] and
        float(scalar('native_q3h3_coriolis_constant_s_inverse'))==
            identity['coriolis_constant_s_inverse'] and
        float(scalar('native_q3h3_betac'))==identity['betac'] and
        float(scalar('native_q3h3_scalar_diffusivity_m2_s'))==
            identity['scalar_diffusivity_m2_s'] and
        float(scalar('native_q3h3_von_karman_constant'))==
            identity['von_karman_constant'] and
        int(scalar('native_q3h3_transport_direction'))==1 and
        int(scalar('native_q3h3_inflow_mode'))==identity['inflow_mode'],
        'native v57 prognostic-secondary identity differs from authenticated metadata')


def _install_inline_secondary_identity(data,identity,route):
    identity=validate_secondary_foundation_execution_document(identity)
    expected={
        'ordered_parts':['capacity_execution_fingerprint',
                         'capacity_profile_fingerprint',
                         'hiding_execution_fingerprint',
                         'secondary_execution_fingerprint'],
        'capacity_execution_fingerprint':identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint':identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint':identity[
            'hiding_execution_fingerprint'],
        'secondary_execution_fingerprint':identity['fingerprint'],
    }
    require(route==expected and identity['method_id']=='secondary.equilibrium' and
            identity['model']==1 and identity['capacity_law']==int(data['bedlaw']) and
            identity['roughness_method_id']==
                data.get('capacity_roughness_method_id') and
            identity['roughness_fingerprint']==
                data.get('delft_forcing_fingerprint') and
            identity['coriolis_disposition']=='zero' and
            identity['coriolis_constant_s_inverse']==0. and
            identity['coriolis_source']=='constant_zero_planar' and
            identity['betac']==0. and
            int(data.get('secondary_flow_model',0))==0 and
            int(data.get('secondary_flow',0))==0 and
            int(data.get('secondary_flow_transport_direction',0))==0 and
            float(data.get('secondary_flow_momentum_beta',0.))==0. and
            float(data.get('secondary_flow_espir',1.))==1.,
            'v56 secondary identity differs from nested native controls')
    data.update(secondary_flow_model=1,secondary_flow=1,
                secondary_flow_transport_direction=1,
                secondary_flow_momentum_beta=0.,
                secondary_flow_espir=identity['espir'])
    data[SECONDARY_EXECUTION_KEY]=identity
    data[SECONDARY_ORDERED_ROUTE_KEY]=route


def _install_inline_prognostic_secondary_identity(data,identity,route):
    """Install an authenticated v57 identity without legacy model-2 fallback."""
    identity=validate_prognostic_secondary_execution_document(identity)
    expected={
        'capacity_execution_fingerprint':identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint':identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint':identity[
            'hiding_execution_fingerprint'],
        'prognostic_secondary_execution_fingerprint':identity['fingerprint'],
    }
    require(route==expected and identity['method_id']=='secondary.prognostic' and
            identity['model']==2 and identity['capacity_law']==int(data['bedlaw']) and
            identity['roughness_method_id']==
                data.get('capacity_roughness_method_id') and
            identity['roughness_fingerprint']==
                data.get('delft_forcing_fingerprint') and
            identity['coriolis_disposition']=='zero' and
            identity['coriolis_constant_s_inverse']==0. and
            identity['coriolis_source']=='constant_zero_planar' and
            identity['betac']==0. and
            identity['scalar_diffusivity_m2_s']==0. and
            int(data.get('secondary_flow_model',-1))==2 and
            int(data.get('secondary_flow',0))==1 and
            int(data.get('secondary_flow_transport_direction',0))==1 and
            int(data.get('secondary_flow_inflow_mode',-1))==1 and
            float(data.get('secondary_flow_momentum_beta',np.nan))==0. and
            float(data.get('secondary_flow_espir',np.nan))==identity['espir'],
            'v57 prognostic-secondary identity differs from native controls')
    layout=identity['runtime_layout']
    state=np.asarray(data.get('state'))
    require(state.shape==(layout['runtime_state_count'],int(data['nc'])),
            'v57 prognostic-secondary state layout differs from identity')
    secondary=np.ascontiguousarray(
        state[layout['secondary_state_index']])
    require('secondary_mass' not in data or
            np.array_equal(np.asarray(data['secondary_mass']),secondary),
            'v57 prognostic-secondary derived hI state differs')
    # ``secondary_mass`` is a runtime/viewer alias, not an additional native
    # record.  Derive it only after the archived native inventory has been
    # authenticated byte-for-byte.
    data['secondary_mass']=secondary
    data[PROGNOSTIC_SECONDARY_EXECUTION_KEY]=identity
    data[PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY]=route


def _component_rows(profile):
    return [dict(component_id=row.component_id, schema=row.schema,
        scope_id=row.scope_id, semantic_id=row.semantic_id,
        fingerprint=row.fingerprint, document_sha256=row.document_sha256)
        for row in profile.component_profiles]


def _resolved_execution_profile_data(project, sediment, profile):
    """Translate authenticated host/native metadata to worker scalar IDs."""
    def numeric(value, nested_key):
        return value.get(nested_key) if isinstance(value, dict) else value

    secondary = sediment.get('secondary_flow')
    require(isinstance(secondary, dict),
            'execution-profile preparation lacks secondary-flow identity')
    secondary_model = 0
    if secondary.get('enabled') is True:
        secondary_model = {
            'equilibrium': 1, 'prognostic': 2, 'gaia': 3,
        }.get(secondary.get('model'))
    shear = project.get('sediment_shear', {'mode': 'total'})
    shear_mode = ({'total': 0, 'soulsby_clarke_skin': 1}.get(
        shear.get('mode')) if isinstance(shear, dict) else None)
    fractions = sediment.get('fraction_ids')
    profile_fractions = [row.fraction_id for row in profile.fraction_layout]
    require(isinstance(fractions, list) and secondary_model is not None and
            shear_mode is not None,
            'execution-profile preparation has unrepresented numeric methods')
    # The native single-material layout historically stores no explicit
    # fraction-ID vector even though its conserved sediment channel is the
    # canonical ``single`` fraction.  All multi-fraction decks must match
    # exactly; this one representation equivalence is explicit and bounded.
    require(fractions == profile_fractions or
            (fractions == [] and profile_fractions == ['single']),
            'execution-profile fractions differ from prepared native layout')
    roughness = project.get('roughness', {})
    roughness_mode = roughness.get('mode', 'manning') if isinstance(
        roughness, dict) else None
    roughness_method = {
        'chezy': 'roughness.chezy',
        'manning': 'roughness.manning',
        'white_colebrook': ('roughness.white_colebrook_waqua'
            if roughness.get('white_colebrook_chzk2d', False) is True else
            'roughness.white_colebrook'),
    }.get(roughness_mode)
    return {
        'nfrac': len(profile_fractions),
        'bedlaw': sediment.get('bedload_law'),
        'exchange_mode': sediment.get('suspended_exchange_mode'),
        'native_sediment_shear_mode': shear_mode,
        'hiding_exposure': numeric(sediment.get('hiding_exposure'), 'mode'),
        'bed_slope': numeric(sediment.get('bed_slope'), 'islope'),
        'secondary_flow_model': secondary_model,
        'capacity_roughness_method_id': roughness_method,
    }


def _validate_execution_profile_worker_scope(profile, worker_scope):
    """Require the packaged worker to carry this exact validator and ledger."""
    package = worker_scope.get('sediment_execution_profile_v2')
    ledger = load_selector_ledger()
    current_registry = ledger.get('admission_registry_id')
    historical = is_frozen_q3sp_execution_profile(profile)
    completeness_source = Path(
        validate_fixed_grid_profile_completeness.__code__.co_filename)
    require(isinstance(package, dict) and
            package.get('semantic_id') == profile.semantic_id and
            package.get('ledger_semantic_id') == ledger.get('semantic_id') and
            (historical or profile.ledger_semantic_id == ledger.get('semantic_id')) and
            package.get('admission_registry_id') == current_registry and
            package.get('ledger_sha256') == _file_sha256(SELECTOR_LEDGER_FILE) and
            package.get('completeness_source') ==
            'braids_core/sediment/fixed_grid_profile_completeness.py' and
            completeness_source.is_file() and
            package.get('completeness_source_sha256') ==
            _file_sha256(completeness_source) and
            BACKEND_GPU in package.get('backends', ()),
            'selected worker has a different execution-profile validator/ledger')


def validate_q3h3_worker_source_scope(worker_scope):
    """Require the qualified resident source pair and v18 initializer route."""
    generations = (worker_scope.get('native_exporter_generations_by_variant')
                   if isinstance(worker_scope, dict) else None)
    transport_generations = (generations.get(VARIANT)
                             if isinstance(generations, dict) else None)
    require(isinstance(worker_scope, dict) and
            worker_scope.get(_Q3H3_WORKER_SOURCE_KEY) ==
                _Q3H3_WORKER_SOURCE_CONTRACT and
            isinstance(transport_generations, dict) and
            transport_generations.get('delft_components_v18_q3h3') ==
                'braids_gpu/native_exporter_v18',
            'selected worker lacks the exact Q3-H3 prognostic-secondary runtime identity')


def _project_requests_q3h3(project):
    """Recognize only the explicit classic prognostic project declaration."""
    if not isinstance(project, dict) or _Q3H3_PROJECT_BINDING_KEY not in project:
        return False
    binding = project.get(_Q3H3_PROJECT_BINDING_KEY)
    require(isinstance(binding, dict) and
            binding.get('execution_scope_id') ==
                _Q3H3_WORKER_SOURCE_CONTRACT['execution_scope_id'] and
            binding.get('method_id') ==
                _Q3H3_WORKER_SOURCE_CONTRACT['method_id'] and
            binding.get('gpu_checkpoint_schema') ==
                _Q3H3_WORKER_SOURCE_CONTRACT['checkpoint_schema'] and
            binding.get('automatic_cpu_fallback') is False,
            'Q3-H3 project binding differs from the explicit classic prognostic route')
    return True


def _require_exact_law11_total_profile(profile, checked_documents):
    require(len(profile.fraction_layout) == 1 and
            profile.fraction_layout[0].transport_class == 1,
            'public law-11 total route requires one class-1 fraction')
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
    require(selected == expected,
            'public schema-v2 preparation is limited to the exact law-11 total-load chain')
    capacity = checked_documents.get('capacity')
    require(isinstance(capacity, dict) and capacity.get('backend') is None and
            len(capacity.get('fractions', ())) == 1 and
            capacity['fractions'][0].get('fraction_id') == fraction_id and
            capacity['fractions'][0].get('method_id') == 'capacity.general' and
            capacity['fractions'][0].get('formula_code') == 11,
            'public law-11 preparation requires its exact neutral capacity component')
    return profile, checked_documents


def _require_exact_q3c_total_profile(profile, checked_documents):
    layout = tuple(profile.fraction_layout)
    require(layout and all(row.transport_class == 1 for row in layout),
            'public Q3-C total route requires class-1 fractions')
    fraction_ids = tuple(row.fraction_id for row in layout)
    capacity = checked_documents.get('capacity')
    channel = checked_documents.get('channel_composition')
    require(isinstance(capacity, dict) and isinstance(channel, dict) and
            capacity.get('backend') is None and
            [row.get('fraction_id') for row in capacity.get('fractions', ())] ==
                list(fraction_ids) and
            channel.get('fractions') == [
                {'fraction_id': value, 'transport_class': 1}
                for value in fraction_ids],
            'public Q3-C components differ from the fraction layout')
    laws = {row.get('formula_code') for row in capacity['fractions']}
    methods = {row.get('method_id') for row in capacity['fractions']}
    require(len(laws) == 1 and len(methods) == 1 and
            _Q3C_METHODS.get(next(iter(laws))) == next(iter(methods)),
            'public Q3-C capacity formula differs')
    law, method_id = next(iter(laws)), next(iter(methods))
    selected = {(row.entry_id, tuple(row.scope))
                for row in profile.method_selections}
    expected = set()
    for fraction_id in fraction_ids:
        scope = (('fraction_id', fraction_id),)
        expected.update(((method_id, scope), ('transport_route.total', scope),
                         ('suspension.off', scope)))
    expected.update({
        ('shear.total_roughness', (('model_id', 'model'),)),
        ('hiding.none', (('model_id', 'model'),)),
        ('bed_slope.off', (('model_id', 'model'),)),
        ('secondary.off', (('model_id', 'model'),)),
    })
    roughness = {row.entry_id for row in profile.method_selections
                 if row.entry_id in _ROUGHNESS_METHODS and
                 tuple(row.scope) == (('model_id', 'model'),)}
    require(len(roughness) == 1,
            'public Q3-C requires one typed hydraulic-roughness selector')
    expected.add((next(iter(roughness)), (('model_id', 'model'),)))
    require(selected == expected,
            'public Q3-C execution profile has a different operator chain')
    return profile, checked_documents, law, method_id, next(iter(roughness))


def _public_q3c_admission_document(profile, checked_documents):
    profile, checked_documents, law, method_id, roughness = \
        _require_exact_q3c_total_profile(profile, checked_documents)
    require(profile.backend == BACKEND_GPU,
            'public Q3-C admission requires the GPU execution profile')
    bindings = {row.component_id: row for row in profile.component_profiles}
    require(set(bindings) == {'capacity', 'channel_composition'} and
            set(checked_documents) == set(bindings),
            'public Q3-C admission has a different component inventory')
    return {
        'schema': 1, 'admission_id': _Q3C_PUBLIC_ADMISSION_ID,
        'backend': BACKEND_GPU, 'formula_code': law,
        'method_id': method_id,
        'fraction_ids': [row.fraction_id for row in profile.fraction_layout],
        'roughness_method_id': roughness,
        'execution_profile_fingerprint': profile.fingerprint,
        'execution_profile_ledger_semantic_id': profile.ledger_semantic_id,
        'capacity_document_sha256': bindings['capacity'].document_sha256,
        'channel_composition_document_sha256':
            bindings['channel_composition'].document_sha256,
    }


def _public_q3h1_admission_document(profile, checked_documents, identity):
    base = _public_q3c_admission_document(profile, checked_documents)
    identity = validate_hiding_foundation_execution_document(identity)
    require(base['formula_code'] in (2, 11) and
            identity['capacity_law'] == base['formula_code'] and
            len(base['fraction_ids']) == 2 and
            identity['fraction_ids'] == base['fraction_ids'] and
            identity['capacity_profile_fingerprint'] ==
                checked_documents['capacity']['fingerprint'],
            'Q3-H1 admission differs from its Q3-C profile')
    return {
        **base,
        'admission_id': _Q3H1_PUBLIC_ADMISSION_ID,
        'base_admission_id': _Q3C_PUBLIC_ADMISSION_ID,
        'hiding_mode': identity['mode'],
        'hiding_method_id': identity['method_id'],
        'hiding_execution_fingerprint': identity['fingerprint'],
        'hiding_execution_semantic_id': identity['semantic_id'],
    }


def _public_law11_admission_document(profile, checked_documents):
    """Mirror the worker's receipt-bound public admission marker.

    The QGIS host must remain NumPy-only and cannot import the CUDA worker.
    Constructing the marker from the already authenticated execution profile
    keeps that separation while giving the worker a value it can independently
    recompute from its receipt-authenticated sidecar.
    """
    _require_exact_law11_total_profile(profile, checked_documents)
    require(profile.backend == BACKEND_GPU,
            'public law-11 admission requires the GPU execution profile')
    bindings = {row.component_id: row for row in profile.component_profiles}
    require(set(bindings) == {'capacity', 'channel_composition'} and
            set(checked_documents) == set(bindings),
            'public law-11 admission has a different component inventory')
    fraction_id = profile.fraction_layout[0].fraction_id
    return {
        'schema': 1,
        'admission_id': _M1_PUBLIC_ADMISSION_ID,
        'backend': BACKEND_GPU,
        'fraction_id': fraction_id,
        'execution_profile_fingerprint': profile.fingerprint,
        'execution_profile_ledger_semantic_id': profile.ledger_semantic_id,
        'capacity_document_sha256': bindings['capacity'].document_sha256,
        'channel_composition_document_sha256':
            bindings['channel_composition'].document_sha256,
    }


def resolve_public_law11_total_profile(project):
    binding = project.get(EXECUTION_PROFILE_BINDING_KEY)
    require(isinstance(binding, dict) and binding.get('state') == 'canonical',
            'public law-11 GPU preparation requires a canonical schema-v2 profile')
    documents = binding.get('component_documents')
    require(isinstance(documents, dict),
            'schema-v2 execution profile lacks component documents')
    try:
        profile, checked_documents = runtime_execution_profile(
            binding, backend=BACKEND_GPU, component_documents=documents)
        validate_fixed_grid_profile_completeness(
            profile, component_documents=checked_documents)
    except (TypeError, ValueError) as error:
        raise ValueError(
            'GPU live transport: schema-v2 GPU profile is not admitted: ' +
            str(error)) from error
    if any(row.entry_id in _ROUGHNESS_METHODS for row in profile.method_selections):
        checked = _require_exact_q3c_total_profile(profile, checked_documents)
        return checked[0], checked[1]
    return _require_exact_law11_total_profile(profile, checked_documents)


def has_public_law11_total_prepared_profile(data):
    """Return true only for a sidecar-authenticated prepared law-11 profile."""
    if not isinstance(data, dict):
        return False
    try:
        bedlaw = int(data.get('bedlaw', -1))
    except (TypeError, ValueError, OverflowError):
        return False
    if bedlaw not in (2, 8, 11):
        return False
    bound = data.get(_EXECUTION_PROFILE_DATA_KEY)
    if not isinstance(bound, dict):
        return False
    profile = bound.get('profile')
    documents = bound.get('component_documents')
    if profile is None or not isinstance(documents, dict):
        return False
    try:
        if any(row.entry_id in _ROUGHNESS_METHODS
               for row in profile.method_selections):
            hiding = data.get(HIDING_EXECUTION_KEY)
            marker = (_public_q3h1_admission_document(
                          profile, documents, hiding)
                      if hiding is not None else
                      _public_q3c_admission_document(profile, documents))
            if marker['formula_code'] != bedlaw:
                return False
        else:
            if bedlaw != 11:
                return False
            _require_exact_law11_total_profile(profile, documents)
            marker = _public_law11_admission_document(profile, documents)
    except (KeyError, TypeError, ValueError):
        return False
    return (profile.backend == BACKEND_GPU and
            data.get(_M1_PUBLIC_ADMISSION_KEY) == marker and
            'm1_profile_research_override' not in data)


def validate_public_law11_total_preparation(project, worker_scope,
                                             gpu_numerical_variant):
    """Preflight the first public schema-v2 GPU route before mesh creation.

    Q3-S/P deliberately admits exactly one algebraic total-load profile: one
    class-1 fraction, general law 11, total-roughness shear, and no separate
    suspension, hiding, slope, or secondary-flow operator.  The additive v2
    resolver is the admission authority; the older capacity-profile catalogue
    must remain frozen and is not used to relabel the neutral component.
    """
    require(gpu_numerical_variant == VARIANT,
            'schema-v2 execution profiles require the common fixed-grid route')
    profile, checked_documents = resolve_public_law11_total_profile(project)
    _validate_execution_profile_worker_scope(profile, worker_scope)
    if _project_requests_q3h3(project):
        validate_q3h3_worker_source_scope(worker_scope)
    return profile, checked_documents


def materialize_execution_profile_sidecar(
        project, native_root, worker_scope, gpu_numerical_variant):
    """Write the admitted public GPU sidecar before the worker is launched."""
    binding = project.get(EXECUTION_PROFILE_BINDING_KEY)
    if not isinstance(binding, dict) or binding.get('state') != 'canonical':
        return None
    require(gpu_numerical_variant == VARIANT,
            'schema-v2 execution profiles require the common fixed-grid route')
    native_root = Path(native_root)
    resolved_path = native_root/'resolved_inputs.json'
    require(resolved_path.is_file() and not resolved_path.is_symlink(),
            'execution-profile preparation lacks resolved inputs')
    resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
    sediment = resolved.get('sediment') if isinstance(resolved, dict) else None
    require(isinstance(sediment, dict),
            'execution-profile preparation lacks sediment identity')
    neutral_document = sediment.get('execution_profile_v2')
    documents = sediment.get('execution_profile_v2_components')
    require(isinstance(neutral_document, dict) and isinstance(documents, dict),
            'execution-profile preparation is incomplete')
    profile, checked_documents = runtime_execution_profile(
        binding, backend=BACKEND_GPU, component_documents=documents)
    validate_fixed_grid_profile_completeness(
        profile, component_documents=checked_documents,
        prepared_data=_resolved_execution_profile_data(
            project, sediment, profile))
    native_neutral = load_execution_profile(neutral_document)
    if is_frozen_q3sp_execution_profile(native_neutral):
        neutral = native_neutral
    else:
        neutral = resolve_execution_profile(
            fraction_layout=profile.fraction_layout,
            method_selections=profile.method_selections,
            parameter_bindings=profile.parameter_bindings,
            control_bindings=profile.control_bindings,
            # Component ordering is part of the persisted project fingerprint.
            # Runtime resolution canonicalizes the document map for validation;
            # reconstruct the native-neutral comparison in its original order.
            component_profiles=native_neutral.component_profiles, backend=None)
    require(execution_profile_document(neutral) == neutral_document,
            'native and GPU execution-profile physical identities differ')
    _validate_execution_profile_worker_scope(profile, worker_scope)
    if _project_requests_q3h3(project):
        validate_q3h3_worker_source_scope(worker_scope)
    sidecar = dict(schema=1, backend=BACKEND_GPU, admission='runtime',
        profile=execution_profile_document(profile),
        component_documents=dict(checked_documents))
    path = native_root/EXECUTION_PROFILE_ARTIFACT
    serialized = json.dumps(sidecar, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=True, allow_nan=False)
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink() and
                path.read_text(encoding='ascii') == serialized,
                'prepared execution-profile sidecar differs from admitted GPU profile')
    else:
        path.write_text(serialized, encoding='ascii')
    return path


def _bind_execution_profile_components(data, profile, documents):
    fraction_ids = tuple(row.fraction_id for row in profile.fraction_layout)
    require(int(data['nfrac']) == len(fraction_ids),
            'execution-profile fraction layout differs from prepared state')
    bindings = {
        'capacity': ('capacity_runtime_profile', 'capacity_profile_fingerprint'),
        'channel_composition': (
            'channel_composition_profile', 'channel_composition_fingerprint'),
        'suspension': (
            'suspension_runtime_profile', 'suspension_profile_fingerprint'),
    }
    for component_id, (document_key, fingerprint_key) in bindings.items():
        if component_id not in documents:
            continue
        document = documents[component_id]
        require(document_key not in data or data[document_key] == document,
                'prepared component document differs: '+component_id)
        require(fingerprint_key not in data or
                data[fingerprint_key] == document['fingerprint'],
                'prepared component fingerprint differs: '+component_id)
        data[document_key] = document
        data[fingerprint_key] = document['fingerprint']
    if 'capacity' in documents:
        require('capacity_fraction_ids' not in data or
                tuple(data['capacity_fraction_ids']) == fraction_ids,
                'prepared capacity fraction IDs differ from execution profile')
        data['capacity_fraction_ids'] = fraction_ids
    if int(data.get('bedlaw', -1)) in (2, 8, 11):
        q3c = any(row.entry_id in _ROUGHNESS_METHODS
                  for row in profile.method_selections)
        if q3c:
            _, _, law, _, roughness_method = _require_exact_q3c_total_profile(
                profile, documents)
            require(law == int(data.get('bedlaw', -1)),
                    'public Q3-C formula differs from prepared bedlaw')
        else:
            require(int(data.get('bedlaw', -1)) == 11,
                    'historical schema-v2 admission is law-11 only')
            _require_exact_law11_total_profile(profile, documents)
        require(profile.backend == BACKEND_GPU and
                isinstance(data.get('delft_forcing_binding'), dict) and
                data.get('capacity_sediment_shear_method_id') ==
                    'shear.total_roughness' and
                 data.get('capacity_roughness_method_id') in
                     _ROUGHNESS_METHODS and
                type(data.get('capacity_roughness_work_depth_m')) in
                    (float, np.float64) and
                type(data.get('capacity_skin_kinematic_viscosity_m2_s')) in
                    (float, np.float64),
                'public scalar profile lacks authenticated roughness/shear forcing')
        # These are deterministic runtime adapters derived from the v53
        # profile and forcing sidecars.  They are installed after the frozen
        # native NPZ comparison; no unauthenticated physical value is created.
        base_marker = (
            _public_q3c_admission_document(profile, documents) if q3c else
            _public_law11_admission_document(profile, documents))
        derived = {
            'm1_profile_runtime_schema': np.int32(2),
            'capacity_suspended_route':
                'resident_capacity_diagnostic_only_m2_pending',
            'capacity_kinematic_viscosity_m2_s': np.float64(
                data['capacity_skin_kinematic_viscosity_m2_s']),
            _M1_PUBLIC_ADMISSION_KEY: base_marker,
        }
        if q3c:
            marker = base_marker
            forcing = data['delft_forcing_binding']
            execution = capacity_foundation_execution_document(
                method_id=marker['method_id'],
                formula_code=marker['formula_code'],
                forcing_fingerprint=forcing['fingerprint'],
                roughness_method_id=forcing['roughness_method_id'],
                capacity_profile_fingerprint=
                    documents['capacity']['fingerprint'])
            derived['capacity_foundation_execution_identity'] = execution
            derived['capacity_foundation_ordered_route_identity'] = {
                'ordered_parts': ['capacity_execution_fingerprint',
                                  'capacity_profile_fingerprint'],
                'capacity_execution_fingerprint': execution['fingerprint'],
                'capacity_profile_fingerprint':
                    documents['capacity']['fingerprint'],
            }
            hiding = data.get(HIDING_EXECUTION_KEY)
            if hiding is not None:
                identity = validate_hiding_foundation_execution_document(hiding)
                expected_route = {
                    'ordered_parts': [
                        'capacity_execution_fingerprint',
                        'capacity_profile_fingerprint',
                        'hiding_execution_fingerprint'],
                    'capacity_execution_fingerprint': execution['fingerprint'],
                    'capacity_profile_fingerprint':
                        documents['capacity']['fingerprint'],
                    'hiding_execution_fingerprint': identity['fingerprint'],
                }
                require(marker['formula_code'] in (2, 11) and
                        identity['capacity_law'] == marker['formula_code'] and
                        identity['fraction_ids'] == marker['fraction_ids'] and
                        len(marker['fraction_ids']) == 2 and
                        identity['capacity_execution_fingerprint'] ==
                            execution['fingerprint'] and
                        identity['capacity_profile_fingerprint'] ==
                            documents['capacity']['fingerprint'] and
                        data.get(HIDING_ORDERED_ROUTE_KEY) == expected_route and
                        int(data.get('hiding_exposure', -1)) == identity['mode'] and
                        float(data.get('hiding_parker_alpha', np.nan)) ==
                            identity['parker_alpha'] and
                        float(data.get('hiding_wu_exponent', np.nan)) ==
                            identity['wu_exponent'],
                        'Q3-H1 hiding identity differs from its Q3-C foundation')
                derived[_M1_PUBLIC_ADMISSION_KEY] = \
                    _public_q3h1_admission_document(
                        profile, documents, identity)
        require('m1_profile_research_override' not in data,
                'public law-11 profile cannot carry the research override')
        for key, value in derived.items():
            require(key not in data or np.array_equal(data[key], value),
                    'prepared public law-11 runtime field differs: '+key)
            data[key] = value


def execution_profile_sidecar(root, receipt, *, data=None):
    """Authenticate the public runtime sidecar without importing the worker."""
    artifacts = receipt.get('artifacts', {})
    path = root/EXECUTION_PROFILE_ARTIFACT
    declared = EXECUTION_PROFILE_ARTIFACT in artifacts
    require(declared == (path.exists() or path.is_symlink()),
            'execution-profile sidecar is unlisted or missing')
    if not declared:
        return None
    expected_sha256 = artifacts[EXECUTION_PROFILE_ARTIFACT]
    require(path.is_file() and not path.is_symlink() and
            _file_sha256(path) == expected_sha256,
            'execution-profile sidecar differs from preparation receipt')
    raw = _strict_json_object(path)
    require(set(raw) == {
        'schema', 'backend', 'admission', 'profile', 'component_documents'},
        'execution-profile sidecar inventory differs')
    require(type(raw['schema']) is int and
            raw['schema'] == EXECUTION_PROFILE_SIDECAR_SCHEMA and
            raw['backend'] == BACKEND_GPU and raw['admission'] == 'runtime' and
            isinstance(raw['component_documents'], dict),
            'QGIS accepts only an admitted runtime GPU execution profile')
    profile = load_execution_profile(raw['profile'])
    require(profile.backend == BACKEND_GPU and
            execution_profile_document(profile) == raw['profile'],
            'runtime execution profile is not the exact canonical GPU profile')
    expected = {row.component_id: row for row in profile.component_profiles}
    documents = raw['component_documents']
    require(set(documents) == set(expected),
            'execution-profile component document inventory differs')
    for component_id in sorted(documents):
        document = documents[component_id]
        require(isinstance(document, dict) and
                bind_component_profile(component_id, document) == expected[component_id],
                'execution-profile component document differs: '+component_id)
    profile_prepared_data = data
    if data is not None:
        neutral={}
        if HIDING_EXECUTION_KEY in data:
            # The nested Q3-C profile is deliberately unchanged and still owns
            # hiding.none.  Validate that foundation through a neutral view;
            # the independently authenticated outer v55 identity owns the
            # actual native hiding mode and is cross-checked below.
            neutral['hiding_exposure']=np.int32(0)
        if isinstance(receipt.get('prognostic_secondary_identity'),dict):
            # Likewise v57 is a sibling outer composition over a frozen
            # secondary.off v54/v55 profile.  Native v57 already owns the hI
            # slot and reports model 2, so validate only the nested selector
            # through an explicit neutral view.  The authenticated outer
            # identity is installed and checked after this call.
            neutral.update(
                secondary_flow_model=np.int32(0),
                secondary_flow=np.int32(0),
                secondary_flow_transport_direction=np.int32(0),
                secondary_flow_momentum_beta=np.float64(0.))
        if neutral:
            profile_prepared_data=dict(data,**neutral)
    validate_fixed_grid_profile_completeness(
        profile, component_documents=documents,
        prepared_data=profile_prepared_data)
    components = _component_rows(profile)
    identities = {
        'execution_profile_v2_sidecar_sha256': expected_sha256,
        'execution_profile_v2_semantic_id': profile.semantic_id,
        'execution_profile_v2_ledger_semantic_id': profile.ledger_semantic_id,
        'execution_profile_v2_fingerprint': profile.fingerprint,
        'execution_profile_v2_components_sha256': hashlib.sha256(
            _canonical_json(documents).encode('ascii')).hexdigest(),
        'execution_profile_v2_admission': 'runtime',
        'execution_profile_v2_backend': BACKEND_GPU,
        'execution_profile_v2_profile_backend': BACKEND_GPU,
    }
    contract = dict(sidecar_schema=EXECUTION_PROFILE_SIDECAR_SCHEMA,
        identities=identities, components=components,
        component_documents=documents, profile=profile)
    if data is not None:
        _bind_execution_profile_components(data, profile, documents)
        data[_EXECUTION_PROFILE_DATA_KEY] = contract
    return contract


def validate_worker_integrator_capability(scope, token, variant, launch_mode):
    """Require the frozen package to advertise the selected implicit ABI.

    Historical explicit methods predate this versioned package field and keep
    their established gates.  The new method is admitted only by an exact,
    authenticated source-worker declaration; an absent or broader declaration
    is not interpreted as support.
    """
    if token != _IMPLICIT_TOKEN:
        return
    entries = scope.get('implicit_integrators') if isinstance(scope, dict) else None
    catalog = scope.get('sediment_method_catalog') if isinstance(scope, dict) else None
    require(isinstance(catalog, dict) and
            catalog.get('schema') == CATALOG_SCHEMA and
            catalog.get('scope_id') == CATALOG_SCOPE_ID and
            catalog.get('semantic_catalog_id') == semantic_catalog_id(),
            'selected worker uses a different sediment-method catalogue')
    require(isinstance(entries, dict) and
            entries.get(token) == _IMPLICIT_CAPABILITY and
            variant == _IMPLICIT_VARIANT and launch_mode == 'plain',
            'selected worker lacks the exact implicit f-wave midpoint capability')


def validate_implicit_project_scope(project, *, launch_mode, variant):
    """Fail early on project controls outside the runtime's implicit envelope.

    Initial wetness and nondimensional velocity limits depend on the exported
    arrays and are checked separately by :func:`validate_implicit_prepared_scope`.
    This function is intentionally read-only and does not reinterpret a saved
    physical setting to make it admissible.
    """
    require(isinstance(project, dict) and isinstance(project.get('numerics'), dict),
            'implicit midpoint needs explicit project numerics')
    numerics = project['numerics']
    internal = project.get('internal_conditions', {})
    initial = project.get('initial', {})
    require(variant == _IMPLICIT_VARIANT and launch_mode == 'plain',
            'implicit midpoint is limited to the plain fixed-grid common transport route')
    amr = numerics.get('amr', {}).get('enabled', False)
    require(type(amr) is bool and not amr,
            'implicit midpoint does not support AMR')
    bedlaw, exchange = effective_project_transport(project)
    binding = project.get(EXECUTION_PROFILE_BINDING_KEY)
    profile_document = (binding.get('profile', {})
                        if isinstance(binding, dict) else {})
    selections = profile_document.get('method_selections', ()) \
        if isinstance(profile_document, dict) else ()
    q3c_scalar = (bedlaw in _IMPLICIT_SCHEMA_V2_BEDLOAD_LAWS and
                  any(isinstance(row, dict) and
                      row.get('entry_id') in _ROUGHNESS_METHODS
                      for row in selections))
    schema_v2_law11 = bedlaw == 11 and not q3c_scalar
    if schema_v2_law11 or q3c_scalar:
        resolve_public_law11_total_profile(project)
    require(type(bedlaw) is int and
            (bedlaw in _IMPLICIT_BEDLOAD_LAWS or schema_v2_law11 or
             q3c_scalar) and
            type(exchange) is int and exchange == 0,
            'implicit midpoint supports its admitted unsuspended bedload profiles')
    require(type(numerics.get('morphological_factor', 1.)) in (int, float) and
            not isinstance(numerics.get('morphological_factor', 1.), bool) and
            numerics.get('morphological_factor', 1.) == 1.,
            'implicit midpoint requires MorFac 1')
    hiding_mode = numerics.get('hiding_exposure', 0)
    q3h_hiding = (type(hiding_mode) is int and hiding_mode in (2, 3, 4, 5))
    require(hiding_mode == 0 or
            (q3h_hiding and q3c_scalar and bedlaw in (2, 11)),
            'implicit midpoint hiding requires the admitted Q3-H1/Q3-C route')
    require(numerics.get('bed_slope', 0) in _IMPLICIT_BED_SLOPE_MODES,
            'implicit midpoint requires an admitted Delft3D bed-slope option')
    viscosity = numerics.get('horizontal_viscosity_m2_s', 0.)
    require(type(viscosity) in (int, float) and np.isfinite(viscosity) and viscosity >= 0. and
            (viscosity == 0. or numerics.get('horizontal_viscosity_formulation') == 'velocity_laplacian'),
            'implicit midpoint viscosity requires the velocity_laplacian formulation')
    for name in ('bank_failure', 'bank_morphology', 'general_avalanche',
                 'bedform_roughness', 'bedload_adaptation',
                 'morphological_acceleration'):
        value = internal.get(name, {}).get('enabled', False)
        require(type(value) is bool and not value,
                'implicit midpoint excludes '+name)
    secondary = internal.get('secondary_flow', {})
    require(type(secondary.get('enabled', False)) is bool and
            (not secondary.get('enabled', False) or
             secondary.get('model', 'equilibrium') in ('equilibrium', 'prognostic')),
            'implicit midpoint secondary flow must be OFF/equilibrium/prognostic')
    if q3h_hiding:
        declaration = project.get('hiding_foundation_binding')
        require(isinstance(declaration, dict) and
                declaration.get('mode') == hiding_mode and
                declaration.get('state') ==
                    'deferred_prepared_capacity_execution' and
                numerics.get('bed_slope', 0) == 0 and
                not secondary.get('enabled', False),
                'implicit Q3-H1 requires its deferred identity with slope and '
                'secondary flow off')
    boundaries = project.get('boundaries')
    require(isinstance(boundaries, dict) and bool(boundaries),
            'implicit midpoint needs explicit constant boundaries')
    for boundary in boundaries.values():
        require(isinstance(boundary, dict) and
                boundary.get('hydraulic_series') is None and
                boundary.get('sediment_series') is None and
                not boundary.get('fraction_sediment_series'),
                'implicit midpoint requires constant hydraulic and sediment forcing')
        concentrations = boundary.get('fraction_concentration_kg_m3') or {}
        require(not boundary.get('concentration_kg_m3', 0.) and
                not any(concentrations.values()),
                'implicit midpoint excludes suspended concentration forcing')
    fractions = project.get('sediment_fractions')
    sediment_enabled = numerics.get('sediment_enabled', True)
    require(type(sediment_enabled) is bool, 'implicit midpoint sediment switch must be boolean')
    if bedlaw == 0:
        require(not secondary.get('enabled', False) and viscosity == 0. and
                numerics.get('bed_slope', 0) == 0 and
                sediment_enabled is False and not fractions and
                initial.get('active_layer_mode', 'constant') == 'constant' and
                initial.get('active_layer_thickness_m', 0.) == 0.,
                'implicit midpoint bedlaw 0 requires the native inactive static-bed material')
    elif schema_v2_law11:
        require(sediment_enabled is True and not fractions and
                initial.get('active_layer_mode', 'constant') == 'constant' and
                initial.get('active_layer_thickness_m', 0.) == 0.,
                'implicit schema-v2 law-11 requires the canonical single-material layout')
    elif q3c_scalar and not fractions:
        require(sediment_enabled is True and
                initial.get('active_layer_mode', 'constant') == 'constant' and
                initial.get('active_layer_thickness_m', 0.) == 0.,
                'implicit Q3-C single-material route requires its canonical layout')
    else:
        require(sediment_enabled is True and isinstance(fractions, list) and
                1 <= len(fractions) <= 32,
                'implicit midpoint bedload requires 1 to 32 sediment classes')
    if q3h_hiding:
        require(isinstance(fractions, list) and len(fractions) == 2 and
                initial.get('active_layer_mode', 'constant') == 'constant' and
                type(initial.get('active_layer_thickness_m')) in (int, float) and
                not isinstance(initial.get('active_layer_thickness_m'), bool) and
                initial['active_layer_thickness_m'] > 0.,
                'implicit Q3-H1 requires two fractions and a positive active layer')


def _validate_implicit_q(data, q):
    """Validate the static, safely-wet portion of the runtime state envelope.

    The runtime owns the velocity-dependent Froude and advective-CFL checks on
    every *actual* trial and nonlinear stage.  Repeating those checks here with
    ``dtmax`` would incorrectly reject runs whose first trial is clipped or
    transactionally retried with a smaller step.
    """
    nc = int(data['nc'])
    q = np.asarray(q)
    dry = float(data['dry'])
    require(q.shape == (3, nc) and q.dtype == np.float64 and
            np.isfinite(q).all() and np.all(q[0] > 20.*dry),
            'implicit midpoint requires a safely fully wet FP64 state')
    return q


def validate_implicit_prepared_scope(data, *, dtmax=None):
    """Mirror the authenticated runtime's prepared-array admission on CPU."""
    nc, ng, nl = (int(data[key]) for key in ('nc', 'nfrac', 'nbed_layers'))
    active = np.asarray(data.get('active'))
    require(active.shape == (nc,) and np.all(active == 1),
            'implicit midpoint excludes inactive/cut-out cells')
    require(not np.any(data.get('amr_enabled', 0)) and
            int(data['exchange_mode']) == 0 and
            not bool(data.get('suspended_transport_active', False)) and
            not np.any(data.get('suspended_mass', 0)) and
            not np.any(data.get('fraction_concentration', 0)) and
            not np.any(data.get('fraction_series_concentration', 0)) and
            not suspended_enabled(data),
            'implicit midpoint excludes AMR, suspended transport and exchange')
    secondary = int(data['secondary_flow_model'])
    mass = np.asarray(data.get('secondary_mass', np.zeros(nc)))
    require(secondary in _IMPLICIT_SECONDARY_MODELS and
            mass.shape == (nc,) and np.isfinite(mass).all() and
            (secondary == 2 or not np.any(mass)),
            'implicit midpoint secondary inventory differs from its selected model')
    for key in ('bank_failure', 'bank_front_tracking', 'bank_toe_erosion_mode',
                'bank_avalanche', 'bank_erosion_fraction', 'general_avalanche',
                'bedform_roughness', 'bedload_adaptation',
                'morphological_acceleration', 'nmorfac'):
        require(not np.any(data.get(key, 0)),
                'implicit midpoint unsupported prepared control: '+key)
    viscosity = float(data.get('horizontal_viscosity', 0.))
    require(np.isfinite(viscosity) and viscosity >= 0. and
            (viscosity == 0. or (int(data.get('horizontal_viscosity_mode', 0)) == 1 and
                                int(data.get('horizontal_viscosity_geometry', 0)) == 1)) and
            float(data.get('morfac', data.get('morphological_factor', 1.))) == 1.,
            'implicit midpoint requires resident velocity-Laplacian viscosity and MorFac 1')
    require(not np.any(data.get('nhydro_series', 0)) and
            not np.any(data.get('nfraction_series', 0)) and
            not np.any(data.get('nseries', 0)),
            'implicit midpoint requires constant boundary forcing')
    bedlaw = int(data['bedlaw'])
    marker = data.get(_M1_PUBLIC_ADMISSION_KEY)
    q3c_scalar = (isinstance(marker, dict) and
                   marker.get('admission_id') in (
                       _Q3C_PUBLIC_ADMISSION_ID,
                       _Q3H1_PUBLIC_ADMISSION_ID) and
                   bedlaw in _IMPLICIT_SCHEMA_V2_BEDLOAD_LAWS)
    q3h_hiding = (isinstance(marker, dict) and
                   marker.get('admission_id') == _Q3H1_PUBLIC_ADMISSION_ID)
    q3h2_secondary = SECONDARY_EXECUTION_KEY in data
    q3h3_secondary = PROGNOSTIC_SECONDARY_EXECUTION_KEY in data
    require(not (q3h2_secondary and q3h3_secondary),
            'implicit midpoint cannot compose equilibrium and prognostic secondary')
    schema_v2_law11 = bedlaw == 11 and not q3c_scalar
    if q3c_scalar or schema_v2_law11:
        bound = data.get(_EXECUTION_PROFILE_DATA_KEY)
        require(isinstance(bound, dict) and
                bound.get('profile') is not None and
                isinstance(bound.get('component_documents'), dict),
                'implicit scalar-capacity input lacks its admitted schema-v2 profile')
        if q3c_scalar:
            _require_exact_q3c_total_profile(
                bound['profile'], bound['component_documents'])
        else:
            _require_exact_law11_total_profile(
                bound['profile'], bound['component_documents'])
    require(bedlaw in _IMPLICIT_BEDLOAD_LAWS or schema_v2_law11 or q3c_scalar,
            'implicit midpoint bedload profile is not admitted')
    if bedlaw == 0:
        require(secondary == 0 and viscosity == 0. and int(data['bed_slope']) == 0 and
                ng == 1 and nl == 1 and int(data['nsubstrate_layers']) == 0,
                'implicit midpoint bedlaw0 requires the inactive static-bed layout')
    elif schema_v2_law11:
        require(ng == 1 and nl == 1 and
                int(data['nsubstrate_layers']) == 0 and
                int(data['bed_slope']) == 0 and
                int(data['hiding_exposure']) == 0,
                'implicit schema-v2 law-11 requires its one-layer single-material layout')
    elif q3h_hiding:
        identity = validate_hiding_foundation_execution_document(
            data.get(HIDING_EXECUTION_KEY))
        require(ng == 2 and nl >= 2 and
                int(data['nsubstrate_layers']) == nl - 1 and
                int(data['bed_slope']) == 0 and
                secondary == (2 if q3h3_secondary else
                              1 if q3h2_secondary else 0) and
                int(data['hiding_exposure']) == identity['mode'] and
                bedlaw == identity['capacity_law'] and
                tuple(data.get('capacity_fraction_ids', ())) ==
                    tuple(identity['fraction_ids']),
                'implicit Q3-H1 requires its two-fraction active-surface route')
    elif q3c_scalar:
        require(1 <= ng <= 32 and nl >= 1 and
                int(data['bed_slope']) == 0 and
                int(data['hiding_exposure']) == 0 and
                secondary == (2 if q3h3_secondary else
                              1 if q3h2_secondary else 0),
                'implicit Q3-C requires 1 to 32 class-1 fractions with OFF auxiliaries')
    else:
        require(1 <= ng <= 32 and nl >= 2 and
                int(data['bed_slope']) in _IMPLICIT_BED_SLOPE_MODES and
                int(data['hiding_exposure']) == 0,
                'implicit midpoint bedload requires 1 to 32 classes with hiding OFF')
    if q3h2_secondary:
        identity=validate_secondary_foundation_execution_document(
            data.get(SECONDARY_EXECUTION_KEY))
        route=data.get(SECONDARY_ORDERED_ROUTE_KEY)
        require(q3c_scalar and identity['capacity_law']==bedlaw and
                identity['model']==1 and
                identity['method_id']=='secondary.equilibrium' and
                identity['base_envelope_version']==(55 if q3h_hiding else 54) and
                identity['hiding_execution_fingerprint']==(
                    data.get(HIDING_EXECUTION_KEY,{}).get('fingerprint')
                    if q3h_hiding else None) and
                route=={
                    'ordered_parts':['capacity_execution_fingerprint',
                                     'capacity_profile_fingerprint',
                                     'hiding_execution_fingerprint',
                                     'secondary_execution_fingerprint'],
                    'capacity_execution_fingerprint':identity[
                        'capacity_execution_fingerprint'],
                    'capacity_profile_fingerprint':identity[
                        'capacity_profile_fingerprint'],
                    'hiding_execution_fingerprint':identity[
                        'hiding_execution_fingerprint'],
                    'secondary_execution_fingerprint':identity['fingerprint']} and
                identity['coriolis_disposition']=='zero' and
                identity['coriolis_constant_s_inverse']==0. and
                identity['betac']==0. and secondary==0 and
                int(data['bed_slope'])==0 and
                not np.any(data.get('amr_enabled',0)),
                'implicit Q3-H2 requires its authenticated fixed-grid classic secondary route')
    if q3h3_secondary:
        identity=validate_public_q3h3_controls(data,secondary)
        require(q3c_scalar and identity['capacity_law']==bedlaw and
                identity['model']==2 and
                identity['method_id']=='secondary.prognostic' and
                identity['gpu_checkpoint_schema']==8 and
                int(data['bed_slope'])==0 and viscosity==0. and
                not np.any(data.get('amr_enabled',0)),
                'implicit Q3-H3 requires its authenticated fixed-grid classic '
                'prognostic-secondary route')
    for key in ('bed', 'area', 'length', 'normal', 'owner', 'neighbour',
                'distance', 'manning_cell'):
        require(np.isfinite(np.asarray(data[key])).all(),
                'nonfinite implicit midpoint input: '+key)
    require(np.all(np.asarray(data['area']) > 0.) and
            np.all(np.asarray(data['length']) > 0.) and
            np.all(np.asarray(data['manning_cell']) >= 0.),
            'implicit midpoint geometry/friction is invalid')
    internal = np.asarray(data['neighbour']) >= 0
    distance = np.asarray(data['distance'])[0] + np.asarray(data['distance'])[1]
    require(np.any(internal) and np.all(distance[internal] > 0.),
            'implicit midpoint needs positive internal face distances')
    q = np.asarray(data.get('q', np.asarray(data['state'])[:3]))
    # Kept as a keyword for a stable host call site.  Trial-size admission is
    # deliberately runtime-owned; only reject malformed explicit overrides.
    if dtmax is not None:
        require(type(dtmax) in (int, float) and not isinstance(dtmax, bool) and
                np.isfinite(dtmax) and dtmax > 0.,
                'implicit midpoint requires a finite positive timestep limit')
    return _validate_implicit_q(data, q)


def _q3c_profile_identity(data):
    """Derive the authenticated Q3-C/Q3-H1/Q3-H2/Q3-H3 identity."""
    marker = data.get(_M1_PUBLIC_ADMISSION_KEY)
    hiding = (isinstance(marker, dict) and
              marker.get('admission_id') == _Q3H1_PUBLIC_ADMISSION_ID)
    prognostic = PROGNOSTIC_SECONDARY_EXECUTION_KEY in data
    require(isinstance(marker, dict) and
            marker.get('admission_id') in (
                _Q3C_PUBLIC_ADMISSION_ID, _Q3H1_PUBLIC_ADMISSION_ID),
            'Q3-C/Q3-H1 profile identity lacks its public admission')
    profile = load_capacity_profile(data.get('capacity_runtime_profile'))
    channel = load_channel_composition_profile(
        data.get('channel_composition_profile'))
    law = int(data['bedlaw'])
    require(law in _Q3C_METHODS and marker.get('formula_code') == law and
            marker.get('method_id') == _Q3C_METHODS[law] and
            data.get('capacity_profile_fingerprint') == profile.fingerprint and
            data.get('channel_composition_fingerprint') == channel.fingerprint and
            data.get('capacity_suspended_route') ==
                'resident_capacity_diagnostic_only_m2_pending' and
            int(data.get('exchange_mode', -1)) == 0,
            'Q3-C profile identity differs from authenticated components')
    forcing = forcing_identity(data)
    require(forcing['hydraulic_roughness_method_id'] ==
                data.get('capacity_roughness_method_id') and
            forcing['sediment_shear_method_id'] == 'shear.total_roughness',
            'Q3-C forcing/profile identities differ')
    fraction_ids = tuple(data.get('capacity_fraction_ids', ()))
    require(tuple(row.fraction_id for row in profile.fractions) == fraction_ids and
            tuple(row.fraction_id for row in channel.fractions) == fraction_ids and
            all(row.formula_code == law and
                row.method_id == _Q3C_METHODS[law]
                for row in profile.fractions),
            'Q3-C fraction/profile order differs')
    result = dict(
        m1_profile_runtime_schema=2,
        m1_profile_generation=(
            'implicit_q3h3_classic_flow_prognostic_secondary_profile_v1'
            if prognostic else
            'implicit_q3h1_hiding_capacity_profile_v1' if hiding else
            'implicit_q3c_scalar_capacity_profile_v1'),
        capacity_profile_fingerprint=profile.fingerprint,
        capacity_profile_semantic_id=profile.semantic_id,
        capacity_parameter_catalog_id=profile.parameter_catalog_id,
        capacity_formula=_Q3C_METHODS[law],
        capacity_formula_code=law,
        capacity_bed_route='resident_reconstructed_face_exner_v1',
        capacity_suspended_route=
            'resident_capacity_diagnostic_only_m2_pending',
        channel_composition_schema=channel.schema,
        channel_composition_scope_id=channel.scope_id,
        channel_composition_semantic_id=channel.semantic_id,
        channel_composition_fingerprint=channel.fingerprint,
        channel_composition_bed_factor=channel.bed_factor,
        channel_composition_suspended_factor=channel.suspended_factor,
        channel_composition_epsilon=channel.epsilon,
        channel_transport_classes=[row.transport_class
                                   for row in channel.fractions],
        m1_profile_admission=marker['admission_id'],
        automatic_CPU_fallback=False,
    )
    if hiding:
        identity = validate_hiding_foundation_execution_document(
            data.get(HIDING_EXECUTION_KEY))
        require(marker == _public_q3h1_admission_document(
                    data[_EXECUTION_PROFILE_DATA_KEY]['profile'],
                    data[_EXECUTION_PROFILE_DATA_KEY]['component_documents'],
                    identity),
                'Q3-H1 public admission differs from its execution identity')
        result.update(
            hiding_execution_schema=identity['schema'],
            hiding_execution_scope_id=identity['scope_id'],
            hiding_execution_semantic_id=identity['semantic_id'],
            hiding_execution_fingerprint=identity['fingerprint'],
            hiding_exposure_mode=identity['mode'],
            hiding_method_id=identity['method_id'],
            hiding_parker_alpha=identity['parker_alpha'],
            hiding_wu_exponent=identity['wu_exponent'],
            hiding_active_volume_source=identity['active_volume_source'],
            hiding_composition_order=identity['composition_order'],
        )
    if prognostic:
        identity=validate_prognostic_secondary_execution_document(
            data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY))
        foundation=load_capacity_foundation_execution(
            data.get('capacity_foundation_execution_identity'))
        expected_route={
            'capacity_execution_fingerprint':foundation['fingerprint'],
            'capacity_profile_fingerprint':profile.fingerprint,
            'hiding_execution_fingerprint':identity[
                'hiding_execution_fingerprint'],
            'prognostic_secondary_execution_fingerprint':identity['fingerprint'],
        }
        require(SECONDARY_EXECUTION_KEY not in data and
                data.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)==expected_route and
                identity['capacity_execution_fingerprint']==foundation['fingerprint'] and
                identity['capacity_law']==law and
                identity['capacity_profile_fingerprint']==profile.fingerprint and
                identity['roughness_method_id']==
                    forcing['hydraulic_roughness_method_id'] and
                identity['roughness_fingerprint']==
                    forcing['delft_forcing_fingerprint'] and
                identity['hiding_execution_fingerprint']==(
                    result.get('hiding_execution_fingerprint')),
                'Q3-H3 identity differs from its Q3-C/Q3-H1 foundation')
        result.update(
            prognostic_secondary_execution_schema=identity['schema'],
            prognostic_secondary_execution_scope_id=identity['scope_id'],
            prognostic_secondary_execution_semantic_id=identity['semantic_id'],
            prognostic_secondary_execution_fingerprint=identity['fingerprint'],
            prognostic_secondary_method_id=identity['method_id'],
            prognostic_secondary_model=identity['model'],
            prognostic_secondary_state_semantics=identity['state_semantics'],
            prognostic_secondary_integrated_ledger_units=
                identity['integrated_ledger_units'],
            prognostic_secondary_base_envelope_version=
                identity['base_envelope_version'],
            prognostic_secondary_capacity_law=identity['capacity_law'],
            prognostic_secondary_rmincf=identity['rmincf'],
            prognostic_secondary_chzmin=identity['chzmin'],
            prognostic_secondary_espir=identity['espir'],
            prognostic_secondary_betac=identity['betac'],
            prognostic_secondary_scalar_diffusivity_m2_s=
                identity['scalar_diffusivity_m2_s'],
            prognostic_secondary_source_sink_policy=
                identity['source_sink_policy'],
            prognostic_secondary_open_boundary_policy=
                identity['open_boundary_policy'],
            prognostic_secondary_bed_boundary_rotation_policy=
                identity['bed_boundary_rotation_policy'],
            prognostic_secondary_depth_pair_policy=
                identity['depth_pair_policy'],
            prognostic_secondary_target_forcing_policy=
                identity['target_forcing_policy'],
            prognostic_secondary_composition_order=identity['composition_order'],
            prognostic_secondary_checkpoint_schema=
                identity['gpu_checkpoint_schema'],
            prognostic_secondary_backend_selection=identity['backend_selection'],
            automatic_CPU_fallback=False,
        )
    elif SECONDARY_EXECUTION_KEY in data:
        identity=validate_secondary_foundation_execution_document(
            data.get(SECONDARY_EXECUTION_KEY))
        expected_route={
            'ordered_parts':['capacity_execution_fingerprint',
                             'capacity_profile_fingerprint',
                             'hiding_execution_fingerprint',
                             'secondary_execution_fingerprint'],
            'capacity_execution_fingerprint':identity[
                'capacity_execution_fingerprint'],
            'capacity_profile_fingerprint':identity[
                'capacity_profile_fingerprint'],
            'hiding_execution_fingerprint':identity[
                'hiding_execution_fingerprint'],
            'secondary_execution_fingerprint':identity['fingerprint'],
        }
        require(data.get(SECONDARY_ORDERED_ROUTE_KEY)==expected_route and
                identity['capacity_law']==law and
                identity['capacity_profile_fingerprint']==profile.fingerprint and
                identity['roughness_method_id']==
                    forcing['hydraulic_roughness_method_id'] and
                identity['roughness_fingerprint']==
                    forcing['delft_forcing_fingerprint'] and
                identity['hiding_execution_fingerprint']==(
                    result.get('hiding_execution_fingerprint')),
                'Q3-H2 identity differs from its Q3-C/Q3-H1 foundation')
        result.update(
            secondary_execution_schema=identity['schema'],
            secondary_execution_scope_id=identity['scope_id'],
            secondary_execution_semantic_id=identity['semantic_id'],
            secondary_execution_fingerprint=identity['fingerprint'],
            secondary_method_id=identity['method_id'],
            secondary_model=identity['model'],
            secondary_base_envelope_version=identity['base_envelope_version'],
            secondary_capacity_law=identity['capacity_law'],
            secondary_rmincf=identity['rmincf'],
            secondary_chzmin=identity['chzmin'],
            secondary_espir=identity['espir'],
            secondary_betac=identity['betac'],
            secondary_curvature_discretization=
                identity['curvature_discretization'],
            secondary_curvature_cap_policy=
                identity['curvature_cap_policy'],
            secondary_metric_policy=identity['metric_policy'],
            secondary_roughness_policy=identity['roughness_policy'],
            secondary_refresh_timing=identity['refresh_timing'],
            secondary_boundary_policy=identity['boundary_policy'],
            secondary_composition_order=identity['composition_order'],
            secondary_raw_bed_rotation_scope=
                identity['raw_bed_rotation_scope'],
            suspended_channel_rotation=identity['suspended_channel_rotation'],
            secondary_backend_selection=identity['backend_selection'],
            automatic_CPU_fallback=False,
        )
    return result


def _validate_implicit_method(method, manifest, data):
    """Bind run/readback to the exact qualified method and checkpoint ABI."""
    implicit = method.get('integrator') == _IMPLICIT_LABEL
    if not implicit:
        require(method.get('implicit_midpoint_schema') is None and
                method.get('predictor_checkpoint') is None and
                manifest.get('implicit_fwave_midpoint') is not True and
                manifest.get('implicit_predictor_checkpoint') is not True,
                'implicit method metadata is present for another integrator')
        return False
    selected_schema = implicit_checkpoint_schema(data)
    coupling = implicit_coupling_identity(data)
    expected = dict(_IMPLICIT_METHOD,
        integrator=_IMPLICIT_LABEL, euler_stages=1, launch_mode='plain',
        stage_slots=[1], stage_fractions=[.5],
        transport_checkpoint_schema=selected_schema,
        predictor_checkpoint='implicit_predictor_rate_and_valid_flag_v1',
        coupling=coupling)
    if BINDING_FIELD in data:
        # Typed C/M/W routes replace the historical Manning-only split token
        # with the exact forcing-bound friction identity emitted by the
        # worker.  Keeping both tokens would falsely claim two simultaneous
        # friction implementations.
        expected.pop('manning_split')
        forcing = forcing_identity(data)
        expected.update(
            hydraulic_friction_split=(
                'symmetric_half_step_legacy_manning_v1'
                if forcing['hydraulic_roughness_method_id'] ==
                    'roughness.manning'
                else 'symmetric_half_step_typed_resident_v1'),
            **forcing)
    if expected['coupling'] == 'weak_midpoint_delft_components_v2':
        expected.update(_IMPLICIT_COMPONENT_METHOD)
    elif coupling == \
            'weak_midpoint_q3h3_classic_flow_prognostic_secondary_composition_v1':
        identity=validate_prognostic_secondary_execution_document(
            data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY))
        expected.update(_Q3H3_COMPONENT_METHOD)
        expected.update(
            secondary_state_semantics=identity['state_semantics'],
            secondary_integrated_ledger_units=identity[
                'integrated_ledger_units'],
            secondary_boundary_rotation=identity[
                'bed_boundary_rotation_policy'])
        if HIDING_EXECUTION_KEY in data:
            expected.update({
                key:value for key,value in _Q3H1_COMPONENT_METHOD.items()
                if key.startswith('hiding_')})
        expected.update(_q3c_profile_identity(data))
    elif coupling in ('weak_midpoint_q3c_scalar_capacity_composition_v1',
                      'weak_midpoint_q3h1_hiding_capacity_composition_v1',
                      'weak_midpoint_q3h2_classic_flow_equilibrium_secondary_composition_v1'):
        expected.update(
            _Q3H2_COMPONENT_METHOD if coupling.endswith(
                'q3h2_classic_flow_equilibrium_secondary_composition_v1') else
            _Q3H1_COMPONENT_METHOD if coupling.endswith(
                'q3h1_hiding_capacity_composition_v1') else
            _Q3C_COMPONENT_METHOD)
        if (coupling.endswith(
                'q3h2_classic_flow_equilibrium_secondary_composition_v1') and
                HIDING_EXECUTION_KEY in data):
            expected.update({
                key:value for key,value in _Q3H1_COMPONENT_METHOD.items()
                if key.startswith('hiding_')})
        expected.update(_q3c_profile_identity(data))
    require(all(method.get(key) == value for key, value in expected.items()),
            'implicit midpoint method identity differs from the qualified contract')
    require(manifest.get('integrator') == _IMPLICIT_LABEL and
            manifest.get('euler_stages') == 1 and
            manifest.get('transport_checkpoint_schema') == selected_schema and
            manifest.get('implicit_fwave_midpoint') is True and
            manifest.get('implicit_predictor_checkpoint') is True and
            manifest.get('weakly_coupled_sediment') is
                (int(data['bedlaw']) in (2, 8, 11)),
            'implicit midpoint manifest/checkpoint identity differs')
    return True


def validate_integrator_scope(*, bedlaw, exchange_mode, integrator, launch_mode,
                              bedform_roughness=False, bedload_adaptation=False,
                              general_avalanche=False,
                              schema_v2_law11=False):
    """Validate the explicitly selected method for the common fixed-grid route.

    ``cpu_time_integrator`` belongs to the zero-step native exporter and is
    deliberately not consulted here.  The GPU method is the explicit run
    choice recorded in the method identity.
    """
    require(launch_mode == 'plain',
            'the common fixed-grid route requires explicit plain launches')
    if integrator == 'SSPRK33':
        return
    optional = (bedform_roughness, bedload_adaptation, general_avalanche)
    if integrator == _IMPLICIT_LABEL:
        require(type(schema_v2_law11) is bool and
                type(bedlaw) is int and
                (bedlaw in _IMPLICIT_BEDLOAD_LAWS or
                 (bedlaw == 11 and schema_v2_law11)) and
                type(exchange_mode) is int and exchange_mode == 0 and
                all(type(value) in (bool, int) and not value for value in optional),
                'implicit midpoint is limited to admitted bedload profiles with '
                'suspension and optional morphology transactions off')
        return
    require(integrator == 'SSPRK43' and type(bedlaw) is int and bedlaw == 2 and
            type(exchange_mode) is int and exchange_mode == 0 and
            all(type(value) in (bool, int) and not value for value in optional),
            'SSPRK43 is limited to fixed-grid nonbank MPM2 with suspended exchange, '
            'bedform roughness, bedload adaptation and general avalanche off')


def _validate_sampled_discharge(data):
    """Host mirror of production mode1 geometry, without numerical imports."""
    nc, nf, nb = (int(data[k]) for k in ('nc','nf','nb'))
    modes = np.asarray(data['mode'])
    flags, delta = np.asarray(data['sampled']), np.asarray(data['boundary_bed_offset'])
    owner, neighbour, bid, offsets, faces = (np.asarray(data[k]) for k in
        ('owner','neighbour','bid','boundary_offset','boundary_faces'))
    require(flags.shape == (nf,) and flags.dtype == np.int32 and np.isin(flags,(0,1)).all() and
            delta.shape == (nf,) and delta.dtype == np.float64 and np.isfinite(delta).all() and
            not np.any(delta[flags == 0]), 'invalid sampled mode1 flags/offsets')
    require(owner.shape == neighbour.shape == bid.shape == (nf,) and
            all(v.dtype == np.int32 for v in (owner,neighbour,bid,offsets,faces)) and
            np.all((owner >= 0)&(owner < nc)) and np.all((bid >= -1)&(bid < nb)),
            'invalid sampled mode1 topology')
    require(offsets.shape == (nb+1,) and offsets[0] == 0 and offsets[-1] == len(faces) and
            np.all(np.diff(offsets) >= 0) and faces.ndim == 1 and
            np.all((faces >= 0)&(faces < nf)) and len(np.unique(faces)) == len(faces) and
            np.array_equal(np.sort(faces),np.flatnonzero(bid >= 0)), 'invalid sampled boundary CSR')
    fs = np.flatnonzero(flags)
    require(len(fs) > 0 and np.all(bid[fs] >= 0) and np.all(modes[bid[fs]] == 1) and
            np.all(neighbour[fs] == -1) and data.get('production_section_geometry') == 1,
            'only whole production mode1 sections may be sampled')
    selected = np.zeros(nb,dtype=bool); selected[np.unique(bid[fs])] = True
    for b in range(nb):
        members = faces[offsets[b]:offsets[b+1]]
        require(np.all(bid[members] == b) and np.all(neighbour[members] == -1) and
                (not selected[b] or len(members)>0 and np.all(flags[members] == 1)),
                'partly sampled or inconsistent legacy section')
    for key in ('native_section_seed_node','native_boundary_seed_position'):
        raw = np.asarray(data.get(key))
        require(raw.shape == (nb,) and raw.dtype == np.int32 and np.all(raw >= 0),
                'invalid native sampled section seed')
        count = np.diff(offsets)
        limit = count+(key == 'native_section_seed_node')
        require(np.all(raw[~selected] == 0) and np.all(raw[selected] <= limit[selected]),
                'sampled section seed is outside or unrelated to its section')
    normal, length = np.asarray(data['normal']), np.asarray(data['length'])
    require(normal.shape == (2,nf) and normal.dtype == np.float64 and np.isfinite(normal).all() and
            np.allclose(np.sum(normal*normal,axis=0),1.,rtol=0.,atol=1.e-12) and
            length.shape == (nf,) and length.dtype == np.float64 and np.isfinite(length).all() and
            np.all(length > 0.), 'invalid sampled face geometry')
    forcing = np.asarray(data['forcing'])
    require(forcing.ndim == 2 and forcing.shape[0] >= 2 and forcing.shape[1] == nb and
            np.isfinite(forcing[:2]).all() and np.all(forcing[0,selected] >= 0.),
            'invalid sampled Q/reference stage')


def validate_boundary_scope(data, *, allow_q3h3_closed_wall=False):
    """Host mirror of the ordinary boundary family; not an AMR policy."""
    nb = int(data['nb'])
    control, modes = np.asarray(data.get('section_control')), np.asarray(data['mode'])
    if nb == 0:
        require(allow_q3h3_closed_wall and
                modes.shape == control.shape == (0,) and
                modes.dtype == control.dtype == np.int32 and
                np.asarray(data['boundary_offset']).shape == (1,) and
                np.asarray(data['boundary_offset']).dtype == np.int32 and
                int(np.asarray(data['boundary_offset'])[0]) == 0 and
                np.asarray(data['boundary_faces']).shape == (0,) and
                np.asarray(data['boundary_faces']).dtype == np.int32 and
                np.all(np.asarray(data['bid']) == -1) and
                all(np.asarray(data[key]).shape == (0,) for key in
                    ('section_seed_node', 'boundary_seed_position')),
                'closed-wall boundary inventory requires authenticated Q3-H3')
        return False
    require(nb >= 1 and modes.shape == (nb,) and modes.dtype == np.int32 and
            control.dtype == np.int32 and control.shape == (nb,), 'invalid boundary controls')
    surveyed = bool(np.any(modes == 9))
    if surveyed:
        require(modes.shape == (2,) and modes[0] == 9 and modes[1] in (2,4) and
                control[0] in (0,1,2) and control[1] == 0 and
                data.get('production_section_geometry') == 1,
                'invalid surveyed inlet/stage or critical outlet')
    else:
        require(np.isin(modes,(1,2,4,6,8)).all() and not np.any(control),
                'unsupported stock boundary controls')
        if np.any(data.get('sampled',0)):
            _validate_sampled_discharge(data)
        else:
            if np.any(modes == 1):
                require(not np.any(data['boundary_bed_offset']), 'unsampled mode1 requires zero-offset faces')
            for key in ('native_section_seed_node','native_boundary_seed_position'):
                require(np.shape(data.get(key)) == (nb,) and not np.any(data[key]),
                        'stock boundary metadata cannot carry unrelated surveyed seeds')
    require(not np.any(data['nseries']), 'unsupported legacy sediment forcing')
    outlets = np.isin(modes,(2,4))
    require(not np.any(np.asarray(data['nfraction_series'])[:, outlets]) and
            not np.any(np.asarray(data['fraction_concentration'])[:, outlets]) and
            not np.any(np.asarray(data['fraction_bed_input'])[:, outlets]), 'unsupported outlet sediment forcing')
    require(not np.any(np.asarray(data['nhydro_series'])[modes == 4]),
            'unsupported critical outlet hydraulic forcing')
    return surveyed


def numerical_identity(receipt):
    artifacts = receipt.get('artifacts')
    require(isinstance(artifacts, dict) and ARTIFACTS <= set(artifacts) <=
            ARTIFACTS | OPTIONAL_ARTIFACTS, 'artifact inventory differs')
    return hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()


def validate_receipt(root, receipt):
    from .gpu_prepared_io import _authenticated_member
    require(type(receipt.get('schema')) is int and receipt['schema'] in (6, 7) and
        receipt.get('variant') == VARIANT and ARTIFACTS <= set(receipt.get('artifacts', {})) <=
        ARTIFACTS | OPTIONAL_ARTIFACTS and
        receipt.get('exporter_executable_sha256') in
        (EXPORTER_SHA256, AVALANCHE_EXPORTER_SHA256, EH_EXPORTER_SHA256,
         *DELFT_EXPORTER_GENERATIONS,
         *GAIA_EXPORTER_GENERATIONS) and
        receipt.get('initialization_CPU_only') is True and receipt.get('GPU_solver_executed') is False and
        type(receipt.get('native_solver_steps_performed')) is int and
        receipt['native_solver_steps_performed'] == 0,
        'not an authenticated zero-step schema-6/7 export')
    exporter = receipt.get('exporter_executable_sha256')
    require((exporter == DELFT_EXPORTER_V17_SHA256) ==
            isinstance(receipt.get('secondary_foundation_identity'), dict),
            'v56 preparation receipt presence differs')
    prognostic_receipt_keys=(
        'prognostic_secondary_execution_fingerprint',
        'prognostic_secondary_state_semantics',
        'prognostic_secondary_integrated_ledger_units',
        'prognostic_secondary_bed_boundary_rotation_policy',
        'prognostic_secondary_state_units')
    require((exporter == DELFT_EXPORTER_V18_SHA256) ==
            isinstance(receipt.get('prognostic_secondary_identity'), dict) and
            all((key in receipt)==(exporter==DELFT_EXPORTER_V18_SHA256)
                for key in prognostic_receipt_keys),
            'v57 preparation receipt presence differs')
    if exporter == DELFT_EXPORTER_V18_SHA256:
        outer=receipt['prognostic_secondary_identity']
        identity=validate_prognostic_secondary_execution_document(
            outer.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)
            if isinstance(outer,dict) else None)
        require(receipt['prognostic_secondary_execution_fingerprint']==
                    identity['fingerprint'] and
                receipt['prognostic_secondary_state_semantics']==
                    identity['state_semantics'] and
                receipt['prognostic_secondary_integrated_ledger_units']==
                    identity['integrated_ledger_units']=='m4_s' and
                receipt['prognostic_secondary_bed_boundary_rotation_policy']==
                    identity['bed_boundary_rotation_policy'] and
                receipt['prognostic_secondary_state_units']=='m2_s',
                'v57 preparation visible routing receipt differs')
    if exporter in DELFT_EXPORTER_GENERATIONS:
        require(receipt.get('initializer_generation') ==
                DELFT_EXPORTER_GENERATIONS[exporter],
                'Delft component initializer generation/hash pair differs')
        if exporter in _DELFT_FORCING_EXPORTERS:
            require(receipt.get('schema') == 7 and
                    FORCING_ARTIFACT in receipt['artifacts'],
                    'Delft component initializer lacks its forcing binding')
        else:
            require(receipt.get('schema') == 6 and
                    FORCING_ARTIFACT not in receipt['artifacts'],
                    'historical Delft initializer acquired a forcing binding')
    else:
        require(FORCING_ARTIFACT not in receipt['artifacts'],
                'historical initializer cannot acquire a forcing binding')
    if exporter in GAIA_EXPORTER_GENERATIONS:
        require(receipt.get('initializer_generation') == GAIA_EXPORTER_GENERATIONS[exporter],
                'GAIA initializer generation/hash pair differs')
    require(AVALANCHE_ARTIFACT not in receipt['artifacts'] or
            receipt['exporter_executable_sha256'] in
            (AVALANCHE_EXPORTER_SHA256, EH_EXPORTER_SHA256,
             *DELFT_EXPORTER_GENERATIONS),
            'avalanche sidecar requires its pinned initializer')
    require(not (root/'preparation_receipt.json').is_symlink(), 'linked receipt')
    files = receipt.get('origin', {}).get('files')
    require(isinstance(files, dict) and 'case.data' in files, 'missing input provenance')
    for name, digest in files.items(): _authenticated_member(root, name, digest, nested=True)
    if 'geometry.npz' not in files:
        # QGIS's small v1 deck stores cell/face geometry inline. Its exact deck
        # and metadata must both be authenticated; never invent a mesh sidecar.
        require('case.data.json' in files, 'inline geometry metadata is unauthenticated')
        deck_lines = (root/'case.data').read_text(
            encoding='utf-8').splitlines()
        header = deck_lines[:2]
        require(len(header) == 2 and
                re.fullmatch(r'[0-9a-fA-F]{64}',header[1].strip()) is not None,
                'unsupported inline geometry header')
        metadata = json.loads((root/'case.data.json').read_text(encoding='utf-8'))
        legacy_inline = (header[0].strip() == '1' and
                         metadata.get('format') == 'xbraids_cutcell-v1')
        gaia = metadata.get('gaia_engelund_chollet_cunge')
        slope = metadata.get('bed_slope')
        secondary = metadata.get('secondary_flow')
        gaia_inline = (
            header[0].strip() == '43' and
            exporter in GAIA_EXPORTER_GENERATIONS and
            metadata.get('format') == 'xbraids_cutcell-v43' and
            metadata.get('bedload_law') == 9 and isinstance(gaia, dict) and
            gaia.get('skin_roughness_ratio') == 3. and
            gaia.get('slope_magnitude_beta') == 1.3 and
            gaia.get('talmon_deviation_beta2') == .85 and
            gaia.get('secondary_alpha') == 1. and
            isinstance(slope, dict) and slope.get('islope') == 4 and
            isinstance(secondary, dict) and secondary.get('model') == 'gaia')
        execution_profile = metadata.get('execution_profile_v2')
        q3c_law = metadata.get('bedload_law')
        profile_document = (execution_profile.get('profile')
                            if isinstance(execution_profile, dict) else None)
        method_rows = (profile_document.get('method_selections')
                       if isinstance(profile_document, dict) else None)
        profile_roughness = ({
            row.get('entry_id') for row in method_rows
            if isinstance(row, dict) and
            row.get('entry_id') in _ROUGHNESS_METHODS
        } if isinstance(method_rows, list) else None)
        generation_profile = (
            (exporter == DELFT_EXPORTER_V13_SHA256 and
             profile_roughness == set()) or
            (exporter == DELFT_EXPORTER_V14_SHA256 and
             isinstance(profile_roughness, set) and
             len(profile_roughness) == 1) or
            (exporter == DELFT_EXPORTER_V15_SHA256 and
             isinstance(profile_roughness, set) and
             len(profile_roughness) == 1) or
            (exporter == DELFT_EXPORTER_V17_SHA256 and
             isinstance(profile_roughness, set) and
             len(profile_roughness) == 1) or
            (exporter == DELFT_EXPORTER_V18_SHA256 and
             isinstance(profile_roughness, set) and
             len(profile_roughness) == 1))
        capacity_route = metadata.get('capacity_route_identity')
        historical_v53 = (
            header[0].strip() == '53' and
            exporter == DELFT_EXPORTER_V13_SHA256 and q3c_law == 11 and
            metadata.get('format') == 'xbraids_cutcell-v53')
        q3c_v54 = (
            header[0].strip() == '54' and
            exporter == DELFT_EXPORTER_V14_SHA256 and
            q3c_law in (2, 8, 11) and
            metadata.get('format') == 'xbraids_cutcell-v54' and
            isinstance(capacity_route, dict) and
            capacity_route.get('envelope_version') == 54)
        q3h1_v55 = (
            header[0].strip() == '55' and
            exporter == DELFT_EXPORTER_V15_SHA256 and
            q3c_law in (2, 11) and
            metadata.get('format') == 'xbraids_cutcell-v55' and
            isinstance(capacity_route, dict) and
            capacity_route.get('envelope_version') == 54)
        q3h2_v56 = (
            header[0].strip() == '56' and
            exporter == DELFT_EXPORTER_V17_SHA256 and
            q3c_law in (2, 8, 11) and
            metadata.get('format') == 'xbraids_cutcell-v56' and
            isinstance(capacity_route, dict) and
            capacity_route.get('envelope_version') == 54)
        q3h3_v57 = (
            header[0].strip() == '57' and
            exporter == DELFT_EXPORTER_V18_SHA256 and
            q3c_law in (2, 8, 11) and
            metadata.get('format') == 'xbraids_cutcell-v57' and
            isinstance(capacity_route, dict) and
            capacity_route.get('envelope_version') == 54)
        if q3h1_v55:
            _v55_hiding_identity(metadata, deck_lines)
            require(metadata.get('secondary_foundation_identity') is None,
                    'v55 preparation cannot acquire a v56 secondary identity')
        elif q3h2_v56:
            _v56_secondary_identity(metadata, deck_lines)
            require(receipt.get('secondary_foundation_identity') ==
                    metadata.get('secondary_foundation_identity'),
                    'v56 preparation receipt differs from case metadata')
        elif q3h3_v57:
            _v57_prognostic_secondary_identity(metadata,deck_lines)
            require(receipt.get('prognostic_secondary_identity')==
                    metadata.get('prognostic_secondary_identity') and
                    metadata.get('secondary_foundation_identity') is None,
                    'v57 preparation receipt differs from case metadata')
        else:
            require(metadata.get('hiding_foundation_identity') is None,
                    'v53/v54 preparation cannot acquire a v55 hiding identity')
            require(metadata.get('secondary_foundation_identity') is None,
                    'v53/v54 preparation cannot acquire a v56 secondary identity')
            require(metadata.get('prognostic_secondary_identity') is None,
                    'pre-v57 preparation cannot acquire a prognostic secondary identity')
        public_v53_inline = (
            (historical_v53 or q3c_v54 or q3h1_v55 or q3h2_v56 or
             q3h3_v57) and
            generation_profile and
            isinstance(execution_profile, dict) and
            execution_profile.get('envelope_version') == 53 and
            execution_profile.get('runtime_backend') == BACKEND_GPU and
            execution_profile.get('profile_backend') ==
                'backend_neutral_physical_profile' and
            isinstance(execution_profile.get('profile'), dict) and
            execution_profile['profile'].get('backend') is None and
            set(execution_profile.get('component_documents', {})) == {
                'capacity', 'channel_composition'})
        require((legacy_inline or gaia_inline or public_v53_inline) and
                metadata.get('fingerprint') == header[1].strip() and
                metadata.get('sidecars',{}).get('native_binary') is None and
                metadata.get('sidecars',{}).get('geometry') == 'geometry.json' and
                'geometry.json' in files,
                'inline geometry fingerprint differs')
    for line in (root/'case.data').read_text(encoding='utf-8').splitlines():
        row = line.split()
        if len(row) == 2 and re.fullmatch(r'[0-9a-fA-F]{64}', row[1]):
            require(files.get(row[0]) == row[1].lower(), 'native deck dependency is unauthenticated')
    expected_bins = {name for name in receipt['artifacts']
                     if name.endswith('.bin')}
    require({p.name for p in root.glob('*.bin') if p.name != 'case.data.bin'} == expected_bins,
            'unexpected native sidecar')
    execution_profile_sidecar(root, receipt)


def validate_diffusion(data):
    """Native cache structure only; GPU operator retains its numerical checks."""
    nu = data.get('horizontal_viscosity')
    require(type(nu) in (int, float) and np.isfinite(nu) and nu >= 0., 'invalid viscosity')
    cached = {name[10:] for name in data if name.startswith('diffusion_')}
    if nu == 0.:
        require(not cached, 'zero viscosity cannot carry a diffusion cache')
        return
    require(data.get('horizontal_viscosity_mode') == 1 and data.get('horizontal_viscosity_geometry') == 1,
            'viscosity requires native mode1 geometry1')
    nc, nf = int(data['nc']), int(data['nf'])
    cells = np.asarray(data.get('diffusion_recovery_cells'))
    require(cells.ndim == 1 and cells.dtype == np.int32 and
            np.all((cells >= 0) & (cells < nc)), 'invalid diffusion recovery cells')
    nr = len(cells)
    shapes = dict(conductance=(nf,), face_diffusivity=(nf,), tangent_offset=(nf,), normal_asymmetry=(nf,),
        face_bound=(nf,), rate=(nc,), recovery_inverse=(5,5,nr), recovery_affine_inverse=(2,2,nr),
        corrected_face=(nf,), recovery_slot=(nc,), recovery_cells=(nr,), recovery_order=(nr,),
        recovery_valid=(nr,), recovery_stencils=(256,nr), recovery_stencil_counts=(nr,), maximum_rate=())
    integers = {'corrected_face','recovery_slot','recovery_cells','recovery_order',
                'recovery_valid','recovery_stencils','recovery_stencil_counts'}
    require(cached == set(shapes), 'diffusion cache field inventory differs')
    for name, shape in shapes.items():
        value = np.asarray(data['diffusion_'+name])
        require(value.shape == shape and value.dtype == (np.int32 if name in integers else np.float64) and
                np.isfinite(value).all(), 'invalid diffusion cache '+name)
    for name in ('conductance','face_diffusivity','face_bound','rate','maximum_rate'):
        require(np.all(np.asarray(data['diffusion_'+name]) >= 0.), 'negative diffusion cache '+name)
    counts, stencils = data['diffusion_recovery_stencil_counts'], data['diffusion_recovery_stencils']
    require(np.all((counts >= 0) & (counts <= 256)), 'invalid diffusion stencil counts')
    for index, count in enumerate(counts):
        require(np.all((stencils[:count,index] >= 0) & (stencils[:count,index] < nc)),
                'invalid diffusion stencil cells')
    require(np.isin(data['diffusion_corrected_face'], (0,1)).all() and
            np.isin(data['diffusion_recovery_valid'], (0,1)).all() and
            np.all((data['diffusion_recovery_slot'] >= -1) & (data['diffusion_recovery_slot'] < nr)),
            'invalid diffusion cache flags/slots')


def validate_mpm_controls(data):
    mode = data.get('mpm_friction_mode')
    require(type(mode) is int and mode in (0,1), 'invalid MPM friction mode')
    for key in ('mpm_cf','mpm_critical'):
        require(type(data.get(key)) in (int,float) and np.isfinite(data[key]) and data[key] >= 0.,
                'invalid MPM coefficient '+key)
    if mode == 1:
        require(abs(data['mpm_critical']-.047) <= 1.e-14 and
                np.isfinite(data['manning_cell']).all() and np.all(data['manning_cell'] > 0.),
                'Manning MPM requires positive roughness and critical Shields0.047')


def validate_avalanche_controls(data):
    flag = data.get('general_avalanche')
    require(type(flag) is int and flag in (0,1), 'invalid avalanche switch')
    if flag:
        require(not data.get('bedform_roughness',0), 'avalanche excludes bedform roughness')
        for key in ('wet_slope','dry_slope','time','relaxation_length','wet_depth_switch'):
            value = data.get('general_avalanche_'+key)
            require(type(value) in (int,float) and np.isfinite(value) and
                    (value >= 0. if key == 'wet_depth_switch' else value > 0.),
                    'invalid avalanche control '+key)


def validate_mpm_method(method,data):
    # Historical Manning/bedlaw0 records predate these explicit fields. A new
    # specified-Cf branch may never masquerade as one of those records.
    if data['bedlaw'] == 2 and data['mpm_friction_mode'] == 0 or 'mpm_friction_mode' in method or 'mpm_cf' in method:
        require(type(method.get('mpm_friction_mode')) is int and
                method['mpm_friction_mode'] == data['mpm_friction_mode'] and
                type(method.get('mpm_cf')) is float and method['mpm_cf'] == float(data['mpm_cf']),
                'MPM method differs from native inputs')


def validate_engelund_hansen_controls(data, model):
    """Exact frozen v42/law-8 host envelope; no equation substitution."""
    chezy = data.get('engelund_hansen_chezy')
    calibration = data.get('transport_calibration')
    require(data['bedlaw'] == 8 and data['exchange_mode'] == 0 and
            data['bed_slope'] in (0, 2, 3, 5) and data['hiding_exposure'] == 0 and
            data.get('mpm_friction_mode') == 0 and
            type(chezy) in (int, float) and np.isfinite(chezy) and chezy > 0. and
            type(calibration) in (int, float) and calibration == 1. and
            model in (0, 1, 2),
            'Engelund-Hansen law 8 differs from its authenticated v42 envelope')
    if data['bed_slope'] == 3:
        require(float(data.get('bed_slope_alfabs', np.nan)) >= 0. and
                float(data.get('bed_slope_alfabn', np.nan)) > 0. and
                float(data.get('bed_friction_angle_deg', np.nan)) > 0.,
                'Koch-Flokstra law-8 controls differ')
    require(np.all(np.asarray(data['manning_cell']) > 0.),
            'Engelund-Hansen law 8 requires positive Manning roughness')


def validate_gaia_controls(data, model):
    """Exact authenticated v43 GAIA correspondence admitted by the GPU worker."""
    exact = {
        'gaia_skin_roughness_ratio': 3.,
        'gaia_slope_beta': 1.3,
        'gaia_deviation_beta2': .85,
        'gaia_secondary_alpha': 1.,
    }
    require(data['bedlaw'] == 9 and data['exchange_mode'] == 0 and
            data['bed_slope'] == 4 and data['hiding_exposure'] == 0 and
            data.get('mpm_friction_mode') == 0 and model == 3 and
            data.get('secondary_flow') == 1 and
            data.get('secondary_flow_transport_direction') == 1 and
            float(data.get('secondary_flow_momentum_beta', np.nan)) == 0. and
            float(data.get('secondary_flow_espir', np.nan)) == 1. and
            data.get('secondary_flow_inflow_mode') == 1 and
            data.get('nsubstrate_layers') == 1 and
            float(data.get('morphological_factor', np.nan)) == 1. and
            not data.get('morphological_acceleration', 0) and
            not data.get('bedload_adaptation', 0) and
            not data.get('bedform_roughness', 0) and
            not data.get('general_avalanche', 0) and
            float(data.get('horizontal_viscosity', np.nan)) == 0. and
            float(data.get('transport_calibration', np.nan)) == 1. and
            not suspended_enabled(data) and
            all(type(data.get(key)) in (int, float) and data[key] == value
                for key, value in exact.items()) and
            np.all(np.asarray(data['manning_cell']) > 0.),
            'TELEMAC-GAIA law 9 differs from its authenticated v43 envelope')


def validate_public_law11_controls(data, model):
    """Admit only the receipt-bound one-layer Q3-S/P total-load route."""
    require(has_public_law11_total_prepared_profile(data),
            'law 11 requires its authenticated public execution profile')
    require(int(data['nfrac']) == 1 and int(data['nbed_layers']) == 1 and
            int(data['nsubstrate_layers']) == 0 and
            int(data['exchange_mode']) == 0 and model == 0 and
            int(data['hiding_exposure']) == 0 and
            int(data['bed_slope']) == 0 and
            int(data.get('mpm_friction_mode', -1)) == 0 and
            not suspended_enabled(data) and
            not data.get('bedload_adaptation', 0) and
            not data.get('bedform_roughness', 0) and
            not data.get('general_avalanche', 0) and
            int(data.get('active_layer_mode', -1)) == 0 and
            float(data.get('active_layer_thickness', np.nan)) == 0. and
            float(data.get('active_layer_depth_fraction', np.nan)) == 0. and
            float(data.get('active_layer_minimum_thickness', np.nan)) == 0. and
            float(data.get('transport_calibration', np.nan)) == 1.,
            'public law-11 total route requires its exact one-layer, '
            'no-substrate controls')


def validate_public_q3c_controls(data, model):
    """Validate the common forcing-bound scalar route, not legacy law shells."""
    marker=data.get(_M1_PUBLIC_ADMISSION_KEY)
    law=int(data.get('bedlaw',-1))
    require(isinstance(marker,dict) and
            marker.get('admission_id')==_Q3C_PUBLIC_ADMISSION_ID and
            marker.get('formula_code')==law and
            has_public_law11_total_prepared_profile(data),
            'Q3-C scalar capacity requires its authenticated public profile')
    require(law in (2,8,11) and int(data['exchange_mode'])==0 and model==0 and
            int(data['hiding_exposure'])==0 and int(data['bed_slope'])==0 and
            int(data.get('mpm_friction_mode',-1))==(1 if law==2 else 0) and
            not suspended_enabled(data) and
            not data.get('bedload_adaptation',0) and
            not data.get('bedform_roughness',0) and
            not data.get('general_avalanche',0) and
            float(data.get('transport_calibration',np.nan))==1. and
            isinstance(data.get('delft_forcing_binding'),dict),
            'Q3-C scalar capacity controls differ from its authenticated profile')


def validate_public_q3h1_controls(data, model):
    """Validate additive hiding while preserving the exact Q3-C foundation."""
    marker = data.get(_M1_PUBLIC_ADMISSION_KEY)
    identity = validate_hiding_foundation_execution_document(
        data.get(HIDING_EXECUTION_KEY))
    law = int(data.get('bedlaw', -1))
    expected_route = {
        'ordered_parts': [
            'capacity_execution_fingerprint',
            'capacity_profile_fingerprint',
            'hiding_execution_fingerprint'],
        'capacity_execution_fingerprint': identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint': identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint': identity['fingerprint'],
    }
    require(isinstance(marker, dict) and
            marker.get('admission_id') == _Q3H1_PUBLIC_ADMISSION_ID and
            marker.get('formula_code') == law and
            marker.get('hiding_execution_fingerprint') ==
                identity['fingerprint'] and
            has_public_law11_total_prepared_profile(data) and
            law in (2, 11) and int(data['nfrac']) == 2 and
            int(data['nbed_layers']) >= 2 and
            int(data['nsubstrate_layers']) == int(data['nbed_layers']) - 1 and
            int(data['exchange_mode']) == 0 and model == 0 and
            int(data['hiding_exposure']) == identity['mode'] and
            int(data['bed_slope']) == 0 and
            int(data.get('mpm_friction_mode', -1)) == (1 if law == 2 else 0) and
            tuple(data.get('capacity_fraction_ids', ())) ==
                tuple(identity['fraction_ids']) and
            data.get(HIDING_ORDERED_ROUTE_KEY) == expected_route and
            not suspended_enabled(data) and
            not data.get('bedload_adaptation', 0) and
            not data.get('bedform_roughness', 0) and
            not data.get('general_avalanche', 0) and
            float(data.get('transport_calibration', np.nan)) == 1. and
            isinstance(data.get('delft_forcing_binding'), dict),
            'Q3-H1 hiding controls differ from their authenticated identity')


def validate_public_q3h2_controls(data,model):
    identity=validate_secondary_foundation_execution_document(
        data.get(SECONDARY_EXECUTION_KEY))
    route=data.get(SECONDARY_ORDERED_ROUTE_KEY)
    marker=data.get(_M1_PUBLIC_ADMISSION_KEY)
    hiding=data.get(HIDING_EXECUTION_KEY)
    expected_hiding=(
        validate_hiding_foundation_execution_document(hiding)['fingerprint']
        if hiding is not None else None)
    require(isinstance(marker,dict) and marker.get('admission_id') in (
                _Q3C_PUBLIC_ADMISSION_ID,_Q3H1_PUBLIC_ADMISSION_ID) and
            identity['method_id']=='secondary.equilibrium' and
            identity['model']==1 and identity['capacity_law']==data['bedlaw'] and
            identity['base_envelope_version']==(55 if hiding is not None else 54) and
            identity['hiding_execution_fingerprint']==expected_hiding and
            identity['roughness_method_id']==
                data.get('capacity_roughness_method_id') and
            identity['roughness_fingerprint']==
                data.get('delft_forcing_fingerprint') and
            identity['coriolis_disposition']=='zero' and
            identity['coriolis_constant_s_inverse']==0. and
            identity['coriolis_source']=='constant_zero_planar' and
            identity['betac']==0. and model==1 and
            int(data.get('secondary_flow',0))==1 and
            int(data.get('secondary_flow_transport_direction',0))==1 and
            float(data.get('secondary_flow_momentum_beta',0.))==0. and
            float(data.get('secondary_flow_espir',np.nan))==identity['espir'] and
            int(data['exchange_mode'])==0 and int(data['bed_slope'])==0 and
            not np.any(data.get('amr_enabled',0)) and
            not data.get('bedload_adaptation',0) and
            not data.get('bedform_roughness',0) and
            route=={
                'ordered_parts':['capacity_execution_fingerprint',
                                 'capacity_profile_fingerprint',
                                 'hiding_execution_fingerprint',
                                 'secondary_execution_fingerprint'],
                'capacity_execution_fingerprint':identity[
                    'capacity_execution_fingerprint'],
                'capacity_profile_fingerprint':identity[
                    'capacity_profile_fingerprint'],
                'hiding_execution_fingerprint':expected_hiding,
                'secondary_execution_fingerprint':identity['fingerprint']},
            'Q3-H2 secondary controls differ from authenticated identity')


def validate_public_q3h3_controls(data,model):
    """Authenticate the bounded v57 classic prognostic-secondary route."""
    require(SECONDARY_EXECUTION_KEY not in data,
            'Q3-H3 cannot also execute the Q3-H2 equilibrium rotation')
    identity=validate_prognostic_secondary_execution_document(
        data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY))
    foundation=load_capacity_foundation_execution(
        data.get('capacity_foundation_execution_identity'))
    hiding=data.get(HIDING_EXECUTION_KEY)
    expected_hiding=(
        validate_hiding_foundation_execution_document(hiding)['fingerprint']
        if hiding is not None else None)
    marker=data.get(_M1_PUBLIC_ADMISSION_KEY)
    expected_marker=(_Q3H1_PUBLIC_ADMISSION_ID if hiding is not None else
                     _Q3C_PUBLIC_ADMISSION_ID)
    route={
        'capacity_execution_fingerprint':foundation['fingerprint'],
        'capacity_profile_fingerprint':data.get(
            'capacity_profile_fingerprint'),
        'hiding_execution_fingerprint':expected_hiding,
        'prognostic_secondary_execution_fingerprint':identity['fingerprint'],
    }
    layout=identity['runtime_layout']
    require(isinstance(marker,dict) and
            marker.get('admission_id')==expected_marker and
            data.get(PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY)==route and
            identity['capacity_execution_fingerprint']==foundation['fingerprint'] and
            identity['capacity_profile_fingerprint']==
                data.get('capacity_profile_fingerprint') and
            identity['hiding_execution_fingerprint']==expected_hiding and
            identity['base_envelope_version']==(55 if hiding is not None else 54) and
            identity['capacity_law']==int(data['bedlaw']) and
            identity['roughness_method_id']==
                data.get('capacity_roughness_method_id') and
            identity['roughness_fingerprint']==
                data.get('delft_forcing_fingerprint') and
            layout['fraction_count']==int(data['nfrac']) and
            layout['bed_layer_count']==int(data['nbed_layers']) and
            layout['physical_state_count']==int(data['nphysical_state']) and
            layout['runtime_state_count']==int(data['nstate']) and
            layout['secondary_state_index']==int(data['secondary_index']) and
            layout['physical_primitive_count']==int(data['nphysical_primitive']) and
            layout['runtime_primitive_count']==int(data['nprimitive']) and
            layout['secondary_primitive_index']==
                int(data['secondary_primitive_index']) and
            layout['physical_ledger_count']==int(data['nphysical_ledger']) and
            layout['runtime_ledger_count']==int(data['nledger']) and
            layout['secondary_ledger_index']==
                int(data['secondary_ledger_index']),
            'Q3-H3 identity differs from capacity, hiding, layout, or roughness')
    require(identity['transport_direction'] is True and
            identity['inflow_mode']==1 and
            identity['coriolis_disposition']=='zero' and
            identity['coriolis_constant_s_inverse']==0. and
            identity['coriolis_source']=='constant_zero_planar' and
            identity['betac']==0. and
            identity['scalar_diffusivity_m2_s']==0. and model==2 and
            int(data.get('secondary_flow',0))==1 and
            int(data.get('secondary_flow_transport_direction',0))==1 and
            int(data.get('secondary_flow_inflow_mode',-1))==1 and
            float(data.get('secondary_flow_momentum_beta',np.nan))==0. and
            float(data.get('secondary_flow_espir',np.nan))==identity['espir'] and
            int(data['exchange_mode'])==0 and int(data['bed_slope'])==0 and
            not np.any(data.get('amr_enabled',0)) and
            not data.get('bedload_adaptation',0) and
            not data.get('bedform_roughness',0) and
            float(data.get('horizontal_viscosity',np.nan))==0. and
            not np.any(data.get('bank_failure',0)) and
            not np.any(data.get('bank_front_tracking',0)) and
            not np.any(data.get('bank_toe_erosion_mode',0)) and
            not np.any(data.get('bank_avalanche',0)) and
            not np.any(data.get('general_avalanche',0)) and
            not np.any(data.get('bank_erosion_fraction',0.)),
            'Q3-H3 first route requires fixed-grid prognostic secondary, '
            'zero Coriolis/Betac/diffusivity, and slope/exchange/adaptation/'
            'bedform/viscosity/bank/avalanche physics off')
    state=np.asarray(data.get('state'))
    secondary=np.asarray(data.get('secondary_mass'))
    require(secondary.shape==(int(data['nc']),) and
            state.shape[0]==layout['runtime_state_count'] and
            np.array_equal(secondary,state[layout['secondary_state_index']]),
            'Q3-H3 transported hI state differs from its authenticated slot')
    return identity


def validate_scope(data):
    from .gpu_prepared_io import _bank_secondary_layout
    nc, ng, nl, nb = (int(data[k]) for k in ('nc', 'nfrac', 'nbed_layers', 'nb'))
    inert = nl == 1 and ng == 1 and data.get('nsubstrate_layers') == 0
    q3h3_secondary=PROGNOSTIC_SECONDARY_EXECUTION_KEY in data
    require(nc > 0 and 1 <= ng <= 32 and (2 <= nl <= 33 or inert) and
            (nb >= 0 if q3h3_secondary else nb >= 1),
            'invalid material/grid dimensions')
    model = _bank_secondary_layout(data, allow_gaia=data.get('bedlaw') == 9)
    # A boundary-free Cartesian domain is admitted only after authenticating
    # the exact v57 composition.  A bare numeric model=2 cannot use this path.
    if nb == 0:
        validate_public_q3h3_controls(data,model)
    ns = 3 + ng*(1+nl)
    state, layers = np.asarray(data.get('state')), np.asarray(data.get('layer_mass'))
    require(state.shape == (ns+(model == 2), nc) and state.dtype == np.float64 and
            np.isfinite(state).all() and layers.shape == (nl, ng, nc) and layers.dtype == np.float64 and
            np.isfinite(layers).all() and np.all(state[0] >= 0.) and np.all(state[3:ns] >= 0.) and
            np.array_equal(data.get('suspended_indices'), np.arange(3, 3+ng)) and
            np.array_equal(data.get('bed_indices'), np.arange(3+ng, ns).reshape(nl, ng)) and
            np.array_equal(layers, state[3+ng:ns].reshape(nl, ng, nc)),
            'invalid native physical state inventory/layout')
    if 'suspended_mass' in data:
        require(np.shape(data['suspended_mass']) == (ng,nc) and
                np.array_equal(data['suspended_mass'],state[3:3+ng]),
                'derived suspended mass differs from state slots')
    public_law11 = has_public_law11_total_prepared_profile(data)
    marker=data.get(_M1_PUBLIC_ADMISSION_KEY)
    q3c_scalar=(isinstance(marker,dict) and
                marker.get('admission_id')==_Q3C_PUBLIC_ADMISSION_ID)
    q3h_scalar=(isinstance(marker,dict) and
                marker.get('admission_id')==_Q3H1_PUBLIC_ADMISSION_ID)
    q3h2_secondary=SECONDARY_EXECUTION_KEY in data
    for key, options in (('scheme', (6,)), ('cpu_time_integrator', (33,)),
            ('cpu_outlet_extension', (0,)), ('exchange_mode', (0, 1, 2)),
            ('settling_mode', (0,)), ('hiding_exposure', (0, 2, 3, 4, 5)),
            ('bed_slope', (0, 2, 3, 4, 5)), ('active_layer_mode', (0,))):
        require(type(data.get(key)) is int and data[key] in options, 'unsupported '+key)
    require(type(data.get('bedlaw')) is int and
            (data['bedlaw'] in (0, 2, 8, 9) or
             (data['bedlaw'] == 11 and public_law11)),
            'unsupported bedlaw')
    surveyed = validate_boundary_scope(
        data,allow_q3h3_closed_wall=q3h3_secondary)
    control = np.asarray(data['section_control'])
    for key in ('amr_enabled', 'bank_failure', 'bank_front_tracking', 'bank_avalanche',
            'bank_erosion_fraction',
            'viscosity_equilibrated_inlet', 'morphological_acceleration', 'nmorfac',
            'bank_toe_erosion_mode'):
        require(not np.any(data.get(key, 0)), 'unsupported '+key)
    require(float(data['morphological_factor']) == 1. and int(data['nsubstrate_layers']) == nl-1,
            'invalid morphology/stratigraphy controls')
    validate_mpm_controls(data)
    validate_avalanche_controls(data)
    if inert:
        expected_inert_model=(2 if q3h3_secondary else
                              1 if q3h2_secondary else 0)
        require((data['bedlaw'] == 0 or public_law11 or q3c_scalar) and
                data['exchange_mode'] == 0 and model==expected_inert_model and
                not suspended_enabled(data) and not data['general_avalanche'] and
                not data.get('bedload_adaptation',0) and not data.get('bedform_roughness',0) and
                data.get('active_layer_thickness') == 0. and data.get('active_layer_mode') == 0 and
                data.get('active_layer_depth_fraction',0.) == 0. and
                data.get('active_layer_minimum_thickness',0.) == 0.,
                'single-layer native controls differ from their authenticated route')
    adaptation = data.get('bedload_adaptation')
    require(type(adaptation) is int and adaptation in (0, 1), 'invalid adaptation switch')
    if adaptation:
        length = data.get('bedload_adaptation_length_diameters')
        require(data['bedlaw'] == 2 and type(length) in (int, float) and np.isfinite(length) and length > 0.,
                'adaptation requires MPM2 and positive finite length')
    foundation_model = 0 if (q3h2_secondary or q3h3_secondary) else model
    if q3h_scalar:
        validate_public_q3h1_controls(data,foundation_model)
    elif q3c_scalar:
        validate_public_q3c_controls(data,foundation_model)
    elif data['bedlaw'] == 0:
        require(model == 0 and data.get('mpm_friction_mode') == 0 and data['hiding_exposure'] == 0 and
                data['bed_slope'] == 0 and not np.any(data['fraction_bed_input']) and
                not np.any(data.get('fraction_series_bed_input', 0)) and not np.any(data['bed_boundary_mode']),
                'no-bedload mode requires neutral MPM/secondary controls and zero bed feed')
    elif data['bedlaw'] == 2:
        require(data.get('mpm_friction_mode') in (0,1) and (not data['hiding_exposure'] or ng > 1),
                'MPM/hiding combination differs')
    elif data['bedlaw'] == 8:
        validate_engelund_hansen_controls(data, model)
    elif data['bedlaw'] == 9:
        validate_gaia_controls(data, model)
    else:
        validate_public_law11_controls(data, model)
    if q3h2_secondary:
        validate_public_q3h2_controls(data,model)
    if q3h3_secondary:
        validate_public_q3h3_controls(data,model)
    roughness = data.get('bedform_roughness')
    require(type(roughness) is int and roughness in (0,1), 'invalid bedform switch')
    if roughness:
        require(data['bedlaw'] == 2 and data['mpm_friction_mode'] == 1 and model == 0 and
                not adaptation and not suspended_enabled(data) and
                not (surveyed and control[0] == 1) and np.all(data['manning_cell'] > 0.) and
                np.isfinite(data['mpm_critical']) and data['mpm_critical'] >= 0.,
                'bedform roughness requires MPM, positive Manning, no suspension/secondary/adaptation or normal inlet')
    for key in ('hiding_parker_alpha', 'hiding_wu_exponent'):
        require(np.isfinite(data.get(key, np.nan)), 'invalid '+key)
    require(inert or (np.isfinite(data.get('active_layer_thickness', np.nan)) and data['active_layer_thickness'] > 0.),
            'positive fixed active-layer thickness required')
    for key in ('d50_fraction', 'd90_fraction', 'bulk_fraction', 'rhos_fraction', 'z0_fraction',
                'reference_height_fraction', 'dstar_fraction', 'taucr_fraction'):
        value = np.asarray(data.get(key))
        require(value.shape == (ng,) and value.dtype == np.float64 and
                np.isfinite(value).all() and np.all(value > 0.), 'invalid exchange parameter '+key)
    for key in ('ws_fraction', 'sedthr_fraction'):
        value = np.asarray(data.get(key))
        require(value.shape == (ng,) and value.dtype == np.float64 and
                np.isfinite(value).all() and np.all(value >= 0.), 'invalid exchange parameter '+key)
    require(np.all(np.asarray(data['d90_fraction']) >= data['d50_fraction']) and
            np.all(np.asarray(data['rhos_fraction']) > float(data['rho'])) and
            np.shape(data.get('sediment_type_fraction')) == (ng,) and
            not np.any(data['sediment_type_fraction']), 'invalid/non-sand material')
    for key in ('rho', 'grav', 'dry_tolerance', 'hindered_settling_reference_concentration'):
        require(type(data.get(key)) in (int, float) and np.isfinite(data[key]) and data[key] > 0.,
                'invalid positive exchange control '+key)
    require(np.shape(data.get('manning_cell')) == (nc,) and np.isfinite(data['manning_cell']).all() and
            np.all(data['manning_cell'] >= 0.) and np.shape(data.get('sediment_roughness_fraction')) == (ng,) and
            np.isin(data['sediment_roughness_fraction'], (0, 1)).all(), 'invalid exchange roughness')
    validate_diffusion(data)
    return model


def _validate_gaia_fixed_face_contract(data, exporter):
    """Authenticate historical absence or an explicit native face bed."""
    keys = {'fixed_boundary_bed_mode', 'fixed_boundary_bed'}
    present = keys.intersection(data)
    if exporter in DELFT_EXPORTER_GENERATIONS:
        # v7 declares the record inventory explicitly for every law. These
        # GAIA-only arrays must be neutral for the Delft3D component route.
        require(present == keys and int(data['bedlaw']) in (0,2,8,11),
                'v7 Delft component export lacks explicit neutral fixed-face data')
        fixed_mode = np.asarray(data['fixed_boundary_bed_mode'])
        fixed_bed = np.asarray(data['fixed_boundary_bed'])
        require(fixed_mode.shape == (int(data['nb']),) and fixed_mode.dtype == np.int32 and
                fixed_bed.shape == (int(data['nf']),) and fixed_bed.dtype == np.float64 and
                np.isfinite(fixed_bed).all() and not np.any(fixed_mode) and
                not np.any(fixed_bed) and not np.any(np.signbit(fixed_bed)),
                'v7 non-GAIA fixed-face arrays must be inert positive zero')
        return 'explicit_native_v5'
    if exporter == GAIA_EXPORTER_V4_SHA256:
        require(not present, 'historical v4 GAIA export has unexpected fixed-face data')
        return 'historical_absent'
    if exporter not in (GAIA_EXPORTER_V5_SHA256, GAIA_EXPORTER_V6_SHA256):
        require(not present, 'historical transport export has unexpected fixed-face data')
        return 'historical_absent'
    generation = 'v5' if exporter == GAIA_EXPORTER_V5_SHA256 else 'v6'
    require(present == keys, 'native '+generation+' GAIA export lacks fixed-face data')
    nb, nf, nc = (int(data[name]) for name in ('nb', 'nf', 'nc'))
    fixed_mode = np.asarray(data['fixed_boundary_bed_mode'])
    fixed_bed = np.asarray(data['fixed_boundary_bed'])
    bid, owner = np.asarray(data['bid']), np.asarray(data['owner'])
    bed_mode = np.asarray(data['bed_boundary_mode'])
    mode = np.asarray(data['mode'])
    allowed_bed_modes = ((0, 1) if exporter == GAIA_EXPORTER_V5_SHA256
                         else (0, 1, 2))
    expected_fixed_mode = ((bed_mode == 1) if exporter == GAIA_EXPORTER_V5_SHA256
                           else np.isin(bed_mode, (1, 2)))
    require(fixed_mode.shape == (nb,) and fixed_mode.dtype == np.int32 and
            np.isin(fixed_mode, (0, 1)).all() and
            fixed_bed.shape == (nf,) and fixed_bed.dtype == np.float64 and
            np.isfinite(fixed_bed).all() and bid.shape == owner.shape == (nf,) and
            np.all((owner >= 0) & (owner < nc)) and bed_mode.shape == mode.shape == (nb,) and
            bed_mode.dtype == np.int32 and np.isin(bed_mode, allowed_bed_modes).all() and
            np.array_equal(fixed_mode, expected_fixed_mode.astype(np.int32)) and
            np.all(mode[fixed_mode == 1] == 6),
            'native '+generation+' GAIA fixed-face mode differs')
    prescribed = bed_mode == 2
    if np.any(prescribed):
        ng = int(data['nfrac'])
        constant = np.asarray(data['fraction_bed_input'])
        series = np.asarray(data['fraction_series_bed_input'])
        require(constant.shape == (ng, nb) and series.ndim == 3 and
                series.shape[1:] == (ng, nb) and
                np.isfinite(constant).all() and np.isfinite(series).all() and
                np.all(constant >= 0.) and np.all(series >= 0.) and
                np.all(np.any(constant[:, prescribed] > 0., axis=0) |
                       np.any(series[:, :, prescribed] > 0., axis=(0, 1))),
                'native v6 prescribed fixed-face inlet lacks per-fraction feed')
    face_fixed = np.zeros(nf, dtype=bool)
    exterior = bid >= 0
    face_fixed[exterior] = fixed_mode[bid[exterior]] == 1
    bed = np.asarray(data['bed'])
    offset = np.asarray(data['boundary_bed_offset'])
    require(bed.shape == (nc,) and offset.shape == (nf,),
            'native '+generation+' GAIA fixed-face geometry differs')
    expected = bed[owner[face_fixed]] + offset[face_fixed]
    require(np.array_equal(fixed_bed[face_fixed], expected) and
            not np.any(fixed_bed[~face_fixed]) and
            not np.any(np.signbit(fixed_bed[~face_fixed])),
            'native '+generation+' GAIA fixed-face elevation differs')
    return 'explicit_native_v5'


def read_records(root, data, receipt):
    from .gpu_prepared_io import _read_records, _bank_npz_dtype
    merged = dict(data)
    diffusion = root/DIFFUSION_ARTIFACT
    require(diffusion.is_file() == (data.get('horizontal_viscosity', 0.) > 0.),
            'native diffusion artifact does not match viscosity')
    if diffusion.is_file():
        for key, value in _read_records(diffusion).items():
            require('diffusion_'+key not in merged, 'duplicate diffusion cache record')
            merged['diffusion_'+key] = value
    for key, value in _read_records(root/'normal_metadata.bin').items():
        require(key not in merged or np.array_equal(merged[key], value), 'duplicate native record differs: '+key)
        merged[key] = value
    avalanche = root/AVALANCHE_ARTIFACT
    require(avalanche.is_file() == bool(merged.get('general_avalanche',0)),
            'native avalanche artifact does not match switch')
    if avalanche.is_file():
        for key,value in _read_records(avalanche).items():
            require(key not in merged or np.array_equal(merged[key],value),
                    'duplicate avalanche record differs: '+key)
            merged[key] = value
    merged.update(q=np.ascontiguousarray(data['state'][:3]), dry=data['dry_tolerance'],
        outlet_extension=data['cpu_outlet_extension'], dt=data['dtmax'],
        bedoffset=data['boundary_bed_offset'], bc_hydraulic=np.ascontiguousarray(data['forcing'][:2]),
        bc_bed_input=np.asarray(data['fraction_bed_input']).copy())
    for key in ('section_seed_node', 'boundary_seed_position'):
        raw = np.asarray(merged['native_'+key])
        require(raw.shape == (int(data['nb']),) and raw.dtype == np.int32 and np.all(raw >= 0), 'invalid native '+key)
        merged[key] = np.ascontiguousarray(raw-1)
    exporter = receipt.get('exporter_executable_sha256')
    fixed_contract = _validate_gaia_fixed_face_contract(merged, exporter)
    if fixed_contract == 'explicit_native_v5':
        merged['gaia_fixed_inlet_contract'] = fixed_contract
    with np.load(root/'coupled_input.npz', allow_pickle=False) as archive:
        if (fixed_contract == 'historical_absent' and
                'gaia_fixed_inlet_contract' in archive.files):
            merged['gaia_fixed_inlet_contract'] = fixed_contract
        require(len(archive.files) == len(set(archive.files)) and set(archive.files) == set(merged) and
            all(archive[k].dtype == _bank_npz_dtype(merged[k]) and np.array_equal(archive[k], merged[k])
                for k in archive.files), 'NPZ differs from native records')
    if fixed_contract == 'historical_absent' and 'gaia_fixed_inlet_contract' not in merged:
        # This marker is descriptive only and deliberately added after the
        # frozen v4 NPZ comparison; no missing physical array is fabricated.
        merged['gaia_fixed_inlet_contract'] = fixed_contract
    if exporter in _DELFT_FORCING_EXPORTERS:
        binding = json.loads((root/FORCING_ARTIFACT).read_text(encoding='utf-8'))
        apply_forcing_binding(merged, binding)
    if exporter == DELFT_EXPORTER_V15_SHA256:
        metadata = json.loads((root/'case.data.json').read_text(encoding='utf-8'))
        deck_lines = (root/'case.data').read_text(
            encoding='utf-8').splitlines()
        hiding_identity, hiding_route = _v55_hiding_identity(
            metadata, deck_lines)
        _validate_native_v55_hiding(merged, hiding_identity)
        merged[HIDING_EXECUTION_KEY] = hiding_identity
        merged[HIDING_ORDERED_ROUTE_KEY] = hiding_route
    elif exporter == DELFT_EXPORTER_V17_SHA256:
        metadata = json.loads((root/'case.data.json').read_text(encoding='utf-8'))
        deck_lines = (root/'case.data').read_text(
            encoding='utf-8').splitlines()
        (secondary_identity, secondary_route,
         hiding_identity, hiding_route) = _v56_secondary_identity(
             metadata, deck_lines)
        if hiding_identity is not None:
            _validate_native_v55_hiding(merged, hiding_identity)
            merged[HIDING_EXECUTION_KEY] = hiding_identity
            merged[HIDING_ORDERED_ROUTE_KEY] = hiding_route
        _validate_native_v56_secondary(merged, secondary_identity)
    elif exporter == DELFT_EXPORTER_V18_SHA256:
        metadata=json.loads((root/'case.data.json').read_text(encoding='utf-8'))
        deck_lines=(root/'case.data').read_text(
            encoding='utf-8').splitlines()
        (prognostic_identity,prognostic_route,
         hiding_identity,hiding_route)=_v57_prognostic_secondary_identity(
             metadata,deck_lines)
        if hiding_identity is not None:
            _validate_native_v55_hiding(merged,hiding_identity)
            merged[HIDING_EXECUTION_KEY]=hiding_identity
            merged[HIDING_ORDERED_ROUTE_KEY]=hiding_route
        _validate_native_v57_prognostic_secondary(merged,prognostic_identity)
    execution_profile_sidecar(root, receipt, data=merged)
    if exporter == DELFT_EXPORTER_V17_SHA256:
        # First authenticate the frozen nested profile with secondary.off,
        # then install the additive outer composition for result/runtime use.
        _install_inline_secondary_identity(
            merged, secondary_identity, secondary_route)
    if exporter == DELFT_EXPORTER_V18_SHA256:
        _install_inline_prognostic_secondary_identity(
            merged,prognostic_identity,prognostic_route)
    validate_scope(merged)
    return merged


def _execution_profile_run_context(data, method, manifest):
    bound = data.get(_EXECUTION_PROFILE_DATA_KEY)
    if bound is None:
        require(not any(key in method or key in manifest
                        for key in EXECUTION_PROFILE_IDENTITY_FIELDS) and
                method.get('execution_profile_v2_bound') is None and
                method.get('execution_profile_v2_sidecar_schema') is None and
                method.get('execution_profile_v2_components') is None and
                manifest.get('execution_profile_v2') is None,
                'run declares an execution profile absent from preparation')
        return None
    identities = bound['identities']
    components = bound['components']
    require(all(method.get(key) == value and manifest.get(key) == value
                for key, value in identities.items()) and
            method.get('execution_profile_v2_bound') is True and
            method.get('execution_profile_v2_sidecar_schema') ==
                EXECUTION_PROFILE_SIDECAR_SCHEMA and
            method.get('execution_profile_v2_components') == components,
            'run execution-profile method/identity differs from preparation')
    expected = dict(schema=EXECUTION_PROFILE_SIDECAR_SCHEMA,
                    **identities, components=components)
    require(manifest.get('execution_profile_v2') == expected,
            'run execution-profile manifest differs from preparation')
    runtime_identities=dict(identities)
    prognostic=data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)
    if prognostic is not None:
        identity=validate_prognostic_secondary_execution_document(prognostic)
        runtime_identities.update(
            prognostic_secondary_execution_fingerprint=identity['fingerprint'],
            prognostic_secondary_state_semantics=identity['state_semantics'],
            prognostic_secondary_integrated_ledger_units=
                identity['integrated_ledger_units'],
            prognostic_secondary_bed_boundary_rotation_policy=
                identity['bed_boundary_rotation_policy'],
            prognostic_secondary_composition_order=
                list(identity['composition_order']))
        require(all(method.get(key)==value for key,value in
                    _q3c_profile_identity(data).items()),
                'run prognostic-secondary identity differs from preparation')
    return dict(identities=runtime_identities, components=list(components),
                record=expected)


def preparation_context(root, receipt, manifest, ncell):
    from .gpu_prepared_io import read_gpu_prepared
    method = manifest['method']
    catalog_fields = (
        'sediment_catalog_schema', 'sediment_catalog_scope_id',
        'semantic_catalog_id',
    )
    catalog_values = tuple(method.get(key) for key in catalog_fields)
    if all(value is None for value in catalog_values):
        # Frozen schema-6 evidence predates the catalogue fields.  Its pinned
        # exporter, preparation and run identities remain independently
        # authenticated.  New schema-7 preparation cannot use this exception,
        # and a partial modern identity is never accepted.
        require(receipt.get('schema') == 6,
                'current run lacks its sediment-method catalogue')
    else:
        require(method.get('sediment_catalog_schema') == CATALOG_SCHEMA and
                method.get('sediment_catalog_scope_id') == CATALOG_SCOPE_ID and
                method.get('semantic_catalog_id') == semantic_catalog_id(),
                'run uses a different sediment-method catalogue')
    if receipt.get('schema') in (6, 7):
        require(method.get('variant') == VARIANT,
                'schema-6/7 preparation needs its transport variant')
    if method.get('variant') != VARIANT: return None
    # The prepared data owns the coupling-specific implicit ABI.  It is read
    # below, so defer the exact schema comparison until after authentication.
    require(receipt.get('schema') in (6, 7) and
            type(manifest.get('suspended_enabled')) is bool and
            type(method.get('transport_checkpoint_schema')) is int and
            type(manifest.get('transport_checkpoint_schema')) is int,
            'method/prepared schema differs')
    data = read_gpu_prepared(root, receipt_sha256=manifest['input_preparation_sha256'])
    checkpoint_schema = (implicit_checkpoint_schema(data)
                         if method.get('integrator') == _IMPLICIT_LABEL else 1)
    require(method['transport_checkpoint_schema'] == checkpoint_schema and
            manifest['transport_checkpoint_schema'] == checkpoint_schema,
            'method/prepared schema differs')
    execution_profile = _execution_profile_run_context(data, method, manifest)
    validate_integrator_scope(bedlaw=data['bedlaw'], exchange_mode=data['exchange_mode'],
                              integrator=method.get('integrator'),
                              launch_mode=method.get('launch_mode'),
                              bedform_roughness=data.get('bedform_roughness', 0),
                              bedload_adaptation=data.get('bedload_adaptation', 0),
                              general_avalanche=data.get('general_avalanche', 0),
                              schema_v2_law11=(
                                  int(data['bedlaw']) == 11 and
                                  execution_profile is not None))
    implicit = _validate_implicit_method(method, manifest, data)
    if implicit:
        validate_implicit_prepared_scope(data, dtmax=method.get('dtmax_s'))
    require(data['nc'] == ncell and manifest.get('input_identity') == numerical_identity(receipt),
            'prepared numerical identity differs')
    suspended = suspended_enabled(data)
    validate_mpm_method(method,data)
    # One physical row is also the authenticated Q3-C active reservoir for a
    # single sediment class.  It is static only when the numerical transport
    # operators are actually OFF, matching the worker's exact predicate.
    static = (data['nbed_layers'] == 1 and data['nsubstrate_layers'] == 0 and
              int(data['bedlaw']) == 0 and int(data['exchange_mode']) == 0)
    require(type(method.get('static_bed',False)) is bool and
            method.get('static_bed',False) == static, 'static-bed method differs from native inputs')
    require(manifest['suspended_enabled'] == suspended, 'suspended allocation differs from native inputs')
    if data['exchange_mode'] == 0 or 'suspended_enabled' in method:
        require(type(method.get('suspended_enabled')) is bool and method['suspended_enabled'] == suspended,
                'suspended method differs from native inputs')
    bedform = bool(data['bedform_roughness'])
    for record in (method, manifest):
        require(type(record.get('bedform_roughness', False)) is bool and
                record.get('bedform_roughness', False) == bedform, 'bedform method/manifest differs')
    avalanche = bool(data['general_avalanche'])
    for record in (method,manifest):
        require(type(record.get('general_avalanche',False)) is bool and
                record.get('general_avalanche',False) == avalanche, 'avalanche method/manifest differs')
    if avalanche:
        require(method.get('avalanche_order') == 'each_euler_after_exner_before_rebalance',
                'avalanche transaction order differs')
    for target, source in (('secondary_flow_model', 'secondary_flow_model'), ('bedlaw', 'bedlaw'),
            ('exchange_mode', 'exchange_mode'), ('hiding_exposure', 'hiding_exposure'),
            ('bed_slope', 'bed_slope'), ('fractions', 'nfrac'), ('bed_layers', 'nbed_layers')):
        require(type(method.get(target)) is int and method[target] == data[source], 'method differs: '+target)
    nu = float(data['horizontal_viscosity'])
    require(type(manifest.get('diffusion_enabled', False)) is bool and
            manifest.get('diffusion_enabled', False) == (nu > 0.), 'diffusion manifest differs')
    if nu > 0. or 'horizontal_viscosity' in method:
        require(type(method.get('horizontal_viscosity')) is float and method['horizontal_viscosity'] == nu,
                'viscosity method differs')
        for key in ('horizontal_viscosity_mode','horizontal_viscosity_geometry'):
            require(type(method.get(key)) is int and method[key] == data[key], 'method differs: '+key)
    adaptation = bool(data['bedload_adaptation'])
    require(type(method.get('bedload_adaptation', False)) is bool and
            method.get('bedload_adaptation', False) == adaptation, 'adaptation method differs')
    length = method.get('bedload_adaptation_length_diameters')
    if adaptation or length is not None:
        require(type(length) is float and length == float(data['bedload_adaptation_length_diameters']),
                'adaptation length differs')
    gaia_metadata = method.get('gaia_secondary_diagnostic')
    gaia_diagnostic = gaia_metadata is not None
    if gaia_diagnostic:
        require(data['bedlaw'] == 9 and data['secondary_flow_model'] == 3 and
                gaia_metadata == _gaia_diagnostic_method() and
                manifest.get('gaia_secondary_diagnostic_schema') == 1,
                'GAIA secondary-stress diagnostic method differs')
    else:
        # Historical authenticated model-3 runs did not carry this derived
        # group.  Continue to read them; new writers declare schema 1.
        require(manifest.get('gaia_secondary_diagnostic_schema') in (None,),
                'undeclared GAIA secondary-stress diagnostic schema')
    event = manifest.get('boundary_events')
    require(isinstance(event, dict) and event.get('identity') == manifest.get('event_identity') ==
            method.get('event_identity') and isinstance(event.get('knots_s'), list), 'forcing identity differs')
    return dict(nfrac=data['nfrac'], nlayer=data['nbed_layers'], nb=data['nb'],
        nc=data['nc'], nf=data['nf'], implicit=implicit,
        checkpoint_schema=checkpoint_schema,
        execution_profile=execution_profile,
        secondary_model=data['secondary_flow_model'], bedlaw=data['bedlaw'],
        adaptation=adaptation, suspended_enabled=suspended, bedform=bedform,
        gaia_diagnostic=gaia_diagnostic,
        dry_tolerance=float(data['dry_tolerance']), rock=np.asarray(data['rock']),
        dry=float(data['dry']), grav=float(data['grav']),
        area=np.asarray(data['area']), length=np.asarray(data['length']),
        normal=np.asarray(data['normal']), owner=np.asarray(data['owner']),
        neighbour=np.asarray(data['neighbour']), bulk=np.asarray(data['bulk_fraction']))


def validate_frame(archive, manifest, context, required):
    from .gpu_frame_io import _scalar, _string_scalar, _identity
    require(context is not None, 'missing authenticated preparation context')
    implicit = context.get('implicit', False)
    checkpoint = context.get('checkpoint', False)
    require(type(checkpoint) is bool,
            'invalid transport checkpoint readback context')
    expected = (set(required) | _EXPLICIT_COMPACT |
                (_IMPLICIT_COMPACT if implicit and checkpoint else frozenset()) |
                {'event_identity', 'next_boundary_knot'})
    execution_profile = context.get('execution_profile')
    if execution_profile is not None:
        expected |= set(EXECUTION_PROFILE_IDENTITY_FIELDS)
    if context['suspended_enabled']: expected |= SUSPENDED
    if ADAPTATION & set(archive):
        require(context.get('adaptation', False) and ADAPTATION <= set(archive),
                'unexpected or incomplete adaptation diagnostics')
        expected |= ADAPTATION
    if BEDFORM & set(archive):
        require(context.get('bedform', False) and BEDFORM <= set(archive),
                'unexpected or incomplete bedform diagnostics')
        expected |= BEDFORM
    if context.get('gaia_diagnostic', False):
        expected |= GAIA_DIAGNOSTICS
    require(set(archive) == expected and
            int(_scalar(archive, 'transport_checkpoint_schema', 'int32')) ==
                (int(context.get('checkpoint_schema', 2)) if implicit else 1),
            'compact physical field inventory differs')
    if execution_profile is not None:
        for key, value in execution_profile['identities'].items():
            actual=np.asarray(archive[key])
            expected_value=np.asarray(value)
            require(actual.shape==expected_value.shape and
                    actual.dtype.kind=='U' and expected_value.dtype.kind=='U' and
                    np.array_equal(actual,expected_value),
                    'compact execution-profile identity differs: '+key)
    expected_digest = _identity(_string_scalar(archive, 'transport_payload_sha256'), 'transport payload')
    digest = hashlib.sha256()
    for key in sorted(expected - {'transport_payload_sha256'}):
        value = np.ascontiguousarray(archive[key])
        digest.update(key.encode()+b'\0'+value.dtype.str.encode()+b'\0')
        digest.update(json.dumps(value.shape).encode()+b'\0'+value.tobytes())
    require(digest.hexdigest() == expected_digest, 'compact payload differs')
    q, layers = (archive[key] for key in ('q', 'layers'))
    if implicit and checkpoint:
        predictor = archive['implicit_predictor_rate']
        have_predictor = archive['implicit_have_predictor']
        require(predictor.shape == (3, context['nc']) and
                predictor.dtype == np.float64 and np.isfinite(predictor).all() and
                have_predictor.shape == () and have_predictor.dtype == np.int32 and
                int(have_predictor) in (0, 1),
                'implicit predictor checkpoint differs')
    if implicit:
        _validate_implicit_q(context, q)
    suspended = archive['suspended'] if context['suspended_enabled'] else np.empty(0)
    require(layers.shape[:2] == (context['nlayer'], context['nfrac']) and
            archive['ledger_water'].shape == (context['nb'],), 'material/boundary layout differs')
    dry = q[0] <= context['dry_tolerance']
    # Exchange columns are SIGNED gains in suspended/active reservoirs, not
    # positive erosion/deposition totals. Their conservation is kernel-checked.
    require(np.all(q[0] >= 0.) and np.all(layers >= 0.) and np.all(suspended >= 0.) and
            not np.any(q[1:, dry]) and not np.any(archive['secondary'][dry]),
            'negative state or dry momentum')
    if ADAPTATION & set(archive):
        require(int(_scalar(archive, 'bedload_adaptation_iterations', 'int32')) >= 0 and
                float(_scalar(archive, 'bedload_adaptation_residual', 'float64')) >= 0.,
                'invalid adaptation diagnostics')
    if BEDFORM & set(archive):
        require(all(np.all(archive[key] >= 0) for key in BEDFORM), 'negative bedform diagnostics')
    gaia = None
    if context.get('gaia_diagnostic', False):
        schema = archive['gaia_secondary_diagnostic_schema']
        values = archive['gaia_secondary_diagnostic']
        require(schema.shape == () and schema.dtype == np.int32 and int(schema) == 1 and
                values.shape == (len(GAIA_DIAGNOSTIC_FIELDS), context['nfrac'], q.shape[1]) and
                values.dtype == np.float64 and np.isfinite(values).all(),
                'invalid GAIA secondary-stress diagnostic layout')
        active = values[0]
        enabled = active.astype(bool)
        require(np.isin(active, (0., 1.)).all() and np.all(values[5] >= 0.) and
                np.all(values[5][enabled] > 0.) and
                np.all(np.abs(values[12:14]) <= np.pi + 32*np.finfo(float).eps),
                'invalid GAIA secondary-stress diagnostic values')
        tolerance = 512*np.finfo(float).eps*np.maximum(1., np.maximum(
            np.abs(values[10:12]), np.abs(values[6:8])))
        require(np.all(np.abs(values[10:12]-(values[6:8]+values[8:10])) <= tolerance),
                'GAIA stress decomposition differs')
        if np.any(enabled):
            qmag = np.hypot(values[14], values[15])
            expected_x = np.zeros_like(qmag)
            expected_y = np.zeros_like(qmag)
            expected_x[enabled] = qmag[enabled]*values[10][enabled]/values[5][enabled]
            expected_y[enabled] = qmag[enabled]*values[11][enabled]/values[5][enabled]
            scale = np.maximum(1.e-20, np.maximum.reduce((np.abs(expected_x),
                np.abs(expected_y), np.abs(values[16]), np.abs(values[17]))))
            require(np.all(np.abs(expected_x[enabled]-values[16][enabled]) <=
                           512*np.finfo(float).eps*scale[enabled]) and
                    np.all(np.abs(expected_y[enabled]-values[17][enabled]) <=
                           512*np.finfo(float).eps*scale[enabled]),
                    'GAIA diagnostic capacity/stress relation differs')
        gaia=dict(schema=1,fields=GAIA_DIAGNOSTIC_FIELDS,values=values.copy())
    if context['secondary_model'] != 2:
        require(not any(np.any(archive[key]) for key in ('secondary', 'ledger_secondary', 'ledger_source')),
                'nonprognostic secondary state/ledgers must be zero')
    if context['bedlaw'] == 0: require(not np.any(archive['ledger_bed']), 'no-bedload ledger must be zero')
    bed = context['rock'] + (layers/context['bulk'][None, :, None]).sum(axis=(0, 1))
    require(np.allclose(archive['bed_elevation'], bed, rtol=0., atol=1.e-8), 'bed elevation differs from layers')
    return gaia
