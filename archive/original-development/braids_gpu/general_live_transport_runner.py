"""Ordinary fixed-grid physical-process driver for the shared GPU lifecycle."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from braids_core.sediment_method_catalog import (
    CATALOG_SCHEMA, CATALOG_SCOPE_ID, semantic_catalog_id)
from braids_core.sediment.execution_profile_v2 import (
    BACKEND_GPU, bind_component_profile, execution_profile_document,
    load_execution_profile)
from braids_core.sediment.fixed_grid_profile_completeness import (
    validate_fixed_grid_profile_completeness)
from braids_core.sediment.hiding_foundation_execution import (
    EXECUTION_KEY as HIDING_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as HIDING_ORDERED_ROUTE_KEY,
    validate_hiding_foundation_execution_document,
)
from braids_core.sediment.secondary_foundation_execution import (
    EXECUTION_KEY as SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as SECONDARY_ORDERED_ROUTE_KEY,
    validate_secondary_foundation_execution_document,
)
from braids_core.sediment.prognostic_secondary_execution import (
    EXECUTION_KEY as PROGNOSTIC_SECONDARY_EXECUTION_KEY,
    ORDERED_ROUTE_KEY as PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY,
    validate_prognostic_secondary_execution_document,
)

from .atomic_output import save_arrays_atomic as save_arrays, write_json_atomic as write_json
from .general_live_bank_runner import (require, sha, digest, authenticated_member,
    packaged_native_exporter, ADAPTATION_DIAGNOSTICS)
from .graded_native_exporter import EXPORTER_SHA256, REQUIRED_DLLS, export_environment
from .delft_components_exporter import (EXPORTER_SHA256 as DELFT_EXPORTER_SHA256,
    EXPORTER_GENERATIONS as DELFT_EXPORTER_GENERATIONS,
    GENERATION as DELFT_EXPORTER_GENERATION, EXPORTER_NAME as DELFT_EXPORTER_NAME,
    EXPORTER_V17_SHA256 as Q3H2_EXPORTER_SHA256,
    EXPORTER_V18_GENERATION as Q3H3_EXPORTER_GENERATION,
    EXPORTER_V18_SHA256 as Q3H3_EXPORTER_SHA256,
    HISTORICAL_EXPORTER_V13_GENERATION, HISTORICAL_EXPORTER_V13_SHA256,
    verify_source_exporter)

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'live_normal_critical_transport_ssp33'
ARTIFACTS = frozenset(('coupled_input.bin', 'coupled_input.npz', 'normal_metadata.bin'))
FORCING_ARTIFACT = 'delft_forcing_binding.json'
EXECUTION_PROFILE_ARTIFACT = 'execution_profile_v2.json'
EXECUTION_PROFILE_SIDECAR_SCHEMA = 1
EXECUTION_PROFILE_ADMISSIONS = frozenset((
    'runtime', 'validation_internal_neutral'))
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
INTEGRATED_EXPORTER_SHA256 = '44b50de40f6d7851957f435a76a16f0995947012bd1ecfd7fcd8776c3915e226'
EH_EXPORTER_SHA256 = '898f467518789453ee2be202095eef2a81725c4e262f0c1c22b0625be56970fd'
GAIA_EXPORTER_V4_SHA256 = '90eeb6a5740ba0fa7aee3613855380e6185724209a636d4603ff77a5db7c203f'
GAIA_EXPORTER_V5_SHA256 = '970dc63b032a08de59ca18506c9eb804c7c0d3efdd646f698000dbf981018eb5'
GAIA_EXPORTER_V6_SHA256 = '1c1c8503b7e297e6df07fa53685c41635937ea982254f6719bfe205d253be838'
# The unqualified name is the only GAIA initializer used for new preparation.
GAIA_EXPORTER_SHA256 = GAIA_EXPORTER_V6_SHA256
GAIA_EXPORTER_GENERATIONS = {
    GAIA_EXPORTER_V4_SHA256: 'native_exporter_v4',
    GAIA_EXPORTER_V5_SHA256: 'native_exporter_v5',
    GAIA_EXPORTER_V6_SHA256: 'native_exporter_v6',
}
EXPORTERS = frozenset((EXPORTER_SHA256, INTEGRATED_EXPORTER_SHA256,
                       EH_EXPORTER_SHA256, *DELFT_EXPORTER_GENERATIONS,
                       *GAIA_EXPORTER_GENERATIONS))
DELFT_FORCING_EXPORTERS = frozenset(DELFT_EXPORTER_GENERATIONS)
TYPED_ROUGHNESS_METHODS = frozenset((
    'roughness.chezy', 'roughness.manning', 'roughness.white_colebrook',
    'roughness.white_colebrook_waqua'))
Q3C_EXPORTER_GENERATION = 'native_exporter_v14'
Q3H1_EXPORTER_GENERATION = 'native_exporter_v15'
Q3H2_EXPORTER_GENERATION = 'native_exporter_v17'
OPTIONAL_ARTIFACTS = frozenset((
    'coupled_diffusion.bin', 'avalanche_metadata.bin', FORCING_ARTIFACT,
    EXECUTION_PROFILE_ARTIFACT))
BEDFORM_DIAGNOSTICS = frozenset(('bedform_effective_manning', 'bedform_dune_height',
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


def gaia_diagnostic_method(data):
    if int(data['bedlaw']) != 9:
        return None
    return dict(schema=1, fields=list(GAIA_DIAGNOSTIC_FIELDS),
        clock='accepted_output_frame', eta_gradient='grad(z+h)',
        bed_gradient='grad(z)_positive_uphill',
        decomposition='slope_stress+secondary_delta=total_stress',
        angle='signed_counter_clockwise_radians_minus_pi_to_pi',
        capacity='cell_centred_equilibrium_not_accepted_face_flux',
        checkpointed=False)


def artifact_names(data):
    return (ARTIFACTS | ({'coupled_diffusion.bin'} if float(data['horizontal_viscosity']) > 0 else set()) |
            ({'avalanche_metadata.bin'} if bool(data['general_avalanche']) else set()) |
            ({FORCING_ARTIFACT} if 'delft_forcing_binding' in data else set()) |
            ({EXECUTION_PROFILE_ARTIFACT}
             if _EXECUTION_PROFILE_DATA_KEY in data else set()))


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _strict_json_object(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result,
                    'Duplicate execution-profile sidecar key: '+str(key))
            result[key] = value
        return result

    try:
        value = json.loads(Path(path).read_bytes().decode('ascii'),
            object_pairs_hook=object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError('Non-finite execution-profile JSON: '+token)))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError('Invalid execution-profile sidecar JSON') from error
    require(isinstance(value, dict),
            'Execution-profile sidecar must be an object')
    return value


def _component_rows(profile):
    return [dict(component_id=row.component_id, schema=row.schema,
        scope_id=row.scope_id, semantic_id=row.semantic_id,
        fingerprint=row.fingerprint, document_sha256=row.document_sha256)
        for row in profile.component_profiles]


def _v55_hiding_identity(metadata, deck_lines, *, nested=False):
    """Authenticate the additive H1 document and its intact v54 parent.

    The first v55 payload is canonical JSON encoded as bounded hexadecimal
    chunks.  The very next native envelope must be v54; v55 never substitutes
    for or reinterprets that frozen Q3-C foundation.
    """
    raw = metadata.get('hiding_foundation_identity')
    require(isinstance(raw, dict) and set(raw) == {
        'envelope_version', 'canonical_preimage_ascii',
        'canonical_preimage_sha256', HIDING_EXECUTION_KEY,
        HIDING_ORDERED_ROUTE_KEY},
        'Transport v55 hiding-foundation metadata differs')
    identity = validate_hiding_foundation_execution_document(
        raw[HIDING_EXECUTION_KEY])
    preimage = raw['canonical_preimage_ascii']
    require(raw['envelope_version'] == 55 and isinstance(preimage, str) and
            preimage.isascii() and _canonical_json(identity) == preimage and
            hashlib.sha256(preimage.encode('ascii')).hexdigest() ==
                raw['canonical_preimage_sha256'],
            'Transport v55 hiding-foundation preimage differs')
    route = raw[HIDING_ORDERED_ROUTE_KEY]
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
    require(route == expected_route and isinstance(base, dict) and
            base.get('envelope_version') == 54 and
            base.get('capacity_execution_fingerprint') ==
                identity['capacity_execution_fingerprint'] and
            base.get('capacity_profile_fingerprint') ==
                identity['capacity_profile_fingerprint'] and
            metadata.get('bedload_law') == identity['capacity_law'],
            'Transport v55 ordered hiding/Q3-C route differs')

    header_index = 1 if nested else 2
    require(len(deck_lines) > header_index + 1 and
            deck_lines[0].strip() == '55',
            'Transport v55 deck envelope is incomplete')
    try:
        schema, byte_count, chunk_count = (
            int(token) for token in deck_lines[header_index].split())
    except (TypeError, ValueError) as error:
        raise ValueError('Transport v55 hiding header differs') from error
    require(schema == 1 and byte_count == len(preimage) and chunk_count > 0 and
            len(deck_lines) > header_index + 1 + chunk_count,
            'Transport v55 hiding header differs')
    chunks = deck_lines[header_index + 1:header_index + 1 + chunk_count]
    require(all(re.fullmatch(r'[0-9a-f]+', chunk) is not None
                for chunk in chunks),
            'Transport v55 hiding payload is not lowercase hexadecimal')
    try:
        decoded = bytes.fromhex(''.join(chunks)).decode('ascii')
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError('Transport v55 hiding payload is invalid') from error
    require(len(decoded) == byte_count and decoded == preimage and
            deck_lines[header_index + 1 + chunk_count].strip() == '54',
            'Transport v55 does not wrap its exact v54 foundation')
    return identity, expected_route


def _v56_secondary_identity(metadata, deck_lines):
    """Authenticate v56 and retain its v54/v55 parent as an opaque base."""
    raw=metadata.get('secondary_foundation_identity')
    require(isinstance(raw,dict) and set(raw)=={
        'envelope_version','canonical_preimage_ascii',
        'canonical_preimage_sha256',SECONDARY_EXECUTION_KEY,
        SECONDARY_ORDERED_ROUTE_KEY},
        'Transport v56 secondary-foundation metadata differs')
    identity=validate_secondary_foundation_execution_document(
        raw[SECONDARY_EXECUTION_KEY])
    preimage=raw['canonical_preimage_ascii']
    require(raw['envelope_version']==56 and isinstance(preimage,str) and
            preimage.isascii() and _canonical_json(identity)==preimage and
            hashlib.sha256(preimage.encode('ascii')).hexdigest()==
                raw['canonical_preimage_sha256'],
            'Transport v56 secondary-foundation preimage differs')
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
            'Transport v56 ordered secondary/foundation route differs')
    require(len(deck_lines)>=4 and deck_lines[0].strip()=='56',
            'Transport v56 deck envelope is incomplete')
    try:
        schema,byte_count,chunk_count=(int(token)
                                      for token in deck_lines[2].split())
    except (TypeError,ValueError) as error:
        raise ValueError('Transport v56 secondary header differs') from error
    require(schema==1 and byte_count==len(preimage) and chunk_count>0 and
            len(deck_lines)>3+chunk_count,
            'Transport v56 secondary header differs')
    chunks=deck_lines[3:3+chunk_count]
    require(all(re.fullmatch(r'[0-9a-f]+',chunk) is not None and
                len(chunk)%2==0 for chunk in chunks),
            'Transport v56 secondary payload is not lowercase hexadecimal')
    try:decoded=bytes.fromhex(''.join(chunks)).decode('ascii')
    except (UnicodeDecodeError,ValueError) as error:
        raise ValueError('Transport v56 secondary payload is invalid') from error
    base_index=3+chunk_count
    require(len(decoded)==byte_count and decoded==preimage and
            deck_lines[base_index].strip()==str(identity['base_envelope_version']),
            'Transport v56 does not wrap its declared v54/v55 foundation')
    hiding_identity=hiding_route=None
    if identity['base_envelope_version']==55:
        hiding_identity,hiding_route=_v55_hiding_identity(
            metadata,deck_lines[base_index:],nested=True)
        require(hiding_identity['fingerprint']==
                    identity['hiding_execution_fingerprint'],
                'Transport v56 hiding fingerprint differs from nested v55')
    else:
        require(metadata.get('hiding_foundation_identity') is None and
                identity['hiding_execution_fingerprint'] is None,
                'Transport v56/v54 foundation unexpectedly carries hiding')
    return identity,route,hiding_identity,hiding_route


def _v57_prognostic_secondary_identity(metadata, deck_lines):
    """Authenticate v57 directly over its intact v54/v55 foundation."""
    raw = metadata.get('prognostic_secondary_identity')
    require(isinstance(raw, dict) and set(raw) == {
        'envelope_version', 'canonical_preimage_ascii',
        'canonical_preimage_sha256',
        'prognostic_secondary_execution_fingerprint',
        'prognostic_secondary_bed_boundary_rotation_policy',
        'prognostic_secondary_state_units',
        'prognostic_secondary_integrated_ledger_units',
        PROGNOSTIC_SECONDARY_EXECUTION_KEY,
        PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY},
        'Transport v57 prognostic-secondary metadata differs')
    identity = validate_prognostic_secondary_execution_document(
        raw[PROGNOSTIC_SECONDARY_EXECUTION_KEY])
    preimage = raw['canonical_preimage_ascii']
    require(raw['envelope_version'] == 57 and isinstance(preimage, str) and
            preimage.isascii() and _canonical_json(identity) == preimage and
            raw['prognostic_secondary_execution_fingerprint'] ==
                identity['fingerprint'] and
            raw['prognostic_secondary_bed_boundary_rotation_policy'] ==
                identity['bed_boundary_rotation_policy'] and
            raw['prognostic_secondary_state_units'] == 'm2_s' and
            raw['prognostic_secondary_integrated_ledger_units'] ==
                identity['integrated_ledger_units'] and
            hashlib.sha256(preimage.encode('ascii')).hexdigest() ==
                raw['canonical_preimage_sha256'],
            'Transport v57 prognostic-secondary preimage differs')
    route = {
        'capacity_execution_fingerprint':
            identity['capacity_execution_fingerprint'],
        'capacity_profile_fingerprint':
            identity['capacity_profile_fingerprint'],
        'hiding_execution_fingerprint':
            identity['hiding_execution_fingerprint'],
        'prognostic_secondary_execution_fingerprint':
            identity['fingerprint'],
    }
    base = metadata.get('capacity_route_identity')
    require(raw[PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY] == route and
            metadata.get('secondary_foundation_identity') is None and
            isinstance(base, dict) and base.get('envelope_version') == 54 and
            base.get('capacity_execution_fingerprint') ==
                identity['capacity_execution_fingerprint'] and
            base.get('capacity_profile_fingerprint') ==
                identity['capacity_profile_fingerprint'] and
            metadata.get('bedload_law') == identity['capacity_law'],
            'Transport v57 ordered prognostic/foundation route differs')
    require(len(deck_lines) >= 4 and deck_lines[0].strip() == '57',
            'Transport v57 deck envelope is incomplete')
    try:
        schema, byte_count, chunk_count = (
            int(token) for token in deck_lines[2].split())
    except (TypeError, ValueError) as error:
        raise ValueError(
            'Transport v57 prognostic-secondary header differs') from error
    require(schema == 1 and byte_count == len(preimage) and chunk_count > 0 and
            len(deck_lines) > 3 + chunk_count,
            'Transport v57 prognostic-secondary header differs')
    chunks = deck_lines[3:3 + chunk_count]
    require(all(re.fullmatch(r'[0-9a-f]+', chunk) is not None and
                len(chunk) % 2 == 0 for chunk in chunks),
            'Transport v57 prognostic-secondary payload is not lowercase hexadecimal')
    try:
        decoded = bytes.fromhex(''.join(chunks)).decode('ascii')
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            'Transport v57 prognostic-secondary payload is invalid') from error
    base_index = 3 + chunk_count
    require(len(decoded) == byte_count and decoded == preimage and
            deck_lines[base_index].strip() ==
                str(identity['base_envelope_version']),
            'Transport v57 does not directly wrap its declared v54/v55 foundation')
    hiding_identity = hiding_route = None
    if identity['base_envelope_version'] == 55:
        hiding_identity, hiding_route = _v55_hiding_identity(
            metadata, deck_lines[base_index:], nested=True)
        require(hiding_identity['fingerprint'] ==
                    identity['hiding_execution_fingerprint'],
                'Transport v57 hiding fingerprint differs from nested v55')
    else:
        require(metadata.get('hiding_foundation_identity') is None and
                identity['hiding_execution_fingerprint'] is None,
                'Transport v57/v54 foundation unexpectedly carries hiding')
    return identity, route, hiding_identity, hiding_route


def _install_inline_hiding_identity(data, inline):
    """Install only an already authenticated outer identity into host data."""
    identity = inline.get('hiding_execution_identity')
    route = inline.get('hiding_ordered_route_identity')
    if identity is None:
        require(route is None and int(np.asarray(data['hiding_exposure'])) == 0,
                'Transport v53/v54 cannot carry generic hiding')
        return
    identity = validate_hiding_foundation_execution_document(identity)
    require(route == {
        'ordered_parts': [
            'capacity_execution_fingerprint',
            'capacity_profile_fingerprint',
            'hiding_execution_fingerprint'],
        'capacity_execution_fingerprint': identity[
            'capacity_execution_fingerprint'],
        'capacity_profile_fingerprint': identity[
            'capacity_profile_fingerprint'],
        'hiding_execution_fingerprint': identity['fingerprint'],
    } and int(np.asarray(data['hiding_exposure'])) == identity['mode'] and
            float(np.asarray(data['hiding_parker_alpha'])) ==
                identity['parker_alpha'] and
            float(np.asarray(data['hiding_wu_exponent'])) ==
                identity['wu_exponent'] and
            int(np.asarray(data['bedlaw'])) == identity['capacity_law'],
            'Transport v55 hiding identity differs from native controls')
    data[HIDING_EXECUTION_KEY] = identity
    data[HIDING_ORDERED_ROUTE_KEY] = route


def _install_inline_secondary_identity(data,inline):
    identity=inline.get('secondary_execution_identity')
    route=inline.get('secondary_ordered_route_identity')
    if identity is None:
        require(route is None,'Transport v53/v54/v55 acquired Q3-H2 secondary')
        return
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
            'Transport v56 secondary identity differs from nested controls')
    # The nested v54/v55 native deck deliberately keeps its frozen
    # ``secondary.off`` controls.  Only after authenticating the v56 outer
    # envelope do we materialize the effective resident-GPU controls; this is
    # not a legacy native source activation or a fallback.
    data.update(secondary_flow_model=1,secondary_flow=1,
                secondary_flow_transport_direction=1,
                secondary_flow_momentum_beta=0.,
                secondary_flow_espir=identity['espir'])
    data[SECONDARY_EXECUTION_KEY]=identity
    data[SECONDARY_ORDERED_ROUTE_KEY]=route


def _install_inline_prognostic_secondary_identity(data, inline):
    """Attach only an already authenticated v57 identity to native state."""
    identity = inline.get('prognostic_secondary_execution_identity')
    route = inline.get('prognostic_secondary_ordered_route_identity')
    if identity is None:
        require(route is None,
                'Transport v53/v54/v55/v56 acquired Q3-H3 secondary')
        return
    identity = validate_prognostic_secondary_execution_document(identity)
    expected = {
        'capacity_execution_fingerprint':
            identity['capacity_execution_fingerprint'],
        'capacity_profile_fingerprint':
            identity['capacity_profile_fingerprint'],
        'hiding_execution_fingerprint':
            identity['hiding_execution_fingerprint'],
        'prognostic_secondary_execution_fingerprint':
            identity['fingerprint'],
    }
    require(route == expected and
            identity['method_id'] == 'secondary.prognostic' and
            identity['model'] == 2 and
            identity['capacity_law'] == int(data['bedlaw']) and
            identity['roughness_method_id'] ==
                data.get('capacity_roughness_method_id') and
            identity['roughness_fingerprint'] ==
                data.get('delft_forcing_fingerprint') and
            identity['coriolis_disposition'] == 'zero' and
            identity['coriolis_constant_s_inverse'] == 0. and
            identity['coriolis_source'] == 'constant_zero_planar' and
            identity['betac'] == 0. and
            identity['scalar_diffusivity_m2_s'] == 0. and
            int(data.get('secondary_flow_model', -1)) == 2 and
            int(data.get('secondary_flow', 0)) == 1 and
            int(data.get('secondary_flow_transport_direction', 0)) == 1 and
            int(data.get('secondary_flow_inflow_mode', -1)) == 1 and
            float(data.get('secondary_flow_momentum_beta', np.nan)) == 0. and
            float(data.get('secondary_flow_espir', np.nan)) ==
                identity['espir'],
            'Transport v57 prognostic-secondary identity differs from native controls')
    data[PROGNOSTIC_SECONDARY_EXECUTION_KEY] = identity
    data[PROGNOSTIC_SECONDARY_ORDERED_ROUTE_KEY] = route


def _native_identity_ascii(data,key):
    value=np.asarray(data.get(key))
    require(value.shape==(64,) and value.dtype.kind in 'iu' and
            np.all((value>=32)&(value<=126)),
            'Transport native Q3-H2 identity record differs: '+key)
    return bytes(int(item) for item in value).decode('ascii').rstrip()


def _validate_native_v56_secondary(data,identity):
    scalar=lambda key:np.asarray(data.get(key))
    for key in ('native_q3h2_secondary_verified',
                'native_q3h2_secondary_envelope','native_q3h2_model'):
        value=scalar(key)
        require(value.shape==() and value.dtype.kind in 'iu',
                'Transport native Q3-H2 scalar differs: '+key)
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
            'Transport native v56 secondary identity differs')


def _native_q3h3_ascii(data, key):
    value = np.asarray(data.get(key))
    require(value.shape == (64,) and value.dtype.kind in 'iu' and
            np.all((value >= 32) & (value <= 126)),
            'Transport native Q3-H3 identity record differs: ' + key)
    return bytes(int(item) for item in value).decode('ascii').rstrip()


def _validate_native_v57_prognostic_secondary(data, identity):
    scalar = lambda key: np.asarray(data.get(key))
    integer_keys = (
        'native_q3h3_secondary_verified',
        'native_q3h3_secondary_envelope', 'native_q3h3_model',
        'native_q3h3_base_envelope', 'native_q3h3_transport_direction',
        'native_q3h3_inflow_mode')
    real_keys = (
        'native_q3h3_rmincf', 'native_q3h3_chzmin',
        'native_q3h3_espir', 'native_q3h3_coriolis_constant_s_inverse',
        'native_q3h3_betac', 'native_q3h3_scalar_diffusivity_m2_s',
        'native_q3h3_von_karman_constant')
    for key in integer_keys:
        value = scalar(key)
        require(value.shape == () and value.dtype.kind in 'iu',
                'Transport native Q3-H3 scalar differs: ' + key)
    for key in real_keys:
        value = scalar(key)
        require(value.shape == () and value.dtype.kind == 'f' and
                np.isfinite(value),
                'Transport native Q3-H3 scalar differs: ' + key)
    hiding = identity['hiding_execution_fingerprint'] or ''
    require(
        int(scalar('native_q3h3_secondary_verified')) == 1 and
        int(scalar('native_q3h3_secondary_envelope')) == 57 and
        int(scalar('native_q3h3_model')) == 2 and
        int(scalar('native_q3h3_base_envelope')) ==
            identity['base_envelope_version'] and
        _native_q3h3_ascii(
            data, 'native_q3h3_execution_fingerprint_ascii') ==
                identity['fingerprint'] and
        _native_q3h3_ascii(
            data, 'native_q3h3_capacity_execution_fingerprint_ascii') ==
                identity['capacity_execution_fingerprint'] and
        _native_q3h3_ascii(
            data, 'native_q3h3_capacity_profile_fingerprint_ascii') ==
                identity['capacity_profile_fingerprint'] and
        _native_q3h3_ascii(
            data, 'native_q3h3_hiding_execution_fingerprint_ascii') ==
                hiding and
        _native_q3h3_ascii(
            data, 'native_q3h3_roughness_fingerprint_ascii') ==
                identity['roughness_fingerprint'] and
        _native_q3h3_ascii(data, 'native_q3h3_method_id_ascii') ==
                identity['method_id'] and
        _native_q3h3_ascii(
            data, 'native_q3h3_canonical_preimage_sha256_ascii') ==
                hashlib.sha256(_canonical_json(identity).encode(
                    'ascii')).hexdigest() and
        float(scalar('native_q3h3_rmincf')) == identity['rmincf'] and
        float(scalar('native_q3h3_chzmin')) == identity['chzmin'] and
        float(scalar('native_q3h3_espir')) == identity['espir'] and
        float(scalar('native_q3h3_coriolis_constant_s_inverse')) ==
            identity['coriolis_constant_s_inverse'] and
        float(scalar('native_q3h3_betac')) == identity['betac'] and
        float(scalar('native_q3h3_scalar_diffusivity_m2_s')) ==
            identity['scalar_diffusivity_m2_s'] and
        float(scalar('native_q3h3_von_karman_constant')) ==
            identity['von_karman_constant'] and
        int(scalar('native_q3h3_transport_direction')) == 1 and
        int(scalar('native_q3h3_inflow_mode')) == identity['inflow_mode'],
        'Transport native v57 prognostic-secondary identity differs')


def _v53_inline_identity(metadata, header, exporter):
    """Authenticate v53/v54 and additive v55/v56/v57 inline preimages."""
    generation = DELFT_EXPORTER_GENERATIONS.get(exporter)
    bedlaw = metadata.get('bedload_law')
    outer_versions = {
        HISTORICAL_EXPORTER_V13_GENERATION: 53,
        Q3C_EXPORTER_GENERATION: 54,
        Q3H1_EXPORTER_GENERATION: 55,
        Q3H2_EXPORTER_GENERATION: 56,
        Q3H3_EXPORTER_GENERATION: 57,
    }
    outer_version = outer_versions.get(generation)
    supported_law = (
        bedlaw == 11 if generation == 'native_exporter_v13'
        else bedlaw in (2, 8, 11) if generation == Q3C_EXPORTER_GENERATION
        else bedlaw in (2, 11) if generation == Q3H1_EXPORTER_GENERATION
        else bedlaw in (2, 8, 11) if generation == Q3H2_EXPORTER_GENERATION
        else bedlaw in (2, 8, 11) if generation == Q3H3_EXPORTER_GENERATION
        else False
    )
    route = metadata.get('capacity_route_identity')
    require(header[0].strip() == str(outer_version) and
            generation is not None and supported_law and
            metadata.get('format') == f'xbraids_cutcell-v{outer_version}' and
            (generation == HISTORICAL_EXPORTER_V13_GENERATION or
             isinstance(route, dict) and route.get('envelope_version') == 54) and
            metadata.get('fingerprint') == header[1].strip() and
            metadata.get('sidecars', {}).get('native_binary') is None and
            metadata.get('sidecars', {}).get('geometry') == 'geometry.json',
            'Transport v53 inline geometry identity differs')
    execution = metadata.get('execution_profile_v2')
    require(isinstance(execution, dict) and set(execution) == {
        'envelope_version', 'runtime_backend', 'profile_backend', 'schema',
        'semantic_id', 'fingerprint', 'canonical_preimage_sha256',
        'canonical_preimage_ascii', 'profile', 'component_documents'},
        'Transport v53 execution-profile metadata differs')
    preimage = execution['canonical_preimage_ascii']
    require(execution['envelope_version'] == 53 and
            execution['runtime_backend'] == BACKEND_GPU and
            execution['profile_backend'] ==
                'backend_neutral_physical_profile' and
            execution['schema'] == 2 and isinstance(preimage, str) and
            preimage.isascii() and
            hashlib.sha256(preimage.encode('ascii')).hexdigest() ==
                execution['canonical_preimage_sha256'],
            'Transport v53 execution-profile preimage differs')
    try:
        preimage_document = json.loads(preimage)
        neutral = load_execution_profile(execution['profile'])
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            'Transport v53 execution-profile document is invalid') from error
    profile_payload = dict(execution['profile'])
    profile_fingerprint = profile_payload.pop('fingerprint', None)
    require(preimage_document == profile_payload and
            _canonical_json(preimage_document) == preimage and
            execution_profile_document(neutral) == execution['profile'] and
            neutral.backend is None and
            hashlib.sha256(preimage.encode('ascii')).hexdigest() ==
                profile_fingerprint and
            neutral.fingerprint == execution['fingerprint'] and
            neutral.semantic_id == execution['semantic_id'],
            'Transport v53 execution-profile document differs')
    roughness = {row.entry_id for row in neutral.method_selections
                 if row.entry_id in TYPED_ROUGHNESS_METHODS}
    require((generation == HISTORICAL_EXPORTER_V13_GENERATION and
             not roughness) or
            (generation in (Q3C_EXPORTER_GENERATION,
                            Q3H1_EXPORTER_GENERATION,
                            Q3H2_EXPORTER_GENERATION,
                            Q3H3_EXPORTER_GENERATION) and
             len(roughness) == 1),
            'Transport v53 profile/exporter generation differs')
    receipts = execution['component_documents']
    expected = {row.component_id: row for row in neutral.component_profiles}
    require(isinstance(receipts, dict) and set(receipts) == set(expected),
            'Transport v53 component-document inventory differs')
    documents = {}
    receipt_keys = {
        'canonical_preimage_ascii', 'document_sha256', 'fingerprint',
        'schema', 'scope_id', 'semantic_id'}
    for component_id in sorted(receipts):
        receipt = receipts[component_id]
        require(isinstance(receipt, dict) and set(receipt) == receipt_keys,
                'Transport v53 component receipt differs: '+component_id)
        component_preimage = receipt['canonical_preimage_ascii']
        require(isinstance(component_preimage, str) and
                component_preimage.isascii() and
                hashlib.sha256(component_preimage.encode('ascii')).hexdigest() ==
                    receipt['document_sha256'],
                'Transport v53 component preimage differs: '+component_id)
        try:
            document = json.loads(component_preimage)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                'Transport v53 component document is invalid: '+component_id
            ) from error
        binding = bind_component_profile(component_id, document)
        require(_canonical_json(document) == component_preimage and
                binding == expected[component_id] and
                receipt == dict(
                    canonical_preimage_ascii=component_preimage,
                    document_sha256=binding.document_sha256,
                    fingerprint=binding.fingerprint, schema=binding.schema,
                    scope_id=binding.scope_id,
                    semantic_id=binding.semantic_id),
                'Transport v53 component document differs: '+component_id)
        documents[component_id] = document
    validate_fixed_grid_profile_completeness(
        neutral, component_documents=documents)
    hiding_identity = None
    hiding_route = None
    secondary_identity = None
    secondary_route = None
    prognostic_secondary_identity = None
    prognostic_secondary_route = None
    if outer_version == 55:
        hiding_identity, hiding_route = _v55_hiding_identity(metadata, header)
        require(metadata.get('secondary_foundation_identity') is None,
                'Transport v55 cannot acquire an outer secondary identity')
        require(metadata.get('prognostic_secondary_identity') is None,
                'Transport v55 cannot acquire an outer prognostic identity')
    elif outer_version == 56:
        (secondary_identity,secondary_route,hiding_identity,hiding_route)=\
            _v56_secondary_identity(metadata,header)
        require(metadata.get('prognostic_secondary_identity') is None,
                'Transport v56 cannot acquire an outer prognostic identity')
    elif outer_version == 57:
        (prognostic_secondary_identity, prognostic_secondary_route,
         hiding_identity, hiding_route) = \
            _v57_prognostic_secondary_identity(metadata, header)
    else:
        require(metadata.get('hiding_foundation_identity') is None,
                'Transport v53/v54 cannot acquire an outer hiding identity')
        require(metadata.get('secondary_foundation_identity') is None,
                'Transport v53/v54 cannot acquire an outer secondary identity')
        require(metadata.get('prognostic_secondary_identity') is None,
                'Transport v53/v54 cannot acquire an outer prognostic identity')
    return dict(
        profile=neutral, component_documents=documents,
        hiding_execution_identity=hiding_identity,
        hiding_ordered_route_identity=hiding_route,
        secondary_execution_identity=secondary_identity,
        secondary_ordered_route_identity=secondary_route,
        prognostic_secondary_execution_identity=
            prognostic_secondary_identity,
        prognostic_secondary_ordered_route_identity=
            prognostic_secondary_route)


def _select_delft_exporter(metadata, header):
    """Select the pinned zero-step exporter from the exact outer wire."""
    execution = metadata.get('execution_profile_v2')
    profile = execution.get('profile') if isinstance(execution, dict) else None
    selections = (profile.get('method_selections')
                  if isinstance(profile, dict) else None)
    version = header[0].strip()
    require(version in ('53', '54', '55', '56', '57') and
            isinstance(selections, list),
            'Delft component preparation requires an authenticated '
            'v53/v54/v55/v56/v57 profile')
    roughness = {
        row.get('entry_id') for row in selections
        if isinstance(row, dict) and row.get('entry_id') in
            TYPED_ROUGHNESS_METHODS}
    by_generation = {
        generation: identity
        for identity, generation in DELFT_EXPORTER_GENERATIONS.items()}
    if not roughness:
        require(version == '53',
                'Historical Q3-S/P preparation requires its frozen v53 wire')
        exporter = HISTORICAL_EXPORTER_V13_SHA256
    elif len(roughness) == 1:
        generation = {
            '54': Q3C_EXPORTER_GENERATION,
            '55': Q3H1_EXPORTER_GENERATION,
            '56': Q3H2_EXPORTER_GENERATION,
            '57': Q3H3_EXPORTER_GENERATION,
        }.get(version)
        require(generation in by_generation,
                'Typed Q3-C/Q3-H1 wire has no pinned zero-step exporter')
        exporter = by_generation[generation]
    else:
        raise ValueError(
            'Transport v53 profile has ambiguous typed roughness')
    _v53_inline_identity(metadata, header, exporter)
    return (DELFT_EXPORTER_GENERATIONS[exporter], exporter)


def _crosscheck_v53_runtime(inline, contract):
    """Bind the neutral native wire to the admitted GPU sidecar profile."""
    require(isinstance(inline, dict) and isinstance(contract, dict),
            'Transport v53 runtime profile is missing')
    neutral_document = execution_profile_document(inline['profile'])
    runtime_document = execution_profile_document(contract['profile'])
    neutral_components = {
        row['component_id']: row
        for row in neutral_document.get('component_profiles', ())}
    runtime_components = {
        row['component_id']: row
        for row in runtime_document.get('component_profiles', ())}
    require(neutral_document.get('backend') is None and
            runtime_document.get('backend') == BACKEND_GPU and
            all(neutral_document[key] == runtime_document[key]
                for key in neutral_document
                if key not in {'backend', 'fingerprint',
                               'component_profiles'}) and
            neutral_components == runtime_components and
            inline['component_documents'] == contract['component_documents'],
            'Transport v53 native/runtime execution profiles differ')


def _bind_execution_profile_components(data, profile, documents):
    fraction_ids = tuple(row.fraction_id for row in profile.fraction_layout)
    require(int(data['nfrac']) == len(fraction_ids),
            'Execution-profile fraction layout differs from prepared state')
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
                'Prepared component document differs: '+component_id)
        require(fingerprint_key not in data or
                data[fingerprint_key] == document['fingerprint'],
                'Prepared component fingerprint differs: '+component_id)
        data[document_key] = document
        data[fingerprint_key] = document['fingerprint']
    if 'capacity' in documents:
        require('capacity_fraction_ids' not in data or
                tuple(data['capacity_fraction_ids']) == fraction_ids,
                'Prepared capacity fraction IDs differ from execution profile')
        data['capacity_fraction_ids'] = fraction_ids


def _load_execution_profile_sidecar(path, expected_sha256, *,
                                    allow_validation_internal_neutral=False,
                                    data=None,
                                    neutral_prognostic_secondary=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and
            sha(path) == expected_sha256,
            'Execution-profile sidecar differs from preparation receipt')
    raw = _strict_json_object(path)
    require(set(raw) == {
        'schema', 'backend', 'admission', 'profile', 'component_documents'},
        'Execution-profile sidecar inventory differs')
    require(type(raw['schema']) is int and
            raw['schema'] == EXECUTION_PROFILE_SIDECAR_SCHEMA and
            raw['backend'] == BACKEND_GPU and
            raw['admission'] in EXECUTION_PROFILE_ADMISSIONS and
            isinstance(raw['component_documents'], dict),
            'Execution-profile sidecar schema/backend/admission differs')
    profile = load_execution_profile(raw['profile'])
    require(execution_profile_document(profile) == raw['profile'],
            'Execution-profile document is not the exact canonical document')
    expected = {row.component_id: row for row in profile.component_profiles}
    documents = raw['component_documents']
    require(set(documents) == set(expected),
            'Execution-profile component document inventory differs')
    for component_id in sorted(documents):
        document = documents[component_id]
        require(isinstance(document, dict) and
                bind_component_profile(component_id, document) ==
                expected[component_id],
                'Execution-profile component document differs: '+component_id)
    profile_prepared_data = data
    if (data is not None and
            (HIDING_EXECUTION_KEY in data or neutral_prognostic_secondary)):
        profile_prepared_data = dict(data)
    if data is not None and HIDING_EXECUTION_KEY in data:
        # The nested, frozen Q3-C profile deliberately declares hiding.none.
        # Validate that foundation against an explicit neutral view; the
        # authenticated v55 outer document owns and separately cross-checks
        # the actual nonzero native hiding control.
        profile_prepared_data['hiding_exposure'] = np.int32(0)
    if neutral_prognostic_secondary:
        # The native v57 initializer necessarily exports model=2 and its hI
        # state layout.  The execution-profile sidecar, however, is the
        # unchanged nested v54/v55 Q3-C foundation and therefore declares
        # secondary.off.  Its completeness check must see an explicit neutral
        # *view* of those five selectors.  The real arrays/layout remain in
        # ``data`` and the already authenticated v57 outer identity is
        # installed and cross-checked immediately after this foundation gate.
        require(data is not None and
                int(np.asarray(data.get(
                    'native_q3h3_secondary_verified', -1))) == 1,
                'Transport v57 foundation view lacks native authentication')
        profile_prepared_data.update(
            secondary_flow_model=np.int32(0),
            secondary_flow=np.int32(0),
            secondary_flow_transport_direction=np.int32(0),
            secondary_flow_momentum_beta=np.float64(0.0),
            secondary_flow_espir=np.float64(1.0))
    validate_fixed_grid_profile_completeness(
        profile, component_documents=documents,
        prepared_data=profile_prepared_data)
    admission = raw['admission']
    if admission == 'runtime':
        require(profile.backend == BACKEND_GPU,
                'Runtime execution profile must select the GPU backend')
    else:
        require(allow_validation_internal_neutral and profile.backend is None,
                'Internal-neutral execution profile is validation-only')
    components = _component_rows(profile)
    components_sha256 = hashlib.sha256(
        _canonical_json(documents).encode('ascii')).hexdigest()
    identities = {
        'execution_profile_v2_sidecar_sha256': expected_sha256,
        'execution_profile_v2_semantic_id': profile.semantic_id,
        'execution_profile_v2_ledger_semantic_id': profile.ledger_semantic_id,
        'execution_profile_v2_fingerprint': profile.fingerprint,
        'execution_profile_v2_components_sha256': components_sha256,
        'execution_profile_v2_admission': admission,
        'execution_profile_v2_backend': BACKEND_GPU,
        'execution_profile_v2_profile_backend': (
            profile.backend if profile.backend is not None else 'null'),
    }
    contract = dict(sidecar_schema=EXECUTION_PROFILE_SIDECAR_SCHEMA,
        identities=identities, components=components,
        component_documents=documents, profile=profile)
    if data is not None:
        data[_EXECUTION_PROFILE_DATA_KEY] = contract
        _bind_execution_profile_components(data, profile, documents)
        if admission == 'runtime' and int(data.get('bedlaw', -1)) in (2, 8, 11):
            from .m1_profile_transport import (
                bind_public_hiding_runtime, bind_public_law11_runtime,
                bind_public_scalar_runtime)
            methods = {row.entry_id for row in profile.method_selections}
            if methods & TYPED_ROUGHNESS_METHODS:
                hiding_identity = data.get(HIDING_EXECUTION_KEY)
                if hiding_identity is None:
                    require(int(np.asarray(data['hiding_exposure'])) == 0 and
                            HIDING_ORDERED_ROUTE_KEY not in data,
                            'Q3-C execution profile cannot carry generic hiding')
                    bind_public_scalar_runtime(data, profile, documents)
                else:
                    identity = validate_hiding_foundation_execution_document(
                        hiding_identity)
                    bind_public_hiding_runtime(
                        data, profile, documents, mode=identity['mode'],
                        method_id=identity['method_id'],
                        parker_alpha=identity['parker_alpha'],
                        wu_exponent=identity['wu_exponent'])
            elif int(data.get('bedlaw', -1)) == 11:
                require(int(np.asarray(data['hiding_exposure'])) == 0 and
                        HIDING_EXECUTION_KEY not in data and
                        HIDING_ORDERED_ROUTE_KEY not in data,
                        'Historical law-11 profile cannot carry generic hiding')
                bind_public_law11_runtime(data, profile, documents)
    return contract


def candidate_module():
    from . import general_live_transport
    return general_live_transport


def _read_prepared(folder, *, allow_validation_internal_neutral=False):
    folder = Path(folder).resolve()
    path = folder/'preparation_receipt.json'
    require(path.is_file() and not path.is_symlink(), 'Missing transport preparation receipt')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    require(type(receipt.get('schema')) is int and receipt['schema'] in (6, 7) and
            receipt.get('variant') == VARIANT and
            receipt.get('exporter_executable_sha256') in EXPORTERS and
            receipt.get('initialization_CPU_only') is True and
            type(receipt.get('native_solver_steps_performed')) is int and
            receipt['native_solver_steps_performed'] == 0 and
            receipt.get('GPU_solver_executed') is False, 'Transport zero-step preparation identity differs')
    exporter = receipt.get('exporter_executable_sha256')
    if exporter in DELFT_EXPORTER_GENERATIONS:
        require(receipt.get('initializer_generation') ==
                DELFT_EXPORTER_GENERATIONS[exporter],
                'Delft component initializer generation/hash pair differs')
    if exporter in GAIA_EXPORTER_GENERATIONS:
        require(receipt.get('initializer_generation') == GAIA_EXPORTER_GENERATIONS[exporter],
                'Transport GAIA initializer generation/hash pair differs')
    require((exporter == Q3H2_EXPORTER_SHA256) ==
            isinstance(receipt.get('secondary_foundation_identity'), dict),
            'Transport preparation v56 receipt presence differs')
    require((exporter == Q3H3_EXPORTER_SHA256) ==
            isinstance(receipt.get('prognostic_secondary_identity'), dict),
            'Transport preparation v57 receipt presence differs')
    if exporter == Q3H3_EXPORTER_SHA256:
        outer = receipt['prognostic_secondary_identity']
        identity = validate_prognostic_secondary_execution_document(
            outer.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)
            if isinstance(outer, dict) else None)
        require(
            receipt.get('prognostic_secondary_execution_fingerprint') ==
                identity['fingerprint'] and
            receipt.get('prognostic_secondary_state_semantics') ==
                identity['state_semantics'] and
            receipt.get('prognostic_secondary_integrated_ledger_units') ==
                identity['integrated_ledger_units'] and
            receipt.get(
                'prognostic_secondary_bed_boundary_rotation_policy') ==
                identity['bed_boundary_rotation_policy'] and
            receipt.get('prognostic_secondary_state_units') == 'm2_s' and
            identity['integrated_ledger_units'] == 'm4_s',
            'Transport preparation v57 visible routing receipt differs')
    artifacts = receipt.get('artifacts', {})
    require(ARTIFACTS <= set(artifacts) <= ARTIFACTS | OPTIONAL_ARTIFACTS,
            'Transport preparation artifact inventory differs')
    require('avalanche_metadata.bin' not in artifacts or
            receipt.get('exporter_executable_sha256') in
            (INTEGRATED_EXPORTER_SHA256, EH_EXPORTER_SHA256,
             *DELFT_EXPORTER_GENERATIONS),
            'Avalanche preparation requires the integrated initializer')
    for name, identity in artifacts.items():
        authenticated_member(folder, name, identity)
    profile_path = folder/EXECUTION_PROFILE_ARTIFACT
    require((EXECUTION_PROFILE_ARTIFACT in artifacts) ==
            (profile_path.exists() or profile_path.is_symlink()),
            'Execution-profile sidecar is unlisted or missing')
    origins = receipt.get('origin', {}).get('files', {})
    require('case.data' in origins, 'Transport deck identity missing')
    inline_v53 = None
    if 'geometry.npz' not in origins:
        # QGIS writes small native decks with geometry in case.data itself.
        # Authenticate that inline form rather than requiring a nonexistent
        # standalone mesh sidecar.
        require('case.data.json' in origins, 'Transport inline geometry metadata missing')
        deck_lines = (folder/'case.data').read_text(
            encoding='utf-8').splitlines()
        header = deck_lines[:2]
        require(len(header) == 2 and
                re.fullmatch(r'[0-9a-fA-F]{64}', header[1].strip()) is not None,
                'Transport inline geometry deck header differs')
        metadata = json.loads((folder/'case.data.json').read_text(encoding='utf-8'))
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
        if header[0].strip() in ('53', '54', '55', '56', '57'):
            require(EXECUTION_PROFILE_ARTIFACT in artifacts,
                    'Transport v53/v54/v55/v56/v57 input lacks its execution-profile sidecar')
            inline_v53 = _v53_inline_identity(
                metadata, deck_lines, exporter)
            if header[0].strip() == '56':
                require(receipt.get('secondary_foundation_identity') ==
                        metadata.get('secondary_foundation_identity'),
                        'Transport preparation v56 receipt differs from deck')
            if header[0].strip() == '57':
                require(receipt.get('prognostic_secondary_identity') ==
                        metadata.get('prognostic_secondary_identity'),
                        'Transport preparation v57 receipt differs from deck')
        require((legacy_inline or gaia_inline or inline_v53 is not None) and
                metadata.get('fingerprint') == header[1].strip() and
                metadata.get('sidecars', {}).get('native_binary') is None and
                metadata.get('sidecars', {}).get('geometry') == 'geometry.json' and
                'geometry.json' in origins,
                'Transport inline geometry fingerprint differs')
    for name, identity in origins.items():
        authenticated_member(folder, name, identity, nested=True)
    for line in (folder/'case.data').read_text().splitlines():
        tokens = line.split()
        if len(tokens) == 2 and re.fullmatch(r'[0-9a-fA-F]{64}', tokens[1]):
            require(origins.get(tokens[0]) == tokens[1].lower(),
                    'Transport native deck dependency is not authenticated')
    data = candidate_module().read_records(folder)
    if exporter in DELFT_FORCING_EXPORTERS:
        require(receipt['schema'] == 7 and FORCING_ARTIFACT in artifacts,
                'Delft initializer lacks authenticated forcing binding')
        from .forcing_profile import apply_forcing_binding
        binding = json.loads((folder/FORCING_ARTIFACT).read_text(encoding='utf-8'))
        apply_forcing_binding(data, binding)
    else:
        require(FORCING_ARTIFACT not in artifacts,
                'Historical initializer cannot acquire a forcing binding')
    if inline_v53 is not None:
        _install_inline_hiding_identity(data, inline_v53)
        if DELFT_EXPORTER_GENERATIONS.get(exporter)==Q3H2_EXPORTER_GENERATION:
            _validate_native_v56_secondary(
                data,inline_v53['secondary_execution_identity'])
        if DELFT_EXPORTER_GENERATIONS.get(exporter)==Q3H3_EXPORTER_GENERATION:
            _validate_native_v57_prognostic_secondary(
                data, inline_v53[
                    'prognostic_secondary_execution_identity'])
    if exporter == GAIA_EXPORTER_V4_SHA256:
        require(data.get('gaia_fixed_inlet_contract') == 'historical_absent' and
                'fixed_boundary_bed_mode' not in data and 'fixed_boundary_bed' not in data,
                'Historical v4 GAIA preparation has unexpected fixed-face semantics')
    elif exporter == GAIA_EXPORTER_V5_SHA256:
        require(data.get('gaia_fixed_inlet_contract') == 'explicit_native_v5' and
                'fixed_boundary_bed_mode' in data and 'fixed_boundary_bed' in data,
                'Current v5 GAIA preparation lacks explicit fixed-face semantics')
        require(np.isin(np.asarray(data['bed_boundary_mode']), (0, 1)).all(),
                'Historical v5 GAIA preparation has unsupported mode-2 semantics')
    elif exporter == GAIA_EXPORTER_V6_SHA256:
        require(data.get('gaia_fixed_inlet_contract') == 'explicit_native_v5' and
                'fixed_boundary_bed_mode' in data and 'fixed_boundary_bed' in data and
                np.isin(np.asarray(data['bed_boundary_mode']), (0, 1, 2)).all(),
                'Current v6 GAIA preparation lacks explicit mode-2 fixed-face semantics')
    if EXECUTION_PROFILE_ARTIFACT in artifacts:
        profile_contract = _load_execution_profile_sidecar(
            profile_path, artifacts[EXECUTION_PROFILE_ARTIFACT],
            allow_validation_internal_neutral=allow_validation_internal_neutral,
            data=data,
            neutral_prognostic_secondary=(
                exporter == Q3H3_EXPORTER_SHA256))
        if inline_v53 is not None:
            _crosscheck_v53_runtime(inline_v53, profile_contract)
    if inline_v53 is not None:
        # Validate the frozen v54/v55 profile while its numerical selector is
        # still secondary.off, then compose the authenticated v56 outer route.
        _install_inline_secondary_identity(data, inline_v53)
        _install_inline_prognostic_secondary_identity(data, inline_v53)
    require(set(artifacts) == artifact_names(data), 'Transport viscosity/cache artifacts differ')
    with np.load(folder/'coupled_input.npz', allow_pickle=False) as archive:
        require(all(archive[k].dtype == np.asarray(data[k]).dtype for k in archive.files),
                'Transport NPZ dtype differs from native records')
    return data, hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest(), sha(path)


def read_prepared(folder):
    """Read a public prepared input; validation-neutral profiles are forbidden."""
    return _read_prepared(folder, allow_validation_internal_neutral=False)


def read_prepared_validation_internal(folder):
    """Read a bounded direct-prepared fixture with a neutral internal profile.

    The shared public lifecycle never calls this entry point.  It exists only
    to exercise currently unadmitted law-11 selector identities without
    relabelling them as an admitted GPU runtime route.
    """
    return _read_prepared(folder, allow_validation_internal_neutral=True)


def scope(data, **overrides):
    controls = dict(start=float(data.get('time', 0.)), end=float(data['endtime']),
        interval=float(data['output_dt']), dtmax=float(data['dtmax']),
        launch_mode='plain', integrator='ssp33')
    controls.update(overrides)
    return candidate_module().validate_scope(data, **controls)


def integration_identity(integrator, launch_mode):
    """Bind the user-selected GPU method independently of CPU initialization."""
    return candidate_module().integration_identity(integrator, launch_mode)


def integrator_execution_contract(data, integrator, contract):
    """Bind checkpoint/coupling identity to the selected time integrator."""
    if integrator != 'implicit_fwave_midpoint':
        return contract
    from .implicit_midpoint_fwave import (
        checkpoint_schema, method_identity, coupling_identity,
        validate_scope as validate_implicit_scope)
    validate_implicit_scope(data)
    result = dict(contract)
    selected_schema = checkpoint_schema(data)
    result['schema'] = selected_schema
    result['method'] = dict(contract['method'],
        **method_identity(coupling=coupling_identity(data), data=data),
        transport_checkpoint_schema=selected_schema,
        predictor_checkpoint='implicit_predictor_rate_and_valid_flag_v1')
    public_marker = data.get('m1_profile_public_admission')
    scalar_public = (
        isinstance(public_marker, dict) and
        public_marker.get('admission_id') in (
            'q3c.scalar_capacity_total_v1',
            'q3h1.scalar_capacity_hiding_v1') and
        int(data['bedlaw']) in (2, 8, 11))
    if int(data['bedlaw']) == 11 or scalar_public:
        from .m1_profile_transport import profile_identity
        result['method'].update(profile_identity(data))
    result['manifest'] = dict(contract['manifest'],
        transport_checkpoint_schema=selected_schema,
        implicit_fwave_midpoint=True,
        implicit_predictor_checkpoint=True,
        flow2dh_boundary_memory_checkpoint=(
            int(data.get('exchange_mode', 0)) == 4 and
            selected_schema in (3, 4)),
        weakly_coupled_sediment=bool(
            int(data['bedlaw']) in (2, 5, 6, 7, 8, 11)))
    return result


def prepare(deck, output, *, stop_file=None):
    from validation.gpu_fwave.coupled_reference import copy_inputs
    deck, output = Path(deck).resolve(), Path(output).resolve()
    require(deck.name == 'case.data', 'Transport preparation requires a native case.data deck')

    def check_stop():
        if stop_file is not None and Path(stop_file).exists():
            raise InterruptedError('GPU transport preparation cancelled')

    check_stop()
    deck_lines = deck.read_text(encoding='utf-8').splitlines()
    header = deck_lines[:2]
    require(len(header) == 2 and re.fullmatch(r'[0-9a-fA-F]{64}', header[1].strip()),
            'Transport deck header differs')
    metadata_path = deck.with_name(deck.name + '.json')
    require(metadata_path.is_file(), 'Transport deck metadata missing')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    gaia = metadata.get('gaia_engelund_chollet_cunge')
    slope = metadata.get('bed_slope')
    secondary = metadata.get('secondary_flow')
    use_gaia = header[0].strip() == '43'
    delft_generation = delft_exporter = None
    if use_gaia:
        require(metadata.get('format') == 'xbraids_cutcell-v43' and
                metadata.get('fingerprint') == header[1].strip() and
                metadata.get('bedload_law') == 9 and isinstance(gaia, dict) and
                gaia.get('skin_roughness_ratio') == 3. and
                gaia.get('slope_magnitude_beta') == 1.3 and
                gaia.get('talmon_deviation_beta2') == .85 and
                gaia.get('secondary_alpha') == 1. and
                isinstance(slope, dict) and slope.get('islope') == 4 and
                isinstance(secondary, dict) and secondary.get('model') == 'gaia',
                'Transport v43 law-9 initializer envelope differs')
    else:
        delft_generation, delft_exporter = _select_delft_exporter(
            metadata, deck_lines)
    if (ROOT/'PACKAGE-MANIFEST.json').exists():
        relative, name, identity = (
            ('braids_gpu/native_exporter_v6', 'xbraids_gaia_export.exe', GAIA_EXPORTER_V6_SHA256)
            if use_gaia else
            ('braids_gpu/'+delft_generation, DELFT_EXPORTER_NAME,
             delft_exporter))
        runtime, executable, env = packaged_native_exporter(relative, name, identity)
    else:
        runtime = ROOT/('braids_gpu/native_exporter_v6' if use_gaia else
                       'braids_gpu/'+delft_generation)
        executable = runtime/('xbraids_gaia_export.exe' if use_gaia else
                              DELFT_EXPORTER_NAME)
        expected = GAIA_EXPORTER_V6_SHA256 if use_gaia else delft_exporter
        if not use_gaia:
            verify_source_exporter(ROOT, exporter_sha256=delft_exporter)
        require(executable.is_file() and sha(executable) == expected,
                'Pinned transport zero-step exporter differs')
        build = json.loads((runtime/'build.json').read_text())
        require(build['executable_sha256'] == expected and
                build.get('generation') ==
                    ('native_exporter_v6' if use_gaia else delft_generation) and
                build.get('native_solver_steps_performed') == 0 and
                (not use_gaia or build.get('numerical_source_unchanged') is True) and
                build.get('Delft3D_source_modified') is False,
                'Transport exporter receipt differs')
        compiler = Path(build['command'][0]).parent
        require(all((compiler/name).is_file() for name in REQUIRED_DLLS), 'Transport initializer DLLs missing')
        env = export_environment(runtime)
        env['PATH'] = str(compiler)+os.pathsep+env['PATH']
    origin = copy_inputs(deck, output, stop_file=stop_file)
    if 'geometry.npz' not in origin['files']:
        # QGIS inlines the numerical mesh in case.data, while its result
        # renderer needs the original polygon vertices and CRS from this
        # separately declared visualization sidecar.
        metadata_path = output/'case.data.json'
        require(metadata_path.is_file(), 'Inline QGIS deck metadata missing')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        require(metadata.get('sidecars', {}).get('geometry') == 'geometry.json',
                'Inline QGIS geometry sidecar differs')
        geometry = deck.parent/'geometry.json'
        require(geometry.is_file() and not geometry.is_symlink(),
                'Inline QGIS polygon geometry missing')
        shutil.copy2(geometry, output/'geometry.json')
        origin['files']['geometry.json'] = sha(output/'geometry.json')
    with (output/'initial.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([str(executable), deck.name], cwd=output, env=env,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        began = time.monotonic()
        try:
            while process.poll() is None:
                check_stop()
                if time.monotonic()-began > 240:
                    raise TimeoutError('Transport zero-step initialization exceeded 240 seconds')
                time.sleep(.05)
        except (InterruptedError, TimeoutError):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, [str(executable), deck.name])
    check_stop()
    native_data = candidate_module().read_records(output)
    data = dict(native_data)
    if not use_gaia:
        from .forcing_profile import (apply_forcing_binding,
                                      forcing_binding_from_metadata)
        metadata = json.loads((output/'case.data.json').read_text(encoding='utf-8'))
        native_skin = dict(
            kinematic_viscosity_m2_s=float(native_data[
                'native_skin_kinematic_viscosity_m2_s']),
            sand_roughness_m=float(native_data['native_skin_sand_roughness_m']),
            silt_roughness_m=float(native_data['native_skin_silt_roughness_m']),
            critical_mud_fraction_low=float(native_data[
                'native_skin_critical_mud_fraction_low']),
            critical_mud_fraction_high=float(native_data[
                'native_skin_critical_mud_fraction_high']),
            fixed_fluff_fraction=float(native_data['native_skin_fluff_fraction']))
        binding = forcing_binding_from_metadata(
            metadata, np.asarray(native_data['manning_cell']),
            native_skin_controls=native_skin)
        apply_forcing_binding(data, binding)
        write_json(output/FORCING_ARTIFACT, binding)
        inline_profile = _v53_inline_identity(
            metadata, deck_lines, delft_exporter)
        _install_inline_hiding_identity(data, inline_profile)
        if delft_generation==Q3H2_EXPORTER_GENERATION:
            _validate_native_v56_secondary(
                data,inline_profile['secondary_execution_identity'])
        if delft_generation==Q3H3_EXPORTER_GENERATION:
            _validate_native_v57_prognostic_secondary(
                data, inline_profile[
                    'prognostic_secondary_execution_identity'])
    profile_source = deck.parent/EXECUTION_PROFILE_ARTIFACT
    profile_target = output/EXECUTION_PROFILE_ARTIFACT
    declares_profile = metadata.get('execution_profile_v2') is not None
    require(not declares_profile or profile_source.is_file(),
            'Schema-v2 deck requires an explicit GPU execution-profile sidecar')
    require(not profile_source.is_symlink(),
            'Execution-profile preparation source cannot be linked')
    if profile_source.is_file():
        if profile_target.exists() or profile_target.is_symlink():
            require(profile_target.is_file() and not profile_target.is_symlink() and
                    sha(profile_target) == sha(profile_source),
                    'Copied execution-profile sidecar differs from deck source')
        else:
            shutil.copy2(profile_source, profile_target)
        _load_execution_profile_sidecar(
            profile_target, sha(profile_target),
            allow_validation_internal_neutral=False, data=data,
            neutral_prognostic_secondary=(
                delft_generation == Q3H3_EXPORTER_GENERATION))
    else:
        require(not (profile_target.exists() or profile_target.is_symlink()),
                'Unexpected execution-profile sidecar in preparation output')
    if not use_gaia:
        # The sidecar owns the unchanged nested v54/v55 profile.  Materialize
        # the authenticated v56/v57 GPU composition only after that base passes.
        _install_inline_secondary_identity(data, inline_profile)
        _install_inline_prognostic_secondary_identity(data, inline_profile)
    # Preparation performs no solver step, but it still validates the exact
    # runtime envelope.  A public law-11 sidecar explicitly selects the
    # implicit-fwave GPU backend, so validate that method rather than the
    # historical SSPRK33 default.  This is profile-owned dispatch, never a
    # mesh-size heuristic or fallback.
    preparation_integrator = (
        'implicit_fwave_midpoint'
        if (int(data.get('bedlaw', -1)) in (2, 8, 11) and
            data.get('m1_profile_public_admission') is not None)
        else 'ssp33')
    events = scope(data, integrator=preparation_integrator)
    save_arrays(output/'coupled_input.npz', **native_data)
    receipt = dict(schema=7 if not use_gaia else 6, variant=VARIANT, origin=origin,
        artifacts={name:sha(output/name) for name in sorted(artifact_names(data))},
        exporter_executable_sha256=(GAIA_EXPORTER_V6_SHA256 if use_gaia else delft_exporter),
        initializer_generation=('native_exporter_v6' if use_gaia else delft_generation),
        initialization_CPU_only=True,
        native_solver_steps_performed=0, GPU_solver_executed=False)
    if delft_generation == Q3H2_EXPORTER_GENERATION:
        receipt['secondary_foundation_identity'] = \
            metadata['secondary_foundation_identity']
    if delft_generation == Q3H3_EXPORTER_GENERATION:
        identity = inline_profile['prognostic_secondary_execution_identity']
        receipt.update(
            prognostic_secondary_identity=
                metadata['prognostic_secondary_identity'],
            prognostic_secondary_execution_fingerprint=
                identity['fingerprint'],
            prognostic_secondary_state_semantics=
                identity['state_semantics'],
            prognostic_secondary_integrated_ledger_units=
                identity['integrated_ledger_units'],
            prognostic_secondary_bed_boundary_rotation_policy=
                identity['bed_boundary_rotation_policy'],
            prognostic_secondary_state_units='m2_s')
    write_json(output/'preparation_receipt.json', receipt)
    read_prepared(output)
    return dict(status='prepared', output=str(output), variant=VARIANT,
        cells=int(data['nc']), fractions=int(data['nfrac']), layers=int(data['nbed_layers']),
        forcing_knots_s=events.knots.tolist(), native_solver_steps_performed=0,
        preparation_receipt_sha256=sha(output/'preparation_receipt.json'), archive_bound=False)


def execution_profile_contract(data):
    """Return additive identity fields for the shared fixed-grid lifecycle."""
    bound = data.get(_EXECUTION_PROFILE_DATA_KEY)
    if bound is None:
        return dict(identities={}, method={}, manifest={}, result={})
    require(isinstance(bound, dict) and set(bound) == {
        'sidecar_schema', 'identities', 'components',
        'component_documents', 'profile'},
        'Prepared execution-profile contract differs')
    profile_identities = dict(bound['identities'])
    require(tuple(profile_identities) == EXECUTION_PROFILE_IDENTITY_FIELDS,
            'Prepared execution-profile identity inventory differs')
    identities = dict(profile_identities)
    prognostic = data.get(PROGNOSTIC_SECONDARY_EXECUTION_KEY)
    if prognostic is not None:
        identity = validate_prognostic_secondary_execution_document(prognostic)
        identities.update(
            prognostic_secondary_execution_fingerprint=
                identity['fingerprint'],
            prognostic_secondary_state_semantics=
                identity['state_semantics'],
            prognostic_secondary_integrated_ledger_units=
                identity['integrated_ledger_units'],
            prognostic_secondary_bed_boundary_rotation_policy=
                identity['bed_boundary_rotation_policy'],
            prognostic_secondary_composition_order=
                list(identity['composition_order']),
        )
    components = list(bound['components'])
    # The implicit method contract already carries the visible prognostic
    # fields.  Keep them in the lifecycle identities (and therefore every
    # checkpoint/result) without colliding with that method record.
    method = dict(profile_identities,
        execution_profile_v2_bound=True,
        execution_profile_v2_sidecar_schema=bound['sidecar_schema'],
        execution_profile_v2_components=components)
    profile_record = dict(schema=bound['sidecar_schema'],
        **profile_identities, components=components)
    return dict(identities=identities, method=method,
        manifest={'execution_profile_v2': profile_record},
        result=dict(identities, execution_profile_v2=profile_record))


def archive_sources(folder, data, *, integrator='ssp33'):
    folder = Path(folder)
    candidate_module().prepare_sources(folder, data, integrator=integrator)
    for name in ('general_live_transport_runner.py', 'general_live_bank_runner.py',
                 'graded_native_exporter.py', 'fixed_grid_lifecycle.py', 'integrated.py',
                 'runner.py', '__main__.py', 'atomic_output.py', 'delft_components_exporter.py'):
        copied = folder/'worker_python'/name
        copied.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'braids_gpu'/name, copied)
    shutil.copy2(ROOT/'braids_core/gpu_capabilities.py', folder/'worker_python/gpu_capabilities.py')
    shutil.copy2(ROOT/'braids_core/sediment_method_catalog.py',
                 folder/'worker_python/sediment_method_catalog.py')
    profile_contract = execution_profile_contract(data)
    if profile_contract['identities']:
        profile_root = folder/'worker_python/braids_core/sediment'
        profile_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'braids_core/sediment/execution_profile_v2.py',
                     profile_root/'execution_profile_v2.py')
        shutil.copy2(
            ROOT/'braids_core/sediment/fixed_grid_profile_completeness.py',
            profile_root/'fixed_grid_profile_completeness.py')
        shutil.copy2(
            ROOT/'braids_core/sediment/capacity_foundation_execution.py',
            profile_root/'capacity_foundation_execution.py')
        if HIDING_EXECUTION_KEY in data:
            shutil.copy2(
                ROOT/'braids_core/sediment/hiding_foundation_execution.py',
                profile_root/'hiding_foundation_execution.py')
        if SECONDARY_EXECUTION_KEY in data:
            shutil.copy2(
                ROOT/'braids_core/sediment/secondary_foundation_execution.py',
                profile_root/'secondary_foundation_execution.py')
        shutil.copy2(ROOT/'braids_core/sediment/flow2dh_selector_ledger_v2.json',
                     profile_root/'flow2dh_selector_ledger_v2.json')
        catalog = folder/'worker_python/braids_core/sediment_method_catalog.py'
        catalog.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'braids_core/sediment_method_catalog.py', catalog)
    files = {file.relative_to(folder).as_posix():sha(file) for file in folder.rglob('*') if file.is_file()}
    method = integration_identity(integrator, 'plain')
    if integrator == 'implicit_fwave_midpoint':
        from .implicit_midpoint_fwave import checkpoint_schema as schema_for
        checkpoint_schema = schema_for(data)
    else:
        checkpoint_schema = 1
    runtime_manifest = dict(schema=1, variant=VARIANT, source_files=files,
        transport_checkpoint_schema=checkpoint_schema, CUDA_copied_verbatim=True,
        normal_root_rule=candidate_module().NORMAL_ROOT_RULE,
        normal_face_policy=candidate_module().normal_face_policy(data, integrator),
        integrator=method['integrator'], euler_stages=method['euler_stages'],
        native_cpu_time_integrator=int(data['cpu_time_integrator']))
    if profile_contract['identities']:
        runtime_manifest['execution_profile_v2'] = \
            profile_contract['manifest']['execution_profile_v2']
    write_json(folder/'runtime_manifest.json', runtime_manifest)
    return sha(folder/'runtime_manifest.json')


def checkpoint_preflight(path, data, events, *, identities, start, end, expected_schema=None):
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'Duplicate transport checkpoint field')
        fields = {k:archive[k].copy() for k in archive.files}
    nc, ng, nl, nb = [int(data[k]) for k in ('nc', 'nfrac', 'nbed_layers', 'nb')]
    suspended = candidate_module().suspended_enabled(data)
    shapes = dict(q=(3, nc), layers=(nl, ng, nc), secondary=(nc,),
        bed_elevation=(nc,), ledger_water=(nb,), ledger_bed=(ng, nb), ledger_secondary=(nb,),
        ledger_source=())
    if suspended:
        shapes.update(suspended=(ng, nc), ledger_suspended=(ng, nb), ledger_exchange=(ng, 2))
    implicit = expected_schema in (2, 3, 4, 5, 6, 7, 8)
    implicit_fields = {'implicit_predictor_rate', 'implicit_have_predictor'} if implicit else set()
    flow2dh_memory = (
        expected_schema in (3, 4) and
        int(data.get('exchange_mode', 0)) == 4)
    if flow2dh_memory:
        implicit_fields |= {
            'flow2dh_boundary_last_outflow_concentration',
            'flow2dh_boundary_return_clock',
        }
    other = {'time', 'dt_hint', 'steps', 'rejected', 'next_boundary_knot',
             'transport_checkpoint_schema', 'transport_payload_sha256'}
    require(set(fields) == set(shapes) | set(identities) | other | implicit_fields,
            'Transport checkpoint inventory differs')
    payload = fields.pop('transport_payload_sha256')
    require(payload.shape == () and payload.dtype.kind == 'U' and payload.item() == digest(fields),
            'Transport checkpoint payload differs')
    for key, identity in identities.items():
        value = fields[key]
        expected = np.asarray(identity)
        require(value.shape == expected.shape and value.dtype.kind == 'U' and
                expected.dtype.kind == 'U' and np.array_equal(value, expected),
                'Transport checkpoint identity differs: '+key)
    for key, shape in shapes.items():
        value = fields[key]
        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(),
                'Transport checkpoint field differs: '+key)
    if implicit:
        predictor = fields['implicit_predictor_rate']
        valid = fields['implicit_have_predictor']
        require(predictor.shape == (3, nc) and predictor.dtype == np.float64 and
                np.isfinite(predictor).all() and valid.shape == () and
                valid.dtype == np.int32 and int(valid) in (0, 1),
                 'Implicit predictor checkpoint differs')
    if flow2dh_memory:
        for key in ('flow2dh_boundary_last_outflow_concentration',
                    'flow2dh_boundary_return_clock'):
            value = fields[key]
            require(value.shape == (ng, len(data['owner'])) and
                    value.dtype == np.float64 and np.isfinite(value).all() and
                    np.all(value >= 0.0),
                    'FLOW 2DH boundary-memory checkpoint differs: ' + key)
    for key in ('time', 'dt_hint', 'next_boundary_knot'):
        require(fields[key].shape == () and fields[key].dtype == np.float64 and np.isfinite(fields[key]),
                'Invalid transport checkpoint scalar: '+key)
    for key, dtype in (('steps', np.int64), ('rejected', np.int64), ('transport_checkpoint_schema', np.int32)):
        require(fields[key].shape == () and fields[key].dtype == dtype and int(fields[key]) >= 0,
                'Invalid transport checkpoint integer: '+key)
    current = float(fields['time'])
    require(start <= current <= end and float(fields['dt_hint']) > 0 and
            float(fields['next_boundary_knot']) == events.next_after(current) and
            int(fields['transport_checkpoint_schema']) == (expected_schema or 1) and
            expected_schema in (None, 1, 2, 3, 4, 5, 6, 7, 8),
            'Transport checkpoint clock/schema differs')
    # Exchange columns are signed gains in the suspended and bed reservoirs,
    # not nonnegative entrainment/deposition counters.
    require(np.min(fields['q'][0]) >= 0 and np.min(fields['layers']) >= 0 and
            (not suspended or np.min(fields['suspended']) >= 0),
            'Transport checkpoint negative physical inventory')
    dry = fields['q'][0] <= float(data['dry_tolerance'])
    require(not np.any(fields['q'][1:, dry]) and not np.any(fields['secondary'][dry]),
            'Transport checkpoint contains dry-cell momentum or secondary mass')
    if int(data['secondary_flow_model']) != 2:
        require(not any(np.any(fields[key]) for key in ('secondary', 'ledger_secondary', 'ledger_source')),
                'OFF/equilibrium transport checkpoint contains prognostic inventory')
    if int(data['bedlaw']) == 0:
        require(not np.any(fields['ledger_bed']), 'No-bedload checkpoint contains bedload export')
    if suspended and int(data['exchange_mode']) == 0:
        require(not np.any(fields['ledger_exchange']), 'Exchange-OFF checkpoint contains exchange transfer')
    bed = data['rock'] + (fields['layers']/np.asarray(data['bulk_fraction'])[None, :, None]).sum(axis=(0, 1))
    require(np.allclose(fields['bed_elevation'], bed, rtol=0., atol=1.e-8),
            'Transport checkpoint bed differs from inventory')
    return fields


def execution_contract(data):
    candidate = candidate_module()
    suspended = candidate.suspended_enabled(data)
    bedform = bool(data['bedform_roughness'])
    gaia = gaia_diagnostic_method(data)
    forcing = None
    if 'delft_forcing_binding' in data:
        from .forcing_profile import forcing_identity
        forcing = forcing_identity(data)
    return dict(schema=1, schema_key='transport_checkpoint_schema', payload_key='transport_payload_sha256',
        diagnostics=set(ADAPTATION_DIAGNOSTICS) | BEDFORM_DIAGNOSTICS |
            (set(GAIA_DIAGNOSTICS) if gaia is not None else set()),
        method=dict(transport_checkpoint_schema=1, bedlaw=int(data['bedlaw']),
            sediment_catalog_schema=CATALOG_SCHEMA,
            sediment_catalog_scope_id=CATALOG_SCOPE_ID,
            semantic_catalog_id=semantic_catalog_id(),
            native_initializer_time_integrator=int(data['cpu_time_integrator']),
            native_initializer_outlet_extension=int(data['cpu_outlet_extension']),
            suspended_enabled=suspended, bedform_roughness=bedform,
            general_avalanche=bool(data['general_avalanche']),
            static_bed=candidate.static_bed_enabled(data),
            avalanche_order='each_euler_after_exner_before_rebalance',
            mpm_friction_mode=int(data['mpm_friction_mode']), mpm_cf=float(data['mpm_cf']),
            exchange_mode=int(data['exchange_mode']), secondary_flow_model=int(data['secondary_flow_model']),
            hiding_exposure=int(data['hiding_exposure']), bed_slope=int(data['bed_slope']),
            bedload_adaptation=bool(data['bedload_adaptation']),
            bedload_adaptation_length_diameters=float(data['bedload_adaptation_length_diameters']),
            adaptation_output_clock='accepted_time',
            adaptation_checkpoint_policy='derived_recompute_not_persistent',
            horizontal_viscosity=float(data['horizontal_viscosity']),
            horizontal_viscosity_mode=int(data['horizontal_viscosity_mode']),
            horizontal_viscosity_geometry=int(data['horizontal_viscosity_geometry']),
            settling_mode=int(data['settling_mode']), gaia_secondary_diagnostic=gaia,
            normal_root_rule=candidate.NORMAL_ROOT_RULE,
            normal_trace_rule=candidate.NORMAL_TRACE_RULE,
            normal_face_policy=candidate.NORMAL_TRACE_RULE,
            delft_forcing=forcing),
        manifest=dict(transport_checkpoint_schema=1, CUDA_copied_verbatim=True,
            normal_root_rule=candidate.NORMAL_ROOT_RULE, suspended_enabled=suspended,
            bedform_roughness=bedform,
            general_avalanche=bool(data['general_avalanche']),
            gaia_secondary_diagnostic_schema=(1 if gaia is not None else None),
            diffusion_enabled=bool(float(data['horizontal_viscosity']) > 0),
            delft_forcing=forcing))


def endpoint_artifacts(solver, current, destination):
    """Freeze the actual accepted-state CUDA law-9 inlet face flux."""
    if int(solver.data['bedlaw']) != 9:
        return None
    result=solver.gaia_fixed_inlet_diagnostic(float(current))
    rows=result['rows'];fixed=np.ascontiguousarray(result['fixed_boundary_bed'],dtype='<f8')
    require(len(rows)==1 and fixed.shape==(solver.nf,) and np.isfinite(fixed).all() and
            np.isfinite(result.get('diagnostic_dt_s')) and result['diagnostic_dt_s']>0,
            'GAIA endpoint diagnostic inventory differs')
    row=rows[0]
    require(row['boundary_id']==1 and np.isfinite(row['bedload_flux_kg_s']) and
            np.isfinite(row['owner_bed_max_abs_change_m']) and
            type(row['owner_cells_mobile']) is bool,
            'GAIA endpoint diagnostic value differs')
    destination=Path(destination)
    fixed_path=destination/'gaia_fixed_boundary_bed.f64'
    with fixed_path.open('xb') as stream:
        stream.write(fixed.tobytes(order='C'))
    fixed_sha=sha(fixed_path)
    raw_path=destination/'gaia_fixed_inlet_flux_cuda_native.json'
    write_json(raw_path,dict(schema=1,backend='gpu',
        implementation='actual accepted-state CUDA production face pipeline at one-ULP diagnostic dt',
        execution_order=('after final accepted checkpoint/frame capture; scratch buffers may change; '
                         'accepted state and ledgers byte-verified unchanged'),
        diagnostic_dt_s=float(result['diagnostic_dt_s']),
        rows=rows,fixed_boundary_bed_artifact=fixed_path.name,
        fixed_boundary_bed_sha256=fixed_sha))
    raw_sha=sha(raw_path)
    wrapper=destination/'gaia_fixed_inlet_flux.json'
    write_json(wrapper,dict(schema=1,backend='gpu',gaia_fixed_inlet=dict(
        time_s=float(row['time_s']),boundary_id=int(row['boundary_id']),
        bedload_flux_kg_s=float(row['bedload_flux_kg_s']),
        sign_convention='positive_into_domain',fixed_boundary_bed_sha256=fixed_sha,
        owner_cells_mobile=row['owner_cells_mobile'],
        owner_bed_max_abs_change_m=float(row['owner_bed_max_abs_change_m']),
        owner_mobility_tolerance_m=float(row['owner_mobility_tolerance_m']),
        native_raw_artifact=raw_path.name,native_raw_sha256=raw_sha)))
    return dict(
        gaia_fixed_inlet_flux_json=dict(file=wrapper.name,sha256=sha(wrapper)),
        native_cuda_face_flux=dict(file=raw_path.name,sha256=raw_sha),
        native_fixed_boundary_bed=dict(file=fixed_path.name,sha256=fixed_sha,
                                       bytes=fixed_path.stat().st_size),
        diagnostic_order='after_final_accepted_output_noncommitting_scratch_only')


def restore_extra(solver, restored):
    # All persistent suspension/exchange fields live in state and ledger.
    solver._glt_accepted_time = float(restored['time'])
    if solver.implicit_midpoint is not None:
        solver.implicit_midpoint.restore_checkpoint(restored)


def checkpoint_extra(solver):
    if solver.implicit_midpoint is None:
        return {}
    return solver.implicit_midpoint.checkpoint_fields()


def run(**kwargs):
    from .fixed_grid_lifecycle import run as run_fixed_grid
    import sys
    return run_fixed_grid(sys.modules[__name__], **kwargs)
