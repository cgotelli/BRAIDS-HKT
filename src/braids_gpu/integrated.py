"""Common GPU entry point: select operators from authenticated preparation.

The user has already selected GPU. This is physical-module dispatch, not a
backend or mesh-size heuristic; each driver still checks every input control.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

from braids_core.gpu_capabilities import (INTEGRATED, AMR, AMR_GENERAL, BANK, TRANSPORT,
    route_for_preparation_schema, resolve_project_route)
from braids_core.sediment.qgis_product_v109 import (
    GPU_BACKEND, PRODUCT_SIDECAR, load_product_document_v109,
    verify_worker_capabilities_v109,
)


QGIS_PREPARED_BINDING = "qgis_product_prepared_v109.json"


def _verify_qgis_product(folder, scope):
    """Verify the host-bound portable product before any GPU allocation."""
    folder=Path(folder).resolve()
    sidecar=folder/PRODUCT_SIDECAR
    receipt_path=folder/QGIS_PREPARED_BINDING
    if not sidecar.is_file() or sidecar.is_symlink() or \
            not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError('Q10 GPU prepared input has no portable product binding')
    profile=load_product_document_v109(json.loads(sidecar.read_text(encoding='ascii')))
    if profile.backend!=GPU_BACKEND:
        raise ValueError('Q10 GPU prepared input names another backend')
    verify_worker_capabilities_v109(profile,scope)
    receipt=json.loads(receipt_path.read_text(encoding='ascii'))
    digest=hashlib.sha256(sidecar.read_bytes()).hexdigest()
    expected={
        'schema':'braids.qgis_product_prepared.v109',
        'product_fingerprint':profile.fingerprint,
        'product_sidecar':PRODUCT_SIDECAR,
        'product_sidecar_sha256':digest,
        'worker_capability':scope['qgis_product_v109'],
    }
    if receipt!=expected:
        raise ValueError('Q10 GPU prepared product receipt differs')
    return profile


def preparation_route(deck):
    """Route ordinary writer metadata; the native driver admits actual physics.

    Metadata is only an initializer hint, never a replacement for native state
    or controls. A stale/missing description cannot select a fallback solver.
    """
    deck = Path(deck).resolve()
    metadata_path = deck.with_name(deck.name + '.json')
    if deck.name != 'case.data' or not deck.is_file() or not metadata_path.is_file():
        raise ValueError('Integrated preparation needs ordinary case.data and case.data.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    with deck.open(encoding='utf-8') as stream:
        version, fingerprint = stream.readline().strip(), stream.readline().strip()
    if (metadata.get('format') != 'xbraids_cutcell-v' + version or
            not re.fullmatch(r'[0-9a-f]{64}', fingerprint) or
            metadata.get('fingerprint') != fingerprint):
        raise ValueError('Integrated preparation metadata differs from native deck identity')
    amr, bank = metadata.get('amr'), metadata.get('bank_failure')
    if amr is None:
        amr = dict(enabled=False)
    if (not isinstance(amr, dict) or type(amr.get('enabled')) is not bool or
            not isinstance(bank, dict) or type(bank.get('enabled')) is not bool):
        raise ValueError('Integrated preparation needs explicit AMR and bank controls')
    return (AMR if bank['enabled'] else AMR_GENERAL) if amr['enabled'] else (
        BANK if bank['enabled'] else TRANSPORT)


def prepare(deck, output, *, stop_file=None, evolving_banks=False,
            selector_policy=None, recursive_bank_bedform_avalanche=False):
    from .amr_general_session import validate_selector_policy
    validate_selector_policy(selector_policy)
    if stop_file is not None and Path(stop_file).exists():
        raise InterruptedError('Integrated GPU preparation cancelled')
    route = preparation_route(deck)
    if selector_policy is not None and route != AMR_GENERAL:
        raise ValueError('General AMR selector policy requires nonbank AMR input')
    if evolving_banks and route != AMR:
        raise ValueError('Evolving-bank AMR preparation requires an AMR input deck')
    if (type(recursive_bank_bedform_avalanche) is not bool or
            recursive_bank_bedform_avalanche and
            (route != AMR or not evolving_banks)):
        raise ValueError('Recursive bedform-bank preparation requires evolving bank AMR')
    if route in (AMR, AMR_GENERAL):
        from .amr_on_demand_runner import prepare as initialize
        return initialize(deck, output, stop_file=stop_file,
                          evolving_banks=evolving_banks,
                          recursive_bank_bedform_avalanche=
                            recursive_bank_bedform_avalanche,
                          **(dict(selector_policy=selector_policy)
                             if selector_policy is not None else {}))
    if route == BANK:
        from .general_live_bank_runner import prepare as initialize
    else:
        from .general_live_transport_runner import prepare as initialize
    return initialize(deck, output, stop_file=stop_file)


def resolve_prepared(folder):
    path = Path(folder).resolve()/'preparation_receipt.json'
    if not path.is_file() or path.is_symlink():
        raise ValueError('Integrated GPU execution requires an ordinary preparation receipt')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    route = route_for_preparation_schema(receipt.get('schema'), receipt.get('variant'))
    manifest_path = Path(__file__).resolve().parents[1]/'PACKAGE-MANIFEST.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        scope = manifest.get('source_worker_scope', {})
        if route not in scope.get('supported_gpu_variants', ()) or route in scope.get('excluded_gpu_variants', {}):
            raise ValueError('This GPU package does not contain the selected physical operators: '+route)
        if manifest.get('QGIS_release') is True:
            _verify_qgis_product(folder,scope)
    return route
