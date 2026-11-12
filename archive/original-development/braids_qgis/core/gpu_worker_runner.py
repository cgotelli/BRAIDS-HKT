"""QProcess launcher for an explicitly selected external BRAIDS GPU worker.

This module is host-only.  In particular it neither imports CuPy nor selects a
CPU fallback.  Source-package and Python executable hashes are independent
trust decisions; callers must supply both expected SHA-256 identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from pathlib import Path, PurePosixPath

import numpy as np
from braids_core.gpu_capabilities import (INTEGRATED, gpu_integrator_label,
    route_for_preparation_schema)
from braids_core.windows_paths import filesystem_path
from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from .gpu_amr_frame_io import read_gpu_amr_run
from .gpu_amr_prepared_io import inspect_gpu_amr_prepared
from .gpu_on_demand_prepared import inspect_gpu_on_demand_prepared
from .gpu_frame_io import read_gpu_run
from .gpu_live_transport_io import (has_public_law11_total_prepared_profile,
    validate_implicit_prepared_scope, validate_integrator_scope,
    validate_worker_integrator_capability,
    validate_q3h3_worker_source_scope)
from .gpu_prepared_io import _g1_numerical_identity, _read_records, read_gpu_prepared
from .gpu_worker_identity import inspect_gpu_source_worker


_BANK_AMR_VARIANT = "bank_amr_wetfront_ssp33"
_ON_DEMAND_AMR_VARIANT = "on_demand_bank_amr_ssp33"
_ON_DEMAND_GENERAL_AMR_VARIANT = "on_demand_transport_amr_ssp33"
_GENERAL_AMR_SLOPE_POLICY = "local_eta_plane_v1"
_GENERAL_AMR_SLOPE_SOURCES = frozenset((
    "amr_slope_indicator_v2.py", "amr_slope_indicator_research.py"))
_GRADED_LIVE_BEDLOAD_VARIANT = "live_normal_critical_graded_bedload_ssp33"
_GENERAL_LIVE_BANK_VARIANT = "live_normal_critical_bank_ssp33"
_GENERAL_LIVE_TRANSPORT_VARIANT = "live_normal_critical_transport_ssp33"
_G1_VARIANT = "live_mode9_section_choice_ssp33"
_BANK_AMR_CHECKPOINT_KIND = "braids-gpu-bank-amr-checkpoint-v1"
_RECURSIVE_STEPPER = 'recursive_ssprk33_fine_replaced_register_v1'
_RECURSIVE_BANK_CHECKPOINT = 'recursive_bank_amr_v1'
_RECURSIVE_GENERAL_CHECKPOINT = 'recursive_general_amr_v1'
_RECURSIVE_GENERAL_AVALANCHE_POLICY = 'parent_post_reflux_once_per_macro_v1'
_RECURSIVE_GENERAL_VISCOSITY_POLICY = 'parent_post_reflux_ssprk2_once_per_macro_v1'
_RECURSIVE_GENERAL_VISCOSITY_METHOD = dict(
    policy=_RECURSIVE_GENERAL_VISCOSITY_POLICY,
    placement='external_parent_post_reflux_before_avalanche',
    calls_per_macro=1,
    integrator='SSPRK2_subcycled_area_weighted_velocity_L2_limited',
    limited_norm='area_weighted_velocity_L2',
    area_weighted_velocity_l2_nonincrease=True,
    pairwise_velocity_face_flux=True,
    momentum_conservation='variable_depth_not_exact',
    dynamic_wet_geometry=True,
    general_avalanche_modes=[0, 1],
    composition_order='viscosity_then_avalanche')
_RECURSIVE_GENERAL_BEDFORM_METHOD = dict(
    policy='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False)
_RECURSIVE_BANK_BEDFORM_METHOD = dict(
    policy='recursive_bedform_mode3_parent_avalanche_v1',
    bedform_order='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    bank_transaction='distributed_mode3_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    calls_per_macro={'bank':1,'avalanche':1})
_RECURSIVE_PHYSICAL_TOE_BEDFORM_METHOD = dict(
    policy='recursive_bedform_mode2_physical_toe_parent_avalanche_v1',
    bedform_order='level_ssprk33_stage_capacity_and_pre_friction_v1',
    stage_capacity_evaluations_per_level_step=3,
    pre_friction_evaluations_per_level_step=1,
    accepted_diagnostics='lazy_from_accepted_state',
    checkpointed_diagnostics=False,
    accepted_flux_owner='completed_root_exner_register_v1',
    bank_transaction=
        'physical_toe_supply_undercut_collapse_post_reflux_once_per_macro_v1',
    avalanche_transaction='parent_post_bank_once_per_macro_v1',
    mapping_transaction='reconcile_then_evolving_toe_rebuild_before_commit_v1',
    calls_per_macro={'bank':1,'avalanche':1})
_BEDFORM_BANK_RUNTIME = 'braids_gpu/native_amr_bedform_bank_v3'
_BEDFORM_PHYSICAL_TOE_RUNTIME = \
    'braids_gpu/native_amr_bedform_physical_toe_v1'
_BANK_HISTORY_FLOAT = (
    "bank_history_undercut", "bank_history_reference_height",
    "bank_history_toe_height", "bank_history_reference_length",
    "bank_history_toe_cumulative", "bank_history_collapse_cumulative",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_general_amr_selector_scope(scope: dict, variant: str,
                                      selector_policy: str | None) -> None:
    """Admit an opt-in selector only when this worker explicitly advertises it."""
    if selector_policy is None:
        return  # Absent persisted policy is the historical selector.
    if selector_policy != _GENERAL_AMR_SLOPE_POLICY or variant != _ON_DEMAND_GENERAL_AMR_VARIANT:
        raise ValueError("Unsupported GPU AMR selector policy for the selected route")
    policies = scope.get("general_amr_selector_policies_by_variant", {})
    if (not isinstance(policies, dict) or
            policies.get(variant) != [_GENERAL_AMR_SLOPE_POLICY]):
        raise ValueError("Selected GPU worker does not advertise the slope-aware general AMR selector")
    source = scope.get("source_archives_by_variant", {}).get(variant)
    if source not in {"braids_gpu/frozen_sources/on_demand_transport_amr_v2",
                      "braids_gpu/frozen_sources/on_demand_transport_amr_v3",
                      "braids_gpu/frozen_sources/on_demand_transport_amr_v4",
                      "braids_gpu/frozen_sources/on_demand_transport_amr_v5",
                      "braids_gpu/frozen_sources/on_demand_transport_amr_v6",
                      "braids_gpu/frozen_sources/on_demand_transport_amr_v7",
                      "braids_gpu/frozen_sources/on_demand_transport_amr_v8"}:
        raise ValueError("Slope-aware general AMR requires a versioned packaged source archive")


def isolated_worker_command(worker_root: str | Path, command: str,
                            arguments: list[str]) -> tuple[list[str], str]:
    """Return arguments and cwd for a verified worker's external Python.

    The caller authenticates ``worker_root`` first.  Python's isolated mode
    ignores PYTHONPATH and cwd; the only injected import path is the selected
    source-worker folder.  ``command`` may be ``prepare`` or ``run``.
    """
    if command not in {"prepare", "run"}:
        raise ValueError("Unsupported GPU worker command")
    root = Path(worker_root).expanduser().resolve()
    bootstrap = ("import sys;sys.path.insert(0," + repr(str(root)) + ");"
                 "from braids_gpu.__main__ import main;"
                 "raise SystemExit(main(sys.argv[1:]))")
    return ["-I", "-c", bootstrap, command, *map(str, arguments)], str(root)


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _verify_prepared_input(root: Path) -> str:
    """Authenticate the prepared numerical artifacts before launching Python.

    The worker repeats its own admission, but QGIS should reject a damaged or
    substituted input before it creates a run directory or initializes CUDA.
    Preparation receipts from both supported schemas list direct-child files.
    """
    receipt_path = root / "preparation_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("GPU prepared input has no regular preparation receipt")
    receipt = _json_object(receipt_path)
    artifacts = receipt.get("artifacts")
    if (receipt.get("schema") not in (1, 2, 3, 4, 5, 6, 7) or
            not isinstance(artifacts, dict) or
            "coupled_input.bin" not in artifacts):
        raise ValueError("GPU prepared input has an incomplete artifact inventory")
    for name, expected in artifacts.items():
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name or
                PurePosixPath(name).is_absolute() or
                PurePosixPath(name).as_posix() != name or
                len(PurePosixPath(name).parts) != 1 or name in (".", "..") or
                not isinstance(expected, str) or len(expected) != 64 or
                any(ch not in "0123456789abcdef" for ch in expected)):
            raise ValueError("GPU prepared input has an invalid artifact identity")
        path = root / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected:
            raise ValueError("GPU prepared artifact differs: " + name)
    if receipt["schema"] in (3, 4, 5, 6, 7):
        # Public live routes authenticate copied deck/geometry origin as well
        # as all exported numerical files before QProcess starts.
        read_gpu_prepared(root)
    return _sha256(receipt_path)


def _verify_g1_prepared_input(root: Path) -> str:
    receipt = _json_object(root / "preparation_receipt.json")
    if receipt.get("schema") != 4 or receipt.get("variant") != _G1_VARIANT:
        raise ValueError("Inlet GPU route requires schema-4 prepared input")
    return _verify_prepared_input(root)


def _allowed_prepared_sidecars(scope: dict, variant: str) -> set[str]:
    """Permit the normal-section sidecar only for its scoped live route."""
    policy = scope.get("allowed_prepared_sidecars_by_variant", {})
    if not isinstance(policy, dict):
        raise ValueError("GPU worker has invalid prepared-sidecar allowance policy")
    for selected, names in policy.items():
        expected = (['avalanche_metadata.bin', 'bank_history_metadata.bin', 'bank_toe_metadata.bin',
                     'distributed_bank_metadata.bin', 'normal_metadata.bin']
                    if selected == _GENERAL_LIVE_BANK_VARIANT else ['normal_metadata.bin'])
        if (selected == _GENERAL_LIVE_TRANSPORT_VARIANT and isinstance(names,list) and
                names == sorted(set(names)) and {'normal_metadata.bin'} <= set(names) <=
                {'normal_metadata.bin','coupled_diffusion.bin','avalanche_metadata.bin',
                 'delft_forcing_binding.json','execution_profile_v2.json'}):
            expected = names
        if selected == _GENERAL_LIVE_BANK_VARIANT and names == sorted(expected+['coupled_diffusion.bin']):
            expected = names
        if (selected not in {_GRADED_LIVE_BEDLOAD_VARIANT, _G1_VARIANT, _GENERAL_LIVE_BANK_VARIANT,
                             _GENERAL_LIVE_TRANSPORT_VARIANT} or
                names != expected):
            raise ValueError("GPU worker has an unsupported prepared-sidecar allowance")
    return set(policy.get(variant, ()))


def _source_archive_for_variant(scope: dict, worker: Path,
                                variant: str) -> Path | None:
    """Resolve a package-declared numerical bundle, never a host checkout path.

    The bundle's files are authenticated by the worker package manifest before
    this function is called.  Require its complete inventory to be declared,
    including non-executable manifests, so an injected JSON cannot alter the
    source selected for a GPU run.
    """
    policy = scope.get("source_archives_by_variant", {})
    if not isinstance(policy, dict) or any(
            selected not in {_GRADED_LIVE_BEDLOAD_VARIANT, _ON_DEMAND_AMR_VARIANT,
                             _ON_DEMAND_GENERAL_AMR_VARIANT}
            for selected in policy):
        raise ValueError("GPU worker has an unsupported source-archive policy")
    name = policy.get(variant)
    if variant in {_GRADED_LIVE_BEDLOAD_VARIANT, _ON_DEMAND_AMR_VARIANT,
                   _ON_DEMAND_GENERAL_AMR_VARIANT} and name is None:
        raise ValueError("GPU worker has no packaged numerical source archive")
    if name is None:
        return None
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name):
        raise ValueError("GPU worker has an invalid source-archive path")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or relative.as_posix() != name or
            any(part in ("", ".", "..") for part in relative.parts) or
            relative.parts[:2] != ("braids_gpu", "frozen_sources") or
            len(relative.parts) < 3):
        raise ValueError("GPU worker has an invalid source-archive path")
    archive = worker.joinpath(*relative.parts)
    if (archive.is_symlink() or not archive.is_dir() or
            not archive.resolve().is_relative_to(worker)):
        raise ValueError("GPU worker numerical source archive is missing or linked")
    files = _json_object(worker / "PACKAGE-MANIFEST.json").get("files")
    if not isinstance(files, dict):
        raise ValueError("GPU worker has no authenticated payload inventory")
    members = list(archive.rglob("*"))
    if (not members or not (archive / "runtime_manifest.json").is_file() or
            any(path.is_symlink() or (path.is_file() and
                path.relative_to(worker).as_posix() not in files)
                for path in members)):
        raise ValueError("GPU worker numerical source archive is not fully authenticated")
    if variant == _ON_DEMAND_AMR_VARIANT:
        if name not in {f"braids_gpu/frozen_sources/on_demand_amr_v{revision}" for revision in range(1,10)}:
            raise ValueError("GPU worker has an unsupported on-demand AMR archive")
        # This route authenticates/selects its package-local payload internally;
        # its CLI intentionally rejects arbitrary --gpu-source-archive overrides.
        return None
    if variant == _ON_DEMAND_GENERAL_AMR_VARIANT:
        if name not in {"braids_gpu/frozen_sources/on_demand_transport_amr_v1",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v2",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v3",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v4",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v5",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v6",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v7",
                        "braids_gpu/frozen_sources/on_demand_transport_amr_v8"}:
            raise ValueError("GPU worker has an unsupported general transport AMR archive")
        # The package-local AMR source selector authenticates this bundle and
        # does not accept a caller-supplied source override.
        return None
    return archive


def _verify_amr_shared_method_pair(root: Path, source: str, manifest: dict) -> None:
    """Dispatch only declared, authenticated bundle/revision combinations.

    The v7 directory contains two historical source-pair revisions. Select
    their exact manifest declaration, never retry another policy on failure.
    Package inventory authentication must precede this additional ABI check.
    """
    from .gpu_amr_method_pair import (
        verify_method_pair, REVISION, PREVIOUS_REVISION, LEGACY_REVISION)
    allowed = {
        'braids_gpu/frozen_sources/on_demand_amr_v6': {LEGACY_REVISION},
        'braids_gpu/frozen_sources/on_demand_amr_v7': {PREVIOUS_REVISION, REVISION},
        'braids_gpu/frozen_sources/on_demand_amr_v8': {REVISION},
        'braids_gpu/frozen_sources/on_demand_amr_v9': {REVISION},
    }
    pair = manifest.get('interface_source_pair')
    revision = pair.get('revision') if isinstance(pair, dict) else None
    if (source not in allowed or not isinstance(revision, str) or
            revision not in allowed[source]):
        raise ValueError('GPU AMR bundle/shared source pairing revision differs')
    verify_method_pair(root, root/source, manifest, revision=revision)


def _verify_recursive_adaptation_capability(
        scope: dict, variant: str = _ON_DEMAND_AMR_VARIANT) -> None:
    """An authenticated OFF-only worker must never acquire the ON closure."""
    bank_mode=0 if variant == _ON_DEMAND_GENERAL_AMR_VARIANT else 3
    expected=dict(schema=1,policy='full_leaf_steady_v1',bank_mode=bank_mode,
                  secondary_flow_model=2,boundary_modes=[9,4],section_controls=[0,1,2])
    policy=scope.get('recursive_bedload_adaptation_by_variant')
    selected=policy.get(variant) if isinstance(policy,dict) else None
    if (not isinstance(policy,dict) or
            variant not in {_ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT} or
            not {variant} <= set(policy) <= {
                _ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT} or
            not isinstance(selected,dict) or selected!=expected or
            any(type(selected[key]) is not type(value) for key,value in expected.items()) or
            any(type(value) is not int for key in ('boundary_modes','section_controls')
                for value in selected[key])):
        raise ValueError('GPU AMR worker lacks explicit full-leaf recursive bedload adaptation capability')


def _verify_recursive_general_avalanche_capability(scope: dict, variant: str) -> None:
    """Require an explicit ON closure without invalidating historical OFF workers."""
    expected=dict(schema=1,policy=_RECURSIVE_GENERAL_AVALANCHE_POLICY,
        bank_mode=0,secondary_flow_model=2,boundary_modes=[9,4],
        checkpoint_schema=6)
    policies=scope.get('recursive_general_avalanche_by_variant')
    selected=policies.get(variant) if isinstance(policies,dict) else None
    if (variant != _ON_DEMAND_GENERAL_AMR_VARIANT or
            not isinstance(policies,dict) or set(policies)!={variant} or
            not isinstance(selected,dict) or selected!=expected or
            any(type(selected[key]) is not type(value)
                for key,value in expected.items()) or
            any(type(value) is not int for value in selected['boundary_modes'])):
        raise ValueError('GPU AMR worker lacks explicit recursive general-avalanche capability')


def _verify_recursive_general_viscosity_capability(scope: dict, variant: str) -> None:
    """Require the exact positive-viscosity closure; zero stays historical."""
    expected=dict(schema=1,bank_mode=0,secondary_flow_model=2,
        boundary_modes=[9,4],viscosity_mode=1,viscosity_geometry=1,
        formulation='velocity_laplacian',checkpoint_schema=6,
        **_RECURSIVE_GENERAL_VISCOSITY_METHOD)
    policies=scope.get('recursive_general_viscosity_by_variant')
    selected=policies.get(variant) if isinstance(policies,dict) else None
    if (variant != _ON_DEMAND_GENERAL_AMR_VARIANT or
            not isinstance(policies,dict) or set(policies)!={variant} or
            not isinstance(selected,dict) or selected!=expected or
            any(type(selected[key]) is not type(value)
                for key,value in expected.items()) or
            any(type(value) is not int for value in selected['boundary_modes']) or
            any(type(value) is not int
                for value in selected['general_avalanche_modes'])):
        raise ValueError('GPU AMR worker lacks explicit recursive general-viscosity capability')


def _verify_recursive_general_bedform_capability(scope: dict, variant: str) -> None:
    """Require the exact equilibrium-bedform branch for its ON/OFF pair."""
    expected=dict(schema=1,bank_mode=0,boundary_mode_counts={'1':1,'2':1},
        bedform_modes=[0,1],unsampled_zero_offset=True,scheduled_forcing=False,bedlaw=2,
        mpm_friction_mode=1,bed_slope=0,hiding_exposure=0,exchange_mode=0,
        secondary_flow_model=0,bedload_adaptation=0,horizontal_viscosity=0.,
        general_avalanche=0,morphological_factor=1.,nsubstrate_layers=1,
        checkpoint_schema=6,**_RECURSIVE_GENERAL_BEDFORM_METHOD)
    policies=scope.get('recursive_general_bedform_by_variant')
    selected=policies.get(variant) if isinstance(policies,dict) else None
    if (variant != _ON_DEMAND_GENERAL_AMR_VARIANT or
            not isinstance(policies,dict) or set(policies)!={variant} or
            not isinstance(selected,dict) or selected!=expected or
            any(type(selected[key]) is not type(value)
                for key,value in expected.items()) or
            any(type(key) is not str or type(value) is not int
                for key,value in selected.get('boundary_mode_counts',{}).items()) or
            any(type(value) is not int
                for value in selected.get('bedform_modes',()))):
        raise ValueError('GPU AMR worker lacks explicit recursive general-bedform capability')


def _verify_recursive_bank_bedform_avalanche_capability(
        scope: dict, variant: str = _ON_DEMAND_AMR_VARIANT,
        bank_mode: int = 3) -> None:
    """Require the exact recursive composition for the prepared bank mode."""
    from .gpu_on_demand_frame_io import _typed_equal
    common=dict(schema=1,bank_mode=bank_mode,evolving_banks=True,
        boundary_mode_counts={'2':1},unsampled_zero_offset=True,
        scheduled_forcing=False,bank_failure=1,bedform_roughness=1,
        general_avalanche=1,bedlaw=2,mpm_friction_mode=1,bed_slope=0,
        hiding_exposure=0,exchange_mode=0,secondary_flow_model=0,
        bedload_adaptation=0,horizontal_viscosity=0.,morphological_factor=1.,
        nsubstrate_layers=1,checkpoint_kind='recursive_bank_amr')
    if bank_mode == 2:
        expected=dict(common, toe_geometry_seed_only=True,
            checkpoint_schema=2,frame_schema=8,
            **_RECURSIVE_PHYSICAL_TOE_BEDFORM_METHOD)
        policy_key='recursive_physical_toe_bedform_avalanche_by_variant'
        exporter_key=\
            'recursive_physical_toe_bedform_native_exporters_by_variant'
        runtime=_BEDFORM_PHYSICAL_TOE_RUNTIME
    elif bank_mode == 3:
        expected=dict(common, checkpoint_schema=1,frame_schema=7,
            **_RECURSIVE_BANK_BEDFORM_METHOD)
        policy_key='recursive_bank_bedform_avalanche_by_variant'
        exporter_key='recursive_bank_bedform_native_exporters_by_variant'
        runtime=_BEDFORM_BANK_RUNTIME
    else:
        raise ValueError('Recursive bank bedform capability has invalid bank mode')
    policies=scope.get(policy_key)
    selected=policies.get(variant) if isinstance(policies,dict) else None
    initializers=scope.get(exporter_key)
    if (variant != _ON_DEMAND_AMR_VARIANT or
            not isinstance(policies,dict) or set(policies)!={variant} or
            not _typed_equal(selected,expected) or
            initializers != {variant:runtime}):
        raise ValueError('GPU AMR worker lacks explicit recursive bank bedform/avalanche capability')


def _verify_recursive_amr_capability(worker: dict, prepared: dict) -> None:
    """Older synchronous packages cannot acquire subcycling from new inputs."""
    subcycling=prepared.get('amr_settings',{}).get('subcycling',False)
    if type(subcycling) is not bool:
        raise ValueError('GPU AMR subcycling selection must be an authenticated boolean')
    if not subcycling:
        return
    variant=prepared.get('variant',
        _ON_DEMAND_AMR_VARIANT if 'bank_mode' in prepared else None)
    from .gpu_on_demand_prepared import (
        _recursive_adaptation_scope, _recursive_general_avalanche_scope,
        _recursive_general_viscosity_scope, _recursive_general_bedform_scope,
        _recursive_bank_bedform_avalanche_scope)
    if _recursive_adaptation_scope(prepared.get('initial_data',{})):
        _verify_recursive_adaptation_capability(worker['scope'],variant)
    avalanche=(variant == _ON_DEMAND_GENERAL_AMR_VARIANT and
               _recursive_general_avalanche_scope(
                   prepared.get('initial_data',{})))
    if avalanche:
        _verify_recursive_general_avalanche_capability(worker['scope'],variant)
    viscosity=(variant == _ON_DEMAND_GENERAL_AMR_VARIANT and
               _recursive_general_viscosity_scope(
                   prepared.get('initial_data',{})))
    if viscosity:
        _verify_recursive_general_viscosity_capability(worker['scope'],variant)
    bedform=(variant == _ON_DEMAND_GENERAL_AMR_VARIANT and
             _recursive_general_bedform_scope(
                 prepared.get('initial_data',{})))
    if bedform:
        _verify_recursive_general_bedform_capability(worker['scope'],variant)
    bank_bedform=False
    if variant == _ON_DEMAND_AMR_VARIANT:
        initial=prepared.get('initial_data',{})
        flag=initial.get('bedform_roughness',0)
        if type(flag) is not int or flag not in (0,1):
            raise ValueError('Recursive bank bedform switch differs')
        bank_bedform=bool(flag)
        if bank_bedform:
            _recursive_bank_bedform_avalanche_scope(initial)
            _verify_recursive_bank_bedform_avalanche_capability(
                worker['scope'],variant,
                bank_mode=initial.get('bank_toe_erosion_mode'))
    policy=worker['scope'].get('recursive_amr_by_variant')
    selected=policy.get(variant) if isinstance(policy,dict) else None
    if not isinstance(policy,dict) or not isinstance(selected,dict):
        raise ValueError('GPU AMR worker lacks the explicit recursive capability')
    if variant == _ON_DEMAND_GENERAL_AMR_VARIANT:
        expected=dict(schema=1,bank_mode=0,stepper=_RECURSIVE_STEPPER,
            checkpoint_kind='recursive_general_amr',checkpoint_schema=6)
        if (selected != expected or
                any(type(selected[key]) is not type(value)
                    for key,value in expected.items()) or
                type(prepared.get('bank_mode')) is not int or
                prepared['bank_mode'] != 0 or
                prepared.get('evolving_banks') is not False or
                prepared.get('general_transport') is not True or
                prepared.get('checkpoint_schema') != _RECURSIVE_GENERAL_CHECKPOINT or
                prepared.get('session_manifest',{}).get('schema') != 3):
            raise ValueError('GPU AMR worker lacks the explicit recursive general capability')
        source=worker['scope'].get('source_archives_by_variant',{}).get(variant)
        allowed_sources=({
                'braids_gpu/frozen_sources/on_demand_transport_amr_v7',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v8'}
            if bedform else {
                'braids_gpu/frozen_sources/on_demand_transport_amr_v6',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v7',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v8'}
            if viscosity else {
                'braids_gpu/frozen_sources/on_demand_transport_amr_v4',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v5',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v6',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v7',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v8'}
            if not avalanche else {
                'braids_gpu/frozen_sources/on_demand_transport_amr_v5',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v6',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v7',
                'braids_gpu/frozen_sources/on_demand_transport_amr_v8'})
        if source not in allowed_sources:
            raise ValueError('Recursive general GPU AMR requires its current source bundle')
        files=_json_object(worker['root']/'PACKAGE-MANIFEST.json').get('files',{})
        modules=('amr_general_recursive_route','amr_general_subcycle_stepper',
                 'amr_general_subcycle_register','amr_recursive_engine',
                 'amr_general_checkpoint')
        required={'braids_gpu/'+name+'.py' for name in modules}
        required.add('braids_gpu/amr_bank_subcycle.cu')
        if avalanche:
            required|={
                'validation/gpu_fwave/avalanche_operator.py',
                'validation/gpu_fwave/avalanche_proposal.py',
                'validation/gpu_fwave/avalanche_proposal.cu',
                'validation/gpu_fwave/surface_transfer.py',
                'validation/gpu_fwave/surface_transfer.cu',
            }
        if viscosity:
            required|={
                'validation/gpu_fwave/momentum_diffusion.py',
                'validation/gpu_fwave/momentum_diffusion.cu',
                'validation/gpu_fwave/momentum_geometry.py',
                'validation/gpu_fwave/momentum_geometry.cu',
            }
        if bedform:
            required|={
                'validation/gpu_fwave/bedform_roughness.py',
                'validation/gpu_fwave/bedform_roughness.cu',
            }
        if not isinstance(files,dict) or not required<=files.keys():
            raise ValueError('Recursive general GPU AMR worker has an incomplete component inventory')
        return
    if variant != _ON_DEMAND_AMR_VARIANT:
        raise ValueError('GPU AMR worker has no recursive capability for this route')
    bank_mode=prepared.get('bank_mode')
    evolving=prepared.get('evolving_banks')
    session_schema=prepared.get('session_manifest',{}).get('schema')
    legacy=dict(schema=1,bank_mode=3,stepper=_RECURSIVE_STEPPER,
        checkpoint_kind='recursive_bank_amr',checkpoint_schema=1)
    if selected == legacy:
        if (type(bank_mode) is not int or bank_mode!=3 or evolving is not True or
                session_schema!=2):
            raise ValueError('Legacy recursive capability admits only explicit evolving mode 3')
        source_expected='braids_gpu/frozen_sources/on_demand_amr_v7'
    else:
        routes=[
            dict(bank_mode=2,evolving_banks=False,
                 checkpoint_kind='recursive_bank_amr',checkpoint_schema=2),
            dict(bank_mode=2,evolving_banks=True,
                 checkpoint_kind='recursive_bank_amr',checkpoint_schema=2),
            dict(bank_mode=3,evolving_banks=True,
                 checkpoint_kind='recursive_bank_amr',checkpoint_schema=1),
        ]
        expected=dict(schema=2,stepper=_RECURSIVE_STEPPER,routes=routes)
        route=dict(bank_mode=bank_mode,evolving_banks=evolving,
                   checkpoint_kind='recursive_bank_amr',
                   checkpoint_schema=2 if bank_mode==2 else 1)
        if (selected != expected or
                type(bank_mode) is not int or type(evolving) is not bool or
                route not in routes or session_schema != (2 if evolving else 1) or
                prepared.get('checkpoint_schema') != _RECURSIVE_BANK_CHECKPOINT):
            raise ValueError('GPU AMR worker lacks the selected recursive bank capability')
        source_expected=('braids_gpu/frozen_sources/on_demand_amr_v9'
                         if bank_bedform else None)
    source=worker['scope'].get('source_archives_by_variant',{}).get(_ON_DEMAND_AMR_VARIANT)
    allowed_sources=({source_expected} if source_expected is not None else {
        'braids_gpu/frozen_sources/on_demand_amr_v8',
        'braids_gpu/frozen_sources/on_demand_amr_v9'})
    if source not in allowed_sources:
        raise ValueError('Recursive GPU AMR requires the current paired worker bundle')
    from .gpu_amr_method_pair import REVISION
    manifest=_json_object(worker['root']/source/'runtime_manifest.json')
    pair=manifest.get('interface_source_pair')
    if not isinstance(pair,dict) or pair.get('revision')!=REVISION:
        raise ValueError('Recursive GPU AMR requires the current secondary source pair')
    files=_json_object(worker['root']/'PACKAGE-MANIFEST.json').get('files',{})
    modules=('amr_recursive_route','amr_bank_subcycle','amr_boundary_forcing',
             'amr_level_euler','amr_bank_repair','amr_bank_wet','amr_level_secondary')
    if selected != legacy:
        modules=(*modules,'amr_recursive_engine')
    kernels=('amr_bank_subcycle','amr_level_euler','amr_bank_repair',
             'amr_bank_wet','amr_level_secondary')
    required={'braids_gpu/'+name+'.py' for name in modules}
    required|={'braids_gpu/'+name+'.cu' for name in kernels}
    if bank_bedform:
        required|={
            'validation/gpu_fwave/bedform_roughness.py',
            'validation/gpu_fwave/bedform_roughness.cu',
            'validation/gpu_fwave/avalanche_operator.py',
            'validation/gpu_fwave/avalanche_proposal.py',
            'validation/gpu_fwave/avalanche_proposal.cu',
            'validation/gpu_fwave/surface_transfer.py',
            'validation/gpu_fwave/surface_transfer.cu'}
    if not isinstance(files,dict) or not required<=files.keys():
        raise ValueError('Recursive GPU AMR worker has an incomplete component inventory')


def _verify_on_demand_runtime(worker: dict, prepared: dict) -> None:
    """Bind a prepared session to the selected package's initializer, host-only.

    Package inventory verification has already authenticated the executable and
    DLLs. No recorded checkout runtime or compiler is imported/executed here.
    """
    # Historical bank preparations did not persist a variant discriminator.
    # Only the explicit general value selects the newer nonbank route.
    variant = prepared.get('variant', _ON_DEMAND_AMR_VARIANT)
    if variant == _ON_DEMAND_GENERAL_AMR_VARIANT:
        relative = "braids_gpu/native_amr_general_v1"
        _verify_recursive_amr_capability(worker,prepared)
        selector_policy = prepared.get('selector_policy')
        verify_general_amr_selector_scope(worker['scope'],
            _ON_DEMAND_GENERAL_AMR_VARIANT, selector_policy)
        source = worker['scope'].get('source_archives_by_variant', {}).get(
            _ON_DEMAND_GENERAL_AMR_VARIANT)
        policy = worker['scope'].get('native_exporters_by_variant', {})
        recursive=prepared.get('amr_settings',{}).get('subcycling',False)
        expected_checkpoint=(_RECURSIVE_GENERAL_CHECKPOINT if recursive else 5)
        if (prepared.get('general_transport') is not True or
                prepared.get('evolving_banks') is not False or
                prepared.get('bank_mode') != 0 or
                prepared.get('checkpoint_schema') != expected_checkpoint or
                prepared.get('session_manifest', {}).get('schema') != 3 or
                not isinstance(policy, dict) or
                policy.get(_ON_DEMAND_GENERAL_AMR_VARIANT) != relative or
                source not in ('braids_gpu/frozen_sources/on_demand_transport_amr_v1',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v2',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v3',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v4',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v5',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v6',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v7',
                               'braids_gpu/frozen_sources/on_demand_transport_amr_v8')):
            raise ValueError('GPU worker lacks the general transport AMR contract')
        if selector_policy is not None:
            sources = prepared.get('selector_source_sha256')
            if (not isinstance(sources, dict) or
                    set(sources) != _GENERAL_AMR_SLOPE_SOURCES or
                    any(_sha256(worker['root']/'braids_gpu'/name) != sources[name]
                        for name in _GENERAL_AMR_SLOPE_SOURCES)):
                raise ValueError('Slope-aware selector source differs from the packaged worker')
        runtime = worker['root']/relative
        if (runtime.is_symlink() or
                prepared['runtime_location'].resolve() != runtime.resolve() or
                _sha256(runtime/'build.json') != prepared['runtime_build_sha256']):
            raise ValueError('General AMR preparation uses a different packaged initializer')
        manifest = _json_object(worker['root']/source/'runtime_manifest.json')
        if (manifest.get('schema') != 1 or
                manifest.get('variant') != _ON_DEMAND_GENERAL_AMR_VARIANT or
                manifest.get('native_preparation_schema') != 3):
            raise ValueError('General AMR source archive contract differs')
        return
    if variant != _ON_DEMAND_AMR_VARIANT:
        raise ValueError('Unsupported on-demand AMR prepared route')
    _verify_recursive_amr_capability(worker,prepared)
    bedform_bank=(prepared.get('session_manifest',{}).get(
        'recursive_bank_bedform_avalanche') is True)
    relative = "braids_gpu/native_amr_v1"
    live_export=prepared['runtime_build'].get('live_boundary_metadata') is True
    if bedform_bank:
        if prepared.get('bank_mode') == 2:
            relative=_BEDFORM_PHYSICAL_TOE_RUNTIME
        elif prepared.get('bank_mode') == 3:
            relative=_BEDFORM_BANK_RUNTIME
        else:
            raise ValueError('Recursive bank bedform preparation has invalid bank mode')
    elif live_export:
        relative="braids_gpu/native_amr_live_v1"
    policy_key = (
        'recursive_physical_toe_bedform_native_exporters_by_variant'
        if bedform_bank and prepared.get('bank_mode') == 2 else
        'recursive_bank_bedform_native_exporters_by_variant'
        if bedform_bank else
        'evolving_native_exporters_by_variant' if live_export else
        'native_exporters_by_variant')
    policy = worker["scope"].get(policy_key, {})
    if not isinstance(policy, dict) or policy.get(_ON_DEMAND_AMR_VARIANT) != relative:
        raise ValueError("GPU AMR worker has no supported packaged initializer")
    runtime = worker["root"] / relative
    if (runtime.is_symlink() or prepared["runtime_location"].resolve() != runtime.resolve() or
            _sha256(runtime / "build.json") != prepared["runtime_build_sha256"]):
        raise ValueError("GPU AMR preparation uses a different packaged initializer")
    if prepared['session_manifest']['schema']==2:
        source=worker['scope'].get('source_archives_by_variant',{}).get(_ON_DEMAND_AMR_VARIANT)
        if source not in ('braids_gpu/frozen_sources/on_demand_amr_v5',
                          'braids_gpu/frozen_sources/on_demand_amr_v6',
                          'braids_gpu/frozen_sources/on_demand_amr_v7',
                          'braids_gpu/frozen_sources/on_demand_amr_v8',
                          'braids_gpu/frozen_sources/on_demand_amr_v9'):
            raise ValueError('Evolving AMR preparation requires a supported versioned worker source contract')
        manifest=_json_object(worker['root']/source/'runtime_manifest.json')
        if source.endswith(('/on_demand_amr_v6','/on_demand_amr_v7',
                            '/on_demand_amr_v8','/on_demand_amr_v9')):
            _verify_amr_shared_method_pair(worker['root'],source,manifest)
        else:
            # Retain the older v5 workflow, but do not let its lack of the
            # later ABI checks admit a mislabeled new shared implementation.
            pair=manifest.get('interface_source_pair')
            if (not isinstance(pair,dict) or
                    pair.get('revision')!='inventory_bounded_geometry_endpoint_v4'):
                raise ValueError('GPU AMR version-5 source pairing revision differs')
        required={'evolving_toe_map.cu','amr_evolving_history.cu',
                  'amr_evolving_support.cu','amr_general_indicators.cu'}
        pair=manifest.get('evolving_interface_source_pair',{})
        revision=pair.get('revision')
        versions={'evolving_bank_on_demand_v1':1,
                  'evolving_bank_on_demand_indicators_v2':2}
        abi=versions.get(revision)
        expected_abis={
            'evolving_toe_map.cu':'#define BRAIDS_EVOLVING_TOE_ABI 2',
            'amr_evolving_history.cu':'#define BRAIDS_AMR_EVOLVING_HISTORY_ABI 1',
            'amr_evolving_support.cu':'#define BRAIDS_AMR_EVOLVING_SUPPORT_ABI 1',
            'amr_general_indicators.cu':f'#define BRAIDS_AMR_GENERAL_INDICATORS_ABI {abi}'}
        python_files={
            'validation/gpu_fwave/evolving_toe_map.py',
            'validation/gpu_fwave/amr_evolving_history.py',
            'validation/gpu_fwave/amr_general_indicators.py',
            'braids_gpu/amr_evolving_bank.py','braids_gpu/amr_face_correspondence.py',
            'braids_gpu/amr_general_selector.py','braids_gpu/amr_on_demand.py',
            'braids_gpu/amr_catalog_bank.py','braids_gpu/amr_catalog_bank_checkpoint.py'}
        if abi==2:
            python_files.add('braids_gpu/amr_indicator_reference.py')
        extended=any(prepared.get('amr_settings',{}).get(key,False) for key in
                     ('refine_walls','refine_bathymetry','refine_sediment'))
        if (manifest.get('added_cuda_kernels')!=sorted(required) or
                abi is None or (extended and abi!=2) or pair.get('schema')!=1 or
                set(pair)!={'schema','revision','required_cuda_abi','cuda_sha256','python_sha256'} or
                pair.get('required_cuda_abi')!=expected_abis or
                set(pair.get('cuda_sha256',{}))!=required or
                set(pair.get('python_sha256',{}))!=python_files):
            raise ValueError('GPU AMR worker lacks the evolving bank source pairing')
        for filename in sorted(required):
            path=worker['root']/source/filename
            if (_sha256(path)!=pair['cuda_sha256'][filename] or
                    sum(line.strip()==expected_abis[filename] for line in
                        path.read_text(encoding='utf-8').splitlines())!=1):
                raise ValueError('GPU AMR worker kernel pairing differs: '+filename)
        for filename in sorted(python_files):
            if _sha256(worker['root']/filename)!=pair['python_sha256'][filename]:
                raise ValueError('GPU AMR worker Python pairing differs: '+filename)


def _verify_amr_prepared_input(root: Path) -> str:
    """Authenticate the complete bank-AMR package before worker creation.

    The topology reader checks every declared artifact, tree, selected-node
    cover, geometry and receipt.  Schema 1 is a different research route and
    cannot be silently promoted by the bank-AMR worker variant.
    """
    prepared = inspect_gpu_amr_prepared(root)
    if _json_object(root / "amr_prepared.json").get("schema") != 2:
        raise ValueError("Bank-AMR worker requires schema-2 prepared input")
    return prepared["manifest_sha256"]


def _verify_frozen_source(manifest_path: Path) -> str:
    """Verify the numerical files behind a saved source-manifest digest."""
    if (manifest_path.parent.is_symlink() or manifest_path.is_symlink() or
            not manifest_path.is_file()):
        raise ValueError("GPU frozen source manifest is missing or linked")
    manifest = _json_object(manifest_path)
    files = manifest.get("source_files", manifest)
    if not isinstance(files, dict) or not files:
        raise ValueError("GPU frozen source inventory is missing")
    source_root = manifest_path.parent.resolve()
    for name, expected in files.items():
        member = PurePosixPath(name) if isinstance(name, str) else None
        if (member is None or not name or "\\" in name or ":" in name or
                member.is_absolute() or member.as_posix() != name or
                not member.parts or any(part in (".", "..") for part in member.parts) or
                not isinstance(expected, str) or len(expected) != 64 or
                any(ch not in "0123456789abcdef" for ch in expected)):
            raise ValueError("GPU frozen source inventory has an invalid member")
        path = manifest_path.parent.joinpath(*member.parts)
        if (any(parent.is_symlink() for parent in
                (path, *tuple(path.parents)[:len(member.parts) - 1])) or
                not path.resolve().is_relative_to(source_root) or
                not path.is_file() or _sha256(path) != expected):
            raise ValueError("GPU frozen source file differs: " + name)
    return _sha256(manifest_path)


def _checkpoint_scalar(archive, name: str):
    if name not in archive.files:
        raise ValueError("GPU checkpoint is missing " + name)
    value = archive[name]
    if value.shape != ():
        raise ValueError("GPU checkpoint has non-scalar " + name)
    return value.item()


def _amr_fields_digest(fields: dict[str, np.ndarray]) -> str:
    """Independent host implementation of the accepted-field byte digest."""
    digest = hashlib.sha256()
    for name in sorted(fields):
        value = np.ascontiguousarray(fields[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


_G1_CHECKPOINT_FIELDS = frozenset({
    "q", "layers", "secondary", "bed_elevation", "ledger_water", "ledger_bed",
    "ledger_secondary", "ledger_source", "time", "steps", "rejected", "dt_hint",
    "input_identity", "source_identity", "method_identity", "event_identity",
    "next_boundary_knot", "compact_checkpoint_schema", "g1_payload_sha256",
})


def _verify_g1_checkpoint(checkpoint: Path, manifest: dict,
                          final_frame: Path | None = None) -> None:
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("GPU inlet checkpoint is missing or linked")
    with np.load(checkpoint, allow_pickle=False) as archive:
        if (len(archive.files) != len(_G1_CHECKPOINT_FIELDS) or
                set(archive.files) != _G1_CHECKPOINT_FIELDS):
            raise ValueError("GPU inlet checkpoint field inventory differs")
        fields = {name: archive[name] for name in archive.files}
    payload = fields.pop("g1_payload_sha256")
    if (payload.shape != () or payload.dtype.kind != "U" or
            payload.item() != _amr_fields_digest(fields) or
            fields["compact_checkpoint_schema"].shape != () or
            fields["compact_checkpoint_schema"].dtype != np.int32 or
            int(fields["compact_checkpoint_schema"]) != 1):
        raise ValueError("GPU inlet checkpoint payload or schema differs")
    for name, expected in (("input_identity", manifest.get("input_identity")),
                           ("source_identity", manifest.get("source_identity")),
                           ("method_identity", manifest.get("method_identity")),
                           ("event_identity", (manifest.get("boundary_events") or {}).get("identity"))):
        if (fields[name].shape != () or fields[name].dtype.kind != "U" or
                fields[name].item() != expected):
            raise ValueError("GPU inlet checkpoint identity differs: " + name)
    if (fields["time"].shape != () or fields["time"].dtype != np.float64 or
            fields["steps"].shape != () or fields["steps"].dtype != np.int64 or
            not math.isclose(float(fields["time"]), float(manifest.get("time_s")),
                             rel_tol=0, abs_tol=1e-12) or
            int(fields["steps"]) != manifest.get("steps")):
        raise ValueError("GPU inlet checkpoint terminal clock differs")
    if final_frame is not None:
        with np.load(final_frame, allow_pickle=False) as archive:
            if (set(archive.files) != _G1_CHECKPOINT_FIELDS or
                    any(not np.array_equal(archive[name], fields[name])
                        for name in fields) or
                    not np.array_equal(archive["g1_payload_sha256"], payload)):
                raise ValueError("GPU inlet checkpoint differs from final accepted frame")


def _amr_array(fields: dict, name: str, shape: tuple[int, ...], dtype,
               *, nonnegative: bool = False) -> np.ndarray:
    value = fields.get(name)
    if (value is None or value.shape != shape or value.dtype != np.dtype(dtype) or
            (value.dtype.kind == "f" and not np.isfinite(value).all()) or
            (nonnegative and np.any(value < 0))):
        raise ValueError("GPU AMR checkpoint has invalid " + name)
    return value


def _amr_text(fields: dict, name: str, expected: str | None = None) -> str:
    value = fields.get(name)
    if (value is None or value.shape != () or value.dtype.kind not in "US" or
            (expected is not None and value.item() != expected)):
        raise ValueError("GPU AMR checkpoint identity differs: " + name)
    return str(value.item())


def _amr_json(fields: dict, name: str):
    raw = _amr_text(fields, name)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU AMR checkpoint has invalid JSON: " + name) from exc
    if json.dumps(value, sort_keys=True, separators=(",", ":"),
                  allow_nan=False) != raw:
        raise ValueError("GPU AMR checkpoint has noncanonical JSON: " + name)
    return value


def _amr_toe_masks(index: dict) -> dict[str, np.ndarray]:
    """Read the already authenticated bank-toe sidecar for each topology."""
    manifest = _json_object(index["root"] / "amr_prepared.json")
    rows = manifest["topologies"]
    if len(rows) != len(index["topologies"]):
        raise ValueError("GPU AMR checkpoint topology inventory differs")
    masks = {}
    for row in rows:
        name = row["name"]
        topology = index["topologies"][name]
        sidecars = row.get("sidecars")
        relative = sidecars.get("bank_toe") if isinstance(sidecars, dict) else None
        if (not isinstance(relative, str) or
                relative != (topology["folder"] / "bank_toe_metadata.bin").relative_to(
                    index["root"]).as_posix() or
                relative not in manifest["artifacts"]):
            raise ValueError("GPU AMR checkpoint lacks authenticated bank-toe mapping")
        toe = _read_records(index["root"] / relative)
        nf = int(topology["initial_data"]["nf"])
        value = np.asarray(toe.get("is_toe_face"))
        if (value.shape != (nf,) or value.dtype != np.dtype("int32") or
                not np.isin(value, (0, 1)).all()):
            raise ValueError("GPU AMR checkpoint has invalid prepared toe-face mask")
        masks[name] = value.astype(bool)
    return masks


def _amr_toe_diagnostics(fields: dict, prefix: str, mask: np.ndarray) -> None:
    shape = (2, len(mask))
    signed = _amr_array(fields, prefix + "signed", shape, "float64")
    gross = _amr_array(fields, prefix + "gross", shape, "float64",
                       nonnegative=True)
    if (np.any(gross + 1e-12 < np.abs(signed)) or
            np.any(signed[:, ~mask] != 0) or np.any(gross[:, ~mask] != 0)):
        raise ValueError("GPU AMR checkpoint toe-face diagnostics differ")


def _verify_amr_checkpoint(root: Path, result: dict) -> None:
    """Verify the complete v1 bank-AMR restart state, not merely a result frame.

    This is a *future* public-worker ABI.  No current source package advertises
    it.  A worker must atomically write and inventory all accepted state,
    ledgers, history, face epochs, regrid events and transfer cache.  Unknown
    checkpoint schemas/fields fail closed so a pretty frame cannot masquerade
    as a recoverable calculation.
    """
    manifest = result["manifest"]
    index = result["prepared_index"]
    frames = result["frames"]
    names = list(index["topologies"])
    method = manifest.get("method")
    if (not isinstance(method, dict) or method.get("variant") != _BANK_AMR_VARIANT or
            method.get("integrator") != "SSPRK33" or
            method.get("launch_mode") != "plain"):
        raise ValueError("GPU AMR checkpoint method is not the declared bank route")
    checkpoint_record = manifest.get("checkpoint")
    checkpoint = root / "checkpoint.npz"
    if (not isinstance(checkpoint_record, dict) or
            checkpoint_record.get("file") != "checkpoint.npz" or
            not isinstance(checkpoint_record.get("sha256"), str) or
            len(checkpoint_record["sha256"]) != 64 or
            any(ch not in "0123456789abcdef" for ch in checkpoint_record["sha256"]) or
            checkpoint.is_symlink() or not checkpoint.is_file() or
            _sha256(checkpoint) != checkpoint_record["sha256"]):
        raise ValueError("GPU AMR final checkpoint is absent or differs")
    with np.load(checkpoint, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    digest = fields.pop("checkpoint_payload_sha256", None)
    if (digest is None or digest.shape != () or digest.dtype.kind not in "US" or
            digest.item() != _amr_fields_digest(fields) or
            digest.item() != checkpoint_record.get("payload_sha256")):
        raise ValueError("GPU AMR checkpoint payload digest differs")
    if int(_amr_array(fields, "checkpoint_schema", (), "int64")) != 1:
        raise ValueError("Unsupported GPU AMR checkpoint version")
    _amr_text(fields, "checkpoint_kind", _BANK_AMR_CHECKPOINT_KIND)
    for key, expected in (("input_identity", manifest["input_preparation_sha256"]),
                          ("source_identity", manifest["source_identity"]),
                          ("method_identity", manifest["method_identity"]),
                          ("final_frame_sha256", manifest["frames"][-1]["sha256"])):
        _amr_text(fields, key, expected)
    final = frames[-1]
    reported_steps = manifest.get("steps")
    reported_time = manifest.get("final_time_s")
    status = result["status"]
    if (type(reported_steps) is not int or reported_steps != final["steps"] or
            type(reported_time) not in (float, int) or
            not math.isclose(reported_time, final["time_s"], rel_tol=0,
                             abs_tol=1e-12) or
            status.get("steps") != final["steps"] or
            type(status.get("time_s")) not in (float, int) or
            not math.isclose(status["time_s"], final["time_s"], rel_tol=0,
                             abs_tol=1e-12)):
        raise ValueError("GPU AMR checkpoint terminal status/frame clock differs")
    name = _amr_text(fields, "topology_name", final["topology_name"])
    topology = index["topologies"][name]
    _amr_text(fields, "operator_identity", topology["operator_sha256"])
    if int(_amr_array(fields, "regrid_mesh_index", (), "int64")) != names.index(name):
        raise ValueError("GPU AMR checkpoint selected mesh index differs")
    n = len(topology["node_ids"])
    initial = topology["initial_data"]
    nfrac = int(initial["nfrac"])
    nlayer = int(initial["nbed_layers"])
    nb = int(initial["nb"])
    nf = int(initial["nf"])
    nstate = int(initial["nstate"])
    if (nstate != 4 + nfrac * (1 + nlayer) or
            int(initial["secondary_index"]) != nstate - 1):
        raise ValueError("GPU AMR checkpoint numerical state layout differs")
    state = _amr_array(fields, "state", (n, nstate), "float64")
    node_ids = _amr_array(fields, "node_ids", (n,), "int64")
    if (not np.array_equal(node_ids, topology["node_ids"]) or
            np.any(state[:, 0] < -1e-10) or
            np.any(state[:, 3:nstate-1] < -1e-10) or
            np.any(state[:, 3:3+nfrac] != 0) or
            not np.array_equal(state[:, :nstate-1], final["q"]) or
            not np.array_equal(state[:, nstate-1], final["secondary"])):
        raise ValueError("GPU AMR checkpoint state differs from final accepted frame")
    steps = int(_amr_array(fields, "steps", (), "int64"))
    clock = float(_amr_array(fields, "time_s", (), "float64"))
    rejected = int(_amr_array(fields, "rejected", (), "int64"))
    hint = float(_amr_array(fields, "dt_hint", (), "float64"))
    cfl = float(_amr_array(fields, "maximum_cfl", (), "float64"))
    euler_calls = int(_amr_array(fields, "rk_euler_calls", (), "int64"))
    if (steps != final["steps"] or
            not math.isclose(clock, final["time_s"], rel_tol=0, abs_tol=1e-12) or
            rejected < 0 or hint <= 0 or cfl < 0 or
            cfl > float(initial["cfl_target"]) + 1e-12 or
            euler_calls < 3 * steps):
        raise ValueError("GPU AMR checkpoint accepted clock/timestep differs")
    for key, shape in (("ledger_water", (nb,)),
                       ("ledger_bed", (nfrac, nb)),
                       ("ledger_secondary", (nb,)),
                       ("ledger_source", ())):
        value = _amr_array(fields, key, shape, "float64")
        if not np.array_equal(value, final["ledger"][key]):
            raise ValueError("GPU AMR checkpoint ledger differs: " + key)
    bank_history = final["bank_history"]
    if bank_history is None:
        raise ValueError("GPU AMR bank checkpoint requires complete frame histories")
    for key in _BANK_HISTORY_FLOAT:
        value = _amr_array(fields, key, (n,), "float64", nonnegative=True)
        if not np.array_equal(value, bank_history[key]):
            raise ValueError("GPU AMR checkpoint bank history differs: " + key)
    counts = _amr_array(fields, "bank_history_collapse_count", (n,), "int32",
                        nonnegative=True)
    if not np.array_equal(counts, bank_history["bank_history_collapse_count"]):
        raise ValueError("GPU AMR checkpoint collapse history differs")
    masks = _amr_toe_masks(index)
    _amr_toe_diagnostics(fields, "toe_", masks[name])
    policy = _amr_json(fields, "regrid_policy_json")
    if (not isinstance(policy, dict) or policy != method.get("regrid_policy") or
            set(policy) != {"min_dwell_steps", "coarsen_spread_m", "refine_jump_m"} or
            type(policy["min_dwell_steps"]) is not int or
            policy["min_dwell_steps"] < 0 or
            any(type(policy[key]) not in (float, int) or not math.isfinite(policy[key]) or
                policy[key] <= 0 for key in ("coarsen_spread_m", "refine_jump_m"))):
        raise ValueError("GPU AMR checkpoint regrid policy differs")
    events = _amr_json(fields, "regrid_events_json")
    if not isinstance(events, list):
        raise ValueError("GPU AMR checkpoint regrid events are invalid")
    epoch_count = int(_amr_array(fields, "toe_epoch_count", (), "int64"))
    if epoch_count != len(events):
        raise ValueError("GPU AMR checkpoint face epoch count differs")
    current = names[0]
    prior_step = 0
    prior_time = 0.
    for number, event in enumerate(events):
        if not isinstance(event, dict) or set(event) != {
                "step", "time_s", "old_topology", "new_topology",
                "old_operator_identity", "new_operator_identity",
                "face_epoch_sealed", "cache_parent_ids"}:
            raise ValueError("GPU AMR checkpoint regrid event schema differs")
        old, new = event["old_topology"], event["new_topology"]
        step, when = event["step"], event["time_s"]
        if (old != current or new not in names or new == old or
                type(step) is not int or not prior_step < step <= steps or
                type(when) not in (int, float) or not math.isfinite(when) or
                not prior_time < when <= clock or
                event["old_operator_identity"] != index["topologies"][old]["operator_sha256"] or
                event["new_operator_identity"] != index["topologies"][new]["operator_sha256"] or
                event["face_epoch_sealed"] is not True or
                not isinstance(event["cache_parent_ids"], list) or
                any(type(node) is not int for node in event["cache_parent_ids"])):
            raise ValueError("GPU AMR checkpoint regrid transition differs")
        prefix = f"toe_epoch_{number}_"
        _amr_text(fields, prefix + "topology", old)
        _amr_toe_diagnostics(fields, prefix, masks[old])
        current, prior_step, prior_time = new, step, float(when)
    last_step = int(_amr_array(fields, "regrid_last_step", (), "int64"))
    if (current != name or last_step != prior_step or
            last_step > steps or
            int(_amr_array(fields, "boundary_event_cursor", (), "int64")) != 0):
        raise ValueError("GPU AMR checkpoint final route/cursor differs")
    position = 0
    current = names[0]
    for frame in frames:
        while position < len(events) and events[position]["step"] <= frame["steps"]:
            if events[position]["time_s"] > frame["time_s"] + 1e-12:
                raise ValueError("GPU AMR checkpoint event postdates its frame")
            current = events[position]["new_topology"]
            position += 1
        if frame["topology_name"] != current:
            raise ValueError("GPU AMR checkpoint events do not explain frame topology")
    cache_ids = fields.get("regrid_cache_node_ids")
    if (cache_ids is None or cache_ids.dtype != np.dtype("int64") or
            cache_ids.ndim != 1 or np.any(np.diff(cache_ids) <= 0)):
        raise ValueError("GPU AMR checkpoint transfer cache IDs differ")
    cache = _amr_array(fields, "regrid_cache_state",
                       (len(cache_ids), 4, nstate), "float64")
    if (np.any(cache[:, :, 0] < -1e-10) or
            np.any(cache[:, :, 3:nstate-1] < -1e-10) or
            np.any(cache[:, :, 3:3+nfrac] != 0)):
        raise ValueError("GPU AMR checkpoint transfer cache state is not physical")
    parent = index["tree"]["parent"]
    tree_area = index["tree"]["area"]
    selected = set(map(int, topology["node_ids"]))
    child_counts = np.bincount(parent[parent >= 0], minlength=len(parent))
    cached_children = {int(node): [] for node in cache_ids}
    for child, owner in enumerate(parent):
        if int(owner) in cached_children:
            cached_children[int(owner)].append(child)
    event = events[-1] if events else None
    witness = next((frame for frame in frames if event is not None and
                    frame["steps"] == event["step"] and
                    frame["topology_name"] == event["new_topology"]), None)
    for item, node in enumerate(cache_ids):
        if (int(node) not in selected or
                child_counts[node] != 4):
            raise ValueError("GPU AMR checkpoint transfer cache parent differs")
        if witness is None:
            raise ValueError("GPU AMR checkpoint transfer cache lacks regrid frame")
        children = cached_children[int(node)]
        restricted = np.sum(cache[item] * tree_area[children, None], axis=0) / tree_area[node]
        cell = np.flatnonzero(witness["node_ids"] == node)
        if cell.size != 1 or not np.allclose(restricted,
                np.r_[witness["q"][cell[0]], witness["secondary"][cell[0]]],
                rtol=1e-10, atol=1e-10):
            raise ValueError("GPU AMR checkpoint transfer cache is not conservative")
    if not events or events[-1]["cache_parent_ids"] != list(map(int, cache_ids)):
        if len(cache_ids) or events and events[-1]["cache_parent_ids"]:
            raise ValueError("GPU AMR checkpoint event/cache binding differs")
    expected = {
        "checkpoint_schema", "checkpoint_kind", "input_identity",
        "source_identity", "method_identity", "final_frame_sha256",
        "topology_name", "operator_identity", "regrid_mesh_index",
        "time_s", "steps", "rejected", "dt_hint", "maximum_cfl",
        "rk_euler_calls", "state", "node_ids", "ledger_water",
        "ledger_bed", "ledger_secondary", "ledger_source",
        *_BANK_HISTORY_FLOAT, "bank_history_collapse_count",
        "toe_signed", "toe_gross", "toe_epoch_count",
        "regrid_policy_json", "regrid_events_json", "regrid_last_step",
        "regrid_cache_node_ids", "regrid_cache_state",
        "boundary_event_cursor",
    }
    for number in range(epoch_count):
        expected.update({f"toe_epoch_{number}_{suffix}" for suffix in
                         ("topology", "signed", "gross")})
    if set(fields) != expected:
        raise ValueError("GPU AMR checkpoint complete field inventory differs")
    start = index["topologies"][names[0]]
    first = np.asarray(start["initial_data"]["state"], dtype=np.float64).T
    first_area = np.asarray(start["geometry"]["area_m2"], dtype=np.float64)
    final_area = np.asarray(topology["geometry"]["area_m2"], dtype=np.float64)
    first_frame = frames[0]
    if (first_frame["steps"] != 0 or first_frame["time_s"] != 0. or
            first_frame["topology_name"] != names[0] or
            not np.array_equal(np.column_stack((first_frame["q"],
                                                first_frame["secondary"])), first) or
            any(np.any(value != 0) for value in first_frame["ledger"].values())):
        raise ValueError("GPU AMR checkpoint run does not begin at prepared state")

    def balanced(before, after, ledger):
        delta = float(np.dot(final_area, after) - np.dot(first_area, before) -
                      np.sum(ledger))
        scale = max(1., abs(float(np.dot(first_area, before))),
                    abs(float(np.dot(final_area, after))))
        return abs(delta) <= 1e-9 + 1e-10 * scale

    if not balanced(first[:, 0], state[:, 0], fields["ledger_water"]):
        raise ValueError("GPU AMR checkpoint water inventory differs")
    for grain in range(nfrac):
        columns = [3 + grain] + [3 + nfrac + layer*nfrac + grain
                                 for layer in range(nlayer)]
        if not balanced(first[:, columns].sum(axis=1),
                        state[:, columns].sum(axis=1),
                        fields["ledger_bed"][grain]):
            raise ValueError("GPU AMR checkpoint material inventory differs")
    secondary_ledger = (fields["ledger_secondary"].sum() +
                        fields["ledger_source"])
    if not balanced(first[:, nstate-1], state[:, nstate-1], secondary_ledger):
        raise ValueError("GPU AMR checkpoint secondary inventory differs")


class GpuWorkerRunner(QObject):
    """Launch one pinned GPU run and observe its atomic status file.

    A new instance may be reused after ``finished``.  A resumed run is another
    call to ``start`` with a *new* output folder and the old checkpoint path.
    Cancellation is cooperative: no implicit terminate/kill can strand an
    apparently completed output.  A hung worker requires explicit external
    process management and is reported as still running.
    """

    started = pyqtSignal()
    stdout = pyqtSignal(str)
    stderr = pyqtSignal(str)
    progress = pyqtSignal(float)
    status = pyqtSignal(dict)
    finished = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.started.connect(self.started)
        self.process.readyReadStandardOutput.connect(self._stdout)
        self.process.readyReadStandardError.connect(self._stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_status)
        self._finalized = True
        self._output_root = None
        self._stop_file = None
        self._worker_root = None
        self._manifest_sha256 = None
        self._prepared_root = None
        self._input_sha256 = None
        self._prepared_kind = None
        self._selected_variant = None
        self._selected_integrator = None
        self._selected_launch_mode = None
        self._run_controls = None
        self._last_status = None
        self._stop_requested = False

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    @property
    def output_root(self) -> Path | None:
        return self._output_root

    @property
    def stop_file(self) -> Path | None:
        return self._stop_file

    def start(self, *, python_executable: str | Path, expected_python_sha256: str,
              worker_root: str | Path, expected_manifest_sha256: str,
              prepared_root: str | Path, output_root: str | Path,
              gpu_launch_mode: str = "graph", gpu_integrator: str = "ssp43",
              gpu_numerical_variant: str = "auto", until: float | None = None,
              output_every: float | None = None, dt_max: float | None = None,
              max_steps: int | None = None, resume: str | Path | None = None,
              gpu_amr_selector_policy: str | None = None) -> None:
        if self.is_running or not self._finalized:
            raise RuntimeError("A GPU worker is already running")
        if not all(isinstance(digest, str) and len(digest) == 64 and
                   all(ch in "0123456789abcdef" for ch in digest)
                   for digest in (expected_python_sha256, expected_manifest_sha256)):
            raise ValueError("Both GPU worker and Python SHA-256 pins are required")
        worker = inspect_gpu_source_worker(
            worker_root, expected_manifest_sha256=expected_manifest_sha256)
        scope = worker["scope"]
        if gpu_numerical_variant == INTEGRATED:
            receipt = _json_object(filesystem_path(Path(prepared_root).expanduser()).resolve()/
                                   'preparation_receipt.json')
            gpu_numerical_variant = route_for_preparation_schema(receipt.get('schema'), receipt.get('variant'))
        if gpu_numerical_variant not in scope["supported_gpu_variants"]:
            raise ValueError("GPU variant is not in the selected worker's supported scope")
        if gpu_numerical_variant in scope["excluded_gpu_variants"]:
            raise ValueError("GPU variant is excluded by this source worker")
        verify_general_amr_selector_scope(scope, gpu_numerical_variant,
                                          gpu_amr_selector_policy)
        try:
            selected_integrator = gpu_integrator_label(gpu_integrator)
        except ValueError as error:
            raise ValueError("Unsupported GPU launch mode or integrator") from error
        if gpu_launch_mode not in {"graph", "plain"}:
            raise ValueError("Unsupported GPU launch mode or integrator")
        if gpu_integrator == "ssp33" and gpu_launch_mode != "plain":
            raise ValueError("GPU SSPRK(3,3) requires plain launches")
        amr = gpu_numerical_variant == _BANK_AMR_VARIANT
        on_demand = gpu_numerical_variant in {
            _ON_DEMAND_AMR_VARIANT, _ON_DEMAND_GENERAL_AMR_VARIANT}
        g1 = gpu_numerical_variant == _G1_VARIANT
        bank = gpu_numerical_variant == _GENERAL_LIVE_BANK_VARIANT
        transport = gpu_numerical_variant == _GENERAL_LIVE_TRANSPORT_VARIANT
        if gpu_integrator == 'implicit_fwave_midpoint' and (
                not transport or gpu_launch_mode != 'plain'):
            raise ValueError('GPU implicit f-wave midpoint requires the plain fixed-grid common transport route')
        validate_worker_integrator_capability(scope, gpu_integrator,
                                              gpu_numerical_variant, gpu_launch_mode)
        if (amr or on_demand) and (gpu_integrator != "ssp33" or gpu_launch_mode != "plain"):
            raise ValueError("AMR GPU route requires plain SSPRK(3,3)")
        if (g1 or bank) and (gpu_integrator != "ssp33" or gpu_launch_mode != "plain"):
            raise ValueError("Inlet GPU route requires plain SSPRK(3,3)")
        python = filesystem_path(Path(python_executable).expanduser()).resolve()
        if not python.is_file() or _sha256(python) != expected_python_sha256:
            raise ValueError("Selected external Python executable differs from its pin")
        prepared = filesystem_path(Path(prepared_root).expanduser()).resolve()
        on_demand_input = inspect_gpu_on_demand_prepared(prepared) if on_demand else None
        if on_demand:
            if on_demand_input.get('selector_policy') != gpu_amr_selector_policy:
                raise ValueError('Selected GPU AMR selector differs from the prepared input')
            _verify_on_demand_runtime(worker, on_demand_input)
        input_sha256 = (on_demand_input["receipt_sha256"] if on_demand else
                        _verify_amr_prepared_input(prepared) if amr else
                        _verify_g1_prepared_input(prepared) if g1 else
                        _verify_prepared_input(prepared))
        if (not amr and not g1 and
                _json_object(prepared / "preparation_receipt.json").get("schema") == 4):
            raise ValueError("Schema-4 inlet input requires the selected inlet GPU variant")
        if (gpu_numerical_variant == _GRADED_LIVE_BEDLOAD_VARIANT and
                _json_object(prepared / "preparation_receipt.json").get("schema") != 3):
            raise ValueError("Graded GPU route requires schema-3 prepared input")
        if not on_demand and not amr:
            bank_input = _json_object(prepared/'preparation_receipt.json').get('schema') == 5
            if bank != bank_input:
                raise ValueError('Schema-5 bank input requires the selected general live-bank GPU variant')
            transport_input = _json_object(prepared/'preparation_receipt.json').get('schema') in (6, 7)
            if transport != transport_input:
                raise ValueError('Schema-6/7 transport input requires the selected general live-transport GPU variant')
        if transport:
            transport_data = read_gpu_prepared(prepared)
            if 'prognostic_secondary_execution_identity' in transport_data:
                validate_q3h3_worker_source_scope(scope)
            validate_integrator_scope(
                bedlaw=transport_data['bedlaw'],
                exchange_mode=transport_data['exchange_mode'],
                integrator=selected_integrator,
                launch_mode=gpu_launch_mode,
                bedform_roughness=transport_data.get('bedform_roughness', 0),
                bedload_adaptation=transport_data.get('bedload_adaptation', 0),
                general_avalanche=transport_data.get('general_avalanche', 0),
                schema_v2_law11=
                    has_public_law11_total_prepared_profile(transport_data))
            if gpu_integrator == 'implicit_fwave_midpoint':
                validate_implicit_prepared_scope(transport_data,
                    dtmax=(float(transport_data['dtmax']) if dt_max is None else float(dt_max)))
        allowed_sidecars = _allowed_prepared_sidecars(scope, gpu_numerical_variant)
        if g1 and allowed_sidecars != {"normal_metadata.bin"}:
            raise ValueError("Inlet GPU worker lacks the normal-metadata sidecar allowance")
        if transport and (not {'normal_metadata.bin'} <= allowed_sidecars or
                any((prepared/name).is_file() and name not in allowed_sidecars
                    for name in ('coupled_diffusion.bin','avalanche_metadata.bin',
                                 'delft_forcing_binding.json','execution_profile_v2.json'))):
            raise ValueError('Live-transport GPU worker lacks its authenticated sidecar allowance')
        if bank:
            required_sidecars = {'avalanche_metadata.bin', 'bank_history_metadata.bin',
                'bank_toe_metadata.bin', 'distributed_bank_metadata.bin', 'normal_metadata.bin'}
            if (not required_sidecars <= allowed_sidecars or
                    ((prepared/'coupled_diffusion.bin').is_file() and 'coupled_diffusion.bin' not in allowed_sidecars)):
                raise ValueError('Live-bank GPU worker lacks its authenticated sidecar allowance')
        for sidecar in scope["excluded_prepared_sidecars"]:
            if (prepared / sidecar).exists() and sidecar not in allowed_sidecars:
                raise ValueError("GPU prepared input has an excluded sidecar: " + sidecar)
        source_archive = _source_archive_for_variant(scope, worker["root"],
                                                     gpu_numerical_variant)
        output = filesystem_path(Path(output_root).expanduser()).resolve()
        if output.exists():
            raise FileExistsError("GPU run output must be a new folder: " + str(output))
        if not output.parent.is_dir():
            raise ValueError("GPU run output parent folder does not exist")
        checkpoint = None
        prior_on_demand = None
        if resume is not None:
            checkpoint = filesystem_path(Path(resume).expanduser()).resolve()
            if on_demand:
                from .gpu_resume import inspect_gpu_resume_checkpoint
                if not checkpoint.is_dir() or checkpoint.parent.name != "checkpoints":
                    raise ValueError("GPU AMR resume requires a committed checkpoint directory")
                prior_on_demand = inspect_gpu_resume_checkpoint(checkpoint.parent.parent, prepared)
                if checkpoint != prior_on_demand["checkpoint"].resolve():
                    raise ValueError("GPU AMR resume must use the manifest-selected checkpoint")
            elif not checkpoint.is_file() or checkpoint.name != "checkpoint.npz":
                raise ValueError("GPU resume requires an existing checkpoint.npz")
            if checkpoint.parent == output:
                raise ValueError("GPU resume output must be a new folder")
            if amr:
                prior = read_gpu_amr_run(checkpoint.parent, prepared_root=prepared)
                if prior["manifest"].get("input_preparation_sha256") != input_sha256:
                    raise ValueError("GPU AMR resume uses a different prepared input")
                _verify_amr_checkpoint(checkpoint.parent, prior)
            elif g1:
                prior = read_gpu_run(checkpoint.parent, prepared_root=prepared)
                if (prior["manifest"].get("input_preparation_sha256") != input_sha256 or
                        prior["manifest"].get("method", {}).get("variant") != _G1_VARIANT):
                    raise ValueError("GPU inlet resume uses a different prepared input/variant")
                final_name = prior["manifest"]["frames"][-1]["file"]
                _verify_g1_checkpoint(checkpoint, prior["manifest"],
                                      checkpoint.parent / final_name)
            elif bank or transport:
                from .gpu_resume import (inspect_gpu_resume_checkpoint,
                    validate_fixed_grid_restart_selection)
                prior = inspect_gpu_resume_checkpoint(checkpoint.parent, prepared)
                validate_fixed_grid_restart_selection(
                    prior, token=gpu_integrator, variant=gpu_numerical_variant,
                    launch_mode=gpu_launch_mode,
                    dtmax_s=(float(transport_data['dtmax'] if dt_max is None else dt_max)
                             if gpu_integrator == 'implicit_fwave_midpoint' else None),
                    output_every_s=(float(transport_data['output_dt'] if output_every is None else output_every)
                                    if gpu_integrator == 'implicit_fwave_midpoint' else None))
        for name, value in (("until", until), ("output_every", output_every), ("dt_max", dt_max)):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(name + " must be finite and positive")
        if max_steps is not None and (isinstance(max_steps, bool) or int(max_steps) < 1):
            raise ValueError("max_steps must be positive")
        run_controls = None
        if on_demand:
            run_controls = dict(
                end_s=float(on_demand_input["end_s"] if until is None else until),
                output_every_s=float(on_demand_input["output_every_s"] if output_every is None else output_every),
                dtmax_s=float(on_demand_input["dtmax_s"] if dt_max is None else dt_max))
            if (run_controls["end_s"] > on_demand_input["end_s"] or
                    run_controls["end_s"] <= on_demand_input["time_s"] or
                    run_controls["dtmax_s"] > on_demand_input["dtmax_s"]):
                raise ValueError("GPU AMR time controls exceed prepared limits")
            if prior_on_demand is not None:
                method = prior_on_demand["method"]
                if (method.get("variant") != gpu_numerical_variant or
                        method.get('selector_policy') != gpu_amr_selector_policy or
                        method.get("integrator") != "SSPRK33" or method.get("launch_mode") != "plain" or
                        method.get("output_every_s") != run_controls["output_every_s"] or
                        method.get("dtmax_s") != run_controls["dtmax_s"] or
                        prior_on_demand["time_s"] > run_controls["end_s"]):
                    raise ValueError("GPU AMR restart changes the saved method/output clock")
        stop_file = output.parent / ("." + output.name + ".gpu-stop-" + secrets.token_hex(16))
        if stop_file.exists():
            raise FileExistsError("GPU stop marker already exists")
        arguments = ["--backend", "gpu", "--input", str(prepared), "--output", str(output),
                     "--gpu-launch-mode", gpu_launch_mode, "--gpu-integrator", gpu_integrator,
                     "--gpu-numerical-variant", gpu_numerical_variant,
                     "--stop-file", str(stop_file)]
        for flag, value in (("--until", until), ("--output-every", output_every),
                            ("--dt-max", dt_max), ("--max-steps", max_steps)):
            if value is not None:
                arguments.extend((flag, str(value)))
        if checkpoint is not None:
            arguments.extend(("--resume", str(checkpoint)))
        if source_archive is not None:
            arguments.extend(("--gpu-source-archive", str(source_archive)))
        python_arguments, cwd = isolated_worker_command(worker["root"], "run", arguments)
        environment = QProcessEnvironment.systemEnvironment()
        # GPU method selection is carried by authenticated command arguments
        # and method identities, never by a shell variable intended for the
        # native CPU solver.
        environment.remove("BRAIDS_TIME_INTEGRATOR")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(cwd)
        self.process.setProgram(str(python))
        self.process.setArguments(python_arguments)
        self._output_root = output
        self._stop_file = stop_file
        self._worker_root = worker["root"]
        self._manifest_sha256 = worker["manifest_sha256"]
        self._prepared_root = prepared
        self._input_sha256 = input_sha256
        self._prepared_kind = "amr-on-demand" if on_demand else "amr-prepared" if amr else "fixed-prepared"
        self._selected_variant = gpu_numerical_variant
        self._selected_integrator = selected_integrator
        self._selected_launch_mode = gpu_launch_mode
        self._selected_selector_policy = gpu_amr_selector_policy
        self._run_controls = run_controls
        self._last_status = None
        self._stop_requested = False
        self._finalized = False
        self._poll_timer.start()
        self.process.start()

    def stop(self) -> None:
        if not self.is_running or self._stop_requested:
            return
        try:
            with self._stop_file.open("x", encoding="ascii") as stream:
                stream.write("BRAIDS GPU cooperative cancellation\n")
            self._stop_requested = True
        except OSError as exc:
            self.stderr.emit(f"Could not request GPU cancellation: {exc}\n")

    def _stdout(self) -> None:
        value = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if value:
            self.stdout.emit(value)

    def _stderr(self) -> None:
        value = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if value:
            self.stderr.emit(value)

    def _poll_status(self) -> None:
        if self._output_root is None:
            return
        path = self._output_root / "status.json"
        try:
            record = _json_object(path)
            if record.get("backend") != "gpu":
                raise ValueError("GPU status has a different backend")
            percent = float(record["percent"])
            if not math.isfinite(percent) or not 0 <= percent <= 100:
                raise ValueError("GPU status has invalid progress")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # The worker creates the output asynchronously and writes status
            # atomically.  Finalization reports a malformed/missing status.
            return
        if record != self._last_status:
            self._last_status = record
            self.status.emit(record)
            self.progress.emit(percent)

    def _verify_completion(self) -> None:
        root = self._output_root
        status = _json_object(root / "status.json")
        manifest = _json_object(root / "run.json")
        if (status.get("status") != "completed" or status.get("backend") != "gpu" or
                manifest.get("status") != "completed" or manifest.get("backend") != "gpu" or
                manifest.get("backend_selected_by") != "user" or
                manifest.get("automatic_backend_selection") is not False):
            raise ValueError("GPU worker did not report an explicitly selected completed run")
        if manifest.get("input_preparation_sha256") != self._input_sha256:
            raise ValueError("GPU run input identity differs")
        if self._prepared_kind == "amr-on-demand":
            from .gpu_resume import inspect_gpu_resume_checkpoint
            method = manifest.get("method", {})
            expected_kind = ('amr-on-demand-general' if
                self._selected_variant == _ON_DEMAND_GENERAL_AMR_VARIANT else
                'amr-on-demand')
            if (self._selected_variant not in {_ON_DEMAND_AMR_VARIANT,
                    _ON_DEMAND_GENERAL_AMR_VARIANT} or
                    manifest.get("schema") != 3 or manifest.get("format") != "braids-gpu-amr-run-v2" or
                    manifest.get("input_kind") != expected_kind or
                    method.get("variant") != self._selected_variant or
                    method.get('selector_policy') != self._selected_selector_policy or
                    method.get("integrator") != "SSPRK33" or method.get("launch_mode") != "plain" or
                    manifest.get("validation_projection_only") is True or
                    manifest.get("GPU_solver_executed_in_this_output") is False or
                    any(manifest.get(key) != value for key, value in self._run_controls.items())):
                raise ValueError("GPU output is not the selected on-demand AMR run")
            # This host inspector verifies the result frames, operators and the
            # complete manifest-selected four-file accepted checkpoint once.
            verified = inspect_gpu_resume_checkpoint(root, self._prepared_root)
            if (verified["status"] != "completed" or
                    not math.isclose(verified["time_s"], self._run_controls["end_s"], rel_tol=0, abs_tol=1e-12)):
                raise ValueError("GPU AMR run did not reach the requested end time")
            worker = inspect_gpu_source_worker(self._worker_root,
                expected_manifest_sha256=self._manifest_sha256)
            prepared = inspect_gpu_on_demand_prepared(self._prepared_root,
                expected_receipt_sha256=self._input_sha256)
            _verify_on_demand_runtime(worker, prepared)
            return
        if self._prepared_kind == "amr-prepared":
            method = manifest.get("method")
            if (self._selected_variant != _BANK_AMR_VARIANT or
                    manifest.get("schema") != 2 or
                    manifest.get("format") != "braids-gpu-amr-run-v1" or
                    manifest.get("input_kind") != "amr-prepared" or
                    not isinstance(method, dict) or
                    method.get("variant") != self._selected_variant or
                    manifest.get("validation_projection_only") is True or
                    manifest.get("GPU_solver_executed_in_this_output") is False):
                raise ValueError("GPU AMR output is not the selected numerical worker route")
            # The topology-aware reader verifies the source snapshot and each
            # complete frame once.  Reusing the fixed-grid frame hash loop
            # here would double completion I/O on large river reaches.
            verified = read_gpu_amr_run(root, prepared_root=self._prepared_root)
            _verify_amr_checkpoint(root, verified)
            inspect_gpu_source_worker(self._worker_root,
                                      expected_manifest_sha256=self._manifest_sha256)
            if _verify_amr_prepared_input(self._prepared_root) != self._input_sha256:
                raise ValueError("GPU AMR prepared input identity changed during execution")
            return
        method = manifest.get("method")
        if not isinstance(method, dict):
            raise ValueError("GPU run method identity is absent")
        g1 = self._selected_variant == _G1_VARIANT
        if g1:
            receipt = _json_object(self._prepared_root / "preparation_receipt.json")
            if (method.get("variant") != _G1_VARIANT or
                    method.get("integrator") != "SSPRK33" or
                    method.get("launch_mode") != "plain" or
                    method.get("compact_checkpoint_schema") != 1 or
                    manifest.get("compact_checkpoint_schema") != 1 or
                    manifest.get("input_identity") != _g1_numerical_identity(receipt)):
                raise ValueError("GPU inlet output is not the selected compact numerical route")
        method_sha = hashlib.sha256(json.dumps(method, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        if method_sha != manifest.get("method_identity"):
            raise ValueError("GPU run method identity differs")
        source = root / "source"
        source_manifest = source / "runtime_manifest.json"
        if not source_manifest.is_file():
            source_manifest = source / "manifest.json"
        if _verify_frozen_source(source_manifest) != manifest.get("source_identity"):
            raise ValueError("GPU frozen source identity differs")
        if method.get("source_identity") != manifest.get("source_identity"):
            raise ValueError("GPU method uses a different frozen source")
        frames = manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("GPU run has no frames")
        seen_frames = set()
        previous_time = -math.inf
        previous_steps = -1
        for frame in frames:
            if (not isinstance(frame, dict) or
                    not isinstance(frame.get("file"), str) or
                    not frame["file"].startswith("frame_") or
                    not frame["file"].endswith(".npz") or
                    Path(frame["file"]).name != frame["file"] or
                    frame["file"] in seen_frames or
                    not isinstance(frame.get("sha256"), str) or
                    (root / frame["file"]).is_symlink() or
                    not (root / frame["file"]).is_file() or
                    _sha256(root / frame["file"]) != frame["sha256"]):
                raise ValueError("GPU frame identity differs")
            frame_time = frame.get("time_s")
            frame_steps = frame.get("steps")
            if (isinstance(frame_time, bool) or
                    not isinstance(frame_time, (int, float)) or
                    not math.isfinite(frame_time) or frame_time < previous_time or
                    isinstance(frame_steps, bool) or
                    not isinstance(frame_steps, int) or
                    frame_steps < previous_steps):
                raise ValueError("GPU frame clock or step sequence differs")
            seen_frames.add(frame["file"])
            previous_time, previous_steps = frame_time, frame_steps
        last_frame = frames[-1]
        terminal_time = manifest.get("time_s") if g1 else manifest.get("final_time_s")
        if (not isinstance(last_frame, dict) or
                last_frame.get("steps") != manifest.get("steps") or
                not math.isclose(float(last_frame.get("time_s")),
                                 float(terminal_time), rel_tol=0, abs_tol=1e-12)):
            raise ValueError("GPU final frame identity or clock differs")
        if status.get("steps") != manifest.get("steps"):
            raise ValueError("GPU status and manifest steps differ")
        checkpoint = root / "checkpoint.npz"
        if not checkpoint.is_file():
            raise ValueError("GPU final checkpoint is absent")
        if g1:
            # The full reader authenticates all committed frame payloads and
            # their geometry before the launcher reports completion.
            read_gpu_run(root, prepared_root=self._prepared_root)
            _verify_g1_checkpoint(checkpoint, manifest, root / last_frame["file"])
        elif self._selected_variant in (_GENERAL_LIVE_BANK_VARIANT, _GENERAL_LIVE_TRANSPORT_VARIANT):
            if (method.get('variant') != self._selected_variant or
                    method.get('integrator') != self._selected_integrator or
                    method.get('launch_mode') != self._selected_launch_mode):
                raise ValueError('GPU compact output uses a different selected method')
            from .gpu_resume import inspect_gpu_resume_checkpoint
            read_gpu_run(root, prepared_root=self._prepared_root)
            inspect_gpu_resume_checkpoint(root, self._prepared_root)
        else:
            with np.load(checkpoint, allow_pickle=False) as archive:
                for name, expected in (("input_identity", self._input_sha256),
                                       ("source_identity", manifest["source_identity"]),
                                       ("method_identity", method_sha)):
                    if _checkpoint_scalar(archive, name) != expected:
                        raise ValueError("GPU checkpoint " + name + " differs")
                if _checkpoint_scalar(archive, "steps") != manifest["steps"]:
                    raise ValueError("GPU checkpoint steps differ")
                if not math.isclose(float(_checkpoint_scalar(archive, "time")),
                                    float(terminal_time), rel_tol=0, abs_tol=1e-12):
                    raise ValueError("GPU checkpoint time differs")
        # Detect a source-worker mutation across a long run, without loading
        # any frame arrays.  This is still not a substitute for frame-reader QA.
        inspect_gpu_source_worker(self._worker_root,
                                  expected_manifest_sha256=self._manifest_sha256)
        current_input = (_verify_g1_prepared_input(self._prepared_root) if g1 else
                         _verify_prepared_input(self._prepared_root))
        if current_input != self._input_sha256:
            raise ValueError("GPU prepared input identity changed during execution")

    def _process_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.stderr.emit("Could not start pinned GPU Python: " + self.process.errorString() + "\n")
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, exit_code, exit_status) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._poll_timer.stop()
        self._stdout()
        self._stderr()
        self._poll_status()
        success = False
        if (not self._stop_requested and int(exit_code) == 0 and
                exit_status == QProcess.ExitStatus.NormalExit):
            try:
                self._verify_completion()
                success = True
            except Exception as exc:
                # Bad ZIP members, filesystem races and malformed worker JSON
                # must all finish the UI lifecycle as failures, never successes.
                self.stderr.emit("GPU completion verification failed: " + str(exc) + "\n")
        if success:
            self.progress.emit(100.)
        self.finished.emit(int(exit_code), success)
