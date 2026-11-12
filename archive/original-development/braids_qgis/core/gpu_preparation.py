"""Cancellable QGIS-side preparation for an explicitly chosen GPU worker.

The native executable only initializes the case; the external worker owns the
GPU-ready receipt.  This module never imports CUDA into QGIS and never changes
the requested execution backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Callable
from braids_core.windows_paths import filesystem_path
from braids_core.gpu_capabilities import (INTEGRATED, effective_project_transport,
    gpu_integrator_label, resolve_project_route)

from .gpu_worker_identity import inspect_gpu_source_worker
from .gpu_worker_runner import (isolated_worker_command,
    _verify_recursive_adaptation_capability, _verify_recursive_amr_capability,
    verify_general_amr_selector_scope)
from .gpu_prepared_io import read_gpu_prepared
from .gpu_on_demand_prepared import inspect_gpu_on_demand_prepared
from .gpu_live_transport_io import (validate_implicit_prepared_scope,
    validate_implicit_project_scope, validate_integrator_scope,
    validate_worker_integrator_capability,
    materialize_execution_profile_sidecar,
    validate_public_law11_total_preparation,
    resolve_public_law11_total_profile)
from .project import load_project, prepare_run, reject_raw_gpu_persistent_front
from .recursive_product import write_recursive_prepared_product
from .sediment_product import (
    PROJECT_KEY as QGIS_PRODUCT_KEY,
    materialize_gpu_prepared_product,
    verify_gpu_worker_for_project,
)
from braids_core.sediment.execution_profile_binding import (
    EXECUTION_PROFILE_BINDING_KEY)
from braids_core.sediment.execution_profile_v2 import BACKEND_GPU


_GRADED_LIVE_BEDLOAD_VARIANT = "live_normal_critical_graded_bedload_ssp33"
_GENERAL_LIVE_BANK_VARIANT = "live_normal_critical_bank_ssp33"
_GENERAL_LIVE_TRANSPORT_VARIANT = "live_normal_critical_transport_ssp33"
_ON_DEMAND_AMR_VARIANT = "on_demand_bank_amr_ssp33"
_ON_DEMAND_GENERAL_AMR_VARIANT = "on_demand_transport_amr_ssp33"


def _sha256(path: Path) -> str:
    with filesystem_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _stop_unresponsive_worker(process: subprocess.Popen) -> None:
    """Terminate only the preparation process tree started by this task."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, text=True, timeout=10, check=False)
    else:
        process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


_PLAIN_SSP33_VARIANTS = frozenset((
    "scheduled_inlet_ssp33", "bedform_feedback_ssp33",
    "general_avalanche_plain_ssp33", "physical_toe_mode2_plain_ssp33",
    "physical_toe_mode2_avalanche_ssp33",
    "distributed_slope_mode3_plain_ssp33",
    "distributed_slope_mode3_avalanche_ssp33",
    "bedload_adaptation_plain_ssp33",
    _GRADED_LIVE_BEDLOAD_VARIANT,
))
_MPM_ONLY_VARIANTS = _PLAIN_SSP33_VARIANTS - frozenset((
    "scheduled_inlet_ssp33", "general_avalanche_plain_ssp33",
))
_DEFERRED_BOUNDARY_MODES = frozenset((
    "prescribed_trace", "prescribed_depth_trace", "supercritical", "weir",
))
_BANK_VARIANT_MODES = {
    "physical_toe_mode2_plain_ssp33": "noncohesive_transport_capacity",
    "physical_toe_mode2_avalanche_ssp33": "noncohesive_transport_capacity",
    "distributed_slope_mode3_plain_ssp33": "distributed_slope_transport",
    "distributed_slope_mode3_avalanche_ssp33": "distributed_slope_transport",
}


def _has_bed_feed(boundaries: dict) -> bool:
    for row in boundaries.values():
        if (row.get("bedload_boundary_mode") == "equilibrium_capacity" or
                row.get("bed_input_kg_m_s", 0) or row.get("bed_input_kg_s", 0) or
                any(row.get("fraction_bed_input_kg_s", {}).values())):
            return True
        series = row.get("sediment_series")
        if series and any(series.get("bed_input_kg_s", ())):
            return True
        if any(any(series.get("bed_input_kg_s", ())) for series in
               row.get("fraction_sediment_series", {}).values()):
            return True
    return False


def _validate_graded_live_bedload_project(project: dict) -> None:
    """Fail early on controls outside the common seven-class GPU route.

    Face counts, sampled normal-section geometry and forcing-knot alignment
    are checked by the native exporter and numerical worker after preparation.
    This preflight prevents an obviously incompatible QGIS project from
    spending time on native mesh construction first.
    """
    numerics = project["numerics"]
    initial = project.get("initial", {})
    internal = project.get("internal_conditions", {})
    fractions = project.get("sediment_fractions")
    boundaries = project.get("boundaries", {})
    if (numerics.get("bedload") != 2 or numerics.get("exchange") != 0 or
            numerics.get("mpm_friction_mode") != 1 or
            numerics.get("hiding_exposure") != 4 or
            numerics.get("bed_slope") != 2 or
            not isinstance(fractions, list) or len(fractions) != 7 or
            any(not isinstance(row, dict) or row.get("sediment_type") != "sand" or
                row.get("roughness_mode") != "manning" for row in fractions) or
            initial.get("substrate_layer_count") != 3 or
            float(initial.get("active_layer_thickness_m", 0)) <= 0 or
            not (0 < float(numerics.get("cfl", 0)) <= .8)):
        raise ValueError("The graded live GPU route requires seven Manning-roughness sand classes, four bed layers, MPM bedload 2, friction 1, hiding 4, slope 2, exchange 0 and CFL at most 0.8")
    if sorted(row.get("mode") for row in boundaries.values()) != ["critical", "inlet_normal"]:
        raise ValueError("The graded live GPU route requires one normal-depth inlet and one critical-depth outlet")
    for name in ("secondary_flow", "bank_failure", "general_avalanche",
                 "bedform_roughness", "bank_morphology"):
        if internal.get(name, {}).get("enabled", False):
            raise ValueError("The graded live GPU route does not yet support " + name)
    adaptation = internal.get("bedload_adaptation", {})
    if adaptation.get("enabled", False) and float(adaptation.get("length_diameters", 0)) <= 0:
        raise ValueError("GPU bedload adaptation requires a positive grain-diameter length")


def _validate_ordinary_boundaries(boundaries, fraction_ids, end, *,
                                  forbid_suspension=False,
                                  allow_viscous_profile=False):
    """Shared fixed-grid boundary policy; never used to broaden AMR."""
    surveyed = any(row.mode in ('inlet_stage','inlet_normal','inlet_critical','inlet_depth')
                   for row in boundaries)
    if surveyed:
        if (len(boundaries) != 2 or sum(row.mode in ('stage','critical') for row in boundaries) != 1 or
                sum(row.mode in ('inlet_stage','inlet_depth','inlet_normal','inlet_critical')
                    for row in boundaries) != 1):
            raise ValueError('Ordinary GPU needs one surveyed inlet and one stage or critical outlet')
    elif not boundaries or any(row.mode not in ('discharge','stage','total_discharge','critical')
                               for row in boundaries):
        raise ValueError('Ordinary GPU supports stock modes 1/2/4/6 only')
    for boundary in boundaries:
        profile_ok = (boundary.total_discharge_profile == 'stock_conveyance' or
                      (allow_viscous_profile and
                       boundary.total_discharge_profile ==
                           'viscous_manning_equilibrium' and
                       boundary.mode == 'total_discharge'))
        if (not profile_ok or boundary.sediment_series is not None or
                boundary.concentration_kg_m3 or boundary.bed_input_kg_m_s or boundary.bed_input_kg_s is not None):
            raise ValueError(
                'Ordinary GPU requires stock conveyance, or the exact admitted '
                'viscosity-equilibrated total-discharge profile, and explicitly '
                'classed sediment forcing')
        referenced = (set(boundary.fraction_concentration_kg_m3) | set(boundary.fraction_bed_input_kg_s) |
                      set(boundary.fraction_sediment_series))
        if not referenced <= fraction_ids:
            raise ValueError('Ordinary GPU boundary refers to an unknown class')
        if forbid_suspension and (any(boundary.fraction_concentration_kg_m3.values()) or
                any(any(row.concentration_kg_m3) for row in boundary.fraction_sediment_series.values())):
            raise ValueError('Ordinary GPU banks exclude suspended concentration forcing')
        if boundary.mode == 'critical':
            if (boundary.hydraulic_series is not None or boundary.fraction_sediment_series or
                    boundary.bedload_boundary_mode != 'prescribed'):
                raise ValueError('Critical outlet cannot prescribe hydraulic/sediment forcing')
            continue
        hydraulic = boundary.hydraulic_series
        if hydraulic is None:
            if surveyed and boundary.is_inlet and boundary.discharge_m3_s <= 0.:
                raise ValueError('Surveyed inlet requires positive discharge')
        elif (hydraulic.times_s[0] > 0. or hydraulic.times_s[-1] < end or
                (surveyed and boundary.is_inlet and min(hydraulic.discharge_m3_s) <= 0.)):
            raise ValueError('Hydraulic schedule must cover the run with admissible discharge')
        for series in boundary.fraction_sediment_series.values():
            if series.times_s[0] > 0. or series.times_s[-1] < end:
                raise ValueError('Each class sediment schedule must cover the run')


def _validate_live_amr_project(project: dict, *, fixed_grid_bank: bool = False) -> None:
    """Validate project controls only; exported surveyed faces remain authoritative."""
    from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction

    def number(value, label, *, minimum=None, positive=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or (minimum is not None and value < minimum) or
                (positive and value <= 0)):
            raise ValueError('Invalid evolving AMR ' + label)
        return value

    numerics, initial = project['numerics'], project.get('initial', {})
    internal = project.get('internal_conditions', {})
    end = number(numerics.get('end_time_s'), 'end time', positive=True)
    for key in ('hiding_exposure', 'bed_slope'):
        choices = ((0, 2, 3, 4, 5) if fixed_grid_bank else (0, 4)) if key == 'hiding_exposure' else (0, 2)
        if type(numerics.get(key)) is not int or numerics[key] not in choices:
            raise ValueError('Unsupported evolving AMR ' + key)
    for key, default in (('hiding_parker_alpha', .6), ('hiding_wu_exponent', -.6)):
        number(numerics.get(key, default), key)
    for key, default in (('bed_slope_alfabs', 1.), ('bed_slope_alfabn', 1.5),
                         ('mpm_cf', .01), ('mpm_critical_shields', .047)):
        number(numerics.get(key, default), key, minimum=0.)
    if fixed_grid_bank and numerics.get('mpm_friction_mode') == 1 and not math.isclose(
            numerics.get('mpm_critical_shields',.047),.047,rel_tol=0.,abs_tol=1.e-14):
        raise ValueError('Manning MPM requires critical Shields0.047')
    angle = number(numerics.get('bed_friction_angle_deg', 30.), 'bed friction angle')
    if not 0 < angle < 90:
        raise ValueError('Evolving AMR bed friction angle must be between 0 and 90 degrees')
    adaptation = internal.get('bedload_adaptation', {})
    if type(adaptation.get('enabled', False)) is not bool:
        raise ValueError('Evolving AMR adaptation switch must be boolean')
    number(adaptation.get('length_diameters', 100.), 'adaptation length',
           positive=adaptation.get('enabled', False))
    distributed = internal.get('bank_failure', {}).get('toe_erosion_mode') == 'distributed_slope_transport'
    avalanche = internal.get('general_avalanche', {}).get('enabled')
    if (type(avalanche) is not bool or (not fixed_grid_bank and not distributed and avalanche is not True)):
        raise ValueError('Evolving bank AMR requires the enabled general-avalanche transaction')
    fractions = project.get('sediment_fractions')
    if not isinstance(fractions, list) or not 1 <= len(fractions) <= 32:
        raise ValueError('Evolving AMR requires 1 to 32 explicit sediment classes')
    try:
        materials = [CutSedimentFraction(**row) for row in fractions]
        boundaries = {name: CutBoundary(**row) for name, row in project['boundaries'].items()}
    except (TypeError, KeyError) as error:
        raise ValueError('Malformed evolving AMR material/boundary controls') from error
    fraction_ids = {row.fraction_id for row in materials}
    if len(fraction_ids) != len(materials) or any(row.sediment_type != 'sand' for row in materials):
        raise ValueError('Evolving AMR requires unique noncohesive sand classes')
    if fixed_grid_bank and numerics['hiding_exposure'] and len(materials) < 2:
        raise ValueError('GPU hiding/exposure requires multiple sediment classes')
    layers = initial.get('substrate_layer_count')
    if type(layers) is not int or not 1 <= layers <= 32:
        raise ValueError('Evolving AMR requires 1 to 32 substrate layers')
    number(initial.get('active_layer_thickness_m'), 'active-layer thickness', positive=True)
    if layers > 1:
        number(initial.get('substrate_layer_thickness_m'), 'substrate-layer thickness', positive=True)
    if fixed_grid_bank:
        _validate_ordinary_boundaries(list(boundaries.values()),fraction_ids,end,forbid_suspension=True)
        if internal.get('horizontal_viscosity', {}).get('enabled', False):
            raise ValueError('Use the explicit horizontal_viscosity_m2_s project control')
        return
    if (len(boundaries) != 2 or sum(row.mode == 'critical' for row in boundaries.values()) != 1 or
            sum(row.mode in ('inlet_stage', 'inlet_normal', 'inlet_critical')
                for row in boundaries.values()) != 1):
        raise ValueError('Evolving AMR requires one surveyed stage/normal/critical inlet and a critical outlet')
    for boundary in boundaries.values():
        if boundary.total_discharge_profile != 'stock_conveyance':
            raise ValueError('Evolving AMR requires the stock inlet discharge profile')
        if (boundary.concentration_kg_m3 or any(boundary.fraction_concentration_kg_m3.values()) or
                (boundary.sediment_series and any(boundary.sediment_series.concentration_kg_m3)) or
                any(any(row.concentration_kg_m3) for row in boundary.fraction_sediment_series.values())):
            raise ValueError('Evolving AMR excludes suspended concentration forcing')
        referenced = (set(boundary.fraction_concentration_kg_m3) |
                      set(boundary.fraction_bed_input_kg_s) | set(boundary.fraction_sediment_series))
        if not referenced <= fraction_ids:
            raise ValueError('Evolving AMR boundary refers to an unknown sediment class')
        number(boundary.bed_input_kg_m_s, 'bed feed', minimum=0.)
        if boundary.bed_input_kg_s is not None:
            number(boundary.bed_input_kg_s, 'total bed feed', minimum=0.)
        if boundary.mode == 'critical':
            if (boundary.hydraulic_series is not None or boundary.sediment_series is not None or
                    boundary.fraction_sediment_series or boundary.bedload_boundary_mode != 'prescribed'):
                raise ValueError('Evolving AMR critical outlet cannot prescribe hydraulic or sediment forcing')
            continue
        series = boundary.hydraulic_series
        if (series is None or series.times_s[0] > 0 or series.times_s[-1] < end or
                min(series.discharge_m3_s) <= 0):
            raise ValueError('Evolving AMR inlet needs positive scheduled discharge covering the run')
        for feed in ([boundary.sediment_series] if boundary.sediment_series else []) + list(
                boundary.fraction_sediment_series.values()):
            if feed.times_s[0] > 0 or feed.times_s[-1] < end:
                raise ValueError('Evolving AMR bed feed schedule must cover the run')
    if internal.get('horizontal_viscosity', {}).get('enabled', False):
        raise ValueError('Evolving live AMR excludes horizontal viscosity')


def _validate_general_live_bank_project(project: dict, *, gpu_integrator: str,
                                       gpu_launch_mode: str | None) -> None:
    """Ordinary fixed-grid bank scope; intentionally separate from AMR policy."""
    numerics = project['numerics']
    internal = project.get('internal_conditions', {})
    if gpu_integrator != 'ssp33' or gpu_launch_mode != 'plain':
        raise ValueError('General live-bank GPU requires explicit Plain SSPRK33')
    if numerics.get('sediment_enabled', True) is not True:
        raise ValueError('General live-bank GPU requires enabled sediment transport')
    for key, expected in (('bedload', 2), ('exchange', 0)):
        if type(numerics.get(key)) is not int or numerics[key] != expected:
            raise ValueError('General live-bank GPU requires unsuspended MPM2')
    if type(numerics.get('mpm_friction_mode')) is not int or numerics['mpm_friction_mode'] not in (0,1):
        raise ValueError('General live-bank GPU requires MPM friction0 or1')
    if project.get('initial', {}).get('active_layer_mode', 'constant') != 'constant':
        raise ValueError('General live-bank GPU requires a constant active-layer thickness')
    bank = internal.get('bank_failure', {})
    if (bank.get('enabled') is not True or bank.get('toe_erosion_mode') not in
            ('noncohesive_transport_capacity', 'distributed_slope_transport') or
            bank.get('front_tracking_enabled', False)):
        raise ValueError('General live-bank GPU requires noncohesive mode2/3 without persistent fronts')
    for key in ('bank_morphology', 'bedform_roughness', 'morphological_acceleration'):
        if internal.get(key, {}).get('enabled', False):
            raise ValueError('General live-bank GPU excludes '+key)
    if numerics.get('horizontal_viscosity',0.) != 0.:
        raise ValueError('Use the explicit horizontal_viscosity_m2_s project control')
    viscosity=numerics.get('horizontal_viscosity_m2_s',0.)
    if (type(viscosity) not in (int,float) or not math.isfinite(viscosity) or viscosity < 0. or
            (viscosity > 0. and numerics.get('horizontal_viscosity_formulation','depth_conservative') !=
             'velocity_laplacian')):
        raise ValueError('General live-bank GPU viscosity requires nonnegative velocity_laplacian input')
    secondary = internal.get('secondary_flow', {})
    enabled = secondary.get('enabled', False)
    model = secondary.get('model', 'equilibrium')
    direction = secondary.get('transport_direction', True)
    beta = secondary.get('momentum_beta', 0.)
    espir = secondary.get('espir', 1.)
    if (type(enabled) is not bool or model not in ('equilibrium', 'prognostic') or
            type(direction) is not bool or any(type(value) not in (int, float) or
            not math.isfinite(value) or value < 0. for value in (beta, espir))):
        raise ValueError('Invalid fixed-grid GPU secondary-flow controls')
    if enabled and model == 'equilibrium' and (not direction or beta != 0.):
        raise ValueError('Equilibrium secondary flow requires direction-only coupling and zero momentum Beta')
    # Disabled project settings may retain UI preferences; the native exporter
    # records model0 with zero direction/Beta and no prognostic state/ledger.
    _validate_live_amr_project(project, fixed_grid_bank=True)


def _validate_general_live_transport_project(project: dict, *, gpu_integrator: str,
                                            gpu_launch_mode: str | None) -> None:
    """Ordinary nonbanked transport; do not inherit or expand AMR policy."""
    from braids_core.boundaries.cut_cell_io import CutBoundary, CutSedimentFraction

    def number(value, label, *, positive=False):
        if (type(value) not in (int, float) or not math.isfinite(value) or
                (value <= 0. if positive else value < 0.)):
            raise ValueError('Invalid live-transport '+label)
        return value

    n, initial = project['numerics'], project.get('initial', {})
    bedlaw, exchange = effective_project_transport(project)
    schema_v2_law11 = bedlaw == 11
    if schema_v2_law11:
        resolve_public_law11_total_profile(project)
    disabled = not n.get('sediment_enabled', True)
    if disabled:
        # This detached admission view mirrors prepare_run's existing explicit
        # sediment-off semantics. Saved choices and actual material stay intact.
        n = dict(n, bedload=bedlaw, exchange=exchange, mpm_friction_mode=0,
                 hiding_exposure=0, bed_slope=0, settling_mode=0, morphological_factor=1.)
        from dataclasses import asdict, replace
        boundaries = {name:asdict(replace(CutBoundary(**row), concentration_kg_m3=0.,
            bed_input_kg_m_s=0., bed_input_kg_s=None, bedload_boundary_mode='prescribed',
            sediment_series=None, fraction_concentration_kg_m3={},
            fraction_bed_input_kg_s={}, fraction_sediment_series={}))
            for name,row in project.get('boundaries',{}).items()}
        project = dict(project, numerics=n, boundaries=boundaries)
    internal = project.get('internal_conditions', {})
    if gpu_integrator == 'implicit_fwave_midpoint':
        validate_implicit_project_scope(project, launch_mode=gpu_launch_mode,
                                        variant=_GENERAL_LIVE_TRANSPORT_VARIANT)
    if schema_v2_law11:
        if (gpu_integrator != 'implicit_fwave_midpoint' or
                gpu_launch_mode != 'plain' or exchange != 0 or
                any(internal.get(name, {}).get('enabled', False) for name in
                    ('bedform_roughness', 'bedload_adaptation',
                     'general_avalanche'))):
            raise ValueError(
                'Schema-v2 law-11 total load requires Plain implicit_fwave_midpoint '
                'with exchange and optional morphology transactions off')
    else:
        validate_integrator_scope(
            bedlaw=bedlaw, exchange_mode=exchange,
            integrator=gpu_integrator_label(gpu_integrator),
            launch_mode=gpu_launch_mode,
            bedform_roughness=internal.get('bedform_roughness', {}).get('enabled', False),
            bedload_adaptation=internal.get('bedload_adaptation', {}).get('enabled', False),
            general_avalanche=internal.get('general_avalanche', {}).get('enabled', False))
    for key, allowed in (('bedload', (0, 2, 8, 11)), ('exchange', (0, 1, 2)),
            ('settling_mode', (0,)), ('hiding_exposure', (0, 2, 3, 4, 5)), ('bed_slope', (0, 2, 3, 5))):
        value = n.get(key, 0)
        if type(value) is not int or value not in allowed:
            raise ValueError('Unsupported live-transport '+key)
    for key in ('bank_failure', 'bank_morphology',
                'horizontal_viscosity', 'morphological_acceleration'):
        value = internal.get(key, {}).get('enabled', False)
        if type(value) is not bool or value:
            raise ValueError('Live transport excludes '+key)
    if internal.get('bank_failure', {}).get('front_tracking_enabled', False):
        raise ValueError('Live transport excludes persistent bank fronts')
    avalanche = internal.get('general_avalanche',{})
    if type(avalanche.get('enabled',False)) is not bool:
        raise ValueError('General avalanche switch must be boolean')
    if avalanche.get('enabled',False):
        for key, default in (('wet_maximum_slope',10.),('dry_maximum_slope',1.),
                ('avalanche_time_s',86400.),('relaxation_length_m',1.)):
            number(avalanche.get(key,default),key,positive=True)
        number(avalanche.get('wet_depth_switch_m',.1),'avalanche wet depth')
    adaptation = internal.get('bedload_adaptation', {})
    if type(adaptation.get('enabled', False)) is not bool:
        raise ValueError('Live transport adaptation switch must be boolean')
    if adaptation.get('enabled', False):
        if n['bedload'] != 2:
            raise ValueError('Live transport adaptation requires MPM2 bedload')
        number(adaptation.get('length_diameters'), 'adaptation length', positive=True)
    if n.get('horizontal_viscosity',0.) != 0.:
        raise ValueError('Use the explicit horizontal_viscosity_m2_s project control')
    viscosity=number(n.get('horizontal_viscosity_m2_s',0.),'horizontal viscosity')
    if viscosity > 0. and n.get('horizontal_viscosity_formulation','depth_conservative') != 'velocity_laplacian':
        raise ValueError('Live transport viscosity requires the velocity_laplacian formulation')
    allow_viscous_profile = (
        gpu_integrator == 'implicit_fwave_midpoint' and
        gpu_launch_mode == 'plain' and viscosity > 0. and
        n.get('horizontal_viscosity_formulation') == 'velocity_laplacian')
    if number(n.get('morphological_factor', 1.), 'MorFac', positive=True) != 1.:
        raise ValueError('Live transport requires MorFac 1')
    number(n.get('hindered_settling_reference_concentration_kg_m3', 1000.),
           'hindered settling reference concentration', positive=True)
    end = number(n.get('end_time_s'), 'end time', positive=True)
    implicit_hydraulic = disabled and not project.get('sediment_fractions')
    if implicit_hydraulic:
        if (initial.get('active_layer_mode', 'constant') != 'constant' or
                initial.get('active_layer_thickness_m', 0.) != 0.):
            raise ValueError('Implicit hydraulic material requires its native inactive single bed layer')
        boundaries = [CutBoundary(**row) for row in project['boundaries'].values()]
        _validate_ordinary_boundaries(
            boundaries, set(), end,
            allow_viscous_profile=allow_viscous_profile)
        return
    fractions = project.get('sediment_fractions')
    single_law11 = schema_v2_law11 and fractions is None
    if initial.get('active_layer_mode', 'constant') != 'constant':
        raise ValueError('Live transport requires constant active-layer thickness')
    try:
        boundaries = [CutBoundary(**row) for row in project['boundaries'].values()]
    except (TypeError, KeyError) as error:
        raise ValueError('Malformed live-transport material/boundary controls') from error
    if schema_v2_law11 and any(
            (row.is_inlet and
             row.bedload_boundary_mode != 'equilibrium_capacity') or
            (not row.is_inlet and row.bedload_boundary_mode != 'prescribed')
            for row in boundaries):
        raise ValueError(
            'Schema-v2 law-11 total load requires equilibrium-capacity inlets '
            'and prescribed free-capacity outlets')
    if single_law11:
        if (initial.get('active_layer_thickness_m', 0.) != 0. or
                initial.get('substrate_layer_count', 1) != 1):
            raise ValueError(
                'Schema-v2 law-11 total load requires the native one-layer '
                'single-material layout')
        materials = []
        ids = {'single'}
    else:
        count = initial.get('substrate_layer_count')
        if type(count) is not int or not 1 <= count <= 32:
            raise ValueError('Live transport requires 1 to 32 substrate layers')
        number(initial.get('active_layer_thickness_m'),
               'active-layer thickness', positive=True)
        if count > 1:
            number(initial.get('substrate_layer_thickness_m'),
                   'substrate-layer thickness', positive=True)
        if not isinstance(fractions, list) or not 1 <= len(fractions) <= 32:
            raise ValueError('Live transport requires 1 to 32 explicit sediment classes')
        try:
            materials = [CutSedimentFraction(**row) for row in fractions]
        except (TypeError, KeyError) as error:
            raise ValueError('Malformed live-transport material controls') from error
        ids = {row.fraction_id for row in materials}
        if len(ids) != len(materials) or any(
                row.sediment_type != 'sand' for row in materials):
            raise ValueError('Live transport requires unique noncohesive sand classes')
    secondary = internal.get('secondary_flow', {})
    enabled, model = secondary.get('enabled', False), secondary.get('model', 'equilibrium')
    direction, beta = secondary.get('transport_direction', True), secondary.get('momentum_beta', 0.)
    if type(enabled) is not bool or model not in ('equilibrium', 'prognostic') or type(direction) is not bool:
        raise ValueError('Invalid live-transport secondary controls')
    number(beta, 'secondary momentum Beta'); number(secondary.get('espir', 1.), 'secondary espir')
    if enabled and model == 'equilibrium' and (not direction or beta != 0.):
        raise ValueError('Equilibrium secondary flow requires direction-only coupling and zero Beta')
    if n['bedload'] == 0:
        if enabled or n.get('mpm_friction_mode', 0) != 0 or n.get('hiding_exposure', 0) or n.get('bed_slope', 0):
            raise ValueError('No-bedload transport requires neutral MPM and secondary controls')
        if _has_bed_feed(project.get('boundaries', {})):
            raise ValueError('No-bedload transport forbids bed feed')
    elif n['bedload'] == 8:
        if n.get('mpm_friction_mode', 0) != 0 or n.get('hiding_exposure', 0) or exchange != 0:
            raise ValueError('Engelund-Hansen requires neutral MPM/hiding and bedload-only exchange0')
        number(n.get('engelund_hansen_chezy', 28.8), 'Engelund-Hansen Chezy', positive=True)
    elif n['bedload'] == 11:
        if (n.get('mpm_friction_mode', 0) != 0 or
                n.get('hiding_exposure', 0) != 0 or
                n.get('bed_slope', 0) != 0 or exchange != 0 or enabled):
            raise ValueError(
                'Schema-v2 law-11 total load requires total shear with hiding, '
                'slope, secondary flow and separate suspension off')
    elif (type(n.get('mpm_friction_mode')) is not int or n['mpm_friction_mode'] not in (0,1) or
            (n.get('hiding_exposure', 0) and len(materials) < 2)):
        raise ValueError('Live MPM transport requires friction0/1 and multiple classes for hiding')
    for key, default in (('mpm_cf', .01), ('mpm_critical_shields', 0.)):
        number(n.get(key, default), key)
    if n['bedload'] == 2 and n['mpm_friction_mode'] == 1 and not math.isclose(
            n.get('mpm_critical_shields', .047), .047, rel_tol=0., abs_tol=1.e-14):
        raise ValueError('Manning MPM requires critical Shields 0.047')
    for key, default in (('hiding_parker_alpha', .6), ('hiding_wu_exponent', -.6)):
        value = n.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Invalid live-transport '+key)
    _validate_ordinary_boundaries(
        boundaries,ids,end,
        allow_viscous_profile=allow_viscous_profile)
    bedform = internal.get('bedform_roughness', {}).get('enabled', False)
    if type(bedform) is not bool:
        raise ValueError('Bedform roughness switch must be boolean')
    if bedform:
        from braids_core.gpu_capabilities import has_suspended_transport
        concentrations = [tuple(row.fraction_concentration_kg_m3.values()) for row in boundaries]
        schedules = [series.concentration_kg_m3 for row in boundaries
                     for series in row.fraction_sediment_series.values()]
        modes = [row.mode for row in boundaries]
        if (n['bedload'] != 2 or n.get('mpm_friction_mode') != 1 or
                n.get('bed_slope', 0) != 0 or n.get('hiding_exposure', 0) != 0 or
                initial.get('substrate_layer_count') != 1 or enabled or
                avalanche.get('enabled',False) or viscosity != 0. or
                adaptation.get('enabled', False) or
                has_suspended_transport(n['exchange'], (), concentrations, schedules) or
                sorted(modes) != ['discharge', 'stage'] or
                any(row.hydraulic_series is not None or
                    row.fraction_sediment_series for row in boundaries)):
            raise ValueError('Bedform roughness requires the qualified unscheduled discharge/stage, '
                             'unsuspended one-substrate MPM-friction1 envelope with '
                             'slope/hiding/secondary/adaptation/viscosity/avalanche off')
        number(project.get('roughness', {}).get('default_manning_n', n.get('manning', .03)),
               'bedform Manning coefficient', positive=True)


def _recursive_bedform_project_scope(project: dict) -> bool:
    """Recognize and validate the qualified ON/OFF recursive bedform pair."""
    numerics = project['numerics']
    initial = project.get('initial', {})
    internal = project.get('internal_conditions', {})
    bedform = internal.get('bedform_roughness', {}).get('enabled', False)
    modes = sorted(row.get('mode') for row in project.get('boundaries', {}).values())
    candidate = bedform is True or modes == ['discharge', 'stage']
    if not candidate:
        return False
    secondary = internal.get('secondary_flow', {})
    boundaries = project.get('boundaries', {}).values()
    if (type(bedform) is not bool or modes != ['discharge', 'stage'] or
            numerics.get('bedload') != 2 or numerics.get('exchange') != 0 or
            numerics.get('mpm_friction_mode') != 1 or
            numerics.get('bed_slope', 0) != 0 or
            numerics.get('hiding_exposure', 0) != 0 or
            numerics.get('horizontal_viscosity_m2_s', 0.) != 0. or
            numerics.get('morphological_factor', 1.) != 1. or
            initial.get('active_layer_mode', 'constant') != 'constant' or
            initial.get('substrate_layer_count') != 1 or
            secondary.get('enabled', False) is not False or
            any(internal.get(name, {}).get('enabled', False) for name in
                ('bedload_adaptation', 'general_avalanche', 'bank_failure',
                 'bank_morphology', 'morphological_acceleration')) or
            any(row.get('hydraulic_series') is not None or
                row.get('sediment_series') is not None or
                bool(row.get('fraction_sediment_series'))
                for row in boundaries)):
        raise ValueError('Recursive bedform AMR requires the exact unscheduled mode-1/'
                         'stage-2, unsuspended one-substrate MPM-friction1 ON/OFF '
                         'envelope with slope/hiding/secondary/adaptation/viscosity/'
                         'avalanche/banks off and MorFac 1')
    return True


def _recursive_bank_bedform_avalanche_project_scope(
        project: dict, *, gpu_amr_evolving_banks: bool) -> bool:
    """Recognize the exact recursive mode-2/3 bank/bedform/avalanche route."""
    numerics = project.get('numerics', {})
    amr = numerics.get('amr', {})
    initial = project.get('initial', {})
    internal = project.get('internal_conditions', {})
    bedform = internal.get('bedform_roughness', {}).get('enabled', False)
    if type(bedform) is not bool:
        raise ValueError('Bedform roughness switch must be boolean')
    if not bedform:
        return False
    bank = internal.get('bank_failure', {})
    avalanche = internal.get('general_avalanche', {})
    secondary = internal.get('secondary_flow', {})
    boundaries = list(project.get('boundaries', {}).values())
    if (gpu_amr_evolving_banks is not True or
            amr.get('enabled') is not True or amr.get('subcycling') is not True or
            bank.get('enabled') is not True or
            bank.get('toe_erosion_mode') not in (
                'noncohesive_transport_capacity',
                'distributed_slope_transport') or
            bank.get('front_tracking_enabled', False) is not False or
            avalanche.get('enabled') is not True or
            secondary.get('enabled', False) is not False or
            numerics.get('bedload') != 2 or numerics.get('exchange') != 0 or
            numerics.get('mpm_friction_mode') != 1 or
            numerics.get('bed_slope', 0) != 0 or
            numerics.get('hiding_exposure', 0) != 0 or
            numerics.get('horizontal_viscosity_m2_s', 0.) != 0. or
            numerics.get('morphological_factor', 1.) != 1. or
            initial.get('active_layer_mode', 'constant') != 'constant' or
            initial.get('substrate_layer_count') != 1 or
            any(internal.get(name, {}).get('enabled', False) for name in
                ('bedload_adaptation','bank_morphology','morphological_acceleration')) or
            len(boundaries) != 1 or boundaries[0].get('mode') != 'stage' or
            boundaries[0].get('hydraulic_series') is not None or
            boundaries[0].get('sediment_series') is not None or
            bool(boundaries[0].get('fraction_sediment_series')) or
            bool(boundaries[0].get('fraction_concentration_kg_m3')) or
            boundaries[0].get('concentration_kg_m3', 0) != 0):
        raise ValueError('Recursive bank bedform composition requires recursive GPU AMR, '
                         'explicit evolving physical-toe mode-2 or distributed mode-3 banks, '
                         'parent avalanche, '
                         'one constant-stage boundary, unsuspended one-substrate '
                         'MPM-friction1, slope/hiding/secondary/adaptation/viscosity off '
                         'and MorFac 1')
    return True


def _validate_on_demand_amr_project(project: dict, *, gpu_integrator: str,
                                    gpu_launch_mode: str | None,
                                    gpu_amr_evolving_banks: bool = False) -> None:
    """Preflight the explicitly selected synchronous/recursive bank-AMR scope.

    The schema-6 prepared reader remains the authoritative check of the
    exported original native controls and geometry.
    """
    numerics = project["numerics"]
    amr = numerics.get("amr", {})
    internal = project.get("internal_conditions", {})
    bank = internal.get("bank_failure", {})
    secondary = internal.get("secondary_flow", {})
    boundaries = project.get('boundaries', {})
    bedform_composition = _recursive_bank_bedform_avalanche_project_scope(
        project, gpu_amr_evolving_banks=gpu_amr_evolving_banks)
    live = gpu_amr_evolving_banks and not (len(boundaries) == 1 and
        next(iter(boundaries.values())).get('mode') == 'stage')
    if gpu_integrator != "ssp33" or gpu_launch_mode != "plain":
        raise ValueError("On-demand GPU bank AMR requires Plain launches and SSPRK(3,3)")
    for name in ("enabled", "subcycling", "refine_wet_dry", "refine_hydraulics",
                 "refine_walls", "refine_bathymetry", "refine_sediment"):
        if type(amr.get(name)) is not bool:
            raise ValueError("On-demand GPU bank AMR needs explicit boolean " + name)
    if not amr.get("enabled", False):
        raise ValueError("On-demand GPU bank AMR requires AMR enabled")
    if amr['subcycling']:
        toe_mode = bank.get('toe_erosion_mode')
        if not (toe_mode == 'noncohesive_transport_capacity' or
                (toe_mode == 'distributed_slope_transport' and
                 gpu_amr_evolving_banks)):
            raise ValueError('Recursive GPU bank AMR requires physical-toe mode 2 or explicit evolving mode 3')
        if live and toe_mode == 'noncohesive_transport_capacity':
            raise ValueError('Recursive surveyed bank sections currently require distributed mode 3')
    if (type(amr.get("refinement_ratio", 2)) is not int or amr.get("refinement_ratio", 2) != 2 or
            not amr.get("refine_wet_dry", False) or
            not amr.get("refine_hydraulics", False)):
        raise ValueError("On-demand GPU bank AMR requires 2:1 wet/dry and hydraulic indicators")
    extended = any(amr[name] for name in ("refine_walls", "refine_bathymetry", "refine_sediment"))
    if extended and not gpu_amr_evolving_banks:
        raise ValueError("Extended AMR indicators require the explicit evolving-bank option")
    if extended:
        for name in ("bed_tolerance_m", "concentration_tolerance_kg_m3", "composition_tolerance",
                     "bed_change_tolerance_m"):
            value = amr.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0 or
                    (name == "composition_tolerance" and value > 1)):
                raise ValueError("Invalid finite positive extended AMR indicator control " + name)
    if (numerics.get("bedload") != 2 or numerics.get("exchange") != 0 or
            numerics.get("mpm_friction_mode") != 1 or
            (not live and numerics.get("bed_slope") != 0) or
            (not live and numerics.get("hiding_exposure") != 0) or
            numerics.get("settling_mode", 0) != 0):
        raise ValueError("On-demand GPU bank AMR requires MPM bedload 2, friction 1, and suspended exchange off" +
                         ("" if live else "; legacy slope and hiding must be off"))
    distributed = bank.get("toe_erosion_mode") == "distributed_slope_transport"
    if (not bank.get("enabled", False) or
            bank.get("toe_erosion_mode") not in ("noncohesive_transport_capacity", "distributed_slope_transport") or
            (distributed and not gpu_amr_evolving_banks)):
        raise ValueError("On-demand GPU bank AMR requires mode 2, or explicitly enabled evolving mode 3")
    if distributed:
        for key, default in (("distributed_transport_fraction", .1),
                             ("distributed_critical_slope", .35), ("critical_undercut_m", .1)):
            value = bank.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or
                    (not 0 <= value <= 1 if key == "distributed_transport_fraction" else value <= 0)):
                raise ValueError("Invalid distributed-bank control " + key)
    if bank.get("front_tracking_enabled", False):
        raise ValueError("GPU persistent bank-front tracking is excluded")
    if bedform_composition or (distributed and not live):
        if secondary.get("enabled", False):
            raise ValueError("Stage-only composed/distributed bank AMR requires secondary flow off")
    elif not secondary.get("enabled", False) or secondary.get("model") != "prognostic":
        raise ValueError("On-demand GPU bank AMR requires enabled prognostic secondary flow")
    for name in ("bank_morphology", "bedload_adaptation", "bedform_roughness",
                  "morphological_acceleration"):
        if name == 'bedload_adaptation' and live:
            continue
        if name == 'bedform_roughness' and bedform_composition:
            continue
        if internal.get(name, {}).get("enabled", False):
            raise ValueError("On-demand GPU bank AMR does not support " + name)
    if project.get("initial", {}).get("active_layer_mode", "constant") != "constant":
        raise ValueError("On-demand GPU bank AMR requires a constant active-layer thickness")
    if live:
        _validate_live_amr_project(project)
        return
    if len(boundaries) != 1 or next(iter(boundaries.values())).get("mode") != "stage":
        raise ValueError("On-demand GPU bank AMR requires exactly one constant stage outlet")
    boundary = next(iter(boundaries.values()))
    if (boundary.get("hydraulic_series") is not None or
            boundary.get("sediment_series") is not None or
            boundary.get("fraction_sediment_series") or
            boundary.get("fraction_concentration_kg_m3") or
            boundary.get("concentration_kg_m3", 0) or _has_bed_feed(boundaries)):
        raise ValueError("On-demand GPU bank AMR does not support scheduled or sediment boundary forcing")


def _validate_general_on_demand_amr_project(project: dict, *, gpu_integrator: str,
                                            gpu_launch_mode: str | None) -> None:
    """Preflight the explicitly selected synchronous/recursive nonbank route."""
    amr = project['numerics'].get('amr', {})
    if gpu_integrator != 'ssp33' or gpu_launch_mode != 'plain':
        raise ValueError('General transport AMR requires Plain SSPRK(3,3)')
    for name in ('enabled', 'subcycling', 'refine_wet_dry', 'refine_hydraulics',
                 'refine_walls', 'refine_bathymetry', 'refine_sediment'):
        if type(amr.get(name)) is not bool:
            raise ValueError('General transport AMR needs explicit boolean '+name)
    if (not amr['enabled'] or
            not amr['refine_wet_dry'] or not amr['refine_hydraulics'] or
            type(amr.get('refinement_ratio', 2)) is not int or
            amr.get('refinement_ratio', 2) != 2):
        raise ValueError('General transport AMR requires 2:1 wet/dry and hydraulic refinement')
    if any(amr[name] for name in ('refine_walls', 'refine_bathymetry', 'refine_sediment')):
        for name in ('bed_tolerance_m', 'concentration_tolerance_kg_m3',
                     'composition_tolerance', 'bed_change_tolerance_m'):
            value = amr.get(name)
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    value <= 0 or (name == 'composition_tolerance' and value > 1)):
                raise ValueError('Invalid general AMR indicator control '+name)
    if project.get('internal_conditions', {}).get('bank_failure', {}).get('enabled', False):
        raise ValueError('General transport AMR requires bank failure off')
    internal = project.get('internal_conditions', {})
    bedform_branch = _recursive_bedform_project_scope(project)
    boundaries = list(project.get('boundaries', {}).values())
    if bedform_branch:
        if (amr['subcycling'] is not True or len(boundaries) != 2 or
                sorted(row.get('mode') for row in boundaries) != ['discharge', 'stage']):
            raise ValueError('Recursive bedform AMR requires one unscheduled mode-1 discharge inlet '
                             'and one constant stage outlet')
    elif (len(boundaries) != 2 or
            sum(row.get('mode') == 'critical' for row in boundaries) != 1 or
            sum(row.get('mode') in ('inlet_stage', 'inlet_normal', 'inlet_critical',
                                    'inlet_depth') for row in boundaries) != 1):
        raise ValueError('General transport AMR requires one surveyed inlet and one critical outlet')
    _validate_general_live_transport_project(project, gpu_integrator=gpu_integrator,
                                             gpu_launch_mode=gpu_launch_mode)
    if amr['subcycling']:
        numerics = project['numerics']
        secondary = internal.get('secondary_flow', {})
        viscosity = numerics.get('horizontal_viscosity_m2_s', 0.)
        if bedform_branch:
            return
        if (numerics.get('bedload') != 2 or numerics.get('exchange') != 2 or
                secondary.get('enabled') is not True or
                secondary.get('model') != 'prognostic' or
                type(viscosity) not in (int, float) or
                not math.isfinite(viscosity) or viscosity < 0. or
                (viscosity > 0. and
                 numerics.get('horizontal_viscosity_formulation') !=
                 'velocity_laplacian')):
            raise ValueError('Recursive general AMR requires MPM2, exchange 2 and prognostic secondary flow; '
                             'positive viscosity requires velocity_laplacian mode1/geometry1; '
                             'banks are unsupported')


def validate_gpu_project(project: dict, *, gpu_integrator: str = "ssp43",
                         gpu_launch_mode: str | None = None,
                         gpu_numerical_variant: str = "auto",
                         gpu_amr_evolving_banks: bool = False) -> None:
    """Reject unsupported GPU preparation choices before writing a native run."""
    if gpu_numerical_variant == INTEGRATED:
        gpu_numerical_variant = resolve_project_route(project)
    numerics = project["numerics"]
    if (gpu_numerical_variant not in (_GENERAL_LIVE_TRANSPORT_VARIANT,_GENERAL_LIVE_BANK_VARIANT,
                                     _ON_DEMAND_GENERAL_AMR_VARIANT) and
            numerics.get('horizontal_viscosity_m2_s',0.) != 0.):
        raise ValueError('This GPU route does not yet support project horizontal viscosity')
    if type(gpu_amr_evolving_banks) is not bool or (gpu_amr_evolving_banks and
            gpu_numerical_variant != _ON_DEMAND_AMR_VARIANT):
        raise ValueError('Explicit evolving-bank selection requires the on-demand AMR variant')
    if numerics["scheme"] != 6:
        raise ValueError("GPU preparation requires scheme 6 (augmented f-wave)")
    if gpu_numerical_variant == _ON_DEMAND_AMR_VARIANT:
        _validate_on_demand_amr_project(project, gpu_integrator=gpu_integrator,
                                        gpu_launch_mode=gpu_launch_mode,
                                        gpu_amr_evolving_banks=gpu_amr_evolving_banks)
        return
    if gpu_numerical_variant == _ON_DEMAND_GENERAL_AMR_VARIANT:
        _validate_general_on_demand_amr_project(project, gpu_integrator=gpu_integrator,
                                                gpu_launch_mode=gpu_launch_mode)
        return
    if numerics.get("amr", {}).get("enabled", False):
        raise ValueError("The selectable GPU worker does not yet support AMR; turn it off or use CPU")
    try:
        gpu_integrator_label(gpu_integrator)
    except ValueError as error:
        raise ValueError("Select a supported GPU time integrator before preparation") from error
    if gpu_integrator == 'implicit_fwave_midpoint' and gpu_launch_mode not in (None, 'plain'):
        raise ValueError('GPU implicit f-wave midpoint requires Plain launches')
    if gpu_integrator not in ("ssp43", "ssp33", 'implicit_fwave_midpoint'):
        raise ValueError("Select a supported GPU time integrator before preparation")
    if (gpu_integrator == 'implicit_fwave_midpoint' and
            gpu_numerical_variant != _GENERAL_LIVE_TRANSPORT_VARIANT):
        raise ValueError('GPU implicit f-wave midpoint requires the fixed-grid common transport route')
    if gpu_launch_mode is not None and gpu_launch_mode not in ("graph", "plain"):
        raise ValueError("Select a supported GPU launch mode before preparation")
    if gpu_integrator == "ssp33" and gpu_launch_mode not in (None, "plain"):
        raise ValueError("GPU SSPRK(3,3) requires Plain launches")
    if gpu_numerical_variant == _GENERAL_LIVE_BANK_VARIANT:
        _validate_general_live_bank_project(project, gpu_integrator=gpu_integrator,
                                           gpu_launch_mode=gpu_launch_mode)
        return
    if gpu_numerical_variant == _GENERAL_LIVE_TRANSPORT_VARIANT:
        _validate_general_live_transport_project(project, gpu_integrator=gpu_integrator,
                                                 gpu_launch_mode=gpu_launch_mode)
        return
    bedlaw = numerics.get("bedload", 2)
    exchange = numerics.get("exchange", 0)
    if bedlaw not in (0, 2):
        raise ValueError("The selected GPU backend supports only no-bedload or MPM bedload 2")
    if exchange not in (0, 1, 2):
        raise ValueError("The selected GPU backend does not support cohesive or mixed suspended exchange")
    if exchange in (1, 2) and gpu_integrator != "ssp33":
        raise ValueError("GPU suspended exchange modes 1/2 require SSPRK(3,3); select it before Prepare Run")
    if exchange in (1, 2) and gpu_launch_mode not in (None, "plain"):
        raise ValueError("GPU suspended exchange modes 1/2 require Plain launches and SSPRK(3,3)")
    internal = project.get("internal_conditions", {})
    if exchange in (1, 2) and numerics.get("settling_mode", 0) != 0:
        raise ValueError("GPU hindered settling is deferred; select constant per-class settling velocity")
    if internal.get("morphological_acceleration", {}).get("enabled", False):
        raise ValueError("GPU morphological acceleration is deferred; turn it off and use MorFac 1")
    if project.get("initial", {}).get("active_layer_mode", "constant") != "constant":
        raise ValueError("GPU water-depth-scaled active layer is deferred; select constant thickness")
    for name, boundary in project.get("boundaries", {}).items():
        if boundary.get("mode") in _DEFERRED_BOUNDARY_MODES:
            raise ValueError(f"GPU boundary {name!r} uses a deferred mode: {boundary['mode']}")
        if boundary.get("total_discharge_profile", "stock_conveyance") != "stock_conveyance":
            raise ValueError(f"GPU boundary {name!r} uses a deferred viscosity-equilibrated inlet profile")
    bank_failure = internal.get("bank_failure", {})
    if bank_failure.get("front_tracking_enabled", False):
        raise ValueError("GPU persistent bank-front tracking is excluded from this noncohesive goal")
    if bank_failure.get("enabled", False) and bank_failure.get(
            "toe_erosion_mode", "cohesive_excess_shear") == "cohesive_excess_shear":
        raise ValueError("GPU cohesive bank failure is excluded from this noncohesive goal")
    if (bank_failure.get("enabled", False) and
            gpu_numerical_variant in _BANK_VARIANT_MODES and
            bank_failure.get("toe_erosion_mode") != _BANK_VARIANT_MODES[gpu_numerical_variant]):
        raise ValueError("Selected GPU bank variant does not match the project's toe-erosion mode")
    if bedlaw == 0:
        if gpu_integrator != "ssp33" or gpu_launch_mode not in (None, "plain"):
            raise ValueError("GPU no-bedload mode requires Plain launches and SSPRK(3,3)")
        if (numerics.get("mpm_friction_mode", 0) != 0 or numerics.get("bed_slope", 0) != 0 or
                numerics.get("hiding_exposure", 0) != 0 or
                internal.get("secondary_flow", {}).get("enabled", False) or
                internal.get("morphological_acceleration", {}).get("enabled", False)):
            raise ValueError("GPU no-bedload mode does not support MPM, slope, hiding, secondary flow or morphological acceleration controls")
        if _has_bed_feed(project.get("boundaries", {})):
            raise ValueError("GPU no-bedload mode forbids prescribed or equilibrium bed feed")
    if gpu_numerical_variant in _MPM_ONLY_VARIANTS and bedlaw != 2:
        raise ValueError("Selected GPU numerical variant requires MPM bedload 2")
    if gpu_numerical_variant == "general_avalanche_plain_ssp33" and bedlaw != 0:
        raise ValueError("Selected GPU general-avalanche variant requires no bedload")
    if gpu_numerical_variant == "scheduled_inlet_ssp33" and bedlaw == 0 and exchange in (1, 2):
        raise ValueError("Scheduled GPU exchange 1/2 with no bedload is not yet qualified")
    if (internal.get("bedform_roughness", {}).get("enabled", False) and
            gpu_numerical_variant != "bedform_feedback_ssp33"):
        raise ValueError("GPU bedform roughness requires the bedform_feedback_ssp33 numerical variant")
    if (internal.get("bedload_adaptation", {}).get("enabled", False) and
            gpu_numerical_variant not in (
                "bedload_adaptation_plain_ssp33", _GRADED_LIVE_BEDLOAD_VARIANT)):
        raise ValueError("GPU bedload adaptation requires a compatible adaptation numerical variant")
    if gpu_numerical_variant == "bedload_adaptation_plain_ssp33" and not internal.get(
            "bedload_adaptation", {}).get("enabled", False):
        raise ValueError("GPU bedload adaptation variant requires the project adaptation switch")
    if internal.get("bank_morphology", {}).get("enabled", False):
        raise ValueError("The selected GPU backend does not support legacy bank morphology")
    if (internal.get("bank_failure", {}).get("enabled", False) and
            gpu_numerical_variant not in (
                "physical_toe_mode2_plain_ssp33", "physical_toe_mode2_avalanche_ssp33",
                "distributed_slope_mode3_plain_ssp33",
                "distributed_slope_mode3_avalanche_ssp33")):
        raise ValueError("GPU bank failure requires an explicit matching bank numerical variant")
    if (internal.get("general_avalanche", {}).get("enabled", False) and
            gpu_numerical_variant not in (
                "general_avalanche_plain_ssp33", "physical_toe_mode2_avalanche_ssp33",
                "distributed_slope_mode3_avalanche_ssp33")):
        raise ValueError("GPU general avalanche requires an explicit avalanche numerical variant")
    if gpu_numerical_variant in _PLAIN_SSP33_VARIANTS and (
            gpu_integrator != "ssp33" or gpu_launch_mode not in (None, "plain")):
        raise ValueError("Selected GPU numerical variant requires Plain launches and SSPRK(3,3)")
    if gpu_numerical_variant == _GRADED_LIVE_BEDLOAD_VARIANT:
        _validate_graded_live_bedload_project(project)


def validate_gpu_worker_variant(scope: dict, gpu_numerical_variant: str) -> None:
    """Reject a selected worker route that normal QGIS preparation cannot produce."""
    if gpu_numerical_variant not in scope["supported_gpu_variants"]:
        raise ValueError("GPU variant is not in the selected worker's supported scope")
    if gpu_numerical_variant in scope["excluded_gpu_variants"]:
        raise ValueError("GPU variant is excluded by this source worker")
    external = scope.get("requires_external_prepared_sidecars", {})
    if gpu_numerical_variant in external:
        raise ValueError("GPU variant requires external preparation not available from QGIS: " +
                         ", ".join(external[gpu_numerical_variant]))


def prepare_gpu_run(
    project_path: str | Path, *, worker_root: str | Path,
    expected_manifest_sha256: str, python_executable: str | Path,
    expected_python_sha256: str,
    gpu_integrator: str = "ssp43",
    gpu_launch_mode: str = "graph",
    gpu_numerical_variant: str = "auto",
    gpu_amr_evolving_banks: bool = False,
    gpu_amr_selector_policy: str | None = None,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    native_ready: Callable[[Path], None] | None = None,
) -> tuple[Path, Path]:
    """Prepare a native deck, then export its GPU state in a separate process.

    Returns ``(native_run_root, gpu_prepared_root)``. A cancelled/failed GPU
    export never has a committed receipt; its native deck remains available
    for diagnosis and a fresh preparation attempt.
    """
    project_path = filesystem_path(Path(project_path).expanduser()).resolve()
    if project_path.is_file():
        reject_raw_gpu_persistent_front(project_path)
    project = load_project(project_path)
    if (QGIS_PRODUCT_KEY in project and
            gpu_integrator != 'implicit_fwave_midpoint'):
        raise ValueError(
            'The saved Q10 GPU product requires implicit_fwave_midpoint; '
            'the integrator is never changed silently')
    if gpu_numerical_variant == INTEGRATED:
        gpu_numerical_variant = resolve_project_route(project)
    validate_gpu_project(project, gpu_integrator=gpu_integrator,
                         gpu_launch_mode=gpu_launch_mode,
                         gpu_numerical_variant=gpu_numerical_variant,
                         gpu_amr_evolving_banks=gpu_amr_evolving_banks)
    worker = inspect_gpu_source_worker(
        worker_root, expected_manifest_sha256=expected_manifest_sha256)
    if QGIS_PRODUCT_KEY in project:
        verify_gpu_worker_for_project(project, worker["scope"])
    validate_gpu_worker_variant(worker["scope"], gpu_numerical_variant)
    validate_worker_integrator_capability(worker['scope'], gpu_integrator,
                                          gpu_numerical_variant, gpu_launch_mode)
    canonical_execution_profile = (
        isinstance(project.get(EXECUTION_PROFILE_BINDING_KEY), dict) and
        project[EXECUTION_PROFILE_BINDING_KEY].get('state') == 'canonical')
    # A canonical schema-v2 document records sediment-method identity; it does
    # not by itself select a hydraulic time integrator.  The first public
    # law-11 profile is admitted only on the common fixed-grid route and owns
    # the implicit backend sidecar.  Recursive/bank AMR routes instead retain
    # their independently qualified Plain SSPRK(3,3) contract.
    public_schema_v2_profile = (
        canonical_execution_profile and
        gpu_numerical_variant == _GENERAL_LIVE_TRANSPORT_VARIANT)
    if public_schema_v2_profile:
        if (gpu_integrator != 'implicit_fwave_midpoint' or
                gpu_launch_mode not in (None, 'plain')):
            raise ValueError(
                'Canonical schema-v2 GPU preparation requires explicitly selected '
                'implicit_fwave_midpoint with Plain launches')
        validate_public_law11_total_preparation(
            project, worker['scope'], gpu_numerical_variant)
    verify_general_amr_selector_scope(worker['scope'], gpu_numerical_variant,
                                      gpu_amr_selector_policy)
    recursive=project['numerics'].get('amr', {}).get('subcycling', False)
    recursive_bank_bedform_avalanche = False
    if recursive and gpu_numerical_variant in (
            _ON_DEMAND_AMR_VARIANT,_ON_DEMAND_GENERAL_AMR_VARIANT):
        if gpu_numerical_variant == _ON_DEMAND_GENERAL_AMR_VARIANT:
            avalanche=project.get('internal_conditions',{}).get(
                'general_avalanche',{}).get('enabled',False)
            bedform=project.get('internal_conditions',{}).get(
                'bedform_roughness',{}).get('enabled',False)
            boundary_modes=sorted(row.get('mode') for row in
                project.get('boundaries',{}).values())
            bedform_branch=(bedform or boundary_modes==['discharge','stage'])
            viscosity=project['numerics'].get('horizontal_viscosity_m2_s',0.)
            if bedform_branch:
                initial_data=dict(bedload_adaptation=0,general_avalanche=0,
                    bank_toe_erosion_mode=0,bank_failure=0,
                    bank_front_tracking=0,bedlaw=2,mpm_friction_mode=1,
                    bed_slope=0,hiding_exposure=0,exchange_mode=0,
                    secondary_flow_model=0,horizontal_viscosity=0.,
                    horizontal_viscosity_mode=0,horizontal_viscosity_geometry=0,
                    bedform_roughness=int(bedform),mode=[1,2],sampled=[0],
                    boundary_bed_offset=[0.],section_control=[0,0],
                    nseries=[0,0],nhydro_series=[0,0],
                    nfraction_series=[[0,0]],forcing=[[1.,0.],[0.,0.]],
                    morphological_factor=1.,nsubstrate_layers=1,
                    suspended_transport_active=0,fraction_concentration=[[0.,0.]],
                    suspended_indices=[3],state=[[0.],[0.],[0.],[0.]])
            else:
                initial_data=dict(bedload_adaptation=0,
                    general_avalanche=int(avalanche),
                    bank_toe_erosion_mode=0,bank_failure=0,
                    bank_front_tracking=0,bedlaw=2,exchange_mode=2,
                    secondary_flow_model=2,horizontal_viscosity=float(viscosity),
                    horizontal_viscosity_mode=1 if viscosity > 0. else 0,
                    horizontal_viscosity_geometry=1 if viscosity > 0. else 0,
                    bedform_roughness=0,mode=[9,4])
            recursive_prepared=dict(variant=gpu_numerical_variant,
                amr_settings=dict(subcycling=True),bank_mode=0,
                evolving_banks=False,general_transport=True,
                checkpoint_schema='recursive_general_amr_v1',
                session_manifest=dict(schema=3),
                initial_data=initial_data)
        else:
            toe=project.get('internal_conditions',{}).get(
                'bank_failure',{}).get('toe_erosion_mode')
            bank_mode=(2 if toe=='noncohesive_transport_capacity' else
                       3 if toe=='distributed_slope_transport' else -1)
            composition=_recursive_bank_bedform_avalanche_project_scope(
                project,gpu_amr_evolving_banks=gpu_amr_evolving_banks)
            recursive_bank_bedform_avalanche = composition
            initial_data=dict(bedload_adaptation=0,bedform_roughness=0)
            if composition:
                initial_data.update(bank_toe_erosion_mode=bank_mode,bank_failure=1,
                    general_avalanche=1,bedlaw=2,mpm_friction_mode=1,
                    bed_slope=0,hiding_exposure=0,exchange_mode=0,
                    secondary_flow_model=0,nsubstrate_layers=1,nbed_layers=2,
                    cpu_time_integrator=33,suspended_transport_active=0,
                    horizontal_viscosity=0.,horizontal_viscosity_mode=0,
                    horizontal_viscosity_geometry=0,morphological_factor=1.,
                    morfac=1.,bank_front_tracking=0,bank_avalanche=0,
                    bank_erosion_fraction=0,morphological_acceleration=0,nmorfac=0,
                    mode=[2],nb=1,nf=1,sampled=[0],boundary_bed_offset=[0.],
                    section_control=[0],nseries=[0],nhydro_series=[0],
                    nfraction_series=[[0]],fraction_concentration=[[0.]],
                    suspended_indices=[3],state=[[1.],[0.],[0.],[0.]],
                    bedform_roughness=1)
            recursive_prepared=dict(variant=gpu_numerical_variant,
                amr_settings=dict(subcycling=True),bank_mode=bank_mode,
                evolving_banks=gpu_amr_evolving_banks,
                checkpoint_schema='recursive_bank_amr_v1',
                session_manifest=dict(schema=2 if gpu_amr_evolving_banks else 1),
                initial_data=initial_data)
        _verify_recursive_amr_capability(worker,recursive_prepared)
        if project.get('internal_conditions', {}).get(
                'bedload_adaptation', {}).get('enabled', False):
            # Refuse an older OFF-only package before mesh/native preparation.
            _verify_recursive_adaptation_capability(
                worker['scope'],gpu_numerical_variant)
    needs_legacy_initializer = (project["numerics"].get("exchange", 0) in (1, 2) and
                                gpu_numerical_variant not in (
                                    _GENERAL_LIVE_TRANSPORT_VARIANT,
                                    _ON_DEMAND_GENERAL_AMR_VARIANT))
    if needs_legacy_initializer and "legacy" not in worker["scope"].get(
            "native_preparation_integrators", ()):
        raise ValueError("The selected GPU worker cannot prepare suspended exchange; select a worker with legacy native initialization")
    python = Path(python_executable).expanduser().resolve()
    if not python.is_file() or _sha256(python) != expected_python_sha256:
        raise ValueError("Selected GPU Python executable differs from its pin")
    if cancelled and cancelled():
        raise InterruptedError("GPU preparation cancelled")
    native_root = prepare_run(
        project_path,
        progress=(None if progress is None else lambda value: progress(float(value) * .75)),
        cancelled=cancelled,
        execution_backend=(BACKEND_GPU if public_schema_v2_profile else 'cpu'),
        product_backend=BACKEND_GPU,
    )
    native_root = filesystem_path(native_root)
    if native_ready:
        native_ready(native_root)
    if cancelled and cancelled():
        raise InterruptedError("GPU preparation cancelled")
    deck = native_root / "case.data"
    if not deck.is_file():
        raise ValueError("Native preparation did not produce case.data")
    if public_schema_v2_profile:
        materialize_execution_profile_sidecar(
            project, native_root, worker['scope'], gpu_numerical_variant)
    prepared_root = native_root / "gpu_prepared"
    stop_file = native_root / (".gpu-prepare-stop-" + secrets.token_hex(16))
    worker_args = [
        "--input", str(deck), "--output", str(prepared_root),
        "--stop-file", str(stop_file),
    ]
    if gpu_numerical_variant in (_GRADED_LIVE_BEDLOAD_VARIANT, _ON_DEMAND_AMR_VARIANT,
                                _ON_DEMAND_GENERAL_AMR_VARIANT,
                                _GENERAL_LIVE_BANK_VARIANT, _GENERAL_LIVE_TRANSPORT_VARIANT):
        worker_args.extend(("--gpu-preparation-variant", gpu_numerical_variant))
    if gpu_amr_evolving_banks:
        worker_args.append('--evolving-banks')
    if recursive_bank_bedform_avalanche:
        worker_args.append('--recursive-bank-bedform-avalanche')
    if gpu_amr_selector_policy is not None:
        worker_args.extend(('--gpu-amr-selector-policy', gpu_amr_selector_policy))
    if needs_legacy_initializer:
        worker_args.extend(("--native-integrator", "legacy"))
    args, cwd = isolated_worker_command(worker["root"], "prepare", worker_args)
    if progress:
        progress(75.)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    environment = dict(os.environ)
    # The worker receives any permitted native initializer explicitly through
    # --native-integrator.  Do not inherit an unrelated diagnostic override.
    environment.pop("BRAIDS_TIME_INTEGRATOR", None)
    process = subprocess.Popen(
        [str(python), *args], cwd=cwd, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        creationflags=flags, env=environment,
    )
    stop_requested = False
    stop_requested_at = None
    while True:
        if cancelled and cancelled() and not stop_requested:
            try:
                stop_file.write_text("BRAIDS GPU preparation cancellation\n", encoding="ascii")
            except OSError:
                _stop_unresponsive_worker(process)
                raise
            stop_requested = True
            stop_requested_at = time.monotonic()
        if stop_requested and time.monotonic() - stop_requested_at > 20:
            _stop_unresponsive_worker(process)
            raise InterruptedError("GPU preparation worker did not stop within 20 s; its process tree was terminated")
        try:
            output, _ = process.communicate(timeout=.25)
            break
        except subprocess.TimeoutExpired:
            if progress:
                if gpu_numerical_variant in (_ON_DEMAND_AMR_VARIANT,
                                             _ON_DEMAND_GENERAL_AMR_VARIANT):
                    try:
                        import json
                        status = json.loads((prepared_root / "preparation_status.json").read_text(encoding="utf-8"))
                        if (status.get("variant") == gpu_numerical_variant and
                                status.get("status") == "preparing"):
                            percent = float(status["percent"])
                            if 0 <= percent <= 100:
                                progress(75. + .2 * percent)
                    except (OSError, ValueError, KeyError, TypeError):
                        pass  # A partially published progress receipt is transient.
                elif (prepared_root / "coupled_input.bin").is_file():
                    progress(90.)
    if stop_requested:
        raise InterruptedError("GPU preparation cancelled")
    if cancelled and cancelled():
        raise InterruptedError("GPU preparation cancelled")
    if process.returncode:
        # Preserve the complete exception chain; the compact dialog is not
        # sufficient evidence when a cleanup error follows an initializer error.
        (native_root / 'gpu_preparation_worker.log').write_text(output or '', encoding='utf-8')
        raise RuntimeError("GPU initializer failed (exit " + str(process.returncode) + "): " +
                           (output or "").strip()[-1600:])
    receipt_path = prepared_root / "preparation_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("GPU initializer did not commit a preparation receipt")
    if progress:
        progress(95.)
    # The same strict reader used by the QGIS viewer checks schema, every
    # declared artifact and the actual initialized array layout.
    if gpu_numerical_variant in (_ON_DEMAND_AMR_VARIANT,
                                 _ON_DEMAND_GENERAL_AMR_VARIANT):
        prepared = inspect_gpu_on_demand_prepared(prepared_root)
        if (prepared.get('variant') != gpu_numerical_variant or
                prepared.get('evolving_banks', False) is not gpu_amr_evolving_banks or
                prepared.get('selector_policy') != gpu_amr_selector_policy):
            raise ValueError('Prepared GPU AMR contract differs from the explicit selection')
    else:
        prepared_data = read_gpu_prepared(prepared_root)
        if gpu_integrator == 'implicit_fwave_midpoint':
            validate_implicit_prepared_scope(prepared_data)
    if QGIS_PRODUCT_KEY in project:
        materialize_gpu_prepared_product(
            native_root, prepared_root, project, worker["scope"])
        requirements=project[QGIS_PRODUCT_KEY].get("worker_requirements",{})
        if requirements.get("required_worker_variant")=="recursive_bank_amr_v118":
            write_recursive_prepared_product(prepared_root,project)
    inspect_gpu_source_worker(worker["root"],
                              expected_manifest_sha256=expected_manifest_sha256)
    if progress:
        progress(100.)
    return native_root, prepared_root
