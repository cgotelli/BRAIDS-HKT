"""Strict new-format river projects and binding to the common numerical owner.

No legacy reader or migration is provided. This format deliberately separates
model selection from physical values and from the internal mesh carrier.
"""
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

from .river_model import ModelError, verify_resolved, SURVEY_BANKS
from .sediment.river_dependent_layers import PARAMETERS as DEPENDENT_LAYER_PARAMETERS, methods_for
from .sediment.river_vr1993 import REFERENCE_PARAMETERS as VAN_RIJN_REFERENCE_PARAMETERS
from .sediment.river_skin_shear import PARAMETERS as SKIN_SHEAR_PARAMETERS, validate_parameters as validate_skin_parameters
from .sediment.river_wet_avalanche import PARAMETERS as WET_AVALANCHE_PARAMETERS, validate_parameters as validate_avalanche_parameters

SCHEMA = "braids.river-project.v41"
SETTING_KEYS = frozenset((
    "domain", "assignments", "grid", "levels", "breaklines", "solids", "controls", "columns", "hydraulics",
    "sediment_boundaries", "bank", "suspension", "adaptation", "adaptive",
    "end_time_s", "dt_max_s", "output_interval_s", "checkpoint_interval_s", "sediment_start_time_s",
))
# Output selection concerns saved result frames only: it never changes the
# numerical state, timestep, checkpoint contents, or turbine performance CSV.
# Older current-schema projects have no choice recorded and retain their full
# historic output inventory when replayed.
OUTPUT_FIELD_IDS = frozenset((
    'initial_bed', 'bedrock', 'bedload', 'suspended', 'advanced_hydraulics',
    'bank_and_collapse', 'boundary_ledgers', 'bedforms', 'bed_material',
))
DEFAULT_OUTPUT_FIELDS = frozenset((
    'initial_bed', 'bedrock', 'bedload', 'suspended', 'advanced_hydraulics',
    'bank_and_collapse', 'bedforms',
))


def output_fields(settings, *, legacy=True):
    """Return the selected frame-field groups without changing physics.

    A missing key is deliberately a legacy full-output replay, not a silent
    rewrite of existing projects.  New desktop documents always save the
    explicit selection.
    """
    raw=settings.get('output_fields')
    if raw is None:
        return frozenset(OUTPUT_FIELD_IDS if legacy else DEFAULT_OUTPUT_FIELDS)
    if (not isinstance(raw, list) or any(type(name) is not str for name in raw)
            or len(set(raw)) != len(raw) or not set(raw) <= OUTPUT_FIELD_IDS):
        fail('settings.output_fields', 'Choose unique known result-field groups')
    return frozenset(raw)


def select_output_fields(fields, settings):
    """Filter optional frame arrays; q/z and output-clock metadata stay saved."""
    selected=output_fields(settings)
    groups={
        'initial_bed': {'initial_bed_m'},
        'bedrock': {'rock'},
        # These are public result fields, not solver diagnostics.  In
        # particular, do not write the mobile-fraction reconstruction,
        # suspension initialisation, or the last sampling interval merely
        # because a bedload/suspended process was active.
        'bedload': {'bedload_vector_m2_s','bedload_capacity_m2_s'},
        'suspended': {'suspended'},
        'advanced_hydraulics': {'W','secondary_fields','secondary_hI_m2_s',
                                'secondary_intensity_m_s','dispersion_flux','dispersion_flux_m2_s'},
        'bank_and_collapse': {'bank','collapse','bank_gross','collapse_gross',
                              'bank_failure_history','bank_failure_count','bank_failure_class_cumulative'},
        'boundary_ledgers': {'water','bed_boundary','sus_boundary'},
        'bedforms': {'bedforms'},
        # Kept for strict validation of older documents, but there is no
        # current public bed-material result in the Desktop output inventory.
        'bed_material': set(),
    }
    # ``state`` is a raw Jiang--Cao solver snapshot.  It is never a public
    # result field and must not be written merely because it is available.
    # ``legacy_internal`` names fields produced by existing workers but not
    # exposed by the current Desktop report contract.  They must be removed
    # explicitly rather than falling through as unknown fields.
    legacy_internal={'state','mobile_bedload_concentration','bedload_sample_dt_s',
                     'suspended_initial','acceleration','mass','sv','basal_ledger','area'}
    optional=set().union(*groups.values())|legacy_internal
    retained={'q','z','time','steps','water_vector_m2_s','bedload_flux_sampling'}
    for name in selected:retained.update(groups[name])
    return {name:value for name,value in fields.items() if name not in optional or name in retained}
SLOPE_PARAMETERS = dict(alfabs=0., alfabn=0., ashld=.85, bshld=.5,
                        cshld=0., dshld=0., alfpa=1., thcrpa=.05)
HIDING_PARAMETERS = dict(capacity=dict(parker_alpha=.5, wu_exponent=-.6),
                         slope=dict(parker_alpha=.5, wu_exponent=-.6))
ROUGHNESS_PARAMETERS = dict(chezy_m_half_s=45., manning_s_m_third=.03,
                            nikuradse_m=.003, white_colebrook_relation="flow_log_layer")
BEDFORM_METHODS = dict(height="van_rijn_1984", length="van_rijn_1984",
                      roughness="van_rijn_1984", evolution="equilibrium")
GENERAL_PARAMETERS = dict(ACal=8., PowerB=0., PowerC=1.5, RipFac=1., ThetaC=.047,
                          SusACal=2.5, SusPowerB=2., SusPowerC=1., SusRipFac=.9,
                          SusThetaC=.035)


def fail(path, message, category="invalid_input"):
    raise ModelError(category, path, message)


def exact(value, keys, path):
    if not isinstance(value, dict) or set(value) != set(keys):
        fail(path, "Required fields: " + ", ".join(sorted(keys)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(key, "Duplicate JSON field")
        result[key] = value
    return result


def read_json(text):
    try:
        return json.loads(text, object_pairs_hook=_pairs,
                          parse_constant=lambda s: fail("project", f"Nonfinite value: {s}"))
    except json.JSONDecodeError as exc:
        raise ModelError("invalid_input", "project", "Invalid JSON") from exc


def _number(value, path, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        fail(path, "A finite number is required")
    if positive and value <= 0:
        fail(path, "A positive value is required")


def checkpoint_editor_state(project):
    """Resolve only the saved editor choice, never infer or change execution.

    An existing current-format document without the optional annotation keeps
    its explicit interval as a custom value, even if it equals the end time.
    """
    settings=project['settings']
    editor=project.get('editor_state',{})
    if 'checkpoint_clock' not in editor:
        return dict(mode='custom',custom_interval_s=settings['checkpoint_interval_s'])
    value=editor['checkpoint_clock'];path='editor_state.checkpoint_clock'
    exact(value,('mode','custom_interval_s'),path)
    if value['mode'] not in ('end_time','custom'):
        fail(path+'.mode','Choose end_time or custom')
    _number(value['custom_interval_s'],path+'.custom_interval_s',positive=True)
    resolved=settings['end_time_s'] if value['mode']=='end_time' else value['custom_interval_s']
    if settings['checkpoint_interval_s']!=resolved:
        fail(path,'Checkpoint editor choice differs from the explicit saved interval')
    return dict(value)


def worker_profile(project):
    """Private mesh-carrier representation; never an import/migration reader."""
    from .sediment.adaptive_river_worker import SCHEMA as CARRIER_SCHEMA
    from .river_boundary import carrier_boundaries
    from .river_open_boundary import carrier
    settings=project['settings']
    c=settings['controls']
    simple_hydro=(project['model']['physics']['sediment'] is None and
        not (c.get('betac') or c.get('diffusivity') or c.get('viscosity')) and
        len(settings['columns']['fraction_ids'])==1)
    # Restart publication is an administrative clock, not a native mesh or
    # numerical-step input. The public worker passes it explicitly at runtime.
    carrier_settings={key:value for key,value in settings.items()
                      if key not in ('checkpoint_interval_s','output_fields')}
    from .river_turbines import constraints,mesh_regions
    layout=project.get('power_plan',{}).get('layout',project['turbines'])
    carrier_settings['breaklines']=settings['breaklines']+constraints(layout)
    carrier_settings['local_refinement']=mesh_regions(layout)
    # An absent optional switch in the current project format retains the
    # existing finest-boundary policy. The emitted worker profile is explicit.
    adaptation={'open_boundary_enabled':True,**settings['adaptation']}
    return dict(carrier_settings, adaptation=adaptation,hydraulics=carrier(settings['hydraulics']),
        sediment_boundaries=carrier_boundaries(settings['sediment_boundaries'],settings['columns']),
        schema=CARRIER_SCHEMA, physics_mode="hydrodynamics" if simple_hydro else "combined")


def validate_binding(profile, resolved):
    """Admit only actual operators on this owner, equally on CPU and GPU.

    Missing integrations are reported as such, not as physical incompatibility.
    This ledger is extended as the other existing component owners are composed.
    """
    doc = verify_resolved(resolved).document()
    from .river_jiang_cao import active as jc_active, validate_binding as jc_binding
    if jc_active(doc):return jc_binding(profile, doc)
    h = doc["physics"]["hydrodynamics"]
    s = doc["physics"]["sediment"]
    c, columns = profile["controls"], profile["columns"]
    if s is not None and s['capacity']['method']=='capacity.mpm_braids_5_055':
        from .hydraulics.bedload import validate_execution
        validate_execution(profile)
    simple_hydro = (s is None and not (c['betac'] or c['diffusivity'] or c['viscosity'])
                    and len(columns['fraction_ids']) == 1)

    def pending(path, message):
        fail(path, message, "integration_pending")

    if h["secondary_flow"] == "secondary.off" and (c["betac"] or c["diffusivity"]):
        fail("controls", "Secondary flow is off but its momentum/diffusion coefficients are nonzero")
    if (s is not None and h['secondary_flow'] != 'secondary.off' and
            s['secondary_direction'] == 'capacity_owned' and c['espir'] != 1.):
        fail('controls.espir', 'This transport formula owns its sediment direction and does not use FLOW Espir. Set Espir to 1, or select a compatible scalar-direction formula.', 'integration_pending')
    if h["turbulence"] == "off" and c["viscosity"]:
        fail("controls.viscosity", "Turbulence is off but eddy viscosity is nonzero")
    if c["diameter"] != columns["diameter_m"][0]:
        fail("controls.diameter", "The hydraulic carrier diameter must identify the first material class")
    bank, suspension = profile["bank"], profile["suspension"]
    if s is None:
        if bank["bank_theta"] or bank["collapse"] or suspension["sus_factor"] or suspension["diffusivity"]:
            fail("settings", "Sediment is off but a sediment process has active coefficients")
        for row in profile["sediment_boundaries"].values():
            if any(row["concentration_kg_m3"]) or any(row.get("solid_flux_m2_s", [])):
                fail("sediment_boundaries", "Sediment is off but boundary sediment is nonzero")
        return doc
    if s["capacity"]["method"] not in ("capacity.mpm_braids_5_055", "capacity.general", "capacity.mpm", "capacity.engelund_hansen",
                                     "capacity.van_rijn_1984_modified", "capacity.wang_fredsoe",
                                     "capacity.ashida_michiue_generalized", "capacity.wilcock_crowe",
                                     "capacity.wilcock_crowe_modified", "capacity.gaeuman",
                                     "capacity.soulsby_van_rijn", "capacity.soulsby", "capacity.van_rijn_1993",
                                     "capacity.smart_jaggi", "capacity.ancey_recking_2023", "capacity.recking_2010",
                                     "capacity.camenen_larson_2005", "capacity.wong_parker_2006"):
        pending("sediment.capacity", "Connect the selected capacity, wave derivative and exchange "
                "operators together before running this formula")
    if s["capacity"]["method"] == "capacity.general" and any(
        s["capacity"]["parameters"][key] < 0 for key in ("PowerB", "PowerC", "SusPowerB", "SusPowerC")):
        pending("sediment.capacity", "Negative power-law exponents require a separately qualified wave treatment")
    if s["capacity"]["method"] == "capacity.ashida_michiue_generalized" and any(
        s["capacity"]["parameters"][key] < 0 for key in ("PowerM", "PowerP", "PowerQ")):
        pending("sediment.capacity", "Negative Ashida-Michiue powers require a qualified threshold treatment")
    bed = s["bed"]
    mixed = bed["composition"] == "bed.fully_mixed"
    if columns["bed"] != ("mixed" if mixed else "layered"):
        fail("columns.bed", "Column representation differs from the explicitly selected bed")
    if columns["transport_layer"] != ("depth" if bed["transport_layer"] == "transport_layer.water_depth" else "constant"):
        fail("columns.transport_layer", "Column target differs from the selected thickness method")
    if mixed and (columns["exchange_thickness_m"] or columns["lagrangian_underlayers"]):
        fail("columns", "A fully mixed bed cannot contain exchange or substrate layers")
    grains = s["grains"]
    for field, key in (("id", "fraction_ids"), ("diameter_m", "diameter_m"),
                       ("solid_density_kg_m3", "solid_density_kg_m3")):
        if [g[field] for g in grains] != list(columns[key]):
            fail("columns." + key, "Column material differs from the resolved grain definition")
    if not s["banks"] == "redistribution.dry_bank" and bank["bank_theta"]:
        fail("bank.bank_theta", "FLOW dry-bank sharing is not selected but its coefficient is nonzero")
    if s['banks'] in SURVEY_BANKS:
        if s['transport']=='total_load' or columns['base_update']!=1:
            pending('sediment.banks','Surveyed banks require finite stock and separate bedload/suspended exchange; '
                'choose the finite independent basal rule and disable algebraic total load')
        if s['banks']==SURVEY_BANKS[1] and s['avalanche']=='redistribution.avalanche':
            pending('sediment.avalanche','Retained block release supports parent-owned conservative '
                'wet/dry collapse, not a second FLOW wet-slope flux')
    if bank["collapse"] != (s["avalanche"] in ("redistribution.braids_conservative_wet_dry_avalanche", "redistribution.jiang_cao_repose_exner223_v1")):
        fail("bank.collapse", "Collapse switch differs from the resolved redistribution method")
    if s["suspension"] == "off" and (suspension["sus_factor"] or suspension["diffusivity"]):
        fail("suspension", "Disabled suspension requires zero entrainment and diffusivity")
    if s["suspension"] == "suspension.settling_only" and suspension["sus_factor"]:
        fail("suspension.sus_factor", "Settling-only cannot entrain sediment; set its entrainment factor to zero")
    return doc


def validate_project(value):
    """Validate without constructing a mesh or numerical owner."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        fail("schema", f"Only {SCHEMA} can run. Old projects are not supported or migrated; "
             "create a new project.", "unsupported_format")
    keys = {"schema", "name", "model", "settings", "terrain", "initial", "material_zones",
            "slope_parameters", "hiding_parameters", "roughness_parameters", "bedform_methods",
            "dependent_layer_parameters", "column_methods", "van_rijn_reference_parameters", "skin_shear_parameters",
            "wet_avalanche_parameters", "sediment_schedules", "boundary_states", "boundary_resolution", "bank_survey", "turbines"}
    if 'power_plan' in value:fail('power_plan','Estimates-only Gaussian planning is retired. Use the coupled Thetis turbine extension and configure its speed / Ct / Cp data.')
    selected=value.get('model',{}).get('physics',{}).get('sediment')
    if selected and selected.get('suspension')=='suspension.garcia_parker_1991':
        keys.add('garcia_parker_parameters')
        from .sediment.transport_setup import validate_gp_execution
        validate_gp_execution(value)
    if selected and selected.get('suspension')=='suspension.settling_only':
        keys.add('supplied_suspension_parameters')
        from .sediment.transport_setup import validate_supplied_execution
        validate_supplied_execution(value)
    from .sediment.river_classic_suspension import METHODS as CLASSIC_SUSPENSION
    if selected and selected.get('suspension') in CLASSIC_SUSPENSION:
        keys.add('classic_suspension_parameters')
        from .sediment.transport_setup import validate_classic_execution
        validate_classic_execution(value)
    if selected and selected.get('suspension')=='suspension.de_leeuw_2020':
        keys.add('de_leeuw_parameters')
        from .sediment.transport_setup import validate_dl_execution
        validate_dl_execution(value)
    if "editor_state" in value:
        keys.add("editor_state")
        editor = value["editor_state"]
        editor_keys={"schema", "sediment", "forcing", "schedules", "boundaries", "bank_survey"}
        if isinstance(editor,dict) and 'turbines' in editor:
            editor_keys.add('turbines')
            draft=editor['turbines']
            # An empty inactive UI draft carries no physics; keep ordinary
            # hydraulic projects usable while the editor rewrites that draft.
            inactive_empty=isinstance(draft,dict) and draft.get('schema')==4 and draft.get('rows')==[] and value.get('turbines') is None
            independent=isinstance(draft,dict) and draft.get('schema') in (6,7)
            draft_keys=('schema','rows','parameters','table','surface') if inactive_empty else ('schema','rows','parameters','table')
            exact(draft,(*draft_keys,'curves') if independent else draft_keys,'editor_state.turbines')
            if (draft['schema'] not in ((4,) if inactive_empty else (5,6,7)) or not isinstance(draft['rows'],list) or not isinstance(draft['parameters'],dict) or not isinstance(draft['table'],list)):
                fail('editor_state.turbines','Current Thetis turbine editor state required; reconfigure old active turbine controls.')
            if independent:
                exact(draft['curves'],('thrust_coefficient','power_coefficient'),'editor_state.turbines.curves')
                if any(not isinstance(rows,list) or any(not isinstance(row,list) or len(row)!=2 or any(not isinstance(v,str) for v in row) for row in rows) for rows in draft['curves'].values()):
                    fail('editor_state.turbines.curves','Each curve draft must contain two text cells per row.')
        if isinstance(editor,dict) and 'checkpoint_clock' in editor:editor_keys.add('checkpoint_clock')
        exact(editor, editor_keys, "editor_state")
        if type(editor["schema"]) is not int or editor["schema"] != 6 or not isinstance(editor["sediment"], dict):
            fail("editor_state", "Unknown editor state")
        from .river_forcing import validate_draft
        try:
            validate_draft(editor["forcing"])
            from .river_schedule import ScheduleDraft
            ScheduleDraft(editor["schedules"])
            from .river_boundary_draft import BoundaryDraft
            BoundaryDraft(editor['boundaries'])
        except ValueError as exc:
            fail("editor_state.forcing",str(exc))
        exact(editor["sediment"], ("schema", "widgets", "parameters", "grains", "layers", "layer_drafts", "grain_sizes", "transport_setup"),
              "editor_state.sediment")
        if type(editor["sediment"]["schema"]) is not int or editor["sediment"]["schema"] != 13:
            fail("editor_state.sediment", "Current editor format required; old drafts are not migrated",
                 "unsupported_format")
        widgets = editor['sediment']['widgets']
        if widgets.get('enabled') and widgets.get('suspension_on'):
            from .sediment.transport_setup import validate_execution
            validate_execution(value)
        # Draft text is preserved for the UI only. It never feeds a numerical
        # owner: active physical values must be present in the validated model.
    exact(value, keys, "project")
    from .river_turbines import validate as validate_turbines
    try:validate_turbines(value['turbines'],value)
    except (ValueError,TypeError,KeyError) as exc:fail('turbines',str(exc))
    from .river_material_zones import validate as validate_zones
    try:validate_zones(value['material_zones'])
    except ValueError as exc:fail('material_zones',str(exc))
    try:
        validate_avalanche_parameters(value["wet_avalanche_parameters"])
    except ValueError as exc:
        fail("wet_avalanche_parameters", str(exc))
    try:
        validate_skin_parameters(value["skin_shear_parameters"])
    except ValueError as exc:
        fail("skin_shear_parameters", str(exc))
    exact(value["van_rijn_reference_parameters"], VAN_RIJN_REFERENCE_PARAMETERS, "van_rijn_reference_parameters")
    for name, number in value["van_rijn_reference_parameters"].items():
        _number(number, "van_rijn_reference_parameters."+name, positive=True)
    if value["van_rijn_reference_parameters"]["maximum_volume_concentration"]>1.:
        fail("van_rijn_reference_parameters.maximum_volume_concentration","Volume concentration cannot exceed one")
    if type(value["name"]) is not str or not value["name"].strip():
        fail("name", "A nonempty project name is required")
    verify_resolved(value["model"])
    from .river_jiang_cao import active as jc_active, validate_project as jc_validate
    if jc_active(value['model']):jc_validate(value)
    from .sediment.bank_setup import validate as validate_bank
    selected_bank=value['model']['physics']['sediment']
    if selected_bank is not None:
        validate_bank(selected_bank['banks'], selected_bank['capacity']['method'], selected_bank['suspension'],
                      bedload=bool(value['settings']['suspension']['bed_factor']))
    from .sediment.collapse_setup import validate as validate_collapse
    collapse_selection=value['model']['physics']['sediment']
    if collapse_selection is not None:
        validate_collapse(collapse_selection['avalanche'],collapse_selection['capacity']['method'],
            collapse_selection['suspension'],mf=value['settings']['controls']['morfac'],
            banks=collapse_selection['banks'])
    setting_keys=set(SETTING_KEYS)
    # Existing current-schema projects predate selectable frame outputs; keep
    # their full-output replay valid until the editor saves an explicit choice.
    if 'output_fields' in value['settings']:setting_keys.add('output_fields')
    exact(value["settings"], setting_keys, "settings")
    output_fields(value['settings'])
    from .river_open_boundary import resolve as resolve_boundaries
    try:
        kinds={a['bc_id']:a['kind'] for a in value['settings']['assignments'] if a['kind']!='wall'}
        resolved_boundaries=resolve_boundaries(value['settings']['hydraulics'],value['boundary_states'],kinds,
            sediment=value['model']['physics']['sediment'],ng=len(value['settings']['columns']['fraction_ids']),
            end_time=value['settings']['end_time_s'])
    except (ValueError,KeyError,TypeError) as exc:fail('boundaries',str(exc))
    if value['boundary_resolution']!=resolved_boundaries:fail('boundary_resolution','Resolved boundary identities differ from the active inputs')
    exact(value["terrain"], ("dem_path",), "terrain")
    if type(value["terrain"]["dem_path"]) is not str or not value["terrain"]["dem_path"]:
        fail("terrain.dem_path", "An explicit DEM path is required")
    exact(value["slope_parameters"], SLOPE_PARAMETERS, "slope_parameters")
    exact(value["bedform_methods"], BEDFORM_METHODS, "bedform_methods")
    exact(value["dependent_layer_parameters"], DEPENDENT_LAYER_PARAMETERS, "dependent_layer_parameters")
    for name, number in value["dependent_layer_parameters"].items():
        if name != "hiding_scope":
            _number(number, "dependent_layer_parameters."+name)
    if value["dependent_layer_parameters"]["hiding_scope"] not in ("active", "active_coarse"):
        fail("dependent_layer_parameters.hiding_scope", "Choose active or active_coarse")
    for name, number in value["slope_parameters"].items():
        _number(number, "slope_parameters." + name)
    if value["slope_parameters"]["thcrpa"] < 0:
        fail("slope_parameters.thcrpa", "Critical Shields cannot be negative")
    exact(value["hiding_parameters"], HIDING_PARAMETERS, "hiding_parameters")
    for channel in HIDING_PARAMETERS:
        exact(value["hiding_parameters"][channel], HIDING_PARAMETERS[channel], "hiding_parameters."+channel)
        for name, number in value["hiding_parameters"][channel].items():
            _number(number, "hiding_parameters."+channel+"."+name)
    exact(value["roughness_parameters"], ROUGHNESS_PARAMETERS, "roughness_parameters")
    roughness = value["roughness_parameters"]
    for key in ("chezy_m_half_s", "manning_s_m_third", "nikuradse_m"):
        _number(roughness[key], "roughness_parameters."+key, positive=True)
    if roughness["white_colebrook_relation"] not in ("flow_log_layer", "classical_2d"):
        fail("roughness_parameters.white_colebrook_relation", "Choose flow_log_layer or classical_2d")
    # These are implementation restrictions, not universal compatibility rules.
    if value["bedform_methods"] != BEDFORM_METHODS:
        fail("process_parameters", "The common owner does not yet bind different bedform coefficients",
             "integration_pending")
    from .sediment.current_river_solver import CurrentRiverControls
    from .sediment.bed_columns_v101 import ColumnMethodsV101
    from .sediment.river_field_policy import FieldSettings
    from .sediment.adaptive_river_worker import validate_profile
    p = worker_profile(value)
    if collapse_selection is not None and collapse_selection['banks']!='off':
        from .sediment.bank_setup import validate_inventory
        validate_inventory(p['columns'])
    if collapse_selection is not None and collapse_selection['avalanche']=='redistribution.jiang_cao_repose_exner223_v1':
        col=p['columns']
        if len(col['fraction_ids'])!=1 or col['bed']!='mixed' or col['porosity']!=0 or col['base_update']!=1:
            fail('settings.columns','Repose projection currently requires one bulk material, a fully mixed finite bed and dry-bulk-density packing. Layered/mixture coupling is not yet qualified.','integration_pending')
    for name, cls in (("controls", CurrentRiverControls), ("columns", ColumnMethodsV101),
                      ("adaptation", FieldSettings)):
        exact(p[name], (f.name for f in fields(cls)), "settings." + name)
    if type(p['adaptive']) is not bool or type(p['adaptation']['terrain_enabled']) is not bool:
        fail('settings.adaptive', 'Explicit static-manual, static-terrain or dynamic-depth mesh mode required')
    if p['adaptive'] and p['adaptation']['terrain_enabled']:
        fail('settings.adaptation', 'Choose dynamic depth-only AMR (terrain off), static manual refinement (both off), or static terrain refinement (AMR off). Dynamic AMR and terrain refinement cannot be combined.')
    for name, number in p["controls"].items():
        _number(number, "settings.controls." + name)
    _number(p['controls']['morfac'], 'settings.controls.morfac', positive=True)
    if p["controls"]["chezy"] != roughness["chezy_m_half_s"]:
        fail("controls.chezy", "The inert carrier Chezy must match the explicitly saved Chezy value")
    if type(p["controls"]["threads"]) is not int:
        fail("settings.controls.threads", "An explicit integer thread count is required")
    for name, number in p["suspension"].items():
        _number(number, "settings.suspension." + name)
    for key in ("end_time_s", "dt_max_s", "output_interval_s"):
        _number(p[key], "settings." + key, positive=True)
    _number(value['settings']['checkpoint_interval_s'], 'settings.checkpoint_interval_s', positive=True)
    start=value['settings']['sediment_start_time_s']
    _number(start,'settings.sediment_start_time_s')
    if not 0<=start<=p['end_time_s']:
        fail('settings.sediment_start_time_s','Use a time from zero to the simulation end time')
    sediment=value['model']['physics']['sediment']
    if sediment is not None:
        for name,row in value['settings']['sediment_boundaries'].items():
            if kinds[name]=='outlet' and row['kind']!='outflow':fail('sediment_boundaries.'+name,'Outlets export sediment only; no incoming sediment condition is allowed')
    if start and sediment is not None and (sediment['suspension']!='off' or sediment['banks']!='off' or sediment['avalanche']!='off' or sediment['bedforms']!='off'):
        fail('settings.sediment_start_time_s','Hydraulic warm-up is qualified for bedload only; other sediment processes must be off')
    checkpoint_editor_state(value)
    from .sediment.river_column_methods import validate_identity, topology
    for key in ("porosity", "top_update", "base_update", "lagrangian_underlayers"):
        if type(p["columns"][key]) is not int:
            fail("settings.columns."+key, "An explicit integer wire is required")
    columns = ColumnMethodsV101(**p["columns"])
    validate_identity(value["column_methods"], columns)
    c, columns = validate_profile(p)
    # Complete early forcing checks, before native construction.
    ng = len(columns.fraction_ids)
    for name, row in p["hydraulics"].items():
        key = "stage_m" if row["mode"] == "stage" else "discharge_m3_s"
        _number(row[key], "hydraulics." + name + "." + key)
    for name, row in p["sediment_boundaries"].items():
        if not isinstance(row, dict) or row.get("kind") not in ("feed", "outflow"):
            fail("sediment_boundaries." + name, "Explicit feed/outflow required")
        keys = {"kind", "concentration_kg_m3"} | ({"solid_flux_m2_s"} if row["kind"] == "feed" else set())
        exact(row, keys, "sediment_boundaries." + name)
        for key in keys - {"kind"}:
            if not isinstance(row[key], list) or len(row[key]) != ng:
                fail("sediment_boundaries." + name, "Ordered per-fraction forcing required")
            for v in row[key]:
                _number(v, "sediment_boundaries." + name)
                if v < 0:
                    fail("sediment_boundaries." + name, "Negative sediment forcing is not supported")
    initial = value["initial"]
    initial_keys = ("stage_m", "velocity_m_s", "layer_thickness_m", "layer_fractions", "concentration_kg_m3")
    exact(initial, initial_keys + (("depth_m",) if isinstance(initial,dict) and 'depth_m' in initial else ()), "initial")
    _number(initial["stage_m"], "initial.stage_m")
    if 'depth_m' in initial:
        _number(initial['depth_m'], 'initial.depth_m')
        if initial['depth_m'] < 0:fail('initial.depth_m','Uniform depth cannot be negative')
    for key, count in (("velocity_m_s", 2), ("concentration_kg_m3", ng)):
        if not isinstance(initial[key], list) or len(initial[key]) != count:
            fail("initial." + key, "Wrong number of values")
        for v in initial[key]:
            _number(v, "initial." + key)
    if any(v < 0 for v in initial["concentration_kg_m3"]):
        fail("initial.concentration_kg_m3", "Concentration cannot be negative")
    th, fr = initial["layer_thickness_m"], initial["layer_fractions"]
    if not isinstance(th, list) or not isinstance(fr, list) or len(fr) != len(th):
        fail("initial", "Explicit bed thicknesses and compositions required")
    if (columns.bed == "mixed" and len(th) != 1) or (columns.bed == "layered" and len(th) < 2):
        fail("initial", "A fully mixed bed requires one layer; a layered bed requires active and substrate layers")
    columns.targets([0.], len(th))
    for thickness, row in zip(th, fr):
        _number(thickness, "initial.layer_thickness_m")
        if thickness<0:fail('initial.layer_thickness_m','Initial erodible depth cannot be negative; zero is a bare non-erodible base')
        if not isinstance(row, list) or len(row) != ng:
            fail("initial.layer_fractions", "Ordered per-class composition required in every layer")
        for fraction in row:
            _number(fraction, "initial.layer_fractions")
            if not 0 <= fraction <= 1:
                fail("initial.layer_fractions", "Fractions must be between zero and one")
        if abs(math.fsum(row)-1) > 1e-10:
            fail("initial.layer_fractions", "Layer fractions must sum to one")
    doc = validate_binding(p, value["model"])
    selected = doc["physics"]["sediment"]
    from .river_bank_survey import validate as validate_survey
    survey=value['bank_survey'];survey_on=selected is not None and selected['banks'] in SURVEY_BANKS
    if survey_on:
        try:validate_survey(survey,p['domain']['spatial_reference'])
        except ValueError as exc:fail('bank_survey',str(exc))
        if (survey['critical_undercut_m'] is not None)!=(selected['banks']==SURVEY_BANKS[1]):
            fail('bank_survey.critical_undercut_m','A positive block threshold is required only in retained-collapse mode; toe-only requires null')
    elif survey is not None:
        fail('bank_survey','Inactive bank survey must be null; retain geometry in editor state only')
    if 'editor_state' in value and value['editor_state']['bank_survey'] is not None:
        draft=value['editor_state']['bank_survey']
        try:validate_survey(draft,draft['spatial_reference'])
        except (ValueError,KeyError,TypeError) as exc:fail('editor_state.bank_survey',str(exc))
    from .river_schedule import validate as validate_schedules
    try:
        validate_schedules(value["sediment_schedules"],value['settings']["sediment_boundaries"],ng,p["end_time_s"],
            sediment=selected is not None,suspended=selected is not None and selected["suspension"]!="off")
    except ValueError as exc:
        fail("sediment_schedules",str(exc))
    if selected is None and any(row['kind'] not in ('feed','outflow') for row in value['settings']['sediment_boundaries'].values()):
        fail('sediment_boundaries','Native transport policy cannot be active when sediment is off')
    topology(columns, len(th), coarse=bool(selected and selected["bed"]["coarse_layer"]))
    if selected is not None and (selected["bed"]["coarse_layer"]
                                 or selected["bed"]["transport_layer"] == "transport_layer.dune_height"):
        dependent = methods_for(columns, selected["bed"], value["dependent_layer_parameters"])
        if dependent.hiding_scope == "active_coarse" and not dependent.coarse_layer:
            fail("dependent_layer_parameters.hiding_scope", "Active-plus-coarse hiding needs the coarse layer")
        dependent.targets([0.], len(th), dune_height=[0.])
    if selected is not None and fr[0] != [g["fraction"] for g in selected["grains"]]:
        fail("initial.layer_fractions", "Initial active composition differs from the material table")
    if selected is None or selected["suspension"] == "off":
        if any(initial["concentration_kg_m3"]) or any(
            any(row["concentration_kg_m3"]) for row in p["sediment_boundaries"].values()
        ):
            fail("initial.concentration_kg_m3", "Disabled suspension cannot carry concentration")
    # Validate geometry and all boundary assignments without building a hierarchy.
    from .boundaries.domain import PolygonDomain, BoundaryAssignment, boundary_polyline
    from .boundaries.inlet_io import NativeGrid
    domain = PolygonDomain(**p["domain"])
    from .boundaries.breaklines import validate_breaklines
    try:validate_breaklines(p['breaklines'],domain)
    except (ValueError,TypeError) as exc:fail('settings.breaklines',str(exc))
    from .boundaries.solids import validate_solids
    try:validate_solids(p['solids'],domain)
    except (ValueError,TypeError) as exc:fail('settings.solids',str(exc))
    grid = NativeGrid(**p["grid"])
    if grid.dx_m != grid.dy_m:
        fail("grid", "The common river route requires square Cartesian cells")
    assignments = [BoundaryAssignment(**a) for a in p["assignments"]]
    segments = domain.boundary_segments(assignments)
    ids = {s.bc_id for s in segments if s.kind != "wall"}
    if ids != set(p["hydraulics"]):
        fail("hydraulics", "Assigned open-boundary IDs differ from the defined hydraulic boundaries")
    for bc_id in sorted(ids):
        boundary_polyline(domain, assignments, bc_id)
    # Each named section remains contiguous and straight (checked above), but
    # need not lie on a Cartesian grid line. Preparation retains the true
    # normal/length of its cut faces and merges fragments to the local area
    # floor. Hydraulic traces and accepted flux ledgers use those same faces.
    return read_json(canonical(value))


def load_project(path):
    from .windows_paths import filesystem_path
    return validate_project(read_json(filesystem_path(path).read_text(encoding="utf-8")))
